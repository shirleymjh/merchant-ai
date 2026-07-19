from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage

from merchant_ai.services.artifacts import WorkspaceArtifactStore


RECOVERY_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _model_payload(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(by_alias=True, mode="json")
    if isinstance(value, dict):
        return dict(value)
    return value


def _message_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return _canonical_json(content)


def _tool_payload(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return dict(tool)
    schema: Any = {}
    args_schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            schema = model_json_schema()
        except Exception:
            schema = {}
    return {
        "name": str(getattr(tool, "name", "") or ""),
        "description": str(getattr(tool, "description", "") or ""),
        "schema": schema,
    }


def _serialized_request(
    messages: list[Any],
    system_message: Any,
    tools: list[Any],
) -> str:
    return _canonical_json(
        {
            "system": _message_text(system_message),
            "messages": [
                {
                    "type": str(getattr(message, "type", "") or ""),
                    "content": _message_text(message),
                    "toolCalls": list(
                        getattr(message, "tool_calls", None) or []
                    ),
                }
                for message in messages
            ],
            "tools": [_tool_payload(tool) for tool in tools],
        }
    )


def _conservative_tokens(text: str, structural_items: int = 0) -> int:
    """Return an explicitly labelled conservative fallback estimate.

    UTF-8 bytes are used instead of character count so CJK input is not
    substantially under-counted.  This is never reported as a provider token
    count.
    """

    byte_count = len(str(text or "").encode("utf-8"))
    return max(1, int(math.ceil(byte_count / 3.0)) + max(0, structural_items) * 4 + 8)


@dataclass(frozen=True)
class GroundedTokenCount:
    tokens: int
    source: str
    authority: str
    fallback_used: bool

    def report(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "source": self.source,
            "authority": self.authority,
            "fallbackUsed": self.fallback_used,
        }


class ProviderAwareContextTokenCounter:
    """Count the actual model request with the current model tokenizer first."""

    def __init__(
        self,
        model: Any = None,
        *,
        provider_counter: Optional[
            Callable[[list[Any], Any, list[Any]], int]
        ] = None,
    ) -> None:
        self.model = model
        self.provider_counter = provider_counter

    def count(
        self,
        messages: list[Any],
        system_message: Any,
        tools: list[Any],
    ) -> GroundedTokenCount:
        if self.provider_counter is not None:
            try:
                tokens = int(
                    self.provider_counter(messages, system_message, tools)
                )
                if tokens >= 0:
                    return GroundedTokenCount(
                        tokens=tokens,
                        source="injected_provider_counter",
                        authority="PROVIDER_MODEL",
                        fallback_used=False,
                    )
            except Exception:
                pass

        model_messages = list(messages)
        if system_message is not None:
            model_messages.insert(0, system_message)
        message_counter = getattr(
            self.model,
            "get_num_tokens_from_messages",
            None,
        )
        text_counter = getattr(self.model, "get_num_tokens", None)
        model_name = type(self.model).__name__ if self.model is not None else "unavailable"
        if callable(message_counter):
            if tools:
                try:
                    tokens = int(message_counter(model_messages, tools=tools))
                    if tokens >= 0:
                        return GroundedTokenCount(
                            tokens=tokens,
                            source=(
                                "model.%s.get_num_tokens_from_messages_with_tools"
                                % model_name
                            ),
                            authority="PROVIDER_MODEL",
                            fallback_used=False,
                        )
                except Exception:
                    pass
            try:
                message_tokens = int(message_counter(model_messages))
                if message_tokens >= 0:
                    if not tools:
                        return GroundedTokenCount(
                            tokens=message_tokens,
                            source=(
                                "model.%s.get_num_tokens_from_messages"
                                % model_name
                            ),
                            authority="PROVIDER_MODEL",
                            fallback_used=False,
                        )
                    serialized_tools = _canonical_json(
                        [_tool_payload(tool) for tool in tools]
                    )
                    if callable(text_counter):
                        tool_tokens = int(text_counter(serialized_tools))
                        if tool_tokens >= 0:
                            return GroundedTokenCount(
                                tokens=message_tokens + tool_tokens,
                                source=(
                                    "model.%s.message_and_text_tokenizers"
                                    % model_name
                                ),
                                authority="PROVIDER_MODEL",
                                fallback_used=False,
                            )
                    tool_tokens = _conservative_tokens(
                        serialized_tools,
                        structural_items=len(tools),
                    )
                    return GroundedTokenCount(
                        tokens=message_tokens + tool_tokens,
                        source=(
                            "model.%s.message_tokenizer+"
                            "conservative_utf8_tool_schema_estimate"
                            % model_name
                        ),
                        authority="MIXED",
                        fallback_used=True,
                    )
            except Exception:
                pass

        serialized = _serialized_request(messages, system_message, tools)
        if callable(text_counter):
            try:
                tokens = int(text_counter(serialized))
                if tokens >= 0:
                    return GroundedTokenCount(
                        tokens=tokens,
                        source="model.%s.get_num_tokens" % model_name,
                        authority="PROVIDER_MODEL",
                        fallback_used=False,
                    )
            except Exception:
                pass
        return GroundedTokenCount(
            tokens=_conservative_tokens(
                serialized,
                structural_items=len(messages) + len(tools) + 1,
            ),
            source="conservative_utf8_bytes_estimate",
            authority="CONSERVATIVE_FALLBACK",
            fallback_used=True,
        )


def _semantic_receipts(session: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in list(getattr(session, "core_semantic_evidence", None) or []):
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("refId") or "").strip()
        if not ref_id:
            continue
        receipts.append(
            {
                "refId": ref_id,
                "path": str(item.get("path") or ""),
                "kind": str(item.get("kind") or ""),
                "topic": str(item.get("topic") or ""),
                "table": str(item.get("table") or ""),
                "contentHash": str(item.get("contentHash") or ""),
                "contentComplete": bool(item.get("contentComplete")),
            }
        )
    return sorted(
        receipts,
        key=lambda item: (item["refId"], item["path"], item["contentHash"]),
    )


def _result_artifact_receipts(run_result: Any) -> list[dict[str, Any]]:
    if run_result is None:
        return []
    merged = getattr(run_result, "merged_query_bundle", None)
    bundles = [*list(getattr(run_result, "query_bundles", None) or [])]
    if merged is not None:
        bundles.append(merged)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle in bundles:
        for event in list(getattr(bundle, "runtime_events", None) or []):
            if not isinstance(event, dict):
                continue
            receipt = event.get("resultArtifact")
            if not isinstance(receipt, dict):
                continue
            fingerprint = str(receipt.get("artifactFingerprint") or "")
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            result.append(dict(receipt))
    return sorted(
        result,
        key=lambda item: str(item.get("artifactFingerprint") or ""),
    )


def _query_artifact_receipts(session: Any) -> list[dict[str, Any]]:
    runtime = getattr(session, "runtime", None)
    goal_ids_by_artifact = dict(
        getattr(session, "artifact_goal_ids", None) or {}
    )
    receipts: list[dict[str, Any]] = []
    for artifact in list(
        getattr(runtime, "verified_query_ledger", None) or []
    ):
        run_result = getattr(artifact, "run_result", None)
        bundle = getattr(run_result, "merged_query_bundle", None)
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        receipts.append(
            {
                "queryArtifactId": artifact_id,
                "generation": int(getattr(artifact, "generation", 0) or 0),
                "attemptId": str(getattr(artifact, "attempt_id", "") or ""),
                "contractFingerprint": str(
                    getattr(artifact, "contract_fingerprint", "") or ""
                ),
                "sqlFingerprint": str(
                    getattr(artifact, "sql_fingerprint", "") or ""
                ),
                "executionMode": str(
                    getattr(artifact, "execution_mode", "") or ""
                ),
                "goalIds": sorted(
                    {
                        str(item)
                        for item in goal_ids_by_artifact.get(artifact_id, [])
                        if str(item)
                    }
                ),
                "resultCoverage": str(
                    getattr(bundle, "result_coverage", "") or ""
                ),
                "originalRowCount": int(
                    getattr(bundle, "original_row_count", 0) or 0
                ),
                "storedRowCount": len(
                    list(getattr(bundle, "rows", None) or [])
                ),
                "resultArtifacts": _result_artifact_receipts(run_result),
            }
        )
    return sorted(
        receipts,
        key=lambda item: (
            item["generation"],
            item["queryArtifactId"],
        ),
    )


def build_grounded_recovery_payload(
    session: Any,
    *,
    thread_id: str,
    run_id: str,
) -> dict[str, Any]:
    runtime = getattr(session, "runtime", None)
    workspace = getattr(session, "context_workspace", None)
    goal_contract = _model_payload(
        getattr(session, "question_goal_contract", None)
    )
    graph_receipt = _model_payload(
        getattr(session, "execution_graph_receipt", None)
    )
    graph_edges = [
        _model_payload(item)
        for item in list(
            getattr(session, "execution_graph_edges", None) or []
        )
    ]
    active_contract = _model_payload(
        getattr(runtime, "active_contract", None)
    )
    branches: list[Any] = []
    for branch_id, context in sorted(
        dict(getattr(session, "query_branch_contexts", None) or {}).items()
    ):
        spec = getattr(context, "spec", None)
        ledger = getattr(context, "semantic_ledger", None)
        refs = getattr(ledger, "refs", None)
        paths = getattr(ledger, "paths", None)
        branch_runtime = getattr(context, "runtime", None)
        branches.append(
            {
                "branchId": str(branch_id),
                "spec": _model_payload(spec),
                "status": str(getattr(context, "status", "") or ""),
                "contractScopeQueryIds": list(
                    getattr(context, "contract_scope_query_ids", None) or []
                ),
                "dependencyQueryIds": list(
                    getattr(context, "dependency_query_ids", None) or []
                ),
                "dependencyGoalIds": list(
                    getattr(context, "dependency_goal_ids", None) or []
                ),
                "semanticRefIds": refs() if callable(refs) else [],
                "semanticPaths": paths() if callable(paths) else [],
                "verifiedArtifactIds": list(
                    getattr(context, "verified_artifact_ids", None) or []
                ),
                "lastGaps": [
                    dict(item)
                    for item in list(
                        getattr(context, "last_gaps", None) or []
                    )
                    if isinstance(item, dict)
                ],
                "runtimePhase": str(
                    getattr(branch_runtime, "phase", "") or ""
                ),
                "activeGeneration": int(
                    getattr(branch_runtime, "active_generation", 0) or 0
                ),
                "activeAttemptId": str(
                    getattr(branch_runtime, "active_attempt_id", "") or ""
                ),
            }
        )
    payload: dict[str, Any] = {
        "schemaVersion": RECOVERY_SCHEMA_VERSION,
        "artifactKind": "GROUNDED_CONTEXT_RECOVERY",
        "identityBinding": {
            "threadFingerprint": str(
                getattr(workspace, "thread_fingerprint", "") or ""
            ),
            "runFingerprint": str(
                getattr(workspace, "run_fingerprint", "") or ""
            ),
            "ownerFingerprint": str(
                getattr(workspace, "owner_fingerprint", "") or ""
            ),
            "requestFingerprint": str(
                getattr(workspace, "request_fingerprint", "") or ""
            ),
            "threadIdHash": hashlib.sha256(
                str(thread_id or "").encode("utf-8")
            ).hexdigest(),
            "runIdHash": hashlib.sha256(
                str(run_id or "").encode("utf-8")
            ).hexdigest(),
        },
        "phase": {
            "runtimePhase": str(getattr(runtime, "phase", "") or ""),
            "activeGeneration": int(
                getattr(runtime, "active_generation", 0) or 0
            ),
            "activeAttemptId": str(
                getattr(runtime, "active_attempt_id", "") or ""
            ),
            "activeExecutionMode": str(
                getattr(runtime, "active_execution_mode", "") or ""
            ),
            "dataCollectionSealed": bool(
                getattr(session, "data_collection_sealed", False)
            ),
            "analysisSkillStarted": bool(
                getattr(session, "analysis_skill_started", False)
            ),
        },
        "question": str(getattr(runtime, "question", "") or ""),
        "goalContract": {
            "fingerprint": _fingerprint(goal_contract) if goal_contract else "",
            "contract": goal_contract,
        },
        "semanticReceipts": _semantic_receipts(session),
        "executionGraph": {
            "generation": int(
                getattr(session, "execution_graph_generation", 0) or 0
            ),
            "fingerprint": str(
                getattr(session, "execution_graph_fingerprint", "") or ""
            ),
            "receipt": graph_receipt,
            "edges": graph_edges,
            "branches": branches,
        },
        "activeContract": {
            "fingerprint": _fingerprint(active_contract) if active_contract else "",
            "contract": active_contract,
        },
        "queryArtifactReceipts": _query_artifact_receipts(session),
    }
    payload["recoveryFingerprint"] = _fingerprint(payload)
    return payload


def persist_grounded_recovery_payload(
    session: Any,
    payload: dict[str, Any],
    *,
    settings: Any,
) -> str:
    workspace = getattr(session, "context_workspace", None)
    if workspace is None or settings is None:
        return ""
    fingerprint = str(payload.get("recoveryFingerprint") or "")
    if not fingerprint:
        return ""
    relative_ref = "context/recovery_%s.json" % fingerprint
    result = WorkspaceArtifactStore(
        settings,
        workspace.core_scratch_root,
    ).write_json(
        "context",
        "recovery_%s.json" % fingerprint,
        payload,
        preview_chars=0,
        immutable=True,
    )
    if not result.get("success") or not result.get("immutable"):
        return ""
    return "/workspace/%s" % relative_ref


def build_grounded_model_recovery_message(
    payload: dict[str, Any],
    artifact_ref: str,
) -> HumanMessage:
    goal_contract = dict(payload.get("goalContract") or {})
    goal_payload = goal_contract.get("contract")
    goals = []
    if isinstance(goal_payload, dict):
        goals = list(goal_payload.get("goals") or [])
    semantic_receipts = [
        {
            "refId": str(item.get("refId") or ""),
            "contentHash": str(item.get("contentHash") or ""),
            "path": str(item.get("path") or ""),
        }
        for item in list(payload.get("semanticReceipts") or [])
        if isinstance(item, dict)
    ]
    query_receipts = [
        {
            "queryArtifactId": str(item.get("queryArtifactId") or ""),
            "contractFingerprint": str(
                item.get("contractFingerprint") or ""
            ),
            "sqlFingerprint": str(item.get("sqlFingerprint") or ""),
            "goalIds": list(item.get("goalIds") or []),
            "resultCoverage": str(item.get("resultCoverage") or ""),
            "resultArtifacts": list(item.get("resultArtifacts") or []),
        }
        for item in list(payload.get("queryArtifactReceipts") or [])
        if isinstance(item, dict)
    ]
    summary = {
        "contextRecoverySummary": {
            "schemaVersion": payload.get("schemaVersion"),
            "recoveryFingerprint": payload.get("recoveryFingerprint"),
            "recoveryArtifactRef": artifact_ref,
            "identityBinding": dict(payload.get("identityBinding") or {}),
            "phase": dict(payload.get("phase") or {}),
            "question": str(payload.get("question") or ""),
            "goalContract": {
                "fingerprint": str(goal_contract.get("fingerprint") or ""),
                "goals": goals,
            },
            "semanticReceipts": semantic_receipts,
            "executionGraph": dict(payload.get("executionGraph") or {}),
            "activeContract": dict(payload.get("activeContract") or {}),
            "queryArtifactReceipts": query_receipts,
        },
        "instruction": (
            "The durable checkpoint and raw tool log remain unchanged. Continue "
            "from this server-generated recovery summary. Read recoveryArtifactRef "
            "through /workspace only when a detail omitted from this summary is required."
        ),
    }
    return HumanMessage(content=_canonical_json(summary))


def compact_summary_to_reference_only(
    payload: dict[str, Any],
    artifact_ref: str,
) -> HumanMessage:
    goal_contract = dict(payload.get("goalContract") or {})
    execution_graph = dict(payload.get("executionGraph") or {})
    graph_receipt = execution_graph.get("receipt")
    summary = {
        "contextRecoverySummary": {
            "schemaVersion": payload.get("schemaVersion"),
            "recoveryFingerprint": payload.get("recoveryFingerprint"),
            "recoveryArtifactRef": artifact_ref,
            "identityBinding": dict(payload.get("identityBinding") or {}),
            "phase": dict(payload.get("phase") or {}),
            "question": str(payload.get("question") or ""),
            "goalContractFingerprint": str(
                goal_contract.get("fingerprint") or ""
            ),
            "semanticRefIds": [
                str(item.get("refId") or "")
                for item in list(payload.get("semanticReceipts") or [])
                if isinstance(item, dict) and item.get("refId")
            ],
            "executionGraph": {
                "generation": execution_graph.get("generation"),
                "fingerprint": execution_graph.get("fingerprint"),
                "receipt": graph_receipt,
            },
            "queryArtifactIds": [
                str(item.get("queryArtifactId") or "")
                for item in list(payload.get("queryArtifactReceipts") or [])
                if isinstance(item, dict) and item.get("queryArtifactId")
            ],
        },
        "instruction": (
            "Recover omitted governed details from recoveryArtifactRef. The raw "
            "checkpoint remains authoritative and was not deleted or rewritten."
        ),
    }
    return HumanMessage(content=_canonical_json(summary))
