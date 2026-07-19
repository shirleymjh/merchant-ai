from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from merchant_ai.config import Settings
from merchant_ai.graph.action_contract import (
    action_prerequisite_gaps,
    action_state_flag_ready,
    contract_block_observation,
    state_path_ready,
    state_path_value,
)
from merchant_ai.graph.state import AgentState, mark_terminal_status, merge_agent_state_update
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.models import (
    ActionResult,
    AgentDecision,
    ArtifactRef,
    ContextBudgetReport,
    ContextCompressionEvent,
    GraphValidationGap,
    MiddlewareEvent,
    RetrievalIssue,
    ToolCallExecutionResult,
    ToolCallLedgerEntry,
    ToolCallRecoveryEvent,
    ToolCallRequest,
    WorkspaceManifest,
    WorkspaceManifestEntry,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.context_assembly import ContextAssembler
from merchant_ai.services.context import ContextManager
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.memory import (
    MemoryQueryUnderstandingService,
    create_memory_store,
    memory_query_hash,
    memory_recall_trace_for,
    normalize_memory_recall_issues,
    retrieval_context_from_state,
    truncate_memory_text_by_tokens,
)
from merchant_ai.services.memory_constraints import build_memory_constraints
from merchant_ai.services.observability import artifact_ref_from_path
from merchant_ai.services.text_parsing import safe_ascii_component


TERMINAL_TOOL_STATUSES = {"success", "failed", "error", "timeout", "rate_limited", "circuit_blocked", "skipped"}


class HarnessMiddleware:
    """Small, composable runtime middleware around the LeadAgent loop."""

    name = "middleware"
    failure_policy = "open"
    read_keys: List[str] = []
    write_keys: List[str] = []

    def before_policy(self, state: AgentState) -> AgentState:
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        return state

    def after_action(self, state: AgentState) -> AgentState:
        return state


class MiddlewareChain:
    def __init__(self, middlewares: Iterable[HarnessMiddleware]):
        self.middlewares = list(middlewares)

    def before_policy(self, state: AgentState) -> AgentState:
        self._record_chain_order(state, "before_policy")
        for middleware in self.middlewares:
            state = self._safe_call(middleware, "before_policy", state)
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        self._record_chain_order(state, "before_action")
        for middleware in self.middlewares:
            state = self._safe_call(middleware, "before_action", state, decision)
        return state

    def capture_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        from merchant_ai.graph.policy import AgentActionRegistry

        action = AgentActionRegistry().get(decision.selected_action)
        state["_pending_action_contract"] = {
            "action": decision.selected_action,
            "node": decision.selected_node,
            "expectedStateKeys": list(action.expected_state_keys),
            "expectedStateFlags": list(action.expected_state_flags),
            "preValues": action_contract_values(state, action.expected_state_keys, action.expected_state_flags),
            "inputFingerprint": orchestration_progress_fingerprint(state),
        }
        return state

    def after_action(self, state: AgentState) -> AgentState:
        if not state.get("_pending_action_contract"):
            return state
        self._record_chain_order(state, "after_action")
        for middleware in self.middlewares:
            state = self._safe_call(middleware, "after_action", state)
        # LangGraph state updates cannot observe a local key deletion. Persist
        # an explicit empty value so a terminal finalizer cannot replay the
        # same contract on the following graph edge.
        state["_pending_action_contract"] = {}
        return state

    def _safe_call(self, middleware: HarnessMiddleware, method: str, state: AgentState, *args: Any) -> AgentState:
        before_fingerprint = self._state_fingerprint(state)
        try:
            handler = getattr(middleware, method)
            returned = handler(state, *args)
            merged = merge_agent_state_update(state, returned or {})
            changed_keys = self._changed_keys(before_fingerprint, self._state_fingerprint(merged))
            if changed_keys and not (middleware.name == "context_snapshot" and method == "before_action"):
                append_middleware_event(
                    merged,
                    middleware.name,
                    method,
                    status="observed",
                    code="MIDDLEWARE_STATE_DELTA",
                    message="middleware updated state",
                    metadata={"changedKeys": changed_keys[:20], "changedKeyCount": len(changed_keys)},
                )
            return merged
        except Exception as exc:
            policy = getattr(middleware, "failure_policy", "open") or "open"
            severity = "critical" if policy == "closed" else "error"
            append_middleware_event(
                state,
                middleware.name,
                method,
                status="error",
                code="MIDDLEWARE_ERROR",
                message=str(exc),
                metadata={"failurePolicy": policy},
            )
            if policy == "closed":
                mark_terminal_status(state, "blocked", "MIDDLEWARE_FAIL_CLOSED", middleware.name, str(exc))
                state["middleware_blocked"] = True
                state["middleware_loop_blocked"] = True
                state["chat_bi_completed"] = True
                state.setdefault("safety_finish_reasons", []).append(
                    {
                        "source": middleware.name,
                        "finishReason": "middleware_fail_closed",
                        "message": str(exc)[:500],
                    }
                )
                if not state.get("answer"):
                    state["answer"] = "系统安全中间件执行失败，本轮已停止继续调用工具。"
                append_middleware_event(
                    state,
                    middleware.name,
                    method,
                    status="blocked",
                    code="MIDDLEWARE_FAIL_CLOSED",
                    message=str(exc),
                    metadata={"failurePolicy": policy, "severity": severity},
                )
            return state

    def _record_chain_order(self, state: AgentState, stage: str) -> None:
        key = "_middleware_chain_order_recorded"
        recorded = state.setdefault(key, {})
        if recorded.get(stage):
            return
        recorded[stage] = True
        append_middleware_event(
            state,
            "middleware_chain",
            stage,
            status="observed",
            code="MIDDLEWARE_CHAIN_ORDER",
            message="configured middleware order",
            metadata={"order": [middleware.name for middleware in self.middlewares], "count": len(self.middlewares)},
        )

    def _state_fingerprint(self, state: AgentState) -> Dict[str, str]:
        tracked = [
            "tool_call_results",
            "tool_call_ledger",
            "context_packages",
            "context_manifests",
            "runtime_checkpoints",
            "middleware_events",
            "skill_lifecycle_records",
            "memory_context",
            "runtime_context",
            "summary_context",
            "answer",
            "next_action",
        ]
        return {key: stable_json_hash(state.get(key)) for key in tracked if key in state}

    def _changed_keys(self, before: Dict[str, str], after: Dict[str, str]) -> List[str]:
        keys = sorted(set(before) | set(after))
        return [key for key in keys if before.get(key) != after.get(key)]


class CancellationMiddleware(HarnessMiddleware):
    name = "cancellation"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        run_id = str(state.get("run_id") or "")
        if not run_id or state.get("run_canceled"):
            return state
        path = self.settings.resolved_workspace_path / "run_events" / "runs" / ("%s.json" % run_id)
        status = ""
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = str(payload.get("status") or "").upper()
        except Exception:
            status = ""
        if status in {"CANCELED", "CANCELLED"}:
            state["run_canceled"] = True
            state["chat_bi_completed"] = True
            state["answer"] = "本次运行已取消。"
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="blocked",
                code="RUN_CANCELED",
                message="run status is CANCELED in run event store",
            )
        return state


class PermissionMiddleware(HarnessMiddleware):
    name = "permission"
    failure_policy = "closed"

    def before_policy(self, state: AgentState) -> AgentState:
        slots = state.get("route_slots")
        operation = str(getattr(slots, "operation", "") or "")
        if operation == "write_requested":
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="blocked",
                code="WRITE_OPERATION_REQUIRES_HUMAN",
                message="write operation is routed to human clarification instead of tool execution",
            )
        return state


class RunBudgetMiddleware(HarnessMiddleware):
    name = "run_budget"
    failure_policy = "closed"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        report = build_run_budget_report(state, self.settings)
        state["run_budget_report"] = report
        if not report.get("exhausted"):
            return state
        state["run_budget_exhausted"] = True
        state["chat_bi_completed"] = True
        state.setdefault("safety_finish_reasons", []).append(
            {
                "source": "run_budget",
                "finishReason": "run_budget_exhausted",
                "message": report.get("reason", ""),
                "report": report,
            }
        )
        state["safety_finish_reasons"] = state["safety_finish_reasons"][-50:]
        if not state.get("answer"):
            state["answer"] = "本轮运行已达到系统预算上限，我会基于当前已完成的结果给出回答，并标注未完成部分。"
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="blocked",
            code="RUN_BUDGET_EXHAUSTED",
            message=report.get("reason", "run budget exhausted"),
            metadata=report,
        )
        return state


class ActionContractMiddleware(HarnessMiddleware):
    """Validate the selected action against declared state contracts."""

    name = "action_contract"
    failure_policy = "closed"

    def __init__(self):
        from merchant_ai.graph.policy import AgentActionRegistry

        self.registry = AgentActionRegistry()

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        action = self.registry.get(decision.selected_action)
        missing_keys, missing_flags = action_prerequisite_gaps(state, action)
        if not missing_keys and not missing_flags:
            if action.id != "observe_contract_block":
                state["contract_block_observation"] = {}
                state["contract_block_observed"] = False
            return state

        original_action = decision.selected_action
        original_node = decision.selected_node
        observation_action = self.registry.get("observe_contract_block")
        state["contract_block_observation"] = contract_block_observation(
            action,
            missing_keys,
            missing_flags,
            reason=decision.reason,
        )
        state["contract_block_observed"] = False
        decision.selected_action = observation_action.id
        decision.selected_node = observation_action.node
        decision.source = "contract_block"
        decision.reason = (
            "Action contract blocked %s; return the missing prerequisite observation to LeadAgent "
            "without executing a fallback. missing keys=%s flags=%s. %s"
            % (original_action, missing_keys, missing_flags, decision.reason or "")
        ).strip()
        decision.available_actions = [observation_action.id]
        append_middleware_event(
            state,
            self.name,
            "before_action",
            status="blocked",
            code="ACTION_CONTRACT_BLOCKED",
            message="selected action was blocked; no business fallback was executed",
            metadata={
                "fromAction": original_action,
                "fromNode": original_node,
                "observationAction": observation_action.id,
                "observationNode": observation_action.node,
                "missingStateKeys": missing_keys,
                "missingStateFlags": missing_flags,
            },
        )
        return state

    def after_action(self, state: AgentState) -> AgentState:
        snapshot = state.get("_pending_action_contract") or {}
        if not isinstance(snapshot, dict) or not snapshot.get("action"):
            return state
        action_id = str(snapshot.get("action") or "")
        expected_keys = list(snapshot.get("expectedStateKeys") or [])
        expected_flags = list(snapshot.get("expectedStateFlags") or [])
        missing_keys = [key for key in expected_keys if not state_path_ready(state, key)]
        missing_flags = [flag for flag in expected_flags if not action_state_flag_ready(state, flag)]
        post_values = action_contract_values(state, expected_keys, expected_flags)
        pre_values = snapshot.get("preValues") or {}
        contract_progressed = stable_json_hash(pre_values) != stable_json_hash(post_values)
        output_fingerprint = orchestration_progress_fingerprint(state)
        input_fingerprint = str(snapshot.get("inputFingerprint") or "")
        if missing_keys or missing_flags:
            status = "failed"
            retryable = True
            message = "action did not establish declared postconditions"
        elif output_fingerprint == input_fingerprint and not contract_progressed:
            status = "no_progress"
            retryable = True
            message = "action completed without observable contract progress"
        else:
            status = "success"
            retryable = False
            message = "action established its declared postconditions"
        state["last_action_result"] = ActionResult(
            action=action_id,
            node=str(snapshot.get("node") or ""),
            status=status,
            message=message,
            retryable=retryable,
        )
        for trace in reversed(state.get("action_history") or []):
            if str(getattr(trace, "action", "") or "") == action_id and str(getattr(trace, "status", "") or "") == "selected":
                trace.status = status
                break
        outcome = {
            "action": action_id,
            "node": str(snapshot.get("node") or ""),
            "status": status,
            "inputFingerprint": input_fingerprint,
            "outputFingerprint": output_fingerprint,
            "missingStateKeys": missing_keys,
            "missingStateFlags": missing_flags,
            "preValues": pre_values,
            "postValues": post_values,
        }
        state.setdefault("action_outcomes", []).append(outcome)
        state["action_outcomes"] = state["action_outcomes"][-64:]
        append_middleware_event(
            state,
            self.name,
            "after_action",
            status="ok" if status == "success" else status,
            code="ACTION_CONTRACT_%s" % status.upper(),
            message=message,
            metadata=outcome,
        )
        return state

class ContextBudgetMiddleware(HarnessMiddleware):
    name = "context_budget"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        stage = "policy_round_%s" % int(state.get("react_round") or 0)
        estimated_tokens = estimate_context_tokens(state)
        window = max(1, int(self.settings.context_window_tokens or 1))
        ratio = round(estimated_tokens / window, 4)
        threshold = float(self.settings.context_compaction_threshold_ratio or 0.85)
        report = ContextBudgetReport(
            stage=stage,
            window_tokens=window,
            estimated_tokens=estimated_tokens,
            usage_ratio=ratio,
            threshold_ratio=threshold,
            over_budget=ratio >= threshold,
            protected_fact_count=sum(snapshot_protected_fact_count(snapshot) for snapshot in state.get("context_snapshots", [])),
            artifact_count=len(workspace_entries(state)),
            summary_chars=len(str(state.get("summary_context") or "")),
            decision="summarize" if ratio >= threshold else "keep_inline",
        )
        state.setdefault("context_budget_reports", []).append(report)
        state["context_budget_reports"] = state["context_budget_reports"][-24:]
        return state


class DynamicContextMiddleware(HarnessMiddleware):
    name = "dynamic_context"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assembler = ContextAssembler(settings)

    def before_policy(self, state: AgentState) -> AgentState:
        stage = "policy_round_%s" % int(state.get("react_round") or 0)
        stale_before_render = bool(state.get("_runtime_context_stale"))
        pending_warnings = [str(item) for item in (state.get("pending_tool_loop_warnings") or []) if str(item).strip()]
        if pending_warnings:
            state["tool_loop_warning"] = "\n".join(pending_warnings)
            state["pending_tool_loop_warnings"] = []
        injection = self.assembler.runtime_injection(state, stage=stage)
        state["runtime_injection"] = injection
        state["runtime_context"] = self.assembler.render_runtime_context(
            injection,
            budget_chars=int(self.settings.context_runtime_budget_chars or 6000),
        )
        state["_runtime_context_stale"] = False
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="injected",
            code="RUNTIME_CONTEXT_INJECTED",
            message="dynamic runtime context is injected through middleware",
            input_chars=len(json.dumps(injection, ensure_ascii=False, default=str)),
            output_chars=len(state.get("runtime_context") or ""),
            metadata={
                "stage": stage,
                "artifactCount": (injection.get("workspace") or {}).get("artifactCount", 0),
                "hasThreadContext": bool((injection.get("threadContext") or {}).get("restored")),
                "rerenderedAfterCompaction": stale_before_render,
                "toolFeedbackCount": len(injection.get("toolFeedback") or []),
            },
        )
        return state


class SummarizeMiddleware(HarnessMiddleware):
    name = "summarize"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        reports = state.get("context_budget_reports") or []
        report = reports[-1] if reports else None
        if not report or not context_report_over_budget(report):
            return state
        stage = context_report_stage(report)
        if stage in set(state.get("_summarized_stages", [])):
            return state
        before_tokens = context_report_estimated_tokens(report)
        target_tokens = int(max(1, (self.settings.context_window_tokens or 1) * (self.settings.context_compaction_target_ratio or 0.4)))
        checkpoint = write_middleware_json_artifact(
            state,
            self.settings,
            "context",
            "runtime_checkpoint_%s.json" % stage,
            build_runtime_checkpoint_payload(state, stage),
        )
        full_context_artifact = write_middleware_json_artifact(
            state,
            self.settings,
            "context",
            "pre_compaction_context_%s.json" % stage,
            {
                key: state.get(key)
                for key in [
                    "session_context",
                    "memory_context",
                    "runtime_context",
                    "base_knowledge_context",
                    "topic_asset_context",
                    "recall_context",
                    "tool_context",
                    "summary_context",
                ]
            },
        )
        if checkpoint and (checkpoint.path or checkpoint.relative_path or checkpoint.merchant_uri):
            state.setdefault("runtime_checkpoints", []).append(checkpoint.model_dump(by_alias=True))
            state["runtime_checkpoints"] = state["runtime_checkpoints"][-8:]
        summary = build_compression_summary(state, target_tokens * 4)
        artifact = write_middleware_text_artifact(state, self.settings, "context", "summary_%s.md" % stage, summary)
        state["summary_context"] = summary
        compact_runtime_context_fields(state, target_tokens * 4)
        event = ContextCompressionEvent(
            stage=stage,
            before_tokens=before_tokens,
            after_tokens=estimate_context_tokens(state),
            target_ratio=float(self.settings.context_compaction_target_ratio or 0.4),
            summary_artifact=artifact,
            protected_keys=protected_fact_keys(state),
            reason="context budget exceeded %.2f threshold" % float(self.settings.context_compaction_threshold_ratio or 0.85),
        )
        state.setdefault("context_compression_events", []).append(event)
        state["context_compression_events"] = state["context_compression_events"][-12:]
        state.setdefault("_summarized_stages", []).append(stage)
        state["_runtime_context_stale"] = True
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="compressed",
            code="CONTEXT_SUMMARIZED",
            message="context compressed to summary artifact after runtime checkpoint flush",
            artifact_refs=[
                ref
                for ref in [checkpoint, full_context_artifact, artifact]
                if ref and (ref.path or ref.relative_path or ref.merchant_uri)
            ],
            input_chars=before_tokens * 4,
            output_chars=len(summary),
        )
        return state


class FileSystemContextMiddleware(HarnessMiddleware):
    name = "filesystem_context"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        manifest = build_workspace_manifest(state, self.settings)
        state["workspace_manifest"] = manifest
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if outputs_path:
            try:
                path = Path(outputs_path) / "workspace_manifest.json"
                path.write_text(json.dumps(manifest.model_dump(by_alias=True), ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            except Exception:
                pass
        return state


class ArtifactOffloadMiddleware(HarnessMiddleware):
    name = "artifact_offload"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        run_result = state.get("agent_run_result")
        task_results = getattr(run_result, "task_results", []) if run_result is not None else []
        seen = set(state.get("_middleware_offloaded_tasks", []))
        for task_result in task_results:
            task_id = str(getattr(task_result, "task_id", "") or "")
            if not task_id or task_id in seen:
                continue
            bundle = getattr(task_result, "query_bundle", None)
            rows = getattr(bundle, "rows", []) if bundle is not None else []
            if not isinstance(rows, list) or not rows:
                continue
            estimated_chars = len(json.dumps(rows, ensure_ascii=False, default=str))
            if len(rows) <= int(self.settings.context_artifact_inline_max_rows or 20) and estimated_chars <= int(self.settings.tool_result_offload_chars or 20000):
                continue
            artifact = write_middleware_json_artifact(state, self.settings, "tool_results", "%s_rows.json" % task_id, rows)
            seen.add(task_id)
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="offloaded",
                code="TOOL_RESULT_OFFLOADED",
                message="large node result was written to workspace artifact",
                artifact_refs=[artifact],
                input_chars=estimated_chars,
                output_chars=0,
                metadata={"taskId": task_id, "rowCount": len(rows)},
            )
        state["_middleware_offloaded_tasks"] = sorted(seen)
        return state


class ToolOutputBudgetMiddleware(HarnessMiddleware):
    name = "tool_output_budget"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        budget = max(1000, int(getattr(self.settings, "middleware_tool_output_budget_chars", 12000) or 12000))
        reports: List[Dict[str, Any]] = []
        for result in list(state.get("tool_call_results") or []):
            if not isinstance(result, ToolCallExecutionResult):
                continue
            payload = result.result or {}
            payload_chars = len(json.dumps(payload, ensure_ascii=False, default=str))
            if payload_chars <= budget or payload.get("_offloaded"):
                continue
            artifact = write_middleware_json_artifact(state, self.settings, "tool_outputs", "%s_result.json" % sanitize_artifact_name(result.id or result.name), payload)
            result.result = {
                "_offloaded": True,
                "truncated": True,
                "toolCallId": result.id,
                "toolName": result.name,
                "originalChars": payload_chars,
                "preview": json.dumps(payload, ensure_ascii=False, default=str)[: min(1000, budget)],
                "artifactRef": artifact.model_dump(by_alias=True),
            }
            reports.append({"toolCallId": result.id, "toolName": result.name, "originalChars": payload_chars, "budgetChars": budget, "artifact": artifact.model_dump(by_alias=True)})
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="offloaded",
                code="TOOL_OUTPUT_BUDGET_APPLIED",
                message="tool result exceeded output budget and was offloaded",
                artifact_refs=[artifact],
                input_chars=payload_chars,
                output_chars=len(json.dumps(result.result, ensure_ascii=False, default=str)),
                metadata=reports[-1],
            )
        if reports:
            state.setdefault("tool_output_budget_reports", []).extend(reports)
            state["tool_output_budget_reports"] = state["tool_output_budget_reports"][-50:]
        return state


class TokenUsageMiddleware(HarnessMiddleware):
    name = "token_usage"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        stage = "policy_round_%s" % int(state.get("react_round") or 0)
        report = {
            "stage": stage,
            "estimatedInputTokens": estimate_context_tokens(state),
            "questionTokens": estimate_text_tokens(str(state.get("question") or "")),
            "runtimeContextTokens": estimate_text_tokens(str(state.get("runtime_context") or "")),
            "memoryContextTokens": estimate_text_tokens(str(state.get("memory_context") or "")),
            "summaryContextTokens": estimate_text_tokens(str(state.get("summary_context") or "")),
            "toolResultCount": len(state.get("tool_call_results") or []),
            "nodeResultCount": len(getattr(state.get("agent_run_result"), "task_results", []) or []),
        }
        existing = state.get("token_usage_reports") or []
        if existing and existing[-1].get("stage") == stage:
            existing[-1] = report
        else:
            existing.append(report)
        state["token_usage_reports"] = existing[-50:]
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="observed",
            code="TOKEN_USAGE_ESTIMATED",
            message="estimated token usage for current agent state",
            metadata=report,
        )
        return state


class SafetyFinishReasonMiddleware(HarnessMiddleware):
    name = "safety_finish_reason"

    def before_policy(self, state: AgentState) -> AgentState:
        reasons: List[Dict[str, Any]] = []
        for source, error in [
            ("planner", state.get("planner_provider_error")),
            ("planner_llm", getattr(getattr(state.get("planner"), "llm", None), "last_error", "")),
            ("forced_tool_loop", state.get("forced_tool_loop_stop_message")),
        ]:
            reason = classify_finish_reason(error)
            if reason:
                reasons.append({"source": source, "finishReason": reason, "message": str(error or "")[:500]})
        if state.get("middleware_loop_blocked"):
            reasons.append({"source": "middleware", "finishReason": "tool_loop_hard_stop", "message": str(state.get("forced_tool_loop_stop_message") or "loop guard blocked repeated tool calls")[:500]})
        if not reasons:
            return state
        existing = state.get("safety_finish_reasons") or []
        fingerprints = {stable_json_hash(item) for item in existing}
        for reason in reasons:
            if stable_json_hash(reason) not in fingerprints:
                existing.append(reason)
                append_middleware_event(
                    state,
                    self.name,
                    "before_policy",
                    status="observed",
                    code="SAFETY_FINISH_REASON_RECORDED",
                    message=reason["message"],
                    metadata=reason,
                )
        state["safety_finish_reasons"] = existing[-50:]
        return state


class ToolCallRecoveryMiddleware(HarnessMiddleware):
    name = "tool_call_recovery"

    def before_policy(self, state: AgentState) -> AgentState:
        results = state.get("tool_call_results") or []
        results = list(results) + synthetic_missing_tool_results(state)
        ledger_keys = {
            "%s:%s" % (entry.tool_call_id, entry.status)
            for entry in state.get("tool_call_ledger", [])
            if hasattr(entry, "tool_call_id")
        }
        normalized: List[Any] = []
        for item in results:
            result = normalize_tool_result(item)
            if not result.status or result.status not in TERMINAL_TOOL_STATUSES:
                before = result.status
                result.status = "failed"
                result.error_type = result.error_type or "MISSING_TOOL_RESULT"
                result.error_message = result.error_message or "tool call did not produce a terminal result"
                event = ToolCallRecoveryEvent(
                    tool_call_id=result.id,
                    tool_name=result.name,
                    stage="before_policy",
                    action="patch_missing_terminal_result",
                    reason="non terminal tool status would break provider message order",
                    status_before=before,
                    status_after=result.status,
                )
                state.setdefault("tool_call_recovery_events", []).append(event)
                append_middleware_event(
                    state,
                    self.name,
                    "before_policy",
                    status="patched",
                    code="DANGLING_TOOL_CALL_PATCHED",
                    message="patched non-terminal tool call result",
                    metadata={"toolCallId": result.id, "toolName": result.name, "statusBefore": before},
                )
            key = "%s:%s" % (result.id, result.status)
            if key not in ledger_keys:
                state.setdefault("tool_call_ledger", []).append(
                    ToolCallLedgerEntry(
                        tool_call_id=result.id,
                        tool_name=result.name,
                        stage="before_policy",
                        status=result.status,
                        duration_ms=result.duration_ms,
                        attempts=result.attempts,
                        cache_hit=result.cache_hit,
                        rate_limited=result.rate_limited,
                        target=result.target,
                        error_type=result.error_type,
                        error_message=result.error_message,
                    )
                )
                ledger_keys.add(key)
            normalized.append(result)
        state["tool_call_results"] = normalized[-100:]
        state["tool_call_ledger"] = state.get("tool_call_ledger", [])[-200:]
        state["tool_call_recovery_events"] = state.get("tool_call_recovery_events", [])[-50:]
        return state


class LoopGuardMiddleware(HarnessMiddleware):
    name = "loop_guard"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        if state.get("middleware_loop_blocked"):
            return state
        plan = state.get("plan")
        if state.get("human_clarification_required") or bool(
            plan and getattr(plan, "clarification_needs", None)
        ):
            return state
        self._check_tool_call_loops(state)
        if state.get("middleware_loop_blocked"):
            return state
        outcomes = state.get("action_outcomes") or []
        threshold = max(2, int(self.settings.middleware_loop_guard_threshold or 3))
        if len(outcomes) < threshold:
            return state
        recent = outcomes[-threshold:]
        actions = [str(item.get("action") or "") for item in recent if isinstance(item, dict)]
        if not actions or any(action in {"answer_data", "answer_rule", "cache_answer", "ask_human", "terminal_end"} for action in actions):
            return state
        statuses = [str(item.get("status") or "") for item in recent if isinstance(item, dict)]
        if any(status == "success" for status in statuses):
            return state
        transitions = {
            "%s:%s:%s"
            % (item.get("action") or "", item.get("inputFingerprint") or "", item.get("outputFingerprint") or "")
            for item in recent
            if isinstance(item, dict)
        }
        output_fingerprints = {
            str(item.get("outputFingerprint") or "") for item in recent if isinstance(item, dict)
        }
        same_transition = len(transitions) == 1
        short_cycle_without_progress = len(output_fingerprints) <= 2
        if not same_transition and not short_cycle_without_progress:
            return state
        state["middleware_loop_blocked"] = True
        append_runtime_guard_gap(
            state,
            GraphValidationGap(
                code="LOOP_DETECTED",
                reason="LeadAgent repeated a transition or short action cycle without observable contract progress",
                evidence="actions=%s outcomes=%d" % (actions, threshold),
            ),
        )
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="blocked",
            code="LOOP_DETECTED",
            message="blocked action cycle without observable progress",
            metadata={"actions": actions, "outcomes": recent},
        )
        return state

    def _check_tool_call_loops(self, state: AgentState) -> None:
        calls = collect_tool_call_requests(state)
        if not calls:
            return
        thread_id = str(state.get("thread_id") or state.get("run_id") or "default")
        seen_by_thread = state.setdefault("tool_loop_seen_call_ids", {})
        if not isinstance(seen_by_thread, dict):
            seen_by_thread = {}
            state["tool_loop_seen_call_ids"] = seen_by_thread
        seen_ids = set(seen_by_thread.get(thread_id) or [])
        history_by_thread = state.setdefault("tool_loop_history", {})
        if not isinstance(history_by_thread, dict):
            history_by_thread = {}
            state["tool_loop_history"] = history_by_thread
        history = list(history_by_thread.get(thread_id) or [])
        window_size = max(1, int(getattr(self.settings, "middleware_tool_loop_window_size", 20) or 20))
        combo_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        combo_display: Dict[str, Dict[str, Any]] = {}
        for call in calls:
            call_id = str(call.get("id") or "").strip()
            if call_id and call_id in seen_ids:
                continue
            if call_id:
                seen_ids.add(call_id)
            name = str(call.get("name") or "")
            if not name:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            key_args = loop_guard_key_args(args)
            params_hash = stable_json_hash(key_args)
            fingerprint = "%s:%s" % (name, params_hash)
            history.append({"toolName": name, "fingerprint": fingerprint, "paramsHash": params_hash, "keyArgs": key_args})
            combo_display[fingerprint] = {"toolName": name, "paramsHash": params_hash, "keyArgs": key_args}
        history = history[-window_size:]
        history_by_thread[thread_id] = history
        seen_by_thread[thread_id] = list(seen_ids)[-max(window_size * 3, 50):]
        for item in history:
            name = str(item.get("toolName") or "")
            fingerprint = str(item.get("fingerprint") or "")
            if not name or not fingerprint:
                continue
            combo_counts[fingerprint] = combo_counts.get(fingerprint, 0) + 1
            type_counts[name] = type_counts.get(name, 0) + 1
            combo_display.setdefault(fingerprint, {"toolName": name, "paramsHash": item.get("paramsHash", ""), "keyArgs": item.get("keyArgs", {})})
        repeat_warning = max(1, int(getattr(self.settings, "middleware_tool_repeat_warning_threshold", 3) or 3))
        repeat_stop = max(repeat_warning, int(getattr(self.settings, "middleware_tool_repeat_hard_stop_threshold", 5) or 5))
        type_warning = max(1, int(getattr(self.settings, "middleware_tool_type_warning_threshold", 30) or 30))
        type_stop = max(type_warning, int(getattr(self.settings, "middleware_tool_type_hard_stop_threshold", 50) or 50))
        hard_combo = [(key, count) for key, count in combo_counts.items() if count >= repeat_stop]
        hard_type = [(name, count) for name, count in type_counts.items() if count >= type_stop]
        warn_combo = [(key, count) for key, count in combo_counts.items() if count >= repeat_warning]
        warn_type = [(name, count) for name, count in type_counts.items() if count >= type_warning]
        if hard_combo or hard_type:
            reason = "repeated tool call hard stop"
            if hard_combo:
                item = combo_display.get(hard_combo[0][0], {})
                reason = "same tool+params repeated %d times: %s" % (hard_combo[0][1], item.get("toolName", ""))
            elif hard_type:
                reason = "same tool type repeated %d times: %s" % (hard_type[0][1], hard_type[0][0])
            state["middleware_loop_blocked"] = True
            state["chat_bi_completed"] = True
            state["tool_call_requests"] = []
            state["forced_tool_loop_stop_message"] = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."
            append_runtime_guard_gap(
                state,
                GraphValidationGap(
                    code="TOOL_CALL_LOOP_DETECTED",
                    reason=reason,
                    evidence=json.dumps({"comboCounts": combo_counts, "typeCounts": type_counts}, ensure_ascii=False, default=str)[:1000],
                ),
            )
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="blocked",
                code="TOOL_CALL_LOOP_HARD_STOP",
                message=reason,
                metadata={"comboCounts": combo_counts, "typeCounts": type_counts},
            )
            return
        if warn_combo or warn_type:
            warnings = []
            for key, count in warn_combo[:5]:
                item = combo_display.get(key, {})
                warnings.append("工具 %s 使用相同参数已重复 %d 次，请停止重复调用，基于已有结果总结或换工具。" % (item.get("toolName", ""), count))
            for name, count in warn_type[:5]:
                warnings.append("工具类型 %s 已调用 %d 次，请收敛调用次数，避免工具循环。" % (name, count))
            warning = "\n".join(warnings)
            pending = list(state.get("pending_tool_loop_warnings") or [])
            if warning and warning not in pending:
                pending.append(warning)
            state["pending_tool_loop_warnings"] = pending[-5:]
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="warning",
                code="TOOL_CALL_LOOP_WARNING",
                message=warning,
                metadata={"comboCounts": combo_counts, "typeCounts": type_counts},
            )


class MerchantMemoryRecallMiddleware(HarnessMiddleware):
    """Recall Diana's tenant-scoped personal memory snapshot.

    The explicit name avoids confusion with Deep Agents' AGENTS.md-style
    ``MemoryMiddleware``, which is intentionally not enabled in this runtime.
    """

    name = "memory"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.memory_store = create_memory_store(settings)
        self.query_understanding = MemoryQueryUnderstandingService(settings)

    def before_policy(self, state: AgentState) -> AgentState:
        if state.get("topic_routed") is False:
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="skipped",
                code="MEMORY_WAITING_FOR_TOPIC",
                message="memory recall waits for topic workspace",
            )
            return state
        if state.get("memory_recalled") is False:
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="skipped",
                code="MEMORY_WAITING_FOR_RECALL_NODE",
                message="governed memory recall is owned by the recall_memory node after topic routing",
            )
            return state
        trace = dict(state.get("memory_injection_trace") or {})
        if state.get("_memory_snapshot_locked") and trace:
            status = str(state.get("memory_recall_status") or trace.get("status") or "not_started")
            usable_flag = trace.get("usableSnapshot") if "usableSnapshot" in trace else status in {"success", "empty", "degraded"}
            usable = bool(usable_flag) and status in {"success", "empty", "degraded"}
            if usable and status != "empty":
                self._render_existing_snapshot(state)
                trace = dict(state.get("memory_injection_trace") or trace)
                status = str(state.get("memory_recall_status") or trace.get("status") or status)
                usable_flag = trace.get("usableSnapshot") if "usableSnapshot" in trace else status in {"success", "empty", "degraded"}
                usable = bool(usable_flag) and status in {"success", "empty", "degraded"}
            state["_memory_middleware_snapshot_ready"] = usable
            state["_memory_snapshot_locked"] = True
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="reused" if usable else "error",
                code="MEMORY_REQUEST_SNAPSHOT_LOCKED" if usable else "MEMORY_REQUEST_SNAPSHOT_UNAVAILABLE",
                message=(
                    "recall_memory node already produced the governed request snapshot"
                    if usable
                    else "recall_memory completed without a usable governed snapshot"
                ),
                metadata={
                    "constraintCount": len(state.get("memory_constraints") or []),
                    "contextFingerprint": trace.get("contextFingerprint") or "",
                    "locked": True,
                    "status": status,
                    "usableSnapshot": usable,
                },
            )
            return state
        fingerprint = self._context_fingerprint(state)
        trace_status = str(trace.get("status") or "not_started")
        trace_usable = trace.get("usableSnapshot") if "usableSnapshot" in trace else trace_status in {"success", "empty", "degraded"}
        snapshot_succeeded = bool(trace) and bool(trace_usable) and trace_status in {"success", "empty", "degraded"}
        semantic_refresh = bool(
            state.get("topic_routed")
            and not state.get("_memory_snapshot_locked")
            and trace.get("contextFingerprint") != fingerprint
            and not state.get("_memory_semantic_refresh_attempted")
        )
        if snapshot_succeeded and not semantic_refresh:
            if trace_status != "empty":
                self._render_existing_snapshot(state)
            state["_memory_middleware_snapshot_ready"] = True
            if state.get("topic_routed"):
                state["_memory_snapshot_locked"] = True
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="reused",
                code="MEMORY_REQUEST_SNAPSHOT_REUSED",
                message="request-scoped governed memory snapshot reused",
                metadata={
                    "constraintCount": len(state.get("memory_constraints") or []),
                    "contextFingerprint": fingerprint,
                    "locked": bool(state.get("_memory_snapshot_locked")),
                },
            )
            return state
        if trace_status == "failed" and state.get("_memory_middleware_retry_attempted") and not semantic_refresh:
            return state
        if semantic_refresh:
            state["_memory_semantic_refresh_attempted"] = True
        else:
            state["_memory_middleware_retry_attempted"] = True
        try:
            injection = self.memory_store.select_for_question(
                state,
                budget_tokens=int(self.settings.context_memory_budget_tokens or 1200),
            )
        except Exception as exc:
            issue = RetrievalIssue(
                code="MEMORY_RECALL_RETRY_FAILED",
                message=str(exc)[:500],
                backend=type(self.memory_store).__name__,
                lane="primary",
                stage="acquire",
                severity="blocking",
                resolved=False,
            )
            failed_trace = memory_recall_trace_for({}, [issue], enrichment_status={"acquire": "failed"})
            failed_trace["contextFingerprint"] = fingerprint
            state["memory_injection_trace"] = failed_trace
            state["memory_recall_status"] = "failed"
            state["memory_recall_issues"] = list(failed_trace.get("issues") or [])
            state["_memory_middleware_snapshot_ready"] = False
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="error",
                code=issue.code,
                message=issue.message,
                metadata={"contextFingerprint": fingerprint},
            )
            return state
        state["memory_injection"] = injection
        next_trace = memory_recall_trace_for(injection)
        next_trace["contextFingerprint"] = fingerprint
        state["memory_injection_trace"] = next_trace
        state["memory_recall_status"] = str(next_trace.get("status") or "failed")
        state["memory_recall_issues"] = list(next_trace.get("issues") or [])
        if next_trace.get("status") == "failed":
            state["_memory_middleware_snapshot_ready"] = False
            return state
        try:
            constraints = build_memory_constraints(injection)
        except Exception as exc:
            issue = RetrievalIssue(
                code="MEMORY_CONSTRAINT_COMPILATION_FAILED",
                message=str(exc)[:500],
                backend="memory_contract",
                lane="constraints",
                stage="compile_constraints",
                severity="blocking",
                resolved=False,
            )
            state["memory_injection_raw_snapshot"] = injection
            state["memory_injection"] = {}
            failed_trace = memory_recall_trace_for({}, [*normalize_memory_recall_issues(next_trace.get("issues") or []), issue])
            for key in ["selectedIds", "candidateIds", "candidateCount", "filteredReasons"]:
                if key in next_trace:
                    failed_trace[key] = next_trace[key]
            failed_trace["contextFingerprint"] = fingerprint
            state["memory_injection_trace"] = failed_trace
            state["memory_recall_status"] = "failed"
            state["memory_recall_issues"] = list(failed_trace.get("issues") or [])
            state["memory_constraints"] = []
            state["memory_constraint_trace"] = {"constraintCount": 0, "status": "failed", "error": str(exc)[:500]}
            state["_memory_middleware_snapshot_ready"] = False
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="error",
                code=issue.code,
                message=issue.message,
            )
            return state
        state["memory_constraints"] = constraints
        state["memory_constraint_trace"] = {
            "constraintCount": len(constraints),
            "requiredCount": sum(1 for item in constraints if str(item.get("enforcement") or "") == "required"),
            "clarifyCount": sum(1 for item in constraints if str(item.get("enforcement") or "") == "clarify_or_disclose"),
            "source": injection.get("source", ""),
            "selectedIds": (injection.get("memoryInjectionTrace") or {}).get("selectedIds", []),
            "status": "success" if constraints else str(next_trace.get("status") or "empty"),
        }
        self._render_existing_snapshot(state)
        recent_focus = state.get("recent_focus")
        memory_context = str(state.get("memory_context") or "")
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="observed",
            code="MEMORY_CONTEXT_READY" if memory_context or (recent_focus and not recent_focus.is_empty()) else "MEMORY_CONTEXT_EMPTY",
            message="memory is injected as compact business focus instead of full chat history",
            metadata={
                "memoryContextChars": len(memory_context),
                "recentFocusEmpty": bool(recent_focus.is_empty()) if recent_focus else True,
                "eventCount": len(injection.get("relevantEvents") or []),
                "candidateCount": (injection.get("memoryInjectionTrace") or {}).get("candidateCount", 0),
                "selectedIds": (injection.get("memoryInjectionTrace") or {}).get("selectedIds", []),
                "budgetUsedChars": (injection.get("memoryInjectionTrace") or {}).get("budgetUsedChars", 0),
                "filteredReasons": (injection.get("memoryInjectionTrace") or {}).get("filteredReasons", {}),
                "memorySource": injection.get("source", ""),
                "memoryConstraintCount": len(constraints),
            },
        )
        state["_memory_middleware_snapshot_ready"] = True
        if state.get("topic_routed"):
            state["_memory_snapshot_locked"] = True
        return state

    def _context_fingerprint(self, state: AgentState) -> str:
        self.query_understanding.ensure_state_profile(state)
        merchant_id = str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or "")
        return memory_query_hash(merchant_id, retrieval_context_from_state(state))

    def _render_existing_snapshot(self, state: AgentState) -> None:
        if state.get("memory_context"):
            return
        renderer = getattr(self.memory_store, "render_injection", None)
        if not callable(renderer):
            return
        try:
            rendered = renderer(state.get("memory_injection") or {})
        except Exception as exc:
            issue = RetrievalIssue(
                code="MEMORY_RENDER_FAILED",
                message=str(exc)[:500],
                backend=type(self.memory_store).__name__,
                lane="enrichment",
                stage="render",
                severity="warning",
                resolved=True,
                details={"answerImpact": False},
            )
            previous = dict(state.get("memory_injection_trace") or {})
            issues = normalize_memory_recall_issues([*(previous.get("issues") or []), issue])
            trace = memory_recall_trace_for(
                state.get("memory_injection") or {},
                issues,
                enrichment_status={
                    **dict(previous.get("enrichmentStatus") or {}),
                    "render": "failed",
                },
            )
            if previous.get("contextFingerprint"):
                trace["contextFingerprint"] = previous["contextFingerprint"]
            state["memory_injection_trace"] = trace
            state["memory_recall_status"] = str(trace.get("status") or "degraded")
            state["memory_recall_issues"] = list(trace.get("issues") or [])
            if state.get("memory_injection"):
                state["memory_injection"]["memoryInjectionTrace"] = dict(trace)
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="warning",
                code=issue.code,
                message=issue.message,
                metadata={"answerImpact": False},
            )
            return
        if rendered:
            state["memory_context"] = truncate_memory_text_by_tokens(
                rendered,
                int(self.settings.context_memory_budget_tokens or 1200),
            )


# Compatibility for callers/tests that imported the old domain class name.
# Runtime composition below uses the unambiguous name.
MemoryMiddleware = MerchantMemoryRecallMiddleware


class SkillMiddleware(HarnessMiddleware):
    name = "skill"

    def before_policy(self, state: AgentState) -> AgentState:
        loaded = state.get("loaded_skills") or []
        if loaded and state.get("_skill_middleware_loaded") != loaded:
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="observed",
                code="SKILL_HEADERS_LOADED",
                message="skills are exposed as thin headers; details load through semantic assets",
                metadata={"loadedSkills": loaded},
            )
            state["_skill_middleware_loaded"] = list(loaded)
        return state


class ProviderCompatibilityMiddleware(HarnessMiddleware):
    name = "provider_compatibility"

    def before_policy(self, state: AgentState) -> AgentState:
        trace = state.get("bounded_lead_llm_trace") or {}
        if trace and trace.get("status") == "error":
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="observed",
                code="LEAD_LLM_DECISION_COMPAT_ERROR",
                message=str(trace.get("reason") or "bounded lead LLM decision failed"),
                metadata=trace,
            )
        return state


class ClarificationMiddleware(HarnessMiddleware):
    """Turn human clarification into a virtual tool-call result.

    The frontend contract stays as ChatResponse.clarification, but internally the
    run now has an ask_clarification tool message that mirrors LangGraph-style
    tool interception.
    """

    name = "clarification"

    def before_policy(self, state: AgentState) -> AgentState:
        if state.get("human_clarification_required"):
            return self._intercept(state, stage="before_policy", reason="state_requires_clarification")
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        if decision.selected_action == "ask_human" or state.get("human_clarification_required"):
            return self._intercept(state, stage="before_action", reason="ask_human_selected")
        return state

    def _intercept(self, state: AgentState, stage: str, reason: str) -> AgentState:
        if state.get("_clarification_tool_intercepted"):
            return state
        question = str(state.get("human_clarification_question") or "").strip()
        if not question:
            question = "我还需要补充确认后才能继续处理这个问题。"
        options = [str(item) for item in (state.get("human_clarification_options") or []) if str(item).strip()]
        context = {
            "stage": str(state.get("human_clarification_stage") or ""),
            "type": str(state.get("human_clarification_type") or ""),
            "pendingQuestion": str(state.get("question") or ""),
            "reason": reason,
        }
        tool_call_id = "ask_clarification:%s" % (state.get("qa_id") or state.get("run_id") or uuid.uuid4().hex)
        content = format_clarification_message(question, options)
        tool_message = {
            "toolName": "ask_clarification",
            "status": "success",
            "type": "clarification_request",
            "content": content,
            "question": question,
            "options": options,
            "context": context,
            "endsRun": True,
        }
        result = ToolCallExecutionResult(
            id=tool_call_id,
            name="ask_clarification",
            status="success",
            result={"question": question, "options": options, "context": context},
            service_name="human",
            tool_message=tool_message,
            attempts=1,
            runtime_events=[
                {
                    "eventType": "tool.intercepted",
                    "toolCallId": tool_call_id,
                    "toolName": "ask_clarification",
                    "status": "success",
                    "middleware": self.name,
                    "createdAt": datetime.now().isoformat(),
                    "payload": {"stage": stage, "reason": reason, "goto": "END"},
                }
            ],
        )
        state["clarification_tool_message"] = tool_message
        state["clarification_command"] = {
            "update": {"messages": [tool_message]},
            "goto": "END",
            "reason": "clarification_required",
        }
        state.setdefault("tool_call_results", []).append(result)
        state["tool_call_results"] = state["tool_call_results"][-100:]
        state.setdefault("tool_runtime_events", []).extend(result.runtime_events)
        state["tool_runtime_events"] = state["tool_runtime_events"][-200:]
        ensure_tool_ledger_entry(state, result, stage)
        state["_clarification_tool_intercepted"] = True
        append_middleware_event(
            state,
            self.name,
            stage,
            status="intercepted",
            code="CLARIFICATION_TOOL_INTERCEPTED",
            message="ask_clarification was intercepted and converted to a terminal user clarification",
            metadata={"toolCallId": tool_call_id, "goto": "END", "optionsCount": len(options), "reason": reason},
        )
        return state


class ContextSnapshotMiddleware(HarnessMiddleware):
    name = "context_snapshot"

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        state.setdefault("middleware_action_context_hashes", {})
        packages = state.get("context_packages") or []
        last_hash = ""
        if packages:
            last = packages[-1]
            last_hash = str(getattr(last, "context_hash", "") or (last.get("contextHash", "") if isinstance(last, dict) else ""))
        state["middleware_action_context_hashes"][decision.selected_action] = last_hash
        return state


MiddlewareFactory = Callable[[Settings, ContextManager], HarnessMiddleware]


def default_middleware_registry() -> Dict[str, MiddlewareFactory]:
    return {
        "cancellation": lambda settings, context_manager: CancellationMiddleware(settings),
        "run_budget": lambda settings, context_manager: RunBudgetMiddleware(settings),
        "permission": lambda settings, context_manager: PermissionMiddleware(),
        "action_contract": lambda settings, context_manager: ActionContractMiddleware(),
        "provider_compatibility": lambda settings, context_manager: ProviderCompatibilityMiddleware(),
        "clarification": lambda settings, context_manager: ClarificationMiddleware(),
        "tool_call_recovery": lambda settings, context_manager: ToolCallRecoveryMiddleware(),
        "tool_output_budget": lambda settings, context_manager: ToolOutputBudgetMiddleware(settings),
        "loop_guard": lambda settings, context_manager: LoopGuardMiddleware(settings),
        "safety_finish_reason": lambda settings, context_manager: SafetyFinishReasonMiddleware(),
        "memory": lambda settings, context_manager: MerchantMemoryRecallMiddleware(settings),
        "dynamic_context": lambda settings, context_manager: DynamicContextMiddleware(settings),
        "token_usage": lambda settings, context_manager: TokenUsageMiddleware(settings),
        "context_budget": lambda settings, context_manager: ContextBudgetMiddleware(settings),
        "summarize": lambda settings, context_manager: SummarizeMiddleware(settings),
        "artifact_offload": lambda settings, context_manager: ArtifactOffloadMiddleware(settings),
        "filesystem_context": lambda settings, context_manager: FileSystemContextMiddleware(settings),
        "skill": lambda settings, context_manager: SkillMiddleware(),
        "context_snapshot": lambda settings, context_manager: ContextSnapshotMiddleware(context_manager),
    }


def default_middleware_order() -> List[str]:
    return [
        "cancellation",
        "run_budget",
        "permission",
        "action_contract",
        "provider_compatibility",
        "clarification",
        "tool_call_recovery",
        "tool_output_budget",
        "loop_guard",
        "safety_finish_reason",
        "memory",
        "context_budget",
        "summarize",
        "artifact_offload",
        "filesystem_context",
        "dynamic_context",
        "token_usage",
        "skill",
        "context_snapshot",
    ]


def middleware_names_from_config(value: str) -> List[str]:
    names: List[str] = []
    current: List[str] = []
    for character in str(value or ""):
        if character in {",", ";"} or character.isspace():
            name = "".join(current).strip()
            if name:
                names.append(name)
            current = []
            continue
        current.append(character)
    name = "".join(current).strip()
    if name:
        names.append(name)
    return names


def configured_middleware_order(settings: Settings, registry: Dict[str, MiddlewareFactory]) -> List[str]:
    requested = middleware_names_from_config(getattr(settings, "harness_middleware_order", ""))
    if not requested:
        return default_middleware_order()
    known = [name for name in requested if name in registry]
    remaining = [name for name in default_middleware_order() if name in registry and name not in known]
    return known + remaining


def default_harness_middlewares(settings: Settings, context_manager: ContextManager) -> List[HarnessMiddleware]:
    registry = default_middleware_registry()
    disabled = set(middleware_names_from_config(getattr(settings, "harness_middleware_disabled", "")))
    middlewares: List[HarnessMiddleware] = []
    for name in configured_middleware_order(settings, registry):
        if name in disabled:
            continue
        factory = registry.get(name)
        if factory is None:
            continue
        middlewares.append(factory(settings, context_manager))
    return middlewares


def append_middleware_event(
    state: AgentState,
    middleware: str,
    stage: str,
    status: str = "ok",
    code: str = "",
    message: str = "",
    input_chars: int = 0,
    output_chars: int = 0,
    artifact_refs: Optional[List[ArtifactRef]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    severity = middleware_event_severity(status, code)
    channel = middleware_event_channel(severity, code)
    event = MiddlewareEvent(
        event_id="mw_" + uuid.uuid4().hex,
        middleware=middleware,
        stage=stage,
        status=status,
        severity=severity,
        channel=channel,
        code=code,
        message=message,
        input_chars=input_chars,
        output_chars=output_chars,
        artifact_refs=artifact_refs or [],
        metadata=metadata or {},
    )
    state.setdefault("middleware_events", []).append(event)
    state["middleware_events"] = retain_middleware_events(state["middleware_events"], limit=200)


def middleware_event_severity(status: str, code: str) -> str:
    normalized_status = str(status or "").lower()
    normalized_code = str(code or "").upper()
    if normalized_status in {"blocked", "error"} or "FAIL_CLOSED" in normalized_code:
        return "critical" if normalized_status == "blocked" or "FAIL_CLOSED" in normalized_code else "error"
    if normalized_status in {"warning", "patched", "rerouted", "offloaded", "compressed", "intercepted"}:
        return "warning"
    return "info"


def middleware_event_channel(severity: str, code: str) -> str:
    normalized_code = str(code or "").upper()
    if severity in {"critical", "error"}:
        return "audit"
    if normalized_code in {"RUN_CANCELED", "RUN_BUDGET_EXHAUSTED", "WRITE_OPERATION_REQUIRES_HUMAN", "TOOL_CALL_LOOP_HARD_STOP"}:
        return "audit"
    if normalized_code in {"TOKEN_USAGE_ESTIMATED", "MIDDLEWARE_STATE_DELTA", "MIDDLEWARE_CHAIN_ORDER"}:
        return "debug"
    return "trace"


def retain_middleware_events(events: List[MiddlewareEvent], limit: int = 200) -> List[MiddlewareEvent]:
    if len(events or []) <= limit:
        return events or []
    audit = [event for event in events if getattr(event, "channel", "") == "audit" or getattr(event, "severity", "") in {"critical", "error"}]
    tail_budget = max(0, limit - len(audit))
    tail = [event for event in events if event not in audit][-tail_budget:] if tail_budget else []
    merged = audit + tail
    return merged[-limit:]


def append_runtime_guard_gap(state: AgentState, gap: GraphValidationGap) -> None:
    gaps = list(state.get("runtime_guard_gaps") or [])
    fingerprint = (gap.code, gap.task_id, gap.evidence, gap.reason)
    if not any((item.code, item.task_id, item.evidence, item.reason) == fingerprint for item in gaps):
        gaps.append(gap)
    state["runtime_guard_gaps"] = gaps[-32:]


def action_contract_values(state: AgentState, keys: List[str], flags: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for path in keys:
        values["key:%s" % path] = state_path_value(state, path)
    for flag in flags:
        values["flag:%s" % flag] = action_state_flag_ready(state, flag)
    return values


def orchestration_progress_fingerprint(state: AgentState) -> str:
    plan = state.get("plan")
    run_result = state.get("agent_run_result")
    task_results = list(getattr(run_result, "task_results", None) or [])
    payload = {
        "planFingerprint": query_graph_fingerprint(plan),
        "validationStatus": state.get("query_graph_validation_status"),
        "validatedPlanFingerprint": state.get("validated_query_graph_fingerprint"),
        "pendingKnowledge": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("pending_knowledge_requests", [])
        ],
        "generations": {
            "execution": state.get("execution_generation"),
            "result": state.get("result_generation"),
            "evidence": state.get("evidence_generation"),
            "analysis": state.get("analysis_generation"),
            "contractBlock": state.get("contract_block_generation"),
        },
        "flags": {
            key: bool(state.get(key))
            for key in [
                "topic_routed",
                "memory_recalled",
                "fast_understood",
                "data_discovered",
                "planning_assets_compacted",
                "query_graph_reflected",
                "sql_generated",
                "evidence_graph_verified",
                "evidence_accepted",
                "human_clarification_required",
                "chat_bi_completed",
            ]
        },
        "tasks": [
            {
                "taskId": getattr(item, "task_id", ""),
                "success": bool(getattr(item, "success", False)),
                "failed": bool(getattr(getattr(item, "query_bundle", None), "failed", False)),
                "rowCount": int(getattr(getattr(item, "query_bundle", None), "original_row_count", 0) or len(getattr(getattr(item, "query_bundle", None), "rows", []) or [])),
                "contract": getattr(item, "execution_contract_hash", ""),
            }
            for item in task_results
        ],
        "outputs": {
            "answer": stable_json_hash(state.get("answer")),
            "analysis": stable_json_hash(state.get("analysis_summary")),
            "clarification": stable_json_hash(
                {
                    "question": state.get("human_clarification_question"),
                    "options": state.get("human_clarification_options"),
                }
            ),
        },
    }
    return stable_json_hash(payload)


def format_clarification_message(question: str, options: List[str]) -> str:
    if not options:
        return question
    lines = [question, "", "可选项："]
    for index, option in enumerate(options, start=1):
        lines.append("%d. %s" % (index, option))
    return "\n".join(lines)


def ensure_tool_ledger_entry(state: AgentState, result: ToolCallExecutionResult, stage: str) -> None:
    key = "%s:%s" % (result.id, result.status)
    existing = {
        "%s:%s" % (entry.tool_call_id, entry.status)
        for entry in state.get("tool_call_ledger", [])
        if hasattr(entry, "tool_call_id")
    }
    if key in existing:
        return
    state.setdefault("tool_call_ledger", []).append(
        ToolCallLedgerEntry(
            tool_call_id=result.id,
            tool_name=result.name,
            stage=stage,
            status=result.status,
            duration_ms=result.duration_ms,
            attempts=result.attempts,
            cache_hit=result.cache_hit,
            rate_limited=result.rate_limited,
            target=result.target,
            error_type=result.error_type,
            error_message=result.error_message,
        )
    )
    state["tool_call_ledger"] = state["tool_call_ledger"][-200:]


def stable_json_hash(value: Any) -> str:
    try:
        raw = json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        raw = str(value or {})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def loop_guard_key_args(args: Dict[str, Any]) -> Dict[str, Any]:
    key_fields = [
        "path",
        "url",
        "query",
        "command",
        "cmd",
        "pattern",
        "prompt",
        "description",
        "refId",
        "artifactId",
        "sql",
        "taskId",
    ]
    normalized: Dict[str, Any] = {}
    for field in key_fields:
        if field in args:
            normalized[field] = args.get(field)
    if normalized:
        return normalized
    return dict(args or {})


def sanitize_artifact_name(value: str) -> str:
    text = safe_ascii_component(
        str(value or "artifact").strip(),
        extras=("_", ".", "-"),
        default="artifact",
        strip="._",
    )
    return text.strip("._") or "artifact"


def classify_finish_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if "forced stop" in lower:
        return "forced_stop"
    if "circuit_open" in lower or "circuit open" in lower:
        return "circuit_open"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "empty_response" in lower or "empty response" in lower:
        return "empty_response"
    if "json_parse_error" in lower or "parse" in lower:
        return "parse_error"
    if "provider_error" in lower or "provider" in lower:
        return "provider_error"
    if "length" in lower or "truncated" in lower:
        return "length_or_truncated"
    return ""


def build_run_budget_report(state: AgentState, settings: Settings) -> Dict[str, Any]:
    now = int(time.time() * 1000)
    started = normalize_run_started_at_ms(state.get("run_started_at_ms"), now)
    elapsed_ms = max(0, now - started)
    spans = state.get("trace_spans") or []
    llm_calls = sum(1 for span in spans if span_kind(span) == "llm")
    doris_queries = sum(1 for span in spans if span_kind(span) == "sql" or "doris" in span_name(span))
    tool_events = state.get("tool_runtime_events") or []
    tool_calls = sum(1 for event in tool_events if str(event.get("eventType") or "") == "tool.started")
    token_reports = [item for item in (state.get("token_usage_reports") or []) if isinstance(item, dict)]
    current_estimated_tokens = int((token_reports[-1] or {}).get("estimatedInputTokens") or 0) if token_reports else 0
    peak_estimated_tokens = max([current_estimated_tokens] + [int(item.get("estimatedInputTokens") or 0) for item in token_reports])
    usage = {
        "elapsedMs": elapsed_ms,
        "actions": len(state.get("action_history") or []),
        "llmCalls": llm_calls,
        "dorisQueries": doris_queries,
        "toolCalls": tool_calls,
        "estimatedTokens": current_estimated_tokens,
        "peakEstimatedTokens": peak_estimated_tokens,
    }
    latency_policy = state.get("latency_optimization") or {}
    fast_budget = bool(latency_policy.get("eligible")) and str(latency_policy.get("mode") or "").startswith("fast_path")
    duration_limit = (
        max(1, int(getattr(settings, "run_budget_fast_duration_seconds", 25) or 25))
        if fast_budget
        else max(1, int(getattr(settings, "run_budget_max_duration_seconds", 90) or 90))
    )
    limits = {
        "maxDurationSeconds": duration_limit,
        "maxActions": max(1, int(getattr(settings, "run_budget_max_actions", 20) or 20)),
        "maxLlmCalls": max(1, int(getattr(settings, "run_budget_max_llm_calls", 16) or 16)),
        "maxDorisQueries": max(1, int(getattr(settings, "run_budget_max_doris_queries", 12) or 12)),
        "maxToolCalls": max(1, int(getattr(settings, "run_budget_max_tool_calls", 60) or 60)),
        "maxEstimatedTokens": max(1, int(getattr(settings, "run_budget_max_estimated_tokens", 60000) or 60000)),
        "profile": "fast" if fast_budget else "complex",
    }
    breaches: List[str] = []
    if usage["elapsedMs"] >= limits["maxDurationSeconds"] * 1000:
        breaches.append("duration")
    if usage["actions"] >= limits["maxActions"]:
        breaches.append("actions")
    if usage["llmCalls"] >= limits["maxLlmCalls"]:
        breaches.append("llm_calls")
    if usage["dorisQueries"] >= limits["maxDorisQueries"]:
        breaches.append("doris_queries")
    if usage["toolCalls"] >= limits["maxToolCalls"]:
        breaches.append("tool_calls")
    if usage["peakEstimatedTokens"] >= limits["maxEstimatedTokens"]:
        breaches.append("estimated_tokens")
    return {
        "usage": usage,
        "limits": limits,
        "breaches": breaches,
        "exhausted": bool(breaches),
        "reason": "run budget exhausted: %s" % ", ".join(breaches) if breaches else "",
    }


def normalize_run_started_at_ms(value: Any, now_ms_value: int) -> int:
    try:
        started = int(float(value))
    except Exception:
        return now_ms_value
    if started <= 0:
        return now_ms_value
    monotonic_now = int(time.perf_counter() * 1000)
    if abs(started - monotonic_now) <= 31_536_000_000:
        # Compatibility for checkpoints produced before run_started_at_ms was
        # separated from the monotonic trace clock.  Preserve elapsed duration
        # while translating the value into the epoch domain used by budgets.
        return max(1, now_ms_value - max(0, monotonic_now - started))
    if started < 1_000_000_000:
        return now_ms_value
    if started < 10_000_000_000:
        return started * 1000
    return started


def span_kind(span: Any) -> str:
    if isinstance(span, dict):
        return str(span.get("kind") or "")
    return str(getattr(span, "kind", "") or "")


def span_name(span: Any) -> str:
    if isinstance(span, dict):
        return str(span.get("name") or "")
    return str(getattr(span, "name", "") or "")


def estimate_context_tokens(state: AgentState) -> int:
    chars = 0
    for key in [
        "question",
        "merchant_profile_context",
        "memory_context",
        "runtime_context",
        "base_knowledge_context",
        "topic_asset_context",
        "recall_context",
        "session_context",
        "summary_context",
        "tool_context",
    ]:
        chars += len(str(state.get(key) or ""))
    for key in ["route_slots", "intent_signals", "fast_understanding"]:
        value = state.get(key)
        if value is not None:
            chars += len(safe_json(value))
    plan = state.get("plan")
    if plan is not None:
        chars += len(safe_json(getattr(plan, "question_understanding", {}) or {}))
        chars += min(8000, len(safe_json(getattr(plan, "compiler_trace", []) or [])))
    snapshots = state.get("context_snapshots") or []
    chars += sum(min(4000, len(safe_json(item))) for item in snapshots[-4:])
    chars += sum(min(1200, len(safe_json(item))) for item in state.get("context_assembly_reports", [])[-4:])
    return estimate_text_tokens("x" * chars)


def snapshot_protected_fact_count(snapshot: Any) -> int:
    if isinstance(snapshot, dict):
        return len(snapshot.get("protectedFacts") or snapshot.get("protected_facts") or [])
    return len(getattr(snapshot, "protected_facts", []) or [])


def context_report_over_budget(report: Any) -> bool:
    if isinstance(report, dict):
        return bool(report.get("overBudget") or report.get("over_budget"))
    return bool(getattr(report, "over_budget", False))


def context_report_stage(report: Any) -> str:
    if isinstance(report, dict):
        return str(report.get("stage") or "policy")
    return str(getattr(report, "stage", "policy") or "policy")


def context_report_estimated_tokens(report: Any) -> int:
    if isinstance(report, dict):
        return int(report.get("estimatedTokens") or report.get("estimated_tokens") or 0)
    return int(getattr(report, "estimated_tokens", 0) or 0)


def estimate_text_tokens(text: str) -> int:
    value = str(text or "")
    if not value:
        return 1
    cjk_count = sum(
        1
        for character in value
        if "\u3400" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff"
    )
    non_cjk_count = max(0, len(value) - cjk_count)
    return max(1, cjk_count + int((non_cjk_count + 3) / 4))


def safe_json(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(by_alias=True)
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


def build_compression_summary(state: AgentState, max_chars: int) -> str:
    lines: List[str] = []
    lines.append("# Compressed Runtime Context")
    question = str(state.get("question") or "")
    if question:
        lines.append("\n## Current Question\n%s" % question[:1200])
    facts = []
    for snapshot in state.get("context_snapshots", [])[-4:]:
        if isinstance(snapshot, dict):
            facts.extend(snapshot.get("protectedFacts") or [])
    if facts:
        lines.append("\n## Protected Facts")
        for fact in facts[:24]:
            key = fact.get("key") if isinstance(fact, dict) else getattr(fact, "key", "")
            value = fact.get("value") if isinstance(fact, dict) else getattr(fact, "value", "")
            if key or value:
                lines.append("- %s=%s" % (key, str(value)[:500]))
    history = state.get("action_history") or []
    if history:
        lines.append("\n## Recent Actions")
        for item in history[-10:]:
            action = getattr(item, "action", "") if hasattr(item, "action") else item.get("action", "")
            status = getattr(item, "status", "") if hasattr(item, "status") else item.get("status", "")
            reason = getattr(item, "reason", "") if hasattr(item, "reason") else item.get("reason", "")
            lines.append("- %s %s %s" % (action, status, str(reason)[:240]))
    validation = state.get("query_graph_validation_result")
    gaps = getattr(validation, "gaps", []) if validation is not None else []
    if gaps:
        lines.append("\n## Current Gaps")
        for gap in gaps[:12]:
            lines.append("- %s %s" % (getattr(gap, "code", ""), str(getattr(gap, "message", ""))[:240]))
    return "\n".join(lines)[: max(1000, max_chars)]


def compact_runtime_context_fields(state: AgentState, target_chars: int) -> None:
    total_budget = max(1800, int(target_chars or 0))
    summary_budget = max(800, total_budget // 2)
    state["summary_context"] = str(state.get("summary_context") or "")[:summary_budget]
    remaining = max(1000, total_budget - len(state["summary_context"]))
    fields = [
        "session_context",
        "memory_context",
        "base_knowledge_context",
        "topic_asset_context",
        "recall_context",
        "tool_context",
    ]
    per_field = max(160, remaining // len(fields))
    for key in fields:
        value = str(state.get(key) or "")
        if len(value) <= per_field:
            continue
        head = max(80, per_field // 2)
        tail = max(80, per_field - head - 20)
        state[key] = "%s\n...[compacted]...\n%s" % (value[:head], value[-tail:])
    state["runtime_context"] = ""
    state["context_snapshots"] = list(state.get("context_snapshots") or [])[-2:]
    state["context_assembly_reports"] = list(state.get("context_assembly_reports") or [])[-2:]


def build_runtime_checkpoint_payload(state: AgentState, stage: str) -> Dict[str, Any]:
    validation = state.get("query_graph_validation_result")
    run_result = state.get("agent_run_result")
    plan = state.get("plan")
    manifest = state.get("workspace_manifest")
    return {
        "version": "runtime_checkpoint.v1",
        "stage": stage,
        "createdAt": datetime.now().isoformat(),
        "question": state.get("question", ""),
        "runId": state.get("run_id", ""),
        "threadId": state.get("thread_id", ""),
        "queryGraph": plan.model_dump(by_alias=True) if hasattr(plan, "model_dump") else {},
        "validation": validation.model_dump(by_alias=True) if hasattr(validation, "model_dump") else {},
        "evidenceGaps": [
            gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap
            for gap in getattr(run_result, "evidence_gaps", [])[:24]
        ]
        if run_result is not None
        else [],
        "executionAttemptArtifacts": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("execution_attempt_artifacts", [])[-12:]
        ],
        "memoryConstraints": state.get("memory_constraints", []),
        "memoryConstraintTrace": state.get("memory_constraint_trace", {}),
        "runtimeInjection": state.get("runtime_injection", {}),
        "threadContext": state.get("thread_context", {}),
        "artifactManifest": manifest.model_dump(by_alias=True) if hasattr(manifest, "model_dump") else manifest,
        "recentActions": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("action_history", [])[-12:]
        ],
        "recentSteps": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("run_steps", [])[-12:]
        ],
    }


def protected_fact_keys(state: AgentState) -> List[str]:
    keys: List[str] = []
    for snapshot in state.get("context_snapshots", [])[-4:]:
        if not isinstance(snapshot, dict):
            continue
        for fact in snapshot.get("protectedFacts") or []:
            key = str(fact.get("key") or "")
            if key and key not in keys:
                keys.append(key)
    return keys[:40]


def write_middleware_text_artifact(state: AgentState, settings: Settings, namespace: str, name: str, content: str) -> ArtifactRef:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return ArtifactRef()
    store = WorkspaceArtifactStore(settings, Path(outputs_path) / "artifacts")
    artifact = store.write_text(namespace, name, content)
    ref = artifact_ref_from_path(artifact.get("path", ""), namespace=namespace, reason="middleware context artifact")
    ref.merchant_uri = artifact.get("merchantUri", ref.merchant_uri)
    ref.relative_path = artifact.get("relativePath", ref.relative_path)
    return ref


def write_middleware_json_artifact(state: AgentState, settings: Settings, namespace: str, name: str, payload: Any) -> ArtifactRef:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return ArtifactRef()
    store = WorkspaceArtifactStore(settings, Path(outputs_path) / "artifacts")
    artifact = store.write_json(namespace, name, payload, preview_chars=0)
    ref = artifact_ref_from_path(artifact.get("path", ""), namespace=namespace, reason="middleware offloaded tool result")
    ref.merchant_uri = artifact.get("merchantUri", ref.merchant_uri)
    ref.relative_path = artifact.get("relativePath", ref.relative_path)
    return ref


def workspace_entries(state: AgentState) -> List[Path]:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return []
    root = Path(outputs_path)
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def build_workspace_manifest(state: AgentState, settings: Settings) -> WorkspaceManifest:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    root = Path(outputs_path) if outputs_path else Path()
    entries: List[WorkspaceManifestEntry] = []
    if root and root.exists():
        for path in workspace_entries(state)[: max(1, int(settings.middleware_max_manifest_entries or 200))]:
            if path.name == "workspace_manifest.json":
                continue
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            entries.append(
                WorkspaceManifestEntry(
                    path=str(path),
                    relative_path=relative,
                    namespace=relative.split("/", 1)[0] if "/" in relative else "root",
                    bytes=path.stat().st_size,
                    estimated_chars=path.stat().st_size,
                    sha256=file_sha256(path),
                    merchant_uri=merchant_uri_for_artifact(relative),
                    updated_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                )
            )
    return WorkspaceManifest(
        root=str(root) if outputs_path else "",
        entries=entries,
        entry_count=len(entries),
        total_bytes=sum(item.bytes for item in entries),
        updated_at=datetime.now().isoformat(),
    )


def file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except Exception:
        return ""


def normalize_tool_result(item: Any) -> ToolCallExecutionResult:
    if isinstance(item, ToolCallExecutionResult):
        return item
    if isinstance(item, dict):
        return ToolCallExecutionResult.model_validate(item)
    return ToolCallExecutionResult(id=str(getattr(item, "id", "")), name=str(getattr(item, "name", "")), status=str(getattr(item, "status", "")))


def synthetic_missing_tool_results(state: AgentState) -> List[ToolCallExecutionResult]:
    calls = collect_tool_call_requests(state)
    if not calls:
        return []
    existing_ids = {
        normalize_tool_result(item).id
        for item in state.get("tool_call_results", []) or []
        if normalize_tool_result(item).id
    }
    synthetic: List[ToolCallExecutionResult] = []
    for call in calls:
        call_id = str(call.get("id") or "").strip()
        name = str(call.get("name") or "").strip()
        if not call_id or call_id in existing_ids:
            continue
        synthetic.append(
            ToolCallExecutionResult(
                id=call_id,
                name=name,
                status="failed",
                error_type="MISSING_TOOL_RESULT",
                error_message="tool call had no matching tool result; synthetic failure inserted by recovery middleware",
                retryable=False,
                recommended_action="patch_missing_terminal_result",
                tool_message={
                    "toolName": name,
                    "status": "failed",
                    "errorCode": "MISSING_TOOL_RESULT",
                    "message": "synthetic missing tool result",
                },
            )
        )
        state.setdefault("tool_call_recovery_events", []).append(
            ToolCallRecoveryEvent(
                tool_call_id=call_id,
                tool_name=name,
                stage="before_policy",
                action="patch_missing_tool_result",
                reason="tool call did not have a matching tool result",
                status_before="missing",
                status_after="failed",
            )
        )
        append_middleware_event(
            state,
            "tool_call_recovery",
            "before_policy",
            status="patched",
            code="MISSING_TOOL_RESULT_PATCHED",
            message="inserted synthetic failed tool result for dangling tool call",
            metadata={"toolCallId": call_id, "toolName": name},
        )
        existing_ids.add(call_id)
    return synthetic


def collect_tool_call_requests(state: AgentState) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in state.get("tool_call_requests", []) or []:
        normalized = normalize_tool_call_request(item)
        if normalized:
            calls.append(normalized)
    plan = state.get("plan")
    for item in getattr(plan, "planner_tool_calls", []) or []:
        normalized = normalize_tool_call_request(item)
        if normalized:
            calls.append(normalized)
    answer_tools = state.get("answer_file_tool_results") or {}
    if isinstance(answer_tools, dict):
        for item in answer_tools.get("calls") or []:
            normalized = normalize_tool_call_request(item)
            if normalized:
                calls.append(normalized)
    return calls


def normalize_tool_call_request(item: Any) -> Dict[str, Any]:
    if isinstance(item, ToolCallRequest):
        return {"id": item.id, "name": item.name, "args": dict(item.args or {})}
    if isinstance(item, dict):
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        return {"id": str(item.get("id") or ""), "name": str(item.get("name") or ""), "args": args}
    args = getattr(item, "args", {})
    return {"id": str(getattr(item, "id", "")), "name": str(getattr(item, "name", "")), "args": args if isinstance(args, dict) else {}}
