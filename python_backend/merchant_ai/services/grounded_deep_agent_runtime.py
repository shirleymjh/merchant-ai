from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from pydantic import Field as PydanticField

try:
    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from deepagents.backends.protocol import (
        EditResult,
        FileInfo,
        GlobResult,
        GrepMatch,
        GrepResult,
        LsResult,
        ReadResult,
        WriteResult,
    )

    _DEEPAGENTS_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised through constructor gate
    create_deep_agent = None
    _DEEPAGENTS_IMPORT_ERROR = str(exc)

    @dataclass
    class FilesystemPermission:  # type: ignore[no-redef]
        operations: list[str]
        paths: list[str]
        mode: str

    @dataclass
    class FileInfo:  # type: ignore[no-redef]
        path: str
        is_dir: bool = False
        size: int = 0
        modified_at: str = ""

    @dataclass
    class LsResult:  # type: ignore[no-redef]
        entries: list[FileInfo] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class ReadResult:  # type: ignore[no-redef]
        file_data: Optional[dict[str, Any]] = None
        error: Optional[str] = None

    @dataclass
    class GrepMatch:  # type: ignore[no-redef]
        path: str
        line: int
        text: str

    @dataclass
    class GrepResult:  # type: ignore[no-redef]
        matches: list[GrepMatch] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class GlobResult:  # type: ignore[no-redef]
        matches: list[Any] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class WriteResult:  # type: ignore[no-redef]
        error: Optional[str] = None
        path: str = ""

    @dataclass
    class EditResult:  # type: ignore[no-redef]
        error: Optional[str] = None
        path: str = ""

    class StateBackend:  # type: ignore[no-redef]
        pass

    class FilesystemBackend:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class CompositeBackend:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
try:
    from langchain.tools import ToolRuntime, tool
except ImportError:  # pragma: no cover - compatibility for minimal test runtime
    from langchain_core.tools import tool

    _ContextT = TypeVar("_ContextT")

    class ToolRuntime(Generic[_ContextT]):  # type: ignore[no-redef]
        context: _ContextT

from merchant_ai.models import (
    APIModel,
    ChatResponse,
    MerchantInfo,
    RecallBundle,
    SkillLifecycleRecord,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier
from merchant_ai.services.assets import normalize_semantic_path
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeEvent,
    GroundedRuntimeSession,
)
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
)
from merchant_ai.services.grounded_goal_contract import (
    GoalCoverageBlocked,
    GoalCoverageVerifier,
    OriginalQuestionGoalContract,
    VerifiedArtifactGoalCoverage,
    canonical_goal_id,
    declare_verified_artifact_goal_coverage,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
    required_goal_ids,
)
from merchant_ai.services.time_semantics import has_explicit_time_expression
from merchant_ai.services.grounded_query_contract import GroundedBindingHints
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.grounded_subagent_runtime import (
    IsolatedSubagentJob,
    IsolatedSubagentRuntime,
)


_SEMANTIC_SCOPE: ContextVar[Optional["GroundedDeepAgentSession"]] = ContextVar(
    "grounded_deep_agent_semantic_scope",
    default=None,
)


class GroundedParallelQuerySpec(APIModel):
    """One independent query branch prepared from already-read semantics."""

    query_id: str
    read_ref_ids: list[str] = PydanticField(default_factory=list)
    binding_hints: GroundedBindingHints = PydanticField(
        default_factory=GroundedBindingHints
    )
    goal_ids: list[str] = PydanticField(default_factory=list)


class GroundedParallelExecutionSpec(APIModel):
    """Execution input for a previously prepared independent branch."""

    query_id: str
    sql: str = ""
    rationale: str = ""
    evidence_ref_ids: list[str] = PydanticField(default_factory=list)


def _validated_goal_assignment(
    session: "GroundedDeepAgentSession",
    goal_ids: list[str],
) -> tuple[list[str], dict[str, Any]]:
    contract = session.question_goal_contract
    if contract is None:
        return [], {
            "status": "BLOCKED",
            "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED",
            "nextAction": "DECLARE_ORIGINAL_QUESTION_GOALS",
        }
    try:
        normalized = list(
            dict.fromkeys(canonical_goal_id(item) for item in goal_ids)
        )
    except ValueError as exc:
        return [], {
            "status": "REJECTED",
            "code": "QUERY_GOAL_ID_INVALID",
            "message": str(exc),
        }
    if not normalized:
        return [], {
            "status": "REJECTED",
            "code": "QUERY_GOAL_ASSIGNMENT_REQUIRED",
        }
    known = set(contract.goal_map())
    unknown = [item for item in normalized if item not in known]
    if unknown:
        return [], {
            "status": "REJECTED",
            "code": "QUERY_GOAL_ID_UNKNOWN",
            "unknownGoalIds": unknown,
        }
    return normalized, {}


def _parallel_goal_dependency_issues(
    contract: OriginalQuestionGoalContract,
    goal_ids_by_query_id: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Find declared goal paths that make one query batch non-independent.

    This is a batch-safety check, not an execution planner.  It only rejects
    dependency paths whose two endpoints are assigned somewhere in the same
    requested batch; the Core remains responsible for choosing the serial
    query sequence.
    """

    query_ids_by_goal_id: dict[str, list[str]] = {}
    for query_id, goal_ids in goal_ids_by_query_id.items():
        for goal_id in goal_ids:
            query_ids_by_goal_id.setdefault(goal_id, []).append(query_id)

    for query_ids in query_ids_by_goal_id.values():
        query_ids.sort()

    edges_by_upstream_goal_id: dict[str, list[dict[str, Any]]] = {}

    def add_edge(
        *,
        upstream_goal_id: str,
        downstream_goal_id: str,
        relation_type: str,
        declared_by_goal_id: str,
        dependency_goal_id: str = "",
    ) -> None:
        edge: dict[str, Any] = {
            "relationType": relation_type,
            "upstreamGoalId": upstream_goal_id,
            "downstreamGoalId": downstream_goal_id,
            "declaredByGoalId": declared_by_goal_id,
        }
        if dependency_goal_id:
            edge["dependencyGoalId"] = dependency_goal_id
        existing = edges_by_upstream_goal_id.setdefault(upstream_goal_id, [])
        if edge not in existing:
            existing.append(edge)

    for goal in contract.goals:
        for upstream_goal_id in goal.depends_on_goal_ids:
            add_edge(
                upstream_goal_id=upstream_goal_id,
                downstream_goal_id=goal.goal_id,
                relation_type="DEPENDS_ON_GOAL_IDS",
                declared_by_goal_id=goal.goal_id,
            )
        if str(getattr(goal, "kind", "")) != "DEPENDENCY":
            continue
        for upstream_goal_id in getattr(goal, "upstream_goal_ids", ()):
            for downstream_goal_id in getattr(goal, "downstream_goal_ids", ()):
                add_edge(
                    upstream_goal_id=upstream_goal_id,
                    downstream_goal_id=downstream_goal_id,
                    relation_type="DEPENDENCY_GOAL",
                    declared_by_goal_id=goal.goal_id,
                    dependency_goal_id=goal.goal_id,
                )

    def dependency_path(
        upstream_goal_id: str,
        downstream_goal_id: str,
    ) -> list[dict[str, Any]]:
        pending: list[tuple[str, list[dict[str, Any]]]] = [
            (upstream_goal_id, [])
        ]
        visited = {upstream_goal_id}
        cursor = 0
        while cursor < len(pending):
            current_goal_id, current_path = pending[cursor]
            cursor += 1
            for edge in edges_by_upstream_goal_id.get(current_goal_id, []):
                next_goal_id = str(edge["downstreamGoalId"])
                next_path = [*current_path, edge]
                if next_goal_id == downstream_goal_id:
                    return next_path
                if next_goal_id in visited:
                    continue
                visited.add(next_goal_id)
                pending.append((next_goal_id, next_path))
        return []

    goal_order = {
        goal.goal_id: index
        for index, goal in enumerate(contract.goals)
    }
    assigned_goal_ids = sorted(
        query_ids_by_goal_id,
        key=lambda goal_id: (goal_order.get(goal_id, len(goal_order)), goal_id),
    )
    issues: list[dict[str, Any]] = []
    for upstream_goal_id in assigned_goal_ids:
        for downstream_goal_id in assigned_goal_ids:
            if upstream_goal_id == downstream_goal_id:
                continue
            path_edges = dependency_path(upstream_goal_id, downstream_goal_id)
            if not path_edges:
                continue
            direct = len(path_edges) == 1
            path_goal_ids = [upstream_goal_id]
            path_goal_ids.extend(
                str(edge["downstreamGoalId"])
                for edge in path_edges
            )
            first_edge = path_edges[0]
            if direct and first_edge["relationType"] == "DEPENDS_ON_GOAL_IDS":
                code = "BATCH_GOAL_DEPENDS_ON_EDGE"
            elif direct and first_edge["relationType"] == "DEPENDENCY_GOAL":
                code = "BATCH_DEPENDENCY_GOAL_EDGE"
            else:
                code = "BATCH_TRANSITIVE_GOAL_DEPENDENCY_PATH"
            issue: dict[str, Any] = {
                "code": code,
                "upstreamGoalId": upstream_goal_id,
                "downstreamGoalId": downstream_goal_id,
                "upstreamQueryIds": list(
                    query_ids_by_goal_id[upstream_goal_id]
                ),
                "downstreamQueryIds": list(
                    query_ids_by_goal_id[downstream_goal_id]
                ),
                "requiredExecution": "SERIAL",
                "direct": direct,
                "pathGoalIds": path_goal_ids,
                "pathEdges": [dict(edge) for edge in path_edges],
            }
            dependency_goal_ids = list(
                dict.fromkeys(
                    str(edge.get("dependencyGoalId") or "")
                    for edge in path_edges
                    if str(edge.get("dependencyGoalId") or "")
                )
            )
            if dependency_goal_ids:
                issue["dependencyGoalIds"] = dependency_goal_ids
            if direct and dependency_goal_ids:
                dependency_goal_id = dependency_goal_ids[0]
                issue["dependencyGoalId"] = dependency_goal_id
                issue["dependencyGoalQueryIds"] = list(
                    query_ids_by_goal_id.get(dependency_goal_id, [])
                )
            issues.append(issue)
    return issues


def _goal_assignment_contract_issues(
    session: "GroundedDeepAgentSession",
    goal_ids: list[str],
    contract: Any,
) -> list[dict[str, Any]]:
    goal_contract = session.question_goal_contract
    if goal_contract is None:
        return [{"code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED"}]
    selected_refs = set(getattr(contract, "evidence_refs", ()) or ())
    issues: list[dict[str, Any]] = []
    for goal_id in goal_ids:
        goal = goal_contract.goal_map().get(goal_id)
        if goal is None:
            continue
        declared_refs = set(getattr(goal, "semantic_ref_ids", ()) or ())
        for attribute in ("metric_ref_id", "dimension_ref_id", "entity_ref_id"):
            value = str(getattr(goal, attribute, "") or "").strip()
            if value:
                declared_refs.add(value)
        if declared_refs and not declared_refs.intersection(selected_refs):
            issues.append(
                {
                    "code": "QUERY_GOAL_SEMANTIC_REF_MISMATCH",
                    "goalId": goal_id,
                    "declaredSemanticRefIds": sorted(declared_refs),
                    "contractEvidenceRefs": sorted(selected_refs),
                }
            )
        if str(getattr(goal, "kind", "")) == "TIME_WINDOW":
            time_range = getattr(contract, "time_range", None)
            if time_range is None or not bool(getattr(time_range, "explicit", False)):
                issues.append(
                    {
                        "code": "QUERY_GOAL_TIME_WINDOW_NOT_BOUND",
                        "goalId": goal_id,
                    }
                )
    return issues


@dataclass
class GroundedDeepAgentSession:
    runtime: GroundedRuntimeSession
    core_semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    opened_topics: list[str] = field(default_factory=list)
    topic_index_read: bool = False
    expanded_from_attempt_ids: list[str] = field(default_factory=list)
    skill_runs: list[dict[str, Any]] = field(default_factory=list)
    analysis_skill_headers_disclosed: bool = False
    data_collection_sealed: bool = False
    analysis_skill_started: bool = False
    parallel_branches: dict[str, GroundedRuntimeSession] = field(default_factory=dict)
    parallel_branch_goal_ids: dict[str, list[str]] = field(default_factory=dict)
    artifact_goal_ids: dict[str, list[str]] = field(default_factory=dict)
    active_goal_ids: list[str] = field(default_factory=list)
    question_goal_contract: Optional[OriginalQuestionGoalContract] = None
    goal_coverage_result: dict[str, Any] = field(default_factory=dict)
    operational_failure: dict[str, Any] = field(default_factory=dict)
    runtime_budget_report: dict[str, Any] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)

    def effective_topics(self) -> list[str]:
        return list(
            dict.fromkeys(
                [*self.runtime.workspace_topics, *self.opened_topics]
            )
        )

    def can_expand_topic(self) -> bool:
        if len(self.opened_topics) >= 2 or not self.runtime.attempts:
            return False
        latest_attempt = self.runtime.attempts[-1]
        if latest_attempt.attempt_id in set(self.expanded_from_attempt_ids):
            return False
        latest = latest_attempt.contract
        if latest.status != "REVISE_BINDINGS":
            return False
        structured_gaps = [
            gap
            for gap in latest.unresolved_gaps
            if gap.blocking
            and gap.required_capability
            and (
                "TOPIC_INDEX" in str(gap.search_scope or "").upper()
                or bool(gap.required_capability.get("allowTopicExpansion"))
            )
        ]
        if not structured_gaps:
            return False
        evidence_refs = {
            str(item.get("refId") or "")
            for item in self.core_semantic_evidence
            if str(item.get("refId") or "")
        }
        evidence_topics = {
            str(item.get("topic") or "")
            for item in self.core_semantic_evidence
            if str(item.get("topic") or "")
        }
        for gap in structured_gaps:
            if evidence_refs.intersection(gap.rejected_ref_ids):
                return True
            capability_refs = {
                str(value)
                for value in _nested_values(gap.required_capability)
                if str(value).startswith("semantic:")
            }
            if evidence_refs.intersection(capability_refs):
                return True
            if gap.topic and gap.topic in evidence_topics:
                return True
        return False

    def mark_topic_expanded(self) -> None:
        if not self.runtime.attempts:
            return
        attempt_id = self.runtime.attempts[-1].attempt_id
        if attempt_id not in self.expanded_from_attempt_ids:
            self.expanded_from_attempt_ids.append(attempt_id)


@dataclass(frozen=True)
class GroundedDeepAgentRunContext:
    thread_id: str
    run_id: str
    session: GroundedDeepAgentSession
    budget: Optional[GroundedRuntimeBudget] = None
    listener: Optional[Callable[[str, str, dict[str, Any]], None]] = None


class GroundedSemanticBackend:
    """Read-only semantic filesystem scoped to one Grounded Core session."""

    MAX_EVIDENCE_ITEMS = 64

    def __init__(
        self,
        semantic_catalog: Any,
        *,
        reader_is_core: Optional[Callable[[], bool]] = None,
    ):
        self.semantic_catalog = semantic_catalog
        self.reader_is_core = reader_is_core or _deepagent_reader_is_core
        self._read_receipts: dict[
            tuple[int, str, int, int],
            dict[str, Any],
        ] = {}
        self._receipt_lock = RLock()

    @contextmanager
    def scope(self, session: GroundedDeepAgentSession) -> Iterator[None]:
        token = _SEMANTIC_SCOPE.set(session)
        try:
            yield
        finally:
            _SEMANTIC_SCOPE.reset(token)

    @staticmethod
    def _session() -> Optional[GroundedDeepAgentSession]:
        return _SEMANTIC_SCOPE.get()

    @staticmethod
    def _path(value: str) -> str:
        return normalize_semantic_path(str(value or "/")).strip("/")

    @staticmethod
    def _topic(path: str) -> str:
        parts = [item for item in str(path or "").split("/") if item]
        return parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""

    @staticmethod
    def _manifest_topic(path: str) -> str:
        parts = [item for item in str(path or "").split("/") if item]
        if len(parts) == 3 and parts[0] == "topics" and parts[2] == "manifest.json":
            return parts[1]
        return ""

    def _scope_error(self, path: str) -> str:
        session = self._session()
        if session is None:
            return "GROUNDED_SESSION_REQUIRED"
        normalized = self._path(path)
        if normalized.endswith("/asset.json"):
            return (
                "FULL_TABLE_ASSET_DENIED: read the table detail.json, then only "
                "the required metrics/columns/schema/relationships children"
            )
        if normalized in {"", "topics", "topics/index.json"}:
            return ""
        manifest_topic = self._manifest_topic(normalized)
        if manifest_topic:
            if manifest_topic in session.effective_topics():
                return ""
            if session.topic_index_read and session.can_expand_topic():
                return ""
            if session.topic_index_read:
                return "TOPIC_EXPANSION_REQUIRES_STRUCTURED_GAP"
            return "TOPIC_INDEX_READ_REQUIRED"
        topic = self._topic(normalized)
        if topic and topic not in session.effective_topics():
            return "TOPIC_SCOPE_DENIED:%s" % topic
        return ""

    def ls(self, path: str) -> LsResult:
        normalized = self._path(path)
        scope_error = self._scope_error(normalized)
        if scope_error:
            return LsResult(error=scope_error)
        session = self._session()
        if normalized in {"", "topics"}:
            topics = session.effective_topics() if session else []
            return LsResult(
                entries=[
                    FileInfo(path="/topics/index.json", is_dir=False, size=0, modified_at=""),
                    *[
                        FileInfo(
                            path="/topics/%s/" % topic_name,
                            is_dir=True,
                            size=0,
                            modified_at="",
                        )
                        for topic_name in topics
                    ],
                ]
            )
        try:
            items = self.semantic_catalog.ls(path=normalized, limit=500)
        except Exception as exc:
            return LsResult(error="SEMANTIC_LS_FAILED:%s" % str(exc)[:300])
        return LsResult(
            entries=[
                FileInfo(
                    path="/" + str(item.get("path") or "").lstrip("/"),
                    is_dir=False,
                    size=int(item.get("estimatedChars") or 0),
                    modified_at="",
                )
                for item in items
                if isinstance(item, dict)
                and item.get("path")
                and not str(item.get("path") or "").endswith("/asset.json")
            ]
        )

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        normalized = self._path(file_path)
        scope_error = self._scope_error(normalized)
        if scope_error:
            return ReadResult(error=scope_error)
        try:
            result = self.semantic_catalog.read(
                path=normalized,
                max_chars=2_000_000,
                offset=0,
            )
        except Exception as exc:
            return ReadResult(error="SEMANTIC_READ_FAILED:%s" % str(exc)[:300])
        if not isinstance(result, dict) or not result.get("success"):
            return ReadResult(error=str((result or {}).get("error") or "SEMANTIC_REF_NOT_FOUND"))

        session = self._session()
        kind = str(result.get("kind") or "").upper()
        topic_name = str(result.get("topic") or "")
        if session is not None:
            if kind == "TOPIC_INDEX":
                session.topic_index_read = True
            elif kind == "TOPIC_MANIFEST" and topic_name not in session.effective_topics():
                if not session.topic_index_read:
                    return ReadResult(error="TOPIC_INDEX_READ_REQUIRED")
                if not session.can_expand_topic():
                    return ReadResult(error="TOPIC_EXPANSION_REQUIRES_STRUCTURED_GAP")
                session.opened_topics.append(topic_name)
                session.mark_topic_expanded()
                if topic_name not in session.runtime.workspace_topics:
                    session.runtime.workspace_topics.append(topic_name)

        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        content = "".join(lines[start:end])
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if (
            session is not None
            and complete
            and kind not in {"TOPIC_INDEX", "TOPIC_MANIFEST"}
        ):
            evidence = {
                "refId": str(result.get("refId") or ""),
                "path": str(result.get("path") or normalized).lstrip("/"),
                "kind": kind,
                "topic": topic_name,
                "table": str(result.get("table") or ""),
                "contentSnippet": full_content,
                "contentHash": hashlib.sha256(full_content.encode("utf-8")).hexdigest(),
                "contentComplete": True,
                "offset": 0,
            }
            if evidence["refId"] and topic_name in session.effective_topics():
                receipt_key = (
                    id(session),
                    normalized,
                    int(offset or 0),
                    int(limit or 2000),
                )
                with self._receipt_lock:
                    self._read_receipts[receipt_key] = dict(evidence)
                if self.reader_is_core():
                    self._retain_evidence(session, evidence)
        return ReadResult(file_data={"content": content, "encoding": "utf-8"})

    def record_core_read(
        self,
        session: GroundedDeepAgentSession,
        file_path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> bool:
        """Record one successful root-Core read without relying on thread config.

        Deep Agents may execute filesystem backends in a worker context where
        ``langgraph.config.get_config()`` is unavailable.  Root/subagent
        authority is therefore decided by main-agent tool middleware; this
        method only verifies that the exact semantic file was read completely.
        """

        normalized = self._path(file_path)
        if normalized == "knowledge":
            normalized = ""
        elif normalized.startswith("knowledge/"):
            normalized = normalized[len("knowledge/") :]
        receipt_key = (
            id(session),
            normalized,
            int(offset or 0),
            int(limit or 2000),
        )
        with self._receipt_lock:
            receipt = self._read_receipts.pop(receipt_key, None)
        if receipt is not None:
            self._retain_evidence(session, receipt)
            return True
        try:
            result = self.semantic_catalog.read(
                path=normalized,
                max_chars=2_000_000,
                offset=0,
            )
        except Exception:
            return False
        if not isinstance(result, dict) or not result.get("success"):
            return False
        kind = str(result.get("kind") or "").upper()
        topic_name = str(result.get("topic") or "")
        if kind in {"TOPIC_INDEX", "TOPIC_MANIFEST"}:
            return False
        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if not complete or topic_name not in session.effective_topics():
            return False
        evidence = {
            "refId": str(result.get("refId") or ""),
            "path": str(result.get("path") or normalized).lstrip("/"),
            "kind": kind,
            "topic": topic_name,
            "table": str(result.get("table") or ""),
            "contentSnippet": full_content,
            "contentHash": hashlib.sha256(full_content.encode("utf-8")).hexdigest(),
            "contentComplete": True,
            "offset": 0,
        }
        if not evidence["refId"]:
            return False
        self._retain_evidence(session, evidence)
        return True

    def _retain_evidence(
        self,
        session: GroundedDeepAgentSession,
        evidence: dict[str, Any],
    ) -> None:
        with session.lock:
            retained = [
                item
                for item in session.core_semantic_evidence
                if item.get("refId") != evidence.get("refId")
            ]
            retained.append(dict(evidence))
            session.core_semantic_evidence = retained[-self.MAX_EVIDENCE_ITEMS :]

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        normalized = self._path(path or "")
        scope_error = self._scope_error(normalized)
        if scope_error:
            return GrepResult(error=scope_error)
        session = self._session()
        topics = [self._topic(normalized)] if self._topic(normalized) else (
            session.effective_topics() if session else []
        )
        hits: list[dict[str, Any]] = []
        try:
            for topic_name in topics:
                hits.extend(
                    self.semantic_catalog.grep(
                        query=pattern,
                        topic=topic_name,
                        limit=100,
                        path=normalized if normalized not in {"", "topics"} else "",
                    )
                )
        except Exception as exc:
            return GrepResult(error="SEMANTIC_GREP_FAILED:%s" % str(exc)[:300])
        matches: list[GrepMatch] = []
        for hit in hits:
            hit_path = "/" + str(hit.get("path") or "").lstrip("/")
            if hit_path.endswith("/asset.json"):
                continue
            if glob and not fnmatch.fnmatch(hit_path, glob):
                continue
            snippets = hit.get("snippets") or [hit.get("summary") or hit.get("title") or ""]
            matches.extend(
                GrepMatch(path=hit_path, line=1, text=str(snippet)[:1000])
                for snippet in snippets[:3]
            )
        return GrepResult(matches=matches[:100])

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        listing = self.ls(path or "/")
        if listing.error:
            return GlobResult(error=listing.error)
        return GlobResult(
            matches=[
                item
                for item in listing.entries or []
                if fnmatch.fnmatch(
                    str(item.get("path") if isinstance(item, dict) else item.path),
                    pattern,
                )
            ]
        )

    @staticmethod
    def write(file_path: str, content: str) -> WriteResult:
        return WriteResult(error="READ_ONLY_SEMANTIC_FILESYSTEM", path=file_path)

    @staticmethod
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error="READ_ONLY_SEMANTIC_FILESYSTEM", path=file_path)

    def ls_info(self, path: str) -> list[FileInfo]:
        result = self.ls(path)
        return [] if result.error else list(result.entries)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        result = self.glob(pattern, path)
        return [] if result.error else list(result.matches or [])

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> Any:
        result = self.grep(pattern, path, glob)
        return result.error or list(result.matches)

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def agrep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        return self.grep(pattern, path, glob)

    async def aglob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        return self.glob(pattern, path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def als_info(self, path: str) -> list[FileInfo]:
        return self.ls_info(path)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> Any:
        return self.grep_raw(pattern, path, glob)


class GroundedCoreToolBoundaryMiddleware(AgentMiddleware):
    """Bind semantic-read authority to the root Core's actual tool calls."""

    name = "GroundedCoreToolBoundaryMiddleware"

    def __init__(self, semantic_backend: GroundedSemanticBackend):
        self.semantic_backend = semantic_backend

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call = dict(getattr(request, "tool_call", None) or {})
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        if tool_name == "task":
            return ToolMessage(
                content=(
                    "SubAgent dispatch is disabled in this grounded runtime until "
                    "worker evidence acceptance is independently auditable."
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        result = handler(request)
        if tool_name != "read_file" or getattr(result, "status", "success") == "error":
            return result
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        session = getattr(context, "session", None)
        if not isinstance(session, GroundedDeepAgentSession):
            return result
        args = dict(tool_call.get("args") or {})
        self.semantic_backend.record_core_read(
            session,
            str(args.get("file_path") or ""),
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 2000),
        )
        return result


class GroundedRuntimeBudgetMiddleware(AgentMiddleware):
    """Enforce and measure the actual DeepAgent model/tool loop."""

    name = "GroundedRuntimeBudgetMiddleware"

    @staticmethod
    def _budget_from_runtime(runtime: Any) -> Optional[GroundedRuntimeBudget]:
        context = getattr(runtime, "context", None)
        budget = getattr(context, "budget", None)
        return budget if isinstance(budget, GroundedRuntimeBudget) else None

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        budget = self._budget_from_runtime(getattr(request, "runtime", None))
        if budget is None:
            return handler(request)
        budget.consume_llm_call(name="grounded_core")
        model_settings = dict(getattr(request, "model_settings", None) or {})
        model_settings["timeout"] = budget.clamp_timeout_seconds(
            model_settings.get("timeout")
        )
        override = getattr(request, "override", None)
        if callable(override):
            request = override(model_settings=model_settings)
        with budget.stage("llm.grounded_core"):
            return handler(request)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        budget = self._budget_from_runtime(getattr(request, "runtime", None))
        if budget is None:
            return handler(request)
        tool_name = str((getattr(request, "tool_call", None) or {}).get("name") or "tool")
        budget.consume_tool_call(tool_name)
        with budget.stage("tool.%s" % tool_name):
            return handler(request)


class GroundedDeepAgentRuntime:
    """Single DeepAgent Core backed only by GroundedRuntimeKernel tools."""

    SYSTEM_PROMPT = """You are the single Grounded merchant-analysis Core.

The first user message already contains the automatically selected Topic L0 manifest and one Topic-scoped thin recall. Recall is navigation evidence, never planning authority.
Before execution, inspect userInputRequirements and the progressively-read semantic capabilities. Time is required for analytical aggregates, rankings, trends and unbounded detail lists unless the user supplied it. A concrete entity lookup is different: when the selected semantic field declares an entity identity and its lookupTimePolicy permits global/unbounded lookup, do not ask for time; bind the entity filter and let the Contract gate validate that policy. Never infer this exception from a business-specific field name or value pattern.
The first user message also contains trustedExecutionScope. It is authoritative runtime state, not a user claim. When merchantScopeBound=true, never ask the user for merchant_id and never propose bypassing tenant filtering; the executor binds the declared merchant scope automatically.
Before proposing any query Contract, call declare_original_question_goals exactly once with a typed, complete ledger of the original question's metric, dimension, time-window, comparison, entity and dependency goals. Preserve explicit conjunctions and comparison operands. After exact semantic refs have been read, attach them to the corresponding metric, dimension and entity goals so final coverage can be checked against artifact evidence rather than labels alone. Every serial or parallel query must declare the goalIds it covers. Finalization is blocked until every required goal and dependency is covered by verified artifacts.
Use native ls/read_file/grep progressively under /knowledge. Read exact table detail, metric, column and relationship files before proposing bindings.
Published metric files already contain the governed formula, source columns, unit and time semantics. When metricRefs satisfy the question, do not also submit fieldAggregations for the same measures.
One Grounded Contract represents one coherent execution shape. Never combine metrics whose timeSemantics.selectionPolicy values differ. A period_window metric is a period scalar and must not be grouped by the time dimension; a per_time_grain metric over multiple days must preserve that time dimension. If the gate returns REVISE_BINDINGS, follow requiredCapability and submit a smaller compatible binding set before execution.
For a simple same-table scalar metric query, the expected disclosure path is table detail plus the exact metric files. Do not read schema, columns/index.json, the time column, or metric source-column files unless the question needs a field aggregation, business dimension, filter, join, or a published metric is unavailable.
When thinRecallCandidates already contains an exact readable path, read that path directly instead of opening an index. Never navigate to asset.json or a #fragment path.
Do not read optional name/label columns unless the user explicitly asks for a name/title. For ranking by an entity ID, the ID dimension is sufficient. labelRefs maps semantic ref IDs to the user's display phrase; it never carries a user's entity value. Put literal entity values only in typed entityFilters.
GroundedQueryContract is the only semantic planning authority. After it is READY, inspect executionMode. DETERMINISTIC_METRIC, DETERMINISTIC_MULTI_METRIC, DETERMINISTIC_GROUPED, DETERMINISTIC_TREND, DETERMINISTIC_RANKED and DETERMINISTIC_ENTITY_LOOKUP are runtime-owned deterministic compilation modes and may be executed directly; they compile only the already-grounded Contract and never plan goals or impose an execution order. CORE_SQL_REQUIRED means you must author the complete Doris SELECT/WITH SQL yourself and call submit_grounded_sql_candidate with the exact activeGeneration and contractFingerprint returned by propose_grounded_contract; never reuse these values after another Contract is proposed. Implement sqlObligations exactly. The runtime will not invent semantic bindings, joins, CTEs, windows, complex dependency logic, or fallback SQL for you. Never put merchant/tenant predicates or runtimeInjected upstream entity predicates in your SQL: trusted execution injects them after validation.
propose_grounded_contract.binding_hints has a strict schema. Use only tableRefs, metricRefs, fieldAggregations, dimensionRefs, selectedFields, entityFilters, upstreamEntityBindings, groupByRef, labelRefs, relationshipRefs, ranking, analysisMode and timeExpression. selectedFields contains exact fieldRef/outputAlias projections. entityFilters contains fieldRef/operator/literalValue/requestedPhrase and may only target a read field whose filterOperators allow that operator. upstreamEntityBindings contains only entitySetArtifactId/targetFieldRef/operator/requestedPhrase; never copy or invent its values. Use analysisMode=ENTITY_LOOKUP for a concrete entity lookup and DETAIL for an unbounded detail list. Never invent alternative keys such as tableRef, metricBindings, metrics, timeWindow or timeRange.
Available governed tools are declare_original_question_goals, retrieve_knowledge, propose_grounded_contract, prepare_grounded_query_batch, submit_grounded_sql_candidate, execute_grounded_query, execute_grounded_query_batch, publish_verified_entity_set, finalize_evidence_collection, compose_verified_answer, run_skill and ask_human. There is no action catalog, legacy planner, NodeAgent SQL writer, or complex-query template compiler.
One verified query may be only partial evidence for the user's question. When a later query depends on a verified entity output, call publish_verified_entity_set, progressively read the downstream target field, and propose a new Contract using upstreamEntityBindings. Do not treat a first successful TopN/entity query as the end of data collection. Each query remains an independent grounded QueryGraph chosen dynamically by you, not a fixed workflow.
When two or more query goals are independent and none uses upstreamEntityBindings, prepare them together with prepare_grounded_query_batch and execute them with execute_grounded_query_batch. The runtime gives every branch its own Contract generation and adopts only independently verified artifacts. Never batch an entity chain: publish the upstream entity set first, then run the dependent Contract serially.
Analysis Skill headers are not disclosed by an individual query. The Core cannot read SKILL.md and must never use a Skill procedure or header to choose metrics, dimensions, tables, or Contract shape. After every datum required by the original question is in the verified evidence portfolio, call finalize_evidence_collection. Only that gate may disclose Skill headers and seal data collection. Then select at most one matching Skill or compose the verified answer. run_skill is a one-way isolation boundary: after it starts, do not retrieve more knowledge, propose another Contract, execute another query, or call run_skill again. It mounts the selected full Skill for an independent subagent, workspace and checkpoint, streams progress, and publishes a structured result artifact. Do not use task for Skill execution.
Use retrieve_knowledge only for a targeted supplemental query; it remains inside the active Topic workspace. Governed-rule recall items marked INLINE_ONLY are usable snippets, not filesystem refs and not binding evidence. Do not read /knowledge/topics/index.json or open another Topic merely to compare alternatives. Topic expansion is allowed only after a submitted Contract returns REVISE_BINDINGS with a structured requiredCapability/searchScope gap based on evidence already read in the active Topic. A read relationship may establish that a required endpoint table is outside the current workspace; submit that relationship in the candidate Contract, then follow the returned gap to read the Topic index and exactly one relevant Topic manifest. Never expand from a normal pending request or a failed filename guess.
Do not call task in this runtime. SubAgent dispatch is disabled until worker evidence acceptance is independently auditable.
Never invent a formula, binding, SQL result, evidence status or answer. Finish only after compose_verified_answer, a verified run_skill result, or ask_human succeeds.
"""

    def __init__(
        self,
        kernel: GroundedRuntimeKernel,
        lead_model: Any,
        semantic_catalog: Any,
        *,
        checkpointer: Any = None,
        checkpoint_config_factory: Optional[
            Callable[[str, str], dict[str, Any]]
        ] = None,
        skill_root: Optional[str] = None,
        skill_run_root: Optional[str] = None,
        isolated_subagent_model: Any = None,
        parallel_max_workers: int = 4,
        settings: Any = None,
        agent_factory: Any = None,
        backend: Any = None,
    ):
        self.kernel = kernel
        self.semantic_catalog = semantic_catalog
        self.settings = settings
        self.checkpointer = checkpointer
        self.checkpoint_config_factory = checkpoint_config_factory
        self.parallel_max_workers = max(1, min(int(parallel_max_workers or 1), 8))
        self.skill_root = Path(skill_root).resolve() if skill_root else None
        self.skill_run_root = Path(
            skill_run_root or ".merchant-ai/skill-runs"
        ).resolve()
        self.skill_run_root.mkdir(parents=True, exist_ok=True)
        self.skill_headers = self._load_skill_headers()
        # Native backend reads provide content only. Root-Core evidence authority
        # is recorded by tool middleware, never by ambient thread-local config.
        self.knowledge_backend = GroundedSemanticBackend(
            semantic_catalog,
            reader_is_core=lambda: False,
        )
        self.core_tool_boundary = GroundedCoreToolBoundaryMiddleware(
            self.knowledge_backend
        )
        self.budget_middleware = GroundedRuntimeBudgetMiddleware()
        self.backend = backend or self._build_backend()
        self.tools = self._build_tools()
        self.initialization_error = ""
        self.deep_agent_graph: Any = None
        model = self._resolve_model(lead_model)
        if model is None:
            raise RuntimeError("Grounded DeepAgent initialization failed: model is not configured")
        if agent_factory is None:
            if create_deep_agent is None:
                raise RuntimeError(
                    "Grounded DeepAgent initialization failed: deepagents unavailable: %s"
                    % _DEEPAGENTS_IMPORT_ERROR
                )
            agent_factory = create_deep_agent
        self._model = model
        self._agent_factory = agent_factory
        subagent_model = self._resolve_model(isolated_subagent_model) or model
        self.subagent_runtime = IsolatedSubagentRuntime(
            model=subagent_model,
            agent_factory=agent_factory,
            checkpointer=checkpointer,
            checkpoint_config_factory=checkpoint_config_factory,
        )
        try:
            self.deep_agent_graph = agent_factory(
                model=model,
                tools=self.tools,
                system_prompt=self.SYSTEM_PROMPT,
                middleware=[self.budget_middleware, self.core_tool_boundary],
                subagents=[
                    {
                        # deepagents 0.6.x otherwise appends a default worker
                        # named general-purpose which inherits every custom
                        # tool.  Override that exact identity with an explicit
                        # zero-custom-tool read-only worker.
                        "name": "general-purpose",
                        "description": "Read-only isolated semantic investigation for parallel evidence gathering.",
                        "system_prompt": (
                            "Use only native read-only filesystem tools. Return refs and concise findings. "
                            "Do not route, propose a Contract, execute SQL, verify evidence, answer, or ask the user."
                        ),
                        "tools": [],
                        "skills": None,
                    }
                ],
                skills=None,
                permissions=[
                    FilesystemPermission(
                        operations=["write"],
                        paths=["/knowledge", "/knowledge/**", "/skills", "/skills/**"],
                        mode="deny",
                    )
                ],
                backend=self.backend,
                context_schema=GroundedDeepAgentRunContext,
                checkpointer=checkpointer,
                name="grounded_merchant_core",
            )
        except Exception as exc:
            self.initialization_error = "%s:%s" % (type(exc).__name__, str(exc)[:500])
            raise RuntimeError(
                "Grounded DeepAgent initialization failed: %s" % self.initialization_error
            ) from exc
        if self.deep_agent_graph is None:
            raise RuntimeError("Grounded DeepAgent initialization failed: agent factory returned no graph")

    @staticmethod
    def _resolve_model(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "configured") and not bool(value.configured):
            return None
        if hasattr(value, "chat_model"):
            return value.chat_model()
        return value

    def _build_backend(self) -> CompositeBackend:
        # Full Skill bodies are intentionally absent from the parent Core.
        # Only run_skill mounts one selected Skill into an isolated backend.
        routes: dict[str, Any] = {"/knowledge/": self.knowledge_backend}
        return CompositeBackend(
            default=StateBackend(),
            routes=routes,
            artifacts_root="/workspace",
        )

    def _load_skill_headers(self) -> list[dict[str, str]]:
        if self.skill_root is None or not self.skill_root.is_dir():
            return []
        headers: list[dict[str, str]] = []
        for skill_dir in sorted(self.skill_root.iterdir(), key=lambda path: path.name):
            skill_file = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not skill_file.is_file():
                continue
            metadata = _load_skill_frontmatter(skill_file)
            name = str(metadata.get("name") or "").strip()
            description = str(metadata.get("description") or "").strip()
            if not name or not description or name != skill_dir.name:
                continue
            headers.append(
                {
                    "name": name,
                    "description": description,
                    "title": str(metadata.get("title") or "").strip(),
                    "lifecyclePhase": str(
                        metadata.get("lifecyclePhase")
                        or metadata.get("lifecycle_phase")
                        or "post_query_analysis"
                    ).strip(),
                    "requiresVerifiedEvidence": str(
                        metadata.get("requiresVerifiedEvidence")
                        or metadata.get("requires_verified_evidence")
                        or "true"
                    ).strip().lower(),
                    "outputContract": str(
                        metadata.get("outputContract")
                        or metadata.get("output_contract")
                        or ""
                    ).strip(),
                }
            )
        return headers

    def _build_tools(self) -> list[Any]:
        runtime_owner = self

        @tool("declare_original_question_goals")
        def declare_original_question_goals(
            contract: OriginalQuestionGoalContract,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Commit the immutable, typed coverage ledger for the original question."""

            deep_session = runtime.context.session
            try:
                parsed = parse_original_question_goal_contract(contract)
            except Exception as exc:
                issues = [
                    item.model_dump(by_alias=True)
                    for item in getattr(exc, "issues", ())
                ]
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_INVALID",
                        "message": str(exc)[:500],
                        "issues": issues,
                        "nextAction": "REVISE_GOAL_CONTRACT",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            if parsed.question.strip() != deep_session.runtime.question.strip():
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "GOAL_CONTRACT_QUESTION_MISMATCH",
                        "message": "The goal ledger must retain the exact original question.",
                    },
                    ensure_ascii=False,
                )
            fingerprint = original_question_goal_contract_fingerprint(parsed)
            with deep_session.lock:
                existing = deep_session.question_goal_contract
                if existing is not None:
                    existing_fingerprint = (
                        original_question_goal_contract_fingerprint(existing)
                    )
                    if existing_fingerprint != fingerprint and (
                        deep_session.runtime.attempts
                        or deep_session.runtime.verified_query_ledger
                        or deep_session.parallel_branches
                    ):
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": "GOAL_CONTRACT_IMMUTABLE_AFTER_QUERY_START",
                                "contractFingerprint": existing_fingerprint,
                            },
                            ensure_ascii=False,
                        )
                deep_session.question_goal_contract = parsed.model_copy(deep=True)
            return json.dumps(
                {
                    "status": "ACCEPTED",
                    "contractId": parsed.contract_id,
                    "contractFingerprint": fingerprint,
                    "requiredGoalIds": required_goal_ids(parsed),
                    "goals": [
                        {
                            "goalId": goal.goal_id,
                            "kind": goal.kind,
                            "label": goal.label,
                            "required": goal.required,
                            "dependsOnGoalIds": list(goal.depends_on_goal_ids),
                        }
                        for goal in parsed.goals
                    ],
                    "nextAction": "READ_SEMANTICS_AND_PROPOSE_QUERY_CONTRACTS",
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("retrieve_knowledge")
        def retrieve_knowledge(
            query: str,
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Run one targeted supplemental recall inside the active Topic scope."""

            if (
                runtime.context.session.analysis_skill_started
                or runtime.context.session.data_collection_sealed
            ):
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                        "message": "Analysis Skill execution cannot drive new retrieval or query planning.",
                    },
                    ensure_ascii=False,
                )

            try:
                bundle = runtime_owner.kernel.recall_navigation(
                    runtime.context.session.runtime,
                    query=str(query or "").strip(),
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "RECALL_NAVIGATION_DEGRADED",
                        "code": "RECALL_BACKEND_UNAVAILABLE",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "recallCandidates": [],
                        "scope": runtime.context.session.effective_topics(),
                        "nextAction": "CONTINUE_WITH_FILESYSTEM",
                        "instruction": (
                            "Recall is navigation only. Continue from the mounted Topic L0 manifest with ls/read_file/grep."
                        ),
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "status": "OK",
                    "reason": str(reason or "")[:500],
                    "recallCandidates": _thin_recall(bundle, limit=8),
                    "scope": runtime.context.session.effective_topics(),
                },
                ensure_ascii=False,
            )

        @tool("propose_grounded_contract")
        def propose_grounded_contract(
            read_ref_ids: list[str],
            binding_hints: GroundedBindingHints,
            goal_ids: list[str],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Propose from exact Core reads using the strict typed BindingHints schema."""

            session = runtime.context.session
            if session.analysis_skill_started or session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                        "message": "Analysis Skill execution cannot revise semantic bindings.",
                    },
                    ensure_ascii=False,
                )
            assigned_goal_ids, goal_error = _validated_goal_assignment(
                session,
                goal_ids,
            )
            if goal_error:
                return json.dumps(goal_error, ensure_ascii=False)
            if not isinstance(binding_hints, GroundedBindingHints):
                binding_hints = GroundedBindingHints.model_validate(binding_hints)
            binding_hints = _canonical_binding_hints(binding_hints)
            requested = list(
                dict.fromkeys(
                    _canonical_progressive_ref(str(item or "").strip())
                    for item in read_ref_ids
                    if str(item or "").strip()
                )
            )
            with session.lock:
                evidence_by_ref = {
                    str(item.get("refId") or ""): dict(item)
                    for item in session.core_semantic_evidence
                    if str(item.get("refId") or "")
                }
            missing = [ref_id for ref_id in requested if ref_id not in evidence_by_ref]
            if missing:
                read_next: list[dict[str, str]] = []
                for ref_id in missing:
                    try:
                        resolved = runtime_owner.semantic_catalog.read(
                            ref_id=ref_id,
                            max_chars=1,
                            offset=0,
                        )
                    except Exception:
                        resolved = {}
                    read_next.append(
                        {
                            "refId": ref_id,
                            "path": str((resolved or {}).get("path") or ""),
                        }
                    )
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "SEMANTIC_REF_NOT_READ",
                        "missingRefs": missing,
                        "readNext": read_next,
                        "instruction": (
                            "Read each non-empty /knowledge path completely, then resubmit "
                            "the same typed binding_hints."
                        ),
                    },
                    ensure_ascii=False,
                )
            evidence = [evidence_by_ref[ref_id] for ref_id in requested]
            try:
                attempt = runtime_owner.kernel.propose_contract(
                    session.runtime,
                    evidence,
                    binding_hints,
                    topics=session.effective_topics(),
                )
                assignment_issues = _goal_assignment_contract_issues(
                    session,
                    assigned_goal_ids,
                    attempt.contract,
                )
                if assignment_issues:
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "QUERY_GOAL_ASSIGNMENT_MISMATCH",
                            "issues": assignment_issues,
                            "nextAction": "REVISE_BINDINGS_OR_GOAL_ASSIGNMENT",
                        },
                        ensure_ascii=False,
                    )
                if attempt.contract.ready:
                    attempt = runtime_owner.kernel.activate_contract(
                        session.runtime,
                        attempt.attempt_id,
                    )
            except RuntimeError as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "GROUNDED_CONTRACT_ACTIVATION_BLOCKED",
                        "message": str(exc)[:500],
                        "nextAction": "STOP"
                        if "TERMINAL_GUARD" in str(exc)
                        else "REVISE_BINDINGS",
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "GROUNDED_CONTRACT_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            if attempt.activated:
                with session.lock:
                    session.active_goal_ids = list(assigned_goal_ids)
            contract_fingerprint = grounded_query_contract_fingerprint(
                attempt.contract
            )
            return json.dumps(
                {
                    "attemptId": attempt.attempt_id,
                    "status": attempt.contract.status,
                    "queryShape": attempt.contract.query_shape,
                    "compileStatus": attempt.compile_status,
                    "activationStatus": attempt.activation_status,
                    "activated": attempt.activated,
                    "executionMode": attempt.execution_mode,
                    "executionReasonCodes": attempt.execution_reason_codes,
                    "fastPathEligible": attempt.fast_path_eligible,
                    "fastPathReasonCodes": attempt.fast_path_reason_codes,
                    "fastPathReasonDetails": attempt.fast_path_reason_details,
                    "nextAction": attempt.next_action,
                    "activeGeneration": attempt.active_generation,
                    "contractFingerprint": contract_fingerprint,
                    "sqlObligations": _grounded_contract_sql_obligations(
                        attempt.contract
                    ),
                    "acceptedBindingHints": _core_visible_binding_hints(
                        attempt.contract
                    ),
                    "assignedGoalIds": list(assigned_goal_ids),
                    "gaps": [
                        gap.model_dump(by_alias=True)
                        for gap in attempt.contract.unresolved_gaps
                    ],
                    "rejectedBindings": [
                        item.model_dump(by_alias=True)
                        for item in attempt.contract.rejected_bindings
                    ],
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("prepare_grounded_query_batch")
        def prepare_grounded_query_batch(
            queries: list[GroundedParallelQuerySpec],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Prepare isolated Contract generations for independent query goals."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                    },
                    ensure_ascii=False,
                )
            normalized = [
                item
                if isinstance(item, GroundedParallelQuerySpec)
                else GroundedParallelQuerySpec.model_validate(item)
                for item in queries
            ]
            if not normalized:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_QUERY_BATCH_EMPTY",
                    },
                    ensure_ascii=False,
                )
            if len(normalized) > runtime_owner.parallel_max_workers:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_QUERY_BATCH_TOO_LARGE",
                        "maxWorkers": runtime_owner.parallel_max_workers,
                    },
                    ensure_ascii=False,
                )
            query_ids = [str(item.query_id or "").strip() for item in normalized]
            if any(not item for item in query_ids) or len(set(query_ids)) != len(query_ids):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_QUERY_ID_INVALID",
                    },
                    ensure_ascii=False,
                )
            goal_assignments_by_query_id: dict[str, list[str]] = {}
            goal_errors_by_query_id: dict[str, dict[str, Any]] = {}
            for raw_item, query_id in zip(normalized, query_ids):
                assigned_goal_ids, goal_error = _validated_goal_assignment(
                    deep_session,
                    raw_item.goal_ids,
                )
                goal_assignments_by_query_id[query_id] = assigned_goal_ids
                if goal_error:
                    goal_errors_by_query_id[query_id] = goal_error
            goal_contract = deep_session.question_goal_contract
            dependency_issues = (
                _parallel_goal_dependency_issues(
                    goal_contract,
                    {
                        query_id: goal_ids
                        for query_id, goal_ids in goal_assignments_by_query_id.items()
                        if query_id not in goal_errors_by_query_id
                    },
                )
                if goal_contract is not None
                else []
            )
            if dependency_issues:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_GOAL_DEPENDENCY_DETECTED",
                        "issues": dependency_issues,
                        "instruction": (
                            "Only mutually independent goals may share a batch. "
                            "Dependency-connected goals are ineligible for this batch "
                            "and must use the governed serial query tools."
                        ),
                        "nextAction": "USE_SERIAL_GROUNDED_QUERY_TOOLS",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            with deep_session.lock:
                evidence_by_ref = {
                    str(item.get("refId") or ""): dict(item)
                    for item in deep_session.core_semantic_evidence
                    if str(item.get("refId") or "")
                }
            prepared: list[dict[str, Any]] = []
            for raw_item, query_id in zip(normalized, query_ids):
                assigned_goal_ids = goal_assignments_by_query_id[query_id]
                goal_error = goal_errors_by_query_id.get(query_id, {})
                if goal_error:
                    prepared.append(
                        {
                            "queryId": query_id,
                            **goal_error,
                        }
                    )
                    continue
                hints = _canonical_binding_hints(raw_item.binding_hints)
                if hints.upstream_entity_bindings:
                    prepared.append(
                        {
                            "queryId": query_id,
                            "status": "REJECTED",
                            "code": "DEPENDENT_QUERY_REQUIRES_SERIAL_EXECUTION",
                            "instruction": (
                                "Publish and bind the verified upstream entity set, then use "
                                "the serial propose/submit/execute tools."
                            ),
                        }
                    )
                    continue
                requested = list(
                    dict.fromkeys(
                        _canonical_progressive_ref(str(ref_id or "").strip())
                        for ref_id in raw_item.read_ref_ids
                        if str(ref_id or "").strip()
                    )
                )
                missing = [ref_id for ref_id in requested if ref_id not in evidence_by_ref]
                if missing:
                    prepared.append(
                        {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": "SEMANTIC_REF_NOT_READ",
                            "missingRefs": missing,
                        }
                    )
                    continue
                branch = runtime_owner.kernel.fork_query_branch(
                    deep_session.runtime,
                    query_id,
                )
                try:
                    attempt = runtime_owner.kernel.propose_contract(
                        branch,
                        [evidence_by_ref[ref_id] for ref_id in requested],
                        hints,
                        topics=deep_session.effective_topics(),
                    )
                    assignment_issues = _goal_assignment_contract_issues(
                        deep_session,
                        assigned_goal_ids,
                        attempt.contract,
                    )
                    if assignment_issues:
                        prepared.append(
                            {
                                "queryId": query_id,
                                "status": "BLOCKED",
                                "code": "QUERY_GOAL_ASSIGNMENT_MISMATCH",
                                "issues": assignment_issues,
                            }
                        )
                        continue
                    if attempt.contract.ready:
                        attempt = runtime_owner.kernel.activate_contract(
                            branch,
                            attempt.attempt_id,
                        )
                except GroundedRuntimeBudgetExceeded:
                    raise
                except Exception as exc:
                    prepared.append(
                        {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": "PARALLEL_CONTRACT_PREPARATION_FAILED",
                            "message": "%s:%s"
                            % (type(exc).__name__, str(exc)[:400]),
                        }
                    )
                    continue
                if not attempt.activated:
                    prepared.append(
                        {
                            "queryId": query_id,
                            "status": attempt.contract.status,
                            "code": "PARALLEL_CONTRACT_NOT_READY",
                            "gaps": [
                                gap.model_dump(by_alias=True)
                                for gap in attempt.contract.unresolved_gaps
                            ],
                        }
                    )
                    continue
                with deep_session.lock:
                    deep_session.parallel_branches[query_id] = branch
                    deep_session.parallel_branch_goal_ids[query_id] = list(
                        assigned_goal_ids
                    )
                prepared.append(
                    {
                        "queryId": query_id,
                        "status": "PREPARED",
                        "queryShape": attempt.contract.query_shape,
                        "executionMode": attempt.execution_mode,
                        "activeGeneration": attempt.active_generation,
                        "contractFingerprint": grounded_query_contract_fingerprint(
                            attempt.contract
                        ),
                        "sqlObligations": _grounded_contract_sql_obligations(
                            attempt.contract
                        ),
                        "goalIds": list(assigned_goal_ids),
                        "nextAction": (
                            "EXECUTE_BATCH"
                            if attempt.execution_mode
                            != "CORE_SQL_REQUIRED"
                            else "AUTHOR_SQL_THEN_EXECUTE_BATCH"
                        ),
                    }
                )
            ready_count = sum(item.get("status") == "PREPARED" for item in prepared)
            return json.dumps(
                {
                    "status": "PREPARED" if ready_count else "BLOCKED",
                    "preparedCount": ready_count,
                    "maxWorkers": runtime_owner.parallel_max_workers,
                    "queries": prepared,
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("execute_grounded_query_batch")
        def execute_grounded_query_batch(
            queries: list[GroundedParallelExecutionSpec],
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Execute prepared independent branches concurrently and adopt verified artifacts."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                    },
                    ensure_ascii=False,
                )
            normalized = [
                item
                if isinstance(item, GroundedParallelExecutionSpec)
                else GroundedParallelExecutionSpec.model_validate(item)
                for item in queries
            ]
            if not normalized or len(normalized) > runtime_owner.parallel_max_workers:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_EXECUTION_BATCH_SIZE_INVALID",
                        "maxWorkers": runtime_owner.parallel_max_workers,
                    },
                    ensure_ascii=False,
                )
            query_ids = [str(item.query_id or "").strip() for item in normalized]
            if any(not item for item in query_ids) or len(set(query_ids)) != len(query_ids):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_QUERY_ID_INVALID",
                    },
                    ensure_ascii=False,
                )
            with deep_session.lock:
                branch_by_id = {
                    query_id: deep_session.parallel_branches.get(query_id)
                    for query_id in query_ids
                }
            missing_branches = [
                query_id for query_id, branch in branch_by_id.items() if branch is None
            ]
            if missing_branches:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_BRANCH_NOT_PREPARED",
                        "queryIds": missing_branches,
                    },
                    ensure_ascii=False,
                )

            def execute_one(
                item: GroundedParallelExecutionSpec,
            ) -> tuple[str, GroundedRuntimeSession, dict[str, Any]]:
                query_id = str(item.query_id).strip()
                branch = branch_by_id[query_id]
                assert branch is not None
                try:
                    if str(
                        getattr(
                            branch.active_execution_mode,
                            "value",
                            branch.active_execution_mode,
                        )
                    ) == "CORE_SQL_REQUIRED":
                        if not str(item.sql or "").strip():
                            return query_id, branch, {
                                "queryId": query_id,
                                "status": "BLOCKED",
                                "code": "CORE_SQL_CANDIDATE_REQUIRED",
                            }
                        contract = branch.active_contract
                        if contract is None:
                            raise RuntimeError("parallel branch has no active Contract")
                        submitted = runtime_owner.kernel.submit_sql_candidate(
                            branch,
                            item.sql,
                            expected_generation=branch.active_generation,
                            expected_contract_fingerprint=(
                                grounded_query_contract_fingerprint(contract)
                            ),
                            rationale=item.rationale or reason,
                            evidence_refs=item.evidence_ref_ids,
                        )
                        if submitted.status != "ACCEPTED":
                            return query_id, branch, {
                                "queryId": query_id,
                                "status": submitted.status,
                                "nextAction": submitted.next_action,
                                "gaps": submitted.validation_gaps,
                            }
                    budget = runtime.context.budget
                    if budget is not None:
                        budget.consume_doris_query(
                            name="parallel.%s" % query_id
                        )
                    if budget is None:
                        run_result = runtime_owner.kernel.execute_active(
                            branch,
                            run_id="%s__%s" % (runtime.context.run_id, query_id),
                        )
                        verified = runtime_owner.kernel.verify_active(branch)
                    else:
                        with budget.stage("doris.parallel.%s" % query_id):
                            run_result = runtime_owner.kernel.execute_active(
                                branch,
                                run_id="%s__%s"
                                % (runtime.context.run_id, query_id),
                                runtime_budget=budget,
                            )
                        with budget.stage("evidence.parallel.%s" % query_id):
                            verified = runtime_owner.kernel.verify_active(branch)
                    artifact = runtime_owner.kernel.latest_verified_query_artifact(
                        branch
                    )
                    return query_id, branch, {
                        "queryId": query_id,
                        "status": "VERIFIED" if verified.passed else "VERIFICATION_GAPPED",
                        "queryArtifactId": artifact.artifact_id if artifact else "",
                        "rowCount": len(run_result.merged_query_bundle.rows),
                        "tables": list(run_result.merged_query_bundle.tables),
                        "blockingGaps": [
                            gap.model_dump(by_alias=True)
                            for gap in verified.blocking_gaps
                        ],
                    }
                except GroundedRuntimeBudgetExceeded:
                    raise
                except Exception as exc:
                    return query_id, branch, {
                        "queryId": query_id,
                        "status": "FAILED",
                        "code": "PARALLEL_QUERY_EXECUTION_FAILED",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                    }

            results: list[dict[str, Any]] = []
            successful_branches: list[GroundedRuntimeSession] = []
            with ThreadPoolExecutor(
                max_workers=min(len(normalized), runtime_owner.parallel_max_workers),
                thread_name_prefix="grounded-query",
            ) as pool:
                futures = {pool.submit(execute_one, item): item.query_id for item in normalized}
                for future in as_completed(futures):
                    query_id, branch, result = future.result()
                    results.append(result)
                    if result.get("status") == "VERIFIED":
                        successful_branches.append(branch)
            successful_branches.sort(
                key=lambda branch: query_ids.index(
                    next(
                        query_id
                        for query_id, candidate in branch_by_id.items()
                        if candidate is branch
                    )
                )
            )
            adopted = runtime_owner.kernel.adopt_verified_branches(
                deep_session.runtime,
                successful_branches,
            )
            artifact_ids = {artifact.artifact_id for artifact in adopted}
            with deep_session.lock:
                for result in results:
                    artifact_id = str(result.get("queryArtifactId") or "")
                    query_id = str(result.get("queryId") or "")
                    if artifact_id and artifact_id in artifact_ids:
                        deep_session.artifact_goal_ids[artifact_id] = list(
                            deep_session.parallel_branch_goal_ids.get(query_id) or []
                        )
                for query_id in query_ids:
                    deep_session.parallel_branches.pop(query_id, None)
                    deep_session.parallel_branch_goal_ids.pop(query_id, None)
            results.sort(key=lambda item: query_ids.index(str(item.get("queryId") or "")))
            return json.dumps(
                {
                    "status": (
                        "VERIFIED"
                        if len(adopted) == len(normalized)
                        else "PARTIAL"
                        if adopted
                        else "FAILED"
                    ),
                    "reason": str(reason or "")[:500],
                    "executedInParallel": True,
                    "workerCount": min(
                        len(normalized), runtime_owner.parallel_max_workers
                    ),
                    "adoptedArtifactIds": [item.artifact_id for item in adopted],
                    "queries": results,
                    "nextAction": (
                        "CONTINUE_QUERYING_OR_FINALIZE"
                        if adopted
                        else "REVISE_BINDINGS_OR_SQL"
                    ),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("submit_grounded_sql_candidate")
        def submit_grounded_sql_candidate(
            sql: str,
            expected_generation: int,
            contract_fingerprint: str,
            rationale: str,
            evidence_ref_ids: list[str],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Submit the complete Doris SQL authored by Core for the active Contract."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                        "message": "Analysis Skill execution cannot author or revise SQL.",
                    },
                    ensure_ascii=False,
                )
            try:
                attempt = runtime_owner.kernel.submit_sql_candidate(
                    deep_session.runtime,
                    sql,
                    expected_generation=expected_generation,
                    expected_contract_fingerprint=contract_fingerprint,
                    rationale=rationale,
                    evidence_refs=evidence_ref_ids,
                )
            except GroundedRuntimeBudgetExceeded:
                raise
            except RuntimeError as exc:
                message = str(exc)
                stale = "SQL_CANDIDATE_STALE_CONTRACT" in message
                terminal = "TERMINAL_GUARD" in message
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": (
                            "SQL_CANDIDATE_STALE_CONTRACT"
                            if stale
                            else "TERMINAL_GUARD"
                            if terminal
                            else "CORE_SQL_NOT_AUTHORIZED"
                        ),
                        "message": message[:500],
                        "nextAction": (
                            "USE_LATEST_CONTRACT"
                            if stale
                            else "STOP"
                            if terminal
                            else "PROPOSE_GROUNDED_CONTRACT"
                        ),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "SQL_CANDIDATE_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            internal_error = attempt.status == "VALIDATOR_INTERNAL_ERROR"
            return json.dumps(
                {
                    "candidateId": attempt.candidate_id,
                    "status": "BLOCKED" if internal_error else attempt.status,
                    "code": (
                        "SQL_CANDIDATE_VALIDATOR_INTERNAL_ERROR"
                        if internal_error
                        else ""
                    ),
                    "activeGeneration": attempt.active_generation,
                    "nextAction": (
                        "STOP_INTERNAL"
                        if internal_error
                        else attempt.next_action
                    ),
                    "astFingerprint": attempt.ast_fingerprint,
                    "contractFingerprint": attempt.contract_fingerprint,
                    "outputColumns": attempt.output_columns,
                    "gaps": attempt.validation_gaps,
                    "instruction": (
                        "Execute only when status=ACCEPTED. For REPAIR_SQL, change the SQL AST "
                        "using the exact gap. For REVISE_BINDINGS, progressively read missing "
                        "semantic assets and propose a new Contract generation. Never retry the "
                        "same SQL/error state."
                    ),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("execute_grounded_query")
        def execute_grounded_query(
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Execute the active compiled Contract and immediately verify evidence."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                        "message": "Analysis Skill execution cannot trigger another SQL query.",
                    },
                    ensure_ascii=False,
                )
            session = deep_session.runtime
            try:
                budget = runtime.context.budget
                if budget is not None:
                    budget.consume_doris_query(name="serial.grounded_query")
                if budget is None:
                    run_result = runtime_owner.kernel.execute_active(
                        session,
                        run_id=runtime.context.run_id,
                    )
                    verified = runtime_owner.kernel.verify_active(session)
                else:
                    with budget.stage("doris.serial"):
                        run_result = runtime_owner.kernel.execute_active(
                            session,
                            run_id=runtime.context.run_id,
                            runtime_budget=budget,
                        )
                    with budget.stage("evidence.serial"):
                        verified = runtime_owner.kernel.verify_active(session)
            except GroundedRuntimeBudgetExceeded:
                raise
            except RuntimeError as exc:
                message = str(exc)
                no_progress = "SQL_EXECUTION_NO_PROGRESS" in message
                core_sql_required = "CORE_SQL_REQUIRED" in message or no_progress
                return json.dumps(
                    {
                        "status": "EXECUTION_REVISE_REQUIRED",
                        "code": (
                            "SQL_EXECUTION_NO_PROGRESS"
                            if no_progress
                            else
                            "CORE_SQL_CANDIDATE_REQUIRED"
                            if core_sql_required
                            else "GROUNDED_EXECUTION_COMPATIBILITY_BLOCKED"
                        ),
                        "message": message[:500],
                        "nextAction": (
                            "SUBMIT_GROUNDED_SQL_CANDIDATE"
                            if core_sql_required
                            else "REVISE_BINDINGS"
                        ),
                        "instruction": (
                            (
                                "Submit a materially changed SQL candidate; executing the same accepted AST again is forbidden."
                                if no_progress
                                else "Author and submit the complete SQL for the active Contract before execution."
                            )
                            if core_sql_required
                            else (
                                "Return to the progressively read semantic evidence, submit a "
                                "smaller compatible Grounded Contract, and execute only after "
                                "the Contract gate activates it. Do not retry the same bindings."
                            )
                        ),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "GROUNDED_EXECUTION_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            failed_results = [
                item
                for item in run_result.task_results
                if item.query_bundle.failed or not item.success
            ]
            if failed_results:
                failure_codes = [
                    result.error_code
                    for item in failed_results
                    for result in item.validation_results
                    if result.error_code
                ]
                failure_code = failure_codes[0] if failure_codes else "QUERY_EXECUTION_FAILED"
                access_denied = failure_code in {
                    "ACCESS_DENIED",
                    "MERCHANT_SCOPE_DENIED",
                    "TABLE_DENIED",
                    "TABLE_NOT_ALLOWED",
                    "TABLE_ROLE_DENIED",
                    "COLUMN_DENIED",
                }
                core_sql_mode = str(
                    getattr(
                        session.active_execution_mode,
                        "value",
                        session.active_execution_mode,
                    )
                ) == "CORE_SQL_REQUIRED"
                return json.dumps(
                    {
                        "status": (
                            "ACCESS_DENIED"
                            if access_denied
                            else "SQL_EXECUTION_REPAIR_REQUIRED"
                            if core_sql_mode
                            else "EXECUTION_FAILED"
                        ),
                        "code": failure_code,
                        "nextAction": (
                            "STOP_ACCESS_DENIED"
                            if access_denied
                            else "SUBMIT_GROUNDED_SQL_CANDIDATE"
                            if core_sql_mode
                            else "REVISE_BINDINGS"
                        ),
                        "message": str(
                            failed_results[0].query_bundle.error
                            or failed_results[0].summary
                            or failure_code
                        )[:500],
                        "blockingGaps": [
                            gap.model_dump(by_alias=True)
                            for gap in verified.blocking_gaps
                        ],
                        "instruction": (
                            "Access denial is terminal for this request; do not alter SQL to bypass policy."
                            if access_denied
                            else (
                                "Use the execution error and active Contract to author one changed SQL AST. "
                                "Do not rerun the same accepted candidate."
                            )
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            latest_artifact = getattr(
                runtime_owner.kernel,
                "latest_verified_query_artifact",
                None,
            )
            query_artifact = (
                latest_artifact(session)
                if verified.passed and callable(latest_artifact)
                else None
            )
            if query_artifact is not None:
                with deep_session.lock:
                    deep_session.artifact_goal_ids[
                        query_artifact.artifact_id
                    ] = list(deep_session.active_goal_ids)
            return json.dumps(
                {
                    "status": "VERIFIED" if verified.passed else "VERIFICATION_GAPPED",
                    "reason": str(reason or "")[:500],
                    "queryArtifactId": (
                        query_artifact.artifact_id if query_artifact else ""
                    ),
                    "coveredGoalIds": list(
                        deep_session.active_goal_ids if query_artifact else []
                    ),
                    "rowCount": len(run_result.merged_query_bundle.rows),
                    "tables": list(run_result.merged_query_bundle.tables),
                    "outputColumns": (
                        list(query_artifact.output_columns)
                        if query_artifact
                        else []
                    ),
                    "entitySetEligibleOutputs": (
                        sorted(query_artifact.output_entity_identities)
                        if query_artifact
                        else []
                    ),
                    "blockingGaps": [
                        gap.model_dump(by_alias=True) for gap in verified.blocking_gaps
                    ],
                    "warningGaps": [
                        gap.model_dump(by_alias=True) for gap in verified.warning_gaps
                    ],
                    "dataCollectionStatus": "OPEN",
                    "nextAction": (
                        "PUBLISH_ENTITY_SET_OR_CONTINUE_QUERYING_OR_FINALIZE"
                        if verified.passed
                        else "REVISE_BINDINGS_OR_SQL"
                    ),
                    "skillSelectionPolicy": (
                        "No Skill header is disclosed by an individual query. Finalize the complete "
                        "verified evidence portfolio only after every datum required by the original question is present."
                    ),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("publish_verified_entity_set")
        def publish_verified_entity_set(
            query_artifact_id: str,
            output_column: str,
            limit: int,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Publish a typed entity set from one immutable verified query output."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                        "message": "Analysis Skill execution cannot publish new query inputs.",
                    },
                    ensure_ascii=False,
                )
            try:
                artifact = runtime_owner.kernel.publish_verified_entity_set(
                    deep_session.runtime,
                    query_artifact_id,
                    output_column,
                    limit=limit,
                )
            except RuntimeError as exc:
                message = str(exc)
                code = message.partition(":")[0] or "VERIFIED_ENTITY_SET_REJECTED"
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": code,
                        "message": message[:500],
                        "nextAction": (
                            "USE_LATEST_VERIFIED_QUERY_ARTIFACT"
                            if code == "VERIFIED_QUERY_ARTIFACT_NOT_FOUND"
                            else "REVISE_BINDINGS"
                            if code in {
                                "VERIFIED_ENTITY_OUTPUT_COLUMN_NOT_FOUND",
                                "VERIFIED_ENTITY_SEMANTIC_LINEAGE_REQUIRED",
                                "VERIFIED_ENTITY_IDENTITY_REQUIRED",
                            }
                            else "STOP_WITH_VERIFIED_EMPTY_RESULT"
                            if code == "VERIFIED_ENTITY_SET_EMPTY"
                            else "STOP"
                        ),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "VERIFIED_ENTITY_SET_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "status": "PUBLISHED",
                    "entitySetArtifactId": artifact.artifact_id,
                    "sourceQueryArtifactId": artifact.source_query_artifact_id,
                    "sourceColumn": artifact.source_column,
                    "sourceSemanticRefId": artifact.source_semantic_ref_id,
                    "entityIdentity": artifact.source_entity_identity,
                    "valueCount": artifact.value_count,
                    "truncated": artifact.truncated,
                    "valuesHash": artifact.values_hash,
                    "nextAction": (
                        "STOP_WITH_VERIFIED_EMPTY_RESULT"
                        if artifact.value_count == 0
                        else "REVISE_QUERY_STRATEGY"
                        if artifact.truncated
                        else "READ_DOWNSTREAM_FIELD_AND_PROPOSE_CONTRACT"
                    ),
                    "instruction": (
                        "Bind this artifact by entitySetArtifactId and a progressively read "
                        "targetFieldRef. Never copy or invent entity values in entityFilters."
                    ),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("compose_verified_answer")
        def compose_verified_answer(
            allow_llm: bool,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Compose and store the final answer from verified evidence only."""

            try:
                runtime_owner._require_complete_goal_coverage(
                    runtime.context.session
                )
            except GoalCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                runtime.context.session.goal_coverage_result = result
                return json.dumps(
                    {
                        "status": "GOAL_COVERAGE_INCOMPLETE",
                        "code": "ORIGINAL_QUESTION_GOALS_UNCOVERED",
                        "missingRequiredGoalIds": result.get(
                            "missingRequiredGoalIds", []
                        ),
                        "issues": result.get("issues", []),
                        "nextAction": "CONTINUE_PROGRESSIVE_QUERYING",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except RuntimeError as exc:
                return json.dumps(
                    {
                        "status": "GOAL_CONTRACT_REQUIRED",
                        "code": str(exc),
                        "nextAction": "DECLARE_ORIGINAL_QUESTION_GOALS",
                    },
                    ensure_ascii=False,
                )
            try:
                compose_kwargs: dict[str, Any] = {"allow_llm": allow_llm}
                if runtime.context.budget is not None:
                    compose_kwargs["runtime_budget"] = runtime.context.budget
                answer = runtime_owner.kernel.compose_answer(
                    runtime.context.session.runtime,
                    **compose_kwargs,
                )
            except GroundedRuntimeBudgetExceeded:
                raise
            except RuntimeError as exc:
                message = str(exc)
                incomplete = "EVIDENCE_PORTFOLIO_INCOMPLETE" in message
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE" if incomplete else "BLOCKED",
                        "code": (
                            "EVIDENCE_PORTFOLIO_INCOMPLETE"
                            if incomplete
                            else "ANSWER_COMPOSITION_BLOCKED"
                        ),
                        "message": message[:500],
                        "nextAction": (
                            "CONTINUE_PROGRESSIVE_QUERYING"
                            if incomplete
                            else "STOP"
                        ),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "ANSWER_COMPOSITION_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            state = runtime.context.session.runtime
            return json.dumps(
                {
                    "status": "ANSWERED",
                    "answer": answer,
                    "verifiedQueryArtifactIds": list(
                        state.answer_artifact_ids
                    ),
                },
                ensure_ascii=False,
            )

        @tool("finalize_evidence_collection")
        def finalize_evidence_collection(
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Seal the complete verified portfolio before optional Skill analysis."""

            deep_session = runtime.context.session
            if deep_session.analysis_skill_started:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                    },
                    ensure_ascii=False,
                )
            try:
                coverage = runtime_owner._require_complete_goal_coverage(
                    deep_session
                )
            except GoalCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                deep_session.goal_coverage_result = result
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE",
                        "code": "ORIGINAL_QUESTION_GOALS_UNCOVERED",
                        "missingRequiredGoalIds": result.get(
                            "missingRequiredGoalIds", []
                        ),
                        "issues": result.get("issues", []),
                        "nextAction": "CONTINUE_PROGRESSIVE_QUERYING",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except RuntimeError as exc:
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE",
                        "code": str(exc),
                        "nextAction": "DECLARE_ORIGINAL_QUESTION_GOALS",
                    },
                    ensure_ascii=False,
                )
            try:
                plan, run_result, verified, artifact_ids = (
                    runtime_owner.kernel.verify_portfolio(
                        deep_session.runtime
                    )
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE",
                        "code": "VERIFIED_EVIDENCE_PORTFOLIO_UNAVAILABLE",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "CONTINUE_PROGRESSIVE_QUERYING",
                    },
                    ensure_ascii=False,
                )
            if not verified.passed:
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE",
                        "code": "EVIDENCE_PORTFOLIO_INCOMPLETE",
                        "reason": str(reason or "")[:500],
                        "artifactIds": artifact_ids,
                        "blockingGaps": [
                            item.model_dump(by_alias=True)
                            for item in verified.blocking_gaps
                        ],
                        "nextAction": "CONTINUE_PROGRESSIVE_QUERYING",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            with deep_session.lock:
                deep_session.data_collection_sealed = True
                deep_session.analysis_skill_headers_disclosed = True
                deep_session.runtime.answer_plan = plan.model_copy(deep=True)
                deep_session.runtime.answer_run_result = run_result.model_copy(
                    deep=True
                )
                deep_session.runtime.answer_verified_evidence = (
                    verified.model_copy(deep=True)
                )
                deep_session.runtime.answer_artifact_ids = list(artifact_ids)
            return json.dumps(
                {
                    "status": "EVIDENCE_COLLECTION_SEALED",
                    "reason": str(reason or "")[:500],
                    "verifiedQueryArtifactIds": artifact_ids,
                    "goalCoverage": coverage.model_dump(by_alias=True),
                    "rowCount": len(run_result.merged_query_bundle.rows),
                    "tables": list(run_result.merged_query_bundle.tables),
                    "availableAnalysisSkillHeaders": list(
                        runtime_owner.skill_headers
                    ),
                    "nextAction": "RUN_ONE_MATCHING_SKILL_OR_COMPOSE_ANSWER",
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("run_skill")
        def run_skill(
            skill_name: str,
            objective: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Run one LLM-selected Skill in an isolated subagent/checkpoint."""

            result = runtime_owner._run_isolated_skill(
                runtime.context,
                skill_name=str(skill_name or "").strip(),
                objective=str(objective or "").strip(),
            )
            return json.dumps(result, ensure_ascii=False, default=str)

        @tool("ask_human")
        def ask_human(
            question: str,
            stage: str,
            clarification_type: str,
            options: list[str],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Create a typed human clarification and stop query execution."""

            normalized_type = str(clarification_type or "").strip().upper()
            if normalized_type.startswith(
                (
                    "SYSTEM_",
                    "INTERNAL_",
                    "TOOL_",
                    "COMPILER_",
                    "SEMANTIC_",
                    "CONTRACT_",
                    "SQL_",
                    "EXECUTION_",
                )
            ):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "INTERNAL_FAILURE_IS_NOT_USER_CLARIFICATION",
                        "message": (
                            "ask_human is reserved for genuinely missing user business input; "
                            "use the trusted runtime scope and return an operational failure for internal blockers"
                        ),
                    },
                    ensure_ascii=False,
                )

            request = runtime_owner.kernel.request_clarification(
                runtime.context.session.runtime,
                question,
                stage=stage,
                clarification_type=clarification_type,
                options=options,
            )
            return json.dumps(
                {"status": "CLARIFICATION_REQUIRED", "clarification": request.model_dump(by_alias=True)},
                ensure_ascii=False,
            )

        return [
            declare_original_question_goals,
            retrieve_knowledge,
            propose_grounded_contract,
            prepare_grounded_query_batch,
            submit_grounded_sql_candidate,
            execute_grounded_query,
            execute_grounded_query_batch,
            publish_verified_entity_set,
            finalize_evidence_collection,
            compose_verified_answer,
            run_skill,
            ask_human,
        ]

    def _run_isolated_skill(
        self,
        context: GroundedDeepAgentRunContext,
        *,
        skill_name: str,
        objective: str,
    ) -> dict[str, Any]:
        session = context.session
        state = session.runtime
        normalized_name = _normalized_skill_name(skill_name)
        skill_dir = self._skill_directory(normalized_name)
        if skill_dir is None:
            return {
                "status": "SKILL_NOT_FOUND",
                "skillName": normalized_name,
                "message": "Skill must be selected from the disclosed Skill headers.",
            }
        if not session.data_collection_sealed:
            return {
                "status": "EVIDENCE_COLLECTION_NOT_SEALED",
                "skillName": normalized_name,
                "nextAction": "FINALIZE_EVIDENCE_COLLECTION",
                "message": (
                    "Finish every required data query and seal the verified evidence portfolio before running an analysis Skill."
                ),
            }
        if (
            state.answer_plan is None
            or state.answer_run_result is None
            or state.answer_verified_evidence is None
            or not state.answer_verified_evidence.passed
        ):
            return {
                "status": "VERIFIED_EVIDENCE_REQUIRED",
                "skillName": normalized_name,
                "message": (
                    "Run the grounded query and verification first, then invoke the Skill "
                    "with the verified result artifact."
                ),
            }
        if not session.analysis_skill_headers_disclosed:
            return {
                "status": "SKILL_HEADERS_NOT_DISCLOSED",
                "skillName": normalized_name,
                "message": "Execute and verify the grounded query before selecting an analysis Skill.",
            }
        if session.analysis_skill_started:
            return {
                "status": "SKILL_ALREADY_ATTEMPTED",
                "skillName": normalized_name,
                "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                "message": "An analysis Skill may run only once from immutable verified evidence.",
            }
        if self.checkpointer is None:
            return {
                "status": "SKILL_CHECKPOINT_REQUIRED",
                "skillName": normalized_name,
                "message": "Skill isolation requires an independent checkpoint backend.",
            }

        metadata = _load_skill_frontmatter(skill_dir / "SKILL.md")
        lifecycle_phase = str(
            metadata.get("lifecyclePhase")
            or metadata.get("lifecycle_phase")
            or "post_query_analysis"
        ).strip()
        requires_verified = str(
            metadata.get("requiresVerifiedEvidence")
            or metadata.get("requires_verified_evidence")
            or "true"
        ).strip().lower() not in {"false", "0", "no"}
        output_contract = str(
            metadata.get("outputContract")
            or metadata.get("output_contract")
            or ""
        ).strip()
        if (
            lifecycle_phase != "post_query_analysis"
            or not requires_verified
            or output_contract != "verified_analysis_v1"
        ):
            return {
                "status": "SKILL_LIFECYCLE_UNSUPPORTED",
                "skillName": normalized_name,
                "message": (
                    "The current run_skill boundary only executes post-query analysis "
                    "Skills that require verified evidence."
                ),
            }

        skill_run_id = "skill_%s_%s" % (
            re.sub(r"[^a-z0-9-]+", "-", normalized_name).strip("-") or "run",
            uuid.uuid4().hex[:12],
        )
        skill_thread_id = "%s__%s" % (context.thread_id, skill_run_id)
        workspace = self.skill_run_root / skill_run_id
        try:
            workspace.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            return {
                "status": "SKILL_WORKSPACE_FAILED",
                "skillName": normalized_name,
                "message": "%s:%s"
                % (type(exc).__name__, str(exc)[:400]),
            }
        input_path = workspace / "input.json"
        script_output_path = workspace / "script-output.json"
        result_path = workspace / "result.json"
        checkpoint_ref = {
            "threadId": skill_thread_id,
            "runId": skill_run_id,
            "checkpointNamespace": "deepagent",
        }
        progress: list[dict[str, Any]] = []

        def progress_event(stage: str, status: str, detail: str = "") -> None:
            event = {
                "sequence": len(progress) + 1,
                "stage": stage,
                "status": status,
                "detail": str(detail or "")[:500],
                "skillName": normalized_name,
                "skillRunId": skill_run_id,
                "checkpoint": checkpoint_ref,
            }
            progress.append(event)
            _emit_runtime_listener(
                context.listener,
                "skill.progress",
                "SKILL_RUN",
                event,
            )

        progress_event("matched", "completed", objective)
        skill_payload = self._skill_input_payload(
            session,
            normalized_name,
            objective,
            skill_run_id,
        )
        try:
            input_path.write_text(
                json.dumps(skill_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            return {
                "status": "SKILL_INPUT_ARTIFACT_FAILED",
                "skillName": normalized_name,
                "message": "%s:%s"
                % (type(exc).__name__, str(exc)[:400]),
            }
        progress_event("workspace", "completed", str(workspace))
        with session.lock:
            if session.analysis_skill_started:
                return {
                    "status": "SKILL_ALREADY_ATTEMPTED",
                    "skillName": normalized_name,
                    "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                }
            session.analysis_skill_started = True
        execution_mode = str(
            metadata.get("executionMode")
            or metadata.get("execution_mode")
            or "structured_renderer"
        ).strip()
        script_result: dict[str, Any] = {}
        if execution_mode == "python_script":
            progress_event("script", "started", str(metadata.get("script") or ""))
            script_result = self._execute_declared_skill_script(
                skill_dir,
                metadata,
                input_path,
                script_output_path,
            )
            if not script_result.get("success"):
                progress_event("script", "failed", str(script_result.get("error") or ""))
                failed = {
                    "status": "SKILL_SCRIPT_FAILED",
                    "skillName": normalized_name,
                    "skillRunId": skill_run_id,
                    "checkpoint": checkpoint_ref,
                    "workspace": str(workspace),
                    "progress": progress,
                    "error": str(script_result.get("error") or "skill script failed"),
                }
                result_path.write_text(
                    json.dumps(failed, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                self._record_skill_run(session, failed)
                return failed
            progress_event("script", "completed", str(script_output_path))

        skill_semantic_backend = GroundedSemanticBackend(
            self.semantic_catalog,
            reader_is_core=lambda: False,
        )
        isolated_session = GroundedDeepAgentSession(
            runtime=state.model_copy(deep=True),
            opened_topics=list(session.opened_topics),
        )
        isolated_backend = CompositeBackend(
            default=FilesystemBackend(root_dir=workspace, virtual_mode=True),
            routes={
                "/knowledge/": skill_semantic_backend,
                "/skills/%s/" % normalized_name: FilesystemBackend(
                    root_dir=skill_dir,
                    virtual_mode=True,
                ),
            },
            artifacts_root="/workspace",
        )

        runtime_owner = self

        @tool("retrieve_knowledge")
        def skill_retrieve_knowledge(query: str, reason: str) -> str:
            """Retrieve current Topic knowledge for the isolated Skill only."""

            bundle = runtime_owner.kernel.recall_navigation(
                isolated_session.runtime,
                query=str(query or "").strip(),
            )
            return json.dumps(
                {
                    "status": "OK",
                    "reason": str(reason or "")[:300],
                    "scope": isolated_session.effective_topics(),
                    "recallCandidates": _thin_recall(bundle, limit=8),
                },
                ensure_ascii=False,
            )

        isolated_job = IsolatedSubagentJob(
            job_id=skill_run_id,
            thread_id=skill_thread_id,
            system_prompt=(
                "You are a generic isolated subagent with one mounted Skill resource. "
                "Read the selected SKILL.md and execute its procedure against /input.json "
                "and, when present, /script-output.json. You may read current-Topic "
                "knowledge and call retrieve_knowledge for governed background, but you "
                "may not propose the parent Contract, execute SQL, alter parent evidence, "
                "ask the user, dispatch task, or request that the parent query more data. "
                "Every observed fact must be grounded in the immutable input evidence. "
                "Never replace or extend a governed metric formula. Put measured facts in "
                "observations, governed definitions in semanticDisclosures, calculations "
                "using an already-declared governed formula in derivedFacts, uncertain ideas "
                "in hypotheses, actions in recommendations, and missing evidence in gaps. "
                "Return one JSON object with answerMarkdown, observations, "
                "semanticDisclosures, derivedFacts, hypotheses, recommendations, "
                "evidenceRefs, gaps, and executionConfidence between 0 and 1."
            ),
            user_payload={
                "mountedSkill": "/skills/%s/SKILL.md" % normalized_name,
                "objective": objective,
                "inputArtifact": "/input.json",
                "scriptOutputArtifact": (
                    "/script-output.json" if script_output_path.exists() else ""
                ),
                "resultContract": {
                    "required": [
                        "answerMarkdown",
                        "observations",
                        "semanticDisclosures",
                        "derivedFacts",
                        "hypotheses",
                        "recommendations",
                        "evidenceRefs",
                        "gaps",
                        "executionConfidence",
                    ]
                },
            },
            backend=isolated_backend,
            tools=[skill_retrieve_knowledge],
            # Skill matching already happened in the parent from Header-only
            # metadata. The isolated worker reads the one mounted SKILL.md
            # explicitly; no other Skill directory is visible.
            skills=[],
            middleware=[GroundedCoreToolBoundaryMiddleware(skill_semantic_backend)],
            permissions=[
                FilesystemPermission(
                    operations=["write"],
                    paths=["/knowledge", "/knowledge/**", "/skills", "/skills/**"],
                    mode="deny",
                )
            ],
            subagents=[
                {
                    "name": "general-purpose",
                    "description": "Disabled nested worker inside one isolated job.",
                    "system_prompt": "Do not run; nested task dispatch is disabled.",
                    "tools": [],
                }
            ],
        )
        try:
            with skill_semantic_backend.scope(isolated_session):
                isolated_result = self.subagent_runtime.run(
                    isolated_job,
                    on_progress=progress_event,
                )
        except Exception as exc:
            progress_event(
                "subagent",
                "failed",
                "%s:%s" % (type(exc).__name__, str(exc)[:400]),
            )
            failed = {
                "status": "SKILL_SUBAGENT_FAILED",
                "skillName": normalized_name,
                "skillRunId": skill_run_id,
                "checkpoint": checkpoint_ref,
                "workspace": str(workspace),
                "artifact": str(result_path),
                "progress": progress,
                "error": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
            }
            result_path.write_text(
                json.dumps(failed, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._record_skill_run(session, failed)
            return failed
        checkpoint_ref = dict(isolated_result.checkpoint)
        raw_output = isolated_result.raw_output

        def assess_output(
            raw: str,
            *,
            allow_script_fallback: bool,
        ) -> tuple[dict[str, Any], str, list[str], Any, list[dict[str, Any]]]:
            structured_output = _parse_skill_result(raw)
            if (
                allow_script_fallback
                and not structured_output.get("answerMarkdown")
                and script_result.get("payload")
            ):
                structured_output["answerMarkdown"] = str(
                    (script_result.get("payload") or {}).get("answerMarkdown") or ""
                )
            contract_issues = _skill_output_contract_issues(
                structured_output,
                state.answer_plan or state.active_plan,
            )
            for key in (
                "observations",
                "semanticDisclosures",
                "derivedFacts",
                "hypotheses",
                "recommendations",
                "evidenceRefs",
                "gaps",
            ):
                structured_output.setdefault(key, [])
            structured_output["executionConfidence"] = _confidence(
                structured_output.get("executionConfidence")
            )
            rendered_answer = str(
                structured_output.get("answerMarkdown") or ""
            ).strip()
            if not rendered_answer:
                contract_issues.append(
                    {
                        "code": "ANSWER_MARKDOWN_REQUIRED",
                        "message": "isolated Skill returned no answerMarkdown",
                    }
                )
            selected_artifact_ids = set(state.answer_artifact_ids)
            permitted_refs = {
                ref_id
                for artifact in state.verified_query_ledger
                if artifact.artifact_id in selected_artifact_ids
                for ref_id in artifact.contract.evidence_refs
            }
            untrusted = [
                str(ref_id)
                for ref_id in structured_output.get("evidenceRefs") or []
                if str(ref_id) not in permitted_refs
            ]
            verification = AnswerClaimVerifier().verify(
                state.question,
                state.answer_plan or state.active_plan,
                state.answer_run_result or state.run_result,
                rendered_answer,
                support_context=_skill_claim_support_context(state),
            )
            return (
                structured_output,
                rendered_answer,
                untrusted,
                verification,
                contract_issues,
            )

        structured, answer, untrusted_refs, claim_verification, contract_issues = (
            assess_output(raw_output, allow_script_fallback=True)
        )
        repair_attempted = False
        if untrusted_refs or not claim_verification.passed or contract_issues:
            repair_attempted = True
            progress_event(
                "verification",
                "repairing",
                "isolated output exceeded immutable verified evidence",
            )
            draft_path = workspace / "draft-output.json"
            feedback_path = workspace / "verification-feedback.json"
            draft_path.write_text(raw_output, encoding="utf-8")
            feedback = {
                "contractIssues": contract_issues,
                "untrustedEvidenceRefs": untrusted_refs,
                "unsupportedClaims": [
                    item.model_dump(by_alias=True)
                    for item in claim_verification.unsupported_claims
                ],
                "repairPolicy": (
                    "Revise presentation only from the same immutable /input.json. "
                    "Do not request more data, add metrics, or alter governed formulas."
                ),
            }
            feedback_path.write_text(
                json.dumps(feedback, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            repair_job = replace(
                isolated_job,
                job_id="%s_repair1" % skill_run_id,
                thread_id="%s__repair1" % skill_thread_id,
                system_prompt=(
                    isolated_job.system_prompt
                    + " This is the only permitted repair attempt. Read /draft-output.json "
                    "and /verification-feedback.json, then return a corrected JSON object. "
                    "Use exactly the same immutable evidence and never ask the parent to query."
                ),
                user_payload={
                    **dict(isolated_job.user_payload),
                    "draftArtifact": "/draft-output.json",
                    "verificationFeedbackArtifact": "/verification-feedback.json",
                    "repairAttempt": 1,
                },
            )
            try:
                with skill_semantic_backend.scope(isolated_session):
                    repaired_result = self.subagent_runtime.run(
                        repair_job,
                        on_progress=progress_event,
                    )
                checkpoint_ref = dict(repaired_result.checkpoint)
                structured, answer, untrusted_refs, claim_verification, contract_issues = (
                    assess_output(
                        repaired_result.raw_output,
                        allow_script_fallback=False,
                    )
                )
            except Exception as exc:
                contract_issues.append(
                    {
                        "code": "SKILL_REPAIR_FAILED",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                    }
                )

        if untrusted_refs or not claim_verification.passed or contract_issues:
            progress_event(
                "verification",
                "fallback",
                "Skill repair failed; returning deterministic verified answer",
            )
            fallback_answer = self.kernel.compose_answer(state, allow_llm=False)
            fallback = {
                "status": "SKILL_FALLBACK_ANSWERED",
                "skillName": normalized_name,
                "skillRunId": skill_run_id,
                "checkpoint": checkpoint_ref,
                "workspace": str(workspace),
                "artifact": str(result_path),
                "answerMarkdown": fallback_answer,
                "repairAttempted": repair_attempted,
                "queryMutationAllowed": False,
                "contractIssues": contract_issues,
                "untrustedEvidenceRefs": untrusted_refs,
                "unsupportedClaims": [
                    item.model_dump(by_alias=True)
                    for item in claim_verification.unsupported_claims
                ],
                "progress": progress,
            }
            result_path.write_text(
                json.dumps(fallback, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._record_skill_run(session, fallback)
            return fallback

        progress_event("verification", "completed", "verified evidence only")
        completed = {
            "status": "SKILL_COMPLETED",
            "skillName": normalized_name,
            "skillRunId": skill_run_id,
            "checkpoint": checkpoint_ref,
            "workspace": str(workspace),
            "artifact": str(result_path),
            "answerMarkdown": answer,
            "observations": structured.get("observations") or [],
            "semanticDisclosures": structured.get("semanticDisclosures") or [],
            "derivedFacts": structured.get("derivedFacts") or [],
            "hypotheses": structured.get("hypotheses") or [],
            "recommendations": structured.get("recommendations") or [],
            "evidenceRefs": structured.get("evidenceRefs") or [],
            "gaps": structured.get("gaps") or [],
            "executionConfidence": structured["executionConfidence"],
            "repairAttempted": repair_attempted,
            "queryMutationAllowed": False,
            "progress": progress,
        }
        progress_event("result", "completed", str(result_path))
        completed["progress"] = progress
        result_path.write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._record_skill_run(session, completed)
        with session.lock:
            state.answer = answer
            state.phase = "ANSWERED"
        return completed

    def _skill_directory(self, skill_name: str) -> Optional[Path]:
        if self.skill_root is None or not skill_name:
            return None
        candidate = (self.skill_root / skill_name).resolve()
        if self.skill_root not in candidate.parents or not (candidate / "SKILL.md").is_file():
            return None
        metadata = _load_skill_frontmatter(candidate / "SKILL.md")
        if str(metadata.get("name") or "").strip() != skill_name:
            return None
        return candidate

    @staticmethod
    def _skill_input_payload(
        session: GroundedDeepAgentSession,
        skill_name: str,
        objective: str,
        skill_run_id: str,
    ) -> dict[str, Any]:
        state = session.runtime
        plan = state.answer_plan or state.active_plan
        run_result = state.answer_run_result or state.run_result
        active_contract = state.active_contract
        verified = state.answer_verified_evidence or state.verified_evidence
        selected_artifact_ids = set(state.answer_artifact_ids)
        selected_artifacts = [
            item
            for item in state.verified_query_ledger
            if item.artifact_id in selected_artifact_ids
        ]
        allowed_evidence_refs = list(
            dict.fromkeys(
                ref_id
                for artifact in selected_artifacts
                for ref_id in artifact.contract.evidence_refs
            )
        )
        return {
            "skillName": skill_name,
            "skillRunId": skill_run_id,
            "question": state.question,
            "objective": objective,
            "topics": session.effective_topics(),
            "groundedSummary": {
                "queryShape": (
                    "VERIFIED_EVIDENCE_PORTFOLIO"
                    if len(selected_artifacts) > 1
                    else str(active_contract.query_shape if active_contract else "")
                ),
                "analysisMode": str(active_contract.analysis_mode if active_contract else ""),
                "tables": list(
                    run_result.merged_query_bundle.tables
                    if run_result is not None
                    else []
                ),
                "timeRange": (
                    active_contract.time_range.model_dump(by_alias=True)
                    if active_contract is not None
                    else {}
                ),
                "evidenceRefs": allowed_evidence_refs,
                "verifiedQueryArtifactIds": list(state.answer_artifact_ids),
            },
            "dataRows": (
                list(run_result.merged_query_bundle.rows)[:200]
                if run_result is not None
                else []
            ),
            "metricDisclosures": [
                dict(spec)
                for intent in (plan.intents if plan is not None else [])
                for spec in intent.metric_specs
                if isinstance(spec, dict)
            ],
            "verifiedEvidence": {
                "passed": bool(verified.passed) if verified is not None else False,
                "coveredEvidence": list(verified.covered_evidence) if verified is not None else [],
                "derivedEvidence": list(verified.derived_evidence) if verified is not None else [],
                "requiredDisclosures": list(verified.required_disclosures) if verified is not None else [],
                "blockingGaps": [
                    gap.model_dump(by_alias=True)
                    for gap in (verified.blocking_gaps if verified is not None else [])
                ],
                "warningGaps": [
                    gap.model_dump(by_alias=True)
                    for gap in (verified.warning_gaps if verified is not None else [])
                ],
            },
            "evidenceGaps": [
                gap.model_dump(by_alias=True)
                for gap in (run_result.evidence_gaps[:32] if run_result is not None else [])
            ],
            "allowedEvidenceRefs": list(
                allowed_evidence_refs
            ),
        }

    @staticmethod
    def _execute_declared_skill_script(
        skill_dir: Path,
        metadata: dict[str, str],
        input_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        relative = Path(str(metadata.get("script") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return {"success": False, "error": "invalid Skill script path"}
        script = (skill_dir / relative).resolve()
        if skill_dir not in script.parents or not script.is_file() or script.suffix != ".py":
            return {"success": False, "error": "declared Skill script is unavailable"}
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=str(output_path.parent),
                env={"PYTHONIOENCODING": "utf-8"},
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
        except Exception as exc:
            return {"success": False, "error": "%s:%s" % (type(exc).__name__, str(exc)[:500])}
        if completed.returncode != 0 or not output_path.is_file():
            return {
                "success": False,
                "error": (completed.stderr or completed.stdout or "skill script failed")[:1000],
            }
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"success": False, "error": "invalid script output: %s" % str(exc)[:300]}
        return {"success": True, "payload": payload}

    @staticmethod
    def _record_skill_run(
        session: GroundedDeepAgentSession,
        result: dict[str, Any],
    ) -> None:
        with session.lock:
            session.skill_runs.append(dict(result))
            run_result = (
                session.runtime.answer_run_result
                or session.runtime.run_result
            )
            if run_result is not None:
                run_result.skill_lifecycle_records.append(
                    SkillLifecycleRecord(
                        record_id=str(result.get("skillRunId") or ""),
                        skill_name=str(result.get("skillName") or ""),
                        stage="completed",
                        status=str(result.get("status") or ""),
                        matched_by="core_llm_skill_header",
                        isolated_run_id=str(result.get("skillRunId") or ""),
                        workspace_path=str(result.get("workspace") or ""),
                        progress=[
                            "%s:%s"
                            % (item.get("stage") or "", item.get("status") or "")
                            for item in result.get("progress") or []
                            if isinstance(item, dict)
                        ],
                        summary=str(result.get("answerMarkdown") or result.get("error") or "")[:1000],
                        metadata={
                            "checkpoint": result.get("checkpoint") or {},
                            "artifact": str(result.get("artifact") or ""),
                            "executionConfidence": result.get("executionConfidence"),
                        },
                    )
                )

    @staticmethod
    def _require_complete_goal_coverage(
        session: GroundedDeepAgentSession,
    ) -> Any:
        contract = session.question_goal_contract
        if contract is None:
            raise RuntimeError("ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED")
        declarations: list[VerifiedArtifactGoalCoverage] = []
        for artifact in session.runtime.verified_query_ledger:
            declarations.append(
                declare_verified_artifact_goal_coverage(
                    contract,
                    artifact,
                    session.artifact_goal_ids.get(artifact.artifact_id) or [],
                    evidence_refs=artifact.contract.evidence_refs,
                )
            )
        result = GoalCoverageVerifier().require_complete(
            contract,
            declarations,
        )
        session.goal_coverage_result = result.model_dump(by_alias=True)
        return result

    def run(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: Optional[MerchantInfo] = None,
        access_role: str = "merchant_analyst",
        user_scope: Optional[dict[str, Any]] = None,
        thread_id: str = "",
        run_id: str = "",
        listener: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
    ) -> ChatResponse:
        actual_thread_id = thread_id or "thread_%s" % uuid.uuid4().hex
        actual_run_id = run_id or "run_%s" % uuid.uuid4().hex
        budget = GroundedRuntimeBudget.from_settings(self.settings or object())
        kernel_session: Optional[GroundedRuntimeSession] = None
        session: Optional[GroundedDeepAgentSession] = None
        try:
            with budget.stage("bootstrap.session"):
                kernel_session = self.kernel.new_session(
                    question,
                    merchant_id,
                    merchant=merchant,
                    access_role=access_role,
                    user_scope=user_scope,
                )
            with budget.stage("routing.topic"):
                self.kernel.route_topic(kernel_session)
            try:
                with budget.stage("recall.initial"):
                    self.kernel.recall_navigation(kernel_session)
            except GroundedRuntimeBudgetExceeded:
                raise
            except Exception as exc:
                kernel_session.recall = RecallBundle()
                kernel_session.phase = "RECALL_NAVIGATION_DEGRADED"
                kernel_session.events.append(
                    GroundedRuntimeEvent(
                        sequence=len(kernel_session.events) + 1,
                        stage="recall_navigation",
                        status="DEGRADED",
                        detail="%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                    )
                )
            session = GroundedDeepAgentSession(runtime=kernel_session)
            context = GroundedDeepAgentRunContext(
                thread_id=actual_thread_id,
                run_id=actual_run_id,
                session=session,
                budget=budget,
                listener=listener,
            )
            with budget.stage("bootstrap.context"):
                first_context = self._initial_context(session)
            if self.checkpoint_config_factory is not None:
                config = self.checkpoint_config_factory(actual_thread_id, actual_run_id)
            elif self.checkpointer is not None:
                config = {
                    "configurable": {
                        "thread_id": actual_thread_id,
                        "run_id": actual_run_id,
                    }
                }
            else:
                config = None
            with self.knowledge_backend.scope(session):
                with budget.stage("core.react_loop"):
                    self.deep_agent_graph.invoke(
                        {"messages": [{"role": "user", "content": first_context}]},
                        config=config,
                        context=context,
                    )
        except GroundedRuntimeBudgetExceeded as exc:
            failure = {
                "code": "GROUNDED_RUNTIME_BUDGET_EXHAUSTED",
                "message": str(exc),
                "breaches": list(exc.breaches),
                "report": dict(exc.report),
            }
            if session is None and kernel_session is not None:
                session = GroundedDeepAgentSession(runtime=kernel_session)
            report = budget.finish()
            if session is None:
                _emit_runtime_listener(
                    listener,
                    "runtime.budget_exhausted",
                    "GROUNDED_CORE",
                    failure,
                )
                return ChatResponse(
                    answer=(
                        "本次查数未能在运行预算内完成。系统没有把未完成或未验证的结果当作最终答案；"
                        "请缩小查询范围，或稍后重试。"
                    ),
                    debug_trace={
                        "harness": {
                            "runtime": "grounded_deepagent",
                            "threadId": actual_thread_id,
                            "runId": actual_run_id,
                            "legacyFallbackUsed": False,
                            "runtimeBudget": report,
                            "goalCoverage": {},
                            "operationalFailure": failure,
                        }
                    },
                )
            session.operational_failure = {
                **failure,
                "report": report,
            }
            session.runtime.phase = "BUDGET_EXHAUSTED"
            session.runtime_budget_report = report
            _emit_runtime_listener(
                listener,
                "runtime.budget_exhausted",
                "GROUNDED_CORE",
                session.operational_failure,
            )
            return self._governed_response(
                session,
                actual_thread_id,
                actual_run_id,
            )
        finally:
            if session is not None and not session.runtime_budget_report:
                session.runtime_budget_report = budget.finish()
        assert session is not None
        return self._governed_response(session, actual_thread_id, actual_run_id)

    def _initial_context(self, session: GroundedDeepAgentSession) -> str:
        manifests: list[dict[str, Any]] = []
        for topic_name in session.runtime.workspace_topics:
            result = self.semantic_catalog.read(
                path="topics/%s/manifest.json" % topic_name,
                max_chars=80_000,
                offset=0,
            )
            if not isinstance(result, dict) or not result.get("success"):
                raise RuntimeError(
                    "Grounded bootstrap failed: Topic L0 manifest unavailable for %s"
                    % topic_name
                )
            manifests.append(
                {
                    "topic": topic_name,
                    "refId": str(result.get("refId") or ""),
                    "path": str(result.get("path") or ""),
                    "content": str(result.get("content") or ""),
                }
            )
        if not manifests:
            raise RuntimeError("Grounded bootstrap failed: no routed Topic L0 manifest")
        payload = {
            "question": session.runtime.question,
            "userInputRequirements": {
                "explicitTimeExpression": has_explicit_time_expression(
                    session.runtime.question
                ),
                "timeRequirementPolicy": {
                    "analyticalOrDetailList": "EXPLICIT_TIME_REQUIRED",
                    "resolvedEntityLookup": "USE_FIELD_LOOKUP_TIME_POLICY",
                    "decisionAuthority": "PROGRESSIVELY_READ_SEMANTIC_ASSET",
                    "rule": (
                        "Do not create field-name or business-domain exceptions. A no-time entity "
                        "lookup is allowed only when a selected semantic entity field explicitly "
                        "declares a compatible lookupTimePolicy."
                    ),
                },
            },
            "trustedExecutionScope": {
                "merchantScopeBound": bool(session.runtime.merchant_id),
                "merchantId": session.runtime.merchant_id,
                "merchantName": str(session.runtime.merchant.merchant_name or ""),
                "accessRole": session.runtime.access_role,
                "authorizedStoreIds": [
                    str(item)
                    for item in (
                        session.runtime.user_scope.get("storeIds")
                        or session.runtime.user_scope.get("store_ids")
                        or []
                    )
                    if str(item or "").strip()
                ],
                "tenantFilterPolicy": (
                    "The SQL executor automatically binds the published merchantFilterColumn "
                    "to this merchantId. Do not expose it as a business dimension."
                ),
            },
            "topicRouting": session.runtime.routing.model_dump(by_alias=True),
            "topicL0Manifests": manifests,
            "thinRecallCandidates": _thin_recall(session.runtime.recall, limit=8),
            "originalQuestionGoalPolicy": {
                "requiredBeforeQuery": True,
                "immutableAfterQueryStart": True,
                "goalKinds": [
                    "METRIC",
                    "DIMENSION",
                    "TIME_WINDOW",
                    "COMPARISON",
                    "ENTITY",
                    "DEPENDENCY",
                ],
                "queryAssignmentRequired": True,
                "finalizationRequiresVerifiedCoverage": True,
                "parallelRule": (
                    "Only goals without dependency/upstream entity bindings may share a batch."
                ),
            },
            "analysisSkillPolicy": {
                "lifecyclePhase": "post_query_analysis",
                "requiresGroundedContract": True,
                "requiresExecutedQuery": True,
                "requiresVerifiedEvidence": True,
                "mayInfluenceSemanticBindings": False,
                "mayExecuteSql": False,
                "headersDisclosedAfterEvidenceFinalizationOnly": True,
            },
            "instructions": (
                "Use Topic/recall only for initial navigation. trustedExecutionScope is authoritative. "
                "Progressively read exact files under /knowledge, then use the typed Grounded tools. "
                "No analysis Skill Header is available while data collection remains open. "
                "Only finalize_evidence_collection may disclose matching headers; the parent Core "
                "never has access to full SKILL.md bodies."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _governed_response(
        session: GroundedDeepAgentSession,
        thread_id: str,
        run_id: str,
    ) -> ChatResponse:
        state = session.runtime
        trace = {
            "harness": {
                "runtime": "grounded_deepagent",
                "threadId": thread_id,
                "runId": run_id,
                "activeAttemptId": state.active_attempt_id,
                "activeGeneration": state.active_generation,
                "legacyFallbackUsed": False,
                "verifiedQueryArtifactIds": list(state.answer_artifact_ids),
                "verifiedQueryArtifactCount": len(state.answer_artifact_ids),
                "skillRuns": [
                    {
                        "skillName": item.get("skillName"),
                        "skillRunId": item.get("skillRunId"),
                        "status": item.get("status"),
                        "checkpoint": item.get("checkpoint"),
                        "artifact": item.get("artifact"),
                        "executionConfidence": item.get("executionConfidence"),
                    }
                    for item in session.skill_runs
                ],
                "runtimeBudget": dict(session.runtime_budget_report),
                "goalCoverage": dict(session.goal_coverage_result),
            }
        }
        if session.operational_failure:
            trace["harness"]["operationalFailure"] = dict(
                session.operational_failure
            )
            return ChatResponse(
                answer=(
                    "本次查数未能在运行预算内完成。系统没有把未完成或未验证的结果当作最终答案；"
                    "请缩小查询范围，或稍后重试。"
                ),
                category_name=state.routing.display_summary(),
                debug_trace=trace,
            )
        if state.clarification is not None:
            return ChatResponse(
                answer=state.clarification.question,
                category_name=state.routing.display_summary(),
                clarification=state.clarification,
                debug_trace=trace,
            )
        if state.answer:
            display_run = state.answer_run_result or state.run_result
            rows = (
                list(display_run.merged_query_bundle.rows)
                if display_run is not None
                else []
            )
            tables = (
                list(display_run.merged_query_bundle.tables)
                if display_run is not None
                else []
            )
            sections = [
                {
                    "title": (
                        artifact.contract.query_shape
                        or "Verified query evidence"
                    ),
                    "resultRole": "verified_query_artifact",
                    "dorisTables": list(
                        artifact.run_result.merged_query_bundle.tables
                    ),
                    "dataRows": [
                        {
                            **dict(row),
                            "__evidenceArtifactId": artifact.artifact_id,
                        }
                        for row in artifact.run_result.merged_query_bundle.rows
                    ],
                    "offloaded": bool(
                        artifact.run_result.merged_query_bundle.offloaded_files
                    ),
                    "offloadedFiles": list(
                        artifact.run_result.merged_query_bundle.offloaded_files
                    ),
                    "originalRowCount": (
                        artifact.run_result.merged_query_bundle.effective_row_count()
                    ),
                    "resultSummary": (
                        "artifact=%s; verified=true"
                        % artifact.artifact_id
                    ),
                }
                for artifact in state.verified_query_ledger
                if artifact.artifact_id in set(state.answer_artifact_ids)
            ]
            return ChatResponse(
                answer=state.answer,
                category_name=state.routing.display_summary(),
                doris_tables=tables,
                data_rows=rows,
                data_sections=sections,
                debug_trace=trace,
            )
        raise RuntimeError(
            "Grounded DeepAgent Core ended without verified answer or typed clarification"
        )


def _grounded_contract_sql_obligations(contract: Any) -> dict[str, Any]:
    """Expose the normalized SQL obligations Core must implement exactly."""

    required_outputs = list(
        dict.fromkeys(
            [
                *[str(item.metric_key or "") for item in contract.metrics],
                *[
                    str(item.output_alias or item.column or "")
                    for item in contract.selected_fields
                ],
                *[
                    str(item.column or "")
                    for item in contract.dimensions
                    if item.usage == "group_by"
                ],
            ]
        )
    )
    return {
        "requiredFinalOutputAliases": [
            item for item in required_outputs if item
        ],
        "tables": [
            {
                "table": item.table,
                "timeColumn": item.time_column,
                "tenantColumn": item.merchant_filter_column,
                "instruction": "tenantColumn may be used only as a governed join key; do not compare it to a literal",
            }
            for item in contract.tables
        ],
        "metrics": [
            {
                "outputAlias": item.metric_key,
                "semanticRefId": item.semantic_ref_id,
                "table": item.table,
                "formula": item.formula,
                "sourceColumns": list(item.source_columns),
                "timeColumn": item.time_column,
                "timeSemantics": dict(item.time_semantics),
            }
            for item in contract.metrics
        ],
        "dimensions": [
            {
                "outputAlias": item.column,
                "semanticRefId": item.semantic_ref_id,
                "table": item.table,
                "column": item.column,
                "usage": item.usage,
            }
            for item in contract.dimensions
        ],
        "selectedFields": [
            {
                "outputAlias": item.output_alias or item.column,
                "semanticRefId": item.semantic_ref_id,
                "table": item.table,
                "column": item.column,
            }
            for item in contract.selected_fields
        ],
        "entityFilters": _core_visible_entity_filter_obligations(contract),
        "relationships": [
            {
                "semanticRefId": item.semantic_ref_id,
                "leftTable": item.left_table,
                "rightTable": item.right_table,
                "joinType": item.join_type,
                "keys": [list(pair) for pair in item.keys],
                "cardinality": item.cardinality,
                "fanoutPolicy": item.fanout_policy,
                "grain": item.grain,
            }
            for item in contract.relationships
        ],
        "timeRange": contract.time_range.model_dump(by_alias=True),
    }


def _core_visible_binding_hints(contract: Any) -> dict[str, Any]:
    """Keep kernel-owned upstream values out of the parent Core context."""

    upstream_refs = {
        str(item.target_field_ref or "")
        for item in contract.upstream_entity_bindings
    }
    hints = contract.binding_hints.model_copy(
        update={
            "entity_filters": [
                item
                for item in contract.binding_hints.entity_filters
                if item.field_ref not in upstream_refs
            ]
        },
        deep=True,
    )
    return hints.model_dump(by_alias=True)


def _core_visible_entity_filter_obligations(contract: Any) -> list[dict[str, Any]]:
    upstream_by_ref = {
        str(item.target_field_ref or ""): item
        for item in contract.upstream_entity_bindings
    }
    result: list[dict[str, Any]] = []
    for item in contract.entity_filters:
        upstream = upstream_by_ref.get(item.semantic_ref_id)
        if upstream is None:
            result.append(
                {
                    "semanticRefId": item.semantic_ref_id,
                    "table": item.table,
                    "column": item.column,
                    "operator": item.operator,
                    "literalValue": item.literal_value,
                    "runtimeInjected": False,
                }
            )
            continue
        result.append(
            {
                "semanticRefId": item.semantic_ref_id,
                "table": item.table,
                "column": item.column,
                "operator": item.operator,
                "runtimeInjected": True,
                "entitySetArtifactId": upstream.entity_set_artifact_id,
                "valueCount": upstream.value_count,
                "valuesHash": upstream.values_hash,
                "instruction": "Do not author this predicate; trusted execution injects it.",
            }
        )
    return result


def _thin_recall(bundle: RecallBundle, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bundle.items:
        metadata = dict(item.metadata or {})
        kind = str(metadata.get("semanticKind") or item.source_type or "").upper()
        raw_ref_id = str(metadata.get("semanticRefId") or "").strip()
        if not raw_ref_id and str(item.doc_id or "").startswith("semantic:"):
            raw_ref_id = str(item.doc_id or "").strip()
        ref_id = _canonical_progressive_ref(raw_ref_id)
        if not ref_id.startswith("semantic:"):
            continue
        inline_only = kind == "GOVERNED_RULE"
        path = ""
        if not inline_only:
            path = _canonical_recall_path(
                ref_id,
                str(metadata.get("semanticPath") or metadata.get("path") or ""),
            )
            if not _is_safe_semantic_path(path):
                continue
        if "asset.json" in path or kind == "TABLE_ASSET":
            continue
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        result.append(
            {
                "refId": ref_id,
                "path": path,
                "kind": kind,
                "topic": item.topic,
                "table": item.table,
                "title": item.title,
                "snippet": item.content[:600],
                "score": float(item.fusion_score or 0.0),
                "navigationMode": "INLINE_ONLY" if inline_only else "READ_FILE",
                "bindingEligible": not inline_only,
            }
        )
        if len(result) >= max(1, int(limit or 1)):
            break
    return result


def _is_safe_semantic_path(path: str) -> bool:
    candidate = str(path or "").strip().lstrip("/")
    if not candidate or ".." in candidate.split("/"):
        return False
    if candidate.startswith(("Users/", "private/", "var/", "tmp/")):
        return False
    return candidate.startswith("topics/") and "asset.json" not in candidate


def _canonical_progressive_ref(ref_id: str) -> str:
    value = str(ref_id or "").strip()
    if value.startswith("semantic:") and value.endswith(":asset"):
        value = value[: -len(":asset")] + ":detail"
    if ":column:" in value:
        value = value.replace(":column:", ":field:", 1)
    return value


def _canonical_binding_hints(hints: GroundedBindingHints) -> GroundedBindingHints:
    return hints.model_copy(
        update={
            "table_refs": [_canonical_progressive_ref(item) for item in hints.table_refs],
            "metric_refs": [_canonical_progressive_ref(item) for item in hints.metric_refs],
            "field_aggregations": [
                item.model_copy(
                    update={"field_ref": _canonical_progressive_ref(item.field_ref)}
                )
                for item in hints.field_aggregations
            ],
            "dimension_refs": [
                _canonical_progressive_ref(item) for item in hints.dimension_refs
            ],
            "selected_fields": [
                item.model_copy(
                    update={"field_ref": _canonical_progressive_ref(item.field_ref)}
                )
                for item in hints.selected_fields
            ],
            "entity_filters": [
                item.model_copy(
                    update={"field_ref": _canonical_progressive_ref(item.field_ref)}
                )
                for item in hints.entity_filters
            ],
            "upstream_entity_bindings": [
                item.model_copy(
                    update={
                        "target_field_ref": _canonical_progressive_ref(
                            item.target_field_ref
                        )
                    }
                )
                for item in hints.upstream_entity_bindings
            ],
            "group_by_ref": _canonical_progressive_ref(hints.group_by_ref),
            "label_refs": {
                _canonical_progressive_ref(key): value
                for key, value in hints.label_refs.items()
            },
            "relationship_refs": [
                _canonical_progressive_ref(item) for item in hints.relationship_refs
            ],
            "ranking": hints.ranking.model_copy(
                update={
                    "metric_ref": _canonical_progressive_ref(
                        hints.ranking.metric_ref
                    )
                }
            ),
        }
    )


def _canonical_recall_path(ref_id: str, path: str) -> str:
    """Normalize legacy recall fragments into safe progressive file paths."""

    value = _canonical_progressive_ref(ref_id)
    parts = value.split(":")
    canonical = ""
    if len(parts) >= 3 and parts[0] == "semantic":
        topic = parts[1]
        if parts[2] in {"relationships", "relationship_index"}:
            canonical = "topics/%s/relationships/index.json" % topic
        elif parts[2] == "relationship" and len(parts) >= 4:
            canonical = "topics/%s/relationships/%s.json" % (
                topic,
                ":".join(parts[3:]),
            )
        elif len(parts) >= 5:
            table = parts[2]
            kind = parts[3]
            key = ":".join(parts[4:])
            if kind == "metric" and key:
                canonical = "topics/%s/tables/%s/metrics/%s.json" % (
                    topic,
                    table,
                    key,
                )
            elif kind in {"field", "column"} and key:
                canonical = "topics/%s/tables/%s/columns/%s.json" % (
                    topic,
                    table,
                    key,
                )
        elif len(parts) >= 4 and parts[3] == "detail":
            canonical = "topics/%s/tables/%s/detail.json" % (topic, parts[2])
    if canonical:
        return canonical
    candidate = str(path or "").strip().lstrip("/")
    if not candidate or "asset.json" in candidate or candidate.startswith(("Users/", "private/", "var/")):
        return ""
    return candidate


def _nested_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _nested_values(item)
    else:
        yield value


def _deepagent_reader_is_core() -> bool:
    """Fail closed for filesystem reads originating from a subagent graph."""

    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return False
    configurable = dict(config.get("configurable") or {})
    metadata = dict(config.get("metadata") or {})
    checkpoint_ns = str(
        configurable.get("checkpoint_ns")
        or configurable.get("checkpointNamespace")
        or ""
    ).lower()
    identity_text = json.dumps(
        {"configurable": configurable, "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).lower()
    if "general-purpose" in identity_text or "subagent" in identity_text:
        return False
    # LangGraph nested/subgraph executions receive a checkpoint namespace;
    # the root Core invocation does not. Unknown nested identities never gain
    # executable evidence authority.
    if checkpoint_ns and checkpoint_ns not in {"deepagent", "grounded_deepagent"}:
        return False
    return True


def _normalized_skill_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", str(value or "").strip().lower())
    return re.sub(r"-+", "-", normalized).strip("-")


def _load_skill_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    result: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        result[key.strip()] = raw.strip().strip('"').strip("'")
    return result


def _parse_skill_result(raw_output: str) -> dict[str, Any]:
    text = str(raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return dict(parsed)
    return {"answerMarkdown": text} if text else {}


def _skill_claim_support_context(state: GroundedRuntimeSession) -> str:
    contract = state.active_contract
    plan = state.answer_plan or state.active_plan
    return json.dumps(
        {
            "merchantId": state.merchant_id,
            "verifiedQueryArtifactIds": list(state.answer_artifact_ids),
            "timeRange": (
                contract.time_range.model_dump(by_alias=True)
                if contract is not None
                else {}
            ),
            "metricSpecs": [
                dict(spec)
                for intent in (plan.intents if plan is not None else [])
                for spec in intent.metric_specs
                if isinstance(spec, dict)
            ],
        },
        ensure_ascii=False,
        default=str,
    )


def _skill_output_contract_issues(
    structured: dict[str, Any],
    plan: Any,
) -> list[dict[str, Any]]:
    required = {
        "answerMarkdown",
        "observations",
        "semanticDisclosures",
        "derivedFacts",
        "hypotheses",
        "recommendations",
        "evidenceRefs",
        "gaps",
        "executionConfidence",
    }
    issues = [
        {
            "code": "SKILL_RESULT_FIELD_REQUIRED",
            "field": field_name,
            "message": "Structured analysis Skill output is missing %s" % field_name,
        }
        for field_name in sorted(required - set(structured))
    ]
    specs = [
        dict(spec)
        for intent in (getattr(plan, "intents", []) or [])
        for spec in getattr(intent, "metric_specs", []) or []
        if isinstance(spec, dict)
    ]
    formula_by_ref = {
        str(spec.get("semanticRefId") or ""): _normalized_formula(
            spec.get("metricFormula") or ""
        )
        for spec in specs
        if str(spec.get("semanticRefId") or "")
        and str(spec.get("metricFormula") or "").strip()
    }
    allowed_formulas = {formula for formula in formula_by_ref.values() if formula}

    for item in structured.get("semanticDisclosures") or []:
        if not isinstance(item, dict):
            issues.append(
                {
                    "code": "SEMANTIC_DISCLOSURE_INVALID",
                    "message": "semanticDisclosures entries must be objects",
                }
            )
            continue
        ref_id = str(
            item.get("metricRef")
            or item.get("semanticRefId")
            or item.get("evidenceRef")
            or ""
        )
        disclosure_text = str(item.get("definition") or "")
        raw_formula = str(item.get("formula") or "")
        if not raw_formula and re.search(
            r"\b(?:SUM|COUNT|AVG|MIN|MAX|NULLIF|COALESCE)\s*\(",
            disclosure_text,
            flags=re.I,
        ):
            raw_formula = disclosure_text
        formula = _normalized_formula(raw_formula)
        if ref_id not in formula_by_ref:
            issues.append(
                {
                    "code": "SEMANTIC_DISCLOSURE_REF_UNTRUSTED",
                    "refId": ref_id,
                    "message": "Semantic disclosure must reference a verified metricSpec",
                }
            )
        elif formula and formula != formula_by_ref[ref_id]:
            issues.append(
                {
                    "code": "GOVERNED_FORMULA_DRIFT",
                    "refId": ref_id,
                    "message": "Skill changed the governed metric formula",
                }
            )

    for item in structured.get("derivedFacts") or []:
        if not isinstance(item, dict):
            issues.append(
                {
                    "code": "DERIVED_FACT_INVALID",
                    "message": "derivedFacts entries must be objects",
                }
            )
            continue
        formula = _normalized_formula(item.get("formula") or "")
        if not formula or formula not in allowed_formulas:
            issues.append(
                {
                    "code": "UNGOVERNED_DERIVED_FORMULA",
                    "message": "Derived facts may use only a formula declared by verified metricSpecs",
                }
            )

    for raw_formula in re.findall(r"`([^`]+)`", str(structured.get("answerMarkdown") or "")):
        if not re.search(r"\b(?:SUM|COUNT|AVG|MIN|MAX|NULLIF|COALESCE)\s*\(", raw_formula, flags=re.I):
            continue
        normalized = _normalized_formula(raw_formula)
        if normalized and normalized not in allowed_formulas:
            issues.append(
                {
                    "code": "GOVERNED_FORMULA_DRIFT",
                    "formula": raw_formula,
                    "message": "Answer Markdown contains a formula not declared by verified metricSpecs",
                }
            )
    return issues


def _normalized_formula(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip("`").upper()


def _confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.5
    return max(0.0, min(score, 1.0))


def _emit_runtime_listener(
    listener: Optional[Callable[[str, str, dict[str, Any]], None]],
    event_type: str,
    node: str,
    payload: dict[str, Any],
) -> None:
    if listener is None:
        return
    try:
        listener(event_type, node, dict(payload))
    except Exception:
        return
