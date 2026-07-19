from __future__ import annotations

import fnmatch
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

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
    AgentRunResult,
    APIModel,
    ChatContext,
    ChatResponse,
    ClarificationRequest,
    ConversationMessage,
    DataSnapshotContract,
    MerchantInfo,
    QueryBundle,
    RecallBundle,
    SkillLifecycleRecord,
    VerifiedEvidence,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier
from merchant_ai.services.assets import normalize_semantic_path
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeEvent,
    GroundedRuntimeSession,
    GroundedVerifiedEntitySet,
    GroundedVerifiedQueryArtifact,
    verified_query_artifact_integrity_valid,
)
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
)
from merchant_ai.services.grounded_goal_contract import (
    GoalCoverageBlocked,
    GoalCoverageVerifier,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    VerifiedArtifactGoalCoverage,
    canonical_goal_id,
    declare_verified_artifact_goal_coverage,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
    required_goal_ids,
)
from merchant_ai.services.grounded_answer_coverage import (
    AnswerCoverageBlocked,
    AnswerCoverageVerifier,
    answer_attestation_matches,
    render_verified_query_goal_sections,
    render_verified_rule_goal_bindings,
)
from merchant_ai.services.grounded_analysis_artifact import (
    GroundedDerivedAnalysisArtifact,
    GroundedRunSkillAnalysisPublicationRequest,
    build_grounded_analysis_skill_input,
    grounded_analysis_goal_coverage,
    publish_grounded_analysis_from_skill,
    render_grounded_analysis_artifact,
    verify_grounded_analysis_data_input_coverage,
)
from merchant_ai.services.grounded_goal_proofs import (
    derive_query_artifact_goal_resolutions,
)
from merchant_ai.services.grounded_rule_artifact import (
    GroundedVerifiedRuleArtifact,
    build_verified_rule_artifact,
    render_verified_rule_answer,
    verified_rule_candidate_refs,
)
from merchant_ai.services.time_semantics import has_explicit_time_expression
from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedQueryContract,
    GroundedReferenceScopeBinding,
)
from merchant_ai.services.grounded_query_branches import (
    GroundedBranchBudget,
    GroundedBranchBudgetExceeded,
    GroundedBranchBudgetLimits,
    GroundedBranchPrepareSpec,
    GroundedQueryBranchContext,
    GroundedQueryBranchSpec,
    GroundedSemanticReadLedger,
)
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionEdgeSpec,
    GroundedExecutionGraphProposal,
    GroundedExecutionGraphReplanEvidence,
    GroundedExecutionGraphReceipt,
    GroundedExecutionGraphRevisionProposal,
    GroundedExecutionGraphNodeRuntimeState,
    GroundedExecutionNodeSpec,
    build_grounded_execution_graph_replan_evidence,
    build_grounded_execution_graph_receipt,
    discovery_evidence_snapshot_fingerprint,
    grounded_execution_graph_fingerprint,
    grounded_execution_graph_replan_evidence_set_fingerprint,
    validate_grounded_execution_graph,
    validate_grounded_execution_graph_revision,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphEdge,
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    PopulationPreExecutionNodeReference,
    PopulationPreExecutionReference,
    population_pre_execution_reference_valid,
)
from merchant_ai.services.grounded_graph_revision_journal import (
    GroundedGraphRevisionBaseBranchCheckpoint,
    GroundedGraphRevisionBaseSessionCheckpoint,
    GroundedGraphRevisionJournalError,
    GroundedGraphRevisionRecoveryPayload,
    GroundedGraphRevisionTransactionJournal,
    build_grounded_graph_revision_recovery_payload,
    seal_grounded_graph_revision_base_session_checkpoint,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_semantic_activation import (
    GroundedSemanticActivationSeal,
    semantic_activation_seal_valid,
)
from merchant_ai.services.data_snapshot_contract import (
    derive_multi_query_snapshot_requirement,
    validate_query_bundle_snapshots,
)
from merchant_ai.services.grounded_exploration_coordinator import (
    GroundedExplorationAssignmentSpec,
    GroundedExplorationCoordinator,
    GroundedExplorationCoordinatorError,
    GroundedExplorationCoordinatorState,
    GroundedExplorationScopeAuthority,
    InMemoryVerifiedExplorationArtifactCatalog,
    IsolatedGroundedExplorationWorker,
    VerifiedExplorationObservation,
    VerifiedExplorationSourceView,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.grounded_subagent_runtime import (
    IsolatedSubagentJob,
    IsolatedSubagentRuntime,
)
from merchant_ai.services.grounded_conversation_state import (
    GROUNDED_CONVERSATION_STATE_VERSION,
    GroundedConversationResolution,
    GroundedConversationStateCorruptError,
    grounded_conversation_principal_fingerprint,
    resolve_grounded_conversation_turn,
)
from merchant_ai.services.grounded_conversation_online_authority import (
    GroundedConversationOnlineAuthorityFacade,
)
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
    GroundedContextWorkspaceError,
    grounded_context_owner_fingerprint,
)
from merchant_ai.services.grounded_context_compaction import (
    ProviderAwareContextTokenCounter,
    build_grounded_model_recovery_message,
    build_grounded_recovery_payload,
    compact_summary_to_reference_only,
    persist_grounded_recovery_payload,
)
from merchant_ai.services.context_filesystem import (
    ContextPathOutsideRootError,
    resolve_context_path,
)
from merchant_ai.services.sandbox import (
    MerchantAnalysisSandbox,
    SandboxArtifactAccess,
)
from merchant_ai.services.grounded_skill_artifact_access import (
    GroundedSkillArtifactAccessBundle,
    GroundedSkillArtifactAccessError,
    build_grounded_skill_artifact_access,
)
from merchant_ai.services.authorization_policy import (
    load_authorization_policy,
)


_DEFAULT_GROUNDED_ACCESS_ROLE = load_authorization_policy().default_access_role


class _GroundedFileInfoMapping(dict[str, Any]):
    """Mapping for deepagents plus attribute access for older adapters."""

    @property
    def path(self) -> str:
        return str(self.get("path") or "")

    @property
    def is_dir(self) -> bool:
        return bool(self.get("is_dir"))

    @property
    def size(self) -> int:
        return int(self.get("size") or 0)

    @property
    def modified_at(self) -> str:
        return str(self.get("modified_at") or "")


def _grounded_file_info(
    *,
    path: str,
    is_dir: bool,
    size: int,
    modified_at: str = "",
) -> Any:
    observed = FileInfo(
        path=path,
        is_dir=is_dir,
        size=size,
        modified_at=modified_at,
    )
    if isinstance(observed, Mapping):
        return _GroundedFileInfoMapping(dict(observed))
    return observed


_SEMANTIC_SCOPE: ContextVar[Optional["GroundedDeepAgentSession"]] = ContextVar(
    "grounded_deep_agent_semantic_scope",
    default=None,
)


class GroundedParallelQuerySpec(GroundedBranchPrepareSpec):
    """V1/V2 preparation input for one independent query branch."""


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
        normalized = list(dict.fromkeys(canonical_goal_id(item) for item in goal_ids))
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


def _required_goal_ids_for_kind(
    session: "GroundedDeepAgentSession",
    kind: str,
) -> list[str]:
    contract = session.question_goal_contract
    if contract is None:
        return []
    required = set(required_goal_ids(contract))
    normalized_kind = str(kind or "").strip().upper()
    return [
        goal.goal_id
        for goal in contract.goals
        if goal.goal_id in required and str(goal.kind or "").strip().upper() == normalized_kind
    ]


def _required_non_rule_goal_ids(
    session: "GroundedDeepAgentSession",
) -> list[str]:
    contract = session.question_goal_contract
    if contract is None:
        return []
    required = set(required_goal_ids(contract))
    return [
        goal.goal_id
        for goal in contract.goals
        if goal.goal_id in required and str(goal.kind or "").strip().upper() != "RULE"
    ]


def _required_goals_are_rule_only(
    session: "GroundedDeepAgentSession",
) -> bool:
    return bool(_required_goal_ids_for_kind(session, "RULE") and not _required_non_rule_goal_ids(session))


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
        # Goal dependencies express answer/semantic coverage and do not by
        # themselves require a serial query. Only a typed runtime artifact
        # input creates an execution edge.
        if str(getattr(goal, "kind", "")) != "DEPENDENCY":
            if isinstance(goal, RankingQuestionGoal) and goal.population_scope == "VERIFIED_ENTITY_SET":
                for upstream_goal_id in goal.population_goal_ids:
                    add_edge(
                        upstream_goal_id=upstream_goal_id,
                        downstream_goal_id=goal.goal_id,
                        relation_type="POPULATION_ENTITY_SET",
                        declared_by_goal_id=goal.goal_id,
                    )
            continue
        dependency_type = str(getattr(goal, "dependency_type", "") or "").strip().upper()
        artifact_kind = str(getattr(goal, "artifact_kind", "") or "").strip().upper()
        if dependency_type in {"CONTRACT_SCOPE", "PREDICATE_SCOPE"}:
            continue
        if not (
            artifact_kind
            in {
                "ENTITY_SET",
                "RESULT_ARTIFACT",
                "VERIFIED_ENTITY_SET",
                "VERIFIED_RESULT_ARTIFACT",
            }
            or dependency_type in {"ENTITY_CHAIN", "RESULT_CHAIN"}
        ):
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
        pending: list[tuple[str, list[dict[str, Any]]]] = [(upstream_goal_id, [])]
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

    goal_order = {goal.goal_id: index for index, goal in enumerate(contract.goals)}
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
            path_goal_ids.extend(str(edge["downstreamGoalId"]) for edge in path_edges)
            first_edge = path_edges[0]
            if direct and first_edge["relationType"] == "DEPENDS_ON_GOAL_IDS":
                code = "BATCH_GOAL_DEPENDS_ON_EDGE"
            elif direct and first_edge["relationType"] == "DEPENDENCY_GOAL":
                code = "BATCH_DEPENDENCY_GOAL_EDGE"
            elif direct and first_edge["relationType"] == "POPULATION_ENTITY_SET":
                code = "BATCH_POPULATION_ENTITY_SET_EDGE"
            else:
                code = "BATCH_TRANSITIVE_GOAL_DEPENDENCY_PATH"
            issue: dict[str, Any] = {
                "code": code,
                "upstreamGoalId": upstream_goal_id,
                "downstreamGoalId": downstream_goal_id,
                "upstreamQueryIds": list(query_ids_by_goal_id[upstream_goal_id]),
                "downstreamQueryIds": list(query_ids_by_goal_id[downstream_goal_id]),
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
                issue["dependencyGoalQueryIds"] = list(query_ids_by_goal_id.get(dependency_goal_id, []))
            issues.append(issue)
    return issues


_BRANCH_QUERY_GOAL_KINDS = {
    "METRIC",
    "DIMENSION",
    "TIME_WINDOW",
    "ENTITY",
    "DETAIL",
    "RANKING",
}


def _branch_plan_validation_issues(
    session: "GroundedDeepAgentSession",
    specs: list[GroundedQueryBranchSpec],
) -> tuple[
    dict[str, list[str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate one Core-authored branch grouping without planning queries."""

    contract = session.question_goal_contract
    if contract is None:
        return {}, [], [{"code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED"}]
    issues: list[dict[str, Any]] = []
    assignments: dict[str, list[str]] = {}
    allowed_topics = set(session.effective_topics())
    discovery_evidence_by_ref = {
        str(item.get("refId") or ""): item for item in session.core_semantic_evidence if str(item.get("refId") or "")
    }
    query_ids = [str(item.query_id or "").strip() for item in specs]
    if any(not item for item in query_ids) or len(set(query_ids)) != len(query_ids):
        issues.append({"code": "QUERY_BRANCH_ID_INVALID"})
    for spec in specs:
        query_id = str(spec.query_id or "").strip()
        assigned, goal_error = _validated_goal_assignment(
            session,
            spec.goal_ids,
        )
        assignments[query_id] = assigned
        if goal_error:
            issues.append({"queryId": query_id, **goal_error})
        topics = list(dict.fromkeys(str(item or "").strip() for item in spec.topic_scope if str(item or "").strip()))
        if not topics:
            issues.append(
                {
                    "queryId": query_id,
                    "code": "QUERY_BRANCH_TOPIC_SCOPE_REQUIRED",
                }
            )
        unknown_topics = [item for item in topics if item not in allowed_topics]
        if unknown_topics:
            issues.append(
                {
                    "queryId": query_id,
                    "code": "QUERY_BRANCH_TOPIC_OUT_OF_SCOPE",
                    "unknownTopics": unknown_topics,
                    "allowedTopics": sorted(allowed_topics),
                }
            )
        evidence_ref_ids = list(
            dict.fromkeys(str(item or "").strip() for item in spec.evidence_ref_ids if str(item or "").strip())
        )
        unread_ref_ids = [ref_id for ref_id in evidence_ref_ids if ref_id not in discovery_evidence_by_ref]
        if unread_ref_ids:
            issues.append(
                {
                    "queryId": query_id,
                    "code": "QUERY_BRANCH_EVIDENCE_NOT_READ",
                    "missingRefs": unread_ref_ids,
                }
            )
        evidence_topics = {
            str(discovery_evidence_by_ref[ref_id].get("topic") or "")
            for ref_id in evidence_ref_ids
            if ref_id in discovery_evidence_by_ref and str(discovery_evidence_by_ref[ref_id].get("topic") or "")
        }
        outside_topics = sorted(evidence_topics - set(topics))
        if outside_topics:
            issues.append(
                {
                    "queryId": query_id,
                    "code": "QUERY_BRANCH_EVIDENCE_TOPIC_MISMATCH",
                    "evidenceTopics": outside_topics,
                }
            )

    assigned_goal_ids = {goal_id for goal_ids in assignments.values() for goal_id in goal_ids}
    required_query_goal_ids = {
        goal.goal_id
        for goal in contract.goals
        if goal.required and str(goal.kind or "").upper() in _BRANCH_QUERY_GOAL_KINDS
    }
    missing = sorted(required_query_goal_ids - assigned_goal_ids)
    if missing:
        issues.append(
            {
                "code": "QUERY_BRANCH_REQUIRED_GOALS_UNASSIGNED",
                "missingGoalIds": missing,
            }
        )

    query_ids_by_goal_id: dict[str, set[str]] = {}
    for query_id, goal_ids in assignments.items():
        for goal_id in goal_ids:
            query_ids_by_goal_id.setdefault(goal_id, set()).add(query_id)
    for goal in contract.goals:
        kind = str(goal.kind or "").upper()
        if kind == "RANKING":
            structural_input_goal_ids = [
                *list(getattr(goal, "metric_goal_ids", ()) or ()),
                *list(getattr(goal, "dimension_goal_ids", ()) or ()),
            ]
        elif kind == "DETAIL":
            structural_input_goal_ids = list(getattr(goal, "input_goal_ids", ()) or ())
        else:
            # ANALYSIS/COMPARISON inputs intentionally remain portfolio-level:
            # their primitive query branches may execute independently and a
            # verified derived artifact proves the later analysis goal.
            continue
        owner_query_ids = query_ids_by_goal_id.get(goal.goal_id, set())
        for input_goal_id in structural_input_goal_ids:
            input_query_ids = query_ids_by_goal_id.get(
                input_goal_id,
                set(),
            )
            missing_colocation = sorted(owner_query_ids - input_query_ids)
            if missing_colocation:
                issues.append(
                    {
                        "code": "QUERY_BRANCH_STRUCTURAL_GOALS_NOT_COLOCATED",
                        "goalId": goal.goal_id,
                        "goalKind": kind,
                        "inputGoalId": input_goal_id,
                        "queryIds": missing_colocation,
                        "instruction": (
                            "RANKING metric/dimension goals and DETAIL input "
                            "goals belong to the same coherent query branch."
                        ),
                    }
                )

    spec_by_query_id = {str(item.query_id or "").strip(): item for item in specs}
    for goal in contract.goals:
        if not isinstance(goal, RankingQuestionGoal):
            continue
        if goal.population_scope != "SAME_AS_GOAL":
            continue
        owner_query_ids = query_ids_by_goal_id.get(goal.goal_id, set())
        source_query_ids = {
            query_id
            for population_goal_id in goal.population_goal_ids
            for query_id in query_ids_by_goal_id.get(population_goal_id, set())
        }
        required_topics = {
            topic
            for query_id in source_query_ids
            for topic in (spec_by_query_id[query_id].topic_scope if query_id in spec_by_query_id else [])
        }
        for owner_query_id in owner_query_ids:
            owner_spec = spec_by_query_id.get(owner_query_id)
            owner_topics = set(owner_spec.topic_scope if owner_spec else [])
            missing_topics = sorted(required_topics - owner_topics)
            if missing_topics:
                issues.append(
                    {
                        "code": "RANKING_POPULATION_TOPIC_NOT_IN_BRANCH_SCOPE",
                        "goalId": goal.goal_id,
                        "queryId": owner_query_id,
                        "populationGoalIds": list(goal.population_goal_ids),
                        "missingTopics": missing_topics,
                        "instruction": (
                            "A SAME_AS_GOAL ranking branch must include the Topic "
                            "scope that defines its source population."
                        ),
                    }
                )

    dependency_issues = _parallel_goal_dependency_issues(
        contract,
        assignments,
    )
    # A prerequisite already owned by the same branch is a local Contract
    # dependency, not an entity-chain edge.  RANKING/DETAIL structural inputs
    # are explicitly required to be colocated above.  Only prerequisites that
    # are absent from the downstream branch create a cross-branch wait.
    return assignments, dependency_issues, issues


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
    context_workspace: Optional[GroundedContextWorkspace] = None
    context_artifact_inline_max_rows: int = 1
    core_semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    opened_topics: list[str] = field(default_factory=list)
    topic_index_read: bool = False
    expanded_from_attempt_ids: list[str] = field(default_factory=list)
    skill_runs: list[dict[str, Any]] = field(default_factory=list)
    analysis_skill_headers_disclosed: bool = False
    data_collection_sealed: bool = False
    analysis_skill_started: bool = False
    verified_analysis_ledger: list[GroundedDerivedAnalysisArtifact] = field(default_factory=list)
    analysis_data_input_gate_result: dict[str, Any] = field(default_factory=dict)
    query_branch_contexts: dict[str, GroundedQueryBranchContext] = field(default_factory=dict)
    execution_graph_generation: int = 0
    execution_graph_fingerprint: str = ""
    execution_graph_proposal: Optional[GroundedExecutionGraphProposal] = None
    execution_graph_receipt: Optional[GroundedExecutionGraphReceipt] = None
    execution_graph_edges: list[GroundedExecutionEdgeSpec] = field(default_factory=list)
    execution_graph_data_snapshot: Optional[DataSnapshotContract] = None
    execution_graph_history: list[dict[str, Any]] = field(default_factory=list)
    execution_graph_replan_evidence: dict[
        str,
        GroundedExecutionGraphReplanEvidence,
    ] = field(default_factory=dict)
    execution_graph_used_replan_fingerprints: list[str] = field(default_factory=list)
    execution_graph_revision_count: int = 0
    execution_graph_max_revision_count: int = 2
    execution_graph_revision_discovery_evidence_id: str = ""
    execution_graph_revision_discovery_evidence_ids: list[str] = field(
        default_factory=list
    )
    exploration_states: dict[
        str,
        GroundedExplorationCoordinatorState,
    ] = field(default_factory=dict)
    exploration_reports: list[dict[str, Any]] = field(default_factory=list)
    parallel_branches: dict[str, GroundedRuntimeSession] = field(default_factory=dict)
    parallel_branch_goal_ids: dict[str, list[str]] = field(default_factory=dict)
    artifact_goal_ids: dict[str, list[str]] = field(default_factory=dict)
    active_goal_ids: list[str] = field(default_factory=list)
    question_goal_contract: Optional[OriginalQuestionGoalContract] = None
    population_goal_gate_id: str = ""
    population_gate_enforced: bool = False
    population_goal_gate_result: dict[str, Any] = field(default_factory=dict)
    population_goal_attestation: Optional[PopulationVerificationAttestation] = None
    population_graph_receipt: Optional[PopulationDynamicGraphReceipt] = None
    population_pre_execution_references: dict[
        str,
        PopulationPreExecutionReference,
    ] = field(default_factory=dict)
    population_post_gate_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    population_staged_query_artifacts: dict[
        str,
        GroundedVerifiedQueryArtifact,
    ] = field(default_factory=dict)
    population_artifact_query_node_ids: dict[str, str] = field(
        default_factory=dict
    )
    goal_coverage_result: dict[str, Any] = field(default_factory=dict)
    answer_coverage_result: dict[str, Any] = field(default_factory=dict)
    operational_failure: dict[str, Any] = field(default_factory=dict)
    runtime_budget_report: dict[str, Any] = field(default_factory=dict)
    core_context_reports: list[dict[str, Any]] = field(default_factory=list)
    conversation_context: dict[str, Any] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)

    def effective_topics(self) -> list[str]:
        return list(dict.fromkeys([*self.runtime.workspace_topics, *self.opened_topics]))

    def can_expand_topic(self) -> bool:
        if (
            self.runtime.semantic_activation_execution_started
            or _authorized_verified_query_artifacts(self)
        ):
            return False
        if not self.runtime.attempts:
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
            str(item.get("refId") or "") for item in self.core_semantic_evidence if str(item.get("refId") or "")
        }
        evidence_topics = {
            str(item.get("topic") or "") for item in self.core_semantic_evidence if str(item.get("topic") or "")
        }
        for gap in structured_gaps:
            if evidence_refs.intersection(gap.rejected_ref_ids):
                return True
            capability_refs = {
                str(value) for value in _nested_values(gap.required_capability) if str(value).startswith("semantic:")
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


GroundedExecutionFailureDisposition = Literal[
    "NONE",
    "SECURITY_TERMINAL",
    "OPERATIONAL_TERMINAL",
    "RECOVERABLE_EXECUTION",
]


@dataclass(frozen=True)
class GroundedExecutionFailureClassification:
    disposition: GroundedExecutionFailureDisposition = "NONE"
    code: str = ""
    codes: tuple[str, ...] = ()
    message: str = ""

    @property
    def terminal(self) -> bool:
        return self.disposition in {
            "SECURITY_TERMINAL",
            "OPERATIONAL_TERMINAL",
        }


_GROUNDED_SECURITY_TERMINAL_EXECUTION_CODES = frozenset(
    {
        "ACCESS_DENIED",
        "ACL_POLICY_UNAVAILABLE",
        "ACL_SQL_PARSE_FAILED",
        "MERCHANT_SCOPE_DENIED",
        "TABLE_DENIED",
        "TABLE_NOT_ALLOWED",
        "TABLE_ROLE_DENIED",
        "COLUMN_DENIED",
    }
)
_GROUNDED_RECOVERABLE_EXECUTION_CODES = frozenset({"DORIS_ERROR"})


def _grounded_failed_execution_codes(
    run_result: AgentRunResult,
) -> tuple[str, ...]:
    codes: list[str] = []
    failed_task_observed = False
    for task_result in run_result.task_results:
        validation_codes = [
            str(result.error_code or "").strip()
            for result in task_result.validation_results
            if str(result.error_code or "").strip()
        ]
        task_failed = bool(
            not task_result.success
            or task_result.query_bundle.failed
            or validation_codes
        )
        if not task_failed:
            continue
        failed_task_observed = True
        codes.extend(
            validation_codes
            or ["EXECUTION_OPERATIONAL_FAILURE"]
        )
    if run_result.merged_query_bundle.failed and not failed_task_observed:
        codes.append("EXECUTION_OPERATIONAL_FAILURE")
    return tuple(dict.fromkeys(codes))


def _classify_grounded_execution_result(
    run_result: AgentRunResult,
    verified: VerifiedEvidence | None = None,
) -> GroundedExecutionFailureClassification:
    codes = _grounded_failed_execution_codes(run_result)
    if codes:
        security_codes = [
            code
            for code in codes
            if code in _GROUNDED_SECURITY_TERMINAL_EXECUTION_CODES
        ]
        if security_codes:
            return GroundedExecutionFailureClassification(
                disposition="SECURITY_TERMINAL",
                code=security_codes[0],
                codes=codes,
                message=str(
                    run_result.merged_query_bundle.error
                    or security_codes[0]
                )[:500],
            )
        if all(
            code in _GROUNDED_RECOVERABLE_EXECUTION_CODES
            for code in codes
        ):
            return GroundedExecutionFailureClassification(
                disposition="RECOVERABLE_EXECUTION",
                code=codes[0],
                codes=codes,
                message=str(
                    run_result.merged_query_bundle.error or codes[0]
                )[:500],
            )
        return GroundedExecutionFailureClassification(
            disposition="OPERATIONAL_TERMINAL",
            code=codes[0],
            codes=codes,
            message=str(
                run_result.merged_query_bundle.error or codes[0]
            )[:500],
        )
    if verified is None or verified.passed:
        return GroundedExecutionFailureClassification()
    gap_codes = tuple(
        dict.fromkeys(
            str(gap.code or gap.gap_code or "").strip()
            for gap in verified.blocking_gaps
            if str(gap.code or gap.gap_code or "").strip()
        )
    )
    security_codes = [
        code
        for code in gap_codes
        if code in _GROUNDED_SECURITY_TERMINAL_EXECUTION_CODES
    ]
    return GroundedExecutionFailureClassification(
        disposition=(
            "SECURITY_TERMINAL"
            if security_codes
            else "OPERATIONAL_TERMINAL"
        ),
        code=(
            security_codes[0]
            if security_codes
            else gap_codes[0]
            if gap_codes
            else "EVIDENCE_VERIFICATION_GAPPED"
        ),
        codes=(
            gap_codes
            if gap_codes
            else ("EVIDENCE_VERIFICATION_GAPPED",)
        ),
        message=str(
            verified.partial_answer_reason
            or "The execution result failed independent verification."
        )[:500],
    )


def _classify_grounded_execution_exception(
    exc: BaseException,
) -> GroundedExecutionFailureClassification:
    code = str(exc).partition(":")[0].strip()
    if isinstance(exc, PermissionError) or (
        code in _GROUNDED_SECURITY_TERMINAL_EXECUTION_CODES
    ):
        disposition: GroundedExecutionFailureDisposition = (
            "SECURITY_TERMINAL"
        )
    else:
        disposition = "OPERATIONAL_TERMINAL"
    return GroundedExecutionFailureClassification(
        disposition=disposition,
        code=code or "GROUNDED_EXECUTION_INTERNAL_ERROR",
        codes=(code,) if code else ("GROUNDED_EXECUTION_INTERNAL_ERROR",),
        message="%s:%s" % (type(exc).__name__, str(exc)[:500]),
    )


def _artifact_population_authorized(
    session: GroundedDeepAgentSession,
    artifact_id: str,
) -> bool:
    if not session.population_gate_enforced:
        return True
    normalized_artifact_id = str(artifact_id or "").strip()
    query_node_id = session.population_artifact_query_node_ids.get(
        normalized_artifact_id,
        "",
    )
    post_result = session.population_post_gate_results.get(
        query_node_id,
        {},
    )
    return bool(
        query_node_id
        and post_result.get("accepted") is True
        and str(post_result.get("stage") or "") == "POST_RESULT"
    )


def _authorized_verified_query_artifacts(
    session: GroundedDeepAgentSession,
) -> list[GroundedVerifiedQueryArtifact]:
    """Return the sole query-artifact authority exposed to consumers."""

    return [
        artifact
        for artifact in session.runtime.verified_query_ledger
        if _artifact_population_authorized(
            session,
            artifact.artifact_id,
        )
        and (
            not session.population_gate_enforced
            or (
                artifact.verified_evidence.passed
                and artifact.publication_status == "PUBLISHED"
                and verified_query_artifact_integrity_valid(
                    artifact
                )
            )
        )
    ]


def _record_execution_graph_replan_evidence(
    session: GroundedDeepAgentSession,
    *,
    query_node_id: str,
    trigger_kind: str,
    source_stage: str,
    code: str,
    details: dict[str, Any],
    runtime_budget: GroundedRuntimeBudget | None = None,
) -> Optional[GroundedExecutionGraphReplanEvidence]:
    """Seal one server-observed graph-revision trigger.

    Core never supplies this authority. The trigger is derived only from a
    typed Contract gap, datasource report, or execution exception already
    observed inside a governed tool boundary.
    """

    if runtime_budget is not None:
        runtime_budget.checkpoint()
    normalized_query_id = str(query_node_id or "").strip()
    normalized_code = str(code or "").strip()
    with session.lock:
        receipt = session.execution_graph_receipt
        if (
            receipt is None
            or normalized_query_id not in set(receipt.node_ids.values())
            or not normalized_code
            or trigger_kind not in {"DATA_GAP", "TABLE_DELAY", "EXECUTION_ERROR"}
            or source_stage not in {"CONTRACT", "DATASOURCE", "EXECUTION"}
        ):
            return None
        evidence = build_grounded_execution_graph_replan_evidence(
            trigger_kind=trigger_kind,  # type: ignore[arg-type]
            source_stage=source_stage,  # type: ignore[arg-type]
            source_query_node_id=normalized_query_id,
            code=normalized_code,
            graph_receipt=receipt,
            details=dict(details),
        )
        existing = session.execution_graph_replan_evidence.get(evidence.evidence_id)
        if existing is not None:
            return existing.model_copy(deep=True)
        session.execution_graph_replan_evidence[evidence.evidence_id] = evidence.model_copy(deep=True)
        return evidence.model_copy(deep=True)


def _current_execution_graph_replan_evidence(
    session: GroundedDeepAgentSession,
) -> list[GroundedExecutionGraphReplanEvidence]:
    with session.lock:
        receipt = session.execution_graph_receipt
        if receipt is None:
            return []
        used = set(session.execution_graph_used_replan_fingerprints)
        evidences = [
            evidence.model_copy(deep=True)
            for evidence in session.execution_graph_replan_evidence.values()
            if (
                evidence.graph_id == receipt.graph_id
                and evidence.graph_version == receipt.version
                and evidence.graph_fingerprint == receipt.fingerprint
                and evidence.evidence_fingerprint not in used
            )
        ]
        return sorted(
            evidences,
            key=lambda item: item.evidence_id,
        )


def _selected_execution_graph_replan_evidence(
    session: GroundedDeepAgentSession,
) -> list[GroundedExecutionGraphReplanEvidence]:
    selected_ids = list(
        dict.fromkeys(
            [
                *session.execution_graph_revision_discovery_evidence_ids,
                *(
                    [
                        session.execution_graph_revision_discovery_evidence_id
                    ]
                    if session.execution_graph_revision_discovery_evidence_id
                    else []
                ),
            ]
        )
    )
    if not selected_ids:
        return []
    current_by_id = {
        item.evidence_id: item
        for item in _current_execution_graph_replan_evidence(session)
    }
    if any(evidence_id not in current_by_id for evidence_id in selected_ids):
        return []
    return [current_by_id[evidence_id] for evidence_id in selected_ids]


def _execution_graph_replan_evidence_report(
    evidence: GroundedExecutionGraphReplanEvidence,
) -> dict[str, Any]:
    return evidence.model_dump(by_alias=True, mode="json")


def _execution_graph_node_runtime_states(
    session: GroundedDeepAgentSession,
    receipt: GroundedExecutionGraphReceipt,
) -> list[GroundedExecutionGraphNodeRuntimeState]:
    states: list[GroundedExecutionGraphNodeRuntimeState] = []
    for client_key, query_node_id in receipt.node_ids.items():
        context = session.query_branch_contexts.get(query_node_id)
        if context is None:
            lifecycle = "UNEXECUTED"
        else:
            published = bool(
                context.verified_artifact_ids
                or context.status == "VERIFIED"
            )
            if published:
                lifecycle = "PUBLISHED"
            elif context.status in {
                "FAILED",
                "SNAPSHOT_BLOCKED",
            }:
                lifecycle = "EXECUTION_FAILED"
            elif query_node_id in session.population_pre_execution_references or context.status == "EXECUTING":
                lifecycle = "PRE_AUTHORIZED"
            else:
                lifecycle = "UNEXECUTED"
        states.append(
            GroundedExecutionGraphNodeRuntimeState(
                client_key=client_key,
                query_node_id=query_node_id,
                lifecycle=lifecycle,  # type: ignore[arg-type]
            )
        )
    return states


def _build_graph_revision_base_session_checkpoint(
    session: GroundedDeepAgentSession,
    *,
    execution_proposal: GroundedExecutionGraphProposal,
    execution_receipt: GroundedExecutionGraphReceipt,
    population_receipt: PopulationDynamicGraphReceipt,
    node_states: Sequence[GroundedExecutionGraphNodeRuntimeState],
) -> GroundedGraphRevisionBaseSessionCheckpoint:
    goal_contract = session.question_goal_contract
    if goal_contract is None:
        raise GroundedGraphRevisionJournalError(
            "GRAPH_REVISION_BASE_GOAL_CONTRACT_REQUIRED"
        )
    states_by_key = {
        item.client_key: item for item in node_states
    }
    branches: list[GroundedGraphRevisionBaseBranchCheckpoint] = []
    for node in execution_proposal.nodes:
        query_node_id = execution_receipt.node_ids[node.client_key]
        context = session.query_branch_contexts.get(query_node_id)
        state = states_by_key.get(node.client_key)
        if state is None:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_NODE_STATE_REQUIRED"
            )
        branches.append(
            GroundedGraphRevisionBaseBranchCheckpoint(
                client_key=node.client_key,
                query_node_id=query_node_id,
                objective=node.objective,
                goal_ids=tuple(node.goal_ids),
                topic_scope=tuple(node.topic_scope),
                evidence_ref_ids=tuple(node.evidence_ref_ids),
                dependency_query_node_ids=tuple(
                    context.dependency_query_ids
                    if context is not None
                    else ()
                ),
                contract_scope_query_node_ids=tuple(
                    context.contract_scope_query_ids
                    if context is not None
                    else ()
                ),
                opened_topics=tuple(
                    context.opened_topics
                    if context is not None
                    else ()
                ),
                lifecycle=state.lifecycle,
                status=(
                    context.status
                    if context is not None
                    else "DECLARED"
                ),
                verified_artifact_ids=tuple(
                    context.verified_artifact_ids
                    if context is not None
                    else ()
                ),
                last_gaps=tuple(
                    dict(item)
                    for item in (
                        context.last_gaps
                        if context is not None
                        else ()
                    )
                ),
            )
        )
    runtime = session.runtime
    semantic_activation = (
        runtime.semantic_activation_seal.model_dump(
            by_alias=True,
            mode="json",
        )
        if runtime.semantic_activation_seal is not None
        else {}
    )
    checkpoint = GroundedGraphRevisionBaseSessionCheckpoint(
        question=runtime.question,
        goal_contract=goal_contract.model_dump(
            by_alias=True,
            mode="json",
        ),
        execution_proposal=execution_proposal.model_dump(
            by_alias=True,
            mode="json",
        ),
        execution_receipt=execution_receipt.model_dump(
            by_alias=True,
            mode="json",
        ),
        population_receipt=population_receipt.model_dump(
            by_alias=True,
            mode="json",
        ),
        semantic_evidence=tuple(
            json.loads(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
            for item in session.core_semantic_evidence
        ),
        branches=tuple(branches),
        runtime_state={
            "revision": runtime.revision,
            "activeGeneration": runtime.active_generation,
            "activeGoalContractFingerprint": (
                runtime.active_goal_contract_fingerprint
            ),
            "workspaceTopics": list(runtime.workspace_topics),
            "semanticActivationSeal": semantic_activation,
            "semanticActivationExecutionStarted": bool(
                runtime.semantic_activation_execution_started
            ),
        },
        verified_query_artifacts=tuple(
            item.model_dump(by_alias=True, mode="json")
            for item in runtime.verified_query_ledger
        ),
        verified_entity_sets=tuple(
            item.model_dump(by_alias=True, mode="json")
            for item in runtime.verified_entity_sets
        ),
        verified_rule_artifacts=tuple(
            item.model_dump(by_alias=True, mode="json")
            for item in runtime.verified_rule_ledger
        ),
        artifact_goal_ids={
            artifact_id: tuple(goal_ids)
            for artifact_id, goal_ids in (
                session.artifact_goal_ids.items()
            )
        },
        population_pre_execution_references={
            query_node_id: reference.model_dump(
                by_alias=True,
                mode="json",
            )
            for query_node_id, reference in (
                session.population_pre_execution_references.items()
            )
        },
        population_post_gate_results={
            query_node_id: dict(result)
            for query_node_id, result in (
                session.population_post_gate_results.items()
            )
        },
        population_artifact_query_node_ids=dict(
            session.population_artifact_query_node_ids
        ),
        population_goal_gate_id=session.population_goal_gate_id,
        population_goal_gate_result=dict(
            session.population_goal_gate_result
        ),
        population_goal_attestation=(
            session.population_goal_attestation.model_dump(
                by_alias=True,
                mode="json",
            )
            if session.population_goal_attestation is not None
            else {}
        ),
        execution_graph_data_snapshot=(
            session.execution_graph_data_snapshot.model_dump(
                by_alias=True,
                mode="json",
            )
            if session.execution_graph_data_snapshot is not None
            else {}
        ),
        execution_graph_revision_count=(
            session.execution_graph_revision_count
        ),
        execution_graph_max_revision_count=(
            session.execution_graph_max_revision_count
        ),
        opened_topics=tuple(session.opened_topics),
    )
    return seal_grounded_graph_revision_base_session_checkpoint(
        checkpoint
    )


class _SessionExplorationStateStore:
    """CAS adapter that keeps advisory state inside the active run session."""

    def __init__(self, session: GroundedDeepAgentSession) -> None:
        self.session = session

    def create(self, state: GroundedExplorationCoordinatorState) -> bool:
        assignment_id = state.assignment.assignment_id
        with self.session.lock:
            if assignment_id in self.session.exploration_states:
                return False
            self.session.exploration_states[assignment_id] = state
            return True

    def load(
        self,
        assignment_id: str,
    ) -> GroundedExplorationCoordinatorState | None:
        with self.session.lock:
            return self.session.exploration_states.get(assignment_id)

    def compare_and_swap(
        self,
        assignment_id: str,
        *,
        expected_revision: int,
        replacement: GroundedExplorationCoordinatorState,
    ) -> bool:
        with self.session.lock:
            current = self.session.exploration_states.get(assignment_id)
            if (
                current is None
                or current.ledger.revision != expected_revision
                or replacement.assignment.assignment_id != assignment_id
            ):
                return False
            self.session.exploration_states[assignment_id] = replacement
            return True


@dataclass(frozen=True)
class GroundedDeepAgentRunContext:
    thread_id: str
    run_id: str
    session: GroundedDeepAgentSession
    budget: Optional[GroundedRuntimeBudget] = None
    listener: Optional[Callable[[str, str, dict[str, Any]], None]] = None


class GroundedSemanticBackend:
    """Read-only semantic filesystem scoped to one Grounded Core session."""

    def __init__(
        self,
        semantic_catalog: Any,
        *,
        reader_is_core: Optional[Callable[[], bool]] = None,
        semantic_activation_refresher: Optional[Callable[[GroundedDeepAgentSession], Any]] = None,
    ):
        self.semantic_catalog = semantic_catalog
        self.reader_is_core = reader_is_core or _deepagent_reader_is_core
        self.semantic_activation_refresher = semantic_activation_refresher
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
                    _grounded_file_info(
                        path="/topics/index.json",
                        is_dir=False,
                        size=0,
                    ),
                    *[
                        _grounded_file_info(
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
                _grounded_file_info(
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
                if topic_name not in session.runtime.workspace_topics:
                    session.runtime.workspace_topics.append(topic_name)
                if self.semantic_activation_refresher is not None:
                    try:
                        self.semantic_activation_refresher(session)
                    except Exception as exc:
                        session.opened_topics = [item for item in session.opened_topics if item != topic_name]
                        session.runtime.workspace_topics = [
                            item for item in session.runtime.workspace_topics if item != topic_name
                        ]
                        return ReadResult(
                            error=(
                                "SEMANTIC_ACTIVATION_TOPIC_RESEAL_FAILED:"
                                "%s:%s"
                                % (
                                    type(exc).__name__,
                                    str(exc)[:300],
                                )
                            )
                        )
                session.mark_topic_expanded()

        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        content = "".join(lines[start:end])
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if session is not None and complete and kind not in {"TOPIC_INDEX", "TOPIC_MANIFEST"}:
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
        return bool(
            self.record_core_read_receipt(
                session,
                file_path,
                offset=offset,
                limit=limit,
            )
        )

    def record_core_read_receipt(
        self,
        session: GroundedDeepAgentSession,
        file_path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> dict[str, Any]:
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
            return dict(receipt)
        try:
            result = self.semantic_catalog.read(
                path=normalized,
                max_chars=2_000_000,
                offset=0,
            )
        except Exception:
            return {}
        if not isinstance(result, dict) or not result.get("success"):
            return {}
        kind = str(result.get("kind") or "").upper()
        topic_name = str(result.get("topic") or "")
        if kind in {"TOPIC_INDEX", "TOPIC_MANIFEST"}:
            return {}
        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if not complete or topic_name not in session.effective_topics():
            return {}
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
            return {}
        self._retain_evidence(session, evidence)
        return dict(evidence)

    def _retain_evidence(
        self,
        session: GroundedDeepAgentSession,
        evidence: dict[str, Any],
    ) -> None:
        with session.lock:
            retained = [item for item in session.core_semantic_evidence if item.get("refId") != evidence.get("refId")]
            retained.append(dict(evidence))
            # The run-level governed tool budget bounds discovery. Evidence
            # referenced by a graph is never silently evicted from its
            # immutable snapshot.
            session.core_semantic_evidence = retained

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
        topics = (
            [self._topic(normalized)] if self._topic(normalized) else (session.effective_topics() if session else [])
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
            matches.extend(GrepMatch(path=hit_path, line=1, text=str(snippet)[:1000]) for snippet in snippets[:3])
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


def _published_query_artifact_digests(
    session: GroundedDeepAgentSession,
) -> dict[str, str]:
    """Derive Core read authority only from verifier-committed ledger rows."""

    workspace = session.context_workspace
    if workspace is None:
        return {}
    owner_fingerprint = str(workspace.owner_fingerprint or "")
    semantic_seal = session.runtime.semantic_activation_seal
    if semantic_seal is not None and not semantic_activation_seal_valid(semantic_seal):
        return {}
    expected_semantic_fingerprint = str(
        semantic_seal.semantic_activation_fingerprint if semantic_seal is not None else ""
    )
    expected_seal_fingerprint = str(semantic_seal.seal_fingerprint if semantic_seal is not None else "")
    allowed: dict[str, str] = {}
    conflicts: set[str] = set()
    for artifact in _authorized_verified_query_artifacts(session):
        if (
            not _artifact_population_authorized(
                session,
                artifact.artifact_id,
            )
            or not verified_query_artifact_integrity_valid(artifact)
            or str(getattr(artifact, "publication_status", "") or "") != "PUBLISHED"
            or not bool(
                getattr(
                    getattr(artifact, "verified_evidence", None),
                    "passed",
                    False,
                )
            )
        ):
            continue
        contract_fingerprint = str(getattr(artifact, "contract_fingerprint", "") or "")
        sql_fingerprint = str(getattr(artifact, "sql_fingerprint", "") or "")
        generation = int(getattr(artifact, "generation", 0) or 0)
        attempt_id = str(getattr(artifact, "attempt_id", "") or "")
        attempt_fingerprint = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
        verified = getattr(artifact, "verified_evidence", None)
        verified_payload = (
            verified.model_dump(by_alias=True, mode="json")
            if callable(getattr(verified, "model_dump", None))
            else verified
        )
        verified_fingerprint = hashlib.sha256(
            json.dumps(
                verified_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        receipts = getattr(artifact, "result_artifact_receipts", None)
        if not isinstance(receipts, list) or not receipts:
            continue
        for raw_receipt in receipts:
            if not isinstance(raw_receipt, Mapping):
                continue
            receipt = dict(raw_receipt)
            semantic_fingerprint = str(receipt.get("semanticActivationFingerprint") or "")
            artifact_fingerprint = str(receipt.get("artifactFingerprint") or "")
            if (
                str(receipt.get("contextOwnerFingerprint") or "") != owner_fingerprint
                or not _sha256_value(semantic_fingerprint)
                or not _sha256_value(artifact_fingerprint)
                or receipt.get("executionGeneration") != generation
                or str(receipt.get("attemptFingerprint") or "") != attempt_fingerprint
                or str(receipt.get("contractFingerprint") or "") != contract_fingerprint
                or str(receipt.get("sqlEvidenceFingerprint") or "") != sql_fingerprint
                or str(receipt.get("verifiedEvidenceSha256") or "") != verified_fingerprint
                or (bool(expected_semantic_fingerprint) and semantic_fingerprint != expected_semantic_fingerprint)
                or (
                    bool(expected_semantic_fingerprint)
                    and str(
                        getattr(
                            artifact,
                            "semantic_activation_fingerprint",
                            "",
                        )
                        or ""
                    )
                    != expected_semantic_fingerprint
                )
                or (
                    bool(expected_seal_fingerprint)
                    and str(
                        getattr(
                            artifact,
                            "semantic_activation_seal_fingerprint",
                            "",
                        )
                        or ""
                    )
                    != expected_seal_fingerprint
                )
            ):
                continue
            receipt_paths = (
                (
                    receipt.get("manifestRelativePath"),
                    receipt.get("queryManifestSha256"),
                    receipt.get("manifestContentAddress"),
                ),
                (
                    receipt.get("rowsRelativePath"),
                    receipt.get("rowsSha256"),
                    receipt.get("rowsContentAddress"),
                ),
            )
            for raw_path, raw_digest, raw_address in receipt_paths:
                relative = _safe_artifact_relative_path(raw_path)
                digest = str(raw_digest or "")
                if (
                    not relative
                    or not _sha256_value(digest)
                    or str(raw_address or "") != "sha256:%s" % digest
                    or relative in conflicts
                ):
                    continue
                previous = allowed.get(relative)
                if previous is not None and previous != digest:
                    allowed.pop(relative, None)
                    conflicts.add(relative)
                    continue
                allowed[relative] = digest
    return allowed


def _sha256_value(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _safe_artifact_relative_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return ""
    return path.as_posix()


class GroundedRunFilesystemBackend:
    """Context-aware run filesystem used for artifacts and Core scratch.

    The backend never chooses semantic bindings or query topology. It only
    exposes the identity-bound workspace already created by the server for the
    active run. Query artifacts are read-only; `/workspace` is the durable
    scratch/offload surface owned by Deep Agents middleware.
    """

    MAX_GREP_FILES = 200
    MAX_GREP_MATCHES = 100
    MAX_PAGE_CHARS_CEILING = 64_000

    def __init__(
        self,
        *,
        root_kind: str,
        read_only: bool,
        settings: Any = None,
        allowed_artifact_digests: Optional[Mapping[str, str]] = None,
        allowed_artifact_digest_provider: Optional[Callable[[GroundedDeepAgentSession], Mapping[str, str]]] = None,
    ) -> None:
        if root_kind not in {"artifacts", "scratch"}:
            raise ValueError("unsupported grounded filesystem root kind")
        self.root_kind = root_kind
        self.read_only = bool(read_only)
        self.settings = settings
        self.allowed_artifact_digests = (
            {
                str(Path(str(path))).replace("\\", "/").lstrip("/"): str(digest)
                for path, digest in dict(allowed_artifact_digests or {}).items()
            }
            if allowed_artifact_digests is not None
            else None
        )
        self.allowed_artifact_digest_provider = allowed_artifact_digest_provider

    def _current_allowed_artifact_digests(
        self,
    ) -> Optional[dict[str, str]]:
        if self.root_kind != "artifacts":
            return None
        if self.allowed_artifact_digests is not None:
            raw_allowed: Mapping[str, str] = self.allowed_artifact_digests
        elif self.allowed_artifact_digest_provider is not None:
            session = self._session()
            if session is None:
                return {}
            try:
                raw_allowed = self.allowed_artifact_digest_provider(session)
            except Exception:
                return {}
        else:
            return {}
        normalized: dict[str, str] = {}
        for raw_path, raw_digest in dict(raw_allowed or {}).items():
            relative = _safe_artifact_relative_path(raw_path)
            digest = str(raw_digest or "")
            if relative and _sha256_value(digest):
                normalized[relative] = digest
        return normalized

    @staticmethod
    def _session() -> Optional[GroundedDeepAgentSession]:
        return _SEMANTIC_SCOPE.get()

    def _root(self) -> tuple[Optional[Path], str]:
        session = self._session()
        if session is None or session.context_workspace is None:
            return None, "GROUNDED_CONTEXT_WORKSPACE_REQUIRED"
        workspace = session.context_workspace
        root = workspace.artifacts_root if self.root_kind == "artifacts" else workspace.core_scratch_root
        return root, ""

    def _relative_path(self, value: str) -> str:
        normalized = str(value or "").strip().replace("\\", "/").strip("/")
        prefix = "artifacts" if self.root_kind == "artifacts" else "workspace"
        if normalized == prefix:
            return ""
        if normalized.startswith(prefix + "/"):
            return normalized[len(prefix) + 1 :]
        return normalized

    def _resolve(self, value: str) -> tuple[Optional[Path], Optional[Path], str]:
        root, error = self._root()
        if root is None:
            return None, None, error
        try:
            target = resolve_context_path(root, self._relative_path(value))
        except ContextPathOutsideRootError:
            return root, None, "GROUNDED_CONTEXT_PATH_OUTSIDE_ROOT"
        return root, target, ""

    @staticmethod
    def _internal(path: Path) -> bool:
        return path.name.startswith(
            (
                ".artifact-write-",
                ".artifact-lock-",
                ".artifact-immutable-",
                ".context-",
            )
        )

    @staticmethod
    def _file_info(root: Path, path: Path) -> Any:
        return _grounded_file_info(
            path="/" + str(path.relative_to(root)).replace("\\", "/") + ("/" if path.is_dir() else ""),
            is_dir=path.is_dir(),
            size=0 if path.is_dir() else int(path.stat().st_size),
            modified_at="",
        )

    def _page_chars(self, requested: int) -> int:
        configured = int(
            getattr(
                self.settings,
                "context_file_inline_max_chars",
                12_000,
            )
            or 12_000
        )
        return max(
            1,
            min(
                max(1, int(requested or configured)),
                max(1, configured),
                self.MAX_PAGE_CHARS_CEILING,
            ),
        )

    def _artifact_store(
        self,
        root: Path,
    ) -> Optional[WorkspaceArtifactStore]:
        if self.settings is None:
            return None
        return WorkspaceArtifactStore(self.settings, root)

    def _artifact_file_valid(self, root: Path, path: Path) -> bool:
        if self._internal(path) or not path.is_file():
            return False
        store = self._artifact_store(root)
        if store is None:
            return False
        result = store.read(
            str(path.relative_to(root)),
            offset=0,
            max_chars=1,
            require_immutable=True,
        )
        if not result.get("success"):
            return False
        allowed = self._current_allowed_artifact_digests() or {}
        relative = str(path.relative_to(root)).replace("\\", "/")
        expected = allowed.get(relative)
        return bool(expected) and str(result.get("sha256") or "") == expected

    def _path_allowed(self, root: Path, path: Path) -> bool:
        if self.root_kind != "artifacts":
            return True
        allowed = self._current_allowed_artifact_digests() or {}
        relative = str(path.relative_to(root)).replace("\\", "/")
        if path.is_file():
            return relative in allowed
        prefix = relative.rstrip("/")
        if not prefix or prefix == ".":
            return bool(allowed)
        return any(item.startswith(prefix + "/") for item in allowed)

    def _artifact_directory_has_valid_file(
        self,
        root: Path,
        path: Path,
    ) -> bool:
        if not path.is_dir():
            return False
        allowed = self._current_allowed_artifact_digests() or {}
        directory_relative = (
            str(path.relative_to(root))
            .replace(
                "\\",
                "/",
            )
            .strip("/")
        )
        prefix = "%s/" % directory_relative if directory_relative else ""
        for relative in sorted(allowed):
            if prefix and not relative.startswith(prefix):
                continue
            try:
                safe_candidate = resolve_context_path(root, relative)
            except ContextPathOutsideRootError:
                continue
            if self._artifact_file_valid(root, safe_candidate):
                return True
        return False

    def ls(self, path: str) -> LsResult:
        root, target, error = self._resolve(path)
        if error:
            return LsResult(error=error)
        assert root is not None and target is not None
        if not target.exists():
            return LsResult(error="GROUNDED_CONTEXT_FILE_NOT_FOUND")
        if not self._path_allowed(root, target):
            return LsResult(error="GROUNDED_CONTEXT_FILE_NOT_ALLOWED")
        if target.is_file():
            if self._internal(target):
                return LsResult(error="GROUNDED_CONTEXT_FILE_NOT_FOUND")
            if self.root_kind == "artifacts" and not self._artifact_file_valid(root, target):
                return LsResult(error="GROUNDED_CONTEXT_ARTIFACT_INVALID")
            return LsResult(entries=[self._file_info(root, target)])
        entries: list[FileInfo] = []
        try:
            for child in sorted(target.iterdir(), key=lambda item: item.name):
                safe_child = resolve_context_path(root, child)
                if self._internal(safe_child):
                    continue
                if not self._path_allowed(root, safe_child):
                    continue
                if self.root_kind == "artifacts":
                    if safe_child.is_file() and not self._artifact_file_valid(
                        root,
                        safe_child,
                    ):
                        continue
                    if safe_child.is_dir() and not self._artifact_directory_has_valid_file(
                        root,
                        safe_child,
                    ):
                        continue
                entries.append(self._file_info(root, safe_child))
        except (ContextPathOutsideRootError, OSError) as exc:
            return LsResult(error="GROUNDED_CONTEXT_LS_FAILED:%s" % str(exc)[:160])
        return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        root, target, error = self._resolve(file_path)
        if error:
            return ReadResult(error=error)
        assert root is not None and target is not None
        if self._internal(target) or not target.is_file():
            return ReadResult(error="GROUNDED_CONTEXT_FILE_NOT_FOUND")
        if not self._path_allowed(root, target):
            return ReadResult(error="GROUNDED_CONTEXT_FILE_NOT_ALLOWED")
        start = max(0, int(offset or 0))
        page_chars = self._page_chars(limit)
        try:
            if self.settings is not None:
                store = self._artifact_store(root)
                if store is None:
                    return ReadResult(error="GROUNDED_CONTEXT_ARTIFACT_STORE_REQUIRED")
                read_result = store.read(
                    str(target.relative_to(root)),
                    offset=start,
                    max_chars=page_chars,
                    require_immutable=self.root_kind == "artifacts",
                )
                if not read_result.get("success"):
                    return ReadResult(error=str(read_result.get("error") or "GROUNDED_CONTEXT_READ_FAILED"))
                allowed = self._current_allowed_artifact_digests()
                if allowed is not None:
                    relative = str(target.relative_to(root)).replace(
                        "\\",
                        "/",
                    )
                    expected_digest = allowed.get(relative)
                    if not expected_digest or str(read_result.get("sha256") or "") != expected_digest:
                        return ReadResult(error="GROUNDED_CONTEXT_ARTIFACT_COMMIT_MISMATCH")
                text = str(read_result.get("content") or "")
                next_offset = read_result.get("nextContentOffsetChars")
                content_hash = str(read_result.get("sha256") or "")
                estimated_chars = int(read_result.get("estimatedChars") or len(text))
            else:
                with target.open("r", encoding="utf-8") as stream:
                    if start:
                        stream.read(start)
                    text = stream.read(page_chars)
                    has_more = bool(stream.read(1))
                next_offset = start + len(text) if has_more else None
                content_hash = ""
                estimated_chars = int(target.stat().st_size) if has_more else start + len(text)
        except (OSError, UnicodeError) as exc:
            return ReadResult(error="GROUNDED_CONTEXT_READ_FAILED:%s" % str(exc)[:160])
        file_data: dict[str, Any] = {
            "content": text,
            "encoding": "utf-8",
        }
        if start or next_offset is not None:
            file_data.update(
                {
                    "cursorSemantics": "CHAR_OFFSET",
                    "contentOffsetChars": start,
                    "nextContentOffsetChars": next_offset,
                    "estimatedChars": estimated_chars,
                    "contentHash": content_hash,
                    "resultCoverage": ("PREVIEW" if next_offset is not None else "ALL_CHARS"),
                }
            )
        return ReadResult(file_data=file_data)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        root, target, error = self._resolve(path or "")
        if error:
            return GrepResult(error=error)
        assert root is not None and target is not None
        if not self._path_allowed(root, target):
            return GrepResult(error="GROUNDED_CONTEXT_FILE_NOT_ALLOWED")
        needle = str(pattern or "")[:500].casefold()
        if not needle:
            return GrepResult(matches=[])
        if self.root_kind == "artifacts":
            store = self._artifact_store(root)
            if store is None:
                return GrepResult(error="GROUNDED_CONTEXT_ARTIFACT_STORE_REQUIRED")
            relative_scope = (
                str(target.relative_to(root))
                .replace(
                    "\\",
                    "/",
                )
                .strip("./")
            )
            matches: list[GrepMatch] = []
            remaining_chars = self._page_chars(self.MAX_PAGE_CHARS_CEILING)
            allowed = self._current_allowed_artifact_digests() or {}
            inspected = 0
            for relative, expected_digest in sorted(allowed.items()):
                if inspected >= self.MAX_GREP_FILES:
                    break
                if relative_scope and not (
                    relative == relative_scope or relative.startswith(relative_scope.rstrip("/") + "/")
                ):
                    continue
                routed_path = "/" + relative.lstrip("/")
                if glob and not fnmatch.fnmatch(routed_path, glob):
                    continue
                read_result = store.read(
                    relative,
                    offset=0,
                    max_chars=self.MAX_PAGE_CHARS_CEILING,
                    require_immutable=True,
                )
                if not read_result.get("success") or str(read_result.get("sha256") or "") != expected_digest:
                    continue
                inspected += 1
                for line_number, line in enumerate(
                    str(read_result.get("content") or "").splitlines(),
                    start=1,
                ):
                    if needle not in line[:20_000].casefold():
                        continue
                    if remaining_chars <= 0:
                        return GrepResult(matches=matches)
                    text = line[: min(5_000, remaining_chars)]
                    matches.append(
                        GrepMatch(
                            path=routed_path,
                            line=line_number,
                            text=text,
                        )
                    )
                    remaining_chars -= len(text)
                    if len(matches) >= self.MAX_GREP_MATCHES:
                        return GrepResult(matches=matches)
            return GrepResult(matches=matches)
        candidates = [target] if target.is_file() else target.rglob("*")
        inspected = 0
        matches: list[GrepMatch] = []
        remaining_chars = self._page_chars(self.MAX_PAGE_CHARS_CEILING)
        for candidate in candidates:
            if inspected >= self.MAX_GREP_FILES or len(matches) >= self.MAX_GREP_MATCHES:
                break
            try:
                safe_candidate = resolve_context_path(root, candidate)
            except ContextPathOutsideRootError:
                continue
            if self._internal(safe_candidate) or not safe_candidate.is_file():
                continue
            relative = "/" + str(safe_candidate.relative_to(root)).replace("\\", "/")
            if glob and not fnmatch.fnmatch(relative, glob):
                continue
            inspected += 1
            try:
                with safe_candidate.open("r", encoding="utf-8") as stream:
                    lines = stream.read(self.MAX_PAGE_CHARS_CEILING).splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle not in line[:20_000].casefold():
                    continue
                if remaining_chars <= 0:
                    return GrepResult(matches=matches)
                text = line[: min(5_000, remaining_chars)]
                matches.append(
                    GrepMatch(
                        path=relative,
                        line=line_number,
                        text=text,
                    )
                )
                remaining_chars -= len(text)
                if len(matches) >= self.MAX_GREP_MATCHES:
                    break
        return GrepResult(matches=matches)

    def glob(
        self,
        pattern: str,
        path: Optional[str] = None,
    ) -> GlobResult:
        root, target, error = self._resolve(path or "")
        if error:
            return GlobResult(error=error)
        assert root is not None and target is not None
        if not self._path_allowed(root, target):
            return GlobResult(error="GROUNDED_CONTEXT_FILE_NOT_ALLOWED")
        if self.root_kind == "artifacts":
            matches: list[FileInfo] = []
            allowed = self._current_allowed_artifact_digests() or {}
            for relative in sorted(allowed):
                try:
                    safe_candidate = resolve_context_path(root, relative)
                except ContextPathOutsideRootError:
                    continue
                if target.is_file() and safe_candidate != target:
                    continue
                if target.is_dir() and target not in safe_candidate.parents:
                    continue
                if not self._artifact_file_valid(root, safe_candidate):
                    continue
                routed_path = "/" + str(safe_candidate.relative_to(root)).replace("\\", "/").lstrip("/")
                if fnmatch.fnmatch(routed_path, pattern):
                    matches.append(
                        _grounded_file_info(
                            path=routed_path,
                            is_dir=False,
                            size=int(safe_candidate.stat().st_size),
                            modified_at="",
                        )
                    )
                if len(matches) >= 500:
                    break
            return GlobResult(matches=matches)
        candidates = [target] if target.is_file() else target.rglob("*")
        matches: list[FileInfo] = []
        for candidate in candidates:
            try:
                safe_candidate = resolve_context_path(root, candidate)
            except ContextPathOutsideRootError:
                continue
            if self._internal(safe_candidate):
                continue
            info = self._file_info(root, safe_candidate)
            if fnmatch.fnmatch(info.path, pattern):
                matches.append(info)
        return GlobResult(matches=matches[:500])

    def write(self, file_path: str, content: str) -> WriteResult:
        if self.read_only:
            return WriteResult(
                error="GROUNDED_CONTEXT_FILESYSTEM_READ_ONLY",
                path=file_path,
            )
        session = self._session()
        if session is None or session.context_workspace is None:
            return WriteResult(
                error="GROUNDED_CONTEXT_WORKSPACE_REQUIRED",
                path=file_path,
            )
        try:
            target = session.context_workspace.write_core_scratch(
                self._relative_path(file_path),
                content,
            )
        except GroundedContextWorkspaceError as exc:
            return WriteResult(error=str(exc), path=file_path)
        return WriteResult(
            path="/" + str(target.relative_to(session.context_workspace.core_scratch_root)).replace("\\", "/")
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if self.read_only:
            return EditResult(
                error="GROUNDED_CONTEXT_FILESYSTEM_READ_ONLY",
                path=file_path,
            )
        read_result = self.read(file_path, offset=0, limit=2_000_000)
        if read_result.error or not read_result.file_data:
            return EditResult(error=read_result.error, path=file_path)
        if read_result.file_data.get("nextContentOffsetChars") is not None:
            return EditResult(
                error="GROUNDED_CONTEXT_EDIT_REQUIRES_COMPLETE_FILE",
                path=file_path,
            )
        content = str(read_result.file_data.get("content") or "")
        if not old_string or old_string not in content:
            return EditResult(
                error="GROUNDED_CONTEXT_EDIT_TARGET_NOT_FOUND",
                path=file_path,
            )
        replacement = (
            content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        )
        written = self.write(file_path, replacement)
        return EditResult(error=written.error, path=written.path)

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

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def agrep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        return self.grep(pattern, path, glob)

    async def aglob(
        self,
        pattern: str,
        path: Optional[str] = None,
    ) -> GlobResult:
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

    async def aglob_info(
        self,
        pattern: str,
        path: str = "/",
    ) -> list[FileInfo]:
        return self.glob_info(pattern, path)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> Any:
        return self.grep_raw(pattern, path, glob)


def _message_content_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content or "")


def _knowledge_relative_path(value: str) -> str:
    normalized = GroundedSemanticBackend._path(value)
    if normalized == "knowledge":
        return ""
    if normalized.startswith("knowledge/"):
        return normalized[len("knowledge/") :]
    return normalized


def _filesystem_tool_namespace(
    tool_name: str,
    args: dict[str, Any],
) -> str:
    if tool_name == "retrieve_knowledge":
        return "knowledge"
    raw = str(args.get("file_path") or args.get("path") or "").strip().replace("\\", "/").strip("/")
    if not raw:
        return "unknown"
    first = raw.split("/", 1)[0]
    if first in {"knowledge", "artifacts", "workspace"}:
        return first
    return "unknown"


def _replace_message_content(message: Any, content: str) -> Any:
    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": content})
    return ToolMessage(
        content=content,
        name=str(getattr(message, "name", "") or ""),
        tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
        status=str(getattr(message, "status", "success") or "success"),
    )


def _compact_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, (dict, list, tuple)):
            return {"omitted": True, "itemCount": len(value)}
        return str(value or "")[:240]
    if isinstance(value, dict):
        return {str(key): _compact_json_value(item, depth=depth + 1) for key, item in list(value.items())[:16]}
    if isinstance(value, (list, tuple)):
        return [_compact_json_value(item, depth=depth + 1) for item in list(value)[:12]]
    if isinstance(value, str):
        return value[:400]
    return value


def _semantic_payload_summary(kind: str, content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized_kind = str(kind or "").upper()
    if normalized_kind == "TABLE_DETAIL":
        summary = _compact_json_value(
            {
                key: payload.get(key)
                for key in (
                    "tableName",
                    "title",
                    "businessSummary",
                    "dataGrain",
                    "timeColumn",
                    "merchantFilterColumn",
                    "freshnessType",
                    "supportsDetail",
                    "supportsMetrics",
                    "preferredFor",
                    "children",
                )
                if payload.get(key) not in (None, "", [], {})
            }
        )
        navigation = payload.get("semanticNavigation")
        if isinstance(navigation, dict):

            def compact_leaves(value: Any) -> list[dict[str, Any]]:
                leaves = value if isinstance(value, list) else []
                compacted: list[dict[str, Any]] = []
                # L1 table details are already bounded by the semantic
                # publisher (currently 16 metric and 26 column leaves). Keep
                # every advertised coordinate so a closed broad-index policy
                # never makes the final advertised leaf unreachable.
                for leaf in leaves:
                    if not isinstance(leaf, dict):
                        continue
                    item = {
                        "key": str(leaf.get("key") or ""),
                        "aliases": [
                            str(alias) for alias in (leaf.get("aliases") or [])[:6] if str(alias or "").strip()
                        ],
                        "refId": str(leaf.get("refId") or ""),
                        "path": str(leaf.get("path") or ""),
                    }
                    compacted.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
                return compacted

            summary["semanticNavigation"] = {
                "source": str(navigation.get("source") or ""),
                "questionIndependent": bool(navigation.get("questionIndependent")),
                "bindingEvidence": bool(navigation.get("bindingEvidence")),
                "publishedCounts": dict(navigation.get("publishedCounts") or {}),
                "advertisedCounts": dict(navigation.get("advertisedCounts") or {}),
                "metricLeaves": compact_leaves(navigation.get("metricLeaves")),
                "columnLeaves": compact_leaves(navigation.get("columnLeaves")),
            }
        return summary
    if normalized_kind == "METRIC":
        definition = payload.get("metric") if isinstance(payload.get("metric"), dict) else payload
        return _compact_json_value(
            {
                key: definition.get(key)
                for key in (
                    "metricKey",
                    "businessName",
                    "formula",
                    "unit",
                    "description",
                    "sourceColumns",
                    "aliases",
                    "aggregationPolicy",
                    "metricGrain",
                    "applicableTimeGrain",
                    "timeColumn",
                    "timeSemantics",
                )
                if definition.get(key) not in (None, "", [], {})
            }
        )
    if normalized_kind in {"COLUMN", "FIELD"}:
        definition = payload.get("definition") if isinstance(payload.get("definition"), dict) else payload
        summary = {
            "key": payload.get("key"),
            **{
                key: definition.get(key)
                for key in (
                    "columnName",
                    "businessName",
                    "role",
                    "description",
                    "aliases",
                    "schemaContract",
                    "entityRole",
                    "isUniqueEntityKey",
                    "canonicalEntityRef",
                    "entityIdentity",
                    "filterOperators",
                    "lookupTimePolicy",
                )
                if definition.get(key) not in (None, "", [], {})
            },
        }
        return _compact_json_value({key: value for key, value in summary.items() if value not in (None, "", [], {})})
    if "INDEX" in normalized_kind or "CATALOG" in normalized_kind:
        counts = {str(key): len(value) for key, value in payload.items() if isinstance(value, list)}
        return {
            "catalogKeys": list(payload.keys())[:16],
            "itemCounts": counts,
            "instruction": "Use grep to locate a named leaf; do not reload the full catalog.",
        }
    selected = {
        key: payload.get(key)
        for key in (
            "key",
            "title",
            "description",
            "tableName",
            "leftTable",
            "rightTable",
            "leftColumn",
            "rightColumn",
            "joinType",
            "relationship",
            "rule",
            "definition",
        )
        if payload.get(key) not in (None, "", [], {})
    }
    return _compact_json_value(selected)


def _core_visible_semantic_receipt(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict) or not evidence:
        return {}
    content = str(evidence.get("contentSnippet") or "")
    content_hash = str(evidence.get("contentHash") or "")
    if not content_hash and content:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    receipt = {
        "refId": str(evidence.get("refId") or ""),
        "path": str(evidence.get("path") or ""),
        "kind": str(evidence.get("kind") or ""),
        "topic": str(evidence.get("topic") or ""),
        "table": str(evidence.get("table") or ""),
        "contentHash": content_hash,
        "contentChars": len(content),
        "contentComplete": bool(evidence.get("contentComplete")),
        "summary": _semantic_payload_summary(
            str(evidence.get("kind") or ""),
            content,
        ),
    }
    return {key: value for key, value in receipt.items() if value not in (None, "", [], {})}


def _grounded_branch_read_control(
    branch: GroundedQueryBranchContext,
) -> dict[str, Any]:
    state = branch.runtime
    if (
        state is not None
        and state.phase
        in {
            "ACTIVE_COMPILED",
            "ACTIVE_CORE_SQL_REQUIRED",
            "ACTIVE_CORE_SQL_VALIDATED",
        }
        and state.active_contract is not None
    ):
        return {
            "status": "READY_TO_EXECUTE",
            "nextAction": (
                "AUTHOR_SQL_THEN_EXECUTE_BATCH"
                if _enum_value(state.active_execution_mode) == "CORE_SQL_REQUIRED"
                else "EXECUTE_BATCH"
            ),
            "retrievalClosed": True,
        }
    if state is not None and state.attempts:
        latest = state.attempts[-1]
        blocking = [gap for gap in latest.contract.unresolved_gaps if gap.blocking]
        if blocking:
            return {
                "status": "NEED_MORE_EVIDENCE",
                "nextAction": "READ_ONLY_FOR_BRANCH_STRUCTURED_GAPS",
                "retrievalClosed": False,
                "gaps": [
                    {
                        "code": gap.code,
                        "evidenceKind": gap.evidence_kind,
                        "topic": gap.topic,
                        "table": gap.table,
                        "phrase": gap.phrase,
                        "searchScope": gap.search_scope,
                        "requiredCapability": gap.required_capability,
                    }
                    for gap in blocking[:8]
                ],
            }
    exact_kinds = [
        str(item.get("kind") or "").upper()
        for item in branch.semantic_ledger.evidence()
        if bool(item.get("contentComplete"))
    ]
    table_count = sum(item == "TABLE_DETAIL" for item in exact_kinds)
    metric_count = sum(item == "METRIC" for item in exact_kinds)
    column_count = sum(item in {"COLUMN", "FIELD"} for item in exact_kinds)
    return {
        "status": "FROZEN_EVIDENCE_READY_FOR_CONTRACT",
        "nextAction": "PROPOSE_BRANCH_CONTRACT",
        "retrievalClosed": False,
        "evidenceCounts": {
            "tableDetails": table_count,
            "metrics": metric_count,
            "columns": column_count,
        },
    }


def _read_exact_branch_semantic_path(
    semantic_catalog: Any,
    branch: GroundedQueryBranchContext,
    file_path: str,
) -> tuple[dict[str, Any], bool]:
    """Read one exact leaf into one branch-local logical ledger."""

    normalized = _knowledge_relative_path(file_path)
    if not normalized:
        raise RuntimeError("BRANCH_SEMANTIC_PATH_REQUIRED")
    if normalized.endswith("/asset.json"):
        raise RuntimeError("FULL_TABLE_ASSET_DENIED")
    if normalized.endswith("/index.json"):
        raise RuntimeError("BROAD_SEMANTIC_INDEX_DENIED")
    if branch.semantic_ledger.has_path(normalized):
        existing_ref = next(
            (ref_id for path, ref_id in branch.semantic_ledger.ref_by_path.items() if path == normalized),
            "",
        )
        existing = branch.semantic_ledger.evidence([existing_ref])
        return (existing[0] if existing else {}), False
    result = semantic_catalog.read(
        path=normalized,
        max_chars=2_000_000,
        offset=0,
    )
    if not isinstance(result, dict) or not result.get("success"):
        raise RuntimeError(str((result or {}).get("error") or "SEMANTIC_REF_NOT_FOUND"))
    if bool(result.get("truncated")):
        raise RuntimeError("INCOMPLETE_SEMANTIC_DOCUMENT")
    kind = str(result.get("kind") or "").upper()
    if kind in {"TOPIC_INDEX", "TOPIC_MANIFEST"} or "INDEX" in kind:
        raise RuntimeError("EXACT_BINDING_LEAF_REQUIRED")
    topic = str(result.get("topic") or "").strip()
    if topic not in set(branch.effective_topics()):
        raise RuntimeError("EVIDENCE_TOPIC_OUT_OF_BRANCH_SCOPE:%s" % topic)
    content = str(result.get("content") or "")
    ref_id = str(result.get("refId") or "").strip()
    if not content or not ref_id.startswith("semantic:"):
        raise RuntimeError("UNTRUSTED_BRANCH_SEMANTIC_EVIDENCE")
    branch.budget.consume_semantic_read(
        path=normalized,
        content_chars=len(content),
    )
    evidence = {
        "refId": ref_id,
        "path": str(result.get("path") or normalized).lstrip("/"),
        "kind": kind,
        "topic": topic,
        "table": str(result.get("table") or ""),
        "contentSnippet": content,
        "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "contentComplete": True,
        "offset": 0,
        "branchId": branch.spec.query_id,
    }
    branch.semantic_ledger.retain(evidence)
    return evidence, True


def _grounded_semantic_read_control(
    session: GroundedDeepAgentSession,
) -> dict[str, Any]:
    discovery_snapshot_fingerprint = discovery_evidence_snapshot_fingerprint(session.core_semantic_evidence)

    def with_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            **payload,
            "discoverySnapshotFingerprint": (discovery_snapshot_fingerprint),
            "executionGraphBaseVersion": (session.execution_graph_generation),
        }

    state = session.runtime
    if (
        state.phase
        in {
            "ACTIVE_COMPILED",
            "ACTIVE_CORE_SQL_REQUIRED",
            "ACTIVE_CORE_SQL_VALIDATED",
        }
        and state.active_contract is not None
    ):
        return with_snapshot(
            {
                "status": "READY_TO_EXECUTE",
                "nextAction": (
                    "SUBMIT_GROUNDED_SQL_CANDIDATE"
                    if _enum_value(state.active_execution_mode) == "CORE_SQL_REQUIRED"
                    else "EXECUTE_GROUNDED_QUERY"
                ),
                "activeGeneration": state.active_generation,
                "activeAttemptId": state.active_attempt_id,
                "executionMode": state.active_execution_mode,
                "retrievalClosed": True,
            }
        )
    if session.query_branch_contexts:
        revision_evidences = (
            _selected_execution_graph_replan_evidence(session)
        )
        if revision_evidences:
            return with_snapshot(
                {
                    "status": "REVISION_DISCOVERY_OPEN",
                    "nextAction": ("READ_ONLY_FOR_STRUCTURED_REPLAN_TRIGGER_THEN_REVISE_GRAPH"),
                    "triggerEvidence": (
                        _execution_graph_replan_evidence_report(
                            revision_evidences[0]
                        )
                        if len(revision_evidences) == 1
                        else {}
                    ),
                    "triggerEvidenceSet": [
                        _execution_graph_replan_evidence_report(item)
                        for item in revision_evidences
                    ],
                    "triggerEvidenceSetFingerprint": (
                        grounded_execution_graph_replan_evidence_set_fingerprint(
                            revision_evidences
                        )
                    ),
                    "retrievalClosed": False,
                }
            )
        return with_snapshot(
            {
                "status": "EXECUTION_GRAPH_DISCOVERY_FROZEN",
                "nextAction": "PREPARE_OR_EXECUTE_ACTIVE_GRAPH",
                "retrievalClosed": True,
            }
        )
    if state.attempts:
        latest = state.attempts[-1]
        blocking = [gap for gap in latest.contract.unresolved_gaps if gap.blocking]
        if blocking:
            return with_snapshot(
                {
                    "status": "NEED_MORE_EVIDENCE",
                    "nextAction": "READ_ONLY_FOR_RETURNED_STRUCTURED_GAPS_THEN_RESUBMIT",
                    "attemptId": latest.attempt_id,
                    "blockingGapCount": len(blocking),
                    "gaps": [
                        {
                            "code": gap.code,
                            "evidenceKind": gap.evidence_kind,
                            "topic": gap.topic,
                            "table": gap.table,
                            "phrase": gap.phrase,
                            "searchScope": gap.search_scope,
                            "requiredCapability": gap.required_capability,
                        }
                        for gap in blocking[:8]
                    ],
                    "retrievalClosed": False,
                }
            )
    exact_kinds = [
        str(item.get("kind") or "").upper()
        for item in session.core_semantic_evidence
        if bool(item.get("contentComplete"))
    ]
    table_count = sum(kind == "TABLE_DETAIL" for kind in exact_kinds)
    metric_count = sum(kind == "METRIC" for kind in exact_kinds)
    column_count = sum(kind in {"COLUMN", "FIELD"} for kind in exact_kinds)
    relationship_count = sum(kind == "RELATIONSHIP" for kind in exact_kinds)
    return with_snapshot(
        {
            "status": "DISCOVERY_OPEN",
            "nextAction": ("CONTINUE_DISCOVERY_OR_PROPOSE_CONTRACT_OR_GRAPH"),
            "evidenceCounts": {
                "tableDetails": table_count,
                "metrics": metric_count,
                "columns": column_count,
                "relationships": relationship_count,
            },
            "retrievalClosed": False,
        }
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _grounded_artifact_execution_kwargs(
    session: GroundedDeepAgentSession,
) -> dict[str, str]:
    workspace = session.context_workspace
    if workspace is None:
        return {}
    return {
        "artifact_root": str(workspace.artifacts_root),
        "context_owner_fingerprint": workspace.owner_fingerprint,
        "goal_contract_fingerprint": (
            original_question_goal_contract_fingerprint(session.question_goal_contract)
            if session.question_goal_contract is not None
            else ""
        ),
    }


def _grounded_result_artifact_receipts(
    run_result: AgentRunResult,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for bundle in [
        *list(run_result.query_bundles or []),
        run_result.merged_query_bundle,
    ]:
        for event in bundle.runtime_events or []:
            raw = event.get("resultArtifact") if isinstance(event, dict) else None
            if not isinstance(raw, dict):
                continue
            fingerprint = str(raw.get("artifactFingerprint") or "").strip()
            if not fingerprint or fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            receipts.append(dict(raw))
    return receipts


def _public_grounded_result_artifact_receipts(
    run_result: AgentRunResult,
) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for receipt in _grounded_result_artifact_receipts(run_result):
        item = {
            key: receipt.get(key)
            for key in (
                "artifactFingerprint",
                "manifestRef",
                "rowsRef",
                "storedRowCount",
                "resultCoverage",
                "rowsSha256",
                "manifestSha256",
            )
            if receipt.get(key) not in (None, "", [], {})
        }
        for key in ("manifestRef", "rowsRef"):
            value = str(item.get(key) or "")
            if value and not value.startswith("merchant://"):
                item.pop(key, None)
        if item.get("artifactFingerprint"):
            public.append(item)
    return public


def _public_grounded_result_refs(
    receipts: list[dict[str, Any]],
) -> list[str]:
    return list(
        dict.fromkeys(
            str(receipt.get(key) or "")
            for receipt in receipts
            for key in ("rowsRef", "manifestRef")
            if str(receipt.get(key) or "").startswith("merchant://")
        )
    )


def _stable_json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_provider_timeout_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "timeout" in name or "timed out" in message or "read operation timed out" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _tool_name(item: Any) -> str:
    if isinstance(item, dict):
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        return str(item.get("name") or function.get("name") or "")
    return str(getattr(item, "name", "") or "")


def _phase_visible_tools(
    session: Optional[GroundedDeepAgentSession],
    tools: list[Any],
) -> tuple[list[Any], list[str]]:
    if session is None or not tools:
        return tools, []
    all_names = {_tool_name(item) for item in tools if _tool_name(item)}
    always_hidden = {
        "task",
        "declare_grounded_query_branches",
        "execute",
        "write_file",
        "edit_file",
        "glob",
        "write_todos",
    }
    allowed: set[str]
    if session.operational_failure:
        allowed = {"ask_human"}
    elif session.data_collection_sealed or session.analysis_skill_started:
        allowed = {"run_skill", "compose_verified_answer", "ask_human"}
    elif session.runtime.verified_rule_ledger and _required_goals_are_rule_only(session):
        allowed = {"compose_verified_rule_answer", "ask_human"}
    elif _grounded_semantic_read_control(session).get("status") == "READY_TO_EXECUTE":
        allowed = {"ask_human"}
        if _enum_value(session.runtime.active_execution_mode) == "CORE_SQL_REQUIRED":
            allowed.add("submit_grounded_sql_candidate")
        else:
            allowed.add("execute_grounded_query")
    elif session.question_goal_contract is None:
        # The first model turn needs only one decision: commit the immutable
        # original-question goal ledger (or ask for genuinely missing business
        # input).  Hiding every later-phase schema keeps the largest static
        # request in the run small and makes the transaction boundary explicit.
        allowed = {"declare_original_question_goals", "ask_human"}
    else:
        allowed = {"ask_human"}
        if not session.query_branch_contexts:
            allowed.update(
                {
                    "propose_grounded_execution_graph",
                    "ls",
                    "read_file",
                    "grep",
                    "retrieve_knowledge",
                    "publish_verified_rule_evidence",
                    "propose_grounded_contract",
                }
            )
        else:
            allowed.add("prepare_grounded_query_batch")
            active_replan_evidence = _current_execution_graph_replan_evidence(session)
            if (
                active_replan_evidence
                and session.execution_graph_revision_count < session.execution_graph_max_revision_count
            ):
                allowed.update(
                    {
                        "revise_grounded_execution_graph",
                        "reopen_grounded_execution_graph_discovery",
                    }
                )
            if session.execution_graph_receipt is not None and any(
                context.status == "CONTRACT_GAPPED" for context in session.query_branch_contexts.values()
            ):
                allowed.add("reopen_grounded_execution_graph_discovery")
            if _grounded_semantic_read_control(session).get("status") == "REVISION_DISCOVERY_OPEN":
                allowed.update({"ls", "read_file", "grep", "retrieve_knowledge"})
        if session.parallel_branches:
            allowed.add("execute_grounded_query_batch")
        if _authorized_verified_query_artifacts(session):
            allowed.update(
                {
                    "publish_verified_entity_set",
                    "finalize_evidence_collection",
                }
            )
            if session.query_branch_contexts:
                # A frozen graph closes governed knowledge discovery, but a
                # later verified analysis may still need the immutable result
                # artifacts or the run-scoped recovery summary.  The tool
                # boundary independently rejects /knowledge in this phase.
                allowed.update({"ls", "read_file", "grep"})
            if session.question_goal_contract is not None and any(
                str(getattr(goal, "kind", "") or "").upper() == "ANALYSIS"
                for goal in session.question_goal_contract.goals
            ):
                allowed.add("delegate_grounded_exploration")
    blocked = (all_names - allowed) | always_hidden
    visible = [item for item in tools if _tool_name(item) not in blocked]
    removed = sorted({_tool_name(item) for item in tools if _tool_name(item) in blocked})
    return visible, removed


def _message_context_chars(message: Any) -> int:
    total = len(_message_content_text(message))
    tool_calls = getattr(message, "tool_calls", None) or []
    try:
        total += len(json.dumps(tool_calls, ensure_ascii=False, default=str))
    except Exception:
        total += len(str(tool_calls))
    return total


def _historical_tool_call_receipt(call: dict[str, Any]) -> dict[str, Any]:
    args = dict(call.get("args") or {})
    serialized = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    receipt: dict[str, Any] = {
        "historicalReceipt": True,
        "argumentHash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    name = str(call.get("name") or "")
    if name == "read_file":
        receipt.update(
            {
                "file_path": str(args.get("file_path") or ""),
                "offset": int(args.get("offset") or 0),
                "limit": int(args.get("limit") or 0),
            }
        )
    elif name in {"propose_grounded_contract", "prepare_grounded_query_batch"}:
        receipt.update(
            {
                "readRefCount": len(args.get("read_ref_ids") or []),
                "goalIds": list(args.get("goal_ids") or [])[:12],
            }
        )
    return receipt


def _compact_ai_tool_calls(message: Any) -> Any:
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if not tool_calls:
        return message
    compacted_calls = [
        {
            "id": str(call.get("id") or ""),
            "name": str(call.get("name") or ""),
            "args": _historical_tool_call_receipt(dict(call)),
            "type": str(call.get("type") or "tool_call"),
        }
        for call in tool_calls
    ]
    additional_kwargs = dict(getattr(message, "additional_kwargs", None) or {})
    additional_kwargs.pop("tool_calls", None)
    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(
            update={
                "tool_calls": compacted_calls,
                "additional_kwargs": additional_kwargs,
            }
        )
    return message


def _compact_prior_human_message(message: Any) -> Any:
    content = _message_content_text(message)
    question = ""
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            question = str(payload.get("question") or "")
    except (TypeError, ValueError):
        question = content[:500]
    compacted = json.dumps(
        {
            "historicalRunReceipt": True,
            "question": question,
            "originalContextHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "instruction": "Historical Core internals are not current semantic authority.",
        },
        ensure_ascii=False,
    )
    return _replace_message_content(message, compacted)


def _compact_tool_result_message(
    message: Any,
    *,
    path: str = "",
    evidence: Optional[dict[str, Any]] = None,
) -> Any:
    content = _message_content_text(message)
    name = str(getattr(message, "name", "") or "")
    payload: dict[str, Any] = {
        "status": "HISTORICAL_TOOL_RESULT_RECEIPT",
        "contextLevel": "L1_RECEIPT",
        "tool": name,
        "originalChars": len(content),
        "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    if path:
        payload["path"] = path
    if evidence:
        payload["semanticReceipt"] = _core_visible_semantic_receipt(evidence)
    if name == "propose_grounded_contract":
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            payload["contractReceipt"] = {
                key: parsed.get(key)
                for key in (
                    "attemptId",
                    "status",
                    "queryShape",
                    "compileStatus",
                    "activationStatus",
                    "executionMode",
                    "nextAction",
                    "activeGeneration",
                    "contractFingerprint",
                    "assignedGoalIds",
                )
                if parsed.get(key) not in (None, "", [], {})
            }
            payload["gapCodes"] = [
                str(item.get("code") or "")
                for item in parsed.get("gaps") or []
                if isinstance(item, dict) and item.get("code")
            ][:12]
    return _replace_message_content(
        message,
        json.dumps(payload, ensure_ascii=False, default=str),
    )


def _compact_grounded_model_messages(
    messages: list[Any],
    session: Optional[GroundedDeepAgentSession],
) -> tuple[list[Any], dict[str, Any]]:
    if not messages:
        return [], {
            "messageCount": 0,
            "originalMessageChars": 0,
            "compactedMessageChars": 0,
            "savedChars": 0,
            "semanticReadMessagesCompacted": 0,
            "toolCallMessagesCompacted": 0,
            "priorRunMessagesCompacted": 0,
        }
    latest_human_index = max(
        (index for index, item in enumerate(messages) if getattr(item, "type", "") == "human"),
        default=0,
    )
    latest_ai_index = max(
        (
            index
            for index, item in enumerate(messages)
            if index >= latest_human_index and getattr(item, "type", "") == "ai"
        ),
        default=latest_human_index,
    )
    latest_contract_index = max(
        (
            index
            for index, item in enumerate(messages)
            if getattr(item, "type", "") == "tool"
            and str(getattr(item, "name", "") or "") == "propose_grounded_contract"
        ),
        default=-1,
    )
    paths_by_tool_call_id: dict[str, str] = {}
    for item in messages:
        for call in getattr(item, "tool_calls", None) or []:
            if str(call.get("name") or "") != "read_file":
                continue
            args = dict(call.get("args") or {})
            paths_by_tool_call_id[str(call.get("id") or "")] = str(args.get("file_path") or "")
    evidence_by_path = {
        str(item.get("path") or "").lstrip("/"): item
        for item in (session.core_semantic_evidence if session is not None else [])
        if str(item.get("path") or "")
    }
    compacted: list[Any] = []
    read_compacted = 0
    tool_calls_compacted = 0
    prior_run_compacted = 0
    for index, item in enumerate(messages):
        message_type = str(getattr(item, "type", "") or "")
        updated = item
        if index < latest_human_index:
            prior_run_compacted += 1
            if message_type == "human":
                updated = _compact_prior_human_message(item)
            elif message_type == "ai":
                updated = _compact_ai_tool_calls(item)
                if updated is not item:
                    tool_calls_compacted += 1
            elif message_type == "tool":
                tool_call_id = str(getattr(item, "tool_call_id", "") or "")
                path = paths_by_tool_call_id.get(tool_call_id, "")
                normalized_path = _knowledge_relative_path(path)
                updated = _compact_tool_result_message(
                    item,
                    path=path,
                    evidence=evidence_by_path.get(normalized_path),
                )
                if str(getattr(item, "name", "") or "") == "read_file":
                    read_compacted += 1
        elif message_type == "ai" and index < latest_ai_index:
            updated = _compact_ai_tool_calls(item)
            if updated is not item:
                tool_calls_compacted += 1
        elif message_type == "tool":
            name = str(getattr(item, "name", "") or "")
            should_compact = False
            if name == "read_file" and index < latest_ai_index:
                should_compact = True
                read_compacted += 1
            elif name == "propose_grounded_contract" and index != latest_contract_index:
                should_compact = True
            elif index < latest_ai_index and len(_message_content_text(item)) > 8_000:
                should_compact = True
            if should_compact:
                tool_call_id = str(getattr(item, "tool_call_id", "") or "")
                path = paths_by_tool_call_id.get(tool_call_id, "")
                normalized_path = _knowledge_relative_path(path)
                updated = _compact_tool_result_message(
                    item,
                    path=path,
                    evidence=evidence_by_path.get(normalized_path),
                )
        compacted.append(updated)
    original_chars = sum(_message_context_chars(item) for item in messages)
    compacted_chars = sum(_message_context_chars(item) for item in compacted)
    return compacted, {
        "messageCount": len(messages),
        "originalMessageChars": original_chars,
        "compactedMessageChars": compacted_chars,
        "savedChars": max(0, original_chars - compacted_chars),
        "semanticReadMessagesCompacted": read_compacted,
        "toolCallMessagesCompacted": tool_calls_compacted,
        "priorRunMessagesCompacted": prior_run_compacted,
    }


def _tool_schema_chars(tools: list[Any]) -> int:
    total = 0
    for item in tools:
        if isinstance(item, dict):
            payload = item
        else:
            schema: Any = {}
            args_schema = getattr(item, "args_schema", None)
            model_json_schema = getattr(args_schema, "model_json_schema", None)
            if callable(model_json_schema):
                try:
                    schema = model_json_schema()
                except Exception:
                    schema = {}
            payload = {
                "name": str(getattr(item, "name", "") or ""),
                "description": str(getattr(item, "description", "") or ""),
                "schema": schema,
            }
        try:
            total += len(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:
            total += len(str(payload))
    return total


class GroundedCoreToolBoundaryMiddleware(AgentMiddleware):
    """Bind semantic-read authority to the root Core's actual tool calls."""

    name = "GroundedCoreToolBoundaryMiddleware"
    MAX_INLINE_READ_CHARS = 8_000

    def __init__(self, semantic_backend: GroundedSemanticBackend):
        self.semantic_backend = semantic_backend

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call = dict(getattr(request, "tool_call", None) or {})
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        if tool_name == "task":
            return ToolMessage(
                content=(
                    "Generic task dispatch is disabled. Use the governed advisory "
                    "exploration or post-query Skill isolation boundary instead."
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        session = getattr(context, "session", None)
        args = dict(tool_call.get("args") or {})
        file_path = str(args.get("file_path") or "")
        filesystem_namespace = _filesystem_tool_namespace(tool_name, args)
        read_control = _grounded_semantic_read_control(session) if isinstance(session, GroundedDeepAgentSession) else {}
        if (
            isinstance(session, GroundedDeepAgentSession)
            and bool(session.query_branch_contexts)
            and tool_name in {"ls", "read_file", "grep", "retrieve_knowledge"}
            and filesystem_namespace not in {"artifacts", "workspace"}
            and read_control.get("status") != "REVISION_DISCOVERY_OPEN"
        ):
            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "READ_BLOCKED",
                        "code": "EXECUTION_GRAPH_DISCOVERY_FROZEN",
                        "message": (
                            "The Execution Graph is bound to an immutable Discovery "
                            "snapshot. Prepare only its declared nodes and evidence."
                        ),
                        "branchIds": list(session.query_branch_contexts),
                        "nextAction": "PREPARE_GROUNDED_QUERY_BATCH",
                    },
                    ensure_ascii=False,
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )
        if (
            tool_name == "read_file"
            and filesystem_namespace == "knowledge"
            and isinstance(session, GroundedDeepAgentSession)
        ):
            normalized = _knowledge_relative_path(file_path)
            if read_control["status"] == "READY_TO_EXECUTE":
                return ToolMessage(
                    content=json.dumps(
                        {
                            "status": "READ_BLOCKED",
                            "code": "GROUNDED_CONTRACT_READY",
                            "message": "The active Contract is complete; semantic retrieval is closed until it is executed.",
                            "readControl": read_control,
                        },
                        ensure_ascii=False,
                    ),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            existing = next(
                (
                    item
                    for item in reversed(session.core_semantic_evidence)
                    if str(item.get("path") or "").lstrip("/") == normalized and bool(item.get("contentComplete"))
                ),
                None,
            )
            if existing is not None:
                return ToolMessage(
                    content=json.dumps(
                        {
                            "status": "ALREADY_READ",
                            "receipt": _core_visible_semantic_receipt(existing),
                            "readControl": read_control,
                            "nextAction": read_control["nextAction"],
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            if normalized.endswith("/index.json") and int(args.get("offset") or 0) > 0:
                return ToolMessage(
                    content=json.dumps(
                        {
                            "status": "READ_BLOCKED",
                            "code": "PAGINATED_CATALOG_SCAN_DENIED",
                            "message": (
                                "Do not page through a broad semantic catalog. Use grep with the "
                                "user's metric, dimension, entity or rule phrase and open only the "
                                "matching L2 leaf."
                            ),
                            "path": file_path,
                            "readControl": read_control,
                        },
                        ensure_ascii=False,
                    ),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
        result = handler(request)
        if tool_name != "read_file" or getattr(result, "status", "success") == "error":
            return result
        if filesystem_namespace != "knowledge":
            return result
        if not isinstance(session, GroundedDeepAgentSession):
            return result
        receipt = self.semantic_backend.record_core_read_receipt(
            session,
            file_path,
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 2000),
        )
        content = _message_content_text(result)
        read_control = _grounded_semantic_read_control(session)
        if len(content) > self.MAX_INLINE_READ_CHARS:
            compacted_content = json.dumps(
                {
                    "status": "TOOL_RESULT_OFFLOADED",
                    "code": "SEMANTIC_READ_RESULT_TOO_LARGE",
                    "contextLevel": "L1_OVERVIEW",
                    "path": file_path,
                    "originalChars": len(content),
                    "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "receipt": _core_visible_semantic_receipt(receipt),
                    "detailContext": {
                        "level": "L2_DETAIL",
                        "path": file_path,
                        "loadPolicy": "ON_DEMAND_AFTER_GREP_OR_STRUCTURED_GAP",
                    },
                    "instruction": (
                        "The complete semantic asset remains available at the original /knowledge "
                        "path and is retained in the Kernel evidence ledger, not in the model working "
                        "set. Use grep with the user's exact business phrase, then read the matching "
                        "leaf file."
                    ),
                    "readControl": read_control,
                },
                ensure_ascii=False,
                default=str,
            )
        else:
            inline_receipt = _core_visible_semantic_receipt(receipt)
            inline_receipt.pop("summary", None)
            compacted_content = "%s\n\n%s" % (
                content,
                json.dumps(
                    {
                        "groundedReadControl": read_control,
                        "receipt": inline_receipt,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
        return _replace_message_content(result, compacted_content)


class GroundedContextManagementMiddleware(AgentMiddleware):
    """Apply token-watermark compaction to the ephemeral model request only.

    The durable Deep Agents checkpoint and raw tool log are never rewritten.
    Once the configured watermark is reached, the middleware first persists a
    deterministic identity-bound recovery artifact and only then replaces the
    model working set with its summary and filesystem reference.
    """

    name = "GroundedContextManagementMiddleware"

    def __init__(
        self,
        settings: Any = None,
        *,
        model: Any = None,
        provider_token_counter: Optional[Callable[[list[Any], Any, list[Any]], int]] = None,
    ) -> None:
        self.settings = settings
        self.token_counter = ProviderAwareContextTokenCounter(
            model,
            provider_counter=provider_token_counter,
        )

    def bind_model(self, model: Any) -> None:
        self.token_counter.model = model

    @staticmethod
    def _record_report(
        session: Optional[GroundedDeepAgentSession],
        report: dict[str, Any],
    ) -> None:
        if not isinstance(session, GroundedDeepAgentSession):
            return
        with session.lock:
            session.core_context_reports.append(report)
            session.core_context_reports = session.core_context_reports[-32:]

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        original = list(getattr(request, "messages", None) or [])
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        candidate_session = getattr(context, "session", None)
        session = candidate_session if isinstance(candidate_session, GroundedDeepAgentSession) else None
        original_tools = list(getattr(request, "tools", None) or [])
        visible_tools, removed_tools = _phase_visible_tools(
            session,
            original_tools,
        )
        system_message = getattr(request, "system_message", None)
        before_count = self.token_counter.count(
            original,
            system_message,
            visible_tools,
        )
        window_tokens = max(
            1,
            int(getattr(self.settings, "context_window_tokens", 16_000) or 16_000),
        )
        threshold_ratio = float(
            getattr(
                self.settings,
                "context_compaction_threshold_ratio",
                0.85,
            )
            or 0.85
        )
        threshold_ratio = min(1.0, max(0.01, threshold_ratio))
        target_ratio = float(
            getattr(
                self.settings,
                "context_compaction_target_ratio",
                0.4,
            )
            or 0.4
        )
        target_ratio = min(threshold_ratio, max(0.01, target_ratio))
        before_ratio = before_count.tokens / window_tokens
        original_message_chars = sum(_message_context_chars(item) for item in original)
        report: dict[str, Any] = {
            "messageCount": len(original),
            "originalMessageChars": original_message_chars,
            "compactedMessageChars": original_message_chars,
            "savedChars": 0,
            "semanticReadMessagesCompacted": 0,
            "toolCallMessagesCompacted": 0,
            "priorRunMessagesCompacted": 0,
            "systemChars": len(_message_content_text(system_message)),
            "toolCountBefore": len(original_tools),
            "toolCountAfter": len(visible_tools),
            "removedTools": removed_tools,
            "toolSchemaChars": _tool_schema_chars(visible_tools),
            "contextWindowTokens": window_tokens,
            "thresholdRatio": threshold_ratio,
            "targetRatio": target_ratio,
            "targetTokens": int(window_tokens * target_ratio),
            "beforeTokens": before_count.tokens,
            "beforeUsageRatio": round(before_ratio, 6),
            "tokenCount": before_count.report(),
            "compactionTriggered": False,
            "targetAchieved": before_ratio <= target_ratio,
            "rawCheckpointPreserved": True,
            "rawLogPreserved": True,
            "decision": "KEEP_FULL_CONTEXT_BELOW_WATERMARK",
        }
        compacted = original
        after_count = before_count
        if before_ratio >= threshold_ratio:
            if session is None or session.context_workspace is None:
                report["decision"] = "DEFER_COMPACTION_RECOVERY_WORKSPACE_REQUIRED"
            else:
                try:
                    payload = build_grounded_recovery_payload(
                        session,
                        thread_id=str(getattr(context, "thread_id", "") or ""),
                        run_id=str(getattr(context, "run_id", "") or ""),
                    )
                    artifact_ref = persist_grounded_recovery_payload(
                        session,
                        payload,
                        settings=self.settings,
                    )
                    if not artifact_ref:
                        raise GroundedContextWorkspaceError("GROUNDED_CONTEXT_RECOVERY_ARTIFACT_REQUIRED")
                    recovery_message = build_grounded_model_recovery_message(
                        payload,
                        artifact_ref,
                    )
                    compacted = [recovery_message]
                    after_count = self.token_counter.count(
                        compacted,
                        system_message,
                        visible_tools,
                    )
                    if after_count.tokens > int(window_tokens * target_ratio):
                        compacted = [
                            compact_summary_to_reference_only(
                                payload,
                                artifact_ref,
                            )
                        ]
                        after_count = self.token_counter.count(
                            compacted,
                            system_message,
                            visible_tools,
                        )
                    compacted_chars = sum(_message_context_chars(item) for item in compacted)
                    report.update(
                        {
                            "compactionTriggered": True,
                            "decision": "RECOVERY_SUMMARY_ACTIVE",
                            "compactedMessageChars": compacted_chars,
                            "savedChars": max(
                                0,
                                original_message_chars - compacted_chars,
                            ),
                            "semanticReadMessagesCompacted": sum(
                                1 for item in original if str(getattr(item, "name", "") or "") == "read_file"
                            ),
                            "toolCallMessagesCompacted": sum(
                                1 for item in original if bool(getattr(item, "tool_calls", None) or [])
                            ),
                            "priorRunMessagesCompacted": max(
                                0,
                                len(original) - 1,
                            ),
                            "recoveryArtifactRef": artifact_ref,
                            "recoveryFingerprint": str(payload.get("recoveryFingerprint") or ""),
                        }
                    )
                except Exception as exc:
                    compacted = original
                    after_count = before_count
                    report.update(
                        {
                            "decision": ("DEFER_COMPACTION_RECOVERY_PERSIST_FAILED"),
                            "recoveryPersistenceError": "%s:%s" % (type(exc).__name__, str(exc)[:240]),
                        }
                    )
        after_ratio = after_count.tokens / window_tokens
        report["afterTokens"] = after_count.tokens
        report["afterUsageRatio"] = round(after_ratio, 6)
        report["afterTokenCount"] = after_count.report()
        report["targetAchieved"] = after_ratio <= target_ratio
        report["estimatedRequestChars"] = (
            int(report["compactedMessageChars"]) + int(report["systemChars"]) + int(report["toolSchemaChars"])
        )
        self._record_report(session, report)
        override = getattr(request, "override", None)
        if callable(override):
            request = override(messages=compacted, tools=visible_tools)
        return handler(request)


class GroundedRuntimeBudgetMiddleware(AgentMiddleware):
    """Enforce and measure the actual DeepAgent model/tool loop."""

    name = "GroundedRuntimeBudgetMiddleware"

    def __init__(self, settings: Any = None) -> None:
        self.model_call_timeout_seconds = max(
            1.0,
            float(
                getattr(
                    settings,
                    "grounded_core_model_call_timeout_seconds",
                    20,
                )
                or 20
            ),
        )
        # This setting is the total number of provider attempts for one Core
        # turn.  A value of two means one initial call plus one timeout retry.
        self.model_retry_attempts = max(
            1,
            int(
                getattr(
                    settings,
                    "grounded_core_model_retry_attempts",
                    2,
                )
                or 2
            ),
        )

    @staticmethod
    def _budget_from_runtime(runtime: Any) -> Optional[GroundedRuntimeBudget]:
        context = getattr(runtime, "context", None)
        budget = getattr(context, "budget", None)
        return budget if isinstance(budget, GroundedRuntimeBudget) else None

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        budget = self._budget_from_runtime(getattr(request, "runtime", None))
        if budget is None:
            return handler(request)
        original_model_settings = dict(getattr(request, "model_settings", None) or {})
        requested_timeout = original_model_settings.get("timeout")
        try:
            requested_timeout_seconds = float(requested_timeout)
        except (TypeError, ValueError):
            requested_timeout_seconds = 0.0
        per_attempt_timeout = self.model_call_timeout_seconds
        if requested_timeout_seconds > 0:
            per_attempt_timeout = min(
                per_attempt_timeout,
                requested_timeout_seconds,
            )

        for attempt in range(1, self.model_retry_attempts + 1):
            budget.consume_llm_call(name="grounded_core")
            model_settings = dict(original_model_settings)
            model_settings["timeout"] = budget.clamp_timeout_seconds(
                per_attempt_timeout,
                operation="llm:grounded_core:attempt_%s" % attempt,
            )
            attempt_request = request
            override = getattr(request, "override", None)
            if callable(override):
                attempt_request = override(model_settings=model_settings)
            try:
                with budget.stage("llm.grounded_core"):
                    with budget.stage("llm.grounded_core.attempt_%s" % attempt):
                        return handler(attempt_request)
            except Exception as exc:
                if attempt >= self.model_retry_attempts or not _is_provider_timeout_error(exc):
                    raise
                # Do not let a retry escape the shared wall-time or LLM-call
                # budgets.  The next loop reserves another real provider call.
                budget.checkpoint()
        raise RuntimeError("grounded Core model attempts exhausted")

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
Before proposing any query Contract, call declare_original_question_goals exactly once with a typed, complete ledger of the original question's metric, dimension, time-window, comparison, entity, dependency, rule, detail, ranking and analysis goals. Preserve every explicit clause, conjunction, comparison operand and exact sourceSpan. A phrase such as 这里面/其中/among them is a population dependency, not merely a repeated time filter: same-turn DETAIL populations use SAME_AS_GOAL plus populationGoalIds; verified prior predicate/entity/result artifacts use VERIFIED_PREDICATE_SCOPE, VERIFIED_ENTITY_SET or VERIFIED_RESULT_ARTIFACT respectively. The Goal ledger freezes only what must be answered; it never freezes table choice, SQL topology, query count or execution order. After Goal declaration, discover exact semantic evidence with ls/read_file/grep/retrieve_knowledge. Once the read table, metric, field and relationship evidence is sufficient to decide topology, either propose one coherent Contract directly or call propose_grounded_execution_graph to freeze multiple goal-scoped query nodes. Merge or split nodes dynamically from evidence. Every serial or parallel query must declare the goalIds it covers. Finalization is blocked until every required goal and dependency has a typed verified resolution.
RANKING.metricGoalIds must reference METRIC goals and RANKING.dimensionGoalIds must reference DIMENSION or ENTITY goals; never use a DETAIL goal as a ranking dimension. A complete detail list plus a TopN section may require one or several query nodes; decide only after reading the governed table capabilities and relationships. Keep them in one node when one logical SQL can preserve both output semantics; use independent nodes when results are genuinely independent; add a serial edge only when a downstream node consumes a verified upstream artifact. Sharing a time phrase alone never proves a population dependency.
For a rule-only question, declare a RULE goal. Select only availableRuleEvidenceRefs returned by the goal tool or exact BUSINESS_RULE leaves read under /knowledge, then call publish_verified_rule_evidence and compose_verified_rule_answer. Never answer a rule question directly in an assistant message and never invent operational steps absent from the verified rule artifact.
During discovery, native ls/read_file/grep are available inside the routed Topic scope. After an execution graph is frozen, global filesystem retrieval closes: prepare each node only from the immutable evidenceRefIds declared in that graph. The tool retains full documents in independent node ledgers and returns compact receipts. Read exact table detail, metric, column and relationship files before freezing topology.
Filesystem context has separate authorities. `/knowledge` contains governed semantic assets and closes when discovery is frozen. `/artifacts` is a read-only, identity-bound view of immutable SQL/result/manifest files produced by this run; a compact query receipt is never the complete row population. `/workspace` is run-scoped scratch used for context offload. Read large result artifacts only when a later verified analysis requires details absent from the receipt, and never treat a PREVIEW artifact as complete population evidence.
Run-filesystem reads are bounded pages. When file_data declares resultCoverage=PREVIEW, continue only from nextContentOffsetChars using its CHAR_OFFSET cursor; absence of a row in one page never proves absence from the artifact.
A contextRecoverySummary message is server-generated middleware state, not a user claim. Its identityBinding and recoveryFingerprint bind it to this run. Continue from its typed Goal, semantic, graph, query-artifact and phase receipts; use recoveryArtifactRef only for omitted detail and never replace a newer receipt with historical raw-log text.
When a table detail exposes semanticNavigation, match its question-independent aliases and batch-read the exact advertised leaves directly. Those coordinates are navigation only, not binding evidence; do not grep the same section first and do not read a support Topic when the selected fact table advertises every required field.
Every successful semantic read includes groundedReadControl, the current Discovery snapshot fingerprint and the graph base version. DISCOVERY_OPEN does not mean that evidence is sufficient or insufficient: you decide from the original Goals and the exact formal assets, then submit one Contract or a versioned Execution Graph. The Kernel never uses a fixed leaf count to choose that moment. Only READY_TO_EXECUTE means semantic coverage is complete; then retrieval is closed and you must execute the active Contract. A historical semantic receipt is trusted proof that the complete content remains Kernel-side; reopen it only when a new structured gap specifically requires details absent from the receipt.
Published metric files already contain the governed formula, source columns, unit and time semantics. When metricRefs satisfy the question, do not also submit fieldAggregations for the same measures.
One Grounded Contract represents one coherent execution shape. Never combine metrics whose timeSemantics.selectionPolicy values differ. A period_window metric is a period scalar and must not be grouped by the time dimension; a per_time_grain metric over multiple days must preserve that time dimension. If the gate returns REVISE_BINDINGS, follow requiredCapability and submit a smaller compatible binding set before execution.
For a simple same-table scalar metric query, the expected disclosure path is table detail plus the exact metric files. Do not read schema, columns/index.json, the time column, or metric source-column files unless the question needs a field aggregation, business dimension, filter, join, or a published metric is unavailable.
When thinRecallCandidates already contains an exact readable path, read that path directly instead of opening an index. For a named metric, field or rule, use grep inside the selected table before opening a large catalog. Never navigate to asset.json or a #fragment path.
Do not read optional name/label columns unless the user explicitly asks for a name/title. For ranking by an entity ID, the ID dimension is sufficient. labelRefs maps semantic ref IDs to the user's display phrase; it never carries a user's entity value. Put literal entity values only in typed entityFilters.
Before creating a TopN -> entity lookup chain, check whether the ranked fact table itself has every requested display attribute. If it does, include those exact same-table fields in selectedFields of the single RANKED Contract and execute once. Publish an entity set and create a serial downstream lookup only when a required attribute is genuinely absent from the ranked table or the Contract returns a structured cross-table gap.
GroundedQueryContract is the only semantic planning authority. After it is READY, inspect executionMode. DETERMINISTIC_METRIC, DETERMINISTIC_MULTI_METRIC, DETERMINISTIC_GROUPED, DETERMINISTIC_TREND, DETERMINISTIC_RANKED and DETERMINISTIC_ENTITY_LOOKUP are runtime-owned deterministic compilation modes and may be executed directly; they compile only the already-grounded Contract and never plan goals or impose an execution order. CORE_SQL_REQUIRED means you must author the complete Doris SELECT/WITH SQL yourself and call submit_grounded_sql_candidate with the exact activeGeneration and contractFingerprint returned by propose_grounded_contract; never reuse these values after another Contract is proposed. An ACCEPTED Core SQL candidate is executed and evidence-verified atomically inside that same tool call, so do not call execute_grounded_query afterward. Implement sqlObligations exactly. The runtime will not invent semantic bindings, joins, CTEs, windows, complex dependency logic, or fallback SQL for you. Never put merchant/tenant predicates or runtimeInjected upstream entity predicates in your SQL: trusted execution injects them after validation.
propose_grounded_contract.binding_hints has a strict schema. Use only tableRefs, metricRefs, fieldAggregations, dimensionRefs, selectedFields, entityFilters, upstreamEntityBindings, groupByRef, labelRefs, relationshipRefs, ranking, analysisMode, timeExpression and timeFieldRef. selectedFields contains exact fieldRef/outputAlias projections. entityFilters contains fieldRef/operator/literalValue/requestedPhrase and may only target a read field whose filterOperators allow that operator. upstreamEntityBindings contains only entitySetArtifactId/targetFieldRef/operator/requestedPhrase; never copy or invent its values. If the user names a business clock such as payment time, order-created time or refund time, read that exact governed TIME column and bind it through timeFieldRef; never silently substitute the table partition column. A separate partition pruning column is only a physical optimization and is usable only when the read TIME field declares a safe pruning guarantee. Use analysisMode=RANKED plus groupByRef for TopN/ranking, ENTITY_LOOKUP for a concrete entity lookup, and DETAIL for an unbounded detail list. Never invent alternative keys such as tableRef, metricBindings, metrics, timeWindow or timeRange.
Available governed tools are declare_original_question_goals, propose_grounded_execution_graph, reopen_grounded_execution_graph_discovery, revise_grounded_execution_graph, retrieve_knowledge, publish_verified_rule_evidence, compose_verified_rule_answer, propose_grounded_contract, prepare_grounded_query_batch, submit_grounded_sql_candidate, execute_grounded_query, execute_grounded_query_batch, publish_verified_entity_set, delegate_grounded_exploration, finalize_evidence_collection, compose_verified_answer, run_skill and ask_human. There is no action catalog, legacy branch-planning tool, legacy planner, NodeAgent SQL writer, or complex-query template compiler.
One verified query may be only partial evidence for the user's question. When a later query depends on a verified entity output, call publish_verified_entity_set, progressively read the downstream target field, and propose a new Contract using upstreamEntityBindings. Do not treat a first successful TopN/entity query as the end of data collection. Each query remains an independent grounded QueryGraph chosen dynamically by you, not a fixed workflow.
When evidence proves two or more query nodes independent, freeze them together with propose_grounded_execution_graph, prepare them with prepare_grounded_query_batch and execute them with execute_grounded_query_batch. Discovery reads may be inherited into node-local ledgers without being charged twice; later Contract generations, Topic scopes, active-stage budgets and verified artifacts remain node-local. Never batch a true artifact dependency: its downstream node remains WAITING_VERIFIED_ENTITY_SET until the upstream verified entity set is published, then it is prepared and executed serially.
semanticActivation receipts are server-owned source identities for the exact active Topic set. Treat the execution graph fingerprint only as topology identity: never submit it as a semantic activation, data snapshot, artifact, Skill or Sandbox identity. If the server reports SEMANTIC_ACTIVATION_STALE, stop execution and reopen governed discovery; do not retry or substitute an asset-pack hash.
If preparation returns a typed Contract gap before every node is executed, call reopen_grounded_execution_graph_discovery with the exact graphId/version; that legacy path abandons only the fully unexecuted graph and preserves its version for CAS. After any PRE authorization, execution, failure, or verified publication, graph history is immutable. The server may then expose sealed DATA_GAP, TABLE_DELAY, or EXECUTION_ERROR evidence. Use reopen_grounded_execution_graph_discovery only to open trigger-bound read-only discovery, then call revise_grounded_execution_graph with the exact base graph identity and evidence identity. A revision may carry immutable nodes, replace only explicitly declared unexecuted downstream nodes, or append recovery nodes for the bound failed node. Never mutate a published node or its incoming lineage, never reuse one trigger, never retry an old population receipt, and stop when the server revision budget is exhausted.
Analysis Skill headers are not disclosed by an individual query. The Core cannot read SKILL.md and must never use a Skill procedure or header to choose metrics, dimensions, tables, or Contract shape. After every datum required by the original question is in the verified evidence portfolio, call finalize_evidence_collection. Only that gate may disclose Skill headers and seal data collection. Then select at most one matching Skill or compose the verified answer. run_skill is a one-way isolation boundary: after it starts, do not retrieve more knowledge, propose another Contract, execute another query, or call run_skill again. It mounts the selected full Skill for an independent subagent, workspace and checkpoint, streams progress, and publishes a structured result artifact. Do not use task for Skill execution.
For a declared ANALYSIS Goal, delegate_grounded_exploration may be called only from verified query artifacts before evidence collection is sealed. This is not a Skill: it launches a zero-tool, zero-Skill, zero-backend advisory SubAgent. The worker may return falsifiable hypotheses, competing explanations, a stopping assessment and abstract evidence-capability requests. It cannot select a table, field, formula or SQL; cannot run a query; cannot widen population/time scope; and cannot publish an answer. Every returned request remains PENDING_ROOT_APPROVAL. Review it against the frozen graph and formal assets, then use only the normal Root discovery/Contract path for any evidence you decide is required.
When finalize_evidence_collection returns availableAnalysisGoalIds, run one matching analysis Skill. The isolated Skill returns narrow analysisPublicationRequests containing only verified artifact/column mappings and an allowed deterministic method. It cannot submit analysisType, rows, computed results, conclusions, answer prose, or causal claims. The Kernel recomputes and publishes DerivedAnalysisArtifact, then deterministically renders the final analysis span. Correlation never proves causation. Missing baseline, normalization, comparable grain, or sample size must become typed INSUFFICIENT_EVIDENCE.
compose_verified_answer generates final goal bindings internally from immutable verified artifacts and the actual rendered rows. Never invent or submit a renderer name or answer span. DETAIL, RANKING, ranked COMPARISON and DEPENDENCY sections are mechanically attested or deterministically rendered from their query artifacts; ANALYSIS and analysis-derived COMPARISON conclusions are accepted only from the dedicated verified derived-analysis artifact renderer. Primitive METRIC/DIMENSION/TIME_WINDOW/ENTITY bindings are generated mechanically. A pure RULE question uses compose_verified_rule_answer. A mixed RULE-and-data question continues data collection after publishing rule evidence and finishes once through compose_verified_answer, which binds both ledgers. A normal assistant message or unbound prose can never become the final answer.
Use retrieve_knowledge only for a targeted supplemental query; it remains inside the active Topic workspace. Governed-rule recall items marked INLINE_ONLY are usable snippets, not filesystem refs and not binding evidence. Do not read /knowledge/topics/index.json or open another Topic merely to compare alternatives. Topic expansion is allowed only after a submitted Contract returns REVISE_BINDINGS with a structured requiredCapability/searchScope gap based on evidence already read in the active Topic. A read relationship may establish that a required endpoint table is outside the current workspace; submit that relationship in the candidate Contract, then follow the returned gap to read the Topic index and exactly one relevant Topic manifest. Never expand from a normal pending request or a failed filename guess.
Do not call task in this runtime. Generic SubAgent dispatch remains disabled; only delegate_grounded_exploration and the post-query run_skill isolation boundaries are auditable.
Never invent a formula, binding, SQL result, rule, evidence status or answer. Finish only after compose_verified_rule_answer, compose_verified_answer, a verified run_skill result, or ask_human succeeds.
"""

    def __init__(
        self,
        kernel: GroundedRuntimeKernel,
        lead_model: Any,
        semantic_catalog: Any,
        *,
        checkpointer: Any = None,
        checkpoint_config_factory: Optional[Callable[[str, str], dict[str, Any]]] = None,
        skill_root: Optional[str] = None,
        skill_run_root: Optional[str] = None,
        isolated_subagent_model: Any = None,
        parallel_max_workers: int = 4,
        settings: Any = None,
        agent_factory: Any = None,
        backend: Any = None,
        conversation_state_store: Any = None,
        conversation_online_authority: Optional[GroundedConversationOnlineAuthorityFacade] = None,
        population_execution_gate: GroundedPopulationExecutionGate | None = None,
        population_gate_enforced: bool = False,
        graph_revision_fault_injector: Optional[
            Callable[[str, str], None]
        ] = None,
    ):
        self.kernel = kernel
        self.semantic_catalog = semantic_catalog
        self.settings = settings
        self.conversation_state_store = conversation_state_store
        self.conversation_online_authority = conversation_online_authority
        self.population_execution_gate = population_execution_gate
        self.population_gate_enforced = bool(population_gate_enforced)
        self.graph_revision_fault_injector = (
            graph_revision_fault_injector
        )
        self.checkpointer = checkpointer
        self.checkpoint_config_factory = checkpoint_config_factory
        self.parallel_max_workers = max(1, min(int(parallel_max_workers or 1), 8))
        self.skill_root = Path(skill_root).resolve() if skill_root else None
        self.skill_run_root = Path(skill_run_root or ".merchant-ai/skill-runs").resolve()
        self.skill_run_root.mkdir(parents=True, exist_ok=True)
        self.analysis_sandbox = MerchantAnalysisSandbox(settings) if settings is not None else None
        if self.analysis_sandbox is not None and self.skill_root is not None:
            self.analysis_sandbox.skill_root = self.skill_root
        self.skill_headers = self._load_skill_headers()
        # Native backend reads provide content only. Root-Core evidence authority
        # is recorded by tool middleware, never by ambient thread-local config.
        self.knowledge_backend = GroundedSemanticBackend(
            semantic_catalog,
            reader_is_core=lambda: False,
            semantic_activation_refresher=(
                lambda session: self.kernel.seal_semantic_activation(
                    session.runtime,
                    session.effective_topics(),
                    allow_topic_expansion=True,
                )
            ),
        )
        self.artifact_backend = GroundedRunFilesystemBackend(
            root_kind="artifacts",
            read_only=True,
            settings=settings,
            allowed_artifact_digest_provider=(_published_query_artifact_digests),
        )
        self.scratch_backend = GroundedRunFilesystemBackend(
            root_kind="scratch",
            read_only=False,
            settings=settings,
        )
        self.core_tool_boundary = GroundedCoreToolBoundaryMiddleware(self.knowledge_backend)
        self.context_middleware = GroundedContextManagementMiddleware(
            settings=settings,
        )
        self.budget_middleware = GroundedRuntimeBudgetMiddleware(settings)
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
                    "Grounded DeepAgent initialization failed: deepagents unavailable: %s" % _DEEPAGENTS_IMPORT_ERROR
                )
            agent_factory = create_deep_agent
        self._model = model
        self.context_middleware.bind_model(model)
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
                middleware=[
                    self.context_middleware,
                    self.budget_middleware,
                    self.core_tool_boundary,
                ],
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
                    ),
                    FilesystemPermission(
                        operations=["write"],
                        paths=["/artifacts", "/artifacts/**"],
                        mode="deny",
                    ),
                ],
                backend=self.backend,
                context_schema=GroundedDeepAgentRunContext,
                checkpointer=checkpointer,
                name="grounded_merchant_core",
            )
        except Exception as exc:
            self.initialization_error = "%s:%s" % (type(exc).__name__, str(exc)[:500])
            raise RuntimeError("Grounded DeepAgent initialization failed: %s" % self.initialization_error) from exc
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

    def _verified_exploration_source_views(
        self,
        session: GroundedDeepAgentSession,
        artifact_ids: list[str],
    ) -> tuple[VerifiedExplorationSourceView, ...]:
        requested = list(
            dict.fromkeys(
                str(artifact_id or "").strip() for artifact_id in artifact_ids if str(artifact_id or "").strip()
            )
        )
        artifacts = {
            artifact.artifact_id: artifact
            for artifact in _authorized_verified_query_artifacts(session)
            if artifact.verified_evidence.passed
        }
        if not requested or any(artifact_id not in artifacts for artifact_id in requested):
            raise GroundedExplorationCoordinatorError(
                "VERIFIED_EXPLORATION_SOURCE_INCOMPLETE",
                "Every exploration source must be a verified query artifact in this run.",
            )
        row_limit = max(
            1,
            min(
                int(
                    getattr(
                        self.settings,
                        "grounded_exploration_max_observation_rows",
                        64,
                    )
                    or 64
                ),
                1000,
            ),
        )
        character_limit = max(
            1000,
            min(
                int(
                    getattr(
                        self.settings,
                        "grounded_exploration_max_observation_chars",
                        12000,
                    )
                    or 12000
                ),
                200000,
            ),
        )
        views: list[VerifiedExplorationSourceView] = []
        for artifact_id in requested:
            artifact = artifacts[artifact_id]
            goal_ids = tuple(dict.fromkeys(session.artifact_goal_ids.get(artifact_id) or []))
            if not goal_ids:
                raise GroundedExplorationCoordinatorError(
                    "VERIFIED_EXPLORATION_SOURCE_GOAL_MISMATCH",
                    "A verified exploration source must retain its Goal bindings.",
                )
            bundle = artifact.run_result.merged_query_bundle
            evidence_refs = tuple(
                dict.fromkeys(
                    str(item or "").strip()
                    for item in artifact.verified_evidence.covered_evidence
                    if str(item or "").strip()
                )
            )
            observations = [
                VerifiedExplorationObservation(
                    observation_id="%s.metadata" % artifact_id,
                    statement=json.dumps(
                        {
                            "observationType": "VERIFIED_RESULT_METADATA",
                            "rowCount": bundle.effective_row_count(),
                            "visibleRowCount": len(bundle.rows),
                            "resultCoverage": str(bundle.result_coverage),
                            "isTruncated": bool(bundle.is_truncated),
                            "outputLabels": list(artifact.output_columns),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    evidence_refs=evidence_refs,
                )
            ]
            remaining_characters = character_limit - len(observations[0].statement)
            for row_index, row in enumerate(bundle.rows[:row_limit]):
                visible_row = {str(key): value for key, value in row.items() if not str(key).startswith("__")}
                statement = json.dumps(
                    {
                        "observationType": "VERIFIED_RESULT_ROW",
                        "rowIndex": row_index,
                        "values": visible_row,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if len(statement) > remaining_characters:
                    break
                observations.append(
                    VerifiedExplorationObservation(
                        observation_id="%s.row.%d"
                        % (
                            artifact_id,
                            row_index,
                        ),
                        statement=statement,
                        evidence_refs=evidence_refs,
                    )
                )
                remaining_characters -= len(statement)
            artifact_fingerprint = _stable_json_fingerprint(
                {
                    "artifactId": artifact.artifact_id,
                    "contractFingerprint": artifact.contract_fingerprint,
                    "sqlFingerprint": artifact.sql_fingerprint,
                    "goalIds": goal_ids,
                    "verifiedEvidence": artifact.verified_evidence.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "observations": [item.model_dump(by_alias=True, mode="json") for item in observations],
                }
            )
            views.append(
                VerifiedExplorationSourceView(
                    artifact_id=artifact_id,
                    artifact_fingerprint=artifact_fingerprint,
                    goal_ids=goal_ids,
                    observations=tuple(observations),
                )
            )
        return tuple(views)

    def _build_backend(self) -> CompositeBackend:
        # Full Skill bodies are intentionally absent from the parent Core.
        # Only run_skill mounts one selected Skill into an isolated backend.
        routes: dict[str, Any] = {
            "/knowledge/": self.knowledge_backend,
            "/artifacts/": self.artifact_backend,
            "/workspace/": self.scratch_backend,
        }
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
                        metadata.get("lifecyclePhase") or metadata.get("lifecycle_phase") or "post_query_analysis"
                    ).strip(),
                    "requiresVerifiedEvidence": str(
                        metadata.get("requiresVerifiedEvidence") or metadata.get("requires_verified_evidence") or "true"
                    )
                    .strip()
                    .lower(),
                    "outputContract": str(
                        metadata.get("outputContract") or metadata.get("output_contract") or ""
                    ).strip(),
                }
            )
        return headers

    def _build_tools(self) -> list[Any]:
        runtime_owner = self

        def reconcile_graph_revision_before_mutation(
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str | None:
            try:
                recovered = (
                    runtime_owner._recover_pending_graph_revisions(
                        runtime.context.session,
                        runtime_budget=runtime.context.budget,
                    )
                )
            except (
                GroundedGraphRevisionJournalError,
                RuntimeError,
                ValueError,
            ) as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": (
                            "GRAPH_REVISION_RECOVERY_FAILED"
                        ),
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            if not recovered:
                return None
            receipt = (
                runtime.context.session.execution_graph_receipt
            )
            return json.dumps(
                {
                    "status": "GRAPH_REVISION_RECOVERED",
                    "journalTransactions": recovered,
                    "receipt": (
                        receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        if receipt is not None
                        else {}
                    ),
                    "nextAction": "REPREPARE_ACTIVE_GRAPH_FRONTIER",
                },
                ensure_ascii=False,
                default=str,
            )

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
                issues = [item.model_dump(by_alias=True) for item in getattr(exc, "issues", ())]
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
            population_goal_result = None
            population_gate_id = ""
            population_goal_attestation = None
            with deep_session.lock:
                existing = deep_session.question_goal_contract
                existing_fingerprint = ""
                if existing is not None:
                    existing_fingerprint = original_question_goal_contract_fingerprint(existing)
                    if existing_fingerprint != fingerprint and (
                        deep_session.runtime.attempts
                        or _authorized_verified_query_artifacts(
                            deep_session
                        )
                        or deep_session.parallel_branches
                        or deep_session.query_branch_contexts
                        or deep_session.execution_graph_generation > 0
                    ):
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": "GOAL_CONTRACT_IMMUTABLE_AFTER_QUERY_START",
                                "contractFingerprint": existing_fingerprint,
                            },
                            ensure_ascii=False,
                        )
                goal_gate_already_committed = bool(
                    existing_fingerprint == fingerprint and deep_session.population_goal_gate_result.get("accepted")
                )
                if runtime_owner.population_gate_enforced and not goal_gate_already_committed:
                    gate = runtime_owner.population_execution_gate
                    workspace = deep_session.context_workspace
                    if gate is None or workspace is None:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": "POPULATION_GOAL_GATE_UNAVAILABLE",
                                "nextAction": "STOP_INTERNAL",
                            },
                            ensure_ascii=False,
                        )
                    population_goal_result = gate.commit_goal(
                        context_owner_fingerprint=(workspace.owner_fingerprint),
                        run_authority_fingerprint=(workspace.request_fingerprint),
                        exact_question=parsed.question,
                        goal_contract=parsed,
                    )
                    if not population_goal_result.accepted:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": "POPULATION_GOAL_REJECTED",
                                "populationGateCode": (population_goal_result.code),
                                "message": population_goal_result.message,
                                "nextAction": "REVISE_GOAL_CONTRACT",
                            },
                            ensure_ascii=False,
                        )
                    population_goal_attestation = runtime_owner._validated_population_goal_attestation(
                        population_goal_result,
                        goal_contract_fingerprint=fingerprint,
                    )
                    if population_goal_attestation is None:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("POPULATION_GOAL_ATTESTATION_INVALID"),
                                "nextAction": "STOP_INTERNAL",
                            },
                            ensure_ascii=False,
                        )
                    population_gate_id = gate.gate_id(
                        context_owner_fingerprint=(workspace.owner_fingerprint),
                        run_authority_fingerprint=(workspace.request_fingerprint),
                        goal_contract_fingerprint=fingerprint,
                    )
                deep_session.question_goal_contract = parsed.model_copy(deep=True)
                if population_goal_result is not None:
                    deep_session.population_goal_gate_id = population_gate_id
                    deep_session.population_goal_attestation = (
                        population_goal_attestation.model_copy(deep=True)
                        if population_goal_attestation is not None
                        else None
                    )
                    deep_session.population_goal_gate_result = {
                        "accepted": population_goal_result.accepted,
                        "code": population_goal_result.code,
                        "stage": str(
                            getattr(
                                population_goal_result.stage,
                                "value",
                                population_goal_result.stage,
                            )
                        ),
                    }
            rule_goal_ids = [goal.goal_id for goal in parsed.goals if str(goal.kind or "").upper() == "RULE"]
            rule_candidates = verified_rule_candidate_refs(
                core_semantic_evidence=deep_session.core_semantic_evidence,
                recall_items=deep_session.runtime.recall.items,
            )
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
                    "availableRuleEvidenceRefs": (rule_candidates if rule_goal_ids else []),
                    "nextAction": (
                        "PUBLISH_VERIFIED_RULE_EVIDENCE"
                        if rule_goal_ids and rule_candidates
                        else "READ_EXACT_RULE_EVIDENCE"
                        if rule_goal_ids
                        else "DISCOVER_SEMANTIC_EVIDENCE"
                    ),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("declare_grounded_query_branches")
        def declare_grounded_query_branches(
            branches: list[GroundedQueryBranchSpec],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Freeze a validated execution graph after semantic discovery."""

            deep_session = runtime.context.session
            if deep_session.question_goal_contract is None:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED",
                        "nextAction": "DECLARE_ORIGINAL_QUESTION_GOALS",
                    },
                    ensure_ascii=False,
                )
            normalized = [
                item if isinstance(item, GroundedQueryBranchSpec) else GroundedQueryBranchSpec.model_validate(item)
                for item in branches
            ]
            if not normalized:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "QUERY_BRANCH_DECLARATION_EMPTY",
                    },
                    ensure_ascii=False,
                )
            if len(normalized) > runtime_owner.parallel_max_workers:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "QUERY_BRANCH_DECLARATION_TOO_LARGE",
                        "maxBranches": runtime_owner.parallel_max_workers,
                    },
                    ensure_ascii=False,
                )
            with deep_session.lock:
                existing_contexts = dict(deep_session.query_branch_contexts)
                query_started = bool(
                    deep_session.runtime.attempts
                    or _authorized_verified_query_artifacts(deep_session)
                    or deep_session.parallel_branches
                )
            if existing_contexts:
                existing_specs = {
                    query_id: context.spec.model_dump(by_alias=False) for query_id, context in existing_contexts.items()
                }
                requested_specs = {
                    str(item.query_id or "").strip(): item.model_dump(by_alias=False) for item in normalized
                }
                if existing_specs == requested_specs:
                    return json.dumps(
                        {
                            "status": "ALREADY_DECLARED",
                            "branches": [item.report() for item in existing_contexts.values()],
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "QUERY_BRANCH_PLAN_IMMUTABLE",
                    },
                    ensure_ascii=False,
                )
            if query_started:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "QUERY_BRANCH_DECLARATION_TOO_LATE",
                        "instruction": (
                            "Freeze the execution graph after discovery but before any Contract attempt or query execution."
                        ),
                    },
                    ensure_ascii=False,
                )
            assignments, dependency_issues, issues = _branch_plan_validation_issues(
                deep_session,
                normalized,
            )
            if issues:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "QUERY_BRANCH_DECLARATION_INVALID",
                        "issues": issues,
                    },
                    ensure_ascii=False,
                    default=str,
                )

            seal_activation = getattr(
                runtime_owner.kernel,
                "seal_semantic_activation",
                None,
            )
            semantic_activation_seal = None
            if callable(seal_activation):
                try:
                    semantic_activation_seal = seal_activation(
                        deep_session.runtime,
                        deep_session.effective_topics(),
                    )
                except Exception as exc:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": (str(exc).partition(":")[0] or "SEMANTIC_ACTIVATION_SEAL_FAILED"),
                            "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                            "nextAction": "REOPEN_SEMANTIC_DISCOVERY",
                        },
                        ensure_ascii=False,
                    )

            dependencies_by_query_id: dict[str, set[str]] = {}
            dependency_goals_by_query_id: dict[str, set[str]] = {}
            for issue in dependency_issues:
                upstream_query_ids = {str(item) for item in issue.get("upstreamQueryIds") or [] if str(item)}
                dependency_goal_ids = {str(item) for item in issue.get("dependencyGoalIds") or [] if str(item)}
                for downstream_query_id in issue.get("downstreamQueryIds") or []:
                    normalized_downstream = str(downstream_query_id)
                    if normalized_downstream in upstream_query_ids:
                        continue
                    dependencies_by_query_id.setdefault(
                        normalized_downstream,
                        set(),
                    ).update(upstream_query_ids)
                    dependency_goals_by_query_id.setdefault(
                        normalized_downstream,
                        set(),
                    ).update(dependency_goal_ids)

            limits = GroundedBranchBudgetLimits.from_settings(runtime_owner.settings or object())
            created: dict[str, GroundedQueryBranchContext] = {}
            for raw_spec in normalized:
                query_id = str(raw_spec.query_id or "").strip()
                objective = str(raw_spec.objective or "").strip()
                if not objective:
                    goal_map = deep_session.question_goal_contract.goal_map()
                    objective = "；".join(
                        str(goal_map[goal_id].label or goal_id)
                        for goal_id in assignments[query_id]
                        if goal_id in goal_map
                    )
                spec = raw_spec.model_copy(
                    update={
                        "query_id": query_id,
                        "objective": objective,
                        "goal_ids": assignments[query_id],
                        "topic_scope": list(
                            dict.fromkeys(
                                str(item or "").strip() for item in raw_spec.topic_scope if str(item or "").strip()
                            )
                        ),
                        "evidence_ref_ids": list(
                            dict.fromkeys(
                                str(item or "").strip() for item in raw_spec.evidence_ref_ids if str(item or "").strip()
                            )
                        ),
                    },
                    deep=True,
                )
                dependency_query_ids = sorted(dependencies_by_query_id.get(query_id, set()))
                branch_runtime: Optional[GroundedRuntimeSession] = None
                if not dependency_query_ids:
                    try:
                        branch_runtime = runtime_owner.kernel.fork_query_branch(
                            deep_session.runtime,
                            query_id,
                            workspace_topics=spec.topic_scope,
                            objective=spec.objective,
                        )
                    except TypeError:
                        # Compatibility for injected test/legacy kernels that
                        # still expose the original two-argument method.
                        branch_runtime = runtime_owner.kernel.fork_query_branch(
                            deep_session.runtime,
                            query_id,
                        )
                        branch_runtime.workspace_topics = list(spec.topic_scope)
                        branch_runtime.question = spec.objective
                context = GroundedQueryBranchContext(
                    spec=spec,
                    runtime=branch_runtime,
                    budget=GroundedBranchBudget(
                        query_id,
                        limits,
                        parent=runtime.context.budget,
                    ),
                    dependency_query_ids=dependency_query_ids,
                    dependency_goal_ids=sorted(dependency_goals_by_query_id.get(query_id, set())),
                    status=("WAITING_VERIFIED_ENTITY_SET" if dependency_query_ids else "DECLARED"),
                )
                selected_evidence_refs = set(spec.evidence_ref_ids)
                for evidence in deep_session.core_semantic_evidence:
                    if selected_evidence_refs and str(evidence.get("refId") or "") not in selected_evidence_refs:
                        continue
                    evidence_topic = str(evidence.get("topic") or "").strip()
                    if evidence_topic and evidence_topic not in set(spec.topic_scope):
                        continue
                    context.semantic_ledger.retain(evidence)
                created[query_id] = context
            graph_payload = {
                "goalContractFingerprint": original_question_goal_contract_fingerprint(
                    deep_session.question_goal_contract
                ),
                "nodes": [
                    created[query_id].spec.model_dump(by_alias=True, mode="json") for query_id in sorted(created)
                ],
                "discoveryEvidence": sorted(
                    {
                        str(item.get("contentHash") or item.get("refId") or "")
                        for item in deep_session.core_semantic_evidence
                        if str(item.get("contentHash") or item.get("refId") or "")
                    }
                ),
            }
            graph_fingerprint = hashlib.sha256(
                json.dumps(
                    graph_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            with deep_session.lock:
                deep_session.query_branch_contexts = created
                deep_session.execution_graph_generation += 1
                deep_session.execution_graph_fingerprint = graph_fingerprint
            ready_query_ids = [query_id for query_id, context in created.items() if context.status == "DECLARED"]
            waiting_query_ids = [
                query_id for query_id, context in created.items() if context.status == "WAITING_VERIFIED_ENTITY_SET"
            ]
            return json.dumps(
                {
                    "status": "FROZEN",
                    "contractType": "GROUNDED_EXECUTION_GRAPH",
                    "graphGeneration": deep_session.execution_graph_generation,
                    "graphFingerprint": graph_fingerprint,
                    "semanticActivation": (
                        semantic_activation_seal.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        if semantic_activation_seal is not None
                        else {}
                    ),
                    "branchCount": len(created),
                    "readyQueryIds": ready_query_ids,
                    "waitingForVerifiedEntitySetQueryIds": waiting_query_ids,
                    "dependencyIssues": dependency_issues,
                    "branches": [context.report() for context in created.values()],
                    "nextAction": ("PREPARE_READY_GRAPH_NODES"),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("propose_grounded_execution_graph")
        def propose_grounded_execution_graph(
            proposal: GroundedExecutionGraphProposal,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Validate and freeze a versioned graph after evidence discovery."""

            deep_session = runtime.context.session
            if deep_session.question_goal_contract is None:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED",
                        "nextAction": "DECLARE_ORIGINAL_QUESTION_GOALS",
                    },
                    ensure_ascii=False,
                )
            try:
                parsed = (
                    proposal
                    if isinstance(
                        proposal,
                        GroundedExecutionGraphProposal,
                    )
                    else GroundedExecutionGraphProposal.model_validate(proposal)
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXECUTION_GRAPH_SCHEMA_INVALID",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                    },
                    ensure_ascii=False,
                )

            candidate_fingerprint = grounded_execution_graph_fingerprint(parsed)
            with deep_session.lock:
                existing_receipt = deep_session.execution_graph_receipt
                existing_contexts = bool(deep_session.query_branch_contexts)
                current_version = deep_session.execution_graph_generation
            if existing_receipt is not None and existing_receipt.fingerprint == candidate_fingerprint:
                return json.dumps(
                    {
                        "status": "ALREADY_FROZEN",
                        "contractType": "GROUNDED_EXECUTION_GRAPH",
                        "receipt": existing_receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                    },
                    ensure_ascii=False,
                )
            if existing_contexts:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": ("EXECUTION_GRAPH_IMMUTABLE_AFTER_FREEZE"),
                        "activeVersion": current_version,
                        "activeFingerprint": (deep_session.execution_graph_fingerprint),
                    },
                    ensure_ascii=False,
                )

            validation = validate_grounded_execution_graph(
                parsed,
                goal_contract=deep_session.question_goal_contract,
                discovery_evidence=(deep_session.core_semantic_evidence),
                routed_topics=deep_session.effective_topics(),
                current_version=current_version,
            )
            if not validation.valid:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXECUTION_GRAPH_INVALID",
                        "issues": [issue.model_dump(by_alias=True, mode="json") for issue in validation.issues],
                        "nextAction": ("REVISE_EXECUTION_GRAPH_FROM_CURRENT_DISCOVERY"),
                    },
                    ensure_ascii=False,
                )

            unsupported_artifact_edges = [
                {
                    "edgeIndex": index,
                    "artifactKind": edge.artifact_kind,
                }
                for index, edge in enumerate(parsed.edges)
                if edge.dependency_mode == "VERIFIED_ARTIFACT" and edge.artifact_kind != "VERIFIED_ENTITY_SET"
            ]
            if unsupported_artifact_edges:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": ("EXECUTION_GRAPH_ARTIFACT_CAPABILITY_UNAVAILABLE"),
                        "issues": unsupported_artifact_edges,
                        "supportedArtifactKinds": ["VERIFIED_ENTITY_SET"],
                    },
                    ensure_ascii=False,
                )

            preflight_specs = [
                GroundedQueryBranchSpec(
                    query_id=node.client_key,
                    objective=node.objective,
                    goal_ids=list(node.goal_ids),
                    topic_scope=list(node.topic_scope),
                    evidence_ref_ids=list(node.evidence_ref_ids),
                )
                for node in parsed.nodes
            ]
            _, dependency_issues, topology_issues = _branch_plan_validation_issues(
                deep_session,
                preflight_specs,
            )
            if topology_issues:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": ("EXECUTION_GRAPH_GOAL_TOPOLOGY_INVALID"),
                        "issues": topology_issues,
                    },
                    ensure_ascii=False,
                    default=str,
                )

            artifact_adjacency: dict[str, set[str]] = {node.client_key: set() for node in parsed.nodes}
            for edge in parsed.edges:
                if edge.dependency_mode == "VERIFIED_ARTIFACT":
                    artifact_adjacency.setdefault(
                        edge.source_client_key,
                        set(),
                    ).add(edge.target_client_key)

            def has_artifact_path(
                source_key: str,
                target_key: str,
            ) -> bool:
                pending = [source_key]
                visited = {source_key}
                cursor = 0
                while cursor < len(pending):
                    current = pending[cursor]
                    cursor += 1
                    for candidate in artifact_adjacency.get(
                        current,
                        set(),
                    ):
                        if candidate == target_key:
                            return True
                        if candidate in visited:
                            continue
                        visited.add(candidate)
                        pending.append(candidate)
                return False

            missing_artifact_paths: list[dict[str, Any]] = []
            for issue in dependency_issues:
                for source_key in issue.get("upstreamQueryIds") or []:
                    for target_key in issue.get("downstreamQueryIds") or []:
                        if source_key == target_key or has_artifact_path(
                            str(source_key),
                            str(target_key),
                        ):
                            continue
                        missing_artifact_paths.append(
                            {
                                "sourceClientKey": source_key,
                                "targetClientKey": target_key,
                                "dependency": issue,
                            }
                        )
            if missing_artifact_paths:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": ("EXECUTION_GRAPH_VERIFIED_ARTIFACT_EDGE_REQUIRED"),
                        "issues": missing_artifact_paths,
                    },
                    ensure_ascii=False,
                    default=str,
                )

            seal_activation = getattr(
                runtime_owner.kernel,
                "seal_semantic_activation",
                None,
            )
            semantic_activation_seal = None
            if callable(seal_activation):
                try:
                    semantic_activation_seal = seal_activation(
                        deep_session.runtime,
                        deep_session.effective_topics(),
                    )
                except Exception as exc:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": (str(exc).partition(":")[0] or "SEMANTIC_ACTIVATION_SEAL_FAILED"),
                            "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        },
                        ensure_ascii=False,
                    )
            receipt = build_grounded_execution_graph_receipt(
                parsed,
                version=current_version + 1,
                semantic_activation_fingerprint=str(
                    getattr(
                        semantic_activation_seal,
                        "semantic_activation_fingerprint",
                        "",
                    )
                    or ""
                ),
                semantic_activation_seal_fingerprint=str(
                    getattr(
                        semantic_activation_seal,
                        "seal_fingerprint",
                        "",
                    )
                    or ""
                ),
                semantic_activation_topics=list(
                    getattr(
                        semantic_activation_seal,
                        "exact_topics",
                        [],
                    )
                    or []
                ),
            )
            specs = [
                GroundedQueryBranchSpec(
                    query_id=receipt.node_ids[node.client_key],
                    objective=node.objective,
                    goal_ids=list(node.goal_ids),
                    topic_scope=list(node.topic_scope),
                    evidence_ref_ids=list(node.evidence_ref_ids),
                )
                for node in parsed.nodes
            ]
            legacy_result = json.loads(
                declare_grounded_query_branches.func(
                    branches=specs,
                    runtime=runtime,
                )
            )
            if legacy_result.get("status") != "FROZEN":
                return json.dumps(
                    legacy_result,
                    ensure_ascii=False,
                    default=str,
                )

            evidence_by_ref = {
                str(item.get("refId") or ""): item
                for item in deep_session.core_semantic_evidence
                if str(item.get("refId") or "")
            }
            node_by_client_key: dict[
                str,
                GroundedExecutionNodeSpec,
            ] = {node.client_key: node for node in parsed.nodes}
            with deep_session.lock:
                for client_key, query_id in receipt.node_ids.items():
                    context = deep_session.query_branch_contexts[query_id]
                    node = node_by_client_key[client_key]
                    ledger = GroundedSemanticReadLedger()
                    for ref_id in node.evidence_ref_ids:
                        evidence = evidence_by_ref.get(ref_id)
                        if evidence is not None:
                            ledger.retain(evidence)
                    context.semantic_ledger = ledger
                    context.contract_scope_query_ids = []
                    context.dependency_query_ids = []
                    context.dependency_goal_ids = []
                    context.status = "DECLARED"

                for edge in parsed.edges:
                    source_query_id = receipt.node_ids[edge.source_client_key]
                    target_query_id = receipt.node_ids[edge.target_client_key]
                    target_context = deep_session.query_branch_contexts[target_query_id]
                    if edge.dependency_mode == "CONTRACT_SCOPE":
                        if source_query_id not in (target_context.contract_scope_query_ids):
                            target_context.contract_scope_query_ids.append(source_query_id)
                        continue
                    if source_query_id not in (target_context.dependency_query_ids):
                        target_context.dependency_query_ids.append(source_query_id)
                    target_context.runtime = None
                    target_context.status = "WAITING_VERIFIED_ENTITY_SET"

                deep_session.execution_graph_generation = receipt.version
                deep_session.execution_graph_fingerprint = receipt.fingerprint
                deep_session.execution_graph_proposal = parsed.model_copy(deep=True)
                deep_session.execution_graph_receipt = receipt
                deep_session.execution_graph_edges = [edge.model_copy(deep=True) for edge in parsed.edges]

            ready_query_ids = [
                query_id
                for query_id, context in (deep_session.query_branch_contexts.items())
                if context.status == "DECLARED"
            ]
            waiting_query_ids = [
                query_id
                for query_id, context in (deep_session.query_branch_contexts.items())
                if context.status == "WAITING_VERIFIED_ENTITY_SET"
            ]
            return json.dumps(
                {
                    "status": "FROZEN",
                    "contractType": "GROUNDED_EXECUTION_GRAPH",
                    "receipt": receipt.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "clientNodeIds": dict(receipt.node_ids),
                    "readyQueryIds": ready_query_ids,
                    "waitingForVerifiedArtifactQueryIds": (waiting_query_ids),
                    "branches": [context.report() for context in (deep_session.query_branch_contexts.values())],
                    "nextAction": "PREPARE_READY_GRAPH_NODES",
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("reopen_grounded_execution_graph_discovery")
        def reopen_grounded_execution_graph_discovery(
            graph_id: str,
            version: int,
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
            trigger_evidence_ids: Optional[list[str]] = None,
        ) -> str:
            """Open typed discovery without mutating executed graph history."""

            deep_session = runtime.context.session
            reconciliation = reconcile_graph_revision_before_mutation(
                runtime
            )
            if reconciliation is not None:
                return reconciliation
            if deep_session.operational_failure:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": (
                            "EXECUTION_GRAPH_REVISION_TERMINALLY_CLOSED"
                        ),
                        "nextAction": "STOP",
                    },
                    ensure_ascii=False,
                )
            normalized_reason = str(reason or "").strip()
            if not normalized_reason:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXECUTION_GRAPH_REOPEN_REASON_REQUIRED",
                    },
                    ensure_ascii=False,
                )
            with deep_session.lock:
                receipt = deep_session.execution_graph_receipt
                contexts = dict(deep_session.query_branch_contexts)
                if receipt is None:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "EXECUTION_GRAPH_NOT_ACTIVE",
                        },
                        ensure_ascii=False,
                    )
                if str(graph_id or "").strip() != receipt.graph_id or int(version) != receipt.version:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "EXECUTION_GRAPH_REOPEN_STALE",
                            "activeGraphId": receipt.graph_id,
                            "activeVersion": receipt.version,
                        },
                        ensure_ascii=False,
                    )
                executed_query_ids = [
                    query_id
                    for query_id, context in contexts.items()
                    if context.status in {"EXECUTING", "VERIFIED", "FAILED"}
                    or bool(context.verified_artifact_ids)
                    or bool(context.runtime is not None and context.runtime.verified_query_ledger)
                    or query_id in deep_session.population_pre_execution_references
                ]
                if executed_query_ids:
                    available_triggers = _current_execution_graph_replan_evidence(deep_session)
                    if not available_triggers:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("EXECUTION_GRAPH_REOPEN_AFTER_EXECUTION_FORBIDDEN"),
                                "executedQueryIds": executed_query_ids,
                                "message": (
                                    "Executed history is immutable and no current structured revision trigger exists."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    available_by_id = {
                        item.evidence_id: item
                        for item in available_triggers
                    }
                    selected_ids = list(
                        dict.fromkeys(
                            str(item or "").strip()
                            for item in (trigger_evidence_ids or [])
                            if str(item or "").strip()
                        )
                    )
                    if not selected_ids and len(available_triggers) == 1:
                        selected_ids = [available_triggers[0].evidence_id]
                    if not selected_ids:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("EXECUTION_GRAPH_REPLAN_TRIGGER_SELECTION_REQUIRED"),
                                "availableTriggers": [
                                    _execution_graph_replan_evidence_report(item) for item in available_triggers
                                ],
                            },
                            ensure_ascii=False,
                        )
                    unavailable_ids = sorted(
                        set(selected_ids) - set(available_by_id)
                    )
                    if unavailable_ids:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": (
                                    "EXECUTION_GRAPH_REPLAN_TRIGGER_SELECTION_INVALID"
                                ),
                                "unavailableEvidenceIds": unavailable_ids,
                                "availableTriggers": [
                                    _execution_graph_replan_evidence_report(
                                        item
                                    )
                                    for item in available_triggers
                                ],
                            },
                            ensure_ascii=False,
                        )
                    selected_triggers = [
                        available_by_id[evidence_id]
                        for evidence_id in selected_ids
                    ]
                    failed_query_ids = {
                        query_id
                        for query_id, context in contexts.items()
                        if context.status
                        in {
                            "FAILED",
                            "SNAPSHOT_BLOCKED",
                        }
                    }
                    selected_failed_query_ids = {
                        item.source_query_node_id
                        for item in selected_triggers
                    }.intersection(failed_query_ids)
                    if selected_failed_query_ids != failed_query_ids:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": (
                                    "EXECUTION_GRAPH_REPLAN_TRIGGER_SET_INCOMPLETE"
                                ),
                                "missingFailedQueryIds": sorted(
                                    failed_query_ids
                                    - selected_failed_query_ids
                                ),
                                "availableTriggers": [
                                    _execution_graph_replan_evidence_report(
                                        item
                                    )
                                    for item in available_triggers
                                ],
                            },
                            ensure_ascii=False,
                        )
                    if deep_session.execution_graph_revision_count >= deep_session.execution_graph_max_revision_count:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("EXECUTION_GRAPH_REPLAN_BUDGET_EXHAUSTED"),
                            },
                            ensure_ascii=False,
                        )
                    deep_session.execution_graph_revision_discovery_evidence_ids = list(
                        selected_ids
                    )
                    deep_session.execution_graph_revision_discovery_evidence_id = (
                        selected_ids[0]
                        if len(selected_ids) == 1
                        else ""
                    )
                    return json.dumps(
                        {
                            "status": "REVISION_DISCOVERY_OPENED",
                            "baseVersion": receipt.version,
                            "activeGraphId": receipt.graph_id,
                            "executedQueryIds": executed_query_ids,
                            "triggerEvidence": (
                                _execution_graph_replan_evidence_report(
                                    selected_triggers[0]
                                )
                                if len(selected_triggers) == 1
                                else {}
                            ),
                            "triggerEvidenceSet": [
                                _execution_graph_replan_evidence_report(item)
                                for item in selected_triggers
                            ],
                            "triggerEvidenceSetFingerprint": (
                                grounded_execution_graph_replan_evidence_set_fingerprint(
                                    selected_triggers
                                )
                            ),
                            "discoverySnapshotFingerprint": (
                                discovery_evidence_snapshot_fingerprint(deep_session.core_semantic_evidence)
                            ),
                            "nextAction": ("READ_ONLY_FOR_STRUCTURED_TRIGGER_THEN_REVISE_GRAPH"),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                gapped_contexts = [
                    context
                    for context in contexts.values()
                    if context.status == "CONTRACT_GAPPED" and context.last_gaps
                ]
                if not gapped_contexts:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "EXECUTION_GRAPH_TYPED_GAP_REQUIRED",
                        },
                        ensure_ascii=False,
                    )
                gap_reports = [
                    {
                        "queryId": context.spec.query_id,
                        "goalIds": list(context.spec.goal_ids),
                        "gaps": [dict(item) for item in context.last_gaps],
                    }
                    for context in gapped_contexts
                ]
                deep_session.execution_graph_history.append(
                    {
                        "status": "GAPPED_REOPENED",
                        "receipt": receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "reason": normalized_reason,
                        "gaps": gap_reports,
                        "branches": [context.report() for context in contexts.values()],
                    }
                )
                deep_session.query_branch_contexts = {}
                deep_session.parallel_branches = {}
                deep_session.parallel_branch_goal_ids = {}
                deep_session.execution_graph_receipt = None
                deep_session.execution_graph_proposal = None
                deep_session.execution_graph_edges = []
                deep_session.execution_graph_data_snapshot = None
                deep_session.execution_graph_fingerprint = ""
                deep_session.execution_graph_revision_discovery_evidence_id = ""
                deep_session.execution_graph_revision_discovery_evidence_ids = []
                deep_session.active_goal_ids = []

            return json.dumps(
                {
                    "status": "DISCOVERY_REOPENED",
                    "baseVersion": receipt.version,
                    "previousGraphId": receipt.graph_id,
                    "typedGaps": gap_reports,
                    "discoverySnapshotFingerprint": (
                        discovery_evidence_snapshot_fingerprint(deep_session.core_semantic_evidence)
                    ),
                    "nextAction": ("READ_ONLY_FOR_RETURNED_STRUCTURED_GAPS"),
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("revise_grounded_execution_graph")
        def revise_grounded_execution_graph(
            revision: GroundedExecutionGraphRevisionProposal,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """CAS-append one trigger-bound graph revision."""

            deep_session = runtime.context.session
            try:
                recovered_transactions = (
                    runtime_owner._recover_pending_graph_revisions(
                        deep_session,
                        runtime_budget=runtime.context.budget,
                    )
                )
            except (
                GroundedGraphRevisionJournalError,
                RuntimeError,
                ValueError,
            ) as exc:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": (
                            "GRAPH_REVISION_RECOVERY_FAILED"
                        ),
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            if recovered_transactions:
                latest = recovered_transactions[-1]
                receipt_payload = dict(
                    latest.get("executionReceipt") or {}
                )
                return json.dumps(
                    {
                        "status": "REVISED",
                        "recovered": True,
                        "journalTransactions": (
                            recovered_transactions
                        ),
                        "receipt": receipt_payload,
                        "clientNodeIds": dict(
                            receipt_payload.get("nodeIds") or {}
                        ),
                        "nextAction": "PREPARE_READY_GRAPH_NODES",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            try:
                parsed = (
                    revision
                    if isinstance(
                        revision,
                        GroundedExecutionGraphRevisionProposal,
                    )
                    else GroundedExecutionGraphRevisionProposal.model_validate(revision)
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXECUTION_GRAPH_REVISION_SCHEMA_INVALID",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                    },
                    ensure_ascii=False,
                )

            with deep_session.lock:
                active_receipt = deep_session.execution_graph_receipt
                active_proposal = deep_session.execution_graph_proposal
                goal_contract = deep_session.question_goal_contract
                proposed_trigger_fingerprints = sorted(
                    binding.evidence_fingerprint
                    for binding in parsed.trigger_evidence_set
                )
                if (
                    active_receipt is not None
                    and active_receipt.parent_graph_id
                    == parsed.base_graph_id
                    and active_receipt.parent_version
                    == parsed.base_version
                    and active_receipt.parent_fingerprint
                    == parsed.base_fingerprint
                    and active_receipt.replan_evidence_fingerprints
                    == proposed_trigger_fingerprints
                    and active_receipt.fingerprint
                    == grounded_execution_graph_fingerprint(
                        parsed.graph
                    )
                ):
                    return json.dumps(
                        {
                            "status": "REVISED",
                            "idempotent": True,
                            "receipt": active_receipt.model_dump(
                                by_alias=True,
                                mode="json",
                            ),
                            "clientNodeIds": dict(
                                active_receipt.node_ids
                            ),
                            "nextAction": (
                                "PREPARE_READY_GRAPH_NODES"
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                trigger_evidences = [
                    deep_session.execution_graph_replan_evidence.get(
                        binding.evidence_id
                    )
                    for binding in parsed.trigger_evidence_set
                ]
                if active_receipt is None or active_proposal is None or goal_contract is None:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "EXECUTION_GRAPH_REVISION_BASE_REQUIRED",
                        },
                        ensure_ascii=False,
                    )
                if any(item is None for item in trigger_evidences):
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": ("EXECUTION_GRAPH_REPLAN_EVIDENCE_NOT_FOUND"),
                        },
                        ensure_ascii=False,
                    )
                selected_trigger_evidences = [
                    item
                    for item in trigger_evidences
                    if item is not None
                ]
                opened_evidence_ids = set(
                    deep_session.execution_graph_revision_discovery_evidence_ids
                    or (
                        [
                            deep_session.execution_graph_revision_discovery_evidence_id
                        ]
                        if deep_session.execution_graph_revision_discovery_evidence_id
                        else []
                    )
                )
                proposed_evidence_ids = {
                    item.evidence_id
                    for item in selected_trigger_evidences
                }
                if (
                    opened_evidence_ids
                    and opened_evidence_ids != proposed_evidence_ids
                ):
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": (
                                "EXECUTION_GRAPH_REVISION_DISCOVERY_BINDING_MISMATCH"
                            ),
                        },
                        ensure_ascii=False,
                    )

                node_states = _execution_graph_node_runtime_states(
                    deep_session,
                    active_receipt,
                )
                validation = validate_grounded_execution_graph_revision(
                    parsed,
                    active_proposal=active_proposal,
                    active_receipt=active_receipt,
                    trigger_evidence=selected_trigger_evidences,
                    node_states=node_states,
                    goal_contract=goal_contract,
                    discovery_evidence=(deep_session.core_semantic_evidence),
                    routed_topics=deep_session.effective_topics(),
                    used_trigger_fingerprints=(deep_session.execution_graph_used_replan_fingerprints),
                    completed_revision_count=(deep_session.execution_graph_revision_count),
                    max_revision_count=(deep_session.execution_graph_max_revision_count),
                )
                if not validation.valid:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "EXECUTION_GRAPH_REVISION_INVALID",
                            "issues": [
                                issue.model_dump(
                                    by_alias=True,
                                    mode="json",
                                )
                                for issue in validation.issues
                            ],
                        },
                        ensure_ascii=False,
                        default=str,
                    )

                unsupported_artifact_edges = [
                    {
                        "edgeIndex": index,
                        "artifactKind": edge.artifact_kind,
                    }
                    for index, edge in enumerate(parsed.graph.edges)
                    if (edge.dependency_mode == "VERIFIED_ARTIFACT" and edge.artifact_kind != "VERIFIED_ENTITY_SET")
                ]
                if unsupported_artifact_edges:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": ("EXECUTION_GRAPH_ARTIFACT_CAPABILITY_UNAVAILABLE"),
                            "issues": unsupported_artifact_edges,
                            "supportedArtifactKinds": ["VERIFIED_ENTITY_SET"],
                        },
                        ensure_ascii=False,
                    )

                preflight_specs = [
                    GroundedQueryBranchSpec(
                        query_id=node.client_key,
                        objective=node.objective,
                        goal_ids=list(node.goal_ids),
                        topic_scope=list(node.topic_scope),
                        evidence_ref_ids=list(node.evidence_ref_ids),
                    )
                    for node in parsed.graph.nodes
                ]
                assignments, dependency_issues, topology_issues = _branch_plan_validation_issues(
                    deep_session,
                    preflight_specs,
                )
                if topology_issues:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": ("EXECUTION_GRAPH_GOAL_TOPOLOGY_INVALID"),
                            "issues": topology_issues,
                        },
                        ensure_ascii=False,
                        default=str,
                    )

                artifact_adjacency: dict[str, set[str]] = {node.client_key: set() for node in parsed.graph.nodes}
                for edge in parsed.graph.edges:
                    if edge.dependency_mode == "VERIFIED_ARTIFACT":
                        artifact_adjacency.setdefault(
                            edge.source_client_key,
                            set(),
                        ).add(edge.target_client_key)

                def has_artifact_path(
                    source_key: str,
                    target_key: str,
                ) -> bool:
                    pending = [source_key]
                    visited = {source_key}
                    cursor = 0
                    while cursor < len(pending):
                        current = pending[cursor]
                        cursor += 1
                        for candidate in artifact_adjacency.get(
                            current,
                            set(),
                        ):
                            if candidate == target_key:
                                return True
                            if candidate in visited:
                                continue
                            visited.add(candidate)
                            pending.append(candidate)
                    return False

                missing_artifact_paths: list[dict[str, Any]] = []
                for issue in dependency_issues:
                    for source_key in issue.get("upstreamQueryIds") or []:
                        for target_key in issue.get("downstreamQueryIds") or []:
                            if source_key == target_key or has_artifact_path(
                                str(source_key),
                                str(target_key),
                            ):
                                continue
                            missing_artifact_paths.append(
                                {
                                    "sourceClientKey": source_key,
                                    "targetClientKey": target_key,
                                    "dependency": issue,
                                }
                            )
                if missing_artifact_paths:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": ("EXECUTION_GRAPH_VERIFIED_ARTIFACT_EDGE_REQUIRED"),
                            "issues": missing_artifact_paths,
                        },
                        ensure_ascii=False,
                        default=str,
                    )

                semantic_activation_seal = None
                seal_activation = getattr(
                    runtime_owner.kernel,
                    "seal_semantic_activation",
                    None,
                )
                if callable(seal_activation):
                    try:
                        semantic_activation_seal = seal_activation(
                            deep_session.runtime,
                            deep_session.effective_topics(),
                        )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": (str(exc).partition(":")[0] or "SEMANTIC_ACTIVATION_SEAL_FAILED"),
                                "message": "%s:%s"
                                % (
                                    type(exc).__name__,
                                    str(exc)[:400],
                                ),
                            },
                            ensure_ascii=False,
                        )

                carried_keys = set(validation.carried_forward_client_keys)
                trigger_evidence_set_fingerprint = (
                    grounded_execution_graph_replan_evidence_set_fingerprint(
                        selected_trigger_evidences
                    )
                )
                trigger_evidence_fingerprints = sorted(
                    item.evidence_fingerprint
                    for item in selected_trigger_evidences
                )
                preserved_node_ids = {key: active_receipt.node_ids[key] for key in carried_keys}
                retired_client_keys = set(active_receipt.node_ids) - carried_keys
                retired_node_ids = [active_receipt.node_ids[key] for key in sorted(retired_client_keys)]
                revised_receipt = build_grounded_execution_graph_receipt(
                    parsed.graph,
                    version=active_receipt.version + 1,
                    semantic_activation_fingerprint=str(
                        getattr(
                            semantic_activation_seal,
                            "semantic_activation_fingerprint",
                            active_receipt.semantic_activation_fingerprint,
                        )
                        or ""
                    ),
                    semantic_activation_seal_fingerprint=str(
                        getattr(
                            semantic_activation_seal,
                            "seal_fingerprint",
                            active_receipt.semantic_activation_seal_fingerprint,
                        )
                        or ""
                    ),
                    semantic_activation_topics=list(
                        getattr(
                            semantic_activation_seal,
                            "exact_topics",
                            active_receipt.semantic_activation_topics,
                        )
                        or []
                    ),
                    parent_receipt=active_receipt,
                    replan_evidence_fingerprint=(
                        trigger_evidence_set_fingerprint
                    ),
                    replan_evidence_fingerprints=(
                        trigger_evidence_fingerprints
                    ),
                    preserved_node_ids=preserved_node_ids,
                    retired_node_ids=retired_node_ids,
                )

                old_contexts = dict(deep_session.query_branch_contexts)
                node_by_key = {node.client_key: node for node in parsed.graph.nodes}
                evidence_by_ref = {
                    str(item.get("refId") or ""): item
                    for item in deep_session.core_semantic_evidence
                    if str(item.get("refId") or "")
                }
                limits = GroundedBranchBudgetLimits.from_settings(runtime_owner.settings or object())
                candidate_contexts: dict[
                    str,
                    GroundedQueryBranchContext,
                ] = {}
                for client_key in validation.carried_forward_client_keys:
                    old_query_id = active_receipt.node_ids[client_key]
                    candidate_contexts[old_query_id] = old_contexts[old_query_id]

                incoming_edges: dict[
                    str,
                    list[GroundedExecutionEdgeSpec],
                ] = {}
                for edge in parsed.graph.edges:
                    incoming_edges.setdefault(
                        edge.target_client_key,
                        [],
                    ).append(edge)

                for client_key, node in node_by_key.items():
                    if client_key in carried_keys:
                        continue
                    query_node_id = revised_receipt.node_ids[client_key]
                    assigned_goal_ids = assignments.get(
                        client_key,
                        list(node.goal_ids),
                    )
                    objective = str(node.objective or "").strip()
                    if not objective:
                        goal_map = goal_contract.goal_map()
                        objective = "；".join(
                            str(goal_map[goal_id].label or goal_id)
                            for goal_id in assigned_goal_ids
                            if goal_id in goal_map
                        )
                    spec = GroundedQueryBranchSpec(
                        query_id=query_node_id,
                        objective=objective,
                        goal_ids=list(assigned_goal_ids),
                        topic_scope=list(node.topic_scope),
                        evidence_ref_ids=list(node.evidence_ref_ids),
                    )
                    artifact_dependencies = [
                        edge
                        for edge in incoming_edges.get(
                            client_key,
                            [],
                        )
                        if edge.dependency_mode == "VERIFIED_ARTIFACT"
                    ]
                    branch_runtime: Optional[GroundedRuntimeSession] = None
                    if not artifact_dependencies:
                        try:
                            branch_runtime = runtime_owner.kernel.fork_query_branch(
                                deep_session.runtime,
                                query_node_id,
                                workspace_topics=spec.topic_scope,
                                objective=spec.objective,
                            )
                        except TypeError:
                            branch_runtime = runtime_owner.kernel.fork_query_branch(
                                deep_session.runtime,
                                query_node_id,
                            )
                            branch_runtime.workspace_topics = list(spec.topic_scope)
                            branch_runtime.question = spec.objective
                    branch_context = GroundedQueryBranchContext(
                        spec=spec,
                        runtime=branch_runtime,
                        budget=GroundedBranchBudget(
                            query_node_id,
                            limits,
                            parent=runtime.context.budget,
                        ),
                        status=("WAITING_VERIFIED_ENTITY_SET" if artifact_dependencies else "DECLARED"),
                    )
                    for ref_id in spec.evidence_ref_ids:
                        evidence = evidence_by_ref.get(ref_id)
                        if evidence is not None:
                            branch_context.semantic_ledger.retain(evidence)
                    candidate_contexts[query_node_id] = branch_context

                for client_key, node in node_by_key.items():
                    if client_key in carried_keys:
                        continue
                    target_query_id = revised_receipt.node_ids[client_key]
                    target_context = candidate_contexts[target_query_id]
                    for edge in incoming_edges.get(client_key, []):
                        source_query_id = revised_receipt.node_ids[edge.source_client_key]
                        source_goals = node_by_key[edge.source_client_key].goal_ids
                        if edge.dependency_mode == "CONTRACT_SCOPE":
                            target_context.contract_scope_query_ids.append(source_query_id)
                        else:
                            target_context.dependency_query_ids.append(source_query_id)
                            target_context.dependency_goal_ids.extend(
                                goal_id for goal_id in source_goals if goal_id not in target_context.dependency_goal_ids
                            )

                old_population_receipt = deep_session.population_graph_receipt
                revised_population_receipt = None
                revision_journal: (
                    GroundedGraphRevisionTransactionJournal | None
                ) = None
                revision_journal_record = None
                if old_population_receipt is not None or runtime_owner.population_gate_enforced:
                    if old_population_receipt is None:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("POPULATION_GRAPH_REVISION_BASE_REQUIRED"),
                            },
                            ensure_ascii=False,
                        )
                    population_nodes = tuple(
                        PopulationDynamicGraphNode(
                            query_node_id=query_node_id,
                            consumer_goal_ids=tuple(candidate_contexts[query_node_id].spec.goal_ids),
                        )
                        for query_node_id in revised_receipt.node_ids.values()
                    )
                    population_edges = tuple(
                        PopulationDynamicGraphEdge(
                            source_query_node_id=(revised_receipt.node_ids[edge.source_client_key]),
                            target_query_node_id=(revised_receipt.node_ids[edge.target_client_key]),
                            dependency_mode=edge.dependency_mode,
                            artifact_kind=edge.artifact_kind,
                        )
                        for edge in parsed.graph.edges
                    )
                    revised_population_receipt = seal_population_dynamic_graph_receipt(
                        PopulationDynamicGraphReceipt(
                            graph_id=revised_receipt.graph_id,
                            graph_version=revised_receipt.version,
                            graph_fingerprint=(revised_receipt.fingerprint),
                            nodes=population_nodes,
                            edges=population_edges,
                            parent_receipt_fingerprint=(old_population_receipt.receipt_fingerprint),
                            revision_evidence_fingerprint=(
                                trigger_evidence_set_fingerprint
                            ),
                            carried_forward_query_node_ids=tuple(revised_receipt.carried_forward_node_ids),
                            retired_query_node_ids=tuple(revised_receipt.retired_node_ids),
                        )
                    )

                if runtime_owner.population_gate_enforced:
                    workspace = deep_session.context_workspace
                    if (
                        workspace is None
                        or old_population_receipt is None
                        or revised_population_receipt is None
                    ):
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": (
                                    "POPULATION_GRAPH_REVISION_BASE_REQUIRED"
                                ),
                            },
                            ensure_ascii=False,
                        )
                    try:
                        revision_journal = (
                            GroundedGraphRevisionTransactionJournal(
                                workspace
                            )
                        )
                        base_session_checkpoint = (
                            _build_graph_revision_base_session_checkpoint(
                                deep_session,
                                execution_proposal=active_proposal,
                                execution_receipt=active_receipt,
                                population_receipt=(
                                    old_population_receipt
                                ),
                                node_states=node_states,
                            )
                        )
                        base_session_checkpoint_reference = (
                            revision_journal.persist_base_session_checkpoint(
                                base_session_checkpoint
                            )
                        )
                        recovery_payload = (
                            build_grounded_graph_revision_recovery_payload(
                                execution_proposal=parsed.graph,
                                execution_receipt=revised_receipt,
                                population_receipt=(
                                    revised_population_receipt
                                ),
                                base_session_checkpoint=(
                                    base_session_checkpoint_reference
                                ),
                                assigned_goal_ids_by_client_key={
                                    client_key: list(
                                        candidate_contexts[
                                            revised_receipt.node_ids[
                                                client_key
                                            ]
                                        ].spec.goal_ids
                                    )
                                    for client_key in (
                                        revised_receipt.node_ids
                                    )
                                },
                            )
                        )
                        prepared_transaction = revision_journal.prepare(
                            base_execution_receipt_fingerprint=(
                                active_receipt.fingerprint
                            ),
                            new_execution_receipt_fingerprint=(
                                revised_receipt.fingerprint
                            ),
                            base_population_receipt_fingerprint=(
                                old_population_receipt.receipt_fingerprint
                            ),
                            new_population_receipt_fingerprint=(
                                revised_population_receipt.receipt_fingerprint
                            ),
                            evidence_set_fingerprint=(
                                trigger_evidence_set_fingerprint
                            ),
                            recovery_payload=recovery_payload,
                        )
                        revision_journal_record = (
                            prepared_transaction.record
                        )
                        runtime_owner._graph_revision_fault_checkpoint(
                            "AFTER_PREPARE",
                            revision_journal_record.transaction_id,
                        )
                    except GroundedGraphRevisionJournalError as exc:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": exc.code,
                                "nextAction": "STOP_INTERNAL",
                            },
                            ensure_ascii=False,
                        )

                refreshed_population_references: dict[
                    str,
                    PopulationPreExecutionReference,
                ] = {}
                if runtime_owner.population_gate_enforced:
                    gate = runtime_owner.population_execution_gate
                    workspace = deep_session.context_workspace
                    if gate is None or workspace is None:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("POPULATION_ONLINE_GATE_AUTHORITY_REQUIRED"),
                            },
                            ensure_ascii=False,
                        )
                    assert revised_population_receipt is not None
                    gate_result = gate.revise_graph(
                        context_owner_fingerprint=(workspace.owner_fingerprint),
                        run_authority_fingerprint=(workspace.request_fingerprint),
                        goal_contract_fingerprint=(original_question_goal_contract_fingerprint(goal_contract)),
                        previous_graph_receipt_fingerprint=(old_population_receipt.receipt_fingerprint),
                        revised_graph_receipt=(revised_population_receipt),
                        revision_evidence_fingerprint=(
                            trigger_evidence_set_fingerprint
                        ),
                    )
                    if not gate_result.accepted:
                        return json.dumps(
                            {
                                "status": "REJECTED",
                                "code": ("POPULATION_GRAPH_REVISION_REJECTED"),
                                "populationCode": gate_result.code,
                                "message": gate_result.message,
                            },
                            ensure_ascii=False,
                        )
                    assert revision_journal is not None
                    assert revision_journal_record is not None
                    runtime_owner._graph_revision_fault_checkpoint(
                        "AFTER_POPULATION_CAS",
                        revision_journal_record.transaction_id,
                    )
                    population_transaction = revision_journal.advance(
                        revision_journal_record.transaction_id,
                        target_status="POPULATION_COMMITTED",
                        expected_revision=(
                            revision_journal_record.revision
                        ),
                        expected_record_fingerprint=(
                            revision_journal_record.record_fingerprint
                        ),
                    )
                    revision_journal_record = (
                        population_transaction.record
                    )
                    runtime_owner._graph_revision_fault_checkpoint(
                        "AFTER_POPULATION_COMMITTED",
                        revision_journal_record.transaction_id,
                    )
                    for query_node_id, old_reference in deep_session.population_pre_execution_references.items():
                        if query_node_id not in set(revised_receipt.carried_forward_node_ids):
                            continue
                        refreshed_population_references[query_node_id] = gate.build_pre_execution_reference(
                            context_owner_fingerprint=(workspace.owner_fingerprint),
                            run_authority_fingerprint=(workspace.request_fingerprint),
                            goal_contract_fingerprint=(original_question_goal_contract_fingerprint(goal_contract)),
                            graph_receipt=(revised_population_receipt),
                            node=old_reference.node,
                        )

                has_published_node = any(state.lifecycle == "PUBLISHED" for state in node_states)
                deep_session.execution_graph_history.append(
                    {
                        "status": "REVISED",
                        "parentReceipt": active_receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "receipt": revised_receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "triggerEvidenceSet": [
                            _execution_graph_replan_evidence_report(item)
                            for item in selected_trigger_evidences
                        ],
                        "triggerEvidenceSetFingerprint": (
                            trigger_evidence_set_fingerprint
                        ),
                        "carriedForwardQueryNodeIds": list(revised_receipt.carried_forward_node_ids),
                        "retiredQueryNodeIds": list(revised_receipt.retired_node_ids),
                    }
                )
                deep_session.query_branch_contexts = candidate_contexts
                deep_session.execution_graph_proposal = parsed.graph.model_copy(deep=True)
                deep_session.execution_graph_receipt = revised_receipt.model_copy(deep=True)
                deep_session.execution_graph_edges = [edge.model_copy(deep=True) for edge in parsed.graph.edges]
                deep_session.execution_graph_generation = revised_receipt.version
                deep_session.execution_graph_fingerprint = revised_receipt.fingerprint
                if not has_published_node:
                    deep_session.execution_graph_data_snapshot = None
                deep_session.execution_graph_revision_count += 1
                deep_session.execution_graph_used_replan_fingerprints.extend(
                    fingerprint
                    for fingerprint in trigger_evidence_fingerprints
                    if fingerprint
                    not in set(
                        deep_session.execution_graph_used_replan_fingerprints
                    )
                )
                deep_session.execution_graph_revision_discovery_evidence_id = ""
                deep_session.execution_graph_revision_discovery_evidence_ids = []
                active_query_ids = set(candidate_contexts)
                deep_session.parallel_branches = {
                    query_node_id: branch
                    for query_node_id, branch in (deep_session.parallel_branches.items())
                    if query_node_id in active_query_ids
                }
                deep_session.parallel_branch_goal_ids = {
                    query_node_id: list(goal_ids)
                    for query_node_id, goal_ids in (deep_session.parallel_branch_goal_ids.items())
                    if query_node_id in active_query_ids
                }
                if revised_population_receipt is not None:
                    deep_session.population_graph_receipt = revised_population_receipt.model_copy(deep=True)
                if runtime_owner.population_gate_enforced:
                    deep_session.population_pre_execution_references = refreshed_population_references
                deep_session.population_post_gate_results = {
                    query_node_id: dict(result)
                    for query_node_id, result in (deep_session.population_post_gate_results.items())
                    if query_node_id in set(revised_receipt.carried_forward_node_ids)
                }
                deep_session.population_artifact_query_node_ids = {
                    artifact_id: query_node_id
                    for artifact_id, query_node_id in (
                        deep_session.population_artifact_query_node_ids.items()
                    )
                    if query_node_id
                    in set(revised_receipt.carried_forward_node_ids)
                }

                if revision_journal is not None:
                    assert revision_journal_record is not None
                    runtime_owner._graph_revision_fault_checkpoint(
                        "AFTER_EXECUTION_SWITCH",
                        revision_journal_record.transaction_id,
                    )
                    execution_transaction = revision_journal.advance(
                        revision_journal_record.transaction_id,
                        target_status="EXECUTION_COMMITTED",
                        expected_revision=(
                            revision_journal_record.revision
                        ),
                        expected_record_fingerprint=(
                            revision_journal_record.record_fingerprint
                        ),
                    )
                    revision_journal_record = (
                        execution_transaction.record
                    )

                ready_query_ids = [
                    query_node_id
                    for query_node_id, context in candidate_contexts.items()
                    if context.status == "DECLARED"
                ]
                waiting_query_ids = [
                    query_node_id
                    for query_node_id, context in candidate_contexts.items()
                    if context.status == "WAITING_VERIFIED_ENTITY_SET"
                ]
                return json.dumps(
                    {
                        "status": "REVISED",
                        "contractType": ("GROUNDED_EXECUTION_GRAPH_REVISION"),
                        "receipt": revised_receipt.model_dump(
                            by_alias=True,
                            mode="json",
                        ),
                        "triggerEvidence": (
                            _execution_graph_replan_evidence_report(
                                selected_trigger_evidences[0]
                            )
                            if len(selected_trigger_evidences) == 1
                            else {}
                        ),
                        "triggerEvidenceSet": [
                            _execution_graph_replan_evidence_report(item)
                            for item in selected_trigger_evidences
                        ],
                        "triggerEvidenceSetFingerprint": (
                            trigger_evidence_set_fingerprint
                        ),
                        "clientNodeIds": dict(revised_receipt.node_ids),
                        "carriedForwardQueryNodeIds": list(revised_receipt.carried_forward_node_ids),
                        "retiredQueryNodeIds": list(revised_receipt.retired_node_ids),
                        "readyQueryIds": ready_query_ids,
                        "waitingForVerifiedArtifactQueryIds": (waiting_query_ids),
                        "revisionBudget": {
                            "used": (deep_session.execution_graph_revision_count),
                            "maximum": (deep_session.execution_graph_max_revision_count),
                        },
                        "revisionTransactionId": (
                            revision_journal_record.transaction_id
                            if revision_journal_record is not None
                            else ""
                        ),
                        "nextAction": "PREPARE_READY_GRAPH_NODES",
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

            if runtime.context.session.analysis_skill_started or runtime.context.session.data_collection_sealed:
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
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
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

        @tool("publish_verified_rule_evidence")
        def publish_verified_rule_evidence(
            goal_ids: list[str],
            rule_ref_ids: list[str],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Publish current-session rule reads/recall as a verified artifact."""

            deep_session = runtime.context.session
            assigned_goal_ids, goal_error = _validated_goal_assignment(
                deep_session,
                goal_ids,
            )
            if goal_error:
                return json.dumps(goal_error, ensure_ascii=False)
            assert deep_session.question_goal_contract is not None
            goal_map = deep_session.question_goal_contract.goal_map()
            invalid_goal_ids = [
                goal_id for goal_id in assigned_goal_ids if str(goal_map[goal_id].kind or "").upper() != "RULE"
            ]
            if invalid_goal_ids:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "RULE_ARTIFACT_GOAL_KIND_REQUIRED",
                        "invalidGoalIds": invalid_goal_ids,
                        "nextAction": "REVISE_GOAL_CONTRACT",
                    },
                    ensure_ascii=False,
                )
            candidates = verified_rule_candidate_refs(
                core_semantic_evidence=deep_session.core_semantic_evidence,
                recall_items=deep_session.runtime.recall.items,
            )
            if not candidates:
                try:
                    bundle = runtime_owner.kernel.recall_navigation(
                        deep_session.runtime,
                        query=deep_session.runtime.question,
                    )
                except Exception:
                    bundle = RecallBundle()
                candidates = verified_rule_candidate_refs(
                    core_semantic_evidence=deep_session.core_semantic_evidence,
                    recall_items=bundle.items,
                )
            available_ref_ids = [str(item.get("refId") or "") for item in candidates]
            requested = (
                list(dict.fromkeys(str(item or "").strip() for item in rule_ref_ids if str(item or "").strip()))
                or available_ref_ids[:4]
            )
            try:
                artifact = build_verified_rule_artifact(
                    question=deep_session.runtime.question,
                    goal_contract_fingerprint=(
                        original_question_goal_contract_fingerprint(deep_session.question_goal_contract)
                    ),
                    goal_ids=assigned_goal_ids,
                    requested_ref_ids=requested,
                    core_semantic_evidence=deep_session.core_semantic_evidence,
                    recall_items=deep_session.runtime.recall.items,
                )
                artifact = runtime_owner.kernel.publish_rule_artifact(
                    deep_session.runtime,
                    artifact,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "RULE_EVIDENCE_INCOMPLETE",
                        "code": "VERIFIED_RULE_ARTIFACT_REQUIRED",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                        "availableRuleEvidenceRefs": candidates,
                        "nextAction": "READ_OR_RECALL_EXACT_RULE_EVIDENCE",
                    },
                    ensure_ascii=False,
                )
            coverage = runtime_owner._goal_coverage_snapshot(deep_session)
            remaining_required_goal_ids = list(coverage.missing_required_goal_ids)
            remaining_non_rule_goal_ids = [
                goal_id
                for goal_id in remaining_required_goal_ids
                if goal_id in goal_map and str(goal_map[goal_id].kind or "").strip().upper() != "RULE"
            ]
            remaining_rule_goal_ids = [
                goal_id
                for goal_id in remaining_required_goal_ids
                if goal_id in goal_map and str(goal_map[goal_id].kind or "").strip().upper() == "RULE"
            ]
            if remaining_non_rule_goal_ids:
                next_action = "CONTINUE_GROUNDED_DATA_COLLECTION"
            elif remaining_rule_goal_ids:
                next_action = "PUBLISH_ADDITIONAL_RULE_EVIDENCE"
            elif _required_goals_are_rule_only(deep_session):
                next_action = "COMPOSE_VERIFIED_RULE_ANSWER"
            else:
                next_action = "FINALIZE_EVIDENCE_COLLECTION"
            return json.dumps(
                {
                    "status": "RULE_EVIDENCE_VERIFIED",
                    "artifactId": artifact.artifact_id,
                    "goalIds": artifact.goal_ids,
                    "ruleEvidenceRefs": [item.ref_id for item in artifact.evidence_refs],
                    "remainingRequiredGoalIds": remaining_required_goal_ids,
                    "nextAction": next_action,
                },
                ensure_ascii=False,
            )

        @tool("compose_verified_rule_answer", return_direct=True)
        def compose_verified_rule_answer(
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Finish a rule-only question from verified rule artifacts."""

            deep_session = runtime.context.session
            if _required_non_rule_goal_ids(deep_session):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "MIXED_GOAL_FINALIZER_REQUIRED",
                        "remainingDataGoalIds": (_required_non_rule_goal_ids(deep_session)),
                        "nextAction": "CONTINUE_GROUNDED_DATA_COLLECTION",
                    },
                    ensure_ascii=False,
                )
            try:
                coverage = runtime_owner._require_complete_goal_coverage(deep_session)
            except GoalCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                deep_session.goal_coverage_result = result
                return json.dumps(
                    {
                        "status": "GOAL_COVERAGE_INCOMPLETE",
                        "code": "RULE_GOALS_UNPROVED",
                        "issues": result.get("issues", []),
                    },
                    ensure_ascii=False,
                )
            answer = runtime_owner.kernel.compose_rule_answer(deep_session.runtime)
            deep_session.goal_coverage_result = coverage.model_dump(by_alias=True)
            try:
                answer_coverage = AnswerCoverageVerifier().require_complete(
                    deep_session.question_goal_contract,
                    coverage,
                    answer,
                    render_verified_rule_goal_bindings(
                        deep_session.question_goal_contract,
                        coverage,
                        answer,
                    ),
                    source="compose_verified_rule_answer",
                )
            except AnswerCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                deep_session.answer_coverage_result = result
                runtime_owner._clear_rejected_answer(deep_session)
                return json.dumps(
                    {
                        "status": "ANSWER_COVERAGE_INCOMPLETE",
                        "code": "RULE_ANSWER_GOALS_NOT_RENDERED",
                        "issues": result.get("issues", []),
                    },
                    ensure_ascii=False,
                )
            deep_session.answer_coverage_result = answer_coverage.model_dump(by_alias=True)
            return json.dumps(
                {
                    "status": "ANSWERED",
                    "answer": answer,
                    "verifiedRuleArtifactIds": list(deep_session.runtime.answer_rule_artifact_ids),
                    "goalAnswerCoverage": dict(deep_session.answer_coverage_result),
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
            semantic_activation_seal = None
            try:
                attempt = runtime_owner.kernel.propose_contract(
                    session.runtime,
                    evidence,
                    binding_hints,
                    topics=session.effective_topics(),
                    timezone_name=str(getattr(runtime_owner.settings, "business_timezone", "") or "Asia/Shanghai"),
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
                if attempt.activated:
                    seal_activation = getattr(
                        runtime_owner.kernel,
                        "seal_semantic_activation",
                        None,
                    )
                    if callable(seal_activation):
                        semantic_activation_seal = seal_activation(
                            session.runtime,
                            session.effective_topics(),
                        )
            except RuntimeError as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "GROUNDED_CONTRACT_ACTIVATION_BLOCKED",
                        "message": str(exc)[:500],
                        "nextAction": "STOP" if "TERMINAL_GUARD" in str(exc) else "REVISE_BINDINGS",
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "GROUNDED_CONTRACT_INTERNAL_ERROR",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            if attempt.activated:
                with session.lock:
                    session.active_goal_ids = list(assigned_goal_ids)
            contract_fingerprint = grounded_query_contract_fingerprint(attempt.contract)
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
                    "semanticActivation": (
                        semantic_activation_seal.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        if semantic_activation_seal is not None
                        else {}
                    ),
                    "sqlObligations": _grounded_contract_sql_obligations(attempt.contract),
                    "acceptedBindingHints": _core_visible_binding_hints(attempt.contract),
                    "assignedGoalIds": list(assigned_goal_ids),
                    "semanticCoverage": {
                        "status": (
                            "READY_TO_EXECUTE" if attempt.contract.ready and attempt.activated else "NEED_MORE_EVIDENCE"
                        ),
                        "retrievalClosed": bool(attempt.contract.ready and attempt.activated),
                        "blockingGapCount": len([gap for gap in attempt.contract.unresolved_gaps if gap.blocking]),
                        "nextAction": attempt.next_action,
                    },
                    "gaps": [gap.model_dump(by_alias=True) for gap in attempt.contract.unresolved_gaps],
                    "rejectedBindings": [item.model_dump(by_alias=True) for item in attempt.contract.rejected_bindings],
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
            reconciliation = reconcile_graph_revision_before_mutation(
                runtime
            )
            if reconciliation is not None:
                return reconciliation
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                    },
                    ensure_ascii=False,
                )
            normalized = [
                item if isinstance(item, GroundedParallelQuerySpec) else GroundedParallelQuerySpec.model_validate(item)
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
            if deep_session.query_branch_contexts:
                with deep_session.lock:
                    branch_contexts = {
                        query_id: deep_session.query_branch_contexts.get(query_id) for query_id in query_ids
                    }
                unknown_query_ids = [query_id for query_id, context in branch_contexts.items() if context is None]
                if unknown_query_ids:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "QUERY_BRANCH_NOT_DECLARED",
                            "queryIds": unknown_query_ids,
                        },
                        ensure_ascii=False,
                    )
                selected_query_ids = set(query_ids)
                dependency_conflicts: list[dict[str, Any]] = []
                for query_id, context in branch_contexts.items():
                    assert context is not None
                    selected_dependencies = sorted(selected_query_ids.intersection(context.dependency_query_ids))
                    if selected_dependencies:
                        dependency_conflicts.append(
                            {
                                "queryId": query_id,
                                "dependsOnQueryIds": selected_dependencies,
                                "requiredExecution": "SERIAL",
                            }
                        )
                if dependency_conflicts:
                    return json.dumps(
                        {
                            "status": "REJECTED",
                            "code": "PARALLEL_GOAL_DEPENDENCY_DETECTED",
                            "issues": dependency_conflicts,
                            "nextAction": "PREPARE_UPSTREAM_THEN_PUBLISH_ENTITY_SET",
                        },
                        ensure_ascii=False,
                    )

                def prepare_declared_branch(
                    raw_item: GroundedParallelQuerySpec,
                ) -> dict[str, Any]:
                    query_id = str(raw_item.query_id or "").strip()
                    context = branch_contexts[query_id]
                    assert context is not None
                    declared_goal_ids = list(context.spec.goal_ids)
                    if raw_item.goal_ids:
                        assigned_goal_ids, goal_error = _validated_goal_assignment(
                            deep_session,
                            raw_item.goal_ids,
                        )
                        if goal_error:
                            return {"queryId": query_id, **goal_error}
                        if assigned_goal_ids != declared_goal_ids:
                            return {
                                "queryId": query_id,
                                "status": "REJECTED",
                                "code": "QUERY_BRANCH_GOAL_ASSIGNMENT_IMMUTABLE",
                                "declaredGoalIds": declared_goal_ids,
                                "submittedGoalIds": assigned_goal_ids,
                            }
                    hints = _canonical_binding_hints(raw_item.binding_hints)

                    if context.status == "WAITING_VERIFIED_ENTITY_SET":
                        entity_set_ids = list(
                            dict.fromkeys(
                                str(item.entity_set_artifact_id or "").strip()
                                for item in hints.upstream_entity_bindings
                                if str(item.entity_set_artifact_id or "").strip()
                            )
                        )
                        if not entity_set_ids:
                            return {
                                "queryId": query_id,
                                "status": "WAITING_VERIFIED_ENTITY_SET",
                                "code": "DEPENDENT_QUERY_REQUIRES_VERIFIED_ENTITY_SET",
                                "dependsOnQueryIds": list(context.dependency_query_ids),
                                "nextAction": "PUBLISH_UPSTREAM_ENTITY_SET",
                            }
                        entity_sets = {item.artifact_id: item for item in deep_session.runtime.verified_entity_sets}
                        missing_entity_sets = [item for item in entity_set_ids if item not in entity_sets]
                        if missing_entity_sets:
                            return {
                                "queryId": query_id,
                                "status": "WAITING_VERIFIED_ENTITY_SET",
                                "code": "VERIFIED_ENTITY_SET_NOT_FOUND",
                                "missingEntitySetArtifactIds": missing_entity_sets,
                            }
                        source_artifact_ids = {entity_sets[item].source_query_artifact_id for item in entity_set_ids}
                        source_mismatches: list[str] = []
                        for dependency_query_id in context.dependency_query_ids:
                            upstream_context = deep_session.query_branch_contexts.get(dependency_query_id)
                            if upstream_context is None or not source_artifact_ids.intersection(
                                upstream_context.verified_artifact_ids
                            ):
                                source_mismatches.append(dependency_query_id)
                        if source_mismatches:
                            return {
                                "queryId": query_id,
                                "status": "REJECTED",
                                "code": "ENTITY_SET_SOURCE_BRANCH_MISMATCH",
                                "dependencyQueryIds": source_mismatches,
                            }
                        try:
                            branch_runtime = runtime_owner.kernel.fork_query_branch(
                                deep_session.runtime,
                                query_id,
                                workspace_topics=context.spec.topic_scope,
                                objective=context.spec.objective,
                                inherit_entity_set_ids=entity_set_ids,
                            )
                        except TypeError:
                            return {
                                "queryId": query_id,
                                "status": "BLOCKED",
                                "code": "DEPENDENT_BRANCH_LINEAGE_FORK_UNSUPPORTED",
                            }
                        with context.lock:
                            context.runtime = branch_runtime
                            context.status = "DECLARED"

                    if context.status == "PREPARED":
                        state = context.runtime
                        assert state is not None
                        contract = state.active_contract
                        return {
                            "queryId": query_id,
                            "status": "PREPARED",
                            "queryShape": (contract.query_shape if contract else ""),
                            "executionMode": state.active_execution_mode,
                            "activeGeneration": state.active_generation,
                            "contractFingerprint": (
                                grounded_query_contract_fingerprint(contract) if contract is not None else ""
                            ),
                            "goalIds": declared_goal_ids,
                            "alreadyPrepared": True,
                            "branchBudget": context.budget.report(),
                        }
                    if context.runtime is None:
                        return {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": "QUERY_BRANCH_RUNTIME_UNAVAILABLE",
                        }

                    semantic_paths = list(
                        dict.fromkeys(
                            _knowledge_relative_path(str(item or ""))
                            for item in raw_item.semantic_paths
                            if str(item or "").strip()
                        )
                    )
                    requested_ref_ids = list(
                        dict.fromkeys(
                            _canonical_progressive_ref(str(item or "").strip())
                            for item in raw_item.read_ref_ids
                            if str(item or "").strip()
                        )
                    )
                    if deep_session.execution_graph_receipt is not None:
                        undeclared_paths = [
                            path for path in semantic_paths if not context.semantic_ledger.has_path(path)
                        ]
                        declared_refs = set(context.semantic_ledger.refs())
                        undeclared_refs = [ref_id for ref_id in requested_ref_ids if ref_id not in declared_refs]
                        if undeclared_paths or undeclared_refs:
                            return {
                                "queryId": query_id,
                                "status": "REJECTED",
                                "code": ("EXECUTION_NODE_EVIDENCE_OUTSIDE_FROZEN_GRAPH"),
                                "undeclaredPaths": undeclared_paths,
                                "undeclaredRefs": undeclared_refs,
                                "discoverySnapshotFingerprint": (
                                    deep_session.execution_graph_receipt.discovery_snapshot_fingerprint
                                ),
                            }
                    unresolved_ref_ids: list[str] = []
                    for ref_id in requested_ref_ids:
                        if ref_id in set(context.semantic_ledger.refs()):
                            continue
                        try:
                            resolved = runtime_owner.semantic_catalog.read(
                                ref_id=ref_id,
                                max_chars=1,
                                offset=0,
                            )
                        except Exception:
                            resolved = {}
                        resolved_path = str((resolved or {}).get("path") or "").strip()
                        if resolved_path:
                            semantic_paths.append(_knowledge_relative_path(resolved_path))
                        else:
                            unresolved_ref_ids.append(ref_id)
                    if unresolved_ref_ids:
                        return {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": "SEMANTIC_REF_PATH_UNRESOLVED",
                            "missingRefs": unresolved_ref_ids,
                        }
                    semantic_paths = list(dict.fromkeys(semantic_paths))
                    read_receipts: list[dict[str, Any]] = []
                    try:
                        with context.budget.stage("semantic_retrieval"):
                            for semantic_path in semantic_paths:
                                evidence, newly_read = _read_exact_branch_semantic_path(
                                    runtime_owner.semantic_catalog,
                                    context,
                                    semantic_path,
                                )
                                receipt = _core_visible_semantic_receipt(evidence)
                                receipt["newlyRead"] = newly_read
                                receipt["branchId"] = query_id
                                read_receipts.append(receipt)
                        context.budget.consume_contract_attempt()
                        with context.budget.stage("contract"):
                            attempt = runtime_owner.kernel.propose_contract(
                                context.runtime,
                                context.semantic_ledger.evidence(),
                                hints,
                                topics=context.effective_topics(),
                                timezone_name=str(
                                    getattr(runtime_owner.settings, "business_timezone", "") or "Asia/Shanghai"
                                ),
                            )
                            assignment_issues = _goal_assignment_contract_issues(
                                deep_session,
                                declared_goal_ids,
                                attempt.contract,
                            )
                            if assignment_issues:
                                with context.lock:
                                    context.status = "CONTRACT_GAPPED"
                                    context.last_gaps = assignment_issues
                                replan_evidence = _record_execution_graph_replan_evidence(
                                    deep_session,
                                    query_node_id=query_id,
                                    trigger_kind="DATA_GAP",
                                    source_stage="CONTRACT",
                                    code=("QUERY_GOAL_ASSIGNMENT_MISMATCH"),
                                    details={
                                        "issues": assignment_issues,
                                        "goalIds": declared_goal_ids,
                                    },
                                    runtime_budget=runtime.context.budget,
                                )
                                return {
                                    "queryId": query_id,
                                    "status": "BLOCKED",
                                    "code": "QUERY_GOAL_ASSIGNMENT_MISMATCH",
                                    "issues": assignment_issues,
                                    "replanEvidence": (
                                        _execution_graph_replan_evidence_report(replan_evidence)
                                        if replan_evidence is not None
                                        else {}
                                    ),
                                    "semanticReceipts": read_receipts,
                                    "branchBudget": context.budget.report(),
                                }
                            if attempt.contract.ready:
                                attempt = runtime_owner.kernel.activate_contract(
                                    context.runtime,
                                    attempt.attempt_id,
                                )
                    except GroundedRuntimeBudgetExceeded:
                        raise
                    except GroundedBranchBudgetExceeded as exc:
                        with context.lock:
                            context.status = "BUDGET_BLOCKED"
                        return {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": exc.code,
                            "branchBudget": exc.report,
                        }
                    except Exception as exc:
                        with context.lock:
                            context.status = "FAILED"
                        return {
                            "queryId": query_id,
                            "status": "BLOCKED",
                            "code": "PARALLEL_CONTRACT_PREPARATION_FAILED",
                            "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                            "semanticReceipts": read_receipts,
                            "branchBudget": context.budget.report(),
                        }

                    if not attempt.activated:
                        gaps = [gap.model_dump(by_alias=True) for gap in attempt.contract.unresolved_gaps]
                        with context.lock:
                            context.status = "CONTRACT_GAPPED"
                            context.last_gaps = gaps
                        replan_evidence = _record_execution_graph_replan_evidence(
                            deep_session,
                            query_node_id=query_id,
                            trigger_kind="DATA_GAP",
                            source_stage="CONTRACT",
                            code="PARALLEL_CONTRACT_NOT_READY",
                            details={
                                "contractStatus": (attempt.contract.status),
                                "gaps": gaps,
                                "goalIds": declared_goal_ids,
                            },
                            runtime_budget=runtime.context.budget,
                        )
                        return {
                            "queryId": query_id,
                            "status": attempt.contract.status,
                            "code": "PARALLEL_CONTRACT_NOT_READY",
                            "gaps": gaps,
                            "replanEvidence": (
                                _execution_graph_replan_evidence_report(replan_evidence)
                                if replan_evidence is not None
                                else {}
                            ),
                            "semanticReceipts": read_receipts,
                            "branchReadControl": _grounded_branch_read_control(context),
                            "branchBudget": context.budget.report(),
                        }
                    with context.lock:
                        context.status = "PREPARED"
                        context.last_gaps = []
                    with deep_session.lock:
                        deep_session.parallel_branches[query_id] = context.runtime
                        deep_session.parallel_branch_goal_ids[query_id] = declared_goal_ids
                    return {
                        "queryId": query_id,
                        "status": "PREPARED",
                        "queryShape": attempt.contract.query_shape,
                        "executionMode": attempt.execution_mode,
                        "activeGeneration": attempt.active_generation,
                        "contractFingerprint": (grounded_query_contract_fingerprint(attempt.contract)),
                        "sqlObligations": _grounded_contract_sql_obligations(attempt.contract),
                        "goalIds": declared_goal_ids,
                        "semanticReceipts": read_receipts,
                        "branchReadControl": _grounded_branch_read_control(context),
                        "branchBudget": context.budget.report(),
                        "nextAction": (
                            "EXECUTE_BATCH"
                            if attempt.execution_mode != "CORE_SQL_REQUIRED"
                            else "AUTHOR_SQL_THEN_EXECUTE_BATCH"
                        ),
                    }

                declared_results: list[dict[str, Any]] = []
                with ThreadPoolExecutor(
                    max_workers=min(
                        len(normalized),
                        runtime_owner.parallel_max_workers,
                    ),
                    thread_name_prefix="grounded-prepare",
                ) as pool:
                    futures = {pool.submit(prepare_declared_branch, item): item.query_id for item in normalized}
                    for future in as_completed(futures):
                        declared_results.append(future.result())
                declared_results.sort(key=lambda item: query_ids.index(str(item.get("queryId") or "")))
                ready_count = sum(item.get("status") == "PREPARED" for item in declared_results)
                return json.dumps(
                    {
                        "status": (
                            "PREPARED" if ready_count == len(normalized) else "PARTIAL" if ready_count else "BLOCKED"
                        ),
                        "compatMode": "BRANCH_SCOPED_V2",
                        "preparedCount": ready_count,
                        "preparedInParallel": len(normalized) > 1,
                        "workerCount": min(
                            len(normalized),
                            runtime_owner.parallel_max_workers,
                        ),
                        "queries": declared_results,
                    },
                    ensure_ascii=False,
                    default=str,
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
                        timezone_name=str(getattr(runtime_owner.settings, "business_timezone", "") or "Asia/Shanghai"),
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
                            "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        }
                    )
                    continue
                if not attempt.activated:
                    prepared.append(
                        {
                            "queryId": query_id,
                            "status": attempt.contract.status,
                            "code": "PARALLEL_CONTRACT_NOT_READY",
                            "gaps": [gap.model_dump(by_alias=True) for gap in attempt.contract.unresolved_gaps],
                        }
                    )
                    continue
                with deep_session.lock:
                    deep_session.parallel_branches[query_id] = branch
                    deep_session.parallel_branch_goal_ids[query_id] = list(assigned_goal_ids)
                prepared.append(
                    {
                        "queryId": query_id,
                        "status": "PREPARED",
                        "queryShape": attempt.contract.query_shape,
                        "executionMode": attempt.execution_mode,
                        "activeGeneration": attempt.active_generation,
                        "contractFingerprint": grounded_query_contract_fingerprint(attempt.contract),
                        "sqlObligations": _grounded_contract_sql_obligations(attempt.contract),
                        "goalIds": list(assigned_goal_ids),
                        "nextAction": (
                            "EXECUTE_BATCH"
                            if attempt.execution_mode != "CORE_SQL_REQUIRED"
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
            reconciliation = reconcile_graph_revision_before_mutation(
                runtime
            )
            if reconciliation is not None:
                return reconciliation
            if runtime.context.budget is not None:
                runtime.context.budget.checkpoint()
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
                branch_by_id = {query_id: deep_session.parallel_branches.get(query_id) for query_id in query_ids}
                branch_context_by_id = {
                    query_id: deep_session.query_branch_contexts.get(query_id) for query_id in query_ids
                }
            missing_branches = [query_id for query_id, branch in branch_by_id.items() if branch is None]
            if missing_branches:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "PARALLEL_BRANCH_NOT_PREPARED",
                        "queryIds": missing_branches,
                    },
                    ensure_ascii=False,
                )

            snapshot_requirement = None
            shared_data_snapshot = None
            with deep_session.lock:
                graph_receipt = (
                    deep_session.execution_graph_receipt.model_copy(deep=True)
                    if deep_session.execution_graph_receipt is not None
                    else None
                )
                graph_edges = [edge.model_copy(deep=True) for edge in deep_session.execution_graph_edges]
                goal_contract = (
                    deep_session.question_goal_contract.model_copy(deep=True)
                    if deep_session.question_goal_contract is not None
                    else None
                )
                goal_ids_by_query_id = {
                    query_id: list(context.spec.goal_ids)
                    for query_id, context in (deep_session.query_branch_contexts.items())
                }
                runtime_semantic_seal = (
                    deep_session.runtime.semantic_activation_seal.model_copy(deep=True)
                    if deep_session.runtime.semantic_activation_seal is not None
                    else None
                )
                retained_graph_snapshot = (
                    deep_session.execution_graph_data_snapshot.model_copy(deep=True)
                    if deep_session.execution_graph_data_snapshot is not None
                    else None
                )
            graph_query_ids = list(graph_receipt.node_ids.values()) if graph_receipt is not None else []
            if graph_receipt is not None and len(graph_query_ids) > 1:
                if goal_contract is None:
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED",
                        },
                        ensure_ascii=False,
                    )
                snapshot_requirement = derive_multi_query_snapshot_requirement(
                    graph_query_ids,
                    receipt_node_ids=graph_receipt.node_ids,
                    graph_edges=graph_edges,
                    goal_contract=goal_contract,
                    goal_ids_by_query_id=goal_ids_by_query_id,
                )
                semantic_activation_fingerprint = str(
                    graph_receipt.semantic_activation_fingerprint
                    or getattr(
                        runtime_semantic_seal,
                        "semantic_activation_fingerprint",
                        "",
                    )
                    or ""
                ).strip()
                receipt_seal_fingerprint = str(graph_receipt.semantic_activation_seal_fingerprint or "").strip()
                current_seal_fingerprint = str(
                    getattr(
                        runtime_semantic_seal,
                        "seal_fingerprint",
                        "",
                    )
                    or ""
                ).strip()
                current_semantic_fingerprint = str(
                    getattr(
                        runtime_semantic_seal,
                        "semantic_activation_fingerprint",
                        "",
                    )
                    or ""
                ).strip()
                current_semantic_topics = list(
                    getattr(
                        runtime_semantic_seal,
                        "exact_topics",
                        [],
                    )
                    or []
                )
                semantic_authority_available = getattr(
                    runtime_owner.kernel,
                    "semantic_activation_authority_available",
                    None,
                )
                semantic_authority_required = bool(
                    callable(semantic_authority_available) and semantic_authority_available()
                )
                if semantic_authority_required:
                    revalidate_activation = getattr(
                        runtime_owner.kernel,
                        "revalidate_semantic_activation",
                        None,
                    )
                    try:
                        if callable(revalidate_activation):
                            revalidate_activation(deep_session.runtime)
                    except Exception as exc:
                        return json.dumps(
                            {
                                "status": "BLOCKED",
                                "code": (str(exc).partition(":")[0] or "SEMANTIC_ACTIVATION_STALE"),
                                "message": "%s:%s"
                                % (
                                    type(exc).__name__,
                                    str(exc)[:400],
                                ),
                                "nextAction": ("REOPEN_GRAPH_BEFORE_EXECUTION"),
                            },
                            ensure_ascii=False,
                        )
                if semantic_authority_required and (
                    not semantic_activation_fingerprint
                    or not receipt_seal_fingerprint
                    or not current_seal_fingerprint
                    or receipt_seal_fingerprint != current_seal_fingerprint
                    or semantic_activation_fingerprint != current_semantic_fingerprint
                    or list(graph_receipt.semantic_activation_topics) != current_semantic_topics
                ):
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": ("EXECUTION_GRAPH_SEMANTIC_ACTIVATION_STALE"),
                            "nextAction": ("REOPEN_GRAPH_BEFORE_EXECUTION"),
                        },
                        ensure_ascii=False,
                    )
                shared_data_snapshot = retained_graph_snapshot
                if (
                    shared_data_snapshot is not None
                    and semantic_activation_fingerprint
                    and str(shared_data_snapshot.semantic_activation_fingerprint or "").strip()
                    != semantic_activation_fingerprint
                ):
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": ("EXECUTION_GRAPH_DATA_SNAPSHOT_ACTIVATION_MISMATCH"),
                        },
                        ensure_ascii=False,
                    )
                if shared_data_snapshot is None:
                    capture_snapshot = getattr(
                        runtime_owner.kernel,
                        "capture_data_snapshot",
                        None,
                    )
                    if callable(capture_snapshot):
                        candidate_snapshot = capture_snapshot(semantic_activation_fingerprint)
                        with deep_session.lock:
                            if deep_session.execution_graph_data_snapshot is None:
                                deep_session.execution_graph_data_snapshot = candidate_snapshot.model_copy(deep=True)
                            shared_data_snapshot = deep_session.execution_graph_data_snapshot.model_copy(deep=True)
                snapshot_preflight_issues = validate_query_bundle_snapshots(
                    [
                        QueryBundle(data_snapshot=shared_data_snapshot)
                        if shared_data_snapshot is not None
                        else QueryBundle()
                        for _query_id in graph_query_ids
                    ],
                    require_atomic_multi_query=(snapshot_requirement.require_atomic_multi_query),
                )
                if snapshot_preflight_issues:
                    gap_payload = [
                        {
                            "code": issue,
                            "message": (
                                "The frozen multi-query graph cannot preserve the "
                                "required data-snapshot semantics with the active "
                                "datasource capability."
                            ),
                            "blocking": True,
                            "requiredCapability": {
                                "atomicMultiQuery": (snapshot_requirement.require_atomic_multi_query)
                            },
                        }
                        for issue in snapshot_preflight_issues
                    ]
                    with deep_session.lock:
                        for query_id in graph_query_ids:
                            branch_context = deep_session.query_branch_contexts.get(query_id)
                            if branch_context is None:
                                continue
                            with branch_context.lock:
                                branch_context.status = "CONTRACT_GAPPED"
                                branch_context.last_gaps = [dict(item) for item in gap_payload]
                    replan_evidence = _record_execution_graph_replan_evidence(
                        deep_session,
                        query_node_id=graph_query_ids[0],
                        trigger_kind="DATA_GAP",
                        source_stage="DATASOURCE",
                        code=("MULTI_QUERY_SNAPSHOT_CONTRACT_UNSATISFIED"),
                        details={
                            "snapshotIssues": (snapshot_preflight_issues),
                            "requiredCapability": (
                                snapshot_requirement.model_dump(
                                    by_alias=True,
                                    mode="json",
                                )
                            ),
                        },
                        runtime_budget=runtime.context.budget,
                    )
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "MULTI_QUERY_SNAPSHOT_CONTRACT_UNSATISFIED",
                            "snapshotIssues": snapshot_preflight_issues,
                            "snapshotRequirement": snapshot_requirement.model_dump(
                                by_alias=True,
                                mode="json",
                            ),
                            "consistencyMode": str(
                                getattr(
                                    shared_data_snapshot,
                                    "consistency_mode",
                                    "UNSUPPORTED",
                                )
                                or "UNSUPPORTED"
                            ),
                            "replanEvidence": (
                                _execution_graph_replan_evidence_report(replan_evidence)
                                if replan_evidence is not None
                                else {}
                            ),
                            "nextAction": (
                                "REOPEN_GRAPH_AND_MERGE_ATOMIC_GOALS_OR_CONFIGURE_TRUSTED_SNAPSHOT_CAPABILITY"
                            ),
                        },
                        ensure_ascii=False,
                    )

            def execute_one(
                item: GroundedParallelExecutionSpec,
            ) -> tuple[str, GroundedRuntimeSession, dict[str, Any]]:
                query_id = str(item.query_id).strip()
                branch = branch_by_id[query_id]
                branch_context = branch_context_by_id.get(query_id)
                assert branch is not None
                try:
                    if runtime.context.budget is not None:
                        runtime.context.budget.checkpoint()
                    if branch_context is not None:
                        with branch_context.lock:
                            branch_context.status = "EXECUTING"
                    if (
                        str(
                            getattr(
                                branch.active_execution_mode,
                                "value",
                                branch.active_execution_mode,
                            )
                        )
                        == "CORE_SQL_REQUIRED"
                    ):
                        if not str(item.sql or "").strip():
                            return (
                                query_id,
                                branch,
                                {
                                    "queryId": query_id,
                                    "status": "BLOCKED",
                                    "code": "CORE_SQL_CANDIDATE_REQUIRED",
                                },
                            )
                        contract = branch.active_contract
                        if contract is None:
                            raise RuntimeError("parallel branch has no active Contract")
                        submitted = runtime_owner.kernel.submit_sql_candidate(
                            branch,
                            item.sql,
                            expected_generation=branch.active_generation,
                            expected_contract_fingerprint=(grounded_query_contract_fingerprint(contract)),
                            rationale=item.rationale or reason,
                            evidence_refs=item.evidence_ref_ids,
                        )
                        if submitted.status != "ACCEPTED":
                            return (
                                query_id,
                                branch,
                                {
                                    "queryId": query_id,
                                    "status": submitted.status,
                                    "nextAction": submitted.next_action,
                                    "gaps": submitted.validation_gaps,
                                },
                            )
                    budget = runtime.context.budget
                    if branch_context is not None:
                        branch_context.budget.consume_doris_query()
                    elif budget is not None:
                        budget.consume_doris_query(name="parallel.%s" % query_id)
                    if branch_context is not None:
                        with branch_context.budget.stage("doris"):
                            run_result = runtime_owner.kernel.execute_active(
                                branch,
                                run_id="%s__%s" % (runtime.context.run_id, query_id),
                                runtime_budget=budget,
                                data_snapshot_contract=shared_data_snapshot,
                                **runtime_owner._population_execution_kwargs(
                                    deep_session,
                                    branch,
                                    query_node_id=query_id,
                                ),
                            )
                    elif budget is None:
                        run_result = runtime_owner.kernel.execute_active(
                            branch,
                            run_id="%s__%s" % (runtime.context.run_id, query_id),
                            data_snapshot_contract=shared_data_snapshot,
                            **runtime_owner._population_execution_kwargs(
                                deep_session,
                                branch,
                                query_node_id=query_id,
                            ),
                        )
                    else:
                        with budget.stage("doris.parallel.%s" % query_id):
                            run_result = runtime_owner.kernel.execute_active(
                                branch,
                                run_id="%s__%s" % (runtime.context.run_id, query_id),
                                runtime_budget=budget,
                                data_snapshot_contract=shared_data_snapshot,
                                **runtime_owner._population_execution_kwargs(
                                    deep_session,
                                    branch,
                                    query_node_id=query_id,
                                ),
                            )
                    delayed_reports = [
                        report.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        for task_result in list(
                            getattr(run_result, "task_results", [])
                            or []
                        )
                        for report in list(
                            getattr(
                                task_result,
                                "freshness_reports",
                                [],
                            )
                            or []
                        )
                        if (
                            not bool(
                                getattr(
                                    report,
                                    "coverage_complete",
                                    True,
                                )
                            )
                            or str(
                                getattr(report, "status", "")
                                or ""
                            )
                            == "STALE_REQUIRES_GRAPH_REPREPARATION"
                        )
                    ]
                    pre_verification_failure = (
                        _classify_grounded_execution_result(run_result)
                    )
                    if (
                        pre_verification_failure.disposition == "NONE"
                        and not delayed_reports
                    ):
                        if branch_context is not None:
                            with branch_context.budget.stage("evidence"):
                                verified = runtime_owner.kernel.verify_active(
                                    branch
                                )
                        elif budget is None:
                            verified = runtime_owner.kernel.verify_active(
                                branch
                            )
                        else:
                            with budget.stage(
                                "evidence.parallel.%s" % query_id
                            ):
                                verified = runtime_owner.kernel.verify_active(
                                    branch
                                )
                    else:
                        verified = VerifiedEvidence(passed=False)
                    replan_evidence = None
                    failure = (
                        pre_verification_failure
                        if pre_verification_failure.disposition != "NONE"
                        else GroundedExecutionFailureClassification()
                        if delayed_reports
                        else _classify_grounded_execution_result(
                            run_result,
                            verified,
                        )
                    )
                    result_status = "VERIFIED"
                    result_code = ""
                    result_disposition = failure.disposition
                    if failure.terminal:
                        if branch_context is not None:
                            with branch_context.lock:
                                branch_context.status = "TERMINAL_BLOCKED"
                        result_status = (
                            "ACCESS_DENIED"
                            if failure.disposition == "SECURITY_TERMINAL"
                            else "OPERATIONAL_FAILURE"
                        )
                        result_code = failure.code
                    elif delayed_reports:
                        if branch_context is not None:
                            with branch_context.lock:
                                branch_context.status = "FAILED"
                        result_status = "REPLAN_REQUIRED"
                        result_disposition = "RECOVERABLE_EXECUTION"
                        result_code = (
                            "TABLE_FRESHNESS_COVERAGE_INCOMPLETE"
                        )
                        replan_evidence = _record_execution_graph_replan_evidence(
                            deep_session,
                            query_node_id=query_id,
                            trigger_kind="TABLE_DELAY",
                            source_stage="DATASOURCE",
                            code=("TABLE_FRESHNESS_COVERAGE_INCOMPLETE"),
                            details={
                                "freshnessReports": delayed_reports,
                            },
                            runtime_budget=budget,
                        )
                    elif failure.disposition == "RECOVERABLE_EXECUTION":
                        if branch_context is not None:
                            with branch_context.lock:
                                branch_context.status = "FAILED"
                        result_status = "REPLAN_REQUIRED"
                        result_code = failure.code
                        replan_evidence = _record_execution_graph_replan_evidence(
                            deep_session,
                            query_node_id=query_id,
                            trigger_kind="EXECUTION_ERROR",
                            source_stage="EXECUTION",
                            code=failure.code,
                            details={
                                "failureCodes": list(failure.codes),
                                "message": failure.message,
                            },
                            runtime_budget=budget,
                        )
                    artifact = (
                        runtime_owner.kernel.latest_verified_query_artifact(
                            branch
                        )
                        if result_status == "VERIFIED"
                        else None
                    )
                    return (
                        query_id,
                        branch,
                        {
                            "queryId": query_id,
                            "status": result_status,
                            "code": result_code,
                            "failureDisposition": result_disposition,
                            "message": failure.message,
                            "queryArtifactId": artifact.artifact_id if artifact else "",
                            "rowCount": len(run_result.merged_query_bundle.rows),
                            "tables": list(run_result.merged_query_bundle.tables),
                            "resultArtifacts": _grounded_result_artifact_receipts(run_result),
                            "blockingGaps": [gap.model_dump(by_alias=True) for gap in verified.blocking_gaps],
                            "replanEvidence": (
                                _execution_graph_replan_evidence_report(replan_evidence)
                                if replan_evidence is not None
                                else {}
                            ),
                            "branchBudget": (branch_context.budget.report() if branch_context is not None else {}),
                        },
                    )
                except GroundedRuntimeBudgetExceeded:
                    raise
                except GroundedBranchBudgetExceeded as exc:
                    if branch_context is not None:
                        with branch_context.lock:
                            branch_context.status = "BUDGET_BLOCKED"
                    return (
                        query_id,
                        branch,
                        {
                            "queryId": query_id,
                            "status": "OPERATIONAL_FAILURE",
                            "code": exc.code,
                            "failureDisposition": (
                                "OPERATIONAL_TERMINAL"
                            ),
                            "branchBudget": exc.report,
                        },
                    )
                except Exception as exc:
                    failure = _classify_grounded_execution_exception(exc)
                    if branch_context is not None:
                        with branch_context.lock:
                            branch_context.status = "TERMINAL_BLOCKED"
                    return (
                        query_id,
                        branch,
                        {
                            "queryId": query_id,
                            "status": (
                                "ACCESS_DENIED"
                                if failure.disposition
                                == "SECURITY_TERMINAL"
                                else "OPERATIONAL_FAILURE"
                            ),
                            "code": failure.code,
                            "message": failure.message,
                            "failureDisposition": failure.disposition,
                            "replanEvidence": {},
                        },
                    )

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
                    next(query_id for query_id, candidate in branch_by_id.items() if candidate is branch)
                )
            )
            terminal_results = [
                item
                for item in results
                if item.get("failureDisposition")
                in {
                    "SECURITY_TERMINAL",
                    "OPERATIONAL_TERMINAL",
                }
            ]
            if terminal_results:
                security_terminal = any(
                    item.get("failureDisposition")
                    == "SECURITY_TERMINAL"
                    for item in terminal_results
                )
                operational_failure = {
                    "code": (
                        "GROUNDED_EXECUTION_SECURITY_FAILURE"
                        if security_terminal
                        else "GROUNDED_EXECUTION_OPERATIONAL_FAILURE"
                    ),
                    "failures": [
                        {
                            "queryId": item.get("queryId"),
                            "code": item.get("code"),
                            "disposition": item.get(
                                "failureDisposition"
                            ),
                            "message": item.get("message"),
                        }
                        for item in terminal_results
                    ],
                    "retryable": False,
                }
                with deep_session.lock:
                    if deep_session.execution_graph_replan_evidence:
                        deep_session.execution_graph_history.append(
                            {
                                "status": (
                                    "REPLAN_EVIDENCE_REVOKED_BY_TERMINAL_BATCH"
                                ),
                                "evidence": [
                                    _execution_graph_replan_evidence_report(
                                        item
                                    )
                                    for item in deep_session.execution_graph_replan_evidence.values()
                                ],
                            }
                        )
                    deep_session.execution_graph_replan_evidence = {}
                    deep_session.execution_graph_revision_discovery_evidence_id = ""
                    deep_session.execution_graph_revision_discovery_evidence_ids = []
                    deep_session.operational_failure = dict(
                        operational_failure
                    )
                    deep_session.runtime.phase = (
                        "SECURITY_BLOCKED"
                        if security_terminal
                        else "OPERATIONAL_FAILURE"
                    )
                    for query_id in query_ids:
                        context = (
                            deep_session.query_branch_contexts.get(
                                query_id
                            )
                        )
                        if context is not None:
                            with context.lock:
                                context.status = "TERMINAL_BLOCKED"
                        deep_session.parallel_branches.pop(
                            query_id,
                            None,
                        )
                        deep_session.parallel_branch_goal_ids.pop(
                            query_id,
                            None,
                        )
                for item in results:
                    item["queryArtifactId"] = ""
                    item["resultArtifacts"] = []
                    item["replanEvidence"] = {}
                    if item.get("status") == "VERIFIED":
                        item["status"] = (
                            "CANCELLED_BY_TERMINAL_BATCH_FAILURE"
                        )
                results.sort(
                    key=lambda item: query_ids.index(
                        str(item.get("queryId") or "")
                    )
                )
                return json.dumps(
                    {
                        "status": (
                            "ACCESS_DENIED"
                            if security_terminal
                            else "OPERATIONAL_FAILURE"
                        ),
                        "code": operational_failure["code"],
                        "adoptedArtifactIds": [],
                        "queries": results,
                        "nextAction": "STOP",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            if graph_receipt is not None and len(graph_query_ids) > 1 and successful_branches:
                snapshot_postflight_issues = validate_query_bundle_snapshots(
                    [
                        QueryBundle(
                            data_snapshot=shared_data_snapshot
                            if shared_data_snapshot is not None
                            else DataSnapshotContract()
                        ),
                        *[
                            branch.run_result.merged_query_bundle
                            for branch in successful_branches
                            if branch.run_result is not None
                        ],
                    ],
                    require_atomic_multi_query=bool(
                        snapshot_requirement and snapshot_requirement.require_atomic_multi_query
                    ),
                )
                if snapshot_postflight_issues:
                    affected_query_ids = [
                        query_id
                        for query_id in query_ids
                        if any(
                            branch_by_id.get(query_id) is branch
                            for branch in successful_branches
                        )
                    ]
                    with deep_session.lock:
                        for query_id in affected_query_ids:
                            branch_context = deep_session.query_branch_contexts.get(query_id)
                            if branch_context is not None:
                                with branch_context.lock:
                                    branch_context.status = "SNAPSHOT_BLOCKED"
                    if runtime.context.budget is not None:
                        runtime.context.budget.checkpoint()
                    snapshot_replan_evidences = [
                        evidence
                        for query_id in affected_query_ids
                        if (
                            evidence
                            := _record_execution_graph_replan_evidence(
                                deep_session,
                                query_node_id=query_id,
                                trigger_kind="DATA_GAP",
                                source_stage="DATASOURCE",
                                code=(
                                    "MULTI_QUERY_SNAPSHOT_POSTFLIGHT_FAILED"
                                ),
                                details={
                                    "snapshotIssues": (
                                        snapshot_postflight_issues
                                    ),
                                    "requiredCapability": (
                                        snapshot_requirement.model_dump(
                                            by_alias=True,
                                            mode="json",
                                        )
                                        if snapshot_requirement
                                        is not None
                                        else {}
                                    ),
                                },
                                runtime_budget=runtime.context.budget,
                            )
                        )
                    ]
                    for result in results:
                        if result.get("queryId") not in set(
                            affected_query_ids
                        ):
                            continue
                        result.update(
                            {
                                "status": "REPLAN_REQUIRED",
                                "code": (
                                    "MULTI_QUERY_SNAPSHOT_POSTFLIGHT_FAILED"
                                ),
                                "queryArtifactId": "",
                                "resultArtifacts": [],
                                "replanEvidence": next(
                                    (
                                        _execution_graph_replan_evidence_report(
                                            evidence
                                        )
                                        for evidence in snapshot_replan_evidences
                                        if evidence.source_query_node_id
                                        == result.get("queryId")
                                    ),
                                    {},
                                ),
                            }
                        )
                    results.sort(key=lambda item: query_ids.index(str(item.get("queryId") or "")))
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "MULTI_QUERY_SNAPSHOT_POSTFLIGHT_FAILED",
                            "snapshotIssues": snapshot_postflight_issues,
                            "snapshotRequirement": snapshot_requirement.model_dump(
                                by_alias=True,
                                mode="json",
                            )
                            if snapshot_requirement is not None
                            else {},
                            "adoptedArtifactIds": [],
                            "queries": results,
                            "replanEvidenceSet": [
                                _execution_graph_replan_evidence_report(
                                    evidence
                                )
                                for evidence in snapshot_replan_evidences
                            ],
                            "replanEvidenceSetFingerprint": (
                                grounded_execution_graph_replan_evidence_set_fingerprint(
                                    snapshot_replan_evidences
                                )
                            ),
                            "nextAction": (
                                "REOPEN_GRAPH_FOR_SNAPSHOT_RECOVERY"
                            ),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
            result_by_query_id = {
                str(item.get("queryId") or ""): item
                for item in results
            }
            query_id_by_branch_identity = {
                id(branch): query_id
                for query_id, branch in branch_by_id.items()
            }

            def authorize_population_before_adoption(
                branch: GroundedRuntimeSession,
                staged_artifacts: Sequence[
                    GroundedVerifiedQueryArtifact
                ],
            ) -> bool:
                query_id = query_id_by_branch_identity.get(
                    id(branch),
                    "",
                )
                result = result_by_query_id.get(query_id)
                if not query_id or result is None or not staged_artifacts:
                    return False
                with deep_session.lock:
                    for artifact in staged_artifacts:
                        deep_session.population_staged_query_artifacts[
                            artifact.artifact_id
                        ] = artifact.model_copy(deep=True)
                try:
                    result["populationPostGate"] = (
                        runtime_owner._commit_population_node_post(
                            deep_session,
                            query_id,
                        )
                    )
                    return True
                except RuntimeError as exc:
                    with deep_session.lock:
                        for artifact in staged_artifacts:
                            deep_session.population_staged_query_artifacts.pop(
                                artifact.artifact_id,
                                None,
                            )
                    result.update(
                        {
                            "status": "BLOCKED",
                            "code": (
                                "POPULATION_POST_RESULT_REJECTED"
                            ),
                            "message": str(exc)[:500],
                            "queryArtifactId": "",
                            "resultArtifacts": [],
                            "failureDisposition": (
                                "OPERATIONAL_TERMINAL"
                            ),
                        }
                    )
                    branch_context = (
                        deep_session.query_branch_contexts.get(query_id)
                    )
                    if branch_context is not None:
                        with branch_context.lock:
                            branch_context.status = "TERMINAL_BLOCKED"
                    return False

            if runtime_owner.population_gate_enforced:
                adopted = runtime_owner.kernel.adopt_verified_branches(
                    deep_session.runtime,
                    successful_branches,
                    pre_adoption_authorizer=(
                        authorize_population_before_adoption
                    ),
                )
            else:
                adopted = runtime_owner.kernel.adopt_verified_branches(
                    deep_session.runtime,
                    successful_branches,
                )
            artifact_ids = {
                artifact.artifact_id for artifact in adopted
            }
            population_accepted_artifact_ids = set(artifact_ids)
            with deep_session.lock:
                for artifact_id in list(
                    deep_session.population_staged_query_artifacts
                ):
                    if artifact_id in artifact_ids:
                        deep_session.population_staged_query_artifacts.pop(
                            artifact_id,
                            None,
                        )
            population_post_rejections = [
                item
                for item in results
                if item.get("code")
                == "POPULATION_POST_RESULT_REJECTED"
            ]
            if population_post_rejections:
                with deep_session.lock:
                    deep_session.operational_failure = {
                        "code": (
                            "POPULATION_POST_RESULT_REJECTED"
                        ),
                        "failures": [
                            {
                                "queryId": item.get("queryId"),
                                "message": item.get("message"),
                            }
                            for item in population_post_rejections
                        ],
                        "retryable": False,
                    }
                    deep_session.runtime.phase = "OPERATIONAL_FAILURE"
            with deep_session.lock:
                for result in results:
                    artifact_id = str(result.get("queryArtifactId") or "")
                    query_id = str(result.get("queryId") or "")
                    if artifact_id and artifact_id in population_accepted_artifact_ids:
                        deep_session.artifact_goal_ids[artifact_id] = list(
                            deep_session.parallel_branch_goal_ids.get(query_id) or []
                        )
                        if runtime_owner.population_gate_enforced:
                            deep_session.population_artifact_query_node_ids[
                                artifact_id
                            ] = query_id
                        branch_context = deep_session.query_branch_contexts.get(query_id)
                        if branch_context is not None:
                            with branch_context.lock:
                                branch_context.status = "VERIFIED"
                                if artifact_id not in branch_context.verified_artifact_ids:
                                    branch_context.verified_artifact_ids.append(artifact_id)
                for query_id in query_ids:
                    deep_session.parallel_branches.pop(query_id, None)
                    deep_session.parallel_branch_goal_ids.pop(query_id, None)
            results.sort(key=lambda item: query_ids.index(str(item.get("queryId") or "")))
            accepted_adopted = [item for item in adopted if item.artifact_id in population_accepted_artifact_ids]
            replan_required = any(
                item.get("status") == "REPLAN_REQUIRED"
                for item in results
            )
            batch_replan_evidence_ids = {
                str(
                    (item.get("replanEvidence") or {}).get(
                        "evidenceId"
                    )
                    or ""
                )
                for item in results
                if isinstance(item.get("replanEvidence"), dict)
            }
            batch_replan_evidences = [
                evidence
                for evidence in _current_execution_graph_replan_evidence(
                    deep_session
                )
                if evidence.evidence_id
                in batch_replan_evidence_ids
            ]
            return json.dumps(
                {
                    "status": (
                        "OPERATIONAL_FAILURE"
                        if population_post_rejections
                        else "VERIFIED"
                        if len(accepted_adopted) == len(normalized)
                        else "PARTIAL"
                        if accepted_adopted
                        else "REPLAN_REQUIRED"
                        if replan_required
                        else "FAILED"
                    ),
                    "reason": str(reason or "")[:500],
                    "executedInParallel": len(normalized) > 1,
                    "workerCount": min(len(normalized), runtime_owner.parallel_max_workers),
                    "adoptedArtifactIds": [item.artifact_id for item in accepted_adopted],
                    "queries": results,
                    "replanRequired": replan_required,
                    "replanEvidenceSet": [
                        _execution_graph_replan_evidence_report(
                            evidence
                        )
                        for evidence in batch_replan_evidences
                    ],
                    "replanEvidenceSetFingerprint": (
                        grounded_execution_graph_replan_evidence_set_fingerprint(
                            batch_replan_evidences
                        )
                        if batch_replan_evidences
                        else ""
                    ),
                    "nextAction": (
                        "STOP"
                        if population_post_rejections
                        else "REOPEN_GRAPH_FOR_RECOVERY"
                        if replan_required
                        else "CONTINUE_QUERYING_OR_FINALIZE"
                        if accepted_adopted
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
            """Validate Core SQL and atomically execute it only when accepted."""

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
                            "USE_LATEST_CONTRACT" if stale else "STOP" if terminal else "PROPOSE_GROUNDED_CONTRACT"
                        ),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "SQL_CANDIDATE_INTERNAL_ERROR",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            internal_error = attempt.status == "VALIDATOR_INTERNAL_ERROR"
            candidate_payload = {
                "candidateId": attempt.candidate_id,
                "status": "BLOCKED" if internal_error else attempt.status,
                "code": ("SQL_CANDIDATE_VALIDATOR_INTERNAL_ERROR" if internal_error else ""),
                "activeGeneration": attempt.active_generation,
                "nextAction": ("STOP_INTERNAL" if internal_error else attempt.next_action),
                "astFingerprint": attempt.ast_fingerprint,
                "contractFingerprint": attempt.contract_fingerprint,
                "outputColumns": attempt.output_columns,
                "gaps": attempt.validation_gaps,
                "submittedAndExecuted": False,
                "instruction": (
                    "For REPAIR_SQL, change the SQL AST using the exact gap. For "
                    "REVISE_BINDINGS, progressively read missing semantic assets and "
                    "propose a new Contract generation. Never retry the same SQL/error state."
                ),
            }
            if internal_error or attempt.status != "ACCEPTED":
                return json.dumps(
                    candidate_payload,
                    ensure_ascii=False,
                    default=str,
                )

            # The validator has accepted the complete Core-authored SQL and the
            # Kernel has atomically activated its preparation. Execute and
            # verify inside this same governed tool call so a second LLM turn
            # is not required merely to dispatch the accepted candidate.
            execution_payload = json.loads(
                execute_grounded_query.func(
                    reason="Core SQL candidate accepted; execute and verify atomically",
                    runtime=runtime,
                )
            )
            execution_payload.update(
                {
                    "sqlCandidateStatus": "ACCEPTED",
                    "candidateId": attempt.candidate_id,
                    "activeGeneration": attempt.active_generation,
                    "astFingerprint": attempt.ast_fingerprint,
                    "contractFingerprint": attempt.contract_fingerprint,
                    "submittedAndExecuted": True,
                }
            )
            return json.dumps(
                execution_payload,
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
            population_post: dict[str, Any] = {}

            def authorize_serial_artifact(
                artifact: GroundedVerifiedQueryArtifact,
            ) -> bool:
                nonlocal population_post
                with deep_session.lock:
                    deep_session.population_staged_query_artifacts[
                        artifact.artifact_id
                    ] = artifact.model_copy(deep=True)
                try:
                    population_post = (
                        runtime_owner._commit_population_node_post(
                            deep_session,
                            execution_session=session,
                        )
                    )
                except Exception:
                    with deep_session.lock:
                        deep_session.population_staged_query_artifacts.pop(
                            artifact.artifact_id,
                            None,
                        )
                    raise
                return True

            def verify_serial() -> VerifiedEvidence:
                if runtime_owner.population_gate_enforced:
                    return runtime_owner.kernel.verify_active(
                        session,
                        pre_ledger_authorizer=(
                            authorize_serial_artifact
                        ),
                    )
                return runtime_owner.kernel.verify_active(session)

            try:
                budget = runtime.context.budget
                if budget is not None:
                    budget.consume_doris_query(name="serial.grounded_query")
                if budget is None:
                    run_result = runtime_owner.kernel.execute_active(
                        session,
                        run_id=runtime.context.run_id,
                        **runtime_owner._population_execution_kwargs(
                            deep_session,
                            session,
                        ),
                    )
                else:
                    with budget.stage("doris.serial"):
                        run_result = runtime_owner.kernel.execute_active(
                            session,
                            run_id=runtime.context.run_id,
                            runtime_budget=budget,
                            **runtime_owner._population_execution_kwargs(
                                deep_session,
                                session,
                            ),
                        )
                execution_failure = (
                    _classify_grounded_execution_result(run_result)
                )
                if execution_failure.disposition == "NONE":
                    if budget is None:
                        verified = verify_serial()
                    else:
                        with budget.stage("evidence.serial"):
                            verified = verify_serial()
                else:
                    verified = VerifiedEvidence(passed=False)
            except GroundedRuntimeBudgetExceeded:
                raise
            except RuntimeError as exc:
                message = str(exc)
                if "POPULATION_POST_RESULT_REJECTED" in message:
                    with deep_session.lock:
                        deep_session.operational_failure = {
                            "code": "POPULATION_POST_RESULT_REJECTED",
                            "retryable": False,
                        }
                        deep_session.runtime.phase = (
                            "OPERATIONAL_FAILURE"
                        )
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "POPULATION_POST_RESULT_REJECTED",
                            "message": message[:500],
                            "nextAction": "STOP_INTERNAL",
                        },
                        ensure_ascii=False,
                    )
                no_progress = "SQL_EXECUTION_NO_PROGRESS" in message
                core_sql_required = "CORE_SQL_REQUIRED" in message or no_progress
                return json.dumps(
                    {
                        "status": "EXECUTION_REVISE_REQUIRED",
                        "code": (
                            "SQL_EXECUTION_NO_PROGRESS"
                            if no_progress
                            else "CORE_SQL_CANDIDATE_REQUIRED"
                            if core_sql_required
                            else "GROUNDED_EXECUTION_COMPATIBILITY_BLOCKED"
                        ),
                        "message": message[:500],
                        "nextAction": ("SUBMIT_GROUNDED_SQL_CANDIDATE" if core_sql_required else "REVISE_BINDINGS"),
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
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            if execution_failure.disposition != "NONE":
                core_sql_mode = (
                    str(
                        getattr(
                            session.active_execution_mode,
                            "value",
                            session.active_execution_mode,
                        )
                    )
                    == "CORE_SQL_REQUIRED"
                )
                if execution_failure.terminal:
                    with deep_session.lock:
                        deep_session.operational_failure = {
                            "code": execution_failure.code,
                            "failureDisposition": (
                                execution_failure.disposition
                            ),
                            "retryable": False,
                        }
                        deep_session.runtime.phase = (
                            "SECURITY_BLOCKED"
                            if execution_failure.disposition
                            == "SECURITY_TERMINAL"
                            else "OPERATIONAL_FAILURE"
                        )
                return json.dumps(
                    {
                        "status": (
                            "ACCESS_DENIED"
                            if execution_failure.disposition
                            == "SECURITY_TERMINAL"
                            else "OPERATIONAL_FAILURE"
                            if execution_failure.terminal
                            else "SQL_EXECUTION_REPAIR_REQUIRED"
                            if core_sql_mode
                            else "EXECUTION_FAILED"
                        ),
                        "code": execution_failure.code,
                        "failureDisposition": (
                            execution_failure.disposition
                        ),
                        "nextAction": (
                            "STOP"
                            if execution_failure.terminal
                            else "SUBMIT_GROUNDED_SQL_CANDIDATE"
                            if core_sql_mode
                            else "REVISE_BINDINGS"
                        ),
                        "message": execution_failure.message,
                        "blockingGaps": [gap.model_dump(by_alias=True) for gap in verified.blocking_gaps],
                        "instruction": (
                            "Access denial is terminal for this request; do not alter SQL to bypass policy."
                            if execution_failure.disposition
                            == "SECURITY_TERMINAL"
                            else "Stop: this failure is not a SQL/data recovery trigger."
                            if execution_failure.terminal
                            else (
                                "Use the execution error and active Contract to author one changed SQL AST. "
                                "Do not rerun the same accepted candidate."
                            )
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                )
            verification_failure = (
                _classify_grounded_execution_result(
                    run_result,
                    verified,
                )
            )
            if verification_failure.terminal:
                with deep_session.lock:
                    deep_session.operational_failure = {
                        "code": verification_failure.code,
                        "failureDisposition": (
                            verification_failure.disposition
                        ),
                        "retryable": False,
                    }
                    deep_session.runtime.phase = "OPERATIONAL_FAILURE"
                return json.dumps(
                    {
                        "status": "OPERATIONAL_FAILURE",
                        "code": verification_failure.code,
                        "failureDisposition": (
                            verification_failure.disposition
                        ),
                        "message": verification_failure.message,
                        "nextAction": "STOP",
                    },
                    ensure_ascii=False,
                )
            latest_artifact = getattr(
                runtime_owner.kernel,
                "latest_verified_query_artifact",
                None,
            )
            query_artifact = latest_artifact(session) if verified.passed and callable(latest_artifact) else None
            if runtime_owner.population_gate_enforced and verified.passed:
                if query_artifact is None:
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "POPULATION_PUBLISHED_RESULT_REQUIRED",
                            "nextAction": "STOP_INTERNAL",
                        },
                        ensure_ascii=False,
                    )
            if query_artifact is not None:
                with deep_session.lock:
                    deep_session.population_staged_query_artifacts.pop(
                        query_artifact.artifact_id,
                        None,
                    )
                    deep_session.artifact_goal_ids[query_artifact.artifact_id] = list(deep_session.active_goal_ids)
                    if runtime_owner.population_gate_enforced:
                        population_query_node_id = str(
                            population_post.get("queryNodeId") or ""
                        ).strip()
                        if population_query_node_id:
                            deep_session.population_artifact_query_node_ids[
                                query_artifact.artifact_id
                            ] = population_query_node_id
            return json.dumps(
                {
                    "status": "VERIFIED" if verified.passed else "VERIFICATION_GAPPED",
                    "reason": str(reason or "")[:500],
                    "queryArtifactId": (query_artifact.artifact_id if query_artifact else ""),
                    "populationPostGate": population_post,
                    "coveredGoalIds": list(deep_session.active_goal_ids if query_artifact else []),
                    "rowCount": len(run_result.merged_query_bundle.rows),
                    "tables": list(run_result.merged_query_bundle.tables),
                    "resultArtifacts": _grounded_result_artifact_receipts(run_result),
                    "outputColumns": (list(query_artifact.output_columns) if query_artifact else []),
                    "entitySetEligibleOutputs": (
                        sorted(query_artifact.output_entity_identities) if query_artifact else []
                    ),
                    "blockingGaps": [gap.model_dump(by_alias=True) for gap in verified.blocking_gaps],
                    "warningGaps": [gap.model_dump(by_alias=True) for gap in verified.warning_gaps],
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
            if not _artifact_population_authorized(
                deep_session,
                query_artifact_id,
            ):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": (
                            "POPULATION_POST_ATTESTATION_REQUIRED"
                        ),
                        "nextAction": "STOP",
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
                            if code
                            in {
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
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
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

        @tool("delegate_grounded_exploration")
        def delegate_grounded_exploration(
            analysis_goal_ids: list[str],
            source_query_artifact_ids: list[str],
            objective: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Run a zero-capability SubAgent that may emit advisory hypotheses only."""

            deep_session = runtime.context.session
            goal_contract = deep_session.question_goal_contract
            if deep_session.analysis_skill_started or deep_session.data_collection_sealed:
                return json.dumps(
                    {
                        "status": "POST_QUERY_SKILL_BOUNDARY_CLOSED",
                        "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                    },
                    ensure_ascii=False,
                )
            if goal_contract is None:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED",
                    },
                    ensure_ascii=False,
                )
            normalized_goal_ids = list(
                dict.fromkeys(str(goal_id or "").strip() for goal_id in analysis_goal_ids if str(goal_id or "").strip())
            )
            normalized_artifact_ids = list(
                dict.fromkeys(
                    str(artifact_id or "").strip()
                    for artifact_id in source_query_artifact_ids
                    if str(artifact_id or "").strip()
                )
            )
            normalized_objective = str(objective or "").strip()
            if not normalized_objective:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXPLORATION_OBJECTIVE_REQUIRED",
                    },
                    ensure_ascii=False,
                )
            goal_map = goal_contract.goal_map()
            invalid_goal_ids = [
                goal_id
                for goal_id in normalized_goal_ids
                if goal_id not in goal_map or str(getattr(goal_map[goal_id], "kind", "") or "").upper() != "ANALYSIS"
            ]
            if not normalized_goal_ids or invalid_goal_ids:
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "EXPLORATION_ANALYSIS_GOAL_INVALID",
                        "invalidGoalIds": invalid_goal_ids,
                    },
                    ensure_ascii=False,
                )
            if runtime_owner.checkpointer is None:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "EXPLORATION_CHECKPOINT_REQUIRED",
                    },
                    ensure_ascii=False,
                )
            try:
                source_views = runtime_owner._verified_exploration_source_views(
                    deep_session,
                    normalized_artifact_ids,
                )
                goal_fingerprint = original_question_goal_contract_fingerprint(goal_contract)
                assignment_fingerprint = _stable_json_fingerprint(
                    {
                        "goalContractFingerprint": goal_fingerprint,
                        "objective": normalized_objective,
                        "analysisGoalIds": sorted(normalized_goal_ids),
                        "sourceArtifactFingerprints": sorted(item.artifact_fingerprint for item in source_views),
                    }
                )
                assignment_id = "exploration_%s" % assignment_fingerprint[:24]
                max_assignments = max(
                    1,
                    min(
                        int(
                            getattr(
                                runtime_owner.settings,
                                "grounded_exploration_max_assignments",
                                2,
                            )
                            or 2
                        ),
                        16,
                    ),
                )
                if (
                    assignment_id not in deep_session.exploration_states
                    and len(deep_session.exploration_states) >= max_assignments
                ):
                    return json.dumps(
                        {
                            "status": "BLOCKED",
                            "code": "EXPLORATION_ASSIGNMENT_BUDGET_EXHAUSTED",
                            "maxAssignments": max_assignments,
                        },
                        ensure_ascii=False,
                    )
                cached_report = next(
                    (
                        item
                        for item in reversed(deep_session.exploration_reports)
                        if item.get("assignmentId") == assignment_id
                    ),
                    None,
                )
                if cached_report is not None:
                    return json.dumps(
                        {
                            **dict(cached_report),
                            "status": "IDEMPOTENT_REPLAY",
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                population_scope_fingerprint = _stable_json_fingerprint(
                    {
                        "sourceArtifactFingerprints": [item.artifact_fingerprint for item in source_views],
                        "goalPopulationScopes": [
                            {
                                "goalId": goal_id,
                                "populationScope": str(
                                    getattr(
                                        goal_map[goal_id],
                                        "population_scope",
                                        "",
                                    )
                                    or ""
                                ),
                                "populationGoalIds": list(
                                    getattr(
                                        goal_map[goal_id],
                                        "population_goal_ids",
                                        (),
                                    )
                                    or ()
                                ),
                            }
                            for goal_id in normalized_goal_ids
                        ],
                    }
                )
                artifact_by_id = {
                    artifact.artifact_id: artifact
                    for artifact in _authorized_verified_query_artifacts(
                        deep_session
                    )
                }
                time_scope_fingerprint = _stable_json_fingerprint(
                    [
                        {
                            "artifactId": artifact_id,
                            "timeRange": artifact_by_id[artifact_id].contract.time_range.model_dump(
                                by_alias=True,
                                mode="json",
                            ),
                            "timeField": artifact_by_id[artifact_id].contract.time_field.model_dump(
                                by_alias=True,
                                mode="json",
                            ),
                        }
                        for artifact_id in normalized_artifact_ids
                    ]
                )

                def progress_event(
                    stage: str,
                    status: str,
                    detail: str,
                ) -> None:
                    _emit_runtime_listener(
                        runtime.context.listener,
                        "exploration.progress",
                        "EXPLORATION_SUBAGENT",
                        {
                            "assignmentId": assignment_id,
                            "stage": stage,
                            "status": status,
                            "detail": str(detail or "")[:500],
                        },
                    )

                budget = runtime.context.budget
                configured_timeout = max(
                    1.0,
                    float(
                        getattr(
                            runtime_owner.settings,
                            "grounded_exploration_timeout_seconds",
                            15,
                        )
                        or 15
                    ),
                )
                exploration_timeout = (
                    budget.clamp_timeout_seconds(
                        configured_timeout,
                        minimum_seconds=0.001,
                        operation="exploration_subagent_timeout",
                    )
                    if budget is not None
                    else configured_timeout
                )
                worker = IsolatedGroundedExplorationWorker(
                    runtime_owner.subagent_runtime,
                    parent_thread_id=runtime.context.thread_id,
                    model_timeout_seconds=exploration_timeout,
                    on_progress=progress_event,
                )
                coordinator = GroundedExplorationCoordinator(
                    artifact_catalog=(InMemoryVerifiedExplorationArtifactCatalog(source_views)),
                    state_store=_SessionExplorationStateStore(deep_session),
                    worker=worker,
                )
                state = deep_session.exploration_states.get(assignment_id)
                if state is None:
                    receipt = coordinator.issue_assignment(
                        goal_contract,
                        GroundedExplorationAssignmentSpec(
                            assignment_id=assignment_id,
                            objective=normalized_objective,
                            authorized_goal_ids=tuple(normalized_goal_ids),
                            source_artifact_ids=tuple(normalized_artifact_ids),
                            scope_authority=GroundedExplorationScopeAuthority(
                                population_scope_fingerprint=(population_scope_fingerprint),
                                time_scope_fingerprint=time_scope_fingerprint,
                            ),
                        ),
                    )
                    expected_revision = receipt.ledger_revision
                else:
                    expected_revision = state.ledger.revision
                if budget is None:
                    report = coordinator.run_assignment(
                        assignment_id,
                        goal_contract,
                        expected_revision=expected_revision,
                    )
                else:
                    budget.consume_llm_call(name="exploration_subagent")
                    with budget.stage("llm.exploration_subagent"):
                        report = coordinator.run_assignment(
                            assignment_id,
                            goal_contract,
                            expected_revision=expected_revision,
                        )
            except GroundedExplorationCoordinatorError as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": exc.code,
                        "message": exc.message,
                        "issues": [item.model_dump(by_alias=True, mode="json") for item in exc.issues],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except GroundedRuntimeBudgetExceeded:
                raise
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "EXPLORATION_INTERNAL_ERROR",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                    },
                    ensure_ascii=False,
                )
            response = {
                "status": report.status,
                "assignmentId": report.assignment_id,
                "ledgerRevision": report.ledger_revision,
                "authority": report.authority,
                "publishableAsFinal": report.publishable_as_final,
                "queryExecuted": report.query_executed,
                "hypotheses": [item.model_dump(by_alias=True, mode="json") for item in report.artifact.hypotheses],
                "analysisPlan": (
                    report.artifact.analysis_plan.model_dump(
                        by_alias=True,
                        mode="json",
                    )
                    if report.artifact.analysis_plan is not None
                    else {}
                ),
                "stoppingAssessment": (
                    report.artifact.stopping_assessment.model_dump(
                        by_alias=True,
                        mode="json",
                    )
                ),
                "pendingCapabilityRequests": [
                    item.model_dump(by_alias=True, mode="json") for item in report.pending_capability_requests
                ],
                "nextAction": (
                    "ROOT_REVIEW_REQUESTS_AGAINST_FROZEN_GRAPH"
                    if deep_session.query_branch_contexts
                    else "ROOT_TRANSLATE_APPROVED_REQUESTS_TO_NORMAL_DISCOVERY"
                ),
            }
            with deep_session.lock:
                deep_session.exploration_reports.append(dict(response))
            return json.dumps(
                response,
                ensure_ascii=False,
                default=str,
            )

        @tool("compose_verified_answer", return_direct=True)
        def compose_verified_answer(
            allow_llm: bool,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Compose and attest the final answer from verified evidence only.

            Strict goal bindings are generated internally from immutable query
            artifacts and the actual rendered rows.  The Core cannot submit a
            renderer name or answer span as provenance. Primitive query goals
            retain a narrow compatibility binding from the verified renderer.
            """

            try:
                goal_coverage = runtime_owner._require_complete_goal_coverage(runtime.context.session)
            except GoalCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                runtime.context.session.goal_coverage_result = result
                return json.dumps(
                    {
                        "status": "GOAL_COVERAGE_INCOMPLETE",
                        "code": "ORIGINAL_QUESTION_GOALS_UNCOVERED",
                        "missingRequiredGoalIds": result.get("missingRequiredGoalIds", []),
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
            required_rule_goal_ids = _required_goal_ids_for_kind(
                runtime.context.session,
                "RULE",
            )
            verified_rule_artifacts: list[Any] = []
            verified_rule_artifact_ids: list[str] = []
            rule_answer_span = ""
            if required_rule_goal_ids:
                verified_rule_artifact_ids = list(
                    dict.fromkeys(
                        artifact_id
                        for goal_id in required_rule_goal_ids
                        for artifact_id in (
                            goal_coverage.resolution_artifact_ids_by_goal_id.get(
                                goal_id,
                                [],
                            )
                        )
                    )
                )
                rule_ledger_by_id = {
                    artifact.artifact_id: artifact
                    for artifact in (runtime.context.session.runtime.verified_rule_ledger)
                    if artifact.verification_passed
                }
                missing_rule_artifact_ids = [
                    artifact_id for artifact_id in verified_rule_artifact_ids if artifact_id not in rule_ledger_by_id
                ]
                if not verified_rule_artifact_ids or missing_rule_artifact_ids:
                    return json.dumps(
                        {
                            "status": "EVIDENCE_INCOMPLETE",
                            "code": "VERIFIED_RULE_ARTIFACT_REQUIRED",
                            "missingRuleArtifactIds": (missing_rule_artifact_ids),
                            "nextAction": "PUBLISH_VERIFIED_RULE_EVIDENCE",
                        },
                        ensure_ascii=False,
                    )
                verified_rule_artifacts = [rule_ledger_by_id[artifact_id] for artifact_id in verified_rule_artifact_ids]
                rendered_rule_answer = render_verified_rule_answer(verified_rule_artifacts)
                rule_answer_span = "### 规则依据\n\n%s" % rendered_rule_answer
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
                        "code": ("EVIDENCE_PORTFOLIO_INCOMPLETE" if incomplete else "ANSWER_COMPOSITION_BLOCKED"),
                        "message": message[:500],
                        "nextAction": ("CONTINUE_PROGRESSIVE_QUERYING" if incomplete else "STOP"),
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "ANSWER_COMPOSITION_INTERNAL_ERROR",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
                        "nextAction": "STOP_INTERNAL",
                    },
                    ensure_ascii=False,
                )
            state = runtime.context.session.runtime
            rendered = render_verified_query_goal_sections(
                runtime.context.session.question_goal_contract,
                goal_coverage,
                answer,
                _authorized_verified_query_artifacts(
                    runtime.context.session
                ),
            )
            answer = rendered.answer_markdown
            if rule_answer_span and rule_answer_span not in answer:
                answer = "\n\n".join(item for item in (answer, rule_answer_span) if item)
            state.answer = answer
            state.answer_rule_artifact_ids = list(verified_rule_artifact_ids)
            bindings = list(rendered.bindings)
            if rule_answer_span:
                bindings.extend(
                    render_verified_rule_goal_bindings(
                        runtime.context.session.question_goal_contract,
                        goal_coverage,
                        rule_answer_span,
                    )
                )
            try:
                answer_coverage = AnswerCoverageVerifier().require_complete(
                    runtime.context.session.question_goal_contract,
                    goal_coverage,
                    answer,
                    bindings,
                    source="compose_verified_answer",
                    auto_bind_verified_primitives=True,
                )
            except AnswerCoverageBlocked as exc:
                result = exc.result.model_dump(by_alias=True)
                runtime.context.session.answer_coverage_result = result
                runtime_owner._clear_rejected_answer(runtime.context.session)
                return json.dumps(
                    {
                        "status": "ANSWER_COVERAGE_INCOMPLETE",
                        "code": "FINAL_ANSWER_GOALS_NOT_RENDERED",
                        "missingGoalIds": result.get("missingGoalIds", []),
                        "issues": result.get("issues", []),
                        "nextAction": "RECOMPOSE_WITH_TYPED_GOAL_BINDINGS",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            runtime.context.session.answer_coverage_result = answer_coverage.model_dump(by_alias=True)
            return json.dumps(
                {
                    "status": "ANSWERED",
                    "answer": answer,
                    "verifiedQueryArtifactIds": list(state.answer_artifact_ids),
                    "verifiedRuleArtifactIds": list(state.answer_rule_artifact_ids),
                    "goalAnswerCoverage": dict(runtime.context.session.answer_coverage_result),
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
            analysis_gate = None
            try:
                coverage = runtime_owner._require_complete_goal_coverage(deep_session)
            except GoalCoverageBlocked as exc:
                coverage = exc.result
                analysis_gate = verify_grounded_analysis_data_input_coverage(
                    goal_contract=deep_session.question_goal_contract,
                    query_goal_coverage=coverage,
                )
                deep_session.analysis_data_input_gate_result = analysis_gate.model_dump(by_alias=True)
                if not analysis_gate.skill_start_allowed:
                    result = coverage.model_dump(by_alias=True)
                    deep_session.goal_coverage_result = result
                    return json.dumps(
                        {
                            "status": "EVIDENCE_INCOMPLETE",
                            "code": "ORIGINAL_QUESTION_GOALS_UNCOVERED",
                            "missingRequiredGoalIds": result.get("missingRequiredGoalIds", []),
                            "issues": result.get("issues", []),
                            "analysisDataInputGate": (analysis_gate.model_dump(by_alias=True)),
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
                plan, run_result, verified, artifact_ids = runtime_owner.kernel.verify_portfolio(deep_session.runtime)
            except Exception as exc:
                return json.dumps(
                    {
                        "status": "EVIDENCE_INCOMPLETE",
                        "code": "VERIFIED_EVIDENCE_PORTFOLIO_UNAVAILABLE",
                        "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
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
                        "blockingGaps": [item.model_dump(by_alias=True) for item in verified.blocking_gaps],
                        "nextAction": "CONTINUE_PROGRESSIVE_QUERYING",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            with deep_session.lock:
                deep_session.data_collection_sealed = True
                deep_session.analysis_skill_headers_disclosed = True
                deep_session.runtime.answer_plan = plan.model_copy(deep=True)
                deep_session.runtime.answer_run_result = run_result.model_copy(deep=True)
                deep_session.runtime.answer_verified_evidence = verified.model_copy(deep=True)
                deep_session.runtime.answer_artifact_ids = list(artifact_ids)
                if analysis_gate is not None:
                    deep_session.analysis_data_input_gate_result = analysis_gate.model_dump(by_alias=True)
            return json.dumps(
                {
                    "status": "EVIDENCE_COLLECTION_SEALED",
                    "reason": str(reason or "")[:500],
                    "verifiedQueryArtifactIds": artifact_ids,
                    "goalCoverage": coverage.model_dump(by_alias=True),
                    "analysisDataInputGate": (
                        analysis_gate.model_dump(by_alias=True) if analysis_gate is not None else {}
                    ),
                    "availableAnalysisGoalIds": (
                        list(analysis_gate.deferred_goal_ids) if analysis_gate is not None else []
                    ),
                    "rowCount": len(run_result.merged_query_bundle.rows),
                    "tables": list(run_result.merged_query_bundle.tables),
                    "availableAnalysisSkillHeaders": list(runtime_owner.skill_headers),
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
            analysis_publication_requests: Optional[list[dict[str, Any]]] = None,
        ) -> str:
            """Run one LLM-selected Skill in an isolated subagent/checkpoint."""

            result = runtime_owner._run_isolated_skill(
                runtime.context,
                skill_name=str(skill_name or "").strip(),
                objective=str(objective or "").strip(),
                analysis_publication_requests=(list(analysis_publication_requests or [])),
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
            propose_grounded_execution_graph,
            reopen_grounded_execution_graph_discovery,
            revise_grounded_execution_graph,
            retrieve_knowledge,
            publish_verified_rule_evidence,
            compose_verified_rule_answer,
            propose_grounded_contract,
            prepare_grounded_query_batch,
            submit_grounded_sql_candidate,
            execute_grounded_query,
            execute_grounded_query_batch,
            publish_verified_entity_set,
            delegate_grounded_exploration,
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
        analysis_publication_requests: Optional[list[dict[str, Any]]] = None,
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
            metadata.get("lifecyclePhase") or metadata.get("lifecycle_phase") or "post_query_analysis"
        ).strip()
        requires_verified = str(
            metadata.get("requiresVerifiedEvidence") or metadata.get("requires_verified_evidence") or "true"
        ).strip().lower() not in {"false", "0", "no"}
        output_contract = str(metadata.get("outputContract") or metadata.get("output_contract") or "").strip()
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
            _normalized_skill_name(normalized_name) or "run",
            uuid.uuid4().hex[:12],
        )
        skill_thread_id = "%s__%s" % (context.thread_id, skill_run_id)
        if session.context_workspace is None:
            return {
                "status": "SKILL_CONTEXT_WORKSPACE_REQUIRED",
                "skillName": normalized_name,
                "code": "SKILL_CONTEXT_WORKSPACE_REQUIRED",
            }
        try:
            workspace = session.context_workspace.subagent_workspace(
                "skill",
                skill_run_id,
            )
        except Exception:
            return {
                "status": "SKILL_WORKSPACE_FAILED",
                "skillName": normalized_name,
                "code": "SKILL_WORKSPACE_FAILED",
            }
        try:
            revalidate_semantic_activation = getattr(
                self.kernel,
                "revalidate_semantic_activation",
                None,
            )
            if callable(revalidate_semantic_activation):
                revalidate_semantic_activation(state)
            semantic_seal = state.semantic_activation_seal
            selected_artifact_ids = self._selected_skill_artifact_ids(session)
            artifact_access_bundle = build_grounded_skill_artifact_access(
                settings=self.settings,
                trusted_workspace_root=(session.context_workspace.root),
                artifact_root=session.context_workspace.artifacts_root,
                sandbox_staging_root=(session.context_workspace.staging_root),
                owner_fingerprint=(session.context_workspace.owner_fingerprint),
                verified_query_artifacts=(
                    _authorized_verified_query_artifacts(session)
                ),
                selected_artifact_ids=selected_artifact_ids,
                skill_run_id=skill_run_id,
                expected_semantic_activation_fingerprint=str(
                    semantic_seal.semantic_activation_fingerprint if semantic_seal is not None else ""
                ),
                expected_semantic_activation_seal_fingerprint=str(
                    semantic_seal.seal_fingerprint if semantic_seal is not None else ""
                ),
            )
        except GroundedSkillArtifactAccessError as exc:
            return {
                "status": "SKILL_ARTIFACT_ACCESS_REJECTED",
                "skillName": normalized_name,
                "code": exc.code,
            }
        except Exception:
            return {
                "status": "SKILL_ARTIFACT_ACCESS_REJECTED",
                "skillName": normalized_name,
                "code": "SKILL_ARTIFACT_ACCESS_BUILD_FAILED",
            }
        input_path = workspace / "input.json"
        script_output_path = workspace / "script-output.json"
        result_path = workspace / "result.json"

        def write_job_text(
            path: Path,
            content: str,
            *,
            immutable: bool = False,
        ) -> None:
            session.context_workspace.write_subagent_file(
                workspace,
                path.name,
                content,
                immutable=immutable,
            )

        def write_job_json(
            path: Path,
            payload: Any,
            *,
            immutable: bool = False,
        ) -> None:
            write_job_text(
                path,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                immutable=immutable,
            )

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
            artifact_access_bundle,
        )
        try:
            write_job_json(
                input_path,
                skill_payload,
                immutable=True,
            )
        except Exception as exc:
            return {
                "status": "SKILL_INPUT_ARTIFACT_FAILED",
                "skillName": normalized_name,
                "message": "%s:%s" % (type(exc).__name__, str(exc)[:400]),
            }
        progress_event(
            "workspace",
            "completed",
            "isolated run workspace ready",
        )
        with session.lock:
            if session.analysis_skill_started:
                return {
                    "status": "SKILL_ALREADY_ATTEMPTED",
                    "skillName": normalized_name,
                    "nextAction": "USE_SKILL_RESULT_OR_VERIFIED_FALLBACK",
                }
            session.analysis_skill_started = True
        execution_mode = str(
            metadata.get("executionMode") or metadata.get("execution_mode") or "structured_renderer"
        ).strip()
        script_result: dict[str, Any] = {}
        if execution_mode == "python_script":
            progress_event("script", "started", str(metadata.get("script") or ""))
            script_result = self._execute_declared_skill_script(
                skill_dir,
                metadata,
                input_path,
                script_output_path,
                context_workspace=session.context_workspace,
                artifact_access=artifact_access_bundle.access,
            )
            if not script_result.get("success"):
                progress_event("script", "failed", str(script_result.get("error") or ""))
                failed = {
                    "status": "SKILL_SCRIPT_FAILED",
                    "skillName": normalized_name,
                    "skillRunId": skill_run_id,
                    "checkpoint": checkpoint_ref,
                    "progress": progress,
                    "error": str(script_result.get("error") or "skill script failed"),
                }
                write_job_json(result_path, failed)
                self._record_skill_run(session, failed)
                return failed
            progress_event(
                "script",
                "completed",
                "isolated script output captured",
            )

        skill_semantic_backend = GroundedSemanticBackend(
            self.semantic_catalog,
            reader_is_core=lambda: False,
        )
        isolated_session = GroundedDeepAgentSession(
            runtime=state.model_copy(deep=True),
            context_workspace=session.context_workspace,
            opened_topics=list(session.opened_topics),
        )
        isolated_artifact_backend = GroundedRunFilesystemBackend(
            root_kind="artifacts",
            read_only=True,
            settings=self.settings,
            allowed_artifact_digests=(artifact_access_bundle.allowed_artifact_digests),
        )
        isolated_backend = CompositeBackend(
            default=FilesystemBackend(root_dir=workspace, virtual_mode=True),
            routes={
                "/knowledge/": skill_semantic_backend,
                "/artifacts/": isolated_artifact_backend,
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
                "The verifiedArtifactAccess catalog in /input.json is the only data "
                "authority. Read selected immutable rows through /artifacts with paging; "
                "unselected artifacts are outside your authority. PREVIEW and OBSERVATION "
                "inputs are samples only and must never be treated as a complete population. "
                "Never replace or extend a governed metric formula. Put measured facts in "
                "observations, governed definitions in semanticDisclosures, calculations "
                "using an already-declared governed formula in derivedFacts, uncertain ideas "
                "in hypotheses, actions in recommendations, and missing evidence in gaps. "
                "When /input.json contains analysisGoals, also return "
                "analysisPublicationRequests: one request per analysis goal using only "
                "that goal's publicationInterface schema. Select mappings and an allowed "
                "deterministic method; never return computed results, conclusions, causal "
                "claims, rows, or answerMarkdown inside a publication request. "
                "Return one JSON object with answerMarkdown, observations, "
                "semanticDisclosures, derivedFacts, hypotheses, recommendations, "
                "evidenceRefs, gaps, executionConfidence between 0 and 1, and when "
                "required analysisPublicationRequests."
            ),
            user_payload={
                "mountedSkill": "/skills/%s/SKILL.md" % normalized_name,
                "objective": objective,
                "inputArtifact": "/input.json",
                "scriptOutputArtifact": ("/script-output.json" if script_output_path.exists() else ""),
                "verifiedArtifactIds": list(artifact_access_bundle.selected_artifact_ids),
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
                    ],
                    "conditional": {"whenInputAnalysisGoalsPresent": ["analysisPublicationRequests"]},
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
                    paths=[
                        "/knowledge",
                        "/knowledge/**",
                        "/skills",
                        "/skills/**",
                        "/artifacts",
                        "/artifacts/**",
                        "/input.json",
                        "/script-output.json",
                        "/draft-output.json",
                        "/verification-feedback.json",
                    ],
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
                "progress": progress,
                "error": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
            }
            write_job_json(result_path, failed)
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
            if allow_script_fallback and not structured_output.get("answerMarkdown") and script_result.get("payload"):
                structured_output["answerMarkdown"] = str(
                    (script_result.get("payload") or {}).get("answerMarkdown") or ""
                )
            contract_issues = _skill_output_contract_issues(
                structured_output,
                state.answer_plan or state.active_plan,
            )
            expected_analysis_goal_ids = {
                str(item.get("analysisGoalId") or "")
                for item in skill_payload.get("analysisGoals") or []
                if isinstance(item, dict) and str(item.get("analysisGoalId") or "")
            }
            publication_requests = list(analysis_publication_requests or [])
            if not publication_requests:
                raw_requests = structured_output.get("analysisPublicationRequests") or []
                if isinstance(raw_requests, dict):
                    raw_requests = [raw_requests]
                if isinstance(raw_requests, list):
                    publication_requests = [dict(item) for item in raw_requests if isinstance(item, dict)]
            if not publication_requests and allow_script_fallback and isinstance(script_result.get("payload"), dict):
                raw_requests = (script_result.get("payload") or {}).get("analysisPublicationRequests") or []
                if isinstance(raw_requests, dict):
                    raw_requests = [raw_requests]
                if isinstance(raw_requests, list):
                    publication_requests = [dict(item) for item in raw_requests if isinstance(item, dict)]
            submitted_goal_ids = {
                str(item.get("analysisGoalId") or item.get("analysis_goal_id") or "") for item in publication_requests
            }
            if expected_analysis_goal_ids != submitted_goal_ids:
                contract_issues.append(
                    {
                        "code": "ANALYSIS_PUBLICATION_REQUEST_REQUIRED",
                        "message": (
                            "Return exactly one narrow analysis publication "
                            "request for every typed deferred analysis goal."
                        ),
                        "expectedAnalysisGoalIds": sorted(expected_analysis_goal_ids),
                        "submittedAnalysisGoalIds": sorted(item for item in submitted_goal_ids if item),
                    }
                )
            structured_output["_groundedAnalysisPublicationRequests"] = publication_requests
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
            structured_output["executionConfidence"] = _confidence(structured_output.get("executionConfidence"))
            rendered_answer = str(structured_output.get("answerMarkdown") or "").strip()
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
                for artifact in _authorized_verified_query_artifacts(
                    session
                )
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

        structured, answer, untrusted_refs, claim_verification, contract_issues = assess_output(
            raw_output, allow_script_fallback=True
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
            write_job_text(
                draft_path,
                raw_output,
                immutable=True,
            )
            feedback = {
                "contractIssues": contract_issues,
                "untrustedEvidenceRefs": untrusted_refs,
                "unsupportedClaims": [item.model_dump(by_alias=True) for item in claim_verification.unsupported_claims],
                "repairPolicy": (
                    "Revise presentation only from the same immutable /input.json. "
                    "Do not request more data, add metrics, or alter governed formulas."
                ),
            }
            write_job_json(
                feedback_path,
                feedback,
                immutable=True,
            )
            repair_job = replace(
                isolated_job,
                job_id="%s_repair1" % skill_run_id,
                thread_id="%s__repair1" % skill_thread_id,
                system_prompt=(
                    isolated_job.system_prompt + " This is the only permitted repair attempt. Read /draft-output.json "
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
                structured, answer, untrusted_refs, claim_verification, contract_issues = assess_output(
                    repaired_result.raw_output,
                    allow_script_fallback=False,
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
                "answerMarkdown": fallback_answer,
                "repairAttempted": repair_attempted,
                "queryMutationAllowed": False,
                "contractIssues": contract_issues,
                "untrustedEvidenceRefs": untrusted_refs,
                "unsupportedClaims": [item.model_dump(by_alias=True) for item in claim_verification.unsupported_claims],
                "progress": progress,
            }
            write_job_json(result_path, fallback)
            self._record_skill_run(session, fallback)
            return fallback

        publication_requests = list(structured.get("_groundedAnalysisPublicationRequests") or [])
        try:
            if session.question_goal_contract is None:
                # Compatibility for direct unit/legacy harness construction.
                # The online Core cannot reach run_skill without first
                # declaring the immutable original-question Goal Contract.
                analysis_artifacts = []
                final_goal_coverage = None
                final_answer_coverage = None
                with session.lock:
                    state.answer = answer
                    state.phase = "ANSWERED"
            else:
                (
                    answer,
                    analysis_artifacts,
                    final_goal_coverage,
                    final_answer_coverage,
                ) = self._finalize_attested_skill_answer(
                    session,
                    publication_requests,
                )
        except Exception as exc:
            progress_event(
                "analysis_publication",
                "failed",
                "%s:%s" % (type(exc).__name__, str(exc)[:400]),
            )
            failed = {
                "status": "SKILL_ANALYSIS_PUBLICATION_FAILED",
                "skillName": normalized_name,
                "skillRunId": skill_run_id,
                "checkpoint": checkpoint_ref,
                "queryMutationAllowed": False,
                "error": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                "progress": progress,
            }
            write_job_json(result_path, failed)
            self._record_skill_run(session, failed)
            return failed

        progress_event("verification", "completed", "verified evidence only")
        completed = {
            "status": "SKILL_COMPLETED",
            "skillName": normalized_name,
            "skillRunId": skill_run_id,
            "checkpoint": checkpoint_ref,
            "answerMarkdown": answer,
            "verifiedAnalysisArtifactIds": [item.artifact_id for item in analysis_artifacts],
            "goalCoverage": (final_goal_coverage.model_dump(by_alias=True) if final_goal_coverage is not None else {}),
            "answerCoverage": (
                final_answer_coverage.model_dump(by_alias=True) if final_answer_coverage is not None else {}
            ),
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
        progress_event(
            "result",
            "completed",
            "verified Skill result recorded",
        )
        completed["progress"] = progress
        write_job_json(result_path, completed)
        self._record_skill_run(session, completed)
        return completed

    def _finalize_attested_skill_answer(
        self,
        session: GroundedDeepAgentSession,
        publication_requests: list[dict[str, Any]],
    ) -> tuple[str, list[GroundedDerivedAnalysisArtifact], Any, Any]:
        """Publish deterministic analysis artifacts and attest the final text."""

        published: list[GroundedDerivedAnalysisArtifact] = []
        for raw_request in publication_requests:
            request = GroundedRunSkillAnalysisPublicationRequest.model_validate(raw_request)
            artifact = publish_grounded_analysis_from_skill(
                goal_contract=session.question_goal_contract,
                publication_request=request,
                verified_query_artifacts=(
                    _authorized_verified_query_artifacts(session)
                ),
                artifact_goal_ids=session.artifact_goal_ids,
            )
            with session.lock:
                existing = next(
                    (item for item in session.verified_analysis_ledger if item.artifact_id == artifact.artifact_id),
                    None,
                )
                if existing is None:
                    session.verified_analysis_ledger.append(artifact.model_copy(deep=True))
                else:
                    artifact = existing.model_copy(deep=True)
            published.append(artifact)

        coverage = self._require_complete_goal_coverage(session)
        state = session.runtime
        base_answer = self.kernel.compose_answer(state, allow_llm=False)
        query_render = render_verified_query_goal_sections(
            session.question_goal_contract,
            coverage,
            base_answer,
            _authorized_verified_query_artifacts(session),
        )
        answer = query_render.answer_markdown
        bindings = list(query_render.bindings)
        for artifact in session.verified_analysis_ledger:
            rendered = render_grounded_analysis_artifact(artifact)
            answer = "\n\n".join(item for item in (answer, rendered.answer_markdown) if str(item or "").strip())
            bindings.append(rendered.binding)
        answer_coverage = AnswerCoverageVerifier().require_complete(
            session.question_goal_contract,
            coverage,
            answer,
            bindings,
            source="run_skill",
            auto_bind_verified_primitives=True,
        )
        with session.lock:
            state.answer = answer
            state.phase = "ANSWERED"
            session.goal_coverage_result = coverage.model_dump(by_alias=True)
            session.answer_coverage_result = answer_coverage.model_dump(by_alias=True)
        return answer, published, coverage, answer_coverage

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
    def _selected_skill_artifact_ids(
        session: GroundedDeepAgentSession,
    ) -> tuple[str, ...]:
        answer_ids = tuple(
            dict.fromkeys(str(item).strip() for item in session.runtime.answer_artifact_ids if str(item).strip())
        )
        if not answer_ids:
            raise GroundedSkillArtifactAccessError("SKILL_ANSWER_ARTIFACT_SELECTION_REQUIRED")
        raw_gate = session.analysis_data_input_gate_result
        if not raw_gate:
            return answer_ids
        gate_ids_value = raw_gate.get("verifiedInputArtifactIds")
        if gate_ids_value is None:
            gate_ids_value = raw_gate.get("verified_input_artifact_ids")
        if not isinstance(gate_ids_value, list) or isinstance(gate_ids_value, (str, bytes)):
            raise GroundedSkillArtifactAccessError("SKILL_ANALYSIS_GATE_ARTIFACT_SELECTION_REQUIRED")
        gate_ids = tuple(dict.fromkeys(str(item).strip() for item in gate_ids_value if str(item).strip()))
        if not gate_ids:
            raise GroundedSkillArtifactAccessError("SKILL_ANALYSIS_GATE_ARTIFACT_SELECTION_REQUIRED")
        answer_scope = set(answer_ids)
        if any(item not in answer_scope for item in gate_ids):
            raise GroundedSkillArtifactAccessError("SKILL_ANALYSIS_GATE_ARTIFACT_SCOPE_MISMATCH")
        selected = tuple(item for item in answer_ids if item in set(gate_ids))
        if not selected:
            raise GroundedSkillArtifactAccessError("SKILL_ANALYSIS_GATE_ARTIFACT_SCOPE_MISMATCH")
        return selected

    @staticmethod
    def _skill_input_payload(
        session: GroundedDeepAgentSession,
        skill_name: str,
        objective: str,
        skill_run_id: str,
        artifact_access: GroundedSkillArtifactAccessBundle,
    ) -> dict[str, Any]:
        state = session.runtime
        plan = state.answer_plan or state.active_plan
        run_result = state.answer_run_result or state.run_result
        active_contract = state.active_contract
        verified = state.answer_verified_evidence or state.verified_evidence
        selected_artifact_ids = set(artifact_access.selected_artifact_ids)
        selected_artifacts = [
            item
            for item in _authorized_verified_query_artifacts(session)
            if item.artifact_id in selected_artifact_ids
        ]
        allowed_evidence_refs = list(
            dict.fromkeys(ref_id for artifact in selected_artifacts for ref_id in artifact.contract.evidence_refs)
        )
        analysis_inputs: list[dict[str, Any]] = []
        if session.question_goal_contract is not None:
            coverage = GroundedDeepAgentRuntime._goal_coverage_snapshot(session)
            gate = verify_grounded_analysis_data_input_coverage(
                goal_contract=session.question_goal_contract,
                query_goal_coverage=coverage,
            )
            if gate.skill_start_allowed:
                for goal_id in gate.deferred_goal_ids:
                    input_goal_ids = gate.deferred_input_goal_ids_by_goal_id.get(goal_id, [])
                    requested_ids = list(
                        dict.fromkeys(
                            artifact_id
                            for input_goal_id in input_goal_ids
                            for artifact_id in (coverage.coverage_by_goal_id.get(input_goal_id, []))
                        )
                    )
                    analysis_input = build_grounded_analysis_skill_input(
                        goal_contract=session.question_goal_contract,
                        analysis_goal_id=goal_id,
                        requested_artifact_ids=requested_ids,
                        verified_query_artifacts=(
                            _authorized_verified_query_artifacts(
                                session
                            )
                        ),
                        artifact_goal_ids=session.artifact_goal_ids,
                        include_rows=False,
                    )
                    analysis_payload = analysis_input.model_dump(by_alias=True)
                    for verified_input in analysis_payload.get("verifiedInputs") or []:
                        if not isinstance(verified_input, dict):
                            continue
                        verified_input.pop("rows", None)
                        artifact_id = str(verified_input.get("artifactId") or "")
                        matching_catalog = [
                            dict(item)
                            for item in artifact_access.artifact_catalog
                            if str(item.get("queryArtifactId") or "") == artifact_id
                        ]
                        if matching_catalog:
                            verified_input["rowRef"] = str(matching_catalog[0].get("rowsRef") or "")
                        verified_input["artifactAccess"] = matching_catalog
                    analysis_inputs.append(analysis_payload)
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
                "tables": list(run_result.merged_query_bundle.tables if run_result is not None else []),
                "timeRange": (
                    active_contract.time_range.model_dump(by_alias=True) if active_contract is not None else {}
                ),
                "evidenceRefs": allowed_evidence_refs,
                "verifiedQueryArtifactIds": list(artifact_access.selected_artifact_ids),
            },
            "verifiedArtifactAccess": {
                "schemaVersion": 1,
                "transport": "READ_ONLY_PUBLISHED_ARTIFACTS",
                "inlineRows": False,
                "selectedQueryArtifactIds": list(artifact_access.selected_artifact_ids),
                "artifacts": [dict(item) for item in artifact_access.artifact_catalog],
                "populationPolicy": (
                    "Only COMPLETE_POPULATION may represent every row in its "
                    "declared query population; PREVIEW/OBSERVATION never does."
                ),
            },
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
                    gap.model_dump(by_alias=True) for gap in (verified.blocking_gaps if verified is not None else [])
                ],
                "warningGaps": [
                    gap.model_dump(by_alias=True) for gap in (verified.warning_gaps if verified is not None else [])
                ],
            },
            "evidenceGaps": [
                gap.model_dump(by_alias=True)
                for gap in (run_result.evidence_gaps[:32] if run_result is not None else [])
            ],
            "allowedEvidenceRefs": list(allowed_evidence_refs),
            "analysisGoals": analysis_inputs,
        }

    def _execute_declared_skill_script(
        self,
        skill_dir: Path,
        metadata: dict[str, str],
        input_path: Path,
        output_path: Path,
        context_workspace: GroundedContextWorkspace,
        artifact_access: SandboxArtifactAccess | None = None,
    ) -> dict[str, Any]:
        relative = Path(str(metadata.get("script") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return {"success": False, "error": "invalid Skill script path"}
        script = (skill_dir / relative).resolve()
        if skill_dir not in script.parents or not script.is_file() or script.suffix != ".py":
            return {"success": False, "error": "declared Skill script is unavailable"}
        if self.analysis_sandbox is None:
            return {
                "success": False,
                "error": "SKILL_SANDBOX_NOT_CONFIGURED",
            }
        if artifact_access is None or not artifact_access.verified_query_artifact_commits:
            return {
                "success": False,
                "error": "SKILL_VERIFIED_ARTIFACT_ACCESS_REQUIRED",
            }
        completed = self.analysis_sandbox.run_python(
            script,
            [
                "--input",
                input_path.name,
                "--output",
                output_path.name,
            ],
            output_path.parent,
            90,
            artifact_access=artifact_access,
        )
        if completed.returncode != 0:
            return {
                "success": False,
                "error": (completed.stderr or completed.stdout or "skill script failed")[:1000],
            }
        try:
            context_workspace.read_subagent_file(
                input_path.parent,
                input_path.name,
                require_immutable=True,
            )
            output_text = context_workspace.read_subagent_file(
                output_path.parent,
                output_path.name,
            )
            context_workspace.write_subagent_file(
                output_path.parent,
                output_path.name,
                output_text,
                immutable=True,
            )
            output_text = context_workspace.read_subagent_file(
                output_path.parent,
                output_path.name,
                require_immutable=True,
            )
            payload = json.loads(output_text)
        except Exception:
            return {
                "success": False,
                "error": "SKILL_SCRIPT_OUTPUT_ARTIFACT_INVALID",
            }
        serialized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        forbidden_paths = {
            str(input_path.parent.resolve()),
            str(skill_dir.resolve()),
            str(Path(artifact_access.run_artifact_root).resolve()),
            str(Path(artifact_access.trusted_workspace_root).resolve()),
        }
        if any(path and path in serialized_payload for path in forbidden_paths):
            return {
                "success": False,
                "error": "SKILL_OUTPUT_HOST_PATH_REJECTED",
            }
        return {"success": True, "payload": payload}

    @staticmethod
    def _record_skill_run(
        session: GroundedDeepAgentSession,
        result: dict[str, Any],
    ) -> None:
        with session.lock:
            session.skill_runs.append(dict(result))
            run_result = session.runtime.answer_run_result or session.runtime.run_result
            if run_result is not None:
                run_result.skill_lifecycle_records.append(
                    SkillLifecycleRecord(
                        record_id=str(result.get("skillRunId") or ""),
                        skill_name=str(result.get("skillName") or ""),
                        stage="completed",
                        status=str(result.get("status") or ""),
                        matched_by="core_llm_skill_header",
                        isolated_run_id=str(result.get("skillRunId") or ""),
                        progress=[
                            "%s:%s" % (item.get("stage") or "", item.get("status") or "")
                            for item in result.get("progress") or []
                            if isinstance(item, dict)
                        ],
                        summary=str(result.get("answerMarkdown") or result.get("error") or "")[:1000],
                        metadata={
                            "checkpoint": result.get("checkpoint") or {},
                            "executionConfidence": result.get("executionConfidence"),
                        },
                    )
                )

    @staticmethod
    def _clear_rejected_answer(session: GroundedDeepAgentSession) -> None:
        """Remove an answer snapshot that failed final answer-goal coverage."""

        state = session.runtime
        with session.lock:
            state.answer = ""
            state.answer_plan = None
            state.answer_run_result = None
            state.answer_verified_evidence = None
            state.answer_artifact_ids = []
            state.answer_rule_artifact_ids = []
            if state.phase == "ANSWERED":
                state.phase = "ANSWER_COVERAGE_INCOMPLETE"

    @staticmethod
    def _answer_is_attested(session: GroundedDeepAgentSession) -> bool:
        return answer_attestation_matches(
            session.runtime.answer,
            session.answer_coverage_result,
        )

    @staticmethod
    def _goal_coverage_declarations(
        session: GroundedDeepAgentSession,
    ) -> list[VerifiedArtifactGoalCoverage]:
        contract = session.question_goal_contract
        if contract is None:
            raise RuntimeError("ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED")
        declarations: list[VerifiedArtifactGoalCoverage] = []
        query_artifacts = _authorized_verified_query_artifacts(session)
        for artifact in query_artifacts:
            assigned_goal_ids = session.artifact_goal_ids.get(artifact.artifact_id) or []
            declarations.append(
                declare_verified_artifact_goal_coverage(
                    contract,
                    artifact,
                    assigned_goal_ids,
                    evidence_refs=artifact.contract.evidence_refs,
                    goal_resolutions=derive_query_artifact_goal_resolutions(
                        goal_contract=contract,
                        artifact=artifact,
                        assigned_goal_ids=assigned_goal_ids,
                        artifact_goal_ids=session.artifact_goal_ids,
                        all_artifacts=query_artifacts,
                    ),
                )
            )
        for artifact in session.runtime.verified_rule_ledger:
            evidence_refs = [item.ref_id for item in artifact.evidence_refs]
            declarations.append(
                VerifiedArtifactGoalCoverage(
                    artifact_id=artifact.artifact_id,
                    goal_contract_fingerprint=(artifact.goal_contract_fingerprint),
                    covered_goal_ids=list(artifact.goal_ids),
                    verification_passed=artifact.verification_passed,
                    evidence_refs=evidence_refs,
                    goal_resolutions=[
                        {
                            "goalId": goal_id,
                            "goalKind": "RULE",
                            "resolution": "PROVED",
                            "proofType": "VERIFIED_RULE_ARTIFACT",
                            "evidenceRefs": evidence_refs,
                            "ruleRefIds": evidence_refs,
                            "citationRefs": evidence_refs,
                        }
                        for goal_id in artifact.goal_ids
                    ],
                )
            )
        for artifact in session.verified_analysis_ledger:
            declarations.append(grounded_analysis_goal_coverage(contract, artifact))
        return declarations

    def _population_graph_for_execution(
        self,
        session: GroundedDeepAgentSession,
        execution_session: GroundedRuntimeSession,
        *,
        query_node_id: str = "",
    ) -> PopulationDynamicGraphReceipt:
        with session.lock:
            existing = session.population_graph_receipt
            execution_receipt = (
                session.execution_graph_receipt.model_copy(deep=True)
                if session.execution_graph_receipt is not None
                else None
            )
            contexts = dict(session.query_branch_contexts)
            graph_edges = tuple(item.model_copy(deep=True) for item in session.execution_graph_edges)
            goal_contract = session.question_goal_contract
            active_goal_ids = tuple(session.active_goal_ids)
            graph_fingerprint = session.execution_graph_fingerprint
            graph_generation = session.execution_graph_generation
        if existing is not None:
            return existing.model_copy(deep=True)
        if goal_contract is None:
            raise RuntimeError("ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED")

        nodes: tuple[PopulationDynamicGraphNode, ...]
        edges: tuple[PopulationDynamicGraphEdge, ...]
        graph_id: str
        version: int
        if execution_receipt is not None:
            nodes = tuple(
                PopulationDynamicGraphNode(
                    query_node_id=query_id,
                    consumer_goal_ids=tuple(contexts[query_id].spec.goal_ids),
                )
                for query_id in execution_receipt.node_ids.values()
            )
            edges = tuple(
                PopulationDynamicGraphEdge(
                    source_query_node_id=(execution_receipt.node_ids[item.source_client_key]),
                    target_query_node_id=(execution_receipt.node_ids[item.target_client_key]),
                    dependency_mode=item.dependency_mode,
                    artifact_kind=item.artifact_kind,
                )
                for item in graph_edges
            )
            graph_id = execution_receipt.graph_id
            version = execution_receipt.version
            graph_fingerprint = execution_receipt.fingerprint
        elif contexts:
            nodes = tuple(
                PopulationDynamicGraphNode(
                    query_node_id=query_id,
                    consumer_goal_ids=tuple(context.spec.goal_ids),
                )
                for query_id, context in sorted(contexts.items())
            )
            derived_edges: list[PopulationDynamicGraphEdge] = []
            for target_id, context in sorted(contexts.items()):
                derived_edges.extend(
                    PopulationDynamicGraphEdge(
                        source_query_node_id=source_id,
                        target_query_node_id=target_id,
                        dependency_mode="CONTRACT_SCOPE",
                    )
                    for source_id in sorted(set(context.contract_scope_query_ids))
                )
                derived_edges.extend(
                    PopulationDynamicGraphEdge(
                        source_query_node_id=source_id,
                        target_query_node_id=target_id,
                        dependency_mode="VERIFIED_ARTIFACT",
                        artifact_kind="VERIFIED_RESULT_ARTIFACT",
                    )
                    for source_id in sorted(set(context.dependency_query_ids))
                )
            edges = tuple(derived_edges)
            graph_fingerprint = graph_fingerprint or _stable_json_fingerprint(
                {
                    "goalContractFingerprint": (original_question_goal_contract_fingerprint(goal_contract)),
                    "nodes": [item.model_dump(by_alias=True, mode="json") for item in nodes],
                    "edges": [item.model_dump(by_alias=True, mode="json") for item in edges],
                }
            )
            graph_id = "population_graph_%s" % graph_fingerprint[:24]
            version = max(1, int(graph_generation or 0))
        else:
            preparation = execution_session.active_preparation
            plan = getattr(preparation, "plan", None)
            intents = tuple(getattr(plan, "intents", ()) or ())
            node_id = str(query_node_id or "").strip()
            if not node_id and len(intents) == 1:
                node_id = str(intents[0].plan_task_id or "").strip()
            if not node_id or not active_goal_ids:
                raise RuntimeError("POPULATION_DYNAMIC_NODE_IDENTITY_REQUIRED")
            nodes = (
                PopulationDynamicGraphNode(
                    query_node_id=node_id,
                    consumer_goal_ids=active_goal_ids,
                ),
            )
            edges = ()
            graph_fingerprint = _stable_json_fingerprint(
                {
                    "goalContractFingerprint": (original_question_goal_contract_fingerprint(goal_contract)),
                    "nodes": [item.model_dump(by_alias=True, mode="json") for item in nodes],
                }
            )
            graph_id = "population_graph_%s" % graph_fingerprint[:24]
            version = 1
        candidate = seal_population_dynamic_graph_receipt(
            PopulationDynamicGraphReceipt(
                graph_id=graph_id,
                graph_version=version,
                graph_fingerprint=graph_fingerprint,
                nodes=nodes,
                edges=edges,
            )
        )
        with session.lock:
            if session.population_graph_receipt is None:
                session.population_graph_receipt = candidate.model_copy(deep=True)
            elif session.population_graph_receipt.receipt_fingerprint != candidate.receipt_fingerprint:
                raise RuntimeError("POPULATION_DYNAMIC_GRAPH_CHANGED")
            return session.population_graph_receipt.model_copy(deep=True)

    def _population_execution_kwargs(
        self,
        session: GroundedDeepAgentSession,
        execution_session: GroundedRuntimeSession,
        *,
        query_node_id: str = "",
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(_grounded_artifact_execution_kwargs(session))
        with session.lock:
            session.population_gate_enforced = bool(
                self.population_gate_enforced
            )
        if not self.population_gate_enforced:
            return kwargs
        gate = self.population_execution_gate
        workspace = session.context_workspace
        contract = execution_session.active_contract
        if gate is None or workspace is None or contract is None:
            raise RuntimeError("POPULATION_ONLINE_GATE_AUTHORITY_REQUIRED")
        graph_receipt = self._population_graph_for_execution(
            session,
            execution_session,
            query_node_id=query_node_id,
        )
        node_id = str(query_node_id or "").strip()
        if not node_id:
            preparation = execution_session.active_preparation
            plan = getattr(preparation, "plan", None)
            intents = tuple(getattr(plan, "intents", ()) or ())
            if len(intents) == 1:
                node_id = str(intents[0].plan_task_id or "").strip()
        matching_nodes = tuple(item for item in graph_receipt.nodes if item.query_node_id == node_id)
        if len(matching_nodes) != 1:
            raise RuntimeError("POPULATION_DYNAMIC_NODE_BINDING_REQUIRED")
        sql_validation = execution_session.active_sql_validation
        expected_ast = str(getattr(sql_validation, "ast_fingerprint", "") or "").strip()
        reference = gate.build_pre_execution_reference(
            context_owner_fingerprint=workspace.owner_fingerprint,
            run_authority_fingerprint=workspace.request_fingerprint,
            goal_contract_fingerprint=(original_question_goal_contract_fingerprint(session.question_goal_contract)),
            graph_receipt=graph_receipt,
            node=PopulationPreExecutionNodeReference(
                query_node_id=node_id,
                consumer_goal_ids=matching_nodes[0].consumer_goal_ids,
                generation=execution_session.active_generation,
                attempt_id=execution_session.active_attempt_id,
                query_contract_fingerprint=(grounded_query_contract_fingerprint(contract)),
                expected_sql_ast_fingerprint=expected_ast,
            ),
        )
        with session.lock:
            session.population_pre_execution_references[node_id] = reference.model_copy(deep=True)
        kwargs["population_pre_execution_reference"] = reference
        kwargs["population_query_node_id"] = node_id
        return kwargs

    def _graph_revision_fault_checkpoint(
        self,
        stage: str,
        transaction_id: str,
    ) -> None:
        injector = self.graph_revision_fault_injector
        if injector is not None:
            injector(
                str(stage or ""),
                str(transaction_id or ""),
            )

    def _revision_contexts_from_recovery_payload(
        self,
        session: GroundedDeepAgentSession,
        payload: GroundedGraphRevisionRecoveryPayload,
        *,
        runtime_budget: GroundedRuntimeBudget | None,
    ) -> dict[str, GroundedQueryBranchContext]:
        proposal = GroundedExecutionGraphProposal.model_validate(
            payload.execution_proposal
        )
        receipt = GroundedExecutionGraphReceipt.model_validate(
            payload.execution_receipt
        )
        declarations = {
            item.client_key: item
            for item in payload.candidate_node_declarations
        }
        with session.lock:
            old_contexts = dict(session.query_branch_contexts)
            evidence_by_ref = {
                str(item.get("refId") or ""): item
                for item in session.core_semantic_evidence
                if str(item.get("refId") or "")
            }
        carried_query_ids = set(receipt.carried_forward_node_ids)
        limits = GroundedBranchBudgetLimits.from_settings(
            self.settings or object()
        )
        node_by_client_key = {
            node.client_key: node for node in proposal.nodes
        }
        dependency_goal_ids_by_target_key: dict[
            str,
            list[str],
        ] = {}
        for edge in proposal.edges:
            if edge.dependency_mode != "VERIFIED_ARTIFACT":
                continue
            target_goal_ids = (
                dependency_goal_ids_by_target_key.setdefault(
                    edge.target_client_key,
                    [],
                )
            )
            for goal_id in node_by_client_key[
                edge.source_client_key
            ].goal_ids:
                if goal_id not in target_goal_ids:
                    target_goal_ids.append(goal_id)
        contexts: dict[str, GroundedQueryBranchContext] = {}
        for node in proposal.nodes:
            declaration = declarations[node.client_key]
            query_node_id = receipt.node_ids[node.client_key]
            if query_node_id in carried_query_ids:
                carried = old_contexts.get(query_node_id)
                if carried is None:
                    raise RuntimeError(
                        "GRAPH_REVISION_RECOVERY_BASE_SESSION_REQUIRED"
                    )
                contexts[query_node_id] = carried
                continue
            spec = GroundedQueryBranchSpec(
                query_id=query_node_id,
                objective=declaration.objective,
                goal_ids=list(declaration.goal_ids),
                topic_scope=list(declaration.topic_scope),
                evidence_ref_ids=list(
                    declaration.evidence_ref_ids
                ),
            )
            branch_runtime: GroundedRuntimeSession | None = None
            if declaration.initial_status == "DECLARED":
                try:
                    branch_runtime = self.kernel.fork_query_branch(
                        session.runtime,
                        query_node_id,
                        workspace_topics=spec.topic_scope,
                        objective=spec.objective,
                    )
                except TypeError:
                    branch_runtime = self.kernel.fork_query_branch(
                        session.runtime,
                        query_node_id,
                    )
                    branch_runtime.workspace_topics = list(
                        spec.topic_scope
                    )
                    branch_runtime.question = spec.objective
            context = GroundedQueryBranchContext(
                spec=spec,
                runtime=branch_runtime,
                budget=GroundedBranchBudget(
                    query_node_id,
                    limits,
                    parent=runtime_budget,
                ),
                status=declaration.initial_status,
                dependency_query_ids=list(
                    declaration.dependency_query_node_ids
                ),
                dependency_goal_ids=list(
                    dependency_goal_ids_by_target_key.get(
                        node.client_key,
                        [],
                    )
                ),
                contract_scope_query_ids=list(
                    declaration.contract_scope_query_node_ids
                ),
            )
            for ref_id in spec.evidence_ref_ids:
                evidence = evidence_by_ref.get(ref_id)
                if evidence is not None:
                    context.semantic_ledger.retain(evidence)
            contexts[query_node_id] = context
        return contexts

    def _install_recovered_execution_graph_revision(
        self,
        session: GroundedDeepAgentSession,
        payload: GroundedGraphRevisionRecoveryPayload,
        *,
        expected_base_execution_fingerprint: str,
        runtime_budget: GroundedRuntimeBudget | None,
    ) -> bool:
        proposal = GroundedExecutionGraphProposal.model_validate(
            payload.execution_proposal
        )
        receipt = GroundedExecutionGraphReceipt.model_validate(
            payload.execution_receipt
        )
        population_receipt = PopulationDynamicGraphReceipt.model_validate(
            payload.population_receipt
        )
        with session.lock:
            active_receipt = session.execution_graph_receipt
            if (
                active_receipt is not None
                and active_receipt.fingerprint == receipt.fingerprint
                and active_receipt.version == receipt.version
            ):
                return False
            if (
                active_receipt is None
                or active_receipt.fingerprint
                != expected_base_execution_fingerprint
            ):
                raise RuntimeError(
                    "GRAPH_REVISION_RECOVERY_EXECUTION_BASE_MISMATCH"
                )
            old_population_references = dict(
                session.population_pre_execution_references
            )
        contexts = self._revision_contexts_from_recovery_payload(
            session,
            payload,
            runtime_budget=runtime_budget,
        )
        refreshed_population_references: dict[
            str,
            PopulationPreExecutionReference,
        ] = {}
        if self.population_gate_enforced:
            gate = self.population_execution_gate
            workspace = session.context_workspace
            goal_contract = session.question_goal_contract
            if (
                gate is None
                or workspace is None
                or goal_contract is None
            ):
                raise RuntimeError(
                    "POPULATION_ONLINE_GATE_AUTHORITY_REQUIRED"
                )
            carried_query_ids = set(
                receipt.carried_forward_node_ids
            )
            for query_node_id, old_reference in (
                old_population_references.items()
            ):
                if query_node_id not in carried_query_ids:
                    continue
                refreshed_population_references[
                    query_node_id
                ] = gate.build_pre_execution_reference(
                    context_owner_fingerprint=(
                        workspace.owner_fingerprint
                    ),
                    run_authority_fingerprint=(
                        workspace.request_fingerprint
                    ),
                    goal_contract_fingerprint=(
                        original_question_goal_contract_fingerprint(
                            goal_contract
                        )
                    ),
                    graph_receipt=population_receipt,
                    node=old_reference.node,
                )
        carried_query_ids = set(receipt.carried_forward_node_ids)
        with session.lock:
            current = session.execution_graph_receipt
            if (
                current is not None
                and current.fingerprint == receipt.fingerprint
                and current.version == receipt.version
            ):
                return False
            if (
                current is None
                or current.fingerprint
                != expected_base_execution_fingerprint
            ):
                raise RuntimeError(
                    "GRAPH_REVISION_RECOVERY_EXECUTION_CAS_MISMATCH"
                )
            session.execution_graph_history.append(
                {
                    "status": "REVISED_RECOVERED",
                    "parentReceipt": current.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                    "receipt": receipt.model_dump(
                        by_alias=True,
                        mode="json",
                    ),
                }
            )
            session.query_branch_contexts = contexts
            session.execution_graph_proposal = proposal.model_copy(
                deep=True
            )
            session.execution_graph_receipt = receipt.model_copy(
                deep=True
            )
            session.execution_graph_edges = [
                edge.model_copy(deep=True)
                for edge in proposal.edges
            ]
            session.execution_graph_generation = receipt.version
            session.execution_graph_fingerprint = receipt.fingerprint
            if not any(
                context.status == "VERIFIED"
                or bool(context.verified_artifact_ids)
                for context in contexts.values()
            ):
                session.execution_graph_data_snapshot = None
            session.execution_graph_revision_count = max(
                session.execution_graph_revision_count,
                max(0, receipt.version - 1),
            )
            session.execution_graph_used_replan_fingerprints.extend(
                fingerprint
                for fingerprint in receipt.replan_evidence_fingerprints
                if fingerprint
                not in set(
                    session.execution_graph_used_replan_fingerprints
                )
            )
            session.execution_graph_revision_discovery_evidence_id = ""
            session.execution_graph_revision_discovery_evidence_ids = []
            session.parallel_branches = {
                query_node_id: branch
                for query_node_id, branch in session.parallel_branches.items()
                if query_node_id in contexts
            }
            session.parallel_branch_goal_ids = {
                query_node_id: list(goal_ids)
                for query_node_id, goal_ids in (
                    session.parallel_branch_goal_ids.items()
                )
                if query_node_id in contexts
            }
            session.population_graph_receipt = (
                population_receipt.model_copy(deep=True)
            )
            session.population_pre_execution_references = (
                refreshed_population_references
            )
            session.population_post_gate_results = {
                query_node_id: dict(result)
                for query_node_id, result in (
                    session.population_post_gate_results.items()
                )
                if query_node_id in carried_query_ids
            }
            session.population_artifact_query_node_ids = {
                artifact_id: query_node_id
                for artifact_id, query_node_id in (
                    session.population_artifact_query_node_ids.items()
                )
                if query_node_id in carried_query_ids
            }
        return True

    def _recover_pending_graph_revisions(
        self,
        session: GroundedDeepAgentSession,
        *,
        runtime_budget: GroundedRuntimeBudget | None,
    ) -> list[dict[str, Any]]:
        if not self.population_gate_enforced:
            return []
        workspace = session.context_workspace
        gate = self.population_execution_gate
        if workspace is None or gate is None:
            return []
        journal = GroundedGraphRevisionTransactionJournal(workspace)
        pending = journal.discover_pending()
        if not pending:
            return []
        goal_contract = session.question_goal_contract
        if goal_contract is None:
            raise RuntimeError(
                "GRAPH_REVISION_RECOVERY_BASE_SESSION_REQUIRED"
            )
        reports: list[dict[str, Any]] = []
        for recovery in pending:
            if runtime_budget is not None:
                runtime_budget.checkpoint()
            current = recovery
            payload = current.recovery_payload
            revised_population_receipt = (
                PopulationDynamicGraphReceipt.model_validate(
                    payload.population_receipt
                )
            )
            if current.next_action == "COMMIT_POPULATION":
                gate_result = gate.revise_graph(
                    context_owner_fingerprint=(
                        workspace.owner_fingerprint
                    ),
                    run_authority_fingerprint=(
                        workspace.request_fingerprint
                    ),
                    goal_contract_fingerprint=(
                        original_question_goal_contract_fingerprint(
                            goal_contract
                        )
                    ),
                    previous_graph_receipt_fingerprint=(
                        current.record.base_population_receipt_fingerprint
                    ),
                    revised_graph_receipt=(
                        revised_population_receipt
                    ),
                    revision_evidence_fingerprint=(
                        current.record.evidence_set_fingerprint
                    ),
                )
                if not gate_result.accepted:
                    raise RuntimeError(
                        "GRAPH_REVISION_RECOVERY_POPULATION_REJECTED:%s"
                        % gate_result.code
                    )
                advanced = journal.advance(
                    current.transaction_id,
                    target_status="POPULATION_COMMITTED",
                    expected_revision=current.record.revision,
                    expected_record_fingerprint=(
                        current.record.record_fingerprint
                    ),
                )
                current = journal.load_recovery(
                    advanced.record.transaction_id
                )
                if current is None:
                    raise RuntimeError(
                        "GRAPH_REVISION_RECOVERY_JOURNAL_MISSING"
                    )
            if current.next_action == "COMMIT_EXECUTION":
                self._install_recovered_execution_graph_revision(
                    session,
                    current.recovery_payload,
                    expected_base_execution_fingerprint=(
                        current.record.base_execution_receipt_fingerprint
                    ),
                    runtime_budget=runtime_budget,
                )
                advanced = journal.advance(
                    current.transaction_id,
                    target_status="EXECUTION_COMMITTED",
                    expected_revision=current.record.revision,
                    expected_record_fingerprint=(
                        current.record.record_fingerprint
                    ),
                )
                reports.append(
                    {
                        "transactionId": (
                            advanced.record.transaction_id
                        ),
                        "status": advanced.record.status,
                        "executionReceipt": (
                            current.recovery_payload.execution_receipt
                        ),
                    }
                )
        return reports

    def _commit_population_node_post(
        self,
        session: GroundedDeepAgentSession,
        query_node_id: str = "",
        *,
        execution_session: GroundedRuntimeSession | None = None,
    ) -> dict[str, Any]:
        if not self.population_gate_enforced:
            return {"accepted": True, "code": "NOT_ENFORCED"}
        gate = self.population_execution_gate
        with session.lock:
            if not query_node_id and execution_session is not None:
                candidates = tuple(
                    node_id
                    for node_id, item in (session.population_pre_execution_references.items())
                    if item.node.generation == execution_session.active_generation
                    and item.node.attempt_id == execution_session.active_attempt_id
                )
                if len(candidates) == 1:
                    query_node_id = candidates[0]
            reference = session.population_pre_execution_references.get(query_node_id)
        if gate is None or reference is None:
            raise RuntimeError("POPULATION_NODE_PRE_REFERENCE_REQUIRED")
        result = gate.commit_node_post_result(reference=reference)
        payload = {
            "accepted": result.accepted,
            "code": result.code,
            "stage": str(getattr(result.stage, "value", result.stage)),
            "queryNodeId": query_node_id,
        }
        with session.lock:
            session.population_post_gate_results[query_node_id] = dict(payload)
        if not result.accepted:
            raise RuntimeError("POPULATION_POST_RESULT_REJECTED:%s" % result.code)
        return payload

    @staticmethod
    def _population_goal_attestation_is_valid(
        attestation: PopulationVerificationAttestation | None,
        *,
        goal_contract_fingerprint: str,
    ) -> bool:
        if not isinstance(attestation, PopulationVerificationAttestation):
            return False
        return bool(
            attestation.stage == PopulationVerificationStage.GOAL_DECLARATION
            and attestation.passed
            and attestation.gate_open
            and attestation.goal_contract_fingerprint == goal_contract_fingerprint
            and attestation.attestation_fingerprint
            and attestation.attestation_fingerprint == population_attestation_fingerprint(attestation)
        )

    @staticmethod
    def _validated_population_goal_attestation(
        result: Any,
        *,
        goal_contract_fingerprint: str,
    ) -> PopulationVerificationAttestation | None:
        transition = getattr(result, "transition", None)
        state = getattr(transition, "state", None)
        attestation = getattr(state, "goal_attestation", None)
        if not GroundedDeepAgentRuntime._population_goal_attestation_is_valid(
            attestation,
            goal_contract_fingerprint=goal_contract_fingerprint,
        ):
            return None
        return attestation.model_copy(deep=True)

    @staticmethod
    def _goal_coverage_snapshot(
        session: GroundedDeepAgentSession,
    ) -> Any:
        contract = session.question_goal_contract
        if contract is None:
            raise RuntimeError("ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED")
        result = GoalCoverageVerifier().verify(
            contract,
            GroundedDeepAgentRuntime._goal_coverage_declarations(session),
        )
        session.goal_coverage_result = result.model_dump(by_alias=True)
        return result

    def _require_complete_goal_coverage(
        self,
        session: GroundedDeepAgentSession,
    ) -> Any:
        if self.population_gate_enforced:
            gate = self.population_execution_gate
            with session.lock:
                contract = (
                    session.question_goal_contract.model_copy(deep=True)
                    if session.question_goal_contract is not None
                    else None
                )
                goal_gate_id = session.population_goal_gate_id
                goal_gate_accepted = bool(session.population_goal_gate_result.get("accepted"))
                goal_attestation = (
                    session.population_goal_attestation.model_copy(deep=True)
                    if session.population_goal_attestation is not None
                    else None
                )
                references = tuple(session.population_pre_execution_references.values())
            if contract is None:
                raise RuntimeError("ORIGINAL_QUESTION_GOAL_CONTRACT_REQUIRED")
            goal_contract_fingerprint = original_question_goal_contract_fingerprint(contract)
            if (
                gate is None
                or not goal_gate_id
                or not goal_gate_accepted
                or not self._population_goal_attestation_is_valid(
                    goal_attestation,
                    goal_contract_fingerprint=goal_contract_fingerprint,
                )
            ):
                raise RuntimeError("POPULATION_GOAL_ATTESTATION_REQUIRED")
            if goal_attestation.accepted_scopes:
                if not references:
                    raise RuntimeError("POPULATION_GRAPH_COMPLETION_REQUIRED")
                completion = gate.require_graph_complete(reference=references[0])
                if not completion.accepted:
                    raise RuntimeError("POPULATION_GRAPH_INCOMPLETE:%s" % completion.code)
        result = GroundedDeepAgentRuntime._goal_coverage_snapshot(session)
        if not result.finalization_allowed:
            raise GoalCoverageBlocked(result)
        return result

    def run(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: Optional[MerchantInfo] = None,
        access_role: str = _DEFAULT_GROUNDED_ACCESS_ROLE,
        user_scope: Optional[dict[str, Any]] = None,
        thread_id: str = "",
        run_id: str = "",
        listener: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        request_context: Optional[ChatContext] = None,
        message_history: Optional[list[ConversationMessage]] = None,
    ) -> ChatResponse:
        actual_thread_id = thread_id or "thread_%s" % uuid.uuid4().hex
        actual_run_id = run_id or "run_%s" % uuid.uuid4().hex
        store = self.conversation_state_store
        transaction = store.locked(actual_thread_id) if store is not None else nullcontext()
        with transaction:
            persisted_snapshot: dict[str, Any] = {}
            persisted_revision = 0
            if store is not None:
                try:
                    persisted = store.load(actual_thread_id)
                except GroundedConversationStateCorruptError:
                    clarification = ClarificationRequest(
                        question=(
                            "当前会话的服务端状态无法安全恢复。为避免沿用错误的数据范围，"
                            "请重新说明这次查询的完整时间范围和筛选条件。"
                        ),
                        stage="conversation_state",
                        type="conversation_state_corrupt",
                        pending_question=str(question or "").strip(),
                    )
                    return ChatResponse(
                        answer=clarification.question,
                        clarification=clarification,
                        debug_trace={
                            "harness": {
                                "runtime": "grounded_deepagent",
                                "threadId": actual_thread_id,
                                "runId": actual_run_id,
                                "conversationResolution": {
                                    "status": "CORRUPT_STATE_REJECTED",
                                },
                            }
                        },
                    )
                if persisted is not None:
                    persisted_snapshot = dict(persisted.snapshot)
                    persisted_revision = int(persisted.revision)
                    expected_principal = str(persisted_snapshot.get("principalFingerprint") or "")
                    actual_principal = grounded_conversation_principal_fingerprint(
                        merchant_id,
                        user_scope,
                    )
                    if expected_principal and expected_principal != actual_principal:
                        raise PermissionError("GROUNDED_CONVERSATION_PRINCIPAL_SCOPE_MISMATCH")
            if self.conversation_online_authority is not None:
                active_scope = dict(persisted_snapshot.get("activeScope") or {})
                resolution = self.conversation_online_authority.resolve(
                    question,
                    persisted_snapshot=persisted_snapshot,
                    persisted_revision=persisted_revision,
                    request_context=request_context,
                    expected_principal_fingerprint=(
                        grounded_conversation_principal_fingerprint(
                            merchant_id,
                            user_scope,
                        )
                    ),
                    expected_context_owner_fingerprint=(
                        grounded_context_owner_fingerprint(
                            merchant_id,
                            access_role,
                            user_scope,
                        )
                    ),
                    expected_semantic_activation_fingerprint=str(
                        active_scope.get("semanticActivationFingerprint") or ""
                    ),
                )
            else:
                resolution = resolve_grounded_conversation_turn(
                    question,
                    persisted_snapshot=persisted_snapshot,
                    persisted_revision=persisted_revision,
                    message_history=message_history,
                    request_context=request_context,
                )
            if resolution.needs_clarification:
                clarification = ClarificationRequest(
                    question=resolution.clarification_question,
                    stage="conversation_reference",
                    type=(resolution.clarification_type or "CONVERSATION_REFERENCE_CLARIFICATION"),
                    options=list(resolution.clarification_options),
                    pending_question=resolution.original_question,
                )
                response = ChatResponse(
                    answer=clarification.question,
                    clarification=clarification,
                    debug_trace={
                        "harness": {
                            "runtime": "grounded_deepagent",
                            "threadId": actual_thread_id,
                            "runId": actual_run_id,
                            "conversationResolution": resolution.trace(),
                        }
                    },
                )
                self._persist_conversation_state(
                    actual_thread_id,
                    actual_run_id,
                    merchant_id,
                    user_scope,
                    resolution,
                    response,
                    previous_snapshot=persisted_snapshot,
                    expected_revision=persisted_revision,
                    session=None,
                )
                return response
            try:
                reference_scope = _trusted_reference_scope_binding(
                    resolution,
                    persisted_snapshot,
                )
            except (TypeError, ValueError) as exc:
                clarification = ClarificationRequest(
                    question=(
                        "上一轮结果的验证信息已经失效或不完整，不能安全地继续沿用。"
                        "请重新说明要分析的时间范围和筛选条件。"
                    ),
                    stage="conversation_reference",
                    type="reference_artifact_invalid",
                    pending_question=resolution.original_question,
                )
                response = ChatResponse(
                    answer=clarification.question,
                    clarification=clarification,
                    debug_trace={
                        "harness": {
                            "runtime": "grounded_deepagent",
                            "threadId": actual_thread_id,
                            "runId": actual_run_id,
                            "conversationResolution": resolution.trace(),
                            "referenceBindingError": str(exc)[:240],
                        }
                    },
                )
                self._persist_conversation_state(
                    actual_thread_id,
                    actual_run_id,
                    merchant_id,
                    user_scope,
                    resolution,
                    response,
                    previous_snapshot=persisted_snapshot,
                    expected_revision=persisted_revision,
                    session=None,
                )
                return response
            return self._run_once(
                resolution.effective_question,
                merchant_id,
                merchant=merchant,
                access_role=access_role,
                user_scope=user_scope,
                thread_id=actual_thread_id,
                run_id=actual_run_id,
                listener=listener,
                conversation_resolution=resolution,
                previous_conversation_snapshot=persisted_snapshot,
                expected_conversation_revision=persisted_revision,
                reference_scope=reference_scope,
            )

    def _run_once(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: Optional[MerchantInfo] = None,
        access_role: str = _DEFAULT_GROUNDED_ACCESS_ROLE,
        user_scope: Optional[dict[str, Any]] = None,
        thread_id: str = "",
        run_id: str = "",
        listener: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        conversation_resolution: Optional[GroundedConversationResolution] = None,
        previous_conversation_snapshot: Optional[dict[str, Any]] = None,
        expected_conversation_revision: int = 0,
        reference_scope: Optional[GroundedReferenceScopeBinding] = None,
    ) -> ChatResponse:
        actual_thread_id = thread_id or "thread_%s" % uuid.uuid4().hex
        actual_run_id = run_id or "run_%s" % uuid.uuid4().hex
        budget = GroundedRuntimeBudget.from_settings(self.settings or object())
        kernel_session: Optional[GroundedRuntimeSession] = None
        session: Optional[GroundedDeepAgentSession] = None
        try:
            with budget.stage("bootstrap.session"):
                context_workspace = (
                    GroundedContextWorkspace.open(
                        self.settings,
                        thread_id=actual_thread_id,
                        run_id=actual_run_id,
                        merchant_id=merchant_id,
                        access_role=access_role,
                        user_scope=user_scope,
                        question=question,
                    )
                    if (
                        self.settings is not None
                        and getattr(
                            self.settings,
                            "resolved_workspace_path",
                            None,
                        )
                        is not None
                    )
                    else None
                )
                session_kwargs: dict[str, Any] = {
                    "merchant": merchant,
                    "access_role": access_role,
                    "user_scope": user_scope,
                }
                if reference_scope is not None and reference_scope.enabled:
                    session_kwargs["reference_scope"] = reference_scope
                kernel_session = self.kernel.new_session(
                    question,
                    merchant_id,
                    **session_kwargs,
                )
            with budget.stage("routing.topic"):
                if isinstance(self.kernel, GroundedRuntimeKernel):
                    self.kernel.route_topic(
                        kernel_session,
                        runtime_budget=budget,
                    )
                else:
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
                        detail="%s:%s" % (type(exc).__name__, str(exc)[:500]),
                    )
                )
            session = GroundedDeepAgentSession(
                runtime=kernel_session,
                context_workspace=context_workspace,
                context_artifact_inline_max_rows=max(
                    1,
                    int(
                        getattr(
                            self.settings,
                            "context_artifact_inline_max_rows",
                            1,
                        )
                        or 1
                    ),
                ),
                execution_graph_max_revision_count=max(
                    1,
                    int(
                        getattr(
                            self.settings,
                            "grounded_execution_graph_max_revisions",
                            2,
                        )
                        or 2
                    ),
                ),
                population_gate_enforced=self.population_gate_enforced,
                conversation_context=(
                    {
                        "resolution": conversation_resolution.trace(),
                        "activeScope": dict((previous_conversation_snapshot or {}).get("activeScope") or {}),
                    }
                    if conversation_resolution is not None
                    else {}
                ),
            )
            if self.population_gate_enforced:
                if self.population_execution_gate is None or context_workspace is None:
                    raise RuntimeError("POPULATION_ONLINE_GATE_AUTHORITY_REQUIRED")
                self.population_execution_gate.register_run(
                    workspace=context_workspace,
                    ledger_provider=(
                        lambda active=session: tuple(
                            _authorized_verified_query_artifacts(active)
                        )
                        + tuple(
                            active.population_staged_query_artifacts.values()
                        )
                    ),
                )
                try:
                    recovered_graph_revisions = (
                        self._recover_pending_graph_revisions(
                            session,
                            runtime_budget=budget,
                        )
                    )
                except (
                    GroundedGraphRevisionJournalError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    session.operational_failure = {
                        "code": (
                            "GRAPH_REVISION_RECOVERY_FAILED"
                        ),
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                        "retryable": False,
                    }
                    session.runtime.phase = "OPERATIONAL_FAILURE"
                else:
                    if recovered_graph_revisions:
                        session.execution_graph_history.append(
                            {
                                "status": (
                                    "JOURNAL_RECOVERY_COMPLETED_AT_BOOTSTRAP"
                                ),
                                "transactions": (
                                    recovered_graph_revisions
                                ),
                            }
                        )
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
                response = ChatResponse(
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
                            "performance": {
                                "totalDurationMs": report.get(
                                    "elapsedMs",
                                    0,
                                ),
                                "runtimeBudget": report,
                            },
                            "goalCoverage": {},
                            "operationalFailure": failure,
                        }
                    },
                )
                if conversation_resolution is not None:
                    response.debug_trace["harness"]["conversationResolution"] = conversation_resolution.trace()
                return response
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
            response = self._governed_response(
                session,
                actual_thread_id,
                actual_run_id,
            )
            return self._finalize_conversation_response(
                response,
                session=session,
                resolution=conversation_resolution,
                thread_id=actual_thread_id,
                run_id=actual_run_id,
                merchant_id=merchant_id,
                user_scope=user_scope,
                previous_snapshot=previous_conversation_snapshot or {},
                expected_revision=expected_conversation_revision,
            )
        except Exception as exc:
            if session is not None and session.runtime.answer and self._answer_is_attested(session):
                # compose_verified_answer is terminal, but preserve a verified
                # answer if a provider/framework performs an unnecessary tail
                # turn and that tail fails.  A post-answer transport error must
                # never overwrite successfully verified evidence.
                session.runtime.events.append(
                    GroundedRuntimeEvent(
                        sequence=len(session.runtime.events) + 1,
                        stage="post_answer_tail",
                        status="IGNORED",
                        detail="%s:%s" % (type(exc).__name__, str(exc)[:300]),
                    )
                )
            elif _is_provider_timeout_error(exc):
                if session is None and kernel_session is not None:
                    session = GroundedDeepAgentSession(runtime=kernel_session)
                if session is None:
                    raise
                session.operational_failure = {
                    "code": "GROUNDED_PROVIDER_TIMEOUT",
                    "message": "%s:%s" % (type(exc).__name__, str(exc)[:500]),
                    "retryable": True,
                }
                session.runtime.phase = "PROVIDER_TIMEOUT"
                _emit_runtime_listener(
                    listener,
                    "runtime.provider_timeout",
                    "GROUNDED_CORE",
                    session.operational_failure,
                )
            else:
                raise
        finally:
            if session is not None and not session.runtime_budget_report:
                session.runtime_budget_report = budget.finish()
        assert session is not None
        response = self._governed_response(session, actual_thread_id, actual_run_id)
        return self._finalize_conversation_response(
            response,
            session=session,
            resolution=conversation_resolution,
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            merchant_id=merchant_id,
            user_scope=user_scope,
            previous_snapshot=previous_conversation_snapshot or {},
            expected_revision=expected_conversation_revision,
        )

    def _finalize_conversation_response(
        self,
        response: ChatResponse,
        *,
        session: GroundedDeepAgentSession,
        resolution: Optional[GroundedConversationResolution],
        thread_id: str,
        run_id: str,
        merchant_id: str,
        user_scope: Optional[dict[str, Any]],
        previous_snapshot: dict[str, Any],
        expected_revision: int,
    ) -> ChatResponse:
        harness = dict((response.debug_trace or {}).get("harness") or {})
        if resolution is not None:
            harness["conversationResolution"] = resolution.trace()
        response.debug_trace = {
            **dict(response.debug_trace or {}),
            "harness": harness,
        }
        if resolution is not None:
            self._persist_conversation_state(
                thread_id,
                run_id,
                merchant_id,
                user_scope,
                resolution,
                response,
                previous_snapshot=previous_snapshot,
                expected_revision=expected_revision,
                session=session,
            )
        return response

    def _persist_conversation_state(
        self,
        thread_id: str,
        run_id: str,
        merchant_id: str,
        user_scope: Optional[dict[str, Any]],
        resolution: GroundedConversationResolution,
        response: ChatResponse,
        *,
        previous_snapshot: dict[str, Any],
        expected_revision: int,
        session: Optional[GroundedDeepAgentSession],
    ) -> None:
        store = self.conversation_state_store
        if store is None:
            return
        previous = dict(previous_snapshot or {})
        now = datetime.now(timezone.utc)
        ttl_seconds = max(
            60,
            int(
                getattr(
                    self.settings,
                    "thread_context_summary_ttl_seconds",
                    2_592_000,
                )
                or 2_592_000
            ),
        )
        operational_failure = bool(((response.debug_trace or {}).get("harness") or {}).get("operationalFailure"))
        verified_scope = (
            self._verified_conversation_scope(session) if session is not None and not operational_failure else {}
        )
        active_scope = verified_scope if verified_scope.get("artifactIds") else dict(previous.get("activeScope") or {})
        pending: dict[str, Any] = {}
        if response.clarification is not None:
            pending = {
                "stage": str(response.clarification.stage or ""),
                "type": str(response.clarification.type or ""),
                "pendingQuestion": str(response.clarification.pending_question or resolution.effective_question),
                "options": list(response.clarification.options),
                "sourceRunId": run_id,
            }
        turn_status = (
            "CLARIFICATION_REQUIRED"
            if response.clarification is not None
            else "FAILED"
            if operational_failure
            else "VERIFIED"
            if verified_scope.get("artifactIds")
            else "COMPLETED"
        )
        turns = [dict(item) for item in (previous.get("turns") or []) if isinstance(item, dict)][-11:]
        turns.append(
            {
                "runId": run_id,
                "originalQuestion": resolution.original_question,
                "effectiveQuestion": resolution.effective_question,
                "resolutionStatus": resolution.status,
                "status": turn_status,
                "artifactIds": list(verified_scope.get("artifactIds") or []),
                "createdAt": now.isoformat().replace("+00:00", "Z"),
            }
        )
        snapshot = {
            "stateVersion": GROUNDED_CONVERSATION_STATE_VERSION,
            "merchantId": str(merchant_id or "").strip(),
            "principalFingerprint": grounded_conversation_principal_fingerprint(
                merchant_id,
                user_scope,
            ),
            "lastTurn": turns[-1],
            "turns": turns,
            "activeScope": active_scope,
            "pendingClarification": pending,
            "expiresAt": (now + timedelta(seconds=ttl_seconds)).isoformat().replace("+00:00", "Z"),
        }
        store.save_snapshot(
            thread_id,
            snapshot,
            expected_revision=expected_revision,
        )

    def _verified_conversation_scope(
        self,
        session: Optional[GroundedDeepAgentSession],
    ) -> dict[str, Any]:
        if session is None or session.context_workspace is None or self.settings is None:
            return {}
        try:
            artifact_root_relative_path = str(
                session.context_workspace.artifacts_root.relative_to(self.settings.resolved_workspace_path)
            )
        except (AttributeError, ValueError):
            return {}
        state = session.runtime
        selected_ids = set(state.answer_artifact_ids)
        artifacts = [
            item
            for item in _authorized_verified_query_artifacts(session)
            if item.verified_evidence.passed
            and item.publication_status == "PUBLISHED"
            and verified_query_artifact_integrity_valid(item)
            and (not selected_ids or item.artifact_id in selected_ids)
        ][:8]
        if selected_ids and {item.artifact_id for item in artifacts} != selected_ids:
            return {}
        if not artifacts:
            return {}
        result_sets: list[dict[str, Any]] = []
        source_artifacts: list[dict[str, Any]] = []
        all_times: list[str] = []
        all_filters: list[str] = []
        semantic_activation_fingerprint = ""
        context_owner_fingerprint = ""
        retained_artifact_ids: list[str] = []
        for artifact in artifacts:
            result_artifact_receipts = _grounded_result_artifact_receipts(artifact.run_result)
            if len(result_artifact_receipts) != 1:
                return {}
            publication_receipt = dict(result_artifact_receipts[0])
            receipt_owner = str(publication_receipt.get("contextOwnerFingerprint") or "").strip()
            receipt_activation = str(publication_receipt.get("semanticActivationFingerprint") or "").strip()
            if (
                receipt_owner != session.context_workspace.owner_fingerprint
                or len(receipt_activation) != 64
                or any(character not in "0123456789abcdef" for character in receipt_activation)
                or (semantic_activation_fingerprint and semantic_activation_fingerprint != receipt_activation)
                or (context_owner_fingerprint and context_owner_fingerprint != receipt_owner)
            ):
                return {}
            semantic_activation_fingerprint = receipt_activation
            context_owner_fingerprint = receipt_owner
            contract = artifact.contract
            time_range = contract.time_range
            absolute_time = ""
            if bool(time_range.explicit):
                start = str(time_range.execution_start_date or time_range.start_date or "").strip()
                end = str(time_range.execution_end_date or time_range.end_date or "").strip()
                absolute_time = (
                    "%s 至 %s" % (start, end)
                    if start and end and start != end
                    else start or end or str(time_range.label or "").strip()
                )
            filter_summaries = [_conversation_filter_summary(item) for item in contract.entity_filters]
            filter_summaries = [item for item in filter_summaries if item]
            times = [absolute_time] if absolute_time else []
            goal_ids = list(session.artifact_goal_ids.get(artifact.artifact_id) or [])
            bundle = artifact.run_result.merged_query_bundle
            result_set = {
                "label": "%s %s"
                % (
                    "/".join(contract.topics),
                    str(contract.query_shape or ""),
                ),
                "queryShape": str(contract.query_shape or ""),
                "topics": list(contract.topics),
                "tables": list(bundle.tables),
                "timeExpressions": times,
                "filterSummaries": filter_summaries,
                "goalIds": goal_ids,
                "queryArtifactId": artifact.artifact_id,
                "contractFingerprint": artifact.contract_fingerprint,
                "sqlFingerprint": artifact.sql_fingerprint,
                "previewRowCount": len(bundle.rows),
                "completeRowCount": int(bundle.effective_row_count()),
                "offloaded": bool(bundle.offloaded_files),
                "coverageStatus": str(
                    getattr(bundle, "coverage_status", "")
                    or getattr(bundle, "result_coverage", "")
                    or (
                        "TOP_N"
                        if str(contract.query_shape or "").upper() == "RANKED"
                        else "PREVIEW"
                        if bool(getattr(bundle, "is_truncated", False))
                        else "UNKNOWN"
                    )
                ).upper(),
                "hasMore": bool(getattr(bundle, "has_more", False) or getattr(bundle, "is_truncated", False)),
                "entityIdentities": list(
                    dict.fromkeys(
                        str(item).strip() for item in artifact.output_entity_identities.values() if str(item).strip()
                    )
                ),
                "outputEntityIdentities": dict(artifact.output_entity_identities),
                "dataGrains": list(
                    dict.fromkeys(
                        str(item.data_grain or "").strip()
                        for item in contract.tables
                        if str(item.data_grain or "").strip()
                    )
                ),
                "snapshotSemantics": "ABSOLUTE_PREDICATE_SNAPSHOT",
            }
            result_sets.append(result_set)
            retained_artifact_ids.append(artifact.artifact_id)
            source_artifacts.append(
                {
                    "queryArtifactId": artifact.artifact_id,
                    "publicationStatus": "PUBLISHED",
                    "artifactRootRelativePath": (artifact_root_relative_path),
                    "contractFingerprint": artifact.contract_fingerprint,
                    "sqlFingerprint": artifact.sql_fingerprint,
                    "goalIds": goal_ids,
                    "contract": contract.model_dump(by_alias=True, mode="json"),
                    "resultArtifactReceipts": [publication_receipt],
                }
            )
            all_times.extend(times)
            all_filters.extend(filter_summaries)
        return {
            "artifactIds": retained_artifact_ids,
            "timeExpressions": list(dict.fromkeys(all_times)),
            "filterSummaries": list(dict.fromkeys(all_filters)),
            "resultSets": result_sets,
            "sourceArtifacts": source_artifacts,
            "contextOwnerFingerprint": context_owner_fingerprint,
            "semanticActivationFingerprint": (semantic_activation_fingerprint),
            "scopeSemantics": "VERIFIED_QUERY_PREDICATE_SNAPSHOT",
            "previewRowsAreCompletePopulation": False,
        }

    def _initial_context(self, session: GroundedDeepAgentSession) -> str:
        manifests: list[dict[str, Any]] = []
        for topic_name in session.runtime.workspace_topics:
            result = self.semantic_catalog.read(
                path="topics/%s/manifest.json" % topic_name,
                max_chars=80_000,
                offset=0,
            )
            if not isinstance(result, dict) or not result.get("success"):
                raise RuntimeError("Grounded bootstrap failed: Topic L0 manifest unavailable for %s" % topic_name)
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
        conversation_context = dict(session.conversation_context or {})
        active_conversation_scope = dict(conversation_context.get("activeScope") or {})
        compact_source_artifacts = [
            {
                "queryArtifactId": str(item.get("queryArtifactId") or ""),
                "contractFingerprint": str(item.get("contractFingerprint") or ""),
                "sqlFingerprint": str(item.get("sqlFingerprint") or ""),
                "goalIds": list(item.get("goalIds") or []),
            }
            for item in active_conversation_scope.get("sourceArtifacts") or []
            if isinstance(item, dict)
        ][:8]
        payload = {
            "question": session.runtime.question,
            "trustedConversationContext": {
                "resolution": dict(conversation_context.get("resolution") or {}),
                "activeScope": {
                    "scopeSemantics": str(active_conversation_scope.get("scopeSemantics") or ""),
                    "artifactIds": list(active_conversation_scope.get("artifactIds") or []),
                    "timeExpressions": list(active_conversation_scope.get("timeExpressions") or []),
                    "filterSummaries": list(active_conversation_scope.get("filterSummaries") or []),
                    "resultSets": list(active_conversation_scope.get("resultSets") or [])[:8],
                    "sourceArtifacts": compact_source_artifacts,
                    "previewRowsAreCompletePopulation": False,
                },
                "authority": (
                    "Server-generated from verified query artifacts. Reuse the full "
                    "predicate scope; never infer population membership from preview rows."
                ),
            },
            "userInputRequirements": {
                "explicitTimeExpression": has_explicit_time_expression(session.runtime.question),
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
                        session.runtime.user_scope.get("storeIds") or session.runtime.user_scope.get("store_ids") or []
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
            "thinRecallCandidates": _thin_recall(session.runtime.recall, limit=4),
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
                    "RULE",
                    "DETAIL",
                    "RANKING",
                    "ANALYSIS",
                ],
                "queryAssignmentRequired": True,
                "queryTopologyDecision": "LATE_BOUND_AFTER_FORMAL_EVIDENCE",
                "executionGraphFreezePoint": "IMMEDIATELY_BEFORE_QUERY_PREPARATION",
                "topologyAuthority": "CORE_LLM_WITHIN_KERNEL_VALIDATED_EVIDENCE",
                "branchTopicScopeAuthority": "CORE_LLM_WITHIN_ROUTED_TOPICS",
                "finalizationRequiresVerifiedCoverage": True,
                "parallelRule": ("Only goals without dependency/upstream entity bindings may share a batch."),
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
                "Declare the immutable original-question Goals first, then progressively read formal assets. "
                "Do not freeze query branches or topology before that evidence is sufficient. Immediately "
                "before query preparation, either propose one grounded Contract or freeze a validated graph; "
                "the Core may merge, split, parallelize, or serialize nodes only from formal evidence and "
                "typed population/artifact dependencies. For a frozen multi-node graph, submit exact "
                "semanticPaths through prepare_grounded_query_batch instead of using global filesystem tools. "
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
                "verifiedRuleArtifactIds": list(state.answer_rule_artifact_ids),
                "verifiedRuleArtifactCount": len(state.answer_rule_artifact_ids),
                "verifiedAnalysisArtifactIds": [item.artifact_id for item in session.verified_analysis_ledger],
                "verifiedAnalysisArtifactCount": len(session.verified_analysis_ledger),
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
                "performance": {
                    "totalDurationMs": session.runtime_budget_report.get(
                        "elapsedMs",
                        0,
                    ),
                    "runtimeBudget": dict(session.runtime_budget_report),
                },
                "queryBranches": [context.report() for context in session.query_branch_contexts.values()],
                "goalCoverage": dict(session.goal_coverage_result),
                "answerCoverage": dict(session.answer_coverage_result),
                "analysisDataInputGate": dict(session.analysis_data_input_gate_result),
                "contextManagement": {
                    "modelCallCount": len(session.core_context_reports),
                    "latest": (dict(session.core_context_reports[-1]) if session.core_context_reports else {}),
                    "calls": [dict(item) for item in session.core_context_reports[-16:]],
                },
            }
        }
        if session.operational_failure:
            trace["harness"]["operationalFailure"] = dict(session.operational_failure)
            return ChatResponse(
                answer=(
                    "本次查数未能在模型调用时限内完成。系统没有把未完成或未验证的结果当作最终答案；请稍后重试。"
                    if session.operational_failure.get("code") == "GROUNDED_PROVIDER_TIMEOUT"
                    else "本次查数未能在运行预算内完成。系统没有把未完成或未验证的结果当作最终答案；"
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
        if state.answer and GroundedDeepAgentRuntime._answer_is_attested(session):
            display_run = state.answer_run_result or state.run_result
            inline_limit = max(
                1,
                int(session.context_artifact_inline_max_rows or 1),
            )
            rows = list(display_run.merged_query_bundle.rows)[:inline_limit] if display_run is not None else []
            tables = list(display_run.merged_query_bundle.tables) if display_run is not None else []
            sections: list[dict[str, Any]] = []
            answered_artifact_ids = set(state.answer_artifact_ids)
            for artifact in _authorized_verified_query_artifacts(
                session
            ):
                if artifact.artifact_id not in answered_artifact_ids:
                    continue
                bundle = artifact.run_result.merged_query_bundle
                artifact_receipts = _public_grounded_result_artifact_receipts(artifact.run_result)
                preview_rows = [
                    {
                        **dict(row),
                        "__evidenceArtifactId": artifact.artifact_id,
                    }
                    for row in list(bundle.rows)[:inline_limit]
                ]
                result_coverage = str(bundle.result_coverage or "UNKNOWN")
                original_row_count = max(
                    0,
                    int(bundle.original_row_count or 0),
                )
                original_row_count_exact = original_row_count > 0
                if not original_row_count_exact and result_coverage in {"ALL_ROWS", "TOP_N"}:
                    original_row_count = len(bundle.rows)
                    original_row_count_exact = True
                has_more = (
                    len(bundle.rows) > len(preview_rows)
                    or original_row_count > len(preview_rows)
                    or result_coverage == "PREVIEW"
                    or bool(bundle.is_truncated)
                )
                sections.append(
                    {
                        "title": (artifact.contract.query_shape or "Verified query evidence"),
                        "resultRole": "verified_query_artifact",
                        "dorisTables": list(bundle.tables),
                        "dataRows": preview_rows,
                        "offloaded": bool(artifact_receipts),
                        "offloadedFiles": _public_grounded_result_refs(artifact_receipts),
                        "resultArtifacts": artifact_receipts,
                        "previewRowCount": len(preview_rows),
                        "originalRowCount": original_row_count,
                        "originalRowCountExact": (original_row_count_exact),
                        "resultCoverage": result_coverage,
                        "hasMore": has_more,
                        "resultSummary": ("artifact=%s; verified=true" % artifact.artifact_id),
                    }
                )
            return ChatResponse(
                answer=state.answer,
                category_name=state.routing.display_summary(),
                doris_tables=tables,
                data_rows=rows,
                data_sections=sections,
                debug_trace=trace,
            )
        if state.answer:
            raise RuntimeError("Grounded final answer exists without a matching answer-coverage attestation")
        raise RuntimeError("Grounded DeepAgent Core ended without verified answer or typed clarification")


def _grounded_contract_sql_obligations(contract: Any) -> dict[str, Any]:
    """Expose the normalized SQL obligations Core must implement exactly."""

    required_outputs = list(
        dict.fromkeys(
            [
                *[str(item.metric_key or "") for item in contract.metrics],
                *[str(item.output_alias or item.column or "") for item in contract.selected_fields],
                *[str(item.column or "") for item in contract.dimensions if item.usage == "group_by"],
            ]
        )
    )
    return {
        "requiredFinalOutputAliases": [item for item in required_outputs if item],
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
        "timeField": contract.time_field.model_dump(by_alias=True),
        "referenceScope": (
            {
                "enabled": True,
                "referentType": contract.reference_scope.referent_type,
                "downstreamOperation": contract.reference_scope.downstream_operation,
                "sourceArtifactId": contract.reference_scope.source_artifact_id,
                "sourceContractFingerprint": contract.reference_scope.source_contract_fingerprint,
                "sourceSqlFingerprint": contract.reference_scope.source_sql_fingerprint,
                "sourceQueryShape": contract.reference_scope.source_query_shape,
                "sourceTables": [
                    {
                        "topic": item.topic,
                        "table": item.table,
                        "timeColumn": item.time_column,
                        "tenantColumn": item.merchant_filter_column,
                        "dataGrain": item.data_grain,
                    }
                    for item in contract.reference_scope.source_tables
                ],
                "sourceTimeRange": contract.reference_scope.source_time_range.model_dump(by_alias=True),
                "sourceTimeColumns": dict(contract.reference_scope.source_time_columns),
                "sourceEntityFilters": [
                    {
                        "semanticRefId": item.semantic_ref_id,
                        "table": item.table,
                        "column": item.column,
                        "operator": item.operator,
                        "literalValue": item.literal_value,
                    }
                    for item in contract.reference_scope.source_entity_filters
                ],
                "coverageStatus": contract.reference_scope.coverage_status,
                "snapshotSemantics": contract.reference_scope.snapshot_semantics,
                "populationRequired": contract.reference_scope.population_required,
                "membershipHandleType": contract.reference_scope.membership_handle_type,
                "membershipHandleId": contract.reference_scope.membership_handle_id,
                "membershipValuesHash": contract.reference_scope.membership_values_hash,
                "instruction": (
                    "Preserve this typed source lineage exactly. Predicate scope means the full "
                    "verified source predicate, never the visible preview rows."
                ),
            }
            if contract.reference_scope.enabled
            else {"enabled": False}
        ),
    }


def _core_visible_binding_hints(contract: Any) -> dict[str, Any]:
    """Keep kernel-owned upstream values out of the parent Core context."""

    upstream_refs = {str(item.target_field_ref or "") for item in contract.upstream_entity_bindings}
    hints = contract.binding_hints.model_copy(
        update={
            "entity_filters": [
                item for item in contract.binding_hints.entity_filters if item.field_ref not in upstream_refs
            ]
        },
        deep=True,
    )
    return hints.model_dump(by_alias=True)


def _core_visible_entity_filter_obligations(contract: Any) -> list[dict[str, Any]]:
    upstream_by_ref = {str(item.target_field_ref or ""): item for item in contract.upstream_entity_bindings}
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
                "snippet": item.content[:600] if inline_only else item.content[:160],
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
                item.model_copy(update={"field_ref": _canonical_progressive_ref(item.field_ref)})
                for item in hints.field_aggregations
            ],
            "dimension_refs": [_canonical_progressive_ref(item) for item in hints.dimension_refs],
            "selected_fields": [
                item.model_copy(update={"field_ref": _canonical_progressive_ref(item.field_ref)})
                for item in hints.selected_fields
            ],
            "entity_filters": [
                item.model_copy(update={"field_ref": _canonical_progressive_ref(item.field_ref)})
                for item in hints.entity_filters
            ],
            "upstream_entity_bindings": [
                item.model_copy(update={"target_field_ref": _canonical_progressive_ref(item.target_field_ref)})
                for item in hints.upstream_entity_bindings
            ],
            "group_by_ref": _canonical_progressive_ref(hints.group_by_ref),
            "time_field_ref": _canonical_progressive_ref(hints.time_field_ref),
            "label_refs": {_canonical_progressive_ref(key): value for key, value in hints.label_refs.items()},
            "relationship_refs": [_canonical_progressive_ref(item) for item in hints.relationship_refs],
            "ranking": hints.ranking.model_copy(
                update={"metric_ref": _canonical_progressive_ref(hints.ranking.metric_ref)}
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


def _reference_contract_time_columns(
    contract: GroundedQueryContract,
) -> dict[str, list[str]]:
    table_bindings = {item.table: item for item in contract.tables if item.table}
    result: dict[str, list[str]] = {}

    def add(table: str, column: str) -> None:
        table_name = str(table or "").strip()
        column_name = str(column or "").strip()
        if not table_name or not column_name:
            return
        result.setdefault(table_name, [])
        if column_name not in result[table_name]:
            result[table_name].append(column_name)

    for metric in contract.metrics:
        table = table_bindings.get(metric.table)
        add(metric.table, metric.time_column or (table.time_column if table else ""))
    primary = table_bindings.get(contract.primary_table)
    if primary is not None:
        add(primary.table, primary.time_column)
    for entity_filter in contract.entity_filters:
        table = table_bindings.get(entity_filter.table)
        if table is not None:
            add(table.table, table.time_column)
    if not result:
        for table in contract.tables:
            add(table.table, table.time_column)
    return result


def _trusted_reference_scope_binding(
    resolution: GroundedConversationResolution,
    persisted_snapshot: Optional[dict[str, Any]],
) -> GroundedReferenceScopeBinding:
    """Rehydrate one typed reference only from a sealed server artifact.

    The model-visible conversation text is intentionally irrelevant here.  A
    source Contract must still be present, parseable and byte-semantically
    equivalent to its stored fingerprint after a restart or deployment.
    """

    reference = resolution.reference_contract
    if not reference.bound or not reference.source_artifact_id:
        return GroundedReferenceScopeBinding()
    active_scope = dict((persisted_snapshot or {}).get("activeScope") or {})
    source_records = [
        dict(item)
        for item in active_scope.get("sourceArtifacts") or []
        if isinstance(item, dict) and str(item.get("queryArtifactId") or "").strip() == reference.source_artifact_id
    ]
    if len(source_records) != 1:
        raise ValueError("REFERENCE_SOURCE_ARTIFACT_NOT_UNIQUE")
    source_record = source_records[0]
    raw_contract = source_record.get("contract")
    if not isinstance(raw_contract, dict):
        raise ValueError("REFERENCE_SOURCE_CONTRACT_MISSING")
    source_contract = GroundedQueryContract.model_validate(raw_contract)
    actual_fingerprint = grounded_query_contract_fingerprint(source_contract)
    expected_fingerprints = {
        str(reference.source_contract_fingerprint or "").strip(),
        str(source_record.get("contractFingerprint") or "").strip(),
    }
    expected_fingerprints.discard("")
    if not expected_fingerprints or expected_fingerprints != {actual_fingerprint}:
        raise ValueError("REFERENCE_SOURCE_CONTRACT_FINGERPRINT_MISMATCH")
    expected_sql = str(reference.source_sql_fingerprint or "").strip()
    stored_sql = str(source_record.get("sqlFingerprint") or "").strip()
    if expected_sql and stored_sql and expected_sql != stored_sql:
        raise ValueError("REFERENCE_SOURCE_SQL_FINGERPRINT_MISMATCH")
    if source_contract.status != "READY" or not source_contract.ready:
        raise ValueError("REFERENCE_SOURCE_CONTRACT_NOT_READY")
    return GroundedReferenceScopeBinding(
        enabled=True,
        status="BOUND",
        referent_type=reference.referent_type,
        downstream_operation=reference.downstream_operation,
        source_artifact_id=reference.source_artifact_id,
        source_contract_fingerprint=actual_fingerprint,
        source_sql_fingerprint=stored_sql or expected_sql,
        source_query_shape=reference.source_query_shape or source_contract.query_shape,
        source_contract_version=source_contract.contract_version,
        source_topics=list(reference.source_topics or source_contract.topics),
        source_tables=[item.model_copy(deep=True) for item in source_contract.tables],
        source_entity_filters=[item.model_copy(deep=True) for item in source_contract.entity_filters],
        source_time_range=source_contract.time_range.model_copy(deep=True),
        source_time_columns=_reference_contract_time_columns(source_contract),
        source_goal_ids=list(reference.source_goal_ids),
        source_entity_identities=list(reference.source_entity_identities),
        source_data_grains=(
            list(reference.source_data_grains)
            or [
                str(item.data_grain or "").strip()
                for item in source_contract.tables
                if str(item.data_grain or "").strip()
            ]
        ),
        source_evidence_refs=list(source_contract.evidence_refs),
        coverage_status=reference.coverage_status,
        snapshot_semantics=reference.snapshot_semantics,
        population_required=reference.population_required,
        complete_membership_required=reference.complete_membership_required,
        membership_handle_type=reference.membership_handle_type,
        membership_handle_id=reference.membership_handle_id,
        membership_values_hash=reference.membership_values_hash,
        current_turn_explicit_time=has_explicit_time_expression(resolution.original_question),
        verified_server_side=True,
    )


def _conversation_filter_summary(binding: Any) -> str:
    """Render one verified non-tenant predicate for reference resolution."""

    phrase = str(getattr(binding, "requested_phrase", "") or "").strip()
    if phrase:
        return phrase[:300]
    label = str(
        getattr(binding, "business_name", "")
        or getattr(binding, "column", "")
        or getattr(binding, "semantic_ref_id", "")
        or ""
    ).strip()
    operator = str(getattr(binding, "operator", "") or "EQ").strip().upper()
    value = getattr(binding, "literal_value", None)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        rendered = ",".join(str(item)[:80] for item in items[:10])
        if len(items) > 10:
            rendered += ",…(%d values)" % len(items)
    else:
        rendered = str(value)[:300]
    if not label or value is None:
        return ""
    return "%s %s %s" % (label, operator, rendered)


def _deepagent_reader_is_core() -> bool:
    """Fail closed for filesystem reads originating from a subagent graph."""

    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return False
    configurable = dict(config.get("configurable") or {})
    metadata = dict(config.get("metadata") or {})
    checkpoint_ns = str(configurable.get("checkpoint_ns") or configurable.get("checkpointNamespace") or "").lower()
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
    output: list[str] = []
    separator_pending = False
    for character in str(value or "").strip().lower():
        if character.isascii() and (character.islower() or character.isdigit()):
            if separator_pending and output:
                output.append("-")
            output.append(character)
            separator_pending = False
            continue
        separator_pending = bool(output)
    return "".join(output).strip("-")


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
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
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
            "timeRange": (contract.time_range.model_dump(by_alias=True) if contract is not None else {}),
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
        str(spec.get("semanticRefId") or ""): _normalized_formula(spec.get("metricFormula") or "")
        for spec in specs
        if str(spec.get("semanticRefId") or "") and str(spec.get("metricFormula") or "").strip()
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
        ref_id = str(item.get("metricRef") or item.get("semanticRefId") or item.get("evidenceRef") or "")
        disclosure_text = str(item.get("definition") or "")
        raw_formula = str(item.get("formula") or "")
        if not raw_formula and _contains_sql_formula_call(disclosure_text):
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

    for raw_formula in _backtick_segments(str(structured.get("answerMarkdown") or "")):
        if not _contains_sql_formula_call(raw_formula):
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
    return "".join(str(value or "").split()).strip("`").upper()


_GOVERNED_SQL_FORMULA_FUNCTIONS = {
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "NULLIF",
    "COALESCE",
}


def _contains_sql_formula_call(value: Any) -> bool:
    text = str(value or "")
    index = 0
    while index < len(text):
        character = text[index]
        if not (character.isascii() and (character.isalpha() or character == "_")):
            index += 1
            continue
        end = index + 1
        while end < len(text):
            candidate = text[end]
            if not (candidate.isascii() and (candidate.isalnum() or candidate == "_")):
                break
            end += 1
        token = text[index:end].upper()
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if token in _GOVERNED_SQL_FORMULA_FUNCTIONS and cursor < len(text) and text[cursor] == "(":
            return True
        index = end
    return False


def _backtick_segments(value: Any) -> list[str]:
    text = str(value or "")
    segments: list[str] = []
    start: Optional[int] = None
    for index, character in enumerate(text):
        if character != "`":
            continue
        if start is None:
            start = index + 1
            continue
        segments.append(text[start:index])
        start = None
    return segments


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
