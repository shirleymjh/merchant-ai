from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

from pydantic import Field

from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.models import (
    APIModel,
    AgentRunResult,
    AnswerMode,
    ClarificationRequest,
    DataSnapshotContract,
    EvidenceGap,
    EntityFilterObligation,
    EntityReference,
    ExtractedKeywords,
    IntentType,
    KnowledgeRetrievalRequest,
    MerchantInfo,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
    RecallRoundTrace,
    RetrievalIssue,
    ResultCoverage,
    RouteSlots,
    TaskRole,
    TopicRoutingDecision,
    VerifiedEvidence,
)


from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedContractGap,
    GroundedEntityFilterHint,
    GroundedQueryContract,
    GroundedQueryContractBuilder,
    GroundedReferenceScopeBinding,
    GroundedUpstreamArtifactBinding,
    GroundedUpstreamEntityBinding,
    compile_deterministic_grounded_query,
    materialize_grounded_asset_pack,
    requested_semantic_label,
)
from merchant_ai.services.grounded_contract_repair import (
    build_contract_repair_directive,
)
from merchant_ai.services.grounded_execution_policy import (
    DETERMINISTIC_EXECUTION_MODES,
    GroundedExecutionMode,
    evaluate_deterministic_execution,
)
from merchant_ai.services.entity_contracts import (
    canonical_entity_values,
    entity_comparison_policy,
    entity_filter_contract_hash,
    entity_value_hash,
)
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlCandidate,
    GroundedSqlCandidateValidator,
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.grounded_query_proof import (
    build_grounded_query_proof,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
    RouteSlotExtractor,
    TopicRouterService,
)
from merchant_ai.services.grounded_rule_artifact import (
    GroundedVerifiedRuleArtifact,
    render_verified_rule_answer,
)
from merchant_ai.services.grounded_context_workspace import (
    grounded_context_owner_fingerprint,
)
from merchant_ai.services.grounded_semantic_activation import (
    GroundedSemanticActivationSeal,
    build_semantic_activation_seal,
    canonical_semantic_topics,
    semantic_activation_seal_valid,
    valid_semantic_activation_fingerprint,
)
from merchant_ai.services.grounded_semantic_ir import (
    GroundedTrustedArtifactDescriptor,
    compose_query_calculation_graph,
    trusted_entity_set_descriptor,
    trusted_query_artifact_descriptor,
)

if TYPE_CHECKING:
    from merchant_ai.services.grounded_population_runtime_gate import (
        PopulationPreExecutionReference,
    )
else:
    PopulationPreExecutionReference = Any

DEFAULT_ACCESS_ROLE = load_authorization_policy().default_access_role


class GroundedRuntimeAttempt(APIModel):
    attempt_id: str
    contract: GroundedQueryContract
    contract_version: int = 1
    parent_attempt_id: str = ""
    parent_contract_fingerprint: str = ""
    repair_type: str = "NONE"
    repair_directive: dict[str, Any] = Field(default_factory=dict)
    status: str = "PROPOSED"
    compile_status: str = "NOT_ATTEMPTED"
    activation_status: str = "NOT_ATTEMPTED"
    execution_mode: GroundedExecutionMode = GroundedExecutionMode.UNDECIDED
    execution_reason_codes: list[str] = Field(default_factory=list)
    fast_path_eligible: bool = False
    fast_path_reason_codes: list[str] = Field(default_factory=list)
    fast_path_reason_details: dict[str, str] = Field(default_factory=dict)
    next_action: str = "RESOLVE_CONTRACT"
    activated: bool = False
    active_generation: int = 0
    error: str = ""
    validation_gaps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""


class GroundedRuntimeSqlCandidateAttempt(APIModel):
    candidate_id: str
    active_generation: int
    candidate_version: int = 1
    parent_candidate_id: str = ""
    parent_ast_fingerprint: str = ""
    status: str = "PROPOSED"
    next_action: str = "REPAIR_SQL"
    ast_fingerprint: str = ""
    contract_fingerprint: str = ""
    progress_fingerprint: str = ""
    output_columns: list[str] = Field(default_factory=list)
    validation_gaps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""


class GroundedVerifiedQueryArtifact(APIModel):
    """Immutable snapshot of one independently grounded and verified query."""

    artifact_id: str
    generation: int
    attempt_id: str = ""
    contract_fingerprint: str
    sql_fingerprint: str
    contract: GroundedQueryContract
    plan: QueryPlan
    run_result: AgentRunResult
    verified_evidence: VerifiedEvidence
    execution_mode: str = ""
    sql_validation: Optional[GroundedSqlValidationResult] = None
    ranking_semantics_verified: bool = False
    output_columns: list[str] = Field(default_factory=list)
    output_semantic_refs: dict[str, str] = Field(default_factory=dict)
    output_entity_identities: dict[str, str] = Field(default_factory=dict)
    output_lineage: dict[str, list[str]] = Field(default_factory=dict)
    sealed_entity_values: dict[str, list[Any]] = Field(default_factory=dict)
    sealed_entity_values_truncated: bool = False
    publication_status: str = "VERIFIED_IN_MEMORY"
    result_artifact_receipts: list[dict[str, Any]] = Field(
        default_factory=list
    )
    plan_fingerprint: str = ""
    run_result_fingerprint: str = ""
    semantic_activation_fingerprint: str = ""
    semantic_activation_seal_fingerprint: str = ""
    semantic_activation_topics: list[str] = Field(default_factory=list)
    query_proof_version: str = "grounded_query_proof.v1"
    query_proof_fingerprint: str = ""
    executed_sql_ast_fingerprint: str = ""
    ledger_fingerprint: str = ""
    trusted_descriptor: Optional[GroundedTrustedArtifactDescriptor] = None
    created_at: str = ""


class GroundedPendingQueryPublication(APIModel):
    """Server-private capability for staged, not-yet-visible query bytes."""

    pending_artifact_id: str
    generation: int
    attempt_id: str
    receipt: dict[str, Any] = Field(default_factory=dict, exclude=True)
    receipt_fingerprint: str = ""
    run_result_fingerprint: str = ""
    status: str = "PENDING"


class GroundedVerifiedEntitySet(APIModel):
    """Kernel-retained entity values published from verified query evidence."""

    artifact_id: str
    source_query_artifact_id: str
    source_column: str
    source_semantic_ref_id: str
    source_entity_identity: str = ""
    values: list[Any] = Field(default_factory=list)
    value_count: int = 0
    truncated: bool = False
    values_hash: str = ""
    trusted_descriptor: Optional[GroundedTrustedArtifactDescriptor] = None
    created_at: str = ""


@dataclass(frozen=True)
class GroundedCoreSqlPreparation:
    """Accepted Core SQL plus the evidence-only plan used downstream."""

    plan: QueryPlan
    candidate_id: str
    sql_candidate: GroundedSqlCandidate
    candidate_validation: GroundedSqlValidationResult
    asset_pack_fingerprint: str

    @property
    def executable(self) -> bool:
        return bool(self.candidate_validation.valid)


class GroundedRuntimeEvent(APIModel):
    sequence: int
    stage: str
    status: str
    detail: str = ""
    attempt_id: str = ""


class GroundedRuntimeSession(APIModel):
    session_id: str
    question: str
    retrieval_question: str = ""
    merchant_id: str
    merchant: MerchantInfo = Field(default_factory=MerchantInfo)
    access_role: str = DEFAULT_ACCESS_ROLE
    user_scope: dict[str, Any] = Field(default_factory=dict)
    reference_scope: GroundedReferenceScopeBinding = Field(
        default_factory=GroundedReferenceScopeBinding
    )
    phase: str = "CREATED"
    revision: int = 0
    keywords: ExtractedKeywords = Field(default_factory=ExtractedKeywords)
    route_slots: RouteSlots = Field(default_factory=RouteSlots)
    routing: TopicRoutingDecision = Field(default_factory=TopicRoutingDecision)
    workspace_topics: list[str] = Field(default_factory=list)
    semantic_activation_seal: Optional[
        GroundedSemanticActivationSeal
    ] = None
    semantic_activation_execution_started: bool = False
    recall: RecallBundle = Field(default_factory=RecallBundle)
    recall_rounds: list[RecallRoundTrace] = Field(default_factory=list)
    recall_index_version: str = ""
    recall_semantic_source_hash: str = ""
    recall_retrieval_status: str = "not_started"
    recall_retrieval_issues: list[RetrievalIssue] = Field(default_factory=list)
    attempts: list[GroundedRuntimeAttempt] = Field(default_factory=list)
    active_generation: int = 0
    active_attempt_id: str = ""
    active_goal_contract_fingerprint: str = ""
    active_execution_mode: GroundedExecutionMode = GroundedExecutionMode.UNDECIDED
    active_execution_reason_codes: list[str] = Field(default_factory=list)
    active_contract: Optional[GroundedQueryContract] = None
    active_pack: Optional[PlanningAssetPack] = None
    active_plan: Optional[QueryPlan] = None
    active_preparation: Any = None
    active_sql_candidate: Optional[GroundedSqlCandidate] = None
    active_sql_validation: Optional[GroundedSqlValidationResult] = None
    sql_candidate_attempts: list[GroundedRuntimeSqlCandidateAttempt] = Field(
        default_factory=list
    )
    sql_execution_repair_context: dict[str, Any] = Field(default_factory=dict)
    rejected_sql_candidate_fingerprints: list[str] = Field(default_factory=list)
    executed_sql_candidate_fingerprints: list[str] = Field(default_factory=list)
    repair_exhausted_contract_fingerprints: list[str] = Field(default_factory=list)
    terminal_guard_code: str = ""
    run_result: Optional[AgentRunResult] = None
    verified_evidence: Optional[VerifiedEvidence] = None
    verified_query_ledger: list[GroundedVerifiedQueryArtifact] = Field(
        default_factory=list
    )
    pending_query_publications: list[GroundedPendingQueryPublication] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )
    artifact_publication_required: bool = Field(
        default=False,
        exclude=True,
        repr=False,
    )
    defer_artifact_publication: bool = Field(
        default=False,
        exclude=True,
        repr=False,
    )
    publication_authority_run_result: Optional[AgentRunResult] = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    publication_authority_fingerprint: str = Field(
        default="",
        exclude=True,
        repr=False,
    )
    verified_rule_ledger: list[GroundedVerifiedRuleArtifact] = Field(
        default_factory=list
    )
    verified_entity_sets: list[GroundedVerifiedEntitySet] = Field(
        default_factory=list
    )
    upstream_artifact_bindings: list[GroundedUpstreamArtifactBinding] = Field(
        default_factory=list
    )
    answer_plan: Optional[QueryPlan] = None
    answer_run_result: Optional[AgentRunResult] = None
    answer_verified_evidence: Optional[VerifiedEvidence] = None
    answer_artifact_ids: list[str] = Field(default_factory=list)
    answer_rule_artifact_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    clarification: Optional[ClarificationRequest] = None
    events: list[GroundedRuntimeEvent] = Field(default_factory=list)


def _recall_knowledge_ref_ids(recall: RecallBundle) -> set[str]:
    refs: set[str] = set()
    for item in recall.items:
        metadata = item.metadata or {}
        for raw in (
            item.doc_id,
            metadata.get("semanticRefId"),
            metadata.get("semantic_ref_id"),
            metadata.get("knowledgeRefId"),
            metadata.get("knowledge_ref_id"),
        ):
            value = str(raw or "").strip()
            if value:
                refs.add(value)
    return refs


def _plan_has_knowledge_evidence_contract(plan: QueryPlan) -> bool:
    return any(
        str(
            contract.get("evidenceSource")
            or contract.get("evidence_source")
            or ""
        ).lower()
        in {"knowledge_ref", "knowledge", "rule"}
        for contract in plan.evidence_contracts
    )


class GroundedRuntimeKernel:
    """Independent runtime for the progressively grounded query path.

    The kernel owns state transitions but delegates domain work to narrow
    services. It deliberately has no graph/workflow dependency: a candidate
    Contract is routed by a capability-based execution policy. Safe single-
    table scalar, multi-metric, grouped, trend, ranked and entity lookup shapes
    use deterministic compilation. JOIN/CTE/window/dependency topology remains
    Core-authored SQL.
    """

    def __init__(
        self,
        topic_assets: Any,
        *,
        keyword_service: Any | None = None,
        route_slot_extractor: Any | None = None,
        topic_router: Any | None = None,
        recall_service: Any | None = None,
        contract_builder: Any | None = None,
        asset_materializer: Callable[[GroundedQueryContract, Any], PlanningAssetPack]
        | None = None,
        compiler: Callable[[GroundedQueryContract, PlanningAssetPack], Any] | None = None,
        sql_candidate_validator: Any | None = None,
        executor: Any | None = None,
        verifier: Any | None = None,
        answer_composer: Any | None = None,
    ):
        self.topic_assets = topic_assets
        self.keyword_service = keyword_service or KeywordExtractService(topic_assets)
        self.route_slot_extractor = (
            route_slot_extractor or RouteSlotExtractor(topic_assets)
        )
        self.topic_router = topic_router or TopicRouterService(topic_assets)
        self.recall_service = recall_service
        self.contract_builder = contract_builder or GroundedQueryContractBuilder()
        self.asset_materializer = asset_materializer or materialize_grounded_asset_pack
        self.compiler = compiler or compile_deterministic_grounded_query
        self.sql_candidate_validator = (
            sql_candidate_validator or GroundedSqlCandidateValidator()
        )
        self.executor = executor
        self.verifier = verifier
        self.answer_composer = answer_composer
        executor_settings = getattr(executor, "settings", None)
        configured_preview_rows = getattr(
            executor_settings,
            "context_artifact_inline_max_rows",
            None,
        )
        self._result_preview_rows = (
            max(0, int(configured_preview_rows))
            if configured_preview_rows is not None
            else None
        )
        self._lock = RLock()

    def new_session(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: MerchantInfo | None = None,
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: dict[str, Any] | None = None,
        reference_scope: GroundedReferenceScopeBinding | dict[str, Any] | None = None,
        retrieval_question: str = "",
        session_id: str = "",
    ) -> GroundedRuntimeSession:
        normalized_question = str(question or "").strip()
        if not normalized_question:
            raise ValueError("grounded runtime requires a non-empty question")
        normalized_retrieval_question = str(retrieval_question or "").strip()
        if not normalized_retrieval_question:
            normalized_retrieval_question = normalized_question
        normalized_merchant_id = str(merchant_id or "").strip()
        if not normalized_merchant_id:
            raise ValueError("grounded runtime requires merchant_id")
        return GroundedRuntimeSession(
            session_id=session_id or "grounded_%s" % uuid.uuid4().hex[:16],
            question=normalized_question,
            retrieval_question=normalized_retrieval_question,
            merchant_id=normalized_merchant_id,
            merchant=(merchant or MerchantInfo(merchant_id=normalized_merchant_id)).model_copy(
                update={"merchant_id": normalized_merchant_id}
            ),
            access_role=access_role or DEFAULT_ACCESS_ROLE,
            user_scope=dict(user_scope or {}),
            reference_scope=(
                reference_scope.model_copy(deep=True)
                if isinstance(reference_scope, GroundedReferenceScopeBinding)
                else GroundedReferenceScopeBinding.model_validate(
                    reference_scope or {}
                )
            ),
        )

    def fork_query_branch(
        self,
        session: GroundedRuntimeSession,
        branch_id: str,
        *,
        workspace_topics: Sequence[str] | None = None,
        objective: str = "",
        inherit_entity_set_ids: Sequence[str] | None = None,
        inherit_query_artifact_ids: Sequence[str] | None = None,
        upstream_artifact_bindings: Sequence[
            GroundedUpstreamArtifactBinding
        ] | None = None,
    ) -> GroundedRuntimeSession:
        """Create an isolated execution branch for one independent query goal.

        Active Contract generations are intentionally branch-local.  This lets
        independent queries compile and execute concurrently without making a
        sibling SQL candidate stale.  Verified artifacts are adopted into the
        parent only through :meth:`adopt_verified_branches`.
        """

        token = str(branch_id or "").strip()
        if (
            not token
            or len(token) > 128
            or any(
                not (character.isascii() and (character.isalnum() or character in {"_", "-"}))
                for character in token
            )
        ):
            raise ValueError(
                "parallel query branch_id must be a typed ASCII identifier"
            )
        self.seal_semantic_activation(
            session,
            session.workspace_topics,
        )
        branch = self.new_session(
            str(objective or session.question).strip(),
            session.merchant_id,
            merchant=session.merchant.model_copy(deep=True),
            access_role=session.access_role,
            user_scope=dict(session.user_scope),
            reference_scope=session.reference_scope.model_copy(deep=True),
            session_id="%s__branch_%s_%s"
            % (session.session_id, token[:48], uuid.uuid4().hex[:8]),
        )
        branch.defer_artifact_publication = True
        branch.keywords = session.keywords.model_copy(deep=True)
        branch.routing = session.routing.model_copy(deep=True)
        branch.workspace_topics = _dedupe(
            workspace_topics or session.workspace_topics
        )
        with self._lock:
            parent_semantic_seal = (
                session.semantic_activation_seal.model_copy(deep=True)
                if session.semantic_activation_seal is not None
                else None
            )
            parent_execution_started = bool(
                session.semantic_activation_execution_started
            )
        if parent_semantic_seal is not None:
            if any(
                topic not in set(parent_semantic_seal.exact_topics)
                for topic in branch.workspace_topics
            ):
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_BRANCH_TOPIC_SCOPE_MISMATCH"
                )
            branch.semantic_activation_seal = parent_semantic_seal
            branch.semantic_activation_execution_started = (
                parent_execution_started
            )
        branch.recall = session.recall.model_copy(deep=True)
        branch.recall_rounds = [item.model_copy(deep=True) for item in session.recall_rounds]
        branch.recall_index_version = str(session.recall_index_version or "")
        branch.recall_semantic_source_hash = str(session.recall_semantic_source_hash or "")
        branch.recall_retrieval_status = str(session.recall_retrieval_status or "not_started")
        branch.recall_retrieval_issues = [
            item.model_copy(deep=True) for item in session.recall_retrieval_issues
        ]
        branch.sql_execution_repair_context = deepcopy(
            session.sql_execution_repair_context
        )
        branch.active_goal_contract_fingerprint = str(
            session.active_goal_contract_fingerprint or ""
        )
        requested_entity_set_ids = _dedupe(inherit_entity_set_ids or [])
        if requested_entity_set_ids:
            with self._lock:
                entity_sets = {
                    item.artifact_id: item.model_copy(deep=True)
                    for item in session.verified_entity_sets
                    if item.artifact_id in set(requested_entity_set_ids)
                }
                source_query_ids = {
                    item.source_query_artifact_id
                    for item in entity_sets.values()
                }
                source_queries = [
                    item.model_copy(deep=True)
                    for item in session.verified_query_ledger
                    if item.artifact_id in source_query_ids
                    and item.verified_evidence.passed
                ]
            missing_entity_sets = [
                item
                for item in requested_entity_set_ids
                if item not in entity_sets
            ]
            if missing_entity_sets:
                raise RuntimeError(
                    "VERIFIED_ENTITY_SET_NOT_FOUND:%s"
                    % ",".join(missing_entity_sets)
                )
            if len(source_queries) != len(source_query_ids):
                raise RuntimeError("VERIFIED_ENTITY_SET_SOURCE_QUERY_REQUIRED")
            branch.verified_entity_sets = [
                entity_sets[item].model_copy(deep=True)
                for item in requested_entity_set_ids
            ]
            branch.verified_query_ledger = source_queries
        requested_query_artifact_ids = _dedupe(
            inherit_query_artifact_ids or []
        )
        if requested_query_artifact_ids:
            with self._lock:
                query_artifacts = {
                    item.artifact_id: item.model_copy(deep=True)
                    for item in session.verified_query_ledger
                    if item.artifact_id
                    in set(requested_query_artifact_ids)
                    and item.verified_evidence.passed
                }
            missing_query_artifacts = [
                artifact_id
                for artifact_id in requested_query_artifact_ids
                if artifact_id not in query_artifacts
            ]
            if missing_query_artifacts:
                raise RuntimeError(
                    "VERIFIED_QUERY_ARTIFACT_NOT_FOUND:%s"
                    % ",".join(missing_query_artifacts)
                )
            inherited = {
                item.artifact_id: item
                for item in branch.verified_query_ledger
            }
            inherited.update(query_artifacts)
            branch.verified_query_ledger = list(inherited.values())
        normalized_upstream_bindings = [
            item.model_copy(deep=True)
            for item in (upstream_artifact_bindings or [])
        ]
        expected_scope_fingerprint = grounded_context_owner_fingerprint(
            session.merchant_id,
            session.access_role,
            session.user_scope,
        )
        inherited_query_ids = {
            item.artifact_id for item in branch.verified_query_ledger
        }
        for binding in normalized_upstream_bindings:
            descriptor = binding.descriptor
            if not descriptor.immutable:
                raise RuntimeError("UPSTREAM_ARTIFACT_MUST_BE_IMMUTABLE")
            if descriptor.artifact_id != binding.artifact_id:
                raise RuntimeError("UPSTREAM_ARTIFACT_IDENTITY_MISMATCH")
            if (
                descriptor.merchant_scope_fingerprint
                and descriptor.merchant_scope_fingerprint
                != expected_scope_fingerprint
            ):
                raise RuntimeError("UPSTREAM_ARTIFACT_MERCHANT_SCOPE_MISMATCH")
            if binding.artifact_id not in inherited_query_ids:
                raise RuntimeError(
                    "UPSTREAM_QUERY_ARTIFACT_NOT_INHERITED:%s"
                    % binding.artifact_id
                )
            if not binding.target_binding_ref:
                raise RuntimeError("UPSTREAM_ARTIFACT_TARGET_BINDING_REQUIRED")
        branch.upstream_artifact_bindings = normalized_upstream_bindings
        branch.phase = "BRANCH_CREATED"
        self._event(
            branch,
            "fork_query_branch",
            "READY",
            "parent=%s;branch=%s" % (session.session_id, token),
        )
        return branch

    def adopt_verified_branches(
        self,
        session: GroundedRuntimeSession,
        branches: Sequence[GroundedRuntimeSession],
        *,
        pre_adoption_authorizer: Optional[
            Callable[
                [
                    GroundedRuntimeSession,
                    Sequence[GroundedVerifiedQueryArtifact],
                ],
                bool,
            ]
        ] = None,
    ) -> list[GroundedVerifiedQueryArtifact]:
        """Atomically merge successful independent branches into the parent.

        Failed or unverified branch state never mutates the parent.  The last
        successful branch becomes the active compatibility snapshot so legacy
        answer composition and a later serial entity-chain query can continue.
        """

        if self.semantic_activation_authority_available():
            self.revalidate_semantic_activation(session)

        with self._lock:
            verified_branches = [
                branch
                for branch in branches
                if branch.verified_query_ledger
                and branch.verified_evidence is not None
                and branch.verified_evidence.passed
            ]
            if not verified_branches:
                return []
            parent_semantic_seal = (
                session.semantic_activation_seal.model_copy(deep=True)
                if session.semantic_activation_seal is not None
                else None
            )
            branch_semantic_seals = [
                branch.semantic_activation_seal.model_copy(deep=True)
                for branch in verified_branches
                if branch.semantic_activation_seal is not None
            ]
            if self.semantic_activation_authority_available():
                if (
                    parent_semantic_seal is None
                    or not semantic_activation_seal_valid(
                        parent_semantic_seal
                    )
                    or len(branch_semantic_seals)
                    != len(verified_branches)
                    or any(
                        not semantic_activation_seal_valid(item)
                        or item.seal_fingerprint
                        != parent_semantic_seal.seal_fingerprint
                        for item in branch_semantic_seals
                    )
                ):
                    raise RuntimeError(
                        "VERIFIED_BRANCH_SEMANTIC_ACTIVATION_MISMATCH"
                    )
            elif branch_semantic_seals and any(
                item.seal_fingerprint
                != branch_semantic_seals[0].seal_fingerprint
                for item in branch_semantic_seals[1:]
            ):
                raise RuntimeError(
                    "VERIFIED_BRANCH_SEMANTIC_ACTIVATION_MISMATCH"
                )
            parent_cas = {
                "generation": session.active_generation,
                "revision": session.revision,
                "ledgerIds": tuple(
                    item.artifact_id
                    for item in session.verified_query_ledger
                ),
            }
            branch_snapshots: list[dict[str, Any]] = []
            for branch in verified_branches:
                if branch.run_result is None or branch.verified_evidence is None:
                    raise RuntimeError("VERIFIED_BRANCH_RUN_RESULT_REQUIRED")
                generation = branch.active_generation
                branch_snapshots.append(
                    {
                        "branch": branch,
                        "generation": generation,
                        "attemptId": branch.active_attempt_id,
                        "verified": branch.verified_evidence.model_copy(
                            deep=True
                        ),
                        "verifiedFingerprint": _stable_json_hash(
                            branch.verified_evidence.model_dump(
                                by_alias=True,
                                mode="json",
                            )
                        ),
                        "publication": self._publication_snapshot_locked(
                            branch,
                            generation,
                        ),
                        "ledgerRefs": list(branch.verified_query_ledger),
                        "ledgerToken": tuple(
                            (
                                id(item),
                                item.artifact_id,
                                item.publication_status,
                                item.ledger_fingerprint,
                            )
                            for item in branch.verified_query_ledger
                        ),
                    }
                )

        # Snapshot postflight is performed by the graph runtime before this
        # call. All hash/copy/publish work remains outside the global lock.
        publication_batches: list[dict[str, Any]] = []
        for snapshot in branch_snapshots:
            try:
                source_integrity = tuple(
                    (
                        id(source),
                        verified_query_artifact_integrity_fingerprint(
                            source
                        ),
                    )
                    for source in snapshot["ledgerRefs"]
                )
                if any(
                    source.ledger_fingerprint
                    and source.ledger_fingerprint != observed
                    for source, (_identity, observed) in zip(
                        snapshot["ledgerRefs"],
                        source_integrity,
                    )
                ):
                    raise RuntimeError("VERIFIED_BRANCH_LEDGER_CORRUPT")
                requests = self._pending_publication_requests(
                    snapshot["publication"],
                    snapshot["verified"],
                )
                receipts = self._publish_requests(requests) if requests else []
                published_run_result = snapshot["publication"][
                    "runResult"
                ].model_copy(deep=True)
                published_run_result.verified_evidence = snapshot[
                    "verified"
                ].model_copy(deep=True)
                if receipts:
                    self._attach_published_receipts(
                        published_run_result,
                        snapshot["publication"]["pending"],
                        receipts,
                    )
                parent_ledger: list[GroundedVerifiedQueryArtifact] = []
                for source in snapshot["ledgerRefs"]:
                    if source.generation != snapshot["generation"]:
                        continue
                    artifact = self._copy_verified_artifact_with_run_result(
                        source,
                        published_run_result,
                    )
                    if (
                        artifact.publication_status == "PENDING"
                        and receipts
                    ):
                        artifact.publication_status = "PUBLISHED"
                        artifact.result_artifact_receipts = [
                            dict(item) for item in receipts
                        ]
                    artifact.ledger_fingerprint = (
                        verified_query_artifact_integrity_fingerprint(
                            artifact
                        )
                    )
                    parent_ledger.append(artifact)
                branch_public_result = self._bounded_run_result_projection(
                    published_run_result
                )
                branch_audit_ledger = [
                    self._copy_verified_artifact_with_run_result(
                        item,
                        branch_public_result,
                    )
                    for item in parent_ledger
                ]
                for item in branch_audit_ledger:
                    item.run_result_fingerprint = (
                        _query_run_result_fingerprint(item.run_result)
                    )
                    item.ledger_fingerprint = (
                        verified_query_artifact_integrity_fingerprint(item)
                    )
                source_integrity_post = tuple(
                    (
                        id(source),
                        verified_query_artifact_integrity_fingerprint(
                            source
                        ),
                    )
                    for source in snapshot["ledgerRefs"]
                )
                if source_integrity_post != source_integrity:
                    raise RuntimeError("VERIFIED_BRANCH_STATE_STALE")
                publication_batches.append(
                    {
                        **snapshot,
                        "requests": requests,
                        "receipts": receipts,
                        "runResult": branch_public_result,
                        "parentRunResult": branch_public_result.model_copy(
                            deep=True
                        ),
                        "preparedLedger": branch_audit_ledger,
                        "parentLedger": parent_ledger,
                    }
                )
            except Exception as exc:
                # No parent allowlist/ledger mutation has happened. Any files
                # already written are inert without the atomic commit below.
                raise RuntimeError(
                    "VERIFIED_BRANCH_ARTIFACT_PUBLICATION_FAILED:%s:%s"
                    % (type(exc).__name__, str(exc)[:300])
                ) from exc

        if pre_adoption_authorizer is not None:
            authorized_batches: list[dict[str, Any]] = []
            for batch in publication_batches:
                authorized = pre_adoption_authorizer(
                    batch["branch"],
                    tuple(
                        artifact.model_copy(deep=True)
                        for artifact in batch["parentLedger"]
                    ),
                )
                if authorized:
                    authorized_batches.append(batch)
            publication_batches = authorized_batches
            verified_branches = [
                batch["branch"] for batch in publication_batches
            ]
            if not publication_batches:
                return []

        with self._lock:
            if (
                session.active_generation != parent_cas["generation"]
                or session.revision != parent_cas["revision"]
                or tuple(
                    item.artifact_id
                    for item in session.verified_query_ledger
                )
                != parent_cas["ledgerIds"]
            ):
                raise RuntimeError("PARENT_VERIFIED_LEDGER_STALE")
            for batch in publication_batches:
                branch = batch["branch"]
                generation = int(batch["generation"])
                if (
                    branch.active_generation != generation
                    or branch.active_attempt_id != batch["attemptId"]
                    or branch.verified_evidence is None
                    or _stable_json_hash(
                        branch.verified_evidence.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                    )
                    != batch["verifiedFingerprint"]
                    or tuple(
                        (
                            id(item),
                            item.artifact_id,
                            item.publication_status,
                            item.ledger_fingerprint,
                        )
                        for item in branch.verified_query_ledger
                    )
                    != batch["ledgerToken"]
                ):
                    raise RuntimeError("VERIFIED_BRANCH_STATE_STALE")
                if batch["requests"]:
                    self._assert_pending_publication_cas_locked(
                        branch,
                        generation,
                        batch["requests"],
                    )

            # One short commit makes every branch receipt and the parent
            # allowlist ledger visible together.
            for batch in publication_batches:
                branch = batch["branch"]
                branch.run_result = batch["runResult"]
                branch.verified_query_ledger = batch["preparedLedger"]
                for pending in branch.pending_query_publications:
                    if pending.generation == batch["generation"]:
                        pending.status = "PUBLISHED"
                self._scrub_pending_publication_receipts(
                    branch,
                    int(batch["generation"]),
                )
                branch.publication_authority_run_result = None
                branch.publication_authority_fingerprint = ""
            artifacts = [
                artifact
                for batch in publication_batches
                for artifact in batch["parentLedger"]
                if artifact.verified_evidence.passed
                and artifact.publication_status
                in {"PUBLISHED", "VERIFIED_IN_MEMORY"}
            ]
            if not artifacts:
                return []
            existing_ids = {
                item.artifact_id for item in session.verified_query_ledger
            }
            adopted = [
                artifact for artifact in artifacts if artifact.artifact_id not in existing_ids
            ]
            if not adopted:
                return []
            session.verified_query_ledger.extend(adopted)
            last = verified_branches[-1]
            last_batch = publication_batches[-1]
            session.active_generation += 1
            if (
                session.semantic_activation_seal is None
                and branch_semantic_seals
            ):
                session.semantic_activation_seal = (
                    branch_semantic_seals[0].model_copy(deep=True)
                )
            if branch_semantic_seals:
                session.semantic_activation_execution_started = True
            session.active_attempt_id = last.active_attempt_id
            session.active_execution_mode = last.active_execution_mode
            session.active_execution_reason_codes = list(
                last.active_execution_reason_codes
            )
            session.active_contract = (
                last.active_contract.model_copy(deep=True)
                if last.active_contract is not None
                else None
            )
            session.active_pack = (
                last.active_pack.model_copy(deep=True)
                if last.active_pack is not None
                else None
            )
            session.active_plan = (
                last.active_plan.model_copy(deep=True)
                if last.active_plan is not None
                else None
            )
            session.active_preparation = deepcopy(last.active_preparation)
            session.active_sql_candidate = (
                last.active_sql_candidate.model_copy(deep=True)
                if last.active_sql_candidate is not None
                else None
            )
            session.active_sql_validation = (
                last.active_sql_validation.model_copy(deep=True)
                if last.active_sql_validation is not None
                else None
            )
            session.run_result = last_batch["parentRunResult"]
            session.verified_evidence = (
                last.verified_evidence.model_copy(deep=True)
                if last.verified_evidence is not None
                else None
            )
            if last.terminal_guard_code:
                session.terminal_guard_code = last.terminal_guard_code
            self._clear_answer_snapshot(session)
            session.phase = "PARALLEL_BRANCHES_ADOPTED"
            session.revision += 1
            self._event(
                session,
                "adopt_verified_branches",
                "ADOPTED",
                "branches=%d;artifacts=%d"
                % (len(verified_branches), len(adopted)),
            )
        adopted_projection: list[GroundedVerifiedQueryArtifact] = []
        for item in adopted:
            projected = self._copy_verified_artifact_with_run_result(
                item,
                self._bounded_run_result_projection(item.run_result),
            )
            projected.ledger_fingerprint = (
                verified_query_artifact_integrity_fingerprint(projected)
            )
            adopted_projection.append(projected)
        return adopted_projection

    def route_topic(
        self,
        session: GroundedRuntimeSession,
        *,
        runtime_budget: Any = None,
    ) -> TopicRoutingDecision:
        # Grounded RAG routes from the complete question and published Topic
        # cards.  It deliberately does not pre-classify metrics, dimensions,
        # time, ranking shape or analysis intent before initial recall.
        keywords = ExtractedKeywords(
            normalized_question=session.question,
        )
        route_slots = RouteSlots()
        semantic_route = getattr(self.topic_router, "route_with_budget", None)
        if callable(semantic_route):
            routing = semantic_route(
                session.question,
                keywords=keywords,
                route_slots=route_slots,
                runtime_budget=runtime_budget,
            )
        else:
            route = self.topic_router.route
            try:
                routing = route(
                    session.question,
                    keywords,
                    route_slots=route_slots,
                )
            except TypeError as exc:
                # Test and compatibility routers may expose the historical
                # two-argument signature. Only retry that exact signature
                # mismatch; do not hide a TypeError raised by router logic.
                if "route_slots" not in str(exc):
                    raise
                routing = route(session.question, keywords)
        categories = routing.recall_topics()
        topic_names = list(self.topic_assets.topic_names_for_categories(categories))
        with self._lock:
            session.keywords = keywords
            session.route_slots = route_slots
            session.routing = routing
            session.workspace_topics = _dedupe(topic_names)
            session.phase = "TOPIC_ROUTED"
            self._event(session, "route_topic", "OK", routing.routing_mode)
        return routing

    def recall_navigation(
        self,
        session: GroundedRuntimeSession,
        *,
        query: str = "",
        history_rows: Sequence[dict[str, Any]] | None = None,
        knowledge_context: str = "",
        target_goal_ids: Sequence[str] | None = None,
        required_capabilities: Sequence[str] | None = None,
        coverage_receipt_id: str = "",
        strict_topic_scope: bool = True,
    ) -> RecallBundle:
        if self.recall_service is None:
            raise RuntimeError(
                "grounded recall service is not configured; refusing to fall back to legacy retrieval"
            )
        categories = list(session.routing.recall_topics())
        resolve_category = getattr(self.topic_assets, "resolve_topic_category", None)
        if callable(resolve_category):
            for topic_name in session.workspace_topics:
                category = resolve_category(topic_name)
                if category and category not in categories:
                    categories.append(category)
        retrieval_query = str(
            query or session.retrieval_question or session.question
        ).strip()
        retrieve = getattr(self.recall_service, "retrieve", None)
        knowledge_bundle = None
        if callable(retrieve):
            knowledge_bundle = retrieve(
                KnowledgeRetrievalRequest(
                    query=retrieval_query,
                    keywords=[],
                    history_rows=list(history_rows or []),
                    knowledge_context=knowledge_context,
                    merchant_id=session.merchant_id,
                    access_role=session.access_role,
                    permissions=[
                        str(item)
                        for item in session.user_scope.get("permissions", [])
                        if str(item or "").strip()
                    ],
                    topic_categories=categories,
                    route_slots={},
                    intent_kind="",
                    complexity="",
                    target_goal_ids=list(target_goal_ids or []),
                    required_capabilities=list(required_capabilities or []),
                    coverage_receipt_id=str(coverage_receipt_id or ""),
                    strict_topic_scope=bool(strict_topic_scope),
                )
            )
            bundle = knowledge_bundle.recall_bundle
        else:
            bundle = self.recall_service.recall(
                retrieval_query,
                ExtractedKeywords(normalized_question=retrieval_query),
                list(history_rows or []),
                knowledge_context,
                session.merchant_id,
                categories,
            )
        if not isinstance(bundle, RecallBundle):
            bundle = RecallBundle.model_validate(bundle)
        if query:
            bundle = _merge_recall_bundles(session.recall, bundle)
        with self._lock:
            session.recall = bundle
            if knowledge_bundle is not None:
                session.recall_rounds = [
                    *session.recall_rounds,
                    *[
                        item.model_copy(deep=True)
                        for item in knowledge_bundle.recall_rounds
                    ],
                ][-32:]
                session.recall_index_version = str(knowledge_bundle.index_version or "")
                session.recall_semantic_source_hash = str(
                    knowledge_bundle.semantic_source_hash or ""
                )
                session.recall_retrieval_status = str(
                    knowledge_bundle.retrieval_status or "not_started"
                )
                session.recall_retrieval_issues = [
                    item.model_copy(deep=True)
                    for item in knowledge_bundle.retrieval_issues
                ][-24:]
            session.phase = "NAVIGATION_RECALLED"
            stop_reason = (
                str(knowledge_bundle.recall_rounds[-1].retrieval_stop_reason or "")
                if knowledge_bundle is not None and knowledge_bundle.recall_rounds
                else ""
            )
            self._event(
                session,
                "recall_navigation",
                "OK",
                "query=%s;items=%d;stop=%s"
                % (retrieval_query[:120], len(bundle.items), stop_reason or "unknown"),
            )
        return bundle

    def recall_goal_gaps(
        self,
        session: GroundedRuntimeSession,
        requests: Sequence[Mapping[str, Any]],
        *,
        max_requests: int = 4,
    ) -> list[dict[str, Any]]:
        """Run bounded Goal-targeted recall without selecting semantic refs."""

        results: list[dict[str, Any]] = []
        for raw in list(requests)[: max(0, int(max_requests or 0))]:
            request_id = str(
                raw.get("requestId") or raw.get("request_id") or ""
            ).strip()
            fingerprint = str(
                raw.get("requestFingerprint")
                or raw.get("request_fingerprint")
                or ""
            ).strip()
            try:
                bundle = self.recall_navigation(
                    session,
                    query=str(raw.get("query") or "").strip(),
                    target_goal_ids=list(
                        raw.get("targetGoalIds")
                        or raw.get("target_goal_ids")
                        or []
                    ),
                    required_capabilities=list(
                        raw.get("requiredCapabilities")
                        or raw.get("required_capabilities")
                        or []
                    ),
                    coverage_receipt_id=str(
                        raw.get("coverageReceiptId")
                        or raw.get("coverage_receipt_id")
                        or ""
                    ),
                    strict_topic_scope=False,
                )
            except Exception as exc:
                results.append(
                    {
                        "requestId": request_id,
                        "requestFingerprint": fingerprint,
                        "status": "DEGRADED",
                        "code": "GOAL_SUPPLEMENTAL_RECALL_FAILED",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                    }
                )
                continue
            results.append(
                {
                    "requestId": request_id,
                    "requestFingerprint": fingerprint,
                    "status": "COMPLETED",
                    "candidateCount": len(bundle.items),
                    "recallIndexVersion": session.recall_index_version,
                }
            )
        return results

    def propose_contract(
        self,
        session: GroundedRuntimeSession,
        core_semantic_evidence: Iterable[dict[str, Any]],
        binding_hints: dict[str, Any] | GroundedBindingHints | None = None,
        *,
        topics: Sequence[str] | None = None,
        timezone_name: str = "Asia/Shanghai",
        now: datetime | None = None,
        default_days: int = 7,
    ) -> GroundedRuntimeAttempt:
        with self._lock:
            if session.terminal_guard_code:
                raise RuntimeError(
                    "TERMINAL_GUARD:%s" % session.terminal_guard_code
                )
        selected_topics = _dedupe(topics or session.workspace_topics)
        normalized_hints = (
            binding_hints
            if isinstance(binding_hints, GroundedBindingHints)
            else GroundedBindingHints.model_validate(binding_hints or {})
        )
        (
            normalized_hints,
            upstream_bindings,
            upstream_gaps,
        ) = self._resolve_upstream_entity_hints(session, normalized_hints)
        contract = self.contract_builder.build(
            session.question,
            selected_topics,
            list(core_semantic_evidence),
            binding_hints=normalized_hints,
            timezone_name=timezone_name,
            now=now,
            default_days=default_days,
        )
        if session.upstream_artifact_bindings:
            query_calculation_graph, output_node_by_ref = (
                compose_query_calculation_graph(
                    contract.metrics,
                    contract.internal_metrics,
                    session.upstream_artifact_bindings,
                )
            )
            contract = contract.model_copy(
                update={
                    "upstream_artifact_bindings": [
                        item.model_copy(deep=True)
                        for item in session.upstream_artifact_bindings
                    ],
                    "calculation_graph": query_calculation_graph,
                    "requested_outputs": [
                        item.model_copy(
                            update={
                                "calculation_node_id": output_node_by_ref.get(
                                    item.semantic_ref_id,
                                    item.calculation_node_id,
                                )
                            },
                            deep=True,
                        )
                        for item in contract.requested_outputs
                    ],
                },
                deep=True,
            )
            validator = getattr(self.contract_builder, "validator", None)
            if validator is not None and callable(
                getattr(validator, "validate", None)
            ):
                validation = validator.validate(contract)
                combined_gaps = _dedupe_contract_gaps(
                    [
                        *contract.unresolved_gaps,
                        *validation.gaps,
                    ]
                )
                contract = contract.model_copy(
                    update={
                        "status": (
                            "UNRESOLVED"
                            if any(gap.blocking for gap in combined_gaps)
                            else "READY"
                        ),
                        "unresolved_gaps": combined_gaps,
                    },
                    deep=True,
                )
        if session.reference_scope.enabled:
            contract = self._attach_reference_scope(
                contract,
                session.reference_scope,
            )
        if upstream_bindings or upstream_gaps:
            resolved_bindings, identity_gaps = self._bind_upstream_targets(
                contract,
                upstream_bindings,
            )
            combined_gaps = [
                *contract.unresolved_gaps,
                *upstream_gaps,
                *identity_gaps,
            ]
            contract = contract.model_copy(
                update={
                    "status": (
                        "UNRESOLVED"
                        if any(item.blocking for item in combined_gaps)
                        else contract.status
                    ),
                    "upstream_entity_bindings": resolved_bindings,
                    "unresolved_gaps": combined_gaps,
                },
                deep=True,
            )
        attempt_id = "attempt_%s" % uuid.uuid4().hex[:12]
        with self._lock:
            parent_attempt = session.attempts[-1] if session.attempts else None
            parent_attempt_id = str(
                getattr(parent_attempt, "attempt_id", "") or ""
            )
            parent_contract_fingerprint = (
                grounded_query_contract_fingerprint(parent_attempt.contract)
                if parent_attempt is not None
                else ""
            )
            contract_version = (
                int(getattr(parent_attempt, "contract_version", 0) or 0) + 1
                if parent_attempt is not None
                else 1
            )
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        repair_directive = build_contract_repair_directive(
            contract.unresolved_gaps,
            contract_status=contract.status,
            base_attempt_id=attempt_id,
            base_contract_fingerprint=contract_fingerprint,
            contract_version=contract_version,
            parent_attempt_id=parent_attempt_id,
            parent_contract_fingerprint=parent_contract_fingerprint,
        )
        attempt = GroundedRuntimeAttempt(
            attempt_id=attempt_id,
            contract=contract,
            contract_version=contract_version,
            parent_attempt_id=parent_attempt_id,
            parent_contract_fingerprint=parent_contract_fingerprint,
            repair_type=repair_directive.repair_type,
            repair_directive=repair_directive.model_dump(
                by_alias=True,
                mode="json",
            ),
            status=contract.status,
            validation_gaps=[
                gap.model_dump(by_alias=True, mode="json")
                for gap in contract.unresolved_gaps
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._apply_execution_policy(attempt)
        if repair_directive.repair_type != "NONE":
            attempt.next_action = repair_directive.next_action
        repair_phase = (
            "QUERY_REPAIR_REQUIRED"
            if repair_directive.repair_type != "NONE"
            else "CONTRACT_PROPOSED"
        )
        with self._lock:
            session.attempts.append(attempt)
            session.phase = repair_phase
            self._event(
                session,
                "propose_contract",
                contract.status,
                ",".join(gap.code for gap in contract.unresolved_gaps if gap.blocking),
                attempt.attempt_id,
            )
        return attempt

    def _attach_reference_scope(
        self,
        contract: GroundedQueryContract,
        reference_scope: GroundedReferenceScopeBinding,
    ) -> GroundedQueryContract:
        """Attach server-owned lineage and re-run Contract validation.

        Core never authors or edits this object.  It is copied from the
        session after the persisted source artifact and fingerprint have been
        checked by the application runtime.
        """

        updated = contract.model_copy(
            update={"reference_scope": reference_scope.model_copy(deep=True)},
            deep=True,
        )
        validation = self.contract_builder.validator.validate(updated)
        combined: list[GroundedContractGap] = []
        seen: set[tuple[str, str, str, str]] = set()
        for gap in [*updated.unresolved_gaps, *validation.gaps]:
            identity = (gap.code, gap.topic, gap.table, gap.phrase)
            if identity in seen:
                continue
            seen.add(identity)
            combined.append(gap)
        return updated.model_copy(
            update={
                "status": (
                    "UNRESOLVED"
                    if any(item.blocking for item in combined)
                    else "READY"
                ),
                "unresolved_gaps": combined,
            },
            deep=True,
        )

    def _resolve_upstream_entity_hints(
        self,
        session: GroundedRuntimeSession,
        hints: GroundedBindingHints,
    ) -> tuple[
        GroundedBindingHints,
        list[GroundedUpstreamEntityBinding],
        list[GroundedContractGap],
    ]:
        """Resolve artifact references inside the kernel, never in Core context."""

        if not hints.upstream_entity_bindings:
            return hints, [], []
        with self._lock:
            entity_sets = {
                item.artifact_id: item.model_copy(deep=True)
                for item in session.verified_entity_sets
            }
            query_artifacts = {
                item.artifact_id: item.model_copy(deep=True)
                for item in session.verified_query_ledger
            }
        filters = [item.model_copy(deep=True) for item in hints.entity_filters]
        resolved: list[GroundedUpstreamEntityBinding] = []
        gaps: list[GroundedContractGap] = []
        occupied_targets = {item.field_ref for item in filters}
        for item in hints.upstream_entity_bindings:
            artifact_id = str(item.entity_set_artifact_id or "").strip()
            target_ref = str(item.target_field_ref or "").strip()
            entity_set = entity_sets.get(artifact_id)
            if entity_set is None:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_SET_NOT_FOUND",
                        message="Verified entity set %s does not exist" % artifact_id,
                        evidence_kind="VERIFIED_ENTITY_SET",
                        phrase=item.requested_phrase,
                        resolution="Publish an entity set from a verified query artifact first.",
                    )
                )
                continue
            source_query = query_artifacts.get(entity_set.source_query_artifact_id)
            if source_query is None or not source_query.verified_evidence.passed:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_QUERY_EVIDENCE_NOT_VERIFIED",
                        message="Entity set source query is not present in the verified ledger",
                        evidence_kind="VERIFIED_QUERY_ARTIFACT",
                        phrase=item.requested_phrase,
                        resolution="Use only an entity set retained by the active verified ledger.",
                    )
                )
                continue
            if entity_set.truncated:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_SET_TRUNCATED",
                        message="A truncated entity set cannot authorize a complete downstream query",
                        evidence_kind="VERIFIED_ENTITY_SET",
                        phrase=item.requested_phrase,
                        resolution="Publish a bounded complete entity set or revise the query strategy.",
                    )
                )
                continue
            if not entity_set.values:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_SET_EMPTY",
                        message="Verified entity set contains no usable values",
                        evidence_kind="VERIFIED_ENTITY_SET",
                        phrase=item.requested_phrase,
                        resolution="Treat the empty upstream result as verified final evidence; do not query an unrelated entity.",
                    )
                )
                continue
            operator = str(item.operator or "IN").strip().upper()
            if operator != "IN":
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_OPERATOR_INVALID",
                        message="Verified entity sets use the stable typed IN protocol",
                        evidence_kind="VERIFIED_ENTITY_SET",
                        phrase=item.requested_phrase,
                    )
                )
                continue
            if not target_ref:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_TARGET_REQUIRED",
                        message="Verified entity binding requires a target semantic field ref",
                        evidence_kind="COLUMN",
                        phrase=item.requested_phrase,
                    )
                )
                continue
            if target_ref in occupied_targets:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_FILTER_CONFLICT",
                        message="Target field already has another entity filter binding",
                        evidence_kind="COLUMN",
                        phrase=item.requested_phrase,
                        rejected_ref_ids=[target_ref],
                    )
                )
                continue
            occupied_targets.add(target_ref)
            literal_value: Any = list(entity_set.values)
            filters.append(
                GroundedEntityFilterHint(
                    field_ref=target_ref,
                    operator=operator,
                    literal_value=literal_value,
                    requested_phrase=(
                        item.requested_phrase
                        or "verified entity set %s" % artifact_id
                    ),
                )
            )
            resolved.append(
                GroundedUpstreamEntityBinding(
                    entity_set_artifact_id=artifact_id,
                    source_query_artifact_id=entity_set.source_query_artifact_id,
                    source_contract_fingerprint=source_query.contract_fingerprint,
                    source_sql_fingerprint=source_query.sql_fingerprint,
                    source_column=entity_set.source_column,
                    source_semantic_ref_id=entity_set.source_semantic_ref_id,
                    source_entity_identity=entity_set.source_entity_identity,
                    target_field_ref=target_ref,
                    operator=operator,
                    value_count=entity_set.value_count,
                    values_hash=entity_set.values_hash,
                    requested_phrase=item.requested_phrase,
                )
            )
        return hints.model_copy(update={"entity_filters": filters}), resolved, gaps

    @staticmethod
    def _bind_upstream_targets(
        contract: GroundedQueryContract,
        bindings: Sequence[GroundedUpstreamEntityBinding],
    ) -> tuple[list[GroundedUpstreamEntityBinding], list[GroundedContractGap]]:
        by_ref = {item.semantic_ref_id: item for item in contract.entity_filters}
        resolved: list[GroundedUpstreamEntityBinding] = []
        gaps: list[GroundedContractGap] = []
        for item in bindings:
            target = by_ref.get(item.target_field_ref)
            if target is None:
                resolved.append(item)
                continue
            bound = item.model_copy(
                update={
                    "target_table": target.table,
                    "target_column": target.column,
                    "target_entity_identity": target.entity_identity,
                }
            )
            resolved.append(bound)
            source_identity = str(item.source_entity_identity or "").strip()
            target_identity = str(target.entity_identity or "").strip()
            if not source_identity or not target_identity:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_IDENTITY_UNDECLARED",
                        message="Both source and target fields must declare a canonical entity identity",
                        evidence_kind="COLUMN",
                        table=target.table,
                        phrase=item.requested_phrase,
                        resolution="Publish canonicalEntityRef/entityIdentity in the semantic field assets.",
                        rejected_ref_ids=[item.source_semantic_ref_id, item.target_field_ref],
                    )
                )
            elif source_identity != target_identity:
                gaps.append(
                    GroundedContractGap(
                        code="UPSTREAM_ENTITY_IDENTITY_MISMATCH",
                        message="Verified source entity identity does not match the downstream target field",
                        evidence_kind="COLUMN",
                        table=target.table,
                        phrase=item.requested_phrase,
                        required_capability={
                            "sourceEntityIdentity": source_identity,
                            "targetEntityIdentity": target_identity,
                        },
                        rejected_ref_ids=[item.source_semantic_ref_id, item.target_field_ref],
                    )
                )
        return resolved, gaps

    def activate_contract(
        self,
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        """Activate a READY candidate according to its execution authority.

        This method invokes deterministic compilation only for modes admitted
        by the capability gate in ``DETERMINISTIC_EXECUTION_MODES``.
        ``CORE_SQL_REQUIRED`` activates the Contract as the allowed semantic
        scope without materializing or compiling a template query.
        """

        with self._lock:
            attempt = self._attempt(session, attempt_id)
            contract = attempt.contract.model_copy(deep=True)
            self._apply_execution_policy(attempt)
            if not contract.ready or contract.status != "READY":
                attempt.compile_status = "SKIPPED_NOT_READY"
                attempt.activation_status = "SKIPPED_NOT_READY"
                attempt.error = "candidate Contract is not READY"
                self._event(
                    session,
                    "compile_candidate",
                    attempt.compile_status,
                    attempt.error,
                    attempt_id,
                )
                return attempt
            contract_fingerprint = grounded_query_contract_fingerprint(contract)
            active_contract_fingerprint = (
                grounded_query_contract_fingerprint(session.active_contract)
                if session.active_contract is not None
                else ""
            )
            if (
                session.active_generation > 0
                and active_contract_fingerprint == contract_fingerprint
            ):
                attempt.compile_status = "NOT_APPLICABLE_CONTRACT_UNCHANGED"
                attempt.activation_status = "ACTIVE_CONTRACT_REUSED"
                attempt.activated = True
                attempt.active_generation = session.active_generation
                if (
                    contract_fingerprint
                    in session.repair_exhausted_contract_fingerprints
                ):
                    attempt.next_action = "REVISE_BINDINGS_WITH_CHANGED_EVIDENCE"
                elif session.active_preparation is not None:
                    attempt.next_action = "EXECUTE_GROUNDED_QUERY"
                elif (
                    attempt.execution_mode
                    == GroundedExecutionMode.CORE_SQL_REQUIRED
                ):
                    attempt.next_action = "SUBMIT_GROUNDED_SQL_CANDIDATE"
                else:
                    attempt.next_action = "EXECUTE_GROUNDED_QUERY"
                self._event(
                    session,
                    "activate_contract",
                    attempt.activation_status,
                    "generation=%d;contract_unchanged" % session.active_generation,
                    attempt_id,
                )
                return attempt
            if attempt.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED:
                next_generation = session.active_generation + 1
                self._activate_scope(
                    session,
                    attempt=attempt,
                    contract=contract,
                    generation=next_generation,
                )
                attempt.compile_status = "NOT_APPLICABLE_CORE_SQL_REQUIRED"
                attempt.activation_status = "ACTIVATED"
                attempt.activated = True
                attempt.active_generation = next_generation
                attempt.next_action = "SUBMIT_GROUNDED_SQL_CANDIDATE"
                self._event(
                    session,
                    "activate_candidate",
                    "CORE_SQL_REQUIRED",
                    "generation=%d;reasons=%s"
                    % (
                        next_generation,
                        ",".join(attempt.fast_path_reason_codes),
                    ),
                    attempt_id,
                )
                return attempt
            if attempt.execution_mode not in DETERMINISTIC_EXECUTION_MODES:
                attempt.compile_status = "EXECUTION_MODE_UNDECIDED"
                attempt.activation_status = "NOT_ACTIVATED"
                attempt.error = "READY Contract has no permitted execution mode"
                self._event(
                    session,
                    "activate_candidate",
                    attempt.compile_status,
                    attempt.error,
                    attempt_id,
                )
                return attempt
            try:
                candidate_pack = self.asset_materializer(contract, self.topic_assets)
                candidate_preparation = self.compiler(contract, candidate_pack)
            except Exception as exc:
                attempt.compile_status = "COMPILE_FAILED"
                attempt.activation_status = "NOT_ACTIVATED"
                attempt.error = str(exc)
                self._event(
                    session,
                    "compile_candidate",
                    attempt.compile_status,
                    attempt.error,
                    attempt_id,
                )
                return attempt

            validation = getattr(candidate_preparation, "validation", None)
            if validation is None or not bool(getattr(validation, "valid", False)):
                attempt.compile_status = "VALIDATION_FAILED"
                attempt.activation_status = "NOT_ACTIVATED"
                attempt.validation_gaps = [
                    item.model_dump(by_alias=True)
                    if hasattr(item, "model_dump")
                    else dict(item)
                    if isinstance(item, dict)
                    else {"reason": str(item)}
                    for item in list(getattr(validation, "gaps", []) or [])
                ]
                attempt.error = "candidate execution preparation is invalid"
                self._event(
                    session,
                    "compile_candidate",
                    attempt.compile_status,
                    attempt.error,
                    attempt_id,
                )
                return attempt

            candidate_plan = getattr(candidate_preparation, "plan", None)
            if not isinstance(candidate_plan, QueryPlan):
                attempt.compile_status = "COMPILE_FAILED"
                attempt.activation_status = "NOT_ACTIVATED"
                attempt.error = "compiler did not return a typed QueryPlan preparation"
                self._event(
                    session,
                    "compile_candidate",
                    attempt.compile_status,
                    attempt.error,
                    attempt_id,
                )
                return attempt

            # Atomic active-generation switch: no active field changes before
            # every candidate artifact has passed its own validation.
            next_generation = session.active_generation + 1
            session.active_contract = contract
            session.active_execution_mode = attempt.execution_mode
            session.active_execution_reason_codes = list(
                attempt.execution_reason_codes
            )
            session.active_pack = candidate_pack
            session.active_plan = candidate_plan.model_copy(deep=True)
            session.active_preparation = candidate_preparation
            session.active_sql_candidate = None
            session.active_sql_validation = None
            session.sql_execution_repair_context = {}
            session.active_attempt_id = attempt_id
            session.active_generation = next_generation
            session.run_result = None
            self._reset_publication_state(session)
            session.verified_evidence = None
            self._clear_answer_snapshot(session)
            session.clarification = None
            session.phase = "ACTIVE_COMPILED"
            session.revision += 1
            attempt.compile_status = "VALID"
            attempt.activation_status = "ACTIVATED"
            attempt.activated = True
            attempt.active_generation = next_generation
            attempt.next_action = "EXECUTE_GROUNDED_QUERY"
            self._event(
                session,
                "compile_candidate",
                "ACTIVATED",
                "generation=%d" % next_generation,
                attempt_id,
            )
            return attempt

    def submit_sql_candidate(
        self,
        session: GroundedRuntimeSession,
        sql: str,
        *,
        expected_generation: int,
        expected_contract_fingerprint: str,
        rationale: str = "",
        evidence_refs: Sequence[str] | None = None,
    ) -> GroundedRuntimeSqlCandidateAttempt:
        """Validate and atomically activate one complete SQL authored by Core."""

        with self._lock:
            generation = int(session.active_generation or 0)
            if session.terminal_guard_code:
                raise RuntimeError(
                    "TERMINAL_GUARD:%s" % session.terminal_guard_code
                )
            if (
                generation <= 0
                or session.active_execution_mode
                != GroundedExecutionMode.CORE_SQL_REQUIRED
                or session.active_contract is None
            ):
                raise RuntimeError(
                    "CORE_SQL_NOT_AUTHORIZED: activate a READY complex Contract first"
                )
            contract = session.active_contract.model_copy(deep=True)
            contract_fingerprint = grounded_query_contract_fingerprint(contract)
            if (
                int(expected_generation or 0) != generation
                or str(expected_contract_fingerprint or "").strip()
                != contract_fingerprint
            ):
                raise RuntimeError(
                    "SQL_CANDIDATE_STALE_CONTRACT: expected generation/fingerprint "
                    "does not match the active Contract"
                )
            if (
                contract_fingerprint
                in session.repair_exhausted_contract_fingerprints
            ):
                existing = next(
                    (
                        item
                        for item in reversed(session.sql_candidate_attempts)
                        if item.active_generation == generation
                        and item.status == "REPAIR_EXHAUSTED"
                    ),
                    None,
                )
                if existing is not None:
                    return existing.model_copy(deep=True)
                raise RuntimeError(
                    "SQL_CANDIDATE_REPAIR_EXHAUSTED: revise bindings with changed evidence"
                )
            prior_attempts = [
                item
                for item in session.sql_candidate_attempts
                if item.active_generation == generation
            ]
        parent_candidate = prior_attempts[-1] if prior_attempts else None
        candidate_version = (
            max(int(item.candidate_version or 0) for item in prior_attempts) + 1
            if prior_attempts
            else 1
        )
        parent_candidate_id = str(
            getattr(parent_candidate, "candidate_id", "") or ""
        )
        parent_ast_fingerprint = str(
            getattr(parent_candidate, "ast_fingerprint", "") or ""
        )
        if len(prior_attempts) >= 3:
            exhausted = GroundedRuntimeSqlCandidateAttempt(
                candidate_id="sql_%s" % uuid.uuid4().hex[:12],
                active_generation=generation,
                candidate_version=candidate_version,
                parent_candidate_id=parent_candidate_id,
                parent_ast_fingerprint=parent_ast_fingerprint,
                status="REPAIR_EXHAUSTED",
                next_action="REVISE_BINDINGS",
                validation_gaps=[
                    {
                        "code": "SQL_CANDIDATE_REPAIR_EXHAUSTED",
                        "message": (
                            "Initial SQL plus two Core repair attempts were consumed for "
                            "this Contract generation"
                        ),
                        "blocking": True,
                        "resolution": (
                            "Revise the grounded bindings before authoring another SQL candidate."
                        ),
                    }
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._require_generation(session, generation)
                session.sql_candidate_attempts.append(exhausted)
                if (
                    contract_fingerprint
                    not in session.repair_exhausted_contract_fingerprints
                ):
                    session.repair_exhausted_contract_fingerprints.append(
                        contract_fingerprint
                    )
                self._clear_active_sql_candidate(session)
                session.phase = "CORE_SQL_REPAIR_EXHAUSTED"
                session.revision += 1
                self._event(
                    session,
                    "submit_sql_candidate",
                    exhausted.status,
                    "generation=%d" % generation,
                    session.active_attempt_id,
                )
            return exhausted

        trusted_refs = set(contract.evidence_refs)
        candidate = GroundedSqlCandidate(
            sql=str(sql or ""),
            contract_fingerprint=contract_fingerprint,
            evidence_refs=[
                ref
                for ref in _dedupe(evidence_refs or [])
                if ref in trusted_refs
            ],
            rationale=str(rationale or "")[:2000],
        )
        try:
            validation = self.sql_candidate_validator.validate(candidate, contract)
        except Exception as exc:
            failed = GroundedRuntimeSqlCandidateAttempt(
                candidate_id="sql_%s" % uuid.uuid4().hex[:12],
                active_generation=generation,
                candidate_version=candidate_version,
                parent_candidate_id=parent_candidate_id,
                parent_ast_fingerprint=parent_ast_fingerprint,
                status="VALIDATOR_INTERNAL_ERROR",
                next_action="STOP_INTERNAL",
                contract_fingerprint=contract_fingerprint,
                validation_gaps=[
                    {
                        "code": "SQL_CANDIDATE_VALIDATOR_INTERNAL_ERROR",
                        "message": "%s:%s"
                        % (type(exc).__name__, str(exc)[:400]),
                        "blocking": True,
                        "resolution": (
                            "Stop this execution attempt; validator failures are not repairable by changing business bindings."
                        ),
                    }
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._require_generation(session, generation)
                session.sql_candidate_attempts.append(failed)
                self._clear_active_sql_candidate(session)
                session.phase = "CORE_SQL_VALIDATOR_INTERNAL_ERROR"
                session.revision += 1
                self._event(
                    session,
                    "submit_sql_candidate",
                    failed.status,
                    "SQL_CANDIDATE_VALIDATOR_INTERNAL_ERROR",
                    session.active_attempt_id,
                )
            return failed
        leading_code = next(
            (gap.code for gap in validation.gaps if gap.blocking),
            "VALID",
        )
        sql_identity = validation.ast_fingerprint or hashlib.sha256(
            str(sql or "").strip().encode("utf-8")
        ).hexdigest()
        progress_fingerprint = hashlib.sha256(
            (
                "%s:%s:%s"
                % (
                    contract_fingerprint,
                    sql_identity,
                    leading_code,
                )
            ).encode("utf-8")
        ).hexdigest()
        duplicate = next(
            (
                item
                for item in prior_attempts
                if item.progress_fingerprint == progress_fingerprint
            ),
            None,
        )
        if duplicate is not None:
            if (
                duplicate.status == "ACCEPTED"
                and isinstance(
                    session.active_preparation,
                    GroundedCoreSqlPreparation,
                )
                and session.active_preparation.candidate_id
                == duplicate.candidate_id
                and session.active_sql_validation is not None
                and session.active_sql_validation.valid
            ):
                return duplicate.model_copy(deep=True)
            no_progress = GroundedRuntimeSqlCandidateAttempt(
                candidate_id="sql_%s" % uuid.uuid4().hex[:12],
                active_generation=generation,
                candidate_version=candidate_version,
                parent_candidate_id=parent_candidate_id,
                parent_ast_fingerprint=parent_ast_fingerprint,
                status="NO_PROGRESS",
                next_action=(
                    "REVISE_BINDINGS"
                    if duplicate.next_action == "REVISE_BINDINGS"
                    else "REPAIR_SQL_OR_BINDINGS"
                    if duplicate.next_action == "REPAIR_SQL_OR_BINDINGS"
                    else "REPAIR_SQL"
                ),
                ast_fingerprint=validation.ast_fingerprint,
                contract_fingerprint=contract_fingerprint,
                progress_fingerprint=progress_fingerprint,
                validation_gaps=[
                    {
                        "code": "SQL_CANDIDATE_NO_PROGRESS",
                        "message": (
                            "Core resubmitted the same canonical SQL with the same validation state"
                        ),
                        "blocking": True,
                        "resolution": (
                            "Change the SQL AST or revise semantic bindings; do not retry the same candidate."
                        ),
                    }
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._require_generation(session, generation)
                session.sql_candidate_attempts.append(no_progress)
                if progress_fingerprint not in session.rejected_sql_candidate_fingerprints:
                    session.rejected_sql_candidate_fingerprints.append(
                        progress_fingerprint
                    )
                self._clear_active_sql_candidate(session)
                session.phase = (
                    "CORE_BINDING_REPAIR_REQUIRED"
                    if no_progress.next_action == "REVISE_BINDINGS"
                    else "CORE_QUERY_REPAIR_REQUIRED"
                    if no_progress.next_action == "REPAIR_SQL_OR_BINDINGS"
                    else "CORE_SQL_NO_PROGRESS"
                )
                session.revision += 1
                self._event(
                    session,
                    "submit_sql_candidate",
                    no_progress.status,
                    leading_code,
                    session.active_attempt_id,
                )
            return no_progress

        next_action = self._sql_candidate_next_action(validation)
        attempt = GroundedRuntimeSqlCandidateAttempt(
            candidate_id="sql_%s" % uuid.uuid4().hex[:12],
            active_generation=generation,
            candidate_version=candidate_version,
            parent_candidate_id=parent_candidate_id,
            parent_ast_fingerprint=parent_ast_fingerprint,
            status="ACCEPTED" if validation.valid else "REJECTED",
            next_action=(
                "EXECUTE_GROUNDED_QUERY" if validation.valid else next_action
            ),
            ast_fingerprint=validation.ast_fingerprint,
            contract_fingerprint=contract_fingerprint,
            progress_fingerprint=progress_fingerprint,
            output_columns=list(validation.output_columns),
            validation_gaps=[
                gap.model_dump(by_alias=True) for gap in validation.gaps
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        if not validation.valid:
            with self._lock:
                self._require_generation(session, generation)
                session.sql_candidate_attempts.append(attempt)
                session.rejected_sql_candidate_fingerprints.append(
                    progress_fingerprint
                )
                self._clear_active_sql_candidate(session)
                session.phase = (
                    "CORE_BINDING_REPAIR_REQUIRED"
                    if next_action == "REVISE_BINDINGS"
                    else "CORE_QUERY_REPAIR_REQUIRED"
                    if next_action == "REPAIR_SQL_OR_BINDINGS"
                    else "CORE_SQL_REPAIR_REQUIRED"
                )
                session.revision += 1
                self._event(
                    session,
                    "submit_sql_candidate",
                    attempt.status,
                    leading_code,
                    session.active_attempt_id,
                )
            return attempt

        try:
            candidate_pack = self.asset_materializer(contract, self.topic_assets)
            candidate_plan = self._build_core_sql_evidence_plan(
                contract,
                validation,
            )
            preparation = GroundedCoreSqlPreparation(
                plan=candidate_plan,
                candidate_id=attempt.candidate_id,
                sql_candidate=candidate,
                candidate_validation=validation,
                asset_pack_fingerprint=_stable_json_hash(
                    candidate_pack.model_dump(by_alias=True, mode="json")
                ),
            )
        except Exception as exc:
            attempt.status = "PREPARATION_FAILED"
            attempt.next_action = "REVISE_BINDINGS"
            attempt.validation_gaps.append(
                {
                    "code": "SQL_CANDIDATE_PREPARATION_FAILED",
                    "message": str(exc)[:500],
                    "blocking": True,
                    "resolution": (
                        "Repair the progressively disclosed semantic assets or Contract bindings."
                    ),
                }
            )
            with self._lock:
                self._require_generation(session, generation)
                session.sql_candidate_attempts.append(attempt)
                session.rejected_sql_candidate_fingerprints.append(
                    progress_fingerprint
                )
                self._clear_active_sql_candidate(session)
                session.phase = "CORE_SQL_PREPARATION_FAILED"
                session.revision += 1
            return attempt

        with self._lock:
            self._require_generation(session, generation)
            session.sql_candidate_attempts.append(attempt)
            session.active_pack = candidate_pack
            session.active_plan = candidate_plan.model_copy(deep=True)
            session.active_preparation = preparation
            session.active_sql_candidate = candidate
            session.active_sql_validation = validation
            session.run_result = None
            self._reset_publication_state(session)
            session.verified_evidence = None
            self._clear_answer_snapshot(session)
            session.phase = "ACTIVE_CORE_SQL_VALIDATED"
            session.revision += 1
            self._event(
                session,
                "submit_sql_candidate",
                attempt.status,
                "outputs=%s" % ",".join(validation.output_columns),
                session.active_attempt_id,
            )
        return attempt

    def semantic_activation_authority_available(self) -> bool:
        return callable(
            getattr(self.topic_assets, "semantic_source_hash", None)
        )

    def seal_semantic_activation(
        self,
        session: GroundedRuntimeSession,
        topics: Sequence[str],
        *,
        allow_topic_expansion: bool = False,
    ) -> GroundedSemanticActivationSeal | None:
        """Seal the exact governed Topic source set with optimistic CAS.

        A missing provider is supported only for narrow injected test kernels.
        The production ``TopicAssetService`` always provides this authority.
        Once any query has started or verified evidence exists, neither the
        Topic set nor a changed source digest may be silently re-sealed.
        """

        source_hash = getattr(
            self.topic_assets,
            "semantic_source_hash",
            None,
        )
        if not callable(source_hash):
            return None
        requested_topics = canonical_semantic_topics(topics)
        if not requested_topics:
            raise RuntimeError("SEMANTIC_ACTIVATION_TOPIC_SET_REQUIRED")
        active_topic_names = getattr(
            self.topic_assets,
            "all_topic_names",
            None,
        )
        if callable(active_topic_names):
            active_topics = set(
                canonical_semantic_topics(active_topic_names())
            )
            missing_topics = [
                topic
                for topic in requested_topics
                if topic not in active_topics
            ]
            if missing_topics:
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_TOPIC_NOT_ACTIVE:%s"
                    % ",".join(missing_topics)
                )

        with self._lock:
            existing = (
                session.semantic_activation_seal.model_copy(deep=True)
                if session.semantic_activation_seal is not None
                else None
            )
            if existing is not None and not semantic_activation_seal_valid(
                existing
            ):
                raise RuntimeError("SEMANTIC_ACTIVATION_SEAL_CORRUPT")
            existing_topics = set(
                existing.exact_topics if existing is not None else []
            )
            added_topics = [
                topic
                for topic in requested_topics
                if topic not in existing_topics
            ]
            if added_topics and existing is not None:
                if not allow_topic_expansion:
                    raise RuntimeError(
                        "SEMANTIC_ACTIVATION_TOPIC_EXPANSION_REQUIRES_RESEAL"
                    )
                if (
                    session.semantic_activation_execution_started
                    or session.verified_query_ledger
                ):
                    raise RuntimeError(
                        "SEMANTIC_ACTIVATION_TOPIC_EXPANSION_AFTER_EXECUTION_FORBIDDEN"
                    )
            target_topics = canonical_semantic_topics(
                [
                    *(existing.exact_topics if existing is not None else []),
                    *requested_topics,
                ]
            )
            cas_seal_fingerprint = (
                existing.seal_fingerprint if existing is not None else ""
            )
            cas_execution_started = bool(
                session.semantic_activation_execution_started
            )
            cas_ledger_ids = tuple(
                item.artifact_id
                for item in session.verified_query_ledger
            )

        if existing is not None:
            observed_existing_fingerprint = str(
                source_hash(existing.exact_topics) or ""
            ).strip()
            if not valid_semantic_activation_fingerprint(
                observed_existing_fingerprint
            ):
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_REVALIDATION_UNAVAILABLE"
                )
            if (
                observed_existing_fingerprint
                != existing.semantic_activation_fingerprint
            ):
                raise RuntimeError("SEMANTIC_ACTIVATION_STALE")
        observed_source_fingerprint = str(
            source_hash(target_topics) or ""
        ).strip()
        if not valid_semantic_activation_fingerprint(
            observed_source_fingerprint
        ):
            raise RuntimeError(
                "SEMANTIC_ACTIVATION_SOURCE_FINGERPRINT_UNAVAILABLE"
            )
        next_version = (
            int(existing.version) + 1 if existing is not None else 1
        )
        candidate = build_semantic_activation_seal(
            topics=target_topics,
            semantic_activation_fingerprint=(
                observed_source_fingerprint
            ),
            version=next_version,
        )

        with self._lock:
            current = session.semantic_activation_seal
            current_fingerprint = (
                current.seal_fingerprint if current is not None else ""
            )
            current_ledger_ids = tuple(
                item.artifact_id
                for item in session.verified_query_ledger
            )
            if (
                current_fingerprint != cas_seal_fingerprint
                or bool(session.semantic_activation_execution_started)
                != cas_execution_started
                or current_ledger_ids != cas_ledger_ids
            ):
                if (
                    current is not None
                    and semantic_activation_seal_valid(current)
                    and current.exact_topics == target_topics
                    and current.semantic_activation_fingerprint
                    == observed_source_fingerprint
                ):
                    return current.model_copy(deep=True)
                raise RuntimeError("SEMANTIC_ACTIVATION_SEAL_CAS_STALE")
            if (
                existing is not None
                and existing.exact_topics == target_topics
                and existing.semantic_activation_fingerprint
                == observed_source_fingerprint
            ):
                return existing.model_copy(deep=True)
            session.semantic_activation_seal = candidate.model_copy(
                deep=True
            )
            session.revision += 1
            self._event(
                session,
                "seal_semantic_activation",
                "SEALED",
                "version=%d;topics=%d;activation=%s"
                % (
                    candidate.version,
                    len(candidate.exact_topics),
                    candidate.semantic_activation_fingerprint[:16],
                ),
                session.active_attempt_id,
            )
            return candidate.model_copy(deep=True)

    def revalidate_semantic_activation(
        self,
        session: GroundedRuntimeSession,
    ) -> GroundedSemanticActivationSeal | None:
        source_hash = getattr(
            self.topic_assets,
            "semantic_source_hash",
            None,
        )
        if not callable(source_hash):
            return None
        with self._lock:
            seal = (
                session.semantic_activation_seal.model_copy(deep=True)
                if session.semantic_activation_seal is not None
                else None
            )
        if seal is None:
            raise RuntimeError("SEMANTIC_ACTIVATION_SEAL_REQUIRED")
        if not semantic_activation_seal_valid(seal):
            raise RuntimeError("SEMANTIC_ACTIVATION_SEAL_CORRUPT")
        observed = str(
            source_hash(seal.exact_topics) or ""
        ).strip()
        if not valid_semantic_activation_fingerprint(observed):
            raise RuntimeError(
                "SEMANTIC_ACTIVATION_REVALIDATION_UNAVAILABLE"
            )
        if observed != seal.semantic_activation_fingerprint:
            raise RuntimeError("SEMANTIC_ACTIVATION_STALE")
        with self._lock:
            current = session.semantic_activation_seal
            if (
                current is None
                or current.seal_fingerprint != seal.seal_fingerprint
            ):
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_SEAL_CAS_STALE"
                )
        return seal

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        semantic_fingerprint = str(
            semantic_activation_fingerprint or ""
        ).strip()
        if self.executor is None:
            return DataSnapshotContract(
                semantic_activation_fingerprint=semantic_fingerprint,
                unsupported_reason="GROUNDED_EXECUTOR_NOT_CONFIGURED"
            )
        capture = getattr(self.executor, "capture_data_snapshot", None)
        if not callable(capture):
            return DataSnapshotContract(
                semantic_activation_fingerprint=semantic_fingerprint,
                unsupported_reason="DATA_SNAPSHOT_CAPABILITY_UNAVAILABLE"
            )
        snapshot = capture(semantic_fingerprint)
        if not isinstance(snapshot, DataSnapshotContract):
            snapshot = DataSnapshotContract.model_validate(snapshot)
        observed = str(
            snapshot.semantic_activation_fingerprint or ""
        ).strip()
        if observed != semantic_fingerprint:
            raise RuntimeError(
                "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_MISMATCH"
            )
        return snapshot

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        *,
        knowledge_context: str = "",
        run_id: str = "",
        artifact_root: str = "",
        context_owner_fingerprint: str = "",
        runtime_budget: Any = None,
        data_snapshot_contract: DataSnapshotContract | None = None,
        goal_contract_fingerprint: str = "",
        population_pre_execution_reference: (
            PopulationPreExecutionReference | None
        ) = None,
        population_query_node_id: str = "",
    ) -> AgentRunResult:
        with self._lock:
            execution_mode = session.active_execution_mode
            active_preparation = session.active_preparation
            latest_sql_attempt = next(
                (
                    item
                    for item in reversed(session.sql_candidate_attempts)
                    if item.active_generation == session.active_generation
                ),
                None,
            )
            if execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED:
                if (
                    not isinstance(active_preparation, GroundedCoreSqlPreparation)
                    or latest_sql_attempt is None
                    or latest_sql_attempt.status != "ACCEPTED"
                    or latest_sql_attempt.candidate_id
                    != active_preparation.candidate_id
                ):
                    raise RuntimeError(
                        "CORE_SQL_REQUIRED: the latest SQL candidate is not the accepted active candidate"
                    )
            active_sql_execution_fingerprint = (
                "%d:%s"
                % (
                    session.active_generation,
                    session.active_sql_validation.ast_fingerprint,
                )
                if session.active_sql_validation is not None
                else ""
            )
            if (
                execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
                and active_sql_execution_fingerprint
                and active_sql_execution_fingerprint
                in session.executed_sql_candidate_fingerprints
            ):
                raise RuntimeError(
                    "SQL_EXECUTION_NO_PROGRESS: the active Core SQL candidate was already "
                    "executed; submit a changed SQL candidate before retrying"
                )
        if (
            execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
            and active_preparation is None
        ):
            raise RuntimeError(
                "CORE_SQL_REQUIRED: submit and validate a Core-authored SQL "
                "candidate before execution"
            )
        if self.executor is None:
            raise RuntimeError(
                "grounded executor is not configured; refusing to call workflow execution"
            )
        with self._lock:
            generation, plan, pack = self._active_snapshot(session)
            contract = session.active_contract.model_copy(deep=True)
            runtime_preparation = session.active_preparation
        if not bool(getattr(runtime_preparation, "executable", False)):
            raise RuntimeError("active grounded graph failed runtime preparation")
        execute_contract = getattr(self.executor, "execute_contract", None)
        if not callable(execute_contract):
            raise RuntimeError(
                "grounded executor must implement execute_contract; "
                "legacy execute_plan/NodeAgent execution is forbidden"
            )
        semantic_seal = self.seal_semantic_activation(
            session,
            contract.topics or session.workspace_topics,
        )
        if self.semantic_activation_authority_available():
            semantic_seal = self.revalidate_semantic_activation(
                session
            )
            if semantic_seal is None:
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_SEAL_REQUIRED"
                )
        semantic_activation_fingerprint = (
            semantic_seal.semantic_activation_fingerprint
            if semantic_seal is not None
            else ""
        )
        semantic_seal_fingerprint = (
            semantic_seal.seal_fingerprint
            if semantic_seal is not None
            else ""
        )
        normalized_goal_contract_fingerprint = str(
            goal_contract_fingerprint
            or session.active_goal_contract_fingerprint
            or ""
        ).strip()
        if artifact_root and not _valid_sha256_hex(
            normalized_goal_contract_fingerprint
        ):
            raise RuntimeError(
                "QUERY_EXECUTION_GOAL_CONTRACT_FINGERPRINT_INVALID"
            )
        if data_snapshot_contract is not None and semantic_seal is not None:
            if (
                str(
                    data_snapshot_contract.semantic_activation_fingerprint
                    or ""
                ).strip()
                != semantic_activation_fingerprint
            ):
                raise RuntimeError(
                    "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_MISMATCH"
                )
        with self._lock:
            self._require_generation(session, generation)
            current_seal = session.semantic_activation_seal
            if semantic_seal is not None and (
                current_seal is None
                or current_seal.seal_fingerprint
                != semantic_seal_fingerprint
            ):
                raise RuntimeError(
                    "SEMANTIC_ACTIVATION_SEAL_CAS_STALE"
                )
            if not session.semantic_activation_execution_started:
                session.semantic_activation_execution_started = True
                session.revision += 1
                self._event(
                    session,
                    "semantic_activation_execution_gate",
                    "PASSED",
                    "activation=%s"
                    % semantic_activation_fingerprint[:16],
                    session.active_attempt_id,
                )
        execution_kwargs = {
            "run_id": run_id,
            "access_role": session.access_role,
            "user_scope": dict(session.user_scope),
            "execution_reference_scope": (
                session.reference_scope.model_copy(deep=True)
            ),
            "execution_goal_contract_fingerprint": (
                normalized_goal_contract_fingerprint
            ),
            "expected_semantic_activation_fingerprint": (
                semantic_activation_fingerprint
            ),
            "execution_preparation": runtime_preparation,
        }
        if artifact_root:
            execution_kwargs["artifact_root"] = artifact_root
            execution_kwargs["context_owner_fingerprint"] = (
                context_owner_fingerprint
            )
            execution_kwargs["execution_generation"] = generation
            execution_kwargs["execution_attempt_id"] = (
                session.active_attempt_id
            )
        # Keep non-budget callers fully backward-compatible. Grounded tools
        # explicitly pass the one shared run budget so the executor can clamp
        # Doris immediately before the repository call.
        if runtime_budget is not None:
            execution_kwargs["runtime_budget"] = runtime_budget
        if data_snapshot_contract is not None:
            execution_kwargs["data_snapshot_contract"] = data_snapshot_contract
        if population_pre_execution_reference is not None:
            execution_kwargs["population_pre_execution_reference"] = (
                population_pre_execution_reference
            )
            execution_kwargs["population_query_node_id"] = str(
                population_query_node_id or ""
            ).strip()
        run_result = execute_contract(
            session.merchant_id,
            contract,
            runtime_preparation.plan,
            pack,
            session.question,
            **execution_kwargs,
        )
        if not isinstance(run_result, AgentRunResult):
            run_result = AgentRunResult.model_validate(run_result)
        if semantic_seal is not None:
            seen_bundles: set[int] = set()
            for bundle in self._all_result_bundles(run_result):
                if id(bundle) in seen_bundles:
                    continue
                seen_bundles.add(id(bundle))
                observed_semantic_fingerprint = str(
                    bundle.data_snapshot.semantic_activation_fingerprint
                    or ""
                ).strip()
                if (
                    not observed_semantic_fingerprint
                    and data_snapshot_contract is not None
                ):
                    bundle.data_snapshot = (
                        data_snapshot_contract.model_copy(deep=True)
                    )
                    observed_semantic_fingerprint = str(
                        bundle.data_snapshot.semantic_activation_fingerprint
                        or ""
                    ).strip()
                if not observed_semantic_fingerprint and bundle.failed:
                    continue
                if not observed_semantic_fingerprint:
                    raise RuntimeError(
                        "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_REQUIRED"
                    )
                if (
                    observed_semantic_fingerprint
                    != semantic_activation_fingerprint
                ):
                    raise RuntimeError(
                        "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_MISMATCH"
                    )
        pending_receipts = self._extract_pending_result_artifacts(run_result)
        # The executor result becomes the single private full-result
        # authority.  Core receives a separate, configuration-bounded
        # projection, so the session does not retain two full row populations
        # while it waits for independent verification.
        publication_authority = run_result
        publication_authority_fingerprint = (
            _query_run_result_fingerprint(publication_authority)
        )
        core_run_result = self._bounded_run_result_projection(run_result)
        access_terminal_codes = {
            "ACCESS_DENIED",
            "ACL_POLICY_UNAVAILABLE",
            "ACL_SQL_PARSE_FAILED",
            "MERCHANT_SCOPE_DENIED",
            "TABLE_DENIED",
            "TABLE_NOT_ALLOWED",
            "TABLE_ROLE_DENIED",
            "COLUMN_DENIED",
        }
        terminal_code = next(
            (
                validation.error_code
                for item in run_result.task_results
                for validation in item.validation_results
                if validation.error_code in access_terminal_codes
            ),
            "",
        )
        with self._lock:
            self._require_generation(session, generation)
            if semantic_seal is not None:
                current_seal = session.semantic_activation_seal
                if (
                    current_seal is None
                    or current_seal.seal_fingerprint
                    != semantic_seal_fingerprint
                ):
                    raise RuntimeError(
                        "SEMANTIC_ACTIVATION_SEAL_CAS_STALE"
                    )
            if (
                active_sql_execution_fingerprint
                and active_sql_execution_fingerprint
                not in session.executed_sql_candidate_fingerprints
            ):
                session.executed_sql_candidate_fingerprints.append(
                    active_sql_execution_fingerprint
                )
            session.active_plan = runtime_preparation.plan.model_copy(deep=True)
            session.active_preparation = runtime_preparation
            session.active_goal_contract_fingerprint = (
                normalized_goal_contract_fingerprint
            )
            session.run_result = core_run_result
            session.publication_authority_run_result = (
                publication_authority
            )
            session.publication_authority_fingerprint = (
                publication_authority_fingerprint
            )
            session.artifact_publication_required = bool(artifact_root)
            session.pending_query_publications = [
                GroundedPendingQueryPublication(
                    pending_artifact_id=str(
                        item.get("pendingArtifactId") or ""
                    ),
                    generation=generation,
                    attempt_id=session.active_attempt_id,
                    receipt=dict(item),
                    receipt_fingerprint=_stable_json_hash(dict(item)),
                    run_result_fingerprint=(
                        publication_authority_fingerprint
                    ),
                )
                for item in pending_receipts
                if str(item.get("pendingArtifactId") or "").strip()
            ]
            if terminal_code:
                session.terminal_guard_code = terminal_code
            session.verified_evidence = None
            self._clear_answer_snapshot(session)
            session.phase = "EXECUTED"
            session.revision += 1
            self._event(
                session,
                "execute_active",
                "OK",
                "tasks=%d" % len(run_result.task_results),
                session.active_attempt_id,
            )
        return core_run_result

    def verify_active(
        self,
        session: GroundedRuntimeSession,
        *,
        pre_ledger_authorizer: Optional[
            Callable[[GroundedVerifiedQueryArtifact], bool]
        ] = None,
    ) -> VerifiedEvidence:
        if self.verifier is None:
            raise RuntimeError(
                "grounded evidence verifier is not configured; refusing unverified answering"
            )
        with self._lock:
            generation, plan, _pack = self._active_snapshot(session)
            if session.run_result is None:
                raise RuntimeError("active grounded query has not executed")
            run_result_source = session.publication_authority_run_result
            if run_result_source is None:
                raise RuntimeError("QUERY_RESULT_PUBLICATION_AUTHORITY_REQUIRED")
            authority_fingerprint = str(
                session.publication_authority_fingerprint or ""
            )
            defer_publication = bool(session.defer_artifact_publication)
            publication_required = bool(
                session.artifact_publication_required
            )
            allowed_knowledge_refs = _recall_knowledge_ref_ids(session.recall)
        if self.semantic_activation_authority_available():
            self.revalidate_semantic_activation(session)
        verifier_run_result = run_result_source.model_copy(deep=True)
        if (
            authority_fingerprint
            and _query_run_result_fingerprint(verifier_run_result)
            != authority_fingerprint
        ):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_AUTHORITY_CORRUPT")
        verifier_kwargs: dict[str, Any] = {}
        if _plan_has_knowledge_evidence_contract(plan):
            verifier_kwargs["allowed_knowledge_refs"] = allowed_knowledge_refs
        semantic_activation_fingerprint = str(
            (
                session.semantic_activation_seal.semantic_activation_fingerprint
                if session.semantic_activation_seal is not None
                else ""
            )
            or ""
        )
        query_proof = build_grounded_query_proof(
            question=session.question,
            contract=session.active_contract,
            execution_plan=plan,
            run_result=verifier_run_result,
            merchant_scope_fingerprint=grounded_context_owner_fingerprint(
                session.merchant_id,
                session.access_role,
                session.user_scope,
            ),
            semantic_activation_fingerprint=semantic_activation_fingerprint,
            sql_validation=session.active_sql_validation,
        )
        verify_proof = getattr(self.verifier, "verify_proof", None)
        verified = (
            verify_proof(query_proof, **verifier_kwargs)
            if callable(verify_proof)
            else self.verifier.verify(
                session.question,
                plan,
                verifier_run_result,
                **verifier_kwargs,
            )
        )
        if not isinstance(verified, VerifiedEvidence):
            verified = VerifiedEvidence.model_validate(verified)
        if (
            authority_fingerprint
            and _query_run_result_fingerprint(verifier_run_result)
            != authority_fingerprint
        ):
            raise RuntimeError("QUERY_RESULT_VERIFIER_MUTATED_AUTHORITY")
        publication_requests: list[dict[str, Any]] = []
        published_receipts: list[dict[str, Any]] = []
        publication_error = ""
        if verified.passed and not defer_publication:
            with self._lock:
                try:
                    publication_snapshot = (
                        self._publication_snapshot_locked(
                            session,
                            generation,
                        )
                    )
                except Exception as exc:
                    publication_error = "%s:%s" % (
                        type(exc).__name__,
                        str(exc)[:400],
                    )
                    publication_snapshot = {}
            if publication_snapshot and not publication_error:
                try:
                    publication_requests = (
                        self._pending_publication_requests(
                            publication_snapshot,
                            verified,
                        )
                    )
                except Exception as exc:
                    publication_error = "%s:%s" % (
                        type(exc).__name__,
                        str(exc)[:400],
                    )
            if publication_requests and not publication_error:
                try:
                    # Potentially large hashing/copying is deliberately
                    # outside the global kernel lock and runtime budget clock.
                    published_receipts = self._publish_requests(
                        publication_requests
                    )
                except Exception as exc:
                    publication_error = "%s:%s" % (
                        type(exc).__name__,
                        str(exc)[:400],
                    )
        with self._lock:
            self._require_generation(session, generation)
            if publication_requests:
                self._assert_pending_publication_cas_locked(
                    session,
                    generation,
                    publication_requests,
                )
            if publication_error:
                verified = self._publication_failure_evidence(
                    session.active_attempt_id,
                    publication_error,
                )
            if verified.passed and published_receipts:
                for pending in session.pending_query_publications:
                    if pending.generation == generation:
                        pending.status = "PUBLISHED"
            session.verified_evidence = verified
            public_run_result = session.run_result
            private_run_result = session.publication_authority_run_result
            if public_run_result is None or private_run_result is None:
                raise RuntimeError("QUERY_RESULT_PUBLICATION_STATE_STALE")
            if published_receipts:
                self._attach_published_receipts(
                    public_run_result,
                    publication_snapshot.get("pending") or [],
                    published_receipts,
                )
            public_run_result.verified_evidence = verified.model_copy(
                deep=True
            )
            if verified.passed:
                deferred_publication = bool(
                    defer_publication and publication_required
                )
                if not deferred_publication:
                    if published_receipts:
                        self._attach_published_receipts(
                            private_run_result,
                            publication_snapshot.get("pending") or [],
                            published_receipts,
                        )
                    private_run_result.verified_evidence = (
                        verified.model_copy(deep=True)
                    )
                verified_artifact = self._record_verified_query_artifact(
                    session,
                    generation=generation,
                    plan=plan,
                    run_result=private_run_result,
                    verified=verified,
                    publication_status=(
                        "PENDING"
                        if defer_publication
                        and publication_required
                        else "PUBLISHED"
                        if published_receipts
                        else "VERIFIED_IN_MEMORY"
                    ),
                    result_artifact_receipts=published_receipts,
                    query_proof_fingerprint=query_proof.proof_fingerprint,
                    executed_sql_ast_fingerprint=(
                        query_proof.sql_ast_fingerprint
                    ),
                    append_to_ledger=(
                        pre_ledger_authorizer is None
                    ),
                )
                if pre_ledger_authorizer is not None:
                    try:
                        authorized = bool(
                            pre_ledger_authorizer(
                                verified_artifact.model_copy(
                                    deep=True
                                )
                            )
                        )
                    except Exception:
                        self._scrub_pending_publication_receipts(
                            session,
                            generation,
                        )
                        for pending in session.pending_query_publications:
                            if pending.generation == generation:
                                pending.status = (
                                    "POST_AUTHORIZATION_REJECTED"
                                )
                        session.verified_evidence = None
                        session.run_result = None
                        self._clear_answer_snapshot(session)
                        session.publication_authority_run_result = None
                        session.publication_authority_fingerprint = ""
                        session.phase = (
                            "VERIFIED_ARTIFACT_ADOPTION_REJECTED"
                        )
                        session.revision += 1
                        raise
                    if not authorized:
                        self._scrub_pending_publication_receipts(
                            session,
                            generation,
                        )
                        for pending in session.pending_query_publications:
                            if pending.generation == generation:
                                pending.status = (
                                    "POST_AUTHORIZATION_REJECTED"
                                )
                        session.verified_evidence = None
                        session.run_result = None
                        self._clear_answer_snapshot(session)
                        session.publication_authority_run_result = None
                        session.publication_authority_fingerprint = ""
                        session.phase = (
                            "VERIFIED_ARTIFACT_ADOPTION_REJECTED"
                        )
                        session.revision += 1
                        raise RuntimeError(
                            "VERIFIED_QUERY_ARTIFACT_ADOPTION_REJECTED"
                        )
                    if all(
                        item.artifact_id
                        != verified_artifact.artifact_id
                        for item in session.verified_query_ledger
                    ):
                        session.verified_query_ledger.append(
                            verified_artifact
                        )
            elif publication_error:
                for pending in session.pending_query_publications:
                    if pending.generation == generation:
                        pending.status = "PUBLICATION_FAILED"
            elif not verified.passed:
                for pending in session.pending_query_publications:
                    if pending.generation == generation:
                        pending.status = "VERIFICATION_FAILED"
            if not (
                verified.passed
                and defer_publication
                and publication_required
            ):
                self._scrub_pending_publication_receipts(
                    session,
                    generation,
                )
            session.publication_authority_run_result = None
            session.publication_authority_fingerprint = ""
            session.phase = "VERIFIED" if verified.passed else "VERIFICATION_GAPPED"
            session.revision += 1
            self._event(
                session,
                "verify_active",
                "PASSED" if verified.passed else "GAPPED",
                "blocking=%d" % len(verified.blocking_gaps),
                session.active_attempt_id,
            )
        return verified.model_copy(deep=True)

    def latest_verified_query_artifact(
        self,
        session: GroundedRuntimeSession,
    ) -> GroundedVerifiedQueryArtifact | None:
        with self._lock:
            if not session.verified_query_ledger:
                return None
            return session.verified_query_ledger[-1].model_copy(deep=True)

    def attach_verified_artifact_goals(
        self,
        session: GroundedRuntimeSession,
        artifact_id: str,
        goal_ids: Sequence[str],
    ) -> GroundedVerifiedQueryArtifact:
        """Version the trusted descriptor with server-assigned Goal coverage."""

        normalized_goal_ids = _dedupe(goal_ids)
        with self._lock:
            index = next(
                (
                    index
                    for index, item in enumerate(session.verified_query_ledger)
                    if item.artifact_id == artifact_id
                ),
                -1,
            )
            if index < 0:
                raise RuntimeError(
                    "VERIFIED_QUERY_ARTIFACT_NOT_FOUND:%s" % artifact_id
                )
            artifact = session.verified_query_ledger[index].model_copy(
                deep=True
            )
            descriptor = artifact.trusted_descriptor or (
                trusted_query_artifact_descriptor(
                    artifact,
                    merchant_scope_fingerprint=(
                        grounded_context_owner_fingerprint(
                            session.merchant_id,
                            session.access_role,
                            session.user_scope,
                        )
                    ),
                )
            )
            descriptor = descriptor.model_copy(
                update={"covered_goal_ids": normalized_goal_ids},
                deep=True,
            )
            artifact.trusted_descriptor = descriptor
            artifact.ledger_fingerprint = (
                verified_query_artifact_integrity_fingerprint(artifact)
            )
            session.verified_query_ledger[index] = artifact
            session.revision += 1
            self._event(
                session,
                "attach_verified_artifact_goals",
                "UPDATED",
                "artifact=%s;goals=%d"
                % (artifact_id, len(normalized_goal_ids)),
                artifact.attempt_id,
            )
            return artifact.model_copy(deep=True)

    @staticmethod
    def _all_result_bundles(
        run_result: AgentRunResult,
    ) -> list[QueryBundle]:
        bundles = [
            *[
                item.query_bundle
                for item in run_result.task_results
                if item.query_bundle is not None
            ],
            *list(run_result.query_bundles or []),
            run_result.merged_query_bundle,
        ]
        return bundles

    def _bounded_run_result_projection(
        self,
        run_result: AgentRunResult,
    ) -> AgentRunResult:
        """Return the Core-visible result without duplicating a full population."""

        projection = run_result.model_copy(deep=True)
        limit = self._result_preview_rows
        if limit is None:
            return projection
        seen_bundles: set[int] = set()
        for bundle in self._all_result_bundles(projection):
            if id(bundle) in seen_bundles:
                continue
            seen_bundles.add(id(bundle))
            if len(bundle.rows) <= limit:
                continue
            bundle.rows = [dict(row) for row in bundle.rows[:limit]]
            bundle.is_truncated = True
            bundle.result_coverage = ResultCoverage.PREVIEW
        for task_result in projection.task_results:
            entity_set = task_result.entity_set
            if entity_set is None:
                continue
            values_were_truncated = len(entity_set.values) > limit
            entity_set.values = list(entity_set.values[:limit])
            for column, values in list(entity_set.column_values.items()):
                if len(values) > limit:
                    values_were_truncated = True
                entity_set.column_values[column] = list(values[:limit])
            entity_set.truncated = bool(
                entity_set.truncated or values_were_truncated
            )
        return projection

    @classmethod
    def _extract_pending_result_artifacts(
        cls,
        run_result: AgentRunResult,
    ) -> list[dict[str, Any]]:
        """Move pending receipts out of every Core-visible result projection."""

        pending: list[dict[str, Any]] = []
        seen: set[str] = set()
        for bundle in cls._all_result_bundles(run_result):
            sanitized_events: list[dict[str, Any]] = []
            for raw_event in bundle.runtime_events or []:
                event = dict(raw_event or {})
                raw_receipt = event.pop(
                    "_serverPrivatePendingResultArtifact",
                    None,
                )
                if isinstance(raw_receipt, dict):
                    pending_id = str(
                        raw_receipt.get("pendingArtifactId") or ""
                    ).strip()
                    if pending_id and pending_id not in seen:
                        seen.add(pending_id)
                        pending.append(dict(raw_receipt))
                # An executor is not allowed to smuggle a published receipt
                # into an unverified execution result.
                event.pop("resultArtifact", None)
                sanitized_events.append(event)
            bundle.runtime_events = sanitized_events
            bundle.offloaded_files = []
            bundle.source_artifact_refs = {}
        return pending

    @staticmethod
    def _scrub_pending_publication_receipts(
        session: GroundedRuntimeSession,
        generation: int,
    ) -> None:
        """Release private capabilities once no future publish may use them."""

        for pending in session.pending_query_publications:
            if pending.generation != generation:
                continue
            pending.receipt = {}
            pending.receipt_fingerprint = ""
            pending.run_result_fingerprint = ""

    def _publication_snapshot_locked(
        self,
        session: GroundedRuntimeSession,
        generation: int,
    ) -> dict[str, Any]:
        self._require_generation(session, generation)
        authority_run_result = session.publication_authority_run_result
        authority_fingerprint = str(
            session.publication_authority_fingerprint or ""
        )
        authority_artifact_id = ""
        if authority_run_result is None:
            authority_artifact = next(
                (
                    artifact
                    for artifact in reversed(session.verified_query_ledger)
                    if artifact.generation == generation
                    and artifact.publication_status
                    in {"PENDING", "VERIFIED_IN_MEMORY"}
                ),
                None,
            )
            if authority_artifact is not None:
                authority_run_result = authority_artifact.run_result
                authority_fingerprint = str(
                    authority_artifact.run_result_fingerprint or ""
                )
                if (
                    not authority_fingerprint
                    and authority_artifact.publication_status
                    == "VERIFIED_IN_MEMORY"
                ):
                    authority_fingerprint = _query_run_result_fingerprint(
                        authority_run_result
                    )
                authority_artifact_id = authority_artifact.artifact_id
        if (
            authority_run_result is None
            and not session.artifact_publication_required
            and session.run_result is not None
        ):
            authority_run_result = session.run_result
            authority_fingerprint = _query_run_result_fingerprint(
                authority_run_result
            )
        if authority_run_result is None or session.active_contract is None:
            raise RuntimeError("QUERY_RESULT_PUBLICATION_STATE_REQUIRED")
        semantic_seal = session.semantic_activation_seal
        if self.semantic_activation_authority_available() and (
            semantic_seal is None
            or not semantic_activation_seal_valid(semantic_seal)
        ):
            raise RuntimeError(
                "QUERY_RESULT_SEMANTIC_ACTIVATION_SEAL_REQUIRED"
            )
        return {
            "generation": generation,
            "attemptId": session.active_attempt_id,
            "goalContractFingerprint": str(
                session.active_goal_contract_fingerprint or ""
            ),
            "merchantId": session.merchant_id,
            "accessRole": session.access_role,
            "userScope": dict(session.user_scope),
            "referenceScope": session.reference_scope.model_copy(deep=True),
            "contract": session.active_contract.model_copy(deep=True),
            "sqlValidation": (
                session.active_sql_validation.model_copy(deep=True)
                if session.active_sql_validation is not None
                else None
            ),
            "semanticActivationFingerprint": str(
                semantic_seal.semantic_activation_fingerprint
                if semantic_seal is not None
                else ""
                or ""
            ),
            "semanticActivationSealFingerprint": str(
                semantic_seal.seal_fingerprint
                if semantic_seal is not None
                else ""
            ),
            "semanticActivationTopics": list(
                semantic_seal.exact_topics
                if semantic_seal is not None
                else []
            ),
            # This private deep copy was sealed when execute_active accepted
            # the result. Core only receives session.run_result, never this
            # publication authority object.
            "runResult": authority_run_result,
            "runResultToken": id(authority_run_result),
            "runResultFingerprint": authority_fingerprint,
            "runResultArtifactId": authority_artifact_id,
            "publicationRequired": bool(
                session.artifact_publication_required
            ),
            "pending": [
                item.model_copy(deep=True)
                for item in session.pending_query_publications
                if item.generation == generation
            ],
        }

    def _pending_publication_requests(
        self,
        snapshot: dict[str, Any],
        verified: VerifiedEvidence,
    ) -> list[dict[str, Any]]:
        if not bool(snapshot.get("publicationRequired")):
            return []
        generation = int(snapshot.get("generation") or 0)
        attempt_id = str(snapshot.get("attemptId") or "")
        pending = [
            item
            for item in snapshot.get("pending") or []
            if item.generation == generation
            and item.attempt_id == attempt_id
            and item.status == "PENDING"
        ]
        if not pending:
            raise RuntimeError("QUERY_RESULT_PENDING_RECEIPT_REQUIRED")
        if not callable(
            getattr(
                self.executor,
                "publish_pending_result_artifact",
                None,
            )
        ):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_GATE_UNAVAILABLE")
        contract = snapshot.get("contract")
        if not isinstance(contract, GroundedQueryContract):
            raise RuntimeError("QUERY_RESULT_ACTIVE_CONTRACT_REQUIRED")
        run_result = snapshot.get("runResult")
        if not isinstance(run_result, AgentRunResult):
            raise RuntimeError("QUERY_RESULT_RUN_RESULT_REQUIRED")
        authority_fingerprint = str(
            snapshot.get("runResultFingerprint") or ""
        )
        if (
            not authority_fingerprint
            or _query_run_result_fingerprint(run_result)
            != authority_fingerprint
        ):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_AUTHORITY_MISMATCH")
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        sql_fingerprint = self._sql_evidence_fingerprint(
            snapshot.get("sqlValidation"),
            contract_fingerprint,
            run_result,
        )
        semantic_activation_fingerprint = str(
            snapshot.get("semanticActivationFingerprint") or ""
        ).strip()
        if not semantic_activation_fingerprint:
            raise RuntimeError(
                "QUERY_RESULT_SEMANTIC_ACTIVATION_FINGERPRINT_REQUIRED"
            )
        bundle = run_result.merged_query_bundle
        if (
            str(
                bundle.data_snapshot.semantic_activation_fingerprint
                or ""
            ).strip()
            != semantic_activation_fingerprint
        ):
            raise RuntimeError(
                "QUERY_RESULT_DATA_SNAPSHOT_ACTIVATION_MISMATCH"
            )
        context_owner_fingerprint = grounded_context_owner_fingerprint(
            str(snapshot.get("merchantId") or ""),
            str(snapshot.get("accessRole") or ""),
            dict(snapshot.get("userScope") or {}),
        )
        rows_canonical_sha256 = _stable_json_hash(bundle.rows or [])
        requests: list[dict[str, Any]] = []
        for item in pending:
            private_receipt = dict(item.receipt)
            receipt_fingerprint = _stable_json_hash(private_receipt)
            if (
                not item.receipt_fingerprint
                or item.receipt_fingerprint != receipt_fingerprint
                or item.run_result_fingerprint != authority_fingerprint
            ):
                raise RuntimeError("QUERY_RESULT_PENDING_AUTHORITY_MISMATCH")
            requests.append(
                {
                    "pendingArtifactId": item.pending_artifact_id,
                    "generation": generation,
                    "attemptId": attempt_id,
                    "runResultToken": int(
                        snapshot.get("runResultToken") or 0
                    ),
                    "runResultArtifactId": str(
                        snapshot.get("runResultArtifactId") or ""
                    ),
                    "pendingReceiptFingerprint": receipt_fingerprint,
                    "runResultFingerprint": authority_fingerprint,
                    "receipt": private_receipt,
                    "publishKwargs": {
                        "verified_evidence": verified.model_copy(deep=True),
                        "expected_generation": generation,
                        "expected_attempt_id": attempt_id,
                        "expected_contract_fingerprint": contract_fingerprint,
                        "expected_goal_contract_fingerprint": str(
                            snapshot.get("goalContractFingerprint") or ""
                        ),
                        "expected_merchant_id": str(
                            snapshot.get("merchantId") or ""
                        ),
                        "expected_access_role": str(
                            snapshot.get("accessRole") or ""
                        ),
                        "expected_user_scope": dict(
                            snapshot.get("userScope") or {}
                        ),
                        "expected_reference_scope": snapshot[
                            "referenceScope"
                        ].model_copy(deep=True),
                        "expected_sql_fingerprint": sql_fingerprint,
                        "expected_context_owner_fingerprint": (
                            context_owner_fingerprint
                        ),
                        "expected_semantic_activation_fingerprint": (
                            semantic_activation_fingerprint
                        ),
                        "expected_data_snapshot": (
                            bundle.data_snapshot.model_copy(deep=True)
                        ),
                        "expected_result_coverage": str(
                            bundle.result_coverage or ""
                        ),
                        "expected_result_is_truncated": bool(
                            bundle.is_truncated
                        ),
                        "expected_stored_row_count": len(bundle.rows),
                        "expected_exact_result_row_count": max(
                            0,
                            int(bundle.original_row_count or 0),
                        ),
                        "expected_rows_canonical_sha256": (
                            rows_canonical_sha256
                        ),
                    },
                }
            )
        return requests

    def _publish_requests(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        publish = getattr(
            self.executor,
            "publish_pending_result_artifact",
            None,
        )
        if not callable(publish):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_GATE_UNAVAILABLE")
        published: list[dict[str, Any]] = []
        for request in requests:
            receipt = publish(
                dict(request.get("receipt") or {}),
                **dict(request.get("publishKwargs") or {}),
            )
            self._validate_published_receipt(receipt)
            if str(receipt.get("pendingArtifactId") or "") != str(
                request.get("pendingArtifactId") or ""
            ):
                raise RuntimeError(
                    "QUERY_RESULT_PUBLICATION_PENDING_ID_MISMATCH"
                )
            published.append(dict(receipt))
        return published

    @staticmethod
    def _assert_pending_publication_cas_locked(
        session: GroundedRuntimeSession,
        generation: int,
        requests: Sequence[dict[str, Any]],
    ) -> None:
        if session.active_generation != generation:
            raise RuntimeError("stale grounded runtime generation")
        for request in requests:
            artifact_id = str(
                request.get("runResultArtifactId") or ""
            )
            if artifact_id:
                artifact = next(
                    (
                        item
                        for item in session.verified_query_ledger
                        if item.artifact_id == artifact_id
                        and item.generation == generation
                        and item.publication_status == "PENDING"
                    ),
                    None,
                )
                authority = artifact.run_result if artifact is not None else None
                authority_fingerprint = (
                    artifact.run_result_fingerprint
                    if artifact is not None
                    else ""
                )
            else:
                authority = session.publication_authority_run_result
                authority_fingerprint = (
                    session.publication_authority_fingerprint
                )
            if (
                session.active_attempt_id
                != str(request.get("attemptId") or "")
                or id(authority)
                != int(request.get("runResultToken") or 0)
                or authority_fingerprint
                != str(request.get("runResultFingerprint") or "")
            ):
                raise RuntimeError("QUERY_RESULT_PUBLICATION_STATE_STALE")
        current = {
            item.pending_artifact_id: item
            for item in session.pending_query_publications
            if item.generation == generation
        }
        for request in requests:
            pending_id = str(request.get("pendingArtifactId") or "")
            item = current.get(pending_id)
            if (
                item is None
                or item.status != "PENDING"
                or item.attempt_id != str(request.get("attemptId") or "")
                or item.receipt_fingerprint
                != str(request.get("pendingReceiptFingerprint") or "")
                or item.run_result_fingerprint
                != str(request.get("runResultFingerprint") or "")
                or _stable_json_hash(dict(item.receipt))
                != item.receipt_fingerprint
            ):
                raise RuntimeError("QUERY_RESULT_PENDING_RECEIPT_STALE")

    @staticmethod
    def _sql_evidence_fingerprint(
        validation: GroundedSqlValidationResult | None,
        contract_fingerprint: str,
        run_result: AgentRunResult,
    ) -> str:
        if validation is not None and validation.ast_fingerprint:
            return str(validation.ast_fingerprint)
        return hashlib.sha256(
            (
                "%s:%s"
                % (
                    contract_fingerprint,
                    run_result.merged_query_bundle.sql,
                )
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _validate_published_receipt(receipt: Any) -> None:
        if not isinstance(receipt, dict):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_RECEIPT_INVALID")
        required = (
            "artifactFingerprint",
            "queryManifestSha256",
            "rowsSha256",
            "sqlSha256",
            "resultCoverage",
            "manifestRelativePath",
            "manifestRef",
            "rowsRef",
            "sqlRef",
            "verifiedEvidenceSha256",
        )
        if any(not str(receipt.get(key) or "").strip() for key in required):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_RECEIPT_INCOMPLETE")
        for key in ("manifestRef", "rowsRef", "sqlRef"):
            if not str(receipt.get(key) or "").startswith("merchant://"):
                raise RuntimeError("QUERY_RESULT_PUBLICATION_REF_INVALID")
        for key in (
            "artifactFingerprint",
            "queryManifestSha256",
            "rowsSha256",
            "sqlSha256",
            "verifiedEvidenceSha256",
        ):
            value = str(receipt.get(key) or "")
            if len(value) != 64 or any(
                character not in "0123456789abcdef"
                for character in value
            ):
                raise RuntimeError(
                    "QUERY_RESULT_PUBLICATION_DIGEST_INVALID:%s" % key
                )
        for key, value in receipt.items():
            if "path" in str(key).lower() and str(value or "").startswith(
                "/"
            ):
                raise RuntimeError(
                    "QUERY_RESULT_PUBLICATION_ABSOLUTE_PATH_FORBIDDEN"
                )

    @staticmethod
    def _publication_failure_evidence(
        attempt_id: str,
        reason: str,
    ) -> VerifiedEvidence:
        gap = EvidenceGap(
            code="QUERY_RESULT_ARTIFACT_PUBLICATION_FAILED",
            evidence=str(attempt_id or ""),
            reason=str(reason or "")[:500],
            severity="blocking",
        )
        return VerifiedEvidence(
            passed=False,
            gaps=[gap.model_copy(deep=True)],
            blocking_gaps=[gap.model_copy(deep=True)],
            answer_guard_required=True,
            partial_answer_reason=(
                "Verified rows could not be committed to the artifact "
                "allowlist ledger."
            ),
        )

    @classmethod
    def _attach_published_receipts(
        cls,
        run_result: AgentRunResult,
        pending: Sequence[GroundedPendingQueryPublication],
        published: Sequence[dict[str, Any]],
    ) -> None:
        by_pending_id = {
            str(item.get("pendingArtifactId") or ""): dict(item)
            for item in published
            if str(item.get("pendingArtifactId") or "").strip()
        }
        task_fingerprints = {
            item.pending_artifact_id: str(
                item.receipt.get("identity", {}).get("taskFingerprint")
                or ""
            )
            for item in pending
        }
        for bundle in cls._all_result_bundles(run_result):
            bundle.offloaded_files = []
            for event in bundle.runtime_events or []:
                task_id = str(event.get("taskId") or "")
                task_fingerprint = hashlib.sha256(
                    task_id.encode("utf-8")
                ).hexdigest()
                matches = [
                    receipt
                    for pending_id, receipt in by_pending_id.items()
                    if task_fingerprints.get(pending_id) == task_fingerprint
                ]
                if len(matches) == 1:
                    event["resultArtifact"] = dict(matches[0])
                    refs = [
                        str(matches[0].get(key) or "")
                        for key in ("manifestRef", "rowsRef", "sqlRef")
                        if str(matches[0].get(key) or "").startswith(
                            "merchant://"
                        )
                    ]
                    if task_id and refs:
                        bundle.source_artifact_refs[task_id] = refs

    def publish_verified_entity_set(
        self,
        session: GroundedRuntimeSession,
        query_artifact_id: str,
        output_column: str,
        *,
        limit: int = 500,
    ) -> GroundedVerifiedEntitySet:
        """Publish typed entity values from immutable verified evidence.

        Full values remain kernel-side. Downstream Contracts reference the
        returned artifact ID and cannot replace or amend its contents.
        """

        artifact_id = str(query_artifact_id or "").strip()
        column = str(output_column or "").strip()
        bounded_limit = max(1, min(int(limit or 500), 5000))
        with self._lock:
            source = next(
                (
                    item.model_copy(deep=True)
                    for item in session.verified_query_ledger
                    if item.artifact_id == artifact_id
                ),
                None,
            )
        if source is None:
            raise RuntimeError(
                "VERIFIED_QUERY_ARTIFACT_NOT_FOUND:%s" % artifact_id
            )
        if not source.verified_evidence.passed:
            raise RuntimeError("VERIFIED_QUERY_ARTIFACT_REQUIRED")
        source_bundle = source.run_result.merged_query_bundle
        source_coverage = str(
            source_bundle.result_coverage or ResultCoverage.UNKNOWN.value
        )
        if (
            source.sealed_entity_values_truncated
            or source_bundle.is_truncated
            or source_coverage
            not in {
                ResultCoverage.ALL_ROWS.value,
                ResultCoverage.TOP_N.value,
            }
        ):
            raise RuntimeError(
                "VERIFIED_ENTITY_SET_INCOMPLETE_COVERAGE:%s"
                % source_coverage
            )
        if not column or column not in source.output_columns:
            raise RuntimeError(
                "VERIFIED_ENTITY_OUTPUT_COLUMN_NOT_FOUND:%s" % column
            )
        semantic_ref = str(source.output_semantic_refs.get(column) or "").strip()
        if not semantic_ref:
            raise RuntimeError(
                "VERIFIED_ENTITY_SEMANTIC_LINEAGE_REQUIRED:%s" % column
            )
        entity_identity = str(
            source.output_entity_identities.get(column) or ""
        ).strip()
        if not entity_identity:
            raise RuntimeError(
                "VERIFIED_ENTITY_IDENTITY_REQUIRED:%s" % column
            )

        sealed_values = list(source.sealed_entity_values.get(column) or [])
        if not sealed_values and source.run_result.merged_query_bundle.rows:
            raise RuntimeError(
                "VERIFIED_ENTITY_PRE_MASK_VALUES_REQUIRED:%s" % column
            )
        values_by_identity: dict[str, Any] = {}
        for value in sealed_values:
            identity = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            values_by_identity.setdefault(identity, value)
        ordered_identities = sorted(values_by_identity)
        all_values_count = len(ordered_identities)
        values = [
            values_by_identity[identity]
            for identity in ordered_identities[:bounded_limit]
        ]
        values_hash = _stable_json_hash(values)
        entity_artifact_id = "entity_set_%s" % hashlib.sha256(
            (
                "%s:%s:%s:%s"
                % (artifact_id, column, semantic_ref, values_hash)
            ).encode("utf-8")
        ).hexdigest()[:16]
        published = GroundedVerifiedEntitySet(
            artifact_id=entity_artifact_id,
            source_query_artifact_id=artifact_id,
            source_column=column,
            source_semantic_ref_id=semantic_ref,
            source_entity_identity=entity_identity,
            values=values,
            value_count=all_values_count,
            truncated=(
                source.sealed_entity_values_truncated
                or all_values_count > len(values)
            ),
            values_hash=values_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        published = published.model_copy(
            update={
                "trusted_descriptor": trusted_entity_set_descriptor(
                    published,
                    merchant_scope_fingerprint=grounded_context_owner_fingerprint(
                        session.merchant_id,
                        session.access_role,
                        session.user_scope,
                    ),
                )
            },
            deep=True,
        )
        with self._lock:
            existing = next(
                (
                    item
                    for item in session.verified_entity_sets
                    if item.artifact_id == published.artifact_id
                ),
                None,
            )
            if existing is not None:
                return existing.model_copy(deep=True)
            session.verified_entity_sets.append(published.model_copy(deep=True))
            session.revision += 1
            self._event(
                session,
                "publish_verified_entity_set",
                "PUBLISHED",
                "artifact=%s;column=%s;values=%d;truncated=%s"
                % (
                    published.artifact_id,
                    column,
                    published.value_count,
                    str(published.truncated).lower(),
                ),
                source.attempt_id,
            )
        return published.model_copy(deep=True)

    @staticmethod
    def _record_verified_query_artifact(
        session: GroundedRuntimeSession,
        *,
        generation: int,
        plan: QueryPlan,
        run_result: AgentRunResult,
        verified: VerifiedEvidence,
        publication_status: str = "VERIFIED_IN_MEMORY",
        result_artifact_receipts: Sequence[dict[str, Any]] = (),
        query_proof_fingerprint: str = "",
        executed_sql_ast_fingerprint: str = "",
        append_to_ledger: bool = True,
    ) -> GroundedVerifiedQueryArtifact:
        if session.active_contract is None:
            raise RuntimeError("verified query has no active grounded Contract")
        semantic_seal = session.semantic_activation_seal
        if semantic_seal is not None and not semantic_activation_seal_valid(
            semantic_seal
        ):
            raise RuntimeError("SEMANTIC_ACTIVATION_SEAL_CORRUPT")
        contract = session.active_contract.model_copy(deep=True)
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        validation = session.active_sql_validation
        sql_fingerprint = (
            str(validation.ast_fingerprint or "")
            if validation is not None
            else ""
        )
        if not sql_fingerprint:
            sql_fingerprint = hashlib.sha256(
                (
                    "%s:%s"
                    % (
                        contract_fingerprint,
                        run_result.merged_query_bundle.sql,
                    )
                ).encode("utf-8")
            ).hexdigest()
        existing = next(
            (
                item
                for item in session.verified_query_ledger
                if item.generation == generation
                and item.contract_fingerprint == contract_fingerprint
                and item.sql_fingerprint == sql_fingerprint
            ),
            None,
        )
        if existing is not None:
            return existing

        expected_columns = _contract_output_columns(contract)
        observed_columns = _observed_output_columns(run_result)
        output_columns = _dedupe([*expected_columns, *observed_columns])
        semantic_refs, entity_identities = _contract_output_semantics(contract)
        output_lineage = (
            dict(validation.output_lineage)
            if validation is not None
            else dict(
                plan.question_understanding.get("outputLineage") or {}
            )
        )
        sealed_entity_values: dict[str, list[Any]] = {}
        result_bundle = run_result.merged_query_bundle
        result_coverage = str(result_bundle.result_coverage or "UNKNOWN")
        sealed_values_truncated = bool(
            result_bundle.is_truncated
            or result_coverage
            not in {
                ResultCoverage.ALL_ROWS.value,
                ResultCoverage.TOP_N.value,
            }
        )
        for task_result in run_result.task_results:
            entity_set = task_result.entity_set
            if entity_set is None:
                continue
            sealed_values_truncated = (
                sealed_values_truncated or bool(entity_set.truncated)
            )
            for output, values in entity_set.column_values.items():
                if output in output_columns:
                    sealed_entity_values[output] = list(values)
        artifact_id = "query_artifact_%s" % hashlib.sha256(
            (
                "%s:%d:%s:%s"
                % (
                    session.session_id,
                    generation,
                    contract_fingerprint,
                    sql_fingerprint,
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        artifact = GroundedVerifiedQueryArtifact(
            artifact_id=artifact_id,
            generation=generation,
            attempt_id=session.active_attempt_id,
            contract_fingerprint=contract_fingerprint,
            sql_fingerprint=sql_fingerprint,
            contract=contract,
            plan=plan.model_copy(deep=True),
            # ``run_result`` is a kernel-owned sealed object at this point.
            # Taking ownership avoids retaining another complete row
            # population beside the verified ledger.
            run_result=run_result,
            verified_evidence=verified.model_copy(deep=True),
            execution_mode=str(session.active_execution_mode or ""),
            sql_validation=(
                validation.model_copy(deep=True)
                if validation is not None
                else None
            ),
            ranking_semantics_verified=bool(
                str(contract.query_shape or "").upper() != "RANKED"
                or (
                    validation is not None
                    and validation.valid
                    and validation.contract_fingerprint == contract_fingerprint
                    and bool(validation.ast_fingerprint)
                )
                or session.active_execution_mode
                == GroundedExecutionMode.DETERMINISTIC_RANKED
            ),
            output_columns=output_columns,
            output_semantic_refs=semantic_refs,
            output_entity_identities=entity_identities,
            output_lineage=output_lineage,
            sealed_entity_values=sealed_entity_values,
            sealed_entity_values_truncated=sealed_values_truncated,
            publication_status=str(
                publication_status or "VERIFIED_IN_MEMORY"
            ),
            result_artifact_receipts=[
                dict(item) for item in result_artifact_receipts
            ],
            plan_fingerprint=_stable_json_hash(
                plan.model_dump(by_alias=True, mode="json")
            ),
            run_result_fingerprint=_query_run_result_fingerprint(run_result),
            semantic_activation_fingerprint=(
                semantic_seal.semantic_activation_fingerprint
                if semantic_seal is not None
                else ""
            ),
            semantic_activation_seal_fingerprint=(
                semantic_seal.seal_fingerprint
                if semantic_seal is not None
                else ""
            ),
            semantic_activation_topics=(
                list(semantic_seal.exact_topics)
                if semantic_seal is not None
                else []
            ),
            query_proof_fingerprint=str(
                query_proof_fingerprint or ""
            ),
            executed_sql_ast_fingerprint=str(
                executed_sql_ast_fingerprint or ""
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        artifact = artifact.model_copy(
            update={
                "trusted_descriptor": trusted_query_artifact_descriptor(
                    artifact,
                    merchant_scope_fingerprint=grounded_context_owner_fingerprint(
                        session.merchant_id,
                        session.access_role,
                        session.user_scope,
                    ),
                )
            },
            deep=True,
        )
        artifact.ledger_fingerprint = (
            verified_query_artifact_integrity_fingerprint(artifact)
        )
        if append_to_ledger:
            session.verified_query_ledger.append(artifact)
        return artifact

    @staticmethod
    def _copy_verified_artifact_with_run_result(
        source: GroundedVerifiedQueryArtifact,
        run_result: AgentRunResult,
    ) -> GroundedVerifiedQueryArtifact:
        """Copy ledger metadata without first copying its full row payload."""

        artifact = source.model_copy(deep=False)
        artifact.contract = source.contract.model_copy(deep=True)
        artifact.plan = source.plan.model_copy(deep=True)
        artifact.run_result = run_result
        artifact.verified_evidence = source.verified_evidence.model_copy(
            deep=True
        )
        artifact.sql_validation = (
            source.sql_validation.model_copy(deep=True)
            if source.sql_validation is not None
            else None
        )
        artifact.output_columns = list(source.output_columns)
        artifact.output_semantic_refs = dict(source.output_semantic_refs)
        artifact.output_entity_identities = dict(
            source.output_entity_identities
        )
        artifact.output_lineage = {
            key: list(values)
            for key, values in source.output_lineage.items()
        }
        artifact.sealed_entity_values = {
            key: list(values)
            for key, values in source.sealed_entity_values.items()
        }
        artifact.result_artifact_receipts = [
            dict(item) for item in source.result_artifact_receipts
        ]
        artifact.run_result_fingerprint = _query_run_result_fingerprint(
            run_result
        )
        artifact.ledger_fingerprint = ""
        return artifact

    def compose_answer(
        self,
        session: GroundedRuntimeSession,
        *,
        knowledge_context: str = "",
        analysis_summary: str = "",
        allow_llm: bool = True,
        rule_context: str = "",
        personalization_context: dict[str, Any] | None = None,
        runtime_budget: Any | None = None,
    ) -> str:
        if self.answer_composer is None:
            raise RuntimeError(
                "grounded answer composer is not configured; refusing workflow fallback"
            )
        with self._lock:
            generation, _active_plan, _pack = self._active_snapshot(session)
            plan, run_result, portfolio_verified, artifact_ids = (
                self._verified_portfolio_snapshot(session)
            )
        if self.verifier is not None:
            portfolio_verified = self.verifier.verify(
                session.question,
                plan,
                run_result,
            )
            if not isinstance(portfolio_verified, VerifiedEvidence):
                portfolio_verified = VerifiedEvidence.model_validate(
                    portfolio_verified
                )
            run_result.verified_evidence = portfolio_verified.model_copy(
                deep=True
            )
        if not portfolio_verified.passed:
            codes = _dedupe(
                gap.code or gap.gap_code
                for gap in portfolio_verified.blocking_gaps
            )
            raise RuntimeError(
                "EVIDENCE_PORTFOLIO_INCOMPLETE:%s"
                % (",".join(codes[:8]) or "UNVERIFIED")
            )
        answer = self.answer_composer.compose(
            session.question,
            session.merchant,
            plan,
            run_result,
            knowledge_context,
            analysis_summary=analysis_summary,
            allow_llm=allow_llm,
            rule_context=rule_context,
            personalization_context=personalization_context,
            runtime_budget=runtime_budget,
        )
        with self._lock:
            self._require_generation(session, generation)
            session.answer = str(answer or "")
            session.answer_plan = plan.model_copy(deep=True)
            session.answer_run_result = run_result.model_copy(deep=True)
            session.answer_verified_evidence = portfolio_verified.model_copy(
                deep=True
            )
            session.answer_artifact_ids = list(artifact_ids)
            session.phase = "ANSWERED"
            session.revision += 1
            self._event(
                session,
                "compose_answer",
                "OK",
                "chars=%d" % len(session.answer),
                session.active_attempt_id,
            )
        return session.answer

    def publish_rule_artifact(
        self,
        session: GroundedRuntimeSession,
        artifact: GroundedVerifiedRuleArtifact,
    ) -> GroundedVerifiedRuleArtifact:
        if not artifact.verification_passed:
            raise ValueError("rule artifact must be verified before publication")
        if artifact.question.strip() != session.question.strip():
            raise ValueError("rule artifact question does not match the active session")
        with self._lock:
            existing = next(
                (
                    item
                    for item in session.verified_rule_ledger
                    if item.artifact_id == artifact.artifact_id
                ),
                None,
            )
            if existing is not None:
                return existing.model_copy(deep=True)
            session.verified_rule_ledger.append(artifact.model_copy(deep=True))
            session.phase = "RULE_EVIDENCE_VERIFIED"
            session.revision += 1
            self._event(
                session,
                "publish_rule_artifact",
                "VERIFIED",
                "artifact=%s;refs=%d"
                % (artifact.artifact_id, len(artifact.evidence_refs)),
            )
        return artifact.model_copy(deep=True)

    def compose_rule_answer(
        self,
        session: GroundedRuntimeSession,
        *,
        artifact_ids: Sequence[str] = (),
    ) -> str:
        requested = {str(item or "").strip() for item in artifact_ids if str(item or "").strip()}
        with self._lock:
            artifacts = [
                item.model_copy(deep=True)
                for item in session.verified_rule_ledger
                if not requested or item.artifact_id in requested
            ]
        if not artifacts:
            raise RuntimeError("VERIFIED_RULE_EVIDENCE_REQUIRED")
        answer = render_verified_rule_answer(artifacts)
        with self._lock:
            session.answer = answer
            session.answer_artifact_ids = []
            session.answer_rule_artifact_ids = [
                item.artifact_id for item in artifacts
            ]
            session.phase = "ANSWERED"
            session.revision += 1
            self._event(
                session,
                "compose_rule_answer",
                "OK",
                "chars=%d;artifacts=%d" % (len(answer), len(artifacts)),
            )
        return answer

    def verified_portfolio(
        self,
        session: GroundedRuntimeSession,
    ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
        with self._lock:
            return self._verified_portfolio_snapshot(session)

    def verify_portfolio(
        self,
        session: GroundedRuntimeSession,
    ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
        with self._lock:
            plan, run_result, verified, artifact_ids = (
                self._verified_portfolio_snapshot(session)
            )
        if self.verifier is not None:
            verified = self.verifier.verify(
                session.question,
                plan,
                run_result,
            )
            if not isinstance(verified, VerifiedEvidence):
                verified = VerifiedEvidence.model_validate(verified)
            run_result.verified_evidence = verified.model_copy(deep=True)
        return plan, run_result, verified, artifact_ids

    @staticmethod
    def _verified_portfolio_snapshot(
        session: GroundedRuntimeSession,
    ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
        artifacts = [
            item.model_copy(deep=True)
            for item in session.verified_query_ledger
            if item.verified_evidence.passed
        ]
        if not artifacts:
            raise RuntimeError(
                "grounded answer requires at least one verified query artifact"
            )
        plans: list[QueryPlan] = []
        runs: list[AgentRunResult] = []
        for artifact in artifacts:
            plans.append(
                _namespace_artifact_plan(
                    artifact.plan,
                    artifact.artifact_id,
                )
            )
            runs.append(
                _namespace_artifact_run(
                    artifact.run_result,
                    artifact.artifact_id,
                )
            )
        combined_plan = _combine_artifact_plans(
            plans,
            [item.artifact_id for item in artifacts],
        )
        combined_verified = _combine_verified_evidence(
            [item.verified_evidence for item in artifacts]
        )
        combined_run = _combine_artifact_runs(
            runs,
            [item.artifact_id for item in artifacts],
            combined_verified,
        )
        return (
            combined_plan,
            combined_run,
            combined_verified,
            [item.artifact_id for item in artifacts],
        )

    def request_clarification(
        self,
        session: GroundedRuntimeSession,
        question: str,
        *,
        stage: str,
        clarification_type: str,
        options: Sequence[str] | None = None,
    ) -> ClarificationRequest:
        request = ClarificationRequest(
            question=str(question or "").strip(),
            stage=str(stage or "grounded_runtime"),
            type=str(clarification_type or "missing_information"),
            options=_dedupe(options or []),
            pending_question=session.question,
        )
        if not request.question:
            raise ValueError("clarification question is required")
        with self._lock:
            session.clarification = request
            session.phase = "CLARIFICATION_REQUIRED"
            session.revision += 1
            self._event(session, "request_clarification", "REQUIRED", request.type)
        return request

    @staticmethod
    def _attempt(
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        attempt = next(
            (item for item in session.attempts if item.attempt_id == attempt_id),
            None,
        )
        if attempt is None:
            raise KeyError("unknown grounded runtime attempt: %s" % attempt_id)
        return attempt

    @staticmethod
    def _sql_candidate_next_action(
        validation: GroundedSqlValidationResult,
    ) -> str:
        revise_binding_codes = {
            "GROUNDED_CONTRACT_NOT_READY",
            "SQL_GOVERNED_FORMULA_INVALID",
            "SQL_GOVERNED_COMPOSITE_FORMULA_INVALID",
        }
        if any(
            gap.blocking and gap.code in revise_binding_codes
            for gap in validation.gaps
        ):
            return "REVISE_BINDINGS"
        sql_or_binding_codes = {
            "SQL_TABLE_NOT_GROUNDED",
            "SQL_COLUMN_NOT_GROUNDED",
            "SQL_JOIN_SOURCE_UNRESOLVED",
            "SQL_JOIN_NOT_GOVERNED",
            "SQL_CORRELATED_RELATIONSHIP_UNPROVEN",
        }
        if any(
            gap.blocking and gap.code in sql_or_binding_codes
            for gap in validation.gaps
        ):
            return "REPAIR_SQL_OR_BINDINGS"
        return "REPAIR_SQL"

    @staticmethod
    def _build_core_sql_evidence_plan(
        contract: GroundedQueryContract,
        validation: GroundedSqlValidationResult,
    ) -> QueryPlan:
        """Project accepted SQL outputs into evidence metadata, not SQL topology."""

        shape = str(contract.query_shape or "").strip().upper()
        answer_mode = {
            "SCALAR": AnswerMode.METRIC,
            "GROUPED": AnswerMode.GROUP_AGG,
            "TREND": AnswerMode.GROUP_AGG,
            "RANKED": AnswerMode.TOPN,
            "DETAIL": AnswerMode.DETAIL,
            "ENTITY_LOOKUP": AnswerMode.DETAIL,
        }.get(shape, AnswerMode.DERIVED)
        primary_table = str(
            contract.primary_table
            or (validation.referenced_tables[0] if validation.referenced_tables else "")
        ).strip()
        task_id = "grounded_core_sql_%s" % (
            validation.ast_fingerprint[:12]
            or hashlib.sha256(validation.canonical_sql.encode("utf-8")).hexdigest()[:12]
        )
        metric_specs = [
            {
                "metricName": metric.metric_key,
                "displayName": (
                    metric.requested_phrase
                    or metric.business_name
                    or metric.metric_key
                ),
                "sourcePhrase": (
                    metric.requested_phrase
                    or metric.business_name
                    or metric.metric_key
                ),
                "businessName": metric.business_name or metric.metric_key,
                "metricColumn": (
                    metric.source_columns[0] if metric.source_columns else ""
                ),
                "metricFormula": metric.formula,
                "sourceColumns": list(metric.source_columns),
                "sourceTaskId": task_id,
                "semanticRefId": metric.semantic_ref_id,
                "ownerTable": metric.table,
                "aggregationPolicy": metric.aggregation_policy,
                "metricGrain": metric.metric_grain,
                "applicableTimeGrain": metric.applicable_time_grain,
                "timeColumn": metric.time_column,
                "timeSemantics": dict(metric.time_semantics),
                "unit": metric.unit,
                "bindingType": metric.binding_type,
                "fieldAggregation": metric.field_aggregation,
                "sourceFieldRefId": metric.source_field_ref_id,
            }
            for metric in contract.metrics
        ]
        first_metric = contract.metrics[0] if contract.metrics else None
        grouped_dimensions = [
            item for item in contract.dimensions if item.usage == "group_by"
        ]
        entity_obligations: list[EntityFilterObligation] = []
        for index, item in enumerate(contract.entity_filters):
            values = (
                list(item.literal_value)
                if item.operator == "IN"
                and isinstance(item.literal_value, (list, tuple))
                else [item.literal_value]
            )
            entity_obligations.append(
                EntityFilterObligation(
                    obligation_id="grounded_core_sql_entity_%d" % (index + 1),
                    task_id=task_id,
                    required=True,
                    reference=EntityReference(
                        semantic_ref_id=item.semantic_ref_id,
                        field=item.column,
                        table=item.table,
                        raw_label=item.requested_phrase,
                        raw_value=str(item.literal_value),
                        values=values,
                        comparison_policy=item.operator.lower(),
                        source="grounded_core_sql_candidate",
                        confidence=1.0,
                        status="bound",
                        time_scope_explicit=bool(contract.time_range.explicit),
                        lookup_time_policy=dict(item.lookup_time_policy),
                    ),
                    status="bound",
                    reason="accepted Core SQL contains the mandatory typed predicate",
                )
            )
        output_columns = list(validation.output_columns)
        output_labels = {
            str(item.column): str(item.business_name or item.column)
            for item in contract.dimensions
            if str(item.column or "").strip()
        }
        output_labels.update(
            {
                str(item.output_alias or item.column): str(
                    requested_semantic_label(
                        contract.question,
                        item.aliases,
                        item.business_name or item.output_alias or item.column,
                    )
                )
                for item in contract.selected_fields
                if str(item.output_alias or item.column or "").strip()
            }
        )
        metric_resolution = (
            {
                "requestedMetricRef": first_metric.semantic_ref_id,
                "metricKey": first_metric.metric_key,
                "displayName": (
                    first_metric.requested_phrase
                    or first_metric.business_name
                    or first_metric.metric_key
                ),
                "ownerTable": first_metric.table,
                "semanticRefId": first_metric.semantic_ref_id,
                "formula": first_metric.formula,
                "sourceColumns": list(first_metric.source_columns),
                "resolutionSource": "grounded_core_sql_candidate",
                "aggregationPolicy": first_metric.aggregation_policy,
                "metricGrain": first_metric.metric_grain,
                "applicableTimeGrain": first_metric.applicable_time_grain,
                "timeColumn": first_metric.time_column,
                "timeSemantics": dict(first_metric.time_semantics),
                "unit": first_metric.unit,
                "bindingType": first_metric.binding_type,
            }
            if first_metric
            else {}
        )
        if output_labels:
            metric_resolution["sourceColumnLabels"] = output_labels
        intent = QuestionIntent(
            question=contract.question,
            intent_type=IntentType.VALID,
            answer_mode=answer_mode,
            plan_task_id=task_id,
            task_role=TaskRole.ANCHOR,
            preferred_table=primary_table,
            metric_column=(
                first_metric.source_columns[0]
                if first_metric and first_metric.source_columns
                else ""
            ),
            metric_name=first_metric.metric_key if first_metric else "",
            metric_formula=first_metric.formula if first_metric else "",
            metric_specs=metric_specs,
            group_by_column=(
                grouped_dimensions[0].column
                if len(grouped_dimensions) == 1
                else ""
            ),
            group_by_name=(
                grouped_dimensions[0].business_name
                if len(grouped_dimensions) == 1
                else ""
            ),
            entity_reference=(
                entity_obligations[0].reference.model_copy(deep=True)
                if entity_obligations
                else EntityReference()
            ),
            days=int(contract.time_range.days or 0),
            limit=(
                int(contract.ranking.limit or 0)
                if contract.ranking.enabled
                else 100 if answer_mode == AnswerMode.DETAIL else 20
            ),
            required_evidence=output_columns,
            output_keys=output_columns,
            knowledge_ref_ids=list(contract.evidence_refs),
            analysis_source="grounded_core_sql_candidate",
            analysis_note=(
                "Core authored complete SQL; harness validated semantic lineage and obligations"
            ),
            sql_strategy="core_llm_grounded_sql",
            sql=validation.canonical_sql,
            metric_resolution=metric_resolution,
            time_range=contract.time_range.model_copy(deep=True),
        )
        return QueryPlan(
            intents=[intent],
            final_required_evidence=output_columns,
            final_evidence_column_hints={task_id: output_columns},
            entity_filter_obligations=entity_obligations,
            agent_trace=[
                "planner=single_core_progressive_grounding",
                "sql_author=core_llm",
                "planner_llm_calls=0",
            ],
            question_understanding={
                "source": "grounded_query_contract",
                "contractVersion": contract.contract_version,
                "queryShape": contract.query_shape,
                "executionMode": GroundedExecutionMode.CORE_SQL_REQUIRED.value,
                "semanticSelectionRefs": list(contract.evidence_refs),
                "sqlAstFingerprint": validation.ast_fingerprint,
                "outputLineage": dict(validation.output_lineage),
            },
            compiler_trace=[
                "CORE_SQL_ACCEPTED:%s" % validation.ast_fingerprint[:16],
                "NO_SQL_TEMPLATE_COMPILATION",
            ],
            planner_loaded_refs=list(contract.evidence_refs),
        )

    @staticmethod
    def _apply_execution_policy(attempt: GroundedRuntimeAttempt) -> None:
        decision = evaluate_deterministic_execution(attempt.contract)
        attempt.fast_path_eligible = decision.eligible
        attempt.fast_path_reason_codes = list(decision.reason_codes)
        attempt.fast_path_reason_details = dict(decision.reason_details)
        attempt.execution_mode = decision.execution_mode
        attempt.execution_reason_codes = list(decision.execution_reason_codes)
        if attempt.execution_mode == GroundedExecutionMode.UNDECIDED:
            attempt.next_action = "RESOLVE_CONTRACT"
            return
        if attempt.execution_mode in DETERMINISTIC_EXECUTION_MODES:
            attempt.next_action = "ACTIVATE_DETERMINISTIC_QUERY"
            return
        attempt.next_action = "SUBMIT_GROUNDED_SQL_CANDIDATE"

    @staticmethod
    def _activate_scope(
        session: GroundedRuntimeSession,
        *,
        attempt: GroundedRuntimeAttempt,
        contract: GroundedQueryContract,
        generation: int,
    ) -> None:
        """Atomically activate semantic authority without inventing query SQL."""

        session.active_contract = contract
        session.active_execution_mode = GroundedExecutionMode.CORE_SQL_REQUIRED
        session.active_execution_reason_codes = list(attempt.execution_reason_codes)
        session.active_pack = None
        session.active_plan = None
        session.active_preparation = None
        session.active_sql_candidate = None
        session.active_sql_validation = None
        session.sql_execution_repair_context = {}
        session.active_attempt_id = attempt.attempt_id
        session.active_generation = generation
        session.run_result = None
        GroundedRuntimeKernel._reset_publication_state(session)
        session.verified_evidence = None
        GroundedRuntimeKernel._clear_answer_snapshot(session)
        session.clarification = None
        session.phase = "ACTIVE_CORE_SQL_REQUIRED"
        session.revision += 1

    @staticmethod
    def _clear_active_sql_candidate(session: GroundedRuntimeSession) -> None:
        """Invalidate every executable artifact after the latest SQL is rejected."""

        session.active_pack = None
        session.active_plan = None
        session.active_preparation = None
        session.active_sql_candidate = None
        session.active_sql_validation = None
        session.run_result = None
        GroundedRuntimeKernel._reset_publication_state(session)
        session.verified_evidence = None
        GroundedRuntimeKernel._clear_answer_snapshot(session)

    @staticmethod
    def _clear_answer_snapshot(session: GroundedRuntimeSession) -> None:
        session.answer = ""
        session.answer_plan = None
        session.answer_run_result = None
        session.answer_verified_evidence = None
        session.answer_artifact_ids = []

    @staticmethod
    def _reset_publication_state(session: GroundedRuntimeSession) -> None:
        session.publication_authority_run_result = None
        session.publication_authority_fingerprint = ""
        session.pending_query_publications = []
        session.artifact_publication_required = False

    @staticmethod
    def _active_snapshot(
        session: GroundedRuntimeSession,
    ) -> tuple[int, QueryPlan, PlanningAssetPack]:
        if (
            not session.active_generation
            or session.active_plan is None
            or session.active_pack is None
            or session.active_contract is None
        ):
            raise RuntimeError("grounded runtime has no active compiled Contract")
        return (
            session.active_generation,
            session.active_plan.model_copy(deep=True),
            session.active_pack.model_copy(deep=True),
        )

    @staticmethod
    def _require_generation(session: GroundedRuntimeSession, generation: int) -> None:
        if session.active_generation != generation:
            raise RuntimeError(
                "active grounded Contract changed while the operation was running; result discarded"
            )

    @staticmethod
    def _event(
        session: GroundedRuntimeSession,
        stage: str,
        status: str,
        detail: str = "",
        attempt_id: str = "",
    ) -> None:
        session.events.append(
            GroundedRuntimeEvent(
                sequence=len(session.events) + 1,
                stage=stage,
                status=status,
                detail=str(detail or "")[:1000],
                attempt_id=attempt_id,
            )
        )


def _stable_json_hash(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _query_run_result_fingerprint(run_result: AgentRunResult) -> str:
    return _stable_json_hash(
        run_result.model_dump(by_alias=True, mode="json")
    )


def verified_query_artifact_integrity_fingerprint(
    artifact: GroundedVerifiedQueryArtifact,
) -> str:
    """Return the complete content seal for a verified query ledger item.

    ``ledger_fingerprint`` is deliberately excluded from its own seal.  The
    declared run-result fingerprint and the observed payload fingerprint are
    both included so mutating nested rows cannot be hidden by leaving the
    declaration unchanged.
    """

    return _stable_json_hash(
        {
            "artifactId": artifact.artifact_id,
            "generation": artifact.generation,
            "attemptId": artifact.attempt_id,
            "contractFingerprint": artifact.contract_fingerprint,
            "observedContractFingerprint": (
                grounded_query_contract_fingerprint(artifact.contract)
            ),
            "sqlFingerprint": artifact.sql_fingerprint,
            "planFingerprint": artifact.plan_fingerprint,
            "observedPlanFingerprint": _stable_json_hash(
                artifact.plan.model_dump(by_alias=True, mode="json")
            ),
            "runResultFingerprint": artifact.run_result_fingerprint,
            "observedRunResultFingerprint": _query_run_result_fingerprint(
                artifact.run_result
            ),
            "semanticActivationFingerprint": (
                artifact.semantic_activation_fingerprint
            ),
            "semanticActivationSealFingerprint": (
                artifact.semantic_activation_seal_fingerprint
            ),
            "semanticActivationTopics": list(
                artifact.semantic_activation_topics
            ),
            "queryProofVersion": artifact.query_proof_version,
            "queryProofFingerprint": artifact.query_proof_fingerprint,
            "executedSqlAstFingerprint": (
                artifact.executed_sql_ast_fingerprint
            ),
            "verifiedEvidenceFingerprint": _stable_json_hash(
                artifact.verified_evidence.model_dump(
                    by_alias=True,
                    mode="json",
                )
            ),
            "executionMode": artifact.execution_mode,
            "sqlValidation": (
                artifact.sql_validation.model_dump(
                    by_alias=True,
                    mode="json",
                )
                if artifact.sql_validation is not None
                else None
            ),
            "rankingSemanticsVerified": (
                artifact.ranking_semantics_verified
            ),
            "outputColumns": list(artifact.output_columns),
            "outputSemanticRefs": dict(artifact.output_semantic_refs),
            "outputEntityIdentities": dict(
                artifact.output_entity_identities
            ),
            "outputLineage": dict(artifact.output_lineage),
            "sealedEntityValues": dict(artifact.sealed_entity_values),
            "sealedEntityValuesTruncated": bool(
                artifact.sealed_entity_values_truncated
            ),
            "publicationStatus": artifact.publication_status,
            "resultArtifactReceipts": [
                dict(item) for item in artifact.result_artifact_receipts
            ],
            "trustedDescriptor": (
                artifact.trusted_descriptor.model_dump(
                    by_alias=True,
                    mode="json",
                )
                if artifact.trusted_descriptor is not None
                else None
            ),
        }
    )


def verified_query_artifact_integrity_valid(
    artifact: GroundedVerifiedQueryArtifact,
) -> bool:
    """Fail closed unless a ledger item matches its complete content seal."""

    declared = str(artifact.ledger_fingerprint or "").strip()
    return bool(
        declared
        and declared
        == verified_query_artifact_integrity_fingerprint(artifact)
    )


def _contract_output_columns(contract: GroundedQueryContract) -> list[str]:
    return _dedupe(
        [item.metric_key for item in contract.metrics]
        + [
            item.output_alias or item.column
            for item in contract.selected_fields
        ]
        + [
            item.column
            for item in contract.dimensions
            if item.usage == "group_by"
        ]
    )


def _observed_output_columns(run_result: AgentRunResult) -> list[str]:
    columns: list[str] = []
    for row in run_result.merged_query_bundle.rows:
        for key in row:
            name = str(key or "").strip()
            if name and not name.startswith("__") and name not in columns:
                columns.append(name)
    return columns


def _contract_output_semantics(
    contract: GroundedQueryContract,
) -> tuple[dict[str, str], dict[str, str]]:
    semantic_refs: dict[str, str] = {}
    entity_identities: dict[str, str] = {}
    for metric in contract.metrics:
        if metric.metric_key:
            semantic_refs[metric.metric_key] = metric.semantic_ref_id
    for dimension in contract.dimensions:
        if dimension.usage != "group_by" or not dimension.column:
            continue
        semantic_refs[dimension.column] = dimension.semantic_ref_id
        if dimension.entity_identity:
            entity_identities[dimension.column] = dimension.entity_identity
    for field in contract.selected_fields:
        output = field.output_alias or field.column
        if not output:
            continue
        semantic_refs[output] = field.semantic_ref_id
        if field.entity_identity:
            entity_identities[output] = field.entity_identity
    return semantic_refs, entity_identities


def _namespace_task_id(artifact_id: str, task_id: str) -> str:
    value = str(task_id or "").strip()
    if not value:
        return value
    return "%s::%s" % (artifact_id, value)


def _namespace_artifact_plan(plan: QueryPlan, artifact_id: str) -> QueryPlan:
    result = plan.model_copy(deep=True)
    for intent in result.intents:
        intent.plan_task_id = _namespace_task_id(
            artifact_id,
            intent.plan_task_id,
        )
        intent.depends_on_task_ids = [
            _namespace_task_id(artifact_id, item)
            for item in intent.depends_on_task_ids
        ]
    for dependency in result.dependencies:
        dependency.anchor_task_id = _namespace_task_id(
            artifact_id,
            dependency.anchor_task_id,
        )
        dependency.dependent_task_id = _namespace_task_id(
            artifact_id,
            dependency.dependent_task_id,
        )
    for obligation in result.entity_filter_obligations:
        obligation.task_id = _namespace_task_id(
            artifact_id,
            obligation.task_id,
        )
        if obligation.obligation_id:
            obligation.obligation_id = _namespace_task_id(
                artifact_id,
                obligation.obligation_id,
            )
    for obligation in result.semantic_filter_obligations:
        obligation.task_id = _namespace_task_id(
            artifact_id,
            obligation.task_id,
        )
        obligation.node_id = _namespace_task_id(
            artifact_id,
            obligation.node_id,
        )
        if obligation.obligation_id:
            obligation.obligation_id = _namespace_task_id(
                artifact_id,
                obligation.obligation_id,
            )
    result.final_evidence_column_hints = {
        _namespace_task_id(artifact_id, task_id): list(columns)
        for task_id, columns in result.final_evidence_column_hints.items()
    }
    result.evidence_contracts = [
        {
            **dict(contract),
            **(
                {
                    "taskId": _namespace_task_id(
                        artifact_id,
                        str(contract.get("taskId") or ""),
                    )
                }
                if str(contract.get("taskId") or "").strip()
                else {}
            ),
            **(
                {
                    "task_id": _namespace_task_id(
                        artifact_id,
                        str(contract.get("task_id") or ""),
                    )
                }
                if str(contract.get("task_id") or "").strip()
                else {}
            ),
        }
        for contract in result.evidence_contracts
    ]
    understanding = dict(result.question_understanding or {})
    understanding["verifiedQueryArtifactId"] = artifact_id
    result.question_understanding = understanding
    result.agent_trace.append("VERIFIED_QUERY_ARTIFACT:%s" % artifact_id)
    return result


def _annotate_artifact_bundle(
    bundle: QueryBundle,
    artifact_id: str,
) -> QueryBundle:
    result = bundle.model_copy(deep=True)
    result.rows = [
        {
            **dict(row),
            "__evidenceArtifactId": artifact_id,
        }
        for row in result.rows
    ]
    result.runtime_events = [
        *result.runtime_events,
        {
            "event": "verified_evidence_portfolio.member",
            "artifactId": artifact_id,
        },
    ]
    return result


def _namespace_artifact_run(
    run_result: AgentRunResult,
    artifact_id: str,
) -> AgentRunResult:
    result = run_result.model_copy(deep=True)
    for task in result.tasks:
        task.task_id = _namespace_task_id(artifact_id, task.task_id)
        task.depends_on = [
            _namespace_task_id(artifact_id, item)
            for item in task.depends_on
        ]
        for dependency in task.plan_dependencies:
            dependency.anchor_task_id = _namespace_task_id(
                artifact_id,
                dependency.anchor_task_id,
            )
            dependency.dependent_task_id = _namespace_task_id(
                artifact_id,
                dependency.dependent_task_id,
            )
    for task_result in result.task_results:
        task_result.task_id = _namespace_task_id(
            artifact_id,
            task_result.task_id,
        )
        task_result.query_bundle = _annotate_artifact_bundle(
            task_result.query_bundle,
            artifact_id,
        )
        task_result.node_plan_contract.task_id = _namespace_task_id(
            artifact_id,
            task_result.node_plan_contract.task_id,
        )
        task_result.node_task_profile.task_id = _namespace_task_id(
            artifact_id,
            task_result.node_task_profile.task_id,
        )
        task_result.entity_filter_verification.task_id = _namespace_task_id(
            artifact_id,
            task_result.entity_filter_verification.task_id,
        )
        _refresh_namespaced_entity_filter_proof(task_result)
        task_result.semantic_filter_verification.task_id = _namespace_task_id(
            artifact_id,
            task_result.semantic_filter_verification.task_id,
        )
        if task_result.entity_set is not None:
            task_result.entity_set.task_id = _namespace_task_id(
                artifact_id,
                task_result.entity_set.task_id,
            )
        for report in task_result.freshness_reports:
            report.task_id = _namespace_task_id(
                artifact_id,
                report.task_id,
            )
    result.query_bundles = [
        _annotate_artifact_bundle(item, artifact_id)
        for item in result.query_bundles
    ]
    result.merged_query_bundle = _annotate_artifact_bundle(
        result.merged_query_bundle,
        artifact_id,
    )
    for item in result.node_plan_contracts:
        item.task_id = _namespace_task_id(artifact_id, item.task_id)
    for item in result.node_task_profiles:
        item.task_id = _namespace_task_id(artifact_id, item.task_id)
    for item in result.freshness_reports:
        item.task_id = _namespace_task_id(artifact_id, item.task_id)
    for fact in result.verified_facts:
        fact.task_id = _namespace_task_id(artifact_id, fact.task_id)
        if fact.fact_id:
            fact.fact_id = _namespace_task_id(artifact_id, fact.fact_id)
    return result


def _refresh_namespaced_entity_filter_proof(task_result: Any) -> None:
    """Upgrade legacy task-bound proofs after portfolio namespacing."""

    contract = task_result.node_plan_contract
    proof = task_result.entity_filter_verification
    obligations = [
        item
        for item in contract.entity_filter_obligations
        if item.required and item.status == "bound"
    ]
    if not obligations or proof.status == "not_required" or not proof.contract_hash:
        return
    new_contract_hash = entity_filter_contract_hash(contract)
    if proof.contract_hash == new_contract_hash:
        return
    requested = {
        value
        for obligation in obligations
        for value in canonical_entity_values(
            obligation.reference.values,
            entity_comparison_policy(obligation.reference),
        )
    }
    old_requested_hashes = sorted(
        entity_value_hash(value, proof.contract_hash)
        for value in requested
    )
    if old_requested_hashes != sorted(proof.requested_value_hashes):
        return
    proof.contract_hash = new_contract_hash
    proof.requested_value_hashes = sorted(
        entity_value_hash(value, new_contract_hash)
        for value in requested
    )


def _combine_artifact_plans(
    plans: Sequence[QueryPlan],
    artifact_ids: Sequence[str],
) -> QueryPlan:
    combined = plans[0].model_copy(deep=True)
    for plan in plans[1:]:
        combined.intents.extend(item.model_copy(deep=True) for item in plan.intents)
        combined.dependencies.extend(
            item.model_copy(deep=True) for item in plan.dependencies
        )
        combined.knowledge_requests.extend(
            item.model_copy(deep=True) for item in plan.knowledge_requests
        )
        combined.evidence_contracts.extend(
            dict(item) for item in plan.evidence_contracts
        )
        combined.clarification_needs.extend(plan.clarification_needs)
        combined.final_required_evidence.extend(plan.final_required_evidence)
        combined.final_evidence_column_hints.update(
            {
                key: list(value)
                for key, value in plan.final_evidence_column_hints.items()
            }
        )
        combined.semantic_filter_obligations.extend(
            item.model_copy(deep=True)
            for item in plan.semantic_filter_obligations
        )
        combined.entity_filter_obligations.extend(
            item.model_copy(deep=True)
            for item in plan.entity_filter_obligations
        )
        combined.agent_trace.extend(plan.agent_trace)
        combined.compiler_trace.extend(plan.compiler_trace)
        combined.planner_loaded_refs.extend(plan.planner_loaded_refs)
    combined.final_required_evidence = _dedupe(
        combined.final_required_evidence
    )
    combined.planner_loaded_refs = _dedupe(combined.planner_loaded_refs)
    combined.question_understanding = {
        **dict(combined.question_understanding or {}),
        "source": "verified_evidence_portfolio",
        "verifiedQueryArtifactIds": list(artifact_ids),
        "artifactCount": len(artifact_ids),
    }
    combined.agent_trace.append(
        "VERIFIED_EVIDENCE_PORTFOLIO:%d" % len(artifact_ids)
    )
    return combined


def _combine_verified_evidence(
    items: Sequence[VerifiedEvidence],
) -> VerifiedEvidence:
    return VerifiedEvidence(
        passed=bool(items) and all(item.passed for item in items),
        covered_evidence=_dedupe(
            value
            for item in items
            for value in item.covered_evidence
        ),
        derived_evidence=[
            dict(value)
            for item in items
            for value in item.derived_evidence
        ],
        gaps=[
            value.model_copy(deep=True)
            for item in items
            for value in item.gaps
        ],
        blocking_gaps=[
            value.model_copy(deep=True)
            for item in items
            for value in item.blocking_gaps
        ],
        warning_gaps=[
            value.model_copy(deep=True)
            for item in items
            for value in item.warning_gaps
        ],
        answer_guard_required=any(
            item.answer_guard_required for item in items
        ),
        required_disclosures=_dedupe(
            value
            for item in items
            for value in item.required_disclosures
        ),
        partial_answer_reason="; ".join(
            _dedupe(item.partial_answer_reason for item in items)
        ),
    )


def _combine_artifact_runs(
    runs: Sequence[AgentRunResult],
    artifact_ids: Sequence[str],
    verified: VerifiedEvidence,
) -> AgentRunResult:
    combined = AgentRunResult()
    for run in runs:
        combined.tasks.extend(item.model_copy(deep=True) for item in run.tasks)
        combined.task_results.extend(
            item.model_copy(deep=True) for item in run.task_results
        )
        combined.query_bundles.extend(
            item.model_copy(deep=True) for item in run.query_bundles
        )
        combined.sql_repairs.extend(
            item.model_copy(deep=True) for item in run.sql_repairs
        )
        combined.evidence_gaps.extend(
            item.model_copy(deep=True) for item in run.evidence_gaps
        )
        combined.reflection_notes.extend(run.reflection_notes)
        combined.node_tool_traces.extend(
            item.model_copy(deep=True) for item in run.node_tool_traces
        )
        combined.node_task_profiles.extend(
            item.model_copy(deep=True) for item in run.node_task_profiles
        )
        combined.freshness_reports.extend(
            item.model_copy(deep=True) for item in run.freshness_reports
        )
        combined.node_plan_contracts.extend(
            item.model_copy(deep=True) for item in run.node_plan_contracts
        )
        combined.skill_lifecycle_records.extend(
            item.model_copy(deep=True)
            for item in run.skill_lifecycle_records
        )
        combined.verified_facts.extend(
            item.model_copy(deep=True) for item in run.verified_facts
        )
    rows = [
        dict(row)
        for run in runs
        for row in run.merged_query_bundle.rows
    ]
    tables = _dedupe(
        table
        for run in runs
        for table in run.merged_query_bundle.tables
    )
    combined.merged_query_bundle = QueryBundle(
        tables=tables,
        rows=rows,
        original_row_count=sum(
            run.merged_query_bundle.effective_row_count()
            for run in runs
        ),
        is_truncated=any(
            run.merged_query_bundle.is_truncated for run in runs
        ),
        source_artifact_refs={
            artifact_id: [artifact_id]
            for artifact_id in artifact_ids
        },
        runtime_events=[
            {
                "event": "verified_evidence_portfolio.composed",
                "artifactIds": list(artifact_ids),
                "artifactCount": len(artifact_ids),
            }
        ],
    )
    combined.verified_evidence = verified.model_copy(deep=True)
    combined.partial_answer_reason = verified.partial_answer_reason
    combined.executed_query_graph_fingerprint = _stable_json_hash(
        list(artifact_ids)
    )
    return combined


def _dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _dedupe_contract_gaps(
    gaps: Iterable[GroundedContractGap],
) -> list[GroundedContractGap]:
    result: list[GroundedContractGap] = []
    seen: set[str] = set()
    for gap in gaps:
        identity = _stable_json_hash(
            gap.model_dump(by_alias=True, mode="json")
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(gap)
    return result


def _valid_sha256_hex(value: Any) -> bool:
    normalized = str(value or "").strip()
    return bool(
        len(normalized) == 64
        and all(
            character in "0123456789abcdef"
            for character in normalized
        )
    )


def _merge_recall_bundles(primary: RecallBundle, secondary: RecallBundle) -> RecallBundle:
    by_ref: dict[str, Any] = {}
    for item in [*primary.items, *secondary.items]:
        ref_id = str(
            (item.metadata or {}).get("semanticRefId")
            or item.doc_id
            or ""
        ).strip()
        if ref_id and ref_id not in by_ref:
            by_ref[ref_id] = item
    items = list(by_ref.values())
    return RecallBundle(
        items=items,
        top_score=max([float(item.fusion_score or 0.0) for item in items] or [0.0]),
        merged_context="\n\n".join(
            "召回片段 [%s] %s\n%s"
            % (item.source_type, item.title, item.content[:1200])
            for item in items
        ),
    )
