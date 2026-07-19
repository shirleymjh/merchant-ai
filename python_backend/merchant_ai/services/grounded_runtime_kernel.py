from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import Field

from merchant_ai.models import (
    APIModel,
    AgentRunResult,
    AnswerMode,
    ClarificationRequest,
    DataSnapshotContract,
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
    GroundedUpstreamEntityBinding,
    compile_deterministic_grounded_query,
    materialize_grounded_asset_pack,
    requested_semantic_label,
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
from merchant_ai.services.routing import KeywordExtractService, TopicRouterService
from merchant_ai.services.grounded_rule_artifact import (
    GroundedVerifiedRuleArtifact,
    render_verified_rule_answer,
)


class GroundedRuntimeAttempt(APIModel):
    attempt_id: str
    contract: GroundedQueryContract
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
    created_at: str = ""


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
    created_at: str = ""


@dataclass(frozen=True)
class GroundedCoreSqlPreparation:
    """Accepted Core SQL plus the evidence-only plan used downstream."""

    plan: QueryPlan
    candidate_id: str
    sql_candidate: GroundedSqlCandidate
    candidate_validation: GroundedSqlValidationResult

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
    merchant_id: str
    merchant: MerchantInfo = Field(default_factory=MerchantInfo)
    access_role: str = "merchant_analyst"
    user_scope: dict[str, Any] = Field(default_factory=dict)
    reference_scope: GroundedReferenceScopeBinding = Field(
        default_factory=GroundedReferenceScopeBinding
    )
    phase: str = "CREATED"
    revision: int = 0
    keywords: ExtractedKeywords = Field(default_factory=ExtractedKeywords)
    routing: TopicRoutingDecision = Field(default_factory=TopicRoutingDecision)
    workspace_topics: list[str] = Field(default_factory=list)
    recall: RecallBundle = Field(default_factory=RecallBundle)
    attempts: list[GroundedRuntimeAttempt] = Field(default_factory=list)
    active_generation: int = 0
    active_attempt_id: str = ""
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
    rejected_sql_candidate_fingerprints: list[str] = Field(default_factory=list)
    executed_sql_candidate_fingerprints: list[str] = Field(default_factory=list)
    repair_exhausted_contract_fingerprints: list[str] = Field(default_factory=list)
    terminal_guard_code: str = ""
    run_result: Optional[AgentRunResult] = None
    verified_evidence: Optional[VerifiedEvidence] = None
    verified_query_ledger: list[GroundedVerifiedQueryArtifact] = Field(
        default_factory=list
    )
    verified_rule_ledger: list[GroundedVerifiedRuleArtifact] = Field(
        default_factory=list
    )
    verified_entity_sets: list[GroundedVerifiedEntitySet] = Field(
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
        self._lock = RLock()

    def new_session(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: MerchantInfo | None = None,
        access_role: str = "merchant_analyst",
        user_scope: dict[str, Any] | None = None,
        reference_scope: GroundedReferenceScopeBinding | dict[str, Any] | None = None,
        session_id: str = "",
    ) -> GroundedRuntimeSession:
        normalized_question = str(question or "").strip()
        if not normalized_question:
            raise ValueError("grounded runtime requires a non-empty question")
        normalized_merchant_id = str(merchant_id or "").strip()
        if not normalized_merchant_id:
            raise ValueError("grounded runtime requires merchant_id")
        return GroundedRuntimeSession(
            session_id=session_id or "grounded_%s" % uuid.uuid4().hex[:16],
            question=normalized_question,
            merchant_id=normalized_merchant_id,
            merchant=(merchant or MerchantInfo(merchant_id=normalized_merchant_id)).model_copy(
                update={"merchant_id": normalized_merchant_id}
            ),
            access_role=access_role or "merchant_analyst",
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
        branch.keywords = session.keywords.model_copy(deep=True)
        branch.routing = session.routing.model_copy(deep=True)
        branch.workspace_topics = _dedupe(
            workspace_topics or session.workspace_topics
        )
        branch.recall = session.recall.model_copy(deep=True)
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
    ) -> list[GroundedVerifiedQueryArtifact]:
        """Atomically merge successful independent branches into the parent.

        Failed or unverified branch state never mutates the parent.  The last
        successful branch becomes the active compatibility snapshot so legacy
        answer composition and a later serial entity-chain query can continue.
        """

        verified_branches = [
            branch
            for branch in branches
            if branch.verified_query_ledger
            and branch.verified_evidence is not None
            and branch.verified_evidence.passed
        ]
        if not verified_branches:
            return []
        artifacts = [
            artifact.model_copy(deep=True)
            for branch in verified_branches
            for artifact in branch.verified_query_ledger
            if artifact.verified_evidence.passed
        ]
        if not artifacts:
            return []
        with self._lock:
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
            session.active_generation += 1
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
            session.run_result = (
                last.run_result.model_copy(deep=True)
                if last.run_result is not None
                else None
            )
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
        return [item.model_copy(deep=True) for item in adopted]

    def route_topic(
        self,
        session: GroundedRuntimeSession,
        *,
        runtime_budget: Any = None,
    ) -> TopicRoutingDecision:
        keywords = self.keyword_service.extract(session.question)
        semantic_route = getattr(self.topic_router, "route_with_budget", None)
        if callable(semantic_route):
            routing = semantic_route(
                session.question,
                runtime_budget=runtime_budget,
            )
        else:
            routing = self.topic_router.route(session.question, keywords)
        categories = routing.recall_topics()
        topic_names = list(self.topic_assets.topic_names_for_categories(categories))
        with self._lock:
            session.keywords = keywords
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
        retrieval_query = str(query or session.question).strip()
        retrieval_keywords = (
            session.keywords
            if retrieval_query == session.question
            else self.keyword_service.extract(retrieval_query)
        )
        retrieve = getattr(self.recall_service, "retrieve", None)
        if callable(retrieve):
            knowledge_bundle = retrieve(
                KnowledgeRetrievalRequest(
                    query=retrieval_query,
                    keywords=list(retrieval_keywords.keywords),
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
                    strict_topic_scope=True,
                )
            )
            bundle = knowledge_bundle.recall_bundle
        else:
            bundle = self.recall_service.recall(
                retrieval_query,
                retrieval_keywords,
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
            session.phase = "NAVIGATION_RECALLED"
            self._event(
                session,
                "recall_navigation",
                "OK",
                "query=%s;items=%d" % (retrieval_query[:120], len(bundle.items)),
            )
        return bundle

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
        attempt = GroundedRuntimeAttempt(
            attempt_id="attempt_%s" % uuid.uuid4().hex[:12],
            contract=contract,
            status=contract.status,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._apply_execution_policy(attempt)
        with self._lock:
            session.attempts.append(attempt)
            session.phase = "CONTRACT_PROPOSED"
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
            session.active_attempt_id = attempt_id
            session.active_generation = next_generation
            session.run_result = None
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
        if len(prior_attempts) >= 3:
            exhausted = GroundedRuntimeSqlCandidateAttempt(
                candidate_id="sql_%s" % uuid.uuid4().hex[:12],
                active_generation=generation,
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
                status="NO_PROGRESS",
                next_action=(
                    "REVISE_BINDINGS"
                    if duplicate.next_action == "REVISE_BINDINGS"
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
                session.phase = "CORE_SQL_NO_PROGRESS"
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
                session.phase = "CORE_SQL_REPAIR_REQUIRED"
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

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        if self.executor is None:
            return DataSnapshotContract(
                unsupported_reason="GROUNDED_EXECUTOR_NOT_CONFIGURED"
            )
        capture = getattr(self.executor, "capture_data_snapshot", None)
        if not callable(capture):
            return DataSnapshotContract(
                unsupported_reason="DATA_SNAPSHOT_CAPABILITY_UNAVAILABLE"
            )
        snapshot = capture(str(semantic_activation_fingerprint or "").strip())
        if isinstance(snapshot, DataSnapshotContract):
            return snapshot
        return DataSnapshotContract.model_validate(snapshot)

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
        execution_kwargs = {
            "run_id": run_id,
            "access_role": session.access_role,
            "user_scope": dict(session.user_scope),
            "execution_preparation": runtime_preparation,
        }
        if artifact_root:
            execution_kwargs["artifact_root"] = artifact_root
            execution_kwargs["context_owner_fingerprint"] = (
                context_owner_fingerprint
            )
        # Keep non-budget callers fully backward-compatible. Grounded tools
        # explicitly pass the one shared run budget so the executor can clamp
        # Doris immediately before the repository call.
        if runtime_budget is not None:
            execution_kwargs["runtime_budget"] = runtime_budget
        if data_snapshot_contract is not None:
            execution_kwargs["data_snapshot_contract"] = data_snapshot_contract
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
            session.run_result = run_result
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
        return run_result

    def verify_active(self, session: GroundedRuntimeSession) -> VerifiedEvidence:
        if self.verifier is None:
            raise RuntimeError(
                "grounded evidence verifier is not configured; refusing unverified answering"
            )
        with self._lock:
            generation, plan, _pack = self._active_snapshot(session)
            if session.run_result is None:
                raise RuntimeError("active grounded query has not executed")
            run_result = session.run_result.model_copy(deep=True)
        verified = self.verifier.verify(session.question, plan, run_result)
        if not isinstance(verified, VerifiedEvidence):
            verified = VerifiedEvidence.model_validate(verified)
        with self._lock:
            self._require_generation(session, generation)
            session.verified_evidence = verified
            if session.run_result is not None:
                session.run_result.verified_evidence = verified.model_copy(deep=True)
            if verified.passed:
                self._record_verified_query_artifact(
                    session,
                    generation=generation,
                    plan=plan,
                    run_result=session.run_result or run_result,
                    verified=verified,
                )
            session.phase = "VERIFIED" if verified.passed else "VERIFICATION_GAPPED"
            session.revision += 1
            self._event(
                session,
                "verify_active",
                "PASSED" if verified.passed else "GAPPED",
                "blocking=%d" % len(verified.blocking_gaps),
                session.active_attempt_id,
            )
        return verified

    def latest_verified_query_artifact(
        self,
        session: GroundedRuntimeSession,
    ) -> GroundedVerifiedQueryArtifact | None:
        with self._lock:
            if not session.verified_query_ledger:
                return None
            return session.verified_query_ledger[-1].model_copy(deep=True)

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
    ) -> GroundedVerifiedQueryArtifact:
        if session.active_contract is None:
            raise RuntimeError("verified query has no active grounded Contract")
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
        sealed_values_truncated = False
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
            run_result=run_result.model_copy(deep=True),
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
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        session.verified_query_ledger.append(artifact)
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
            "SQL_TABLE_NOT_GROUNDED",
            "SQL_COLUMN_NOT_GROUNDED",
            "SQL_JOIN_SOURCE_UNRESOLVED",
            "SQL_JOIN_NOT_GOVERNED",
            "SQL_CORRELATED_RELATIONSHIP_UNPROVEN",
            "SQL_GOVERNED_FORMULA_INVALID",
        }
        if any(
            gap.blocking and gap.code in revise_binding_codes
            for gap in validation.gaps
        ):
            return "REVISE_BINDINGS"
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
        session.active_attempt_id = attempt.attempt_id
        session.active_generation = generation
        session.run_result = None
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
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


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
