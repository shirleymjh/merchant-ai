from __future__ import annotations

import hashlib
import uuid
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
    EntityFilterObligation,
    EntityReference,
    ExtractedKeywords,
    IntentType,
    KnowledgeRetrievalRequest,
    MerchantInfo,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
    TaskRole,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedQueryContractBuilder,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.grounded_execution_policy import (
    GroundedExecutionMode,
    GroundedExecutionReason,
    evaluate_single_metric_fast_path,
)
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlCandidate,
    GroundedSqlCandidateValidator,
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.routing import KeywordExtractService, TopicRouterService


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
    answer: str = ""
    clarification: Optional[ClarificationRequest] = None
    events: list[GroundedRuntimeEvent] = Field(default_factory=list)


class GroundedRuntimeKernel:
    """Independent runtime for the progressively grounded query path.

    The kernel owns state transitions but delegates domain work to narrow
    services.  It deliberately has no graph/workflow dependency: a candidate
    Contract is routed by a generic execution policy. Only an extremely simple
    published scalar metric may use deterministic compilation. Every other
    READY Contract activates only its grounded semantic scope and waits for a
    Core-authored SQL candidate.
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
        self.compiler = compiler or compile_grounded_query
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
        )

    def route_topic(self, session: GroundedRuntimeSession) -> TopicRoutingDecision:
        keywords = self.keyword_service.extract(session.question)
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
        binding_hints: dict[str, Any] | None = None,
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
        contract = self.contract_builder.build(
            session.question,
            selected_topics,
            list(core_semantic_evidence),
            binding_hints=binding_hints,
            timezone_name=timezone_name,
            now=now,
            default_days=default_days,
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

    def activate_contract(
        self,
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        """Activate a READY candidate according to its execution authority.

        This method may invoke deterministic compilation only for
        ``DETERMINISTIC_METRIC``.
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
            if attempt.execution_mode != GroundedExecutionMode.DETERMINISTIC_METRIC:
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
            session.active_execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
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
            session.answer = ""
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
        validation = self.sql_candidate_validator.validate(candidate, contract)
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
            session.answer = ""
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

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        *,
        knowledge_context: str = "",
        run_id: str = "",
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
        run_result = execute_contract(
            session.merchant_id,
            contract,
            runtime_preparation.plan,
            pack,
            session.question,
            run_id=run_id,
            access_role=session.access_role,
            user_scope=dict(session.user_scope),
            execution_preparation=runtime_preparation,
        )
        if not isinstance(run_result, AgentRunResult):
            run_result = AgentRunResult.model_validate(run_result)
        access_terminal_codes = {
            "ACCESS_DENIED",
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
            session.answer = ""
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

    def compose_answer(
        self,
        session: GroundedRuntimeSession,
        *,
        knowledge_context: str = "",
        analysis_summary: str = "",
        allow_llm: bool = True,
        rule_context: str = "",
        personalization_context: dict[str, Any] | None = None,
    ) -> str:
        if self.answer_composer is None:
            raise RuntimeError(
                "grounded answer composer is not configured; refusing workflow fallback"
            )
        with self._lock:
            generation, plan, _pack = self._active_snapshot(session)
            if session.run_result is None or session.verified_evidence is None:
                raise RuntimeError("grounded answer requires executed and verified evidence")
            run_result = session.run_result.model_copy(deep=True)
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
        )
        with self._lock:
            self._require_generation(session, generation)
            session.answer = str(answer or "")
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
            metric_resolution=(
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
            ),
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
        decision = evaluate_single_metric_fast_path(attempt.contract)
        attempt.fast_path_eligible = decision.eligible
        attempt.fast_path_reason_codes = list(decision.reason_codes)
        attempt.fast_path_reason_details = dict(decision.reason_details)
        if not attempt.contract.ready or attempt.contract.status != "READY":
            attempt.execution_mode = GroundedExecutionMode.UNDECIDED
            attempt.execution_reason_codes = [
                GroundedExecutionReason.CONTRACT_NOT_READY.value
            ]
            attempt.next_action = "RESOLVE_CONTRACT"
            return
        if decision.eligible:
            attempt.execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
            attempt.execution_reason_codes = [
                GroundedExecutionReason.SINGLE_METRIC_FAST_PATH_ELIGIBLE.value
            ]
            attempt.next_action = "ACTIVATE_DETERMINISTIC_METRIC"
            return
        attempt.execution_mode = GroundedExecutionMode.CORE_SQL_REQUIRED
        attempt.execution_reason_codes = [
            GroundedExecutionReason.COMPLEX_QUERY_REQUIRES_CORE_SQL.value,
            *decision.reason_codes,
        ]
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
        session.answer = ""
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
        session.answer = ""

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
