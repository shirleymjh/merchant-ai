from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic import Field

from merchant_ai.models import (
    APIModel,
    AgentRunResult,
    ClarificationRequest,
    ExtractedKeywords,
    KnowledgeRetrievalRequest,
    MerchantInfo,
    PlanningAssetPack,
    QueryPlan,
    RecallBundle,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedQueryContractBuilder,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.routing import KeywordExtractService, TopicRouterService


class GroundedRuntimeAttempt(APIModel):
    attempt_id: str
    contract: GroundedQueryContract
    status: str = "PROPOSED"
    compile_status: str = "NOT_ATTEMPTED"
    activated: bool = False
    active_generation: int = 0
    error: str = ""
    validation_gaps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""


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
    active_contract: Optional[GroundedQueryContract] = None
    active_pack: Optional[PlanningAssetPack] = None
    active_plan: Optional[QueryPlan] = None
    active_preparation: Any = None
    run_result: Optional[AgentRunResult] = None
    verified_evidence: Optional[VerifiedEvidence] = None
    answer: str = ""
    clarification: Optional[ClarificationRequest] = None
    events: list[GroundedRuntimeEvent] = Field(default_factory=list)


class GroundedRuntimeKernel:
    """Independent runtime for the progressively grounded query path.

    The kernel owns state transitions but delegates domain work to narrow
    services.  It deliberately has no graph/workflow dependency: a candidate
    Contract is materialized and compiled in isolation, and only a READY
    Contract with a valid execution preparation may atomically replace the
    active query.
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

    def compile_candidate(
        self,
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        """Compile a candidate transactionally and activate only on success."""

        with self._lock:
            attempt = self._attempt(session, attempt_id)
            contract = attempt.contract.model_copy(deep=True)
            if not contract.ready or contract.status != "READY":
                attempt.compile_status = "SKIPPED_NOT_READY"
                attempt.error = "candidate Contract is not READY"
                self._event(
                    session,
                    "compile_candidate",
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
            session.active_pack = candidate_pack
            session.active_plan = candidate_plan.model_copy(deep=True)
            session.active_preparation = candidate_preparation
            session.active_attempt_id = attempt_id
            session.active_generation = next_generation
            session.run_result = None
            session.verified_evidence = None
            session.answer = ""
            session.clarification = None
            session.phase = "ACTIVE_COMPILED"
            session.revision += 1
            attempt.compile_status = "VALID"
            attempt.activated = True
            attempt.active_generation = next_generation
            self._event(
                session,
                "compile_candidate",
                "ACTIVATED",
                "generation=%d" % next_generation,
                attempt_id,
            )
            return attempt

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        *,
        knowledge_context: str = "",
        run_id: str = "",
    ) -> AgentRunResult:
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
        with self._lock:
            self._require_generation(session, generation)
            session.active_plan = runtime_preparation.plan.model_copy(deep=True)
            session.active_preparation = runtime_preparation
            session.run_result = run_result
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
