from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import (
    ChatContext,
    ChatResponse,
    ClarificationRequest,
    ConversationMessage,
    PendingAnswer,
    QuestionRoute,
)
from merchant_ai.services.answer import AnswerComposeService
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.services.assets import (
    HybridRecallService,
    TopicAssetService,
)
from merchant_ai.services.checkpoints import CheckpointManager
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_deep_agent_runtime import GroundedDeepAgentRuntime
from merchant_ai.services.grounded_conversation_state import (
    GroundedConversationStateStore,
)
from merchant_ai.services.grounded_conversation_online_authority import (
    GroundedConversationOnlineAuthorityFacade,
    grounded_conversation_authority_fingerprint,
)
from merchant_ai.services.grounded_conversation_semantic_provider import (
    StructuredConversationSemanticProvider,
)
from merchant_ai.services.grounded_query_executor import GroundedQueryExecutionKernel
from merchant_ai.services.grounded_population_online_gate import (
    StructuredPopulationSemanticModelProvider,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeKernel
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.memory import create_memory_store
from merchant_ai.services.merchant_profile import MerchantProfileStore
from merchant_ai.services.repositories import (
    AnswerRepository,
    DorisRepository,
    MerchantService,
    PendingAnswerStore,
)
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    HybridKnowledgeRetrievalService,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
    PreflightUnderstandingService,
    RouteSlotExtractor,
    SemanticPreflightRouteClassifier,
    SemanticTopicRouterService,
)
from merchant_ai.services.runtime_bindings import SemanticRuntimeBindingRegistry
from merchant_ai.services.security import identity_scope_hash, identity_scope_payload


EventListener = Callable[[str, str, dict[str, Any]], None]


class GroundedOnlineRuntimeUnavailable(RuntimeError):
    """Typed fail-closed signal for an unavailable grounded data plane."""

    code = "GROUNDED_ONLINE_RUNTIME_UNAVAILABLE"

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "required model authority is unavailable")
        super().__init__("%s: %s" % (self.code, self.reason))


@dataclass(frozen=True)
class RuntimeServices:
    topic_assets: TopicAssetService
    recall_service: HybridRecallService
    knowledge_retriever: Any
    doris_repository: DorisRepository
    access_control: Any
    merchant_service: MerchantService
    answer_repository: AnswerRepository
    pending_store: PendingAnswerStore
    keyword_service: KeywordExtractService
    preflight_understanding: PreflightUnderstandingService
    answer_service: AnswerComposeService
    memory_store: Any
    merchant_profile_store: MerchantProfileStore
    recall_cache_clearers: tuple[Callable[[], None], ...]


class GroundedApplicationRuntime:
    """API-compatible facade around the independent Grounded Core runtime.

    This is deliberately a composition root, not a workflow.  It exposes the
    repositories used by the FastAPI management endpoints while online query
    answering is owned exclusively by :class:`GroundedDeepAgentRuntime`.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        core: Optional[GroundedDeepAgentRuntime],
        services: RuntimeServices,
        checkpoint_manager: CheckpointManager,
        unavailable_reason: str = "",
    ):
        self.settings = settings
        self.core = core
        self.services = services
        self._checkpoint_manager = checkpoint_manager
        self._unavailable_reason = str(unavailable_reason or "").strip()
        self._closed = False
        self.runtime_kind = "grounded_deepagent"

    def run(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Optional[EventListener] = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[Sequence[ConversationMessage]] = None,
    ) -> ChatResponse:
        effective_merchant_id = str(merchant_id or self.settings.merchant_id).strip()
        actual_thread_id = thread_id or "thread_%s" % uuid.uuid4().hex
        actual_run_id = run_id or "run_%s" % uuid.uuid4().hex
        _emit(
            listener,
            "runtime.started",
            "GROUNDED_CORE",
            {
                "runtime": self.runtime_kind,
                "threadId": actual_thread_id,
                "runId": actual_run_id,
                "messageHistoryCount": len(message_history or []),
                "preflightEnabled": True,
            },
        )
        preflight = self._preflight_understanding(
            question,
            context=context,
            message_history=message_history,
            listener=listener,
        )
        if preflight is not None:
            route = preflight.routing_decision.route
            if route == QuestionRoute.GREETING:
                semantic_route = str(
                    (preflight.semantic_trace or {}).get("route") or ""
                ).strip().upper()
                response = ChatResponse(
                    answer=(
                        "您好，我是 yshopping 商家 AI 助手，可以查询已接入的经营指标、"
                        "明细和规则信息。"
                    ),
                    category_name=(
                        "BUSINESS_CHAT"
                        if semantic_route == "BUSINESS_CHAT"
                        else "GREETING"
                    ),
                    suggestions=[
                        "查询最近7天经营指标",
                        "查看订单或退款明细",
                        "了解已发布业务规则",
                    ],
                    debug_trace={
                        "harness": {
                            "runtimePath": "preflight_fast_path",
                            "coreInvoked": False,
                        }
                    },
                )
                return self._finalize_response(
                    response,
                    question=question,
                    merchant_id=effective_merchant_id,
                    context=context,
                    listener=listener,
                    thread_id=actual_thread_id,
                    run_id=actual_run_id,
                    preflight=preflight,
                    core_invoked=False,
                )
            if route == QuestionRoute.INVALID:
                write_operation = bool(
                    (preflight.surface_signals or {}).get("writeOperation")
                )
                clarification_question = (
                    "当前 BI Agent 只支持只读查询和分析，不能执行写操作。请改成只读查询问题。"
                    if write_operation
                    else (
                        str(preflight.clarification_question or "").strip()
                        or "请补充要看的业务对象、指标、时间范围或分析目标。"
                    )
                )
                clarification = ClarificationRequest(
                    question=clarification_question,
                    stage=("UNSUPPORTED_OPERATION" if write_operation else "BUSINESS_SCOPE"),
                    type=("write_operation" if write_operation else "business_scope"),
                    options=(
                        ["改成只读查询", "取消本次操作"]
                        if write_operation
                        else [
                            "查询经营指标",
                            "查看业务明细",
                            "分析经营变化",
                        ]
                    ),
                    pending_question=str(question or ""),
                )
                response = ChatResponse(
                    answer=clarification_question,
                    category_name=(
                        "UNSUPPORTED_WRITE" if write_operation else "INVALID"
                    ),
                    clarification=clarification,
                    debug_trace={
                        "harness": {
                            "runtimePath": "preflight_fast_path",
                            "coreInvoked": False,
                        }
                    },
                )
                return self._finalize_response(
                    response,
                    question=question,
                    merchant_id=effective_merchant_id,
                    context=context,
                    listener=listener,
                    thread_id=actual_thread_id,
                    run_id=actual_run_id,
                    preflight=preflight,
                    core_invoked=False,
                )
        if self.core is None:
            reason = self._unavailable_reason or "required model authority is unavailable"
            _emit(
                listener,
                "runtime.failed",
                "GROUNDED_CORE",
                {
                    "runtime": self.runtime_kind,
                    "errorType": "GroundedOnlineRuntimeUnavailable",
                    "error": reason,
                },
            )
            raise GroundedOnlineRuntimeUnavailable(reason)
        identity = getattr(context, "user_identity", None) if context is not None else None
        user_scope = identity_scope_payload(identity, effective_merchant_id)
        access_role = load_authorization_policy().access_role_for_identity(str(user_scope.get("role") or ""))
        merchant = self.services.merchant_service.current_merchant(effective_merchant_id)

        try:
            response = self.core.run(
                question,
                effective_merchant_id,
                merchant=merchant,
                access_role=access_role,
                user_scope=user_scope,
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            listener=listener,
            request_context=context,
            message_history=list(message_history or []),
        )
        except Exception as exc:
            _emit(
                listener,
                "runtime.failed",
                "GROUNDED_CORE",
                {
                    "runtime": self.runtime_kind,
                    "errorType": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            )
            raise

        return self._finalize_response(
            response,
            question=question,
            merchant_id=effective_merchant_id,
            context=context,
            listener=listener,
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            preflight=preflight,
            core_invoked=True,
        )

    def _preflight_understanding(
        self,
        question: str,
        *,
        context: Optional[ChatContext],
        message_history: Optional[Sequence[ConversationMessage]],
        listener: Optional[EventListener],
    ) -> Any:
        service = getattr(self.services, "preflight_understanding", None)
        if service is None or not hasattr(service, "understand"):
            return None
        pending_context = bool(
            message_history
            or (
                context
                and (
                    str(context.pending_clarification_stage or "").strip()
                    or str(context.pending_clarification_type or "").strip()
                    or str(context.pending_question or "").strip()
                    or str(context.question or "").strip()
                    or str(context.topic or "").strip()
                    or bool(context.topics)
                    or bool(context.metric_keys)
                    or bool(context.dimension_keys)
                    or str(context.context_summary or "").strip()
                )
            )
        )
        try:
            return service.understand(
                question,
                pending_context=pending_context,
            )
        except Exception as exc:
            # The preflight gate is an optimization and routing guard. An
            # internal classifier failure must not incorrectly reject a valid
            # business task; the grounded Core remains the fail-safe path.
            _emit(
                listener,
                "runtime.preflight_failed",
                "PREFLIGHT_ROUTE",
                {
                    "runtime": self.runtime_kind,
                    "errorType": type(exc).__name__,
                    "error": str(exc)[:300],
                    "fallback": "GROUNDED_CORE",
                },
            )
            return None

    def _finalize_response(
        self,
        response: ChatResponse,
        *,
        question: str,
        merchant_id: str,
        context: Optional[ChatContext],
        listener: Optional[EventListener],
        thread_id: str,
        run_id: str,
        preflight: Any,
        core_invoked: bool,
    ) -> ChatResponse:
        if not response.id:
            response.id = "qa_%s" % uuid.uuid4().hex
        response_context = (context or ChatContext()).model_copy(deep=True)
        if response.clarification is not None:
            response_context.pending_clarification_stage = str(
                response.clarification.stage or ""
            )
            response_context.pending_clarification_type = str(
                response.clarification.type or ""
            )
            response_context.pending_question = str(
                response.clarification.pending_question or question
            )
            response_context.pending_clarification_options = list(
                response.clarification.options
            )
            response_context.clarification_resolved = False
        else:
            response_context.pending_clarification_stage = ""
            response_context.pending_clarification_type = ""
            response_context.pending_question = ""
            response_context.pending_clarification_options = []
        response.context = response_context
        harness = dict((response.debug_trace or {}).get("harness") or {})
        harness.update(
            {
                "runtime": self.runtime_kind,
                "threadId": thread_id,
                "runId": run_id,
                "legacyFallbackUsed": False,
                "coreInvoked": bool(core_invoked),
                "preflight": self._preflight_trace(preflight),
                "checkpoint": self._checkpoint_manager.deep_agent_ref(
                    thread_id,
                    run_id,
                ),
            }
        )
        response.debug_trace = {**dict(response.debug_trace or {}), "harness": harness}
        self._persist_verified_answer(
            response,
            question=question,
            merchant_id=merchant_id,
            context=response_context,
            thread_id=thread_id,
            run_id=run_id,
            core_invoked=core_invoked,
        )
        _emit(
            listener,
            "answer.ready" if response.clarification is None else "clarification.required",
            "GROUNDED_CORE" if core_invoked else "PREFLIGHT_ROUTE",
            {
                "runtime": self.runtime_kind,
                "answer": response.answer if response.clarification is None else "",
                "clarificationType": (
                    str(response.clarification.type) if response.clarification is not None else ""
                ),
            },
        )
        return response

    def _persist_verified_answer(
        self,
        response: ChatResponse,
        *,
        question: str,
        merchant_id: str,
        context: ChatContext,
        thread_id: str,
        run_id: str,
        core_invoked: bool,
    ) -> None:
        """Persist only an attested answer; never turn an error into a record.

        PendingAnswer remains the feedback attribution authority even when the
        MySQL repository is temporarily degraded.  Persistence is a post-answer
        side effect and therefore cannot invalidate an already verified reply.
        """

        harness = dict((response.debug_trace or {}).get("harness") or {})
        if (
            not core_invoked
            or response.clarification is not None
            or not str(response.answer or "").strip()
            or bool(harness.get("operationalFailure"))
        ):
            return
        verified_query_ids = [
            str(item).strip()
            for item in harness.get("verifiedQueryArtifactIds") or []
            if str(item).strip()
        ]
        verified_rule_ids = [
            str(item).strip()
            for item in harness.get("verifiedRuleArtifactIds") or []
            if str(item).strip()
        ]
        if not verified_query_ids and not verified_rule_ids:
            return
        identity = context.user_identity
        scope = identity_scope_payload(identity, merchant_id)
        merchant_name = str(merchant_id or "")
        try:
            merchant = self.services.merchant_service.current_merchant(merchant_id)
            merchant_name = str(getattr(merchant, "merchant_name", "") or merchant_name)
        except Exception:
            # Merchant lookup is enrichment only; the trusted merchant id is
            # already bound by the runtime and must not block answer delivery.
            pass
        tables: list[str] = []
        for section in response.data_sections or []:
            tables.extend(str(item) for item in section.doris_tables or [] if str(item).strip())
        tables.extend(str(item) for item in response.doris_tables or [] if str(item).strip())
        pending = PendingAnswer(
            id=response.id,
            question=str(question or ""),
            answer=str(response.answer or ""),
            merchant_id=str(merchant_id or ""),
            merchant_name=merchant_name,
            category_name=str(response.category_name or ""),
            doris_tables=",".join(dict.fromkeys(tables)),
            suggested_questions=json.dumps(response.suggestions or [], ensure_ascii=False),
            thread_id=str(thread_id or ""),
            user_id=str(scope.get("userId") or ""),
            identity_scope_hash=identity_scope_hash(identity, merchant_id),
            store_ids=list(scope.get("storeIds") or []),
            permissions=list(scope.get("permissions") or []),
        )
        pending_written = False
        answer_written = False
        try:
            self.services.pending_store.put(pending)
            pending_written = True
        except Exception as exc:
            harness["pendingAnswerPersistenceError"] = "%s:%s" % (
                type(exc).__name__,
                str(exc)[:300],
            )
        try:
            answer_written = bool(self.services.answer_repository.insert_answer(pending))
        except Exception as exc:
            harness["answerRepositoryPersistenceError"] = "%s:%s" % (
                type(exc).__name__,
                str(exc)[:300],
            )
        memory_written = False
        if pending_written and self.services.memory_store is not None:
            try:
                topics = [str(item.value if hasattr(item, "value") else item) for item in context.topics]
                metrics = [str(item) for item in context.metric_keys if str(item).strip()]
                days = int(context.resolved_time_window_days or context.days or 0)
                plan_intent = SimpleNamespace(
                    category=str(context.topic or (topics[0] if topics else "")),
                    metric_resolution={"metricKey": metrics[0] if metrics else ""},
                    metric_name=metrics[0] if metrics else "",
                    days=days,
                )
                memory_state = {
                    "requested_merchant_id": merchant_id,
                    "question": question,
                    "answer": response.answer,
                    "plan": SimpleNamespace(
                        intents=[plan_intent],
                        question_understanding={"analysisIntent": str(context.answer_mode or "")},
                    ),
                    "route_slots": {"timeWindow": {"days": days}},
                    "user_identity": identity,
                    "persisted": bool(answer_written or pending_written),
                    "agent_run_result": SimpleNamespace(
                        task_results=[object()],
                        verified_evidence=SimpleNamespace(passed=True),
                        evidence_gaps=[],
                    ),
                    "semantic_evidence": [],
                }
                self.services.memory_store.update_from_state(memory_state)
                memory_written = True
            except Exception as exc:
                harness["memoryPersistenceError"] = "%s:%s" % (
                    type(exc).__name__,
                    str(exc)[:300],
                )
        response.persisted = bool(answer_written)
        harness["persistence"] = {
            "pendingAnswerWritten": pending_written,
            "answerRepositoryWritten": answer_written,
            "memoryWritten": memory_written,
            "verifiedQueryArtifactIds": verified_query_ids,
            "verifiedRuleArtifactIds": verified_rule_ids,
            "feedbackPending": pending_written,
            "runId": run_id,
        }
        response.debug_trace = {**dict(response.debug_trace or {}), "harness": harness}

    @staticmethod
    def _preflight_trace(preflight: Any) -> dict[str, Any]:
        if preflight is None:
            return {
                "status": "FAILED_OPEN_TO_CORE",
                "route": "BUSINESS",
            }
        decision = getattr(preflight, "routing_decision", None)
        route = getattr(decision, "route", QuestionRoute.INVALID)
        return {
            "status": "COMPLETED",
            "route": str(getattr(route, "value", route) or ""),
            "reason": str(getattr(decision, "reason", "") or "")[:300],
            "semantic": dict(getattr(preflight, "semantic_trace", {}) or {}),
            "surfaceSignals": dict(
                getattr(preflight, "surface_signals", {}) or {}
            ),
        }

    async def run_async(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Optional[EventListener] = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[Sequence[ConversationMessage]] = None,
    ) -> ChatResponse:
        return await asyncio.to_thread(
            self.run,
            question,
            merchant_id,
            context,
            listener,
            thread_id,
            run_id,
            message_history,
        )

    def checkpoint_state_summary(self, thread_id: str, run_id: str) -> dict[str, Any]:
        if self.core is None:
            raise GroundedOnlineRuntimeUnavailable(
                self._unavailable_reason
                or "required model authority is unavailable"
            )
        config = self._checkpoint_manager.config_for_deep_agent(thread_id, run_id)
        snapshot = self.core.deep_agent_graph.get_state(config)
        values = snapshot.values if hasattr(snapshot, "values") else {}
        metadata = snapshot.metadata if hasattr(snapshot, "metadata") else {}
        tasks = snapshot.tasks if hasattr(snapshot, "tasks") else ()
        next_nodes = snapshot.next if hasattr(snapshot, "next") else ()
        return {
            "checkpointRef": self._checkpoint_manager.deep_agent_ref(thread_id, run_id),
            "metadata": metadata or {},
            "next": list(next_nodes or []),
            "taskCount": len(tasks or []),
            "valueKeys": sorted(list((values or {}).keys()))[:80]
            if isinstance(values, dict)
            else [],
            "hasValues": bool(values),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._checkpoint_manager.close()

    def runtime_trace(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_kind,
            "onlineReady": self.core is not None,
            "preflightReady": bool(
                getattr(self.services, "preflight_understanding", None)
            ),
            "unavailableReason": self._unavailable_reason,
            "planner": None,
            "nodeAgent": None,
            "dataEngine": "GroundedQueryExecutionKernel",
            "degraded": {
                "answerRepository": self.services.answer_repository.trace()
                if hasattr(self.services.answer_repository, "trace")
                else {},
                "merchantService": self.services.merchant_service.trace()
                if hasattr(self.services.merchant_service, "trace")
                else {},
                "doris": self.services.doris_repository.cache_trace()
                if hasattr(self.services.doris_repository, "cache_trace")
                else {},
            },
        }


def create_runtime(settings: Optional[Settings] = None) -> Any:
    runtime_settings = settings or get_settings()
    mode = str(runtime_settings.agent_mode or "deepagent").strip().lower()
    if mode == "deepagent":
        return create_grounded_runtime(runtime_settings)
    raise ValueError(
        "Unsupported agent_mode: %s; online query authority is grounded deepagent only"
        % mode
    )


def create_grounded_runtime(settings: Settings) -> GroundedApplicationRuntime:
    doris_repository = DorisRepository(settings)
    answer_repository = AnswerRepository(settings)
    pending_store = PendingAnswerStore(settings)
    topic_assets = TopicAssetService(settings)
    recall_service = HybridRecallService(settings, topic_assets)
    semantic_catalog = recall_service.semantic_catalog
    knowledge_retriever: Any
    if settings.es_enabled:
        knowledge_retriever = EsKnowledgeRetrievalService(settings, topic_assets)
    else:
        knowledge_retriever = HybridKnowledgeRetrievalService(recall_service)

    keyword_service = KeywordExtractService(topic_assets)
    route_slot_extractor = RouteSlotExtractor(topic_assets)
    preflight_understanding = PreflightUnderstandingService(
        settings,
        keyword_service,
        slot_extractor=route_slot_extractor,
        semantic_classifier=SemanticPreflightRouteClassifier(settings),
    )
    topic_router = SemanticTopicRouterService(settings, topic_assets)
    answer_service = AnswerComposeService(LlmClient(settings))
    merchant_service = MerchantService(
        settings,
        doris_repository,
        SemanticRuntimeBindingRegistry(settings).resolve("principal_profile"),
    )
    checkpoint_manager = CheckpointManager(settings)
    memory_store = create_memory_store(settings)
    merchant_profile_store = MerchantProfileStore(settings)

    def runtime_services(access_control: Any) -> RuntimeServices:
        return RuntimeServices(
            topic_assets=topic_assets,
            recall_service=recall_service,
            knowledge_retriever=knowledge_retriever,
            doris_repository=doris_repository,
            access_control=access_control,
            merchant_service=merchant_service,
            answer_repository=answer_repository,
            pending_store=pending_store,
            keyword_service=keyword_service,
            preflight_understanding=preflight_understanding,
            answer_service=answer_service,
            memory_store=memory_store,
            merchant_profile_store=merchant_profile_store,
            recall_cache_clearers=(
                recall_service.clear_cache,
                doris_repository.clear_cache,
                keyword_service.reload_semantic_lexicon,
            ),
        )

    population_model_name = str(
        settings.llm_balanced_model
        or settings.llm_fast_model
        or settings.openai_model
    )
    population_timeout_seconds = max(
        1,
        int(settings.llm_analysis_timeout_seconds or 0),
    )
    deep_agent_timeout_seconds = _deep_agent_timeout_seconds(settings)
    lead_model = LlmClient(settings).chat_model(
        timeout_seconds=deep_agent_timeout_seconds
    )
    population_model = LlmClient(
        settings,
        model_name=population_model_name,
    ).chat_model(timeout_seconds=population_timeout_seconds)
    if lead_model is None or population_model is None:
        # Management/control-plane APIs must remain available for health,
        # identity, uploads and governance when inference credentials are not
        # installed.  The data plane stays fail-closed: no population gate,
        # query executor or Core is constructed. Lightweight preflight routes
        # remain available, while BUSINESS tasks are rejected explicitly by
        # the facade.
        return GroundedApplicationRuntime(
            settings=settings,
            core=None,
            services=runtime_services(AccessControlService(settings)),
            checkpoint_manager=checkpoint_manager,
            unavailable_reason="required grounded model authority is not configured",
        )
    population_deployment = {
        "model": population_model_name,
        "baseUrl": str(settings.openai_base_url or "").rstrip("/"),
        "protocol": "population_semantic_reviewer.v1",
    }
    semantic_authority = (
        GroundedPopulationExecutionGate.authority_fingerprint(
            "population_semantic_reviewer",
            population_deployment,
        )
    )
    population_provider = StructuredPopulationSemanticModelProvider(
        population_model,
        authority_fingerprint=semantic_authority,
    )
    population_gate = GroundedPopulationExecutionGate(
        settings=settings,
        semantic_provider=population_provider,
        declaration_author_fingerprint=(
            GroundedPopulationExecutionGate.authority_fingerprint(
                "core_goal_declaration",
                {
                    "model": str(settings.openai_model),
                    "protocol": "grounded_core_goal_contract.v1",
                },
            )
        ),
        semantic_authority_fingerprint=semantic_authority,
        lineage_authority_fingerprint=(
            GroundedPopulationExecutionGate.authority_fingerprint(
                "validated_sql_ast_lineage",
                {"protocol": "grounded_sql_lineage.v1"},
            )
        ),
        artifact_authority_fingerprint=(
            GroundedPopulationExecutionGate.authority_fingerprint(
                "immutable_result_artifact",
                {"protocol": "grounded_result_artifact.v2"},
            )
        ),
        ledger_authority_fingerprint=(
            GroundedPopulationExecutionGate.authority_fingerprint(
                "published_query_ledger",
                {"protocol": "grounded_query_ledger.v1"},
            )
        ),
        semantic_timeout_seconds=population_timeout_seconds,
    )
    conversation_deployment = {
        "model": population_model_name,
        "baseUrl": str(settings.openai_base_url or "").rstrip("/"),
        "protocol": "conversation_semantic_resolver.v1",
    }
    conversation_semantic_authority = (
        grounded_conversation_authority_fingerprint(
            "conversation_semantic_reviewer",
            conversation_deployment,
        )
    )
    conversation_provider = StructuredConversationSemanticProvider(
        population_model,
        authority_fingerprint=conversation_semantic_authority,
    )
    conversation_online_authority = (
        GroundedConversationOnlineAuthorityFacade(
            workspace_root=settings.resolved_workspace_path,
            semantic_provider=conversation_provider,
            trusted_reviewer_authority_fingerprints=(
                conversation_semantic_authority,
            ),
            core_authority_fingerprint=(
                grounded_conversation_authority_fingerprint(
                    "grounded_core",
                    {
                        "model": str(settings.openai_model),
                        "baseUrl": str(
                            settings.openai_base_url or ""
                        ).rstrip("/"),
                        "protocol": "grounded_core_goal_contract.v1",
                    },
                )
            ),
            review_timeout_seconds=population_timeout_seconds,
        )
    )
    query_executor = GroundedQueryExecutionKernel(
        doris_repository,
        settings,
        population_execution_gate=population_gate,
    )
    kernel = GroundedRuntimeKernel(
        topic_assets,
        keyword_service=keyword_service,
        route_slot_extractor=route_slot_extractor,
        topic_router=topic_router,
        recall_service=knowledge_retriever,
        executor=query_executor,
        verifier=EvidenceVerifier(),
        answer_composer=answer_service,
    )
    core = GroundedDeepAgentRuntime(
        kernel,
        lead_model=lead_model,
        isolated_subagent_model=LlmClient(
            settings,
            model_name=str(
                settings.llm_balanced_model
                or settings.llm_fast_model
                or settings.openai_model
            ),
        ).chat_model(
            timeout_seconds=deep_agent_timeout_seconds
        ),
        semantic_catalog=semantic_catalog,
        checkpointer=checkpoint_manager.saver(),
        checkpoint_config_factory=checkpoint_manager.config_for_deep_agent,
        skill_root=str(settings.resources_root / "runtime" / "agent_skills"),
        skill_run_root=str(settings.resolved_workspace_path / "skill_runs"),
        parallel_max_workers=int(settings.tool_max_concurrency or 4),
        settings=settings,
        memory_store=memory_store,
        conversation_state_store=GroundedConversationStateStore(settings),
        conversation_online_authority=conversation_online_authority,
        population_execution_gate=population_gate,
        population_gate_enforced=True,
    )
    return GroundedApplicationRuntime(
        settings=settings,
        core=core,
        services=runtime_services(query_executor.access_control),
        checkpoint_manager=checkpoint_manager,
    )


def _deep_agent_timeout_seconds(settings: Settings) -> int:
    """Return one timeout budget for both Core and isolated LLM turns.

    DeepAgent model turns include filesystem and Skill middleware context and
    therefore must not inherit the short single-shot service timeout. Keeping
    Core and isolated subagents on the same budget also prevents the parent
    from timing out before an isolated Skill can start.
    """

    return max(
        60,
        int(settings.llm_request_timeout_seconds or 0),
        int(settings.llm_lead_timeout_seconds or 0),
        int(settings.llm_analysis_timeout_seconds or 0),
    )


def _emit(
    listener: Optional[EventListener],
    event_type: str,
    node: str,
    payload: dict[str, Any],
) -> None:
    if listener is None:
        return
    try:
        listener(event_type, node, payload)
    except Exception:
        # Observability must not become query authority or alter the Core loop.
        return
