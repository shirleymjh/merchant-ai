from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import ChatContext, ChatResponse, ConversationMessage
from merchant_ai.services.answer import AnswerComposeService
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
from merchant_ai.services.routing import KeywordExtractService, SemanticTopicRouterService
from merchant_ai.services.runtime_bindings import SemanticRuntimeBindingRegistry
from merchant_ai.services.security import identity_scope_payload


EventListener = Callable[[str, str, dict[str, Any]], None]


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
        core: GroundedDeepAgentRuntime,
        services: RuntimeServices,
        checkpoint_manager: CheckpointManager,
    ):
        self.settings = settings
        self.core = core
        self.services = services
        self._checkpoint_manager = checkpoint_manager
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
        identity = getattr(context, "user_identity", None) if context is not None else None
        user_scope = identity_scope_payload(identity, effective_merchant_id)
        access_role = _merchant_access_role(str(user_scope.get("role") or ""))
        merchant = self.services.merchant_service.current_merchant(effective_merchant_id)

        _emit(
            listener,
            "runtime.started",
            "GROUNDED_CORE",
            {
                "runtime": self.runtime_kind,
                "threadId": actual_thread_id,
                "runId": actual_run_id,
                "messageHistoryCount": len(message_history or []),
            },
        )
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
                "threadId": actual_thread_id,
                "runId": actual_run_id,
                "legacyFallbackUsed": False,
                "checkpoint": self._checkpoint_manager.deep_agent_ref(
                    actual_thread_id,
                    actual_run_id,
                ),
            }
        )
        response.debug_trace = {**dict(response.debug_trace or {}), "harness": harness}
        _emit(
            listener,
            "answer.ready" if response.clarification is None else "clarification.required",
            "GROUNDED_CORE",
            {
                "runtime": self.runtime_kind,
                "answer": response.answer if response.clarification is None else "",
                "clarificationType": (
                    str(response.clarification.type) if response.clarification is not None else ""
                ),
            },
        )
        return response

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
    pending_store = PendingAnswerStore()
    topic_assets = TopicAssetService(settings)
    recall_service = HybridRecallService(settings, topic_assets)
    semantic_catalog = recall_service.semantic_catalog
    knowledge_retriever: Any
    if settings.es_enabled:
        knowledge_retriever = EsKnowledgeRetrievalService(settings, topic_assets)
    else:
        knowledge_retriever = HybridKnowledgeRetrievalService(recall_service)

    keyword_service = KeywordExtractService(topic_assets)
    topic_router = SemanticTopicRouterService(settings, topic_assets)
    population_model_name = str(
        settings.llm_balanced_model
        or settings.llm_fast_model
        or settings.openai_model
    )
    population_timeout_seconds = max(
        1,
        int(settings.llm_analysis_timeout_seconds or 0),
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
        LlmClient(
            settings,
            model_name=population_model_name,
        ).chat_model(timeout_seconds=population_timeout_seconds),
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
        LlmClient(
            settings,
            model_name=population_model_name,
        ).chat_model(timeout_seconds=population_timeout_seconds),
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
    answer_service = AnswerComposeService(LlmClient(settings))
    merchant_service = MerchantService(
        settings,
        doris_repository,
        SemanticRuntimeBindingRegistry(settings).resolve("principal_profile"),
    )
    checkpoint_manager = CheckpointManager(settings)
    kernel = GroundedRuntimeKernel(
        topic_assets,
        keyword_service=keyword_service,
        topic_router=topic_router,
        recall_service=knowledge_retriever,
        executor=query_executor,
        verifier=EvidenceVerifier(),
        answer_composer=answer_service,
    )
    deep_agent_timeout_seconds = _deep_agent_timeout_seconds(settings)
    core = GroundedDeepAgentRuntime(
        kernel,
        lead_model=LlmClient(settings).chat_model(
            timeout_seconds=deep_agent_timeout_seconds
        ),
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
        conversation_state_store=GroundedConversationStateStore(settings),
        conversation_online_authority=conversation_online_authority,
        population_execution_gate=population_gate,
        population_gate_enforced=True,
    )
    return GroundedApplicationRuntime(
        settings=settings,
        core=core,
        services=RuntimeServices(
            topic_assets=topic_assets,
            recall_service=recall_service,
            knowledge_retriever=knowledge_retriever,
            doris_repository=doris_repository,
            access_control=query_executor.access_control,
            merchant_service=merchant_service,
            answer_repository=answer_repository,
            pending_store=pending_store,
            keyword_service=keyword_service,
            answer_service=answer_service,
            memory_store=create_memory_store(settings),
            merchant_profile_store=MerchantProfileStore(settings),
            recall_cache_clearers=(
                recall_service.clear_cache,
                doris_repository.clear_cache,
                keyword_service.reload_semantic_lexicon,
            ),
        ),
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


def _merchant_access_role(role: str) -> str:
    return {
        "platform_operator": "merchant_admin",
        "merchant_owner": "merchant_admin",
        "merchant_operator": "merchant_analyst",
        "merchant_finance": "merchant_finance",
        "merchant_customer_service": "merchant_service",
        "merchant_goods": "merchant_goods",
        "merchant_fulfillment": "merchant_fulfillment",
    }.get(str(role or ""), "merchant_analyst")


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
