from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import os
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.action_contract import action_state_flag_ready, state_path_ready
from merchant_ai.graph.policy import REPAIRABLE_QUERY_GRAPH_GAP_CODES, V2AgentPolicy
from merchant_ai.graph.query_graph_contract import (
    graph_validation_attempted,
    graph_validation_failure_reason,
    graph_validation_passed,
    invalidate_graph_validation,
    mark_graph_validation_stale,
    query_graph_fingerprint,
    query_graph_structure_fingerprint,
    record_graph_validation,
)
from merchant_ai.graph.message_history import (
    MAX_SHORT_TERM_MESSAGES,
    MAX_SHORT_TERM_CONTEXT_CHARS,
    append_context_section,
    compact_file_tool_results_for_prompt,
    normalize_message_history,
    preserve_priority_context_window,
    render_message_history_context,
)
from merchant_ai.graph.state import (
    AgentState,
    GraphEventListener,
    add_step,
    emit,
    increment_round,
    knowledge_context,
    register_event_listener,
    unregister_event_listener,
)
from merchant_ai.models import (
    ActionResult,
    AgentDecision,
    AgentActionTrace,
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    ChatContext,
    ChatDataSection,
    ChatResponse,
    ClarificationRequest,
    ContextManifest,
    ConversationMessage,
    ExecutionAttemptArtifact,
    ExtractedKeywords,
    EvidenceGap,
    FastUnderstandingResult,
    GraphValidationGap,
    GraphValidationResult,
    HypothesisEvidenceLedger,
    IntentSignals,
    IntentType,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    KnowledgeRequest,
    KnowledgeRequestType,
    MerchantRecentFocus,
    WorkspaceManifest,
    PendingAnswer,
    PlanningAssetPack,
    PlannerRepairInput,
    PlannerReflectionResult,
    PlannerRepairRequest,
    QueryBundle,
    QueryGraphRepairDelta,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    QuestionRoute,
    RecallBundle,
    RecallItem,
    RetrievalIssue,
    RouteSlots,
    SkillDraft,
    SkillLifecycleRecord,
    SkillMatchState,
    SubAgentDelegationPlan,
    SubAgentDelegationTask,
    ToolCallRequest,
    ToolCachePolicy,
    RoutingDecision,
    ThreadData,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.langgraph_compat import END, START, StateGraph
from merchant_ai.services.answer import (
    AnswerComposeService,
    answer_result_role,
    answer_skill_headers,
    build_response_context,
    boolish,
    intent_by_task_id,
    joined_categories,
    section_title_for_intent,
    select_answer_skill,
    visible_successful_tasks,
)
from merchant_ai.services.analysis_worker import AnalysisWorkerExecutor
from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.services.assets import (
    HybridRecallService,
    PlanningAssetPackBuilder,
    SemanticCatalogService,
    SkillLoader,
    TopicAssetService,
    metric_direct_match_label,
    normalize_for_match,
    recalled_metric_evidence_matches_phrase,
)
from merchant_ai.services.checkpoints import CheckpointManager
from merchant_ai.services.clarification import ClarificationResolutionService
from merchant_ai.services.context import ContextManager
from merchant_ai.services.context_assembly import (
    ContextAssembler,
    ThreadContextService,
    build_llm_context_blocks,
    context_cache_layout,
    context_quarantine_policy,
    extract_reusable_entity_sets,
)
from merchant_ai.services.context_filesystem import add_context_uri, context_lineage_record, merchant_uri_for_artifact, merchant_uri_for_semantic_ref
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_query_contract import (
    GroundedContractGap,
    GroundedQueryContract,
    GroundedQueryContractBuilder,
    GroundedQueryContractValidator,
    compile_grounded_query as compile_grounded_query_contract,
    materialize_grounded_asset_pack,
    merge_grounded_rejected_bindings,
)
from merchant_ai.services.latency import LatencyOptimizer
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.llm_recovery import (
    bounded_single_retry_count,
    classify_llm_failure,
    retry_timeout_with_answer_reserve,
)
from merchant_ai.services.knowledge_requests import (
    canonical_knowledge_request_payload,
    dedupe_knowledge_requests,
    knowledge_request_identity,
    normalize_knowledge_request_text as normalize_canonical_knowledge_request_text,
)
from merchant_ai.services.memory import (
    create_memory_store,
    memory_query_hash,
    memory_recall_trace_for,
    normalize_memory_recall_issues,
    retrieval_context_from_state,
    truncate_memory_text_by_tokens,
)
from merchant_ai.services.memory_constraints import build_memory_constraints
from merchant_ai.services.merchant_profile import MerchantProfileStore, MerchantProfileSummaryService
from merchant_ai.services.middleware import MiddlewareChain, default_harness_middlewares
from merchant_ai.services.observability import append_span, artifact_ref_from_path, now_ms, performance_summary, start_step, finish_step
from merchant_ai.services.planning import (
    compile_semantic_metric_fallback_graph,
    PlannerReflectionAgent,
    QueryGraphPlanner,
    QueryGraphValidator,
    query_plan_question_coverage_gaps,
    semantic_workspace_manifest_from_asset_pack,
)
from merchant_ai.services.planning_tooling import planner_failure_gap_code
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.quick_metrics import (
    is_metric_definition_question,
    published_semantic_quick_metrics,
    quick_metric_response,
)
from merchant_ai.services.query import (
    NodeWorkerExecutor,
    SqlValidationService,
    merge_task_result_bundles,
    prepare_execution_graph,
    query_bundle_complete_rows,
)
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, MerchantService, PendingAnswerStore
from merchant_ai.services.runtime_bindings import SemanticRuntimeBindingRegistry
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    HybridKnowledgeRetrievalService,
    KnowledgeRetrievalService,
    dedupe_retrieval_issues,
    failed_knowledge_bundle,
    merge_knowledge_fallback,
    normalize_knowledge_bundle_status,
    recall_item_sort_key,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
    PreflightUnderstandingService,
    QuestionRoutingService,
    RouteSlotExtractor,
    SemanticPreflightRouteClassifier,
    TopicRouterService,
    planning_hints_from_extracted_keywords,
    route_primary_topic,
)
from merchant_ai.services.skill_drafts import SkillDraftService
from merchant_ai.services.distributed_workers import (
    DistributedSubAgentClient,
    builtin_worker_handlers,
    coerce_handler_outcome,
    normalize_subagent_result,
)
from merchant_ai.services.tools import (
    artifact_file_tool_definitions,
    artifact_file_tool_schemas,
    lead_action_selection_tool,
    delegate_subagent_tool,
    node_runtime_tool_schemas,
    semantic_file_tool_definitions,
    semantic_file_tool_schemas,
)
from merchant_ai.services.time_semantics import apply_time_window_contract_to_plan, resolve_time_range, resolve_time_window_contract
from merchant_ai.services.text_parsing import contains_any_literal, safe_ascii_component, split_on_characters
from merchant_ai.services.tool_runtime import tool_runtime_scope
from merchant_ai.services.security import identity_scope_hash, identity_scope_payload

class MerchantQaWorkflow:
    def __init__(
        self,
        settings: Settings,
        merchant_service: MerchantService,
        answer_repository: AnswerRepository,
        pending_store: PendingAnswerStore,
        keyword_service: KeywordExtractService,
        routing_service: QuestionRoutingService,
        topic_router: TopicRouterService,
        recall_service: HybridRecallService,
        knowledge_retriever: KnowledgeRetrievalService,
        asset_builder: PlanningAssetPackBuilder,
        planner: QueryGraphPlanner,
        graph_validator: QueryGraphValidator,
        node_worker: NodeWorkerExecutor,
        evidence_verifier: EvidenceVerifier,
        answer_service: AnswerComposeService,
    ):
        self.settings = settings
        self.merchant_service = merchant_service
        self.answer_repository = answer_repository
        self.pending_store = pending_store
        self.keyword_service = keyword_service
        self.routing_service = routing_service
        self.topic_router = topic_router
        self.route_slot_extractor = RouteSlotExtractor(getattr(keyword_service, "topic_assets", None))
        self.recall_service = recall_service
        self.knowledge_retriever = knowledge_retriever
        self.asset_builder = asset_builder
        self.semantic_catalog = getattr(recall_service, "semantic_catalog", SemanticCatalogService(asset_builder.topic_assets))
        self.planner = planner
        self.planner_reflection_agent = PlannerReflectionAgent()
        self.graph_validator = graph_validator
        self.node_worker = node_worker
        self.evidence_verifier = evidence_verifier
        self.answer_service = answer_service
        self.policy = V2AgentPolicy(settings)
        self.semantic_route_classifier = SemanticPreflightRouteClassifier(settings)
        self.preflight_understanding = PreflightUnderstandingService(
            settings,
            keyword_service,
            routing_service,
            self.route_slot_extractor,
            self.semantic_route_classifier,
        )
        self.prompt_assembler = PromptAssembler()
        self.clarification_resolver = ClarificationResolutionService()
        self.merchant_profile_summary_service = MerchantProfileSummaryService()
        self.merchant_profile_store = MerchantProfileStore(settings)
        self.latency_optimizer = LatencyOptimizer()
        self.context_manager = ContextManager(settings)
        self.context_assembler = ContextAssembler(settings)
        self.thread_context_service = ThreadContextService(settings)
        self.memory_store = create_memory_store(settings)
        self.skill_draft_service = SkillDraftService(settings)
        self.middleware_chain = MiddlewareChain(default_harness_middlewares(settings, self.context_manager))
        self.checkpoint_manager = CheckpointManager(settings)
        self.graph = self._build_graph()

    def run(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Optional[GraphEventListener] = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[List[ConversationMessage]] = None,
    ) -> ChatResponse:
        effective_merchant_id = merchant_id or self.settings.merchant_id
        state = self._initial_state(question, effective_merchant_id, context, listener, thread_id, run_id, message_history)
        config = self.checkpoint_manager.config_for_run(state["thread_id"], state["run_id"])
        register_event_listener(state["run_id"], listener)
        try:
            with tool_runtime_scope(effective_merchant_id, state["thread_id"], state["run_id"]):
                final_state = self.graph.invoke(state, config=config)
                response = self.to_response(final_state)
                self.schedule_post_answer_tail(final_state)
                return response
        finally:
            unregister_event_listener(state["run_id"])

    async def run_async(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Optional[GraphEventListener] = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[List[ConversationMessage]] = None,
    ) -> ChatResponse:
        effective_merchant_id = merchant_id or self.settings.merchant_id
        state = self._initial_state(question, effective_merchant_id, context, listener, thread_id, run_id, message_history)
        config = self.checkpoint_manager.config_for_run(state["thread_id"], state["run_id"])
        register_event_listener(state["run_id"], listener)
        try:
            with tool_runtime_scope(effective_merchant_id, state["thread_id"], state["run_id"]):
                final_state = await asyncio.to_thread(self.graph.invoke, state, config)
                response = self.to_response(final_state)
                self.schedule_post_answer_tail(final_state)
                return response
        finally:
            unregister_event_listener(state["run_id"])

    def _initial_state(
        self,
        question: str,
        merchant_id: str,
        context: Optional[ChatContext],
        listener: Optional[GraphEventListener],
        thread_id: str,
        run_id: str,
        message_history: Optional[List[ConversationMessage]] = None,
    ) -> AgentState:
        qa_id = "qa_" + uuid.uuid4().hex
        actual_thread_id = thread_id or "thread_" + uuid.uuid4().hex
        actual_run_id = run_id or "run_" + uuid.uuid4().hex
        workspace = self.settings.resolved_workspace_path / "threads" / actual_thread_id / "runs" / actual_run_id
        workspace.mkdir(parents=True, exist_ok=True)
        return AgentState(
            qa_id=qa_id,
            question=(question or "").strip(),
            original_question=question or "",
            requested_merchant_id=merchant_id,
            request_context=context,
            user_identity=(context.user_identity.model_dump(by_alias=True) if context and context.user_identity else {}),
            access_role=merchant_access_role(context.user_identity.role if context and context.user_identity else ""),
            response_context=None,
            message_history=normalize_message_history(message_history),
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            checkpoint_thread_id=self.checkpoint_manager.thread_id_for_run(actual_thread_id, actual_run_id),
            thread_data=ThreadData(
                thread_id=actual_thread_id,
                run_id=actual_run_id,
                workspace_path=str(workspace),
                uploads_path=str(workspace / "uploads"),
                outputs_path=str(workspace / "outputs"),
            ),
            event_listener=None,
            merchant=self.merchant_service.current_merchant(merchant_id),
            recent_focus=MerchantRecentFocus(merchant_id=merchant_id),
            routing_decision=RoutingDecision(),
            topic_routing_decision=TopicRoutingDecision(),
            topic_workspace={},
            semantic_workspace_opened_topics=[],
            semantic_topic_index_read=False,
            analysis_scope={},
            knowledge_refresh={},
            route_slots=RouteSlots(),
            route_decision_trace=[],
            clarification_resolution={},
            clarification_root_question=(question or "").strip(),
            bounded_route_llm_trace={},
            bounded_lead_llm_trace={},
            _pending_lead_action_failure_observations=[],
            fast_gate_decision_trace={},
            main_agent_observations=[],
            fast_understanding=FastUnderstandingResult(),
            fast_metric_attempted=False,
            fast_metric_completed=False,
            fast_metric_response=None,
            plan=QueryPlan(),
            recall_bundle=RecallBundle(),
            initial_topic_recall_completed=False,
            initial_topic_recall_trace={},
            knowledge_bundle=KnowledgeBundle(),
            recall_rounds=[],
            intent_signals=IntentSignals(),
            planning_asset_pack=PlanningAssetPack(),
            planning_authority="legacy_question_understanding",
            legacy_planning_disabled=False,
            grounded_query_contract=None,
            grounded_query_contract_attempt=None,
            grounded_asset_pack=PlanningAssetPack(),
            grounded_contract_validation={},
            grounded_contract_ready=False,
            grounded_rejected_bindings=[],
            grounded_query_compiled=False,
            grounded_compile_trace={},
            grounded_compile_reason="",
            grounded_runtime_failure={},
            semantic_evidence_ledger=[],
            query_graph_validation_result=GraphValidationResult(),
            pending_knowledge_requests=[],
            knowledge_request_attempts={},
            knowledge_request_fingerprints={},
            blocked_knowledge_request_keys=[],
            knowledge_request_lineage={},
            knowledge_request_gaps=[],
            knowledge_retrieval_status="not_started",
            knowledge_retrieval_issues=[],
            knowledge_retrieval_outcomes=[],
            agent_run_result=AgentRunResult(),
            query_bundle=QueryBundle(),
            query_bundles=[],
            execution_attempt_artifacts=[],
            available_actions=[],
            lead_decisions=[],
            action_history=[],
            action_outcomes=[],
            action_catalog_contract_blocks=[],
            contract_block_observation={},
            contract_block_observed=False,
            contract_block_generation=0,
            lead_arbitration_observed=False,
            lead_provider_error="",
            last_action_result=ActionResult(),
            planner_reflection=PlannerReflectionResult(),
            node_tool_traces=[],
            freshness_reports=[],
            context_snapshots=[],
            context_packages=[],
            context_manifests=[],
            active_context_manifest={},
            active_context_package={},
            context_budget_reports=[],
            context_assembly_reports=[],
            context_compression_events=[],
            runtime_checkpoints=[],
            middleware_events=[],
            tool_call_ledger=[],
            tool_call_recovery_events=[],
            tool_call_requests=[],
            tool_loop_warning="",
            pending_tool_loop_warnings=[],
            tool_loop_history={},
            tool_loop_seen_call_ids={},
            forced_tool_loop_stop_message="",
            tool_output_budget_reports=[],
            token_usage_reports=[],
            safety_finish_reasons=[],
            run_budget_report={},
            run_budget_exhausted=False,
            # Run budgets use wall-clock epoch milliseconds so checkpoints can
            # be compared across processes.  Trace spans intentionally use the
            # monotonic ``now_ms`` helper and must not share this field.
            run_started_at_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            workspace_manifest=WorkspaceManifest(),
            run_steps=[],
            trace_spans=[],
            planner_repair_requests=[],
            planner_repair_input=PlannerRepairInput(),
            tool_failures=[],
            circuit_breakers=[],
            tool_runtime_policies=[],
            tool_call_results=[],
            tool_runtime_events=[],
            answer_file_tool_results={},
            clarification_tool_message={},
            clarification_command={},
            agent_decision_reason="",
            planner_repair_reason="",
            planner_provider_error="",
            planner_degraded={},
            base_knowledge_context="",
            topic_asset_context="",
            always_apply_context="",
            always_apply_rules=[],
            recall_context="",
            merchant_profile_context="",
            memory_context="",
            runtime_context="",
            session_context="",
            summary_context="",
            tool_context="",
            thread_context={},
            runtime_injection={},
            memory_injection={},
            memory_injection_raw_snapshot={},
            memory_injection_trace={},
            memory_recall_status="not_started",
            memory_recall_issues=[],
            memory_ingestion_trace={},
            memory_constraints=[],
            memory_constraint_trace={},
            memory_recalled=False,
            merchant_profile_summary={},
            open_diagnostic_scope="",
            open_diagnostic_intent="",
            open_diagnostic_goal="",
            open_diagnostic_profile={},
            open_diagnostic_seed_topics=[],
            hypothesis_exploration={},
            hypothesis_results=[],
            hypothesis_exploration_completed=False,
            hypothesis_exploration_status={"status": "pending", "source": "runtime"},
            hypothesis_exploration_rounds=0,
            hypothesis_selected_ids=[],
            hypothesis_evidence_ledger=HypothesisEvidenceLedger(),
            candidate_query_graphs={},
            strategy_switch_trace=[],
            latency_optimization={},
            execution_tier_policy={},
            node_execution_mode="auto",
            answer="",
            analysis_summary="",
            analysis_worker_trace={},
            analysis_worker_completed=False,
            analysis_worker_status={"status": "pending", "source": "runtime"},
            analysis_skill_trace={},
            subagent_delegation_plan={},
            subagent_delegation_results=[],
            subagent_delegation_attempted=False,
            subagent_delegation_completed=False,
            confirmation_evidence_reused=False,
            confirmation_token="",
            confirmation_source_run_id="",
            analysis_skill_bypassed=False,
            skill_worker_completed=False,
            analysis_skill_status={"status": "pending", "source": "runtime"},
            skill_match=SkillMatchState(),
            skill_draft=SkillDraft(),
            skill_lifecycle_records=[],
            merchant_experience={},
            answer_used_llm=False,
            suggestions=[],
            thinking_steps=[],
            history_rows=[],
            react_round=0,
            query_graph_retrieve_count=0,
            query_graph_supplemental_retrieve_count=0,
            query_graph_plan_attempts=0,
            query_graph_repair_attempts=0,
            query_graph_repair_attempted=False,
            query_graph_repair_progressed=False,
            query_graph_repair_scope_attempts={},
            query_graph_repair_scope_key="",
            query_graph_repair_scope_attempt_count=0,
            query_graph_repair_exhausted=False,
            query_graph_repair_history=[],
            last_query_graph_repair_delta=QueryGraphRepairDelta(),
            execution_generation=0,
            result_generation=-1,
            evidence_generation=-1,
            analysis_generation=-1,
            fast_understood=False,
            query_metric_attempted=False,
            query_metric_completed=False,
            planning_assets_compacted=False,
            skills_loaded=False,
            loaded_skills=[],
            rule_recall_ready=False,
            rule_recall_refs=[],
            rule_recall_context="",
            query_graph_validated=False,
            query_graph_validation_attempted=False,
            query_graph_validation_passed=False,
            query_graph_validation_status="not_run",
            validated_query_graph_fingerprint="",
            query_graph_reflected=False,
            sql_repair_reviewed=False,
            evidence_graph_verified=False,
            verification_status="not_run",
            evidence_accepted=False,
            supervised=False,
            scope_clarified=False,
            context_loaded=False,
            topic_routed=False,
            data_discovered=False,
            sql_generated=False,
            chat_bi_completed=False,
            run_canceled=False,
            middleware_loop_blocked=False,
            runtime_guard_gaps=[],
            terminal_status={},
            should_persist=False,
            persisted=False,
            post_answer_tail_pending=False,
            human_clarification_required=False,
            human_clarification_question="",
            human_clarification_stage="",
            human_clarification_type="",
            human_clarification_options=[],
        )

    def _build_graph(self):
        builder = StateGraph(AgentState)
        registry = self.policy.registry
        policy_routes = registry.policy_routing_map()
        for node in ["preflight_route", "inherit_context", "runtime_bootstrap", "policy", "finalize_action_contract"]:
            builder.add_node(node, getattr(self, "policy_node" if node == "policy" else node))
        for node in policy_routes:
            handler = getattr(self, node, None)
            if not callable(handler):
                raise ValueError("registered agent action has no workflow node handler: %s" % node)
            builder.add_node(node, handler)

        builder.add_edge(START, "preflight_route")
        builder.add_conditional_edges(
            "preflight_route",
            lambda state: "inherit_context" if self.preflight_needs_full_context(state) else "policy",
            {
                "inherit_context": "inherit_context",
                "policy": "policy",
            },
        )
        builder.add_edge("inherit_context", "runtime_bootstrap")
        builder.add_edge("runtime_bootstrap", "policy")
        builder.add_conditional_edges(
            "policy",
            lambda state: state.get("_next_action", registry.node_for("cache_answer")),
            policy_routes,
        )

        # The graph owns only the ReAct control loop. Business ordering must
        # never be encoded as action-to-action edges: every completed tool
        # returns its observation to LeadAgent, which selects again from the
        # registry catalog after middleware has filtered unsafe actions.
        human_node = registry.node_for("ask_human")
        cache_node = registry.node_for("cache_answer")
        terminal_node = registry.node_for("terminal_end")
        terminal_nodes = {human_node, cache_node, terminal_node}
        for node in terminal_nodes:
            builder.add_edge(node, "finalize_action_contract")
        builder.add_edge("finalize_action_contract", END)
        for node in policy_routes.keys() - terminal_nodes:
            builder.add_edge(node, "policy")
        return builder.compile(checkpointer=self.checkpoint_manager.saver())

    def finalize_action_contract(self, state: AgentState) -> AgentState:
        """Close the selected action contract before a terminal graph edge."""

        if state.get("_pending_action_contract"):
            state = self.middleware_chain.after_action(state)
        return state

    def observe_contract_block(self, state: AgentState) -> AgentState:
        """Expose a prerequisite failure as a neutral LeadAgent observation."""

        started = now_ms()
        observation = dict(state.get("contract_block_observation") or {})
        blocked_label = str(observation.get("blockedAction") or "").strip() or ",".join(
            str(item) for item in observation.get("blockedActions") or []
        )
        step = self.start_run_step(
            state,
            "observe_contract_block",
            "Runtime",
            "OBSERVE_CONTRACT_BLOCK",
            input_summary="blockedAction=%s" % blocked_label,
        )
        increment_round(state)
        emit(state, "node.started", "OBSERVE_CONTRACT_BLOCK", observation)
        state["contract_block_generation"] = int(state.get("contract_block_generation") or 0) + 1
        observation.update(
            {
                "status": "observed",
                "generation": state["contract_block_generation"],
                "reactRound": int(state.get("react_round") or 0),
            }
        )
        state["contract_block_observation"] = observation
        state["contract_block_observed"] = True
        add_step(
            state,
            "ActionContract：动作 %s 前置条件未满足，已作为 observation 返回 LeadAgent；未执行任何业务 fallback"
            % (blocked_label or "unknown"),
        )
        emit(state, "node.completed", "OBSERVE_CONTRACT_BLOCK", observation)
        self.record_span(
            state,
            "runtime",
            "action_contract.block_observation",
            started,
            status="gap",
            error_code="ACTION_CONTRACT_BLOCKED",
            metadata=observation,
        )
        self.finish_run_step(
            state,
            step,
            "gap",
            output_summary="missingKeys=%d missingFlags=%d"
            % (
                len(observation.get("missingStateKeys") or []),
                len(observation.get("missingStateFlags") or []),
            ),
            error_code="ACTION_CONTRACT_BLOCKED",
        )
        return state

    def lead_arbitrate(self, state: AgentState) -> AgentState:
        """Fail closed if an unresolved Lead arbitration ever reaches dispatch."""

        state["lead_arbitration_observed"] = True
        state["human_clarification_required"] = True
        state["human_clarification_question"] = "主 Agent 未能完成安全工具选择，请稍后重试。"
        state["human_clarification_stage"] = "LEAD_DECISION"
        state["human_clarification_type"] = "lead_decision_unavailable"
        state["human_clarification_options"] = []
        return state

    def terminal_or_human_node(self, state: AgentState) -> str:
        if state.get("run_canceled") or (state.get("terminal_status") or {}).get("active"):
            return self.policy.registry.node_for("terminal_end")
        if state.get("human_clarification_required"):
            return self.policy.registry.node_for("ask_human")
        return ""

    def can_retry_knowledge(self, state: AgentState) -> bool:
        return self.policy.can_retrieve_supplemental(state) and not self.policy.knowledge_recall_stalled(state)

    def can_repair_graph(self, state: AgentState) -> bool:
        return self.policy.graph_repair_attempt_count(state) < self.policy.max_graph_repair_actions

    def invalidate_execution_outputs(self, state: AgentState, reason: str) -> None:
        state["execution_generation"] = int(state.get("execution_generation") or 0) + 1
        had_outputs = bool(
            state.get("sql_generated")
            or state.get("evidence_graph_verified")
            or getattr(state.get("agent_run_result"), "task_results", None)
            or state.get("analysis_summary")
        )
        state["sql_generated"] = False
        state["sql_repair_reviewed"] = False
        state["evidence_graph_verified"] = False
        state["verification_status"] = "not_run"
        state["evidence_accepted"] = False
        state["result_generation"] = -1
        state["evidence_generation"] = -1
        state["analysis_generation"] = -1
        state["agent_run_result"] = AgentRunResult(
            execution_attempt_artifacts=normalized_execution_attempt_artifacts(state),
        )
        state["query_bundle"] = QueryBundle()
        state["query_bundles"] = []
        state["node_tool_traces"] = []
        state["freshness_reports"] = []
        state["analysis_summary"] = ""
        state["analysis_worker_result"] = {}
        state["analysis_worker_trace"] = {}
        state["analysis_worker_completed"] = False
        state["analysis_worker_status"] = {"status": "pending", "source": reason}
        state["analysis_skill_trace"] = {}
        state["skill_worker_completed"] = False
        skill_status = state.get("analysis_skill_status") or {}
        user_declined_skill = state.get("analysis_skill_bypassed") and skill_status.get("status") == "declined" and skill_status.get("source") == "user"
        if not user_declined_skill:
            state["analysis_skill_bypassed"] = False
            state["analysis_skill_status"] = {"status": "pending", "source": reason}
        if had_outputs:
            add_step(state, "Runtime Guard：%s，已清空旧 SQL/证据/分析输出" % reason)

    def clear_analysis_outputs(self, state: AgentState, reason: str) -> None:
        had_analysis = bool(state.get("analysis_summary") or state.get("analysis_worker_completed") or state.get("skill_worker_completed"))
        state["analysis_summary"] = ""
        state["analysis_generation"] = -1
        state["analysis_worker_result"] = {}
        state["analysis_worker_trace"] = {}
        state["analysis_worker_completed"] = False
        state["analysis_worker_status"] = {"status": "pending", "source": reason}
        state["analysis_skill_trace"] = {}
        state["skill_worker_completed"] = False
        skill_status = state.get("analysis_skill_status") or {}
        user_declined_skill = state.get("analysis_skill_bypassed") and skill_status.get("status") == "declined" and skill_status.get("source") == "user"
        if not user_declined_skill:
            state["analysis_skill_status"] = {"status": "pending", "source": reason}
        if had_analysis:
            add_step(state, "Runtime Guard：%s，已清空旧分析输出" % reason)

    def guard_unaccepted_evidence_for_answer(self, state: AgentState) -> None:
        run_result = state.get("agent_run_result") or AgentRunResult()
        if not getattr(run_result, "task_results", None):
            return
        if evidence_accepted_for_state(state):
            return
        existing = list(getattr(run_result, "evidence_gaps", []) or [])
        if not existing:
            existing = [
                EvidenceGap(
                    code="UNVERIFIED_EVIDENCE",
                    severity="blocking",
                    reason="查询结果尚未通过证据校验，不能作为完整业务结论。",
                    answer_instruction="本轮查询结果尚未通过证据校验，只能说明当前证据不完整，不能给出完整业务判断。",
                )
            ]
        blocking = [
            gap
            for gap in existing
            if str(getattr(gap, "severity", "") or "") == "blocking"
        ] or existing[:1]
        run_result.evidence_gaps = existing
        run_result.partial_answer_reason = run_result.partial_answer_reason or "查询结果尚未通过证据校验"
        run_result.verified_evidence = VerifiedEvidence(
            passed=False,
            gaps=existing,
            blocking_gaps=blocking,
            answer_guard_required=True,
            partial_answer_reason=run_result.partial_answer_reason,
        )
        state["agent_run_result"] = run_result
        state["evidence_accepted"] = False

    def route_after_retrieve_knowledge(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_rule_recall_ready(state) or self.should_answer_with_rule_recall(state):
            return "policy"
        if state.get("data_discovered") and not state.get("planning_assets_compacted"):
            return self.policy.registry.node_for("compact_assets")
        return "policy"

    def route_after_plan_query_graph(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_pending_knowledge_requests(state) and self.can_retry_knowledge(state):
            return "policy"
        plan = state.get("plan") or QueryPlan()
        if not plan.intents:
            return "policy"
        if self.policy.fast_path_bypasses_reflection(state):
            return self.policy.registry.node_for("validate_graph")
        return self.policy.registry.node_for("reflect_plan")

    def route_after_reflect_query_graph(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_pending_knowledge_requests(state) and self.can_retry_knowledge(state):
            return "policy"
        reflection = state.get("planner_reflection") or PlannerReflectionResult()
        if getattr(reflection, "passed", True):
            return self.policy.registry.node_for("validate_graph")
        return "policy"

    def route_after_repair_query_graph(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_pending_knowledge_requests(state) and self.can_retry_knowledge(state):
            return "policy"
        delta = state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()
        if state.get("query_graph_repair_exhausted") or not getattr(delta, "changed", False):
            return "policy"
        plan = state.get("plan") or QueryPlan()
        if not plan.intents:
            return "policy"
        return self.policy.registry.node_for("reflect_plan")

    def route_after_repair_sql(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_graph_repairable_execution_gap(state):
            return "policy"
        return self.policy.registry.node_for("verify_evidence")

    def preflight_route(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "preflight_route", "LeadAgent", "PREFLIGHT_ROUTE", input_summary=state.get("question", ""))
        emit(state, "node.started", "PREFLIGHT_ROUTE", {})
        state["_preflight_question"] = state.get("question", "")
        context = state.get("request_context")
        state["_preflight_requires_full_context"] = bool(
            context
            and (
                getattr(context, "pending_clarification_stage", "")
                or getattr(context, "pending_clarification_type", "") == "skill_confirm"
            )
        )
        understanding = self.preflight_understanding.understand(
            state["question"],
            pending_context=bool(state.get("_preflight_requires_full_context")),
        )
        state["semantic_preflight_route_trace"] = understanding.semantic_trace
        state["preflight_understanding"] = {
            "surfaceSignals": understanding.surface_signals,
            "clarificationQuestion": understanding.clarification_question,
        }
        state["routing_decision"] = understanding.routing_decision
        state["route_decision_trace"] = list(understanding.trace)
        route = state["routing_decision"].route
        if route == QuestionRoute.GREETING:
            add_step(state, "Preflight Route Gate：识别为轻量对话，不做 Topic/Metric 解析")
        elif route == QuestionRoute.INVALID and not state.get("_preflight_requires_full_context"):
            if understanding.surface_signals.get("writeOperation"):
                clarification = "当前 BI Agent 只支持只读查询和分析，不能执行写操作。请改成只读查询问题。"
                options = ["改成只读查询", "取消本次操作"]
                clarification_type = "write_operation"
                stage = "UNSUPPORTED_OPERATION"
            else:
                clarification = understanding.clarification_question or self.build_scope_clarification_prompt(state)
                options = business_scope_examples(self.recall_service.topic_assets)
                clarification_type = "business_scope"
                stage = "BUSINESS_SCOPE"
            self.request_human_clarification(state, clarification, stage, clarification_type, options)
            add_step(state, "Preflight Route Gate：入口信息不足或不支持，先 ask_human；未做 Topic/Metric 解析")
        else:
            add_step(state, "Preflight Route Gate：判断为可进入业务链路；Topic/Metric 留给后续节点解析")
        self.record_span(state, "action", "preflight_route", started)
        self.finish_run_step(state, step, "success", output_summary="route=%s" % enum_value(route))
        emit(state, "node.completed", "PREFLIGHT_ROUTE", {"route": enum_value(route)})
        return state

    def preflight_needs_full_context(self, state: AgentState) -> bool:
        route = state.get("routing_decision") or RoutingDecision()
        return bool(route.route == QuestionRoute.BUSINESS or state.get("_preflight_requires_full_context"))

    def merge_semantic_preflight_route(
        self,
        state: AgentState,
        rule_route: RoutingDecision,
        semantic_trace: Dict[str, Any],
        keywords: ExtractedKeywords,
        route_slots: RouteSlots,
    ) -> RoutingDecision:
        if not semantic_trace or semantic_trace.get("status") != "success":
            return rule_route
        pending_context = bool(state.get("_preflight_requires_full_context"))
        semantic_route = str(semantic_trace.get("route") or "")
        confidence = float(semantic_trace.get("confidence") or 0)
        min_conf = float(getattr(self.settings, "preflight_semantic_route_min_confidence", 0.62) or 0.62)
        high_conf = float(getattr(self.settings, "preflight_semantic_route_high_confidence", 0.86) or 0.86)
        strong_task_signal = self.has_strong_preflight_task_signal(keywords, route_slots)
        weak_business_signal = self.has_weak_preflight_business_signal(keywords, route_slots)

        if pending_context and semantic_route == "CLARIFICATION_REPLY" and confidence >= min_conf:
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=False, reason="语义路由：上一轮澄清/确认承接回复")
        if pending_context:
            return rule_route
        if confidence < min_conf:
            return rule_route
        if semantic_route == "GREETING":
            return RoutingDecision(route=QuestionRoute.GREETING, complex=False, reason="语义路由：寒暄/闲聊")
        if semantic_route == "BUSINESS_CHAT":
            if strong_task_signal and rule_route.route == QuestionRoute.BUSINESS and confidence < high_conf:
                return rule_route
            return RoutingDecision(route=QuestionRoute.GREETING, complex=False, reason="语义路由：经营闲聊，不触发查数")
        if semantic_route == "BUSINESS_TASK":
            if strong_task_signal or confidence >= high_conf:
                return RoutingDecision(
                    route=QuestionRoute.BUSINESS,
                    complex=bool(rule_route.complex),
                    reason="语义路由：明确经营任务；%s" % str(rule_route.reason or "进入完整链路"),
                )
            if weak_business_signal:
                return RoutingDecision(route=QuestionRoute.INVALID, complex=False, reason="语义路由：业务意图不完整，需要补充指标、对象或分析目标")
            return rule_route
        if semantic_route == "INVALID":
            if strong_task_signal and rule_route.route == QuestionRoute.BUSINESS and confidence < high_conf:
                return rule_route
            return RoutingDecision(route=QuestionRoute.INVALID, complex=False, reason=str(semantic_trace.get("reason") or "语义路由：非业务或范围不清"))
        if semantic_route == "CLARIFICATION_REPLY":
            return rule_route
        return rule_route

    def has_strong_preflight_task_signal(self, keywords: ExtractedKeywords, route_slots: RouteSlots) -> bool:
        has_metric = bool(getattr(keywords, "metric_keywords", []) or [])
        has_object = bool(getattr(route_slots, "object_refs", []) or [])
        has_time = bool(getattr(keywords, "time_keywords", []) or []) or bool(getattr(getattr(route_slots, "time_window", None), "days", 0))
        has_action = bool(getattr(keywords, "action_keywords", []) or getattr(keywords, "ranking_keywords", []))
        has_topic = bool(getattr(keywords, "topic_keywords", []) or getattr(route_slots, "topic_candidates", []))
        return bool(has_object or (has_metric and (has_time or has_action or has_topic)) or (has_topic and has_time and has_action))

    def has_weak_preflight_business_signal(self, keywords: ExtractedKeywords, route_slots: RouteSlots) -> bool:
        return bool(
            getattr(keywords, "business_keywords", [])
            or getattr(keywords, "topic_keywords", [])
            or getattr(keywords, "metric_keywords", [])
            or getattr(route_slots, "topic_candidates", [])
            or getattr(route_slots, "object_refs", [])
        )

    def should_recall_memory(self, state: AgentState) -> bool:
        if state.get("memory_recalled"):
            return False
        if not state.get("topic_routed"):
            return False
        if state.get("confirmation_evidence_reused"):
            return False
        if state.get("human_clarification_required"):
            return False
        route = state.get("routing_decision") or RoutingDecision()
        return bool(route.route == QuestionRoute.BUSINESS)

    def inherit_context(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "inherit_context", "LeadAgent", "INHERIT_CONTEXT", input_summary=state.get("question", ""))
        emit(state, "node.started", "INHERIT_CONTEXT", {"qaId": state["qa_id"]})
        thread_context = self.thread_context_service.restore(state)
        if thread_context.get("restored"):
            add_step(
                state,
                "Thread Context：已恢复上轮线程上下文 previousRun=%s reusableEntitySets=%d artifacts=%d"
                % (
                    thread_context.get("previousRunId", ""),
                    len(thread_context.get("reusableEntitySets") or []),
                    len(thread_context.get("previousArtifacts") or []),
                ),
            )
        context = state.get("request_context")
        if context and context.pending_clarification_type == "skill_confirm":
            resolution = self.clarification_resolver.resolve_context(
                context,
                state.get("question", ""),
            )
            if resolution:
                state["clarification_resolution"] = resolution
            self.restore_confirmation_evidence(state)
        if context and context.pending_clarification_stage and context.pending_question and not state.get("confirmation_evidence_reused"):
            resolution = self.clarification_resolver.resolve_context(context, state.get("question", ""))
            pending_question = str(context.pending_question or "").strip()
            if resolution:
                state["clarification_resolution"] = resolution
                state["question"] = merge_clarification_question(
                    pending_question,
                    str(resolution.get("normalizedAnswer") or state.get("question") or ""),
                )
                state["clarification_root_question"] = state["question"]
                add_step(state, "Clarification Resolver：已结构化回填 %s" % json.dumps(resolution, ensure_ascii=False, default=str))
                add_step(state, "Context Middleware：已合并上一轮澄清问题与规范化选项")
            else:
                state["question"] = pending_question
                state["clarification_root_question"] = pending_question
                add_step(state, "Context Middleware：澄清回复未解析，保留原问题且不累积无效回复")
        history_payload = render_message_history_context(
            state.get("message_history") or [],
            question=state.get("question", ""),
            llm=getattr(self.planner, "llm", None),
        )
        history_context = str(history_payload.get("context") or "")
        if history_context:
            message_history_source = (
                "server_thread_runs"
                if any(str(getattr(item, "local_id", "") or "").startswith("server_thread:") for item in state.get("message_history") or [])
                else "client_fallback"
            )
            state.setdefault("thread_context", thread_context or {})["messageHistorySummary"] = {
                "source": message_history_source,
                "usedLlm": bool(history_payload.get("usedLlm")),
                "fallback": bool(history_payload.get("fallback")),
                "summarySourceMessages": int(history_payload.get("summarySourceMessages") or 0),
                "recentMessages": int(history_payload.get("recentMessages") or 0),
            }
            state["session_context"] = append_context_section(
                state.get("session_context") or "",
                history_context,
                preserve_existing_chars=max(0, MAX_SHORT_TERM_CONTEXT_CHARS - 2000),
            )
            thread_context = state.setdefault("thread_context", thread_context or {})
            thread_context["messageHistory"] = [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in (state.get("message_history") or [])[-MAX_SHORT_TERM_MESSAGES:]
            ]
            add_step(
                state,
                "Short-term Memory：已接入当前会话多轮 messages=%d，旧消息%s压缩"
                % (len(state.get("message_history") or []), "由 LLM " if history_payload.get("usedLlm") else "规则兜底"),
            )
        self.record_span(state, "action", "inherit_context", started)
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary=state["question"][:500],
            artifact_paths=[item.get("path", "") for item in (thread_context.get("previousArtifacts") or [])[:6]],
        )
        emit(state, "node.completed", "INHERIT_CONTEXT", {"question": state["question"]})
        return state

    def apply_clarification_to_route_slots(self, state: AgentState, route_slots: RouteSlots) -> RouteSlots:
        route_slots, trace_items = self.clarification_resolver.apply_to_route_slots(route_slots, state.get("clarification_resolution") or {})
        if trace_items:
            state.setdefault("route_decision_trace", []).extend(trace_items)
        return route_slots

    def append_route_slots_trace(self, state: AgentState, route_slots: RouteSlots, stage: str = "extract_route_slots") -> None:
        state.setdefault("route_decision_trace", []).append(
            {
                "stage": stage,
                "operation": route_slots.operation,
                "riskLevel": route_slots.risk_level,
                "objectRefs": [item.model_dump(by_alias=True) for item in route_slots.object_refs],
                "timeWindow": route_slots.time_window.model_dump(by_alias=True),
                "analysisSignals": route_slots.analysis_signals,
                "routeConfidence": route_slots.route_confidence,
                "warnings": route_slots.route_warnings,
            }
        )

    def runtime_bootstrap(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "runtime_bootstrap", "LeadAgent", "LANGGRAPH_RUNTIME", input_summary=state.get("question", ""))
        emit(state, "node.started", "LANGGRAPH_RUNTIME", {})
        increment_round(state)
        requested_merchant_id = str(state.get("requested_merchant_id") or "").strip()
        merchant = state.get("merchant")
        merchant_id = str(getattr(merchant, "merchant_id", "") or "").strip() if merchant is not None else ""
        if not merchant_id or (requested_merchant_id and merchant_id != requested_merchant_id):
            merchant = self.merchant_service.current_merchant(requested_merchant_id)
        state["merchant"] = merchant
        state["merchant_profile_context"] = state["merchant"].profile_markdown()
        identity = getattr(state.get("request_context"), "user_identity", None)
        if identity:
            state["merchant_profile_context"] += "\n\n## 当前用户画像\n" + identity.prompt_markdown()
        if state.get("confirmation_evidence_reused"):
            state["supervised"] = True
            add_step(state, "LangGraph Runtime：已复用上一轮校验证据，跳过 Topic、召回、规划与 SQL 执行")
            self.refresh_context_snapshot(state, "runtime_bootstrap")
            self.record_span(state, "action", "runtime_bootstrap", started, metadata={"confirmationEvidenceReused": True})
            self.finish_run_step(state, step, "success", output_summary="restored verified evidence")
            emit(state, "node.completed", "LANGGRAPH_RUNTIME", {"confirmationEvidenceReused": True})
            return state
        state["supervised"] = True
        state["context_loaded"] = True
        add_step(state, "LangGraph Runtime：完成会话接入，已预加载店铺静态画像；业务理解留给 route_topic")
        self.refresh_context_snapshot(state, "runtime_bootstrap")
        self.record_span(state, "action", "runtime_bootstrap", started)
        self.finish_run_step(state, step, "success", output_summary="context_loaded")
        emit(state, "node.completed", "LANGGRAPH_RUNTIME", {"contextLoaded": True})
        return state

    def recall_memory(self, state: AgentState) -> AgentState:
        """Recall governed long-term memory before planning and execution."""
        started = now_ms()
        step = self.start_run_step(
            state,
            "recall_memory",
            "LeadAgent",
            "MEMORY_RECALL",
            input_summary=state.get("question", ""),
        )
        emit(state, "node.started", "MEMORY_RECALL", {})
        if not self.should_recall_memory(state):
            route = (state.get("routing_decision") or RoutingDecision()).route
            if not state.get("topic_routed"):
                state["memory_injection"] = {}
                state["memory_injection_trace"] = {"status": "skipped", "reason": "topic_not_routed", "route": enum_value(route)}
                state["memory_recall_status"] = "skipped"
                state["memory_recall_issues"] = []
                state["memory_constraints"] = []
                state["memory_constraint_trace"] = {"constraintCount": 0, "status": "skipped", "reason": "topic_not_routed"}
                add_step(state, "Long-term Memory：Topic workspace 未就绪，暂不召回长期记忆")
                self.record_span(state, "action", "recall_memory", started, metadata={"skipped": True, "route": enum_value(route), "reason": "topic_not_routed"})
                self.finish_run_step(state, step, "skipped", output_summary="topic_not_routed")
                emit(state, "node.completed", "MEMORY_RECALL", {"selectedCount": 0, "constraintCount": 0, "skipped": True})
                return state
            state["memory_injection"] = {}
            state["memory_injection_trace"] = {"status": "skipped", "reason": "non_business_route", "route": enum_value(route)}
            state["memory_recall_status"] = "skipped"
            state["memory_recall_issues"] = []
            state["memory_constraints"] = []
            state["memory_constraint_trace"] = {"constraintCount": 0, "status": "skipped", "reason": "non_business_route"}
            state["memory_recalled"] = True
            state["_memory_snapshot_locked"] = True
            add_step(state, "Long-term Memory：非业务/轻量请求跳过长期记忆召回")
            self.record_span(state, "action", "recall_memory", started, metadata={"skipped": True, "route": enum_value(route)})
            self.finish_run_step(state, step, "skipped", output_summary="non_business_route")
            emit(state, "node.completed", "MEMORY_RECALL", {"selectedCount": 0, "constraintCount": 0, "skipped": True})
            return state
        fingerprint = memory_query_hash(
            str(state.get("requested_merchant_id") or ""),
            retrieval_context_from_state(state),
        )
        state["memory_injection_raw_snapshot"] = {}
        issues: List[RetrievalIssue] = []
        try:
            injection = self.memory_store.select_for_question(
                state,
                budget_tokens=int(self.settings.context_memory_budget_tokens or 1200),
            )
        except Exception as exc:
            injection = {}
            issues = [
                RetrievalIssue(
                    code="MEMORY_RECALL_FAILED",
                    message=str(exc)[:500],
                    backend=type(self.memory_store).__name__,
                    lane="primary",
                    stage="acquire",
                    severity="blocking",
                    resolved=False,
                )
            ]
        else:
            issues = normalize_memory_recall_issues(
                (injection.get("memoryInjectionTrace") or {}).get("issues") or []
            )

        trace = memory_recall_trace_for(
            injection,
            issues,
            enrichment_status={"acquire": "failed" if not injection and issues else "success"},
        )
        if trace.get("status") == "failed" and not issues:
            issues = [
                RetrievalIssue(
                    code="MEMORY_RECALL_FAILED",
                    message="Memory recall returned an unusable snapshot without a structured issue",
                    backend=type(self.memory_store).__name__,
                    lane="primary",
                    stage="acquire",
                    severity="blocking",
                    resolved=False,
                )
            ]
            trace = memory_recall_trace_for(injection, issues, enrichment_status={"acquire": "failed"})

        state["memory_injection"] = injection
        state["memory_constraints"] = []
        state["memory_constraint_trace"] = {
            "constraintCount": 0,
            "status": trace.get("status", "failed"),
            "source": injection.get("source", "") if isinstance(injection, dict) else "",
        }
        if trace.get("status") != "failed":
            try:
                constraints = build_memory_constraints(injection)
            except Exception as exc:
                state["memory_injection_raw_snapshot"] = injection
                state["memory_injection"] = {}
                state["memory_constraints"] = []
                issues = normalize_memory_recall_issues(
                    [
                        *issues,
                        RetrievalIssue(
                            code="MEMORY_CONSTRAINT_COMPILATION_FAILED",
                            message=str(exc)[:500],
                            backend="memory_contract",
                            lane="constraints",
                            stage="compile_constraints",
                            severity="blocking",
                            resolved=False,
                        ),
                    ]
                )
                trace = memory_recall_trace_for(
                    {},
                    issues,
                    enrichment_status={"acquire": "success", "constraints": "failed"},
                )
                source_trace = dict(injection.get("memoryInjectionTrace") or {})
                for key in ["selectedIds", "candidateIds", "candidateCount", "filteredReasons"]:
                    if key in source_trace:
                        trace[key] = source_trace[key]
                state["memory_constraint_trace"] = {
                    "constraintCount": 0,
                    "status": "failed",
                    "error": str(exc)[:500],
                }
            else:
                state["memory_constraints"] = constraints
                state["memory_constraint_trace"] = {
                    "constraintCount": len(constraints),
                    "requiredCount": sum(
                        1 for item in constraints if str(item.get("enforcement") or "") == "required"
                    ),
                    "clarifyCount": sum(
                        1
                        for item in constraints
                        if str(item.get("enforcement") or "") == "clarify_or_disclose"
                    ),
                    "source": injection.get("source", ""),
                    "status": "success" if constraints else trace.get("status", "empty"),
                }
                trace["enrichmentStatus"] = {
                    **dict(trace.get("enrichmentStatus") or {}),
                    "constraints": "success",
                }

        if trace.get("status") != "failed":
            try:
                state["merchant_profile_summary"] = self.merchant_profile_summary_service.summarize(
                    merchant=state["merchant"],
                    memory_injection=state["memory_injection"],
                    memory_constraints=state["memory_constraints"],
                    route_slots=state.get("route_slots", RouteSlots()),
                    fast_understanding=state.get("fast_understanding", FastUnderstandingResult()),
                )
            except Exception as exc:
                issues = normalize_memory_recall_issues(
                    [
                        *issues,
                        RetrievalIssue(
                            code="MEMORY_PROFILE_SUMMARY_FAILED",
                            message=str(exc)[:500],
                            backend="profile_summary",
                            lane="enrichment",
                            stage="profile_summary",
                            severity="warning",
                            resolved=True,
                            details={"answerImpact": False},
                        ),
                    ]
                )
                trace = memory_recall_trace_for(
                    state["memory_injection"],
                    issues,
                    enrichment_status={
                        **dict(trace.get("enrichmentStatus") or {}),
                        "profileSummary": "failed",
                    },
                )
            else:
                trace["enrichmentStatus"] = {
                    **dict(trace.get("enrichmentStatus") or {}),
                    "profileSummary": "success",
                }

            try:
                renderer = getattr(self.memory_store, "render_injection", None)
                rendered = renderer(state["memory_injection"]) if callable(renderer) else ""
                state["memory_context"] = (
                    truncate_memory_text_by_tokens(
                        rendered,
                        int(self.settings.context_memory_budget_tokens or 1200),
                    )
                    if rendered
                    else ""
                )
            except Exception as exc:
                state["memory_context"] = ""
                issues = normalize_memory_recall_issues(
                    [
                        *issues,
                        RetrievalIssue(
                            code="MEMORY_RENDER_FAILED",
                            message=str(exc)[:500],
                            backend=type(self.memory_store).__name__,
                            lane="enrichment",
                            stage="render",
                            severity="warning",
                            resolved=True,
                            details={"answerImpact": False},
                        ),
                    ]
                )
                trace = memory_recall_trace_for(
                    state["memory_injection"],
                    issues,
                    enrichment_status={
                        **dict(trace.get("enrichmentStatus") or {}),
                        "render": "failed",
                    },
                )
            else:
                trace["enrichmentStatus"] = {
                    **dict(trace.get("enrichmentStatus") or {}),
                    "render": "success",
                }

        trace["contextFingerprint"] = fingerprint
        trace["issues"] = [issue.model_dump(by_alias=True) for issue in normalize_memory_recall_issues(issues)]
        state["memory_injection_trace"] = trace
        state["memory_recall_status"] = str(trace.get("status") or "failed")
        state["memory_recall_issues"] = list(trace.get("issues") or [])
        state["memory_recalled"] = True
        state["_memory_snapshot_locked"] = True
        if state.get("memory_injection"):
            state["memory_injection"]["memoryInjectionTrace"] = dict(trace)

        selected_count = len(trace.get("selectedIds") or [])
        constraint_count = len(state.get("memory_constraints") or [])
        status = str(trace.get("status") or "failed")
        if status == "failed":
            add_step(state, "Long-term Memory：回答前召回不可用，已保留结构化失败证据")
        elif status == "degraded":
            add_step(
                state,
                "Long-term Memory：回答前召回部分降级 selected=%d constraints=%d issues=%d"
                % (selected_count, constraint_count, len(issues)),
            )
        elif status == "empty":
            add_step(state, "Long-term Memory：召回成功，本轮没有匹配的受治理记忆")
        else:
            add_step(
                state,
                "Long-term Memory：回答前召回完成 selected=%d constraints=%d"
                % (selected_count, constraint_count),
            )
        self.finish_run_step(
            state,
            step,
            "failed" if status == "failed" else ("partial" if status == "degraded" else "success"),
            output_summary="status=%s selected=%d constraints=%d issues=%d"
            % (status, selected_count, constraint_count, len(issues)),
            error_code=(
                next((issue.code for issue in issues if not issue.resolved), "MEMORY_RECALL_FAILED")
                if status == "failed"
                else ""
            ),
        )
        self.record_span(
            state,
            "action",
            "recall_memory",
            started,
            status="failed" if status == "failed" else ("partial" if status == "degraded" else "success"),
            error_code=(
                next((issue.code for issue in issues if not issue.resolved), "")
                if status == "failed"
                else ""
            ),
            metadata={"status": status, "issueCount": len(issues), "usableSnapshot": bool(trace.get("usableSnapshot"))},
        )
        emit(
            state,
            "node.completed",
            "MEMORY_RECALL",
            {
                "selectedCount": len(state.get("memory_injection_trace", {}).get("selectedIds") or []),
                "constraintCount": len(state.get("memory_constraints") or []),
                "status": state.get("memory_recall_status", "failed"),
                "issues": state.get("memory_recall_issues", []),
            },
        )
        return state

    def policy_node(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "policy", "LeadAgent", "MAIN_AGENT_POLICY", input_summary=state.get("agent_decision_reason", ""))
        state = self.middleware_chain.after_action(state)
        self.materialize_plan_clarification(state)
        state = self.middleware_chain.before_policy(state)
        self.refresh_execution_tier_policy(state)
        observation = self.main_agent_observation(state)
        state.setdefault("main_agent_observations", []).append(observation)
        state["main_agent_observations"] = state["main_agent_observations"][-24:]
        state["lead_decision_context"] = self.build_lead_decision_context(state, observation)
        decision = self.policy.decide(state)
        decision.observation = observation.get("summary", "")
        decision = self.arbitrate_lead_action_if_needed(state, decision)
        state = self.middleware_chain.before_action(state, decision)
        if (state.get("terminal_status") or {}).get("active"):
            action = self.policy.registry.get("terminal_end")
            decision = AgentDecision(
                selected_action=action.id,
                selected_node=action.node,
                available_actions=[action.id],
                reason="terminal_status became active before action dispatch",
                budget_exhausted=True,
                source="terminal_status",
            )
        state = self.middleware_chain.capture_action(state, decision)
        state["_next_action"] = decision.selected_node
        state["available_actions"] = self.policy.registry.actions(decision.available_actions)
        state["agent_decision_reason"] = decision.reason
        self.ensure_terminal_planning_gap(state, decision)
        state.setdefault("lead_decisions", []).append(decision)
        selected = self.policy.registry.get(decision.selected_action)
        state.setdefault("action_history", []).append(
            AgentActionTrace(
                round=int(state.get("react_round") or 0),
                action=decision.selected_action,
                node=decision.selected_node,
                agent=selected.agent,
                status="selected",
                reason=decision.reason,
                available_actions=decision.available_actions,
                observation=decision.observation,
            )
        )
        emit(
            state,
            "agent.action.selected",
            "MAIN_AGENT_POLICY",
            {
                "action": decision.selected_action,
                "node": decision.selected_node,
                "reactRound": state.get("react_round", 0),
                "availableActions": [item.model_dump(by_alias=True) for item in state["available_actions"]],
                "reason": decision.reason,
                "observation": observation,
                "decisionContext": state.get("lead_decision_context", {}),
                "source": decision.source,
            },
        )
        self.record_span(state, "action", "policy", started)
        self.finish_run_step(state, step, "success", output_summary="%s->%s" % (decision.selected_action, decision.selected_node))
        return state

    def refresh_execution_tier_policy(self, state: AgentState) -> None:
        plan = state.get("plan") or QueryPlan()
        if not plan.intents:
            return
        run_result = state.get("agent_run_result") or AgentRunResult()
        prior_failures = sum(1 for item in run_result.task_results if item.query_bundle.failed)
        context = state.get("request_context")
        state["execution_tier_policy"] = self.latency_optimizer.execution_tier_policy(
            state.get("question", ""),
            plan,
            state.get("fast_understanding") or FastUnderstandingResult(),
            remaining_run_budget_seconds(state, self.settings),
            prior_failure_count=prior_failures,
            has_attachments=bool(getattr(context, "offloaded_files", None)),
        )

    def reconcile_fast_request_agent_gates(self, state: AgentState) -> None:
        """Keep expensive-agent gates aligned with the one-way fast-path state."""

        policy = state.get("latency_optimization") or {}
        if self.latency_optimizer.blocks_expensive_agents(policy):
            state["hypothesis_exploration"] = {}
            state["hypothesis_results"] = []
            state["hypothesis_exploration_completed"] = True
            state["hypothesis_exploration_status"] = {"status": "skipped", "source": "simple_request_fast_path"}
            state["analysis_skill_bypassed"] = True
            state["analysis_skill_status"] = {"status": "skipped", "source": "simple_request_fast_path"}
            return
        if (state.get("hypothesis_exploration_status") or {}).get("source") == "simple_request_fast_path":
            state["hypothesis_exploration"] = {}
            state["hypothesis_results"] = []
            state["hypothesis_exploration_completed"] = True
            state["hypothesis_exploration_status"] = {"status": "retired", "source": "lead_react"}
        if (state.get("analysis_skill_status") or {}).get("source") == "simple_request_fast_path":
            state["analysis_skill_bypassed"] = False
            state["analysis_skill_status"] = {"status": "pending", "source": "runtime"}

    def escalate_fast_request(self, state: AgentState, reason: str) -> None:
        state["latency_optimization"] = self.latency_optimizer.upgrade_to_standard(
            state.get("latency_optimization") or {},
            reason,
        )
        self.reconcile_fast_request_agent_gates(state)

    def arbitrate_lead_action_if_needed(self, state: AgentState, decision: AgentDecision) -> AgentDecision:
        mode = str(getattr(self.settings, "lead_action_llm_mode", "adaptive") or "adaptive").lower()
        policy_allowed = [str(item) for item in decision.available_actions if item]
        allowed = self.lead_llm_action_catalog(policy_allowed, state)
        is_fast_gate = {"query_metric", "plan_graph"}.issubset(set(allowed))
        observation = state.get("main_agent_observations", [{}])[-1] if state.get("main_agent_observations") else {}
        trace: Dict[str, Any] = {
            "mode": mode,
            "status": "skipped",
            "reason": "lead_action_llm_disabled",
            "deterministicAction": decision.selected_action,
            "policyAllowedActions": policy_allowed,
            "allowedActions": allowed,
            "observation": observation,
        }
        state["bounded_lead_llm_trace"] = trace
        deterministic_reason = self.lead_action_deterministic_skip_reason(state, decision, allowed)
        if deterministic_reason:
            trace.update(
                {
                    "status": "skipped",
                    "reason": deterministic_reason,
                    "policySource": "deterministic",
                    "selectedAction": decision.selected_action,
                }
            )
            return decision
        if mode in {"off", "false", "0", "disabled"}:
            return self.lead_decision_unavailable(state, decision, allowed, trace, "lead_action_llm_disabled")
        if len(allowed) <= 1:
            trace["reason"] = "single_available_action"
            trace["policySource"] = "deterministic"
            return decision
        should_call = mode == "always" or (mode == "fast_gate" and is_fast_gate) or (
            mode == "low_confidence"
            and (
                bool(state.get("pending_knowledge_requests"))
                or bool(state.get("planner_repair_requests"))
                or (
                    graph_validation_attempted(state)
                    and not graph_validation_passed(state)
                )
            )
        )
        if mode == "adaptive":
            should_call = self.adaptive_lead_llm_needed(state, allowed, is_fast_gate)
        if not should_call:
            return self.lead_decision_unavailable(state, decision, allowed, trace, "lead_action_arbitration_not_available")
        llm = getattr(self.planner, "llm", None)
        if not llm or not getattr(llm, "configured", False):
            return self.lead_decision_unavailable(state, decision, allowed, trace, "lead_action_llm_not_configured")
        trace.update({"status": "calling", "reason": "lead_llm_arbitration_needed", "policySource": "lead_llm_arbitration"})
        payload = {
            "question": state.get("question", ""),
            "observation": observation,
            "fastUnderstanding": (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
            "deterministicDecision": decision.model_dump(by_alias=True),
            "allowedActions": allowed,
            "actionCatalog": [
                self.policy.registry.get(action_id).model_dump(by_alias=True)
                for action_id in allowed
            ],
            "recentActions": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in state.get("action_history", [])[-6:]
            ],
            "pendingKnowledgeRequests": [item.model_dump(by_alias=True) for item in state.get("pending_knowledge_requests", [])[:5]],
            "plannerRepairRequests": [item.model_dump(by_alias=True) for item in state.get("planner_repair_requests", [])[:5]],
            "plannerReflection": (state.get("planner_reflection") or PlannerReflectionResult()).model_dump(by_alias=True),
            "queryGraphValidation": (state.get("query_graph_validation_result") or GraphValidationResult()).model_dump(by_alias=True),
            "decisionContext": state.get("lead_decision_context", {}),
            "instruction": self.lead_agent_tool_instruction(is_fast_gate),
        }
        lead_timeout_seconds = max(
            1,
            int(
                getattr(
                    self.settings,
                    "llm_lead_timeout_seconds",
                    getattr(self.settings, "llm_request_timeout_seconds", 12),
                )
                or 1
            ),
        )
        prompt_text = json.dumps(payload, ensure_ascii=False, default=str)
        tool = lead_action_selection_tool(allowed) if hasattr(llm, "tool_json_chat") else None
        if tool is not None:
            trace["tool"] = tool.trace_schema()
        max_retries = bounded_single_retry_count(
            int(getattr(self.settings, "agent_lead_action_retries", 1) or 0)
        )
        trace["maxRetries"] = max_retries
        trace["attempts"] = []
        llm_payload: Dict[str, Any] = {}
        attempt = 0
        attempt_timeout_seconds = lead_timeout_seconds
        while True:
            lead_llm_started = now_ms()
            raised_error = ""
            try:
                if tool is not None:
                    llm_payload = llm.tool_json_chat(
                        "你是主 Agent 的 ReAct 决策器。先读 observation，再只能调用 select_agent_action 选择下一步。",
                        prompt_text,
                        tool.openai_schema(),
                        {},
                        timeout_seconds=attempt_timeout_seconds,
                    )
                else:
                    llm_payload = llm.json_chat(
                        "你是 BI Agent 的受限 LeadAction 选择器，只能在给定 action registry 候选中改选下一步。",
                        prompt_text,
                        fallback={},
                        timeout_seconds=attempt_timeout_seconds,
                    )
            except Exception as exc:
                raised_error = "%s: %s" % (type(exc).__name__, str(exc))
                llm_payload = {}

            if llm_payload:
                self.record_span(
                    state,
                    "llm",
                    "lead_action.select",
                    lead_llm_started,
                    model=self.settings.openai_model,
                    provider=self.settings.openai_base_url,
                    estimated_prompt_chars=len(prompt_text),
                    estimated_completion_chars=len(json.dumps(llm_payload, ensure_ascii=False, default=str)),
                    metadata={"attempt": attempt + 1, "retried": attempt > 0},
                )
                state["lead_provider_error"] = ""
                break

            llm_error = raised_error or str(getattr(llm, "last_error", "") or "")
            error_code, retryable = classify_lead_llm_failure(llm_error)
            error_message = llm_error or "provider returned no Lead action tool payload"
            failure_observation = self.preserve_lead_action_failure_observation(
                state,
                trace,
                attempt=attempt + 1,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
            trace["attempts"].append(failure_observation)
            self.record_span(
                state,
                "llm",
                "lead_action.select",
                lead_llm_started,
                status="failed",
                error_code=error_code,
                error_message=error_message[:300],
                model=self.settings.openai_model,
                provider=self.settings.openai_base_url,
                estimated_prompt_chars=len(prompt_text),
                estimated_completion_chars=0,
                metadata={"attempt": attempt + 1, "retryable": retryable},
            )
            retry_timeout_seconds = 0
            if retryable and attempt < max_retries:
                retry_timeout_seconds = self.lead_action_retry_timeout_seconds(state, lead_timeout_seconds)
            if retry_timeout_seconds > 0:
                failure_observation["retryScheduled"] = True
                # The sync LLM timeout worker cannot be force-cancelled. Lead
                # selection is read-only/idempotent, so permit at most one
                # overlapping retry and only while the global budget can cover it.
                attempt += 1
                attempt_timeout_seconds = retry_timeout_seconds
                trace.update(
                    {
                        "status": "retrying",
                        "reason": "lead_action_llm_retryable_failure",
                        "errorCode": error_code,
                        "errorMessage": error_message[:300],
                        "retryAttempt": attempt,
                        "retryTimeoutSeconds": retry_timeout_seconds,
                    }
                )
                continue

            failure_observation["retryScheduled"] = False
            failure_reason = lead_llm_failure_reason(error_code)
            trace.update(
                {
                    "status": "failed",
                    "reason": failure_reason,
                    "errorCode": error_code,
                    "errorMessage": error_message[:300],
                    "payload": {},
                }
            )
            if retryable and attempt < max_retries:
                trace["retrySkippedReason"] = "insufficient_run_budget"
            return self.lead_decision_unavailable(
                state,
                decision,
                allowed,
                trace,
                failure_reason,
                error_code=error_code,
                error_message=error_message,
            )
        trace["recoveredAfterRetry"] = attempt > 0
        selected_action = str(
            (llm_payload or {}).get("actionId")
            or (llm_payload or {}).get("action_id")
            or (llm_payload or {}).get("selectedAction")
            or (llm_payload or {}).get("selected_action")
            or ""
        )
        if selected_action not in allowed:
            trace.update({"status": "ignored", "reason": "llm_selected_action_not_allowed", "payload": llm_payload or {}})
            return self.lead_decision_unavailable(
                state,
                decision,
                allowed,
                trace,
                "llm_selected_action_not_allowed",
                error_code="LEAD_ACTION_INVALID",
                error_message="Lead model selected an action outside the governed catalog",
            )
        state["_lead_llm_decision_fingerprint"] = lead_decision_fingerprint(state, allowed)
        if selected_action == decision.selected_action:
            llm_reason = str((llm_payload or {}).get("reason") or "")[:300]
            reason = "bounded Lead LLM selected %s from registry. %s" % (selected_action, llm_reason)
            trace.update({"status": "accepted", "selectedAction": selected_action, "reason": reason, "payload": llm_payload or {}})
            return AgentDecision(
                selected_action=decision.selected_action,
                selected_node=decision.selected_node,
                available_actions=allowed,
                reason=reason,
                budget_exhausted=decision.budget_exhausted,
                observation=str(observation.get("summary") or decision.observation),
                source="lead_llm_tool",
            )
        action = self.policy.registry.get(selected_action)
        reason = "bounded Lead LLM selected %s from registry; deterministic was %s. %s" % (
            selected_action,
            decision.selected_action,
            str((llm_payload or {}).get("reason") or "")[:300],
        )
        trace.update({"status": "accepted", "selectedAction": selected_action, "payload": llm_payload or {}, "reason": reason})
        return AgentDecision(
            selected_action=action.id,
            selected_node=action.node,
            available_actions=allowed,
            reason=reason,
            budget_exhausted=decision.budget_exhausted,
            observation=str(observation.get("summary") or decision.observation),
            source="lead_llm_tool",
        )

    def lead_action_retry_timeout_seconds(self, state: AgentState, lead_timeout_seconds: int) -> int:
        """Return one full retry timeout only when the run can still answer safely."""

        return retry_timeout_with_answer_reserve(
            remaining_run_budget_seconds(state, self.settings),
            lead_timeout_seconds,
            int(getattr(self.settings, "llm_answer_timeout_seconds", 10) or 10),
        )

    def preserve_lead_action_failure_observation(
        self,
        state: AgentState,
        trace: Dict[str, Any],
        *,
        attempt: int,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> Dict[str, Any]:
        """Persist an operational Lead failure for the next ReAct observation."""

        prior_observation = deepcopy(trace.get("observation") or {})
        prior_observation.pop("leadActionFailures", None)
        failure = {
            "stage": "lead_action.select",
            "status": "retryable_failure" if retryable else "failed",
            "attempt": max(1, int(attempt)),
            "errorCode": str(error_code or "LEAD_LLM_FAILED"),
            "errorMessage": str(error_message or "")[:300],
            "retryable": bool(retryable),
            "deterministicAction": str(trace.get("deterministicAction") or ""),
            "allowedActions": [str(item) for item in trace.get("allowedActions", []) if str(item)],
            "priorObservation": prior_observation,
        }
        pending = list(state.get("_pending_lead_action_failure_observations") or [])
        pending.append(failure)
        state["_pending_lead_action_failure_observations"] = pending[-4:]
        return failure

    def runtime_safe_lead_recovery_action(
        self,
        state: AgentState,
        decision: AgentDecision,
        allowed: List[str],
    ) -> tuple[str, str, List[Dict[str, object]]]:
        """Select a contract-safe progress action when model arbitration is unavailable."""

        catalog = list(dict.fromkeys(str(item) for item in allowed if str(item)))
        deterministic_action = str(decision.selected_action or "")
        if deterministic_action in catalog:
            return deterministic_action, "preserve_governed_policy_action", []

        safe, blocked = self.policy.contract_safe_action_ids(state, catalog)

        if state.get("pending_knowledge_requests") and "retrieve_knowledge" in safe:
            return "retrieve_knowledge", "pending_knowledge_contract", blocked

        final_actions = {"answer_data", "ask_human", "cache_answer", "terminal_end"}
        progress_actions = [action_id for action_id in safe if action_id not in final_actions]
        if progress_actions:
            selected_action = progress_actions[0]
            selected_score = -1
            for action_id in progress_actions:
                action = self.policy.registry.get(action_id)
                unmet_postconditions = sum(
                    1 for path in action.expected_state_keys if not state_path_ready(state, path)
                ) + sum(
                    1 for flag in action.expected_state_flags if not action_state_flag_ready(state, flag)
                )
                if unmet_postconditions > selected_score:
                    selected_action = action_id
                    selected_score = unmet_postconditions
            return selected_action, "unmet_postcondition_progress", blocked

        if safe:
            return safe[0], "only_contract_safe_terminal_action", blocked
        return "lead_arbitrate", "no_contract_safe_recovery_action", blocked

    def lead_decision_unavailable(
        self,
        state: AgentState,
        decision: AgentDecision,
        allowed: List[str],
        trace: Dict[str, Any],
        reason: str,
        *,
        error_code: str = "LEAD_DECISION_UNAVAILABLE",
        error_message: str = "",
    ) -> AgentDecision:
        """Preserve a safe policy choice or surface an explicit Lead failure.

        Provider failures are operational failures, not clarification needs.
        Runtime may preserve an already-selected governed action, but it never
        invents a business action outside the filtered catalog.
        """

        selected_action, recovery_strategy, blocked = self.runtime_safe_lead_recovery_action(
            state,
            decision,
            allowed,
        )
        policy_source = "runtime_safe_fallback" if selected_action in allowed else "runtime_explicit_failure"
        if error_message or error_code.startswith("LEAD_LLM"):
            state["lead_provider_error"] = error_message or reason
        action = self.policy.registry.get(selected_action)
        available = list(dict.fromkeys([*allowed, selected_action]))
        trace.update(
            {
                "status": "failed_closed",
                "reason": reason,
                "errorCode": error_code,
                "errorMessage": str(error_message or "")[:300],
                "selectedAction": selected_action,
                "policySource": policy_source,
                "recoverySelection": {
                    "strategy": recovery_strategy,
                    "selectedAction": selected_action,
                    "blockedCandidates": blocked,
                },
            }
        )
        return AgentDecision(
            selected_action=action.id,
            selected_node=action.node,
            available_actions=available,
            reason="%s; Lead arbitration failed and Runtime applied its explicit safety contract" % reason,
            budget_exhausted=decision.budget_exhausted,
            observation=str((trace.get("observation") or {}).get("summary") or decision.observation),
            source=policy_source,
        )

    def apply_bounded_lead_llm_decision(self, state: AgentState, decision: AgentDecision) -> AgentDecision:
        return self.arbitrate_lead_action_if_needed(state, decision)

    def lead_action_deterministic_skip_reason(self, state: AgentState, decision: AgentDecision, allowed: List[str]) -> str:
        """Return a hard runtime reason that forbids model arbitration."""

        del allowed
        selected = str(decision.selected_action or "")
        if selected in {"terminal_end", "cache_answer", "ask_human"}:
            return "%s_is_terminal_or_user_blocked" % selected
        if bool(decision.budget_exhausted):
            return "budget_or_runtime_guard_decision"
        if state.get("human_clarification_required"):
            return "human_clarification_required"
        if selected == "verify_evidence":
            return "evidence_verification_is_mandatory"
        if selected == "retrieve_knowledge" and state.get("pending_knowledge_requests"):
            return "pending_knowledge_request_requires_retrieve"
        return ""

    def lead_llm_action_catalog(self, action_ids: List[str], state: Optional[AgentState] = None) -> List[str]:
        """Return the safe policy catalog without adding workflow preferences."""

        del state
        return list(dict.fromkeys(action_ids))

    def adaptive_lead_llm_needed(self, state: AgentState, allowed: List[str], is_fast_gate: bool) -> bool:
        del is_fast_gate
        if len(allowed) <= 1:
            return False
        if remaining_run_budget_seconds(state, self.settings) <= 12:
            return False
        fingerprint = lead_decision_fingerprint(state, allowed)
        return fingerprint != str(state.get("_lead_llm_decision_fingerprint") or "")

    def lead_agent_tool_instruction(self, is_fast_gate: bool) -> str:
        base = (
            "你是商家经营分析主 Agent。根据用户目标、observation、工具结果、证据缺口和 actionCatalog 自主选择下一项工具。"
            "Harness 已经移除了不满足权限或安全前置条件的工具；不要机械遵循固定流水线，也不要重复没有新增信息的动作。"
            "在证据不足时继续检索、规划、修复或执行；只有证据已经校验或必须明确披露缺口时才回答。"
            "专项深度分析只在已取得并校验经营数据后选择。只能从 allowedActions 中选择一个 action id，不创造新 action。"
            "delegate_subagent 仅用于 Runtime 治理的附件、Python、query 或 Skill Worker；普通只读上下文隔离由外层 DeepAgent task 负责；"
            "当已验证数据足够但问题属于开放、长尾、非固定 SOP 的分析，可选择 run_analysis_worker；"
            "只有明确匹配已发布可复用 Skill 时才选择 run_analysis_skill；不要把普通复杂分析硬塞进 Skill。"
            "如果当前证据已经足以回答，且不需要隔离分析或 Skill，选择 answer_data。"
            "返回 JSON: {selectedAction:'', reason:''}。"
        )
        if not is_fast_gate:
            return base
        return base + (
            "当前已完成 Topic workspace 召回和资产压缩：只有一个受控语义指标、无需归因/排行/明细/跨域分析时可选择 query_metric；"
            "如果指标歧义、资产缺口、需要解释原因或多轮探索，选择 plan_graph 或后续 ask_human。"
            "query_metric 只接收当前 Topic 资产包里的语义引用并产出已校验 QueryGraph；执行与证据校验必须作为后续独立动作再次由 Lead 选择。"
        )

    def main_agent_observation(self, state: AgentState) -> Dict[str, Any]:
        validation = state.get("query_graph_validation_result") or GraphValidationResult()
        run_result = state.get("agent_run_result") or AgentRunResult()
        plan = state.get("plan") or QueryPlan()
        execution_observations = self.execution_observation_payload(state, run_result)
        repair_input = state.get("planner_repair_input") or PlannerRepairInput()
        if isinstance(repair_input, dict):
            try:
                repair_input = PlannerRepairInput.model_validate(repair_input)
            except Exception:
                repair_input = PlannerRepairInput()
        planner_operational_observations = list(
            (plan.planner_prompt_stats or {}).get("operationalAttempts") or []
        )
        lead_action_failures = list(state.get("_pending_lead_action_failure_observations") or [])
        state["_pending_lead_action_failure_observations"] = []
        pending_requests = state.get("pending_knowledge_requests") or []
        graph_gaps = getattr(validation, "gaps", []) or []
        evidence_gaps = getattr(run_result, "evidence_gaps", []) or []
        summary_parts = [
            "round=%s" % int(state.get("react_round") or 0),
            "topicRouted=%s" % bool(state.get("topic_routed")),
            "retrieved=%s" % bool(state.get("data_discovered")),
            "assets=%s" % bool(state.get("planning_assets_compacted")),
            "planNodes=%d" % len(plan.intents),
            "validated=%s" % bool(state.get("query_graph_validated")),
            "sqlGenerated=%s" % bool(state.get("sql_generated")),
            "evidenceChecked=%s" % bool(state.get("evidence_graph_verified")),
            "evidenceAccepted=%s" % bool(evidence_accepted_for_state(state)),
        ]
        if pending_requests:
            summary_parts.append("pendingKnowledge=%d" % len(pending_requests))
        if graph_gaps:
            summary_parts.append("graphGaps=%d" % len(graph_gaps))
        if evidence_gaps:
            summary_parts.append("evidenceGaps=%d" % len(evidence_gaps))
        execution_tasks = execution_observations["tasks"]
        if execution_tasks:
            summary_parts.append("executionTasks=%d" % len(execution_tasks))
        failed_task_ids = [
            str(item.get("taskId") or "")
            for item in execution_tasks
            if item.get("failed") and item.get("taskId")
        ]
        if failed_task_ids:
            summary_parts.append("executionFailedTasks=%s" % ",".join(failed_task_ids[:6]))
        zero_row_task_ids = execution_observations["zeroRowTaskIds"]
        if zero_row_task_ids:
            summary_parts.append("zeroRowTasks=%s" % ",".join(zero_row_task_ids[:8]))
        freshness_items = execution_observations["freshness"]
        if freshness_items:
            summary_parts.append(
                "freshness=%s"
                % ",".join(
                    "%s:%s(max=%s,fallback=%s,coverage=%s)"
                    % (
                        str(item.get("taskId") or "unknown"),
                        str(item.get("status") or "NOT_CHECKED"),
                        str(item.get("maxTimeValue") or "none"),
                        str(item.get("fallbackTable") or "none"),
                        bool(item.get("coverageComplete")),
                    )
                    for item in freshness_items[:4]
                )
            )
        snapshot = execution_observations["snapshotAlignment"]
        if snapshot.get("status") and snapshot.get("status") != "NOT_APPLICABLE":
            summary_parts.append(
                "snapshotAlignment=%s(complete=%s)"
                % (snapshot.get("status"), bool(snapshot.get("complete")))
            )
        if execution_observations["sqlRepairAttempts"]:
            summary_parts.append(
                "sqlRepairAttempts=%d" % len(execution_observations["sqlRepairAttempts"])
            )
        if lead_action_failures:
            summary_parts.append(
                "leadActionFailures=%s"
                % ",".join(str(item.get("errorCode") or "LEAD_LLM_FAILED") for item in lead_action_failures)
            )
        if planner_operational_observations:
            recovered_planner_observations = [
                item
                for item in planner_operational_observations
                if isinstance(item, dict) and item.get("recovered") is True
            ]
            terminal_planner_observations = [
                item
                for item in planner_operational_observations
                if isinstance(item, dict) and item.get("recovered") is not True
            ]
            if recovered_planner_observations:
                summary_parts.append(
                    "plannerOperationalRetriesRecovered=%s"
                    % ",".join(
                        str(item.get("errorCode") or "PLANNER_LLM_FAILED")
                        for item in recovered_planner_observations
                    )
                )
            if terminal_planner_observations:
                summary_parts.append(
                    "plannerOperationalFailures=%s"
                    % ",".join(
                        str(item.get("errorCode") or "PLANNER_LLM_FAILED")
                        for item in terminal_planner_observations
                    )
                )
        if state.get("human_clarification_required"):
            summary_parts.append("needsHuman=true")
        contract_block = state.get("contract_block_observation") or {}
        if contract_block:
            summary_parts.append("contractBlock=%s" % contract_block.get("blockedAction", "unknown"))
        repair_delta = state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()
        if getattr(repair_delta, "attempt", 0):
            summary_parts.append(
                "graphRepair=%s(scopeAttempt=%d,changed=%s,exhausted=%s)"
                % (
                    repair_delta.status,
                    repair_delta.scope_attempt,
                    repair_delta.changed,
                    repair_delta.exhausted,
                )
            )
        last_action = {}
        if state.get("action_history"):
            item = state["action_history"][-1]
            last_action = item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
        return {
            "summary": "; ".join(summary_parts),
            "lastAction": last_action,
            "pendingKnowledgeRequests": [item.model_dump(by_alias=True) for item in pending_requests[:4]],
            "graphGaps": [gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap for gap in graph_gaps[:6]],
            "evidenceGaps": [gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap for gap in evidence_gaps[:6]],
            "executionObservations": execution_observations,
            "toolRuntimeFailures": state.get("tool_failures", [])[-4:],
            "leadActionFailures": lead_action_failures,
            "plannerOperationalObservations": planner_operational_observations,
            "contractBlockObservation": contract_block,
            "catalogContractBlocks": state.get("action_catalog_contract_blocks", []),
            "plannerReflection": (
                state.get("planner_reflection") or PlannerReflectionResult()
            ).model_dump(by_alias=True),
            "plannerRepairRequests": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in (state.get("planner_repair_requests") or [])
            ],
            # PlannerCritic/Repair is a ReAct observation contract, not transient
            # workflow state. Keep the immutable typed input intact so the next
            # decision can reason over the original issue codes and repair scope.
            "plannerRepairInput": repair_input.model_dump(by_alias=True),
            "queryGraphRepairDelta": (
                repair_delta.model_dump(by_alias=True) if getattr(repair_delta, "attempt", 0) else {}
            ),
        }

    def build_lead_decision_context(self, state: AgentState, observation: Dict[str, Any]) -> Dict[str, Any]:
        validation = state.get("query_graph_validation_result") or GraphValidationResult()
        run_result = state.get("agent_run_result") or AgentRunResult()
        graph_gaps = list(getattr(validation, "gaps", []) or [])
        evidence_gaps = list(getattr(run_result, "evidence_gaps", []) or [])
        sql_failures = self.sql_failure_payloads(run_result)
        recall_refs = self.current_recall_refs(state)
        seen_refs = [str(item) for item in state.get("_lead_seen_recall_refs", []) if str(item)]
        seen_set = set(seen_refs)
        new_refs = [ref for ref in recall_refs if ref not in seen_set]
        if recall_refs:
            state["_lead_seen_recall_refs"] = (seen_refs + new_refs)[-240:]
        current_counts = {
            "graphGaps": len(graph_gaps),
            "evidenceGaps": len(evidence_gaps),
            "sqlFailures": len(sql_failures),
            "pendingKnowledge": len(state.get("pending_knowledge_requests") or []),
        }
        previous_counts = state.get("_lead_previous_gap_counts") or {}
        gap_delta = {
            key: current_counts[key] - int(previous_counts.get(key, 0) or 0)
            for key in current_counts
        }
        state["_lead_previous_gap_counts"] = current_counts
        last_action, repeat_count = self.last_action_repeat_count(state)
        knowledge_recall_stalled = bool(
            last_action == "retrieve_knowledge"
            and int(state.get("query_graph_retrieve_count") or 0) > 0
            and not new_refs
        )
        decision_hints: List[str] = []
        if knowledge_recall_stalled:
            decision_hints.append("retrieve_knowledge_has_no_new_refs")
        if graph_gaps:
            decision_hints.append("graph_validation_gaps_present")
        if evidence_gaps:
            decision_hints.append("evidence_gaps_present")
        if sql_failures:
            decision_hints.append("sql_failures_present")
        if state.get("human_clarification_required"):
            decision_hints.append("human_clarification_required")
        context = {
            "observationSummary": observation.get("summary", ""),
            "progress": {
                "lastAction": last_action,
                "lastActionRepeatCount": repeat_count,
                "newRecallRefsCount": len(new_refs),
                "newRecallRefs": new_refs[:12],
                "recallRefsTotal": len(recall_refs),
                "gapDelta": gap_delta,
                "knowledgeRecallStalled": knowledge_recall_stalled,
            },
            "stateFlags": {
                "topicRouted": bool(state.get("topic_routed")),
                "fastUnderstood": bool(state.get("fast_understood")),
                "dataDiscovered": bool(state.get("data_discovered")),
                "planningAssetsCompacted": bool(state.get("planning_assets_compacted")),
                "queryGraphReflected": bool(state.get("query_graph_reflected")),
                "queryGraphValidated": bool(state.get("query_graph_validated")),
                "sqlGenerated": bool(state.get("sql_generated")),
                "sqlRepairReviewed": bool(state.get("sql_repair_reviewed")),
                "evidenceGraphVerified": bool(state.get("evidence_graph_verified")),
                "evidenceAccepted": bool(evidence_accepted_for_state(state)),
                "analysisWorkerCompleted": bool(state.get("analysis_worker_completed")),
                "skillWorkerCompleted": bool(state.get("skill_worker_completed")),
                "hypothesisExplorationCompleted": bool(state.get("hypothesis_exploration_completed")),
                "chatBiCompleted": bool(state.get("chat_bi_completed")),
            },
            "budgets": {
                "reactRound": int(state.get("react_round") or 0),
                "queryGraphRetrieveCount": int(state.get("query_graph_retrieve_count") or 0),
                "queryGraphSupplementalRetrieveCount": int(state.get("query_graph_supplemental_retrieve_count") or 0),
                "queryGraphPlanAttempts": int(state.get("query_graph_plan_attempts") or 0),
                "queryGraphRepairAttempts": int(state.get("query_graph_repair_attempts") or 0),
            },
            "gaps": {
                "graph": self.gap_payloads(graph_gaps, 8),
                "evidence": self.gap_payloads(evidence_gaps, 8),
                "pendingKnowledge": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in (state.get("pending_knowledge_requests") or [])[:6]
                ],
            },
            "plannerRepair": {
                "reflection": observation.get("plannerReflection", {}),
                "requests": observation.get("plannerRepairRequests", []),
                "input": observation.get("plannerRepairInput", {}),
                "delta": observation.get("queryGraphRepairDelta", {}),
            },
            "executionFailures": {
                "sql": sql_failures[:8],
                "toolRuntime": state.get("tool_failures", [])[-6:],
            },
            "executionObservations": observation.get("executionObservations", {}),
            "retrievalStrategy": state.get("recall_strategy", {}),
            "controlledExploration": {
                "hypotheses": (state.get("hypothesis_exploration") or {}).get("hypotheses", [])[:3],
                "candidateGraphs": (state.get("candidate_query_graphs") or {}).get("candidates", [])[:3],
                "budget": (state.get("hypothesis_exploration") or {}).get("budget", {}),
            },
            "workerDispatch": state.get("worker_dispatch_context") or self.worker_dispatch_context(state),
            "analysisDispatch": self.analysis_dispatch_context(state),
            "skillDispatch": state.get("skill_dispatch_context") or self.skill_dispatch_context(state),
            "decisionHints": decision_hints,
        }
        return context

    def recall_strategy_payload(self, fast_understanding: FastUnderstandingResult, round_traces: List[Any]) -> Dict[str, Any]:
        traces: List[Dict[str, Any]] = []
        for trace in round_traces[-8:]:
            traces.append(trace.model_dump(by_alias=True) if hasattr(trace, "model_dump") else dict(trace or {}))
        profiles = [trace.get("retrievalProfile") or {} for trace in traces]
        query_types = dedupe_texts([str(profile.get("queryType") or trace.get("queryType") or "") for profile, trace in zip(profiles, traces)])
        profile_kinds = dedupe_texts([str(profile.get("profileKind") or "") for profile in profiles])
        lanes: List[Dict[str, Any]] = []
        for trace in traces:
            for lane in trace.get("retrievalLanes", []) or trace.get("retrieval_lanes", []) or []:
                if isinstance(lane, dict):
                    lane_key = "%s:%s:%s" % (lane.get("lane"), lane.get("enabled"), lane.get("topK"))
                    if not any(item.get("_key") == lane_key for item in lanes):
                        next_lane = dict(lane)
                        next_lane["_key"] = lane_key
                        lanes.append(next_lane)
        for lane in lanes:
            lane.pop("_key", None)
        top_k = {}
        if profiles:
            last = profiles[-1]
            top_k = {
                "textTopK": int(last.get("textTopK") or 0),
                "vectorTopK": int(last.get("vectorTopK") or 0),
                "broadTextTopK": int(last.get("broadTextTopK") or 0),
                "broadVectorTopK": int(last.get("broadVectorTopK") or 0),
                "hybridTopK": int(last.get("hybridTopK") or 0),
            }
        return {
            "intentKind": fast_understanding.intent_kind,
            "complexity": fast_understanding.complexity,
            "queryTypes": query_types,
            "profileKinds": profile_kinds,
            "topK": top_k,
            "lanes": lanes[:8],
            "roundCount": len(traces),
        }

    def worker_dispatch_context(self, state: AgentState) -> Dict[str, Any]:
        fast = state.get("fast_understanding") or FastUnderstandingResult()
        plan = state.get("plan") or QueryPlan()
        intents = list(getattr(plan, "intents", []) or [])
        node_count = len(intents)
        dependencies = list(getattr(plan, "dependencies", []) or [])
        dependent_ids = {str(getattr(item, "dependent_task_id", "") or "") for item in dependencies}
        root_nodes = [
            str(getattr(intent, "plan_task_id", "") or "")
            for intent in intents
            if str(getattr(intent, "plan_task_id", "") or "") not in dependent_ids
        ]
        parallelizable = node_count > 1 and len([item for item in root_nodes if item]) > 1
        complex_task = fast.complexity in {"medium", "complex"} or fast.intent_kind in {"multi_hop", "analysis", "rule_data_mix"} or node_count > 1
        worker_type = "NodeWorker" if node_count else ""
        reason = "no_query_graph_yet"
        if worker_type and parallelizable:
            reason = "query_graph_has_parallel_roots"
        elif worker_type and complex_task:
            reason = "complex_or_multi_node_query_graph"
        elif worker_type:
            reason = "single_node_controlled_sql"
        return {
            "workerType": worker_type,
            "shouldDispatch": bool(worker_type),
            "complexTask": bool(complex_task),
            "parallelizable": bool(parallelizable),
            "nodeCount": node_count,
            "rootNodeCount": len([item for item in root_nodes if item]),
            "intentKind": fast.intent_kind,
            "complexity": fast.complexity,
            "reason": reason,
        }

    def skill_dispatch_context(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or QueryPlan()
        run_result = state.get("agent_run_result") or AgentRunResult()
        has_rule_context = bool(state.get("rule_recall_context", ""))
        candidate = select_answer_skill(plan, run_result, has_rule_context)
        needs_skill = self.policy.analysis_skill_needed(state)
        headers = answer_skill_headers(self.settings.resources_root / "runtime" / "agent_skills") if needs_skill else []
        return {
            "needsSkillWorker": bool(needs_skill),
            "candidateSkill": candidate,
            "matchMode": str(getattr(self.settings, "answer_skill_match_mode", "")),
            "availableSkillHeaders": self.skill_header_payloads(headers),
        }

    def analysis_dispatch_context(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or QueryPlan()
        understanding = plan.question_understanding or {}
        run_result = state.get("agent_run_result") or AgentRunResult()
        evidence_accepted = evidence_accepted_for_state(state)
        return {
            "genericAnalysisWorkerAvailable": bool(self.policy.analysis_worker_needed(state)),
            "skillWorkerAvailable": bool(self.policy.analysis_skill_needed(state)),
            "directAnswerAvailable": bool(evidence_accepted and getattr(run_result, "task_results", None)),
            "gapAnswerAvailable": bool(state.get("evidence_graph_verified") and not evidence_accepted),
            "analysisIntent": str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or ""),
            "requiresExplanation": boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation"))),
            "evidenceAccepted": bool(evidence_accepted),
            "currentAnalysisSummaryChars": len(str(state.get("analysis_summary") or "")),
        }

    def skill_header_payloads(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for item in headers[:8]:
            payloads.append(
                {
                    "name": str(item.get("name") or ""),
                    "description": str(item.get("description") or "")[:240],
                    "whenToUse": str(item.get("when_to_use") or item.get("whenToUse") or "")[:360],
                    "constraints": [str(value)[:220] for value in (item.get("constraints") or [])[:6]],
                    "requiredInputs": [str(value)[:220] for value in (item.get("required_inputs") or item.get("requiredInputs") or [])[:6]],
                    "path": str(item.get("path") or ""),
                }
            )
        return payloads

    def current_recall_refs(self, state: AgentState) -> List[str]:
        refs: List[str] = []
        bundle = state.get("knowledge_bundle")
        refs.extend(str(item) for item in getattr(bundle, "source_refs", []) or [] if str(item))
        recall_bundle = state.get("recall_bundle")
        for item in getattr(recall_bundle, "items", []) or []:
            ref = str(getattr(item, "doc_id", "") or ((getattr(item, "metadata", {}) or {}).get("semanticRefId")) or "")
            if ref:
                refs.append(ref)
        for round_trace in state.get("recall_rounds", []) or []:
            payload = round_trace.model_dump(by_alias=True) if hasattr(round_trace, "model_dump") else round_trace
            if isinstance(payload, dict):
                refs.extend(str(item) for item in payload.get("sourceRefs", []) or payload.get("source_refs", []) or [] if str(item))
        seen: set[str] = set()
        ordered: List[str] = []
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            ordered.append(ref)
        return ordered

    def gap_payloads(self, gaps: List[Any], limit: int) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for gap in gaps[:limit]:
            if hasattr(gap, "model_dump"):
                payload = gap.model_dump(by_alias=True)
            elif isinstance(gap, dict):
                payload = dict(gap)
            else:
                payload = {"reason": str(gap)}
            payloads.append(
                {
                    "code": str(payload.get("code") or ""),
                    "taskId": str(payload.get("taskId") or payload.get("task_id") or ""),
                    "reason": str(payload.get("reason") or payload.get("message") or "")[:240],
                    "evidence": str(payload.get("evidence") or "")[:240],
                    "severity": str(payload.get("severity") or ""),
                }
            )
        return payloads

    def sql_failure_payloads(self, run_result: AgentRunResult) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for task_result in getattr(run_result, "task_results", []) or []:
            bundle = getattr(task_result, "query_bundle", None) or QueryBundle()
            if not getattr(bundle, "failed", False):
                continue
            validations = getattr(task_result, "validation_results", []) or []
            last_validation = validations[-1] if validations else None
            failures.append(
                {
                    "taskId": str(getattr(task_result, "task_id", "") or ""),
                    "summary": str(getattr(task_result, "summary", "") or "")[:180],
                    "error": str(getattr(bundle, "error", "") or "")[:240],
                    "validationErrorCode": str(getattr(last_validation, "error_code", "") or ""),
                    "validationMessage": str(getattr(last_validation, "message", "") or "")[:240],
                    "repairAttempts": len(getattr(task_result, "sql_repairs", []) or []),
                }
            )
        return failures

    def execution_observation_payload(
        self,
        state: AgentState,
        run_result: AgentRunResult,
    ) -> Dict[str, Any]:
        """Return bounded, result-only signals for the next Core ReAct turn."""

        task_payloads: List[Dict[str, Any]] = []
        zero_row_task_ids: List[str] = []
        task_results = list(getattr(run_result, "task_results", None) or [])[:12]
        for task_result in task_results:
            bundle = getattr(task_result, "query_bundle", None) or QueryBundle()
            task_id = str(getattr(task_result, "task_id", "") or "")
            row_count = int(bundle.effective_row_count())
            failed = bool(getattr(bundle, "failed", False) or not getattr(task_result, "success", False))
            error = str(
                getattr(bundle, "error", "")
                or (getattr(task_result, "summary", "") if failed else "")
                or ""
            )[:300]
            tables = list(
                dict.fromkeys(
                    str(table)
                    for table in (getattr(bundle, "tables", None) or [])
                    if str(table)
                )
            )[:6]
            task_payloads.append(
                {
                    "taskId": task_id,
                    "tables": tables,
                    "rowCount": row_count,
                    "failed": failed,
                    "error": error,
                }
            )
            if task_id and not failed and row_count == 0:
                zero_row_task_ids.append(task_id)

        freshness_payloads: List[Dict[str, Any]] = []
        freshness_reports = list(
            getattr(run_result, "freshness_reports", None)
            or state.get("freshness_reports")
            or []
        )[:12]
        for report in freshness_reports:
            freshness_payloads.append(
                {
                    "taskId": str(getattr(report, "task_id", "") or ""),
                    "table": str(getattr(report, "table", "") or ""),
                    "requestedDays": int(getattr(report, "requested_days", 0) or 0),
                    "status": str(getattr(report, "status", "") or ""),
                    "maxTimeValue": str(getattr(report, "max_time_value", "") or ""),
                    "fallbackTable": str(getattr(report, "fallback_table", "") or ""),
                    "coverageComplete": bool(getattr(report, "coverage_complete", False)),
                    "alignmentStatus": str(getattr(report, "alignment_status", "") or ""),
                    "reason": str(getattr(report, "reason", "") or "")[:240],
                }
            )

        repairs = list(getattr(run_result, "sql_repairs", None) or [])
        if not repairs:
            repairs = [
                repair
                for task_result in task_results
                for repair in (getattr(task_result, "sql_repairs", None) or [])
            ]
        repair_payloads = [
            {
                "taskId": str(getattr(repair, "task_id", "") or ""),
                "round": int(getattr(repair, "round", 0) or 0),
                "errorCode": str(getattr(repair, "error_code", "") or ""),
                "status": str(getattr(repair, "status", "") or ""),
                "success": bool(getattr(repair, "success", False)),
                "progressed": bool(getattr(repair, "progressed", False)),
                "exhausted": bool(getattr(repair, "exhausted", False)),
                "observation": str(getattr(repair, "observation", "") or "")[:240],
            }
            for repair in repairs[:12]
        ]

        alignment = getattr(run_result, "snapshot_alignment", None)
        snapshot_payload = {
            "status": str(getattr(alignment, "status", "") or "NOT_APPLICABLE"),
            "aligned": bool(getattr(alignment, "aligned", False)),
            "complete": bool(getattr(alignment, "complete", False)),
            "commonAnchorTimeValue": str(
                getattr(alignment, "common_anchor_time_value", "") or ""
            ),
            "disclosureRequired": bool(
                getattr(alignment, "disclosure_required", False)
            ),
            "reason": str(getattr(alignment, "reason", "") or "")[:240],
        }
        return {
            "tasks": task_payloads,
            "freshness": freshness_payloads,
            "zeroRowTaskIds": list(dict.fromkeys(zero_row_task_ids))[:12],
            "sqlRepairAttempts": repair_payloads,
            "snapshotAlignment": snapshot_payload,
        }

    def last_action_repeat_count(self, state: AgentState) -> tuple[str, int]:
        history = state.get("action_history") or []
        if not history:
            return "", 0
        last = str(getattr(history[-1], "action", "") or "")
        count = 0
        for item in reversed(history):
            if str(getattr(item, "action", "") or "") != last:
                break
            count += 1
        return last, count

    def ensure_terminal_planning_gap(self, state: AgentState, decision: Any) -> None:
        if getattr(decision, "selected_action", "") not in {"answer_data", "answer"}:
            return
        if state.get("chat_bi_completed"):
            return
        plan = state.get("plan") or QueryPlan()
        validation = state.get("query_graph_validation_result") or GraphValidationResult()
        run_result = state.get("agent_run_result")
        has_task_results = bool(getattr(run_result, "task_results", None))
        if plan.intents or validation.gaps or has_task_results:
            return
        reason = str(getattr(decision, "reason", "") or "LeadAgent selected answer without executable QueryGraph")
        gap_code = "AGENT_DECISION_EXHAUSTED" if getattr(decision, "budget_exhausted", False) else "MISSING_QUERY_GRAPH"
        record_graph_validation(
            state,
            GraphValidationResult(
                valid=False,
                repairable=False,
                gaps=[
                    GraphValidationGap(
                        code=gap_code,
                        reason=reason,
                    )
                ],
            ),
        )

    def route_topic(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "route_topic", "LeadAgent", "ROUTE_TOPIC", input_summary=state.get("question", ""))
        increment_round(state)
        emit(state, "node.started", "ROUTE_TOPIC", {})
        question_changed = str(state.get("_route_understanding_question") or "") != str(state.get("question") or "")
        if question_changed or not state.get("_route_slots_bootstrapped"):
            keywords = self.keyword_service.extract(state["question"])
            state["extracted_keywords"] = keywords
            route_slots = self.route_slot_extractor.extract(state["question"], keywords)
            route_slots = self.apply_clarification_to_route_slots(state, route_slots)
            state["route_slots"] = route_slots
            state["routing_decision"] = self.routing_service.route(state["question"], keywords, RecallBundle())
            state["_route_slots_bootstrapped"] = True
            state["_route_understanding_question"] = state["question"]
            self.append_route_slots_trace(state, route_slots)
        else:
            keywords = state.get("extracted_keywords") or ExtractedKeywords()
            route_slots = state.get("route_slots") or self.route_slot_extractor.extract(state["question"], keywords)
            state["route_slots"] = route_slots
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            state["topic_routed"] = True
            self.record_span(state, "action", "route_topic", started)
            self.finish_run_step(state, step, "skipped", output_summary="non_business")
            return state
        context_topic = state["request_context"].topic if state.get("request_context") else ""
        session_topics = self.session_topic_categories(state)
        state["route_slots"] = route_slots
        if route_slots.operation == "write_requested":
            self.request_human_clarification(
                state,
                "当前 BI Agent 只支持只读查询和分析，不能执行写操作。请改成只读查询问题。",
                "UNSUPPORTED_OPERATION",
                "write_operation",
                ["改成只读查询", "取消本次操作"],
            )
            state["topic_routing_decision"] = TopicRoutingDecision(
                primary_topic=QuestionCategory.UNKNOWN,
                candidate_topics=[],
                confidence=route_slots.route_confidence,
                clarification_required=True,
                reason="检测到写操作请求；route_topic 只允许只读 BI 查询",
            )
            state["topic_routed"] = True
            state["context_loaded"] = True
            state["scope_clarified"] = False
            add_step(state, "RouteSlots：检测到写操作请求，进入 ask_human，不进入 BI QueryGraph")
            self.record_span(state, "action", "route_topic", started)
            self.finish_run_step(state, step, "success", output_summary="write_operation_blocked")
            emit(state, "node.completed", "ROUTE_TOPIC", {"routeSlots": route_slots.model_dump(by_alias=True), "clarificationRequired": True})
            return state
        decision = self.topic_router.route(
            state["question"],
            state.get("extracted_keywords", ExtractedKeywords()),
            context_topic,
            route_slots=route_slots,
            context_topics=session_topics,
            context_locked=self.session_topic_scope_is_locked(state),
        )
        decision, route_llm_trace = self.apply_bounded_route_llm_decision(state, decision, route_slots)
        state["bounded_route_llm_trace"] = route_llm_trace
        self.bootstrap_asset_diagnostic_state(state)
        state["route_decision_trace"].append(
            {
                "stage": "topic_router",
                "candidateTopics": [enum_value(item) for item in decision.candidate_topics],
                "confidence": decision.confidence,
                "selectionMode": decision.selection_mode,
                "selectionEvidence": dict(decision.selection_evidence or {}),
                "reason": decision.reason,
            }
        )
        forced_clarification_reason = self.topic_clarification_gate_reason(state, decision, route_slots)
        if forced_clarification_reason:
            decision.clarification_required = True
            decision.reason = "; ".join([item for item in [decision.reason, forced_clarification_reason] if item])
            state["route_decision_trace"].append(
                {
                    "stage": "topic_clarification_gate",
                    "candidateTopics": [enum_value(item) for item in decision.recall_topics()],
                    "confidence": decision.confidence,
                    "routeConfidence": route_slots.route_confidence,
                    "warnings": route_slots.route_warnings,
                    "reason": forced_clarification_reason,
                }
            )
        self.apply_topic_workspace_policy(state, decision, route_slots)
        state["topic_routing_decision"] = decision
        try:
            always_rules = self.recall_service.topic_assets.always_apply_rules(
                decision.recall_topics(),
                user_scope=state.get("user_identity") or {},
                merchant_id=state.get("requested_merchant_id", ""),
            )
        except TypeError:
            always_rules = self.recall_service.topic_assets.always_apply_rules(decision.recall_topics())
        state["always_apply_rules"] = always_rules
        state["always_apply_context"] = "\n".join(
            "- [%s/%s] %s：%s" % (
                item.get("topic", ""),
                item.get("tableName", ""),
                item.get("title", "强制规则"),
                item.get("content", ""),
            )
            for item in always_rules[:40]
        )
        state["knowledge_refresh"] = {
            "policy": "refresh_each_business_turn",
            "refreshedAt": datetime.now().isoformat(),
            "topics": list((state.get("topic_workspace") or {}).get("topics") or []),
            "alwaysApplyRuleCount": len(always_rules),
            "historyAuthoritative": False,
        }
        if always_rules:
            add_step(state, "Always Apply：已绕过普通召回并强制注入 %d 条 Topic 业务规则" % len(always_rules))
        state["topic_routed"] = True
        state["context_loaded"] = True
        state["scope_clarified"] = not decision.clarification_required
        if decision.clarification_required:
            self.request_human_clarification(
                state,
                self.build_topic_clarification_prompt(state),
                "BUSINESS_SCOPE",
                "topic_required",
                business_scope_examples(self.recall_service.topic_assets),
            )
        else:
            diagnostic_topics = self.apply_open_diagnostic_policy(state, decision)
            if state.get("human_clarification_required"):
                add_step(state, "Open Diagnostic Policy：开放优先级建议需要先确认排序目标")
                emit(
                    state,
                    "node.completed",
                    "ROUTE_TOPIC",
                    {
                        "topic": decision.display_summary(),
                        "openDiagnostic": self.open_diagnostic_debug(state),
                        "clarificationRequired": True,
                    },
                )
                self.record_span(state, "action", "route_topic", started)
                self.finish_run_step(state, step, "success", output_summary="clarification_required")
                return state
            topics = self._merge_topic_categories(decision.recall_topics(), diagnostic_topics)
            topic_names = self._topic_names_for_categories(topics)
            state["topic_asset_context"] = self.recall_service.topic_assets.load_topic_context(topic_names)
            if diagnostic_topics:
                add_step(
                    state,
                    "Open Diagnostic Policy：识别为开放诊断，先用诊断 seed topics 做窄口径 discovery，不选择 anchor",
                )
            else:
                add_step(state, "Topic Router：已将分析范围收敛到 " + decision.display_summary())
        self.record_span(state, "action", "route_topic", started)
        self.finish_run_step(state, step, "success", output_summary=decision.display_summary())
        emit(
            state,
            "node.completed",
            "ROUTE_TOPIC",
            {"topic": decision.display_summary(), "openDiagnostic": self.open_diagnostic_debug(state), "routeSlots": route_slots.model_dump(by_alias=True)},
        )
        return state

    def apply_topic_workspace_policy(
        self,
        state: AgentState,
        decision: TopicRoutingDecision,
        route_slots: RouteSlots,
    ) -> None:
        """Set the semantic workspace boundary before any knowledge retrieval."""
        topics = decision.recall_topics()
        confidence = max(float(decision.confidence or 0.0), float(route_slots.route_confidence or 0.0))
        context = state.get("request_context")
        open_discovery = bool(
            decision.routing_mode == "open_discovery"
            or state.get("open_diagnostic_intent")
        )
        explicit_single_topic_lock = bool(
            len(topics) == 1
            and (
                contains_any_literal(
                    state.get("question") or "",
                    load_language_policy().routing.scope_lock_markers,
                )
                or (
                    context
                    and context.pending_clarification_type == "topic_required"
                    and getattr(context, "clarification_resolved", False)
                )
            )
        )
        restored_workspace = (state.get("thread_context") or {}).get("topicWorkspace") or {}
        restored_user_lock = bool(
            str(restored_workspace.get("mode") or "") == "explicit_topic_scope"
            or str(restored_workspace.get("expansionPolicy") or "") == "user_locked"
        )
        clarification_confirmed = bool(
            context
            and getattr(context, "clarification_resolved", False)
            and str(getattr(context, "pending_clarification_type", "") or "") == "topic_required"
        )
        user_confirmed = bool(explicit_single_topic_lock or clarification_confirmed or restored_user_lock)
        if open_discovery:
            mode = "open_discovery"
            decision.clarification_required = False
        elif decision.clarification_required or not topics:
            mode = "clarification_required"
            decision.clarification_required = True
        elif explicit_single_topic_lock:
            mode = "explicit_topic_scope"
        elif len(topics) >= 1:
            mode = "topic_workspace"
        else:
            mode = "clarification_required"
            decision.clarification_required = True
        topic_names = self._topic_names_for_categories(topics)
        high_risk = str(route_slots.risk_level or "") in {"high_risk", "rule_sensitive"}
        decision.routing_mode = mode
        decision.workspace_topics = topics if mode not in {"clarification_required", "open_discovery"} else []
        decision.scope_disclosure_required = bool(high_risk or mode == "topic_workspace")
        if mode == "explicit_topic_scope":
            topic_role = "explicit_boundary"
            expansion_policy = "user_locked"
        elif mode == "topic_workspace":
            topic_role = "seed_workspace"
            expansion_policy = "filesystem_manifest_browse"
        elif mode == "open_discovery":
            topic_role = "discovery_seed"
            expansion_policy = "coverage_and_relationship_driven"
        else:
            topic_role = "needs_clarification"
            expansion_policy = "ask_human"
        workspace = {
            "mode": mode,
            "topics": topic_names,
            "topicIds": [enum_value(item) for item in topics],
            "confidence": round(confidence, 3),
            "confirmedByUser": user_confirmed,
            "isolated": mode == "explicit_topic_scope",
            "allowCrossTopic": mode == "open_discovery",
            "topicRole": topic_role,
            "expansionPolicy": expansion_policy,
            "scopeDisclosureRequired": decision.scope_disclosure_required,
            "knowledgeRefreshPolicy": "refresh_each_business_turn",
            "openedTopics": [],
            "effectiveTopics": topic_names,
        }
        state["topic_workspace"] = workspace
        state["analysis_scope"] = {
            **workspace,
            "riskLevel": str(route_slots.risk_level or "normal"),
            "timeWindow": route_slots.time_window.model_dump(by_alias=True),
            "objectRefs": [item.model_dump(by_alias=True) for item in route_slots.object_refs],
            "displayText": "当前 Seed Topic：%s；覆盖不足时先浏览全局 Topic Index，再读取候选 manifest"
            % (" + ".join(topic_names) if topic_names else "待确认"),
        }
        state.setdefault("route_decision_trace", []).append({"stage": "topic_workspace", **workspace})

    def fast_understand(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "fast_understand", "LeadAgent", "FAST_UNDERSTAND", input_summary=state.get("question", ""))
        increment_round(state)
        emit(state, "node.started", "FAST_UNDERSTAND", {})
        route = state.get("routing_decision") or RoutingDecision()
        slots = state.get("route_slots") or RouteSlots()
        keywords = state.get("extracted_keywords") or ExtractedKeywords()
        topics = self._effective_topic_categories(state)
        has_data = self.requires_bi_execution(state, keywords)
        has_rule = slots.risk_level in {"rule_sensitive", "high_risk"}
        business_topic_count = len(self._asset_backed_topic_categories(topics))
        object_refs: Dict[str, List[str]] = {}
        for ref in slots.object_refs:
            object_refs.setdefault(ref.ref_type, [])
            if ref.value not in object_refs[ref.ref_type]:
                object_refs[ref.ref_type].append(ref.value)
        metric_phrases = dedupe_texts(list(keywords.metric_keywords or keywords.business_keywords or [])[:12])
        clarified_metric_focus = str((state.get("clarification_resolution") or {}).get("metricFocus") or "")
        if clarified_metric_focus:
            metric_phrases = dedupe_texts([clarified_metric_focus, *metric_phrases])
        # Routing may flag that a semantic goal still needs interpretation, but
        # it must not name that goal.  The Planner sees the raw question and
        # owns the exact analysis relation.
        analysis_requested = bool(
            slots.analysis_signals
            or str(keywords.analysis_intent or "").strip().lower() == "unresolved"
            or state.get("open_diagnostic_intent")
        )
        if route.route == QuestionRoute.GREETING:
            intent_kind = "chat"
            complexity = "simple"
        elif route.route == QuestionRoute.INVALID:
            intent_kind = "invalid"
            complexity = "simple"
        elif slots.operation == "write_requested":
            intent_kind = "write_requested"
            complexity = "simple"
        elif has_rule and not has_data:
            intent_kind = "rule_only"
            complexity = "simple"
        elif has_rule and has_data:
            intent_kind = "rule_data_mix"
            complexity = "complex" if analysis_requested else "medium"
        elif business_topic_count >= 4:
            intent_kind = "multi_hop"
            complexity = "complex"
        elif object_refs and not analysis_requested and len(topics) <= 3:
            intent_kind = "detail_lookup"
            complexity = "simple" if len(topics) <= 2 else "medium"
        # Multiple independently requested measures require one semantic
        # understanding pass even when the user only asks for their values.
        # Whether those measures are related by a comparison, ratio, trend, or
        # no analysis relation at all is Planner-owned; a connective word must
        # never decide that relationship in the routing layer.
        elif len(metric_phrases) >= 2:
            intent_kind = "multi_metric"
            complexity = "medium"
        elif analysis_requested:
            intent_kind = "analysis"
            complexity = "complex"
        elif business_topic_count >= 3:
            intent_kind = "multi_hop"
            complexity = "complex"
        elif has_data:
            intent_kind = "metric_query"
            complexity = "simple"
        else:
            intent_kind = "unknown"
            complexity = "unknown"
        needs_planner = intent_kind not in {"chat", "invalid", "write_requested", "rule_only"} and complexity in {"medium", "complex", "unknown"}
        needs_knowledge = intent_kind not in {"chat", "invalid", "write_requested"}
        suggested_actions = ["answer_rule"] if intent_kind == "rule_only" else ["retrieve_knowledge"]
        if needs_planner:
            suggested_actions.extend(["compact_assets", "plan_graph"])
        confidence = 0.85
        reasons = [
            "topics=%s" % ",".join(enum_value(topic) for topic in topics[:6]),
            "objectRefs=%d" % sum(len(values) for values in object_refs.values()),
            "analysisSignals=%d" % len(slots.analysis_signals),
            "hasRule=%s hasData=%s" % (has_rule, has_data),
            "analysisIntent=%s" % keywords.analysis_intent,
        ]
        if clarified_metric_focus:
            reasons.append("clarifiedMetricFocus=%s" % clarified_metric_focus)
        if complexity == "unknown":
            confidence = 0.45
        elif intent_kind in {"multi_hop", "analysis", "rule_data_mix"}:
            confidence = 0.75
        result = FastUnderstandingResult(
            complexity=complexity,
            intent_kind=intent_kind,
            analysis_intent=keywords.analysis_intent,
            topics=topics,
            object_refs=object_refs,
            time_window_days=slots.time_window.days,
            metric_phrases=metric_phrases,
            needs_planner=needs_planner,
            needs_knowledge=needs_knowledge,
            suggested_actions=dedupe_texts(suggested_actions),
            confidence=confidence,
            reasons=reasons,
            time_range=resolve_time_range(state.get("question", ""), self.settings.business_timezone),
        )
        state["fast_understanding"] = result
        state.setdefault("capability_decisions", {})["metricFastEntry"] = self.policy.fast_metric_decision(result).model_dump(by_alias=True)
        state["latency_optimization"] = self.latency_optimizer.initial_policy(result)
        if state.get("memory_recalled") and state.get("merchant") is not None:
            try:
                state["merchant_profile_summary"] = self.merchant_profile_summary_service.summarize(
                    merchant=state["merchant"],
                    memory_injection=state.get("memory_injection") or {},
                    memory_constraints=state.get("memory_constraints") or [],
                    route_slots=state.get("route_slots", RouteSlots()),
                    fast_understanding=result,
                )
            except Exception as exc:
                append_memory_recall_issue_to_state(
                    state,
                    RetrievalIssue(
                        code="MEMORY_PROFILE_SUMMARY_FAILED",
                        message=str(exc)[:500],
                        backend="profile_summary",
                        lane="enrichment",
                        stage="fast_understand_profile_summary",
                        severity="warning",
                        resolved=True,
                        details={"answerImpact": False},
                    ),
                    enrichment_key="profileSummaryRefresh",
                )
                add_step(state, "Long-term Memory：画像摘要刷新失败，继续使用已召回的记忆与约束")
            self.refresh_context_snapshot(state, "fast_understand")
        context = state.get("request_context")
        if getattr(context, "offloaded_files", None):
            self.escalate_fast_request(state, "attachments require the standard path")
        else:
            self.reconcile_fast_request_agent_gates(state)
        state["fast_understood"] = True
        clarification = self.merchant_clarification_need(state, result)
        if clarification:
            self.request_human_clarification(
                state,
                clarification["question"],
                clarification["stage"],
                clarification["type"],
                clarification["options"],
            )
            state.setdefault("route_decision_trace", []).append(
                {
                    "stage": "merchant_clarification_gate",
                    "type": clarification["type"],
                    "reason": clarification["reason"],
                    "options": clarification["options"],
                }
            )
            add_step(state, "Merchant Clarification：%s" % clarification["reason"])
        add_step(
            state,
            "Fast Understanding：intent=%s complexity=%s needsPlanner=%s"
            % (result.intent_kind, result.complexity, result.needs_planner),
        )
        if (state.get("latency_optimization") or {}).get("eligible"):
            add_step(state, "Latency Optimizer：命中 fast path，后续可跳过反思、Skill 和 Answer LLM")
        self.record_span(
            state,
            "action",
            "fast_understand",
            started,
            metadata=result.model_dump(by_alias=True),
        )
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary="intent=%s complexity=%s" % (result.intent_kind, result.complexity),
        )
        emit(state, "node.completed", "FAST_UNDERSTAND", result.model_dump(by_alias=True))
        return state

    def try_fast_metric(self, state: AgentState) -> AgentState:
        """Try the deterministic path for one published semantic metric only."""
        started = now_ms()
        step = self.start_run_step(
            state,
            "try_fast_metric",
            "LeadAgent",
            "TRY_FAST_METRIC",
            input_summary=state.get("question", ""),
        )
        increment_round(state)
        emit(state, "node.started", "TRY_FAST_METRIC", {})
        state["fast_metric_attempted"] = True
        if str(state.get("memory_recall_status") or "") == "failed":
            state["fast_metric_completed"] = False
            self.escalate_fast_request(state, "governed memory recall is unavailable")
            add_step(state, "Lead Agent Fast Tool：长期记忆召回不可用，拒绝直接快速回答并进入完整证据链")
            emit(
                state,
                "node.completed",
                "TRY_FAST_METRIC",
                {"supported": False, "fallback": "retrieve_knowledge", "reason": "memory_recall_failed"},
            )
            self.record_span(
                state,
                "tool",
                "try_fast_metric",
                started,
                status="gap",
                error_code="MEMORY_RECALL_FAILED",
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="memory recall failed -> full evidence path",
                error_code="MEMORY_RECALL_FAILED",
            )
            return state
        merchant = state.get("merchant")
        semantic_metrics = published_semantic_quick_metrics(
            self.recall_service.topic_assets,
            self._topic_names_for_categories(self._effective_topic_categories(state)),
        )
        response = quick_metric_response(
            state.get("question", ""),
            getattr(merchant, "merchant_id", ""),
            self.node_worker.doris_repository,
            state.get("extracted_keywords"),
            semantic_metrics,
            timezone_name=self.settings.business_timezone,
            artifact_refs=list(
                getattr(state.get("request_context"), "offloaded_files", None) or []
            ),
        )
        if response is None:
            state["fast_metric_completed"] = False
            self.escalate_fast_request(state, "fast metric did not uniquely resolve a published semantic contract")
            add_step(state, "Lead Agent Fast Tool：未唯一命中一个已发布语义指标，回退语义召回和 Planner")
            emit(
                state,
                "node.completed",
                "TRY_FAST_METRIC",
                {"supported": False, "fallback": "retrieve_knowledge", "publishedSemanticMetricCount": len(semantic_metrics)},
            )
            self.record_span(state, "tool", "try_fast_metric", started, status="gap", error_code="FAST_UNSUPPORTED")
            self.finish_run_step(state, step, "gap", output_summary="unsupported -> Planner", error_code="FAST_UNSUPPORTED")
            return state
        debug_trace = response.debug_trace or {}
        semantic_identity = dict(debug_trace.get("semanticMetric") or {})
        if not semantic_identity:
            semantic_identities = list(debug_trace.get("semanticMetrics") or [])
            if len(semantic_identities) == 1:
                semantic_identity = dict(semantic_identities[0] or {})
        if not semantic_identity.get("semanticRefId") or semantic_identity.get("governanceStatus") != "published":
            state["fast_metric_completed"] = False
            self.escalate_fast_request(state, "fast metric result lacked published semantic lineage")
            add_step(state, "Lead Agent Fast Tool：结果缺少已发布语义指标血缘，拒绝快速回答并回退 Planner")
            emit(state, "node.completed", "TRY_FAST_METRIC", {"supported": False, "fallback": "retrieve_knowledge", "reason": "semantic_lineage_missing"})
            self.record_span(state, "tool", "try_fast_metric", started, status="gap", error_code="FAST_SEMANTIC_LINEAGE_MISSING")
            self.finish_run_step(state, step, "gap", output_summary="semantic lineage missing -> Planner", error_code="FAST_SEMANTIC_LINEAGE_MISSING")
            return state
        self.apply_turn_knowledge_refresh_to_fast_response(state, response)
        state["latency_optimization"] = self.latency_optimizer.mark_verified(
            state.get("latency_optimization") or {},
            "single published semantic metric passed the deterministic fast executor",
        )
        self.reconcile_fast_request_agent_gates(state)
        response.id = state["qa_id"]
        state["fast_metric_completed"] = True
        state["fast_metric_response"] = response
        state["answer"] = response.answer
        state["suggestions"] = list(response.suggestions or [])
        state["thinking_steps"] = list(response.thinking_steps or [])
        state["merchant_experience"] = dict(response.merchant_experience or {})
        state["query_bundle"] = QueryBundle(
            tables=list(response.doris_tables or []),
            rows=list(response.data_rows or []),
            summary="published semantic metric definition" if debug_trace.get("definitionOnly") else "verified published semantic single-metric result",
            cache_hit=bool(debug_trace.get("quickMetricCacheHit")),
        )
        if debug_trace.get("definitionOnly"):
            self.attach_fast_metric_definition_state(state, response, semantic_identity)
            state["should_persist"] = False
        else:
            self.attach_fast_metric_evidence_state(state, response, semantic_identity)
            state["should_persist"] = True
        append_memory_recall_outcomes_to_run_result(state, state["agent_run_result"])
        memory_gaps = [
            gap
            for gap in state["agent_run_result"].evidence_gaps
            if str(gap.source or "") == "memory_recall"
        ]
        if memory_gaps:
            verified = state["agent_run_result"].verified_evidence
            verified.gaps = list(state["agent_run_result"].evidence_gaps)
            verified.blocking_gaps = [gap for gap in verified.gaps if gap.severity == "blocking"]
            verified.warning_gaps = [gap for gap in verified.gaps if gap.severity == "warning"]
            verified.answer_guard_required = True
            guarded = self.answer_service._apply_answer_guard(response.answer, state["agent_run_result"])
            response.answer = guarded
            state["answer"] = guarded
        state["chat_bi_completed"] = True
        if debug_trace.get("definitionOnly"):
            add_step(state, "Lead Agent Fast Tool：唯一已发布语义指标覆盖口径问题，采用语义资产口径说明")
        else:
            add_step(state, "Lead Agent Fast Tool：唯一已发布语义指标完整覆盖本轮问题，采用已校验快速结果")
        emit(
            state,
            "node.completed",
            "TRY_FAST_METRIC",
            {"supported": True, "metric": semantic_identity},
        )
        self.record_span(
            state,
            "tool",
            "try_fast_metric",
            started,
            metadata={"supported": True, "tables": response.doris_tables, "semanticMetric": semantic_identity},
        )
        self.finish_run_step(state, step, "success", output_summary="single published semantic metric accepted")
        return state

    def apply_turn_knowledge_refresh_to_fast_response(self, state: AgentState, response: ChatResponse) -> None:
        rules = list(state.get("always_apply_rules") or [])
        refresh = {
            **dict(state.get("knowledge_refresh") or {}),
            "policy": "refresh_each_business_turn",
            "historyAuthoritative": False,
            "fastPathUsesLatestMandatoryRules": True,
        }
        public_rules = [
            {
                "topic": str(item.get("topic") or ""),
                "tableName": str(item.get("tableName") or ""),
                "title": str(item.get("title") or "强制规则"),
                "content": str(item.get("content") or "")[:500],
            }
            for item in rules[:8]
            if str(item.get("content") or "").strip()
        ]
        response.merchant_experience = dict(response.merchant_experience or {})
        response.merchant_experience["knowledgeRefresh"] = refresh
        if public_rules:
            response.merchant_experience["platformRules"] = public_rules
        response.debug_trace = dict(response.debug_trace or {})
        response.debug_trace["knowledgeRefresh"] = {
            **refresh,
            "alwaysApplyRuleCount": len(rules),
            "ruleRefs": [
                "%s/%s/%s"
                % (
                    str(item.get("topic") or ""),
                    str(item.get("tableName") or ""),
                    str(item.get("ruleId") or item.get("title") or "rule"),
                )
                for item in rules[:12]
            ],
        }
        steps = list(response.thinking_steps or [])
        if "刷新本轮平台/Topic 强制规则" not in steps:
            response.thinking_steps = ["刷新本轮平台/Topic 强制规则", *steps]
        if rules:
            add_step(state, "Fast Metric Knowledge Refresh：已将 %d 条本轮 Topic 强制规则注入快速指标答案" % len(rules))

    def attach_fast_metric_definition_state(self, state: AgentState, response: ChatResponse, semantic_identity: Dict[str, Any]) -> None:
        debug_trace = response.debug_trace or {}
        metric_disclosures = (response.merchant_experience or {}).get("metricDisclosures") or [{}]
        metric_key = str(semantic_identity.get("metricKey") or "")
        display_name = str(metric_disclosures[0].get("displayName") or metric_key or debug_trace.get("metric") or "指标口径")
        table = str(semantic_identity.get("table") or ((response.doris_tables or [""])[0] if response.doris_tables else ""))
        topic_name = str(semantic_identity.get("category") or semantic_identity.get("topic") or "")
        category = self.recall_service.topic_assets.resolve_topic_category(topic_name)
        state["plan"] = QueryPlan(
            intents=[
                QuestionIntent(
                    question=state.get("question", ""),
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.RULE,
                    category=category,
                    plan_task_id="quick_metric_definition",
                    preferred_table=table,
                    metric_name=display_name,
                    metric_resolution={
                        "metricKey": metric_key,
                        "displayName": display_name,
                        "semanticRefId": str(semantic_identity.get("semanticRefId") or ""),
                        "metricGovernanceMode": "published_semantic_definition",
                    },
                )
            ],
            agent_trace=["fast_metric_published_semantic_definition"],
        )
        state["agent_run_result"] = AgentRunResult(
            query_bundles=[state.get("query_bundle") or QueryBundle()],
            merged_query_bundle=state.get("query_bundle") or QueryBundle(),
            verified_evidence=VerifiedEvidence(passed=False, summary="semantic definition only; no Doris evidence required"),
        )
        state["sql_generated"] = False
        state["evidence_graph_verified"] = False
        state["verification_status"] = "semantic_definition"
        state["evidence_accepted"] = False

    def attach_fast_metric_evidence_state(self, state: AgentState, response: ChatResponse, semantic_identity: Dict[str, Any]) -> None:
        debug_trace = response.debug_trace or {}
        metric_disclosures = (response.merchant_experience or {}).get("metricDisclosures") or [{}]
        metric_key = str(semantic_identity.get("metricKey") or "")
        display_name = str(metric_disclosures[0].get("displayName") or metric_key or debug_trace.get("metric") or "value")
        table = str(semantic_identity.get("table") or ((response.doris_tables or [""])[0] if response.doris_tables else ""))
        topic_name = str(semantic_identity.get("category") or semantic_identity.get("topic") or "")
        category = self.recall_service.topic_assets.resolve_topic_category(topic_name)
        days = int(debug_trace.get("days") or ((debug_trace.get("timeRange") or {}).get("days") or 0))
        summary_rows: List[Dict[str, Any]] = []
        for section in response.data_sections or []:
            if str(getattr(section, "result_role", "") or "") == "summary":
                summary_rows.extend(list(getattr(section, "data_rows", []) or []))
        if not summary_rows:
            summary_rows = list(response.data_rows or [])[:1]
        summary_bundle = QueryBundle(
            tables=[table] if table else list(response.doris_tables or []),
            rows=summary_rows,
            original_row_count=len(summary_rows),
            summary="verified published semantic single-metric result",
        )
        trend_bundle = QueryBundle(
            tables=[table] if table else list(response.doris_tables or []),
            rows=list(response.data_rows or []),
            original_row_count=len(response.data_rows or []),
            summary="verified published semantic single-metric trend result",
        )
        state["plan"] = QueryPlan(
            intents=[
                QuestionIntent(
                    question=state.get("question", ""),
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.METRIC,
                    category=category,
                    plan_task_id="quick_metric_summary",
                    preferred_table=table,
                    metric_name=display_name,
                    metric_column="value",
                    metric_resolution={
                        "metricKey": metric_key,
                        "displayName": display_name,
                        "semanticRefId": str(semantic_identity.get("semanticRefId") or ""),
                    },
                    days=days,
                )
            ],
            agent_trace=["fast_metric_verified_semantic_contract"],
        )
        state["agent_run_result"] = AgentRunResult(
            task_results=[
                AgentTaskResult(task_id="quick_metric_summary", success=True, query_bundle=summary_bundle),
                AgentTaskResult(task_id="quick_metric_trend", success=True, query_bundle=trend_bundle),
            ],
            query_bundles=[summary_bundle, trend_bundle],
            merged_query_bundle=summary_bundle,
            verified_evidence=VerifiedEvidence(passed=True),
        )
        state["sql_generated"] = True
        state["evidence_graph_verified"] = True
        state["verification_status"] = "passed"
        state["evidence_accepted"] = True
        state["result_generation"] = int(state.get("execution_generation") or 0)
        state["evidence_generation"] = int(state.get("execution_generation") or 0)

    def merchant_clarification_need(self, state: AgentState, fast: FastUnderstandingResult) -> Dict[str, Any]:
        if not bool(getattr(self.settings, "merchant_clarification_enabled", True)):
            return {}
        if state.get("human_clarification_required"):
            return {}
        route = state.get("routing_decision") or RoutingDecision()
        if route.route != QuestionRoute.BUSINESS:
            return {}
        context = state.get("request_context")
        if context and context.pending_clarification_stage:
            return {}
        slots = state.get("route_slots") or RouteSlots()
        topics = set(fast.topics or self._effective_topic_categories(state))
        business_topics = set(self._asset_backed_topic_categories(list(topics)))
        if fast.intent_kind in {"chat", "invalid", "write_requested", "rule_only"}:
            return {}
        if is_metric_definition_question(str(state.get("question") or "")):
            return {}
        if state.get("open_diagnostic_intent"):
            return {}
        if (
            slots.time_window.days <= 0
            and not (state.get("extracted_keywords") or ExtractedKeywords()).time_keywords
            and business_topics
            and self.question_needs_time_clarification(fast)
        ):
            contract = topic_clarification_contract(self.recall_service.topic_assets, "time_window")
            return {
                "stage": "BUSINESS_SCOPE",
                "type": "time_window",
                "question": str(contract.get("prompt") or "请补充本次查询的时间范围。"),
                "options": list(contract.get("options") or []),
                "reason": "经营查询缺少时间范围，先确认口径避免默认时间误导",
            }
        if (
            not fast.metric_phrases
            and fast.intent_kind in {"analysis", "multi_hop", "unknown"}
            and self.question_needs_metric_clarification(fast)
        ):
            contract = topic_clarification_contract(self.recall_service.topic_assets, "metric_focus")
            return {
                "stage": "BUSINESS_SCOPE",
                "type": "metric_focus",
                "question": str(contract.get("prompt") or "请补充希望分析的指标或目标。"),
                "options": list(contract.get("options") or []),
                "reason": "开放经营分析缺少目标指标，先确认分析重心",
            }
        return {}

    def question_needs_time_clarification(self, fast: FastUnderstandingResult) -> bool:
        return str(fast.intent_kind or "") not in {"chat", "invalid", "write_requested", "rule_only", "detail_lookup"}

    def question_needs_metric_clarification(self, fast: FastUnderstandingResult) -> bool:
        return not bool(fast.metric_phrases)

    def apply_bounded_route_llm_decision(
        self,
        state: AgentState,
        decision: TopicRoutingDecision,
        route_slots: RouteSlots,
    ) -> tuple[TopicRoutingDecision, Dict[str, Any]]:
        mode = str(getattr(self.settings, "route_llm_mode", "low_confidence") or "low_confidence").lower()
        trace: Dict[str, Any] = {
            "mode": mode,
            "status": "skipped",
            "reason": "deterministic_route_confident",
            "allowedTopics": [enum_value(item) for item in decision.recall_topics()],
            "selectionMode": decision.selection_mode,
            "selectionEvidence": dict(decision.selection_evidence or {}),
        }
        if mode in {"off", "false", "0", "disabled"}:
            trace["reason"] = "route_llm_disabled"
            return decision, trace
        should_call = mode == "always" or (
            mode == "low_confidence"
            and (
                float(route_slots.route_confidence or 0.0) < 0.55
                or bool(route_slots.route_warnings)
                or len(decision.recall_topics()) >= 5
            )
        )
        if not should_call:
            return decision, trace
        llm = getattr(self.planner, "llm", None)
        if not llm or not getattr(llm, "configured", False):
            trace.update({"status": "skipped", "reason": "llm_not_configured"})
            return decision, trace
        allowed = [enum_value(item) for item in decision.recall_topics()]
        if not allowed:
            allowed = [enum_value(item.topic) for item in route_slots.topic_candidates if item.topic != QuestionCategory.UNKNOWN]
        prompt = {
            "question": state.get("question", ""),
            "routeSlots": route_slots.model_dump(by_alias=True),
            "allowedTopics": allowed,
            "selectionEvidence": dict(decision.selection_evidence or {}),
            "instruction": (
                "Topic 是内部取数工作区，不让商家选择。只允许从 allowedTopics 中保留或删除 topic，"
                "不允许新增未知 topic；区分业务归属 businessTopics 与实际 servingTopics，"
                "根据查询粒度、指标能力和 detailMetricRef 选择。返回 JSON: "
                "{topics:[], confidence:0-1, reason:''}"
            ),
        }
        try:
            payload = llm.json_chat(
                "你是 BI Agent 的受限路由确认器，只能在给定 topic 集合内做选择。",
                json.dumps(prompt, ensure_ascii=False),
                fallback={},
                timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
            )
        except Exception as exc:
            route_slots.route_warnings.append("ROUTE_LLM_TIMEOUT")
            trace.update({"status": "failed", "errorCode": "ROUTE_LLM_TIMEOUT", "errorMessage": str(exc)[:300]})
            return decision, trace
        if not payload and getattr(llm, "last_error", ""):
            code = "ROUTE_LLM_TIMEOUT" if "timeout" in str(llm.last_error).lower() else "ROUTE_LLM_FAILED"
            route_slots.route_warnings.append(code)
            trace.update({"status": "failed", "errorCode": code, "errorMessage": str(llm.last_error)[:300]})
            return decision, trace
        topics = [str(item) for item in (payload or {}).get("topics", []) if str(item) in allowed]
        if not topics:
            trace.update({"status": "ignored", "reason": "llm_returned_no_allowed_topics", "payload": payload or {}})
            return decision, trace
        categories = []
        for item in topics:
            try:
                categories.append(QuestionCategory(item))
            except Exception:
                continue
        if not categories:
            trace.update({"status": "ignored", "reason": "llm_topics_failed_enum_parse", "payload": payload or {}})
            return decision, trace
        decision.candidate_topics = categories
        decision.primary_topic = route_primary_topic(categories)
        decision.dimension_topics = [] if decision.primary_topic == QuestionCategory.UNKNOWN else categories[1:]
        decision.confidence = float((payload or {}).get("confidence") or decision.confidence or route_slots.route_confidence or 0.0)
        decision.selection_mode = "automatic_model_confirmed"
        decision.selection_evidence = {
            **dict(decision.selection_evidence or {}),
            "modelSelectedTopics": topics,
            "modelReason": str((payload or {}).get("reason") or ""),
        }
        decision.reason = "受限 route LLM 确认；多 topic 时 primaryTopic 保持 UNKNOWN，不表示 anchor。%s" % str(
            (payload or {}).get("reason") or ""
        )
        trace.update({"status": "applied", "topics": topics, "payload": payload or {}})
        return decision, trace

    def topic_clarification_gate_reason(self, state: AgentState, decision: TopicRoutingDecision, route_slots: RouteSlots) -> str:
        if not bool(getattr(self.settings, "route_force_clarification_enabled", True)):
            return ""
        if decision.routing_mode == "open_discovery":
            return ""
        context = state.get("request_context")
        if state.get("open_diagnostic_intent"):
            return ""
        profile = state.get("open_diagnostic_profile") or {}
        if context and str(context.pending_clarification_type or "") == str(profile.get("clarificationType") or ""):
            return ""
        topics = decision.recall_topics()
        business_topics = [topic for topic in topics if topic != QuestionCategory.UNKNOWN]
        data_topics = self._asset_backed_topic_categories(business_topics)
        confidence = max(float(decision.confidence or 0.0), float(route_slots.route_confidence or 0.0))
        min_confidence = float(getattr(self.settings, "route_topic_min_confidence", 0.52) or 0.52)
        max_candidates = int(getattr(self.settings, "route_topic_max_candidates", 4) or 4)
        mixed_min_confidence = float(getattr(self.settings, "route_mixed_rule_data_min_confidence", 0.75) or 0.75)
        if not business_topics and confidence < min_confidence:
            return "业务意图低置信且没有明确指标或对象，先让用户补充业务目标；不要求用户选择内部 Topic"
        if confidence < min_confidence and ("NO_EXPLICIT_TOPIC" in route_slots.route_warnings or not route_slots.object_refs):
            return "缺少明确对象或业务关键词，先确认要看的指标、对象或时间范围；Topic 仍由系统自动选择"
        if len(business_topics) > max_candidates and confidence < 0.82:
            return "候选工作区过多，先确认优先看的业务指标或对象；不向用户暴露内部 Topic 选择"
        if route_slots.risk_level == "rule_sensitive" and data_topics and confidence < mixed_min_confidence:
            return "规则问题和数据分析域混在一起且置信度不足，先确认是查规则还是查经营数据"
        if "BROAD_TOPIC_SET" in route_slots.route_warnings and confidence < mixed_min_confidence:
            return "路由命中范围过宽，先确认要看的指标或对象"
        return ""

    def load_skill_policies_for_retrieval(self, state: AgentState) -> List[str]:
        skills = self.asset_builder.skill_loader.select(state["question"], self._effective_topic_categories(state))
        state["loaded_skills"] = [skill.domain for skill in skills]
        state["skills_loaded"] = True
        return state["loaded_skills"]

    def retrieve_knowledge(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "retrieve_knowledge", "KnowledgeAgent", "RETRIEVE_KNOWLEDGE", input_summary=state.get("question", ""))
        # The first Topic-scoped recall is context bootstrap, not a business
        # planning action chosen by the Core.  It runs immediately after Topic
        # selection so the first Core observation can contain both the full L0
        # table manifest and thin retrieval hints.  Supplemental recalls remain
        # normal ReAct actions and therefore consume a round.
        initial_topic_recall = bool(state.get("_initial_topic_recall"))
        core_targeted_topic_recall = bool(state.get("_core_targeted_topic_recall"))
        strict_topic_recall = initial_topic_recall or core_targeted_topic_recall
        if not initial_topic_recall:
            increment_round(state)
        self.configure_artifact_roots(state)
        state["query_graph_retrieve_count"] = int(state.get("query_graph_retrieve_count") or 0) + 1
        emit(state, "node.started", "RETRIEVE_KNOWLEDGE", {})
        pending_requests = dedupe_workflow_knowledge_requests(list(state.get("pending_knowledge_requests") or []))
        blocked_request_keys = set(state.get("blocked_knowledge_request_keys") or [])
        active_pending_requests = [
            request for request in pending_requests if knowledge_request_key(request) not in blocked_request_keys
        ]
        stalled_pending_requests = [
            request for request in pending_requests if knowledge_request_key(request) in blocked_request_keys
        ]
        if stalled_pending_requests:
            state["knowledge_request_gaps"] = append_knowledge_request_gaps(
                state.get("knowledge_request_gaps", []),
                stalled_pending_requests,
                "METRIC_EVIDENCE_UNCHANGED",
            )
        state["pending_knowledge_requests"] = active_pending_requests
        had_pending_requests = bool(active_pending_requests)
        resume_repair_after_retrieval = bool(
            had_pending_requests
            and self.awaiting_query_graph_repair_input(state) is not None
        )
        was_data_discovered = bool(state.get("data_discovered"))
        if was_data_discovered or had_pending_requests:
            state["query_graph_supplemental_retrieve_count"] = int(state.get("query_graph_supplemental_retrieve_count") or 0) + 1
        base_topics = self._effective_topic_categories(state)
        fast_understanding = state.get("fast_understanding") or FastUnderstandingResult()
        query_scopes: List[tuple[str, List[QuestionCategory], Optional[KnowledgeRequest]]] = [(state["question"], base_topics, None)]
        # Initial retrieval must preserve the complete user question.  Keyword
        # slots and fast-understanding phrases are advisory signals only; using
        # them to rewrite the first query would give them planning authority and
        # can expand a narrow Topic merely because one word matched elsewhere.
        if not initial_topic_recall:
            route_query = self.route_recall_query(state)
            if route_query and route_query != state["question"]:
                query_scopes.insert(0, (route_query, base_topics, None))
        merged = state.get("recall_bundle") or RecallBundle()
        all_items = {item.doc_id: item for item in merged.items}
        existing_refs = set(all_items)
        round_traces = list(state.get("recall_rounds") or [])
        expanded_topics = list(state.get("knowledge_expanded_topics") or [])
        knowledge_bundles: List[KnowledgeBundle] = []
        retrieval_outcomes: List[Dict[str, Any]] = []
        stage_items: Dict[str, Dict[str, RecallItem]] = {}
        request_result_items: Dict[str, Dict[str, RecallItem]] = {}

        def run_recall_scopes(scopes: List[tuple[str, List[QuestionCategory], Optional[KnowledgeRequest]]], stage: str) -> None:
            nonlocal expanded_topics
            stage_bucket = stage_items.setdefault(stage, {})
            for query, query_topics, request in scopes[:5]:
                request_key = knowledge_request_key(request) if request is not None else ""
                request_bucket = request_result_items.setdefault(request_key, {}) if request_key else {}
                expanded_topics = self._merge_topic_categories(expanded_topics, query_topics)
                keywords = self.keyword_service.extract(query)
                retrieval_request = KnowledgeRetrievalRequest(
                    query=query,
                    keywords=keywords.keywords,
                    history_rows=state.get("history_rows", []),
                    knowledge_context=knowledge_context(state),
                    merchant_id=state["merchant"].merchant_id,
                    access_role=state.get("access_role", load_authorization_policy().default_access_role),
                    permissions=list((state.get("user_identity") or {}).get("permissions") or []),
                    previous_user_question=previous_user_question(
                        state.get("message_history") or [],
                        current_question=state.get("question", ""),
                    ),
                    session_context=preserve_priority_context_window(str(state.get("session_context") or ""), 4000),
                    topic_categories=query_topics,
                    knowledge_request=request,
                    route_slots=(state.get("route_slots") or RouteSlots()).model_dump(by_alias=True),
                    intent_kind=fast_understanding.intent_kind,
                    complexity=fast_understanding.complexity,
                    round=int(state.get("query_graph_retrieve_count") or 0),
                    strict_topic_scope=strict_topic_recall,
                )
                retriever_backend = str(getattr(self.knowledge_retriever, "backend_name", "knowledge") or "knowledge")
                try:
                    knowledge_bundle = normalize_knowledge_bundle_status(
                        self.knowledge_retriever.retrieve(retrieval_request)
                    )
                except Exception as exc:
                    knowledge_bundle = failed_knowledge_bundle(
                        retrieval_request,
                        retriever_backend,
                        "KNOWLEDGE_RETRIEVER_FAILED",
                        str(exc),
                    )
                if not knowledge_bundle.recall_bundle.items and str(knowledge_bundle.backend or "").lower().startswith("es"):
                    try:
                        fallback_bundle = HybridKnowledgeRetrievalService(self.recall_service).retrieve(retrieval_request)
                    except Exception as exc:
                        fallback_bundle = failed_knowledge_bundle(
                            retrieval_request,
                            "hybrid",
                            "KNOWLEDGE_FALLBACK_RETRIEVER_FAILED",
                            str(exc),
                            stage="fallback",
                        )
                    knowledge_bundle = merge_knowledge_fallback(knowledge_bundle, fallback_bundle)
                knowledge_bundles.append(knowledge_bundle)
                retrieval_outcomes.append(
                    {
                        "stage": stage,
                        "requestKey": request_key,
                        "query": query[:500],
                        "backend": knowledge_bundle.backend,
                        "status": knowledge_bundle.retrieval_status,
                        "itemCount": len(knowledge_bundle.recall_bundle.items),
                        "issues": [
                            issue.model_dump(by_alias=True)
                            for issue in knowledge_bundle.retrieval_issues
                        ],
                    }
                )
                for trace in knowledge_bundle.recall_rounds:
                    trace.new_refs = [ref for ref in trace.source_refs if ref not in existing_refs]
                    existing_refs.update(trace.source_refs)
                    trace_payload = trace.model_dump(by_alias=True)
                    trace_payload["stage"] = stage
                    round_traces.append(trace_payload)
                bundle = knowledge_bundle.recall_bundle
                for item in bundle.items:
                    if request_key:
                        current_request_item = request_bucket.get(item.doc_id)
                        if current_request_item is None or recall_item_sort_key(item) >= recall_item_sort_key(current_request_item):
                            request_bucket[item.doc_id] = item
                    current_stage_item = stage_bucket.get(item.doc_id)
                    if current_stage_item is None or recall_item_sort_key(item) >= recall_item_sort_key(current_stage_item):
                        stage_bucket[item.doc_id] = item
                    current = all_items.get(item.doc_id)
                    if current is not None:
                        item = merge_recall_item_queries(current, item)
                    if current is None or recall_item_sort_key(item) >= recall_item_sort_key(current) or set((item.metadata or {}).get("recallQueries") or []) != set((current.metadata or {}).get("recallQueries") or []):
                        all_items[item.doc_id] = item

        run_recall_scopes(query_scopes, "topic_workspace")
        pending_scopes = [
            (
                request.query,
                (
                    base_topics
                    if strict_topic_recall
                    else self._knowledge_request_topics(request, base_topics)
                ),
                request.model_copy(update={"request_key": knowledge_request_key(request)}),
            )
            for request in active_pending_requests
            if request.query
        ]
        if pending_scopes:
            run_recall_scopes(pending_scopes, "pending_knowledge_request")
        # A pending knowledge request describes a missing fact inside the
        # current workspace; it is not authority to widen the Topic boundary.
        # DeepAgent supplemental recall therefore remains strict just like the
        # automatic first recall. Cross-Topic expansion must be represented by
        # a separate, explicit routing/relationship decision.
        expansion_topics = (
            []
            if strict_topic_recall
            else self.knowledge_recall_expansion_topics(state, base_topics, active_pending_requests)
        )
        coverage_reason = self.knowledge_recall_coverage_gap_reason(
            state,
            list((stage_items.get("topic_workspace") or {}).values()),
            base_topics,
            expansion_topics,
            had_pending_requests=had_pending_requests,
        )
        expansion_scopes: List[tuple[str, List[QuestionCategory], Optional[KnowledgeRequest]]] = []
        if coverage_reason and expansion_topics:
            expanded_scope_topics = self._merge_topic_categories(base_topics, expansion_topics)
            expansion_query = self.route_recall_query(state) or state["question"]
            expansion_scopes.append((expansion_query, expanded_scope_topics, None))
            for request in active_pending_requests[:4]:
                if request.query:
                    expansion_scopes.append(
                        (
                            request.query,
                            self._merge_topic_categories(expanded_scope_topics, self._knowledge_request_topics(request, expanded_scope_topics)),
                            request.model_copy(update={"request_key": knowledge_request_key(request)}),
                        )
                    )
            add_step(
                state,
                "KnowledgeAgent：初始 Topic workspace 召回覆盖不足(%s)，扩展候选 Topic=%s 后二次召回"
                % (coverage_reason, ",".join(enum_value(item) for item in expansion_topics)),
            )
            state.setdefault("knowledge_recall_coverage", {})["topicExpansion"] = {
                "reason": coverage_reason,
                "baseTopics": [enum_value(item) for item in base_topics],
                "expandedTopics": [enum_value(item) for item in expansion_topics],
            }
            run_recall_scopes(expansion_scopes, "topic_expansion")
        else:
            state.setdefault("knowledge_recall_coverage", {})["topicExpansion"] = {
                "reason": coverage_reason or "not_needed",
                "baseTopics": [enum_value(item) for item in base_topics],
                "expandedTopics": [],
            }
        no_match_requests = [
            request
            for request in active_pending_requests
            if not request_result_items.get(knowledge_request_key(request))
        ]
        if no_match_requests:
            blocked = set(state.get("blocked_knowledge_request_keys") or [])
            blocked.update(knowledge_request_key(request) for request in no_match_requests)
            state["blocked_knowledge_request_keys"] = sorted(blocked)
            failed_request_keys = {
                str(outcome.get("requestKey") or "")
                for outcome in retrieval_outcomes
                if str(outcome.get("status") or "") == "failed"
            }
            retrieval_failed_requests = [
                request
                for request in no_match_requests
                if knowledge_request_key(request) in failed_request_keys
            ]
            genuine_no_match_requests = [
                request
                for request in no_match_requests
                if knowledge_request_key(request) not in failed_request_keys
            ]
            gaps = state.get("knowledge_request_gaps", [])
            if retrieval_failed_requests:
                gaps = append_knowledge_request_gaps(
                    gaps,
                    retrieval_failed_requests,
                    "KNOWLEDGE_RETRIEVAL_FAILED",
                )
            if genuine_no_match_requests:
                gaps = append_knowledge_request_gaps(
                    gaps,
                    genuine_no_match_requests,
                    "KNOWLEDGE_REQUEST_NO_MATCH",
                )
            state["knowledge_request_gaps"] = gaps
            add_step(
                state,
                "KnowledgeAgent：%d 个补知识请求未获得 request-specific 证据（召回失败=%d，真实零命中=%d），记录结构化缺口并停止重试"
                % (len(no_match_requests), len(retrieval_failed_requests), len(genuine_no_match_requests)),
            )
        lineage_items = list(all_items.values())
        items = sorted(lineage_items, key=recall_item_sort_key, reverse=True)[:24]
        state["recall_bundle"] = RecallBundle(
            items=items,
            top_score=items[0].fusion_score if items else 0.0,
            merged_context="\n\n".join("召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items),
        )
        if initial_topic_recall:
            state["initial_topic_recall_completed"] = True
            state["initial_topic_recall_trace"] = {
                "query": state.get("question", ""),
                "topics": [enum_value(item) for item in base_topics],
                "itemCount": len(items),
                "topicExpansion": False,
                "role": "navigation_candidates_before_core_table_selection",
            }
        retrieval_issues = dedupe_retrieval_issues(
            [
                issue
                for bundle in knowledge_bundles
                for issue in bundle.retrieval_issues
            ]
        )
        scope_statuses = {
            str(outcome.get("status") or "")
            for outcome in retrieval_outcomes
        }
        if "failed" in scope_statuses:
            retrieval_status = "failed"
        elif retrieval_issues or "degraded" in scope_statuses:
            retrieval_status = "degraded"
        else:
            retrieval_status = "success" if items else "empty"
        backend = next((bundle.backend for bundle in knowledge_bundles if bundle.backend), "hybrid")
        state["knowledge_bundle"] = KnowledgeBundle(
            recall_bundle=state["recall_bundle"],
            source_refs=sorted({item.doc_id for item in items if item.doc_id}),
            recall_rounds=[],
            backend=backend,
            index_version=next((bundle.index_version for bundle in knowledge_bundles if bundle.index_version), ""),
            semantic_source_hash=next((bundle.semantic_source_hash for bundle in knowledge_bundles if bundle.semantic_source_hash), ""),
            retrieval_status=retrieval_status,
            retrieval_issues=retrieval_issues,
        )
        state["knowledge_retrieval_status"] = retrieval_status
        state["knowledge_retrieval_issues"] = [
            issue.model_dump(by_alias=True)
            for issue in retrieval_issues
        ]
        state["knowledge_retrieval_outcomes"] = retrieval_outcomes
        state["recall_rounds"] = round_traces
        state["recall_strategy"] = self.recall_strategy_payload(fast_understanding, round_traces)
        self.update_knowledge_request_lineage(state, active_pending_requests, lineage_items)
        state["recall_context"] = state["recall_bundle"].merged_context
        state["knowledge_expanded_topics"] = expanded_topics
        loaded_skills = self.load_skill_policies_for_retrieval(state)
        state["pending_knowledge_requests"] = []
        state["data_discovered"] = True
        strict_rule_refs = self.rule_recall_ref_ids(state, fallback=False)
        if strict_rule_refs:
            state["rule_recall_refs"] = strict_rule_refs
            state["rule_recall_context"] = self.rule_recall_context(state, fallback=False)
        else:
            state["rule_recall_refs"] = []
            state["rule_recall_context"] = ""
        if self.should_answer_with_rule_recall(state):
            if not state["rule_recall_refs"]:
                state["rule_recall_refs"] = self.rule_recall_ref_ids(state)
                state["rule_recall_context"] = self.rule_recall_context(state)
            state["rule_recall_ready"] = True
            state["should_persist"] = False
            add_step(state, "Rule Recall：命中平台规则知识，等待 LeadAgent 选择 answer_rule")
        else:
            state["rule_recall_ready"] = False
            if state["rule_recall_context"]:
                add_step(state, "Rule Recall：命中平台规则知识，作为后续 Answer 证据，不短路 BI QueryGraph")
        state["intent_signals"] = self.build_intent_signals(state)
        grounded_authority = str(state.get("planning_authority") or "") == "grounded_query_contract"
        if grounded_authority:
            # Grounded-mode recall is navigation only.  It may add thin search
            # candidates, but it has no authority to invalidate an immutable
            # READY Contract, its direct QueryGraph, or verified execution
            # artifacts.  A replacement becomes authoritative only through a
            # successful transactional Contract commit and direct compilation.
            state["grounded_recall_navigation_only"] = True
            add_step(
                state,
                "Grounded Recall：仅更新导航候选，未修改 READY Contract、QueryGraph 或执行证据",
            )
        else:
            state["planning_assets_compacted"] = False
            invalidate_graph_validation(state)
            state["query_graph_reflected"] = False
            if was_data_discovered or had_pending_requests:
                self.invalidate_execution_outputs(state, "知识召回已更新")
        self.planner.artifact_store.write_json("recall", "recall_bundle.json", state["recall_bundle"].model_dump(by_alias=True), preview_chars=0)
        if grounded_authority:
            pass
        elif had_pending_requests and resume_repair_after_retrieval:
            consumed_request_keys = {
                knowledge_request_key(request)
                for request in active_pending_requests
            }
            retained_plan = (state.get("plan") or QueryPlan()).model_copy(deep=True)
            retained_plan.knowledge_requests = [
                request
                for request in retained_plan.knowledge_requests
                if knowledge_request_key(request) not in consumed_request_keys
            ]
            state["plan"] = retained_plan
            self.restore_query_graph_repair_after_knowledge(state)
        elif had_pending_requests:
            state["plan"] = QueryPlan()
            state["query_graph_plan_attempts"] = 0
            state["planner_provider_error"] = ""
        add_step(
            state,
            "Main Agent Tool retrieve_knowledge：检索状态=%s，命中 %d 条候选知识/资产片段，issues=%d，profile=%s skillPolicies=%s"
            % (
                retrieval_status,
                len(items),
                len(retrieval_issues),
                ",".join(state["recall_strategy"].get("profileKinds") or []) or "unknown",
                loaded_skills or [],
            ),
        )
        self.record_span(
            state,
            "semantic_tool",
            "retrieve_knowledge",
            started,
            row_count=len(items),
            status="failed" if retrieval_status == "failed" else ("partial" if retrieval_status == "degraded" else "success"),
            error_code=(
                next((issue.code for issue in retrieval_issues if not issue.resolved), "KNOWLEDGE_RETRIEVAL_FAILED")
                if retrieval_status == "failed"
                else ""
            ),
            metadata={
                "skillPolicies": loaded_skills,
                "pendingRequests": had_pending_requests,
                "recallStrategy": state.get("recall_strategy", {}),
                "retrievalStatus": retrieval_status,
                "retrievalIssues": state.get("knowledge_retrieval_issues", []),
            },
        )
        self.finish_run_step(
            state,
            step,
            "gap" if retrieval_status == "failed" else ("partial" if retrieval_status == "degraded" else "success"),
            output_summary="status=%s recallItems=%d issues=%d skills=%s"
            % (retrieval_status, len(items), len(retrieval_issues), loaded_skills or []),
            error_code=(
                next((issue.code for issue in retrieval_issues if not issue.resolved), "KNOWLEDGE_RETRIEVAL_FAILED")
                if retrieval_status == "failed"
                else ""
            ),
        )
        emit(
            state,
            "node.completed",
            "RETRIEVE_KNOWLEDGE",
            {
                "recallItems": len(items),
                "skillPolicies": loaded_skills,
                "recallStrategy": state.get("recall_strategy", {}),
                "retrievalStatus": retrieval_status,
                "retrievalIssueCount": len(retrieval_issues),
            },
        )
        return state

    def knowledge_recall_expansion_topics(
        self,
        state: AgentState,
        base_topics: List[QuestionCategory],
        active_pending_requests: List[KnowledgeRequest],
    ) -> List[QuestionCategory]:
        workspace = state.get("topic_workspace") or {}
        if workspace.get("mode") == "explicit_topic_scope" or workspace.get("isolated"):
            return []
        candidates: List[QuestionCategory] = []
        slots = state.get("route_slots") or RouteSlots()
        for candidate in slots.topic_candidates:
            candidates = self._merge_topic_categories(candidates, [candidate.topic])
        keywords = state.get("extracted_keywords") or ExtractedKeywords()
        for category_value in getattr(keywords, "topic_scores", {}) or {}:
            try:
                candidates = self._merge_topic_categories(candidates, [QuestionCategory(category_value)])
            except Exception:
                continue
        for request in active_pending_requests:
            candidates = self._merge_topic_categories(candidates, self._knowledge_request_topics(request, base_topics))
        expanded = [
            topic
            for topic in self._merge_topic_categories(candidates, [])
            if topic not in set(base_topics) and topic != QuestionCategory.UNKNOWN
        ]
        return expanded[:4]

    def knowledge_recall_coverage_gap_reason(
        self,
        state: AgentState,
        items: List[RecallItem],
        base_topics: List[QuestionCategory],
        expansion_topics: List[QuestionCategory],
        had_pending_requests: bool = False,
    ) -> str:
        if not expansion_topics:
            return ""
        workspace = state.get("topic_workspace") or {}
        if workspace.get("mode") == "explicit_topic_scope" or workspace.get("isolated"):
            return ""
        if not items:
            return "no_recall_items"
        top_score = max([float(item.fusion_score or 0.0) for item in items] or [0.0])
        bundle = RecallBundle(items=items, top_score=top_score)
        if not bundle.has_strong_match():
            return "weak_recall_match"
        # A pending request is a semantic need, not permission to widen the
        # Topic workspace. Expansion, where legacy callers still support it,
        # must be justified by actual coverage failure rather than request
        # existence alone.
        slot_topics = self._merge_topic_categories(
            [candidate.topic for candidate in (state.get("route_slots") or RouteSlots()).topic_candidates],
            [],
        )
        if len(slot_topics) > len(set(base_topics)):
            return "uncovered_candidate_topics"
        return ""

    def update_knowledge_request_lineage(
        self,
        state: AgentState,
        requests: List[KnowledgeRequest],
        items: List[RecallItem],
    ) -> None:
        if not requests:
            return
        attempts = dict(state.get("knowledge_request_attempts") or {})
        fingerprints = dict(state.get("knowledge_request_fingerprints") or {})
        lineage = dict(state.get("knowledge_request_lineage") or {})
        blocked = set(state.get("blocked_knowledge_request_keys") or [])
        unchanged_requests: List[KnowledgeRequest] = []
        for request in requests:
            key = knowledge_request_key(request)
            fingerprint = knowledge_request_recall_fingerprint(items, request)
            previous = fingerprints.get(key)
            if previous is not None and previous == fingerprint:
                attempts[key] = attempts.get(key, 0) + 1
                blocked.add(key)
                unchanged_requests.append(request)
            else:
                attempts[key] = 0
                fingerprints[key] = fingerprint
            lineage[key] = {
                "request": request.model_dump(by_alias=True),
                "attempts": attempts.get(key, 0),
                "fingerprint": fingerprint,
                "blocked": key in blocked,
            }
        state["knowledge_request_attempts"] = attempts
        state["knowledge_request_fingerprints"] = fingerprints
        state["knowledge_request_lineage"] = lineage
        state["blocked_knowledge_request_keys"] = sorted(blocked)
        if unchanged_requests:
            state["knowledge_request_gaps"] = append_knowledge_request_gaps(
                state.get("knowledge_request_gaps", []),
                unchanged_requests,
                "METRIC_EVIDENCE_UNCHANGED",
            )
            add_step(
                state,
                "KnowledgeAgent：%d 个补知识请求二次召回无新增证据，停止重试"
                % len(unchanged_requests),
            )

    def awaiting_query_graph_repair_input(
        self,
        state: AgentState,
    ) -> Optional[PlannerRepairInput]:
        """Return the suspended repair contract while knowledge is being acquired."""

        delta = state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()
        if isinstance(delta, dict):
            try:
                delta = QueryGraphRepairDelta.model_validate(delta)
            except Exception:
                return None
        plan = state.get("plan") or QueryPlan()
        if (
            str(getattr(delta, "status", "") or "") != "awaiting_knowledge"
            or bool(getattr(delta, "changed", False))
            or bool(getattr(delta, "exhausted", False))
            or not plan.intents
        ):
            return None
        raw_repair_input = state.get("planner_repair_input")
        if isinstance(raw_repair_input, PlannerRepairInput):
            repair_input = raw_repair_input
        elif isinstance(raw_repair_input, dict):
            try:
                repair_input = PlannerRepairInput.model_validate(raw_repair_input)
            except Exception:
                return None
        else:
            return None
        if repair_input.reflection.passed:
            return None
        return repair_input

    def restore_query_graph_repair_after_knowledge(self, state: AgentState) -> bool:
        """Resume the same Critic contract after supplemental knowledge retrieval."""

        repair_input = self.awaiting_query_graph_repair_input(state)
        if repair_input is None:
            return False
        reflection = repair_input.reflection.model_copy(deep=True)
        retained_gaps = dedupe_graph_validation_gaps(
            retained_query_graph_repair_gaps(repair_input)
        )
        state["planner_reflection"] = reflection
        state["planner_repair_reason"] = reflection.repair_reason
        state["planner_repair_requests"] = [
            item.model_copy(deep=True)
            for item in repair_input.repair_requests
        ]
        state["query_graph_reflected"] = True
        state["query_graph_repair_exhausted"] = False
        if retained_gaps:
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=True,
                    gaps=retained_gaps,
                ),
            )
            state["last_query_graph_validation_gaps"] = retained_gaps
        return True

    def route_recall_query(self, state: AgentState) -> str:
        slots = state.get("route_slots") or RouteSlots()
        parts = [state.get("question", "")]
        fast = state.get("fast_understanding") or FastUnderstandingResult()
        if fast.intent_kind:
            parts.append("intentKind:%s complexity:%s" % (fast.intent_kind, fast.complexity))
        if fast.metric_phrases:
            parts.append("metricPhrases:%s" % ",".join(fast.metric_phrases[:8]))
        if slots.object_refs:
            parts.append(" ".join("%s:%s" % (item.ref_type, item.value) for item in slots.object_refs))
        if slots.analysis_signals:
            parts.append("analysisSignals:%s" % ",".join(slots.analysis_signals))
        if slots.risk_level and slots.risk_level != "normal":
            parts.append("riskLevel:%s" % slots.risk_level)
        if slots.time_window.raw:
            parts.append("timeWindow:%s" % slots.time_window.raw)
        return " ".join(part for part in parts if part).strip()

    def build_intent_signals(self, state: AgentState) -> IntentSignals:
        topics = self._effective_topic_categories(state)
        data_topics = self._asset_backed_topic_categories(topics)
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        rule_items = [item for item in recall_items[:8] if rule_recall_item(item)]
        keywords = state.get("extracted_keywords") or ExtractedKeywords()
        route_slots = state.get("route_slots") or RouteSlots()
        has_data_intent = self.requires_bi_execution(state, keywords)
        has_analysis_intent = bool(state.get("open_diagnostic_intent"))
        rule_confidence = max([float(item.fusion_score or 0.0) for item in rule_items] or [0.0])
        data_confidence = max([float(item.fusion_score or 0.0) for item in recall_items if not rule_recall_item(item)] or [0.0])
        has_rule_signal = route_slots.risk_level in {"rule_sensitive", "high_risk"}
        rule_needed = bool(rule_items) and (has_rule_signal or not has_data_intent)
        rule_refs = [item.doc_id for item in rule_items if item.doc_id] if rule_needed else []
        suggested_actions: List[str] = []
        observations: List[str] = []
        if rule_refs:
            observations.append("retrieved_rule_evidence")
        elif rule_items:
            observations.append("rule_candidate_recalled_but_not_required")
        if has_data_intent:
            observations.append("data_intent_present")
        if route_slots.analysis_signals:
            observations.append("route_analysis_hint_present")
        if has_analysis_intent:
            observations.append("analysis_intent_signal_present")
        if route_slots.operation == "write_requested":
            observations.append("write_operation_blocked")
        if rule_refs and not has_data_intent:
            suggested_actions.append("answer_rule")
        if has_data_intent:
            suggested_actions.extend(["compact_assets", "plan_graph"])
        if not rule_refs and has_rule_signal:
            observations.append("rule_topic_without_rule_evidence")
        return IntentSignals(
            has_rule_evidence=bool(rule_refs),
            rule_evidence_refs=rule_refs,
            rule_evidence_count=len(rule_refs),
            has_data_intent=has_data_intent,
            data_topics=data_topics,
            has_analysis_intent=has_analysis_intent,
            open_diagnostic_intent=state.get("open_diagnostic_intent", ""),
            rule_confidence=rule_confidence,
            data_confidence=data_confidence,
            suggested_actions=dedupe_texts(suggested_actions),
            observations=dedupe_texts(observations),
        )

    def should_answer_with_rule_recall(self, state: AgentState) -> bool:
        keywords = state.get("extracted_keywords") or ExtractedKeywords()
        if self.requires_bi_execution(state, keywords):
            return False
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        route_slots = state.get("route_slots") or RouteSlots()
        has_rule_signal = route_slots.risk_level in {"rule_sensitive", "high_risk"}
        has_rule_recall = any(rule_recall_item(item) for item in recall_items[:6])
        return has_rule_signal or has_rule_recall

    def requires_bi_execution(self, state: AgentState, keywords: ExtractedKeywords | None = None) -> bool:
        keywords = keywords or state.get("extracted_keywords") or ExtractedKeywords()
        route_slots = state.get("route_slots") or RouteSlots()
        if route_slots.operation == "write_requested":
            return False
        if route_slots.object_refs:
            return True
        topics = self._effective_topic_categories(state)
        data_topics = self._asset_backed_topic_categories(topics)
        if not data_topics:
            return False
        if keywords.metric_keywords or keywords.dimension_keywords:
            return True
        if keywords.action_keywords or keywords.ranking_keywords or route_slots.analysis_signals:
            return True
        if getattr(keywords, "time_keywords", []):
            return True
        return False

    def has_rule_plan(self, state: AgentState) -> bool:
        plan = state.get("plan")
        if not plan or not plan.intents:
            return False
        return all(intent.intent_type == IntentType.VALID and intent.answer_mode == AnswerMode.RULE for intent in plan.intents)

    def rule_recall_ref_ids(self, state: AgentState, fallback: bool = True) -> List[str]:
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        ref_ids = [item.doc_id for item in recall_items[:8] if item.doc_id and rule_recall_item(item)]
        if not ref_ids and fallback:
            ref_ids = [item.doc_id for item in recall_items[:3] if item.doc_id]
        return ref_ids

    def rule_recall_context(self, state: AgentState, fallback: bool = True) -> str:
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        rule_items = [item for item in recall_items[:8] if rule_recall_item(item)]
        if not rule_items and fallback:
            rule_items = recall_items[:3]
        return "\n\n".join("召回规则片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in rule_items)

    def build_rule_recall_plan(self, state: AgentState) -> QueryPlan:
        rule_items = [
            item
            for item in (state.get("recall_bundle") or RecallBundle()).items
            if rule_recall_item(item)
        ]
        ref_ids = list(state.get("rule_recall_refs") or self.rule_recall_ref_ids(state))
        rule_topic = ""
        if rule_items:
            metadata = rule_items[0].metadata or {}
            rule_topic = str(
                rule_items[0].topic
                or metadata.get("categoryId")
                or metadata.get("questionCategory")
                or metadata.get("topic")
                or rule_items[0].source_type
                or ""
            )
        intent = QuestionIntent(
            question=state["question"],
            intent_type=IntentType.VALID,
            category=self.recall_service.topic_assets.resolve_topic_category(rule_topic),
            answer_mode=AnswerMode.RULE,
            plan_task_id="rule_recall_answer",
            knowledge_ref_ids=ref_ids,
            analysis_source="rule_recall",
            analysis_note="retrieved rule knowledge; skip BI QueryGraph and SQL",
        )
        return QueryPlan(
            intents=[intent],
            agent_trace=["planner=rule_recall_short_circuit", "rule.recall_refs=%d" % len(ref_ids)],
            final_required_evidence=["retrieved_rule_knowledge"],
        )

    def compact_assets(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "compact_assets", "KnowledgeAgent", "COMPACT_ASSETS", input_summary=state.get("recall_context", "")[:1000])
        increment_round(state)
        self.configure_artifact_roots(state)
        emit(state, "node.started", "COMPACT_ASSETS", {})
        resume_repair_after_compaction = self.awaiting_query_graph_repair_input(state) is not None
        self.invalidate_execution_outputs(state, "PlanningAssetPack 重新生成")
        planning_hints = planning_hints_from_extracted_keywords(
            state["question"],
            state.get("extracted_keywords") or ExtractedKeywords(),
        )
        pack = self.asset_builder.compact(
            state["question"],
            state["recall_bundle"],
            self._effective_topic_categories(state),
            self.open_diagnostic_debug(state),
            planning_hints,
        )
        slots = state.get("route_slots") or RouteSlots()
        pack.metric_compaction["routeSlots"] = {
            "objectRefs": [item.model_dump(by_alias=True) for item in slots.object_refs],
            "timeWindow": slots.time_window.model_dump(by_alias=True),
            "riskLevel": slots.risk_level,
            "analysisSignals": slots.analysis_signals,
        }
        pack.metric_compaction["fastUnderstanding"] = (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True)
        knowledge_bundle = state.get("knowledge_bundle") or KnowledgeBundle()
        pack.metric_compaction["recallLineage"] = list(state.get("recall_rounds") or [])
        pack.metric_compaction["requestLineage"] = state.get("knowledge_request_lineage") or {}
        retrieval_health = knowledge_retrieval_health_payload(state)
        pack.metric_compaction["retrievalHealth"] = retrieval_health
        planning_gaps = [
            *list(pack.metric_compaction.get("knowledgeRequestGaps") or []),
            *list(state.get("knowledge_request_gaps") or []),
        ]
        known_gap_ids = {
            (str(item.get("code") or ""), str(item.get("requestKey") or ""), str(item.get("backend") or ""), str(item.get("lane") or ""))
            for item in planning_gaps
            if isinstance(item, dict)
        }
        for gap in knowledge_retrieval_planning_gaps(state):
            identity = (
                str(gap.get("code") or ""),
                str(gap.get("requestKey") or ""),
                str(gap.get("backend") or ""),
                str(gap.get("lane") or ""),
            )
            if identity not in known_gap_ids:
                planning_gaps.append(gap)
                known_gap_ids.add(identity)
        pack.metric_compaction["knowledgeRequestGaps"] = planning_gaps
        pack.metric_compaction["loadedSourceRefs"] = sorted(pack.source_refs.keys())
        pack.metric_compaction["recallBackend"] = knowledge_bundle.backend or "hybrid"
        pack.metric_compaction["semanticSourceHash"] = (
            knowledge_bundle.semantic_source_hash
            or pack.metric_compaction.get("cache", {}).get("semanticSourceHash", "")
        )
        pack.metric_compaction["indexVersion"] = knowledge_bundle.index_version
        state["planning_asset_pack"] = pack
        # Hypothesis analysis is delegated only through Lead-selected generic
        # workers.  The former controller created fixed templates here and then
        # ran a hidden multi-stage workflow, which is intentionally retired.
        state["hypothesis_exploration"] = {}
        state["hypothesis_results"] = []
        state["hypothesis_exploration_completed"] = True
        state["hypothesis_exploration_status"] = {"status": "retired", "source": "lead_react"}
        state["planning_assets_compacted"] = True
        self.reconcile_fast_request_agent_gates(state)
        invalidate_graph_validation(state)
        if resume_repair_after_compaction:
            self.restore_query_graph_repair_after_knowledge(state)
        else:
            state["query_graph_reflected"] = False
            state["planner_reflection"] = PlannerReflectionResult()
        self.planner.artifact_store.write_json("planner", "planning_asset_pack.json", pack.model_dump(by_alias=True), preview_chars=0)
        add_step(
            state,
            "Main Agent Tool compact_assets：生成 PlanningAssetPack，tables=%d, metrics=%d, fields=%d, relationships=%d"
            % (len(pack.tables), len(pack.metrics), len(pack.fields), len(pack.relationships)),
        )
        self.refresh_context_snapshot(state, "compact_assets")
        self.record_span(
            state,
            "semantic_tool",
            "compact_assets",
            started,
            row_count=len(pack.tables),
            metadata={"metrics": len(pack.metrics), "fields": len(pack.fields), "relationships": len(pack.relationships)},
        )
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary="tables=%d metrics=%d relationships=%d" % (len(pack.tables), len(pack.metrics), len(pack.relationships)),
            artifact_paths=[str(Path(state["thread_data"].outputs_path) / "artifacts" / "planner" / "planning_asset_pack.json")],
        )
        emit(state, "node.completed", "COMPACT_ASSETS", {"tables": pack.known_tables()[:12]})
        return state

    def commit_grounded_query_contract(
        self,
        state: AgentState,
        binding_hints: Dict[str, Any],
    ) -> AgentState:
        """Seal Core-selected semantic refs into the only planning authority.

        The Core must have read every selected ref through the Topic-scoped
        filesystem.  Recall candidates and L0 summaries remain navigation
        evidence and cannot become executable bindings by appearing in the
        request payload alone.
        """

        started = now_ms()
        step = self.start_run_step(
            state,
            "commit_grounded_query_contract",
            "CoreAgent",
            "COMMIT_GROUNDED_QUERY_CONTRACT",
            input_summary=str(binding_hints)[:1000],
        )
        increment_round(state)
        self.configure_artifact_roots(state)
        previous_contract_value = state.get("grounded_query_contract")
        try:
            previous_contract = (
                previous_contract_value
                if isinstance(previous_contract_value, GroundedQueryContract)
                else GroundedQueryContract.model_validate(previous_contract_value)
                if previous_contract_value
                else None
            )
        except Exception:
            previous_contract = None
        previous_ready = bool(previous_contract and previous_contract.ready)

        topics = list(
            dict.fromkeys(
                str(item).strip()
                for item in [
                    *list((state.get("topic_workspace") or {}).get("topics") or []),
                    *list(state.get("semantic_workspace_opened_topics") or []),
                ]
                if str(item).strip()
            )
        )
        contract = GroundedQueryContractBuilder().build(
            question=str(state.get("question") or ""),
            topics=topics,
            core_semantic_evidence=list(state.get("core_semantic_evidence") or []),
            binding_hints=binding_hints,
            timezone_name=str(self.settings.business_timezone or "Asia/Shanghai"),
        )
        prior_rejections = merge_grounded_rejected_bindings(
            state.get("grounded_rejected_bindings") or [],
            [],
        )
        prior_fingerprints = {item.fingerprint for item in prior_rejections}
        reused_rejections = [
            item
            for item in contract.rejected_bindings
            if item.fingerprint in prior_fingerprints
        ]
        merged_rejections = merge_grounded_rejected_bindings(
            prior_rejections,
            contract.rejected_bindings,
        )
        if reused_rejections:
            reused_gaps = [
                GroundedContractGap(
                    code="REJECTED_BINDING_REUSED",
                    message=(
                        "Rejected table %s was selected again for the same capability; "
                        "return to the manifest and widen filesystem search"
                    )
                    % item.table,
                    topic=item.topic,
                    table=item.table,
                    resolution="EXPAND_SEARCH_SCOPE",
                    search_scope="READ_BINDINGS_THEN_TABLE_MANIFEST_THEN_TOPIC_INDEX",
                    required_capability=dict(item.required_capability),
                    rejected_ref_ids=list(item.ref_ids),
                )
                for item in reused_rejections
            ]
            contract = contract.model_copy(
                update={
                    "status": "REVISE_BINDINGS",
                    "unresolved_gaps": [*contract.unresolved_gaps, *reused_gaps],
                }
            )
        contract = contract.model_copy(update={"rejected_bindings": merged_rejections})
        state["grounded_rejected_bindings"] = [
            item.model_dump(by_alias=True)
            for item in merged_rejections
        ]
        validation = GroundedQueryContractValidator().validate(contract)
        candidate_ready = bool(contract.ready and validation.valid)
        state["grounded_query_contract_attempt"] = contract
        state["grounded_contract_attempt_validation"] = validation.model_dump(by_alias=True)

        previous_payload = (
            previous_contract.model_dump(by_alias=True)
            if previous_contract is not None
            else {}
        )
        candidate_payload = contract.model_dump(by_alias=True)
        same_as_previous = bool(
            previous_payload
            and hashlib.sha256(
                json.dumps(previous_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            == hashlib.sha256(
                json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        )
        # A malformed/partial proposal must never destroy the last READY
        # authority.  READY replacements and semantic REVISE_BINDINGS results
        # are authoritative; an UNRESOLVED proposal is merely an attempt until
        # it becomes complete.
        accepted = bool(
            not previous_ready
            or candidate_ready
            or contract.status == "REVISE_BINDINGS"
        )
        if accepted:
            if not same_as_previous:
                self.invalidate_execution_outputs(state, "GroundedQueryContract 原子替换")
                invalidate_graph_validation(state)
                state["plan"] = QueryPlan()
                state["grounded_query_compiled"] = False
                state["planning_asset_pack"] = PlanningAssetPack()
                state["planning_assets_compacted"] = False
            state["grounded_query_contract"] = contract
            state["grounded_contract_validation"] = validation.model_dump(by_alias=True)
            state["grounded_contract_ready"] = candidate_ready
            if candidate_ready:
                state["grounded_asset_pack"] = materialize_grounded_asset_pack(
                    contract,
                    self.asset_builder.topic_assets,
                )
            else:
                state["grounded_asset_pack"] = PlanningAssetPack()
        else:
            state["grounded_contract_ready"] = previous_ready
            add_step(
                state,
                "Core Grounding：候选 Contract 未完成，保留上一份 READY Contract 与 QueryGraph",
            )
        if any(gap.code == "TIME_RANGE_REQUIRED" for gap in contract.unresolved_gaps):
            self.request_human_clarification(
                state,
                "这类查询需要明确时间范围，请问要查询最近多久？",
                "QUERY_PLAN",
                "time_window",
                list(load_language_policy().routing.time_clarification_options),
            )
            add_step(state, "Core Grounding：用户未提供时间范围，转入 ask_human，未编译或执行 SQL")
        self.planner.artifact_store.write_json(
            "planner",
            "grounded_query_contract_attempt.json",
            candidate_payload,
            preview_chars=0,
        )
        if accepted:
            self.planner.artifact_store.write_json(
                "planner",
                "grounded_query_contract.json",
                candidate_payload,
                preview_chars=0,
            )
        add_step(
            state,
            "Core Grounding：Contract candidate status=%s accepted=%s tables=%d metrics=%d dimensions=%d gaps=%d"
            % (
                contract.status,
                accepted,
                len(contract.tables),
                len(contract.metrics),
                len(contract.dimensions),
                len(contract.unresolved_gaps),
            ),
        )
        self.record_span(
            state,
            "semantic_tool",
            "commit_grounded_query_contract",
            started,
            status="success" if candidate_ready else "gap",
            error_code=(
                ""
                if candidate_ready
                else ",".join(gap.code for gap in contract.unresolved_gaps[:4])
            ),
            metadata={
                "status": contract.status,
                "accepted": accepted,
                "evidenceRefs": list(contract.evidence_refs),
                "gaps": [gap.model_dump(by_alias=True) for gap in contract.unresolved_gaps[:12]],
            },
        )
        self.finish_run_step(
            state,
            step,
            "success" if candidate_ready else "gap",
            output_summary="status=%s evidence=%d gaps=%d"
            % (contract.status, len(contract.evidence_refs), len(contract.unresolved_gaps)),
        )
        emit(
            state,
            "node.completed",
            "COMMIT_GROUNDED_QUERY_CONTRACT",
            {
                "status": contract.status,
                "ready": candidate_ready,
                "accepted": accepted,
                "activeReady": state["grounded_contract_ready"],
                "gaps": len(contract.unresolved_gaps),
            },
        )
        return state

    def compile_grounded_query(self, state: AgentState) -> AgentState:
        """Compile QueryGraph deterministically from GroundedQueryContract."""

        started = now_ms()
        step = self.start_run_step(
            state,
            "compile_grounded_query",
            "QueryGraphCompiler",
            "COMPILE_GROUNDED_QUERY",
            input_summary=str(state.get("grounded_compile_reason") or "")[:500],
        )
        increment_round(state)
        self.configure_artifact_roots(state)
        self.invalidate_execution_outputs(state, "GroundedQueryContract 重新编译")
        contract_value = state.get("grounded_query_contract")
        try:
            contract = (
                contract_value
                if isinstance(contract_value, GroundedQueryContract)
                else GroundedQueryContract.model_validate(contract_value or {})
            )
        except Exception as exc:
            contract = None
            validation = GraphValidationResult(
                valid=False,
                repairable=False,
                gaps=[
                    GraphValidationGap(
                        code="GROUNDED_CONTRACT_INVALID",
                        reason="GroundedQueryContract could not be parsed: %s" % str(exc)[:240],
                    )
                ],
            )
            record_graph_validation(state, validation)
        else:
            validation = None

        if contract is None or not contract.ready:
            if validation is None:
                validation = GraphValidationResult(
                    valid=False,
                    repairable=False,
                    gaps=[
                        GraphValidationGap(
                            code="GROUNDED_CONTRACT_UNRESOLVED",
                            reason="GroundedQueryContract has blocking semantic gaps",
                        )
                    ],
                )
                record_graph_validation(state, validation)
            state["grounded_query_compiled"] = False
            state["grounded_compile_trace"] = {
                "status": "rejected",
                "plannerLlmCalls": 0,
                "reason": "contract_not_ready",
            }
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="contract not ready",
                error_code="GROUNDED_CONTRACT_UNRESOLVED",
            )
            emit(state, "node.completed", "COMPILE_GROUNDED_QUERY", state["grounded_compile_trace"])
            return state

        pack = state.get("grounded_asset_pack") or PlanningAssetPack()
        if not isinstance(pack, PlanningAssetPack):
            pack = PlanningAssetPack.model_validate(pack)
        if pack.is_empty():
            pack = materialize_grounded_asset_pack(contract, self.asset_builder.topic_assets)
        try:
            preparation = compile_grounded_query_contract(contract, pack)
            plan = preparation.plan
            validation = preparation.validation
        except Exception as exc:
            plan = QueryPlan()
            validation = GraphValidationResult(
                valid=False,
                repairable=False,
                gaps=[
                    GraphValidationGap(
                        code="GROUNDED_QUERY_COMPILE_FAILED",
                        reason="Deterministic QueryGraph compiler failed: %s" % str(exc)[:240],
                    )
                ],
            )

        state["grounded_asset_pack"] = pack
        # Compatibility view for the mature validator/NodeWorker only.  This
        # pack is a projection of bound refs and is never a planning input.
        state["planning_asset_pack"] = pack
        state["planning_assets_compacted"] = False
        state["plan"] = plan
        state["query_graph_reflected"] = True
        state["planner_reflection"] = PlannerReflectionResult()
        state["planner_provider_error"] = ""
        state["grounded_query_compiled"] = bool(plan.intents)
        record_graph_validation(state, validation, plan)
        state["last_query_graph_validation_gaps"] = [] if validation.valid else list(validation.gaps)
        state["grounded_compile_trace"] = {
            "status": "validated" if validation.valid else ("compiled_with_gaps" if plan.intents else "failed"),
            "plannerLlmCalls": 0,
            "source": "grounded_query_contract",
            "logicalIntentCount": len(plan.intents),
            "validationGaps": [gap.model_dump(by_alias=True) for gap in validation.gaps[:12]],
        }
        self.planner.artifact_store.write_json(
            "planner",
            "query_graph.json",
            plan.model_dump(by_alias=True),
            preview_chars=0,
        )
        self.planner.artifact_store.write_json(
            "planner",
            "grounded_execution_asset_pack.json",
            pack.model_dump(by_alias=True),
            preview_chars=0,
        )
        add_step(
            state,
            "QueryGraph Compiler：从 GroundedQueryContract 确定性编译，nodes=%d valid=%s PlannerLLM=0"
            % (len(plan.intents), validation.valid),
        )
        self.record_span(
            state,
            "planner",
            "compile_grounded_query",
            started,
            status="success" if validation.valid else "failed",
            error_code="" if validation.valid else ",".join(gap.code for gap in validation.gaps[:4]),
            metadata=state["grounded_compile_trace"],
        )
        self.finish_run_step(
            state,
            step,
            "success" if validation.valid else "gap",
            output_summary="nodes=%d valid=%s" % (len(plan.intents), validation.valid),
            error_code="" if validation.valid else ",".join(gap.code for gap in validation.gaps[:4]),
        )
        emit(state, "node.completed", "COMPILE_GROUNDED_QUERY", state["grounded_compile_trace"])
        return state

    def query_metric_ambiguity(self, pack: PlanningAssetPack, question: str) -> Dict[str, Any]:
        normalized_question = normalize_for_match(question)
        if not normalized_question:
            return {}
        buckets: Dict[str, List[Dict[str, Any]]] = {}

        def add_candidate(raw: Dict[str, Any], source: str, ambiguous: bool) -> None:
            label = str(
                raw.get("matchedMetricLabel")
                or raw.get("recallQuery")
                or raw.get("businessName")
                or raw.get("title")
                or raw.get("metricKey")
                or ""
            ).strip()
            label_norm = normalize_for_match(label)
            if not label_norm:
                return
            if label_norm not in normalized_question and not ambiguous:
                return
            table = str(raw.get("ownerTable") or raw.get("tableName") or raw.get("table") or "").strip()
            metric_key = str(raw.get("metricKey") or raw.get("metricRef") or "").strip()
            if not table or not metric_key:
                return
            display = str(raw.get("businessName") or raw.get("title") or raw.get("displayName") or metric_key).strip()
            bucket = buckets.setdefault(label_norm, [])
            if any(item["table"] == table and item["metricKey"] == metric_key for item in bucket):
                return
            bucket.append(
                {
                    "label": label,
                    "display": display,
                    "table": table,
                    "metricKey": metric_key,
                    "source": source,
                    "ambiguous": bool(ambiguous),
                }
            )

        for evidence in (pack.metric_compaction or {}).get("recalledMetricEvidence") or []:
            if not isinstance(evidence, dict):
                continue
            if not recalled_metric_evidence_matches_phrase(evidence, question) and not evidence.get("metricResolutionAmbiguous"):
                continue
            add_candidate(evidence, "recalled_metric_evidence", bool(evidence.get("metricResolutionAmbiguous")))

        for candidate in (pack.metric_compaction or {}).get("catalogMetricCandidates") or []:
            if not isinstance(candidate, dict):
                continue
            add_candidate(candidate, "catalog_metric_candidate", bool(candidate.get("ambiguous")))

        for metric in pack.metrics:
            matched_label = metric_direct_match_label(metric, question)
            if not matched_label:
                continue
            label_norm = normalize_for_match(matched_label)
            if not label_norm:
                continue
            metric_metadata = metric.metadata or {}
            sibling_metrics = []
            for sibling in pack.metrics:
                if sibling.table == metric.table and sibling.key == metric.key:
                    continue
                sibling_metadata = sibling.metadata or {}
                sibling_level = str(sibling_metadata.get("metricLevel") or sibling_metadata.get("metric_level") or "").lower()
                if sibling_level != "business_variant":
                    continue
                sibling_labels = [
                    sibling.key,
                    sibling.title,
                    str(sibling_metadata.get("businessName") or ""),
                    str(sibling_metadata.get("displayName") or ""),
                    *sibling.aliases,
                    *[str(alias) for alias in sibling_metadata.get("aliases") or []],
                ]
                if any(label_norm and label_norm in normalize_for_match(label) for label in sibling_labels if str(label or "").strip()):
                    sibling_metrics.append(sibling)
            if not sibling_metrics:
                continue
            add_candidate(
                {
                    "matchedMetricLabel": matched_label,
                    "businessName": str(metric_metadata.get("businessName") or metric.title or metric.key),
                    "ownerTable": metric.table,
                    "metricKey": metric.key,
                },
                "semantic_metric_variant_family",
                True,
            )
            for sibling in sibling_metrics[:5]:
                sibling_metadata = sibling.metadata or {}
                add_candidate(
                    {
                        "matchedMetricLabel": matched_label,
                        "businessName": str(sibling_metadata.get("businessName") or sibling.title or sibling.key),
                        "ownerTable": sibling.table,
                        "metricKey": sibling.key,
                    },
                    "semantic_metric_variant_family",
                    True,
                )

        for _label_norm, candidates in buckets.items():
            if len(candidates) < 2:
                continue
            if not any(item.get("ambiguous") for item in candidates):
                continue
            label = next((item["label"] for item in candidates if item.get("label")), "这个指标")
            options = [
                "%s（%s.%s）" % (item["display"], item["table"], item["metricKey"])
                for item in candidates[:6]
            ]
            return {
                "label": label,
                "options": dedupe_texts(options),
                "candidates": candidates,
            }
        return {}

    def request_query_metric_ambiguity_clarification(
        self,
        state: AgentState,
        ambiguity: Dict[str, Any],
        started: int,
        step: Any,
    ) -> AgentState:
        label = str(ambiguity.get("label") or "这个指标").strip()
        options = [str(item) for item in ambiguity.get("options") or [] if str(item).strip()]
        self.request_human_clarification(
            state,
            "你想看哪个 %s 口径？" % label,
            "METRIC_SCOPE",
            "metric_focus",
            options,
        )
        state["query_metric_trace"] = {
            **dict(state.get("query_metric_trace") or {}),
            "status": "needs_clarification",
            "ambiguity": ambiguity,
        }
        add_step(state, "Metric Tool query_metric：指标候选多义，调用 ask_human 确认口径")
        self.record_span(
            state,
            "semantic_tool",
            "query_metric",
            started,
            status="gap",
            error_code="METRIC_AMBIGUITY_REQUIRES_CLARIFICATION",
            metadata=state["query_metric_trace"],
        )
        self.finish_run_step(
            state,
            step,
            "gap",
            output_summary="metric ambiguity -> ask_human",
            error_code="METRIC_AMBIGUITY_REQUIRES_CLARIFICATION",
        )
        emit(state, "node.completed", "QUERY_METRIC", {"supported": False, "clarificationRequired": True})
        return state

    def query_metric(self, state: AgentState) -> AgentState:
        """Execute a governed single-metric tool after Topic recall and asset compaction."""
        started = now_ms()
        step = self.start_run_step(
            state,
            "query_metric",
            "MetricTool",
            "QUERY_METRIC",
            input_summary=state.get("question", ""),
        )
        increment_round(state)
        self.configure_artifact_roots(state)
        emit(state, "node.started", "QUERY_METRIC", {})
        state["query_metric_attempted"] = True
        state["query_metric_completed"] = False
        self.invalidate_execution_outputs(state, "query_metric 重新生成受控 QueryGraph")
        pack = state.get("planning_asset_pack") or PlanningAssetPack()
        time_window_contract: Dict[str, Any] = {}
        if self.settings.calendar_time_semantics_enabled:
            time_window_contract = dict(
                state.get("time_window_contract")
                or resolve_time_window_contract(state["question"], self.settings.business_timezone)
            )
            state["time_window_contract"] = time_window_contract
        planner_context = {
            "fastUnderstanding": (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
            "timeWindowContract": time_window_contract,
            "memoryConstraints": state.get("memory_constraints", []),
            "memoryRecall": memory_recall_health_payload(state),
        }
        recall_bundle = state.get("recall_bundle") or RecallBundle()
        candidate_hints = (
            self.planner._semantic_candidate_hints(
                state.get("question", ""),
                recall_bundle,
                pack,
                planner_context=planner_context,
            )
            if hasattr(self.planner, "_semantic_candidate_hints")
            else {}
        )
        plan = QueryPlan(agent_trace=["query_metric.semantic_candidates=advisory_only"])
        fast = state.get("fast_understanding") or FastUnderstandingResult()
        metric_phrase_count = len([phrase for phrase in fast.metric_phrases if str(phrase or "").strip()])
        if metric_phrase_count <= 1:
            ambiguity = self.query_metric_ambiguity(pack, state.get("question", ""))
            if ambiguity:
                return self.request_query_metric_ambiguity_clarification(state, ambiguity, started, step)
        plan = compile_semantic_metric_fallback_graph(state.get("question", ""), pack)
        if plan.intents:
            plan.agent_trace.extend(
                [
                    "query_metric.compiler=published_asset_contract",
                    "query_metric.semantic_candidates=advisory_only",
                ]
            )
        catalog_expansion_traces: List[str] = []
        if not plan.intents:
            catalog_expansion_traces = self.asset_builder.expand_for_metric_catalog_resolution(
                pack,
                state.get("question", ""),
            )
            if catalog_expansion_traces:
                state["planning_asset_pack"] = pack
                plan = compile_semantic_metric_fallback_graph(state.get("question", ""), pack)
                if plan.intents:
                    plan.agent_trace.extend(
                        [
                            "query_metric.compiler=published_asset_contract_after_catalog_expansion",
                            "query_metric.semantic_candidates=advisory_only",
                        ]
                    )
            if not plan.intents and metric_phrase_count <= 1:
                ambiguity = self.query_metric_ambiguity(pack, state.get("question", ""))
                if ambiguity:
                    return self.request_query_metric_ambiguity_clarification(state, ambiguity, started, step)
        state["query_metric_trace"] = {
            "status": "compiled" if plan.intents else "unsupported",
            "agentTrace": list(plan.agent_trace or []),
            "compilerTrace": list(plan.compiler_trace or []),
            "catalogExpansion": list(catalog_expansion_traces),
            "catalogMetricCandidates": list((pack.metric_compaction or {}).get("catalogMetricCandidates") or []),
            "semanticCandidateHints": {
                "authority": str(candidate_hints.get("authority") or "advisory_only"),
                "candidateRefs": list(candidate_hints.get("candidateRefs") or []),
                "provenance": str(candidate_hints.get("provenance") or ""),
            } if candidate_hints else {},
        }
        if not plan.intents:
            self.escalate_fast_request(state, "query_metric could not resolve one governed semantic metric from the current Topic workspace")
            add_step(state, "Metric Tool query_metric：当前 Topic 资产包未能唯一解析受控单指标，交回 Planner 多轮探索")
            self.record_span(
                state,
                "semantic_tool",
                "query_metric",
                started,
                status="gap",
                error_code="QUERY_METRIC_UNSUPPORTED",
                metadata=state["query_metric_trace"],
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="unsupported -> Planner",
                error_code="QUERY_METRIC_UNSUPPORTED",
            )
            emit(state, "node.completed", "QUERY_METRIC", {"supported": False, "fallback": "plan_query_graph"})
            return state
        if time_window_contract:
            plan = apply_time_window_contract_to_plan(plan, time_window_contract)
        state["plan"] = plan
        state["query_graph_reflected"] = True
        state["planner_reflection"] = PlannerReflectionResult()
        state["latency_optimization"] = self.latency_optimizer.update_after_plan(
            state.get("latency_optimization") or {},
            plan,
        )
        try:
            execution_preparation = prepare_execution_graph(
                state["question"],
                plan,
                pack,
                self.graph_validator,
                state.get("memory_constraints", []),
            )
            plan = execution_preparation.plan
            state["plan"] = plan
            validation = execution_preparation.validation
        except Exception as exc:
            execution_preparation = None
            validation = GraphValidationResult(
                valid=False,
                repairable=False,
                gaps=[
                    GraphValidationGap(
                        code="EXECUTION_GRAPH_PREPARATION_FAILED",
                        reason="QueryGraph could not be normalized and validated before SQL dispatch: %s"
                        % str(exc)[:240],
                    )
                ],
            )
        record_graph_validation(state, validation, plan)
        state["last_query_graph_validation_gaps"] = [] if validation.valid else list(validation.gaps)
        state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
            state,
            dedupe_workflow_knowledge_requests(
                list(state.get("pending_knowledge_requests") or []) + list(validation.recommended_knowledge_requests or [])
            ),
        )
        state["latency_optimization"] = self.latency_optimizer.update_after_validation(
            state.get("latency_optimization") or {},
            validation,
        )
        self.reconcile_fast_request_agent_gates(state)
        if not validation.valid:
            self.escalate_fast_request(state, "query_metric graph validation failed")
            add_step(
                state,
                "Metric Tool query_metric：已解析单指标但校验发现缺口，gaps=%d，交回 Planner/补知识"
                % len(validation.gaps),
            )
            self.record_span(
                state,
                "validator",
                "query_metric.validate",
                started,
                status="failed",
                error_code=",".join(gap.code for gap in validation.gaps[:4]),
                metadata={"gaps": [gap.model_dump(by_alias=True) for gap in validation.gaps[:12]]},
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="validation failed gaps=%d" % len(validation.gaps),
                error_code=",".join(gap.code for gap in validation.gaps[:4]),
            )
            emit(state, "node.completed", "QUERY_METRIC", {"supported": True, "valid": False, "gaps": len(validation.gaps)})
            return state
        state["query_metric_trace"] = {
            **dict(state.get("query_metric_trace") or {}),
            "status": "validated",
            "executionDeferredToLead": True,
            "validatedQueryGraphFingerprint": state.get("validated_query_graph_fingerprint", ""),
            "sourceQueryGraphFingerprint": (
                execution_preparation.source_plan_fingerprint
                if execution_preparation is not None
                else ""
            ),
        }
        state["query_metric_completed"] = False
        add_step(
            state,
            "Metric Tool query_metric：受控单指标 QueryGraph 已解析并校验，执行动作交回 LeadAgent 选择",
        )
        self.record_span(
            state,
            "semantic_tool",
            "query_metric",
            started,
            status="success",
            metadata={
                "validationGaps": len(validation.gaps),
                "trace": state.get("query_metric_trace") or {},
            },
        )
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary="validated metric graph; execution deferred to LeadAgent",
        )
        emit(
            state,
            "node.completed",
            "QUERY_METRIC",
            {
                "supported": True,
                "valid": True,
                "executionDeferredToLead": True,
            },
        )
        return state

    def plan_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "plan_query_graph", "PlannerAgent", "PLAN_QUERY_GRAPH", input_summary=state.get("question", ""))
        increment_round(state)
        state["query_graph_plan_attempts"] = int(state.get("query_graph_plan_attempts") or 0) + 1
        emit(state, "node.started", "PLAN_QUERY_GRAPH", {})
        self.configure_artifact_roots(state)
        self.invalidate_execution_outputs(state, "QueryGraph 重新规划")
        time_window_contract: Dict[str, Any] = {}
        if self.settings.calendar_time_semantics_enabled:
            time_window_contract = dict(
                state.get("time_window_contract")
                or resolve_time_window_contract(state["question"], self.settings.business_timezone)
            )
            state["time_window_contract"] = time_window_contract
        planner_package = self.prepare_scoped_context_package(
            state,
            "plan_query_graph",
            "PlannerAgent",
            allowed_tables=(state.get("planning_asset_pack") or PlanningAssetPack()).known_tables()[:12],
            allowed_metrics=self.planning_metric_keys_for_context(state.get("planning_asset_pack") or PlanningAssetPack())[:24],
        )
        planner_context = {
            "contextPackage": self.compact_context_package(planner_package),
            "openDiagnostic": self.open_diagnostic_debug(state),
            "previousUnderstanding": (state.get("plan") or QueryPlan()).question_understanding,
            "fastUnderstanding": (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
            "timeWindowContract": time_window_contract,
            "threadContext": state.get("thread_context", {}),
            "conversationContext": planner_conversation_context(state),
            "runtimeInjection": state.get("runtime_injection", {}),
            "memoryInjection": state.get("memory_injection", {}),
            "memoryConstraints": state.get("memory_constraints", []),
            "memoryRecall": memory_recall_health_payload(state),
        }
        planner_context = self.context_assembler.assemble_payload(
            state,
            "planner_context",
            "PlannerAgent",
            planner_context,
            budget_chars=int(self.settings.context_planner_budget_chars or 12000),
        )
        # Only the outer DeepAgent Core may assert this authority marker.  Its
        # Topic-scoped knowledge backend records successful read_file calls in
        # core_semantic_evidence before dispatching plan_graph.  Planner
        # consumes that immutable hand-off and must not start a hidden file
        # tool loop of its own.
        if bool(state.get("core_managed_filesystem")):
            planner_context["coreManagedFilesystem"] = True
            planner_context["coreSemanticEvidence"] = [
                dict(item)
                for item in state.get("core_semantic_evidence") or []
                if isinstance(item, dict) and str(item.get("refId") or "").startswith("semantic:")
            ]
        planner_remaining_seconds = remaining_run_budget_seconds(state, self.settings)
        planner_context["runtimeBudget"] = {
            "deadlineEpochMs": datetime.now(timezone.utc).timestamp() * 1000
            + int(planner_remaining_seconds * 1000),
            "remainingSecondsAtDispatch": planner_remaining_seconds,
        }
        # The DeepAgent path hands Planner only the files the Core actually
        # read. Legacy harness callers keep the empty context and their
        # explicitly configured compatibility behavior.
        planner_knowledge_context = (
            json.dumps(planner_context.get("coreSemanticEvidence") or [], ensure_ascii=False)
            if planner_context.get("coreManagedFilesystem")
            else ""
        )
        plan, requests, reason = self.planner.plan(
            state["question"],
            state.get("history_rows", []),
            planner_knowledge_context,
            state["recall_bundle"],
            state["planning_asset_pack"],
            state["query_graph_validation_result"].gaps or state.get("last_query_graph_validation_gaps", []),
            state.get("thinking_steps", []),
            planner_context,
        )
        if time_window_contract:
            plan = apply_time_window_contract_to_plan(plan, time_window_contract)
        state["plan"] = plan
        self.materialize_plan_clarification(state)
        state["candidate_query_graphs"] = {}
        state["latency_optimization"] = self.latency_optimizer.update_after_plan(
            state.get("latency_optimization") or {},
            plan,
        )
        self.reconcile_fast_request_agent_gates(state)
        self.planner.artifact_store.write_json("planner", "query_graph.json", plan.model_dump(by_alias=True), preview_chars=0)
        self.planner.artifact_store.write_json("planner", "candidate_query_graphs.json", state["candidate_query_graphs"], preview_chars=0)
        plan_requests = list(getattr(plan, "knowledge_requests", []) or [])
        pending_requests = dedupe_workflow_knowledge_requests(plan_requests + list(requests or []))
        blocked_request_keys = set(state.get("blocked_knowledge_request_keys") or [])
        blocked_requests = [
            request for request in pending_requests if knowledge_request_key(request) in blocked_request_keys
        ]
        active_requests = [
            request for request in pending_requests if knowledge_request_key(request) not in blocked_request_keys
        ]
        if blocked_requests:
            blocked_gap_codes = knowledge_request_gap_codes_by_key(state.get("knowledge_request_gaps") or [])
            existing_code_requests: Dict[str, List[KnowledgeRequest]] = {}
            fallback_blocked_requests: List[KnowledgeRequest] = []
            for request in blocked_requests:
                code = blocked_gap_codes.get(knowledge_request_key(request))
                if code:
                    existing_code_requests.setdefault(code, []).append(request)
                else:
                    fallback_blocked_requests.append(request)
            gaps = state.get("knowledge_request_gaps", [])
            for code, code_requests in existing_code_requests.items():
                gaps = append_knowledge_request_gaps(gaps, code_requests, code)
            if fallback_blocked_requests:
                gaps = append_knowledge_request_gaps(gaps, fallback_blocked_requests, "METRIC_EVIDENCE_UNCHANGED")
            state["knowledge_request_gaps"] = gaps
            trace = list(plan.compiler_trace or [])
            for request in blocked_requests:
                marker = "%s:%s" % (blocked_gap_codes.get(knowledge_request_key(request)) or "METRIC_EVIDENCE_UNCHANGED", request.query)
                if marker not in trace:
                    trace.append(marker)
            plan.compiler_trace = trace
            plan.knowledge_requests = active_requests
        state["pending_knowledge_requests"] = active_requests
        state["planner_degraded"] = planner_degraded_state(
            self.planner.llm.last_error,
            plan,
            reason,
        )
        state["planner_provider_error"] = (
            str((state.get("planner_degraded") or {}).get("reason") or "")
            if not plan.intents
            else ""
        )
        if planner_degraded_stops_expensive_work(state):
            state["hypothesis_exploration_status"] = {"status": "skipped", "source": "planner_degraded"}
            state["analysis_skill_status"] = {"status": "skipped", "source": "planner_degraded"}
            state["hypothesis_evidence_ledger"] = HypothesisEvidenceLedger(
                ledger_id="ledger_%s" % state.get("run_id", "run"),
                budget={
                    "skipped": True,
                    "reason": "PLANNER_DEGRADED_FAIL_FAST",
                    "plannerDegraded": state["planner_degraded"],
                },
            )
        else:
            if (state.get("hypothesis_exploration_status") or {}).get("source") == "planner_degraded":
                state["hypothesis_exploration_status"] = {"status": "pending", "source": "runtime"}
                state["hypothesis_exploration_completed"] = False
            if (state.get("analysis_skill_status") or {}).get("source") == "planner_degraded":
                state["analysis_skill_status"] = {"status": "pending", "source": "runtime"}
                state["analysis_skill_bypassed"] = False
        invalidate_graph_validation(state)
        state["query_graph_reflected"] = False
        state["planner_reflection"] = PlannerReflectionResult()
        if active_requests:
            add_step(
                state,
                "Main Agent Tool plan_query_graph：planner 请求补知识，requests=%d，reason=%s"
                % (len(active_requests), reason),
            )
        elif blocked_requests:
            add_step(
                state,
                "Main Agent Tool plan_query_graph：补知识请求无新增 evidence，转结构化 gap，blocked=%d"
                % len(blocked_requests),
            )
        else:
            add_step(state, "Main Agent Tool plan_query_graph：生成 QueryGraph，nodes=%d, edges=%d" % (len(plan.intents), len(plan.dependencies)))
        self.refresh_context_snapshot(state, "plan_query_graph")
        status = "success" if plan.intents else "gap"
        error_code = (
            state.get("planner_provider_error", "")
            or ("NEED_MORE_KNOWLEDGE" if active_requests else "")
            or ("METRIC_EVIDENCE_UNCHANGED" if blocked_requests else "")
        )
        planner_prompt_stats = plan.planner_prompt_stats or {}
        estimated_prompt_chars = int(
            planner_prompt_stats.get("totalChars")
            or len(state.get("recall_context", ""))
            + len(json.dumps(self.planning_asset_debug(state["planning_asset_pack"]), ensure_ascii=False))
        )
        planner_used_semantic_fast_path = reason == "SEMANTIC_FAST_PATH" or any(
            "planner.semantic_fast_path" in str(item) for item in (plan.agent_trace or [])
        )
        self.record_span(
            state,
            "planner" if planner_used_semantic_fast_path else "llm",
            "planner.semantic_fast_path" if planner_used_semantic_fast_path else "planner.question_understanding",
            started,
            status="success" if plan.intents else "failed",
            model=self.settings.openai_model,
            provider=self.settings.openai_base_url,
            estimated_prompt_chars=0 if planner_used_semantic_fast_path else estimated_prompt_chars,
            estimated_completion_chars=len(json.dumps(plan.question_understanding or {}, ensure_ascii=False)),
            error_code=error_code,
            error_message=self.planner.llm.last_error if not plan.intents else "",
            metadata={
                "plannerPromptStats": planner_prompt_stats,
                "plannerDegraded": state.get("planner_degraded") or {},
            },
        )
        self.finish_run_step(
            state,
            step,
            status,
            output_summary="nodes=%d edges=%d requests=%d" % (len(plan.intents), len(plan.dependencies), len(active_requests)),
            error_code=error_code,
            error_message=self.planner.llm.last_error if not plan.intents else "",
            artifact_paths=[str(Path(state["thread_data"].outputs_path) / "artifacts" / "planner" / "query_graph.json")],
        )
        emit(
            state,
            "node.completed",
            "PLAN_QUERY_GRAPH",
            {
                "nodes": len(plan.intents),
                "requests": len(active_requests),
                "degraded": state.get("planner_degraded") or {},
            },
        )
        return state

    def configure_artifact_roots(self, state: AgentState) -> None:
        thread_data = state.get("thread_data")
        if not thread_data:
            return
        artifact_root = Path(thread_data.outputs_path) / "artifacts"
        self.planner.with_artifact_root(str(artifact_root))
        if hasattr(self.node_worker, "with_artifact_root"):
            self.node_worker.with_artifact_root(str(artifact_root))

    def start_run_step(self, state: AgentState, action_id: str, agent: str, node: str, reason: str = "", input_summary: str = ""):
        step = start_step(state, action_id, agent, node, reason, input_summary)
        emit(state, "run.step.started", node, step.model_dump(by_alias=True))
        return step

    def finish_run_step(
        self,
        state: AgentState,
        step,
        status: str = "success",
        output_summary: str = "",
        error_code: str = "",
        error_message: str = "",
        artifact_paths: Optional[List[str]] = None,
    ) -> None:
        refs = [artifact_ref_from_path(path, reason="step output artifact") for path in artifact_paths or []]
        finish_step(state, step, status, output_summary, error_code, error_message, refs)
        emit(state, "run.step.completed", step.node, step.model_dump(by_alias=True))

    def record_span(
        self,
        state: AgentState,
        kind: str,
        name: str,
        started_ms: float,
        status: str = "success",
        **kwargs,
    ) -> None:
        span = append_span(state, kind, name, started_ms, status=status, **kwargs)
        emit(state, "run.span.recorded", name, span.model_dump(by_alias=True))

    @contextmanager
    def run_node_step(
        self,
        state: AgentState,
        action_id: str,
        agent: str,
        node: str,
        reason: str = "",
        input_summary: str = "",
    ):
        started = now_ms()
        step = self.start_run_step(state, action_id, agent, node, reason=reason, input_summary=input_summary)
        try:
            yield step
        except Exception as exc:
            self.record_span(state, "action", action_id, started, status="error", error_code=type(exc).__name__, error_message=str(exc))
            if getattr(step, "status", "") == "running":
                self.finish_run_step(state, step, "error", error_code=type(exc).__name__, error_message=str(exc))
            raise
        else:
            if getattr(step, "status", "") == "running":
                self.record_span(state, "action", action_id, started)
                self.finish_run_step(state, step, "success", output_summary=getattr(step, "output_summary", ""))

    def reflect_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "reflect_plan", "PlannerCriticAgent", "REFLECT_QUERY_GRAPH", input_summary="nodes=%d" % len(state["plan"].intents))
        increment_round(state)
        emit(state, "node.started", "REFLECT_QUERY_GRAPH", {})
        try:
            reflection = self.planner_reflection_agent.reflect(
                state["question"],
                state["plan"],
                state["planning_asset_pack"],
            )
        except Exception as exc:
            failure_reason = "PlannerCritic callback failed: %s: %s" % (
                type(exc).__name__,
                str(exc)[:300],
            )
            failure_gap = GraphValidationGap(
                code="PLANNER_CRITIC_FAILED",
                evidence=type(exc).__name__,
                reason=failure_reason,
            )
            reflection = PlannerReflectionResult(
                passed=False,
                issues=[
                    {
                        "code": failure_gap.code,
                        "severity": "error",
                        "evidence": failure_gap.evidence,
                        "reason": failure_gap.reason,
                    }
                ],
                suggested_actions=["repair_graph"],
                repair_reason=failure_gap.code,
            )
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=True,
                    gaps=[failure_gap],
                ),
            )
            state["last_action_result"] = ActionResult(
                action="reflect_plan",
                node="reflect_query_graph",
                status="failed",
                message=failure_reason,
                retryable=True,
            )
        state["planner_reflection"] = reflection
        state["planner_repair_reason"] = reflection.repair_reason
        state["planner_repair_requests"] = reflection.repair_requests
        state["query_graph_reflected"] = True
        repair_input = planner_repair_input_from_state(state)
        scope_attempts = dict(state.get("query_graph_repair_scope_attempts") or {})
        scope_attempt_count = int(scope_attempts.get(repair_input.scope_key, 0) or 0)
        state["planner_repair_input"] = repair_input
        state["query_graph_repair_scope_key"] = repair_input.scope_key
        state["query_graph_repair_scope_attempt_count"] = scope_attempt_count
        state["query_graph_repair_exhausted"] = bool(
            not reflection.passed
            and scope_attempt_count >= max(0, int(self.policy.max_graph_repair_actions))
        )
        if state["query_graph_repair_exhausted"]:
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=False,
                    gaps=dedupe_graph_validation_gaps(
                        [
                            query_graph_repair_exhaustion_gap(repair_input, scope_attempt_count),
                            *retained_query_graph_repair_gaps(repair_input),
                        ]
                    ),
                ),
            )
        reflection_requests = list(reflection.suggested_knowledge_requests or [])
        for repair_request in reflection.repair_requests or []:
            reflection_requests.extend(repair_request.knowledge_requests or [])
        if reflection_requests:
            state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
                state,
                dedupe_workflow_knowledge_requests(list(state.get("pending_knowledge_requests") or []) + reflection_requests),
            )
        if reflection.passed:
            add_step(state, "Planner Critic Tool reflect_plan：QueryGraph 自检通过，issues=%d" % len(reflection.issues))
        else:
            add_step(
                state,
                "Planner Critic Tool reflect_plan：发现 %d 个计划问题，suggested=%s"
                % (len(reflection.issues), reflection.suggested_actions[:3]),
            )
        self.record_span(
            state,
            "critic",
            "reflect_plan",
            started,
            status="success" if reflection.passed else "failed",
            error_code=reflection.repair_reason,
            metadata={"issues": reflection.issues[:12], "repairRequests": [item.model_dump(by_alias=True) for item in reflection.repair_requests[:8]]},
        )
        self.finish_run_step(
            state,
            step,
            "success" if reflection.passed else "gap",
            output_summary="passed=%s issues=%d" % (reflection.passed, len(reflection.issues)),
            error_code=reflection.repair_reason,
        )
        emit(
            state,
            "node.completed",
            "REFLECT_QUERY_GRAPH",
            {"passed": reflection.passed, "issues": len(reflection.issues), "suggestedActions": reflection.suggested_actions},
        )
        return state

    def validate_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "validate_graph", "PlannerCriticAgent", "VALIDATE_QUERY_GRAPH", input_summary="nodes=%d" % len(state["plan"].intents))
        increment_round(state)
        emit(state, "node.started", "VALIDATE_QUERY_GRAPH", {})
        result = self.graph_validator.validate(
            state["question"],
            state["plan"],
            state["planning_asset_pack"],
            state.get("memory_constraints", []),
        )
        result = validation_with_question_coverage(
            state["question"],
            state["plan"],
            state["planning_asset_pack"],
            result,
        )
        record_graph_validation(state, result)
        state["latency_optimization"] = self.latency_optimizer.update_after_validation(
            state.get("latency_optimization") or {},
            result,
        )
        self.reconcile_fast_request_agent_gates(state)
        state["last_query_graph_validation_gaps"] = [] if result.valid else list(result.gaps)
        state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
            state,
            dedupe_workflow_knowledge_requests(list(state.get("pending_knowledge_requests") or []) + list(result.recommended_knowledge_requests or [])),
        )
        if result.valid:
            repair_delta = state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()
            reflection = state.get("planner_reflection") or PlannerReflectionResult()
            repair_revalidated = bool(
                repair_delta.status == "pending_revalidation"
                and repair_delta.changed
                and state.get("query_graph_reflected")
                and reflection.passed
                and repair_delta.after_graph_fingerprint == query_graph_fingerprint(state.get("plan"))
            )
            if repair_revalidated:
                repair_delta = repair_delta.model_copy(update={"status": "success"})
                state["last_query_graph_repair_delta"] = repair_delta
                history = list(state.get("query_graph_repair_history") or [])
                for index in range(len(history) - 1, -1, -1):
                    item = history[index]
                    if item.attempt == repair_delta.attempt and item.scope_key == repair_delta.scope_key:
                        history[index] = repair_delta.model_copy(deep=True)
                        break
                state["query_graph_repair_history"] = history
            state["planner_reflection"] = PlannerReflectionResult()
            state["planner_repair_reason"] = ""
            state["planner_repair_requests"] = []
            state["should_persist"] = any(intent.answer_mode != AnswerMode.RULE for intent in state["plan"].intents)
            add_step(state, "Main Agent Tool validate_query_graph：QueryGraph 通过校验，edges=%d" % len(state["plan"].dependencies))
        else:
            add_step(state, "Main Agent Tool validate_query_graph：发现 %d 个图缺口，repairable=%s" % (len(result.gaps), result.repairable))
        self.record_span(
            state,
            "validator",
            "validate_query_graph",
            started,
            status="success" if result.valid else "failed",
            error_code=",".join(gap.code for gap in result.gaps[:4]),
            metadata={"gaps": [gap.model_dump(by_alias=True) for gap in result.gaps[:12]]},
        )
        self.finish_run_step(
            state,
            step,
            "success" if result.valid else "gap",
            output_summary="valid=%s gaps=%d" % (result.valid, len(result.gaps)),
            error_code=",".join(gap.code for gap in result.gaps[:4]),
        )
        emit(state, "node.completed", "VALIDATE_QUERY_GRAPH", {"valid": result.valid, "gaps": len(result.gaps)})
        return state

    def repair_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "repair_graph", "PlannerAgent", "REPAIR_QUERY_GRAPH", reason=state.get("planner_repair_reason", ""))
        increment_round(state)
        repair_input = planner_repair_input_from_state(state)
        state["planner_repair_input"] = repair_input
        scope_attempts = dict(state.get("query_graph_repair_scope_attempts") or {})
        prior_scope_attempts = int(scope_attempts.get(repair_input.scope_key, 0) or 0)
        max_scope_attempts = max(0, int(self.policy.max_graph_repair_actions))
        state["query_graph_repair_scope_key"] = repair_input.scope_key
        state["query_graph_repair_scope_attempt_count"] = prior_scope_attempts
        state["query_graph_repair_exhausted"] = prior_scope_attempts >= max_scope_attempts
        state["query_graph_repair_attempted"] = False
        state["query_graph_repair_progressed"] = False
        emit(
            state,
            "node.started",
            "REPAIR_QUERY_GRAPH",
            {
                "scopeKey": repair_input.scope_key,
                "scopeAttempt": prior_scope_attempts + 1,
                "beforeGraphFingerprint": repair_input.graph_fingerprint,
                "reflection": repair_input.reflection.model_dump(by_alias=True),
                "repairRequests": [item.model_dump(by_alias=True) for item in repair_input.repair_requests],
            },
        )
        if prior_scope_attempts >= max_scope_attempts:
            gap = query_graph_repair_exhaustion_gap(repair_input, prior_scope_attempts)
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=False,
                    gaps=[
                        gap,
                        *retained_query_graph_repair_gaps(repair_input),
                    ],
                ),
            )
            add_step(state, "Main Agent Tool repair_query_graph：相同 QueryGraph 与问题集合的修复预算已耗尽")
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="scopeAttempt=%d exhausted=true" % prior_scope_attempts,
                error_code="QUERY_GRAPH_REPAIR_EXHAUSTED",
            )
            emit(
                state,
                "node.completed",
                "REPAIR_QUERY_GRAPH",
                {
                    "scopeKey": repair_input.scope_key,
                    "scopeAttempt": prior_scope_attempts,
                    "changed": False,
                    "exhausted": True,
                    "errorCode": "QUERY_GRAPH_REPAIR_EXHAUSTED",
                },
            )
            return state

        total_attempt = int(state.get("query_graph_repair_attempts") or 0) + 1
        scope_attempt = prior_scope_attempts + 1
        scope_attempts[repair_input.scope_key] = scope_attempt
        state["query_graph_repair_attempts"] = total_attempt
        state["query_graph_repair_attempted"] = True
        state["query_graph_repair_scope_attempts"] = scope_attempts
        state["query_graph_repair_scope_attempt_count"] = scope_attempt
        before_plan = (state.get("plan") or QueryPlan()).model_copy(deep=True)
        before_nodes = len(before_plan.intents)
        try:
            repaired_plan = self.planner.repair(
                state["question"],
                before_plan.model_copy(deep=True),
                state["planning_asset_pack"],
                [item.model_copy(deep=True) for item in repair_input.repair_gaps],
                state.get("history_rows", []),
                knowledge_context(state),
                state["recall_bundle"],
            )
        except Exception as exc:
            failure_gap = query_graph_repair_failure_gap(repair_input, exc)
            exhausted = scope_attempt >= max_scope_attempts
            failure_gaps = [failure_gap]
            if exhausted:
                failure_gaps.append(query_graph_repair_exhaustion_gap(repair_input, scope_attempt))
            failure_gaps.extend(retained_query_graph_repair_gaps(repair_input))
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=not exhausted,
                    gaps=dedupe_graph_validation_gaps(failure_gaps),
                ),
            )
            delta = QueryGraphRepairDelta(
                attempt=total_attempt,
                scope_attempt=scope_attempt,
                scope_key=repair_input.scope_key,
                status="failed",
                changed=False,
                exhausted=exhausted,
                before_graph_fingerprint=repair_input.graph_fingerprint,
                after_graph_fingerprint=repair_input.graph_fingerprint,
                before_nodes=before_nodes,
                after_nodes=before_nodes,
                repair_reason=failure_gap.reason,
                reflection=repair_input.reflection.model_copy(deep=True),
                repair_requests=[item.model_copy(deep=True) for item in repair_input.repair_requests],
                repair_gaps=[failure_gap, *[item.model_copy(deep=True) for item in repair_input.repair_gaps]],
            )
            state["plan"] = before_plan
            state["last_query_graph_repair_delta"] = delta
            state.setdefault("query_graph_repair_history", []).append(delta.model_copy(deep=True))
            state["query_graph_repair_history"] = state["query_graph_repair_history"][-32:]
            state["query_graph_repair_exhausted"] = exhausted
            state["query_graph_reflected"] = True
            state["last_action_result"] = ActionResult(
                action="repair_graph",
                node="repair_query_graph",
                status="failed",
                message=failure_gap.reason,
                retryable=not exhausted,
            )
            add_step(state, "Main Agent Tool repair_query_graph：修复回调失败，已保留原图与 PlannerCritic 契约")
            self.record_span(
                state,
                "planner_repair",
                "repair_query_graph",
                started,
                status="failed",
                error_code=failure_gap.code,
                error_message=failure_gap.reason,
                metadata={
                    "repairInput": repair_input.model_dump(by_alias=True),
                    "repairDelta": delta.model_dump(by_alias=True),
                },
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="failed scopeAttempt=%d" % scope_attempt,
                error_code=failure_gap.code,
                error_message=failure_gap.reason,
            )
            emit(
                state,
                "node.completed",
                "REPAIR_QUERY_GRAPH",
                {
                    "attempt": total_attempt,
                    "scopeAttempt": scope_attempt,
                    "scopeKey": repair_input.scope_key,
                    "changed": False,
                    "exhausted": exhausted,
                    "errorCode": failure_gap.code,
                    "repairDelta": delta.model_dump(by_alias=True),
                },
            )
            return state
        if isinstance(repaired_plan, dict):
            try:
                repaired_plan = QueryPlan.model_validate(repaired_plan)
            except Exception:
                repaired_plan = before_plan.model_copy(deep=True)
        if not isinstance(repaired_plan, QueryPlan):
            repaired_plan = before_plan.model_copy(deep=True)
        # The independent raw-question condition ledger is an immutable input
        # to Repair, not Planner-authored graph material.  A repair may change
        # semanticQuery bindings/structure, but may not delete or rewrite the
        # source obligations it is supposed to satisfy.
        before_understanding = before_plan.question_understanding or {}
        frozen_condition_ledger = (
            before_understanding.get("sourceConditionLedger")
            or before_understanding.get("source_condition_ledger")
        )
        if isinstance(frozen_condition_ledger, dict) and frozen_condition_ledger:
            repaired_understanding = dict(repaired_plan.question_understanding or {})
            repaired_understanding["sourceConditionLedger"] = deepcopy(frozen_condition_ledger)
            repaired_understanding["sourceConditionAuditRequired"] = True
            repaired_plan = repaired_plan.model_copy(
                update={"question_understanding": repaired_understanding}
            )
        added_knowledge_requests = new_query_graph_repair_knowledge_requests(before_plan, repaired_plan)
        handled_knowledge_request_keys = {
            str(key)
            for key in (state.get("knowledge_request_lineage") or {})
            if str(key)
        } | {
            str(key)
            for key in (state.get("blocked_knowledge_request_keys") or [])
            if str(key)
        }
        if handled_knowledge_request_keys:
            repaired_plan.knowledge_requests = [
                request
                for request in repaired_plan.knowledge_requests
                if knowledge_request_key(request) not in handled_knowledge_request_keys
            ]
            added_knowledge_requests = [
                request
                for request in added_knowledge_requests
                if knowledge_request_key(request) not in handled_knowledge_request_keys
            ]
        after_fingerprint = query_graph_fingerprint(repaired_plan)
        before_structure_fingerprint = query_graph_structure_fingerprint(before_plan)
        after_structure_fingerprint = query_graph_structure_fingerprint(repaired_plan)
        structure_changed = after_structure_fingerprint != before_structure_fingerprint

        # Repair only proposes a candidate.  It cannot act as its own Critic:
        # structural change is committed as pending revalidation and the next
        # workflow observation must run PlannerCritic/Validator independently.
        changed = structure_changed
        awaiting_knowledge = bool(not changed and added_knowledge_requests)
        if awaiting_knowledge:
            # Asking for governed knowledge suspends this repair. It is an
            # orchestration transition, not a failed executable-graph attempt,
            # so the same Critic contract keeps its remaining repair budget.
            if prior_scope_attempts:
                scope_attempts[repair_input.scope_key] = prior_scope_attempts
            else:
                scope_attempts.pop(repair_input.scope_key, None)
            state["query_graph_repair_scope_attempts"] = scope_attempts
            state["query_graph_repair_scope_attempt_count"] = prior_scope_attempts
        exhausted = bool(not changed and not awaiting_knowledge and scope_attempt >= max_scope_attempts)
        repair_status = (
            "pending_revalidation"
            if changed
            else ("awaiting_knowledge" if awaiting_knowledge else "no_progress")
        )
        delta = QueryGraphRepairDelta(
            attempt=total_attempt,
            scope_attempt=scope_attempt,
            scope_key=repair_input.scope_key,
            status=repair_status,
            changed=changed,
            exhausted=exhausted,
            before_graph_fingerprint=repair_input.graph_fingerprint,
            after_graph_fingerprint=after_fingerprint,
            before_nodes=before_nodes,
            after_nodes=len(repaired_plan.intents),
            repair_reason=str(state.get("planner_repair_reason") or repair_input.reflection.repair_reason or ""),
            reflection=repair_input.reflection.model_copy(deep=True),
            repair_requests=[item.model_copy(deep=True) for item in repair_input.repair_requests],
            repair_gaps=[item.model_copy(deep=True) for item in repair_input.repair_gaps],
        )
        state["last_query_graph_repair_delta"] = delta
        state.setdefault("query_graph_repair_history", []).append(delta.model_copy(deep=True))
        state["query_graph_repair_history"] = state["query_graph_repair_history"][-32:]
        state["query_graph_repair_exhausted"] = exhausted

        if changed:
            state["plan"] = repaired_plan
            state["query_graph_repair_progressed"] = True
            self.materialize_plan_clarification(state)
            state["last_action_result"] = ActionResult(
                action="repair_graph",
                node="repair_query_graph",
                status="pending_revalidation",
                message="repair produced a structural candidate; PlannerCritic and Validator must revalidate it",
                retryable=True,
            )
        elif awaiting_knowledge:
            state["plan"] = repaired_plan
            state["last_action_result"] = ActionResult(
                action="repair_graph",
                node="repair_query_graph",
                status="awaiting_knowledge",
                message="repair preserved the executable graph and requested supplemental governed knowledge",
                retryable=True,
            )
        else:
            # The repairer may append traces in place.  A trace-only mutation is
            # not an executable graph repair, so retain the exact input graph.
            state["plan"] = before_plan
            state["last_action_result"] = ActionResult(
                action="repair_graph",
                node="repair_query_graph",
                status="no_progress",
                message="repair did not change the executable/evidence-bearing graph contract",
                retryable=not exhausted,
            )

        if (changed or awaiting_knowledge) and state["plan"].knowledge_requests:
            state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
                state,
                dedupe_workflow_knowledge_requests(
                    list(state.get("pending_knowledge_requests") or []) + list(state["plan"].knowledge_requests or [])
                ),
            )
        if changed:
            state["query_graph_reflected"] = False
            pending_gaps = dedupe_graph_validation_gaps(
                [
                    GraphValidationGap(
                        code="QUERY_GRAPH_REPAIR_PENDING_REVALIDATION",
                        evidence=after_structure_fingerprint,
                        reason="repair changed the executable graph; original typed gaps remain blocking until independent PlannerCritic/Validator revalidation passes",
                    ),
                    *retained_query_graph_repair_gaps(repair_input),
                ]
            )
            record_graph_validation(
                state,
                GraphValidationResult(valid=False, repairable=True, gaps=pending_gaps),
            )
            mark_graph_validation_stale(state)
            state["last_query_graph_validation_gaps"] = pending_gaps
            self.invalidate_execution_outputs(state, "QueryGraph 修复后需要重新执行")
            add_step(state, "Main Agent Tool repair_query_graph：候选图已改变，保留原 typed gaps，等待 PlannerCritic/Validator 重验")
        elif awaiting_knowledge:
            state["query_graph_reflected"] = True
            awaiting_gap = query_graph_repair_awaiting_knowledge_gap(
                repair_input,
                added_knowledge_requests,
            )
            projected_gaps = dedupe_graph_validation_gaps(
                [awaiting_gap, *retained_query_graph_repair_gaps(repair_input)]
            )
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=True,
                    gaps=projected_gaps,
                ),
            )
            state["last_query_graph_validation_gaps"] = projected_gaps
            add_step(
                state,
                "Main Agent Tool repair_query_graph：仅新增补知识请求，保留原图和 PlannerCritic 契约，状态=awaiting_knowledge",
            )
        else:
            state["query_graph_reflected"] = True
            add_step(
                state,
                "Main Agent Tool repair_query_graph：候选修复未通过目标 gap 重验证，保留 PlannerCritic 问题与修复请求",
            )
            no_progress_gap = query_graph_repair_no_progress_gap(repair_input, scope_attempt)
            projected_gaps = [no_progress_gap]
            if exhausted:
                projected_gaps.insert(0, query_graph_repair_exhaustion_gap(repair_input, scope_attempt))
            projected_gaps.extend(retained_query_graph_repair_gaps(repair_input))
            projected_gaps = dedupe_graph_validation_gaps(projected_gaps)
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    repairable=not exhausted,
                    gaps=projected_gaps,
                ),
            )
            state["last_query_graph_validation_gaps"] = projected_gaps
        repair_artifact = Path(state["thread_data"].outputs_path) / "artifacts" / "planner" / ("repair_attempt_%d.json" % total_attempt)
        try:
            self.planner.artifact_store.write_json(
                "planner",
                "repair_attempt_%d.json" % total_attempt,
                {
                    "repairInput": repair_input.model_dump(by_alias=True),
                    "repairDelta": delta.model_dump(by_alias=True),
                    "plan": state["plan"].model_dump(by_alias=True),
                },
                preview_chars=0,
            )
        except Exception:
            pass
        repair_error_code = (
            "QUERY_GRAPH_REPAIR_PENDING_REVALIDATION"
            if changed
            else (
                "QUERY_GRAPH_REPAIR_AWAITING_KNOWLEDGE"
                if awaiting_knowledge
                else ("QUERY_GRAPH_REPAIR_EXHAUSTED" if exhausted else "QUERY_GRAPH_REPAIR_NO_PROGRESS")
            )
        )
        self.record_span(
            state,
            "planner_repair",
            "repair_query_graph",
            started,
            status="pending" if (changed or awaiting_knowledge) else "failed",
            error_code=repair_error_code,
            metadata={
                "repairInput": repair_input.model_dump(by_alias=True),
                "repairDelta": delta.model_dump(by_alias=True),
            },
        )
        self.finish_run_step(
            state,
            step,
            "gap",
            output_summary="status=%s changed=%s scopeAttempt=%d before=%s after=%s"
            % (
                repair_status,
                changed,
                scope_attempt,
                repair_input.graph_fingerprint,
                after_fingerprint,
            ),
            error_code=repair_error_code,
            artifact_paths=[str(repair_artifact)],
        )
        emit(
            state,
            "node.completed",
            "REPAIR_QUERY_GRAPH",
            {
                "attempt": total_attempt,
                "scopeAttempt": scope_attempt,
                "scopeKey": repair_input.scope_key,
                "status": repair_status,
                "changed": changed,
                "awaitingKnowledge": awaiting_knowledge,
                "exhausted": exhausted,
                "beforeGraphFingerprint": repair_input.graph_fingerprint,
                "afterGraphFingerprint": after_fingerprint,
                "reflection": repair_input.reflection.model_dump(by_alias=True),
                "repairRequests": [item.model_dump(by_alias=True) for item in repair_input.repair_requests],
                "repairDelta": delta.model_dump(by_alias=True),
            },
        )
        return state

    def execute_query_graph_direct(self, state: AgentState) -> AgentState:
        state["node_execution_mode"] = "direct"
        return self.execute_query_graph(state)

    def execute_query_graph_agent(self, state: AgentState) -> AgentState:
        state["node_execution_mode"] = "subagent"
        return self.execute_query_graph(state)

    def execute_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "execute_graph", "NodeAgent", "EXECUTE_QUERY_GRAPH", input_summary="nodes=%d" % len(state["plan"].intents))
        increment_round(state)
        self.configure_artifact_roots(state)
        state["worker_dispatch_context"] = self.worker_dispatch_context(state)
        emit(state, "node.started", "EXECUTE_QUERY_GRAPH", {"workerDispatch": state.get("worker_dispatch_context", {})})
        validation_failure = graph_validation_failure_reason(state)
        execution_preparation = None
        if not validation_failure:
            try:
                execution_preparation = self.node_worker.prepare_runtime_execution_graph(
                    state["merchant"].merchant_id,
                    state["plan"],
                    state.get("planning_asset_pack") or PlanningAssetPack(),
                    state["question"],
                    graph_validator=self.graph_validator,
                    memory_constraints=state.get("memory_constraints", []),
                    access_role=state.get("access_role", load_authorization_policy().default_access_role),
                    user_scope=state.get("user_identity", {}),
                )
                state["plan"] = execution_preparation.plan
                record_graph_validation(state, execution_preparation.validation, execution_preparation.plan)
            except Exception as exc:
                record_graph_validation(
                    state,
                    GraphValidationResult(
                        valid=False,
                        repairable=False,
                        gaps=[
                            GraphValidationGap(
                                code="EXECUTION_GRAPH_PREPARATION_FAILED",
                                reason="QueryGraph could not be normalized and validated before SQL dispatch: %s"
                                % str(exc)[:240],
                            )
                        ],
                    ),
                    state["plan"],
                )
            validation_failure = graph_validation_failure_reason(state)
        validation = state.get("query_graph_validation_result")
        if validation_failure:
            if validation_failure == "QUERY_GRAPH_CHANGED_AFTER_VALIDATION":
                mark_graph_validation_stale(state)
            validation_gaps = list(getattr(validation, "gaps", None) or [])
            if not validation_gaps:
                validation_gaps = [
                    GraphValidationGap(
                        code=validation_failure,
                        reason="The current QueryGraph does not have a matching passed validation contract",
                    )
                ]
            gaps = [
                EvidenceGap(
                    code=gap.code,
                    task_id=gap.task_id,
                    evidence=gap.evidence,
                    reason=gap.reason or "QueryGraph validation failed before SQL execution",
                    severity="blocking",
                    source="query_graph_validator",
                    answer_instruction="不要执行 SQL；先补知识或修复 QueryGraph 后再回答。",
                )
                for gap in validation_gaps
            ]
            run_result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    failed=True,
                    error=validation_failure,
                    summary="QueryGraph execution was rejected by the validation contract",
                ),
                evidence_gaps=gaps,
                partial_answer_reason=validation_failure,
                reflection_notes=["NodeAgent skipped because the current graph has no matching passed validation"],
            )
            append_active_planner_degraded_reason(state, run_result)
            archive_execution_attempt(state, run_result, "query_graph_validation_rejection")
            state["agent_run_result"] = run_result
            state["query_bundle"] = run_result.merged_query_bundle
            state["query_bundles"] = []
            state["node_tool_traces"] = []
            state["freshness_reports"] = []
            state["sql_generated"] = True
            state["result_generation"] = int(state.get("execution_generation") or 0)
            self.planner.artifact_store.write_json("node", "agent_run_result.json", run_result.model_dump(by_alias=True), preview_chars=0)
            add_step(state, "Main Agent Tool execute_query_graph：validation contract rejected execution; NodeWorker was not called")
            self.record_span(
                state,
                "action",
                "execute_query_graph",
                started,
                status="failed",
                error_code=validation_failure,
                metadata={"gaps": [gap.model_dump(by_alias=True) for gap in validation_gaps[:12]]},
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="skipped SQL because validation=%s gaps=%d" % (validation_failure, len(validation_gaps)),
                error_code=validation_failure,
                artifact_paths=[str(Path(state["thread_data"].outputs_path) / "artifacts" / "node" / "agent_run_result.json")],
            )
            emit(state, "node.completed", "EXECUTE_QUERY_GRAPH", {"tasks": 0, "rows": 0, "skipped": True})
            return state
        try:
            node_package = self.prepare_scoped_context_package(
                state,
                "execute_query_graph",
                "NodeWorker",
                allowed_tables=(state.get("planning_asset_pack") or PlanningAssetPack()).known_tables()[:12],
                allowed_metrics=[
                    item.key
                    for item in (state.get("planning_asset_pack") or PlanningAssetPack()).metrics[:24]
                    if item.key
                ],
            )
            node_knowledge_context = append_context_section(
                knowledge_context(state),
                self.render_context_package_for_prompt(node_package),
                max_chars=int(self.settings.context_runtime_budget_chars or 6000),
            )
            run_result = self.node_worker.execute_plan(
                state["merchant"].merchant_id,
                state["plan"],
                state["planning_asset_pack"],
                node_knowledge_context,
                state["question"],
                resume_task_results=(state.get("agent_run_result") or AgentRunResult()).task_results,
                run_id=state.get("run_id", ""),
                access_role=state.get("access_role", load_authorization_policy().default_access_role),
                user_scope=state.get("user_identity", {}),
                execution_mode=state.get("node_execution_mode", "auto"),
                execution_preparation=execution_preparation,
            )
            executed_fingerprint = str(run_result.executed_query_graph_fingerprint or "")
            validated_fingerprint = str(state.get("validated_query_graph_fingerprint") or "")
            if executed_fingerprint != validated_fingerprint:
                run_result = AgentRunResult(
                    executed_query_graph_fingerprint=executed_fingerprint,
                    merged_query_bundle=QueryBundle(
                        failed=True,
                        error="EXECUTED_QUERY_GRAPH_FINGERPRINT_MISMATCH",
                        summary="NodeWorker result was rejected because its graph fingerprint differs from the validated execution graph",
                    ),
                    evidence_gaps=[
                        EvidenceGap(
                            code="EXECUTED_QUERY_GRAPH_FINGERPRINT_MISMATCH",
                            reason="validated=%s executed=%s" % (validated_fingerprint, executed_fingerprint),
                            severity="blocking",
                            source="execution_graph_contract",
                            answer_instruction="禁止使用该 SQL 结果；重新准备并校验最终执行图。",
                        )
                    ],
                    reflection_notes=["NodeWorker execution graph fingerprint did not match workflow validation"],
                )
        except Exception as exc:
            run_result = AgentRunResult(
                merged_query_bundle=QueryBundle(failed=True, error=str(exc), summary="NodeWorker 执行失败"),
                reflection_notes=["NodeWorker 执行失败: %s" % str(exc)[:200]],
            )
        append_active_planner_degraded_reason(state, run_result)
        archive_execution_attempt(state, run_result, "query_graph_execution")
        state["agent_run_result"] = run_result
        state["query_bundle"] = run_result.merged_query_bundle
        state["query_bundles"] = run_result.query_bundles
        state["node_tool_traces"] = run_result.node_tool_traces
        state["freshness_reports"] = run_result.freshness_reports
        self.planner.artifact_store.write_json("node", "agent_run_result.json", run_result.model_dump(by_alias=True), preview_chars=0)
        self.sync_tool_runtime_state(state)
        state["sql_generated"] = True
        state["result_generation"] = int(state.get("execution_generation") or 0)
        for task_result in run_result.task_results:
            bundle = task_result.query_bundle
            sql_span_start = max(0, now_ms() - int(bundle.duration_ms or 0))
            self.record_span(
                state,
                "sql",
                "node_sql:%s" % task_result.task_id,
                sql_span_start,
                status="failed" if bundle.failed else "success",
                sql=bundle.sql,
                table=",".join(bundle.tables),
                row_count=bundle.effective_row_count(),
                error_code="SQL_EXECUTION_FAILED" if bundle.failed else "",
                error_message=bundle.error,
                retry_or_fallback_count=sum(1 for repair in run_result.sql_repairs if repair.task_id == task_result.task_id),
                metadata={"taskSummary": task_result.summary, "cacheHit": bundle.cache_hit, "cacheKey": bundle.cache_key},
            )
            emit(
                state,
                "task.completed" if task_result.success else "task.failed",
                "NODE_WORKER",
                {
                    "taskId": task_result.task_id,
                    "success": task_result.success,
                    "summary": task_result.summary,
                    "tables": task_result.query_bundle.tables,
                    "rows": task_result.query_bundle.effective_row_count(),
                },
            )
        add_step(
            state,
            "Main Agent Tool execute_query_graph：派发 NodeWorker Agent 执行 QueryGraph nodes，tasks=%d reason=%s"
            % (len(run_result.tasks), state["worker_dispatch_context"].get("reason") or "unknown"),
        )
        self.refresh_context_snapshot(state, "execute_query_graph")
        failed_tasks = sum(1 for item in run_result.task_results if item.query_bundle.failed)
        self.record_span(
            state,
            "action",
            "execute_query_graph",
            started,
            status="failed" if failed_tasks else "success",
            row_count=run_result.merged_query_bundle.effective_row_count(),
            error_code="SQL_TASK_FAILED" if failed_tasks else "",
            retry_or_fallback_count=len(run_result.sql_repairs),
            metadata={"tasks": len(run_result.tasks), "failedTasks": failed_tasks, "workerDispatch": state.get("worker_dispatch_context", {})},
        )
        self.finish_run_step(
            state,
            step,
            "success" if not failed_tasks else "partial",
            output_summary="tasks=%d rows=%d failedTasks=%d" % (len(run_result.tasks), run_result.merged_query_bundle.effective_row_count(), failed_tasks),
            error_code="SQL_TASK_FAILED" if failed_tasks else "",
            artifact_paths=[str(Path(state["thread_data"].outputs_path) / "artifacts" / "node" / "agent_run_result.json")],
        )
        emit(state, "node.completed", "EXECUTE_QUERY_GRAPH", {"tasks": len(run_result.tasks), "rows": run_result.merged_query_bundle.effective_row_count()})
        return state

    def repair_sql(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "repair_sql", "NodeAgent", "REPAIR_SQL", input_summary="failedTasks=%d" % sum(1 for item in state["agent_run_result"].task_results if item.query_bundle.failed))
        increment_round(state)
        state["sql_repair_reviewed"] = True
        graph_gaps = graph_gaps_from_node_failures(state["agent_run_result"].task_results)
        if graph_gaps:
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    gaps=graph_gaps,
                    repairable=True,
                ),
            )
            state["last_action_result"] = ActionResult(
                action="repair_sql",
                node="repair_sql",
                status="graph_repair_required",
                message="node failures indicate graph dependency repair is required",
                retryable=True,
            )
        else:
            state["last_action_result"] = ActionResult(
                action="repair_sql",
                node="repair_sql",
                status="reviewed",
                message="node-level SQL repairs reviewed; continue to evidence verification",
            )
        repaired_count = len(state["agent_run_result"].sql_repairs)
        failed_count = sum(1 for item in state["agent_run_result"].task_results if item.query_bundle.failed)
        add_step(state, "Main Agent Tool repair_sql：汇总 NodeAgent SQL 修复，repairs=%d, failedTasks=%d" % (repaired_count, failed_count))
        status = "graph_repair_required" if graph_gaps else "success"
        self.record_span(
            state,
            "sql_repair",
            "repair_sql_review",
            started,
            status="failed" if graph_gaps else "success",
            error_code="PLAN_CONTRACT_MISMATCH" if graph_gaps else "",
            retry_or_fallback_count=repaired_count,
            metadata={"failedTasks": failed_count, "graphGaps": [gap.model_dump(by_alias=True) for gap in graph_gaps]},
        )
        self.finish_run_step(
            state,
            step,
            status,
            output_summary="repairs=%d failedTasks=%d graphGaps=%d" % (repaired_count, failed_count, len(graph_gaps)),
            error_code="PLAN_CONTRACT_MISMATCH" if graph_gaps else "",
        )
        return state

    def verify_evidence_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "verify_evidence", "EvidenceVerifierAgent", "VERIFY_EVIDENCE_GRAPH", input_summary="tasks=%d" % len(state["agent_run_result"].task_results))
        increment_round(state)
        append_knowledge_retrieval_outcomes_to_run_result(state, state["agent_run_result"])
        append_memory_recall_outcomes_to_run_result(state, state["agent_run_result"])
        append_subagent_outcomes_to_run_result(state, state["agent_run_result"])
        verified = self.evidence_verifier.verify(
            state["question"],
            state["plan"],
            state["agent_run_result"],
            state.get("memory_constraints", []),
            recall_knowledge_ref_ids(state),
        )
        state["agent_run_result"].verified_evidence = verified
        state["agent_run_result"].evidence_gaps = verified.gaps
        state["agent_run_result"].partial_answer_reason = verified.partial_answer_reason
        graph_repair_gaps = graph_repair_validation_gaps(verified.gaps)
        if graph_repair_gaps:
            record_graph_validation(
                state,
                GraphValidationResult(
                    valid=False,
                    gaps=graph_repair_gaps,
                    repairable=True,
                ),
            )
        add_step(state, "Main Agent Tool verify_evidence_graph：" + ("证据门禁通过" if verified.passed else "证据存在缺口 %d 个" % len(verified.gaps)))
        state["evidence_graph_verified"] = True
        state["verification_status"] = "passed" if verified.passed else "failed"
        state["evidence_accepted"] = bool(verified.passed)
        state["evidence_generation"] = int(state.get("execution_generation") or 0) if verified.passed else -1
        if not verified.passed and self.latency_optimizer.blocks_expensive_agents(state.get("latency_optimization") or {}):
            self.escalate_fast_request(state, "evidence verification failed on the fast path")
        if not verified.passed:
            state["hypothesis_results"] = []
            state["hypothesis_exploration_status"] = {"status": "blocked", "source": "evidence_gate"}
        self.refresh_context_snapshot(state, "verify_evidence_graph")
        self.record_span(
            state,
            "verifier",
            "verify_evidence_graph",
            started,
            status="success" if verified.passed else "failed",
            error_code=",".join(gap.code for gap in verified.gaps[:4]),
            metadata={"partialReason": verified.partial_answer_reason, "gaps": [gap.model_dump(by_alias=True) for gap in verified.gaps[:12]]},
        )
        self.finish_run_step(
            state,
            step,
            "success" if verified.passed else "partial",
            output_summary="passed=%s gaps=%d" % (verified.passed, len(verified.gaps)),
            error_code=",".join(gap.code for gap in verified.gaps[:4]),
        )
        return state

    def explore_hypotheses(self, state: AgentState) -> AgentState:
        """Compatibility no-op for the retired fixed hypothesis controller.

        Hypothesis work is now expressed as ordinary Lead-selected delegation or
        analysis-worker tasks.  Keeping this unregistered method makes restored
        checkpoints fail closed instead of replaying the former hidden workflow.
        """

        state["hypothesis_exploration"] = {}
        state["hypothesis_results"] = []
        state["candidate_query_graphs"] = {}
        state["hypothesis_exploration_completed"] = True
        state["hypothesis_exploration_status"] = {"status": "retired", "source": "lead_react"}
        state["last_action_result"] = ActionResult(
            action="explore_hypotheses",
            node="explore_hypotheses",
            status="blocked",
            message="fixed hypothesis controller retired; Lead ReAct must use generic delegated tools",
        )
        return state

    def answer_rule(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "answer_rule", "RuleAnswerAgent", "ANSWER_RULE", input_summary="refs=%d" % len(state.get("rule_recall_refs") or []))
        increment_round(state)
        emit(state, "node.started", "ANSWER_RULE", {})
        append_knowledge_retrieval_outcomes_to_run_result(state, state["agent_run_result"])
        append_memory_recall_outcomes_to_run_result(state, state["agent_run_result"])
        state["plan"] = self.build_rule_recall_plan(state)
        record_graph_validation(state, GraphValidationResult(valid=True, repairable=False))
        state["query_graph_reflected"] = True
        state["planner_provider_error"] = ""
        rule_context = state.get("rule_recall_context") or knowledge_context(state)
        rule_context = self.context_assembler.compact_text_context(
            state,
            "answer_rule_context",
            "RuleAnswerAgent",
            rule_context,
            budget_chars=int(self.settings.context_answer_budget_chars or 10000),
        )
        state["answer"] = self.answer_service.compose(
            state["question"],
            state["merchant"],
            state["plan"],
            state["agent_run_result"],
            rule_context,
        )
        state["answer_used_llm"] = bool(self.answer_service.llm.configured and rule_context)
        state["suggestions"] = self.answer_service.contextual_suggestions(
            state["question"],
            state["plan"].intents,
            run_result=state.get("agent_run_result"),
            merchant=state.get("merchant"),
            personalization_context=self.answer_personalization_context(state),
        )
        state["merchant_experience"] = self.answer_service.merchant_experience(
            state["question"],
            state["plan"],
            state.get("agent_run_result"),
            merchant=state.get("merchant"),
            sections=self.answer_service.build_sections(state["plan"], state["agent_run_result"]),
            suggestions=state.get("suggestions", []),
            personalization_context=self.answer_personalization_context(state),
        )
        state["chat_bi_completed"] = True
        state["should_persist"] = False
        state["persisted"] = False
        add_step(state, "RuleAnswerAgent：基于召回规则知识回答，未生成 QueryGraph/SQL")
        self.record_span(
            state,
            "llm" if state.get("answer_used_llm") else "answer",
            "answer.rule_llm" if state.get("answer_used_llm") else "answer.rule_structured",
            started,
            model=self.settings.openai_model,
            provider=self.settings.openai_base_url,
            estimated_prompt_chars=len(rule_context),
            estimated_completion_chars=len(state.get("answer", "")),
            metadata={"refs": state.get("rule_recall_refs") or []},
        )
        self.finish_run_step(state, step, "success", output_summary="answerChars=%d" % len(state.get("answer", "")))
        emit(state, "node.completed", "ANSWER_RULE", {"answerReady": bool(state["answer"])})
        return state

    def delegate_subagent(self, state: AgentState) -> AgentState:
        """Formal Lead Agent action for bounded, worker-independent delegation."""
        started = now_ms()
        step = self.start_run_step(
            state,
            "delegate_subagent",
            "LeadAgent",
            "DELEGATE_SUBAGENT",
            input_summary=str(state.get("question") or "")[:300],
        )
        increment_round(state)
        state["subagent_delegation_attempted"] = True
        emit(state, "node.started", "DELEGATE_SUBAGENT", {})
        allowed_kinds = self._allowed_delegation_kinds(state)
        plan = self._build_delegation_plan(state, allowed_kinds)
        state["subagent_delegation_plan"] = plan.model_dump(by_alias=True)
        if not plan.tasks:
            contract = normalize_subagent_result(
                "delegate_subagent",
                "failed",
                {},
                "no safe, bounded Sub-Agent task could be constructed",
            )
            contract["recommendedNextAction"] = "continue_in_lead_agent"
            state["subagent_delegation_results"] = [contract]
            state["subagent_delegation_completed"] = True
            self.finish_run_step(state, step, "partial", output_summary="no delegable task", error_code="NO_DELEGABLE_TASK")
            emit(state, "node.completed", "DELEGATE_SUBAGENT", {"completed": True, "taskCount": 0})
            return state

        tasks = list(plan.tasks)[: max(1, int(self.settings.max_sub_agent_tasks or 1))]
        results: List[Dict[str, Any]] = []
        if plan.parallel and len(tasks) > 1:
            workers = min(len(tasks), max(1, int(self.settings.max_concurrent_sub_agents or 1)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lead-delegate") as pool:
                futures = {
                    submit_with_current_context(
                        pool,
                        self._execute_delegation_task,
                        state,
                        task,
                        plan.failure_strategy,
                        plan.read_artifact_policy,
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        results.append(normalize_subagent_result(task.task_kind, "failed", {}, str(exc)))
        else:
            for task in tasks:
                results.append(self._execute_delegation_task(state, task, plan.failure_strategy, plan.read_artifact_policy))

        state["subagent_delegation_results"] = results
        state["subagent_delegation_completed"] = True
        completed = [item for item in results if item.get("status") == "completed"]
        partial = [item for item in results if item.get("status") == "partial"]
        failed = [item for item in results if item.get("status") not in {"completed", "partial"}]
        analysis_summary_task_kinds = {"document_analysis", "analysis_worker", "analysis_skill", "hypothesis_review"}
        summaries: List[str] = []
        for item in [*completed, *partial]:
            task_kind = str(item.get("taskKind") or item.get("task_kind") or "")
            summary = str(item.get("summary") or "").strip()
            if task_kind not in analysis_summary_task_kinds or not summary:
                continue
            if item.get("status") == "partial":
                codes = [
                    str(gap.get("code") or "")
                    for gap in item.get("gaps") or []
                    if isinstance(gap, dict) and str(gap.get("code") or "")
                ]
                summary = "[partial%s] %s" % (":" + ",".join(codes[:3]) if codes else "", summary)
            summaries.append(summary)
        if summaries:
            delegated_summary = "\n".join("- %s" % text for text in summaries)
            prior = str(state.get("analysis_summary") or "").strip()
            state["analysis_summary"] = "\n".join(item for item in (prior, "Sub-Agent 结果：\n%s" % delegated_summary) if item)
            state["analysis_generation"] = int(state.get("execution_generation") or 0)
        append_subagent_outcomes_to_run_result(state, state.get("agent_run_result") or AgentRunResult())
        observations = list(state.get("main_agent_observations") or [])
        observations.append(
            {
                "stage": "delegate_subagent",
                "summary": "%d/%d Sub-Agent tasks completed; partial=%d failed=%d"
                % (len(completed), len(results), len(partial), len(failed)),
                "plan": plan.model_dump(by_alias=True),
                "results": results,
            }
        )
        state["main_agent_observations"] = observations
        status = "success" if len(completed) == len(results) else "partial"
        self.record_span(
            state,
            "worker",
            "lead_agent.delegate_subagent",
            started,
            status=status,
            metadata={"taskKinds": [task.task_kind for task in tasks], "parallel": plan.parallel, "failureStrategy": plan.failure_strategy},
        )
        self.finish_run_step(
            state,
            step,
            status,
            output_summary="completed=%d total=%d" % (len(completed), len(results)),
            error_code="" if status == "success" else "SUBAGENT_PARTIAL_FAILURE",
        )
        emit(
            state,
            "node.completed",
            "DELEGATE_SUBAGENT",
            {
                "completed": True,
                "status": status,
                "taskCount": len(results),
                "successCount": len(completed),
                "partialCount": len(partial),
                "failedCount": len(failed),
                "parallel": plan.parallel,
            },
        )
        return state

    def _allowed_delegation_kinds(self, state: AgentState) -> List[str]:
        kinds: List[str] = []
        context = state.get("request_context")
        files = list(getattr(context, "offloaded_files", None) or [])
        question = str(state.get("question") or "").lower()
        if files or "[用户附件上下文]" in question:
            kinds.append("document_analysis")
        if any(str(path).lower().endswith(".py") for path in files):
            kinds.append("python_batch")
        if state.get("query_graph_validated") and not state.get("sql_generated") and getattr(state.get("plan"), "intents", None):
            kinds.append("query_node")
        if self.policy.analysis_worker_needed(state):
            kinds.append("analysis_worker")
        if self.policy.analysis_skill_needed(state):
            kinds.append("analysis_skill")
        return list(dict.fromkeys(kinds))

    def _build_delegation_plan(self, state: AgentState, allowed_kinds: List[str]) -> SubAgentDelegationPlan:
        fallback = self._fallback_delegation_plan(state, allowed_kinds)
        if not allowed_kinds:
            return fallback
        llm = getattr(self.planner, "llm", None)
        if not llm or not getattr(llm, "configured", False) or not hasattr(llm, "tool_json_chat"):
            return fallback
        tool = delegate_subagent_tool(allowed_kinds)
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        payload = {
            "question": state.get("question", ""),
            "availableTaskKinds": allowed_kinds,
            "hasValidatedQueryGraph": bool(state.get("query_graph_validated")),
            "evidenceChecked": bool(state.get("evidence_graph_verified")),
            "evidenceAccepted": bool(evidence_accepted_for_state(state)),
            "attachmentRefs": list(getattr(state.get("request_context"), "offloaded_files", None) or []),
            "queryTasks": [
                {"taskId": item.plan_task_id, "question": item.question, "answerMode": item.answer_mode}
                for item in (state.get("plan") or QueryPlan()).intents[: max(1, int(self.settings.max_sub_agent_tasks or 1))]
            ],
            "matchedSkill": getattr(state.get("skill_match"), "skill_name", ""),
            "hypothesisIds": [str(item.get("hypothesisId") or "") for item in hypotheses if isinstance(item, dict)],
            "instruction": "Only delegate self-contained work that benefits from isolation. Tasks marked parallel must be independent.",
        }
        try:
            raw = llm.tool_json_chat(
                "你是 Lead Agent 的受限委派规划器。只可使用给定 task kind；文件、查询和数据输入会由 Runtime 重新校验。",
                json.dumps(payload, ensure_ascii=False, default=str),
                tool.openai_schema(),
                fallback.model_dump(by_alias=True),
                timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
            )
            plan = SubAgentDelegationPlan.model_validate(raw or fallback.model_dump(by_alias=True))
        except Exception:
            return fallback
        valid_tasks = [task for task in plan.tasks if task.task_kind in allowed_kinds and task.objective.strip()]
        return plan.model_copy(
            update={
                "tasks": valid_tasks[: max(1, int(self.settings.max_sub_agent_tasks or 1))],
                "isolation_mode": "worker",
                "failure_strategy": plan.failure_strategy if plan.failure_strategy in {"retry", "fallback", "repair", "continue_partial"} else "continue_partial",
                "read_artifact_policy": plan.read_artifact_policy if plan.read_artifact_policy in {"on_completion", "summary_first"} else "on_completion",
            }
        )

    def _fallback_delegation_plan(self, state: AgentState, allowed_kinds: List[str]) -> SubAgentDelegationPlan:
        question = str(state.get("question") or "")
        context = state.get("request_context")
        files = list(getattr(context, "offloaded_files", None) or [])
        task_kind = ""
        if "python_batch" in allowed_kinds and any(str(path).casefold().endswith(".py") for path in files):
            task_kind = "python_batch"
        elif "document_analysis" in allowed_kinds:
            task_kind = "document_analysis"
        elif "hypothesis_review" in allowed_kinds:
            task_kind = "hypothesis_review"
        elif "analysis_worker" in allowed_kinds:
            task_kind = "analysis_worker"
        elif "analysis_skill" in allowed_kinds:
            task_kind = "analysis_skill"
        elif "query_node" in allowed_kinds:
            task_kind = "query_node"
        tasks = []
        if task_kind:
            tasks.append(
                SubAgentDelegationTask(
                    task_kind=task_kind,
                    objective=question or "完成隔离分析并返回证据与缺口",
                    inputs={},
                    expected_outputs=["summary", "evidenceRefs", "gaps"],
                    timeout=min(120, int(self.settings.distributed_worker_result_timeout_seconds or 120)),
                )
            )
        return SubAgentDelegationPlan(tasks=tasks, parallel=False, reason="deterministic safe delegation fallback")

    def _execute_delegation_task(
        self,
        state: AgentState,
        task: SubAgentDelegationTask,
        failure_strategy: str,
        read_artifact_policy: str,
    ) -> Dict[str, Any]:
        timeout = max(1, min(int(task.timeout or 60), int(self.settings.distributed_worker_result_timeout_seconds or 180)))
        try:
            request = self._delegation_request(state, task, timeout)
        except Exception as exc:
            contract = normalize_subagent_result(task.task_kind, "failed", {}, "invalid delegation input: %s" % str(exc))
            contract["recommendedNextAction"] = "repair_delegation"
            return contract
        attempts = 2 if failure_strategy == "retry" else 1
        contract: Dict[str, Any] = {}
        actual_attempts = 0
        for attempt in range(attempts):
            actual_attempts = attempt + 1
            task_kind = safe_ascii_component(task.task_kind, extras=("_", "-"), default="worker")
            task_id = "delegate_%s_%s" % (task_kind, uuid.uuid4().hex[:12])
            if bool(self.settings.distributed_subagents_enabled):
                result = DistributedSubAgentClient(self.settings).execute(
                    str(state.get("run_id") or state.get("qa_id") or "delegation"),
                    task_id,
                    task.task_kind,
                    request,
                    timeout,
                    read_artifact=read_artifact_policy == "on_completion",
                )
                contract = dict(result.contract or normalize_subagent_result(task.task_kind, result.status, result.result, result.error, result.artifact_uri))
            else:
                handler = builtin_worker_handlers(self.settings).get(task.task_kind)
                if not handler:
                    contract = normalize_subagent_result(task.task_kind, "failed", {}, "unsupported task kind")
                else:
                    try:
                        outcome = coerce_handler_outcome(handler(request, lambda: bool(state.get("run_canceled"))))
                        contract = normalize_subagent_result(
                            task.task_kind,
                            outcome.status,
                            outcome.payload,
                            outcome.error,
                        )
                    except Exception as exc:
                        contract = normalize_subagent_result(task.task_kind, "failed", {}, "%s: %s" % (type(exc).__name__, str(exc)))
            if contract.get("status") == "completed" or not contract.get("retryable") or attempt + 1 >= attempts:
                break
        contract["objective"] = task.objective
        contract["expectedOutputs"] = list(task.expected_outputs)
        contract["attempts"] = actual_attempts
        if contract.get("status") != "completed" and failure_strategy == "fallback":
            contract["recommendedNextAction"] = "fallback_to_lead_agent"
        elif contract.get("status") != "completed" and failure_strategy == "repair":
            contract["recommendedNextAction"] = "repair_delegation"
        return contract

    def _delegation_request(self, state: AgentState, task: SubAgentDelegationTask, timeout: int) -> Dict[str, Any]:
        inputs = dict(task.inputs or {})
        if task.task_kind == "document_analysis":
            content = self._delegation_document_content(state)
            if not content:
                raise ValueError("no readable document content in current task workspace")
            return {"content": content, "question": task.objective or state.get("question", "")}
        if task.task_kind == "python_batch":
            script = self._approved_delegation_script(state, str(inputs.get("scriptPath") or ""))
            return {
                "scriptPath": str(script),
                "workspacePath": state["thread_data"].outputs_path,
                "args": [str(item) for item in inputs.get("args") or []][:20],
                "timeoutSeconds": timeout,
            }
        if task.task_kind == "hypothesis_review":
            return {
                "hypotheses": dict(state.get("hypothesis_exploration") or {}),
                "runResult": (state.get("agent_run_result") or AgentRunResult()).model_dump(by_alias=True),
            }
        if task.task_kind == "analysis_worker":
            return {
                "question": task.objective or state.get("question", ""),
                "plan": (state.get("plan") or QueryPlan()).model_dump(by_alias=True),
                "runResult": (state.get("agent_run_result") or AgentRunResult()).model_dump(by_alias=True),
                "outputsPath": state["thread_data"].outputs_path,
                "ruleContext": state.get("rule_recall_context", ""),
                "merchant": state["merchant"].model_dump(by_alias=True),
                "personalizationContext": self.answer_personalization_context(state),
                "initialTrace": {"parentRunId": state.get("run_id") or state.get("qa_id") or "analysis_worker"},
            }
        if task.task_kind == "analysis_skill":
            match = state.get("skill_match") or SkillMatchState()
            skill_name = str(inputs.get("skillName") or getattr(match, "skill_name", "") or "")
            if not skill_name:
                raise ValueError("no reviewed analysis skill is available")
            return {
                "question": task.objective or state.get("question", ""),
                "plan": (state.get("plan") or QueryPlan()).model_dump(by_alias=True),
                "runResult": (state.get("agent_run_result") or AgentRunResult()).model_dump(by_alias=True),
                "outputsPath": state["thread_data"].outputs_path,
                "ruleContext": state.get("rule_recall_context", ""),
                "skillName": skill_name,
                "merchant": state["merchant"].model_dump(by_alias=True),
                "personalizationContext": self.answer_personalization_context(state),
            }
        if task.task_kind == "query_node":
            plan = state.get("plan") or QueryPlan()
            if not state.get("query_graph_validated") or not plan.intents:
                raise ValueError("query_node delegation requires a validated QueryGraph")
            requested_id = str(inputs.get("taskId") or "")
            intent = next((item for item in plan.intents if item.plan_task_id == requested_id), plan.intents[0])
            return {
                "intent": intent.model_dump(by_alias=True),
                "assetPack": (state.get("planning_asset_pack") or PlanningAssetPack()).model_dump(by_alias=True),
                "knowledgeContext": knowledge_context(state),
                "context": {
                    "merchantId": state.get("requested_merchant_id", ""),
                    "effectiveUserId": str((state.get("user_identity") or {}).get("userId") or ""),
                    "authorizedRegion": str((state.get("user_identity") or {}).get("region") or ""),
                    "authorizedStoreIds": list((state.get("user_identity") or {}).get("storeIds") or []),
                    "accessRole": state.get("access_role", load_authorization_policy().default_access_role),
                    "question": task.objective or state.get("question", ""),
                    "subAgentRunId": str(state.get("run_id") or ""),
                    "workspacePath": state["thread_data"].workspace_path,
                },
            }
        raise ValueError("unsupported task kind: %s" % task.task_kind)

    def _delegation_document_content(self, state: AgentState) -> str:
        context = state.get("request_context")
        refs = list(getattr(context, "offloaded_files", None) or [])
        allowed_roots = [Path(state["thread_data"].workspace_path).resolve(), self.settings.resolved_workspace_path.resolve()]
        chunks: List[str] = []
        for ref in refs[:10]:
            path = Path(str(ref)).expanduser().resolve()
            if not any(path.is_relative_to(root) for root in allowed_roots) or not path.is_file():
                continue
            try:
                chunks.append("## %s\n%s" % (path.name, path.read_text(encoding="utf-8", errors="replace")[:100_000]))
            except OSError:
                continue
        question = str(state.get("question") or "")
        if "[用户附件上下文]" in question:
            chunks.append(question.split("[用户附件上下文]", 1)[1][:100_000])
        return "\n\n".join(chunks)[:100_000]

    def _approved_delegation_script(self, state: AgentState, requested: str) -> Path:
        context = state.get("request_context")
        refs = [Path(str(item)).expanduser().resolve() for item in (getattr(context, "offloaded_files", None) or []) if str(item).lower().endswith(".py")]
        candidate = requested or (refs[0] if refs else "")
        if not candidate:
            raise ValueError("python_batch requires a .py attachment")
        path = Path(candidate).expanduser().resolve()
        if path.suffix.lower() != ".py" or not path.is_file() or path not in refs:
            raise ValueError("scriptPath is not a Python attachment from the current request")
        return path

    def run_analysis_worker(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(
            state,
            "run_analysis_worker",
            "AnalysisWorker",
            "RUN_ANALYSIS_WORKER",
            input_summary="tasks=%d" % len(state["agent_run_result"].task_results),
        )
        increment_round(state)
        emit(state, "node.started", "RUN_ANALYSIS_WORKER", {})
        if not evidence_accepted_for_state(state):
            self.clear_analysis_outputs(state, "证据未通过，阻止 AnalysisWorker 使用旧分析")
            state["analysis_worker_status"] = {"status": "blocked", "source": "evidence_gate"}
            self.finish_run_step(state, step, "gap", output_summary="blocked: evidence verification failed", error_code="EVIDENCE_NOT_ACCEPTED")
            emit(state, "node.completed", "RUN_ANALYSIS_WORKER", {"completed": False, "evidenceAccepted": False})
            return state
        if planner_degraded_stops_expensive_work(state):
            self.clear_analysis_outputs(state, "planner degraded，跳过 AnalysisWorker")
            state["analysis_worker_status"] = {"status": "skipped", "source": "planner_degraded"}
            self.finish_run_step(state, step, "partial", output_summary="skipped: planner degraded fail-fast", error_code="PLANNER_DEGRADED_FAIL_FAST")
            emit(state, "node.completed", "RUN_ANALYSIS_WORKER", {"completed": False, "degradedSkipped": True})
            return state

        personalization_context = self.answer_personalization_context(state)
        result = AnalysisWorkerExecutor(self.answer_service.llm).execute(
            state["question"],
            state["plan"],
            state["agent_run_result"],
            state["thread_data"].outputs_path,
            state.get("rule_recall_context", ""),
            merchant=state.get("merchant"),
            personalization_context=personalization_context,
            initial_trace={"parentRunId": state.get("run_id") or state.get("qa_id") or "analysis_worker"},
        )
        state["analysis_summary"] = result.answer
        state["analysis_generation"] = int(state.get("execution_generation") or 0)
        trace = result.trace
        state["analysis_worker_trace"] = trace
        state["analysis_worker_completed"] = bool(state.get("analysis_summary")) and not bool(trace.get("error"))
        state["analysis_worker_status"] = {
            "status": "completed" if state["analysis_worker_completed"] else "failed",
            "source": "runtime",
        }
        add_step(
            state,
            "LeadAgent Tool run_analysis_worker：已调度通用 AnalysisWorker，status=%s"
            % trace.get("lifecycleStage", "none"),
        )
        self.record_span(
            state,
            "worker",
            "analysis_worker:general",
            started,
            status="success" if state.get("analysis_worker_completed") else "failed",
            error_code=str(trace.get("error") or ""),
            estimated_completion_chars=len(state.get("analysis_summary") or ""),
            metadata={"analysisWorkerTrace": trace, "isolatedExecution": True},
        )
        self.finish_run_step(
            state,
            step,
            "success" if state.get("analysis_worker_completed") else "partial",
            output_summary="summaryChars=%d" % len(state.get("analysis_summary") or ""),
            error_code=str(trace.get("error") or ""),
        )
        emit(
            state,
            "node.completed",
            "RUN_ANALYSIS_WORKER",
            {
                "completed": bool(state.get("analysis_worker_completed")),
                "summaryChars": len(state.get("analysis_summary") or ""),
                "workerType": trace.get("workerType"),
            },
        )
        return state

    def run_analysis_skill(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(
            state,
            "run_analysis_skill",
            "SkillWorker",
            "RUN_ANALYSIS_SKILL",
            input_summary="tasks=%d" % len(state["agent_run_result"].task_results),
        )
        increment_round(state)
        emit(state, "node.started", "RUN_ANALYSIS_SKILL", {})
        if not evidence_accepted_for_state(state):
            self.clear_analysis_outputs(state, "证据未通过，阻止 SkillWorker 使用旧分析")
            state["analysis_skill_status"] = {"status": "blocked", "source": "evidence_gate"}
            self.finish_run_step(state, step, "gap", output_summary="blocked: evidence verification failed", error_code="EVIDENCE_NOT_ACCEPTED")
            emit(state, "node.completed", "RUN_ANALYSIS_SKILL", {"completed": False, "evidenceAccepted": False})
            return state
        if planner_degraded_stops_expensive_work(state):
            self.clear_analysis_outputs(state, "planner degraded，跳过 SkillWorker")
            state["analysis_skill_status"] = {"status": "skipped", "source": "planner_degraded"}
            self.finish_run_step(
                state,
                step,
                "partial",
                output_summary="skipped: planner degraded fail-fast",
                error_code="PLANNER_DEGRADED_FAIL_FAST",
            )
            emit(
                state,
                "node.completed",
                "RUN_ANALYSIS_SKILL",
                {"completed": False, "degradedSkipped": True},
            )
            return state
        match = self.match_analysis_skill(state)
        if state.get("analysis_skill_bypassed"):
            self.clear_analysis_outputs(state, "商家跳过 SkillWorker")
            self.finish_run_step(state, step, "success", output_summary="merchant chose current verified result")
            emit(state, "node.completed", "RUN_ANALYSIS_SKILL", {"completed": False, "bypassed": True})
            return state
        if not match.skill_name:
            self.clear_analysis_outputs(state, "未匹配 SkillWorker")
            state["analysis_skill_trace"] = {
                "error": "NO_MATCHED_SKILL",
                "skillName": "",
                "matchStatus": match.status,
                "candidateSkills": list(match.candidate_skills or []),
                "reason": match.reason,
            }
            state["analysis_skill_status"] = {"status": "no_match", "source": "skill_match"}
            self.finish_run_step(state, step, "partial", output_summary="no matched skill", error_code="NO_MATCHED_SKILL")
            emit(state, "node.completed", "RUN_ANALYSIS_SKILL", {"completed": False, "reason": "NO_MATCHED_SKILL"})
            return state
        if self.maybe_request_skill_confirmation(state):
            self.clear_analysis_outputs(state, "等待 SkillWorker 人工确认")
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="skill confirmation required",
                error_code="SKILL_CONFIRMATION_REQUIRED",
            )
            emit(state, "node.completed", "RUN_ANALYSIS_SKILL", {"confirmationRequired": True})
            return state
        match = state.get("skill_match") if isinstance(state.get("skill_match"), SkillMatchState) else match
        if isinstance(match, SkillMatchState):
            match = match.model_copy(update={"confirmed": True, "status": "confirmed" if match.requires_confirmation else "ready"})
            state["skill_match"] = match
            dispatch_trace = dict(getattr(self.answer_service, "last_analysis_skill_trace", {}) or {})
            dispatch_trace.update(
                {
                    "requiresConfirmation": match.requires_confirmation,
                    "confirmed": match.confirmed,
                    "confirmationStatus": match.status,
                    "parentRunId": state.get("run_id") or state.get("qa_id") or "skill_run",
                }
            )
            self.answer_service.last_analysis_skill_trace = dispatch_trace
        personalization_context = self.answer_personalization_context(state)
        self.emit_skill_lifecycle_event(state, "confirmed", match, {"confirmed": True})
        parallel_skill_names = self.parallel_skill_names(match)
        if len(parallel_skill_names) > 1:
            self.emit_skill_lifecycle_event(state, "parallel_isolated_execute", match, {"workerType": "SKILL_WORKER_BATCH", "skillNames": parallel_skill_names})
            state["analysis_summary"] = self.answer_service.run_parallel_analysis_skills(
                state["question"],
                state["plan"],
                state["agent_run_result"],
                parallel_skill_names,
                state["thread_data"].outputs_path,
                state.get("rule_recall_context", ""),
                merchant=state["merchant"],
                personalization_context=personalization_context,
            )
        else:
            self.emit_skill_lifecycle_event(state, "isolated_execute", match, {"workerType": "SKILL_WORKER"})
            state["analysis_summary"] = self.answer_service.run_analysis_skill(
                state["question"],
                state["plan"],
                state["agent_run_result"],
                state["thread_data"].outputs_path,
                state.get("rule_recall_context", ""),
                skill_name=match.skill_name,
                merchant=state["merchant"],
                personalization_context=personalization_context,
            )
        state["analysis_generation"] = int(state.get("execution_generation") or 0)
        state["analysis_skill_trace"] = dict(getattr(self.answer_service, "last_analysis_skill_trace", {}) or {})
        self.record_skill_lifecycle(state, state["analysis_skill_trace"])
        trace = state.get("analysis_skill_trace") or {}
        if trace.get("progress"):
            self.emit_skill_lifecycle_event(state, "progress_synced", match, {"progress": trace.get("progress") or []})
        state["skill_worker_completed"] = bool(state.get("analysis_summary")) and not bool(trace.get("error"))
        state["analysis_skill_status"] = {
            "status": "completed" if state["skill_worker_completed"] else "failed",
            "source": "runtime",
        }
        if isinstance(state.get("skill_match"), SkillMatchState):
            state["skill_match"] = state["skill_match"].model_copy(
                update={
                    "status": "completed" if state.get("skill_worker_completed") else "failed",
                    "confirmed": True,
                    "trace": trace,
                }
            )
        add_step(
            state,
            "LeadAgent Tool run_analysis_skill：已调度 SkillWorker，skill=%s status=%s"
            % (trace.get("skillName") or "", trace.get("lifecycleStage") or "none"),
        )
        self.record_span(
            state,
            "worker",
            "skill_worker:%s" % (trace.get("skillName") or "unknown"),
            started,
            status="success" if state.get("skill_worker_completed") else "failed",
            error_code=str(trace.get("error") or ""),
            estimated_completion_chars=len(state.get("analysis_summary") or ""),
            metadata={
                "skillTrace": trace,
                "subAgentType": trace.get("subAgentType") or trace.get("workerType"),
                "isolatedExecution": trace.get("isolatedExecution"),
            },
        )
        self.finish_run_step(
            state,
            step,
            "success" if state.get("skill_worker_completed") else "partial",
            output_summary="skill=%s summaryChars=%d" % (trace.get("skillName") or "", len(state.get("analysis_summary") or "")),
            error_code=str(trace.get("error") or ""),
        )
        emit(
            state,
            "node.completed",
            "RUN_ANALYSIS_SKILL",
            {
                "skillName": trace.get("skillName"),
                "completed": bool(state.get("skill_worker_completed")),
                "summaryChars": len(state.get("analysis_summary") or ""),
                "workerType": trace.get("workerType"),
            },
        )
        self.emit_skill_lifecycle_event(
            state,
            "completed" if state.get("skill_worker_completed") else "failed",
            match,
            {"summaryChars": len(state.get("analysis_summary") or ""), "error": trace.get("error") or ""},
        )
        return state

    def parallel_skill_names(self, match: SkillMatchState) -> List[str]:
        if not bool(getattr(self.settings, "skill_worker_parallel_enabled", False)):
            return [match.skill_name] if match.skill_name else []
        names: List[str] = []
        reviewed_parallel = list((match.trace or {}).get("parallelSkillNames") or (match.trace or {}).get("parallel_skill_names") or [])
        for candidate in [match.skill_name] + reviewed_parallel:
            name = str(candidate or "").strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return []
        limit = max(1, int(getattr(self.settings, "max_concurrent_skill_workers", 2) or 2))
        return names[:limit]

    def match_analysis_skill(self, state: AgentState) -> SkillMatchState:
        current = state.get("skill_match")
        if isinstance(current, SkillMatchState) and current.skill_name:
            return current
        plan = state.get("plan") or QueryPlan()
        run_result = state.get("agent_run_result") or AgentRunResult()
        has_rule_context = bool(state.get("rule_recall_context", ""))
        selected = self.answer_service.propose_answer_skill(
            state.get("question", ""),
            plan,
            run_result,
            has_rule_context,
        )
        trace = dict(getattr(self.answer_service, "last_analysis_skill_trace", {}) or {})
        headers = self.skill_header_payloads(answer_skill_headers(self.settings.resources_root / "runtime" / "agent_skills"))
        match = SkillMatchState(
            skill_name=selected,
            status="matched" if selected else "no_match",
            matched_by=str(trace.get("matchedBy") or ""),
            match_source=str(trace.get("matchMode") or ""),
            confidence=float(trace.get("confidence") or (0.6 if selected else 0.0)),
            reason=str(trace.get("reason") or ("fallback skill selected" if selected else "no skill selected")),
            candidate_skills=[str(item) for item in trace.get("candidateSkills", []) if str(item)],
            fallback_skill=str(trace.get("fallbackSkill") or ""),
            requires_confirmation=bool(getattr(self.settings, "skill_confirmation_required", False)),
            confirmed=not bool(getattr(self.settings, "skill_confirmation_required", False)),
            headers=headers,
            trace=trace,
        )
        state["skill_match"] = match
        state["skill_dispatch_context"] = self.skill_dispatch_context(state)
        self.emit_skill_lifecycle_event(state, "matched", match, {"candidateSkills": match.candidate_skills, "reason": match.reason})
        record = SkillLifecycleRecord(
            record_id=self.skill_lifecycle_record_id(match.skill_name, "matched"),
            skill_name=match.skill_name,
            stage="matched",
            status=match.status,
            matched_by=match.matched_by,
            requires_confirmation=match.requires_confirmation,
            confirmed=match.confirmed,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            progress=["matched"],
            summary=match.reason,
            metadata={"confidence": match.confidence, "fallbackSkill": match.fallback_skill},
        )
        if match.skill_name:
            state.setdefault("skill_lifecycle_records", []).append(record)
            state["agent_run_result"].skill_lifecycle_records.append(record)
        return match

    def emit_skill_lifecycle_event(
        self,
        state: AgentState,
        stage: str,
        match: SkillMatchState,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "skillName": match.skill_name,
            "stage": stage,
            "status": stage,
            "matchedBy": match.matched_by,
            "requiresConfirmation": match.requires_confirmation,
            "confirmed": match.confirmed if stage not in {"confirmed", "isolated_execute", "progress_synced", "completed"} else True,
            "confidence": match.confidence,
        }
        payload.update(extra or {})
        emit(state, "skill.lifecycle", "RUN_ANALYSIS_SKILL", payload)

    def answer_analysis(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "answer", "AnswerAgent", "ANSWER_ANALYSIS", input_summary="verified=%s" % state["agent_run_result"].verified_evidence.passed)
        increment_round(state)
        emit(state, "node.started", "ANSWER_ANALYSIS", {})
        append_knowledge_retrieval_outcomes_to_run_result(state, state["agent_run_result"])
        append_memory_recall_outcomes_to_run_result(state, state["agent_run_result"])
        personalization_context = self.answer_personalization_context(state)
        route = state["routing_decision"].route
        allow_answer_llm = self.latency_optimizer.answer_allows_llm(state.get("latency_optimization") or {})
        answer_llm_reserve = max(3, int(getattr(self.settings, "llm_answer_timeout_seconds", 15) or 15) + 2)
        if remaining_run_budget_seconds(state, self.settings) <= answer_llm_reserve:
            allow_answer_llm = False
            state.setdefault("route_decision_trace", []).append(
                {
                    "stage": "answer_latency_budget",
                    "decision": "structured_answer_only",
                    "remainingSeconds": remaining_run_budget_seconds(state, self.settings),
                    "requiredSeconds": answer_llm_reserve,
                }
            )
        if route == QuestionRoute.GREETING:
            state["plan"] = QueryPlan(
                intents=[
                    QuestionIntent(
                        question=state["question"],
                        answer_mode=AnswerMode.CHAT,
                        intent_type="GREETING",
                        category="UNKNOWN",
                        plan_task_id="chat_1",
                    )
                ]
            )
            state["answer"] = self.answer_service.compose(
                state["question"],
                state["merchant"],
                state["plan"],
                state["agent_run_result"],
                self.context_assembler.compact_text_context(
                    state,
                    "answer_chat_context",
                    "AnswerAgent",
                    knowledge_context(state),
                    budget_chars=int(self.settings.context_answer_budget_chars or 10000),
                ),
                personalization_context=personalization_context,
            )
        elif route == QuestionRoute.INVALID:
            self.request_human_clarification(
                state,
                self.build_scope_clarification_prompt(state),
                "BUSINESS_SCOPE",
                "business_scope",
                business_scope_examples(self.recall_service.topic_assets),
            )
            return self.human_in_loop(state)
        else:
            self.guard_unaccepted_evidence_for_answer(state)
            answer_package = self.prepare_scoped_context_package(
                state,
                "answer_analysis",
                "AnswerAgent",
                allowed_tables=(state.get("planning_asset_pack") or PlanningAssetPack()).known_tables()[:12],
                allowed_metrics=[
                    item.key
                    for item in (state.get("planning_asset_pack") or PlanningAssetPack()).metrics[:24]
                    if item.key
                ],
            )
            answer_file_context = self.answer_file_tool_context(state, answer_package)
            answer_context_source = append_context_section(
                knowledge_context(state),
                append_context_section(
                    self.render_context_package_for_prompt(answer_package),
                    answer_file_context,
                    max_chars=int(self.settings.context_answer_budget_chars or 10000),
                ),
                max_chars=int(self.settings.context_answer_budget_chars or 10000),
            )
            answer_context = self.context_assembler.compact_text_context(
                state,
                "answer_knowledge_context",
                "AnswerAgent",
                answer_context_source,
                budget_chars=int(self.settings.context_answer_budget_chars or 10000),
            )
            state["answer"] = self.answer_service.compose(
                state["question"],
                state["merchant"],
                state["plan"],
                state["agent_run_result"],
                answer_context,
                current_analysis_summary_for_state(state),
                allow_llm=allow_answer_llm,
                rule_context=state.get("rule_recall_context", ""),
                personalization_context=personalization_context,
            )
            skill_trace = state.get("analysis_skill_trace") or {}
            worker_trace = state.get("analysis_worker_trace") or {}
            state["answer_used_llm"] = bool(
                skill_trace.get("llmFallbackUsed")
                or (worker_trace.get("analysisTrace") or {}).get("llmFallbackUsed")
                or getattr(self.answer_service, "last_compose_used_llm", False)
            )
        state["suggestions"] = self.answer_service.contextual_suggestions(
            state["question"],
            state["plan"].intents,
            run_result=state.get("agent_run_result"),
            merchant=state.get("merchant"),
            personalization_context=personalization_context,
        )
        state["merchant_experience"] = self.answer_service.merchant_experience(
            state["question"],
            state["plan"],
            state.get("agent_run_result"),
            merchant=state.get("merchant"),
            sections=self.answer_service.build_sections(state["plan"], state["agent_run_result"]),
            suggestions=state.get("suggestions", []),
            personalization_context=personalization_context,
        )
        state["chat_bi_completed"] = True
        add_step(state, "Result Loop：完成结果解读、建议生成与可视化数据组织")
        self.record_span(
            state,
            "llm" if state.get("answer_used_llm") else "answer",
            "answer.compose" if state.get("answer_used_llm") else "answer.compose_structured",
            started,
            model=self.settings.openai_model,
            provider=self.settings.openai_base_url,
            estimated_prompt_chars=(
                int(getattr(self.answer_service, "last_prompt_chars", 0) or 0)
                if state.get("answer_used_llm")
                else len(state.get("summary_context", "")) + len(json.dumps(state["agent_run_result"].model_dump(by_alias=True), ensure_ascii=False, default=str))
            ),
            estimated_completion_chars=len(state.get("answer", "")),
            metadata={"usedLlm": bool(state.get("answer_used_llm"))},
        )
        self.finish_run_step(state, step, "success", output_summary="answerChars=%d" % len(state.get("answer", "")))
        emit(state, "node.completed", "ANSWER_ANALYSIS", {"answerReady": bool(state["answer"])})
        return state

    def maybe_request_skill_confirmation(self, state: AgentState) -> bool:
        match = state.get("skill_match")
        if not isinstance(match, SkillMatchState) or not match.skill_name:
            match = self.match_analysis_skill(state)
        if not isinstance(match, SkillMatchState) or not match.requires_confirmation:
            return False
        context = state.get("request_context")
        if context and context.pending_clarification_type == "skill_confirm":
            if skill_confirmation_declined(state.get("clarification_resolution") or {}):
                state["analysis_skill_bypassed"] = True
                state["analysis_skill_status"] = {"status": "declined", "source": "user"}
                return False
            match = state.get("skill_match")
            if isinstance(match, SkillMatchState):
                state["skill_match"] = match.model_copy(update={"confirmed": True, "status": "confirmed"})
                state["analysis_skill_status"] = {"status": "confirmed", "source": "user"}
                self.emit_skill_lifecycle_event(state, "confirmed", state["skill_match"], {"confirmed": True})
            return False
        if isinstance(match, SkillMatchState) and match.confirmed:
            return False
        skill_name = match.skill_name
        if not skill_name:
            return False
        state["skill_match"] = match.model_copy(update={"status": "waiting_confirmation", "requires_confirmation": True, "confirmed": False})
        record = SkillLifecycleRecord(
            record_id=self.skill_lifecycle_record_id(skill_name, "confirmation_required"),
            skill_name=skill_name,
            stage="confirmation_required",
            status="waiting_confirmation",
            matched_by=match.matched_by or "skill_match",
            requires_confirmation=True,
            confirmed=False,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            progress=["matched", "waiting_confirmation"],
            summary="等待用户确认是否执行分析技能",
        )
        state.setdefault("skill_lifecycle_records", []).append(record)
        state["agent_run_result"].skill_lifecycle_records.append(record)
        self.persist_confirmation_evidence(state)
        self.emit_skill_lifecycle_event(
            state,
            "confirmation_required",
            state["skill_match"],
            {"status": "waiting_confirmation"},
        )
        published_skills = (
            state.get("planning_asset_pack", PlanningAssetPack()).skills
            if state.get("planning_asset_pack")
            else []
        )
        action_label = merchant_analysis_action_label(skill_name, published_skills)
        self.request_human_clarification(
            state,
            "为了让结论更可靠，建议继续进行“%s”。系统会基于已校验的经营数据做专项拆解，不会修改任何业务数据。是否开始？"
            % action_label,
            "DEEP_ANALYSIS",
            "skill_confirm",
            ["开始深度分析", "先看当前结果"],
        )
        add_step(state, "专项分析确认：命中 %s，等待商家确认是否继续深挖" % action_label)
        return True

    def confirmation_evidence_path(self, state: AgentState, source_run_id: str = "") -> Path:
        run_id = source_run_id or state.get("run_id", "")
        return (
            self.settings.resolved_workspace_path
            / "threads"
            / str(state.get("thread_id") or "thread")
            / "runs"
            / str(run_id or "run")
            / "outputs"
            / "confirmation_evidence.json"
        )

    def persist_confirmation_evidence(self, state: AgentState) -> None:
        token = uuid.uuid4().hex
        source_run_id = str(state.get("run_id") or "")
        original_question = str(state.get("original_question") or state.get("question") or "")
        plan_payload = state["plan"].model_dump(by_alias=True)
        run_result_payload = state["agent_run_result"].model_dump(by_alias=True)
        skill_name = str(getattr(state.get("skill_match"), "skill_name", "") or "")
        semantic_topics = self.confirmation_semantic_topics(state)
        semantic_source_hash = self.recall_service.topic_assets.semantic_source_hash(semantic_topics)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        payload = {
            "version": 3,
            "sourceRunId": source_run_id,
            "nonce": token,
            "expiresAt": expires_at.isoformat(),
            "consumedAt": "",
            "merchantId": state.get("requested_merchant_id", ""),
            "userIdentity": state.get("user_identity") or {},
            "originalQuestion": original_question,
            "questionHash": stable_payload_hash(original_question.strip()),
            "planFingerprint": stable_payload_hash(plan_payload),
            "evidenceFingerprint": stable_payload_hash(run_result_payload),
            "semanticTopics": semantic_topics,
            "semanticSourceHash": semantic_source_hash,
            "skillName": skill_name,
            "routingDecision": state["routing_decision"].model_dump(by_alias=True),
            "topicRoutingDecision": state["topic_routing_decision"].model_dump(by_alias=True),
            "plan": plan_payload,
            "planningAssetPack": state["planning_asset_pack"].model_dump(by_alias=True),
            "queryGraphValidation": state["query_graph_validation_result"].model_dump(by_alias=True),
            "agentRunResult": run_result_payload,
            "skillMatch": state["skill_match"].model_dump(by_alias=True),
            "ruleRecallContext": state.get("rule_recall_context", ""),
            "ruleRecallRefs": state.get("rule_recall_refs") or [],
            "baseKnowledgeContext": state.get("base_knowledge_context", ""),
            "topicAssetContext": state.get("topic_asset_context", ""),
            "alwaysApplyContext": state.get("always_apply_context", ""),
            "latencyOptimization": state.get("latency_optimization") or {},
            "hypothesisExplorationCompleted": bool(state.get("hypothesis_exploration_completed")),
            "hypothesisExplorationRounds": int(state.get("hypothesis_exploration_rounds") or 0),
            "hypothesisSelectedIds": list(state.get("hypothesis_selected_ids") or []),
            "hypothesisResults": list(state.get("hypothesis_results") or []),
            "hypothesisEvidenceLedger": (state.get("hypothesis_evidence_ledger") or HypothesisEvidenceLedger()).model_dump(by_alias=True),
            "savedAt": datetime.now().isoformat(),
        }
        path = self.confirmation_evidence_path(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        state["confirmation_token"] = token
        state["confirmation_source_run_id"] = source_run_id

    def restore_confirmation_evidence(self, state: AgentState) -> bool:
        context = state.get("request_context")
        source_run_id = str(getattr(context, "confirmation_run_id", "") or "") if context else ""
        token = str(getattr(context, "confirmation_token", "") or "") if context else ""
        pending_question = str(getattr(context, "pending_question", "") or "") if context else ""
        if not source_run_id or not token or not pending_question:
            return False
        path = self.confirmation_evidence_path(state, source_run_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            payload = {}
        if (
            not payload
            or int(payload.get("version") or 0) != 3
            or str(payload.get("sourceRunId") or "") != source_run_id
            or str(payload.get("nonce") or "") != token
            or payload.get("consumedAt")
            or str(payload.get("merchantId") or "") != str(state.get("requested_merchant_id") or "")
        ):
            return False
        try:
            expires_at = datetime.fromisoformat(str(payload.get("expiresAt") or ""))
        except ValueError:
            return False
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            return False
        stored_identity = payload.get("userIdentity") or {}
        current_identity = state.get("user_identity") or {}
        stored_user = str(stored_identity.get("userId") or stored_identity.get("user_id") or "")
        current_user = str(current_identity.get("userId") or current_identity.get("user_id") or "")
        if stored_user and current_user and stored_user != current_user:
            return False
        if stable_payload_hash(pending_question.strip()) != str(payload.get("questionHash") or ""):
            return False
        if stable_payload_hash(payload.get("plan") or {}) != str(payload.get("planFingerprint") or ""):
            return False
        if stable_payload_hash(payload.get("agentRunResult") or {}) != str(payload.get("evidenceFingerprint") or ""):
            return False
        semantic_topics = [str(item) for item in payload.get("semanticTopics") or [] if str(item)]
        stored_semantic_hash = str(payload.get("semanticSourceHash") or "")
        current_semantic_hash = self.recall_service.topic_assets.semantic_source_hash(semantic_topics)
        if not semantic_topics or not stored_semantic_hash or current_semantic_hash != stored_semantic_hash:
            state["confirmation_restore_status"] = {
                "status": "stale",
                "code": "CONFIRMATION_SEMANTIC_VERSION_CHANGED",
                "sourceRunId": source_run_id,
                "semanticTopics": semantic_topics,
                "storedSemanticSourceHash": stored_semantic_hash,
                "currentSemanticSourceHash": current_semantic_hash,
                "recommendedAction": "refresh_recall_and_replan",
            }
            state.setdefault("route_decision_trace", []).append(dict(state["confirmation_restore_status"]))
            return False
        run_result = AgentRunResult.model_validate(payload.get("agentRunResult") or {})
        if not run_result.task_results or not run_result.verified_evidence.passed:
            return False
        restored_skill = SkillMatchState.model_validate(payload.get("skillMatch") or {})
        if str(restored_skill.skill_name or "") != str(payload.get("skillName") or ""):
            return False
        consumed_path = path.with_name("confirmation_evidence.consumed.%s.json" % state.get("run_id", "run"))
        try:
            path.replace(consumed_path)
        except OSError:
            return False
        state["routing_decision"] = RoutingDecision.model_validate(payload.get("routingDecision") or {})
        state["topic_routing_decision"] = TopicRoutingDecision.model_validate(payload.get("topicRoutingDecision") or {})
        state["plan"] = QueryPlan.model_validate(payload.get("plan") or {})
        state["planning_asset_pack"] = PlanningAssetPack.model_validate(payload.get("planningAssetPack") or {})
        restored_validation = GraphValidationResult.model_validate(payload.get("queryGraphValidation") or {})
        record_graph_validation(state, restored_validation)
        state["agent_run_result"] = run_result
        state["query_bundle"] = run_result.merged_query_bundle
        state["query_bundles"] = run_result.query_bundles
        state["skill_match"] = restored_skill
        state["rule_recall_context"] = str(payload.get("ruleRecallContext") or "")
        state["rule_recall_refs"] = list(payload.get("ruleRecallRefs") or [])
        state["base_knowledge_context"] = str(payload.get("baseKnowledgeContext") or "")
        state["topic_asset_context"] = str(payload.get("topicAssetContext") or "")
        state["always_apply_context"] = str(payload.get("alwaysApplyContext") or "")
        state["latency_optimization"] = dict(payload.get("latencyOptimization") or {})
        state["hypothesis_exploration_completed"] = bool(payload.get("hypothesisExplorationCompleted"))
        state["hypothesis_exploration_rounds"] = int(payload.get("hypothesisExplorationRounds") or 0)
        state["hypothesis_selected_ids"] = list(payload.get("hypothesisSelectedIds") or [])
        state["hypothesis_results"] = list(payload.get("hypothesisResults") or [])
        state["hypothesis_evidence_ledger"] = HypothesisEvidenceLedger.model_validate(payload.get("hypothesisEvidenceLedger") or {})
        state["question"] = str(payload.get("originalQuestion") or state.get("question") or "")
        state["confirmation_evidence_reused"] = True
        state["confirmation_source_run_id"] = source_run_id
        state["confirmation_token"] = ""
        state["analysis_skill_bypassed"] = skill_confirmation_declined(
            state.get("clarification_resolution") or {}
        )
        state["topic_routed"] = True
        state["fast_understood"] = True
        state["data_discovered"] = True
        state["planning_assets_compacted"] = True
        state["query_graph_reflected"] = True
        state["sql_generated"] = True
        state["result_generation"] = int(state.get("execution_generation") or 0)
        state["sql_repair_reviewed"] = True
        state["evidence_graph_verified"] = True
        state["verification_status"] = "passed"
        state["evidence_accepted"] = True
        state["evidence_generation"] = int(state.get("execution_generation") or 0)
        if state.get("analysis_summary"):
            state["analysis_generation"] = int(state.get("execution_generation") or 0)
        add_step(state, "确认续跑：恢复已验证 QueryGraph、数据结果与专项分析选择")
        return True

    def confirmation_semantic_topics(self, state: AgentState) -> List[str]:
        topics: List[str] = []
        pack = state.get("planning_asset_pack") or PlanningAssetPack()
        for item in pack.tables:
            topic = str(item.topic or "")
            if topic and topic not in topics:
                topics.append(topic)
        decision = state.get("topic_routing_decision") or TopicRoutingDecision()
        for topic in self.recall_service.topic_assets.topic_names_for_categories(decision.recall_topics()):
            if topic and topic not in topics:
                topics.append(topic)
        return sorted(topics)

    def record_skill_lifecycle(self, state: AgentState, trace: Dict[str, Any]) -> None:
        for child_trace in trace.get("skillBatchResults") or []:
            if isinstance(child_trace, dict):
                self.record_skill_lifecycle(state, child_trace)
        if not trace or not trace.get("skillName"):
            return
        stage = str(trace.get("lifecycleStage") or ("completed" if trace.get("activated") and not trace.get("error") else "matched"))
        status = "success" if stage == "completed" and not trace.get("error") else ("failed" if trace.get("error") else stage)
        context_package = trace.get("contextPackage") or {}
        checkpoint_path = str(trace.get("checkpointPath") or "")
        artifact_refs = []
        for path, reason in [
            (checkpoint_path, "skill worker checkpoint"),
            (str(trace.get("contextPackagePath") or ""), "skill worker context package"),
        ]:
            if path:
                artifact_refs.append(artifact_ref_from_path(path, namespace="skill_worker", reason=reason))
        record = SkillLifecycleRecord(
            record_id=self.skill_lifecycle_record_id(str(trace.get("skillName") or ""), stage, str(trace.get("isolatedRunId") or "")),
            skill_name=str(trace.get("skillName") or ""),
            stage=stage,
            status=status,
            matched_by=str(trace.get("matchedBy") or ""),
            requires_confirmation=bool(trace.get("requiresConfirmation")),
            confirmed=bool(trace.get("confirmed", True)),
            isolated_run_id=str(trace.get("isolatedRunId") or ""),
            workspace_path=str(trace.get("workspacePath") or ""),
            checkpoint_path=checkpoint_path,
            progress=[str(item) for item in (trace.get("progress") or [])],
            reuse_candidate=bool(trace.get("reuseCandidate")),
            context_hash=str(context_package.get("contextHash") or trace.get("contextHash") or ""),
            artifact_refs=artifact_refs,
            started_at=str(trace.get("startedAt") or ""),
            completed_at=str(trace.get("completedAt") or (datetime.now().isoformat() if status in {"success", "failed"} else "")),
            duration_ms=int(trace.get("durationMs") or 0),
            summary=str(trace.get("error") or ("skill completed" if status == "success" else stage)),
            metadata={
                "executionMode": trace.get("executionMode"),
                "workerType": trace.get("workerType"),
                "subAgentType": trace.get("subAgentType"),
                "isolatedExecution": trace.get("isolatedExecution"),
                "contextPackagePath": trace.get("contextPackagePath"),
                "contextPackage": trace.get("contextPackage") or {},
                "inputArtifact": trace.get("inputArtifact"),
                "outputArtifact": trace.get("outputArtifact"),
                "metadata": trace.get("metadata") or {},
            },
        )
        existing = {item.isolated_run_id for item in state.get("skill_lifecycle_records", []) if item.isolated_run_id}
        if record.isolated_run_id and record.isolated_run_id in existing:
            return
        state.setdefault("skill_lifecycle_records", []).append(record)
        state["agent_run_result"].skill_lifecycle_records.append(record)
        add_step(state, "AnswerAgent Skill：%s lifecycle=%s status=%s" % (record.skill_name, record.stage, record.status))

    def skill_lifecycle_record_id(self, skill_name: str, stage: str, isolated_run_id: str = "") -> str:
        seed = "%s:%s:%s" % (skill_name or "unknown", stage or "unknown", isolated_run_id or uuid.uuid4().hex)
        return "skill_lifecycle_%s" % uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:16]

    def answer_personalization_context(self, state: AgentState) -> Dict[str, Any]:
        answer_memory_injection = answer_safe_memory_injection(state.get("memory_injection") or {})
        return {
            "merchantProfileContext": state.get("merchant_profile_context", ""),
            "sessionContext": state.get("session_context", ""),
            "memoryContext": state.get("memory_context", ""),
            "runtimeContext": state.get("runtime_context", ""),
            "runtimeInjection": state.get("runtime_injection", {}),
            "memoryInjection": answer_memory_injection,
            "memoryInjectionTrace": state.get("memory_injection_trace", {}),
            "memoryConstraints": state.get("memory_constraints", []),
            "memoryConstraintTrace": state.get("memory_constraint_trace", {}),
            "merchantProfileSummary": state.get("merchant_profile_summary", {}),
            "threadContext": state.get("thread_context", {}),
        }

    def human_in_loop(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "ask_human", "LeadAgent", "HUMAN_IN_LOOP", input_summary=state.get("human_clarification_question", ""))
        increment_round(state)
        prompt = state.get("human_clarification_question") or self.build_scope_clarification_prompt(state)
        state["answer"] = prompt
        state["should_persist"] = False
        state["persisted"] = False
        state["suggestions"] = list(state.get("human_clarification_options") or [])
        if state.get("clarification_tool_message"):
            add_step(state, "ClarificationMiddleware：已拦截 ask_clarification 工具调用，并以 Command(goto=END) 语义暂停当前 run")
        add_step(state, "Human-in-the-loop / ask_human Tool：已用业务问题暂停自动推进，等待商家补充确认")
        self.emit_answer_ready(state)
        self.record_span(state, "action", "ask_human", started, status="gap", error_code="HUMAN_CLARIFICATION_REQUIRED")
        self.finish_run_step(state, step, "gap", output_summary=prompt[:500], error_code="HUMAN_CLARIFICATION_REQUIRED")
        return state

    def cache_answer(self, state: AgentState) -> AgentState:
        if self.run_cancellation_requested(state):
            return self.finish_canceled_run(state)
        if not state.get("answer"):
            state["answer"] = "当前没有足够证据生成回答，请补充更明确的业务范围或稍后重试。"
        if self.run_cancellation_requested(state):
            return self.finish_canceled_run(state)
        started = now_ms()
        step = self.start_run_step(state, "cache_answer", "LeadAgent", "CACHE_ANSWER", input_summary="answerChars=%d" % len(state.get("answer", "")))
        emit(state, "node.started", "CACHE_ANSWER", {})
        fast_response = state.get("fast_metric_response")
        sections = (
            list(getattr(fast_response, "data_sections", []) or [])
            if fast_response is not None
            else self.answer_service.build_sections(state["plan"], state["agent_run_result"])
        )
        pending_scope = identity_scope_payload(
            state.get("user_identity") or {},
            str(state.get("requested_merchant_id") or state["merchant"].merchant_id),
        )
        pending = PendingAnswer(
            id=state["qa_id"],
            question=state["question"],
            answer=state["answer"],
            merchant_id=state["merchant"].merchant_id,
            merchant_name=state["merchant"].merchant_name,
            category_name=(getattr(fast_response, "category_name", "") if fast_response is not None else joined_categories(state["plan"])),
            doris_tables=",".join(
                list(getattr(fast_response, "doris_tables", []) or [])
                if fast_response is not None
                else state["query_bundle"].tables
            ),
            suggested_questions=json.dumps(state.get("suggestions", []), ensure_ascii=False),
            thread_id=str(state.get("thread_id") or ""),
            user_id=str(pending_scope.get("userId") or ""),
            identity_scope_hash=identity_scope_hash(
                state.get("user_identity") or {},
                str(state.get("requested_merchant_id") or state["merchant"].merchant_id),
            ),
            store_ids=list(pending_scope.get("storeIds") or []),
            permissions=list(pending_scope.get("permissions") or []),
            create_time=datetime.now(),
        )
        if state.get("should_persist"):
            self.pending_store.put(pending)
            state["persisted"] = False
            add_step(state, "当前已关闭问答记录立即写入，已缓存待反馈回答")
        else:
            if fast_response is not None and (getattr(fast_response, "debug_trace", {}) or {}).get("definitionOnly"):
                add_step(state, "Fast Metric Definition：口径说明类答案不写入问答记录和长期记忆")
            else:
                add_step(state, "寒暄、无效意图或非持久化答案不写入问答记录")
        if self.run_cancellation_requested(state):
            self.pending_store.remove(state["qa_id"])
            return self.finish_canceled_run(state)
        if state.get("should_persist"):
            state["memory_ingestion_trace"] = {
                **dict(state.get("memory_ingestion_trace") or {}),
                "status": "pending",
                "written": False,
                "commitMode": "post_answer_async",
            }
            add_step(state, "Merchant Memory：已安排回答后异步抽象个人长期记忆")
        self.emit_answer_ready(state)
        state["post_answer_tail_pending"] = bool(state.get("should_persist"))
        state["response_context"] = build_response_context(
            state["question"],
            state["plan"],
            state["merchant"],
            sections,
            state.get("human_clarification_stage", ""),
            state.get("human_clarification_type", ""),
            state.get("human_clarification_options", []),
            pending_question=state.get("clarification_root_question", ""),
        )
        request_context = state.get("request_context")
        if request_context and getattr(request_context, "user_identity", None):
            state["response_context"].user_identity = request_context.user_identity
        if fast_response is not None:
            state["response_context"].category = str(getattr(fast_response, "category_name", "") or joined_categories(state["plan"]))
            state["response_context"].topic = state["response_context"].category
            state["response_context"].topics = list(
                (state.get("topic_routing_decision") or TopicRoutingDecision()).recall_topics()
            )
            state["response_context"].metric_keys = [
                str(item.get("metricKey") or "")
                for item in (getattr(fast_response, "merchant_experience", {}) or {}).get("metricDisclosures", [])
                if item.get("metricKey")
            ]
            fast_debug = getattr(fast_response, "debug_trace", {}) or {}
            time_contract = fast_debug.get("timeWindowContract") if isinstance(fast_debug.get("timeWindowContract"), dict) else {}
            declared_time_column = str(time_contract.get("partitionColumn") or time_contract.get("timeColumn") or "")
            state["response_context"].dimension_keys = (
                [] if fast_debug.get("definitionOnly") or not declared_time_column else [declared_time_column]
            )
            state["response_context"].data_catalog = ",".join(getattr(fast_response, "doris_tables", []) or [])
        self.synchronize_response_topic_context(state)
        emit(state, "node.completed", "CACHE_ANSWER", {"persisted": state["persisted"]})
        self.refresh_context_snapshot(state, "cache_answer")
        self.record_span(state, "action", "cache_answer", started, metadata={"persisted": state.get("persisted")})
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary="persisted=%s" % state.get("persisted"),
            artifact_paths=[str(Path(state["thread_data"].outputs_path) / "trace_replay.json")],
        )
        state = self.finalize_action_contract(state)
        self.write_trace_replay(state, sections)
        if state.get("should_persist") and not self.run_cancellation_requested(state):
            self.publish_thread_summary(state)
        return state

    def schedule_post_answer_tail(self, state: AgentState) -> None:
        if not state.get("post_answer_tail_pending"):
            return
        if self.run_cancellation_requested(state):
            self.discard_canceled_answer_side_effects(state)
            state["post_answer_tail_pending"] = False
            return
        context = copy_context()
        thread = Thread(
            target=context.run,
            args=(self.run_post_answer_tail, state),
            name="merchant-ai-post-answer",
            daemon=True,
        )
        thread.start()

    def commit_memory_before_response(self, state: AgentState) -> None:
        """Commit long-term memory before returning the answer to the caller."""

        try:
            memory_payload = self.update_memory_after_answer(state)
        except Exception as exc:
            state["memory_ingestion_trace"] = {
                "status": "failed",
                "written": False,
                "error": str(exc)[:500],
                "commitMode": "synchronous_before_response",
            }
            self.pending_store.remove(str(state.get("qa_id") or ""))
            raise
        trace = dict(memory_payload.get("memoryIngestionTrace") or {})
        trace["status"] = "success"
        trace["commitMode"] = "synchronous_before_response"
        state["memory_ingestion_trace"] = trace
        curator_trace = trace.get("knowledgeCurator") or {}
        governance_suggestions = self.merchant_knowledge_suggestions_for_response(memory_payload)
        if governance_suggestions:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["knowledgeSuggestions"] = governance_suggestions
            state["merchant_experience"]["knowledgeGovernance"] = {
                "mode": "shared_knowledge_review" if curator_trace.get("authoritative") else "merchant_knowledge_confirmation",
                "status": "pending_review",
                "description": "从本轮对话中提取候选业务知识，商家确认或平台审核后才会生效",
            }
            if curator_trace.get("authoritative"):
                add_step(
                    state,
                    "Knowledge Curator：模型已从用户原话提取 %d 条待确认知识"
                    % int(curator_trace.get("candidateCount") or len(governance_suggestions)),
                )
        add_step(state, "Merchant Memory：回答返回前已可靠提交个人长期记忆 events=%d" % len(memory_payload.get("events") or []))

    def run_post_answer_tail(self, state: AgentState) -> None:
        try:
            if self.run_cancellation_requested(state):
                return
            if state.get("should_persist"):
                try:
                    memory_payload = self.update_memory_after_answer(state)
                    trace = dict(memory_payload.get("memoryIngestionTrace") or {})
                    trace["status"] = "success"
                    trace["commitMode"] = "post_answer_async"
                    state["memory_ingestion_trace"] = trace
                    curator_trace = trace.get("knowledgeCurator") or {}
                    governance_suggestions = self.merchant_knowledge_suggestions_for_response(memory_payload)
                    if governance_suggestions:
                        state["merchant_experience"] = dict(state.get("merchant_experience") or {})
                        state["merchant_experience"]["knowledgeSuggestions"] = governance_suggestions
                        state["merchant_experience"]["knowledgeGovernance"] = {
                            "mode": "shared_knowledge_review" if curator_trace.get("authoritative") else "merchant_knowledge_confirmation",
                            "status": "pending_review",
                            "description": "从本轮对话中提取候选业务知识，商家确认或平台审核后才会生效",
                        }
                        if curator_trace.get("authoritative"):
                            add_step(
                                state,
                                "Knowledge Curator：模型已从用户原话提取 %d 条待确认知识"
                                % int(curator_trace.get("candidateCount") or len(governance_suggestions)),
                            )
                    add_step(state, "Merchant Memory：回答后已异步提交个人长期记忆 events=%d" % len(memory_payload.get("events") or []))
                except Exception as exc:
                    state["memory_ingestion_trace"] = {
                        "status": "failed",
                        "written": False,
                        "error": str(exc)[:500],
                        "commitMode": "post_answer_async",
                    }
                    add_step(state, "Merchant Memory：回答后异步长期记忆提交失败 %s" % str(exc)[:180])
                if self.run_cancellation_requested(state):
                    return
            runtime_profile_summary = self.merchant_profile_summary_service.summarize(
                merchant=state["merchant"],
                memory_injection=state.get("memory_injection") or {},
                memory_constraints=state.get("memory_constraints") or [],
                route_slots=state.get("route_slots", RouteSlots()),
                fast_understanding=state.get("fast_understanding", FastUnderstandingResult()),
            )
            if self.run_cancellation_requested(state):
                return
            state["merchant_profile_summary"] = self.merchant_profile_store.merge_runtime_summary(
                state["merchant"].merchant_id,
                runtime_profile_summary,
            )
            if self.run_cancellation_requested(state):
                return
            # Per-turn metrics, inferred Topics, and anomaly signals are
            # recent-focus/operating context, not reviewed merchant identity.
            # Memory ingestion above already records them as soft, governed
            # observations.  Never self-promote those observations into the
            # stable profile store or they will bias the next Topic route and
            # manufacture an apparent user preference. Stable merchant master
            # profile fields are a separate controlled-data domain (not
            # personal Memory and not shared Knowledge) and require an
            # explicit profile edit.
            state["merchant_profile_summary"]["persistencePolicy"] = "explicit_profile_update_required"
            try:
                draft_payload = self.skill_draft_service.maybe_create_from_state(
                    state,
                    cancel_check=lambda: self.run_cancellation_requested(state),
                )
                if draft_payload:
                    state["skill_draft"] = SkillDraft.model_validate(draft_payload)
                    add_step(state, "Skill Governance：已异步生成待审核 SkillDraft %s" % draft_payload.get("draftId", ""))
            except Exception as exc:
                add_step(state, "Skill Governance：异步 SkillDraft 生成失败 %s" % str(exc)[:180])
            if self.run_cancellation_requested(state):
                return
            if self.run_cancellation_requested(state):
                self.discard_canceled_answer_side_effects(state)
        finally:
            if self.run_cancellation_requested(state):
                self.discard_canceled_answer_side_effects(state)
            state["post_answer_tail_pending"] = False

    def update_memory_after_answer(self, state: AgentState) -> Dict[str, Any]:
        update = self.memory_store.update_from_state
        try:
            signature = inspect.signature(update)
            supports_cancel = "cancel_check" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            supports_cancel = False
        if supports_cancel:
            return update(state, cancel_check=lambda: self.run_cancellation_requested(state))
        return update(state)

    def discard_canceled_answer_side_effects(self, state: AgentState) -> None:
        self.pending_store.remove(str(state.get("qa_id") or ""))
        path = self.thread_summary_path(state)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def write_trace_replay(self, state: AgentState, sections: List[Any]) -> None:
        if not self.settings.agent_trace_replay_enabled:
            return
        try:
            thread_data = state.get("thread_data")
            if not thread_data:
                return
            path = Path(thread_data.outputs_path) / "trace_replay.json"
            payload = {
                "version": "v2",
                "threadId": state.get("thread_id", ""),
                "runId": state.get("run_id", ""),
                "status": "completed" if state.get("chat_bi_completed") else "partial",
                "createdAt": datetime.now().isoformat(),
                "question": state.get("question", ""),
                "answer": state.get("answer", ""),
                "thinkingSteps": state.get("thinking_steps", []),
                "plan": state["plan"].model_dump(by_alias=True),
                "assetPack": self.planning_asset_debug(state["planning_asset_pack"]),
                "actionTimeline": [item.model_dump(by_alias=True) for item in state.get("run_steps", [])],
                "spanTimeline": [item.model_dump(by_alias=True) for item in state.get("trace_spans", [])],
                "performance": performance_summary(state),
                "checkpoint": self.checkpoint_debug(state),
                "artifactManifest": self.artifact_manifest(state),
                "actionHistory": [item.model_dump(by_alias=True) for item in state.get("action_history", [])],
                "actionOutcomes": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("action_outcomes", [])
                ],
                "lastActionResult": action_result_payload(state.get("last_action_result")),
                "contractBlockObservation": state.get("contract_block_observation", {}),
                "actionCatalogContractBlocks": state.get("action_catalog_contract_blocks", []),
                "leadDecisions": [item.model_dump(by_alias=True) for item in state.get("lead_decisions", [])],
                "mainAgentObservations": state.get("main_agent_observations", []),
                "promptManagement": {
                    "templates": self.prompt_assembler.catalog_summary(),
                    "leadPrompt": self.prompt_assembler.lead_prompt_summary(
                        self.policy.registry.public_action_ids(),
                        state.get("loaded_skills", []),
                        self.settings.max_concurrent_sub_agents,
                    ),
                    "plannerRuntime": self.planner_runtime_debug(state),
                },
                "toolCalling": self.tool_calling_debug(state),
                "threadContext": state.get("thread_context", {}),
                "messageHistory": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("message_history", [])
                ],
                "runtimeInjection": state.get("runtime_injection", {}),
                "memoryInjection": state.get("memory_injection", {}),
                "memoryConstraints": state.get("memory_constraints", []),
                "merchantProfileSummary": state.get("merchant_profile_summary", {}),
                "dataFreshness": self.data_freshness_for_response(state, sections),
                "securityAudit": self.security_audit_for_response(state, state.get("query_bundle", QueryBundle()).tables),
                "memory": self.memory_debug(state),
                "contextSnapshots": state.get("context_snapshots", []),
                "contextManifests": state.get("context_manifests", []),
                "contextManagement": self.context_management_debug(state),
                "observability": observability_summary(state),
                "middleware": self.middleware_debug(state),
                "contextLineage": self.context_lineage_debug(state),
                "toolRuntime": self.tool_runtime_debug(state),
                "cache": self.cache_debug(),
                "openDiagnostic": self.open_diagnostic_debug(state),
                "routeSlots": state.get("route_slots", RouteSlots()).model_dump(by_alias=True),
                "routeDecisionTrace": state.get("route_decision_trace", []),
                "boundedRouteLlmTrace": state.get("bounded_route_llm_trace", {}),
                "intentSignals": state.get("intent_signals", IntentSignals()).model_dump(by_alias=True),
                "plannerReflection": state.get("planner_reflection", PlannerReflectionResult()).model_dump(by_alias=True),
                "plannerRepairReason": state.get("planner_repair_reason", ""),
                "plannerRepairRequests": [item.model_dump(by_alias=True) for item in state.get("planner_repair_requests", [])],
                "plannerRepairInput": (state.get("planner_repair_input") or PlannerRepairInput()).model_dump(by_alias=True),
                "queryGraphRepairDelta": (state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()).model_dump(by_alias=True),
                "queryGraphRepairHistory": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("query_graph_repair_history", [])
                ],
                "questionUnderstanding": state["plan"].question_understanding,
                "compilerTrace": state["plan"].compiler_trace,
                "plannerToolCalls": state["plan"].planner_tool_calls,
                "plannerToolResults": state["plan"].planner_tool_results,
                "plannerLoadedRefs": state["plan"].planner_loaded_refs,
                "plannerContextFiles": state["plan"].planner_context_files,
                "metricResolution": metric_resolutions_for_debug(state["plan"]),
                "answerGuard": answer_guard_debug(state["agent_run_result"]),
                "analysisSkill": state.get("analysis_skill_trace", {}),
                "answerFileToolResults": state.get("answer_file_tool_results", {}),
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
                "nodeExecutionBatches": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_execution_batches],
                "nodePlanContracts": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_contracts],
                "nodePlanCritiques": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_critiques],
                "sqlDraftDecisions": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_draft_decisions],
                "freshnessReports": [item.model_dump(by_alias=True) for item in state.get("freshness_reports", [])],
                "validation": state["query_graph_validation_result"].model_dump(by_alias=True),
                "tasks": [item.model_dump(by_alias=True) for item in state["agent_run_result"].task_results],
                "executionAttemptArtifacts": [
                    item.model_dump(by_alias=True)
                    for item in normalized_execution_attempt_artifacts(state)
                ],
                "evidenceGaps": [item.model_dump(by_alias=True) for item in state["agent_run_result"].evidence_gaps],
                "evidenceTimeline": {
                    "verifiedEvidence": state["agent_run_result"].verified_evidence.model_dump(by_alias=True),
                    "partialAnswerReason": state["agent_run_result"].partial_answer_reason,
                },
                "sections": [section.model_dump(by_alias=True) for section in sections],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        except Exception:
            return

    def to_response(self, state: AgentState) -> ChatResponse:
        append_knowledge_retrieval_outcomes_to_run_result(state, state["agent_run_result"])
        append_memory_recall_outcomes_to_run_result(state, state["agent_run_result"])
        append_subagent_outcomes_to_run_result(state, state["agent_run_result"])
        execution_attempts = normalized_execution_attempt_artifacts(state)
        state["execution_attempt_artifacts"] = execution_attempts
        state["agent_run_result"].execution_attempt_artifacts = [
            item.model_copy(deep=True) for item in execution_attempts
        ]
        fast_response = state.get("fast_metric_response")
        if fast_response is not None:
            response = ChatResponse.model_validate(
                fast_response.model_dump(by_alias=True) if hasattr(fast_response, "model_dump") else fast_response
            )
            response.id = state["qa_id"]
            response.answer = state.get("answer", response.answer)
            response.persisted = bool(state.get("persisted"))
            response.context = state.get("response_context")
            response.suggestions = list(state.get("suggestions") or response.suggestions)
            response.thinking_steps = list(state.get("thinking_steps") or response.thinking_steps)
            response.merchant_experience = {
                **dict(response.merchant_experience or {}),
                **dict(state.get("merchant_experience") or {}),
                "analysisScope": state.get("analysis_scope", {}),
            }
            response.debug_trace = {
                **dict(response.debug_trace or {}),
                "leadAgentFastDecision": True,
                "singleMetricOnly": True,
                "boundedLeadLlmTrace": state.get("fast_gate_decision_trace", {}),
                "actionHistory": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("action_history", [])
                ],
                "actionOutcomes": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("action_outcomes", [])
                ],
                "lastActionResult": action_result_payload(state.get("last_action_result")),
                "contractBlockObservation": state.get("contract_block_observation", {}),
                "actionCatalogContractBlocks": state.get("action_catalog_contract_blocks", []),
                "leadDecisions": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("lead_decisions", [])
                ],
                "capabilityDecisions": state.get("capability_decisions", {}),
                "degradedReasons": list(state["agent_run_result"].degraded_reasons or []),
                "evidenceGaps": [item.model_dump(by_alias=True) for item in state["agent_run_result"].evidence_gaps],
            }
            return response
        sections = self.answer_service.build_sections(state["plan"], state["agent_run_result"])
        if state.get("response_context") is None:
            state["response_context"] = build_response_context(
                state["question"],
                state["plan"],
                state["merchant"],
                sections,
                state.get("human_clarification_stage", ""),
                state.get("human_clarification_type", ""),
                state.get("human_clarification_options", []),
                pending_question=state.get("clarification_root_question", ""),
            )
        self.synchronize_response_topic_context(state)
        state["response_context"].confirmation_token = str(state.get("confirmation_token") or "")
        state["response_context"].confirmation_run_id = str(state.get("confirmation_source_run_id") or "")
        clarification = None
        if state.get("human_clarification_required"):
            clarification = ClarificationRequest(
                question=state.get("human_clarification_question", ""),
                stage=state.get("human_clarification_stage", ""),
                type=state.get("human_clarification_type", ""),
                options=state.get("human_clarification_options", []),
                pending_question=state.get("clarification_root_question") or state.get("question", ""),
            )
        data_rows: List[Dict[str, Any]] = []
        tables: List[str] = []
        for section in sections:
            data_rows.extend(section.data_rows)
            for table in section.doris_tables:
                if table not in tables:
                    tables.append(table)
        if not sections:
            data_rows = state["query_bundle"].rows
            tables = state["query_bundle"].tables
        if not tables:
            for task_result in (state.get("agent_run_result") or AgentRunResult()).task_results:
                if task_result.query_bundle.failed:
                    continue
                for table in task_result.query_bundle.tables:
                    if table and table not in tables:
                        tables.append(table)
        if not state.get("merchant_experience"):
            state["merchant_experience"] = self.answer_service.merchant_experience(
                state["question"],
                state["plan"],
                state.get("agent_run_result"),
                merchant=state.get("merchant"),
                sections=sections,
                suggestions=state.get("suggestions", []),
                personalization_context=self.answer_personalization_context(state),
            )
        state["merchant_experience"] = dict(state.get("merchant_experience") or {})
        state["merchant_experience"]["analysisScope"] = state.get("analysis_scope", {})
        governance = self.knowledge_governance_debug_payload(state)
        if governance:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"].setdefault("knowledgeGovernance", governance)
        applied_constraints = self.applied_memory_constraints_for_response(state)
        if applied_constraints:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["appliedMemoryConstraints"] = applied_constraints
        profile_summary = state.get("merchant_profile_summary") or {}
        if profile_summary:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["merchantProfileSummary"] = profile_summary
        data_freshness = self.data_freshness_for_response(state, sections)
        if data_freshness:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["dataFreshness"] = data_freshness
        security_audit = self.security_audit_for_response(state, tables)
        if security_audit:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["securityAudit"] = security_audit
        controlled_react = self.controlled_react_for_response(state)
        if controlled_react:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["controlledReact"] = controlled_react
        human_loop = self.human_loop_for_response(state)
        if human_loop:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["humanLoop"] = human_loop
        skill_ecosystem = self.skill_ecosystem_for_response(state)
        if skill_ecosystem:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["skillEcosystem"] = skill_ecosystem
        downloads = self.result_download_artifacts(state, sections)
        if downloads:
            state["merchant_experience"] = dict(state.get("merchant_experience") or {})
            state["merchant_experience"]["downloadArtifacts"] = downloads
        return ChatResponse(
            id=state["qa_id"],
            answer=state.get("answer", ""),
            category_name=joined_categories(state["plan"]),
            persisted=bool(state.get("persisted")),
            doris_tables=tables,
            suggestions=state.get("suggestions", []),
            thinking_steps=state.get("thinking_steps", []),
            data_rows=data_rows,
            data_sections=sections,
            context=state["response_context"],
            clarification=clarification,
            merchant_experience=state.get("merchant_experience", {}),
            debug_trace={
                "displayPolicy": state["plan"].display_policy,
                "displayTitle": state["plan"].display_title,
                "agentTrace": state["plan"].agent_trace or legacy_agent_trace_from_actions(state),
                "harness": {
                    "mode": self.settings.agent_mode,
                    "actions": self.policy.registry.public_action_ids(),
                    "availableActions": [item.model_dump(by_alias=True) for item in state.get("available_actions", [])],
                    "actionHistory": [item.model_dump(by_alias=True) for item in state.get("action_history", [])],
                    "actionOutcomes": [
                        item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                        for item in state.get("action_outcomes", [])
                    ],
                    "lastActionResult": action_result_payload(state.get("last_action_result")),
                    "contractBlockObservation": state.get("contract_block_observation", {}),
                    "actionCatalogContractBlocks": state.get("action_catalog_contract_blocks", []),
                    "leadDecisions": [item.model_dump(by_alias=True) for item in state.get("lead_decisions", [])],
                    "mainAgentObservations": state.get("main_agent_observations", []),
                    "decisionReason": state.get("agent_decision_reason", ""),
                    "performance": performance_summary(state),
                    "checkpoint": self.checkpoint_debug(state),
                    "traceReplay": self.trace_replay_debug(state),
                    "llmLastError": self.planner.llm.last_error or self.node_worker.llm.last_error,
                    "llmErrors": (self.planner.llm.error_events or [])[-12:],
                    "loadedSkills": state.get("loaded_skills", []),
                    "promptManagement": {
                        "templates": self.prompt_assembler.catalog_summary(),
                        "leadPrompt": self.prompt_assembler.lead_prompt_summary(
                            self.policy.registry.public_action_ids(),
                            state.get("loaded_skills", []),
                            self.settings.max_concurrent_sub_agents,
                        ),
                        "plannerRuntime": self.planner_runtime_debug(state),
                    },
                    "toolCalling": self.tool_calling_debug(state),
                    "threadContext": state.get("thread_context", {}),
                    "contextManagement": self.context_management_debug(state),
                    "observability": observability_summary(state),
                    "middleware": self.middleware_debug(state),
                    "contextLineage": self.context_lineage_debug(state),
                    "toolRuntime": self.tool_runtime_debug(state),
                    "cache": self.cache_debug(),
                    "runtimeInjection": state.get("runtime_injection", {}),
                    "memoryInjection": state.get("memory_injection", {}),
                    "memory": self.memory_debug(state),
                    "openDiagnostic": self.open_diagnostic_debug(state),
                    "routeSlots": state.get("route_slots", RouteSlots()).model_dump(by_alias=True),
                    "routeDecisionTrace": state.get("route_decision_trace", []),
                    "boundedRouteLlmTrace": state.get("bounded_route_llm_trace", {}),
                    "fastUnderstanding": (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
                    "capabilityRegistry": {
                        "version": self.policy.capabilities.version,
                        "source": self.policy.capabilities.source,
                        "decisions": state.get("capability_decisions", {}),
                    },
                    "boundedLeadLlmTrace": state.get("bounded_lead_llm_trace", {}),
                    "fastGateDecisionTrace": state.get("fast_gate_decision_trace", {}),
                    "intentSignals": state.get("intent_signals", IntentSignals()).model_dump(by_alias=True),
                    "knowledgeRetrieval": {
                        "backend": (state.get("knowledge_bundle") or KnowledgeBundle()).backend,
                        "status": state.get("knowledge_retrieval_status") or (state.get("knowledge_bundle") or KnowledgeBundle()).retrieval_status,
                        "sourceRefs": (state.get("knowledge_bundle") or KnowledgeBundle()).source_refs,
                        "rounds": state.get("recall_rounds", []),
                        "requestLineage": state.get("knowledge_request_lineage", {}),
                        "issues": state.get("knowledge_retrieval_issues", []),
                        "outcomes": state.get("knowledge_retrieval_outcomes", []),
                    },
                },
                "plannerReflection": state.get("planner_reflection", PlannerReflectionResult()).model_dump(by_alias=True),
                "plannerRepairReason": state.get("planner_repair_reason", ""),
                "plannerRepairRequests": [item.model_dump(by_alias=True) for item in state.get("planner_repair_requests", [])],
                "plannerRepairInput": (state.get("planner_repair_input") or PlannerRepairInput()).model_dump(by_alias=True),
                "queryGraphRepairDelta": (state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()).model_dump(by_alias=True),
                "queryGraphRepairHistory": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("query_graph_repair_history", [])
                ],
                "questionUnderstanding": state["plan"].question_understanding,
                "compilerTrace": state["plan"].compiler_trace,
                "plannerToolCalls": state["plan"].planner_tool_calls,
                "plannerToolResults": state["plan"].planner_tool_results,
                "plannerLoadedRefs": state["plan"].planner_loaded_refs,
                "plannerContextFiles": state["plan"].planner_context_files,
                "metricResolution": metric_resolutions_for_debug(state["plan"]),
                "answerGuard": answer_guard_debug(state["agent_run_result"]),
                "analysisSkill": state.get("analysis_skill_trace", {}),
                "answerFileToolResults": state.get("answer_file_tool_results", {}),
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
                "nodePlanContracts": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_contracts],
                "nodePlanCritiques": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_critiques],
                "sqlDraftDecisions": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_draft_decisions],
                "freshnessReports": [item.model_dump(by_alias=True) for item in state.get("freshness_reports", [])],
                "merchantProfileSummary": state.get("merchant_profile_summary", {}),
                "dataFreshness": self.data_freshness_for_response(state, sections),
                "securityAudit": self.security_audit_for_response(state, tables),
                "controlledReact": self.controlled_react_for_response(state),
                "humanLoop": self.human_loop_for_response(state),
                "skillEcosystem": self.skill_ecosystem_for_response(state),
                "planningAssetPack": self.planning_asset_debug(state["planning_asset_pack"]),
                "queryGraphValidation": state["query_graph_validation_result"].model_dump(by_alias=True),
                "pendingKnowledgeRequests": [item.model_dump(by_alias=True) for item in state.get("pending_knowledge_requests", [])],
                "knowledgeRequestGaps": state.get("knowledge_request_gaps", []),
                "blockedKnowledgeRequestKeys": state.get("blocked_knowledge_request_keys", []),
                "agentBudgets": {
                    "reactRound": state.get("react_round", 0),
                    "retrieveCount": state.get("query_graph_retrieve_count", 0),
                    "planAttempts": state.get("query_graph_plan_attempts", 0),
                    "graphRepairAttempts": state.get("query_graph_repair_attempts", 0),
                },
                "planIntents": [intent.model_dump(by_alias=True) for intent in state["plan"].intents],
                "dependencies": [dep.model_dump(by_alias=True) for dep in state["plan"].dependencies],
                "taskResults": [item.model_dump(by_alias=True) for item in state["agent_run_result"].task_results],
                "executionAttemptArtifacts": [item.model_dump(by_alias=True) for item in execution_attempts],
                "evidenceGaps": [item.model_dump(by_alias=True) for item in state["agent_run_result"].evidence_gaps],
                "degradedReasons": list(state["agent_run_result"].degraded_reasons or []),
                "sqlRepairs": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_repairs],
                "verifiedEvidence": state["agent_run_result"].verified_evidence.model_dump(by_alias=True),
                "partialAnswerReason": state["agent_run_result"].partial_answer_reason,
            },
        )

    def merchant_knowledge_suggestions_for_response(self, memory_payload: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        trace = memory_payload.get("memoryIngestionTrace") or {}
        target_id = str(trace.get("knowledgeSuggestionId") or "")
        suggestions = [item for item in (memory_payload.get("knowledgeSuggestions") or []) if isinstance(item, dict)]
        if target_id:
            suggestions = [item for item in suggestions if str(item.get("suggestionId") or "") == target_id] + [
                item for item in suggestions if str(item.get("suggestionId") or "") != target_id
            ]
        result: List[Dict[str, Any]] = []
        for item in suggestions:
            if str(item.get("suggestionType") or "") in {"personal_preference", "merchant_preference"}:
                # New personal preferences are auto-written to Merchant
                # Memory. Legacy misclassified proposals must not surface as a
                # Knowledge review card.
                continue
            status = str(item.get("status") or "candidate")
            if status in {"merchant_active", "platform_suggested", "dismissed", "rejected", "published", "indexed"}:
                continue
            if status not in {"candidate", "review_required", "pending", "reviewed"} and str(item.get("suggestionId") or "") != target_id:
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            proposed_scope = str(item.get("scopeType") or payload.get("proposedScope") or "").lower()
            if str(payload.get("memoryType") or "") == "metric_dispute":
                proposed_scope = "platform"
            if proposed_scope not in {"merchant", "platform"}:
                # Old candidates without an explicit scope must be classified
                # before either merchant activation or shared publication.
                continue
            correction_text = str(payload.get("correctionText") or payload.get("question") or "")[:240]
            title = str(payload.get("title") or item.get("metricName") or "业务知识")
            if proposed_scope == "platform":
                prompt = "这条内容涉及平台指标或公共口径，提交后需要平台审核。"
                user_actions = [
                    {"actionId": "submit_feedback", "label": "提交反馈", "style": "primary"},
                    {"actionId": "dismiss", "label": "取消", "style": "secondary"},
                ]
                notice_type = "platform_feedback"
            else:
                prompt = "是否将这条规则用于本商家后续分析？"
                user_actions = [
                    {"actionId": "confirm_use", "label": "确认使用", "style": "primary"},
                    {"actionId": "dismiss", "label": "暂不使用", "style": "secondary"},
                ]
                notice_type = "merchant_rule_confirmation"
            result.append(
                {
                    "suggestionId": str(item.get("suggestionId") or ""),
                    "noticeType": notice_type,
                    "status": status,
                    "topic": str(item.get("topic") or ""),
                    "metricName": str(item.get("metricName") or ""),
                    "title": title,
                    "message": prompt,
                    "ruleText": correction_text,
                    "evidenceQuote": str(payload.get("evidenceQuote") or "")[:240],
                    "userActions": user_actions,
                }
            )
            if len(result) >= limit:
                break
        return result

    def knowledge_governance_debug_payload(self, state: AgentState) -> Dict[str, Any]:
        trace = state.get("memory_ingestion_trace") or {}
        suggestion_id = str(trace.get("knowledgeSuggestionId") or "")
        if not suggestion_id:
            return {}
        return {
            "mode": "merchant_knowledge_confirmation",
            "status": "pending_review" if trace.get("knowledgeSuggestionWritten") else "existing_candidate",
            "suggestionId": suggestion_id,
            "description": "候选规则需要审核后发布到商家知识库或语义资产",
        }

    def memory_governance_debug_payload(self, state: AgentState) -> Dict[str, Any]:
        """Backward-compatible alias for pre-split trace consumers."""

        return self.knowledge_governance_debug_payload(state)

    def applied_memory_constraints_for_response(self, state: AgentState, limit: int = 4) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for item in state.get("memory_constraints") or []:
            enforcement = str(item.get("enforcement") or "")
            if enforcement not in {"required", "clarify_or_disclose"}:
                continue
            result.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": str(item.get("type") or ""),
                    "enforcement": enforcement,
                    "instruction": str(item.get("instruction") or "")[:240],
                    "targetMetrics": [str(value) for value in (item.get("targetMetrics") or [])[:8]],
                    "source": str(item.get("source") or ""),
                    "approvedBy": str(item.get("approvedBy") or ""),
                }
            )
            if len(result) >= limit:
                break
        return result

    def data_freshness_for_response(self, state: AgentState, sections: List[Any]) -> Dict[str, Any]:
        reports = state.get("freshness_reports") or []
        report_payloads = [item.model_dump(by_alias=True) for item in reports[:12]]
        run_result = state.get("agent_run_result") or AgentRunResult()
        alignment = run_result.snapshot_alignment
        alignment_applicable = bool(
            alignment.sources
            or alignment.common_anchor_time_value
            or alignment.disclosure_required
            or str(alignment.status or "").upper() not in {"", "NOT_APPLICABLE"}
        )
        alignment_incomplete = bool(alignment_applicable and not (alignment.aligned and alignment.complete))
        report_statuses = {str(item.get("status") or "").upper() for item in report_payloads}
        gap_codes = {
            str(gap.gap_code or gap.code or "").upper()
            for gap in run_result.evidence_gaps
        }
        degraded = alignment_incomplete or any(
            code in gap_codes for code in {"ZERO_ROWS", "PARTIAL_EVIDENCE", "RESOURCE_DEGRADED_QUERY"}
        )
        fallback_reports = [item for item in report_payloads if item.get("fallbackTable")]
        realtime_fallback_used = "STALE_USE_REALTIME_FALLBACK" in report_statuses or bool(fallback_reports)
        offline_stale = realtime_fallback_used or any("STALE" in status for status in report_statuses)
        unchecked = bool(sections) and not reports
        if realtime_fallback_used:
            status = "realtime_fallback_used"
        elif offline_stale:
            status = "offline_stale"
        elif alignment_incomplete:
            status = "aligned_partial_coverage" if alignment.aligned else "alignment_incomplete"
        elif degraded:
            status = "partial_or_degraded"
        elif any(str(item.get("status") or "").lower() in {"stale", "fallback", "partial"} for item in report_payloads):
            status = "checked_with_warning"
        elif reports:
            status = "checked"
        elif unchecked:
            status = "not_checked"
        else:
            status = "no_query_data"
        tables: List[str] = []
        for section in sections or []:
            for table in getattr(section, "doris_tables", []) or []:
                if table and table not in tables:
                    tables.append(table)
        notes: List[str] = []
        if unchecked:
            notes.append("当前回答有查询结果，但未生成分区新鲜度检查报告")
        if offline_stale:
            notes.append("离线表分区可能滞后，不能把缺失数据直接解释为业务为 0")
        if realtime_fallback_used:
            notes.append("已识别离线表延迟并切换到实时/近实时 fallback 表")
        if degraded:
            notes.append("存在空结果、部分证据或降级查询，答案需要披露数据覆盖风险")
        if alignment_incomplete:
            notes.append("部分来源未覆盖统一时间窗口，相关缺失不可解释为业务为 0")
        common_anchor = str(alignment.common_anchor_time_value or "") if alignment_applicable else ""
        latest_data_at = common_anchor
        if not latest_data_at:
            for item in report_payloads:
                latest_data_at = max(latest_data_at, str(item.get("maxPt") or item.get("max_pt") or ""))
        source_cutoffs = [
            {
                "taskId": source.task_id,
                "table": source.table,
                "sourceLatestDataAt": source.source_max_time_value,
                "effectiveStartTimeValue": source.effective_start_time_value,
                "effectiveEndTimeValue": source.effective_end_time_value,
                "aggregationPolicy": source.aggregation_policy,
                "timeSelectionPolicy": source.time_selection_policy,
                "coverageComplete": source.coverage_complete,
                "compatible": source.compatible,
            }
            for source in alignment.sources[:12]
        ]
        return {
            "status": status,
            "checked": bool(reports),
            "tables": tables[:12],
            "reports": report_payloads,
            "latestDataAt": latest_data_at,
            "commonAnchorTimeValue": common_anchor,
            "sourceCutoffs": source_cutoffs,
            "snapshotAlignment": alignment.model_dump(by_alias=True) if alignment_applicable else {},
            "offlineDelayDetected": offline_stale,
            "realtimeFallbackUsed": realtime_fallback_used,
            "fallbackSuggested": degraded or offline_stale or any(item.get("fallbackTable") for item in report_payloads),
            "missingDataPolicy": "missing partition or zero rows must be disclosed as data coverage risk, not treated as business zero",
            "answerDisclosure": {
                "required": degraded or offline_stale or unchecked,
                "message": "数据可能存在离线延迟或覆盖不足，回答需披露数据更新时间和缺失风险" if (degraded or offline_stale or unchecked) else "",
            },
            "notes": notes,
        }

    def security_audit_for_response(self, state: AgentState, tables: List[str]) -> Dict[str, Any]:
        merchant = state.get("merchant")
        merchant_id = getattr(merchant, "merchant_id", "") or state.get("requested_merchant_id", "")
        allowed_merchants = sorted(self.settings.allowed_merchants)
        contracts = (state.get("agent_run_result") or AgentRunResult()).node_plan_contracts
        masked_columns = sorted(
            {
                column
                for contract in contracts
                for column in (contract.masked_columns or {}).keys()
                if str(column or "").strip()
            }
        )
        restricted_columns = sorted(
            {
                column
                for contract in contracts
                for column, policy in (contract.column_display_policy or {}).items()
                if isinstance(policy, dict) and str(policy.get("level") or "").lower() in {"restricted", "hidden", "internal"}
            }
        )
        audit_summary = self.node_worker.access_control.audit_summary(limit=5) if hasattr(self.node_worker, "access_control") else {}
        tenant_filter_columns = sorted(
            {
                str(contract.merchant_filter_column)
                for contract in contracts
                if str(contract.merchant_filter_column or "").strip()
            }
        )
        return {
            "policy": "readonly_merchant_scoped",
            "merchantId": merchant_id,
            "requestedMerchantId": state.get("requested_merchant_id", ""),
            "tenantScoped": bool(merchant_id),
            "allowedMerchantIdsConfigured": bool(getattr(self.settings, "allowed_merchant_ids", "")),
            "allowedMerchantCount": len(allowed_merchants),
            "readOnly": True,
            "rowLevelSecurity": {
                "enabled": bool(merchant_id),
                "filterColumns": tenant_filter_columns,
                "source": "validated node plan contracts",
            },
            "tableAccess": {
                "checked": True,
                "tableCount": len(tables or []),
                "tables": tables[:12],
            },
            "tables": tables[:12],
            "columnMasking": {
                "enabled": bool(masked_columns),
                "maskedColumns": masked_columns[:20],
            },
            "columnPolicy": {
                "restrictedColumnCount": len(restricted_columns),
                "restrictedColumns": restricted_columns[:20],
            },
            "sqlPolicy": {
                "readOnlyOnly": True,
                "crossTenantGuard": bool(merchant_id),
                "injectionGuard": "sql validator and semantic contract enforce read-only scoped SQL",
            },
            "auditLog": {
                "enabled": True,
                "path": audit_summary.get("auditPath", ""),
                "recentCount": len(audit_summary.get("items") or []),
                "recent": audit_summary.get("items", [])[:3],
            },
            "auditSource": "workflow_response",
        }

    def controlled_react_for_response(self, state: AgentState) -> Dict[str, Any]:
        action_history = state.get("action_history") or []
        lead_decisions = state.get("lead_decisions") or []
        repair_requests = state.get("planner_repair_requests") or []
        gaps = (state.get("agent_run_result") or AgentRunResult()).evidence_gaps
        state["strategy_switch_trace"] = []
        return {
            "mode": "lead_react_querygraph",
            "tradeoff": "Lead selects tools dynamically; Runtime enforces semantic, SQL and evidence contracts",
            "exploration": state.get("hypothesis_exploration") or {},
            "parallelHypothesisResults": state.get("hypothesis_results") or [],
            "candidateQueryGraphs": state.get("candidate_query_graphs") or {},
            "hypothesisEvidenceLedger": (state.get("hypothesis_evidence_ledger") or HypothesisEvidenceLedger()).model_dump(by_alias=True),
            "executionTierPolicy": state.get("execution_tier_policy") or {},
            "latencyOptimization": self.latency_optimizer.response_payload(state.get("latency_optimization") or {}),
            "steps": {
                "leadDecisions": len(lead_decisions),
                "actions": len(action_history),
                "plannerRepairs": len(state.get("query_graph_repair_history") or []),
                "pendingPlannerRepairRequests": len(repair_requests),
                "toolCalls": len(state.get("tool_call_ledger") or []),
                "evidenceGaps": len(gaps),
            },
            "strategySwitches": [],
            "guardrails": ["semantic_asset_contract", "query_graph_validation", "readonly_sql", "evidence_verification"],
        }

    def human_loop_for_response(self, state: AgentState) -> Dict[str, Any]:
        if not state.get("human_clarification_required") and not state.get("clarification_resolution"):
            return {}
        checkpoint = self.checkpoint_debug(state)
        confirmation_type = state.get("human_clarification_type", "")
        return {
            "status": "waiting_confirmation" if state.get("human_clarification_required") else "resolved",
            "confirmationCard": {
                "type": confirmation_type,
                "title": self.human_loop_card_title(confirmation_type),
                "question": state.get("human_clarification_question", ""),
                "options": state.get("human_clarification_options", []),
                "confirmationToken": state.get("confirmation_token", ""),
                "confirmationRunId": state.get("confirmation_source_run_id", ""),
            },
            "checkpoint": checkpoint,
            "resumePolicy": "consume a run-scoped, single-use confirmation token in a new run",
            "resolution": state.get("clarification_resolution") or {},
            "knowledgeFeedback": {
                "candidateSuggestionEnabled": True,
                "reviewRequiredBeforePublish": True,
                "description": "personal preferences persist through the automatic Memory gate; reusable rules and definitions become separate Knowledge proposals before publish",
            },
        }

    def human_loop_card_title(self, confirmation_type: str) -> str:
        mapping = {
            "time_window": "确认分析时间范围",
            "metric_focus": "确认指标口径",
            "planner_clarification": "确认查询口径",
            "priority_goal": "确认优化目标",
            "skill_confirmation": "是否开始深度分析",
            "skill_confirm": "是否开始深度分析",
            "business_scope": "确认业务范围",
            "write_operation_blocked": "确认危险操作",
        }
        return mapping.get(str(confirmation_type or ""), "确认分析口径")

    def skill_ecosystem_for_response(self, state: AgentState) -> Dict[str, Any]:
        draft = state.get("skill_draft")
        draft_payload = draft.model_dump(by_alias=True) if hasattr(draft, "model_dump") else (dict(draft) if isinstance(draft, dict) else {})
        if draft_payload and not draft_payload.get("draftId"):
            draft_payload = {}
        lifecycle_records = [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else dict(item)
            for item in (state.get("skill_lifecycle_records") or [])
        ][:20]
        skills = state.get("planning_asset_pack", PlanningAssetPack()).skills if state.get("planning_asset_pack") else []
        market_items: List[Dict[str, Any]] = []
        for skill in skills[:12]:
            market_items.append(
                {
                    "name": getattr(skill, "domain", "") or getattr(skill, "display_name", ""),
                    "displayName": getattr(skill, "display_name", "") or getattr(skill, "domain", ""),
                    "metrics": list(getattr(skill, "metrics", []) or [])[:8],
                    "tables": list(getattr(skill, "tables", []) or [])[:8],
                    "sourcePath": getattr(skill, "source_path", ""),
                    "version": "runtime_manifest",
                    "reuseScope": "merchant_or_industry_template",
                }
            )
        return {
            "mode": "governed_skill_ecosystem",
            "creator": {
                "enabled": True,
                "source": "verified free exploration or reusable SOP",
                "draft": draft_payload,
                "draftStatus": str(draft_payload.get("status") or "none"),
            },
            "review": {
                "required": True,
                "service": "SkillDraftService",
                "approvedDraftsBecomeCallable": True,
            },
            "market": {
                "enabled": True,
                "items": market_items[:15],
                "reuseScopes": ["merchant", "industry", "global_sop"],
            },
            "versioning": {
                "enabled": True,
                "policy": "published Skill gets immutable folder/version metadata; draft remains non-callable until review",
            },
            "runtimeRecords": lifecycle_records,
            "fromExploration": {
                "enabled": True,
                "eligibility": "complex run with verified evidence and no blocking evidence gaps",
            },
        }

    def result_download_artifacts(self, state: AgentState, sections: List[ChatDataSection]) -> List[Dict[str, Any]]:
        run_result = state.get("agent_run_result")
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not run_result or not outputs_path:
            return []
        min_rows = max(1, int(getattr(self.settings, "result_csv_download_min_rows", 50) or 50))
        intent_map = intent_by_task_id(state.get("plan") or QueryPlan())
        section_by_role = {section.result_role: section for section in sections}
        downloads: List[Dict[str, Any]] = []
        root = Path(outputs_path) / "artifacts" / "downloads"
        root.mkdir(parents=True, exist_ok=True)
        for task in visible_successful_tasks(state.get("plan") or QueryPlan(), run_result):
            bundle = task.query_bundle
            rows, complete = query_bundle_complete_rows(bundle)
            if bundle.failed or not complete or len(rows) < min_rows:
                continue
            intent = intent_map.get(task.task_id)
            title = section_title_for_intent(state.get("plan") or QueryPlan(), intent, task.task_id) if intent else task.task_id
            filename = "%s.csv" % sanitize_download_name(task.task_id or title or "query_result")
            target = root / filename
            write_rows_csv(target, rows)
            relative = str(target.relative_to(Path(outputs_path) / "artifacts"))
            merchant_uri = merchant_uri_for_artifact(relative, namespace="downloads")
            section = section_by_role.get(answer_result_role(intent)) if intent else None
            downloads.append(
                {
                    "type": "csv",
                    "label": "下载%sCSV" % (title or "查询结果"),
                    "taskId": task.task_id,
                    "title": title,
                    "rowCount": len(rows),
                    "previewRowCount": len(section.data_rows) if section else min(len(rows), self.settings.tool_result_preview_rows),
                    "path": str(target),
                    "relativePath": relative,
                    "merchantUri": merchant_uri,
                    "downloadUrl": merchant_uri,
                }
            )
        return downloads[:12]

    def planning_asset_debug(self, pack: PlanningAssetPack) -> Dict[str, Any]:
        return {
            "tables": [item.table for item in pack.tables[:20]],
            "metrics": self.planning_metric_keys_for_context(pack)[:40],
            "relationships": [item.relationship_id for item in pack.relationships[:30]],
            "skills": [skill.domain for skill in pack.skills],
            "schemaSource": pack.schema_source,
            "missingLiveColumns": pack.missing_live_columns,
            "relationshipClosure": pack.relationship_closure,
            "metricCompaction": pack.metric_compaction,
            "skillSemanticGaps": pack.skill_semantic_gaps,
            "semanticCatalogVersion": {
                key: value.model_dump(by_alias=True) for key, value in pack.semantic_catalog_version.items()
            },
            "schemaDriftReports": [item.model_dump(by_alias=True) for item in pack.schema_drift_reports],
            "semanticWorkspace": semantic_workspace_manifest_from_asset_pack(pack, limit=12),
            "semanticFileContext": self.semantic_catalog.context_manifest(
                pack.source_refs,
                allowed_tables=pack.known_tables(),
                allowed_relationship_topics=relationship_topics_from_pack(pack),
            ),
        }

    def planning_metric_keys_for_context(self, pack: PlanningAssetPack) -> List[str]:
        keys: List[str] = []
        for metric in pack.metrics:
            if metric.key and metric.key not in keys:
                keys.append(metric.key)
        for item in (pack.metric_compaction or {}).get("recalledMetricEvidence") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("metricKey") or "")
            if key and key not in keys:
                keys.append(key)
        return keys

    def context_lineage_debug(self, state: AgentState) -> Dict[str, Any]:
        pack = state.get("planning_asset_pack") or PlanningAssetPack()
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for ref_id, item in list(pack.source_refs.items())[:40]:
            metadata = item.metadata or {}
            semantic_ref_id = str(metadata.get("semanticRefId") or ref_id or item.doc_id or "")
            semantic_path = str(metadata.get("semanticPath") or "")
            if semantic_ref_id:
                source = add_context_uri(
                    {
                        "refId": semantic_ref_id,
                        "path": semantic_path,
                        "kind": metadata.get("semanticKind") or item.source_type,
                        "topic": item.topic,
                        "table": item.table,
                        "title": item.title,
                    },
                    ref_id=semantic_ref_id,
                    topic=item.topic,
                    table=item.table,
                    kind=str(metadata.get("semanticKind") or item.source_type),
                    path=semantic_path,
                )
                key = source.get("merchantUri") or semantic_ref_id
                if key not in seen:
                    seen.add(key)
                    records.append(context_lineage_record("knowledge_retrieval", source, "load_source_ref"))
        plan = state.get("plan") or QueryPlan()
        for intent in plan.intents[:40]:
            resolution = intent.metric_resolution or {}
            fallback_ref_id = intent.knowledge_ref_ids[0] if intent.knowledge_ref_ids else ""
            semantic_ref_id = str(resolution.get("semanticRefId") or fallback_ref_id or "")
            if semantic_ref_id:
                source = {
                    "merchantUri": merchant_uri_for_semantic_ref(
                        semantic_ref_id,
                        topic=str(resolution.get("topic") or ""),
                        table=intent.preferred_table,
                        kind="METRIC",
                        key=intent.metric_name or intent.metric_column,
                    ),
                    "refId": semantic_ref_id,
                    "path": "",
                    "contextLayer": "L1",
                    "kind": "METRIC",
                    "title": intent.metric_name or intent.metric_column or intent.plan_task_id,
                }
                key = source["merchantUri"]
                if key not in seen:
                    seen.add(key)
                    records.append(context_lineage_record("metric_resolution", source, "bind_metric"))
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        for artifact in self.artifact_manifest(state)[:40]:
            uri = artifact.get("merchantUri") or merchant_uri_for_artifact(artifact.get("relativePath") or artifact.get("path") or "", namespace=artifact.get("namespace") or "")
            source = {
                "merchantUri": uri,
                "path": artifact.get("relativePath") or artifact.get("path") or "",
                "contextLayer": "L2",
                "kind": artifact.get("namespace") or "artifact",
                "title": artifact.get("title") or artifact.get("relativePath") or "",
            }
            key = uri or source["path"]
            if key and key not in seen:
                seen.add(key)
                records.append(context_lineage_record("artifact", source, "offload"))
        return {
            "uriScheme": "merchant://",
            "records": records[:80],
            "recordCount": len(records),
            "workspace": outputs_path,
        }

    def artifact_manifest(self, state: AgentState) -> List[Dict[str, Any]]:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return []
        root = Path(outputs_path)
        paths = [
            root / "trace_replay.json",
            root / "context_snapshot.json",
            root / "workspace_manifest.json",
            root / "artifacts" / "planner" / "planning_asset_pack.json",
            root / "artifacts" / "planner" / "query_graph.json",
            root / "artifacts" / "node" / "agent_run_result.json",
        ]
        manifest = []
        for path in paths:
            if not path.exists() and path.name != "trace_replay.json":
                continue
            manifest.append(artifact_ref_from_path(str(path), namespace="trace", reason="trace replay v2 artifact").model_dump(by_alias=True))
        for package in state.get("context_packages", [])[-6:]:
            if hasattr(package, "artifact_refs"):
                manifest.extend([item.model_dump(by_alias=True) for item in package.artifact_refs[:4]])
            elif isinstance(package, dict):
                manifest.extend(package.get("artifactRefs") or [])
        deduped: Dict[str, Dict[str, Any]] = {}
        for item in manifest:
            key = str(item.get("path") or item.get("relativePath") or item.get("title") or "")
            if key:
                deduped[key] = item
        return list(deduped.values())

    def trace_replay_debug(self, state: AgentState) -> Dict[str, Any]:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        path = str(Path(outputs_path) / "trace_replay.json") if outputs_path else ""
        return {
            "version": "v2",
            "path": path,
            "actionTimelineCount": len(state.get("run_steps", [])),
            "spanTimelineCount": len(state.get("trace_spans", [])),
            "artifactCount": len(self.artifact_manifest(state)),
        }

    def checkpoint_debug(self, state: AgentState) -> Dict[str, Any]:
        ref = self.checkpoint_manager.run_ref(state.get("thread_id", ""), state.get("run_id", ""))
        ref["checkpointThreadId"] = state.get("checkpoint_thread_id") or ref.get("checkpointThreadId", "")
        return ref

    def checkpoint_state_summary(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        config = self.checkpoint_manager.config_for_run(thread_id, run_id)
        snapshot = self.graph.get_state(config)
        values = snapshot.values if hasattr(snapshot, "values") else {}
        metadata = snapshot.metadata if hasattr(snapshot, "metadata") else {}
        tasks = snapshot.tasks if hasattr(snapshot, "tasks") else ()
        next_nodes = snapshot.next if hasattr(snapshot, "next") else ()
        return {
            "checkpointRef": self.checkpoint_manager.run_ref(thread_id, run_id),
            "metadata": metadata or {},
            "next": list(next_nodes or []),
            "taskCount": len(tasks or []),
            "valueKeys": sorted(list((values or {}).keys()))[:80] if isinstance(values, dict) else [],
            "hasValues": bool(values),
        }

    def open_diagnostic_debug(self, state: AgentState) -> Dict[str, Any]:
        return {
            "scope": state.get("open_diagnostic_scope", ""),
            "intent": state.get("open_diagnostic_intent", ""),
            "goal": state.get("open_diagnostic_goal", ""),
            "seedTopics": [enum_value(item) for item in state.get("open_diagnostic_seed_topics", [])],
        }

    def prepare_scoped_context_package(
        self,
        state: AgentState,
        stage: str,
        agent: str,
        task_id: str = "",
        allowed_tables: Optional[List[str]] = None,
        allowed_metrics: Optional[List[str]] = None,
    ):
        snapshot = self.context_manager.snapshot(state, stage)
        package = self.context_manager.package(
            state,
            stage=stage,
            agent=agent,
            snapshot=snapshot,
            task_id=task_id,
            allowed_tables=allowed_tables or [],
            allowed_metrics=allowed_metrics or [],
        )
        state.setdefault("context_packages", []).append(package)
        state["context_packages"] = state["context_packages"][-12:]
        state["active_context_package"] = self.compact_context_package(package)
        self.record_context_manifest(state, package)
        self.context_manager.persist_package(state, package)
        add_step(
            state,
            "Context Package：%s 使用最小上下文 package=%s artifacts=%d"
            % (agent, package.package_id, len(package.artifact_refs)),
        )
        return package

    def compact_context_package(self, package: Any) -> Dict[str, Any]:
        refs = []
        for ref in getattr(package, "artifact_refs", [])[:8]:
            refs.append(
                {
                    "title": ref.title,
                    "relativePath": ref.relative_path,
                    "merchantUri": ref.merchant_uri,
                    "reason": ref.reason,
                }
            )
        return {
            "packageId": getattr(package, "package_id", ""),
            "stage": getattr(package, "stage", ""),
            "agent": getattr(package, "agent", ""),
            "taskId": getattr(package, "task_id", ""),
            "contextHash": getattr(package, "context_hash", ""),
            "summary": str(getattr(package, "summary", "") or "")[:1200],
            "constraints": list(getattr(package, "constraints", []) or [])[:6],
            "allowedTables": list(getattr(package, "allowed_tables", []) or [])[:12],
            "allowedMetrics": list(getattr(package, "allowed_metrics", []) or [])[:24],
            "agentContextPolicy": dict(getattr(package, "agent_context_policy", {}) or {}),
            "evidenceGaps": list(getattr(package, "evidence_gaps", []) or [])[:8],
            "artifactRefs": refs,
        }

    def render_context_package_for_prompt(self, package: Any) -> str:
        compact = self.compact_context_package(package)
        return "## ContextPackage\n%s" % json.dumps(compact, ensure_ascii=False, default=str)

    def record_context_manifest(self, state: AgentState, package: Any) -> Dict[str, Any]:
        manifest = self.build_context_manifest(state, package)
        package_id = str(manifest.get("contextPackageId") or "")
        existing = [
            item
            for item in state.get("context_manifests", [])
            if str((item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item).get("contextPackageId") or "") != package_id
        ]
        existing.append(manifest)
        state["context_manifests"] = existing[-24:]
        state["active_context_manifest"] = manifest
        return manifest

    def build_context_manifest(self, state: AgentState, package: Any) -> Dict[str, Any]:
        budget_report = latest_context_budget_report(state, getattr(package, "stage", "") or "")
        artifact_refs = []
        for ref in getattr(package, "artifact_refs", [])[:12]:
            artifact_refs.append(ref if hasattr(ref, "model_dump") else ref)
        blocks = build_llm_context_blocks(state, package, budget_report)
        manifest = ContextManifest(
            stage=str(getattr(package, "stage", "") or ""),
            agent=str(getattr(package, "agent", "") or ""),
            context_package_id=str(getattr(package, "package_id", "") or ""),
            context_hash=str(getattr(package, "context_hash", "") or ""),
            blocks=blocks,
            cache_layout=context_cache_layout(blocks),
            quarantine_policy=context_quarantine_policy(blocks),
            allowed_tables=list(getattr(package, "allowed_tables", []) or [])[:12],
            allowed_metrics=list(getattr(package, "allowed_metrics", []) or [])[:24],
            memory_ids=context_memory_ids(state),
            semantic_ref_ids=context_semantic_ref_ids(state)[:40],
            artifact_refs=[
                item if hasattr(item, "model_dump") else artifact_ref_from_path(str(item.get("path") or item.get("relativePath") or ""), namespace=str(item.get("namespace") or "context"))
                for item in artifact_refs
            ],
            budget_report=budget_report,
        )
        return manifest.model_dump(by_alias=True)

    def answer_file_tool_context(self, state: AgentState, package: Any) -> str:
        max_rounds = int(getattr(self.settings, "answer_file_tool_rounds", 0) or 0)
        llm = getattr(self.answer_service, "llm", None)
        if max_rounds <= 0 or not llm or not getattr(llm, "configured", False) or not hasattr(llm, "tool_chat"):
            return ""
        request_context = state.get("request_context")
        has_attachments = bool(getattr(request_context, "offloaded_files", None))
        has_offloaded_output = bool(state.get("tool_output_budget_reports") or state.get("_middleware_offloaded_tasks"))
        task_results = list(getattr(state.get("agent_run_result"), "task_results", []) or [])
        large_result = any(
            len(getattr(getattr(item, "query_bundle", None), "rows", []) or [])
            > int(getattr(self.settings, "context_artifact_inline_max_rows", 20) or 20)
            for item in task_results
        )
        if not (has_attachments or has_offloaded_output or large_result):
            # Small verified query results are already inline.  A model call to
            # decide that no file needs reading is pure latency and was hidden
            # inside the structured-answer span.
            state.setdefault("route_decision_trace", []).append(
                {"stage": "answer_file_context", "decision": "skip", "reason": "verified_results_inline"}
            )
            return ""
        tools = semantic_file_tool_definitions() + artifact_file_tool_definitions()
        tool_schemas = [tool.openai_schema() for tool in tools]
        handlers = self.planner._semantic_tool_handlers() if hasattr(self.planner, "_semantic_tool_handlers") else {}
        results: List[Dict[str, Any]] = []
        calls_trace: List[Dict[str, Any]] = []
        for round_index in range(max_rounds):
            payload = {
                "question": state.get("question", ""),
                "contextPackage": self.compact_context_package(package),
                "verifiedEvidence": state["agent_run_result"].verified_evidence.model_dump(by_alias=True),
                "evidenceGaps": [gap.model_dump(by_alias=True) for gap in state["agent_run_result"].evidence_gaps[:8]],
                "artifactManifest": self.artifact_manifest(state)[:16],
                "previousToolResults": compact_file_tool_results_for_prompt(results),
                "instruction": (
                    "如果 contextPackage 和 verifiedEvidence 已足够回答，不要调用工具。"
                    "如果需要查看大结果、证据缺口、SQL 行预览或语义说明，只能读取相关 artifact/semantic 文件。"
                    "不要根据未验证文件内容创造新结论；读取结果只能作为组织回答和缺口说明的证据补充。"
                ),
            }
            llm_result = llm.tool_chat(
                "你是 AnswerAgent 的文件上下文选择器。按需读取文件，最终回答仍必须受 verifiedEvidence 约束。",
                json.dumps(payload, ensure_ascii=False, default=str),
                tool_schemas,
                {"content": "", "toolCalls": []},
                timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
            )
            calls = [
                ToolCallRequest(id=str(call.get("id") or "answer_file_%d_%d" % (round_index, idx)), name=str(call.get("name") or ""), args=call.get("args") or {})
                for idx, call in enumerate(llm_result.get("toolCalls") or [])
                if str(call.get("name") or "") in handlers
            ][:4]
            if not calls:
                break
            cache_policies = {
                "semantic_ls": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "semantic_read": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "semantic_grep": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_ls": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_read": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_grep": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
            }
            executed = self.planner.tool_runtime_service.execute_many(calls, handlers, cache_policies=cache_policies)
            serialized = [item.model_dump(by_alias=True) for item in executed]
            calls_trace.extend([call.model_dump(by_alias=True) for call in calls])
            results.extend(serialized)
        if not results:
            return ""
        state["answer_file_tool_results"] = {"calls": calls_trace, "results": results}
        add_step(state, "AnswerAgent File Context：按需读取 artifact/semantic 文件 results=%d" % len(results))
        return "## Answer File Tool Results\n%s" % json.dumps(compact_file_tool_results_for_prompt(results), ensure_ascii=False, default=str)

    def refresh_context_snapshot(self, state: AgentState, stage: str) -> None:
        snapshot = self.context_manager.refresh_state(state, stage)
        packages = state.get("context_packages") or []
        if packages:
            self.record_context_manifest(state, packages[-1])
        add_step(state, "Context Manager：刷新上下文快照 stage=%s protectedFacts=%d" % (stage, len(snapshot.protected_facts)))

    def sync_tool_runtime_state(self, state: AgentState) -> None:
        failure_trace = {"failures": [], "circuits": []}
        runtime_events: List[Dict[str, Any]] = []
        registries = []
        runtime_services = []
        if hasattr(self.node_worker, "tool_failure_registry"):
            registries.append(self.node_worker.tool_failure_registry)
        if hasattr(self.planner, "tool_failure_registry"):
            registries.append(self.planner.tool_failure_registry)
        if hasattr(self.node_worker, "tool_runtime_service"):
            runtime_services.append(("node", self.node_worker.tool_runtime_service))
        if hasattr(self.planner, "tool_runtime_service"):
            runtime_services.append(("planner", self.planner.tool_runtime_service))
        for registry in registries:
            trace = registry.trace()
            failure_trace["failures"].extend(trace.get("failures", []))
            failure_trace["circuits"].extend(trace.get("circuits", []))
        for runtime_name, service in runtime_services:
            for event in service.events()[-100:]:
                next_event = dict(event)
                next_event["runtime"] = runtime_name
                runtime_events.append(next_event)
        state["tool_failures"] = failure_trace.get("failures", [])
        state["circuit_breakers"] = failure_trace.get("circuits", [])
        seen_ids = set(state.get("_emitted_tool_runtime_event_ids") or [])
        for event in runtime_events:
            event_id = str(event.get("eventId") or "")
            if event_id and event_id not in seen_ids:
                emit(state, str(event.get("eventType") or "tool.runtime"), "TOOL_RUNTIME", event)
                seen_ids.add(event_id)
        state["_emitted_tool_runtime_event_ids"] = list(seen_ids)[-500:]
        state["tool_runtime_events"] = runtime_events[-200:]
        if hasattr(self.node_worker, "tool_runtime_policies"):
            state["tool_runtime_policies"] = self.node_worker.tool_runtime_policies.trace()

    def context_management_debug(self, state: AgentState) -> Dict[str, Any]:
        snapshots = state.get("context_snapshots", [])
        packages = state.get("context_packages", [])
        manifest = state.get("workspace_manifest", WorkspaceManifest())
        return {
            "strategy": "lead keeps global goal/progress; node agents receive scoped node context; snapshots preserve protected facts with source refs",
            "snapshotCount": len(snapshots),
            "packageCount": len(packages),
            "budgetReports": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in state.get("context_budget_reports", [])[-8:]
            ],
            "assemblyReports": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in state.get("context_assembly_reports", [])[-12:]
            ],
            "contextManifests": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in state.get("context_manifests", [])[-8:]
            ],
            "activeContextManifest": state.get("active_context_manifest", {}),
            "compressionEvents": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in state.get("context_compression_events", [])[-8:]
            ],
            "runtimeCheckpoints": state.get("runtime_checkpoints", [])[-8:],
            "threadContext": state.get("thread_context", {}),
            "runtimeInjectionPreview": json.dumps(state.get("runtime_injection", {}), ensure_ascii=False, default=str)[:1200],
            "memoryInjectionPreview": json.dumps(state.get("memory_injection", {}), ensure_ascii=False, default=str)[:1200],
            "workspaceManifest": manifest.model_dump(by_alias=True) if hasattr(manifest, "model_dump") else manifest,
            "recentSnapshots": snapshots[-4:],
            "recentPackages": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in packages[-4:]
            ],
            "summaryContextPreview": str(state.get("summary_context") or "")[:1200],
            "artifactCount": len(self.artifact_manifest(state)),
            "inlineRows": sum(len(task.query_bundle.rows) for task in state.get("agent_run_result", AgentRunResult()).task_results),
            "inlineChars": sum(len(str(task.query_bundle.rows[:3])) for task in state.get("agent_run_result", AgentRunResult()).task_results),
            "recoverySources": [
                "recent state",
                "context_snapshot.json",
                "trace_replay.json",
                "context_packages/*.json",
                "workspace files",
                "user clarification",
            ],
        }

    def memory_debug(self, state: AgentState) -> Dict[str, Any]:
        injection = state.get("memory_injection") or {}
        trace = state.get("memory_injection_trace") or injection.get("memoryInjectionTrace") or {}
        ingestion = state.get("memory_ingestion_trace") or {}
        constraints = state.get("memory_constraints") or []
        constraint_trace = state.get("memory_constraint_trace") or {}
        candidates = trace.get("candidates") or []
        conflicts = ingestion.get("conflict") or {}
        return {
            "ingestion": ingestion,
            "retrieval": {
                "status": state.get("memory_recall_status") or trace.get("status", "not_started"),
                "usableSnapshot": bool(trace.get("usableSnapshot")),
                "issues": list(state.get("memory_recall_issues") or trace.get("issues") or []),
                "enrichmentStatus": dict(trace.get("enrichmentStatus") or {}),
                "candidateCount": trace.get("candidateCount", 0),
                "selectedIds": trace.get("selectedIds", []),
                "candidateIds": trace.get("candidateIds", []),
                "filteredReasons": trace.get("filteredReasons", {}),
                "topCandidates": candidates[:8],
            },
            "injection": {
                "budgetChars": trace.get("budgetChars", 0),
                "budgetUsedChars": trace.get("budgetUsedChars", 0),
                "truncated": bool(trace.get("truncated")),
                "recentFocus": injection.get("recentFocus", {}),
                "correctionCount": len(injection.get("relevantCorrections") or []),
                "preferenceCount": len(injection.get("relevantPreferences") or []),
                "factCount": len(injection.get("relevantFacts") or []),
                "eventCount": len(injection.get("relevantEvents") or []),
                "pastCaseCount": len(injection.get("relevantPastCases") or []),
                "quarantinedMemoryCount": len(injection.get("quarantinedMemories") or []),
            },
            "knowledgeSuggestions": {
                "count": int((ingestion or {}).get("knowledgeSuggestionCount") or 0),
                "lastSuggestionId": (ingestion or {}).get("knowledgeSuggestionId", ""),
                "written": bool((ingestion or {}).get("knowledgeSuggestionWritten")),
            },
            "constraints": {
                "constraintCount": constraint_trace.get("constraintCount", len(constraints)),
                "requiredCount": constraint_trace.get(
                    "requiredCount",
                    sum(1 for item in constraints if str(item.get("enforcement") or "") == "required"),
                ),
                "clarifyCount": constraint_trace.get(
                    "clarifyCount",
                    sum(1 for item in constraints if str(item.get("enforcement") or "") == "clarify_or_disclose"),
                ),
                "items": constraints[:8],
            },
            "conflicts": [conflicts] if conflicts else [],
            "decay": {
                "strategy": "confidence * time_decay * feedback_weight * hit_count_boost",
                "recentFocusWeighted": bool((injection.get("recentFocus") or {}).get("updatedBy")),
            },
        }

    def planner_runtime_debug(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or QueryPlan()
        stats = dict(getattr(plan, "planner_prompt_stats", None) or {})
        tool_calls = list(getattr(plan, "planner_tool_calls", None) or [])
        tool_results = list(getattr(plan, "planner_tool_results", None) or [])
        semantic_reads = [
            call for call in tool_calls if str(call.get("name") or call.get("toolName") or "") == "semantic_read"
        ]
        semantic_refs = []
        for call in semantic_reads:
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            ref_id = str(args.get("refId") or args.get("ref_id") or "")
            if ref_id and ref_id not in semantic_refs:
                semantic_refs.append(ref_id)
        fast_path = any("planner.semantic_fast_path" in str(item) for item in (getattr(plan, "agent_trace", []) or []))
        return {
            "promptTotalChars": int(stats.get("totalChars") or 0),
            "maxRoundTotalChars": int(stats.get("maxRoundTotalChars") or stats.get("totalChars") or 0),
            "budgetLevel": int(stats.get("budgetLevel") or 0),
            "toolRounds": len(stats.get("toolRounds") or []),
            "toolCallCount": len(tool_calls),
            "semanticReadCalls": len(semantic_reads),
            "semanticReadRefs": semantic_refs[:12],
            "plannerToolResultCount": len(tool_results),
            "loadedRefCount": len(getattr(plan, "planner_loaded_refs", []) or []),
            "contextFileCount": len(getattr(plan, "planner_context_files", []) or []),
            "schemaMode": str(stats.get("schemaMode") or ""),
            "fastPath": fast_path,
            "degraded": state.get("planner_degraded") or {},
        }

    def middleware_debug(self, state: AgentState) -> Dict[str, Any]:
        events = state.get("middleware_events", [])
        ledger = state.get("tool_call_ledger", [])
        recoveries = state.get("tool_call_recovery_events", [])
        return {
            "enabled": True,
            "middlewares": [item.name for item in self.middleware_chain.middlewares],
            "eventCount": len(events),
            "recentEvents": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in events[-24:]
            ],
            "toolCallLedger": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in ledger[-24:]
            ],
            "toolCallRecoveryEvents": [
                item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                for item in recoveries[-24:]
            ],
            "runBudgetReport": state.get("run_budget_report", {}),
            "safetyFinishReasons": state.get("safety_finish_reasons", [])[-12:],
            "runCanceled": bool(state.get("run_canceled")),
            "runBudgetExhausted": bool(state.get("run_budget_exhausted")),
            "loopBlocked": bool(state.get("middleware_loop_blocked")),
            "actionContextHashes": state.get("middleware_action_context_hashes", {}),
        }

    def tool_runtime_debug(self, state: AgentState) -> Dict[str, Any]:
        policies = state.get("tool_runtime_policies", [])
        failures = state.get("tool_failures", [])
        circuits = state.get("circuit_breakers", [])
        runtime_traces = {}
        if hasattr(self.node_worker, "tool_runtime_policies"):
            policies = self.node_worker.tool_runtime_policies.trace()
        for name, owner in [("node", self.node_worker), ("planner", self.planner)]:
            if hasattr(owner, "tool_runtime_service"):
                runtime_traces[name] = owner.tool_runtime_service.trace()
        merged_alerts = []
        merged_metrics = []
        merged_events = []
        merged_rate_limits: Dict[str, Any] = {}
        merged_load_balancer: Dict[str, Any] = {}
        for name, trace in runtime_traces.items():
            merged_alerts.extend(trace.get("alerts", []))
            for event in trace.get("events", []):
                next_event = dict(event)
                next_event["runtime"] = name
                merged_events.append(next_event)
            for item in trace.get("metrics", {}).get("tools", []):
                next_item = dict(item)
                next_item["runtime"] = name
                merged_metrics.append(next_item)
            merged_rate_limits[name] = trace.get("rateLimits", {})
            merged_load_balancer[name] = trace.get("loadBalancer", {})
        self.sync_tool_runtime_state(state)
        failures = state.get("tool_failures", failures)
        circuits = state.get("circuit_breakers", circuits)
        return {
            "policies": policies,
            "failures": failures,
            "circuits": circuits,
            "metrics": {"tools": merged_metrics},
            "events": state.get("tool_runtime_events", merged_events[-200:]),
            "rateLimits": merged_rate_limits,
            "loadBalancer": merged_load_balancer,
            "alerts": merged_alerts,
            "parallelism": {
                "maxConcurrentNodeAgents": self.settings.max_concurrent_sub_agents,
                "maxConcurrentToolCalls": self.settings.tool_max_concurrency,
                "resultPairing": "tool_call id",
                "failureIsolation": True,
            },
            "circuitBreaker": {
                "repeatedIdenticalFailureBlocks": True,
                "toolFailureThresholdBlocksTool": True,
                "repeatThreshold": self.settings.tool_failure_repeat_threshold,
                "circuitThreshold": self.settings.tool_circuit_threshold,
                "cooldownSeconds": self.settings.tool_circuit_cooldown_seconds,
            },
        }

    def cache_debug(self) -> Dict[str, Any]:
        return {
            "enabled": self.settings.cache_enabled,
            "memoryMaxEntries": self.settings.cache_memory_max_entries,
            "recall": self.recall_service.cache_trace() if hasattr(self.recall_service, "cache_trace") else {},
            "knowledgeRetriever": self.knowledge_retriever.cache_trace()
            if hasattr(self.knowledge_retriever, "cache_trace")
            else {},
            "assetBuilder": self.asset_builder.cache_trace() if hasattr(self.asset_builder, "cache_trace") else {},
            "doris": self.node_worker.doris_repository.cache_trace()
            if hasattr(self.node_worker.doris_repository, "cache_trace")
            else {},
            "llm": self.planner.llm.cache_trace() if hasattr(self.planner.llm, "cache_trace") else {},
            "policy": {
                "dorisSelectTtlSeconds": self.settings.cache_doris_select_ttl_seconds,
                "recallTtlSeconds": self.settings.cache_recall_ttl_seconds,
                "assetPackTtlSeconds": self.settings.cache_asset_pack_ttl_seconds,
                "llmTtlSeconds": self.settings.cache_llm_ttl_seconds,
            },
        }

    def tool_calling_debug(self, state: AgentState) -> Dict[str, Any]:
        selected_tools: List[str] = []
        for profile in state.get("agent_run_result", AgentRunResult()).node_task_profiles:
            for tool in profile.selected_tools:
                if tool not in selected_tools:
                    selected_tools.append(tool)
        if not selected_tools:
            selected_tools = list(self.node_worker.node_agent.TOOL_REGISTRY.keys())
        return {
            "nativeToolCallingSupported": hasattr(self.planner.llm, "tool_chat"),
            "leadActionTool": lead_action_selection_tool(self.policy.registry.public_action_ids()).trace_schema(),
            "semanticFileTools": semantic_file_tool_schemas(),
            "artifactFileTools": artifact_file_tool_schemas(),
            "nodeToolSchemas": node_runtime_tool_schemas(self.node_worker.node_agent.TOOL_REGISTRY, selected_tools),
            "structuredOutputTools": [
                "emit_question_understanding",
                "draft_sql",
                "repair_sql",
            ],
            "contextPolicy": {
                "lead": ["global goal", "routing state", "action history", "budgets", "final summary"],
                "node": ["single QueryGraph node", "node-local schema/assets", "upstream entity set", "preview rows"],
                "answer": ["verified evidence", "evidence gaps", "result rows preview", "plan summary"],
            },
        }

    def materialize_plan_clarification(self, state: AgentState) -> bool:
        """Project PlannerAgent's terminal clarification contract into workflow state."""

        if state.get("human_clarification_required"):
            return True
        plan = state.get("plan") or QueryPlan()
        needs = dedupe_texts(
            [str(item).strip() for item in (getattr(plan, "clarification_needs", None) or []) if str(item or "").strip()]
        )
        if not needs:
            return False
        question = needs[0]
        if len(needs) > 1:
            question = "为确保查询口径一致，请确认以下问题：\n" + "\n".join(
                "%d. %s" % (index, item) for index, item in enumerate(needs, start=1)
            )
        self.request_human_clarification(state, question, "QUERY_PLAN", "planner_clarification", [])
        add_step(state, "PlannerAgent：查询计划需要确认业务口径，已转入 ask_human，未进入 QueryGraph 校验")
        return True

    def request_human_clarification(self, state: AgentState, question: str, stage: str, type_: str, options: List[str]) -> None:
        question_text = str(question or "").strip()
        state["human_clarification_required"] = True
        state["scope_clarified"] = False
        state["human_clarification_question"] = question_text
        state["human_clarification_stage"] = stage
        state["human_clarification_type"] = type_
        state["human_clarification_options"] = options

    def thread_summary_path(self, state: AgentState) -> Path:
        return (
            self.settings.resolved_workspace_path
            / "threads"
            / str(state.get("thread_id") or "thread")
            / "published"
            / ("%s.summary.json" % str(state.get("run_id") or "run"))
        )

    def publish_thread_summary(self, state: AgentState) -> None:
        if self.run_cancellation_requested(state):
            return
        path = self.thread_summary_path(state)
        if path.exists():
            return
        identity_scope = identity_scope_payload(state.get("user_identity") or {}, state.get("requested_merchant_id", ""))
        run_result = state.get("agent_run_result") or AgentRunResult()
        run_result_payload = run_result.model_dump(by_alias=True) if hasattr(run_result, "model_dump") else {}
        artifacts = self.artifact_manifest(state)[:20]
        reusable_entity_sets = extract_reusable_entity_sets(run_result_payload)[:12]
        thread_context = state.get("thread_context") or {}
        payload = {
            "version": 2,
            "runId": state.get("run_id", ""),
            "threadId": state.get("thread_id", ""),
            "merchantId": state.get("requested_merchant_id", ""),
            "ownerUserId": identity_scope.get("userId", ""),
            "ownerRole": identity_scope.get("role", ""),
            "identityScopeHash": identity_scope_hash(
                state.get("user_identity") or {},
                state.get("requested_merchant_id", ""),
            ),
            "question": state.get("question", ""),
            "answerPreview": str(state.get("answer") or "")[:1000],
            "summary": str(state.get("summary_context") or state.get("answer") or "")[:2000],
            "messageHistorySummary": thread_context.get("messageHistorySummary") or {},
            "topicWorkspace": dict(state.get("topic_workspace") or {}),
            "reusableEntitySets": reusable_entity_sets,
            "publishedAt": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
        try:
            with temp_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_path, path)
        except FileExistsError:
            return
        except OSError as exc:
            add_step(state, "Thread Context：发布不可变摘要失败 %s" % str(exc)[:180])
        finally:
            temp_path.unlink(missing_ok=True)

    def terminal_end(self, state: AgentState) -> AgentState:
        terminal = state.get("terminal_status") or {}
        if not state.get("answer"):
            state["answer"] = str(terminal.get("message") or "本轮运行已被安全策略终止，未继续调用工具。")
        state["should_persist"] = False
        state["persisted"] = False
        state["chat_bi_completed"] = True
        self.emit_answer_ready(state)
        return state

    def emit_answer_ready(self, state: AgentState) -> None:
        answer = str(state.get("answer") or "")
        if not answer or state.get("_answer_ready_emitted"):
            return
        state["_answer_ready_emitted"] = True
        emit(
            state,
            "answer.ready",
            "ANSWER_STREAM",
            {
                "answer": answer,
                "answerLength": len(answer),
            },
        )

    def run_cancellation_requested(self, state: AgentState) -> bool:
        if state.get("run_canceled"):
            return True
        run_id = str(state.get("run_id") or "")
        store = getattr(self.node_worker, "runtime_state_store", None)
        try:
            canceled = bool(run_id and store and store.run_canceled(run_id))
        except Exception:
            canceled = False
        if canceled:
            state["run_canceled"] = True
        return canceled

    def finish_canceled_run(self, state: AgentState) -> AgentState:
        state["run_canceled"] = True
        state["answer"] = "本次运行已取消。"
        state["should_persist"] = False
        state["persisted"] = False
        state["chat_bi_completed"] = True
        self.emit_answer_ready(state)
        return state

    def build_scope_clarification_prompt(self, state: AgentState) -> str:
        del state
        return "请补充你想看的业务指标或对象，以及时间范围；内部 Topic 会由系统自动选择。"

    def build_topic_clarification_prompt(self, state: AgentState) -> str:
        del state
        return "这个问题的业务目标还不够明确。请补充要看的指标、业务对象或分析维度；内部 Topic 由系统自动选择。"

    def bootstrap_asset_diagnostic_state(self, state: AgentState) -> None:
        profile = diagnostic_profile_for_question(
            state.get("question", ""),
            self.recall_service.topic_assets,
            state.get("request_context"),
        )
        if not profile:
            return
        state["open_diagnostic_profile"] = profile
        state["open_diagnostic_scope"] = "OPEN_DIAGNOSTIC"
        state["open_diagnostic_intent"] = str(profile.get("id") or "")
        if not state.get("open_diagnostic_goal"):
            state["open_diagnostic_goal"] = str(profile.get("defaultGoal") or "")

    def build_diagnostic_goal_clarification_prompt(self, profile: Dict[str, Any]) -> str:
        prompt = str(profile.get("clarificationPrompt") or "").strip()
        if prompt:
            return prompt
        return "请补充本次分析的排序目标或约束；系统只会使用已发布语义资产。"

    def apply_open_diagnostic_policy(self, state: AgentState, decision: TopicRoutingDecision) -> List[QuestionCategory]:
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            return []
        del decision
        profile = state.get("open_diagnostic_profile") or {}
        intent = str(state.get("open_diagnostic_intent") or profile.get("id") or "")
        if not intent:
            return []
        context = state.get("request_context")
        clarification_type = str(profile.get("clarificationType") or "diagnostic_goal")
        if context and str(context.pending_clarification_type or "") == clarification_type:
            goal = str(
                (state.get("clarification_resolution") or {}).get("diagnosticGoal")
                or getattr(context, "priority_goal", "")
                or state.get("question")
                or ""
            ).strip()
            return self.mark_open_diagnostic(state, intent=intent, goal=goal)
        goal = str(state.get("open_diagnostic_goal") or profile.get("defaultGoal") or "").strip()
        if bool(profile.get("goalRequired")) and not goal:
            self.mark_open_diagnostic(state, intent=intent, goal="")
            self.request_human_clarification(
                state,
                self.build_diagnostic_goal_clarification_prompt(profile),
                "OPEN_DIAGNOSTIC",
                clarification_type,
                [str(item) for item in profile.get("goalOptions") or []],
            )
            return []
        return self.mark_open_diagnostic(state, intent=intent, goal=goal)

    def mark_open_diagnostic(self, state: AgentState, intent: str, goal: str) -> List[QuestionCategory]:
        seed_topics = diagnostic_seed_topics(intent, self.recall_service.topic_assets)
        state["open_diagnostic_scope"] = "OPEN_DIAGNOSTIC"
        state["open_diagnostic_intent"] = intent
        state["open_diagnostic_goal"] = goal
        state["open_diagnostic_seed_topics"] = seed_topics
        state["knowledge_expanded_topics"] = self._merge_topic_categories(state.get("knowledge_expanded_topics") or [], seed_topics)
        return seed_topics

    def _topic_names_for_categories(self, categories: List[Any]) -> List[str]:
        topic_asset_service = self.recall_service.topic_assets
        return topic_asset_service.topic_names_for_categories(categories)

    def session_topic_categories(self, state: AgentState) -> List[QuestionCategory]:
        """Return the previous Topic set as a weak continuation prior.

        The normal chat response is produced by the Agent, so merely round-
        tripping ``context.topic(s)`` cannot be treated as user confirmation.
        ``TopicRouterService`` may replace this prior whenever the current
        merchant question carries stronger business-domain evidence.
        """
        context = state.get("request_context")
        raw_categories: List[Any] = list(getattr(context, "topics", []) or []) if context else []
        if not raw_categories:
            restored_workspace = (state.get("thread_context") or {}).get("topicWorkspace") or {}
            raw_categories = list(restored_workspace.get("topicIds") or [])
            if not raw_categories:
                raw_categories = list(restored_workspace.get("topics") or [])
        categories: List[QuestionCategory] = []
        for raw in raw_categories:
            try:
                category = QuestionCategory(raw)
            except (TypeError, ValueError):
                category = self.recall_service.topic_assets.resolve_topic_category(str(raw or ""))
            if category and category != QuestionCategory.UNKNOWN and category not in categories:
                categories.append(category)
        return categories

    def session_topic_scope_is_locked(self, state: AgentState) -> bool:
        """Only an explicit user scope or resolved Topic clarification is sticky."""
        context = state.get("request_context")
        if bool(
            context
            and getattr(context, "clarification_resolved", False)
            and str(getattr(context, "pending_clarification_type", "") or "") == "topic_required"
        ):
            return True
        workspace = (state.get("thread_context") or {}).get("topicWorkspace") or {}
        return bool(
            str(workspace.get("mode") or "") == "explicit_topic_scope"
            or str(workspace.get("expansionPolicy") or "") == "user_locked"
        )

    def synchronize_response_topic_context(self, state: AgentState) -> None:
        """Persist the routed workspace instead of deriving Topic from graph nodes."""
        response_context = state.get("response_context")
        if response_context is None:
            return
        workspace = state.get("topic_workspace") or {}
        categories: List[QuestionCategory] = []
        for raw in list(workspace.get("topicIds") or []):
            try:
                category = QuestionCategory(raw)
            except (TypeError, ValueError):
                continue
            if category != QuestionCategory.UNKNOWN and category not in categories:
                categories.append(category)
        if not categories:
            categories = (state.get("topic_routing_decision") or TopicRoutingDecision()).recall_topics()
        topic_names = list(workspace.get("topics") or self._topic_names_for_categories(categories))
        if categories:
            response_context.topics = categories
        if topic_names:
            response_context.topic = "、".join(str(item) for item in topic_names)

    def _asset_backed_topic_categories(self, categories: List[Any]) -> List[QuestionCategory]:
        result: List[QuestionCategory] = []
        for item in categories:
            category = QuestionCategory(item)
            if category == QuestionCategory.UNKNOWN:
                continue
            if self._topic_names_for_categories([category]) and category not in result:
                result.append(category)
        return result

    def _effective_topic_categories(self, state: AgentState) -> List[QuestionCategory]:
        base = state["topic_routing_decision"].recall_topics()
        expanded = state.get("knowledge_expanded_topics") or []
        opened: List[QuestionCategory] = []
        for topic_name in state.get("semantic_workspace_opened_topics") or []:
            category = self.recall_service.topic_assets.resolve_topic_category(topic_name)
            if category != QuestionCategory.UNKNOWN and category not in opened:
                opened.append(category)
        return self._merge_topic_categories(self._merge_topic_categories(base, expanded), opened)

    def _knowledge_request_topics(self, request: KnowledgeRequest, base_topics: List[QuestionCategory]) -> List[QuestionCategory]:
        text = "%s %s" % (request.query or "", request.reason or "")
        decision = self.topic_router.route(text, self.keyword_service.extract(text), "")
        topics = self._merge_topic_categories(base_topics, decision.recall_topics())
        topics = self._merge_topic_categories(topics, self._topics_from_table_mentions(text))
        if request.type == KnowledgeRequestType.RELATIONSHIP:
            topics = self._merge_topic_categories(topics, self._relationship_path_topics_from_text(text))
        return topics

    def _topics_from_table_mentions(self, text: str) -> List[QuestionCategory]:
        lowered = (text or "").lower()
        found: List[QuestionCategory] = []
        for topic_name in self.recall_service.topic_assets.all_topic_names():
            category = self.recall_service.topic_assets.resolve_topic_category(topic_name)
            if not category:
                continue
            for manifest_item in self.recall_service.topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if table and table.lower() in lowered and category not in found:
                    found.append(category)
        return found

    def _relationship_path_topics_from_text(self, text: str) -> List[QuestionCategory]:
        mentioned_tables = self._mentioned_relationship_tables(text)
        if len(mentioned_tables) < 2:
            return []
        table_topic = self._table_topic_index()
        adjacency: Dict[str, List[str]] = {}
        for topic_name in self.recall_service.topic_assets.all_topic_names():
            for rel in self.recall_service.topic_assets.load_relationships(topic_name):
                left = str(rel.get("leftTable") or "")
                right = str(rel.get("rightTable") or "")
                if left and right:
                    adjacency.setdefault(left, []).append(right)
                    adjacency.setdefault(right, []).append(left)
        topics: List[QuestionCategory] = []
        for index, start in enumerate(mentioned_tables):
            for target in mentioned_tables[index + 1 :]:
                for table in shortest_table_path(start, target, adjacency):
                    topic_name = table_topic.get(table, "")
                    category = self.recall_service.topic_assets.resolve_topic_category(topic_name)
                    if category and category not in topics:
                        topics.append(category)
        return topics

    def _mentioned_relationship_tables(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        tables: List[str] = []
        for topic_name in self.recall_service.topic_assets.all_topic_names():
            for rel in self.recall_service.topic_assets.load_relationships(topic_name):
                for table in [str(rel.get("leftTable") or ""), str(rel.get("rightTable") or "")]:
                    if table and table.lower() in lowered and table not in tables:
                        tables.append(table)
        return tables

    def _table_topic_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for topic_name in self.recall_service.topic_assets.all_topic_names():
            for manifest_item in self.recall_service.topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if table and table not in index:
                    index[table] = topic_name
        return index

    def _merge_topic_categories(self, first: List[Any], second: List[Any]) -> List[QuestionCategory]:
        merged: List[QuestionCategory] = []
        for item in list(first or []) + list(second or []):
            try:
                category = item if isinstance(item, QuestionCategory) else QuestionCategory(str(item))
            except Exception:
                continue
            if category != QuestionCategory.UNKNOWN and category not in merged:
                merged.append(category)
        return merged


def business_scope_options(topic_assets: Any = None) -> List[str]:
    contracts = topic_assets.topic_contracts() if topic_assets is not None else []
    options = [
        str(item.get("displayName") or item.get("topic") or "")
        for item in contracts
        if str(item.get("categoryId") or "") != str(QuestionCategory.UNKNOWN)
    ]
    return dedupe_texts(options)[:6]


def business_scope_examples(topic_assets: Any = None) -> List[str]:
    """Read user-facing examples from published Topic clarification assets."""

    contract = topic_clarification_contract(topic_assets, "business_scope")
    return list(contract.get("options") or business_scope_options(topic_assets))


def topic_clarification_contract(topic_assets: Any, clarification_type: str) -> Dict[str, Any]:
    """Merge an optional clarification contract declared by published Topics."""

    if topic_assets is None or not clarification_type:
        return {}
    prompts: List[str] = []
    options: List[str] = []
    for topic in topic_assets.topic_contracts():
        metadata = topic.get("metadata") if isinstance(topic.get("metadata"), dict) else {}
        contracts = metadata.get("clarificationContracts")
        contract: Dict[str, Any] = {}
        if isinstance(contracts, dict):
            candidate = contracts.get(clarification_type)
            contract = candidate if isinstance(candidate, dict) else {}
        elif isinstance(contracts, list):
            contract = next(
                (
                    item
                    for item in contracts
                    if isinstance(item, dict) and str(item.get("type") or "") == clarification_type
                ),
                {},
            )
        prompt = str(contract.get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)
        options.extend(str(item) for item in contract.get("options") or [] if str(item or "").strip())
    return {"prompt": prompts[0] if prompts else "", "options": dedupe_texts(options)[:8]}


def submit_with_current_context(executor: ThreadPoolExecutor, fn: Any, *args: Any):
    context = copy_context()
    return executor.submit(context.run, fn, *args)


def merge_clarification_question(pending_question: str, normalized_answer: str) -> str:
    pending = str(pending_question or "").strip()
    answer = str(normalized_answer or "").strip()
    if not pending:
        return answer
    if not answer or answer == pending or pending.endswith(answer):
        return pending
    return "%s %s" % (pending, answer)


def merchant_analysis_action_label(skill_name: str, skills: Optional[List[Any]] = None) -> str:
    requested = str(skill_name or "").strip()
    for skill in skills or []:
        domain = str(getattr(skill, "domain", "") or "").strip()
        if domain == requested:
            return str(getattr(skill, "display_name", "") or domain)
    return requested or "已发布分析技能"


def skill_confirmation_declined(resolution: Dict[str, Any]) -> bool:
    return str((resolution or {}).get("confirmationDecision") or "") == "declined"


def merchant_access_role(role: str) -> str:
    return load_authorization_policy().access_role_for_identity(role)


def diagnostic_profile_for_question(
    question: str,
    topic_assets: Any = None,
    request_context: Any = None,
) -> Dict[str, Any]:
    """Resolve an open-diagnostic intent only from published Topic metadata."""

    if topic_assets is None:
        return {}
    profiles: Dict[str, Dict[str, Any]] = {}
    for contract in topic_assets.topic_contracts():
        metadata = contract.get("metadata") if isinstance(contract.get("metadata"), dict) else {}
        declared = metadata.get("diagnosticIntents") or []
        for raw_profile in declared if isinstance(declared, list) else []:
            if not isinstance(raw_profile, dict):
                continue
            profile_id = str(raw_profile.get("id") or raw_profile.get("profile") or "").strip()
            if not profile_id:
                continue
            target = profiles.setdefault(profile_id, {"id": profile_id, "triggerTerms": [], "goalOptions": []})
            for key in ("defaultGoal", "goalRequired", "clarificationType", "clarificationPrompt"):
                if key in raw_profile and raw_profile.get(key) not in (None, ""):
                    target[key] = raw_profile.get(key)
            for source_key, target_key in (("triggerTerms", "triggerTerms"), ("aliases", "triggerTerms"), ("goalOptions", "goalOptions")):
                values = raw_profile.get(source_key) or []
                target[target_key] = dedupe_texts(
                    [*target.get(target_key, []), *[str(item) for item in values if str(item or "").strip()]]
                )
    pending_type = str(getattr(request_context, "pending_clarification_type", "") or "")
    if pending_type:
        matched = next(
            (profile for profile in profiles.values() if str(profile.get("clarificationType") or "") == pending_type),
            {},
        )
        if matched:
            return dict(matched)
    normalized_question = normalize_for_match(question)
    for profile in profiles.values():
        for term in profile.get("triggerTerms") or []:
            normalized_term = normalize_for_match(term)
            if normalized_term and normalized_term in normalized_question:
                return dict(profile)
    return {}


def diagnostic_seed_topics(intent: str, topic_assets: Any = None) -> List[QuestionCategory]:
    """Select diagnostic scope only from asset-declared topic contracts."""

    if topic_assets is None:
        return []
    selected: List[QuestionCategory] = []
    for contract in topic_assets.topic_contracts():
        metadata = contract.get("metadata") or {}
        profiles = [str(item) for item in metadata.get("diagnosticProfiles") or []]
        enabled = bool(metadata.get("openDiagnostic")) or bool(profiles)
        if not enabled or (profiles and intent not in profiles):
            continue
        category = QuestionCategory(contract.get("categoryId") or contract.get("topic"))
        if category != QuestionCategory.UNKNOWN and category not in selected:
            selected.append(category)
    return selected


def dedupe_texts(values: List[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.append(text)
    return seen


def shortest_table_path(start: str, target: str, adjacency: Dict[str, List[str]]) -> List[str]:
    if not start or not target:
        return []
    if start == target:
        return [start]
    queue: List[List[str]] = [[start]]
    visited = {start}
    while queue:
        path = queue.pop(0)
        node = path[-1]
        for next_node in adjacency.get(node, []):
            if next_node in visited:
                continue
            next_path = path + [next_node]
            if next_node == target:
                return next_path
            visited.add(next_node)
            queue.append(next_path)
    return []


def graph_repair_validation_gaps(evidence_gaps: List[Any]) -> List[GraphValidationGap]:
    gaps: List[GraphValidationGap] = []
    for gap in evidence_gaps:
        code = str(getattr(gap, "code", ""))
        if code not in REPAIRABLE_QUERY_GRAPH_GAP_CODES:
            continue
        gaps.append(
            GraphValidationGap(
                code=code,
                task_id=str(getattr(gap, "task_id", "")),
                evidence=str(getattr(gap, "evidence", "")),
                reason=str(getattr(gap, "reason", ""))
                or "execution evidence indicates QueryGraph dependency repair is needed",
            )
        )
    return gaps


def planner_repair_gaps_from_state(state: AgentState) -> List[GraphValidationGap]:
    """Merge validator and PlannerCritic failures into one typed repair input."""

    validation = state.get("query_graph_validation_result") or GraphValidationResult()
    gaps = list(getattr(validation, "gaps", []) or [])
    reflection = state.get("planner_reflection") or PlannerReflectionResult()
    issues = reflection.get("issues", []) if isinstance(reflection, dict) else reflection.issues
    for issue in issues or []:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "error").strip().lower()
        if severity != "error":
            continue
        gap = GraphValidationGap(
            code=str(issue.get("code") or ""),
            evidence=str(issue.get("evidence") or ""),
            task_id=str(issue.get("taskId") or issue.get("task_id") or ""),
            reason=str(issue.get("reason") or ""),
        )
        if not gap.code:
            continue
        if any(
            current.code == gap.code
            and current.task_id == gap.task_id
            and current.evidence == gap.evidence
            for current in gaps
        ):
            continue
        gaps.append(gap)
    for request in normalized_planner_repair_requests(state.get("planner_repair_requests") or []):
        payload = request.model_dump(by_alias=True, mode="json")
        gap = GraphValidationGap(
            code=str(request.reason or "PLANNER_REPAIR_REQUEST"),
            evidence=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            task_id=str(request.task_id or ""),
            reason=(
                "PlannerCritic requested %s at %s%s"
                % (
                    str(request.action or "graph_repair"),
                    str(request.stage or "planner_reflection"),
                    (" from " + str(request.source)) if request.source else "",
                )
            ),
        )
        if any(
            current.code == gap.code
            and current.task_id == gap.task_id
            and current.evidence == gap.evidence
            for current in gaps
        ):
            continue
        gaps.append(gap)
    return gaps


def normalized_planner_repair_requests(values: List[Any]) -> List[PlannerRepairRequest]:
    requests: List[PlannerRepairRequest] = []
    for value in values or []:
        if isinstance(value, PlannerRepairRequest):
            requests.append(value.model_copy(deep=True))
            continue
        if isinstance(value, dict):
            try:
                requests.append(PlannerRepairRequest.model_validate(value))
            except Exception:
                continue
    return requests


QUERY_GRAPH_REPAIR_OPERATIONAL_GAP_CODES = frozenset(
    {
        "QUERY_GRAPH_REPAIR_AWAITING_KNOWLEDGE",
        "QUERY_GRAPH_REPAIR_EXHAUSTED",
        "QUERY_GRAPH_REPAIR_FAILED",
        "QUERY_GRAPH_REPAIR_NO_PROGRESS",
        "QUERY_GRAPH_REPAIR_PENDING_REVALIDATION",
    }
)


def planner_repair_scope_key(state: AgentState, plan: QueryPlan | None = None) -> str:
    """Scope repair budgets to one executable graph and one critic issue set."""

    reflection_value = state.get("planner_reflection") or PlannerReflectionResult()
    if isinstance(reflection_value, PlannerReflectionResult):
        reflection = reflection_value
    elif isinstance(reflection_value, dict):
        try:
            reflection = PlannerReflectionResult.model_validate(reflection_value)
        except Exception:
            reflection = PlannerReflectionResult()
    else:
        reflection = PlannerReflectionResult()
    validation = state.get("query_graph_validation_result") or GraphValidationResult()
    validation_gaps = [
        item
        for item in list(getattr(validation, "gaps", []) or [])
        if str(getattr(item, "code", "") or "") not in QUERY_GRAPH_REPAIR_OPERATIONAL_GAP_CODES
        and not planner_repair_request_projection_gap(item)
    ]
    reflection_issues = [
        issue
        for issue in reflection.issues
        if isinstance(issue, dict)
        and str(issue.get("severity") or "error").strip().lower() == "error"
    ]
    repair_requests = normalized_planner_repair_requests(
        state.get("planner_repair_requests") or reflection.repair_requests
    )
    payload = {
        "graphStructureFingerprint": query_graph_structure_fingerprint(plan or state.get("plan")),
        # When PlannerCritic has typed errors it owns the repair-budget
        # identity. Validator observations are still passed to Repair, but new
        # validator prose/codes cannot reset the same Critic scope.
        "issues": canonical_repair_issue_payloads(
            reflection_issues if reflection_issues else validation_gaps
        ),
        # A changed repair strategy deserves a fresh scoped attempt even when
        # the underlying critic issue is unchanged.  Exclude prose-only hints
        # and request reasons so harmless wording changes cannot reset budget.
        "repairRequests": canonical_repair_payloads(
            [planner_repair_request_scope_payload(request) for request in repair_requests]
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_repair_payloads(values: List[Any]) -> List[Any]:
    payloads: Dict[str, Any] = {}
    for value in values or []:
        if hasattr(value, "model_dump"):
            payload = value.model_dump(by_alias=True, mode="json")
        elif isinstance(value, dict):
            payload = dict(value)
        else:
            payload = {"value": str(value)}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        payloads.setdefault(encoded, payload)
    return [payloads[key] for key in sorted(payloads)]


def canonical_repair_issue_payloads(values: List[Any]) -> List[Dict[str, str]]:
    payloads: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in values or []:
        if hasattr(value, "model_dump"):
            raw = value.model_dump(by_alias=True, mode="json")
        elif isinstance(value, dict):
            raw = value
        else:
            continue
        payload = {
            "code": normalize_knowledge_request_text(raw.get("code") or ""),
            "taskId": normalize_knowledge_request_text(
                raw.get("taskId") or raw.get("task_id") or ""
            ),
        }
        identity = (payload["code"], payload["taskId"])
        if not payload["code"] or identity in seen:
            continue
        seen.add(identity)
        payloads.append(payload)
    return sorted(
        payloads,
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def planner_repair_request_scope_payload(request: PlannerRepairRequest) -> Dict[str, Any]:
    knowledge_requests = [
        canonical_knowledge_request_payload(item)
        for item in dedupe_knowledge_requests(request.knowledge_requests)
    ]
    return {
        "action": normalize_knowledge_request_text(request.action),
        "stage": normalize_knowledge_request_text(request.stage),
        "taskId": normalize_knowledge_request_text(request.task_id),
        "query": normalize_knowledge_request_text(request.query),
        "knowledgeRequests": canonical_repair_payloads(knowledge_requests),
    }


def planner_repair_request_projection_gap(value: Any) -> bool:
    evidence = str(getattr(value, "evidence", "") or "")
    if not evidence.startswith("{"):
        return False
    try:
        payload = json.loads(evidence)
    except Exception:
        return False
    return bool(
        isinstance(payload, dict)
        and ("action" in payload or "repairHints" in payload or "repair_hints" in payload)
        and ("stage" in payload or "knowledgeRequests" in payload or "knowledge_requests" in payload)
    )


def planner_repair_input_from_state(state: AgentState) -> PlannerRepairInput:
    reflection_value = state.get("planner_reflection") or PlannerReflectionResult()
    if isinstance(reflection_value, PlannerReflectionResult):
        reflection = reflection_value.model_copy(deep=True)
    elif isinstance(reflection_value, dict):
        try:
            reflection = PlannerReflectionResult.model_validate(reflection_value)
        except Exception:
            reflection = PlannerReflectionResult()
    else:
        reflection = PlannerReflectionResult()
    validation = state.get("query_graph_validation_result") or GraphValidationResult()
    requests = normalized_planner_repair_requests(
        state.get("planner_repair_requests") or reflection.repair_requests
    )
    return PlannerRepairInput(
        scope_key=planner_repair_scope_key(state),
        graph_fingerprint=query_graph_fingerprint(state.get("plan")),
        reflection=reflection,
        repair_requests=requests,
        validation_gaps=[item.model_copy(deep=True) for item in list(getattr(validation, "gaps", []) or [])],
        repair_gaps=[item.model_copy(deep=True) for item in planner_repair_gaps_from_state(state)],
    )


def query_graph_repair_exhaustion_gap(
    repair_input: PlannerRepairInput,
    scope_attempts: int,
) -> GraphValidationGap:
    issue_summaries: List[str] = []
    for issue in repair_input.reflection.issues:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or "PLANNER_CRITIC_ISSUE")
        detail = str(issue.get("reason") or issue.get("evidence") or "").strip()
        summary = "%s%s" % (code, (": " + detail) if detail else "")
        if summary not in issue_summaries:
            issue_summaries.append(summary)
    request_summaries: List[str] = []
    for request in repair_input.repair_requests:
        summary = "%s/%s%s" % (
            str(request.reason or "PLANNER_REPAIR_REQUEST"),
            str(request.action or "graph_repair"),
            (" task=" + str(request.task_id)) if request.task_id else "",
        )
        if summary not in request_summaries:
            request_summaries.append(summary)
    details = []
    if issue_summaries:
        details.append("issues=" + " | ".join(issue_summaries))
    if request_summaries:
        details.append("requests=" + " | ".join(request_summaries))
    if not details:
        details.append("issues=" + " | ".join(gap.code for gap in repair_input.repair_gaps if gap.code))
    task_ids = unique_workflow_strings(
        [
            *[str(issue.get("taskId") or issue.get("task_id") or "") for issue in repair_input.reflection.issues if isinstance(issue, dict)],
            *[str(item.task_id or "") for item in repair_input.repair_requests],
            *[str(item.task_id or "") for item in repair_input.repair_gaps],
        ]
    )
    evidence = {
        "scopeKey": repair_input.scope_key,
        "graphFingerprint": repair_input.graph_fingerprint,
        "scopeAttempts": int(scope_attempts),
        "reflectionIssues": repair_input.reflection.issues,
        "repairRequests": [item.model_dump(by_alias=True, mode="json") for item in repair_input.repair_requests],
    }
    return GraphValidationGap(
        code="QUERY_GRAPH_REPAIR_EXHAUSTED",
        task_id=task_ids[0] if len(task_ids) == 1 else "",
        reason=(
            "PlannerCritic repair made no executable QueryGraph progress after %d scoped attempts; %s"
            % (int(scope_attempts), "; ".join(details))
        ),
        evidence=json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
    )


def dedupe_graph_validation_gaps(gaps: List[GraphValidationGap]) -> List[GraphValidationGap]:
    deduped: List[GraphValidationGap] = []
    seen: set[tuple[str, str, str, str]] = set()
    for gap in gaps or []:
        identity = (
            str(gap.code or ""),
            str(gap.task_id or ""),
            str(gap.evidence or ""),
            str(gap.reason or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(gap.model_copy(deep=True))
    return deduped


def retained_query_graph_repair_gaps(repair_input: PlannerRepairInput) -> List[GraphValidationGap]:
    """Preserve original typed Critic/validator gaps across repair outcomes."""

    retained = [
        item
        for item in [*repair_input.validation_gaps, *repair_input.repair_gaps]
        if str(item.code or "") not in QUERY_GRAPH_REPAIR_OPERATIONAL_GAP_CODES
    ]
    return dedupe_graph_validation_gaps(retained)


def query_graph_repair_no_progress_gap(
    repair_input: PlannerRepairInput,
    scope_attempt: int,
) -> GraphValidationGap:
    return GraphValidationGap(
        code="QUERY_GRAPH_REPAIR_NO_PROGRESS",
        reason=(
            "Planner repair completed without changing executable QueryGraph structure; "
            "the original PlannerCritic gaps remain unresolved"
        ),
        evidence=json.dumps(
            {
                "scopeKey": repair_input.scope_key,
                "graphFingerprint": repair_input.graph_fingerprint,
                "scopeAttempt": int(scope_attempt),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def query_graph_repair_failure_gap(
    repair_input: PlannerRepairInput,
    error: Exception,
) -> GraphValidationGap:
    return GraphValidationGap(
        code="QUERY_GRAPH_REPAIR_FAILED",
        reason="Planner repair callback failed: %s: %s" % (
            type(error).__name__,
            str(error)[:300],
        ),
        evidence=json.dumps(
            {
                "scopeKey": repair_input.scope_key,
                "graphFingerprint": repair_input.graph_fingerprint,
                "errorType": type(error).__name__,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def query_graph_repair_awaiting_knowledge_gap(
    repair_input: PlannerRepairInput,
    requests: List[KnowledgeRequest],
) -> GraphValidationGap:
    task_ids = unique_workflow_strings([str(item.needed_for_task_id or "") for item in requests])
    return GraphValidationGap(
        code="QUERY_GRAPH_REPAIR_AWAITING_KNOWLEDGE",
        task_id=task_ids[0] if len(task_ids) == 1 else "",
        reason=(
            "Repair preserved the executable QueryGraph and requested supplemental governed knowledge; "
            "this is pending orchestration work, not a successful graph repair"
        ),
        evidence=json.dumps(
            {
                "scopeKey": repair_input.scope_key,
                "requests": [item.model_dump(by_alias=True, mode="json") for item in requests],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def new_query_graph_repair_knowledge_requests(
    before: QueryPlan,
    after: QueryPlan,
) -> List[KnowledgeRequest]:
    before_ids = {knowledge_request_identity(item) for item in before.knowledge_requests}
    return [
        item.model_copy(deep=True)
        for item in dedupe_knowledge_requests(after.knowledge_requests)
        if knowledge_request_identity(item) not in before_ids
    ]


def graph_gaps_from_node_failures(task_results: List[Any]) -> List[GraphValidationGap]:
    gaps: List[GraphValidationGap] = []
    for task_result in task_results:
        message = str(task_result.query_bundle.error or task_result.summary or "")
        matched = next((code for code in REPAIRABLE_QUERY_GRAPH_GAP_CODES if code in message), "")
        if not matched:
            continue
        if matched == "MISSING_UPSTREAM_ENTITY" and upstream_missing_is_execution_result(task_result):
            continue
        gaps.append(
            GraphValidationGap(
                code=matched,
                task_id=task_result.task_id,
                evidence=task_result.query_bundle.sql[:240],
                reason=message[:300],
            )
        )
    return gaps


def upstream_missing_is_execution_result(task_result: Any) -> bool:
    contract = getattr(task_result, "node_plan_contract", None)
    if not contract:
        return False
    for entity in getattr(contract, "upstream_entity_sets", []) or []:
        reason = ""
        if isinstance(entity, dict):
            reason = str(entity.get("missingReason") or entity.get("missing_reason") or "")
        if reason in {"UPSTREAM_SQL_FAILED", "UPSTREAM_ZERO_ROWS"}:
            return True
    return False


def enum_value(value: Any) -> str:
    return getattr(value, "value", value)


def relationship_topics_from_pack(pack: PlanningAssetPack) -> List[str]:
    topics: List[str] = []
    for relationship in pack.relationships:
        ref_id = str(relationship.source_ref_id or "")
        if not ref_id.startswith("semantic:"):
            continue
        parts = ref_id.split(":")
        if len(parts) >= 3 and parts[2] == "relationship" and parts[1] not in topics:
            topics.append(parts[1])
    return topics


def planner_provider_error(error: str) -> str:
    text = str(error or "")
    markers = ["timeout:", "provider_error:", "json_parse_error:"]
    return text if any(marker in text for marker in markers) else ""


def planner_degraded_state(error: str, plan: QueryPlan, reason: str = "") -> Dict[str, Any]:
    provider_error = planner_provider_error(error)
    failure_trace = [
        str(item)
        for item in plan.agent_trace or []
        if str(item).startswith("PLANNER_")
        or "after_llm_failure" in str(item)
        or "validated_after_llm_failure" in str(item)
        or "failure_fallback" in str(item)
    ]
    failure_code = planner_failure_gap_code(plan)
    if not provider_error and failure_code and failure_trace:
        # Exceptions caught around the provider call may not mutate the
        # adapter's last_error. Preserve the structured Planner observation.
        provider_error = failure_trace[0]
    if not provider_error or (plan.intents and not failure_trace):
        return {}
    understanding = plan.question_understanding or {}
    allow_expensive_recovery = bool(
        understanding.get("allowDegradedHypothesisExploration")
        or understanding.get("allow_degraded_hypothesis_exploration")
    )
    if failure_code:
        code = failure_code
    elif "timeout:" in provider_error:
        code = "PLANNER_LLM_TIMEOUT"
    elif "provider_error:" in provider_error:
        code = "PLANNER_PROVIDER_ERROR"
    else:
        code = "PLANNER_RESPONSE_INVALID"
    coverage_rejected = any("coverage_rejected" in item or "fail_closed_coverage" in item for item in failure_trace)
    return {
        "active": True,
        "stage": "planner",
        "code": code,
        "reason": provider_error,
        "timeout": code == "PLANNER_LLM_TIMEOUT",
        "fallbackUsed": bool(plan.intents),
        "fallbackCoveragePassed": bool(plan.intents) and not coverage_rejected,
        "stopExpensivePostProcessing": not allow_expensive_recovery,
        "allowDegradedHypothesisExploration": allow_expensive_recovery,
        "plannerReason": reason,
        "trace": failure_trace[:12],
    }


def append_active_planner_degraded_reason(
    state: AgentState,
    run_result: AgentRunResult,
) -> None:
    """Project the active Planner operational state into execution evidence."""

    raw = state.get("planner_degraded") or {}
    if not isinstance(raw, dict) or not raw.get("active"):
        return
    payload = json.loads(json.dumps(raw, ensure_ascii=False, default=str))
    payload.setdefault("stage", "planner")
    reasons = [dict(item) for item in run_result.degraded_reasons or [] if isinstance(item, dict)]
    identity = planner_degraded_reason_identity(payload)
    if not any(planner_degraded_reason_identity(item) == identity for item in reasons):
        reasons.append(payload)
    run_result.degraded_reasons = reasons


def memory_recall_issues_from_state(state: AgentState) -> List[RetrievalIssue]:
    raw_items = list(state.get("memory_recall_issues") or [])
    if not raw_items:
        raw_items = list((state.get("memory_injection_trace") or {}).get("issues") or [])
    return normalize_memory_recall_issues(raw_items)


def memory_recall_health_payload(state: AgentState) -> Dict[str, Any]:
    trace = dict(state.get("memory_injection_trace") or {})
    issues = memory_recall_issues_from_state(state)
    return {
        "status": str(state.get("memory_recall_status") or trace.get("status") or "not_started"),
        "usableSnapshot": bool(trace.get("usableSnapshot")),
        "selectedIds": list(trace.get("selectedIds") or []),
        "issues": [issue.model_dump(by_alias=True) for issue in issues],
        "enrichmentStatus": dict(trace.get("enrichmentStatus") or {}),
        "contextFingerprint": str(trace.get("contextFingerprint") or ""),
    }


def append_memory_recall_issue_to_state(
    state: AgentState,
    issue: RetrievalIssue,
    enrichment_key: str = "",
) -> None:
    issues = normalize_memory_recall_issues([*memory_recall_issues_from_state(state), issue])
    previous = dict(state.get("memory_injection_trace") or {})
    enrichment = dict(previous.get("enrichmentStatus") or {})
    if enrichment_key:
        enrichment[enrichment_key] = "failed"
    trace = memory_recall_trace_for(
        state.get("memory_injection") or {},
        issues,
        enrichment_status=enrichment,
    )
    for key in ["contextFingerprint", "selectedIds", "candidateIds", "candidateCount"]:
        if key in previous and key not in trace:
            trace[key] = previous[key]
    state["memory_injection_trace"] = trace
    state["memory_recall_status"] = str(trace.get("status") or "failed")
    state["memory_recall_issues"] = list(trace.get("issues") or [])
    if state.get("memory_injection"):
        state["memory_injection"]["memoryInjectionTrace"] = dict(trace)


def append_memory_recall_outcomes_to_run_result(
    state: AgentState,
    run_result: AgentRunResult,
) -> None:
    """Keep memory availability and degradation monotonic through answering."""

    status = str(
        state.get("memory_recall_status")
        or (state.get("memory_injection_trace") or {}).get("status")
        or "not_started"
    )
    issues = memory_recall_issues_from_state(state)
    if status not in {"failed", "degraded"} and not issues:
        return
    if status == "failed" and not issues:
        issues = [
            RetrievalIssue(
                code="MEMORY_RECALL_FAILED",
                message="Memory recall failed without a structured operational issue",
                backend="unknown",
                lane="primary",
                stage="acquire",
                severity="blocking",
                resolved=False,
            )
        ]
    reasons = [dict(item) for item in run_result.degraded_reasons or [] if isinstance(item, dict)]
    gaps = [item.model_copy(deep=True) for item in run_result.evidence_gaps or []]
    reason_ids = {
        (str(item.get("stage") or ""), str(item.get("code") or ""), str(item.get("lane") or ""))
        for item in reasons
    }
    gap_ids = {
        (str(item.code or item.gap_code), str(item.source_node_id or item.task_id))
        for item in gaps
    }
    for issue in issues:
        severity = (
            "blocking"
            if status == "failed" and not issue.resolved
            else "warning"
        )
        details = {
            **dict(issue.details or {}),
            "memoryRecallStatus": status,
            "backend": issue.backend,
            "lane": issue.lane,
            "stage": issue.stage,
            "retryable": issue.retryable,
            "fallbackUsed": issue.fallback_used,
            "resolved": issue.resolved,
        }
        reason_identity = ("memory_recall", issue.code, issue.lane)
        if reason_identity not in reason_ids:
            reasons.append(
                {
                    "active": True,
                    "stage": "memory_recall",
                    "code": issue.code or "MEMORY_RECALL_FAILED",
                    "reason": issue.message[:1000] or issue.code,
                    "status": status,
                    "severity": severity,
                    "backend": issue.backend,
                    "lane": issue.lane,
                    "resolved": issue.resolved,
                    "details": details,
                }
            )
            reason_ids.add(reason_identity)
        if details.get("answerImpact") is False:
            continue
        code = issue.code or "MEMORY_RECALL_FAILED"
        source_node_id = "memory:%s:%s" % (issue.backend or "unknown", issue.lane or "unknown")
        gap_identity = (code, source_node_id)
        if gap_identity in gap_ids:
            continue
        gaps.append(
            EvidenceGap(
                code=code,
                source_node_id=source_node_id,
                evidence=issue.backend,
                reason=issue.message[:1000] or code,
                severity=severity,
                disclosure_required=True,
                source="memory_recall",
                answer_instruction=(
                    "本轮长期记忆不可用，不能声称已应用历史偏好或纠正规则；恢复召回后再完成受历史记忆约束的回答。"
                    if severity == "blocking"
                    else "说明长期记忆使用了降级来源，本轮结论以当前问题和已发布语义资产为准。"
                ),
                suggested_action="retry_memory_recall" if severity == "blocking" else "answer_with_memory_disclosure",
                details=details,
            )
        )
        gap_ids.add(gap_identity)
    run_result.degraded_reasons = reasons
    run_result.evidence_gaps = gaps


def retrieval_issues_from_state(state: AgentState) -> List[RetrievalIssue]:
    raw_items = list(state.get("knowledge_retrieval_issues") or [])
    if not raw_items:
        raw_items = list((state.get("knowledge_bundle") or KnowledgeBundle()).retrieval_issues or [])
    issues: List[RetrievalIssue] = []
    for raw in raw_items:
        try:
            issues.append(raw if isinstance(raw, RetrievalIssue) else RetrievalIssue.model_validate(raw))
        except Exception:
            continue
    return dedupe_retrieval_issues(issues)


def knowledge_retrieval_health_payload(state: AgentState) -> Dict[str, Any]:
    bundle = normalize_knowledge_bundle_status(state.get("knowledge_bundle") or KnowledgeBundle())
    status = str(state.get("knowledge_retrieval_status") or bundle.retrieval_status or "not_started")
    issues = retrieval_issues_from_state(state)
    return {
        "status": status,
        "backend": bundle.backend,
        "sourceRefCount": len(bundle.source_refs),
        "issueCount": len(issues),
        "issues": [issue.model_dump(by_alias=True) for issue in issues],
        "outcomes": list(state.get("knowledge_retrieval_outcomes") or []),
    }


def knowledge_retrieval_planning_gaps(state: AgentState) -> List[Dict[str, Any]]:
    """Expose unavailable retrieval as planner input, never as a zero match."""

    status = str(state.get("knowledge_retrieval_status") or "")
    outcomes = list(state.get("knowledge_retrieval_outcomes") or [])
    gaps: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for issue in retrieval_issues_from_state(state):
        if issue.resolved:
            continue
        if status != "failed" and str(issue.severity or "").lower() != "blocking":
            continue
        query = next(
            (
                str(outcome.get("query") or "")
                for outcome in outcomes
                if (
                    not issue.request_key
                    or str(outcome.get("requestKey") or "") == issue.request_key
                )
                and (
                    not issue.stage
                    or str(outcome.get("stage") or "") == issue.stage
                    or issue.stage in {"retrieve", "fallback"}
                )
            ),
            "",
        )
        identity = (issue.code, issue.request_key, issue.backend, issue.lane)
        if identity in seen:
            continue
        seen.add(identity)
        gaps.append(
            {
                "code": issue.code or "KNOWLEDGE_RETRIEVAL_FAILED",
                "requestKey": issue.request_key,
                "type": "retrieval_failure",
                "query": query[:500],
                "reason": issue.message[:500] or "Knowledge retrieval was unavailable",
                "severity": "blocking",
                "backend": issue.backend,
                "lane": issue.lane,
            }
        )
    if status == "failed" and not gaps:
        gaps.append(
            {
                "code": "KNOWLEDGE_RETRIEVAL_FAILED",
                "requestKey": "",
                "type": "retrieval_failure",
                "query": "",
                "reason": "Knowledge retrieval failed without a structured backend issue",
                "severity": "blocking",
                "backend": (state.get("knowledge_bundle") or KnowledgeBundle()).backend,
                "lane": "unknown",
            }
        )
    return gaps


def append_knowledge_retrieval_outcomes_to_run_result(
    state: AgentState,
    run_result: AgentRunResult,
) -> None:
    """Keep retrieval availability monotonic through planning and answering."""

    status = str(state.get("knowledge_retrieval_status") or "")
    issues = retrieval_issues_from_state(state)
    if status not in {"failed", "degraded"} and not issues:
        return
    if status == "failed" and not issues:
        issues = [
            RetrievalIssue(
                code="KNOWLEDGE_RETRIEVAL_FAILED",
                message="Knowledge retrieval failed without a structured backend issue",
                backend=(state.get("knowledge_bundle") or KnowledgeBundle()).backend,
                lane="unknown",
                stage="retrieve",
                severity="blocking",
            )
        ]
    reasons = [dict(item) for item in run_result.degraded_reasons or [] if isinstance(item, dict)]
    evidence_gaps = [item.model_copy(deep=True) for item in run_result.evidence_gaps or []]
    reason_ids = {
        (
            str(item.get("stage") or ""),
            str(item.get("code") or ""),
            str(item.get("backend") or ""),
            str(item.get("lane") or ""),
            str(item.get("requestKey") or ""),
        )
        for item in reasons
    }
    gap_ids = {
        (
            str(item.code or item.gap_code),
            str(item.source_node_id or item.task_id),
            str((item.details or {}).get("requestKey") or ""),
        )
        for item in evidence_gaps
    }
    for issue in issues:
        severity = (
            "warning"
            if issue.resolved or (status == "degraded" and str(issue.severity or "").lower() != "blocking")
            else "blocking"
        )
        code = issue.code or "KNOWLEDGE_RETRIEVAL_FAILED"
        source_node_id = "retrieval:%s:%s" % (issue.backend or "unknown", issue.lane or "unknown")
        details = {
            **dict(issue.details or {}),
            "retrievalStatus": status,
            "backend": issue.backend,
            "lane": issue.lane,
            "stage": issue.stage,
            "requestKey": issue.request_key,
            "retryable": issue.retryable,
            "fallbackUsed": issue.fallback_used,
            "resolved": issue.resolved,
        }
        reason_identity = ("retrieval", code, issue.backend, issue.lane, issue.request_key)
        if reason_identity not in reason_ids:
            reasons.append(
                {
                    "active": True,
                    "stage": "retrieval",
                    "code": code,
                    "reason": issue.message[:1000] or code,
                    "status": status,
                    "severity": severity,
                    "backend": issue.backend,
                    "lane": issue.lane,
                    "requestKey": issue.request_key,
                    "fallbackUsed": issue.fallback_used,
                    "resolved": issue.resolved,
                    "details": details,
                }
            )
            reason_ids.add(reason_identity)
        gap_identity = (code, source_node_id, issue.request_key)
        if gap_identity in gap_ids:
            continue
        evidence_gaps.append(
            EvidenceGap(
                code=code,
                source_node_id=source_node_id,
                evidence=issue.backend,
                reason=issue.message[:1000] or code,
                severity=severity,
                disclosure_required=True,
                source="retrieval",
                answer_instruction=(
                    "说明召回后端或检索通道不可用，本轮不能把空候选解释为业务上不存在；先恢复召回再完成回答。"
                    if severity == "blocking"
                    else "说明主召回或部分检索通道失败，当前答案使用了可用的降级证据来源。"
                ),
                suggested_action="retry_knowledge_retrieval" if severity == "blocking" else "answer_with_retrieval_disclosure",
                details=details,
            )
        )
        gap_ids.add(gap_identity)
    run_result.degraded_reasons = reasons
    run_result.evidence_gaps = evidence_gaps


def append_subagent_outcomes_to_run_result(
    state: AgentState,
    run_result: AgentRunResult,
) -> None:
    """Keep logical Sub-Agent degradation monotonic through later execution.

    Delegation may happen before or after SQL execution, and SQL execution may
    replace the current ``AgentRunResult``.  Re-projecting the stored contracts
    at the evidence and response boundaries prevents a useful partial summary
    from being mistaken for a completed analysis or silently disappearing.
    The projection is task-kind agnostic: status and structured gaps are the
    authority.
    """

    contracts = [
        dict(item)
        for item in state.get("subagent_delegation_results") or []
        if isinstance(item, dict)
    ]
    if not contracts:
        return
    reasons = [dict(item) for item in run_result.degraded_reasons or [] if isinstance(item, dict)]
    evidence_gaps = [item.model_copy(deep=True) for item in run_result.evidence_gaps or []]
    projected_gaps: List[EvidenceGap] = []
    for contract_index, contract in enumerate(contracts):
        status = str(contract.get("status") or "failed").strip().lower()
        raw_gaps = [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else dict(item)
            for item in contract.get("gaps") or []
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
        if status == "completed" and not raw_gaps:
            continue
        if not raw_gaps:
            raw_gaps = [
                {
                    "code": "SUBAGENT_PARTIAL" if status == "partial" else "SUBAGENT_FAILED",
                    "message": str(contract.get("summary") or "Sub-Agent did not complete")[:1000],
                }
            ]
        task_kind = str(contract.get("taskKind") or contract.get("task_kind") or "subagent")
        objective = str(contract.get("objective") or "")
        for gap_index, raw_gap in enumerate(raw_gaps):
            code = str(raw_gap.get("code") or "SUBAGENT_DEGRADED").strip() or "SUBAGENT_DEGRADED"
            reason = str(raw_gap.get("reason") or raw_gap.get("message") or contract.get("summary") or code).strip()
            severity = str(raw_gap.get("severity") or "warning").strip().lower()
            if severity not in {"blocking", "warning", "info"}:
                severity = "warning"
            source_node_id = "%s:%d" % (task_kind, contract_index)
            details = {
                **dict(raw_gap.get("details") or {}),
                "taskKind": task_kind,
                "outcomeStatus": status,
                "objective": objective[:500],
                "contractIndex": contract_index,
                "gapIndex": gap_index,
            }
            degraded = {
                "active": True,
                "stage": "subagent",
                "source": "subagent",
                "taskKind": task_kind,
                "status": status,
                "code": code,
                "reason": reason[:1000],
                "fallbackUsed": bool((contract.get("payload") or {}).get("fallbackUsed")),
                "summary": str(contract.get("summary") or "")[:1000],
                "details": details,
            }
            degraded_identity = (
                str(degraded["stage"]),
                task_kind,
                code,
                str(degraded["reason"]),
            )
            if not any(
                (
                    str(item.get("stage") or ""),
                    str(item.get("taskKind") or item.get("task_kind") or ""),
                    str(item.get("code") or ""),
                    str(item.get("reason") or ""),
                )
                == degraded_identity
                for item in reasons
            ):
                reasons.append(degraded)
            projected_gaps.append(
                EvidenceGap(
                    code=code,
                    source_node_id=source_node_id,
                    evidence=task_kind,
                    reason=reason[:1000],
                    severity=severity,
                    disclosure_required=bool(raw_gap.get("disclosureRequired", True)),
                    source="subagent",
                    answer_instruction=str(
                        raw_gap.get("answerInstruction")
                        or raw_gap.get("answer_instruction")
                        or "说明 Sub-Agent 未完整执行，降级或部分输出不能表述为已完成的分析。"
                    ),
                    suggested_action=str(raw_gap.get("suggestedAction") or "answer_with_subagent_gap"),
                    details=details,
                )
            )
    if not projected_gaps:
        return
    existing_gap_ids = {
        (
            str(item.gap_code or item.code),
            str(item.source_node_id or item.task_id),
            str(item.reason or ""),
        )
        for item in evidence_gaps
    }
    for gap in projected_gaps:
        identity = (str(gap.gap_code or gap.code), str(gap.source_node_id or gap.task_id), str(gap.reason or ""))
        if identity not in existing_gap_ids:
            evidence_gaps.append(gap)
            existing_gap_ids.add(identity)
    run_result.degraded_reasons = reasons
    run_result.evidence_gaps = evidence_gaps

    verified = run_result.verified_evidence
    verified_gap_ids = {
        (
            str(item.gap_code or item.code),
            str(item.source_node_id or item.task_id),
            str(item.reason or ""),
        )
        for item in verified.gaps or []
    }
    verified_gaps = [item.model_copy(deep=True) for item in verified.gaps or []]
    for gap in projected_gaps:
        identity = (str(gap.gap_code or gap.code), str(gap.source_node_id or gap.task_id), str(gap.reason or ""))
        if identity not in verified_gap_ids:
            verified_gaps.append(gap.model_copy(deep=True))
            verified_gap_ids.add(identity)
    blocking_gaps = [item for item in verified_gaps if item.severity == "blocking"]
    warning_gaps = [item for item in verified_gaps if item.severity == "warning"]
    verified.gaps = verified_gaps
    verified.blocking_gaps = blocking_gaps
    verified.warning_gaps = warning_gaps
    verified.passed = bool(verified.passed and not blocking_gaps)
    verified.answer_guard_required = bool(
        verified.answer_guard_required
        or blocking_gaps
        or warning_gaps
        or any(item.disclosure_required for item in projected_gaps)
    )
    if blocking_gaps and not verified.partial_answer_reason:
        verified.partial_answer_reason = "；".join(item.reason for item in blocking_gaps[:3])
    if blocking_gaps and not run_result.partial_answer_reason:
        run_result.partial_answer_reason = verified.partial_answer_reason


def planner_degraded_reason_identity(payload: Dict[str, Any]) -> tuple[str, str, str]:
    stage = str(payload.get("stage") or "planner").strip().lower()
    code = str(payload.get("code") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    if code or reason:
        return stage, code, reason
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return stage, "", canonical


def planner_degraded_stops_expensive_work(state: AgentState) -> bool:
    degraded = state.get("planner_degraded") or {}
    return bool(degraded.get("active") and degraded.get("stopExpensivePostProcessing", True))


def validation_with_question_coverage(
    question: str,
    plan: QueryPlan,
    asset_pack: PlanningAssetPack,
    validation: GraphValidationResult,
) -> GraphValidationResult:
    coverage_gaps = query_plan_question_coverage_gaps(question, plan, asset_pack)
    if not coverage_gaps:
        return validation
    gaps = list(validation.gaps or [])
    seen = {(gap.code, gap.task_id, gap.evidence) for gap in gaps}
    for gap in coverage_gaps:
        identity = (gap.code, gap.task_id, gap.evidence)
        if identity not in seen:
            gaps.append(gap)
            seen.add(identity)
    return validation.model_copy(
        update={
            "valid": False,
            "gaps": gaps,
            "repairable": False,
        }
    )


def metric_resolutions_for_debug(plan: QueryPlan) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for intent in plan.intents:
        if intent.metric_resolution:
            items.append(intent.metric_resolution)
    for contract in plan.evidence_contracts:
        resolution = contract.get("metricResolution") or contract.get("metric_resolution")
        if isinstance(resolution, dict):
            items.append(resolution)
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = "%s:%s:%s" % (item.get("requestedMetricRef"), item.get("ownerTable"), item.get("metricKey"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def legacy_agent_trace_from_actions(state: AgentState) -> List[str]:
    traces: List[str] = []
    for item in state.get("action_history", []) or []:
        action = str(getattr(item, "action", "") or "")
        node = str(getattr(item, "node", "") or "")
        status = str(getattr(item, "status", "") or "")
        reason = str(getattr(item, "reason", "") or "")
        if not action and not node:
            continue
        label = action or node
        if node and node != action:
            label = "%s:%s" % (label, node)
        if status:
            label = "%s:%s" % (label, status)
        if reason:
            label = "%s - %s" % (label, reason)
        traces.append(label)
    return traces


def answer_guard_debug(run_result: AgentRunResult) -> Dict[str, Any]:
    verified = run_result.verified_evidence if run_result else None
    if not verified:
        return {
            "required": False,
            "requiredDisclosures": [],
            "blockingGapCodes": [],
            "warningGapCodes": [],
            "claimVerification": {},
        }
    return {
        "required": verified.answer_guard_required,
        "requiredDisclosures": verified.required_disclosures,
        "blockingGapCodes": [gap.code for gap in verified.blocking_gaps],
        "warningGapCodes": [gap.code for gap in verified.warning_gaps],
        "verifiedFactCount": len(run_result.verified_facts or []),
        "claimVerification": run_result.answer_claim_verification.model_dump(by_alias=True),
    }


def rule_recall_item(item: Any) -> bool:
    answer_mode = str(getattr(item, "answer_mode", "") or "").upper()
    source_type = str(getattr(item, "source_type", "") or "").upper()
    metadata = getattr(item, "metadata", {}) if isinstance(getattr(item, "metadata", {}), dict) else {}
    declared_role = str(
        metadata.get("knowledgeRole")
        or metadata.get("topicRole")
        or metadata.get("routingRole")
        or metadata.get("semanticKind")
        or ""
    ).upper()
    answer_modes = {part.strip() for part in split_on_characters(answer_mode, ",|/") if part.strip()}
    if answer_modes & {"RULE", "RULE_ANSWER", "RULE_REFERENCE"}:
        return True
    return source_type == "GOVERNED_RULE" or declared_role in {"RULE", "GOVERNED_RULE", "RULE_REFERENCE"}


def merge_recall_item_queries(current: RecallItem, incoming: RecallItem) -> RecallItem:
    current_metadata = dict(current.metadata or {})
    incoming_metadata = dict(incoming.metadata or {})
    queries: List[str] = []
    for metadata in [current_metadata, incoming_metadata]:
        for query in metadata.get("recallQueries") or []:
            query_text = str(query or "")
            if query_text and query_text not in queries:
                queries.append(query_text)
        query_text = str(metadata.get("recallQuery") or "")
        if query_text and query_text not in queries:
            queries.append(query_text)
    base = incoming if recall_item_sort_key(incoming) >= recall_item_sort_key(current) else current
    other = current if base is incoming else incoming
    merged_metadata = {**dict(other.metadata or {}), **dict(base.metadata or {}), "recallQueries": queries}
    if queries:
        merged_metadata["recallQuery"] = queries[-1]
    return base.model_copy(update={"metadata": merged_metadata})


def normalize_knowledge_request_text(value: str) -> str:
    return normalize_canonical_knowledge_request_text(value)


def knowledge_request_key(request: KnowledgeRequest) -> str:
    return knowledge_request_identity(request)


def recall_item_queries(item: RecallItem) -> List[str]:
    metadata = item.metadata or {}
    queries: List[str] = []
    for query in metadata.get("recallQueries") or []:
        query_text = str(query or "")
        if query_text and query_text not in queries:
            queries.append(query_text)
    query_text = str(metadata.get("recallQuery") or "")
    if query_text and query_text not in queries:
        queries.append(query_text)
    return queries


def knowledge_request_recall_fingerprint(items: List[RecallItem], request: KnowledgeRequest) -> str:
    request_query = normalize_knowledge_request_text(request.query)
    records: List[Dict[str, Any]] = []
    for item in items:
        queries = recall_item_queries(item)
        normalized_queries = [normalize_knowledge_request_text(query) for query in queries]
        if request_query and request_query not in normalized_queries:
            continue
        records.append(
            {
                "docId": item.doc_id,
                "sourceType": item.source_type,
                "title": item.title,
                "fusionScore": round(float(item.fusion_score or 0.0), 4),
                "semanticRefId": str((item.metadata or {}).get("semanticRefId") or ""),
            }
        )
    records.sort(key=lambda item: (str(item.get("docId") or ""), str(item.get("semanticRefId") or "")))
    return json.dumps(records, ensure_ascii=False, sort_keys=True)


def append_knowledge_request_gaps(
    current: List[Dict[str, Any]],
    requests: List[KnowledgeRequest],
    code: str,
) -> List[Dict[str, Any]]:
    merged = list(current or [])
    seen = {
        "%s|%s" % (str(item.get("code") or ""), str(item.get("requestKey") or ""))
        for item in merged
        if isinstance(item, dict)
    }
    for request in requests:
        key = knowledge_request_key(request)
        identity = "%s|%s" % (code, key)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(
            {
                "code": code,
                "requestKey": key,
                "type": str(request.type or ""),
                "query": request.query,
                "neededForTaskId": request.needed_for_task_id,
                "reason": request.reason,
            }
        )
    return merged


def knowledge_request_gap_codes_by_key(gaps: List[Dict[str, Any]]) -> Dict[str, str]:
    codes: Dict[str, str] = {}
    for item in gaps or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("requestKey") or "")
        code = str(item.get("code") or "")
        if key and code and key not in codes:
            codes[key] = code
    return codes


def filter_blocked_knowledge_requests(
    state: AgentState,
    requests: List[KnowledgeRequest],
) -> List[KnowledgeRequest]:
    deduped = dedupe_workflow_knowledge_requests(list(requests or []))
    blocked_keys = set(state.get("blocked_knowledge_request_keys") or [])
    blocked_requests = [request for request in deduped if knowledge_request_key(request) in blocked_keys]
    if blocked_requests:
        blocked_gap_codes = knowledge_request_gap_codes_by_key(state.get("knowledge_request_gaps") or [])
        gaps = state.get("knowledge_request_gaps", [])
        by_code: Dict[str, List[KnowledgeRequest]] = {}
        for request in blocked_requests:
            by_code.setdefault(blocked_gap_codes.get(knowledge_request_key(request)) or "METRIC_EVIDENCE_UNCHANGED", []).append(request)
        for code, code_requests in by_code.items():
            gaps = append_knowledge_request_gaps(gaps, code_requests, code)
        state["knowledge_request_gaps"] = gaps
    return [request for request in deduped if knowledge_request_key(request) not in blocked_keys]


def answer_safe_memory_injection(injection: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(injection or {})
    safe.pop("relevantPastCases", None)
    safe.pop("relevantProcedures", None)
    safe.pop("quarantinedMemories", None)
    safe.pop("candidateMemories", None)  # pre-boundary persisted snapshots
    trace = dict(safe.get("memoryInjectionTrace") or {})
    if trace:
        trace.pop("candidates", None)
        safe["memoryInjectionTrace"] = trace
    return safe


def context_memory_ids(state: AgentState) -> List[str]:
    trace = state.get("memory_injection_trace") or (state.get("memory_injection") or {}).get("memoryInjectionTrace") or {}
    ids: List[str] = []
    for raw in list(trace.get("selectedIds") or []) + list(trace.get("candidateIds") or []):
        text = str(raw or "").strip()
        if text and text not in ids:
            ids.append(text)
    return ids


def context_semantic_ref_ids(state: AgentState) -> List[str]:
    refs: List[str] = []
    bundle = state.get("recall_bundle") or RecallBundle()
    for item in getattr(bundle, "items", []) or []:
        metadata = item.metadata or {}
        ref = str(metadata.get("semanticRefId") or item.doc_id or "")
        if ref:
            refs.append(ref)
    plan = state.get("plan") or QueryPlan()
    for intent in plan.intents:
        refs.extend([str(ref) for ref in intent.knowledge_ref_ids if ref])
        resolution = intent.metric_resolution or {}
        ref = str(resolution.get("semanticRefId") or resolution.get("semantic_ref_id") or "")
        if ref:
            refs.append(ref)
    pack = state.get("planning_asset_pack") or PlanningAssetPack()
    for ref_id, item in list((pack.source_refs or {}).items())[:80]:
        metadata = item.metadata or {}
        refs.append(str(metadata.get("semanticRefId") or ref_id or item.doc_id or ""))
    return unique_workflow_strings(refs)


def recall_knowledge_ref_ids(state: AgentState) -> set[str]:
    """Return only references actually present in this run's recall result."""
    refs: List[str] = []
    bundle = state.get("recall_bundle") or RecallBundle()
    for item in getattr(bundle, "items", []) or []:
        metadata = item.metadata or {}
        for raw in [
            item.doc_id,
            metadata.get("semanticRefId"),
            metadata.get("semantic_ref_id"),
            metadata.get("knowledgeRefId"),
            metadata.get("knowledge_ref_id"),
        ]:
            value = str(raw or "").strip()
            if value:
                refs.append(value)
    return set(unique_workflow_strings(refs))


def latest_context_budget_report(state: AgentState, stage: str) -> Dict[str, Any]:
    for item in reversed(state.get("context_budget_reports") or []):
        payload = item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
        if isinstance(payload, dict) and (not stage or str(payload.get("stage") or "") == stage):
            return payload
    return {}


def observability_summary(state: AgentState) -> Dict[str, Any]:
    validation = state.get("query_graph_validation_result") or GraphValidationResult()
    run_result = state.get("agent_run_result") or AgentRunResult()
    active_manifest = state.get("active_context_manifest") or {}
    repair_delta = state.get("last_query_graph_repair_delta") or QueryGraphRepairDelta()
    return {
        "selectedMemoryIds": context_memory_ids(state),
        "semanticRefIds": context_semantic_ref_ids(state)[:40],
        "contextHash": str(active_manifest.get("contextHash") or ""),
        "contextPackageId": str(active_manifest.get("contextPackageId") or ""),
        "validationGaps": [
            gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap
            for gap in getattr(validation, "gaps", [])[:12]
        ],
        "evidenceGaps": [
            gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap
            for gap in getattr(run_result, "evidence_gaps", [])[:12]
        ],
        "repairCount": int(state.get("query_graph_repair_attempts") or 0),
        "repairScopeAttemptCount": int(state.get("query_graph_repair_scope_attempt_count") or 0),
        "repairExhausted": bool(state.get("query_graph_repair_exhausted")),
        "lastRepairDelta": repair_delta.model_dump(by_alias=True) if getattr(repair_delta, "attempt", 0) else {},
    }


def unique_workflow_strings(values: List[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def previous_user_question(messages: List[Any], current_question: str = "") -> str:
    current = str(current_question or "").strip()
    for message in reversed(list(messages or [])):
        role = str(getattr(message, "role", "") or (message.get("role") if isinstance(message, dict) else "")).strip().lower()
        text = str(getattr(message, "text", "") or (message.get("text") if isinstance(message, dict) else "")).strip()
        if role == "user" and text and text != current:
            return text[:1200]
    return ""


def planner_conversation_context(state: AgentState) -> Dict[str, Any]:
    thread_context = state.get("thread_context") or {}
    current = str(state.get("question") or "").strip()
    recent_messages: List[Dict[str, str]] = []
    for item in state.get("message_history") or []:
        role = str(getattr(item, "role", "") or (item.get("role") if isinstance(item, dict) else "")).strip().lower()
        text = str(getattr(item, "text", "") or (item.get("text") if isinstance(item, dict) else "")).strip()
        if role not in {"user", "assistant"} or not text:
            continue
        recent_messages.append({"role": role, "text": text[:800]})
    for index in range(len(recent_messages) - 1, -1, -1):
        if recent_messages[index]["role"] == "user" and recent_messages[index]["text"] == current:
            recent_messages.pop(index)
            break
    return {
        "trust": "untrusted_conversation_data",
        "previousQuestion": str(thread_context.get("previousQuestion") or "")[:600],
        "previousAnswerPreview": str(thread_context.get("previousAnswerPreview") or "")[:800],
        "previousSummary": str(thread_context.get("previousSummary") or "")[:1000],
        "recentMessages": recent_messages[-6:],
    }


def remaining_run_budget_seconds(state: AgentState, settings: Settings) -> float:
    now = now_ms()
    started = int(state.get("run_started_at_ms") or now)
    elapsed_seconds = max(0, now - started) / 1000.0
    limit_seconds = int(getattr(settings, "run_budget_max_duration_seconds", 90) or 90)
    latency = state.get("latency_optimization") or {}
    if latency.get("eligible") and str(latency.get("mode") or "").startswith("fast_path"):
        limit_seconds = int(getattr(settings, "run_budget_fast_duration_seconds", 25) or 25)
    return max(0.0, float(limit_seconds) - elapsed_seconds)


def classify_lead_llm_failure(error: str) -> tuple[str, bool]:
    """Map the shared transport classification onto Lead's typed error codes."""

    classified = classify_llm_failure(error)
    code = {
        "TIMEOUT": "LEAD_LLM_TIMEOUT",
        "EMPTY_RESPONSE": "LEAD_LLM_EMPTY_RESPONSE",
        "PROVIDER_ERROR": "LEAD_LLM_PROVIDER_ERROR",
    }.get(classified.kind, "LEAD_LLM_FAILED")
    return code, classified.retryable


def lead_llm_failure_reason(error_code: str) -> str:
    return {
        "LEAD_LLM_TIMEOUT": "lead_action_llm_timeout",
        "LEAD_LLM_EMPTY_RESPONSE": "lead_action_llm_empty_response",
        "LEAD_LLM_PROVIDER_ERROR": "lead_action_llm_provider_error",
    }.get(str(error_code or ""), "lead_action_llm_failed")


def lead_decision_fingerprint(state: AgentState, allowed: List[str]) -> str:
    validation = state.get("query_graph_validation_result") or GraphValidationResult()
    run_result = state.get("agent_run_result") or AgentRunResult()
    last_action = state.get("last_action_result") or ActionResult()
    payload = {
        "allowedActions": sorted(str(item) for item in allowed if item),
        "validationGapCodes": sorted(str(item.code) for item in (validation.gaps or [])),
        "evidenceGapCodes": sorted(str(item.code) for item in (run_result.evidence_gaps or [])),
        "pendingKnowledgeCount": len(state.get("pending_knowledge_requests") or []),
        "repairRequestCount": len(state.get("planner_repair_requests") or []),
            "flags": {
                "fastMetricAttempted": bool(state.get("fast_metric_attempted")),
                "graphValidated": bool(state.get("query_graph_validated")),
                "evidenceVerified": bool(state.get("evidence_graph_verified")),
                "evidenceAccepted": bool(evidence_accepted_for_state(state)),
                "hypothesisExplored": bool(state.get("hypothesis_exploration_completed")),
                "analysisWorkerCompleted": bool(state.get("analysis_worker_completed")),
                "skillWorkerCompleted": bool(state.get("skill_worker_completed")),
            },
        "lastAction": {
            "action": last_action.action,
            "status": last_action.status,
            "message": last_action.message,
        },
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def action_result_payload(value: Any) -> Dict[str, Any]:
    """Serialize the latest ActionContract result for response and replay audit."""

    result = value or ActionResult()
    if hasattr(result, "model_dump"):
        return result.model_dump(by_alias=True)
    return dict(result) if isinstance(result, dict) else {}


def namespace_query_plan(plan: QueryPlan, namespace: str) -> QueryPlan:
    if not plan.intents:
        return plan
    mapping: Dict[str, str] = {}
    for index, intent in enumerate(plan.intents, start=1):
        old_id = str(intent.plan_task_id or "node_%d" % index)
        normalized_id = safe_ascii_component(old_id, default="node_%d" % index)
        mapping[old_id] = "%s_%s" % (namespace, normalized_id)
    intents = []
    for index, intent in enumerate(plan.intents, start=1):
        old_id = str(intent.plan_task_id or "node_%d" % index)
        intents.append(
            intent.model_copy(
                deep=True,
                update={
                    "plan_task_id": mapping[old_id],
                    "depends_on_task_ids": [mapping.get(str(item), str(item)) for item in intent.depends_on_task_ids],
                },
            )
        )
    dependencies = [
        dependency.model_copy(
            deep=True,
            update={
                "anchor_task_id": mapping.get(dependency.anchor_task_id, dependency.anchor_task_id),
                "dependent_task_id": mapping.get(dependency.dependent_task_id, dependency.dependent_task_id),
            },
        )
        for dependency in plan.dependencies
    ]
    contracts = []
    for contract in plan.evidence_contracts:
        next_contract = dict(contract)
        task_id = str(next_contract.get("taskId") or next_contract.get("task_id") or "")
        if task_id in mapping:
            next_contract["taskId"] = mapping[task_id]
            next_contract.pop("task_id", None)
        contracts.append(next_contract)
    return plan.model_copy(
        deep=True,
        update={
            "intents": intents,
            "dependencies": dependencies,
            "evidence_contracts": contracts,
            "agent_trace": list(plan.agent_trace or []) + ["query_graph.namespace=%s" % namespace],
        },
    )


def query_plan_fingerprint(plan: QueryPlan) -> str:
    task_mapping = {
        str(intent.plan_task_id or "node_%d" % index): "node_%d" % index
        for index, intent in enumerate(plan.intents, start=1)
    }
    payload = {
        "intents": [],
        "dependencies": [],
        "evidenceContracts": list(plan.evidence_contracts or []),
        "finalRequiredEvidence": list(plan.final_required_evidence or []),
    }
    for index, intent in enumerate(plan.intents, start=1):
        dumped = intent.model_dump(by_alias=True)
        dumped["planTaskId"] = "node_%d" % index
        dumped["dependsOnTaskIds"] = [
            task_mapping.get(str(task_id), str(task_id)) for task_id in intent.depends_on_task_ids
        ]
        payload["intents"].append(dumped)
    for dependency in plan.dependencies:
        dumped = dependency.model_dump(by_alias=True)
        dumped["anchorTaskId"] = task_mapping.get(dependency.anchor_task_id, dependency.anchor_task_id)
        dumped["dependentTaskId"] = task_mapping.get(dependency.dependent_task_id, dependency.dependent_task_id)
        payload["dependencies"].append(dumped)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def stable_payload_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def normalized_execution_attempt_artifacts(state: AgentState) -> List[ExecutionAttemptArtifact]:
    artifacts: List[ExecutionAttemptArtifact] = []
    for raw in state.get("execution_attempt_artifacts", []) or []:
        try:
            artifact = raw if isinstance(raw, ExecutionAttemptArtifact) else ExecutionAttemptArtifact.model_validate(raw)
        except Exception:
            continue
        artifacts.append(artifact)
    return artifacts


def build_execution_attempt_artifact(
    state: AgentState,
    run_result: AgentRunResult,
    phase: str,
) -> ExecutionAttemptArtifact:
    task_errors = [
        str(item.query_bundle.error or item.summary or "").strip()
        for item in run_result.task_results
        if item.query_bundle.failed or not item.success
    ]
    error = str(run_result.merged_query_bundle.error or "").strip()
    if not error:
        error = "; ".join(item for item in task_errors if item)[:2000]
    failed = bool(
        run_result.merged_query_bundle.failed
        or any(item.query_bundle.failed or not item.success for item in run_result.task_results)
    )
    return ExecutionAttemptArtifact(
        attempt_id="execution_attempt_%s" % uuid.uuid4().hex,
        phase=str(phase or "node_execution"),
        execution_generation=int(state.get("execution_generation") or 0),
        query_graph_fingerprint=query_graph_fingerprint(state.get("plan")),
        recorded_at=datetime.now().isoformat(),
        failed=failed,
        error=error,
        task_results=[item.model_copy(deep=True) for item in run_result.task_results],
        query_bundles=[item.model_copy(deep=True) for item in run_result.query_bundles],
        merged_query_bundle=run_result.merged_query_bundle.model_copy(deep=True),
        sql_repairs=[item.model_copy(deep=True) for item in run_result.sql_repairs],
        node_tool_traces=[item.model_copy(deep=True) for item in run_result.node_tool_traces],
        reflection_notes=list(run_result.reflection_notes or []),
    )


def archive_execution_attempt(
    state: AgentState,
    run_result: AgentRunResult,
    phase: str,
) -> ExecutionAttemptArtifact:
    """Append an immutable execution snapshot before replaceable answer state is updated."""

    archive = normalized_execution_attempt_artifacts(state)
    artifact = build_execution_attempt_artifact(state, run_result, phase)
    archive.append(artifact)
    state["execution_attempt_artifacts"] = archive
    run_result.execution_attempt_artifacts = [item.model_copy(deep=True) for item in archive]
    return artifact


def evidence_accepted_for_state(state: AgentState) -> bool:
    run_result = state.get("agent_run_result") or AgentRunResult()
    if "execution_generation" in state:
        current_generation = int(state.get("execution_generation") or 0)
        result_generation = int(state.get("result_generation") if state.get("result_generation") is not None else -1)
        evidence_generation = int(state.get("evidence_generation") if state.get("evidence_generation") is not None else -1)
        if (result_generation >= 0 or evidence_generation >= 0) and (
            result_generation != current_generation or evidence_generation != current_generation
        ):
            return False
    accepted = state.get("evidence_accepted")
    if accepted is None:
        accepted = state.get("evidence_graph_verified")
    elif accepted is False and state.get("evidence_graph_verified") and str(state.get("verification_status") or "") != "failed":
        accepted = True
    return bool(accepted and run_result.verified_evidence.passed)


def current_analysis_summary_for_state(state: AgentState) -> str:
    summary = str(state.get("analysis_summary") or "")
    if not summary or "execution_generation" not in state:
        return summary
    analysis_generation = int(state.get("analysis_generation") if state.get("analysis_generation") is not None else -1)
    if analysis_generation < 0:
        return summary
    if analysis_generation != int(state.get("execution_generation") or 0):
        return ""
    return summary


def merge_query_plans(left: QueryPlan, right: QueryPlan) -> QueryPlan:
    merged = left.model_copy(deep=True)
    existing = {intent.plan_task_id for intent in merged.intents}
    merged.intents.extend(intent.model_copy(deep=True) for intent in right.intents if intent.plan_task_id not in existing)
    dependency_keys = {
        (item.anchor_task_id, item.dependent_task_id, item.join_key, item.anchor_column, item.dependent_column)
        for item in merged.dependencies
    }
    for dependency in right.dependencies:
        key = (dependency.anchor_task_id, dependency.dependent_task_id, dependency.join_key, dependency.anchor_column, dependency.dependent_column)
        if key not in dependency_keys:
            merged.dependencies.append(dependency.model_copy(deep=True))
            dependency_keys.add(key)
    contract_keys = {
        (str(item.get("taskId") or item.get("task_id") or ""), str(item.get("semanticLabel") or item.get("semantic_label") or ""))
        for item in merged.evidence_contracts
    }
    for contract in right.evidence_contracts:
        key = (str(contract.get("taskId") or contract.get("task_id") or ""), str(contract.get("semanticLabel") or contract.get("semantic_label") or ""))
        if key not in contract_keys:
            merged.evidence_contracts.append(dict(contract))
            contract_keys.add(key)
    merged.final_required_evidence = list(dict.fromkeys([*merged.final_required_evidence, *right.final_required_evidence]))
    merged.agent_trace = list(dict.fromkeys([*merged.agent_trace, *right.agent_trace]))
    return merged


def merge_agent_run_results(left: AgentRunResult, right: AgentRunResult) -> AgentRunResult:
    merged = left.model_copy(deep=True)
    list_fields = [
        "tasks",
        "task_results",
        "query_bundles",
        "sql_repairs",
        "evidence_gaps",
        "reflection_notes",
        "node_tool_traces",
        "node_task_profiles",
        "freshness_reports",
        "node_plan_contracts",
        "node_plan_critiques",
        "sql_draft_decisions",
        "node_execution_batches",
        "skill_lifecycle_records",
        "resumed_task_ids",
        "degraded_reasons",
        "execution_attempt_artifacts",
    ]
    for field in list_fields:
        current = list(getattr(merged, field) or [])
        current.extend(item.model_copy(deep=True) if hasattr(item, "model_copy") else dict(item) if isinstance(item, dict) else item for item in getattr(right, field) or [])
        setattr(merged, field, current)
    merged.merged_query_bundle = merge_task_result_bundles(merged.task_results)
    return merged


def dedupe_workflow_knowledge_requests(items: List[KnowledgeRequest]) -> List[KnowledgeRequest]:
    return dedupe_knowledge_requests(items)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            column = str(key)
            if column not in seen:
                seen.add(column)
                columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sanitize_download_name(value: str) -> str:
    return safe_ascii_component(
        str(value or "query_result").strip(),
        extras=("_", ".", "-"),
        default="query_result",
        strip="._",
    )


def create_workflow(settings: Optional[Settings] = None) -> Any:
    settings = settings or get_settings()
    doris_repository = DorisRepository(settings)
    answer_repository = AnswerRepository(settings)
    pending_store = PendingAnswerStore(settings)
    planner_llm = LlmClient(settings)
    node_llm = LlmClient(settings)
    answer_llm = LlmClient(settings)
    topic_assets = TopicAssetService(settings)
    recall_service = HybridRecallService(settings, topic_assets)
    # One governed knowledge service is shared by the Core Agent filesystem,
    # QueryGraph planning, recall, and NodeWorker evidence binding.  Separate
    # facade instances would share files but not cache/read scope or audit state.
    semantic_catalog = recall_service.semantic_catalog
    knowledge_retriever: KnowledgeRetrievalService
    if settings.es_enabled:
        knowledge_retriever = EsKnowledgeRetrievalService(settings, topic_assets)
    else:
        knowledge_retriever = HybridKnowledgeRetrievalService(recall_service)
    skill_loader = SkillLoader(settings)
    asset_builder = PlanningAssetPackBuilder(topic_assets, skill_loader, doris_repository)
    domain_workflow = MerchantQaWorkflow(
        settings=settings,
        merchant_service=MerchantService(
            settings,
            doris_repository,
            SemanticRuntimeBindingRegistry(settings).resolve("principal_profile"),
        ),
        answer_repository=answer_repository,
        pending_store=pending_store,
        keyword_service=KeywordExtractService(topic_assets),
        routing_service=QuestionRoutingService(topic_assets),
        topic_router=TopicRouterService(topic_assets),
        recall_service=recall_service,
        knowledge_retriever=knowledge_retriever,
        asset_builder=asset_builder,
        planner=QueryGraphPlanner(planner_llm, semantic_catalog=semantic_catalog, settings=settings),
        graph_validator=QueryGraphValidator(),
        node_worker=NodeWorkerExecutor(node_llm, doris_repository, SqlValidationService(), settings, semantic_catalog=semantic_catalog),
        evidence_verifier=EvidenceVerifier(),
        answer_service=AnswerComposeService(answer_llm),
    )
    if str(settings.agent_mode or "").strip().lower() == "deepagent":
        from merchant_ai.services.deep_agent_runtime import DeepAgentWorkflowAdapter

        return DeepAgentWorkflowAdapter(
            domain_workflow=domain_workflow,
            lead_llm=LlmClient(settings),
            semantic_catalog=semantic_catalog,
        )
    return domain_workflow


try:
    graph = create_workflow().graph
except Exception:
    graph = None
