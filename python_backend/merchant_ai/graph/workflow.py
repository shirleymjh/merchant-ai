from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.policy import REPAIRABLE_QUERY_GRAPH_GAP_CODES, V2AgentPolicy
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
    ExtractedKeywords,
    EvidenceGap,
    FastUnderstandingResult,
    GraphValidationGap,
    GraphValidationResult,
    HypothesisEvidenceLedger,
    HypothesisEvidenceRecord,
    HypothesisLedgerEntry,
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
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    QuestionRoute,
    RecallBundle,
    RecallItem,
    RouteSlots,
    SkillDraft,
    SkillLifecycleRecord,
    SkillMatchState,
    SubAgentDelegationPlan,
    SubAgentDelegationTask,
    TOPIC_TO_CATEGORY,
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
from merchant_ai.services.controlled_react import ControlledReactExplorer
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.latency import LatencyOptimizer
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.memory import create_memory_store, memory_query_hash, retrieval_context_from_state, truncate_memory_text_by_tokens
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
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.quick_metrics import published_semantic_quick_metrics, quick_metric_response
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService, merge_task_result_bundles
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, MerchantService, PendingAnswerStore
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    HybridKnowledgeRetrievalService,
    KnowledgeRetrievalService,
    recall_item_sort_key,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
    PreflightUnderstandingService,
    QuestionRoutingService,
    RouteSlotExtractor,
    SemanticPreflightRouteClassifier,
    TopicRouterService,
    route_primary_topic,
)
from merchant_ai.services.skill_drafts import SkillDraftService
from merchant_ai.services.distributed_workers import (
    DistributedSubAgentClient,
    builtin_worker_handlers,
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
        self.route_slot_extractor = RouteSlotExtractor()
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
        self.controlled_react_explorer = ControlledReactExplorer()
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
            analysis_scope={},
            knowledge_refresh={},
            route_slots=RouteSlots(),
            route_decision_trace=[],
            clarification_resolution={},
            clarification_root_question=(question or "").strip(),
            bounded_route_llm_trace={},
            bounded_lead_llm_trace={},
            fast_gate_decision_trace={},
            main_agent_observations=[],
            fast_understanding=FastUnderstandingResult(),
            fast_metric_attempted=False,
            fast_metric_completed=False,
            fast_metric_response=None,
            plan=QueryPlan(),
            recall_bundle=RecallBundle(),
            knowledge_bundle=KnowledgeBundle(),
            recall_rounds=[],
            intent_signals=IntentSignals(),
            planning_asset_pack=PlanningAssetPack(),
            query_graph_validation_result=GraphValidationResult(),
            pending_knowledge_requests=[],
            knowledge_request_attempts={},
            knowledge_request_fingerprints={},
            blocked_knowledge_request_keys=[],
            knowledge_request_lineage={},
            knowledge_request_gaps=[],
            agent_run_result=AgentRunResult(),
            query_bundle=QueryBundle(),
            query_bundles=[],
            available_actions=[],
            lead_decisions=[],
            action_history=[],
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
            run_started_at_ms=now_ms(),
            workspace_manifest=WorkspaceManifest(),
            run_steps=[],
            trace_spans=[],
            planner_repair_requests=[],
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
            memory_injection_trace={},
            memory_ingestion_trace={},
            memory_constraints=[],
            memory_constraint_trace={},
            memory_recalled=False,
            merchant_profile_summary={},
            open_diagnostic_scope="",
            open_diagnostic_intent="",
            open_diagnostic_goal="",
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
        builder.add_node("preflight_route", self.preflight_route)
        builder.add_node("inherit_context", self.inherit_context)
        builder.add_node("runtime_bootstrap", self.runtime_bootstrap)
        builder.add_node("recall_memory", self.recall_memory)
        builder.add_node("policy", self.policy_node)
        builder.add_node("route_topic", self.route_topic)
        builder.add_node("fast_understand", self.fast_understand)
        builder.add_node("try_fast_metric", self.try_fast_metric)
        builder.add_node("retrieve_knowledge", self.retrieve_knowledge)
        builder.add_node("compact_assets", self.compact_assets)
        builder.add_node("query_metric", self.query_metric)
        builder.add_node("plan_query_graph", self.plan_query_graph)
        builder.add_node("reflect_query_graph", self.reflect_query_graph)
        builder.add_node("validate_query_graph", self.validate_query_graph)
        builder.add_node("repair_query_graph", self.repair_query_graph)
        builder.add_node("execute_query_graph", self.execute_query_graph)
        builder.add_node("execute_query_graph_direct", self.execute_query_graph_direct)
        builder.add_node("execute_query_graph_agent", self.execute_query_graph_agent)
        builder.add_node("repair_sql", self.repair_sql)
        builder.add_node("verify_evidence_graph", self.verify_evidence_graph)
        builder.add_node("explore_hypotheses", self.explore_hypotheses)
        builder.add_node("run_analysis_worker", self.run_analysis_worker)
        builder.add_node("run_analysis_skill", self.run_analysis_skill)
        builder.add_node("delegate_subagent", self.delegate_subagent)
        builder.add_node("answer_rule", self.answer_rule)
        builder.add_node("answer_analysis", self.answer_analysis)
        builder.add_node("human_in_loop", self.human_in_loop)
        builder.add_node("cache_answer", self.cache_answer)
        builder.add_node("terminal_end", self.terminal_end)

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
        builder.add_edge("recall_memory", "policy")
        builder.add_conditional_edges(
            "policy",
            lambda state: state.get("_next_action", "cache_answer"),
            {
                "route_topic": "route_topic",
                "recall_memory": "recall_memory",
                "fast_understand": "fast_understand",
                "try_fast_metric": "try_fast_metric",
                "retrieve_knowledge": "retrieve_knowledge",
                "compact_assets": "compact_assets",
                "query_metric": "query_metric",
                "plan_query_graph": "plan_query_graph",
                "reflect_query_graph": "reflect_query_graph",
                "validate_query_graph": "validate_query_graph",
                "repair_query_graph": "repair_query_graph",
                "execute_query_graph": "execute_query_graph",
                "execute_query_graph_direct": "execute_query_graph_direct",
                "execute_query_graph_agent": "execute_query_graph_agent",
                "repair_sql": "repair_sql",
                "verify_evidence_graph": "verify_evidence_graph",
                "explore_hypotheses": "explore_hypotheses",
                "run_analysis_worker": "run_analysis_worker",
                "run_analysis_skill": "run_analysis_skill",
                "delegate_subagent": "delegate_subagent",
                "answer_rule": "answer_rule",
                "answer_analysis": "answer_analysis",
                "human_in_loop": "human_in_loop",
                "cache_answer": "cache_answer",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_edge("route_topic", "policy")
        builder.add_edge("fast_understand", "policy")
        builder.add_edge("try_fast_metric", "policy")
        builder.add_conditional_edges(
            "retrieve_knowledge",
            self.route_after_retrieve_knowledge,
            {
                "compact_assets": "compact_assets",
                "policy": "policy",
                "human_in_loop": "human_in_loop",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_edge("compact_assets", "policy")
        builder.add_edge("query_metric", "policy")
        builder.add_conditional_edges(
            "plan_query_graph",
            self.route_after_plan_query_graph,
            {
                "reflect_query_graph": "reflect_query_graph",
                "validate_query_graph": "validate_query_graph",
                "policy": "policy",
                "human_in_loop": "human_in_loop",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_conditional_edges(
            "reflect_query_graph",
            self.route_after_reflect_query_graph,
            {
                "validate_query_graph": "validate_query_graph",
                "policy": "policy",
                "human_in_loop": "human_in_loop",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_conditional_edges(
            "repair_query_graph",
            self.route_after_repair_query_graph,
            {
                "reflect_query_graph": "reflect_query_graph",
                "policy": "policy",
                "human_in_loop": "human_in_loop",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_edge("validate_query_graph", "policy")
        builder.add_edge("execute_query_graph", "repair_sql")
        builder.add_edge("execute_query_graph_direct", "repair_sql")
        builder.add_edge("execute_query_graph_agent", "repair_sql")
        builder.add_conditional_edges(
            "repair_sql",
            self.route_after_repair_sql,
            {
                "verify_evidence_graph": "verify_evidence_graph",
                "policy": "policy",
                "human_in_loop": "human_in_loop",
                "terminal_end": "terminal_end",
            },
        )
        builder.add_edge("verify_evidence_graph", "policy")
        builder.add_edge("explore_hypotheses", "policy")
        builder.add_edge("run_analysis_worker", "policy")
        builder.add_edge("run_analysis_skill", "policy")
        builder.add_edge("delegate_subagent", "policy")
        builder.add_edge("answer_rule", "cache_answer")
        builder.add_edge("answer_analysis", "cache_answer")
        builder.add_edge("human_in_loop", END)
        builder.add_edge("cache_answer", END)
        builder.add_edge("terminal_end", END)
        return builder.compile(checkpointer=self.checkpoint_manager.saver())

    def terminal_or_human_node(self, state: AgentState) -> str:
        if state.get("run_canceled") or (state.get("terminal_status") or {}).get("active"):
            return "terminal_end"
        if state.get("human_clarification_required"):
            return "human_in_loop"
        return ""

    def can_retry_knowledge(self, state: AgentState) -> bool:
        return self.policy.can_retrieve_supplemental(state) and not self.policy.knowledge_recall_stalled(state)

    def can_repair_graph(self, state: AgentState) -> bool:
        return int(state.get("query_graph_repair_attempts") or 0) < self.policy.max_graph_repair_actions

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
        state["agent_run_result"] = AgentRunResult()
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
            return "compact_assets"
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
            return "validate_query_graph"
        return "reflect_query_graph"

    def route_after_reflect_query_graph(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_pending_knowledge_requests(state) and self.can_retry_knowledge(state):
            return "policy"
        reflection = state.get("planner_reflection") or PlannerReflectionResult()
        if getattr(reflection, "passed", True):
            return "validate_query_graph"
        return "policy"

    def route_after_repair_query_graph(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_pending_knowledge_requests(state) and self.can_retry_knowledge(state):
            return "policy"
        plan = state.get("plan") or QueryPlan()
        if not plan.intents:
            return "policy"
        return "reflect_query_graph"

    def route_after_repair_sql(self, state: AgentState) -> str:
        terminal = self.terminal_or_human_node(state)
        if terminal:
            return terminal
        if self.policy.has_graph_repairable_execution_gap(state):
            return "policy"
        return "verify_evidence_graph"

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
                clarification = "当前 BI Agent 只支持只读查询和分析，不能执行删除、修改、创建或重建等写操作。请改成只读问题，例如“查看最近30天相关数据”。"
                options = ["改成只读查询", "取消本次操作"]
                clarification_type = "write_operation"
                stage = "UNSUPPORTED_OPERATION"
            else:
                clarification = understanding.clarification_question or self.build_scope_clarification_prompt(state)
                options = business_scope_options()
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
                state["memory_constraints"] = []
                state["memory_constraint_trace"] = {"constraintCount": 0, "status": "skipped", "reason": "topic_not_routed"}
                add_step(state, "Long-term Memory：Topic workspace 未就绪，暂不召回长期记忆")
                self.record_span(state, "action", "recall_memory", started, metadata={"skipped": True, "route": enum_value(route), "reason": "topic_not_routed"})
                self.finish_run_step(state, step, "skipped", output_summary="topic_not_routed")
                emit(state, "node.completed", "MEMORY_RECALL", {"selectedCount": 0, "constraintCount": 0, "skipped": True})
                return state
            state["memory_injection"] = {}
            state["memory_injection_trace"] = {"status": "skipped", "reason": "non_business_route", "route": enum_value(route)}
            state["memory_constraints"] = []
            state["memory_constraint_trace"] = {"constraintCount": 0, "status": "skipped", "reason": "non_business_route"}
            state["memory_recalled"] = True
            state["_memory_snapshot_locked"] = True
            add_step(state, "Long-term Memory：非业务/轻量请求跳过长期记忆召回")
            self.record_span(state, "action", "recall_memory", started, metadata={"skipped": True, "route": enum_value(route)})
            self.finish_run_step(state, step, "skipped", output_summary="non_business_route")
            emit(state, "node.completed", "MEMORY_RECALL", {"selectedCount": 0, "constraintCount": 0, "skipped": True})
            return state
        try:
            injection = self.memory_store.select_for_question(
                state,
                budget_tokens=int(self.settings.context_memory_budget_tokens or 1200),
            )
            state["memory_injection"] = injection
            state["memory_recalled"] = True
            trace = dict(injection.get("memoryInjectionTrace") or {})
            trace.update(
                {
                    "status": "success",
                    "contextFingerprint": memory_query_hash(
                        str(state.get("requested_merchant_id") or ""),
                        retrieval_context_from_state(state),
                    ),
                }
            )
            state["memory_injection_trace"] = trace
            state["_memory_snapshot_locked"] = True
            state["memory_constraints"] = build_memory_constraints(injection)
            state["memory_constraint_trace"] = {
                "constraintCount": len(state["memory_constraints"]),
                "requiredCount": sum(
                    1 for item in state["memory_constraints"] if str(item.get("enforcement") or "") == "required"
                ),
                "clarifyCount": sum(
                    1
                    for item in state["memory_constraints"]
                    if str(item.get("enforcement") or "") == "clarify_or_disclose"
                ),
                "source": injection.get("source", ""),
            }
            state["merchant_profile_summary"] = self.merchant_profile_summary_service.summarize(
                merchant=state["merchant"],
                memory_injection=injection,
                memory_constraints=state["memory_constraints"],
                route_slots=state.get("route_slots", RouteSlots()),
                fast_understanding=state.get("fast_understanding", FastUnderstandingResult()),
            )
            renderer = getattr(self.memory_store, "render_injection", None)
            rendered = renderer(injection) if callable(renderer) else ""
            state["memory_context"] = (
                truncate_memory_text_by_tokens(
                    rendered,
                    int(self.settings.context_memory_budget_tokens or 1200),
                )
                if rendered
                else ""
            )
            add_step(
                state,
                "Long-term Memory：回答前召回完成 selected=%d constraints=%d"
                % (
                    len(state["memory_injection_trace"].get("selectedIds") or []),
                    len(state["memory_constraints"]),
                ),
            )
            self.finish_run_step(
                state,
                step,
                "success",
                output_summary="selected=%d constraints=%d"
                % (
                    len(state["memory_injection_trace"].get("selectedIds") or []),
                    len(state["memory_constraints"]),
                ),
            )
        except Exception as exc:
            state["memory_injection"] = {}
            state["memory_recalled"] = True
            state["_memory_snapshot_locked"] = True
            state["memory_injection_trace"] = {
                "status": "failed",
                "error": str(exc)[:500],
                "contextFingerprint": memory_query_hash(
                    str(state.get("requested_merchant_id") or ""),
                    retrieval_context_from_state(state),
                ),
            }
            state["memory_constraints"] = []
            state["memory_constraint_trace"] = {"constraintCount": 0, "error": str(exc)[:500]}
            add_step(state, "Long-term Memory：回答前召回失败，已降级为空记忆")
            self.finish_run_step(state, step, "failed", output_summary=str(exc)[:500])
        self.record_span(state, "action", "recall_memory", started)
        emit(
            state,
            "node.completed",
            "MEMORY_RECALL",
            {
                "selectedCount": len(state.get("memory_injection_trace", {}).get("selectedIds") or []),
                "constraintCount": len(state.get("memory_constraints") or []),
            },
        )
        return state

    def policy_node(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "policy", "LeadAgent", "MAIN_AGENT_POLICY", input_summary=state.get("agent_decision_reason", ""))
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
            pack = state.get("planning_asset_pack") or PlanningAssetPack()
            state["hypothesis_exploration"] = (
                self.controlled_react_explorer.build_hypotheses(
                    state.get("question", ""),
                    pack,
                    (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
                )
                if pack.known_tables()
                else {}
            )
            state["hypothesis_results"] = []
            state["hypothesis_exploration_completed"] = False
            state["hypothesis_exploration_status"] = {"status": "pending", "source": "runtime"}
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
        mode = str(getattr(self.settings, "lead_action_llm_mode", "always") or "always").lower()
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
        is_preknowledge_retrieval_guard = bool(
            not state.get("data_discovered")
            and state.get("fast_understood")
            and decision.selected_action == "retrieve_knowledge"
        )
        if is_preknowledge_retrieval_guard:
            state["fast_gate_decision_trace"] = trace
            fast_guard_reason = self.fast_gate_retrieval_guard_reason(state)
            if fast_guard_reason:
                action = self.policy.registry.get("retrieve_knowledge")
                trace.update(
                    {
                        "status": "forced",
                        "reason": fast_guard_reason,
                        "selectedAction": "retrieve_knowledge",
                        "historyAuthoritative": False,
                        "knowledgeRefreshPolicy": "refresh_each_business_turn",
                    }
                )
                state["fast_gate_decision_trace"] = trace
                return AgentDecision(
                    selected_action=action.id,
                    selected_node=action.node,
                    available_actions=allowed,
                    reason=fast_guard_reason,
                    budget_exhausted=decision.budget_exhausted,
                    observation=str(observation.get("summary") or decision.observation),
                    source="knowledge_refresh_guard",
                )
        if self.latency_optimizer.blocks_expensive_agents(state.get("latency_optimization") or {}):
            trace.update({"status": "skipped", "reason": "simple_request_fast_path_blocks_lead_llm"})
            return decision
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
            trace["policySource"] = "deterministic"
            return decision
        if len(allowed) <= 1:
            trace["reason"] = "single_available_action"
            trace["policySource"] = "deterministic"
            return decision
        should_call = mode == "always" or (mode == "fast_gate" and is_fast_gate) or (
            mode == "low_confidence"
            and (
                bool(state.get("pending_knowledge_requests"))
                or bool(state.get("planner_repair_requests"))
                or not bool(state.get("query_graph_validated"))
                or bool(getattr(state.get("query_graph_validation_result"), "gaps", None))
            )
        )
        if mode == "adaptive":
            should_call = self.adaptive_lead_llm_needed(state, allowed, is_fast_gate)
        if not should_call:
            trace["reason"] = "deterministic_decision_confident"
            trace["policySource"] = "deterministic"
            return decision
        llm = getattr(self.planner, "llm", None)
        if not llm or not getattr(llm, "configured", False):
            trace.update({"status": "skipped", "reason": "llm_not_configured"})
            return decision
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
        state["_lead_llm_decision_fingerprint"] = lead_decision_fingerprint(state, allowed)
        lead_llm_started = now_ms()
        try:
            if hasattr(llm, "tool_json_chat"):
                tool = lead_action_selection_tool(allowed)
                llm_payload = llm.tool_json_chat(
                    "你是主 Agent 的 ReAct 决策器。先读 observation，再只能调用 select_agent_action 选择下一步。",
                    json.dumps(payload, ensure_ascii=False, default=str),
                    tool.openai_schema(),
                    {},
                    timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
                )
                trace["tool"] = tool.trace_schema()
            else:
                llm_payload = llm.json_chat(
                    "你是 BI Agent 的受限 LeadAction 选择器，只能在给定 action registry 候选中改选下一步。",
                    json.dumps(payload, ensure_ascii=False, default=str),
                    fallback={},
                    timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
                )
        except Exception as exc:
            self.record_span(
                state,
                "llm",
                "lead_action.select",
                lead_llm_started,
                status="failed",
                error_code="LEAD_LLM_FAILED",
                error_message=str(exc)[:300],
                model=self.settings.openai_model,
                provider=self.settings.openai_base_url,
            )
            trace.update({"status": "failed", "errorCode": "LEAD_LLM_FAILED", "errorMessage": str(exc)[:300]})
            return decision
        self.record_span(
            state,
            "llm",
            "lead_action.select",
            lead_llm_started,
            model=self.settings.openai_model,
            provider=self.settings.openai_base_url,
            estimated_prompt_chars=len(json.dumps(payload, ensure_ascii=False, default=str)),
            estimated_completion_chars=len(json.dumps(llm_payload or {}, ensure_ascii=False, default=str)),
        )
        selected_action = str(
            (llm_payload or {}).get("actionId")
            or (llm_payload or {}).get("action_id")
            or (llm_payload or {}).get("selectedAction")
            or (llm_payload or {}).get("selected_action")
            or ""
        )
        if selected_action not in allowed:
            trace.update({"status": "ignored", "reason": "llm_selected_action_not_allowed", "payload": llm_payload or {}})
            return decision
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

    def apply_bounded_lead_llm_decision(self, state: AgentState, decision: AgentDecision) -> AgentDecision:
        return self.arbitrate_lead_action_if_needed(state, decision)

    def lead_action_deterministic_skip_reason(self, state: AgentState, decision: AgentDecision, allowed: List[str]) -> str:
        """Return why LeadAction LLM arbitration is unnecessary for settled states.

        This does not prescribe the business workflow.  It only avoids asking a
        second model to re-decide states that the harness has already proven:
        terminal answers, clarification stops, verified evidence, and required
        knowledge refreshes.
        """

        selected = str(decision.selected_action or "")
        allowed_set = {str(item) for item in allowed if item}
        analysis_dispatch_actions = {"delegate_subagent", "explore_hypotheses", "run_analysis_worker", "run_analysis_skill"}
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
        if selected == "answer_rule":
            return "rule_answer_ready"
        if selected == "answer_data":
            if state.get("query_metric_completed") and evidence_accepted_for_state(state):
                return "verified_query_metric_answer_ready"
            if evidence_accepted_for_state(state) and not (analysis_dispatch_actions & allowed_set):
                return "verified_evidence_answer_ready"
            route = state.get("routing_decision") or RoutingDecision()
            if route.route in {QuestionRoute.GREETING, QuestionRoute.INVALID}:
                return "preflight_terminal_answer"
        return ""

    def fast_gate_retrieval_guard_reason(self, state: AgentState) -> str:
        route = state.get("routing_decision") or RoutingDecision()
        if route.route != QuestionRoute.BUSINESS:
            return ""
        fast = state.get("fast_understanding") or FastUnderstandingResult()
        slots = state.get("route_slots") or RouteSlots()
        question = str(state.get("question") or "")
        knowledge_sensitive_terms = [
            "为什么",
            "原因",
            "归因",
            "诊断",
            "异常",
            "建议",
            "口径",
            "规则",
            "含义",
            "什么意思",
            "状态",
            "status",
            "枚举",
        ]
        if re.search(r"口径|定义|含义|什么意思|是否扣|怎么算|计算方式", question) and re.search(
            r"gmv|销售额|成交额|订单|退款|退款率|工单|客诉|赔付",
            question.lower(),
        ):
            return ""
        if str(getattr(slots, "risk_level", "") or "") in {"rule_sensitive", "high_risk"}:
            return "本轮问题涉及平台规则或业务口径，必须先刷新 Topic 知识，history 不作为权威依据"
        topic_values = {enum_value(topic) for topic in list(getattr(fast, "topics", []) or [])}
        if QuestionCategory.PLATFORM_RULE.value in topic_values:
            return "本轮问题命中平台规则 Topic，必须走最新知识检索"
        if str(getattr(fast, "intent_kind", "") or "") in {"rule_only", "rule_data_mix", "multi_hop"}:
            return "本轮问题不是简单指标查询，必须走最新知识检索后再分析"
        if str(getattr(fast, "intent_kind", "") or "") in {"analysis", "unknown"} and any(term in question for term in knowledge_sensitive_terms):
            return "本轮问题涉及归因、口径或规则解释，必须走最新知识检索后再分析"
        if str(getattr(fast, "complexity", "") or "") == "complex" and any(term in question for term in knowledge_sensitive_terms):
            return "本轮问题复杂度较高，必须走最新知识检索后再规划"
        return ""

    def lead_llm_action_catalog(self, action_ids: List[str], state: Optional[AgentState] = None) -> List[str]:
        """Hide equivalent compatibility actions while retaining deterministic fallbacks."""
        catalog = list(dict.fromkeys(action_ids))
        analysis_dispatch_actions = {"delegate_subagent", "explore_hypotheses", "run_analysis_worker", "run_analysis_skill"}
        if "answer_data" in catalog and len(catalog) > 1 and not (analysis_dispatch_actions & set(catalog)):
            catalog.remove("answer_data")
        return catalog

    def adaptive_lead_llm_needed(self, state: AgentState, allowed: List[str], is_fast_gate: bool) -> bool:
        if remaining_run_budget_seconds(state, self.settings) <= 12:
            return False
        allowed_set = set(allowed)
        # Deterministic policy already knows whether the governed fast metric
        # can be attempted, which execution tier is cheapest, and whether a
        # matched Skill is required.  Asking an LLM to repeat those decisions
        # adds several network round trips without improving correctness.
        # Keep the bounded Lead LLM only for genuinely ambiguous recovery.
        analysis_dispatch = bool(
            "answer_data" in allowed_set
            and ({"run_analysis_worker", "run_analysis_skill", "delegate_subagent", "explore_hypotheses"} & allowed_set)
            and evidence_accepted_for_state(state)
        )
        strategic = bool(
            ("repair_graph" in allowed_set and bool(getattr(state.get("query_graph_validation_result"), "gaps", None)))
            or ("retrieve_knowledge" in allowed_set and bool(state.get("pending_knowledge_requests")))
            or analysis_dispatch
        )
        if not strategic:
            return False
        fingerprint = lead_decision_fingerprint(state, allowed)
        return fingerprint != str(state.get("_lead_llm_decision_fingerprint") or "")

    def lead_agent_tool_instruction(self, is_fast_gate: bool) -> str:
        base = (
            "你是商家经营分析主 Agent。根据用户目标、observation、工具结果、证据缺口和 actionCatalog 自主选择下一项工具。"
            "Harness 已经移除了不满足权限或安全前置条件的工具；不要机械遵循固定流水线，也不要重复没有新增信息的动作。"
            "在证据不足时继续检索、规划、修复或执行；只有证据已经校验或必须明确披露缺口时才回答。"
            "专项深度分析只在已取得并校验经营数据后选择。只能从 allowedActions 中选择一个 action id，不创造新 action。"
            "当任务适合隔离上下文、文档分析、批量 Python 或多个独立 Worker 时可选择 delegate_subagent；"
            "当已验证数据足够但问题属于开放、长尾、非固定 SOP 的分析，可选择 run_analysis_worker；"
            "只有明确匹配已发布可复用 Skill 时才选择 run_analysis_skill；不要把普通复杂分析硬塞进 Skill。"
            "当问题要求原因、归因、异常诊断或优先建议，且存在多个可验证经营假设时，优先选择 explore_hypotheses；"
            "explore_hypotheses 会生成并执行新的独立 QueryGraph；delegate_subagent 的 hypothesis_review 只复核已有假设和证据，两者不可互换。"
            "如果当前证据已经足以回答，且不需要隔离分析或 Skill，选择 answer_data。"
            "返回 JSON: {selectedAction:'', reason:''}。"
        )
        if not is_fast_gate:
            return base
        return base + (
            "当前已完成 Topic workspace 召回和资产压缩：只有一个受控语义指标、无需归因/排行/明细/跨域分析时可选择 query_metric；"
            "如果指标歧义、资产缺口、需要解释原因或多轮探索，选择 plan_graph 或后续 ask_human。"
            "query_metric 只接收当前 Topic 资产包里的语义引用，并会自行完成校验、执行和证据门禁。"
        )

    def main_agent_observation(self, state: AgentState) -> Dict[str, Any]:
        validation = state.get("query_graph_validation_result") or GraphValidationResult()
        run_result = state.get("agent_run_result") or AgentRunResult()
        plan = state.get("plan") or QueryPlan()
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
        if state.get("human_clarification_required"):
            summary_parts.append("needsHuman=true")
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
            "toolRuntimeFailures": state.get("tool_failures", [])[-4:],
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
            "executionFailures": {
                "sql": sql_failures[:8],
                "toolRuntime": state.get("tool_failures", [])[-6:],
            },
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
        state["query_graph_validation_result"] = GraphValidationResult(
            valid=False,
            repairable=False,
            gaps=[
                GraphValidationGap(
                    code=gap_code,
                    reason=reason,
                )
            ],
        )
        state["query_graph_validated"] = True

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
        state["route_slots"] = route_slots
        if route_slots.operation == "write_requested":
            self.request_human_clarification(
                state,
                "当前 BI Agent 只支持只读查询和分析，不能执行删除、修改、创建或重建等写操作。请改成只读问题，例如“查看最近30天相关数据”。",
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
            context_topics=list(getattr(state.get("request_context"), "topics", []) or []),
        )
        decision, route_llm_trace = self.apply_bounded_route_llm_decision(state, decision, route_slots)
        state["bounded_route_llm_trace"] = route_llm_trace
        state["route_decision_trace"].append(
            {
                "stage": "topic_router",
                "candidateTopics": [enum_value(item) for item in decision.candidate_topics],
                "confidence": decision.confidence,
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
            self.request_human_clarification(state, self.build_topic_clarification_prompt(state), "BUSINESS_SCOPE", "topic_required", business_scope_options())
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
        user_confirmed = bool(
            context
            and (
                getattr(context, "clarification_resolved", False)
                or getattr(context, "topics", None)
                or getattr(context, "topic", "")
            )
        )
        open_discovery = bool(
            is_store_health_overview_question(state.get("question", ""))
            or is_priority_recommendation_question(state.get("question", ""))
            or (context and context.pending_clarification_type == "priority_goal")
        )
        explicit_single_topic_lock = bool(
            len(topics) == 1
            and (
                re.search(r"(?:只|仅|单独)(?:看|查|分析|关注)", str(state.get("question") or ""))
                or (
                    context
                    and context.pending_clarification_type == "topic_required"
                    and getattr(context, "clarification_resolved", False)
                )
            )
        )
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
            topic_role = "topic_boundary"
            expansion_policy = "on_gap_or_tool_request"
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
        }
        state["topic_workspace"] = workspace
        state["analysis_scope"] = {
            **workspace,
            "riskLevel": str(route_slots.risk_level or "normal"),
            "timeWindow": route_slots.time_window.model_dump(by_alias=True),
            "objectRefs": [item.model_dump(by_alias=True) for item in route_slots.object_refs],
            "displayText": "当前 Topic 边界：%s；仅在证据缺口或工具请求时补充关联 Topic"
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
        has_rule = QuestionCategory.PLATFORM_RULE in set(topics) or slots.risk_level in {"rule_sensitive", "high_risk"}
        business_topic_count = len([topic for topic in topics if topic not in {QuestionCategory.UNKNOWN, QuestionCategory.MERCHANT_OTHER}])
        object_refs: Dict[str, List[str]] = {}
        for ref in slots.object_refs:
            object_refs.setdefault(ref.ref_type, [])
            if ref.value not in object_refs[ref.ref_type]:
                object_refs[ref.ref_type].append(ref.value)
        metric_phrases = dedupe_texts(list(keywords.metric_keywords or keywords.business_keywords or [])[:12])
        clarified_metric_focus = str((state.get("clarification_resolution") or {}).get("metricFocus") or "")
        if clarified_metric_focus:
            metric_phrases = dedupe_texts([clarified_metric_focus, *metric_phrases])
        analysis_requested = bool(slots.analysis_signals or state.get("open_diagnostic_intent"))
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
        elif analysis_requested:
            intent_kind = "analysis"
            complexity = "complex"
        elif business_topic_count >= 3:
            intent_kind = "multi_hop"
            complexity = "complex"
        elif len(metric_phrases) >= 3:
            intent_kind = "multi_metric"
            complexity = "medium"
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
            state["merchant_profile_summary"] = self.merchant_profile_summary_service.summarize(
                merchant=state["merchant"],
                memory_injection=state.get("memory_injection") or {},
                memory_constraints=state.get("memory_constraints") or [],
                route_slots=state.get("route_slots", RouteSlots()),
                fast_understanding=result,
            )
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
        category = TOPIC_TO_CATEGORY.get(topic_name, QuestionCategory.UNKNOWN)
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
        category = TOPIC_TO_CATEGORY.get(topic_name, QuestionCategory.UNKNOWN)
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
        text = state.get("question", "")
        slots = state.get("route_slots") or RouteSlots()
        topics = set(fast.topics or self._effective_topic_categories(state))
        business_topics = {
            topic
            for topic in topics
            if topic not in {QuestionCategory.UNKNOWN, QuestionCategory.MERCHANT_OTHER, QuestionCategory.IDENTITY}
        }
        if fast.intent_kind in {"chat", "invalid", "write_requested", "rule_only"}:
            return {}
        if is_store_health_overview_question(text) or state.get("open_diagnostic_intent") == "STORE_HEALTH_DIAGNOSIS":
            return {}
        if is_priority_recommendation_question(text) and not state.get("open_diagnostic_goal"):
            return {
                "stage": "OPEN_DIAGNOSTIC",
                "type": "priority_goal",
                "question": self.build_priority_goal_clarification_prompt(state),
                "options": priority_goal_options(),
                "reason": "开放优先级问题缺少排序目标，先确认商家最关心的经营目标",
            }
        if (
            slots.time_window.days <= 0
            and not (state.get("extracted_keywords") or ExtractedKeywords()).time_keywords
            and business_topics
            and self.question_needs_time_clarification(text, fast)
        ):
            return {
                "stage": "BUSINESS_SCOPE",
                "type": "time_window",
                "question": "你想按哪个时间范围看？我可以先按近7天，也可以切到昨天、近30天或你指定的时间。",
                "options": ["近7天", "昨天", "近30天", "我补充具体时间"],
                "reason": "经营查询缺少时间范围，先确认口径避免默认时间误导",
            }
        if (
            not fast.metric_phrases
            and fast.intent_kind in {"analysis", "multi_hop", "unknown"}
            and self.question_needs_metric_clarification(text)
        ):
            return {
                "stage": "BUSINESS_SCOPE",
                "type": "metric_focus",
                "question": "你说的表现/异常主要想看哪个指标？我可以按 GMV、订单量、退款率、客诉/工单或综合经营风险来分析。",
                "options": ["综合经营风险", "GMV/销售额", "订单量", "退款率", "客诉/工单"],
                "reason": "开放经营分析缺少目标指标，先确认分析重心",
            }
        return {}

    def question_needs_time_clarification(self, text: str, fast: FastUnderstandingResult) -> bool:
        lowered = (text or "").lower()
        if re.search(r"今天|昨日|昨天|近\d+天|最近\d+天|本周|上周|本月|上月|\d{4}[-/年]\d{1,2}", lowered):
            return False
        if re.search(r"口径|定义|含义|什么意思|是否扣|规则|枚举|status|状态", lowered):
            return False
        if fast.object_refs and re.search(r"明细|详情|记录|单号|流水|状态", lowered):
            return False
        if fast.intent_kind in {"detail_lookup"} and not re.search(r"趋势|变化|下降|上升|异常|表现|为什么|原因|归因|分析|多少|情况", lowered):
            return False
        return bool(re.search(r"多少|情况|趋势|变化|下降|上升|异常|表现|为什么|原因|归因|分析|top|排行|退款|订单|gmv|销售额|工单|客诉|赔付", lowered))

    def question_needs_metric_clarification(self, text: str) -> bool:
        lowered = (text or "").lower()
        if re.search(r"gmv|销售额|成交额|订单|退款|退款率|退货|工单|客诉|赔付|优惠券|库存|履约|发货|商品|转化", lowered):
            return False
        return bool(re.search(r"表现|异常|不好|下滑|下降|变差|风险|优先|原因|归因|分析|怎么了", lowered))

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
            "instruction": "只允许从 allowedTopics 中保留或删除 topic，不允许新增未知 topic。返回 JSON: {topics:[], confidence:0-1, reason:''}",
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
        decision.reason = "受限 route LLM 确认；多 topic 时 primaryTopic 保持 UNKNOWN，不表示 anchor。%s" % str(
            (payload or {}).get("reason") or ""
        )
        trace.update({"status": "applied", "topics": topics, "payload": payload or {}})
        return decision, trace

    def topic_clarification_gate_reason(self, state: AgentState, decision: TopicRoutingDecision, route_slots: RouteSlots) -> str:
        if not bool(getattr(self.settings, "route_force_clarification_enabled", True)):
            return ""
        text = state.get("question", "")
        context = state.get("request_context")
        if is_store_health_overview_question(text) or is_priority_recommendation_question(text):
            return ""
        if context and context.pending_clarification_type == "priority_goal":
            return ""
        topics = decision.recall_topics()
        business_topics = [
            topic
            for topic in topics
            if topic not in {QuestionCategory.UNKNOWN, QuestionCategory.MERCHANT_OTHER}
        ]
        data_topics = [topic for topic in business_topics if topic != QuestionCategory.PLATFORM_RULE]
        confidence = max(float(decision.confidence or 0.0), float(route_slots.route_confidence or 0.0))
        min_confidence = float(getattr(self.settings, "route_topic_min_confidence", 0.52) or 0.52)
        max_candidates = int(getattr(self.settings, "route_topic_max_candidates", 4) or 4)
        mixed_min_confidence = float(getattr(self.settings, "route_mixed_rule_data_min_confidence", 0.75) or 0.75)
        if not business_topics and confidence < min_confidence:
            return "业务域低置信且没有明确候选 Topic，先确认分析范围"
        if confidence < min_confidence and ("NO_EXPLICIT_TOPIC" in route_slots.route_warnings or not route_slots.object_refs):
            return "Topic 低置信，缺少明确对象或业务关键词，先确认分析范围"
        if len(business_topics) > max_candidates and confidence < 0.82:
            return "候选 Topic 过多且未形成稳定主域，先确认优先业务范围"
        if QuestionCategory.PLATFORM_RULE in business_topics and data_topics and confidence < mixed_min_confidence:
            return "规则问题和数据分析域混在一起且置信度不足，先确认是查规则还是查经营数据"
        if "BROAD_TOPIC_SET" in route_slots.route_warnings and confidence < mixed_min_confidence:
            return "路由命中范围过宽，先确认要看的业务域"
        return ""

    def load_skill_policies_for_retrieval(self, state: AgentState) -> List[str]:
        skills = self.asset_builder.skill_loader.select(state["question"], self._effective_topic_categories(state))
        state["loaded_skills"] = [skill.domain for skill in skills]
        state["skills_loaded"] = True
        return state["loaded_skills"]

    def retrieve_knowledge(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "retrieve_knowledge", "KnowledgeAgent", "RETRIEVE_KNOWLEDGE", input_summary=state.get("question", ""))
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
        was_data_discovered = bool(state.get("data_discovered"))
        if was_data_discovered or had_pending_requests:
            state["query_graph_supplemental_retrieve_count"] = int(state.get("query_graph_supplemental_retrieve_count") or 0) + 1
        base_topics = self._effective_topic_categories(state)
        fast_understanding = state.get("fast_understanding") or FastUnderstandingResult()
        query_scopes: List[tuple[str, List[QuestionCategory], Optional[KnowledgeRequest]]] = [(state["question"], base_topics, None)]
        route_query = self.route_recall_query(state)
        if route_query and route_query != state["question"]:
            query_scopes.insert(0, (route_query, base_topics, None))
        merged = state.get("recall_bundle") or RecallBundle()
        all_items = {item.doc_id: item for item in merged.items}
        existing_refs = set(all_items)
        round_traces = list(state.get("recall_rounds") or [])
        expanded_topics = list(state.get("knowledge_expanded_topics") or [])
        knowledge_bundles: List[KnowledgeBundle] = []
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
                    access_role=state.get("access_role", "merchant_analyst"),
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
                )
                knowledge_bundle = self.knowledge_retriever.retrieve(retrieval_request)
                if not knowledge_bundle.recall_bundle.items and str(knowledge_bundle.backend or "").lower().startswith("es"):
                    fallback_bundle = HybridKnowledgeRetrievalService(self.recall_service).retrieve(retrieval_request)
                    if fallback_bundle.recall_bundle.items:
                        fallback_bundle.backend = "es_fallback_hybrid"
                        fallback_bundle.recall_rounds = list(knowledge_bundle.recall_rounds or []) + list(fallback_bundle.recall_rounds or [])
                        knowledge_bundle = fallback_bundle
                knowledge_bundles.append(knowledge_bundle)
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
                self._knowledge_request_topics(request, base_topics),
                request.model_copy(update={"request_key": knowledge_request_key(request)}),
            )
            for request in active_pending_requests
            if request.query
        ]
        if pending_scopes:
            run_recall_scopes(pending_scopes, "pending_knowledge_request")
        expansion_topics = self.knowledge_recall_expansion_topics(state, base_topics, active_pending_requests)
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
            state["knowledge_request_gaps"] = append_knowledge_request_gaps(
                state.get("knowledge_request_gaps", []),
                no_match_requests,
                "KNOWLEDGE_REQUEST_NO_MATCH",
            )
            add_step(
                state,
                "KnowledgeAgent：%d 个补知识请求未召回到 request-specific 证据，记录缺口并停止重试"
                % len(no_match_requests),
            )
        lineage_items = list(all_items.values())
        items = sorted(lineage_items, key=recall_item_sort_key, reverse=True)[:24]
        state["recall_bundle"] = RecallBundle(
            items=items,
            top_score=items[0].fusion_score if items else 0.0,
            merged_context="\n\n".join("召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items),
        )
        backend = next((bundle.backend for bundle in knowledge_bundles if bundle.backend), "hybrid")
        state["knowledge_bundle"] = KnowledgeBundle(
            recall_bundle=state["recall_bundle"],
            source_refs=sorted({item.doc_id for item in items if item.doc_id}),
            recall_rounds=[],
            backend=backend,
            index_version=next((bundle.index_version for bundle in knowledge_bundles if bundle.index_version), ""),
            semantic_source_hash=next((bundle.semantic_source_hash for bundle in knowledge_bundles if bundle.semantic_source_hash), ""),
        )
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
        state["planning_assets_compacted"] = False
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        if was_data_discovered or had_pending_requests:
            self.invalidate_execution_outputs(state, "知识召回已更新")
        self.planner.artifact_store.write_json("recall", "recall_bundle.json", state["recall_bundle"].model_dump(by_alias=True), preview_chars=0)
        if had_pending_requests:
            state["plan"] = QueryPlan()
            state["query_graph_plan_attempts"] = 0
            state["planner_provider_error"] = ""
        add_step(
            state,
            "Main Agent Tool retrieve_knowledge：完成检索，命中 %d 条候选知识/资产片段，profile=%s skillPolicies=%s"
            % (len(items), ",".join(state["recall_strategy"].get("profileKinds") or []) or "unknown", loaded_skills or []),
        )
        self.record_span(
            state,
            "semantic_tool",
            "retrieve_knowledge",
            started,
            row_count=len(items),
            metadata={"skillPolicies": loaded_skills, "pendingRequests": had_pending_requests, "recallStrategy": state.get("recall_strategy", {})},
        )
        self.finish_run_step(state, step, "success", output_summary="recallItems=%d skills=%s" % (len(items), loaded_skills or []))
        emit(
            state,
            "node.completed",
            "RETRIEVE_KNOWLEDGE",
            {"recallItems": len(items), "skillPolicies": loaded_skills, "recallStrategy": state.get("recall_strategy", {})},
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
            if topic not in set(base_topics)
            and topic
            not in {
                QuestionCategory.UNKNOWN,
                QuestionCategory.IDENTITY,
                QuestionCategory.MERCHANT_OTHER,
            }
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
        if had_pending_requests:
            return "pending_knowledge_topic_expansion"
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
        data_topics = [
            topic
            for topic in topics
            if topic
            not in {
                QuestionCategory.UNKNOWN,
                QuestionCategory.PLATFORM_RULE,
                QuestionCategory.MERCHANT_OTHER,
                QuestionCategory.IDENTITY,
            }
        ]
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        rule_items = [item for item in recall_items[:8] if rule_recall_item(item)]
        keywords = state.get("extracted_keywords") or ExtractedKeywords()
        route_slots = state.get("route_slots") or RouteSlots()
        has_data_intent = self.requires_bi_execution(state, keywords)
        has_analysis_intent = bool(state.get("open_diagnostic_intent"))
        rule_confidence = max([float(item.fusion_score or 0.0) for item in rule_items] or [0.0])
        data_confidence = max([float(item.fusion_score or 0.0) for item in recall_items if not rule_recall_item(item)] or [0.0])
        has_rule_topic = QuestionCategory.PLATFORM_RULE in set(topics)
        rule_needed = bool(rule_items) and (has_rule_topic or not has_data_intent)
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
        if not rule_refs and has_rule_topic:
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
        topics = set(self._effective_topic_categories(state))
        recall_items = (state.get("recall_bundle") or RecallBundle()).items
        has_rule_topic = QuestionCategory.PLATFORM_RULE in topics
        has_rule_recall = any(rule_recall_item(item) for item in recall_items[:6])
        return has_rule_topic or has_rule_recall

    def requires_bi_execution(self, state: AgentState, keywords: ExtractedKeywords | None = None) -> bool:
        question = state.get("question", "")
        text = question.lower()
        keywords = keywords or state.get("extracted_keywords") or ExtractedKeywords()
        route_slots = state.get("route_slots") or RouteSlots()
        if route_slots.operation == "write_requested":
            return False
        if route_slots.object_refs:
            return True
        if re.search(r"\b(order_id|sub_order_id|spu_id|sku_id|refund_id|ticket_id|bill_id)_[a-z0-9_]+\b", text):
            return True
        topics = set(self._effective_topic_categories(state))
        data_topics = {
            topic
            for topic in topics
            if topic
            not in {
                QuestionCategory.UNKNOWN,
                QuestionCategory.PLATFORM_RULE,
                QuestionCategory.MERCHANT_OTHER,
                QuestionCategory.IDENTITY,
            }
        }
        if not data_topics:
            return False
        data_action_terms = [
            "查询",
            "查看",
            "看下",
            "看看",
            "关联",
            "对应",
            "明细",
            "详情",
            "列表",
            "记录",
            "订单量",
            "下单量",
            "退款量",
            "金额",
            "退款率",
            "趋势",
            "top",
            "前",
            "最高",
            "最多",
            "多少",
        ]
        has_data_action = any(term in text for term in data_action_terms)
        if has_data_action:
            return True
        if getattr(keywords, "time_keywords", []) and QuestionCategory.PLATFORM_RULE not in topics:
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
        ref_ids = list(state.get("rule_recall_refs") or self.rule_recall_ref_ids(state))
        intent = QuestionIntent(
            question=state["question"],
            intent_type=IntentType.VALID,
            category=QuestionCategory.PLATFORM_RULE,
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
        self.invalidate_execution_outputs(state, "PlanningAssetPack 重新生成")
        pack = self.asset_builder.compact(
            state["question"],
            state["recall_bundle"],
            self._effective_topic_categories(state),
            self.open_diagnostic_debug(state),
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
        pack.metric_compaction["knowledgeRequestGaps"] = list(state.get("knowledge_request_gaps") or [])
        pack.metric_compaction["loadedSourceRefs"] = sorted(pack.source_refs.keys())
        pack.metric_compaction["recallBackend"] = knowledge_bundle.backend or "hybrid"
        pack.metric_compaction["semanticSourceHash"] = (
            knowledge_bundle.semantic_source_hash
            or pack.metric_compaction.get("cache", {}).get("semanticSourceHash", "")
        )
        pack.metric_compaction["indexVersion"] = knowledge_bundle.index_version
        state["planning_asset_pack"] = pack
        if self.latency_optimizer.blocks_expensive_agents(state.get("latency_optimization") or {}):
            state["hypothesis_exploration"] = {}
        else:
            state["hypothesis_exploration"] = self.controlled_react_explorer.build_hypotheses(
                state["question"],
                pack,
                (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
            )
        state["planning_assets_compacted"] = True
        self.reconcile_fast_request_agent_gates(state)
        state["query_graph_validation_result"] = GraphValidationResult()
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        state["planner_reflection"] = PlannerReflectionResult()
        self.planner.artifact_store.write_json("planner", "planning_asset_pack.json", pack.model_dump(by_alias=True), preview_chars=0)
        add_step(
            state,
            "Main Agent Tool compact_assets：生成 PlanningAssetPack，tables=%d, metrics=%d, fields=%d, relationships=%d"
            % (len(pack.tables), len(pack.metrics), len(pack.fields), len(pack.relationships)),
        )
        add_step(
            state,
            "Controlled ReAct：生成受控探索假设，hypotheses=%d"
            % len((state.get("hypothesis_exploration") or {}).get("hypotheses") or []),
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
        planner_context = {
            "fastUnderstanding": (state.get("fast_understanding") or FastUnderstandingResult()).model_dump(by_alias=True),
            "memoryConstraints": state.get("memory_constraints", []),
        }
        selection_payload: Dict[str, Any] = {}
        plan = QueryPlan(agent_trace=["query_metric.semantic_asset_selection.skipped"])
        selector_attempted = False
        if getattr(getattr(self.planner, "llm", None), "configured", False) and hasattr(self.planner, "semantic_asset_selection_plan"):
            selector_attempted = True
            plan, selection_payload = self.planner.semantic_asset_selection_plan(
                state.get("question", ""),
                state.get("recall_bundle") or RecallBundle(),
                pack,
                planner_context=planner_context,
            )
            if plan.intents:
                plan.agent_trace.append("query_metric.semantic_asset_selection=compiled")
        if not plan.intents and not selector_attempted:
            plan = compile_semantic_metric_fallback_graph(state.get("question", ""), pack)
        catalog_expansion_traces: List[str] = []
        if not plan.intents:
            catalog_expansion_traces = self.asset_builder.expand_for_metric_catalog_resolution(
                pack,
                state.get("question", ""),
            )
            if catalog_expansion_traces:
                state["planning_asset_pack"] = pack
                if selector_attempted:
                    plan, selection_payload = self.planner.semantic_asset_selection_plan(
                        state.get("question", ""),
                        state.get("recall_bundle") or RecallBundle(),
                        pack,
                        planner_context=planner_context,
                    )
                    if plan.intents:
                        plan.agent_trace.append("query_metric.semantic_asset_selection=compiled_after_catalog_expansion")
                else:
                    plan = compile_semantic_metric_fallback_graph(state.get("question", ""), pack)
            fast = state.get("fast_understanding") or FastUnderstandingResult()
            metric_phrase_count = len([phrase for phrase in fast.metric_phrases if str(phrase or "").strip()])
            if not plan.intents and not selector_attempted and metric_phrase_count <= 1:
                ambiguity = self.query_metric_ambiguity(pack, state.get("question", ""))
                if ambiguity:
                    return self.request_query_metric_ambiguity_clarification(state, ambiguity, started, step)
        state["query_metric_trace"] = {
            "status": "compiled" if plan.intents else "unsupported",
            "agentTrace": list(plan.agent_trace or []),
            "compilerTrace": list(plan.compiler_trace or []),
            "catalogExpansion": list(catalog_expansion_traces),
            "catalogMetricCandidates": list((pack.metric_compaction or {}).get("catalogMetricCandidates") or []),
            "semanticAssetSelection": {
                "status": str(selection_payload.get("status") or ""),
                "action": str(selection_payload.get("action") or ""),
                "selectedRefs": list(selection_payload.get("selectedRefs") or []),
                "reason": str(selection_payload.get("reason") or ""),
            } if selection_payload else {},
        }
        selection_action = str(selection_payload.get("action") or "").strip().lower()
        selection_status = str(selection_payload.get("status") or "").strip().upper()
        if not plan.intents and selection_payload and (
            selection_action in {"ask_human", "clarify", "clarification"}
            or selection_status in {"AMBIGUOUS", "NEED_CLARIFICATION"}
        ):
            clarifications = [item for item in (selection_payload.get("clarifications") or []) if isinstance(item, dict)]
            first = clarifications[0] if clarifications else {}
            question = str(first.get("question") or selection_payload.get("reason") or "请确认你想看的指标口径。").strip()
            options: List[str] = []
            for option in first.get("options") or []:
                if isinstance(option, dict):
                    label = str(option.get("label") or option.get("ref") or "").strip()
                else:
                    label = str(option or "").strip()
                if label and label not in options:
                    options.append(label)
            self.request_human_clarification(state, question, "METRIC_SCOPE", "metric_focus", options)
            state["query_metric_trace"] = {
                **dict(state.get("query_metric_trace") or {}),
                "status": "needs_clarification",
                "clarification": {
                    "question": question,
                    "options": options,
                    "reason": str(selection_payload.get("reason") or ""),
                    "clarifications": clarifications,
                },
            }
            add_step(state, "Metric Tool query_metric：语义选择需要确认指标口径，调用 ask_human")
            self.record_span(
                state,
                "semantic_tool",
                "query_metric",
                started,
                status="gap",
                error_code="METRIC_SELECTION_REQUIRES_CLARIFICATION",
                metadata=state["query_metric_trace"],
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="semantic selection -> ask_human",
                error_code="METRIC_SELECTION_REQUIRES_CLARIFICATION",
            )
            emit(state, "node.completed", "QUERY_METRIC", {"supported": False, "clarificationRequired": True})
            return state
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
        state["plan"] = plan
        state["query_graph_reflected"] = True
        state["planner_reflection"] = PlannerReflectionResult()
        state["latency_optimization"] = self.latency_optimizer.update_after_plan(
            state.get("latency_optimization") or {},
            plan,
        )
        validation = self.graph_validator.validate(
            state["question"],
            plan,
            pack,
            state.get("memory_constraints", []),
        )
        validation = validation_with_question_coverage(state["question"], plan, pack, validation)
        state["query_graph_validation_result"] = validation
        state["query_graph_validated"] = True
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
        state["worker_dispatch_context"] = {
            **self.worker_dispatch_context(state),
            "tool": "query_metric",
            "executionMode": "direct",
            "reason": "single_metric_tool_executes_governed_query_graph",
        }
        try:
            node_package = self.prepare_scoped_context_package(
                state,
                "query_metric",
                "MetricTool",
                allowed_tables=pack.known_tables()[:12],
                allowed_metrics=self.planning_metric_keys_for_context(pack)[:24],
            )
            node_knowledge_context = append_context_section(
                knowledge_context(state),
                self.render_context_package_for_prompt(node_package),
                max_chars=int(self.settings.context_runtime_budget_chars or 6000),
            )
            run_result = self.node_worker.execute_plan(
                state["merchant"].merchant_id,
                plan,
                pack,
                node_knowledge_context,
                state["question"],
                resume_task_results=[],
                run_id=state.get("run_id", ""),
                access_role=state.get("access_role", "merchant_operator"),
                user_scope=state.get("user_identity", {}),
                execution_mode="direct",
            )
        except Exception as exc:
            run_result = AgentRunResult(
                merged_query_bundle=QueryBundle(failed=True, error=str(exc), summary="query_metric NodeWorker 执行失败"),
                reflection_notes=["query_metric NodeWorker 执行失败: %s" % str(exc)[:200]],
            )
        state["agent_run_result"] = run_result
        state["query_bundle"] = run_result.merged_query_bundle
        state["query_bundles"] = run_result.query_bundles
        state["node_tool_traces"] = run_result.node_tool_traces
        state["freshness_reports"] = run_result.freshness_reports
        state["sql_generated"] = True
        state["result_generation"] = int(state.get("execution_generation") or 0)
        self.sync_tool_runtime_state(state)
        verified = self.evidence_verifier.verify(
            state["question"],
            plan,
            run_result,
            state.get("memory_constraints", []),
            recall_knowledge_ref_ids(state),
        )
        run_result.verified_evidence = verified
        run_result.evidence_gaps = verified.gaps
        run_result.partial_answer_reason = verified.partial_answer_reason
        graph_repair_gaps = graph_repair_validation_gaps(verified.gaps)
        if graph_repair_gaps:
            state["query_graph_validation_result"] = GraphValidationResult(
                valid=False,
                gaps=graph_repair_gaps,
                repairable=True,
            )
        state["evidence_graph_verified"] = True
        state["verification_status"] = "passed" if verified.passed else "failed"
        state["evidence_accepted"] = bool(verified.passed)
        state["evidence_generation"] = int(state.get("execution_generation") or 0) if verified.passed else -1
        state["agent_run_result"] = run_result
        self.planner.artifact_store.write_json("node", "query_metric_agent_run_result.json", run_result.model_dump(by_alias=True), preview_chars=0)
        failed_tasks = sum(1 for item in run_result.task_results if item.query_bundle.failed)
        if verified.passed and not failed_tasks:
            state["query_metric_completed"] = True
            state["should_persist"] = True
            state["latency_optimization"] = self.latency_optimizer.mark_verified(
                state.get("latency_optimization") or {},
                "query_metric resolved, executed and verified one governed semantic metric",
            )
            self.reconcile_fast_request_agent_gates(state)
            add_step(state, "Metric Tool query_metric：受控单指标 QueryGraph 执行并通过证据校验")
        else:
            self.escalate_fast_request(state, "query_metric evidence verification failed")
            add_step(
                state,
                "Metric Tool query_metric：执行后证据未通过，failedTasks=%d gaps=%d，交回标准链路"
                % (failed_tasks, len(verified.gaps)),
            )
        self.record_span(
            state,
            "semantic_tool",
            "query_metric",
            started,
            status="success" if state.get("query_metric_completed") else "failed",
            row_count=run_result.merged_query_bundle.effective_row_count(),
            error_code="" if state.get("query_metric_completed") else (",".join(gap.code for gap in verified.gaps[:4]) or "QUERY_METRIC_EVIDENCE_FAILED"),
            metadata={
                "tasks": len(run_result.tasks),
                "failedTasks": failed_tasks,
                "validationGaps": len(validation.gaps),
                "evidenceGaps": len(verified.gaps),
                "trace": state.get("query_metric_trace") or {},
            },
        )
        self.finish_run_step(
            state,
            step,
            "success" if state.get("query_metric_completed") else "gap",
            output_summary="tasks=%d rows=%d evidencePassed=%s"
            % (len(run_result.tasks), run_result.merged_query_bundle.effective_row_count(), verified.passed),
            error_code="" if state.get("query_metric_completed") else (",".join(gap.code for gap in verified.gaps[:4]) or "QUERY_METRIC_EVIDENCE_FAILED"),
            artifact_paths=[str(Path(state["thread_data"].outputs_path) / "artifacts" / "node" / "query_metric_agent_run_result.json")],
        )
        emit(
            state,
            "node.completed",
            "QUERY_METRIC",
            {
                "supported": True,
                "valid": True,
                "tasks": len(run_result.tasks),
                "rows": run_result.merged_query_bundle.effective_row_count(),
                "evidencePassed": verified.passed,
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
            "threadContext": state.get("thread_context", {}),
            "conversationContext": planner_conversation_context(state),
            "runtimeInjection": state.get("runtime_injection", {}),
            "memoryInjection": state.get("memory_injection", {}),
            "memoryConstraints": state.get("memory_constraints", []),
        }
        planner_context = self.context_assembler.assemble_payload(
            state,
            "planner_context",
            "PlannerAgent",
            planner_context,
            budget_chars=int(self.settings.context_planner_budget_chars or 12000),
        )
        # Planner follows Diana-style progressive semantic reads: start from
        # workspace refs/catalog only, then semantic_read details on demand.
        planner_knowledge_context = ""
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
        if self.settings.calendar_time_semantics_enabled:
            time_window_contract = resolve_time_window_contract(state["question"], self.settings.business_timezone)
            state["time_window_contract"] = time_window_contract
            plan = apply_time_window_contract_to_plan(plan, time_window_contract)
        state["plan"] = plan
        state["candidate_query_graphs"] = self.controlled_react_explorer.evaluate_candidates(
            state.get("hypothesis_exploration") or {},
            state["planning_asset_pack"],
            plan,
        )
        for candidate in state["candidate_query_graphs"].get("candidates", []):
            candidate_plan = QueryPlan.model_validate(candidate.get("queryGraph") or {})
            candidate_validation = self.graph_validator.validate(
                state["question"],
                candidate_plan,
                state["planning_asset_pack"],
                state.get("memory_constraints", []),
            )
            candidate_validation = validation_with_question_coverage(
                state["question"],
                candidate_plan,
                state["planning_asset_pack"],
                candidate_validation,
            )
            candidate["validation"] = candidate_validation.model_dump(by_alias=True)
            candidate["safeToExecute"] = bool(candidate_validation.valid)
            if not candidate_validation.valid:
                candidate["score"] = max(0, int(candidate.get("score") or 0) - 40)
        state["candidate_query_graphs"]["candidates"].sort(
            key=lambda item: (bool(item.get("safeToExecute")), int(item.get("score") or 0)),
            reverse=True,
        )
        for index, candidate in enumerate(state["candidate_query_graphs"].get("candidates", [])):
            candidate["status"] = "selected" if index == 0 else "validated_alternative"
        if state["candidate_query_graphs"].get("candidates"):
            state["candidate_query_graphs"]["selectedCandidateId"] = state["candidate_query_graphs"]["candidates"][0]["candidateId"]
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
        state["query_graph_validated"] = False
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
        reflection = self.planner_reflection_agent.reflect(state["question"], state["plan"], state["planning_asset_pack"])
        state["planner_reflection"] = reflection
        state["planner_repair_reason"] = reflection.repair_reason
        state["planner_repair_requests"] = reflection.repair_requests
        state["query_graph_reflected"] = True
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
        state["query_graph_validation_result"] = result
        state["query_graph_validated"] = True
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
        state["query_graph_repair_attempts"] = int(state.get("query_graph_repair_attempts") or 0) + 1
        emit(state, "node.started", "REPAIR_QUERY_GRAPH", {})
        before_nodes = len(state["plan"].intents)
        repair_reason = state.get("planner_repair_reason", "")
        repair_requests = list(state.get("planner_repair_requests", []))
        state["plan"] = self.planner.repair(
            state["question"],
            state["plan"],
            state["planning_asset_pack"],
            state["query_graph_validation_result"].gaps,
            state.get("history_rows", []),
            knowledge_context(state),
            state["recall_bundle"],
        )
        if state["plan"].knowledge_requests:
            state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
                state,
                dedupe_workflow_knowledge_requests(
                    list(state.get("pending_knowledge_requests") or []) + list(state["plan"].knowledge_requests or [])
                ),
            )
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        state["planner_reflection"] = PlannerReflectionResult()
        state["planner_repair_reason"] = ""
        self.invalidate_execution_outputs(state, "QueryGraph 修复后需要重新执行")
        add_step(state, "Main Agent Tool repair_query_graph：完成 QueryGraph 修复尝试")
        repair_artifact = Path(state["thread_data"].outputs_path) / "artifacts" / "planner" / ("repair_attempt_%d.json" % state["query_graph_repair_attempts"])
        try:
            self.planner.artifact_store.write_json(
                "planner",
                "repair_attempt_%d.json" % state["query_graph_repair_attempts"],
                {
                    "attempt": state["query_graph_repair_attempts"],
                    "repairReason": repair_reason,
                    "repairRequests": [item.model_dump(by_alias=True) for item in repair_requests],
                    "beforeNodes": before_nodes,
                    "afterNodes": len(state["plan"].intents),
                    "plan": state["plan"].model_dump(by_alias=True),
                },
                preview_chars=0,
            )
        except Exception:
            pass
        self.record_span(
            state,
            "planner_repair",
            "repair_query_graph",
            started,
            metadata={
                "repairReason": repair_reason,
                "repairRequests": [item.model_dump(by_alias=True) for item in repair_requests],
                "beforeNodes": before_nodes,
                "afterNodes": len(state["plan"].intents),
                "attempt": state["query_graph_repair_attempts"],
            },
        )
        self.finish_run_step(
            state,
            step,
            "success",
            output_summary="beforeNodes=%d afterNodes=%d" % (before_nodes, len(state["plan"].intents)),
            artifact_paths=[str(repair_artifact)],
        )
        emit(state, "node.completed", "REPAIR_QUERY_GRAPH", {"attempt": state["query_graph_repair_attempts"]})
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
        validation = state.get("query_graph_validation_result")
        if state.get("query_graph_validated") and validation and not validation.valid:
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
                for gap in validation.gaps
            ]
            run_result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    failed=True,
                    error="QueryGraph validation failed before SQL execution",
                    summary="QueryGraph 未通过校验，NodeAgent 未执行 SQL",
                ),
                evidence_gaps=gaps,
                partial_answer_reason="QUERY_GRAPH_VALIDATION_FAILED",
                reflection_notes=["NodeAgent skipped because QueryGraph validation failed"],
            )
            state["agent_run_result"] = run_result
            state["query_bundle"] = run_result.merged_query_bundle
            state["query_bundles"] = []
            state["node_tool_traces"] = []
            state["freshness_reports"] = []
            state["sql_generated"] = True
            state["result_generation"] = int(state.get("execution_generation") or 0)
            self.planner.artifact_store.write_json("node", "agent_run_result.json", run_result.model_dump(by_alias=True), preview_chars=0)
            add_step(state, "Main Agent Tool execute_query_graph：QueryGraph 未通过校验，跳过 NodeWorker SQL 执行")
            self.record_span(
                state,
                "action",
                "execute_query_graph",
                started,
                status="failed",
                error_code="QUERY_GRAPH_VALIDATION_FAILED",
                metadata={"gaps": [gap.model_dump(by_alias=True) for gap in validation.gaps[:12]]},
            )
            self.finish_run_step(
                state,
                step,
                "gap",
                output_summary="skipped SQL because queryGraph valid=false gaps=%d" % len(validation.gaps),
                error_code="QUERY_GRAPH_VALIDATION_FAILED",
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
                access_role=state.get("access_role", "merchant_operator"),
                user_scope=state.get("user_identity", {}),
                execution_mode=state.get("node_execution_mode", "auto"),
            )
        except Exception as exc:
            run_result = AgentRunResult(
                merged_query_bundle=QueryBundle(failed=True, error=str(exc), summary="NodeWorker 执行失败"),
                reflection_notes=["NodeWorker 执行失败: %s" % str(exc)[:200]],
            )
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
            state["query_graph_validation_result"] = GraphValidationResult(
                valid=False,
                gaps=graph_gaps,
                repairable=True,
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
            state["query_graph_validation_result"] = GraphValidationResult(
                valid=False,
                gaps=graph_repair_gaps,
                repairable=True,
            )
        add_step(state, "Main Agent Tool verify_evidence_graph：" + ("证据门禁通过" if verified.passed else "证据存在缺口 %d 个" % len(verified.gaps)))
        state["evidence_graph_verified"] = True
        state["verification_status"] = "passed" if verified.passed else "failed"
        state["evidence_accepted"] = bool(verified.passed)
        state["evidence_generation"] = int(state.get("execution_generation") or 0) if verified.passed else -1
        if not verified.passed and self.latency_optimizer.blocks_expensive_agents(state.get("latency_optimization") or {}):
            self.escalate_fast_request(state, "evidence verification failed on the fast path")
        if (
            verified.passed
            and
            not state.get("hypothesis_exploration_completed")
            and not self.latency_optimizer.blocks_expensive_agents(state.get("latency_optimization") or {})
        ):
            distributed_client = getattr(self.node_worker, "distributed_subagent_client", None)
            if distributed_client and (state.get("hypothesis_exploration") or {}).get("hypotheses"):
                task_id = "hypothesis_review_%s" % uuid.uuid4().hex[:10]
                distributed = distributed_client.execute(
                    state.get("run_id") or state.get("qa_id") or "hypothesis_run",
                    task_id,
                    "hypothesis_review",
                    {
                        "hypotheses": state.get("hypothesis_exploration") or {},
                        "runResult": (state.get("agent_run_result") or AgentRunResult()).model_dump(by_alias=True),
                    },
                    timeout_seconds=max(1, int(self.settings.agent_node_timeout_seconds or 1)),
                )
                state["hypothesis_results"] = list(distributed.result.get("reviews") or []) if distributed.status == "completed" else []
                state.setdefault("tool_runtime_events", []).append(
                    {
                        "event": "hypothesis.review.distributed",
                        "taskId": task_id,
                        "status": distributed.status,
                        "resultArtifactUri": distributed.artifact_uri,
                    }
                )
            else:
                state["hypothesis_results"] = self.controlled_react_explorer.run_parallel_evidence_reviews(
                    state.get("hypothesis_exploration") or {},
                    state.get("agent_run_result") or AgentRunResult(),
                )
            if state["hypothesis_results"]:
                add_step(state, "多假设预检查：已基于基线证据判断 %d 个经营假设是否值得独立查询" % len(state["hypothesis_results"]))
        elif not verified.passed:
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
        started = now_ms()
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        limit = max(1, min(3, int(getattr(self.settings, "hypothesis_max_candidates", 3) or 3)))
        hypotheses = hypotheses[:limit]
        step = self.start_run_step(
            state,
            "explore_hypotheses",
            "LeadAgent",
            "EXPLORE_HYPOTHESES",
            input_summary="hypotheses=%d" % len(hypotheses),
        )
        increment_round(state)
        emit(state, "node.started", "EXPLORE_HYPOTHESES", {"hypotheses": len(hypotheses)})
        recovery_mode = bool(
            state.get("planner_provider_error")
            and state.get("planning_assets_compacted")
            and not (state.get("plan") and getattr(state.get("plan"), "intents", None))
        )
        if not evidence_accepted_for_state(state) and not recovery_mode:
            state["hypothesis_exploration_status"] = {"status": "blocked", "source": "evidence_gate"}
            self.finish_run_step(state, step, "gap", output_summary="blocked: evidence verification failed", error_code="EVIDENCE_NOT_ACCEPTED")
            emit(state, "node.completed", "EXPLORE_HYPOTHESES", {"completed": False, "executed": 0, "evidenceAccepted": False})
            return state
        if planner_degraded_stops_expensive_work(state):
            state["hypothesis_exploration_status"] = {"status": "skipped", "source": "planner_degraded"}
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
                "EXPLORE_HYPOTHESES",
                {"completed": False, "executed": 0, "degradedSkipped": True},
            )
            return state
        answer_reserve = max(5, int(getattr(self.settings, "hypothesis_answer_reserve_seconds", 15) or 15))
        remaining_at_start = remaining_run_budget_seconds(state, self.settings)
        planner_allowance = max(5, min(25, int(getattr(self.settings, "llm_planner_timeout_seconds", 25) or 25)))
        minimum_stage_budget = answer_reserve + planner_allowance + 5
        if remaining_at_start <= minimum_stage_budget:
            state["hypothesis_exploration_status"] = {"status": "skipped", "source": "run_budget"}
            state["hypothesis_evidence_ledger"] = HypothesisEvidenceLedger(
                ledger_id="ledger_%s" % state.get("run_id", "run"),
                budget={
                    "remainingSeconds": remaining_at_start,
                    "requiredSeconds": minimum_stage_budget,
                    "answerReserveSeconds": answer_reserve,
                    "skipped": True,
                    "reason": "INSUFFICIENT_RUN_BUDGET",
                },
            )
            self.finish_run_step(state, step, "partial", output_summary="skipped: insufficient run budget", error_code="INSUFFICIENT_RUN_BUDGET")
            emit(state, "node.completed", "EXPLORE_HYPOTHESES", {"completed": True, "executed": 0, "budgetSkipped": True})
            return state
        if len(hypotheses) < 2:
            state["hypothesis_exploration_completed"] = True
            state["hypothesis_exploration_status"] = {"status": "not_applicable", "source": "runtime"}
            self.finish_run_step(state, step, "success", output_summary="not enough competing hypotheses")
            emit(state, "node.completed", "EXPLORE_HYPOTHESES", {"completed": True, "executed": 0})
            return state

        max_parallel = max(1, min(limit, int(getattr(self.settings, "hypothesis_max_parallel_queries", 3) or 3)))
        planned: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                submit_with_current_context(executor, self._generate_independent_hypothesis_plan, state, hypothesis, 1, None, None): hypothesis
                for hypothesis in hypotheses
            }
            for future in as_completed(futures):
                hypothesis = futures[future]
                try:
                    planned.append(future.result())
                except Exception as exc:
                    planned.append(
                        {
                            "hypothesis": hypothesis,
                            "hypothesisId": str(hypothesis.get("hypothesisId") or ""),
                            "plan": QueryPlan(),
                            "validation": GraphValidationResult(valid=False),
                            "planningError": str(exc)[:500],
                            "round": 1,
                        }
                    )
        safe_plans = [item for item in planned if item["plan"].intents and item["validation"].valid]
        executions = self._execute_hypothesis_plans_parallel(state, safe_plans, max_parallel)
        executed_ids = {str(item.get("hypothesisId") or "") for item in executions}
        for item in planned:
            if str(item.get("hypothesisId") or "") in executed_ids:
                continue
            executions.append(
                {
                    **item,
                    "runResult": AgentRunResult(),
                    "semanticScore": self._hypothesis_semantic_score(state, str(item.get("hypothesisId") or "")),
                    "executionError": item.get("planningError") or "QUERY_GRAPH_VALIDATION_FAILED",
                }
            )
        comparison = self.controlled_react_explorer.compare_independent_executions(
            executions,
            min_score=int(getattr(self.settings, "hypothesis_min_survivor_score", 45) or 45),
            max_survivors=int(getattr(self.settings, "hypothesis_max_survivors", 2) or 2),
        )
        rounds_used = 1
        max_rounds = max(1, min(2, int(getattr(self.settings, "hypothesis_max_rounds", 2) or 2)))
        if max_rounds > 1 and comparison.get("survivorIds"):
            survivors = [item for item in comparison["ranked"] if item.get("decision") == "survive"]
            followup_plans: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(survivors)))) as executor:
                futures = {}
                for survivor in survivors:
                    decision = self.controlled_react_explorer.followup_decision(
                        survivor,
                        remaining_seconds=remaining_run_budget_seconds(state, self.settings),
                        minimum_information_gain=float(getattr(self.settings, "hypothesis_second_round_min_information_gain", 0.35) or 0.35),
                        answer_reserve_seconds=answer_reserve,
                    )
                    survivor["followupDecision"] = decision
                    if decision.get("action") == "stop":
                        continue
                    future = submit_with_current_context(
                        executor,
                        self._generate_independent_hypothesis_plan,
                        state,
                        survivor["hypothesis"],
                        2,
                        survivor,
                        decision,
                    )
                    futures[future] = survivor
                for future in as_completed(futures):
                    survivor = futures[future]
                    try:
                        followup_plans.append(future.result())
                    except Exception as exc:
                        survivor["followupError"] = str(exc)[:500]
            safe_followups = [item for item in followup_plans if item["plan"].intents and item["validation"].valid]
            followup_executions = self._execute_hypothesis_plans_parallel(state, safe_followups, max_parallel)
            if followup_executions:
                rounds_used = 2
                by_id = {str(item.get("hypothesisId") or ""): item for item in comparison["ranked"]}
                for followup in followup_executions:
                    hypothesis_id = str(followup.get("hypothesisId") or "")
                    if hypothesis_id not in by_id:
                        continue
                    base = by_id[hypothesis_id]
                    base.setdefault("followups", []).append(followup)
                    base["plan"] = merge_query_plans(base["plan"], followup["plan"])
                    base["runResult"] = merge_agent_run_results(base["runResult"], followup["runResult"])
                    base["runResult"].verified_evidence = self.evidence_verifier.verify(
                        state["question"],
                        base["plan"],
                        base["runResult"],
                        state.get("memory_constraints", []),
                        recall_knowledge_ref_ids(state),
                    )
                comparison = self.controlled_react_explorer.compare_independent_executions(
                    list(by_id.values()),
                    min_score=int(getattr(self.settings, "hypothesis_min_survivor_score", 45) or 45),
                    max_survivors=int(getattr(self.settings, "hypothesis_max_survivors", 2) or 2),
                )

        selected = [item for item in comparison.get("ranked", []) if item.get("decision") == "survive" and item.get("runResult")]
        ledger = self._build_hypothesis_evidence_ledger(state, comparison, rounds_used)
        state["hypothesis_evidence_ledger"] = ledger
        promoted_count = self._promote_hypothesis_winner_when_baseline_missing(state, selected)
        state["hypothesis_exploration_completed"] = True
        state["hypothesis_exploration_status"] = {"status": "completed", "source": "runtime"}
        state["hypothesis_exploration_rounds"] = rounds_used
        state["hypothesis_selected_ids"] = list(comparison.get("survivorIds") or [])
        state["hypothesis_results"] = [self._public_hypothesis_execution(item) for item in comparison.get("ranked", [])]
        state.setdefault("candidate_query_graphs", {})["independentExecutions"] = state["hypothesis_results"]
        state["candidate_query_graphs"]["comparison"] = {
            "winnerId": comparison.get("winnerId", ""),
            "survivorIds": comparison.get("survivorIds", []),
            "prunedIds": comparison.get("prunedIds", []),
            "roundsUsed": rounds_used,
            "comparisonPolicy": comparison.get("comparisonPolicy", ""),
        }
        state["runtime_context"] = append_context_section(
            state.get("runtime_context") or "",
            self._render_hypothesis_ledger_for_answer(ledger),
            max_chars=int(self.settings.context_runtime_budget_chars or 6000),
        )
        if promoted_count:
            state["sql_generated"] = True
            state["result_generation"] = int(state.get("execution_generation") or 0)
            state["query_graph_validated"] = True
            state["query_graph_reflected"] = True
            state["planner_provider_error"] = ""
            state["should_persist"] = True
        state["last_action_result"] = ActionResult(
            action="explore_hypotheses",
            node="explore_hypotheses",
            status="success" if selected else "partial",
            message="independent hypotheses executed=%d survivors=%d promotedWinnerTasks=%d" % (len(executions), len(selected), promoted_count),
        )
        self.planner.artifact_store.write_json(
            "planner",
            "hypothesis_exploration.json",
            {
                "hypotheses": state["hypothesis_results"],
                "comparison": state["candidate_query_graphs"]["comparison"],
                "evidenceLedger": ledger.model_dump(by_alias=True),
            },
            preview_chars=0,
        )
        add_step(
            state,
            "多假设独立探索：执行 %d 个独立 QueryGraph，保留 %d 个，淘汰 %d 个，探索轮次=%d"
            % (len(executions), len(selected), len(comparison.get("prunedIds") or []), rounds_used),
        )
        self.record_span(
            state,
            "agent",
            "explore_hypotheses",
            started,
            status="success" if selected else "failed",
            row_count=sum(int(item.get("rowCount") or 0) for item in comparison.get("ranked", [])),
            metadata={
                "survivorIds": comparison.get("survivorIds", []),
                "prunedIds": comparison.get("prunedIds", []),
                "roundsUsed": rounds_used,
                "promotedWinnerTasks": promoted_count,
            },
        )
        self.finish_run_step(
            state,
            step,
            "success" if selected else "partial",
            output_summary="executed=%d survivors=%d rounds=%d" % (len(executions), len(selected), rounds_used),
        )
        emit(
            state,
            "node.completed",
            "EXPLORE_HYPOTHESES",
            {
                "completed": True,
                "executed": len(executions),
                "survivorIds": comparison.get("survivorIds", []),
                "prunedIds": comparison.get("prunedIds", []),
                "roundsUsed": rounds_used,
            },
        )
        return state

    def _generate_independent_hypothesis_plan(
        self,
        state: AgentState,
        hypothesis: Dict[str, Any],
        round_index: int,
        previous: Optional[Dict[str, Any]],
        followup: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypothesis_id = str(hypothesis.get("hypothesisId") or "hypothesis")
        namespace = "hyp_%s_r%d" % (re.sub(r"[^a-zA-Z0-9_]+", "_", hypothesis_id), round_index)
        root = Path(state["thread_data"].outputs_path) / "hypotheses" / hypothesis_id / ("round_%d" % round_index)
        root.mkdir(parents=True, exist_ok=True)
        planner = QueryGraphPlanner(
            LlmClient(self.settings),
            semantic_catalog=self.semantic_catalog,
            artifact_store=self.planner.artifact_store.with_root(str(root)),
            settings=self.settings,
        )
        previous_summary = self._hypothesis_result_summary(previous) if previous else {}
        followup_action = str((followup or {}).get("action") or "")
        instruction = (
            "%s\n\n你正在独立验证一个经营假设，不得把其他假设的结论当作证据。\n"
            "假设：%s\n依据：%s\n建议指标：%s\n所需证据：%s\n"
            "请为这个假设单独生成可执行 QueryGraph。每个节点必须使用语义资产中的真实表和字段。"
            % (
                state["question"],
                hypothesis.get("title", ""),
                hypothesis.get("reason", ""),
                ", ".join(str(item) for item in hypothesis.get("metricHints") or []),
                ", ".join(str(item) for item in hypothesis.get("requiredEvidence") or []),
            )
        )
        if round_index > 1:
            instruction += (
                "\n这是第二轮探索。上一轮证据摘要：%s。主 Agent 决定：%s（%s）。"
                "不要原样重复上一轮 QueryGraph；优先增加业务维度、补充证据节点或切换到能够验证同一假设的其他语义表。"
                % (json.dumps(previous_summary, ensure_ascii=False, default=str), followup_action, (followup or {}).get("reason", ""))
            )
        plan, requests, reason = planner.plan(
            instruction,
            [],
            knowledge_context(state),
            state["recall_bundle"],
            state["planning_asset_pack"],
            [],
            ["independent_hypothesis=%s" % hypothesis_id, "exploration_round=%d" % round_index],
            {
                "hypothesis": hypothesis,
                "explorationRound": round_index,
                "previousEvidence": previous_summary,
                "followupDecision": followup or {},
                "memoryConstraints": state.get("memory_constraints", []),
            },
        )
        planning_mode = "independent_planner"
        if not plan.intents and previous and followup_action:
            plan = self.controlled_react_explorer.fallback_followup_plan(
                previous.get("plan") or QueryPlan(),
                hypothesis,
                state["planning_asset_pack"],
                followup_action,
                namespace,
            )
            planning_mode = "safe_semantic_followup_fallback"
        if not plan.intents and round_index == 1:
            plan = self.controlled_react_explorer._independent_candidate_plan(state["plan"], hypothesis)
            planning_mode = "safe_projected_fallback"
        if not plan.intents and round_index == 1:
            context = state.get("request_context")
            days = int(getattr(context, "days", 0) or 7) if context else 7
            plan = self.controlled_react_explorer.fallback_hypothesis_seed_plan(
                hypothesis,
                state["planning_asset_pack"],
                state["question"],
                namespace,
                days=days,
            )
            planning_mode = "safe_semantic_seed_fallback"
        plan = namespace_query_plan(plan, namespace)
        if round_index > 1 and previous and query_plan_fingerprint(plan) == query_plan_fingerprint(previous.get("plan") or QueryPlan()):
            fallback = self.controlled_react_explorer.fallback_followup_plan(
                previous.get("plan") or QueryPlan(),
                hypothesis,
                state["planning_asset_pack"],
                followup_action,
                namespace,
            )
            if fallback.intents:
                plan = namespace_query_plan(fallback, namespace)
                planning_mode = "safe_semantic_followup_fallback"
        validation = self.graph_validator.validate(
            state["question"],
            plan,
            state["planning_asset_pack"],
            state.get("memory_constraints", []),
        )
        validation = validation_with_question_coverage(
            state["question"],
            plan,
            state["planning_asset_pack"],
            validation,
        )
        return {
            "hypothesis": hypothesis,
            "hypothesisId": hypothesis_id,
            "plan": plan,
            "validation": validation,
            "round": round_index,
            "planningMode": planning_mode,
            "planningReason": reason,
            "knowledgeRequests": [item.model_dump(by_alias=True) for item in requests or []],
            "semanticScore": self._hypothesis_semantic_score(state, hypothesis_id),
        }

    def _execute_hypothesis_plans_parallel(
        self,
        state: AgentState,
        planned: List[Dict[str, Any]],
        max_parallel: int,
    ) -> List[Dict[str, Any]]:
        if not planned:
            return []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in planned:
            grouped.setdefault(query_plan_fingerprint(item.get("plan") or QueryPlan()), []).append(item)
        representatives = [items[0] for items in grouped.values()]
        executions: List[Dict[str, Any]] = []
        completed_by_fingerprint: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(representatives)))) as executor:
            futures = {
                submit_with_current_context(executor, self._execute_hypothesis_plan, state, item): (
                    item,
                    query_plan_fingerprint(item.get("plan") or QueryPlan()),
                )
                for item in representatives
            }
            for future in as_completed(futures):
                item, fingerprint = futures[future]
                try:
                    execution = future.result()
                except Exception as exc:
                    execution = {**item, "runResult": AgentRunResult(), "executionError": str(exc)[:500]}
                completed_by_fingerprint[fingerprint] = execution
                executions.append(execution)
        for fingerprint, items in grouped.items():
            representative = completed_by_fingerprint.get(fingerprint)
            if representative is None:
                continue
            for duplicate in items[1:]:
                executions.append(self._reuse_hypothesis_execution(state, representative, duplicate))
        return executions

    def _reuse_hypothesis_execution(
        self,
        state: AgentState,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_plan = source.get("plan") or QueryPlan()
        target_plan = target.get("plan") or QueryPlan()
        source_result = source.get("runResult") or AgentRunResult()
        reused_result = source_result.model_copy(deep=True)
        task_id_mapping = {
            source_intent.plan_task_id: target_intent.plan_task_id
            for source_intent, target_intent in zip(source_plan.intents, target_plan.intents)
        }
        for task in reused_result.tasks:
            task.task_id = task_id_mapping.get(task.task_id, task.task_id)
            task.depends_on = [task_id_mapping.get(item, item) for item in task.depends_on]
        for task_result in reused_result.task_results:
            task_result.task_id = task_id_mapping.get(task_result.task_id, task_result.task_id)
        reused_result.verified_evidence = self.evidence_verifier.verify(
            state["question"], target_plan, reused_result, state.get("memory_constraints", []), recall_knowledge_ref_ids(state)
        )
        reused_result.evidence_gaps = reused_result.verified_evidence.gaps
        return {
            **target,
            "runResult": reused_result,
            "reusedFromHypothesisId": str(source.get("hypothesisId") or ""),
            "executionReuse": "identical_query_plan_fingerprint",
        }

    def _execute_hypothesis_plan(self, state: AgentState, planned: Dict[str, Any]) -> Dict[str, Any]:
        coverage_gaps = query_plan_question_coverage_gaps(
            state["question"],
            planned["plan"],
            state["planning_asset_pack"],
        )
        if coverage_gaps:
            return {
                **planned,
                "runResult": AgentRunResult(),
                "executionError": "QUESTION_COVERAGE_REJECTED:%s"
                % ",".join("%s=%s" % (gap.code, gap.evidence) for gap in coverage_gaps[:8]),
            }
        hypothesis_id = str(planned.get("hypothesisId") or "hypothesis")
        round_index = int(planned.get("round") or 1)
        root = Path(state["thread_data"].outputs_path) / "hypotheses" / hypothesis_id / ("round_%d" % round_index) / "worker"
        worker = NodeWorkerExecutor(
            LlmClient(self.settings),
            self.node_worker.doris_repository,
            SqlValidationService(),
            self.settings,
            semantic_catalog=self.semantic_catalog,
        )
        worker.with_artifact_root(str(root))
        run_result = worker.execute_plan(
            state["merchant"].merchant_id,
            planned["plan"],
            state["planning_asset_pack"],
            knowledge_context(state),
            state["question"],
            run_id="%s_%s_r%d" % (state.get("run_id", "run"), hypothesis_id, round_index),
            access_role=state.get("access_role", "merchant_analyst"),
            user_scope=state.get("user_identity", {}),
            execution_mode=str((state.get("execution_tier_policy") or {}).get("defaultMode") or "auto"),
        )
        run_result.verified_evidence = self.evidence_verifier.verify(
            state["question"],
            planned["plan"],
            run_result,
            state.get("memory_constraints", []),
            recall_knowledge_ref_ids(state),
        )
        run_result.evidence_gaps = run_result.verified_evidence.gaps
        return {**planned, "runResult": run_result}

    def _build_hypothesis_evidence_ledger(
        self,
        state: AgentState,
        comparison: Dict[str, Any],
        rounds_used: int,
    ) -> HypothesisEvidenceLedger:
        entries: List[HypothesisLedgerEntry] = []
        for execution in comparison.get("ranked", []):
            hypothesis = execution.get("hypothesis") or {}
            hypothesis_id = str(execution.get("hypothesisId") or "")
            plan = execution.get("plan") or QueryPlan()
            run_result = execution.get("runResult") or AgentRunResult()
            intents = {intent.plan_task_id: intent for intent in plan.intents}
            records: List[HypothesisEvidenceRecord] = []
            for task_result in run_result.task_results:
                bundle = task_result.query_bundle
                intent = intents.get(task_result.task_id) or QuestionIntent()
                matching_gaps = [
                    gap.model_dump(by_alias=True)
                    for gap in run_result.verified_evidence.gaps
                    if not gap.task_id or gap.task_id == task_result.task_id
                ][:8]
                if bundle.failed:
                    status = "failed"
                elif bundle.effective_row_count() <= 0:
                    status = "insufficient"
                else:
                    status = "supporting"
                evidence_id = "evidence_%s" % hashlib.sha256(
                    ("%s|%s|%s|%s" % (hypothesis_id, task_result.task_id, bundle.sql, bundle.effective_row_count())).encode("utf-8")
                ).hexdigest()[:16]
                records.append(
                    HypothesisEvidenceRecord(
                        evidence_id=evidence_id,
                        hypothesis_id=hypothesis_id,
                        round=2 if "_r2_" in task_result.task_id else 1,
                        task_id=task_result.task_id,
                        claim_key=intent.analysis_note or intent.metric_name or intent.metric_column or hypothesis.get("title", ""),
                        metric_name=intent.metric_name or intent.metric_column,
                        metric_formula=intent.metric_formula,
                        table=intent.preferred_table or (bundle.tables[0] if bundle.tables else ""),
                        time_range="最近%d天" % int(intent.days or 0) if int(intent.days or 0) else "",
                        sql_hash=hashlib.sha256(str(bundle.sql or "").encode("utf-8")).hexdigest()[:16] if bundle.sql else "",
                        row_count=bundle.effective_row_count(),
                        status=status,
                        confidence=(float(execution.get("evidenceScore") or 0) / 100.0) if status == "supporting" else 0.0,
                        evidence_preview=list(bundle.rows or [])[:3] if status == "supporting" else [],
                        gaps=matching_gaps,
                        failure_reason=bundle.error or task_result.summary if status in {"failed", "insufficient"} else "",
                        reused_from_hypothesis_id=str(execution.get("reusedFromHypothesisId") or ""),
                    )
                )
            supporting = [item.evidence_id for item in records if item.status == "supporting"]
            insufficient = [item.evidence_id for item in records if item.status == "insufficient"]
            failed = [item.evidence_id for item in records if item.status == "failed"]
            decision = str(execution.get("decision") or "pruned")
            elimination_reason = ""
            if decision == "pruned":
                if execution.get("executionError"):
                    elimination_reason = str(execution.get("executionError"))[:300]
                elif not supporting:
                    elimination_reason = "没有取得可支持该假设的独立查询证据"
                else:
                    elimination_reason = "与其他假设相比证据得分或覆盖度较低"
            entries.append(
                HypothesisLedgerEntry(
                    hypothesis_id=hypothesis_id,
                    title=str(hypothesis.get("title") or ""),
                    reason=str(hypothesis.get("reason") or ""),
                    status=decision,
                    rank=int(execution.get("rank") or 0),
                    evidence_score=int(execution.get("evidenceScore") or 0),
                    semantic_score=int(execution.get("semanticScore") or 0),
                    confidence=float(execution.get("evidenceScore") or 0) / 100.0,
                    query_graphs=[plan.model_dump(by_alias=True)],
                    evidence=records,
                    supporting_evidence_ids=supporting,
                    insufficient_evidence_ids=insufficient,
                    failed_evidence_ids=failed,
                    evidence_gaps=[gap.model_dump(by_alias=True) for gap in run_result.verified_evidence.gaps[:12]],
                    elimination_reason=elimination_reason,
                    followup_decision=dict(execution.get("followupDecision") or {}),
                )
            )
        budget = dict(state.get("run_budget_report") or {})
        budget["remainingSeconds"] = remaining_run_budget_seconds(state, self.settings)
        return HypothesisEvidenceLedger(
            ledger_id="ledger_%s" % state.get("run_id", "run"),
            winner_id=str(comparison.get("winnerId") or ""),
            survivor_ids=list(comparison.get("survivorIds") or []),
            pruned_ids=list(comparison.get("prunedIds") or []),
            entries=entries,
            rounds_used=rounds_used,
            budget=budget,
            comparison_policy=str(comparison.get("comparisonPolicy") or ""),
        )

    def _promote_hypothesis_winner_when_baseline_missing(self, state: AgentState, selected: List[Dict[str, Any]]) -> int:
        baseline = state.get("agent_run_result") or AgentRunResult()
        if any(not item.query_bundle.failed and item.query_bundle.effective_row_count() > 0 for item in baseline.task_results):
            return 0
        winner_id = str((state.get("hypothesis_evidence_ledger") or HypothesisEvidenceLedger()).winner_id or "")
        winner = next((item for item in selected if str(item.get("hypothesisId") or "") == winner_id), None)
        if not winner:
            return 0
        source_result = winner.get("runResult") or AgentRunResult()
        winner_validation = winner.get("validation") or GraphValidationResult()
        if not getattr(winner_validation, "valid", False) or not source_result.verified_evidence.passed:
            return 0
        successful = [
            item.model_copy(deep=True)
            for item in source_result.task_results
            if not item.query_bundle.failed and item.query_bundle.effective_row_count() > 0
        ]
        if not successful:
            return 0
        executable_ids = {
            intent.plan_task_id
            for intent in (winner.get("plan") or QueryPlan()).intents
            if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE
        }
        if {item.task_id for item in successful} != executable_ids:
            return 0
        curated = AgentRunResult(
            task_results=successful,
            query_bundles=[item.query_bundle for item in successful],
            node_tool_traces=[trace for item in successful for trace in item.node_tool_traces],
            freshness_reports=[report for item in successful for report in item.freshness_reports],
            reflection_notes=["promoted from verified hypothesis evidence ledger winner=%s" % winner_id],
        )
        curated.merged_query_bundle = merge_task_result_bundles(curated.task_results)
        winner_plan = winner.get("plan") or QueryPlan()
        successful_ids = {item.task_id for item in successful}
        promoted_understanding = dict(winner_plan.question_understanding or {})
        fast = state.get("fast_understanding") or FastUnderstandingResult()
        diagnostic_intent = str(getattr(fast, "analysis_intent", "") or state.get("open_diagnostic_intent") or "").strip().lower()
        if diagnostic_intent and diagnostic_intent not in {"none", "metric_query", "detail_lookup"}:
            # A hypothesis winner is supporting evidence, not a replacement
            # for the user's original analysis objective. Preserve a governed
            # diagnosis contract so the matched Skill runs after promotion.
            promoted_understanding.update(
                {
                    "originalQuestion": state.get("question", ""),
                    "analysisIntent": "diagnosis",
                    "requiresExplanation": True,
                    "reusableAnalysis": True,
                    "hypothesisEvidenceOnly": True,
                }
            )
        curated_plan = winner_plan.model_copy(
            deep=True,
            update={
                "intents": [intent.model_copy(deep=True) for intent in winner_plan.intents if intent.plan_task_id in successful_ids],
                "dependencies": [
                    dep.model_copy(deep=True)
                    for dep in winner_plan.dependencies
                    if dep.anchor_task_id in successful_ids and dep.dependent_task_id in successful_ids
                ],
                "question_understanding": promoted_understanding,
            },
        )
        curated_validation = self.graph_validator.validate(
            state["question"],
            curated_plan,
            state["planning_asset_pack"],
            state.get("memory_constraints", []),
        )
        curated_validation = validation_with_question_coverage(
            state["question"],
            curated_plan,
            state["planning_asset_pack"],
            curated_validation,
        )
        if not curated_validation.valid:
            return 0
        curated.verified_evidence = self.evidence_verifier.verify(
            state["question"], curated_plan, curated, state.get("memory_constraints", []), recall_knowledge_ref_ids(state)
        )
        curated.evidence_gaps = curated.verified_evidence.gaps
        if not curated.verified_evidence.passed:
            return 0
        state["plan"] = curated_plan
        state["agent_run_result"] = curated
        state["query_bundle"] = curated.merged_query_bundle
        state["query_bundles"] = curated.query_bundles
        state["node_tool_traces"] = curated.node_tool_traces
        state["freshness_reports"] = curated.freshness_reports
        state["query_graph_validation_result"] = curated_validation
        return len(successful)

    def _render_hypothesis_ledger_for_answer(self, ledger: HypothesisEvidenceLedger) -> str:
        payload = {
            "winnerId": ledger.winner_id,
            "survivorIds": ledger.survivor_ids,
            "hypotheses": [],
        }
        for entry in ledger.entries:
            item = {
                "hypothesisId": entry.hypothesis_id,
                "title": entry.title,
                "status": entry.status,
                "confidence": entry.confidence,
                "eliminationReason": entry.elimination_reason,
                "supportingEvidence": [],
                "evidenceGaps": entry.evidence_gaps[:4],
            }
            if entry.status == "survive":
                item["supportingEvidence"] = [
                    {
                        "evidenceId": evidence.evidence_id,
                        "claimKey": evidence.claim_key,
                        "metric": evidence.metric_name,
                        "table": evidence.table,
                        "timeRange": evidence.time_range,
                        "rowCount": evidence.row_count,
                        "confidence": evidence.confidence,
                        "preview": evidence.evidence_preview,
                    }
                    for evidence in entry.evidence
                    if evidence.status == "supporting"
                ][:6]
            payload["hypotheses"].append(item)
        return "假设—证据账本（最终回答只能引用 survive 假设的 supportingEvidence，pruned/failed 只能解释淘汰原因）：\n%s" % json.dumps(
            payload, ensure_ascii=False, default=str
        )

    def _hypothesis_semantic_score(self, state: AgentState, hypothesis_id: str) -> int:
        for candidate in (state.get("candidate_query_graphs") or {}).get("candidates", []):
            if str(candidate.get("hypothesisId") or "") == hypothesis_id:
                return int(candidate.get("score") or 0)
        return 30

    def _hypothesis_result_summary(self, execution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not execution:
            return {}
        run_result = execution.get("runResult") or AgentRunResult()
        bundle = run_result.merged_query_bundle
        return {
            "tables": list(bundle.tables or []),
            "rowCount": bundle.effective_row_count(),
            "rowsPreview": list(bundle.rows or [])[:5],
            "verifiedPassed": bool(run_result.verified_evidence.passed),
            "coveredEvidence": list(run_result.verified_evidence.covered_evidence or [])[:20],
            "gaps": [gap.model_dump(by_alias=True) for gap in run_result.verified_evidence.gaps[:8]],
        }

    def _public_hypothesis_execution(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        run_result = execution.get("runResult") or AgentRunResult()
        plan = execution.get("plan") or QueryPlan()
        return {
            "hypothesisId": execution.get("hypothesisId", ""),
            "title": (execution.get("hypothesis") or {}).get("title", ""),
            "rank": execution.get("rank", 0),
            "decision": execution.get("decision", ""),
            "evidenceScore": execution.get("evidenceScore", 0),
            "semanticScore": execution.get("semanticScore", 0),
            "rowCount": execution.get("rowCount", 0),
            "verifiedPassed": execution.get("verifiedPassed", False),
            "gapCount": execution.get("gapCount", 0),
            "planningMode": execution.get("planningMode", ""),
            "round": execution.get("round", 1),
            "tables": list(run_result.merged_query_bundle.tables or []),
            "queryGraph": plan.model_dump(by_alias=True),
            "evidencePreview": list(run_result.merged_query_bundle.rows or [])[:5],
            "evidenceGaps": [gap.model_dump(by_alias=True) for gap in run_result.verified_evidence.gaps[:8]],
            "followupDecision": execution.get("followupDecision") or {},
            "followupCount": len(execution.get("followups") or []),
            "executionError": execution.get("executionError", ""),
        }

    def answer_rule(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "answer_rule", "RuleAnswerAgent", "ANSWER_RULE", input_summary="refs=%d" % len(state.get("rule_recall_refs") or []))
        increment_round(state)
        emit(state, "node.started", "ANSWER_RULE", {})
        state["plan"] = self.build_rule_recall_plan(state)
        state["query_graph_validation_result"] = GraphValidationResult(valid=True, repairable=False)
        state["query_graph_validated"] = True
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
        analysis_summary_task_kinds = {"document_analysis", "analysis_worker", "analysis_skill", "hypothesis_review"}
        summaries = [
            str(item.get("summary") or "").strip()
            for item in completed
            if str(item.get("taskKind") or item.get("task_kind") or "") in analysis_summary_task_kinds
            and str(item.get("summary") or "").strip()
        ]
        if summaries:
            delegated_summary = "\n".join("- %s" % text for text in summaries)
            prior = str(state.get("analysis_summary") or "").strip()
            state["analysis_summary"] = "\n".join(item for item in (prior, "Sub-Agent 结果：\n%s" % delegated_summary) if item)
            state["analysis_generation"] = int(state.get("execution_generation") or 0)
        observations = list(state.get("main_agent_observations") or [])
        observations.append(
            {
                "stage": "delegate_subagent",
                "summary": "%d/%d Sub-Agent tasks completed" % (len(completed), len(results)),
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
            {"completed": True, "taskCount": len(results), "successCount": len(completed), "parallel": plan.parallel},
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
        hypotheses = list((state.get("hypothesis_exploration") or {}).get("hypotheses") or [])
        if self.policy.hypothesis_exploration_needed(state) and hypotheses and getattr(state.get("agent_run_result"), "task_results", None):
            kinds.append("hypothesis_review")
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
        lowered = question.lower()
        task_kind = ""
        python_requested = any(token in lowered for token in ("python", "批量分析", "批处理", "模拟计算", "运行脚本"))
        if "python_batch" in allowed_kinds and python_requested:
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
            task_id = "delegate_%s_%s" % (re.sub(r"[^a-zA-Z0-9_-]+", "_", task.task_kind), uuid.uuid4().hex[:12])
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
                        payload = handler(request, lambda: bool(state.get("run_canceled")))
                        contract = normalize_subagent_result(task.task_kind, "completed", payload)
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
                    "accessRole": state.get("access_role", "merchant_analyst"),
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
            self.request_human_clarification(state, self.build_scope_clarification_prompt(state), "BUSINESS_SCOPE", "business_scope", business_scope_options())
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
            if skill_confirmation_declined(state.get("original_question") or state.get("question") or ""):
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
        self.request_human_clarification(
            state,
            "为了让结论更可靠，建议继续进行“%s”。系统会基于已校验的经营数据做专项拆解，不会修改任何业务数据。是否开始？"
            % merchant_analysis_action_label(skill_name),
            "DEEP_ANALYSIS",
            "skill_confirm",
            ["开始深度分析", "先看当前结果"],
        )
        add_step(state, "专项分析确认：命中 %s，等待商家确认是否继续深挖" % merchant_analysis_action_label(skill_name))
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
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        payload = {
            "version": 2,
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
            or int(payload.get("version") or 0) != 2
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
        state["query_graph_validation_result"] = GraphValidationResult.model_validate(payload.get("queryGraphValidation") or {})
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
        state["analysis_skill_bypassed"] = skill_confirmation_declined(state.get("original_question") or "")
        state["topic_routed"] = True
        state["fast_understood"] = True
        state["data_discovered"] = True
        state["planning_assets_compacted"] = True
        state["query_graph_reflected"] = True
        state["query_graph_validated"] = bool(state["query_graph_validation_result"].valid)
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
        state["suggestions"] = state.get("human_clarification_options") or business_scope_options()
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
            add_step(state, "Memory Middleware：已安排回答后异步抽象长期记忆")
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
            state["response_context"].dimension_keys = [] if (getattr(fast_response, "debug_trace", {}) or {}).get("definitionOnly") else ["pt"]
            state["response_context"].data_catalog = ",".join(getattr(fast_response, "doris_tables", []) or [])
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
                "mode": "llm_curator_review" if curator_trace.get("authoritative") else "merchant_memory_review",
                "status": "pending_review",
                "description": "从本轮对话中提取候选业务知识，商家确认或平台审核后才会生效",
            }
            if curator_trace.get("authoritative"):
                add_step(
                    state,
                    "Knowledge Curator：模型已从用户原话提取 %d 条待确认知识"
                    % int(curator_trace.get("candidateCount") or len(governance_suggestions)),
                )
        add_step(state, "Memory Middleware：回答返回前已可靠提交结构化长期记忆 events=%d" % len(memory_payload.get("events") or []))

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
                            "mode": "llm_curator_review" if curator_trace.get("authoritative") else "merchant_memory_review",
                            "status": "pending_review",
                            "description": "从本轮对话中提取候选业务知识，商家确认或平台审核后才会生效",
                        }
                        if curator_trace.get("authoritative"):
                            add_step(
                                state,
                                "Knowledge Curator：模型已从用户原话提取 %d 条待确认知识"
                                % int(curator_trace.get("candidateCount") or len(governance_suggestions)),
                            )
                    add_step(state, "Memory Middleware：回答后已异步提交结构化长期记忆 events=%d" % len(memory_payload.get("events") or []))
                except Exception as exc:
                    state["memory_ingestion_trace"] = {
                        "status": "failed",
                        "written": False,
                        "error": str(exc)[:500],
                        "commitMode": "post_answer_async",
                    }
                    add_step(state, "Memory Middleware：回答后异步长期记忆提交失败 %s" % str(exc)[:180])
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
            self.merchant_profile_store.upsert_profile(
                state["merchant"].merchant_id,
                {
                    "defaultTimeWindow": state["merchant_profile_summary"].get("defaultTimeWindow") or 7,
                    "preferredMetrics": state["merchant_profile_summary"].get("preferredMetrics") or [],
                    "businessFocus": state["merchant_profile_summary"].get("businessFocus") or [],
                    "recentRisks": state["merchant_profile_summary"].get("recentRisks") or [],
                    "confirmedRules": state["merchant_profile_summary"].get("confirmedRules") or [],
                    "confirmedRuleTexts": state["merchant_profile_summary"].get("confirmedRuleTexts") or [],
                },
                reviewer="runtime_profile_store",
                review_status="reviewed",
                cancel_check=lambda: self.run_cancellation_requested(state),
            )
            if self.run_cancellation_requested(state):
                return
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
                "leadDecisions": [
                    item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
                    for item in state.get("lead_decisions", [])
                ],
                "capabilityDecisions": state.get("capability_decisions", {}),
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
        governance = self.memory_governance_debug_payload(state)
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
                        "sourceRefs": (state.get("knowledge_bundle") or KnowledgeBundle()).source_refs,
                        "rounds": state.get("recall_rounds", []),
                        "requestLineage": state.get("knowledge_request_lineage", {}),
                    },
                },
                "plannerReflection": state.get("planner_reflection", PlannerReflectionResult()).model_dump(by_alias=True),
                "plannerRepairReason": state.get("planner_repair_reason", ""),
                "plannerRepairRequests": [item.model_dump(by_alias=True) for item in state.get("planner_repair_requests", [])],
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
                "evidenceGaps": [item.model_dump(by_alias=True) for item in state["agent_run_result"].evidence_gaps],
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
            status = str(item.get("status") or "candidate")
            if status in {"merchant_active", "platform_suggested", "dismissed", "rejected", "published", "indexed"}:
                continue
            if status not in {"candidate", "review_required", "pending", "reviewed"} and str(item.get("suggestionId") or "") != target_id:
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            proposed_scope = str(item.get("scopeType") or payload.get("proposedScope") or "merchant").lower()
            if str(payload.get("memoryType") or "") == "metric_dispute":
                proposed_scope = "platform"
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

    def memory_governance_debug_payload(self, state: AgentState) -> Dict[str, Any]:
        trace = state.get("memory_ingestion_trace") or {}
        suggestion_id = str(trace.get("knowledgeSuggestionId") or "")
        if not suggestion_id:
            return {}
        return {
            "mode": "merchant_memory_review",
            "status": "pending_review" if trace.get("knowledgeSuggestionWritten") else "existing_candidate",
            "suggestionId": suggestion_id,
            "description": "候选规则需要审核后发布到商家知识库或语义资产",
        }

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
        report_statuses = {str(item.get("status") or "").upper() for item in report_payloads}
        gap_codes = {
            str(gap.gap_code or gap.code or "").upper()
            for gap in (state.get("agent_run_result") or AgentRunResult()).evidence_gaps
        }
        degraded = any(code in gap_codes for code in {"ZERO_ROWS", "PARTIAL_EVIDENCE", "RESOURCE_DEGRADED_QUERY"})
        fallback_reports = [item for item in report_payloads if item.get("fallbackTable")]
        realtime_fallback_used = "STALE_USE_REALTIME_FALLBACK" in report_statuses or bool(fallback_reports)
        offline_stale = realtime_fallback_used or any("STALE" in status for status in report_statuses)
        unchecked = bool(sections) and not reports
        if realtime_fallback_used:
            status = "realtime_fallback_used"
        elif offline_stale:
            status = "offline_stale"
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
        latest_data_at = ""
        for item in report_payloads:
            latest_data_at = max(latest_data_at, str(item.get("maxPt") or item.get("max_pt") or ""))
        return {
            "status": status,
            "checked": bool(reports),
            "tables": tables[:12],
            "reports": report_payloads,
            "latestDataAt": latest_data_at,
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
                "filter": "merchant_id/seller_id scoped by request merchant",
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
        validation = state.get("query_graph_validation_result") or GraphValidationResult()
        run_result = state.get("agent_run_result") or AgentRunResult()
        gaps = (state.get("agent_run_result") or AgentRunResult()).evidence_gaps
        strategy_switches = self.controlled_react_explorer.strategy_switch_trace(state, validation, run_result)
        state["strategy_switch_trace"] = strategy_switches
        return {
            "mode": "controlled_react_querygraph",
            "tradeoff": "limits free-form exploration in exchange for verifiable BI answers",
            "exploration": state.get("hypothesis_exploration") or {},
            "parallelHypothesisResults": state.get("hypothesis_results") or [],
            "candidateQueryGraphs": state.get("candidate_query_graphs") or {},
            "hypothesisEvidenceLedger": (state.get("hypothesis_evidence_ledger") or HypothesisEvidenceLedger()).model_dump(by_alias=True),
            "executionTierPolicy": state.get("execution_tier_policy") or {},
            "latencyOptimization": self.latency_optimizer.response_payload(state.get("latency_optimization") or {}),
            "steps": {
                "leadDecisions": len(lead_decisions),
                "actions": len(action_history),
                "plannerRepairs": len(repair_requests),
                "toolCalls": len(state.get("tool_call_ledger") or []),
                "evidenceGaps": len(gaps),
            },
            "strategySwitches": strategy_switches,
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
                "description": "confirmed business preferences can be written as memory suggestions, then reviewed before semantic publish",
            },
        }

    def human_loop_card_title(self, confirmation_type: str) -> str:
        mapping = {
            "time_window": "确认分析时间范围",
            "metric_focus": "确认指标口径",
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
        built_in = [
            {"name": "gmv_drop_diagnosis", "displayName": "GMV 下跌归因", "version": "v1", "reuseScope": "merchant_sop"},
            {"name": "refund_rate_diagnosis", "displayName": "退款率升高归因", "version": "v1", "reuseScope": "merchant_sop"},
            {"name": "merchant_daily_briefing", "displayName": "经营简报", "version": "v1", "reuseScope": "merchant_sop"},
        ]
        market_names = {item["name"] for item in market_items if item.get("name")}
        for item in built_in:
            if item["name"] not in market_names:
                market_items.append(item)
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
            rows = list(bundle.rows or [])
            if bundle.failed or len(rows) < min_rows:
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
                "candidateMemoryCount": len(injection.get("candidateMemories") or []),
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

    def request_human_clarification(self, state: AgentState, question: str, stage: str, type_: str, options: List[str]) -> None:
        question_text = str(question or "").strip()
        metric_gap = re.search(r"(?:当前)?候选(?:指标)?中没有明确的[“\"]?(.+?)[”\"]?指标", question_text)
        if metric_gap:
            metric_label = metric_gap.group(1).strip(" “”\"")
            question_text = "我还不能确定你想看的%s口径，请确认一下。" % (metric_label or "指标")
        else:
            question_text = (
                question_text.replace("当前候选中", "当前可用口径里")
                .replace("当前候选指标中", "当前可用口径里")
                .replace("候选指标中", "当前可用口径里")
                .replace("候选中", "当前可用口径里")
                .replace("QueryGraph", "查询计划")
            )
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
        return "你想看哪个范围？我可以按最近7天、最近30天或昨天来查，也可以直接看交易、退款、客服或商品。"

    def build_topic_clarification_prompt(self, state: AgentState) -> str:
        return "这个问题可能涉及多个业务域。你想优先看交易、退款售后、客服工单、商品还是供应链？"

    def build_priority_goal_clarification_prompt(self, state: AgentState) -> str:
        return "你希望“优先处理”按什么目标排序？我可以按综合经营风险默认评估，也可以更偏向退款/赔付损失、GMV 下单或客服压力。"

    def apply_open_diagnostic_policy(self, state: AgentState, decision: TopicRoutingDecision) -> List[QuestionCategory]:
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            return []
        text = state.get("question", "")
        context = state.get("request_context")
        if context and context.pending_clarification_type == "priority_goal":
            goal = str((state.get("clarification_resolution") or {}).get("priorityGoal") or getattr(context, "priority_goal", "") or state.get("original_question") or text or "综合经营风险").strip()
            return self.mark_open_diagnostic(
                state,
                intent="PRIORITY_RECOMMENDATION",
                goal=goal,
            )
        if is_store_health_overview_question(text) or state["routing_decision"].reason == "店铺整体经营问题":
            return self.mark_open_diagnostic(state, intent="STORE_HEALTH_DIAGNOSIS", goal="综合经营健康度")
        if decision.recall_topics():
            return []
        if is_priority_recommendation_question(text):
            self.mark_open_diagnostic(state, intent="PRIORITY_RECOMMENDATION", goal="")
            self.request_human_clarification(
                state,
                self.build_priority_goal_clarification_prompt(state),
                "OPEN_DIAGNOSTIC",
                "priority_goal",
                priority_goal_options(),
            )
            return []
        return []

    def mark_open_diagnostic(self, state: AgentState, intent: str, goal: str) -> List[QuestionCategory]:
        seed_topics = diagnostic_seed_topics(intent)
        state["open_diagnostic_scope"] = "OPEN_DIAGNOSTIC"
        state["open_diagnostic_intent"] = intent
        state["open_diagnostic_goal"] = goal
        state["open_diagnostic_seed_topics"] = seed_topics
        state["knowledge_expanded_topics"] = self._merge_topic_categories(state.get("knowledge_expanded_topics") or [], seed_topics)
        return seed_topics

    def _topic_names_for_categories(self, categories: List[Any]) -> List[str]:
        topic_asset_service = self.recall_service.topic_assets
        return topic_asset_service.topic_names_for_categories(categories)

    def _effective_topic_categories(self, state: AgentState) -> List[QuestionCategory]:
        base = state["topic_routing_decision"].recall_topics()
        expanded = state.get("knowledge_expanded_topics") or []
        return self._merge_topic_categories(base, expanded)

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
            category = TOPIC_TO_CATEGORY.get(topic_name)
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
                    category = TOPIC_TO_CATEGORY.get(topic_name)
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


def business_scope_options() -> List[str]:
    return ["最近7天整体经营", "最近30天退款售后", "昨天客服工单"]


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


def merchant_analysis_action_label(skill_name: str) -> str:
    return {
        "bi_trend_attribution": "指标波动原因深挖",
        "gmv_drop_diagnosis": "GMV下降原因诊断",
        "merchant_daily_briefing": "店铺经营体检",
        "new_product_risk": "新品经营风险排查",
        "ratio_analysis": "占比口径核验",
        "refund_rate_diagnosis": "退款压力专项诊断",
        "risk_analysis": "经营风险优先级分析",
        "rule_compliance": "平台规则影响核对",
    }.get(str(skill_name or ""), "经营专项分析")


def skill_confirmation_declined(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return any(term in normalized for term in ["先看当前结果", "暂不", "不用", "取消", "不开始", "先不", "no"])


def merchant_access_role(role: str) -> str:
    return {
        "platform_operator": "merchant_admin",
        "merchant_owner": "merchant_admin",
        "merchant_operator": "merchant_analyst",
        "merchant_finance": "merchant_finance",
        "merchant_customer_service": "merchant_service",
        "merchant_goods": "merchant_goods",
        "merchant_fulfillment": "merchant_fulfillment",
    }.get(str(role or ""), "merchant_analyst")


def priority_goal_options() -> List[str]:
    return ["综合经营风险", "降低退款/赔付损失", "稳住 GMV 和下单", "降低客服压力"]


def diagnostic_seed_topics(intent: str) -> List[QuestionCategory]:
    base = [
        QuestionCategory.TRADE,
        QuestionCategory.REFUND,
        QuestionCategory.CS_TICKET,
        QuestionCategory.COMPENSATION,
        QuestionCategory.GOODS,
    ]
    if intent == "PRIORITY_RECOMMENDATION":
        return base + [QuestionCategory.COUPON, QuestionCategory.SCM]
    return base


def dedupe_texts(values: List[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.append(text)
    return seen


def is_priority_recommendation_question(question: str) -> bool:
    text = question or ""
    decision_terms = ["优先处理", "先处理", "优先解决", "最值得优先", "建议我先", "只优先处理", "排优先级"]
    return any(term in text for term in decision_terms)


def is_store_health_overview_question(question: str) -> bool:
    text = question or ""
    subject_terms = ["店铺", "商家", "我店", "当前店铺"]
    overview_terms = ["整体经营", "经营情况", "经营概况", "店铺情况", "风险和机会", "总结风险", "经营健康", "店铺整体"]
    return any(term in text for term in subject_terms) and any(term in text for term in overview_terms)


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
    if not provider_error or (plan.intents and not failure_trace):
        return {}
    understanding = plan.question_understanding or {}
    allow_expensive_recovery = bool(
        understanding.get("allowDegradedHypothesisExploration")
        or understanding.get("allow_degraded_hypothesis_exploration")
    )
    if "timeout:" in provider_error:
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
    title = str(getattr(item, "title", "") or "")
    doc_id = str(getattr(item, "doc_id", "") or "")
    answer_modes = {part.strip() for part in re.split(r"[,|/]", answer_mode) if part.strip()}
    if answer_modes & {"RULE", "RULE_ANSWER", "PLATFORM_RULE", "RULE_REFERENCE"}:
        return True
    return source_type == "GOVERNED_RULE" and ("rule" in doc_id.lower() or "规则" in title)


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
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def knowledge_request_reason_code(request: KnowledgeRequest) -> str:
    reason = str(request.reason or "").strip()
    if not reason:
        return ""
    match = re.search(r":\s*([a-zA-Z0-9_]+)", reason)
    if match:
        return match.group(1)
    return normalize_knowledge_request_text(reason).split(" ")[0]


def knowledge_request_key(request: KnowledgeRequest) -> str:
    parts = [
        str(request.type or ""),
        normalize_knowledge_request_text(request.query),
        normalize_knowledge_request_text(request.needed_for_task_id),
        knowledge_request_reason_code(request),
    ]
    return "|".join(parts)


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
    safe.pop("candidateMemories", None)
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


def namespace_query_plan(plan: QueryPlan, namespace: str) -> QueryPlan:
    if not plan.intents:
        return plan
    mapping: Dict[str, str] = {}
    for index, intent in enumerate(plan.intents, start=1):
        old_id = str(intent.plan_task_id or "node_%d" % index)
        mapping[old_id] = "%s_%s" % (namespace, re.sub(r"[^a-zA-Z0-9_]+", "_", old_id))
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
    ]
    for field in list_fields:
        current = list(getattr(merged, field) or [])
        current.extend(item.model_copy(deep=True) if hasattr(item, "model_copy") else dict(item) if isinstance(item, dict) else item for item in getattr(right, field) or [])
        setattr(merged, field, current)
    merged.merged_query_bundle = merge_task_result_bundles(merged.task_results)
    return merged


def dedupe_workflow_knowledge_requests(items: List[KnowledgeRequest]) -> List[KnowledgeRequest]:
    deduped: List[KnowledgeRequest] = []
    seen: set[str] = set()
    for item in items:
        key = knowledge_request_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "query_result").strip())
    return text.strip("._") or "query_result"


def create_workflow(settings: Optional[Settings] = None) -> MerchantQaWorkflow:
    settings = settings or get_settings()
    doris_repository = DorisRepository(settings)
    answer_repository = AnswerRepository(settings)
    pending_store = PendingAnswerStore()
    planner_llm = LlmClient(settings)
    node_llm = LlmClient(settings)
    answer_llm = LlmClient(settings)
    topic_assets = TopicAssetService(settings)
    semantic_catalog = SemanticCatalogService(topic_assets)
    recall_service = HybridRecallService(settings, topic_assets)
    knowledge_retriever: KnowledgeRetrievalService
    if settings.es_enabled:
        knowledge_retriever = EsKnowledgeRetrievalService(settings, topic_assets)
    else:
        knowledge_retriever = HybridKnowledgeRetrievalService(recall_service)
    skill_loader = SkillLoader(settings)
    asset_builder = PlanningAssetPackBuilder(topic_assets, skill_loader, doris_repository)
    return MerchantQaWorkflow(
        settings=settings,
        merchant_service=MerchantService(settings, doris_repository),
        answer_repository=answer_repository,
        pending_store=pending_store,
        keyword_service=KeywordExtractService(topic_assets),
        routing_service=QuestionRoutingService(),
        topic_router=TopicRouterService(),
        recall_service=recall_service,
        knowledge_retriever=knowledge_retriever,
        asset_builder=asset_builder,
        planner=QueryGraphPlanner(planner_llm, semantic_catalog=semantic_catalog, settings=settings),
        graph_validator=QueryGraphValidator(),
        node_worker=NodeWorkerExecutor(node_llm, doris_repository, SqlValidationService(), settings, semantic_catalog=semantic_catalog),
        evidence_verifier=EvidenceVerifier(),
        answer_service=AnswerComposeService(answer_llm),
    )


try:
    graph = create_workflow().graph
except Exception:
    graph = None
