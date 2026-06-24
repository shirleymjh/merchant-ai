from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.policy import V2AgentPolicy
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
    AgentActionTrace,
    AgentRunResult,
    AnswerMode,
    ChatContext,
    ChatResponse,
    ClarificationRequest,
    ExtractedKeywords,
    GraphValidationGap,
    GraphValidationResult,
    IntentSignals,
    IntentType,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    KnowledgeRequest,
    KnowledgeRequestType,
    MerchantRecentFocus,
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
    TOPIC_TO_CATEGORY,
    RoutingDecision,
    ThreadData,
    TopicRoutingDecision,
)
from merchant_ai.services.answer import AnswerComposeService, analysis_summary_required, build_response_context, joined_categories
from merchant_ai.services.assets import HybridRecallService, PlanningAssetPackBuilder, SemanticCatalogService, SkillLoader, TopicAssetService, WikiMemoryService
from merchant_ai.services.context import ContextManager
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.observability import append_span, artifact_ref_from_path, now_ms, performance_summary, start_step, finish_step
from merchant_ai.services.planning import PlannerReflectionAgent, QueryGraphPlanner, QueryGraphValidator
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, MerchantService, PendingAnswerStore
from merchant_ai.services.retrieval import EsKnowledgeRetrievalService, HybridKnowledgeRetrievalService, KnowledgeRetrievalService
from merchant_ai.services.routing import KeywordExtractService, QuestionRoutingService, RouteSlotExtractor, TopicRouterService
from merchant_ai.services.tools import artifact_file_tool_schemas, lead_action_selection_tool, node_runtime_tool_schemas, semantic_file_tool_schemas


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
        wiki_memory: WikiMemoryService,
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
        self.wiki_memory = wiki_memory
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
        self.prompt_assembler = PromptAssembler()
        self.context_manager = ContextManager(settings)
        self.graph = self._build_graph()

    def run(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Optional[GraphEventListener] = None,
        thread_id: str = "",
        run_id: str = "",
    ) -> ChatResponse:
        effective_merchant_id = merchant_id or self.settings.merchant_id
        state = self._initial_state(question, effective_merchant_id, context, listener, thread_id, run_id)
        config = {"configurable": {"thread_id": state["thread_id"]}, "recursion_limit": 80}
        register_event_listener(state["run_id"], listener)
        try:
            final_state = self.graph.invoke(state, config=config)
            return self.to_response(final_state)
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
    ) -> ChatResponse:
        effective_merchant_id = merchant_id or self.settings.merchant_id
        state = self._initial_state(question, effective_merchant_id, context, listener, thread_id, run_id)
        config = {"configurable": {"thread_id": state["thread_id"]}, "recursion_limit": 80}
        register_event_listener(state["run_id"], listener)
        try:
            final_state = await asyncio.to_thread(self.graph.invoke, state, config)
            return self.to_response(final_state)
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
    ) -> AgentState:
        qa_id = "qa_" + uuid.uuid4().hex
        actual_thread_id = thread_id or "thread_" + uuid.uuid4().hex
        actual_run_id = run_id or "run_" + uuid.uuid4().hex
        workspace = self.settings.resolved_workspace_path / "threads" / actual_thread_id
        workspace.mkdir(parents=True, exist_ok=True)
        return AgentState(
            qa_id=qa_id,
            question=(question or "").strip(),
            original_question=question or "",
            requested_merchant_id=merchant_id,
            request_context=context,
            response_context=None,
            thread_id=actual_thread_id,
            run_id=actual_run_id,
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
            route_slots=RouteSlots(),
            route_decision_trace=[],
            bounded_route_llm_trace={},
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
            run_steps=[],
            trace_spans=[],
            planner_repair_requests=[],
            tool_failures=[],
            circuit_breakers=[],
            tool_runtime_policies=[],
            tool_call_results=[],
            agent_decision_reason="",
            planner_repair_reason="",
            planner_provider_error="",
            base_knowledge_context="",
            topic_asset_context="",
            recall_context="",
            merchant_profile_context="",
            memory_context="",
            session_context="",
            summary_context="",
            tool_context="",
            open_diagnostic_scope="",
            open_diagnostic_intent="",
            open_diagnostic_goal="",
            open_diagnostic_seed_topics=[],
            answer="",
            analysis_summary="",
            analysis_skill_trace={},
            answer_used_llm=False,
            suggestions=[],
            thinking_steps=[],
            history_rows=[],
            react_round=0,
            query_graph_retrieve_count=0,
            query_graph_plan_attempts=0,
            query_graph_repair_attempts=0,
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
            supervised=False,
            scope_clarified=False,
            context_loaded=False,
            topic_routed=False,
            data_discovered=False,
            sql_generated=False,
            chat_bi_completed=False,
            should_persist=False,
            persisted=False,
            human_clarification_required=False,
            human_clarification_question="",
            human_clarification_stage="",
            human_clarification_type="",
            human_clarification_options=[],
        )

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("inherit_context", self.inherit_context)
        builder.add_node("runtime_bootstrap", self.runtime_bootstrap)
        builder.add_node("policy", self.policy_node)
        builder.add_node("route_topic", self.route_topic)
        builder.add_node("retrieve_knowledge", self.retrieve_knowledge)
        builder.add_node("compact_assets", self.compact_assets)
        builder.add_node("plan_query_graph", self.plan_query_graph)
        builder.add_node("reflect_query_graph", self.reflect_query_graph)
        builder.add_node("validate_query_graph", self.validate_query_graph)
        builder.add_node("repair_query_graph", self.repair_query_graph)
        builder.add_node("execute_query_graph", self.execute_query_graph)
        builder.add_node("repair_sql", self.repair_sql)
        builder.add_node("verify_evidence_graph", self.verify_evidence_graph)
        builder.add_node("answer_rule", self.answer_rule)
        builder.add_node("answer_analysis", self.answer_analysis)
        builder.add_node("human_in_loop", self.human_in_loop)
        builder.add_node("cache_answer", self.cache_answer)

        builder.add_edge(START, "inherit_context")
        builder.add_edge("inherit_context", "runtime_bootstrap")
        builder.add_edge("runtime_bootstrap", "policy")
        builder.add_conditional_edges(
            "policy",
            lambda state: state.get("_next_action", "cache_answer"),
            {
                "route_topic": "route_topic",
                "retrieve_knowledge": "retrieve_knowledge",
                "compact_assets": "compact_assets",
                "plan_query_graph": "plan_query_graph",
                "reflect_query_graph": "reflect_query_graph",
                "validate_query_graph": "validate_query_graph",
                "repair_query_graph": "repair_query_graph",
                "execute_query_graph": "execute_query_graph",
                "repair_sql": "repair_sql",
                "verify_evidence_graph": "verify_evidence_graph",
                "answer_rule": "answer_rule",
                "answer_analysis": "answer_analysis",
                "human_in_loop": "human_in_loop",
                "cache_answer": "cache_answer",
            },
        )
        for node in [
            "route_topic",
            "retrieve_knowledge",
            "compact_assets",
            "plan_query_graph",
            "reflect_query_graph",
            "validate_query_graph",
            "repair_query_graph",
            "execute_query_graph",
            "repair_sql",
            "verify_evidence_graph",
        ]:
            builder.add_edge(node, "policy")
        builder.add_edge("answer_rule", "cache_answer")
        builder.add_edge("answer_analysis", "cache_answer")
        builder.add_edge("human_in_loop", END)
        builder.add_edge("cache_answer", END)
        return builder.compile(checkpointer=MemorySaver())

    def inherit_context(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "inherit_context", "LeadAgent", "INHERIT_CONTEXT", input_summary=state.get("question", ""))
        emit(state, "node.started", "INHERIT_CONTEXT", {"qaId": state["qa_id"]})
        context = state.get("request_context")
        if context and context.pending_clarification_stage and context.pending_question:
            state["question"] = ("%s %s" % (context.pending_question, state["question"])).strip()
            add_step(state, "Context Middleware：已合并上一轮澄清问题")
        self.record_span(state, "action", "inherit_context", started)
        self.finish_run_step(state, step, "success", output_summary=state["question"][:500])
        emit(state, "node.completed", "INHERIT_CONTEXT", {"question": state["question"]})
        return state

    def runtime_bootstrap(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "runtime_bootstrap", "LeadAgent", "LANGGRAPH_RUNTIME", input_summary=state.get("question", ""))
        emit(state, "node.started", "LANGGRAPH_RUNTIME", {})
        increment_round(state)
        state["merchant"] = self.merchant_service.current_merchant(state.get("requested_merchant_id", ""))
        state["merchant_profile_context"] = state["merchant"].profile_markdown()
        keywords = self.keyword_service.extract(state["question"])
        state["extracted_keywords"] = keywords
        state["routing_decision"] = self.routing_service.route(state["question"], keywords, state["recall_bundle"])
        state["supervised"] = True
        add_step(state, "LangGraph Runtime：完成会话接入，已预加载店铺静态画像")
        route = state["routing_decision"].route
        if route == QuestionRoute.GREETING:
            add_step(state, "LangGraph Runtime：闲聊/问候类问题走轻量回答")
        elif route == QuestionRoute.INVALID:
            self.request_human_clarification(state, self.build_scope_clarification_prompt(state), "BUSINESS_SCOPE", "business_scope", business_scope_options())
            add_step(state, "LangGraph Runtime：当前问题范围不清晰，准备进入 ask_human")
        else:
            add_step(state, "LangGraph Runtime：业务问题进入 Main Agent ReAct Runtime")
        self.refresh_context_snapshot(state, "runtime_bootstrap")
        self.record_span(state, "action", "runtime_bootstrap", started)
        self.finish_run_step(state, step, "success", output_summary="route=%s" % enum_value(route))
        emit(state, "node.completed", "LANGGRAPH_RUNTIME", {"route": enum_value(route)})
        return state

    def policy_node(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "policy", "LeadAgent", "MAIN_AGENT_POLICY", input_summary=state.get("agent_decision_reason", ""))
        decision = self.policy.decide(state)
        state["_next_action"] = decision.selected_node
        state["available_actions"] = self.policy.available_actions(state)
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
            },
        )
        self.record_span(state, "action", "policy", started)
        self.finish_run_step(state, step, "success", output_summary="%s->%s" % (decision.selected_action, decision.selected_node))
        return state

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
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            state["topic_routed"] = True
            self.record_span(state, "action", "route_topic", started)
            self.finish_run_step(state, step, "skipped", output_summary="non_business")
            return state
        state["base_knowledge_context"] = self.wiki_memory.load_base_wiki()
        context_topic = state["request_context"].topic if state.get("request_context") else ""
        route_slots = self.route_slot_extractor.extract(state["question"], state.get("extracted_keywords", ExtractedKeywords()))
        state["route_slots"] = route_slots
        state["route_decision_trace"] = [
            {
                "stage": "extract_route_slots",
                "operation": route_slots.operation,
                "riskLevel": route_slots.risk_level,
                "objectRefs": [item.model_dump(by_alias=True) for item in route_slots.object_refs],
                "timeWindow": route_slots.time_window.model_dump(by_alias=True),
                "analysisSignals": route_slots.analysis_signals,
                "routeConfidence": route_slots.route_confidence,
                "warnings": route_slots.route_warnings,
            }
        ]
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
        state["topic_routing_decision"] = decision
        state["topic_routed"] = True
        state["context_loaded"] = True
        state["scope_clarified"] = True
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
            state["topic_asset_context"] = self.wiki_memory.load_relevant_wiki(topic_names)
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
        decision.primary_topic = categories[0]
        decision.dimension_topics = categories[1:]
        decision.confidence = float((payload or {}).get("confidence") or decision.confidence or route_slots.route_confidence or 0.0)
        decision.reason = "受限 route LLM 确认；primaryTopic 仅兼容字段，不表示 anchor。%s" % str((payload or {}).get("reason") or "")
        trace.update({"status": "applied", "topics": topics, "payload": payload or {}})
        return decision, trace

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
        loaded_skills = self.load_skill_policies_for_retrieval(state)
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
        base_topics = self._effective_topic_categories(state)
        query_scopes: List[tuple[str, List[QuestionCategory], Optional[KnowledgeRequest]]] = [(state["question"], base_topics, None)]
        route_query = self.route_recall_query(state)
        if route_query and route_query != state["question"]:
            query_scopes.insert(0, (route_query, base_topics, None))
        if active_pending_requests:
            query_scopes = [
                (
                    request.query,
                    self._knowledge_request_topics(request, base_topics),
                    request.model_copy(update={"request_key": knowledge_request_key(request)}),
                )
                for request in active_pending_requests
                if request.query
            ] or query_scopes
        merged = state.get("recall_bundle") or RecallBundle()
        all_items = {item.doc_id: item for item in merged.items}
        existing_refs = set(all_items)
        round_traces = list(state.get("recall_rounds") or [])
        expanded_topics = list(state.get("knowledge_expanded_topics") or [])
        knowledge_bundles: List[KnowledgeBundle] = []
        for query, query_topics, request in query_scopes[:5]:
            expanded_topics = self._merge_topic_categories(expanded_topics, query_topics)
            keywords = self.keyword_service.extract(query)
            knowledge_bundle = self.knowledge_retriever.retrieve(
                KnowledgeRetrievalRequest(
                    query=query,
                    keywords=keywords.keywords,
                    history_rows=state.get("history_rows", []),
                    knowledge_context=knowledge_context(state),
                    merchant_id=state["merchant"].merchant_id,
                    topic_categories=query_topics,
                    knowledge_request=request,
                    route_slots=(state.get("route_slots") or RouteSlots()).model_dump(by_alias=True),
                    round=int(state.get("query_graph_retrieve_count") or 0),
                )
            )
            knowledge_bundles.append(knowledge_bundle)
            for trace in knowledge_bundle.recall_rounds:
                trace.new_refs = [ref for ref in trace.source_refs if ref not in existing_refs]
                existing_refs.update(trace.source_refs)
                round_traces.append(trace.model_dump(by_alias=True))
            bundle = knowledge_bundle.recall_bundle
            for item in bundle.items:
                current = all_items.get(item.doc_id)
                if current is not None:
                    item = merge_recall_item_queries(current, item)
                if current is None or item.fusion_score >= current.fusion_score or set((item.metadata or {}).get("recallQueries") or []) != set((current.metadata or {}).get("recallQueries") or []):
                    all_items[item.doc_id] = item
        lineage_items = list(all_items.values())
        items = sorted(lineage_items, key=lambda item: item.fusion_score, reverse=True)[:12]
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
        self.update_knowledge_request_lineage(state, active_pending_requests, lineage_items)
        state["recall_context"] = state["recall_bundle"].merged_context
        state["knowledge_expanded_topics"] = expanded_topics
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
        self.planner.artifact_store.write_json("recall", "recall_bundle.json", state["recall_bundle"].model_dump(by_alias=True), preview_chars=0)
        if had_pending_requests:
            state["plan"] = QueryPlan()
            state["query_graph_plan_attempts"] = 0
            state["planner_provider_error"] = ""
        add_step(state, "Main Agent Tool retrieve_knowledge：完成检索，命中 %d 条候选知识/资产片段，skillPolicies=%s" % (len(items), loaded_skills or []))
        self.record_span(
            state,
            "semantic_tool",
            "retrieve_knowledge",
            started,
            row_count=len(items),
            metadata={"skillPolicies": loaded_skills, "pendingRequests": had_pending_requests},
        )
        self.finish_run_step(state, step, "success", output_summary="recallItems=%d skills=%s" % (len(items), loaded_skills or []))
        emit(state, "node.completed", "RETRIEVE_KNOWLEDGE", {"recallItems": len(items), "skillPolicies": loaded_skills})
        return state

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
        if getattr(keywords, "time_keywords", []):
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
        return any(term in text for term in data_action_terms)

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
        knowledge_bundle = state.get("knowledge_bundle") or KnowledgeBundle()
        pack.metric_compaction["recallLineage"] = list(state.get("recall_rounds") or [])
        pack.metric_compaction["requestLineage"] = state.get("knowledge_request_lineage") or {}
        pack.metric_compaction["loadedSourceRefs"] = sorted(pack.source_refs.keys())
        pack.metric_compaction["recallBackend"] = knowledge_bundle.backend or "hybrid"
        pack.metric_compaction["semanticSourceHash"] = (
            knowledge_bundle.semantic_source_hash
            or pack.metric_compaction.get("cache", {}).get("semanticSourceHash", "")
        )
        pack.metric_compaction["indexVersion"] = knowledge_bundle.index_version
        state["planning_asset_pack"] = pack
        state["planning_assets_compacted"] = True
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

    def plan_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "plan_query_graph", "PlannerAgent", "PLAN_QUERY_GRAPH", input_summary=state.get("question", ""))
        increment_round(state)
        state["query_graph_plan_attempts"] = int(state.get("query_graph_plan_attempts") or 0) + 1
        emit(state, "node.started", "PLAN_QUERY_GRAPH", {})
        self.configure_artifact_roots(state)
        plan, requests, reason = self.planner.plan(
            state["question"],
            state.get("history_rows", []),
            knowledge_context(state),
            state["recall_bundle"],
            state["planning_asset_pack"],
            state["query_graph_validation_result"].gaps or state.get("last_query_graph_validation_gaps", []),
            state.get("thinking_steps", []),
            {
                "openDiagnostic": self.open_diagnostic_debug(state),
                "previousUnderstanding": (state.get("plan") or QueryPlan()).question_understanding,
            },
        )
        state["plan"] = plan
        self.planner.artifact_store.write_json("planner", "query_graph.json", plan.model_dump(by_alias=True), preview_chars=0)
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
            state["knowledge_request_gaps"] = append_knowledge_request_gaps(
                state.get("knowledge_request_gaps", []),
                blocked_requests,
                "METRIC_EVIDENCE_UNCHANGED",
            )
            trace = list(plan.compiler_trace or [])
            for request in blocked_requests:
                marker = "METRIC_EVIDENCE_UNCHANGED:%s" % request.query
                if marker not in trace:
                    trace.append(marker)
            plan.compiler_trace = trace
            plan.knowledge_requests = active_requests
        state["pending_knowledge_requests"] = active_requests
        state["planner_provider_error"] = planner_provider_error(self.planner.llm.last_error) if not plan.intents else ""
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
        self.record_span(
            state,
            "llm",
            "planner.question_understanding",
            started,
            status="success" if plan.intents else "failed",
            model=self.settings.openai_model,
            provider=self.settings.openai_base_url,
            estimated_prompt_chars=estimated_prompt_chars,
            estimated_completion_chars=len(json.dumps(plan.question_understanding or {}, ensure_ascii=False)),
            error_code=error_code,
            error_message=self.planner.llm.last_error if not plan.intents else "",
            metadata={"plannerPromptStats": planner_prompt_stats},
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
        emit(state, "node.completed", "PLAN_QUERY_GRAPH", {"nodes": len(plan.intents), "requests": len(active_requests)})
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
        if reflection.suggested_knowledge_requests:
            state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
                state,
                reflection.suggested_knowledge_requests,
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
        result = self.graph_validator.validate(state["question"], state["plan"], state["planning_asset_pack"])
        state["query_graph_validation_result"] = result
        state["query_graph_validated"] = True
        state["last_query_graph_validation_gaps"] = [] if result.valid else list(result.gaps)
        state["pending_knowledge_requests"] = filter_blocked_knowledge_requests(
            state,
            result.recommended_knowledge_requests,
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
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        state["planner_reflection"] = PlannerReflectionResult()
        state["planner_repair_reason"] = ""
        state["sql_generated"] = False
        state["sql_repair_reviewed"] = False
        state["evidence_graph_verified"] = False
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

    def execute_query_graph(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "execute_graph", "NodeAgent", "EXECUTE_QUERY_GRAPH", input_summary="nodes=%d" % len(state["plan"].intents))
        increment_round(state)
        self.configure_artifact_roots(state)
        emit(state, "node.started", "EXECUTE_QUERY_GRAPH", {})
        try:
            run_result = self.node_worker.execute_plan(
                state["merchant"].merchant_id,
                state["plan"],
                state["planning_asset_pack"],
                knowledge_context(state),
                state["question"],
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
        add_step(state, "Main Agent Tool execute_query_graph：派发 NodeWorker Agent 执行 QueryGraph nodes，tasks=%d" % len(run_result.tasks))
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
            metadata={"tasks": len(run_result.tasks), "failedTasks": failed_tasks},
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
        verified = self.evidence_verifier.verify(state["question"], state["plan"], state["agent_run_result"])
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
        state["answer"] = self.answer_service.compose(
            state["question"],
            state["merchant"],
            state["plan"],
            state["agent_run_result"],
            rule_context,
        )
        state["answer_used_llm"] = bool(self.answer_service.llm.configured and rule_context)
        state["suggestions"] = self.answer_service.contextual_suggestions(state["question"], state["plan"].intents)
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

    def answer_analysis(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "answer", "AnswerAgent", "ANSWER_ANALYSIS", input_summary="verified=%s" % state["agent_run_result"].verified_evidence.passed)
        increment_round(state)
        emit(state, "node.started", "ANSWER_ANALYSIS", {})
        route = state["routing_decision"].route
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
            state["answer"] = self.answer_service.compose(state["question"], state["merchant"], state["plan"], state["agent_run_result"], knowledge_context(state))
        elif route == QuestionRoute.INVALID:
            self.request_human_clarification(state, self.build_scope_clarification_prompt(state), "BUSINESS_SCOPE", "business_scope", business_scope_options())
            return self.human_in_loop(state)
        else:
            answer_needs_llm = analysis_summary_required(state["plan"])
            analysis_summary_attempted = False
            if answer_needs_llm and not state.get("analysis_summary"):
                analysis_summary_attempted = True
                state["analysis_summary"] = self.answer_service.summarize_analysis(
                    state["question"],
                    state["plan"],
                    state["agent_run_result"],
                    state["thread_data"].outputs_path,
                    state.get("rule_recall_context", ""),
                )
                state["analysis_skill_trace"] = dict(getattr(self.answer_service, "last_analysis_skill_trace", {}) or {})
            state["answer"] = self.answer_service.compose(
                state["question"],
                state["merchant"],
                state["plan"],
                state["agent_run_result"],
                knowledge_context(state),
                state.get("analysis_summary", ""),
                allow_llm=not (analysis_summary_attempted and not state.get("analysis_summary")),
                rule_context=state.get("rule_recall_context", ""),
            )
            skill_trace = state.get("analysis_skill_trace") or {}
            state["answer_used_llm"] = bool(answer_needs_llm and skill_trace.get("llmFallbackUsed"))
        state["suggestions"] = self.answer_service.contextual_suggestions(state["question"], state["plan"].intents)
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
                else len(knowledge_context(state)) + len(json.dumps(state["agent_run_result"].model_dump(by_alias=True), ensure_ascii=False, default=str))
            ),
            estimated_completion_chars=len(state.get("answer", "")),
            metadata={"usedLlm": bool(state.get("answer_used_llm"))},
        )
        self.finish_run_step(state, step, "success", output_summary="answerChars=%d" % len(state.get("answer", "")))
        emit(state, "node.completed", "ANSWER_ANALYSIS", {"answerReady": bool(state["answer"])})
        return state

    def human_in_loop(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "ask_human", "LeadAgent", "HUMAN_IN_LOOP", input_summary=state.get("human_clarification_question", ""))
        increment_round(state)
        prompt = state.get("human_clarification_question") or self.build_scope_clarification_prompt(state)
        state["answer"] = prompt
        state["should_persist"] = False
        state["persisted"] = False
        state["suggestions"] = state.get("human_clarification_options") or business_scope_options()
        add_step(state, "Human-in-the-loop / ask_human Tool：已用业务问题暂停自动推进，等待商家补充确认")
        self.record_span(state, "action", "ask_human", started, status="gap", error_code="HUMAN_CLARIFICATION_REQUIRED")
        self.finish_run_step(state, step, "gap", output_summary=prompt[:500], error_code="HUMAN_CLARIFICATION_REQUIRED")
        return state

    def cache_answer(self, state: AgentState) -> AgentState:
        started = now_ms()
        step = self.start_run_step(state, "cache_answer", "LeadAgent", "CACHE_ANSWER", input_summary="answerChars=%d" % len(state.get("answer", "")))
        emit(state, "node.started", "CACHE_ANSWER", {})
        if not state.get("answer"):
            state["answer"] = "当前没有足够证据生成回答，请补充更明确的业务范围或稍后重试。"
        sections = self.answer_service.build_sections(state["plan"], state["agent_run_result"])
        pending = PendingAnswer(
            id=state["qa_id"],
            question=state["question"],
            answer=state["answer"],
            merchant_id=state["merchant"].merchant_id,
            merchant_name=state["merchant"].merchant_name,
            category_name=joined_categories(state["plan"]),
            doris_tables=",".join(state["query_bundle"].tables),
            suggested_questions=json.dumps(state.get("suggestions", []), ensure_ascii=False),
            create_time=datetime.now(),
        )
        if state.get("should_persist"):
            self.pending_store.put(pending)
            state["persisted"] = False
            add_step(state, "当前已关闭问答记录立即写入，已缓存待反馈回答")
        else:
            add_step(state, "寒暄或无效意图不写入问答记录")
        state["response_context"] = build_response_context(
            state["question"],
            state["plan"],
            state["merchant"],
            sections,
            state.get("human_clarification_stage", ""),
            state.get("human_clarification_type", ""),
            state.get("human_clarification_options", []),
        )
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
        return state

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
                "artifactManifest": self.artifact_manifest(state),
                "actionHistory": [item.model_dump(by_alias=True) for item in state.get("action_history", [])],
                "leadDecisions": [item.model_dump(by_alias=True) for item in state.get("lead_decisions", [])],
                "promptManagement": {
                    "templates": self.prompt_assembler.catalog_summary(),
                    "leadPrompt": self.prompt_assembler.lead_prompt_summary(
                        self.policy.registry.public_action_ids(),
                        state.get("loaded_skills", []),
                        self.settings.max_concurrent_sub_agents,
                    ),
                },
                "toolCalling": self.tool_calling_debug(state),
                "contextSnapshots": state.get("context_snapshots", []),
                "contextManagement": self.context_management_debug(state),
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
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
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
            )
        clarification = None
        if state.get("human_clarification_required"):
            clarification = ClarificationRequest(
                question=state.get("human_clarification_question", ""),
                stage=state.get("human_clarification_stage", ""),
                type=state.get("human_clarification_type", ""),
                options=state.get("human_clarification_options", []),
                pending_question=state.get("question", ""),
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
                    "decisionReason": state.get("agent_decision_reason", ""),
                    "performance": performance_summary(state),
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
                    },
                    "toolCalling": self.tool_calling_debug(state),
                    "contextManagement": self.context_management_debug(state),
                    "toolRuntime": self.tool_runtime_debug(state),
                    "cache": self.cache_debug(),
                    "openDiagnostic": self.open_diagnostic_debug(state),
                    "routeSlots": state.get("route_slots", RouteSlots()).model_dump(by_alias=True),
                    "routeDecisionTrace": state.get("route_decision_trace", []),
                    "boundedRouteLlmTrace": state.get("bounded_route_llm_trace", {}),
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
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
                "nodePlanContracts": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_contracts],
                "nodePlanCritiques": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_critiques],
                "sqlDraftDecisions": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_draft_decisions],
                "freshnessReports": [item.model_dump(by_alias=True) for item in state.get("freshness_reports", [])],
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

    def planning_asset_debug(self, pack: PlanningAssetPack) -> Dict[str, Any]:
        return {
            "tables": [item.table for item in pack.tables[:20]],
            "metrics": [item.key for item in pack.metrics[:40]],
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
            "semanticFileContext": self.semantic_catalog.context_manifest(
                pack.source_refs,
                allowed_tables=pack.known_tables(),
                allowed_relationship_topics=relationship_topics_from_pack(pack),
            ),
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

    def open_diagnostic_debug(self, state: AgentState) -> Dict[str, Any]:
        return {
            "scope": state.get("open_diagnostic_scope", ""),
            "intent": state.get("open_diagnostic_intent", ""),
            "goal": state.get("open_diagnostic_goal", ""),
            "seedTopics": [enum_value(item) for item in state.get("open_diagnostic_seed_topics", [])],
        }

    def refresh_context_snapshot(self, state: AgentState, stage: str) -> None:
        snapshot = self.context_manager.refresh_state(state, stage)
        add_step(state, "Context Manager：刷新上下文快照 stage=%s protectedFacts=%d" % (stage, len(snapshot.protected_facts)))

    def sync_tool_runtime_state(self, state: AgentState) -> None:
        failure_trace = {}
        if hasattr(self.node_worker, "tool_failure_registry"):
            failure_trace = self.node_worker.tool_failure_registry.trace()
        state["tool_failures"] = failure_trace.get("failures", [])
        state["circuit_breakers"] = failure_trace.get("circuits", [])
        if hasattr(self.node_worker, "tool_runtime_policies"):
            state["tool_runtime_policies"] = self.node_worker.tool_runtime_policies.trace()

    def context_management_debug(self, state: AgentState) -> Dict[str, Any]:
        snapshots = state.get("context_snapshots", [])
        packages = state.get("context_packages", [])
        return {
            "strategy": "lead keeps global goal/progress; node agents receive scoped node context; snapshots preserve protected facts with source refs",
            "snapshotCount": len(snapshots),
            "packageCount": len(packages),
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

    def tool_runtime_debug(self, state: AgentState) -> Dict[str, Any]:
        policies = state.get("tool_runtime_policies", [])
        failures = state.get("tool_failures", [])
        circuits = state.get("circuit_breakers", [])
        if hasattr(self.node_worker, "tool_runtime_policies"):
            policies = self.node_worker.tool_runtime_policies.trace()
        if hasattr(self.node_worker, "tool_failure_registry"):
            failure_trace = self.node_worker.tool_failure_registry.trace()
            failures = failure_trace.get("failures", failures)
            circuits = failure_trace.get("circuits", circuits)
        return {
            "policies": policies,
            "failures": failures,
            "circuits": circuits,
            "parallelism": {
                "maxConcurrentNodeAgents": self.settings.max_concurrent_sub_agents,
                "resultPairing": "tool_call id",
                "failureIsolation": True,
            },
            "circuitBreaker": {
                "repeatedIdenticalFailureBlocks": True,
                "toolFailureThresholdBlocksTool": True,
            },
        }

    def cache_debug(self) -> Dict[str, Any]:
        return {
            "enabled": self.settings.cache_enabled,
            "memoryMaxEntries": self.settings.cache_memory_max_entries,
            "recall": self.recall_service.cache_trace() if hasattr(self.recall_service, "cache_trace") else {},
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
        state["human_clarification_required"] = True
        state["human_clarification_question"] = question
        state["human_clarification_stage"] = stage
        state["human_clarification_type"] = type_
        state["human_clarification_options"] = options

    def build_scope_clarification_prompt(self, state: AgentState) -> str:
        return "我还需要确认您要分析的业务范围。可以补充时间范围、业务域或具体指标，例如“最近7天GMV趋势”或“昨天退款明细”。"

    def build_topic_clarification_prompt(self, state: AgentState) -> str:
        return "这个问题可能涉及多个业务域，请确认优先看哪个范围：交易、退款、客服工单、赔付、商品或供应链。"

    def build_priority_goal_clarification_prompt(self, state: AgentState) -> str:
        return "你希望“优先处理”按什么目标排序？我可以按综合经营风险默认评估，也可以更偏向退款/赔付损失、GMV 下单或客服压力。"

    def apply_open_diagnostic_policy(self, state: AgentState, decision: TopicRoutingDecision) -> List[QuestionCategory]:
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            return []
        text = state.get("question", "")
        context = state.get("request_context")
        if context and context.pending_clarification_type == "priority_goal":
            return self.mark_open_diagnostic(
                state,
                intent="PRIORITY_RECOMMENDATION",
                goal=(state.get("original_question") or text or "综合经营风险").strip(),
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
    return ["最近7天总订单量趋势", "昨天退款明细", "商品审核拒绝原因"]


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
    repairable_codes = {
        "JOIN_KEY_NOT_PRODUCED",
        "DEPENDENCY_KEY_NOT_IN_SCHEMA",
        "DEPENDENCY_KEY_NOT_PRODUCED",
        "PLAN_CONTRACT_MISMATCH",
        "MISSING_METRIC_COLUMN",
        "MISSING_GROUP_BY_COLUMN",
        "MISSING_OUTPUT_KEY",
        "MISSING_UPSTREAM_ENTITY",
        "CONTRACT_REQUIRED_EVIDENCE_GAP",
    }
    gaps: List[GraphValidationGap] = []
    for gap in evidence_gaps:
        code = str(getattr(gap, "code", ""))
        if code not in repairable_codes:
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
    repairable_codes = {
        "JOIN_KEY_NOT_PRODUCED",
        "DEPENDENCY_KEY_NOT_IN_SCHEMA",
        "DEPENDENCY_KEY_NOT_PRODUCED",
        "JOIN_KEY_VALUES_EMPTY",
        "PLAN_CONTRACT_MISMATCH",
        "MISSING_METRIC_COLUMN",
        "MISSING_GROUP_BY_COLUMN",
        "MISSING_OUTPUT_KEY",
        "MISSING_UPSTREAM_ENTITY",
        "CONTRACT_REQUIRED_EVIDENCE_GAP",
    }
    gaps: List[GraphValidationGap] = []
    for task_result in task_results:
        message = str(task_result.query_bundle.error or task_result.summary or "")
        matched = next((code for code in repairable_codes if code in message), "")
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
        return {"required": False, "requiredDisclosures": [], "blockingGapCodes": [], "warningGapCodes": []}
    return {
        "required": verified.answer_guard_required,
        "requiredDisclosures": verified.required_disclosures,
        "blockingGapCodes": [gap.code for gap in verified.blocking_gaps],
        "warningGapCodes": [gap.code for gap in verified.warning_gaps],
    }


def rule_recall_item(item: Any) -> bool:
    answer_mode = str(getattr(item, "answer_mode", "") or "").upper()
    source_type = str(getattr(item, "source_type", "") or "").upper()
    title = str(getattr(item, "title", "") or "")
    doc_id = str(getattr(item, "doc_id", "") or "")
    answer_modes = {part.strip() for part in re.split(r"[,|/]", answer_mode) if part.strip()}
    if answer_modes & {"RULE", "RULE_ANSWER", "PLATFORM_RULE"}:
        return True
    return source_type == "BASE_WIKI" and ("rule" in doc_id.lower() or "规则" in title)


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
    merged_metadata = {**current_metadata, **incoming_metadata, "recallQueries": queries}
    if queries:
        merged_metadata["recallQuery"] = queries[-1]
    base = incoming if incoming.fusion_score >= current.fusion_score else current
    return base.model_copy(update={"fusion_score": max(current.fusion_score, incoming.fusion_score), "metadata": merged_metadata})


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


def filter_blocked_knowledge_requests(
    state: AgentState,
    requests: List[KnowledgeRequest],
) -> List[KnowledgeRequest]:
    deduped = dedupe_workflow_knowledge_requests(list(requests or []))
    blocked_keys = set(state.get("blocked_knowledge_request_keys") or [])
    blocked_requests = [request for request in deduped if knowledge_request_key(request) in blocked_keys]
    if blocked_requests:
        state["knowledge_request_gaps"] = append_knowledge_request_gaps(
            state.get("knowledge_request_gaps", []),
            blocked_requests,
            "METRIC_EVIDENCE_UNCHANGED",
        )
    return [request for request in deduped if knowledge_request_key(request) not in blocked_keys]


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


def create_workflow(settings: Optional[Settings] = None) -> MerchantQaWorkflow:
    settings = settings or get_settings()
    doris_repository = DorisRepository(settings)
    answer_repository = AnswerRepository(settings)
    pending_store = PendingAnswerStore()
    llm = LlmClient(settings)
    topic_assets = TopicAssetService(settings)
    semantic_catalog = SemanticCatalogService(topic_assets)
    wiki_memory = WikiMemoryService(settings)
    recall_service = HybridRecallService(settings, topic_assets, wiki_memory)
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
        keyword_service=KeywordExtractService(),
        routing_service=QuestionRoutingService(),
        topic_router=TopicRouterService(),
        wiki_memory=wiki_memory,
        recall_service=recall_service,
        knowledge_retriever=knowledge_retriever,
        asset_builder=asset_builder,
        planner=QueryGraphPlanner(llm, semantic_catalog=semantic_catalog, settings=settings),
        graph_validator=QueryGraphValidator(),
        node_worker=NodeWorkerExecutor(llm, doris_repository, SqlValidationService(), settings),
        evidence_verifier=EvidenceVerifier(),
        answer_service=AnswerComposeService(llm),
    )


try:
    graph = create_workflow().graph
except Exception:
    graph = None
