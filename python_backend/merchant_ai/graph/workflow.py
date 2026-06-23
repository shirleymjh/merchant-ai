from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.state import AgentState, GraphEventListener, add_step, emit, increment_round, knowledge_context
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
    TOPIC_TO_CATEGORY,
    RoutingDecision,
    ThreadData,
    TopicRoutingDecision,
)
from merchant_ai.services.answer import AnswerComposeService, build_response_context, joined_categories
from merchant_ai.services.assets import HybridRecallService, PlanningAssetPackBuilder, SemanticCatalogService, SkillLoader, TopicAssetService, WikiMemoryService
from merchant_ai.services.context import ContextManager
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.planning import PlannerReflectionAgent, QueryGraphPlanner, QueryGraphValidator
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, MerchantService, PendingAnswerStore
from merchant_ai.services.routing import KeywordExtractService, QuestionRoutingService, TopicRouterService
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
        self.wiki_memory = wiki_memory
        self.recall_service = recall_service
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
        final_state = self.graph.invoke(state, config=config)
        return self.to_response(final_state)

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
        final_state = await asyncio.to_thread(self.graph.invoke, state, config)
        return self.to_response(final_state)

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
            event_listener=listener,
            merchant=self.merchant_service.current_merchant(merchant_id),
            recent_focus=MerchantRecentFocus(merchant_id=merchant_id),
            routing_decision=RoutingDecision(),
            topic_routing_decision=TopicRoutingDecision(),
            plan=QueryPlan(),
            recall_bundle=RecallBundle(),
            planning_asset_pack=PlanningAssetPack(),
            query_graph_validation_result=GraphValidationResult(),
            pending_knowledge_requests=[],
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
        builder.add_edge("answer_analysis", "cache_answer")
        builder.add_edge("human_in_loop", END)
        builder.add_edge("cache_answer", END)
        return builder.compile(checkpointer=MemorySaver())

    def inherit_context(self, state: AgentState) -> AgentState:
        emit(state, "node.started", "INHERIT_CONTEXT", {"qaId": state["qa_id"]})
        context = state.get("request_context")
        if context and context.pending_clarification_stage and context.pending_question:
            state["question"] = ("%s %s" % (context.pending_question, state["question"])).strip()
            add_step(state, "Context Middleware：已合并上一轮澄清问题")
        emit(state, "node.completed", "INHERIT_CONTEXT", {"question": state["question"]})
        return state

    def runtime_bootstrap(self, state: AgentState) -> AgentState:
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
        emit(state, "node.completed", "LANGGRAPH_RUNTIME", {"route": enum_value(route)})
        return state

    def policy_node(self, state: AgentState) -> AgentState:
        decision = self.policy.decide(state)
        state["_next_action"] = decision.selected_node
        state["available_actions"] = self.policy.available_actions(state)
        state["agent_decision_reason"] = decision.reason
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
        return state

    def route_topic(self, state: AgentState) -> AgentState:
        increment_round(state)
        emit(state, "node.started", "ROUTE_TOPIC", {})
        if state["routing_decision"].route != QuestionRoute.BUSINESS:
            state["topic_routed"] = True
            return state
        state["base_knowledge_context"] = self.wiki_memory.load_base_wiki()
        context_topic = state["request_context"].topic if state.get("request_context") else ""
        decision = self.topic_router.route(state["question"], state.get("extracted_keywords", ExtractedKeywords()), context_topic)
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
        emit(state, "node.completed", "ROUTE_TOPIC", {"topic": decision.display_summary(), "openDiagnostic": self.open_diagnostic_debug(state)})
        return state

    def load_skill_policies_for_retrieval(self, state: AgentState) -> List[str]:
        skills = self.asset_builder.skill_loader.select(state["question"], self._effective_topic_categories(state))
        state["loaded_skills"] = [skill.domain for skill in skills]
        state["skills_loaded"] = True
        return state["loaded_skills"]

    def retrieve_knowledge(self, state: AgentState) -> AgentState:
        increment_round(state)
        self.configure_artifact_roots(state)
        state["query_graph_retrieve_count"] = int(state.get("query_graph_retrieve_count") or 0) + 1
        emit(state, "node.started", "RETRIEVE_KNOWLEDGE", {})
        loaded_skills = self.load_skill_policies_for_retrieval(state)
        had_pending_requests = bool(state.get("pending_knowledge_requests"))
        base_topics = self._effective_topic_categories(state)
        query_scopes = [(state["question"], base_topics)]
        if state.get("pending_knowledge_requests"):
            query_scopes = [
                (request.query, self._knowledge_request_topics(request, base_topics))
                for request in state["pending_knowledge_requests"]
                if request.query
            ] or query_scopes
        merged = state.get("recall_bundle") or RecallBundle()
        all_items = {item.doc_id: item for item in merged.items}
        expanded_topics = list(state.get("knowledge_expanded_topics") or [])
        for query, query_topics in query_scopes[:5]:
            expanded_topics = self._merge_topic_categories(expanded_topics, query_topics)
            bundle = self.recall_service.recall(
                query,
                self.keyword_service.extract(query),
                state.get("history_rows", []),
                knowledge_context(state),
                state["merchant"].merchant_id,
                query_topics,
            )
            for item in bundle.items:
                current = all_items.get(item.doc_id)
                if current is None or item.fusion_score > current.fusion_score:
                    all_items[item.doc_id] = item
        items = sorted(all_items.values(), key=lambda item: item.fusion_score, reverse=True)[:12]
        state["recall_bundle"] = RecallBundle(
            items=items,
            top_score=items[0].fusion_score if items else 0.0,
            merged_context="\n\n".join("召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items),
        )
        state["recall_context"] = state["recall_bundle"].merged_context
        state["knowledge_expanded_topics"] = expanded_topics
        state["pending_knowledge_requests"] = []
        state["data_discovered"] = True
        state["planning_assets_compacted"] = False
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        self.planner.artifact_store.write_json("recall", "recall_bundle.json", state["recall_bundle"].model_dump(by_alias=True), preview_chars=0)
        if had_pending_requests:
            state["plan"] = QueryPlan()
            state["query_graph_plan_attempts"] = 0
            state["planner_provider_error"] = ""
        add_step(state, "Main Agent Tool retrieve_knowledge：完成检索，命中 %d 条候选知识/资产片段，skillPolicies=%s" % (len(items), loaded_skills or []))
        emit(state, "node.completed", "RETRIEVE_KNOWLEDGE", {"recallItems": len(items), "skillPolicies": loaded_skills})
        return state

    def compact_assets(self, state: AgentState) -> AgentState:
        increment_round(state)
        self.configure_artifact_roots(state)
        emit(state, "node.started", "COMPACT_ASSETS", {})
        pack = self.asset_builder.compact(
            state["question"],
            state["recall_bundle"],
            self._effective_topic_categories(state),
            self.open_diagnostic_debug(state),
        )
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
        emit(state, "node.completed", "COMPACT_ASSETS", {"tables": pack.known_tables()[:12]})
        return state

    def plan_query_graph(self, state: AgentState) -> AgentState:
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
            state["query_graph_validation_result"].gaps,
            state.get("thinking_steps", []),
            {"openDiagnostic": self.open_diagnostic_debug(state)},
        )
        state["plan"] = plan
        self.planner.artifact_store.write_json("planner", "query_graph.json", plan.model_dump(by_alias=True), preview_chars=0)
        state["pending_knowledge_requests"] = [] if plan.intents else requests
        state["planner_provider_error"] = planner_provider_error(self.planner.llm.last_error) if not plan.intents else ""
        state["query_graph_validated"] = False
        state["query_graph_reflected"] = False
        state["planner_reflection"] = PlannerReflectionResult()
        if requests:
            add_step(state, "Main Agent Tool plan_query_graph：planner 请求补知识，requests=%d，reason=%s" % (len(requests), reason))
        else:
            add_step(state, "Main Agent Tool plan_query_graph：生成 QueryGraph，nodes=%d, edges=%d" % (len(plan.intents), len(plan.dependencies)))
        self.refresh_context_snapshot(state, "plan_query_graph")
        emit(state, "node.completed", "PLAN_QUERY_GRAPH", {"nodes": len(plan.intents), "requests": len(requests)})
        return state

    def configure_artifact_roots(self, state: AgentState) -> None:
        thread_data = state.get("thread_data")
        if not thread_data:
            return
        artifact_root = Path(thread_data.outputs_path) / "artifacts"
        self.planner.with_artifact_root(str(artifact_root))
        if hasattr(self.node_worker, "with_artifact_root"):
            self.node_worker.with_artifact_root(str(artifact_root))

    def reflect_query_graph(self, state: AgentState) -> AgentState:
        increment_round(state)
        emit(state, "node.started", "REFLECT_QUERY_GRAPH", {})
        reflection = self.planner_reflection_agent.reflect(state["question"], state["plan"], state["planning_asset_pack"])
        state["planner_reflection"] = reflection
        state["planner_repair_reason"] = reflection.repair_reason
        state["query_graph_reflected"] = True
        if reflection.suggested_knowledge_requests:
            state["pending_knowledge_requests"] = reflection.suggested_knowledge_requests
        if reflection.passed:
            add_step(state, "Planner Critic Tool reflect_plan：QueryGraph 自检通过，issues=%d" % len(reflection.issues))
        else:
            add_step(
                state,
                "Planner Critic Tool reflect_plan：发现 %d 个计划问题，suggested=%s"
                % (len(reflection.issues), reflection.suggested_actions[:3]),
            )
        emit(
            state,
            "node.completed",
            "REFLECT_QUERY_GRAPH",
            {"passed": reflection.passed, "issues": len(reflection.issues), "suggestedActions": reflection.suggested_actions},
        )
        return state

    def validate_query_graph(self, state: AgentState) -> AgentState:
        increment_round(state)
        emit(state, "node.started", "VALIDATE_QUERY_GRAPH", {})
        result = self.graph_validator.validate(state["question"], state["plan"], state["planning_asset_pack"])
        state["query_graph_validation_result"] = result
        state["query_graph_validated"] = True
        state["pending_knowledge_requests"] = result.recommended_knowledge_requests
        if result.valid:
            state["should_persist"] = any(intent.answer_mode != AnswerMode.RULE for intent in state["plan"].intents)
            add_step(state, "Main Agent Tool validate_query_graph：QueryGraph 通过校验，edges=%d" % len(state["plan"].dependencies))
        else:
            add_step(state, "Main Agent Tool validate_query_graph：发现 %d 个图缺口，repairable=%s" % (len(result.gaps), result.repairable))
        emit(state, "node.completed", "VALIDATE_QUERY_GRAPH", {"valid": result.valid, "gaps": len(result.gaps)})
        return state

    def repair_query_graph(self, state: AgentState) -> AgentState:
        increment_round(state)
        state["query_graph_repair_attempts"] = int(state.get("query_graph_repair_attempts") or 0) + 1
        emit(state, "node.started", "REPAIR_QUERY_GRAPH", {})
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
        emit(state, "node.completed", "REPAIR_QUERY_GRAPH", {"attempt": state["query_graph_repair_attempts"]})
        return state

    def execute_query_graph(self, state: AgentState) -> AgentState:
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
        emit(state, "node.completed", "EXECUTE_QUERY_GRAPH", {"tasks": len(run_result.tasks), "rows": run_result.merged_query_bundle.effective_row_count()})
        return state

    def repair_sql(self, state: AgentState) -> AgentState:
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
        return state

    def verify_evidence_graph(self, state: AgentState) -> AgentState:
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
        return state

    def answer_analysis(self, state: AgentState) -> AgentState:
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
            if not state.get("analysis_summary"):
                state["analysis_summary"] = self.answer_service.summarize_analysis(state["question"], state["plan"], state["agent_run_result"])
            state["answer"] = self.answer_service.compose(
                state["question"],
                state["merchant"],
                state["plan"],
                state["agent_run_result"],
                knowledge_context(state),
                state.get("analysis_summary", ""),
            )
        state["suggestions"] = self.answer_service.contextual_suggestions(state["question"], state["plan"].intents)
        state["chat_bi_completed"] = True
        add_step(state, "Result Loop：完成结果解读、建议生成与可视化数据组织")
        emit(state, "node.completed", "ANSWER_ANALYSIS", {"answerReady": bool(state["answer"])})
        return state

    def human_in_loop(self, state: AgentState) -> AgentState:
        increment_round(state)
        prompt = state.get("human_clarification_question") or self.build_scope_clarification_prompt(state)
        state["answer"] = prompt
        state["should_persist"] = False
        state["persisted"] = False
        state["suggestions"] = state.get("human_clarification_options") or business_scope_options()
        add_step(state, "Human-in-the-loop / ask_human Tool：已用业务问题暂停自动推进，等待商家补充确认")
        return state

    def cache_answer(self, state: AgentState) -> AgentState:
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
                "question": state.get("question", ""),
                "answer": state.get("answer", ""),
                "thinkingSteps": state.get("thinking_steps", []),
                "plan": state["plan"].model_dump(by_alias=True),
                "assetPack": self.planning_asset_debug(state["planning_asset_pack"]),
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
                "openDiagnostic": self.open_diagnostic_debug(state),
                "plannerReflection": state.get("planner_reflection", PlannerReflectionResult()).model_dump(by_alias=True),
                "plannerRepairReason": state.get("planner_repair_reason", ""),
                "questionUnderstanding": state["plan"].question_understanding,
                "compilerTrace": state["plan"].compiler_trace,
                "plannerToolCalls": state["plan"].planner_tool_calls,
                "plannerToolResults": state["plan"].planner_tool_results,
                "plannerLoadedRefs": state["plan"].planner_loaded_refs,
                "plannerContextFiles": state["plan"].planner_context_files,
                "metricResolution": metric_resolutions_for_debug(state["plan"]),
                "answerGuard": answer_guard_debug(state["agent_run_result"]),
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
                "nodePlanContracts": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_contracts],
                "nodePlanCritiques": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_critiques],
                "sqlDraftDecisions": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_draft_decisions],
                "freshnessReports": [item.model_dump(by_alias=True) for item in state.get("freshness_reports", [])],
                "validation": state["query_graph_validation_result"].model_dump(by_alias=True),
                "tasks": [item.model_dump(by_alias=True) for item in state["agent_run_result"].task_results],
                "evidenceGaps": [item.model_dump(by_alias=True) for item in state["agent_run_result"].evidence_gaps],
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
                "agentTrace": state["plan"].agent_trace,
                "harness": {
                    "mode": self.settings.agent_mode,
                    "actions": self.policy.registry.public_action_ids(),
                    "availableActions": [item.model_dump(by_alias=True) for item in state.get("available_actions", [])],
                    "actionHistory": [item.model_dump(by_alias=True) for item in state.get("action_history", [])],
                    "leadDecisions": [item.model_dump(by_alias=True) for item in state.get("lead_decisions", [])],
                    "decisionReason": state.get("agent_decision_reason", ""),
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
                    "openDiagnostic": self.open_diagnostic_debug(state),
                },
                "plannerReflection": state.get("planner_reflection", PlannerReflectionResult()).model_dump(by_alias=True),
                "plannerRepairReason": state.get("planner_repair_reason", ""),
                "questionUnderstanding": state["plan"].question_understanding,
                "compilerTrace": state["plan"].compiler_trace,
                "plannerToolCalls": state["plan"].planner_tool_calls,
                "plannerToolResults": state["plan"].planner_tool_results,
                "plannerLoadedRefs": state["plan"].planner_loaded_refs,
                "plannerContextFiles": state["plan"].planner_context_files,
                "metricResolution": metric_resolutions_for_debug(state["plan"]),
                "answerGuard": answer_guard_debug(state["agent_run_result"]),
                "nodeToolTraces": [item.model_dump(by_alias=True) for item in state.get("node_tool_traces", [])],
                "nodeTaskProfiles": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_task_profiles],
                "nodePlanContracts": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_contracts],
                "nodePlanCritiques": [item.model_dump(by_alias=True) for item in state["agent_run_result"].node_plan_critiques],
                "sqlDraftDecisions": [item.model_dump(by_alias=True) for item in state["agent_run_result"].sql_draft_decisions],
                "freshnessReports": [item.model_dump(by_alias=True) for item in state.get("freshness_reports", [])],
                "planningAssetPack": self.planning_asset_debug(state["planning_asset_pack"]),
                "queryGraphValidation": state["query_graph_validation_result"].model_dump(by_alias=True),
                "pendingKnowledgeRequests": [item.model_dump(by_alias=True) for item in state.get("pending_knowledge_requests", [])],
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
            "semanticFileContext": self.semantic_catalog.context_manifest(
                pack.source_refs,
                allowed_tables=pack.known_tables(),
                allowed_relationship_topics=relationship_topics_from_pack(pack),
            ),
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
        return {
            "strategy": "lead keeps global goal/progress; node agents receive scoped node context; snapshots preserve protected facts with source refs",
            "snapshotCount": len(snapshots),
            "recentSnapshots": snapshots[-4:],
            "summaryContextPreview": str(state.get("summary_context") or "")[:1200],
            "recoverySources": [
                "recent state",
                "context_snapshot.json",
                "trace_replay.json",
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
        gaps.append(
            GraphValidationGap(
                code=matched,
                task_id=task_result.task_id,
                evidence=task_result.query_bundle.sql[:240],
                reason=message[:300],
            )
        )
    return gaps


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
