from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from merchant_ai.models import (
    ActionResult,
    AgentAction,
    AgentActionTrace,
    AgentDecision,
    AgentRunResult,
    ChatContext,
    CircuitBreakerState,
    ContextSnapshot,
    FreshnessCheckResult,
    GraphValidationResult,
    KnowledgeRequest,
    MerchantInfo,
    MerchantRecentFocus,
    NodeToolCall,
    PlanningAssetPack,
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    RecallBundle,
    RoutingDecision,
    ThreadData,
    ToolCallExecutionResult,
    ToolFailureRecord,
    ToolRuntimePolicy,
    TopicRoutingDecision,
)


GraphEventListener = Callable[[str, str, Dict[str, Any]], None]


class AgentState(TypedDict, total=False):
    qa_id: str
    question: str
    original_question: str
    requested_merchant_id: str
    request_context: Optional[ChatContext]
    response_context: Optional[ChatContext]
    thread_id: str
    run_id: str
    thread_data: ThreadData
    event_listener: Optional[GraphEventListener]

    merchant: MerchantInfo
    recent_focus: MerchantRecentFocus
    routing_decision: RoutingDecision
    topic_routing_decision: TopicRoutingDecision
    extracted_keywords: Any
    plan: QueryPlan
    recall_bundle: RecallBundle
    planning_asset_pack: PlanningAssetPack
    query_graph_validation_result: GraphValidationResult
    pending_knowledge_requests: List[KnowledgeRequest]
    agent_run_result: AgentRunResult
    query_bundle: QueryBundle
    query_bundles: List[QueryBundle]
    available_actions: List[AgentAction]
    lead_decisions: List[AgentDecision]
    action_history: List[AgentActionTrace]
    last_action_result: ActionResult
    planner_reflection: PlannerReflectionResult
    node_tool_traces: List[NodeToolCall]
    freshness_reports: List[FreshnessCheckResult]
    context_snapshots: List[ContextSnapshot]
    tool_failures: List[ToolFailureRecord]
    circuit_breakers: List[CircuitBreakerState]
    tool_runtime_policies: List[ToolRuntimePolicy]
    tool_call_results: List[ToolCallExecutionResult]
    agent_decision_reason: str
    planner_repair_reason: str
    planner_provider_error: str

    base_knowledge_context: str
    topic_asset_context: str
    recall_context: str
    merchant_profile_context: str
    memory_context: str
    session_context: str
    summary_context: str
    tool_context: str
    open_diagnostic_scope: str
    open_diagnostic_intent: str
    open_diagnostic_goal: str
    open_diagnostic_seed_topics: List[QuestionCategory]

    answer: str
    analysis_summary: str
    suggestions: List[str]
    thinking_steps: List[str]
    history_rows: List[Dict[str, Any]]

    react_round: int
    query_graph_retrieve_count: int
    query_graph_plan_attempts: int
    query_graph_repair_attempts: int
    planning_assets_compacted: bool
    skills_loaded: bool
    loaded_skills: List[str]
    query_graph_validated: bool
    query_graph_reflected: bool
    sql_repair_reviewed: bool
    evidence_graph_verified: bool

    supervised: bool
    scope_clarified: bool
    context_loaded: bool
    topic_routed: bool
    data_discovered: bool
    sql_generated: bool
    chat_bi_completed: bool
    should_persist: bool
    persisted: bool

    human_clarification_required: bool
    human_clarification_question: str
    human_clarification_stage: str
    human_clarification_type: str
    human_clarification_options: List[str]
    _next_action: str


def emit(state: AgentState, event_type: str, node: str, payload: Dict[str, Any]) -> None:
    listener = state.get("event_listener")
    if listener:
        listener(event_type, node, payload)


def add_step(state: AgentState, text: str) -> None:
    state.setdefault("thinking_steps", [])
    if text:
        state["thinking_steps"].append(text)


def increment_round(state: AgentState) -> None:
    state["react_round"] = int(state.get("react_round") or 0) + 1


def knowledge_context(state: AgentState) -> str:
    parts: List[str] = []
    for title, key in [
        ("店铺画像", "merchant_profile_context"),
        ("长期记忆", "memory_context"),
        ("基础业务知识", "base_knowledge_context"),
        ("Topic资产", "topic_asset_context"),
        ("召回上下文", "recall_context"),
        ("会话上下文", "session_context"),
        ("历史摘要", "summary_context"),
        ("工具上下文", "tool_context"),
    ]:
        value = state.get(key) or ""
        if value:
            parts.append("## %s\n%s" % (title, value))
    return "\n\n".join(parts).strip()
