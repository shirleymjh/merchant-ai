from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from merchant_ai.models import (
    ActionResult,
    AgentAction,
    AgentActionTrace,
    AgentDecision,
    AgentRunResult,
    ContextPackage,
    ChatContext,
    CircuitBreakerState,
    ContextSnapshot,
    FreshnessCheckResult,
    GraphValidationResult,
    IntentSignals,
    KnowledgeBundle,
    KnowledgeRequest,
    MerchantInfo,
    MerchantRecentFocus,
    NodeToolCall,
    PlannerRepairRequest,
    PlanningAssetPack,
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    RecallBundle,
    RouteSlots,
    RoutingDecision,
    ThreadData,
    ToolCallExecutionResult,
    ToolFailureRecord,
    ToolRuntimePolicy,
    RunStep,
    TraceSpan,
    TopicRoutingDecision,
)


GraphEventListener = Callable[[str, str, Dict[str, Any]], None]
_EVENT_LISTENERS: Dict[str, GraphEventListener] = {}


def register_event_listener(run_id: str, listener: Optional[GraphEventListener]) -> None:
    if run_id and listener:
        _EVENT_LISTENERS[run_id] = listener


def unregister_event_listener(run_id: str) -> None:
    if run_id:
        _EVENT_LISTENERS.pop(run_id, None)


class AgentState(TypedDict, total=False):
    qa_id: str
    question: str
    original_question: str
    requested_merchant_id: str
    request_context: Optional[ChatContext]
    response_context: Optional[ChatContext]
    thread_id: str
    run_id: str
    checkpoint_thread_id: str
    thread_data: ThreadData
    event_listener: Optional[GraphEventListener]

    merchant: MerchantInfo
    recent_focus: MerchantRecentFocus
    routing_decision: RoutingDecision
    topic_routing_decision: TopicRoutingDecision
    route_slots: RouteSlots
    route_decision_trace: List[Dict[str, Any]]
    bounded_route_llm_trace: Dict[str, Any]
    extracted_keywords: Any
    plan: QueryPlan
    recall_bundle: RecallBundle
    knowledge_bundle: KnowledgeBundle
    recall_rounds: List[Any]
    knowledge_request_lineage: Dict[str, Any]
    intent_signals: IntentSignals
    planning_asset_pack: PlanningAssetPack
    query_graph_validation_result: GraphValidationResult
    pending_knowledge_requests: List[KnowledgeRequest]
    knowledge_request_attempts: Dict[str, int]
    knowledge_request_fingerprints: Dict[str, str]
    blocked_knowledge_request_keys: List[str]
    knowledge_request_gaps: List[Dict[str, Any]]
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
    context_packages: List[ContextPackage]
    run_steps: List[RunStep]
    trace_spans: List[TraceSpan]
    planner_repair_requests: List[PlannerRepairRequest]
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
    analysis_skill_trace: Dict[str, Any]
    answer_used_llm: bool
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
    rule_recall_ready: bool
    rule_recall_refs: List[str]
    rule_recall_context: str
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
    listener = _EVENT_LISTENERS.get(str(state.get("run_id") or "")) or state.get("event_listener")
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
