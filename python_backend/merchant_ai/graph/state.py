from __future__ import annotations

import uuid
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, TypedDict

from merchant_ai.models import (
    ActionResult,
    AgentAction,
    AgentActionTrace,
    AgentDecision,
    AgentRunResult,
    ContextPackage,
    ChatContext,
    ConversationMessage,
    CircuitBreakerState,
    ContextBudgetReport,
    ContextAssemblyReport,
    ContextCompressionEvent,
    ContextManifest,
    ContextSnapshot,
    ExecutionAttemptArtifact,
    FreshnessCheckResult,
    FastUnderstandingResult,
    GraphValidationGap,
    GraphValidationResult,
    IntentSignals,
    KnowledgeBundle,
    KnowledgeRequest,
    HypothesisEvidenceLedger,
    MerchantInfo,
    MerchantRecentFocus,
    MiddlewareEvent,
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
    ToolCallLedgerEntry,
    ToolCallRequest,
    ToolCallRecoveryEvent,
    ToolFailureRecord,
    ToolRuntimePolicy,
    RunStep,
    SkillDraft,
    SkillLifecycleRecord,
    SkillMatchState,
    TraceSpan,
    TopicRoutingDecision,
    WorkspaceManifest,
)


GraphEventListener = Callable[[str, str, Dict[str, Any]], None]
_EVENT_LISTENERS: Dict[str, GraphEventListener] = {}
_EVENT_LISTENERS_LOCK = RLock()


def register_event_listener(run_id: str, listener: Optional[GraphEventListener]) -> None:
    if run_id and listener:
        with _EVENT_LISTENERS_LOCK:
            _EVENT_LISTENERS[run_id] = listener


def unregister_event_listener(run_id: str) -> None:
    if run_id:
        with _EVENT_LISTENERS_LOCK:
            _EVENT_LISTENERS.pop(run_id, None)


class AgentState(TypedDict, total=False):
    qa_id: str
    question: str
    original_question: str
    requested_merchant_id: str
    request_context: Optional[ChatContext]
    user_identity: Dict[str, Any]
    access_role: str
    response_context: Optional[ChatContext]
    message_history: List[ConversationMessage]
    thread_id: str
    run_id: str
    checkpoint_thread_id: str
    thread_data: ThreadData
    event_listener: Optional[GraphEventListener]

    merchant: MerchantInfo
    recent_focus: MerchantRecentFocus
    routing_decision: RoutingDecision
    topic_routing_decision: TopicRoutingDecision
    topic_workspace: Dict[str, Any]
    analysis_scope: Dict[str, Any]
    knowledge_refresh: Dict[str, Any]
    route_slots: RouteSlots
    route_decision_trace: List[Dict[str, Any]]
    clarification_resolution: Dict[str, Any]
    clarification_root_question: str
    bounded_route_llm_trace: Dict[str, Any]
    bounded_lead_llm_trace: Dict[str, Any]
    fast_gate_decision_trace: Dict[str, Any]
    lead_decision_context: Dict[str, Any]
    recall_strategy: Dict[str, Any]
    worker_dispatch_context: Dict[str, Any]
    skill_dispatch_context: Dict[str, Any]
    main_agent_observations: List[Dict[str, Any]]
    extracted_keywords: Any
    plan: QueryPlan
    recall_bundle: RecallBundle
    knowledge_bundle: KnowledgeBundle
    recall_rounds: List[Any]
    knowledge_request_lineage: Dict[str, Any]
    intent_signals: IntentSignals
    fast_understanding: FastUnderstandingResult
    query_metric_attempted: bool
    query_metric_completed: bool
    query_metric_trace: Dict[str, Any]
    fast_metric_attempted: bool
    fast_metric_completed: bool
    fast_metric_response: Any
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
    execution_attempt_artifacts: List[ExecutionAttemptArtifact]
    available_actions: List[AgentAction]
    lead_decisions: List[AgentDecision]
    action_history: List[AgentActionTrace]
    action_outcomes: List[Dict[str, Any]]
    action_catalog_contract_blocks: List[Dict[str, Any]]
    contract_block_observation: Dict[str, Any]
    contract_block_observed: bool
    contract_block_generation: int
    _pending_action_contract: Dict[str, Any]
    last_action_result: ActionResult
    planner_reflection: PlannerReflectionResult
    node_tool_traces: List[NodeToolCall]
    freshness_reports: List[FreshnessCheckResult]
    context_snapshots: List[ContextSnapshot]
    context_packages: List[ContextPackage]
    context_manifests: List[ContextManifest]
    active_context_manifest: Dict[str, Any]
    active_context_package: Dict[str, Any]
    context_budget_reports: List[ContextBudgetReport]
    context_assembly_reports: List[ContextAssemblyReport]
    context_compression_events: List[ContextCompressionEvent]
    runtime_checkpoints: List[Dict[str, Any]]
    middleware_events: List[MiddlewareEvent]
    tool_call_ledger: List[ToolCallLedgerEntry]
    tool_call_recovery_events: List[ToolCallRecoveryEvent]
    tool_call_requests: List[ToolCallRequest]
    tool_loop_warning: str
    pending_tool_loop_warnings: List[str]
    tool_loop_history: Dict[str, List[Dict[str, Any]]]
    tool_loop_seen_call_ids: Dict[str, List[str]]
    forced_tool_loop_stop_message: str
    tool_output_budget_reports: List[Dict[str, Any]]
    token_usage_reports: List[Dict[str, Any]]
    safety_finish_reasons: List[Dict[str, Any]]
    run_budget_report: Dict[str, Any]
    run_budget_exhausted: bool
    run_started_at_ms: int
    workspace_manifest: WorkspaceManifest
    run_steps: List[RunStep]
    trace_spans: List[TraceSpan]
    planner_repair_requests: List[PlannerRepairRequest]
    tool_failures: List[ToolFailureRecord]
    circuit_breakers: List[CircuitBreakerState]
    tool_runtime_policies: List[ToolRuntimePolicy]
    tool_call_results: List[ToolCallExecutionResult]
    tool_runtime_events: List[Dict[str, Any]]
    answer_file_tool_results: Dict[str, Any]
    clarification_tool_message: Dict[str, Any]
    clarification_command: Dict[str, Any]
    agent_decision_reason: str
    planner_repair_reason: str
    planner_provider_error: str
    planner_degraded: Dict[str, Any]

    base_knowledge_context: str
    topic_asset_context: str
    always_apply_context: str
    always_apply_rules: List[Dict[str, Any]]
    recall_context: str
    merchant_profile_context: str
    memory_context: str
    runtime_context: str
    session_context: str
    summary_context: str
    tool_context: str
    thread_context: Dict[str, Any]
    runtime_injection: Dict[str, Any]
    memory_injection: Dict[str, Any]
    memory_injection_trace: Dict[str, Any]
    memory_ingestion_trace: Dict[str, Any]
    memory_constraints: List[Dict[str, Any]]
    memory_constraint_trace: Dict[str, Any]
    memory_recalled: bool
    merchant_profile_summary: Dict[str, Any]
    open_diagnostic_scope: str
    open_diagnostic_intent: str
    open_diagnostic_goal: str
    open_diagnostic_profile: Dict[str, Any]
    open_diagnostic_seed_topics: List[QuestionCategory]
    hypothesis_exploration: Dict[str, Any]
    hypothesis_results: List[Dict[str, Any]]
    hypothesis_exploration_completed: bool
    hypothesis_exploration_status: Dict[str, Any]
    hypothesis_exploration_rounds: int
    hypothesis_selected_ids: List[str]
    hypothesis_evidence_ledger: HypothesisEvidenceLedger
    candidate_query_graphs: Dict[str, Any]
    strategy_switch_trace: List[Dict[str, Any]]
    latency_optimization: Dict[str, Any]
    execution_tier_policy: Dict[str, Any]
    node_execution_mode: str

    answer: str
    analysis_summary: str
    analysis_worker_trace: Dict[str, Any]
    analysis_worker_completed: bool
    analysis_worker_status: Dict[str, Any]
    analysis_skill_trace: Dict[str, Any]
    subagent_delegation_plan: Dict[str, Any]
    subagent_delegation_results: List[Dict[str, Any]]
    subagent_delegation_attempted: bool
    subagent_delegation_completed: bool
    confirmation_evidence_reused: bool
    confirmation_token: str
    confirmation_source_run_id: str
    analysis_skill_bypassed: bool
    skill_worker_completed: bool
    analysis_skill_status: Dict[str, Any]
    skill_match: SkillMatchState
    skill_draft: SkillDraft
    skill_lifecycle_records: List[SkillLifecycleRecord]
    merchant_experience: Dict[str, Any]
    answer_used_llm: bool
    suggestions: List[str]
    thinking_steps: List[str]
    history_rows: List[Dict[str, Any]]

    react_round: int
    query_graph_retrieve_count: int
    query_graph_supplemental_retrieve_count: int
    query_graph_plan_attempts: int
    query_graph_repair_attempts: int
    query_graph_repair_attempted: bool
    execution_generation: int
    result_generation: int
    evidence_generation: int
    analysis_generation: int
    planning_assets_compacted: bool
    fast_understood: bool
    skills_loaded: bool
    loaded_skills: List[str]
    rule_recall_ready: bool
    rule_recall_refs: List[str]
    rule_recall_context: str
    query_graph_validated: bool
    query_graph_validation_attempted: bool
    query_graph_validation_passed: bool
    query_graph_validation_status: str
    validated_query_graph_fingerprint: str
    query_graph_reflected: bool
    sql_repair_reviewed: bool
    evidence_graph_verified: bool
    verification_status: str
    evidence_accepted: bool

    supervised: bool
    scope_clarified: bool
    context_loaded: bool
    topic_routed: bool
    data_discovered: bool
    sql_generated: bool
    chat_bi_completed: bool
    run_canceled: bool
    middleware_loop_blocked: bool
    runtime_guard_gaps: List[GraphValidationGap]
    terminal_status: Dict[str, Any]
    should_persist: bool
    persisted: bool
    post_answer_tail_pending: bool

    human_clarification_required: bool
    human_clarification_question: str
    human_clarification_stage: str
    human_clarification_type: str
    human_clarification_options: List[str]
    _next_action: str


def emit(state: AgentState, event_type: str, node: str, payload: Dict[str, Any]) -> None:
    run_id = str(state.get("run_id") or "")
    with _EVENT_LISTENERS_LOCK:
        listener = _EVENT_LISTENERS.get(run_id)
    listener = listener or state.get("event_listener")
    if listener:
        listener(event_type, node, event_payload(state, event_type, node, payload))


def event_payload(state: AgentState, event_type: str, node: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("eventEnvelopeVersion", "v1")
    data.setdefault("eventId", "evt_" + uuid.uuid4().hex)
    data.setdefault("eventType", event_type)
    data.setdefault("node", node)
    data.setdefault("runId", str(state.get("run_id") or ""))
    data.setdefault("threadId", str(state.get("thread_id") or ""))
    data.setdefault("timestamp", datetime.now().isoformat())
    correlation_id = (
        data.get("correlationId")
        or data.get("toolCallId")
        or data.get("stepId")
        or str(state.get("_active_step_id") or "")
        or str(state.get("run_id") or "")
    )
    data.setdefault("correlationId", correlation_id)
    return data


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
        ("运行时注入", "runtime_context"),
        ("基础业务知识", "base_knowledge_context"),
        ("Topic资产", "topic_asset_context"),
        ("强制业务规则（Always Apply）", "always_apply_context"),
        ("召回上下文", "recall_context"),
        ("会话上下文", "session_context"),
        ("历史摘要", "summary_context"),
        ("工具上下文", "tool_context"),
    ]:
        value = state.get(key) or ""
        if value:
            parts.append("## %s\n%s" % (title, value))
    return "\n\n".join(parts).strip()


STATE_LIST_MERGE_KEYS: Dict[str, str] = {
    "context_packages": "packageId",
    "context_manifests": "manifestId",
    "runtime_checkpoints": "path",
    "middleware_events": "eventId",
    "tool_call_ledger": "toolCallId",
    "tool_call_recovery_events": "eventId",
    "tool_call_requests": "id",
    "tool_call_results": "id",
    "tool_failures": "fingerprint",
    "circuit_breakers": "circuitKey",
    "run_steps": "stepId",
    "trace_spans": "spanId",
    "skill_lifecycle_records": "recordId",
    "freshness_reports": "reportId",
}

STATE_LIST_LIMITS: Dict[str, int] = {
    "context_packages": 12,
    "context_manifests": 24,
    "runtime_checkpoints": 8,
    "middleware_events": 200,
    "tool_call_ledger": 200,
    "tool_call_recovery_events": 50,
    "tool_call_requests": 100,
    "tool_call_results": 100,
    "tool_failures": 100,
    "circuit_breakers": 100,
    "run_steps": 200,
    "trace_spans": 400,
    "skill_lifecycle_records": 120,
    "freshness_reports": 50,
}


def merge_agent_state_update(existing: AgentState, update: Dict[str, Any]) -> AgentState:
    """Merge partial state updates with explicit reducers for shared runtime lists."""

    merged: AgentState = dict(existing)
    payload = dict(update or {})
    replacements = payload.pop("__replace__", {})
    deletions = payload.pop("__delete__", [])
    if isinstance(replacements, dict):
        for key, value in replacements.items():
            merged[str(key)] = value
    for key in deletions if isinstance(deletions, list) else []:
        merged.pop(str(key), None)
    for key, value in payload.items():
        if key in STATE_LIST_MERGE_KEYS:
            merged[key] = merge_state_list(
                existing.get(key) or [],
                value or [],
                id_key=STATE_LIST_MERGE_KEYS[key],
                limit=STATE_LIST_LIMITS.get(key, 200),
            )
        elif isinstance(value, dict) and isinstance(existing.get(key), dict):
            merged[key] = {**(existing.get(key) or {}), **value}
        else:
            merged[key] = value
    return merged


def mark_terminal_status(
    state: AgentState,
    status: str,
    code: str,
    source: str,
    message: str = "",
) -> Dict[str, Any]:
    """Set a run terminal state once; later middleware cannot silently downgrade it."""

    existing = state.get("terminal_status") or {}
    if existing.get("active"):
        return existing
    terminal = {
        "active": True,
        "status": str(status or "blocked"),
        "code": str(code or "TERMINAL_STOP"),
        "source": str(source or "runtime"),
        "message": str(message or "")[:500],
    }
    state["terminal_status"] = terminal
    return terminal


def merge_state_list(existing: Any, incoming: Any, id_key: str, limit: int = 200) -> List[Any]:
    items = list(existing or []) + list(incoming or [])
    deduped: Dict[str, Any] = {}
    anonymous: List[Any] = []
    for item in items:
        item_id = state_item_id(item, id_key)
        if item_id:
            deduped[item_id] = item
        else:
            anonymous.append(item)
    merged = [*anonymous, *deduped.values()]
    return merged[-limit:] if limit > 0 else merged


def state_item_id(item: Any, id_key: str) -> str:
    candidates = [
        id_key,
        snake_case_key(id_key),
        "id",
        "taskId",
        "task_id",
        "toolCallId",
        "tool_call_id",
    ]
    if isinstance(item, dict):
        for key in candidates:
            value = item.get(key)
            if value:
                return str(value)
        return ""
    for key in candidates:
        value = getattr(item, key, "")
        if value:
            return str(value)
    if hasattr(item, "model_dump"):
        try:
            return state_item_id(item.model_dump(by_alias=True), id_key)
        except Exception:
            return ""
    return ""


def snake_case_key(value: str) -> str:
    chars: List[str] = []
    for char in str(value or ""):
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)
