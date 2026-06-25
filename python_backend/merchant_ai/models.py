from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    pieces = value.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel, use_enum_values=True)


class QuestionRoute(str, Enum):
    GREETING = "GREETING"
    INVALID = "INVALID"
    BUSINESS = "BUSINESS"


class QuestionCategory(str, Enum):
    PLATFORM_RULE = "PLATFORM_RULE"
    TRADE = "TRADE"
    REFUND = "REFUND"
    CS_TICKET = "CS_TICKET"
    COMPENSATION = "COMPENSATION"
    COUPON = "COUPON"
    GOODS = "GOODS"
    MERCHANT_OTHER = "MERCHANT_OTHER"
    IDENTITY = "IDENTITY"
    SCM = "SCM"
    UNKNOWN = "UNKNOWN"


QUESTION_CATEGORY_DISPLAY: Dict[QuestionCategory, str] = {
    QuestionCategory.PLATFORM_RULE: "平台商家规则",
    QuestionCategory.TRADE: "电商交易",
    QuestionCategory.REFUND: "电商退货",
    QuestionCategory.CS_TICKET: "电商客服工单",
    QuestionCategory.COMPENSATION: "电商理赔/赔付",
    QuestionCategory.COUPON: "电商优惠券",
    QuestionCategory.GOODS: "商品管理",
    QuestionCategory.MERCHANT_OTHER: "商家其他信息",
    QuestionCategory.IDENTITY: "身份信息",
    QuestionCategory.SCM: "供应链",
    QuestionCategory.UNKNOWN: "未知",
}


TOPIC_TO_CATEGORY: Dict[str, QuestionCategory] = {
    "平台商家规则": QuestionCategory.PLATFORM_RULE,
    "电商交易": QuestionCategory.TRADE,
    "电商退货": QuestionCategory.REFUND,
    "客服工单": QuestionCategory.CS_TICKET,
    "电商客服工单": QuestionCategory.CS_TICKET,
    "客服理赔": QuestionCategory.COMPENSATION,
    "电商理赔/赔付": QuestionCategory.COMPENSATION,
    "电商优惠券": QuestionCategory.COUPON,
    "商品管理": QuestionCategory.GOODS,
    "商家其他信息": QuestionCategory.MERCHANT_OTHER,
    "身份信息": QuestionCategory.IDENTITY,
    "供应链": QuestionCategory.SCM,
    "经营画像": QuestionCategory.TRADE,
}


def category_display(category: QuestionCategory) -> str:
    if isinstance(category, QuestionCategory):
        return QUESTION_CATEGORY_DISPLAY.get(category, category.value)
    try:
        enum_category = QuestionCategory(str(category))
        return QUESTION_CATEGORY_DISPLAY.get(enum_category, enum_category.value)
    except Exception:
        return str(category or QuestionCategory.UNKNOWN.value)


class AnswerMode(str, Enum):
    METRIC = "METRIC"
    DETAIL = "DETAIL"
    TOPN = "TOPN"
    GROUP_AGG = "GROUP_AGG"
    DERIVED = "DERIVED"
    RULE = "RULE"
    IDENTITY = "IDENTITY"
    CHAT = "CHAT"
    INVALID = "INVALID"


class IntentType(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    GREETING = "GREETING"


class DisplayPolicy(str, Enum):
    DEFAULT = "DEFAULT"
    SHOW_ALL = "SHOW_ALL"
    HIDE_INTERMEDIATE = "HIDE_INTERMEDIATE"


class TaskRole(str, Enum):
    QUERY = "QUERY"
    ANCHOR = "ANCHOR"
    DEPENDENT = "DEPENDENT"
    BENCHMARK = "BENCHMARK"


class ResultScope(str, Enum):
    OVERALL = "OVERALL"
    DAILY = "DAILY"
    DETAIL = "DETAIL"
    TOPN = "TOPN"


class AgentRunStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class KnowledgeRequestType(str, Enum):
    TABLE = "TABLE"
    FIELD = "FIELD"
    METRIC = "METRIC"
    RELATIONSHIP = "RELATIONSHIP"
    BUSINESS_RULE = "BUSINESS_RULE"
    FRESHNESS = "FRESHNESS"
    REALTIME_FALLBACK = "REALTIME_FALLBACK"


class ChatContext(APIModel):
    question: str = ""
    time_expression: str = ""
    days: int = 0
    category: str = ""
    answer_mode: str = ""
    topic: str = ""
    data_catalog: str = ""
    user_preference: str = ""
    merchant_profile: str = ""
    context_summary: str = ""
    offloaded_files: List[str] = Field(default_factory=list)
    pending_clarification_stage: str = ""
    pending_clarification_type: str = ""
    pending_question: str = ""
    pending_clarification_options: List[str] = Field(default_factory=list)


class ChatDataSection(APIModel):
    title: str = ""
    doris_tables: List[str] = Field(default_factory=list)
    data_rows: List[Dict[str, Any]] = Field(default_factory=list)
    offloaded: bool = False
    offloaded_files: List[str] = Field(default_factory=list)
    original_row_count: int = 0
    result_summary: str = ""


class ClarificationRequest(APIModel):
    question: str = ""
    stage: str = ""
    type: str = ""
    options: List[str] = Field(default_factory=list)
    pending_question: str = ""


class ChatRequest(APIModel):
    message: str
    merchant_id: str = ""
    context: Optional[ChatContext] = None


class RunCreateRequest(APIModel):
    message: str = ""
    merchant_id: str = ""
    thread_id: str = ""
    context: Optional[ChatContext] = None


class FeedbackRequest(APIModel):
    adopted: Optional[bool] = None
    liked: Optional[bool] = None
    disliked: Optional[bool] = None


class ChatResponse(APIModel):
    id: str = ""
    answer: str = ""
    category_name: str = ""
    persisted: bool = False
    doris_tables: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    thinking_steps: List[str] = Field(default_factory=list)
    data_rows: List[Dict[str, Any]] = Field(default_factory=list)
    data_sections: List[ChatDataSection] = Field(default_factory=list)
    context: Optional[ChatContext] = None
    clarification: Optional[ClarificationRequest] = None
    debug_trace: Dict[str, Any] = Field(default_factory=dict)


class DailyReportResponse(APIModel):
    merchant_id: str = ""
    merchant_name: str = ""
    date: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)


class WikiCompressRequest(APIModel):
    category_name: str = ""
    manual_markdown: str = ""


class KnowledgeSuggestionReviewRequest(APIModel):
    approved: bool = False
    reviewer: str = ""
    review_note: str = ""


class TopicBuildRequest(APIModel):
    topic: str = ""
    table_name: str = ""
    merchant_id: str = ""
    sample_limit: int = 20
    manual_notes: str = ""
    business_knowledge: str = ""
    sample_sqls: List[str] = Field(default_factory=list)
    enum_discovery_enabled: bool = True
    enum_value_limit: int = 20
    schema_ddl: str = ""


class TopicReviewRequest(APIModel):
    approved: bool = False
    reviewer: str = ""
    review_note: str = ""


class MerchantInfo(APIModel):
    merchant_id: str = ""
    merchant_name: str = ""
    company_name: str = ""
    rows: Dict[str, Any] = Field(default_factory=dict)

    def profile_markdown(self) -> str:
        parts = [
            f"- 商家ID：{self.merchant_id}",
            f"- 商家名称：{self.merchant_name or self.company_name or '未知商家'}",
        ]
        for key, value in list(self.rows.items())[:24]:
            if value is not None and str(value) != "":
                parts.append(f"- {key}：{value}")
        return "\n".join(parts)


class MerchantRecentFocus(APIModel):
    merchant_id: str = ""
    top_categories: List[str] = Field(default_factory=list)
    top_metrics: List[str] = Field(default_factory=list)
    common_time_ranges: List[str] = Field(default_factory=list)
    focus_pattern: str = ""
    last_active_at: str = ""
    updated_at: str = ""

    def is_empty(self) -> bool:
        return not (self.top_categories or self.top_metrics or self.common_time_ranges or self.focus_pattern)


class ExtractedKeywords(APIModel):
    keywords: List[str] = Field(default_factory=list)
    business_keywords: List[str] = Field(default_factory=list)
    time_keywords: List[str] = Field(default_factory=list)
    action_keywords: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.keywords or self.business_keywords or self.time_keywords or self.action_keywords)

    def summary(self) -> str:
        return "业务词=%s，时间词=%s，动作词=%s" % (
            self.business_keywords[:8],
            self.time_keywords[:6],
            self.action_keywords[:6],
        )


class RoutingDecision(APIModel):
    route: QuestionRoute = QuestionRoute.INVALID
    complex: bool = False
    reason: str = ""


class TopicRoutingDecision(APIModel):
    primary_topic: QuestionCategory = QuestionCategory.UNKNOWN
    candidate_topics: List[QuestionCategory] = Field(default_factory=list)
    dimension_topics: List[QuestionCategory] = Field(default_factory=list)
    confidence: float = 0.0
    clarification_required: bool = False
    reason: str = ""

    def recall_topics(self) -> List[QuestionCategory]:
        seen: List[QuestionCategory] = []
        for category in [self.primary_topic] + self.candidate_topics + self.dimension_topics:
            if category != QuestionCategory.UNKNOWN and category not in seen:
                seen.append(category)
        return seen

    def display_summary(self) -> str:
        topics = [category_display(item) for item in self.recall_topics()]
        return "、".join(topics) if topics else "未知"


class RouteObjectRef(APIModel):
    ref_type: str = ""
    value: str = ""
    raw: str = ""
    confidence: float = 0.0


class RouteTimeWindow(APIModel):
    days: int = 0
    raw: str = ""
    needs_freshness_check: bool = False


class RouteTopicCandidate(APIModel):
    topic: QuestionCategory = QuestionCategory.UNKNOWN
    score: int = 0
    evidence: List[str] = Field(default_factory=list)


class RouteSlots(APIModel):
    object_refs: List[RouteObjectRef] = Field(default_factory=list)
    time_window: RouteTimeWindow = Field(default_factory=RouteTimeWindow)
    operation: str = "read"
    risk_level: str = "normal"
    topic_candidates: List[RouteTopicCandidate] = Field(default_factory=list)
    analysis_signals: List[str] = Field(default_factory=list)
    route_confidence: float = 0.0
    route_warnings: List[str] = Field(default_factory=list)


class RecallItem(APIModel):
    doc_id: str = ""
    title: str = ""
    content: str = ""
    source_type: str = ""
    topic: str = ""
    table: str = ""
    answer_mode: str = ""
    fusion_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecallBundle(APIModel):
    items: List[RecallItem] = Field(default_factory=list)
    top_score: float = 0.0
    merged_context: str = ""

    def has_strong_match(self) -> bool:
        return self.top_score >= 4.0 or any(item.fusion_score >= 4.0 for item in self.items)


class IntentSignals(APIModel):
    has_rule_evidence: bool = False
    rule_evidence_refs: List[str] = Field(default_factory=list)
    rule_evidence_count: int = 0
    has_data_intent: bool = False
    data_topics: List[QuestionCategory] = Field(default_factory=list)
    has_analysis_intent: bool = False
    open_diagnostic_intent: str = ""
    rule_confidence: float = 0.0
    data_confidence: float = 0.0
    suggested_actions: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)


class KnowledgeRequest(APIModel):
    type: KnowledgeRequestType = KnowledgeRequestType.TABLE
    query: str = ""
    needed_for_task_id: str = ""
    reason: str = ""
    source_phrase: str = ""
    expected_refs: List[str] = Field(default_factory=list)
    round: int = 0
    request_key: str = ""


class KnowledgeRetrievalRequest(APIModel):
    query: str = ""
    keywords: List[str] = Field(default_factory=list)
    history_rows: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_context: str = ""
    merchant_id: str = ""
    topic_categories: List[QuestionCategory] = Field(default_factory=list)
    knowledge_request: Optional[KnowledgeRequest] = None
    route_slots: Dict[str, Any] = Field(default_factory=dict)
    round: int = 0


class RecallRoundTrace(APIModel):
    request_key: str = ""
    query: str = ""
    topics: List[str] = Field(default_factory=list)
    backend: str = "hybrid"
    recall_queries: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    new_refs: List[str] = Field(default_factory=list)
    blocked_reason: str = ""
    item_count: int = 0


class KnowledgeBundle(APIModel):
    recall_bundle: RecallBundle = Field(default_factory=RecallBundle)
    source_refs: List[str] = Field(default_factory=list)
    recall_rounds: List[RecallRoundTrace] = Field(default_factory=list)
    backend: str = "hybrid"
    index_version: str = ""
    semantic_source_hash: str = ""


class KnowledgeRef(APIModel):
    ref_id: str = ""
    ref_type: str = ""
    table: str = ""
    column: str = ""
    relationship_id: str = ""
    title: str = ""
    reason: str = ""
    score: float = 0.0


class AgentAction(APIModel):
    id: str = ""
    node: str = ""
    agent: str = ""
    description: str = ""


class AgentDecision(APIModel):
    selected_action: str = ""
    selected_node: str = ""
    available_actions: List[str] = Field(default_factory=list)
    reason: str = ""
    budget_exhausted: bool = False


class ActionResult(APIModel):
    action: str = ""
    node: str = ""
    status: str = ""
    message: str = ""
    retryable: bool = False


class AgentActionTrace(APIModel):
    round: int = 0
    action: str = ""
    node: str = ""
    agent: str = ""
    status: str = ""
    reason: str = ""
    available_actions: List[str] = Field(default_factory=list)


class PlannerReflectionResult(APIModel):
    passed: bool = True
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    suggested_knowledge_requests: List[KnowledgeRequest] = Field(default_factory=list)
    repair_hints: List[str] = Field(default_factory=list)
    repair_reason: str = ""
    repair_requests: List["PlannerRepairRequest"] = Field(default_factory=list)


class NodeToolResult(APIModel):
    status: str = ""
    summary: str = ""
    error_type: str = ""


class NodeToolCall(APIModel):
    task_id: str = ""
    tool_name: str = ""
    status: str = ""
    input_summary: str = ""
    output_summary: str = ""
    error_type: str = ""
    repair_round: int = 0
    duration_ms: int = 0


class SourceRef(APIModel):
    ref_type: str = ""
    path: str = ""
    title: str = ""
    locator: str = ""
    reason: str = ""
    merchant_uri: str = ""
    context_layer: str = ""


class ArtifactRef(APIModel):
    artifact_id: str = ""
    namespace: str = ""
    path: str = ""
    relative_path: str = ""
    title: str = ""
    reason: str = ""
    bytes: int = 0
    estimated_chars: int = 0
    sha256: str = ""
    merchant_uri: str = ""
    context_layer: str = "L2"


class ImportantFact(APIModel):
    key: str = ""
    value: str = ""
    category: str = ""
    priority: int = 0
    source_refs: List[SourceRef] = Field(default_factory=list)


class ContextSnapshot(APIModel):
    stage: str = ""
    summary: str = ""
    protected_facts: List[ImportantFact] = Field(default_factory=list)
    source_refs: List[SourceRef] = Field(default_factory=list)
    token_budget: int = 0
    truncated: bool = False


class ContextDelta(APIModel):
    context_hash: str = ""
    unchanged_refs: List[str] = Field(default_factory=list)
    changed_refs: List[str] = Field(default_factory=list)
    inline_chars: int = 0
    artifact_refs: List[str] = Field(default_factory=list)


class ContextPackage(APIModel):
    package_id: str = ""
    run_id: str = ""
    thread_id: str = ""
    stage: str = ""
    agent: str = ""
    task_id: str = ""
    question: str = ""
    merchant_id: str = ""
    goal: str = ""
    constraints: List[str] = Field(default_factory=list)
    protected_facts: List[ImportantFact] = Field(default_factory=list)
    source_refs: List[SourceRef] = Field(default_factory=list)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    allowed_tables: List[str] = Field(default_factory=list)
    allowed_metrics: List[str] = Field(default_factory=list)
    evidence_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    inline_budget_chars: int = 0
    input_chars: int = 0
    offload_reason: str = ""
    context_hash: str = ""
    context_delta: ContextDelta = Field(default_factory=ContextDelta)


class TraceSpan(APIModel):
    span_id: str = ""
    parent_span_id: str = ""
    run_id: str = ""
    step_id: str = ""
    kind: str = ""
    name: str = ""
    status: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_prompt_chars: int = 0
    estimated_completion_chars: int = 0
    sql_hash: str = ""
    table: str = ""
    row_count: int = 0
    error_code: str = ""
    error_message: str = ""
    retry_or_fallback_count: int = 0
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunStep(APIModel):
    step_id: str = ""
    run_id: str = ""
    action_id: str = ""
    agent: str = ""
    node: str = ""
    status: str = ""
    reason: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    input_summary: str = ""
    output_summary: str = ""
    error_code: str = ""
    error_message: str = ""
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)


class SemanticCatalogVersion(APIModel):
    semantic_version: str = ""
    schema_version: str = ""
    topic: str = ""
    table: str = ""
    source_hash: str = ""
    published_at: str = ""
    live_schema_checked_at: str = ""


class SchemaDriftReport(APIModel):
    topic: str = ""
    table: str = ""
    semantic_version: str = ""
    schema_version: str = ""
    source_hash: str = ""
    live_schema_checked_at: str = ""
    missing_live_columns: List[str] = Field(default_factory=list)
    extra_live_columns: List[str] = Field(default_factory=list)
    type_changed_columns: List[Dict[str, Any]] = Field(default_factory=list)
    live_column_count: int = 0
    semantic_column_count: int = 0


class PlannerRepairRequest(APIModel):
    reason: str = ""
    stage: str = ""
    action: str = ""
    query: str = ""
    task_id: str = ""
    evidence: str = ""
    repair_hints: List[str] = Field(default_factory=list)
    knowledge_requests: List[KnowledgeRequest] = Field(default_factory=list)
    source: str = ""


class ToolRuntimePolicy(APIModel):
    tool_name: str = ""
    timeout_seconds: int = 0
    max_retries: int = 0
    backoff_seconds: float = 0.0
    retryable_errors: List[str] = Field(default_factory=list)
    non_retryable_errors: List[str] = Field(default_factory=list)


class ToolCachePolicy(APIModel):
    enabled: bool = False
    namespace: str = ""
    ttl_seconds: int = 0
    key_fields: List[str] = Field(default_factory=list)


class LoadBalancerTarget(APIModel):
    name: str = ""
    endpoint: str = ""
    weight: int = 1
    healthy: bool = True


class ToolRuntimeMetrics(APIModel):
    tool_name: str = ""
    calls: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    circuit_blocked: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    total_duration_ms: int = 0
    p95_duration_ms: int = 0
    p99_duration_ms: int = 0
    last_error_type: str = ""
    last_target: str = ""


class RuntimeAlert(APIModel):
    alert_id: str = ""
    severity: str = "warning"
    code: str = ""
    message: str = ""
    tool_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    created_at: str = ""


class ToolFailureRecord(APIModel):
    fingerprint: str = ""
    tool_name: str = ""
    error_type: str = ""
    error_message: str = ""
    count: int = 0
    blocked: bool = False


class CircuitBreakerState(APIModel):
    tool_name: str = ""
    open: bool = False
    failure_count: int = 0
    reason: str = ""
    opened_at_ms: int = 0
    open_until_ms: int = 0


class ToolCallRequest(APIModel):
    id: str = ""
    name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolCallExecutionResult(APIModel):
    id: str = ""
    name: str = ""
    status: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""
    duration_ms: int = 0
    attempts: int = 0
    cache_hit: bool = False
    cache_key: str = ""
    rate_limited: bool = False
    target: str = ""


class FreshnessCheckResult(APIModel):
    task_id: str = ""
    table: str = ""
    checked: bool = False
    status: str = ""
    pt_column: str = "pt"
    requested_days: int = 0
    min_pt: str = ""
    max_pt: str = ""
    fallback_table: str = ""
    reason: str = ""


class NodePlanContract(APIModel):
    task_id: str = ""
    question: str = ""
    preferred_table: str = ""
    allowed_columns: List[str] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)
    metric_column: str = ""
    metric_name: str = ""
    metric_formula: str = ""
    metric_specs: List[Dict[str, Any]] = Field(default_factory=list)
    group_by_column: str = ""
    output_keys: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)
    days: int = 0
    limit: int = 0
    merchant_id: str = ""
    merchant_filter_column: str = ""
    answer_mode: str = ""
    task_role: str = ""
    sql_strategy: str = ""
    upstream_entity_sets: List[Dict[str, Any]] = Field(default_factory=list)
    metric_resolution: Dict[str, Any] = Field(default_factory=dict)


class NodePlanCritiqueResult(APIModel):
    task_id: str = ""
    valid: bool = True
    code: str = ""
    message: str = ""
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    graph_repairable: bool = False


class SqlDraftDecision(APIModel):
    task_id: str = ""
    source: str = ""
    llm_attempted: bool = False
    structured_fallback_used: bool = False
    fallback_reason: str = ""
    reason: str = ""


class NodeAgentContext(APIModel):
    task_id: str = ""
    task_kind: str = ""
    selected_tools: List[str] = Field(default_factory=list)


class NodeTaskProfile(APIModel):
    task_id: str = ""
    task_kind: str = ""
    sql_strategy: str = ""
    selected_tools: List[str] = Field(default_factory=list)
    reason: str = ""
    risk_controls: List[str] = Field(default_factory=list)
    contract_status: str = ""
    sql_draft_source: str = ""
    contract_critique_reason: str = ""


class NodeExecutionBatch(APIModel):
    batch_id: str = ""
    ready_task_ids: List[str] = Field(default_factory=list)
    submitted_task_ids: List[str] = Field(default_factory=list)
    completed_task_ids: List[str] = Field(default_factory=list)
    timed_out_task_ids: List[str] = Field(default_factory=list)
    blocked_task_ids: List[str] = Field(default_factory=list)
    max_concurrency: int = 0
    timeout_seconds: int = 0
    duration_ms: int = 0


class SkillManifest(APIModel):
    domain: str = ""
    display_name: str = ""
    trigger_terms: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    entity_keys: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    graph_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_hints: List[str] = Field(default_factory=list)
    field_warnings: List[str] = Field(default_factory=list)
    answer_guidelines: List[str] = Field(default_factory=list)
    source_path: str = ""


class PlanningAssetEntry(APIModel):
    key: str = ""
    table: str = ""
    topic: str = ""
    title: str = ""
    columns: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    source_ref_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipEntry(APIModel):
    relationship_id: str = ""
    left_table: str = ""
    right_table: str = ""
    join_keys: List[Dict[str, str]] = Field(default_factory=list)
    grain: str = ""
    path_semantics: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)
    source_ref_id: str = ""
    description: str = ""


class PlanningAssetPack(APIModel):
    tables: List[PlanningAssetEntry] = Field(default_factory=list)
    fields: List[PlanningAssetEntry] = Field(default_factory=list)
    metrics: List[PlanningAssetEntry] = Field(default_factory=list)
    entity_keys: List[PlanningAssetEntry] = Field(default_factory=list)
    relationships: List[RelationshipEntry] = Field(default_factory=list)
    rules: List[PlanningAssetEntry] = Field(default_factory=list)
    terms: List[PlanningAssetEntry] = Field(default_factory=list)
    freshness: List[PlanningAssetEntry] = Field(default_factory=list)
    realtime_fallbacks: List[PlanningAssetEntry] = Field(default_factory=list)
    skills: List[SkillManifest] = Field(default_factory=list)
    source_refs: Dict[str, RecallItem] = Field(default_factory=dict)
    schema_source: Dict[str, str] = Field(default_factory=dict)
    missing_live_columns: Dict[str, List[str]] = Field(default_factory=dict)
    relationship_closure: List[str] = Field(default_factory=list)
    skill_semantic_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    metric_compaction: Dict[str, Any] = Field(default_factory=dict)
    semantic_catalog_version: Dict[str, SemanticCatalogVersion] = Field(default_factory=dict)
    schema_drift_reports: List[SchemaDriftReport] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.tables or self.fields or self.metrics or self.relationships or self.rules or self.terms)

    def known_tables(self) -> List[str]:
        return sorted({item.table or item.key for item in self.tables if item.table or item.key})

    def known_columns(self, table: str) -> List[str]:
        cols = set()
        for item in self.tables:
            if (item.table or item.key) == table:
                cols.update(item.columns)
        for item in self.fields + self.metrics + self.entity_keys:
            if item.table == table and item.key:
                cols.add(item.key)
        return sorted(cols)


class GraphValidationGap(APIModel):
    code: str = ""
    evidence: str = ""
    task_id: str = ""
    reason: str = ""


class GraphValidationResult(APIModel):
    valid: bool = False
    gaps: List[GraphValidationGap] = Field(default_factory=list)
    repairable: bool = True
    recommended_knowledge_requests: List[KnowledgeRequest] = Field(default_factory=list)


class SqlValidationResult(APIModel):
    valid: bool = False
    error_code: str = ""
    message: str = ""
    base_tables: List[str] = Field(default_factory=list)
    cte_names: List[str] = Field(default_factory=list)
    unknown_tables: List[str] = Field(default_factory=list)
    unknown_columns: List[str] = Field(default_factory=list)


class SqlRepairAttempt(APIModel):
    task_id: str = ""
    round: int = 0
    original_sql: str = ""
    repaired_sql: str = ""
    error_code: str = ""
    error_message: str = ""
    success: bool = False


class EntitySet(APIModel):
    task_id: str = ""
    join_key: str = ""
    values: List[Any] = Field(default_factory=list)
    column_values: Dict[str, List[Any]] = Field(default_factory=dict)
    truncated: bool = False
    source_row_count: int = 0
    source_key: str = ""
    requested_join_key: str = ""
    missing_reason: str = ""


class NodeExecutionContext(APIModel):
    merchant_id: str = ""
    question: str = ""
    upstream_entity_sets: List[EntitySet] = Field(default_factory=list)
    upstream_rows: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceGap(APIModel):
    code: str = ""
    task_id: str = ""
    evidence: str = ""
    reason: str = ""
    severity: str = ""
    disclosure_required: bool = False
    source: str = ""
    answer_instruction: str = ""


class VerifiedEvidence(APIModel):
    passed: bool = False
    covered_evidence: List[str] = Field(default_factory=list)
    derived_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[EvidenceGap] = Field(default_factory=list)
    blocking_gaps: List[EvidenceGap] = Field(default_factory=list)
    warning_gaps: List[EvidenceGap] = Field(default_factory=list)
    answer_guard_required: bool = False
    required_disclosures: List[str] = Field(default_factory=list)
    partial_answer_reason: str = ""


class PlanDependency(APIModel):
    anchor_task_id: str = ""
    dependent_task_id: str = ""
    join_key: str = ""
    anchor_column: str = ""
    dependent_column: str = ""
    relation_type: str = "LOOKUP"


class QuestionIntent(APIModel):
    question: str = ""
    intent_type: IntentType = IntentType.INVALID
    category: QuestionCategory = QuestionCategory.UNKNOWN
    answer_mode: AnswerMode = AnswerMode.INVALID
    plan_task_id: str = ""
    task_role: TaskRole = TaskRole.QUERY
    preferred_table: str = ""
    metric_column: str = ""
    metric_name: str = ""
    metric_formula: str = ""
    metric_specs: List[Dict[str, Any]] = Field(default_factory=list)
    group_by_column: str = ""
    group_by_name: str = ""
    filter_column: str = ""
    filter_value: str = ""
    days: int = 7
    limit: int = 20
    required_evidence: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)
    depends_on_task_ids: List[str] = Field(default_factory=list)
    knowledge_ref_ids: List[str] = Field(default_factory=list)
    knowledge_refs: List[KnowledgeRef] = Field(default_factory=list)
    display_policy: DisplayPolicy = DisplayPolicy.DEFAULT
    analysis_source: str = ""
    analysis_note: str = ""
    sql_strategy: str = "llm_plan_bound_first"
    sql: str = ""
    metric_resolution: Dict[str, Any] = Field(default_factory=dict)


class QueryPlan(APIModel):
    intents: List[QuestionIntent] = Field(default_factory=list)
    dependencies: List[PlanDependency] = Field(default_factory=list)
    knowledge_requests: List[KnowledgeRequest] = Field(default_factory=list)
    evidence_contracts: List[Dict[str, Any]] = Field(default_factory=list)
    clarification_needs: List[str] = Field(default_factory=list)
    final_required_evidence: List[str] = Field(default_factory=list)
    final_evidence_column_hints: Dict[str, List[str]] = Field(default_factory=dict)
    agent_trace: List[str] = Field(default_factory=list)
    question_understanding: Dict[str, Any] = Field(default_factory=dict)
    compiler_trace: List[str] = Field(default_factory=list)
    planner_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    planner_tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    planner_loaded_refs: List[str] = Field(default_factory=list)
    planner_context_files: List[Dict[str, Any]] = Field(default_factory=list)
    planner_prompt_stats: Dict[str, Any] = Field(default_factory=dict)
    display_policy: DisplayPolicy = DisplayPolicy.DEFAULT
    display_title: str = ""

    def categories(self) -> List[QuestionCategory]:
        seen: List[QuestionCategory] = []
        for intent in self.intents:
            if intent.category != QuestionCategory.UNKNOWN and intent.category not in seen:
                seen.append(intent.category)
        return seen


class SqlDraft(APIModel):
    sql: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class QueryBundle(APIModel):
    sql: str = ""
    tables: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    failed: bool = False
    error: str = ""
    summary: str = ""
    offloaded_files: List[str] = Field(default_factory=list)
    original_row_count: int = 0
    duration_ms: int = 0
    cache_hit: bool = False
    cache_key: str = ""

    def effective_row_count(self) -> int:
        return self.original_row_count or len(self.rows)


class ReActStep(APIModel):
    round: int = 0
    reason: str = ""
    action: str = ""
    observation: str = ""


class AgentTask(APIModel):
    task_id: str = ""
    plan_index: int = 0
    sub_agent_type: str = "NODE_WORKER"
    instruction: str = ""
    depends_on: List[str] = Field(default_factory=list)
    plan_dependencies: List[PlanDependency] = Field(default_factory=list)


class AgentTaskResult(APIModel):
    task_id: str = ""
    sub_agent_type: str = "NODE_WORKER"
    success: bool = False
    summary: str = ""
    query_bundle: QueryBundle = Field(default_factory=QueryBundle)
    react_trace: List[ReActStep] = Field(default_factory=list)
    sql_repairs: List[SqlRepairAttempt] = Field(default_factory=list)
    validation_results: List[SqlValidationResult] = Field(default_factory=list)
    entity_set: Optional[EntitySet] = None
    node_tool_traces: List[NodeToolCall] = Field(default_factory=list)
    node_task_profile: NodeTaskProfile = Field(default_factory=NodeTaskProfile)
    freshness_reports: List[FreshnessCheckResult] = Field(default_factory=list)
    node_plan_contract: NodePlanContract = Field(default_factory=NodePlanContract)
    node_plan_critique: NodePlanCritiqueResult = Field(default_factory=NodePlanCritiqueResult)
    sql_draft_decision: SqlDraftDecision = Field(default_factory=SqlDraftDecision)


class EvidenceCheckResult(APIModel):
    passed: bool = False
    summary: str = ""
    gaps: List[str] = Field(default_factory=list)


class AgentRunResult(APIModel):
    tasks: List[AgentTask] = Field(default_factory=list)
    task_results: List[AgentTaskResult] = Field(default_factory=list)
    query_bundles: List[QueryBundle] = Field(default_factory=list)
    merged_query_bundle: QueryBundle = Field(default_factory=QueryBundle)
    evidence_check: EvidenceCheckResult = Field(default_factory=EvidenceCheckResult)
    verified_evidence: VerifiedEvidence = Field(default_factory=VerifiedEvidence)
    sql_repairs: List[SqlRepairAttempt] = Field(default_factory=list)
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    partial_answer_reason: str = ""
    reflection_notes: List[str] = Field(default_factory=list)
    node_tool_traces: List[NodeToolCall] = Field(default_factory=list)
    node_task_profiles: List[NodeTaskProfile] = Field(default_factory=list)
    freshness_reports: List[FreshnessCheckResult] = Field(default_factory=list)
    node_plan_contracts: List[NodePlanContract] = Field(default_factory=list)
    node_plan_critiques: List[NodePlanCritiqueResult] = Field(default_factory=list)
    sql_draft_decisions: List[SqlDraftDecision] = Field(default_factory=list)
    node_execution_batches: List[NodeExecutionBatch] = Field(default_factory=list)


class PendingAnswer(APIModel):
    id: str
    question: str
    answer: str
    merchant_id: str
    merchant_name: str
    category_name: str
    doris_tables: str
    suggested_questions: str
    langfuse_trace_id: str = ""
    langfuse_session_id: str = ""
    create_time: datetime = Field(default_factory=datetime.now)


class ThreadData(APIModel):
    thread_id: str = ""
    run_id: str = ""
    workspace_path: str = ""
    uploads_path: str = ""
    outputs_path: str = ""
    offloaded_files: List[str] = Field(default_factory=list)


class AgentThreadRecord(APIModel):
    thread_id: str
    merchant_id: str
    topic: str = ""
    context: Optional[ChatContext] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AgentRunEventRecord(APIModel):
    event_id: str
    run_id: str
    thread_id: str
    event_type: str
    node: str = ""
    step_id: str = ""
    tool_call_id: str = ""
    parent_id: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class AgentRunRecord(APIModel):
    run_id: str
    thread_id: str
    merchant_id: str
    question: str = ""
    status: AgentRunStatus = AgentRunStatus.CREATED
    answer: Optional[ChatResponse] = None
    error: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    final_answer_hash: str = ""
    performance_summary: Dict[str, Any] = Field(default_factory=dict)
    trace_path: str = ""
    checkpoint_ref: Dict[str, Any] = Field(default_factory=dict)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    resumable: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
