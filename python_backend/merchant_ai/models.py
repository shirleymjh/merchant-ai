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


class SubAgentResultEnvelope(APIModel):
    """Worker-independent result contract consumed by the Lead Agent."""

    status: str = "completed"
    summary: str = ""
    evidence_refs: List[Any] = Field(default_factory=list)
    artifact_refs: List[Any] = Field(default_factory=list)
    gaps: List[Any] = Field(default_factory=list)
    recommended_next_action: str = ""
    retryable: bool = False
    payload: Dict[str, Any] = Field(default_factory=dict)
    task_kind: str = ""


class SubAgentDelegationTask(APIModel):
    task_kind: str
    objective: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: List[str] = Field(default_factory=list)
    timeout: int = 60


class SubAgentDelegationPlan(APIModel):
    tasks: List[SubAgentDelegationTask] = Field(default_factory=list)
    parallel: bool = False
    isolation_mode: str = "worker"
    read_artifact_policy: str = "on_completion"
    failure_strategy: str = "continue_partial"
    reason: str = ""


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


class UserIdentity(APIModel):
    user_id: str = ""
    merchant_id: str = ""
    display_name: str = ""
    role: str = "merchant_operator"
    region: str = ""
    language: str = "zh-CN"
    store_ids: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)

    def prompt_markdown(self) -> str:
        return "\n".join(
            [
                f"- 用户：{self.display_name or self.user_id or '当前商家用户'}",
                f"- 角色：{self.role or 'merchant_operator'}",
                f"- Region：{self.region or '未限定'}",
                f"- Language：{self.language or 'zh-CN'}",
                f"- 门店范围：{', '.join(self.store_ids) if self.store_ids else '当前商家全部授权门店'}",
            ]
        )


class ChatContext(APIModel):
    question: str = ""
    time_expression: str = ""
    days: int = 0
    category: str = ""
    answer_mode: str = ""
    topic: str = ""
    topics: List[QuestionCategory] = Field(default_factory=list)
    metric_keys: List[str] = Field(default_factory=list)
    dimension_keys: List[str] = Field(default_factory=list)
    data_catalog: str = ""
    user_preference: str = ""
    merchant_profile: str = ""
    context_summary: str = ""
    offloaded_files: List[str] = Field(default_factory=list)
    clarification_resolved: bool = False
    resolved_time_window_days: int = 0
    metric_focus: str = ""
    priority_goal: str = ""
    pending_clarification_stage: str = ""
    pending_clarification_type: str = ""
    pending_question: str = ""
    pending_clarification_options: List[str] = Field(default_factory=list)
    user_identity: UserIdentity = Field(default_factory=UserIdentity)


class ConversationMessage(APIModel):
    role: str = ""
    text: str = ""
    id: str = ""
    local_id: str = ""
    created_at: str = ""


class ChatDataSection(APIModel):
    title: str = ""
    result_role: str = ""
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


class AttachmentReference(APIModel):
    id: str = ""
    name: str = ""
    type: str = ""
    size: int = 0


class ChatRequest(APIModel):
    message: str
    merchant_id: str = ""
    context: Optional[ChatContext] = None
    message_history: List[ConversationMessage] = Field(default_factory=list)
    attachments: List[AttachmentReference] = Field(default_factory=list)
    user_identity: UserIdentity = Field(default_factory=UserIdentity)


class RunCreateRequest(APIModel):
    message: str = ""
    merchant_id: str = ""
    thread_id: str = ""
    context: Optional[ChatContext] = None
    message_history: List[ConversationMessage] = Field(default_factory=list)
    attachments: List[AttachmentReference] = Field(default_factory=list)
    user_identity: UserIdentity = Field(default_factory=UserIdentity)


class FeedbackRequest(APIModel):
    adopted: Optional[bool] = None
    liked: Optional[bool] = None
    disliked: Optional[bool] = None


class MetricDefinitionPreferenceRequest(APIModel):
    action: str = "confirm_default"
    merchant_id: str = ""
    metric_key: str = ""
    display_name: str = ""
    description: str = ""
    formula: str = ""
    semantic_ref: str = ""
    source_table: str = ""
    question: str = ""
    answer_id: str = ""
    note: str = ""
    reviewer: str = ""


class MemoryItemPatchRequest(APIModel):
    status: Optional[str] = None
    confidence: Optional[float] = None
    valid_until: Optional[str] = None
    retention_days: Optional[int] = None
    visibility: Optional[str] = None
    allowed_roles: Optional[List[str]] = None
    approved_by: Optional[str] = None


class MemoryCleanupRequest(APIModel):
    hard_delete: bool = False
    dry_run: bool = False


class MemoryRecallEvalCase(APIModel):
    case_id: str = ""
    question: str = ""
    expected_memory_ids: List[str] = Field(default_factory=list)
    unexpected_memory_ids: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    time_windows: List[int] = Field(default_factory=list)
    access_role: str = "merchant_analyst"


class MemoryRecallEvaluationRequest(APIModel):
    cases: List[MemoryRecallEvalCase] = Field(default_factory=list)
    budget_tokens: int = 0
    budget_chars: int = 0


class GoldenEvaluationRequest(APIModel):
    merchant_id: str = ""
    case_ids: List[str] = Field(default_factory=list)
    limit: int = 0
    persist_report: bool = True
    persist_governance_items: bool = True
    cases_path: str = ""
    partition_date_anchor_enabled: bool = True


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
    merchant_experience: Dict[str, Any] = Field(default_factory=dict)
    debug_trace: Dict[str, Any] = Field(default_factory=dict)


class DailyReportResponse(APIModel):
    merchant_id: str = ""
    merchant_name: str = ""
    date: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    anomaly_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    drill_down_actions: List[Dict[str, Any]] = Field(default_factory=list)
    traceability: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)


class KnowledgeSuggestionReviewRequest(APIModel):
    approved: bool = False
    reviewer: str = ""
    review_note: str = ""
    action: str = "review"


class KnowledgeSuggestionActionRequest(APIModel):
    action: str = ""
    merchant_id: str = ""
    actor: str = ""
    note: str = ""
    conflict_resolution: str = ""


class KnowledgeSuggestionPublishRequest(APIModel):
    reviewer: str = ""
    review_note: str = ""
    topic: str = ""
    table_name: str = ""
    auto_index: bool = True


class SkillDraftReviewRequest(APIModel):
    approved: bool = False
    reviewer: str = ""
    review_note: str = ""


class SkillEvaluationCase(APIModel):
    case_id: str = ""
    question: str = ""
    expected_skill: str = ""
    expect_trigger: bool = True
    question_understanding: Dict[str, Any] = Field(default_factory=dict)
    planned_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_rows: List[Dict[str, Any]] = Field(default_factory=list)
    has_rule_context: bool = False


class SkillEvaluationRequest(APIModel):
    cases: List[SkillEvaluationCase] = Field(default_factory=list)


class KnowledgeSuggestion(APIModel):
    suggestion_id: str = ""
    suggestion_type: str = "metric"
    status: str = "candidate"
    source: str = ""
    source_memory_id: str = ""
    source_refs: List[str] = Field(default_factory=list)
    topic: str = ""
    metric_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    source_table: str = ""
    source_fields: List[str] = Field(default_factory=list)
    aggregation: str = ""
    filter_conditions: List[str] = Field(default_factory=list)
    dependency_fields: List[str] = Field(default_factory=list)
    reviewer: str = ""
    review_note: str = ""
    approved_by: str = ""
    reviewed_at: str = ""
    publish_requested_at: str = ""
    publish_requested_by: str = ""
    published_ref_id: str = ""
    indexed_at: str = ""
    scope_type: str = "merchant"
    merchant_action: str = ""
    actioned_by: str = ""
    actioned_at: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


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


class KeywordMention(APIModel):
    phrase: str = ""
    canonical_key: str = ""
    display_name: str = ""
    kind: str = ""
    topic: QuestionCategory = QuestionCategory.UNKNOWN
    score: float = 0.0
    source: str = ""


class ExtractedKeywords(APIModel):
    normalized_question: str = ""
    keywords: List[str] = Field(default_factory=list)
    business_keywords: List[str] = Field(default_factory=list)
    topic_keywords: List[str] = Field(default_factory=list)
    metric_keywords: List[str] = Field(default_factory=list)
    dimension_keywords: List[str] = Field(default_factory=list)
    time_keywords: List[str] = Field(default_factory=list)
    action_keywords: List[str] = Field(default_factory=list)
    ranking_keywords: List[str] = Field(default_factory=list)
    mentions: List[KeywordMention] = Field(default_factory=list)
    topic_scores: Dict[str, float] = Field(default_factory=dict)
    analysis_intent: str = "lookup"
    confidence: float = 0.0
    unresolved_phrases: List[str] = Field(default_factory=list)
    excluded_topics: List[QuestionCategory] = Field(default_factory=list)
    excluded_metric_keywords: List[str] = Field(default_factory=list)
    ambiguous_metric_keywords: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.keywords or self.business_keywords or self.time_keywords or self.action_keywords or self.mentions)

    def summary(self) -> str:
        return "Topic词=%s，指标词=%s，维度词=%s，时间词=%s，动作词=%s，意图=%s，置信度=%.2f" % (
            self.topic_keywords[:8],
            self.metric_keywords[:8],
            self.dimension_keywords[:6],
            self.time_keywords[:6],
            self.action_keywords[:6],
            self.analysis_intent,
            self.confidence,
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
    routing_mode: str = "open"
    workspace_topics: List[QuestionCategory] = Field(default_factory=list)
    scope_disclosure_required: bool = False
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
    score: float = 0.0
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
        versioned_items = [item for item in self.items if str((item.metadata or {}).get("scoreVersion") or "") == "recall_v2"]
        if versioned_items:
            versioned_strong = any(
                int((item.metadata or {}).get("protectionTier") or 0) >= 1
                or float((item.metadata or {}).get("finalScore") if (item.metadata or {}).get("finalScore") is not None else item.fusion_score or 0.0) >= 0.5
                for item in versioned_items
            )
            legacy_strong = any(
                item.fusion_score >= 4.0
                for item in self.items
                if str((item.metadata or {}).get("scoreVersion") or "") != "recall_v2"
            )
            return versioned_strong or legacy_strong
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


class FastUnderstandingResult(APIModel):
    complexity: str = "unknown"
    intent_kind: str = "unknown"
    analysis_intent: str = "lookup"
    topics: List[QuestionCategory] = Field(default_factory=list)
    object_refs: Dict[str, List[str]] = Field(default_factory=dict)
    time_window_days: int = 0
    metric_phrases: List[str] = Field(default_factory=list)
    needs_planner: bool = True
    needs_knowledge: bool = True
    suggested_actions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasons: List[str] = Field(default_factory=list)


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
    access_role: str = "merchant_operator"
    permissions: List[str] = Field(default_factory=list)
    previous_user_question: str = ""
    session_context: str = ""
    topic_categories: List[QuestionCategory] = Field(default_factory=list)
    knowledge_request: Optional[KnowledgeRequest] = None
    route_slots: Dict[str, Any] = Field(default_factory=dict)
    intent_kind: str = ""
    complexity: str = ""
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
    recall_channels: List[str] = Field(default_factory=list)
    source_type_top_k: Dict[str, int] = Field(default_factory=dict)
    vector_enabled: bool = False
    vector_disabled: bool = False
    metric_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_profile: Dict[str, Any] = Field(default_factory=dict)
    query_type: str = ""
    intent_kind: str = ""
    complexity: str = ""
    retrieval_lanes: List[Dict[str, Any]] = Field(default_factory=list)
    rewritten_query: str = ""
    governance_filtered: Dict[str, int] = Field(default_factory=dict)
    rerank_applied: bool = False


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
    required_state_keys: List[str] = Field(default_factory=list)
    required_state_flags: List[str] = Field(default_factory=list)
    expected_state_keys: List[str] = Field(default_factory=list)
    expected_state_flags: List[str] = Field(default_factory=list)
    fallback_action: str = ""


class AgentDecision(APIModel):
    selected_action: str = ""
    selected_node: str = ""
    available_actions: List[str] = Field(default_factory=list)
    reason: str = ""
    budget_exhausted: bool = False
    observation: str = ""
    source: str = "policy"


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
    agent_context_policy: Dict[str, Any] = Field(default_factory=dict)
    evidence_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    inline_budget_chars: int = 0
    input_chars: int = 0
    offload_reason: str = ""
    context_hash: str = ""
    context_delta: ContextDelta = Field(default_factory=ContextDelta)


class MiddlewareEvent(APIModel):
    event_id: str = ""
    middleware: str = ""
    stage: str = ""
    status: str = "ok"
    severity: str = "info"
    channel: str = "trace"
    code: str = ""
    message: str = ""
    input_chars: int = 0
    output_chars: int = 0
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ContextBudgetReport(APIModel):
    stage: str = ""
    window_tokens: int = 0
    estimated_tokens: int = 0
    usage_ratio: float = 0.0
    threshold_ratio: float = 0.0
    over_budget: bool = False
    protected_fact_count: int = 0
    artifact_count: int = 0
    summary_chars: int = 0
    decision: str = ""


class ContextCompressionEvent(APIModel):
    stage: str = ""
    before_tokens: int = 0
    after_tokens: int = 0
    target_ratio: float = 0.0
    summary_artifact: ArtifactRef = Field(default_factory=ArtifactRef)
    protected_keys: List[str] = Field(default_factory=list)
    reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class ContextAssemblyReport(APIModel):
    stage: str = ""
    agent: str = ""
    input_chars: int = 0
    output_chars: int = 0
    budget_chars: int = 0
    compacted: bool = False
    trimmed_sections: List[str] = Field(default_factory=list)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    context_hash: str = ""
    reason: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class ContextManifest(APIModel):
    stage: str = ""
    agent: str = ""
    context_package_id: str = ""
    context_hash: str = ""
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    cache_layout: Dict[str, Any] = Field(default_factory=dict)
    quarantine_policy: Dict[str, Any] = Field(default_factory=dict)
    allowed_tables: List[str] = Field(default_factory=list)
    allowed_metrics: List[str] = Field(default_factory=list)
    memory_ids: List[str] = Field(default_factory=list)
    semantic_ref_ids: List[str] = Field(default_factory=list)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    budget_report: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ToolCallLedgerEntry(APIModel):
    tool_call_id: str = ""
    tool_name: str = ""
    idempotency_key: str = ""
    stage: str = ""
    status: str = ""
    duration_ms: int = 0
    attempts: int = 0
    cache_hit: bool = False
    rate_limited: bool = False
    target: str = ""
    error_type: str = ""
    error_message: str = ""
    result_ref: ArtifactRef = Field(default_factory=ArtifactRef)
    created_at: datetime = Field(default_factory=datetime.now)


class ToolCallRecoveryEvent(APIModel):
    tool_call_id: str = ""
    tool_name: str = ""
    stage: str = ""
    action: str = ""
    reason: str = ""
    status_before: str = ""
    status_after: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class WorkspaceManifestEntry(APIModel):
    path: str = ""
    relative_path: str = ""
    namespace: str = ""
    bytes: int = 0
    estimated_chars: int = 0
    sha256: str = ""
    merchant_uri: str = ""
    updated_at: str = ""


class WorkspaceManifest(APIModel):
    root: str = ""
    entries: List[WorkspaceManifestEntry] = Field(default_factory=list)
    entry_count: int = 0
    total_bytes: int = 0
    updated_at: str = ""


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
    fallback_tools: List[str] = Field(default_factory=list)


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


class ToolRecoveryAction(APIModel):
    error_type: str = ""
    tool_kind: str = ""
    action: str = ""
    retryable: bool = False
    fallback_tools: List[str] = Field(default_factory=list)
    message: str = ""


class ToolFailureRecord(APIModel):
    fingerprint: str = ""
    tool_name: str = ""
    service_name: str = ""
    target: str = ""
    merchant_id: str = ""
    thread_id: str = ""
    params_hash: str = ""
    circuit_key: str = ""
    error_type: str = ""
    error_message: str = ""
    count: int = 0
    blocked: bool = False
    first_failed_at_ms: int = 0
    last_failed_at_ms: int = 0
    recovery_action: str = ""


class CircuitBreakerState(APIModel):
    circuit_key: str = ""
    tool_name: str = ""
    service_name: str = ""
    target: str = ""
    merchant_id: str = ""
    thread_id: str = ""
    params_hash: str = ""
    state: str = "closed"
    open: bool = False
    failure_count: int = 0
    reason: str = ""
    opened_at_ms: int = 0
    open_until_ms: int = 0
    half_open_probe_in_flight: bool = False
    last_probe_at_ms: int = 0
    recovery_action: str = ""
    recommended_action: str = ""
    fallback_tools: List[str] = Field(default_factory=list)


class ToolCallRequest(APIModel):
    id: str = ""
    name: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolCallExecutionResult(APIModel):
    id: str = ""
    name: str = ""
    status: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""
    params_hash: str = ""
    error_type: str = ""
    error_code: str = ""
    error_message: str = ""
    timeout_type: str = ""
    duration_ms: int = 0
    attempts: int = 0
    cache_hit: bool = False
    cache_key: str = ""
    rate_limited: bool = False
    target: str = ""
    service_name: str = ""
    circuit_key: str = ""
    retryable: bool = False
    recommended_action: str = ""
    fallback_tools: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    contract: Dict[str, Any] = Field(default_factory=dict)
    result_hash: str = ""
    tool_message: Dict[str, Any] = Field(default_factory=dict)
    runtime_events: List[Dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.error_code and self.error_type:
            self.error_code = self.error_type
        if self.status in {"failed", "blocked", "error"} and not self.details:
            self.details = {
                key: value
                for key, value in {
                    "toolCallId": self.id,
                    "toolName": self.name,
                    "serviceName": self.service_name,
                    "target": self.target,
                    "paramsHash": self.params_hash,
                    "idempotencyKey": self.idempotency_key,
                    "circuitKey": self.circuit_key,
                    "timeoutType": self.timeout_type,
                    "attempts": self.attempts,
                    "durationMs": self.duration_ms,
                    "cacheHit": self.cache_hit,
                    "rateLimited": self.rate_limited,
                }.items()
                if value not in ("", 0, False, None, [], {})
            }
        if self.status in {"failed", "blocked", "error"} and not self.tool_message:
            self.tool_message = {
                "toolCallId": self.id,
                "toolName": self.name,
                "status": self.status,
                "errorType": self.error_type,
                "errorCode": self.error_code,
                "message": self.error_message,
                "retryable": self.retryable,
                "recommendedAction": self.recommended_action,
                "fallbackTools": list(self.fallback_tools),
                "details": dict(self.details),
            }
        elif self.tool_message:
            self.tool_message.setdefault("toolCallId", self.id)
            self.tool_message.setdefault("toolName", self.name)
            self.tool_message.setdefault("status", self.status)
            if self.error_type:
                self.tool_message.setdefault("errorType", self.error_type)
            if self.error_code:
                self.tool_message.setdefault("errorCode", self.error_code)
            if self.details:
                self.tool_message.setdefault("details", dict(self.details))


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
    visible_columns: List[str] = Field(default_factory=list)
    internal_only_columns: List[str] = Field(default_factory=list)
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
    effective_user_id: str = ""
    authorized_region: str = ""
    authorized_store_ids: List[str] = Field(default_factory=list)
    region_filter_column: str = ""
    store_filter_column: str = ""
    access_role: str = "merchant_analyst"
    row_scope_policy: Dict[str, Any] = Field(default_factory=dict)
    column_access_policy: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    column_display_policy: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    masked_columns: Dict[str, str] = Field(default_factory=dict)
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
    failed_task_ids: List[str] = Field(default_factory=list)
    timed_out_task_ids: List[str] = Field(default_factory=list)
    resumed_task_ids: List[str] = Field(default_factory=list)
    blocked_task_ids: List[str] = Field(default_factory=list)
    max_concurrency: int = 0
    timeout_seconds: int = 0
    duration_ms: int = 0
    runtime_events: List[Dict[str, Any]] = Field(default_factory=list)


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
    effective_user_id: str = ""
    authorized_region: str = ""
    authorized_store_ids: List[str] = Field(default_factory=list)
    access_role: str = "merchant_analyst"
    question: str = ""
    upstream_entity_sets: List[EntitySet] = Field(default_factory=list)
    upstream_rows: List[Dict[str, Any]] = Field(default_factory=list)
    sub_agent_run_id: str = ""
    checkpoint_path: str = ""
    workspace_path: str = ""
    context_package: Dict[str, Any] = Field(default_factory=dict)
    cancel_event: Any = Field(default=None, exclude=True)


class EvidenceGap(APIModel):
    code: str = ""
    gap_code: str = ""
    task_id: str = ""
    source_node_id: str = ""
    evidence: str = ""
    reason: str = ""
    severity: str = ""
    disclosure_required: bool = False
    source: str = ""
    answer_instruction: str = ""
    suggested_action: str = ""
    missing_metric: str = ""
    missing_dimension: str = ""
    missing_time_range: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.gap_code and self.code:
            self.gap_code = self.code
        if not self.source_node_id and self.task_id:
            self.source_node_id = self.task_id
        if not self.suggested_action and self.answer_instruction:
            self.suggested_action = self.answer_instruction
        if not self.details:
            self.details = {
                key: value
                for key, value in {
                    "gapCode": self.gap_code or self.code,
                    "taskId": self.task_id,
                    "sourceNodeId": self.source_node_id,
                    "evidence": self.evidence,
                    "source": self.source,
                    "severity": self.severity,
                }.items()
                if value not in ("", None, [], {})
            }


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


class VerifiedAnswerContext(APIModel):
    question: str = ""
    business_context: Dict[str, Any] = Field(default_factory=dict)
    tables: List[str] = Field(default_factory=list)
    row_count: int = 0
    data_rows: List[Dict[str, Any]] = Field(default_factory=list)
    data_sections: List[Dict[str, Any]] = Field(default_factory=list)
    metric_disclosures: List[Dict[str, Any]] = Field(default_factory=list)
    lightweight_metric_disclosures: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    degraded_reasons: List[Dict[str, Any]] = Field(default_factory=list)
    rule_evidence: Any = ""
    verified_passed: bool = False
    partial_answer_reason: str = ""

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "businessContext": self.business_context,
            "tables": self.tables,
            "rowCount": self.row_count,
            "dataRows": self.data_rows,
            "dataSections": self.data_sections,
            "metricDisclosures": self.metric_disclosures,
            "lightweightMetricDisclosures": self.lightweight_metric_disclosures,
            "evidenceGaps": self.evidence_gaps,
            "degradedReasons": self.degraded_reasons,
            "ruleEvidence": self.rule_evidence,
            "verifiedPassed": self.verified_passed,
            "partialAnswerReason": self.partial_answer_reason,
        }


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
    params: List[Any] = Field(default_factory=list)
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
    runtime_events: List[Dict[str, Any]] = Field(default_factory=list)

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
    sub_agent_run_id: str = ""
    sub_agent_checkpoint_path: str = ""
    sub_agent_workspace: str = ""
    sub_agent_context: Dict[str, Any] = Field(default_factory=dict)
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
    file_tool_results: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceCheckResult(APIModel):
    passed: bool = False
    summary: str = ""
    gaps: List[str] = Field(default_factory=list)


class SkillLifecycleRecord(APIModel):
    record_id: str = ""
    skill_name: str = ""
    stage: str = ""
    status: str = ""
    matched_by: str = ""
    requires_confirmation: bool = False
    confirmed: bool = False
    isolated_run_id: str = ""
    workspace_path: str = ""
    checkpoint_path: str = ""
    progress: List[str] = Field(default_factory=list)
    reuse_candidate: bool = False
    context_hash: str = ""
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillMatchState(APIModel):
    skill_name: str = ""
    status: str = "matched"
    matched_by: str = ""
    match_source: str = ""
    confidence: float = 0.0
    reason: str = ""
    candidate_skills: List[str] = Field(default_factory=list)
    fallback_skill: str = ""
    requires_confirmation: bool = False
    confirmed: bool = False
    headers: List[Dict[str, Any]] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)


class SkillDraft(APIModel):
    draft_id: str = ""
    status: str = "pending_review"
    callable: bool = False
    source_thread_id: str = ""
    source_run_id: str = ""
    source_qa_id: str = ""
    merchant_id: str = ""
    title: str = ""
    description: str = ""
    applicability: List[str] = Field(default_factory=list)
    required_inputs: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    evidence_requirements: List[str] = Field(default_factory=list)
    example_questions: List[str] = Field(default_factory=list)
    source_artifacts: Dict[str, Any] = Field(default_factory=dict)
    review_note: str = ""
    reviewer: str = ""
    created_at: str = ""
    reviewed_at: str = ""
    published_skill_name: str = ""


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
    skill_lifecycle_records: List[SkillLifecycleRecord] = Field(default_factory=list)
    resumed_task_ids: List[str] = Field(default_factory=list)
    degraded_reasons: List[Dict[str, Any]] = Field(default_factory=list)


class HypothesisEvidenceRecord(APIModel):
    evidence_id: str = ""
    hypothesis_id: str = ""
    round: int = 1
    task_id: str = ""
    claim_key: str = ""
    metric_name: str = ""
    metric_formula: str = ""
    table: str = ""
    time_range: str = ""
    sql_hash: str = ""
    row_count: int = 0
    status: str = "insufficient"
    confidence: float = 0.0
    evidence_preview: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    failure_reason: str = ""
    reused_from_hypothesis_id: str = ""


class HypothesisLedgerEntry(APIModel):
    hypothesis_id: str = ""
    title: str = ""
    reason: str = ""
    status: str = "candidate"
    rank: int = 0
    evidence_score: int = 0
    semantic_score: int = 0
    confidence: float = 0.0
    query_graphs: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[HypothesisEvidenceRecord] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    insufficient_evidence_ids: List[str] = Field(default_factory=list)
    failed_evidence_ids: List[str] = Field(default_factory=list)
    evidence_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    elimination_reason: str = ""
    followup_decision: Dict[str, Any] = Field(default_factory=dict)


class HypothesisEvidenceLedger(APIModel):
    ledger_id: str = ""
    winner_id: str = ""
    survivor_ids: List[str] = Field(default_factory=list)
    pruned_ids: List[str] = Field(default_factory=list)
    entries: List[HypothesisLedgerEntry] = Field(default_factory=list)
    rounds_used: int = 0
    budget: Dict[str, Any] = Field(default_factory=dict)
    comparison_policy: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


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


class MemoryEvent(APIModel):
    event_id: str = ""
    memory_type: str = "query_event"
    memory_tier: str = "retrieval"
    memory_class: str = "interaction_event"
    question: str = ""
    answer_preview: str = ""
    topics: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    time_windows: List[int] = Field(default_factory=list)
    analysis_intent: str = ""
    is_follow_up: bool = False
    feedback_signal: str = ""
    correction_text: str = ""
    confidence: float = 0.5
    source: str = "answer_run"
    source_event_id: str = ""
    hit_count: int = 0
    last_used_at: str = ""
    decay_score: float = 1.0
    valid_until: str = ""
    retention_days: int = 0
    supersedes: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    scope: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    visibility: str = "merchant"
    allowed_roles: List[str] = Field(default_factory=list)
    approved_by: str = ""
    review_status: str = "auto"
    write_policy: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    case_payload: Dict[str, Any] = Field(default_factory=dict)
    case_summary: str = ""
    created_at: str = ""


class MemoryFact(APIModel):
    fact_id: str = ""
    memory_type: str = "business_fact"
    memory_tier: str = "core"
    memory_class: str = "semantic_fact"
    content: str = ""
    topics: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    source: str = ""
    source_event_id: str = ""
    hit_count: int = 0
    last_used_at: str = ""
    decay_score: float = 1.0
    valid_until: str = ""
    retention_days: int = 0
    supersedes: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    scope: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    visibility: str = "merchant"
    allowed_roles: List[str] = Field(default_factory=list)
    approved_by: str = ""
    review_status: str = "auto"
    write_policy: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    created_at: str = ""


class MemoryPreference(APIModel):
    preference_id: str = ""
    memory_type: str = "preference"
    memory_tier: str = "core"
    memory_class: str = "preference"
    key: str = ""
    value: str = ""
    topics: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    source: str = ""
    hit_count: int = 0
    last_used_at: str = ""
    decay_score: float = 1.0
    valid_until: str = ""
    retention_days: int = 0
    scope: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    visibility: str = "merchant"
    allowed_roles: List[str] = Field(default_factory=list)
    approved_by: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    review_status: str = "merchant_confirmed"
    write_policy: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class MemoryRetrievalCandidate(APIModel):
    memory_id: str = ""
    memory_type: str = ""
    score: float = 0.0
    reasons: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    filtered: bool = False
    filter_reason: str = ""


class MemoryInjectionTrace(APIModel):
    merchant_id: str = ""
    budget_tokens: int = 0
    budget_chars: int = 0
    candidate_count: int = 0
    injected_event_count: int = 0
    injected_preference_count: int = 0
    injected_correction_count: int = 0
    injected_fact_count: int = 0
    budget_used_tokens: int = 0
    budget_used_chars: int = 0
    truncated: bool = False
    selected_ids: List[str] = Field(default_factory=list)
    filtered_reasons: Dict[str, int] = Field(default_factory=dict)
    candidates: List[MemoryRetrievalCandidate] = Field(default_factory=list)
    candidate_ids: List[str] = Field(default_factory=list)
    past_case_count: int = 0
    core_memory_count: int = 0
    retrieval_memory_count: int = 0
    core_selected_ids: List[str] = Field(default_factory=list)


class MemoryConflictResolution(APIModel):
    conflict_id: str = ""
    winner_id: str = ""
    loser_id: str = ""
    reason: str = ""
    action: str = ""
    created_at: str = ""


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
