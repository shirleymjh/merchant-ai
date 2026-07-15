from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set

from merchant_ai.models import EvidenceGap, GraphValidationGap, QueryPlan


REQUIRED_MEMORY_CONSTRAINT_TYPES = {"metric_correction", "business_correction"}
CONSTRAINT_APPROVED_STATUSES = {"", "active", "approved", "reviewed", "published", "indexed"}
STOP_TERMS = {
    "最近",
    "这些",
    "这个",
    "那个",
    "以后",
    "不是",
    "应该",
    "当前",
    "问题",
    "多少",
    "怎么",
    "看看",
    "分析",
}
FOCUS_TERMS = {"售后", "风险", "口径", "偏好", "关注", "纠正", "默认", "习惯"}


def build_memory_constraints(memory_injection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn selected memory payloads into auditable, semantic-layer-bounded hints."""

    if not isinstance(memory_injection, dict):
        return []
    constraints: List[Dict[str, Any]] = []
    core_memory = memory_injection.get("coreMemory") or {}
    if isinstance(core_memory, dict):
        for item in core_memory.get("corePreferences") or []:
            constraint = constraint_from_memory_payload(item, "business_preference")
            if constraint:
                constraint["source"] = "coreMemory"
                constraints.append(constraint)
        for item in core_memory.get("coreFacts") or []:
            constraint = constraint_from_memory_payload(item, "business_fact")
            if constraint:
                constraint["source"] = "coreMemory"
                constraints.append(constraint)
        for item in core_memory.get("coreCorrections") or []:
            constraint = constraint_from_memory_payload(item, "metric_correction")
            if constraint:
                constraint["source"] = "coreMemory"
                constraints.append(constraint)
    for item in memory_injection.get("relevantCorrections") or []:
        constraint = constraint_from_memory_payload(item, "metric_correction")
        if constraint:
            constraints.append(constraint)
    for item in memory_injection.get("relevantMetricDisputes") or []:
        constraint = constraint_from_memory_payload(item, "metric_dispute")
        if constraint:
            constraints.append(constraint)
    for item in memory_injection.get("relevantFacts") or []:
        constraint = constraint_from_memory_payload(item, "business_fact")
        if constraint:
            constraints.append(constraint)
    for item in memory_injection.get("relevantPreferences") or []:
        constraint = constraint_from_memory_payload(item, "business_preference")
        if constraint:
            constraints.append(constraint)
    recent_focus = memory_injection.get("recentFocus") or {}
    if isinstance(recent_focus, dict) and recent_focus.get("summary"):
        constraints.append(
            {
                "id": "recent_focus",
                "type": "recent_focus",
                "enforcement": "advisory",
                "instruction": str(recent_focus.get("summary") or "")[:400],
                "targetMetrics": [
                    str((item or {}).get("metric") or "")
                    for item in recent_focus.get("topMetrics") or []
                    if isinstance(item, dict) and (item or {}).get("metric")
                ][:8],
                "topics": [
                    str((item or {}).get("topic") or "")
                    for item in recent_focus.get("topTopics") or []
                    if isinstance(item, dict) and (item or {}).get("topic")
                ][:8],
                "confidence": 0.5,
                "source": "recentFocus",
            }
        )
    return dedupe_constraints(constraints)


def constraint_from_memory_payload(payload: Any, fallback_type: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    status = memory_payload_status(payload)
    if status not in CONSTRAINT_APPROVED_STATUSES:
        return {}
    memory_type = str(payload.get("memoryType") or fallback_type)
    constraint_type = memory_constraint_type(memory_type, fallback_type)
    metrics = unique_strings(payload.get("metrics") or [])
    topics = unique_strings(payload.get("topics") or [])
    instruction = memory_instruction_text(payload)
    if not instruction and not metrics and not topics:
        return {}
    confidence = safe_float(payload.get("confidence"), 0.5)
    enforcement = memory_constraint_enforcement(constraint_type, confidence, metrics)
    return {
        "id": str(payload.get("id") or payload.get("memoryId") or ""),
        "type": constraint_type,
        "sourceMemoryType": memory_type,
        "enforcement": enforcement,
        "instruction": instruction[:600],
        "targetMetrics": metrics[:12],
        "topics": topics[:8],
        "timeWindows": unique_ints(payload.get("timeWindows") or [])[:6],
        "confidence": confidence,
        "source": str(payload.get("source") or "memory"),
        "status": status or "active",
        "scope": payload.get("scope") if isinstance(payload.get("scope"), dict) else {},
        "approvedBy": str(payload.get("approvedBy") or ""),
        "evidenceRefs": [str(item) for item in payload.get("evidenceRefs") or [] if item][:12],
        "hitReasons": [str(item) for item in payload.get("hitReasons") or [] if item][:8],
        "governanceInstruction": str(payload.get("governanceInstruction") or ""),
    }


def memory_payload_status(payload: Dict[str, Any]) -> str:
    value = str(payload.get("status") or payload.get("governanceStatus") or "").strip()
    return value.lower() if value else "active"


def memory_constraint_type(memory_type: str, fallback_type: str) -> str:
    normalized = str(memory_type or fallback_type)
    if normalized == "metric_dispute":
        return "metric_dispute"
    if normalized == "correction":
        return "metric_correction"
    if normalized in {"metric_habit", "time_window_habit", "user_preference", "business_focus", "preference"}:
        return "business_preference"
    if normalized in {"fact", "business_fact"}:
        return "business_fact"
    return fallback_type


def memory_constraint_enforcement(constraint_type: str, confidence: float, metrics: Sequence[str]) -> str:
    if constraint_type == "metric_dispute":
        return "clarify_or_disclose"
    if constraint_type in REQUIRED_MEMORY_CONSTRAINT_TYPES and confidence >= 0.7 and metrics:
        return "required"
    return "advisory"


def memory_instruction_text(payload: Dict[str, Any]) -> str:
    parts = []
    for key in ["correctionText", "content", "value", "question", "answerPreview", "governanceInstruction"]:
        value = str(payload.get(key) or "").strip()
        if value:
            parts.append(value)
    return "；".join(unique_strings(parts))


def memory_constraint_validation_gaps(
    question: str,
    plan: QueryPlan,
    constraints: Iterable[Dict[str, Any]],
    supported_metrics: Iterable[str] | None = None,
) -> List[GraphValidationGap]:
    gaps: List[GraphValidationGap] = []
    plan_metrics = plan_metric_tokens(plan)
    supported_metric_set = {normalize_token(metric) for metric in supported_metrics or [] if normalize_token(metric)}
    for constraint in constraints or []:
        if not memory_constraint_is_required(constraint):
            continue
        if not memory_constraint_applies(question, plan, constraint, plan_metrics):
            continue
        planned_targets = [
            metric
            for metric in target_metrics(constraint)
            if metric in plan_metrics or normalize_token(metric) in plan_metrics
        ]
        unsupported = [
            metric
            for metric in target_metrics(constraint)
            if supported_metric_set and normalize_token(metric) not in supported_metric_set
            and metric not in planned_targets
            and normalize_token(metric) not in planned_targets
        ]
        if unsupported:
            gaps.append(
                GraphValidationGap(
                    code="MEMORY_CONSTRAINT_ASSET_MISSING",
                    evidence=",".join(unsupported),
                    reason="长期记忆要求的指标不在当前 PlanningAssetPack 中；必须先召回正式 semanticCatalog 资产，不能用 memory 改写语义层。sourceMemoryId=%s"
                    % (constraint.get("id", "")),
                )
            )
            continue
        missing = [metric for metric in target_metrics(constraint) if metric not in plan_metrics and normalize_token(metric) not in plan_metrics]
        if not missing:
            continue
        gaps.append(
            GraphValidationGap(
                code="MEMORY_CONSTRAINT_UNAPPLIED",
                evidence=",".join(missing),
                reason="长期记忆约束未落到 QueryGraph；只允许从 semanticCatalog 选择已有指标，不能用 memory 改写语义层。sourceMemoryId=%s instruction=%s"
                % (constraint.get("id", ""), str(constraint.get("instruction") or "")[:180]),
            )
        )
    return gaps


def memory_constraint_evidence_gaps(question: str, plan: QueryPlan, constraints: Iterable[Dict[str, Any]]) -> List[EvidenceGap]:
    gaps: List[EvidenceGap] = []
    plan_metrics = plan_metric_tokens(plan)
    for constraint in constraints or []:
        if str(constraint.get("type") or "") == "metric_dispute" and memory_dispute_applies(question, plan, constraint, plan_metrics):
            gaps.append(
                EvidenceGap(
                    code="MEMORY_METRIC_DISPUTE_REQUIRES_CLARIFICATION",
                    evidence=",".join(target_metrics(constraint)) or str(constraint.get("id") or ""),
                    reason=str(constraint.get("governanceInstruction") or constraint.get("instruction") or "记忆中存在指标口径争议，不能覆盖语义层标准定义"),
                    severity="warning",
                    disclosure_required=True,
                    source="memory",
                    answer_instruction="说明该长期记忆只是口径争议信号，当前仍以语义层/指标中心定义为准；必要时请用户确认口径。",
                )
            )
            continue
        if not memory_constraint_is_required(constraint):
            continue
        if not memory_constraint_applies(question, plan, constraint, plan_metrics):
            continue
        missing = [metric for metric in target_metrics(constraint) if metric not in plan_metrics]
        if missing:
            gaps.append(
                EvidenceGap(
                    code="MEMORY_CONSTRAINT_UNAPPLIED",
                    evidence=",".join(missing),
                    reason="长期记忆约束未被当前 QueryGraph/证据覆盖：%s" % str(constraint.get("instruction") or "")[:220],
                    source="memory",
                    answer_instruction="不要声称已遵守该历史偏好或纠错；需要修复 QueryGraph 或披露未应用原因。",
                )
            )
    return gaps


def memory_constraint_is_required(constraint: Dict[str, Any]) -> bool:
    return str(constraint.get("enforcement") or "") == "required" and bool(target_metrics(constraint))


def memory_dispute_applies(question: str, plan: QueryPlan, constraint: Dict[str, Any], plan_metrics: Set[str]) -> bool:
    metrics = target_metrics(constraint)
    if metrics and any(metric in plan_metrics for metric in metrics):
        return True
    return bool(significant_term_overlap(question, str(constraint.get("instruction") or "")))


def memory_constraint_applies(question: str, plan: QueryPlan, constraint: Dict[str, Any], plan_metrics: Set[str]) -> bool:
    metrics = target_metrics(constraint)
    normalized_question = normalize_token(question)
    if any(metric and normalize_token(metric) in normalized_question for metric in metrics):
        return True
    if any(metric in plan_metrics for metric in metrics):
        return True
    constraint_topics = {normalize_token(topic) for topic in constraint.get("topics") or [] if normalize_token(topic)}
    if constraint_topics & plan_topic_tokens(plan):
        return True
    overlap = significant_term_overlap(question, str(constraint.get("instruction") or ""))
    if overlap & FOCUS_TERMS or len(overlap) >= 2:
        return True
    # An empty/incomplete graph is not evidence that every merchant memory is
    # relevant. Applying all required memories here creates cross-domain
    # pollution and can prevent a valid graph from ever being planned.
    return False


def plan_topic_tokens(plan: QueryPlan) -> Set[str]:
    topics: Set[str] = set()
    for intent in getattr(plan, "intents", []) or []:
        category = getattr(intent, "category", "")
        value = getattr(category, "value", category)
        normalized = normalize_token(value)
        if normalized:
            topics.add(normalized)
    understanding = getattr(plan, "question_understanding", {}) or {}
    if isinstance(understanding, dict):
        for key in ["topics", "candidateTopics", "candidate_topics", "primaryTopic", "primary_topic"]:
            values = understanding.get(key) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if isinstance(value, dict):
                    value = value.get("topic") or value.get("name") or value.get("key")
                normalized = normalize_token(value)
                if normalized:
                    topics.add(normalized)
    return topics


def plan_metric_tokens(plan: QueryPlan) -> Set[str]:
    tokens: Set[str] = set()
    for intent in getattr(plan, "intents", []) or []:
        for value in [
            getattr(intent, "metric_name", ""),
            getattr(intent, "metric_column", ""),
            getattr(intent, "metric_formula", ""),
        ]:
            add_token(tokens, value)
        for value in getattr(intent, "required_evidence", []) or []:
            add_token(tokens, value)
        resolution = getattr(intent, "metric_resolution", {}) or {}
        if isinstance(resolution, dict):
            for key in ["metricKey", "metric_key", "requestedMetricRef", "requested_metric_ref", "field", "column"]:
                add_token(tokens, resolution.get(key))
            for key in ["sourceColumns", "source_columns"]:
                for value in resolution.get(key) or []:
                    add_token(tokens, value)
        for spec in getattr(intent, "metric_specs", []) or []:
            if isinstance(spec, dict):
                for key in ["metricKey", "metric_key", "key", "field", "column", "sourcePhrase"]:
                    add_token(tokens, spec.get(key))
    understanding = getattr(plan, "question_understanding", {}) or {}
    if isinstance(understanding, dict):
        collect_understanding_metric_tokens(tokens, understanding.get("rankingObjective") or understanding.get("ranking_objective"))
        collect_understanding_metric_tokens(tokens, understanding.get("requestedMeasures") or understanding.get("requested_measures"))
        collect_understanding_metric_tokens(
            tokens,
            understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents"),
        )
    for evidence in getattr(plan, "final_required_evidence", []) or []:
        add_token(tokens, evidence)
    for contract in getattr(plan, "evidence_contracts", []) or []:
        collect_understanding_metric_tokens(tokens, contract)
    return tokens


def collect_understanding_metric_tokens(tokens: Set[str], value: Any) -> None:
    if isinstance(value, dict):
        for key in ["metricRef", "metric_ref", "metricKey", "metric_key", "semanticLabel", "semantic_label", "sourcePhrase"]:
            add_token(tokens, value.get(key))
        for child in value.values():
            collect_understanding_metric_tokens(tokens, child)
    elif isinstance(value, list):
        for item in value:
            collect_understanding_metric_tokens(tokens, item)
    else:
        add_token(tokens, value)


def add_token(tokens: Set[str], value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    tokens.add(text)
    normalized = normalize_token(text)
    if normalized:
        tokens.add(normalized)


def target_metrics(constraint: Dict[str, Any]) -> List[str]:
    metrics: List[str] = []
    for metric in constraint.get("targetMetrics") or constraint.get("metrics") or []:
        text = str(metric or "").strip()
        if not text:
            continue
        metrics.append(text)
        normalized = normalize_token(text)
        if normalized and normalized != text:
            metrics.append(normalized)
    return unique_strings(metrics)


def analysis_intent(plan: QueryPlan) -> str:
    understanding = getattr(plan, "question_understanding", {}) or {}
    if isinstance(understanding, dict):
        return str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "").lower()
    return ""


def significant_term_overlap(left: str, right: str) -> Set[str]:
    return significant_terms(left) & significant_terms(right)


def significant_terms(text: str) -> Set[str]:
    terms: Set[str] = set()
    value = str(text or "")
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", value):
        if raw in STOP_TERMS:
            continue
        terms.add(raw)
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", raw):
            for size in [2, 3, 4]:
                for index in range(0, max(0, len(raw) - size + 1)):
                    part = raw[index : index + size]
                    if part not in STOP_TERMS:
                        terms.add(part)
    return terms


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def dedupe_constraints(constraints: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for constraint in constraints:
        if not constraint:
            continue
        key = "%s|%s|%s" % (
            constraint.get("id", ""),
            constraint.get("type", ""),
            ",".join(target_metrics(constraint)),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(constraint)
    return result[:24]


def unique_strings(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for item in items or []:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def unique_ints(items: Iterable[Any]) -> List[int]:
    result: List[int] = []
    seen: Set[int] = set()
    for item in items or []:
        try:
            value = int(item)
        except Exception:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
