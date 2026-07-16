from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Iterable, Optional

from merchant_ai.models import (
    AnswerClaim,
    AnswerClaimVerification,
    ChatDataSection,
    ChatResponse,
)
from merchant_ai.services.cache import TTLCache
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.semantic_request import semantic_request_cache_key
from merchant_ai.services.time_semantics import (
    CALENDAR_ANCHOR_POLICY,
    latest_partition_window_predicate,
    partition_date_matches,
    resolve_time_range,
    resolve_time_window_contract,
    time_window_contract_payload,
)


COMPLEX_TERMS = ["明细", "详情", "列表", "记录", "对应", "关联", "拆解", "归因"]
ANALYSIS_TERMS = ["为什么", "原因", "分析", "归因", "诊断", "异常", "建议"]
DEFINITION_TERMS = ["口径", "定义", "含义", "什么意思", "是否扣", "怎么算", "计算方式"]
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TIME_DIMENSION_KEY = "time_dimension"
QUICK_RESPONSE_CACHE = TTLCache("quick_metric_response", max_entries=512, ttl_seconds=30)


def is_metric_definition_question(question: str) -> bool:
    text = str(question or "")
    return any(term in text for term in DEFINITION_TERMS)


def quick_metric_response(
    question: str,
    merchant_id: str,
    repository: Any,
    extracted_keywords: Any = None,
    semantic_metrics: Optional[list[Dict[str, Any]]] = None,
    timezone_name: str = "Asia/Shanghai",
) -> Optional[ChatResponse]:
    # Fast answers are deliberately a one-metric capability.  The caller must
    # provide contracts compiled from the published semantic layer; an empty,
    # ambiguous, or multi-metric match always falls back to QueryGraph.
    if not semantic_metrics:
        return None
    metric_phrases = list(getattr(extracted_keywords, "metric_keywords", None) or [])
    matched_metrics = resolve_metrics(question, semantic_metrics or [], metric_phrases)
    if len(matched_metrics) != 1:
        return None
    if structured_keywords_require_planner(extracted_keywords, matched_metrics) or "[用户附件上下文]" in question:
        return None
    if is_metric_definition_question(question):
        return quick_metric_definition_response(question, merchant_id, matched_metrics)
    if any(term in question for term in ANALYSIS_TERMS):
        return None
    if any(term in question for term in COMPLEX_TERMS):
        return None
    if not any(
        term in question.lower() for term in ["多少", "总", "趋势", "走势", "变化", "最近", "近", "为什么", "原因"]
    ):
        return None
    time_range = resolve_time_range(question, timezone_name)
    temporal_contract = resolve_time_window_contract(question, timezone_name)
    temporal_mode = quick_metric_temporal_mode(time_range, temporal_contract)
    if temporal_mode == "unsupported":
        return None
    original_metric = matched_metrics[0]
    metric = resolve_quick_metric_temporal_variant(original_metric, semantic_metrics, temporal_mode)
    if metric is None:
        return None
    days = time_range.days
    cache_key = semantic_request_cache_key(
        "quick_metric",
        topics=[metric.get("topic")],
        metrics=[{"metricKey": metric.get("key"), "ownerTable": metric.get("table")}],
        dimensions=[metric.get("time_column")],
        filters=[],
        time_range=time_range,
        asset_version={"semanticContract": semantic_metric_identity(metric)},
        scope={"merchantId": merchant_id, "temporalMode": temporal_mode},
    )
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    formula = metric["compiled_formula"]
    table = metric["table"]
    time_column = metric["time_column"]
    tenant_column = metric["tenant_column"]
    time_filter, time_params = quick_metric_time_filter(time_range, table, time_column, tenant_column, merchant_id)
    rows = repository.query(
        "SELECT `%s` AS `%s`, %s AS value FROM `%s` "
        "WHERE `%s`=%%s AND %s "
        "GROUP BY `%s` ORDER BY `%s`"
        % (
            time_column,
            TIME_DIMENSION_KEY,
            formula,
            table,
            tenant_column,
            time_filter,
            time_column,
            time_column,
        ),
        [merchant_id] + time_params,
    )
    if not rows:
        return None
    latest_partition = row_time_dimension(rows[-1], time_column)
    if time_range.kind == "exact_date" and not partition_date_matches(latest_partition, time_range.end_date):
        return None
    values = [float(row.get("value") or 0) for row in rows]
    # Never roll up daily aggregates in application code: SUM of daily
    # COUNT(DISTINCT), AVG, or ratios can change the governed metric meaning.
    normalized_rows = [
        {
            "metric_name": metric["label"],
            TIME_DIMENSION_KEY: row_time_dimension(row, time_column),
            "value": float(row.get("value") or 0),
        }
        for row in rows
    ]
    daily_series_value_only = (
        temporal_mode == "daily_series" and metric_aggregation_policy(metric) == "daily_value_only"
    )
    if daily_series_value_only:
        # A daily precomputed value has no governed multi-day scalar.  The
        # grouped series is authoritative and its final point is disclosed as
        # a latest-day value; never execute an interval MAX/AVG as a summary.
        total = values[-1]
    else:
        total_rows = repository.query(
            "SELECT %s AS value FROM `%s` "
            "WHERE `%s`=%%s AND %s"
            % (
                formula,
                table,
                tenant_column,
                time_filter,
            ),
            [merchant_id] + time_params,
        )
        if not total_rows or total_rows[0].get("value") is None:
            return None
        total = float(total_rows[0]["value"])
    first, last = values[0], values[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    delta = abs(last - first)
    direction_text = "整体持平" if delta == 0 else "整体%s" % direction
    peak_index = max(range(len(values)), key=values.__getitem__)
    peak = normalized_rows[peak_index]
    total_text = format_value(total, metric)
    all_zero = bool(values) and all(value == 0 for value in values)
    advice = zero_metric_advice(metric["label"]) if all_zero else metric_advice(metric["label"])
    time_label = time_range.label or time_range_label(question, days)
    freshness_sentence = ""
    if time_range.kind == "rolling" and not partition_date_matches(latest_partition, time_range.end_date):
        freshness_sentence = "数据日期截至 %s。\n\n" % latest_partition
    trend_sentence = (
        f"从每日表现看，{metric['label']}各日均为 {format_value(0, metric)}。"
        if all_zero
        else (
            f"从每日表现看，{metric['label']}由 {format_value(first, metric)} 变化到 {format_value(last, metric)}，{direction_text}；"
            f"峰值日期为 {peak[TIME_DIMENSION_KEY]}，峰值为 {format_value(peak['value'], metric)}。"
        )
    )
    summary_sentence = (
        f"{time_label}，店铺{metric['label']}最新日值为 {total_text}。"
        if daily_series_value_only
        else f"{time_label}，店铺{metric['label']}{summary_predicate(metric)} {total_text}。"
    )
    answer = f"{summary_sentence}\n\n{freshness_sentence}{trend_sentence}\n\n建议：\n- {advice[0]}\n- {advice[1]}"
    suggestions = metric_suggestions(metric["label"], days)
    traceability = {
        "sourceSummary": "Doris 快速指标查询",
        "merchantId": merchant_id,
        "timeRange": time_label,
        "timeWindowContract": time_window_contract_payload(time_range, table, time_column, tenant_column),
        "dataUpdatedAt": normalized_rows[-1][TIME_DIMENSION_KEY],
        "rowCount": len(normalized_rows),
        "sourceTables": [table],
        "evidenceStatus": "verified",
        "summarySemantics": "latest_day_value" if daily_series_value_only else "period_aggregate",
    }
    response = ChatResponse(
        id="quick_" + uuid.uuid4().hex,
        answer=answer,
        category_name=metric_category_name(metric),
        persisted=False,
        doris_tables=[table],
        suggestions=suggestions,
        thinking_steps=["识别简单指标问题", "读取指标口径", "查询 Doris 数据", "校验结果", "生成经营建议"],
        data_rows=normalized_rows,
        data_sections=[
            ChatDataSection(
                title=f"{metric['label']}趋势",
                result_role="trend_context",
                doris_tables=[table],
                data_rows=normalized_rows,
            ),
            ChatDataSection(
                title=(f"{metric['label']}最新日值" if daily_series_value_only else metric["label"]),
                result_role="summary",
                doris_tables=[table],
                data_rows=[
                    {
                        "metric_name": metric["label"],
                        "value": total,
                        **(
                            {TIME_DIMENSION_KEY: normalized_rows[-1][TIME_DIMENSION_KEY]}
                            if daily_series_value_only
                            else {}
                        ),
                    }
                ],
            ),
        ],
        merchant_experience={
            "version": "v1",
            "businessAdvice": advice,
            "suggestedQuestions": suggestions,
            "anomalyAlerts": [],
            "metricDisclosures": [semantic_metric_disclosure(metric)],
            "traceability": traceability,
            "drillDownActions": [{"label": "继续下钻", "question": suggestions[0], "actionType": "follow_up_question"}],
        },
        debug_trace={
            "quickMetricPath": True,
            "days": days,
            "timeRange": time_range.model_dump(by_alias=True),
            "timeWindowContract": time_window_contract_payload(time_range, table, time_column, tenant_column),
            "actualLatestPartition": latest_partition,
            "metric": metric["label"],
            "metricTerms": metric.get("terms") or [],
            "semanticMetric": semantic_metric_identity(metric),
            "temporalMode": temporal_mode,
            "temporalMetricRedirected": semantic_metric_ref_id(metric) != semantic_metric_ref_id(original_metric),
            "requestedSemanticMetric": semantic_metric_identity(original_metric),
            "summarySemantics": "latest_day_value" if daily_series_value_only else "period_aggregate",
        },
    )
    verification = verify_quick_metric_answer(
        question,
        metric,
        normalized_rows,
        total,
        answer,
        summary_semantics="latest_day_value" if daily_series_value_only else "period_aggregate",
    )
    if not verification.passed:
        return None
    response.debug_trace["answerClaimVerification"] = verification.model_dump(by_alias=True)
    QUICK_RESPONSE_CACHE.set(cache_key, response.model_dump(by_alias=True))
    return response


def quick_metric_temporal_mode(time_range: Any, temporal_contract: Dict[str, Any]) -> str:
    """Return the governed shape a quick metric query must produce.

    Time interpretation comes from the shared time-window tool.  The fast path
    has no comparison executor, so a second window is always returned to the
    Planner rather than silently answering only the primary window.
    """

    if bool((temporal_contract or {}).get("requiresComparison")):
        return "unsupported"
    if str(getattr(time_range, "kind", "") or "") == "exact_date":
        return "exact_day"
    return "daily_series" if str((temporal_contract or {}).get("grain") or "").lower() == "day" else "period_summary"


def resolve_quick_metric_temporal_variant(
    selected: Dict[str, Any],
    semantic_metrics: list[Dict[str, Any]],
    temporal_mode: str,
) -> Optional[Dict[str, Any]]:
    """Resolve a time-compatible member of a published metric family.

    Family membership is established only by ``temporalVariants`` and
    ``linkedVariantOf`` references.  Labels, metric names, table names, and
    formula text never infer a relationship.
    """

    family = temporal_metric_family(selected, semantic_metrics)
    compatible = [metric for metric in family if quick_metric_supports_temporal_mode(metric, temporal_mode)]
    if not compatible:
        return None
    scores = [(quick_metric_temporal_preference(metric, temporal_mode), metric) for metric in compatible]
    highest = max(score for score, _metric in scores)
    winners = [metric for score, metric in scores if score == highest]
    identities = {semantic_metric_ref_id(metric) for metric in winners}
    if len(identities) != 1:
        return None
    return winners[0]


def temporal_metric_family(
    selected: Dict[str, Any],
    semantic_metrics: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    by_ref = {semantic_metric_ref_id(metric): metric for metric in semantic_metrics}
    selected_ref = semantic_metric_ref_id(selected)
    if selected_ref not in by_ref:
        by_ref[selected_ref] = selected
    family_refs: set[str] = {selected_ref}
    pending = [selected_ref]
    while pending:
        current_ref = pending.pop()
        current = by_ref[current_ref]
        for candidate_ref, candidate in by_ref.items():
            if candidate_ref in family_refs:
                continue
            if metric_temporal_reference_targets(current, candidate) or metric_temporal_reference_targets(
                candidate, current
            ):
                family_refs.add(candidate_ref)
                pending.append(candidate_ref)
    return [by_ref[reference] for reference in family_refs]


def metric_temporal_reference_targets(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    return any(
        temporal_reference_matches_metric(reference, source, target) for reference in metric_temporal_references(source)
    )


def metric_temporal_references(metric: Dict[str, Any]) -> list[str]:
    references: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                references.append(text)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                visit(child)

    visit(metric.get("temporal_variants") or metric.get("temporalVariants") or {})
    visit(metric.get("linked_variant_of") or metric.get("linkedVariantOf") or "")
    return dedupe_texts(references)


def temporal_reference_matches_metric(
    reference: str,
    source: Dict[str, Any],
    target: Dict[str, Any],
) -> bool:
    value = str(reference or "").strip()
    if not value:
        return False
    target_ref = semantic_metric_ref_id(target)
    if value == target_ref:
        return True
    # A short metric key is scoped to its publishing topic/table.  This avoids
    # linking equal keys that happen to exist in unrelated semantic assets.
    target_key = str(target.get("key") or "").strip()
    if value != target_key:
        return False
    return str(source.get("topic") or "") == str(target.get("topic") or "") and str(source.get("table") or "") == str(
        target.get("table") or ""
    )


def quick_metric_supports_temporal_mode(metric: Dict[str, Any], temporal_mode: str) -> bool:
    policy = metric_aggregation_policy(metric)
    grains = metric_applicable_time_grains(metric)
    if policy == "daily_value_only":
        return temporal_mode in {"exact_day", "daily_series"} and "day" in grains
    if temporal_mode in {"exact_day", "daily_series"}:
        return not grains or "day" in grains
    if temporal_mode == "period_summary":
        return not grains or "period" in grains
    return False


def quick_metric_temporal_preference(metric: Dict[str, Any], temporal_mode: str) -> int:
    policy = metric_aggregation_policy(metric)
    grains = metric_applicable_time_grains(metric)
    if temporal_mode in {"exact_day", "daily_series"}:
        if policy == "daily_value_only":
            return 300
        return 200 if "day" in grains else 100
    if policy in {"ratio_of_sums", "period_rollup"}:
        return 300
    return 200 if "period" in grains else 100


def metric_aggregation_policy(metric: Dict[str, Any]) -> str:
    return str(metric.get("aggregation_policy") or metric.get("aggregationPolicy") or "").strip().lower()


def metric_applicable_time_grains(metric: Dict[str, Any]) -> set[str]:
    raw = metric.get("applicable_time_grain")
    if raw in (None, ""):
        raw = metric.get("applicableTimeGrain")
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return {str(item or "").strip().lower() for item in values if str(item or "").strip()}


def quick_metric_time_filter(
    time_range: Any,
    table: str,
    time_column: str,
    tenant_column: str,
    merchant_id: str,
) -> tuple[str, list[Any]]:
    if (
        getattr(time_range, "anchor_policy", "") == CALENDAR_ANCHOR_POLICY
        and getattr(time_range, "start_date", "")
        and getattr(time_range, "end_date", "")
    ):
        if time_range.start_date == time_range.end_date:
            return "`%s` = %%s" % time_column, [time_range.end_date]
        return "`%s` BETWEEN %%s AND %%s" % time_column, [time_range.start_date, time_range.end_date]
    predicate = latest_partition_window_predicate(
        table,
        getattr(time_range, "days", 0) or 1,
        partition_column=time_column,
        tenant_column=tenant_column,
        tenant_value_sql="%s",
    )
    return predicate, [merchant_id, merchant_id]


def row_time_dimension(row: Dict[str, Any], declared_time_column: str) -> str:
    """Read the generic projection or the exact semantic column used by test adapters."""

    value = row.get(TIME_DIMENSION_KEY)
    if value in (None, "") and declared_time_column:
        value = row.get(declared_time_column)
    return str(value or "")


def verify_quick_metric_answer(
    question: str,
    metric: Dict[str, Any],
    trend_rows: list[Dict[str, Any]],
    total: float,
    answer: str,
    summary_semantics: str = "period_aggregate",
) -> AnswerClaimVerification:
    coverage_claim = (
        "quick_metric_latest_day_coverage"
        if summary_semantics == "latest_day_value"
        else "quick_metric_period_total_coverage"
    )
    supported_claim = AnswerClaim(
        text=coverage_claim,
        numeric_values=[format_value(total, metric)],
        supported=True,
    )
    unsupported_claims: list[AnswerClaim] = []
    if not answer_contains_number(answer, total):
        unsupported_claims.append(
            AnswerClaim(
                text=coverage_claim,
                numeric_values=[format_value(total, metric)],
                supported=False,
                reasons=["missing_quick_metric_summary_value"],
            )
        )
    extra_numbers = unsupported_quick_answer_numbers(answer, question, trend_rows, total)
    if extra_numbers:
        unsupported_claims.append(
            AnswerClaim(
                text="quick_metric_extra_numbers",
                numeric_values=extra_numbers,
                supported=False,
                reasons=["unsupported_extra_value:%s" % value for value in extra_numbers],
            )
        )
    return AnswerClaimVerification(
        passed=not unsupported_claims,
        fact_count=1,
        claims=[supported_claim] + unsupported_claims,
        unsupported_claims=unsupported_claims,
    )


def unsupported_quick_answer_numbers(
    answer: str,
    question: str,
    trend_rows: list[Dict[str, Any]],
    total: float,
) -> list[str]:
    allowed = [float(total)]
    allowed.extend(float(row.get("value") or 0) for row in trend_rows[:40])
    allowed.extend(value for _, value in numeric_token_pairs(question))
    unsupported: list[str] = []
    for raw, value in numeric_token_pairs(answer):
        if any(numbers_close(value, candidate) for candidate in allowed):
            continue
        if abs(value) < 10 and not raw.endswith("%") and "." not in raw:
            continue
        unsupported.append(raw)
    return dedupe_strings(unsupported)


def dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def answer_contains_number(answer: str, expected: Any) -> bool:
    value = numeric_token_value(expected)
    return value is not None and any(numbers_close(value, candidate) for _, candidate in numeric_token_pairs(answer))


def numeric_token_pairs(text: str) -> list[tuple[str, float]]:
    scrubbed = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", str(text or ""))
    pairs: list[tuple[str, float]] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?%?", scrubbed):
        raw = match.group(0)
        value = numeric_token_value(raw)
        if value is not None:
            pairs.append((raw, value))
    return pairs


def numeric_token_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    percent = text.endswith("%")
    text = text.rstrip("%")
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if percent else number


def numbers_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.005, abs(right) * 0.000001)


def quick_metric_definition_response(
    question: str, merchant_id: str, metrics: list[Dict[str, Any]]
) -> Optional[ChatResponse]:
    if len(metrics) != 1:
        return None
    selected = metrics
    disclosures = [semantic_metric_disclosure(metric) for metric in selected]
    lines = ["当前语义层里，%s 的口径如下：" % "、".join(metric["label"] for metric in selected)]
    for metric in selected:
        lines.append(
            "- %s：公式 `%s`，来源表 `%s`。%s"
            % (
                metric["label"],
                metric["formula"],
                metric["table"],
                metric.get("description") or "暂无更细业务说明",
            )
        )
    suggestions = metric_definition_suggestions(selected)
    advice = metric_definition_advice(question, selected)
    return ChatResponse(
        id="quick_" + uuid.uuid4().hex,
        answer="\n".join(lines),
        category_name=metric_definition_category_name(selected),
        persisted=False,
        doris_tables=dedupe_texts([metric["table"] for metric in selected]),
        suggestions=suggestions,
        thinking_steps=["识别指标口径问题", "读取已发布语义指标", "生成口径说明"],
        data_rows=[],
        data_sections=[],
        merchant_experience={
            "version": "v1",
            "businessAdvice": advice,
            "suggestedQuestions": suggestions,
            "anomalyAlerts": [],
            "metricDisclosures": disclosures,
            "traceability": {
                "sourceSummary": "已发布语义指标口径",
                "merchantId": merchant_id,
                "sourceTables": dedupe_texts([metric["table"] for metric in selected]),
                "evidenceStatus": "semantic_definition",
            },
            "drillDownActions": [
                {"label": "按该口径看趋势", "question": suggestions[0], "actionType": "follow_up_question"}
            ],
        },
        debug_trace={
            "quickMetricPath": True,
            "definitionOnly": True,
            "metricTerms": [term for metric in selected for term in (metric.get("terms") or [])],
            "semanticMetrics": [semantic_metric_identity(metric) for metric in selected],
        },
    )


def metric_definition_category_name(metrics: list[Dict[str, Any]]) -> str:
    categories = dedupe_texts(metric_category_name(metric) for metric in metrics)
    return "、".join(categories) if categories else "经营指标"


def metric_definition_suggestions(metrics: list[Dict[str, Any]]) -> list[str]:
    metric = metrics[0] if metrics else {}
    label = str(metric.get("label") or metric.get("key") or "该指标").strip()
    return [
        "最近7天%s趋势" % label,
        "昨天%s是多少？" % label,
        "最近30天%s按天走势" % label,
    ]


def metric_definition_advice(question: str, metrics: list[Dict[str, Any]]) -> list[str]:
    label = str((metrics[0] if metrics else {}).get("label") or "该指标").strip()
    return ["后续分析请沿用“%s”的已发布语义口径。" % label, "把来源表、公式和单位一起展示，避免不同报表口径混用。"]


def structured_keywords_require_planner(keywords: Any, matched_metrics: Optional[list[Dict[str, Any]]] = None) -> bool:
    if keywords is None:
        return False
    if getattr(keywords, "ranking_keywords", None):
        return True
    if getattr(keywords, "unresolved_phrases", None):
        return True
    structured_metrics = list(getattr(keywords, "metric_keywords", None) or [])
    analysis_intent = str(getattr(keywords, "analysis_intent", "") or "")
    if analysis_intent in {"attribution", "ranking", "detail", "advice"}:
        return True
    if analysis_intent == "ratio" and not (
        matched_metrics is not None and len(matched_metrics) == 1 and len(structured_metrics) <= 1
    ):
        return True
    if matched_metrics is not None and len(structured_metrics) > len(matched_metrics):
        return True
    metric_candidates: Dict[str, set[str]] = {}
    for item in getattr(keywords, "mentions", None) or []:
        if str(getattr(item, "kind", "") or "") != "metric":
            continue
        metric_candidates.setdefault(str(getattr(item, "phrase", "") or ""), set()).add(
            "%s:%s"
            % (
                str(getattr(item, "canonical_key", "") or ""),
                str(getattr(item, "topic", "") or ""),
            )
        )
    if any(len(candidates) > 1 for candidates in metric_candidates.values()):
        return True
    declared_time_columns = {
        str(metric.get("time_column") or "").strip()
        for metric in (matched_metrics or [])
        if str(metric.get("time_column") or "").strip()
    }
    return any(
        str(getattr(item, "kind", "") or "") == "dimension"
        and str(getattr(item, "canonical_key", "") or "") not in declared_time_columns
        for item in (getattr(keywords, "mentions", None) or [])
    )


def published_semantic_quick_metrics(
    topic_assets: Any, preferred_topics: Optional[Iterable[str]] = None
) -> list[Dict[str, Any]]:
    """Compile safe fast-path contracts exclusively from published topic assets."""
    preferred = [str(item) for item in (preferred_topics or []) if str(item or "").strip()]
    all_topics = list(topic_assets.all_topic_names())
    topics = preferred + [topic for topic in all_topics if topic not in preferred]
    contracts: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for topic in topics:
        for manifest in topic_assets.load_manifest(topic):
            if not bool(manifest.get("supportsMetrics")):
                continue
            table = str(manifest.get("tableName") or "").strip()
            time_column = str(manifest.get("timeColumn") or "").strip()
            tenant_column = str(manifest.get("merchantFilterColumn") or "").strip()
            if not all(SAFE_IDENTIFIER.fullmatch(item or "") for item in [table, time_column, tenant_column]):
                continue
            for metric in topic_assets.load_table_metrics(topic, table):
                contract = compile_semantic_quick_metric(metric, topic, table, time_column, tenant_column)
                if not contract:
                    continue
                identity = (contract["table"], contract["key"], contract["formula"])
                if identity in seen:
                    continue
                seen.add(identity)
                contracts.append(contract)
    return contracts


def compile_semantic_quick_metric(
    metric: Dict[str, Any],
    topic: str,
    table: str,
    time_column: str,
    tenant_column: str,
) -> Optional[Dict[str, Any]]:
    formula = str(metric.get("formula") or metric.get("metricFormula") or "").strip()
    source_columns = [str(item or "").strip() for item in metric.get("sourceColumns") or [] if str(item or "").strip()]
    if not source_columns or not all(SAFE_IDENTIFIER.fullmatch(column) for column in source_columns):
        return None
    compiled_formula = compile_metric_formula(formula, set(source_columns))
    if not compiled_formula:
        return None
    metric_key = str(metric.get("metricKey") or metric.get("canonicalMetricKey") or "").strip()
    if not metric_key:
        return None
    label = str(metric.get("displayName") or metric.get("businessName") or metric_key).strip()
    aliases = [label, metric_key, *source_columns, *(metric.get("aliases") or [])]
    terms = list(dict.fromkeys(str(item or "").strip() for item in aliases if str(item or "").strip()))
    if not terms:
        return None
    return {
        "key": metric_key,
        "label": label,
        "unit": str(metric.get("unit") or "").strip(),
        "formula": formula,
        "compiled_formula": compiled_formula,
        "source_columns": source_columns,
        "terms": terms,
        "table": table,
        "time_column": time_column,
        "tenant_column": tenant_column,
        "topic": topic,
        "description": str(metric.get("description") or "").strip(),
        "evidence": str(metric.get("evidence") or "").strip(),
        "natural_name": str(metric.get("naturalName") or "").strip(),
        "metric_grain": str(metric.get("metricGrain") or "").strip(),
        "metric_intent": str(metric.get("metricIntent") or "").strip(),
        "selection_guidance": str(metric.get("selectionGuidance") or "").strip(),
        "aggregation_policy": str(metric.get("aggregationPolicy") or "").strip(),
        "applicable_time_grain": metric.get("applicableTimeGrain"),
        "temporal_variants": metric.get("temporalVariants") or {},
        "linked_variant_of": str(metric.get("linkedVariantOf") or "").strip(),
    }


def semantic_metric_identity(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic": str(metric.get("topic") or ""),
        "category": metric_topic_category(metric),
        "table": str(metric.get("table") or ""),
        "metricKey": str(metric.get("key") or ""),
        "formula": str(metric.get("formula") or ""),
        "aggregationPolicy": metric_aggregation_policy(metric),
        "applicableTimeGrain": sorted(metric_applicable_time_grains(metric)),
        "temporalVariants": metric.get("temporal_variants") or metric.get("temporalVariants") or {},
        "linkedVariantOf": str(metric.get("linked_variant_of") or metric.get("linkedVariantOf") or ""),
        "semanticRefId": semantic_metric_ref_id(metric),
        "governanceStatus": "published",
    }


def metric_topic_category(metric: Dict[str, Any]) -> str:
    return str(metric.get("topic") or "")


def semantic_metric_disclosure(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metricKey": metric["key"],
        "displayName": metric["label"],
        "formula": metric["formula"],
        "description": metric.get("description") or "来自已发布语义资产",
        "semanticRef": semantic_metric_ref_id(metric),
        "aggregationPolicy": metric_aggregation_policy(metric),
        "applicableTimeGrain": sorted(metric_applicable_time_grains(metric)),
        "governanceStatus": "published",
    }


def semantic_metric_ref_id(metric: Dict[str, Any]) -> str:
    return "semantic:%s:%s:metric:%s" % (metric["topic"], metric["table"], metric["key"])


def dedupe_texts(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def resolve_metrics(
    question: str,
    semantic_metrics: list[Dict[str, Any]],
    metric_phrases: Optional[list[str]] = None,
) -> list[Dict[str, Any]]:
    text = question.lower()
    phrases = [str(item or "").strip() for item in (metric_phrases or []) if str(item or "").strip()]
    if not phrases:
        matched_terms = {
            term for metric in semantic_metrics for term in metric.get("terms") or [] if term and term.lower() in text
        }
        phrases = [
            term
            for term in matched_terms
            if not any(term != other and term.lower() in other.lower() for other in matched_terms)
        ]
    identities: set[tuple[str, str]] = set()
    result: list[Dict[str, Any]] = []
    for phrase in phrases:
        scored = [(semantic_phrase_score(metric, phrase), metric) for metric in semantic_metrics]
        if unresolved_cross_table_metric_candidates(scored):
            return []
        top_score = max([score for score, _metric in scored] or [0])
        winners = [metric for score, metric in scored if score == top_score and score > 0]
        winner_identities = {(metric["table"], metric["key"], metric["formula"]) for metric in winners}
        if len(winner_identities) != 1:
            return []
        metric = winners[0]
        identity = (metric["table"], metric["key"])
        if identity in identities:
            continue
        identities.add(identity)
        result.append(metric)
    return result


def unresolved_cross_table_metric_candidates(scored: list[tuple[int, Dict[str, Any]]]) -> bool:
    candidates = [(score, metric) for score, metric in scored if score >= 100]
    if len({(metric["table"], metric["key"]) for _score, metric in candidates}) < 2:
        return False
    top_score = max(score for score, _metric in candidates)
    top_tables = {metric["table"] for score, metric in candidates if score == top_score}
    return len(top_tables) > 1


def semantic_phrase_score(metric: Dict[str, Any], phrase: str) -> int:
    normalized = normalize_question(phrase)
    if not normalized:
        return 0
    best = 0
    for term in metric.get("terms") or []:
        candidate = normalize_question(term)
        if not candidate:
            continue
        if candidate == normalized:
            best = max(best, 1000 + len(candidate))
        elif candidate in normalized:
            best = max(best, 500 + len(candidate))
        elif normalized in candidate:
            best = max(best, 100 + len(normalized))
    return best


def extract_days(question: str) -> int:
    if "昨天" in question or "昨日" in question:
        return 1
    if "今天" in question or "今日" in question:
        return 1
    match = re.search(r"(?:最近|近)?\s*(\d{1,3})\s*天", question)
    return max(1, min(int(match.group(1)), 180)) if match else 7


def time_range_label(question: str, days: int) -> str:
    if "昨天" in question or "昨日" in question:
        return "昨天"
    if "今天" in question or "今日" in question:
        return "今天"
    return "最近%d天" % days


def metric_category_name(metric: Dict[str, Any]) -> str:
    return str(metric.get("topic") or "").strip() or "经营指标"


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", str(question or "").lower()).replace("？", "?")


def fresh_cached_response(payload: Dict[str, Any]) -> ChatResponse:
    response = ChatResponse.model_validate(payload)
    response.id = "quick_" + uuid.uuid4().hex
    response.persisted = False
    response.debug_trace = dict(response.debug_trace or {})
    response.debug_trace["quickMetricCacheHit"] = True
    return response


def format_value(value: float, metric: Dict[str, Any]) -> str:
    if metric["unit"] == "%":
        return f"{value * 100:.2f}%"
    if metric["unit"] == "元":
        return f"¥{value:,.2f}"
    if not metric["unit"] and float(value).is_integer() is False:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,.0f}{metric['unit']}"


def summary_predicate(metric: Dict[str, Any]) -> str:
    label = str(metric.get("label") or "")
    key = str(metric.get("key") or "").lower()
    formula = str(metric.get("formula") or "").strip().upper()
    unit = str(metric.get("unit") or "")
    if unit == "%" or "rate" in key or "率" in label or "比例" in label:
        return "为"
    if formula.startswith("AVG(") or key.startswith("avg_") or "平均" in label or "score" in key or "duration" in key:
        return "平均为"
    return "合计为"


def metric_advice(label: str) -> list[str]:
    return [f"复盘{label}峰值日期对应的主要维度。", f"将{label}与语义层中相关指标联动观察，确认变化来源。"]


def zero_metric_advice(label: str) -> list[str]:
    return [f"当前{label}为 0，先确认是否符合预期。", "后续可继续观察该指标是否出现新增波动。"]


def metric_suggestions(label: str, days: int) -> list[str]:
    return [
        f"{label}波动最大的日期有哪些？",
        f"最近{days}天{label}按维度拆解",
        f"{label}异常原因是什么？",
    ]
