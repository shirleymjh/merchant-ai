from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Iterable, Optional

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    ChatDataSection,
    ChatResponse,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier
from merchant_ai.services.cache import TTLCache
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.semantic_request import semantic_request_cache_key
from merchant_ai.services.time_semantics import partition_date_matches, resolve_time_range


COMPLEX_TERMS = ["明细", "详情", "列表", "记录", "对应", "关联", "拆解", "归因"]
ANALYSIS_TERMS = ["为什么", "原因", "分析", "归因", "诊断", "异常", "建议"]
DEFINITION_TERMS = ["口径", "定义", "含义", "什么意思", "是否扣", "怎么算", "计算方式"]
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
QUICK_RESPONSE_CACHE = TTLCache("quick_metric_response", max_entries=512, ttl_seconds=30)


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
    if any(term in question for term in DEFINITION_TERMS):
        return quick_metric_definition_response(question, merchant_id, matched_metrics)
    if any(term in question for term in ANALYSIS_TERMS):
        return None
    metric = matched_metrics[0]
    if any(term in question for term in COMPLEX_TERMS):
        return None
    if not any(term in question.lower() for term in ["多少", "总", "趋势", "走势", "变化", "最近", "近", "为什么", "原因"]):
        return None
    time_range = resolve_time_range(question, timezone_name)
    days = time_range.days
    cache_key = semantic_request_cache_key(
        "quick_metric",
        topics=[metric.get("topic")],
        metrics=[{"metricKey": metric.get("key"), "ownerTable": metric.get("table")}],
        dimensions=[metric.get("time_column")],
        filters=[],
        time_range=time_range,
        asset_version={"semanticContract": semantic_metric_identity(metric)},
        scope={"merchantId": merchant_id},
    )
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    formula = metric["compiled_formula"]
    table = metric["table"]
    time_column = metric["time_column"]
    tenant_column = metric["tenant_column"]
    rows = repository.query(
        "SELECT `%s` AS pt, %s AS value FROM `%s` "
        "WHERE `%s`=%%s AND `%s` >= DATE_SUB((SELECT MAX(`%s`) FROM `%s` WHERE `%s`=%%s), INTERVAL %d DAY) "
        "GROUP BY `%s` ORDER BY `%s`"
        % (
            time_column,
            formula,
            table,
            tenant_column,
            time_column,
            time_column,
            table,
            tenant_column,
            max(0, days - 1),
            time_column,
            time_column,
        ),
        [merchant_id, merchant_id],
    )
    if not rows:
        return None
    latest_partition = str(rows[-1].get("pt") or "")
    if time_range.kind == "exact_date" and not partition_date_matches(latest_partition, time_range.end_date):
        return None
    total_rows = repository.query(
        "SELECT %s AS value FROM `%s` "
        "WHERE `%s`=%%s AND `%s` >= DATE_SUB((SELECT MAX(`%s`) FROM `%s` WHERE `%s`=%%s), INTERVAL %d DAY)"
        % (
            formula,
            table,
            tenant_column,
            time_column,
            time_column,
            table,
            tenant_column,
            max(0, days - 1),
        ),
        [merchant_id, merchant_id],
    )
    if not total_rows or total_rows[0].get("value") is None:
        return None
    values = [float(row.get("value") or 0) for row in rows]
    # Never roll up daily aggregates in application code: SUM of daily
    # COUNT(DISTINCT), AVG, or ratios can change the governed metric meaning.
    # The range total is therefore evaluated independently by Doris using the
    # exact same published semantic formula.
    total = float(total_rows[0]["value"])
    normalized_rows = [{"metric_name": metric["label"], "pt": str(row.get("pt") or ""), "value": float(row.get("value") or 0)} for row in rows]
    first, last = values[0], values[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    delta = abs(last - first)
    direction_text = "整体持平" if delta == 0 else "整体%s %s" % (direction, format_value(delta, metric))
    peak_index = max(range(len(values)), key=values.__getitem__)
    peak = normalized_rows[peak_index]
    total_text = format_value(total, metric)
    advice = metric_advice(metric["label"])
    time_label = time_range.label or time_range_label(question, days)
    freshness_sentence = ""
    if time_range.kind == "rolling" and not partition_date_matches(latest_partition, time_range.end_date):
        freshness_sentence = "数据日期截至 %s。\n\n" % latest_partition
    answer = (
        f"{time_label}，店铺{metric['label']}合计为 {total_text}。\n\n"
        f"{freshness_sentence}"
        f"从每日表现看，{metric['label']}由 {format_value(first, metric)} 变化到 {format_value(last, metric)}，{direction_text}；"
        f"峰值日期为 {peak['pt']}，峰值为 {format_value(peak['value'], metric)}。\n\n"
        "建议：\n"
        f"- {advice[0]}\n"
        f"- {advice[1]}"
    )
    suggestions = metric_suggestions(metric["label"], days)
    traceability = {
        "sourceSummary": "Doris 快速指标查询",
        "merchantId": merchant_id,
        "timeRange": time_label,
        "dataUpdatedAt": normalized_rows[-1]["pt"],
        "rowCount": len(normalized_rows),
        "sourceTables": [table],
        "evidenceStatus": "verified",
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
            ChatDataSection(title=f"{metric['label']}趋势", result_role="trend_context", doris_tables=[table], data_rows=normalized_rows),
            ChatDataSection(title=metric["label"], result_role="summary", doris_tables=[table], data_rows=[{"metric_name": metric["label"], "value": total}]),
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
            "actualLatestPartition": latest_partition,
            "metric": metric["label"],
            "metricTerms": metric.get("terms") or [],
            "semanticMetric": semantic_metric_identity(metric),
        },
    )
    verification = verify_quick_metric_answer(question, metric, normalized_rows, total, answer)
    if not verification.passed:
        return None
    response.debug_trace["answerClaimVerification"] = verification.model_dump(by_alias=True)
    QUICK_RESPONSE_CACHE.set(cache_key, response.model_dump(by_alias=True))
    return response


def verify_quick_metric_answer(
    question: str,
    metric: Dict[str, Any],
    trend_rows: list[Dict[str, Any]],
    total: float,
    answer: str,
):
    summary_task_id = "quick_metric_summary"
    trend_task_id = "quick_metric_trend"
    date_context_task_id = "quick_metric_date_context"
    latest_date = str(trend_rows[-1].get("pt") or "") if trend_rows else ""
    peak_row = max(trend_rows, key=lambda row: float(row.get("value") or 0)) if trend_rows else {}
    peak_date = str(peak_row.get("pt") or "")
    first_value = float(trend_rows[0].get("value") or 0) if trend_rows else 0.0
    last_value = float(trend_rows[-1].get("value") or 0) if trend_rows else 0.0
    delta_value = abs(last_value - first_value)
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                answer_mode=AnswerMode.METRIC,
                plan_task_id=summary_task_id,
                preferred_table=metric["table"],
                metric_name="value",
                metric_resolution={"metricKey": "value", "displayName": metric["label"]},
            ),
            QuestionIntent(
                question=question,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id=trend_task_id,
                preferred_table=metric["table"],
                metric_name="value",
                group_by_column="pt",
                metric_resolution={"metricKey": "value", "displayName": metric["label"], "displayRole": "trend_context"},
            ),
            QuestionIntent(
                question=question,
                answer_mode=AnswerMode.DETAIL,
                plan_task_id=date_context_task_id,
                preferred_table=metric["table"],
                output_keys=["数据日期", "峰值日期"],
            ),
        ]
    )
    summary_bundle = QueryBundle(tables=[metric["table"]], rows=[{"value": total}], original_row_count=1)
    trend_bundle = QueryBundle(tables=[metric["table"]], rows=trend_rows, original_row_count=len(trend_rows))
    date_context_bundle = QueryBundle(
        tables=[metric["table"]],
        rows=[{"数据日期": latest_date, "峰值日期": peak_date, "变化值": delta_value}],
        original_row_count=1,
    )
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id=summary_task_id, success=True, query_bundle=summary_bundle),
            AgentTaskResult(task_id=trend_task_id, success=True, query_bundle=trend_bundle),
            AgentTaskResult(task_id=date_context_task_id, success=True, query_bundle=date_context_bundle),
        ],
        merged_query_bundle=summary_bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    return AnswerClaimVerifier().verify(question, plan, run_result, answer)


def quick_metric_definition_response(question: str, merchant_id: str, metrics: list[Dict[str, Any]]) -> Optional[ChatResponse]:
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
            "drillDownActions": [{"label": "按该口径看趋势", "question": suggestions[0], "actionType": "follow_up_question"}],
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
    if str(getattr(keywords, "analysis_intent", "") or "") in {"attribution", "ranking", "detail", "advice", "ratio"}:
        return True
    structured_metrics = list(getattr(keywords, "metric_keywords", None) or [])
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
    return any(
        str(getattr(item, "kind", "") or "") == "dimension"
        and str(getattr(item, "canonical_key", "") or "") != "pt"
        for item in (getattr(keywords, "mentions", None) or [])
    )


def published_semantic_quick_metrics(topic_assets: Any, preferred_topics: Optional[Iterable[str]] = None) -> list[Dict[str, Any]]:
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
    label = str(metric.get("businessName") or metric.get("displayName") or metric_key).strip()
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
    }


def semantic_metric_identity(metric: Dict[str, Any]) -> Dict[str, str]:
    return {
        "topic": str(metric.get("topic") or ""),
        "category": metric_topic_category(metric),
        "table": str(metric.get("table") or ""),
        "metricKey": str(metric.get("key") or ""),
        "formula": str(metric.get("formula") or ""),
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
            term
            for metric in semantic_metrics
            for term in metric.get("terms") or []
            if term and term.lower() in text
        }
        phrases = [term for term in matched_terms if not any(term != other and term.lower() in other.lower() for other in matched_terms)]
    identities: set[tuple[str, str]] = set()
    result: list[Dict[str, Any]] = []
    for phrase in phrases:
        scored = [(semantic_phrase_score(metric, phrase), metric) for metric in semantic_metrics]
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
    return f"{value:,.0f}{metric['unit']}"


def metric_advice(label: str) -> list[str]:
    return [f"复盘{label}峰值日期对应的主要维度。", f"将{label}与语义层中相关指标联动观察，确认变化来源。"]


def metric_suggestions(label: str, days: int) -> list[str]:
    return [
        f"{label}波动最大的日期有哪些？",
        f"最近{days}天{label}按维度拆解",
        f"{label}异常原因是什么？",
    ]
