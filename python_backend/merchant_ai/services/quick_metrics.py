from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Iterable, Optional

from merchant_ai.models import ChatDataSection, ChatResponse
from merchant_ai.services.cache import TTLCache, stable_cache_key


COMPLEX_TERMS = ["哪些商品", "哪个商品", "类目", "渠道", "订单明细", "对应订单", "拆解", "归因"]
ANALYSIS_TERMS = ["为什么", "原因", "分析", "归因", "诊断", "异常", "建议"]
SIMPLE_FORMULA = re.compile(r"^\s*(SUM|AVG|MAX|MIN)\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*$", re.I)
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
QUICK_RESPONSE_CACHE = TTLCache("quick_metric_response", max_entries=512, ttl_seconds=30)


def quick_metric_response(
    question: str,
    merchant_id: str,
    repository: Any,
    extracted_keywords: Any = None,
    semantic_metrics: Optional[list[Dict[str, Any]]] = None,
) -> Optional[ChatResponse]:
    metric_phrases = list(getattr(extracted_keywords, "metric_keywords", None) or [])
    matched_metrics = resolve_metrics(question, semantic_metrics or [], metric_phrases)
    if structured_keywords_require_planner(extracted_keywords, matched_metrics) or "[用户附件上下文]" in question:
        return None
    if any(term in question for term in ANALYSIS_TERMS):
        return None
    if 1 < len(matched_metrics) <= 3 and not any(term in question for term in COMPLEX_TERMS):
        return quick_multi_metric_response(question, merchant_id, repository, matched_metrics)
    metric = matched_metrics[0] if len(matched_metrics) == 1 else None
    if not metric or any(term in question for term in COMPLEX_TERMS):
        return None
    if not any(term in question.lower() for term in ["多少", "总", "趋势", "走势", "变化", "最近", "近", "为什么", "原因"]):
        return None
    days = extract_days(question)
    cache_key = stable_cache_key(
        "quick_metric",
        {
            "merchantId": merchant_id,
            "question": normalize_question(question),
            "days": days,
            "semanticContract": semantic_metric_identity(metric),
        },
    )
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    column = metric["column"]
    aggregate = metric["agg"]
    table = metric["table"]
    time_column = metric["time_column"]
    tenant_column = metric["tenant_column"]
    rows = repository.query(
        "SELECT `%s` AS pt, `%s` AS value FROM `%s` "
        "WHERE `%s`=%%s AND `%s` >= DATE_SUB((SELECT MAX(`%s`) FROM `%s` WHERE `%s`=%%s), INTERVAL %d DAY) "
        "ORDER BY `%s`"
        % (time_column, column, table, tenant_column, time_column, time_column, table, tenant_column, max(0, days - 1), time_column),
        [merchant_id, merchant_id],
    )
    if not rows:
        return None
    values = [float(row.get("value") or 0) for row in rows]
    total = aggregate_values(values, aggregate)
    normalized_rows = [{"metric_name": metric["label"], "pt": str(row.get("pt") or ""), "value": float(row.get("value") or 0)} for row in rows]
    first, last = values[0], values[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    delta = abs(last - first)
    peak_index = max(range(len(values)), key=values.__getitem__)
    peak = normalized_rows[peak_index]
    total_text = format_value(total, metric)
    advice = metric_advice(metric["label"])
    answer = (
        f"最近{days}天，店铺{metric['label']}合计为 {total_text}。\n\n"
        f"从每日表现看，{metric['label']}由 {format_value(first, metric)} 变化到 {format_value(last, metric)}，整体{direction} {format_value(delta, metric)}；"
        f"峰值出现在 {peak['pt']}，为 {format_value(peak['value'], metric)}。\n\n"
        "建议：\n"
        f"- {advice[0]}\n"
        f"- {advice[1]}"
    )
    suggestions = metric_suggestions(metric["label"], days)
    traceability = {
        "sourceSummary": "Doris 快速指标查询",
        "merchantId": merchant_id,
        "timeRange": f"最近{days}天",
        "dataUpdatedAt": normalized_rows[-1]["pt"],
        "rowCount": len(normalized_rows),
        "sourceTables": [table],
        "evidenceStatus": "verified",
    }
    response = ChatResponse(
        id="quick_" + uuid.uuid4().hex,
        answer=answer,
        category_name="电商交易",
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
        debug_trace={"quickMetricPath": True, "days": days, "metric": metric["label"], "semanticMetric": semantic_metric_identity(metric)},
    )
    QUICK_RESPONSE_CACHE.set(cache_key, response.model_dump(by_alias=True))
    return response


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


def quick_multi_metric_response(question: str, merchant_id: str, repository: Any, metrics: list[Dict[str, Any]]) -> Optional[ChatResponse]:
    if not any(term in question.lower() for term in ["趋势", "走势", "变化", "最近", "近", "一起看", "对比"]):
        return None
    days = extract_days(question)
    scopes = {(item["table"], item["time_column"], item["tenant_column"]) for item in metrics}
    if len(scopes) != 1:
        return None
    table, time_column, tenant_column = next(iter(scopes))
    cache_key = stable_cache_key(
        "quick_multi_metric",
        {
            "merchantId": merchant_id,
            "question": normalize_question(question),
            "metrics": [semantic_metric_identity(item) for item in metrics],
            "days": days,
        },
    )
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    projections = ", ".join("`%s` AS `m%d`" % (metric["column"], index) for index, metric in enumerate(metrics))
    rows = repository.query(
        "SELECT `%s` AS pt, %s FROM `%s` "
        "WHERE `%s`=%%s AND `%s` >= DATE_SUB((SELECT MAX(`%s`) FROM `%s` WHERE `%s`=%%s), INTERVAL %d DAY) "
        "ORDER BY `%s`"
        % (time_column, projections, table, tenant_column, time_column, time_column, table, tenant_column, max(0, days - 1), time_column),
        [merchant_id, merchant_id],
    )
    if not rows:
        return None
    sections: list[ChatDataSection] = []
    all_rows: list[Dict[str, Any]] = []
    summaries: list[str] = []
    for index, metric in enumerate(metrics):
        values = [float(row.get("m%d" % index) or 0) for row in rows]
        aggregate = metric["agg"]
        total = aggregate_values(values, aggregate)
        metric_rows = [
            {"metric_name": metric["label"], "pt": str(row.get("pt") or ""), "value": float(row.get("m%d" % index) or 0)}
            for row in rows
        ]
        all_rows.extend(metric_rows)
        sections.extend([
            ChatDataSection(title=f"{metric['label']}趋势", result_role="trend_context", doris_tables=[table], data_rows=metric_rows),
            ChatDataSection(title=metric["label"], result_role="summary", doris_tables=[table], data_rows=[{"metric_name": metric["label"], "value": total}]),
        ])
        direction = "上升" if values[-1] > values[0] else "下降" if values[-1] < values[0] else "持平"
        summaries.append(f"{metric['label']}为 {format_value(total, metric)}，期末较期初{direction} {format_value(abs(values[-1] - values[0]), metric)}")
    labels = "、".join(metric["label"] for metric in metrics)
    advice = [f"将{labels}放在同一时间轴观察，优先排查不同步变化的日期。", "对波动最大的日期下钻商品、订单和活动来源，确认变化是否来自交易增长或售后压力。"]
    suggestions = [f"{labels}波动最大的日期有哪些？", f"最近{days}天哪些商品影响了{metrics[0]['label']}？", f"{labels}异常时客服工单有什么变化？"]
    answer = f"最近{days}天，" + "；".join(summaries) + "。\n\n建议：\n- " + advice[0] + "\n- " + advice[1]
    response = ChatResponse(
        id="quick_" + uuid.uuid4().hex,
        answer=answer,
        category_name="电商交易",
        doris_tables=[table],
        suggestions=suggestions,
        thinking_steps=["识别多指标问题", "匹配统一数据表", "并行读取指标", "校验时间范围", "生成联动建议"],
        data_rows=all_rows,
        data_sections=sections,
        merchant_experience={
            "version": "v1",
            "businessAdvice": advice,
            "suggestedQuestions": suggestions,
            "anomalyAlerts": [],
            "metricDisclosures": [semantic_metric_disclosure(metric) for metric in metrics],
            "traceability": {
                "sourceSummary": "Doris 多指标快速查询",
                "merchantId": merchant_id,
                "timeRange": f"最近{days}天",
                "dataUpdatedAt": str(rows[-1].get("pt") or ""),
                "rowCount": len(rows),
                "sourceTables": [table],
                "evidenceStatus": "verified",
            },
            "drillDownActions": [{"label": "继续下钻", "question": suggestions[0], "actionType": "follow_up_question"}],
        },
        debug_trace={
            "quickMetricPath": True,
            "multiMetric": True,
            "days": days,
            "metrics": [item["label"] for item in metrics],
            "semanticMetrics": [semantic_metric_identity(item) for item in metrics],
        },
    )
    QUICK_RESPONSE_CACHE.set(cache_key, response.model_dump(by_alias=True))
    return response


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
    match = SIMPLE_FORMULA.fullmatch(formula)
    if not match:
        return None
    aggregate, column = match.group(1).upper(), match.group(2)
    source_columns = [str(item or "").strip() for item in metric.get("sourceColumns") or [] if str(item or "").strip()]
    if not SAFE_IDENTIFIER.fullmatch(column) or (source_columns and set(source_columns) != {column}):
        return None
    metric_key = str(metric.get("metricKey") or metric.get("canonicalMetricKey") or column).strip()
    label = str(metric.get("businessName") or metric.get("displayName") or metric_key).strip()
    aliases = [label, metric_key, column, *(metric.get("aliases") or [])]
    terms = list(dict.fromkeys(str(item or "").strip() for item in aliases if str(item or "").strip()))
    if not terms:
        return None
    return {
        "key": metric_key,
        "column": column,
        "label": label,
        "unit": str(metric.get("unit") or "").strip(),
        "agg": aggregate,
        "formula": formula,
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
        "table": str(metric.get("table") or ""),
        "metricKey": str(metric.get("key") or ""),
        "formula": str(metric.get("formula") or ""),
    }


def semantic_metric_disclosure(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metricKey": metric["key"],
        "displayName": metric["label"],
        "formula": metric["formula"],
        "description": metric.get("description") or "来自已发布语义资产",
        "semanticRef": "%s/%s/%s" % (metric["topic"], metric["table"], metric["key"]),
    }


def aggregate_values(values: list[float], aggregate: str) -> float:
    if not values:
        return 0.0
    if aggregate == "SUM":
        return sum(values)
    if aggregate == "MAX":
        return max(values)
    if aggregate == "MIN":
        return min(values)
    return sum(values) / len(values)


def resolve_metric(
    question: str,
    semantic_metrics: Optional[list[Dict[str, Any]]] = None,
    metric_phrases: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    metrics = resolve_metrics(question, semantic_metrics or [], metric_phrases)
    return metrics[0] if metrics else None


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
    match = re.search(r"(?:最近|近)?\s*(\d{1,3})\s*天", question)
    return max(1, min(int(match.group(1)), 180)) if match else 7


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
    if "退款" in label:
        return ["优先排查退款金额或退款率最高的商品和原因。", "同步查看客服工单与履约情况，处理集中出现的售后问题。"]
    if "工单" in label:
        return ["优先处理数量最多且仍在增长的工单类型。", "将高频问题沉淀为客服话术和商品说明，减少重复咨询。"]
    if "订单量" in label:
        return ["复盘订单量峰值日期对应的商品、活动和流量来源。", "将订单量与GMV、客单价和退款情况联动观察，判断变化来自流量还是转化。"]
    return [f"复盘{label}峰值日期对应的商品、活动和流量来源。", f"建立{label}连续下滑预警，并与订单、退款指标联动观察。"]


def metric_suggestions(label: str, days: int) -> list[str]:
    comparison_metric = "GMV" if "订单量" in label else "订单量"
    return [
        f"{label}波动最大的日期对应哪些商品？",
        f"最近{days}天{label}和{comparison_metric}一起看",
        f"{label}下降时退款和工单有什么变化？",
    ]
