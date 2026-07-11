from __future__ import annotations

import re
import uuid
from typing import Any, Dict, Optional

from merchant_ai.models import ChatDataSection, ChatResponse
from merchant_ai.services.cache import TTLCache, stable_cache_key


METRICS: Dict[str, Dict[str, Any]] = {
    "GMV": {"terms": ["gmv", "成交金额", "交易金额"], "column": "order_gmv_amt_1d", "label": "GMV", "unit": "元", "agg": "SUM"},
    "订单量": {"terms": ["订单量", "订单数", "下单量"], "column": "order_cnt_1d", "label": "订单量", "unit": "单", "agg": "SUM"},
    "退款金额": {"terms": ["退款金额", "退货金额"], "column": "refund_amt_1d", "label": "退款金额", "unit": "元", "agg": "SUM"},
    "退款率": {"terms": ["退款率", "退货率"], "column": "refund_rate_1d", "label": "退款率", "unit": "%", "agg": "AVG"},
    "工单量": {"terms": ["工单量", "工单数"], "column": "cs_ticket_cnt_1d", "label": "客服工单量", "unit": "单", "agg": "SUM"},
    "赔付金额": {"terms": ["赔付金额", "赔偿金额"], "column": "seller_repay_amt_1d", "label": "赔付金额", "unit": "元", "agg": "SUM"},
    "成交订单量": {"terms": ["成交订单量", "支付订单量"], "column": "pay_order_cnt_1d", "label": "成交订单量", "unit": "单", "agg": "SUM"},
    "下单用户量": {"terms": ["下单用户量", "下单人数"], "column": "order_user_cnt_1d", "label": "下单用户量", "unit": "人", "agg": "SUM"},
    "客单价": {"terms": ["客单价", "平均订单金额"], "column": "avg_pay_order_amt_1d", "label": "客单价", "unit": "元", "agg": "AVG"},
    "发货超时量": {"terms": ["发货超时量", "发货超时订单"], "column": "ship_timeout_order_cnt_1d", "label": "发货超时量", "unit": "单", "agg": "SUM"},
    "履约量": {"terms": ["履约量", "签收订单量"], "column": "signed_order_cnt_1d", "label": "履约量", "unit": "单", "agg": "SUM"},
    "优惠金额": {"terms": ["优惠金额", "折扣金额"], "column": "pay_success_discount_amt_1d", "label": "优惠金额", "unit": "元", "agg": "SUM"},
    "在线商品量": {"terms": ["在线商品量", "在售商品量"], "column": "goods_online_cnt_1d", "label": "在线商品量", "unit": "个", "agg": "LAST"},
    "申诉次数": {"terms": ["申诉次数", "申诉量"], "column": "appeal_cnt_1d", "label": "申诉次数", "unit": "次", "agg": "SUM"},
    "处罚次数": {"terms": ["处罚次数", "处罚量"], "column": "punish_cnt_1d", "label": "处罚次数", "unit": "次", "agg": "SUM"},
    "保证金余额": {"terms": ["保证金余额", "保证金"], "column": "deposit_amt", "label": "保证金余额", "unit": "元", "agg": "LAST"},
}

COMPLEX_TERMS = ["哪些商品", "哪个商品", "类目", "渠道", "订单明细", "对应订单", "拆解", "归因"]
QUICK_RESPONSE_CACHE = TTLCache("quick_metric_response", max_entries=512, ttl_seconds=30)


def quick_metric_response(
    question: str,
    merchant_id: str,
    repository: Any,
    extracted_keywords: Any = None,
) -> Optional[ChatResponse]:
    matched_metrics = resolve_metrics(question)
    if structured_keywords_require_planner(extracted_keywords, matched_metrics) or "[用户附件上下文]" in question:
        return None
    if 1 < len(matched_metrics) <= 3 and not any(term in question for term in COMPLEX_TERMS):
        return quick_multi_metric_response(question, merchant_id, repository, matched_metrics)
    metric = matched_metrics[0] if len(matched_metrics) == 1 else None
    if not metric or any(term in question for term in COMPLEX_TERMS):
        return None
    if not any(term in question.lower() for term in ["多少", "总", "趋势", "走势", "变化", "最近", "近", "为什么", "原因"]):
        return None
    days = extract_days(question)
    cache_key = stable_cache_key("quick_metric", {"merchantId": merchant_id, "question": normalize_question(question), "days": days})
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    column = metric["column"]
    aggregate = metric["agg"]
    rows = repository.query(
        "SELECT pt, `%s` AS value FROM ads_merchant_profile "
        "WHERE merchant_id=%%s AND pt >= DATE_SUB((SELECT MAX(pt) FROM ads_merchant_profile WHERE merchant_id=%%s), INTERVAL %d DAY) "
        "ORDER BY pt" % (column, max(0, days - 1)),
        [merchant_id, merchant_id],
    )
    if not rows:
        return None
    values = [float(row.get("value") or 0) for row in rows]
    total = sum(values) if aggregate == "SUM" else values[-1] if aggregate == "LAST" else sum(values) / max(1, len(values))
    normalized_rows = [{"metric_name": metric["label"], "pt": str(row.get("pt") or ""), "value": float(row.get("value") or 0)} for row in rows]
    first, last = values[0], values[-1]
    direction = "上升" if last > first else "下降" if last < first else "持平"
    delta = abs(last - first)
    peak_index = max(range(len(values)), key=values.__getitem__)
    peak = normalized_rows[peak_index]
    total_text = format_value(total, metric)
    diagnostic = any(term in question for term in ["为什么", "原因", "分析"])
    diagnostic_text = (
        f"仅凭{metric['label']}时间序列可以确认波动发生的日期和幅度，但不能直接证明业务原因。"
        f"优先验证订单量、客单价、退款和活动变化，避免把时间相关性误判为因果。\n\n"
        if diagnostic else ""
    )
    advice = metric_advice(metric["label"])
    answer = (
        f"最近{days}天，店铺{metric['label']}合计为 {total_text}。\n\n"
        f"从每日表现看，{metric['label']}由 {format_value(first, metric)} 变化到 {format_value(last, metric)}，整体{direction} {format_value(delta, metric)}；"
        f"峰值出现在 {peak['pt']}，为 {format_value(peak['value'], metric)}。\n\n{diagnostic_text}"
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
        "sourceTables": ["ads_merchant_profile"],
        "evidenceStatus": "verified",
    }
    response = ChatResponse(
        id="quick_" + uuid.uuid4().hex,
        answer=answer,
        category_name="电商交易",
        persisted=False,
        doris_tables=["ads_merchant_profile"],
        suggestions=suggestions,
        thinking_steps=["识别简单指标问题", "读取指标口径", "查询 Doris 数据", "校验结果", "生成经营建议"],
        data_rows=normalized_rows,
        data_sections=[
            ChatDataSection(title=f"{metric['label']}趋势", result_role="trend_context", doris_tables=["ads_merchant_profile"], data_rows=normalized_rows),
            ChatDataSection(title=metric["label"], result_role="summary", doris_tables=["ads_merchant_profile"], data_rows=[{"metric_name": metric["label"], "value": total}]),
        ],
        merchant_experience={
            "version": "v1",
            "businessAdvice": advice,
            "suggestedQuestions": suggestions,
            "anomalyAlerts": [],
            "metricDisclosures": [{"metricKey": column, "displayName": metric["label"], "description": f"按日汇总{metric['label']}"}],
            "traceability": traceability,
            "drillDownActions": [{"label": "继续下钻", "question": suggestions[0], "actionType": "follow_up_question"}],
        },
        debug_trace={"quickMetricPath": True, "days": days, "metric": metric["label"]},
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
    cache_key = stable_cache_key(
        "quick_multi_metric",
        {"merchantId": merchant_id, "question": normalize_question(question), "metrics": [item["column"] for item in metrics], "days": days},
    )
    cached = QUICK_RESPONSE_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return fresh_cached_response(cached)
    projections = ", ".join("`%s` AS `m%d`" % (metric["column"], index) for index, metric in enumerate(metrics))
    rows = repository.query(
        "SELECT pt, %s FROM ads_merchant_profile "
        "WHERE merchant_id=%%s AND pt >= DATE_SUB((SELECT MAX(pt) FROM ads_merchant_profile WHERE merchant_id=%%s), INTERVAL %d DAY) "
        "ORDER BY pt" % (projections, max(0, days - 1)),
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
        total = sum(values) if aggregate == "SUM" else values[-1] if aggregate == "LAST" else sum(values) / max(1, len(values))
        metric_rows = [
            {"metric_name": metric["label"], "pt": str(row.get("pt") or ""), "value": float(row.get("m%d" % index) or 0)}
            for row in rows
        ]
        all_rows.extend(metric_rows)
        sections.extend([
            ChatDataSection(title=f"{metric['label']}趋势", result_role="trend_context", doris_tables=["ads_merchant_profile"], data_rows=metric_rows),
            ChatDataSection(title=metric["label"], result_role="summary", doris_tables=["ads_merchant_profile"], data_rows=[{"metric_name": metric["label"], "value": total}]),
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
        doris_tables=["ads_merchant_profile"],
        suggestions=suggestions,
        thinking_steps=["识别多指标问题", "匹配统一数据表", "并行读取指标", "校验时间范围", "生成联动建议"],
        data_rows=all_rows,
        data_sections=sections,
        merchant_experience={
            "version": "v1",
            "businessAdvice": advice,
            "suggestedQuestions": suggestions,
            "anomalyAlerts": [],
            "metricDisclosures": [
                {"metricKey": metric["column"], "displayName": metric["label"], "description": f"按日汇总{metric['label']}"}
                for metric in metrics
            ],
            "traceability": {
                "sourceSummary": "Doris 多指标快速查询",
                "merchantId": merchant_id,
                "timeRange": f"最近{days}天",
                "dataUpdatedAt": str(rows[-1].get("pt") or ""),
                "rowCount": len(rows),
                "sourceTables": ["ads_merchant_profile"],
                "evidenceStatus": "verified",
            },
            "drillDownActions": [{"label": "继续下钻", "question": suggestions[0], "actionType": "follow_up_question"}],
        },
        debug_trace={"quickMetricPath": True, "multiMetric": True, "days": days, "metrics": [item["label"] for item in metrics]},
    )
    QUICK_RESPONSE_CACHE.set(cache_key, response.model_dump(by_alias=True))
    return response


def resolve_metric(question: str) -> Optional[Dict[str, Any]]:
    metrics = resolve_metrics(question)
    return metrics[0] if metrics else None


def resolve_metrics(question: str) -> list[Dict[str, Any]]:
    text = question.lower()
    ranked = sorted(METRICS.values(), key=lambda item: max(len(term) for term in item["terms"]), reverse=True)
    return [metric for metric in ranked if any(term.lower() in text for term in metric["terms"])]


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
