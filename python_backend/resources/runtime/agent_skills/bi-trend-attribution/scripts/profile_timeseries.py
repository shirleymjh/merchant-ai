#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = load_json(Path(args.input))
    profile = build_profile(payload)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = [row for row in payload.get("dataRows") or [] if isinstance(row, dict)]
    date_key = first_present_key(rows, ["pt", "date", "dt", "biz_date"])
    labels = metric_labels(payload, rows)
    metric_keys = numeric_metric_keys(rows, payload, exclude={date_key, "seller_id", "merchant_id"})
    metrics = [metric_profile(rows, date_key, key, labels) for key in metric_keys[:6]]
    metrics = [item for item in metrics if item]
    findings = build_findings(metrics)
    caveats = []
    if not rows:
        caveats.append("没有可用于分析的当前数据。")
    if not date_key:
        caveats.append("结果缺少日期字段，只能做横截面比较，不能判断走势。")
    if len(rows) < 3:
        caveats.append("可用行数较少，异常判断可信度有限。")
    for gap in payload.get("evidenceGaps") or []:
        if isinstance(gap, dict) and gap.get("code"):
            caveats.append("%s：%s" % (gap.get("code"), gap.get("reason") or gap.get("answerInstruction") or "证据缺口"))
    markdown = render_answer(payload, metrics, findings, caveats)
    return {
        "skillName": "bi-trend-attribution",
        "rowCount": len(rows),
        "dateKey": date_key,
        "metricKeys": metric_keys,
        "metrics": metrics,
        "findings": findings,
        "caveats": dedupe(caveats),
        "answerMarkdown": markdown,
    }


def metric_profile(rows: List[Dict[str, Any]], date_key: str, metric_key: str, labels: Dict[str, str]) -> Dict[str, Any]:
    points = []
    for row in rows:
        value = decimal_value(row.get(metric_key))
        if value is None:
            continue
        points.append(
            {
                "date": str(row.get(date_key) or ""),
                "value": value,
                "row": compact_row(row),
            }
        )
    if not points:
        return {}
    first = points[0]["value"]
    last = points[-1]["value"]
    minimum = min(points, key=lambda item: item["value"])
    maximum = max(points, key=lambda item: item["value"])
    avg = sum((item["value"] for item in points), Decimal("0")) / Decimal(len(points))
    delta = last - first
    delta_pct = safe_pct(delta, first)
    max_vs_avg = safe_pct(maximum["value"] - avg, avg)
    return {
        "metric": metric_key,
        "label": labels.get(metric_key) or fallback_metric_label(metric_key),
        "points": len(points),
        "first": format_decimal(first),
        "last": format_decimal(last),
        "delta": format_decimal(delta),
        "deltaPct": format_decimal(delta_pct) if delta_pct is not None else "",
        "average": format_decimal(avg),
        "max": {"date": maximum["date"], "value": format_decimal(maximum["value"]), "row": maximum["row"]},
        "min": {"date": minimum["date"], "value": format_decimal(minimum["value"]), "row": minimum["row"]},
        "maxVsAvgPct": format_decimal(max_vs_avg) if max_vs_avg is not None else "",
    }


def build_findings(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for item in metrics:
        metric = item.get("metric") or "metric"
        label = item.get("label") or metric
        has_date = bool((item.get("max") or {}).get("date") or (item.get("min") or {}).get("date"))
        delta = decimal_value(item.get("delta"))
        max_vs_avg = decimal_value(item.get("maxVsAvgPct"))
        if has_date and delta is not None and delta != 0:
            direction = "上升" if delta > 0 else "下降"
            findings.append(
                {
                    "title": "%s 期初到期末%s" % (label, direction),
                    "evidence": "%s 从 %s 变化到 %s，变化 %s。" % (label, item.get("first"), item.get("last"), item.get("delta")),
                }
            )
        if max_vs_avg is not None and max_vs_avg.copy_abs() >= Decimal("30"):
            peak = item.get("max") or {}
            findings.append(
                {
                    "title": "%s 存在高峰日" % label,
                    "evidence": "%s 在 %s 达到 %s，较均值高 %s%%。"
                    % (label, peak.get("date") or "未知日期", peak.get("value"), item.get("maxVsAvgPct")),
                }
            )
    return findings[:8]


def render_answer(payload: Dict[str, Any], metrics: List[Dict[str, Any]], findings: List[Dict[str, Any]], caveats: List[str]) -> str:
    lines = []
    time_phrase = extract_question_time_phrase(str(payload.get("question") or ""))
    if metrics:
        primary = metrics[0]
        label = primary.get("label") or primary.get("metric") or "指标"
        direction = trend_direction(primary)
        prefix = "%s，" % time_phrase if time_phrase else ""
        if direction:
            first = format_business_number(primary.get("first"))
            last = format_business_number(primary.get("last"))
            delta = format_business_number(abs_decimal_text(primary.get("delta")))
            lines.append(
                "%s%s从 %s 变化到 %s，整体%s %s。"
                % (prefix, label, first, last, direction, delta)
            )
        else:
            lines.append("%s%s整体比较平稳，当前没有看到明显单边变化。" % (prefix, label))
        peak = primary.get("max") or {}
        trough = primary.get("min") or {}
        if peak.get("date") and trough.get("date") and peak.get("date") != trough.get("date"):
            lines.append(
                "峰值出现在 %s，为 %s；低点出现在 %s，为 %s。"
                % (peak.get("date"), format_business_number(peak.get("value")), trough.get("date"), format_business_number(trough.get("value")))
            )
        if len(metrics) > 1:
            extra = []
            for item in metrics[1:3]:
                item_label = item.get("label") or item.get("metric") or "指标"
                item_direction = trend_direction(item)
                if item_direction:
                    delta = format_business_number(abs_decimal_text(item.get("delta")))
                    extra.append("%s%s %s" % (item_label, item_direction, delta))
            if extra:
                lines.append("同时看到%s。" % "，".join(extra))
    else:
        lines.append("当前查询结果还不足以判断趋势。")
    disclosures = payload.get("metricDisclosures") or []
    if disclosures and question_asks_metric_disclosure(str(payload.get("question") or "")):
        lines.append("")
        lines.append("口径：")
        for item in disclosures[:6]:
            if isinstance(item, dict):
                metric = item.get("displayName") or item.get("metricKey") or item.get("requestedMetricRef")
                if not metric:
                    continue
                formula = item.get("formula") or ""
                warning = item.get("fieldWarning") or ""
                detail = "，".join(part for part in [str(formula), str(warning)] if part)
                lines.append("- %s：%s" % (metric, detail))
    if caveats:
        lines.append("")
        lines.append("说明：")
        for item in dedupe(caveats)[:6]:
            lines.append("- %s" % item)
    return "\n".join(lines)


def metric_labels(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for item in payload.get("metricDisclosures") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("metricKey") or item.get("requestedMetricRef") or item.get("metric") or "").strip()
        label = str(item.get("displayName") or "").strip()
        if key and label:
            labels[key] = label
    for row in rows[:40]:
        key = str(row.get("__metricKey") or "").strip()
        label = str(row.get("__metricName") or "").strip()
        if key and label:
            labels.setdefault(key, label)
    return labels


def fallback_metric_label(metric_key: str) -> str:
    mapping = {
        "order_gmv_amt_1d": "GMV",
        "pay_gmv_amt_1d": "支付GMV",
        "trade_success_gmv_amt_1d": "交易成功GMV",
        "refund_amt_1d": "退款金额",
        "pay_amt": "退款金额",
        "seller_repay_amt_1d": "赔付金额",
        "cs_ticket_cnt_1d": "咨询工单量",
        "order_detail_cnt": "订单量",
    }
    if metric_key in mapping:
        return mapping[metric_key]
    text = str(metric_key or "")
    dictionary = {
        "order": "订单",
        "detail": "明细",
        "cnt": "数量",
        "amt": "金额",
        "gmv": "GMV",
        "refund": "退款",
        "trade": "交易",
        "success": "成功",
        "pay": "支付",
        "ticket": "工单",
    }
    label = "".join(dictionary.get(part, "") for part in re.split(r"[_\s]+", text.lower()))
    return label or text


def question_asks_metric_disclosure(question: str) -> bool:
    return bool(re.search(r"(口径|怎么算|计算方式|字段|来源表|SQL|sql)", str(question or ""), flags=re.I))


def numeric_metric_keys(rows: List[Dict[str, Any]], payload: Dict[str, Any], exclude: set[str]) -> List[str]:
    disclosed = disclosed_metric_keys(payload)
    keys: List[str] = []
    for row in rows[:40]:
        for key, value in row.items():
            name = str(key)
            if name in exclude or name in keys:
                continue
            if entity_or_code_key(name):
                continue
            if disclosed and name not in disclosed:
                continue
            if not disclosed and not metric_shaped_key(name):
                continue
            if decimal_value(value) is not None:
                keys.append(name)
    return keys


def extract_question_time_phrase(question: str) -> str:
    for pattern in [
        r"最近\s*\d+\s*[天日周月]",
        r"近\s*\d+\s*[天日周月]",
        r"过去\s*\d+\s*[天日周月]",
        r"昨天",
        r"今日",
        r"今天",
        r"本周",
        r"本月",
    ]:
        match = re.search(pattern, str(question or ""))
        if match:
            return re.sub(r"\s+", "", match.group(0))
    return ""


def trend_direction(item: Dict[str, Any]) -> str:
    delta = decimal_value(item.get("delta"))
    if delta is None or delta == 0:
        return ""
    return "下降" if delta < 0 else "上升"


def abs_decimal_text(value: Any) -> str:
    numeric = decimal_value(value)
    if numeric is None:
        return str(value or "")
    return format_decimal(abs(numeric))


def format_business_number(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    numeric = decimal_value(text)
    if numeric is None:
        return text
    formatted = format_decimal(numeric).rstrip("0").rstrip(".")
    return formatted or "0"


def disclosed_metric_keys(payload: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in payload.get("metricDisclosures") or []:
        if not isinstance(item, dict):
            continue
        for field in ["metricKey", "requestedMetricRef"]:
            value = str(item.get(field) or "").strip()
            if value:
                keys.add(value)
    return keys


def entity_or_code_key(name: str) -> bool:
    text = str(name or "").strip().lower()
    if not text:
        return True
    return text == "id" or text.endswith("_id") or text.endswith("_code") or text.endswith("_no")


def metric_shaped_key(name: str) -> bool:
    text = str(name or "").strip().lower()
    return text.endswith(("_cnt", "_amt", "_rate", "_gmv")) or "gmv" in text


def first_present_key(rows: List[Dict[str, Any]], candidates: List[str]) -> str:
    for candidate in candidates:
        if any(candidate in row for row in rows[:10]):
            return candidate
    return ""


def decimal_value(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def safe_pct(delta: Decimal, base: Decimal) -> Decimal | None:
    if base == 0:
        return None
    return (delta / base) * Decimal("100")


def format_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    return str(value.quantize(Decimal("0.01")))


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): row.get(key) for key in list(row.keys())[:8]}


def dedupe(items: List[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
