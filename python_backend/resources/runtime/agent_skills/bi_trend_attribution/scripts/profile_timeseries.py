#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    metric_keys = numeric_metric_keys(rows, exclude={date_key, "seller_id", "merchant_id"})
    metrics = [metric_profile(rows, date_key, key) for key in metric_keys[:6]]
    metrics = [item for item in metrics if item]
    findings = build_findings(metrics)
    caveats = []
    if not rows:
        caveats.append("没有可用于分析的 verified data rows。")
    if not date_key:
        caveats.append("结果缺少日期字段，只能做横截面比较，不能判断走势。")
    if len(rows) < 3:
        caveats.append("可用行数较少，异常判断可信度有限。")
    for gap in payload.get("evidenceGaps") or []:
        if isinstance(gap, dict) and gap.get("code"):
            caveats.append("%s：%s" % (gap.get("code"), gap.get("reason") or gap.get("answerInstruction") or "证据缺口"))
    markdown = render_answer(payload, metrics, findings, caveats)
    return {
        "skillName": "bi_trend_attribution",
        "rowCount": len(rows),
        "dateKey": date_key,
        "metricKeys": metric_keys,
        "metrics": metrics,
        "findings": findings,
        "caveats": dedupe(caveats),
        "answerMarkdown": markdown,
    }


def metric_profile(rows: List[Dict[str, Any]], date_key: str, metric_key: str) -> Dict[str, Any]:
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
        delta = decimal_value(item.get("delta"))
        max_vs_avg = decimal_value(item.get("maxVsAvgPct"))
        if delta is not None and delta != 0:
            direction = "上升" if delta > 0 else "下降"
            findings.append(
                {
                    "title": "%s 期初到期末%s" % (metric, direction),
                    "evidence": "%s 从 %s 变化到 %s，变化 %s。" % (metric, item.get("first"), item.get("last"), item.get("delta")),
                }
            )
        if max_vs_avg is not None and max_vs_avg.copy_abs() >= Decimal("30"):
            peak = item.get("max") or {}
            findings.append(
                {
                    "title": "%s 存在高峰日" % metric,
                    "evidence": "%s 在 %s 达到 %s，较均值高 %s%%。"
                    % (metric, peak.get("date") or "未知日期", peak.get("value"), item.get("maxVsAvgPct")),
                }
            )
    return findings[:8]


def render_answer(payload: Dict[str, Any], metrics: List[Dict[str, Any]], findings: List[Dict[str, Any]], caveats: List[str]) -> str:
    lines = ["分析结论："]
    if findings:
        lines.append("- 当前证据显示存在可解释的波动点，不能简单判断为业务为 0 或无异常。")
    else:
        lines.append("- 当前 verified evidence 没有显示足够强的异常信号，建议结合更长时间窗或更多维度复核。")
    lines.append("")
    lines.append("关键证据：")
    if findings:
        for item in findings[:6]:
            lines.append("- %s：%s" % (item.get("title"), item.get("evidence")))
    else:
        for item in metrics[:4]:
            lines.append(
                "- %s：首值 %s，末值 %s，均值 %s，峰值 %s。"
                % (
                    item.get("metric"),
                    item.get("first"),
                    item.get("last"),
                    item.get("average"),
                    (item.get("max") or {}).get("value"),
                )
            )
    disclosures = payload.get("metricDisclosures") or []
    if disclosures:
        lines.append("")
        lines.append("口径：")
        for item in disclosures[:6]:
            if isinstance(item, dict):
                metric = item.get("metricKey") or item.get("displayName") or item.get("requestedMetricRef")
                if not metric:
                    continue
                formula = item.get("formula") or ""
                table = item.get("ownerTable") or ""
                warning = item.get("fieldWarning") or ""
                detail = "，".join(part for part in [str(table), str(formula), str(warning)] if part)
                lines.append("- %s：%s" % (metric, detail))
    if caveats:
        lines.append("")
        lines.append("限制：")
        for item in dedupe(caveats)[:6]:
            lines.append("- %s" % item)
    lines.append("")
    lines.append("建议：优先核对峰值日期对应的订单、退款原因和活动/履约变化，再判断是否需要按商品或订单继续下钻。")
    return "\n".join(lines)


def numeric_metric_keys(rows: List[Dict[str, Any]], exclude: set[str]) -> List[str]:
    keys: List[str] = []
    for row in rows[:40]:
        for key, value in row.items():
            name = str(key)
            if name in exclude or name in keys:
                continue
            if decimal_value(value) is not None:
                keys.append(name)
    return keys


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
