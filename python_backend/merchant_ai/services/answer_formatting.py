from __future__ import annotations

import re
from typing import Any, Mapping


def answer_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_metric_value_for_answer(
    value: Any,
    metric_key: str = "",
    label: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    del metric_key, label
    text = format_cell(value)
    numeric = answer_numeric_value(value)
    if numeric is None:
        return text
    contract = dict(metadata or {})
    value_format = str(contract.get("valueFormat") or contract.get("value_format") or "").strip().lower()
    unit = str(contract.get("unit") or "").strip()
    decimals = contract.get("decimalPlaces", contract.get("decimal_places", 2))
    try:
        decimal_places = max(0, min(int(decimals), 8))
    except (TypeError, ValueError):
        decimal_places = 2
    if value_format in {"percent", "percentage", "ratio"} or unit == "%":
        percent = numeric * 100 if abs(numeric) <= 1 else numeric
        return "%s%%" % _format_number(percent, decimal_places)
    if value_format in {"integer", "int", "count"}:
        rendered = str(int(numeric)) if float(numeric).is_integer() else _format_number(numeric, decimal_places)
    else:
        rendered = _format_number(numeric, decimal_places)
    return "%s%s" % (rendered, unit) if unit else rendered


def _format_number(value: float, decimal_places: int) -> str:
    if float(value).is_integer():
        return str(int(value))
    rendered = ("%%.%df" % decimal_places) % value
    return rendered.rstrip("0").rstrip(".")


def extract_question_time_phrase(question: str) -> str:
    text = str(question or "")
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
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", "", match.group(0))
    return ""


def humanize_column_name(column: str) -> str:
    text = str(column or "").strip()
    return "指标" if text else ""


def source_aware_metric_label(column: str, table: str = "", category: str = "") -> str:
    del column, table, category
    return ""


def identifier_like_column(column: str) -> bool:
    text = str(column or "").strip().lower()
    return text == "id" or text.endswith("_id") or text.endswith("_no")


def format_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ")[:80]
