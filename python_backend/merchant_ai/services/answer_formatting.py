from __future__ import annotations

from typing import Any, Mapping

from merchant_ai.services.language_policy import load_language_policy


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
    if value_format in {"currency", "money", "fixed", "fixed_decimal"}:
        rendered = ("%%.%df" % decimal_places) % numeric
    elif value_format in {"integer", "int", "count"}:
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
    text = "".join(str(question or "").split())
    policy = load_language_policy().temporal
    fixed_phrases = tuple(
        phrase
        for phrase, semantic in policy.named_window_semantics.items()
        if semantic in {"previous_day", "current_day", "current_week", "current_month"}
    )
    observed = [
        (text.find(phrase), phrase)
        for phrase in fixed_phrases
        if text.find(phrase) >= 0
    ]
    for start in range(len(text)):
        for prefix in policy.rolling_month_prefixes:
            if not text.startswith(prefix, start):
                continue
            cursor = start + len(prefix)
            number_start = cursor
            while cursor < len(text) and text[cursor].isascii() and text[cursor].isdigit():
                cursor += 1
            if cursor == number_start:
                continue
            unit = next(
                (
                    candidate
                    for candidate in sorted(policy.units, key=len, reverse=True)
                    if text.startswith(candidate, cursor)
                ),
                "",
            )
            if not unit:
                continue
            observed.append((start, text[start : cursor + len(unit)]))
    if observed:
        return min(observed, key=lambda item: item[0])[1]
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
