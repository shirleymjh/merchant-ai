from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from merchant_ai.models import (
    AnswerMode,
    GraphValidationGap,
    QueryPlan,
    ResolvedTimeRange,
    RouteLexicalSpan,
    RouteSpanType,
)
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.text_parsing import (
    ASCII_DIGITS,
    contains_any_literal,
    leading_iso_date_parts,
    literal_spans,
    safe_ascii_component,
    separator_contains_only,
)


CALENDAR_ANCHOR_POLICY = "calendar"
LATEST_PARTITION_ANCHOR_POLICY = "latest_available_partition"
_TEMPORAL_LANGUAGE = load_language_policy().temporal
COMPARISON_MARKERS = _TEMPORAL_LANGUAGE.comparison_markers
DAILY_GRAIN_MARKERS = _TEMPORAL_LANGUAGE.daily_grain_markers
TIME_SERIES_ANALYSIS_MARKERS = _TEMPORAL_LANGUAGE.time_series_markers
TEMPORAL_PREFIXES = _TEMPORAL_LANGUAGE.prefixes
TEMPORAL_UNITS = tuple(sorted(_TEMPORAL_LANGUAGE.units, key=lambda item: (-len(item), item)))
TEMPORAL_NAMED_WINDOWS = _TEMPORAL_LANGUAGE.named_windows
TEMPORAL_NAMED_WINDOW_SEMANTICS = _TEMPORAL_LANGUAGE.named_window_semantics
TEMPORAL_UNIT_ALIASES = _TEMPORAL_LANGUAGE.units
COORDINATED_TIME_WINDOW_SEPARATOR_CHARACTERS = _TEMPORAL_LANGUAGE.coordinated_separator_characters
COMPARISON_TIME_WINDOW_SEPARATOR_CHARACTERS = _TEMPORAL_LANGUAGE.comparison_separator_characters
CHINESE_TEMPORAL_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _temporal_label(key: str, *, days: int = 0) -> str:
    template = str(_TEMPORAL_LANGUAGE.labels[key])
    return template.replace("{days}", str(max(0, int(days or 0))))


def _named_semantics_in_text(value: str) -> set[str]:
    return {
        str(TEMPORAL_NAMED_WINDOW_SEMANTICS.get(phrase) or "")
        for _start, _end, phrase in literal_spans(value, TEMPORAL_NAMED_WINDOW_SEMANTICS)
        if TEMPORAL_NAMED_WINDOW_SEMANTICS.get(phrase)
    }


def extract_temporal_lexical_spans(text: str) -> list[RouteLexicalSpan]:
    """Return canonical typed time spans without interpreting business text.

    Numeric windows are parsed as ``value + unit`` for any value.  Their
    prefix contributes a temporal role, while named calendar expressions use
    the same output contract.  Offsets always refer to the supplied text.
    """

    value = str(text or "")
    spans: list[RouteLexicalSpan] = []
    cursor = 0
    while cursor < len(value):
        parsed = _temporal_quantity_span_at(value, cursor)
        if parsed is None:
            cursor += 1
            continue
        end, prefix, quantity_text, unit_text = parsed
        quantity = parse_temporal_quantity(quantity_text)
        if quantity <= 0:
            cursor += 1
            continue
        spans.append(
            RouteLexicalSpan(
                span_type=RouteSpanType.TEMPORAL,
                start=cursor,
                end=end,
                text=value[cursor:end],
                source="quantity_window",
                value=quantity,
                unit=TEMPORAL_UNIT_ALIASES.get(unit_text, ""),
                role="previous_period" if prefix == "前" else "primary",
            )
        )
        cursor = end
    for start, end, window_text in literal_spans(value, TEMPORAL_NAMED_WINDOWS):
        spans.append(
            RouteLexicalSpan(
                span_type=RouteSpanType.TEMPORAL,
                start=start,
                end=end,
                text=window_text,
                source="named_calendar_window",
                value=1,
                unit=TEMPORAL_NAMED_WINDOWS[window_text],
                role="primary",
            )
        )
    spans.sort(key=lambda item: (item.start, -(item.end - item.start), item.source))
    accepted: list[RouteLexicalSpan] = []
    for candidate in spans:
        if any(candidate.start < current.end and current.start < candidate.end for current in accepted):
            continue
        accepted.append(candidate)
    return accepted


def _temporal_quantity_span_at(value: str, start: int) -> tuple[int, str, str, str] | None:
    cursor = start
    prefix = next((item for item in TEMPORAL_PREFIXES if value.startswith(item, cursor)), "")
    if prefix:
        cursor += len(prefix)
        cursor = _skip_whitespace(value, cursor)
    quantity_start = cursor
    allowed_quantity = {*CHINESE_TEMPORAL_DIGITS, "十", "百"}
    while cursor < len(value) and (value[cursor].isdigit() or value[cursor] in allowed_quantity):
        cursor += 1
    if cursor == quantity_start:
        return None
    quantity_text = value[quantity_start:cursor]
    cursor = _skip_whitespace(value, cursor)
    unit = next((item for item in TEMPORAL_UNITS if value.startswith(item, cursor)), "")
    if not unit:
        return None
    return cursor + len(unit), prefix, quantity_text, unit


def _skip_whitespace(value: str, cursor: int) -> int:
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    return cursor


def _extract_explicit_dates(value: str) -> list[date]:
    dates: list[date] = []
    cursor = 0
    while cursor < len(value):
        parsed = _explicit_date_at(value, cursor)
        if parsed is None:
            cursor += 1
            continue
        end, parts = parsed
        target = safe_date(*parts)
        if target is not None:
            dates.append(target)
        cursor = end
    return dates


def _explicit_date_at(value: str, start: int) -> tuple[int, tuple[str, str, str]] | None:
    if start > 0 and value[start - 1] in ASCII_DIGITS:
        return None
    year_end = start + 4
    if year_end >= len(value) or any(character not in ASCII_DIGITS for character in value[start:year_end]):
        return None
    if value[year_end] not in {"-", "/", "年"}:
        return None
    cursor = year_end + 1
    month_start = cursor
    while cursor < len(value) and value[cursor] in ASCII_DIGITS and cursor - month_start < 2:
        cursor += 1
    month = value[month_start:cursor]
    if not month or cursor >= len(value) or value[cursor] not in {"-", "/", "月"}:
        return None
    cursor += 1
    day_start = cursor
    while cursor < len(value) and value[cursor] in ASCII_DIGITS and cursor - day_start < 2:
        cursor += 1
    day = value[day_start:cursor]
    if not day:
        return None
    if cursor < len(value) and value[cursor] == "日":
        cursor += 1
    if cursor < len(value) and value[cursor] in ASCII_DIGITS:
        return None
    return cursor, (value[start:year_end], month, day)


def parse_temporal_quantity(value: object) -> int:
    """Parse Arabic or small Chinese quantities used in relative windows."""

    text = str(value or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    if any(character not in {*CHINESE_TEMPORAL_DIGITS, "十", "百"} for character in text):
        return 0
    total = 0
    section = 0
    digit = 0
    for character in text:
        if character in CHINESE_TEMPORAL_DIGITS:
            digit = CHINESE_TEMPORAL_DIGITS[character]
            continue
        unit = 10 if character == "十" else 100
        section += (digit or 1) * unit
        digit = 0
    total = section + digit
    return total if total > 0 else 0


def rolling_span_start(anchor: date, span: RouteLexicalSpan) -> date:
    """Resolve a typed rolling quantity without duration-specific branches."""

    amount = max(1, int(span.value or 0))
    unit = str(span.unit or "")
    if unit == "day":
        return anchor - timedelta(days=amount - 1)
    if unit == "week":
        return anchor - timedelta(weeks=amount) + timedelta(days=1)
    if unit == "month":
        # A rolling month follows calendar arithmetic.  It is deliberately not
        # approximated as 30 days because month lengths vary.
        return shift_month(anchor, -amount) + timedelta(days=1)
    raise ValueError("unsupported temporal quantity unit: %s" % unit)


def resolve_time_range(
    question: str,
    timezone_name: str = "Asia/Shanghai",
    now: Optional[datetime] = None,
    default_days: int = 7,
) -> ResolvedTimeRange:
    today = local_today(timezone_name, now)
    text = str(question or "")
    explicit_dates = _extract_explicit_dates(text)
    if len(explicit_dates) >= 2:
        start, end = sorted(explicit_dates[:2])
        return resolved_range("explicit_range", start, end, timezone_name, "%s 至 %s" % (start, end), "question_dates")
    if len(explicit_dates) == 1:
        target = explicit_dates[0]
        return resolved_range("exact_date", target, target, timezone_name, target.isoformat(), "question_date")
    named_span = next(
        (
            span
            for span in extract_temporal_lexical_spans(text)
            if span.source == "named_calendar_window"
        ),
        None,
    )
    if named_span:
        named_range = resolved_time_range_for_span(named_span, timezone_name, now=now)
        if named_range is not None:
            return named_range
    quantity_span = next(
        (
            span
            for span in extract_temporal_lexical_spans(text)
            if span.source == "quantity_window" and span.role == "primary"
        ),
        None,
    )
    if quantity_span:
        start = rolling_span_start(today, quantity_span)
        label = str(quantity_span.text or "").strip()
        source = "relative_%s_quantity" % str(quantity_span.unit or "")
        return resolved_range("rolling", start, today, timezone_name, label, source)
    days = max(1, int(default_days or 7))
    start = today - timedelta(days=days - 1)
    return resolved_range(
        "rolling",
        start,
        today,
        timezone_name,
        _temporal_label("default_rolling_days", days=days),
        "default_days",
    )


def resolve_time_window_contract(
    question: str,
    timezone_name: str = "Asia/Shanghai",
    now: Optional[datetime] = None,
    default_days: int = 7,
) -> Dict[str, Any]:
    """Resolve reusable time semantics for BI planning.

    The contract is intentionally independent from topics, metrics, and SQL. It
    gives the planner a stable tool result for primary windows, comparison
    windows, and output grain without asking the LLM to rediscover time grammar.
    """

    text = str(question or "")
    explicit_windows, window_relation = resolve_coordinated_time_windows(
        text,
        timezone_name,
        now=now,
    )
    primary = (
        explicit_windows[0]
        if explicit_windows
        else resolve_time_range(text, timezone_name, now=now, default_days=default_days)
    )
    primary.window_role = "primary"
    if len(explicit_windows) >= 2:
        secondary = explicit_windows[1]
        if window_relation == "explicit_comparison":
            secondary.window_role = "comparison"
            secondary.comparison_type = "explicit_comparison"
        else:
            secondary.window_role = "additional_1"
            secondary.comparison_type = ""
        comparison = secondary if window_relation == "explicit_comparison" else None
    else:
        secondary = None
        comparison = resolve_comparison_time_range(text, primary, timezone_name, now=now)
    grain = resolve_time_grain(text, comparison or secondary)
    windows = [primary.model_dump(by_alias=True)]
    if secondary:
        windows.append(secondary.model_dump(by_alias=True))
    elif comparison:
        windows.append(comparison.model_dump(by_alias=True))
    if len(explicit_windows) > 2:
        for index, window in enumerate(explicit_windows[2:], start=2):
            window.window_role = (
                "comparison_%d" % index
                if window_relation == "explicit_comparison"
                else "additional_%d" % index
            )
            window.comparison_type = "explicit_comparison" if window_relation == "explicit_comparison" else ""
            windows.append(window.model_dump(by_alias=True))
    requires_comparison = bool(comparison)
    contract: Dict[str, Any] = {
        "primary": primary.model_dump(by_alias=True),
        "comparison": comparison.model_dump(by_alias=True) if comparison else {},
        "additionalWindows": (
            [dict(item) for item in windows[1:]]
            if len(windows) > 1 and not requires_comparison
            else []
        ),
        "windows": windows,
        "grain": grain,
        "requiresComparison": requires_comparison,
        "requiresMultipleWindows": len(windows) > 1,
        "comparisonType": comparison.comparison_type if comparison else "",
        "windowRelation": window_relation or ("comparison" if comparison else "single"),
        "source": "time_window_tool",
        "trace": time_window_contract_trace(text, primary, comparison, grain),
    }
    if len(windows) > 1:
        contract["trace"].append(
            "explicit_windows=%d:relation=%s" % (len(windows), contract["windowRelation"])
        )
    if len(explicit_windows) > 2:
        contract.setdefault("ambiguities", []).append(
            {
                "code": "MULTI_WINDOW_EXECUTION_LIMIT",
                "message": "当前 QueryGraph 只支持两个显式窗口；其余窗口已保留并将在校验阶段阻止不完整执行。",
                "windowCount": len(explicit_windows),
            }
        )
    if ambiguous_recent_month(text):
        contract.setdefault("ambiguities", []).append(
            {
                "code": "RECENT_MONTH_AMBIGUOUS",
                "message": "最近N个月可按滚动日历月或完整自然月理解；当前按以锚点结束的滚动日历月处理。",
                "default": "rolling_calendar_months",
            }
        )
    return contract


def resolve_coordinated_time_windows(
    text: str,
    timezone_name: str,
    now: Optional[datetime] = None,
) -> tuple[list[ResolvedTimeRange], str]:
    """Resolve two adjacent, explicitly coordinated windows in mention order.

    The normal scalar resolver intentionally remains backward compatible. This
    helper recognizes only structurally clear expressions such as
    ``近三天和今天`` or ``今天比昨天``. It does not cross-product windows that
    are separated by metric/object text (for example ``今天订单和近七天退款``).
    """

    value = str(text or "")
    spans = [span for span in extract_temporal_lexical_spans(value) if span.role == "primary"]
    resolved: list[tuple[RouteLexicalSpan, ResolvedTimeRange]] = []
    for span in spans:
        window = resolved_time_range_for_span(span, timezone_name, now=now)
        if window is not None:
            resolved.append((span, window))
    if len(resolved) < 2:
        return [], ""
    coordinated = all(
        separator_contains_only(
            value[current[0].end : following[0].start],
            COORDINATED_TIME_WINDOW_SEPARATOR_CHARACTERS,
        )
        for current, following in zip(resolved, resolved[1:])
    )
    explicit_comparison = contains_any_literal(value, COMPARISON_MARKERS) and all(
        separator_contains_only(
            value[current[0].end : following[0].start],
            COMPARISON_TIME_WINDOW_SEPARATOR_CHARACTERS,
        )
        for current, following in zip(resolved, resolved[1:])
    )
    if not coordinated and not explicit_comparison:
        return [], ""
    windows: list[ResolvedTimeRange] = []
    seen: set[tuple[str, str, str]] = set()
    for _, window in resolved:
        identity = (window.kind, window.start_date, window.end_date)
        if identity in seen:
            continue
        seen.add(identity)
        windows.append(window)
    if len(windows) < 2:
        return [], ""
    relation = "explicit_comparison" if explicit_comparison else "explicit_conjunction"
    return windows, relation


def resolved_time_range_for_span(
    span: RouteLexicalSpan,
    timezone_name: str,
    now: Optional[datetime] = None,
) -> Optional[ResolvedTimeRange]:
    today = local_today(timezone_name, now)
    label = str(span.text or "").strip()
    if span.source == "quantity_window":
        if span.role != "primary":
            return None
        return resolved_range(
            "rolling",
            rolling_span_start(today, span),
            today,
            timezone_name,
            label,
            "relative_%s_quantity" % str(span.unit or ""),
        )
    if span.source != "named_calendar_window":
        return None
    semantic = str(TEMPORAL_NAMED_WINDOW_SEMANTICS.get(label) or "")
    if semantic == "current_day":
        return resolved_range("exact_date", today, today, timezone_name, label, "relative_today")
    if semantic == "previous_day":
        target = today - timedelta(days=1)
        return resolved_range("exact_date", target, target, timezone_name, label, "relative_yesterday")
    if semantic == "two_days_ago":
        target = today - timedelta(days=2)
        return resolved_range("exact_date", target, target, timezone_name, label, "relative_two_days_ago")
    if semantic == "current_week":
        start = today - timedelta(days=today.weekday())
        return resolved_range("calendar_week", start, today, timezone_name, label, "calendar_current_week")
    if semantic == "previous_week":
        end = today - timedelta(days=today.weekday() + 1)
        return resolved_range("calendar_week", end - timedelta(days=6), end, timezone_name, label, "calendar_previous_week")
    if semantic == "current_month":
        return resolved_range(
            "calendar_month",
            today.replace(day=1),
            today,
            timezone_name,
            label,
            "calendar_current_month",
        )
    if semantic == "previous_month":
        end = today.replace(day=1) - timedelta(days=1)
        return resolved_range(
            "calendar_month",
            end.replace(day=1),
            end,
            timezone_name,
            label,
            "calendar_previous_month",
        )
    return None


def has_explicit_time_expression(question: str) -> bool:
    """Use the canonical resolver to distinguish user time from defaults."""

    return resolve_time_range(str(question or "")).source != "default_days"


def resolve_comparison_time_range(
    text: str,
    primary: ResolvedTimeRange,
    timezone_name: str,
    now: Optional[datetime] = None,
) -> Optional[ResolvedTimeRange]:
    if not comparison_requested(text):
        return None
    today = local_today(timezone_name, now)
    previous_span = next(
        (
            span
            for span in extract_temporal_lexical_spans(text)
            if span.source == "quantity_window" and span.role == "previous_period"
        ),
        None,
    )
    if contains_any_literal(text, _TEMPORAL_LANGUAGE.year_over_year_markers):
        start = parse_iso_date(primary.start_date)
        end = parse_iso_date(primary.end_date)
        if start and end:
            return resolved_comparison_range(
                "year_over_year",
                shift_year(start, -1),
                shift_year(end, -1),
                timezone_name,
                _temporal_label("year_over_year"),
                "year_over_year",
                comparison_type="year_over_year",
            )
    named_semantics = _named_semantics_in_text(text)
    period_over_period = contains_any_literal(text, _TEMPORAL_LANGUAGE.period_over_period_markers)
    if primary.kind in {"calendar_month"} and (
        "previous_month" in named_semantics or period_over_period
    ):
        end = parse_iso_date(primary.start_date) or today.replace(day=1)
        end = end - timedelta(days=1)
        start = end.replace(day=1)
        return resolved_comparison_range(
            "calendar_month",
            start,
            end,
            timezone_name,
            _temporal_label("previous_month"),
            "previous_calendar_month",
        )
    if primary.kind in {"calendar_week"} and (
        "previous_week" in named_semantics or period_over_period
    ):
        start_primary = parse_iso_date(primary.start_date) or (today - timedelta(days=today.weekday()))
        end = start_primary - timedelta(days=1)
        start = end - timedelta(days=6)
        return resolved_comparison_range(
            "calendar_week",
            start,
            end,
            timezone_name,
            _temporal_label("previous_week"),
            "previous_calendar_week",
        )
    if primary.kind == "exact_date":
        target = (parse_iso_date(primary.start_date) or today) - timedelta(days=1)
        return resolved_comparison_range(
            "exact_date",
            target,
            target,
            timezone_name,
            _temporal_label("previous_day"),
            "previous_day",
            offset_days=1,
        )
    end = (parse_iso_date(primary.start_date) or today) - timedelta(days=1)
    if previous_span:
        start = rolling_span_start(end, previous_span)
        label = str(previous_span.text or "").strip()
        source = "previous_%s_quantity" % str(previous_span.unit or "")
    else:
        days = max(1, int(primary.days or 1))
        start = end - timedelta(days=days - 1)
        label = _temporal_label("previous_days", days=days)
        source = "previous_period"
    # offsetDays shifts the comparison anchor away from the primary anchor;
    # it is independent of the comparison window's own duration.
    offset = max(1, int(primary.days or 1))
    return resolved_comparison_range(
        "previous_period",
        start,
        end,
        timezone_name,
        label,
        source,
        offset_days=offset,
    )


def comparison_requested(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    if contains_any_literal(
        value,
        (*_TEMPORAL_LANGUAGE.year_over_year_markers, *_TEMPORAL_LANGUAGE.period_over_period_markers),
    ):
        return True
    if any(span.role == "previous_period" for span in extract_temporal_lexical_spans(value)) and contains_any_literal(value, COMPARISON_MARKERS):
        return True
    if (
        _named_semantics_in_text(value) & _TEMPORAL_LANGUAGE.comparison_named_semantics
        and contains_any_literal(value, COMPARISON_MARKERS)
    ):
        return True
    if contains_any_literal(value, _TEMPORAL_LANGUAGE.previous_period_markers) and contains_any_literal(value, COMPARISON_MARKERS):
        return True
    return False


def ambiguous_recent_month(text: str) -> bool:
    value = str(text or "")
    if contains_any_literal(value, _TEMPORAL_LANGUAGE.natural_month_markers):
        return False
    return any(
        span.source == "quantity_window"
        and span.role == "primary"
        and span.unit == "month"
        and str(span.text or "").strip().startswith(_TEMPORAL_LANGUAGE.rolling_month_prefixes)
        for span in extract_temporal_lexical_spans(value)
    )


def resolve_time_grain(text: str, comparison: Optional[ResolvedTimeRange]) -> str:
    value = str(text or "")
    if contains_any_literal(value, DAILY_GRAIN_MARKERS):
        return "day"
    # If the user explicitly asks for a period-to-period comparison, keep the
    # answer at period grain unless they also ask for daily/point trends.
    if comparison:
        return "period"
    if contains_any_literal(value, TIME_SERIES_ANALYSIS_MARKERS):
        return "day"
    return "period"


def resolved_comparison_range(
    kind: str,
    start: date,
    end: date,
    timezone_name: str,
    label: str,
    source: str,
    comparison_type: str = "previous_period",
    offset_days: int = 0,
) -> ResolvedTimeRange:
    payload = resolved_range(kind, start, end, timezone_name, label, source)
    payload.window_role = "comparison"
    payload.comparison_type = comparison_type
    payload.offset_days = max(0, int(offset_days or 0))
    if kind == "previous_period" and payload.offset_days > 0:
        payload.anchor_policy = LATEST_PARTITION_ANCHOR_POLICY
    return payload


def time_window_contract_trace(
    text: str,
    primary: ResolvedTimeRange,
    comparison: Optional[ResolvedTimeRange],
    grain: str,
) -> list[str]:
    trace = [
        "primary=%s:%s:%s..%s" % (primary.kind, primary.label, primary.start_date, primary.end_date),
        "grain=%s" % grain,
    ]
    if comparison:
        trace.append(
            "comparison=%s:%s:offsetDays=%d"
            % (comparison.kind, comparison.label, int(comparison.offset_days or 0))
        )
    if ambiguous_recent_month(text):
        trace.append("ambiguous_recent_month=rolling_calendar_months")
    return trace


def apply_time_range_to_plan(
    plan: QueryPlan,
    time_range: ResolvedTimeRange,
    *,
    force: bool = False,
) -> QueryPlan:
    if not plan.intents:
        return plan
    lookup_policy = plan_entity_lookup_time_policy(plan)
    lookup_mode = str(lookup_policy.get("mode") or "").strip().lower()
    lookup_explicit = bool(lookup_policy.get("timeScopeExplicit"))
    if lookup_policy and not lookup_explicit and lookup_mode in {
        "not_required",
        "global",
        "unbounded",
        "all_partitions",
    }:
        # The default calendar contract is not user evidence.  A published
        # global lookup contract must survive the workflow unchanged.
        return plan
    if lookup_policy and not lookup_explicit and lookup_mode in {
        "bounded_default",
        "default_window",
    }:
        policy_days = safe_positive_int(lookup_policy.get("defaultDays"))
        if policy_days:
            end = parse_iso_date(time_range.end_date)
            if end:
                start = end - timedelta(days=policy_days - 1)
                time_range = resolved_range(
                    "rolling",
                    start,
                    end,
                    time_range.timezone,
                    "语义资产默认%d天" % policy_days,
                    "entity_lookup_policy",
                ).model_copy(update={"explicit": False})
    intents = []
    for intent in plan.intents:
        current = intent.time_range
        resolved = (
            current
            if not force and current and current.start_date and current.end_date
            else time_range
        )
        intents.append(intent.model_copy(update={"time_range": resolved, "days": resolved.days or intent.days}))
    understanding = dict(plan.question_understanding or {})
    understanding["timeRange"] = time_range.model_dump(by_alias=True)
    return plan.model_copy(update={"intents": intents, "question_understanding": understanding})


def plan_entity_lookup_time_policy(plan: QueryPlan) -> Dict[str, Any]:
    references = [
        obligation.reference
        for obligation in plan.entity_filter_obligations
        if obligation.required and obligation.reference.status == "resolved"
    ]
    if not references:
        references = [
            intent.entity_reference
            for intent in plan.intents
            if intent.entity_reference.status == "resolved"
        ]
    if not references:
        return {}
    reference = references[0]
    policy = dict(reference.lookup_time_policy or {})
    if not policy:
        return {}
    policy["timeScopeExplicit"] = bool(reference.time_scope_explicit)
    return policy


def safe_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def apply_time_window_contract_to_plan(plan: QueryPlan, contract: Dict[str, Any]) -> QueryPlan:
    if not plan.intents or not isinstance(contract, dict):
        return plan
    lookup_policy = plan_entity_lookup_time_policy(plan)
    lookup_mode = str(lookup_policy.get("mode") or "").strip().lower()
    lookup_explicit = bool(lookup_policy.get("timeScopeExplicit"))
    if lookup_policy and not lookup_explicit and lookup_mode in {
        "not_required",
        "global",
        "unbounded",
        "all_partitions",
    }:
        return plan
    primary = time_range_from_contract(contract.get("primary") or {})
    if not primary:
        return plan
    if lookup_policy and not lookup_explicit and lookup_mode in {
        "bounded_default",
        "default_window",
    }:
        policy_days = safe_positive_int(lookup_policy.get("defaultDays"))
        end = parse_iso_date(primary.end_date)
        if policy_days and end:
            primary = resolved_range(
                "rolling",
                end - timedelta(days=policy_days - 1),
                end,
                primary.timezone,
                "语义资产默认%d天" % policy_days,
                "entity_lookup_policy",
            ).model_copy(update={"explicit": False})
            contract = dict(contract)
            contract["primary"] = primary.model_dump(by_alias=True)
            contract["source"] = "entity_lookup_policy"
            contract["trace"] = [
                *list(contract.get("trace") or []),
                "entity_lookup_policy=bounded_default:%d" % policy_days,
            ]
    primary.window_role = "primary"
    plan = apply_time_range_to_plan(
        plan,
        primary,
        force=bool(contract.get("requiresMultipleWindows")),
    )
    understanding = dict(plan.question_understanding or {})
    understanding["timeWindowContract"] = contract
    if contract.get("grain") == "day":
        plan = apply_time_grain_to_plan(plan, "day", primary, contract)
        understanding = dict(plan.question_understanding or understanding)
        understanding["timeWindowContract"] = contract
        understanding["analysisGrain"] = "day"
    elif str(contract.get("grain") or "").strip().lower() == "period":
        plan = apply_period_grain_to_plan(plan, contract)
        understanding = dict(plan.question_understanding or understanding)
        understanding["timeWindowContract"] = contract
    if contract.get("requiresMultipleWindows"):
        plan = add_secondary_time_windows_to_plan(plan, contract)
        understanding = dict(plan.question_understanding or understanding)
        understanding["timeWindowContract"] = contract
    if contract.get("requiresComparison"):
        understanding["analysisIntent"] = "comparison"
    return plan.model_copy(update={"question_understanding": understanding})


def time_window_contract_validation_gaps(plan: QueryPlan) -> list[GraphValidationGap]:
    """Fail closed when a sealed multi-window contract loses a window.

    QueryGraph nodes carry the executable window, while
    ``questionUnderstanding.timeWindowContract.windows`` is the immutable user
    obligation. Validation compares exact logical bounds so a plan cannot answer
    only ``今天`` after the user requested ``近三天和今天``.
    """

    understanding = dict(plan.question_understanding or {})
    contract = understanding.get("timeWindowContract") or understanding.get("time_window_contract") or {}
    if not isinstance(contract, dict) or not contract.get("requiresMultipleWindows"):
        return []
    expected_windows = [item for item in contract.get("windows") or [] if isinstance(item, dict)]
    if len(expected_windows) < 2:
        return [
            GraphValidationGap(
                code="INVALID_TIME_WINDOW_CONTRACT",
                reason="multi-window contract must retain at least two structured windows",
            )
        ]
    actual = {
        (
            str(intent.time_range.window_role or ""),
            str(intent.time_range.start_date or ""),
            str(intent.time_range.end_date or ""),
        )
        for intent in plan.intents
        if intent.time_range.start_date and intent.time_range.end_date
    }
    gaps: list[GraphValidationGap] = []
    for index, window in enumerate(expected_windows):
        role = str(window.get("windowRole") or ("primary" if index == 0 else "comparison"))
        start = str(window.get("startDate") or "")
        end = str(window.get("endDate") or "")
        if start and end and (role, start, end) in actual:
            continue
        gaps.append(
            GraphValidationGap(
                code="TIME_WINDOW_NOT_PLANNED",
                evidence="%s:%s..%s" % (str(window.get("label") or role), start, end),
                reason="QueryGraph does not cover every window in the sealed multi-window contract",
            )
        )
    return gaps


def apply_period_grain_to_plan(plan: QueryPlan, time_contract: Dict[str, Any]) -> QueryPlan:
    """Turn an explicitly period-grained temporal aggregate into a scalar.

    This is deliberately driven by the structured analysis/time contracts.  It
    never guesses a physical date field from its identifier, and it leaves
    entity/ranking groups untouched.
    """

    if not plan.intents:
        return plan
    understanding = dict(plan.question_understanding or {})
    analysis_grain = str(
        understanding.get("analysisGrain") or understanding.get("analysis_grain") or ""
    ).strip().lower()
    temporal_grains = {"time", "temporal", "date", "day", "daily", "time_series"}
    intents = []
    scalarized_task_ids: list[str] = []
    for intent in plan.intents:
        mode = str(getattr(intent.answer_mode, "value", intent.answer_mode) or "").upper()
        group_column = str(intent.group_by_column or "").strip()
        if mode not in {"METRIC", "GROUP_AGG"} or not group_column:
            intents.append(intent)
            continue
        resolution = dict(intent.metric_resolution or {})
        resolution_grain = str(
            resolution.get("timeGrain")
            or resolution.get("time_grain")
            or resolution.get("analysisGrain")
            or resolution.get("analysis_grain")
            or ""
        ).strip().lower()
        explicitly_declared_time_columns = {
            str(value).strip()
            for source in [
                time_contract,
                resolution,
                resolution.get("timeWindowContract") or resolution.get("time_window_contract") or {},
            ]
            if isinstance(source, dict)
            for value in [
                source.get("timeColumn"),
                source.get("time_column"),
                source.get("partitionColumn"),
                source.get("partition_column"),
            ]
            if str(value or "").strip()
        }
        temporal_group = bool(
            analysis_grain in temporal_grains
            or resolution_grain in temporal_grains
            or group_column in explicitly_declared_time_columns
        )
        if not temporal_group:
            intents.append(intent)
            continue
        for key in ["groupByColumn", "group_by_column"]:
            if str(resolution.get(key) or "").strip() == group_column:
                resolution.pop(key, None)
        resolution["timeWindowGrain"] = "period"
        resolution["displayRole"] = "summary"
        if str(resolution.get("visualization") or "").lower() in {
            "line_chart",
            "area_chart",
            "time_series",
        }:
            resolution.pop("visualization", None)
        intents.append(
            intent.model_copy(
                update={
                    "answer_mode": AnswerMode.METRIC,
                    "group_by_column": "",
                    "limit": 1,
                    "required_evidence": [
                        item for item in intent.required_evidence if str(item or "") != group_column
                    ],
                    "output_keys": [
                        item for item in intent.output_keys if str(item or "") != group_column
                    ],
                    "metric_resolution": resolution,
                    "analysis_note": append_note(intent.analysis_note, "time grain period scalar"),
                }
            )
        )
        if intent.plan_task_id:
            scalarized_task_ids.append(intent.plan_task_id)
    if not scalarized_task_ids:
        return plan
    selected_metrics = []
    for item in understanding.get("selectedMetrics") or understanding.get("selected_metrics") or []:
        if not isinstance(item, dict):
            continue
        selected = dict(item)
        for key in ["groupByColumn", "group_by_column"]:
            selected.pop(key, None)
        selected_metrics.append(selected)
    if "selectedMetrics" in understanding or selected_metrics:
        understanding["selectedMetrics"] = selected_metrics
    understanding["analysisGrain"] = "period"
    trace = list(plan.compiler_trace or [])
    trace.append("TIME_WINDOW_GRAIN:period:%s" % ",".join(scalarized_task_ids))
    updated = plan.model_copy(
        update={
            "intents": intents,
            "question_understanding": understanding,
            "compiler_trace": trace,
        }
    )
    return updated.model_copy(update={"evidence_contracts": evidence_contracts_from_current_intents(updated)})


def apply_time_grain_to_plan(
    plan: QueryPlan,
    grain: str,
    time_range: ResolvedTimeRange,
    time_contract: Optional[Dict[str, Any]] = None,
) -> QueryPlan:
    if grain != "day" or not plan.intents:
        return plan
    intents = []
    time_column_by_task: Dict[str, str] = {}
    unbound_task_ids: list[str] = []
    for intent in plan.intents:
        mode = str(getattr(intent.answer_mode, "value", intent.answer_mode) or "").upper()
        if mode not in {"METRIC", "GROUP_AGG", "DERIVED"}:
            intents.append(intent)
            continue
        time_column = declared_time_column_for_intent(intent, time_contract)
        if not time_column:
            intents.append(intent)
            if intent.plan_task_id:
                unbound_task_ids.append(intent.plan_task_id)
            continue
        resolution = dict(intent.metric_resolution or {})
        resolution.setdefault("displayRole", "trend_context")
        resolution.setdefault("visualization", "line_chart")
        resolution["timeColumn"] = time_column
        resolution["groupByColumn"] = time_column
        if intent.plan_task_id:
            time_column_by_task[intent.plan_task_id] = time_column
        intents.append(
            intent.model_copy(
                update={
                    "answer_mode": AnswerMode.DERIVED if mode == "DERIVED" else AnswerMode.GROUP_AGG,
                    "group_by_column": time_column,
                    "limit": max(int(time_range.days or intent.days or 0), int(intent.limit or 0), 7),
                    "required_evidence": time_grain_evidence_keys(intent.required_evidence, time_column),
                    "output_keys": time_grain_evidence_keys(intent.output_keys, time_column),
                    "metric_resolution": resolution,
                    "analysis_note": append_note(intent.analysis_note, "time grain day"),
                }
            )
        )
    dependencies = []
    for dep in plan.dependencies:
        anchor_time_column = time_column_by_task.get(dep.anchor_task_id, "")
        dependent_time_column = time_column_by_task.get(dep.dependent_task_id, "")
        if anchor_time_column and dependent_time_column and dep.relation_type == "DERIVED_COMPONENT":
            dependencies.append(
                dep.model_copy(
                    update={
                        "join_key": anchor_time_column,
                        "anchor_column": anchor_time_column,
                        "dependent_column": dependent_time_column,
                    }
                )
            )
        else:
            dependencies.append(dep)
    trace = list(plan.compiler_trace or [])
    if "TIME_WINDOW_GRAIN:day" not in trace:
        trace.append("TIME_WINDOW_GRAIN:day")
    trace.extend("TIME_WINDOW_GRAIN_UNBOUND:%s" % task_id for task_id in unbound_task_ids)
    updated = plan.model_copy(update={"intents": intents, "dependencies": dependencies, "compiler_trace": trace})
    return updated.model_copy(update={"evidence_contracts": evidence_contracts_from_current_intents(updated)})


def time_grain_evidence_keys(values: list[str], group_key: str) -> list[str]:
    keys = [str(item) for item in values or [] if str(item or "").strip()]
    updated = list(keys)
    if group_key and group_key not in updated:
        updated.insert(0, group_key)
    return dedupe_text(updated)


def dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def evidence_contracts_from_current_intents(plan: QueryPlan) -> list[Dict[str, Any]]:
    contracts: list[Dict[str, Any]] = []
    for intent in plan.intents:
        label = ""
        resolution = dict(intent.metric_resolution or {})
        if resolution:
            label = str(resolution.get("displayName") or resolution.get("naturalName") or resolution.get("metricKey") or "")
        label = label or str(intent.metric_name or intent.metric_column or intent.preferred_table or intent.plan_task_id or "")
        columns: list[str] = []
        for column in [intent.group_by_column, intent.filter_column, metric_contract_column(intent)]:
            text = str(column or "").strip()
            if text and text not in columns:
                columns.append(text)
        if not columns:
            columns = time_grain_evidence_keys(
                list(intent.output_keys or []) + list(intent.required_evidence or []),
                declared_time_column_for_intent(intent),
            )
        contract: Dict[str, Any] = {
            "taskId": intent.plan_task_id,
            "table": intent.preferred_table,
            "semanticLabel": label,
            "requiredLevel": "required",
            "columns": columns[:8],
        }
        if resolution:
            contract["metricResolution"] = resolution
        contracts.append(contract)
    return contracts


def metric_contract_column(intent: Any) -> str:
    resolution = dict(getattr(intent, "metric_resolution", None) or {})
    for key in ["metricKey", "metric_key"]:
        value = str(resolution.get(key) or "").strip()
        if value:
            return value
    return str(getattr(intent, "metric_name", "") or getattr(intent, "metric_column", "") or "").strip()


def declared_time_column_for_intent(
    intent: Any,
    time_contract: Optional[Dict[str, Any]] = None,
    *,
    accept_selected_group: bool = False,
) -> str:
    """Return a physical time column only when a runtime contract declares it.

    The semantic layer may bind the column on the metric resolution, a nested
    time-window contract, or the selected intent grouping.  There is deliberately
    no warehouse-specific fallback.
    """

    resolution = dict(getattr(intent, "metric_resolution", None) or {})
    nested_contract = resolution.get("timeWindowContract") or resolution.get("time_window_contract") or {}
    sources = [nested_contract, resolution, time_contract or {}]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ["partitionColumn", "partition_column", "timeColumn", "time_column"]:
            value = str(source.get(key) or "").strip()
            if value:
                return value
    selected_group = str(
        resolution.get("groupByColumn")
        or resolution.get("group_by_column")
        or getattr(intent, "group_by_column", "")
        or ""
    ).strip()
    if not selected_group:
        return ""
    temporal_role = str(resolution.get("displayRole") or resolution.get("display_role") or "").lower()
    visualization = str(resolution.get("visualization") or "").lower()
    semantic_grain = str(
        resolution.get("timeGrain")
        or resolution.get("time_grain")
        or resolution.get("analysisGrain")
        or resolution.get("analysis_grain")
        or ""
    ).lower()
    mode = str(getattr(getattr(intent, "answer_mode", ""), "value", getattr(intent, "answer_mode", "")) or "").upper()
    question_declares_series = mode in {"GROUP_AGG", "DERIVED"} and bool(
        contains_any_literal(str(getattr(intent, "question", "") or ""), DAILY_GRAIN_MARKERS)
        or contains_any_literal(str(getattr(intent, "question", "") or ""), TIME_SERIES_ANALYSIS_MARKERS)
    )
    if (
        accept_selected_group
        or temporal_role == "trend_context"
        or visualization in {"line_chart", "area_chart", "time_series"}
        or semantic_grain in {"day", "date", "time", "daily"}
        or question_declares_series
    ):
        return selected_group
    return ""


def add_secondary_time_windows_to_plan(plan: QueryPlan, contract: Dict[str, Any]) -> QueryPlan:
    """Clone the primary graph for every additional structured time window.

    Multiple requested windows are an execution-shape obligation.  They only
    become an analysis comparison when the time contract explicitly declares
    that relation; conjunction windows keep independent roles and values.
    """

    raw_windows = [item for item in contract.get("windows") or [] if isinstance(item, dict)]
    secondary_payloads = raw_windows[1:]
    if not secondary_payloads and isinstance(contract.get("comparison"), dict) and contract.get("comparison"):
        secondary_payloads = [dict(contract["comparison"])]
    if not plan.intents or not secondary_payloads:
        return plan

    base_intents = list(plan.intents)
    base_dependencies = list(plan.dependencies)
    existing_ids = {str(intent.plan_task_id or "") for intent in plan.intents if intent.plan_task_id}
    cloned_intents = []
    cloned_dependencies = []
    trace_entries: list[str] = []
    agent_entries: list[str] = []
    for index, payload in enumerate(secondary_payloads, start=1):
        window = time_range_from_contract(payload)
        if not window:
            continue
        role = str(window.window_role or payload.get("windowRole") or "additional_%d" % index)
        window.window_role = role
        if any(
            intent.time_range.window_role == role
            and intent.time_range.start_date == window.start_date
            and intent.time_range.end_date == window.end_date
            for intent in plan.intents
        ):
            continue
        task_id_map: Dict[str, str] = {}
        for intent in base_intents:
            if not intent.plan_task_id:
                continue
            clone_id = unique_time_window_task_id(intent.plan_task_id, role, existing_ids)
            existing_ids.add(clone_id)
            task_id_map[intent.plan_task_id] = clone_id
        for intent in base_intents:
            clone_id = task_id_map.get(intent.plan_task_id)
            if not clone_id:
                continue
            resolution = dict(intent.metric_resolution or {})
            resolution["timeWindowRole"] = role
            if role == "comparison" or role.startswith("comparison_"):
                resolution["displayRole"] = resolution.get("displayRole") or "comparison_baseline"
                note = "comparison baseline %s" % (window.label or "")
            else:
                resolution["displayRole"] = resolution.get("displayRole") or "summary"
                note = "additional time window %s" % (window.label or "")
            cloned_intents.append(
                intent.model_copy(
                    update={
                        "plan_task_id": clone_id,
                        "depends_on_task_ids": [task_id_map.get(task_id, task_id) for task_id in intent.depends_on_task_ids],
                        "time_range": window,
                        "days": window.days or intent.days,
                        "metric_resolution": resolution,
                        "analysis_note": append_note(intent.analysis_note, note),
                    }
                )
            )
        cloned_dependencies.extend(
            dep.model_copy(
                update={
                    "anchor_task_id": task_id_map.get(dep.anchor_task_id, dep.anchor_task_id),
                    "dependent_task_id": task_id_map.get(dep.dependent_task_id, dep.dependent_task_id),
                }
            )
            for dep in base_dependencies
            if dep.anchor_task_id in task_id_map and dep.dependent_task_id in task_id_map
        )
        trace_entries.append("TIME_WINDOW_%s:%s" % (role.upper(), ",".join(task_id_map.values())))
        agent_entries.append("time_window_tool=%s" % role)
    if not cloned_intents:
        return plan
    trace = list(plan.compiler_trace or [])
    trace.extend(trace_entries)
    agent_trace = list(plan.agent_trace or [])
    agent_trace.extend(agent_entries)
    return plan.model_copy(
        update={
            "intents": list(plan.intents) + cloned_intents,
            "dependencies": list(plan.dependencies) + cloned_dependencies,
            "compiler_trace": trace,
            "agent_trace": agent_trace,
        }
    )


def add_comparison_baseline_to_plan(plan: QueryPlan, contract: Dict[str, Any]) -> QueryPlan:
    """Backward-compatible wrapper for callers with a comparison contract."""

    return add_secondary_time_windows_to_plan(plan, contract)


def time_range_from_contract(payload: Dict[str, Any]) -> Optional[ResolvedTimeRange]:
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return ResolvedTimeRange.model_validate(payload)
    except Exception:
        return None


def unique_baseline_task_id(task_id: str, existing_ids: set[str]) -> str:
    base = "%s_baseline" % str(task_id or "task").strip()
    candidate = base
    index = 2
    while candidate in existing_ids:
        candidate = "%s_%d" % (base, index)
        index += 1
    return candidate


def unique_time_window_task_id(task_id: str, role: str, existing_ids: set[str]) -> str:
    if role == "comparison":
        return unique_baseline_task_id(task_id, existing_ids)
    safe_role = safe_ascii_component(role, default="additional")
    base = "%s_%s" % (str(task_id or "task").strip(), safe_role)
    candidate = base
    index = 2
    while candidate in existing_ids:
        candidate = "%s_%d" % (base, index)
        index += 1
    return candidate


def append_note(existing: str, note: str) -> str:
    parts = [str(existing or "").strip(), str(note or "").strip()]
    return "; ".join(part for part in parts if part)


def partition_date_matches(value: object, expected: str) -> bool:
    normalized = normalize_partition_date(value)
    return bool(normalized and expected and normalized == expected)


def normalize_partition_date(value: object) -> str:
    text = str(value or "").strip()
    if len(text) == 8 and text.isascii() and text.isdigit():
        return "%s-%s-%s" % (text[:4], text[4:6], text[6:8])
    date_parts = leading_iso_date_parts(text)
    if date_parts is None:
        return ""
    target = safe_date(*date_parts)
    return target.isoformat() if target else ""


def local_today(timezone_name: str, now: Optional[datetime]) -> date:
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    if now is None:
        return datetime.now(zone).date()
    if now.tzinfo is None:
        now = now.replace(tzinfo=zone)
    return now.astimezone(zone).date()


def time_window_anchor_policy(kind: str) -> str:
    return LATEST_PARTITION_ANCHOR_POLICY if kind == "rolling" else CALENDAR_ANCHOR_POLICY


def resolved_range(kind: str, start: date, end: date, timezone_name: str, label: str, source: str) -> ResolvedTimeRange:
    return ResolvedTimeRange(
        kind=kind,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        days=(end - start).days + 1,
        label=label,
        timezone=timezone_name,
        anchor_policy=time_window_anchor_policy(kind),
        explicit=True,
        source=source,
    )


def quote_sql_identifier(value: object) -> str:
    return "`%s`" % str(value or "").strip().replace("`", "")


def latest_partition_anchor_sql(
    table: str,
    partition_column: str = "",
    tenant_column: str = "",
    tenant_value_sql: str = "",
) -> str:
    if not str(partition_column or "").strip():
        raise ValueError("partition_column must be declared by the semantic contract")
    table_sql = quote_sql_identifier(table)
    partition_sql = quote_sql_identifier(partition_column)
    where_sql = ""
    if tenant_column and tenant_value_sql:
        where_sql = " WHERE %s = %s" % (quote_sql_identifier(tenant_column), tenant_value_sql)
    return "(SELECT MAX(%s) FROM %s%s)" % (partition_sql, table_sql, where_sql)


def latest_as_of_partition_predicate_sql(
    table: str,
    partition_column: str,
    *,
    anchor_value_sql: str = "",
    tenant_column: str = "",
    tenant_value_sql: str = "",
) -> str:
    """Compile one point-in-time selector shared by Quick and QueryGraph.

    Callers decide whether values are literals or bound placeholders.  The
    temporal operator itself is identical: select the latest tenant-scoped
    observation not after the runtime anchor.
    """

    if not str(partition_column or "").strip():
        raise ValueError("partition_column must be declared by the semantic contract")
    predicates: list[str] = []
    if tenant_column and tenant_value_sql:
        predicates.append("%s = %s" % (quote_sql_identifier(tenant_column), tenant_value_sql))
    if anchor_value_sql:
        predicates.append("%s <= %s" % (quote_sql_identifier(partition_column), anchor_value_sql))
    where_sql = " WHERE " + " AND ".join(predicates) if predicates else ""
    partition_sql = quote_sql_identifier(partition_column)
    return "%s = (SELECT MAX(%s) FROM %s%s)" % (
        partition_sql,
        partition_sql,
        quote_sql_identifier(table),
        where_sql,
    )


def latest_partition_window_predicate(
    table: str,
    days: Any,
    partition_column: str = "",
    tenant_column: str = "",
    tenant_value_sql: str = "",
    offset_days: Any = 0,
) -> str:
    anchor_sql = latest_partition_anchor_sql(table, partition_column, tenant_column, tenant_value_sql)
    try:
        interval_days = max(int(days or 0) - 1, 0)
    except (TypeError, ValueError):
        interval_days = 0
    try:
        offset = max(int(offset_days or 0), 0)
    except (TypeError, ValueError):
        offset = 0
    partition_sql = quote_sql_identifier(partition_column)
    upper_anchor_sql = anchor_sql if offset <= 0 else "DATE_SUB(%s, INTERVAL %d DAY)" % (anchor_sql, offset)
    lower_interval = interval_days if offset <= 0 else offset + interval_days
    return "%s BETWEEN DATE_SUB(%s, INTERVAL %d DAY) AND %s" % (
        partition_sql,
        anchor_sql,
        lower_interval,
        upper_anchor_sql,
    )


def time_window_contract_payload(
    time_range: ResolvedTimeRange,
    table: str = "",
    partition_column: str = "",
    tenant_column: str = "",
) -> Dict[str, Any]:
    anchor_policy = time_range.anchor_policy or time_window_anchor_policy(time_range.kind)
    contract = {
        "kind": time_range.kind,
        "label": time_range.label,
        "days": time_range.days,
        "startDate": time_range.start_date,
        "endDate": time_range.end_date,
        "timezone": time_range.timezone,
        "anchorPolicy": anchor_policy,
        "source": time_range.source,
        "windowRole": time_range.window_role,
        "offsetDays": int(time_range.offset_days or 0),
        "comparisonType": time_range.comparison_type,
    }
    if time_range.execution_start_date and time_range.execution_end_date:
        contract.update(
            {
                "executionStartDate": time_range.execution_start_date,
                "executionEndDate": time_range.execution_end_date,
                "executionStartValue": time_range.execution_start_value or time_range.execution_start_date,
                "executionEndValue": time_range.execution_end_value or time_range.execution_end_date,
                "executionAnchorPolicy": time_range.execution_anchor_policy or "source_snapshot",
            }
        )
    if partition_column:
        contract["partitionColumn"] = partition_column
    if table:
        contract["table"] = table
    if tenant_column:
        contract["tenantColumn"] = tenant_column
    if contract.get("executionStartValue") and contract.get("executionEndValue"):
        contract["executionRule"] = "use the runtime-bound executionStartValue/executionEndValue directly"
    elif anchor_policy == LATEST_PARTITION_ANCHOR_POLICY and partition_column:
        contract["executionRule"] = "relative windows must anchor to MAX(%s) after tenant filter" % quote_sql_identifier(partition_column)
    elif anchor_policy == LATEST_PARTITION_ANCHOR_POLICY:
        contract["executionRule"] = "relative window execution requires a semantic partitionColumn binding"
    else:
        contract["executionRule"] = "calendar/exact windows use startDate/endDate directly"
    return contract


def parse_iso_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except (TypeError, ValueError):
        return None


def shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def shift_month(value: date, months: int) -> date:
    """Shift by calendar months while preserving a valid day-of-month."""

    month_index = value.year * 12 + (value.month - 1) + int(months)
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def safe_date(year: str, month: str, day: str) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None
