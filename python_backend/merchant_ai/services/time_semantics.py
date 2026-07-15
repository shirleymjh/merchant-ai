from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from merchant_ai.models import QueryPlan, ResolvedTimeRange


CALENDAR_ANCHOR_POLICY = "calendar"
LATEST_PARTITION_ANCHOR_POLICY = "latest_available_partition"
EXPLICIT_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})(?:日)?(?!\d)")
ROLLING_DAYS_PATTERN = re.compile(r"(?:最近|近)\s*(\d{1,3})\s*天")
ROLLING_WEEK_PATTERN = re.compile(r"(?:最近|近)\s*(?:一|1)\s*(?:周|星期)")
ROLLING_MONTH_PATTERN = re.compile(r"(?:最近|近)\s*(?:一|1)\s*个?月")
PREVIOUS_DAYS_PATTERN = re.compile(r"(?:前|上一个|上一|上期|上一期)\s*(\d{1,3})?\s*天")
COMPARISON_MARKER_PATTERN = re.compile(r"相比|对比|比较|变化|环比|较|比|上升|下降|增长|减少|升高|降低")
DAILY_GRAIN_PATTERN = re.compile(r"每天|每日|按天|逐日|日趋势|每天的|按日|走势|趋势")


def resolve_time_range(
    question: str,
    timezone_name: str = "Asia/Shanghai",
    now: Optional[datetime] = None,
    default_days: int = 7,
) -> ResolvedTimeRange:
    today = local_today(timezone_name, now)
    text = str(question or "")
    explicit_dates = [safe_date(*match.groups()) for match in EXPLICIT_DATE_PATTERN.finditer(text)]
    explicit_dates = [item for item in explicit_dates if item]
    if len(explicit_dates) >= 2:
        start, end = sorted(explicit_dates[:2])
        return resolved_range("explicit_range", start, end, timezone_name, "%s 至 %s" % (start, end), "question_dates")
    if len(explicit_dates) == 1:
        target = explicit_dates[0]
        return resolved_range("exact_date", target, target, timezone_name, target.isoformat(), "question_date")
    if "昨天" in text or "昨日" in text:
        target = today - timedelta(days=1)
        return resolved_range("exact_date", target, target, timezone_name, "昨天", "relative_yesterday")
    if "今天" in text or "今日" in text:
        return resolved_range("exact_date", today, today, timezone_name, "今天", "relative_today")
    if "本月" in text or "这个月" in text:
        start = today.replace(day=1)
        return resolved_range("calendar_month", start, today, timezone_name, "本月", "calendar_current_month")
    if "上月" in text or "上个月" in text:
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return resolved_range("calendar_month", start, end, timezone_name, "上月", "calendar_previous_month")
    if "本周" in text or "这周" in text:
        start = today - timedelta(days=today.weekday())
        return resolved_range("calendar_week", start, today, timezone_name, "本周", "calendar_current_week")
    if "上周" in text or "上星期" in text:
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return resolved_range("calendar_week", start, end, timezone_name, "上周", "calendar_previous_week")
    match = ROLLING_DAYS_PATTERN.search(text)
    if match:
        days = max(1, min(int(match.group(1)), 180))
        source = "relative_days"
    elif ROLLING_WEEK_PATTERN.search(text):
        days = 7
        source = "relative_week_phrase"
    elif ROLLING_MONTH_PATTERN.search(text):
        days = 30
        source = "relative_month_phrase"
    else:
        days = max(1, int(default_days or 7))
        source = "default_days"
    start = today - timedelta(days=days - 1)
    label = "最近%d天" % days
    return resolved_range("rolling", start, today, timezone_name, label, source)


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
    primary = resolve_time_range(text, timezone_name, now=now, default_days=default_days)
    primary.window_role = "primary"
    comparison = resolve_comparison_time_range(text, primary, timezone_name, now=now)
    grain = "day" if DAILY_GRAIN_PATTERN.search(text) else "period"
    contract: Dict[str, Any] = {
        "primary": primary.model_dump(by_alias=True),
        "comparison": comparison.model_dump(by_alias=True) if comparison else {},
        "grain": grain,
        "requiresComparison": bool(comparison),
        "comparisonType": comparison.comparison_type if comparison else "",
        "source": "time_window_tool",
        "trace": time_window_contract_trace(text, primary, comparison, grain),
    }
    if ambiguous_recent_month(text):
        contract["ambiguities"] = [
            {
                "code": "RECENT_MONTH_AMBIGUOUS",
                "message": "最近一个月可按最近30天或自然月理解；当前按最近30天处理。",
                "default": "rolling_30_days",
            }
        ]
    return contract


def resolve_comparison_time_range(
    text: str,
    primary: ResolvedTimeRange,
    timezone_name: str,
    now: Optional[datetime] = None,
) -> Optional[ResolvedTimeRange]:
    if not comparison_requested(text):
        return None
    today = local_today(timezone_name, now)
    previous_days = PREVIOUS_DAYS_PATTERN.search(text)
    days = max(1, min(int(previous_days.group(1)), 180)) if previous_days and previous_days.group(1) else int(primary.days or 0)
    days = max(1, days or 1)
    if "同比" in text or "去年同期" in text:
        start = parse_iso_date(primary.start_date)
        end = parse_iso_date(primary.end_date)
        if start and end:
            return resolved_comparison_range(
                "year_over_year",
                shift_year(start, -1),
                shift_year(end, -1),
                timezone_name,
                "去年同期",
                "year_over_year",
                comparison_type="year_over_year",
            )
    if primary.kind in {"calendar_month"} and ("上月" in text or "上个月" in text or "环比" in text or "上期" in text):
        end = parse_iso_date(primary.start_date) or today.replace(day=1)
        end = end - timedelta(days=1)
        start = end.replace(day=1)
        return resolved_comparison_range("calendar_month", start, end, timezone_name, "上月", "previous_calendar_month")
    if primary.kind in {"calendar_week"} and ("上周" in text or "上星期" in text or "环比" in text or "上期" in text):
        start_primary = parse_iso_date(primary.start_date) or (today - timedelta(days=today.weekday()))
        end = start_primary - timedelta(days=1)
        start = end - timedelta(days=6)
        return resolved_comparison_range("calendar_week", start, end, timezone_name, "上周", "previous_calendar_week")
    if primary.kind == "exact_date":
        target = (parse_iso_date(primary.start_date) or today) - timedelta(days=1)
        return resolved_comparison_range(
            "exact_date",
            target,
            target,
            timezone_name,
            "前一天",
            "previous_day",
            offset_days=1,
        )
    offset = max(int(primary.days or days or 1), days)
    end = (parse_iso_date(primary.start_date) or today) - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return resolved_comparison_range(
        "previous_period",
        start,
        end,
        timezone_name,
        "前%d天" % days,
        "previous_period",
        offset_days=offset,
    )


def comparison_requested(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    if "去年同期" in value or "同比" in value or "环比" in value:
        return True
    if PREVIOUS_DAYS_PATTERN.search(value) and COMPARISON_MARKER_PATTERN.search(value):
        return True
    if any(marker in value for marker in ["上月", "上个月", "上周", "上星期", "前天"]) and COMPARISON_MARKER_PATTERN.search(value):
        return True
    if any(marker in value for marker in ["上期", "上一期", "上一周期"]) and COMPARISON_MARKER_PATTERN.search(value):
        return True
    return False


def ambiguous_recent_month(text: str) -> bool:
    return bool(ROLLING_MONTH_PATTERN.search(str(text or "")) and "自然月" not in str(text or ""))


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
        trace.append("ambiguous_recent_month=rolling_30_days")
    return trace


def apply_time_range_to_plan(plan: QueryPlan, time_range: ResolvedTimeRange) -> QueryPlan:
    if not plan.intents:
        return plan
    intents = []
    for intent in plan.intents:
        current = intent.time_range
        resolved = current if current and current.start_date and current.end_date else time_range
        intents.append(intent.model_copy(update={"time_range": resolved, "days": resolved.days or intent.days}))
    understanding = dict(plan.question_understanding or {})
    understanding["timeRange"] = time_range.model_dump(by_alias=True)
    return plan.model_copy(update={"intents": intents, "question_understanding": understanding})


def apply_time_window_contract_to_plan(plan: QueryPlan, contract: Dict[str, Any]) -> QueryPlan:
    if not plan.intents or not isinstance(contract, dict):
        return plan
    primary = time_range_from_contract(contract.get("primary") or {})
    if not primary:
        return plan
    primary.window_role = "primary"
    plan = apply_time_range_to_plan(plan, primary)
    understanding = dict(plan.question_understanding or {})
    understanding["timeWindowContract"] = contract
    if contract.get("requiresComparison"):
        understanding["analysisIntent"] = "comparison"
        plan = add_comparison_baseline_to_plan(plan, contract)
        understanding = dict(plan.question_understanding or understanding)
        understanding["timeWindowContract"] = contract
        understanding["analysisIntent"] = "comparison"
    return plan.model_copy(update={"question_understanding": understanding})


def add_comparison_baseline_to_plan(plan: QueryPlan, contract: Dict[str, Any]) -> QueryPlan:
    comparison = time_range_from_contract(contract.get("comparison") or {})
    if not comparison or any(str(intent.plan_task_id or "").endswith("_baseline") for intent in plan.intents):
        return plan
    comparison.window_role = "comparison"
    existing_ids = {str(intent.plan_task_id or "") for intent in plan.intents if intent.plan_task_id}
    task_id_map: Dict[str, str] = {}
    baseline_intents = []
    for intent in plan.intents:
        if not intent.plan_task_id:
            continue
        baseline_id = unique_baseline_task_id(intent.plan_task_id, existing_ids)
        existing_ids.add(baseline_id)
        task_id_map[intent.plan_task_id] = baseline_id
    for intent in plan.intents:
        baseline_id = task_id_map.get(intent.plan_task_id)
        if not baseline_id:
            continue
        resolution = dict(intent.metric_resolution or {})
        resolution["timeWindowRole"] = "comparison"
        resolution["displayRole"] = resolution.get("displayRole") or "comparison_baseline"
        baseline_intents.append(
            intent.model_copy(
                update={
                    "plan_task_id": baseline_id,
                    "depends_on_task_ids": [task_id_map.get(task_id, task_id) for task_id in intent.depends_on_task_ids],
                    "time_range": comparison,
                    "days": comparison.days or intent.days,
                    "metric_resolution": resolution,
                    "analysis_note": append_note(intent.analysis_note, "comparison baseline %s" % (comparison.label or "")),
                }
            )
        )
    baseline_deps = [
        dep.model_copy(
            update={
                "anchor_task_id": task_id_map.get(dep.anchor_task_id, dep.anchor_task_id),
                "dependent_task_id": task_id_map.get(dep.dependent_task_id, dep.dependent_task_id),
            }
        )
        for dep in plan.dependencies
        if dep.anchor_task_id in task_id_map and dep.dependent_task_id in task_id_map
    ]
    trace = list(plan.compiler_trace or [])
    trace.append("TIME_WINDOW_COMPARISON_BASELINE:%s" % ",".join(task_id_map.values()))
    agent_trace = list(plan.agent_trace or [])
    agent_trace.append("time_window_tool=comparison_baseline")
    return plan.model_copy(
        update={
            "intents": list(plan.intents) + baseline_intents,
            "dependencies": list(plan.dependencies) + baseline_deps,
            "compiler_trace": trace,
            "agent_trace": agent_trace,
        }
    )


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


def append_note(existing: str, note: str) -> str:
    parts = [str(existing or "").strip(), str(note or "").strip()]
    return "; ".join(part for part in parts if part)


def partition_date_matches(value: object, expected: str) -> bool:
    normalized = normalize_partition_date(value)
    return bool(normalized and expected and normalized == expected)


def normalize_partition_date(value: object) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return "%s-%s-%s" % (text[:4], text[4:6], text[6:8])
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if not match:
        return ""
    target = safe_date(*match.groups())
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
    partition_column: str = "pt",
    tenant_column: str = "",
    tenant_value_sql: str = "",
) -> str:
    table_sql = quote_sql_identifier(table)
    partition_sql = quote_sql_identifier(partition_column or "pt")
    where_sql = ""
    if tenant_column and tenant_value_sql:
        where_sql = " WHERE %s = %s" % (quote_sql_identifier(tenant_column), tenant_value_sql)
    return "(SELECT MAX(%s) FROM %s%s)" % (partition_sql, table_sql, where_sql)


def latest_partition_window_predicate(
    table: str,
    days: Any,
    partition_column: str = "pt",
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
    partition_sql = quote_sql_identifier(partition_column or "pt")
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
    partition_column: str = "pt",
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
        "partitionColumn": partition_column or "pt",
        "source": time_range.source,
        "windowRole": time_range.window_role,
        "offsetDays": int(time_range.offset_days or 0),
        "comparisonType": time_range.comparison_type,
    }
    if table:
        contract["table"] = table
    if tenant_column:
        contract["tenantColumn"] = tenant_column
    if anchor_policy == LATEST_PARTITION_ANCHOR_POLICY:
        contract["executionRule"] = "relative windows must anchor to MAX(%s) after merchant filter" % quote_sql_identifier(partition_column or "pt")
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


def safe_date(year: str, month: str, day: str) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None
