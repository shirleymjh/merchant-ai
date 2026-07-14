from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from merchant_ai.models import QueryPlan, ResolvedTimeRange


EXPLICIT_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})(?:日)?(?!\d)")
ROLLING_DAYS_PATTERN = re.compile(r"(?:最近|近)\s*(\d{1,3})\s*天")


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
    match = ROLLING_DAYS_PATTERN.search(text)
    days = max(1, min(int(match.group(1)), 180)) if match else max(1, int(default_days or 7))
    start = today - timedelta(days=days - 1)
    label = "最近%d天" % days
    return resolved_range("rolling", start, today, timezone_name, label, "relative_days" if match else "default_days")


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


def resolved_range(kind: str, start: date, end: date, timezone_name: str, label: str, source: str) -> ResolvedTimeRange:
    return ResolvedTimeRange(
        kind=kind,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        days=(end - start).days + 1,
        label=label,
        timezone=timezone_name,
        anchor_policy="calendar",
        explicit=True,
        source=source,
    )


def safe_date(year: str, month: str, day: str) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None
