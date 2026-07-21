from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import AnswerMode, QueryPlan, QuestionIntent
from merchant_ai.services.routing import extract_days
from merchant_ai.services.time_semantics import (
    apply_time_range_to_plan,
    normalize_partition_date,
    resolve_time_range,
    resolve_time_window_contract,
)


def test_resolve_yesterday_to_absolute_business_date():
    resolved = resolve_time_range(
        "昨天退款金额是多少？",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert resolved.kind == "exact_date"
    assert resolved.start_date == "2026-07-12"
    assert resolved.end_date == "2026-07-12"
    assert resolved.calendar_anchor_policy == "runtime_current_date"
    assert resolved.data_as_of_policy == ""


def test_resolve_rolling_window_to_absolute_bounds():
    resolved = resolve_time_range(
        "最近7天订单量",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert resolved.start_date == "2026-07-07"
    assert resolved.end_date == "2026-07-13"
    assert resolved.days == 7
    assert resolved.calendar_anchor_policy == "runtime_current_date"
    assert resolved.data_as_of_policy == ""


def test_resolve_arbitrary_week_quantity_from_shared_temporal_span():
    resolved = resolve_time_range(
        "过去12周订单量",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert resolved.start_date == "2026-04-21"
    assert resolved.end_date == "2026-07-13"
    assert resolved.days == 84
    assert resolved.label == "过去12周"
    assert resolved.source == "relative_week_quantity"


def test_resolve_arbitrary_month_quantity_with_calendar_arithmetic_not_fixed_30_days():
    now = datetime(2026, 7, 31, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    resolved = resolve_time_range("最近3个月订单量", "Asia/Shanghai", now=now)

    assert resolved.start_date == "2026-05-01"
    assert resolved.end_date == "2026-07-31"
    assert resolved.days == 92
    assert resolved.label == "最近3个月"
    assert extract_days("最近3个月订单量") == resolve_time_range("最近3个月订单量").days


def test_previous_quantity_span_reuses_shared_contract_and_keeps_adjacent_windows():
    contract = resolve_time_window_contract(
        "最近30天订单量与前30天相比有什么变化？",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert contract["primary"]["startDate"] == "2026-06-14"
    assert contract["primary"]["endDate"] == "2026-07-13"
    assert contract["primary"]["days"] == 30
    assert contract["comparison"]["startDate"] == "2026-05-15"
    assert contract["comparison"]["endDate"] == "2026-06-13"
    assert contract["comparison"]["days"] == 30
    assert contract["comparison"]["offsetDays"] == 30
    assert contract["comparison"]["label"] == "前30天"


def test_apply_time_range_seals_every_query_intent():
    plan = QueryPlan(
        intents=[
            QuestionIntent(answer_mode=AnswerMode.METRIC, plan_task_id="order_metric", days=7),
            QuestionIntent(answer_mode=AnswerMode.METRIC, plan_task_id="refund_metric", days=7),
        ]
    )
    resolved = resolve_time_range(
        "昨天订单和退款",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    updated = apply_time_range_to_plan(plan, resolved)

    assert all(intent.time_range.end_date == "2026-07-12" for intent in updated.intents)
    assert all(intent.days == 1 for intent in updated.intents)
    assert (
        updated.question_understanding["timeRange"]["calendarAnchorPolicy"]
        == "runtime_current_date"
    )


def test_partition_date_normalization_accepts_common_pt_formats():
    assert normalize_partition_date("20260712") == "2026-07-12"
    assert normalize_partition_date("2026-07-12 00:00:00") == "2026-07-12"
