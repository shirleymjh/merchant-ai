from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import AnswerMode, QueryPlan, QuestionIntent
from merchant_ai.services.time_semantics import apply_time_range_to_plan, normalize_partition_date, resolve_time_range


def test_resolve_yesterday_to_absolute_business_date():
    resolved = resolve_time_range(
        "昨天退款金额是多少？",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert resolved.kind == "exact_date"
    assert resolved.start_date == "2026-07-12"
    assert resolved.end_date == "2026-07-12"
    assert resolved.anchor_policy == "calendar"


def test_resolve_rolling_window_to_absolute_bounds():
    resolved = resolve_time_range(
        "最近7天订单量",
        "Asia/Shanghai",
        now=datetime(2026, 7, 13, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert resolved.start_date == "2026-07-07"
    assert resolved.end_date == "2026-07-13"
    assert resolved.days == 7
    assert resolved.anchor_policy == "latest_available_partition"


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
    assert updated.question_understanding["timeRange"]["anchorPolicy"] == "calendar"


def test_partition_date_normalization_accepts_common_pt_formats():
    assert normalize_partition_date("20260712") == "2026-07-12"
    assert normalize_partition_date("2026-07-12 00:00:00") == "2026-07-12"
