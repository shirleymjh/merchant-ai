from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import PlanningAssetPack, QueryPlan, QuestionIntent
from merchant_ai.services.planning import QueryGraphValidator
from merchant_ai.services.time_semantics import (
    apply_time_window_contract_to_plan,
    extract_temporal_lexical_spans,
    resolve_time_range,
    resolve_time_window_contract,
)


NOW = datetime(2026, 7, 16, 9, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_chinese_quantity_and_today_are_retained_as_two_explicit_windows() -> None:
    question = "近三天和今天有多少活跃用户"

    spans = extract_temporal_lexical_spans(question)
    contract = resolve_time_window_contract(question, now=NOW)

    assert [(span.text, span.value) for span in spans] == [("近三天", 3), ("今天", 1)]
    assert contract["requiresMultipleWindows"] is True
    assert contract["requiresComparison"] is True
    assert contract["comparisonType"] == "explicit_multi_window"
    assert contract["windowRelation"] == "explicit_conjunction"
    assert [window["label"] for window in contract["windows"]] == ["近三天", "今天"]
    assert contract["primary"]["startDate"] == "2026-07-14"
    assert contract["primary"]["endDate"] == "2026-07-16"
    assert contract["comparison"]["startDate"] == "2026-07-16"
    assert contract["comparison"]["endDate"] == "2026-07-16"
    assert resolve_time_range("近三天活跃用户", now=NOW).days == 3


def test_multi_window_contract_overrides_scalar_hint_and_builds_two_query_nodes() -> None:
    contract = resolve_time_window_contract("近三天和今天有多少活跃用户", now=NOW)
    scalar_today = resolve_time_range("今天有多少活跃用户", now=NOW)
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="active_users",
                time_range=scalar_today,
            )
        ]
    )

    updated = apply_time_window_contract_to_plan(plan, contract)

    assert [intent.time_range.label for intent in updated.intents] == ["近三天", "今天"]
    assert [intent.time_range.window_role for intent in updated.intents] == ["primary", "comparison"]
    assert [intent.days for intent in updated.intents] == [3, 1]
    assert QueryGraphValidator().validate(
        "近三天和今天有多少活跃用户",
        updated,
        PlanningAssetPack(),
    ).valid


def test_query_graph_validation_fails_closed_if_one_explicit_window_is_dropped() -> None:
    contract = resolve_time_window_contract("近三天和今天有多少活跃用户", now=NOW)
    plan = apply_time_window_contract_to_plan(
        QueryPlan(intents=[QuestionIntent(plan_task_id="active_users")]),
        contract,
    )
    incomplete = plan.model_copy(update={"intents": [plan.intents[1]]})

    gaps = QueryGraphValidator().validate(
        "近三天和今天有多少活跃用户",
        incomplete,
        PlanningAssetPack(),
    ).gaps

    assert any(gap.code == "TIME_WINDOW_NOT_PLANNED" and "近三天" in gap.evidence for gap in gaps)
