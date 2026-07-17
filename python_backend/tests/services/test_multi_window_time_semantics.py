from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    NodePlanContract,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.answer import deterministic_structured_answer
from merchant_ai.services.evidence import EvidenceVerifier
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
    assert contract["requiresComparison"] is False
    assert contract["comparisonType"] == ""
    assert contract["windowRelation"] == "explicit_conjunction"
    assert [window["label"] for window in contract["windows"]] == ["近三天", "今天"]
    assert [window["windowRole"] for window in contract["windows"]] == ["primary", "additional_1"]
    assert contract["comparison"] == {}
    assert [window["label"] for window in contract["additionalWindows"]] == ["今天"]
    assert contract["primary"]["startDate"] == "2026-07-14"
    assert contract["primary"]["endDate"] == "2026-07-16"
    assert contract["additionalWindows"][0]["startDate"] == "2026-07-16"
    assert contract["additionalWindows"][0]["endDate"] == "2026-07-16"
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
    assert [intent.time_range.window_role for intent in updated.intents] == ["primary", "additional_1"]
    assert [intent.days for intent in updated.intents] == [3, 1]
    assert updated.question_understanding.get("analysisIntent") != "comparison"
    assert QueryGraphValidator().validate(
        "近三天和今天有多少活跃用户",
        updated,
        PlanningAssetPack(),
    ).valid


def test_explicit_relation_between_two_windows_remains_a_comparison() -> None:
    contract = resolve_time_window_contract("今天比昨天活跃用户变化多少", now=NOW)
    plan = apply_time_window_contract_to_plan(
        QueryPlan(intents=[QuestionIntent(plan_task_id="active_users")]),
        contract,
    )

    assert contract["requiresMultipleWindows"] is True
    assert contract["requiresComparison"] is True
    assert contract["windowRelation"] == "explicit_comparison"
    assert [intent.time_range.window_role for intent in plan.intents] == ["primary", "comparison"]
    assert plan.question_understanding["analysisIntent"] == "comparison"


def test_independent_windows_keep_both_values_without_inventing_change() -> None:
    contract = resolve_time_window_contract("近三天和今天有多少活跃用户", now=NOW)
    plan = apply_time_window_contract_to_plan(
        QueryPlan(
            question_understanding={"analysisIntent": "none"},
            intents=[
                QuestionIntent(
                    question="近三天和今天有多少活跃用户",
                    answer_mode=AnswerMode.METRIC,
                    plan_task_id="active_users",
                    preferred_table="fact_active_users",
                    metric_name="active_users",
                    metric_column="active_users",
                    metric_formula="SUM(active_users)",
                    metric_specs=[
                        {
                            "metricName": "active_users",
                            "metricColumn": "active_users",
                            "metricFormula": "SUM(active_users)",
                            "sourceColumns": ["active_users"],
                            "semanticRefId": "semantic:test:active_users",
                            "displayName": "活跃用户数",
                            "unit": "人",
                        }
                    ],
                    metric_resolution={"metricKey": "active_users", "displayName": "活跃用户数"},
                )
            ],
        ),
        contract,
    )
    values = {"primary": 300, "additional_1": 120}
    tasks = [
        AgentTaskResult(
            task_id=intent.plan_task_id,
            success=True,
            query_bundle=QueryBundle(
                tables=["fact_active_users"],
                rows=[
                    {
                        "active_users": values[intent.time_range.window_role],
                        "__timeWindowRole": intent.time_range.window_role,
                        "__timeWindowLabel": intent.time_range.label,
                    }
                ],
            ),
            node_plan_contract=NodePlanContract(
                task_id=intent.plan_task_id,
                preferred_table="fact_active_users",
                metric_specs=intent.metric_specs,
            ),
        )
        for intent in plan.intents
    ]
    run = AgentRunResult(
        task_results=tasks,
        query_bundles=[task.query_bundle for task in tasks],
        merged_query_bundle=QueryBundle(rows=[row for task in tasks for row in task.query_bundle.rows]),
    )

    run.verified_evidence = EvidenceVerifier().verify("近三天和今天有多少活跃用户", plan, run)
    answer = deterministic_structured_answer("近三天和今天有多少活跃用户", plan, run)

    assert run.verified_evidence.passed
    assert "近三天：活跃用户数为 300人" in answer
    assert "今天：活跃用户数为 120人" in answer
    assert "上升" not in answer
    assert "下降" not in answer
    assert "持平" not in answer


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
