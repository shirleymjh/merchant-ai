from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    PlanDependency,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.answer import (
    TIME_DIMENSION_KEY,
    merchant_traceability,
    metric_series_rows_for_intent,
    multi_trend_metric_sentence,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier, build_verified_facts
from merchant_ai.services.quick_metrics import quick_metric_response
from merchant_ai.services.time_semantics import apply_time_window_contract_to_plan, resolve_time_window_contract


def test_daily_grain_uses_each_intents_declared_time_column():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天每天指标A走势",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_a",
                metric_name="metric_a",
                group_by_column="event_day",
                output_keys=["metric_a"],
                metric_resolution={"metricKey": "metric_a", "timeColumn": "event_day"},
            ),
            QuestionIntent(
                question="最近7天每天指标B走势",
                answer_mode=AnswerMode.DERIVED,
                plan_task_id="metric_b",
                metric_name="metric_b",
                group_by_column="booked_on",
                output_keys=["metric_b"],
                metric_resolution={"metricKey": "metric_b", "timeColumn": "booked_on"},
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="metric_a",
                dependent_task_id="metric_b",
                relation_type="DERIVED_COMPONENT",
            )
        ],
    )
    contract = resolve_time_window_contract(
        "最近7天每天指标A和指标B走势",
        now=datetime(2026, 7, 16, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    updated = apply_time_window_contract_to_plan(plan, contract)

    assert [intent.group_by_column for intent in updated.intents] == ["event_day", "booked_on"]
    assert updated.intents[0].output_keys[0] == "event_day"
    assert updated.intents[1].output_keys[0] == "booked_on"
    assert updated.dependencies[0].anchor_column == "event_day"
    assert updated.dependencies[0].dependent_column == "booked_on"


def test_daily_grain_does_not_relabel_a_scope_group_as_time_without_contract():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天每天指标A走势",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_a",
                metric_name="metric_a",
                group_by_column="tenant_key",
                metric_resolution={"metricKey": "metric_a"},
            )
        ]
    )
    contract = resolve_time_window_contract(
        "最近7天每天指标A走势",
        now=datetime(2026, 7, 16, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    updated = apply_time_window_contract_to_plan(plan, contract)

    assert updated.intents[0].answer_mode == AnswerMode.METRIC
    assert updated.intents[0].group_by_column == "tenant_key"
    assert "timeColumn" not in updated.intents[0].metric_resolution
    assert "TIME_WINDOW_GRAIN_UNBOUND:metric_a" in updated.compiler_trace


def test_quick_metric_projects_declared_event_day_to_generic_time_dimension():
    calls = []
    metric = {
        "key": "metric_a",
        "label": "指标A",
        "unit": "",
        "formula": "SUM(metric_a)",
        "compiled_formula": "SUM(`metric_a`)",
        "source_columns": ["metric_a"],
        "terms": ["指标a"],
        "table": "metric_table",
        "time_column": "event_day",
        "tenant_column": "tenant_key",
        "topic": "runtime_topic",
    }

    class Repository:
        def query(self, sql, params=None):
            calls.append((sql, params))
            if "GROUP BY" in sql:
                return [
                    {"time_dimension": "2026-07-15", "value": 10},
                    {"time_dimension": "2026-07-16", "value": 12},
                ]
            return [{"value": 22}]

    response = quick_metric_response("最近2天指标A趋势", "tenant-1", Repository(), semantic_metrics=[metric])

    assert response is not None
    assert "`event_day` AS `time_dimension`" in calls[0][0]
    assert "MAX(`event_day`)" in calls[0][0]
    assert response.data_rows[0][TIME_DIMENSION_KEY] == "2026-07-15"
    assert response.debug_trace["timeWindowContract"]["partitionColumn"] == "event_day"


def test_answer_and_claim_verification_follow_dynamic_event_day_contract():
    intent = QuestionIntent(
        question="最近2天指标A趋势",
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="metric_a",
        preferred_table="metric_table",
        metric_name="metric_a",
        metric_column="metric_a",
        group_by_column="event_day",
        output_keys=["event_day", "metric_a"],
        metric_resolution={
            "metricKey": "metric_a",
            "displayName": "指标A",
            "timeColumn": "event_day",
            "displayRole": "trend_context",
        },
    )
    rows = [
        {"event_day": "2026-07-15", "metric_a": 10},
        {"event_day": "2026-07-16", "metric_a": 15},
    ]
    bundle = QueryBundle(tables=["metric_table"], rows=rows, original_row_count=2)
    plan = QueryPlan(intents=[intent])
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_a", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    points = metric_series_rows_for_intent(plan, intent, rows)
    facts = build_verified_facts(plan, run)
    answer = multi_trend_metric_sentence("最近2天指标A趋势", plan, run)
    verification = AnswerClaimVerifier().verify(
        "最近2天指标A趋势",
        plan,
        run,
        "2026-07-15 的指标A为 10。",
    )
    traceability = merchant_traceability("最近2天指标A趋势", plan, run, None, [])

    assert points == [
        {"metric_name": "指标A", "metric_key": "metric_a", TIME_DIMENSION_KEY: "2026-07-15", "value": 10.0},
        {"metric_name": "指标A", "metric_key": "metric_a", TIME_DIMENSION_KEY: "2026-07-16", "value": 15.0},
    ]
    assert "2026-07-15" in answer and "2026-07-16" in answer
    assert next(fact for fact in facts if fact.column == "event_day").value_type == "date"
    assert all(fact.result_role == "trend_context" for fact in facts)
    assert verification.passed is True
    assert traceability["dataUpdatedAt"] == "2026-07-16"
