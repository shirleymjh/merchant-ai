from datetime import datetime
from zoneinfo import ZoneInfo

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    IntentType,
    NodePlanContract,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    ResolvedTimeRange,
)
from merchant_ai.services.answer import deterministic_structured_answer
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.planning import QueryGraphPlanner
from merchant_ai.services.query import NodeWorkerExecutor
from merchant_ai.services.time_semantics import (
    apply_time_window_contract_to_plan,
    resolve_time_window_contract,
)


class UnconfiguredLlm:
    configured = False
    last_error = ""
    error_events = []


def period_comparison_contract() -> dict:
    return resolve_time_window_contract(
        "最近30天指标甲与前30天相比有什么变化",
        now=datetime(2026, 7, 16, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_period_time_contract_scalarizes_temporal_group_before_comparison_clone():
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "time",
            "selectedMetrics": [{"metricRef": "measure_a", "groupByColumn": "event_day"}],
        },
        intents=[
            QuestionIntent(
                question="最近30天指标甲与前30天相比有什么变化",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="measure_a",
                preferred_table="fact_window",
                metric_name="measure_a",
                metric_column="measure_a",
                metric_formula="SUM(measure_a)",
                group_by_column="event_day",
                required_evidence=["event_day", "measure_a"],
                output_keys=["event_day"],
                limit=1,
                metric_resolution={"metricKey": "measure_a", "timeColumn": "event_day"},
            )
        ],
    )

    updated = apply_time_window_contract_to_plan(plan, period_comparison_contract())

    assert len(updated.intents) == 2
    assert all(intent.answer_mode == AnswerMode.METRIC for intent in updated.intents)
    assert all(not intent.group_by_column for intent in updated.intents)
    assert all("event_day" not in intent.output_keys for intent in updated.intents)
    assert [intent.time_range.window_role for intent in updated.intents] == ["primary", "comparison"]
    assert updated.question_understanding["analysisGrain"] == "period"


def test_period_time_contract_does_not_scalarize_entity_group():
    plan = QueryPlan(
        question_understanding={"analysisGrain": "entity"},
        intents=[
            QuestionIntent(
                question="最近30天各对象指标甲与前30天相比",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="by_entity",
                metric_name="measure_a",
                group_by_column="entity_key",
            )
        ],
    )

    updated = apply_time_window_contract_to_plan(plan, period_comparison_contract())

    assert all(intent.group_by_column == "entity_key" for intent in updated.intents)
    assert all(intent.answer_mode == AnswerMode.GROUP_AGG for intent in updated.intents)


def test_independent_semantic_metrics_compile_as_period_scalars_without_default_time_group():
    refs = ["semantic:test:fact_window:metric:measure_a", "semantic:test:fact_window:metric:measure_b"]
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="fact_window",
                table="fact_window",
                columns=["tenant_key", "event_day", "measure_a", "measure_b"],
                metadata={"timeColumn": "event_day"},
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="measure_a",
                table="fact_window",
                columns=["measure_a"],
                title="指标甲",
                source_ref_id=refs[0],
                metadata={"formula": "SUM(measure_a)", "sourceColumns": ["measure_a"], "unit": "个"},
            ),
            PlanningAssetEntry(
                key="measure_b",
                table="fact_window",
                columns=["measure_b"],
                title="指标乙",
                source_ref_id=refs[1],
                metadata={"formula": "SUM(measure_b)", "sourceColumns": ["measure_b"], "unit": "元"},
            ),
        ],
    )
    payload = {
        "status": "SELECTED",
        "queryContract": {"contractType": "independent_metrics", "timeWindowDays": 30},
        "selectedRefs": refs,
        "selectedAssets": [
            {
                "semanticRefId": ref,
                "metricRef": metric,
                "ownerTable": "fact_window",
                "sourcePhrase": phrase,
            }
            for ref, metric, phrase in zip(refs, ["measure_a", "measure_b"], ["指标甲", "指标乙"])
        ],
    }

    plan = QueryGraphPlanner(UnconfiguredLlm())._compile_semantic_asset_selection_payload(
        "最近30天指标甲和指标乙分别是多少",
        payload,
        pack,
    )

    assert len(plan.intents) == 2
    assert all(intent.answer_mode == AnswerMode.METRIC for intent in plan.intents)
    assert all(not intent.group_by_column for intent in plan.intents)
    assert all(item.get("groupByColumn") == "" for item in plan.question_understanding["selectedMetrics"])


def comparison_specs(task_id: str) -> list[dict]:
    return [
        {
            "metricName": "measure_count",
            "metricColumn": "measure_count",
            "metricFormula": "SUM(measure_count)",
            "sourceColumns": ["measure_count"],
            "sourceTaskId": task_id,
            "semanticRefId": "semantic:test:measure_count",
            "displayName": "数量指标",
            "unit": "单",
            "valueFormat": "integer",
        },
        {
            "metricName": "measure_amount",
            "metricColumn": "measure_amount",
            "metricFormula": "SUM(measure_amount)",
            "sourceColumns": ["measure_amount"],
            "sourceTaskId": task_id,
            "semanticRefId": "semantic:test:measure_amount",
            "displayName": "金额指标",
            "unit": "元",
            "valueFormat": "decimal",
        },
        {
            "metricName": "measure_ratio",
            "metricColumn": "measure_numerator",
            "metricFormula": "SUM(measure_numerator) / NULLIF(SUM(measure_denominator), 0)",
            "sourceColumns": ["measure_numerator", "measure_denominator"],
            "sourceTaskId": task_id,
            "semanticRefId": "semantic:test:measure_ratio",
            "displayName": "比例指标",
            "unit": "%",
            "valueFormat": "percentage",
        },
    ]


def comparison_intent(task_id: str, role: str, label: str) -> QuestionIntent:
    specs = comparison_specs(task_id)
    return QuestionIntent(
        question="最近30天三个指标与前30天相比",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.METRIC,
        plan_task_id=task_id,
        preferred_table="fact_window",
        metric_name="measure_count",
        metric_column="measure_count",
        metric_formula="SUM(measure_count)",
        metric_specs=specs,
        time_range=ResolvedTimeRange(window_role=role, label=label, days=30),
        metric_resolution={"metricKey": "measure_count", "displayName": "数量指标"},
    )


def comparison_task(intent: QuestionIntent, values: dict, role: str, label: str) -> AgentTaskResult:
    rows = [{**values, "__timeWindowRole": role, "__timeWindowLabel": label}]
    return AgentTaskResult(
        task_id=intent.plan_task_id,
        success=True,
        query_bundle=QueryBundle(tables=["fact_window"], rows=rows),
        node_plan_contract=NodePlanContract(
            task_id=intent.plan_task_id,
            preferred_table="fact_window",
            metric_specs=intent.metric_specs,
        ),
    )


def comparison_plan_and_run() -> tuple[QueryPlan, AgentRunResult]:
    primary = comparison_intent("window_primary", "primary", "当前窗口")
    comparison = comparison_intent("window_comparison", "comparison", "对比窗口")
    plan = QueryPlan(
        intents=[primary, comparison],
        question_understanding={"timeWindowContract": {"requiresComparison": True, "grain": "period"}},
    )
    primary_task = comparison_task(
        primary,
        {"measure_count": 120, "measure_amount": 250, "measure_ratio": 0.12},
        "primary",
        "当前窗口",
    )
    comparison_task_result = comparison_task(
        comparison,
        {"measure_count": 100, "measure_amount": 200, "measure_ratio": 0.10},
        "comparison",
        "对比窗口",
    )
    tasks = [primary_task, comparison_task_result]
    return plan, AgentRunResult(
        task_results=tasks,
        query_bundles=[item.query_bundle for item in tasks],
        merged_query_bundle=QueryBundle(
            tables=["fact_window"],
            rows=[row for item in tasks for row in item.query_bundle.rows],
        ),
    )


def test_structured_period_sql_uses_published_ratio_formula_without_group_or_top_one():
    intent = comparison_intent("window_primary", "primary", "当前窗口")
    contract = NodePlanContract(
        task_id=intent.plan_task_id,
        preferred_table="fact_window",
        visible_columns=[
            "measure_count",
            "measure_amount",
            "measure_numerator",
            "measure_denominator",
        ],
        metric_specs=intent.metric_specs,
    )
    worker = object.__new__(NodeWorkerExecutor)

    sql = worker._draft_structured_aggregate_sql(
        intent,
        "fact_window",
        {"measure_count", "measure_amount", "measure_numerator", "measure_denominator", "event_day"},
        "",
        contract,
    )

    assert "GROUP BY" not in sql
    assert "ORDER BY" not in sql
    assert "LIMIT 1" not in sql
    assert (
        "SUM(`measure_numerator`) / NULLIF(SUM(`measure_denominator`), 0) AS `measure_ratio`"
        in sql
    )


def test_evidence_and_answer_are_verified_per_metric_spec_and_window_role():
    plan, run = comparison_plan_and_run()

    verified = EvidenceVerifier().verify("最近30天三个指标与前30天相比", plan, run)
    run.verified_evidence = verified
    answer = deterministic_structured_answer("最近30天三个指标与前30天相比", plan, run)

    assert verified.passed
    assert len(verified.derived_evidence) == 6
    assert all(item["covered"] for item in verified.derived_evidence)
    assert "数量指标为 120单" in answer
    assert "对比窗口为 100单" in answer
    assert "上升 20单，+20%" in answer
    assert "金额指标为 250元" in answer
    assert "比例指标为 12%" in answer
    assert "上升 2个百分点，+20%" in answer


def test_metric_spec_verifier_fails_closed_on_alias_source_and_window_role_mismatch():
    plan, run = comparison_plan_and_run()
    primary = run.task_results[0]
    primary.query_bundle.rows[0].pop("measure_ratio")
    primary.query_bundle.rows[0]["__timeWindowRole"] = "comparison"
    primary.node_plan_contract.metric_specs[1]["metricFormula"] = "MAX(measure_amount)"

    verified = EvidenceVerifier().verify("最近30天三个指标与前30天相比", plan, run)
    codes = {gap.code for gap in verified.gaps}

    assert not verified.passed
    assert "MISSING_METRIC_ALIAS" in codes
    assert "METRIC_SOURCE_CONTRACT_MISMATCH" in codes
    assert "TIME_WINDOW_ROLE_MISMATCH" in codes
    assert "MISSING_TIME_WINDOW_EVIDENCE" in codes
