import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.query import (
    ExecutionGraphPreparationRequired,
    ExecutionGraphValidationError,
    NodeWorkerExecutor,
    SqlValidationService,
    prepare_execution_graph,
)


class UnconfiguredLlm:
    configured = False


class NeverCalledRepository:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("invalid or unprepared execution graphs must not reach Doris")


def governed_metric_intent(metric_key: str, label: str) -> QuestionIntent:
    return QuestionIntent(
        question=label,
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id=metric_key,
        preferred_table="merchant_daily",
        group_by_column="pt",
        metric_column=metric_key,
        metric_name=metric_key,
        metric_formula="SUM(`%s`)" % metric_key,
        output_keys=["pt"],
        required_evidence=["pt", metric_key],
        days=30,
        limit=30,
        sql_strategy="structured_first",
    )


def governed_multi_metric_graph() -> tuple[str, QueryPlan, PlanningAssetPack]:
    question = "最近30天指标甲和指标乙有什么变化？"
    plan = QueryPlan(
        intents=[
            governed_metric_intent("metric_a", "指标甲"),
            governed_metric_intent("metric_b", "指标乙"),
        ]
    )
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="merchant_daily",
                topic="test",
                columns=["merchant_id", "pt", "metric_a", "metric_b"],
                metadata={"timeColumn": "pt", "timeGrain": "day"},
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="pt",
                table="merchant_daily",
                title="日期",
                metadata={"semanticRole": "TIME", "groupable": True},
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="metric_a",
                table="merchant_daily",
                title="指标甲",
                aliases=["指标甲"],
                columns=["metric_a"],
                metadata={
                    "formula": "SUM(metric_a)",
                    "sourceColumns": ["metric_a"],
                    "unit": "单",
                },
            ),
            PlanningAssetEntry(
                key="metric_b",
                table="merchant_daily",
                title="指标乙",
                aliases=["指标乙"],
                columns=["metric_b"],
                metadata={
                    "formula": "SUM(metric_b)",
                    "sourceColumns": ["metric_b"],
                    "unit": "单",
                },
            ),
        ],
    )
    return question, plan, pack


def test_prepare_execution_graph_is_pure_and_validates_the_merged_graph() -> None:
    question, plan, pack = governed_multi_metric_graph()
    original = plan.model_dump(by_alias=True, mode="json")

    prepared = prepare_execution_graph(question, plan, pack)

    assert plan.model_dump(by_alias=True, mode="json") == original
    assert len(plan.intents) == 2
    assert prepared.changed is True
    assert prepared.executable is True
    assert prepared.validation.valid is True
    assert prepared.validator_name == "QueryGraphValidator"
    assert prepared.source_plan_fingerprint != prepared.execution_plan_fingerprint
    assert len(prepared.plan.intents) == 1
    assert {spec["metricName"] for spec in prepared.plan.intents[0].metric_specs} == {
        "metric_a",
        "metric_b",
    }
    assert any("same_table_metric_merge" in note for note in prepared.optimization_notes)


def test_node_worker_refuses_to_normalize_the_callers_graph_in_place() -> None:
    question, plan, pack = governed_multi_metric_graph()
    original = plan.model_dump(by_alias=True, mode="json")
    repository = NeverCalledRepository()
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        repository,
        SqlValidationService(),
        get_settings(),
    )

    with pytest.raises(ExecutionGraphPreparationRequired):
        worker.execute_plan("100", plan, pack, "", question)

    assert plan.model_dump(by_alias=True, mode="json") == original
    assert repository.calls == 0


def test_invalid_prepared_graph_cannot_reach_node_execution() -> None:
    question, _, pack = governed_multi_metric_graph()
    invalid_plan = QueryPlan(
        intents=[
            governed_metric_intent("missing_metric", "不存在的指标"),
        ]
    )
    prepared = prepare_execution_graph(question, invalid_plan, pack)
    repository = NeverCalledRepository()
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        repository,
        SqlValidationService(),
        get_settings(),
    )

    assert prepared.validation.valid is False
    with pytest.raises(ExecutionGraphValidationError):
        worker.execute_plan(
            "100",
            prepared.plan,
            pack,
            "",
            question,
            execution_preparation=prepared,
        )

    assert repository.calls == 0
