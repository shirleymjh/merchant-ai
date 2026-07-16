from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    FreshnessCheckResult,
    IntentType,
    NodeExecutionContext,
    NodePlanContract,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    ResolvedTimeRange,
)
from merchant_ai.services.query import (
    NodeWorkerExecutor,
    SqlValidationService,
    bind_runtime_snapshot_alignment,
    metric_execution_contract_issue,
    same_table_metric_base_key,
)
from merchant_ai.services.semantic_metrics import seal_semantic_metric_resolution


class UnconfiguredLlm:
    configured = False
    last_error = ""
    error_events: list[object] = []


class RepositoryStub:
    pass


def worker() -> NodeWorkerExecutor:
    return NodeWorkerExecutor(
        UnconfiguredLlm(),
        RepositoryStub(),
        SqlValidationService(),
        get_settings(),
    )


def asset_pack() -> PlanningAssetPack:
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="fact_snapshot",
                columns=["tenant_key", "event_day", "measure_value"],
                metadata={"merchantFilterColumn": "tenant_key", "timeColumn": "event_day"},
            )
        ]
    )


def metric_intent(policy: str, *, task_id: str = "measure") -> QuestionIntent:
    return QuestionIntent(
        question="opaque metric question",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.METRIC,
        plan_task_id=task_id,
        preferred_table="fact_snapshot",
        metric_name="measure_value",
        metric_column="measure_value",
        metric_formula="SUM(measure_value)",
        metric_resolution={
            "metricKey": "measure_value",
            "ownerTable": "fact_snapshot",
            "formula": "SUM(measure_value)",
            "sourceColumns": ["measure_value"],
            "aggregationPolicy": policy,
        },
        days=7,
        time_range=ResolvedTimeRange(
            kind="rolling",
            days=7,
            anchor_policy="latest_partition_after_tenant_filter",
            execution_start_date="2026-07-10" if policy == "latest_value_only" else "2026-07-04",
            execution_end_date="2026-07-10",
            execution_start_value="2026-07-10" if policy == "latest_value_only" else "2026-07-04",
            execution_end_value="2026-07-10",
            execution_anchor_policy="common_latest_partition",
        ),
    )


def node_contract(policy: str) -> NodePlanContract:
    selection = "latest_as_of" if policy == "latest_value_only" else "period_window"
    start = "2026-07-10" if selection == "latest_as_of" else "2026-07-04"
    return NodePlanContract(
        task_id="measure",
        preferred_table="fact_snapshot",
        allowed_columns=["tenant_key", "event_day", "measure_value"],
        visible_columns=["event_day", "measure_value"],
        required_columns=["tenant_key", "event_day", "measure_value"],
        metric_column="measure_value",
        metric_name="measure_value",
        metric_formula="SUM(measure_value)",
        metric_specs=[
            {
                "metricName": "measure_value",
                "metricColumn": "measure_value",
                "metricFormula": "SUM(measure_value)",
                "sourceColumns": ["measure_value"],
                "aggregationPolicy": policy,
            }
        ],
        merchant_filter_column="tenant_key",
        answer_mode=AnswerMode.METRIC.value,
        metric_resolution={
            "metricKey": "measure_value",
            "ownerTable": "fact_snapshot",
            "formula": "SUM(measure_value)",
            "sourceColumns": ["measure_value"],
            "aggregationPolicy": policy,
        },
        time_window_contract={
            "partitionColumn": "event_day",
            "tenantColumn": "tenant_key",
            "executionStartValue": start,
            "executionEndValue": "2026-07-10",
            "metricAggregationPolicy": policy,
            "timeSelectionPolicy": selection,
        },
    )


def test_latest_value_only_compiles_to_one_as_of_partition() -> None:
    executor = worker()
    intent = metric_intent("latest_value_only")
    contract = node_contract("latest_value_only")

    sql = executor._draft_structured_sql(
        intent,
        asset_pack(),
        NodeExecutionContext(merchant_id="tenant_1"),
        contract,
    )

    assert "SUM(`measure_value`)" in sql
    assert "`event_day` = (SELECT MAX(`event_day`)" in sql
    assert "`event_day` <= '2026-07-10'" in sql
    assert "`tenant_key` = 'tenant_1'" in sql
    assert "BETWEEN '2026-07-10'" not in sql
    validation = executor._contract_scope_validation(
        SqlValidationService().validate(sql, asset_pack()),
        intent,
        sql,
        contract,
    )
    assert validation.valid, validation.model_dump()


def test_latest_value_only_rejects_period_between_sql() -> None:
    executor = worker()
    intent = metric_intent("latest_value_only")
    contract = node_contract("latest_value_only")
    sql = (
        "SELECT SUM(`measure_value`) AS `measure_value` FROM `fact_snapshot` "
        "WHERE `tenant_key` = 'tenant_1' AND `event_day` BETWEEN '2026-07-10' AND '2026-07-10'"
    )

    validation = executor._contract_scope_validation(
        SqlValidationService().validate(sql, asset_pack()),
        intent,
        sql,
        contract,
    )

    assert not validation.valid
    assert validation.error_code == "METRIC_TIME_SEMANTICS_MISMATCH"


def test_period_rollup_keeps_the_bound_window() -> None:
    executor = worker()
    intent = metric_intent("period_rollup")
    contract = node_contract("period_rollup")

    sql = executor._draft_structured_sql(
        intent,
        asset_pack(),
        NodeExecutionContext(merchant_id="tenant_1"),
        contract,
    )

    assert "`event_day` BETWEEN '2026-07-04' AND '2026-07-10'" in sql
    assert "SELECT MAX(`event_day`)" not in sql


def test_runtime_alignment_uses_one_anchor_but_policy_specific_coverage() -> None:
    period = metric_intent("period_rollup", task_id="period")
    snapshot = metric_intent("latest_value_only", task_id="snapshot")
    for intent in (period, snapshot):
        intent.time_range = ResolvedTimeRange(
            kind="rolling",
            days=7,
            anchor_policy="latest_partition_after_tenant_filter",
        )
    plan = QueryPlan(intents=[period, snapshot])
    reports = [
        FreshnessCheckResult(
            task_id="period",
            table="fact_snapshot",
            checked=True,
            status="AVAILABLE",
            time_column="event_day",
            min_time_value="2026-06-01",
            max_time_value="2026-07-10",
        ),
        FreshnessCheckResult(
            task_id="snapshot",
            table="fact_snapshot",
            checked=True,
            status="AVAILABLE",
            time_column="event_day",
            min_time_value="2026-07-11",
            max_time_value="2026-07-11",
        ),
    ]

    alignment = bind_runtime_snapshot_alignment(plan, reports)

    assert alignment.common_anchor_time_value == "2026-07-10"
    assert alignment.status == "ALIGNED_PARTIAL_COVERAGE"
    assert period.time_range.execution_start_date == "2026-07-04"
    assert snapshot.time_range.execution_start_date == "2026-07-10"
    snapshot_source = next(source for source in alignment.sources if source.task_id == "snapshot")
    assert snapshot_source.time_selection_policy == "latest_as_of"
    assert not snapshot_source.coverage_complete


def test_same_table_optimizer_does_not_merge_different_temporal_policies() -> None:
    period = metric_intent("period_rollup", task_id="period")
    snapshot = metric_intent("latest_value_only", task_id="snapshot")

    assert same_table_metric_base_key(period, asset_pack()) != same_table_metric_base_key(snapshot, asset_pack())


def test_metric_time_column_overrides_table_default_through_the_node_contract() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="fact_snapshot",
                columns=["tenant_key", "event_day", "ingest_day", "measure_value"],
                metadata={"merchantFilterColumn": "tenant_key", "timeColumn": "ingest_day"},
            )
        ]
    )
    intent = metric_intent("latest_value_only")
    intent.metric_resolution = seal_semantic_metric_resolution(
        {
            **intent.metric_resolution,
            "semanticRefId": "semantic:test:fact_snapshot:metric:measure_value",
            "timeColumn": "event_day",
        }
    )

    contract = worker()._node_plan_contract(
        intent,
        pack,
        NodeExecutionContext(merchant_id="tenant_1"),
    )

    assert contract.time_window_contract["partitionColumn"] == "event_day"
    assert "event_day" in contract.allowed_columns


def test_published_metric_cannot_execute_with_an_identity_only_seal() -> None:
    resolution = seal_semantic_metric_resolution(
        {
            "semanticRefId": "semantic:test:fact_snapshot:metric:measure_value",
            "metricKey": "measure_value",
            "ownerTable": "fact_snapshot",
            "formula": "SUM(measure_value)",
            "sourceColumns": ["measure_value"],
            "aggregationPolicy": "period_rollup",
        }
    )
    contract = NodePlanContract(
        preferred_table="fact_snapshot",
        allowed_columns=["tenant_key", "event_day", "measure_value"],
        metric_governance_mode="published_semantic",
        metric_resolution=resolution,
    )

    assert metric_execution_contract_issue(contract) == "published metric has no metricGrain execution contract"


def test_published_metric_executes_only_with_the_complete_nested_temporal_contract() -> None:
    resolution = seal_semantic_metric_resolution(
        {
            "semanticRefId": "semantic:test:fact_snapshot:metric:measure_value",
            "metricKey": "measure_value",
            "ownerTable": "fact_snapshot",
            "formula": "SUM(measure_value)",
            "sourceColumns": ["measure_value"],
            "aggregationPolicy": "period_rollup",
            "metricGrain": "tenant_event",
            "applicableTimeGrain": "period",
            "timeColumn": "event_day",
            "timeSemantics": {
                "selectionPolicy": "period_window",
                "asOfPolicy": "latest_available_partition",
                "missingDataPolicy": "disclose_unknown",
                "zeroValuePolicy": "preserve_observed_zero",
            },
        }
    )
    contract = NodePlanContract(
        preferred_table="fact_snapshot",
        allowed_columns=["tenant_key", "event_day", "measure_value"],
        metric_governance_mode="published_semantic",
        metric_resolution=resolution,
    )

    assert metric_execution_contract_issue(contract) == ""
    assert resolution["semanticContract"]["missingValuePolicy"] == "disclose_unknown"
    assert resolution["semanticContract"]["zeroValueMeaning"] == "preserve_observed_zero"
