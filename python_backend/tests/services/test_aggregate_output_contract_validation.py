from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionIntent,
    QueryPlan,
    TaskRole,
)
from merchant_ai.services.planning import QueryGraphValidator, compiled_metric_intent


def aggregate_pack() -> PlanningAssetPack:
    table = "table_alpha"
    columns = [
        "time_axis",
        "group_axis",
        "group_label",
        "transfer_key",
        "metric_value",
        "unrelated_value",
    ]
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table=table,
                columns=columns,
                metadata={"timeColumn": "time_axis"},
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="time_axis",
                table=table,
                metadata={"semantic": {"role": "TIME"}},
            ),
            PlanningAssetEntry(
                key="group_axis",
                table=table,
                metadata={"entityGrain": "entity_alpha"},
            ),
            PlanningAssetEntry(
                key="group_label",
                table=table,
                metadata={"entityGrain": "entity_alpha"},
            ),
            PlanningAssetEntry(key="transfer_key", table=table),
            PlanningAssetEntry(key="metric_value", table=table),
            PlanningAssetEntry(
                key="unrelated_value",
                table=table,
                metadata={
                    "semantic": {
                        "role": "ATTRIBUTE",
                        "defaultVisible": True,
                        "visibilityPolicy": {"level": "public"},
                    }
                },
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="metric_alpha",
                table=table,
                columns=["metric_value"],
                metadata={
                    "formula": "SUM(metric_value)",
                    "sourceColumns": ["metric_value"],
                },
            )
        ],
    )


def aggregate_intent(**updates: object) -> QuestionIntent:
    base = QuestionIntent(
        question="neutral aggregate",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="aggregate_alpha",
        task_role=TaskRole.ANCHOR,
        preferred_table="table_alpha",
        metric_column="metric_value",
        metric_name="metric_alpha",
        metric_formula="SUM(metric_value)",
        group_by_column="time_axis",
        required_evidence=["time_axis", "metric_value"],
        output_keys=["time_axis", "metric_value", "unrelated_value"],
    )
    return base.model_copy(update=updates)


def mismatch_gaps(plan: QueryPlan, pack: PlanningAssetPack):
    result = QueryGraphValidator().validate("neutral aggregate", plan, pack)
    return result, [gap for gap in result.gaps if gap.code == "AGGREGATE_OUTPUT_CONTRACT_MISMATCH"]


def group_contract_gaps(plan: QueryPlan, pack: PlanningAssetPack):
    result = QueryGraphValidator().validate("neutral aggregate", plan, pack)
    return result, [gap for gap in result.gaps if gap.code == "GROUP_BY_CONTRACT_MISMATCH"]


def test_validator_rejects_uncontracted_physical_aggregate_outputs() -> None:
    result, gaps = mismatch_gaps(QueryPlan(intents=[aggregate_intent()]), aggregate_pack())

    assert result.repairable
    assert len(gaps) == 1
    assert gaps[0].task_id == "aggregate_alpha"
    assert gaps[0].evidence == "unrelated_value"


def test_validator_allows_metric_group_time_and_same_grain_outputs() -> None:
    pack = aggregate_pack()
    clean = aggregate_intent(output_keys=["time_axis", "metric_value"])
    same_grain = aggregate_intent(
        plan_task_id="aggregate_by_entity",
        group_by_column="group_axis",
        output_keys=["group_axis", "group_label", "metric_value"],
    )

    _, clean_gaps = mismatch_gaps(QueryPlan(intents=[clean]), pack)
    _, same_grain_gaps = mismatch_gaps(QueryPlan(intents=[same_grain]), pack)
    _, clean_group_gaps = group_contract_gaps(QueryPlan(intents=[clean]), pack)
    _, same_grain_group_gaps = group_contract_gaps(QueryPlan(intents=[same_grain]), pack)

    assert not clean_gaps
    assert not same_grain_gaps
    assert not clean_group_gaps
    assert not same_grain_group_gaps


def test_validator_rejects_metric_or_undeclared_grouping_roles() -> None:
    pack = aggregate_pack()
    metric_group = aggregate_intent(
        plan_task_id="metric_group",
        group_by_column="metric_value",
        output_keys=["metric_value"],
    )
    undeclared_group = aggregate_intent(
        plan_task_id="undeclared_group",
        group_by_column="transfer_key",
        output_keys=["transfer_key", "metric_value"],
    )

    result, gaps = group_contract_gaps(
        QueryPlan(intents=[metric_group, undeclared_group]),
        pack,
    )

    assert result.repairable
    assert {gap.task_id for gap in gaps} == {"metric_group", "undeclared_group"}
    assert next(gap for gap in gaps if gap.task_id == "metric_group").evidence.endswith(":METRIC")
    assert next(gap for gap in gaps if gap.task_id == "undeclared_group").evidence.endswith(":UNDECLARED")


def test_validator_allows_an_explicit_dependency_output_key() -> None:
    pack = aggregate_pack()
    anchor = aggregate_intent(output_keys=["time_axis", "transfer_key"])
    dependent = QuestionIntent(
        question="neutral dependent",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="dependent_alpha",
        task_role=TaskRole.DEPENDENT,
        preferred_table="table_alpha",
        metric_column="metric_value",
        metric_name="metric_alpha",
        metric_formula="SUM(metric_value)",
        group_by_column="time_axis",
        depends_on_task_ids=["aggregate_alpha"],
        output_keys=["time_axis", "transfer_key"],
    )
    plan = QueryPlan(
        intents=[anchor, dependent],
        dependencies=[
            PlanDependency(
                anchor_task_id="aggregate_alpha",
                dependent_task_id="dependent_alpha",
                join_key="transfer_key",
                anchor_column="transfer_key",
                dependent_column="transfer_key",
            )
        ],
    )

    _, gaps = mismatch_gaps(plan, pack)

    assert not gaps


def test_metric_compiler_does_not_promote_default_display_fields_to_aggregate_keys() -> None:
    pack = aggregate_pack()

    intent = compiled_metric_intent(
        question="neutral aggregate",
        metric=pack.metrics[0],
        task_id="compiled_alpha",
        role=TaskRole.ANCHOR,
        mode=AnswerMode.GROUP_AGG,
        grain="time",
        group_by="time_axis",
        depends_on=[],
        limit=20,
        asset_pack=pack,
    )

    assert intent is not None
    assert intent.output_keys == ["time_axis"]
    assert "unrelated_value" not in intent.output_keys
    _, gaps = mismatch_gaps(QueryPlan(intents=[intent]), pack)
    assert not gaps
