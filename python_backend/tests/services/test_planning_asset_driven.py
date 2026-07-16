from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
)
from merchant_ai.services.planning import (
    EvidenceContractBuilder,
    category_for_table,
    compact_metric_entry,
    metric_table_kind,
    query_plan_question_coverage_gaps,
    select_planner_columns,
    semantic_domain_for_metric,
    semantic_domain_for_table,
    semantic_fast_path_grain,
    semantic_fast_path_group_by,
    semantic_fast_path_requested_grains,
    trend_metric_family,
)


def neutral_asset_pack() -> PlanningAssetPack:
    table = PlanningAssetEntry(
        key="table_alpha",
        table="table_alpha",
        topic="topic_alpha",
        columns=["entity_alpha", "axis_alpha", "value_alpha"],
        metadata={
            "semanticDomain": "domain_alpha",
            "questionCategory": "IDENTITY",
            "timeColumn": "axis_alpha",
            "defaultGroupByColumn": "entity_alpha",
            "dataGrain": "grain_alpha",
        },
    )
    entity = PlanningAssetEntry(
        key="entity_alpha",
        table="table_alpha",
        title="维度甲",
        aliases=["对象甲"],
        metadata={"semanticRole": "ENTITY", "grain": "grain_alpha"},
    )
    axis = PlanningAssetEntry(
        key="axis_alpha",
        table="table_alpha",
        title="时间轴甲",
        aliases=["轴甲"],
        metadata={"semanticRole": "TIME", "grain": "time_alpha"},
    )
    metric = PlanningAssetEntry(
        key="metric_alpha",
        table="table_alpha",
        title="测量甲",
        aliases=["指标甲"],
        columns=["value_alpha"],
        metadata={
            "formula": "SUM(value_alpha)",
            "sourceColumns": ["value_alpha"],
            "aggregationPolicy": "period_rollup",
            "semanticRole": "measure",
        },
    )
    return PlanningAssetPack(
        tables=[table],
        fields=[entity, axis],
        entity_keys=[entity],
        metrics=[metric],
    )


def test_namespace_and_category_come_only_from_asset_metadata():
    pack = neutral_asset_pack()
    metric = pack.metrics[0]

    assert semantic_domain_for_table("table_alpha", pack) == "domain_alpha"
    assert semantic_domain_for_metric(metric, pack) == "domain_alpha"
    assert category_for_table("table_alpha", pack) == QuestionCategory.IDENTITY

    opaque_pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(key="table_omega", table="table_omega")]
    )
    assert semantic_domain_for_table("table_omega", opaque_pack) == ""
    assert category_for_table("table_omega", opaque_pack) == QuestionCategory.UNKNOWN
    assert category_for_table("table_not_declared", pack) == QuestionCategory.UNKNOWN


def test_fast_path_grouping_and_grain_follow_field_aliases_and_metadata():
    pack = neutral_asset_pack()
    metric = pack.metrics[0]
    question = "按对象甲查看指标甲"

    assert semantic_fast_path_group_by(question, metric, pack) == "entity_alpha"
    assert semantic_fast_path_requested_grains(question, pack) == ["grain_alpha"]
    assert semantic_fast_path_grain(question, "entity_alpha", pack, "table_alpha") == "grain_alpha"


def test_temporal_metric_family_uses_declared_variant_links_not_key_similarity():
    period = PlanningAssetEntry(
        key="metric_alpha_period",
        table="table_alpha",
        metadata={
            "temporalVariants": {"series": {"metricKey": "metric_alpha_series"}}
        },
    )
    series = PlanningAssetEntry(
        key="metric_alpha_series",
        table="table_alpha",
        metadata={"temporalVariants": {"summary": "metric_alpha_period"}},
    )
    similar_but_unlinked = PlanningAssetEntry(
        key="metric_alpha_period_extra",
        table="table_alpha",
    )
    pack = PlanningAssetPack(metrics=[period, series, similar_but_unlinked])

    assert trend_metric_family(period, pack) == trend_metric_family(series, pack)
    assert trend_metric_family(period, pack) != trend_metric_family(similar_but_unlinked, pack)


def test_planner_column_selection_uses_declared_columns_and_schema_labels():
    columns = ["noise_alpha", "value_zeta", "axis_tau", "key_omega"]
    selected = select_planner_columns(
        columns,
        "signal zeta",
        {
            "primaryKey": "key_omega",
            "timeColumn": "axis_tau",
            "schemaColumns": [
                {
                    "columnName": "value_zeta",
                    "businessName": "signal zeta",
                    "aliases": ["measure zeta"],
                }
            ],
        },
    )

    assert selected[:3] == ["key_omega", "axis_tau", "value_zeta"]


def test_structured_metric_coverage_is_asset_identity_driven():
    pack = neutral_asset_pack()
    understanding = {
        "requestedMeasures": [
            {
                "metricRef": "metric_alpha",
                "ownerTable": "table_alpha",
                "sourcePhrase": "指标甲",
            }
        ]
    }
    covered = QueryPlan(
        intents=[
            QuestionIntent(
                question="指标甲",
                intent_type=IntentType.VALID,
                category=QuestionCategory.IDENTITY,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="task_alpha",
                preferred_table="table_alpha",
                metric_name="metric_alpha",
            )
        ],
        question_understanding=understanding,
    )
    missing = QueryPlan(question_understanding=understanding)

    assert query_plan_question_coverage_gaps("指标甲", covered, pack) == []
    assert {
        gap.code for gap in query_plan_question_coverage_gaps("指标甲", missing, pack)
    } == {"QUESTION_METRIC_NOT_COVERED"}


def test_table_kind_is_not_inferred_from_an_opaque_name():
    opaque = PlanningAssetEntry(table="table_delta")
    declared = PlanningAssetEntry(
        table="table_delta", metadata={"tableKind": "kind_delta"}
    )

    assert metric_table_kind(opaque) == "semantic_table"
    assert metric_table_kind(declared) == "kind_delta"


def test_evidence_contract_does_not_invent_metric_name_from_column_shape():
    intent = QuestionIntent(
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.METRIC,
        plan_task_id="task_alpha",
        preferred_table="table_alpha",
        metric_column="value_alpha",
        metric_resolution={"outputAliases": ["alias_alpha"]},
    )
    builder = EvidenceContractBuilder()

    contract = builder.contracts_from_intents([intent])[0]

    assert contract["semanticLabel"] == "value_alpha"
    assert contract["columns"] == ["value_alpha"]
    assert contract["semanticAliases"] == {
        "value_alpha": ["value_alpha", "alias_alpha"]
    }
    assert builder.final_evidence_labels([intent]) == ["value_alpha"]


def test_compact_metric_preserves_aliases_and_temporal_variant_structure():
    metric = PlanningAssetEntry(
        key="metric_alpha_period",
        table="table_alpha",
        aliases=["指标甲周期"],
        metadata={
            "semanticRole": "measure",
            "aggregationPolicy": "period_rollup",
            "selectionGuidance": "guidance_alpha",
            "temporalVariants": {
                "series": {"metricKey": "metric_alpha_series"}
            },
        },
    )

    compacted = compact_metric_entry(metric)

    assert compacted["aliases"] == ["指标甲周期"]
    assert compacted["metadata"]["semanticRole"] == "measure"
    assert compacted["metadata"]["temporalVariants"] == {
        "series": {"metricKey": "metric_alpha_series"}
    }
