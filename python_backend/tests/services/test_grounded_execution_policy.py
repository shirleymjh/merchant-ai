from __future__ import annotations

import pytest

from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.grounded_execution_policy import (
    GroundedExecutionMode,
    GroundedFastPathReason,
    evaluate_deterministic_execution,
    evaluate_single_metric_fast_path,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedEntityFilterHint,
    GroundedFieldAggregationHint,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRankingBinding,
    GroundedRankingHint,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
    GroundedSelectedFieldHint,
    GroundedTableBinding,
    GroundedUpstreamEntityHint,
)


def eligible_contract() -> GroundedQueryContract:
    topic = "topic_alpha"
    table = "table_alpha"
    return GroundedQueryContract(
        status="READY",
        question="deliberately ignored by the policy",
        topics=[topic],
        query_shape="SCALAR",
        primary_table=table,
        tables=[GroundedTableBinding(topic=topic, table=table)],
        metrics=[
            GroundedMetricBinding(
                requested_phrase="ignored",
                semantic_ref_id="semantic:topic_alpha:table_alpha:metric:metric_alpha",
                topic=topic,
                table=table,
                metric_key="metric_alpha",
                formula="SUM(value_alpha)",
                source_columns=["value_alpha"],
                aggregation_policy="period_rollup",
                metric_grain="entity_day",
                applicable_time_grain="period",
                time_column="event_day",
                time_semantics={
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "calendar",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
                binding_type="published_metric",
            )
        ],
        time_range=ResolvedTimeRange(
            days=30,
            explicit=True,
            window_role="primary",
        ),
    )


def reason_codes(contract: GroundedQueryContract) -> set[str]:
    return set(evaluate_single_metric_fast_path(contract).reason_codes)


def ranked_contract() -> GroundedQueryContract:
    contract = eligible_contract()
    dimension_ref = "semantic:topic_alpha:table_alpha:column:dimension_alpha"
    contract.query_shape = "RANKED"
    contract.dimensions = [
        GroundedDimensionBinding(
            requested_phrase="ignored",
            semantic_ref_id=dimension_ref,
            topic="topic_alpha",
            table="table_alpha",
            column="dimension_alpha",
            usage="group_by",
        )
    ]
    contract.ranking = GroundedRankingBinding(
        enabled=True,
        direction="DESC",
        limit=1,
        metric_ref_id=contract.metrics[0].semantic_ref_id,
        dimension_ref_id=dimension_ref,
    )
    return contract


def entity_lookup_contract() -> GroundedQueryContract:
    contract = eligible_contract()
    entity_ref = "semantic:topic_alpha:table_alpha:field:entity_alpha"
    contract.query_shape = "ENTITY_LOOKUP"
    contract.metrics = []
    contract.selected_fields = [
        GroundedSelectedFieldBinding(
            semantic_ref_id=("semantic:topic_alpha:table_alpha:field:published_alpha"),
            topic="topic_alpha",
            table="table_alpha",
            column="published_alpha",
            output_alias="published_alpha",
        )
    ]
    contract.entity_filters = [
        GroundedEntityFilterBinding(
            semantic_ref_id=entity_ref,
            topic="topic_alpha",
            table="table_alpha",
            column="entity_alpha",
            operator="IN",
            literal_value=["entity-1"],
            entity_identity="entity:alpha",
            allowed_operators=["EQ", "IN"],
            lookup_time_policy={"mode": "global"},
        )
    ]
    contract.time_range.explicit = False
    return contract


def same_table_multi_metric_contract() -> GroundedQueryContract:
    contract = eligible_contract()
    contract.execution_shape = "same_table_multi_metric"
    contract.metrics.append(
        contract.metrics[0].model_copy(
            update={
                "semantic_ref_id": (
                    "semantic:topic_alpha:table_alpha:metric:metric_beta"
                ),
                "metric_key": "metric_beta",
                "formula": "COUNT(DISTINCT value_beta)",
                "source_columns": ["value_beta"],
            },
            deep=True,
        )
    )
    return contract


def grouped_contract(*, trend: bool = False) -> GroundedQueryContract:
    contract = eligible_contract()
    dimension_ref = "semantic:topic_alpha:table_alpha:column:dimension_alpha"
    contract.query_shape = "TREND" if trend else "GROUPED"
    contract.execution_shape = "grouped_metric"
    contract.dimensions = [
        GroundedDimensionBinding(
            requested_phrase="ignored",
            semantic_ref_id=dimension_ref,
            topic="topic_alpha",
            table="table_alpha",
            column="event_day" if trend else "dimension_alpha",
            role="TIME" if trend else "CATEGORY",
            usage="group_by",
        )
    ]
    return contract


def field_aggregation_contract(
    aggregation: str = "COUNT_DISTINCT",
) -> GroundedQueryContract:
    contract = eligible_contract()
    field_ref = "semantic:topic_alpha:table_alpha:column:value_alpha"
    metric = contract.metrics[0]
    metric.semantic_ref_id = field_ref
    metric.source_field_ref_id = field_ref
    metric.binding_type = "field_aggregation"
    metric.field_aggregation = aggregation
    metric.metric_key = (
        "count_distinct_value_alpha"
        if aggregation == "COUNT_DISTINCT"
        else "count_value_alpha"
    )
    metric.formula = (
        "COUNT(DISTINCT `value_alpha`)"
        if aggregation == "COUNT_DISTINCT"
        else "COUNT(`value_alpha`)"
    )
    metric.calculation_capabilities = {
        "allowedAggregations": [aggregation],
        "declaredAggregation": aggregation,
    }
    contract.binding_hints.field_aggregations = [
        GroundedFieldAggregationHint(
            field_ref=field_ref,
            aggregation=aggregation,
        )
    ]
    return contract


def literal_filtered_metric_contract() -> GroundedQueryContract:
    contract = eligible_contract()
    field_ref = "semantic:topic_alpha:table_alpha:column:status_alpha"
    contract.entity_filters = [
        GroundedEntityFilterBinding(
            semantic_ref_id=field_ref,
            topic="topic_alpha",
            table="table_alpha",
            column="status_alpha",
            operator="IN",
            literal_value=["open", "pending"],
            allowed_operators=["EQ", "IN"],
        )
    ]
    contract.binding_hints.entity_filters = [
        GroundedEntityFilterHint(
            field_ref=field_ref,
            operator="IN",
            literal_value=["open", "pending"],
        )
    ]
    return contract


def test_allows_only_one_executable_published_scalar_metric() -> None:
    decision = evaluate_single_metric_fast_path(eligible_contract())

    assert decision.eligible is True
    assert decision.reason_codes == []
    assert decision.reason_details == {}


def test_unified_policy_preserves_scalar_deterministic_mode() -> None:
    decision = evaluate_deterministic_execution(eligible_contract())

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC


@pytest.mark.parametrize("aggregation", ["COUNT", "COUNT_DISTINCT"])
def test_unified_policy_allows_exact_column_allowlisted_count_derivation(
    aggregation: str,
) -> None:
    contract = field_aggregation_contract(aggregation)

    decision = evaluate_deterministic_execution(contract)

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC
    assert decision.reason_codes == []
    assert (
        GroundedFastPathReason.FIELD_AGGREGATION_PRESENT.value
        in evaluate_single_metric_fast_path(contract).reason_codes
    )


def test_field_aggregation_requires_exact_derivation_and_explicit_column_allowlist() -> None:
    unallowlisted = field_aggregation_contract()
    unallowlisted.metrics[0].calculation_capabilities["allowedAggregations"] = []
    tampered = field_aggregation_contract()
    tampered.metrics[0].formula = "SUM(value_alpha)"

    assert GroundedFastPathReason.FIELD_AGGREGATION_NOT_ALLOWLISTED.value in (
        evaluate_deterministic_execution(unallowlisted).reason_codes
    )
    assert GroundedFastPathReason.FIELD_AGGREGATION_BINDING_INVALID.value in (
        evaluate_deterministic_execution(tampered).reason_codes
    )


def test_unified_policy_allows_exact_same_table_literal_metric_filters() -> None:
    contract = literal_filtered_metric_contract()

    decision = evaluate_deterministic_execution(contract)

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC
    assert (
        GroundedFastPathReason.ENTITY_FILTERS_PRESENT.value
        in evaluate_single_metric_fast_path(contract).reason_codes
    )


def test_literal_metric_filter_rejects_hint_drift_and_upstream_dependency() -> None:
    drifted = literal_filtered_metric_contract()
    drifted.binding_hints.entity_filters[0].literal_value = ["closed"]
    upstream = literal_filtered_metric_contract()
    upstream.binding_hints.upstream_entity_bindings = [
        GroundedUpstreamEntityHint(
            entity_set_artifact_id="entity-set-1",
            target_field_ref=upstream.entity_filters[0].semantic_ref_id,
        )
    ]

    assert GroundedFastPathReason.ENTITY_FILTER_HINT_MISMATCH.value in (
        evaluate_deterministic_execution(drifted).reason_codes
    )
    assert GroundedFastPathReason.UPSTREAM_ENTITY_BINDINGS_PRESENT.value in (
        evaluate_deterministic_execution(upstream).reason_codes
    )


def test_allows_fully_grounded_single_table_ranked_shape() -> None:
    decision = evaluate_deterministic_execution(ranked_contract())

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_RANKED
    assert decision.reason_codes == []


def test_allows_ascending_ranked_shape_without_core_sql() -> None:
    contract = ranked_contract()
    contract.ranking.direction = "ASC"

    decision = evaluate_deterministic_execution(contract)

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_RANKED


def test_allows_same_table_ranked_display_fields_with_safe_business_aliases() -> None:
    contract = ranked_contract()
    contract.selected_fields = [
        GroundedSelectedFieldBinding(
            semantic_ref_id="semantic:topic_alpha:table_alpha:field:product_name",
            topic="topic_alpha",
            table="table_alpha",
            column="product_name",
            output_alias="商品名称",
        ),
        GroundedSelectedFieldBinding(
            semantic_ref_id="semantic:topic_alpha:table_alpha:field:brand_name",
            topic="topic_alpha",
            table="table_alpha",
            column="brand_name",
            output_alias="品牌名称",
        ),
    ]

    decision = evaluate_deterministic_execution(contract)

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_RANKED
    assert decision.reason_codes == []


def test_allows_same_table_multi_metric_scalar_without_core_sql() -> None:
    decision = evaluate_deterministic_execution(same_table_multi_metric_contract())

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_MULTI_METRIC
    assert decision.reason_codes == []


def test_allows_simple_grouped_and_temporal_trend_shapes() -> None:
    grouped = evaluate_deterministic_execution(grouped_contract())
    trend = evaluate_deterministic_execution(grouped_contract(trend=True))

    assert grouped.eligible is True
    assert grouped.execution_mode == GroundedExecutionMode.DETERMINISTIC_GROUPED
    assert trend.eligible is True
    assert trend.execution_mode == GroundedExecutionMode.DETERMINISTIC_TREND


def test_grouped_and_trend_allow_multiple_published_metrics_on_same_table() -> None:
    grouped = grouped_contract()
    grouped.metrics = same_table_multi_metric_contract().metrics
    grouped.execution_shape = "same_table_multi_metric"
    trend = grouped_contract(trend=True)
    trend.metrics = same_table_multi_metric_contract().metrics
    trend.execution_shape = "same_table_multi_metric"

    assert evaluate_deterministic_execution(grouped).eligible is True
    assert evaluate_deterministic_execution(trend).eligible is True


def test_trend_rejects_non_temporal_grouping_dimension() -> None:
    contract = grouped_contract(trend=True)
    contract.dimensions[0].role = "CATEGORY"
    contract.dimensions[0].column = "dimension_alpha"

    decision = evaluate_deterministic_execution(contract)

    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert (
        GroundedFastPathReason.TREND_DIMENSION_NOT_TEMPORAL.value
        in decision.reason_codes
    )


def test_metric_shapes_reject_cross_table_metric_or_complex_execution_shape() -> None:
    cross_table = same_table_multi_metric_contract()
    cross_table.metrics[1].table = "table_beta"
    windowed = grouped_contract()
    windowed.execution_shape = "windowed_metric"

    cross_table_decision = evaluate_deterministic_execution(cross_table)
    windowed_decision = evaluate_deterministic_execution(windowed)

    assert GroundedFastPathReason.METRIC_TABLE_MISMATCH.value in (
        cross_table_decision.reason_codes
    )
    assert GroundedFastPathReason.EXECUTION_SHAPE_NOT_DETERMINISTIC.value in (
        windowed_decision.reason_codes
    )


def test_window_formula_cannot_enter_the_deterministic_metric_compiler() -> None:
    contract = eligible_contract()
    contract.metrics[0].formula = (
        "SUM(value_alpha) OVER (PARTITION BY value_alpha)"
    )

    decision = evaluate_deterministic_execution(contract)

    assert (
        GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE.value
        in decision.reason_codes
    )


def test_multi_metric_requires_compatible_time_columns_and_selection_policy() -> None:
    contract = same_table_multi_metric_contract()
    contract.metrics[1].time_column = "other_event_day"
    contract.metrics[1].time_semantics["selectionPolicy"] = "per_time_grain"

    decision = evaluate_deterministic_execution(contract)

    assert set(decision.reason_codes) >= {
        GroundedFastPathReason.METRIC_TIME_COLUMN_INCOMPATIBLE.value,
        GroundedFastPathReason.METRIC_TIME_POLICY_INCOMPATIBLE.value,
    }


def test_ranked_allows_additional_same_table_metrics_but_caps_result_size() -> None:
    contract = ranked_contract()
    contract.metrics = same_table_multi_metric_contract().metrics
    contract.ranking.metric_ref_id = contract.metrics[1].semantic_ref_id

    assert evaluate_deterministic_execution(contract).eligible is True

    contract.ranking.limit = 1001
    decision = evaluate_deterministic_execution(contract)
    assert (
        GroundedFastPathReason.RANKING_LIMIT_EXCEEDS_MAXIMUM.value
        in decision.reason_codes
    )


def test_ranked_policy_is_independent_of_business_lexicon() -> None:
    first = ranked_contract()
    second = ranked_contract()
    second.question = "任意问题文本"
    second.metrics[0].requested_phrase = "任意指标名"
    second.dimensions[0].requested_phrase = "任意维度名"
    second.tables[0].title = "任意表名"

    assert evaluate_deterministic_execution(first) == evaluate_deterministic_execution(second)


def test_ranked_rejects_unsupported_direction_or_unbound_ranking_refs() -> None:
    contract = ranked_contract()
    contract.ranking.direction = "SIDEWAYS"
    contract.ranking.metric_ref_id = "semantic:other:metric"
    contract.ranking.dimension_ref_id = "semantic:other:dimension"

    decision = evaluate_deterministic_execution(contract)

    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert set(decision.reason_codes) >= {
        GroundedFastPathReason.RANKING_DIRECTION_NOT_SUPPORTED.value,
        GroundedFastPathReason.RANKING_METRIC_NOT_BOUND.value,
        GroundedFastPathReason.RANKING_DIMENSION_NOT_BOUND.value,
    }


def test_ranked_rejects_multi_table_or_relationship_execution() -> None:
    contract = ranked_contract()
    contract.tables.append(GroundedTableBinding(topic="topic_beta", table="table_beta"))
    contract.relationships.append(
        GroundedRelationshipBinding(
            semantic_ref_id="semantic:topic_alpha:relationships:r1",
            topic="topic_alpha",
            name="r1",
            left_table="table_alpha",
            right_table="table_beta",
        )
    )

    decision = evaluate_deterministic_execution(contract)

    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert set(decision.reason_codes) >= {
        GroundedFastPathReason.TABLE_COUNT_NOT_ONE.value,
        GroundedFastPathReason.RELATIONSHIPS_PRESENT.value,
    }


def test_allows_single_table_entity_lookup_with_typed_filter() -> None:
    decision = evaluate_deterministic_execution(entity_lookup_contract())

    assert decision.eligible is True
    assert decision.execution_mode == GroundedExecutionMode.DETERMINISTIC_ENTITY_LOOKUP
    assert decision.reason_codes == []


def test_entity_lookup_does_not_require_unique_key_when_identity_is_typed() -> None:
    contract = entity_lookup_contract()
    assert contract.entity_filters[0].is_unique_key is False

    assert evaluate_deterministic_execution(contract).eligible is True


def test_entity_lookup_rejects_cross_table_or_undeclared_filter_operator() -> None:
    contract = entity_lookup_contract()
    contract.selected_fields[0].table = "table_beta"
    contract.entity_filters[0].operator = "GT"

    decision = evaluate_deterministic_execution(contract)

    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert set(decision.reason_codes) >= {
        GroundedFastPathReason.SELECTED_FIELD_BINDING_INVALID.value,
        GroundedFastPathReason.ENTITY_FILTER_OPERATOR_NOT_DECLARED.value,
    }


def test_entity_lookup_rejects_untyped_or_empty_in_filter() -> None:
    contract = entity_lookup_contract()
    contract.entity_filters[0].entity_identity = ""
    contract.entity_filters[0].literal_value = []

    decision = evaluate_deterministic_execution(contract)

    assert set(decision.reason_codes) >= {
        GroundedFastPathReason.TYPED_ENTITY_FILTER_MISSING.value,
        GroundedFastPathReason.ENTITY_FILTER_BINDING_INVALID.value,
    }


def test_entity_lookup_rejects_more_values_than_deterministic_result_bound() -> None:
    contract = entity_lookup_contract()
    contract.entity_filters[0].literal_value = list(range(101))

    decision = evaluate_deterministic_execution(contract)

    assert (
        GroundedFastPathReason.ENTITY_FILTER_VALUE_LIMIT_EXCEEDED.value
        in decision.reason_codes
    )


def test_entity_lookup_with_upstream_dependency_remains_core_owned() -> None:
    contract = entity_lookup_contract()
    contract.binding_hints.upstream_entity_bindings = [
        GroundedUpstreamEntityHint(
            entity_set_artifact_id="verified-entity-set",
            target_field_ref=contract.entity_filters[0].semantic_ref_id,
        )
    ]

    decision = evaluate_deterministic_execution(contract)

    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert GroundedFastPathReason.UPSTREAM_ENTITY_BINDINGS_PRESENT.value in (
        decision.reason_codes
    )


def test_unknown_shapes_remain_core_sql_required() -> None:
    contract = eligible_contract()
    contract.query_shape = "MULTI_TABLE"

    decision = evaluate_deterministic_execution(contract)

    assert decision.eligible is False
    assert decision.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert decision.reason_codes == [GroundedFastPathReason.QUERY_SHAPE_NOT_DETERMINISTIC.value]


def test_does_not_read_question_metric_or_table_names() -> None:
    first = eligible_contract()
    second = eligible_contract().model_copy(deep=True)
    second.question = "完全不同的自然语言，包括窗口、排行和任意关键词"
    second.metrics[0].requested_phrase = "任意展示名"
    second.metrics[0].business_name = "另一个指标展示名"
    second.tables[0].title = "任意表展示名"

    assert evaluate_single_metric_fast_path(first) == evaluate_single_metric_fast_path(second)


def test_rejects_non_ready_non_scalar_and_wrong_cardinality() -> None:
    contract = eligible_contract()
    contract.status = "UNRESOLVED"
    contract.query_shape = "GROUPED"
    contract.tables.append(GroundedTableBinding(topic="topic_beta", table="table_beta"))
    contract.metrics.append(contract.metrics[0].model_copy(deep=True))

    assert reason_codes(contract) >= {
        GroundedFastPathReason.CONTRACT_NOT_READY.value,
        GroundedFastPathReason.QUERY_SHAPE_NOT_SCALAR.value,
        GroundedFastPathReason.TABLE_COUNT_NOT_ONE.value,
        GroundedFastPathReason.METRIC_COUNT_NOT_ONE.value,
    }


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda contract: contract.dimensions.append(
                GroundedDimensionBinding(
                    requested_phrase="ignored",
                    semantic_ref_id="semantic:topic_alpha:table_alpha:column:dimension_alpha",
                    topic="topic_alpha",
                    table="table_alpha",
                    column="dimension_alpha",
                    usage="label",
                )
            ),
            GroundedFastPathReason.DIMENSIONS_PRESENT,
        ),
        (
            lambda contract: setattr(
                contract.binding_hints,
                "group_by_ref",
                "semantic:topic_alpha:table_alpha:column:dimension_alpha",
            ),
            GroundedFastPathReason.GROUPING_PRESENT,
        ),
        (
            lambda contract: setattr(
                contract,
                "ranking",
                GroundedRankingBinding(
                    enabled=True,
                    direction="DESC",
                    limit=10,
                ),
            ),
            GroundedFastPathReason.RANKING_PRESENT,
        ),
        (
            lambda contract: contract.selected_fields.append(
                GroundedSelectedFieldBinding(
                    semantic_ref_id="semantic:topic_alpha:table_alpha:field:field_alpha",
                    topic="topic_alpha",
                    table="table_alpha",
                    column="field_alpha",
                )
            ),
            GroundedFastPathReason.SELECTED_FIELDS_PRESENT,
        ),
        (
            lambda contract: contract.entity_filters.append(
                GroundedEntityFilterBinding(
                    semantic_ref_id="semantic:topic_alpha:table_alpha:field:id_alpha",
                    topic="topic_alpha",
                    table="table_alpha",
                    column="id_alpha",
                    operator="EQ",
                    literal_value="id-1",
                )
            ),
            GroundedFastPathReason.ENTITY_FILTERS_PRESENT,
        ),
        (
            lambda contract: contract.relationships.append(
                GroundedRelationshipBinding(
                    semantic_ref_id="semantic:topic_alpha:relationships:r1",
                    topic="topic_alpha",
                    name="r1",
                    left_table="table_alpha",
                    right_table="table_beta",
                )
            ),
            GroundedFastPathReason.RELATIONSHIPS_PRESENT,
        ),
    ],
)
def test_rejects_any_non_scalar_query_structure(mutate, expected) -> None:
    contract = eligible_contract()
    mutate(contract)

    assert expected.value in reason_codes(contract)


def test_rejects_grouping_dimension_with_both_dimension_reasons() -> None:
    contract = eligible_contract()
    contract.dimensions.append(
        GroundedDimensionBinding(
            requested_phrase="ignored",
            semantic_ref_id="semantic:topic_alpha:table_alpha:column:dimension_alpha",
            topic="topic_alpha",
            table="table_alpha",
            column="dimension_alpha",
            usage="group_by",
        )
    )

    assert reason_codes(contract) >= {
        GroundedFastPathReason.DIMENSIONS_PRESENT.value,
        GroundedFastPathReason.GROUPING_PRESENT.value,
    }


def test_fails_closed_on_unresolved_complexity_left_in_structured_hints() -> None:
    contract = eligible_contract()
    hints = contract.binding_hints
    hints.dimension_refs = ["semantic:topic_alpha:table_alpha:column:dimension_alpha"]
    hints.selected_fields = [GroundedSelectedFieldHint(field_ref="semantic:topic_alpha:table_alpha:field:field_alpha")]
    hints.entity_filters = [
        GroundedEntityFilterHint(
            field_ref="semantic:topic_alpha:table_alpha:field:id_alpha",
            literal_value="id-1",
        )
    ]
    hints.relationship_refs = ["semantic:topic_alpha:relationships:r1"]
    hints.field_aggregations = [
        GroundedFieldAggregationHint(
            field_ref="semantic:topic_alpha:table_alpha:field:value_alpha",
            aggregation="SUM",
        )
    ]
    hints.ranking = GroundedRankingHint(order="DESC", limit=10)

    assert reason_codes(contract) >= {
        GroundedFastPathReason.DIMENSIONS_PRESENT.value,
        GroundedFastPathReason.SELECTED_FIELDS_PRESENT.value,
        GroundedFastPathReason.ENTITY_FILTERS_PRESENT.value,
        GroundedFastPathReason.RELATIONSHIPS_PRESENT.value,
        GroundedFastPathReason.FIELD_AGGREGATION_PRESENT.value,
        GroundedFastPathReason.RANKING_PRESENT.value,
    }


def test_rejects_field_aggregation_even_if_it_has_an_executable_formula() -> None:
    contract = eligible_contract()
    metric = contract.metrics[0]
    metric.binding_type = "field_aggregation"
    metric.field_aggregation = "COUNT_DISTINCT"
    metric.source_field_ref_id = "semantic:topic_alpha:table_alpha:field:id_alpha"
    metric.formula = "COUNT(DISTINCT id_alpha)"
    metric.source_columns = ["id_alpha"]

    assert reason_codes(contract) >= {
        GroundedFastPathReason.FIELD_AGGREGATION_PRESENT.value,
        GroundedFastPathReason.METRIC_NOT_PUBLISHED.value,
    }


def test_rejects_non_published_metric_identity() -> None:
    contract = eligible_contract()
    contract.metrics[0].semantic_ref_id = "runtime:invented:metric"

    assert GroundedFastPathReason.METRIC_NOT_PUBLISHED.value in reason_codes(contract)


def test_rejects_metric_owned_by_another_table() -> None:
    contract = eligible_contract()
    contract.metrics[0].table = "table_beta"

    assert GroundedFastPathReason.METRIC_TABLE_MISMATCH.value in reason_codes(contract)


@pytest.mark.parametrize(
    "time_range",
    [
        ResolvedTimeRange(days=30, window_role="comparison"),
        ResolvedTimeRange(days=30, comparison_type="previous_period"),
        ResolvedTimeRange(days=30, offset_days=-30),
    ],
)
def test_rejects_secondary_or_comparison_window_semantics(
    time_range: ResolvedTimeRange,
) -> None:
    contract = eligible_contract()
    contract.time_range = time_range

    assert GroundedFastPathReason.MULTI_WINDOW_PRESENT.value in reason_codes(contract)


def test_rejects_metric_without_declared_source_columns() -> None:
    contract = eligible_contract()
    contract.metrics[0].formula = "COUNT(*)"
    contract.metrics[0].source_columns = []

    assert GroundedFastPathReason.METRIC_SOURCE_COLUMNS_MISSING.value in reason_codes(contract)


def test_rejects_formula_that_cannot_compile_against_declared_sources() -> None:
    contract = eligible_contract()
    contract.metrics[0].formula = "SUM(undeclared_column)"

    assert GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE.value in reason_codes(contract)


def test_rejects_unsupported_time_semantics_with_machine_readable_detail() -> None:
    contract = eligible_contract()
    contract.metrics[0].time_semantics["selectionPolicy"] = "latest_as_of"

    decision = evaluate_single_metric_fast_path(contract)

    code = GroundedFastPathReason.METRIC_TIME_SEMANTICS_NOT_EXECUTABLE.value
    assert decision.eligible is False
    assert code in decision.reason_codes
    assert "conflicts" in decision.reason_details[code]
