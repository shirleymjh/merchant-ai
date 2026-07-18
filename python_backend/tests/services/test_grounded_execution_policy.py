from __future__ import annotations

import pytest

from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.grounded_execution_policy import (
    GroundedFastPathReason,
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


def test_allows_only_one_executable_published_scalar_metric() -> None:
    decision = evaluate_single_metric_fast_path(eligible_contract())

    assert decision.eligible is True
    assert decision.reason_codes == []
    assert decision.reason_details == {}


def test_does_not_read_question_metric_or_table_names() -> None:
    first = eligible_contract()
    second = eligible_contract().model_copy(deep=True)
    second.question = "完全不同的自然语言，包括窗口、排行和任意关键词"
    second.metrics[0].requested_phrase = "任意展示名"
    second.metrics[0].business_name = "另一个指标展示名"
    second.tables[0].title = "任意表展示名"

    assert evaluate_single_metric_fast_path(first) == evaluate_single_metric_fast_path(
        second
    )


def test_rejects_non_ready_non_scalar_and_wrong_cardinality() -> None:
    contract = eligible_contract()
    contract.status = "UNRESOLVED"
    contract.query_shape = "GROUPED"
    contract.tables.append(
        GroundedTableBinding(topic="topic_beta", table="table_beta")
    )
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
    hints.selected_fields = [
        GroundedSelectedFieldHint(
            field_ref="semantic:topic_alpha:table_alpha:field:field_alpha"
        )
    ]
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

    assert (
        GroundedFastPathReason.METRIC_SOURCE_COLUMNS_MISSING.value
        in reason_codes(contract)
    )


def test_rejects_formula_that_cannot_compile_against_declared_sources() -> None:
    contract = eligible_contract()
    contract.metrics[0].formula = "SUM(undeclared_column)"

    assert (
        GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE.value
        in reason_codes(contract)
    )


def test_rejects_unsupported_time_semantics_with_machine_readable_detail() -> None:
    contract = eligible_contract()
    contract.metrics[0].time_semantics["selectionPolicy"] = "latest_as_of"

    decision = evaluate_single_metric_fast_path(contract)

    code = GroundedFastPathReason.METRIC_TIME_SEMANTICS_NOT_EXECUTABLE.value
    assert decision.eligible is False
    assert code in decision.reason_codes
    assert "conflicts" in decision.reason_details[code]
