from __future__ import annotations

from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.grounded_execution_policy import (
    GroundedFastPathReason,
    evaluate_single_metric_fast_path,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedQueryContractValidator,
    GroundedRankingBinding,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
)


TOPIC = "topic_alpha"
TABLE = "fact_alpha"
TABLE_REF = "semantic:topic_alpha:fact_alpha:detail"
METRIC_REF = "semantic:topic_alpha:fact_alpha:metric:amount"


def _table(
    table: str = TABLE,
    *,
    topic: str = TOPIC,
) -> GroundedTableBinding:
    return GroundedTableBinding(
        topic=topic,
        table=table,
        time_column="event_day",
        merchant_filter_column="tenant_id",
        detail_ref_id="semantic:%s:%s:detail" % (topic, table),
    )


def _metric() -> GroundedMetricBinding:
    return GroundedMetricBinding(
        requested_phrase="amount",
        semantic_ref_id=METRIC_REF,
        topic=TOPIC,
        table=TABLE,
        metric_key="amount",
        formula="SUM(amount_value)",
        source_columns=["amount_value"],
        aggregation_policy="period_rollup",
        time_column="event_day",
        time_semantics={"selectionPolicy": "period_window"},
    )


def _dimension(column: str, *, role: str = "DIMENSION") -> GroundedDimensionBinding:
    return GroundedDimensionBinding(
        requested_phrase=column,
        semantic_ref_id="semantic:topic_alpha:fact_alpha:field:%s" % column,
        topic=TOPIC,
        table=TABLE,
        column=column,
        role=role,
        usage="group_by",
    )


def _contract(
    *,
    query_shape: str = "SCALAR",
    dimensions: list[GroundedDimensionBinding] | None = None,
    ranking: GroundedRankingBinding | None = None,
) -> GroundedQueryContract:
    bound_dimensions = list(dimensions or [])
    refs = [TABLE_REF, METRIC_REF, *(item.semantic_ref_id for item in bound_dimensions)]
    return GroundedQueryContract(
        status="READY",
        question="complex grounded question",
        topics=[TOPIC],
        query_shape=query_shape,
        primary_table=TABLE,
        tables=[_table()],
        metrics=[_metric()],
        dimensions=bound_dimensions,
        ranking=ranking or GroundedRankingBinding(),
        evidence_refs=refs,
        time_range=ResolvedTimeRange(days=30, explicit=True),
    )


def _codes(contract: GroundedQueryContract) -> set[str]:
    return {
        gap.code
        for gap in GroundedQueryContractValidator().validate(contract).gaps
    }


def test_contract_accepts_metric_under_detail_shape() -> None:
    contract = _contract(query_shape="DETAIL")

    result = GroundedQueryContractValidator().validate(contract)

    assert result.valid is True, result.model_dump(by_alias=True)
    assert "DETAIL_METRIC_BINDING_FORBIDDEN" not in _codes(contract)
    assert "DETAIL_PROJECTION_REQUIRED" not in _codes(contract)


def test_contract_allows_multiple_grouping_dimensions_and_ranking_topology() -> None:
    dimensions = [_dimension("category_id"), _dimension("region_code")]
    contract = _contract(
        query_shape="SCALAR",
        dimensions=dimensions,
        ranking=GroundedRankingBinding(
            enabled=True,
            direction="DESC",
            limit=20,
            metric_ref_id=METRIC_REF,
            dimension_ref_id=dimensions[0].semantic_ref_id,
        ),
    )

    validation = GroundedQueryContractValidator().validate(contract)
    fast_path = evaluate_single_metric_fast_path(contract)

    assert validation.valid is True, validation.model_dump(by_alias=True)
    assert fast_path.eligible is False
    assert set(fast_path.reason_codes) >= {
        GroundedFastPathReason.DIMENSIONS_PRESENT.value,
        GroundedFastPathReason.GROUPING_PRESENT.value,
        GroundedFastPathReason.RANKING_PRESENT.value,
    }


def test_entity_filter_is_an_sql_predicate_obligation_not_an_output_shape() -> None:
    identity_ref = "semantic:topic_alpha:fact_alpha:field:entity_id"
    output_ref = "semantic:topic_alpha:fact_alpha:field:published_at"
    contract = GroundedQueryContract(
        status="READY",
        question="look up entity-100 and return its publication time",
        topics=[TOPIC],
        query_shape="ENTITY_LOOKUP",
        primary_table=TABLE,
        tables=[_table()],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id=output_ref,
                topic=TOPIC,
                table=TABLE,
                column="published_at",
            )
        ],
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id=identity_ref,
                topic=TOPIC,
                table=TABLE,
                column="entity_id",
                operator="EQ",
                literal_value="entity-100",
                is_unique_key=True,
                entity_identity="PRIMARY_ENTITY",
                allowed_operators=["EQ"],
                lookup_time_policy={
                    "mode": "identity_lookup",
                    "timeRequired": False,
                },
            )
        ],
        evidence_refs=[TABLE_REF, identity_ref, output_ref],
        time_range=ResolvedTimeRange(days=0, explicit=False),
    )

    result = GroundedQueryContractValidator().validate(contract)

    assert result.valid is True, result.model_dump(by_alias=True)
    assert "ENTITY_FILTER_PROJECTION_REQUIRED" not in _codes(contract)


def test_contract_keeps_multiple_safe_relationships_for_sql_ast_selection() -> None:
    right_topic = "topic_beta"
    right_table = "dim_beta"
    right_ref = "semantic:topic_beta:dim_beta:detail"
    left_output_ref = "semantic:topic_alpha:fact_alpha:field:entity_id"
    right_output_ref = "semantic:topic_beta:dim_beta:field:label"
    relationships = [
        GroundedRelationshipBinding(
            semantic_ref_id="semantic:topic_alpha:relationship:primary_edge",
            topic=TOPIC,
            name="primary_edge",
            left_table=TABLE,
            right_table=right_table,
            join_type="LEFT",
            keys=[["tenant_id", "tenant_id"], ["beta_id", "id"]],
            grain="entity_beta",
            cardinality="MANY_TO_ONE",
            fanout_policy="PRESERVE_LEFT_GRAIN",
        ),
        GroundedRelationshipBinding(
            semantic_ref_id="semantic:topic_alpha:relationship:alternate_edge",
            topic=TOPIC,
            name="alternate_edge",
            left_table=TABLE,
            right_table=right_table,
            join_type="LEFT",
            keys=[["tenant_id", "tenant_id"], ["alternate_beta_id", "id"]],
            grain="entity_alternate_beta",
            cardinality="MANY_TO_ONE",
            fanout_policy="PRESERVE_LEFT_GRAIN",
        ),
    ]
    contract = GroundedQueryContract(
        status="READY",
        question="return the related labels selected by the authored SQL",
        topics=[TOPIC, right_topic],
        query_shape="DETAIL",
        primary_table=TABLE,
        tables=[_table(), _table(right_table, topic=right_topic)],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id=left_output_ref,
                topic=TOPIC,
                table=TABLE,
                column="entity_id",
            ),
            GroundedSelectedFieldBinding(
                semantic_ref_id=right_output_ref,
                topic=right_topic,
                table=right_table,
                column="label",
            ),
        ],
        relationships=relationships,
        evidence_refs=[
            TABLE_REF,
            right_ref,
            left_output_ref,
            right_output_ref,
            *(item.semantic_ref_id for item in relationships),
        ],
        time_range=ResolvedTimeRange(days=30, explicit=True),
    )

    result = GroundedQueryContractValidator().validate(contract)

    assert result.valid is True, result.model_dump(by_alias=True)
    assert "DETAIL_RELATIONSHIP_BINDING_AMBIGUOUS" not in _codes(contract)
    assert "DETAIL_RELATIONSHIP_EXECUTION_PROOF_REQUIRED" not in _codes(contract)


def test_cross_table_semantic_relationship_obligations_are_shape_independent() -> None:
    right_topic = "topic_beta"
    right_table = "dim_beta"
    right_ref = "semantic:topic_beta:dim_beta:detail"
    right_dimension = GroundedDimensionBinding(
        requested_phrase="label",
        semantic_ref_id="semantic:topic_beta:dim_beta:field:label",
        topic=right_topic,
        table=right_table,
        column="label",
        usage="group_by",
    )
    relationship = GroundedRelationshipBinding(
        semantic_ref_id="semantic:topic_alpha:relationship:fact_dim",
        topic=TOPIC,
        name="fact_dim",
        left_table=TABLE,
        right_table=right_table,
        keys=[["tenant_id", "tenant_id"], ["beta_id", "id"]],
        grain="entity_beta",
        cardinality="MANY_TO_ONE",
        fanout_policy="",
    )
    contract = _contract(query_shape="GROUPED", dimensions=[right_dimension])
    contract.topics.append(right_topic)
    contract.tables.append(_table(right_table, topic=right_topic))
    contract.relationships = [relationship]
    contract.evidence_refs.extend([right_ref, relationship.semantic_ref_id])

    result = GroundedQueryContractValidator().validate(contract)

    assert result.valid is False
    assert "RELATIONSHIP_FANOUT_POLICY_REQUIRED" in {
        gap.code for gap in result.gaps
    }


def test_shape_relaxation_does_not_relax_semantic_evidence() -> None:
    contract = _contract(
        query_shape="GROUPED",
        dimensions=[_dimension("category_id"), _dimension("region_code")],
    )
    contract.evidence_refs.remove(METRIC_REF)

    result = GroundedQueryContractValidator().validate(contract)

    assert result.valid is False
    assert "METRIC_EVIDENCE_REF_MISSING" in {gap.code for gap in result.gaps}
