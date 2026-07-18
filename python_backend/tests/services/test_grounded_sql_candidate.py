from __future__ import annotations

from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
)
from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlCandidate,
    GroundedSqlCandidateValidator,
    grounded_query_contract_fingerprint,
)


def _contract() -> GroundedQueryContract:
    table_specs = [
        ("topic_a", "fact_a"),
        ("topic_b", "dim_b"),
        ("topic_c", "dim_c"),
    ]
    tables = [
        GroundedTableBinding(
            topic=topic,
            table=table,
            time_column="event_time" if table == "fact_a" else "",
            merchant_filter_column="tenant_id",
            detail_ref_id="semantic:%s:%s:detail" % (topic, table),
        )
        for topic, table in table_specs
    ]
    field_specs = [
        ("topic_a", "fact_a", "id"),
        ("topic_b", "dim_b", "label"),
        ("topic_c", "dim_c", "score"),
    ]
    fields = [
        GroundedSelectedFieldBinding(
            semantic_ref_id="semantic:%s:%s:field:%s" % (topic, table, column),
            topic=topic,
            table=table,
            column=column,
            output_alias=column,
        )
        for topic, table, column in field_specs
    ]
    entity = GroundedEntityFilterBinding(
        semantic_ref_id="semantic:topic_a:fact_a:field:id",
        topic="topic_a",
        table="fact_a",
        column="id",
        operator="EQ",
        literal_value="entity-100",
        is_unique_key=True,
        allowed_operators=["EQ"],
    )
    group_dimension = GroundedDimensionBinding(
        requested_phrase="group",
        semantic_ref_id="semantic:topic_a:fact_a:field:group_key",
        topic="topic_a",
        table="fact_a",
        column="group_key",
        usage="label",
    )
    relationships = [
        GroundedRelationshipBinding(
            semantic_ref_id="semantic:topic_a:relationship:fact_a_dim_b",
            topic="topic_a",
            name="fact_a_dim_b",
            left_table="fact_a",
            right_table="dim_b",
            join_type="INNER",
            keys=[["tenant_id", "tenant_id"], ["b_id", "id"]],
            cardinality="MANY_TO_ONE",
            fanout_policy="PRESERVE_LEFT_GRAIN",
        ),
        GroundedRelationshipBinding(
            semantic_ref_id="semantic:topic_b:relationship:dim_b_dim_c",
            topic="topic_b",
            name="dim_b_dim_c",
            left_table="dim_b",
            right_table="dim_c",
            join_type="LEFT",
            keys=[["tenant_id", "tenant_id"], ["c_id", "id"]],
            cardinality="MANY_TO_ONE",
            fanout_policy="PRESERVE_LEFT_GRAIN",
        ),
    ]
    evidence_refs = [
        *(item.detail_ref_id for item in tables),
        *(item.semantic_ref_id for item in fields),
        group_dimension.semantic_ref_id,
        *(item.semantic_ref_id for item in relationships),
    ]
    return GroundedQueryContract(
        status="READY",
        question="look up one entity and rank its related records",
        topics=["topic_a", "topic_b", "topic_c"],
        query_shape="ENTITY_LOOKUP",
        primary_table="fact_a",
        tables=tables,
        selected_fields=fields,
        dimensions=[group_dimension],
        entity_filters=[entity],
        relationships=relationships,
        evidence_refs=evidence_refs,
    )


def _metric_contract(*, explicit_time: bool = True) -> GroundedQueryContract:
    detail_ref = "semantic:topic_a:fact_metric:detail"
    metric_ref = "semantic:topic_a:fact_metric:metric:total_amount"
    return GroundedQueryContract(
        status="READY",
        question="total amount for the requested period",
        topics=["topic_a"],
        query_shape="SCALAR",
        primary_table="fact_metric",
        tables=[
            GroundedTableBinding(
                topic="topic_a",
                table="fact_metric",
                time_column="event_date",
                merchant_filter_column="tenant_id",
                detail_ref_id=detail_ref,
            )
        ],
        metrics=[
            GroundedMetricBinding(
                requested_phrase="total amount",
                semantic_ref_id=metric_ref,
                topic="topic_a",
                table="fact_metric",
                metric_key="total_amount",
                formula="SUM(amount)",
                source_columns=["amount"],
                aggregation_policy="period_rollup",
                time_column="event_date",
            )
        ],
        time_range=ResolvedTimeRange(
            kind="absolute",
            start_date="2026-06-01",
            end_date="2026-06-30",
            days=30,
            explicit=explicit_time,
            source="explicit" if explicit_time else "default_days",
        ),
        evidence_refs=[detail_ref, metric_ref],
    )


def _two_table_entity_contract() -> GroundedQueryContract:
    contract = _contract()
    tables = [item for item in contract.tables if item.table in {"fact_a", "dim_b"}]
    selected = [
        item
        for item in contract.selected_fields
        if (item.table, item.column) in {("fact_a", "id"), ("dim_b", "label")}
    ]
    relationships = [contract.relationships[0]]
    evidence_refs = [
        *(item.detail_ref_id for item in tables),
        *(item.semantic_ref_id for item in selected),
        contract.entity_filters[0].semantic_ref_id,
        relationships[0].semantic_ref_id,
    ]
    return contract.model_copy(
        update={
            "topics": ["topic_a", "topic_b"],
            "tables": tables,
            "selected_fields": selected,
            "dimensions": [],
            "relationships": relationships,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
        }
    )


def _fanout_metric_contract(
    *,
    cardinality: str = "ONE_TO_MANY",
    fanout_policy: str = "ALLOW_DECLARED_FANOUT",
) -> GroundedQueryContract:
    metric = _metric_contract()
    dim_detail = GroundedTableBinding(
        topic="topic_b",
        table="dim_b",
        merchant_filter_column="tenant_id",
        detail_ref_id="semantic:topic_b:dim_b:detail",
    )
    dimension = GroundedDimensionBinding(
        requested_phrase="label",
        semantic_ref_id="semantic:topic_b:dim_b:field:label",
        topic="topic_b",
        table="dim_b",
        column="label",
        usage="group_by",
    )
    relationship = GroundedRelationshipBinding(
        semantic_ref_id="semantic:topic_a:relationship:fact_metric_dim_b",
        topic="topic_a",
        name="fact_metric_dim_b",
        left_table="fact_metric",
        right_table="dim_b",
        join_type="INNER",
        keys=[["tenant_id", "tenant_id"], ["b_id", "id"]],
        grain="metric_to_dimension",
        cardinality=cardinality,
        fanout_policy=fanout_policy,
    )
    return metric.model_copy(
        update={
            "topics": ["topic_a", "topic_b"],
            "tables": [*metric.tables, dim_detail],
            "dimensions": [dimension],
            "relationships": [relationship],
            "evidence_refs": [
                *metric.evidence_refs,
                dim_detail.detail_ref_id,
                dimension.semantic_ref_id,
                relationship.semantic_ref_id,
            ],
        }
    )


def _codes(result: object) -> set[str]:
    return {item.code for item in result.gaps}


def test_accepts_cte_window_and_three_table_governed_join() -> None:
    contract = _contract()
    sql = """
        WITH ranked AS (
            SELECT
                a.id,
                a.group_key,
                b.label,
                c.score,
                ROW_NUMBER() OVER (
                    PARTITION BY a.group_key
                    ORDER BY c.score DESC, a.event_time DESC
                ) AS row_num
            FROM fact_a AS a
            JOIN dim_b AS b
              ON a.tenant_id = b.tenant_id
             AND a.b_id = b.id
            LEFT JOIN dim_c AS c
              ON b.tenant_id = c.tenant_id
             AND b.c_id = c.id
            WHERE a.id = 'entity-100'
        )
        SELECT id, group_key, label, score
        FROM ranked
        WHERE row_num = 1
    """

    result = GroundedSqlCandidateValidator().validate(
        GroundedSqlCandidate(
            sql=sql,
            contract_fingerprint=grounded_query_contract_fingerprint(contract),
        ),
        contract,
    )

    assert result.valid is True, result.model_dump(by_alias=True)
    assert result.canonical_sql.startswith("WITH ranked AS")
    assert len(result.ast_fingerprint) == 64
    assert result.referenced_tables == ["dim_b", "dim_c", "fact_a"]
    assert result.relationship_refs == [
        "semantic:topic_a:relationship:fact_a_dim_b",
        "semantic:topic_b:relationship:dim_b_dim_c",
    ]
    assert result.output_columns == ["id", "group_key", "label", "score"]
    assert result.output_lineage["id"] == ["fact_a.id"]
    assert result.output_lineage["group_key"] == ["fact_a.group_key"]
    assert result.output_lineage["label"] == ["dim_b.label"]
    assert result.output_lineage["score"] == ["dim_c.score"]


def test_rejects_unknown_column_even_inside_window() -> None:
    result = GroundedSqlCandidateValidator().validate(
        """
        SELECT a.id,
               ROW_NUMBER() OVER (PARTITION BY a.not_read ORDER BY a.event_time) AS row_num
        FROM fact_a a
        WHERE a.id = 'entity-100'
        """,
        _contract(),
    )

    assert result.valid is False
    assert "SQL_COLUMN_NOT_GROUNDED" in _codes(result)


def test_rejects_join_that_does_not_use_a_bound_relationship_key_set() -> None:
    result = GroundedSqlCandidateValidator().validate(
        """
        SELECT a.id, b.label
        FROM fact_a a
        JOIN dim_b b
          ON a.tenant_id = b.tenant_id
         AND a.id = b.id
        WHERE a.id = 'entity-100'
        """,
        _contract(),
    )

    assert result.valid is False
    assert "SQL_JOIN_NOT_GOVERNED" in _codes(result)


def test_rejects_missing_entity_literal_predicate() -> None:
    result = GroundedSqlCandidateValidator().validate(
        "SELECT a.id FROM fact_a a WHERE a.event_time >= '2026-01-01'",
        _contract(),
    )

    assert result.valid is False
    assert "SQL_ENTITY_PREDICATE_MISSING" in _codes(result)


def test_entity_literal_under_or_does_not_satisfy_obligation() -> None:
    result = GroundedSqlCandidateValidator().validate(
        "SELECT a.id FROM fact_a a WHERE a.id = 'entity-100' OR 1 = 1",
        _contract(),
    )

    assert result.valid is False
    assert "SQL_ENTITY_PREDICATE_MISSING" in _codes(result)


def test_rejects_cartesian_and_llm_authored_tenant_scope() -> None:
    cartesian = GroundedSqlCandidateValidator().validate(
        "SELECT a.id, b.label FROM fact_a a CROSS JOIN dim_b b WHERE a.id = 'entity-100'",
        _contract(),
    )
    tenant = GroundedSqlCandidateValidator().validate(
        "SELECT a.id FROM fact_a a WHERE a.id = 'entity-100' AND a.tenant_id <> 'other'",
        _contract(),
    )

    assert "SQL_CARTESIAN_PRODUCT_FORBIDDEN" in _codes(cartesian)
    assert "SQL_TENANT_SCOPE_AUTHORED_BY_LLM" in _codes(tenant)


def test_rejects_multiple_or_mutating_statements() -> None:
    validator = GroundedSqlCandidateValidator()

    multiple = validator.validate(
        "SELECT id FROM fact_a; SELECT id FROM fact_a",
        _contract(),
    )
    mutation = validator.validate("DELETE FROM fact_a WHERE id = 'entity-100'", _contract())

    assert "SQL_SINGLE_STATEMENT_REQUIRED" in _codes(multiple)
    assert "SQL_READ_ONLY_REQUIRED" in _codes(mutation)


def test_metric_formula_explicit_time_and_final_alias_are_obligations() -> None:
    contract = _metric_contract()
    valid = GroundedSqlCandidateValidator().validate(
        """
        WITH metric_value AS (
          SELECT SUM(f.amount) AS total_amount
          FROM fact_metric f
          WHERE f.event_date >= '2026-06-01'
            AND f.event_date <= '2026-06-30'
        )
        SELECT total_amount FROM metric_value
        """,
        contract,
    )
    wrong_formula = GroundedSqlCandidateValidator().validate(
        """
        SELECT SUM(f.amount * 2) AS total_amount
        FROM fact_metric f
        WHERE f.event_date >= '2026-06-01'
          AND f.event_date <= '2026-06-30'
        """,
        contract,
    )
    missing_time = GroundedSqlCandidateValidator().validate(
        "SELECT SUM(f.amount) AS total_amount FROM fact_metric f",
        contract,
    )
    wrong_alias = GroundedSqlCandidateValidator().validate(
        """
        SELECT SUM(f.amount) AS arbitrary_name
        FROM fact_metric f
        WHERE f.event_date >= '2026-06-01'
          AND f.event_date <= '2026-06-30'
        """,
        contract,
    )

    assert valid.valid is True, valid.model_dump(by_alias=True)
    assert valid.output_columns == ["total_amount"]
    assert valid.output_lineage == {"total_amount": ["fact_metric.amount"]}
    assert "SQL_METRIC_FORMULA_NOT_PRESERVED" in _codes(wrong_formula)
    assert "SQL_TIME_PREDICATE_MISSING" in _codes(missing_time)
    assert "SQL_OUTPUT_BINDING_MISSING" in _codes(wrong_alias)


def test_non_explicit_time_does_not_create_a_time_predicate_obligation() -> None:
    result = GroundedSqlCandidateValidator().validate(
        "SELECT SUM(f.amount) AS total_amount FROM fact_metric f",
        _metric_contract(explicit_time=False),
    )

    assert result.valid is True, result.model_dump(by_alias=True)


def test_final_output_set_aliases_and_metric_expression_fail_closed() -> None:
    contract = _metric_contract()
    base_from = """
        FROM fact_metric f
        WHERE f.event_date >= '2026-06-01'
          AND f.event_date <= '2026-06-30'
    """
    wrapper = GroundedSqlCandidateValidator().validate(
        "SELECT SUM(f.amount) + 1 AS total_amount " + base_from,
        contract,
    )
    case_wrapper = GroundedSqlCandidateValidator().validate(
        "SELECT CASE WHEN SUM(f.amount) > 0 THEN SUM(f.amount) ELSE 0 END AS total_amount "
        + base_from,
        contract,
    )
    extra = GroundedSqlCandidateValidator().validate(
        "SELECT SUM(f.amount) AS total_amount, f.event_date AS extra "
        + base_from
        + " GROUP BY f.event_date",
        contract,
    )
    duplicate = GroundedSqlCandidateValidator().validate(
        "SELECT SUM(f.amount) AS total_amount, SUM(f.amount) AS total_amount "
        + base_from,
        contract,
    )

    assert "SQL_METRIC_OUTPUT_EXPRESSION_MISMATCH" in _codes(wrapper)
    assert "SQL_METRIC_OUTPUT_EXPRESSION_MISMATCH" in _codes(case_wrapper)
    assert "SQL_OUTPUT_SET_MISMATCH" in _codes(extra)
    assert "SQL_OUTPUT_ALIAS_DUPLICATE" in _codes(duplicate)


def test_current_right_join_scan_cannot_borrow_keys_from_previous_alias() -> None:
    result = GroundedSqlCandidateValidator().validate(
        """
        SELECT a.id, current_b.label
        FROM fact_a a
        JOIN dim_b previous_b
          ON a.tenant_id = previous_b.tenant_id
         AND a.b_id = previous_b.id
        JOIN dim_b current_b
          ON a.tenant_id = current_b.tenant_id
         AND a.b_id = previous_b.id
        WHERE a.id = 'entity-100'
        """,
        _two_table_entity_contract(),
    )

    assert result.valid is False
    assert "SQL_JOIN_NOT_GOVERNED" in _codes(result)


def test_set_operation_is_blocked_before_branch_proof_exists() -> None:
    result = GroundedSqlCandidateValidator().validate(
        """
        SELECT a.id, b.label
        FROM fact_a a
        JOIN dim_b b
          ON a.tenant_id = b.tenant_id AND a.b_id = b.id
        WHERE a.id = 'entity-100'
        UNION ALL
        SELECT a.id, b.label
        FROM fact_a a
        JOIN dim_b b
          ON a.tenant_id = b.tenant_id AND a.b_id = b.id
        """,
        _two_table_entity_contract(),
    )

    assert result.valid is False
    assert "SQL_SET_OPERATION_UNPROVEN" in _codes(result)


def test_preaggregation_fanout_and_blocking_policy_are_rejected() -> None:
    sql = """
        SELECT b.label, SUM(f.amount) AS total_amount
        FROM fact_metric f
        JOIN dim_b b
          ON f.tenant_id = b.tenant_id
         AND f.b_id = b.id
        WHERE f.event_date >= '2026-06-01'
          AND f.event_date <= '2026-06-30'
        GROUP BY b.label
    """
    one_to_many = GroundedSqlCandidateValidator().validate(
        sql,
        _fanout_metric_contract(cardinality="ONE_TO_MANY"),
    )
    blocked_policy = GroundedSqlCandidateValidator().validate(
        sql,
        _fanout_metric_contract(
            cardinality="MANY_TO_ONE",
            fanout_policy="FORBID_FANOUT",
        ),
    )

    assert "SQL_METRIC_FANOUT_UNSAFE" in _codes(one_to_many)
    assert "SQL_METRIC_FANOUT_UNSAFE" in _codes(blocked_policy)
