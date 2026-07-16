from __future__ import annotations

import pytest

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    EntityFilterObligation,
    EntityReference,
    NodeExecutionContext,
    NodePlanContract,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.assets import validate_semantic_asset
from merchant_ai.services.planning import (
    compile_entity_detail_graph_from_question_entity,
    compile_entity_detail_graph_from_understanding,
    entity_filter_obligation_validation_gaps,
    entity_reference_from_understanding,
    resolve_entity_reference,
)
from merchant_ai.services.query import (
    NodePlanCritic,
    build_entity_filter_verification_proof,
    entity_filter_result_contract_error,
    sql_has_entity_filter_predicate,
)
from merchant_ai.services.query_sql_binding import bind_node_sql_parameters
from merchant_ai.services.time_semantics import (
    apply_time_window_contract_to_plan,
    resolve_time_window_contract,
)


def entity_asset_pack(
    *,
    time_column: str = "",
    lookup_policy: dict[str, object] | None = None,
) -> PlanningAssetPack:
    """Neutral semantic assets: all matching behavior must come from metadata."""

    table_metadata: dict[str, object] = {
        "semanticDomain": "domain_alpha",
        "questionCategory": "IDENTITY",
        "tableKind": "detail",
    }
    columns = ["order_id", "sub_order_id", "buyer_id", "amount_value"]
    if time_column:
        columns.append(time_column)
        table_metadata["timeColumn"] = time_column
    if lookup_policy is not None:
        table_metadata["entityLookupPolicy"] = lookup_policy

    table = PlanningAssetEntry(
        key="fact_alpha",
        table="fact_alpha",
        title="对象明细甲",
        columns=columns,
        source_ref_id="semantic:domain_alpha:fact_alpha:table",
        metadata=table_metadata,
    )
    entity_keys = [
        PlanningAssetEntry(
            key="order_id",
            table="fact_alpha",
            title="主订单ID",
            aliases=["主订单号", "交易订单编号"],
            source_ref_id="semantic:domain_alpha:fact_alpha:entity:order_id",
            metadata={"semanticRole": "ENTITY", "schema": {"dataType": "VARCHAR"}},
        ),
        PlanningAssetEntry(
            key="sub_order_id",
            table="fact_alpha",
            title="子订单ID",
            aliases=["子订单号", "订单号"],
            source_ref_id="semantic:domain_alpha:fact_alpha:entity:sub_order_id",
            metadata={"semanticRole": "ENTITY", "schema": {"dataType": "STRING"}},
        ),
        PlanningAssetEntry(
            key="buyer_id",
            table="fact_alpha",
            title="买家ID",
            aliases=["买家编号", "用户ID"],
            source_ref_id="semantic:domain_alpha:fact_alpha:entity:buyer_id",
            metadata={"semanticRole": "ENTITY", "schema": {"dataType": "BIGINT"}},
        ),
    ]
    fields = [
        PlanningAssetEntry(
            key="amount_value",
            table="fact_alpha",
            title="数值甲",
            source_ref_id="semantic:domain_alpha:fact_alpha:field:amount_value",
            metadata={
                "semanticRole": "ATTRIBUTE",
                "semantic": {
                    "displayPolicy": {"defaultVisible": True},
                    "visibilityPolicy": {"level": "visible"},
                },
            },
        )
    ]
    return PlanningAssetPack(tables=[table], fields=fields, entity_keys=entity_keys)


def published_entity_asset(policy: object = None) -> dict[str, object]:
    asset: dict[str, object] = {
        "tableName": "fact_alpha",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "order_id", "dataType": "VARCHAR"},
            {"columnName": "event_day", "dataType": "DATE"},
        ],
        "semanticColumns": [
            {
                "columnName": "order_id",
                "semanticRole": "ENTITY_KEY",
                "canonicalEntityRef": "entity:order",
                "comparisonPolicy": "exact",
            }
        ],
    }
    if policy is not None:
        asset["entityLookupPolicy"] = policy
    return asset


def entity_asset_validation_codes(asset: dict[str, object]) -> set[str]:
    return {
        str(item.get("code") or "")
        for item in validate_semantic_asset(asset, [])["errors"]
    }


def test_partitioned_entity_asset_requires_a_published_lookup_policy() -> None:
    assert "ENTITY_LOOKUP_POLICY_UNDECLARED" in entity_asset_validation_codes(
        published_entity_asset()
    )


@pytest.mark.parametrize(
    ("policy", "expected_code"),
    [
        ({"mode": "guess_latest", "timeColumn": "event_day"}, "ENTITY_LOOKUP_POLICY_MODE_INVALID"),
        ({"mode": "bounded_default", "timeColumn": "event_day", "defaultDays": "90d"}, "ENTITY_LOOKUP_DEFAULT_DAYS_INVALID"),
        ({"mode": "bounded_default", "timeColumn": "unknown_day", "defaultDays": 90}, "ENTITY_LOOKUP_TIME_COLUMN_MISSING"),
    ],
)
def test_malformed_entity_lookup_policy_is_blocked_at_publish(
    policy: dict[str, object],
    expected_code: str,
) -> None:
    assert expected_code in entity_asset_validation_codes(published_entity_asset(policy))


def test_valid_entity_lookup_policy_passes_publish_validation() -> None:
    codes = entity_asset_validation_codes(
        published_entity_asset(
            {"mode": "bounded_default", "timeColumn": "event_day", "defaultDays": 90}
        )
    )

    assert not {code for code in codes if code.startswith("ENTITY_LOOKUP_")}


@pytest.mark.parametrize(
    ("question", "expected_field", "expected_values"),
    [
        ("查询主订单号为AB-123的详细明细", "order_id", ["AB-123"]),
        ("子订单ID为SUB_456的记录", "sub_order_id", ["SUB_456"]),
        ("订单号为A-9的详情", "sub_order_id", ["A-9"]),
        ("主订单ID为A-1，B_2的明细", "order_id", ["A-1", "B_2"]),
        ("order_id_100 的详细记录", "order_id", ["order_id_100"]),
    ],
)
def test_entity_reference_resolution_is_alias_and_value_shape_driven(
    question: str,
    expected_field: str,
    expected_values: list[str],
) -> None:
    reference = resolve_entity_reference(question, entity_asset_pack())

    assert reference.status == "resolved"
    assert reference.field == expected_field
    assert reference.values == expected_values
    assert reference.semantic_ref_id


def test_entity_reference_coerces_numeric_values_from_published_schema() -> None:
    reference = resolve_entity_reference("买家编号为00123", entity_asset_pack())

    assert reference.status == "resolved"
    assert reference.field == "buyer_id"
    assert reference.value_type == "integer"
    assert reference.values == [123]
    assert isinstance(reference.values[0], int)


@pytest.mark.parametrize("value", ["ABC", "1.5"])
def test_numeric_entity_reference_rejects_non_integer_literals(value: str) -> None:
    reference = resolve_entity_reference("买家编号为%s" % value, entity_asset_pack())

    assert reference.status == "invalid"
    assert reference.value_type == "integer"


def test_bare_id_fails_closed_when_multiple_entity_types_are_published() -> None:
    reference = resolve_entity_reference("id为12345", entity_asset_pack())

    assert reference.status == "ambiguous"
    assert reference.field == ""
    assert set(reference.candidate_ref_ids) == {
        "semantic:domain_alpha:fact_alpha:entity:order_id",
        "semantic:domain_alpha:fact_alpha:entity:sub_order_id",
        "semantic:domain_alpha:fact_alpha:entity:buyer_id",
    }


def test_same_physical_id_name_across_unrelated_entities_is_ambiguous() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="buyer_fact", columns=["id"]),
            PlanningAssetEntry(table="order_fact", columns=["id"]),
        ],
        entity_keys=[
            PlanningAssetEntry(
                key="id",
                table="buyer_fact",
                title="买家编号",
                aliases=["编号"],
                source_ref_id="semantic:buyer:id",
            ),
            PlanningAssetEntry(
                key="id",
                table="order_fact",
                title="订单编号",
                aliases=["编号"],
                source_ref_id="semantic:order:id",
            ),
        ],
    )

    reference = resolve_entity_reference("id为123", pack)

    assert reference.status == "ambiguous"
    assert set(reference.candidate_ref_ids) == {"semantic:buyer:id", "semantic:order:id"}


def test_placeholder_id_never_becomes_an_executable_plan() -> None:
    pack = entity_asset_pack()
    reference = resolve_entity_reference("主订单ID为XXX的详细明细", pack)
    plan = compile_entity_detail_graph_from_question_entity("主订单ID为XXX的详细明细", pack)

    assert reference.status == "placeholder"
    assert reference.placeholder is True
    assert not plan.intents
    assert plan.clarification_needs
    assert any("真实 ID" in item for item in plan.clarification_needs)


def test_unknown_llm_filter_field_cannot_override_the_semantic_catalog() -> None:
    plan = compile_entity_detail_graph_from_understanding(
        "查询外部对象ID为A-1的明细",
        {
            "analysisIntent": "lookup",
            "filters": [{"field": "external_object_id", "value": "A-1"}],
        },
        entity_asset_pack(),
    )

    assert not plan.intents
    assert plan.entity_filter_obligations
    assert plan.entity_filter_obligations[0].reference.status == "unresolved"


def test_embedded_resolved_entity_reference_is_revalidated_against_assets() -> None:
    reference = entity_reference_from_understanding(
        "查询外部对象ID为A-1的明细",
        {
            "filters": [
                {
                    "entityReference": {
                        "semanticRefId": "invented:external_object_id",
                        "field": "external_object_id",
                        "table": "invented_table",
                        "rawLabel": "外部对象ID",
                        "rawValue": "A-1",
                        "values": ["A-1"],
                        "status": "resolved",
                    }
                }
            ]
        },
        entity_asset_pack(),
    )

    assert reference.status == "unresolved"
    assert reference.field == ""
    assert reference.semantic_ref_id == ""


def test_partitioned_detail_lookup_without_time_policy_requires_clarification() -> None:
    plan = compile_entity_detail_graph_from_question_entity(
        "查询主订单ID为A-1的详细明细",
        entity_asset_pack(time_column="event_day"),
    )

    assert not plan.intents
    assert plan.clarification_needs
    assert "ENTITY_LOOKUP_TIME_REQUIRED" in "|".join(plan.compiler_trace)
    assert plan.entity_filter_obligations[0].status == "pending_time_scope"


def test_explicit_time_scope_binds_the_same_entity_obligation() -> None:
    plan = compile_entity_detail_graph_from_question_entity(
        "最近7天查询主订单ID为A-1的详细明细",
        entity_asset_pack(time_column="event_day"),
    )

    assert len(plan.intents) == 1
    assert plan.intents[0].days == 7
    assert plan.intents[0].filter_column == "order_id"
    assert plan.intents[0].entity_reference.values == ["A-1"]
    assert plan.entity_filter_obligations[0].status == "bound"


def test_global_lookup_policy_does_not_invent_a_rolling_window() -> None:
    plan = compile_entity_detail_graph_from_question_entity(
        "查询主订单ID为A-1的详细明细",
        entity_asset_pack(
            time_column="event_day",
            lookup_policy={"mode": "global", "timeColumn": "event_day"},
        ),
    )

    assert len(plan.intents) == 1
    assert plan.intents[0].days == 0
    assert not plan.clarification_needs


def test_published_bounded_lookup_policy_uses_its_declared_window() -> None:
    plan = compile_entity_detail_graph_from_question_entity(
        "查询主订单ID为A-1的详细明细",
        entity_asset_pack(
            time_column="event_day",
            lookup_policy={
                "mode": "bounded_default",
                "timeColumn": "event_day",
                "defaultDays": 120,
            },
        ),
    )

    assert len(plan.intents) == 1
    assert plan.intents[0].days == 120
    assert plan.intents[0].entity_reference.lookup_time_policy["defaultDays"] == 120


def test_workflow_time_tool_cannot_override_global_entity_lookup_policy() -> None:
    question = "查询主订单ID为A-1的详细明细"
    plan = compile_entity_detail_graph_from_question_entity(
        question,
        entity_asset_pack(
            time_column="event_day",
            lookup_policy={"mode": "global", "timeColumn": "event_day"},
        ),
    )

    applied = apply_time_window_contract_to_plan(
        plan,
        resolve_time_window_contract(question),
    )

    assert applied.intents[0].days == 0
    assert not applied.intents[0].time_range.start_date


def test_workflow_time_tool_preserves_bounded_entity_lookup_policy() -> None:
    question = "查询主订单ID为A-1的详细明细"
    plan = compile_entity_detail_graph_from_question_entity(
        question,
        entity_asset_pack(
            time_column="event_day",
            lookup_policy={
                "mode": "bounded_default",
                "timeColumn": "event_day",
                "defaultDays": 120,
            },
        ),
    )

    applied = apply_time_window_contract_to_plan(
        plan,
        resolve_time_window_contract(question),
    )

    assert applied.intents[0].days == 120
    assert applied.intents[0].time_range.days == 120
    assert applied.intents[0].time_range.source == "entity_lookup_policy"
    assert applied.question_understanding["timeWindowContract"]["primary"]["days"] == 120


def bound_entity_contract(
    values: list[object] | None = None,
    *,
    value_type: str = "string",
    comparison_policy: str = "exact",
) -> NodePlanContract:
    requested = list(values or ["A-1"])
    reference = EntityReference(
        semantic_ref_id="semantic:domain_alpha:fact_alpha:entity:order_id",
        field="order_id",
        table="fact_alpha",
        raw_label="主订单ID",
        raw_value=",".join(str(item) for item in requested),
        values=requested,
        value_type=value_type,
        comparison_policy=comparison_policy,
        status="resolved",
    )
    obligation = EntityFilterObligation(
        obligation_id="entity_filter_alpha",
        task_id="detail_alpha",
        required=True,
        reference=reference,
        status="bound",
    )
    return NodePlanContract(
        task_id="detail_alpha",
        preferred_table="fact_alpha",
        allowed_columns=["order_id", "amount_value"],
        visible_columns=["order_id", "amount_value"],
        required_columns=["order_id"],
        filter_column="order_id",
        filter_values=requested,
        entity_filter_obligations=[obligation],
        output_keys=["order_id", "amount_value"],
        answer_mode=AnswerMode.DETAIL.value,
    )


def test_node_critic_rejects_dropped_or_changed_entity_filter() -> None:
    contract = bound_entity_contract()

    assert NodePlanCritic().review(contract).valid

    dropped = NodePlanCritic().review(
        contract.model_copy(update={"filter_column": "", "filter_values": []})
    )
    changed = NodePlanCritic().review(
        contract.model_copy(update={"filter_values": ["OTHER-9"]})
    )

    assert not dropped.valid
    assert "MISSING_ENTITY_FILTER" in {item["code"] for item in dropped.issues}
    assert not changed.valid
    assert "ENTITY_FILTER_CONTRACT_MISMATCH" in {
        item["code"] for item in changed.issues
    }


def test_graph_repair_cannot_drop_typed_entity_reference_and_keep_only_text() -> None:
    reference = resolve_entity_reference("主订单ID为A-1", entity_asset_pack())
    obligation = EntityFilterObligation(
        obligation_id="entity_filter_alpha",
        task_id="detail_alpha",
        required=True,
        reference=reference,
        status="bound",
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="detail_alpha",
                preferred_table="fact_alpha",
                filter_column="order_id",
                filter_value="A-1",
            )
        ],
        entity_filter_obligations=[obligation],
    )

    gaps = entity_filter_obligation_validation_gaps(plan, entity_asset_pack())

    assert {gap.code for gap in gaps} == {"ENTITY_FILTER_OBLIGATION_NOT_PLANNED"}


def test_sql_must_contain_the_sealed_entity_predicate() -> None:
    assert sql_has_entity_filter_predicate(
        "SELECT order_id, amount_value FROM fact_alpha WHERE order_id = 'A-1'",
        "order_id",
    )
    assert not sql_has_entity_filter_predicate(
        "SELECT order_id, amount_value FROM fact_alpha WHERE buyer_id = 1",
        "order_id",
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT order_id = 'A-1' AS entity_match, amount_value FROM fact_alpha",
        "SELECT order_id, amount_value FROM fact_alpha WHERE NOT order_id = 'A-1'",
        "SELECT order_id, amount_value FROM fact_alpha WHERE order_id = 'A-1' OR 1 = 1",
        (
            "SELECT order_id, amount_value FROM fact_alpha "
            "WHERE order_id IN (SELECT order_id FROM fact_alpha WHERE amount_value > 0)"
        ),
    ],
)
def test_entity_predicate_must_restrict_every_returned_row(sql: str) -> None:
    assert not sql_has_entity_filter_predicate(sql, "order_id")


def executable_entity_intent(values: list[object]) -> QuestionIntent:
    reference = EntityReference(
        semantic_ref_id="semantic:domain_alpha:fact_alpha:entity:order_id",
        field="order_id",
        table="fact_alpha",
        raw_label="主订单ID",
        raw_value=",".join(str(item) for item in values),
        values=values,
        status="resolved",
    )
    return QuestionIntent(
        preferred_table="fact_alpha",
        filter_column="order_id",
        filter_value=reference.raw_value,
        entity_reference=reference,
    )


def test_multi_id_contract_cannot_be_reduced_to_the_first_equality_value() -> None:
    bound_sql, params, error = bind_node_sql_parameters(
        "SELECT order_id, amount_value FROM fact_alpha WHERE order_id = 'draft-value'",
        executable_entity_intent(["A-1", "B_2"]),
        entity_asset_pack(),
        NodeExecutionContext(),
    )

    assert not error
    assert " IN " in bound_sql.upper()
    assert params == ["A-1", "B_2"]
    assert bound_sql.count("%s") == len(params)


def test_entity_filter_value_limit_fails_closed_without_truncation() -> None:
    intent = executable_entity_intent(["A-1", "B_2", "C-3"])
    bound_sql, params, error = bind_node_sql_parameters(
        "SELECT order_id FROM fact_alpha WHERE order_id IN ('draft')",
        intent,
        entity_asset_pack(),
        NodeExecutionContext(),
        max_filter_values=2,
    )

    assert "ENTITY_FILTER_VALUE_LIMIT_EXCEEDED" in error
    assert params == []
    assert bound_sql.count("%s") == 0


def test_node_critic_rejects_entity_values_over_runtime_contract_limit() -> None:
    contract = bound_entity_contract(["A-1", "B_2", "C-3"]).model_copy(
        update={"filter_value_limit": 2}
    )

    critique = NodePlanCritic().review(contract)

    assert not critique.valid
    assert "ENTITY_FILTER_VALUE_LIMIT_EXCEEDED" in {
        item["code"] for item in critique.issues
    }


def test_in_subquery_binding_never_emits_unused_parameters() -> None:
    bound_sql, params, error = bind_node_sql_parameters(
        (
            "SELECT order_id, amount_value FROM fact_alpha "
            "WHERE order_id IN (SELECT order_id FROM fact_alpha WHERE amount_value > 0)"
        ),
        executable_entity_intent(["A-1"]),
        entity_asset_pack(),
        NodeExecutionContext(),
    )

    assert error or bound_sql.count("%s") == len(params)


@pytest.mark.parametrize(
    ("rows", "error_fragment"),
    [
        ([{"order_id": "OTHER-9", "amount_value": 2}], "outside the requested set"),
        ([{"amount_value": 2}], "do not expose the governed entity key"),
    ],
)
def test_result_rows_are_reverse_checked_against_the_requested_id(
    rows: list[dict[str, object]],
    error_fragment: str,
) -> None:
    error = entity_filter_result_contract_error(bound_entity_contract(), rows)

    assert error_fragment in error


def test_string_entity_ids_are_case_sensitive_by_default() -> None:
    error = entity_filter_result_contract_error(
        bound_entity_contract(["AbC"]),
        [{"order_id": "abc", "amount_value": 2}],
    )

    assert "outside the requested set" in error


def test_numeric_entity_ids_use_numeric_equivalence() -> None:
    error = entity_filter_result_contract_error(
        bound_entity_contract([123], value_type="integer", comparison_policy="integer"),
        [{"order_id": 123.0, "amount_value": 2}],
    )

    assert error == ""


def test_case_insensitive_entity_matching_requires_published_policy() -> None:
    error = entity_filter_result_contract_error(
        bound_entity_contract(["AbC"], comparison_policy="case_insensitive"),
        [{"order_id": "abc", "amount_value": 2}],
    )

    assert error == ""


def test_evidence_gate_blocks_a_wrong_result_id() -> None:
    contract = bound_entity_contract()
    rows = [{"order_id": "OTHER-9", "amount_value": 2}]
    sql = "SELECT order_id, amount_value FROM fact_alpha WHERE order_id = %s"
    proof = build_entity_filter_verification_proof(contract, rows, sql)
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="detail_alpha",
                success=True,
                query_bundle=QueryBundle(
                    sql=sql,
                    tables=["fact_alpha"],
                    rows=rows,
                    original_row_count=1,
                ),
                node_plan_contract=contract,
                entity_filter_verification=proof,
            )
        ],
        merged_query_bundle=QueryBundle(
            tables=["fact_alpha"],
            rows=rows,
            original_row_count=1,
        ),
    )

    verified = EvidenceVerifier().verify("主订单ID为A-1的明细", QueryPlan(), run_result)

    assert not verified.passed
    assert "ENTITY_FILTER_RESULT_MISMATCH" in {
        gap.code for gap in verified.blocking_gaps
    }


def test_complete_multi_id_result_discloses_each_missing_id() -> None:
    contract = bound_entity_contract(["A-1", "B_2"])
    rows = [{"order_id": "A-1", "amount_value": 2}]
    sql = "SELECT order_id, amount_value FROM fact_alpha WHERE order_id IN (%s, %s)"
    proof = build_entity_filter_verification_proof(contract, rows, sql)
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="detail_alpha",
                success=True,
                query_bundle=QueryBundle(
                    sql=sql,
                    tables=["fact_alpha"],
                    rows=rows,
                    original_row_count=1,
                ),
                node_plan_contract=contract,
                entity_filter_verification=proof,
            )
        ],
        merged_query_bundle=QueryBundle(
            tables=["fact_alpha"],
            rows=rows,
            original_row_count=1,
        ),
    )

    verified = EvidenceVerifier().verify("查询两个对象", QueryPlan(), run_result)
    warning = next(
        gap for gap in verified.warning_gaps if gap.code == "ENTITY_FILTER_VALUE_NOT_FOUND"
    )

    assert verified.passed
    assert warning.details["missingValues"] == ["B_2"]
    assert warning.disclosure_required is True


def test_global_limit_cannot_prove_a_missing_multi_id_was_queried_completely() -> None:
    contract = bound_entity_contract(["A-1", "B_2"])
    rows = [{"order_id": "A-1", "amount_value": 2}]
    sql = "SELECT order_id, amount_value FROM fact_alpha WHERE order_id IN (%s, %s) LIMIT 20"
    proof = build_entity_filter_verification_proof(contract, rows, sql)
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="detail_alpha",
                success=True,
                query_bundle=QueryBundle(
                    sql=sql,
                    tables=["fact_alpha"],
                    rows=rows,
                    original_row_count=1,
                ),
                node_plan_contract=contract,
                entity_filter_verification=proof,
            )
        ],
        merged_query_bundle=QueryBundle(tables=["fact_alpha"], rows=rows, original_row_count=1),
    )

    verified = EvidenceVerifier().verify("查询两个对象", QueryPlan(), run_result)

    assert proof.status == "partial"
    assert proof.code == "ENTITY_FILTER_COVERAGE_INCOMPLETE"
    assert not verified.passed
    assert "ENTITY_FILTER_COVERAGE_INCOMPLETE" in {
        gap.code for gap in verified.blocking_gaps
    }


def test_evidence_uses_pre_mask_identity_proof_instead_of_display_id() -> None:
    contract = bound_entity_contract(["A-1"])
    raw_rows = [{"order_id": "A-1", "amount_value": 2}]
    display_rows = [{"order_id": "***", "amount_value": 2}]
    sql = "SELECT order_id, amount_value FROM fact_alpha WHERE order_id = %s"
    proof = build_entity_filter_verification_proof(contract, raw_rows, sql)
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="detail_alpha",
                success=True,
                query_bundle=QueryBundle(
                    sql=sql,
                    tables=["fact_alpha"],
                    rows=display_rows,
                    original_row_count=1,
                ),
                node_plan_contract=contract,
                entity_filter_verification=proof,
            )
        ],
        merged_query_bundle=QueryBundle(
            tables=["fact_alpha"],
            rows=display_rows,
            original_row_count=1,
        ),
    )

    verified = EvidenceVerifier().verify("主订单ID为A-1的明细", QueryPlan(), run_result)

    assert verified.passed
    assert not any(gap.code.startswith("ENTITY_FILTER_") for gap in verified.gaps)
