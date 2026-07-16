from __future__ import annotations

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    NodeExecutionContext,
    NodePlanContract,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QuestionIntent,
    SemanticFilterNode,
    SemanticFilterObligation,
    SemanticQuerySpec,
)
from merchant_ai.services.evidence import semantic_filter_evidence_gaps
from merchant_ai.services.query import (
    NodeWorkerExecutor,
    build_semantic_filter_verification_proof,
    compile_semantic_filter_sql,
    time_window_sql_contract_error,
)
from merchant_ai.services.query_contracts import tenant_scope_binding_error
from merchant_ai.services.query_sql_binding import bind_node_sql_parameters


def predicate(node_id: str, field: str, values: list, operator: str = "eq", data_type: str = "string") -> SemanticFilterNode:
    return SemanticFilterNode(
        node_id=node_id,
        node_type="predicate",
        semantic_ref_id="semantic:orders:field:%s" % field,
        source_phrase="%s %s" % (field, operator),
        operator=operator,
        raw_values=list(values),
        resolved_values=list(values),
        bound_table="orders",
        bound_field=field,
        member_kind="dimension",
        data_type=data_type,
        resolution_status="resolved",
    )


def obligation(node: SemanticFilterNode) -> SemanticFilterObligation:
    return SemanticFilterObligation(
        obligation_id="obligation_%s" % node.node_id,
        task_id="detail",
        node_id=node.node_id,
        predicate_id=node.node_id,
        semantic_ref_id=node.semantic_ref_id,
        source_phrase=node.source_phrase,
        operator=node.operator,
        raw_values=list(node.raw_values),
        resolved_values=list(node.resolved_values),
        bound_table=node.bound_table,
        bound_field=node.bound_field,
        member_kind=node.member_kind,
        data_type=node.data_type,
        status="bound",
    )


def semantic_contract(node: SemanticFilterNode) -> NodePlanContract:
    return NodePlanContract(
        task_id="detail",
        preferred_table="orders",
        allowed_columns=["tenant_id", "region_id", "store_id", "status", "order_id", "pt"],
        visible_columns=["status", "order_id", "pt"],
        semantic_query=SemanticQuerySpec(
            filter_nodes=[node],
            root_filter_node_id=node.node_id,
            binding_status="resolved",
        ),
        semantic_filter_obligations=[obligation(node)],
        merchant_filter_column="tenant_id",
        merchant_id="T-1",
        answer_mode=AnswerMode.DETAIL.value,
    )


def test_semantic_proof_rejects_extra_constant_and_contradictory_user_predicates() -> None:
    contract = semantic_contract(predicate("status", "status", ["paid"]))
    base = "SELECT * FROM orders WHERE tenant_id = 'T-1' AND status = 'paid'"

    constant = build_semantic_filter_verification_proof(contract, base + " AND 1 = 0")
    contradictory = build_semantic_filter_verification_proof(contract, base + " AND status = 'refunded'")

    assert constant.code == "SEMANTIC_FILTER_EXTRA_PREDICATE"
    assert contradictory.code == "SEMANTIC_FILTER_EXTRA_PREDICATE"
    assert not constant.verified and not contradictory.verified


def test_semantic_proof_audits_outer_cte_and_every_union_output_branch() -> None:
    contract = semantic_contract(predicate("status", "status", ["paid"]))
    legal_cte = "WITH x AS (SELECT * FROM orders WHERE tenant_id='T-1' AND status='paid') SELECT * FROM x"
    narrowed_cte = legal_cte + " WHERE 1=0"
    legal_union = (
        "SELECT * FROM orders WHERE tenant_id='T-1' AND status='paid' "
        "UNION ALL SELECT * FROM orders WHERE tenant_id='T-1' AND status='paid'"
    )
    unsafe_union = legal_union.rsplit("status='paid'", 1)[0] + "status='refunded'"

    assert build_semantic_filter_verification_proof(contract, legal_cte).verified
    assert build_semantic_filter_verification_proof(contract, narrowed_cte).code == "SEMANTIC_FILTER_EXTRA_PREDICATE"
    assert build_semantic_filter_verification_proof(contract, legal_union).verified
    assert not build_semantic_filter_verification_proof(contract, unsafe_union).verified
    self_join = (
        "SELECT o.* FROM orders o JOIN orders x ON 1=0 "
        "WHERE o.tenant_id='T-1' AND o.status='paid'"
    )
    assert build_semantic_filter_verification_proof(contract, self_join).code == "SEMANTIC_FILTER_EXTRA_PREDICATE"


def test_bound_parameter_values_are_part_of_predicate_and_evidence_proof() -> None:
    contract = semantic_contract(predicate("status", "status", ["paid"]))
    sql = "SELECT * FROM orders WHERE tenant_id = %s AND status = %s"
    proof = build_semantic_filter_verification_proof(contract, sql, ["T-1", "paid"])

    assert proof.verified
    assert proof.predicate_proofs[0].parameter_names == ["p1"]
    assert not build_semantic_filter_verification_proof(contract, sql, ["T-1", "refunded"]).verified

    task = AgentTaskResult(
        task_id="detail",
        success=True,
        query_bundle=QueryBundle(sql=sql, params=["T-1", "paid"]),
        node_plan_contract=contract,
        semantic_filter_verification=proof,
    )
    run = AgentRunResult(task_results=[task])
    assert semantic_filter_evidence_gaps(run) == []
    task.query_bundle.params = ["T-1", "refunded"]
    assert [gap.code for gap in semantic_filter_evidence_gaps(run)] == ["SEMANTIC_FILTER_VERIFICATION_MISSING"]


def test_percent_s_inside_governed_string_literal_is_not_treated_as_db_parameter() -> None:
    node = predicate("status", "status", ["A%sB"], "contains")
    contract = semantic_contract(node)
    compiled = compile_semantic_filter_sql(contract)
    sql = "SELECT * FROM orders WHERE tenant_id='T-1' AND %s" % compiled.where_sql

    assert compiled.valid
    assert build_semantic_filter_verification_proof(contract, sql).verified


def test_mandatory_dimension_eq_is_parameterized_end_to_end_for_detail_id() -> None:
    node = predicate("order", "order_id", ["ORDER-XXX"])
    intent = QuestionIntent(
        preferred_table="orders",
        semantic_query=SemanticQuerySpec(
            filter_nodes=[node],
            root_filter_node_id="order",
            binding_status="resolved",
        ),
    )
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="orders",
                columns=["tenant_id", "order_id"],
                metadata={"merchantFilterColumn": "tenant_id"},
            )
        ]
    )
    context = NodeExecutionContext(merchant_id="T-1")
    bound_sql, params, error = bind_node_sql_parameters(
        "SELECT order_id FROM orders WHERE tenant_id='T-1' AND order_id='ORDER-XXX'",
        intent,
        pack,
        context,
    )
    contract = semantic_contract(node)

    assert error == ""
    assert bound_sql.count("%s") == 2
    assert params == ["T-1", "ORDER-XXX"]
    assert build_semantic_filter_verification_proof(contract, bound_sql, params).verified


def test_scope_fields_must_be_exact_root_and_bound_in_every_base_table_scope() -> None:
    contract = semantic_contract(predicate("status", "status", ["paid"]))
    contract.authorized_region = "R-1"
    contract.region_filter_column = "region_id"
    contract.authorized_store_ids = ["S-1", "S-2"]
    contract.store_filter_column = "store_id"
    context = NodeExecutionContext(merchant_id="T-1")
    valid = (
        "SELECT * FROM orders WHERE tenant_id=%s AND region_id=%s AND store_id IN (%s,%s) "
        "AND (status=%s OR status=%s)"
    )

    assert tenant_scope_binding_error(valid, ["T-1", "R-1", "S-1", "S-2", "paid", "refund"], contract, context) == ""
    assert tenant_scope_binding_error(
        "SELECT * FROM orders WHERE tenant_id=%s OR status=%s",
        ["T-1", "paid"],
        contract,
        context,
    )
    assert tenant_scope_binding_error(
        "SELECT * FROM orders WHERE tenant_id=%s AND EXISTS (SELECT 1 FROM orders WHERE status=%s)",
        ["T-1", "paid"],
        contract,
        context,
    )
    assert tenant_scope_binding_error(
        "SELECT * FROM orders WHERE tenant_id=%s AND (region_id=%s OR status=%s) AND store_id IN (%s,%s)",
        ["T-1", "R-1", "paid", "S-1", "S-2"],
        contract,
        context,
    )
    assert tenant_scope_binding_error(
        "SELECT * FROM orders WHERE tenant_id=%s AND region_id=%s AND NOT (store_id IN (%s,%s))",
        ["T-1", "R-1", "S-1", "S-2"],
        contract,
        context,
    )


def test_same_semantic_tenant_predicate_fulfils_scope_once_and_conflict_fails_closed() -> None:
    node = predicate("tenant", "tenant_id", ["T-1"])
    contract = semantic_contract(node)
    intent = QuestionIntent(
        preferred_table="orders",
        semantic_query=contract.semantic_query,
        filter_column="tenant_id",
        filter_value="T-1",
    )
    compiled = compile_semantic_filter_sql(contract)
    where = NodeWorkerExecutor._structured_where(
        object(),
        intent,
        "orders",
        {"tenant_id"},
        NodeExecutionContext(merchant_id="T-1"),
        contract,
        semantic_where_sql=compiled.where_sql,
    )

    assert " AND ".join(where).count("tenant_id") == 1

    conflict = semantic_contract(predicate("tenant", "tenant_id", ["T-2"]))
    assert compile_semantic_filter_sql(conflict).code == "SEMANTIC_SCOPE_FILTER_CONFLICT"


def test_time_window_requires_exact_mandatory_positive_boundaries() -> None:
    contract = semantic_contract(predicate("status", "status", ["paid"]))
    contract.time_window_contract = {
        "partitionColumn": "pt",
        "executionStartValue": "2026-07-04",
        "executionEndValue": "2026-07-10",
        "timeSelectionPolicy": "period_window",
    }
    valid = "SELECT * FROM orders WHERE tenant_id='T-1' AND status='paid' AND pt BETWEEN '2026-07-04' AND '2026-07-10'"
    reversed_window = valid.replace("'2026-07-04' AND '2026-07-10'", "'2026-07-10' AND '2026-07-04'")
    or_bypass = "SELECT * FROM orders WHERE tenant_id='T-1' AND status='paid' OR pt BETWEEN '2026-07-04' AND '2026-07-10'"

    assert time_window_sql_contract_error(valid, contract) == ("", "")
    assert time_window_sql_contract_error(reversed_window, contract)[0] == "RUNTIME_TIME_ALIGNMENT_MISMATCH"
    assert time_window_sql_contract_error(or_bypass, contract)[0] == "RUNTIME_TIME_ALIGNMENT_MISMATCH"
    assert build_semantic_filter_verification_proof(contract, valid).verified


def test_semantic_explicit_date_filter_must_equal_runtime_window_and_is_emitted_once() -> None:
    time_node = predicate("time", "pt", ["2026-07-04", "2026-07-10"], "between", "date")
    contract = semantic_contract(time_node)
    contract.time_window_contract = {
        "partitionColumn": "pt",
        "executionStartValue": "2026-07-04",
        "executionEndValue": "2026-07-10",
        "timeSelectionPolicy": "period_window",
    }
    compiled = compile_semantic_filter_sql(contract)
    intent = QuestionIntent(preferred_table="orders", semantic_query=contract.semantic_query)
    where = NodeWorkerExecutor._structured_where(
        object(),
        intent,
        "orders",
        {"tenant_id", "pt"},
        NodeExecutionContext(merchant_id="T-1"),
        contract,
        semantic_where_sql=compiled.where_sql,
    )

    assert compiled.valid
    assert " AND ".join(where).count("BETWEEN") == 1
    assert build_semantic_filter_verification_proof(
        contract,
        "SELECT * FROM orders WHERE tenant_id='T-1' AND pt BETWEEN '2026-07-04' AND '2026-07-10'",
    ).verified

    mismatch = contract.model_copy(deep=True)
    mismatch.semantic_query.filter_nodes[0].resolved_values = ["2026-07-05", "2026-07-10"]
    mismatch.semantic_filter_obligations[0].resolved_values = ["2026-07-05", "2026-07-10"]
    assert compile_semantic_filter_sql(mismatch).code == "SEMANTIC_TIME_WINDOW_CONFLICT"
