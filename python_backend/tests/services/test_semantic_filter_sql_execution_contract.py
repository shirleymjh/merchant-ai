from __future__ import annotations

import pytest

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    NodeExecutionContext,
    NodePlanContract,
    QueryBundle,
    QuestionIntent,
    SemanticFilterNode,
    SemanticFilterObligation,
    SemanticQuerySpec,
)
from merchant_ai.services.evidence import semantic_filter_evidence_gaps
from merchant_ai.services.query import (
    NodeExecutionContractValidator,
    NodeWorkerExecutor,
    build_semantic_filter_verification_proof,
    compile_semantic_filter_sql,
)


def predicate(
    node_id: str,
    field: str,
    operator: str,
    values: list,
    *,
    data_type: str = "string",
    member_kind: str = "dimension",
    semantic_ref_id: str = "",
) -> SemanticFilterNode:
    return SemanticFilterNode(
        node_id=node_id,
        node_type="predicate",
        semantic_ref_id=semantic_ref_id or "semantic:trade:orders:field:%s" % field,
        source_phrase="%s %s" % (field, operator),
        operator=operator,
        raw_values=list(values),
        resolved_values=list(values),
        bound_table="orders",
        bound_field=field,
        member_kind=member_kind,
        data_type=data_type,
        resolution_status="resolved",
    )


def obligation(node: SemanticFilterNode, task_id: str = "detail") -> SemanticFilterObligation:
    return SemanticFilterObligation(
        obligation_id="obligation_%s" % node.node_id,
        task_id=task_id,
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


def contract_for(
    nodes: list[SemanticFilterNode],
    root_id: str,
    *,
    answer_mode: str = AnswerMode.DETAIL.value,
    metric_specs: list[dict] | None = None,
) -> NodePlanContract:
    predicates = [item for item in nodes if item.node_type == "predicate"]
    physical_fields = [item.bound_field for item in predicates if item.member_kind != "measure"]
    physical_fields.extend(["amount", "status", "tenant_id"])
    return NodePlanContract(
        task_id="detail",
        preferred_table="orders",
        allowed_columns=list(dict.fromkeys(physical_fields)),
        visible_columns=list(dict.fromkeys(physical_fields)),
        semantic_query=SemanticQuerySpec(
            filter_nodes=nodes,
            root_filter_node_id=root_id,
            binding_status="resolved",
        ),
        semantic_filter_obligations=[obligation(item) for item in predicates],
        answer_mode=answer_mode,
        group_by_column="status" if answer_mode != AnswerMode.DETAIL.value else "",
        metric_specs=metric_specs or [],
        filter_value_limit=200,
        merchant_filter_column="tenant_id",
        merchant_id="T-1",
    )


@pytest.mark.parametrize(
    ("operator", "values", "data_type"),
    [
        ("eq", ["paid"], "string"),
        ("neq", ["paid"], "string"),
        ("in", ["paid", "refund"], "string"),
        ("not_in", ["cancelled"], "string"),
        ("gt", [100], "number"),
        ("gte", [100], "number"),
        ("lt", [100], "number"),
        ("lte", [100], "number"),
        ("between", [100, 200], "number"),
        ("is_null", [], "string"),
        ("is_not_null", [], "string"),
        ("contains", ["A%_B"], "string"),
        ("starts_with", ["A"], "string"),
        ("ends_with", ["B"], "string"),
    ],
)
def test_supported_predicates_compile_and_are_proven_in_sql(operator, values, data_type):
    node = predicate("p", "amount" if data_type == "number" else "status", operator, values, data_type=data_type)
    contract = contract_for([node], "p")

    compiled = compile_semantic_filter_sql(contract)
    proof = build_semantic_filter_verification_proof(
        contract,
        "SELECT * FROM `orders` WHERE `tenant_id` = 'T-1' AND (%s)" % compiled.where_sql,
    )

    assert compiled.valid, compiled.reason
    assert proof.verified is True
    assert proof.verified_count == 1


def test_and_or_not_structure_is_preserved_and_proven():
    merchant = predicate("merchant", "merchant_id", "eq", ["M-1"])
    paid = predicate("paid", "status", "eq", ["paid"])
    cancelled = predicate("cancelled", "status", "eq", ["cancelled"])
    not_cancelled = SemanticFilterNode(
        node_id="not_cancelled",
        node_type="group",
        logical_operator="not",
        child_node_ids=["cancelled"],
        resolution_status="resolved",
    )
    status_group = SemanticFilterNode(
        node_id="status_group",
        node_type="group",
        logical_operator="or",
        child_node_ids=["paid", "not_cancelled"],
        resolution_status="resolved",
    )
    root = SemanticFilterNode(
        node_id="root",
        node_type="group",
        logical_operator="and",
        child_node_ids=["merchant", "status_group"],
        resolution_status="resolved",
    )
    contract = contract_for([merchant, paid, cancelled, not_cancelled, status_group, root], "root")

    compiled = compile_semantic_filter_sql(contract)
    valid = build_semantic_filter_verification_proof(
        contract,
        "SELECT * FROM orders WHERE tenant_id = 'T-1' AND (%s)" % compiled.where_sql,
    )
    changed = build_semantic_filter_verification_proof(
        contract,
        "SELECT * FROM orders WHERE merchant_id = 'M-1' AND status = 'paid' AND NOT (status = 'cancelled')",
    )

    assert valid.verified is True
    assert changed.verified is False
    assert changed.code == "SEMANTIC_FILTER_BOOLEAN_STRUCTURE_MISMATCH"


def test_root_and_safely_splits_dimension_where_and_governed_measure_having():
    status = predicate("status", "status", "eq", ["paid"])
    gmv = predicate(
        "gmv",
        "gmv",
        "gt",
        [1000],
        data_type="number",
        member_kind="measure",
        semantic_ref_id="semantic:trade:orders:metric:gmv",
    )
    root = SemanticFilterNode(
        node_id="root",
        node_type="group",
        logical_operator="and",
        child_node_ids=["status", "gmv"],
        resolution_status="resolved",
    )
    contract = contract_for(
        [status, gmv, root],
        "root",
        answer_mode=AnswerMode.GROUP_AGG.value,
        metric_specs=[
            {
                "semanticRefId": gmv.semantic_ref_id,
                "metricName": "gmv",
                "metricFormula": "SUM(amount)",
                "sourceColumns": ["amount"],
            }
        ],
    )

    compiled = compile_semantic_filter_sql(contract)
    sql = (
        "SELECT status, SUM(amount) AS gmv FROM orders WHERE tenant_id = 'T-1' AND (%s) "
        "GROUP BY status HAVING %s" % (compiled.where_sql, compiled.having_sql)
    )

    assert compiled.valid is True
    assert "status" in compiled.where_sql
    assert "SUM" in compiled.having_sql
    assert build_semantic_filter_verification_proof(contract, sql).verified is True


def test_or_across_where_and_having_fails_closed_without_rewriting_logic():
    status = predicate("status", "status", "eq", ["paid"])
    gmv = predicate(
        "gmv",
        "gmv",
        "gt",
        [1000],
        data_type="number",
        member_kind="measure",
        semantic_ref_id="semantic:trade:orders:metric:gmv",
    )
    root = SemanticFilterNode(
        node_id="root",
        node_type="group",
        logical_operator="or",
        child_node_ids=["status", "gmv"],
        resolution_status="resolved",
    )
    contract = contract_for(
        [status, gmv, root],
        "root",
        answer_mode=AnswerMode.GROUP_AGG.value,
        metric_specs=[
            {
                "semanticRefId": gmv.semantic_ref_id,
                "metricName": "gmv",
                "metricFormula": "SUM(amount)",
            }
        ],
    )

    compiled = compile_semantic_filter_sql(contract)
    critique = NodeExecutionContractValidator().review(contract)

    assert compiled.code == "SEMANTIC_FILTER_MIXED_BOOLEAN_SCOPE_UNSUPPORTED"
    assert critique.valid is False
    assert any(item["code"] == compiled.code for item in critique.issues)


def test_obligation_drift_is_rejected_before_sql_execution():
    node = predicate("status", "status", "eq", ["paid"])
    contract = contract_for([node], "status")
    contract.semantic_filter_obligations[0].resolved_values = ["refunded"]

    critique = NodeExecutionContractValidator().review(contract)

    assert critique.valid is False
    assert any(item["code"] == "SEMANTIC_FILTER_CONTRACT_MISMATCH" for item in critique.issues)


def test_semantic_filter_replaces_equivalent_legacy_user_predicate_without_duplication():
    node = predicate("merchant", "merchant_id", "eq", ["M-1"])
    contract = contract_for([node], "merchant")
    contract.filter_column = "merchant_id"
    contract.filter_values = ["M-1"]
    intent = QuestionIntent(filter_column="merchant_id", filter_value="M-1")
    compiled = compile_semantic_filter_sql(contract)

    where = NodeWorkerExecutor._structured_where(
        object(),
        intent,
        "orders",
        {"merchant_id"},
        NodeExecutionContext(),
        contract,
        semantic_where_sql=compiled.where_sql,
    )

    assert " AND ".join(where).count("`merchant_id` = 'M-1'") == 1


def test_evidence_blocks_missing_or_stale_semantic_filter_proof():
    node = predicate("status", "status", "eq", ["paid"])
    contract = contract_for([node], "status")
    compiled = compile_semantic_filter_sql(contract)
    sql = "SELECT * FROM orders WHERE %s" % compiled.where_sql
    task = AgentTaskResult(
        task_id="detail",
        success=True,
        query_bundle=QueryBundle(sql=sql),
        node_plan_contract=contract,
    )
    run = AgentRunResult(task_results=[task])

    assert [item.code for item in semantic_filter_evidence_gaps(run)] == [
        "SEMANTIC_FILTER_VERIFICATION_MISSING"
    ]

    task.semantic_filter_verification = build_semantic_filter_verification_proof(contract, sql)
    assert semantic_filter_evidence_gaps(run) == []
