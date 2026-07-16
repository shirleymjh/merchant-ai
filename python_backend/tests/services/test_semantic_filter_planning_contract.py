from __future__ import annotations

from decimal import Decimal

from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.planning import (
    QueryGraphContractValidator,
    QuestionUnderstandingCompiler,
    apply_understanding_filters,
    bind_semantic_query_from_understanding,
)


def field(
    key: str,
    table: str,
    *,
    data_type: str = "varchar",
    role: str = "DIMENSION",
    comparison_policy: str = "exact",
    extra_semantic: dict | None = None,
) -> PlanningAssetEntry:
    semantic = {
        "role": role,
        "comparisonPolicy": comparison_policy,
        **(extra_semantic or {}),
    }
    return PlanningAssetEntry(
        key=key,
        table=table,
        source_ref_id=f"semantic:runtime:{table}:field:{key}",
        metadata={"schema": {"dataType": data_type}, "semantic": semantic},
    )


def intent(table: str, task_id: str = "task") -> QuestionIntent:
    return QuestionIntent(
        question="runtime question",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.DETAIL,
        plan_task_id=task_id,
        preferred_table=table,
    )


def semantic_query(*nodes: dict, root: str) -> dict:
    return {"semanticQuery": {"filterNodes": list(nodes), "rootFilterNodeId": root}}


def predicate(node_id: str, ref: str, operator: str, values: list, phrase: str) -> dict:
    return {
        "nodeId": node_id,
        "nodeType": "predicate",
        "semanticRefId": ref,
        "operator": operator,
        "rawValues": values,
        "sourcePhrase": phrase,
    }


def test_all_same_table_predicates_are_bound_without_first_filter_loss() -> None:
    merchant = field("merchant_id", "orders", role="KEY")
    amount = field("amount", "orders", data_type="decimal(18,2)")
    status = field(
        "status_code",
        "orders",
        extra_semantic={
            "enumValues": [],
            "enumMappings": {"R": "refunded"},
            "enumMetadata": {"reviewStatus": "APPROVED"},
        },
    )
    pack = PlanningAssetPack(fields=[merchant, amount, status])
    understanding = semantic_query(
        predicate("p1", merchant.source_ref_id, "eq", ["M-7"], "merchant is M-7"),
        predicate("p2", amount.source_ref_id, "gt", ["100.50"], "amount over 100.50"),
        predicate("p3", status.source_ref_id, "eq", ["refunded"], "status is refunded"),
        {
            "nodeId": "root",
            "nodeType": "group",
            "logicalOperator": "and",
            "childNodeIds": ["p1", "p2", "p3"],
        },
        root="root",
    )

    planned = apply_understanding_filters(QueryPlan(intents=[intent("orders")]), understanding, pack)

    bound = planned.intents[0].semantic_query
    assert [node.node_id for node in bound.filter_nodes] == ["p1", "p2", "p3", "root"]
    by_id = {node.node_id: node for node in bound.filter_nodes}
    assert by_id["p1"].resolved_values == ["M-7"]
    assert by_id["p2"].resolved_values == [Decimal("100.50")]
    assert by_id["p3"].resolved_values == ["R"]
    assert {item.node_id for item in planned.semantic_filter_obligations} == {"p1", "p2", "p3"}
    assert {item.source_phrase for item in planned.semantic_filter_obligations} == {
        "merchant is M-7",
        "amount over 100.50",
        "status is refunded",
    }
    assert QueryGraphContractValidator().validate(
        planned.model_copy(update={"question_understanding": understanding}), pack
    ) == []


def test_legacy_filter_list_is_converted_to_full_and_graph() -> None:
    left = field("merchant_id", "orders", role="KEY")
    right = field("channel", "orders")
    pack = PlanningAssetPack(fields=[left, right])
    understanding = {
        "filters": [
            {"field": "merchant_id", "value": "M-8", "sourcePhrase": "merchant M-8"},
            {"field": "channel", "value": "store", "sourcePhrase": "store channel"},
        ]
    }

    planned = apply_understanding_filters(QueryPlan(intents=[intent("orders")]), understanding, pack)
    spec = planned.intents[0].semantic_query

    assert spec.root_filter_node_id == "legacy_filter_root"
    assert len([node for node in spec.filter_nodes if node.node_type == "predicate"]) == 2
    assert planned.intents[0].filter_column == "merchant_id"
    assert len(planned.semantic_filter_obligations) == 2


def test_native_physical_binding_fields_cannot_bypass_semantic_ref() -> None:
    pack = PlanningAssetPack(fields=[field("merchant_id", "orders", role="KEY")])
    understanding = semantic_query(
        {
            "nodeId": "p1",
            "nodeType": "predicate",
            "semanticRefId": "",
            "boundTable": "orders",
            "boundField": "merchant_id",
            "operator": "eq",
            "rawValues": ["M-9"],
            "sourcePhrase": "merchant M-9",
        },
        root="p1",
    )
    plan = apply_understanding_filters(QueryPlan(intents=[intent("orders")]), understanding, pack)
    plan = plan.model_copy(update={"question_understanding": understanding})

    assert plan.intents[0].semantic_query.filter_nodes == []
    assert "FILTER_SEMANTIC_REF_UNKNOWN" in {
        gap.code for gap in QueryGraphContractValidator().validate(plan, pack)
    }


def test_unknown_or_ambiguous_enum_value_remains_unresolved() -> None:
    state = field(
        "state_code",
        "orders",
        extra_semantic={
            "enumMappings": {"A": "closed", "B": "closed"},
            "enumMetadata": {"reviewStatus": "APPROVED"},
        },
    )
    pack = PlanningAssetPack(fields=[state])
    understanding = semantic_query(
        predicate("p1", state.source_ref_id, "eq", ["closed"], "state is closed"),
        root="p1",
    )
    plan = apply_understanding_filters(QueryPlan(intents=[intent("orders")]), understanding, pack)
    plan = plan.model_copy(update={"question_understanding": understanding})

    node = bind_semantic_query_from_understanding(understanding, pack).filter_nodes[0]
    assert node.resolution_status == "unresolved"
    assert node.candidate_values == ["A", "B"]
    assert "FILTER_VALUE_UNRESOLVED" in {
        gap.code for gap in QueryGraphContractValidator().validate(plan, pack)
    }


def test_metric_ref_binds_as_measure_for_having_lane() -> None:
    metric = PlanningAssetEntry(
        key="gross_value",
        table="merchant_daily",
        source_ref_id="semantic:runtime:merchant_daily:metric:gross_value",
        metadata={"formula": "SUM(amount)", "dataType": "decimal(18,2)"},
    )
    pack = PlanningAssetPack(metrics=[metric])
    understanding = semantic_query(
        predicate("p1", metric.source_ref_id, "gt", ["10000"], "gross value over 10000"),
        root="p1",
    )

    planned = apply_understanding_filters(
        QueryPlan(intents=[intent("merchant_daily")]), understanding, pack
    )
    node = planned.intents[0].semantic_query.filter_nodes[0]

    assert node.member_kind == "measure"
    assert node.bound_field == "gross_value"
    assert node.resolved_values == [Decimal("10000")]
    assert planned.intents[0].output_keys == []


def test_cross_table_boolean_expression_fails_closed() -> None:
    first = field("merchant_id", "orders", role="KEY")
    second = field("risk_level", "risk_profile")
    pack = PlanningAssetPack(fields=[first, second])
    understanding = semantic_query(
        predicate("p1", first.source_ref_id, "eq", ["M-10"], "merchant M-10"),
        predicate("p2", second.source_ref_id, "eq", ["high"], "high risk"),
        {
            "nodeId": "root",
            "nodeType": "group",
            "logicalOperator": "or",
            "childNodeIds": ["p1", "p2"],
        },
        root="root",
    )
    plan = apply_understanding_filters(
        QueryPlan(intents=[intent("orders", "orders"), intent("risk_profile", "risk")]),
        understanding,
        pack,
    ).model_copy(update={"question_understanding": understanding})

    assert all(not item.semantic_query.filter_nodes for item in plan.intents)
    assert "FILTER_SCOPE_UNSUPPORTED" in {
        gap.code for gap in QueryGraphContractValidator().validate(plan, pack)
    }


def test_entity_predicate_under_or_is_not_projected_as_global_entity_filter() -> None:
    merchant = field("merchant_id", "orders", role="KEY")
    state = field("state", "orders")
    pack = PlanningAssetPack(fields=[merchant, state])
    understanding = semantic_query(
        predicate("merchant", merchant.source_ref_id, "eq", ["M-12"], "merchant M-12"),
        predicate("state", state.source_ref_id, "eq", ["refunded"], "state refunded"),
        {
            "nodeId": "root",
            "nodeType": "group",
            "logicalOperator": "or",
            "childNodeIds": ["merchant", "state"],
        },
        root="root",
    )

    planned = apply_understanding_filters(QueryPlan(intents=[intent("orders")]), understanding, pack)

    assert planned.intents[0].semantic_query.filter_nodes
    assert planned.intents[0].filter_column == ""
    assert planned.intents[0].entity_reference.status == "unresolved"
    assert planned.entity_filter_obligations == []
    assert {item.node_id for item in planned.semantic_filter_obligations} == {"merchant", "state"}


def test_filter_graph_topology_and_type_errors_are_typed_gaps() -> None:
    amount = field("amount", "orders", data_type="decimal(18,2)")
    understanding = semantic_query(
        predicate("p1", amount.source_ref_id, "contains", ["100"], "amount contains 100"),
        {
            "nodeId": "root",
            "nodeType": "group",
            "logicalOperator": "not",
            "childNodeIds": ["root", "missing"],
        },
        root="root",
    )
    pack = PlanningAssetPack(fields=[amount])
    plan = QueryPlan(question_understanding=understanding, intents=[intent("orders")])

    codes = {gap.code for gap in QueryGraphContractValidator().validate(plan, pack)}

    assert "FILTER_OPERATOR_UNSUPPORTED" in codes
    assert "FILTER_CHILD_UNKNOWN" in codes
    assert "FILTER_GROUP_ARITY_INVALID" in codes
    assert "FILTER_CYCLE" in codes


def test_detail_anchor_uses_governed_semantic_key_predicate() -> None:
    merchant = field("merchant_id", "orders", role="KEY")
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="orders",
                table="orders",
                columns=["merchant_id", "amount"],
                metadata={"entityLookupPolicy": {"mode": "not_required"}},
            )
        ],
        fields=[merchant, field("amount", "orders", data_type="decimal(18,2)")],
    )
    understanding = semantic_query(
        predicate("merchant", merchant.source_ref_id, "eq", ["M-11"], "merchant M-11"),
        root="merchant",
    )
    understanding["rankingObjective"] = {}

    plan = QuestionUnderstandingCompiler().compile("show merchant M-11 details", understanding, pack)

    assert plan.intents
    assert plan.intents[0].preferred_table == "orders"
    assert plan.intents[0].filter_column == "merchant_id"
    assert plan.intents[0].filter_value == "M-11"
    assert plan.intents[0].semantic_query.filter_nodes[0].semantic_ref_id == merchant.source_ref_id
    assert plan.semantic_filter_obligations[0].node_id == "merchant"
