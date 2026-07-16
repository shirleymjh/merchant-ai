from __future__ import annotations

from merchant_ai.graph.query_graph_contract import query_graph_structure_fingerprint
from merchant_ai.models import (
    QueryPlan,
    QuestionIntent,
    SemanticFilterNode,
    SemanticFilterObligation,
    SemanticQuerySpec,
)
from merchant_ai.services.planning_tooling import planner_structured_output_validation_errors
from merchant_ai.services.tools import question_understanding_tool


def semantic_query_payload() -> dict:
    return {
        "resultMode": "detail",
        "filterNodes": [
            {
                "nodeId": "merchant",
                "nodeType": "predicate",
                "semanticRefId": "semantic:trade:orders:field:merchant_id",
                "sourcePhrase": "商家 ID 为 M-1",
                "operator": "eq",
                "rawValues": ["M-1"],
                "logicalOperator": "",
                "childNodeIds": [],
                "knowledgeRefIds": ["semantic:trade:orders:field:merchant_id"],
                "reason": "governed field",
            },
            {
                "nodeId": "amount",
                "nodeType": "predicate",
                "semanticRefId": "semantic:trade:orders:field:pay_amt",
                "sourcePhrase": "金额大于 100",
                "operator": "gt",
                "rawValues": ["100"],
                "logicalOperator": "",
                "childNodeIds": [],
                "knowledgeRefIds": ["semantic:trade:orders:field:pay_amt"],
                "reason": "governed field",
            },
            {
                "nodeId": "root",
                "nodeType": "group",
                "semanticRefId": "",
                "sourcePhrase": "商家 ID 为 M-1 且金额大于 100",
                "operator": "",
                "rawValues": [],
                "logicalOperator": "and",
                "childNodeIds": ["merchant", "amount"],
                "knowledgeRefIds": [],
                "reason": "user conjunction",
            },
        ],
        "rootFilterNodeId": "root",
        "selectRefIds": [],
        "measureRefIds": [],
        "dimensionRefIds": [],
        "sourceRefIds": ["semantic:trade:orders:table"],
        "relationshipRefIds": [],
        "joinStrategy": "single_source",
        "orderBy": [],
        "limit": 20,
        "bindingStatus": "unresolved",
    }


def test_question_understanding_tool_exposes_non_recursive_semantic_query_graph():
    tool = question_understanding_tool()
    understanding = tool.parameters["properties"]["questionUnderstanding"]
    semantic_query = understanding["properties"]["semanticQuery"]

    assert "semanticQuery" not in understanding["required"]  # legacy Planner compatibility
    assert semantic_query["properties"]["filterNodes"]["items"]["type"] == "object"
    assert "$ref" not in str(semantic_query)
    assert "oneOf" not in str(semantic_query)
    assert "boundField" not in semantic_query["properties"]["filterNodes"]["items"]["properties"]


def test_semantic_query_model_round_trips_camel_case_contract():
    spec = SemanticQuerySpec.model_validate(semantic_query_payload())
    dumped = spec.model_dump(by_alias=True, mode="json")

    assert dumped["rootFilterNodeId"] == "root"
    assert dumped["filterNodes"][1]["operator"] == "gt"
    assert dumped["relationshipRefIds"] == []


def test_planner_structured_output_accepts_semantic_query_without_requiring_legacy_filters():
    tool = question_understanding_tool()
    payload = {
        "status": "UNDERSTOOD",
        "questionUnderstanding": {
            "analysisGrain": "order",
            "analysisIntent": "none",
            "requiresExplanation": False,
            "requiredEvidenceIntents": [],
            "anchorMetric": {},
            "supportMetrics": [],
            "metricCandidateDecisions": [],
            "calculationIntents": [],
            "scopeConstraints": [],
            "filters": [],
            "semanticQuery": semantic_query_payload(),
            "timeWindowDays": 7,
        },
        "reason": "detail lookup",
    }

    assert planner_structured_output_validation_errors(payload, tool.parameters) == []


def test_semantic_filter_changes_executable_graph_fingerprint_but_reason_does_not():
    spec = SemanticQuerySpec.model_validate(semantic_query_payload())
    plan = QueryPlan(
        intents=[QuestionIntent(plan_task_id="detail", preferred_table="orders", semantic_query=spec)],
        semantic_filter_obligations=[
            SemanticFilterObligation(
                obligation_id="filter_merchant",
                task_id="detail",
                node_id="merchant",
                semantic_ref_id="semantic:trade:orders:field:merchant_id",
                source_phrase="商家 ID 为 M-1",
                operator="eq",
                raw_values=["M-1"],
                status="bound",
            )
        ],
    )
    original = query_graph_structure_fingerprint(plan)

    prose_only = plan.model_copy(deep=True)
    prose_only.intents[0].semantic_query.filter_nodes[0].reason = "rewritten prose"
    prose_only.semantic_filter_obligations[0].reason = "rewritten prose"
    assert query_graph_structure_fingerprint(prose_only) == original

    reordered = plan.model_copy(deep=True)
    reordered.intents[0].semantic_query.filter_nodes = list(
        reversed(reordered.intents[0].semantic_query.filter_nodes)
    )
    root = next(
        node for node in reordered.intents[0].semantic_query.filter_nodes if node.node_id == "root"
    )
    root.child_node_ids = list(reversed(root.child_node_ids))
    assert query_graph_structure_fingerprint(reordered) == original

    changed = plan.model_copy(deep=True)
    changed.intents[0].semantic_query.filter_nodes[0].raw_values = ["M-2"]
    changed.semantic_filter_obligations[0].raw_values = ["M-2"]
    assert query_graph_structure_fingerprint(changed) != original
