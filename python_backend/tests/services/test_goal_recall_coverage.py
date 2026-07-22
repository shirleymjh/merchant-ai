from __future__ import annotations

from merchant_ai.models import RecallItem
from merchant_ai.services.goal_recall_coverage import (
    GoalRecallCoverageService,
    attach_goal_recall_capabilities,
    filter_and_tag_goal_recall_items,
    load_goal_recall_capability_protocol,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    TimeWindowQuestionGoal,
)


def _metric_item(
    ref_id: str,
    label: str,
) -> RecallItem:
    return RecallItem(
        doc_id=ref_id,
        title=label,
        source_type="SEMANTIC_METRIC",
        topic="test-topic",
        table="test-table",
        metadata={
            "semanticRefId": ref_id,
            "matchedMetricLabel": label,
        },
    )


def test_capability_protocol_loads_aliases_and_enriches_items() -> None:
    protocol = load_goal_recall_capability_protocol()
    item = attach_goal_recall_capabilities(
        _metric_item("semantic:test:metric:orders", "订单量"),
        protocol=protocol,
    )

    assert protocol.protocol_version == "goal_recall_capabilities.v1"
    assert protocol.requirements_for("metric") == ["METRIC_DEFINITION"]
    assert item.metadata["goalRecallCapabilities"] == ["METRIC_DEFINITION"]
    assert item.metadata["goalRecallCapabilityProtocolVersion"] == protocol.protocol_version


def test_coverage_plans_supplement_only_for_missing_goal() -> None:
    contract = OriginalQuestionGoalContract(
        question="订单量和退款金额分别是多少？",
        goals=[
            MetricQuestionGoal(
                goal_id="metric.orders",
                label="订单量",
                source_spans=["订单量"],
            ),
            MetricQuestionGoal(
                goal_id="metric.refunds",
                label="退款金额",
                source_spans=["退款金额"],
            ),
            TimeWindowQuestionGoal(
                goal_id="time.current",
                label="当前",
                time_expression="当前",
            ),
        ],
    )

    receipt = GoalRecallCoverageService().evaluate(
        contract,
        [_metric_item("semantic:test:metric:orders", "订单量")],
        index_version="semantic-v7",
        scope_fingerprint="scope-1",
    )

    assert receipt.status == "PARTIAL"
    assert receipt.covered_goal_ids == ["metric.orders"]
    assert receipt.missing_goal_ids == ["metric.refunds"]
    assert receipt.items[-1].status == "NOT_APPLICABLE"
    assert len(receipt.supplemental_requests) == 1
    request = receipt.supplemental_requests[0]
    assert request.target_goal_ids == ["metric.refunds"]
    assert request.query_terms == ["退款金额"]
    assert request.required_capabilities == ["METRIC_DEFINITION"]


def test_multiple_exact_candidates_are_ambiguous_not_missing() -> None:
    contract = OriginalQuestionGoalContract(
        question="订单量是多少？",
        goals=[
            MetricQuestionGoal(
                goal_id="metric.orders",
                label="订单量",
                source_spans=["订单量"],
            )
        ],
    )

    receipt = GoalRecallCoverageService().evaluate(
        contract,
        [
            _metric_item("semantic:a:metric:orders", "订单量"),
            _metric_item("semantic:b:metric:orders", "订单量"),
        ],
    )

    assert receipt.status == "COMPLETE"
    assert receipt.ambiguous_goal_ids == ["metric.orders"]
    assert receipt.missing_goal_ids == []
    assert receipt.supplemental_requests == []


def test_single_goal_accepts_one_unambiguous_exact_unlabelled_candidate() -> None:
    contract = OriginalQuestionGoalContract(
        question="订单量是多少？",
        goals=[MetricQuestionGoal(goal_id="metric.orders", label="订单量")],
    )
    item = RecallItem(
        doc_id="semantic:test:metric:orders",
        source_type="SEMANTIC_METRIC",
        topic="test-topic",
        table="test-table",
        metadata={
            "semanticRefId": "semantic:test:metric:orders",
            "metricResolutionType": "exact_semantic_label",
            "metricResolutionConfidence": 0.97,
            "metricResolutionAmbiguous": False,
        },
    )

    receipt = GoalRecallCoverageService().evaluate(contract, [item])

    assert receipt.covered_goal_ids == ["metric.orders"]
    assert receipt.missing_goal_ids == []


def test_targeted_recall_filters_capability_and_tags_goal_receipt() -> None:
    metric = _metric_item("semantic:test:metric:orders", "订单量")
    table = RecallItem(
        doc_id="semantic:test:table:detail",
        title="订单明细表",
        source_type="SEMANTIC_TABLE_ASSET",
        topic="test-topic",
        table="test-table",
    )

    items = filter_and_tag_goal_recall_items(
        [metric, table],
        target_goal_ids=["metric.orders"],
        required_capabilities=["METRIC_DEFINITION"],
        coverage_receipt_id="goal_recall_123",
    )

    assert [item.doc_id for item in items] == [metric.doc_id]
    assert items[0].metadata["targetGoalIds"] == ["metric.orders"]
    assert items[0].metadata["goalRecallCoverageReceiptId"] == "goal_recall_123"
