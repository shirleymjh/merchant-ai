import pytest

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    EntitySet,
    EvidenceGap,
    PlanDependency,
    QueryBundle,
    QueryPlan,
)
from merchant_ai.services.evidence import EvidenceVerifier


def test_evidence_verifier_preserves_upstream_typed_gap_across_reverification():
    run_result = AgentRunResult(
        evidence_gaps=[
            EvidenceGap(
                code="QUERY_GRAPH_REPAIR_EXHAUSTED",
                evidence="metric_node",
                reason="PlannerCritic repair made no executable progress",
                severity="blocking",
                source="query_graph_repair",
            )
        ]
    )
    verifier = EvidenceVerifier()

    first = verifier.verify("最近30天指标是多少", QueryPlan(), run_result)
    run_result.evidence_gaps = first.gaps
    second = verifier.verify("最近30天指标是多少", QueryPlan(), run_result)

    assert first.passed is False
    assert second.passed is False
    assert [gap.code for gap in second.gaps] == ["QUERY_GRAPH_REPAIR_EXHAUSTED"]
    assert second.blocking_gaps[0].source == "query_graph_repair"


def test_execution_failure_without_task_result_cannot_pass_evidence_gate():
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            failed=True,
            error="NodeWorker bootstrap failed before task dispatch",
        )
    )

    verified = EvidenceVerifier().verify("最近7天指标是多少", QueryPlan(), run_result)

    assert verified.passed is False
    assert [gap.code for gap in verified.blocking_gaps] == ["EXECUTION_OPERATIONAL_FAILURE"]
    assert "不能输出业务结论" in verified.blocking_gaps[0].answer_instruction


@pytest.mark.parametrize("severity", ["error", "critical", "fatal", "blocking", "unexpected_vendor_value"])
def test_error_like_or_unknown_severity_never_bypasses_blocking_gate(severity):
    run_result = AgentRunResult(
        evidence_gaps=[
            EvidenceGap(
                code="PLAN_CONTRACT_MISMATCH",
                reason="node output violated its contract",
                severity=severity,
            )
        ]
    )

    verified = EvidenceVerifier().verify("最近7天指标是多少", QueryPlan(), run_result)

    assert verified.passed is False
    assert verified.blocking_gaps[0].severity == "blocking"


def test_short_column_name_cannot_satisfy_a_more_specific_evidence_identifier():
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["refund_orders"],
            rows=[{"id": "refund_1"}],
            original_row_count=1,
        )
    )
    plan = QueryPlan(final_required_evidence=["refund_order_id"])

    verified = EvidenceVerifier().verify("退款订单是什么", plan, run_result)

    assert verified.passed is False
    assert [gap.code for gap in verified.blocking_gaps] == ["MISSING_REQUIRED_EVIDENCE"]


def test_natural_evidence_label_may_embed_an_exact_governed_identifier():
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["refund_orders"],
            rows=[{"refund_order_id": "refund_1"}],
            original_row_count=1,
        )
    )
    plan = QueryPlan(final_required_evidence=["退款订单号(refund_order_id)"])

    verified = EvidenceVerifier().verify("退款订单是什么", plan, run_result)

    assert verified.passed is True


def test_standalone_detail_entity_preview_does_not_claim_downstream_truncation() -> None:
    task = AgentTaskResult(
        task_id="orders",
        success=True,
        query_bundle=QueryBundle(
            tables=["orders"],
            rows=[{"order_id": "order-1"}],
        ),
        entity_set=EntitySet(
            task_id="orders",
            join_key="order_id",
            values=["order-1"],
            truncated=True,
            source_row_count=58,
        ),
    )
    run_result = AgentRunResult(
        task_results=[task],
        merged_query_bundle=task.query_bundle.model_copy(deep=True),
    )

    verified = EvidenceVerifier().verify(
        "最近10天订单明细",
        QueryPlan(),
        run_result,
    )

    assert "ENTITY_SET_TRUNCATED" not in {
        gap.code for gap in verified.warning_gaps
    }


def test_consumed_truncated_entity_set_keeps_downstream_coverage_warning() -> None:
    task = AgentTaskResult(
        task_id="orders",
        success=True,
        query_bundle=QueryBundle(
            tables=["orders"],
            rows=[{"order_id": "order-1"}],
        ),
        entity_set=EntitySet(
            task_id="orders",
            join_key="order_id",
            values=["order-1"],
            truncated=True,
            source_row_count=58,
        ),
    )
    run_result = AgentRunResult(
        task_results=[task],
        merged_query_bundle=task.query_bundle.model_copy(deep=True),
    )
    plan = QueryPlan(
        dependencies=[
            PlanDependency(
                anchor_task_id="orders",
                dependent_task_id="refunds",
                join_key="order_id",
            )
        ]
    )

    verified = EvidenceVerifier().verify(
        "从这些订单中查询退款",
        plan,
        run_result,
    )

    assert "ENTITY_SET_TRUNCATED" in {
        gap.code for gap in verified.warning_gaps
    }
