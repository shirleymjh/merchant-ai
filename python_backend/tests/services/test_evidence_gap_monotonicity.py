import pytest

from merchant_ai.models import AgentRunResult, EvidenceGap, QueryBundle, QueryPlan
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
