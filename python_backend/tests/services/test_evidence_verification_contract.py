from merchant_ai.graph.evidence_verification_contract import (
    EvidenceVerificationStatus,
    evidence_verification_attempted,
    evidence_verification_passed,
    invalidate_evidence_verification,
    record_evidence_verification,
)
from merchant_ai.models import AgentRunResult, VerifiedEvidence


def test_evidence_verification_has_one_canonical_status_writer() -> None:
    state = {
        "agent_run_result": AgentRunResult(
            verified_evidence=VerifiedEvidence(passed=True),
        )
    }

    status = record_evidence_verification(
        state,
        state["agent_run_result"].verified_evidence,
    )

    assert status == EvidenceVerificationStatus.PASSED
    assert state["verification_status"] == "passed"
    assert state["evidence_graph_verified"] is True
    assert state["evidence_accepted"] is True
    assert evidence_verification_attempted(state) is True
    assert evidence_verification_passed(state) is True


def test_evidence_verification_rejects_stale_generation() -> None:
    state = {
        "agent_run_result": AgentRunResult(
            verified_evidence=VerifiedEvidence(passed=True),
        ),
        "execution_generation": 2,
        "result_generation": 1,
        "evidence_generation": 1,
    }
    record_evidence_verification(
        state,
        state["agent_run_result"].verified_evidence,
    )

    assert evidence_verification_passed(state) is True
    assert evidence_verification_passed(
        state,
        require_current_generation=True,
    ) is False


def test_evidence_verification_invalidation_resets_all_projections() -> None:
    state = {
        "verification_status": "passed",
        "evidence_graph_verified": True,
        "evidence_accepted": True,
    }

    invalidate_evidence_verification(state)

    assert state == {
        "verification_status": "not_run",
        "evidence_graph_verified": False,
        "evidence_accepted": False,
    }
    assert evidence_verification_attempted(state) is False
