from __future__ import annotations

from enum import Enum
from typing import Any, MutableMapping

from merchant_ai.models import AgentRunResult, VerifiedEvidence


class EvidenceVerificationStatus(str, Enum):
    """Canonical evidence-verification lifecycle for compatibility workflows."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    SEMANTIC_DEFINITION = "semantic_definition"


def record_evidence_verification(
    state: MutableMapping[str, Any],
    verified: VerifiedEvidence,
    *,
    status: EvidenceVerificationStatus | str | None = None,
) -> EvidenceVerificationStatus:
    """Write the canonical status and derive legacy boolean projections.

    ``evidence_graph_verified`` and ``evidence_accepted`` remain serialized for
    compatibility, but callers must no longer write them independently.
    """

    canonical = _normalize_status(
        status
        or (
            EvidenceVerificationStatus.PASSED
            if verified.passed
            else EvidenceVerificationStatus.FAILED
        )
    )
    attempted = canonical in {
        EvidenceVerificationStatus.PASSED,
        EvidenceVerificationStatus.FAILED,
    }
    accepted = bool(
        canonical == EvidenceVerificationStatus.PASSED and verified.passed
    )
    state["verification_status"] = canonical.value
    state["evidence_graph_verified"] = attempted
    state["evidence_accepted"] = accepted
    return canonical


def invalidate_evidence_verification(
    state: MutableMapping[str, Any],
    *,
    status: EvidenceVerificationStatus | str = EvidenceVerificationStatus.NOT_RUN,
) -> EvidenceVerificationStatus:
    """Reset evidence authority while keeping compatibility projections aligned."""

    canonical = _normalize_status(status)
    state["verification_status"] = canonical.value
    state["evidence_graph_verified"] = False
    state["evidence_accepted"] = False
    return canonical


def evidence_verification_attempted(state: MutableMapping[str, Any]) -> bool:
    return _normalize_status(state.get("verification_status")) in {
        EvidenceVerificationStatus.PASSED,
        EvidenceVerificationStatus.FAILED,
    }


def evidence_verification_passed(
    state: MutableMapping[str, Any],
    *,
    require_current_generation: bool = False,
) -> bool:
    if _normalize_status(state.get("verification_status")) != EvidenceVerificationStatus.PASSED:
        return False
    run_result = _run_result(state)
    if not run_result.verified_evidence.passed:
        return False
    if require_current_generation and not _generation_matches(state):
        return False
    return True


def _run_result(state: MutableMapping[str, Any]) -> AgentRunResult:
    raw = state.get("agent_run_result")
    if isinstance(raw, AgentRunResult):
        return raw
    if isinstance(raw, dict):
        return AgentRunResult.model_validate(raw)
    return AgentRunResult()


def _generation_matches(state: MutableMapping[str, Any]) -> bool:
    if "execution_generation" not in state:
        return True
    current = int(state.get("execution_generation") or 0)
    result = int(
        state.get("result_generation")
        if state.get("result_generation") is not None
        else -1
    )
    evidence = int(
        state.get("evidence_generation")
        if state.get("evidence_generation") is not None
        else -1
    )
    return not (
        (result >= 0 or evidence >= 0)
        and (result != current or evidence != current)
    )


def _normalize_status(value: EvidenceVerificationStatus | str | Any) -> EvidenceVerificationStatus:
    if isinstance(value, EvidenceVerificationStatus):
        return value
    normalized = str(value or "").strip().lower()
    if normalized == "verified":
        normalized = EvidenceVerificationStatus.PASSED.value
    try:
        return EvidenceVerificationStatus(normalized)
    except ValueError:
        return EvidenceVerificationStatus.NOT_RUN
