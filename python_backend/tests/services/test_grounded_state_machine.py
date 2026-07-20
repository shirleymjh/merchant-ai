import pytest

from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentSession,
    _grounded_state_semantics,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession


def _session(phase: str) -> GroundedDeepAgentSession:
    return GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="state-session",
            question="state question",
            merchant_id="merchant",
            phase=phase,
        )
    )


@pytest.mark.parametrize(
    ("phase", "state_class", "failure_category"),
    [
        ("VERIFICATION_GAPPED", "SEMANTIC_REPLAN_REQUIRED", "SEMANTIC_GAP"),
        ("CORE_SQL_REPAIR_REQUIRED", "SQL_REPAIR", "SQL_REPAIR"),
        ("DATASOURCE_RECOVERY_REQUIRED", "DATASOURCE_RECOVERY", "DATASOURCE"),
        ("ACTIVE_COMPILED", "EXECUTION_READY", "NONE"),
    ],
)
def test_runtime_phases_have_one_typed_state_semantics(
    phase: str,
    state_class: str,
    failure_category: str,
) -> None:
    result = _grounded_state_semantics(_session(phase))

    assert result["stateClass"] == state_class
    assert result["failureCategory"] == failure_category


def test_security_and_internal_failures_cannot_be_reclassified_as_semantic_gap() -> None:
    security = _session("SECURITY_BLOCKED")
    security.operational_failure = {
        "failureDisposition": "SECURITY_TERMINAL",
        "code": "ACCESS_DENIED",
    }
    internal = _session("OPERATIONAL_FAILURE")
    internal.operational_failure = {
        "failureDisposition": "OPERATIONAL_TERMINAL",
        "code": "PUBLICATION_INTERNAL_ERROR",
    }

    assert _grounded_state_semantics(security) == {
        "stateClass": "SECURITY_BLOCKED",
        "failureCategory": "SECURITY",
        "terminal": True,
        "retryable": False,
        "nextAction": "STOP",
    }
    assert _grounded_state_semantics(internal)["failureCategory"] == "SYSTEM"
