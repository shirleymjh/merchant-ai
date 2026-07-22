from __future__ import annotations

from merchant_ai.models import AgentRunResult, QueryPlan
from merchant_ai.services.evidence import EvidenceVerifier


def test_empty_verification_input_never_passes() -> None:
    verified = EvidenceVerifier().verify(
        "最近30天 GMV 是多少",
        QueryPlan(),
        AgentRunResult(),
    )

    assert verified.passed is False
    assert "EVIDENCE_INPUT_REQUIRED" in {gap.code for gap in verified.blocking_gaps}


def test_knowledge_contract_requires_explicit_allowed_reference_set() -> None:
    plan = QueryPlan(
        evidence_contracts=[
            {
                "evidenceSource": "knowledge_ref",
                "knowledgeRefs": ["rule-verified-1"],
                "semanticLabel": "GMV business rule",
                "requiredLevel": "required",
            }
        ]
    )

    missing_authority = EvidenceVerifier().verify(
        "GMV 口径是什么",
        plan,
        AgentRunResult(),
    )
    authorized = EvidenceVerifier().verify(
        "GMV 口径是什么",
        plan,
        AgentRunResult(),
        allowed_knowledge_refs={"rule-verified-1"},
    )

    assert missing_authority.passed is False
    assert authorized.passed is True
