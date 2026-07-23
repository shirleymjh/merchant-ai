from __future__ import annotations

import json

from merchant_ai.services.grounded_outcome_completion import (
    OutcomeCompletionDecision,
    OutcomeCompletionStatus,
    OutcomeCompletionVerifier,
    OutcomeEvidenceKind,
    StructuredOutcomeCompletionProvider,
    UserOutcomeAssessment,
    outcome_attestation_matches,
    render_partial_outcome_gaps,
)


class _StructuredModel:
    def __init__(self, output) -> None:
        self.output = output
        self.schema = None
        self.method = ""
        self.strict = False
        self.messages = []

    def with_structured_output(self, schema, *, method: str, strict: bool):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self

    def invoke(self, messages):
        self.messages = list(messages)
        return self.output


def _satisfied_decision(*, artifact_id: str = "artifact.orders"):
    return OutcomeCompletionDecision(
        overall_status=OutcomeCompletionStatus.SATISFIED,
        outcomes=[
            UserOutcomeAssessment(
                outcome_id="orders.scalar",
                requirement="return the requested order count",
                source_spans=["order count"],
                status=OutcomeCompletionStatus.SATISFIED,
                evidence_kind=OutcomeEvidenceKind.DATA,
                query_artifact_ids=[artifact_id],
                evidence_refs=["semantic:metric:orders"],
            )
        ],
    )


def test_provider_is_zero_tool_strict_and_receives_compact_artifacts() -> None:
    model = _StructuredModel(_satisfied_decision())
    provider = StructuredOutcomeCompletionProvider(model)

    decision = provider.evaluate(
        question="order count",
        required_goals=[{"goalId": "metric.orders", "kind": "METRIC"}],
        candidate_answer="The order count is 12.",
        verified_query_artifacts=[
            {
                "artifactId": "artifact.orders",
                "queryShape": "SCALAR",
                "outputColumns": ["order_count"],
            }
        ],
        verified_rule_artifacts=[],
        known_gaps=[],
        timeout_seconds=2.0,
    )

    assert decision.overall_status == OutcomeCompletionStatus.SATISFIED
    assert model.schema is OutcomeCompletionDecision
    assert model.method == "json_schema"
    assert model.strict is True
    supplied = json.loads(model.messages[1][1])
    assert supplied["verifiedQueryArtifacts"][0]["artifactId"] == (
        "artifact.orders"
    )
    assert "sql" not in supplied["verifiedQueryArtifacts"][0]
    assert "rows" not in supplied["verifiedQueryArtifacts"][0]


def test_verifier_accepts_only_real_verified_artifact_references() -> None:
    result = OutcomeCompletionVerifier().verify(
        _satisfied_decision(),
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=True,
    )

    assert result.completion_allowed is True
    assert result.query_artifact_ids == ["artifact.orders"]
    assert outcome_attestation_matches(
        "The order count is 12.", result
    )

    forged = OutcomeCompletionVerifier().verify(
        _satisfied_decision(artifact_id="artifact.forged"),
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=True,
    )

    assert forged.completion_allowed is False
    assert {
        issue.code for issue in forged.issues
    } >= {"OUTCOME_ARTIFACT_REFERENCE_INVALID"}


def test_partial_answer_requires_explicit_acceptance_and_discloses_gap() -> None:
    decision = OutcomeCompletionDecision(
        overall_status=OutcomeCompletionStatus.PARTIAL,
        outcomes=[
            _satisfied_decision().outcomes[0],
            UserOutcomeAssessment(
                outcome_id="refunds.detail",
                requirement="return the requested refund detail",
                source_spans=["refund detail"],
                status=OutcomeCompletionStatus.INSUFFICIENT_EVIDENCE,
                evidence_kind=OutcomeEvidenceKind.DATA,
                missing_reason="refund detail evidence is unavailable",
            ),
        ],
        missing_requirements=["refund detail evidence is unavailable"],
    )
    verifier = OutcomeCompletionVerifier()
    blocked = verifier.verify(
        decision,
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=True,
        allow_partial=False,
    )
    accepted = verifier.verify(
        decision,
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=True,
        allow_partial=True,
    )

    assert blocked.completion_allowed is False
    assert accepted.completion_allowed is True
    assert accepted.partial_answer is True
    rendered = render_partial_outcome_gaps(
        "The order count is 12.", accepted.missing_requirements
    )
    assert "### 未完成项" in rendered
    assert "refund detail evidence is unavailable" in rendered


def test_data_outcome_and_claim_checks_cannot_be_overridden_by_model() -> None:
    no_artifact = _satisfied_decision().model_copy(
        update={
            "outcomes": [
                _satisfied_decision().outcomes[0].model_copy(
                    update={"query_artifact_ids": []}
                )
            ]
        }
    )
    verifier = OutcomeCompletionVerifier()

    missing_artifact = verifier.verify(
        no_artifact,
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=True,
    )
    failed_claims = verifier.verify(
        _satisfied_decision(),
        answer_markdown="The order count is 12.",
        query_artifact_evidence={
            "artifact.orders": ["semantic:metric:orders"]
        },
        rule_artifact_evidence={},
        data_outcome_required=True,
        rule_outcome_required=False,
        claim_verification_passed=False,
    )

    assert missing_artifact.completion_allowed is False
    assert failed_claims.completion_allowed is False
    assert "ANSWER_CLAIM_VERIFICATION_REQUIRED" in {
        issue.code for issue in failed_claims.issues
    }
