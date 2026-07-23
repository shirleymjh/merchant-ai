from __future__ import annotations

import json
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from pydantic import ConfigDict, Field, ValidationError

from merchant_ai.models import APIModel, to_camel
from merchant_ai.services.grounded_answer_coverage import answer_fingerprint


class OutcomeCompletionStatus(str, Enum):
    SATISFIED = "SATISFIED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"


class OutcomeEvidenceKind(str, Enum):
    DATA = "DATA"
    RULE = "RULE"
    DATA_AND_RULE = "DATA_AND_RULE"
    NONE = "NONE"


class UserOutcomeAssessment(APIModel):
    """One user-visible result assessed by the isolated completion model."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        use_enum_values=True,
        extra="forbid",
    )

    outcome_id: str
    requirement: str
    source_spans: list[str] = Field(default_factory=list)
    status: OutcomeCompletionStatus
    evidence_kind: OutcomeEvidenceKind = OutcomeEvidenceKind.NONE
    query_artifact_ids: list[str] = Field(default_factory=list)
    rule_artifact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    missing_reason: str = ""


class OutcomeCompletionDecision(APIModel):
    """Strict semantic decision. It carries no execution authority."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        use_enum_values=True,
        extra="forbid",
    )

    overall_status: OutcomeCompletionStatus
    outcomes: list[UserOutcomeAssessment] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    needs_user_input_reason: str = ""
    summary: str = ""


class OutcomeCompletionIssue(APIModel):
    code: str
    message: str
    blocking: bool = True
    outcome_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class OutcomeCompletionResult(APIModel):
    passed: bool = False
    completion_allowed: bool = False
    partial_answer: bool = False
    needs_user_input: bool = False
    answer_fingerprint: str = ""
    source: str = "compose_verified_answer"
    evaluator_used: bool = True
    overall_status: OutcomeCompletionStatus = OutcomeCompletionStatus.INSUFFICIENT_EVIDENCE
    outcomes: list[UserOutcomeAssessment] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    query_artifact_ids: list[str] = Field(default_factory=list)
    rule_artifact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    issues: list[OutcomeCompletionIssue] = Field(default_factory=list)


class StructuredOutcomeCompletionProvider:
    """Zero-tool model adapter that judges only user-visible completeness."""

    _SYSTEM_PROMPT = """You are an isolated final-answer completion evaluator for an enterprise BI agent.
You receive the original user question, a compact Goal ledger, a candidate answer, summaries of server-verified artifacts, and known gaps.
Identify the small set of user-visible outcomes requested by the question and decide whether each one is satisfied, partial, unsupported, or genuinely needs user input.
Goals are planning and evidence hints, not a requirement to expose every internal metric, time, dimension, join, helper query, or exploration step as a separate final outcome.
Judge semantic completeness only. Never plan another query, write SQL, call a tool, change permissions, or declare an artifact valid.
For every satisfied data outcome, cite the exact queryArtifactIds supplied in the input. For every satisfied rule outcome, cite the exact ruleArtifactIds supplied in the input. Cite only supplied evidenceRefs.
Do not mark the answer SATISFIED when a user-visible request is omitted. Use NEEDS_USER_INPUT only for genuine business ambiguity, not for an internal failure or missing query work.
Return only the strict structured schema."""

    def __init__(self, model: Any) -> None:
        if model is None:
            raise ValueError("structured outcome completion model is required")
        self.model = model

    def evaluate(
        self,
        *,
        question: str,
        required_goals: Sequence[Mapping[str, Any]],
        candidate_answer: str,
        verified_query_artifacts: Sequence[Mapping[str, Any]],
        verified_rule_artifacts: Sequence[Mapping[str, Any]],
        known_gaps: Sequence[Mapping[str, Any]],
        timeout_seconds: float,
    ) -> OutcomeCompletionDecision:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        bind = getattr(self.model, "with_structured_output", None)
        if not callable(bind):
            raise RuntimeError("STRUCTURED_OUTCOME_COMPLETION_MODEL_REQUIRED")
        structured_model = bind(
            OutcomeCompletionDecision,
            method="json_schema",
            strict=True,
        )
        payload = {
            "question": str(question or "").strip(),
            "requiredGoals": [dict(item) for item in required_goals],
            "candidateAnswer": str(candidate_answer or "").strip(),
            "verifiedQueryArtifacts": [dict(item) for item in verified_query_artifacts],
            "verifiedRuleArtifacts": [dict(item) for item in verified_rule_artifacts],
            "knownGaps": [dict(item) for item in known_gaps],
        }
        raw = structured_model.invoke(
            [
                ("system", self._SYSTEM_PROMPT),
                (
                    "human",
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ),
                ),
            ]
        )
        return self._parse_decision(raw)

    @staticmethod
    def _parse_decision(value: Any) -> OutcomeCompletionDecision:
        if isinstance(value, OutcomeCompletionDecision):
            return value
        if isinstance(value, Mapping):
            return OutcomeCompletionDecision.model_validate(dict(value))
        content: Optional[Any] = getattr(value, "content", None)
        if isinstance(content, Mapping):
            return OutcomeCompletionDecision.model_validate(dict(content))
        raise TypeError("structured outcome completion model returned an invalid value")


class OutcomeCompletionVerifier:
    """Deterministically checks every authority claim made by the evaluator."""

    def verify(
        self,
        decision: OutcomeCompletionDecision | Mapping[str, Any],
        *,
        answer_markdown: str,
        query_artifact_evidence: Mapping[str, Sequence[str]],
        rule_artifact_evidence: Mapping[str, Sequence[str]],
        data_outcome_required: bool,
        rule_outcome_required: bool,
        claim_verification_passed: bool,
        allow_partial: bool = False,
    ) -> OutcomeCompletionResult:
        parsed = (
            decision
            if isinstance(decision, OutcomeCompletionDecision)
            else OutcomeCompletionDecision.model_validate(decision)
        )
        answer = str(answer_markdown or "").strip()
        issues: list[OutcomeCompletionIssue] = []
        if not answer:
            issues.append(
                OutcomeCompletionIssue(
                    code="FINAL_ANSWER_MARKDOWN_REQUIRED",
                    message="candidate answer is empty",
                )
            )
        if not claim_verification_passed:
            issues.append(
                OutcomeCompletionIssue(
                    code="ANSWER_CLAIM_VERIFICATION_REQUIRED",
                    message="candidate answer did not pass deterministic claim verification",
                )
            )
        if not parsed.outcomes:
            issues.append(
                OutcomeCompletionIssue(
                    code="USER_OUTCOME_ASSESSMENT_REQUIRED",
                    message="completion evaluator returned no user-visible outcomes",
                )
            )

        outcome_ids: set[str] = set()
        referenced_query_ids: list[str] = []
        referenced_rule_ids: list[str] = []
        referenced_evidence_refs: list[str] = []
        satisfied_count = 0
        unsatisfied_count = 0
        needs_user_input = False
        for outcome in parsed.outcomes:
            outcome_id = str(outcome.outcome_id or "").strip()
            if not outcome_id or outcome_id in outcome_ids:
                issues.append(
                    OutcomeCompletionIssue(
                        code="OUTCOME_ID_INVALID",
                        message="outcome IDs must be non-empty and unique",
                        outcome_id=outcome_id,
                    )
                )
            outcome_ids.add(outcome_id)
            status = OutcomeCompletionStatus(outcome.status)
            evidence_kind = OutcomeEvidenceKind(outcome.evidence_kind)
            query_ids = _dedupe(outcome.query_artifact_ids)
            rule_ids = _dedupe(outcome.rule_artifact_ids)
            evidence_refs = _dedupe(outcome.evidence_refs)
            referenced_query_ids.extend(query_ids)
            referenced_rule_ids.extend(rule_ids)
            referenced_evidence_refs.extend(evidence_refs)

            unknown_query_ids = [item for item in query_ids if item not in query_artifact_evidence]
            unknown_rule_ids = [item for item in rule_ids if item not in rule_artifact_evidence]
            if unknown_query_ids or unknown_rule_ids:
                issues.append(
                    OutcomeCompletionIssue(
                        code="OUTCOME_ARTIFACT_REFERENCE_INVALID",
                        message="completion evaluator cited an unknown or unauthorized artifact",
                        outcome_id=outcome_id,
                        details={
                            "unknownQueryArtifactIds": unknown_query_ids,
                            "unknownRuleArtifactIds": unknown_rule_ids,
                        },
                    )
                )

            allowed_refs = {
                str(ref or "").strip()
                for artifact_id in query_ids
                for ref in query_artifact_evidence.get(artifact_id, ())
                if str(ref or "").strip()
            }
            allowed_refs.update(
                str(ref or "").strip()
                for artifact_id in rule_ids
                for ref in rule_artifact_evidence.get(artifact_id, ())
                if str(ref or "").strip()
            )
            unknown_refs = [item for item in evidence_refs if item not in allowed_refs]
            if unknown_refs:
                issues.append(
                    OutcomeCompletionIssue(
                        code="OUTCOME_EVIDENCE_REFERENCE_INVALID",
                        message="completion evaluator cited evidence outside its verified artifacts",
                        outcome_id=outcome_id,
                        details={"unknownEvidenceRefs": unknown_refs},
                    )
                )

            if status == OutcomeCompletionStatus.SATISFIED:
                satisfied_count += 1
                if evidence_kind == OutcomeEvidenceKind.NONE and (
                    data_outcome_required or rule_outcome_required
                ):
                    issues.append(
                        OutcomeCompletionIssue(
                            code="SATISFIED_OUTCOME_EVIDENCE_KIND_REQUIRED",
                            message=(
                                "a satisfied governed outcome must declare its "
                                "verified evidence kind"
                            ),
                            outcome_id=outcome_id,
                        )
                    )
                if evidence_kind in {
                    OutcomeEvidenceKind.DATA,
                    OutcomeEvidenceKind.DATA_AND_RULE,
                } and not query_ids:
                    issues.append(
                        OutcomeCompletionIssue(
                            code="SATISFIED_DATA_OUTCOME_ARTIFACT_REQUIRED",
                            message="a satisfied data outcome must cite a verified query artifact",
                            outcome_id=outcome_id,
                        )
                    )
                if evidence_kind in {
                    OutcomeEvidenceKind.RULE,
                    OutcomeEvidenceKind.DATA_AND_RULE,
                } and not rule_ids:
                    issues.append(
                        OutcomeCompletionIssue(
                            code="SATISFIED_RULE_OUTCOME_ARTIFACT_REQUIRED",
                            message="a satisfied rule outcome must cite a verified rule artifact",
                            outcome_id=outcome_id,
                        )
                    )
            else:
                unsatisfied_count += 1
                needs_user_input = needs_user_input or status == OutcomeCompletionStatus.NEEDS_USER_INPUT
                if not str(outcome.missing_reason or "").strip():
                    issues.append(
                        OutcomeCompletionIssue(
                            code="OUTCOME_MISSING_REASON_REQUIRED",
                            message="an incomplete outcome must explain what is missing",
                            outcome_id=outcome_id,
                        )
                    )

        overall_status = OutcomeCompletionStatus(parsed.overall_status)
        if overall_status == OutcomeCompletionStatus.SATISFIED:
            if unsatisfied_count or parsed.missing_requirements:
                issues.append(
                    OutcomeCompletionIssue(
                        code="OUTCOME_STATUS_INCONSISTENT",
                        message="overall SATISFIED conflicts with incomplete outcomes",
                    )
                )
            if data_outcome_required and not _dedupe(referenced_query_ids):
                issues.append(
                    OutcomeCompletionIssue(
                        code="VERIFIED_QUERY_ARTIFACT_REQUIRED",
                        message="a complete data answer must cite a verified query artifact",
                    )
                )
            if rule_outcome_required and not _dedupe(referenced_rule_ids):
                issues.append(
                    OutcomeCompletionIssue(
                        code="VERIFIED_RULE_ARTIFACT_REQUIRED",
                        message="a complete rule answer must cite a verified rule artifact",
                    )
                )
        elif overall_status == OutcomeCompletionStatus.PARTIAL:
            if not satisfied_count or not (unsatisfied_count or parsed.missing_requirements):
                issues.append(
                    OutcomeCompletionIssue(
                        code="OUTCOME_STATUS_INCONSISTENT",
                        message="overall PARTIAL requires both completed and missing user-visible work",
                    )
                )
        elif overall_status == OutcomeCompletionStatus.NEEDS_USER_INPUT:
            needs_user_input = True

        authority_issues = any(issue.blocking for issue in issues)
        complete = overall_status == OutcomeCompletionStatus.SATISFIED
        partial = overall_status == OutcomeCompletionStatus.PARTIAL
        completion_allowed = bool(
            not authority_issues
            and (
                complete
                or (
                    partial
                    and allow_partial
                    and satisfied_count > 0
                    and not needs_user_input
                )
            )
        )
        if not completion_allowed and not authority_issues:
            issues.append(
                OutcomeCompletionIssue(
                    code=(
                        "USER_INPUT_REQUIRED"
                        if needs_user_input
                        else "USER_OUTCOME_INCOMPLETE"
                    ),
                    message=(
                        parsed.needs_user_input_reason
                        if needs_user_input
                        else "one or more user-visible outcomes remain incomplete"
                    ),
                )
            )
        return OutcomeCompletionResult(
            passed=completion_allowed,
            completion_allowed=completion_allowed,
            partial_answer=bool(completion_allowed and partial),
            needs_user_input=needs_user_input,
            answer_fingerprint=answer_fingerprint(answer),
            overall_status=overall_status,
            outcomes=[item.model_copy(deep=True) for item in parsed.outcomes],
            missing_requirements=_dedupe(
                [
                    *parsed.missing_requirements,
                    *[
                        item.missing_reason
                        for item in parsed.outcomes
                        if OutcomeCompletionStatus(item.status)
                        != OutcomeCompletionStatus.SATISFIED
                    ],
                ]
            ),
            query_artifact_ids=_dedupe(referenced_query_ids),
            rule_artifact_ids=_dedupe(referenced_rule_ids),
            evidence_refs=_dedupe(referenced_evidence_refs),
            issues=issues,
        )


def outcome_attestation_matches(
    answer_markdown: str,
    result: OutcomeCompletionResult | Mapping[str, Any] | None,
) -> bool:
    if not result:
        return False
    try:
        parsed = (
            result
            if isinstance(result, OutcomeCompletionResult)
            else OutcomeCompletionResult.model_validate(result)
        )
    except (TypeError, ValueError, ValidationError):
        return False
    return bool(
        parsed.passed
        and parsed.completion_allowed
        and parsed.source == "compose_verified_answer"
        and parsed.answer_fingerprint
        and parsed.answer_fingerprint == answer_fingerprint(answer_markdown)
    )


def render_partial_outcome_gaps(
    answer_markdown: str,
    missing_requirements: Sequence[str],
) -> str:
    missing = _dedupe(missing_requirements)
    answer = str(answer_markdown or "").strip()
    if not missing:
        return answer
    section = "### 未完成项\n\n" + "\n".join("- %s" % item for item in missing[:6])
    return "\n\n".join(item for item in (answer, section) if item)


def _dedupe(values: Sequence[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value or "").strip()
            for value in values
            if str(value or "").strip()
        )
    )
