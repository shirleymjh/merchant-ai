from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Mapping, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_answer_coverage import GoalAnswerBinding
from merchant_ai.services.grounded_goal_contract import (
    AnalysisGoalProofResolution,
    AnalysisQuestionGoal,
    ComparisonGoalProofResolution,
    ComparisonQuestionGoal,
    GoalCoverageResult,
    OriginalQuestionGoalContract,
    VerifiedArtifactGoalCoverage,
    canonical_goal_id,
    declare_verified_artifact_goal_coverage,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
    required_goal_ids,
)


MIN_CORRELATION_SAMPLES = 5
MAX_DERIVED_ROW_PREVIEW = 100


class _StrictAnalysisModel(APIModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=APIModel.model_config.get("alias_generator"),
        use_enum_values=True,
    )


class GroundedAnalysisSeriesBinding(_StrictAnalysisModel):
    series_id: str
    artifact_id: str
    value_column: str


class GroundedAnalysisBaselinePair(_StrictAnalysisModel):
    current_series_id: str
    baseline_series_id: str


class GroundedRunSkillAnalysisPublicationRequest(_StrictAnalysisModel):
    """The only structured payload an isolated analysis Skill may publish.

    It intentionally has no ``analysisType``, rows, result, conclusion, or
    prose field.  The type comes from a typed ANALYSIS/COMPARISON goal and all
    data is reloaded from the Kernel's verified query ledger.
    """

    analysis_goal_id: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    series_bindings: list[GroundedAnalysisSeriesBinding] = Field(default_factory=list)
    observation_keys: list[str] = Field(default_factory=list)
    method: str = ""
    normalization_method: str = ""
    left_series_id: str = ""
    right_series_id: str = ""
    baseline_pairs: list[GroundedAnalysisBaselinePair] = Field(default_factory=list)
    anomaly_threshold: float | None = None
    minimum_samples: int = Field(default=MIN_CORRELATION_SAMPLES, ge=1, le=10_000)

    @model_validator(mode="after")
    def _validate_references(self) -> "GroundedRunSkillAnalysisPublicationRequest":
        self.analysis_goal_id = canonical_goal_id(self.analysis_goal_id)
        self.input_artifact_ids = _dedupe_strings(self.input_artifact_ids)
        self.observation_keys = _dedupe_strings(self.observation_keys)
        if not self.input_artifact_ids:
            raise ValueError("analysis publication requires verified input artifact IDs")
        if not self.series_bindings:
            raise ValueError("analysis publication requires at least one series binding")
        series_ids = [str(item.series_id or "").strip() for item in self.series_bindings]
        if any(not item for item in series_ids) or len(set(series_ids)) != len(series_ids):
            raise ValueError("analysis series IDs must be non-empty and unique")
        input_ids = set(self.input_artifact_ids)
        unknown_inputs = sorted(
            {item.artifact_id for item in self.series_bindings if item.artifact_id not in input_ids}
        )
        if unknown_inputs:
            raise ValueError("analysis series reference undeclared input artifacts: %s" % ",".join(unknown_inputs))
        known_series = set(series_ids)
        pointers = [
            self.left_series_id,
            self.right_series_id,
            *[value for pair in self.baseline_pairs for value in (pair.current_series_id, pair.baseline_series_id)],
        ]
        unknown_series = sorted({value for value in pointers if value and value not in known_series})
        if unknown_series:
            raise ValueError("analysis publication references unknown series: %s" % ",".join(unknown_series))
        return self


class GroundedVerifiedAnalysisInput(_StrictAnalysisModel):
    artifact_id: str
    goal_ids: list[str] = Field(default_factory=list)
    row_ref: str
    rows_hash: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    output_lineage: dict[str, list[str]] = Field(default_factory=dict)


class GroundedAnalysisSkillInput(_StrictAnalysisModel):
    input_contract_version: str = "grounded_analysis_skill_input.v1"
    goal_contract_fingerprint: str
    analysis_goal_id: str
    analysis_type: str
    input_goal_ids: list[str] = Field(default_factory=list)
    baseline_goal_ids: list[str] = Field(default_factory=list)
    verified_inputs: list[GroundedVerifiedAnalysisInput] = Field(default_factory=list)
    publication_interface: dict[str, Any] = Field(default_factory=dict)


class GroundedAnalysisDataInputGateIssue(_StrictAnalysisModel):
    code: str
    message: str
    goal_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class GroundedAnalysisDataInputGateResult(_StrictAnalysisModel):
    """Pre-Skill gate; deliberately weaker than final Goal coverage."""

    passed: bool = False
    skill_start_allowed: bool = False
    goal_contract_fingerprint: str = ""
    deferred_goal_ids: list[str] = Field(default_factory=list)
    deferred_input_goal_ids_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    required_query_goal_ids: list[str] = Field(default_factory=list)
    proved_query_goal_ids: list[str] = Field(default_factory=list)
    missing_proved_input_goal_ids: list[str] = Field(default_factory=list)
    unresolved_query_goal_ids: list[str] = Field(default_factory=list)
    input_artifact_ids_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    verified_input_artifact_ids: list[str] = Field(default_factory=list)
    issues: list[GroundedAnalysisDataInputGateIssue] = Field(default_factory=list)


class GroundedAnalysisEvidenceGap(_StrictAnalysisModel):
    code: str
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class GroundedDerivedAnalysisVerification(_StrictAnalysisModel):
    passed: bool = True
    publication_status: Literal["PROVED", "INSUFFICIENT_EVIDENCE"]
    evidence_refs: list[str] = Field(default_factory=list)
    gaps: list[GroundedAnalysisEvidenceGap] = Field(default_factory=list)


class GroundedDerivedAnalysisArtifact(_StrictAnalysisModel):
    artifact_type: str = "grounded_derived_analysis.v1"
    artifact_id: str
    goal_contract_fingerprint: str
    analysis_goal_id: str
    goal_kind: Literal["ANALYSIS", "COMPARISON"] = "ANALYSIS"
    analysis_type: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    input_row_refs: list[str] = Field(default_factory=list)
    input_lineage: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    method: str = ""
    normalization_method: str = ""
    publication_status: Literal["PROVED", "INSUFFICIENT_EVIDENCE"]
    result_ref: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    correlation_not_causation: bool = False
    causal_claim_allowed: bool = False
    goal_resolutions: list[AnalysisGoalProofResolution | ComparisonGoalProofResolution] = Field(default_factory=list)
    verified_evidence: GroundedDerivedAnalysisVerification
    created_at: str = ""


class GroundedAnalysisRenderResult(_StrictAnalysisModel):
    """Trusted visible span plus the final-answer coverage binding."""

    answer_markdown: str
    binding: GoalAnswerBinding


class _AnalysisEvidenceInsufficient(RuntimeError):
    def __init__(self, code: str, reason: str, **details: Any):
        self.gap = GroundedAnalysisEvidenceGap(
            code=code,
            reason=reason,
            details=details,
        )
        super().__init__(code)


@dataclass(frozen=True)
class _SeriesData:
    binding: GroundedAnalysisSeriesBinding
    source: GroundedVerifiedAnalysisInput
    values: tuple[Decimal, ...]
    keyed_values: dict[tuple[Any, ...], Decimal]
    value_lineage: tuple[str, ...]
    observation_lineage: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _AnalysisComputation:
    method: str
    normalization_method: str
    result: dict[str, Any]
    baseline_refs: tuple[str, ...] = ()
    correlation_not_causation: bool = False


def grounded_analysis_run_skill_publication_schema() -> dict[str, Any]:
    """Return the narrow schema mounted for a future ``run_skill`` publisher."""

    schema = GroundedRunSkillAnalysisPublicationRequest.model_json_schema(by_alias=True)
    schema["x-grounded-boundary"] = {
        "analysisTypeSource": ("typed ANALYSIS.analysisType or typed COMPARISON.comparisonType only"),
        "dataSource": "verified query artifact IDs; Kernel reloads rows and lineage",
        "forbiddenOutputs": [
            "analysisType",
            "rows",
            "result",
            "conclusion",
            "answerMarkdown",
            "causalClaim",
        ],
    }
    return schema


def verify_grounded_analysis_data_input_coverage(
    *,
    goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    query_goal_coverage: GoalCoverageResult | Mapping[str, Any],
) -> GroundedAnalysisDataInputGateResult:
    """Allow Skill startup only after every derived goal input is proved.

    This gate exists because query artifacts cannot prove interpretive goals.
    It defers only structurally typed goals: every ``ANALYSIS`` goal and a
    ``COMPARISON`` whose ``comparisonType`` is explicitly anomaly/correlation.
    Labels and the original question are never inspected.

    Passing this gate does *not* authorize a final answer.  After the Skill
    publishes ``GroundedDerivedAnalysisArtifact``, callers must run the normal
    complete Goal coverage gate and the final answer coverage gate.
    """

    contract = parse_original_question_goal_contract(goal_contract)
    coverage = (
        query_goal_coverage
        if isinstance(query_goal_coverage, GoalCoverageResult)
        else GoalCoverageResult.model_validate(query_goal_coverage)
    )
    fingerprint = original_question_goal_contract_fingerprint(contract)
    issues: list[GroundedAnalysisDataInputGateIssue] = []
    if not coverage.goal_contract_fingerprint:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="DATA_INPUT_GATE_CONTRACT_FINGERPRINT_REQUIRED",
                message="query Goal coverage has no Goal Contract fingerprint",
            )
        )
    elif coverage.goal_contract_fingerprint != fingerprint:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="DATA_INPUT_GATE_CONTRACT_FINGERPRINT_MISMATCH",
                message="query Goal coverage belongs to another Goal Contract",
            )
        )

    deferred_inputs: dict[str, list[str]] = {}
    for goal in contract.goals:
        if isinstance(goal, AnalysisQuestionGoal):
            deferred_inputs[goal.goal_id] = _dedupe_strings([*goal.input_goal_ids, *goal.baseline_goal_ids])
        elif isinstance(goal, ComparisonQuestionGoal) and _canonical_analysis_type(goal.comparison_type) in {
            "ANOMALY",
            "CORRELATION",
        }:
            deferred_inputs[goal.goal_id] = _dedupe_strings([*goal.left_goal_ids, *goal.right_goal_ids])

    deferred_goal_ids = [goal.goal_id for goal in contract.goals if goal.goal_id in deferred_inputs]
    if not deferred_goal_ids:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="DATA_INPUT_GATE_NO_TYPED_DERIVED_GOAL",
                message="Goal Contract has no typed analysis/anomaly/correlation goal to defer",
            )
        )

    prematurely_covered = sorted(set(deferred_goal_ids).intersection(coverage.covered_goal_ids))
    for goal_id in prematurely_covered:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="QUERY_ARTIFACT_CANNOT_PROVE_DERIVED_GOAL",
                message="query-only coverage cannot prove a deferred analysis goal",
                goal_id=goal_id,
            )
        )

    required_ids = required_goal_ids(contract)
    required_query_ids = [goal_id for goal_id in required_ids if goal_id not in deferred_inputs]
    proved = set(coverage.covered_goal_ids)
    resolved = set(coverage.resolved_goal_ids)
    required_derived_inputs = _dedupe_strings(
        input_goal_id for goal_id in deferred_goal_ids for input_goal_id in deferred_inputs[goal_id]
    )
    missing_proved_inputs = [goal_id for goal_id in required_derived_inputs if goal_id not in proved]
    for goal_id in missing_proved_inputs:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="DERIVED_GOAL_INPUT_NOT_PROVED",
                message="analysis input/baseline/operand is not proved by query artifacts",
                goal_id=goal_id,
            )
        )
    unresolved_query_ids = [
        goal_id for goal_id in required_query_ids if goal_id not in resolved and goal_id not in proved
    ]
    for goal_id in unresolved_query_ids:
        issues.append(
            GroundedAnalysisDataInputGateIssue(
                code="NON_DERIVED_GOAL_UNRESOLVED_BEFORE_SKILL",
                message="a required non-derived goal is unresolved before Skill startup",
                goal_id=goal_id,
            )
        )

    input_artifact_ids_by_goal_id = {
        goal_id: _dedupe_strings(coverage.coverage_by_goal_id.get(goal_id) or [])
        for goal_id in required_derived_inputs
        if goal_id in proved
    }
    for goal_id in required_derived_inputs:
        if goal_id in proved and not input_artifact_ids_by_goal_id.get(goal_id):
            issues.append(
                GroundedAnalysisDataInputGateIssue(
                    code="DERIVED_GOAL_INPUT_ARTIFACT_MAPPING_MISSING",
                    message="proved derived-goal input has no verified artifact mapping",
                    goal_id=goal_id,
                )
            )
    verified_input_artifact_ids = _dedupe_strings(
        artifact_id
        for goal_id in required_derived_inputs
        for artifact_id in input_artifact_ids_by_goal_id.get(goal_id, [])
    )
    passed = not issues
    return GroundedAnalysisDataInputGateResult(
        passed=passed,
        skill_start_allowed=passed,
        goal_contract_fingerprint=fingerprint,
        deferred_goal_ids=deferred_goal_ids,
        deferred_input_goal_ids_by_goal_id=deferred_inputs,
        required_query_goal_ids=required_query_ids,
        proved_query_goal_ids=[goal_id for goal_id in required_query_ids if goal_id in proved],
        missing_proved_input_goal_ids=missing_proved_inputs,
        unresolved_query_goal_ids=unresolved_query_ids,
        input_artifact_ids_by_goal_id=input_artifact_ids_by_goal_id,
        verified_input_artifact_ids=verified_input_artifact_ids,
        issues=issues,
    )


def build_grounded_analysis_skill_input(
    *,
    goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    analysis_goal_id: str,
    requested_artifact_ids: Sequence[str],
    verified_query_artifacts: Sequence[Any],
    artifact_goal_ids: Mapping[str, Sequence[str]],
) -> GroundedAnalysisSkillInput:
    """Mount verified rows/lineage for one typed derived-analysis goal."""

    contract, goal, analysis_type = _derived_analysis_goal(
        goal_contract,
        analysis_goal_id,
    )
    inputs = _verified_analysis_inputs(
        requested_artifact_ids,
        verified_query_artifacts,
        artifact_goal_ids,
    )
    _require_goal_inputs(goal, inputs)
    return GroundedAnalysisSkillInput(
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
        analysis_goal_id=goal.goal_id,
        analysis_type=analysis_type,
        input_goal_ids=_derived_goal_input_ids(goal),
        baseline_goal_ids=_derived_goal_baseline_ids(goal),
        verified_inputs=inputs,
        publication_interface={
            "publisher": "publish_grounded_analysis_from_skill",
            "requestSchema": grounded_analysis_run_skill_publication_schema(),
            "policy": (
                "Submit mappings and deterministic method only. The publisher "
                "reloads verified rows and never accepts analysis prose or causes."
            ),
        },
    )


def publish_grounded_analysis_from_skill(
    *,
    goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    publication_request: GroundedRunSkillAnalysisPublicationRequest | Mapping[str, Any],
    verified_query_artifacts: Sequence[Any],
    artifact_goal_ids: Mapping[str, Sequence[str]],
) -> GroundedDerivedAnalysisArtifact:
    """Validate and deterministically publish one isolated Skill result.

    The Skill never supplies analysis type, input rows, a computed result, or
    prose.  It only maps verified columns to a supported deterministic method.
    """

    request = (
        publication_request
        if isinstance(
            publication_request,
            GroundedRunSkillAnalysisPublicationRequest,
        )
        else GroundedRunSkillAnalysisPublicationRequest.model_validate(publication_request)
    )
    contract, goal, analysis_type = _derived_analysis_goal(
        goal_contract,
        request.analysis_goal_id,
    )
    inputs = _verified_analysis_inputs(
        request.input_artifact_ids,
        verified_query_artifacts,
        artifact_goal_ids,
    )
    _require_goal_inputs(goal, inputs)
    gaps: list[GroundedAnalysisEvidenceGap] = []
    computation: _AnalysisComputation | None = None
    try:
        computation = _compute_analysis(
            analysis_type,
            request,
            inputs,
        )
    except _AnalysisEvidenceInsufficient as exc:
        gaps.append(exc.gap)

    status: Literal["PROVED", "INSUFFICIENT_EVIDENCE"] = (
        "PROVED" if computation is not None else "INSUFFICIENT_EVIDENCE"
    )
    method = computation.method if computation is not None else _canonical_token(request.method)
    normalization = (
        computation.normalization_method if computation is not None else _canonical_token(request.normalization_method)
    )
    result = computation.result if computation is not None else {}
    correlation_disclosure = bool(computation is not None and computation.correlation_not_causation)
    contract_fingerprint = original_question_goal_contract_fingerprint(contract)
    fingerprint_payload = {
        "goalContractFingerprint": contract_fingerprint,
        "analysisGoalId": goal.goal_id,
        "analysisType": analysis_type,
        "request": request.model_dump(by_alias=False, exclude_none=False),
        "inputRows": [{"artifactId": item.artifact_id, "rowsHash": item.rows_hash} for item in inputs],
        "status": status,
        "method": method,
        "normalization": normalization,
        "result": result,
        "gaps": [item.model_dump(by_alias=False) for item in gaps],
    }
    artifact_id = (
        "analysis_artifact_%s" % hashlib.sha256(_stable_json(fingerprint_payload).encode("utf-8")).hexdigest()[:16]
    )
    result_ref = "analysis:%s:result" % artifact_id if status == "PROVED" else ""
    input_row_refs = [item.row_ref for item in inputs]
    gap_refs = ["analysis:%s:gap:%s" % (artifact_id, item.code.lower()) for item in gaps]
    baseline_refs = list(computation.baseline_refs) if computation else []
    input_goal_ids = _derived_goal_input_ids(goal)
    if status == "PROVED" and isinstance(goal, AnalysisQuestionGoal):
        resolution = AnalysisGoalProofResolution(
            goal_id=goal.goal_id,
            goal_kind="ANALYSIS",
            resolution="PROVED",
            proof_type="DETERMINISTIC_DERIVED_ANALYSIS",
            evidence_refs=_dedupe_strings([*input_row_refs, *baseline_refs, result_ref]),
            analysis_type=analysis_type,
            input_goal_ids=input_goal_ids,
            analysis_method=method,
            result_ref=result_ref,
            baseline_refs=baseline_refs,
            normalization_method=normalization,
            details={
                "correlationNotCausation": correlation_disclosure,
                "causalClaimAllowed": False,
            },
        )
    elif status == "PROVED":
        resolution = ComparisonGoalProofResolution(
            goal_id=goal.goal_id,
            goal_kind="COMPARISON",
            resolution="PROVED",
            proof_type="DETERMINISTIC_DERIVED_ANALYSIS",
            evidence_refs=_dedupe_strings([*input_row_refs, *baseline_refs, result_ref]),
            comparison_type=analysis_type,
            operand_goal_ids=input_goal_ids,
            comparison_method=method,
            result_ref=result_ref,
            baseline_refs=baseline_refs,
            normalization_method=normalization,
            details={
                "correlationNotCausation": correlation_disclosure,
                "causalClaimAllowed": False,
            },
        )
    elif isinstance(goal, AnalysisQuestionGoal):
        reason = "; ".join(item.reason for item in gaps) or (
            "verified evidence does not support the requested analysis"
        )
        resolution = AnalysisGoalProofResolution(
            goal_id=goal.goal_id,
            goal_kind="ANALYSIS",
            resolution="INSUFFICIENT_EVIDENCE",
            proof_type="DETERMINISTIC_DERIVED_ANALYSIS",
            evidence_refs=gap_refs,
            reason=reason,
            analysis_type=analysis_type,
            input_goal_ids=input_goal_ids,
            analysis_method=method,
            baseline_refs=baseline_refs,
            normalization_method=normalization,
            details={
                "gapCodes": [item.code for item in gaps],
                "correlationNotCausation": analysis_type == "CORRELATION",
                "causalClaimAllowed": False,
            },
        )
    else:
        reason = "; ".join(item.reason for item in gaps) or (
            "verified evidence does not support the requested comparison"
        )
        resolution = ComparisonGoalProofResolution(
            goal_id=goal.goal_id,
            goal_kind="COMPARISON",
            resolution="INSUFFICIENT_EVIDENCE",
            proof_type="DETERMINISTIC_DERIVED_ANALYSIS",
            evidence_refs=gap_refs,
            reason=reason,
            comparison_type=analysis_type,
            operand_goal_ids=input_goal_ids,
            comparison_method=method,
            baseline_refs=baseline_refs,
            normalization_method=normalization,
            details={
                "gapCodes": [item.code for item in gaps],
                "correlationNotCausation": analysis_type == "CORRELATION",
                "causalClaimAllowed": False,
            },
        )
    evidence_refs = _dedupe_strings([*input_row_refs, *baseline_refs, result_ref, *gap_refs])
    return GroundedDerivedAnalysisArtifact(
        artifact_id=artifact_id,
        goal_contract_fingerprint=contract_fingerprint,
        analysis_goal_id=goal.goal_id,
        goal_kind=goal.kind,
        analysis_type=analysis_type,
        input_artifact_ids=[item.artifact_id for item in inputs],
        input_row_refs=input_row_refs,
        input_lineage={item.artifact_id: item.output_lineage for item in inputs},
        method=method,
        normalization_method=normalization,
        publication_status=status,
        result_ref=result_ref,
        result=result,
        correlation_not_causation=(correlation_disclosure or analysis_type == "CORRELATION"),
        causal_claim_allowed=False,
        goal_resolutions=[resolution],
        verified_evidence=GroundedDerivedAnalysisVerification(
            publication_status=status,
            evidence_refs=evidence_refs,
            gaps=gaps,
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def grounded_analysis_goal_coverage(
    goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    artifact: GroundedDerivedAnalysisArtifact,
) -> VerifiedArtifactGoalCoverage:
    """Build the sidecar consumed by the existing goal coverage gate."""

    return declare_verified_artifact_goal_coverage(
        goal_contract,
        artifact,
        [artifact.analysis_goal_id],
        evidence_refs=artifact.verified_evidence.evidence_refs,
        goal_resolutions=artifact.goal_resolutions,
    )


def render_grounded_analysis_artifact(
    artifact: GroundedDerivedAnalysisArtifact,
) -> GroundedAnalysisRenderResult:
    """Render a deterministic answer span from a verified derived artifact.

    Isolated Skill prose is never accepted.  This renderer owns both the
    visible conclusion and its ``VERIFIED_ANALYSIS_ARTIFACT_RENDERER`` binding.
    """

    if not artifact.verified_evidence.passed:
        raise ValueError("analysis artifact is not verified")
    if artifact.publication_status == "INSUFFICIENT_EVIDENCE":
        gap_refs = [ref for ref in artifact.verified_evidence.evidence_refs if ":gap:" in ref]
        if not artifact.verified_evidence.gaps or not gap_refs:
            raise ValueError("insufficient analysis artifact requires a typed gap and ref")
        reasons = _dedupe_strings(item.reason for item in artifact.verified_evidence.gaps)
        span = "证据不足：%s" % "；".join(reasons)
        markdown = "%s\n\n证据缺口：`%s`" % (span, gap_refs[0])
        return GroundedAnalysisRenderResult(
            answer_markdown=markdown,
            binding=GoalAnswerBinding(
                goal_id=artifact.analysis_goal_id,
                resolution="INSUFFICIENT_EVIDENCE",
                answer_text=span,
                artifact_ids=[],
                evidence_refs=[gap_refs[0]],
                insufficiency_ref=gap_refs[0],
                renderer="VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
            ),
        )

    if not artifact.result_ref or not artifact.result:
        raise ValueError("proved analysis artifact has no deterministic result")
    span = _render_proved_analysis_span(artifact)
    markdown = "%s\n\n方法：`%s`。证据：`%s`" % (
        span,
        artifact.method,
        artifact.result_ref,
    )
    return GroundedAnalysisRenderResult(
        answer_markdown=markdown,
        binding=GoalAnswerBinding(
            goal_id=artifact.analysis_goal_id,
            resolution="PROVED",
            answer_text=span,
            artifact_ids=[artifact.artifact_id],
            evidence_refs=[artifact.result_ref],
            renderer="VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
        ),
    )


def _render_proved_analysis_span(
    artifact: GroundedDerivedAnalysisArtifact,
) -> str:
    result = artifact.result
    if artifact.analysis_type == "ANOMALY":
        most_anomalous = str(result.get("mostAnomalousSeriesId") or "").strip()
        scores = [item for item in result.get("seriesScores") or [] if isinstance(item, Mapping)]
        selected = next(
            (item for item in scores if str(item.get("currentSeriesId") or "") == most_anomalous),
            scores[0] if scores else {},
        )
        score = str(selected.get("normalizedScore") or "").strip()
        threshold = str(selected.get("threshold") or result.get("threshold") or "").strip()
        if not most_anomalous or not score or not threshold:
            raise ValueError("proved anomaly result is incomplete")
        conclusion = "超过阈值" if bool(selected.get("exceedsThreshold")) else "未超过阈值"
        return "异常分析：最异常序列为 %s，标准化分数为 %s，阈值为 %s，%s。" % (
            most_anomalous,
            score,
            threshold,
            conclusion,
        )
    if artifact.analysis_type == "CORRELATION":
        coefficient = result.get("coefficient")
        sample_count = result.get("sampleCount")
        if coefficient is None or sample_count is None:
            raise ValueError("proved correlation result is incomplete")
        return "相关性分析：系数为 %s，基于 %s 个对齐观测，方向为 %s、强度为 %s；相关不等于因果。" % (
            coefficient,
            sample_count,
            result.get("direction") or "UNKNOWN",
            result.get("strength") or "UNKNOWN",
        )
    if artifact.analysis_type == "TREND":
        profiles = [item for item in result.get("series") or [] if isinstance(item, Mapping)]
        if not profiles:
            raise ValueError("proved trend result is incomplete")
        details = [
            "%s：%s → %s，变化 %s，OLS 斜率 %s"
            % (
                item.get("seriesId") or "series",
                item.get("first"),
                item.get("last"),
                item.get("delta"),
                item.get("olsSlopePerObservation"),
            )
            for item in profiles[:4]
        ]
        return "趋势分析（按 %s）：%s。" % (
            "、".join(str(item) for item in result.get("observationGrain") or []),
            "；".join(details),
        )
    if artifact.analysis_type == "DIFFERENCE":
        rows = [item for item in result.get("rows") or [] if isinstance(item, Mapping)]
        if not rows:
            raise ValueError("proved difference result is incomplete")
        first = rows[0]
        return "差异分析：共对齐 %s 个观测；首个观测左值 %s、右值 %s，差值 %s，相对右值差异 %s%%。" % (
            result.get("alignedRowCount"),
            first.get("left"),
            first.get("right"),
            first.get("difference"),
            first.get("percentDifference"),
        )
    raise ValueError("proved analysis type has no trusted deterministic renderer: %s" % artifact.analysis_type)


def _compute_analysis(
    analysis_type: str,
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> _AnalysisComputation:
    if analysis_type == "ANOMALY":
        return _compute_anomaly(request, inputs)
    if analysis_type == "CORRELATION":
        return _compute_correlation(request, inputs)
    if analysis_type == "DIFFERENCE":
        return _compute_difference(request, inputs)
    if analysis_type == "TREND":
        return _compute_trend(request, inputs)
    if analysis_type in {"CAUSAL", "ATTRIBUTION", "DIAGNOSIS", "IMPACT"}:
        raise _AnalysisEvidenceInsufficient(
            "CAUSAL_INFERENCE_NOT_SUPPORTED",
            ("observational query artifacts can show association but cannot prove that one metric caused another"),
            analysisType=analysis_type,
            correlationNotCausation=True,
        )
    raise _AnalysisEvidenceInsufficient(
        "ANALYSIS_TYPE_UNSUPPORTED",
        "the typed Goal Contract requests an unsupported deterministic analysis",
        analysisType=analysis_type,
    )


def _compute_anomaly(
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> _AnalysisComputation:
    method = _require_method(request.method, {"BASELINE_NORMALIZED_DEVIATION"})
    normalization = _canonical_token(request.normalization_method)
    if normalization not in {"PERCENT_CHANGE", "Z_SCORE"}:
        raise _AnalysisEvidenceInsufficient(
            "ANOMALY_NORMALIZATION_REQUIRED",
            "anomaly publication requires PERCENT_CHANGE or Z_SCORE normalization",
        )
    if not request.baseline_pairs:
        raise _AnalysisEvidenceInsufficient(
            "ANOMALY_COMPARABLE_BASELINE_REQUIRED",
            "anomaly publication requires at least one current/baseline series pair",
        )
    series = _series_map(request, inputs)
    threshold = request.anomaly_threshold
    if normalization == "PERCENT_CHANGE" and (threshold is None or threshold <= 0):
        raise _AnalysisEvidenceInsufficient(
            "ANOMALY_THRESHOLD_REQUIRED",
            "PERCENT_CHANGE anomaly publication requires a positive threshold",
        )
    if normalization == "Z_SCORE" and threshold is None:
        threshold = 2.0
    if threshold is None or threshold <= 0:
        raise _AnalysisEvidenceInsufficient(
            "ANOMALY_THRESHOLD_REQUIRED",
            "anomaly publication requires a positive threshold",
        )

    scored: list[dict[str, Any]] = []
    baseline_refs: list[str] = []
    for pair in request.baseline_pairs:
        current = series[pair.current_series_id]
        baseline = series[pair.baseline_series_id]
        if not set(current.value_lineage).intersection(baseline.value_lineage):
            raise _AnalysisEvidenceInsufficient(
                "ANOMALY_BASELINE_LINEAGE_INCOMPARABLE",
                "current and baseline values do not share verified semantic lineage",
                currentSeriesId=current.binding.series_id,
                baselineSeriesId=baseline.binding.series_id,
            )
        baseline_refs.append(baseline.source.row_ref)
        if normalization == "PERCENT_CHANGE":
            pairs = _aligned_values(current, baseline, request.observation_keys)
            current_total = sum((left for _key, left, _right in pairs), Decimal("0"))
            baseline_total = sum((right for _key, _left, right in pairs), Decimal("0"))
            if baseline_total == 0:
                raise _AnalysisEvidenceInsufficient(
                    "ANOMALY_BASELINE_ZERO",
                    "percent-change normalization cannot use a zero baseline",
                    baselineSeriesId=baseline.binding.series_id,
                )
            score = (current_total - baseline_total) / abs(baseline_total) * Decimal("100")
        else:
            if len(baseline.values) < MIN_CORRELATION_SAMPLES:
                raise _AnalysisEvidenceInsufficient(
                    "ANOMALY_BASELINE_SAMPLE_INSUFFICIENT",
                    "Z_SCORE baseline requires at least five verified observations",
                    baselineSeriesId=baseline.binding.series_id,
                    sampleCount=len(baseline.values),
                )
            baseline_mean = _mean(baseline.values)
            variance = _mean(tuple((value - baseline_mean) ** 2 for value in baseline.values))
            if variance <= 0:
                raise _AnalysisEvidenceInsufficient(
                    "ANOMALY_BASELINE_VARIANCE_ZERO",
                    "Z_SCORE baseline has no variance",
                    baselineSeriesId=baseline.binding.series_id,
                )
            score = Decimal(str((float(_mean(current.values) - baseline_mean)) / math.sqrt(float(variance))))
        scored.append(
            {
                "currentSeriesId": current.binding.series_id,
                "baselineSeriesId": baseline.binding.series_id,
                "normalizedScore": _decimal_text(score),
                "absoluteScore": _decimal_text(abs(score)),
                "direction": "UP" if score > 0 else "DOWN" if score < 0 else "FLAT",
                "threshold": _decimal_text(Decimal(str(threshold))),
                "exceedsThreshold": abs(score) >= Decimal(str(threshold)),
            }
        )
    if not scored:
        raise _AnalysisEvidenceInsufficient(
            "ANOMALY_RESULT_MISSING",
            "no normalized anomaly result could be computed",
        )
    ranked = sorted(
        scored,
        key=lambda item: (
            -abs(_decimal(item["normalizedScore"]) or Decimal("0")),
            str(item["currentSeriesId"]),
        ),
    )
    return _AnalysisComputation(
        method=method,
        normalization_method=normalization,
        baseline_refs=tuple(_dedupe_strings(baseline_refs)),
        result={
            "method": method,
            "normalizationMethod": normalization,
            "threshold": _decimal_text(Decimal(str(threshold))),
            "seriesScores": ranked,
            "mostAnomalousSeriesId": ranked[0]["currentSeriesId"],
            "resultHash": _stable_hash(ranked),
        },
    )


def _compute_correlation(
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> _AnalysisComputation:
    method = _require_method(request.method, {"PEARSON_CORRELATION"})
    if not request.observation_keys:
        raise _AnalysisEvidenceInsufficient(
            "CORRELATION_OBSERVATION_GRAIN_REQUIRED",
            "correlation requires explicit aligned observation keys",
        )
    series = _series_map(request, inputs)
    left, right = _left_right_series(request, series)
    aligned = _aligned_values(left, right, request.observation_keys)
    minimum = max(MIN_CORRELATION_SAMPLES, request.minimum_samples)
    if len(aligned) < minimum:
        raise _AnalysisEvidenceInsufficient(
            "CORRELATION_SAMPLE_INSUFFICIENT",
            "correlation requires at least %d aligned observations" % minimum,
            sampleCount=len(aligned),
            requiredSampleCount=minimum,
        )
    left_values = tuple(item[1] for item in aligned)
    right_values = tuple(item[2] for item in aligned)
    left_mean = _mean(left_values)
    right_mean = _mean(right_values)
    numerator = sum(
        (
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(left_values, right_values)
        ),
        Decimal("0"),
    )
    left_ss = sum(((value - left_mean) ** 2 for value in left_values), Decimal("0"))
    right_ss = sum(((value - right_mean) ** 2 for value in right_values), Decimal("0"))
    if left_ss <= 0 or right_ss <= 0:
        raise _AnalysisEvidenceInsufficient(
            "CORRELATION_VARIANCE_REQUIRED",
            "correlation cannot be computed for a constant series",
        )
    coefficient = float(numerator) / math.sqrt(float(left_ss * right_ss))
    coefficient = max(-1.0, min(1.0, coefficient))
    magnitude = abs(coefficient)
    strength = "STRONG" if magnitude >= 0.7 else "MODERATE" if magnitude >= 0.3 else "WEAK"
    return _AnalysisComputation(
        method=method,
        normalization_method="",
        correlation_not_causation=True,
        result={
            "method": method,
            "leftSeriesId": left.binding.series_id,
            "rightSeriesId": right.binding.series_id,
            "observationGrain": list(request.observation_keys),
            "sampleCount": len(aligned),
            "coefficient": round(coefficient, 12),
            "direction": "POSITIVE" if coefficient > 0 else "NEGATIVE" if coefficient < 0 else "NONE",
            "strength": strength,
            "correlationNotCausation": True,
            "causalClaimAllowed": False,
            "alignedRowsHash": _stable_hash(
                [
                    {
                        "key": list(key),
                        "left": _decimal_text(left_value),
                        "right": _decimal_text(right_value),
                    }
                    for key, left_value, right_value in aligned
                ]
            ),
        },
    )


def _compute_difference(
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> _AnalysisComputation:
    method = _require_method(
        request.method,
        {"ALIGNED_ABSOLUTE_AND_PERCENT_DIFFERENCE"},
    )
    series = _series_map(request, inputs)
    left, right = _left_right_series(request, series)
    aligned = _aligned_values(left, right, request.observation_keys)
    derived_rows: list[dict[str, Any]] = []
    for key, left_value, right_value in aligned:
        difference = left_value - right_value
        percent = difference / abs(right_value) * Decimal("100") if right_value != 0 else None
        derived_rows.append(
            {
                "observation": _observation_payload(request.observation_keys, key),
                "left": _decimal_text(left_value),
                "right": _decimal_text(right_value),
                "difference": _decimal_text(difference),
                "percentDifference": _decimal_text(percent) if percent is not None else None,
            }
        )
    if not derived_rows:
        raise _AnalysisEvidenceInsufficient(
            "DIFFERENCE_RESULT_MISSING",
            "no aligned numeric values are available for difference analysis",
        )
    return _AnalysisComputation(
        method=method,
        normalization_method="RIGHT_VALUE_PERCENT",
        result={
            "method": method,
            "leftSeriesId": left.binding.series_id,
            "rightSeriesId": right.binding.series_id,
            "observationGrain": list(request.observation_keys),
            "alignedRowCount": len(derived_rows),
            "percentDifferenceDenominator": "absolute right value",
            "rows": derived_rows[:MAX_DERIVED_ROW_PREVIEW],
            "rowsTruncated": len(derived_rows) > MAX_DERIVED_ROW_PREVIEW,
            "derivedRowsHash": _stable_hash(derived_rows),
        },
    )


def _compute_trend(
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> _AnalysisComputation:
    method = _require_method(request.method, {"FIRST_LAST_DELTA_WITH_OLS"})
    if not request.observation_keys:
        raise _AnalysisEvidenceInsufficient(
            "TREND_OBSERVATION_GRAIN_REQUIRED",
            "trend analysis requires an explicit ordered observation grain",
        )
    series = _series_map(request, inputs)
    profiles: list[dict[str, Any]] = []
    for series_id in sorted(series):
        item = series[series_id]
        ordered = sorted(
            item.keyed_values.items(),
            key=lambda entry: _observation_sort_key(entry[0]),
        )
        if len(ordered) < 2:
            raise _AnalysisEvidenceInsufficient(
                "TREND_SAMPLE_INSUFFICIENT",
                "trend analysis requires at least two ordered observations",
                seriesId=series_id,
                sampleCount=len(ordered),
            )
        first_key, first = ordered[0]
        last_key, last = ordered[-1]
        delta = last - first
        delta_pct = delta / abs(first) * Decimal("100") if first != 0 else None
        slope = _ols_slope(tuple(value for _key, value in ordered))
        profiles.append(
            {
                "seriesId": series_id,
                "sampleCount": len(ordered),
                "firstObservation": _observation_payload(request.observation_keys, first_key),
                "lastObservation": _observation_payload(request.observation_keys, last_key),
                "first": _decimal_text(first),
                "last": _decimal_text(last),
                "delta": _decimal_text(delta),
                "deltaPercent": _decimal_text(delta_pct) if delta_pct is not None else None,
                "olsSlopePerObservation": _decimal_text(slope),
                "direction": "UP" if slope > 0 else "DOWN" if slope < 0 else "FLAT",
                "orderedPointsHash": _stable_hash(
                    [{"key": list(key), "value": _decimal_text(value)} for key, value in ordered]
                ),
            }
        )
    return _AnalysisComputation(
        method=method,
        normalization_method="",
        result={
            "method": method,
            "observationGrain": list(request.observation_keys),
            "series": profiles,
            "resultHash": _stable_hash(profiles),
        },
    )


def _derived_analysis_goal(
    goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    analysis_goal_id: str,
) -> tuple[
    OriginalQuestionGoalContract,
    AnalysisQuestionGoal | ComparisonQuestionGoal,
    str,
]:
    contract = parse_original_question_goal_contract(goal_contract)
    goal_id = canonical_goal_id(analysis_goal_id)
    goal = contract.goal_map().get(goal_id)
    if isinstance(goal, AnalysisQuestionGoal):
        raw_analysis_type = goal.analysis_type
    elif isinstance(goal, ComparisonQuestionGoal) and _canonical_analysis_type(goal.comparison_type) in {
        "ANOMALY",
        "CORRELATION",
    }:
        raw_analysis_type = goal.comparison_type
    else:
        raise ValueError(
            "grounded analysis may only publish for typed ANALYSIS or anomaly/correlation COMPARISON goals"
        )
    # Deliberately inspect only typed Goal Contract fields.  The original
    # question string and goal label are never read by type resolution.
    analysis_type = _canonical_analysis_type(raw_analysis_type)
    return contract, goal, analysis_type


def _derived_goal_input_ids(
    goal: AnalysisQuestionGoal | ComparisonQuestionGoal,
) -> list[str]:
    if isinstance(goal, AnalysisQuestionGoal):
        return list(goal.input_goal_ids)
    return _dedupe_strings([*goal.left_goal_ids, *goal.right_goal_ids])


def _derived_goal_baseline_ids(
    goal: AnalysisQuestionGoal | ComparisonQuestionGoal,
) -> list[str]:
    if isinstance(goal, AnalysisQuestionGoal):
        return list(goal.baseline_goal_ids)
    return []


def _canonical_analysis_type(value: Any) -> str:
    normalized = _canonical_token(value)
    aliases = {
        "ANOMALY_CHECK": "ANOMALY",
        "ANOMALY_ANALYSIS": "ANOMALY",
        "ANOMALY_COMPARISON": "ANOMALY",
        "CORRELATION_ANALYSIS": "CORRELATION",
        "CORRELATION_CHECK": "CORRELATION",
        "CORRELATION_COMPARISON": "CORRELATION",
        "RELATIONSHIP": "CORRELATION",
        "RELATION": "CORRELATION",
        "DIFFERENCE_ANALYSIS": "DIFFERENCE",
        "COMPARISON": "DIFFERENCE",
        "TREND_CHECK": "TREND",
        "TREND_ANALYSIS": "TREND",
        "CAUSAL_ANALYSIS": "CAUSAL",
        "ROOT_CAUSE": "CAUSAL",
        "ROOT_CAUSE_ANALYSIS": "CAUSAL",
    }
    return aliases.get(normalized, normalized)


def _verified_analysis_inputs(
    requested_artifact_ids: Sequence[str],
    verified_query_artifacts: Sequence[Any],
    artifact_goal_ids: Mapping[str, Sequence[str]],
) -> list[GroundedVerifiedAnalysisInput]:
    requested = _dedupe_strings(requested_artifact_ids)
    if not requested:
        raise ValueError("analysis requires at least one verified query artifact")
    ledger: dict[str, Any] = {}
    for artifact in verified_query_artifacts:
        artifact_id = str(_object_value(artifact, "artifact_id", "artifactId") or "").strip()
        if artifact_id:
            ledger[artifact_id] = artifact
    missing = [artifact_id for artifact_id in requested if artifact_id not in ledger]
    if missing:
        raise ValueError("analysis input artifacts are not in the verified ledger: %s" % ",".join(missing))
    results: list[GroundedVerifiedAnalysisInput] = []
    for artifact_id in requested:
        artifact = ledger[artifact_id]
        verified = _object_value(artifact, "verified_evidence", "verifiedEvidence")
        if not bool(_object_value(verified, "passed")):
            raise ValueError("analysis input artifact is not verified: %s" % artifact_id)
        run_result = _object_value(artifact, "run_result", "runResult")
        bundle = _object_value(
            run_result,
            "merged_query_bundle",
            "mergedQueryBundle",
        )
        raw_rows = _object_value(bundle, "rows") or []
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise ValueError("verified query artifact rows must be a list")
        declared_columns = _dedupe_strings(_object_value(artifact, "output_columns", "outputColumns") or [])
        declared_columns = [column for column in declared_columns if not column.startswith("__")]
        if not declared_columns:
            declared_columns = _dedupe_strings(
                key for row in raw_rows if isinstance(row, Mapping) for key in row if not str(key).startswith("__")
            )
        rows = [
            {column: row[column] for column in declared_columns if column in row}
            for row in raw_rows
            if isinstance(row, Mapping)
        ]
        lineage_raw = (
            _object_value(
                artifact,
                "output_lineage",
                "outputLineage",
            )
            or {}
        )
        lineage = {
            str(column): _dedupe_strings(
                refs if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else [refs]
            )
            for column, refs in dict(lineage_raw).items()
            if str(column) in declared_columns
        }
        rows_hash = _stable_hash(rows)
        results.append(
            GroundedVerifiedAnalysisInput(
                artifact_id=artifact_id,
                goal_ids=_dedupe_strings(artifact_goal_ids.get(artifact_id) or []),
                row_ref="query:%s:rows:%s" % (artifact_id, rows_hash[:16]),
                rows_hash=rows_hash,
                rows=rows,
                output_columns=declared_columns,
                output_lineage=lineage,
            )
        )
    return results


def _require_goal_inputs(
    goal: AnalysisQuestionGoal | ComparisonQuestionGoal,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> None:
    available_goal_ids = {canonical_goal_id(goal_id) for item in inputs for goal_id in item.goal_ids}
    required = {
        canonical_goal_id(goal_id)
        for goal_id in [
            *_derived_goal_input_ids(goal),
            *_derived_goal_baseline_ids(goal),
        ]
    }
    missing = sorted(required - available_goal_ids)
    if missing:
        raise ValueError("verified analysis artifacts do not cover declared input goals: %s" % ",".join(missing))


def _series_map(
    request: GroundedRunSkillAnalysisPublicationRequest,
    inputs: Sequence[GroundedVerifiedAnalysisInput],
) -> dict[str, _SeriesData]:
    by_artifact = {item.artifact_id: item for item in inputs}
    result: dict[str, _SeriesData] = {}
    for binding in request.series_bindings:
        source = by_artifact[binding.artifact_id]
        if binding.value_column not in source.output_columns:
            raise _AnalysisEvidenceInsufficient(
                "ANALYSIS_VALUE_COLUMN_UNVERIFIED",
                "series value column is absent from the verified artifact output",
                seriesId=binding.series_id,
                artifactId=binding.artifact_id,
                valueColumn=binding.value_column,
            )
        value_lineage = tuple(source.output_lineage.get(binding.value_column) or [])
        if not value_lineage:
            raise _AnalysisEvidenceInsufficient(
                "ANALYSIS_VALUE_LINEAGE_REQUIRED",
                "series value column has no verified output lineage",
                seriesId=binding.series_id,
                valueColumn=binding.value_column,
            )
        observation_lineage: dict[str, tuple[str, ...]] = {}
        for key in request.observation_keys:
            if key not in source.output_columns:
                raise _AnalysisEvidenceInsufficient(
                    "OBSERVATION_GRAIN_COLUMN_UNVERIFIED",
                    "observation key is absent from the verified artifact output",
                    seriesId=binding.series_id,
                    observationKey=key,
                )
            refs = tuple(source.output_lineage.get(key) or [])
            if not refs:
                raise _AnalysisEvidenceInsufficient(
                    "OBSERVATION_GRAIN_LINEAGE_REQUIRED",
                    "observation key has no verified semantic lineage",
                    seriesId=binding.series_id,
                    observationKey=key,
                )
            observation_lineage[key] = refs

        values: list[Decimal] = []
        keyed_values: dict[tuple[Any, ...], Decimal] = {}
        for row in source.rows:
            value = _decimal(row.get(binding.value_column))
            if value is None:
                continue
            values.append(value)
            if request.observation_keys:
                if any(row.get(key) is None for key in request.observation_keys):
                    continue
                observation = tuple(row.get(key) for key in request.observation_keys)
                if observation in keyed_values:
                    raise _AnalysisEvidenceInsufficient(
                        "OBSERVATION_GRAIN_DUPLICATE",
                        "series contains more than one row at the declared observation grain",
                        seriesId=binding.series_id,
                        observation=list(observation),
                    )
                keyed_values[observation] = value
        if not values:
            raise _AnalysisEvidenceInsufficient(
                "ANALYSIS_NUMERIC_VALUES_REQUIRED",
                "series has no verified numeric values",
                seriesId=binding.series_id,
            )
        result[binding.series_id] = _SeriesData(
            binding=binding,
            source=source,
            values=tuple(values),
            keyed_values=keyed_values,
            value_lineage=value_lineage,
            observation_lineage=observation_lineage,
        )
    return result


def _left_right_series(
    request: GroundedRunSkillAnalysisPublicationRequest,
    series: Mapping[str, _SeriesData],
) -> tuple[_SeriesData, _SeriesData]:
    if not request.left_series_id or not request.right_series_id:
        raise _AnalysisEvidenceInsufficient(
            "ANALYSIS_OPERAND_SERIES_REQUIRED",
            "analysis requires explicit left and right verified series",
        )
    return series[request.left_series_id], series[request.right_series_id]


def _aligned_values(
    left: _SeriesData,
    right: _SeriesData,
    observation_keys: Sequence[str],
) -> list[tuple[tuple[Any, ...], Decimal, Decimal]]:
    if observation_keys:
        for key in observation_keys:
            left_refs = set(left.observation_lineage.get(key) or [])
            right_refs = set(right.observation_lineage.get(key) or [])
            if not left_refs.intersection(right_refs):
                raise _AnalysisEvidenceInsufficient(
                    "OBSERVATION_GRAIN_LINEAGE_MISMATCH",
                    "series observation keys do not share verified semantic lineage",
                    observationKey=key,
                    leftSeriesId=left.binding.series_id,
                    rightSeriesId=right.binding.series_id,
                )
        shared = sorted(
            set(left.keyed_values).intersection(right.keyed_values),
            key=_observation_sort_key,
        )
        if not shared:
            raise _AnalysisEvidenceInsufficient(
                "ALIGNED_OBSERVATIONS_REQUIRED",
                "series have no observations at the same verified grain",
                leftSeriesId=left.binding.series_id,
                rightSeriesId=right.binding.series_id,
            )
        return [(key, left.keyed_values[key], right.keyed_values[key]) for key in shared]
    if len(left.values) != 1 or len(right.values) != 1:
        raise _AnalysisEvidenceInsufficient(
            "SCALAR_OR_OBSERVATION_GRAIN_REQUIRED",
            "multiple values require explicit observation keys for alignment",
            leftSampleCount=len(left.values),
            rightSampleCount=len(right.values),
        )
    return [((), left.values[0], right.values[0])]


def _require_method(value: Any, allowed: set[str]) -> str:
    method = _canonical_token(value)
    if method not in allowed:
        raise _AnalysisEvidenceInsufficient(
            "ANALYSIS_METHOD_REQUIRED",
            "analysis requires one supported reproducible method",
            supportedMethods=sorted(allowed),
            requestedMethod=method,
        )
    return method


def _mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise _AnalysisEvidenceInsufficient(
            "ANALYSIS_NUMERIC_VALUES_REQUIRED",
            "numeric observations are required",
        )
    return sum(values, Decimal("0")) / Decimal(len(values))


def _ols_slope(values: Sequence[Decimal]) -> Decimal:
    count = len(values)
    mean_x = Decimal(count - 1) / Decimal("2")
    mean_y = _mean(values)
    numerator = sum(
        ((Decimal(index) - mean_x) * (value - mean_y) for index, value in enumerate(values)),
        Decimal("0"),
    )
    denominator = sum(
        ((Decimal(index) - mean_x) ** 2 for index in range(count)),
        Decimal("0"),
    )
    return numerator / denominator if denominator else Decimal("0")


def _observation_payload(
    keys: Sequence[str],
    observation: tuple[Any, ...],
) -> dict[str, Any]:
    return {key: value for key, value in zip(keys, observation)}


def _observation_sort_key(observation: tuple[Any, ...]) -> tuple[Any, ...]:
    result: list[tuple[int, Any]] = []
    for value in observation:
        numeric = _decimal(value)
        if numeric is not None:
            result.append((0, numeric))
        else:
            result.append((1, str(value)))
    return tuple(result)


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    normalized = value.normalize()
    text = format(normalized, "f")
    return "0" if text in {"-0", ""} else text


def _canonical_token(value: Any) -> str:
    return re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(value or "").strip().upper(),
    ).strip("_")


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _object_value(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
