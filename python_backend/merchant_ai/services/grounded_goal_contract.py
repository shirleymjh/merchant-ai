from __future__ import annotations

import hashlib
import json
import unicodedata
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Protocol, Sequence, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from merchant_ai.models import APIModel


class QuestionGoalKind(str, Enum):
    """Closed structural goal kinds; business concepts remain asset-defined."""

    METRIC = "METRIC"
    DIMENSION = "DIMENSION"
    TIME_WINDOW = "TIME_WINDOW"
    COMPARISON = "COMPARISON"
    ENTITY = "ENTITY"
    DEPENDENCY = "DEPENDENCY"
    RULE = "RULE"
    DETAIL = "DETAIL"
    RANKING = "RANKING"
    ANALYSIS = "ANALYSIS"


_GOAL_DECLARATION_HIDDEN_SCHEMA_FIELDS = {
    "semantic_ref_ids",
    "semanticRefIds",
    "metric_ref_id",
    "metricRefId",
    "dimension_ref_id",
    "dimensionRefId",
    "entity_ref_id",
    "entityRefId",
    "rule_ref_ids",
    "ruleRefIds",
    "required_field_ref_ids",
    "requiredFieldRefIds",
    "population_scope",
    "populationScope",
    "population_goal_ids",
    "populationGoalIds",
    "artifact_kind",
    "artifactKind",
}


def _scrub_goal_declaration_schema(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for field_name in _GOAL_DECLARATION_HIDDEN_SCHEMA_FIELDS:
                properties.pop(field_name, None)
        required = value.get("required")
        if isinstance(required, list):
            value["required"] = [
                item
                for item in required
                if item not in _GOAL_DECLARATION_HIDDEN_SCHEMA_FIELDS
            ]
        for child in value.values():
            _scrub_goal_declaration_schema(child)
    elif isinstance(value, list):
        for child in value:
            _scrub_goal_declaration_schema(child)


class _StrictGoalModel(APIModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: Any,
    ) -> dict[str, Any]:
        schema = handler(core_schema)
        _scrub_goal_declaration_schema(schema)
        return schema


PopulationScope: TypeAlias = Literal[
    "ALL_MATCHING_ROWS",
    "SAME_AS_GOAL",
    "VERIFIED_ENTITY_SET",
    "VERIFIED_PREDICATE_SCOPE",
    "VERIFIED_RESULT_ARTIFACT",
]


class QuestionGoalBase(_StrictGoalModel):
    goal_id: str
    kind: str
    label: str
    required: bool = True
    depends_on_goal_ids: list[str] = Field(default_factory=list)
    source_spans: list[str] = Field(default_factory=list)
    semantic_ref_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class MetricQuestionGoal(QuestionGoalBase):
    kind: Literal["METRIC"] = "METRIC"
    metric_ref_id: str = ""
    result_role: str = "value"


class DimensionQuestionGoal(QuestionGoalBase):
    kind: Literal["DIMENSION"] = "DIMENSION"
    dimension_ref_id: str = ""
    usage: str = "group_by"


class TimeWindowQuestionGoal(QuestionGoalBase):
    kind: Literal["TIME_WINDOW"] = "TIME_WINDOW"
    # Every time Goal must retain the exact user-facing expression.  Making
    # this required in the tool schema prevents a later validator from
    # rejecting an input that the model was previously told was optional.
    time_expression: str
    start: str = ""
    end: str = ""
    timezone: str = ""
    granularity: str = ""
    # Optional execution semantics retained when Core has them.  Relative
    # windows normally only carry ``time_expression``; the Kernel still has
    # to prove the concrete values from the query artifact.
    days: int = 0
    calendar_anchor_policy: str = ""
    data_as_of_policy: str = ""
    window_role: str = ""
    time_range_kind: str = ""
    applies_to_goal_ids: list[str] = Field(default_factory=list)


class ComparisonQuestionGoal(QuestionGoalBase):
    kind: Literal["COMPARISON"] = "COMPARISON"
    comparison_type: str = ""
    left_goal_ids: list[str] = Field(default_factory=list)
    right_goal_ids: list[str] = Field(default_factory=list)


class EntityQuestionGoal(QuestionGoalBase):
    kind: Literal["ENTITY"] = "ENTITY"
    entity_ref_id: str = ""
    entity_identity: str = ""
    role: str = "subject"
    source_goal_ids: list[str] = Field(default_factory=list)


class DependencyQuestionGoal(QuestionGoalBase):
    kind: Literal["DEPENDENCY"] = "DEPENDENCY"
    dependency_type: str = "entity_chain"
    upstream_goal_ids: list[str] = Field(default_factory=list)
    downstream_goal_ids: list[str] = Field(default_factory=list)
    artifact_kind: str = ""


class RuleQuestionGoal(QuestionGoalBase):
    kind: Literal["RULE"] = "RULE"
    rule_ref_ids: list[str] = Field(default_factory=list)
    requested_action: str = ""


class DetailQuestionGoal(QuestionGoalBase):
    kind: Literal["DETAIL"] = "DETAIL"
    required_field_ref_ids: list[str] = Field(default_factory=list)
    input_goal_ids: list[str] = Field(default_factory=list)
    requested_field_phrases: list[str] = Field(default_factory=list)
    request_all_fields: bool = False
    # Retained for checkpoint and execution-graph compatibility.  The goal
    # declaration normalizer still strips these fields from new non-RANKING
    # model payloads; trusted historical contracts may need them to prove a
    # cross-query population snapshot.
    population_scope: PopulationScope = "ALL_MATCHING_ROWS"
    population_goal_ids: list[str] = Field(default_factory=list)


class RankingQuestionGoal(QuestionGoalBase):
    kind: Literal["RANKING"] = "RANKING"
    # A ranking over the rows selected by its own current query is the safe,
    # ordinary case.  External/history-backed populations must be opted into
    # explicitly with one of the VERIFIED_* scopes below.
    population_scope: PopulationScope = "ALL_MATCHING_ROWS"
    population_goal_ids: list[str] = Field(default_factory=list)
    metric_goal_ids: list[str] = Field(default_factory=list)
    dimension_goal_ids: list[str] = Field(default_factory=list)
    direction: str = "DESC"
    limit: int = 0
    limit_source: Literal["USER_EXPLICIT", "SYSTEM_DEFAULT"] = "USER_EXPLICIT"


class AnalysisQuestionGoal(QuestionGoalBase):
    kind: Literal["ANALYSIS"] = "ANALYSIS"
    analysis_type: str = ""
    input_goal_ids: list[str] = Field(default_factory=list)
    baseline_goal_ids: list[str] = Field(default_factory=list)


QuestionGoal: TypeAlias = Annotated[
    MetricQuestionGoal
    | DimensionQuestionGoal
    | TimeWindowQuestionGoal
    | ComparisonQuestionGoal
    | EntityQuestionGoal
    | DependencyQuestionGoal
    | RuleQuestionGoal
    | DetailQuestionGoal
    | RankingQuestionGoal
    | AnalysisQuestionGoal,
    Field(discriminator="kind"),
]


class OriginalQuestionGoalContract(_StrictGoalModel):
    """Core-authored, immutable intent ledger for one original question.

    The contract records *what* must be answered. It deliberately does not
    select tables, write SQL, or infer business semantics from question text.
    Those responsibilities remain with grounded semantic resolution.
    """

    contract_version: str = "original_question_goal_contract.v1"
    contract_id: str = ""
    question: str
    goals: list[QuestionGoal] = Field(default_factory=list)
    source: str = "core"

    def goal_map(self) -> dict[str, QuestionGoal]:
        return {goal.goal_id: goal for goal in self.goals}


class OriginalQuestionGoalDeclaration(_StrictGoalModel):
    """Core-authored Goal payload before trusted question binding.

    The original question already belongs to the server-owned run state, so
    the model must not duplicate it inside the tool call.  The Harness turns
    this declaration into an immutable ``OriginalQuestionGoalContract`` by
    binding the exact retained question at the tool boundary.
    """

    contract_version: str = "original_question_goal_contract.v1"
    contract_id: str = ""
    goals: list[QuestionGoal] = Field(default_factory=list)
    source: str = "core"


class GoalContractIssue(APIModel):
    code: str
    message: str
    blocking: bool = True
    goal_id: str = ""
    reference_goal_id: str = ""
    path: str = ""


class GoalContractValidationResult(APIModel):
    valid: bool = False
    contract: OriginalQuestionGoalContract | None = None
    issues: list[GoalContractIssue] = Field(default_factory=list)


class GoalContractQuestionVerifier(Protocol):
    """Optional semantic completeness verifier owned outside the Kernel.

    Implementations may use a grounded model or another language-aware
    service to compare the retained question with Core's typed declaration.
    The Kernel treats verifier errors and malformed verifier output as
    blocking, but contains no language or business vocabulary itself.
    """

    def verify(
        self,
        contract: OriginalQuestionGoalContract,
    ) -> Sequence[GoalContractIssue | Mapping[str, Any]]: ...


class QuestionStructuralHints(APIModel):
    """Non-authoritative Unicode shape audit of the retained question.

    Goal kinds and relationships cannot be inferred reliably from surface
    words. This model therefore exposes only language-agnostic token and
    punctuation facts. It is suitable for observability, never for allowing
    or blocking a Goal Contract.
    """

    token_count: int = 0
    number_tokens: list[str] = Field(default_factory=list)
    punctuation_tokens: list[str] = Field(default_factory=list)
    clause_count: int = 1


class GoalResolutionStatus(str, Enum):
    """Whether evidence proves a goal or explicitly records why it cannot."""

    PROVED = "PROVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GoalProofResolutionBase(_StrictGoalModel):
    goal_id: str
    goal_kind: str
    resolution: Literal["PROVED", "INSUFFICIENT_EVIDENCE"]
    proof_type: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class MetricGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["METRIC"] = "METRIC"
    metric_ref_ids: list[str] = Field(default_factory=list)
    value_refs: list[str] = Field(default_factory=list)


class DimensionGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["DIMENSION"] = "DIMENSION"
    dimension_ref_ids: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)


class TimeWindowGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["TIME_WINDOW"] = "TIME_WINDOW"
    time_expression: str = ""
    start: str = ""
    end: str = ""
    timezone: str = ""
    granularity: str = ""
    days: int = 0
    label: str = ""
    explicit: bool = False
    calendar_anchor_policy: str = ""
    data_as_of_policy: str = ""
    window_role: str = ""
    time_range_kind: str = ""


class ComparisonGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["COMPARISON"] = "COMPARISON"
    comparison_type: str = ""
    operand_goal_ids: list[str] = Field(default_factory=list)
    comparison_method: str = ""
    result_ref: str = ""
    baseline_refs: list[str] = Field(default_factory=list)
    normalization_method: str = ""


class EntityGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["ENTITY"] = "ENTITY"
    entity_ref_ids: list[str] = Field(default_factory=list)
    identity_fields: list[str] = Field(default_factory=list)
    entity_set_ref: str = ""


class DependencyGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["DEPENDENCY"] = "DEPENDENCY"
    upstream_artifact_ids: list[str] = Field(default_factory=list)
    downstream_artifact_ids: list[str] = Field(default_factory=list)
    lineage_refs: list[str] = Field(default_factory=list)


class RuleGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["RULE"] = "RULE"
    rule_ref_ids: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)


class DetailGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["DETAIL"] = "DETAIL"
    output_fields: list[str] = Field(default_factory=list)
    output_semantic_refs: list[str] = Field(default_factory=list)
    row_set_ref: str = ""
    row_count: int | None = None


class RankingGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["RANKING"] = "RANKING"
    order_by_goal_ids: list[str] = Field(default_factory=list)
    dimension_goal_ids: list[str] = Field(default_factory=list)
    ranking_metric_ref_id: str = ""
    ranking_dimension_ref_id: str = ""
    direction: str = ""
    limit: int = 0
    row_set_ref: str = ""
    population_scope: str = ""
    population_goal_ids: list[str] = Field(default_factory=list)
    population_lineage_refs: list[str] = Field(default_factory=list)


class AnalysisGoalProofResolution(GoalProofResolutionBase):
    goal_kind: Literal["ANALYSIS"] = "ANALYSIS"
    analysis_type: str = ""
    input_goal_ids: list[str] = Field(default_factory=list)
    analysis_method: str = ""
    result_ref: str = ""
    baseline_refs: list[str] = Field(default_factory=list)
    normalization_method: str = ""


GoalProofResolution: TypeAlias = Annotated[
    MetricGoalProofResolution
    | DimensionGoalProofResolution
    | TimeWindowGoalProofResolution
    | ComparisonGoalProofResolution
    | EntityGoalProofResolution
    | DependencyGoalProofResolution
    | RuleGoalProofResolution
    | DetailGoalProofResolution
    | RankingGoalProofResolution
    | AnalysisGoalProofResolution,
    Field(discriminator="goal_kind"),
]


class VerifiedArtifactGoalCoverage(_StrictGoalModel):
    """Coverage declaration retained beside one kernel-verified artifact."""

    artifact_id: str
    goal_contract_fingerprint: str
    covered_goal_ids: list[str] = Field(default_factory=list)
    verification_passed: bool = False
    evidence_refs: list[str] = Field(default_factory=list)
    goal_resolutions: list[GoalProofResolution] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_goal_resolutions(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        raw = normalized.get("goal_resolutions", normalized.get("goalResolutions", []))
        if raw is not None:
            normalized["goal_resolutions"] = [_normalize_goal_resolution_payload(item) for item in raw]
        normalized.pop("goalResolutions", None)
        return normalized


class GoalCoverageIssue(APIModel):
    code: str
    message: str
    blocking: bool = True
    goal_id: str = ""
    artifact_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class GoalCoverageResult(APIModel):
    passed: bool = False
    finalization_allowed: bool = False
    goal_contract_fingerprint: str = ""
    required_goal_ids: list[str] = Field(default_factory=list)
    claimed_covered_goal_ids: list[str] = Field(default_factory=list)
    covered_goal_ids: list[str] = Field(default_factory=list)
    resolved_goal_ids: list[str] = Field(default_factory=list)
    insufficient_evidence_goal_ids: list[str] = Field(default_factory=list)
    missing_required_goal_ids: list[str] = Field(default_factory=list)
    unproved_required_goal_ids: list[str] = Field(default_factory=list)
    optional_uncovered_goal_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    coverage_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    resolution_by_goal_id: dict[str, str] = Field(default_factory=dict)
    resolution_proof_types_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    resolution_artifact_ids_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    resolution_evidence_refs_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
    insufficiency_reason_by_goal_id: dict[str, str] = Field(default_factory=dict)
    issues: list[GoalCoverageIssue] = Field(default_factory=list)


class GoalContractValidationError(ValueError):
    def __init__(self, issues: Sequence[GoalContractIssue]):
        self.issues = tuple(issues)
        super().__init__("invalid original-question goal contract: " + "; ".join(issue.code for issue in issues))


class GoalCoverageBlocked(RuntimeError):
    def __init__(self, result: GoalCoverageResult):
        self.result = result
        super().__init__(
            "original-question goal coverage incomplete: "
            + ", ".join(result.missing_required_goal_ids or [issue.code for issue in result.issues if issue.blocking])
        )


_GOAL_KIND_ALIASES = {
    "METRIC": "METRIC",
    "METRICS": "METRIC",
    "DIMENSION": "DIMENSION",
    "DIMENSIONS": "DIMENSION",
    "TIME": "TIME_WINDOW",
    "TIMEWINDOW": "TIME_WINDOW",
    "TIME_WINDOW": "TIME_WINDOW",
    "TIME_WINDOWS": "TIME_WINDOW",
    "COMPARISON": "COMPARISON",
    "COMPARISONS": "COMPARISON",
    "ENTITY": "ENTITY",
    "ENTITIES": "ENTITY",
    "ENTITY_GOAL": "ENTITY",
    "DEPENDENCY": "DEPENDENCY",
    "DEPENDENCIES": "DEPENDENCY",
    "ENTITY_DEPENDENCY": "DEPENDENCY",
    "RULE": "RULE",
    "RULES": "RULE",
    "POLICY": "RULE",
    "DETAIL": "DETAIL",
    "DETAILS": "DETAIL",
    "DETAIL_ROWS": "DETAIL",
    "RANKING": "RANKING",
    "RANKINGS": "RANKING",
    "RANKED": "RANKING",
    "TOP_N": "RANKING",
    "ANALYSIS": "ANALYSIS",
    "ANALYSES": "ANALYSIS",
    "ANOMALY": "ANALYSIS",
}

_GROUPED_GOAL_FIELDS = {
    "metrics": "METRIC",
    "dimensions": "DIMENSION",
    "timeWindows": "TIME_WINDOW",
    "time_windows": "TIME_WINDOW",
    "comparisons": "COMPARISON",
    "entities": "ENTITY",
    "entityGoals": "ENTITY",
    "entity_goals": "ENTITY",
    "dependencies": "DEPENDENCY",
    "dependencyGoals": "DEPENDENCY",
    "dependency_goals": "DEPENDENCY",
    "rules": "RULE",
    "ruleGoals": "RULE",
    "rule_goals": "RULE",
    "details": "DETAIL",
    "detailGoals": "DETAIL",
    "detail_goals": "DETAIL",
    "rankings": "RANKING",
    "rankingGoals": "RANKING",
    "ranking_goals": "RANKING",
    "analyses": "ANALYSIS",
    "analysisGoals": "ANALYSIS",
    "analysis_goals": "ANALYSIS",
}

def canonical_goal_id(value: Any) -> str:
    """Normalize a Core/artifact goal ID into one stable ledger key."""

    raw = str(value or "").strip().lower()
    characters: list[str] = []
    previous_was_space = False
    for character in raw:
        if character.isspace():
            if not previous_was_space:
                characters.append("_")
            previous_was_space = True
            continue
        characters.append(character)
        previous_was_space = False
    normalized = "".join(characters)
    allowed_tail = {".", "_", ":", "-"}
    if (
        not 1 <= len(normalized) <= 128
        or not _is_ascii_letter_or_digit(normalized[0])
        or any(
            not _is_ascii_letter_or_digit(character) and character not in allowed_tail
            for character in normalized[1:]
        )
    ):
        raise ValueError("goal IDs must be 1-128 ASCII letters/digits with optional '.', '_', ':', or '-' separators")
    return normalized


def inspect_question_structure(question: str) -> QuestionStructuralHints:
    """Inspect language-agnostic Unicode shape without inferring intent."""

    tokens = _unicode_question_tokens(str(question or "").strip())
    return QuestionStructuralHints(
        token_count=len(tokens),
        number_tokens=[value for kind, value in tokens if kind == "NUMBER"],
        punctuation_tokens=[value for kind, value in tokens if kind == "PUNCTUATION"],
        clause_count=_unicode_clause_count(tokens),
    )


def validate_original_question_goal_contract(
    payload: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    structural_checks: bool = True,
    question_verifier: GoalContractQuestionVerifier | None = None,
) -> GoalContractValidationResult:
    """Parse and cross-validate one Core-supplied typed contract.

    ``structural_checks`` is retained for caller compatibility. It controls an
    optional question verifier, not an embedded keyword heuristic.
    """

    try:
        normalized = _normalize_contract_payload(payload)
        contract = OriginalQuestionGoalContract.model_validate(normalized)
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        return GoalContractValidationResult(
            valid=False,
            issues=[
                GoalContractIssue(
                    code="GOAL_CONTRACT_SCHEMA_INVALID",
                    message=str(exc),
                    path=_validation_error_path(exc),
                )
            ],
        )

    issues = _contract_issues(contract)
    if structural_checks and question_verifier is not None:
        issues.extend(_question_verifier_issues(contract, question_verifier))
    return GoalContractValidationResult(
        valid=not any(issue.blocking for issue in issues),
        contract=contract,
        issues=issues,
    )


def parse_original_question_goal_contract(
    payload: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    structural_checks: bool = True,
    question_verifier: GoalContractQuestionVerifier | None = None,
) -> OriginalQuestionGoalContract:
    result = validate_original_question_goal_contract(
        payload,
        structural_checks=structural_checks,
        question_verifier=question_verifier,
    )
    if not result.valid or result.contract is None:
        raise GoalContractValidationError(result.issues)
    return result.contract


def original_question_goal_contract_fingerprint(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
) -> str:
    parsed = parse_original_question_goal_contract(contract)
    canonical = json.dumps(
        parsed.model_dump(by_alias=False, exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def required_goal_ids(contract: OriginalQuestionGoalContract) -> list[str]:
    """Return explicit required goals plus the dependency closure they need."""

    graph = _goal_dependency_graph(contract)
    required = {goal.goal_id for goal in contract.goals if goal.required}
    pending = list(required)
    while pending:
        goal_id = pending.pop()
        for dependency_id in graph.get(goal_id, set()):
            if dependency_id not in required:
                required.add(dependency_id)
                pending.append(dependency_id)
    return [goal.goal_id for goal in contract.goals if goal.goal_id in required]


def declare_verified_artifact_goal_coverage(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    artifact: Any,
    covered_goal_ids: Sequence[str],
    *,
    evidence_refs: Sequence[str] = (),
    goal_resolutions: Sequence[GoalProofResolution | Mapping[str, Any]] = (),
) -> VerifiedArtifactGoalCoverage:
    """Build a typed sidecar only from an already-verified artifact.

    Primitive goal claims retain a compatibility proof derived from the
    kernel-verified result.  Relational/interpretive goals (comparison,
    dependency, rule, detail, ranking and analysis) always require an explicit
    typed resolution and can no longer be completed by ``covered_goal_ids``.
    """

    parsed = parse_original_question_goal_contract(contract)
    artifact_id = str(_object_value(artifact, "artifact_id", "artifactId") or "").strip()
    verified_evidence = _object_value(artifact, "verified_evidence", "verifiedEvidence")
    verification_passed = bool(_object_value(verified_evidence, "passed"))
    if not artifact_id:
        raise ValueError("verified query artifact is missing artifact_id")
    if not verification_passed:
        raise ValueError("goal coverage may only be declared by a verified query artifact")
    canonical_covered = _canonical_goal_id_list(covered_goal_ids)
    normalized_resolutions = [_normalize_goal_resolution_payload(item) for item in goal_resolutions]
    resolution_goal_ids = {canonical_goal_id(item.get("goal_id")) for item in normalized_resolutions}
    goal_map = parsed.goal_map()
    for goal_id in canonical_covered:
        goal = goal_map.get(goal_id)
        if goal is None or goal_id in resolution_goal_ids:
            continue
        legacy_resolution = _legacy_primitive_goal_resolution(
            goal,
            artifact_id=artifact_id,
            evidence_refs=evidence_refs,
            artifact=artifact,
        )
        if legacy_resolution is not None:
            normalized_resolutions.append(legacy_resolution)
    return VerifiedArtifactGoalCoverage(
        artifact_id=artifact_id,
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(parsed),
        covered_goal_ids=canonical_covered,
        verification_passed=True,
        evidence_refs=_normalized_string_list(evidence_refs),
        goal_resolutions=normalized_resolutions,
    )


class GoalCoverageVerifier:
    """Fail-closed finalization gate over the immutable artifact ledger."""

    def verify(
        self,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        artifacts: Sequence[VerifiedArtifactGoalCoverage | Mapping[str, Any] | Any],
    ) -> GoalCoverageResult:
        validation = validate_original_question_goal_contract(contract)
        if not validation.valid or validation.contract is None:
            issues = [
                GoalCoverageIssue(
                    code=issue.code,
                    message=issue.message,
                    blocking=issue.blocking,
                    goal_id=issue.goal_id,
                )
                for issue in validation.issues
            ]
            return GoalCoverageResult(
                required_goal_ids=(required_goal_ids(validation.contract) if validation.contract else []),
                missing_required_goal_ids=(required_goal_ids(validation.contract) if validation.contract else []),
                issues=issues,
            )

        parsed = validation.contract
        fingerprint = original_question_goal_contract_fingerprint(parsed)
        goal_map = parsed.goal_map()
        issues: list[GoalCoverageIssue] = [
            GoalCoverageIssue(code=issue.code, message=issue.message, blocking=issue.blocking, goal_id=issue.goal_id)
            for issue in validation.issues
        ]
        declarations: list[VerifiedArtifactGoalCoverage] = []
        for index, artifact in enumerate(artifacts):
            try:
                declarations.append(_coerce_artifact_coverage(artifact))
            except (TypeError, ValueError, ValidationError) as exc:
                issues.append(
                    GoalCoverageIssue(
                        code="ARTIFACT_GOAL_COVERAGE_INVALID",
                        message=str(exc),
                        details={"artifactIndex": index},
                    )
                )

        seen_artifact_ids: set[str] = set()
        accepted_artifact_ids: list[str] = []
        accepted_declarations: list[VerifiedArtifactGoalCoverage] = []
        for declaration in declarations:
            artifact_id = str(declaration.artifact_id or "").strip()
            if not artifact_id:
                issues.append(
                    GoalCoverageIssue(
                        code="ARTIFACT_ID_MISSING",
                        message="an artifact goal-coverage declaration is missing artifactId",
                    )
                )
                continue
            if artifact_id in seen_artifact_ids:
                issues.append(
                    GoalCoverageIssue(
                        code="DUPLICATE_ARTIFACT_GOAL_COVERAGE",
                        message=f"artifact {artifact_id!r} declared goal coverage more than once",
                        artifact_id=artifact_id,
                    )
                )
                continue
            seen_artifact_ids.add(artifact_id)

            if not declaration.verification_passed:
                issues.append(
                    GoalCoverageIssue(
                        code="ARTIFACT_NOT_VERIFIED",
                        message=f"artifact {artifact_id!r} cannot contribute goal coverage before evidence verification passes",
                        artifact_id=artifact_id,
                    )
                )
                continue
            if declaration.goal_contract_fingerprint != fingerprint:
                issues.append(
                    GoalCoverageIssue(
                        code="GOAL_CONTRACT_FINGERPRINT_MISMATCH",
                        message=f"artifact {artifact_id!r} declared coverage for a different goal contract",
                        artifact_id=artifact_id,
                        details={
                            "expectedFingerprint": fingerprint,
                            "actualFingerprint": declaration.goal_contract_fingerprint,
                        },
                    )
                )
                continue

            accepted_artifact_ids.append(artifact_id)
            accepted_declarations.append(declaration)

        accepted_artifact_id_set = set(accepted_artifact_ids)
        claimed: set[str] = set()
        resolution_candidates: dict[str, list[tuple[str, GoalProofResolution]]] = {
            goal.goal_id: [] for goal in parsed.goals
        }
        for declaration in accepted_declarations:
            artifact_id = declaration.artifact_id
            if not declaration.covered_goal_ids and not declaration.goal_resolutions:
                issues.append(
                    GoalCoverageIssue(
                        code="ARTIFACT_COVERS_NO_GOALS",
                        message=f"artifact {artifact_id!r} is verified but declares no original-question goal coverage",
                        blocking=False,
                        artifact_id=artifact_id,
                    )
                )

            resolution_by_id: dict[str, GoalProofResolution] = {}
            for resolution in declaration.goal_resolutions:
                goal_id = resolution.goal_id
                if goal_id in resolution_by_id:
                    issues.append(
                        GoalCoverageIssue(
                            code="DUPLICATE_ARTIFACT_GOAL_RESOLUTION",
                            message=(
                                f"artifact {artifact_id!r} contains more than one resolution for goal {goal_id!r}"
                            ),
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                        )
                    )
                    continue
                resolution_by_id[goal_id] = resolution

            for goal_id in declaration.covered_goal_ids:
                if goal_id not in goal_map:
                    issues.append(
                        GoalCoverageIssue(
                            code="UNKNOWN_COVERED_GOAL_ID",
                            message=f"artifact {artifact_id!r} declared unknown goal ID {goal_id!r}",
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                        )
                    )
                    continue
                claimed.add(goal_id)

            for goal_id, resolution in resolution_by_id.items():
                if goal_id not in goal_map:
                    issues.append(
                        GoalCoverageIssue(
                            code="UNKNOWN_GOAL_RESOLUTION_ID",
                            message=f"artifact {artifact_id!r} resolved unknown goal ID {goal_id!r}",
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                        )
                    )
                    continue

                goal = goal_map[goal_id]
                if resolution.goal_kind != goal.kind:
                    issues.append(
                        GoalCoverageIssue(
                            code="GOAL_RESOLUTION_KIND_MISMATCH",
                            message=(
                                f"artifact {artifact_id!r} resolved {goal_id!r} as "
                                f"{resolution.goal_kind}, but the contract declares {goal.kind}"
                            ),
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                        )
                    )
                    continue

                if resolution.resolution == GoalResolutionStatus.PROVED.value:
                    claimed.add(goal_id)
                elif goal_id in declaration.covered_goal_ids:
                    issues.append(
                        GoalCoverageIssue(
                            code="INSUFFICIENT_EVIDENCE_CANNOT_PROVE_GOAL",
                            message=(
                                f"artifact {artifact_id!r} marked goal {goal_id!r} as covered "
                                "but explicitly resolved it as INSUFFICIENT_EVIDENCE"
                            ),
                            blocking=False,
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                        )
                    )

                resolution_issues = _goal_resolution_issues(
                    goal,
                    resolution,
                    artifact_id=artifact_id,
                    accepted_artifact_ids=accepted_artifact_id_set,
                    goal_map=goal_map,
                )
                issues.extend(resolution_issues)
                if any(issue.blocking for issue in resolution_issues):
                    continue

                if resolution.resolution == GoalResolutionStatus.PROVED.value:
                    semantic_issue = _semantic_evidence_issue(
                        goal,
                        artifact_id=artifact_id,
                        artifact_evidence_refs=declaration.evidence_refs,
                        resolution_evidence_refs=resolution.evidence_refs,
                    )
                    if semantic_issue is not None:
                        issues.append(semantic_issue)
                        continue
                resolution_candidates[goal_id].append((artifact_id, resolution))

            for goal_id in declaration.covered_goal_ids:
                if goal_id not in goal_map or goal_id in resolution_by_id:
                    continue
                goal = goal_map[goal_id]
                legacy_resolution = _legacy_primitive_goal_resolution(
                    goal,
                    artifact_id=artifact_id,
                    evidence_refs=declaration.evidence_refs,
                    artifact=next(
                        (
                            item
                            for item in artifacts
                            if str(_object_value(item, "artifact_id", "artifactId") or "").strip()
                            == artifact_id
                        ),
                        None,
                    ),
                )
                if legacy_resolution is None:
                    issues.append(
                        GoalCoverageIssue(
                            code="GOAL_TYPED_PROOF_REQUIRED",
                            message=(
                                f"artifact {artifact_id!r} claimed {goal.kind} goal {goal_id!r} "
                                "without a typed proof/resolution"
                            ),
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                            details={"goalKind": goal.kind},
                        )
                    )
                    continue
                semantic_issue = _semantic_evidence_issue(
                    goal,
                    artifact_id=artifact_id,
                    artifact_evidence_refs=declaration.evidence_refs,
                    resolution_evidence_refs=legacy_resolution.get("evidence_refs", []),
                )
                if semantic_issue is not None:
                    issues.append(semantic_issue)
                    continue
                resolution_candidates[goal_id].append(
                    (
                        artifact_id,
                        _parse_goal_resolution(legacy_resolution),
                    )
                )

        raw_resolution_by_goal_id: dict[str, str] = {}
        coverage_by_goal_id: dict[str, list[str]] = {}
        resolution_proof_types_by_goal_id: dict[str, list[str]] = {}
        resolution_artifact_ids_by_goal_id: dict[str, list[str]] = {}
        resolution_evidence_refs_by_goal_id: dict[str, list[str]] = {}
        insufficiency_reason_by_goal_id: dict[str, str] = {}
        for goal in parsed.goals:
            candidates = resolution_candidates[goal.goal_id]
            proved = [
                (artifact_id, resolution)
                for artifact_id, resolution in candidates
                if resolution.resolution == GoalResolutionStatus.PROVED.value
            ]
            insufficient = [
                (artifact_id, resolution)
                for artifact_id, resolution in candidates
                if resolution.resolution == GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value
            ]
            if proved:
                raw_resolution_by_goal_id[goal.goal_id] = GoalResolutionStatus.PROVED.value
                coverage_by_goal_id[goal.goal_id] = list(dict.fromkeys(artifact_id for artifact_id, _ in proved))
                resolution_artifact_ids_by_goal_id[goal.goal_id] = list(coverage_by_goal_id[goal.goal_id])
                resolution_evidence_refs_by_goal_id[goal.goal_id] = list(
                    dict.fromkeys(evidence_ref for _, resolution in proved for evidence_ref in resolution.evidence_refs)
                )
                resolution_proof_types_by_goal_id[goal.goal_id] = list(
                    dict.fromkeys(resolution.proof_type for _, resolution in proved if resolution.proof_type)
                )
                if insufficient:
                    issues.append(
                        GoalCoverageIssue(
                            code="CONFLICTING_GOAL_RESOLUTIONS",
                            message=(
                                f"goal {goal.goal_id!r} has both PROVED and "
                                "INSUFFICIENT_EVIDENCE resolutions; verified proof takes precedence"
                            ),
                            blocking=False,
                            goal_id=goal.goal_id,
                        )
                    )
            elif insufficient:
                raw_resolution_by_goal_id[goal.goal_id] = GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value
                resolution_artifact_ids_by_goal_id[goal.goal_id] = list(
                    dict.fromkeys(artifact_id for artifact_id, _ in insufficient)
                )
                resolution_evidence_refs_by_goal_id[goal.goal_id] = list(
                    dict.fromkeys(
                        evidence_ref for _, resolution in insufficient for evidence_ref in resolution.evidence_refs
                    )
                )
                insufficiency_reason_by_goal_id[goal.goal_id] = next(
                    (resolution.reason for _, resolution in insufficient if resolution.reason),
                    "",
                )
                resolution_proof_types_by_goal_id[goal.goal_id] = list(
                    dict.fromkeys(resolution.proof_type for _, resolution in insufficient if resolution.proof_type)
                )

        graph = _goal_dependency_graph(parsed)
        effective_covered = {
            goal_id
            for goal_id, resolution in raw_resolution_by_goal_id.items()
            if resolution == GoalResolutionStatus.PROVED.value
        }
        while True:
            invalid = {
                goal_id for goal_id in effective_covered if not graph.get(goal_id, set()).issubset(effective_covered)
            }
            if not invalid:
                break
            effective_covered.difference_update(invalid)

        dependency_issue_goal_ids: set[str] = set()
        for goal_id in [
            goal.goal_id
            for goal in parsed.goals
            if raw_resolution_by_goal_id.get(goal.goal_id) == GoalResolutionStatus.PROVED.value
            and goal.goal_id not in effective_covered
        ]:
            missing_dependencies = sorted(graph.get(goal_id, set()) - effective_covered)
            issues.append(
                GoalCoverageIssue(
                    code="COVERED_GOAL_DEPENDENCY_UNCOVERED",
                    message=f"goal {goal_id!r} was claimed covered while prerequisite goals remain uncovered",
                    goal_id=goal_id,
                    details={"missingDependencyGoalIds": missing_dependencies},
                )
            )
            dependency_issue_goal_ids.add(goal_id)

        for goal_id in [
            goal.goal_id
            for goal in parsed.goals
            if goal.goal_id in claimed
            and goal.goal_id not in effective_covered
            and goal.goal_id not in dependency_issue_goal_ids
            and graph.get(goal.goal_id, set())
            and not graph.get(goal.goal_id, set()).issubset(effective_covered)
        ]:
            issues.append(
                GoalCoverageIssue(
                    code="COVERED_GOAL_DEPENDENCY_UNCOVERED",
                    message=f"goal {goal_id!r} was claimed covered while prerequisite goals remain uncovered",
                    goal_id=goal_id,
                    details={"missingDependencyGoalIds": sorted(graph.get(goal_id, set()) - effective_covered)},
                )
            )

        effective_resolved = set(raw_resolution_by_goal_id)
        while True:
            unresolved = {
                goal_id for goal_id in effective_resolved if not graph.get(goal_id, set()).issubset(effective_resolved)
            }
            if not unresolved:
                break
            effective_resolved.difference_update(unresolved)

        for goal_id in [
            goal.goal_id
            for goal in parsed.goals
            if goal.goal_id in raw_resolution_by_goal_id
            and goal.goal_id not in effective_resolved
            and goal.goal_id not in effective_covered
        ]:
            issues.append(
                GoalCoverageIssue(
                    code="RESOLVED_GOAL_DEPENDENCY_UNRESOLVED",
                    message=(f"goal {goal_id!r} has a typed resolution while prerequisite goals remain unresolved"),
                    goal_id=goal_id,
                    details={"missingDependencyGoalIds": sorted(graph.get(goal_id, set()) - effective_resolved)},
                )
            )

        required = required_goal_ids(parsed)
        missing = [goal_id for goal_id in required if goal_id not in effective_resolved]
        for goal_id in missing:
            goal = goal_map[goal_id]
            issues.append(
                GoalCoverageIssue(
                    code="REQUIRED_GOAL_UNCOVERED",
                    message=f"required original-question goal {goal.label!r} ({goal_id}) has no effective verified coverage",
                    goal_id=goal_id,
                )
            )

        insufficient_evidence = [
            goal.goal_id
            for goal in parsed.goals
            if goal.goal_id in effective_resolved
            and raw_resolution_by_goal_id.get(goal.goal_id) == GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value
        ]
        for goal_id in insufficient_evidence:
            issues.append(
                GoalCoverageIssue(
                    code="GOAL_RESOLVED_INSUFFICIENT_EVIDENCE",
                    message=(
                        f"goal {goal_id!r} was explicitly resolved as "
                        "INSUFFICIENT_EVIDENCE and is not counted as proved coverage"
                    ),
                    blocking=False,
                    goal_id=goal_id,
                )
            )

        unproved_required = [goal_id for goal_id in required if goal_id not in effective_covered]

        optional_uncovered = [
            goal.goal_id
            for goal in parsed.goals
            if goal.goal_id not in required and goal.goal_id not in effective_resolved
        ]
        blocking = any(issue.blocking for issue in issues)
        finalization_allowed = not blocking and not missing
        return GoalCoverageResult(
            passed=finalization_allowed and not unproved_required,
            finalization_allowed=finalization_allowed,
            goal_contract_fingerprint=fingerprint,
            required_goal_ids=required,
            claimed_covered_goal_ids=[goal.goal_id for goal in parsed.goals if goal.goal_id in claimed],
            covered_goal_ids=[goal.goal_id for goal in parsed.goals if goal.goal_id in effective_covered],
            resolved_goal_ids=[goal.goal_id for goal in parsed.goals if goal.goal_id in effective_resolved],
            insufficient_evidence_goal_ids=insufficient_evidence,
            missing_required_goal_ids=missing,
            unproved_required_goal_ids=unproved_required,
            optional_uncovered_goal_ids=optional_uncovered,
            artifact_ids=accepted_artifact_ids,
            coverage_by_goal_id={
                goal_id: artifact_ids
                for goal_id, artifact_ids in coverage_by_goal_id.items()
                if goal_id in effective_covered and artifact_ids
            },
            resolution_by_goal_id={
                goal_id: raw_resolution_by_goal_id[goal_id]
                for goal_id in [goal.goal_id for goal in parsed.goals]
                if goal_id in effective_resolved
            },
            resolution_proof_types_by_goal_id={
                goal_id: resolution_proof_types_by_goal_id.get(goal_id, [])
                for goal_id in required
                if goal_id in effective_resolved
            },
            resolution_artifact_ids_by_goal_id={
                goal_id: resolution_artifact_ids_by_goal_id.get(goal_id, [])
                for goal_id in [goal.goal_id for goal in parsed.goals]
                if goal_id in effective_resolved
            },
            resolution_evidence_refs_by_goal_id={
                goal_id: resolution_evidence_refs_by_goal_id.get(goal_id, [])
                for goal_id in [goal.goal_id for goal in parsed.goals]
                if goal_id in effective_resolved
            },
            insufficiency_reason_by_goal_id={
                goal_id: insufficiency_reason_by_goal_id[goal_id]
                for goal_id in [goal.goal_id for goal in parsed.goals]
                if goal_id in effective_resolved and goal_id in insufficiency_reason_by_goal_id
            },
            issues=issues,
        )

    def require_complete(
        self,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        artifacts: Sequence[VerifiedArtifactGoalCoverage | Mapping[str, Any] | Any],
    ) -> GoalCoverageResult:
        result = self.verify(contract, artifacts)
        if not result.finalization_allowed:
            raise GoalCoverageBlocked(result)
        return result


def _normalize_contract_payload(
    payload: OriginalQuestionGoalContract | Mapping[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(payload, OriginalQuestionGoalContract):
        raw: Any = payload.model_dump(by_alias=False)
    elif isinstance(payload, str):
        raw = json.loads(payload)
    elif isinstance(payload, Mapping):
        raw = dict(payload)
    else:
        raise TypeError("goal contract must be a mapping, JSON object string, or OriginalQuestionGoalContract")
    if not isinstance(raw, Mapping):
        raise TypeError("goal contract JSON must contain an object")

    normalized = dict(raw)
    unified_goals = normalized.get("goals", [])
    if unified_goals is None:
        unified_goals = []
    if not isinstance(unified_goals, list):
        raise TypeError("goal contract goals must be a list")
    goals: list[Any] = list(unified_goals)
    for field_name, kind in _GROUPED_GOAL_FIELDS.items():
        grouped = normalized.pop(field_name, None)
        if grouped is None:
            continue
        if not isinstance(grouped, list):
            raise TypeError(f"goal contract {field_name} must be a list")
        for item in grouped:
            if not isinstance(item, Mapping):
                raise TypeError(f"goal contract {field_name} entries must be objects")
            typed_item = dict(item)
            typed_item.setdefault("kind", kind)
            goals.append(typed_item)

    normalized["goals"] = [_normalize_goal_payload(goal) for goal in goals]
    if "contractVersion" in normalized and "contract_version" not in normalized:
        normalized["contract_version"] = normalized.pop("contractVersion")
    if "contractId" in normalized and "contract_id" not in normalized:
        normalized["contract_id"] = normalized.pop("contractId")
    normalized["question"] = str(normalized.get("question") or "").strip()
    if "contract_id" in normalized:
        normalized["contract_id"] = str(normalized.get("contract_id") or "").strip()
    if "source" in normalized:
        normalized["source"] = str(normalized.get("source") or "").strip()
    return normalized


def _normalize_goal_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, QuestionGoalBase):
        normalized = payload.model_dump(by_alias=False)
    elif isinstance(payload, Mapping):
        normalized = dict(payload)
    else:
        raise TypeError("each goal must be an object")

    if "goalType" in normalized and "kind" not in normalized:
        normalized["kind"] = normalized.pop("goalType")
    if "goalId" in normalized and "goal_id" not in normalized:
        normalized["goal_id"] = normalized.pop("goalId")
    if "id" in normalized and "goal_id" not in normalized:
        normalized["goal_id"] = normalized.pop("id")
    normalized["kind"] = _canonical_goal_kind(normalized.get("kind"))
    normalized["goal_id"] = canonical_goal_id(normalized.get("goal_id"))

    id_list_fields = (
        "depends_on_goal_ids",
        "applies_to_goal_ids",
        "left_goal_ids",
        "right_goal_ids",
        "source_goal_ids",
        "upstream_goal_ids",
        "downstream_goal_ids",
        "input_goal_ids",
        "baseline_goal_ids",
        "metric_goal_ids",
        "dimension_goal_ids",
        "population_goal_ids",
    )
    string_list_fields = (
        "source_spans",
        "semantic_ref_ids",
        "rule_ref_ids",
        "required_field_ref_ids",
        "requested_field_phrases",
    )
    for snake_name in id_list_fields + string_list_fields:
        camel_name = _camel_name(snake_name)
        if camel_name in normalized and snake_name not in normalized:
            normalized[snake_name] = normalized.pop(camel_name)
    for field_name in id_list_fields:
        if field_name in normalized:
            normalized[field_name] = _canonical_goal_id_list(normalized[field_name])
    for field_name in string_list_fields:
        if field_name in normalized:
            normalized[field_name] = _normalized_string_list(normalized[field_name])

    for field_name, value in list(normalized.items()):
        if isinstance(value, str):
            normalized[field_name] = value.strip()
    if normalized.get("kind") == "RANKING":
        try:
            limit = int(normalized.get("limit") or 0)
        except (TypeError, ValueError):
            limit = 0
        normalized["limit"] = limit
        normalized.setdefault("limit_source", "USER_EXPLICIT")
        if "direction" in normalized:
            normalized["direction"] = str(normalized.get("direction") or "").strip().upper()
        population_scope = str(
            normalized.get("population_scope")
            or normalized.pop("populationScope", "")
        ).strip().upper()
        if population_scope:
            normalized["population_scope"] = population_scope
    else:
        # Population is an execution-graph concern, not a property of an
        # original-question coverage Goal.  Older checkpoints may still carry
        # these fields on DETAIL/ENTITY/etc.; discard them during recovery so
        # a plain answer obligation cannot accidentally activate the legacy
        # population gate before semantic discovery.
        normalized.pop("population_scope", None)
        normalized.pop("populationScope", None)
        normalized.pop("population_goal_ids", None)
        normalized.pop("populationGoalIds", None)
    return normalized


def _contract_issues(contract: OriginalQuestionGoalContract) -> list[GoalContractIssue]:
    issues: list[GoalContractIssue] = []
    if contract.contract_version != "original_question_goal_contract.v1":
        issues.append(
            GoalContractIssue(
                code="GOAL_CONTRACT_VERSION_UNSUPPORTED",
                message=f"unsupported goal contract version {contract.contract_version!r}",
                path="contractVersion",
            )
        )
    if not contract.question:
        issues.append(
            GoalContractIssue(
                code="GOAL_CONTRACT_QUESTION_MISSING",
                message="the original question must be retained in the goal contract",
                path="question",
            )
        )
    if not contract.goals:
        issues.append(
            GoalContractIssue(
                code="GOAL_CONTRACT_EMPTY",
                message="a grounded business question requires at least one explicit goal",
                path="goals",
            )
        )

    goal_ids = [goal.goal_id for goal in contract.goals]
    duplicate_ids = sorted({goal_id for goal_id in goal_ids if goal_ids.count(goal_id) > 1})
    for goal_id in duplicate_ids:
        issues.append(
            GoalContractIssue(
                code="DUPLICATE_GOAL_ID",
                message=f"goal ID {goal_id!r} occurs more than once after normalization",
                goal_id=goal_id,
            )
        )

    goal_id_set = set(goal_ids)
    goal_by_id = contract.goal_map()
    for goal in contract.goals:
        if not goal.label:
            issues.append(
                GoalContractIssue(
                    code="GOAL_LABEL_MISSING",
                    message=f"goal {goal.goal_id!r} must retain a human-readable label from Core",
                    goal_id=goal.goal_id,
                )
            )
        for source_span in goal.source_spans:
            if source_span not in contract.question:
                issues.append(
                    GoalContractIssue(
                        code="GOAL_SOURCE_SPAN_NOT_IN_QUESTION",
                        message=(
                            f"goal {goal.goal_id!r} sourceSpan {source_span!r} is not present in the original question"
                        ),
                        goal_id=goal.goal_id,
                        path="sourceSpans",
                    )
                )
        if isinstance(goal, TimeWindowQuestionGoal) and not (goal.time_expression or (goal.start and goal.end)):
            issues.append(
                GoalContractIssue(
                    code="TIME_WINDOW_DEFINITION_MISSING",
                    message=f"time-window goal {goal.goal_id!r} requires timeExpression or both start and end",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, ComparisonQuestionGoal) and (not goal.left_goal_ids or not goal.right_goal_ids):
            issues.append(
                GoalContractIssue(
                    code="COMPARISON_OPERANDS_MISSING",
                    message=f"comparison goal {goal.goal_id!r} requires non-empty leftGoalIds and rightGoalIds",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, DependencyQuestionGoal) and (not goal.upstream_goal_ids or not goal.downstream_goal_ids):
            issues.append(
                GoalContractIssue(
                    code="DEPENDENCY_ENDPOINTS_MISSING",
                    message=f"dependency goal {goal.goal_id!r} requires upstreamGoalIds and downstreamGoalIds",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, RankingQuestionGoal):
            if not goal.metric_goal_ids:
                issues.append(
                    GoalContractIssue(
                        code="RANKING_METRIC_GOAL_MISSING",
                        message=f"ranking goal {goal.goal_id!r} requires metricGoalIds",
                        goal_id=goal.goal_id,
                    )
                )
            if goal.direction not in {"ASC", "DESC"}:
                issues.append(
                    GoalContractIssue(
                        code="RANKING_DIRECTION_INVALID",
                        message=f"ranking goal {goal.goal_id!r} requires ASC or DESC direction",
                        goal_id=goal.goal_id,
                    )
                )
            if goal.limit <= 0:
                issues.append(
                    GoalContractIssue(
                        code="RANKING_LIMIT_REQUIRED",
                        message=f"ranking goal {goal.goal_id!r} requires an explicit positive limit",
                        goal_id=goal.goal_id,
                    )
                )
            if goal.population_scope == "ALL_MATCHING_ROWS":
                if goal.population_goal_ids:
                    issues.append(
                        GoalContractIssue(
                            code="RANKING_POPULATION_GOALS_UNEXPECTED",
                            message=(
                                f"ranking goal {goal.goal_id!r} declares ALL_MATCHING_ROWS "
                                "but also supplies populationGoalIds"
                            ),
                            goal_id=goal.goal_id,
                        )
                    )
            elif goal.population_scope in {
                "SAME_AS_GOAL",
                "VERIFIED_ENTITY_SET",
            } and not goal.population_goal_ids:
                issues.append(
                    GoalContractIssue(
                        code="RANKING_POPULATION_GOALS_REQUIRED",
                        message=(
                            f"ranking goal {goal.goal_id!r} populationScope "
                            f"{goal.population_scope!r} requires populationGoalIds"
                        ),
                        goal_id=goal.goal_id,
                    )
                )
            elif goal.population_scope in {
                "VERIFIED_PREDICATE_SCOPE",
                "VERIFIED_RESULT_ARTIFACT",
            } and goal.population_goal_ids:
                issues.append(
                    GoalContractIssue(
                        code="RANKING_CROSS_TURN_POPULATION_GOALS_UNEXPECTED",
                        message=(
                            f"ranking goal {goal.goal_id!r} uses a verified prior artifact "
                            "and must not reference current-turn populationGoalIds"
                        ),
                        goal_id=goal.goal_id,
                    )
                )
            for population_goal_id in goal.population_goal_ids:
                population_goal = goal_by_id.get(population_goal_id)
                if population_goal is not None and not isinstance(
                    population_goal,
                    (DetailQuestionGoal, EntityQuestionGoal, RankingQuestionGoal),
                ):
                    issues.append(
                        GoalContractIssue(
                            code="RANKING_POPULATION_GOAL_KIND_INVALID",
                            message=(
                                f"ranking goal {goal.goal_id!r} populationGoalId "
                                f"{population_goal_id!r} must reference DETAIL, "
                                "ENTITY or RANKING"
                            ),
                            goal_id=goal.goal_id,
                            reference_goal_id=population_goal_id,
                        )
                    )
            for metric_goal_id in goal.metric_goal_ids:
                referenced = goal_by_id.get(metric_goal_id)
                if referenced is not None and not isinstance(referenced, MetricQuestionGoal):
                    issues.append(
                        GoalContractIssue(
                            code="RANKING_METRIC_GOAL_KIND_INVALID",
                            message=(
                                f"ranking goal {goal.goal_id!r} metricGoalId "
                                f"{metric_goal_id!r} must reference a METRIC goal"
                            ),
                            goal_id=goal.goal_id,
                            reference_goal_id=metric_goal_id,
                        )
                    )
            for dimension_goal_id in goal.dimension_goal_ids:
                referenced = goal_by_id.get(dimension_goal_id)
                if referenced is not None and not isinstance(
                    referenced,
                    (DimensionQuestionGoal, EntityQuestionGoal),
                ):
                    issues.append(
                        GoalContractIssue(
                            code="RANKING_DIMENSION_GOAL_KIND_INVALID",
                            message=(
                                f"ranking goal {goal.goal_id!r} dimensionGoalId "
                                f"{dimension_goal_id!r} must reference a "
                                "DIMENSION or ENTITY goal"
                            ),
                            goal_id=goal.goal_id,
                            reference_goal_id=dimension_goal_id,
                        )
                    )
        if isinstance(goal, AnalysisQuestionGoal):
            if not goal.analysis_type:
                issues.append(
                    GoalContractIssue(
                        code="ANALYSIS_TYPE_MISSING",
                        message=f"analysis goal {goal.goal_id!r} requires analysisType",
                        goal_id=goal.goal_id,
                    )
                )
            if not goal.input_goal_ids:
                issues.append(
                    GoalContractIssue(
                        code="ANALYSIS_INPUT_GOALS_MISSING",
                        message=f"analysis goal {goal.goal_id!r} requires inputGoalIds",
                        goal_id=goal.goal_id,
                    )
                )

        for reference_id in _all_goal_references(goal):
            if reference_id == goal.goal_id:
                issues.append(
                    GoalContractIssue(
                        code="SELF_REFERENTIAL_GOAL",
                        message=f"goal {goal.goal_id!r} cannot reference itself",
                        goal_id=goal.goal_id,
                        reference_goal_id=reference_id,
                    )
                )
            elif reference_id not in goal_id_set:
                issues.append(
                    GoalContractIssue(
                        code="UNKNOWN_GOAL_REFERENCE",
                        message=f"goal {goal.goal_id!r} references unknown goal {reference_id!r}",
                        goal_id=goal.goal_id,
                        reference_goal_id=reference_id,
                    )
                )

    if not duplicate_ids and not any(issue.code == "UNKNOWN_GOAL_REFERENCE" for issue in issues):
        cycle = _dependency_cycle(_goal_dependency_graph(contract))
        if cycle:
            issues.append(
                GoalContractIssue(
                    code="GOAL_DEPENDENCY_CYCLE",
                    message="goal dependency graph contains a cycle: " + " -> ".join(cycle),
                    goal_id=cycle[0],
                )
            )
    return issues


def _question_verifier_issues(
    contract: OriginalQuestionGoalContract,
    verifier: GoalContractQuestionVerifier,
) -> list[GoalContractIssue]:
    """Invoke an explicit semantic verifier and fail closed on its failure."""

    try:
        raw_issues = verifier.verify(contract.model_copy(deep=True))
        if isinstance(raw_issues, (str, bytes)) or not isinstance(raw_issues, Sequence):
            raise TypeError("question verifier must return a sequence of typed issues")
        issues: list[GoalContractIssue] = []
        for raw_issue in raw_issues:
            issue = (
                raw_issue.model_copy(deep=True)
                if isinstance(raw_issue, GoalContractIssue)
                else GoalContractIssue.model_validate(raw_issue)
            )
            issue.code = str(issue.code or "").strip()
            issue.message = str(issue.message or "").strip()
            if not issue.code or not issue.message:
                raise ValueError("question verifier issues require non-empty code and message")
            issues.append(issue)
        return issues
    except Exception as exc:
        return [
            GoalContractIssue(
                code="GOAL_CONTRACT_QUESTION_VERIFIER_FAILED",
                message=f"the configured question verifier failed closed: {exc}",
                path="question",
            )
        ]


def _goal_dependency_graph(contract: OriginalQuestionGoalContract) -> dict[str, set[str]]:
    graph = {goal.goal_id: set(goal.depends_on_goal_ids) for goal in contract.goals}
    for goal in contract.goals:
        if isinstance(goal, ComparisonQuestionGoal):
            graph[goal.goal_id].update(goal.left_goal_ids)
            graph[goal.goal_id].update(goal.right_goal_ids)
        elif isinstance(goal, EntityQuestionGoal):
            graph[goal.goal_id].update(goal.source_goal_ids)
        elif isinstance(goal, TimeWindowQuestionGoal):
            for target_goal_id in goal.applies_to_goal_ids:
                graph.setdefault(target_goal_id, set()).add(goal.goal_id)
        elif isinstance(goal, DependencyQuestionGoal):
            graph[goal.goal_id].update(goal.upstream_goal_ids)
            for downstream_goal_id in goal.downstream_goal_ids:
                graph.setdefault(downstream_goal_id, set()).add(goal.goal_id)
        elif isinstance(goal, DetailQuestionGoal):
            graph[goal.goal_id].update(goal.input_goal_ids)
        elif isinstance(goal, RankingQuestionGoal):
            graph[goal.goal_id].update(goal.metric_goal_ids)
            graph[goal.goal_id].update(goal.dimension_goal_ids)
            graph[goal.goal_id].update(goal.population_goal_ids)
        elif isinstance(goal, AnalysisQuestionGoal):
            graph[goal.goal_id].update(goal.input_goal_ids)
            graph[goal.goal_id].update(goal.baseline_goal_ids)
    return graph


def goal_dependency_closure(
    contract: OriginalQuestionGoalContract,
    target_goal_ids: Sequence[str],
) -> set[str]:
    """Return every typed direct or transitive input of target Goals."""

    graph = _goal_dependency_graph(contract)
    targets = {
        str(goal_id or "").strip()
        for goal_id in target_goal_ids
        if str(goal_id or "").strip()
    }
    pending = list(targets)
    dependencies: set[str] = set()
    cursor = 0
    while cursor < len(pending):
        current = pending[cursor]
        cursor += 1
        for dependency in graph.get(current, set()):
            if dependency in dependencies or dependency in targets:
                continue
            dependencies.add(dependency)
            pending.append(dependency)
    return dependencies


def _all_goal_references(goal: QuestionGoal) -> list[str]:
    references = list(goal.depends_on_goal_ids)
    if isinstance(goal, TimeWindowQuestionGoal):
        references.extend(goal.applies_to_goal_ids)
    elif isinstance(goal, ComparisonQuestionGoal):
        references.extend(goal.left_goal_ids)
        references.extend(goal.right_goal_ids)
    elif isinstance(goal, EntityQuestionGoal):
        references.extend(goal.source_goal_ids)
    elif isinstance(goal, DependencyQuestionGoal):
        references.extend(goal.upstream_goal_ids)
        references.extend(goal.downstream_goal_ids)
    elif isinstance(goal, DetailQuestionGoal):
        references.extend(goal.input_goal_ids)
    elif isinstance(goal, RankingQuestionGoal):
        references.extend(goal.metric_goal_ids)
        references.extend(goal.dimension_goal_ids)
        references.extend(goal.population_goal_ids)
    elif isinstance(goal, AnalysisQuestionGoal):
        references.extend(goal.input_goal_ids)
        references.extend(goal.baseline_goal_ids)
    return list(dict.fromkeys(references))


def _dependency_cycle(graph: Mapping[str, set[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(goal_id: str) -> list[str]:
        if goal_id in visiting:
            start = path.index(goal_id)
            return path[start:] + [goal_id]
        if goal_id in visited:
            return []
        visiting.add(goal_id)
        path.append(goal_id)
        for dependency_id in sorted(graph.get(goal_id, set())):
            cycle = visit(dependency_id)
            if cycle:
                return cycle
        path.pop()
        visiting.remove(goal_id)
        visited.add(goal_id)
        return []

    for goal_id in graph:
        cycle = visit(goal_id)
        if cycle:
            return cycle
    return []


def _coerce_artifact_coverage(artifact: Any) -> VerifiedArtifactGoalCoverage:
    if isinstance(artifact, VerifiedArtifactGoalCoverage):
        payload = artifact.model_dump(by_alias=False)
    elif isinstance(artifact, Mapping):
        payload = {
            "artifact_id": _object_value(artifact, "artifact_id", "artifactId"),
            "goal_contract_fingerprint": _object_value(
                artifact, "goal_contract_fingerprint", "goalContractFingerprint"
            ),
            "covered_goal_ids": _object_value(artifact, "covered_goal_ids", "coveredGoalIds"),
            "verification_passed": _object_value(artifact, "verification_passed", "verificationPassed"),
            "evidence_refs": _object_value(
                artifact,
                "evidence_refs",
                "evidenceRefs",
                "goal_coverage_evidence_refs",
                "goalCoverageEvidenceRefs",
            ),
            "goal_resolutions": _object_value(
                artifact,
                "goal_resolutions",
                "goalResolutions",
                "goal_proofs",
                "goalProofs",
            ),
        }
        verified_evidence = _object_value(artifact, "verified_evidence", "verifiedEvidence")
        if payload["verification_passed"] is None and verified_evidence is not None:
            payload["verification_passed"] = bool(_object_value(verified_evidence, "passed"))
    else:
        payload = {
            "artifact_id": _object_value(artifact, "artifact_id", "artifactId"),
            "goal_contract_fingerprint": _object_value(
                artifact, "goal_contract_fingerprint", "goalContractFingerprint"
            ),
            "covered_goal_ids": _object_value(artifact, "covered_goal_ids", "coveredGoalIds"),
            "evidence_refs": _object_value(artifact, "goal_coverage_evidence_refs", "goalCoverageEvidenceRefs") or [],
            "goal_resolutions": _object_value(
                artifact,
                "goal_resolutions",
                "goalResolutions",
                "goal_proofs",
                "goalProofs",
            )
            or [],
        }
        verified_evidence = _object_value(artifact, "verified_evidence", "verifiedEvidence")
        payload["verification_passed"] = bool(_object_value(verified_evidence, "passed"))

    if "verification_passed" not in payload and "verificationPassed" not in payload:
        verified_evidence = _object_value(payload, "verified_evidence", "verifiedEvidence")
        if verified_evidence is not None:
            payload["verification_passed"] = bool(_object_value(verified_evidence, "passed"))
    if "evidence_refs" not in payload and "evidenceRefs" not in payload:
        artifact_evidence_refs = _object_value(
            payload,
            "goal_coverage_evidence_refs",
            "goalCoverageEvidenceRefs",
        )
        if artifact_evidence_refs is not None:
            payload["evidence_refs"] = artifact_evidence_refs
    payload.pop("verified_evidence", None)
    payload.pop("verifiedEvidence", None)
    payload.pop("goal_coverage_evidence_refs", None)
    payload.pop("goalCoverageEvidenceRefs", None)

    for snake_name in (
        "artifact_id",
        "goal_contract_fingerprint",
        "covered_goal_ids",
        "verification_passed",
        "evidence_refs",
        "goal_resolutions",
    ):
        camel_name = _camel_name(snake_name)
        if camel_name in payload and snake_name not in payload:
            payload[snake_name] = payload.pop(camel_name)
    payload["artifact_id"] = str(payload.get("artifact_id") or "").strip()
    payload["goal_contract_fingerprint"] = str(payload.get("goal_contract_fingerprint") or "").strip()
    payload["covered_goal_ids"] = _canonical_goal_id_list(payload.get("covered_goal_ids") or [])
    payload["evidence_refs"] = _normalized_string_list(payload.get("evidence_refs") or [])
    payload["goal_resolutions"] = [
        _normalize_goal_resolution_payload(item) for item in payload.get("goal_resolutions") or []
    ]
    return VerifiedArtifactGoalCoverage.model_validate(payload)


def _normalize_goal_resolution_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, GoalProofResolutionBase):
        normalized = payload.model_dump(by_alias=False)
    elif isinstance(payload, Mapping):
        normalized = dict(payload)
    else:
        raise TypeError("goal resolutions must be objects")

    aliases = {
        "goalId": "goal_id",
        "goalKind": "goal_kind",
        "kind": "goal_kind",
        "status": "resolution",
        "resolutionStatus": "resolution",
    }
    for alias, field_name in aliases.items():
        if alias in normalized and field_name not in normalized:
            normalized[field_name] = normalized.pop(alias)

    normalized["goal_id"] = canonical_goal_id(normalized.get("goal_id"))
    normalized["goal_kind"] = _canonical_goal_kind(normalized.get("goal_kind"))
    normalized["resolution"] = _canonical_resolution_status(normalized.get("resolution"))

    goal_id_list_fields = (
        "operand_goal_ids",
        "order_by_goal_ids",
        "dimension_goal_ids",
        "input_goal_ids",
        "population_goal_ids",
    )
    string_list_fields = (
        "evidence_refs",
        "metric_ref_ids",
        "value_refs",
        "dimension_ref_ids",
        "output_fields",
        "output_semantic_refs",
        "baseline_refs",
        "entity_ref_ids",
        "identity_fields",
        "upstream_artifact_ids",
        "downstream_artifact_ids",
        "lineage_refs",
        "rule_ref_ids",
        "citation_refs",
        "population_lineage_refs",
    )
    for snake_name in goal_id_list_fields + string_list_fields:
        camel_name = _camel_name(snake_name)
        if camel_name in normalized and snake_name not in normalized:
            normalized[snake_name] = normalized.pop(camel_name)
    for field_name in goal_id_list_fields:
        if field_name in normalized:
            normalized[field_name] = _canonical_goal_id_list(normalized[field_name])
    for field_name in string_list_fields:
        if field_name in normalized:
            normalized[field_name] = _normalized_string_list(normalized[field_name])
    for field_name, value in list(normalized.items()):
        if isinstance(value, str):
            normalized[field_name] = value.strip()
    if "direction" in normalized:
        normalized["direction"] = str(normalized["direction"] or "").upper()
    return normalized


def _parse_goal_resolution(payload: GoalProofResolution | Mapping[str, Any]) -> GoalProofResolution:
    if isinstance(payload, GoalProofResolutionBase):
        return payload
    return TypeAdapter(GoalProofResolution).validate_python(_normalize_goal_resolution_payload(payload))


def _legacy_primitive_goal_resolution(
    goal: QuestionGoal,
    *,
    artifact_id: str,
    evidence_refs: Sequence[str],
    artifact: Any | None = None,
) -> dict[str, Any] | None:
    """Create a narrow compatibility proof for already-verified primitives.

    This intentionally excludes goals whose truth depends on ordering,
    comparison, lineage, row-shape or interpretation.  Those goals must always
    publish an explicit typed resolution.
    """

    base: dict[str, Any] = {
        "goal_id": goal.goal_id,
        "goal_kind": goal.kind,
        "resolution": GoalResolutionStatus.PROVED.value,
        "proof_type": "KERNEL_VERIFIED_PRIMITIVE_RESULT",
        "evidence_refs": _normalized_string_list(evidence_refs),
    }
    artifact_ref = f"artifact:{artifact_id}"
    if isinstance(goal, MetricQuestionGoal):
        base.update(
            {
                "metric_ref_ids": _normalized_string_list([goal.metric_ref_id, *goal.semantic_ref_ids]),
                "value_refs": [f"{artifact_ref}:metric-value"],
            }
        )
    elif isinstance(goal, DimensionQuestionGoal):
        base.update(
            {
                "dimension_ref_ids": _normalized_string_list([goal.dimension_ref_id, *goal.semantic_ref_ids]),
                "output_fields": [goal.dimension_ref_id or f"{artifact_ref}:dimension-output"],
            }
        )
    elif isinstance(goal, TimeWindowQuestionGoal):
        # A time window is not a primitive fact of the question declaration.
        # It is proved only by the executed artifact's resolved time contract.
        # In particular, never copy the Goal's expression/bounds into a proof:
        # doing so would let a 30-day query masquerade as "最近7天".
        actual = _artifact_time_window_payload(artifact)
        if actual is None:
            return None
        base.update(actual)
    elif isinstance(goal, EntityQuestionGoal):
        base.update(
            {
                "entity_ref_ids": _normalized_string_list([goal.entity_ref_id, *goal.semantic_ref_ids]),
                "identity_fields": [goal.entity_identity or goal.entity_ref_id or f"{artifact_ref}:entity-identity"],
                "entity_set_ref": f"{artifact_ref}:entity-set",
            }
        )
    else:
        return None
    return base


def _artifact_time_window_payload(artifact: Any | None) -> dict[str, Any] | None:
    """Return only time facts sealed by a verified query contract.

    Runtime execution dates are preferred because they are the snapshot bounds
    actually used by SQL; declared dates are the fallback for older artifacts.
    ``binding_hints.time_expression`` is the formal query-contract expression,
    while the resolved range label is retained as a safe fallback.
    """

    contract = _object_value(artifact, "contract")
    time_range = _object_value(contract, "time_range", "timeRange")
    if time_range is None:
        return None
    execution_start = str(
        _object_value(time_range, "execution_start_date", "executionStartDate")
        or _object_value(time_range, "execution_start_value", "executionStartValue")
        or _object_value(time_range, "start_date", "startDate")
        or ""
    ).strip()
    execution_end = str(
        _object_value(time_range, "execution_end_date", "executionEndDate")
        or _object_value(time_range, "execution_end_value", "executionEndValue")
        or _object_value(time_range, "end_date", "endDate")
        or ""
    ).strip()
    hints = _object_value(contract, "binding_hints", "bindingHints")
    expression = str(
        _object_value(time_range, "label")
        or _object_value(hints, "time_expression", "timeExpression")
        or ""
    ).strip()
    return {
        "time_expression": expression,
        "start": execution_start,
        "end": execution_end,
        "timezone": str(_object_value(time_range, "timezone") or "").strip(),
        "granularity": str(_object_value(time_range, "granularity") or "").strip(),
        "days": int(_object_value(time_range, "days") or 0),
        "label": str(_object_value(time_range, "label") or "").strip(),
        "explicit": bool(_object_value(time_range, "explicit")),
        "calendar_anchor_policy": str(
            _object_value(time_range, "calendar_anchor_policy", "calendarAnchorPolicy")
            or ""
        ).strip(),
        "data_as_of_policy": str(
            _object_value(time_range, "data_as_of_policy", "dataAsOfPolicy")
            or ""
        ).strip(),
        "window_role": str(_object_value(time_range, "window_role", "windowRole") or "").strip(),
        "time_range_kind": str(_object_value(time_range, "kind") or "").strip(),
    }


def _goal_semantic_refs(goal: QuestionGoal) -> set[str]:
    refs = set(goal.semantic_ref_ids)
    for attribute in ("metric_ref_id", "dimension_ref_id", "entity_ref_id"):
        value = str(getattr(goal, attribute, "") or "").strip()
        if value:
            refs.add(value)
    for attribute in ("rule_ref_ids", "required_field_ref_ids"):
        refs.update(_normalized_string_list(getattr(goal, attribute, []) or []))
    return refs


def _semantic_evidence_issue(
    goal: QuestionGoal,
    *,
    artifact_id: str,
    artifact_evidence_refs: Sequence[str],
    resolution_evidence_refs: Sequence[str],
) -> GoalCoverageIssue | None:
    declared_semantic_refs = _canonical_semantic_refs(_goal_semantic_refs(goal))
    evidence = _canonical_semantic_refs(
        [*artifact_evidence_refs, *resolution_evidence_refs]
    )
    if not declared_semantic_refs or declared_semantic_refs.intersection(evidence):
        return None
    return GoalCoverageIssue(
        code="GOAL_SEMANTIC_EVIDENCE_UNCOVERED",
        message=(
            f"artifact {artifact_id!r} claimed goal {goal.goal_id!r} without any of its declared semantic evidence refs"
        ),
        goal_id=goal.goal_id,
        artifact_id=artifact_id,
        details={
            "declaredSemanticRefIds": sorted(declared_semantic_refs),
            "artifactEvidenceRefs": sorted(evidence),
        },
    )


def _goal_resolution_issues(
    goal: QuestionGoal,
    resolution: GoalProofResolution,
    *,
    artifact_id: str,
    accepted_artifact_ids: set[str],
    goal_map: Mapping[str, QuestionGoal],
) -> list[GoalCoverageIssue]:
    issues: list[GoalCoverageIssue] = []

    def add(code: str, message: str, **details: Any) -> None:
        issues.append(
            GoalCoverageIssue(
                code=code,
                message=message,
                goal_id=goal.goal_id,
                artifact_id=artifact_id,
                details=details,
            )
        )

    if resolution.resolution == GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value:
        if not resolution.reason:
            add(
                "INSUFFICIENT_EVIDENCE_REASON_MISSING",
                f"goal {goal.goal_id!r} requires a reason for INSUFFICIENT_EVIDENCE",
            )
        return issues

    if isinstance(resolution, MetricGoalProofResolution):
        if not resolution.value_refs:
            add(
                "METRIC_PROOF_VALUE_MISSING",
                f"metric goal {goal.goal_id!r} has no verified value reference",
            )
    elif isinstance(resolution, DimensionGoalProofResolution):
        if not resolution.output_fields:
            add(
                "DIMENSION_PROOF_OUTPUT_MISSING",
                f"dimension goal {goal.goal_id!r} has no verified output field",
            )
    elif isinstance(resolution, TimeWindowGoalProofResolution):
        if not resolution.time_expression and not (resolution.start and resolution.end):
            add(
                "TIME_WINDOW_PROOF_BOUNDARY_MISSING",
                f"time goal {goal.goal_id!r} has no verified expression or boundaries",
            )
        if not resolution.explicit:
            add(
                "TIME_WINDOW_PROOF_NOT_EXPLICIT",
                f"time goal {goal.goal_id!r} was not proved by an explicit query time range",
            )
        if resolution.days <= 0:
            add(
                "TIME_WINDOW_PROOF_DAYS_INVALID",
                f"time goal {goal.goal_id!r} has no positive executed window length",
                actualDays=resolution.days,
            )
        if isinstance(goal, TimeWindowQuestionGoal):
            actual_expression = resolution.time_expression or resolution.label
            _add_time_value_mismatch(
                add,
                goal_id=goal.goal_id,
                field_name="timeExpression",
                expected=goal.time_expression,
                actual=actual_expression,
                code="TIME_WINDOW_PROOF_EXPRESSION_MISMATCH",
            )
            for field_name, expected, actual in (
                ("start", goal.start, resolution.start),
                ("end", goal.end, resolution.end),
                ("timezone", goal.timezone, resolution.timezone),
                (
                    "calendarAnchorPolicy",
                    goal.calendar_anchor_policy,
                    resolution.calendar_anchor_policy,
                ),
                (
                    "dataAsOfPolicy",
                    goal.data_as_of_policy,
                    resolution.data_as_of_policy,
                ),
            ):
                _add_time_value_mismatch(
                    add,
                    goal_id=goal.goal_id,
                    field_name=field_name,
                    expected=expected,
                    actual=actual,
                    code="TIME_WINDOW_PROOF_%s_MISMATCH"
                    % _canonical_ascii_symbol(field_name),
                )
            if not _time_range_kinds_equivalent(
                goal.time_range_kind,
                resolution.time_range_kind,
            ):
                _add_time_value_mismatch(
                    add,
                    goal_id=goal.goal_id,
                    field_name="timeRangeKind",
                    expected=goal.time_range_kind,
                    actual=resolution.time_range_kind,
                    code="TIME_WINDOW_PROOF_TIMERANGEKIND_MISMATCH",
                )
            if not _time_window_roles_equivalent(
                goal.window_role,
                resolution.window_role,
                goal_map=goal_map,
            ):
                _add_time_value_mismatch(
                    add,
                    goal_id=goal.goal_id,
                    field_name="windowRole",
                    expected=goal.window_role,
                    actual=resolution.window_role,
                    code="TIME_WINDOW_PROOF_WINDOWROLE_MISMATCH",
                )
            # ``ResolvedTimeRange`` does not carry an executed day/week/month
            # grain.  TREND query shape proves only that a series was
            # requested, not its concrete grain, so Goal granularity must be
            # checked by a future authoritative query-contract field rather
            # than compared with this necessarily empty proof attribute.
            if goal.days > 0 and resolution.days != goal.days:
                add(
                    "TIME_WINDOW_PROOF_DAYS_MISMATCH",
                    f"time goal {goal.goal_id!r} executed a different number of days",
                    expectedDays=goal.days,
                    actualDays=resolution.days,
                )
    elif isinstance(resolution, ComparisonGoalProofResolution):
        expected_operands = set(getattr(goal, "left_goal_ids", []) + getattr(goal, "right_goal_ids", []))
        actual_operands = set(resolution.operand_goal_ids)
        if not expected_operands.issubset(actual_operands):
            add(
                "COMPARISON_PROOF_OPERANDS_INCOMPLETE",
                f"comparison goal {goal.goal_id!r} does not prove every declared operand",
                expectedOperandGoalIds=sorted(expected_operands),
                actualOperandGoalIds=sorted(actual_operands),
            )
        if not resolution.comparison_method:
            add(
                "COMPARISON_PROOF_METHOD_MISSING",
                f"comparison goal {goal.goal_id!r} has no deterministic comparison method",
            )
        if not resolution.result_ref:
            add(
                "COMPARISON_PROOF_RESULT_MISSING",
                f"comparison goal {goal.goal_id!r} has no verified comparison result",
            )
        if _is_anomaly_goal(goal, resolution):
            if not resolution.baseline_refs:
                add(
                    "ANOMALY_PROOF_BASELINE_MISSING",
                    f"anomaly goal {goal.goal_id!r} requires comparable baseline evidence",
                )
            if not resolution.normalization_method:
                add(
                    "ANOMALY_PROOF_NORMALIZATION_MISSING",
                    f"anomaly goal {goal.goal_id!r} requires a normalization method",
                )
    elif isinstance(resolution, EntityGoalProofResolution):
        if not resolution.entity_set_ref and not resolution.identity_fields:
            add(
                "ENTITY_PROOF_IDENTITY_MISSING",
                f"entity goal {goal.goal_id!r} has no verified identity or entity set",
            )
    elif isinstance(resolution, DependencyGoalProofResolution):
        if not resolution.upstream_artifact_ids:
            add(
                "DEPENDENCY_PROOF_UPSTREAM_ARTIFACT_MISSING",
                f"dependency goal {goal.goal_id!r} has no upstream artifact lineage",
            )
        if not resolution.downstream_artifact_ids:
            add(
                "DEPENDENCY_PROOF_DOWNSTREAM_ARTIFACT_MISSING",
                f"dependency goal {goal.goal_id!r} has no downstream artifact lineage",
            )
        if not resolution.lineage_refs:
            add(
                "DEPENDENCY_PROOF_LINEAGE_MISSING",
                f"dependency goal {goal.goal_id!r} has no verified entity/artifact lineage reference",
            )
        referenced_artifacts = set(resolution.upstream_artifact_ids) | set(resolution.downstream_artifact_ids)
        unknown_artifacts = sorted(referenced_artifacts - accepted_artifact_ids)
        if unknown_artifacts:
            add(
                "DEPENDENCY_PROOF_ARTIFACT_UNKNOWN",
                f"dependency goal {goal.goal_id!r} references artifacts outside the verified ledger",
                unknownArtifactIds=unknown_artifacts,
            )
    elif isinstance(resolution, RuleGoalProofResolution):
        if not resolution.citation_refs and not resolution.evidence_refs:
            add(
                "RULE_PROOF_CITATION_MISSING",
                f"rule goal {goal.goal_id!r} requires at least one verified citation",
            )
    elif isinstance(resolution, DetailGoalProofResolution):
        if not resolution.row_set_ref:
            add(
                "DETAIL_PROOF_ROW_SET_MISSING",
                f"detail goal {goal.goal_id!r} has no verified row-set reference",
            )
        if not resolution.output_fields:
            add(
                "DETAIL_PROOF_FIELDS_MISSING",
                f"detail goal {goal.goal_id!r} has no verified output fields",
            )
        if resolution.row_count is not None and resolution.row_count < 0:
            add(
                "DETAIL_PROOF_ROW_COUNT_INVALID",
                f"detail goal {goal.goal_id!r} has an invalid negative row count",
            )
        if isinstance(goal, DetailQuestionGoal):
            # Generic semantic refs prove that the Detail Goal is grounded;
            # they may identify a table, metric, or rule and are checked by
            # ``_semantic_evidence_issue``.  Only explicitly requested field
            # refs are obligations on the artifact's output columns.
            expected_refs = _canonical_semantic_refs(
                goal.required_field_ref_ids
            )
            actual_refs = _canonical_semantic_refs(
                resolution.output_semantic_refs
            )
            missing_refs = sorted(expected_refs - actual_refs)
            if missing_refs:
                add(
                    "DETAIL_PROOF_REQUIRED_FIELDS_MISSING",
                    f"detail goal {goal.goal_id!r} does not output every requested semantic field",
                    missingSemanticRefIds=missing_refs,
                    actualOutputSemanticRefIds=sorted(actual_refs),
                    rejectedRefIds=missing_refs,
                    readNext="read the published field definitions and regenerate the detail output binding",
                )
    elif isinstance(resolution, RankingGoalProofResolution):
        expected_order_goals = set(getattr(goal, "metric_goal_ids", []))
        actual_order_goals = set(resolution.order_by_goal_ids)
        if expected_order_goals != actual_order_goals:
            add(
                "RANKING_PROOF_ORDER_GOAL_MISMATCH",
                f"ranking goal {goal.goal_id!r} was not ordered by exactly its declared metric goals",
                expectedOrderGoalIds=sorted(expected_order_goals),
                actualOrderGoalIds=sorted(actual_order_goals),
            )
        expected_dimension_goals = set(getattr(goal, "dimension_goal_ids", []))
        actual_dimension_goals = set(resolution.dimension_goal_ids)
        if expected_dimension_goals != actual_dimension_goals:
            add(
                "RANKING_PROOF_DIMENSION_GOAL_MISMATCH",
                f"ranking goal {goal.goal_id!r} was not grouped by exactly its declared dimension goals",
                expectedDimensionGoalIds=sorted(expected_dimension_goals),
                actualDimensionGoalIds=sorted(actual_dimension_goals),
            )
        if not resolution.ranking_metric_ref_id:
            add(
                "RANKING_PROOF_METRIC_REF_MISSING",
                f"ranking goal {goal.goal_id!r} has no actual ranking metric semantic ref",
            )
        elif not _ranking_ref_matches_goals(
            resolution.ranking_metric_ref_id,
            resolution.order_by_goal_ids,
            goal_map,
            allow_unique_binding=(
                str(resolution.details.get("metricBindingMode") or "")
                == "UNIQUE_CONTRACT_BINDING"
            ),
        ):
            add(
                "RANKING_PROOF_METRIC_REF_MISMATCH",
                f"ranking goal {goal.goal_id!r} actual metric ref does not match its metric goal",
                actualMetricRefId=resolution.ranking_metric_ref_id,
                actualOrderGoalIds=list(resolution.order_by_goal_ids),
            )
        if expected_dimension_goals and not resolution.ranking_dimension_ref_id:
            add(
                "RANKING_PROOF_DIMENSION_REF_MISSING",
                f"ranking goal {goal.goal_id!r} has no actual ranking dimension semantic ref",
            )
        elif resolution.ranking_dimension_ref_id and not _ranking_ref_matches_goals(
            resolution.ranking_dimension_ref_id,
            resolution.dimension_goal_ids,
            goal_map,
            allow_unique_binding=(
                str(resolution.details.get("dimensionBindingMode") or "")
                == "UNIQUE_CONTRACT_BINDING"
            ),
        ):
            add(
                "RANKING_PROOF_DIMENSION_REF_MISMATCH",
                f"ranking goal {goal.goal_id!r} actual dimension ref does not match its dimension goal",
                actualDimensionRefId=resolution.ranking_dimension_ref_id,
                actualDimensionGoalIds=list(resolution.dimension_goal_ids),
            )
        expected_direction = str(getattr(goal, "direction", "") or "").upper()
        if resolution.direction not in {"ASC", "DESC"}:
            add(
                "RANKING_PROOF_DIRECTION_INVALID",
                f"ranking goal {goal.goal_id!r} requires ASC or DESC direction",
            )
        elif resolution.direction != expected_direction:
            add(
                "RANKING_PROOF_DIRECTION_MISMATCH",
                f"ranking goal {goal.goal_id!r} executed a different sort direction",
                expectedDirection=expected_direction,
                actualDirection=resolution.direction,
            )
        if resolution.limit <= 0:
            add(
                "RANKING_PROOF_LIMIT_INVALID",
                f"ranking goal {goal.goal_id!r} requires a positive verified limit",
            )
        elif resolution.limit != int(getattr(goal, "limit", 0) or 0):
            add(
                "RANKING_PROOF_LIMIT_MISMATCH",
                f"ranking goal {goal.goal_id!r} executed a different Top-N limit",
                expectedLimit=int(getattr(goal, "limit", 0) or 0),
                actualLimit=resolution.limit,
            )
        if not resolution.row_set_ref:
            add(
                "RANKING_PROOF_ROW_SET_MISSING",
                f"ranking goal {goal.goal_id!r} has no verified ordered row set",
            )
        if resolution.population_scope != goal.population_scope:
            add(
                "RANKING_PROOF_POPULATION_SCOPE_MISMATCH",
                f"ranking goal {goal.goal_id!r} does not prove its declared population scope",
                expectedPopulationScope=goal.population_scope,
                actualPopulationScope=resolution.population_scope,
            )
        if not set(goal.population_goal_ids).issubset(
            set(resolution.population_goal_ids)
        ):
            add(
                "RANKING_PROOF_POPULATION_GOALS_MISSING",
                f"ranking goal {goal.goal_id!r} does not prove its population goals",
                expectedPopulationGoalIds=list(goal.population_goal_ids),
                actualPopulationGoalIds=list(resolution.population_goal_ids),
            )
        if (
            goal.population_scope != "ALL_MATCHING_ROWS"
            and not resolution.population_lineage_refs
        ):
            add(
                "RANKING_PROOF_POPULATION_LINEAGE_MISSING",
                f"ranking goal {goal.goal_id!r} has no verified population lineage",
            )
    elif isinstance(resolution, AnalysisGoalProofResolution):
        expected_inputs = set(getattr(goal, "input_goal_ids", []))
        if not expected_inputs.issubset(set(resolution.input_goal_ids)):
            add(
                "ANALYSIS_PROOF_INPUTS_INCOMPLETE",
                f"analysis goal {goal.goal_id!r} does not reference every declared input",
                expectedInputGoalIds=sorted(expected_inputs),
                actualInputGoalIds=sorted(resolution.input_goal_ids),
            )
        if not resolution.analysis_method:
            add(
                "ANALYSIS_PROOF_METHOD_MISSING",
                f"analysis goal {goal.goal_id!r} has no reproducible method",
            )
        if not resolution.result_ref:
            add(
                "ANALYSIS_PROOF_RESULT_MISSING",
                f"analysis goal {goal.goal_id!r} has no verified result reference",
            )
        if _is_anomaly_goal(goal, resolution):
            if not resolution.baseline_refs:
                add(
                    "ANOMALY_PROOF_BASELINE_MISSING",
                    f"anomaly analysis {goal.goal_id!r} requires comparable baseline evidence",
                )
            if not resolution.normalization_method:
                add(
                    "ANOMALY_PROOF_NORMALIZATION_MISSING",
                    f"anomaly analysis {goal.goal_id!r} requires a normalization method",
                )
    return issues


def _is_anomaly_goal(
    goal: QuestionGoal,
    resolution: GoalProofResolution,
) -> bool:
    capability_types = {
        _canonical_analysis_capability(getattr(goal, "comparison_type", "")),
        _canonical_analysis_capability(getattr(goal, "analysis_type", "")),
        _canonical_analysis_capability(getattr(resolution, "comparison_type", "")),
        _canonical_analysis_capability(getattr(resolution, "analysis_type", "")),
    }
    return any(value == "ANOMALY" or value.startswith("ANOMALY_") for value in capability_types if value)


def _canonical_semantic_ref(value: Any) -> str:
    """Normalize equivalent published field coordinates for proof comparison."""

    text = unicodedata.normalize("NFKC", str(value or "").strip()).casefold()
    # Older assets called a field a ``column`` while newer assets use
    # ``field``.  They identify the same governed coordinate.
    return text.replace(":column:", ":field:")


def _canonical_semantic_refs(values: Sequence[Any]) -> set[str]:
    return {
        normalized
        for normalized in (_canonical_semantic_ref(value) for value in values)
        if normalized
    }


def _ranking_ref_matches_goals(
    actual_ref: str,
    goal_ids: Sequence[str],
    goal_map: Mapping[str, QuestionGoal],
    *,
    allow_unique_binding: bool = False,
) -> bool:
    actual = _canonical_semantic_ref(actual_ref)
    if not actual or not goal_ids:
        return False
    if any(
        actual in _canonical_semantic_refs(_goal_semantic_refs(goal_map[goal_id]))
        for goal_id in goal_ids
        if goal_id in goal_map
    ):
        return True
    # The proof producer may use this only when the executed Contract had one
    # metric/dimension binding and the Goal declared one corresponding slot.
    return allow_unique_binding and len(goal_ids) == 1


def _add_time_value_mismatch(
    add: Any,
    *,
    goal_id: str,
    field_name: str,
    expected: Any,
    actual: Any,
    code: str,
) -> None:
    expected_text = str(expected or "").strip()
    if not expected_text:
        return
    actual_text = str(actual or "").strip()
    if _canonical_time_value(expected_text) == _canonical_time_value(actual_text):
        return
    add(
        code,
        f"time goal {goal_id!r} has a different executed {field_name}",
        expected=expected_text,
        actual=actual_text,
        rejectedRefIds=["time:%s" % field_name],
        readNext="read the executed query time contract before retrying",
    )


def _time_window_roles_equivalent(
    expected: Any,
    actual: Any,
    *,
    goal_map: Mapping[str, QuestionGoal],
) -> bool:
    """Compare Goal and execution window roles without conflating vocabularies.

    Core may describe the only ordinary window by its query usage (``filter``),
    while the executable time contract describes the same window by position
    (``primary``).  With exactly one declared time window that distinction is
    not semantic.  Once multiple windows exist, roles bind proofs to windows
    and remain exact so comparison/baseline evidence cannot be swapped.
    """

    expected_role = _canonical_time_value(expected)
    actual_role = _canonical_time_value(actual)
    if expected_role == actual_role:
        return True
    time_goal_count = sum(
        isinstance(candidate, TimeWindowQuestionGoal)
        for candidate in goal_map.values()
    )
    if actual_role != "primary":
        return False
    if time_goal_count == 1:
        return True

    # A multi-branch detail request can contain several independent filter
    # windows.  Each branch executes its own sole window as ``primary`` even
    # though Core correctly declares the user-facing role as ``filter``.  Do
    # not apply this alias to comparison/baseline windows: those roles must
    # remain exact so evidence cannot be swapped between windows.
    comparison_markers = {
        "comparison",
        "baseline",
        "previous",
        "prior",
        "secondary",
    }
    time_goals = [
        candidate
        for candidate in goal_map.values()
        if isinstance(candidate, TimeWindowQuestionGoal)
    ]
    return all(
        str(candidate.window_role or "").strip().casefold() not in comparison_markers
        and not str(candidate.window_role or "").strip().casefold().startswith(
            ("comparison_", "baseline_", "previous_", "prior_")
        )
        for candidate in time_goals
    )


def _time_range_kinds_equivalent(expected: Any, actual: Any) -> bool:
    expected_kind = _canonical_ascii_symbol(expected)
    actual_kind = _canonical_ascii_symbol(actual)
    if not expected_kind:
        return True
    aliases = {
        "RELATIVE": "ROLLING",
        "ROLLING": "ROLLING",
        "RELATIVE_WINDOW": "ROLLING",
        "ROLLING_WINDOW": "ROLLING",
    }
    return aliases.get(expected_kind, expected_kind) == aliases.get(
        actual_kind,
        actual_kind,
    )


def _canonical_time_value(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _canonical_analysis_capability(value: Any) -> str:
    return _canonical_ascii_symbol(value)


def _canonical_resolution_status(value: Any) -> str:
    normalized = _canonical_ascii_symbol(getattr(value, "value", value))
    aliases = {
        "PROVED": GoalResolutionStatus.PROVED.value,
        "VERIFIED": GoalResolutionStatus.PROVED.value,
        "INSUFFICIENT": GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value,
        "INSUFFICIENT_EVIDENCE": GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value,
        "EVIDENCE_GAP": GoalResolutionStatus.INSUFFICIENT_EVIDENCE.value,
    }
    status = aliases.get(normalized)
    if status is None:
        raise ValueError(f"unsupported goal resolution {value!r}")
    return status


def _canonical_goal_kind(value: Any) -> str:
    normalized = _canonical_ascii_symbol(getattr(value, "value", value))
    kind = _GOAL_KIND_ALIASES.get(normalized)
    if not kind:
        raise ValueError(f"unsupported goal kind {value!r}")
    return kind


def _canonical_goal_id_list(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("goal ID collections must be lists")
    return list(dict.fromkeys(canonical_goal_id(value) for value in values))


def _normalized_string_list(values: Any) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("string collections must be lists")
    return list(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))


def _canonical_ascii_symbol(value: Any) -> str:
    """Canonicalize a typed enum-like value without a pattern engine."""

    result: list[str] = []
    separator_pending = False
    for character in str(value or "").strip().upper():
        if _is_ascii_letter_or_digit(character):
            if separator_pending and result and result[-1] != "_":
                result.append("_")
            result.append(character)
            separator_pending = False
        elif result:
            separator_pending = True
    return "".join(result)


def _is_ascii_letter_or_digit(character: str) -> bool:
    if len(character) != 1:
        return False
    codepoint = ord(character)
    return (
        ord("a") <= codepoint <= ord("z")
        or ord("A") <= codepoint <= ord("Z")
        or ord("0") <= codepoint <= ord("9")
    )


def _unicode_question_tokens(text: str) -> list[tuple[str, str]]:
    """Tokenize by Unicode character classes, independent of any language."""

    tokens: list[tuple[str, str]] = []
    current_kind = ""
    current_characters: list[str] = []

    def flush() -> None:
        nonlocal current_kind, current_characters
        if current_characters:
            tokens.append((current_kind, "".join(current_characters)))
        current_kind = ""
        current_characters = []

    for character in text:
        if character.isspace():
            flush()
            continue
        category = unicodedata.category(character)
        if category.startswith("N"):
            kind = "NUMBER"
        elif category.startswith("P"):
            kind = "PUNCTUATION"
        elif category.startswith("S"):
            kind = "SYMBOL"
        else:
            kind = "TEXT"
        if current_kind and kind != current_kind:
            flush()
        current_kind = kind
        current_characters.append(character)
    flush()
    return tokens


def _unicode_clause_count(tokens: Sequence[tuple[str, str]]) -> int:
    clauses = 0
    has_content = False
    for kind, _ in tokens:
        if kind == "PUNCTUATION":
            if has_content:
                clauses += 1
                has_content = False
        else:
            has_content = True
    if has_content:
        clauses += 1
    return max(1, clauses)


def _validation_error_path(error: BaseException) -> str:
    if not isinstance(error, ValidationError) or not error.errors():
        return ""
    return ".".join(str(part) for part in error.errors()[0].get("loc", ()))


def _camel_name(value: str) -> str:
    pieces = value.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _object_value(value: Any, *names: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None
