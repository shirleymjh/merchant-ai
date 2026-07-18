from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Sequence, TypeAlias

from pydantic import ConfigDict, Field, ValidationError

from merchant_ai.models import APIModel


class QuestionGoalKind(str, Enum):
    """Closed structural goal kinds; business concepts remain asset-defined."""

    METRIC = "METRIC"
    DIMENSION = "DIMENSION"
    TIME_WINDOW = "TIME_WINDOW"
    COMPARISON = "COMPARISON"
    ENTITY = "ENTITY"
    DEPENDENCY = "DEPENDENCY"


class _StrictGoalModel(APIModel):
    model_config = ConfigDict(extra="forbid")


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
    time_expression: str = ""
    start: str = ""
    end: str = ""
    timezone: str = ""
    granularity: str = ""
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


QuestionGoal: TypeAlias = Annotated[
    MetricQuestionGoal
    | DimensionQuestionGoal
    | TimeWindowQuestionGoal
    | ComparisonQuestionGoal
    | EntityQuestionGoal
    | DependencyQuestionGoal,
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


class QuestionStructuralHints(APIModel):
    """Conservative syntax-only hints; never business-goal inference."""

    time_cues: list[str] = Field(default_factory=list)
    comparison_cues: list[str] = Field(default_factory=list)
    conjunction_cues: list[str] = Field(default_factory=list)
    clause_count: int = 1


class VerifiedArtifactGoalCoverage(_StrictGoalModel):
    """Coverage declaration retained beside one kernel-verified artifact."""

    artifact_id: str
    goal_contract_fingerprint: str
    covered_goal_ids: list[str] = Field(default_factory=list)
    verification_passed: bool = False
    evidence_refs: list[str] = Field(default_factory=list)


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
    missing_required_goal_ids: list[str] = Field(default_factory=list)
    optional_uncovered_goal_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    coverage_by_goal_id: dict[str, list[str]] = Field(default_factory=dict)
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
}

_GOAL_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")
_SPACE_PATTERN = re.compile(r"\s+")
_TIME_CUE_PATTERNS = (
    re.compile(r"\b(?:last|past|previous|next)\s+\d+\s*(?:days?|weeks?|months?|years?)\b", re.IGNORECASE),
    re.compile(r"(?:最近|过去|未来|前|后)\s*\d+\s*(?:天|日|周|星期|个月|月|年)"),
    re.compile(
        r"\b\d{4}-\d{1,2}-\d{1,2}\b\s*(?:to|through|until|[-–—~～至到])\s*\b\d{4}-\d{1,2}-\d{1,2}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:today|yesterday|this\s+(?:week|month|year)|last\s+(?:week|month|year))\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:今天|昨天|本周|上周|本月|上月|今年|去年)"),
)
_COMPARISON_CUE_PATTERNS = (
    re.compile(r"\b(?:vs\.?|versus|compared\s+(?:to|with)|compare|comparison)\b", re.IGNORECASE),
    re.compile(r"(?:同比|环比|对比|比较)"),
)
_CONJUNCTION_CUE_PATTERNS = (
    re.compile(r"\b(?:and|plus|as\s+well\s+as)\b", re.IGNORECASE),
    re.compile(r"(?:以及|并且|同时)"),
)


def canonical_goal_id(value: Any) -> str:
    """Normalize a Core/artifact goal ID into one stable ledger key."""

    normalized = _SPACE_PATTERN.sub("_", str(value or "").strip().lower())
    if not _GOAL_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "goal IDs must be 1-128 ASCII letters/digits with optional '.', '_', ':', or '-' separators"
        )
    return normalized


def inspect_question_structure(question: str) -> QuestionStructuralHints:
    """Return syntax-only cues that can audit a Core-authored contract.

    This intentionally never guesses metric names, dimensions, entities, or
    semantic references. It only recognizes explicit time/comparison syntax
    and a small set of unambiguous conjunction tokens.
    """

    normalized = str(question or "").strip()
    time_cues = _matched_cues(normalized, _TIME_CUE_PATTERNS)
    comparison_cues = _matched_cues(normalized, _COMPARISON_CUE_PATTERNS)
    conjunction_cues = _matched_cues(normalized, _CONJUNCTION_CUE_PATTERNS)
    clauses = [part.strip() for part in re.split(r"[;；。!?！？]+", normalized) if part.strip()]
    return QuestionStructuralHints(
        time_cues=time_cues,
        comparison_cues=comparison_cues,
        conjunction_cues=conjunction_cues,
        clause_count=max(1, len(clauses)),
    )


def validate_original_question_goal_contract(
    payload: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    structural_checks: bool = True,
) -> GoalContractValidationResult:
    """Parse, normalize, and cross-validate one Core-supplied contract."""

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
    if structural_checks:
        issues.extend(_structural_contract_issues(contract))
    return GoalContractValidationResult(
        valid=not any(issue.blocking for issue in issues),
        contract=contract,
        issues=issues,
    )


def parse_original_question_goal_contract(
    payload: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    structural_checks: bool = True,
) -> OriginalQuestionGoalContract:
    result = validate_original_question_goal_contract(payload, structural_checks=structural_checks)
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
) -> VerifiedArtifactGoalCoverage:
    """Build a sidecar declaration only from an already-verified artifact."""

    parsed = parse_original_question_goal_contract(contract)
    artifact_id = str(_object_value(artifact, "artifact_id", "artifactId") or "").strip()
    verified_evidence = _object_value(artifact, "verified_evidence", "verifiedEvidence")
    verification_passed = bool(_object_value(verified_evidence, "passed"))
    if not artifact_id:
        raise ValueError("verified query artifact is missing artifact_id")
    if not verification_passed:
        raise ValueError("goal coverage may only be declared by a verified query artifact")
    return VerifiedArtifactGoalCoverage(
        artifact_id=artifact_id,
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(parsed),
        covered_goal_ids=_canonical_goal_id_list(covered_goal_ids),
        verification_passed=True,
        evidence_refs=_normalized_string_list(evidence_refs),
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
        claimed: set[str] = set()
        coverage_by_goal_id: dict[str, list[str]] = {goal.goal_id: [] for goal in parsed.goals}
        accepted_artifact_ids: list[str] = []
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
            if not declaration.covered_goal_ids:
                issues.append(
                    GoalCoverageIssue(
                        code="ARTIFACT_COVERS_NO_GOALS",
                        message=f"artifact {artifact_id!r} is verified but declares no original-question goal coverage",
                        blocking=False,
                        artifact_id=artifact_id,
                    )
                )
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
                goal = goal_map[goal_id]
                declared_semantic_refs = set(goal.semantic_ref_ids)
                for attribute in (
                    "metric_ref_id",
                    "dimension_ref_id",
                    "entity_ref_id",
                ):
                    semantic_ref_id = str(
                        getattr(goal, attribute, "") or ""
                    ).strip()
                    if semantic_ref_id:
                        declared_semantic_refs.add(semantic_ref_id)
                artifact_evidence_refs = set(declaration.evidence_refs)
                if declared_semantic_refs and not declared_semantic_refs.intersection(
                    artifact_evidence_refs
                ):
                    issues.append(
                        GoalCoverageIssue(
                            code="GOAL_SEMANTIC_EVIDENCE_UNCOVERED",
                            message=(
                                f"artifact {artifact_id!r} claimed goal {goal_id!r} "
                                "without any of its declared semantic evidence refs"
                            ),
                            goal_id=goal_id,
                            artifact_id=artifact_id,
                            details={
                                "declaredSemanticRefIds": sorted(
                                    declared_semantic_refs
                                ),
                                "artifactEvidenceRefs": sorted(
                                    artifact_evidence_refs
                                ),
                            },
                        )
                    )
                    continue
                claimed.add(goal_id)
                coverage_by_goal_id[goal_id].append(artifact_id)

        graph = _goal_dependency_graph(parsed)
        effective_covered = set(claimed)
        while True:
            invalid = {
                goal_id
                for goal_id in effective_covered
                if not graph.get(goal_id, set()).issubset(effective_covered)
            }
            if not invalid:
                break
            effective_covered.difference_update(invalid)

        for goal_id in [goal.goal_id for goal in parsed.goals if goal.goal_id in claimed - effective_covered]:
            missing_dependencies = sorted(graph.get(goal_id, set()) - effective_covered)
            issues.append(
                GoalCoverageIssue(
                    code="COVERED_GOAL_DEPENDENCY_UNCOVERED",
                    message=f"goal {goal_id!r} was claimed covered while prerequisite goals remain uncovered",
                    goal_id=goal_id,
                    details={"missingDependencyGoalIds": missing_dependencies},
                )
            )

        required = required_goal_ids(parsed)
        missing = [goal_id for goal_id in required if goal_id not in effective_covered]
        for goal_id in missing:
            goal = goal_map[goal_id]
            issues.append(
                GoalCoverageIssue(
                    code="REQUIRED_GOAL_UNCOVERED",
                    message=f"required original-question goal {goal.label!r} ({goal_id}) has no effective verified coverage",
                    goal_id=goal_id,
                )
            )

        optional_uncovered = [
            goal.goal_id
            for goal in parsed.goals
            if goal.goal_id not in required and goal.goal_id not in effective_covered
        ]
        blocking = any(issue.blocking for issue in issues)
        return GoalCoverageResult(
            passed=not blocking and not missing,
            finalization_allowed=not blocking and not missing,
            goal_contract_fingerprint=fingerprint,
            required_goal_ids=required,
            claimed_covered_goal_ids=[goal.goal_id for goal in parsed.goals if goal.goal_id in claimed],
            covered_goal_ids=[goal.goal_id for goal in parsed.goals if goal.goal_id in effective_covered],
            missing_required_goal_ids=missing,
            optional_uncovered_goal_ids=optional_uncovered,
            artifact_ids=accepted_artifact_ids,
            coverage_by_goal_id={goal_id: artifact_ids for goal_id, artifact_ids in coverage_by_goal_id.items() if artifact_ids},
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
    )
    string_list_fields = ("source_spans", "semantic_ref_ids")
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
    for goal in contract.goals:
        if not goal.label:
            issues.append(
                GoalContractIssue(
                    code="GOAL_LABEL_MISSING",
                    message=f"goal {goal.goal_id!r} must retain a human-readable label from Core",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, TimeWindowQuestionGoal) and not (
            goal.time_expression or (goal.start and goal.end)
        ):
            issues.append(
                GoalContractIssue(
                    code="TIME_WINDOW_DEFINITION_MISSING",
                    message=f"time-window goal {goal.goal_id!r} requires timeExpression or both start and end",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, ComparisonQuestionGoal) and (
            not goal.left_goal_ids or not goal.right_goal_ids
        ):
            issues.append(
                GoalContractIssue(
                    code="COMPARISON_OPERANDS_MISSING",
                    message=f"comparison goal {goal.goal_id!r} requires non-empty leftGoalIds and rightGoalIds",
                    goal_id=goal.goal_id,
                )
            )
        if isinstance(goal, DependencyQuestionGoal) and (
            not goal.upstream_goal_ids or not goal.downstream_goal_ids
        ):
            issues.append(
                GoalContractIssue(
                    code="DEPENDENCY_ENDPOINTS_MISSING",
                    message=f"dependency goal {goal.goal_id!r} requires upstreamGoalIds and downstreamGoalIds",
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


def _structural_contract_issues(contract: OriginalQuestionGoalContract) -> list[GoalContractIssue]:
    hints = inspect_question_structure(contract.question)
    kinds = {str(goal.kind) for goal in contract.goals}
    issues: list[GoalContractIssue] = []
    if hints.time_cues and QuestionGoalKind.TIME_WINDOW.value not in kinds:
        issues.append(
            GoalContractIssue(
                code="STRUCTURAL_TIME_GOAL_MISSING",
                message="the original question has an explicit time cue but the Core contract has no TIME_WINDOW goal",
                path="question",
            )
        )
    if hints.comparison_cues and QuestionGoalKind.COMPARISON.value not in kinds:
        issues.append(
            GoalContractIssue(
                code="STRUCTURAL_COMPARISON_GOAL_MISSING",
                message="the original question has an explicit comparison cue but the Core contract has no COMPARISON goal",
                path="question",
            )
        )
    required_answer_goals = [
        goal
        for goal in contract.goals
        if goal.required and not isinstance(goal, (TimeWindowQuestionGoal, DependencyQuestionGoal))
    ]
    if hints.conjunction_cues and len(required_answer_goals) < 2:
        issues.append(
            GoalContractIssue(
                code="STRUCTURAL_CONJUNCTION_REVIEW_REQUIRED",
                message="the original question has a conjunction cue but fewer than two required answer goals; Core should review coverage",
                blocking=False,
                path="question",
            )
        )
    return issues


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
    return graph


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
            "verification_passed": _object_value(
                artifact, "verification_passed", "verificationPassed"
            ),
            "evidence_refs": _object_value(
                artifact,
                "evidence_refs",
                "evidenceRefs",
                "goal_coverage_evidence_refs",
                "goalCoverageEvidenceRefs",
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
    ):
        camel_name = _camel_name(snake_name)
        if camel_name in payload and snake_name not in payload:
            payload[snake_name] = payload.pop(camel_name)
    payload["artifact_id"] = str(payload.get("artifact_id") or "").strip()
    payload["goal_contract_fingerprint"] = str(payload.get("goal_contract_fingerprint") or "").strip()
    payload["covered_goal_ids"] = _canonical_goal_id_list(payload.get("covered_goal_ids") or [])
    payload["evidence_refs"] = _normalized_string_list(payload.get("evidence_refs") or [])
    return VerifiedArtifactGoalCoverage.model_validate(payload)


def _canonical_goal_kind(value: Any) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(getattr(value, "value", value) or "").strip().upper()).strip("_")
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


def _matched_cues(text: str, patterns: Sequence[re.Pattern[str]]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(0) for match in pattern.finditer(text))
    return list(dict.fromkeys(matches))


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
