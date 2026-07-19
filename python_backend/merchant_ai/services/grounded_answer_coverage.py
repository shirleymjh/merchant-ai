from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from pydantic import ConfigDict, Field, ValidationError

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import (
    GoalCoverageResult,
    OriginalQuestionGoalContract,
    QuestionGoal,
    canonical_goal_id,
    parse_original_question_goal_contract,
)


class GoalAnswerBinding(APIModel):
    """One visible answer span bound to an already-resolved original goal."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    resolution: str
    answer_text: str
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    insufficiency_ref: str = ""
    renderer: str = ""


class AnswerCoverageIssue(APIModel):
    code: str
    message: str
    blocking: bool = True
    goal_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class AnswerCoverageResult(APIModel):
    passed: bool = False
    answer_fingerprint: str = ""
    source: str = ""
    required_goal_ids: list[str] = Field(default_factory=list)
    mapped_goal_ids: list[str] = Field(default_factory=list)
    missing_goal_ids: list[str] = Field(default_factory=list)
    bindings: list[GoalAnswerBinding] = Field(default_factory=list)
    issues: list[AnswerCoverageIssue] = Field(default_factory=list)


class TrustedAnswerRenderResult(APIModel):
    answer_markdown: str = ""
    bindings: list[GoalAnswerBinding] = Field(default_factory=list)
    appended_goal_ids: list[str] = Field(default_factory=list)


class AnswerCoverageBlocked(RuntimeError):
    def __init__(self, result: AnswerCoverageResult):
        self.result = result
        super().__init__(
            "final answer does not cover every resolved goal: "
            + ", ".join(
                result.missing_goal_ids
                or [issue.code for issue in result.issues if issue.blocking]
            )
        )


_PRIMITIVE_QUERY_GOAL_KINDS = {
    "METRIC",
    "DIMENSION",
    "TIME_WINDOW",
    "ENTITY",
}

_STRICT_RENDERER_BY_GOAL_KIND = {
    "COMPARISON": "VERIFIED_COMPARISON_RENDERER",
    "DEPENDENCY": "VERIFIED_DEPENDENCY_RENDERER",
    "RULE": "VERIFIED_RULE_ARTIFACT_RENDERER",
    "DETAIL": "VERIFIED_DETAIL_RENDERER",
    "RANKING": "VERIFIED_RANKING_RENDERER",
    "ANALYSIS": "VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
}


class AnswerCoverageVerifier:
    """Fail closed unless the visible answer maps every resolved required goal.

    Artifact coverage proves that evidence exists.  This verifier separately
    proves that the final answer actually exposes a span for every goal and
    binds that span to the same verified artifact/evidence reference.
    """

    def verify(
        self,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        goal_coverage: GoalCoverageResult | Mapping[str, Any],
        answer_markdown: str,
        bindings: Sequence[GoalAnswerBinding | Mapping[str, Any]],
        *,
        source: str,
        auto_bind_verified_primitives: bool = False,
    ) -> AnswerCoverageResult:
        parsed = parse_original_question_goal_contract(contract)
        coverage = (
            goal_coverage
            if isinstance(goal_coverage, GoalCoverageResult)
            else GoalCoverageResult.model_validate(goal_coverage)
        )
        answer = str(answer_markdown or "").strip()
        issues: list[AnswerCoverageIssue] = []
        if not answer:
            issues.append(
                AnswerCoverageIssue(
                    code="FINAL_ANSWER_MARKDOWN_REQUIRED",
                    message="final answer markdown is empty",
                )
            )

        parsed_bindings: list[GoalAnswerBinding] = []
        for index, raw in enumerate(bindings):
            try:
                parsed_bindings.append(_parse_binding(raw))
            except (TypeError, ValueError, ValidationError) as exc:
                issues.append(
                    AnswerCoverageIssue(
                        code="ANSWER_GOAL_BINDING_INVALID",
                        message=str(exc),
                        details={"bindingIndex": index},
                    )
                )

        supplied_goal_ids = {item.goal_id for item in parsed_bindings}
        if auto_bind_verified_primitives:
            parsed_bindings.extend(
                _primitive_bindings(
                    parsed,
                    coverage,
                    answer,
                    exclude_goal_ids=supplied_goal_ids,
                )
            )

        goal_map = parsed.goal_map()
        binding_by_goal_id: dict[str, GoalAnswerBinding] = {}
        for binding in parsed_bindings:
            if binding.goal_id in binding_by_goal_id:
                issues.append(
                    AnswerCoverageIssue(
                        code="DUPLICATE_ANSWER_GOAL_BINDING",
                        message=(
                            f"final answer contains more than one binding for "
                            f"goal {binding.goal_id!r}"
                        ),
                        goal_id=binding.goal_id,
                    )
                )
                continue
            binding_by_goal_id[binding.goal_id] = binding

            goal = goal_map.get(binding.goal_id)
            if goal is None:
                issues.append(
                    AnswerCoverageIssue(
                        code="UNKNOWN_ANSWER_GOAL_ID",
                        message=(
                            f"final answer mapped unknown goal ID "
                            f"{binding.goal_id!r}"
                        ),
                        goal_id=binding.goal_id,
                    )
                )
                continue
            issues.extend(
                _binding_issues(
                    goal,
                    binding,
                    coverage,
                    answer,
                    source=source,
                )
            )

        required = list(coverage.required_goal_ids)
        resolved = set(coverage.resolved_goal_ids)
        missing: list[str] = []
        for goal_id in required:
            if goal_id not in resolved:
                issues.append(
                    AnswerCoverageIssue(
                        code="ANSWER_GOAL_NOT_RESOLVED",
                        message=(
                            f"required goal {goal_id!r} was not resolved before "
                            "answer composition"
                        ),
                        goal_id=goal_id,
                    )
                )
                missing.append(goal_id)
            elif goal_id not in binding_by_goal_id:
                issues.append(
                    AnswerCoverageIssue(
                        code="ANSWER_GOAL_BINDING_REQUIRED",
                        message=(
                            f"final answer has no verified payload mapping for "
                            f"goal {goal_id!r}"
                        ),
                        goal_id=goal_id,
                        details={
                            "goalKind": str(getattr(goal_map.get(goal_id), "kind", ""))
                        },
                    )
                )
                missing.append(goal_id)

        blocking = any(issue.blocking for issue in issues)
        return AnswerCoverageResult(
            passed=not blocking and not missing,
            answer_fingerprint=answer_fingerprint(answer),
            source=source,
            required_goal_ids=required,
            mapped_goal_ids=[
                goal.goal_id
                for goal in parsed.goals
                if goal.goal_id in binding_by_goal_id
            ],
            missing_goal_ids=missing,
            bindings=[
                binding_by_goal_id[goal.goal_id]
                for goal in parsed.goals
                if goal.goal_id in binding_by_goal_id
            ],
            issues=issues,
        )

    def require_complete(
        self,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        goal_coverage: GoalCoverageResult | Mapping[str, Any],
        answer_markdown: str,
        bindings: Sequence[GoalAnswerBinding | Mapping[str, Any]],
        *,
        source: str,
        auto_bind_verified_primitives: bool = False,
    ) -> AnswerCoverageResult:
        result = self.verify(
            contract,
            goal_coverage,
            answer_markdown,
            bindings,
            source=source,
            auto_bind_verified_primitives=auto_bind_verified_primitives,
        )
        if not result.passed:
            raise AnswerCoverageBlocked(result)
        return result


def answer_fingerprint(answer_markdown: str) -> str:
    return hashlib.sha256(str(answer_markdown or "").encode("utf-8")).hexdigest()


def answer_attestation_matches(
    answer_markdown: str,
    result: AnswerCoverageResult | Mapping[str, Any] | None,
) -> bool:
    if not result:
        return False
    try:
        parsed = (
            result
            if isinstance(result, AnswerCoverageResult)
            else AnswerCoverageResult.model_validate(result)
        )
    except (TypeError, ValueError, ValidationError):
        return False
    return bool(
        parsed.passed
        and parsed.source
        in {
            "compose_verified_answer",
            "compose_verified_rule_answer",
            "run_skill",
        }
        and parsed.answer_fingerprint
        and parsed.answer_fingerprint == answer_fingerprint(answer_markdown)
    )


def render_verified_query_goal_sections(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    goal_coverage: GoalCoverageResult | Mapping[str, Any],
    answer_markdown: str,
    query_artifacts: Sequence[Any],
) -> TrustedAnswerRenderResult:
    """Derive strict bindings only from real query artifacts and their rows.

    DETAIL/RANKING/ranked-COMPARISON/DEPENDENCY sections are detected in the
    composed answer from actual row values.  If absent, a deterministic table
    is appended from the immutable artifact.  No model-supplied renderer name,
    span or provenance is accepted by the runtime path.
    """

    parsed = parse_original_question_goal_contract(contract)
    coverage = (
        goal_coverage
        if isinstance(goal_coverage, GoalCoverageResult)
        else GoalCoverageResult.model_validate(goal_coverage)
    )
    artifacts_by_id = {
        str(getattr(item, "artifact_id", "") or ""): item
        for item in query_artifacts
        if str(getattr(item, "artifact_id", "") or "")
    }
    answer = str(answer_markdown or "").strip()
    bindings: list[GoalAnswerBinding] = []
    appended: list[str] = []
    for goal in parsed.goals:
        if goal.goal_id not in coverage.required_goal_ids:
            continue
        resolution = coverage.resolution_by_goal_id.get(goal.goal_id)
        if resolution == "INSUFFICIENT_EVIDENCE":
            refs = list(
                coverage.resolution_evidence_refs_by_goal_id.get(
                    goal.goal_id, []
                )
            )
            reason = str(
                coverage.insufficiency_reason_by_goal_id.get(goal.goal_id, "")
                or "现有已验证证据不足，无法得出可靠结论。"
            ).strip()
            if not refs:
                continue
            fragment = f"### {goal.label}\n\n证据不足：{reason}"
            if fragment not in answer:
                answer = "\n\n".join(item for item in (answer, fragment) if item)
                appended.append(goal.goal_id)
            bindings.append(
                GoalAnswerBinding(
                    goal_id=goal.goal_id,
                    resolution="INSUFFICIENT_EVIDENCE",
                    answer_text=fragment,
                    evidence_refs=[refs[0]],
                    insufficiency_ref=refs[0],
                    renderer="VERIFIED_INSUFFICIENCY_RENDERER",
                )
            )
            continue
        if resolution != "PROVED":
            continue
        renderer = _STRICT_RENDERER_BY_GOAL_KIND.get(goal.kind, "")
        if goal.kind not in {"DETAIL", "RANKING", "COMPARISON", "DEPENDENCY"}:
            continue
        allowed_ids = coverage.resolution_artifact_ids_by_goal_id.get(
            goal.goal_id, []
        )
        artifact = next(
            (
                artifacts_by_id[artifact_id]
                for artifact_id in allowed_ids
                if artifact_id in artifacts_by_id
                and _artifact_can_render_goal(goal, artifacts_by_id[artifact_id])
            ),
            None,
        )
        if artifact is None:
            continue
        span = _artifact_answer_span(answer, artifact, goal_kind=goal.kind)
        if not span:
            fragment = _render_artifact_fragment(goal, artifact)
            if not fragment:
                continue
            answer = "\n\n".join(item for item in (answer, fragment) if item)
            span = fragment
            appended.append(goal.goal_id)
        artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        allowed_refs = set(
            coverage.resolution_evidence_refs_by_goal_id.get(goal.goal_id, [])
        )
        artifact_refs = set(
            getattr(getattr(artifact, "contract", None), "evidence_refs", [])
            or []
        )
        bindings.append(
            GoalAnswerBinding(
                goal_id=goal.goal_id,
                resolution="PROVED",
                answer_text=span,
                artifact_ids=[artifact_id],
                evidence_refs=sorted(allowed_refs.intersection(artifact_refs)),
                renderer=renderer,
            )
        )
    return TrustedAnswerRenderResult(
        answer_markdown=answer,
        bindings=bindings,
        appended_goal_ids=appended,
    )


def render_verified_rule_goal_bindings(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    goal_coverage: GoalCoverageResult | Mapping[str, Any],
    answer_markdown: str,
) -> list[GoalAnswerBinding]:
    parsed = parse_original_question_goal_contract(contract)
    coverage = (
        goal_coverage
        if isinstance(goal_coverage, GoalCoverageResult)
        else GoalCoverageResult.model_validate(goal_coverage)
    )
    answer = str(answer_markdown or "").strip()
    return [
        GoalAnswerBinding(
            goal_id=goal.goal_id,
            resolution="PROVED",
            answer_text=answer,
            artifact_ids=list(
                coverage.resolution_artifact_ids_by_goal_id.get(goal.goal_id, [])
            ),
            evidence_refs=list(
                coverage.resolution_evidence_refs_by_goal_id.get(goal.goal_id, [])
            ),
            renderer="VERIFIED_RULE_ARTIFACT_RENDERER",
        )
        for goal in parsed.goals
        if goal.goal_id in coverage.required_goal_ids and goal.kind == "RULE"
    ]
def _parse_binding(raw: GoalAnswerBinding | Mapping[str, Any]) -> GoalAnswerBinding:
    if isinstance(raw, GoalAnswerBinding):
        payload = raw.model_dump(by_alias=False)
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise TypeError("answer goal bindings must be objects")
    aliases = {
        "goalId": "goal_id",
        "answerText": "answer_text",
        "artifactIds": "artifact_ids",
        "evidenceRefs": "evidence_refs",
        "insufficiencyRef": "insufficiency_ref",
    }
    for alias, field_name in aliases.items():
        if alias in payload and field_name not in payload:
            payload[field_name] = payload.pop(alias)
    payload["goal_id"] = canonical_goal_id(payload.get("goal_id"))
    payload["resolution"] = _canonical_resolution(payload.get("resolution"))
    for field_name in ("artifact_ids", "evidence_refs"):
        values = payload.get(field_name) or []
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{field_name} must be a list")
        payload[field_name] = _dedupe_strings(values)
    for field_name in ("answer_text", "insufficiency_ref", "renderer"):
        payload[field_name] = str(payload.get(field_name) or "").strip()
    payload["renderer"] = payload["renderer"].upper()
    return GoalAnswerBinding.model_validate(payload)


def _primitive_bindings(
    contract: OriginalQuestionGoalContract,
    coverage: GoalCoverageResult,
    answer_markdown: str,
    *,
    exclude_goal_ids: set[str],
) -> list[GoalAnswerBinding]:
    result: list[GoalAnswerBinding] = []
    for goal in contract.goals:
        if (
            goal.goal_id in exclude_goal_ids
            or goal.goal_id not in coverage.required_goal_ids
            or goal.kind not in _PRIMITIVE_QUERY_GOAL_KINDS
            or coverage.resolution_by_goal_id.get(goal.goal_id) != "PROVED"
        ):
            continue
        result.append(
            GoalAnswerBinding(
                goal_id=goal.goal_id,
                resolution="PROVED",
                answer_text=answer_markdown,
                artifact_ids=list(
                    coverage.resolution_artifact_ids_by_goal_id.get(
                        goal.goal_id, []
                    )
                ),
                evidence_refs=list(
                    coverage.resolution_evidence_refs_by_goal_id.get(
                        goal.goal_id, []
                    )
                ),
                renderer="VERIFIED_QUERY_RENDERER",
            )
        )
    return result


def _binding_issues(
    goal: QuestionGoal,
    binding: GoalAnswerBinding,
    coverage: GoalCoverageResult,
    answer_markdown: str,
    *,
    source: str,
) -> list[AnswerCoverageIssue]:
    issues: list[AnswerCoverageIssue] = []

    def add(code: str, message: str, **details: Any) -> None:
        issues.append(
            AnswerCoverageIssue(
                code=code,
                message=message,
                goal_id=goal.goal_id,
                details=details,
            )
        )

    expected_resolution = coverage.resolution_by_goal_id.get(goal.goal_id, "")
    if binding.resolution != expected_resolution:
        add(
            "ANSWER_GOAL_RESOLUTION_MISMATCH",
            (
                f"answer binding for {goal.goal_id!r} says "
                f"{binding.resolution or 'UNKNOWN'}, expected "
                f"{expected_resolution or 'UNRESOLVED'}"
            ),
        )

    if not binding.answer_text:
        add(
            "ANSWER_GOAL_TEXT_REQUIRED",
            f"answer binding for {goal.goal_id!r} has no visible answer span",
        )
    elif _normalized_text(binding.answer_text) not in _normalized_text(
        answer_markdown
    ):
        add(
            "ANSWER_GOAL_TEXT_NOT_RENDERED",
            (
                f"the declared answer span for {goal.goal_id!r} is absent "
                "from final answerMarkdown"
            ),
            answerText=binding.answer_text,
        )

    expected_renderer = _expected_goal_renderer(
        goal,
        coverage,
        expected_resolution,
        source=source,
    )
    if expected_renderer and binding.renderer != expected_renderer:
        add(
            "ANSWER_GOAL_RENDERER_REQUIRED",
            (
                f"{goal.kind} goal {goal.goal_id!r} requires "
                f"{expected_renderer}, not ordinary answer text"
            ),
            expectedRenderer=expected_renderer,
            actualRenderer=binding.renderer,
        )
    if goal.kind == "RULE" and source != "compose_verified_rule_answer":
        add(
            "RULE_ANSWER_RENDERER_BOUNDARY_VIOLATION",
            "RULE goals may only be finalized by compose_verified_rule_answer",
        )
    proof_types = {
        str(item or "").strip().upper()
        for item in coverage.resolution_proof_types_by_goal_id.get(
            goal.goal_id,
            [],
        )
    }
    derived_analysis_proof = "DETERMINISTIC_DERIVED_ANALYSIS" in proof_types
    trusted_analysis_source = source in {
        "trusted_analysis_renderer",
        "run_skill",
    }
    if (
        derived_analysis_proof
        and goal.kind in {"ANALYSIS", "COMPARISON"}
        and not trusted_analysis_source
    ):
        add(
            "DERIVED_ANALYSIS_RENDERER_SOURCE_REQUIRED",
            (
                "derived ANALYSIS/COMPARISON conclusions may only be bound "
                "by the trusted analysis artifact renderer"
            ),
        )
    if (
        binding.renderer == "VERIFIED_ANALYSIS_ARTIFACT_RENDERER"
        and not derived_analysis_proof
    ):
        add(
            "DERIVED_ANALYSIS_PROOF_REQUIRED",
            (
                "trusted analysis renderer source requires a "
                "DETERMINISTIC_DERIVED_ANALYSIS goal proof"
            ),
        )

    allowed_artifacts = set(
        coverage.resolution_artifact_ids_by_goal_id.get(goal.goal_id, [])
    )
    allowed_refs = set(
        coverage.resolution_evidence_refs_by_goal_id.get(goal.goal_id, [])
    )
    supplied_artifacts = set(binding.artifact_ids)
    supplied_refs = set(binding.evidence_refs)
    unknown_artifacts = sorted(supplied_artifacts - allowed_artifacts)
    unknown_refs = sorted(supplied_refs - allowed_refs)
    if unknown_artifacts:
        add(
            "ANSWER_GOAL_ARTIFACT_REF_UNVERIFIED",
            f"answer binding for {goal.goal_id!r} references unverified artifacts",
            unverifiedArtifactIds=unknown_artifacts,
        )
    if unknown_refs:
        add(
            "ANSWER_GOAL_EVIDENCE_REF_UNVERIFIED",
            f"answer binding for {goal.goal_id!r} references unverified evidence",
            unverifiedEvidenceRefs=unknown_refs,
        )

    if expected_resolution == "INSUFFICIENT_EVIDENCE":
        if not binding.insufficiency_ref:
            add(
                "ANSWER_GOAL_INSUFFICIENCY_REF_REQUIRED",
                (
                    f"insufficient-evidence answer for {goal.goal_id!r} "
                    "requires a typed gap reference"
                ),
            )
        elif binding.insufficiency_ref not in allowed_refs:
            add(
                "ANSWER_GOAL_INSUFFICIENCY_REF_UNVERIFIED",
                (
                    f"insufficiency ref for {goal.goal_id!r} is not in "
                    "the verified goal resolution"
                ),
                insufficiencyRef=binding.insufficiency_ref,
            )
        if binding.artifact_ids:
            add(
                "INSUFFICIENT_ANSWER_CANNOT_CLAIM_PROOF_ARTIFACT",
                (
                    f"insufficient-evidence answer for {goal.goal_id!r} "
                    "cannot present artifacts as proof"
                ),
            )
    elif not (
        supplied_artifacts.intersection(allowed_artifacts)
        or supplied_refs.intersection(allowed_refs)
    ):
        add(
            "ANSWER_GOAL_VERIFIED_REF_REQUIRED",
            (
                f"answer binding for proved goal {goal.goal_id!r} must map "
                "to its verified artifact or evidence ref"
            ),
        )
    return issues


def _expected_goal_renderer(
    goal: QuestionGoal,
    coverage: GoalCoverageResult,
    expected_resolution: str,
    *,
    source: str,
) -> str:
    proof_types = {
        str(item or "").strip().upper()
        for item in coverage.resolution_proof_types_by_goal_id.get(
            goal.goal_id,
            [],
        )
    }
    if (
        "DETERMINISTIC_DERIVED_ANALYSIS" in proof_types
        and source in {"trusted_analysis_renderer", "run_skill"}
    ):
        return "VERIFIED_ANALYSIS_ARTIFACT_RENDERER"
    if expected_resolution == "INSUFFICIENT_EVIDENCE":
        return "VERIFIED_INSUFFICIENCY_RENDERER"
    return _STRICT_RENDERER_BY_GOAL_KIND.get(goal.kind, "")


def _artifact_can_render_goal(goal: QuestionGoal, artifact: Any) -> bool:
    if goal.kind != "COMPARISON":
        return True
    comparison_type = re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(getattr(goal, "comparison_type", "") or "").upper(),
    ).strip("_")
    if comparison_type.startswith("ANOMAL"):
        return False
    contract = getattr(artifact, "contract", None)
    ranking = getattr(contract, "ranking", None)
    return bool(
        str(getattr(contract, "query_shape", "") or "").upper() == "RANKED"
        and getattr(ranking, "enabled", False)
    )


def _artifact_answer_span(
    answer_markdown: str,
    artifact: Any,
    *,
    goal_kind: str,
) -> str:
    answer = str(answer_markdown or "").strip()
    rows = _artifact_rows(artifact)
    if not rows:
        no_data_markers = (
            "未返回数据",
            "暂无数据",
            "没有数据",
            "无符合",
            "no data",
            "empty result",
        )
        normalized = answer.lower()
        return answer if any(marker in normalized for marker in no_data_markers) else ""

    first_row = rows[0]
    values = [
        value
        for value in first_row.values()
        if value is not None and str(value).strip() not in {"", "-"}
    ]
    matched = [value for value in values if _value_is_rendered(answer, value)]
    minimum = 1 if len(values) <= 1 or goal_kind == "DEPENDENCY" else 2
    return answer if len(matched) >= min(minimum, len(values)) else ""


def _render_artifact_fragment(goal: QuestionGoal, artifact: Any) -> str:
    rows = _artifact_rows(artifact)
    heading = f"### {str(getattr(goal, 'label', '') or goal.goal_id).strip()}"
    bundle = getattr(getattr(artifact, "run_result", None), "merged_query_bundle", None)
    offloaded_files = list(getattr(bundle, "offloaded_files", None) or [])
    if not rows:
        if offloaded_files:
            rendered_files = "\n".join(
                f"- `{_markdown_cell(item)}`" for item in offloaded_files
            )
            return f"{heading}\n\n明细已验证并写入结果文件：\n{rendered_files}"
        return f"{heading}\n\n已验证查询未返回数据。"

    output_columns = list(getattr(artifact, "output_columns", None) or [])
    columns = [
        column
        for column in output_columns
        if any(column in row for row in rows)
    ]
    if not columns:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        return ""
    header = "| " + " | ".join(_markdown_cell(item) for item in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_markdown_cell(row.get(column)) for column in columns)
        + " |"
        for row in rows[:20]
    ]
    suffix = "\n\n仅展示前 20 行。" if len(rows) > 20 else ""
    return "\n".join([heading, "", header, divider, *body]) + suffix


def _artifact_rows(artifact: Any) -> list[dict[str, Any]]:
    bundle = getattr(getattr(artifact, "run_result", None), "merged_query_bundle", None)
    return [
        dict(row)
        for row in (getattr(bundle, "rows", None) or [])
        if isinstance(row, Mapping)
    ]


def _value_is_rendered(answer_markdown: str, value: Any) -> bool:
    answer = str(answer_markdown or "")
    text = str(value).strip()
    if not text:
        return False
    if text in answer:
        return True
    compact_answer = re.sub(r"[,_\s]", "", answer).lower()
    compact_value = re.sub(r"[,_\s]", "", text).lower()
    if compact_value and compact_value in compact_answer:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    numeric_variants = {
        str(number),
        f"{number:g}",
        f"{number:,.2f}",
        f"{number:,.0f}",
    }
    return any(item in answer for item in numeric_variants)


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace(
        "\n", " "
    )


def _canonical_resolution(value: Any) -> str:
    normalized = re.sub(
        r"[^A-Z0-9]+", "_", str(value or "").strip().upper()
    ).strip("_")
    aliases = {
        "PROVED": "PROVED",
        "VERIFIED": "PROVED",
        "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        "EVIDENCE_GAP": "INSUFFICIENT_EVIDENCE",
    }
    status = aliases.get(normalized)
    if status is None:
        raise ValueError(f"unsupported answer goal resolution {value!r}")
    return status


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
