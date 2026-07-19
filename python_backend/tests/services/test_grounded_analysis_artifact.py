from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from merchant_ai.services.grounded_analysis_artifact import (
    GroundedRunSkillAnalysisPublicationRequest,
    build_grounded_analysis_skill_input,
    grounded_analysis_goal_coverage,
    grounded_analysis_run_skill_publication_schema,
    publish_grounded_analysis_from_skill,
    render_grounded_analysis_artifact,
    verify_grounded_analysis_data_input_coverage,
)
from merchant_ai.services.grounded_answer_coverage import AnswerCoverageVerifier
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    ComparisonQuestionGoal,
    GoalCoverageVerifier,
    GoalCoverageResult,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    VerifiedArtifactGoalCoverage,
    original_question_goal_contract_fingerprint,
)


def contract(
    analysis_type: str,
    *,
    input_goal_ids: list[str],
    baseline_goal_ids: list[str] | None = None,
    question: str = "opaque question",
) -> OriginalQuestionGoalContract:
    metric_ids = list(dict.fromkeys([*input_goal_ids, *(baseline_goal_ids or [])]))
    return OriginalQuestionGoalContract(
        question=question,
        goals=[
            *[MetricQuestionGoal(goal_id=goal_id, label=goal_id) for goal_id in metric_ids],
            AnalysisQuestionGoal(
                goal_id="analysis.main",
                label="analysis",
                analysis_type=analysis_type,
                input_goal_ids=input_goal_ids,
                baseline_goal_ids=baseline_goal_ids or [],
            ),
        ],
    )


def query_artifact(
    artifact_id: str,
    rows: list[dict[str, object]],
    lineage: dict[str, list[str]],
    *,
    passed: bool = True,
) -> SimpleNamespace:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    return SimpleNamespace(
        artifact_id=artifact_id,
        verified_evidence=SimpleNamespace(passed=passed),
        output_columns=columns,
        output_lineage=lineage,
        run_result=SimpleNamespace(merged_query_bundle=SimpleNamespace(rows=rows)),
    )


def query_coverage(
    goal_contract: OriginalQuestionGoalContract,
    covered_goal_ids: list[str],
    *,
    coverage_by_goal_id: dict[str, list[str]] | None = None,
) -> GoalCoverageResult:
    return GoalCoverageResult(
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(goal_contract),
        covered_goal_ids=covered_goal_ids,
        resolved_goal_ids=covered_goal_ids,
        coverage_by_goal_id=coverage_by_goal_id or {goal_id: ["query-1"] for goal_id in covered_goal_ids},
    )


def test_data_input_gate_defers_typed_analysis_but_requires_proved_inputs() -> None:
    goal_contract = contract(
        "correlation",
        input_goal_ids=["metric.x", "metric.y"],
    )
    ready = verify_grounded_analysis_data_input_coverage(
        goal_contract=goal_contract,
        query_goal_coverage=query_coverage(
            goal_contract,
            ["metric.x", "metric.y"],
            coverage_by_goal_id={
                "metric.x": ["query-x"],
                "metric.y": ["query-y"],
            },
        ),
    )

    assert ready.skill_start_allowed is True
    assert ready.deferred_goal_ids == ["analysis.main"]
    assert ready.deferred_input_goal_ids_by_goal_id == {"analysis.main": ["metric.x", "metric.y"]}
    assert ready.verified_input_artifact_ids == ["query-x", "query-y"]

    blocked = verify_grounded_analysis_data_input_coverage(
        goal_contract=goal_contract,
        query_goal_coverage=query_coverage(goal_contract, ["metric.x"]),
    )
    assert blocked.skill_start_allowed is False
    assert blocked.missing_proved_input_goal_ids == ["metric.y"]
    assert "DERIVED_GOAL_INPUT_NOT_PROVED" in {item.code for item in blocked.issues}


def test_data_input_gate_defers_only_typed_anomaly_or_correlation_comparison() -> None:
    anomaly_contract = OriginalQuestionGoalContract(
        question="label intentionally opaque",
        goals=[
            MetricQuestionGoal(goal_id="metric.x", label="x"),
            MetricQuestionGoal(goal_id="metric.y", label="y"),
            ComparisonQuestionGoal(
                goal_id="comparison.main",
                label="not inspected",
                comparison_type="anomaly",
                left_goal_ids=["metric.x"],
                right_goal_ids=["metric.y"],
            ),
        ],
    )
    ready = verify_grounded_analysis_data_input_coverage(
        goal_contract=anomaly_contract,
        query_goal_coverage=query_coverage(
            anomaly_contract,
            ["metric.x", "metric.y"],
        ),
    )
    assert ready.skill_start_allowed is True
    assert ready.deferred_goal_ids == ["comparison.main"]

    ranking_contract = OriginalQuestionGoalContract(
        question="question says anomaly but typed comparison is ranking",
        goals=[
            MetricQuestionGoal(goal_id="metric.x", label="异常异常异常"),
            MetricQuestionGoal(goal_id="metric.y", label="y"),
            ComparisonQuestionGoal(
                goal_id="comparison.rank",
                label="最异常",
                comparison_type="rank_desc_top_3",
                left_goal_ids=["metric.x"],
                right_goal_ids=["metric.y"],
            ),
        ],
    )
    blocked = verify_grounded_analysis_data_input_coverage(
        goal_contract=ranking_contract,
        query_goal_coverage=query_coverage(
            ranking_contract,
            ["metric.x", "metric.y"],
        ),
    )
    assert blocked.skill_start_allowed is False
    assert blocked.deferred_goal_ids == []
    assert "DATA_INPUT_GATE_NO_TYPED_DERIVED_GOAL" in {item.code for item in blocked.issues}


def test_data_input_gate_rejects_query_claim_over_deferred_goal() -> None:
    goal_contract = contract("trend", input_goal_ids=["metric.x"])
    result = verify_grounded_analysis_data_input_coverage(
        goal_contract=goal_contract,
        query_goal_coverage=query_coverage(
            goal_contract,
            ["metric.x", "analysis.main"],
        ),
    )

    assert result.skill_start_allowed is False
    assert "QUERY_ARTIFACT_CANNOT_PROVE_DERIVED_GOAL" in {item.code for item in result.issues}


def test_typed_correlation_comparison_closes_only_after_derived_artifact() -> None:
    goal_contract = OriginalQuestionGoalContract(
        question="x 与 y 是否相关",
        goals=[
            MetricQuestionGoal(goal_id="metric.x", label="x"),
            MetricQuestionGoal(goal_id="metric.y", label="y"),
            ComparisonQuestionGoal(
                goal_id="comparison.correlation",
                label="相关性",
                comparison_type="correlation",
                left_goal_ids=["metric.x"],
                right_goal_ids=["metric.y"],
            ),
        ],
    )
    source = query_artifact(
        "query-series",
        [{"pt": day, "x": day, "y": day * 2} for day in range(1, 7)],
        {"pt": ["field:date"], "x": ["metric:x"], "y": ["metric:y"]},
    )
    derived = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "comparison.correlation",
            "inputArtifactIds": ["query-series"],
            "seriesBindings": [
                {"seriesId": "x", "artifactId": "query-series", "valueColumn": "x"},
                {"seriesId": "y", "artifactId": "query-series", "valueColumn": "y"},
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[source],
        artifact_goal_ids={"query-series": ["metric.x", "metric.y"]},
    )

    assert derived.goal_kind == "COMPARISON"
    assert derived.goal_resolutions[0].goal_kind == "COMPARISON"
    primitive_coverage = VerifiedArtifactGoalCoverage(
        artifact_id="query-series",
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(goal_contract),
        covered_goal_ids=["metric.x", "metric.y"],
        verification_passed=True,
        evidence_refs=["query:query-series:rows"],
        goal_resolutions=[
            {
                "goalId": "metric.x",
                "goalKind": "METRIC",
                "resolution": "PROVED",
                "proofType": "QUERY_VALUE",
                "valueRefs": ["query:query-series:x"],
            },
            {
                "goalId": "metric.y",
                "goalKind": "METRIC",
                "resolution": "PROVED",
                "proofType": "QUERY_VALUE",
                "valueRefs": ["query:query-series:y"],
            },
        ],
    )
    final_coverage = GoalCoverageVerifier().verify(
        goal_contract,
        [
            primitive_coverage,
            grounded_analysis_goal_coverage(goal_contract, derived),
        ],
    )

    assert final_coverage.finalization_allowed is True
    assert final_coverage.covered_goal_ids == [
        "metric.x",
        "metric.y",
        "comparison.correlation",
    ]
    rendered = render_grounded_analysis_artifact(derived)
    visible = AnswerCoverageVerifier().verify(
        goal_contract,
        final_coverage,
        rendered.answer_markdown,
        [rendered.binding],
        source="trusted_analysis_renderer",
        auto_bind_verified_primitives=True,
    )
    assert visible.passed is True
    assert rendered.binding.renderer == "VERIFIED_ANALYSIS_ARTIFACT_RENDERER"
    assert "相关不等于因果" in rendered.answer_markdown


def test_analysis_type_comes_only_from_typed_analysis_goal() -> None:
    goal_contract = contract(
        "correlation",
        input_goal_ids=["metric.x", "metric.y"],
        question="这个问题反复说异常、趋势、为什么，但 Goal 明确要求相关性",
    )
    artifact = query_artifact(
        "query-1",
        [{"pt": index, "x": index, "y": index * 2} for index in range(1, 6)],
        {"pt": ["field:date"], "x": ["metric:x"], "y": ["metric:y"]},
    )

    skill_input = build_grounded_analysis_skill_input(
        goal_contract=goal_contract,
        analysis_goal_id="analysis.main",
        requested_artifact_ids=["query-1"],
        verified_query_artifacts=[artifact],
        artifact_goal_ids={"query-1": ["metric.x", "metric.y"]},
    )

    assert skill_input.analysis_type == "CORRELATION"
    schema = grounded_analysis_run_skill_publication_schema()
    assert "analysisType" not in schema["properties"]
    with pytest.raises(ValidationError):
        GroundedRunSkillAnalysisPublicationRequest.model_validate(
            {
                "analysisGoalId": "analysis.main",
                "analysisType": "anomaly",
                "inputArtifactIds": ["query-1"],
                "seriesBindings": [
                    {
                        "seriesId": "x",
                        "artifactId": "query-1",
                        "valueColumn": "x",
                    }
                ],
            }
        )


def test_skill_input_contains_only_verified_rows_and_lineage() -> None:
    goal_contract = contract("trend", input_goal_ids=["metric.x"])
    artifact = query_artifact(
        "query-1",
        [{"pt": "2026-01-01", "x": 3, "__private": "no"}],
        {"pt": ["field:date"], "x": ["metric:x"]},
    )

    mounted = build_grounded_analysis_skill_input(
        goal_contract=goal_contract,
        analysis_goal_id="analysis.main",
        requested_artifact_ids=["query-1"],
        verified_query_artifacts=[artifact],
        artifact_goal_ids={"query-1": ["metric.x"]},
    ).model_dump(by_alias=True)

    verified_input = mounted["verifiedInputs"][0]
    assert set(verified_input) == {
        "artifactId",
        "goalIds",
        "rowRef",
        "rowsHash",
        "rows",
        "outputColumns",
        "outputLineage",
    }
    assert "__private" not in verified_input["rows"][0]
    serialized = str(mounted)
    assert "sql" not in serialized.lower()
    assert "plan" not in serialized.lower()


def test_unverified_or_unknown_query_artifacts_are_rejected() -> None:
    goal_contract = contract("trend", input_goal_ids=["metric.x"])
    failed = query_artifact(
        "query-failed",
        [{"pt": 1, "x": 1}],
        {"pt": ["field:date"], "x": ["metric:x"]},
        passed=False,
    )

    with pytest.raises(ValueError) as unverified_error:
        build_grounded_analysis_skill_input(
            goal_contract=goal_contract,
            analysis_goal_id="analysis.main",
            requested_artifact_ids=["query-failed"],
            verified_query_artifacts=[failed],
            artifact_goal_ids={"query-failed": ["metric.x"]},
        )
    assert "not verified" in str(unverified_error.value)

    with pytest.raises(ValueError) as unknown_artifact_error:
        build_grounded_analysis_skill_input(
            goal_contract=goal_contract,
            analysis_goal_id="analysis.main",
            requested_artifact_ids=["missing"],
            verified_query_artifacts=[],
            artifact_goal_ids={},
        )
    assert "not in the verified ledger" in str(unknown_artifact_error.value)


@pytest.mark.parametrize(
    ("method", "normalization", "baseline_pairs", "expected_gap"),
    [
        ("", "PERCENT_CHANGE", [{"currentSeriesId": "now", "baselineSeriesId": "before"}], "ANALYSIS_METHOD_REQUIRED"),
        (
            "BASELINE_NORMALIZED_DEVIATION",
            "",
            [{"currentSeriesId": "now", "baselineSeriesId": "before"}],
            "ANOMALY_NORMALIZATION_REQUIRED",
        ),
        ("BASELINE_NORMALIZED_DEVIATION", "PERCENT_CHANGE", [], "ANOMALY_COMPARABLE_BASELINE_REQUIRED"),
    ],
)
def test_anomaly_missing_method_normalization_or_baseline_is_insufficient(
    method: str,
    normalization: str,
    baseline_pairs: list[dict[str, str]],
    expected_gap: str,
) -> None:
    goal_contract = contract(
        "anomaly",
        input_goal_ids=["metric.current"],
        baseline_goal_ids=["metric.baseline"],
    )
    current = query_artifact(
        "current",
        [{"value": 120}],
        {"value": ["metric:same"]},
    )
    baseline = query_artifact(
        "baseline",
        [{"value": 100}],
        {"value": ["metric:same"]},
    )

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["current", "baseline"],
            "seriesBindings": [
                {"seriesId": "now", "artifactId": "current", "valueColumn": "value"},
                {"seriesId": "before", "artifactId": "baseline", "valueColumn": "value"},
            ],
            "method": method,
            "normalizationMethod": normalization,
            "baselinePairs": baseline_pairs,
            "anomalyThreshold": 10,
        },
        verified_query_artifacts=[current, baseline],
        artifact_goal_ids={
            "current": ["metric.current"],
            "baseline": ["metric.baseline"],
        },
    )

    assert artifact.publication_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.result == {}
    assert artifact.goal_resolutions[0].resolution == "INSUFFICIENT_EVIDENCE"
    assert expected_gap in {item.code for item in artifact.verified_evidence.gaps}
    rendered = render_grounded_analysis_artifact(artifact)
    assert rendered.binding.resolution == "INSUFFICIENT_EVIDENCE"
    assert rendered.binding.artifact_ids == []
    assert rendered.binding.insufficiency_ref
    assert rendered.binding.renderer == "VERIFIED_ANALYSIS_ARTIFACT_RENDERER"
    assert "证据不足" in rendered.answer_markdown
    visible = AnswerCoverageVerifier().verify(
        goal_contract,
        GoalCoverageResult(
            required_goal_ids=["analysis.main"],
            resolved_goal_ids=["analysis.main"],
            insufficient_evidence_goal_ids=["analysis.main"],
            resolution_by_goal_id={"analysis.main": "INSUFFICIENT_EVIDENCE"},
            resolution_proof_types_by_goal_id={"analysis.main": ["DETERMINISTIC_DERIVED_ANALYSIS"]},
            resolution_evidence_refs_by_goal_id={"analysis.main": [rendered.binding.insufficiency_ref]},
            insufficiency_reason_by_goal_id={"analysis.main": artifact.goal_resolutions[0].reason},
        ),
        rendered.answer_markdown,
        [rendered.binding],
        source="run_skill",
    )
    assert visible.passed is True


def test_anomaly_uses_comparable_baseline_normalization_method_and_result() -> None:
    goal_contract = contract(
        "anomaly",
        input_goal_ids=["metric.current"],
        baseline_goal_ids=["metric.baseline"],
    )
    current = query_artifact(
        "current",
        [{"value": 125}],
        {"value": ["metric:same"]},
    )
    baseline = query_artifact(
        "baseline",
        [{"value": 100}],
        {"value": ["metric:same"]},
    )
    request = {
        "analysisGoalId": "analysis.main",
        "inputArtifactIds": ["current", "baseline"],
        "seriesBindings": [
            {"seriesId": "now", "artifactId": "current", "valueColumn": "value"},
            {"seriesId": "before", "artifactId": "baseline", "valueColumn": "value"},
        ],
        "method": "BASELINE_NORMALIZED_DEVIATION",
        "normalizationMethod": "PERCENT_CHANGE",
        "baselinePairs": [{"currentSeriesId": "now", "baselineSeriesId": "before"}],
        "anomalyThreshold": 20,
    }

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request=request,
        verified_query_artifacts=[current, baseline],
        artifact_goal_ids={
            "current": ["metric.current"],
            "baseline": ["metric.baseline"],
        },
    )

    assert artifact.publication_status == "PROVED"
    assert artifact.normalization_method == "PERCENT_CHANGE"
    assert artifact.result["seriesScores"][0]["normalizedScore"] == "25"
    assert artifact.result["seriesScores"][0]["exceedsThreshold"] is True
    resolution = artifact.goal_resolutions[0]
    assert resolution.resolution == "PROVED"
    assert resolution.analysis_method == "BASELINE_NORMALIZED_DEVIATION"
    assert resolution.baseline_refs
    assert resolution.result_ref == artifact.result_ref
    rendered = render_grounded_analysis_artifact(artifact)
    assert "最异常序列为 now" in rendered.answer_markdown
    assert "标准化分数为 25" in rendered.answer_markdown
    assert "阈值为 20" in rendered.answer_markdown
    assert rendered.binding.artifact_ids == [artifact.artifact_id]
    assert rendered.binding.evidence_refs == [artifact.result_ref]


def test_anomaly_rejects_semantically_incomparable_baseline() -> None:
    goal_contract = contract(
        "anomaly",
        input_goal_ids=["metric.current"],
        baseline_goal_ids=["metric.baseline"],
    )
    current = query_artifact("current", [{"value": 2}], {"value": ["metric:a"]})
    baseline = query_artifact("baseline", [{"value": 1}], {"value": ["metric:b"]})

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["current", "baseline"],
            "seriesBindings": [
                {"seriesId": "now", "artifactId": "current", "valueColumn": "value"},
                {"seriesId": "before", "artifactId": "baseline", "valueColumn": "value"},
            ],
            "method": "BASELINE_NORMALIZED_DEVIATION",
            "normalizationMethod": "PERCENT_CHANGE",
            "baselinePairs": [{"currentSeriesId": "now", "baselineSeriesId": "before"}],
            "anomalyThreshold": 10,
        },
        verified_query_artifacts=[current, baseline],
        artifact_goal_ids={
            "current": ["metric.current"],
            "baseline": ["metric.baseline"],
        },
    )

    assert artifact.publication_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.verified_evidence.gaps[0].code == ("ANOMALY_BASELINE_LINEAGE_INCOMPARABLE")


def test_correlation_aligns_grain_requires_samples_and_disclaims_causation() -> None:
    goal_contract = contract(
        "correlation",
        input_goal_ids=["metric.x", "metric.y"],
    )
    source = query_artifact(
        "series",
        [{"pt": day, "x": day, "y": day * 3} for day in range(1, 7)],
        {"pt": ["field:date"], "x": ["metric:x"], "y": ["metric:y"]},
    )

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["series"],
            "seriesBindings": [
                {"seriesId": "x", "artifactId": "series", "valueColumn": "x"},
                {"seriesId": "y", "artifactId": "series", "valueColumn": "y"},
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[source],
        artifact_goal_ids={"series": ["metric.x", "metric.y"]},
    )

    assert artifact.publication_status == "PROVED"
    assert artifact.result["sampleCount"] == 6
    assert artifact.result["coefficient"] == 1.0
    assert artifact.result["correlationNotCausation"] is True
    assert artifact.correlation_not_causation is True
    assert artifact.causal_claim_allowed is False
    assert artifact.goal_resolutions[0].details["causalClaimAllowed"] is False
    coverage = grounded_analysis_goal_coverage(goal_contract, artifact)
    assert coverage.goal_resolutions[0].resolution == "PROVED"
    rendered = render_grounded_analysis_artifact(artifact)
    assert "系数为 1.0" in rendered.answer_markdown
    assert "6 个对齐观测" in rendered.answer_markdown
    assert "相关不等于因果" in rendered.answer_markdown


def test_correlation_with_too_few_samples_is_insufficient() -> None:
    goal_contract = contract(
        "correlation",
        input_goal_ids=["metric.x", "metric.y"],
    )
    source = query_artifact(
        "series",
        [{"pt": day, "x": day, "y": day * 2} for day in range(1, 5)],
        {"pt": ["field:date"], "x": ["metric:x"], "y": ["metric:y"]},
    )

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["series"],
            "seriesBindings": [
                {"seriesId": "x", "artifactId": "series", "valueColumn": "x"},
                {"seriesId": "y", "artifactId": "series", "valueColumn": "y"},
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[source],
        artifact_goal_ids={"series": ["metric.x", "metric.y"]},
    )

    assert artifact.publication_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.verified_evidence.gaps[0].code == "CORRELATION_SAMPLE_INSUFFICIENT"


def test_correlation_requires_matching_observation_lineage() -> None:
    goal_contract = contract(
        "correlation",
        input_goal_ids=["metric.x", "metric.y"],
    )
    left = query_artifact(
        "left",
        [{"pt": day, "x": day} for day in range(1, 7)],
        {"pt": ["field:date-a"], "x": ["metric:x"]},
    )
    right = query_artifact(
        "right",
        [{"pt": day, "y": day} for day in range(1, 7)],
        {"pt": ["field:date-b"], "y": ["metric:y"]},
    )

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["left", "right"],
            "seriesBindings": [
                {"seriesId": "x", "artifactId": "left", "valueColumn": "x"},
                {"seriesId": "y", "artifactId": "right", "valueColumn": "y"},
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[left, right],
        artifact_goal_ids={"left": ["metric.x"], "right": ["metric.y"]},
    )

    assert artifact.publication_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.verified_evidence.gaps[0].code == ("OBSERVATION_GRAIN_LINEAGE_MISMATCH")


def test_difference_and_trend_are_reproducible() -> None:
    difference_contract = contract(
        "difference",
        input_goal_ids=["metric.left", "metric.right"],
    )
    scalar = query_artifact(
        "scalar",
        [{"left": 120, "right": 100}],
        {"left": ["metric:left"], "right": ["metric:right"]},
    )
    difference_request = {
        "analysisGoalId": "analysis.main",
        "inputArtifactIds": ["scalar"],
        "seriesBindings": [
            {"seriesId": "left", "artifactId": "scalar", "valueColumn": "left"},
            {"seriesId": "right", "artifactId": "scalar", "valueColumn": "right"},
        ],
        "method": "ALIGNED_ABSOLUTE_AND_PERCENT_DIFFERENCE",
        "leftSeriesId": "left",
        "rightSeriesId": "right",
    }
    first = publish_grounded_analysis_from_skill(
        goal_contract=difference_contract,
        publication_request=difference_request,
        verified_query_artifacts=[scalar],
        artifact_goal_ids={"scalar": ["metric.left", "metric.right"]},
    )
    second = publish_grounded_analysis_from_skill(
        goal_contract=difference_contract,
        publication_request=difference_request,
        verified_query_artifacts=[scalar],
        artifact_goal_ids={"scalar": ["metric.left", "metric.right"]},
    )

    assert first.artifact_id == second.artifact_id
    assert first.result["rows"][0]["difference"] == "20"
    assert first.result["rows"][0]["percentDifference"] == "20"
    difference_rendered = render_grounded_analysis_artifact(first)
    assert "左值 120、右值 100，差值 20" in difference_rendered.answer_markdown

    trend_contract = contract("trend", input_goal_ids=["metric.value"])
    trend_source = query_artifact(
        "trend",
        [
            {"pt": "2026-01-03", "value": 9},
            {"pt": "2026-01-01", "value": 1},
            {"pt": "2026-01-02", "value": 5},
        ],
        {"pt": ["field:date"], "value": ["metric:value"]},
    )
    trend = publish_grounded_analysis_from_skill(
        goal_contract=trend_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["trend"],
            "seriesBindings": [{"seriesId": "value", "artifactId": "trend", "valueColumn": "value"}],
            "observationKeys": ["pt"],
            "method": "FIRST_LAST_DELTA_WITH_OLS",
        },
        verified_query_artifacts=[trend_source],
        artifact_goal_ids={"trend": ["metric.value"]},
    )

    profile = trend.result["series"][0]
    assert profile["first"] == "1"
    assert profile["last"] == "9"
    assert profile["delta"] == "8"
    assert profile["olsSlopePerObservation"] == "4"
    trend_rendered = render_grounded_analysis_artifact(trend)
    assert "1 → 9，变化 8，OLS 斜率 4" in trend_rendered.answer_markdown


def test_causal_goal_is_never_published_as_a_causal_claim() -> None:
    goal_contract = contract(
        "attribution",
        input_goal_ids=["metric.x", "metric.y"],
        question="是不是 X 导致 Y？",
    )
    source = query_artifact(
        "series",
        [{"pt": day, "x": day, "y": day} for day in range(1, 7)],
        {"pt": ["field:date"], "x": ["metric:x"], "y": ["metric:y"]},
    )

    artifact = publish_grounded_analysis_from_skill(
        goal_contract=goal_contract,
        publication_request={
            "analysisGoalId": "analysis.main",
            "inputArtifactIds": ["series"],
            "seriesBindings": [
                {"seriesId": "x", "artifactId": "series", "valueColumn": "x"},
                {"seriesId": "y", "artifactId": "series", "valueColumn": "y"},
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[source],
        artifact_goal_ids={"series": ["metric.x", "metric.y"]},
    )

    assert artifact.publication_status == "INSUFFICIENT_EVIDENCE"
    assert artifact.causal_claim_allowed is False
    assert artifact.verified_evidence.gaps[0].code == ("CAUSAL_INFERENCE_NOT_SUPPORTED")
    assert "cannot prove" in artifact.goal_resolutions[0].reason

    with pytest.raises(ValidationError):
        GroundedRunSkillAnalysisPublicationRequest.model_validate(
            {
                "analysisGoalId": "analysis.main",
                "inputArtifactIds": ["series"],
                "seriesBindings": [
                    {
                        "seriesId": "x",
                        "artifactId": "series",
                        "valueColumn": "x",
                    }
                ],
                "conclusion": "X caused Y",
            }
        )
