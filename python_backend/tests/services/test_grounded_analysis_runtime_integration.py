from __future__ import annotations

from types import SimpleNamespace

from merchant_ai.models import AgentRunResult, QueryBundle, VerifiedEvidence
from merchant_ai.services.grounded_analysis_artifact import (
    grounded_analysis_goal_coverage,
    publish_grounded_analysis_from_skill,
    render_grounded_analysis_artifact,
)
from merchant_ai.services.grounded_answer_coverage import (
    AnswerCoverageVerifier,
    answer_attestation_matches,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    GoalCoverageVerifier,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    VerifiedArtifactGoalCoverage,
    original_question_goal_contract_fingerprint,
)


def test_compose_verified_answer_closes_analysis_goal_and_attests_answer() -> None:
    """Exercise the public artifact/coverage/renderer boundary directly.

    The former test called a removed private Runtime method and therefore did
    not protect the real integration contract.  This test publishes the
    deterministic artifact, verifies Goal coverage, renders it, and attests
    the exact final-answer source used by ``compose_verified_answer``.
    """

    contract = OriginalQuestionGoalContract(
        question="分析 x 与 y 的相关性",
        goals=[
            MetricQuestionGoal(
                goal_id="metric.x",
                label="x",
                metric_ref_id="metric:x",
            ),
            MetricQuestionGoal(
                goal_id="metric.y",
                label="y",
                metric_ref_id="metric:y",
            ),
            AnalysisQuestionGoal(
                goal_id="analysis.correlation",
                label="x 与 y 的相关性",
                analysis_type="correlation",
                input_goal_ids=["metric.x", "metric.y"],
            ),
        ],
    )
    rows = [{"pt": day, "x": day, "y": day * 2} for day in range(1, 7)]
    query_artifact = SimpleNamespace(
        artifact_id="query-series",
        contract=SimpleNamespace(
            evidence_refs=["metric:x", "metric:y", "field:pt"],
            query_shape="GROUPED",
            ranking=SimpleNamespace(enabled=False),
        ),
        run_result=AgentRunResult(
            merged_query_bundle=QueryBundle(rows=rows, tables=["series"])
        ),
        verified_evidence=VerifiedEvidence(passed=True),
        output_columns=["pt", "x", "y"],
        output_lineage={
            "pt": ["field:pt"],
            "x": ["metric:x"],
            "y": ["metric:y"],
        },
    )
    derived = publish_grounded_analysis_from_skill(
        goal_contract=contract,
        publication_request={
            "analysisGoalId": "analysis.correlation",
            "inputArtifactIds": ["query-series"],
            "seriesBindings": [
                {
                    "seriesId": "x",
                    "artifactId": "query-series",
                    "valueColumn": "x",
                },
                {
                    "seriesId": "y",
                    "artifactId": "query-series",
                    "valueColumn": "y",
                },
            ],
            "observationKeys": ["pt"],
            "method": "PEARSON_CORRELATION",
            "leftSeriesId": "x",
            "rightSeriesId": "y",
        },
        verified_query_artifacts=[query_artifact],
        artifact_goal_ids={"query-series": ["metric.x", "metric.y"]},
    )

    primitive_coverage = VerifiedArtifactGoalCoverage(
        artifact_id="query-series",
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(
            contract
        ),
        covered_goal_ids=["metric.x", "metric.y"],
        verification_passed=True,
        evidence_refs=["metric:x", "metric:y"],
        goal_resolutions=[
            {
                "goalId": "metric.x",
                "goalKind": "METRIC",
                "resolution": "PROVED",
                "proofType": "QUERY_VALUE",
                "evidenceRefs": ["metric:x"],
                "metricRefIds": ["metric:x"],
                "valueRefs": ["query:query-series:x"],
            },
            {
                "goalId": "metric.y",
                "goalKind": "METRIC",
                "resolution": "PROVED",
                "proofType": "QUERY_VALUE",
                "evidenceRefs": ["metric:y"],
                "metricRefIds": ["metric:y"],
                "valueRefs": ["query:query-series:y"],
            },
        ],
    )
    coverage = GoalCoverageVerifier().verify(
        contract,
        [
            primitive_coverage,
            grounded_analysis_goal_coverage(contract, derived),
        ],
    )
    rendered = render_grounded_analysis_artifact(derived)
    answer_coverage = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        rendered.answer_markdown,
        [rendered.binding],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert derived.publication_status == "PROVED"
    assert coverage.finalization_allowed is True
    assert coverage.covered_goal_ids == [
        "metric.x",
        "metric.y",
        "analysis.correlation",
    ]
    assert "相关不等于因果" in rendered.answer_markdown
    assert answer_coverage.passed is True
    assert answer_attestation_matches(rendered.answer_markdown, answer_coverage)
