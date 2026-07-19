from __future__ import annotations

from types import SimpleNamespace

from merchant_ai.models import AgentRunResult, QueryBundle, VerifiedEvidence
from merchant_ai.services.grounded_answer_coverage import (
    answer_attestation_matches,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)


def test_run_skill_publication_closes_analysis_goal_and_attests_answer() -> None:
    contract = OriginalQuestionGoalContract(
        question="分析 x 与 y 的相关性",
        goals=[
            MetricQuestionGoal(goal_id="metric.x", label="x"),
            MetricQuestionGoal(goal_id="metric.y", label="y"),
            AnalysisQuestionGoal(
                goal_id="analysis.correlation",
                label="x 与 y 的相关性",
                analysis_type="correlation",
                input_goal_ids=["metric.x", "metric.y"],
            ),
        ],
    )
    rows = [
        {"pt": day, "x": day, "y": day * 2}
        for day in range(1, 7)
    ]
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(rows=rows, tables=["series"])
    )
    query_artifact = SimpleNamespace(
        artifact_id="query-series",
        contract=SimpleNamespace(
            evidence_refs=["metric:x", "metric:y", "field:pt"],
            query_shape="GROUPED",
            ranking=SimpleNamespace(enabled=False),
        ),
        run_result=run_result,
        verified_evidence=VerifiedEvidence(passed=True),
        output_columns=["pt", "x", "y"],
        output_lineage={
            "pt": ["field:pt"],
            "x": ["metric:x"],
            "y": ["metric:y"],
        },
    )
    state = SimpleNamespace(
        verified_query_ledger=[query_artifact],
        verified_rule_ledger=[],
        answer="",
        phase="VERIFIED",
    )
    session = GroundedDeepAgentSession(
        runtime=state,
        question_goal_contract=contract,
        artifact_goal_ids={
            "query-series": ["metric.x", "metric.y"]
        },
    )

    class Kernel:
        @staticmethod
        def compose_answer(runtime_state: object, *, allow_llm: bool) -> str:
            assert allow_llm is False
            runtime_state.answer = "x 与 y 的已验证序列已收集。"
            return runtime_state.answer

    runtime = object.__new__(GroundedDeepAgentRuntime)
    runtime.kernel = Kernel()

    answer, artifacts, coverage, answer_coverage = (
        runtime._finalize_attested_skill_answer(
            session,
            [
                {
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
                }
            ],
        )
    )

    assert len(artifacts) == 1
    assert coverage.finalization_allowed is True
    assert coverage.covered_goal_ids == [
        "metric.x",
        "metric.y",
        "analysis.correlation",
    ]
    assert "相关不等于因果" in answer
    assert answer_coverage.passed is True
    assert answer_attestation_matches(answer, session.answer_coverage_result)
    assert session.runtime.phase == "ANSWERED"
