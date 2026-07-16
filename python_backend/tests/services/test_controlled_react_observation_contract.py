from __future__ import annotations

from typing import Any

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    EvidenceGap,
    GraphValidationGap,
    GraphValidationResult,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    VerifiedEvidence,
)
from merchant_ai.services.controlled_react import ControlledReactExplorer


FORBIDDEN_DECISION_KEYS = {
    "action",
    "decision",
    "rank",
    "selectedCandidateId",
    "survivorIds",
    "winnerId",
}


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in nested_keys(item)}
    return set()


def test_hypotheses_are_model_supplied_only_and_never_seeded_from_question_or_assets():
    explorer = ControlledReactExplorer()
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="synthetic_table")],
        metrics=[PlanningAssetEntry(key="synthetic_metric", table="synthetic_table")],
    )

    empty = explorer.build_hypotheses("a question that used to trigger a template", pack, {})
    supplied = explorer.build_hypotheses(
        "question text is not used to invent candidates",
        pack,
        {
            "questionUnderstanding": {
                "analysisHypotheses": [
                    {
                        "hypothesis_id": "model_h1",
                        "statement": "Model-provided proposition",
                        "rationale": "Model-provided rationale",
                        "metric_hints": ["synthetic_metric"],
                        "required_evidence": ["aligned_observation"],
                    }
                ]
            }
        },
    )

    assert empty["hypotheses"] == []
    assert supplied["hypotheses"] == [
        {
            "hypothesisId": "model_h1",
            "title": "Model-provided proposition",
            "reason": "Model-provided rationale",
            "metricHints": ["synthetic_metric"],
            "requiredEvidence": ["aligned_observation"],
            "status": "candidate",
            "source": "lead_or_planner_model",
        }
    ]
    assert supplied["guardrails"] == ["model_supplied_only", "observation_only", "no_automatic_selection"]
    assert not (nested_keys(supplied) & FORBIDDEN_DECISION_KEYS)


def test_parallel_review_emits_only_evidence_observations_in_input_order():
    explorer = ControlledReactExplorer()
    hypotheses = {
        "hypotheses": [
            {"hypothesisId": "h1", "title": "First", "metricHints": ["metric_a"]},
            {"hypothesisId": "h2", "title": "Second", "metricHints": ["metric_b"]},
        ]
    }
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(rows=[{"period": "p1", "metric_a": 10, "metric_b": 20}])
    )

    reviews = explorer.run_parallel_evidence_reviews(hypotheses, run_result)

    assert [item["hypothesisId"] for item in reviews] == ["h1", "h2"]
    assert all(item["checks"]["rowsPresent"] for item in reviews)
    assert all(item["checks"]["hintMatched"] for item in reviews)
    assert all(item["automaticSelection"] is False for item in reviews)
    assert not (nested_keys(reviews) & FORBIDDEN_DECISION_KEYS)


def test_execution_gate_ledger_reports_checks_without_ranking_or_promotion():
    explorer = ControlledReactExplorer()
    passed_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="node_1",
                success=True,
                query_bundle=QueryBundle(rows=[{"metric_a": 1}]),
            )
        ],
        merged_query_bundle=QueryBundle(rows=[{"metric_a": 1}]),
        verified_evidence=VerifiedEvidence(passed=True, covered_evidence=["metric_a"]),
    )
    failed_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="node_2",
                success=False,
                query_bundle=QueryBundle(failed=True, error="synthetic failure"),
            )
        ],
        merged_query_bundle=QueryBundle(failed=True),
        verified_evidence=VerifiedEvidence(
            passed=False,
            gaps=[EvidenceGap(code="SYNTHETIC_GAP", reason="missing evidence")],
        ),
    )

    ledger = explorer.build_execution_gate_ledger(
        [
            {
                "hypothesisId": "h1",
                "validation": GraphValidationResult(valid=True),
                "runResult": passed_result,
            },
            {
                "hypothesisId": "h2",
                "validation": GraphValidationResult(
                    valid=False,
                    gaps=[GraphValidationGap(code="SYNTHETIC_VALIDATION_GAP")],
                ),
                "runResult": failed_result,
            },
        ]
    )

    assert [item["hypothesisId"] for item in ledger["observations"]] == ["h1", "h2"]
    assert ledger["observations"][0]["checks"] == {
        "queryGraphValid": True,
        "evidenceVerified": True,
        "resultRowsPresent": True,
    }
    assert ledger["observations"][1]["checks"] == {
        "queryGraphValid": False,
        "evidenceVerified": False,
        "resultRowsPresent": False,
    }
    assert ledger["automaticSelection"] is False
    assert ledger["decisionOwner"] == "lead_react_model"
    assert not (nested_keys(ledger) & FORBIDDEN_DECISION_KEYS)
