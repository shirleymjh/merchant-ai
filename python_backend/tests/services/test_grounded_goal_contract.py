from __future__ import annotations

from types import SimpleNamespace

import pytest

from merchant_ai.services.grounded_goal_contract import (
    ComparisonQuestionGoal,
    DependencyQuestionGoal,
    GoalContractValidationError,
    GoalCoverageBlocked,
    GoalCoverageVerifier,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    TimeWindowQuestionGoal,
    VerifiedArtifactGoalCoverage,
    declare_verified_artifact_goal_coverage,
    inspect_question_structure,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
    required_goal_ids,
    validate_original_question_goal_contract,
)


def simple_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "Return metric alpha by dimension beta",
            "goals": [
                {
                    "goalId": "metric.alpha",
                    "kind": "metric",
                    "label": "metric alpha",
                },
                {
                    "goalId": "dimension.beta",
                    "kind": "dimension",
                    "label": "dimension beta",
                },
            ],
        }
    )


def declaration(
    contract: OriginalQuestionGoalContract,
    artifact_id: str,
    goal_ids: list[str],
    *,
    verified: bool = True,
    fingerprint: str = "",
) -> VerifiedArtifactGoalCoverage:
    return VerifiedArtifactGoalCoverage(
        artifact_id=artifact_id,
        goal_contract_fingerprint=fingerprint or original_question_goal_contract_fingerprint(contract),
        covered_goal_ids=goal_ids,
        verification_passed=verified,
    )


def test_grouped_core_payload_is_flattened_typed_and_normalized() -> None:
    contract = parse_original_question_goal_contract(
        {
            "contractVersion": "original_question_goal_contract.v1",
            "question": "Show metric alpha for the last 30 days",
            "metrics": [
                {
                    "goalId": " Metric.Alpha ",
                    "label": " metric alpha ",
                    "semanticRefIds": [" semantic:metric:alpha ", "semantic:metric:alpha"],
                }
            ],
            "timeWindows": [
                {
                    "goalId": " Time.Primary ",
                    "label": " last 30 days ",
                    "timeExpression": " last 30 days ",
                    "appliesToGoalIds": [" METRIC.ALPHA "],
                }
            ],
        }
    )

    assert [goal.goal_id for goal in contract.goals] == ["metric.alpha", "time.primary"]
    assert isinstance(contract.goals[0], MetricQuestionGoal)
    assert contract.goals[0].semantic_ref_ids == ["semantic:metric:alpha"]
    assert isinstance(contract.goals[1], TimeWindowQuestionGoal)
    assert contract.goals[1].applies_to_goal_ids == ["metric.alpha"]


def test_duplicate_ids_after_normalization_are_rejected() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return two goals",
            "metrics": [
                {"goalId": "Metric One", "label": "first"},
                {"goalId": "metric_one", "label": "second"},
            ],
        }
    )

    assert result.valid is False
    assert {issue.code for issue in result.issues} == {"DUPLICATE_GOAL_ID"}
    with pytest.raises(GoalContractValidationError):
        parse_original_question_goal_contract(
            {
                "question": "Return two goals",
                "metrics": [
                    {"goalId": "Metric One", "label": "first"},
                    {"goalId": "metric_one", "label": "second"},
                ],
            }
        )


def test_unknown_references_and_dependency_cycles_fail_closed() -> None:
    unknown = validate_original_question_goal_contract(
        {
            "question": "Return a goal",
            "metrics": [
                {
                    "goalId": "metric.alpha",
                    "label": "metric alpha",
                    "dependsOnGoalIds": ["missing.goal"],
                }
            ],
        }
    )
    assert "UNKNOWN_GOAL_REFERENCE" in {issue.code for issue in unknown.issues}

    cyclic = validate_original_question_goal_contract(
        {
            "question": "Return two dependent goals",
            "goals": [
                {
                    "goalId": "metric.alpha",
                    "kind": "metric",
                    "label": "metric alpha",
                    "dependsOnGoalIds": ["dimension.beta"],
                },
                {
                    "goalId": "dimension.beta",
                    "kind": "dimension",
                    "label": "dimension beta",
                    "dependsOnGoalIds": ["metric.alpha"],
                },
            ],
        }
    )
    assert "GOAL_DEPENDENCY_CYCLE" in {issue.code for issue in cyclic.issues}


def test_explicit_time_and_comparison_cues_require_structural_goals() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Compare metric alpha for the last 30 days versus the previous 30 days",
            "metrics": [{"goalId": "metric.alpha", "label": "metric alpha"}],
        }
    )

    assert result.valid is False
    assert {issue.code for issue in result.issues} >= {
        "STRUCTURAL_TIME_GOAL_MISSING",
        "STRUCTURAL_COMPARISON_GOAL_MISSING",
    }

    coverage = GoalCoverageVerifier().verify(result.contract, [])
    assert coverage.finalization_allowed is False
    assert {issue.code for issue in coverage.issues} >= {
        "STRUCTURAL_TIME_GOAL_MISSING",
        "STRUCTURAL_COMPARISON_GOAL_MISSING",
    }


def test_conjunction_hint_only_requests_review_and_never_invents_business_goals() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return alpha and beta",
            "metrics": [{"goalId": "metric.alpha", "label": "alpha"}],
        }
    )
    hints = inspect_question_structure("Return alpha and beta")

    assert result.valid is True
    issue = next(issue for issue in result.issues if issue.code == "STRUCTURAL_CONJUNCTION_REVIEW_REQUIRED")
    assert issue.blocking is False
    assert hints.conjunction_cues == ["and"]


def test_required_goals_include_entity_chain_dependency_closure() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "Resolve a dependent entity result",
            "goals": [
                {
                    "goalId": "metric.upstream",
                    "kind": "metric",
                    "label": "upstream result",
                    "required": False,
                },
                {
                    "goalId": "chain.lookup",
                    "kind": "dependency",
                    "label": "typed entity chain",
                    "required": False,
                    "upstreamGoalIds": ["metric.upstream"],
                    "downstreamGoalIds": ["entity.result"],
                },
                {
                    "goalId": "entity.result",
                    "kind": "entity",
                    "label": "dependent entity result",
                    "required": True,
                },
            ],
        }
    )

    assert isinstance(contract.goals[1], DependencyQuestionGoal)
    assert required_goal_ids(contract) == ["metric.upstream", "chain.lookup", "entity.result"]


def test_multiple_verified_artifacts_can_jointly_cover_the_question() -> None:
    contract = simple_contract()
    result = GoalCoverageVerifier().verify(
        contract,
        [
            declaration(contract, "artifact-metric", ["metric.alpha"]),
            declaration(contract, "artifact-dimension", ["dimension.beta"]),
        ],
    )

    assert result.passed is True
    assert result.finalization_allowed is True
    assert result.covered_goal_ids == ["metric.alpha", "dimension.beta"]
    assert result.coverage_by_goal_id == {
        "metric.alpha": ["artifact-metric"],
        "dimension.beta": ["artifact-dimension"],
    }


def test_missing_required_goal_blocks_finalization() -> None:
    contract = simple_contract()
    verifier = GoalCoverageVerifier()
    result = verifier.verify(
        contract,
        [declaration(contract, "artifact-metric", ["metric.alpha"])],
    )

    assert result.finalization_allowed is False
    assert result.missing_required_goal_ids == ["dimension.beta"]
    assert "REQUIRED_GOAL_UNCOVERED" in {issue.code for issue in result.issues}
    with pytest.raises(GoalCoverageBlocked) as exc_info:
        verifier.require_complete(contract, [declaration(contract, "artifact-metric", ["metric.alpha"])])
    assert exc_info.value.result.missing_required_goal_ids == ["dimension.beta"]


def test_unverified_or_stale_artifacts_cannot_contribute_coverage() -> None:
    contract = simple_contract()
    result = GoalCoverageVerifier().verify(
        contract,
        [
            declaration(contract, "artifact-unverified", ["metric.alpha"], verified=False),
            declaration(
                contract,
                "artifact-stale",
                ["dimension.beta"],
                fingerprint="0" * 64,
            ),
        ],
    )

    assert result.covered_goal_ids == []
    assert result.finalization_allowed is False
    assert {issue.code for issue in result.issues} >= {
        "ARTIFACT_NOT_VERIFIED",
        "GOAL_CONTRACT_FINGERPRINT_MISMATCH",
    }


def test_declared_goal_semantic_refs_require_matching_artifact_evidence() -> None:
    metric_ref = "semantic:orders:metric:gmv"
    contract = parse_original_question_goal_contract(
        {
            "question": "Return GMV",
            "goals": [
                {
                    "goalId": "metric.gmv",
                    "kind": "metric",
                    "label": "GMV",
                    "metricRefId": metric_ref,
                }
            ],
        }
    )
    fingerprint = original_question_goal_contract_fingerprint(contract)

    mismatched = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-wrong-semantic",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=["metric.gmv"],
                verification_passed=True,
                evidence_refs=["semantic:orders:metric:order_count"],
            )
        ],
    )

    assert mismatched.finalization_allowed is False
    assert mismatched.missing_required_goal_ids == ["metric.gmv"]
    assert any(
        issue.code == "GOAL_SEMANTIC_EVIDENCE_UNCOVERED"
        for issue in mismatched.issues
    )

    matched = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-gmv",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=["metric.gmv"],
                verification_passed=True,
                evidence_refs=[metric_ref],
            )
        ],
    )

    assert matched.finalization_allowed is True
    assert matched.covered_goal_ids == ["metric.gmv"]


def test_unknown_claimed_goal_id_is_a_blocking_contract_drift_signal() -> None:
    contract = simple_contract()
    result = GoalCoverageVerifier().verify(
        contract,
        [declaration(contract, "artifact-1", ["metric.alpha", "unknown.goal", "dimension.beta"])],
    )

    assert result.covered_goal_ids == ["metric.alpha", "dimension.beta"]
    assert result.missing_required_goal_ids == []
    assert result.finalization_allowed is False
    assert "UNKNOWN_COVERED_GOAL_ID" in {issue.code for issue in result.issues}


def test_claimed_comparison_is_not_effective_until_its_operands_are_covered() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "Compare alpha versus beta",
            "goals": [
                {
                    "goalId": "metric.alpha",
                    "kind": "metric",
                    "label": "alpha",
                    "required": False,
                },
                {
                    "goalId": "metric.beta",
                    "kind": "metric",
                    "label": "beta",
                    "required": False,
                },
                {
                    "goalId": "comparison.alpha_beta",
                    "kind": "comparison",
                    "label": "alpha versus beta",
                    "leftGoalIds": ["metric.alpha"],
                    "rightGoalIds": ["metric.beta"],
                },
            ],
        }
    )
    assert isinstance(contract.goals[2], ComparisonQuestionGoal)

    result = GoalCoverageVerifier().verify(
        contract,
        [declaration(contract, "artifact-comparison", ["comparison.alpha_beta"])],
    )

    assert result.claimed_covered_goal_ids == ["comparison.alpha_beta"]
    assert result.covered_goal_ids == []
    assert result.missing_required_goal_ids == [
        "metric.alpha",
        "metric.beta",
        "comparison.alpha_beta",
    ]
    assert "COVERED_GOAL_DEPENDENCY_UNCOVERED" in {issue.code for issue in result.issues}


def test_declaration_helper_reads_kernel_verification_instead_of_trusting_caller() -> None:
    contract = simple_contract()
    verified_artifact = SimpleNamespace(
        artifact_id="artifact-verified",
        verified_evidence=SimpleNamespace(passed=True),
    )
    failed_artifact = SimpleNamespace(
        artifact_id="artifact-failed",
        verified_evidence=SimpleNamespace(passed=False),
    )

    declared = declare_verified_artifact_goal_coverage(
        contract,
        verified_artifact,
        [" Metric.Alpha ", "metric.alpha"],
        evidence_refs=["evidence:1"],
    )

    assert declared.covered_goal_ids == ["metric.alpha"]
    assert declared.verification_passed is True
    assert declared.goal_contract_fingerprint == original_question_goal_contract_fingerprint(contract)
    with pytest.raises(ValueError, match="verified query artifact"):
        declare_verified_artifact_goal_coverage(contract, failed_artifact, ["metric.alpha"])


def test_verifier_can_read_coverage_fields_directly_from_kernel_artifact_shape() -> None:
    contract = simple_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)
    artifact = SimpleNamespace(
        artifact_id="artifact-direct",
        goal_contract_fingerprint=fingerprint,
        covered_goal_ids=["metric.alpha", "dimension.beta"],
        goal_coverage_evidence_refs=["verified:evidence"],
        verified_evidence=SimpleNamespace(passed=True),
    )

    result = GoalCoverageVerifier().verify(contract, [artifact])

    assert result.finalization_allowed is True
    assert result.artifact_ids == ["artifact-direct"]
