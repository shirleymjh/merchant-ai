from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import merchant_ai.services.grounded_goal_contract as goal_contract_module
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    ComparisonQuestionGoal,
    DetailQuestionGoal,
    DependencyQuestionGoal,
    DimensionQuestionGoal,
    GoalContractIssue,
    GoalContractValidationError,
    GoalCoverageBlocked,
    GoalCoverageVerifier,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    OriginalQuestionGoalDeclaration,
    RankingQuestionGoal,
    RuleQuestionGoal,
    TimeWindowQuestionGoal,
    VerifiedArtifactGoalCoverage,
    declare_verified_artifact_goal_coverage,
    goal_dependency_closure,
    inspect_question_structure,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
    required_goal_ids,
    validate_original_question_goal_contract,
)


def test_goal_declaration_schema_matches_runtime_validation() -> None:
    schema = OriginalQuestionGoalDeclaration.model_json_schema(by_alias=True)
    assert "question" not in schema.get("properties", {})
    time_schema = schema["$defs"]["TimeWindowQuestionGoal"]
    assert "timeExpression" in time_schema.get("required", [])
    assert "anchorPolicy" not in time_schema.get("properties", {})
    assert "calendarAnchorPolicy" in time_schema.get("properties", {})
    assert "dataAsOfPolicy" in time_schema.get("properties", {})


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


def test_question_semantics_are_delegated_to_explicit_fail_closed_verifier() -> None:
    payload = {
        "question": "Compare metric alpha for the last 30 days versus the previous 30 days",
        "metrics": [{"goalId": "metric.alpha", "label": "metric alpha"}],
    }
    local_result = validate_original_question_goal_contract(payload)

    class IncompleteDeclarationVerifier:
        def verify(self, contract):
            assert contract.question == payload["question"]
            return [
                GoalContractIssue(
                    code="QUESTION_GOAL_DECLARATION_INCOMPLETE",
                    message="the external semantic verifier found omitted typed goals",
                )
            ]

    verified_result = validate_original_question_goal_contract(
        payload,
        question_verifier=IncompleteDeclarationVerifier(),
    )

    assert local_result.valid is True
    assert local_result.issues == []
    assert verified_result.valid is False
    assert {issue.code for issue in verified_result.issues} == {
        "QUESTION_GOAL_DECLARATION_INCOMPLETE"
    }

    coverage = GoalCoverageVerifier().verify(verified_result.contract, [])
    assert coverage.finalization_allowed is False


def test_question_verifier_failure_and_malformed_output_fail_closed() -> None:
    payload = {
        "question": "Return alpha",
        "metrics": [{"goalId": "metric.alpha", "label": "metric alpha"}],
    }

    class FailedVerifier:
        def verify(self, contract):
            raise RuntimeError("unavailable")

    class MalformedVerifier:
        def verify(self, contract):
            return "not a typed issue sequence"

    for verifier in (FailedVerifier(), MalformedVerifier()):
        result = validate_original_question_goal_contract(
            payload,
            question_verifier=verifier,
        )
        assert result.valid is False
        assert {issue.code for issue in result.issues} == {
            "GOAL_CONTRACT_QUESTION_VERIFIER_FAILED"
        }


def test_unicode_question_inspection_is_non_authoritative_and_vocabulary_free() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Alpha 42；β ٣!",
            "metrics": [{"goalId": "metric.alpha", "label": "alpha"}],
        }
    )
    hints = inspect_question_structure("Alpha 42；β ٣!")

    assert result.valid is True
    assert result.issues == []
    assert hints.token_count == 6
    assert hints.number_tokens == ["42", "٣"]
    assert hints.punctuation_tokens == ["；", "!"]
    assert hints.clause_count == 2


def test_goal_contract_module_has_no_pattern_engine_dependency_or_calls() -> None:
    source_path = Path(goal_contract_module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    blocked_modules = {"re", "regex"}
    imported_modules = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    blocked_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in blocked_modules
    ]

    assert imported_modules.isdisjoint(blocked_modules)
    assert blocked_calls == []


def test_ranking_limit_is_never_silently_defaulted() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return a ranked result",
            "goals": [
                {"goalId": "metric.alpha", "kind": "METRIC", "label": "alpha"},
                {
                    "goalId": "ranking.alpha",
                    "kind": "RANKING",
                    "label": "rank alpha",
                    "metricGoalIds": ["metric.alpha"],
                    "populationScope": "ALL_MATCHING_ROWS",
                },
            ],
        }
    )

    assert result.valid is False
    assert result.contract is not None
    assert result.contract.goal_map()["ranking.alpha"].limit == 0
    assert "RANKING_LIMIT_REQUIRED" in {issue.code for issue in result.issues}


def test_ranking_defaults_to_its_own_matching_rows_without_execution_scope() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return a ranked result",
            "goals": [
                {"goalId": "metric.alpha", "kind": "METRIC", "label": "alpha"},
                {
                    "goalId": "ranking.alpha",
                    "kind": "RANKING",
                    "label": "rank alpha",
                    "metricGoalIds": ["metric.alpha"],
                    "limit": 3,
                },
            ],
        }
    )

    assert result.valid is True
    assert result.contract is not None
    ranking = result.contract.goal_map()["ranking.alpha"]
    assert ranking.population_scope == "ALL_MATCHING_ROWS"
    assert ranking.population_goal_ids == []


def test_legacy_population_fields_on_detail_are_discarded() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return recent order details",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "DETAIL",
                    "label": "order details",
                    "populationScope": "VERIFIED_PREDICATE_SCOPE",
                    "populationGoalIds": ["entity.orders"],
                }
            ],
        }
    )

    assert result.valid is True
    assert result.contract is not None
    detail = result.contract.goal_map()["detail.orders"]
    # Compatibility fields remain on trusted checkpoint models, but model
    # declarations cannot set them. The normalizer restores safe defaults.
    assert detail.population_scope == "ALL_MATCHING_ROWS"
    assert detail.population_goal_ids == []


def test_explicit_all_matching_rows_population_is_accepted() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "Return a ranked result",
            "goals": [
                {"goalId": "metric.alpha", "kind": "METRIC", "label": "alpha"},
                {
                    "goalId": "ranking.alpha",
                    "kind": "RANKING",
                    "label": "rank alpha",
                    "metricGoalIds": ["metric.alpha"],
                    "limit": 3,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
            ],
        }
    )

    assert result.valid is True
    assert result.contract is not None
    assert result.contract.goal_map()["ranking.alpha"].population_scope == (
        "ALL_MATCHING_ROWS"
    )


def test_ranking_can_bind_an_explicit_prior_goal_population() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "Return two typed outputs over one declared population",
            "goals": [
                {"goalId": "detail.rows", "kind": "DETAIL", "label": "rows"},
                {"goalId": "metric.alpha", "kind": "METRIC", "label": "alpha"},
                {
                    "goalId": "ranking.alpha",
                    "kind": "RANKING",
                    "label": "rank alpha",
                    "metricGoalIds": ["metric.alpha"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                    "populationGoalIds": ["detail.rows"],
                },
            ],
        }
    )

    ranking = contract.goal_map()["ranking.alpha"]
    assert ranking.population_goal_ids == ["detail.rows"]
    assert required_goal_ids(contract) == [
        "detail.rows",
        "metric.alpha",
        "ranking.alpha",
    ]


def test_multi_clause_detail_then_rank_cannot_drop_either_operation() -> None:
    question = "我想看最近7天的订单明细，然后告诉我这里面退款最多的三单"

    class RequiredKindsVerifier:
        def verify(self, contract):
            declared_kinds = {goal.kind for goal in contract.goals}
            return [
                GoalContractIssue(
                    code="QUESTION_REQUIRED_KIND_MISSING",
                    message=f"required typed kind {kind} was not declared",
                )
                for kind in ("DETAIL", "RANKING")
                if kind not in declared_kinds
            ]

    missing_ranking = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {"goalId": "detail.orders", "kind": "detail", "label": "订单明细"},
                {
                    "goalId": "time.7d",
                    "kind": "time_window",
                    "label": "最近7天",
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": ["detail.orders"],
                },
            ],
        },
        question_verifier=RequiredKindsVerifier(),
    )
    missing_detail = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {"goalId": "metric.refund", "kind": "metric", "label": "退款金额"},
                {
                    "goalId": "ranking.refund",
                    "kind": "ranking",
                    "label": "退款最多三单",
                    "metricGoalIds": ["metric.refund"],
                    "limit": 3,
                    "populationScope": "VERIFIED_PREDICATE_SCOPE",
                },
                {
                    "goalId": "time.7d",
                    "kind": "time_window",
                    "label": "最近7天",
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": ["ranking.refund"],
                },
            ],
        },
        question_verifier=RequiredKindsVerifier(),
    )

    assert missing_ranking.valid is False
    assert missing_detail.valid is False
    assert {item.code for item in missing_ranking.issues} == {
        "QUESTION_REQUIRED_KIND_MISSING"
    }
    assert {item.code for item in missing_detail.issues} == {
        "QUESTION_REQUIRED_KIND_MISSING"
    }


def test_typed_same_turn_population_scope_requires_population_goal_reference() -> None:
    question = "我想看最近7天的订单明细，然后告诉我这里面退款最多的三单"
    result = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {"goalId": "metric.refund", "kind": "metric", "label": "退款金额"},
                {"goalId": "detail.orders", "kind": "detail", "label": "订单明细"},
                {
                    "goalId": "ranking.refund",
                    "kind": "ranking",
                    "label": "退款最多三单",
                    "metricGoalIds": ["metric.refund"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                },
                {
                    "goalId": "time.7d",
                    "kind": "time_window",
                    "label": "最近7天",
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": ["detail.orders", "ranking.refund"],
                },
            ],
        }
    )

    assert "RANKING_POPULATION_GOALS_REQUIRED" in {
        item.code for item in result.issues
    }


def test_complete_same_turn_detail_rank_contract_passes_structural_audit() -> None:
    question = "我想看最近7天的订单明细，然后告诉我这里面退款最多的三单"
    result = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {
                    "goalId": "metric.refund",
                    "kind": "metric",
                    "label": "退款金额",
                    "sourceSpans": ["退款最多"],
                },
                {
                    "goalId": "detail.orders",
                    "kind": "detail",
                    "label": "订单明细",
                    "sourceSpans": ["订单明细"],
                },
                {
                    "goalId": "ranking.refund",
                    "kind": "ranking",
                    "label": "退款最多三单",
                    "sourceSpans": ["这里面退款最多的三单"],
                    "metricGoalIds": ["metric.refund"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                    "populationGoalIds": ["detail.orders"],
                },
                {
                    "goalId": "time.7d",
                    "kind": "time_window",
                    "label": "最近7天",
                    "sourceSpans": ["最近7天"],
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": ["detail.orders", "ranking.refund"],
                },
            ],
        }
    )

    assert result.valid is True, [item.model_dump() for item in result.issues]


def test_external_verifier_can_enforce_question_specific_scope_and_limit() -> None:
    question = "这里面退款最多的三单"

    class ExpectedRankingVerifier:
        def verify(self, contract):
            ranking = contract.goal_map().get("ranking.refund")
            if not isinstance(ranking, RankingQuestionGoal):
                return [
                    GoalContractIssue(
                        code="QUESTION_RANKING_DECLARATION_MISSING",
                        message="the expected typed ranking declaration is missing",
                    )
                ]
            issues = []
            if ranking.limit != 3:
                issues.append(
                    GoalContractIssue(
                        code="QUESTION_RANKING_LIMIT_MISMATCH",
                        message="the typed ranking limit does not match the verified question semantics",
                    )
                )
            if ranking.population_scope != "VERIFIED_PREDICATE_SCOPE":
                issues.append(
                    GoalContractIssue(
                        code="QUESTION_POPULATION_SCOPE_MISMATCH",
                        message="the typed population scope does not match the verified question semantics",
                    )
                )
            return issues

    unsafe = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {"goalId": "metric.refund", "kind": "metric", "label": "退款金额"},
                {
                    "goalId": "ranking.refund",
                    "kind": "ranking",
                    "label": "退款排名",
                    "metricGoalIds": ["metric.refund"],
                    "limit": 5,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
            ],
        },
        question_verifier=ExpectedRankingVerifier(),
    )
    safe = validate_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {"goalId": "metric.refund", "kind": "metric", "label": "退款金额"},
                {
                    "goalId": "ranking.refund",
                    "kind": "ranking",
                    "label": "退款最多三单",
                    "metricGoalIds": ["metric.refund"],
                    "limit": 3,
                    "populationScope": "VERIFIED_PREDICATE_SCOPE",
                },
            ],
        },
        question_verifier=ExpectedRankingVerifier(),
    )

    unsafe_codes = {item.code for item in unsafe.issues}
    assert "QUESTION_POPULATION_SCOPE_MISMATCH" in unsafe_codes
    assert "QUESTION_RANKING_LIMIT_MISMATCH" in unsafe_codes
    assert safe.valid is True, [item.model_dump() for item in safe.issues]


def test_goal_source_span_must_be_verbatim_from_question() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "最近7天订单明细",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "detail",
                    "label": "订单明细",
                    "sourceSpans": ["最近30天"],
                },
                {
                    "goalId": "time.7d",
                    "kind": "time_window",
                    "label": "最近7天",
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": ["detail.orders"],
                },
            ],
        }
    )

    assert "GOAL_SOURCE_SPAN_NOT_IN_QUESTION" in {
        item.code for item in result.issues
    }


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
    assert any(issue.code == "GOAL_SEMANTIC_EVIDENCE_UNCOVERED" for issue in mismatched.issues)

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
    with pytest.raises(ValueError) as exc_info:
        declare_verified_artifact_goal_coverage(contract, failed_artifact, ["metric.alpha"])
    assert "verified query artifact" in str(exc_info.value)


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


def anomaly_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "最近7天 GMV、退款金额、催单工单量分别是多少，哪个环节最异常？",
            "goals": [
                {
                    "goalId": "metric.gmv",
                    "kind": "metric",
                    "label": "GMV",
                },
                {
                    "goalId": "metric.refund_amount",
                    "kind": "metric",
                    "label": "退款金额",
                },
                {
                    "goalId": "metric.urge_ticket_count",
                    "kind": "metric",
                    "label": "催单工单量",
                },
                {
                    "goalId": "time.last_7_days",
                    "kind": "time_window",
                    "label": "最近7天",
                    "timeExpression": "最近7天",
                    "appliesToGoalIds": [
                        "metric.gmv",
                        "metric.refund_amount",
                        "metric.urge_ticket_count",
                    ],
                },
                {
                    "goalId": "comparison.most_anomalous",
                    "kind": "comparison",
                    "label": "哪个环节最异常",
                    "comparisonType": "anomaly",
                    "leftGoalIds": ["metric.gmv", "metric.refund_amount"],
                    "rightGoalIds": ["metric.urge_ticket_count"],
                },
            ],
        }
    )


def test_scalar_artifact_cannot_claim_anomaly_analysis_is_covered() -> None:
    """Regression: three returned scalars are not an anomaly proof."""

    contract = anomaly_contract()
    goal_ids = [goal.goal_id for goal in contract.goals]
    artifact = SimpleNamespace(
        artifact_id="artifact-three-scalars",
        verified_evidence=SimpleNamespace(passed=True),
        contract=SimpleNamespace(
            time_range=SimpleNamespace(
                label="最近7天",
                start_date="2026-07-14",
                end_date="2026-07-20",
                days=7,
                timezone="Asia/Shanghai",
                explicit=True,
                calendar_anchor_policy="runtime_current_date",
                window_role="primary",
                kind="rolling",
            ),
            binding_hints=SimpleNamespace(time_expression="最近7天"),
        ),
    )
    result = GoalCoverageVerifier().verify(
        contract,
        [declare_verified_artifact_goal_coverage(contract, artifact, goal_ids)],
    )

    assert result.finalization_allowed is False
    assert result.covered_goal_ids == [
        "metric.gmv",
        "metric.refund_amount",
        "metric.urge_ticket_count",
        "time.last_7_days",
    ]
    assert result.missing_required_goal_ids == ["comparison.most_anomalous"]
    assert result.resolution_by_goal_id.get("comparison.most_anomalous") is None
    assert "GOAL_TYPED_PROOF_REQUIRED" in {issue.code for issue in result.issues}


def test_independent_filter_windows_accept_branch_local_primary_role() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "最近7天订单明细和最近10天退款明细",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "detail",
                    "label": "订单明细",
                },
                {
                    "goalId": "time.orders",
                    "kind": "time_window",
                    "label": "订单时间窗",
                    "timeExpression": "最近7天",
                    "windowRole": "filter",
                    "appliesToGoalIds": ["detail.orders"],
                },
                {
                    "goalId": "detail.refunds",
                    "kind": "detail",
                    "label": "退款明细",
                },
                {
                    "goalId": "time.refunds",
                    "kind": "time_window",
                    "label": "退款时间窗",
                    "timeExpression": "最近10天",
                    "windowRole": "filter",
                    "appliesToGoalIds": ["detail.refunds"],
                },
            ],
        }
    )

    assert goal_contract_module._time_window_roles_equivalent(
        "filter",
        "primary",
        goal_map=contract.goal_map(),
    ) is True
    assert goal_contract_module._time_window_roles_equivalent(
        "detail_scope",
        "primary",
        goal_map=contract.goal_map(),
    ) is True

    comparison_contract = contract.model_copy(
        update={
            "goals": [
                *contract.goals[:-1],
                contract.goals[-1].model_copy(update={"window_role": "comparison"}),
            ]
        },
        deep=True,
    )
    assert goal_contract_module._time_window_roles_equivalent(
        "filter",
        "primary",
        goal_map=comparison_contract.goal_map(),
    ) is False


def test_anomaly_proof_requires_baseline_and_normalization() -> None:
    contract = anomaly_contract()
    goal_ids = [goal.goal_id for goal in contract.goals]
    result = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-unscaled-comparison",
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
                covered_goal_ids=goal_ids,
                verification_passed=True,
                goal_resolutions=[
                    {
                        "goalId": "time.last_7_days",
                        "goalKind": "time_window",
                        "resolution": "proved",
                        "proofType": "VERIFIED_QUERY_TIME_RANGE",
                        "timeExpression": "最近7天",
                        "start": "2026-07-14",
                        "end": "2026-07-20",
                        "timezone": "Asia/Shanghai",
                        "days": 7,
                        "label": "最近7天",
                        "explicit": True,
                        "calendarAnchorPolicy": "runtime_current_date",
                        "windowRole": "primary",
                        "timeRangeKind": "rolling",
                    },
                    {
                        "goalId": "comparison.most_anomalous",
                        "goalKind": "comparison",
                        "resolution": "proved",
                        "operandGoalIds": [
                            "metric.gmv",
                            "metric.refund_amount",
                            "metric.urge_ticket_count",
                        ],
                        "comparisonMethod": "largest absolute scalar",
                        "resultRef": "artifact:comparison-result",
                    }
                ],
            )
        ],
    )

    assert result.finalization_allowed is False
    assert "comparison.most_anomalous" not in result.covered_goal_ids
    assert {issue.code for issue in result.issues} >= {
        "ANOMALY_PROOF_BASELINE_MISSING",
        "ANOMALY_PROOF_NORMALIZATION_MISSING",
    }


def test_explicit_insufficient_evidence_resolves_but_does_not_prove_goal() -> None:
    contract = anomaly_contract()
    primitive_goal_ids = [goal.goal_id for goal in contract.goals if goal.kind != "COMPARISON"]
    result = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-no-baseline",
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
                covered_goal_ids=[
                    *primitive_goal_ids,
                    "comparison.most_anomalous",
                ],
                verification_passed=True,
                goal_resolutions=[
                    {
                        "goalId": "time.last_7_days",
                        "goalKind": "time_window",
                        "resolution": "proved",
                        "proofType": "VERIFIED_QUERY_TIME_RANGE",
                        "timeExpression": "最近7天",
                        "start": "2026-07-14",
                        "end": "2026-07-20",
                        "timezone": "Asia/Shanghai",
                        "days": 7,
                        "label": "最近7天",
                        "explicit": True,
                        "calendarAnchorPolicy": "runtime_current_date",
                        "windowRole": "primary",
                        "timeRangeKind": "rolling",
                    },
                    {
                        "goalId": "comparison.most_anomalous",
                        "goalKind": "comparison",
                        "resolution": "insufficient_evidence",
                        "reason": "没有上一周期或标准化基线，不能比较不同单位指标的异常程度",
                        "evidenceRefs": ["gap:comparable-baseline-missing"],
                    }
                ],
            )
        ],
    )

    assert result.finalization_allowed is True
    assert result.passed is False
    assert result.missing_required_goal_ids == []
    assert result.unproved_required_goal_ids == ["comparison.most_anomalous"]
    assert result.insufficient_evidence_goal_ids == ["comparison.most_anomalous"]
    assert "comparison.most_anomalous" in result.resolved_goal_ids
    assert "comparison.most_anomalous" not in result.covered_goal_ids
    assert "comparison.most_anomalous" in result.claimed_covered_goal_ids
    assert result.resolution_by_goal_id["comparison.most_anomalous"] == "INSUFFICIENT_EVIDENCE"
    assert "INSUFFICIENT_EVIDENCE_CANNOT_PROVE_GOAL" in {issue.code for issue in result.issues}


def dependency_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "查销量最高商品，再查这些商品的退款明细",
            "goals": [
                {
                    "goalId": "metric.sales",
                    "kind": "metric",
                    "label": "销量",
                    "metricRefId": "metric:sales",
                    "required": False,
                },
                {
                    "goalId": "ranking.top_products",
                    "kind": "ranking",
                    "label": "销量最高商品",
                    "required": False,
                    "metricGoalIds": ["metric.sales"],
                    "limit": 1,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
                {
                    "goalId": "chain.product_refunds",
                    "kind": "dependency",
                    "label": "商品集合传给退款查询",
                    "required": False,
                    "upstreamGoalIds": ["ranking.top_products"],
                    "downstreamGoalIds": ["detail.refund_rows"],
                },
                {
                    "goalId": "detail.refund_rows",
                    "kind": "detail",
                    "label": "对应退款明细",
                },
            ],
        }
    )


def test_dependency_proof_without_artifact_lineage_is_rejected() -> None:
    contract = dependency_contract()
    result = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-flat-result",
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
                covered_goal_ids=[goal.goal_id for goal in contract.goals],
                verification_passed=True,
                goal_resolutions=[
                    {
                        "goalId": "chain.product_refunds",
                        "goalKind": "dependency",
                        "resolution": "proved",
                    }
                ],
            )
        ],
    )

    assert result.finalization_allowed is False
    assert {issue.code for issue in result.issues} >= {
        "DEPENDENCY_PROOF_UPSTREAM_ARTIFACT_MISSING",
        "DEPENDENCY_PROOF_DOWNSTREAM_ARTIFACT_MISSING",
        "DEPENDENCY_PROOF_LINEAGE_MISSING",
    }


def test_dependency_proof_with_verified_lineage_is_accepted() -> None:
    contract = dependency_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)
    result = GoalCoverageVerifier().verify(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-top-products",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=["metric.sales", "ranking.top_products"],
                verification_passed=True,
                evidence_refs=["metric:sales"],
                goal_resolutions=[
                    {
                        "goalId": "ranking.top_products",
                        "goalKind": "ranking",
                        "resolution": "proved",
                        "orderByGoalIds": ["metric.sales"],
                        "rankingMetricRefId": "metric:sales",
                        "direction": "DESC",
                        "limit": 1,
                        "rowSetRef": "artifact-top-products",
                        "populationScope": "ALL_MATCHING_ROWS",
                        "details": {"metricBindingMode": "SEMANTIC_REF_MATCH"},
                    }
                ],
            ),
            VerifiedArtifactGoalCoverage(
                artifact_id="artifact-refund-details",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=[
                    "chain.product_refunds",
                    "detail.refund_rows",
                ],
                verification_passed=True,
                goal_resolutions=[
                    {
                        "goalId": "chain.product_refunds",
                        "goalKind": "dependency",
                        "resolution": "proved",
                        "upstreamArtifactIds": ["artifact-top-products"],
                        "downstreamArtifactIds": ["artifact-refund-details"],
                        "lineageRefs": ["entity-set:verified:top-products"],
                    },
                    {
                        "goalId": "detail.refund_rows",
                        "goalKind": "detail",
                        "resolution": "proved",
                        "outputFields": ["refund_id"],
                        "rowSetRef": "artifact-refund-details",
                        "rowCount": 3,
                    }
                ],
            ),
        ],
    )

    assert result.finalization_allowed is True
    assert result.passed is True
    assert result.covered_goal_ids == [
        "metric.sales",
        "ranking.top_products",
        "chain.product_refunds",
        "detail.refund_rows",
    ]


def test_rule_detail_ranking_and_analysis_goal_kinds_are_strictly_normalized() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "Return rule guidance, rows, top 3, and analysis",
            "metrics": [{"goalId": "metric.alpha", "label": "alpha"}],
            "rules": [{"goalId": "rule.guidance", "label": "guidance"}],
            "details": [
                {
                    "goalId": "detail.rows",
                    "label": "rows",
                    "inputGoalIds": ["metric.alpha"],
                }
            ],
            "rankings": [
                {
                    "goalId": "ranking.top3",
                    "label": "top 3",
                    "metricGoalIds": ["metric.alpha"],
                    "limit": 3,
                    "populationScope": "ALL_MATCHING_ROWS",
                }
            ],
            "analyses": [
                {
                    "goalId": "analysis.alpha",
                    "label": "analysis",
                    "analysisType": "diagnostic",
                    "inputGoalIds": ["metric.alpha"],
                }
            ],
        }
    )

    assert isinstance(contract.goal_map()["rule.guidance"], RuleQuestionGoal)
    assert isinstance(contract.goal_map()["detail.rows"], DetailQuestionGoal)
    assert isinstance(contract.goal_map()["ranking.top3"], RankingQuestionGoal)
    assert isinstance(contract.goal_map()["analysis.alpha"], AnalysisQuestionGoal)

    with pytest.raises(GoalContractValidationError):
        parse_original_question_goal_contract(
            {
                "question": "bad strict rule",
                "rules": [
                    {
                        "goalId": "rule.bad",
                        "label": "bad",
                        "inventedField": "must be rejected",
                    }
                ],
            }
        )


def test_ranking_inputs_must_reference_metric_and_dimension_goal_kinds() -> None:
    result = validate_original_question_goal_contract(
        {
            "question": "退款明细并找金额最高前5单",
            "goals": [
                {
                    "goalId": "detail.refunds",
                    "kind": "DETAIL",
                    "label": "退款明细",
                },
                {
                    "goalId": "metric.refund_amount",
                    "kind": "METRIC",
                    "label": "退款金额",
                },
                {
                    "goalId": "ranking.top5",
                    "kind": "RANKING",
                    "label": "退款最高前5单",
                    "metricGoalIds": ["metric.refund_amount"],
                    "dimensionGoalIds": ["detail.refunds"],
                    "direction": "DESC",
                    "limit": 5,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
            ],
        }
    )

    assert result.valid is False
    assert "RANKING_DIMENSION_GOAL_KIND_INVALID" in {issue.code for issue in result.issues}


def test_goal_dependency_closure_covers_complex_bi_lineage_transitively() -> None:
    contract = OriginalQuestionGoalContract(
        question="rank, compare, then analyze",
        goals=[
            MetricQuestionGoal(
                goal_id="metric.raw",
                label="raw metric",
            ),
            MetricQuestionGoal(
                goal_id="metric.primary",
                label="primary metric",
                depends_on_goal_ids=["metric.raw"],
            ),
            MetricQuestionGoal(
                goal_id="metric.baseline",
                label="baseline metric",
            ),
            DimensionQuestionGoal(
                goal_id="dimension.entity",
                label="entity",
            ),
            RankingQuestionGoal(
                goal_id="ranking.top",
                label="top entities",
                population_scope="SAME_AS_GOAL",
                population_goal_ids=["metric.primary"],
                metric_goal_ids=["metric.primary"],
                dimension_goal_ids=["dimension.entity"],
                limit=3,
            ),
            ComparisonQuestionGoal(
                goal_id="comparison.delta",
                label="ranking versus baseline",
                left_goal_ids=["ranking.top"],
                right_goal_ids=["metric.baseline"],
            ),
            AnalysisQuestionGoal(
                goal_id="analysis.diagnosis",
                label="diagnosis",
                input_goal_ids=["comparison.delta"],
                baseline_goal_ids=["metric.baseline"],
            ),
        ],
    )

    assert goal_dependency_closure(
        contract,
        ["analysis.diagnosis"],
    ) == {
        "comparison.delta",
        "ranking.top",
        "metric.primary",
        "metric.raw",
        "metric.baseline",
        "dimension.entity",
    }
