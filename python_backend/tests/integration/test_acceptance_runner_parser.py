from __future__ import annotations

import hashlib

import pytest

from scripts.acceptance_runner import (
    SUPPLEMENTAL_CASES,
    USER_ACCEPTANCE_CASES,
    AcceptanceCase,
    parse_acceptance_response,
)


EXPECTED_USER_QUESTIONS = [
    "最近30天订单量和退款金额分别是多少，并分析退款有没有影响订单表现。",
    "最近10天保证金缴纳流水和退款金额有没有明显关系？",
    "上个月销售额最高的前3个商品，以及这些商品最近7天退款量是多少？",
    "最近7天按工单状态统计工单量，再看催单工单量有没有升高。",
    "最近10天商品审核拒绝明细给我看一下，再告诉我拒绝最多的商品有哪些。",
    "最近30天订单量、退货量、发货超时订单量为什么一起上升？",
    "最近7天履约量、发货超时订单量和催单工单量分别是多少，并分析是不是履约问题导致催单增加。",
    "最近10天退款最多的商品有哪些？再给我看这些商品对应的退款明细。",
    "最近7天支付订单量和交易成功订单量分别是多少，差异大不大？",
    "最近30天保证金充值流水、申诉次数和处罚次数分别是多少，有没有异常？",
    "最近7天订单明细和退款明细都给我看一下，并找出退款金额最高的前3单。",
    "最近7天商品审核通过量、审核拒绝量和上架商品量分别是多少，帮我分析商品侧有没有问题。",
    "最近10天工单明细给我看一下，再按工单状态统计数量。",
    "最近30天 GMV 下降是不是和退款金额、发货超时订单量有关？",
    "最近7天优惠金额、优惠订单量和 GMV 分别是多少，优惠有没有带来成交提升？",
]


def _successful_payload(answer: str = "已验证结论") -> dict:
    fingerprint = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return {
        "answer": answer,
        "categoryName": "经营画像 / 客服理赔",
        "debugTrace": {
            "harness": {
                "topicRouting": {"primaryTopic": "经营画像", "relatedTopics": ["客服理赔"]},
                "originalQuestionGoalContract": {
                    "goals": [
                        {"goalId": "metric.orders", "kind": "METRIC"},
                        {"goalId": "time.main", "kind": "TIME_WINDOW"},
                        {"goalId": "analysis.main", "kind": "ANALYSIS"},
                    ]
                },
                "queryBranches": [
                    {
                        "queryId": "orders",
                        "goalIds": ["metric.orders", "time.main"],
                        "topicScope": ["经营画像"],
                        "status": "VERIFIED",
                        "verifiedArtifactIds": ["query_artifact_orders"],
                    }
                ],
                "verifiedQueryArtifactIds": ["query_artifact_orders"],
                "verifiedAnalysisArtifactIds": ["analysis_artifact_main"],
                "runtimeBudget": {
                    "elapsedMs": 1234.5,
                    "usage": {"dorisQueries": 1, "llmCalls": 3, "toolCalls": 8},
                    "stages": {
                        "routing.topic": {"totalDurationMs": 12.5},
                        "core.react_loop": {"totalDurationMs": 1100.0},
                    },
                },
                "goalCoverage": {
                    "passed": True,
                    "finalizationAllowed": True,
                    "requiredGoalIds": [
                        "metric.orders",
                        "time.main",
                        "analysis.main",
                    ],
                    "coveredGoalIds": [
                        "metric.orders",
                        "time.main",
                        "analysis.main",
                    ],
                    "resolvedGoalIds": [
                        "metric.orders",
                        "time.main",
                        "analysis.main",
                    ],
                    "artifactIds": ["query_artifact_orders", "analysis_artifact_main"],
                    "resolutionByGoalId": {
                        "metric.orders": "PROVED",
                        "time.main": "PROVED",
                        "analysis.main": "PROVED",
                    },
                    "resolutionProofTypesByGoalId": {
                        "metric.orders": ["KERNEL_VERIFIED_PRIMITIVE_RESULT"],
                        "time.main": ["KERNEL_VERIFIED_PRIMITIVE_RESULT"],
                        "analysis.main": ["DETERMINISTIC_DERIVED_ANALYSIS"]
                    },
                    "resolutionArtifactIdsByGoalId": {
                        "metric.orders": ["query_artifact_orders"],
                        "time.main": ["query_artifact_orders"],
                        "analysis.main": ["analysis_artifact_main"],
                    },
                    "resolutionEvidenceRefsByGoalId": {
                        "metric.orders": ["semantic:orders"],
                        "time.main": ["semantic:orders"],
                        "analysis.main": ["analysis:evidence"],
                    },
                },
                "answerCoverage": {
                    "passed": True,
                    "source": "run_skill",
                    "answerFingerprint": fingerprint,
                    "requiredGoalIds": [
                        "metric.orders",
                        "time.main",
                        "analysis.main",
                    ],
                    "mappedGoalIds": [
                        "metric.orders",
                        "time.main",
                        "analysis.main",
                    ],
                    "bindings": [
                        {
                            "goalId": "metric.orders",
                            "resolution": "PROVED",
                            "answerText": answer,
                            "artifactIds": ["query_artifact_orders"],
                            "evidenceRefs": ["semantic:orders"],
                            "renderer": "VERIFIED_QUERY_RENDERER",
                        },
                        {
                            "goalId": "time.main",
                            "resolution": "PROVED",
                            "answerText": answer,
                            "artifactIds": ["query_artifact_orders"],
                            "evidenceRefs": ["semantic:orders"],
                            "renderer": "VERIFIED_QUERY_RENDERER",
                        },
                        {
                            "goalId": "analysis.main",
                            "resolution": "PROVED",
                            "answerText": answer,
                            "artifactIds": ["analysis_artifact_main"],
                            "evidenceRefs": ["analysis:evidence"],
                            "renderer": "VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
                        }
                    ],
                },
            }
        },
    }


def _single_node_population_payload() -> dict:
    answer = "已验证排名"
    artifact_id = "query_artifact_population"
    goal_ids = ["detail.population", "metric.refund", "ranking.top"]
    evidence_by_goal = {
        "detail.population": ["semantic:detail"],
        "metric.refund": ["semantic:refund"],
        "ranking.top": ["query-artifact:query_artifact_population"],
    }
    proof_by_goal = {
        "detail.population": ["VERIFIED_QUERY_ROW_SET"],
        "metric.refund": ["KERNEL_VERIFIED_PRIMITIVE_RESULT"],
        "ranking.top": ["VERIFIED_ORDERED_ROW_SET"],
    }
    renderer_by_goal = {
        "detail.population": "VERIFIED_DETAIL_RENDERER",
        "metric.refund": "VERIFIED_QUERY_RENDERER",
        "ranking.top": "VERIFIED_RANKING_RENDERER",
    }
    return {
        "answer": answer,
        "debugTrace": {
            "harness": {
                "originalQuestionGoalContract": {
                    "goals": [
                        {"goalId": "detail.population", "kind": "DETAIL"},
                        {"goalId": "metric.refund", "kind": "METRIC"},
                        {
                            "goalId": "ranking.top",
                            "kind": "RANKING",
                            "populationScope": "SAME_AS_GOAL",
                            "populationGoalIds": ["detail.population"],
                        },
                    ]
                },
                "queryBranches": [
                    {
                        "queryId": "population_rank",
                        "goalIds": goal_ids,
                        "topicScope": ["topic"],
                        "status": "VERIFIED",
                        "verifiedArtifactIds": [artifact_id],
                    }
                ],
                "verifiedQueryArtifactIds": [artifact_id],
                "goalCoverage": {
                    "passed": True,
                    "finalizationAllowed": True,
                    "requiredGoalIds": goal_ids,
                    "coveredGoalIds": goal_ids,
                    "resolvedGoalIds": goal_ids,
                    "artifactIds": [artifact_id],
                    "resolutionByGoalId": {
                        goal_id: "PROVED" for goal_id in goal_ids
                    },
                    "resolutionProofTypesByGoalId": proof_by_goal,
                    "resolutionArtifactIdsByGoalId": {
                        goal_id: [artifact_id] for goal_id in goal_ids
                    },
                    "resolutionEvidenceRefsByGoalId": evidence_by_goal,
                },
                "answerCoverage": {
                    "passed": True,
                    "source": "compose_verified_answer",
                    "answerFingerprint": hashlib.sha256(
                        answer.encode("utf-8")
                    ).hexdigest(),
                    "requiredGoalIds": goal_ids,
                    "mappedGoalIds": goal_ids,
                    "bindings": [
                        {
                            "goalId": goal_id,
                            "resolution": "PROVED",
                            "answerText": answer,
                            "artifactIds": [artifact_id],
                            "evidenceRefs": evidence_by_goal[goal_id],
                            "renderer": renderer_by_goal[goal_id],
                        }
                        for goal_id in goal_ids
                    ],
                },
            }
        },
    }


def test_catalogue_contains_the_exact_user_15_and_six_supplemental_shapes() -> None:
    assert [case.question for case in USER_ACCEPTANCE_CASES] == EXPECTED_USER_QUESTIONS
    assert len(SUPPLEMENTAL_CASES) == 6
    assert {case.case_id for case in SUPPLEMENTAL_CASES} == {
        "base_single_metric",
        "base_same_table_two_metrics",
        "base_cross_table",
        "base_multiple_details",
        "base_topn_entity_chain",
        "base_rule",
    }


def test_parser_summarizes_routing_goals_branches_artifacts_usage_and_coverage() -> None:
    case = USER_ACCEPTANCE_CASES[0]
    summary = parse_acceptance_response(case, _successful_payload(), status_code=200, elapsed_ms=1400.0)

    assert summary["status"] == "PASS"
    assert summary["topicRouting"]["primaryTopic"] == "经营画像"
    assert summary["goalKinds"] == ["ANALYSIS", "METRIC", "TIME_WINDOW"]
    assert summary["branches"][0]["topicScope"] == ["经营画像"]
    assert summary["branches"][0]["status"] == "VERIFIED"
    assert summary["artifactIds"] == ["query_artifact_orders", "analysis_artifact_main"]
    assert summary["usage"] == {"dorisQueries": 1, "llmCalls": 3, "toolCalls": 8}
    assert summary["stageTimingsMs"]["core.react_loop"] == 1100.0
    assert summary["elapsedMs"] == 1400.0
    assert summary["goalCoverage"]["passed"] is True
    assert summary["answerCoverage"]["passed"] is True
    assert summary["operationalFailure"] == {}
    assert summary["architecturePassed"] is True


def test_verified_single_contract_does_not_require_a_declared_execution_graph() -> None:
    payload = _successful_payload()
    payload["debugTrace"]["harness"]["queryBranches"] = []

    summary = parse_acceptance_response(USER_ACCEPTANCE_CASES[0], payload)

    assert summary["status"] == "PASS"
    assert summary["branches"] == []


def test_population_scope_can_be_proved_inside_one_query_node() -> None:
    case = AcceptanceCase(
        "single_population",
        "typed population",
        "test",
        ("DETAIL", "METRIC", "RANKING"),
        requires_entity_lineage=True,
    )

    summary = parse_acceptance_response(case, _single_node_population_payload())

    assert summary["status"] == "PASS"
    assert len(summary["branches"]) == 1


@pytest.mark.parametrize(
    ("mutation", "expected_violation"),
    [
        (
            lambda harness: harness["originalQuestionGoalContract"]["goals"][2].update(
                populationScope=""
            ),
            "RANKING_POPULATION_SCOPE_NOT_DECLARED",
        ),
        (
            lambda harness: harness["goalCoverage"]["resolutionProofTypesByGoalId"].update(
                {"ranking.top": ["KERNEL_VERIFIED_PRIMITIVE_RESULT"]}
            ),
            "RANKING_REQUIRES_VERIFIED_POPULATION_PROOF",
        ),
        (
            lambda harness: harness["goalCoverage"]["resolutionEvidenceRefsByGoalId"].update(
                {"ranking.top": []}
            ),
            "RANKING_POPULATION_LINEAGE_NOT_ATTESTED",
        ),
    ],
)
def test_population_acceptance_fails_closed_without_scope_and_lineage_proof(
    mutation,
    expected_violation: str,
) -> None:
    payload = _single_node_population_payload()
    mutation(payload["debugTrace"]["harness"])

    summary = parse_acceptance_response(
        AcceptanceCase(
            "population_invalid",
            "typed population",
            "test",
            ("DETAIL", "METRIC", "RANKING"),
            requires_entity_lineage=True,
        ),
        payload,
    )

    assert summary["status"] == "FAIL"
    assert expected_violation in summary["architectureViolations"]


def test_analysis_accepts_typed_insufficiency_instead_of_a_proved_analysis_artifact() -> None:
    payload = _successful_payload("证据不足：样本量不足。")
    harness = payload["debugTrace"]["harness"]
    harness["verifiedAnalysisArtifactIds"] = []
    harness["goalCoverage"]["resolutionByGoalId"]["analysis.main"] = "INSUFFICIENT_EVIDENCE"
    harness["goalCoverage"]["resolutionProofTypesByGoalId"]["analysis.main"] = ["TYPED_EVIDENCE_GAP"]
    harness["goalCoverage"]["resolutionArtifactIdsByGoalId"]["analysis.main"] = ["analysis_gap_artifact"]
    harness["goalCoverage"]["resolutionEvidenceRefsByGoalId"].update(
        {"analysis.main": ["analysis:gap:sample"]}
    )
    harness["goalCoverage"]["insufficiencyReasonByGoalId"] = {"analysis.main": "样本量不足"}
    harness["goalCoverage"]["artifactIds"] = ["query_artifact_orders", "analysis_gap_artifact"]
    harness["answerCoverage"]["bindings"] = [
        *harness["answerCoverage"]["bindings"][:2],
        {
            "goalId": "analysis.main",
            "resolution": "INSUFFICIENT_EVIDENCE",
            "answerText": payload["answer"],
            "renderer": "VERIFIED_INSUFFICIENCY_RENDERER",
            "insufficiencyRef": "analysis:gap:sample",
        }
    ]
    harness["answerCoverage"]["answerFingerprint"] = hashlib.sha256(
        payload["answer"].encode("utf-8")
    ).hexdigest()

    summary = parse_acceptance_response(USER_ACCEPTANCE_CASES[0], payload)

    assert summary["status"] == "PASS"
    assert summary["analysisArtifactIds"] == []
    assert summary["architecturePassed"] is True


def test_entity_chain_accepts_verified_cross_node_lineage_without_prescribing_node_count() -> None:
    case = AcceptanceCase(
        "chain",
        "TopN 后查实体",
        "test",
        ("RANKING", "ENTITY", "DEPENDENCY", "DETAIL"),
        requires_entity_lineage=True,
    )
    payload = _successful_payload()
    harness = payload["debugTrace"]["harness"]
    harness["originalQuestionGoalContract"] = {
        "goals": [
            {
                "goalId": "rank.top",
                "kind": "RANKING",
                "populationScope": "ALL_MATCHING_ROWS",
            },
            {"goalId": "entity.top", "kind": "ENTITY"},
            {"goalId": "detail.rows", "kind": "DETAIL"},
            {
                "goalId": "dependency.details",
                "kind": "DEPENDENCY",
                "dependencyType": "ENTITY_CHAIN",
                "artifactKind": "VERIFIED_ENTITY_SET",
                "upstreamGoalIds": ["entity.top"],
                "downstreamGoalIds": ["detail.rows"],
            },
        ]
    }
    harness["queryBranches"] = [
        {
            "queryId": "top",
            "goalIds": ["rank.top", "entity.top"],
            "topicScope": ["电商退货"],
            "status": "VERIFIED",
            "verifiedArtifactIds": ["query_artifact_top"],
        },
        {
            "queryId": "details",
            "goalIds": ["detail.rows"],
            "topicScope": ["电商退货"],
            "status": "VERIFIED",
            "dependencyQueryIds": ["top"],
            "verifiedArtifactIds": ["query_artifact_details"],
        },
    ]
    harness["verifiedQueryArtifactIds"] = ["query_artifact_top", "query_artifact_details"]
    harness["verifiedAnalysisArtifactIds"] = []
    harness["goalCoverage"] = {
        "passed": True,
        "finalizationAllowed": True,
        "requiredGoalIds": [
            "rank.top",
            "entity.top",
            "detail.rows",
            "dependency.details",
        ],
        "coveredGoalIds": [
            "rank.top",
            "entity.top",
            "detail.rows",
            "dependency.details",
        ],
        "resolvedGoalIds": [
            "rank.top",
            "entity.top",
            "detail.rows",
            "dependency.details",
        ],
        "artifactIds": ["query_artifact_top", "query_artifact_details"],
        "resolutionByGoalId": {
            "rank.top": "PROVED",
            "entity.top": "PROVED",
            "detail.rows": "PROVED",
            "dependency.details": "PROVED",
        },
        "resolutionProofTypesByGoalId": {
            "rank.top": ["VERIFIED_ORDERED_ROW_SET"],
            "entity.top": ["KERNEL_VERIFIED_PRIMITIVE_RESULT"],
            "detail.rows": ["VERIFIED_QUERY_ROW_SET"],
            "dependency.details": ["VERIFIED_ARTIFACT_LINEAGE"],
        },
        "resolutionArtifactIdsByGoalId": {
            "rank.top": ["query_artifact_top"],
            "entity.top": ["query_artifact_top"],
            "detail.rows": ["query_artifact_details"],
            "dependency.details": ["query_artifact_details"],
        },
        "resolutionEvidenceRefsByGoalId": {
            "rank.top": ["query-artifact:query_artifact_top"],
            "entity.top": ["semantic:entity"],
            "detail.rows": ["semantic:detail"],
            "dependency.details": ["lineage:entity-set"],
        },
    }
    answer = payload["answer"]
    harness["answerCoverage"] = {
        "passed": True,
        "source": "compose_verified_answer",
        "answerFingerprint": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "requiredGoalIds": [
            "rank.top",
            "entity.top",
            "detail.rows",
            "dependency.details",
        ],
        "mappedGoalIds": [
            "rank.top",
            "entity.top",
            "detail.rows",
            "dependency.details",
        ],
        "bindings": [
            {
                "goalId": "rank.top",
                "resolution": "PROVED",
                "answerText": answer,
                "artifactIds": ["query_artifact_top"],
                "evidenceRefs": ["query-artifact:query_artifact_top"],
                "renderer": "VERIFIED_RANKING_RENDERER",
            },
            {
                "goalId": "entity.top",
                "resolution": "PROVED",
                "answerText": answer,
                "artifactIds": ["query_artifact_top"],
                "evidenceRefs": ["semantic:entity"],
                "renderer": "VERIFIED_QUERY_RENDERER",
            },
            {
                "goalId": "detail.rows",
                "resolution": "PROVED",
                "answerText": answer,
                "artifactIds": ["query_artifact_details"],
                "evidenceRefs": ["semantic:detail"],
                "renderer": "VERIFIED_DETAIL_RENDERER",
            },
            {
                "goalId": "dependency.details",
                "resolution": "PROVED",
                "answerText": answer,
                "artifactIds": ["query_artifact_details"],
                "evidenceRefs": ["lineage:entity-set"],
                "renderer": "VERIFIED_DEPENDENCY_RENDERER",
            },
        ],
    }

    summary = parse_acceptance_response(case, payload)

    assert summary["status"] == "PASS"
    assert summary["architecturePassed"] is True

    harness["queryBranches"][1]["dependencyQueryIds"] = []
    unsafe = parse_acceptance_response(case, payload)
    assert unsafe["status"] == "FAIL"
    assert "CROSS_NODE_LINEAGE_REQUIRES_EXECUTION_DEPENDENCY" in unsafe["architectureViolations"]


@pytest.mark.parametrize(
    ("mutation", "expected_violation"),
    [
        (
            lambda harness, payload: harness["answerCoverage"].update(answerFingerprint="wrong"),
            "FINAL_ANSWER_ATTESTATION_FINGERPRINT_MISMATCH",
        ),
        (
            lambda harness, payload: (
                harness.update(verifiedAnalysisArtifactIds=[]),
                harness["goalCoverage"]["resolutionProofTypesByGoalId"].update(
                    {"analysis.main": ["UNVERIFIED_ANALYSIS"]}
                ),
                harness["answerCoverage"]["bindings"][2].update(
                    renderer="VERIFIED_QUERY_RENDERER"
                ),
            ),
            "ANALYSIS_REQUIRES_VERIFIED_ARTIFACT_OR_TYPED_INSUFFICIENCY",
        ),
        (
            lambda harness, payload: harness["goalCoverage"][
                "resolutionArtifactIdsByGoalId"
            ].update({"metric.orders": []}),
            "PROVED_GOAL_REQUIRES_VERIFIED_ARTIFACT",
        ),
    ],
)
def test_parser_fails_closed_on_architecture_regressions(mutation, expected_violation: str) -> None:
    payload = _successful_payload()
    mutation(payload["debugTrace"]["harness"], payload)

    summary = parse_acceptance_response(USER_ACCEPTANCE_CASES[0], payload)

    assert summary["status"] == "FAIL"
    assert expected_violation in summary["architectureViolations"]


def test_typed_operational_failure_is_reported_without_treating_safe_fallback_as_verified_answer() -> None:
    payload = {
        "answer": "本次查数未完成，系统没有把未验证结果作为答案。",
        "debugTrace": {
            "harness": {
                "operationalFailure": {"code": "GROUNDED_PROVIDER_TIMEOUT", "retryable": True},
                "runtimeBudget": {"usage": {"dorisQueries": 1, "llmCalls": 2, "toolCalls": 4}},
            }
        },
    }

    summary = parse_acceptance_response(USER_ACCEPTANCE_CASES[0], payload)

    assert summary["status"] == "OPERATIONAL_FAILURE"
    assert summary["operationalFailure"]["code"] == "GROUNDED_PROVIDER_TIMEOUT"
    assert summary["architecturePassed"] is True
