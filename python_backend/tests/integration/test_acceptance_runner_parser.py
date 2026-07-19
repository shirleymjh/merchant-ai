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
                    "artifactIds": ["query_artifact_orders", "analysis_artifact_main"],
                    "resolutionByGoalId": {
                        "metric.orders": "PROVED",
                        "time.main": "PROVED",
                        "analysis.main": "PROVED",
                    },
                    "resolutionProofTypesByGoalId": {
                        "analysis.main": ["DETERMINISTIC_DERIVED_ANALYSIS"]
                    },
                    "resolutionArtifactIdsByGoalId": {"analysis.main": ["analysis_artifact_main"]},
                },
                "answerCoverage": {
                    "passed": True,
                    "source": "run_skill",
                    "answerFingerprint": fingerprint,
                    "bindings": [
                        {
                            "goalId": "analysis.main",
                            "resolution": "PROVED",
                            "artifactIds": ["analysis_artifact_main"],
                            "renderer": "VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
                        }
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


def test_analysis_accepts_typed_insufficiency_instead_of_a_proved_analysis_artifact() -> None:
    payload = _successful_payload("证据不足：样本量不足。")
    harness = payload["debugTrace"]["harness"]
    harness["verifiedAnalysisArtifactIds"] = []
    harness["goalCoverage"]["resolutionByGoalId"]["analysis.main"] = "INSUFFICIENT_EVIDENCE"
    harness["goalCoverage"]["resolutionProofTypesByGoalId"]["analysis.main"] = ["TYPED_EVIDENCE_GAP"]
    harness["goalCoverage"]["resolutionArtifactIdsByGoalId"]["analysis.main"] = ["analysis_gap_artifact"]
    harness["goalCoverage"]["resolutionEvidenceRefsByGoalId"] = {"analysis.main": ["analysis:gap:sample"]}
    harness["goalCoverage"]["insufficiencyReasonByGoalId"] = {"analysis.main": "样本量不足"}
    harness["goalCoverage"]["artifactIds"] = ["query_artifact_orders", "analysis_gap_artifact"]
    harness["answerCoverage"]["bindings"] = [
        {
            "goalId": "analysis.main",
            "resolution": "INSUFFICIENT_EVIDENCE",
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


def test_entity_chain_requires_observed_wait_and_verified_upstream_lineage() -> None:
    case = AcceptanceCase(
        "chain",
        "TopN 后查实体",
        "test",
        ("RANKING", "ENTITY", "DEPENDENCY"),
        minimum_branch_count=2,
        requires_entity_lineage=True,
    )
    payload = _successful_payload()
    harness = payload["debugTrace"]["harness"]
    harness["originalQuestionGoalContract"] = {
        "goals": [
            {"goalId": "rank.top", "kind": "RANKING"},
            {"goalId": "entity.top", "kind": "ENTITY"},
            {"goalId": "dependency.details", "kind": "DEPENDENCY"},
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
            "goalIds": ["dependency.details"],
            "topicScope": ["电商退货"],
            "status": "VERIFIED",
            "dependencyQueryIds": ["top"],
            "statusHistory": ["WAITING_VERIFIED_ENTITY_SET", "PREPARED", "VERIFIED"],
            "verifiedArtifactIds": ["query_artifact_details"],
        },
    ]
    harness["verifiedQueryArtifactIds"] = ["query_artifact_top", "query_artifact_details"]
    harness["goalCoverage"]["artifactIds"] = ["query_artifact_top", "query_artifact_details"]
    harness["answerCoverage"]["bindings"] = []

    summary = parse_acceptance_response(case, payload)

    assert summary["status"] == "PASS"
    assert summary["branches"][1]["lineageWaitObserved"] is True

    harness["queryBranches"][1]["statusHistory"] = ["PREPARED", "VERIFIED"]
    unsafe = parse_acceptance_response(case, payload)
    assert unsafe["status"] == "FAIL"
    assert "ENTITY_CHAIN_MUST_WAIT_FOR_VERIFIED_LINEAGE" in unsafe["architectureViolations"]


@pytest.mark.parametrize(
    ("mutation", "expected_violation"),
    [
        (
            lambda harness, payload: harness["answerCoverage"].update(answerFingerprint="wrong"),
            "FINAL_ANSWER_ATTESTATION_FINGERPRINT_MISMATCH",
        ),
        (
            lambda harness, payload: harness.update(verifiedAnalysisArtifactIds=[]),
            "ANALYSIS_REQUIRES_VERIFIED_ARTIFACT_OR_TYPED_INSUFFICIENCY",
        ),
        (
            lambda harness, payload: harness.update(queryBranches=[]),
            "INDEPENDENT_GOALS_REQUIRE_DECLARED_BRANCHES",
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
