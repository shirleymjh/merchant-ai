from __future__ import annotations

from types import SimpleNamespace

from merchant_ai.services.grounded_answer_coverage import (
    AnswerCoverageVerifier,
    GoalAnswerBinding,
    answer_attestation_matches,
    render_verified_query_goal_sections,
)
from merchant_ai.services.grounded_goal_contract import (
    GoalCoverageResult,
    GoalCoverageVerifier,
    OriginalQuestionGoalContract,
    VerifiedArtifactGoalCoverage,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
)


def anomaly_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "GMV、退款金额、催单工单量分别是多少，哪个环节最异常？",
            "goals": [
                {"goalId": "metric.gmv", "kind": "metric", "label": "GMV"},
                {
                    "goalId": "metric.refund",
                    "kind": "metric",
                    "label": "退款金额",
                },
                {
                    "goalId": "metric.urge",
                    "kind": "metric",
                    "label": "催单工单量",
                },
                {
                    "goalId": "comparison.anomaly",
                    "kind": "comparison",
                    "label": "哪个环节最异常",
                    "comparisonType": "anomaly",
                    "leftGoalIds": ["metric.gmv", "metric.refund"],
                    "rightGoalIds": ["metric.urge"],
                },
            ],
        }
    )


def proved_anomaly_coverage():
    contract = anomaly_contract()
    artifact_id = "analysis-artifact-1"
    declaration = VerifiedArtifactGoalCoverage(
        artifact_id=artifact_id,
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(
            contract
        ),
        covered_goal_ids=[goal.goal_id for goal in contract.goals],
        verification_passed=True,
        evidence_refs=["evidence:aligned-baseline"],
        goal_resolutions=[
            {
                "goalId": "comparison.anomaly",
                "goalKind": "comparison",
                "resolution": "proved",
                "proofType": "VERIFIED_NORMALIZED_ANOMALY",
                "evidenceRefs": ["evidence:aligned-baseline"],
                "operandGoalIds": [
                    "metric.gmv",
                    "metric.refund",
                    "metric.urge",
                ],
                "comparisonMethod": "period-over-period z-score",
                "resultRef": "analysis:anomaly-result",
                "baselineRefs": ["evidence:aligned-baseline"],
                "normalizationMethod": "z-score",
            }
        ],
    )
    coverage = GoalCoverageVerifier().require_complete(contract, [declaration])
    return contract, coverage, artifact_id


def test_data_and_anomaly_artifact_do_not_pass_when_answer_omits_anomaly_goal() -> None:
    """Data/proof availability is separate from visible answer coverage."""

    contract, coverage, _ = proved_anomaly_coverage()
    answer = "GMV 为 100 元，退款金额为 20 元，催单工单量为 5 单。"

    result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert result.passed is False
    assert result.missing_goal_ids == ["comparison.anomaly"]
    assert "ANSWER_GOAL_BINDING_REQUIRED" in {
        issue.code for issue in result.issues
    }


def test_proved_artifact_binding_fails_when_declared_conclusion_is_not_rendered() -> None:
    contract, coverage, artifact_id = proved_anomaly_coverage()
    answer = "GMV 为 100 元，退款金额为 20 元，催单工单量为 5 单。"

    result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [
            GoalAnswerBinding(
                goal_id="comparison.anomaly",
                resolution="PROVED",
                answer_text="标准化后退款环节最异常。",
                artifact_ids=[artifact_id],
                evidence_refs=["evidence:aligned-baseline"],
                renderer="VERIFIED_COMPARISON_RENDERER",
            )
        ],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert result.passed is False
    assert "ANSWER_GOAL_TEXT_NOT_RENDERED" in {
        issue.code for issue in result.issues
    }


def test_proved_anomaly_conclusion_passes_only_with_visible_verified_mapping() -> None:
    contract, coverage, artifact_id = proved_anomaly_coverage()
    conclusion = "标准化后退款环节最异常。"
    answer = (
        "GMV 为 100 元，退款金额为 20 元，催单工单量为 5 单。"
        + conclusion
    )
    result = AnswerCoverageVerifier().require_complete(
        contract,
        coverage,
        answer,
        [
            {
                "goalId": "comparison.anomaly",
                "resolution": "PROVED",
                "answerText": conclusion,
                "artifactIds": [artifact_id],
                "evidenceRefs": ["evidence:aligned-baseline"],
                "renderer": "VERIFIED_COMPARISON_RENDERER",
            }
        ],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert result.passed is True
    assert result.mapped_goal_ids == [
        "metric.gmv",
        "metric.refund",
        "metric.urge",
        "comparison.anomaly",
    ]
    assert answer_attestation_matches(answer, result) is True
    assert answer_attestation_matches(answer + "未经验证的改写", result) is False


def test_rule_goal_cannot_be_rendered_by_query_compose_source() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "商品审核被拒怎么办",
            "rules": [
                {
                    "goalId": "rule.audit_rejection",
                    "label": "审核拒绝处理规则",
                    "ruleRefIds": ["rule:goods:audit-rejection"],
                }
            ],
        }
    )
    artifact_id = "rule-artifact-1"
    ref_id = "rule:goods:audit-rejection"
    coverage = GoalCoverageVerifier().require_complete(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id=artifact_id,
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(
                    contract
                ),
                covered_goal_ids=["rule.audit_rejection"],
                verification_passed=True,
                evidence_refs=[ref_id],
                goal_resolutions=[
                    {
                        "goalId": "rule.audit_rejection",
                        "goalKind": "rule",
                        "resolution": "proved",
                        "evidenceRefs": [ref_id],
                        "ruleRefIds": [ref_id],
                        "citationRefs": [ref_id],
                    }
                ],
            )
        ],
    )
    answer = "请先根据拒绝原因修改商品信息后重新提交审核。"
    binding = {
        "goalId": "rule.audit_rejection",
        "resolution": "PROVED",
        "answerText": answer,
        "artifactIds": [artifact_id],
        "evidenceRefs": [ref_id],
        "renderer": "VERIFIED_RULE_ARTIFACT_RENDERER",
    }

    query_result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [binding],
        source="compose_verified_answer",
    )
    rule_result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [binding],
        source="compose_verified_rule_answer",
    )

    assert query_result.passed is False
    assert "RULE_ANSWER_RENDERER_BOUNDARY_VIOLATION" in {
        issue.code for issue in query_result.issues
    }
    assert rule_result.passed is True


def test_mixed_rule_and_query_answer_accepts_one_verified_attestation() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "说明适用规则，并返回已验证数据指标",
            "goals": [
                {
                    "goalId": "rule.policy",
                    "kind": "rule",
                    "label": "适用规则",
                    "ruleRefIds": ["rule:policy"],
                },
                {
                    "goalId": "metric.value",
                    "kind": "metric",
                    "label": "数据指标",
                    "metricRefId": "metric:value",
                },
            ],
        }
    )
    fingerprint = original_question_goal_contract_fingerprint(contract)
    coverage = GoalCoverageVerifier().require_complete(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id="rule-artifact",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=["rule.policy"],
                verification_passed=True,
                evidence_refs=["rule:policy"],
                goal_resolutions=[
                    {
                        "goalId": "rule.policy",
                        "goalKind": "rule",
                        "resolution": "proved",
                        "proofType": "VERIFIED_RULE_ARTIFACT",
                        "evidenceRefs": ["rule:policy"],
                        "ruleRefIds": ["rule:policy"],
                        "citationRefs": ["rule:policy"],
                    }
                ],
            ),
            VerifiedArtifactGoalCoverage(
                artifact_id="query-artifact",
                goal_contract_fingerprint=fingerprint,
                covered_goal_ids=["metric.value"],
                verification_passed=True,
                evidence_refs=["metric:value"],
                goal_resolutions=[
                    {
                        "goalId": "metric.value",
                        "goalKind": "metric",
                        "resolution": "proved",
                        "proofType": "VERIFIED_QUERY_RESULT",
                        "evidenceRefs": ["metric:value"],
                        "metricRefIds": ["metric:value"],
                        "valueRefs": ["query-artifact"],
                    }
                ],
            ),
        ],
    )
    rule_text = "根据已发布规则：仅采用已验证的规则依据。"
    answer = f"数据指标为 12。\n\n{rule_text}"
    result = AnswerCoverageVerifier().require_complete(
        contract,
        coverage,
        answer,
        [
            {
                "goalId": "rule.policy",
                "resolution": "PROVED",
                "answerText": rule_text,
                "artifactIds": ["rule-artifact"],
                "evidenceRefs": ["rule:policy"],
                "renderer": "VERIFIED_RULE_ARTIFACT_RENDERER",
            }
        ],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert result.passed is True
    assert result.mapped_goal_ids == ["rule.policy", "metric.value"]
    assert answer_attestation_matches(answer, result) is True


def test_detail_goal_rejects_ordinary_text_renderer() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "订单明细给我看一下",
            "details": [
                {
                    "goalId": "detail.orders",
                    "label": "订单明细",
                    "requiredFieldRefIds": ["field:order_id"],
                }
            ],
        }
    )
    artifact_id = "detail-artifact-1"
    field_ref = "field:order_id"
    coverage = GoalCoverageVerifier().require_complete(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id=artifact_id,
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(
                    contract
                ),
                covered_goal_ids=["detail.orders"],
                verification_passed=True,
                evidence_refs=[field_ref],
                goal_resolutions=[
                    {
                        "goalId": "detail.orders",
                        "goalKind": "detail",
                        "resolution": "proved",
                        "evidenceRefs": [field_ref],
                        "outputFields": ["order_id"],
                        "outputSemanticRefs": [field_ref],
                        "rowSetRef": artifact_id,
                        "rowCount": 1,
                    }
                ],
            )
        ],
    )
    answer = "订单 1001。"
    result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [
            {
                "goalId": "detail.orders",
                "resolution": "PROVED",
                "answerText": answer,
                "artifactIds": [artifact_id],
                "evidenceRefs": [field_ref],
                "renderer": "VERIFIED_QUERY_RENDERER",
            }
        ],
        source="compose_verified_answer",
    )

    assert result.passed is False
    assert "ANSWER_GOAL_RENDERER_REQUIRED" in {
        issue.code for issue in result.issues
    }


def test_ranked_answer_binding_is_generated_from_real_rows_not_model_provenance() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "销量最高的前3个商品",
            "goals": [
                {
                    "goalId": "metric.sales",
                    "kind": "metric",
                    "label": "销量",
                    "required": False,
                },
                {
                    "goalId": "ranking.top3",
                    "kind": "ranking",
                    "label": "销量最高前3个商品",
                    "metricGoalIds": ["metric.sales"],
                    "limit": 3,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
            ],
        }
    )
    artifact_id = "query-ranked-1"
    coverage = GoalCoverageResult(
        passed=True,
        finalization_allowed=True,
        required_goal_ids=["metric.sales", "ranking.top3"],
        covered_goal_ids=["metric.sales", "ranking.top3"],
        resolved_goal_ids=["metric.sales", "ranking.top3"],
        resolution_by_goal_id={
            "metric.sales": "PROVED",
            "ranking.top3": "PROVED",
        },
        resolution_artifact_ids_by_goal_id={
            "metric.sales": [artifact_id],
            "ranking.top3": [artifact_id],
        },
        resolution_evidence_refs_by_goal_id={
            "metric.sales": ["metric:sales"],
            "ranking.top3": ["metric:sales"],
        },
    )
    artifact = SimpleNamespace(
        artifact_id=artifact_id,
        output_columns=["商品名称", "销量"],
        contract=SimpleNamespace(
            query_shape="RANKED",
            ranking=SimpleNamespace(enabled=True),
            evidence_refs=["metric:sales"],
        ),
        run_result=SimpleNamespace(
            merged_query_bundle=SimpleNamespace(
                rows=[
                    {"商品名称": "商品A", "销量": 30},
                    {"商品名称": "商品B", "销量": 20},
                    {"商品名称": "商品C", "销量": 10},
                ],
                offloaded_files=[],
            )
        ),
    )

    rendered = render_verified_query_goal_sections(
        contract,
        coverage,
        "已完成销量查询。",
        [artifact],
    )
    result = AnswerCoverageVerifier().require_complete(
        contract,
        coverage,
        rendered.answer_markdown,
        rendered.bindings,
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert rendered.appended_goal_ids == ["ranking.top3"]
    assert "商品A" in rendered.answer_markdown
    assert "| 商品名称 | 销量 |" in rendered.answer_markdown
    assert rendered.bindings[0].renderer == "VERIFIED_RANKING_RENDERER"
    assert result.passed is True


def test_analysis_goal_cannot_be_attested_from_an_ordinary_query_artifact() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "分析退款是否影响订单表现",
            "goals": [
                {
                    "goalId": "metric.orders",
                    "kind": "metric",
                    "label": "订单量",
                    "required": False,
                },
                {
                    "goalId": "analysis.impact",
                    "kind": "analysis",
                    "label": "退款是否影响订单表现",
                    "analysisType": "impact",
                    "inputGoalIds": ["metric.orders"],
                },
            ],
        }
    )
    artifact_id = "ordinary-query-artifact"
    coverage = GoalCoverageResult(
        passed=True,
        finalization_allowed=True,
        required_goal_ids=["metric.orders", "analysis.impact"],
        covered_goal_ids=["metric.orders", "analysis.impact"],
        resolved_goal_ids=["metric.orders", "analysis.impact"],
        resolution_by_goal_id={
            "metric.orders": "PROVED",
            "analysis.impact": "PROVED",
        },
        resolution_artifact_ids_by_goal_id={
            "metric.orders": [artifact_id],
            "analysis.impact": [artifact_id],
        },
    )
    artifact = SimpleNamespace(
        artifact_id=artifact_id,
        contract=SimpleNamespace(
            query_shape="SCALAR",
            ranking=SimpleNamespace(enabled=False),
            evidence_refs=[],
        ),
        run_result=SimpleNamespace(
            merged_query_bundle=SimpleNamespace(
                rows=[{"订单量": 100}], offloaded_files=[]
            )
        ),
    )

    rendered = render_verified_query_goal_sections(
        contract,
        coverage,
        "订单量为100，退款导致了订单下降。",
        [artifact],
    )
    result = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        rendered.answer_markdown,
        rendered.bindings,
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    assert rendered.bindings == []
    assert result.passed is False
    assert result.missing_goal_ids == ["analysis.impact"]


def test_query_renderer_uses_typed_comparison_capability_not_goal_label() -> None:
    def comparison_case(*, label: str, comparison_type: str):
        contract = parse_original_question_goal_contract(
            {
                "question": "typed comparison",
                "goals": [
                    {
                        "goalId": "metric.sales",
                        "kind": "metric",
                        "label": "销量",
                        "required": False,
                    },
                    {
                        "goalId": "metric.entity",
                        "kind": "metric",
                        "label": "商品",
                        "required": False,
                    },
                    {
                        "goalId": "comparison.typed",
                        "kind": "comparison",
                        "label": label,
                        "comparisonType": comparison_type,
                        "leftGoalIds": ["metric.sales"],
                        "rightGoalIds": ["metric.entity"],
                    },
                ],
            }
        )
        artifact_id = "query-comparison-typed"
        coverage = GoalCoverageResult(
            passed=True,
            finalization_allowed=True,
            required_goal_ids=[
                "metric.sales",
                "metric.entity",
                "comparison.typed",
            ],
            covered_goal_ids=[
                "metric.sales",
                "metric.entity",
                "comparison.typed",
            ],
            resolved_goal_ids=[
                "metric.sales",
                "metric.entity",
                "comparison.typed",
            ],
            resolution_by_goal_id={
                "metric.sales": "PROVED",
                "metric.entity": "PROVED",
                "comparison.typed": "PROVED",
            },
            resolution_artifact_ids_by_goal_id={
                "metric.sales": [artifact_id],
                "metric.entity": [artifact_id],
                "comparison.typed": [artifact_id],
            },
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id,
            output_columns=["商品", "销量"],
            contract=SimpleNamespace(
                query_shape="RANKED",
                ranking=SimpleNamespace(enabled=True),
                evidence_refs=[],
            ),
            run_result=SimpleNamespace(
                merged_query_bundle=SimpleNamespace(
                    rows=[{"商品": "商品A", "销量": 30}],
                    offloaded_files=[],
                )
            ),
        )
        return render_verified_query_goal_sections(
            contract,
            coverage,
            "已验证查询完成。",
            [artifact],
        )

    ranked = comparison_case(label="异常商品排名", comparison_type="RANK")
    anomaly = comparison_case(label="普通比较结果", comparison_type="ANOMALY")

    assert [item.goal_id for item in ranked.bindings] == ["comparison.typed"]
    assert ranked.bindings[0].renderer == "VERIFIED_COMPARISON_RENDERER"
    assert anomaly.bindings == []
    assert anomaly.appended_goal_ids == []


def test_verified_insufficiency_ref_is_rendered_without_masquerading_as_proof() -> None:
    contract = anomaly_contract()
    artifact_id = "query-scalars-no-baseline"
    gap_ref = "gap:comparable-baseline-missing"
    coverage = GoalCoverageVerifier().require_complete(
        contract,
        [
            VerifiedArtifactGoalCoverage(
                artifact_id=artifact_id,
                goal_contract_fingerprint=original_question_goal_contract_fingerprint(
                    contract
                ),
                covered_goal_ids=[
                    "metric.gmv",
                    "metric.refund",
                    "metric.urge",
                ],
                verification_passed=True,
                goal_resolutions=[
                    {
                        "goalId": "comparison.anomaly",
                        "goalKind": "comparison",
                        "resolution": "insufficient_evidence",
                        "reason": "缺少统一口径的历史基线",
                        "evidenceRefs": [gap_ref],
                    }
                ],
            )
        ],
    )
    artifact = SimpleNamespace(
        artifact_id=artifact_id,
        contract=SimpleNamespace(
            query_shape="SCALAR",
            ranking=SimpleNamespace(enabled=False),
            evidence_refs=[],
        ),
        run_result=SimpleNamespace(
            merged_query_bundle=SimpleNamespace(rows=[], offloaded_files=[])
        ),
    )
    rendered = render_verified_query_goal_sections(
        contract,
        coverage,
        "GMV、退款金额和催单工单量已经查到。",
        [artifact],
    )
    result = AnswerCoverageVerifier().require_complete(
        contract,
        coverage,
        rendered.answer_markdown,
        rendered.bindings,
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )

    gap_binding = next(
        item
        for item in result.bindings
        if item.goal_id == "comparison.anomaly"
    )
    assert "证据不足：缺少统一口径的历史基线" in rendered.answer_markdown
    assert gap_binding.resolution == "INSUFFICIENT_EVIDENCE"
    assert gap_binding.insufficiency_ref == gap_ref
    assert gap_binding.artifact_ids == []
    assert result.passed is True


def test_derived_analysis_binding_accepts_internal_compose_verified_answer_source() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "退款与订单是否相关",
            "goals": [
                {
                    "goalId": "metric.series",
                    "kind": "metric",
                    "label": "对齐序列",
                    "required": False,
                },
                {
                    "goalId": "analysis.correlation",
                    "kind": "analysis",
                    "label": "相关性分析",
                    "analysisType": "CORRELATION",
                    "inputGoalIds": ["metric.series"],
                }
            ],
        }
    )
    coverage = GoalCoverageResult(
        passed=True,
        finalization_allowed=True,
        required_goal_ids=["metric.series", "analysis.correlation"],
        covered_goal_ids=["metric.series", "analysis.correlation"],
        resolved_goal_ids=["metric.series", "analysis.correlation"],
        resolution_by_goal_id={
            "metric.series": "PROVED",
            "analysis.correlation": "PROVED",
        },
        resolution_proof_types_by_goal_id={
            "analysis.correlation": ["DETERMINISTIC_DERIVED_ANALYSIS"]
        },
        resolution_artifact_ids_by_goal_id={
            "metric.series": ["query-series-1"],
            "analysis.correlation": ["analysis-artifact-1"]
        },
        resolution_evidence_refs_by_goal_id={
            "metric.series": ["metric:series"],
            "analysis.correlation": ["analysis:result:1"]
        },
    )
    answer = "相关系数为 0.8；相关不等于因果。"
    binding = {
        "goalId": "analysis.correlation",
        "resolution": "PROVED",
        "answerText": answer,
        "artifactIds": ["analysis-artifact-1"],
        "evidenceRefs": ["analysis:result:1"],
        "renderer": "VERIFIED_ANALYSIS_ARTIFACT_RENDERER",
    }

    composed = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [binding],
        source="compose_verified_answer",
        auto_bind_verified_primitives=True,
    )
    trusted = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [binding],
        source="run_skill",
        auto_bind_verified_primitives=True,
    )

    untrusted = AnswerCoverageVerifier().verify(
        contract,
        coverage,
        answer,
        [binding],
        source="ordinary_model_answer",
        auto_bind_verified_primitives=True,
    )

    assert composed.passed is True
    assert trusted.passed is True
    assert untrusted.passed is False
    assert "DERIVED_ANALYSIS_RENDERER_SOURCE_REQUIRED" in {
        issue.code for issue in untrusted.issues
    }


def test_multi_artifact_detail_renderer_does_not_reuse_another_goal_section() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "最近7天订单明细和最近10天退款明细",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "detail",
                    "label": "最近7天订单明细",
                },
                {
                    "goalId": "detail.refunds",
                    "kind": "detail",
                    "label": "最近10天退款明细",
                },
            ],
        }
    )
    coverage = GoalCoverageResult(
        passed=True,
        finalization_allowed=True,
        required_goal_ids=["detail.orders", "detail.refunds"],
        covered_goal_ids=["detail.orders", "detail.refunds"],
        resolved_goal_ids=["detail.orders", "detail.refunds"],
        resolution_by_goal_id={
            "detail.orders": "PROVED",
            "detail.refunds": "PROVED",
        },
        resolution_artifact_ids_by_goal_id={
            "detail.orders": ["artifact-orders"],
            "detail.refunds": ["artifact-refunds"],
        },
        resolution_evidence_refs_by_goal_id={
            "detail.orders": ["semantic:orders"],
            "detail.refunds": ["semantic:refunds"],
        },
    )

    def artifact(
        artifact_id: str,
        evidence_ref: str,
        row: dict[str, object],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            artifact_id=artifact_id,
            output_columns=list(row),
            contract=SimpleNamespace(
                query_shape="DETAIL",
                ranking=SimpleNamespace(enabled=False),
                evidence_refs=[evidence_ref],
            ),
            run_result=SimpleNamespace(
                merged_query_bundle=SimpleNamespace(
                    rows=[row],
                    offloaded_files=[],
                )
            ),
        )

    rendered = render_verified_query_goal_sections(
        contract,
        coverage,
        (
            "### 最近7天订单明细\n\n"
            "| order_id | status |\n| --- | --- |\n"
            "| order_id_shared | paid |"
        ),
        [
            artifact(
                "artifact-orders",
                "semantic:orders",
                {"order_id": "order_id_shared", "status": "paid"},
            ),
            artifact(
                "artifact-refunds",
                "semantic:refunds",
                {
                    "order_id": "order_id_shared",
                    "refund_status": "success",
                },
            ),
        ],
    )

    assert rendered.appended_goal_ids == ["detail.refunds"]
    assert "### 最近10天退款明细" in rendered.answer_markdown
    assert "| order_id_shared | success |" in rendered.answer_markdown
    bindings = {item.goal_id: item for item in rendered.bindings}
    assert bindings["detail.orders"].artifact_ids == ["artifact-orders"]
    assert bindings["detail.refunds"].artifact_ids == ["artifact-refunds"]
    assert "最近10天退款明细" not in bindings["detail.orders"].answer_text
