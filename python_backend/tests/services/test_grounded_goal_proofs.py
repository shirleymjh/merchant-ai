from __future__ import annotations

from types import SimpleNamespace

from merchant_ai.models import QueryBundle, ResultCoverage
from merchant_ai.services.grounded_goal_contract import (
    ComparisonQuestionGoal,
    DependencyQuestionGoal,
    DetailQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
)
from merchant_ai.services.grounded_goal_proofs import (
    derive_query_artifact_goal_resolutions,
)


def artifact(
    *,
    shape: str = "RANKED",
    result_coverage: str | None = None,
    rows: list[dict[str, object]] | None = None,
    is_truncated: bool = False,
) -> SimpleNamespace:
    result_rows = (
        rows if rows is not None else [{"spu_id": "1", "metric": 3}]
    )
    coverage = result_coverage or (
        ResultCoverage.TOP_N.value
        if shape == "RANKED"
        else ResultCoverage.UNKNOWN.value
    )
    return SimpleNamespace(
        artifact_id="artifact-1",
        contract_fingerprint="contract-fp",
        sql_fingerprint="sql-fp",
        verified_evidence=SimpleNamespace(passed=True),
        ranking_semantics_verified=shape == "RANKED",
        sql_validation=None,
        contract=SimpleNamespace(
            query_shape=shape,
            evidence_refs=["metric-ref", "dimension-ref"],
            ranking=SimpleNamespace(enabled=shape == "RANKED", direction="DESC", limit=5),
            reference_scope=SimpleNamespace(enabled=False),
            upstream_entity_bindings=[],
        ),
        run_result=SimpleNamespace(
            merged_query_bundle=QueryBundle(
                rows=result_rows,
                original_row_count=(
                    len(result_rows)
                    if coverage
                    in {ResultCoverage.ALL_ROWS.value, ResultCoverage.TOP_N.value}
                    else 0
                ),
                is_truncated=is_truncated,
                result_coverage=coverage,
            )
        ),
        output_columns=["spu_id", "metric"],
        output_lineage={"spu_id": ["dimension-ref"]},
    )


def test_derives_detail_ranking_and_rank_comparison_proofs() -> None:
    contract = OriginalQuestionGoalContract(
        question="最近7天退款明细并找出最高前5单",
        goals=[
            MetricQuestionGoal(goal_id="metric", label="退款金额"),
            DetailQuestionGoal(
                goal_id="detail",
                label="退款明细",
                input_goal_ids=["metric"],
            ),
            RankingQuestionGoal(
                goal_id="ranking",
                label="退款金额最高前5单",
                metric_goal_ids=["metric"],
                direction="DESC",
                limit=5,
                population_scope="ALL_MATCHING_ROWS",
            ),
            ComparisonQuestionGoal(
                goal_id="comparison",
                label="退款金额Top5",
                comparison_type="rank_desc_top_5",
                left_goal_ids=["metric"],
                right_goal_ids=["detail"],
            ),
        ],
    )
    assigned = ["metric", "detail", "ranking", "comparison"]

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=artifact(),
        assigned_goal_ids=assigned,
        artifact_goal_ids={"artifact-1": assigned},
        all_artifacts=[artifact()],
    )

    by_goal = {item["goalId"]: item for item in resolutions}
    assert by_goal["detail"]["resolution"] == "INSUFFICIENT_EVIDENCE"
    assert by_goal["detail"]["details"]["resultCoverage"] == "TOP_N"
    assert by_goal["ranking"]["limit"] == 5
    assert by_goal["comparison"]["comparisonMethod"] == "ORDER_BY_DESC_LIMIT_5"


def test_complete_detail_row_set_proves_detail_goal() -> None:
    contract = OriginalQuestionGoalContract(
        question="最近7天订单明细",
        goals=[DetailQuestionGoal(goal_id="detail", label="订单明细")],
    )
    complete_artifact = artifact(
        shape="DETAIL",
        result_coverage=ResultCoverage.ALL_ROWS.value,
    )

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=complete_artifact,
        assigned_goal_ids=["detail"],
        artifact_goal_ids={"artifact-1": ["detail"]},
        all_artifacts=[complete_artifact],
    )

    assert resolutions[0]["resolution"] == "PROVED"
    assert resolutions[0]["rowSetRef"] == "artifact-1"


def test_equal_visible_and_original_counts_do_not_imply_complete_detail() -> None:
    contract = OriginalQuestionGoalContract(
        question="全部订单明细",
        goals=[DetailQuestionGoal(goal_id="detail", label="订单明细")],
    )
    rows = [{"spu_id": str(index), "metric": index} for index in range(100)]
    unknown_bundle = QueryBundle(rows=rows, original_row_count=len(rows))
    unknown_artifact = artifact(
        shape="DETAIL",
        result_coverage=ResultCoverage.UNKNOWN.value,
        rows=rows,
    )
    # Preserve the adversarial legacy representation: equal counts and no
    # truncation flag, but no producer-attested coverage.
    unknown_artifact.run_result.merged_query_bundle = unknown_bundle

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=unknown_artifact,
        assigned_goal_ids=["detail"],
        artifact_goal_ids={"artifact-1": ["detail"]},
        all_artifacts=[unknown_artifact],
    )

    assert unknown_bundle.has_complete_detail_coverage() is False
    assert resolutions[0]["resolution"] == "INSUFFICIENT_EVIDENCE"
    assert resolutions[0]["details"]["resultCoverage"] == "UNKNOWN"


def test_preview_coverage_is_truncated_even_when_counts_match() -> None:
    rows = [{"id": index} for index in range(100)]

    bundle = QueryBundle(
        rows=rows,
        original_row_count=len(rows),
        result_coverage=ResultCoverage.PREVIEW.value,
    )

    assert bundle.is_truncated is True
    assert bundle.has_complete_detail_coverage() is False


def test_ranking_population_cannot_be_proved_from_goal_declaration_or_semantic_refs() -> None:
    contract = OriginalQuestionGoalContract(
        question="订单明细中退款最多的三单",
        goals=[
            MetricQuestionGoal(goal_id="refund", label="退款金额"),
            DetailQuestionGoal(goal_id="orders", label="订单明细"),
            RankingQuestionGoal(
                goal_id="ranking",
                label="退款最多三单",
                metric_goal_ids=["refund"],
                direction="DESC",
                limit=3,
                population_scope="SAME_AS_GOAL",
                population_goal_ids=["orders"],
            ),
        ],
    )
    ranked_artifact = artifact()

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=ranked_artifact,
        assigned_goal_ids=["refund", "ranking"],
        artifact_goal_ids={
            "artifact-orders": ["orders"],
            "artifact-1": ["refund", "ranking"],
        },
        all_artifacts=[ranked_artifact],
    )

    assert not any(item["goalId"] == "ranking" for item in resolutions)


def test_cross_turn_predicate_population_proof_uses_artifact_fingerprints() -> None:
    contract = OriginalQuestionGoalContract(
        question="这里面退款最多的三单",
        goals=[
            MetricQuestionGoal(goal_id="refund", label="退款金额"),
            RankingQuestionGoal(
                goal_id="ranking",
                label="退款最多三单",
                metric_goal_ids=["refund"],
                direction="DESC",
                limit=3,
                population_scope="VERIFIED_PREDICATE_SCOPE",
            ),
        ],
    )
    ranked_artifact = artifact(
        rows=[
            {"spu_id": "1", "metric": 30},
            {"spu_id": "2", "metric": 20},
            {"spu_id": "3", "metric": 10},
        ]
    )
    ranked_artifact.contract.ranking.limit = 3
    ranked_artifact.contract.reference_scope = SimpleNamespace(
        enabled=True,
        executable=True,
        population_required=True,
        referent_type="PREDICATE_SCOPE",
        source_artifact_id="orders-7d",
        source_contract_fingerprint="orders-contract-fp",
        source_sql_fingerprint="orders-sql-fp",
    )

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=ranked_artifact,
        assigned_goal_ids=["refund", "ranking"],
        artifact_goal_ids={"artifact-1": ["refund", "ranking"]},
        all_artifacts=[ranked_artifact],
    )

    ranking = next(item for item in resolutions if item["goalId"] == "ranking")
    assert ranking["populationScope"] == "VERIFIED_PREDICATE_SCOPE"
    assert ranking["populationGoalIds"] == []
    assert ranking["populationLineageRefs"] == [
        "query-artifact:orders-7d",
        "contract-fingerprint:orders-contract-fp",
        "sql-fingerprint:orders-sql-fp",
    ]


def test_ranking_proof_requires_kernel_ranking_semantics_receipt() -> None:
    contract = OriginalQuestionGoalContract(
        question="退款最多的五单",
        goals=[
            MetricQuestionGoal(goal_id="refund", label="退款金额"),
            RankingQuestionGoal(
                goal_id="ranking",
                label="退款最多五单",
                metric_goal_ids=["refund"],
                direction="DESC",
                limit=5,
                population_scope="ALL_MATCHING_ROWS",
            ),
        ],
    )
    ranked_artifact = artifact()
    ranked_artifact.ranking_semantics_verified = False

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=ranked_artifact,
        assigned_goal_ids=["refund", "ranking"],
        artifact_goal_ids={"artifact-1": ["refund", "ranking"]},
        all_artifacts=[ranked_artifact],
    )

    assert not any(item["goalId"] == "ranking" for item in resolutions)


def test_scalar_artifact_cannot_prove_anomaly_comparison() -> None:
    contract = OriginalQuestionGoalContract(
        question="哪个环节最异常",
        goals=[
            MetricQuestionGoal(goal_id="gmv", label="GMV"),
            MetricQuestionGoal(goal_id="refund", label="退款金额"),
            ComparisonQuestionGoal(
                goal_id="anomaly",
                label="哪个环节最异常",
                comparison_type="ANOMALY",
                left_goal_ids=["gmv"],
                right_goal_ids=["refund"],
            ),
        ],
    )

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=artifact(shape="SCALAR"),
        assigned_goal_ids=["gmv", "refund", "anomaly"],
        artifact_goal_ids={"artifact-1": ["gmv", "refund", "anomaly"]},
        all_artifacts=[artifact(shape="SCALAR")],
    )

    assert not any(item["goalId"] == "anomaly" for item in resolutions)


def test_derives_same_artifact_dependency_lineage() -> None:
    contract = OriginalQuestionGoalContract(
        question="Top商品及品牌",
        goals=[
            MetricQuestionGoal(goal_id="rank", label="销量"),
            DetailQuestionGoal(goal_id="brand", label="品牌"),
            DependencyQuestionGoal(
                goal_id="dependency",
                label="商品到品牌",
                upstream_goal_ids=["rank"],
                downstream_goal_ids=["brand"],
                dependency_type="same_table_projection",
            ),
        ],
    )
    assigned = ["rank", "brand", "dependency"]

    resolutions = derive_query_artifact_goal_resolutions(
        goal_contract=contract,
        artifact=artifact(),
        assigned_goal_ids=assigned,
        artifact_goal_ids={"artifact-1": assigned},
        all_artifacts=[artifact()],
    )

    dependency = next(item for item in resolutions if item["goalId"] == "dependency")
    assert dependency["upstreamArtifactIds"] == ["artifact-1"]
    assert dependency["downstreamArtifactIds"] == ["artifact-1"]
    assert "dimension-ref" in dependency["lineageRefs"]
