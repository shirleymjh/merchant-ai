from __future__ import annotations

from types import SimpleNamespace

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


def artifact(*, shape: str = "RANKED") -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id="artifact-1",
        contract=SimpleNamespace(
            query_shape=shape,
            evidence_refs=["metric-ref", "dimension-ref"],
            ranking=SimpleNamespace(enabled=shape == "RANKED", direction="DESC", limit=5),
        ),
        run_result=SimpleNamespace(
            merged_query_bundle=SimpleNamespace(rows=[{"spu_id": "1", "metric": 3}])
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
    assert by_goal["detail"]["rowSetRef"] == "artifact-1"
    assert by_goal["ranking"]["limit"] == 5
    assert by_goal["comparison"]["comparisonMethod"] == "ORDER_BY_DESC_LIMIT_5"


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
