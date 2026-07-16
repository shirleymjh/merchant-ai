from merchant_ai.models import AnswerMode, PlanningAssetEntry, PlanningAssetPack, QuestionIntent
from merchant_ai.services.query import compile_node_metric_contract
from merchant_ai.services.semantic_metrics import seal_semantic_metric_resolution, semantic_metric_contract_issue


def test_node_compiles_typed_amount_as_sealed_local_contract():
    intent = QuestionIntent(
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="coupon_amount",
        preferred_table="coupons",
        metric_name="coupon_amt",
        metric_column="coupon_amt",
    )

    contract = compile_node_metric_contract(
        intent,
        PlanningAssetPack(
            tables=[PlanningAssetEntry(table="coupons", columns=["seller_id", "coupon_amt", "pt"])],
            metrics=[
                PlanningAssetEntry(
                    key="coupon_amt",
                    table="coupons",
                    columns=["coupon_amt"],
                    metadata={"formula": "SUM(coupon_amt)", "sourceColumns": ["coupon_amt"]},
                )
            ],
        ),
        {"seller_id", "coupon_amt", "pt"},
    )

    assert contract["mode"] == "compiled_local"
    assert contract["formula"] == "SUM(coupon_amt)"
    assert contract["resolution"]["semanticRefId"].startswith("semantic:compiled_local:")
    assert contract["resolution"]["contractProvenance"]["kind"] == "execution_contract"
    assert semantic_metric_contract_issue(contract["resolution"], "coupons") == ""


def test_node_does_not_downgrade_incomplete_published_contract_to_local():
    intent = QuestionIntent(
        answer_mode=AnswerMode.METRIC,
        preferred_table="orders",
        metric_name="gmv",
        metric_column="pay_amt",
        metric_resolution={
            "semanticRefId": "semantic:trade:orders:metric:gmv",
            "metricKey": "gmv",
            "ownerTable": "orders",
            "sourceColumns": ["pay_amt"],
        },
    )

    contract = compile_node_metric_contract(
        intent,
        PlanningAssetPack(tables=[PlanningAssetEntry(table="orders", columns=["seller_id", "pay_amt"])]),
        {"seller_id", "pay_amt"},
    )

    assert contract["mode"] == "published_semantic"
    assert semantic_metric_contract_issue(contract["resolution"], "orders") == "semantic metric contract is incomplete"


def test_node_preserves_planner_compiled_local_provenance():
    resolution = seal_semantic_metric_resolution(
        {
            "semanticRefId": "semantic:compiled_local:orders:metric:order_cnt",
            "metricKey": "order_cnt",
            "ownerTable": "orders",
            "formula": "COUNT(DISTINCT order_id)",
            "sourceColumns": ["order_id"],
            "metricGovernanceMode": "compiled_local",
            "localCompilationPolicy": "declared_formula",
            "contractProvenance": {
                "kind": "planning_asset",
                "ownerTable": "orders",
                "metricKey": "order_cnt",
            },
        },
        force=True,
    )
    intent = QuestionIntent(
        answer_mode=AnswerMode.METRIC,
        preferred_table="orders",
        metric_name="order_cnt",
        metric_column="order_id",
        metric_resolution=resolution,
    )

    contract = compile_node_metric_contract(
        intent,
        PlanningAssetPack(tables=[PlanningAssetEntry(table="orders", columns=["seller_id", "order_id"])]),
        {"seller_id", "order_id"},
    )

    assert contract["mode"] == "compiled_local"
    assert contract["resolution"]["contractProvenance"]["kind"] == "planning_asset"
    assert semantic_metric_contract_issue(contract["resolution"], "orders") == ""
