from merchant_ai.services.semantic_metrics import (
    seal_semantic_metric_resolution,
    semantic_metric_contract_issue,
)
from merchant_ai.models import AgentRunResult, AgentTaskResult, AnswerMode, PlanningAssetPack, QueryBundle, QueryPlan, QuestionIntent
from merchant_ai.services.answer import (
    deterministic_cross_task_detail_answer,
    ensure_required_field_answer_coverage,
    merchant_friendly_data_answer,
    metric_value_column_for_rows,
)
from merchant_ai.services.memory import reusable_knowledge_assertion_present
from merchant_ai.services.query import governed_realtime_metric_resolution, merge_task_result_bundles


def governed_resolution():
    return {
        "semanticRefId": "semantic:trade:orders:metric:gmv",
        "metricKey": "gmv",
        "ownerTable": "orders",
        "formula": "SUM(pay_amount)",
        "sourceColumns": ["pay_amount"],
        "unit": "元",
    }


def test_semantic_metric_contract_is_sealed_and_valid():
    resolution = seal_semantic_metric_resolution(governed_resolution())

    assert resolution["semanticContractHash"]
    assert semantic_metric_contract_issue(resolution, "orders") == ""


def test_semantic_metric_contract_rejects_late_formula_override():
    resolution = seal_semantic_metric_resolution(governed_resolution())
    resolution["formula"] = "COUNT(DISTINCT order_id)"
    resolution["sourceColumns"] = ["order_id"]

    assert semantic_metric_contract_issue(resolution, "orders") == "semantic metric contract drifted after resolution"


def test_semantic_metric_contract_allows_execution_identifier_quoting():
    resolution = seal_semantic_metric_resolution(
        {
            "semanticRefId": "semantic:trade:orders:metric:order_count",
            "metricKey": "order_count",
            "ownerTable": "orders",
            "formula": "COUNT(DISTINCT order_id)",
            "sourceColumns": ["order_id"],
            "unit": "单",
        }
    )
    resolution["formula"] = "COUNT(DISTINCT `order_id`)"

    assert semantic_metric_contract_issue(resolution, "orders") == ""


def test_semantic_metric_contract_rejects_late_table_override():
    resolution = seal_semantic_metric_resolution(governed_resolution())
    resolution["ownerTable"] = "refunds"

    assert semantic_metric_contract_issue(resolution, "refunds") == "semantic metric contract drifted after resolution"


def test_answer_does_not_substitute_an_unrelated_numeric_column_for_a_metric():
    resolution = seal_semantic_metric_resolution(governed_resolution())
    intent = QuestionIntent(
        answer_mode=AnswerMode.METRIC,
        preferred_table="orders",
        metric_name="gmv",
        metric_column="pay_amount",
        metric_formula="SUM(pay_amount)",
        metric_resolution=resolution,
    )

    assert metric_value_column_for_rows(QueryPlan(intents=[intent]), intent, [{"unrelated_count": 42}]) == ""


def test_answer_binds_same_named_metric_to_its_semantic_owner_task():
    refund_resolution = seal_semantic_metric_resolution(
        {
            **governed_resolution(),
            "semanticRefId": "semantic:refund:refunds:metric:gmv",
            "ownerTable": "refunds",
        }
    )
    refund_resolution["displayName"] = "退款金额"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="anchor_order",
                preferred_table="orders",
                filter_column="order_id",
            ),
            QuestionIntent(
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="refund_lookup",
                preferred_table="refunds",
                metric_name="gmv",
                metric_column="pay_amount",
                metric_formula="SUM(pay_amount)",
                metric_resolution=refund_resolution,
            ),
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="anchor_order",
                success=True,
                query_bundle=QueryBundle(tables=["orders"], rows=[{"order_id": "order_1", "pay_amount": 122}]),
            ),
            AgentTaskResult(
                task_id="refund_lookup",
                success=True,
                query_bundle=QueryBundle(tables=["refunds"], rows=[{"order_id": "order_1", "gmv": 121.5}]),
            ),
        ],
        merged_query_bundle=QueryBundle(rows=[{"order_id": "order_1", "pay_amount": 122}]),
    )

    answer = merchant_friendly_data_answer("订单 order_1 的退款金额是多少？", plan, run.merged_query_bundle, run)

    assert "121.5" in answer
    assert "122" not in answer


def test_answer_appends_verified_required_semantic_field_omitted_by_prose():
    plan = QueryPlan(
        question_understanding={
            "requiredEvidenceIntents": [
                {
                    "sourcePhrase": "退货用户",
                    "suggestedTables": ["refunds"],
                    "suggestedFields": ["buyer_name"],
                }
            ]
        },
        intents=[
            QuestionIntent(
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="refund_lookup",
                preferred_table="refunds",
            )
        ],
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refund_lookup",
                success=True,
                query_bundle=QueryBundle(tables=["refunds"], rows=[{"buyer_name": "buyer_100"}]),
            )
        ]
    )

    answer = ensure_required_field_answer_coverage("退款金额为 121.50元。", plan, run)

    assert "退货用户：buyer_100" in answer


def test_merged_rows_record_field_owner_conflicts_instead_of_hiding_them():
    merged = merge_task_result_bundles(
        [
            AgentTaskResult(
                task_id="order_lookup",
                success=True,
                query_bundle=QueryBundle(rows=[{"order_id": "order_1", "pay_amt": 122}]),
            ),
            AgentTaskResult(
                task_id="refund_lookup",
                success=True,
                query_bundle=QueryBundle(rows=[{"order_id": "order_1", "pay_amt": 121.5}]),
            ),
        ]
    )

    row = merged.rows[0]
    assert row["pay_amt"] == 122
    assert row["refund_lookup__pay_amt"] == 121.5
    assert row["__fieldLineage"]["pay_amt"] == ["order_lookup", "refund_lookup"]
    assert {item["taskId"] for item in row["__fieldConflicts"]["pay_amt"]} == {"order_lookup", "refund_lookup"}


def test_knowledge_curator_hot_path_skips_ordinary_one_shot_query():
    state = {"question": "最近7天订单量是多少？", "message_history": []}

    assert not reusable_knowledge_assertion_present(state, {"memoryType": "query_event"})


def test_knowledge_curator_keeps_explicit_reusable_user_assertion():
    state = {"question": "请记住，以后默认使用支付口径。", "message_history": []}

    assert reusable_knowledge_assertion_present(state, {"memoryType": "query_event"})


def test_cross_task_detail_answer_keeps_each_tasks_owned_fields():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="order_lookup",
                answer_mode=AnswerMode.DETAIL,
                preferred_table="orders",
                output_keys=["order_id", "pt", "pay_amt"],
                required_evidence=["order_id", "pt", "pay_amt"],
            ),
            QuestionIntent(
                plan_task_id="refund_lookup",
                answer_mode=AnswerMode.GROUP_AGG,
                preferred_table="refunds",
                output_keys=["order_id", "buyer_name", "refund_create_time", "pay_amt"],
                required_evidence=["buyer_name", "refund_create_time", "pay_amt"],
                metric_name="pay_amt",
                metric_resolution={"metricKey": "pay_amt", "displayName": "退款金额", "sourceColumns": ["pay_amt"]},
            ),
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="order_lookup",
                success=True,
                query_bundle=QueryBundle(rows=[{"order_id": "order_1", "pt": "2026-07-10", "pay_amt": 122}]),
            ),
            AgentTaskResult(
                task_id="refund_lookup",
                success=True,
                query_bundle=QueryBundle(rows=[{"order_id": "order_1", "buyer_name": "buyer_1", "refund_create_time": "2026-07-10 10:00:00", "pay_amt": 121.5}]),
            ),
        ]
    )

    answer = deterministic_cross_task_detail_answer("查询订单 order_1 的退款信息", plan, run)

    assert "2026-07-10" in answer
    assert "122" in answer
    assert "buyer_1" in answer
    assert "2026-07-10 10:00:00" in answer
    assert "121.50元" in answer


def test_realtime_fallback_requires_an_explicit_semantic_metric_mapping():
    resolution = seal_semantic_metric_resolution(governed_resolution())

    assert governed_realtime_metric_resolution(resolution, "orders", "orders_rt", {}, PlanningAssetPack()) == {}
