import json
from datetime import datetime

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    MerchantInfo,
    NodePlanContract,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    ResolvedTimeRange,
    VerifiedEvidence,
)
from merchant_ai.services.answer import (
    AnswerComposeService,
    answer_requirement_coverage,
    deterministic_structured_answer,
    lightweight_answer_contract_verification,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier, build_verified_facts
from merchant_ai.services.evaluation import GoldenEvaluationService


class ClaimAnswerLlm:
    configured = True
    settings = get_settings()

    def __init__(self, answer):
        self.answer = answer
        self.payload = {}
        self.payloads = []

    def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
        self.payload = json.loads(user_prompt)
        self.payloads.append(self.payload)
        return self.answer


def test_mandatory_skeleton_does_not_duplicate_a_ranking_already_contained_in_it():
    service = object.__new__(AnswerComposeService)
    ranking = """当前查询范围内，按工单量排序如下：

| 商品 | 工单量 |
| --- | --- |
| spu_1 | 3 |"""
    skeleton = ranking + "\n\n说明：\n- 数据截至 2026-07-11。"

    answer = service._ensure_mandatory_answer_skeleton(
        ranking,
        skeleton,
        "最近30天工单量最多的商品",
        QueryPlan(),
        None,
    )

    assert answer == skeleton
    assert answer.count("当前查询范围内") == 1


def test_compound_ranking_lookup_answer_uses_field_label_without_duplicate_ranking():
    question = "最近30天工单量最多的商品，同时看这个商品的商品发布时间"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.TOPN,
                plan_task_id="ticket_rank",
                preferred_table="dwm_cs_ticket_detail_di",
                metric_name="ticket_cnt",
                group_by_column="spu_id",
                group_by_name="商品id",
                output_keys=["spu_id", "ticket_cnt"],
                required_evidence=["spu_id", "ticket_cnt"],
                metric_resolution={
                    "metricKey": "ticket_cnt",
                    "displayName": "客服工单明细量",
                    "sourceColumnLabels": {"spu_id": "商品id"},
                    "unit": "单",
                },
            ),
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="goods_lookup",
                preferred_table="dwm_goods_detail_df",
                output_keys=["spu_id", "spu_apply_create_time"],
                required_evidence=["spu_id", "spu_apply_create_time"],
                metric_resolution={
                    "sourceColumnLabels": {
                        "spu_id": "商品id",
                        "spu_apply_create_time": "商品发布时间",
                    }
                },
            ),
        ]
    )
    rank = QueryBundle(
        tables=["dwm_cs_ticket_detail_di"],
        rows=[{"spu_id": "1", "ticket_cnt": 37}],
        original_row_count=1,
    )
    lookup = QueryBundle(
        tables=["dwm_goods_detail_df"],
        rows=[
            {
                "spu_id": "1",
                "spu_apply_create_time": "2026-04-06 15:54:00",
            }
        ],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="ticket_rank", success=True, query_bundle=rank),
            AgentTaskResult(task_id="goods_lookup", success=True, query_bundle=lookup),
        ],
        merged_query_bundle=QueryBundle(
            tables=["dwm_cs_ticket_detail_di", "dwm_goods_detail_df"],
            rows=[*rank.rows, *lookup.rows],
            original_row_count=2,
        ),
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = AnswerComposeService(ClaimAnswerLlm("")).compose(
        question,
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        allow_llm=False,
    )

    assert "商品发布时间" in answer
    assert "2026-04-06 15:54:00" in answer
    assert "UNKNOWN" not in answer
    assert answer.count("按客服工单明细量排序") <= 1


def test_same_table_ranked_answer_keeps_all_explicit_lookup_fields():
    question = "最近10天卖的最多的商品是哪个？他的品牌名字，货号是多少"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.TOPN,
                plan_task_id="sales_rank",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="sku_cnt",
                group_by_column="spu_id",
                group_by_name="商品ID",
                output_keys=["spu_id", "商品名称", "品牌名称", "货号"],
                required_evidence=["spu_id", "商品名称", "品牌名称", "货号", "sku_cnt"],
                metric_resolution={
                    "metricKey": "sku_cnt",
                    "displayName": "销量",
                    "sourceColumnLabels": {
                        "spu_id": "商品ID",
                        "商品名称": "商品名称",
                        "品牌名称": "品牌名称",
                        "货号": "货号",
                    },
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["dwm_trade_order_detail_di"],
        rows=[
            {
                "spu_id": "6",
                "商品名称": "Cotton Casual Pants Khaki",
                "品牌名称": "brand_name_569",
                "货号": "article_id_569",
                "sku_cnt": 3,
            }
        ],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="sales_rank", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = deterministic_structured_answer(question, plan, run)

    assert "Cotton Casual Pants Khaki" in answer
    assert "brand_name_569" in answer
    assert "article_id_569" in answer
    assert "3" in answer
    assert "| 商品ID | 商品名称 | 品牌名称 | 货号 | 销量 |" in answer
    assert "| 指标 |" not in answer


def detail_plan():
    return QueryPlan(
        intents=[
            QuestionIntent(
                question="订单 order_1 为什么异常？",
                intent_type="VALID",
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="order_detail",
                preferred_table="orders",
                filter_column="order_id",
                filter_value="order_1",
                output_keys=["order_id", "pay_amt"],
                metric_resolution={"sourceColumnLabels": {"pay_amt": "支付金额"}},
            )
        ]
    )


def detail_run(row):
    bundle = QueryBundle(tables=["orders"], rows=[row], original_row_count=1)
    return AgentRunResult(
        task_results=[AgentTaskResult(task_id="order_detail", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )


def metric_plan():
    return QueryPlan(
        question_understanding={"source": "semantic_metric_fallback", "analysisIntent": "none"},
        intents=[
            QuestionIntent(
                question="最近7天GMV是多少？",
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_gmv",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                group_by_column="merchant_id",
                metric_resolution={
                    "semanticRefId": "semantic:经营画像:ads_merchant_profile:metric:order_gmv_amt_1d",
                    "metricKey": "order_gmv_amt_1d",
                    "ownerTable": "ads_merchant_profile",
                    "displayName": "总GMV金额",
                    "sourceColumns": ["order_gmv_amt_1d"],
                },
            )
        ],
    )


def metric_run(value=188.0):
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"merchant_id": "100", "order_gmv_amt_1d": value}],
        original_row_count=1,
    )
    return AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_gmv", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )


def comparison_metric_plan_and_run():
    question = "最近30天GMV是多少？与前30天相比有什么变化？"
    resolution = {
        "semanticRefId": "semantic:经营画像:ads_merchant_profile:metric:order_gmv_amt_1d",
        "metricKey": "order_gmv_amt_1d",
        "ownerTable": "ads_merchant_profile",
        "displayName": "GMV",
        "unit": "元",
        "valueFormat": "decimal",
        "sourceColumns": ["order_gmv_amt_1d"],
    }
    plan = QueryPlan(
        question_understanding={"timeWindowContract": {"requiresComparison": True}},
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="gmv_primary",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                metric_resolution=resolution,
                time_range=ResolvedTimeRange(label="最近30天", window_role="primary"),
            ),
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="gmv_comparison",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                metric_resolution=resolution,
                time_range=ResolvedTimeRange(label="前30天", window_role="comparison"),
            ),
        ],
    )
    primary = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"merchant_id": "100", "order_gmv_amt_1d": 27352.5}],
        original_row_count=1,
    )
    comparison = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"merchant_id": "100", "order_gmv_amt_1d": 27471.0}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="gmv_primary", success=True, query_bundle=primary),
            AgentTaskResult(task_id="gmv_comparison", success=True, query_bundle=comparison),
        ],
        merged_query_bundle=QueryBundle(
            tables=["ads_merchant_profile"],
            rows=[*primary.rows, *comparison.rows],
            original_row_count=2,
        ),
        verified_evidence=VerifiedEvidence(passed=True),
    )
    return question, plan, run


def test_verified_facts_are_task_and_cell_bound():
    plan = detail_plan()
    facts = build_verified_facts(plan, detail_run({"order_id": "order_1", "pay_amt": 121.5}))

    assert {(fact.task_id, fact.column, fact.value) for fact in facts} == {
        ("order_detail", "order_id", "order_1"),
        ("order_detail", "pay_amt", 121.5),
    }
    assert all(fact.fact_id.startswith("fact_") for fact in facts)


def test_verified_facts_load_complete_offloaded_rows_beyond_inline_preview(tmp_path):
    rows = [
        {"business_date": "2026-07-%02d" % day, "metric_value": day - 1}
        for day in range(1, 31)
    ]
    artifact = tmp_path / "metric_series_rows.json"
    artifact.write_text(json.dumps(rows), encoding="utf-8")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="show the governed daily series",
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="metric_series",
                preferred_table="daily_profile",
                metric_name="metric_value",
                metric_column="metric_value",
                group_by_column="business_date",
                output_keys=["business_date"],
                metric_resolution={
                    "metricKey": "metric_value",
                    "ownerTable": "daily_profile",
                    "sourceColumns": ["metric_value"],
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["daily_profile"],
        rows=rows[:20],
        original_row_count=30,
        offloaded_files=[str(artifact)],
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_series", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    facts = build_verified_facts(plan, run)

    assert any(fact.column == "business_date" and fact.value == "2026-07-30" for fact in facts)
    assert any(fact.column == "metric_value" and fact.value == 29 for fact in facts)


def temporal_series_plan_and_run(aggregation_policy):
    question = "最近3天流量指标是多少？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="metric_series",
                preferred_table="daily_profile",
                metric_name="metric_value",
                metric_column="metric_value",
                group_by_column="business_date",
                output_keys=["business_date"],
                metric_resolution={
                    "metricKey": "metric_value",
                    "displayName": "流量指标",
                    "sourceColumns": ["metric_value"],
                    "aggregationPolicy": aggregation_policy,
                },
            )
        ]
    )
    rows = [
        {"business_date": "2026-07-01", "metric_value": 1},
        {"business_date": "2026-07-02", "metric_value": 2},
        {"business_date": "2026-07-03", "metric_value": 3},
    ]
    bundle = QueryBundle(tables=["daily_profile"], rows=rows, original_row_count=3)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_series", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    return question, plan, run


def test_claim_verifier_accepts_period_total_only_for_published_period_rollup():
    question, plan, run = temporal_series_plan_and_run("period_rollup")

    result = AnswerClaimVerifier().verify(
        question,
        plan,
        run,
        "最近3天，流量指标周期合计为 6。",
    )

    assert result.passed is True
    assert {fact.aggregation_policy for fact in run.verified_facts if fact.column == "metric_value"} == {
        "period_rollup"
    }


def test_claim_verifier_rejects_series_sum_for_non_additive_policies():
    for aggregation_policy in ["latest_value_only", "daily_value_only", "ratio_of_sums", ""]:
        question, plan, run = temporal_series_plan_and_run(aggregation_policy)

        result = AnswerClaimVerifier().verify(
            question,
            plan,
            run,
            "最近3天，流量指标周期合计为 6。",
        )

        assert result.passed is False, aggregation_policy


def test_multi_metric_long_claim_binds_each_value_to_contract_alias_in_its_clause():
    question = "最近30天指标甲和指标乙有什么变化？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="multi_metric",
                preferred_table="daily_profile",
                metric_name="metric_a",
                metric_column="metric_a",
                group_by_column="business_date",
                output_keys=["business_date"],
                metric_resolution={"metricKey": "metric_a", "displayName": "指标甲"},
            )
        ]
    )
    rows = [
        {"business_date": "2026-07-01", "metric_a": 9, "metric_b": 100071},
        {"business_date": "2026-07-02", "metric_a": 2, "metric_b": 100100},
    ]
    bundle = QueryBundle(tables=["daily_profile"], rows=rows, original_row_count=2)
    contract = NodePlanContract(
        task_id="multi_metric",
        visible_columns=["business_date", "metric_a", "metric_b"],
        metric_specs=[
            {"metricName": "metric_a", "metricColumn": "metric_a", "displayName": "指标甲", "naturalName": "甲指标"},
            {"metricName": "metric_b", "metricColumn": "metric_b", "displayName": "指标乙", "naturalName": "乙指标"},
        ],
        group_by_column="business_date",
        time_window_contract={"timeColumn": "business_date"},
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="multi_metric", success=True, query_bundle=bundle, node_plan_contract=contract)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    verification = AnswerClaimVerifier().verify(
        question,
        plan,
        run,
        "指标甲从 9 下降到 2，减少 7；指标乙从 100071 上升到 100100，增加 29。",
    )

    assert verification.passed is True
    facts = build_verified_facts(plan, run)
    assert all({"指标乙", "乙指标"} <= set(fact.label_aliases) for fact in facts if fact.column == "metric_b")


def test_multi_metric_long_claim_cannot_borrow_a_different_clause_label():
    plan = detail_plan()
    bundle = QueryBundle(
        tables=["orders"],
        rows=[{"order_id": "order_1", "refund_amt": 122, "pay_amt": 121.5}],
        original_row_count=1,
    )
    contract = NodePlanContract(
        task_id="order_detail",
        visible_columns=["order_id", "refund_amt", "pay_amt"],
        metric_specs=[
            {"metricColumn": "refund_amt", "displayName": "退款金额"},
            {"metricColumn": "pay_amt", "displayName": "支付金额"},
        ],
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="order_detail", success=True, query_bundle=bundle, node_plan_contract=contract)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    verification = AnswerClaimVerifier().verify(
        "订单 order_1 的支付金额和退款金额是多少？",
        plan,
        run,
        "支付金额为 122；退款金额为 122。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:122"]


def test_local_day_direction_is_checked_against_adjacent_points_not_overall_trend():
    question = "最近3天指标值有什么变化？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="metric_series",
                preferred_table="daily_profile",
                metric_name="metric_value",
                metric_column="metric_value",
                group_by_column="business_date",
                output_keys=["business_date"],
                metric_resolution={"metricKey": "metric_value", "displayName": "指标值"},
            )
        ]
    )
    rows = [
        {"business_date": "2026-07-01", "metric_value": 10},
        {"business_date": "2026-07-02", "metric_value": 20},
        {"business_date": "2026-07-03", "metric_value": 5},
    ]
    bundle = QueryBundle(tables=["daily_profile"], rows=rows, original_row_count=3)
    contract = NodePlanContract(
        task_id="metric_series",
        visible_columns=["business_date", "metric_value"],
        metric_specs=[{"metricName": "metric_value", "metricColumn": "source_value", "displayName": "指标值"}],
        group_by_column="business_date",
        time_window_contract={"timeColumn": "business_date"},
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_series", success=True, query_bundle=bundle, node_plan_contract=contract)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    verification = AnswerClaimVerifier().verify(
        question,
        plan,
        run,
        "指标值从 2026-07-01 的 10 变化到 2026-07-03 的 5，整体下降 5；"
        "指标值在 2026-07-02 较前一日上升 10。",
    )

    assert verification.passed is True


def test_direction_only_claim_is_verified_against_the_executed_series():
    question = "最近两天余额有没有异常？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="balance_series",
                preferred_table="daily_profile",
                metric_name="balance",
                metric_column="balance",
                group_by_column="pt",
                output_keys=["pt", "balance"],
                metric_resolution={
                    "metricKey": "balance",
                    "displayName": "余额",
                    "sourceColumns": ["balance"],
                    "timeColumn": "pt",
                },
            )
        ]
    )
    rows = [
        {"pt": "2026-07-10", "balance": 100},
        {"pt": "2026-07-11", "balance": 100},
    ]
    bundle = QueryBundle(tables=["daily_profile"], rows=rows, original_row_count=2)
    contract = NodePlanContract(
        task_id="balance_series",
        visible_columns=["pt", "balance"],
        metric_specs=[{"metricName": "balance", "metricColumn": "balance", "displayName": "余额"}],
        group_by_column="pt",
        time_window_contract={"timeColumn": "pt"},
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="balance_series",
                success=True,
                query_bundle=bundle,
                node_plan_contract=contract,
            )
        ],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    verification = AnswerClaimVerifier().verify(
        question,
        plan,
        run,
        "余额明显下降并出现异常。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_trend_direction:down"]


def test_answer_claim_verifier_rejects_value_missing_from_evidence():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 为什么异常？",
        detail_plan(),
        detail_run({"order_id": "order_1"}),
        "订单 order_1 的支付金额为 122元。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:122"]


def test_question_time_window_cannot_support_an_unrelated_metric_value():
    verification = AnswerClaimVerifier().verify(
        "最近30天订单支付金额是多少？",
        detail_plan(),
        detail_run({"order_id": "order_1", "pay_amt": 121.5}),
        "最近30天，订单支付金额为30元。",
    )

    assert verification.passed is False
    assert "unsupported_value:30" in verification.unsupported_claims[0].reasons


def test_equal_number_from_different_metric_cannot_support_claim():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 的支付金额是多少？",
        detail_plan(),
        detail_run({"order_id": "order_1", "order_cnt": 122}),
        "订单 order_1 的支付金额为122元。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:122"]


def test_explanation_heading_cannot_hide_unsupported_fact():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 为什么异常？",
        detail_plan(),
        detail_run({"order_id": "order_1", "pay_amt": 121.5}),
        "订单信息如下。\n说明：支付金额为999元。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:999"]


def test_full_datetime_is_one_claim_and_must_bind_to_an_executed_fact():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 的创建时间是什么？",
        detail_plan(),
        detail_run({"order_id": "order_1", "created_time": datetime(2026, 7, 13, 12, 34, 56)}),
        "订单 order_1 的创建时间为 2026-07-13 12:34:56。",
    )

    assert verification.passed is True
    assert verification.claims[0].numeric_values == ["2026-07-13 12:34:56"]
    assert verification.claims[0].fact_ids

    changed_time = AnswerClaimVerifier().verify(
        "订单 order_1 的创建时间是什么？",
        detail_plan(),
        detail_run({"order_id": "order_1", "created_time": datetime(2026, 7, 13, 12, 34, 56)}),
        "订单 order_1 的创建时间为 2026-07-13 12:34:57。",
    )

    assert changed_time.passed is False
    assert changed_time.unsupported_claims[0].reasons == ["unsupported_value:2026-07-13 12:34:57"]


def test_markdown_table_uses_headers_and_ignores_only_deterministic_row_position():
    verification = AnswerClaimVerifier().verify(
        "支付金额排名是什么？",
        detail_plan(),
        detail_run({"merchant_id": "merchant_1", "pay_amt": 121.5}),
        "| 排名 | 商户ID | 支付金额 |\n"
        "| ---: | --- | ---: |\n"
        "| 1 | merchant_1 | 121.5 |",
    )

    assert verification.passed is True
    assert verification.claims[0].text == "排名：1；商户ID：merchant_1；支付金额：121.5"
    assert verification.claims[0].numeric_values == ["121.5"]
    assert verification.claims[0].fact_ids


def test_ranking_accepts_numeric_entity_id_via_governed_group_label() -> None:
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天工单量最多的商品",
                intent_type="VALID",
                answer_mode=AnswerMode.TOPN,
                plan_task_id="ticket_top_product",
                preferred_table="dwm_cs_ticket_detail_di",
                metric_column="ticket_id",
                metric_name="ticket_cnt",
                metric_formula="COUNT(DISTINCT ticket_id)",
                metric_specs=[
                    {
                        "metricName": "ticket_cnt",
                        "metricColumn": "ticket_id",
                        "displayName": "客服工单明细量",
                        "sourceColumns": ["ticket_id"],
                    }
                ],
                metric_resolution={
                    "metricKey": "ticket_cnt",
                    "displayName": "客服工单明细量",
                    "sourceColumns": ["ticket_id"],
                },
                group_by_column="spu_id",
                group_by_name="商品id",
                output_keys=["spu_id", "ticket_cnt"],
                required_evidence=["spu_id", "ticket_cnt"],
            )
        ],
        final_evidence_column_hints={
            "ticket_top_product": ["spu_id", "ticket_cnt"]
        },
    )
    bundle = QueryBundle(
        tables=["dwm_cs_ticket_detail_di"],
        rows=[{"spu_id": "1", "ticket_cnt": 37}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="ticket_top_product",
                success=True,
                query_bundle=bundle,
            )
        ],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    verification = AnswerClaimVerifier().verify(
        plan.intents[0].question,
        plan,
        run,
        "| 商品id | 客服工单明细量 |\n"
        "| --- | ---: |\n"
        "| 1 | 37 |",
    )

    assert verification.passed is True
    assert verification.claims[0].fact_ids


def test_markdown_table_metric_value_still_requires_matching_header_semantics():
    verification = AnswerClaimVerifier().verify(
        "支付金额排名是什么？",
        detail_plan(),
        detail_run({"merchant_id": "merchant_1", "refund_amt": 121.5}),
        "| 序号 | 商户ID | 支付金额 |\n"
        "| ---: | --- | ---: |\n"
        "| 1 | merchant_1 | 121.5 |",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:121.5"]


def test_statistical_formula_constants_are_not_treated_as_business_facts():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 的支付金额是多少？",
        detail_plan(),
        detail_run({"order_id": "order_1", "pay_amt": 121.5}),
        "订单 order_1 的支付金额为121.5元。\n"
        "统计说明：支付笔数按已发布语义公式 "
        "SUM(CASE WHEN pay_status = 1 THEN 1 ELSE 0 END) 计算。",
    )

    assert verification.passed is True
    assert all("1" not in claim.numeric_values and "0" not in claim.numeric_values for claim in verification.claims)


def test_statistical_formula_exemption_cannot_hide_an_unverified_business_value():
    verification = AnswerClaimVerifier().verify(
        "订单 order_1 的支付金额是多少？",
        detail_plan(),
        detail_run({"order_id": "order_1", "pay_amt": 121.5}),
        "统计说明：支付金额为999元；支付笔数按已发布语义公式 "
        "SUM(CASE WHEN pay_status = 1 THEN 1 ELSE 0 END) 计算。",
    )

    assert verification.passed is False
    assert verification.unsupported_claims[0].numeric_values == ["999"]
    assert verification.unsupported_claims[0].reasons == ["unsupported_value:999"]


def test_answer_compose_discards_unsupported_llm_fact_and_uses_deterministic_fallback():
    llm = ClaimAnswerLlm("订单 order_1 的支付金额为 122元。")
    service = AnswerComposeService(llm)

    answer = service.compose(
        "订单 order_1 为什么异常？",
        MerchantInfo(merchant_id="100"),
        detail_plan(),
        detail_run({"order_id": "order_1"}),
        "",
    )

    assert "122" not in answer
    assert service.last_answer_claim_trace["passed"] is True
    assert service.last_answer_claim_trace["fallbackUsed"] is True
    assert any(payload.get("verifiedFacts") for payload in llm.payloads)
    assert len(llm.payloads) == 1


def test_answer_compose_keeps_supported_llm_fact():
    llm = ClaimAnswerLlm("订单 order_1 的支付金额为 121.5元。")
    service = AnswerComposeService(llm)

    answer = service.compose(
        "订单 order_1 为什么异常？",
        MerchantInfo(merchant_id="100"),
        detail_plan(),
        detail_run({"order_id": "order_1", "pay_amt": 121.5}),
        "",
    )

    assert "121.5" in answer
    assert service.last_answer_claim_trace["passed"] is True
    assert service.last_answer_claim_trace["fallbackUsed"] is False
    assert len(llm.payloads) == 1


def test_answer_compose_keeps_verified_single_metric_fallback_when_llm_fact_is_wrong():
    llm = ClaimAnswerLlm("最近7天总GMV金额为 999元。")
    service = AnswerComposeService(llm)

    answer = service.compose(
        "最近7天GMV是多少？",
        MerchantInfo(merchant_id="100"),
        metric_plan(),
        metric_run(188.0),
        "",
    )

    assert "188" in answer
    assert "999" not in answer
    assert "以结构化数据区域为准" not in answer
    assert service.last_answer_claim_trace["passed"] is True
    assert service.last_answer_claim_trace["fallbackUsed"] is False
    assert llm.payloads == []


def test_lightweight_contract_allows_absolute_rendering_of_negative_change():
    question, plan, run = comparison_metric_plan_and_run()

    answer = deterministic_structured_answer(question, plan, run)
    verification = lightweight_answer_contract_verification(question, plan, run, answer)

    assert "下降 118.5元" in answer
    assert verification is not None
    assert verification.passed is True
    assert verification.unsupported_claims == []


def test_comparison_answer_keeps_verified_values_while_disclosing_missing_rate():
    _, plan, run = comparison_metric_plan_and_run()
    question = "最近30天GMV和退款率分别是多少？与前30天相比有什么变化？"
    service = AnswerComposeService(ClaimAnswerLlm(""))

    answer = service.compose(
        question,
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        allow_llm=False,
    )

    assert "缺少的关键结果：退款率" in answer
    assert "GMV为 27352.5元" in answer
    assert "下降 118.5元" in answer
    assert service.last_answer_claim_trace["passed"] is True


def test_finalize_answer_recovers_verified_facts_when_supplied_fallback_also_fails():
    service = AnswerComposeService(ClaimAnswerLlm(""))
    plan = metric_plan()
    run = metric_run(188.0)

    answer = service._finalize_answer(
        "最近7天总GMV金额为 999元。",
        "最近7天GMV是多少？",
        plan,
        run,
        fallback_answer="这题目前不能直接给出完整结论。已覆盖的结果：GMV。",
    )

    assert "188" in answer
    assert "999" not in answer
    assert service.last_answer_claim_trace["passed"] is True
    assert service.last_answer_claim_trace["fallbackUsed"] is True
    assert any(
        "unsupported_extra_value:999" in claim.get("reasons", [])
        for claim in service.last_answer_claim_trace["rejectedClaims"]
    )


def test_rate_request_is_a_named_missing_requirement_instead_of_empty_block():
    question = "最近30天GMV和退款率分别是多少？"
    plan = metric_plan()
    run = metric_run(188.0)

    coverage = answer_requirement_coverage(question, plan, run)
    answer = AnswerComposeService(ClaimAnswerLlm("")).compose(
        question,
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        allow_llm=False,
    )

    assert coverage["shouldBlockDirectAnswer"] is True
    assert [item["label"] for item in coverage["missing"]] == ["退款率"]
    assert "缺少的关键结果：退款率" in answer
    assert "188" in answer


def test_rate_request_does_not_add_a_duplicate_gap_when_rate_metric_is_available():
    question = "最近30天退款率是多少？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="refund_rate",
                preferred_table="ads_merchant_profile",
                metric_name="direct_refund_rate_by_pay_order",
                metric_column="direct_refund_rate_by_pay_order",
                metric_resolution={
                    "metricKey": "direct_refund_rate_by_pay_order",
                    "displayName": "退款率",
                    "sourceColumns": ["direct_refund_rate_by_pay_order"],
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"direct_refund_rate_by_pay_order": 0.12}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="refund_rate", success=True, query_bundle=bundle)],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    coverage = answer_requirement_coverage(question, plan, run)

    assert coverage["missing"] == []
    assert coverage["shouldBlockDirectAnswer"] is False


def test_golden_answer_score_honors_runtime_claim_verification():
    service = GoldenEvaluationService(get_settings())
    case = {"answerMustMention": ["订单"]}

    failed = service._score_answer(
        case,
        "订单量为 122。",
        {"answerGuard": {"claimVerification": {"passed": False, "fallbackUsed": False}}},
    )
    recovered = service._score_answer(
        case,
        "订单结果已返回，未展示无法核验的数值。",
        {
            "answerGuard": {
                "claimVerification": {
                    "passed": True,
                    "fallbackUsed": True,
                    "rejectedClaims": [{"text": "订单量为 122。"}],
                }
            }
        },
    )

    assert failed["passed"] is False
    assert "answer_claim_verification_failed" in failed["reasons"]
    assert recovered["passed"] is True
    assert recovered["details"]["claimFallbackUsed"] is True
    assert recovered["details"]["rejectedClaimCount"] == 1


def test_skill_style_claims_accept_scope_formula_list_number_and_negated_gap():
    question = "最近30天退款率为什么高？"
    metric_ref = "semantic:经营画像:ads_merchant_profile:metric:refund_rate_by_pay_order"
    formula = "SUM(return_cnt_1d) / NULLIF(SUM(pay_order_cnt_1d), 0)"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="refund_rate",
                preferred_table="ads_merchant_profile",
                metric_name="refund_rate_by_pay_order",
                metric_column="refund_rate_by_pay_order",
                metric_specs=[
                    {
                        "metricName": "refund_rate_by_pay_order",
                        "displayName": "退款率",
                        "metricFormula": formula,
                        "sourceColumns": ["return_cnt_1d", "pay_order_cnt_1d"],
                        "semanticRefId": metric_ref,
                        "ownerTable": "ads_merchant_profile",
                    }
                ],
                metric_resolution={
                    "metricKey": "refund_rate_by_pay_order",
                    "displayName": "退款率",
                    "sourceColumns": ["return_cnt_1d", "pay_order_cnt_1d"],
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"refund_rate_by_pay_order": 0.2295081967213115}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="refund_rate", success=True, query_bundle=bundle)
        ],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    answer = """基于最近30天（2026-06-18 至 2026-07-17）merchant_id=100 的已验证结果：
1. 退款率为 22.95%。
口径：`SUM(return_cnt_1d) / NULLIF(SUM(pay_order_cnt_1d), 0)`。
不能确认退款率是否上升，因为缺少上一周期基准证据。"""

    verification = AnswerClaimVerifier().verify(
        question,
        plan,
        run,
        answer,
        support_context=(
            "最近30天 2026-06-18 至 2026-07-17 merchant_id=100"
        ),
    )

    assert verification.passed is True, [
        claim.model_dump(by_alias=True) for claim in verification.unsupported_claims
    ]
