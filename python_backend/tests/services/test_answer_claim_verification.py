import json
from datetime import datetime

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.answer import AnswerComposeService
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


def test_verified_facts_are_task_and_cell_bound():
    plan = detail_plan()
    facts = build_verified_facts(plan, detail_run({"order_id": "order_1", "pay_amt": 121.5}))

    assert {(fact.task_id, fact.column, fact.value) for fact in facts} == {
        ("order_detail", "order_id", "order_1"),
        ("order_detail", "pay_amt", 121.5),
    }
    assert all(fact.fact_id.startswith("fact_") for fact in facts)


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
