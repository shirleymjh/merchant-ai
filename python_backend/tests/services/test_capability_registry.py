from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.models import (
    AnswerMode,
    FastUnderstandingResult,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionCategory,
    QuestionIntent,
    QueryPlan,
)
from merchant_ai.services.capabilities import (
    CapabilityRegistry,
    features_from_fast_understanding,
    features_from_query_plan,
)
from merchant_ai.services.planning import semantic_fast_path_can_bypass_configured_llm


def test_capability_registry_loads_versioned_runtime_contracts():
    registry = CapabilityRegistry.from_settings(get_settings())

    assert registry.version == "1.0.0"
    assert registry.contract("metric_fast_entry") is not None
    assert registry.contract("semantic_plan_fast").target_p95_ms == 5000


def test_metric_fast_entry_accepts_only_typed_single_metric_features():
    registry = CapabilityRegistry.from_settings(get_settings())
    eligible = features_from_fast_understanding(
        FastUnderstandingResult(
            intent_kind="metric_query",
            complexity="simple",
            analysis_intent="lookup",
            topics=[QuestionCategory.TRADE],
            metric_phrases=["订单量"],
            needs_planner=False,
            confidence=0.99,
        )
    )
    analysis = eligible.model_copy(
        update={
            "intent_kind": "analysis",
            "complexity": "complex",
            "analysis_intent": "attribution",
            "requires_explanation": True,
            "needs_planner": True,
        }
    )

    assert registry.evaluate("metric_fast_entry", eligible).eligible is True
    assert registry.evaluate("metric_fast_entry", analysis).eligible is False


def test_semantic_plan_fast_rejects_multi_node_or_explanation_plan():
    simple = QueryPlan(
        intents=[
                QuestionIntent(
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.METRIC,
                    category=QuestionCategory.TRADE,
                    plan_task_id="order_metric",
                    preferred_table="dwm_trade_order_detail_di",
                    metric_name="order_detail_cnt",
                    metric_column="order_detail_cnt",
                    metric_resolution={"metricKey": "order_detail_cnt", "semanticRefId": "semantic:trade:order_count"},
                )
            ],
        question_understanding={"analysisIntent": "lookup", "requiresExplanation": False},
    )
    complex_plan = simple.model_copy(
        update={
            "intents": [
                *simple.intents,
                    QuestionIntent(
                        intent_type=IntentType.VALID,
                        answer_mode=AnswerMode.METRIC,
                        category=QuestionCategory.REFUND,
                        plan_task_id="refund_metric",
                        preferred_table="dwm_trade_refund_detail_di",
                        metric_name="refund_amt",
                        metric_column="refund_amt",
                        metric_resolution={"metricKey": "refund_amt", "semanticRefId": "semantic:refund:refund_amt"},
                    ),
            ],
            "question_understanding": {"analysisIntent": "attribution", "requiresExplanation": True},
        }
    )

    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pt", "order_detail_cnt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "refund_amt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["order_detail_cnt"],
                source_ref_id="semantic:trade:order_count",
            ),
            PlanningAssetEntry(
                key="refund_amt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_amt"],
                source_ref_id="semantic:refund:refund_amt",
            ),
        ],
    )

    assert semantic_fast_path_can_bypass_configured_llm("最近7天订单量", simple, pack) is True
    assert features_from_query_plan(complex_plan).requires_explanation is True
    assert semantic_fast_path_can_bypass_configured_llm("分析订单和退款原因", complex_plan, pack) is False


def test_policy_does_not_try_fast_metric_without_structured_fast_understanding():
    policy = V2AgentPolicy(get_settings().model_copy(update={"lead_agent_autonomous_enabled": False}))

    decision = policy.decide(
        {
            "data_discovered": False,
            "topic_routed": True,
            "fast_understood": True,
            "question": "分析一份还没有上传的报告",
        }
    )

    assert decision.selected_action == "retrieve_knowledge"
