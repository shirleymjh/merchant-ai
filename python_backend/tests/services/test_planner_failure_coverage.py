import pytest

from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.workflow import (
    create_workflow,
    planner_degraded_state,
    planner_degraded_stops_expensive_work,
    validation_with_question_coverage,
)
from merchant_ai.models import (
    AgentRunResult,
    AnswerMode,
    GraphValidationResult,
    IntentType,
    PlanningAssetPack,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    QuestionRoute,
    RecallBundle,
    RoutingDecision,
)
from merchant_ai.services.assets import PlanningAssetPackBuilder, SkillLoader, TopicAssetService
from merchant_ai.services.planning import QueryGraphPlanner


class TimeoutPlannerLlm:
    configured = True
    last_error = "timeout: provider call exceeded 20 seconds"
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None, **kwargs):
        return {}


def compact_pack(question: str, categories: list[QuestionCategory]) -> PlanningAssetPack:
    settings = get_settings()
    return PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        RecallBundle(),
        categories,
    )


def single_domain_plan(question: str, pack: PlanningAssetPack, table: str) -> QueryPlan:
    columns = pack.known_columns(table)
    return QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="single_domain_fallback",
                preferred_table=table,
                required_evidence=columns[:5],
                output_keys=columns[:5],
            )
        ]
    )


@pytest.mark.parametrize(
    ("question", "categories", "fallback_table", "missing_domains"),
    [
        (
            "最近30天退款订单，并查看商品发布时间",
            [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
            "dwm_trade_refund_detail_di",
            {"order", "goods"},
        ),
        (
            "最近30天用了优惠券的订单下单情况",
            [QuestionCategory.TRADE, QuestionCategory.COUPON],
            "dwm_coupon_detail_di",
            {"order"},
        ),
        (
            "商品审核拒绝后订单和退款情况",
            [QuestionCategory.GOODS, QuestionCategory.TRADE, QuestionCategory.REFUND],
            "dwm_goods_detail_df",
            {"order", "refund"},
        ),
    ],
)
def test_planner_provider_failure_rejects_single_domain_fallback(
    question,
    categories,
    fallback_table,
    missing_domains,
):
    pack = compact_pack(question, categories)
    incomplete = single_domain_plan(question, pack, fallback_table)
    planner = QueryGraphPlanner(TimeoutPlannerLlm())
    planner._semantic_fast_path = lambda _question, _pack: incomplete
    planner._entity_detail_fallback = lambda _question, _pack: incomplete
    planner._recalled_metric_diagnostic_fallback = lambda _question, _pack: QueryPlan()
    planner._multi_metric_trend_fallback = lambda _question, _pack: QueryPlan()

    plan, requests, reason = planner.plan(
        question,
        [],
        "",
        RecallBundle(),
        pack,
        [],
        [],
    )

    assert not plan.intents
    assert not requests
    assert "PLANNER_LLM_TIMEOUT" in reason
    rejected = {
        item.rsplit(":", 1)[-1]
        for item in plan.agent_trace
        if "QUESTION_DOMAIN_NOT_COVERED" in item
    }
    assert missing_domains <= rejected
    assert "planner.failure_fallback=fail_closed_coverage" in plan.agent_trace


def test_hypothesis_validation_and_execution_reject_incomplete_question_coverage():
    question = "最近30天退款订单，并查看商品发布时间"
    pack = compact_pack(
        question,
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    incomplete = single_domain_plan(question, pack, "dwm_trade_refund_detail_di")
    validation = validation_with_question_coverage(
        question,
        incomplete,
        pack,
        GraphValidationResult(valid=True),
    )

    assert not validation.valid
    assert {gap.evidence for gap in validation.gaps if gap.code == "QUESTION_DOMAIN_NOT_COVERED"} >= {"order", "goods"}

    workflow = create_workflow(get_settings())
    execution = workflow._execute_hypothesis_plan(
        {"question": question, "planning_asset_pack": pack},
        {"hypothesisId": "partial_refund", "plan": incomplete, "round": 1},
    )

    assert not execution["runResult"].task_results
    assert execution["executionError"].startswith("QUESTION_COVERAGE_REJECTED:")


def test_planner_timeout_degraded_marker_stops_expensive_recovery_by_default():
    plan = QueryPlan(
        intents=[QuestionIntent(plan_task_id="fallback", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC)],
        agent_trace=[
            "PLANNER_LLM_TIMEOUT: timeout: provider call exceeded 20 seconds",
            "planner.semantic_fast_path=validated_after_llm_failure",
        ],
    )
    degraded = planner_degraded_state("timeout: provider call exceeded 20 seconds", plan, "SEMANTIC_FAST_PATH")
    state = {"planner_degraded": degraded}

    assert degraded["active"] is True
    assert degraded["timeout"] is True
    assert degraded["fallbackUsed"] is True
    assert planner_degraded_stops_expensive_work(state) is True


def test_policy_does_not_launch_hypothesis_or_skill_chain_after_planner_timeout():
    settings = get_settings().model_copy(
        update={
            "hypothesis_query_exploration_enabled": True,
            "lead_agent_autonomous_enabled": True,
        }
    )
    policy = V2AgentPolicy(settings)
    state = {
        "question": "GMV下降原因分析",
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "planner_provider_error": "timeout: provider call exceeded 20 seconds",
        "planner_degraded": {
            "active": True,
            "code": "PLANNER_LLM_TIMEOUT",
            "stopExpensivePostProcessing": True,
        },
        "plan": QueryPlan(),
        "planning_asset_pack": PlanningAssetPack(metrics=[]),
        "hypothesis_exploration_completed": False,
        "hypothesis_exploration": {"hypotheses": [{"hypothesisId": "h1"}, {"hypothesisId": "h2"}]},
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "agent_run_result": AgentRunResult(),
        "react_round": 4,
    }

    decision = policy.decide(state)

    assert decision.selected_action == "validate_graph"
    assert "explore_hypotheses" not in decision.available_actions
    assert "run_analysis_skill" not in decision.available_actions
    assert policy.hypothesis_recovery_needed(state) is False
    assert policy.analysis_skill_needed(state) is False
