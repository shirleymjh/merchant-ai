from __future__ import annotations

from types import SimpleNamespace

import pytest

from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import MerchantQaWorkflow, business_scope_examples
from merchant_ai.models import ChatContext, QuestionCategory, RouteSlots, TopicRoutingDecision
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.deep_agent_runtime import DeepAgentWorkflowAdapter, _DianaLeadSession, _ResultSink
from merchant_ai.services.routing import KeywordExtractService, TopicRouterService


class _TopicPolicyHarness:
    """Exercise workflow Topic policy without constructing database services."""

    apply_topic_workspace_policy = MerchantQaWorkflow.apply_topic_workspace_policy
    session_topic_categories = MerchantQaWorkflow.session_topic_categories

    def __init__(self, topic_assets: TopicAssetService):
        self.recall_service = SimpleNamespace(topic_assets=topic_assets)

    def _topic_names_for_categories(self, categories):
        return self.recall_service.topic_assets.topic_names_for_categories(categories)


@pytest.fixture(scope="module")
def topic_runtime():
    assets = TopicAssetService(get_settings())
    return SimpleNamespace(
        assets=assets,
        policy=_TopicPolicyHarness(assets),
        keywords=KeywordExtractService(assets),
        router=TopicRouterService(assets),
    )


def test_inferred_context_is_not_marked_as_user_confirmed(topic_runtime) -> None:
    """Assistant-produced response context is history, not a user scope lock."""

    decision = TopicRoutingDecision(
        candidate_topics=[QuestionCategory("TRADE")],
        confidence=0.9,
    )
    state = {
        "question": "最近 7 天订单量怎么样",
        # This is the inferred context returned by a prior model turn.  Merely
        # carrying its Topic forward must not manufacture user confirmation.
        "request_context": ChatContext(
            topic="电商交易",
            topics=[QuestionCategory("TRADE")],
            clarification_resolved=False,
        ),
        "route_decision_trace": [],
    }

    topic_runtime.policy.apply_topic_workspace_policy(state, decision, RouteSlots(route_confidence=0.9))

    assert state["topic_workspace"]["confirmedByUser"] is False
    assert state["topic_workspace"]["expansionPolicy"] != "user_locked"


def test_strong_current_signal_can_replace_an_inferred_historical_workspace(topic_runtime) -> None:
    """An inferred prior Topic is a weak prior, never a sticky session boundary."""

    question = "查看商品审核拒绝明细"
    keywords = topic_runtime.keywords.extract(question)
    state = {
        "request_context": ChatContext(),
        "thread_context": {
            "topicWorkspace": {
                "mode": "topic_workspace",
                "topics": ["电商交易"],
                "topicIds": ["TRADE"],
                "confirmedByUser": False,
                "expansionPolicy": "on_gap_or_tool_request",
            }
        },
    }

    historical_topics = topic_runtime.policy.session_topic_categories(state)
    decision = topic_runtime.router.route(question, keywords, context_topics=historical_topics)

    assert keywords.topic_scores["GOODS"] >= 1.0
    assert decision.recall_topics() == [QuestionCategory("GOODS")]
    assert decision.routing_mode != "topic_workspace"


@pytest.mark.parametrize(
    ("question", "expected_policy", "expected_isolated"),
    [
        ("看看最近 7 天订单情况", "filesystem_manifest_browse", False),
        ("只看交易订单，不要扩展其他业务域", "user_locked", True),
    ],
)
def test_only_explicit_user_scope_language_locks_the_topic(
    topic_runtime,
    question: str,
    expected_policy: str,
    expected_isolated: bool,
) -> None:
    decision = TopicRoutingDecision(
        candidate_topics=[QuestionCategory("TRADE")],
        confidence=0.9,
    )
    state = {
        "question": question,
        "request_context": ChatContext(),
        "route_decision_trace": [],
    }

    topic_runtime.policy.apply_topic_workspace_policy(state, decision, RouteSlots(route_confidence=0.9))

    assert state["topic_workspace"]["expansionPolicy"] == expected_policy
    assert state["topic_workspace"]["isolated"] is expected_isolated


def test_no_topic_candidate_stays_in_discovery_without_disclosing_all_tables(topic_runtime) -> None:
    question = "一个资产尚未覆盖的新问题"
    keywords = topic_runtime.keywords.extract(question)
    decision = topic_runtime.router.route(question, keywords)

    assert decision.recall_topics() == []
    assert decision.routing_mode in {"open_discovery", "clarification_required"}

    state = {
        "react_round": 0,
        "topic_workspace": {
            "mode": decision.routing_mode,
            "topics": [],
            "topicIds": [],
            "confirmedByUser": False,
        },
    }
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = SimpleNamespace(policy=SimpleNamespace(max_main_actions=16))
    adapter.semantic_catalog = SimpleNamespace()
    session = _DianaLeadSession(state=state, sink=_ResultSink(), available_actions=())

    payload = adapter._turn_payload(session)

    assert "tableManifest" not in payload
    assert "knowledgeRoots" not in payload
    assert session.table_manifest_disclosed is False


def test_ticket_ranking_by_product_prioritizes_fact_then_leaves_l0_choice_to_core(topic_runtime) -> None:
    """Lineage ranks discovery; Core still chooses from candidate L0 summaries."""

    question = "最近30天工单量最高的商品"
    keywords = topic_runtime.keywords.extract(question)
    slots = RouteSlots(
        topic_candidates=[],
        route_confidence=keywords.confidence,
    )
    decision = topic_runtime.router.route(question, keywords, route_slots=slots)

    assert decision.recall_topics()[0] == QuestionCategory("CS_TICKET")
    assert {"CS_TICKET", "GOODS", "经营画像"}.issubset(set(keywords.topic_scores))
    assert keywords.topic_scores["CS_TICKET"] > keywords.topic_scores["GOODS"]
    assert keywords.topic_scores["CS_TICKET"] > keywords.topic_scores["经营画像"]


def test_profile_metric_stays_queryable_but_causal_question_discovers_detail_l0(topic_runtime) -> None:
    summary_question = "最近7天工单量是多少"
    summary_keywords = topic_runtime.keywords.extract(summary_question)
    summary_decision = topic_runtime.router.route(summary_question, summary_keywords)

    assert summary_decision.recall_topics() == [QuestionCategory("经营画像")]

    causal_question = "工单量为什么上涨"
    causal_keywords = topic_runtime.keywords.extract(causal_question)
    causal_decision = topic_runtime.router.route(causal_question, causal_keywords)

    assert causal_decision.recall_topics() == [QuestionCategory("CS_TICKET")]
    assert QuestionCategory("经营画像") in causal_keywords.topic_scores


def test_order_and_refund_summary_keeps_business_owners_separate_from_serving_topic(
    topic_runtime,
) -> None:
    """Topic is an automatic serving decision, not a merchant-facing category choice."""

    question = "只查询最近30天的订单数和退款总额"
    keywords = topic_runtime.keywords.extract(question)
    decision = topic_runtime.router.route(question, keywords)

    assert decision.selection_mode == "automatic"
    assert decision.recall_topics() == [QuestionCategory("经营画像")]
    assert decision.selection_evidence["queryShape"] == "summary_or_total"
    assert decision.selection_evidence["sameTableSummaryCandidate"] is True
    assert decision.selection_evidence["servingTopics"] == ["经营画像"]
    assert decision.selection_evidence["servingTables"] == ["ads_merchant_profile"]
    assert decision.selection_evidence["businessTopics"] == ["电商交易", "电商退货"]
    assert {
        item["metricKey"] for item in decision.selection_evidence["matchedMetrics"]
    } == {"order_cnt_1d", "refund_amt_1d"}
    assert "同一张已治理汇总表" in decision.reason
    assert "业务归属仍保留为 电商交易、电商退货" in decision.reason


def test_order_total_synonym_routes_to_profile_summary(topic_runtime) -> None:
    question = "最近7天订单总数是多少？"
    keywords = topic_runtime.keywords.extract(question)
    decision = topic_runtime.router.route(question, keywords)

    assert decision.recall_topics() == [QuestionCategory("经营画像")]
    assert decision.selection_evidence["queryShape"] == "summary_or_total"
    assert decision.selection_evidence["sameTableSummaryCandidate"] is True
    assert decision.selection_evidence["servingTopics"] == ["经营画像"]
    assert decision.selection_evidence["servingTables"] == [
        "ads_merchant_profile"
    ]
    assert {
        item["metricKey"]
        for item in decision.selection_evidence["matchedMetrics"]
    } == {"order_cnt_1d"}


def test_business_clarification_never_asks_merchant_to_choose_internal_topic() -> None:
    harness = SimpleNamespace()

    scope_prompt = MerchantQaWorkflow.build_scope_clarification_prompt(harness, {})
    ambiguous_prompt = MerchantQaWorkflow.build_topic_clarification_prompt(harness, {})

    assert "系统自动选择" in scope_prompt
    assert "系统自动选择" in ambiguous_prompt
    assert "请从已发布 Topic" not in ambiguous_prompt
    assert business_scope_examples() == []
    published_assets = SimpleNamespace(
        topic_contracts=lambda: [
            {
                "metadata": {
                    "clarificationContracts": {
                        "business_scope": {"options": ["已发布示例一", "已发布示例二"]}
                    }
                }
            }
        ]
    )
    assert business_scope_examples(published_assets) == ["已发布示例一", "已发布示例二"]


def test_core_observation_exposes_auditable_automatic_topic_selection(topic_runtime) -> None:
    question = "只查询最近30天的订单数和退款总额"
    keywords = topic_runtime.keywords.extract(question)
    decision = topic_runtime.router.route(question, keywords)
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = SimpleNamespace(policy=SimpleNamespace(max_main_actions=16))
    adapter.semantic_catalog = SimpleNamespace(topic_assets=topic_runtime.assets)
    session = _DianaLeadSession(
        state={
            "react_round": 0,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["经营画像"]},
            "topic_routing_decision": decision,
        },
        sink=_ResultSink(),
        available_actions=(),
    )

    payload = adapter._turn_payload(session)

    assert payload["topicSelection"]["userChoiceRequired"] is False
    assert payload["topicSelection"]["businessTopics"] == ["电商交易", "电商退货"]
    assert payload["topicSelection"]["servingTopics"] == ["经营画像"]
    assert payload["topicSelection"]["sameTableSummaryCandidate"] is True
