from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import ExtractedKeywords, QuestionCategory
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetLimits,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeKernel
from merchant_ai.services.routing import SemanticTopicRouterService


class FakeTopicAssets:
    def __init__(self) -> None:
        self.manifests = {
            "电商交易": [
                {
                    "businessSummary": "订单交易事实，可按商品分析销量和成交。",
                    "dataGrain": "订单商品明细",
                    "preferredFor": ["METRIC", "TOPN", "GROUP_AGG"],
                }
            ],
            "商品管理": [
                {
                    "businessSummary": "商品主数据和商品属性；不承载交易事实。",
                    "dataGrain": "商品快照",
                    "preferredFor": ["DETAIL"],
                }
            ],
            "电商退货": [
                {
                    "businessSummary": "退款、退货和售后事实。",
                    "dataGrain": "退款单明细",
                    "preferredFor": ["METRIC", "TOPN"],
                }
            ],
            "供应链": [
                {
                    "businessSummary": "库存、入库、出库和供应链履约事实。",
                    "dataGrain": "商品供应链快照",
                    "preferredFor": ["METRIC", "TOPN"],
                }
            ],
        }

    def all_topic_names(self) -> list[str]:
        return list(self.manifests)

    def load_manifest(self, topic: str) -> list[dict[str, Any]]:
        return list(self.manifests.get(topic) or [])

    @staticmethod
    def load_topic_contract(topic: str) -> dict[str, Any]:
        return {
            "topic": topic,
            "categoryId": topic,
            "displayName": topic,
            "aliases": [],
        }

    def resolve_topic_category(self, value: Any) -> QuestionCategory:
        raw = str(getattr(value, "value", value) or "")
        return QuestionCategory(raw if raw in self.manifests else "UNKNOWN")

    def topic_names_for_categories(
        self,
        categories: list[QuestionCategory],
    ) -> list[str]:
        wanted = {str(item) for item in categories}
        return [topic for topic in self.manifests if topic in wanted]


class FakeLlm:
    configured = True
    last_error = ""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def tool_json_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        fallback: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system": system_prompt,
                "user": json.loads(user_prompt),
                "tool": tool,
                "fallback": fallback,
                "timeout": timeout_seconds,
            }
        )
        return self.responses.pop(0) if self.responses else {}


class UnavailableLlm:
    configured = False
    last_error = "not configured"


class ProviderFailureLlm(FakeLlm):
    def __init__(self) -> None:
        super().__init__([])

    def tool_json_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        tool: dict[str, Any],
        fallback: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        super().tool_json_chat(
            system_prompt,
            user_prompt,
            tool,
            fallback,
            timeout_seconds,
        )
        self.last_error = "timeout: provider call exceeded 12 seconds"
        return {}


class WrongKeywordService:
    @staticmethod
    def extract(question: str) -> ExtractedKeywords:
        return ExtractedKeywords(
            normalized_question=question,
            topic_scores={"商品管理": 99.0},
            topic_keywords=["商品"],
        )


def settings(**updates: Any) -> SimpleNamespace:
    values = {
        "topic_semantic_route_timeout_seconds": 12,
        "topic_semantic_route_max_attempts": 2,
        "topic_semantic_route_min_confidence": 0.55,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_router_reads_complete_topic_cards_and_returns_only_scope() -> None:
    assets = FakeTopicAssets()
    llm = FakeLlm(
        [
            {
                "status": "RESOLVED",
                "relevantTopics": ["电商交易", "商品管理"],
                "confidence": 0.96,
            }
        ]
    )
    router = SemanticTopicRouterService(settings(), assets, llm=llm)

    decision = router.route_with_budget(
        "最近10天卖得最多的商品是哪个？品牌和货号是多少？"
    )

    assert decision.primary_topic == QuestionCategory.UNKNOWN
    assert decision.candidate_topics == [
        QuestionCategory("电商交易"),
        QuestionCategory("商品管理"),
    ]
    assert decision.routing_mode == "semantic_topic_scope"
    assert decision.selection_evidence["keywordRoutingUsed"] is False
    prompt = llm.calls[0]
    assert [item["topic"] for item in prompt["user"]["topicDirectory"]] == assets.all_topic_names()
    assert "最近10天卖得最多的商品" in prompt["system"]
    assert "不代表主表" in prompt["system"]
    assert set(prompt["tool"]["function"]["parameters"]["properties"]) == {
        "status",
        "relevantTopics",
        "confidence",
        "ambiguityReason",
    }
    assert "metricKeywords" not in json.dumps(prompt["user"], ensure_ascii=False)


def test_single_topic_still_does_not_assign_primary_or_plan_query() -> None:
    router = SemanticTopicRouterService(
        settings(),
        FakeTopicAssets(),
        llm=FakeLlm(
            [
                {
                    "status": "RESOLVED",
                    "relevantTopics": ["电商交易"],
                    "confidence": 0.98,
                }
            ]
        ),
    )

    decision = router.route_with_budget("最近7天订单量是多少？")

    assert decision.primary_topic == QuestionCategory.UNKNOWN
    assert decision.candidate_topics == [QuestionCategory("电商交易")]
    assert decision.dimension_topics == []


def test_invalid_model_topic_is_rejected_and_repaired_once() -> None:
    llm = FakeLlm(
        [
            {
                "status": "RESOLVED",
                "relevantTopics": ["销售中心"],
                "confidence": 0.99,
            },
            {
                "status": "RESOLVED",
                "relevantTopics": ["电商交易"],
                "confidence": 0.93,
            },
        ]
    )
    router = SemanticTopicRouterService(settings(), FakeTopicAssets(), llm=llm)

    decision = router.route_with_budget("订单表现如何？")

    assert decision.candidate_topics == [QuestionCategory("电商交易")]
    assert decision.selection_evidence["llmAttempts"] == 2
    assert "未发布 Topic" in llm.calls[1]["user"]["repair"]["previousResultProblem"]


def test_ambiguous_result_keeps_all_model_selected_topics_without_primary() -> None:
    router = SemanticTopicRouterService(
        settings(),
        FakeTopicAssets(),
        llm=FakeLlm(
            [
                {
                    "status": "AMBIGUOUS",
                    "relevantTopics": ["电商交易", "供应链", "商品管理"],
                    "confidence": 0.48,
                    "ambiguityReason": "“最多”没有说明是销量还是库存",
                }
            ]
        ),
    )

    decision = router.route_with_budget("哪个商品最多？")

    assert decision.primary_topic == QuestionCategory.UNKNOWN
    assert decision.routing_mode == "semantic_topic_ambiguous"
    assert decision.candidate_topics == [
        QuestionCategory("电商交易"),
        QuestionCategory("供应链"),
        QuestionCategory("商品管理"),
    ]
    assert "销量还是库存" in decision.reason


def test_llm_unavailable_opens_full_published_directory_without_keyword_fallback() -> None:
    assets = FakeTopicAssets()
    router = SemanticTopicRouterService(settings(), assets, llm=UnavailableLlm())

    decision = router.route_with_budget("随便一种用户问法")

    assert decision.routing_mode == "semantic_topic_open_directory"
    assert decision.candidate_topics == [QuestionCategory(item) for item in assets.all_topic_names()]
    assert decision.selection_evidence["keywordRoutingUsed"] is False
    assert "未回退关键词路由" in decision.reason


def test_provider_timeout_does_not_spend_a_second_llm_call_on_format_repair() -> None:
    llm = ProviderFailureLlm()
    router = SemanticTopicRouterService(settings(), FakeTopicAssets(), llm=llm)

    decision = router.route_with_budget("最近7天订单量")

    assert len(llm.calls) == 1
    assert decision.routing_mode == "semantic_topic_open_directory"
    assert "timeout" in decision.selection_evidence["detail"]


def test_kernel_does_not_feed_keyword_topic_scores_into_semantic_router() -> None:
    assets = FakeTopicAssets()
    router = SemanticTopicRouterService(
        settings(),
        assets,
        llm=FakeLlm(
            [
                {
                    "status": "RESOLVED",
                    "relevantTopics": ["电商交易", "商品管理"],
                    "confidence": 0.96,
                }
            ]
        ),
    )
    kernel = GroundedRuntimeKernel(
        assets,
        keyword_service=WrongKeywordService(),
        topic_router=router,
    )
    session = kernel.new_session(
        "最近10天卖得最多的商品是哪个？品牌和货号是多少？",
        "merchant-1",
    )

    kernel.route_topic(session)

    assert session.keywords.topic_scores == {"商品管理": 99.0}
    assert session.workspace_topics == ["电商交易", "商品管理"]
    assert session.routing.selection_evidence["keywordRoutingUsed"] is False


def test_topic_llm_call_uses_shared_runtime_budget_and_stage_telemetry() -> None:
    budget = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=90,
            max_llm_calls=8,
            max_tool_calls=60,
            max_doris_queries=12,
        )
    )
    router = SemanticTopicRouterService(
        settings(topic_semantic_route_max_attempts=1),
        FakeTopicAssets(),
        llm=FakeLlm(
            [
                {
                    "status": "RESOLVED",
                    "relevantTopics": ["电商交易"],
                    "confidence": 0.98,
                }
            ]
        ),
    )

    router.route_with_budget("最近7天订单量", runtime_budget=budget)
    report = budget.report()

    assert report["usage"]["llmCalls"] == 1
    assert report["usage"]["llmCallsByName"] == {"semantic_topic_router": 1}
    assert report["stages"]["llm.topic_route.attempt_1"]["calls"] == 1
