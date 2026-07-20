from types import SimpleNamespace

from merchant_ai.models import RouteSpanType
from merchant_ai.services.routing import (
    KeywordExtractService,
    PreflightUnderstandingService,
    RouteSlotExtractor,
    planning_hints_from_extracted_keywords,
)


def test_action_tokens_remain_lexical_facts_without_becoming_analysis_intent() -> None:
    service = KeywordExtractService()
    question = "最近30天订单量和退款金额分别是多少？"

    keywords = service.extract(question)
    slots = RouteSlotExtractor(service.topic_assets).extract(question, keywords)
    hints = planning_hints_from_extracted_keywords(question, keywords)

    assert keywords.action_keywords == ["分别"]
    assert keywords.analysis_intent == "unresolved"
    assert slots.analysis_signals == []
    assert hints["analysisIntent"] == "unresolved"


def test_non_ranking_action_language_is_left_for_planner_understanding() -> None:
    service = KeywordExtractService()
    question = "最近30天订单量为什么下降？"

    keywords = service.extract(question)
    slots = RouteSlotExtractor(service.topic_assets).extract(question, keywords)

    assert keywords.action_keywords
    assert keywords.analysis_intent == "unresolved"
    assert not any(span.span_type == RouteSpanType.RANKING for span in keywords.lexical_spans)
    assert slots.analysis_signals == []


def test_typed_ranking_span_is_the_only_route_proven_analysis_intent() -> None:
    service = KeywordExtractService()
    question = "最近30天订单量前10的商品"

    keywords = service.extract(question)
    slots = RouteSlotExtractor(service.topic_assets).extract(question, keywords)

    assert keywords.analysis_intent == "ranking"
    assert any(span.span_type == RouteSpanType.RANKING for span in keywords.lexical_spans)
    assert slots.analysis_signals == ["typed_ranking_span"]


def test_preflight_analysis_signal_is_backed_by_typed_ranking_evidence() -> None:
    keyword_service = KeywordExtractService()
    slot_extractor = RouteSlotExtractor(keyword_service.topic_assets)
    preflight = PreflightUnderstandingService(
        settings=None,
        keyword_service=keyword_service,
        routing_service=None,
        slot_extractor=slot_extractor,
        semantic_classifier=None,
    )

    unresolved = preflight.surface_signals("最近30天订单量和退款金额分别是多少？")
    ranked = preflight.surface_signals("最近30天订单量前10的商品")

    assert unresolved["hasTypedRankingSpan"] is False
    assert unresolved["hasAnalysisIntent"] is False
    assert ranked["hasTypedRankingSpan"] is True
    assert ranked["hasAnalysisIntent"] is True


def test_two_metric_lookup_is_structurally_sent_to_planner() -> None:
    service = KeywordExtractService()
    question = "最近30天订单量和退款金额分别是多少？"
    keywords = service.extract(question)

    # This is the same structural branch used by fast_understand: metric count
    # decides whether semantic planning is required, while the connective does
    # not decide the relationship between those metrics.
    metric_phrases = list(keywords.metric_keywords or keywords.business_keywords or [])

    assert metric_phrases == ["订单量", "退款金额"]
    assert len(metric_phrases) >= 2
    assert keywords.analysis_intent == "unresolved"


def test_pending_context_reply_reaches_semantic_clarification_classifier() -> None:
    class Classifier:
        def __init__(self) -> None:
            self.calls = 0

        def classify_surface(
            self,
            question: str,
            surface_signals: dict,
            pending_context: bool = False,
        ) -> dict:
            self.calls += 1
            assert question == "第一个"
            assert pending_context is True
            assert surface_signals["pendingContext"] is True
            return {
                "enabled": True,
                "status": "success",
                "route": "CLARIFICATION_REPLY",
                "confidence": 0.95,
            }

    keyword_service = KeywordExtractService()
    classifier = Classifier()
    service = PreflightUnderstandingService(
        settings=SimpleNamespace(
            preflight_semantic_route_min_confidence=0.62
        ),
        keyword_service=keyword_service,
        routing_service=None,
        slot_extractor=RouteSlotExtractor(keyword_service.topic_assets),
        semantic_classifier=classifier,
    )

    result = service.understand("第一个", pending_context=True)

    assert classifier.calls == 1
    assert result.semantic_trace["route"] == "CLARIFICATION_REPLY"
    assert result.routing_decision.route == "BUSINESS"
