from merchant_ai.models import RouteSpanType
from merchant_ai.services.routing import KeywordExtractService, extract_route_lexical_spans
from merchant_ai.services.time_semantics import extract_temporal_lexical_spans


def test_previous_period_span_is_not_classified_as_top_n_ranking() -> None:
    question = "最近30天订单量、退款金额和退款率的趋势怎么样？与前30天相比有什么变化？"
    keywords = KeywordExtractService().extract(question)
    temporal = [span for span in keywords.lexical_spans if span.span_type == RouteSpanType.TEMPORAL]

    assert keywords.ranking_keywords == []
    assert keywords.analysis_intent != "ranking"
    assert [(span.text, span.start, span.end) for span in temporal] == [
        ("最近30天", question.index("最近30天"), question.index("最近30天") + len("最近30天")),
        ("前30天", question.index("前30天"), question.index("前30天") + len("前30天")),
    ]
    assert all(question[span.start : span.end] == span.text for span in temporal)
    canonical = extract_temporal_lexical_spans(question)
    assert [(span.value, span.unit, span.role) for span in canonical] == [
        (30, "day", "primary"),
        (30, "day", "previous_period"),
    ]


def test_true_top_n_span_remains_ranking_beside_a_time_window() -> None:
    question = "最近30天订单量前10的商品"
    keywords = KeywordExtractService().extract(question)
    ranking = [span for span in keywords.lexical_spans if span.span_type == RouteSpanType.RANKING]

    assert keywords.ranking_keywords == ["前10"]
    assert keywords.analysis_intent == "ranking"
    assert len(ranking) == 1
    assert ranking[0].text == "前10"
    assert ranking[0].start == question.index("前10")
    assert ranking[0].end == ranking[0].start + len("前10")


def test_typed_span_arbitration_excludes_only_overlapping_ranking_candidate() -> None:
    question = "前30天与前10商品"
    spans = extract_route_lexical_spans(question)

    assert [(span.span_type, span.text) for span in spans] == [
        (RouteSpanType.TEMPORAL, "前30天"),
        (RouteSpanType.RANKING, "前10"),
    ]
    assert all(question[span.start : span.end] == span.text for span in spans)


def test_shared_temporal_lexer_parses_dynamic_value_unit_and_exact_boundaries() -> None:
    question = "过去 12 周订单量前5商品"
    keywords = KeywordExtractService().extract(question)
    temporal = [span for span in keywords.lexical_spans if span.span_type == RouteSpanType.TEMPORAL]

    assert len(temporal) == 1
    assert (temporal[0].value, temporal[0].unit, temporal[0].role) == (12, "week", "primary")
    assert temporal[0].start == question.index("过去")
    assert temporal[0].end == question.index("周") + len("周")
    assert question[temporal[0].start : temporal[0].end] == temporal[0].text
    assert keywords.ranking_keywords == ["前5"]
