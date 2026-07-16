from datetime import datetime, timedelta, timezone

from merchant_ai.config import get_settings
from merchant_ai.models import KnowledgeRetrievalRequest, RecallBundle, RecallItem
from merchant_ai.services.context import ContextManager
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    business_rerank_recall_items,
    canonical_metric_family_owner,
    filter_recall_items_by_governance,
    limit_recall_items_by_source_type,
    merge_recall_items,
    metric_candidate_fusion_score,
    rrf_fuse_recall_items,
    rewrite_retrieval_query,
)


class LinkedVariantTopicAssets:
    def all_topic_names(self):
        return ["finance"]

    def load_manifest(self, topic):
        return [{"tableName": "account_metrics"}]

    def load_table_asset(self, topic, table):
        return {"status": "PUBLISHED"}

    def load_table_terms(self, topic, table):
        return []

    def load_table_metrics(self, topic, table):
        return [
            {
                "metricKey": "revenue_period",
                "businessName": "Revenue",
                "aliases": ["Revenue"],
                "formula": "SUM(revenue_amount)",
                "metricGrain": "account_period",
                "metricIntent": "summary",
                "aggregationPolicy": "period_rollup",
                "selectionGuidance": "Use for one value over a bounded period.",
                "temporalVariants": {
                    "series": {"metricKey": "revenue_daily"},
                    "alternatives": ["revenue_snapshot"],
                },
            },
            {
                "metricKey": "revenue_daily",
                "businessName": "Daily Revenue",
                "aliases": ["Daily Revenue"],
                "formula": "SUM(revenue_amount)",
                "metricGrain": "account_day",
                "metricIntent": "series",
                "aggregationPolicy": "daily_value_only",
                "selectionGuidance": "Use only with a daily grouping.",
                "temporalVariants": {"summary": "revenue_period"},
            },
            {
                "metricKey": "revenue_snapshot",
                "businessName": "Revenue Snapshot",
                "aliases": ["Revenue Snapshot"],
                "formula": "MAX(revenue_amount)",
                "metricGrain": "account_snapshot",
                "metricIntent": "snapshot",
                "aggregationPolicy": "latest_value_only",
                "selectionGuidance": "Use only for an explicit snapshot request.",
            },
        ]


def test_follow_up_query_is_rewritten_with_previous_user_question():
    request = KnowledgeRetrievalRequest(
        query="那按区域看呢",
        previous_user_question="Revenue for the last 30 days",
    )

    assert rewrite_retrieval_query(request) == "Revenue for the last 30 days；追问补充：那按区域看呢"


def test_standalone_query_is_not_rewritten():
    request = KnowledgeRetrievalRequest(
        query="Revenue by region for the last 7 days",
        previous_user_question="Revenue for the last 30 days",
    )

    assert rewrite_retrieval_query(request) == request.query


def test_metric_resolver_exposes_all_asset_linked_variants_without_question_based_selection():
    service = EsKnowledgeRetrievalService(get_settings(), LinkedVariantTopicAssets())

    total = service._resolve_metric_candidates("Revenue total", ["finance"])
    trend = service._resolve_metric_candidates("Revenue over time", ["finance"])

    assert {item["metricKey"] for item in total} == {
        "revenue_period",
        "revenue_daily",
        "revenue_snapshot",
    }
    assert [item["metricKey"] for item in total] == [item["metricKey"] for item in trend]
    linked = {item["metricKey"]: item for item in total if item["metricResolutionType"] == "linked_variant"}
    assert linked["revenue_daily"]["linkedVariantPath"] == "series.metricKey"
    assert linked["revenue_snapshot"]["linkedVariantPath"] == "alternatives[0]"
    assert linked["revenue_daily"]["linkedVariantOf"].endswith(":revenue_period")
    assert linked["revenue_daily"]["aggregationPolicy"] == "daily_value_only"
    assert linked["revenue_daily"]["selectionGuidance"] == "Use only with a daily grouping."
    assert all(0.0 <= item["metricResolutionConfidence"] <= 1.0 for item in total)

    cards = service._metric_candidate_items("Revenue total", total)
    daily_card = next(item for item in cards if item.metadata["metricKey"] == "revenue_daily")
    assert daily_card.metadata["linkedVariantPath"] == "series.metricKey"
    assert daily_card.metadata["aggregationPolicy"] == "daily_value_only"


def test_qualified_label_suppresses_embedded_bare_label_unless_both_are_requested():
    service = EsKnowledgeRetrievalService(get_settings(), LinkedVariantTopicAssets())
    service.topic_assets.load_table_metrics = lambda topic, table: [
        {
            "metricKey": "revenue",
            "businessName": "Revenue",
            "aliases": ["Revenue"],
            "formula": "SUM(amount)",
        },
        {
            "metricKey": "net_revenue",
            "businessName": "Net Revenue",
            "aliases": ["Net Revenue"],
            "formula": "SUM(net_amount)",
        },
    ]

    qualified = service._resolve_metric_candidates("Net Revenue", ["finance"])
    both = service._resolve_metric_candidates("Revenue and Net Revenue", ["finance"])

    assert [item["metricKey"] for item in qualified] == ["net_revenue"]
    assert {item["metricKey"] for item in both} == {"revenue", "net_revenue"}


def test_raw_metric_hits_remain_broad_after_resolver_candidates_are_merged():
    raw_items = [
        RecallItem(doc_id="metric:a", source_type="SEMANTIC_METRIC", fusion_score=0.7),
        RecallItem(doc_id="metric:b", source_type="SEMANTIC_METRIC", fusion_score=0.6),
    ]
    resolved_items = [
        RecallItem(
            doc_id="metric:a",
            source_type="SEMANTIC_METRIC",
            fusion_score=0.9,
            metadata={"metricResolutionType": "exact_alias"},
        )
    ]

    merged = merge_recall_items(raw_items, resolved_items)

    assert {item.doc_id for item in merged} == {"metric:a", "metric:b"}
    assert next(item for item in merged if item.doc_id == "metric:a").fusion_score == 0.9


def test_governance_filter_blocks_cross_merchant_role_status_version_and_expiry():
    request = KnowledgeRetrievalRequest(
        query="Revenue",
        merchant_id="merchant-100",
        access_role="merchant_analyst",
        permissions=["metric:read"],
    )
    expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    items = [
        RecallItem(doc_id="ok", metadata={"status": "PUBLISHED", "merchantId": "merchant-100"}),
        RecallItem(doc_id="merchant", metadata={"merchantId": "merchant-200"}),
        RecallItem(doc_id="role", metadata={"allowedRoles": ["merchant_finance"]}),
        RecallItem(doc_id="status", metadata={"status": "PENDING_REVIEW"}),
        RecallItem(doc_id="version", metadata={"version": "v1", "activeVersion": "v2"}),
        RecallItem(doc_id="expired", metadata={"expiresAt": expired}),
        RecallItem(doc_id="permission", metadata={"requiredPermissions": ["schema:restricted"]}),
    ]

    kept, filtered = filter_recall_items_by_governance(items, request)

    assert [item.doc_id for item in kept] == ["ok"]
    assert filtered == {
        "merchant": 1,
        "role": 1,
        "status": 1,
        "version": 1,
        "expired": 1,
        "permission": 1,
    }


def test_business_rerank_prefers_metric_matching_the_question():
    request = KnowledgeRetrievalRequest(query="Gross Revenue", intent_kind="metric_query")
    items = [
        RecallItem(
            doc_id="gross_revenue",
            source_type="SEMANTIC_METRIC",
            fusion_score=10,
            metadata={"businessName": "Gross Revenue", "metricKey": "gross_revenue"},
        ),
        RecallItem(
            doc_id="net_revenue",
            source_type="SEMANTIC_METRIC",
            fusion_score=11,
            metadata={"businessName": "Net Revenue", "metricKey": "net_revenue"},
        ),
    ]

    reranked = business_rerank_recall_items(items, request.query, request)

    assert reranked[0].doc_id == "gross_revenue"
    assert "exact_business_label" in reranked[0].metadata["businessRerankReasons"]
    assert 0.0 <= reranked[0].fusion_score <= 1.0
    assert reranked[0].metadata["scoreVersion"] == "recall_v2"


def test_rrf_scores_are_normalized_for_single_and_multiple_channels():
    text_items = [RecallItem(doc_id="a", fusion_score=10), RecallItem(doc_id="b", fusion_score=8)]
    vector_items = [RecallItem(doc_id="b", fusion_score=0.9)]

    text_only = rrf_fuse_recall_items([("bm25", text_items)], rrf_k=60)
    hybrid = rrf_fuse_recall_items([("bm25", text_items), ("vector", vector_items)], rrf_k=60)

    assert text_only[0].fusion_score == 1.0
    assert hybrid[0].doc_id == "b"
    assert 0.0 <= hybrid[0].fusion_score <= 1.0
    assert hybrid[0].metadata["rrfActiveLaneCount"] == 2
    assert hybrid[0].metadata["rrfDisplayScore"] > 0


def test_metric_resolver_score_is_bounded_and_penalizes_lower_rank():
    first = metric_candidate_fusion_score(0.96, "exact_business_name", 1)
    third = metric_candidate_fusion_score(0.96, "exact_business_name", 3)

    assert 0.0 <= third < first <= 1.0


def test_shared_alias_resolves_to_published_canonical_family_owner():
    owner = {
        "topic": "profile",
        "tableName": "merchant_profile",
        "metricKey": "order_amount",
        "canonicalMetricKey": "order_amount",
        "aliasOf": "",
    }
    variant = {
        "topic": "profile",
        "tableName": "merchant_profile",
        "metricKey": "paid_amount",
        "canonicalMetricKey": "order_amount",
        "aliasOf": "order_amount",
    }

    resolved, aliases = canonical_metric_family_owner([owner, variant])

    assert resolved is owner
    assert aliases == [variant]


def test_shared_alias_stays_ambiguous_across_independent_canonical_families():
    first = {
        "topic": "profile",
        "tableName": "merchant_profile",
        "metricKey": "order_amount",
        "canonicalMetricKey": "order_amount",
    }
    second = {
        "topic": "profile",
        "tableName": "merchant_profile",
        "metricKey": "net_amount",
        "canonicalMetricKey": "net_amount",
    }

    resolved, aliases = canonical_metric_family_owner([first, second])

    assert resolved is None
    assert aliases == []


def test_es_search_uses_active_profile_hybrid_top_k(monkeypatch):
    service = EsKnowledgeRetrievalService(get_settings().model_copy(update={"es_hybrid_top_k": 24}), object())
    service._active_retrieval_profile = {"hybridTopK": 32}
    monkeypatch.setattr(service, "_vector_enabled", lambda: False)
    monkeypatch.setattr(
        service,
        "_text_search",
        lambda query_text, topics, include_rules=False: [RecallItem(doc_id=f"doc-{index}", fusion_score=100 - index) for index in range(30)],
    )

    items = service._search("复杂分析问题", ["电商交易"])

    assert len(items) == 30


def test_exact_unambiguous_metric_uses_protection_tier_instead_of_magic_score():
    request = KnowledgeRetrievalRequest(query="Gross Revenue", intent_kind="metric_query")
    items = [
        RecallItem(
            doc_id="exact_metric",
            source_type="SEMANTIC_METRIC",
            fusion_score=0.55,
            metadata={
                "businessName": "Gross Revenue",
                "metricResolutionType": "exact_business_name",
                "metricResolutionConfidence": 0.97,
                "metricResolutionAmbiguous": False,
                "metricResolverScore": 0.55,
            },
        ),
        RecallItem(doc_id="strong_es", source_type="SEMANTIC_TABLE_ASSET", fusion_score=1.0, metadata={"retrievalScore": 1.0}),
    ]

    reranked = business_rerank_recall_items(items, request.query, request)
    limited = limit_recall_items_by_source_type(reranked, {}, limit=10)

    assert limited[0].doc_id == "exact_metric"
    assert limited[0].metadata["protectionTier"] == 2
    assert limited[0].fusion_score <= 1.0


def test_recall_bundle_strong_match_uses_versioned_normalized_threshold():
    normalized = RecallBundle(
        items=[RecallItem(doc_id="v2", fusion_score=0.72, metadata={"scoreVersion": "recall_v2", "finalScore": 0.72})],
        top_score=0.72,
    )
    weak = RecallBundle(
        items=[RecallItem(doc_id="weak", fusion_score=0.31, metadata={"scoreVersion": "recall_v2", "finalScore": 0.31})],
        top_score=0.31,
    )
    legacy = RecallBundle(items=[RecallItem(doc_id="legacy", fusion_score=4.2)], top_score=4.2)

    assert normalized.has_strong_match() is True
    assert weak.has_strong_match() is False
    assert legacy.has_strong_match() is True


def test_es_hit_is_rechecked_against_current_asset_governance():
    class TopicAssets:
        def load_table_asset(self, topic, table):
            assert (topic, table) == ("电商交易", "orders")
            return {
                "status": "PUBLISHED",
                "version": "v2",
                "allowedRoles": ["merchant_finance"],
            }

    service = EsKnowledgeRetrievalService(get_settings(), TopicAssets())
    old_hit = RecallItem(
        doc_id="orders_metric",
        topic="电商交易",
        table="orders",
        metadata={"status": "PUBLISHED", "version": "v1"},
    )

    refreshed = service._attach_current_asset_governance([old_hit])
    request = KnowledgeRetrievalRequest(access_role="merchant_analyst")
    kept, filtered = filter_recall_items_by_governance(refreshed, request)

    assert kept == []
    assert filtered == {"version": 1}


def test_agent_context_policies_are_stage_specific():
    manager = ContextManager(get_settings())

    planner = manager.context_policy("PlannerAgent")
    node = manager.context_policy("NodeAgent")
    answer = manager.context_policy("AnswerAgent")

    assert "semanticMetrics" in planner["includePriority"]
    assert "partitionColumns" in node["includePriority"]
    assert "verifiedEvidence" in answer["includePriority"]
    assert planner != node != answer
