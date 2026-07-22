from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.models import KnowledgeRetrievalRequest, RecallItem
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.retrieval import EsKnowledgeRetrievalService


def cache_settings(tmp_path):
    return get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_recall_ttl_seconds": 60,
            "cache_memory_max_entries": 16,
            "embedding_api_key": "",
            "llm_api_key": "",
            "es_vector_enabled": False,
            "es_multi_query_enabled": False,
            "es_hierarchical_retrieval_enabled": False,
        }
    )


def install_counting_search(monkeypatch, service):
    calls = {"count": 0}

    def fake_search(query_text, topics, include_rules=False):
        del query_text, topics, include_rules
        calls["count"] += 1
        return [
            RecallItem(
                doc_id="semantic:test:metric:order_amount",
                title="订单金额",
                content="订单金额指标",
                source_type="SEMANTIC_METRIC",
                fusion_score=10,
            )
        ]

    monkeypatch.setattr(service, "_search", fake_search)
    return calls


def test_initial_es_recall_is_cached_by_normalized_retrieval_question(monkeypatch, tmp_path):
    settings = cache_settings(tmp_path)
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    calls = install_counting_search(monkeypatch, service)

    first = service.retrieve(
        KnowledgeRetrievalRequest(query="最近7天 订单金额", merchant_id="merchant-1")
    )
    second = service.retrieve(
        KnowledgeRetrievalRequest(query="  最近7天   订单金额  ", merchant_id="merchant-1")
    )
    service.retrieve(
        KnowledgeRetrievalRequest(query="最近7天退款金额", merchant_id="merchant-1")
    )

    assert first.source_refs == second.source_refs
    assert calls["count"] == 2
    assert service.cache_trace()["esRecall"]["hits"] == 1


def test_initial_es_recall_cache_is_scoped_by_merchant_role_and_permissions(monkeypatch, tmp_path):
    settings = cache_settings(tmp_path)
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    calls = install_counting_search(monkeypatch, service)
    base = {
        "query": "最近7天订单金额",
        "access_role": "merchant",
        "permissions": ["metric:read", "table:read"],
    }

    service.retrieve(KnowledgeRetrievalRequest(**base, merchant_id="merchant-1"))
    service.retrieve(KnowledgeRetrievalRequest(**base, merchant_id="merchant-2"))
    service.retrieve(
        KnowledgeRetrievalRequest(
            **{**base, "access_role": "analyst"},
            merchant_id="merchant-1",
        )
    )
    service.retrieve(
        KnowledgeRetrievalRequest(
            **{**base, "permissions": ["metric:read"]},
            merchant_id="merchant-1",
        )
    )
    service.retrieve(
        KnowledgeRetrievalRequest(
            **{**base, "permissions": ["table:read", "metric:read"]},
            merchant_id="merchant-1",
        )
    )

    assert calls["count"] == 4
    assert service.cache_trace()["esRecall"]["hits"] == 1


def test_initial_es_recall_cache_is_invalidated_by_index_version(monkeypatch, tmp_path):
    settings = cache_settings(tmp_path)
    manifest_path = tmp_path / "recall_index_manifest.json"
    manifest_path.write_text(json.dumps({"indexVersion": "index-v1"}), encoding="utf-8")
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    calls = install_counting_search(monkeypatch, service)
    request = KnowledgeRetrievalRequest(query="最近7天订单金额", merchant_id="merchant-1")

    first = service.retrieve(request)
    cached = service.retrieve(request)
    manifest_path.write_text(json.dumps({"indexVersion": "index-v2"}), encoding="utf-8")
    refreshed = service.retrieve(request)

    assert first.index_version == "index-v1"
    assert cached.index_version == "index-v1"
    assert refreshed.index_version == "index-v2"
    assert calls["count"] == 2


def test_supplemental_es_recall_cache_is_bound_to_coverage_receipt(monkeypatch, tmp_path):
    settings = cache_settings(tmp_path)
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    calls = install_counting_search(monkeypatch, service)
    base = {
        "query": "补充订单金额字段证据",
        "merchant_id": "merchant-1",
        "target_goal_ids": ["goal-1"],
        "required_capabilities": ["FIELD"],
    }

    service.retrieve(KnowledgeRetrievalRequest(**base, coverage_receipt_id="receipt-1"))
    service.retrieve(KnowledgeRetrievalRequest(**base, coverage_receipt_id="receipt-2"))
    service.retrieve(KnowledgeRetrievalRequest(**base, coverage_receipt_id="receipt-1"))

    assert calls["count"] == 2
    assert service.cache_trace()["esRecall"]["hits"] == 1
