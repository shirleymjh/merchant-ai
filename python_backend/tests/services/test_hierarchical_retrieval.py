from merchant_ai.config import get_settings
from merchant_ai.models import KnowledgeRetrievalRequest, QuestionCategory, RecallItem
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    build_retrieval_query_plan,
)


class RetrievalTopicAssets:
    def topic_names_for_categories(self, categories):
        del categories
        return ["交易主题"]

    def load_topic_contract(self, topic):
        assert topic == "交易主题"
        return {"metadata": {"knowledgeCapabilities": ["metric", "relationship"]}}

    def all_topic_names(self):
        return ["交易主题"]

    def load_manifest(self, topic):
        del topic
        return []

    def load_table_asset(self, topic, table):
        del topic, table
        return {"status": "PUBLISHED", "version": "v1"}

    def load_table_metrics(self, topic, table):
        del topic, table
        return []

    def load_table_terms(self, topic, table):
        del topic, table
        return []


def retrieval_settings(**updates):
    return get_settings().model_copy(
        update={
            "cache_enabled": False,
            "embedding_api_key": "",
            "llm_api_key": "",
            "es_vector_enabled": False,
            "es_multi_query_enabled": True,
            "es_multi_query_max_queries": 5,
            "es_hierarchical_retrieval_enabled": True,
            "es_hierarchical_max_directories": 1,
            "es_hierarchical_max_leaf_items": 2,
            **updates,
        }
    )


def published_item(
    ref_id,
    source_type,
    *,
    table="orders",
    score=1.0,
    semantic_kind="",
):
    return RecallItem(
        doc_id=ref_id,
        title=ref_id,
        content=ref_id,
        source_type=source_type,
        topic="交易主题",
        table=table,
        fusion_score=score,
        metadata={
            "semanticRefId": ref_id,
            "semanticKind": semantic_kind,
            "semanticPath": "topics/交易主题/tables/%s/detail.json" % table,
            "status": "PUBLISHED",
            "version": "v1",
        },
    )


def test_query_plan_keeps_base_and_builds_bounded_typed_queries():
    request = KnowledgeRetrievalRequest(
        query="最近30天退款率下降的原因，按商品类目分析",
        route_slots={"dimensions": [{"businessName": "商品类目"}]},
        intent_kind="analysis",
        complexity="complex",
    )

    plan = build_retrieval_query_plan(
        query_text=request.query,
        request=request,
        retrieval_profile={"queryType": "multi_hop_analysis"},
        metric_candidates=[{"businessName": "退款率", "metricKey": "refund_rate"}],
        include_rules=True,
        max_queries=5,
        enabled=True,
    )

    assert plan[0]["id"] == "base"
    assert plan[0]["query"] == request.query
    assert len(plan) == 5
    assert {item["id"] for item in plan} == {
        "base",
        "metrics",
        "relationships",
        "fields",
        "rules",
    }
    assert len({item["query"] for item in plan}) == len(plan)
    assert next(item for item in plan if item["id"] == "fields")["targetSourceTypes"] == [
        "SEMANTIC_COLUMN",
        "SEMANTIC_TERM",
    ]


def test_multi_query_records_each_result_and_strict_scope_skips_broad_search(monkeypatch):
    service = EsKnowledgeRetrievalService(
        retrieval_settings(es_hierarchical_retrieval_enabled=False),
        RetrievalTopicAssets(),
    )
    calls = []

    def fake_search(query_text, topics, include_rules=False):
        calls.append((query_text, list(topics), include_rules))
        if "关联关系" in query_text:
            return [
                published_item(
                    "semantic:交易主题:relationship:orders_refunds",
                    "SEMANTIC_RELATIONSHIP",
                    semantic_kind="RELATIONSHIP",
                )
            ]
        if "字段定义" in query_text:
            return [published_item("semantic:交易主题:orders:field:category", "SEMANTIC_COLUMN")]
        return [published_item("semantic:交易主题:orders:detail", "SEMANTIC_TABLE_ASSET")]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="分析订单和退款并按商品类目查看",
            topic_categories=[QuestionCategory.TRADE],
            intent_kind="analysis",
            complexity="complex",
            strict_topic_scope=True,
        )
    )

    trace = bundle.recall_rounds[0]
    assert len(trace.retrieval_query_plan) >= 3
    assert all(
        item.get("status") in {"SUCCESS", "EMPTY", "SKIPPED"}
        for item in trace.retrieval_query_plan
    )
    assert all(topics == ["交易主题"] for _query, topics, _rules in calls)
    assert any(item["lane"] == "multi_query_lane" and item["enabled"] for item in trace.retrieval_lanes)
    assert trace.retrieval_stop_reason == "FINAL_EVIDENCE_SELECTED"
    assert trace.hierarchical_retrieval_applied is False


def test_hierarchical_retrieval_selects_one_directory_and_only_keeps_exact_leaves(monkeypatch):
    service = EsKnowledgeRetrievalService(retrieval_settings(), RetrievalTopicAssets())
    calls = []

    def fake_search(query_text, topics, include_rules=False):
        calls.append((query_text, list(topics), dict(service._active_directory_scope or {})))
        if "限定目录" in query_text:
            return [
                published_item("semantic:交易主题:orders:field:category", "SEMANTIC_COLUMN", table="orders"),
                published_item("semantic:交易主题:orders:term:category", "SEMANTIC_TERM", table="orders"),
                published_item("semantic:交易主题:refunds:field:reason", "SEMANTIC_COLUMN", table="refunds"),
            ]
        return [
            published_item("semantic:交易主题:orders:detail", "SEMANTIC_TABLE_ASSET", table="orders", score=1.0),
            published_item("semantic:交易主题:refunds:detail", "SEMANTIC_TABLE_ASSET", table="refunds", score=0.7),
        ]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="分析订单退款并按商品类目查看",
            topic_categories=[QuestionCategory.TRADE],
            intent_kind="analysis",
            complexity="complex",
            strict_topic_scope=True,
        )
    )

    refs = {item.doc_id for item in bundle.recall_bundle.items}
    trace = bundle.recall_rounds[0]
    selection = next(item for item in trace.directory_retrieval_trace if item["stage"] == "DIRECTORY_SELECTION")
    expansion = next(item for item in trace.directory_retrieval_trace if item["stage"] == "DIRECTORY_EXPANSION")

    assert selection["selectedDirectories"][0]["table"] == "orders"
    assert selection["eliminatedByDirectoryCap"][0]["table"] == "refunds"
    assert "semantic:交易主题:orders:field:category" in refs
    assert "semantic:交易主题:refunds:field:reason" not in refs
    assert expansion["discardedOutsideDirectory"] == 1
    assert trace.hierarchical_retrieval_applied is True
    assert trace.retrieval_stop_details["hierarchicalStopReason"] == "MAX_LEAF_BUDGET_REACHED"
    assert any(scope.get("tables") == ["orders"] for _query, _topics, scope in calls if scope)


def test_multi_query_only_executes_queries_for_missing_asset_types(monkeypatch):
    service = EsKnowledgeRetrievalService(
        retrieval_settings(es_hierarchical_retrieval_enabled=False),
        RetrievalTopicAssets(),
    )
    calls = []

    def fake_search(query_text, topics, include_rules=False):
        del topics, include_rules
        calls.append(query_text)
        if len(calls) == 1:
            return [
                published_item(
                    "semantic:交易主题:orders:metric:refund_rate",
                    "SEMANTIC_METRIC",
                ),
                published_item(
                    "semantic:交易主题:relationship:orders_refunds",
                    "SEMANTIC_RELATIONSHIP",
                    semantic_kind="RELATIONSHIP",
                ),
            ]
        return [published_item("semantic:交易主题:orders:field:category", "SEMANTIC_COLUMN")]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="分析退款率并按商品类目查看",
            topic_categories=[QuestionCategory.TRADE],
            intent_kind="analysis",
            complexity="complex",
            strict_topic_scope=True,
        )
    )

    plan = {item["id"]: item for item in bundle.recall_rounds[0].retrieval_query_plan}
    assert plan["metrics"]["status"] == "SKIPPED"
    assert plan["relationships"]["status"] == "SKIPPED"
    assert plan["fields"]["status"] == "SUCCESS"
    assert len(calls) == 2


def test_supplemental_query_failure_keeps_base_evidence_and_records_degraded_trace(monkeypatch):
    service = EsKnowledgeRetrievalService(
        retrieval_settings(es_hierarchical_retrieval_enabled=False),
        RetrievalTopicAssets(),
    )

    def fake_search(query_text, topics, include_rules=False):
        del topics, include_rules
        if "检索重点" in query_text:
            raise RuntimeError("supplemental unavailable")
        return [published_item("semantic:交易主题:orders:detail", "SEMANTIC_TABLE_ASSET")]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="分析订单和退款",
            topic_categories=[QuestionCategory.TRADE],
            intent_kind="analysis",
            complexity="complex",
            strict_topic_scope=True,
        )
    )

    trace = bundle.recall_rounds[0]
    assert bundle.retrieval_status == "degraded"
    assert [item.doc_id for item in bundle.recall_bundle.items] == [
        "semantic:交易主题:orders:detail"
    ]
    assert any(item.get("status") == "FAILED" for item in trace.retrieval_query_plan[1:])
    assert trace.retrieval_stop_reason == "FINAL_EVIDENCE_SELECTED_DEGRADED"
    assert any(issue.code == "ES_MULTI_QUERY_FAILED" for issue in bundle.retrieval_issues)
