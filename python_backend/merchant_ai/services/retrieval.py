from __future__ import annotations

import hashlib
import json
from typing import Protocol

import requests

from merchant_ai.config import Settings
from merchant_ai.models import (
    ExtractedKeywords,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    QuestionCategory,
    RecallBundle,
    RecallItem,
    RecallRoundTrace,
    category_display,
)
from merchant_ai.services.assets import HybridRecallService, TopicAssetService, compact_metric_for_recall
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key


class KnowledgeRetrievalService(Protocol):
    """Unified knowledge retrieval boundary used by the agent harness."""

    backend_name: str

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        ...


class HybridKnowledgeRetrievalService:
    """Adapter that exposes the local hybrid recall backend through the unified API."""

    backend_name = "hybrid"

    def __init__(self, recall_service: HybridRecallService):
        self.recall_service = recall_service

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        recall_bundle = self.recall_service.recall(
            request.query,
            ExtractedKeywords(keywords=request.keywords),
            request.history_rows,
            request.knowledge_context,
            request.merchant_id,
            request.topic_categories,
        )
        if request.topic_categories and not request.knowledge_request:
            broad_bundle = self.recall_service.recall(
                request.query,
                ExtractedKeywords(keywords=request.keywords),
                request.history_rows,
                request.knowledge_context,
                request.merchant_id,
                [],
            )
            recall_bundle = merge_recall_bundles(recall_bundle, broad_bundle)
        source_refs = unique_source_refs(recall_bundle.items)
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        trace = RecallRoundTrace(
            request_key=str(request_key or ""),
            query=request.query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
            backend=self.backend_name,
            recall_queries=recall_queries_from_items(recall_bundle.items),
            source_refs=source_refs,
            item_count=len(recall_bundle.items),
        )
        return KnowledgeBundle(
            recall_bundle=recall_bundle,
            source_refs=source_refs,
            recall_rounds=[trace],
            backend=self.backend_name,
            index_version=self._index_version(),
            semantic_source_hash=semantic_hash_for_items(recall_bundle.items),
        )

    def _index_version(self) -> str:
        manifest_path = self.recall_service.settings.resolved_workspace_path / "recall_index_manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("indexVersion") or "")

    def cache_trace(self) -> dict[str, object]:
        return self.recall_service.cache_trace() if hasattr(self.recall_service, "cache_trace") else {}


class EsKnowledgeRetrievalService:
    """Elasticsearch-backed knowledge retrieval adapter.

    The rest of the harness still consumes KnowledgeBundle/RecallItem, so ES is
    a backend choice, not a second recall path.
    """

    backend_name = "es"

    def __init__(self, settings: Settings, topic_assets: TopicAssetService):
        self.settings = settings
        self.topic_assets = topic_assets
        self._recall_cache = build_ttl_cache("es_recall", settings, settings.cache_recall_ttl_seconds)
        self._embedding_cache = build_ttl_cache("es_embedding", settings, settings.cache_recall_ttl_seconds)

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        query_text = retrieval_query_text(request)
        normalized_categories = [category for category in [normalize_question_category(item) for item in request.topic_categories] if category]
        topics = self._allowed_topics(normalized_categories)
        include_base_wiki = QuestionCategory.PLATFORM_RULE in set(normalized_categories) or route_is_rule_sensitive(request)
        cache_key = stable_cache_key(
            "es_recall",
            {
                "query": query_text,
                "merchantId": request.merchant_id,
                "topics": topics,
                "includeBaseWiki": include_base_wiki,
                "requestKey": str(request_key or ""),
                "indexVersion": self._index_version(),
                "vectorEnabled": self._vector_enabled(),
                "embeddingModel": self.settings.embedding_model if self._vector_enabled() else "",
                "rrfK": self.settings.es_rrf_k,
                "hybridTopK": self.settings.es_hybrid_top_k,
            },
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return KnowledgeBundle.model_validate(cached)
        try:
            items = self._search(query_text, topics, include_base_wiki=include_base_wiki)
            if topics and not request.knowledge_request:
                try:
                    items = merge_recall_items(items, self._search(query_text, [], include_base_wiki=False))
                except Exception:
                    pass
            items = merge_recall_items(items, self._exact_metric_evidence(query_text, topics))
            blocked_reason = ""
        except Exception as exc:
            items = []
            blocked_reason = "ES_RETRIEVAL_FAILED:%s" % str(exc)[:240]
        source_refs = unique_source_refs(items)
        trace = RecallRoundTrace(
            request_key=str(request_key or ""),
            query=request.query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
            backend=self.backend_name,
            recall_queries=[query_text] if query_text else [],
            source_refs=source_refs,
            item_count=len(items),
            blocked_reason=blocked_reason,
        )
        merged = "\n\n".join("召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items)
        bundle = KnowledgeBundle(
            recall_bundle=RecallBundle(
                items=items,
                top_score=items[0].fusion_score if items else 0.0,
                merged_context=merged,
            ),
            source_refs=source_refs,
            recall_rounds=[trace],
            backend=self.backend_name,
            index_version=self._index_version(),
            semantic_source_hash=semantic_hash_for_items(items),
        )
        if not blocked_reason:
            self._recall_cache.set(cache_key, bundle.model_dump(by_alias=True))
        return bundle

    def _allowed_topics(self, topic_categories: list[QuestionCategory]) -> list[str]:
        topic_names = self.topic_assets.topic_names_for_categories(topic_categories)
        if topic_names:
            return topic_names
        names: list[str] = []
        for category in topic_categories:
            display = category_display(category)
            if display and display not in names:
                names.append(display)
        return names

    def _search(self, query_text: str, topics: list[str], include_base_wiki: bool = False) -> list[RecallItem]:
        text_items = self._text_search(query_text, topics, include_base_wiki=include_base_wiki)
        if not self._vector_enabled() or not query_text:
            return text_items
        try:
            vector = self._embed_text(query_text)
            vector_items = self._vector_search(query_text, vector, topics, include_base_wiki=include_base_wiki) if vector else []
        except Exception:
            vector_items = []
        if not vector_items:
            return text_items
        return rrf_fuse_recall_items(
            [("bm25", text_items), ("vector", vector_items)],
            rrf_k=self.settings.es_rrf_k,
            score_scale=self.settings.es_rrf_score_scale,
            limit=self.settings.es_hybrid_top_k,
        )

    def _text_search(self, query_text: str, topics: list[str], include_base_wiki: bool = False) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        size = self._text_size(topics)
        query = self._text_query(query_text, topics, include_base_wiki=include_base_wiki)
        response = requests.post(
            "%s/%s/_search" % (self.settings.es_base_url.rstrip("/"), self.settings.es_index),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "query": query},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        return [es_hit_to_recall_item(hit, query_text, channel="bm25") for hit in hits]

    def _vector_search(self, query_text: str, query_vector: list[float], topics: list[str], include_base_wiki: bool = False) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        if not query_vector:
            return []
        size = self._vector_size(topics)
        filters = self._filters(topics, include_base_wiki=include_base_wiki)
        knn: dict[str, object] = {
            "field": self.settings.es_vector_field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(size, int(self.settings.es_vector_num_candidates or 0)),
        }
        if filters:
            knn["filter"] = filters if len(filters) > 1 else filters[0]
        response = requests.post(
            "%s/%s/_search" % (self.settings.es_base_url.rstrip("/"), self.settings.es_index),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "knn": knn},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        return [es_hit_to_recall_item(hit, query_text, channel="vector") for hit in hits]

    def _text_query(self, query_text: str, topics: list[str], include_base_wiki: bool = False) -> dict[str, object]:
        must: list[dict[str, object]] = []
        if query_text:
            must.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "title^3",
                            "content^2",
                            "metadata.businessName^3",
                            "metadata.aliases^2",
                            "metadata.metricKey^3",
                            "metadata.relationshipId^2",
                            "metadata.tableName^2",
                        ],
                    }
                }
            )
        filters = self._filters(topics, include_base_wiki=include_base_wiki)
        if must or filters:
            return {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}
        return {"match_all": {}}

    def _filters(self, topics: list[str], include_base_wiki: bool = False) -> list[dict[str, object]]:
        filters: list[dict[str, object]] = []
        if topics:
            topic_should: list[dict[str, object]] = [
                {"terms": {"topic": topics}},
                {"terms": {"topic.keyword": topics}},
                {"terms": {"metadata.topic": topics}},
                {"terms": {"metadata.topic.keyword": topics}},
            ]
            if include_base_wiki:
                topic_should.append({"term": {"source_type": "BASE_WIKI"}})
            filters.append({"bool": {"should": topic_should, "minimum_should_match": 1}})
        elif include_base_wiki:
            filters.append({"term": {"source_type": "BASE_WIKI"}})
        return filters

    def _text_size(self, topics: list[str]) -> int:
        return max(1, int(self.settings.es_text_top_k if topics else self.settings.es_broad_text_top_k))

    def _vector_size(self, topics: list[str]) -> int:
        return max(1, int(self.settings.es_vector_top_k if topics else self.settings.es_broad_vector_top_k))

    def _vector_enabled(self) -> bool:
        return bool(self.settings.es_vector_enabled and self.settings.es_vector_field and self.settings.embedding_model and self._embedding_api_key())

    def _embedding_api_key(self) -> str:
        return str(self.settings.embedding_api_key or self.settings.llm_api_key or "").strip()

    def _embed_text(self, text: str) -> list[float]:
        value = str(text or "").strip()
        if not value:
            return []
        cache_key = stable_cache_key(
            "embedding",
            {
                "baseUrl": self.settings.embedding_base_url,
                "model": self.settings.embedding_model,
                "dims": self.settings.embedding_dims,
                "text": value,
            },
        )
        cached = self._embedding_cache.get(cache_key)
        if isinstance(cached, list):
            return [float(item) for item in cached]
        payload: dict[str, object] = {"model": self.settings.embedding_model, "input": value}
        if int(self.settings.embedding_dims or 0) > 0:
            payload["dimensions"] = int(self.settings.embedding_dims)
        response = requests.post(
            "%s/embeddings" % self.settings.embedding_base_url.rstrip("/"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self._embedding_api_key(),
            },
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json() or {}
        vector = (((data.get("data") or [{}])[0] or {}).get("embedding") or [])
        result = [float(item) for item in vector if isinstance(item, (int, float))]
        if result:
            self._embedding_cache.set(cache_key, result)
        return result

    def cache_trace(self) -> dict[str, object]:
        trace = {"esRecall": self._recall_cache.trace(), "esEmbedding": self._embedding_cache.trace()}
        trace["hybridRecall"] = {
            "vectorEnabled": self._vector_enabled(),
            "vectorField": self.settings.es_vector_field,
            "textTopK": self.settings.es_text_top_k,
            "vectorTopK": self.settings.es_vector_top_k,
            "broadTextTopK": self.settings.es_broad_text_top_k,
            "broadVectorTopK": self.settings.es_broad_vector_top_k,
            "rrfK": self.settings.es_rrf_k,
            "hybridTopK": self.settings.es_hybrid_top_k,
        }
        return trace

    def _exact_metric_evidence(self, query_text: str, topics: list[str]) -> list[RecallItem]:
        """Protect exact metric evidence from ES topK truncation.

        ES remains the retrieval backend; this only supplements semantic metric
        refs whose governed businessName/alias appears verbatim in the query.
        It does not choose the final metric for the planner.
        """
        query = (query_text or "").strip()
        if not query:
            return []
        topic_names = topics or self.topic_assets.all_topic_names()
        items: list[RecallItem] = []
        seen: set[str] = set()
        for topic in topic_names:
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for metric in self.topic_assets.load_table_metrics(topic, table):
                    if not isinstance(metric, dict):
                        continue
                    metric_key = str(metric.get("metricKey") or "")
                    if not metric_key:
                        continue
                    matched_label = exact_metric_label_in_query(metric, query)
                    if not matched_label:
                        continue
                    semantic_ref_id = "semantic:%s:%s:metric:%s" % (topic, table, metric_key)
                    if semantic_ref_id in seen:
                        continue
                    seen.add(semantic_ref_id)
                    items.append(
                        RecallItem(
                            doc_id=semantic_ref_id,
                            title="%s/%s/%s metric" % (topic, table, metric_key),
                            content=compact_metric_for_recall(topic, table, metric),
                            source_type="SEMANTIC_METRIC",
                            topic=topic,
                            table=table,
                            fusion_score=0.01,
                            metadata={
                                "semanticSource": "metrics",
                                "semanticKind": "METRIC",
                                "semanticRefId": semantic_ref_id,
                                "metricKey": metric_key,
                                "tableName": table,
                                "topic": topic,
                                "businessName": metric.get("businessName") or metric_key,
                                "canonicalMetricKey": metric.get("canonicalMetricKey") or "",
                                "aliasOf": metric.get("aliasOf") or "",
                                "metricLevel": metric.get("metricLevel") or "",
                                "formula": metric.get("formula") or metric.get("metricFormula") or "",
                                "sourceColumns": metric.get("sourceColumns") or [],
                                "aliases": metric.get("aliases") or [],
                                "recallQuery": query,
                                "recallQueries": [query],
                                "matchedExactMetricLabel": matched_label,
                                "recallSupplement": "exact_metric_evidence",
                            },
                        )
                    )
        return items

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.es_api_key:
            headers["Authorization"] = "Bearer %s" % self.settings.es_api_key
        return headers

    def _auth(self) -> tuple[str, str] | None:
        if self.settings.es_api_key:
            return None
        if self.settings.es_username:
            return (self.settings.es_username, self.settings.es_password)
        return None

    def _index_version(self) -> str:
        manifest_path = self.settings.resolved_workspace_path / "recall_index_manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("indexVersion") or "")


def merge_recall_bundles(primary: RecallBundle, secondary: RecallBundle) -> RecallBundle:
    items = merge_recall_items(primary.items, secondary.items)
    return RecallBundle(
        items=items,
        top_score=items[0].fusion_score if items else 0.0,
        merged_context="\n\n".join("召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200]) for item in items),
    )


def merge_recall_items(primary: list[RecallItem], secondary: list[RecallItem]) -> list[RecallItem]:
    by_id: dict[str, RecallItem] = {}
    for item in list(primary or []) + list(secondary or []):
        key = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
        if not key:
            continue
        current = by_id.get(key)
        if current is None or item.fusion_score >= current.fusion_score:
            by_id[key] = item
    return sorted(by_id.values(), key=lambda item: item.fusion_score, reverse=True)


def rrf_fuse_recall_items(
    ranked_groups: list[tuple[str, list[RecallItem]]],
    rrf_k: int = 60,
    score_scale: float = 1000.0,
    limit: int = 24,
) -> list[RecallItem]:
    """Fuse ranked recall lists with reciprocal rank fusion.

    BM25 scores and vector similarities are not comparable. RRF only uses the
    rank position inside each channel, so a document that appears near the top
    in both channels naturally wins without score normalization.
    """
    k = max(1, int(rrf_k or 60))
    scale = float(score_scale or 1.0)
    by_id: dict[str, RecallItem] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    channel_scores: dict[str, dict[str, float]] = {}
    for channel, items in ranked_groups:
        channel_name = str(channel or "unknown")
        seen_in_channel: set[str] = set()
        for rank, item in enumerate(items or [], start=1):
            key = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
            if not key or key in seen_in_channel:
                continue
            seen_in_channel.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[channel_name] = rank
            channel_scores.setdefault(key, {})[channel_name] = float(item.fusion_score or 0.0)
            if key not in by_id:
                by_id[key] = item
            else:
                by_id[key] = merge_recall_item_metadata(by_id[key], item)
    fused: list[RecallItem] = []
    for key, item in by_id.items():
        metadata = dict(item.metadata or {})
        metadata["recallFusion"] = "rrf"
        metadata["rrfScore"] = scores.get(key, 0.0)
        metadata["rrfK"] = k
        metadata["rrfRanks"] = ranks.get(key, {})
        metadata["channelScores"] = channel_scores.get(key, {})
        metadata["recallChannels"] = sorted((ranks.get(key) or {}).keys())
        fused.append(item.model_copy(update={"fusion_score": round(scores.get(key, 0.0) * scale, 6), "metadata": metadata}))
    fused = sorted(fused, key=lambda item: item.fusion_score, reverse=True)
    return fused[: max(1, int(limit or len(fused)))] if limit else fused


def merge_recall_item_metadata(primary: RecallItem, secondary: RecallItem) -> RecallItem:
    metadata = dict(primary.metadata or {})
    other = dict(secondary.metadata or {})
    for key, value in other.items():
        if key not in metadata or is_empty_metadata_value(metadata.get(key)):
            metadata[key] = value
    queries: list[str] = []
    for source in [metadata, other]:
        for raw in list(source.get("recallQueries") or []) + [source.get("recallQuery")]:
            query = str(raw or "").strip()
            if query and query not in queries:
                queries.append(query)
    if queries:
        metadata["recallQueries"] = queries
        metadata["recallQuery"] = queries[0]
    if secondary.content and len(secondary.content) > len(primary.content or ""):
        return secondary.model_copy(update={"metadata": metadata})
    return primary.model_copy(update={"metadata": metadata})


def is_empty_metadata_value(value: object) -> bool:
    return value is None or value == "" or value == []


def unique_source_refs(items: list[RecallItem]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        ref = item.doc_id or str((item.metadata or {}).get("semanticRefId") or "")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def retrieval_query_text(request: KnowledgeRetrievalRequest) -> str:
    parts = [request.query]
    parts.extend(request.keywords or [])
    knowledge_request = request.knowledge_request
    if knowledge_request:
        parts.extend(
            [
                knowledge_request.query,
                knowledge_request.source_phrase,
                knowledge_request.reason,
                " ".join(knowledge_request.expected_refs or []),
            ]
        )
    seen: set[str] = set()
    values: list[str] = []
    for part in parts:
        value = str(part or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return " ".join(values)


GENERIC_METRIC_LABELS = {
    "金额",
    "数量",
    "单量",
    "商品",
    "订单",
    "退款",
    "退货",
    "售后",
    "支付",
    "下单",
    "情况",
}


def exact_metric_label_in_query(metric: dict[str, object], query_text: str) -> str:
    query = normalize_recall_label(query_text)
    if not query:
        return ""
    labels = [
        str(metric.get("businessName") or ""),
        str(metric.get("metricKey") or ""),
        *[str(alias) for alias in metric.get("aliases") or []],
    ]
    for label in labels:
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized):
            continue
        if normalized and normalized in query:
            return label
    return ""


def normalize_recall_label(value: str) -> str:
    return "".join(str(value or "").lower().split())


def is_protective_metric_label(label: str) -> bool:
    if not label or label in GENERIC_METRIC_LABELS:
        return False
    if "_" in label:
        return len(label) >= 4
    return len(label) >= 3


def normalize_question_category(category: object) -> QuestionCategory | None:
    if isinstance(category, QuestionCategory):
        return category
    raw = str(category or "").strip()
    if not raw:
        return None
    try:
        return QuestionCategory(raw)
    except Exception:
        pass
    for item in QuestionCategory:
        if category_display(item) == raw:
            return item
    return None


def route_is_rule_sensitive(request: KnowledgeRetrievalRequest) -> bool:
    slots = request.route_slots or {}
    risk_level = str(slots.get("riskLevel") or slots.get("risk_level") or "").strip()
    return risk_level == "rule_sensitive"


def es_hit_to_recall_item(hit: dict[str, object], query_text: str, channel: str = "bm25") -> RecallItem:
    source = hit.get("_source") if isinstance(hit, dict) else {}
    source = source if isinstance(source, dict) else {}
    metadata = dict(source.get("metadata") or {})
    semantic_ref_id = str(source.get("semantic_ref_id") or metadata.get("semanticRefId") or source.get("doc_id") or hit.get("_id") or "")
    semantic_path = str(source.get("semantic_path") or metadata.get("semanticPath") or "")
    merchant_uri = str(source.get("merchant_uri") or metadata.get("merchantUri") or "")
    context_layer = str(source.get("context_layer") or metadata.get("contextLayer") or "")
    metadata["semanticRefId"] = semantic_ref_id
    if semantic_path:
        metadata["semanticPath"] = semantic_path
    if merchant_uri:
        metadata["merchantUri"] = merchant_uri
    if context_layer:
        metadata["contextLayer"] = context_layer
    metadata["recallQuery"] = query_text
    metadata["recallQueries"] = [query_text] if query_text else []
    metadata["esScore"] = float(hit.get("_score") or 0.0)
    metadata["recallChannel"] = channel
    if channel == "bm25":
        metadata["bm25Score"] = float(hit.get("_score") or 0.0)
    elif channel == "vector":
        metadata["vectorScore"] = float(hit.get("_score") or 0.0)
    return RecallItem(
        doc_id=str(source.get("doc_id") or semantic_ref_id or hit.get("_id") or ""),
        title=str(source.get("title") or ""),
        content=str(source.get("content") or ""),
        source_type=str(source.get("source_type") or ""),
        topic=str(source.get("topic") or metadata.get("topic") or ""),
        table=str(source.get("table") or metadata.get("tableName") or ""),
        answer_mode=str(source.get("answer_mode") or ""),
        fusion_score=float(hit.get("_score") or 0.0),
        metadata=metadata,
    )


def recall_queries_from_items(items: list[RecallItem]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for item in items:
        metadata = item.metadata or {}
        for raw in list(metadata.get("recallQueries") or []) + [metadata.get("recallQuery")]:
            query = str(raw or "").strip()
            if not query or query in seen:
                continue
            seen.add(query)
            queries.append(query)
    return queries


def semantic_hash_for_items(items: list[RecallItem]) -> str:
    records = [
        {
            "docId": item.doc_id,
            "sourceType": item.source_type,
            "semanticRefId": str((item.metadata or {}).get("semanticRefId") or ""),
            "merchantUri": str((item.metadata or {}).get("merchantUri") or ""),
            "sourcePath": str((item.metadata or {}).get("sourcePath") or ""),
        }
        for item in items
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] if records else ""
