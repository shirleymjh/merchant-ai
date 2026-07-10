from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

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
        self._active_retrieval_profile: dict[str, Any] | None = None

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        query_text = retrieval_query_text(request)
        normalized_categories = [category for category in [normalize_question_category(item) for item in request.topic_categories] if category]
        topics = self._allowed_topics(normalized_categories)
        include_base_wiki = QuestionCategory.PLATFORM_RULE in set(normalized_categories) or route_is_rule_sensitive(request)
        metric_candidates = self._resolve_metric_candidates(query_text, topics)
        retrieval_profile = build_retrieval_profile(
            query_text=query_text,
            topics=topics,
            include_base_wiki=include_base_wiki,
            metric_candidates=metric_candidates,
            intent_kind=request.intent_kind,
            complexity=request.complexity,
            settings=self.settings,
        )
        source_type_top_k = source_type_top_k_policy(
            include_base_wiki=include_base_wiki,
            query_text=query_text,
            topics=topics,
            metric_candidates=metric_candidates,
            retrieval_profile=retrieval_profile,
        )
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
                "retrievalProfile": retrieval_profile,
                "sourceTypeTopK": source_type_top_k,
            },
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return KnowledgeBundle.model_validate(cached)
        self._active_retrieval_profile = retrieval_profile
        try:
            try:
                items = self._search(query_text, topics, include_base_wiki=include_base_wiki)
                if topics and not request.knowledge_request and bool(retrieval_profile.get("broadSearchEnabled", True)):
                    try:
                        items = merge_recall_items(items, self._search(query_text, [], include_base_wiki=False))
                    except Exception:
                        pass
                items = merge_recall_items(items, self._metric_candidate_items(query_text, metric_candidates))
                items = merge_recall_items(items, self._exact_metric_evidence(query_text, topics))
                items = limit_recall_items_by_source_type(
                    items,
                    source_type_top_k,
                    limit=max(1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or len(items) or 1)),
                )
                blocked_reason = ""
            except Exception as exc:
                items = []
                blocked_reason = "ES_RETRIEVAL_FAILED:%s" % str(exc)[:240]
        finally:
            self._active_retrieval_profile = None
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
            recall_channels=recall_channels_for_items(items),
            source_type_top_k=source_type_top_k,
            vector_enabled=self._vector_enabled(),
            vector_disabled=not self._vector_enabled(),
            metric_candidates=metric_trace_payload(metric_candidates),
            retrieval_profile=retrieval_profile,
            query_type=str(retrieval_profile.get("queryType") or ""),
            intent_kind=str(request.intent_kind or ""),
            complexity=str(request.complexity or ""),
            retrieval_lanes=retrieval_lane_trace(
                retrieval_profile=retrieval_profile,
                vector_enabled=self._vector_enabled(),
                include_base_wiki=include_base_wiki,
                has_metric_candidates=bool(metric_candidates),
                broad_enabled=bool(topics),
            ),
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
        profile = self._active_retrieval_profile or {}
        key = "textTopK" if topics else "broadTextTopK"
        fallback = self.settings.es_text_top_k if topics else self.settings.es_broad_text_top_k
        return max(1, int(profile.get(key) or fallback))

    def _vector_size(self, topics: list[str]) -> int:
        profile = self._active_retrieval_profile or {}
        key = "vectorTopK" if topics else "broadVectorTopK"
        fallback = self.settings.es_vector_top_k if topics else self.settings.es_broad_vector_top_k
        return max(1, int(profile.get(key) or fallback))

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
            "dynamicTopKEnabled": True,
        }
        return trace

    def _resolve_metric_candidates(self, query_text: str, topics: list[str]) -> list[dict[str, Any]]:
        query = (query_text or "").strip()
        if not query:
            return []
        topic_names = topics or self.topic_assets.all_topic_names()
        candidates: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for topic in topic_names:
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                metrics = [metric for metric in self.topic_assets.load_table_metrics(topic, table) if isinstance(metric, dict)]
                metrics_by_key = {
                    str(metric.get("metricKey") or ""): metric
                    for metric in metrics
                    if str(metric.get("metricKey") or "")
                }
                for metric in metrics:
                    candidate = resolve_metric_candidate(metric, topic, table, query)
                    if candidate is None:
                        continue
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(current.get("metricResolutionConfidence") or 0.0):
                        by_id[semantic_ref_id] = candidate
                for term in self.topic_assets.load_table_terms(topic, table):
                    candidate = resolve_term_metric_candidate(term, metrics_by_key, topic, table, query)
                    if candidate is None:
                        continue
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(current.get("metricResolutionConfidence") or 0.0):
                        by_id[semantic_ref_id] = candidate
        candidates = list(by_id.values())
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            label_key = normalize_recall_label(str(candidate.get("matchedMetricLabel") or ""))
            if label_key:
                label_groups.setdefault(label_key, []).append(candidate)
        for label_key, group in label_groups.items():
            unique_metrics = {
                (str(item.get("topic") or ""), str(item.get("tableName") or ""), str(item.get("metricKey") or ""))
                for item in group
            }
            if len(unique_metrics) <= 1:
                continue
            for item in group:
                item["metricResolutionAmbiguous"] = True
                item["metricResolutionConfidence"] = max(0.4, round(float(item.get("metricResolutionConfidence") or 0.0) - 0.18, 3))
                item["metricResolutionReason"] = "%s; ambiguous_label=%s" % (str(item.get("metricResolutionReason") or ""), label_key)
        candidates.sort(
            key=lambda item: (
                float(item.get("metricResolutionConfidence") or 0.0),
                int(item.get("matchLength") or 0),
                float(item.get("fusionScore") or 0.0),
            ),
            reverse=True,
        )
        return candidates[:6]

    def _metric_candidate_items(self, query_text: str, candidates: list[dict[str, Any]]) -> list[RecallItem]:
        query = (query_text or "").strip()
        items: list[RecallItem] = []
        for rank, candidate in enumerate(candidates or [], start=1):
            semantic_ref_id = str(candidate.get("semanticRefId") or "")
            if not semantic_ref_id:
                continue
            metric = candidate.get("metric") or {}
            topic = str(candidate.get("topic") or "")
            table = str(candidate.get("tableName") or "")
            metric_key = str(candidate.get("metricKey") or "")
            confidence = float(candidate.get("metricResolutionConfidence") or 0.0)
            resolution_type = str(candidate.get("metricResolutionType") or "")
            score = float(candidate.get("fusionScore") or metric_candidate_fusion_score(confidence, resolution_type, rank))
            metadata = {
                "semanticSource": "metrics",
                "semanticKind": "METRIC",
                "semanticRefId": semantic_ref_id,
                "metricKey": metric_key,
                "tableName": table,
                "topic": topic,
                "businessName": candidate.get("businessName") or metric_key,
                "canonicalMetricKey": candidate.get("canonicalMetricKey") or "",
                "aliasOf": candidate.get("aliasOf") or "",
                "metricLevel": candidate.get("metricLevel") or "",
                "formula": candidate.get("formula") or "",
                "sourceColumns": candidate.get("sourceColumns") or [],
                "aliases": candidate.get("aliases") or [],
                "recallQuery": query,
                "recallQueries": [query] if query else [],
                "recallChannel": "metric_resolver",
                "matchedMetricLabel": candidate.get("matchedMetricLabel") or "",
                "metricResolutionType": resolution_type,
                "metricResolutionReason": candidate.get("metricResolutionReason") or "",
                "metricResolutionConfidence": confidence,
                "metricResolutionAmbiguous": bool(candidate.get("metricResolutionAmbiguous") or False),
                "metricCandidateRank": rank,
                "recallSupplement": "metric_candidate_resolution",
            }
            items.append(
                RecallItem(
                    doc_id=semantic_ref_id,
                    title="%s/%s/%s metric" % (topic, table, metric_key),
                    content=compact_metric_for_recall(topic, table, metric if isinstance(metric, dict) else {}),
                    source_type="SEMANTIC_METRIC",
                    topic=topic,
                    table=table,
                    fusion_score=score,
                    metadata=metadata,
                )
            )
        return items

    def _exact_metric_evidence(self, query_text: str, topics: list[str]) -> list[RecallItem]:
        """Compatibility supplement for very high-confidence exact metric matches.

        The primary path is now metric candidate resolution before ranking. This
        adapter keeps an explicit exact-match lane so existing callers and
        diagnostics still have a stable high-confidence fallback.
        """
        resolved = self._resolve_metric_candidates(query_text, topics)
        exact_candidates = [
            candidate
            for candidate in resolved
            if str(candidate.get("metricResolutionType") or "").startswith("exact")
            and float(candidate.get("metricResolutionConfidence") or 0.0) >= 0.9
        ]
        items = self._metric_candidate_items(query_text, exact_candidates)
        for item in items:
            metadata = dict(item.metadata or {})
            metadata["recallChannel"] = "exact"
            metadata["matchedExactMetricLabel"] = metadata.get("matchedMetricLabel") or ""
            metadata["recallSupplement"] = "exact_metric_evidence"
            item.metadata = metadata
            item.fusion_score = round(float(item.fusion_score or 0.0) - 50.0, 6)
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


def resolve_metric_candidate(metric: dict[str, Any], topic: str, table: str, query_text: str) -> dict[str, Any] | None:
    metric_key = str(metric.get("metricKey") or "").strip()
    if not metric_key:
        return None
    query = normalize_recall_label(query_text)
    if not query:
        return None
    labels = [
        ("exact_business_name", str(metric.get("businessName") or ""), 0.99, "businessName"),
        ("exact_metric_key", metric_key, 0.95, "metricKey"),
    ]
    labels.extend(("exact_alias", str(alias), 0.97, "alias") for alias in metric.get("aliases") or [])
    best: dict[str, Any] | None = None
    for resolution_type, raw_label, confidence, source in labels:
        label = str(raw_label or "").strip()
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized) or normalized not in query:
            continue
        candidate = build_metric_candidate(metric, topic, table, label, resolution_type, confidence, source)
        if best is None or compare_metric_candidate(candidate, best) > 0:
            best = candidate
    return best


def resolve_term_metric_candidate(term: dict[str, Any], metrics_by_key: dict[str, dict[str, Any]], topic: str, table: str, query_text: str) -> dict[str, Any] | None:
    if not isinstance(term, dict) or not metrics_by_key:
        return None
    query = normalize_recall_label(query_text)
    if not query:
        return None
    metric = resolve_term_metric_definition(term, metrics_by_key)
    if not metric:
        return None
    labels = [str(term.get("term") or ""), *[str(alias) for alias in term.get("aliases") or []]]
    best: dict[str, Any] | None = None
    for raw_label in labels:
        label = str(raw_label or "").strip()
        normalized = normalize_recall_label(label)
        if not is_protective_metric_label(normalized) or normalized not in query:
            continue
        candidate = build_metric_candidate(metric, topic, table, label, "exact_term", 0.96, "term")
        if best is None or compare_metric_candidate(candidate, best) > 0:
            best = candidate
    return best


def resolve_term_metric_definition(term: dict[str, Any], metrics_by_key: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    canonical = str(term.get("canonicalMetricKey") or "").strip()
    if canonical and canonical in metrics_by_key:
        return metrics_by_key[canonical]
    business_name = str(term.get("businessName") or "").strip()
    for metric in metrics_by_key.values():
        if business_name and business_name == str(metric.get("businessName") or "").strip():
            return metric
    return None


def build_metric_candidate(
    metric: dict[str, Any],
    topic: str,
    table: str,
    matched_label: str,
    resolution_type: str,
    confidence: float,
    reason_source: str,
) -> dict[str, Any]:
    metric_key = str(metric.get("metricKey") or "").strip()
    semantic_ref_id = "semantic:%s:%s:metric:%s" % (topic, table, metric_key)
    score = metric_candidate_fusion_score(confidence, resolution_type, 1)
    return {
        "semanticRefId": semantic_ref_id,
        "topic": topic,
        "tableName": table,
        "metricKey": metric_key,
        "businessName": str(metric.get("businessName") or metric_key),
        "canonicalMetricKey": str(metric.get("canonicalMetricKey") or ""),
        "aliasOf": str(metric.get("aliasOf") or ""),
        "metricLevel": str(metric.get("metricLevel") or ""),
        "formula": str(metric.get("formula") or metric.get("metricFormula") or ""),
        "sourceColumns": metric.get("sourceColumns") or [],
        "aliases": metric.get("aliases") or [],
        "metric": metric,
        "matchedMetricLabel": matched_label,
        "matchLength": len(normalize_recall_label(matched_label)),
        "metricResolutionType": resolution_type,
        "metricResolutionReason": "matched_%s:%s" % (reason_source, matched_label),
        "metricResolutionConfidence": round(float(confidence or 0.0), 3),
        "metricResolutionAmbiguous": False,
        "fusionScore": score,
    }


def compare_metric_candidate(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_score = (
        float(left.get("metricResolutionConfidence") or 0.0),
        int(left.get("matchLength") or 0),
        float(left.get("fusionScore") or 0.0),
    )
    right_score = (
        float(right.get("metricResolutionConfidence") or 0.0),
        int(right.get("matchLength") or 0),
        float(right.get("fusionScore") or 0.0),
    )
    if left_score > right_score:
        return 1
    if left_score < right_score:
        return -1
    return 0


def metric_candidate_fusion_score(confidence: float, resolution_type: str, rank: int) -> float:
    base = {
        "exact_business_name": 12000.0,
        "exact_alias": 11800.0,
        "exact_term": 11600.0,
        "exact_metric_key": 11400.0,
    }.get(str(resolution_type or ""), 11000.0)
    bounded_rank = max(1, int(rank or 1))
    return round(base + float(confidence or 0.0) * 100.0 - bounded_rank, 6)


def metric_trace_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in candidates or []:
        payload.append(
            {
                "semanticRefId": str(item.get("semanticRefId") or ""),
                "topic": str(item.get("topic") or ""),
                "tableName": str(item.get("tableName") or ""),
                "metricKey": str(item.get("metricKey") or ""),
                "businessName": str(item.get("businessName") or ""),
                "matchedMetricLabel": str(item.get("matchedMetricLabel") or ""),
                "metricResolutionType": str(item.get("metricResolutionType") or ""),
                "metricResolutionReason": str(item.get("metricResolutionReason") or ""),
                "metricResolutionConfidence": float(item.get("metricResolutionConfidence") or 0.0),
                "metricResolutionAmbiguous": bool(item.get("metricResolutionAmbiguous") or False),
            }
        )
    return payload


def build_retrieval_profile(
    query_text: str,
    topics: list[str],
    include_base_wiki: bool,
    metric_candidates: list[dict[str, Any]],
    settings: Settings,
    intent_kind: str = "",
    complexity: str = "",
) -> dict[str, Any]:
    query = str(query_text or "").strip()
    lowered = query.lower()
    reasons: list[str] = []
    query_type = query_type_from_fast_understanding(
        intent_kind=intent_kind,
        complexity=complexity,
        include_base_wiki=include_base_wiki,
    )
    if query_type:
        reasons.append("fast_understanding:%s/%s" % (intent_kind or "unknown", complexity or "unknown"))
    else:
        query_type = classify_query_type(query=query, topics=topics, metric_candidates=metric_candidates, include_base_wiki=include_base_wiki, reasons=reasons)
    profile_templates = configured_retrieval_profiles(settings)
    selected = dict(profile_templates.get(query_type) or profile_templates.get("multi_hop_analysis") or {})
    profile_kind = str(selected.get("profileKind") or "balanced")
    text_top_k = int(selected.get("textTopK") or settings.es_text_top_k or 12)
    vector_top_k = int(selected.get("vectorTopK") or settings.es_vector_top_k or 12)
    broad_text_top_k = int(selected.get("broadTextTopK") or settings.es_broad_text_top_k or 4)
    broad_vector_top_k = int(selected.get("broadVectorTopK") or settings.es_broad_vector_top_k or 4)
    hybrid_top_k = int(selected.get("hybridTopK") or settings.es_hybrid_top_k or 24)
    complexity_score = int(selected.get("complexityScore") or estimate_query_complexity(query, topics, metric_candidates, include_base_wiki))
    if any(str(item.get("metricResolutionType") or "").startswith("exact") for item in metric_candidates):
        reasons.append("explicit_metric_candidate")
    return {
        "profileKind": profile_kind,
        "queryType": query_type,
        "intentKind": str(intent_kind or ""),
        "fastComplexity": str(complexity or ""),
        "complexity": complexity_score,
        "reasons": reasons,
        "textTopK": text_top_k,
        "vectorTopK": vector_top_k,
        "broadTextTopK": broad_text_top_k,
        "broadVectorTopK": broad_vector_top_k,
        "hybridTopK": hybrid_top_k,
        "broadSearchEnabled": bool(selected.get("broadSearchEnabled", True)),
        "sourceTypeCaps": selected.get("sourceTypeCaps") or {},
        "queryHash": hashlib.sha256(lowered.encode("utf-8")).hexdigest()[:12] if lowered else "",
    }


def query_type_from_fast_understanding(intent_kind: str, complexity: str, include_base_wiki: bool) -> str:
    kind = str(intent_kind or "").strip().lower()
    level = str(complexity or "").strip().lower()
    if kind == "rule_only" or (include_base_wiki and kind not in {"rule_data_mix", "mixed_rule_data"}):
        return "rule_qa"
    if kind in {"rule_data_mix", "mixed_rule_data"}:
        return "mixed_rule_data"
    if kind == "detail_lookup":
        return "detail_lookup"
    if kind == "multi_metric":
        return "multi_metric"
    if kind in {"multi_hop", "analysis"} or level == "complex":
        return "multi_hop_analysis"
    if kind == "metric_query" or level == "simple":
        return "simple_metric"
    return ""


def configured_retrieval_profiles(settings: Settings) -> dict[str, dict[str, Any]]:
    profiles = default_retrieval_profiles(settings)
    raw = str(getattr(settings, "es_retrieval_profiles_json", "") or "").strip()
    if not raw:
        return profiles
    try:
        payload = json.loads(raw)
    except Exception:
        return profiles
    if not isinstance(payload, dict):
        return profiles
    for query_type, override in payload.items():
        if not isinstance(override, dict):
            continue
        base = dict(profiles.get(str(query_type)) or {})
        source_type_caps = dict(base.get("sourceTypeCaps") or {})
        if isinstance(override.get("sourceTypeCaps"), dict):
            source_type_caps.update({str(key): int(value) for key, value in override.get("sourceTypeCaps", {}).items() if isinstance(value, (int, float))})
        merged = {**base, **override}
        if source_type_caps:
            merged["sourceTypeCaps"] = source_type_caps
        profiles[str(query_type)] = merged
    return profiles


def default_retrieval_profiles(settings: Settings) -> dict[str, dict[str, Any]]:
    return {
        "simple_metric": {
            "profileKind": "focused",
            "textTopK": max(6, int(settings.es_text_top_k or 12) - 4),
            "vectorTopK": max(6, int(settings.es_vector_top_k or 12) - 4),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(2, int(settings.es_broad_vector_top_k or 4) - 1),
            "hybridTopK": max(12, min(int(settings.es_hybrid_top_k or 24), 16)),
            "broadSearchEnabled": True,
            "complexityScore": 1,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 10, "SEMANTIC_RELATIONSHIP": 5, "SEMANTIC_TABLE_ASSET": 4, "BASE_WIKI": 2},
        },
        "multi_metric": {
            "profileKind": "balanced",
            "textTopK": int(settings.es_text_top_k or 12),
            "vectorTopK": int(settings.es_vector_top_k or 12),
            "broadTextTopK": int(settings.es_broad_text_top_k or 4),
            "broadVectorTopK": int(settings.es_broad_vector_top_k or 4),
            "hybridTopK": int(settings.es_hybrid_top_k or 24),
            "broadSearchEnabled": True,
            "complexityScore": 2,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 12, "SEMANTIC_RELATIONSHIP": 7, "SEMANTIC_TABLE_ASSET": 6, "BASE_WIKI": 3},
        },
        "multi_hop_analysis": {
            "profileKind": "broad",
            "textTopK": min(max(int(settings.es_text_top_k or 12), 12) + 4, 18),
            "vectorTopK": min(max(int(settings.es_vector_top_k or 12), 12) + 4, 18),
            "broadTextTopK": min(max(int(settings.es_broad_text_top_k or 4), 4) + 2, 8),
            "broadVectorTopK": min(max(int(settings.es_broad_vector_top_k or 4), 4) + 2, 8),
            "hybridTopK": min(max(int(settings.es_hybrid_top_k or 24), 24) + 4, 32),
            "broadSearchEnabled": True,
            "complexityScore": 5,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 14, "SEMANTIC_RELATIONSHIP": 10, "SEMANTIC_TABLE_ASSET": 8, "BASE_WIKI": 4},
        },
        "rule_qa": {
            "profileKind": "balanced",
            "textTopK": max(8, int(settings.es_text_top_k or 12) - 2),
            "vectorTopK": max(6, int(settings.es_vector_top_k or 12) - 4),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(2, int(settings.es_broad_vector_top_k or 4) - 2),
            "hybridTopK": max(12, min(int(settings.es_hybrid_top_k or 24), 18)),
            "broadSearchEnabled": True,
            "complexityScore": 3,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 8, "SEMANTIC_RELATIONSHIP": 4, "SEMANTIC_TABLE_ASSET": 4, "BASE_WIKI": 6},
        },
        "mixed_rule_data": {
            "profileKind": "broad",
            "textTopK": min(max(int(settings.es_text_top_k or 12), 12) + 2, 16),
            "vectorTopK": min(max(int(settings.es_vector_top_k or 12), 12) + 2, 16),
            "broadTextTopK": min(max(int(settings.es_broad_text_top_k or 4), 4) + 1, 6),
            "broadVectorTopK": min(max(int(settings.es_broad_vector_top_k or 4), 4) + 1, 6),
            "hybridTopK": min(max(int(settings.es_hybrid_top_k or 24), 24) + 2, 28),
            "broadSearchEnabled": True,
            "complexityScore": 4,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 12, "SEMANTIC_RELATIONSHIP": 9, "SEMANTIC_TABLE_ASSET": 7, "BASE_WIKI": 6},
        },
        "detail_lookup": {
            "profileKind": "focused",
            "textTopK": max(6, int(settings.es_text_top_k or 12) - 3),
            "vectorTopK": max(4, int(settings.es_vector_top_k or 12) - 6),
            "broadTextTopK": max(2, int(settings.es_broad_text_top_k or 4) - 1),
            "broadVectorTopK": max(1, int(settings.es_broad_vector_top_k or 4) - 2),
            "hybridTopK": max(10, min(int(settings.es_hybrid_top_k or 24), 14)),
            "broadSearchEnabled": True,
            "complexityScore": 2,
            "sourceTypeCaps": {"SEMANTIC_METRIC": 8, "SEMANTIC_RELATIONSHIP": 6, "SEMANTIC_TABLE_ASSET": 5, "BASE_WIKI": 2},
        },
    }


def classify_query_type(
    query: str,
    topics: list[str],
    metric_candidates: list[dict[str, Any]],
    include_base_wiki: bool,
    reasons: list[str] | None = None,
) -> str:
    out = reasons if reasons is not None else []
    metric_count = len(metric_candidates)
    relationship_tokens = ["关联", "对应", "join", "同时看", "再看", "并看"]
    analysis_tokens = ["趋势", "分析", "波动", "判断", "风险", "最高", "最低", "top", "前", "对比"]
    detail_tokens = ["明细", "详情", "订单号", "sub_order", "id", "查询订单"]
    has_relationship = any(token in query for token in relationship_tokens)
    has_analysis = any(token in query for token in analysis_tokens)
    has_detail = any(token in query for token in detail_tokens)
    if include_base_wiki and (has_analysis or len(topics) >= 2):
        out.append("mixed_rule_data")
        return "mixed_rule_data"
    if include_base_wiki:
        out.append("rule_qa")
        return "rule_qa"
    if has_detail:
        out.append("detail_lookup")
        return "detail_lookup"
    if has_relationship or len(topics) >= 2 or (metric_count >= 2 and has_analysis):
        out.append("multi_hop_analysis")
        return "multi_hop_analysis"
    if metric_count >= 2:
        out.append("multi_metric")
        return "multi_metric"
    out.append("simple_metric")
    return "simple_metric"


def estimate_query_complexity(
    query: str,
    topics: list[str],
    metric_candidates: list[dict[str, Any]],
    include_base_wiki: bool,
) -> int:
    score = 0
    if len(query or "") >= 24:
        score += 1
    if len(topics) >= 2:
        score += 1
    if len(metric_candidates) >= 2:
        score += 1
    if any(token in query for token in ["同时", "并且", "关联", "分别", "趋势", "最高", "最低", "top", "前", "分析", "对比"]):
        score += 1
    if include_base_wiki:
        score += 1
    return score


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


def source_type_top_k_policy(
    include_base_wiki: bool = False,
    query_text: str = "",
    topics: list[str] | None = None,
    metric_candidates: list[dict[str, Any]] | None = None,
    retrieval_profile: dict[str, Any] | None = None,
) -> dict[str, int]:
    profile = retrieval_profile or {}
    configured_caps = profile.get("sourceTypeCaps") or {}
    if isinstance(configured_caps, dict) and configured_caps:
        policy = {
            "SEMANTIC_METRIC": int(configured_caps.get("SEMANTIC_METRIC") or 12),
            "SEMANTIC_RELATIONSHIP": int(configured_caps.get("SEMANTIC_RELATIONSHIP") or 8),
            "SEMANTIC_TABLE_ASSET": int(configured_caps.get("SEMANTIC_TABLE_ASSET") or 6),
            "BASE_WIKI": int(configured_caps.get("BASE_WIKI") or (6 if include_base_wiki else 3)),
        }
    else:
        profile_kind = str(profile.get("profileKind") or "balanced")
        if profile_kind == "focused":
            policy = {
                "SEMANTIC_METRIC": 10,
                "SEMANTIC_RELATIONSHIP": 5,
                "SEMANTIC_TABLE_ASSET": 4,
                "BASE_WIKI": 4 if include_base_wiki else 2,
            }
        elif profile_kind == "broad":
            policy = {
                "SEMANTIC_METRIC": 14,
                "SEMANTIC_RELATIONSHIP": 10,
                "SEMANTIC_TABLE_ASSET": 8,
                "BASE_WIKI": 8 if include_base_wiki else 4,
            }
        else:
            policy = {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 8,
                "SEMANTIC_TABLE_ASSET": 6,
                "BASE_WIKI": 6 if include_base_wiki else 3,
            }
    query = str(query_text or "")
    relationship_heavy = any(token in query for token in ["关联", "对应", "join", "同时看", "再看", "并看"])
    metric_heavy = bool(metric_candidates) or any(token in query for token in ["金额", "率", "量", "GMV", "退款", "下单"])
    if relationship_heavy:
        policy["SEMANTIC_RELATIONSHIP"] = min(policy["SEMANTIC_RELATIONSHIP"] + 2, 12)
    if metric_heavy:
        policy["SEMANTIC_METRIC"] = min(policy["SEMANTIC_METRIC"] + 1, 16)
    if topics and len(topics) >= 3:
        policy["SEMANTIC_TABLE_ASSET"] = min(policy["SEMANTIC_TABLE_ASSET"] + 1, 10)
    return policy


def limit_recall_items_by_source_type(items: list[RecallItem], policy: dict[str, int], limit: int = 24) -> list[RecallItem]:
    if not items:
        return []
    counts: dict[str, int] = {}
    selected: list[RecallItem] = []
    for item in sorted(items, key=lambda value: value.fusion_score, reverse=True):
        source_type = str(item.source_type or "UNKNOWN").upper()
        cap = int(policy.get(source_type, max(1, limit)))
        if counts.get(source_type, 0) < cap:
            selected.append(item)
            counts[source_type] = counts.get(source_type, 0) + 1
    return selected[: max(1, int(limit or len(selected)))]


def recall_channels_for_items(items: list[RecallItem]) -> list[str]:
    channels: list[str] = []
    for item in items or []:
        metadata = item.metadata or {}
        raw_channels = metadata.get("recallChannels") or [metadata.get("recallChannel")]
        for raw in raw_channels or []:
            channel = str(raw or "").strip()
            if channel and channel not in channels:
                channels.append(channel)
    return channels


def retrieval_lane_trace(
    retrieval_profile: dict[str, Any],
    vector_enabled: bool,
    include_base_wiki: bool,
    has_metric_candidates: bool,
    broad_enabled: bool,
) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    lanes.append({"lane": "metric_candidate_lane", "enabled": has_metric_candidates, "topK": 6 if has_metric_candidates else 0})
    lanes.append({"lane": "bm25_lane", "enabled": True, "topK": int(retrieval_profile.get("textTopK") or 0)})
    lanes.append({"lane": "vector_lane", "enabled": vector_enabled, "topK": int(retrieval_profile.get("vectorTopK") or 0) if vector_enabled else 0})
    broad_flag = bool(retrieval_profile.get("broadSearchEnabled", True)) and broad_enabled
    lanes.append({"lane": "broad_bm25_lane", "enabled": broad_flag, "topK": int(retrieval_profile.get("broadTextTopK") or 0) if broad_flag else 0})
    lanes.append({"lane": "broad_vector_lane", "enabled": broad_flag and vector_enabled, "topK": int(retrieval_profile.get("broadVectorTopK") or 0) if broad_flag and vector_enabled else 0})
    lanes.append({"lane": "exact_metric_fallback_lane", "enabled": has_metric_candidates, "topK": 3 if has_metric_candidates else 0})
    if include_base_wiki:
        lanes.append({"lane": "rule_wiki_lane", "enabled": True, "topK": int((retrieval_profile.get("sourceTypeCaps") or {}).get("BASE_WIKI") or 0)})
    return lanes


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
