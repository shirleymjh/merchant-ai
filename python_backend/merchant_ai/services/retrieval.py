from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
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
from merchant_ai.services.semantic_request import semantic_request_cache_key
from merchant_ai.services.time_semantics import resolve_time_range


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
        rewritten_query = rewrite_retrieval_query(request)
        recall_bundle = self.recall_service.recall(
            rewritten_query,
            ExtractedKeywords(keywords=request.keywords),
            request.history_rows,
            request.knowledge_context,
            request.merchant_id,
            request.topic_categories,
        )
        if request.topic_categories and not request.knowledge_request:
            broad_bundle = self.recall_service.recall(
                rewritten_query,
                ExtractedKeywords(keywords=request.keywords),
                request.history_rows,
                request.knowledge_context,
                request.merchant_id,
                [],
            )
            recall_bundle = merge_recall_bundles(recall_bundle, broad_bundle)
        governed_items, filtered = filter_recall_items_by_governance(recall_bundle.items, request)
        reranked_items = business_rerank_recall_items(governed_items, rewritten_query, request)
        source_caps = source_type_top_k_policy(
            include_rules=route_is_rule_sensitive(request),
            query_text=rewritten_query,
            topics=[str(item.value if hasattr(item, "value") else item) for item in request.topic_categories],
        )
        reranked_items = limit_recall_items_by_source_type(reranked_items, source_caps, limit=24)
        recall_bundle = RecallBundle(
            items=reranked_items,
            top_score=reranked_items[0].fusion_score if reranked_items else 0.0,
            merged_context="\n\n".join(
                "召回片段 [%s] %s\n%s" % (item.source_type, item.title, item.content[:1200])
                for item in reranked_items
            ),
        )
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
            rewritten_query=rewritten_query,
            governance_filtered=filtered,
            rerank_applied=bool(reranked_items),
            source_type_top_k=source_caps,
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
        rewritten_query = rewrite_retrieval_query(request)
        query_text = retrieval_query_text(request, rewritten_query=rewritten_query)
        normalized_categories = [category for category in [normalize_question_category(item) for item in request.topic_categories] if category]
        topics = self._allowed_topics(normalized_categories)
        include_rules = topic_categories_support_knowledge_capability(
            self.topic_assets,
            normalized_categories,
            "rule",
        ) or route_is_rule_sensitive(request)
        metric_candidates = self._resolve_metric_candidates(query_text, topics)
        retrieval_profile = build_retrieval_profile(
            query_text=query_text,
            topics=topics,
            include_rules=include_rules,
            metric_candidates=metric_candidates,
            intent_kind=request.intent_kind,
            complexity=request.complexity,
            settings=self.settings,
        )
        source_type_top_k = source_type_top_k_policy(
            include_rules=include_rules,
            query_text=query_text,
            topics=topics,
            metric_candidates=metric_candidates,
            retrieval_profile=retrieval_profile,
        )
        route_slots = request.route_slots if isinstance(request.route_slots, dict) else {}
        object_filters = list(route_slots.get("objectRefs") or route_slots.get("object_refs") or [])
        knowledge_filter = (
            request.knowledge_request.model_dump(by_alias=True)
            if request.knowledge_request is not None
            else {}
        )
        metric_contracts = [
            {
                "metricKey": str(item.get("canonicalMetricKey") or item.get("metricKey") or ""),
                "ownerTable": str(item.get("ownerTable") or item.get("tableName") or ""),
            }
            for item in metric_candidates
            if str(item.get("canonicalMetricKey") or item.get("metricKey") or "")
        ]
        cache_key = (
            semantic_request_cache_key(
                "es_recall",
                topics=topics,
                metrics=metric_contracts,
                dimensions=list(route_slots.get("dimensions") or []),
                filters=[
                    *object_filters,
                    {"includeRules": include_rules, "intentKind": request.intent_kind, "complexity": request.complexity},
                    *([knowledge_filter] if knowledge_filter else []),
                ],
                time_range=resolve_time_range(query_text, self.settings.business_timezone),
                asset_version={
                    "indexVersion": self._index_version(),
                    "vectorEnabled": self._vector_enabled(),
                    "embeddingModel": self.settings.embedding_model if self._vector_enabled() else "",
                    "retrievalPolicy": {"rrfK": self.settings.es_rrf_k, "sourceTypeTopK": source_type_top_k},
                },
                scope={
                    "merchantId": request.merchant_id,
                    "accessRole": request.access_role,
                    "permissions": sorted(request.permissions),
                },
            )
            if metric_contracts or object_filters or knowledge_filter
            else ""
        )
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            return KnowledgeBundle.model_validate(cached)
        self._active_retrieval_profile = retrieval_profile
        try:
            try:
                items = self._search(query_text, topics, include_rules=include_rules)
                if topics and not request.knowledge_request and bool(retrieval_profile.get("broadSearchEnabled", True)):
                    try:
                        broad_items = self._search(query_text, [], include_rules=False)
                        items = rrf_fuse_recall_items(
                            [("topic_scope", items), ("broad_scope", broad_items)],
                            rrf_k=self.settings.es_rrf_k,
                            score_scale=self.settings.es_rrf_score_scale,
                            limit=max(1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24)),
                        )
                    except Exception:
                        pass
                items = merge_recall_items(items, self._metric_candidate_items(query_text, metric_candidates))
                items = merge_recall_items(items, self._exact_metric_evidence(query_text, topics))
                items = self._attach_current_asset_governance(items)
                items, governance_filtered = filter_recall_items_by_governance(items, request)
                items = business_rerank_recall_items(items, query_text, request)
                items = limit_recall_items_by_source_type(
                    items,
                    source_type_top_k,
                    limit=max(1, int(retrieval_profile.get("hybridTopK") or self.settings.es_hybrid_top_k or len(items) or 1)),
                )
                blocked_reason = ""
            except Exception as exc:
                items = []
                governance_filtered = {}
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
                include_rules=include_rules,
                has_metric_candidates=bool(metric_candidates),
                broad_enabled=bool(topics),
            ),
            rewritten_query=rewritten_query,
            governance_filtered=governance_filtered,
            rerank_applied=bool(items),
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

    def _attach_current_asset_governance(self, items: list[RecallItem]) -> list[RecallItem]:
        """Recheck ES hits against the live published semantic asset.

        This keeps role/status/version checks effective even before an older ES
        index has been rebuilt with the latest governance metadata.
        """
        governed: list[RecallItem] = []
        asset_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items or []:
            metadata = dict(item.metadata or {})
            topic = str(item.topic or metadata.get("topic") or "")
            table = str(item.table or metadata.get("tableName") or "")
            if not topic or not table:
                governed.append(item)
                continue
            key = (topic, table)
            if key not in asset_cache:
                asset = self.topic_assets.load_table_asset(topic, table)
                current = recall_governance_metadata(asset)
                current_version = str(asset.get("version") or asset.get("semanticVersion") or "") if isinstance(asset, dict) else ""
                if current_version:
                    current["activeVersion"] = current_version
                asset_cache[key] = current
            current = asset_cache[key]
            merged = {
                **current,
                **metadata,
                "activeVersion": current.get("activeVersion") or metadata.get("activeVersion") or "",
                "assetStatus": current.get("status") or "",
                "assetMerchantId": current.get("merchantId") or "",
                "assetAllowedRoles": current.get("allowedRoles") or [],
                "assetRequiredPermissions": current.get("requiredPermissions") or [],
                "assetVisibilityPolicy": current.get("visibilityPolicy") or {},
                "assetExpiresAt": current.get("expiresAt") or "",
            }
            governed.append(item.model_copy(update={"metadata": merged}))
        return governed

    def _search(self, query_text: str, topics: list[str], include_rules: bool = False) -> list[RecallItem]:
        text_items = self._text_search(query_text, topics, include_rules=include_rules)
        vector_items: list[RecallItem] = []
        if not self._vector_enabled() or not query_text:
            return rrf_fuse_recall_items(
                [("bm25", text_items)],
                rrf_k=self.settings.es_rrf_k,
                score_scale=self.settings.es_rrf_score_scale,
                limit=self._hybrid_size(),
            )
        try:
            vector = self._embed_text(query_text)
            vector_items = self._vector_search(query_text, vector, topics, include_rules=include_rules) if vector else []
        except Exception:
            vector_items = []
        if not vector_items:
            return rrf_fuse_recall_items(
                [("bm25", text_items)],
                rrf_k=self.settings.es_rrf_k,
                score_scale=self.settings.es_rrf_score_scale,
                limit=self._hybrid_size(),
            )
        return rrf_fuse_recall_items(
            [("bm25", text_items), ("vector", vector_items)],
            rrf_k=self.settings.es_rrf_k,
            score_scale=self.settings.es_rrf_score_scale,
            limit=self._hybrid_size(),
        )

    def _text_search(self, query_text: str, topics: list[str], include_rules: bool = False) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        size = self._text_size(topics)
        query = self._text_query(query_text, topics, include_rules=include_rules)
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

    def _vector_search(self, query_text: str, query_vector: list[float], topics: list[str], include_rules: bool = False) -> list[RecallItem]:
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        if not query_vector:
            return []
        size = self._vector_size(topics)
        filters = self._filters(topics, include_rules=include_rules)
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

    def _text_query(self, query_text: str, topics: list[str], include_rules: bool = False) -> dict[str, object]:
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
        filters = self._filters(topics, include_rules=include_rules)
        if must or filters:
            return {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}
        return {"match_all": {}}

    def _filters(self, topics: list[str], include_rules: bool = False) -> list[dict[str, object]]:
        filters: list[dict[str, object]] = []
        if topics:
            topic_should: list[dict[str, object]] = [
                {"terms": {"topic": topics}},
                {"terms": {"topic.keyword": topics}},
                {"terms": {"metadata.topic": topics}},
                {"terms": {"metadata.topic.keyword": topics}},
            ]
            if include_rules:
                topic_should.append({"term": {"source_type": "GOVERNED_RULE"}})
            filters.append({"bool": {"should": topic_should, "minimum_should_match": 1}})
        elif include_rules:
            filters.append({"term": {"source_type": "GOVERNED_RULE"}})
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

    def _hybrid_size(self) -> int:
        profile = self._active_retrieval_profile or {}
        return max(1, int(profile.get("hybridTopK") or self.settings.es_hybrid_top_k or 24))

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
        metrics_by_scope: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        governance_by_scope: dict[tuple[str, str], dict[str, Any]] = {}
        for topic in topic_names:
            for manifest_item in self.topic_assets.load_manifest(topic):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                metrics = [metric for metric in self.topic_assets.load_table_metrics(topic, table) if isinstance(metric, dict)]
                table_asset = self.topic_assets.load_table_asset(topic, table)
                table_governance = recall_governance_metadata(table_asset)
                metrics_by_key = {
                    str(metric.get("metricKey") or ""): metric
                    for metric in metrics
                    if str(metric.get("metricKey") or "")
                }
                metrics_by_scope[(topic, table)] = metrics_by_key
                governance_by_scope[(topic, table)] = table_governance
                for metric in metrics:
                    candidate = resolve_metric_candidate(metric, topic, table, query)
                    if candidate is None:
                        continue
                    candidate["governance"] = {**table_governance, **recall_governance_metadata(metric)}
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(current.get("metricResolutionConfidence") or 0.0):
                        by_id[semantic_ref_id] = candidate
                for term in self.topic_assets.load_table_terms(topic, table):
                    candidate = resolve_term_metric_candidate(term, metrics_by_key, topic, table, query)
                    if candidate is None:
                        continue
                    resolved_metric = candidate.get("metric") if isinstance(candidate.get("metric"), dict) else {}
                    candidate["governance"] = {**table_governance, **recall_governance_metadata(resolved_metric)}
                    semantic_ref_id = str(candidate["semanticRefId"])
                    current = by_id.get(semantic_ref_id)
                    if current is None or float(candidate.get("metricResolutionConfidence") or 0.0) > float(current.get("metricResolutionConfidence") or 0.0):
                        by_id[semantic_ref_id] = candidate
        candidates = suppress_embedded_generic_metric_candidates(query, list(by_id.values()))
        by_id = {str(candidate.get("semanticRefId") or ""): candidate for candidate in candidates}
        for (topic, table), metrics_by_key in metrics_by_scope.items():
            scoped_matches = [
                candidate
                for candidate in candidates
                if str(candidate.get("topic") or "") == topic
                and str(candidate.get("tableName") or "") == table
            ]
            for linked_candidate in linked_metric_variant_candidates(
                scoped_matches,
                metrics_by_key,
                topic,
                table,
                governance_by_scope.get((topic, table), {}),
            ):
                semantic_ref_id = str(linked_candidate.get("semanticRefId") or "")
                current = by_id.get(semantic_ref_id)
                if current is None or compare_metric_candidate(linked_candidate, current) > 0:
                    by_id[semantic_ref_id] = linked_candidate
        candidates = list(by_id.values())
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            label_key = normalize_recall_label(str(candidate.get("matchedMetricLabel") or ""))
            if label_key:
                label_groups.setdefault(label_key, []).append(candidate)
        suppressed_alias_candidates: set[str] = set()
        for label_key, group in label_groups.items():
            unique_metrics = {
                (str(item.get("topic") or ""), str(item.get("tableName") or ""), str(item.get("metricKey") or ""))
                for item in group
            }
            if len(unique_metrics) <= 1:
                continue
            canonical_owner, canonical_aliases = canonical_metric_family_owner(group)
            if canonical_owner is not None:
                canonical_key = str(canonical_owner.get("metricKey") or "")
                canonical_owner["metricResolutionAmbiguous"] = False
                canonical_owner["metricResolutionReason"] = "%s; canonical_family_owner=%s" % (
                    str(canonical_owner.get("metricResolutionReason") or ""),
                    canonical_key,
                )
                canonical_owner["metricResolutionConfidence"] = max(
                    0.9,
                    round(float(canonical_owner.get("metricResolutionConfidence") or 0.0), 3),
                )
                suppressed_alias_candidates.update(
                    str(item.get("semanticRefId") or "")
                    for item in canonical_aliases
                    if str(item.get("semanticRefId") or "")
                )
                continue
            for item in group:
                item["metricResolutionAmbiguous"] = True
                item["metricResolutionConfidence"] = max(0.4, round(float(item.get("metricResolutionConfidence") or 0.0) - 0.18, 3))
                item["metricResolutionReason"] = "%s; ambiguous_label=%s" % (str(item.get("metricResolutionReason") or ""), label_key)
        if suppressed_alias_candidates:
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("semanticRefId") or "") not in suppressed_alias_candidates
            ]
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
            score = metric_candidate_fusion_score(confidence, resolution_type, rank)
            metadata = {
                "semanticSource": "metrics",
                "semanticKind": "METRIC",
                "semanticRefId": semantic_ref_id,
                "semanticPath": "topics/%s/tables/%s/asset.json#metric:%s" % (topic, table, metric_key),
                "metricKey": metric_key,
                "tableName": table,
                "topic": topic,
                "businessName": candidate.get("businessName") or metric_key,
                "canonicalMetricKey": candidate.get("canonicalMetricKey") or "",
                "aliasOf": candidate.get("aliasOf") or "",
                "metricLevel": candidate.get("metricLevel") or "",
                "metricGrain": candidate.get("metricGrain") or "",
                "metricIntent": candidate.get("metricIntent") or "",
                "aggregationPolicy": candidate.get("aggregationPolicy") or "",
                "selectionGuidance": candidate.get("selectionGuidance") or "",
                "preferredUseCases": candidate.get("preferredUseCases") or [],
                "notPreferredUseCases": candidate.get("notPreferredUseCases") or [],
                "temporalVariants": candidate.get("temporalVariants") or {},
                "linkedVariantOf": candidate.get("linkedVariantOf") or "",
                "linkedVariantPath": candidate.get("linkedVariantPath") or "",
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
                "metricResolverScore": score,
                "recallSupplement": "metric_candidate_resolution",
                **dict(candidate.get("governance") or {}),
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


def linked_metric_variant_candidates(
    base_candidates: list[dict[str, Any]],
    metrics_by_key: dict[str, dict[str, Any]],
    topic: str,
    table: str,
    table_governance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand only links explicitly published in a metric's variant contract.

    Retrieval does not decide which linked metric fits the question.  It exposes
    compact candidates with the asset's aggregation and selection metadata so a
    downstream semantic selector can make that decision.
    """

    linked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for base_candidate in base_candidates:
        base_metric = base_candidate.get("metric") if isinstance(base_candidate.get("metric"), dict) else {}
        base_key = str(base_candidate.get("metricKey") or "")
        base_ref = str(base_candidate.get("semanticRefId") or "")
        base_confidence = max(0.0, min(1.0, float(base_candidate.get("metricResolutionConfidence") or 0.0)))
        for link_path, variant_key in metric_linked_variant_refs(base_metric):
            if not variant_key or variant_key == base_key or (base_ref, variant_key) in seen:
                continue
            seen.add((base_ref, variant_key))
            variant_metric = metrics_by_key.get(variant_key)
            if not isinstance(variant_metric, dict):
                continue
            linked_confidence = min(0.94, max(0.4, base_confidence - 0.03))
            candidate = build_metric_candidate(
                variant_metric,
                topic,
                table,
                str(base_candidate.get("matchedMetricLabel") or base_candidate.get("businessName") or variant_key),
                "linked_variant",
                linked_confidence,
                "temporalVariants.%s" % link_path,
            )
            candidate["metricResolutionReason"] = "%s; linked_variant_of=%s; link_path=%s" % (
                str(candidate.get("metricResolutionReason") or ""),
                base_key,
                link_path,
            )
            candidate["linkedVariantOf"] = base_ref
            candidate["linkedVariantPath"] = link_path
            candidate["governance"] = {
                **table_governance,
                **recall_governance_metadata(variant_metric),
            }
            linked.append(candidate)
    return linked


def metric_linked_variant_refs(metric: dict[str, Any]) -> list[tuple[str, str]]:
    variants = metric.get("temporalVariants") or metric.get("temporal_variants") or {}
    if not isinstance(variants, (dict, list, tuple)):
        return []
    refs: list[tuple[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, str):
            metric_ref = value.strip()
            if metric_ref:
                refs.append((path, metric_ref))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = "%s.%s" % (path, key) if path else str(key)
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                child_path = "%s[%d]" % (path, index)
                visit(child, child_path)

    visit(variants, "")
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, metric_ref in refs:
        if metric_ref in seen:
            continue
        seen.add(metric_ref)
        deduped.append((path, metric_ref))
    return deduped


def suppress_embedded_generic_metric_candidates(
    query_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer an explicit qualified label over its embedded generic label.

    If a qualified label contains a shorter independent label, an occurrence
    embedded only inside the qualified phrase does not count as a second request.
    If the query separately contains both labels, both candidates are retained.
    """

    query = normalize_recall_label(query_text)
    if not query or len(candidates) <= 1:
        return candidates
    suppressed: set[str] = set()
    labelled = [
        (candidate, normalize_recall_label(str(candidate.get("matchedMetricLabel") or "")))
        for candidate in candidates
    ]
    for qualified, long_label in labelled:
        if not long_label or long_label not in query:
            continue
        remainder = query.replace(long_label, " ")
        for generic, short_label in labelled:
            if generic is qualified or not short_label or short_label == long_label:
                continue
            if short_label not in long_label or short_label in remainder:
                continue
            qualified_ref = str(qualified.get("semanticRefId") or "")
            generic_ref = str(generic.get("semanticRefId") or "")
            if qualified_ref and generic_ref and qualified_ref != generic_ref:
                suppressed.add(generic_ref)
    if not suppressed:
        return candidates
    return [candidate for candidate in candidates if str(candidate.get("semanticRefId") or "") not in suppressed]


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
        "metricGrain": str(metric.get("metricGrain") or metric.get("grainHint") or ""),
        "metricIntent": str(metric.get("metricIntent") or ""),
        "aggregationPolicy": str(metric.get("aggregationPolicy") or ""),
        "selectionGuidance": str(metric.get("selectionGuidance") or ""),
        "preferredUseCases": metric.get("preferredUseCases") or [],
        "notPreferredUseCases": metric.get("notPreferredUseCases") or [],
        "temporalVariants": metric.get("temporalVariants") or {},
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


def canonical_metric_family_owner(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Return the governed owner when every candidate belongs to one alias family.

    A shared user label is not a real ambiguity when the semantic layer explicitly
    declares every variant as an alias of one canonical metric and publishes that
    canonical metric in the same owner table.  Keeping this rule metadata-driven
    avoids teaching retrieval any business-specific metric names.
    """
    if len(candidates) <= 1:
        return None, []
    families: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        metric_key = str(candidate.get("metricKey") or "").strip()
        canonical_key = str(candidate.get("canonicalMetricKey") or candidate.get("aliasOf") or metric_key).strip()
        topic = str(candidate.get("topic") or "").strip()
        table = str(candidate.get("tableName") or "").strip()
        if not metric_key or not canonical_key or not table:
            return None, []
        families.add((topic, table, canonical_key))
    if len(families) != 1:
        return None, []
    _, _, canonical_key = next(iter(families))
    owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("metricKey") or "").strip() == canonical_key
        and not str(candidate.get("aliasOf") or "").strip()
    ]
    if len(owners) != 1:
        return None, []
    owner = owners[0]
    aliases = [candidate for candidate in candidates if candidate is not owner]
    return owner, aliases


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
    type_score = {
        "exact_business_name": 1.0,
        "exact_alias": 0.98,
        "exact_term": 0.96,
        "exact_metric_key": 0.94,
    }.get(str(resolution_type or ""), 0.72)
    bounded_rank = max(1, int(rank or 1))
    confidence_score = max(0.0, min(float(confidence or 0.0), 1.0))
    rank_penalty = min(0.15, (bounded_rank - 1) * 0.02)
    return round(max(0.0, min(1.0, type_score * 0.55 + confidence_score * 0.45 - rank_penalty)), 6)


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
                "aggregationPolicy": str(item.get("aggregationPolicy") or ""),
                "selectionGuidance": str(item.get("selectionGuidance") or ""),
                "temporalVariants": item.get("temporalVariants") or {},
                "linkedVariantOf": str(item.get("linkedVariantOf") or ""),
                "linkedVariantPath": str(item.get("linkedVariantPath") or ""),
            }
        )
    return payload


def build_retrieval_profile(
    query_text: str,
    topics: list[str],
    include_rules: bool,
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
        include_rules=include_rules,
    )
    if query_type:
        reasons.append("fast_understanding:%s/%s" % (intent_kind or "unknown", complexity or "unknown"))
    else:
        query_type = classify_query_type(query=query, topics=topics, metric_candidates=metric_candidates, include_rules=include_rules, reasons=reasons)
    profile_templates = configured_retrieval_profiles(settings)
    selected = dict(profile_templates.get(query_type) or profile_templates.get("multi_hop_analysis") or {})
    profile_kind = str(selected.get("profileKind") or "balanced")
    text_top_k = int(selected.get("textTopK") or settings.es_text_top_k or 12)
    vector_top_k = int(selected.get("vectorTopK") or settings.es_vector_top_k or 12)
    broad_text_top_k = int(selected.get("broadTextTopK") or settings.es_broad_text_top_k or 4)
    broad_vector_top_k = int(selected.get("broadVectorTopK") or settings.es_broad_vector_top_k or 4)
    hybrid_top_k = int(selected.get("hybridTopK") or settings.es_hybrid_top_k or 24)
    complexity_score = int(selected.get("complexityScore") or estimate_query_complexity(query, topics, metric_candidates, include_rules))
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


def query_type_from_fast_understanding(intent_kind: str, complexity: str, include_rules: bool) -> str:
    kind = str(intent_kind or "").strip().lower()
    level = str(complexity or "").strip().lower()
    if kind == "rule_only" or (include_rules and kind not in {"rule_data_mix", "mixed_rule_data"}):
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 10, "SEMANTIC_RELATIONSHIP": 5, "SEMANTIC_TABLE_ASSET": 4, "GOVERNED_RULE": 2},
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 12, "SEMANTIC_RELATIONSHIP": 7, "SEMANTIC_TABLE_ASSET": 6, "GOVERNED_RULE": 3},
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 14, "SEMANTIC_RELATIONSHIP": 10, "SEMANTIC_TABLE_ASSET": 8, "GOVERNED_RULE": 4},
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 8, "SEMANTIC_RELATIONSHIP": 4, "SEMANTIC_TABLE_ASSET": 4, "GOVERNED_RULE": 6},
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 12, "SEMANTIC_RELATIONSHIP": 9, "SEMANTIC_TABLE_ASSET": 7, "GOVERNED_RULE": 6},
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
            "sourceTypeCaps": {"SEMANTIC_METRIC": 8, "SEMANTIC_RELATIONSHIP": 6, "SEMANTIC_TABLE_ASSET": 5, "GOVERNED_RULE": 2},
        },
    }


def classify_query_type(
    query: str,
    topics: list[str],
    metric_candidates: list[dict[str, Any]],
    include_rules: bool,
    reasons: list[str] | None = None,
) -> str:
    out = reasons if reasons is not None else []
    metric_count = len(metric_candidates)
    relationship_tokens = ["关联", "对应", "join", "同时看", "再看", "并看"]
    analysis_tokens = ["趋势", "分析", "波动", "判断", "风险", "最高", "最低", "top", "前", "对比"]
    detail_tokens = ["明细", "详情", "记录", "id"]
    has_relationship = any(token in query for token in relationship_tokens)
    has_analysis = any(token in query for token in analysis_tokens)
    has_detail = any(token in query for token in detail_tokens)
    if include_rules and (has_analysis or len(topics) >= 2):
        out.append("mixed_rule_data")
        return "mixed_rule_data"
    if include_rules:
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
    include_rules: bool,
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
    if include_rules:
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
        if current is None:
            by_id[key] = item
            continue
        preferred = item if recall_item_sort_key(item) > recall_item_sort_key(current) else current
        other = current if preferred is item else item
        merged = merge_recall_item_metadata(preferred, other)
        has_final_score = any((candidate.metadata or {}).get("finalScore") is not None for candidate in [current, item])
        merged_score = float(preferred.fusion_score or 0.0) if has_final_score else max(float(current.fusion_score or 0.0), float(item.fusion_score or 0.0))
        by_id[key] = merged.model_copy(
            update={
                "fusion_score": merged_score,
            }
        )
    return sorted(by_id.values(), key=recall_item_sort_key, reverse=True)


def source_type_top_k_policy(
    include_rules: bool = False,
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
            "GOVERNED_RULE": int(configured_caps.get("GOVERNED_RULE") or (6 if include_rules else 3)),
        }
    else:
        profile_kind = str(profile.get("profileKind") or "balanced")
        if profile_kind == "focused":
            policy = {
                "SEMANTIC_METRIC": 10,
                "SEMANTIC_RELATIONSHIP": 5,
                "SEMANTIC_TABLE_ASSET": 4,
                "GOVERNED_RULE": 4 if include_rules else 2,
            }
        elif profile_kind == "broad":
            policy = {
                "SEMANTIC_METRIC": 14,
                "SEMANTIC_RELATIONSHIP": 10,
                "SEMANTIC_TABLE_ASSET": 8,
                "GOVERNED_RULE": 8 if include_rules else 4,
            }
        else:
            policy = {
                "SEMANTIC_METRIC": 12,
                "SEMANTIC_RELATIONSHIP": 8,
                "SEMANTIC_TABLE_ASSET": 6,
                "GOVERNED_RULE": 6 if include_rules else 3,
            }
    query = str(query_text or "")
    relationship_heavy = any(token in query for token in ["关联", "对应", "join", "同时看", "再看", "并看"])
    metric_heavy = bool(metric_candidates)
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
    for item in sorted(items, key=recall_item_sort_key, reverse=True):
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
    include_rules: bool,
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
    if include_rules:
        lanes.append({"lane": "governed_rule_lane", "enabled": True, "topK": int((retrieval_profile.get("sourceTypeCaps") or {}).get("GOVERNED_RULE") or 0)})
    return lanes


def rrf_fuse_recall_items(
    ranked_groups: list[tuple[str, list[RecallItem]]],
    rrf_k: int = 60,
    score_scale: float = 1000.0,
    limit: int = 24,
) -> list[RecallItem]:
    """Fuse ranked recall lists with reciprocal rank fusion.

    BM25 scores and vector similarities are not comparable. RRF only uses the
    rank position inside each channel, then normalizes the result to 0..1 so
    downstream ranking keeps the same score semantics when a channel degrades.
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
    active_lane_count = max(1, sum(1 for _, items in ranked_groups if items))
    theoretical_max = active_lane_count / float(k + 1)
    fused: list[RecallItem] = []
    for key, item in by_id.items():
        metadata = dict(item.metadata or {})
        raw_score = scores.get(key, 0.0)
        normalized_score = max(0.0, min(1.0, raw_score / theoretical_max)) if theoretical_max else 0.0
        if metadata.get("rrfRanks"):
            metadata["upstreamRrfRanks"] = metadata.get("rrfRanks")
        if metadata.get("channelScores"):
            metadata["upstreamChannelScores"] = metadata.get("channelScores")
        if metadata.get("rrfNormalizedScore") is not None:
            metadata["upstreamRrfNormalizedScore"] = metadata.get("rrfNormalizedScore")
        metadata["recallFusion"] = "rrf"
        metadata["scoreVersion"] = "recall_v2"
        metadata["rrfScore"] = raw_score
        metadata["rrfNormalizedScore"] = normalized_score
        metadata["rrfDisplayScore"] = raw_score * scale
        metadata["retrievalScore"] = normalized_score
        metadata["rrfK"] = k
        metadata["rrfActiveLaneCount"] = active_lane_count
        metadata["rrfRanks"] = ranks.get(key, {})
        metadata["channelScores"] = channel_scores.get(key, {})
        metadata["recallChannels"] = sorted((ranks.get(key) or {}).keys())
        fused.append(item.model_copy(update={"fusion_score": round(normalized_score, 6), "metadata": metadata}))
    fused = sorted(fused, key=recall_item_sort_key, reverse=True)
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


FOLLOW_UP_QUERY_RE = re.compile(r"^(那|那么|再|然后|还有|这个|这些|它|它们|同样|按|改成|换成|呢|继续)")


def rewrite_retrieval_query(request: KnowledgeRetrievalRequest) -> str:
    """Turn a context-dependent follow-up into a standalone retrieval query.

    This is intentionally deterministic: it only inherits the previous user
    question when the current turn contains an explicit follow-up signal.
    """
    current = re.sub(r"\s+", " ", str(request.query or "")).strip()
    previous = re.sub(r"\s+", " ", str(request.previous_user_question or "")).strip()
    if not current or not previous or current == previous:
        return current
    follow_up = bool(FOLLOW_UP_QUERY_RE.search(current)) or any(
        token in current for token in ["按天看", "按周看", "细分一下", "继续下钻", "换个维度"]
    )
    if not follow_up:
        return current
    return "%s；追问补充：%s" % (previous[:600], current[:300])


def filter_recall_items_by_governance(
    items: list[RecallItem],
    request: KnowledgeRetrievalRequest,
) -> tuple[list[RecallItem], dict[str, int]]:
    kept: list[RecallItem] = []
    filtered: dict[str, int] = {}
    for item in items or []:
        reason = recall_governance_block_reason(item, request)
        if reason:
            filtered[reason] = filtered.get(reason, 0) + 1
            continue
        kept.append(item)
    return kept, filtered


def recall_governance_metadata(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    mappings = {
        "status": payload.get("status") or payload.get("lifecycleStatus"),
        "version": payload.get("version") or payload.get("semanticVersion"),
        "activeVersion": payload.get("activeVersion") or payload.get("currentVersion"),
        "merchantId": payload.get("merchantId"),
        "merchantIds": payload.get("merchantIds") or payload.get("allowedMerchantIds"),
        "allowedRoles": payload.get("allowedRoles"),
        "requiredPermissions": payload.get("requiredPermissions"),
        "visibilityPolicy": payload.get("visibilityPolicy"),
        "expiresAt": payload.get("expiresAt") or payload.get("expiryAt"),
        "confidence": payload.get("confidence") or payload.get("knowledgeConfidence"),
    }
    return {
        key: value
        for key, value in mappings.items()
        if value is not None and value != "" and value != () and value != [] and value != {}
    }


def recall_governance_block_reason(item: RecallItem, request: KnowledgeRetrievalRequest) -> str:
    metadata = dict(item.metadata or {})
    status = str(
        metadata.get("lifecycleStatus")
        or metadata.get("publishStatus")
        or metadata.get("status")
        or ""
    ).strip().lower()
    blocked_statuses = {
        "pending",
        "pending_review",
        "draft",
        "rejected",
        "disabled",
        "inactive",
        "expired",
        "rolled_back",
        "deleted",
        "archived",
        "blocked",
    }
    if status in blocked_statuses:
        return "status"
    if str(metadata.get("assetStatus") or "").strip().lower() in blocked_statuses:
        return "status"

    expires_at = metadata.get("expiresAt") or metadata.get("expiryAt")
    if expires_at and timestamp_is_past(expires_at):
        return "expired"
    if metadata.get("assetExpiresAt") and timestamp_is_past(metadata.get("assetExpiresAt")):
        return "expired"

    active_version = str(metadata.get("activeVersion") or metadata.get("currentVersion") or "").strip()
    item_version = str(metadata.get("semanticVersion") or metadata.get("version") or "").strip()
    if active_version and item_version and active_version != item_version:
        return "version"

    merchant_id = str(request.merchant_id or "").strip()
    scoped_merchants = metadata.get("merchantIds") or metadata.get("allowedMerchantIds") or []
    if isinstance(scoped_merchants, str):
        scoped_merchants = [scoped_merchants]
    item_merchant = str(metadata.get("merchantId") or "").strip()
    if item_merchant and item_merchant not in {"*", "global", merchant_id}:
        return "merchant"
    asset_merchant = str(metadata.get("assetMerchantId") or "").strip()
    if asset_merchant and asset_merchant not in {"*", "global", merchant_id}:
        return "merchant"
    if scoped_merchants and merchant_id not in {str(value) for value in scoped_merchants} and "*" not in scoped_merchants:
        return "merchant"

    visibility = metadata.get("visibilityPolicy") if isinstance(metadata.get("visibilityPolicy"), dict) else {}
    allowed_roles = metadata.get("allowedRoles") or visibility.get("allowedRoles") or []
    if isinstance(allowed_roles, str):
        allowed_roles = [allowed_roles]
    role = str(request.access_role or "merchant_operator").strip().lower()
    normalized_roles = {str(value).strip().lower() for value in allowed_roles if str(value).strip()}
    if normalized_roles and role not in normalized_roles and role not in {"merchant_admin", "admin"}:
        return "role"
    asset_roles = metadata.get("assetAllowedRoles") or []
    if isinstance(asset_roles, str):
        asset_roles = [asset_roles]
    normalized_asset_roles = {str(value).strip().lower() for value in asset_roles if str(value).strip()}
    if normalized_asset_roles and role not in normalized_asset_roles and role not in {"merchant_admin", "admin"}:
        return "role"
    visibility_level = str(visibility.get("level") or metadata.get("visibility") or "").strip().lower()
    if visibility_level == "restricted" and not normalized_roles and role not in {"merchant_admin", "admin"}:
        return "role"
    asset_visibility = metadata.get("assetVisibilityPolicy") if isinstance(metadata.get("assetVisibilityPolicy"), dict) else {}
    if str(asset_visibility.get("level") or "").strip().lower() == "restricted" and not normalized_asset_roles and role not in {"merchant_admin", "admin"}:
        return "role"

    required_permissions = metadata.get("requiredPermissions") or []
    if isinstance(required_permissions, str):
        required_permissions = [required_permissions]
    granted = {str(value).strip() for value in request.permissions if str(value).strip()}
    if required_permissions and not set(map(str, required_permissions)).issubset(granted):
        return "permission"
    asset_permissions = metadata.get("assetRequiredPermissions") or []
    if isinstance(asset_permissions, str):
        asset_permissions = [asset_permissions]
    if asset_permissions and not set(map(str, asset_permissions)).issubset(granted):
        return "permission"
    return ""


def timestamp_is_past(value: object) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= datetime.now(timezone.utc)


def business_rerank_recall_items(
    items: list[RecallItem],
    query_text: str,
    request: KnowledgeRetrievalRequest,
) -> list[RecallItem]:
    query = str(query_text or "").lower()
    intent = str(request.intent_kind or "").lower()
    reranked: list[RecallItem] = []
    fallback_ranks = {
        id(item): rank
        for rank, item in enumerate(sorted(items or [], key=lambda value: float(value.fusion_score or 0.0), reverse=True), start=1)
    }
    for item in items or []:
        metadata = dict(item.metadata or {})
        source_type = str(item.source_type or "").upper()
        business_score = 0.0
        reasons: list[str] = []
        aliases = metadata.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        labels = [
            metadata.get("businessName"),
            metadata.get("metricKey"),
            *aliases,
        ]
        if any(str(label).strip().lower() in query for label in labels if len(str(label).strip()) >= 2):
            business_score += 0.4
            reasons.append("exact_business_label")
        if item.topic and str(item.topic).lower() in query:
            business_score += 0.1
            reasons.append("topic_match")
        if source_type == "SEMANTIC_METRIC" and intent in {"metric_query", "multi_metric", "analysis"}:
            business_score += 0.3
            reasons.append("metric_intent")
        if source_type == "SEMANTIC_RELATIONSHIP" and intent in {"multi_hop", "analysis", "rule_data_mix"}:
            business_score += 0.3
            reasons.append("relationship_intent")
        if source_type == "GOVERNED_RULE" and intent in {"rule_only", "rule_data_mix"}:
            business_score += 0.3
            reasons.append("rule_intent")
        confidence = metadata.get("confidence") or metadata.get("knowledgeConfidence")
        if isinstance(confidence, (int, float)):
            business_score += max(0.0, min(float(confidence), 1.0)) * 0.2
            reasons.append("confidence")
        business_score = max(0.0, min(1.0, business_score))
        retrieval_score = recall_item_retrieval_score(item, fallback_rank=fallback_ranks.get(id(item), 1))
        protection_tier, protection_reasons = metric_protection_tier(metadata, source_type, intent)
        final_score = max(0.0, min(1.0, retrieval_score * 0.75 + business_score * 0.25))
        metadata["scoreVersion"] = "recall_v2"
        metadata["retrievalScore"] = round(retrieval_score, 6)
        metadata["businessScore"] = round(business_score, 6)
        metadata["retrievalWeightedScore"] = round(retrieval_score * 0.75, 6)
        metadata["businessWeightedScore"] = round(business_score * 0.25, 6)
        metadata["finalScore"] = round(final_score, 6)
        metadata["protectionTier"] = protection_tier
        metadata["protectionReasons"] = protection_reasons
        metadata["businessRerankBoost"] = round(business_score * 0.25, 6)
        metadata["businessRerankReasons"] = reasons
        reranked.append(item.model_copy(update={"fusion_score": round(final_score, 6), "metadata": metadata}))
    return sorted(reranked, key=recall_item_sort_key, reverse=True)


def recall_item_retrieval_score(item: RecallItem, fallback_rank: int = 1) -> float:
    metadata = dict(item.metadata or {})
    for key in ["retrievalScore", "rrfNormalizedScore", "metricResolverScore"]:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    raw_score = float(item.fusion_score or 0.0)
    if 0.0 <= raw_score <= 1.0:
        return raw_score
    rank = max(1, int(fallback_rank or 1))
    return round(61.0 / float(60 + rank), 6)


def metric_protection_tier(metadata: dict[str, Any], source_type: str, intent: str) -> tuple[int, list[str]]:
    if source_type != "SEMANTIC_METRIC":
        return 0, []
    resolution_type = str(metadata.get("metricResolutionType") or "")
    confidence = max(0.0, min(1.0, float(metadata.get("metricResolutionConfidence") or 0.0)))
    ambiguous = bool(metadata.get("metricResolutionAmbiguous") or False)
    metric_intent = intent in {"metric_query", "multi_metric", "analysis"}
    exact = resolution_type in {"exact_business_name", "exact_alias", "exact_term", "exact_metric_key"}
    if exact and confidence >= 0.95 and not ambiguous and metric_intent:
        return 2, ["exact_metric", "high_confidence", "unambiguous", "metric_intent"]
    if confidence >= 0.8 and not ambiguous:
        reasons = ["metric_candidate", "high_confidence"]
        if exact:
            reasons.append("exact_metric")
        return 1, reasons
    return 0, []


def recall_item_sort_key(item: RecallItem) -> tuple[int, float, float]:
    metadata = dict(item.metadata or {})
    return (
        int(metadata.get("protectionTier") or 0),
        float(metadata.get("finalScore") if metadata.get("finalScore") is not None else item.fusion_score or 0.0),
        float(metadata.get("retrievalScore") or 0.0),
    )


def retrieval_query_text(request: KnowledgeRetrievalRequest, rewritten_query: str = "") -> str:
    parts = [rewritten_query or request.query]
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
    if not label:
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


def topic_categories_support_knowledge_capability(
    topic_assets: TopicAssetService,
    categories: list[QuestionCategory],
    capability: str,
) -> bool:
    """Resolve retrieval lanes from published topic roles, never topic IDs."""

    expected = re.sub(r"[^A-Z0-9]+", "_", str(capability or "").upper()).strip("_")
    if not expected:
        return False
    for topic in topic_assets.topic_names_for_categories(categories):
        contract = topic_assets.load_topic_contract(topic)
        metadata = contract.get("metadata") if isinstance(contract.get("metadata"), dict) else {}
        declared_values: list[Any] = []
        for source in (contract, metadata):
            for key in (
                "capabilities",
                "knowledgeCapabilities",
                "knowledgeCapability",
                "knowledgeRoles",
                "knowledgeRole",
                "retrievalCapabilities",
                "routingRole",
                "topicRole",
            ):
                value = source.get(key)
                declared_values.extend(value if isinstance(value, list) else [value])
        for value in declared_values:
            normalized = re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")
            tokens = [token for token in normalized.split("_") if token]
            if normalized == expected or expected in tokens:
                return True
    return False


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
