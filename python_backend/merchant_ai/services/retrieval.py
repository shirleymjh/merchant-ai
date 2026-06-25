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
from merchant_ai.services.assets import HybridRecallService, TopicAssetService


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


class EsKnowledgeRetrievalService:
    """Elasticsearch-backed knowledge retrieval adapter.

    The rest of the harness still consumes KnowledgeBundle/RecallItem, so ES is
    a backend choice, not a second recall path.
    """

    backend_name = "es"

    def __init__(self, settings: Settings, topic_assets: TopicAssetService):
        self.settings = settings
        self.topic_assets = topic_assets

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        request_key = request.knowledge_request.request_key if request.knowledge_request else ""
        query_text = retrieval_query_text(request)
        normalized_categories = [category for category in [normalize_question_category(item) for item in request.topic_categories] if category]
        topics = self._allowed_topics(normalized_categories)
        include_base_wiki = QuestionCategory.PLATFORM_RULE in set(normalized_categories) or route_is_rule_sensitive(request)
        try:
            items = self._search(query_text, topics, include_base_wiki=include_base_wiki)
            if topics and not request.knowledge_request:
                try:
                    items = merge_recall_items(items, self._search(query_text, [], include_base_wiki=False))
                except Exception:
                    pass
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
        return KnowledgeBundle(
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
        if not self.settings.es_base_url:
            raise RuntimeError("ES_BASE_URL_MISSING")
        size = 12 if topics else 4
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
        query: dict[str, object]
        if must or filters:
            query = {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}
        else:
            query = {"match_all": {}}
        response = requests.post(
            "%s/%s/_search" % (self.settings.es_base_url.rstrip("/"), self.settings.es_index),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "query": query},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        return [es_hit_to_recall_item(hit, query_text) for hit in hits]

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


def es_hit_to_recall_item(hit: dict[str, object], query_text: str) -> RecallItem:
    source = hit.get("_source") if isinstance(hit, dict) else {}
    source = source if isinstance(source, dict) else {}
    metadata = dict(source.get("metadata") or {})
    semantic_ref_id = str(source.get("semantic_ref_id") or metadata.get("semanticRefId") or source.get("doc_id") or hit.get("_id") or "")
    semantic_path = str(source.get("semantic_path") or metadata.get("semanticPath") or "")
    metadata["semanticRefId"] = semantic_ref_id
    if semantic_path:
        metadata["semanticPath"] = semantic_path
    metadata["recallQuery"] = query_text
    metadata["recallQueries"] = [query_text] if query_text else []
    metadata["esScore"] = float(hit.get("_score") or 0.0)
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
            "sourcePath": str((item.metadata or {}).get("sourcePath") or ""),
        }
        for item in items
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] if records else ""
