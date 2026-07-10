from __future__ import annotations

import json
import math
import os
import re
import hashlib
import pickle
import threading
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.models import (
    KnowledgeSuggestion,
    KnowledgeSuggestionReviewRequest,
    MemoryConflictResolution,
    MemoryEvent,
    MemoryFact,
    MemoryInjectionTrace,
    MemoryPreference,
    MemoryRetrievalCandidate,
    PendingAnswer,
)
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key


MEMORY_SCHEMA_VERSION = "merchant_memory.v2"
MAX_EVENTS = 240
MAX_PREFERENCES = 160
MAX_FACTS = 120
MAX_KNOWLEDGE_SUGGESTIONS = 160
HABIT_CORE_PROMOTION_HIT_COUNT = 2
APPROVED_MEMORY_STATUSES = {"", "active", "approved", "reviewed", "published", "indexed"}
PENDING_MEMORY_STATUSES = {"candidate", "pending", "review_required", "needs_review"}
INACTIVE_MEMORY_STATUSES = {"deleted", "disabled", "inactive", "rejected", "archived", "expired"}
STRONG_CONSTRAINT_STATUSES = {"", "active", "approved", "reviewed", "published", "indexed"}
EXPLICIT_HABIT_TERMS = {
    "以后",
    "后续",
    "默认",
    "固定",
    "优先",
    "每次",
    "一直",
    "长期",
    "常用",
    "习惯",
    "记住",
    "以后都",
    "默认按",
    "优先看",
    "不用",
    "不要只",
}


class MemoryStore:
    """Adapter boundary for long-term memory storage."""

    def load(self, merchant_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, merchant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def select_for_question(self, state: AgentState, budget_tokens: int = 0, budget_chars: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

    def update_from_state(self, state: AgentState) -> Dict[str, Any]:
        raise NotImplementedError


class MemoryIngestionService:
    """Owns long-term memory write rules and normalization side effects."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def update_store(self, store: MemoryStore, state: AgentState) -> Dict[str, Any]:
        merchant_id = merchant_id_from_state(state, self.settings)
        memory = store.load(merchant_id)
        event = memory_event_from_state(state)
        if not event.get("question"):
            return memory
        ingestion_trace = {"eventId": event.get("eventId"), "memoryType": event.get("memoryType"), "written": False, "preferenceUpdates": 0}
        events = [item for item in memory.get("events", []) if isinstance(item, dict)]
        past_case = past_case_event_from_state(state)
        if past_case and not any(event_fingerprint(item) == event_fingerprint(past_case) for item in events[-8:]):
            events.append(past_case)
            ingestion_trace["pastCaseWritten"] = True
        procedure = procedure_event_from_state(state)
        if procedure and not any(event_fingerprint(item) == event_fingerprint(procedure) for item in events[-8:]):
            events.append(procedure)
            ingestion_trace["procedureWritten"] = True
        if not events or event_fingerprint(events[-1]) != event_fingerprint(event):
            events.append(event)
            ingestion_trace["written"] = True
        memory["events"] = events[-MAX_EVENTS:]
        preference_updates = 0 if event.get("memoryType") == "metric_dispute" else upsert_habit_preferences(memory, event)
        ingestion_trace["preferenceUpdates"] = preference_updates
        if event.get("memoryType") == "correction":
            fact = correction_fact_from_event(event)
            memory["facts"] = upsert_fact(memory.get("facts") or [], fact)
            conflict = resolve_memory_conflicts(memory, event)
            if conflict:
                ingestion_trace["conflict"] = conflict.model_dump(by_alias=True)
        suggestion = knowledge_suggestion_from_event(event)
        if suggestion:
            memory["knowledgeSuggestions"], suggestion_written = upsert_knowledge_suggestion(
                memory.get("knowledgeSuggestions") or [],
                suggestion,
            )
            ingestion_trace["knowledgeSuggestionWritten"] = suggestion_written
            ingestion_trace["knowledgeSuggestionId"] = suggestion.get("suggestionId", "")
            ingestion_trace["knowledgeSuggestionCount"] = len(memory.get("knowledgeSuggestions") or [])
        memory["recentFocus"] = aggregate_recent_focus(memory.get("events") or [], memory.get("preferences") or [])
        memory["memoryIngestionTrace"] = ingestion_trace
        saved = store.save(merchant_id, memory)
        saved["memoryIngestionTrace"] = ingestion_trace
        return saved

    def update_feedback(self, store: MemoryStore, pending: Optional[PendingAnswer], adopted: Any = None, liked: Any = None, disliked: Any = None) -> Dict[str, Any]:
        if not pending:
            return {}
        merchant_id = pending.merchant_id or self.settings.merchant_id
        memory = store.load(merchant_id)
        event = memory_event_from_feedback(pending, adopted=adopted, liked=liked, disliked=disliked)
        if not event.get("question"):
            return memory
        events = [item for item in memory.get("events", []) if isinstance(item, dict)]
        if not events or event_fingerprint(events[-1]) != event_fingerprint(event):
            events.append(event)
        memory["events"] = events[-MAX_EVENTS:]
        if bool(disliked):
            reduce_related_memory(memory, event, reason="negative feedback")
        else:
            upsert_habit_preferences(memory, event)
        memory["recentFocus"] = aggregate_recent_focus(memory.get("events") or [], memory.get("preferences") or [])
        memory["memoryIngestionTrace"] = {
            "eventId": event.get("eventId"),
            "memoryType": event.get("memoryType"),
            "feedbackSignal": event.get("feedbackSignal", ""),
            "written": True,
        }
        return store.save(merchant_id, memory)


class MemoryRetrievalService:
    """Owns runtime memory recall, ranking, and scoped injection assembly."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def select_from_store(self, store: MemoryStore, state: AgentState, budget_tokens: int = 0, budget_chars: int = 0) -> Dict[str, Any]:
        merchant_id = merchant_id_from_state(state, self.settings)
        budget = memory_budget_tokens(self.settings, budget_tokens=budget_tokens, budget_chars=budget_chars)
        context = retrieval_context_from_state(state)
        if isinstance(store, EnterpriseMemoryStore):
            return self._select_enterprise(store, merchant_id, context, budget)
        memory = store.load(merchant_id)
        candidates, filtered_reasons = rank_memory_candidates(memory, context)
        selected, trace = allocate_injection(memory, candidates, filtered_reasons, merchant_id, budget, str(store.memory_path(merchant_id)))
        selected["source"] = str(store.memory_path(merchant_id))
        selected["updatedAt"] = memory.get("updatedAt", "")
        selected["memoryInjectionTrace"] = trace.model_dump(by_alias=True)
        if trace.selected_ids:
            record_memory_usage(memory, trace.selected_ids)
            store.save(merchant_id, memory)
        return selected

    def _select_enterprise(self, store: "EnterpriseMemoryStore", merchant_id: str, context: Dict[str, Any], budget: int) -> Dict[str, Any]:
        query_hash = memory_query_hash(merchant_id, context)
        injection_key = "memory_injection:%s:%s" % (merchant_id, query_hash)
        cached = store.hot_cache.get_json(injection_key)
        if isinstance(cached, dict):
            cached = dict(cached)
            for memory_id in memory_ids_from_selected(cached):
                store.hot_cache.increment_hit_delta(memory_id, merchant_id)
            trace = dict(cached.get("memoryInjectionTrace") or {})
            trace["cacheHit"] = True
            trace["cacheKey"] = injection_key
            trace["cacheBackend"] = store.hot_cache.backend_name()
            cached["memoryInjectionTrace"] = trace
            return cached

        memory = store.load(merchant_id)
        vector_ids = store._vector_candidate_ids(merchant_id, context)
        vector_loaded_count = 0
        if vector_ids:
            vector_memory = store._load_vector_memory_items(merchant_id, memory, vector_ids)
            vector_loaded_count = count_memory_items(vector_memory)
            if vector_loaded_count:
                memory = merge_memory_payload(memory, vector_memory)
        candidates, filtered_reasons = rank_memory_candidates(memory, context)
        if vector_ids:
            candidates = boost_vector_candidates(candidates, vector_ids)
            filtered_reasons["vector_candidates"] = len(vector_ids)
            filtered_reasons["vector_loaded"] = vector_loaded_count
            store.hot_cache.set_json(
                "memory_candidates:%s:%s" % (merchant_id, query_hash),
                {
                    "merchantId": merchant_id,
                    "queryHash": query_hash,
                    "vectorIds": vector_ids[:24],
                    "vectorLoadedCount": vector_loaded_count,
                },
            )
        source = memory_source_label(memory, self.settings)
        selected, trace = allocate_injection(memory, candidates, filtered_reasons, merchant_id, budget, source)
        selected["source"] = source
        selected["updatedAt"] = memory.get("updatedAt", "")
        trace_payload = trace.model_dump(by_alias=True)
        trace_payload["cacheHit"] = False
        trace_payload["cacheKey"] = injection_key
        trace_payload["cacheBackend"] = store.hot_cache.backend_name()
        trace_payload["vectorCandidateCount"] = len(vector_ids)
        trace_payload["vectorLoadedCount"] = vector_loaded_count
        selected["memoryInjectionTrace"] = trace_payload
        if trace.selected_ids:
            for memory_id in trace.selected_ids:
                store.hot_cache.increment_hit_delta(memory_id, merchant_id)
            record_memory_usage(memory, trace.selected_ids)
            store.save(merchant_id, memory)
        store.hot_cache.set_json(injection_key, selected)
        return selected


class StructuredMemoryStore(MemoryStore):
    """Governed local long-term memory for merchant BI context."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingestion_service = MemoryIngestionService(settings)
        self.retrieval_service = MemoryRetrievalService(settings)

    def memory_path(self, merchant_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", merchant_id or self.settings.merchant_id or "default")
        return self.settings.resolved_workspace_path / "memory" / ("%s.memory.json" % safe_id)

    def load(self, merchant_id: str) -> Dict[str, Any]:
        path = self.memory_path(merchant_id)
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return normalize_memory(payload, merchant_id or self.settings.merchant_id)
        except Exception:
            return self.empty_memory(merchant_id)
        return self.empty_memory(merchant_id)

    def save(self, merchant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = normalize_memory(payload, merchant_id or self.settings.merchant_id)
        payload["updatedAt"] = datetime.now().isoformat()
        payload["merchantId"] = merchant_id or self.settings.merchant_id
        path = self.memory_path(merchant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        return payload

    def select_for_question(self, state: AgentState, budget_tokens: int = 0, budget_chars: int = 0) -> Dict[str, Any]:
        return self.retrieval_service.select_from_store(self, state, budget_tokens=budget_tokens, budget_chars=budget_chars)

    def render_injection(self, payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        renderable = {
            "merchantId": payload.get("merchantId", ""),
            "recentFocus": payload.get("recentFocus", {}),
            "coreMemory": payload.get("coreMemory", {}),
            "relevantCorrections": payload.get("relevantCorrections", []),
            "relevantMetricDisputes": payload.get("relevantMetricDisputes", []),
            "relevantPreferences": payload.get("relevantPreferences", []),
            "relevantFacts": payload.get("relevantFacts", []),
            "relevantEvents": payload.get("relevantEvents", []),
            "truncated": bool(payload.get("truncated")),
        }
        if not any(
            renderable.get(key)
            for key in [
                "recentFocus",
                "relevantCorrections",
                "relevantMetricDisputes",
                "relevantPreferences",
                "relevantFacts",
                "relevantEvents",
            ]
        ):
            return ""
        return json.dumps(renderable, ensure_ascii=False, default=str, indent=2)

    def update_from_state(self, state: AgentState) -> Dict[str, Any]:
        return self.ingestion_service.update_store(self, state)

    def update_from_feedback(self, pending: Optional[PendingAnswer], adopted: Any = None, liked: Any = None, disliked: Any = None) -> Dict[str, Any]:
        return self.ingestion_service.update_feedback(self, pending, adopted=adopted, liked=liked, disliked=disliked)

    def empty_memory(self, merchant_id: str) -> Dict[str, Any]:
        return {
            "merchantId": merchant_id or self.settings.merchant_id,
            "recentFocus": {},
            "coreMemoryProfile": {},
            "preferences": [],
            "facts": [],
            "events": [],
            "conflicts": [],
            "knowledgeSuggestions": [],
            "memoryIngestionTrace": {},
            "updatedAt": "",
            "schemaVersion": MEMORY_SCHEMA_VERSION,
        }


class EnterpriseMemoryStore(StructuredMemoryStore):
    """Enterprise long-term memory store.

    ES is the authoritative store, Redis is an optional short TTL hot cache,
    vector search reads from the same ES index, and the existing JSON file
    store remains the local/fallback path.
    """

    def __init__(
        self,
        settings: Settings,
        repository: Optional[Any] = None,
        hot_cache: Optional[Any] = None,
        vector_index: Optional[Any] = None,
        fallback_store: Optional[StructuredMemoryStore] = None,
    ):
        super().__init__(settings)
        self.repository = repository or MemoryEsRepository(settings)
        self.hot_cache = hot_cache or MemoryHotCache(settings)
        self.vector_index = vector_index or MemoryVectorIndex(settings)
        self.fallback_store = fallback_store or StructuredMemoryStore(settings)

    def load(self, merchant_id: str) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        try:
            payload = self.repository.load_memory(target)
            payload = normalize_memory(payload, target)
            payload["storageBackend"] = "es"
            return payload
        except Exception as exc:
            if self._fallback_enabled():
                payload = self.fallback_store.load(target)
                payload["storageBackend"] = "json_fallback"
                payload["memoryStorageError"] = str(exc)[:240]
                return payload
            payload = self.empty_memory(target)
            payload["storageBackend"] = "es_unavailable"
            payload["memoryStorageError"] = str(exc)[:240]
            return payload

    def save(self, merchant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        payload = normalize_memory(payload, target)
        payload["updatedAt"] = datetime.now().isoformat()
        try:
            saved = self.repository.save_memory(target, payload)
            saved = normalize_memory(saved, target)
            saved["storageBackend"] = "es"
            self._refresh_hot_cache(target, saved)
            return saved
        except Exception as exc:
            if self._fallback_enabled():
                saved = self.fallback_store.save(target, payload)
                saved["storageBackend"] = "json_fallback"
                saved["memoryStorageError"] = str(exc)[:240]
                self._refresh_hot_cache(target, saved)
                return saved
            payload["storageBackend"] = "es_unavailable"
            payload["memoryStorageError"] = str(exc)[:240]
            return payload

    def select_for_question(self, state: AgentState, budget_tokens: int = 0, budget_chars: int = 0) -> Dict[str, Any]:
        return self.retrieval_service.select_from_store(self, state, budget_tokens=budget_tokens, budget_chars=budget_chars)

    def flush_hit_deltas(self) -> Dict[str, Any]:
        deltas = self.hot_cache.drain_hit_deltas()
        if not deltas:
            return {"flushed": 0}
        try:
            flushed = self.repository.apply_hit_deltas(deltas)
            return {"flushed": flushed}
        except Exception as exc:
            return {"flushed": 0, "error": str(exc)[:240]}

    def _vector_candidate_ids(self, merchant_id: str, context: Dict[str, Any]) -> List[str]:
        query_text = memory_vector_query_text(context)
        if not query_text:
            return []
        try:
            return self.vector_index.search(merchant_id, query_text)
        except Exception:
            return []

    def _load_vector_memory_items(self, merchant_id: str, memory: Dict[str, Any], vector_ids: List[str]) -> Dict[str, Any]:
        existing_ids = memory_id_set(memory)
        missing_ids = [memory_id for memory_id in unique_strings(vector_ids) if memory_id not in existing_ids]
        if not missing_ids:
            return empty_memory_payload(merchant_id)
        try:
            return self.repository.load_memory_items(merchant_id, missing_ids)
        except Exception:
            if self._fallback_enabled():
                fallback_memory = self.fallback_store.load(merchant_id)
                return scan_memory_items_by_ids(fallback_memory, missing_ids, merchant_id)
            return empty_memory_payload(merchant_id)

    def _refresh_hot_cache(self, merchant_id: str, memory: Dict[str, Any]) -> None:
        self.hot_cache.invalidate_merchant(merchant_id)
        self.hot_cache.set_json("recent_focus:%s" % merchant_id, memory.get("recentFocus") or {})
        self.hot_cache.set_json(
            "top_preferences:%s" % merchant_id,
            sorted(memory.get("preferences") or [], key=weighted_memory_value, reverse=True)[:20],
        )

    def _hybrid_enabled(self) -> bool:
        return str(getattr(self.settings, "memory_backend", "file") or "file").strip().lower() == "hybrid"

    def _fallback_enabled(self) -> bool:
        backend = str(getattr(self.settings, "memory_backend", "file") or "file").strip().lower()
        return backend in {"es", "hybrid", "mysql"}


class MemoryEsRepository:
    """ES persistence for governed merchant long-term memory."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._index_ready = False
        self._unavailable_until = 0.0
        self._vector_helper = MemoryVectorIndex(settings)

    def load_memory(self, merchant_id: str) -> Dict[str, Any]:
        self._ensure_ready()
        hits = self._search(
            {
                "size": 1024,
                "sort": [{"updated_at": {"order": "asc", "missing": "_last"}}],
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"merchant_id": merchant_id}},
                            {
                                "terms": {
                                    "doc_type": ["memory_item", "memory_conflict", "knowledge_suggestion", "memory_profile"]
                                }
                            },
                        ]
                    }
                },
            }
        )
        payload = empty_memory_payload(merchant_id)
        latest_updated_at = ""
        for hit in hits:
            source = hit.get("_source") or {}
            doc_type = str(source.get("doc_type") or "")
            status = str(source.get("status") or "")
            latest_updated_at = max(latest_updated_at, str(source.get("updated_at") or ""))
            if status == "deleted":
                continue
            if doc_type == "memory_item":
                group = str(source.get("group") or "")
                item = memory_item_from_es_source(source)
                if group == "event":
                    payload["events"].append(item)
                elif group == "preference":
                    payload["preferences"].append(item)
                elif group == "fact":
                    payload["facts"].append(item)
            elif doc_type == "memory_conflict":
                conflict = conflict_from_es_source(source)
                if conflict:
                    payload["conflicts"].append(conflict)
            elif doc_type == "knowledge_suggestion":
                suggestion = knowledge_suggestion_from_es_source(source)
                if suggestion:
                    payload["knowledgeSuggestions"].append(suggestion)
            elif doc_type == "memory_profile":
                payload["recentFocus"] = source.get("recent_focus") if isinstance(source.get("recent_focus"), dict) else {}
                payload["coreMemoryProfile"] = source.get("core_memory_profile") if isinstance(source.get("core_memory_profile"), dict) else {}
                if source.get("updated_at"):
                    payload["updatedAt"] = str(source.get("updated_at") or "")
        payload["updatedAt"] = payload.get("updatedAt") or latest_updated_at
        return normalize_memory(payload, merchant_id)

    def load_memory_items(self, merchant_id: str, memory_ids: List[str]) -> Dict[str, Any]:
        self._ensure_ready()
        ids = unique_strings(memory_ids)[: max(1, int(self.settings.es_vector_top_k or 12)) * 2]
        if not ids:
            return empty_memory_payload(merchant_id)
        hits = self._search(
            {
                "size": len(ids) * 3 + 8,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"merchant_id": merchant_id}},
                            {"term": {"doc_type": "memory_item"}},
                            {"terms": {"memory_id": ids}},
                        ]
                    }
                },
            }
        )
        order = {memory_id: index for index, memory_id in enumerate(ids)}
        payload = empty_memory_payload(merchant_id)
        items: List[Tuple[int, str, Dict[str, Any]]] = []
        for hit in hits:
            source = hit.get("_source") or {}
            if str(source.get("status") or "") == "deleted":
                continue
            group = str(source.get("group") or "")
            item = memory_item_from_es_source(source)
            memory_id = str(source.get("memory_id") or "")
            items.append((order.get(memory_id, len(order)), group, item))
        items.sort(key=lambda item: (item[0], item[1]))
        for _, group, item in items:
            if group == "event":
                payload["events"].append(item)
            elif group == "preference":
                payload["preferences"].append(item)
            elif group == "fact":
                payload["facts"].append(item)
        return normalize_memory(payload, merchant_id)

    def save_memory(self, merchant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_ready()
        memory = normalize_memory(payload, merchant_id)
        vector_async = self._vector_index_async_enabled()
        docs = memory_es_documents(memory, self.settings, self._vector_helper if self._vector_enabled() and not vector_async else None)
        expected_ids = {str(doc.get("doc_id") or "") for doc in docs if doc.get("doc_id")}
        existing_ids = set(self._existing_doc_ids(merchant_id))
        obsolete_ids = sorted(existing_ids - expected_ids)
        lines: List[str] = []
        for doc in docs:
            doc_id = str(doc.pop("doc_id") or "")
            if not doc_id:
                continue
            lines.append(json.dumps({"index": {"_index": self.index_name(), "_id": doc_id}}, ensure_ascii=False))
            lines.append(json.dumps(doc, ensure_ascii=False, default=str))
        for doc_id in obsolete_ids:
            lines.append(json.dumps({"delete": {"_index": self.index_name(), "_id": doc_id}}, ensure_ascii=False))
        self._bulk(lines)
        if vector_async:
            self._schedule_vector_index(merchant_id, memory)
        return memory

    def sync_vector_index(self, merchant_id: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        if not self._vector_enabled():
            return {"success": True, "enabled": False, "updated": 0}
        self._ensure_ready()
        normalized = normalize_memory(memory, merchant_id)
        lines = memory_es_vector_update_lines(normalized, self.settings, self._vector_helper, self.index_name())
        self._bulk(lines)
        return {"success": True, "enabled": True, "updated": int(len(lines) / 2)}

    def _schedule_vector_index(self, merchant_id: str, memory: Dict[str, Any]) -> None:
        snapshot = deepcopy(memory)
        thread = threading.Thread(
            target=self._sync_vector_index_safely,
            args=(merchant_id, snapshot),
            name="memory-vector-index-%s" % stable_slug(merchant_id)[:40],
            daemon=True,
        )
        thread.start()

    def _sync_vector_index_safely(self, merchant_id: str, memory: Dict[str, Any]) -> None:
        try:
            self.sync_vector_index(merchant_id, memory)
        except Exception:
            return

    def apply_hit_deltas(self, deltas: Dict[str, Dict[str, Any]]) -> int:
        self._ensure_ready()
        lines: List[str] = []
        updated = 0
        for memory_id, delta in (deltas or {}).items():
            memory_id = str(memory_id or "")
            if not memory_id:
                continue
            hits = max(1, int((delta or {}).get("hitCount") or 1))
            merchant_id = str((delta or {}).get("merchantId") or "")
            search_filter: List[Dict[str, Any]] = [
                {"term": {"doc_type": "memory_item"}},
                {"term": {"memory_id": memory_id}},
            ]
            if merchant_id:
                search_filter.append({"term": {"merchant_id": merchant_id}})
            matched = self._search({"size": 24, "query": {"bool": {"filter": search_filter}}})
            for hit in matched:
                source = hit.get("_source") or {}
                doc_id = str(hit.get("_id") or source.get("doc_id") or "")
                if not doc_id:
                    continue
                current_hit_count = int(source.get("hit_count") or 0)
                lines.append(json.dumps({"update": {"_index": self.index_name(), "_id": doc_id}}, ensure_ascii=False))
                lines.append(
                    json.dumps(
                        {
                            "doc": {
                                "hit_count": current_hit_count + hits,
                                "last_used_at": str((delta or {}).get("lastUsedAt") or datetime.now().isoformat()),
                                "decay_score": float((delta or {}).get("decayScore") or 1.0),
                                "updated_at": datetime.now().isoformat(),
                            }
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                )
                updated += 1
        self._bulk(lines)
        return updated

    def index_name(self) -> str:
        return str(getattr(self.settings, "memory_es_index", "") or getattr(self.settings, "memory_vector_index", "") or "merchant_memory")

    def _existing_doc_ids(self, merchant_id: str) -> List[str]:
        hits = self._search(
            {
                "size": 1024,
                "_source": ["doc_id"],
                "query": {"bool": {"filter": [{"term": {"merchant_id": merchant_id}}]}},
            }
        )
        doc_ids: List[str] = []
        for hit in hits:
            doc_id = str(((hit.get("_source") or {}).get("doc_id") or hit.get("_id") or ""))
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        return doc_ids

    def _search(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = requests.post(
            self._url("%s/_search" % self.index_name()),
            headers=self._headers(),
            auth=self._auth(),
            json=body,
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json() or {}
        return ((payload.get("hits") or {}).get("hits") or [])

    def _bulk(self, lines: List[str]) -> None:
        if not lines:
            return
        response = requests.post(
            self._url("_bulk"),
            headers={**self._headers(), "Content-Type": "application/x-ndjson"},
            auth=self._auth(),
            data=("\n".join(lines) + "\n").encode("utf-8"),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json() or {}
        if payload.get("errors"):
            raise RuntimeError("memory es bulk write failed")

    def _ensure_ready(self) -> None:
        if not bool(getattr(self.settings, "es_enabled", False)) or not str(self.settings.es_base_url or "").strip():
            raise RuntimeError("memory es repository is not configured")
        now = time.time()
        if self._unavailable_until and now < self._unavailable_until:
            raise RuntimeError("memory es repository temporarily unavailable")
        if self._index_ready:
            return
        try:
            response = requests.head(self._url(self.index_name()), headers=self._headers(), auth=self._auth(), timeout=5)
            if response.status_code == 200:
                self._index_ready = True
                return
            if response.status_code not in {404, 400}:
                response.raise_for_status()
            created = requests.put(
                self._url(self.index_name()),
                headers=self._headers(),
                auth=self._auth(),
                json=memory_es_mapping(self.settings),
                timeout=20,
            )
            created.raise_for_status()
            self._index_ready = True
        except Exception:
            self._unavailable_until = time.time() + 15.0
            raise

    def _vector_enabled(self) -> bool:
        return bool(getattr(self.settings, "memory_vector_enabled", False) and self._vector_helper.enabled())

    def _vector_index_async_enabled(self) -> bool:
        return bool(self._vector_enabled() and getattr(self.settings, "memory_index_async", True))

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.es_api_key:
            headers["Authorization"] = "Bearer %s" % self.settings.es_api_key
        return headers

    def _auth(self) -> Optional[Tuple[str, str]]:
        if self.settings.es_api_key:
            return None
        if self.settings.es_username:
            return (self.settings.es_username, self.settings.es_password)
        return None

    def _url(self, path: str) -> str:
        return "%s/%s" % (self.settings.es_base_url.rstrip("/"), path.lstrip("/"))


class MemoryKnowledgeGovernanceService:
    """Govern knowledge suggestions before they can become published semantic assets."""

    def __init__(
        self,
        settings: Settings,
        memory_store: Optional[MemoryStore] = None,
        topic_assets: Optional[Any] = None,
        governance_service: Optional[Any] = None,
        doris_repository: Optional[Any] = None,
    ):
        self.settings = settings
        self.memory_store = memory_store or create_memory_store(settings)
        self._topic_assets = topic_assets
        self._governance_service = governance_service
        self._doris_repository = doris_repository

    def review_suggestion(self, merchant_id: str, suggestion_id: str, request: KnowledgeSuggestionReviewRequest) -> Dict[str, Any]:
        memory = self.memory_store.load(merchant_id)
        suggestion = find_knowledge_suggestion(memory, suggestion_id)
        if not suggestion:
            return {"success": False, "status": "NOT_FOUND", "merchantId": merchant_id, "suggestionId": suggestion_id}
        action = str(getattr(request, "action", "") or "review").strip().lower()
        now = datetime.now().isoformat()
        approved = bool(getattr(request, "approved", False))
        if action == "approve":
            suggestion["status"] = "approved" if approved else "reviewed"
            suggestion["approvedBy"] = str(getattr(request, "reviewer", "") or suggestion.get("approvedBy") or "")
        elif action == "reject":
            suggestion["status"] = "rejected"
        else:
            suggestion["status"] = "reviewed" if approved else "rejected"
            if approved:
                suggestion["approvedBy"] = str(getattr(request, "reviewer", "") or suggestion.get("approvedBy") or "")
        suggestion["reviewer"] = str(getattr(request, "reviewer", "") or suggestion.get("reviewer") or "")
        suggestion["reviewNote"] = str(getattr(request, "review_note", "") or suggestion.get("reviewNote") or "")
        suggestion["reviewedAt"] = now
        suggestion["updatedAt"] = now
        memory["knowledgeSuggestions"] = replace_knowledge_suggestion(memory.get("knowledgeSuggestions") or [], suggestion)
        saved = self.memory_store.save(merchant_id, memory)
        return {
            "success": True,
            "status": suggestion.get("status"),
            "merchantId": merchant_id,
            "suggestionId": suggestion_id,
            "suggestion": find_knowledge_suggestion(saved, suggestion_id),
        }

    def publish_suggestion(
        self,
        merchant_id: str,
        suggestion_id: str,
        reviewer: str = "",
        review_note: str = "",
        topic: str = "",
        table_name: str = "",
    ) -> Dict[str, Any]:
        memory = self.memory_store.load(merchant_id)
        suggestion = find_knowledge_suggestion(memory, suggestion_id)
        if not suggestion:
            return {"success": False, "status": "NOT_FOUND", "merchantId": merchant_id, "suggestionId": suggestion_id}
        if knowledge_suggestion_status(suggestion) not in {"approved", "publish_requested", "published", "indexed"}:
            return {
                "success": False,
                "status": "NOT_APPROVED",
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "currentStatus": knowledge_suggestion_status(suggestion),
            }
        publish_topic = str(topic or suggestion.get("topic") or "").strip()
        publish_table = str(table_name or suggestion.get("sourceTable") or "").strip()
        if not publish_topic or not publish_table:
            return {
                "success": False,
                "status": "MISSING_PUBLISH_TARGET",
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "topic": publish_topic,
                "tableName": publish_table,
            }
        topic_assets = self._topic_assets
        governance_service = self._governance_service
        if topic_assets is None or governance_service is None:
            from merchant_ai.services.assets import SemanticAssetGovernanceService, TopicAssetService
            from merchant_ai.services.repositories import DorisRepository

            topic_assets = topic_assets or TopicAssetService(self.settings)
            governance_service = governance_service or SemanticAssetGovernanceService(
                self.settings,
                self._doris_repository or DorisRepository(self.settings),
                topic_assets,
            )
        preflight = governance_service.preflight_publish(publish_topic, publish_table)
        if not bool(preflight.get("publishable")):
            return {
                "success": False,
                "status": "PREFLIGHT_FAILED",
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "topic": publish_topic,
                "tableName": publish_table,
                "preflight": preflight,
            }
        published = topic_assets.publish(publish_topic, publish_table, True, reviewer, review_note)
        if not bool(published.get("success")):
            return {
                "success": False,
                "status": str(published.get("status") or "PUBLISH_FAILED"),
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "topic": publish_topic,
                "tableName": publish_table,
                "preflight": preflight,
                "published": published,
            }
        governed = governance_service.after_publish(publish_topic, publish_table, reviewer, review_note)
        now = datetime.now().isoformat()
        suggestion["status"] = "published"
        effective_reviewer = reviewer or str(suggestion.get("publishRequestedBy") or suggestion.get("reviewer") or "")
        suggestion["reviewer"] = effective_reviewer
        suggestion["reviewNote"] = review_note or str(suggestion.get("reviewNote") or "")
        suggestion["approvedBy"] = str(suggestion.get("approvedBy") or effective_reviewer or "")
        suggestion["reviewedAt"] = str(suggestion.get("reviewedAt") or now)
        suggestion["publishRequestedAt"] = str(suggestion.get("publishRequestedAt") or now)
        suggestion["publishRequestedBy"] = str(suggestion.get("publishRequestedBy") or effective_reviewer or "")
        suggestion["publishedRefId"] = build_published_ref_id(suggestion, publish_topic, publish_table)
        suggestion["updatedAt"] = now
        payload = suggestion.get("payload") if isinstance(suggestion.get("payload"), dict) else {}
        payload["semanticPublish"] = {
            "topic": publish_topic,
            "tableName": publish_table,
            "preflightStatus": preflight.get("status"),
            "publishStatus": published.get("status"),
            "governedStatus": governed.get("status"),
        }
        suggestion["payload"] = payload
        memory["knowledgeSuggestions"] = replace_knowledge_suggestion(memory.get("knowledgeSuggestions") or [], suggestion)
        saved = self.memory_store.save(merchant_id, memory)
        return {
            "success": True,
            "status": "PUBLISHED",
            "merchantId": merchant_id,
            "suggestionId": suggestion_id,
            "topic": publish_topic,
            "tableName": publish_table,
            "preflight": preflight,
            "published": published,
            "governed": governed,
            "suggestion": find_knowledge_suggestion(saved, suggestion_id),
        }

    def request_publish_suggestion(
        self,
        merchant_id: str,
        suggestion_id: str,
        requested_by: str = "",
        review_note: str = "",
    ) -> Dict[str, Any]:
        memory = self.memory_store.load(merchant_id)
        suggestion = find_knowledge_suggestion(memory, suggestion_id)
        if not suggestion:
            return {"success": False, "status": "NOT_FOUND", "merchantId": merchant_id, "suggestionId": suggestion_id}
        if knowledge_suggestion_status(suggestion) not in {"approved", "publish_requested", "published", "indexed"}:
            return {
                "success": False,
                "status": "NOT_APPROVED",
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "currentStatus": knowledge_suggestion_status(suggestion),
            }
        suggestion["status"] = "publish_requested"
        suggestion["publishRequestedAt"] = datetime.now().isoformat()
        suggestion["publishRequestedBy"] = str(requested_by or suggestion.get("publishRequestedBy") or "")
        if review_note:
            suggestion["reviewNote"] = str(review_note)
        suggestion["updatedAt"] = datetime.now().isoformat()
        memory["knowledgeSuggestions"] = replace_knowledge_suggestion(memory.get("knowledgeSuggestions") or [], suggestion)
        saved = self.memory_store.save(merchant_id, memory)
        return {
            "success": True,
            "status": "PUBLISH_REQUESTED",
            "merchantId": merchant_id,
            "suggestionId": suggestion_id,
            "suggestion": find_knowledge_suggestion(saved, suggestion_id),
        }

    def mark_suggestion_indexed(
        self,
        merchant_id: str,
        suggestion_id: str,
        indexed_ref_id: str = "",
    ) -> Dict[str, Any]:
        memory = self.memory_store.load(merchant_id)
        suggestion = find_knowledge_suggestion(memory, suggestion_id)
        if not suggestion:
            return {"success": False, "status": "NOT_FOUND", "merchantId": merchant_id, "suggestionId": suggestion_id}
        if knowledge_suggestion_status(suggestion) not in {"published", "indexed"}:
            return {
                "success": False,
                "status": "NOT_PUBLISHED",
                "merchantId": merchant_id,
                "suggestionId": suggestion_id,
                "currentStatus": knowledge_suggestion_status(suggestion),
            }
        suggestion["status"] = "indexed"
        suggestion["publishedRefId"] = str(indexed_ref_id or suggestion.get("publishedRefId") or "")
        suggestion["indexedAt"] = datetime.now().isoformat()
        suggestion["updatedAt"] = datetime.now().isoformat()
        memory["knowledgeSuggestions"] = replace_knowledge_suggestion(memory.get("knowledgeSuggestions") or [], suggestion)
        saved = self.memory_store.save(merchant_id, memory)
        return {
            "success": True,
            "status": "INDEXED",
            "merchantId": merchant_id,
            "suggestionId": suggestion_id,
            "suggestion": find_knowledge_suggestion(saved, suggestion_id),
        }

    def run_publish_jobs(
        self,
        merchant_id: str,
        reviewer: str = "",
        limit: int = 10,
        auto_index: bool = True,
    ) -> Dict[str, Any]:
        memory = self.memory_store.load(merchant_id)
        queued = [
            dict(item)
            for item in memory.get("knowledgeSuggestions") or []
            if isinstance(item, dict) and knowledge_suggestion_status(item) == "publish_requested"
        ][: max(1, int(limit or 10))]
        results: List[Dict[str, Any]] = []
        for suggestion in queued:
            publish_result = self.publish_suggestion(
                merchant_id,
                str(suggestion.get("suggestionId") or ""),
                reviewer=reviewer or str(suggestion.get("publishRequestedBy") or ""),
                review_note=str(suggestion.get("reviewNote") or ""),
            )
            if auto_index and publish_result.get("success"):
                indexed = self.mark_suggestion_indexed(
                    merchant_id,
                    str(suggestion.get("suggestionId") or ""),
                    indexed_ref_id=str(((publish_result.get("suggestion") or {}).get("publishedRefId") or "")),
                )
                publish_result["indexed"] = indexed
            results.append(publish_result)
        return {
            "success": True,
            "merchantId": merchant_id,
            "queuedCount": len(queued),
            "processedCount": len(results),
            "results": results,
        }


class MemoryManagementService:
    """Governed management API surface for long-term merchant memory."""

    def __init__(self, settings: Settings, memory_store: Optional[MemoryStore] = None):
        self.settings = settings
        self.memory_store = memory_store or create_memory_store(settings)

    def get_memory(self, merchant_id: str, include_inactive: bool = True) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        memory = self.memory_store.load(target)
        view = memory if include_inactive else filter_memory_items(memory, active_only=True)
        return {
            "success": True,
            "merchantId": target,
            "memory": view,
            "counts": memory_item_counts(view),
            "storageBackend": memory.get("storageBackend") or "",
            "source": memory_source_label(memory, self.settings),
        }

    def patch_item(self, merchant_id: str, memory_id: str, patch: Any) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        payload = memory_patch_payload(patch)
        memory = self.memory_store.load(target)
        located = locate_memory_item(memory, memory_id)
        if not located:
            return {"success": False, "status": "NOT_FOUND", "merchantId": target, "memoryId": memory_id}
        group, _, item, _ = located
        try:
            patch_memory_item(item, payload)
        except ValueError as exc:
            return {"success": False, "status": "INVALID_PATCH", "merchantId": target, "memoryId": memory_id, "error": str(exc)}
        refresh_memory_rollups(memory)
        saved = self.memory_store.save(target, memory)
        updated = locate_memory_item(saved, memory_id)
        return {
            "success": True,
            "status": "UPDATED",
            "merchantId": target,
            "memoryId": memory_id,
            "group": group,
            "item": updated[2] if updated else item,
            "counts": memory_item_counts(saved),
        }

    def delete_item(self, merchant_id: str, memory_id: str, hard_delete: bool = False) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        memory = self.memory_store.load(target)
        located = locate_memory_item(memory, memory_id)
        if not located:
            return {"success": False, "status": "NOT_FOUND", "merchantId": target, "memoryId": memory_id}
        group, _, item, index = located
        if hard_delete:
            memory[group].pop(index)
            status = "HARD_DELETED"
        else:
            patch_memory_item(item, {"status": "deleted", "validUntil": datetime.now().isoformat()})
            status = "DELETED"
        refresh_memory_rollups(memory)
        saved = self.memory_store.save(target, memory)
        return {
            "success": True,
            "status": status,
            "merchantId": target,
            "memoryId": memory_id,
            "group": group,
            "hardDelete": bool(hard_delete),
            "counts": memory_item_counts(saved),
        }

    def cleanup_expired(self, merchant_id: str, hard_delete: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        memory = self.memory_store.load(target)
        scanned = count_memory_items(memory)
        cleaned: List[Dict[str, Any]] = []
        for group, _, item, index in list_memory_items(memory, reverse=True):
            if memory_is_inactive(item) or not is_memory_expired(item):
                continue
            cleaned.append({"memoryId": memory_item_id(item), "group": group, "status": memory_status(item)})
            if dry_run:
                continue
            if hard_delete:
                memory[group].pop(index)
            else:
                patch_memory_item(item, {"status": "deleted", "validUntil": datetime.now().isoformat()})
        saved = memory
        if cleaned and not dry_run:
            refresh_memory_rollups(memory)
            saved = self.memory_store.save(target, memory)
        return {
            "success": True,
            "status": "DRY_RUN" if dry_run else "CLEANED",
            "merchantId": target,
            "hardDelete": bool(hard_delete),
            "dryRun": bool(dry_run),
            "scanned": scanned,
            "cleanedCount": len(cleaned),
            "cleaned": list(reversed(cleaned)),
            "counts": memory_item_counts(saved),
        }

    def evaluate_recall(self, merchant_id: str, cases: Any, budget_tokens: int = 0, budget_chars: int = 0) -> Dict[str, Any]:
        target = merchant_id or self.settings.merchant_id
        case_payloads = memory_recall_case_payloads(cases)
        results: List[Dict[str, Any]] = []
        total_expected = 0
        total_hits = 0
        total_false_positives = 0
        for index, case in enumerate(case_payloads):
            question = str(case.get("question") or "")
            state: Dict[str, Any] = {
                "question": question,
                "requested_merchant_id": target,
                "access_role": str(case.get("accessRole") or case.get("access_role") or "merchant_analyst"),
                "memory_eval_context": {
                    "topics": unique_strings(case.get("topics") or []),
                    "metrics": unique_strings(case.get("metrics") or []),
                    "timeWindows": unique_ints(case.get("timeWindows") or case.get("time_windows") or []),
                },
            }
            selected = self.memory_store.select_for_question(state, budget_tokens=budget_tokens, budget_chars=budget_chars)
            trace = selected.get("memoryInjectionTrace") or {}
            selected_ids = unique_strings(trace.get("selectedIds") or memory_ids_from_selected(selected))
            expected_ids = unique_strings(case.get("expectedMemoryIds") or case.get("expected_memory_ids") or [])
            unexpected_ids = unique_strings(case.get("unexpectedMemoryIds") or case.get("unexpected_memory_ids") or [])
            hits = [memory_id for memory_id in expected_ids if memory_id in selected_ids]
            misses = [memory_id for memory_id in expected_ids if memory_id not in selected_ids]
            false_positives = [memory_id for memory_id in unexpected_ids if memory_id in selected_ids]
            total_expected += len(expected_ids)
            total_hits += len(hits)
            total_false_positives += len(false_positives)
            results.append(
                {
                    "caseId": str(case.get("caseId") or case.get("case_id") or "case_%d" % (index + 1)),
                    "question": question,
                    "passed": not misses and not false_positives,
                    "selectedMemoryIds": selected_ids,
                    "expectedMemoryIds": expected_ids,
                    "unexpectedMemoryIds": unexpected_ids,
                    "hitMemoryIds": hits,
                    "missedMemoryIds": misses,
                    "falsePositiveMemoryIds": false_positives,
                    "candidateCount": int(trace.get("candidateCount") or 0),
                    "filteredReasons": trace.get("filteredReasons") or {},
                }
            )
        hit_rate = float(total_hits / total_expected) if total_expected else 1.0
        return {
            "success": True,
            "merchantId": target,
            "caseCount": len(results),
            "passed": all(item["passed"] for item in results),
            "hitRate": round(hit_rate, 4),
            "expectedCount": total_expected,
            "hitCount": total_hits,
            "falsePositiveCount": total_false_positives,
            "results": results,
        }


class MemoryGovernanceService(MemoryKnowledgeGovernanceService):
    """Alias that represents the broader memory governance boundary."""


class MemoryHotCache:
    """Short TTL cache for enterprise memory injection and hit deltas."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(getattr(settings, "memory_redis_enabled", False))
        ttl = int(getattr(settings, "memory_cache_ttl_seconds", 600) or 0) if self.enabled else 0
        cache_settings = settings.model_copy(
            update={
                "redis_enabled": self.enabled,
                "redis_cache_enabled": True,
            }
        )
        self.cache = build_ttl_cache("enterprise_memory", cache_settings, ttl)

    def get_json(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        return self.cache.get(key)

    def set_json(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.cache.set(key, value)

    def invalidate_merchant(self, merchant_id: str) -> None:
        if not self.enabled:
            return
        # This cache stores only reconstructible memory snippets; coarse
        # invalidation avoids stale injection after a memory write.
        self.cache.clear()

    def increment_hit_delta(self, memory_id: str, merchant_id: str = "") -> None:
        if not self.enabled or not memory_id:
            return
        key = "memory_hit_delta:%s" % memory_id
        current = self.cache.get(key) or {}
        current["memoryId"] = memory_id
        if merchant_id:
            current["merchantId"] = merchant_id
        current["hitCount"] = int(current.get("hitCount") or 0) + 1
        current["lastUsedAt"] = datetime.now().isoformat()
        current["decayScore"] = 1.0
        self.cache.set(key, current)

    def drain_hit_deltas(self) -> Dict[str, Dict[str, Any]]:
        if not self.enabled:
            return {}
        deltas: Dict[str, Dict[str, Any]] = {}
        for key, value in memory_cache_items(self.cache).items():
            if not key.startswith("memory_hit_delta:") or not isinstance(value, dict):
                continue
            memory_id = str(value.get("memoryId") or key.removeprefix("memory_hit_delta:"))
            deltas[memory_id] = value
        return deltas

    def backend_name(self) -> str:
        try:
            trace = self.cache.trace()
            return str(trace.get("backend") or "disabled")
        except Exception:
            return "disabled"


class MemoryVectorIndex:
    """ES dense_vector search over the governed memory index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedding_cache = build_ttl_cache("memory_embedding", settings, settings.cache_recall_ttl_seconds)

    def enabled(self) -> bool:
        return bool(
            getattr(self.settings, "memory_vector_enabled", False)
            and getattr(self.settings, "es_enabled", False)
            and self.settings.es_base_url
            and self.settings.es_vector_field
            and self.settings.embedding_model
            and self._embedding_api_key()
        )

    def sync_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "enabled": self.enabled(), "managedBy": "memory_es_repository"}

    def search(self, merchant_id: str, query_text: str, limit: int = 12) -> List[str]:
        if not self.enabled() or not query_text:
            return []
        vector = self._embed_text(query_text)
        if not vector:
            return []
        size = max(1, int(limit or self.settings.es_vector_top_k or 12))
        knn: Dict[str, Any] = {
            "field": self.settings.es_vector_field,
            "query_vector": vector,
            "k": size,
            "num_candidates": max(size, int(self.settings.es_vector_num_candidates or 80)),
            "filter": {
                "bool": {
                    "filter": [
                        {"term": {"merchant_id": merchant_id}},
                        {"term": {"doc_type": "memory_item"}},
                    ]
                }
            },
        }
        response = requests.post(
            self._url("%s/_search" % self.index_name()),
            headers=self._headers(),
            auth=self._auth(),
            json={"size": size, "knn": knn, "_source": ["memory_id"]},
            timeout=10,
        )
        response.raise_for_status()
        hits = ((response.json() or {}).get("hits") or {}).get("hits") or []
        ids: List[str] = []
        for hit in hits:
            memory_id = str(((hit.get("_source") or {}).get("memory_id") or hit.get("_id") or ""))
            if memory_id and memory_id not in ids:
                ids.append(memory_id)
        return ids

    def index_name(self) -> str:
        return str(getattr(self.settings, "memory_es_index", "") or getattr(self.settings, "memory_vector_index", "") or "merchant_memory")

    def _ensure_index(self) -> None:
        response = requests.head(self._url(self.index_name()), headers=self._headers(), auth=self._auth(), timeout=10)
        if response.status_code == 200:
            return
        if response.status_code not in {404, 400}:
            response.raise_for_status()
        put_response = requests.put(
            self._url(self.index_name()),
            headers=self._headers(),
            auth=self._auth(),
            json=memory_es_mapping(self.settings),
            timeout=20,
        )
        put_response.raise_for_status()

    def _embed_text(self, text: str) -> List[float]:
        value = str(text or "").strip()
        if not value:
            return []
        cache_key = stable_cache_key(
            "memory_embedding",
            {
                "baseUrl": self.settings.embedding_base_url,
                "model": self.settings.embedding_model,
                "dims": self.settings.embedding_dims,
                "textHash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            },
        )
        cached = self._embedding_cache.get(cache_key)
        if isinstance(cached, list):
            return [float(item) for item in cached]
        payload: Dict[str, Any] = {"model": self.settings.embedding_model, "input": value}
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
        body = response.json() or {}
        vector = (((body.get("data") or [{}])[0] or {}).get("embedding") or [])
        result = [float(item) for item in vector if isinstance(item, (int, float))]
        if result:
            self._embedding_cache.set(cache_key, result)
        return result

    def _embedding_api_key(self) -> str:
        return str(self.settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or self.settings.llm_api_key or "").strip()

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.es_api_key:
            headers["Authorization"] = "Bearer %s" % self.settings.es_api_key
        return headers

    def _auth(self) -> Optional[Tuple[str, str]]:
        if self.settings.es_api_key:
            return None
        if self.settings.es_username:
            return (self.settings.es_username, self.settings.es_password)
        return None

    def _url(self, path: str) -> str:
        return "%s/%s" % (self.settings.es_base_url.rstrip("/"), path.lstrip("/"))


def create_memory_store(settings: Settings) -> MemoryStore:
    backend = str(getattr(settings, "memory_backend", "file") or "file").strip().lower()
    if backend in {"es", "mysql", "hybrid"}:
        return EnterpriseMemoryStore(settings)
    return StructuredMemoryStore(settings)


def empty_memory_payload(merchant_id: str) -> Dict[str, Any]:
    return {
        "merchantId": merchant_id,
        "recentFocus": {},
        "coreMemoryProfile": {},
        "preferences": [],
        "facts": [],
        "events": [],
        "conflicts": [],
        "knowledgeSuggestions": [],
        "schemaVersion": MEMORY_SCHEMA_VERSION,
    }


def count_memory_items(memory: Dict[str, Any]) -> int:
    if not isinstance(memory, dict):
        return 0
    return sum(len(memory.get(group) or []) for group in ["events", "preferences", "facts"])


def memory_item_id(item: Dict[str, Any]) -> str:
    return str((item or {}).get("eventId") or (item or {}).get("preferenceId") or (item or {}).get("factId") or "")


def memory_id_set(memory: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for group in ["events", "preferences", "facts"]:
        for item in memory.get(group) or []:
            memory_id = memory_item_id(item)
            if memory_id:
                ids.add(memory_id)
    return ids


def list_memory_items(memory: Dict[str, Any], reverse: bool = False) -> List[Tuple[str, str, Dict[str, Any], int]]:
    items: List[Tuple[str, str, Dict[str, Any], int]] = []
    for group, id_key in [("events", "eventId"), ("preferences", "preferenceId"), ("facts", "factId")]:
        group_items = memory.get(group) or []
        indexes = range(len(group_items) - 1, -1, -1) if reverse else range(len(group_items))
        for index in indexes:
            item = group_items[index]
            if isinstance(item, dict):
                items.append((group, id_key, item, index))
    return items


def locate_memory_item(memory: Dict[str, Any], memory_id: str) -> Optional[Tuple[str, str, Dict[str, Any], int]]:
    target = str(memory_id or "")
    if not target:
        return None
    for group, id_key, item, index in list_memory_items(memory):
        if str(item.get(id_key) or item.get("id") or "") == target:
            return group, id_key, item, index
    return None


def memory_item_counts(memory: Dict[str, Any]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, int]] = {}
    total = 0
    active = 0
    inactive = 0
    expired = 0
    for group, _, item, _ in list_memory_items(memory):
        bucket = groups.setdefault(group, {"total": 0, "active": 0, "inactive": 0, "expired": 0})
        bucket["total"] += 1
        total += 1
        if is_memory_expired(item):
            bucket["expired"] += 1
            expired += 1
        if memory_is_inactive(item):
            bucket["inactive"] += 1
            inactive += 1
        elif not is_memory_expired(item):
            bucket["active"] += 1
            active += 1
    return {"total": total, "active": active, "inactive": inactive, "expired": expired, "groups": groups}


def filter_memory_items(memory: Dict[str, Any], active_only: bool = False) -> Dict[str, Any]:
    payload = dict(memory or {})
    if not active_only:
        return payload
    for group in ["events", "preferences", "facts"]:
        payload[group] = [
            item
            for item in payload.get(group) or []
            if isinstance(item, dict) and not memory_is_inactive(item) and not is_memory_expired(item)
        ]
    payload["recentFocus"] = aggregate_recent_focus(payload.get("events") or [], payload.get("preferences") or [])
    payload["coreMemoryProfile"] = build_core_memory_profile(payload)
    return payload


def memory_patch_payload(patch: Any) -> Dict[str, Any]:
    if patch is None:
        return {}
    if hasattr(patch, "model_dump"):
        return patch.model_dump(by_alias=True, exclude_unset=True)
    if isinstance(patch, dict):
        return dict(patch)
    return {}


def patch_memory_item(item: Dict[str, Any], patch: Dict[str, Any]) -> None:
    status = str(patch.get("status") or "").strip().lower()
    if status:
        allowed = APPROVED_MEMORY_STATUSES | PENDING_MEMORY_STATUSES | INACTIVE_MEMORY_STATUSES
        if status not in allowed:
            raise ValueError("unsupported memory status: %s" % status)
        item["status"] = "active" if status == "" else status
    if patch.get("confidence") is not None:
        item["confidence"] = max(0.0, min(1.0, float(patch.get("confidence") or 0.0)))
    if patch.get("validUntil") is not None:
        item["validUntil"] = str(patch.get("validUntil") or "")
    if patch.get("retentionDays") is not None:
        item["retentionDays"] = max(0, int(patch.get("retentionDays") or 0))
    if patch.get("visibility") is not None:
        item["visibility"] = str(patch.get("visibility") or "merchant")
    if patch.get("allowedRoles") is not None:
        item["allowedRoles"] = unique_strings(patch.get("allowedRoles") or [])
    if patch.get("approvedBy") is not None:
        item["approvedBy"] = str(patch.get("approvedBy") or "")
    if memory_status(item) in {"deleted", "expired"} and not item.get("validUntil"):
        item["validUntil"] = datetime.now().isoformat()


def refresh_memory_rollups(memory: Dict[str, Any]) -> None:
    memory["recentFocus"] = aggregate_recent_focus(
        [item for item in memory.get("events") or [] if memory_item_can_drive_recent_focus(item)],
        [item for item in memory.get("preferences") or [] if memory_item_can_drive_recent_focus(item)],
    )
    memory["coreMemoryProfile"] = build_core_memory_profile(memory)


def memory_recall_case_payloads(cases: Any) -> List[Dict[str, Any]]:
    if hasattr(cases, "cases"):
        cases = getattr(cases, "cases")
    if not isinstance(cases, list):
        return []
    payloads: List[Dict[str, Any]] = []
    for item in cases:
        if hasattr(item, "model_dump"):
            payload = item.model_dump(by_alias=True)
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            continue
        if str(payload.get("question") or "").strip():
            payloads.append(payload)
    return payloads


def scan_memory_items_by_ids(memory: Dict[str, Any], memory_ids: List[str], merchant_id: str) -> Dict[str, Any]:
    wanted = set(unique_strings(memory_ids))
    result = empty_memory_payload(merchant_id)
    if not wanted:
        return result
    for group in ["events", "preferences", "facts"]:
        for item in memory.get(group) or []:
            if isinstance(item, dict) and memory_item_id(item) in wanted:
                result[group].append(item)
    return result


def merge_memory_payload(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for group in ["events", "preferences", "facts"]:
        items = [item for item in merged.get(group) or [] if isinstance(item, dict)]
        seen = {memory_item_id(item) for item in items if memory_item_id(item)}
        for item in extra.get(group) or []:
            if not isinstance(item, dict):
                continue
            memory_id = memory_item_id(item)
            if memory_id and memory_id not in seen:
                items.append(item)
                seen.add(memory_id)
        merged[group] = items
    if not merged.get("recentFocus"):
        merged["recentFocus"] = (base or {}).get("recentFocus") or {}
    if not merged.get("coreMemoryProfile"):
        merged["coreMemoryProfile"] = (base or {}).get("coreMemoryProfile") or {}
    return merged


def memory_source_label(memory: Dict[str, Any], settings: Settings) -> str:
    backend = str((memory or {}).get("storageBackend") or "")
    if backend == "es":
        return "es:%s" % target_memory_index(settings)
    if backend == "json_fallback":
        return "json_fallback:%s" % StructuredMemoryStore(settings).memory_path(str((memory or {}).get("merchantId") or settings.merchant_id))
    if backend == "es_unavailable":
        return "es_unavailable"
    return backend or "es:%s" % target_memory_index(settings)


def memory_query_hash(merchant_id: str, context: Dict[str, Any]) -> str:
    payload = {
        "merchantId": merchant_id,
        "topics": sorted(str(item) for item in context.get("topics") or []),
        "metrics": sorted(str(item) for item in context.get("metrics") or []),
        "timeWindows": sorted(int(item) for item in context.get("timeWindows") or []),
        "analysisIntent": context.get("analysisIntent") or "",
        "objectRefs": stable_object_refs(context.get("objectRefs") or {}),
        "accessRole": str(context.get("accessRole") or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def stable_object_refs(value: Any) -> Any:
    if not value:
        return {}
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key in sorted(str(item) for item in value.keys()):
            item = value.get(key)
            if isinstance(item, (list, tuple, set)):
                result[key] = sorted(str(entry) for entry in item if str(entry or "").strip())[:80]
            elif isinstance(item, dict):
                result[key] = stable_object_refs(item)
            elif str(item or "").strip():
                result[key] = str(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value if str(item or "").strip())[:80]
    text = str(value or "").strip()
    return text if text else {}


def memory_budget_tokens(settings: Settings, budget_tokens: int = 0, budget_chars: int = 0) -> int:
    if budget_tokens:
        return max(100, int(budget_tokens))
    if budget_chars:
        return max(100, estimate_memory_tokens("x" * int(budget_chars)))
    configured = int(getattr(settings, "context_memory_budget_tokens", 0) or 0)
    if configured:
        return max(100, configured)
    chars = int(getattr(settings, "context_memory_budget_chars", 0) or 8000)
    return max(100, estimate_memory_tokens("x" * chars))


def estimate_memory_tokens(text: str) -> int:
    return conservative_token_estimate(text)


def conservative_token_estimate(text: str) -> int:
    value = str(text or "")
    if not value:
        return 1
    cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", value))
    non_cjk_count = max(0, len(value) - cjk_count)
    return max(1, cjk_count + int((non_cjk_count + 3) / 4))


def memory_payload_chars(payload: Dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str))


def memory_payload_tokens(payload: Dict[str, Any]) -> int:
    return estimate_memory_tokens(json.dumps(payload, ensure_ascii=False, default=str))


def memory_injection_budget_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    budget_payload = dict(payload or {})
    budget_payload.pop("preferences", None)
    budget_payload.pop("facts", None)
    return budget_payload


def memory_injection_tokens(payload: Dict[str, Any]) -> int:
    return memory_payload_tokens(memory_injection_budget_payload(payload))


def memory_injection_chars(payload: Dict[str, Any]) -> int:
    return memory_payload_chars(memory_injection_budget_payload(payload))


def truncate_memory_text_by_tokens(text: str, budget_tokens: int) -> str:
    value = str(text or "")
    if not value:
        return ""
    budget = max(1, int(budget_tokens or 1))
    if estimate_memory_tokens(value) <= budget:
        return value
    units_budget = budget * 4
    units = 0
    chars: List[str] = []
    for char in value:
        units += 4 if re.match(r"[\u3400-\u9fff\uf900-\ufaff]", char) else 1
        if units > units_budget:
            break
        chars.append(char)
    return "".join(chars)


def memory_vector_query_text(context: Dict[str, Any]) -> str:
    parts = [
        str(context.get("question") or ""),
        " ".join(sorted(str(item) for item in context.get("topics") or [])),
        " ".join(sorted(str(item) for item in context.get("metrics") or [])),
        str(context.get("analysisIntent") or ""),
    ]
    return " ".join(part for part in parts if part).strip()[:1200]


def boost_vector_candidates(candidates: List[MemoryRetrievalCandidate], vector_ids: List[str]) -> List[MemoryRetrievalCandidate]:
    """Fuse keyword/rule ranking with vector rank while keeping MySQL/JSON as source of truth."""

    vector_rank = {memory_id: index + 1 for index, memory_id in enumerate(vector_ids)}
    text_rank = {candidate.memory_id: index + 1 for index, candidate in enumerate(candidates) if candidate.memory_id}
    rrf_k = 60.0
    scale = 120.0
    fused: List[MemoryRetrievalCandidate] = []
    for candidate in candidates:
        rank = vector_rank.get(candidate.memory_id)
        if not rank:
            fused.append(candidate)
            continue
        if candidate.filter_reason in {"expired", "low_confidence"}:
            fused.append(candidate)
            continue
        text_position = text_rank.get(candidate.memory_id, len(candidates) + 1)
        rrf_score = (1.0 / (rrf_k + text_position)) + (1.0 / (rrf_k + rank))
        reasons = list(candidate.reasons or [])
        if "vector_rrf_match" not in reasons:
            reasons.append("vector_rrf_match")
        fused.append(
            candidate.model_copy(
                update={
                    "score": round(float(candidate.score or 0.0) + rrf_score * scale, 4),
                    "reasons": reasons,
                    "filtered": False,
                    "filter_reason": "",
                }
            )
        )
    fused.sort(key=lambda item: (item.filtered is False, item.score), reverse=True)
    return diversify_memory_candidates(fused)


def diversify_memory_candidates(candidates: List[MemoryRetrievalCandidate]) -> List[MemoryRetrievalCandidate]:
    usable = [candidate for candidate in candidates if not candidate.filtered]
    filtered = [candidate for candidate in candidates if candidate.filtered]
    selected: List[MemoryRetrievalCandidate] = []
    remaining = list(usable)
    while remaining:
        best_index = 0
        best_score = diversity_adjusted_score(remaining[0], selected)
        for index, candidate in enumerate(remaining[1:], start=1):
            score = diversity_adjusted_score(candidate, selected)
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected + filtered


def diversity_adjusted_score(candidate: MemoryRetrievalCandidate, selected: List[MemoryRetrievalCandidate]) -> float:
    score = float(candidate.score or 0.0)
    if not selected:
        return score
    selected_types = [str(item.memory_type or "") for item in selected[-4:]]
    if selected_types.count(str(candidate.memory_type or "")) >= 2:
        score -= 1.0
    candidate_topics = set(unique_strings((candidate.payload or {}).get("topics") or []))
    candidate_metrics = set(unique_strings((candidate.payload or {}).get("metrics") or []))
    for item in selected[-4:]:
        item_topics = set(unique_strings((item.payload or {}).get("topics") or []))
        item_metrics = set(unique_strings((item.payload or {}).get("metrics") or []))
        if candidate_topics and item_topics and candidate_topics == item_topics:
            score -= 0.25
        if candidate_metrics and item_metrics and candidate_metrics == item_metrics:
            score -= 0.35
    return score


def target_memory_index(settings: Settings) -> str:
    return str(getattr(settings, "memory_es_index", "") or getattr(settings, "memory_vector_index", "") or "merchant_memory")


def memory_schema_statements() -> List[str]:
    common = """
        memory_id VARCHAR(128) NOT NULL,
        merchant_id VARCHAR(128) NOT NULL,
        memory_type VARCHAR(64) NOT NULL,
        question TEXT NULL,
        answer_preview TEXT NULL,
        content TEXT NULL,
        topics JSON NULL,
        metrics JSON NULL,
        time_windows JSON NULL,
        analysis_intent VARCHAR(128) NULL,
        confidence DOUBLE DEFAULT 0,
        decay_score DOUBLE DEFAULT 1,
        hit_count INT DEFAULT 0,
        last_used_at DATETIME NULL,
        valid_until DATETIME NULL,
        source VARCHAR(128) NULL,
        status VARCHAR(32) DEFAULT 'active',
        created_at DATETIME NULL,
        updated_at DATETIME NULL,
        payload_json JSON NULL,
        PRIMARY KEY (memory_id),
        KEY idx_merchant_type (merchant_id, memory_type),
        KEY idx_merchant_updated (merchant_id, updated_at)
    """
    return [
        "CREATE TABLE IF NOT EXISTS merchant_memory_event (%s) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" % common,
        "CREATE TABLE IF NOT EXISTS merchant_memory_preference (%s) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" % common,
        "CREATE TABLE IF NOT EXISTS merchant_memory_fact (%s) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" % common,
        """
        CREATE TABLE IF NOT EXISTS merchant_memory_conflict (
            conflict_id VARCHAR(128) NOT NULL,
            merchant_id VARCHAR(128) NOT NULL,
            winner_id VARCHAR(128) NULL,
            loser_id TEXT NULL,
            reason TEXT NULL,
            action VARCHAR(128) NULL,
            status VARCHAR(32) DEFAULT 'active',
            created_at DATETIME NULL,
            updated_at DATETIME NULL,
            payload_json JSON NULL,
            PRIMARY KEY (conflict_id),
            KEY idx_merchant_updated (merchant_id, updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS merchant_knowledge_suggestion (
            suggestion_id VARCHAR(128) NOT NULL,
            merchant_id VARCHAR(128) NOT NULL,
            suggestion_type VARCHAR(64) NOT NULL,
            source_memory_id VARCHAR(128) NULL,
            topic VARCHAR(128) NULL,
            metric_name VARCHAR(256) NULL,
            source_table VARCHAR(256) NULL,
            status VARCHAR(32) DEFAULT 'candidate',
            reviewer VARCHAR(128) NULL,
            approved_by VARCHAR(128) NULL,
            created_at DATETIME NULL,
            updated_at DATETIME NULL,
            payload_json JSON NULL,
            PRIMARY KEY (suggestion_id),
            KEY idx_merchant_status (merchant_id, status),
            KEY idx_merchant_updated (merchant_id, updated_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]


def memory_item_from_mysql_row(row: Dict[str, Any], id_alias: str) -> Dict[str, Any]:
    payload = parse_jsonish(row.get("payload_json"))
    if not isinstance(payload, dict):
        payload = {}
    payload[id_alias] = str(payload.get(id_alias) or row.get("memory_id") or "")
    payload["memoryType"] = str(row.get("memory_type") or payload.get("memoryType") or "")
    payload["topics"] = unique_strings(parse_jsonish(row.get("topics")) or payload.get("topics") or [])
    payload["metrics"] = unique_strings(parse_jsonish(row.get("metrics")) or payload.get("metrics") or [])
    payload["timeWindows"] = unique_ints(parse_jsonish(row.get("time_windows")) or payload.get("timeWindows") or [])
    payload["analysisIntent"] = str(row.get("analysis_intent") or payload.get("analysisIntent") or "")
    payload["confidence"] = float(row.get("confidence") if row.get("confidence") is not None else payload.get("confidence") or 0)
    payload["decayScore"] = float(row.get("decay_score") if row.get("decay_score") is not None else payload.get("decayScore") or 1)
    payload["hitCount"] = int(row.get("hit_count") if row.get("hit_count") is not None else payload.get("hitCount") or 0)
    payload["lastUsedAt"] = isoformat_value(row.get("last_used_at") or payload.get("lastUsedAt") or "")
    payload["validUntil"] = isoformat_value(row.get("valid_until") or payload.get("validUntil") or "")
    payload["source"] = str(row.get("source") or payload.get("source") or "")
    payload["status"] = str(row.get("status") or payload.get("status") or "active")
    payload["createdAt"] = isoformat_value(row.get("created_at") or payload.get("createdAt") or "")
    if id_alias == "eventId":
        payload["question"] = str(row.get("question") or payload.get("question") or "")
        payload["answerPreview"] = str(row.get("answer_preview") or payload.get("answerPreview") or "")
        if row.get("content") and not payload.get("correctionText"):
            payload["correctionText"] = str(row.get("content") or "")
    elif id_alias == "preferenceId":
        if row.get("content") and not payload.get("value"):
            payload["value"] = str(row.get("content") or "")
    elif id_alias == "factId":
        payload["content"] = str(row.get("content") or payload.get("content") or "")
    return payload


def conflict_from_mysql_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = parse_jsonish(row.get("payload_json"))
    if not isinstance(payload, dict):
        payload = {}
    payload["conflictId"] = str(payload.get("conflictId") or row.get("conflict_id") or "")
    payload["winnerId"] = str(row.get("winner_id") or payload.get("winnerId") or "")
    payload["loserId"] = str(row.get("loser_id") or payload.get("loserId") or "")
    payload["reason"] = str(row.get("reason") or payload.get("reason") or "")
    payload["action"] = str(row.get("action") or payload.get("action") or "")
    payload["createdAt"] = isoformat_value(row.get("created_at") or payload.get("createdAt") or "")
    return payload


def knowledge_suggestion_from_mysql_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = parse_jsonish(row.get("payload_json"))
    if not isinstance(payload, dict):
        payload = {}
    payload["suggestionId"] = str(payload.get("suggestionId") or row.get("suggestion_id") or "")
    payload["suggestionType"] = str(row.get("suggestion_type") or payload.get("suggestionType") or "metric")
    payload["sourceMemoryId"] = str(row.get("source_memory_id") or payload.get("sourceMemoryId") or "")
    payload["topic"] = str(row.get("topic") or payload.get("topic") or "")
    payload["metricName"] = str(row.get("metric_name") or payload.get("metricName") or "")
    payload["sourceTable"] = str(row.get("source_table") or payload.get("sourceTable") or "")
    payload["status"] = str(row.get("status") or payload.get("status") or "candidate")
    payload["reviewer"] = str(row.get("reviewer") or payload.get("reviewer") or "")
    payload["approvedBy"] = str(row.get("approved_by") or payload.get("approvedBy") or "")
    payload["createdAt"] = isoformat_value(row.get("created_at") or payload.get("createdAt") or "")
    payload["updatedAt"] = isoformat_value(row.get("updated_at") or payload.get("updatedAt") or "")
    return payload


def parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    try:
        return json.loads(str(value))
    except Exception:
        return None


def isoformat_value(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def memory_cache_items(cache: Any) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    now = time.time()
    raw_items = getattr(cache, "_items", None)
    if isinstance(raw_items, dict):
        for key, value in list(raw_items.items()):
            try:
                expires_at, payload = value
                if expires_at >= now:
                    items[str(key)] = payload
                    if str(key).startswith("memory_hit_delta:"):
                        raw_items.pop(key, None)
            except Exception:
                continue
    fallback = getattr(cache, "_fallback", None)
    if fallback is not None:
        items.update(memory_cache_items(fallback))
    client = getattr(cache, "_client", None)
    if client is not None and bool(getattr(cache, "available", False)) and hasattr(cache, "_key"):
        try:
            prefix = cache._key("memory_hit_delta:*")
            for redis_key in client.scan_iter(match=prefix, count=200):
                raw = client.get(redis_key)
                if raw is None:
                    continue
                key_text = redis_key.decode("utf-8", errors="ignore") if isinstance(redis_key, bytes) else str(redis_key)
                memory_key = key_text.split(":ttl:%s:" % cache.name, 1)[-1]
                items[memory_key] = pickle.loads(raw)
                client.delete(redis_key)
        except Exception:
            return items
    return items


def memory_es_mapping(settings: Settings) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "doc_id": {"type": "keyword"},
        "merchant_id": {"type": "keyword"},
        "doc_type": {"type": "keyword"},
        "group": {"type": "keyword"},
        "memory_id": {"type": "keyword"},
        "memory_type": {"type": "keyword"},
        "memory_tier": {"type": "keyword"},
        "memory_class": {"type": "keyword"},
        "status": {"type": "keyword"},
        "scope": {"type": "object", "enabled": False},
        "topics": {"type": "keyword"},
        "metrics": {"type": "keyword"},
        "content_text": {"type": "text"},
        "payload_json": {"type": "object", "enabled": False},
        "recent_focus": {"type": "object", "enabled": False},
        "core_memory_profile": {"type": "object", "enabled": False},
        "confidence": {"type": "float"},
        "hit_count": {"type": "integer"},
        "decay_score": {"type": "float"},
        "valid_until": {"type": "date"},
        "retention_days": {"type": "integer"},
        "visibility": {"type": "keyword"},
        "allowed_roles": {"type": "keyword"},
        "last_used_at": {"type": "date"},
        "created_at": {"type": "date"},
        "updated_at": {"type": "date"},
    }
    if bool(getattr(settings, "memory_vector_enabled", False)) and settings.es_vector_field:
        properties[str(settings.es_vector_field)] = {
            "type": "dense_vector",
            "dims": max(1, int(settings.embedding_dims or 1536)),
            "index": True,
            "similarity": "cosine",
        }
    return {"mappings": {"properties": properties}}


def memory_es_documents(
    memory: Dict[str, Any],
    settings: Settings,
    embedder: Optional["MemoryVectorIndex"] = None,
) -> List[Dict[str, Any]]:
    merchant_id = str(memory.get("merchantId") or "")
    now = datetime.now().isoformat()
    docs: List[Dict[str, Any]] = []
    for group, id_key in [("event", "eventId"), ("preference", "preferenceId"), ("fact", "factId")]:
        source_key = "%ss" % group if group != "fact" else "facts"
        if group == "event":
            source_key = "events"
        elif group == "preference":
            source_key = "preferences"
        for item in memory.get(source_key) or []:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get(id_key) or item.get("id") or "")
            if not memory_id:
                continue
            content_text = memory_content_text(item)[:2000]
            doc = {
                "doc_id": memory_item_doc_id(merchant_id, group, memory_id),
                "merchant_id": merchant_id,
                "doc_type": "memory_item",
                "group": group,
                "memory_id": memory_id,
                "memory_type": str(item.get("memoryType") or group),
                "memory_tier": memory_tier(item),
                "memory_class": memory_class(item),
                "status": memory_status(item),
                "scope": memory_scope_payload(item),
                "topics": unique_strings(item.get("topics") or []),
                "metrics": unique_strings(item.get("metrics") or []),
                "content_text": content_text,
                "payload_json": dict(item),
                "confidence": float(item.get("confidence") or 0.0),
                "hit_count": int(item.get("hitCount") or 0),
                "decay_score": float(item.get("decayScore") or memory_decay_score(item)),
                "valid_until": str(item.get("validUntil") or ""),
                "retention_days": int(item.get("retentionDays") or 0),
                "visibility": str(item.get("visibility") or "merchant"),
                "allowed_roles": unique_strings(item.get("allowedRoles") or []),
                "last_used_at": str(item.get("lastUsedAt") or ""),
                "created_at": str(item.get("createdAt") or now),
                "updated_at": str(item.get("lastUsedAt") or item.get("createdAt") or now),
            }
            if embedder and content_text:
                vector = embedder._embed_text(content_text)
                if vector:
                    doc[settings.es_vector_field] = vector
            docs.append(doc)
    for conflict in memory.get("conflicts") or []:
        if not isinstance(conflict, dict):
            continue
        conflict_id = str(conflict.get("conflictId") or conflict.get("conflict_id") or "")
        if not conflict_id:
            continue
        docs.append(
            {
                "doc_id": memory_conflict_doc_id(merchant_id, conflict_id),
                "merchant_id": merchant_id,
                "doc_type": "memory_conflict",
                "group": "conflict",
                "memory_id": conflict_id,
                "memory_type": "conflict",
                "status": "active",
                "scope": {},
                "topics": [],
                "metrics": [],
                "content_text": str(conflict.get("reason") or "")[:1000],
                "payload_json": dict(conflict),
                "confidence": 1.0,
                "hit_count": 0,
                "decay_score": 1.0,
                "created_at": str(conflict.get("createdAt") or now),
                "updated_at": str(conflict.get("createdAt") or now),
            }
        )
    for suggestion in memory.get("knowledgeSuggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        suggestion_id = str(suggestion.get("suggestionId") or suggestion.get("id") or "")
        if not suggestion_id:
            continue
        content_text = " ".join(
            part
            for part in [
                str(suggestion.get("metricName") or ""),
                " ".join(unique_strings(suggestion.get("aliases") or [])),
                str(suggestion.get("topic") or ""),
                str(suggestion.get("reviewNote") or ""),
            ]
            if part
        )[:2000]
        docs.append(
            {
                "doc_id": knowledge_suggestion_doc_id(merchant_id, suggestion_id),
                "merchant_id": merchant_id,
                "doc_type": "knowledge_suggestion",
                "group": "suggestion",
                "memory_id": suggestion_id,
                "memory_type": str(suggestion.get("suggestionType") or "metric"),
                "memory_tier": "retrieval",
                "memory_class": "governance_candidate",
                "status": knowledge_suggestion_status(suggestion),
                "scope": {"merchantId": merchant_id, "topic": str(suggestion.get("topic") or "")},
                "topics": unique_strings([suggestion.get("topic")]),
                "metrics": unique_strings([suggestion.get("metricName")]),
                "content_text": content_text,
                "payload_json": dict(suggestion),
                "confidence": 1.0,
                "hit_count": 0,
                "decay_score": 1.0,
                "created_at": str(suggestion.get("createdAt") or now),
                "updated_at": str(suggestion.get("updatedAt") or suggestion.get("createdAt") or now),
            }
        )
    docs.append(
        {
            "doc_id": memory_profile_doc_id(merchant_id),
            "merchant_id": merchant_id,
            "doc_type": "memory_profile",
            "group": "profile",
            "memory_id": merchant_id,
            "memory_type": "memory_profile",
            "status": "active",
            "scope": {"merchantId": merchant_id},
            "topics": [],
            "metrics": [],
            "content_text": str(((memory.get("recentFocus") or {}).get("summary") or ""))[:2000],
            "payload_json": {},
            "recent_focus": memory.get("recentFocus") or {},
            "core_memory_profile": memory.get("coreMemoryProfile") or build_core_memory_profile(memory),
            "confidence": 1.0,
            "hit_count": 0,
            "decay_score": 1.0,
            "created_at": str(memory.get("updatedAt") or now),
            "updated_at": str(memory.get("updatedAt") or now),
        }
    )
    return docs


def memory_es_vector_update_lines(
    memory: Dict[str, Any],
    settings: Settings,
    embedder: "MemoryVectorIndex",
    index_name: str,
) -> List[str]:
    lines: List[str] = []
    if not settings.es_vector_field:
        return lines
    merchant_id = str(memory.get("merchantId") or "")
    for group, id_key in [("event", "eventId"), ("preference", "preferenceId"), ("fact", "factId")]:
        source_key = "facts" if group == "fact" else "%ss" % group
        if group == "event":
            source_key = "events"
        elif group == "preference":
            source_key = "preferences"
        for item in memory.get(source_key) or []:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get(id_key) or item.get("id") or "")
            if not memory_id:
                continue
            content_text = memory_content_text(item)[:2000]
            if not content_text:
                continue
            vector = embedder._embed_text(content_text)
            if not vector:
                continue
            lines.append(
                json.dumps(
                    {"update": {"_index": index_name, "_id": memory_item_doc_id(merchant_id, group, memory_id)}},
                    ensure_ascii=False,
                )
            )
            lines.append(json.dumps({"doc": {settings.es_vector_field: vector}}, ensure_ascii=False, default=str))
    return lines


def memory_item_doc_id(merchant_id: str, group: str, memory_id: str) -> str:
    return "memory_item:%s:%s:%s" % (merchant_id, group, memory_id)


def memory_conflict_doc_id(merchant_id: str, conflict_id: str) -> str:
    return "memory_conflict:%s:%s" % (merchant_id, conflict_id)


def knowledge_suggestion_doc_id(merchant_id: str, suggestion_id: str) -> str:
    return "knowledge_suggestion:%s:%s" % (merchant_id, suggestion_id)


def memory_profile_doc_id(merchant_id: str) -> str:
    return "memory_profile:%s" % merchant_id


def memory_item_from_es_source(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = source.get("payload_json")
    if isinstance(payload, dict):
        next_payload = dict(payload)
        if source.get("memory_id"):
            if "eventId" in next_payload:
                next_payload["eventId"] = str(source.get("memory_id") or next_payload.get("eventId") or "")
            if "preferenceId" in next_payload:
                next_payload["preferenceId"] = str(source.get("memory_id") or next_payload.get("preferenceId") or "")
            if "factId" in next_payload:
                next_payload["factId"] = str(source.get("memory_id") or next_payload.get("factId") or "")
        next_payload["memoryType"] = str(source.get("memory_type") or next_payload.get("memoryType") or "")
        next_payload["memoryTier"] = str(source.get("memory_tier") or next_payload.get("memoryTier") or "")
        next_payload["memoryClass"] = str(source.get("memory_class") or next_payload.get("memoryClass") or "")
        next_payload["status"] = str(source.get("status") or next_payload.get("status") or "active")
        next_payload["scope"] = source.get("scope") if isinstance(source.get("scope"), dict) else next_payload.get("scope") or {}
        next_payload["topics"] = unique_strings(source.get("topics") or next_payload.get("topics") or [])
        next_payload["metrics"] = unique_strings(source.get("metrics") or next_payload.get("metrics") or [])
        next_payload["confidence"] = float(source.get("confidence") if source.get("confidence") is not None else next_payload.get("confidence") or 0.0)
        next_payload["hitCount"] = int(source.get("hit_count") if source.get("hit_count") is not None else next_payload.get("hitCount") or 0)
        next_payload["decayScore"] = float(source.get("decay_score") if source.get("decay_score") is not None else next_payload.get("decayScore") or 1.0)
        next_payload["lastUsedAt"] = str(source.get("last_used_at") or next_payload.get("lastUsedAt") or "")
        next_payload["validUntil"] = str(source.get("valid_until") or next_payload.get("validUntil") or "")
        next_payload["retentionDays"] = int(source.get("retention_days") if source.get("retention_days") is not None else next_payload.get("retentionDays") or 0)
        next_payload["visibility"] = str(source.get("visibility") or next_payload.get("visibility") or "merchant")
        next_payload["allowedRoles"] = unique_strings(source.get("allowed_roles") or next_payload.get("allowedRoles") or [])
        next_payload["createdAt"] = str(source.get("created_at") or next_payload.get("createdAt") or "")
        return next_payload
    parsed = parse_jsonish(payload)
    return parsed if isinstance(parsed, dict) else {}


def conflict_from_es_source(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = source.get("payload_json")
    if isinstance(payload, dict):
        return payload
    parsed = parse_jsonish(payload)
    return parsed if isinstance(parsed, dict) else {}


def knowledge_suggestion_from_es_source(source: Dict[str, Any]) -> Dict[str, Any]:
    payload = source.get("payload_json")
    if isinstance(payload, dict):
        return payload
    parsed = parse_jsonish(payload)
    return parsed if isinstance(parsed, dict) else {}


def memory_vector_documents(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    merchant_id = str(memory.get("merchantId") or "")
    docs: List[Dict[str, Any]] = []
    for group, id_key in [("events", "eventId"), ("preferences", "preferenceId"), ("facts", "factId")]:
        for item in memory.get(group) or []:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get(id_key) or "")
            if (
                not memory_id
                or is_memory_expired(item)
                or float(item.get("confidence") or 0) <= 0.05
                or memory_is_pending(item)
                or memory_is_inactive(item)
                or memory_tier(item) == "core"
            ):
                continue
            content_text = memory_content_text(item)
            if not content_text:
                continue
            docs.append(
                {
                    "memory_id": memory_id,
                    "merchant_id": merchant_id,
                    "memory_type": str(item.get("memoryType") or group),
                    "content_text": content_text[:2000],
                    "topics": unique_strings(item.get("topics") or []),
                    "metrics": unique_strings(item.get("metrics") or []),
                    "confidence": float(item.get("confidence") or 0),
                    "status": memory_status(item),
                    "decay_score": float(item.get("decayScore") or memory_decay_score(item)),
                    "updated_at": datetime.now().isoformat(),
                }
            )
    return docs


def memory_content_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("question"),
        item.get("answerPreview"),
        item.get("content"),
        item.get("key"),
        item.get("value"),
        item.get("correctionText"),
        " ".join(unique_strings(item.get("topics") or [])),
        " ".join(unique_strings(item.get("metrics") or [])),
    ]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def memory_status(item: Dict[str, Any]) -> str:
    status = str((item or {}).get("status") or (item or {}).get("governanceStatus") or (item or {}).get("governance_status") or "").strip()
    return status.lower() if status else "active"


def memory_is_pending(item: Dict[str, Any]) -> bool:
    return memory_status(item) in PENDING_MEMORY_STATUSES


def memory_is_inactive(item: Dict[str, Any]) -> bool:
    return memory_status(item) in INACTIVE_MEMORY_STATUSES


def memory_is_approved(item: Dict[str, Any]) -> bool:
    return memory_status(item) in APPROVED_MEMORY_STATUSES


def memory_item_can_drive_recent_focus(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and not memory_is_pending(item)
        and not memory_is_inactive(item)
        and not is_memory_expired(item)
        and float(item.get("confidence") or 0.0) > 0.05
    )


def memory_scope_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    scope = (item or {}).get("scope")
    if isinstance(scope, dict):
        return scope
    result: Dict[str, Any] = {}
    merchant_id = str((item or {}).get("merchantId") or (item or {}).get("merchant_id") or "")
    user_id = str((item or {}).get("userId") or (item or {}).get("user_id") or "")
    org_id = str((item or {}).get("orgId") or (item or {}).get("org_id") or "")
    if merchant_id:
        result["merchantId"] = merchant_id
    if user_id:
        result["userId"] = user_id
    if org_id:
        result["orgId"] = org_id
    topics = unique_strings((item or {}).get("topics") or [])
    metrics = unique_strings((item or {}).get("metrics") or [])
    if topics:
        result["topics"] = topics[:8]
    if metrics:
        result["metrics"] = metrics[:12]
    return result


def memory_scope_from_terms(merchant_id: str, topics: Iterable[Any], metrics: Iterable[Any]) -> Dict[str, Any]:
    scope: Dict[str, Any] = {}
    if merchant_id:
        scope["merchantId"] = merchant_id
    topic_values = unique_strings(topics or [])
    metric_values = unique_strings(metrics or [])
    if topic_values:
        scope["topics"] = topic_values[:8]
    if metric_values:
        scope["metrics"] = metric_values[:12]
    return scope


def default_memory_status(memory_type: str, source: str = "") -> str:
    normalized = str(memory_type or "")
    if normalized == "metric_dispute":
        return "active"
    if normalized == "past_case":
        return "approved"
    if normalized == "correction":
        return "approved"
    if str(source or "") == "feedback":
        return "approved"
    return "active"


def default_memory_class(memory_type: str) -> str:
    normalized = str(memory_type or "")
    if normalized in {"preference", "user_preference", "business_focus", "metric_habit", "time_window_habit"}:
        return "preference"
    if normalized in {"fact", "business_fact"}:
        return "semantic_fact"
    if normalized == "correction":
        return "correction"
    if normalized == "metric_dispute":
        return "governance_signal"
    if normalized == "past_case":
        return "episodic_case"
    if normalized == "procedure":
        return "procedural_rule"
    if normalized in {"negative_feedback", "feedback"}:
        return "feedback_event"
    return "interaction_event"


def default_memory_tier(memory_type: str, status: str = "", confidence: float = 0.0) -> str:
    normalized = str(memory_type or "")
    normalized_status = str(status or "").strip().lower()
    if normalized in {"preference", "user_preference", "business_focus", "metric_habit", "time_window_habit"}:
        return "core"
    if normalized in {"fact", "business_fact"}:
        return "core"
    if normalized == "correction" and normalized_status in STRONG_CONSTRAINT_STATUSES and float(confidence or 0.0) >= 0.9:
        return "core"
    return "retrieval"


def default_preference_memory_tier(item: Dict[str, Any]) -> str:
    existing = str((item or {}).get("memoryTier") or (item or {}).get("memory_tier") or "").strip().lower()
    if existing in {"core", "retrieval"}:
        return existing
    memory_type = str((item or {}).get("memoryType") or (item or {}).get("memory_type") or "preference")
    if memory_type not in {"preference", "user_preference", "business_focus", "metric_habit", "time_window_habit"}:
        return default_memory_tier(memory_type, memory_status(item), float((item or {}).get("confidence") or 0.55))
    if int((item or {}).get("hitCount") or (item or {}).get("hit_count") or 0) >= HABIT_CORE_PROMOTION_HIT_COUNT:
        return "core"
    if habit_source_is_governed(item) or habit_text_is_explicit(item) or positive_feedback_signal(str((item or {}).get("feedbackSignal") or "")):
        return "core"
    return "retrieval"


def habit_source_is_governed(item: Dict[str, Any]) -> bool:
    source = str((item or {}).get("source") or "").strip().lower()
    return bool((item or {}).get("approvedBy") or (item or {}).get("approved_by") or source in {"manual", "feedback", "correction", "governance"})


def habit_text_is_explicit(item: Dict[str, Any]) -> bool:
    text = " ".join(
        str((item or {}).get(key) or "")
        for key in ["question", "answerPreview", "answer_preview", "correctionText", "correction_text", "key", "value", "content"]
    )
    return any(term in text for term in EXPLICIT_HABIT_TERMS)


def positive_feedback_signal(signal: str) -> bool:
    value = str(signal or "")
    return "adopted" in value or "liked" in value


def memory_class(item: Dict[str, Any]) -> str:
    existing = str((item or {}).get("memoryClass") or (item or {}).get("memory_class") or "").strip()
    return existing or default_memory_class(str((item or {}).get("memoryType") or (item or {}).get("memory_type") or ""))


def memory_tier(item: Dict[str, Any]) -> str:
    existing = str((item or {}).get("memoryTier") or (item or {}).get("memory_tier") or "").strip().lower()
    if existing in {"core", "retrieval"}:
        return existing
    return default_memory_tier(
        str((item or {}).get("memoryType") or (item or {}).get("memory_type") or ""),
        memory_status(item),
        float((item or {}).get("confidence") or 0.0),
    )


def default_retention_days(memory_type: str) -> int:
    normalized = str(memory_type or "")
    if normalized == "correction":
        return 365
    if normalized == "metric_dispute":
        return 180
    if normalized in {"past_case", "procedure"}:
        return 180
    if normalized in {"business_fact", "fact"}:
        return 180
    if normalized in {"preference", "metric_habit", "time_window_habit", "business_focus"}:
        return 90
    if normalized in {"query_event", "negative_feedback", "feedback"}:
        return 45
    return 90


def default_memory_visibility(memory_type: str) -> str:
    normalized = str(memory_type or "")
    if normalized in {"past_case", "procedure"}:
        return "planner_only"
    return "merchant"


def default_memory_allowed_roles(memory_type: str) -> List[str]:
    normalized = str(memory_type or "")
    if normalized in {"past_case", "procedure"}:
        return ["merchant_admin", "merchant_analyst", "system"]
    return ["merchant_admin", "merchant_analyst"]


def build_core_memory_profile(memory: Dict[str, Any]) -> Dict[str, Any]:
    preferences = sorted(
        [item for item in memory.get("preferences") or [] if memory_item_can_drive_core_profile(item) and memory_tier(item) == "core"],
        key=core_memory_rank,
        reverse=True,
    )
    core_fact_items = [
        item
        for item in memory.get("facts") or []
        if memory_item_can_drive_core_profile(item) and memory_tier(item) == "core"
    ]
    facts = sorted(
        [item for item in core_fact_items if str(item.get("memoryType") or "") != "correction"],
        key=core_memory_rank,
        reverse=True,
    )
    corrections = sorted(
        [
            item
            for item in [
                *core_fact_items,
                *[event for event in memory.get("events") or [] if memory_item_can_drive_core_profile(event)],
            ]
            if memory_tier(item) == "core" and str(item.get("memoryType") or "") == "correction"
        ],
        key=core_memory_rank,
        reverse=True,
    )
    return {
        "corePreferences": [compact_memory_payload(item, "preference") for item in preferences[:6]],
        "coreFacts": [compact_memory_payload(item, "fact") for item in facts[:6]],
        "coreCorrections": [compact_memory_payload(item, "event") for item in corrections[:4]],
        "summary": build_core_memory_summary(preferences[:3], facts[:3], corrections[:2]),
    }


def memory_item_can_drive_core_profile(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and memory_is_approved(item)
        and not memory_is_inactive(item)
        and not is_memory_expired(item)
        and float(item.get("confidence") or 0.0) > 0.05
    )


def compact_core_memory_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {"summary": "", "corePreferenceIds": [], "coreFactIds": [], "coreCorrectionIds": [], "counts": {}}
    preference_ids = [str(item.get("id") or "") for item in profile.get("corePreferences") or [] if isinstance(item, dict) and item.get("id")]
    fact_ids = [str(item.get("id") or "") for item in profile.get("coreFacts") or [] if isinstance(item, dict) and item.get("id")]
    correction_ids = [str(item.get("id") or "") for item in profile.get("coreCorrections") or [] if isinstance(item, dict) and item.get("id")]
    return {
        "summary": str(profile.get("summary") or "")[:420],
        "corePreferenceIds": preference_ids[:6],
        "coreFactIds": fact_ids[:6],
        "coreCorrectionIds": correction_ids[:4],
        "counts": {
            "preferences": len(preference_ids),
            "facts": len(fact_ids),
            "corrections": len(correction_ids),
        },
    }


def core_memory_rank(item: Dict[str, Any]) -> Tuple[float, int, str]:
    return (
        float(item.get("confidence") or 0.0),
        int(item.get("hitCount") or 0),
        str(item.get("lastUsedAt") or item.get("createdAt") or ""),
    )


def build_core_memory_summary(preferences: List[Dict[str, Any]], facts: List[Dict[str, Any]], corrections: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    pref_terms = [str(item.get("value") or item.get("key") or "") for item in preferences if str(item.get("value") or item.get("key") or "").strip()]
    fact_terms = [str(item.get("content") or "") for item in facts if str(item.get("content") or "").strip()]
    correction_terms = [
        str(item.get("correctionText") or item.get("content") or item.get("question") or "")
        for item in corrections
        if str(item.get("correctionText") or item.get("content") or item.get("question") or "").strip()
    ]
    if pref_terms:
        parts.append("稳定偏好：" + "；".join(pref_terms[:3])[:180])
    if fact_terms:
        parts.append("业务事实：" + "；".join(fact_terms[:2])[:180])
    if correction_terms:
        parts.append("已确认纠正：" + "；".join(correction_terms[:2])[:180])
    return " | ".join(parts)[:420]


def memory_case_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = (item or {}).get("casePayload") or (item or {}).get("case_payload")
    return payload if isinstance(payload, dict) else {}


def knowledge_suggestion_status(item: Dict[str, Any]) -> str:
    value = str((item or {}).get("status") or "").strip()
    return value.lower() if value else "candidate"


def knowledge_suggestion_publishable(item: Dict[str, Any]) -> bool:
    return knowledge_suggestion_status(item) in {"approved", "publish_requested", "published", "indexed"}


def normalize_memory(payload: Dict[str, Any], merchant_id: str) -> Dict[str, Any]:
    memory = {
        "merchantId": payload.get("merchantId") or merchant_id,
        "recentFocus": payload.get("recentFocus") or {},
        "coreMemoryProfile": payload.get("coreMemoryProfile") or payload.get("core_memory_profile") or {},
        "preferences": normalize_preferences(payload.get("preferences") or []),
        "facts": normalize_facts(payload.get("facts") or []),
        "events": normalize_events(payload.get("events") or []),
        "conflicts": [item for item in payload.get("conflicts", []) if isinstance(item, dict)],
        "knowledgeSuggestions": normalize_knowledge_suggestions(payload.get("knowledgeSuggestions") or payload.get("knowledge_suggestions") or []),
        "memoryIngestionTrace": payload.get("memoryIngestionTrace") or {},
        "updatedAt": payload.get("updatedAt") or "",
        "schemaVersion": MEMORY_SCHEMA_VERSION,
    }
    memory["events"] = memory["events"][-MAX_EVENTS:]
    memory["preferences"] = memory["preferences"][-MAX_PREFERENCES:]
    memory["facts"] = memory["facts"][-MAX_FACTS:]
    memory["knowledgeSuggestions"] = memory["knowledgeSuggestions"][-MAX_KNOWLEDGE_SUGGESTIONS:]
    if not memory["recentFocus"]:
        memory["recentFocus"] = aggregate_recent_focus(
            [item for item in memory["events"] if memory_item_can_drive_recent_focus(item)],
            [item for item in memory["preferences"] if memory_item_can_drive_recent_focus(item)],
        )
    memory["coreMemoryProfile"] = build_core_memory_profile(memory)
    return memory


def normalize_events(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        event = MemoryEvent(
            event_id=str(item.get("eventId") or item.get("event_id") or "mem_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f")),
            memory_type=str(item.get("memoryType") or item.get("memory_type") or "query_event"),
            memory_tier=str(item.get("memoryTier") or item.get("memory_tier") or default_memory_tier(str(item.get("memoryType") or item.get("memory_type") or "query_event"), memory_status(item), float(item.get("confidence") or 0.0))),
            memory_class=str(item.get("memoryClass") or item.get("memory_class") or default_memory_class(str(item.get("memoryType") or item.get("memory_type") or "query_event"))),
            question=str(item.get("question") or "")[:1000],
            answer_preview=str(item.get("answerPreview") or item.get("answer_preview") or "")[:1000],
            topics=unique_strings(item.get("topics") or []),
            metrics=unique_strings(item.get("metrics") or []),
            time_windows=unique_ints(item.get("timeWindows") or item.get("time_windows") or []),
            analysis_intent=str(item.get("analysisIntent") or item.get("analysis_intent") or ""),
            is_follow_up=bool(item.get("isFollowUp") or item.get("is_follow_up") or False),
            feedback_signal=str(item.get("feedbackSignal") or item.get("feedback_signal") or ""),
            correction_text=str(item.get("correctionText") or item.get("correction_text") or ""),
            confidence=float(item.get("confidence") or default_confidence(str(item.get("memoryType") or item.get("memory_type") or "query_event"), str(item.get("feedbackSignal") or ""))),
            source=str(item.get("source") or "answer_run"),
            hit_count=int(item.get("hitCount") or item.get("hit_count") or 0),
            last_used_at=str(item.get("lastUsedAt") or item.get("last_used_at") or ""),
            decay_score=float(item.get("decayScore") or item.get("decay_score") or 1.0),
            valid_until=str(item.get("validUntil") or item.get("valid_until") or ""),
            retention_days=int(item.get("retentionDays") or item.get("retention_days") or default_retention_days(str(item.get("memoryType") or item.get("memory_type") or "query_event"))),
            supersedes=unique_strings(item.get("supersedes") or []),
            conflicts_with=unique_strings(item.get("conflictsWith") or item.get("conflicts_with") or []),
            scope=memory_scope_payload(item),
            status=memory_status(item),
            visibility=str(item.get("visibility") or default_memory_visibility(str(item.get("memoryType") or item.get("memory_type") or "query_event"))),
            allowed_roles=unique_strings(item.get("allowedRoles") or item.get("allowed_roles") or default_memory_allowed_roles(str(item.get("memoryType") or item.get("memory_type") or "query_event"))),
            approved_by=str(item.get("approvedBy") or item.get("approved_by") or ""),
            evidence_refs=unique_strings(item.get("evidenceRefs") or item.get("evidence_refs") or []),
            case_payload=memory_case_payload(item),
            case_summary=str(item.get("caseSummary") or item.get("case_summary") or "")[:1000],
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
        )
        normalized.append(event.model_dump(by_alias=True))
    return normalized


def normalize_preferences(items: Any) -> List[Dict[str, Any]]:
    if isinstance(items, dict):
        next_items = []
        for key, value in items.items():
            next_items.append({"key": str(key), "value": str(value), "memoryType": "preference"})
        items = next_items
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        pref = MemoryPreference(
            preference_id=str(item.get("preferenceId") or item.get("preference_id") or "pref_%s" % stable_slug(item_key)),
            memory_type=str(item.get("memoryType") or item.get("memory_type") or "preference"),
            memory_tier=default_preference_memory_tier(item),
            memory_class=str(item.get("memoryClass") or item.get("memory_class") or default_memory_class(str(item.get("memoryType") or item.get("memory_type") or "preference"))),
            key=str(item.get("key") or ""),
            value=str(item.get("value") or ""),
            topics=unique_strings(item.get("topics") or []),
            metrics=unique_strings(item.get("metrics") or []),
            confidence=float(item.get("confidence") or 0.55),
            source=str(item.get("source") or "memory"),
            hit_count=int(item.get("hitCount") or item.get("hit_count") or 0),
            last_used_at=str(item.get("lastUsedAt") or item.get("last_used_at") or ""),
            decay_score=float(item.get("decayScore") or item.get("decay_score") or 1.0),
            valid_until=str(item.get("validUntil") or item.get("valid_until") or ""),
            retention_days=int(item.get("retentionDays") or item.get("retention_days") or default_retention_days(str(item.get("memoryType") or item.get("memory_type") or "preference"))),
            scope=memory_scope_payload(item),
            status=memory_status(item),
            visibility=str(item.get("visibility") or default_memory_visibility(str(item.get("memoryType") or item.get("memory_type") or "preference"))),
            allowed_roles=unique_strings(item.get("allowedRoles") or item.get("allowed_roles") or default_memory_allowed_roles(str(item.get("memoryType") or item.get("memory_type") or "preference"))),
            approved_by=str(item.get("approvedBy") or item.get("approved_by") or ""),
            evidence_refs=unique_strings(item.get("evidenceRefs") or item.get("evidence_refs") or []),
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
        )
        if pref.key and pref.value:
            normalized.append(pref.model_dump(by_alias=True))
    return normalized


def normalize_facts(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        fact = MemoryFact(
            fact_id=str(item.get("factId") or item.get("fact_id") or "fact_%s" % stable_slug(item_key)),
            memory_type=str(item.get("memoryType") or item.get("memory_type") or "business_fact"),
            memory_tier=str(item.get("memoryTier") or item.get("memory_tier") or default_memory_tier(str(item.get("memoryType") or item.get("memory_type") or "business_fact"), memory_status(item), float(item.get("confidence") or 0.6))),
            memory_class=str(item.get("memoryClass") or item.get("memory_class") or default_memory_class(str(item.get("memoryType") or item.get("memory_type") or "business_fact"))),
            content=str(item.get("content") or item.get("text") or "")[:1000],
            topics=unique_strings(item.get("topics") or []),
            metrics=unique_strings(item.get("metrics") or []),
            confidence=float(item.get("confidence") or 0.6),
            source=str(item.get("source") or "memory"),
            hit_count=int(item.get("hitCount") or item.get("hit_count") or 0),
            last_used_at=str(item.get("lastUsedAt") or item.get("last_used_at") or ""),
            decay_score=float(item.get("decayScore") or item.get("decay_score") or 1.0),
            valid_until=str(item.get("validUntil") or item.get("valid_until") or ""),
            retention_days=int(item.get("retentionDays") or item.get("retention_days") or default_retention_days(str(item.get("memoryType") or item.get("memory_type") or "business_fact"))),
            supersedes=unique_strings(item.get("supersedes") or []),
            conflicts_with=unique_strings(item.get("conflictsWith") or item.get("conflicts_with") or []),
            scope=memory_scope_payload(item),
            status=memory_status(item),
            visibility=str(item.get("visibility") or default_memory_visibility(str(item.get("memoryType") or item.get("memory_type") or "business_fact"))),
            allowed_roles=unique_strings(item.get("allowedRoles") or item.get("allowed_roles") or default_memory_allowed_roles(str(item.get("memoryType") or item.get("memory_type") or "business_fact"))),
            approved_by=str(item.get("approvedBy") or item.get("approved_by") or ""),
            evidence_refs=unique_strings(item.get("evidenceRefs") or item.get("evidence_refs") or []),
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
        )
        if fact.content:
            normalized.append(fact.model_dump(by_alias=True))
    return normalized


def normalize_knowledge_suggestions(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        payload_key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        suggestion = KnowledgeSuggestion(
            suggestion_id=str(item.get("suggestionId") or item.get("id") or item.get("suggestion_id") or "ks_%s" % stable_slug(payload_key)),
            suggestion_type=str(item.get("suggestionType") or item.get("type") or item.get("suggestion_type") or "metric"),
            status=knowledge_suggestion_status(item),
            source=str(item.get("source") or "memory"),
            source_memory_id=str(item.get("sourceMemoryId") or item.get("source_memory_id") or ""),
            source_refs=unique_strings(item.get("sourceRefs") or item.get("source_refs") or []),
            topic=str(item.get("topic") or ""),
            metric_name=str(item.get("metricName") or item.get("metric_name") or ""),
            aliases=unique_strings(item.get("aliases") or []),
            source_table=str(item.get("sourceTable") or item.get("source_table") or ""),
            source_fields=unique_strings(item.get("sourceFields") or item.get("source_fields") or []),
            aggregation=str(item.get("aggregation") or ""),
            filter_conditions=unique_strings(item.get("filterConditions") or item.get("filter_conditions") or []),
            dependency_fields=unique_strings(item.get("dependencyFields") or item.get("dependency_fields") or []),
            reviewer=str(item.get("reviewer") or ""),
            review_note=str(item.get("reviewNote") or item.get("review_note") or ""),
            approved_by=str(item.get("approvedBy") or item.get("approved_by") or ""),
            reviewed_at=str(item.get("reviewedAt") or item.get("reviewed_at") or ""),
            publish_requested_at=str(item.get("publishRequestedAt") or item.get("publish_requested_at") or ""),
            publish_requested_by=str(item.get("publishRequestedBy") or item.get("publish_requested_by") or ""),
            published_ref_id=str(item.get("publishedRefId") or item.get("published_ref_id") or ""),
            indexed_at=str(item.get("indexedAt") or item.get("indexed_at") or ""),
            payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
            updated_at=str(item.get("updatedAt") or item.get("updated_at") or ""),
        )
        normalized.append(suggestion.model_dump(by_alias=True))
    return normalized


def memory_event_from_state(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan")
    topics, metrics, time_windows = plan_memory_terms(plan)
    route_slots = state.get("route_slots")
    route_payload = route_slots.model_dump(by_alias=True) if hasattr(route_slots, "model_dump") else (route_slots or {})
    if not time_windows:
        days = int(((route_payload.get("timeWindow") or {}).get("days") or 0) if isinstance(route_payload, dict) else 0)
        if days:
            time_windows.append(days)
    question = str(state.get("question") or "")[:1000]
    correction_text = extract_correction_text(question)
    if correction_text and is_metric_definition_dispute(question, metrics):
        memory_type = "metric_dispute"
    elif correction_text:
        memory_type = "correction"
    else:
        memory_type = "business_focus" if analysis_intent_from_plan(plan) not in {"", "none"} else "query_event"
    feedback_signal = "persisted" if state.get("persisted") else ""
    event = MemoryEvent(
        event_id="mem_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        memory_type=memory_type,
        memory_tier=default_memory_tier(memory_type, default_memory_status(memory_type, source="answer_run"), default_confidence(memory_type, feedback_signal)),
        memory_class=default_memory_class(memory_type),
        question=question,
        answer_preview=str(state.get("answer") or "")[:1000],
        topics=topics[:8],
        metrics=metrics[:12],
        time_windows=sorted(set(time_windows))[:6],
        analysis_intent=analysis_intent_from_plan(plan),
        is_follow_up=bool(state.get("thread_context") or state.get("previous_entities")),
        feedback_signal=feedback_signal,
        correction_text=correction_text,
        confidence=default_confidence(memory_type, feedback_signal),
        source="answer_run",
        scope=memory_scope_from_terms(str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or ""), topics, metrics),
        status=default_memory_status(memory_type, source="answer_run"),
        retention_days=default_retention_days(memory_type),
        visibility=default_memory_visibility(memory_type),
        allowed_roles=default_memory_allowed_roles(memory_type),
        evidence_refs=state_semantic_refs(state)[:16],
        created_at=datetime.now().isoformat(),
    )
    return event.model_dump(by_alias=True)


def memory_event_from_feedback(pending: PendingAnswer, adopted: Any = None, liked: Any = None, disliked: Any = None) -> Dict[str, Any]:
    signal_parts: List[str] = []
    if bool(adopted):
        signal_parts.append("adopted")
    if bool(liked):
        signal_parts.append("liked")
    if bool(disliked):
        signal_parts.append("disliked")
    signal = ",".join(signal_parts)
    memory_type = "negative_feedback" if bool(disliked) else "query_event"
    event = MemoryEvent(
        event_id="memfb_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        memory_type=memory_type,
        memory_tier=default_memory_tier(memory_type, default_memory_status(memory_type, source="feedback"), default_confidence(memory_type, signal)),
        memory_class=default_memory_class(memory_type),
        question=str(pending.question or "")[:1000],
        answer_preview=str(pending.answer or "")[:1000],
        topics=unique_strings([pending.category_name]),
        metrics=extract_metric_like_terms(pending.question),
        time_windows=extract_time_windows(pending.question),
        feedback_signal=signal,
        confidence=default_confidence(memory_type, signal),
        source="feedback",
        scope=memory_scope_from_terms(pending.merchant_id, unique_strings([pending.category_name]), extract_metric_like_terms(pending.question)),
        status=default_memory_status(memory_type, source="feedback"),
        retention_days=default_retention_days(memory_type),
        visibility=default_memory_visibility(memory_type),
        allowed_roles=default_memory_allowed_roles(memory_type),
        created_at=datetime.now().isoformat(),
    )
    return event.model_dump(by_alias=True)


def past_case_event_from_state(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan")
    if plan is None or not getattr(plan, "intents", None):
        return {}
    run_result = state.get("agent_run_result")
    validation = state.get("query_graph_validation_result")
    validation_gaps = getattr(validation, "gaps", []) if validation is not None else []
    evidence_gaps = getattr(run_result, "evidence_gaps", []) if run_result is not None else []
    completed = bool(state.get("chat_bi_completed")) or bool(getattr(run_result, "task_results", None))
    has_failure = bool(validation_gaps or evidence_gaps or state.get("planner_provider_error"))
    if not completed and not has_failure:
        return {}
    topics, metrics, time_windows = plan_memory_terms(plan)
    case_payload = {
        "caseStatus": "failure" if has_failure and not bool(state.get("chat_bi_completed")) else "success",
        "route": str(getattr(state.get("routing_decision"), "route", "") or ""),
        "intentCount": len(getattr(plan, "intents", []) or []),
        "dependencyCount": len(getattr(plan, "dependencies", []) or []),
        "semanticRefIds": state_semantic_refs(state)[:24],
        "recallRefs": state_recall_refs(state)[:24],
        "validationGaps": [gap_code(item) for item in validation_gaps[:12]],
        "evidenceGaps": [gap_code(item) for item in evidence_gaps[:12]],
        "repairActions": state_repair_actions(state)[:12],
        "answerWithGap": bool(evidence_gaps or validation_gaps or state.get("planner_provider_error")),
        "sqlSummaries": state_sql_summaries(state)[:12],
    }
    summary = "QueryGraph案例: topics=%s metrics=%s status=%s gaps=%d" % (
        ",".join(topics[:4]),
        ",".join(metrics[:6]),
        case_payload["caseStatus"],
        len(case_payload["validationGaps"]) + len(case_payload["evidenceGaps"]),
    )
    event = MemoryEvent(
        event_id="case_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        memory_type="past_case",
        memory_tier=default_memory_tier("past_case", "approved", 0.72 if case_payload["caseStatus"] == "success" else 0.62),
        memory_class=default_memory_class("past_case"),
        question=str(state.get("question") or "")[:1000],
        answer_preview=str(state.get("answer") or "")[:500],
        topics=topics[:8],
        metrics=metrics[:12],
        time_windows=sorted(set(time_windows))[:6],
        analysis_intent=analysis_intent_from_plan(plan),
        confidence=0.72 if case_payload["caseStatus"] == "success" else 0.62,
        source="query_graph_run",
        scope=memory_scope_from_terms(str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or ""), topics, metrics),
        status="approved",
        retention_days=default_retention_days("past_case"),
        visibility=default_memory_visibility("past_case"),
        allowed_roles=default_memory_allowed_roles("past_case"),
        evidence_refs=case_payload["semanticRefIds"] + case_payload["recallRefs"][:8],
        case_payload=case_payload,
        case_summary=summary,
        created_at=datetime.now().isoformat(),
    )
    return event.model_dump(by_alias=True)


def procedure_event_from_state(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan")
    if plan is None or not getattr(plan, "intents", None):
        return {}
    repair_actions = state_repair_actions(state)
    validation = state.get("query_graph_validation_result")
    validation_gaps = getattr(validation, "gaps", []) if validation is not None else []
    run_result = state.get("agent_run_result")
    evidence_gaps = getattr(run_result, "evidence_gaps", []) if run_result is not None else []
    sql_repairs = getattr(run_result, "sql_repairs", []) if run_result is not None else []
    if not repair_actions and not validation_gaps and not evidence_gaps and not sql_repairs:
        return {}
    topics, metrics, time_windows = plan_memory_terms(plan)
    payload = {
        "route": str(getattr(state.get("routing_decision"), "route", "") or ""),
        "repairActions": repair_actions[:12],
        "validationGaps": [gap_code(item) for item in validation_gaps[:12]],
        "evidenceGaps": [gap_code(item) for item in evidence_gaps[:12]],
        "sqlRepairCount": len(sql_repairs),
        "reviewed": bool(state.get("sql_repair_reviewed")),
    }
    summary_parts = []
    if payload["repairActions"]:
        summary_parts.append("repair=" + ",".join(payload["repairActions"][:3]))
    if payload["validationGaps"]:
        summary_parts.append("validation=" + ",".join(payload["validationGaps"][:2]))
    if payload["evidenceGaps"]:
        summary_parts.append("evidence=" + ",".join(payload["evidenceGaps"][:2]))
    if payload["sqlRepairCount"]:
        summary_parts.append("sql_repairs=%d" % payload["sqlRepairCount"])
    summary = "Planner/Repair经验: %s" % ("; ".join(summary_parts)[:500] or "repair trace")
    event = MemoryEvent(
        event_id="proc_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        memory_type="procedure",
        memory_tier=default_memory_tier("procedure", "approved" if payload["reviewed"] else "reviewed", 0.68 if payload["reviewed"] else 0.6),
        memory_class=default_memory_class("procedure"),
        question=str(state.get("question") or "")[:1000],
        answer_preview=summary[:500],
        topics=topics[:8],
        metrics=metrics[:12],
        time_windows=sorted(set(time_windows))[:6],
        analysis_intent=analysis_intent_from_plan(plan),
        confidence=0.68 if payload["reviewed"] else 0.6,
        source="planner_repair",
        scope=memory_scope_from_terms(str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or ""), topics, metrics),
        status="approved" if payload["reviewed"] else "reviewed",
        retention_days=default_retention_days("procedure"),
        visibility=default_memory_visibility("procedure"),
        allowed_roles=default_memory_allowed_roles("procedure"),
        evidence_refs=state_semantic_refs(state)[:16],
        case_payload=payload,
        case_summary=summary,
        created_at=datetime.now().isoformat(),
    )
    return event.model_dump(by_alias=True)


def knowledge_suggestion_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    memory_type = str(event.get("memoryType") or "")
    if memory_type not in {"correction", "metric_dispute"}:
        return {}
    metrics = unique_strings(event.get("metrics") or [])
    if not metrics and memory_type == "metric_dispute":
        metrics = unique_strings(extract_metric_like_terms(str(event.get("question") or "")))
    if not metrics:
        return {}
    metric_name = metrics[0]
    source_memory_id = str(event.get("eventId") or "")
    suggestion = KnowledgeSuggestion(
        suggestion_id="ks_%s_%s" % (stable_slug(metric_name), stable_slug(source_memory_id or str(event.get("question") or ""))[:20]),
        suggestion_type="metric",
        status="candidate",
        source=str(event.get("source") or "memory"),
        source_memory_id=source_memory_id,
        source_refs=unique_strings(event.get("evidenceRefs") or []),
        topic=(unique_strings(event.get("topics") or []) or [""])[0],
        metric_name=metric_name,
        aliases=unique_strings([metric_name, *extract_metric_like_terms(str(event.get("question") or ""))])[:12],
        dependency_fields=unique_strings(event.get("metrics") or [])[:12],
        payload={
            "memoryType": memory_type,
            "question": str(event.get("question") or "")[:1000],
            "correctionText": str(event.get("correctionText") or "")[:1000],
            "governance": "candidate only; publish through semantic asset review before rebuilding ES recall index",
        },
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    return suggestion.model_dump(by_alias=True)


def upsert_knowledge_suggestion(items: List[Dict[str, Any]], suggestion: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    normalized = normalize_knowledge_suggestions(items)
    suggestion = normalize_knowledge_suggestions([suggestion])[0] if suggestion else {}
    if not suggestion:
        return normalized, False
    for item in normalized:
        if item.get("suggestionId") == suggestion.get("suggestionId") or (
            item.get("sourceMemoryId") and item.get("sourceMemoryId") == suggestion.get("sourceMemoryId")
        ):
            existing_status = knowledge_suggestion_status(item)
            item.update({k: v for k, v in suggestion.items() if v not in ("", [], {})})
            if existing_status in {"reviewed", "approved", "published", "indexed", "rejected"}:
                item["status"] = existing_status
            item["updatedAt"] = datetime.now().isoformat()
            return normalized[-MAX_KNOWLEDGE_SUGGESTIONS:], False
    normalized.append(suggestion)
    return normalized[-MAX_KNOWLEDGE_SUGGESTIONS:], True


def replace_knowledge_suggestion(items: List[Dict[str, Any]], suggestion: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized = normalize_knowledge_suggestions(items)
    target_id = str((suggestion or {}).get("suggestionId") or "")
    replaced = False
    result: List[Dict[str, Any]] = []
    for item in normalized:
        if target_id and str(item.get("suggestionId") or "") == target_id:
            result.append(normalize_knowledge_suggestions([suggestion])[0])
            replaced = True
        else:
            result.append(item)
    if not replaced and suggestion:
        result.append(normalize_knowledge_suggestions([suggestion])[0])
    return result[-MAX_KNOWLEDGE_SUGGESTIONS:]


def find_knowledge_suggestion(memory: Dict[str, Any], suggestion_id: str) -> Dict[str, Any]:
    target = str(suggestion_id or "")
    for item in memory.get("knowledgeSuggestions") or []:
        if isinstance(item, dict) and str(item.get("suggestionId") or "") == target:
            return dict(item)
    return {}


def build_published_ref_id(suggestion: Dict[str, Any], topic: str, table_name: str) -> str:
    existing = str((suggestion or {}).get("publishedRefId") or "")
    if existing:
        return existing
    suggestion_type = str((suggestion or {}).get("suggestionType") or "metric")
    metric_name = str((suggestion or {}).get("metricName") or "")
    if suggestion_type == "metric" and metric_name:
        return "semantic:%s:%s:metric:%s" % (topic, table_name, metric_name)
    if suggestion_type and topic and table_name:
        return "semantic:%s:%s:%s:%s" % (topic, table_name, suggestion_type, stable_slug(metric_name or suggestion_type))
    return ""


def state_semantic_refs(state: AgentState) -> List[str]:
    refs: List[str] = []
    plan = state.get("plan")
    if plan is not None:
        for intent in getattr(plan, "intents", []) or []:
            for ref in getattr(intent, "knowledge_ref_ids", []) or []:
                if ref:
                    refs.append(str(ref))
            resolution = getattr(intent, "metric_resolution", {}) or {}
            ref = str(resolution.get("semanticRefId") or resolution.get("semantic_ref_id") or "")
            if ref:
                refs.append(ref)
    pack = state.get("planning_asset_pack")
    source_refs = getattr(pack, "source_refs", {}) if pack is not None else {}
    if isinstance(source_refs, dict):
        for ref_id, item in list(source_refs.items())[:80]:
            metadata = getattr(item, "metadata", {}) if item is not None else {}
            ref = str((metadata or {}).get("semanticRefId") or ref_id or getattr(item, "doc_id", "") or "")
            if ref:
                refs.append(ref)
    return unique_strings(refs)


def state_recall_refs(state: AgentState) -> List[str]:
    bundle = state.get("recall_bundle")
    items = getattr(bundle, "items", []) if bundle is not None else []
    refs: List[str] = []
    for item in items or []:
        metadata = getattr(item, "metadata", {}) if item is not None else {}
        ref = str(getattr(item, "doc_id", "") or (metadata or {}).get("semanticRefId") or "")
        if ref:
            refs.append(ref)
    return unique_strings(refs)


def state_repair_actions(state: AgentState) -> List[str]:
    actions: List[str] = []
    for item in state.get("planner_repair_requests") or []:
        action = getattr(item, "suggested_action", "") if hasattr(item, "suggested_action") else (item or {}).get("suggestedAction", "")
        reason = getattr(item, "reason", "") if hasattr(item, "reason") else (item or {}).get("reason", "")
        text = "%s:%s" % (action, str(reason)[:160])
        if text.strip(":"):
            actions.append(text)
    for item in state.get("action_history") or []:
        action = getattr(item, "action", "") if hasattr(item, "action") else (item or {}).get("action", "")
        status = getattr(item, "status", "") if hasattr(item, "status") else (item or {}).get("status", "")
        if action in {"repair_graph", "retrieve_knowledge", "plan_graph"}:
            actions.append("%s:%s" % (action, status))
    return unique_strings(actions)


def state_sql_summaries(state: AgentState) -> List[Dict[str, Any]]:
    run_result = state.get("agent_run_result")
    summaries: List[Dict[str, Any]] = []
    for task in getattr(run_result, "task_results", []) or []:
        bundle = getattr(task, "query_bundle", None)
        summaries.append(
            {
                "taskId": str(getattr(task, "task_id", "") or ""),
                "table": (getattr(bundle, "tables", []) or [""])[0] if bundle is not None else "",
                "success": bool(getattr(task, "success", False)),
                "failed": bool(getattr(bundle, "failed", False)) if bundle is not None else False,
                "rowCount": len(getattr(bundle, "rows", []) or []) if bundle is not None else 0,
            }
        )
    return summaries


def gap_code(item: Any) -> str:
    if hasattr(item, "code"):
        code = getattr(item, "code", "")
        task_id = getattr(item, "task_id", "")
        return "%s:%s" % (code, task_id) if task_id else str(code)
    if isinstance(item, dict):
        code = str(item.get("code") or "")
        task_id = str(item.get("taskId") or item.get("task_id") or "")
        return "%s:%s" % (code, task_id) if task_id else code
    return str(item)


def plan_memory_terms(plan: Any) -> Tuple[List[str], List[str], List[int]]:
    topics: List[str] = []
    metrics: List[str] = []
    time_windows: List[int] = []
    if plan is not None:
        for intent in getattr(plan, "intents", [])[:16]:
            category = str(getattr(intent, "category", "") or "")
            if category and category not in topics:
                topics.append(category)
            resolution = getattr(intent, "metric_resolution", {}) or {}
            metric = str(resolution.get("metricKey") or getattr(intent, "metric_name", "") or "").strip()
            if metric and metric not in metrics:
                metrics.append(metric)
            days = int(getattr(intent, "days", 0) or 0)
            if days:
                time_windows.append(days)
    return topics, metrics, time_windows


def analysis_intent_from_plan(plan: Any) -> str:
    try:
        value = (getattr(plan, "question_understanding", {}) or {}).get("analysisIntent", "")
        return str(value or "")
    except Exception:
        return ""


def retrieval_context_from_state(state: AgentState) -> Dict[str, Any]:
    question = str(state.get("question") or "")
    plan = state.get("plan")
    topics, metrics, time_windows = plan_memory_terms(plan)
    route_slots = state.get("route_slots")
    route_payload = route_slots.model_dump(by_alias=True) if hasattr(route_slots, "model_dump") else (route_slots or {})
    if isinstance(route_payload, dict):
        for candidate in route_payload.get("topicCandidates") or []:
            topic = str((candidate or {}).get("topic") or "")
            if topic and topic not in topics:
                topics.append(topic)
        days = int(((route_payload.get("timeWindow") or {}).get("days") or 0) if route_payload.get("timeWindow") else 0)
        if days and days not in time_windows:
            time_windows.append(days)
    eval_context = state.get("memory_eval_context") if isinstance(state.get("memory_eval_context"), dict) else {}
    for topic in unique_strings((eval_context or {}).get("topics") or []):
        if topic not in topics:
            topics.append(topic)
    for metric in unique_strings((eval_context or {}).get("metrics") or []):
        if metric not in metrics:
            metrics.append(metric)
    for days in unique_ints((eval_context or {}).get("timeWindows") or (eval_context or {}).get("time_windows") or []):
        if days not in time_windows:
            time_windows.append(days)
    return {
        "question": question,
        "terms": set(re.findall(r"[\w\u4e00-\u9fff]{2,}", question)),
        "topics": set(topics),
        "metrics": set(metrics).union(extract_metric_like_terms(question)),
        "timeWindows": set(time_windows or extract_time_windows(question)),
        "analysisIntent": analysis_intent_from_plan(plan),
        "objectRefs": route_payload.get("objectRefs", {}) if isinstance(route_payload, dict) else {},
        "accessRole": str(state.get("access_role") or "merchant_analyst"),
    }


def rank_memory_candidates(memory: Dict[str, Any], context: Dict[str, Any]) -> Tuple[List[MemoryRetrievalCandidate], Dict[str, int]]:
    candidates: List[MemoryRetrievalCandidate] = []
    filtered_reasons: Dict[str, int] = defaultdict(int)
    for group_name, memory_id_key, items in [
        ("event", "eventId", memory.get("events") or []),
        ("preference", "preferenceId", memory.get("preferences") or []),
        ("fact", "factId", memory.get("facts") or []),
    ]:
        for item in items:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get(memory_id_key) or "")
            expired = is_memory_expired(item)
            invalid = float(item.get("confidence") or 0) <= 0.05
            pending = memory_is_pending(item)
            inactive = memory_is_inactive(item)
            permitted = memory_visible_to_role(item, str(context.get("accessRole") or "merchant_analyst"))
            if expired or invalid or pending or inactive or not permitted:
                reason = (
                    "expired"
                    if expired
                    else (
                        "low_confidence"
                        if invalid
                        else ("pending_governance" if pending else ("deleted" if memory_status(item) == "deleted" else ("inactive" if inactive else "role_filtered")))
                    )
                )
                filtered_reasons[reason] += 1
                candidates.append(
                    MemoryRetrievalCandidate(
                        memory_id=memory_id,
                        memory_type=str(item.get("memoryType") or group_name),
                        filtered=True,
                        filter_reason=reason,
                        payload=compact_memory_payload(item, group_name),
                    )
                )
                continue
            score, reasons = memory_relevance_score(item, context, group_name)
            context_match_required = memory_context_match_required(item)
            if (context_match_required and not memory_has_contextual_match(reasons)) or (score <= 0.25 and str(item.get("memoryType") or "") != "correction"):
                filtered_reasons["not_relevant"] += 1
                candidates.append(
                    MemoryRetrievalCandidate(
                        memory_id=memory_id,
                        memory_type=str(item.get("memoryType") or group_name),
                        score=round(score, 4),
                        reasons=reasons,
                        filtered=True,
                        filter_reason="not_relevant",
                        payload=compact_memory_payload(item, group_name),
                    )
                )
                continue
            candidates.append(
                MemoryRetrievalCandidate(
                    memory_id=memory_id,
                    memory_type=str(item.get("memoryType") or group_name),
                    score=round(score, 4),
                    reasons=reasons,
                    payload=compact_memory_payload(item, group_name),
                )
            )
    candidates.sort(key=lambda item: (item.filtered is False, item.score), reverse=True)
    return diversify_memory_candidates(candidates), dict(filtered_reasons)


def memory_relevance_score(item: Dict[str, Any], context: Dict[str, Any], group_name: str) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    text = json.dumps(item, ensure_ascii=False, default=str)
    score = 0.0
    topic_overlap = set(unique_strings(item.get("topics") or [])) & set(context.get("topics") or set())
    metric_overlap = set(unique_strings(item.get("metrics") or [])) & set(context.get("metrics") or set())
    time_overlap = set(unique_ints(item.get("timeWindows") or [])) & set(context.get("timeWindows") or set())
    if topic_overlap:
        score += 3.0 + len(topic_overlap)
        reasons.append("topic_overlap:%s" % ",".join(sorted(topic_overlap))[:80])
    if metric_overlap:
        score += 4.0 + len(metric_overlap) * 1.5
        reasons.append("metric_overlap:%s" % ",".join(sorted(metric_overlap))[:80])
    if time_overlap:
        score += 1.5
        reasons.append("time_window_match")
    term_hits = [term for term in context.get("terms", set()) if term and term in text]
    if term_hits:
        score += min(3.0, len(term_hits) * 0.35)
        reasons.append("text_match:%d" % len(term_hits))
    if str(item.get("analysisIntent") or "") and item.get("analysisIntent") == context.get("analysisIntent"):
        score += 0.8
        reasons.append("analysis_intent_match")
    decay = memory_decay_score(item)
    confidence = float(item.get("confidence") or 0.0)
    score += confidence * 2.0 + decay
    if decay < 0.35:
        reasons.append("old_memory_decay")
    else:
        reasons.append("freshness")
    signal = str(item.get("feedbackSignal") or "")
    if "adopted" in signal:
        score += 1.2
        reasons.append("positive_feedback")
    if "liked" in signal:
        score += 0.6
        reasons.append("liked_feedback")
    if "disliked" in signal:
        score -= 1.6
        reasons.append("negative_feedback")
    if str(item.get("memoryType") or "") == "correction":
        score += 5.0 if (topic_overlap or metric_overlap or term_hits) else 1.0
        reasons.append("correction_priority")
    if str(item.get("memoryType") or "") == "metric_dispute":
        reasons.append("metric_dispute_governance_signal")
    if str(item.get("memoryType") or "") == "past_case":
        score += 2.0 if (topic_overlap or metric_overlap or term_hits) else -0.5
        reasons.append("episodic_case_match")
    if str(item.get("memoryType") or "") == "procedure":
        score += 1.6 if (topic_overlap or metric_overlap or term_hits) else 0.2
        reasons.append("procedure_match")
    if memory_tier(item) == "core":
        score += 1.1
        reasons.append("core_memory")
    if group_name == "preference":
        score += 0.5
    return max(0.0, score), reasons


def memory_context_match_required(item: Dict[str, Any]) -> bool:
    memory_type = str((item or {}).get("memoryType") or "")
    if memory_tier(item) == "core":
        return False
    return memory_type in {"query_event", "past_case", "procedure", "negative_feedback", "feedback", "business_focus"}


def memory_has_contextual_match(reasons: List[str]) -> bool:
    contextual_prefixes = (
        "topic_overlap",
        "metric_overlap",
        "time_window_match",
        "text_match",
        "analysis_intent_match",
    )
    return any(str(reason).startswith(contextual_prefixes) for reason in reasons or [])


def candidate_memory_tier(candidate: MemoryRetrievalCandidate) -> str:
    return str(((candidate.payload or {}).get("memoryTier") or "retrieval")).strip().lower() or "retrieval"


def is_core_payload(payload: Dict[str, Any]) -> bool:
    return str((payload or {}).get("memoryTier") or "").strip().lower() == "core"


def merged_payloads_for(
    candidates: List[MemoryRetrievalCandidate],
    memory_types: set[str],
    max_items: int,
    selected_ids: List[str],
) -> List[Dict[str, Any]]:
    payloads = payloads_for(candidates, memory_types, max_items=max_items, selected_ids=selected_ids, allowed_tiers={"core"})
    if len(payloads) < max_items:
        payloads.extend(
            payloads_for(
                candidates,
                memory_types,
                max_items=max_items - len(payloads),
                selected_ids=selected_ids,
                allowed_tiers={"retrieval"},
            )
        )
    return payloads


def pop_low_priority_payload(items: List[Dict[str, Any]], min_items: int = 0) -> bool:
    if len(items) <= max(0, int(min_items or 0)):
        return False
    for index in range(len(items) - 1, -1, -1):
        if len(items) <= max(0, int(min_items or 0)):
            return False
        if not is_core_payload(items[index]):
            items.pop(index)
            return True
    return False


def pop_last_payload(items: List[Dict[str, Any]], min_items: int = 0) -> bool:
    if len(items) <= max(0, int(min_items or 0)):
        return False
    items.pop()
    return True


def allocate_injection(
    memory: Dict[str, Any],
    candidates: List[MemoryRetrievalCandidate],
    filtered_reasons: Dict[str, int],
    merchant_id: str,
    budget: int,
    source: str,
) -> Tuple[Dict[str, Any], MemoryInjectionTrace]:
    usable = [item for item in candidates if not item.filtered]
    selected_ids: List[str] = []
    corrections = merged_payloads_for(usable, {"correction"}, max_items=3, selected_ids=selected_ids)
    metric_disputes = payloads_for(usable, {"metric_dispute"}, max_items=2, selected_ids=selected_ids)
    preferences = merged_payloads_for(usable, {"user_preference", "business_focus", "metric_habit", "time_window_habit", "preference"}, max_items=4, selected_ids=selected_ids)
    facts = merged_payloads_for(usable, {"fact", "business_fact"}, max_items=4, selected_ids=selected_ids)
    events = payloads_for(usable, {"query_event", "business_focus", "negative_feedback", "feedback"}, max_items=6, selected_ids=selected_ids)
    past_cases = payloads_for(usable, {"past_case"}, max_items=3, selected_ids=selected_ids)
    procedures = payloads_for(usable, {"procedure"}, max_items=3, selected_ids=selected_ids)
    candidate_memories = candidate_payloads_for(candidates, max_items=4)
    core_memory = compact_core_memory_profile(memory.get("coreMemoryProfile") or build_core_memory_profile(memory))
    selected = {
        "merchantId": merchant_id,
        "recentFocus": memory.get("recentFocus") or {},
        "coreMemory": core_memory,
        "relevantCorrections": corrections,
        "relevantMetricDisputes": metric_disputes,
        "relevantPreferences": preferences,
        "relevantFacts": facts,
        "relevantEvents": events,
        "relevantPastCases": past_cases,
        "relevantProcedures": procedures,
        "candidateMemories": candidate_memories,
        "preferences": preferences,
        "facts": facts,
        "source": source,
    }
    truncated = False
    while memory_injection_tokens(selected) > budget and events:
        events.pop()
        truncated = True
    while memory_injection_tokens(selected) > budget and candidate_memories:
        candidate_memories.pop()
        truncated = True
    while memory_injection_tokens(selected) > budget and preferences and pop_low_priority_payload(preferences, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and facts and pop_low_priority_payload(facts, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and procedures:
        procedures.pop()
        truncated = True
    while memory_injection_tokens(selected) > budget and metric_disputes:
        metric_disputes.pop()
        truncated = True
    while memory_injection_tokens(selected) > budget and past_cases:
        past_cases.pop()
        truncated = True
    while memory_injection_tokens(selected) > budget and len(corrections) > 1 and pop_low_priority_payload(corrections, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and len(preferences) > 1 and pop_last_payload(preferences, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and len(facts) > 1 and pop_last_payload(facts, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and len(corrections) > 1 and pop_last_payload(corrections, min_items=1):
        truncated = True
    while memory_injection_tokens(selected) > budget and events:
        events.pop()
        truncated = True
    selected["truncated"] = truncated
    selected_ids = memory_ids_from_selected(selected)
    core_selected_ids = [
        *[str(item.get("id") or "") for item in corrections if is_core_payload(item) and item.get("id")],
        *[str(item.get("id") or "") for item in preferences if is_core_payload(item) and item.get("id")],
        *[str(item.get("id") or "") for item in facts if is_core_payload(item) and item.get("id")],
    ]
    trace = MemoryInjectionTrace(
        merchant_id=merchant_id,
        budget_tokens=budget,
        budget_chars=budget * 4,
        candidate_count=len(candidates),
        injected_event_count=len(events),
        injected_preference_count=len(preferences),
        injected_correction_count=len(corrections),
        injected_fact_count=len(facts),
        past_case_count=len(past_cases),
        budget_used_tokens=memory_injection_tokens(selected),
        budget_used_chars=memory_injection_chars(selected),
        truncated=truncated,
        selected_ids=selected_ids,
        candidate_ids=[str(item.get("id") or "") for item in candidate_memories if item.get("id")],
        core_memory_count=len(core_selected_ids),
        retrieval_memory_count=len([memory_id for memory_id in selected_ids if memory_id not in set(core_selected_ids)]),
        core_selected_ids=unique_strings(core_selected_ids),
        filtered_reasons=filtered_reasons,
        candidates=candidates[:20],
    )
    return selected, trace


def payloads_for(
    candidates: List[MemoryRetrievalCandidate],
    memory_types: set[str],
    max_items: int,
    selected_ids: List[str],
    allowed_tiers: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        memory_type = str(candidate.memory_type or "")
        normalized_type = "fact" if memory_type in {"business_fact"} else memory_type
        if normalized_type not in memory_types:
            continue
        if allowed_tiers and candidate_memory_tier(candidate) not in allowed_tiers:
            continue
        if candidate.memory_id in seen or candidate.memory_id in selected_ids:
            continue
        payload = dict(candidate.payload)
        payload["score"] = candidate.score
        payload["hitReasons"] = list(candidate.reasons or [])
        payloads.append(payload)
        selected_ids.append(candidate.memory_id)
        seen.add(candidate.memory_id)
        if len(payloads) >= max_items:
            break
    return payloads


def candidate_payloads_for(candidates: List[MemoryRetrievalCandidate], max_items: int) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.filter_reason != "pending_governance":
            continue
        if candidate.memory_id in seen:
            continue
        payload = dict(candidate.payload)
        payload["score"] = candidate.score
        payload["hitReasons"] = list(candidate.reasons or [])
        payload["candidateReason"] = "pending memory; use only as review signal, not semantic truth"
        payloads.append(payload)
        seen.add(candidate.memory_id)
        if len(payloads) >= max_items:
            break
    return payloads


def compact_memory_payload(item: Dict[str, Any], group_name: str) -> Dict[str, Any]:
    memory_id = str(item.get("eventId") or item.get("preferenceId") or item.get("factId") or "")
    payload = {
        "id": memory_id,
        "memoryType": item.get("memoryType") or group_name,
        "memoryTier": memory_tier(item),
        "memoryClass": memory_class(item),
        "topics": item.get("topics") or [],
        "metrics": item.get("metrics") or [],
        "timeWindows": item.get("timeWindows") or [],
        "confidence": item.get("confidence"),
        "source": item.get("source", ""),
        "status": memory_status(item),
        "scope": memory_scope_payload(item),
        "visibility": str(item.get("visibility") or "merchant"),
        "allowedRoles": unique_strings(item.get("allowedRoles") or []),
        "retentionDays": int(item.get("retentionDays") or 0),
        "approvedBy": item.get("approvedBy", ""),
        "evidenceRefs": unique_strings(item.get("evidenceRefs") or []),
        "createdAt": item.get("createdAt", ""),
        "hitCount": item.get("hitCount", 0),
    }
    if str(payload.get("memoryType") or "") in {"past_case", "procedure"}:
        payload["casePayload"] = compact_case_payload(memory_case_payload(item))
        if item.get("caseSummary"):
            payload["caseSummary"] = str(item.get("caseSummary") or "")[:500]
    for key in ["question", "answerPreview", "content", "key", "value", "correctionText", "feedbackSignal"]:
        if item.get(key):
            payload[key] = str(item.get(key))[:400]
    if str(payload.get("memoryType") or "") == "metric_dispute":
        payload["governanceInstruction"] = "口径争议信号，不覆盖语义层/指标中心标准定义；回答时应澄清或提示指标治理确认。"
    return payload


def compact_case_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "caseStatus": str(payload.get("caseStatus") or ""),
        "intentCount": int(payload.get("intentCount") or 0),
        "dependencyCount": int(payload.get("dependencyCount") or 0),
        "semanticRefIds": unique_strings(payload.get("semanticRefIds") or [])[:12],
        "recallRefs": unique_strings(payload.get("recallRefs") or [])[:12],
        "validationGaps": unique_strings(payload.get("validationGaps") or [])[:8],
        "evidenceGaps": unique_strings(payload.get("evidenceGaps") or [])[:8],
        "repairActions": unique_strings(payload.get("repairActions") or [])[:8],
        "answerWithGap": bool(payload.get("answerWithGap")),
        "sqlSummaries": [
            item
            for item in (payload.get("sqlSummaries") or [])[:8]
            if isinstance(item, dict)
        ],
    }


def record_memory_usage(memory: Dict[str, Any], selected_ids: List[str]) -> None:
    now = datetime.now().isoformat()
    selected = set(selected_ids)
    for group in ["events", "preferences", "facts"]:
        for item in memory.get(group) or []:
            memory_id = str(item.get("eventId") or item.get("preferenceId") or item.get("factId") or "")
            if memory_id in selected:
                item["hitCount"] = int(item.get("hitCount") or 0) + 1
                item["lastUsedAt"] = now
                item["decayScore"] = round(memory_decay_score(item), 4)


def upsert_habit_preferences(memory: Dict[str, Any], event: Dict[str, Any]) -> int:
    preferences = [item for item in memory.get("preferences", []) if isinstance(item, dict)]
    updates = 0
    for topic in unique_strings(event.get("topics") or []):
        pref = MemoryPreference(
            preference_id="pref_topic_%s" % stable_slug(topic),
            memory_type="business_focus",
            memory_tier=habit_memory_tier_for_event(event, 1),
            memory_class=default_memory_class("business_focus"),
            key="topic:%s" % topic,
            value=topic,
            topics=[topic],
            metrics=[],
            confidence=min(0.95, float(event.get("confidence") or 0.5) + 0.1),
            source=event.get("source") or "answer_run",
            hit_count=1,
            scope=memory_scope_payload(event),
            status=memory_status(event),
            retention_days=default_retention_days("business_focus"),
            visibility=default_memory_visibility("business_focus"),
            allowed_roles=default_memory_allowed_roles("business_focus"),
            approved_by=str(event.get("approvedBy") or ""),
            evidence_refs=unique_strings(event.get("evidenceRefs") or []),
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    for metric in unique_strings(event.get("metrics") or []):
        pref = MemoryPreference(
            preference_id="pref_metric_%s" % stable_slug(metric),
            memory_type="metric_habit",
            memory_tier=habit_memory_tier_for_event(event, 1),
            memory_class=default_memory_class("metric_habit"),
            key="metric:%s" % metric,
            value=metric,
            topics=unique_strings(event.get("topics") or []),
            metrics=[metric],
            confidence=min(0.95, float(event.get("confidence") or 0.5) + 0.1),
            source=event.get("source") or "answer_run",
            hit_count=1,
            scope=memory_scope_payload(event),
            status=memory_status(event),
            retention_days=default_retention_days("metric_habit"),
            visibility=default_memory_visibility("metric_habit"),
            allowed_roles=default_memory_allowed_roles("metric_habit"),
            approved_by=str(event.get("approvedBy") or ""),
            evidence_refs=unique_strings(event.get("evidenceRefs") or []),
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    for days in unique_ints(event.get("timeWindows") or []):
        pref = MemoryPreference(
            preference_id="pref_window_%s" % days,
            memory_type="time_window_habit",
            memory_tier=habit_memory_tier_for_event(event, 1),
            memory_class=default_memory_class("time_window_habit"),
            key="timeWindow:%s" % days,
            value="%s天" % days,
            topics=unique_strings(event.get("topics") or []),
            metrics=unique_strings(event.get("metrics") or []),
            confidence=min(0.9, float(event.get("confidence") or 0.5) + 0.05),
            source=event.get("source") or "answer_run",
            hit_count=1,
            scope=memory_scope_payload(event),
            status=memory_status(event),
            retention_days=default_retention_days("time_window_habit"),
            visibility=default_memory_visibility("time_window_habit"),
            allowed_roles=default_memory_allowed_roles("time_window_habit"),
            approved_by=str(event.get("approvedBy") or ""),
            evidence_refs=unique_strings(event.get("evidenceRefs") or []),
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    memory["preferences"] = preferences[-MAX_PREFERENCES:]
    return updates


def habit_memory_tier_for_event(event: Dict[str, Any], hit_count: int) -> str:
    if int(hit_count or 0) >= HABIT_CORE_PROMOTION_HIT_COUNT:
        return "core"
    if event_promotes_habit_to_core(event):
        return "core"
    return "retrieval"


def event_promotes_habit_to_core(event: Dict[str, Any]) -> bool:
    memory_type = str((event or {}).get("memoryType") or "")
    if memory_type == "correction":
        return True
    if positive_feedback_signal(str((event or {}).get("feedbackSignal") or "")):
        return True
    return habit_text_is_explicit(event)


def upsert_preference(items: List[Dict[str, Any]], preference: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    now = datetime.now().isoformat()
    for item in items:
        if item.get("preferenceId") == preference.get("preferenceId") or item.get("key") == preference.get("key"):
            item["confidence"] = round(max(float(item.get("confidence") or 0), float(preference.get("confidence") or 0)), 4)
            item["topics"] = unique_strings(list(item.get("topics") or []) + list(preference.get("topics") or []))
            item["metrics"] = unique_strings(list(item.get("metrics") or []) + list(preference.get("metrics") or []))
            item["hitCount"] = int(item.get("hitCount") or 0) + 1
            item["lastUsedAt"] = now
            item["decayScore"] = round(memory_decay_score(item), 4)
            if memory_tier(preference) == "core" or int(item.get("hitCount") or 0) >= HABIT_CORE_PROMOTION_HIT_COUNT:
                item["memoryTier"] = "core"
            elif memory_tier(item) != "core":
                item["memoryTier"] = "retrieval"
            return items, True
    items.append(preference)
    return items, True


def correction_fact_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    text = str(event.get("correctionText") or event.get("question") or "")[:1000]
    fact = MemoryFact(
        fact_id="fact_correction_%s" % stable_slug(text[:80]),
        memory_type="business_fact",
        memory_tier=default_memory_tier("business_fact", memory_status(event), 0.95),
        memory_class=default_memory_class("business_fact"),
        content=text,
        topics=unique_strings(event.get("topics") or []),
        metrics=unique_strings(event.get("metrics") or []),
        confidence=0.95,
        source="correction",
        scope=memory_scope_payload(event),
        status=memory_status(event),
        retention_days=default_retention_days("business_fact"),
        visibility=default_memory_visibility("business_fact"),
        allowed_roles=default_memory_allowed_roles("business_fact"),
        approved_by=str(event.get("approvedBy") or ""),
        evidence_refs=unique_strings(event.get("evidenceRefs") or []),
        created_at=datetime.now().isoformat(),
    )
    return fact.model_dump(by_alias=True)


def upsert_fact(items: List[Dict[str, Any]], fact: Dict[str, Any]) -> List[Dict[str, Any]]:
    for item in items:
        if item.get("factId") == fact.get("factId") or item.get("content") == fact.get("content"):
            item["confidence"] = max(float(item.get("confidence") or 0), float(fact.get("confidence") or 0))
            item["hitCount"] = int(item.get("hitCount") or 0) + 1
            item["lastUsedAt"] = datetime.now().isoformat()
            return items[-MAX_FACTS:]
    items.append(fact)
    return items[-MAX_FACTS:]


def resolve_memory_conflicts(memory: Dict[str, Any], event: Dict[str, Any]) -> Optional[MemoryConflictResolution]:
    event_id = str(event.get("eventId") or "")
    scope_terms = set(unique_strings(event.get("topics") or [])) | set(unique_strings(event.get("metrics") or []))
    correction_terms = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", str(event.get("correctionText") or event.get("question") or "")))
    all_terms = scope_terms | correction_terms
    losers: List[str] = []
    for group in ["events", "preferences", "facts"]:
        for item in memory.get(group) or []:
            item_id = str(item.get("eventId") or item.get("preferenceId") or item.get("factId") or "")
            if not item_id or item_id == event_id:
                continue
            text = json.dumps(item, ensure_ascii=False, default=str)
            overlaps = bool(scope_terms & (set(unique_strings(item.get("topics") or [])) | set(unique_strings(item.get("metrics") or []))))
            term_hit = any(term in text for term in all_terms if term)
            if overlaps or term_hit:
                item["confidence"] = round(min(float(item.get("confidence") or 0.5), 0.35), 4)
                item["validUntil"] = item.get("validUntil") or datetime.now().isoformat()
                conflicts = unique_strings(item.get("conflictsWith") or [])
                if event_id not in conflicts:
                    conflicts.append(event_id)
                item["conflictsWith"] = conflicts
                losers.append(item_id)
    if not losers:
        return None
    event["supersedes"] = unique_strings(list(event.get("supersedes") or []) + losers)
    conflict = MemoryConflictResolution(
        conflict_id="conflict_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        winner_id=event_id,
        loser_id=",".join(losers[:12]),
        reason="explicit user correction has higher priority than historical memory",
        action="lower_confidence_and_mark_conflict",
        created_at=datetime.now().isoformat(),
    )
    memory.setdefault("conflicts", []).append(conflict.model_dump(by_alias=True))
    memory["conflicts"] = memory["conflicts"][-80:]
    return conflict


def reduce_related_memory(memory: Dict[str, Any], event: Dict[str, Any], reason: str) -> None:
    topics = set(unique_strings(event.get("topics") or []))
    metrics = set(unique_strings(event.get("metrics") or []))
    for group in ["events", "preferences", "facts"]:
        for item in memory.get(group) or []:
            if item.get("eventId") == event.get("eventId"):
                continue
            overlaps = bool((topics & set(unique_strings(item.get("topics") or []))) or (metrics & set(unique_strings(item.get("metrics") or []))))
            if overlaps:
                item["confidence"] = round(max(0.1, float(item.get("confidence") or 0.5) - 0.25), 4)
                item["conflictsWith"] = unique_strings(list(item.get("conflictsWith") or []) + [str(event.get("eventId") or "")])
    memory.setdefault("conflicts", []).append(
        MemoryConflictResolution(
            conflict_id="conflict_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
            winner_id=str(event.get("eventId") or ""),
            reason=reason,
            action="lower_related_memory_confidence",
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
    )


def aggregate_recent_focus(events: Iterable[Dict[str, Any]], preferences: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    topic_counter: Counter[str] = Counter()
    metric_counter: Counter[str] = Counter()
    days_counter: Counter[int] = Counter()
    for event in list(events)[-120:]:
        weight = weighted_memory_value(event)
        for topic in unique_strings(event.get("topics") or []):
            topic_counter[topic] += weight
        for metric in unique_strings(event.get("metrics") or []):
            metric_counter[metric] += weight
        for days in unique_ints(event.get("timeWindows") or []):
            days_counter[days] += weight
    for pref in preferences:
        weight = weighted_memory_value(pref) * 0.7
        for topic in unique_strings(pref.get("topics") or []):
            topic_counter[topic] += weight
        for metric in unique_strings(pref.get("metrics") or []):
            metric_counter[metric] += weight
        if str(pref.get("memoryType") or "") == "time_window_habit":
            for days in extract_time_windows(str(pref.get("value") or "")):
                days_counter[days] += weight
    return {
        "topTopics": [{"topic": key, "score": round(value, 4), "count": round(value, 4)} for key, value in topic_counter.most_common(6)],
        "topMetrics": [{"metric": key, "score": round(value, 4), "count": round(value, 4)} for key, value in metric_counter.most_common(8)],
        "commonTimeWindows": [{"days": key, "score": round(value, 4), "count": round(value, 4)} for key, value in days_counter.most_common(5)],
        "summary": build_focus_summary(topic_counter, metric_counter, days_counter),
        "updatedBy": "decay_and_feedback_weighted_aggregation",
    }


def build_focus_summary(topic_counter: Counter[str], metric_counter: Counter[str], days_counter: Counter[int]) -> str:
    parts: List[str] = []
    if topic_counter:
        parts.append("关注业务域：" + "、".join(key for key, _ in topic_counter.most_common(3)))
    if metric_counter:
        parts.append("常查指标：" + "、".join(key for key, _ in metric_counter.most_common(4)))
    if days_counter:
        parts.append("常用时间窗：" + "、".join("%d天" % key for key, _ in days_counter.most_common(3)))
    return "；".join(parts)


def rank_memory_events(events: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    context = {"question": question, "terms": set(re.findall(r"[\w\u4e00-\u9fff]{2,}", question or "")), "topics": set(), "metrics": extract_metric_like_terms(question), "timeWindows": set(extract_time_windows(question)), "analysisIntent": ""}
    scored = []
    for index, event in enumerate(events):
        score, _ = memory_relevance_score(event, context, "event")
        if score > 0.25 or index >= len(events) - 8:
            scored.append((score, index, event))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [event for _, _, event in scored]


def event_fingerprint(event: Dict[str, Any]) -> str:
    return "%s|%s|%s|%s" % (
        event.get("question", "")[:120],
        ",".join(event.get("metrics") or []),
        ",".join(str(item) for item in event.get("timeWindows") or []),
        event.get("memoryType", ""),
    )


def merchant_id_from_state(state: AgentState, settings: Settings) -> str:
    return str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or settings.merchant_id)


def default_confidence(memory_type: str, feedback_signal: str = "") -> float:
    if memory_type == "correction":
        return 0.95
    if memory_type == "metric_dispute":
        return 0.45
    if "adopted" in feedback_signal:
        return 0.88
    if "liked" in feedback_signal:
        return 0.78
    if memory_type == "negative_feedback" or "disliked" in feedback_signal:
        return 0.28
    if memory_type == "business_focus":
        return 0.65
    return 0.55


def weighted_memory_value(item: Dict[str, Any]) -> float:
    value = float(item.get("confidence") or 0.5) * memory_decay_score(item)
    signal = str(item.get("feedbackSignal") or "")
    if "adopted" in signal:
        value *= 1.6
    if "liked" in signal:
        value *= 1.25
    if "disliked" in signal:
        value *= 0.35
    if str(item.get("memoryType") or "") == "correction":
        value *= 1.8
    value *= 1 + min(0.4, int(item.get("hitCount") or 0) * 0.04)
    return max(0.0, value)


def memory_decay_score(item: Dict[str, Any]) -> float:
    created = parse_datetime(str(item.get("createdAt") or item.get("lastUsedAt") or ""))
    if not created:
        return 0.7
    days = max(0.0, (datetime.now() - created).total_seconds() / 86400.0)
    half_life = 30.0
    if str(item.get("memoryType") or "") == "correction":
        half_life = 120.0
    score = math.pow(0.5, days / half_life)
    return round(max(0.05, min(1.0, score)), 4)


def is_memory_expired(item: Dict[str, Any]) -> bool:
    valid_until = str(item.get("validUntil") or "")
    if valid_until:
        parsed = parse_datetime(valid_until)
        if parsed and parsed < datetime.now():
            return True
    retention_days = int(item.get("retentionDays") or item.get("retention_days") or 0)
    if retention_days > 0:
        created = parse_datetime(str(item.get("createdAt") or item.get("lastUsedAt") or ""))
        if created and created < datetime.now() - timedelta(days=retention_days):
            return True
    return False


def memory_visible_to_role(item: Dict[str, Any], access_role: str) -> bool:
    visibility = str(item.get("visibility") or "merchant").strip().lower()
    if visibility in {"public", "merchant"}:
        return True
    roles = unique_strings(item.get("allowedRoles") or item.get("allowed_roles") or [])
    if not roles:
        return visibility != "planner_only" or str(access_role or "").strip() in {"merchant_admin", "merchant_analyst", "system"}
    return str(access_role or "merchant_analyst") in roles


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def extract_correction_text(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    correction_markers = ["不是", "不对", "错了", "纠正", "应该是", "改成", "以后按", "以后都", "别再"]
    if any(marker in text for marker in correction_markers):
        return text[:500]
    return ""


def is_metric_definition_dispute(question: str, metrics: Iterable[Any] = ()) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    metric_terms = set(unique_strings(metrics)) | extract_metric_like_terms(text)
    has_metric_context = bool(metric_terms) or any(marker in text for marker in ["指标", "口径", "GMV", "gmv"])
    if not has_metric_context:
        return False
    dispute_markers = [
        "口径",
        "公式",
        "算法",
        "指标定义",
        "统计口径",
        "计算口径",
        "不是这么算",
        "不该这么算",
        "不应该这么算",
        "怎么算",
        "分母",
        "分子",
        "除以",
        "剔除",
        "包含",
        "不包含",
        "来源字段",
    ]
    if any(marker in text for marker in dispute_markers):
        return True
    return bool(re.search(r"(率|金额|订单量|GMV|gmv).{0,16}(=|/|÷|占比|比例)", text))


def extract_metric_like_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or ""))
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", text or ""):
        if any(marker in token for marker in ["金额", "订单", "退款", "退货", "工单", "赔付", "GMV", "gmv", "下单", "优惠", "商品", "审核", "入库", "发货", "超时", "率", "量"]):
            terms.add(token)
    return terms


def extract_time_windows(text: str) -> List[int]:
    days: List[int] = []
    for match in re.findall(r"最近\s*(\d+)\s*天", text or ""):
        try:
            days.append(int(match))
        except Exception:
            pass
    if "最近一周" in text or "近一周" in text:
        days.append(7)
    if "最近一个月" in text or "近一个月" in text:
        days.append(30)
    return sorted(set(days))


def unique_strings(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items or []:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def unique_ints(items: Iterable[Any]) -> List[int]:
    seen: set[int] = set()
    result: List[int] = []
    for item in items or []:
        try:
            value = int(item)
        except Exception:
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def stable_slug(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip())
    return safe[:80] or str(abs(hash(value)))


def memory_ids_from_selected(selected: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in [
        "relevantCorrections",
        "relevantMetricDisputes",
        "relevantPreferences",
        "relevantFacts",
        "relevantEvents",
        "relevantPastCases",
        "relevantProcedures",
    ]:
        for item in selected.get(key) or []:
            memory_id = str(item.get("id") or "")
            if memory_id and memory_id not in ids:
                ids.append(memory_id)
    return ids
