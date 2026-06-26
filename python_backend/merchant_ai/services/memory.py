from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.models import (
    MemoryConflictResolution,
    MemoryEvent,
    MemoryFact,
    MemoryInjectionTrace,
    MemoryPreference,
    MemoryRetrievalCandidate,
    PendingAnswer,
)


MEMORY_SCHEMA_VERSION = "merchant_memory.v2"
MAX_EVENTS = 240
MAX_PREFERENCES = 160
MAX_FACTS = 120


class MemoryStore:
    """Adapter boundary for long-term memory storage."""

    def load(self, merchant_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def save(self, merchant_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def select_for_question(self, state: AgentState, budget_chars: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

    def update_from_state(self, state: AgentState) -> Dict[str, Any]:
        raise NotImplementedError


class StructuredMemoryStore(MemoryStore):
    """Governed local long-term memory for merchant BI context."""

    def __init__(self, settings: Settings):
        self.settings = settings

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

    def select_for_question(self, state: AgentState, budget_chars: int = 0) -> Dict[str, Any]:
        merchant_id = merchant_id_from_state(state, self.settings)
        memory = self.load(merchant_id)
        context = retrieval_context_from_state(state)
        budget = max(400, int(budget_chars or self.settings.context_memory_budget_chars or 1800))
        candidates, filtered_reasons = rank_memory_candidates(memory, context)
        selected, trace = allocate_injection(memory, candidates, filtered_reasons, merchant_id, budget, str(self.memory_path(merchant_id)))
        selected["source"] = str(self.memory_path(merchant_id))
        selected["updatedAt"] = memory.get("updatedAt", "")
        selected["memoryInjectionTrace"] = trace.model_dump(by_alias=True)
        if trace.selected_ids:
            record_memory_usage(memory, trace.selected_ids)
            self.save(merchant_id, memory)
        return selected

    def render_injection(self, payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        renderable = {
            "merchantId": payload.get("merchantId", ""),
            "recentFocus": payload.get("recentFocus", {}),
            "relevantCorrections": payload.get("relevantCorrections", []),
            "relevantPreferences": payload.get("relevantPreferences", []),
            "relevantFacts": payload.get("relevantFacts", []),
            "relevantEvents": payload.get("relevantEvents", []),
            "truncated": bool(payload.get("truncated")),
        }
        if not any(renderable.get(key) for key in ["recentFocus", "relevantCorrections", "relevantPreferences", "relevantFacts", "relevantEvents"]):
            return ""
        return json.dumps(renderable, ensure_ascii=False, default=str, indent=2)

    def update_from_state(self, state: AgentState) -> Dict[str, Any]:
        merchant_id = merchant_id_from_state(state, self.settings)
        memory = self.load(merchant_id)
        event = memory_event_from_state(state)
        if not event.get("question"):
            return memory
        ingestion_trace = {"eventId": event.get("eventId"), "memoryType": event.get("memoryType"), "written": False, "preferenceUpdates": 0}
        events = [item for item in memory.get("events", []) if isinstance(item, dict)]
        if not events or event_fingerprint(events[-1]) != event_fingerprint(event):
            events.append(event)
            ingestion_trace["written"] = True
        events = events[-MAX_EVENTS:]
        memory["events"] = events
        preference_updates = upsert_habit_preferences(memory, event)
        ingestion_trace["preferenceUpdates"] = preference_updates
        if event.get("memoryType") == "correction":
            fact = correction_fact_from_event(event)
            memory["facts"] = upsert_fact(memory.get("facts") or [], fact)
            conflict = resolve_memory_conflicts(memory, event)
            if conflict:
                ingestion_trace["conflict"] = conflict.model_dump(by_alias=True)
        memory["recentFocus"] = aggregate_recent_focus(memory.get("events") or [], memory.get("preferences") or [])
        memory["memoryIngestionTrace"] = ingestion_trace
        saved = self.save(merchant_id, memory)
        saved["memoryIngestionTrace"] = ingestion_trace
        return saved

    def update_from_feedback(self, pending: Optional[PendingAnswer], adopted: Any = None, liked: Any = None, disliked: Any = None) -> Dict[str, Any]:
        if not pending:
            return {}
        merchant_id = pending.merchant_id or self.settings.merchant_id
        memory = self.load(merchant_id)
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
        return self.save(merchant_id, memory)

    def empty_memory(self, merchant_id: str) -> Dict[str, Any]:
        return {
            "merchantId": merchant_id or self.settings.merchant_id,
            "recentFocus": {},
            "preferences": [],
            "facts": [],
            "events": [],
            "conflicts": [],
            "memoryIngestionTrace": {},
            "updatedAt": "",
            "schemaVersion": MEMORY_SCHEMA_VERSION,
        }


def normalize_memory(payload: Dict[str, Any], merchant_id: str) -> Dict[str, Any]:
    memory = {
        "merchantId": payload.get("merchantId") or merchant_id,
        "recentFocus": payload.get("recentFocus") or {},
        "preferences": normalize_preferences(payload.get("preferences") or []),
        "facts": normalize_facts(payload.get("facts") or []),
        "events": normalize_events(payload.get("events") or []),
        "conflicts": [item for item in payload.get("conflicts", []) if isinstance(item, dict)],
        "memoryIngestionTrace": payload.get("memoryIngestionTrace") or {},
        "updatedAt": payload.get("updatedAt") or "",
        "schemaVersion": MEMORY_SCHEMA_VERSION,
    }
    memory["events"] = memory["events"][-MAX_EVENTS:]
    memory["preferences"] = memory["preferences"][-MAX_PREFERENCES:]
    memory["facts"] = memory["facts"][-MAX_FACTS:]
    if not memory["recentFocus"]:
        memory["recentFocus"] = aggregate_recent_focus(memory["events"], memory["preferences"])
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
            supersedes=unique_strings(item.get("supersedes") or []),
            conflicts_with=unique_strings(item.get("conflictsWith") or item.get("conflicts_with") or []),
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
        )
        normalized.append(event.model_dump(by_alias=True))
    return normalized


def normalize_preferences(items: Any) -> List[Dict[str, Any]]:
    if isinstance(items, dict):
        next_items = []
        for key, value in items.items():
            next_items.append({"key": str(key), "value": str(value), "memoryType": "user_preference"})
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
            memory_type=str(item.get("memoryType") or item.get("memory_type") or "user_preference"),
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
            memory_type=str(item.get("memoryType") or item.get("memory_type") or "business_focus"),
            content=str(item.get("content") or item.get("text") or "")[:1000],
            topics=unique_strings(item.get("topics") or []),
            metrics=unique_strings(item.get("metrics") or []),
            confidence=float(item.get("confidence") or 0.6),
            source=str(item.get("source") or "memory"),
            hit_count=int(item.get("hitCount") or item.get("hit_count") or 0),
            last_used_at=str(item.get("lastUsedAt") or item.get("last_used_at") or ""),
            decay_score=float(item.get("decayScore") or item.get("decay_score") or 1.0),
            valid_until=str(item.get("validUntil") or item.get("valid_until") or ""),
            supersedes=unique_strings(item.get("supersedes") or []),
            conflicts_with=unique_strings(item.get("conflictsWith") or item.get("conflicts_with") or []),
            created_at=str(item.get("createdAt") or item.get("created_at") or datetime.now().isoformat()),
        )
        if fact.content:
            normalized.append(fact.model_dump(by_alias=True))
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
    memory_type = "correction" if correction_text else ("business_focus" if analysis_intent_from_plan(plan) not in {"", "none"} else "query_event")
    feedback_signal = "persisted" if state.get("persisted") else ""
    event = MemoryEvent(
        event_id="mem_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        memory_type=memory_type,
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
        question=str(pending.question or "")[:1000],
        answer_preview=str(pending.answer or "")[:1000],
        topics=unique_strings([pending.category_name]),
        metrics=extract_metric_like_terms(pending.question),
        time_windows=extract_time_windows(pending.question),
        feedback_signal=signal,
        confidence=default_confidence(memory_type, signal),
        source="feedback",
        created_at=datetime.now().isoformat(),
    )
    return event.model_dump(by_alias=True)


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
    return {
        "question": question,
        "terms": set(re.findall(r"[\w\u4e00-\u9fff]{2,}", question)),
        "topics": set(topics),
        "metrics": set(metrics).union(extract_metric_like_terms(question)),
        "timeWindows": set(time_windows or extract_time_windows(question)),
        "analysisIntent": analysis_intent_from_plan(plan),
        "objectRefs": route_payload.get("objectRefs", {}) if isinstance(route_payload, dict) else {},
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
            if expired or invalid:
                reason = "expired" if expired else "low_confidence"
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
            if score <= 0.25 and str(item.get("memoryType") or "") != "correction":
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
    return candidates, dict(filtered_reasons)


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
    if group_name == "preference":
        score += 0.5
    return max(0.0, score), reasons


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
    corrections = payloads_for(usable, {"correction"}, max_items=3, selected_ids=selected_ids)
    preferences = payloads_for(usable, {"user_preference", "business_focus", "metric_habit", "time_window_habit"}, max_items=6, selected_ids=selected_ids)
    facts = payloads_for(usable, {"fact", "business_fact"}, max_items=4, selected_ids=selected_ids)
    events = payloads_for(usable, {"query_event", "business_focus", "negative_feedback"}, max_items=8, selected_ids=selected_ids)
    selected = {
        "merchantId": merchant_id,
        "recentFocus": memory.get("recentFocus") or {},
        "relevantCorrections": corrections,
        "relevantPreferences": preferences,
        "relevantFacts": facts,
        "relevantEvents": events,
        "preferences": preferences,
        "facts": facts,
        "source": source,
    }
    truncated = False
    while len(json.dumps(selected, ensure_ascii=False, default=str)) > budget and events:
        events.pop()
        truncated = True
    while len(json.dumps(selected, ensure_ascii=False, default=str)) > budget and preferences:
        preferences.pop()
        truncated = True
    while len(json.dumps(selected, ensure_ascii=False, default=str)) > budget and facts:
        facts.pop()
        truncated = True
    while len(json.dumps(selected, ensure_ascii=False, default=str)) > budget and len(corrections) > 1:
        corrections.pop()
        truncated = True
    selected["truncated"] = truncated
    selected_ids = memory_ids_from_selected(selected)
    trace = MemoryInjectionTrace(
        merchant_id=merchant_id,
        budget_chars=budget,
        candidate_count=len(candidates),
        injected_event_count=len(events),
        injected_preference_count=len(preferences),
        injected_correction_count=len(corrections),
        injected_fact_count=len(facts),
        budget_used_chars=len(json.dumps(selected, ensure_ascii=False, default=str)),
        truncated=truncated,
        selected_ids=selected_ids,
        filtered_reasons=filtered_reasons,
        candidates=candidates[:20],
    )
    return selected, trace


def payloads_for(candidates: List[MemoryRetrievalCandidate], memory_types: set[str], max_items: int, selected_ids: List[str]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        memory_type = str(candidate.memory_type or "")
        normalized_type = "fact" if memory_type in {"business_fact"} else memory_type
        if normalized_type not in memory_types:
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


def compact_memory_payload(item: Dict[str, Any], group_name: str) -> Dict[str, Any]:
    memory_id = str(item.get("eventId") or item.get("preferenceId") or item.get("factId") or "")
    payload = {
        "id": memory_id,
        "memoryType": item.get("memoryType") or group_name,
        "topics": item.get("topics") or [],
        "metrics": item.get("metrics") or [],
        "timeWindows": item.get("timeWindows") or [],
        "confidence": item.get("confidence"),
        "source": item.get("source", ""),
        "createdAt": item.get("createdAt", ""),
        "hitCount": item.get("hitCount", 0),
    }
    for key in ["question", "answerPreview", "content", "key", "value", "correctionText", "feedbackSignal"]:
        if item.get(key):
            payload[key] = str(item.get(key))[:400]
    return payload


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
            key="topic:%s" % topic,
            value=topic,
            topics=[topic],
            metrics=[],
            confidence=min(0.95, float(event.get("confidence") or 0.5) + 0.1),
            source=event.get("source") or "answer_run",
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    for metric in unique_strings(event.get("metrics") or []):
        pref = MemoryPreference(
            preference_id="pref_metric_%s" % stable_slug(metric),
            memory_type="metric_habit",
            key="metric:%s" % metric,
            value=metric,
            topics=unique_strings(event.get("topics") or []),
            metrics=[metric],
            confidence=min(0.95, float(event.get("confidence") or 0.5) + 0.1),
            source=event.get("source") or "answer_run",
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    for days in unique_ints(event.get("timeWindows") or []):
        pref = MemoryPreference(
            preference_id="pref_window_%s" % days,
            memory_type="time_window_habit",
            key="timeWindow:%s" % days,
            value="%s天" % days,
            topics=unique_strings(event.get("topics") or []),
            metrics=unique_strings(event.get("metrics") or []),
            confidence=min(0.9, float(event.get("confidence") or 0.5) + 0.05),
            source=event.get("source") or "answer_run",
            created_at=datetime.now().isoformat(),
        ).model_dump(by_alias=True)
        preferences, did_update = upsert_preference(preferences, pref)
        updates += int(did_update)
    memory["preferences"] = preferences[-MAX_PREFERENCES:]
    return updates


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
            return items, True
    items.append(preference)
    return items, True


def correction_fact_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    text = str(event.get("correctionText") or event.get("question") or "")[:1000]
    fact = MemoryFact(
        fact_id="fact_correction_%s" % stable_slug(text[:80]),
        memory_type="business_fact",
        content=text,
        topics=unique_strings(event.get("topics") or []),
        metrics=unique_strings(event.get("metrics") or []),
        confidence=0.95,
        source="correction",
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
    if not valid_until:
        return False
    parsed = parse_datetime(valid_until)
    return bool(parsed and parsed < datetime.now())


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
    for key in ["relevantCorrections", "relevantPreferences", "relevantFacts", "relevantEvents"]:
        for item in selected.get(key) or []:
            memory_id = str(item.get("id") or "")
            if memory_id and memory_id not in ids:
                ids.append(memory_id)
    return ids
