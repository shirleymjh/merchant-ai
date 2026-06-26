from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState


class StructuredMemoryStore:
    """Local structured long-term memory for merchant BI context."""

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
                    return payload
        except Exception:
            return self.empty_memory(merchant_id)
        return self.empty_memory(merchant_id)

    def select_for_question(self, state: AgentState, budget_chars: int = 0) -> Dict[str, Any]:
        merchant_id = str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or self.settings.merchant_id)
        memory = self.load(merchant_id)
        question = str(state.get("question") or "")
        budget = max(400, int(budget_chars or self.settings.context_memory_budget_chars or 1800))
        relevant_events = rank_memory_events(memory.get("events") or [], question)[:8]
        focus = memory.get("recentFocus") or {}
        selected = {
            "merchantId": merchant_id,
            "recentFocus": focus,
            "preferences": memory.get("preferences") or {},
            "facts": (memory.get("facts") or [])[:10],
            "relevantEvents": relevant_events,
            "updatedAt": memory.get("updatedAt", ""),
            "source": str(self.memory_path(merchant_id)),
        }
        text = json.dumps(selected, ensure_ascii=False, default=str)
        if len(text) > budget:
            selected["relevantEvents"] = relevant_events[:3]
            selected["facts"] = (memory.get("facts") or [])[:4]
            selected["truncated"] = True
        return selected

    def render_injection(self, payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        return json.dumps(payload, ensure_ascii=False, default=str, indent=2)

    def update_from_state(self, state: AgentState) -> Dict[str, Any]:
        merchant_id = str(state.get("requested_merchant_id") or getattr(state.get("merchant"), "merchant_id", "") or self.settings.merchant_id)
        memory = self.load(merchant_id)
        event = memory_event_from_state(state)
        if not event.get("question"):
            return memory
        events = [item for item in memory.get("events", []) if isinstance(item, dict)]
        if not events or event_fingerprint(events[-1]) != event_fingerprint(event):
            events.append(event)
        events = events[-120:]
        memory["events"] = events
        memory["recentFocus"] = aggregate_recent_focus(events)
        memory["updatedAt"] = datetime.now().isoformat()
        memory["merchantId"] = merchant_id
        path = self.memory_path(merchant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(memory, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        return memory

    def empty_memory(self, merchant_id: str) -> Dict[str, Any]:
        return {
            "merchantId": merchant_id or self.settings.merchant_id,
            "recentFocus": {},
            "preferences": {},
            "facts": [],
            "events": [],
            "updatedAt": "",
            "schemaVersion": "merchant_memory.v1",
        }


def memory_event_from_state(state: AgentState) -> Dict[str, Any]:
    plan = state.get("plan")
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
    route_slots = state.get("route_slots")
    route_payload = route_slots.model_dump(by_alias=True) if hasattr(route_slots, "model_dump") else (route_slots or {})
    if not time_windows:
        days = int(((route_payload.get("timeWindow") or {}).get("days") or 0) if isinstance(route_payload, dict) else 0)
        if days:
            time_windows.append(days)
    return {
        "eventId": "mem_%s" % datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "question": str(state.get("question") or "")[:1000],
        "answerPreview": str(state.get("answer") or "")[:1000],
        "topics": topics[:8],
        "metrics": metrics[:12],
        "timeWindows": sorted(set(time_windows))[:6],
        "analysisIntent": (getattr(plan, "question_understanding", {}) or {}).get("analysisIntent", "") if plan is not None else "",
        "accepted": bool(state.get("persisted")),
        "createdAt": datetime.now().isoformat(),
    }


def aggregate_recent_focus(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    recent = list(events)[-60:]
    topic_counter: Counter[str] = Counter()
    metric_counter: Counter[str] = Counter()
    days_counter: Counter[int] = Counter()
    for event in recent:
        topic_counter.update(str(item) for item in event.get("topics", []) if item)
        metric_counter.update(str(item) for item in event.get("metrics", []) if item)
        days_counter.update(int(item) for item in event.get("timeWindows", []) if item)
    return {
        "topTopics": [{"topic": key, "count": value} for key, value in topic_counter.most_common(6)],
        "topMetrics": [{"metric": key, "count": value} for key, value in metric_counter.most_common(8)],
        "commonTimeWindows": [{"days": key, "count": value} for key, value in days_counter.most_common(5)],
        "summary": build_focus_summary(topic_counter, metric_counter, days_counter),
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
    terms = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", question or ""))
    scored = []
    for index, event in enumerate(events):
        text = json.dumps(event, ensure_ascii=False, default=str)
        score = sum(1 for term in terms if term in text)
        if score or index >= len(events) - 8:
            scored.append((score, index, event))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [event for _, _, event in scored]


def event_fingerprint(event: Dict[str, Any]) -> str:
    return "%s|%s|%s" % (
        event.get("question", "")[:120],
        ",".join(event.get("metrics") or []),
        ",".join(str(item) for item in event.get("timeWindows") or []),
    )
