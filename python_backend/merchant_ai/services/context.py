from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from merchant_ai.config import Settings
from merchant_ai.models import ContextSnapshot, ImportantFact, SourceRef


class ImportantFactExtractor:
    """Extract facts that should survive context compaction."""

    def extract(self, state: Dict[str, Any], stage: str) -> List[ImportantFact]:
        facts: List[ImportantFact] = []
        self._add(facts, "question", state.get("question", ""), "user_request", 100, stage)
        merchant = state.get("merchant")
        merchant_id = getattr(merchant, "merchant_id", "") if merchant is not None else state.get("requested_merchant_id", "")
        self._add(facts, "merchant_id", merchant_id, "scope", 95, stage)
        loaded_skills = state.get("loaded_skills") or []
        if loaded_skills:
            self._add(facts, "loaded_skills", ",".join(str(item) for item in loaded_skills), "skills", 75, stage)
        plan = state.get("plan")
        if plan and getattr(plan, "intents", None):
            tables = []
            tasks = []
            for intent in plan.intents[:12]:
                if intent.preferred_table and intent.preferred_table not in tables:
                    tables.append(intent.preferred_table)
                if intent.plan_task_id:
                    tasks.append("%s:%s" % (intent.plan_task_id, intent.preferred_table or intent.answer_mode))
            self._add(facts, "plan_tables", ",".join(tables), "query_graph", 90, stage)
            self._add(facts, "plan_tasks", ";".join(tasks), "query_graph", 85, stage)
        validation = state.get("query_graph_validation_result")
        gaps = getattr(validation, "gaps", []) if validation is not None else []
        if gaps:
            self._add(facts, "validation_gaps", ";".join("%s:%s" % (gap.code, gap.task_id) for gap in gaps[:8]), "gap", 90, stage)
        run_result = state.get("agent_run_result")
        if run_result and getattr(run_result, "task_results", None):
            failures = [
                "%s:%s" % (item.task_id, item.query_bundle.error or item.summary)
                for item in run_result.task_results
                if item.query_bundle.failed
            ]
            if failures:
                self._add(facts, "node_failures", " | ".join(failures[:6]), "tool_result", 90, stage)
        return sorted(facts, key=lambda item: item.priority, reverse=True)[:24]

    def _add(self, facts: List[ImportantFact], key: str, value: Any, category: str, priority: int, stage: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        facts.append(
            ImportantFact(
                key=key,
                value=text[:1000],
                category=category,
                priority=priority,
                source_refs=[SourceRef(ref_type="state", title=stage, locator=key, reason="protected context fact")],
            )
        )


class ContextManager:
    """DeerFlow-style context boundary: compact, cite sources, and persist snapshots."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.extractor = ImportantFactExtractor()

    def refresh_state(self, state: Dict[str, Any], stage: str) -> ContextSnapshot:
        snapshot = self.snapshot(state, stage)
        state.setdefault("context_snapshots", []).append(snapshot.model_dump(by_alias=True))
        state["context_snapshots"] = state["context_snapshots"][-8:]
        state["summary_context"] = self.summary_context(state["context_snapshots"])
        self.persist_snapshot(state, snapshot)
        return snapshot

    def snapshot(self, state: Dict[str, Any], stage: str) -> ContextSnapshot:
        facts = self.extractor.extract(state, stage)
        source_refs = self.source_refs(state, stage)
        summary = "\n".join("- %s=%s" % (fact.key, fact.value) for fact in facts[:12])
        token_budget = int(getattr(self.settings, "context_window_tokens", 0) or 0)
        return ContextSnapshot(
            stage=stage,
            summary=summary[:4000],
            protected_facts=facts,
            source_refs=source_refs,
            token_budget=token_budget,
            truncated=len(summary) > 4000,
        )

    def source_refs(self, state: Dict[str, Any], stage: str) -> List[SourceRef]:
        refs = [SourceRef(ref_type="state", title=stage, locator="AgentState", reason="context snapshot source")]
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if outputs_path:
            refs.append(
                SourceRef(
                    ref_type="file",
                    path=str(Path(outputs_path) / "trace_replay.json"),
                    title="trace_replay.json",
                    reason="full replay can restore compressed details",
                )
            )
        return refs

    def summary_context(self, snapshots: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in snapshots[-3:]:
            stage = str(item.get("stage") or "")
            summary = str(item.get("summary") or "")
            if summary:
                lines.append("## %s\n%s" % (stage, summary))
        return "\n\n".join(lines)[-6000:]

    def persist_snapshot(self, state: Dict[str, Any], snapshot: ContextSnapshot) -> None:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return
        try:
            path = Path(outputs_path) / "context_snapshot.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot.model_dump(by_alias=True), ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        except Exception:
            return
