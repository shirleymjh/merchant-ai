from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings
from merchant_ai.models import ArtifactRef, ContextDelta, ContextPackage, ContextSnapshot, ImportantFact, SourceRef
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact


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
        package = self.package(state, stage=stage, agent=self.agent_for_stage(stage), snapshot=snapshot)
        state.setdefault("context_packages", []).append(package)
        state["context_packages"] = state["context_packages"][-12:]
        self.persist_snapshot(state, snapshot)
        self.persist_package(state, package)
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
        refs = [
            SourceRef(
                ref_type="state",
                title=stage,
                locator="AgentState",
                reason="context snapshot source",
                merchant_uri="merchant://state/agent",
                context_layer="L0",
            )
        ]
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if outputs_path:
            refs.append(
                SourceRef(
                    ref_type="file",
                    path=str(Path(outputs_path) / "trace_replay.json"),
                    title="trace_replay.json",
                    reason="full replay can restore compressed details",
                    merchant_uri=merchant_uri_for_artifact("trace_replay.json", namespace="trace"),
                    context_layer="L2",
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

    def package(
        self,
        state: Dict[str, Any],
        stage: str,
        agent: str,
        snapshot: Optional[ContextSnapshot] = None,
        task_id: str = "",
        artifact_refs: Optional[List[ArtifactRef]] = None,
        allowed_tables: Optional[List[str]] = None,
        allowed_metrics: Optional[List[str]] = None,
    ) -> ContextPackage:
        snapshot = snapshot or self.snapshot(state, stage)
        pack = state.get("planning_asset_pack")
        if allowed_tables is None and pack is not None and hasattr(pack, "known_tables"):
            allowed_tables = pack.known_tables()[:12]
        if allowed_metrics is None and pack is not None:
            allowed_metrics = [getattr(item, "key", "") for item in getattr(pack, "metrics", [])[:24] if getattr(item, "key", "")]
        run_result = state.get("agent_run_result")
        gaps = []
        if run_result is not None:
            for gap in getattr(run_result, "evidence_gaps", [])[:12]:
                gaps.append(gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap)
        question = str(state.get("question") or "")
        summary = snapshot.summary[:4000]
        input_chars = len(question) + len(summary) + sum(len(str(fact.value)) for fact in snapshot.protected_facts)
        artifact_refs = artifact_refs or self.artifact_refs(state)
        hash_payload = {
            "stage": stage,
            "agent": agent,
            "taskId": task_id,
            "question": question,
            "tables": allowed_tables or [],
            "metrics": allowed_metrics or [],
            "sourceRefs": [ref.locator or ref.path or ref.title for ref in snapshot.source_refs[:12]],
            "artifactRefs": [ref.relative_path or ref.path for ref in artifact_refs],
        }
        context_hash = hashlib.sha256(json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
        return ContextPackage(
            package_id="ctx_" + uuid.uuid4().hex,
            run_id=str(state.get("run_id") or ""),
            thread_id=str(state.get("thread_id") or ""),
            stage=stage,
            agent=agent,
            task_id=task_id,
            question=question,
            merchant_id=str(state.get("requested_merchant_id") or ""),
            goal=question[:500],
            constraints=[
                "single-node SQL must stay within node plan contract",
                "answer agent may use verified evidence only",
                "large objects are referenced by artifact path",
            ],
            protected_facts=snapshot.protected_facts[:18],
            source_refs=snapshot.source_refs[:12],
            artifact_refs=artifact_refs,
            allowed_tables=allowed_tables or [],
            allowed_metrics=allowed_metrics or [],
            evidence_gaps=gaps,
            summary=summary,
            inline_budget_chars=int(getattr(self.settings, "context_file_inline_max_chars", 0) or 0),
            input_chars=input_chars,
            offload_reason="large rows/tool results stay in workspace artifacts; context package keeps refs only",
            context_hash=context_hash,
            context_delta=ContextDelta(
                context_hash=context_hash,
                changed_refs=[ref.relative_path or ref.path for ref in artifact_refs],
                inline_chars=input_chars,
                artifact_refs=[ref.relative_path or ref.path for ref in artifact_refs],
            ),
        )

    def artifact_refs(self, state: Dict[str, Any]) -> List[ArtifactRef]:
        refs: List[ArtifactRef] = []
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return refs
        for relative in [
            "trace_replay.json",
            "context_snapshot.json",
            "artifacts/planner/planning_asset_pack.json",
            "artifacts/planner/query_graph.json",
            "artifacts/node/agent_run_result.json",
        ]:
            path = Path(outputs_path) / relative
            if path.exists():
                refs.append(
                    ArtifactRef(
                        artifact_id="artifact_" + uuid.uuid4().hex,
                        namespace=relative.split("/", 1)[0] if "/" in relative else "trace",
                        path=str(path),
                        relative_path=relative,
                        title=path.name,
                        reason="recoverable context artifact",
                        bytes=path.stat().st_size,
                        estimated_chars=path.stat().st_size,
                        merchant_uri=merchant_uri_for_artifact(relative, namespace=relative.split("/", 1)[0] if "/" in relative else "trace"),
                        context_layer="L2",
                    )
                )
        return refs

    def persist_package(self, state: Dict[str, Any], package: ContextPackage) -> None:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return
        try:
            path = Path(outputs_path) / "context_packages" / ("%s.json" % package.stage)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(package.model_dump(by_alias=True), ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        except Exception:
            return

    def agent_for_stage(self, stage: str) -> str:
        if stage in {"plan_query_graph", "compact_assets"}:
            return "PlannerAgent"
        if stage in {"execute_query_graph"}:
            return "NodeAgent"
        if stage in {"verify_evidence_graph"}:
            return "EvidenceVerifierAgent"
        if stage in {"cache_answer", "answer_analysis"}:
            return "AnswerAgent"
        return "LeadAgent"
