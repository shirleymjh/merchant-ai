from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from merchant_ai.config import Settings
from merchant_ai.models import ArtifactRef, ContextDelta, ContextPackage, ContextSnapshot, ImportantFact, SourceRef
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.security import identity_scope_payload
from merchant_ai.services.text_parsing import collapse_whitespace


class ImportantFactExtractor:
    """Extract facts that should survive context compaction."""

    def extract(self, state: Dict[str, Any], stage: str) -> List[ImportantFact]:
        facts: List[ImportantFact] = []
        self._add(facts, "question", state.get("question", ""), "user_request", 100, stage)
        merchant = state.get("merchant")
        merchant_id = getattr(merchant, "merchant_id", "") if merchant is not None else state.get("requested_merchant_id", "")
        self._add(facts, "merchant_id", merchant_id, "scope", 95, stage)
        self._extract_route_facts(facts, state, stage)
        self._extract_understanding_facts(facts, state, stage)
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
                self._add(facts, "intent_time_window:%s" % (intent.plan_task_id or len(tasks)), getattr(intent, "days", 0), "time_window", 88, stage)
                self._add_metric_fact(facts, "intent_metric:%s" % (intent.plan_task_id or len(tasks)), intent, stage)
                if getattr(intent, "filter_column", "") and getattr(intent, "filter_value", ""):
                    self._add(
                        facts,
                        "intent_filter:%s" % (intent.plan_task_id or len(tasks)),
                        "%s=%s" % (intent.filter_column, intent.filter_value),
                        "filter",
                        92,
                        stage,
                    )
            self._add(facts, "plan_tables", ",".join(tables), "query_graph", 90, stage)
            self._add(facts, "plan_tasks", ";".join(tasks), "query_graph", 85, stage)
            self._extract_plan_understanding_facts(facts, plan, stage)
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
            self._extract_entity_set_facts(facts, run_result.task_results, stage)
        if run_result and getattr(run_result, "evidence_gaps", None):
            self._add(
                facts,
                "evidence_gaps",
                ";".join("%s:%s:%s" % (gap.code, gap.task_id, gap.reason or gap.evidence) for gap in run_result.evidence_gaps[:8]),
                "gap",
                93,
                stage,
            )
        self._extract_memory_constraint_facts(facts, state, stage)
        self._extract_user_correction_facts(facts, state, stage)
        return sorted(facts, key=lambda item: item.priority, reverse=True)[:24]

    def _add(self, facts: List[ImportantFact], key: str, value: Any, category: str, priority: int, stage: str) -> None:
        text = str(value or "").strip()
        if not text or text == "0":
            return
        if any(item.key == key and item.value == text for item in facts):
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

    def _extract_route_facts(self, facts: List[ImportantFact], state: Dict[str, Any], stage: str) -> None:
        slots = state.get("route_slots")
        if not slots:
            return
        time_window = getattr(slots, "time_window", None)
        if isinstance(slots, dict):
            time_window = slots.get("timeWindow") or slots.get("time_window") or {}
        days = self._value(time_window, "days")
        raw = self._value(time_window, "raw")
        self._add(facts, "route_time_window", "%s天%s" % (days, ":%s" % raw if raw else ""), "time_window", 94, stage)
        object_refs = self._value(slots, "object_refs") or self._value(slots, "objectRefs") or []
        refs = []
        for item in list(object_refs or [])[:12]:
            ref_type = self._value(item, "ref_type") or self._value(item, "refType")
            value = self._value(item, "value")
            raw_value = self._value(item, "raw")
            if ref_type and (value or raw_value):
                refs.append("%s=%s" % (ref_type, value or raw_value))
        self._add(facts, "route_object_refs", ";".join(refs), "filter", 96, stage)

    def _extract_understanding_facts(self, facts: List[ImportantFact], state: Dict[str, Any], stage: str) -> None:
        fast = state.get("fast_understanding")
        if fast:
            self._add(facts, "fast_time_window", self._value(fast, "time_window_days") or self._value(fast, "timeWindowDays"), "time_window", 88, stage)
            metrics = self._value(fast, "metric_phrases") or self._value(fast, "metricPhrases") or []
            self._add(facts, "semantic_metric_phrases", ",".join(str(item) for item in list(metrics or [])[:12]), "metric", 86, stage)
            object_refs = self._value(fast, "object_refs") or self._value(fast, "objectRefs") or {}
            if isinstance(object_refs, dict):
                refs = ["%s=%s" % (key, ",".join(str(v) for v in list(value or [])[:8])) for key, value in object_refs.items() if value]
                self._add(facts, "fast_object_refs", ";".join(refs), "filter", 90, stage)

    def _extract_plan_understanding_facts(self, facts: List[ImportantFact], plan: Any, stage: str) -> None:
        understanding = getattr(plan, "question_understanding", {}) or {}
        if not isinstance(understanding, dict):
            return
        self._add(facts, "understanding_time_window", understanding.get("timeWindowDays") or understanding.get("time_window_days"), "time_window", 95, stage)
        self._add(facts, "analysis_intent", understanding.get("analysisIntent") or understanding.get("analysis_intent"), "analysis_intent", 84, stage)
        self._extract_metric_object(facts, "ranking_objective", understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}, stage, 96)
        for index, item in enumerate(understanding.get("requestedMeasures") or understanding.get("requested_measures") or []):
            self._extract_metric_object(facts, "requested_measure:%d" % index, item, stage, 91)
        filters = []
        for item in understanding.get("filters") or []:
            field = self._value(item, "field") or self._value(item, "column")
            value = self._value(item, "value")
            if field and value:
                filters.append("%s=%s" % (field, value))
        self._add(facts, "understanding_filters", ";".join(filters), "filter", 94, stage)
        scopes = []
        for item in understanding.get("scopeConstraints") or understanding.get("scope_constraints") or []:
            scope = self._compact_dict(item, ["ownerTable", "owner_table", "field", "value", "metricRef", "metric_ref", "required"])
            if scope:
                scopes.append(scope)
        self._add(facts, "scope_constraints", " | ".join(scopes), "scope", 97, stage)
        required_evidence = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
        self._add(facts, "required_evidence_intents", ",".join(str(item) for item in list(required_evidence or [])[:12]), "evidence_contract", 89, stage)

    def _add_metric_fact(self, facts: List[ImportantFact], key: str, intent: Any, stage: str) -> None:
        pieces = [
            self._value(intent, "metric_name") or self._value(intent, "metricName"),
            self._value(intent, "metric_column") or self._value(intent, "metricColumn"),
            self._value(intent, "metric_formula") or self._value(intent, "metricFormula"),
        ]
        specs = self._value(intent, "metric_specs") or self._value(intent, "metricSpecs") or []
        for spec in list(specs or [])[:4]:
            text = self._compact_dict(spec, ["metricName", "metric_name", "metricRef", "metric_ref", "formula"])
            if text:
                pieces.append(text)
        self._add(facts, key, " | ".join(str(item) for item in pieces if item), "metric", 90, stage)

    def _extract_metric_object(self, facts: List[ImportantFact], key: str, item: Any, stage: str, priority: int) -> None:
        text = self._compact_dict(item, ["metricRef", "metric_ref", "sourcePhrase", "source_phrase", "ownerTable", "owner_table", "groupByColumn", "group_by_column", "objectiveType", "objective_type"])
        self._add(facts, key, text, "metric", priority, stage)

    def _extract_entity_set_facts(self, facts: List[ImportantFact], task_results: Iterable[Any], stage: str) -> None:
        rows = []
        for item in list(task_results or [])[:12]:
            entity = self._value(item, "entity_set") or self._value(item, "entitySet")
            if not entity:
                continue
            task_id = self._value(entity, "task_id") or self._value(entity, "taskId") or self._value(item, "task_id") or self._value(item, "taskId")
            join_key = self._value(entity, "join_key") or self._value(entity, "joinKey")
            values = list(self._value(entity, "values") or [])[:12]
            column_values = self._value(entity, "column_values") or self._value(entity, "columnValues") or {}
            column_keys = ",".join(sorted(list(column_values.keys()))[:8]) if isinstance(column_values, dict) else ""
            if values or column_keys:
                rows.append("task=%s key=%s values=%s columns=%s" % (task_id, join_key, values, column_keys))
        self._add(facts, "reusable_entity_sets", " | ".join(rows), "entity_set", 98, stage)

    def _extract_memory_constraint_facts(self, facts: List[ImportantFact], state: Dict[str, Any], stage: str) -> None:
        rows = []
        for item in list(state.get("memory_constraints") or [])[:8]:
            if not isinstance(item, dict):
                continue
            enforcement = str(item.get("enforcement") or "")
            if enforcement not in {"required", "preferred"}:
                continue
            rows.append(self._compact_dict(item, ["memoryType", "memory_type", "summary", "value", "metrics", "timeWindows", "enforcement"]))
        self._add(facts, "memory_constraints", " | ".join(row for row in rows if row), "memory", 87, stage)

    def _extract_user_correction_facts(self, facts: List[ImportantFact], state: Dict[str, Any], stage: str) -> None:
        rows = []
        injection = state.get("memory_injection") or {}
        for item in list(injection.get("relevantCorrections") or [])[-4:]:
            text = (
                self._value(item, "correctionText")
                or self._value(item, "correction_text")
                or self._value(item, "content")
            )
            normalized = collapse_whitespace(text)
            if normalized:
                rows.append(normalized[:300])
        self._add(facts, "user_corrections", " | ".join(rows[-4:]), "correction", 99, stage)

    def _compact_dict(self, item: Any, keys: List[str]) -> str:
        if not item:
            return ""
        parts = []
        for key in keys:
            value = self._value(item, key)
            if value not in ("", None, [], {}):
                parts.append("%s=%s" % (key, value))
        return ",".join(parts)

    def _value(self, item: Any, key: str) -> Any:
        if item is None:
            return ""
        if isinstance(item, dict):
            return item.get(key, "")
        return getattr(item, key, "")


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
        agent_context_policy = self.context_policy(agent)
        hash_payload = {
            "stage": stage,
            "agent": agent,
            "taskId": task_id,
            "question": question,
            "merchantId": str(state.get("requested_merchant_id") or ""),
            "identityScope": identity_scope_payload(
                state.get("user_identity") or {},
                str(state.get("requested_merchant_id") or ""),
            ),
            "accessRole": str(state.get("access_role") or ""),
            "tables": allowed_tables or [],
            "metrics": allowed_metrics or [],
            "protectedFacts": [fact.model_dump(by_alias=True) for fact in snapshot.protected_facts[:18]],
            "memoryConstraints": list(state.get("memory_constraints") or [])[:12],
            "memoryIds": list((state.get("memory_injection_trace") or {}).get("selectedIds") or [])[:24],
            "memoryInjectionSha256": hashlib.sha256(
                json.dumps(state.get("memory_injection") or {}, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "runtimeInjectionSha256": hashlib.sha256(
                json.dumps(state.get("runtime_injection") or {}, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "sessionContextSha256": hashlib.sha256(str(state.get("session_context") or "").encode("utf-8")).hexdigest(),
            "threadContext": {
                "previousRunId": (state.get("thread_context") or {}).get("previousRunId", ""),
                "previousQuestion": (state.get("thread_context") or {}).get("previousQuestion", ""),
                "recentMessages": list((state.get("thread_context") or {}).get("messageHistory") or [])[-6:],
            },
            "sourceRefs": [
                {
                    "locator": ref.locator or ref.path or ref.title,
                    "merchantUri": ref.merchant_uri,
                }
                for ref in snapshot.source_refs[:12]
            ],
            "artifactRefs": [
                {
                    "path": ref.relative_path or ref.path,
                    "sha256": ref.sha256,
                }
                for ref in artifact_refs
            ],
            "agentContextPolicy": agent_context_policy,
            "evidenceGaps": gaps,
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
                "agent context must follow the declared includePriority and exclude sections",
            ],
            protected_facts=snapshot.protected_facts[:18],
            source_refs=snapshot.source_refs[:12],
            artifact_refs=artifact_refs,
            allowed_tables=allowed_tables or [],
            allowed_metrics=allowed_metrics or [],
            agent_context_policy=agent_context_policy,
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

    def context_policy(self, agent: str) -> Dict[str, Any]:
        normalized = str(agent or "LeadAgent")
        policies: Dict[str, Dict[str, Any]] = {
            "PlannerAgent": {
                "includePriority": ["semanticMetrics", "tableAssets", "relationships", "businessRules", "planningCases"],
                "exclude": ["rawDataRows", "fullSqlLogs"],
                "budgetChars": int(getattr(self.settings, "context_planner_budget_chars", 12000) or 12000),
            },
            "NodeAgent": {
                "includePriority": ["allowedTables", "fields", "partitionColumns", "joinKeys", "rowAccessPolicy"],
                "exclude": ["longTermNarratives", "unverifiedBusinessAdvice", "unscopedTables"],
                "budgetChars": int(getattr(self.settings, "context_runtime_budget_chars", 6000) or 6000),
            },
            "EvidenceVerifierAgent": {
                "includePriority": ["queryGraph", "executedSql", "queryResults", "ruleRefs", "evidenceContracts"],
                "exclude": ["unverifiedRecallClaims", "unexecutedSql"],
                "budgetChars": int(getattr(self.settings, "context_runtime_budget_chars", 6000) or 6000),
            },
            "AnswerAgent": {
                "includePriority": ["verifiedEvidence", "queryResultSummary", "ruleEvidence", "evidenceGaps"],
                "exclude": ["rawSchema", "unverifiedRecallClaims", "fullToolLogs"],
                "budgetChars": int(getattr(self.settings, "context_answer_budget_chars", 10000) or 10000),
            },
            "RuleAnswerAgent": {
                "includePriority": ["publishedRules", "ruleRefs", "effectiveDates", "evidenceGaps"],
                "exclude": ["unverifiedDataClaims", "rawSchema"],
                "budgetChars": int(getattr(self.settings, "context_answer_budget_chars", 10000) or 10000),
            },
        }
        return {"agent": normalized, **policies.get(normalized, {
            "includePriority": ["question", "routeSlots", "governedMemory", "recallSummary"],
            "exclude": ["oversizedToolLogs"],
            "budgetChars": int(getattr(self.settings, "context_runtime_budget_chars", 6000) or 6000),
        })}

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
