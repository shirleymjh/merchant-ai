from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.models import ArtifactRef, ContextAssemblyReport
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.observability import artifact_ref_from_path


class ContextAssembler:
    """Single engineering boundary for dynamic context and large prompt payloads."""

    LARGE_KEYS = {
        "plannerToolResults",
        "_plannerToolResults",
        "dataRows",
        "rows",
        "tasks",
        "taskResults",
        "trace",
        "compilerTrace",
        "semanticFileContext",
        "semanticWorkspace",
        "agentRunResult",
        "runResult",
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    def runtime_injection(self, state: AgentState, stage: str = "") -> Dict[str, Any]:
        merchant = state.get("merchant")
        route_slots = state.get("route_slots")
        manifest = state.get("workspace_manifest")
        entries = getattr(manifest, "entries", []) if manifest is not None else []
        if isinstance(manifest, dict):
            entries = manifest.get("entries") or []
        route_payload = route_slots.model_dump(by_alias=True) if hasattr(route_slots, "model_dump") else (route_slots or {})
        topic_candidates = []
        for item in route_payload.get("topicCandidates") or route_payload.get("topic_candidates") or []:
            topic_candidates.append(item.get("topic") if isinstance(item, dict) else str(item))
        thread_context = state.get("thread_context") or {}
        return {
            "stage": stage,
            "currentDate": datetime.now().date().isoformat(),
            "merchant": {
                "merchantId": getattr(merchant, "merchant_id", "") if merchant is not None else state.get("requested_merchant_id", ""),
                "merchantName": getattr(merchant, "merchant_name", "") if merchant is not None else "",
            },
            "routeSlots": route_payload,
            "topicCandidates": topic_candidates[:8],
            "loadedSkillHeaders": state.get("loaded_skills") or [],
            "threadContext": {
                "restored": bool(thread_context.get("restored")),
                "previousQuestion": thread_context.get("previousQuestion", ""),
                "reusableEntitySets": thread_context.get("reusableEntitySets", [])[:6],
            },
            "workspace": {
                "manifestPath": str(Path(getattr(state.get("thread_data"), "outputs_path", "") or "") / "workspace_manifest.json"),
                "artifactCount": len(entries or []),
                "visibleArtifacts": [
                    {
                        "path": getattr(entry, "relative_path", "") if hasattr(entry, "relative_path") else entry.get("relativePath", ""),
                        "uri": getattr(entry, "merchant_uri", "") if hasattr(entry, "merchant_uri") else entry.get("merchantUri", ""),
                    }
                    for entry in list(entries or [])[:12]
                ],
            },
            "dynamicReminders": [
                "Use verified semantic assets and artifact refs; do not assume full hidden catalog.",
                "Large rows/tool results are in workspace artifacts; request artifact_read when details are needed.",
            ],
        }

    def render_runtime_context(self, injection: Dict[str, Any], budget_chars: int = 0) -> str:
        budget = max(800, int(budget_chars or self.settings.context_runtime_budget_chars or 6000))
        text = json.dumps(injection, ensure_ascii=False, default=str, indent=2)
        if len(text) <= budget:
            return text
        compact = {
            "stage": injection.get("stage"),
            "currentDate": injection.get("currentDate"),
            "merchant": injection.get("merchant"),
            "topicCandidates": injection.get("topicCandidates"),
            "threadContext": injection.get("threadContext"),
            "workspace": {
                "manifestPath": (injection.get("workspace") or {}).get("manifestPath"),
                "artifactCount": (injection.get("workspace") or {}).get("artifactCount"),
            },
            "dynamicReminders": injection.get("dynamicReminders"),
        }
        return json.dumps(compact, ensure_ascii=False, default=str, indent=2)[:budget]

    def compact_text_context(self, state: AgentState, stage: str, agent: str, text: str, budget_chars: int = 0) -> str:
        budget = max(1000, int(budget_chars or self.settings.context_runtime_budget_chars or 6000))
        if len(text or "") <= budget:
            self._record_report(state, stage, agent, len(text or ""), len(text or ""), budget, False, [], [], "inline")
            return text
        artifact = self._write_artifact(state, "context", "%s_%s_full.md" % (stage, agent), text)
        preview = (text or "")[: max(500, budget - 700)]
        compact = (
            "%s\n\n[context_offloaded]\nfullContextArtifact=%s\nmerchantUri=%s\noriginalChars=%d"
            % (preview, artifact.relative_path or artifact.path, artifact.merchant_uri, len(text or ""))
        )
        self._record_report(state, stage, agent, len(text or ""), len(compact), budget, True, ["text"], [artifact], "text context exceeded budget")
        return compact

    def assemble_payload(
        self,
        state: AgentState,
        stage: str,
        agent: str,
        payload: Dict[str, Any],
        budget_chars: int = 0,
    ) -> Dict[str, Any]:
        budget = max(1000, int(budget_chars or self.settings.context_planner_budget_chars or 12000))
        raw = json.dumps(payload or {}, ensure_ascii=False, default=str, sort_keys=True)
        if len(raw) <= budget:
            self._record_report(state, stage, agent, len(raw), len(raw), budget, False, [], [], "inline")
            return payload
        compacted, sections, artifacts = self._compact_payload_sections(state, stage, payload)
        compact_raw = json.dumps(compacted, ensure_ascii=False, default=str, sort_keys=True)
        if len(compact_raw) > budget:
            artifact = self._write_artifact(state, "context", "%s_%s_payload.json" % (stage, agent), payload)
            envelope = {
                key: compacted[key]
                for key in [
                    "question",
                    "status",
                    "reason",
                    "outputContract",
                    "plannerBudgetLevel",
                    "openDiagnostic",
                    "fastUnderstanding",
                    "previousUnderstanding",
                    "threadContext",
                    "runtimeInjection",
                    "memoryInjection",
                ]
                if key in compacted
            }
            envelope["_contextOverBudget"] = {
                "artifact": artifact.relative_path or artifact.path,
                "merchantUri": artifact.merchant_uri,
                "originalChars": len(raw),
                "budgetChars": budget,
            }
            compacted = envelope
            artifacts.append(artifact)
            sections.append("payload")
        final_raw = json.dumps(compacted, ensure_ascii=False, default=str, sort_keys=True)
        self._record_report(
            state,
            stage,
            agent,
            len(raw),
            len(final_raw),
            budget,
            True,
            sorted(set(sections)),
            artifacts,
            "payload exceeded stage budget",
        )
        return compacted

    def _compact_payload_sections(self, state: AgentState, stage: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[ArtifactRef]]:
        compacted: Dict[str, Any] = {}
        sections: List[str] = []
        artifacts: List[ArtifactRef] = []
        for key, value in (payload or {}).items():
            if key not in self.LARGE_KEYS:
                compacted[key] = value
                continue
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) <= self.settings.tool_result_offload_chars:
                compacted[key] = value
                continue
            artifact = self._write_artifact(state, "context", "%s_%s.json" % (stage, key), value)
            artifacts.append(artifact)
            sections.append(key)
            compacted[key] = compact_large_value(value, artifact)
        return compacted, sections, artifacts

    def _write_artifact(self, state: AgentState, namespace: str, name: str, payload: Any) -> ArtifactRef:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return ArtifactRef()
        store = WorkspaceArtifactStore(self.settings, Path(outputs_path) / "artifacts")
        if isinstance(payload, str):
            artifact = store.write_text(namespace, name, payload)
        else:
            artifact = store.write_json(namespace, name, payload, preview_chars=0)
        ref = artifact_ref_from_path(artifact.get("path", ""), namespace=namespace, reason="context assembler offload")
        ref.merchant_uri = artifact.get("merchantUri", ref.merchant_uri)
        ref.relative_path = artifact.get("relativePath", ref.relative_path)
        ref.context_layer = "L2"
        return ref

    def _record_report(
        self,
        state: AgentState,
        stage: str,
        agent: str,
        input_chars: int,
        output_chars: int,
        budget_chars: int,
        compacted: bool,
        trimmed_sections: List[str],
        artifact_refs: List[ArtifactRef],
        reason: str,
    ) -> None:
        context_hash = hashlib.sha256(
            json.dumps(
                {
                    "stage": stage,
                    "agent": agent,
                    "input": input_chars,
                    "output": output_chars,
                    "artifacts": [ref.relative_path or ref.path for ref in artifact_refs],
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        report = ContextAssemblyReport(
            stage=stage,
            agent=agent,
            input_chars=input_chars,
            output_chars=output_chars,
            budget_chars=budget_chars,
            compacted=compacted,
            trimmed_sections=trimmed_sections,
            artifact_refs=artifact_refs,
            context_hash=context_hash,
            reason=reason,
        )
        state.setdefault("context_assembly_reports", []).append(report)
        state["context_assembly_reports"] = state["context_assembly_reports"][-50:]


class ThreadContextService:
    """Restore thread-level recoverable context without resuming an ended run checkpoint."""

    def restore(self, state: AgentState) -> Dict[str, Any]:
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if not outputs_path:
            return {"restored": False, "reason": "missing_outputs_path"}
        root = Path(outputs_path)
        snapshot = self._read_json(root / "context_snapshot.json")
        trace = self._read_json(root / "trace_replay.json")
        run_result = self._read_json(root / "artifacts" / "node" / "agent_run_result.json")
        if trace.get("runId") == state.get("run_id"):
            trace = {}
        entity_sets = extract_reusable_entity_sets(run_result)
        artifacts = self._artifact_refs(root)
        restored = bool(snapshot or trace or entity_sets or artifacts)
        context = {
            "restored": restored,
            "checkpointRef": state.get("checkpoint_thread_id", ""),
            "previousRunId": trace.get("runId", ""),
            "previousQuestion": trace.get("question", ""),
            "previousAnswerPreview": str(trace.get("answer") or "")[:500],
            "previousSummary": snapshot.get("summary", ""),
            "previousArtifacts": artifacts[:20],
            "reusableEntitySets": entity_sets[:10],
            "restoredAt": datetime.now().isoformat(),
        }
        state["thread_context"] = context
        if restored:
            state["session_context"] = append_thread_context_summary(state.get("session_context") or "", context)
        return context

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}

    def _artifact_refs(self, root: Path) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for relative in [
            "trace_replay.json",
            "context_snapshot.json",
            "workspace_manifest.json",
            "artifacts/planner/planning_asset_pack.json",
            "artifacts/planner/query_graph.json",
            "artifacts/node/agent_run_result.json",
        ]:
            path = root / relative
            if not path.exists():
                continue
            refs.append(
                {
                    "relativePath": relative,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "merchantUri": merchant_uri_for_artifact(relative, namespace=relative.split("/", 1)[0] if "/" in relative else "trace"),
                }
            )
        return refs


def compact_large_value(value: Any, artifact: ArtifactRef) -> Any:
    if isinstance(value, list):
        preview = [compact_preview_item(item) for item in value[:5]]
        return {
            "offloaded": True,
            "preview": preview,
            "itemCount": len(value),
            "artifact": artifact.relative_path or artifact.path,
            "merchantUri": artifact.merchant_uri,
        }
    if isinstance(value, dict):
        preview = {key: value[key] for key in list(value.keys())[:12]}
        return {
            "offloaded": True,
            "preview": preview,
            "keyCount": len(value),
            "artifact": artifact.relative_path or artifact.path,
            "merchantUri": artifact.merchant_uri,
        }
    return {
        "offloaded": True,
        "preview": str(value)[:1000],
        "artifact": artifact.relative_path or artifact.path,
        "merchantUri": artifact.merchant_uri,
    }


def compact_preview_item(item: Any) -> Any:
    if not isinstance(item, dict):
        text = str(item)
        return text[:300] + ("..." if len(text) > 300 else "")
    keep: Dict[str, Any] = {}
    for key in ["id", "name", "status", "round", "artifact", "promptArtifact", "toolName", "errorType", "errorMessage"]:
        if key in item:
            keep[key] = item.get(key)
    result = item.get("result")
    if isinstance(result, dict):
        keep["resultPreview"] = {
            key: str(result.get(key))[:300]
            for key in ["refId", "path", "title", "truncated", "estimatedChars", "error"]
            if key in result
        }
        if "content" in result:
            keep.setdefault("resultPreview", {})["contentPreview"] = str(result.get("content") or "")[:300]
        if "items" in result:
            keep.setdefault("resultPreview", {})["itemCount"] = len(result.get("items") or [])
        if "hits" in result:
            keep.setdefault("resultPreview", {})["hitCount"] = len(result.get("hits") or [])
    return keep or {key: str(value)[:200] for key, value in list(item.items())[:6]}


def extract_reusable_entity_sets(run_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = run_result.get("taskResults") or run_result.get("task_results") or []
    entities: List[Dict[str, Any]] = []
    for item in results:
        entity = item.get("entitySet") or item.get("entity_set") or {}
        if not isinstance(entity, dict):
            continue
        values = entity.get("values") or []
        column_values = entity.get("columnValues") or entity.get("column_values") or {}
        if not values and not column_values:
            continue
        entities.append(
            {
                "taskId": entity.get("taskId") or entity.get("task_id") or item.get("taskId") or item.get("task_id") or "",
                "joinKey": entity.get("joinKey") or entity.get("join_key") or "",
                "valuesPreview": list(values)[:20],
                "valueCount": len(values),
                "columnKeys": sorted(list(column_values.keys()))[:12] if isinstance(column_values, dict) else [],
                "sourceRowCount": entity.get("sourceRowCount") or entity.get("source_row_count") or 0,
            }
        )
    return entities


def append_thread_context_summary(existing: str, context: Dict[str, Any]) -> str:
    lines = []
    if existing:
        lines.append(existing)
    lines.append("## 上轮线程上下文")
    if context.get("previousQuestion"):
        lines.append("- previousQuestion=%s" % str(context.get("previousQuestion"))[:300])
    if context.get("previousAnswerPreview"):
        lines.append("- previousAnswerPreview=%s" % str(context.get("previousAnswerPreview"))[:300])
    for entity in context.get("reusableEntitySets", [])[:6]:
        lines.append(
            "- reusableEntitySet task=%s key=%s count=%s preview=%s"
            % (
                entity.get("taskId", ""),
                entity.get("joinKey", ""),
                entity.get("valueCount", 0),
                entity.get("valuesPreview", [])[:8],
            )
        )
    for artifact in context.get("previousArtifacts", [])[:6]:
        lines.append("- artifact=%s uri=%s" % (artifact.get("relativePath", ""), artifact.get("merchantUri", "")))
    return "\n".join(lines)[-6000:]
