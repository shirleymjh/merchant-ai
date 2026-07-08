from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from merchant_ai.config import Settings
from merchant_ai.graph.state import AgentState
from merchant_ai.models import (
    AgentDecision,
    ArtifactRef,
    ContextBudgetReport,
    ContextCompressionEvent,
    GraphValidationGap,
    GraphValidationResult,
    MiddlewareEvent,
    ToolCallExecutionResult,
    ToolCallLedgerEntry,
    ToolCallRecoveryEvent,
    WorkspaceManifest,
    WorkspaceManifestEntry,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.context_assembly import ContextAssembler
from merchant_ai.services.context import ContextManager
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.memory import create_memory_store, truncate_memory_text_by_tokens
from merchant_ai.services.memory_constraints import build_memory_constraints
from merchant_ai.services.observability import artifact_ref_from_path


TERMINAL_TOOL_STATUSES = {"success", "failed", "error", "timeout", "rate_limited", "circuit_blocked", "skipped"}


class HarnessMiddleware:
    """Small, composable runtime middleware around the LeadAgent loop."""

    name = "middleware"

    def before_policy(self, state: AgentState) -> AgentState:
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        return state


class MiddlewareChain:
    def __init__(self, middlewares: Iterable[HarnessMiddleware]):
        self.middlewares = list(middlewares)

    def before_policy(self, state: AgentState) -> AgentState:
        for middleware in self.middlewares:
            state = self._safe_call(middleware, "before_policy", state)
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        for middleware in self.middlewares:
            state = self._safe_call(middleware, "before_action", state, decision)
        return state

    def _safe_call(self, middleware: HarnessMiddleware, method: str, state: AgentState, *args: Any) -> AgentState:
        try:
            handler = getattr(middleware, method)
            return handler(state, *args)
        except Exception as exc:
            append_middleware_event(
                state,
                middleware.name,
                method,
                status="error",
                code="MIDDLEWARE_ERROR",
                message=str(exc),
            )
            return state


class CancellationMiddleware(HarnessMiddleware):
    name = "cancellation"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        run_id = str(state.get("run_id") or "")
        if not run_id or state.get("run_canceled"):
            return state
        path = self.settings.resolved_workspace_path / "run_events" / "runs" / ("%s.json" % run_id)
        status = ""
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = str(payload.get("status") or "").upper()
        except Exception:
            status = ""
        if status in {"CANCELED", "CANCELLED"}:
            state["run_canceled"] = True
            state["chat_bi_completed"] = True
            state["answer"] = "本次运行已取消。"
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="blocked",
                code="RUN_CANCELED",
                message="run status is CANCELED in run event store",
            )
        return state


class PermissionMiddleware(HarnessMiddleware):
    name = "permission"

    def before_policy(self, state: AgentState) -> AgentState:
        slots = state.get("route_slots")
        operation = str(getattr(slots, "operation", "") or "")
        if operation == "write_requested":
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="blocked",
                code="WRITE_OPERATION_REQUIRES_HUMAN",
                message="write operation is routed to human clarification instead of tool execution",
            )
        return state


class ActionContractMiddleware(HarnessMiddleware):
    """Validate the selected action against declared state contracts."""

    name = "action_contract"

    def __init__(self):
        from merchant_ai.graph.policy import AgentActionRegistry

        self.registry = AgentActionRegistry()

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        action = self.registry.get(decision.selected_action)
        missing_keys = [key for key in action.required_state_keys if not state_path_ready(state, key)]
        missing_flags = [flag for flag in action.required_state_flags if not bool(state.get(flag))]
        if not missing_keys and not missing_flags:
            return state

        fallback_id = self.fallback_action(action.fallback_action, missing_keys, missing_flags)
        fallback = self.registry.get(fallback_id)
        original_action = decision.selected_action
        original_node = decision.selected_node
        if fallback.id == original_action:
            fallback = self.registry.get("answer_data")
            fallback_id = fallback.id

        decision.selected_action = fallback.id
        decision.selected_node = fallback.node
        decision.source = "contract"
        decision.reason = (
            "Action contract rerouted %s to %s; missing keys=%s flags=%s. %s"
            % (original_action, fallback.id, missing_keys, missing_flags, decision.reason or "")
        ).strip()
        available = [fallback.id] + [item for item in decision.available_actions if item != fallback.id]
        decision.available_actions = available
        append_middleware_event(
            state,
            self.name,
            "before_action",
            status="rerouted",
            code="ACTION_CONTRACT_MISSING_PREREQUISITE",
            message="selected action was rerouted before execution because required state was not ready",
            metadata={
                "fromAction": original_action,
                "fromNode": original_node,
                "toAction": fallback.id,
                "toNode": fallback.node,
                "missingStateKeys": missing_keys,
                "missingStateFlags": missing_flags,
            },
        )
        return state

    def fallback_action(self, configured: str, missing_keys: List[str], missing_flags: List[str]) -> str:
        if "topic_routed" in missing_flags:
            return "route_topic"
        if "data_discovered" in missing_flags:
            return "retrieve_knowledge"
        if "planning_assets_compacted" in missing_flags:
            return "compact_assets"
        if "plan.intents" in missing_keys:
            return "plan_graph"
        if "query_graph_validated" in missing_flags:
            return "validate_graph"
        if "sql_generated" in missing_flags:
            return "execute_graph"
        if "agent_run_result.task_results" in missing_keys:
            return "execute_graph"
        return configured or "answer_data"


class ContextBudgetMiddleware(HarnessMiddleware):
    name = "context_budget"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        stage = "policy_round_%s" % int(state.get("react_round") or 0)
        estimated_tokens = estimate_context_tokens(state)
        window = max(1, int(self.settings.context_window_tokens or 1))
        ratio = round(estimated_tokens / window, 4)
        threshold = float(self.settings.context_compaction_threshold_ratio or 0.85)
        report = ContextBudgetReport(
            stage=stage,
            window_tokens=window,
            estimated_tokens=estimated_tokens,
            usage_ratio=ratio,
            threshold_ratio=threshold,
            over_budget=ratio >= threshold,
            protected_fact_count=sum(snapshot_protected_fact_count(snapshot) for snapshot in state.get("context_snapshots", [])),
            artifact_count=len(workspace_entries(state)),
            summary_chars=len(str(state.get("summary_context") or "")),
            decision="summarize" if ratio >= threshold else "keep_inline",
        )
        state.setdefault("context_budget_reports", []).append(report)
        state["context_budget_reports"] = state["context_budget_reports"][-24:]
        return state


class DynamicContextMiddleware(HarnessMiddleware):
    name = "dynamic_context"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assembler = ContextAssembler(settings)

    def before_policy(self, state: AgentState) -> AgentState:
        stage = "policy_round_%s" % int(state.get("react_round") or 0)
        injection = self.assembler.runtime_injection(state, stage=stage)
        state["runtime_injection"] = injection
        state["runtime_context"] = self.assembler.render_runtime_context(
            injection,
            budget_chars=int(self.settings.context_runtime_budget_chars or 6000),
        )
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="injected",
            code="RUNTIME_CONTEXT_INJECTED",
            message="dynamic runtime context is injected through middleware",
            input_chars=len(json.dumps(injection, ensure_ascii=False, default=str)),
            output_chars=len(state.get("runtime_context") or ""),
            metadata={
                "stage": stage,
                "artifactCount": (injection.get("workspace") or {}).get("artifactCount", 0),
                "hasThreadContext": bool((injection.get("threadContext") or {}).get("restored")),
            },
        )
        return state


class SummarizeMiddleware(HarnessMiddleware):
    name = "summarize"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        reports = state.get("context_budget_reports") or []
        report = reports[-1] if reports else None
        if not report or not context_report_over_budget(report):
            return state
        stage = context_report_stage(report)
        if stage in set(state.get("_summarized_stages", [])):
            return state
        before_tokens = context_report_estimated_tokens(report)
        target_tokens = int(max(1, (self.settings.context_window_tokens or 1) * (self.settings.context_compaction_target_ratio or 0.4)))
        checkpoint = write_middleware_json_artifact(
            state,
            self.settings,
            "context",
            "runtime_checkpoint_%s.json" % stage,
            build_runtime_checkpoint_payload(state, stage),
        )
        if checkpoint and (checkpoint.path or checkpoint.relative_path or checkpoint.merchant_uri):
            state.setdefault("runtime_checkpoints", []).append(checkpoint.model_dump(by_alias=True))
            state["runtime_checkpoints"] = state["runtime_checkpoints"][-8:]
        summary = build_compression_summary(state, target_tokens * 4)
        artifact = write_middleware_text_artifact(state, self.settings, "context", "summary_%s.md" % stage, summary)
        state["summary_context"] = summary
        event = ContextCompressionEvent(
            stage=stage,
            before_tokens=before_tokens,
            after_tokens=estimate_text_tokens(summary),
            target_ratio=float(self.settings.context_compaction_target_ratio or 0.4),
            summary_artifact=artifact,
            protected_keys=protected_fact_keys(state),
            reason="context budget exceeded %.2f threshold" % float(self.settings.context_compaction_threshold_ratio or 0.85),
        )
        state.setdefault("context_compression_events", []).append(event)
        state["context_compression_events"] = state["context_compression_events"][-12:]
        state.setdefault("_summarized_stages", []).append(stage)
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="compressed",
            code="CONTEXT_SUMMARIZED",
            message="context compressed to summary artifact after runtime checkpoint flush",
            artifact_refs=[ref for ref in [checkpoint, artifact] if ref and (ref.path or ref.relative_path or ref.merchant_uri)],
            input_chars=before_tokens * 4,
            output_chars=len(summary),
        )
        return state


class FileSystemContextMiddleware(HarnessMiddleware):
    name = "filesystem_context"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        manifest = build_workspace_manifest(state, self.settings)
        state["workspace_manifest"] = manifest
        thread_data = state.get("thread_data")
        outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
        if outputs_path:
            try:
                path = Path(outputs_path) / "workspace_manifest.json"
                path.write_text(json.dumps(manifest.model_dump(by_alias=True), ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            except Exception:
                pass
        return state


class ArtifactOffloadMiddleware(HarnessMiddleware):
    name = "artifact_offload"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        run_result = state.get("agent_run_result")
        task_results = getattr(run_result, "task_results", []) if run_result is not None else []
        seen = set(state.get("_middleware_offloaded_tasks", []))
        for task_result in task_results:
            task_id = str(getattr(task_result, "task_id", "") or "")
            if not task_id or task_id in seen:
                continue
            bundle = getattr(task_result, "query_bundle", None)
            rows = getattr(bundle, "rows", []) if bundle is not None else []
            if not isinstance(rows, list) or not rows:
                continue
            estimated_chars = len(json.dumps(rows, ensure_ascii=False, default=str))
            if len(rows) <= int(self.settings.context_artifact_inline_max_rows or 20) and estimated_chars <= int(self.settings.tool_result_offload_chars or 20000):
                continue
            artifact = write_middleware_json_artifact(state, self.settings, "tool_results", "%s_rows.json" % task_id, rows)
            seen.add(task_id)
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="offloaded",
                code="TOOL_RESULT_OFFLOADED",
                message="large node result was written to workspace artifact",
                artifact_refs=[artifact],
                input_chars=estimated_chars,
                output_chars=0,
                metadata={"taskId": task_id, "rowCount": len(rows)},
            )
        state["_middleware_offloaded_tasks"] = sorted(seen)
        return state


class ToolCallRecoveryMiddleware(HarnessMiddleware):
    name = "tool_call_recovery"

    def before_policy(self, state: AgentState) -> AgentState:
        results = state.get("tool_call_results") or []
        ledger_keys = {
            "%s:%s" % (entry.tool_call_id, entry.status)
            for entry in state.get("tool_call_ledger", [])
            if hasattr(entry, "tool_call_id")
        }
        normalized: List[Any] = []
        for item in results:
            result = normalize_tool_result(item)
            if not result.status or result.status not in TERMINAL_TOOL_STATUSES:
                before = result.status
                result.status = "failed"
                result.error_type = result.error_type or "MISSING_TOOL_RESULT"
                result.error_message = result.error_message or "tool call did not produce a terminal result"
                event = ToolCallRecoveryEvent(
                    tool_call_id=result.id,
                    tool_name=result.name,
                    stage="before_policy",
                    action="patch_missing_terminal_result",
                    reason="non terminal tool status would break provider message order",
                    status_before=before,
                    status_after=result.status,
                )
                state.setdefault("tool_call_recovery_events", []).append(event)
                append_middleware_event(
                    state,
                    self.name,
                    "before_policy",
                    status="patched",
                    code="DANGLING_TOOL_CALL_PATCHED",
                    message="patched non-terminal tool call result",
                    metadata={"toolCallId": result.id, "toolName": result.name, "statusBefore": before},
                )
            key = "%s:%s" % (result.id, result.status)
            if key not in ledger_keys:
                state.setdefault("tool_call_ledger", []).append(
                    ToolCallLedgerEntry(
                        tool_call_id=result.id,
                        tool_name=result.name,
                        stage="before_policy",
                        status=result.status,
                        duration_ms=result.duration_ms,
                        attempts=result.attempts,
                        cache_hit=result.cache_hit,
                        rate_limited=result.rate_limited,
                        target=result.target,
                        error_type=result.error_type,
                        error_message=result.error_message,
                    )
                )
                ledger_keys.add(key)
            normalized.append(result)
        state["tool_call_results"] = normalized[-100:]
        state["tool_call_ledger"] = state.get("tool_call_ledger", [])[-200:]
        state["tool_call_recovery_events"] = state.get("tool_call_recovery_events", [])[-50:]
        return state


class LoopGuardMiddleware(HarnessMiddleware):
    name = "loop_guard"

    def __init__(self, settings: Settings):
        self.settings = settings

    def before_policy(self, state: AgentState) -> AgentState:
        if state.get("middleware_loop_blocked"):
            return state
        history = state.get("action_history") or []
        threshold = max(2, int(self.settings.middleware_loop_guard_threshold or 3))
        if len(history) < threshold:
            return state
        recent = history[-threshold:]
        actions = [str(getattr(item, "action", "") or "") for item in recent]
        if len(set(actions)) != 1 or actions[0] in {"answer_data", "answer_rule", "cache_answer", "ask_human"}:
            return state
        state["middleware_loop_blocked"] = True
        state["query_graph_validation_result"] = GraphValidationResult(
            valid=False,
            gaps=[
                GraphValidationGap(
                    code="LOOP_DETECTED",
                    reason="LeadAgent repeated the same action without observable progress",
                    evidence="action=%s repeats=%d" % (actions[0], threshold),
                )
            ],
            repairable=False,
        )
        state["query_graph_validated"] = True
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="blocked",
            code="LOOP_DETECTED",
            message="blocked repeated action pattern: %s" % actions[0],
            metadata={"actions": actions},
        )
        return state


class MemoryMiddleware(HarnessMiddleware):
    name = "memory"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.memory_store = create_memory_store(settings)

    def before_policy(self, state: AgentState) -> AgentState:
        injection = self.memory_store.select_for_question(
            state,
            budget_tokens=int(self.settings.context_memory_budget_tokens or 1200),
        )
        state["memory_injection"] = injection
        state["memory_injection_trace"] = injection.get("memoryInjectionTrace", {})
        constraints = build_memory_constraints(injection)
        state["memory_constraints"] = constraints
        state["memory_constraint_trace"] = {
            "constraintCount": len(constraints),
            "requiredCount": sum(1 for item in constraints if str(item.get("enforcement") or "") == "required"),
            "clarifyCount": sum(1 for item in constraints if str(item.get("enforcement") or "") == "clarify_or_disclose"),
            "source": injection.get("source", ""),
            "selectedIds": (injection.get("memoryInjectionTrace") or {}).get("selectedIds", []),
        }
        rendered = self.memory_store.render_injection(injection)
        if rendered:
            state["memory_context"] = truncate_memory_text_by_tokens(
                rendered,
                int(self.settings.context_memory_budget_tokens or 1200),
            )
        recent_focus = state.get("recent_focus")
        memory_context = str(state.get("memory_context") or "")
        append_middleware_event(
            state,
            self.name,
            "before_policy",
            status="observed",
            code="MEMORY_CONTEXT_READY" if memory_context or (recent_focus and not recent_focus.is_empty()) else "MEMORY_CONTEXT_EMPTY",
            message="memory is injected as compact business focus instead of full chat history",
            metadata={
                "memoryContextChars": len(memory_context),
                "recentFocusEmpty": bool(recent_focus.is_empty()) if recent_focus else True,
                "eventCount": len(injection.get("relevantEvents") or []),
                "candidateCount": (injection.get("memoryInjectionTrace") or {}).get("candidateCount", 0),
                "selectedIds": (injection.get("memoryInjectionTrace") or {}).get("selectedIds", []),
                "budgetUsedChars": (injection.get("memoryInjectionTrace") or {}).get("budgetUsedChars", 0),
                "filteredReasons": (injection.get("memoryInjectionTrace") or {}).get("filteredReasons", {}),
                "memorySource": injection.get("source", ""),
                "memoryConstraintCount": len(constraints),
            },
        )
        return state


class SkillMiddleware(HarnessMiddleware):
    name = "skill"

    def before_policy(self, state: AgentState) -> AgentState:
        loaded = state.get("loaded_skills") or []
        if loaded and state.get("_skill_middleware_loaded") != loaded:
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="observed",
                code="SKILL_HEADERS_LOADED",
                message="skills are exposed as thin headers; details load through semantic assets",
                metadata={"loadedSkills": loaded},
            )
            state["_skill_middleware_loaded"] = list(loaded)
        return state


class ProviderCompatibilityMiddleware(HarnessMiddleware):
    name = "provider_compatibility"

    def before_policy(self, state: AgentState) -> AgentState:
        trace = state.get("bounded_lead_llm_trace") or {}
        if trace and trace.get("status") == "error":
            append_middleware_event(
                state,
                self.name,
                "before_policy",
                status="observed",
                code="LEAD_LLM_DECISION_COMPAT_ERROR",
                message=str(trace.get("reason") or "bounded lead LLM decision failed"),
                metadata=trace,
            )
        return state


class ClarificationMiddleware(HarnessMiddleware):
    """Turn human clarification into a virtual tool-call result.

    The frontend contract stays as ChatResponse.clarification, but internally the
    run now has an ask_clarification tool message that mirrors LangGraph-style
    tool interception.
    """

    name = "clarification"

    def before_policy(self, state: AgentState) -> AgentState:
        if state.get("human_clarification_required"):
            return self._intercept(state, stage="before_policy", reason="state_requires_clarification")
        return state

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        if decision.selected_action == "ask_human" or state.get("human_clarification_required"):
            return self._intercept(state, stage="before_action", reason="ask_human_selected")
        return state

    def _intercept(self, state: AgentState, stage: str, reason: str) -> AgentState:
        if state.get("_clarification_tool_intercepted"):
            return state
        question = str(state.get("human_clarification_question") or "").strip()
        if not question:
            question = "我还需要补充确认后才能继续处理这个问题。"
        options = [str(item) for item in (state.get("human_clarification_options") or []) if str(item).strip()]
        context = {
            "stage": str(state.get("human_clarification_stage") or ""),
            "type": str(state.get("human_clarification_type") or ""),
            "pendingQuestion": str(state.get("question") or ""),
            "reason": reason,
        }
        tool_call_id = "ask_clarification:%s" % (state.get("qa_id") or state.get("run_id") or uuid.uuid4().hex)
        content = format_clarification_message(question, options)
        tool_message = {
            "toolName": "ask_clarification",
            "status": "success",
            "type": "clarification_request",
            "content": content,
            "question": question,
            "options": options,
            "context": context,
            "endsRun": True,
        }
        result = ToolCallExecutionResult(
            id=tool_call_id,
            name="ask_clarification",
            status="success",
            result={"question": question, "options": options, "context": context},
            service_name="human",
            tool_message=tool_message,
            attempts=1,
            runtime_events=[
                {
                    "eventType": "tool.intercepted",
                    "toolCallId": tool_call_id,
                    "toolName": "ask_clarification",
                    "status": "success",
                    "middleware": self.name,
                    "createdAt": datetime.now().isoformat(),
                    "payload": {"stage": stage, "reason": reason, "goto": "END"},
                }
            ],
        )
        state["clarification_tool_message"] = tool_message
        state["clarification_command"] = {
            "update": {"messages": [tool_message]},
            "goto": "END",
            "reason": "clarification_required",
        }
        state.setdefault("tool_call_results", []).append(result)
        state["tool_call_results"] = state["tool_call_results"][-100:]
        state.setdefault("tool_runtime_events", []).extend(result.runtime_events)
        state["tool_runtime_events"] = state["tool_runtime_events"][-200:]
        ensure_tool_ledger_entry(state, result, stage)
        state["_clarification_tool_intercepted"] = True
        append_middleware_event(
            state,
            self.name,
            stage,
            status="intercepted",
            code="CLARIFICATION_TOOL_INTERCEPTED",
            message="ask_clarification was intercepted and converted to a terminal user clarification",
            metadata={"toolCallId": tool_call_id, "goto": "END", "optionsCount": len(options), "reason": reason},
        )
        return state


class ContextSnapshotMiddleware(HarnessMiddleware):
    name = "context_snapshot"

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def before_action(self, state: AgentState, decision: AgentDecision) -> AgentState:
        state.setdefault("middleware_action_context_hashes", {})
        packages = state.get("context_packages") or []
        last_hash = ""
        if packages:
            last = packages[-1]
            last_hash = str(getattr(last, "context_hash", "") or (last.get("contextHash", "") if isinstance(last, dict) else ""))
        state["middleware_action_context_hashes"][decision.selected_action] = last_hash
        return state


def default_harness_middlewares(settings: Settings, context_manager: ContextManager) -> List[HarnessMiddleware]:
    return [
        CancellationMiddleware(settings),
        PermissionMiddleware(),
        ActionContractMiddleware(),
        ProviderCompatibilityMiddleware(),
        ClarificationMiddleware(),
        ToolCallRecoveryMiddleware(),
        LoopGuardMiddleware(settings),
        MemoryMiddleware(settings),
        DynamicContextMiddleware(settings),
        ContextBudgetMiddleware(settings),
        SummarizeMiddleware(settings),
        ArtifactOffloadMiddleware(settings),
        FileSystemContextMiddleware(settings),
        SkillMiddleware(),
        ContextSnapshotMiddleware(context_manager),
    ]


def append_middleware_event(
    state: AgentState,
    middleware: str,
    stage: str,
    status: str = "ok",
    code: str = "",
    message: str = "",
    input_chars: int = 0,
    output_chars: int = 0,
    artifact_refs: Optional[List[ArtifactRef]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    event = MiddlewareEvent(
        event_id="mw_" + uuid.uuid4().hex,
        middleware=middleware,
        stage=stage,
        status=status,
        code=code,
        message=message,
        input_chars=input_chars,
        output_chars=output_chars,
        artifact_refs=artifact_refs or [],
        metadata=metadata or {},
    )
    state.setdefault("middleware_events", []).append(event)
    state["middleware_events"] = state["middleware_events"][-200:]


def state_path_ready(state: AgentState, path: str) -> bool:
    value: Any = state
    for part in [item for item in str(path or "").split(".") if item]:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return False
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def format_clarification_message(question: str, options: List[str]) -> str:
    if not options:
        return question
    lines = [question, "", "可选项："]
    for index, option in enumerate(options, start=1):
        lines.append("%d. %s" % (index, option))
    return "\n".join(lines)


def ensure_tool_ledger_entry(state: AgentState, result: ToolCallExecutionResult, stage: str) -> None:
    key = "%s:%s" % (result.id, result.status)
    existing = {
        "%s:%s" % (entry.tool_call_id, entry.status)
        for entry in state.get("tool_call_ledger", [])
        if hasattr(entry, "tool_call_id")
    }
    if key in existing:
        return
    state.setdefault("tool_call_ledger", []).append(
        ToolCallLedgerEntry(
            tool_call_id=result.id,
            tool_name=result.name,
            stage=stage,
            status=result.status,
            duration_ms=result.duration_ms,
            attempts=result.attempts,
            cache_hit=result.cache_hit,
            rate_limited=result.rate_limited,
            target=result.target,
            error_type=result.error_type,
            error_message=result.error_message,
        )
    )
    state["tool_call_ledger"] = state["tool_call_ledger"][-200:]


def estimate_context_tokens(state: AgentState) -> int:
    chars = 0
    for key in [
        "question",
        "merchant_profile_context",
        "memory_context",
        "runtime_context",
        "base_knowledge_context",
        "topic_asset_context",
        "recall_context",
        "session_context",
        "summary_context",
        "tool_context",
    ]:
        chars += len(str(state.get(key) or ""))
    for key in ["route_slots", "intent_signals", "fast_understanding"]:
        value = state.get(key)
        if value is not None:
            chars += len(safe_json(value))
    plan = state.get("plan")
    if plan is not None:
        chars += len(safe_json(getattr(plan, "question_understanding", {}) or {}))
        chars += min(8000, len(safe_json(getattr(plan, "compiler_trace", []) or [])))
    snapshots = state.get("context_snapshots") or []
    chars += sum(min(4000, len(safe_json(item))) for item in snapshots[-4:])
    chars += sum(min(1200, len(safe_json(item))) for item in state.get("context_assembly_reports", [])[-4:])
    return estimate_text_tokens("x" * chars)


def snapshot_protected_fact_count(snapshot: Any) -> int:
    if isinstance(snapshot, dict):
        return len(snapshot.get("protectedFacts") or snapshot.get("protected_facts") or [])
    return len(getattr(snapshot, "protected_facts", []) or [])


def context_report_over_budget(report: Any) -> bool:
    if isinstance(report, dict):
        return bool(report.get("overBudget") or report.get("over_budget"))
    return bool(getattr(report, "over_budget", False))


def context_report_stage(report: Any) -> str:
    if isinstance(report, dict):
        return str(report.get("stage") or "policy")
    return str(getattr(report, "stage", "policy") or "policy")


def context_report_estimated_tokens(report: Any) -> int:
    if isinstance(report, dict):
        return int(report.get("estimatedTokens") or report.get("estimated_tokens") or 0)
    return int(getattr(report, "estimated_tokens", 0) or 0)


def estimate_text_tokens(text: str) -> int:
    value = str(text or "")
    if not value:
        return 1
    cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", value))
    non_cjk_count = max(0, len(value) - cjk_count)
    return max(1, cjk_count + int((non_cjk_count + 3) / 4))


def safe_json(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(by_alias=True)
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


def build_compression_summary(state: AgentState, max_chars: int) -> str:
    lines: List[str] = []
    lines.append("# Compressed Runtime Context")
    question = str(state.get("question") or "")
    if question:
        lines.append("\n## Current Question\n%s" % question[:1200])
    facts = []
    for snapshot in state.get("context_snapshots", [])[-4:]:
        if isinstance(snapshot, dict):
            facts.extend(snapshot.get("protectedFacts") or [])
    if facts:
        lines.append("\n## Protected Facts")
        for fact in facts[:24]:
            key = fact.get("key") if isinstance(fact, dict) else getattr(fact, "key", "")
            value = fact.get("value") if isinstance(fact, dict) else getattr(fact, "value", "")
            if key or value:
                lines.append("- %s=%s" % (key, str(value)[:500]))
    history = state.get("action_history") or []
    if history:
        lines.append("\n## Recent Actions")
        for item in history[-10:]:
            action = getattr(item, "action", "") if hasattr(item, "action") else item.get("action", "")
            status = getattr(item, "status", "") if hasattr(item, "status") else item.get("status", "")
            reason = getattr(item, "reason", "") if hasattr(item, "reason") else item.get("reason", "")
            lines.append("- %s %s %s" % (action, status, str(reason)[:240]))
    validation = state.get("query_graph_validation_result")
    gaps = getattr(validation, "gaps", []) if validation is not None else []
    if gaps:
        lines.append("\n## Current Gaps")
        for gap in gaps[:12]:
            lines.append("- %s %s" % (getattr(gap, "code", ""), str(getattr(gap, "message", ""))[:240]))
    return "\n".join(lines)[: max(1000, max_chars)]


def build_runtime_checkpoint_payload(state: AgentState, stage: str) -> Dict[str, Any]:
    validation = state.get("query_graph_validation_result")
    run_result = state.get("agent_run_result")
    plan = state.get("plan")
    manifest = state.get("workspace_manifest")
    return {
        "version": "runtime_checkpoint.v1",
        "stage": stage,
        "createdAt": datetime.now().isoformat(),
        "question": state.get("question", ""),
        "runId": state.get("run_id", ""),
        "threadId": state.get("thread_id", ""),
        "queryGraph": plan.model_dump(by_alias=True) if hasattr(plan, "model_dump") else {},
        "validation": validation.model_dump(by_alias=True) if hasattr(validation, "model_dump") else {},
        "evidenceGaps": [
            gap.model_dump(by_alias=True) if hasattr(gap, "model_dump") else gap
            for gap in getattr(run_result, "evidence_gaps", [])[:24]
        ]
        if run_result is not None
        else [],
        "memoryConstraints": state.get("memory_constraints", []),
        "memoryConstraintTrace": state.get("memory_constraint_trace", {}),
        "runtimeInjection": state.get("runtime_injection", {}),
        "threadContext": state.get("thread_context", {}),
        "artifactManifest": manifest.model_dump(by_alias=True) if hasattr(manifest, "model_dump") else manifest,
        "recentActions": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("action_history", [])[-12:]
        ],
        "recentSteps": [
            item.model_dump(by_alias=True) if hasattr(item, "model_dump") else item
            for item in state.get("run_steps", [])[-12:]
        ],
    }


def protected_fact_keys(state: AgentState) -> List[str]:
    keys: List[str] = []
    for snapshot in state.get("context_snapshots", [])[-4:]:
        if not isinstance(snapshot, dict):
            continue
        for fact in snapshot.get("protectedFacts") or []:
            key = str(fact.get("key") or "")
            if key and key not in keys:
                keys.append(key)
    return keys[:40]


def write_middleware_text_artifact(state: AgentState, settings: Settings, namespace: str, name: str, content: str) -> ArtifactRef:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return ArtifactRef()
    store = WorkspaceArtifactStore(settings, Path(outputs_path) / "artifacts")
    artifact = store.write_text(namespace, name, content)
    ref = artifact_ref_from_path(artifact.get("path", ""), namespace=namespace, reason="middleware context artifact")
    ref.merchant_uri = artifact.get("merchantUri", ref.merchant_uri)
    ref.relative_path = artifact.get("relativePath", ref.relative_path)
    return ref


def write_middleware_json_artifact(state: AgentState, settings: Settings, namespace: str, name: str, payload: Any) -> ArtifactRef:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return ArtifactRef()
    store = WorkspaceArtifactStore(settings, Path(outputs_path) / "artifacts")
    artifact = store.write_json(namespace, name, payload, preview_chars=0)
    ref = artifact_ref_from_path(artifact.get("path", ""), namespace=namespace, reason="middleware offloaded tool result")
    ref.merchant_uri = artifact.get("merchantUri", ref.merchant_uri)
    ref.relative_path = artifact.get("relativePath", ref.relative_path)
    return ref


def workspace_entries(state: AgentState) -> List[Path]:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    if not outputs_path:
        return []
    root = Path(outputs_path)
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def build_workspace_manifest(state: AgentState, settings: Settings) -> WorkspaceManifest:
    thread_data = state.get("thread_data")
    outputs_path = getattr(thread_data, "outputs_path", "") if thread_data is not None else ""
    root = Path(outputs_path) if outputs_path else Path()
    entries: List[WorkspaceManifestEntry] = []
    if root and root.exists():
        for path in workspace_entries(state)[: max(1, int(settings.middleware_max_manifest_entries or 200))]:
            if path.name == "workspace_manifest.json":
                continue
            try:
                relative = str(path.relative_to(root))
            except ValueError:
                relative = str(path)
            entries.append(
                WorkspaceManifestEntry(
                    path=str(path),
                    relative_path=relative,
                    namespace=relative.split("/", 1)[0] if "/" in relative else "root",
                    bytes=path.stat().st_size,
                    estimated_chars=path.stat().st_size,
                    sha256=file_sha256(path),
                    merchant_uri=merchant_uri_for_artifact(relative),
                    updated_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                )
            )
    return WorkspaceManifest(
        root=str(root) if outputs_path else "",
        entries=entries,
        entry_count=len(entries),
        total_bytes=sum(item.bytes for item in entries),
        updated_at=datetime.now().isoformat(),
    )


def file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except Exception:
        return ""


def normalize_tool_result(item: Any) -> ToolCallExecutionResult:
    if isinstance(item, ToolCallExecutionResult):
        return item
    if isinstance(item, dict):
        return ToolCallExecutionResult.model_validate(item)
    return ToolCallExecutionResult(id=str(getattr(item, "id", "")), name=str(getattr(item, "name", "")), status=str(getattr(item, "status", "")))
