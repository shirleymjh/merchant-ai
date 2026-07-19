from __future__ import annotations

import json
import hashlib
import inspect
import shutil
import time
import uuid
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from typing import Any, Callable, Dict, List, Optional

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import (
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStatus,
    AgentThreadRecord,
    ArtifactRef,
    ChatContext,
    ChatResponse,
    ConversationMessage,
    RunCreateRequest,
)
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.checkpoints import checkpoint_ref_for_run, prune_completed_sqlite_checkpoints
from merchant_ai.services.runtime_state import create_runtime_state_store
from merchant_ai.services.security import identity_scope_hash, identity_scope_payload
from merchant_ai.services.text_parsing import is_ascii_hex


ANSWER_DELTA_CHARS = 80
SERVER_THREAD_HISTORY_RUNS = 8


def valid_thread_id(thread_id: str) -> bool:
    return _valid_runtime_id(thread_id, "thread_")


def valid_run_id(run_id: str) -> bool:
    return _valid_runtime_id(run_id, "run_")


def _valid_runtime_id(value: str, prefix: str) -> bool:
    text = str(value or "")
    return text.startswith(prefix) and is_ascii_hex(text[len(prefix) :], minimum=32, maximum=32)


def call_run_chat(
    run_chat: Callable[..., ChatResponse],
    message: str,
    merchant_id: str,
    context: Optional[ChatContext],
    listener: Callable[[str, str, Dict[str, Any]], None],
    thread_id: str,
    run_id: str,
    message_history: Optional[List[Any]] = None,
) -> ChatResponse:
    try:
        signature = inspect.signature(run_chat)
        supports_history = "message_history" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
    except Exception:
        supports_history = False
    if supports_history:
        response = run_chat(message, merchant_id, context, listener, thread_id, run_id, message_history=message_history or [])
    else:
        response = run_chat(message, merchant_id, context, listener, thread_id, run_id)
    return response


class FileRunEventStore:
    def __init__(self, settings: Settings):
        self.root = settings.resolved_workspace_path / "run_events"
        self.threads_dir = self.root / "threads"
        self.runs_dir = self.root / "runs"
        self.events_dir = self.root / "events"
        self.traces_dir = self.root / "traces"
        for path in [self.threads_dir, self.runs_dir, self.events_dir, self.traces_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.cleanup_expired(int(settings.agent_run_retention_days or 14))

    def save_thread(self, thread: AgentThreadRecord) -> None:
        self._write_json(self.threads_dir / ("%s.json" % thread.thread_id), thread.model_dump(by_alias=True))

    def load_thread(self, thread_id: str) -> Optional[AgentThreadRecord]:
        data = self._read_json(self.threads_dir / ("%s.json" % thread_id))
        return AgentThreadRecord.model_validate(data) if data else None

    def save_run(self, run: AgentRunRecord) -> None:
        self._write_json(self.runs_dir / ("%s.json" % run.run_id), run.model_dump(by_alias=True))

    def load_run(self, run_id: str) -> Optional[AgentRunRecord]:
        data = self._read_json(self.runs_dir / ("%s.json" % run_id))
        return AgentRunRecord.model_validate(data) if data else None

    def list_runs(self, limit: int = 50, status: str = "", merchant_id: str = "") -> List[AgentRunRecord]:
        runs: List[AgentRunRecord] = []
        for path in sorted(self.runs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            data = self._read_json(path)
            if not data:
                continue
            try:
                run = AgentRunRecord.model_validate(data)
            except Exception:
                continue
            if status and run_status_value(run.status).upper() != status.upper():
                continue
            if merchant_id and run.merchant_id != merchant_id:
                continue
            runs.append(run)
            if len(runs) >= limit:
                break
        runs.sort(key=lambda item: item.start_time, reverse=True)
        return runs

    def append_event(self, event: AgentRunEventRecord) -> None:
        path = self.events_dir / ("%s.jsonl" % event.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(by_alias=True), ensure_ascii=False, default=str) + "\n")

    def load_events(self, run_id: str) -> List[AgentRunEventRecord]:
        path = self.events_dir / ("%s.jsonl" % run_id)
        if not path.exists():
            return []
        events: List[AgentRunEventRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(AgentRunEventRecord.model_validate(json.loads(line)))
            except Exception:
                continue
        return events

    def save_trace(self, run_id: str, trace: Dict[str, Any]) -> Path:
        path = self.traces_dir / ("%s.trace_replay.v2.json" % run_id)
        self._write_json(path, trace)
        return path

    def load_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self.traces_dir / ("%s.trace_replay.v2.json" % run_id))

    def cleanup_expired(self, retention_days: int) -> None:
        cutoff = time.time() - max(1, retention_days) * 86400
        for directory in [self.runs_dir, self.events_dir, self.traces_dir]:
            for path in directory.glob("*"):
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
        workspace_threads = self.root.parent / "threads"
        if workspace_threads.exists():
            for thread_dir in workspace_threads.iterdir():
                try:
                    if thread_dir.is_dir() and thread_dir.stat().st_mtime < cutoff:
                        shutil.rmtree(thread_dir)
                except OSError:
                    continue

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
        except Exception:
            return None
        return None


class AgentRunManager:
    def __init__(self, settings: Optional[Settings] = None, store: Optional[FileRunEventStore] = None):
        self.settings = settings or get_settings()
        self.store = store or FileRunEventStore(self.settings)
        self.threads: Dict[str, AgentThreadRecord] = {}
        self.runs: Dict[str, AgentRunRecord] = {}
        self.run_events: Dict[str, List[AgentRunEventRecord]] = {}
        self.runtime_state_store = create_runtime_state_store(self.settings)
        self._lock = RLock()
        try:
            prune_completed_sqlite_checkpoints(self.settings)
        except (OSError, ValueError):
            pass

    def create_thread(
        self,
        merchant_id: str,
        topic: str = "",
        context: Optional[ChatContext] = None,
        identity: Any = None,
    ) -> AgentThreadRecord:
        thread_id = "thread_" + uuid.uuid4().hex
        identity = identity or (getattr(context, "user_identity", None) if context is not None else None)
        scope = identity_scope_payload(identity, merchant_id)
        record = AgentThreadRecord(
            thread_id=thread_id,
            merchant_id=merchant_id,
            topic=topic or "",
            context=context,
            owner_user_id=str(scope.get("userId") or ""),
            owner_role=str(scope.get("role") or ""),
            owner_scope_hash=identity_scope_hash(identity, merchant_id),
        )
        self.threads[thread_id] = record
        self.store.save_thread(record)
        return record

    def get_thread(self, thread_id: str) -> Optional[AgentThreadRecord]:
        if not valid_thread_id(thread_id):
            return None
        thread = self.threads.get(thread_id) or self.store.load_thread(thread_id)
        if thread:
            self.threads[thread_id] = thread
        return thread

    def create_run(
        self,
        thread_id: str,
        merchant_id: str,
        question: str,
        initial_status: AgentRunStatus = AgentRunStatus.RUNNING,
        identity: Any = None,
    ) -> AgentRunRecord:
        if not valid_thread_id(thread_id):
            raise ValueError("invalid threadId")
        thread = self.get_thread(thread_id)
        if not thread:
            raise ValueError("thread not found")
        if thread.merchant_id != merchant_id:
            raise ValueError("thread merchant does not match run merchant")
        if thread.owner_scope_hash and identity is not None:
            if thread.owner_scope_hash != identity_scope_hash(identity, merchant_id):
                raise ValueError("thread identity scope does not match run identity")
        run_id = "run_" + uuid.uuid4().hex
        run = AgentRunRecord(
            run_id=run_id,
            thread_id=thread_id,
            merchant_id=merchant_id,
            question=question,
            status=initial_status,
            start_time=datetime.now(),
            checkpoint_ref=checkpoint_ref_for_run(self.settings, thread_id, run_id),
            resumable=False,
        )
        with self._lock:
            self.runs[run_id] = run
            self.run_events[run_id] = []
            self.store.save_run(run)
        self.append_event(run_id, thread_id, "run.started", "RUN_MANAGER", {"question": question, "merchantId": merchant_id})
        return run

    def get_run(self, run_id: str) -> Optional[AgentRunRecord]:
        if not valid_run_id(run_id):
            return None
        run = self.runs.get(run_id) or self.store.load_run(run_id)
        if run:
            self.runs[run_id] = run
        return run

    def append_event(self, run_id: str, thread_id: str, event_type: str, node: str = "", payload: Optional[Dict[str, Any]] = None) -> None:
        safe_payload = compact_event_payload(payload or {})
        event = AgentRunEventRecord(
            event_id="event_" + uuid.uuid4().hex,
            run_id=run_id,
            thread_id=thread_id,
            event_type=event_type,
            node=node,
            step_id=str(safe_payload.get("stepId") or safe_payload.get("step_id") or ""),
            tool_call_id=str(safe_payload.get("toolCallId") or safe_payload.get("tool_call_id") or ""),
            parent_id=str(safe_payload.get("parentId") or safe_payload.get("parent_id") or ""),
            payload=safe_payload,
        )
        self.run_events.setdefault(run_id, []).append(event)
        self.store.append_event(event)

    def events(self, run_id: str) -> List[AgentRunEventRecord]:
        events = self.run_events.get(run_id) or self.store.load_events(run_id)
        if events:
            self.run_events[run_id] = events
        return events

    def complete_run(self, run_id: str, response: ChatResponse) -> None:
        run = self.runs.get(run_id)
        if run:
            if run.status == AgentRunStatus.CANCELED:
                self.append_event(run_id, run.thread_id, "run.completion_ignored", "RUN_MANAGER", {"reason": "run canceled"})
                return
            run.status = AgentRunStatus.COMPLETED
            run.answer = response
            run.end_time = datetime.now()
            run.final_answer_hash = hashlib.sha256((response.answer or "").encode("utf-8")).hexdigest()[:16]
            trace = response.debug_trace or {}
            harness = trace.get("harness") or {}
            performance = (
                dict(harness.get("performance"))
                if isinstance(harness.get("performance"), dict)
                else {}
            )
            runtime_budget = (
                dict(harness.get("runtimeBudget"))
                if isinstance(harness.get("runtimeBudget"), dict)
                else {}
            )
            if runtime_budget:
                performance.setdefault("runtimeBudget", runtime_budget)
                performance.setdefault(
                    "totalDurationMs",
                    runtime_budget.get("elapsedMs", 0),
                )
            run.performance_summary = performance
            if isinstance(harness.get("checkpoint"), dict) and harness.get("checkpoint"):
                run.checkpoint_ref = harness["checkpoint"]
                run.resumable = bool(harness["checkpoint"].get("resumable"))
            elif run.checkpoint_ref and str(run.checkpoint_ref.get("backend") or "").lower() != "memory":
                run.checkpoint_ref["resumable"] = True
                run.resumable = True
            run.trace_path = str((harness.get("traceReplay") or {}).get("path") or harness.get("traceReplayPath") or "")
            run.updated_at = datetime.now()
            self.store.save_run(run)
            replay_payload = self._load_replay_payload(run.trace_path) or trace
            if replay_payload:
                trace_path = self.store.save_trace(run_id, replay_payload)
                run.trace_path = str(trace_path)
                run.artifact_refs = artifact_refs_from_trace(replay_payload)
                if not run.artifact_refs:
                    run.artifact_refs = [artifact_ref_from_path(str(trace_path), "trace", "persisted trace replay v2")]
            run.answer = ChatResponse.model_validate(public_response_payload(response))
            self.store.save_run(run)
            if bool(self.settings.agent_compact_success_artifacts_enabled):
                compact_completed_thread_outputs(self.settings.resolved_workspace_path, run.thread_id, run.run_id)
            self.append_event(
                run_id,
                run.thread_id,
                "run.completed",
                "RUN_MANAGER",
                {
                    "status": run.status,
                    "finalAnswerHash": run.final_answer_hash,
                    "performance": run.performance_summary,
                    "tracePath": run.trace_path,
                },
            )

    def fail_run(self, run_id: str, error: str) -> None:
        run = self.runs.get(run_id)
        if run:
            if run.status == AgentRunStatus.CANCELED:
                self.append_event(run_id, run.thread_id, "run.failure_ignored", "RUN_MANAGER", {"reason": "run canceled", "error": error})
                return
            run.status = AgentRunStatus.FAILED
            run.error = error
            run.end_time = datetime.now()
            run.updated_at = datetime.now()
            self.store.save_run(run)
            self.append_event(run_id, run.thread_id, "run.failed", "RUN_MANAGER", {"error": error})

    def cancel_run(self, run_id: str) -> Optional[AgentRunRecord]:
        run = self.runs.get(run_id)
        if run:
            if run.status in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
                self.append_event(
                    run_id,
                    run.thread_id,
                    "run.cancel.ignored",
                    "RUN_MANAGER",
                    {"status": run_status_value(run.status), "reason": "terminal status"},
                )
                return run
            run.status = AgentRunStatus.CANCELED
            run.end_time = datetime.now()
            run.updated_at = datetime.now()
            self.store.save_run(run)
            self.runtime_state_store.cancel_run(run_id, "run_manager.cancel_run")
            self.append_event(run_id, run.thread_id, "run.canceled", "RUN_MANAGER", {})
            return run

    def mark_run_running(self, run_id: str) -> Optional[AgentRunRecord]:
        run = self.runs.get(run_id) or self.store.load_run(run_id)
        if not run:
            return None
        if run.status == AgentRunStatus.CANCELED:
            return run
        run.status = AgentRunStatus.RUNNING
        run.updated_at = datetime.now()
        self.runs[run_id] = run
        self.store.save_run(run)
        self.append_event(run_id, run.thread_id, "run.worker.started", "RUN_MANAGER", {})
        return run

    def is_canceled(self, run_id: str) -> bool:
        run = self.runs.get(run_id) or self.store.load_run(run_id)
        return bool(run and run.status == AgentRunStatus.CANCELED)

    def trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.store.load_trace(run_id)

    def list_runs(self, limit: int = 50, status: str = "", merchant_id: str = "") -> List[AgentRunRecord]:
        stored = self.store.list_runs(limit=limit, status=status, merchant_id=merchant_id)
        by_id = {run.run_id: run for run in stored}
        for run in self.runs.values():
            if status and run_status_value(run.status).upper() != status.upper():
                continue
            if merchant_id and run.merchant_id != merchant_id:
                continue
            by_id[run.run_id] = run
        runs = sorted(by_id.values(), key=lambda item: item.start_time, reverse=True)
        return runs[: max(1, limit)]

    def thread_message_history(
        self,
        thread_id: str,
        merchant_id: str = "",
        exclude_run_id: str = "",
        max_runs: int = SERVER_THREAD_HISTORY_RUNS,
    ) -> List[ConversationMessage]:
        if not valid_thread_id(thread_id):
            return []
        runs = [
            run
            for run in self.list_runs(limit=200, merchant_id=merchant_id)
            if run.thread_id == thread_id
            and run.run_id != exclude_run_id
            and run_status_value(run.status).upper() == AgentRunStatus.COMPLETED.value
            and run.answer
        ]
        runs = sorted(runs, key=lambda item: item.start_time)[-max(1, int(max_runs or SERVER_THREAD_HISTORY_RUNS)) :]
        messages: List[ConversationMessage] = []
        for run in runs:
            question = str(run.question or "").strip()
            answer = str(getattr(run.answer, "answer", "") or "").strip()
            if question:
                messages.append(
                    ConversationMessage(
                        role="user",
                        text=question,
                        id="%s:user" % run.run_id,
                        local_id="server_thread:%s:user" % run.run_id,
                    )
                )
            if answer:
                messages.append(
                    ConversationMessage(
                        role="assistant",
                        text=answer,
                        id="%s:assistant" % run.run_id,
                        local_id="server_thread:%s:assistant" % run.run_id,
                    )
                )
        return messages

    def effective_message_history(
        self,
        thread_id: str,
        merchant_id: str,
        exclude_run_id: str,
        client_history: Optional[List[Any]] = None,
    ) -> List[Any]:
        server_history = self.thread_message_history(thread_id, merchant_id, exclude_run_id=exclude_run_id)
        return server_history if server_history else list(client_history or [])

    def dashboard(self, limit: int = 50, status: str = "", merchant_id: str = "") -> Dict[str, Any]:
        runs = self.list_runs(limit=limit, status=status, merchant_id=merchant_id)
        status_counts: Dict[str, int] = {}
        total_duration_ms = 0.0
        duration_count = 0
        slowest: List[Dict[str, Any]] = []
        recent_errors: List[Dict[str, Any]] = []
        for run in runs:
            status_key = run_status_value(run.status)
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            duration = run_duration_ms(run)
            if duration is not None:
                total_duration_ms += duration
                duration_count += 1
                slowest.append(run_summary_payload(run, duration))
            if run.error or run_status_value(run.status) == AgentRunStatus.FAILED.value:
                recent_errors.append(run_summary_payload(run, duration))
        slowest.sort(key=lambda item: float(item.get("durationMs") or 0), reverse=True)
        return {
            "totalRuns": len(runs),
            "statusCounts": status_counts,
            "avgDurationMs": round(total_duration_ms / duration_count, 2) if duration_count else 0,
            "slowestRuns": slowest[:10],
            "recentErrors": recent_errors[:10],
            "runs": [run_summary_payload(run, run_duration_ms(run)) for run in runs],
        }

    def _load_replay_payload(self, path: str) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        try:
            file_path = Path(path)
            if file_path.exists():
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else None
        except Exception:
            return None
        return None


class AgentRunStreamService:
    def __init__(self, run_manager: AgentRunManager, run_chat: Callable[..., ChatResponse], default_merchant_id: str):
        self.run_manager = run_manager
        self.run_chat = run_chat
        self.default_merchant_id = default_merchant_id

    def stream(self, request: RunCreateRequest):
        q: "Queue[Dict[str, Any]]" = Queue()
        merchant_id = request.merchant_id or self.default_merchant_id
        thread_id = request.thread_id
        if not thread_id:
            thread = self.run_manager.create_thread(
                merchant_id,
                request.context.topic if request.context else "",
                request.context,
                identity=request.user_identity,
            )
            thread_id = thread.thread_id
        identity = request.user_identity or (getattr(request.context, "user_identity", None) if request.context is not None else None)
        run = self.run_manager.create_run(thread_id, merchant_id, request.message, identity=identity)
        message_history = self.run_manager.effective_message_history(
            thread_id,
            merchant_id,
            run.run_id,
            request.message_history,
        )
        started_payload = {"runId": run.run_id, "threadId": thread_id, "merchantId": merchant_id}
        self.run_manager.append_event(run.run_id, thread_id, "run.started", "RUN_STREAM", started_payload)
        q.put({"event": "run.started", "node": "RUN_STREAM", "payload": started_payload})
        streamed_answer = ""

        def enqueue_answer(answer: str) -> None:
            nonlocal streamed_answer
            text = str(answer or "")
            if not text or streamed_answer:
                return
            streamed_answer = text
            for index, chunk in enumerate(answer_chunks(text, ANSWER_DELTA_CHARS)):
                payload = {
                    "runId": run.run_id,
                    "threadId": thread_id,
                    "index": index,
                    "delta": chunk,
                }
                self.run_manager.append_event(run.run_id, thread_id, "answer.delta", "ANSWER_STREAM", payload)
                q.put({"event": "answer.delta", "node": "ANSWER_STREAM", "payload": payload})
            completed_payload = {
                "runId": run.run_id,
                "threadId": thread_id,
                "answerLength": len(text),
            }
            self.run_manager.append_event(run.run_id, thread_id, "answer.completed", "ANSWER_STREAM", completed_payload)
            q.put({"event": "answer.completed", "node": "ANSWER_STREAM", "payload": completed_payload})

        def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
            event_payload = dict(payload or {})
            answer = str(event_payload.pop("answer", "")) if event_type == "answer.ready" else ""
            safe_payload = compact_event_payload(event_payload)
            self.run_manager.append_event(run.run_id, thread_id, event_type, node, safe_payload)
            q.put({"event": event_type, "node": node, "payload": safe_payload})
            if answer:
                enqueue_answer(answer)

        def worker() -> None:
            try:
                response = call_run_chat(
                    self.run_chat,
                    request.message,
                    merchant_id,
                    request.context,
                    listener,
                    thread_id,
                    run.run_id,
                    message_history,
                )
                self.run_manager.complete_run(run.run_id, response)
                enqueue_answer(response.answer)
                q.put({"event": "done", "runId": run.run_id, "threadId": thread_id, "response": public_response_payload(response)})
            except Exception as exc:
                self.run_manager.fail_run(run.run_id, str(exc))
                q.put({"event": "error", "runId": run.run_id, "threadId": thread_id, "message": str(exc)})
            finally:
                q.put({"event": "__end__"})

        Thread(target=worker, daemon=True).start()

        def event_iter():
            while True:
                item = q.get()
                if item.get("event") == "__end__":
                    break
                yield "data: %s\n\n" % json.dumps(item, ensure_ascii=False, default=str)

        return event_iter()


class AgentAsyncRunService:
    def __init__(
        self,
        run_manager: AgentRunManager,
        run_chat: Callable[..., ChatResponse],
        default_merchant_id: str,
        max_workers: int = 3,
    ):
        self.run_manager = run_manager
        self.run_chat = run_chat
        self.default_merchant_id = default_merchant_id
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers or 1)), thread_name_prefix="agent-run")
        self.futures: Dict[str, Future] = {}
        self._lock = RLock()

    def submit(self, request: RunCreateRequest) -> AgentRunRecord:
        merchant_id = request.merchant_id or self.default_merchant_id
        thread_id = request.thread_id
        if not thread_id:
            thread = self.run_manager.create_thread(
                merchant_id,
                request.context.topic if request.context else "",
                request.context,
                identity=request.user_identity,
            )
            thread_id = thread.thread_id
        identity = request.user_identity or (getattr(request.context, "user_identity", None) if request.context is not None else None)
        run = self.run_manager.create_run(
            thread_id,
            merchant_id,
            request.message,
            initial_status=AgentRunStatus.QUEUED,
            identity=identity,
        )
        self.run_manager.append_event(
            run.run_id,
            thread_id,
            "run.queued",
            "ASYNC_RUN_SERVICE",
            {"workerPool": "thread", "merchantId": merchant_id},
        )
        run_snapshot = run.model_copy(deep=True)
        message_history = self.run_manager.effective_message_history(
            thread_id,
            merchant_id,
            run.run_id,
            request.message_history,
        )
        future = self.executor.submit(self._worker, run.run_id, thread_id, merchant_id, request.message, request.context, message_history)
        with self._lock:
            self.futures[run.run_id] = future
        return run_snapshot

    def cancel(self, run_id: str) -> Optional[AgentRunRecord]:
        with self._lock:
            future = self.futures.get(run_id)
        future_cancelled = bool(future.cancel()) if future else False
        run = self.run_manager.cancel_run(run_id)
        if run:
            self.run_manager.append_event(
                run_id,
                run.thread_id,
                "run.cancel.requested",
                "ASYNC_RUN_SERVICE",
                {"futureCancelled": future_cancelled},
            )
        return run

    def _worker(
        self,
        run_id: str,
        thread_id: str,
        merchant_id: str,
        message: str,
        context: Optional[ChatContext],
        message_history: Optional[List[Any]] = None,
    ) -> None:
        if self.run_manager.is_canceled(run_id):
            return
        self.run_manager.mark_run_running(run_id)
        self.run_manager.append_event(run_id, thread_id, "run.started", "ASYNC_RUN_SERVICE", {"merchantId": merchant_id})

        def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
            if not self.run_manager.is_canceled(run_id):
                self.run_manager.append_event(run_id, thread_id, event_type, node, payload)

        try:
            response = call_run_chat(self.run_chat, message, merchant_id, context, listener, thread_id, run_id, message_history)
            self.run_manager.complete_run(run_id, response)
        except Exception as exc:
            self.run_manager.fail_run(run_id, str(exc))
        finally:
            with self._lock:
                self.futures.pop(run_id, None)
            if not self.run_manager.is_canceled(run_id):
                self.run_manager.append_event(run_id, thread_id, "run.worker.finished", "ASYNC_RUN_SERVICE", {})


def compact_completed_thread_outputs(workspace_root: Path, thread_id: str, run_id: str = "") -> None:
    outputs = workspace_root / "threads" / thread_id / "runs" / run_id / "outputs"
    if not outputs.exists():
        return
    removable_directories = [outputs / "artifacts" / "context", outputs / "context_packages"]
    for directory in removable_directories:
        try:
            if directory.exists():
                shutil.rmtree(directory)
        except OSError:
            continue
    removable_files = [
        outputs / "trace_replay.json",
        outputs / "artifacts" / "planner" / "planning_asset_pack.json",
        outputs / "artifacts" / "planner" / "candidate_query_graphs.json",
        outputs / "artifacts" / "recall" / "recall_bundle.json",
    ]
    planner_dir = outputs / "artifacts" / "planner"
    if planner_dir.exists():
        removable_files.extend(planner_dir.glob("planner_round_*_prompt.json"))
    for path in removable_files:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            continue


def answer_chunks(answer: str, chunk_size: int = ANSWER_DELTA_CHARS) -> List[str]:
    text = str(answer or "")
    if not text:
        return []
    size = max(1, int(chunk_size or ANSWER_DELTA_CHARS))
    return [text[index : index + size] for index in range(0, len(text), size)]


def compact_event_payload(payload: Dict[str, Any], max_bytes: int = 65536) -> Dict[str, Any]:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return {"summary": str(payload)[:4000], "payloadCompacted": True}
    if len(encoded) <= max_bytes:
        return payload
    important_keys = {
        "stepId",
        "step_id",
        "toolCallId",
        "tool_call_id",
        "parentId",
        "parent_id",
        "status",
        "message",
        "error",
        "durationMs",
        "table",
        "taskId",
        "intentId",
    }
    compacted = {key: value for key, value in payload.items() if key in important_keys}
    compacted.update({"payloadCompacted": True, "originalBytes": len(encoded)})
    return compacted


def public_response_payload(response: ChatResponse) -> Dict[str, Any]:
    payload = response.model_dump(by_alias=True)
    payload.pop("debugTrace", None)
    payload.pop("debug_trace", None)
    return payload


def run_status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "")


def run_duration_ms(run: AgentRunRecord) -> Optional[float]:
    for key in ["totalDurationMs", "total_duration_ms", "durationMs"]:
        value = run.performance_summary.get(key) if run.performance_summary else None
        if isinstance(value, (int, float)):
            return float(value)
    if run.start_time and run.end_time:
        return round((run.end_time - run.start_time).total_seconds() * 1000, 2)
    return None


def run_summary_payload(run: AgentRunRecord, duration_ms: Optional[float] = None) -> Dict[str, Any]:
    answer = run.answer.answer if run.answer else ""
    return {
        "runId": run.run_id,
        "threadId": run.thread_id,
        "merchantId": run.merchant_id,
        "question": run.question,
        "status": run_status_value(run.status),
        "startTime": run.start_time,
        "endTime": run.end_time,
        "durationMs": duration_ms,
        "error": run.error,
        "answerPreview": answer[:160],
        "finalAnswerHash": run.final_answer_hash,
        "tracePath": run.trace_path,
        "checkpointRef": run.checkpoint_ref,
        "artifactRefs": [item.model_dump(by_alias=True) for item in run.artifact_refs[:12]],
        "resumable": run.resumable,
        "performance": run.performance_summary,
    }


def artifact_refs_from_trace(trace: Dict[str, Any]) -> List[ArtifactRef]:
    manifest = trace.get("artifactManifest") or (trace.get("harness") or {}).get("artifactManifest") or []
    refs: List[ArtifactRef] = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        try:
            refs.append(ArtifactRef.model_validate(item))
        except Exception:
            continue
    return refs


def artifact_ref_from_path(path: str, namespace: str = "", reason: str = "") -> ArtifactRef:
    target = Path(path)
    size = target.stat().st_size if target.exists() else 0
    return ArtifactRef(
        artifact_id="artifact_" + uuid.uuid4().hex,
        namespace=namespace,
        path=str(target),
        relative_path=target.name,
        title=target.name,
        reason=reason,
        bytes=size,
        estimated_chars=size,
        merchant_uri=merchant_uri_for_artifact(target.name, namespace=namespace),
        context_layer="L2",
    )
