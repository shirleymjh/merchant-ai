from __future__ import annotations

import json
import hashlib
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
    RunCreateRequest,
)
from merchant_ai.services.checkpoints import checkpoint_ref_for_run


ANSWER_DELTA_CHARS = 80


class FileRunEventStore:
    def __init__(self, settings: Settings):
        self.root = settings.resolved_workspace_path / "run_events"
        self.threads_dir = self.root / "threads"
        self.runs_dir = self.root / "runs"
        self.events_dir = self.root / "events"
        self.traces_dir = self.root / "traces"
        for path in [self.threads_dir, self.runs_dir, self.events_dir, self.traces_dir]:
            path.mkdir(parents=True, exist_ok=True)

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
        self._lock = RLock()

    def create_thread(self, merchant_id: str, topic: str = "", context: Optional[ChatContext] = None) -> AgentThreadRecord:
        thread_id = "thread_" + uuid.uuid4().hex
        record = AgentThreadRecord(thread_id=thread_id, merchant_id=merchant_id, topic=topic or "", context=context)
        self.threads[thread_id] = record
        self.store.save_thread(record)
        return record

    def get_thread(self, thread_id: str) -> Optional[AgentThreadRecord]:
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
    ) -> AgentRunRecord:
        run_id = "run_" + uuid.uuid4().hex
        run = AgentRunRecord(
            run_id=run_id,
            thread_id=thread_id,
            merchant_id=merchant_id,
            question=question,
            status=initial_status,
            start_time=datetime.now(),
            checkpoint_ref=checkpoint_ref_for_run(self.settings, thread_id, run_id),
            resumable=(self.settings.agent_checkpointer_backend or "sqlite").strip().lower() != "memory",
        )
        with self._lock:
            self.runs[run_id] = run
            self.run_events[run_id] = []
            self.store.save_run(run)
        self.append_event(run_id, thread_id, "run.started", "RUN_MANAGER", {"question": question, "merchantId": merchant_id})
        return run

    def get_run(self, run_id: str) -> Optional[AgentRunRecord]:
        run = self.runs.get(run_id) or self.store.load_run(run_id)
        if run:
            self.runs[run_id] = run
        return run

    def append_event(self, run_id: str, thread_id: str, event_type: str, node: str = "", payload: Optional[Dict[str, Any]] = None) -> None:
        safe_payload = payload or {}
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
            run.performance_summary = harness.get("performance") or {}
            if isinstance(harness.get("checkpoint"), dict) and harness.get("checkpoint"):
                run.checkpoint_ref = harness["checkpoint"]
                run.resumable = bool(harness["checkpoint"].get("resumable"))
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
                self.store.save_run(run)
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
            thread = self.run_manager.create_thread(merchant_id, request.context.topic if request.context else "", request.context)
            thread_id = thread.thread_id
        run = self.run_manager.create_run(thread_id, merchant_id, request.message)

        def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
            self.run_manager.append_event(run.run_id, thread_id, event_type, node, payload)
            q.put({"event": event_type, "node": node, "payload": payload})

        def worker() -> None:
            try:
                response = self.run_chat(request.message, merchant_id, request.context, listener, thread_id, run.run_id)
                self.run_manager.complete_run(run.run_id, response)
                for index, chunk in enumerate(answer_chunks(response.answer, ANSWER_DELTA_CHARS)):
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
                    "answerLength": len(response.answer or ""),
                }
                self.run_manager.append_event(run.run_id, thread_id, "answer.completed", "ANSWER_STREAM", completed_payload)
                q.put({"event": "answer.completed", "node": "ANSWER_STREAM", "payload": completed_payload})
                q.put({"event": "done", "runId": run.run_id, "threadId": thread_id, "response": response.model_dump(by_alias=True)})
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
            thread = self.run_manager.create_thread(merchant_id, request.context.topic if request.context else "", request.context)
            thread_id = thread.thread_id
        run = self.run_manager.create_run(thread_id, merchant_id, request.message, initial_status=AgentRunStatus.QUEUED)
        self.run_manager.append_event(
            run.run_id,
            thread_id,
            "run.queued",
            "ASYNC_RUN_SERVICE",
            {"workerPool": "thread", "merchantId": merchant_id},
        )
        run_snapshot = run.model_copy(deep=True)
        future = self.executor.submit(self._worker, run.run_id, thread_id, merchant_id, request.message, request.context)
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

    def _worker(self, run_id: str, thread_id: str, merchant_id: str, message: str, context: Optional[ChatContext]) -> None:
        if self.run_manager.is_canceled(run_id):
            return
        self.run_manager.mark_run_running(run_id)

        def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
            if not self.run_manager.is_canceled(run_id):
                self.run_manager.append_event(run_id, thread_id, event_type, node, payload)

        try:
            response = self.run_chat(message, merchant_id, context, listener, thread_id, run_id)
            self.run_manager.complete_run(run_id, response)
        except Exception as exc:
            self.run_manager.fail_run(run_id, str(exc))
        finally:
            with self._lock:
                self.futures.pop(run_id, None)
            if not self.run_manager.is_canceled(run_id):
                self.run_manager.append_event(run_id, thread_id, "run.worker.finished", "ASYNC_RUN_SERVICE", {})


def answer_chunks(answer: str, chunk_size: int = ANSWER_DELTA_CHARS) -> List[str]:
    text = str(answer or "")
    if not text:
        return []
    size = max(1, int(chunk_size or ANSWER_DELTA_CHARS))
    return [text[index : index + size] for index in range(0, len(text), size)]


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
    )
