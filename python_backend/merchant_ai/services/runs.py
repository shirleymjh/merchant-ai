from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import (
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStatus,
    AgentThreadRecord,
    ChatContext,
    ChatResponse,
    RunCreateRequest,
)


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

    def create_run(self, thread_id: str, merchant_id: str, question: str) -> AgentRunRecord:
        run_id = "run_" + uuid.uuid4().hex
        run = AgentRunRecord(
            run_id=run_id,
            thread_id=thread_id,
            merchant_id=merchant_id,
            question=question,
            status=AgentRunStatus.RUNNING,
            start_time=datetime.now(),
        )
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
            run.status = AgentRunStatus.COMPLETED
            run.answer = response
            run.end_time = datetime.now()
            run.final_answer_hash = hashlib.sha256((response.answer or "").encode("utf-8")).hexdigest()[:16]
            trace = response.debug_trace or {}
            harness = trace.get("harness") or {}
            run.performance_summary = harness.get("performance") or {}
            run.trace_path = str((harness.get("traceReplay") or {}).get("path") or harness.get("traceReplayPath") or "")
            run.updated_at = datetime.now()
            self.store.save_run(run)
            replay_payload = self._load_replay_payload(run.trace_path) or trace
            if replay_payload:
                trace_path = self.store.save_trace(run_id, replay_payload)
                run.trace_path = str(trace_path)
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
            run.status = AgentRunStatus.FAILED
            run.error = error
            run.end_time = datetime.now()
            run.updated_at = datetime.now()
            self.store.save_run(run)
            self.append_event(run_id, run.thread_id, "run.failed", "RUN_MANAGER", {"error": error})

    def cancel_run(self, run_id: str) -> Optional[AgentRunRecord]:
        run = self.runs.get(run_id)
        if run:
            run.status = AgentRunStatus.CANCELED
            run.end_time = datetime.now()
            run.updated_at = datetime.now()
            self.store.save_run(run)
            self.append_event(run_id, run.thread_id, "run.canceled", "RUN_MANAGER", {})
        return run

    def trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.store.load_trace(run_id)

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


def answer_chunks(answer: str, chunk_size: int = ANSWER_DELTA_CHARS) -> List[str]:
    text = str(answer or "")
    if not text:
        return []
    size = max(1, int(chunk_size or ANSWER_DELTA_CHARS))
    return [text[index : index + size] for index in range(0, len(text), size)]
