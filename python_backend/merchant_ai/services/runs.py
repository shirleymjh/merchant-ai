from __future__ import annotations

import json
import uuid
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

from merchant_ai.models import (
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStatus,
    AgentThreadRecord,
    ChatContext,
    ChatResponse,
    RunCreateRequest,
)


class AgentRunManager:
    def __init__(self):
        self.threads: Dict[str, AgentThreadRecord] = {}
        self.runs: Dict[str, AgentRunRecord] = {}
        self.run_events: Dict[str, List[AgentRunEventRecord]] = {}

    def create_thread(self, merchant_id: str, topic: str = "", context: Optional[ChatContext] = None) -> AgentThreadRecord:
        thread_id = "thread_" + uuid.uuid4().hex
        record = AgentThreadRecord(thread_id=thread_id, merchant_id=merchant_id, topic=topic or "", context=context)
        self.threads[thread_id] = record
        return record

    def get_thread(self, thread_id: str) -> Optional[AgentThreadRecord]:
        return self.threads.get(thread_id)

    def create_run(self, thread_id: str, merchant_id: str, question: str) -> AgentRunRecord:
        run_id = "run_" + uuid.uuid4().hex
        run = AgentRunRecord(run_id=run_id, thread_id=thread_id, merchant_id=merchant_id, question=question, status=AgentRunStatus.RUNNING)
        self.runs[run_id] = run
        self.run_events[run_id] = []
        return run

    def get_run(self, run_id: str) -> Optional[AgentRunRecord]:
        return self.runs.get(run_id)

    def append_event(self, run_id: str, thread_id: str, event_type: str, node: str = "", payload: Optional[Dict[str, Any]] = None) -> None:
        event = AgentRunEventRecord(
            event_id="event_" + uuid.uuid4().hex,
            run_id=run_id,
            thread_id=thread_id,
            event_type=event_type,
            node=node,
            payload=payload or {},
        )
        self.run_events.setdefault(run_id, []).append(event)

    def events(self, run_id: str) -> List[AgentRunEventRecord]:
        return self.run_events.get(run_id, [])

    def complete_run(self, run_id: str, response: ChatResponse) -> None:
        run = self.runs.get(run_id)
        if run:
            run.status = AgentRunStatus.COMPLETED
            run.answer = response
            run.updated_at = datetime.now()

    def fail_run(self, run_id: str, error: str) -> None:
        run = self.runs.get(run_id)
        if run:
            run.status = AgentRunStatus.FAILED
            run.error = error
            run.updated_at = datetime.now()

    def cancel_run(self, run_id: str) -> Optional[AgentRunRecord]:
        run = self.runs.get(run_id)
        if run:
            run.status = AgentRunStatus.CANCELED
            run.updated_at = datetime.now()
        return run


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
