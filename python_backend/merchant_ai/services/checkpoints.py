from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver

from merchant_ai.config import Settings


class CheckpointManager:
    """Owns the LangGraph checkpointer lifecycle and run-level checkpoint refs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = (settings.agent_checkpointer_backend or "sqlite").strip().lower()
        self._context: Optional[AbstractContextManager[Any]] = None
        self._saver: Any = None
        self._path = ""

    def saver(self) -> Any:
        if self._saver is not None:
            return self._saver
        if self.backend in {"", "sqlite"}:
            self._saver = self._sqlite_saver()
            return self._saver
        if self.backend == "postgres":
            self._saver = self._postgres_saver()
            return self._saver
        if self.backend == "memory":
            self._saver = MemorySaver()
            return self._saver
        raise ValueError("Unsupported checkpointer backend: %s" % self.backend)

    def _sqlite_saver(self) -> Any:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = self.settings.resolved_checkpointer_sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._context = SqliteSaver.from_conn_string(str(path))
        saver = self._context.__enter__()
        if hasattr(saver, "setup"):
            saver.setup()
        return saver

    def _postgres_saver(self) -> Any:
        from langgraph.checkpoint.postgres import PostgresSaver

        if not self.settings.agent_checkpointer_postgres_uri:
            raise ValueError("YSHOPPING_AGENT_CHECKPOINTER_POSTGRES_URI is required for postgres checkpointer")
        self._path = self.settings.agent_checkpointer_postgres_uri
        self._context = PostgresSaver.from_conn_string(self.settings.agent_checkpointer_postgres_uri)
        saver = self._context.__enter__()
        if hasattr(saver, "setup"):
            saver.setup()
        return saver

    def thread_id_for_run(self, thread_id: str, run_id: str) -> str:
        return "%s:%s" % (thread_id or "thread", run_id or "run")

    def config_for_run(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        checkpoint_thread_id = self.thread_id_for_run(thread_id, run_id)
        return {
            "configurable": {
                "thread_id": checkpoint_thread_id,
            },
            "metadata": {
                "thread_id": thread_id,
                "run_id": run_id,
                "checkpoint_thread_id": checkpoint_thread_id,
            },
            "recursion_limit": 80,
        }

    def run_ref(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        checkpoint_thread_id = self.thread_id_for_run(thread_id, run_id)
        return {
            "backend": self.backend or "sqlite",
            "threadId": thread_id,
            "runId": run_id,
            "checkpointThreadId": checkpoint_thread_id,
            "checkpointNamespace": "",
            "storage": self.storage_ref(),
            "resumable": (self.backend or "sqlite") != "memory",
        }

    def storage_ref(self) -> str:
        if self.backend == "sqlite" or not self.backend:
            return self._path or str(self.settings.resolved_checkpointer_sqlite_path)
        if self.backend == "postgres":
            return "postgres"
        return self.backend

    def debug(self) -> Dict[str, Any]:
        return {
            "backend": self.backend or "sqlite",
            "storage": self.storage_ref(),
            "persistent": (self.backend or "sqlite") != "memory",
        }

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None
            self._saver = None


def checkpoint_ref_for_run(settings: Settings, thread_id: str, run_id: str) -> Dict[str, Any]:
    backend = (settings.agent_checkpointer_backend or "sqlite").strip().lower()
    storage = ""
    if backend in {"", "sqlite"}:
        storage = str(settings.resolved_checkpointer_sqlite_path)
        Path(storage).parent.mkdir(parents=True, exist_ok=True)
    elif backend == "postgres":
        storage = "postgres"
    else:
        storage = backend or "memory"
    checkpoint_thread_id = "%s:%s" % (thread_id or "thread", run_id or "run")
    return {
        "backend": backend or "sqlite",
        "threadId": thread_id,
        "runId": run_id,
        "checkpointThreadId": checkpoint_thread_id,
        "checkpointNamespace": "",
        "storage": storage,
        "resumable": backend != "memory",
    }
