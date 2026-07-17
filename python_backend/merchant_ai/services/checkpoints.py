from __future__ import annotations

from contextlib import AbstractContextManager
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional

from merchant_ai.config import Settings
from merchant_ai.services.langgraph_compat import MemorySaver, postgres_saver_class, sqlite_saver_class


DEEP_AGENT_CHECKPOINT_NAMESPACE = "deepagent"


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
        SqliteSaver = sqlite_saver_class()
        path = self.settings.resolved_checkpointer_sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._context = SqliteSaver.from_conn_string(str(path))
        saver = self._context.__enter__()
        if hasattr(saver, "setup"):
            saver.setup()
        return saver

    def _postgres_saver(self) -> Any:
        PostgresSaver = postgres_saver_class()
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

    def config_for_deep_agent(self, thread_id: str, run_id: str = "") -> Dict[str, Any]:
        """Address the outer Deep Agent by its durable conversation identity.

        Domain graph checkpoints remain isolated per run under ``thread:run``
        with the legacy empty namespace. The outer agent instead reuses the
        real API conversation thread across runs. A root LangGraph must use the
        empty physical checkpoint namespace: passing ``checkpoint_ns=deepagent``
        makes ``get_state`` interpret it as a subgraph path and fail with
        ``Subgraph deepagent not found``. Logical ownership is retained in
        metadata while collision safety comes from the conversation key versus
        the legacy domain graph's independently run-scoped key.
        """

        checkpoint_thread_id = thread_id or "thread"
        return {
            "configurable": {
                "thread_id": checkpoint_thread_id,
            },
            "metadata": {
                "thread_id": thread_id,
                "run_id": run_id,
                "checkpoint_thread_id": checkpoint_thread_id,
                "checkpoint_namespace": "",
                "logical_checkpoint_namespace": DEEP_AGENT_CHECKPOINT_NAMESPACE,
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
            "resumable": False,
            "purpose": "diagnostic_checkpoint_only",
        }

    def deep_agent_ref(self, thread_id: str, run_id: str = "") -> Dict[str, Any]:
        """Describe the durable outer-agent checkpoint used for a conversation."""

        checkpoint_thread_id = thread_id or "thread"
        return {
            "backend": self.backend or "sqlite",
            "threadId": thread_id,
            "runId": run_id,
            "checkpointThreadId": checkpoint_thread_id,
            "checkpointNamespace": "",
            "logicalCheckpointNamespace": DEEP_AGENT_CHECKPOINT_NAMESPACE,
            "storage": self.storage_ref(),
            "resumable": (self.backend or "sqlite") != "memory",
            "purpose": "deep_agent_conversation_checkpoint",
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
        "resumable": False,
        "purpose": "diagnostic_checkpoint_only",
    }


def prune_completed_sqlite_checkpoints(settings: Settings) -> int:
    """Bound local checkpoint growth without breaking resumable conversations.

    The domain graph is addressed by ``thread_id:run_id`` while the outer Deep
    Agent is addressed by the durable conversation ``thread_id``. Retention
    therefore has to keep both keys for every active run and every completed run
    inside the configured retention window.
    """
    backend = (settings.agent_checkpointer_backend or "sqlite").strip().lower()
    path = settings.resolved_checkpointer_sqlite_path
    if backend not in {"", "sqlite"} or not path.exists():
        return 0
    runs_dir = settings.resolved_workspace_path / "run_events" / "runs"
    completed: list[tuple[str, str, str]] = []
    retained_thread_ids: set[str] = set()
    for run_path in runs_dir.glob("run_*.json"):
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        thread_id = str(payload.get("threadId") or payload.get("thread_id") or "")
        run_id = str(payload.get("runId") or payload.get("run_id") or "")
        if not thread_id or not run_id:
            continue
        checkpoint_thread_id = "%s:%s" % (thread_id, run_id)
        status = str(payload.get("status") or "")
        updated_at = str(payload.get("updatedAt") or payload.get("updated_at") or "")
        if status == "COMPLETED":
            completed.append((updated_at, checkpoint_thread_id, thread_id))
        else:
            retained_thread_ids.add(checkpoint_thread_id)
            retained_thread_ids.add(thread_id)
    limit = max(0, int(settings.agent_completed_checkpoint_limit or 0))
    for _, checkpoint_thread_id, conversation_thread_id in sorted(completed, reverse=True)[:limit]:
        retained_thread_ids.add(checkpoint_thread_id)
        retained_thread_ids.add(conversation_thread_id)
    connection = sqlite3.connect(str(path), timeout=5)
    try:
        try:
            rows = connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
            removable = [str(row[0]) for row in rows if str(row[0]) not in retained_thread_ids]
            for offset in range(0, len(removable), 200):
                batch = removable[offset : offset + 200]
                placeholders = ",".join("?" for _ in batch)
                connection.execute("DELETE FROM writes WHERE thread_id IN (%s)" % placeholders, batch)
                connection.execute("DELETE FROM checkpoints WHERE thread_id IN (%s)" % placeholders, batch)
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return len(removable)
        except sqlite3.Error:
            connection.rollback()
            return 0
    finally:
        connection.close()
