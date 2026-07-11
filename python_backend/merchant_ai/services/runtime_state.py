from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from merchant_ai.config import Settings


@dataclass
class NodeTaskState:
    run_id: str
    task_id: str
    status: str = "pending"
    idempotency_key: str = ""
    attempts: int = 0
    lease_owner: str = ""
    lease_until: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)


class RuntimeStateStore:
    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        raise NotImplementedError

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        raise NotImplementedError

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        raise NotImplementedError

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        raise NotImplementedError

    def run_canceled(self, run_id: str) -> bool:
        raise NotImplementedError

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        raise NotImplementedError

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        raise NotImplementedError

    def complete_node_task(self, run_id: str, task_id: str, status: str, payload: Dict[str, Any] | None = None) -> NodeTaskState:
        raise NotImplementedError


class FileRuntimeStateStore(RuntimeStateStore):
    """File-backed runtime state. It is intentionally small but external to process memory."""

    def __init__(self, settings: Settings):
        self.root = settings.resolved_workspace_path / "runtime_state"
        self.tasks_dir = self.root / "node_tasks"
        self.cancel_dir = self.root / "cancellations"
        self.queue_dir = self.root / "node_queue"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        state.updated_at = datetime.now().isoformat()
        with self._lock:
            self._write_json(self._task_path(state.run_id, state.task_id), state.__dict__)
        return state

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        data = self._read_json(self._task_path(run_id, task_id))
        return NodeTaskState(**data) if data else None

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        result: List[NodeTaskState] = []
        for path in sorted((self.tasks_dir / safe_name(run_id)).glob("*.json")):
            data = self._read_json(path)
            if data:
                result.append(NodeTaskState(**data))
        return result

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        self._write_json(self.cancel_dir / ("%s.json" % safe_name(run_id)), {"runId": run_id, "reason": reason, "canceledAt": datetime.now().isoformat()})

    def run_canceled(self, run_id: str) -> bool:
        return (self.cancel_dir / ("%s.json" % safe_name(run_id))).exists()

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        existing = self.get_node_task(state.run_id, state.task_id)
        if existing and existing.status in {"running", "completed", "failed", "timeout", "canceled"}:
            return existing
        state.status = state.status or "queued"
        if state.status == "pending":
            state.status = "queued"
        self.upsert_node_task(state)
        self._write_json(self._queue_path(state.run_id, state.task_id), state.__dict__)
        return state

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        with self._lock:
            state = self.get_node_task(run_id, task_id)
            if not state or state.status not in {"queued", "pending", "retry"}:
                return None
            state.status = "running"
            state.attempts += 1
            state.lease_owner = lease_owner
            state.lease_until = (datetime.now() + timedelta(seconds=max(1, lease_seconds))).isoformat()
            self.upsert_node_task(state)
            queue_path = self._queue_path(run_id, task_id)
            if queue_path.exists():
                queue_path.unlink()
            return state

    def complete_node_task(self, run_id: str, task_id: str, status: str, payload: Dict[str, Any] | None = None) -> NodeTaskState:
        state = self.get_node_task(run_id, task_id) or NodeTaskState(run_id=run_id, task_id=task_id)
        state.status = status
        state.payload.update(payload or {})
        state.lease_until = ""
        return self.upsert_node_task(state)

    def _task_path(self, run_id: str, task_id: str) -> Path:
        return self.tasks_dir / safe_name(run_id) / ("%s.json" % safe_name(task_id))

    def _queue_path(self, run_id: str, task_id: str) -> Path:
        return self.queue_dir / safe_name(run_id) / ("%s.json" % safe_name(task_id))

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
        return {}


class RedisRuntimeStateStore(RuntimeStateStore):
    def __init__(self, settings: Settings):
        import redis

        self.settings = settings
        self.namespace = safe_name(settings.redis_namespace)
        timeout = max(0.05, float(settings.redis_socket_timeout_seconds or 1.0))
        self.client = redis.Redis.from_url(settings.redis_url, socket_timeout=timeout, socket_connect_timeout=timeout, decode_responses=True)
        self.client.ping()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        state.updated_at = datetime.now().isoformat()
        self.client.hset(self._task_key(state.run_id, state.task_id), mapping=serialize_state(state))
        self.client.sadd(self._run_tasks_key(state.run_id), state.task_id)
        return state

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        data = self.client.hgetall(self._task_key(run_id, task_id))
        return deserialize_state(data) if data else None

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        states: List[NodeTaskState] = []
        for task_id in sorted(self.client.smembers(self._run_tasks_key(run_id)) or []):
            state = self.get_node_task(run_id, task_id)
            if state:
                states.append(state)
        return states

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        self.client.hset(self._cancel_key(run_id), mapping={"run_id": run_id, "reason": reason, "canceled_at": datetime.now().isoformat()})

    def run_canceled(self, run_id: str) -> bool:
        return bool(self.client.exists(self._cancel_key(run_id)))

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        existing = self.get_node_task(state.run_id, state.task_id)
        if existing and existing.status in {"running", "completed", "failed", "timeout", "canceled"}:
            return existing
        state.status = "queued" if state.status in {"", "pending"} else state.status
        self.upsert_node_task(state)
        self.client.sadd(self._queue_key(state.run_id), state.task_id)
        return state

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        state = self.get_node_task(run_id, task_id)
        if not state or state.status not in {"queued", "pending", "retry"}:
            return None
        state.status = "running"
        state.attempts += 1
        state.lease_owner = lease_owner
        state.lease_until = (datetime.now() + timedelta(seconds=max(1, lease_seconds))).isoformat()
        self.upsert_node_task(state)
        self.client.srem(self._queue_key(run_id), task_id)
        return state

    def complete_node_task(self, run_id: str, task_id: str, status: str, payload: Dict[str, Any] | None = None) -> NodeTaskState:
        state = self.get_node_task(run_id, task_id) or NodeTaskState(run_id=run_id, task_id=task_id)
        state.status = status
        state.payload.update(payload or {})
        state.lease_until = ""
        return self.upsert_node_task(state)

    def _key(self, *parts: str) -> str:
        return ":".join([self.namespace, "runtime_state", *[safe_name(part) for part in parts]])

    def _task_key(self, run_id: str, task_id: str) -> str:
        return self._key("task", run_id, task_id)

    def _run_tasks_key(self, run_id: str) -> str:
        return self._key("run_tasks", run_id)

    def _queue_key(self, run_id: str) -> str:
        return self._key("queue", run_id)

    def _cancel_key(self, run_id: str) -> str:
        return self._key("cancel", run_id)


class PostgresRuntimeStateStore(RuntimeStateStore):
    def __init__(self, settings: Settings):
        import psycopg

        uri = settings.runtime_state_postgres_uri or settings.agent_checkpointer_postgres_uri
        if not uri:
            raise ValueError("YSHOPPING_RUNTIME_STATE_POSTGRES_URI is required for postgres runtime state")
        self.conn = psycopg.connect(uri)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS merchant_ai_node_task_state (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_until TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    PRIMARY KEY (run_id, task_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS merchant_ai_run_cancellation (
                    run_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    canceled_at TEXT NOT NULL
                )
                """
            )
        self.conn.commit()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        state.updated_at = datetime.now().isoformat()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merchant_ai_node_task_state
                (run_id, task_id, status, idempotency_key, attempts, lease_owner, lease_until, updated_at, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, task_id) DO UPDATE SET
                    status=EXCLUDED.status,
                    idempotency_key=EXCLUDED.idempotency_key,
                    attempts=EXCLUDED.attempts,
                    lease_owner=EXCLUDED.lease_owner,
                    lease_until=EXCLUDED.lease_until,
                    updated_at=EXCLUDED.updated_at,
                    payload=EXCLUDED.payload
                """,
                state_to_row(state),
            )
        self.conn.commit()
        return state

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, task_id, status, idempotency_key, attempts, lease_owner, lease_until, updated_at, payload FROM merchant_ai_node_task_state WHERE run_id=%s AND task_id=%s",
                (run_id, task_id),
            )
            row = cur.fetchone()
        return row_to_state(row) if row else None

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, task_id, status, idempotency_key, attempts, lease_owner, lease_until, updated_at, payload FROM merchant_ai_node_task_state WHERE run_id=%s ORDER BY task_id",
                (run_id,),
            )
            return [row_to_state(row) for row in cur.fetchall()]

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merchant_ai_run_cancellation (run_id, reason, canceled_at)
                VALUES (%s,%s,%s)
                ON CONFLICT (run_id) DO UPDATE SET reason=EXCLUDED.reason, canceled_at=EXCLUDED.canceled_at
                """,
                (run_id, reason, datetime.now().isoformat()),
            )
        self.conn.commit()

    def run_canceled(self, run_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM merchant_ai_run_cancellation WHERE run_id=%s", (run_id,))
            return cur.fetchone() is not None

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        existing = self.get_node_task(state.run_id, state.task_id)
        if existing and existing.status in {"running", "completed", "failed", "timeout", "canceled"}:
            return existing
        state.status = "queued" if state.status in {"", "pending"} else state.status
        return self.upsert_node_task(state)

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        state = self.get_node_task(run_id, task_id)
        if not state or state.status not in {"queued", "pending", "retry"}:
            return None
        state.status = "running"
        state.attempts += 1
        state.lease_owner = lease_owner
        state.lease_until = (datetime.now() + timedelta(seconds=max(1, lease_seconds))).isoformat()
        return self.upsert_node_task(state)

    def complete_node_task(self, run_id: str, task_id: str, status: str, payload: Dict[str, Any] | None = None) -> NodeTaskState:
        state = self.get_node_task(run_id, task_id) or NodeTaskState(run_id=run_id, task_id=task_id)
        state.status = status
        state.payload.update(payload or {})
        state.lease_until = ""
        return self.upsert_node_task(state)


def create_runtime_state_store(settings: Settings) -> RuntimeStateStore:
    backend = str(settings.runtime_state_backend or "file").strip().lower()
    if backend == "redis":
        return RedisRuntimeStateStore(settings)
    if backend in {"postgres", "postgresql"}:
        return PostgresRuntimeStateStore(settings)
    return FileRuntimeStateStore(settings)


def node_task_idempotency_key(run_id: str, task_id: str, table: str) -> str:
    return "node:%s:%s:%s" % (safe_name(run_id), safe_name(task_id), safe_name(table))


def safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value or "").strip())
    return text or "unknown"


def serialize_state(state: NodeTaskState) -> Dict[str, str]:
    return {
        "run_id": state.run_id,
        "task_id": state.task_id,
        "status": state.status,
        "idempotency_key": state.idempotency_key,
        "attempts": str(int(state.attempts or 0)),
        "lease_owner": state.lease_owner,
        "lease_until": state.lease_until,
        "updated_at": state.updated_at,
        "payload": json.dumps(state.payload or {}, ensure_ascii=False, default=str),
    }


def deserialize_state(data: Dict[str, Any]) -> NodeTaskState:
    payload = data.get("payload") or "{}"
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return NodeTaskState(
        run_id=str(data.get("run_id") or data.get("runId") or ""),
        task_id=str(data.get("task_id") or data.get("taskId") or ""),
        status=str(data.get("status") or "pending"),
        idempotency_key=str(data.get("idempotency_key") or data.get("idempotencyKey") or ""),
        attempts=int(data.get("attempts") or 0),
        lease_owner=str(data.get("lease_owner") or data.get("leaseOwner") or ""),
        lease_until=str(data.get("lease_until") or data.get("leaseUntil") or ""),
        updated_at=str(data.get("updated_at") or data.get("updatedAt") or datetime.now().isoformat()),
        payload=payload if isinstance(payload, dict) else {},
    )


def state_to_row(state: NodeTaskState) -> tuple:
    return (
        state.run_id,
        state.task_id,
        state.status,
        state.idempotency_key,
        int(state.attempts or 0),
        state.lease_owner,
        state.lease_until,
        state.updated_at,
        json.dumps(state.payload or {}, ensure_ascii=False, default=str),
    )


def row_to_state(row: Any) -> NodeTaskState:
    run_id, task_id, status, idempotency_key, attempts, lease_owner, lease_until, updated_at, payload = row
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return NodeTaskState(
        run_id=run_id,
        task_id=task_id,
        status=status,
        idempotency_key=idempotency_key,
        attempts=int(attempts or 0),
        lease_owner=lease_owner,
        lease_until=lease_until,
        updated_at=updated_at,
        payload=payload if isinstance(payload, dict) else {},
    )
