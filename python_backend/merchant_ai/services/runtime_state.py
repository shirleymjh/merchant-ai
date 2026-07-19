from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Optional

from merchant_ai.config import Settings


TERMINAL_EXECUTION_STATUSES = frozenset({"succeeded", "failed", "timed_out", "canceled"})


class StaleNodeTaskFence(RuntimeError):
    """Raised when a worker tries to mutate a task after losing its lease."""


class StaleExecutionFence(RuntimeError):
    """Raised when an execution journal write uses an obsolete fencing token."""


class NodeTaskRequestConflict(RuntimeError):
    """Raised when a task key is reused for a different immutable request."""

    code = "NODE_TASK_REQUEST_CONFLICT"

    def __init__(
        self,
        run_id: str,
        task_id: str,
        existing_fingerprint: str,
        requested_fingerprint: str,
    ):
        self.run_id = str(run_id or "")
        self.task_id = str(task_id or "")
        self.existing_fingerprint = str(existing_fingerprint or "")
        self.requested_fingerprint = str(requested_fingerprint or "")
        super().__init__("node task key is already bound to a different request fingerprint")


@dataclass
class ExecutionJournalEntry:
    idempotency_key: str
    status: str = "in_progress"
    lease_owner: str = ""
    lease_generation: int = 0
    lease_until: str = ""
    result_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionJournalClaim:
    outcome: str
    entry: ExecutionJournalEntry

    @property
    def acquired(self) -> bool:
        return self.outcome == "acquired"


@dataclass
class NodeTaskState:
    run_id: str
    task_id: str
    status: str = "pending"
    idempotency_key: str = ""
    attempts: int = 0
    lease_owner: str = ""
    lease_generation: int = 0
    lease_until: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    request_fingerprint: str = ""


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

    def complete_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        lease_owner: str = "",
        lease_generation: int = 0,
    ) -> NodeTaskState:
        raise NotImplementedError

    def heartbeat_node_task(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int = 300,
        lease_generation: int = 0,
    ) -> bool:
        raise NotImplementedError

    def fence_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        expected_generation: int = 0,
    ) -> NodeTaskState:
        raise NotImplementedError

    def recover_expired_node_tasks(self, max_attempts: int = 3) -> int:
        raise NotImplementedError

    def claim_next_node_task(
        self,
        lease_owner: str,
        lease_seconds: int = 300,
        task_kinds: Optional[List[str]] = None,
    ) -> Optional[NodeTaskState]:
        raise NotImplementedError

    def claim_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_seconds: int = 300,
        *,
        allow_expired_takeover: bool = False,
    ) -> ExecutionJournalClaim:
        raise NotImplementedError

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        raise NotImplementedError

    def complete_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_generation: int,
        status: str,
        payload: Dict[str, Any] | None = None,
        result_hash: str = "",
    ) -> ExecutionJournalEntry:
        raise NotImplementedError


class MemoryRuntimeStateStore(RuntimeStateStore):
    """Process-local atomic runtime state used by tests and single-process workers."""

    def __init__(self):
        self._tasks: Dict[tuple[str, str], NodeTaskState] = {}
        self._cancellations: Dict[str, str] = {}
        self._executions: Dict[str, ExecutionJournalEntry] = {}
        self._lock = RLock()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        with self._lock:
            bind_node_task_request_fingerprint(state)
            state.updated_at = datetime.now().isoformat()
            self._tasks[(state.run_id, state.task_id)] = clone_node_task(state)
            return clone_node_task(state)

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            return clone_node_task(state) if state else None

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        with self._lock:
            return [
                clone_node_task(state)
                for key, state in sorted(self._tasks.items())
                if key[0] == run_id
            ]

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        with self._lock:
            self._cancellations[run_id] = reason

    def run_canceled(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancellations

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        candidate = node_task_enqueue_candidate(state)
        with self._lock:
            existing = self._tasks.get((candidate.run_id, candidate.task_id))
            if existing:
                assert_node_task_request_match(existing, candidate)
                return clone_node_task(existing)
            candidate.updated_at = datetime.now().isoformat()
            self._tasks[(candidate.run_id, candidate.task_id)] = clone_node_task(candidate)
            state.request_fingerprint = candidate.request_fingerprint
            state.status = candidate.status
            state.updated_at = candidate.updated_at
            return clone_node_task(candidate)

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            if not state or state.status not in {"queued", "pending", "retry"}:
                return None
            state.status = "running"
            state.attempts += 1
            state.lease_owner = required_text(lease_owner, "lease_owner")
            state.lease_generation += 1
            state.lease_until = future_timestamp(lease_seconds)
            state.updated_at = datetime.now().isoformat()
            return clone_node_task(state)

    def complete_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        lease_owner: str = "",
        lease_generation: int = 0,
    ) -> NodeTaskState:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            if not state:
                raise StaleNodeTaskFence("node task does not exist")
            assert_node_task_fence(state, lease_owner, lease_generation, status)
            if state.status != "running":
                return clone_node_task(state)
            state.status = status
            state.payload.update(payload or {})
            state.lease_until = ""
            state.updated_at = datetime.now().isoformat()
            return clone_node_task(state)

    def heartbeat_node_task(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int = 300,
        lease_generation: int = 0,
    ) -> bool:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            if not node_task_fence_matches(state, lease_owner, lease_generation):
                return False
            state.lease_until = future_timestamp(lease_seconds)
            state.updated_at = datetime.now().isoformat()
            return True

    def fence_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        expected_generation: int = 0,
    ) -> NodeTaskState:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            if not state:
                raise StaleNodeTaskFence("node task does not exist")
            assert_expected_generation(state, expected_generation)
            if state.status in {"completed", "partial", "failed", "timeout", "canceled"}:
                return clone_node_task(state)
            state.status = status
            state.payload.update(payload or {})
            state.lease_generation += 1
            state.lease_owner = ""
            state.lease_until = ""
            state.updated_at = datetime.now().isoformat()
            return clone_node_task(state)

    def recover_expired_node_tasks(self, max_attempts: int = 3) -> int:
        recovered = 0
        with self._lock:
            for state in self._tasks.values():
                if state.status != "running" or not lease_expired(state.lease_until):
                    continue
                state.status = "retry" if state.attempts < max(1, max_attempts) else "failed"
                state.lease_owner = ""
                state.lease_until = ""
                state.payload["recoveredAfterLeaseExpiry"] = True
                state.updated_at = datetime.now().isoformat()
                recovered += 1
        return recovered

    def claim_next_node_task(
        self,
        lease_owner: str,
        lease_seconds: int = 300,
        task_kinds: Optional[List[str]] = None,
    ) -> Optional[NodeTaskState]:
        allowed = {str(item) for item in (task_kinds or []) if str(item)}
        with self._lock:
            candidates = sorted(self._tasks.values(), key=lambda item: item.updated_at)
            for state in candidates:
                if state.status not in {"queued", "pending", "retry"}:
                    continue
                if allowed and str(state.payload.get("taskKind") or "") not in allowed:
                    continue
                return self.claim_node_task(state.run_id, state.task_id, lease_owner, lease_seconds)
        return None

    def claim_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_seconds: int = 300,
        *,
        allow_expired_takeover: bool = False,
    ) -> ExecutionJournalClaim:
        with self._lock:
            key = required_text(idempotency_key, "idempotency_key")
            owner = required_text(lease_owner, "lease_owner")
            current = self._executions.get(key)
            claim = next_execution_claim(current, key, owner, lease_seconds, allow_expired_takeover)
            if claim.acquired:
                self._executions[key] = clone_execution_entry(claim.entry)
            return ExecutionJournalClaim(claim.outcome, clone_execution_entry(claim.entry))

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        with self._lock:
            entry = self._executions.get(idempotency_key)
            return clone_execution_entry(entry) if entry else None

    def complete_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_generation: int,
        status: str,
        payload: Dict[str, Any] | None = None,
        result_hash: str = "",
    ) -> ExecutionJournalEntry:
        with self._lock:
            entry = self._executions.get(idempotency_key)
            completed = completed_execution_entry(
                entry,
                idempotency_key,
                lease_owner,
                lease_generation,
                status,
                payload,
                result_hash,
            )
            self._executions[idempotency_key] = clone_execution_entry(completed)
            return clone_execution_entry(completed)


class FileRuntimeStateStore(RuntimeStateStore):
    """File-backed runtime state with process-safe compare-and-set operations."""

    def __init__(self, settings: Settings):
        self.root = settings.resolved_workspace_path / "runtime_state"
        self.tasks_dir = self.root / "node_tasks"
        self.cancel_dir = self.root / "cancellations"
        self.queue_dir = self.root / "node_queue"
        self.executions_dir = self.root / "execution_journal"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.cancel_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.executions_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".state.lock"
        self._lock = RLock()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        with self._state_guard():
            return self._upsert_node_task_unlocked(state)

    def get_node_task(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        with self._state_guard(shared=True):
            return self._get_node_task_unlocked(run_id, task_id)

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        with self._state_guard(shared=True):
            result: List[NodeTaskState] = []
            for path in sorted((self.tasks_dir / safe_name(run_id)).glob("*.json")):
                data = self._read_json(path)
                if data:
                    result.append(NodeTaskState(**data))
            return result

    def cancel_run(self, run_id: str, reason: str = "") -> None:
        with self._state_guard():
            self._write_json(
                self.cancel_dir / ("%s.json" % safe_name(run_id)),
                {"runId": run_id, "reason": reason, "canceledAt": datetime.now().isoformat()},
            )

    def run_canceled(self, run_id: str) -> bool:
        with self._state_guard(shared=True):
            return (self.cancel_dir / ("%s.json" % safe_name(run_id))).exists()

    def enqueue_node_task(self, state: NodeTaskState) -> NodeTaskState:
        candidate = node_task_enqueue_candidate(state)
        with self._state_guard():
            existing = self._get_node_task_unlocked(candidate.run_id, candidate.task_id)
            if existing:
                assert_node_task_request_match(existing, candidate)
                return existing
            persisted = self._upsert_node_task_unlocked(candidate)
            self._write_json(self._queue_path(candidate.run_id, candidate.task_id), persisted.__dict__)
            state.request_fingerprint = persisted.request_fingerprint
            state.status = persisted.status
            state.updated_at = persisted.updated_at
            return persisted

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        with self._state_guard():
            return self._claim_node_task_unlocked(run_id, task_id, lease_owner, lease_seconds)

    def complete_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        lease_owner: str = "",
        lease_generation: int = 0,
    ) -> NodeTaskState:
        with self._state_guard():
            state = self._get_node_task_unlocked(run_id, task_id)
            if not state:
                raise StaleNodeTaskFence("node task does not exist")
            assert_node_task_fence(state, lease_owner, lease_generation, status)
            if state.status != "running":
                return state
            state.status = status
            state.payload.update(payload or {})
            state.lease_until = ""
            return self._upsert_node_task_unlocked(state)

    def heartbeat_node_task(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int = 300,
        lease_generation: int = 0,
    ) -> bool:
        with self._state_guard():
            state = self._get_node_task_unlocked(run_id, task_id)
            if not node_task_fence_matches(state, lease_owner, lease_generation):
                return False
            state.lease_until = future_timestamp(lease_seconds)
            self._upsert_node_task_unlocked(state)
            return True

    def fence_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        expected_generation: int = 0,
    ) -> NodeTaskState:
        with self._state_guard():
            state = self._get_node_task_unlocked(run_id, task_id)
            if not state:
                raise StaleNodeTaskFence("node task does not exist")
            assert_expected_generation(state, expected_generation)
            if state.status in {"completed", "partial", "failed", "timeout", "canceled"}:
                return state
            state.status = status
            state.payload.update(payload or {})
            state.lease_generation += 1
            state.lease_owner = ""
            state.lease_until = ""
            return self._upsert_node_task_unlocked(state)

    def recover_expired_node_tasks(self, max_attempts: int = 3) -> int:
        recovered = 0
        now = datetime.now()
        with self._state_guard():
            for path in self.tasks_dir.glob("*/*.json"):
                data = self._read_json(path)
                state = NodeTaskState(**data) if data else None
                if not state or state.status != "running" or not lease_expired(state.lease_until, now):
                    continue
                state.status = "retry" if state.attempts < max(1, max_attempts) else "failed"
                state.lease_owner = ""
                state.lease_until = ""
                state.payload["recoveredAfterLeaseExpiry"] = True
                persisted = self._upsert_node_task_unlocked(state)
                if state.status == "retry":
                    self._write_json(self._queue_path(state.run_id, state.task_id), persisted.__dict__)
                recovered += 1
        return recovered

    def claim_next_node_task(
        self,
        lease_owner: str,
        lease_seconds: int = 300,
        task_kinds: Optional[List[str]] = None,
    ) -> Optional[NodeTaskState]:
        allowed = {str(item) for item in (task_kinds or []) if str(item)}
        with self._state_guard():
            for path in sorted(self.queue_dir.glob("*/*.json"), key=lambda item: item.stat().st_mtime):
                data = self._read_json(path)
                if not data:
                    continue
                state = NodeTaskState(**data)
                if allowed and str(state.payload.get("taskKind") or "") not in allowed:
                    continue
                claimed = self._claim_node_task_unlocked(state.run_id, state.task_id, lease_owner, lease_seconds)
                if claimed:
                    return claimed
        return None

    def claim_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_seconds: int = 300,
        *,
        allow_expired_takeover: bool = False,
    ) -> ExecutionJournalClaim:
        with self._state_guard():
            key = required_text(idempotency_key, "idempotency_key")
            owner = required_text(lease_owner, "lease_owner")
            path = self._execution_path(key)
            current = self._get_execution_unlocked(key)
            claim = next_execution_claim(current, key, owner, lease_seconds, allow_expired_takeover)
            if claim.acquired:
                self._write_json(path, claim.entry.__dict__)
            return ExecutionJournalClaim(claim.outcome, clone_execution_entry(claim.entry))

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        with self._state_guard(shared=True):
            return self._get_execution_unlocked(idempotency_key)

    def complete_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_generation: int,
        status: str,
        payload: Dict[str, Any] | None = None,
        result_hash: str = "",
    ) -> ExecutionJournalEntry:
        with self._state_guard():
            entry = self._get_execution_unlocked(idempotency_key)
            completed = completed_execution_entry(
                entry,
                idempotency_key,
                lease_owner,
                lease_generation,
                status,
                payload,
                result_hash,
            )
            self._write_json(self._execution_path(idempotency_key), completed.__dict__)
            return clone_execution_entry(completed)

    def _upsert_node_task_unlocked(self, state: NodeTaskState) -> NodeTaskState:
        bind_node_task_request_fingerprint(state)
        persisted = clone_node_task(state)
        persisted.updated_at = datetime.now().isoformat()
        self._write_json(self._task_path(persisted.run_id, persisted.task_id), persisted.__dict__)
        state.updated_at = persisted.updated_at
        return persisted

    def _get_node_task_unlocked(self, run_id: str, task_id: str) -> Optional[NodeTaskState]:
        data = self._read_json(self._task_path(run_id, task_id))
        return NodeTaskState(**data) if data else None

    def _claim_node_task_unlocked(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> Optional[NodeTaskState]:
        state = self._get_node_task_unlocked(run_id, task_id)
        if not state or state.status not in {"queued", "pending", "retry"}:
            return None
        state.status = "running"
        state.attempts += 1
        state.lease_owner = required_text(lease_owner, "lease_owner")
        state.lease_generation += 1
        state.lease_until = future_timestamp(lease_seconds)
        persisted = self._upsert_node_task_unlocked(state)
        queue_path = self._queue_path(run_id, task_id)
        if queue_path.exists():
            queue_path.unlink()
        return persisted

    def _get_execution_unlocked(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        data = self._read_json(self._execution_path(idempotency_key))
        return ExecutionJournalEntry(**data) if data else None

    def _task_path(self, run_id: str, task_id: str) -> Path:
        return self.tasks_dir / safe_name(run_id) / ("%s.json" % safe_name(task_id))

    def _queue_path(self, run_id: str, task_id: str) -> Path:
        return self.queue_dir / safe_name(run_id) / ("%s.json" % safe_name(task_id))

    def _execution_path(self, idempotency_key: str) -> Path:
        digest = stable_key_digest(idempotency_key)
        return self.executions_dir / digest[:2] / ("%s.json" % digest)

    @contextmanager
    def _state_guard(self, *, shared: bool = False) -> Iterator[None]:
        lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        with self._lock:
            with self.lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), lock_mode)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, default=str, indent=2).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            self._sync_directory(path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _sync_directory(path: Path) -> None:
        directory_descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        except OSError:
            pass
        finally:
            os.close(directory_descriptor)

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
        bind_node_task_request_fingerprint(state)
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
        candidate = node_task_enqueue_candidate(state)
        task_key = self._task_key(candidate.run_id, candidate.task_id)
        queue_member = "%s|%s" % (candidate.run_id, candidate.task_id)
        serialized = serialize_state(candidate)
        script = """
        if redis.call('EXISTS', KEYS[1]) == 1 then
            local fingerprint = redis.call('HGET', KEYS[1], 'request_fingerprint') or ''
            if fingerprint ~= ARGV[1] or fingerprint == '' then return -1 end
            local status = redis.call('HGET', KEYS[1], 'status') or ''
            if status == 'queued' or status == 'pending' or status == 'retry' then
                redis.call('SADD', KEYS[3], ARGV[3])
                redis.call('SADD', KEYS[4], ARGV[12])
            end
            redis.call('SADD', KEYS[2], ARGV[3])
            return 2
        end
        redis.call('HSET', KEYS[1],
            'run_id', ARGV[2], 'task_id', ARGV[3], 'status', ARGV[4],
            'idempotency_key', ARGV[5], 'request_fingerprint', ARGV[1],
            'attempts', ARGV[6], 'lease_owner', ARGV[7],
            'lease_generation', ARGV[8], 'lease_until', ARGV[9],
            'updated_at', ARGV[10], 'payload', ARGV[11])
        redis.call('SADD', KEYS[2], ARGV[3])
        redis.call('SADD', KEYS[3], ARGV[3])
        redis.call('SADD', KEYS[4], ARGV[12])
        return 1
        """
        if hasattr(self.client, "eval"):
            outcome = int(
                self.client.eval(
                    script,
                    4,
                    task_key,
                    self._run_tasks_key(candidate.run_id),
                    self._queue_key(candidate.run_id),
                    self._global_queue_key(),
                    serialized["request_fingerprint"],
                    serialized["run_id"],
                    serialized["task_id"],
                    serialized["status"],
                    serialized["idempotency_key"],
                    serialized["attempts"],
                    serialized["lease_owner"],
                    serialized["lease_generation"],
                    serialized["lease_until"],
                    serialized["updated_at"],
                    serialized["payload"],
                    queue_member,
                )
                or 0
            )
            existing = self.get_node_task(candidate.run_id, candidate.task_id)
            if not existing:
                raise RuntimeError("node task enqueue did not persist")
            if outcome == -1:
                raise NodeTaskRequestConflict(
                    candidate.run_id,
                    candidate.task_id,
                    existing.request_fingerprint,
                    candidate.request_fingerprint,
                )
            if outcome not in {1, 2}:
                raise RuntimeError("node task enqueue returned an invalid outcome")
            state.request_fingerprint = candidate.request_fingerprint
            state.status = candidate.status
            return existing

        existing = self.get_node_task(candidate.run_id, candidate.task_id)
        if existing:
            assert_node_task_request_match(existing, candidate)
            return existing
        self.upsert_node_task(candidate)
        self.client.sadd(self._queue_key(candidate.run_id), candidate.task_id)
        self.client.sadd(self._global_queue_key(), queue_member)
        state.request_fingerprint = candidate.request_fingerprint
        state.status = candidate.status
        return candidate

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        task_key = self._task_key(run_id, task_id)
        lease_owner = required_text(lease_owner, "lease_owner")
        lease_until = future_timestamp(lease_seconds)
        script = """
        local status = redis.call('HGET', KEYS[1], 'status')
        if status ~= 'queued' and status ~= 'pending' and status ~= 'retry' then return 0 end
        redis.call('HSET', KEYS[1], 'status', 'running', 'lease_owner', ARGV[1], 'lease_until', ARGV[2], 'updated_at', ARGV[3])
        redis.call('HINCRBY', KEYS[1], 'attempts', 1)
        redis.call('HINCRBY', KEYS[1], 'lease_generation', 1)
        redis.call('SREM', KEYS[2], ARGV[4])
        return 1
        """
        if not hasattr(self.client, "eval"):
            state = self.get_node_task(run_id, task_id)
            if not state or state.status not in {"queued", "pending", "retry"}:
                return None
            state.status = "running"
            state.attempts += 1
            state.lease_owner = lease_owner
            state.lease_generation += 1
            state.lease_until = lease_until
            self.upsert_node_task(state)
            self.client.srem(self._queue_key(run_id), task_id)
            self.client.srem(self._global_queue_key(), "%s|%s" % (run_id, task_id))
            return state
        claimed = self.client.eval(script, 2, task_key, self._queue_key(run_id), lease_owner, lease_until, datetime.now().isoformat(), task_id)
        if claimed:
            self.client.srem(self._global_queue_key(), "%s|%s" % (run_id, task_id))
        return self.get_node_task(run_id, task_id) if claimed else None

    def complete_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        lease_owner: str = "",
        lease_generation: int = 0,
    ) -> NodeTaskState:
        task_key = self._task_key(run_id, task_id)
        state = self.get_node_task(run_id, task_id)
        if not state:
            raise StaleNodeTaskFence("node task does not exist")
        assert_node_task_fence(state, lease_owner, lease_generation, status)
        if state.status != "running":
            return state
        next_payload = dict(state.payload)
        next_payload.update(payload or {})
        script = """
        if redis.call('HGET', KEYS[1], 'status') ~= 'running' then return 0 end
        if redis.call('HGET', KEYS[1], 'lease_owner') ~= ARGV[1] then return 0 end
        if tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0') ~= tonumber(ARGV[2]) then return 0 end
        redis.call('HSET', KEYS[1], 'status', ARGV[3], 'payload', ARGV[4], 'lease_until', '', 'updated_at', ARGV[5])
        return 1
        """
        if hasattr(self.client, "eval"):
            updated = self.client.eval(
                script,
                1,
                task_key,
                lease_owner,
                int(lease_generation),
                status,
                json.dumps(next_payload, ensure_ascii=False, default=str),
                datetime.now().isoformat(),
            )
            if not updated:
                raise StaleNodeTaskFence("node task lease was superseded before completion")
            return self.get_node_task(run_id, task_id) or state
        state.status = status
        state.payload = next_payload
        state.lease_until = ""
        return self.upsert_node_task(state)

    def heartbeat_node_task(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int = 300,
        lease_generation: int = 0,
    ) -> bool:
        task_key = self._task_key(run_id, task_id)
        script = """
        if redis.call('HGET', KEYS[1], 'status') ~= 'running' then return 0 end
        if redis.call('HGET', KEYS[1], 'lease_owner') ~= ARGV[1] then return 0 end
        if tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0') ~= tonumber(ARGV[2]) then return 0 end
        redis.call('HSET', KEYS[1], 'lease_until', ARGV[3], 'updated_at', ARGV[4])
        return 1
        """
        if hasattr(self.client, "eval"):
            return bool(
                self.client.eval(
                    script,
                    1,
                    task_key,
                    lease_owner,
                    int(lease_generation),
                    future_timestamp(lease_seconds),
                    datetime.now().isoformat(),
                )
            )
        state = self.get_node_task(run_id, task_id)
        if not node_task_fence_matches(state, lease_owner, lease_generation):
            return False
        state.lease_until = future_timestamp(lease_seconds)
        self.upsert_node_task(state)
        return True

    def fence_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        expected_generation: int = 0,
    ) -> NodeTaskState:
        state = self.get_node_task(run_id, task_id)
        if not state:
            raise StaleNodeTaskFence("node task does not exist")
        assert_expected_generation(state, expected_generation)
        if state.status in {"completed", "partial", "failed", "timeout", "canceled"}:
            return state
        next_payload = dict(state.payload)
        next_payload.update(payload or {})
        script = """
        if tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0') ~= tonumber(ARGV[1]) then return 0 end
        local status = redis.call('HGET', KEYS[1], 'status')
        if status == 'completed' or status == 'partial' or status == 'failed' or status == 'timeout' or status == 'canceled' then return 2 end
        redis.call('HINCRBY', KEYS[1], 'lease_generation', 1)
        redis.call('HSET', KEYS[1], 'status', ARGV[2], 'payload', ARGV[3], 'lease_owner', '', 'lease_until', '', 'updated_at', ARGV[4])
        return 1
        """
        if hasattr(self.client, "eval"):
            updated = self.client.eval(
                script,
                1,
                self._task_key(run_id, task_id),
                int(state.lease_generation),
                status,
                json.dumps(next_payload, ensure_ascii=False, default=str),
                datetime.now().isoformat(),
            )
            if not updated:
                raise StaleNodeTaskFence("node task generation changed before fencing")
            return self.get_node_task(run_id, task_id) or state
        state.status = status
        state.payload = next_payload
        state.lease_generation += 1
        state.lease_owner = ""
        state.lease_until = ""
        return self.upsert_node_task(state)

    def recover_expired_node_tasks(self, max_attempts: int = 3) -> int:
        recovered = 0
        now = datetime.now()
        pattern = self._key("task") + ":*"
        for key in self.client.scan_iter(match=pattern, count=200):
            state = deserialize_state(self.client.hgetall(key))
            if state.status != "running" or not lease_expired(state.lease_until, now):
                continue
            next_status = "retry" if state.attempts < max(1, max_attempts) else "failed"
            next_payload = dict(state.payload)
            next_payload["recoveredAfterLeaseExpiry"] = True
            script = """
            if redis.call('HGET', KEYS[1], 'status') ~= 'running' then return 0 end
            if tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0') ~= tonumber(ARGV[1]) then return 0 end
            local lease_until = redis.call('HGET', KEYS[1], 'lease_until') or ''
            if lease_until == '' or lease_until > ARGV[2] then return 0 end
            redis.call('HSET', KEYS[1], 'status', ARGV[3], 'lease_owner', '', 'lease_until', '',
                'payload', ARGV[4], 'updated_at', ARGV[2])
            return 1
            """
            if hasattr(self.client, "eval"):
                updated = self.client.eval(
                    script,
                    1,
                    key,
                    int(state.lease_generation),
                    now.isoformat(),
                    next_status,
                    json.dumps(next_payload, ensure_ascii=False, default=str),
                )
                if not updated:
                    continue
                state = deserialize_state(self.client.hgetall(key))
            else:
                state.status = next_status
                state.lease_owner = ""
                state.lease_until = ""
                state.payload = next_payload
                self.upsert_node_task(state)
            if state.status == "retry":
                self.client.sadd(self._queue_key(state.run_id), state.task_id)
                self.client.sadd(self._global_queue_key(), "%s|%s" % (state.run_id, state.task_id))
            recovered += 1
        return recovered

    def claim_next_node_task(
        self,
        lease_owner: str,
        lease_seconds: int = 300,
        task_kinds: Optional[List[str]] = None,
    ) -> Optional[NodeTaskState]:
        allowed = {str(item) for item in (task_kinds or []) if str(item)}
        for encoded in sorted(self.client.smembers(self._global_queue_key()) or []):
            run_id, separator, task_id = str(encoded).partition("|")
            if not separator:
                self.client.srem(self._global_queue_key(), encoded)
                continue
            state = self.get_node_task(run_id, task_id)
            if not state or state.status not in {"queued", "pending", "retry"}:
                self.client.srem(self._global_queue_key(), encoded)
                continue
            if allowed and str(state.payload.get("taskKind") or "") not in allowed:
                continue
            claimed = self.claim_node_task(run_id, task_id, lease_owner, lease_seconds)
            if claimed:
                return claimed
        return None

    def claim_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_seconds: int = 300,
        *,
        allow_expired_takeover: bool = False,
    ) -> ExecutionJournalClaim:
        key = required_text(idempotency_key, "idempotency_key")
        owner = required_text(lease_owner, "lease_owner")
        execution_key = self._execution_key(key)
        now = datetime.now().isoformat()
        lease_until = future_timestamp(lease_seconds)
        script = """
        if redis.call('EXISTS', KEYS[1]) == 0 then
            redis.call('HSET', KEYS[1],
                'idempotency_key', ARGV[1], 'status', 'in_progress',
                'lease_owner', ARGV[2], 'lease_generation', 1,
                'lease_until', ARGV[3], 'result_hash', '',
                'created_at', ARGV[4], 'updated_at', ARGV[4], 'payload', '{}')
            return {'acquired', '1'}
        end
        local status = redis.call('HGET', KEYS[1], 'status') or ''
        local generation = tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0')
        if status ~= 'in_progress' then return {'replay', tostring(generation)} end
        local current_until = redis.call('HGET', KEYS[1], 'lease_until') or ''
        if current_until ~= '' and current_until <= ARGV[4] then
            if ARGV[5] ~= '1' then return {'expired_unsafe', tostring(generation)} end
            generation = generation + 1
            redis.call('HSET', KEYS[1], 'lease_owner', ARGV[2], 'lease_generation', generation,
                'lease_until', ARGV[3], 'updated_at', ARGV[4])
            return {'acquired', tostring(generation)}
        end
        return {'in_progress', tostring(generation)}
        """
        if hasattr(self.client, "eval"):
            raw = self.client.eval(
                script,
                1,
                execution_key,
                key,
                owner,
                lease_until,
                now,
                "1" if allow_expired_takeover else "0",
            )
            outcome = str(raw[0]) if isinstance(raw, (list, tuple)) and raw else "in_progress"
            entry = self.get_execution(key)
            if not entry:
                raise RuntimeError("execution journal claim did not persist")
            return ExecutionJournalClaim(outcome, entry)
        current = self.get_execution(key)
        claim = next_execution_claim(current, key, owner, lease_seconds, allow_expired_takeover)
        if claim.acquired:
            self.client.hset(execution_key, mapping=serialize_execution_entry(claim.entry))
        return claim

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        data = self.client.hgetall(self._execution_key(idempotency_key))
        return deserialize_execution_entry(data) if data else None

    def complete_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_generation: int,
        status: str,
        payload: Dict[str, Any] | None = None,
        result_hash: str = "",
    ) -> ExecutionJournalEntry:
        validate_execution_status(status)
        execution_key = self._execution_key(idempotency_key)
        script = """
        if redis.call('HGET', KEYS[1], 'status') ~= 'in_progress' then return 0 end
        if redis.call('HGET', KEYS[1], 'lease_owner') ~= ARGV[1] then return 0 end
        if tonumber(redis.call('HGET', KEYS[1], 'lease_generation') or '0') ~= tonumber(ARGV[2]) then return 0 end
        redis.call('HSET', KEYS[1], 'status', ARGV[3], 'payload', ARGV[4],
            'result_hash', ARGV[5], 'lease_until', '', 'updated_at', ARGV[6])
        return 1
        """
        if hasattr(self.client, "eval"):
            updated = self.client.eval(
                script,
                1,
                execution_key,
                lease_owner,
                int(lease_generation),
                status,
                json.dumps(payload or {}, ensure_ascii=False, default=str),
                result_hash,
                datetime.now().isoformat(),
            )
            if not updated:
                current = self.get_execution(idempotency_key)
                if current and current.status == status and current.lease_owner == lease_owner and current.lease_generation == lease_generation:
                    return current
                raise StaleExecutionFence("execution journal lease was superseded before completion")
            completed = self.get_execution(idempotency_key)
            if not completed:
                raise RuntimeError("execution journal completion was not persisted")
            return completed
        current = self.get_execution(idempotency_key)
        completed = completed_execution_entry(
            current,
            idempotency_key,
            lease_owner,
            lease_generation,
            status,
            payload,
            result_hash,
        )
        self.client.hset(execution_key, mapping=serialize_execution_entry(completed))
        return completed

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

    def _global_queue_key(self) -> str:
        return self._key("global_queue")

    def _execution_key(self, idempotency_key: str) -> str:
        return self._key("execution", stable_key_digest(idempotency_key))


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
                    request_fingerprint TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_generation BIGINT NOT NULL DEFAULT 0,
                    lease_until TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    PRIMARY KEY (run_id, task_id)
                )
                """
            )
            cur.execute(
                "ALTER TABLE merchant_ai_node_task_state ADD COLUMN IF NOT EXISTS lease_generation BIGINT NOT NULL DEFAULT 0"
            )
            cur.execute(
                "ALTER TABLE merchant_ai_node_task_state ADD COLUMN IF NOT EXISTS request_fingerprint TEXT NOT NULL DEFAULT ''"
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS merchant_ai_execution_journal (
                    idempotency_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    lease_owner TEXT NOT NULL,
                    lease_generation BIGINT NOT NULL,
                    lease_until TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
        self.conn.commit()

    def upsert_node_task(self, state: NodeTaskState) -> NodeTaskState:
        bind_node_task_request_fingerprint(state)
        state.updated_at = datetime.now().isoformat()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merchant_ai_node_task_state
                (run_id, task_id, status, idempotency_key, request_fingerprint, attempts, lease_owner,
                 lease_generation, lease_until, updated_at, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, task_id) DO UPDATE SET
                    status=EXCLUDED.status,
                    idempotency_key=EXCLUDED.idempotency_key,
                    request_fingerprint=EXCLUDED.request_fingerprint,
                    attempts=EXCLUDED.attempts,
                    lease_owner=EXCLUDED.lease_owner,
                    lease_generation=EXCLUDED.lease_generation,
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
                "SELECT run_id, task_id, status, idempotency_key, request_fingerprint, attempts, lease_owner, lease_generation, lease_until, updated_at, payload FROM merchant_ai_node_task_state WHERE run_id=%s AND task_id=%s",
                (run_id, task_id),
            )
            row = cur.fetchone()
        return row_to_state(row) if row else None

    def list_node_tasks(self, run_id: str) -> List[NodeTaskState]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, task_id, status, idempotency_key, request_fingerprint, attempts, lease_owner, lease_generation, lease_until, updated_at, payload FROM merchant_ai_node_task_state WHERE run_id=%s ORDER BY task_id",
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
        candidate = node_task_enqueue_candidate(state)
        candidate.updated_at = datetime.now().isoformat()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merchant_ai_node_task_state
                    (run_id, task_id, status, idempotency_key, request_fingerprint, attempts,
                     lease_owner, lease_generation, lease_until, updated_at, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, task_id) DO UPDATE
                SET request_fingerprint=merchant_ai_node_task_state.request_fingerprint
                WHERE merchant_ai_node_task_state.request_fingerprint=EXCLUDED.request_fingerprint
                  AND merchant_ai_node_task_state.request_fingerprint<>''
                RETURNING run_id, task_id, status, idempotency_key, request_fingerprint, attempts,
                          lease_owner, lease_generation, lease_until, updated_at, payload
                """,
                state_to_row(candidate),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            persisted = row_to_state(row)
            state.request_fingerprint = candidate.request_fingerprint
            state.status = candidate.status
            return persisted
        existing = self.get_node_task(candidate.run_id, candidate.task_id)
        if not existing:
            raise RuntimeError("node task enqueue conflict could not be read")
        assert_node_task_request_match(existing, candidate)
        raise RuntimeError("node task enqueue did not return its persisted state")

    def claim_node_task(self, run_id: str, task_id: str, lease_owner: str, lease_seconds: int = 300) -> Optional[NodeTaskState]:
        lease_owner = required_text(lease_owner, "lease_owner")
        lease_until = future_timestamp(lease_seconds)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_node_task_state
                SET status='running', attempts=attempts+1, lease_owner=%s,
                    lease_generation=lease_generation+1, lease_until=%s, updated_at=%s
                WHERE run_id=%s AND task_id=%s AND status IN ('queued','pending','retry')
                RETURNING run_id, task_id, status, idempotency_key, request_fingerprint, attempts,
                          lease_owner, lease_generation, lease_until, updated_at, payload
                """,
                (lease_owner, lease_until, datetime.now().isoformat(), run_id, task_id),
            )
            row = cur.fetchone()
        self.conn.commit()
        return row_to_state(row) if row else None

    def claim_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_seconds: int = 300,
        *,
        allow_expired_takeover: bool = False,
    ) -> ExecutionJournalClaim:
        key = required_text(idempotency_key, "idempotency_key")
        owner = required_text(lease_owner, "lease_owner")
        now = datetime.now().isoformat()
        lease_until = future_timestamp(lease_seconds)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO merchant_ai_execution_journal
                    (idempotency_key, status, lease_owner, lease_generation, lease_until,
                     result_hash, created_at, updated_at, payload)
                VALUES (%s, 'in_progress', %s, 1, %s, '', %s, %s, '{}'::jsonb)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING idempotency_key, status, lease_owner, lease_generation, lease_until,
                          result_hash, created_at, updated_at, payload
                """,
                (key, owner, lease_until, now, now),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            return ExecutionJournalClaim("acquired", row_to_execution_entry(row))
        current = self.get_execution(key)
        if not current:
            raise RuntimeError("execution journal claim could not be read")
        if current.status in TERMINAL_EXECUTION_STATUSES:
            return ExecutionJournalClaim("replay", current)
        if not lease_expired(current.lease_until):
            return ExecutionJournalClaim("in_progress", current)
        if not allow_expired_takeover:
            return ExecutionJournalClaim("expired_unsafe", current)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_execution_journal
                SET lease_owner=%s, lease_generation=lease_generation+1, lease_until=%s, updated_at=%s
                WHERE idempotency_key=%s AND status='in_progress'
                  AND lease_generation=%s AND lease_until <> '' AND lease_until <= %s
                RETURNING idempotency_key, status, lease_owner, lease_generation, lease_until,
                          result_hash, created_at, updated_at, payload
                """,
                (owner, lease_until, now, key, current.lease_generation, now),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            return ExecutionJournalClaim("acquired", row_to_execution_entry(row))
        latest = self.get_execution(key)
        if not latest:
            raise RuntimeError("execution journal disappeared during takeover")
        return ExecutionJournalClaim("replay" if latest.status in TERMINAL_EXECUTION_STATUSES else "in_progress", latest)

    def get_execution(self, idempotency_key: str) -> Optional[ExecutionJournalEntry]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT idempotency_key, status, lease_owner, lease_generation, lease_until,
                       result_hash, created_at, updated_at, payload
                FROM merchant_ai_execution_journal WHERE idempotency_key=%s
                """,
                (idempotency_key,),
            )
            row = cur.fetchone()
        return row_to_execution_entry(row) if row else None

    def complete_execution(
        self,
        idempotency_key: str,
        lease_owner: str,
        lease_generation: int,
        status: str,
        payload: Dict[str, Any] | None = None,
        result_hash: str = "",
    ) -> ExecutionJournalEntry:
        validate_execution_status(status)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_execution_journal
                SET status=%s, payload=%s::jsonb, result_hash=%s, lease_until='', updated_at=%s
                WHERE idempotency_key=%s AND status='in_progress'
                  AND lease_owner=%s AND lease_generation=%s
                RETURNING idempotency_key, status, lease_owner, lease_generation, lease_until,
                          result_hash, created_at, updated_at, payload
                """,
                (
                    status,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                    result_hash,
                    datetime.now().isoformat(),
                    idempotency_key,
                    lease_owner,
                    int(lease_generation),
                ),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            return row_to_execution_entry(row)
        current = self.get_execution(idempotency_key)
        if current and current.status == status and current.lease_owner == lease_owner and current.lease_generation == lease_generation:
            return current
        raise StaleExecutionFence("execution journal lease was superseded before completion")

    def complete_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        lease_owner: str = "",
        lease_generation: int = 0,
    ) -> NodeTaskState:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_node_task_state
                SET status=%s, payload=payload || %s::jsonb, lease_until='', updated_at=%s
                WHERE run_id=%s AND task_id=%s AND status='running'
                  AND lease_owner=%s AND lease_generation=%s
                RETURNING run_id, task_id, status, idempotency_key, request_fingerprint, attempts,
                          lease_owner, lease_generation, lease_until, updated_at, payload
                """,
                (
                    status,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                    datetime.now().isoformat(),
                    run_id,
                    task_id,
                    lease_owner,
                    int(lease_generation),
                ),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            return row_to_state(row)
        current = self.get_node_task(run_id, task_id)
        if current and current.status == status and current.lease_owner == lease_owner and current.lease_generation == lease_generation:
            return current
        raise StaleNodeTaskFence("node task lease was superseded before completion")

    def heartbeat_node_task(
        self,
        run_id: str,
        task_id: str,
        lease_owner: str,
        lease_seconds: int = 300,
        lease_generation: int = 0,
    ) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_node_task_state SET lease_until=%s, updated_at=%s
                WHERE run_id=%s AND task_id=%s AND status='running'
                  AND lease_owner=%s AND lease_generation=%s
                """,
                (
                    future_timestamp(lease_seconds),
                    datetime.now().isoformat(),
                    run_id,
                    task_id,
                    lease_owner,
                    int(lease_generation),
                ),
            )
            updated = cur.rowcount
        self.conn.commit()
        return bool(updated)

    def fence_node_task(
        self,
        run_id: str,
        task_id: str,
        status: str,
        payload: Dict[str, Any] | None = None,
        *,
        expected_generation: int = 0,
    ) -> NodeTaskState:
        current = self.get_node_task(run_id, task_id)
        if not current:
            raise StaleNodeTaskFence("node task does not exist")
        expected = int(expected_generation or current.lease_generation)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_node_task_state
                SET status=%s, payload=payload || %s::jsonb, lease_owner='', lease_until='',
                    lease_generation=lease_generation+1, updated_at=%s
                WHERE run_id=%s AND task_id=%s AND lease_generation=%s
                  AND status NOT IN ('completed','partial','failed','timeout','canceled')
                RETURNING run_id, task_id, status, idempotency_key, request_fingerprint, attempts,
                          lease_owner, lease_generation, lease_until, updated_at, payload
                """,
                (
                    status,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                    datetime.now().isoformat(),
                    run_id,
                    task_id,
                    expected,
                ),
            )
            row = cur.fetchone()
        self.conn.commit()
        if row:
            return row_to_state(row)
        latest = self.get_node_task(run_id, task_id)
        if latest and latest.status in {"completed", "partial", "failed", "timeout", "canceled"}:
            return latest
        raise StaleNodeTaskFence("node task generation changed before fencing")

    def recover_expired_node_tasks(self, max_attempts: int = 3) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE merchant_ai_node_task_state
                SET status=CASE WHEN attempts < %s THEN 'retry' ELSE 'failed' END,
                    lease_owner='', lease_until='', updated_at=%s,
                    payload=payload || '{"recoveredAfterLeaseExpiry": true}'::jsonb
                WHERE status='running' AND lease_until <> '' AND lease_until < %s
                """,
                (max(1, max_attempts), datetime.now().isoformat(), datetime.now().isoformat()),
            )
            updated = cur.rowcount
        self.conn.commit()
        return int(updated or 0)

    def claim_next_node_task(
        self,
        lease_owner: str,
        lease_seconds: int = 300,
        task_kinds: Optional[List[str]] = None,
    ) -> Optional[NodeTaskState]:
        lease_owner = required_text(lease_owner, "lease_owner")
        lease_until = future_timestamp(lease_seconds)
        kinds = [str(item) for item in (task_kinds or []) if str(item)]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT run_id, task_id
                    FROM merchant_ai_node_task_state
                    WHERE status IN ('queued','pending','retry')
                      AND (%s = FALSE OR payload->>'taskKind' = ANY(%s))
                    ORDER BY updated_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE merchant_ai_node_task_state task
                SET status='running', attempts=task.attempts+1, lease_owner=%s,
                    lease_generation=task.lease_generation+1, lease_until=%s, updated_at=%s
                FROM candidate
                WHERE task.run_id=candidate.run_id AND task.task_id=candidate.task_id
                RETURNING task.run_id, task.task_id, task.status, task.idempotency_key,
                          task.request_fingerprint, task.attempts, task.lease_owner,
                          task.lease_generation, task.lease_until, task.updated_at, task.payload
                """,
                (bool(kinds), kinds, lease_owner, lease_until, datetime.now().isoformat()),
            )
            row = cur.fetchone()
        self.conn.commit()
        return row_to_state(row) if row else None


def create_runtime_state_store(settings: Settings) -> RuntimeStateStore:
    backend = str(settings.runtime_state_backend or "file").strip().lower()
    if backend == "memory":
        return MemoryRuntimeStateStore()
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
        "request_fingerprint": state.request_fingerprint,
        "attempts": str(int(state.attempts or 0)),
        "lease_owner": state.lease_owner,
        "lease_generation": str(int(state.lease_generation or 0)),
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
        request_fingerprint=str(data.get("request_fingerprint") or data.get("requestFingerprint") or ""),
        attempts=int(data.get("attempts") or 0),
        lease_owner=str(data.get("lease_owner") or data.get("leaseOwner") or ""),
        lease_generation=int(data.get("lease_generation") or data.get("leaseGeneration") or 0),
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
        state.request_fingerprint,
        int(state.attempts or 0),
        state.lease_owner,
        int(state.lease_generation or 0),
        state.lease_until,
        state.updated_at,
        json.dumps(state.payload or {}, ensure_ascii=False, default=str),
    )


def row_to_state(row: Any) -> NodeTaskState:
    (
        run_id,
        task_id,
        status,
        idempotency_key,
        request_fingerprint,
        attempts,
        lease_owner,
        lease_generation,
        lease_until,
        updated_at,
        payload,
    ) = row
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
        request_fingerprint=str(request_fingerprint or ""),
        attempts=int(attempts or 0),
        lease_owner=lease_owner,
        lease_generation=int(lease_generation or 0),
        lease_until=lease_until,
        updated_at=updated_at,
        payload=payload if isinstance(payload, dict) else {},
    )


def lease_expired(value: str, now: Optional[datetime] = None) -> bool:
    if not value:
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None) <= (now or datetime.now()).replace(tzinfo=None)
    except ValueError:
        return True


def required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("%s is required" % field_name)
    return text


def future_timestamp(seconds: int) -> str:
    return (datetime.now() + timedelta(seconds=max(1, int(seconds or 1)))).isoformat()


def stable_key_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(required_text(value, "key").encode("utf-8")).hexdigest()


def clone_node_task(state: NodeTaskState) -> NodeTaskState:
    return NodeTaskState(
        run_id=state.run_id,
        task_id=state.task_id,
        status=state.status,
        idempotency_key=state.idempotency_key,
        request_fingerprint=state.request_fingerprint,
        attempts=int(state.attempts or 0),
        lease_owner=state.lease_owner,
        lease_generation=int(state.lease_generation or 0),
        lease_until=state.lease_until,
        updated_at=state.updated_at,
        payload=dict(state.payload or {}),
    )


def bind_node_task_request_fingerprint(state: NodeTaskState) -> str:
    fingerprint = str(state.request_fingerprint or "").strip()
    if not fingerprint:
        payload = dict(state.payload or {})
        supplied = str(payload.get("requestFingerprint") or payload.get("request_fingerprint") or "").strip()
        identity = (
            {
                "idempotencyKey": str(state.idempotency_key or ""),
                "suppliedRequestFingerprint": supplied,
            }
            if supplied
            else {
                "idempotencyKey": str(state.idempotency_key or ""),
                "requestPayload": payload,
            }
        )
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = stable_key_digest(canonical)
    state.request_fingerprint = fingerprint
    return fingerprint


def node_task_enqueue_candidate(state: NodeTaskState) -> NodeTaskState:
    candidate = clone_node_task(state)
    bind_node_task_request_fingerprint(candidate)
    candidate.status = "queued" if candidate.status in {"", "pending"} else candidate.status
    state.request_fingerprint = candidate.request_fingerprint
    state.status = candidate.status
    return candidate


def assert_node_task_request_match(existing: NodeTaskState, requested: NodeTaskState) -> None:
    existing_fingerprint = str(existing.request_fingerprint or "").strip()
    requested_fingerprint = str(requested.request_fingerprint or "").strip()
    if existing_fingerprint != requested_fingerprint or not existing_fingerprint:
        raise NodeTaskRequestConflict(
            requested.run_id,
            requested.task_id,
            existing_fingerprint,
            requested_fingerprint,
        )


def node_task_fence_matches(
    state: Optional[NodeTaskState],
    lease_owner: str,
    lease_generation: int,
) -> bool:
    return bool(
        state
        and state.status == "running"
        and state.lease_owner == str(lease_owner or "")
        and int(state.lease_generation or 0) == int(lease_generation or 0)
        and int(lease_generation or 0) > 0
    )


def assert_node_task_fence(
    state: NodeTaskState,
    lease_owner: str,
    lease_generation: int,
    requested_status: str,
) -> None:
    if state.status != "running":
        if (
            state.status == requested_status
            and state.lease_owner == str(lease_owner or "")
            and state.lease_generation == int(lease_generation or 0)
            and state.lease_generation > 0
        ):
            return
        raise StaleNodeTaskFence("node task is no longer owned by this lease")
    if not node_task_fence_matches(state, lease_owner, lease_generation):
        raise StaleNodeTaskFence("node task owner or generation does not match")


def assert_expected_generation(state: NodeTaskState, expected_generation: int) -> None:
    expected = int(expected_generation or state.lease_generation)
    if int(state.lease_generation or 0) != expected:
        raise StaleNodeTaskFence("node task generation changed before fencing")


def clone_execution_entry(entry: ExecutionJournalEntry) -> ExecutionJournalEntry:
    return ExecutionJournalEntry(
        idempotency_key=entry.idempotency_key,
        status=entry.status,
        lease_owner=entry.lease_owner,
        lease_generation=int(entry.lease_generation or 0),
        lease_until=entry.lease_until,
        result_hash=entry.result_hash,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        payload=dict(entry.payload or {}),
    )


def next_execution_claim(
    current: Optional[ExecutionJournalEntry],
    idempotency_key: str,
    lease_owner: str,
    lease_seconds: int,
    allow_expired_takeover: bool,
) -> ExecutionJournalClaim:
    now = datetime.now().isoformat()
    if current is None:
        return ExecutionJournalClaim(
            "acquired",
            ExecutionJournalEntry(
                idempotency_key=idempotency_key,
                status="in_progress",
                lease_owner=lease_owner,
                lease_generation=1,
                lease_until=future_timestamp(lease_seconds),
                created_at=now,
                updated_at=now,
            ),
        )
    if current.status in TERMINAL_EXECUTION_STATUSES:
        return ExecutionJournalClaim("replay", clone_execution_entry(current))
    if not lease_expired(current.lease_until):
        return ExecutionJournalClaim("in_progress", clone_execution_entry(current))
    if not allow_expired_takeover:
        return ExecutionJournalClaim("expired_unsafe", clone_execution_entry(current))
    takeover = clone_execution_entry(current)
    takeover.lease_owner = lease_owner
    takeover.lease_generation += 1
    takeover.lease_until = future_timestamp(lease_seconds)
    takeover.updated_at = now
    return ExecutionJournalClaim("acquired", takeover)


def validate_execution_status(status: str) -> None:
    if status not in TERMINAL_EXECUTION_STATUSES:
        raise ValueError("execution completion status must be terminal")


def completed_execution_entry(
    current: Optional[ExecutionJournalEntry],
    idempotency_key: str,
    lease_owner: str,
    lease_generation: int,
    status: str,
    payload: Dict[str, Any] | None,
    result_hash: str,
) -> ExecutionJournalEntry:
    validate_execution_status(status)
    if current is None:
        raise StaleExecutionFence("execution journal entry does not exist")
    generation = int(lease_generation or 0)
    if current.status in TERMINAL_EXECUTION_STATUSES:
        if current.status == status and current.lease_owner == lease_owner and current.lease_generation == generation:
            return clone_execution_entry(current)
        raise StaleExecutionFence("execution journal entry is already terminal")
    if current.lease_owner != lease_owner or current.lease_generation != generation or generation <= 0:
        raise StaleExecutionFence("execution journal owner or generation does not match")
    completed = clone_execution_entry(current)
    completed.status = status
    completed.payload = dict(payload or {})
    completed.result_hash = str(result_hash or "")
    completed.lease_until = ""
    completed.updated_at = datetime.now().isoformat()
    return completed


def serialize_execution_entry(entry: ExecutionJournalEntry) -> Dict[str, str]:
    return {
        "idempotency_key": entry.idempotency_key,
        "status": entry.status,
        "lease_owner": entry.lease_owner,
        "lease_generation": str(int(entry.lease_generation or 0)),
        "lease_until": entry.lease_until,
        "result_hash": entry.result_hash,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "payload": json.dumps(entry.payload or {}, ensure_ascii=False, default=str),
    }


def deserialize_execution_entry(data: Dict[str, Any]) -> ExecutionJournalEntry:
    payload = data.get("payload") or "{}"
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return ExecutionJournalEntry(
        idempotency_key=str(data.get("idempotency_key") or ""),
        status=str(data.get("status") or "in_progress"),
        lease_owner=str(data.get("lease_owner") or ""),
        lease_generation=int(data.get("lease_generation") or 0),
        lease_until=str(data.get("lease_until") or ""),
        result_hash=str(data.get("result_hash") or ""),
        created_at=str(data.get("created_at") or datetime.now().isoformat()),
        updated_at=str(data.get("updated_at") or datetime.now().isoformat()),
        payload=payload if isinstance(payload, dict) else {},
    )


def row_to_execution_entry(row: Any) -> ExecutionJournalEntry:
    idempotency_key, status, lease_owner, lease_generation, lease_until, result_hash, created_at, updated_at, payload = row
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return ExecutionJournalEntry(
        idempotency_key=str(idempotency_key or ""),
        status=str(status or "in_progress"),
        lease_owner=str(lease_owner or ""),
        lease_generation=int(lease_generation or 0),
        lease_until=str(lease_until or ""),
        result_hash=str(result_hash or ""),
        created_at=str(created_at or ""),
        updated_at=str(updated_at or ""),
        payload=payload if isinstance(payload, dict) else {},
    )
