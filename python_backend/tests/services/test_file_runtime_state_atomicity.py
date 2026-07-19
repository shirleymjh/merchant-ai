from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.runtime_state import (
    FileRuntimeStateStore,
    MemoryRuntimeStateStore,
    NodeTaskRequestConflict,
    NodeTaskState,
    PostgresRuntimeStateStore,
    RedisRuntimeStateStore,
)
from merchant_ai.services.tool_runtime import ToolRuntimeService
from merchant_ai.services.tools import ToolCapability, ToolRegistry


def _claim_task_process(
    workspace: str,
    owner: str,
    start_event: Any,
    result_queue: Any,
) -> None:
    try:
        store = FileRuntimeStateStore(Settings(harness_workspace_path=workspace))
        start_event.wait()
        claim = store.claim_node_task("run", "task", owner, lease_seconds=30)
        result_queue.put(
            {
                "owner": owner,
                "acquired": claim is not None,
                "generation": claim.lease_generation if claim else 0,
            }
        )
    except BaseException as exc:
        result_queue.put({"owner": owner, "error": repr(exc)})


def _claim_and_complete_execution_process(
    workspace: str,
    owner: str,
    start_event: Any,
    execution_count: Any,
    result_queue: Any,
) -> None:
    try:
        store = FileRuntimeStateStore(Settings(harness_workspace_path=workspace))
        start_event.wait()
        claim = store.claim_execution("execution", owner, lease_seconds=30)
        if claim.acquired:
            with execution_count.get_lock():
                execution_count.value += 1
            time.sleep(0.1)
            store.complete_execution(
                "execution",
                owner,
                claim.entry.lease_generation,
                "succeeded",
                {"owner": owner},
                "result-hash",
            )
        result_queue.put(
            {
                "owner": owner,
                "outcome": claim.outcome,
                "generation": claim.entry.lease_generation,
            }
        )
    except BaseException as exc:
        result_queue.put({"owner": owner, "error": repr(exc)})


def _enqueue_task_process(
    workspace: str,
    start_event: Any,
    result_queue: Any,
) -> None:
    try:
        store = FileRuntimeStateStore(Settings(harness_workspace_path=workspace))
        start_event.wait()
        state = store.enqueue_node_task(
            NodeTaskState(
                run_id="run",
                task_id="task",
                request_fingerprint="request-a",
                payload={"input": 1},
            )
        )
        result_queue.put({"operation": "enqueue", "status": state.status})
    except BaseException as exc:
        result_queue.put({"operation": "enqueue", "error": repr(exc)})


def _run_processes(processes: list[Any], start_event: Any, result_queue: Any) -> list[dict[str, Any]]:
    for process in processes:
        process.start()
    start_event.set()
    try:
        results = [result_queue.get(timeout=20) for _process in processes]
    finally:
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    assert not [result for result in results if "error" in result]
    return results


def test_file_task_claim_is_compare_and_set_across_processes(tmp_path: Path):
    store = FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))
    store.enqueue_node_task(NodeTaskState(run_id="run", task_id="task"))
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_claim_task_process,
            args=(str(tmp_path), "worker-%d" % index, start_event, result_queue),
        )
        for index in range(4)
    ]

    results = _run_processes(processes, start_event, result_queue)
    acquired = [result for result in results if result["acquired"]]
    persisted = store.get_node_task("run", "task")

    assert len(acquired) == 1
    assert acquired[0]["generation"] == 1
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.attempts == 1
    assert persisted.lease_generation == 1
    assert persisted.lease_owner == acquired[0]["owner"]


def test_file_execution_journal_runs_duplicate_once_across_processes(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    execution_count = context.Value("i", 0)
    processes = [
        context.Process(
            target=_claim_and_complete_execution_process,
            args=(
                str(tmp_path),
                "worker-%d" % index,
                start_event,
                execution_count,
                result_queue,
            ),
        )
        for index in range(4)
    ]

    results = _run_processes(processes, start_event, result_queue)
    acquired = [result for result in results if result["outcome"] == "acquired"]
    store = FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))
    completed = store.get_execution("execution")

    assert len(acquired) == 1
    assert execution_count.value == 1
    assert {result["outcome"] for result in results}.issubset({"acquired", "in_progress", "replay"})
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.lease_generation == 1
    assert completed.payload["owner"] == acquired[0]["owner"]


def test_file_enqueue_cannot_roll_back_a_concurrent_claim(tmp_path: Path):
    store = FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))
    store.enqueue_node_task(
        NodeTaskState(
            run_id="run",
            task_id="task",
            request_fingerprint="request-a",
            payload={"input": 1},
        )
    )
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_enqueue_task_process,
            args=(str(tmp_path), start_event, result_queue),
        ),
        context.Process(
            target=_claim_task_process,
            args=(str(tmp_path), "worker", start_event, result_queue),
        ),
    ]

    results = _run_processes(processes, start_event, result_queue)
    persisted = store.get_node_task("run", "task")

    assert [result for result in results if result.get("acquired")]
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.attempts == 1
    assert persisted.request_fingerprint == "request-a"


@pytest.mark.parametrize("backend", ["memory", "file"])
def test_enqueue_is_idempotent_by_fingerprint_and_rejects_a_different_request(tmp_path: Path, backend: str):
    store = (
        MemoryRuntimeStateStore()
        if backend == "memory"
        else FileRuntimeStateStore(Settings(harness_workspace_path=str(tmp_path)))
    )
    first = store.enqueue_node_task(
        NodeTaskState(
            run_id="run",
            task_id="task",
            request_fingerprint="request-a",
            payload={"input": 1},
        )
    )
    claimed = store.claim_node_task("run", "task", "worker", lease_seconds=30)
    replay = store.enqueue_node_task(
        NodeTaskState(
            run_id="run",
            task_id="task",
            request_fingerprint="request-a",
            payload={"input": 999},
        )
    )

    with pytest.raises(NodeTaskRequestConflict) as conflict:
        store.enqueue_node_task(
            NodeTaskState(
                run_id="run",
                task_id="task",
                request_fingerprint="request-b",
                payload={"input": 1},
            )
        )

    persisted = store.get_node_task("run", "task")
    assert first.request_fingerprint == "request-a"
    assert claimed is not None
    assert replay.status == "running"
    assert replay.payload == {"input": 1}
    assert conflict.value.code == "NODE_TASK_REQUEST_CONFLICT"
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.request_fingerprint == "request-a"


class AtomicNodeRedisClient:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.lock = threading.RLock()

    def hgetall(self, key: str) -> dict[str, str]:
        with self.lock:
            return dict(self.hashes.get(key) or {})

    def sadd(self, key: str, value: str) -> None:
        with self.lock:
            self.sets.setdefault(key, set()).add(str(value))

    def srem(self, key: str, value: str) -> None:
        with self.lock:
            self.sets.setdefault(key, set()).discard(str(value))

    def smembers(self, key: str) -> set[str]:
        with self.lock:
            return set(self.sets.get(key) or set())

    def eval(self, script: str, key_count: int, *args: Any) -> int:
        keys = [str(value) for value in args[:key_count]]
        values = [str(value) for value in args[key_count:]]
        with self.lock:
            if "suppliedRequestFingerprint" not in script and "request_fingerprint" in script and "return 2" in script:
                task_key, run_tasks_key, queue_key, global_queue_key = keys
                fingerprint, run_id, task_id, status, idempotency_key = values[:5]
                attempts, lease_owner, lease_generation, lease_until = values[5:9]
                updated_at, payload, queue_member = values[9:12]
                existing = self.hashes.get(task_key)
                if existing:
                    if not existing.get("request_fingerprint") or existing.get("request_fingerprint") != fingerprint:
                        return -1
                    if existing.get("status") in {"queued", "pending", "retry"}:
                        self.sets.setdefault(queue_key, set()).add(task_id)
                        self.sets.setdefault(global_queue_key, set()).add(queue_member)
                    self.sets.setdefault(run_tasks_key, set()).add(task_id)
                    return 2
                self.hashes[task_key] = {
                    "run_id": run_id,
                    "task_id": task_id,
                    "status": status,
                    "idempotency_key": idempotency_key,
                    "request_fingerprint": fingerprint,
                    "attempts": attempts,
                    "lease_owner": lease_owner,
                    "lease_generation": lease_generation,
                    "lease_until": lease_until,
                    "updated_at": updated_at,
                    "payload": payload,
                }
                self.sets.setdefault(run_tasks_key, set()).add(task_id)
                self.sets.setdefault(queue_key, set()).add(task_id)
                self.sets.setdefault(global_queue_key, set()).add(queue_member)
                return 1

            task_key, queue_key = keys
            lease_owner, lease_until, updated_at, task_id = values
            existing = self.hashes.get(task_key)
            if not existing or existing.get("status") not in {"queued", "pending", "retry"}:
                return 0
            existing["status"] = "running"
            existing["attempts"] = str(int(existing.get("attempts") or 0) + 1)
            existing["lease_owner"] = lease_owner
            existing["lease_generation"] = str(int(existing.get("lease_generation") or 0) + 1)
            existing["lease_until"] = lease_until
            existing["updated_at"] = updated_at
            self.sets.setdefault(queue_key, set()).discard(task_id)
            return 1


def _redis_node_store(client: AtomicNodeRedisClient) -> RedisRuntimeStateStore:
    store = RedisRuntimeStateStore.__new__(RedisRuntimeStateStore)
    store.namespace = "test"
    store.client = client
    return store


def test_redis_lua_enqueue_is_create_or_get_and_cannot_roll_back_claim():
    store = _redis_node_store(AtomicNodeRedisClient())
    store.enqueue_node_task(
        NodeTaskState(run_id="run", task_id="task", request_fingerprint="request-a", payload={"input": 1})
    )
    start = threading.Barrier(3)
    results: list[Any] = []

    def enqueue() -> None:
        start.wait()
        results.append(
            store.enqueue_node_task(
                NodeTaskState(
                    run_id="run",
                    task_id="task",
                    request_fingerprint="request-a",
                    payload={"input": 999},
                )
            )
        )

    def claim() -> None:
        start.wait()
        results.append(store.claim_node_task("run", "task", "worker", lease_seconds=30))

    threads = [threading.Thread(target=enqueue), threading.Thread(target=claim)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    persisted = store.get_node_task("run", "task")
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.attempts == 1
    assert persisted.payload == {"input": 1}
    with pytest.raises(NodeTaskRequestConflict):
        store.enqueue_node_task(
            NodeTaskState(run_id="run", task_id="task", request_fingerprint="request-b")
        )


class AtomicNodePostgresCursor:
    def __init__(self, connection: "AtomicNodePostgresConnection"):
        self.connection = connection
        self.row: Any = None
        self.rowcount = 0

    def __enter__(self) -> "AtomicNodePostgresCursor":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        statement = " ".join(str(sql).split())
        self.row = None
        self.rowcount = 0
        with self.connection.lock:
            if statement.startswith("INSERT INTO merchant_ai_node_task_state"):
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
                ) = params
                key = (str(run_id), str(task_id))
                existing = self.connection.entries.get(key)
                if existing is None:
                    self.connection.entries[key] = {
                        "run_id": run_id,
                        "task_id": task_id,
                        "status": status,
                        "idempotency_key": idempotency_key,
                        "request_fingerprint": request_fingerprint,
                        "attempts": attempts,
                        "lease_owner": lease_owner,
                        "lease_generation": lease_generation,
                        "lease_until": lease_until,
                        "updated_at": updated_at,
                        "payload": json.loads(payload),
                    }
                    self.row = self.connection.row(key)
                    self.rowcount = 1
                elif existing["request_fingerprint"] == request_fingerprint and request_fingerprint:
                    self.row = self.connection.row(key)
                    self.rowcount = 1
                return
            if statement.startswith("SELECT run_id, task_id"):
                key = (str(params[0]), str(params[1]))
                if key in self.connection.entries:
                    self.row = self.connection.row(key)
                return
            if "SET status='running', attempts=attempts+1" in statement:
                lease_owner, lease_until, updated_at, run_id, task_id = params
                key = (str(run_id), str(task_id))
                existing = self.connection.entries.get(key)
                if existing and existing["status"] in {"queued", "pending", "retry"}:
                    existing["status"] = "running"
                    existing["attempts"] += 1
                    existing["lease_owner"] = lease_owner
                    existing["lease_generation"] += 1
                    existing["lease_until"] = lease_until
                    existing["updated_at"] = updated_at
                    self.row = self.connection.row(key)
                    self.rowcount = 1

    def fetchone(self) -> Any:
        return self.row


class AtomicNodePostgresConnection:
    def __init__(self):
        self.entries: dict[tuple[str, str], dict[str, Any]] = {}
        self.lock = threading.RLock()

    def cursor(self) -> AtomicNodePostgresCursor:
        return AtomicNodePostgresCursor(self)

    def commit(self) -> None:
        return None

    def row(self, key: tuple[str, str]) -> tuple[Any, ...]:
        entry = self.entries[key]
        return (
            entry["run_id"],
            entry["task_id"],
            entry["status"],
            entry["idempotency_key"],
            entry["request_fingerprint"],
            entry["attempts"],
            entry["lease_owner"],
            entry["lease_generation"],
            entry["lease_until"],
            entry["updated_at"],
            entry["payload"],
        )


def _postgres_node_store(connection: AtomicNodePostgresConnection) -> PostgresRuntimeStateStore:
    store = PostgresRuntimeStateStore.__new__(PostgresRuntimeStateStore)
    store.conn = connection
    return store


def test_postgres_enqueue_sql_is_create_or_get_and_cannot_roll_back_claim():
    connection = AtomicNodePostgresConnection()
    store = _postgres_node_store(connection)
    store.enqueue_node_task(
        NodeTaskState(run_id="run", task_id="task", request_fingerprint="request-a", payload={"input": 1})
    )
    start = threading.Barrier(3)
    results: list[Any] = []

    def enqueue() -> None:
        start.wait()
        results.append(
            store.enqueue_node_task(
                NodeTaskState(
                    run_id="run",
                    task_id="task",
                    request_fingerprint="request-a",
                    payload={"input": 999},
                )
            )
        )

    def claim() -> None:
        start.wait()
        results.append(store.claim_node_task("run", "task", "worker", lease_seconds=30))

    threads = [threading.Thread(target=enqueue), threading.Thread(target=claim)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    persisted = store.get_node_task("run", "task")
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.attempts == 1
    assert persisted.payload == {"input": 1}
    with pytest.raises(NodeTaskRequestConflict):
        store.enqueue_node_task(
            NodeTaskState(run_id="run", task_id="task", request_fingerprint="request-b")
        )


def test_two_tool_runtimes_share_the_selected_file_execution_journal(
    tmp_path: Path,
):
    settings = Settings(
        harness_workspace_path=str(tmp_path),
        runtime_state_backend="file",
        cache_enabled=False,
        tool_rate_limit_enabled=False,
    )
    registry = ToolRegistry(
        [
            ToolCapability(
                name="verified_read",
                side_effect_level="read",
                cache_policy="disabled",
                fail_closed=True,
            )
        ]
    )
    first_runtime = ToolRuntimeService(settings, tool_registry=registry)
    second_runtime = ToolRuntimeService(settings, tool_registry=registry)
    calls = {"count": 0}

    def handler(_args: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {"value": calls["count"]}

    first = first_runtime.execute(
        "verified_read",
        {"input": 1},
        handler,
        call_id="shared-call",
    )
    replay = second_runtime.execute(
        "verified_read",
        {"input": 1},
        handler,
        call_id="shared-call",
    )

    assert first.status == replay.status == "success"
    assert first.result == replay.result == {"value": 1}
    assert replay.attempts == 0
    assert calls["count"] == 1
