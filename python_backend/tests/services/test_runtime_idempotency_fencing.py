from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.runtime_state import (
    MemoryRuntimeStateStore,
    NodeTaskState,
    PostgresRuntimeStateStore,
    RedisRuntimeStateStore,
    StaleExecutionFence,
    StaleNodeTaskFence,
)
from merchant_ai.services.distributed_workers import (
    DistributedArtifactStore,
    DistributedSubAgentClient,
    DistributedSubAgentWorker,
)
from merchant_ai.models import ToolCallRequest
from merchant_ai.services.tool_runtime import (
    ToolCallExecutor,
    ToolFailureRegistry,
    ToolRuntimePolicyRegistry,
    ToolRuntimeService,
    current_tool_execution_fence,
    normalize_tool_context,
    tool_idempotency_key,
)
from merchant_ai.services.tools import ToolCapability, ToolRegistry


def _custom_read_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolCapability(
                name="custom_tool",
                side_effect_level="read",
                cache_policy="disabled",
                fail_closed=True,
            )
        ]
    )


def test_old_worker_cannot_complete_after_lease_recovery_and_reclaim():
    store = MemoryRuntimeStateStore()
    store.enqueue_node_task(NodeTaskState(run_id="run", task_id="task"))
    first = store.claim_node_task("run", "task", "worker-a", lease_seconds=1)
    assert first is not None
    first.lease_until = (datetime.now() - timedelta(seconds=1)).isoformat()
    store.upsert_node_task(first)

    assert store.recover_expired_node_tasks(max_attempts=3) == 1
    second = store.claim_node_task("run", "task", "worker-b", lease_seconds=30)
    assert second is not None
    assert second.lease_generation == first.lease_generation + 1

    with pytest.raises(StaleNodeTaskFence):
        store.complete_node_task(
            "run",
            "task",
            "completed",
            {"worker": "a"},
            lease_owner=first.lease_owner,
            lease_generation=first.lease_generation,
        )

    completed = store.complete_node_task(
        "run",
        "task",
        "completed",
        {"worker": "b"},
        lease_owner=second.lease_owner,
        lease_generation=second.lease_generation,
    )
    assert completed.payload["worker"] == "b"


def test_controller_timeout_fences_late_worker_result():
    store = MemoryRuntimeStateStore()
    store.enqueue_node_task(NodeTaskState(run_id="run", task_id="task"))
    lease = store.claim_node_task("run", "task", "worker", lease_seconds=30)
    assert lease is not None

    timed_out = store.fence_node_task(
        "run",
        "task",
        "timeout",
        {"reason": "deadline"},
        expected_generation=lease.lease_generation,
    )
    assert timed_out.lease_generation == lease.lease_generation + 1

    with pytest.raises(StaleNodeTaskFence):
        store.complete_node_task(
            "run",
            "task",
            "completed",
            {"late": True},
            lease_owner=lease.lease_owner,
            lease_generation=lease.lease_generation,
        )
    assert store.get_node_task("run", "task").status == "timeout"


def test_stale_distributed_worker_cannot_publish_over_winning_artifact(
    tmp_path: Path,
):
    settings = Settings(
        harness_workspace_path=str(tmp_path),
        runtime_state_backend="memory",
        distributed_worker_execution_backend="inline",
        distributed_worker_lease_seconds=1,
    )
    store = MemoryRuntimeStateStore()
    artifacts = DistributedArtifactStore(settings)
    client = DistributedSubAgentClient(settings, store, artifacts)
    client.submit("run", "task", "analysis", {"input": 1})
    old_lease = store.claim_node_task("run", "task", "worker-old", 1)
    assert old_lease is not None
    old_entered = threading.Event()
    release_old = threading.Event()

    def old_handler(_request, _canceled):
        old_entered.set()
        release_old.wait(timeout=3)
        return {"answer": "old"}

    old_worker = DistributedSubAgentWorker(
        settings,
        handlers={"analysis": old_handler},
        state_store=store,
        artifact_store=artifacts,
        worker_id="worker-old",
    )
    old_thread = threading.Thread(
        target=old_worker._execute_claimed,
        args=(old_lease,),
    )
    old_thread.start()
    assert old_entered.wait(timeout=1)

    expired = store.get_node_task("run", "task")
    assert expired is not None
    expired.lease_until = (datetime.now() - timedelta(seconds=1)).isoformat()
    store.upsert_node_task(expired)
    assert store.recover_expired_node_tasks() == 1
    winning_lease = store.claim_node_task("run", "task", "worker-new", 30)
    assert winning_lease is not None
    winning_worker = DistributedSubAgentWorker(
        settings,
        handlers={"analysis": lambda _request, _canceled: {"answer": "new"}},
        state_store=store,
        artifact_store=artifacts,
        worker_id="worker-new",
    )
    winning_worker._execute_claimed(winning_lease)
    release_old.set()
    old_thread.join(timeout=3)

    final_state = store.get_node_task("run", "task")
    assert final_state is not None
    assert final_state.status == "completed"
    result_uri = str(final_state.payload["resultArtifactUri"])
    assert "result_g%s_" % winning_lease.lease_generation in result_uri
    assert artifacts.read_json(result_uri)["payload"]["answer"] == "new"
    result_files = list(Path(result_uri).parent.glob("result_g*.json"))
    assert result_files == [Path(result_uri)]


def test_concurrent_duplicate_tool_call_executes_handler_once_and_replays_result():
    settings = Settings(
        cache_enabled=False,
        tool_rate_limit_enabled=False,
        agent_node_timeout_seconds=2,
        runtime_state_backend="memory",
    )
    journal = MemoryRuntimeStateStore()
    runtime = ToolRuntimeService(
        settings,
        execution_journal=journal,
        tool_registry=_custom_read_registry(),
    )
    entered = threading.Event()
    release = threading.Event()
    calls = {"count": 0}
    first_results = []

    def handler(_args):
        calls["count"] += 1
        entered.set()
        release.wait(timeout=1)
        return {"value": 7, "fence": current_tool_execution_fence()}

    thread = threading.Thread(
        target=lambda: first_results.append(runtime.execute("custom_tool", {"value": 7}, handler, call_id="call"))
    )
    thread.start()
    assert entered.wait(timeout=1)
    concurrent = runtime.execute("custom_tool", {"value": 7}, handler, call_id="call")
    release.set()
    thread.join(timeout=2)
    replayed = runtime.execute("custom_tool", {"value": 7}, handler, call_id="call")

    assert calls["count"] == 1
    assert concurrent.status == "blocked"
    assert concurrent.error_type == "IDEMPOTENCY_IN_PROGRESS"
    assert first_results[0].status == "success"
    assert replayed.status == "success"
    assert replayed.attempts == 0
    assert replayed.result["fence"]["fencingGeneration"] == 1


def test_timed_out_tool_call_is_not_automatically_executed_again():
    settings = Settings(
        cache_enabled=False,
        tool_rate_limit_enabled=False,
        agent_node_timeout_seconds=1,
        runtime_state_backend="memory",
    )
    journal = MemoryRuntimeStateStore()
    runtime = ToolRuntimeService(
        settings,
        execution_journal=journal,
        tool_registry=_custom_read_registry(),
    )
    calls = {"count": 0}

    def handler(_args):
        calls["count"] += 1
        time.sleep(1.2)
        return {"value": 1}

    first = runtime.execute("custom_tool", {}, handler, call_id="same-call")
    replayed = runtime.execute("custom_tool", {}, handler, call_id="same-call")
    time.sleep(0.25)

    assert first.error_type == "TIMEOUT"
    assert replayed.error_type == "TIMEOUT"
    assert replayed.attempts == 0
    assert calls["count"] == 1
    assert journal.get_execution(first.idempotency_key).status == "timed_out"


def test_authorization_scope_is_bound_into_the_idempotency_key():
    first = normalize_tool_context(
        "custom_tool",
        {"value": 1, "_scope": {"role": "analyst", "storeIds": ["one"]}},
    )
    second = normalize_tool_context(
        "custom_tool",
        {"value": 1, "_scope": {"role": "viewer", "storeIds": ["two"]}},
    )

    assert first["paramsHash"] != second["paramsHash"]
    assert tool_idempotency_key("custom_tool", "call", first) != tool_idempotency_key(
        "custom_tool", "call", second
    )


def test_unregistered_tool_does_not_default_to_read_semantics():
    capability = ToolRegistry().capability("send_external_command")

    assert capability.side_effect_level == "unknown"
    assert capability.cache_policy == "disabled"
    assert capability.fail_closed is True


def test_unregistered_tool_is_blocked_before_handler_execution():
    settings = Settings(
        cache_enabled=False,
        tool_rate_limit_enabled=False,
        runtime_state_backend="memory",
    )
    called = {"value": False}

    def handler(_args):
        called["value"] = True
        return {"ok": True}

    result = ToolRuntimeService(settings).execute(
        "send_external_command",
        {},
        handler,
        call_id="explicit-call",
    )

    assert result.status == "blocked"
    assert result.error_type == "UNCLASSIFIED_TOOL_CAPABILITY"
    assert called["value"] is False


def test_batch_tool_executor_replays_committed_call_instead_of_invoking_handler_again():
    settings = Settings(agent_node_timeout_seconds=2)
    journal = MemoryRuntimeStateStore()
    executor = ToolCallExecutor(
        ToolRuntimePolicyRegistry(settings),
        ToolFailureRegistry(),
        max_concurrency=1,
        execution_journal=journal,
        tool_registry=_custom_read_registry(),
    )
    calls = {"count": 0}

    def handler(_args):
        calls["count"] += 1
        return {"value": calls["count"]}

    request = ToolCallRequest(id="same", name="custom_tool", args={"input": 1})
    first = executor.execute([request], {"custom_tool": handler})[0]
    replayed = executor.execute([request], {"custom_tool": handler})[0]

    assert first.status == replayed.status == "success"
    assert first.result == replayed.result == {"value": 1}
    assert replayed.attempts == 0
    assert calls["count"] == 1


def test_expired_side_effecting_execution_requires_review_instead_of_takeover():
    store = MemoryRuntimeStateStore()
    first = store.claim_execution("operation", "worker-a", lease_seconds=1)
    store._executions["operation"].lease_until = (datetime.now() - timedelta(seconds=1)).isoformat()

    blocked = store.claim_execution(
        "operation",
        "worker-b",
        lease_seconds=30,
        allow_expired_takeover=False,
    )
    assert blocked.outcome == "expired_unsafe"

    takeover = store.claim_execution(
        "operation",
        "worker-b",
        lease_seconds=30,
        allow_expired_takeover=True,
    )
    assert takeover.acquired
    assert takeover.entry.lease_generation == first.entry.lease_generation + 1
    with pytest.raises(StaleExecutionFence):
        store.complete_execution("operation", "worker-a", first.entry.lease_generation, "succeeded")


class AtomicRedisJournalClient:
    def __init__(self):
        self.hashes = {}
        self.lock = threading.RLock()

    def hgetall(self, key):
        with self.lock:
            return dict(self.hashes.get(key) or {})

    def hset(self, key, mapping):
        with self.lock:
            self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})

    def eval(self, script, _numkeys, *args):
        with self.lock:
            key = args[0]
            values = args[1:]
            if "redis.call('EXISTS', KEYS[1])" in script:
                idempotency_key, owner, lease_until, now, allow_takeover = values
                current = self.hashes.get(key)
                if current is None:
                    self.hashes[key] = {
                        "idempotency_key": idempotency_key,
                        "status": "in_progress",
                        "lease_owner": owner,
                        "lease_generation": "1",
                        "lease_until": lease_until,
                        "result_hash": "",
                        "created_at": now,
                        "updated_at": now,
                        "payload": "{}",
                    }
                    return ["acquired", "1"]
                generation = int(current["lease_generation"])
                if current["status"] != "in_progress":
                    return ["replay", str(generation)]
                if current["lease_until"] and current["lease_until"] <= now:
                    if allow_takeover != "1":
                        return ["expired_unsafe", str(generation)]
                    generation += 1
                    current.update(
                        {
                            "lease_owner": owner,
                            "lease_generation": str(generation),
                            "lease_until": lease_until,
                            "updated_at": now,
                        }
                    )
                    return ["acquired", str(generation)]
                return ["in_progress", str(generation)]
            owner, generation, status, payload, result_hash, now = values
            current = self.hashes.get(key) or {}
            if (
                current.get("status") != "in_progress"
                or current.get("lease_owner") != owner
                or int(current.get("lease_generation") or 0) != int(generation)
            ):
                return 0
            current.update(
                {
                    "status": status,
                    "payload": payload,
                    "result_hash": result_hash,
                    "lease_until": "",
                    "updated_at": now,
                }
            )
            return 1


def test_redis_execution_journal_claim_and_commit_are_atomic():
    store = RedisRuntimeStateStore.__new__(RedisRuntimeStateStore)
    store.namespace = "test"
    store.client = AtomicRedisJournalClient()

    first = store.claim_execution("operation", "worker-a", lease_seconds=30)
    duplicate = store.claim_execution("operation", "worker-b", lease_seconds=30)
    completed = store.complete_execution(
        "operation",
        "worker-a",
        first.entry.lease_generation,
        "succeeded",
        {"value": 1},
        "hash",
    )
    replay = store.claim_execution("operation", "worker-c", lease_seconds=30)

    assert first.acquired
    assert duplicate.outcome == "in_progress"
    assert completed.payload == {"value": 1}
    assert replay.outcome == "replay"


class PostgresJournalCursor:
    def __init__(self, connection):
        self.connection = connection
        self.row = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        statement = " ".join(str(sql).split())
        self.row = None
        self.rowcount = 0
        if statement.startswith("INSERT INTO merchant_ai_execution_journal"):
            key, owner, lease_until, created_at, updated_at = params
            if key not in self.connection.entries:
                self.connection.entries[key] = {
                    "idempotency_key": key,
                    "status": "in_progress",
                    "lease_owner": owner,
                    "lease_generation": 1,
                    "lease_until": lease_until,
                    "result_hash": "",
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "payload": {},
                }
                self.row = self.connection.row(key)
                self.rowcount = 1
            return
        if statement.startswith("SELECT idempotency_key"):
            key = params[0]
            if key in self.connection.entries:
                self.row = self.connection.row(key)
            return
        if "SET lease_owner=%s, lease_generation=lease_generation+1" in statement:
            owner, lease_until, now, key, generation, expiry = params
            entry = self.connection.entries.get(key)
            if (
                entry
                and entry["status"] == "in_progress"
                and entry["lease_generation"] == generation
                and entry["lease_until"]
                and entry["lease_until"] <= expiry
            ):
                entry.update(
                    {
                        "lease_owner": owner,
                        "lease_generation": generation + 1,
                        "lease_until": lease_until,
                        "updated_at": now,
                    }
                )
                self.row = self.connection.row(key)
                self.rowcount = 1
            return
        if statement.startswith("UPDATE merchant_ai_execution_journal"):
            status, payload, result_hash, now, key, owner, generation = params
            entry = self.connection.entries.get(key)
            if (
                entry
                and entry["status"] == "in_progress"
                and entry["lease_owner"] == owner
                and entry["lease_generation"] == generation
            ):
                entry.update(
                    {
                        "status": status,
                        "payload": json.loads(payload),
                        "result_hash": result_hash,
                        "lease_until": "",
                        "updated_at": now,
                    }
                )
                self.row = self.connection.row(key)
                self.rowcount = 1

    def fetchone(self):
        return self.row


class PostgresJournalConnection:
    def __init__(self):
        self.entries = {}

    def cursor(self):
        return PostgresJournalCursor(self)

    def commit(self):
        return None

    def row(self, key):
        entry = self.entries[key]
        return (
            entry["idempotency_key"],
            entry["status"],
            entry["lease_owner"],
            entry["lease_generation"],
            entry["lease_until"],
            entry["result_hash"],
            entry["created_at"],
            entry["updated_at"],
            entry["payload"],
        )


def test_postgres_execution_journal_uses_generation_cas():
    store = PostgresRuntimeStateStore.__new__(PostgresRuntimeStateStore)
    store.conn = PostgresJournalConnection()

    first = store.claim_execution("operation", "worker-a", lease_seconds=30)
    duplicate = store.claim_execution("operation", "worker-b", lease_seconds=30)
    store.complete_execution(
        "operation",
        "worker-a",
        first.entry.lease_generation,
        "succeeded",
        {"value": 1},
    )
    replay = store.claim_execution("operation", "worker-c", lease_seconds=30)

    assert first.acquired
    assert duplicate.outcome == "in_progress"
    assert replay.outcome == "replay"
    with pytest.raises(StaleExecutionFence):
        store.complete_execution("operation", "worker-b", first.entry.lease_generation + 1, "failed")
