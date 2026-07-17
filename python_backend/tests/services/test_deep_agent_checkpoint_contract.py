from __future__ import annotations

import json
import sqlite3

from merchant_ai.config import Settings
from merchant_ai.services.checkpoints import (
    DEEP_AGENT_CHECKPOINT_NAMESPACE,
    CheckpointManager,
    prune_completed_sqlite_checkpoints,
)


def test_deep_agent_checkpoint_uses_durable_conversation_scope_without_domain_collision(tmp_path):
    settings = Settings(
        harness_workspace_path=str(tmp_path),
        agent_checkpointer_backend="sqlite",
        agent_checkpointer_sqlite_path=str(tmp_path / "checkpoints" / "agent.sqlite"),
    )
    thread_id = "thread_" + "a" * 32
    first_run_id = "run_" + "1" * 32
    next_run_id = "run_" + "2" * 32

    manager = CheckpointManager(settings)
    try:
        saver = manager.saver()
        deep_config = manager.config_for_deep_agent(thread_id, first_run_id)
        next_deep_config = manager.config_for_deep_agent(thread_id, next_run_id)
        domain_config = manager.config_for_run(thread_id, first_run_id)
        next_domain_config = manager.config_for_run(thread_id, next_run_id)

        assert saver is manager.saver()
        # This verifies the public addressing contract. LangGraph root graphs
        # currently normalize non-empty checkpoint_ns values; physical namespace
        # nesting requires a subgraph. Collision safety here primarily comes from
        # the real conversation key versus the domain's run-scoped key.
        assert deep_config["configurable"] == {
            "thread_id": thread_id,
            "checkpoint_ns": DEEP_AGENT_CHECKPOINT_NAMESPACE,
        }
        assert domain_config["configurable"]["thread_id"] == "%s:%s" % (thread_id, first_run_id)
        assert domain_config["configurable"].get("checkpoint_ns", "") == ""
        assert deep_config["configurable"] != domain_config["configurable"]
        assert next_deep_config["configurable"] == deep_config["configurable"]
        assert next_deep_config["metadata"]["run_id"] == next_run_id
        assert next_domain_config["configurable"]["thread_id"] != domain_config["configurable"]["thread_id"]

        deep_ref = manager.deep_agent_ref(thread_id, next_run_id)
        assert deep_ref["checkpointThreadId"] == thread_id
        assert deep_ref["checkpointNamespace"] == DEEP_AGENT_CHECKPOINT_NAMESPACE
        assert deep_ref["resumable"] is True
        assert deep_ref["purpose"] == "deep_agent_conversation_checkpoint"
        assert deep_ref["storage"] == str(tmp_path / "checkpoints" / "agent.sqlite")

        domain_ref = manager.run_ref(thread_id, next_run_id)
        assert domain_ref["checkpointThreadId"] == "%s:%s" % (thread_id, next_run_id)
        assert domain_ref["checkpointNamespace"] == ""
        assert domain_ref["resumable"] is False
    finally:
        manager.close()


def test_deep_agent_memory_checkpoint_ref_is_not_durable(tmp_path):
    settings = Settings(
        harness_workspace_path=str(tmp_path),
        agent_checkpointer_backend="memory",
    )
    manager = CheckpointManager(settings)

    ref = manager.deep_agent_ref("thread_" + "b" * 32, "run_" + "3" * 32)

    assert ref["backend"] == "memory"
    assert ref["checkpointNamespace"] == DEEP_AGENT_CHECKPOINT_NAMESPACE
    assert ref["resumable"] is False


def test_pruning_retains_domain_and_deep_agent_conversation_checkpoints(tmp_path):
    checkpoint_path = tmp_path / "checkpoints" / "agent.sqlite"
    checkpoint_path.parent.mkdir(parents=True)
    settings = Settings(
        harness_workspace_path=str(tmp_path),
        agent_checkpointer_backend="sqlite",
        agent_checkpointer_sqlite_path=str(checkpoint_path),
        agent_completed_checkpoint_limit=1,
    )
    runs_dir = tmp_path / "run_events" / "runs"
    runs_dir.mkdir(parents=True)
    run_records = [
        {
            "threadId": "thread_active",
            "runId": "run_active",
            "status": "RUNNING",
            "updatedAt": "2026-01-01T00:00:00Z",
        },
        {
            "threadId": "thread_recent",
            "runId": "run_recent",
            "status": "COMPLETED",
            "updatedAt": "2026-03-01T00:00:00Z",
        },
        {
            "threadId": "thread_old",
            "runId": "run_old",
            "status": "COMPLETED",
            "updatedAt": "2026-02-01T00:00:00Z",
        },
    ]
    for record in run_records:
        (runs_dir / (record["runId"] + ".json")).write_text(json.dumps(record), encoding="utf-8")

    checkpoint_ids = {
        "thread_active",
        "thread_active:run_active",
        "thread_recent",
        "thread_recent:run_recent",
        "thread_old",
        "thread_old:run_old",
        "thread_orphan",
    }
    connection = sqlite3.connect(str(checkpoint_path))
    try:
        connection.execute("CREATE TABLE checkpoints (thread_id TEXT NOT NULL)")
        connection.execute("CREATE TABLE writes (thread_id TEXT NOT NULL)")
        connection.executemany("INSERT INTO checkpoints(thread_id) VALUES (?)", [(item,) for item in checkpoint_ids])
        connection.executemany("INSERT INTO writes(thread_id) VALUES (?)", [(item,) for item in checkpoint_ids])
        connection.commit()
    finally:
        connection.close()

    removed = prune_completed_sqlite_checkpoints(settings)

    assert removed == 3
    connection = sqlite3.connect(str(checkpoint_path))
    try:
        remaining_checkpoints = {
            str(row[0]) for row in connection.execute("SELECT thread_id FROM checkpoints").fetchall()
        }
        remaining_writes = {str(row[0]) for row in connection.execute("SELECT thread_id FROM writes").fetchall()}
    finally:
        connection.close()
    expected = {
        "thread_active",
        "thread_active:run_active",
        "thread_recent",
        "thread_recent:run_recent",
    }
    assert remaining_checkpoints == expected
    assert remaining_writes == expected
