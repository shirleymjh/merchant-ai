from __future__ import annotations

import threading
import time
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    MerchantInfo,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.distributed_workers import (
    DistributedArtifactStore,
    DistributedSubAgentClient,
    DistributedSubAgentWorker,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService
from merchant_ai.services.runtime_state import FileRuntimeStateStore, NodeTaskState
from merchant_ai.services.skill_worker import SkillWorkerExecutor


def distributed_settings(tmp_path: Path, **updates):
    return get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "runtime_state_backend": "file",
            "distributed_artifact_backend": "filesystem",
            "distributed_worker_execution_backend": "inline",
            "distributed_worker_poll_seconds": 0.01,
            "distributed_worker_lease_seconds": 2,
            "distributed_worker_result_timeout_seconds": 3,
            **updates,
        }
    )


def test_file_runtime_store_claims_next_task_across_runs(tmp_path):
    settings = distributed_settings(tmp_path)
    store = FileRuntimeStateStore(settings)
    store.enqueue_node_task(NodeTaskState("run_b", "task_b", status="queued", payload={"taskKind": "document_analysis"}))
    time.sleep(0.01)
    store.enqueue_node_task(NodeTaskState("run_a", "task_a", status="queued", payload={"taskKind": "query_node"}))

    claimed = store.claim_next_node_task("worker-1", lease_seconds=10, task_kinds=["document_analysis"])

    assert claimed is not None
    assert (claimed.run_id, claimed.task_id) == ("run_b", "task_b")
    assert claimed.status == "running"
    assert claimed.lease_owner == "worker-1"


def test_distributed_worker_round_trip_persists_result_artifact(tmp_path):
    settings = distributed_settings(tmp_path)
    store = FileRuntimeStateStore(settings)
    artifacts = DistributedArtifactStore(settings)
    client = DistributedSubAgentClient(settings, store, artifacts)
    worker = DistributedSubAgentWorker(
        settings,
        handlers={"document_analysis": lambda request, canceled: {"answer": request["content"].upper()}},
        state_store=store,
        artifact_store=artifacts,
        worker_id="worker-round-trip",
    )
    client.submit("run_1", "task_1", "document_analysis", {"content": "merchant insight"})

    assert worker.run_once()
    result = client.wait("run_1", "task_1")

    assert result.status == "completed"
    assert result.result == {"answer": "MERCHANT INSIGHT"}
    assert result.artifact_uri
    assert Path(result.artifact_uri).exists()


def test_distributed_client_waits_for_independent_worker_loop(tmp_path):
    settings = distributed_settings(tmp_path)
    store = FileRuntimeStateStore(settings)
    artifacts = DistributedArtifactStore(settings)
    client = DistributedSubAgentClient(settings, store, artifacts)
    worker = DistributedSubAgentWorker(
        settings,
        handlers={"hypothesis_review": lambda request, canceled: {"survivor": request["candidate"]}},
        state_store=store,
        artifact_store=artifacts,
        worker_id="worker-loop",
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    try:
        result = client.execute("run_2", "task_2", "hypothesis_review", {"candidate": "orders"})
    finally:
        worker.stop()
        thread.join(timeout=2)

    assert result.status == "completed"
    assert result.result["survivor"] == "orders"


def test_process_worker_terminates_timed_out_task(tmp_path):
    settings = distributed_settings(
        tmp_path,
        distributed_worker_execution_backend="process",
        distributed_worker_result_timeout_seconds=1,
    )
    store = FileRuntimeStateStore(settings)
    artifacts = DistributedArtifactStore(settings)
    client = DistributedSubAgentClient(settings, store, artifacts)

    def slow_handler(request, canceled):
        time.sleep(5)
        return {"late": True}

    worker = DistributedSubAgentWorker(
        settings,
        handlers={"python_batch": slow_handler},
        state_store=store,
        artifact_store=artifacts,
        worker_id="worker-timeout",
    )
    client.submit("run_timeout", "task_timeout", "python_batch", {}, timeout_seconds=1)
    started = time.monotonic()

    assert worker.run_once()
    result = client.wait("run_timeout", "task_timeout")

    assert time.monotonic() - started < 3
    assert result.status == "timeout"
    assert "terminated" in result.error


def test_node_worker_dispatches_subagent_to_durable_worker(tmp_path):
    settings = distributed_settings(
        tmp_path,
        distributed_subagents_enabled=True,
        agent_node_timeout_seconds=3,
        max_concurrent_sub_agents=1,
    )
    store = FileRuntimeStateStore(settings)
    artifacts = DistributedArtifactStore(settings)

    def query_handler(request, canceled):
        intent = QuestionIntent.model_validate(request["intent"])
        return AgentTaskResult(
            task_id=intent.plan_task_id,
            success=True,
            query_bundle=QueryBundle(tables=[intent.preferred_table], rows=[{"seller_id": "100", "value": 42}]),
        ).model_dump(by_alias=True)

    durable_worker = DistributedSubAgentWorker(
        settings,
        handlers={"query_node": query_handler},
        state_store=store,
        artifact_store=artifacts,
        worker_id="node-durable-worker",
    )
    worker_thread = threading.Thread(target=durable_worker.run_forever, daemon=True)
    worker_thread.start()
    try:
        node_worker = NodeWorkerExecutor(LlmClient(settings), object(), SqlValidationService(), settings)
        plan = QueryPlan(
            intents=[
                QuestionIntent(
                    question="查询指标",
                    intent_type="VALID",
                    answer_mode=AnswerMode.DETAIL,
                    plan_task_id="remote_node",
                    preferred_table="metric_table",
                    output_keys=["seller_id"],
                )
            ]
        )
        pack = PlanningAssetPack(tables=[PlanningAssetEntry(table="metric_table", columns=["seller_id", "value"])])
        result = node_worker.execute_plan("100", plan, pack, "", "查询指标", run_id="run_remote_node", execution_mode="subagent")
    finally:
        durable_worker.stop()
        worker_thread.join(timeout=2)

    assert result.task_results, result.model_dump(by_alias=True)
    assert result.task_results[0].success is True
    assert result.task_results[0].query_bundle.rows[0]["value"] == 42
    assert result.node_execution_batches[0].runtime_events[-1]["executionMode"] == "distributed_subagent"


def test_skill_worker_dispatches_to_durable_worker(tmp_path):
    settings = distributed_settings(
        tmp_path,
        distributed_subagents_enabled=True,
        skill_worker_timeout_seconds=3,
    )
    store = FileRuntimeStateStore(settings)
    artifacts = DistributedArtifactStore(settings)
    durable_worker = DistributedSubAgentWorker(
        settings,
        handlers={
            "analysis_skill": lambda request, canceled: {
                "answer": "distributed skill answer",
                "trace": {
                    "skillName": request["skillName"],
                    "lifecycleStage": "completed",
                    "confirmed": True,
                    "requiresConfirmation": False,
                },
            }
        },
        state_store=store,
        artifact_store=artifacts,
        worker_id="skill-durable-worker",
    )
    worker_thread = threading.Thread(target=durable_worker.run_forever, daemon=True)
    worker_thread.start()
    try:
        result = SkillWorkerExecutor(LlmClient(settings)).execute_answer_skill(
            "分析趋势",
            QueryPlan(),
            AgentRunResult(verified_evidence=VerifiedEvidence(passed=True)),
            str(tmp_path),
            skill_name="bi_trend_attribution",
            merchant=MerchantInfo(merchant_id="100"),
            initial_trace={"parentRunId": "run_remote_skill", "confirmed": True},
        )
    finally:
        durable_worker.stop()
        worker_thread.join(timeout=2)

    assert result.answer == "distributed skill answer"
    assert result.trace["workerType"] == "DISTRIBUTED_SKILL_WORKER"
    assert result.trace["resultArtifactUri"]
