import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    TaskRole,
)
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService


class UnconfiguredLlm:
    configured = False


class NeverCalledRepository:
    def __init__(self):
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("Doris executor must not be called for a blocked DAG")


def intent(task_id: str, dependencies: list[str], role: TaskRole = TaskRole.DEPENDENT) -> QuestionIntent:
    return QuestionIntent(
        question=task_id,
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.DETAIL,
        plan_task_id=task_id,
        task_role=role,
        preferred_table="generic_table",
        output_keys=["entity_id"],
        depends_on_task_ids=dependencies,
        sql="SELECT `entity_id` FROM `generic_table` LIMIT 1",
    )


def worker_for(tmp_path) -> tuple[NodeWorkerExecutor, NeverCalledRepository]:
    repository = NeverCalledRepository()
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "max_concurrent_sub_agents": 1,
        }
    )
    return NodeWorkerExecutor(UnconfiguredLlm(), repository, SqlValidationService(), settings), repository


def forbid_ready_batch(monkeypatch, worker: NodeWorkerExecutor):
    calls = {"count": 0}

    def fail_if_called(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("ready batch executor must not receive blocked DAG tasks")

    monkeypatch.setattr(worker, "_execute_ready_batch", fail_if_called)
    return calls


@pytest.mark.parametrize(
    ("dependencies", "expected_reason"),
    [
        ([], "has no dependency task ids"),
        (["missing_anchor"], "does not exist"),
    ],
)
def test_node_worker_fails_closed_for_unresolved_dependency(
    tmp_path,
    monkeypatch,
    dependencies,
    expected_reason,
):
    worker, repository = worker_for(tmp_path)
    executor_calls = forbid_ready_batch(monkeypatch, worker)

    result = worker.execute_plan(
        "merchant",
        QueryPlan(intents=[intent("blocked_task", dependencies)]),
        PlanningAssetPack(),
        "",
        "generic dependency test",
    )

    assert executor_calls["count"] == 0
    assert repository.calls == 0
    assert len(result.task_results) == 1
    assert result.task_results[0].success is False
    assert result.task_results[0].query_bundle.failed is True
    assert "UNRESOLVED_DEPENDENCY" in result.task_results[0].query_bundle.error
    gap = next(gap for gap in result.evidence_gaps if gap.code == "UNRESOLVED_DEPENDENCY")
    assert gap.task_id == "blocked_task"
    assert expected_reason in gap.reason
    batch = result.node_execution_batches[0]
    assert batch.submitted_task_ids == []
    assert batch.blocked_task_ids == ["blocked_task"]
    assert batch.failed_task_ids == ["blocked_task"]
    assert batch.runtime_events[0]["event"] == "node.dag_fail_closed"
    assert batch.runtime_events[0]["taskFailures"][0]["errorCode"] == "UNRESOLVED_DEPENDENCY"


def test_node_worker_marks_all_tasks_blocked_by_unresolved_dependency_chain(tmp_path, monkeypatch):
    worker, repository = worker_for(tmp_path)
    executor_calls = forbid_ready_batch(monkeypatch, worker)
    plan = QueryPlan(
        intents=[
            intent("missing_parent_task", ["missing_anchor"]),
            intent("downstream_task", ["missing_parent_task"]),
        ]
    )

    result = worker.execute_plan(
        "merchant",
        plan,
        PlanningAssetPack(),
        "",
        "generic dependency chain test",
    )

    assert executor_calls["count"] == 0
    assert repository.calls == 0
    assert {item.task_id for item in result.task_results} == {"missing_parent_task", "downstream_task"}
    assert all(not item.success and item.query_bundle.failed for item in result.task_results)
    gaps = [gap for gap in result.evidence_gaps if gap.code == "UNRESOLVED_DEPENDENCY"]
    assert {gap.task_id for gap in gaps} == {"missing_parent_task", "downstream_task"}
    assert set(result.node_execution_batches[0].blocked_task_ids) == {"missing_parent_task", "downstream_task"}


def test_node_worker_fails_closed_for_cyclic_graph_without_dispatching_tasks(tmp_path, monkeypatch):
    worker, repository = worker_for(tmp_path)
    executor_calls = forbid_ready_batch(monkeypatch, worker)
    plan = QueryPlan(
        intents=[
            intent("task_a", ["task_b"]),
            intent("task_b", ["task_a"]),
        ]
    )

    result = worker.execute_plan(
        "merchant",
        plan,
        PlanningAssetPack(),
        "",
        "generic cycle test",
    )

    assert executor_calls["count"] == 0
    assert repository.calls == 0
    assert {item.task_id for item in result.task_results} == {"task_a", "task_b"}
    assert all(not item.success and item.query_bundle.failed for item in result.task_results)
    assert all("CYCLIC_GRAPH" in item.query_bundle.error for item in result.task_results)
    gaps = [gap for gap in result.evidence_gaps if gap.code == "CYCLIC_GRAPH"]
    assert {gap.task_id for gap in gaps} == {"task_a", "task_b"}
    batch = result.node_execution_batches[0]
    assert set(batch.blocked_task_ids) == {"task_a", "task_b"}
    assert set(batch.failed_task_ids) == {"task_a", "task_b"}
    assert batch.submitted_task_ids == []
    assert batch.runtime_events[0]["event"] == "node.dag_fail_closed"
    assert {item["errorCode"] for item in batch.runtime_events[0]["taskFailures"]} == {"CYCLIC_GRAPH"}
