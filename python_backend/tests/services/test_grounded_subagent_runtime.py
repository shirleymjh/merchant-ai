from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage

from merchant_ai.services.grounded_subagent_runtime import (
    GroundedSubagentBudget,
    GroundedSubagentCapabilityMiddleware,
    GroundedSubagentEvidenceRequirement,
    GroundedSubagentGoalContract,
    IsolatedSubagentJob,
    IsolatedSubagentResult,
    PreparedIsolatedSubagentTask,
    dispatch_prepared_subagent_tasks,
    issue_grounded_subagent_capability_grant,
)


def _goal(sub_goal_id: str, *, generation: int = 1) -> GroundedSubagentGoalContract:
    return GroundedSubagentGoalContract(
        sub_goal_id=sub_goal_id,
        parent_goal_ids=["goal.metric"],
        objective="Inspect one bounded evidence question.",
        required_outputs=["finding"],
        input_artifact_refs=[],
        evidence_requirements=[
            GroundedSubagentEvidenceRequirement(
                requirement_id="requirement.refs",
                description="Return exact evidence refs.",
                accepted_ref_types=["SEMANTIC_REF"],
            )
        ],
        allowed_capabilities=["READ_CONTEXT"],
        budget=GroundedSubagentBudget(
            max_tool_calls=1,
            timeout_seconds=12,
        ),
        generation=generation,
    )


def _raw_output(finding: str) -> str:
    return json.dumps(
        {
            "summary": finding,
            "finding": finding,
            "evidenceRefs": [finding],
            "gaps": [],
            "recommendedNextAction": "ROOT_REVIEW",
            "proposedSubGoals": [],
            "evidenceGaps": [],
        }
    )


def test_task_capability_grant_is_bound_to_goal_contract_and_fails_closed() -> None:
    goal = _goal("subgoal.capability")
    grant = issue_grounded_subagent_capability_grant(
        goal,
        allowed_tool_names=["read_file"],
    )

    assert grant.fingerprint_valid()
    assert grant.sub_goal_id == goal.sub_goal_id
    assert grant.parent_goal_ids == ["goal.metric"]
    assert grant.generation == 1
    assert grant.goal_contract_fingerprint == goal.contract_fingerprint()

    middleware = GroundedSubagentCapabilityMiddleware(grant)
    handled = {"count": 0}

    def handler(request: Any) -> ToolMessage:
        handled["count"] += 1
        call = request.tool_call
        return ToolMessage(
            content="ok",
            name=call["name"],
            tool_call_id=call["id"],
        )

    denied = middleware.wrap_tool_call(
        SimpleNamespace(
            tool_call={"id": "call-denied", "name": "execute_assigned_query"}
        ),
        handler,
    )
    first = middleware.wrap_tool_call(
        SimpleNamespace(tool_call={"id": "call-1", "name": "read_file"}),
        handler,
    )
    exhausted = middleware.wrap_tool_call(
        SimpleNamespace(tool_call={"id": "call-2", "name": "read_file"}),
        handler,
    )

    assert denied.status == "error"
    assert json.loads(str(denied.content))["code"] == "SUBAGENT_TOOL_NOT_GRANTED"
    assert first.status != "error"
    assert exhausted.status == "error"
    assert json.loads(str(exhausted.content))["code"] == (
        "SUBAGENT_TOOL_CALL_BUDGET_EXHAUSTED"
    )
    assert handled["count"] == 1


def test_parallel_dispatch_preserves_contract_order_and_runs_independent_workers() -> None:
    barrier = threading.Barrier(2)
    entered: list[str] = []
    lock = threading.Lock()
    prepared: list[PreparedIsolatedSubagentTask] = []

    for sub_goal_id in ("subgoal.parallel.a", "subgoal.parallel.b"):
        goal = _goal(sub_goal_id)
        grant = issue_grounded_subagent_capability_grant(
            goal,
            allowed_tool_names=["read_file"],
        )
        job = IsolatedSubagentJob(
            job_id=sub_goal_id,
            thread_id="thread.%s" % sub_goal_id,
            system_prompt="bounded",
            user_payload={},
            backend=None,
            capability_grant=grant,
        )

        def runner(
            current_job: IsolatedSubagentJob,
            *,
            observed_goal: GroundedSubagentGoalContract = goal,
        ) -> IsolatedSubagentResult:
            with lock:
                entered.append(observed_goal.sub_goal_id)
            barrier.wait(timeout=2)
            return IsolatedSubagentResult(
                job_id=current_job.job_id,
                thread_id=current_job.thread_id,
                checkpoint={},
                raw_output=_raw_output(observed_goal.sub_goal_id),
                update_count=1,
            )

        prepared.append(
            PreparedIsolatedSubagentTask(
                task=goal,
                grant=grant,
                job=job,
                runner=runner,
            )
        )

    outcomes = dispatch_prepared_subagent_tasks(
        prepared,
        parallel=True,
        max_workers=2,
    )

    assert set(entered) == {"subgoal.parallel.a", "subgoal.parallel.b"}
    assert [item.sub_goal_id for item in outcomes] == [
        "subgoal.parallel.a",
        "subgoal.parallel.b",
    ]
    assert [item.status for item in outcomes] == ["COMPLETED", "COMPLETED"]


def test_worker_cannot_turn_proposed_subgoal_into_executable_authority() -> None:
    goal = _goal("subgoal.proposal")
    grant = issue_grounded_subagent_capability_grant(
        goal,
        allowed_tool_names=["read_file"],
    )
    job = IsolatedSubagentJob(
        job_id="proposal-job",
        thread_id="proposal-thread",
        system_prompt="bounded",
        user_payload={},
        backend=None,
        capability_grant=grant,
    )

    def runner(current_job: IsolatedSubagentJob) -> IsolatedSubagentResult:
        payload = json.loads(_raw_output("proposal"))
        payload["proposedSubGoals"] = [
            {
                "objective": "Try another check",
                "generation": 2,
                "allowedCapabilities": ["QUERY_BRANCH"],
            }
        ]
        return IsolatedSubagentResult(
            job_id=current_job.job_id,
            thread_id=current_job.thread_id,
            checkpoint={},
            raw_output=json.dumps(payload),
            update_count=1,
        )

    outcome = dispatch_prepared_subagent_tasks(
        [
            PreparedIsolatedSubagentTask(
                task=goal,
                grant=grant,
                job=job,
                runner=runner,
            )
        ],
        parallel=False,
        max_workers=1,
    )[0]

    assert outcome.status == "FAILED"
    assert outcome.error == "SUBAGENT_OUTPUT_CONTRACT_REJECTED"
    assert "SUBAGENT_PROPOSED_SUB_GOAL_EXECUTABLE_FIELD_DENIED" in (
        outcome.validation_errors
    )
