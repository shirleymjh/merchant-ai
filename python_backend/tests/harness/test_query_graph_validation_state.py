from __future__ import annotations

from types import SimpleNamespace

import merchant_ai.graph.workflow as workflow_module
from merchant_ai.config import get_settings
from merchant_ai.graph.query_graph_contract import (
    graph_validation_attempted,
    graph_validation_passed,
    record_graph_validation,
)
from merchant_ai.graph.workflow import (
    append_active_planner_degraded_reason,
    archive_execution_attempt,
    create_workflow,
)
from merchant_ai.models import (
    AgentDecision,
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.middleware import ActionContractMiddleware, MiddlewareChain


def _executable_plan(metric_name: str = "metric_alpha") -> QueryPlan:
    return QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="task_alpha",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                metric_name=metric_name,
            )
        ]
    )


def test_failed_validation_is_attempted_but_never_passed():
    state = {"plan": _executable_plan()}

    record_graph_validation(state, GraphValidationResult(valid=False, repairable=False))

    assert graph_validation_attempted(state)
    assert not graph_validation_passed(state)
    assert state["query_graph_validation_status"] == "failed"
    assert state["query_graph_validation_attempted"] is True
    assert state["query_graph_validation_passed"] is False
    assert state["query_graph_validated"] is False


def test_failed_validation_projects_gaps_to_evidence_and_no_execution_answer(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "thread", "run")
    state["plan"] = _executable_plan()

    record_graph_validation(
        state,
        GraphValidationResult(
            valid=False,
            gaps=[
                GraphValidationGap(
                    code="CONTRACT_FIELD_MISSING",
                    task_id="task_alpha",
                    evidence="contract_alpha",
                    reason="declared execution contract is incomplete",
                )
            ],
        ),
    )

    run_result = state["agent_run_result"]
    assert [gap.code for gap in run_result.evidence_gaps] == ["CONTRACT_FIELD_MISSING"]
    assert run_result.evidence_gaps[0].source == "query_graph_validator"
    assert run_result.verified_evidence.answer_guard_required is True
    assert [gap.code for gap in run_result.verified_evidence.blocking_gaps] == ["CONTRACT_FIELD_MISSING"]

    answer = workflow.answer_service.compose(
        "question",
        state["merchant"],
        state["plan"],
        run_result,
        "",
        allow_llm=False,
    )

    assert "完整性校验未通过" in answer
    assert "CONTRACT_FIELD_MISSING" in answer


def test_passed_validation_removes_only_validator_projected_gaps():
    state = {"plan": _executable_plan(), "agent_run_result": AgentRunResult()}
    failed = GraphValidationResult(
        valid=False,
        gaps=[GraphValidationGap(code="CONTRACT_FIELD_MISSING", reason="missing")],
    )
    record_graph_validation(state, failed)

    record_graph_validation(state, GraphValidationResult(valid=True))

    assert state["agent_run_result"].evidence_gaps == []
    assert state["agent_run_result"].verified_evidence.blocking_gaps == []
    assert state["agent_run_result"].partial_answer_reason == ""


def test_execution_attempt_audit_survives_replaceable_output_invalidation(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "thread", "run")
    state["plan"] = _executable_plan()
    failed_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="task_alpha",
                success=False,
                query_bundle=QueryBundle(
                    sql="SELECT metric_alpha FROM table_alpha",
                    failed=True,
                    error="execution failed",
                ),
            )
        ],
        merged_query_bundle=QueryBundle(failed=True, error="execution failed"),
    )
    archive_execution_attempt(state, failed_result, "query_graph_execution")
    state["agent_run_result"] = failed_result
    state["sql_generated"] = True

    workflow.invalidate_execution_outputs(state, "graph contract changed")

    assert state["agent_run_result"].task_results == []
    assert len(state["execution_attempt_artifacts"]) == 1
    artifact = state["agent_run_result"].execution_attempt_artifacts[0]
    assert artifact.failed is True
    assert artifact.task_results[0].query_bundle.sql == "SELECT metric_alpha FROM table_alpha"
    assert artifact.task_results[0].query_bundle.error == "execution failed"
    assert artifact.model_dump(by_alias=True)["taskResults"][0]["queryBundle"]["sql"] == "SELECT metric_alpha FROM table_alpha"


def test_execute_rejects_plan_changed_after_validation(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "thread", "run")
    state["plan"] = _executable_plan()
    record_graph_validation(state, GraphValidationResult(valid=True, repairable=False))
    state["plan"].intents[0].metric_name = "metric_beta"
    calls = []

    def forbidden_execute(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("NodeWorker must not run for a stale validation fingerprint")

    monkeypatch.setattr(workflow.node_worker, "execute_plan", forbidden_execute)

    workflow.execute_query_graph(state)

    assert calls == []
    assert state["agent_run_result"].partial_answer_reason == "QUERY_GRAPH_CHANGED_AFTER_VALIDATION"
    assert state["query_bundle"].failed
    assert state["query_graph_validation_status"] == "not_run"
    assert state["query_graph_validation_passed"] is False
    assert state["query_graph_validated"] is False


def test_validation_rejection_projects_planner_degraded_before_execution_archive(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "degraded_validation_thread", "degraded_validation_run")
    state["plan"] = _executable_plan()
    state["planner_degraded"] = {
        "active": True,
        "stage": "planner",
        "code": "PLANNER_LLM_TIMEOUT",
        "reason": "timeout: provider call exceeded 20 seconds",
        "fallbackUsed": True,
        "fallbackCoveragePassed": True,
    }
    record_graph_validation(state, GraphValidationResult(valid=True, repairable=False))
    state["plan"].intents[0].metric_name = "metric_changed_after_validation"
    archived = []
    original_archive = workflow_module.archive_execution_attempt

    def observe_archive(observed_state, run_result, phase):
        archived.append((phase, list(run_result.degraded_reasons)))
        return original_archive(observed_state, run_result, phase)

    monkeypatch.setattr(workflow_module, "archive_execution_attempt", observe_archive)

    workflow.execute_query_graph(state)

    assert archived[0][0] == "query_graph_validation_rejection"
    assert archived[0][1] == [state["planner_degraded"]]
    assert state["agent_run_result"].degraded_reasons == [state["planner_degraded"]]


def test_execution_exception_projects_planner_degraded_before_execution_archive(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "degraded_exception_thread", "degraded_exception_run")
    state["plan"] = _executable_plan()
    state["planner_degraded"] = {
        "active": True,
        "stage": "planner",
        "code": "PLANNER_PROVIDER_ERROR",
        "reason": "provider_error: unavailable",
        "fallbackUsed": True,
        "fallbackCoveragePassed": True,
    }
    record_graph_validation(state, GraphValidationResult(valid=True, repairable=False))
    monkeypatch.setattr(
        workflow.node_worker,
        "prepare_runtime_execution_graph",
        lambda *_args, **_kwargs: SimpleNamespace(
            plan=state["plan"].model_copy(deep=True),
            validation=GraphValidationResult(valid=True, repairable=False),
        ),
    )
    monkeypatch.setattr(
        workflow.node_worker,
        "execute_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("node execution failed")),
    )
    archived = []
    original_archive = workflow_module.archive_execution_attempt

    def observe_archive(observed_state, run_result, phase):
        archived.append((phase, list(run_result.degraded_reasons)))
        return original_archive(observed_state, run_result, phase)

    monkeypatch.setattr(workflow_module, "archive_execution_attempt", observe_archive)

    workflow.execute_query_graph(state)

    assert archived[0][0] == "query_graph_execution"
    assert archived[0][1] == [state["planner_degraded"]]
    assert state["agent_run_result"].degraded_reasons == [state["planner_degraded"]]
    assert state["agent_run_result"].merged_query_bundle.error == "node execution failed"


def test_successful_execution_deduplicates_planner_degraded_before_archive(tmp_path, monkeypatch):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("question", "merchant", None, None, "degraded_success_thread", "degraded_success_run")
    state["plan"] = _executable_plan()
    state["planner_degraded"] = {
        "active": True,
        "stage": "planner",
        "code": "PLANNER_RESPONSE_INVALID",
        "reason": "json_parse_error: invalid response",
        "fallbackUsed": True,
        "fallbackCoveragePassed": True,
    }
    record_graph_validation(state, GraphValidationResult(valid=True, repairable=False))
    monkeypatch.setattr(
        workflow.node_worker,
        "prepare_runtime_execution_graph",
        lambda *_args, **_kwargs: SimpleNamespace(
            plan=state["plan"].model_copy(deep=True),
            validation=GraphValidationResult(valid=True, repairable=False),
        ),
    )

    def successful_execution(*_args, **_kwargs):
        return AgentRunResult(
            executed_query_graph_fingerprint=state["validated_query_graph_fingerprint"],
            merged_query_bundle=QueryBundle(rows=[{"metric_alpha": 1}]),
            degraded_reasons=[dict(state["planner_degraded"])],
        )

    monkeypatch.setattr(workflow.node_worker, "execute_plan", successful_execution)
    archived = []
    original_archive = workflow_module.archive_execution_attempt

    def observe_archive(observed_state, run_result, phase):
        archived.append((phase, list(run_result.degraded_reasons)))
        return original_archive(observed_state, run_result, phase)

    monkeypatch.setattr(workflow_module, "archive_execution_attempt", observe_archive)

    workflow.execute_query_graph(state)

    assert archived == [("query_graph_execution", [state["planner_degraded"]])]
    assert state["agent_run_result"].degraded_reasons == [state["planner_degraded"]]


def test_planner_degraded_projection_is_active_only_and_deduplicated():
    degraded = {
        "active": True,
        "stage": "planner",
        "code": "PLANNER_LLM_TIMEOUT",
        "reason": "timeout: provider call exceeded 20 seconds",
    }
    run_result = AgentRunResult(degraded_reasons=[dict(degraded)])

    append_active_planner_degraded_reason({"planner_degraded": degraded}, run_result)
    append_active_planner_degraded_reason({"planner_degraded": {"active": False}}, run_result)

    assert run_result.degraded_reasons == [degraded]


def test_action_contract_records_success_and_no_progress_outcomes():
    chain = MiddlewareChain([ActionContractMiddleware()])
    state = {"topic_routed": False, "middleware_events": [], "action_history": [], "action_outcomes": []}
    route = AgentDecision(selected_action="route_topic", selected_node="route_topic")
    chain.capture_action(state, route)
    state["topic_routed"] = True

    chain.after_action(state)

    assert state["last_action_result"].status == "success"
    assert state["action_outcomes"][-1]["status"] == "success"

    repeated = AgentDecision(selected_action="route_topic", selected_node="route_topic")
    chain.capture_action(state, repeated)
    chain.after_action(state)

    assert state["last_action_result"].status == "no_progress"
    assert state["action_outcomes"][-1]["status"] == "no_progress"


def test_action_contract_records_failed_when_postcondition_is_missing():
    chain = MiddlewareChain([ActionContractMiddleware()])
    state = {"topic_routed": False, "middleware_events": [], "action_history": [], "action_outcomes": []}
    decision = AgentDecision(selected_action="route_topic", selected_node="route_topic")
    chain.capture_action(state, decision)

    chain.after_action(state)

    assert state["last_action_result"].status == "failed"
    assert state["action_outcomes"][-1]["missingStateFlags"] == ["topic_routed"]
