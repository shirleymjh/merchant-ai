from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.graph.query_graph_contract import (
    graph_validation_attempted,
    graph_validation_passed,
    record_graph_validation,
)
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.graph.workflow import archive_execution_attempt
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
