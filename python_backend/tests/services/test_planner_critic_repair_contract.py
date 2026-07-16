from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.graph.workflow import create_workflow, planner_repair_scope_key
from merchant_ai.models import (
    AgentDecision,
    AnswerMode,
    ChatContext,
    GraphValidationResult,
    IntentType,
    KnowledgeRequest,
    KnowledgeRequestType,
    PlannerReflectionResult,
    PlannerRepairRequest,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
)
from merchant_ai.services.middleware import ActionContractMiddleware, MiddlewareChain


def _plan(metric_name: str = "governed_metric") -> QueryPlan:
    return QueryPlan(
        intents=[
            QuestionIntent(
                question="检查受治理指标",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_task",
                preferred_table="governed_table",
                metric_name=metric_name,
            )
        ]
    )


def _reflection(issue_code: str = "DETAIL_EVIDENCE_NOT_PLANNED") -> PlannerReflectionResult:
    knowledge_request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="read the governed metric contract",
        needed_for_task_id="metric_task",
        reason="the repair needs the complete metric contract",
    )
    repair_request = PlannerRepairRequest(
        reason=issue_code,
        stage="planner_reflection",
        action="graph_repair",
        query="检查受治理指标",
        task_id="metric_task",
        evidence=json.dumps(
            {
                "code": issue_code,
                "severity": "error",
                "reason": "required row evidence is absent",
            },
            ensure_ascii=False,
        ),
        repair_hints=["preserve the requested evidence contract"],
        knowledge_requests=[knowledge_request],
        source="PlannerReflectionAgent",
    )
    return PlannerReflectionResult(
        passed=False,
        issues=[
            {
                "code": issue_code,
                "severity": "error",
                "taskId": "metric_task",
                "evidence": "required row evidence",
                "reason": "required row evidence is absent",
            }
        ],
        suggested_actions=["repair_graph"],
        repair_hints=["preserve the requested evidence contract"],
        repair_reason=issue_code,
        repair_requests=[repair_request],
    )


def _repair_state(tmp_path, *, rounds: int = 2):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "lead_action_llm_mode": "off",
            "agent_graph_repair_rounds": rounds,
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "检查受治理指标",
        "100",
        ChatContext(),
        None,
        "repair_contract_thread",
        "repair_contract_run",
    )
    state["plan"] = _plan()
    state["planning_asset_pack"] = PlanningAssetPack()
    state["planning_assets_compacted"] = True
    state["recall_bundle"] = RecallBundle()
    state["planner_reflection"] = _reflection()
    state["planner_repair_reason"] = state["planner_reflection"].repair_reason
    state["planner_repair_requests"] = list(state["planner_reflection"].repair_requests)
    state["query_graph_reflected"] = True
    state["query_graph_validation_result"] = GraphValidationResult()
    return workflow, state


def test_noop_repair_preserves_full_critic_input_and_projects_exhaustion(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=2)
    original_fingerprint = query_graph_fingerprint(state["plan"])
    captured_gaps = []

    def noop_repair(_question, plan, _pack, gaps, *_args):
        captured_gaps.extend(gaps)
        repaired = plan.model_copy(deep=True)
        repaired.agent_trace.append("planner.repair.unavailable")
        return repaired

    monkeypatch.setattr(workflow.planner, "repair", noop_repair)

    middleware = MiddlewareChain([ActionContractMiddleware()])
    decision = AgentDecision(
        selected_action="repair_graph",
        selected_node="repair_query_graph",
        available_actions=["repair_graph"],
    )
    middleware.capture_action(state, decision)

    workflow.repair_query_graph(state)
    middleware.after_action(state)

    repair_input = state["planner_repair_input"]
    first_delta = state["last_query_graph_repair_delta"]
    assert repair_input.repair_requests == _reflection().repair_requests
    assert any(
        json.loads(gap.evidence).get("knowledgeRequests", [])[0]["query"]
        == "read the governed metric contract"
        for gap in captured_gaps
        if gap.evidence.startswith("{") and "knowledgeRequests" in gap.evidence
    )
    assert first_delta.before_graph_fingerprint == original_fingerprint
    assert first_delta.after_graph_fingerprint == original_fingerprint
    assert first_delta.status == "no_progress"
    assert first_delta.changed is False
    assert state["query_graph_repair_attempted"] is True
    assert state["query_graph_repair_progressed"] is False
    assert state["planner_reflection"].passed is False
    assert state["planner_repair_requests"] == _reflection().repair_requests
    assert state["query_graph_validation_attempted"] is False
    assert state["last_action_result"].status != "success"
    assert state["action_outcomes"][-1]["status"] != "success"
    assert workflow.route_after_repair_query_graph(state) == "policy"

    workflow.repair_query_graph(state)

    exhausted = state["last_query_graph_repair_delta"]
    assert exhausted.status == "no_progress"
    assert exhausted.exhausted is True
    assert state["query_graph_repair_exhausted"] is True
    assert state["query_graph_validation_result"].gaps[0].code == "QUERY_GRAPH_REPAIR_EXHAUSTED"
    assert "DETAIL_EVIDENCE_NOT_PLANNED" in state["query_graph_validation_result"].gaps[0].reason
    assert state["agent_run_result"].evidence_gaps[0].code == "QUERY_GRAPH_REPAIR_EXHAUSTED"
    assert "DETAIL_EVIDENCE_NOT_PLANNED" in state["agent_run_result"].evidence_gaps[0].answer_instruction
    assert state["planner_reflection"].passed is False
    assert state["planner_repair_requests"] == _reflection().repair_requests

    observation = workflow.main_agent_observation(state)
    assert observation["plannerReflection"]["passed"] is False
    assert observation["plannerRepairRequests"][0]["knowledgeRequests"][0]["query"] == "read the governed metric contract"
    assert observation["queryGraphRepairDelta"]["status"] == "no_progress"
    assert observation["queryGraphRepairDelta"]["exhausted"] is True


def test_executable_graph_change_is_success_and_clears_consumed_requests(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path)
    before = query_graph_fingerprint(state["plan"])

    monkeypatch.setattr(workflow.planner, "repair", lambda *_args, **_kwargs: _plan("repaired_metric"))

    workflow.repair_query_graph(state)

    delta = state["last_query_graph_repair_delta"]
    assert delta.status == "success"
    assert delta.changed is True
    assert delta.before_graph_fingerprint == before
    assert delta.after_graph_fingerprint == query_graph_fingerprint(state["plan"])
    assert delta.after_graph_fingerprint != before
    assert state["query_graph_repair_attempted"] is True
    assert state["query_graph_repair_progressed"] is True
    assert state["planner_reflection"].passed is True
    assert state["planner_repair_requests"] == []
    assert state["query_graph_reflected"] is False


def test_repair_budget_is_scoped_by_graph_and_critic_issue(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=2)
    monkeypatch.setattr(workflow.planner, "repair", lambda _q, plan, *_args, **_kwargs: plan.model_copy(deep=True))

    workflow.repair_query_graph(state)
    first_scope = state["query_graph_repair_scope_key"]
    assert state["query_graph_repair_scope_attempt_count"] == 1

    state["planner_reflection"] = _reflection("MISSING_EVIDENCE_CONTRACT")
    state["planner_repair_reason"] = state["planner_reflection"].repair_reason
    state["planner_repair_requests"] = list(state["planner_reflection"].repair_requests)
    workflow.repair_query_graph(state)

    second_scope = state["query_graph_repair_scope_key"]
    assert second_scope != first_scope
    assert state["query_graph_repair_attempts"] == 2
    assert state["query_graph_repair_scope_attempt_count"] == 1
    assert state["query_graph_repair_exhausted"] is False
    assert state["query_graph_repair_scope_attempts"] == {first_scope: 1, second_scope: 1}
    assert workflow.policy.graph_repair_attempt_count(state) == 1

    workflow.repair_query_graph(state)

    assert state["query_graph_repair_scope_attempt_count"] == 2
    assert state["query_graph_repair_exhausted"] is True
    assert state["query_graph_validation_result"].gaps[0].code == "QUERY_GRAPH_REPAIR_EXHAUSTED"
    assert "MISSING_EVIDENCE_CONTRACT" in state["query_graph_validation_result"].gaps[0].reason


def test_scope_identity_is_stable_when_only_repair_hints_change(tmp_path):
    _workflow, state = _repair_state(tmp_path)
    first_scope = planner_repair_scope_key(state)

    state["planner_repair_requests"][0].repair_hints = ["a newly worded implementation hint"]
    state["planner_repair_requests"][0].knowledge_requests[0].reason = "new wording for the same issue"

    assert planner_repair_scope_key(state) == first_scope
