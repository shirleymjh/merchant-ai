from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.graph.workflow import (
    create_workflow,
    dedupe_workflow_knowledge_requests,
    new_query_graph_repair_knowledge_requests,
    planner_repair_scope_key,
)
from merchant_ai.models import (
    AgentDecision,
    AnswerMode,
    ChatContext,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    KnowledgeBundle,
    KnowledgeRequest,
    KnowledgeRequestType,
    PlannerReflectionResult,
    PlannerRepairRequest,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
    RecallItem,
)
from merchant_ai.services.middleware import ActionContractMiddleware, MiddlewareChain
from merchant_ai.services.planning_tooling import planner_repair_feedback_for_understanding


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
    assert state["query_graph_validation_attempted"] is True
    assert {
        gap.code for gap in state["query_graph_validation_result"].gaps
    } >= {"QUERY_GRAPH_REPAIR_NO_PROGRESS", "DETAIL_EVIDENCE_NOT_PLANNED"}
    assert state["last_action_result"].status != "success"
    assert state["action_outcomes"][-1]["status"] != "success"
    assert workflow.route_after_repair_query_graph(state) == "policy"

    workflow.repair_query_graph(state)

    exhausted = state["last_query_graph_repair_delta"]
    assert exhausted.status == "no_progress"
    assert exhausted.exhausted is True
    assert state["query_graph_repair_exhausted"] is True
    assert state["query_graph_validation_result"].gaps[0].code == "QUERY_GRAPH_REPAIR_EXHAUSTED"
    assert any(
        gap.code == "DETAIL_EVIDENCE_NOT_PLANNED"
        for gap in state["query_graph_validation_result"].gaps
    )
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
    state["query_graph_validation_result"] = GraphValidationResult(
        gaps=[
            GraphValidationGap(
                code="DETAIL_EVIDENCE_NOT_PLANNED",
                task_id="metric_task",
                reason="initial validator wording",
                evidence="initial raw validator evidence",
            )
        ]
    )
    first_scope = planner_repair_scope_key(state)

    state["planner_reflection"].issues[0]["reason"] = "new critic prose for the same issue"
    state["planner_reflection"].issues[0]["evidence"] = "new raw critic evidence wording"
    state["query_graph_validation_result"].gaps[0].reason = "new validator prose"
    state["query_graph_validation_result"].gaps[0].evidence = "new raw validator evidence"
    state["planner_repair_requests"][0].reason = "new prose for the same repair strategy"
    state["planner_repair_requests"][0].evidence = "new raw request evidence"
    state["planner_repair_requests"][0].query = "  检查受治理指标  "
    state["planner_repair_requests"][0].repair_hints = ["a newly worded implementation hint"]
    state["planner_repair_requests"][0].knowledge_requests[0].reason = "new wording for the same issue"
    state["planner_repair_requests"][0].knowledge_requests[0].query = (
        "  READ   THE GOVERNED METRIC CONTRACT  "
    )
    state["planner_repair_requests"][0].knowledge_requests[0].round = 9
    state["planner_repair_requests"][0].knowledge_requests[0].request_key = "runtime-key"
    state["planner_repair_requests"].append(
        state["planner_repair_requests"][0].model_copy(deep=True)
    )

    assert planner_repair_scope_key(state) == first_scope


def test_scope_identity_changes_for_repair_or_retrieval_semantics(tmp_path):
    _workflow, state = _repair_state(tmp_path)
    baseline = planner_repair_scope_key(state)
    original = state["planner_repair_requests"][0].model_copy(deep=True)

    variants = []

    action_changed = original.model_copy(deep=True)
    action_changed.action = "re_understand"
    variants.append(action_changed)

    query_changed = original.model_copy(deep=True)
    query_changed.query = "rebuild a different governed evidence contract"
    variants.append(query_changed)

    request_query_changed = original.model_copy(deep=True)
    request_query_changed.knowledge_requests[0].query = "read a different metric contract"
    variants.append(request_query_changed)

    source_phrase_changed = original.model_copy(deep=True)
    source_phrase_changed.knowledge_requests[0].source_phrase = "用户所说的支付订单量"
    variants.append(source_phrase_changed)

    expected_refs_changed = original.model_copy(deep=True)
    expected_refs_changed.knowledge_requests[0].expected_refs = ["semantic:metric:pay_order_count"]
    variants.append(expected_refs_changed)

    for variant in variants:
        state["planner_repair_requests"] = [variant]
        assert planner_repair_scope_key(state) != baseline


def test_new_repair_knowledge_request_uses_full_retrieval_semantic_identity():
    existing = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="read the governed metric contract",
        needed_for_task_id="metric_task",
        source_phrase="支付订单量",
        expected_refs=["semantic:metric:pay_order_count"],
        reason="initial explanation",
    )
    prose_only_change = existing.model_copy(
        update={"reason": "rewritten explanation", "round": 5, "request_key": "runtime-key"}
    )
    source_phrase_change = existing.model_copy(update={"source_phrase": "支付成功订单量"})
    expected_refs_change = existing.model_copy(
        update={"expected_refs": ["semantic:metric:settled_order_count"]}
    )

    before = QueryPlan(knowledge_requests=[existing])
    assert new_query_graph_repair_knowledge_requests(
        before,
        QueryPlan(knowledge_requests=[prose_only_change]),
    ) == []

    added = new_query_graph_repair_knowledge_requests(
        before,
        QueryPlan(knowledge_requests=[source_phrase_change, expected_refs_change]),
    )
    assert [item.source_phrase for item in added] == ["支付成功订单量", "支付订单量"]
    assert added[1].expected_refs == ["semantic:metric:settled_order_count"]

    deduped = dedupe_workflow_knowledge_requests(
        [existing, prose_only_change, source_phrase_change, expected_refs_change]
    )
    assert deduped == [existing, source_phrase_change, expected_refs_change]


def test_repair_exception_becomes_blocking_gap_and_preserves_critic_input(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=2)
    before = query_graph_fingerprint(state["plan"])

    def fail_repair(*_args, **_kwargs):
        raise RuntimeError("repair provider unavailable")

    monkeypatch.setattr(workflow.planner, "repair", fail_repair)

    workflow.repair_query_graph(state)

    assert query_graph_fingerprint(state["plan"]) == before
    assert state["last_query_graph_repair_delta"].status == "failed"
    assert state["last_query_graph_repair_delta"].changed is False
    assert state["planner_reflection"].passed is False
    assert state["planner_repair_requests"] == _reflection().repair_requests
    assert {
        gap.code for gap in state["query_graph_validation_result"].gaps
    } >= {"QUERY_GRAPH_REPAIR_FAILED", "DETAIL_EVIDENCE_NOT_PLANNED"}
    assert any(
        gap.code == "QUERY_GRAPH_REPAIR_FAILED"
        for gap in state["agent_run_result"].evidence_gaps
    )


def test_critic_exception_becomes_blocking_gap_instead_of_breaking_run(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=2)

    def fail_reflection(*_args, **_kwargs):
        raise RuntimeError("critic provider unavailable")

    monkeypatch.setattr(workflow.planner_reflection_agent, "reflect", fail_reflection)

    workflow.reflect_query_graph(state)

    assert state["planner_reflection"].passed is False
    assert state["planner_reflection"].issues[0]["code"] == "PLANNER_CRITIC_FAILED"
    assert state["query_graph_validation_result"].gaps[0].code == "PLANNER_CRITIC_FAILED"
    assert state["last_action_result"].status == "failed"
    assert any(
        gap.code == "PLANNER_CRITIC_FAILED"
        for gap in state["agent_run_result"].evidence_gaps
    )


def test_knowledge_request_only_repair_is_awaiting_not_graph_success(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=2)
    before = query_graph_fingerprint(state["plan"])
    request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="supplemental governed metric definition",
        needed_for_task_id="metric_task",
        reason="repair needs governed evidence",
    )

    def request_knowledge(_question, plan, *_args, **_kwargs):
        repaired = plan.model_copy(deep=True)
        repaired.knowledge_requests.append(request)
        return repaired

    monkeypatch.setattr(workflow.planner, "repair", request_knowledge)

    workflow.repair_query_graph(state)

    delta = state["last_query_graph_repair_delta"]
    assert delta.status == "awaiting_knowledge"
    assert delta.changed is False
    assert delta.after_graph_fingerprint != before
    assert state["plan"].intents == _plan().intents
    assert state["query_graph_repair_progressed"] is False
    assert state["query_graph_repair_exhausted"] is False
    assert state["query_graph_repair_scope_attempt_count"] == 0
    assert workflow.policy.graph_repair_attempt_count(state) == 0
    assert state["planner_reflection"].passed is False
    assert state["planner_repair_requests"] == _reflection().repair_requests
    assert state["pending_knowledge_requests"][0].query == request.query
    assert state["last_action_result"].status == "awaiting_knowledge"
    assert state["query_graph_validation_result"].gaps[0].code == "QUERY_GRAPH_REPAIR_AWAITING_KNOWLEDGE"
    assert workflow.route_after_repair_query_graph(state) == "policy"


def test_awaiting_knowledge_resumes_same_critic_and_repairs_with_one_attempt_budget(
    tmp_path,
    monkeypatch,
):
    workflow, state = _repair_state(tmp_path, rounds=1)
    request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="supplemental governed metric definition",
        needed_for_task_id="metric_task",
        reason="repair needs governed evidence",
    )
    repair_calls = []

    def repair_after_knowledge(_question, plan, *_args, **_kwargs):
        repair_calls.append(1)
        if len(repair_calls) == 1:
            suspended = plan.model_copy(deep=True)
            suspended.knowledge_requests.append(request)
            return suspended
        return _plan("repaired_metric")

    class RequestAwareRetriever:
        backend_name = "repair-contract"

        def retrieve(self, retrieval_request):
            suffix = "request" if retrieval_request.knowledge_request is not None else "workspace"
            item = RecallItem(
                doc_id="semantic:metric:governed:%s" % suffix,
                title="governed metric",
                content="published governed metric contract",
                source_type="SEMANTIC_METRIC",
                fusion_score=10.0,
            )
            return KnowledgeBundle(
                backend=self.backend_name,
                retrieval_status="success",
                recall_bundle=RecallBundle(items=[item], top_score=10.0),
            )

    monkeypatch.setattr(workflow.planner, "repair", repair_after_knowledge)
    monkeypatch.setattr(workflow, "knowledge_retriever", RequestAwareRetriever())
    monkeypatch.setattr(workflow, "load_skill_policies_for_retrieval", lambda _state: [])
    monkeypatch.setattr(workflow.asset_builder, "compact", lambda *_args, **_kwargs: PlanningAssetPack())

    workflow.repair_query_graph(state)

    assert state["last_query_graph_repair_delta"].status == "awaiting_knowledge"
    assert workflow.policy.graph_repair_attempt_count(state) == 0

    workflow.retrieve_knowledge(state)

    assert state["plan"].intents == _plan().intents
    assert state["plan"].knowledge_requests == []
    assert state["planner_reflection"].issues[0]["code"] == "DETAIL_EVIDENCE_NOT_PLANNED"
    assert any(
        gap.code == "DETAIL_EVIDENCE_NOT_PLANNED"
        for gap in state["query_graph_validation_result"].gaps
    )
    assert workflow.policy.graph_repair_attempt_count(state) == 0

    workflow.compact_assets(state)

    assert "repair_graph" in workflow.policy.autonomous_candidate_action_ids(state)
    assert state["planner_reflection"].issues[0]["code"] == "DETAIL_EVIDENCE_NOT_PLANNED"

    workflow.repair_query_graph(state)

    assert len(repair_calls) == 2
    assert state["last_query_graph_repair_delta"].status == "success"
    assert state["plan"].intents[0].metric_name == "repaired_metric"
    assert state["query_graph_repair_scope_attempt_count"] == 1


def test_same_unmet_knowledge_request_cannot_suspend_repair_forever(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=1)
    request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="missing governed metric definition",
        needed_for_task_id="metric_task",
        reason="repair needs governed evidence",
    )
    repair_calls = []

    def repeat_request(_question, plan, *_args, **_kwargs):
        repair_calls.append(1)
        repeated = plan.model_copy(deep=True)
        repeated.knowledge_requests.append(request)
        return repeated

    class NoRequestMatchRetriever:
        backend_name = "repair-contract-empty"

        def retrieve(self, retrieval_request):
            if retrieval_request.knowledge_request is not None:
                return KnowledgeBundle(
                    backend=self.backend_name,
                    retrieval_status="empty",
                    recall_bundle=RecallBundle(),
                )
            item = RecallItem(
                doc_id="semantic:workspace:unrelated",
                title="workspace context",
                content="unrelated governed context",
                source_type="SEMANTIC_TABLE_ASSET",
                fusion_score=10.0,
            )
            return KnowledgeBundle(
                backend=self.backend_name,
                retrieval_status="success",
                recall_bundle=RecallBundle(items=[item], top_score=10.0),
            )

    monkeypatch.setattr(workflow.planner, "repair", repeat_request)
    monkeypatch.setattr(workflow, "knowledge_retriever", NoRequestMatchRetriever())
    monkeypatch.setattr(workflow, "load_skill_policies_for_retrieval", lambda _state: [])
    monkeypatch.setattr(workflow.asset_builder, "compact", lambda *_args, **_kwargs: PlanningAssetPack())

    workflow.repair_query_graph(state)
    workflow.retrieve_knowledge(state)
    workflow.compact_assets(state)

    assert state["blocked_knowledge_request_keys"]
    assert state["pending_knowledge_requests"] == []
    assert workflow.policy.graph_repair_attempt_count(state) == 0

    workflow.repair_query_graph(state)

    assert len(repair_calls) == 2
    assert state["last_query_graph_repair_delta"].status == "no_progress"
    assert state["last_query_graph_repair_delta"].exhausted is True
    assert state["query_graph_repair_scope_attempt_count"] == 1
    assert state["pending_knowledge_requests"] == []

    workflow.repair_query_graph(state)

    assert len(repair_calls) == 2
    assert state["query_graph_repair_exhausted"] is True


def test_no_progress_projects_critic_gap_before_global_budget_answer(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=3)
    monkeypatch.setattr(
        workflow.planner,
        "repair",
        lambda _question, plan, *_args, **_kwargs: plan.model_copy(deep=True),
    )
    state["react_round"] = workflow.policy.max_main_actions - 1

    workflow.repair_query_graph(state)

    assert state["react_round"] >= workflow.policy.max_main_actions
    assert {
        gap.code for gap in state["agent_run_result"].evidence_gaps
    } >= {"QUERY_GRAPH_REPAIR_NO_PROGRESS", "DETAIL_EVIDENCE_NOT_PLANNED"}
    assert "DETAIL_EVIDENCE_NOT_PLANNED" in state["agent_run_result"].partial_answer_reason


def test_changed_repair_request_strategy_gets_a_fresh_scope_attempt(tmp_path, monkeypatch):
    workflow, state = _repair_state(tmp_path, rounds=1)
    calls = []

    def noop(_question, plan, *_args, **_kwargs):
        calls.append(1)
        return plan.model_copy(deep=True)

    monkeypatch.setattr(workflow.planner, "repair", noop)
    workflow.repair_query_graph(state)
    exhausted_scope = state["query_graph_repair_scope_key"]
    assert state["query_graph_repair_exhausted"] is True

    updated = state["planner_repair_requests"][0].model_copy(deep=True)
    updated.action = "re_understand"
    updated.query = "rebuild the governed evidence contract"
    updated.knowledge_requests[0].query = "a different governed metric contract"
    state["planner_repair_requests"] = [updated]

    fresh_scope = planner_repair_scope_key(state)
    assert fresh_scope != exhausted_scope

    workflow.repair_query_graph(state)

    assert len(calls) == 2
    assert state["query_graph_repair_scope_key"] == fresh_scope
    assert state["query_graph_repair_scope_attempt_count"] == 1


def test_non_calculation_gap_is_returned_as_structured_repair_feedback():
    feedback = planner_repair_feedback_for_understanding(
        [
            GraphValidationGap(
                code="MISSING_EVIDENCE_CONTRACT",
                task_id="metric_task",
                evidence="required evidence",
                reason="the plan omitted its evidence obligation",
            )
        ],
        {
            "anchorMetric": {"metricRef": "governed_metric"},
            "filters": [{"field": "governed_key", "value": "A-1"}],
            "requiredEvidenceIntents": [{"semanticLabel": "requested evidence"}],
        },
    )

    assert feedback["mustFixBeforePlanning"] is True
    assert feedback["queryGraph"][0]["code"] == "MISSING_EVIDENCE_CONTRACT"
    assert feedback["queryGraph"][0]["taskId"] == "metric_task"
    assert "Do not repeat" in feedback["queryGraph"][0]["requiredOutcome"]
    assert feedback["preservedContractCounts"] == {
        "anchorMetric": 1,
        "filters": 1,
        "requiredEvidenceIntents": 1,
    }
