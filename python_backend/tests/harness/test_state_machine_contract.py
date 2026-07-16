from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest
from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import START as LANGGRAPH_START
from langgraph.graph import StateGraph

import merchant_ai.graph.workflow as workflow_module
from merchant_ai.graph.policy import AgentActionRegistry, V2AgentPolicy
from merchant_ai.graph.state import AgentState
from merchant_ai.graph.workflow import MerchantQaWorkflow
from merchant_ai.models import AgentActionTrace, AgentDecision, GraphValidationGap, QuestionCategory
from merchant_ai.services.middleware import ActionContractMiddleware, MiddlewareChain


class RecordingStateGraph:
    def __init__(self, schema: Any):
        self.schema = schema
        self.nodes: Dict[str, Callable[..., Any]] = {}
        self.edges: list[tuple[Any, Any]] = []
        self.conditionals: Dict[str, Dict[str, str]] = {}
        self.checkpointer: Any = None

    def add_node(self, node: str, handler: Callable[..., Any]) -> None:
        self.nodes[node] = handler

    def add_edge(self, source: Any, target: Any) -> None:
        self.edges.append((source, target))

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[..., str],
        path_map: Dict[str, str],
    ) -> None:
        del router
        self.conditionals[source] = dict(path_map)

    def compile(self, checkpointer: Any = None) -> RecordingStateGraph:
        self.checkpointer = checkpointer
        return self


def test_action_registry_has_unique_nodes_and_closed_contracts() -> None:
    registry = AgentActionRegistry()
    actions = registry.actions(registry.public_action_ids())
    routing_actions = registry.actions(registry.routing_action_ids())

    assert len({action.id for action in actions}) == len(actions)
    assert len({action.node for action in actions}) == len(actions)
    assert registry.policy_routing_map() == {
        action.node: action.node for action in routing_actions
    }
    assert "observe_contract_block" not in registry.public_action_ids()
    assert not any(action.fallback_action for action in routing_actions)
    assert registry.get("answer").id == "answer_data"
    assert "answer" not in registry.public_action_ids()
    assert all(action.expected_state_keys or action.expected_state_flags for action in actions)
    with pytest.raises(KeyError):
        registry.get("unregistered_action")


def test_graph_repair_attempt_flag_has_a_persisted_state_channel() -> None:
    assert "query_graph_repair_attempted" in AgentState.__annotations__


def test_cross_node_control_channels_round_trip_through_langgraph_state_schema() -> None:
    samples: Dict[str, Any] = {
        "_active_step_id": "step_1",
        "_answer_ready_emitted": True,
        "_clarification_tool_intercepted": True,
        "_emitted_tool_runtime_event_ids": ["event_1"],
        "_lead_llm_decision_fingerprint": "lead_fingerprint",
        "_lead_previous_gap_counts": {"graphGaps": 2},
        "_lead_seen_recall_refs": ["semantic:domain:table:metric:measure"],
        "_memory_middleware_retry_attempted": True,
        "_memory_middleware_snapshot_ready": True,
        "_memory_semantic_refresh_attempted": True,
        "_memory_snapshot_locked": True,
        "_middleware_offloaded_tasks": ["task_1"],
        "_preflight_question": "governed question",
        "_preflight_requires_full_context": True,
        "_route_slots_bootstrapped": True,
        "_route_understanding_question": "governed question",
        "_runtime_context_stale": True,
        "_skill_middleware_loaded": ["analysis_skill"],
        "_summarized_stages": ["policy_round_1"],
        "analysis_worker_result": {"status": "completed"},
        "capability_decisions": {"metricFastEntry": {"allowed": True}},
        "confirmation_restore_status": {"status": "restored"},
        "knowledge_expanded_topics": [QuestionCategory.ORDER],
        "knowledge_recall_coverage": {"topicExpansion": {"reason": "coverage_gap"}},
        "last_query_graph_validation_gaps": [
            GraphValidationGap(code="MISSING_METRIC", reason="metric contract missing")
        ],
        "middleware_action_context_hashes": {"plan_graph": "context_hash"},
        "middleware_blocked": True,
        "preflight_understanding": {"surfaceSignals": {"business": True}},
        "semantic_preflight_route_trace": {"source": "semantic_assets"},
        "time_window_contract": {"kind": "rolling", "days": 30},
    }
    assert set(samples) <= set(AgentState.__annotations__)

    observed: Dict[str, Any] = {}

    def produce(_: AgentState) -> Dict[str, Any]:
        return dict(samples)

    def consume(state: AgentState) -> Dict[str, Any]:
        observed.update({key: state.get(key) for key in samples})
        return {"_next_action": "done"}

    builder = StateGraph(AgentState)
    builder.add_node("produce", produce)
    builder.add_node("consume", consume)
    builder.add_edge(LANGGRAPH_START, "produce")
    builder.add_edge("produce", "consume")
    builder.add_edge("consume", LANGGRAPH_END)

    result = builder.compile().invoke({})

    assert observed == samples
    assert {key: result.get(key) for key in samples} == samples
    assert result["_next_action"] == "done"


def test_workflow_registers_and_routes_every_action_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_module, "StateGraph", RecordingStateGraph)
    workflow = object.__new__(MerchantQaWorkflow)
    workflow.policy = V2AgentPolicy()
    workflow.checkpoint_manager = SimpleNamespace(saver=lambda: "checkpoint")

    graph = workflow._build_graph()
    registered_nodes = set(workflow.policy.registry.policy_routing_map())

    assert registered_nodes <= set(graph.nodes)
    assert graph.conditionals["policy"] == workflow.policy.registry.policy_routing_map()
    assert len(graph.edges) == len(set(graph.edges))

    outbound_sources = {source for source, _ in graph.edges} | set(graph.conditionals)
    assert set(graph.nodes) <= outbound_sources

    valid_targets = set(graph.nodes) | {workflow_module.END}
    edge_targets = {target for _, target in graph.edges}
    conditional_targets = {
        target
        for path_map in graph.conditionals.values()
        for target in path_map.values()
    }
    assert edge_targets | conditional_targets <= valid_targets
    terminal_nodes = {
        workflow.policy.registry.node_for("ask_human"),
        workflow.policy.registry.node_for("cache_answer"),
        workflow.policy.registry.node_for("terminal_end"),
    }
    assert {(node, "finalize_action_contract") for node in terminal_nodes} <= set(graph.edges)
    assert ("finalize_action_contract", workflow_module.END) in graph.edges
    assert {
        (node, "policy")
        for node in registered_nodes - terminal_nodes
    } <= set(graph.edges)
    assert not {
        (source, target)
        for source, target in graph.edges
        if source in registered_nodes and target in registered_nodes
    }
    assert set(graph.conditionals) == {"preflight_route", "policy"}


def test_terminal_finalizer_closes_the_selected_action_contract() -> None:
    workflow = object.__new__(MerchantQaWorkflow)
    workflow.middleware_chain = MiddlewareChain([ActionContractMiddleware()])
    decision = AgentDecision(
        selected_action="answer_data",
        selected_node="answer_analysis",
        available_actions=["answer_data"],
    )
    state = {
        "answer": "",
        "action_history": [
            AgentActionTrace(action="answer_data", node="answer_analysis", status="selected")
        ],
        "action_outcomes": [],
        "middleware_events": [],
    }
    workflow.middleware_chain.capture_action(state, decision)
    state["answer"] = "complete"

    finalized = workflow.finalize_action_contract(state)

    assert not finalized.get("_pending_action_contract")
    assert finalized["action_history"][-1].status == "success"
    assert finalized["last_action_result"].status == "success"
    assert finalized["action_outcomes"][-1]["status"] == "success"

    finalized_again = workflow.finalize_action_contract(finalized)
    assert len(finalized_again["action_outcomes"]) == 1


def test_action_contract_counts_declared_postcondition_change_as_progress() -> None:
    workflow = object.__new__(MerchantQaWorkflow)
    workflow.middleware_chain = MiddlewareChain([ActionContractMiddleware()])
    decision = AgentDecision(
        selected_action="cache_answer",
        selected_node="cache_answer",
        available_actions=["cache_answer"],
    )
    state = {
        "answer": "complete",
        "response_context": None,
        "action_history": [
            AgentActionTrace(action="cache_answer", node="cache_answer", status="selected")
        ],
        "action_outcomes": [],
        "middleware_events": [],
    }
    workflow.middleware_chain.capture_action(state, decision)
    state["response_context"] = {"question": "governed question"}

    finalized = workflow.finalize_action_contract(state)

    assert finalized["last_action_result"].status == "success"
    assert finalized["action_history"][-1].status == "success"
    assert len(finalized["action_outcomes"]) == 1


def test_validation_and_execution_actions_fail_closed_on_missing_prerequisites() -> None:
    registry = AgentActionRegistry()
    contract = ActionContractMiddleware()
    validation = registry.get("validate_graph")
    assert validation.required_state_flags == ["planning_assets_compacted"]
    assert "query_graph_validation_result" in validation.expected_state_keys
    assert "query_graph_validation_attempted" in validation.expected_state_flags

    validation_decision = AgentDecision(
        selected_action="validate_graph",
        selected_node=validation.node,
        available_actions=["validate_graph"],
    )
    contract.before_action({"middleware_events": []}, validation_decision)
    assert validation_decision.selected_action == "observe_contract_block"

    for action_id in ("execute_graph", "execute_graph_direct", "execute_graph_agent"):
        execution = registry.get(action_id)
        assert "query_graph_validation_passed" in execution.required_state_flags
        assert execution.expected_state_flags == ["sql_generated"]

        execution_decision = AgentDecision(
            selected_action=action_id,
            selected_node=execution.node,
            available_actions=[action_id],
        )
        contract.before_action(
            {
                "plan": {"intents": [{"node": "alpha"}]},
                "planning_assets_compacted": True,
                "query_graph_validation_result": {"valid": False},
                "middleware_events": [],
            },
            execution_decision,
        )
        assert execution_decision.selected_action == "observe_contract_block"
        assert execution_decision.available_actions == ["observe_contract_block"]


def test_invalid_action_is_observed_without_implicit_business_fallback() -> None:
    decision = AgentDecision(
        selected_action="execute_graph",
        selected_node="execute_query_graph",
        available_actions=["execute_graph", "validate_graph", "answer_data"],
        reason="injected invalid selection",
    )
    state = {
        "plan": {"intents": [{"node": "metric"}]},
        "query_graph_validation_result": {"valid": False},
        "middleware_events": [],
    }

    ActionContractMiddleware().before_action(state, decision)

    assert decision.selected_action == "observe_contract_block"
    assert decision.selected_node == "observe_contract_block"
    assert decision.source == "contract_block"
    assert "validate_graph" not in decision.available_actions
    assert "answer_data" not in decision.available_actions
    assert state["contract_block_observation"]["blockedAction"] == "execute_graph"
    assert state["contract_block_observation"]["missingStateFlags"] == [
        "query_graph_validation_passed"
    ]
    assert state["middleware_events"][-1].code == "ACTION_CONTRACT_BLOCKED"


def test_policy_filters_unsafe_catalog_entries_before_lead_selection() -> None:
    class CandidatePolicy(V2AgentPolicy):
        def _candidate_action_ids(self, _state: AgentState):
            return ["execute_graph", "answer_data"], "test catalog", False

    state = {
        "plan": {"intents": [{"node": "metric"}]},
        "query_graph_validation_result": {"valid": False},
    }

    decision = CandidatePolicy().decide(state)

    assert decision.available_actions == ["answer_data"]
    assert decision.selected_action == "answer_data"
    assert state["action_catalog_contract_blocks"] == [
        {
            "action": "execute_graph",
            "node": "execute_query_graph",
            "missingStateKeys": [],
            "missingStateFlags": ["query_graph_validation_passed"],
        }
    ]


def test_policy_never_executes_the_first_business_candidate_before_lead_arbitration() -> None:
    class CandidatePolicy(V2AgentPolicy):
        def _candidate_action_ids(self, _state: AgentState):
            return ["plan_graph", "retrieve_knowledge"], "unordered safe catalog", False

    decision = CandidatePolicy().decide(
        {
            "planning_assets_compacted": True,
            "topic_routed": True,
        }
    )

    assert decision.selected_action == "lead_arbitrate"
    assert decision.selected_action not in decision.available_actions
    assert decision.available_actions == ["plan_graph", "retrieve_knowledge"]
    assert decision.source == "lead_arbitration_pending"


def test_policy_aggregates_multiple_contract_blocks_without_selecting_the_first() -> None:
    class CandidatePolicy(V2AgentPolicy):
        def _candidate_action_ids(self, _state: AgentState):
            return ["execute_graph", "validate_graph"], "unordered blocked catalog", False

    state = {}

    decision = CandidatePolicy().decide(state)

    assert decision.selected_action == "observe_contract_block"
    assert state["contract_block_observation"]["blockedAction"] == ""
    assert state["contract_block_observation"]["blockedActions"] == [
        "execute_graph",
        "validate_graph",
    ]
    assert state["contract_block_observation"]["missingStateKeys"] == [
        "plan.intents",
        "query_graph_validation_result",
    ]
    assert state["contract_block_observation"]["missingStateFlags"] == [
        "planning_assets_compacted",
        "query_graph_validation_passed",
    ]


def test_contract_block_observation_consumes_round_and_returns_to_lead() -> None:
    workflow = object.__new__(MerchantQaWorkflow)
    workflow.start_run_step = lambda *_args, **_kwargs: None
    workflow.finish_run_step = lambda *_args, **_kwargs: None
    workflow.record_span = lambda *_args, **_kwargs: None
    state = {
        "react_round": 3,
        "contract_block_generation": 0,
        "contract_block_observation": {
            "status": "pending",
            "blockedAction": "execute_graph",
            "missingStateFlags": ["query_graph_validation_passed"],
        },
        "thinking_steps": [],
    }

    observed = workflow.observe_contract_block(state)

    assert observed["react_round"] == 4
    assert observed["contract_block_generation"] == 1
    assert observed["contract_block_observed"] is True
    assert observed["contract_block_observation"]["status"] == "observed"


def test_adaptive_lead_selects_between_any_multiple_safe_actions() -> None:
    class ChoosingLeadLlm:
        configured = True

        def __init__(self) -> None:
            self.calls = 0

        def tool_json_chat(self, *_args: Any, **_kwargs: Any) -> Dict[str, str]:
            self.calls += 1
            return {"actionId": "retrieve_knowledge", "reason": "inspect semantic assets first"}

    llm = ChoosingLeadLlm()
    workflow = object.__new__(MerchantQaWorkflow)
    workflow.settings = SimpleNamespace(
        lead_action_llm_mode="adaptive",
        run_budget_max_duration_seconds=90,
        run_budget_fast_duration_seconds=25,
        llm_request_timeout_seconds=20,
        openai_model="test-model",
        openai_base_url="test-provider",
    )
    workflow.policy = V2AgentPolicy()
    workflow.planner = SimpleNamespace(llm=llm)
    workflow.record_span = lambda *_args, **_kwargs: None
    decision = AgentDecision(
        selected_action="plan_graph",
        selected_node=workflow.policy.registry.node_for("plan_graph"),
        available_actions=["plan_graph", "retrieve_knowledge"],
        reason="safe catalog fallback",
    )
    state = {
        "question": "analyse the governed business question",
        "main_agent_observations": [{"summary": "two safe tools remain"}],
        "action_history": [],
        "pending_knowledge_requests": [],
        "planner_repair_requests": [],
    }

    selected = workflow.arbitrate_lead_action_if_needed(state, decision)

    assert selected.selected_action == "retrieve_knowledge"
    assert selected.source == "lead_llm_tool"
    assert llm.calls == 1
    assert state["bounded_lead_llm_trace"]["status"] == "accepted"

    repeated = workflow.arbitrate_lead_action_if_needed(state, decision)
    assert repeated.selected_action == "ask_human"
    assert repeated.source == "runtime_fail_closed"
    assert llm.calls == 1
    assert state["bounded_lead_llm_trace"]["errorCode"] == "LEAD_DECISION_UNAVAILABLE"


def test_lead_catalog_preserves_safe_answer_action_without_business_preference() -> None:
    workflow = object.__new__(MerchantQaWorkflow)

    assert workflow.lead_llm_action_catalog(
        ["repair_graph", "answer_data", "repair_graph"]
    ) == ["repair_graph", "answer_data"]


def test_completed_answer_closes_through_cache_before_other_policy_branches() -> None:
    decision = V2AgentPolicy().decide(
        {
            "chat_bi_completed": True,
            "run_budget_exhausted": True,
            "evidence_graph_verified": True,
        }
    )

    assert decision.selected_action == "cache_answer"
    assert decision.available_actions == ["cache_answer"]
    assert decision.budget_exhausted is False


def test_hypothesis_catalog_uses_structured_analysis_contract_not_complexity_hint() -> None:
    policy = V2AgentPolicy()
    policy.settings = SimpleNamespace(hypothesis_query_exploration_enabled=True)
    state = {
        "fast_understanding": SimpleNamespace(complexity="complex"),
        "evidence_graph_verified": True,
        "evidence_accepted": True,
        "agent_run_result": SimpleNamespace(
            verified_evidence=SimpleNamespace(passed=True)
        ),
        "hypothesis_exploration": {
            "hypotheses": [{"hypothesisId": "h1"}, {"hypothesisId": "h2"}],
            "questionSignals": {"mentionsAttribution": True, "mentionsDrop": True},
        },
        "plan": SimpleNamespace(
            intents=[object()],
            question_understanding={
                "analysisIntent": "comparison",
                "requiresExplanation": False,
                "requiredEvidenceIntents": [],
            },
        ),
    }

    assert policy.hypothesis_exploration_needed(state) is False

    state["plan"].question_understanding = {
        "analysisIntent": "diagnostic",
        "requiresExplanation": True,
        "requiredEvidenceIntents": [{"intent": "driver_analysis"}],
    }
    assert policy.hypothesis_exploration_needed(state) is True
