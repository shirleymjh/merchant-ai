from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest

import merchant_ai.graph.workflow as workflow_module
from merchant_ai.graph.policy import AgentActionRegistry, V2AgentPolicy
from merchant_ai.graph.workflow import MerchantQaWorkflow
from merchant_ai.models import AgentDecision
from merchant_ai.services.middleware import ActionContractMiddleware


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

    assert len({action.id for action in actions}) == len(actions)
    assert len({action.node for action in actions}) == len(actions)
    assert registry.policy_routing_map() == {action.node: action.node for action in actions}
    assert registry.get("answer").id == "answer_data"
    assert "answer" not in registry.public_action_ids()
    assert all(action.expected_state_keys or action.expected_state_flags for action in actions)
    with pytest.raises(KeyError):
        registry.get("unregistered_action")


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
    assert validation_decision.selected_action == "compact_assets"

    for action_id in ("execute_graph", "execute_graph_direct", "execute_graph_agent"):
        execution = registry.get(action_id)
        assert "query_graph_validation_passed" in execution.required_state_flags
        assert "sql_generated" in execution.expected_state_flags
        assert "sql_repair_reviewed" in execution.expected_state_flags

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
        assert execution_decision.selected_action == "validate_graph"
