from __future__ import annotations

import pytest

from merchant_ai.services.data_snapshot_contract import (
    derive_multi_query_snapshot_requirement,
)
from merchant_ai.services.grounded_execution_graph import GroundedExecutionEdgeSpec
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    ComparisonQuestionGoal,
    DetailQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
)


def _contract(*goals):
    return OriginalQuestionGoalContract(
        question="Evaluate the declared structural goals",
        goals=list(goals),
    )


def _metric(goal_id: str) -> MetricQuestionGoal:
    return MetricQuestionGoal(goal_id=goal_id, label=goal_id)


def _derive(
    contract: OriginalQuestionGoalContract,
    assignments: dict[str, list[str]],
    *,
    edges: list[GroundedExecutionEdgeSpec] | None = None,
):
    return derive_multi_query_snapshot_requirement(
        list(assignments),
        receipt_node_ids={
            "node_a": "query_a",
            "node_b": "query_b",
            "node_c": "query_c",
        },
        graph_edges=edges or [],
        goal_contract=contract,
        goal_ids_by_query_id=assignments,
    )


def test_contract_scope_edge_requires_atomic_snapshot() -> None:
    result = _derive(
        _contract(_metric("goal.a"), _metric("goal.b")),
        {"query_a": ["goal.a"], "query_b": ["goal.b"]},
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="node_a",
                target_client_key="node_b",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
    )

    assert result.require_atomic_multi_query is True
    assert result.reasons[0].code == "ATOMIC_SNAPSHOT_CONTRACT_SCOPE_EDGE"
    assert result.reasons[0].query_ids == ["query_a", "query_b"]
    assert result.reasons[0].edge_index == 0


@pytest.mark.parametrize(
    ("derived_goal", "expected_code"),
    [
        (
            ComparisonQuestionGoal(
                goal_id="goal.derived",
                label="comparison",
                left_goal_ids=["goal.a"],
                right_goal_ids=["goal.b"],
            ),
            "ATOMIC_SNAPSHOT_CROSS_NODE_COMPARISON",
        ),
        (
            AnalysisQuestionGoal(
                goal_id="goal.derived",
                label="analysis",
                input_goal_ids=["goal.a"],
                baseline_goal_ids=["goal.b"],
            ),
            "ATOMIC_SNAPSHOT_CROSS_NODE_ANALYSIS",
        ),
    ],
)
def test_cross_node_derived_inputs_require_atomic_snapshot(
    derived_goal,
    expected_code: str,
) -> None:
    result = _derive(
        _contract(_metric("goal.a"), _metric("goal.b"), derived_goal),
        {"query_a": ["goal.a"], "query_b": ["goal.b"]},
    )

    assert result.require_atomic_multi_query is True
    assert [reason.code for reason in result.reasons] == [expected_code]
    assert result.reasons[0].query_ids == ["query_a", "query_b"]
    assert result.reasons[0].goal_ids == [
        "goal.a",
        "goal.b",
        "goal.derived",
    ]


@pytest.mark.parametrize(
    "population_goal",
    [
        RankingQuestionGoal(
            goal_id="goal.consumer",
            label="ranking",
            population_scope="SAME_AS_GOAL",
            population_goal_ids=["goal.population"],
            limit=3,
        ),
        DetailQuestionGoal(
            goal_id="goal.consumer",
            label="detail",
            population_scope="VERIFIED_PREDICATE_SCOPE",
            population_goal_ids=["goal.population"],
        ),
    ],
)
def test_cross_node_population_requires_atomic_snapshot(population_goal) -> None:
    result = _derive(
        _contract(_metric("goal.population"), population_goal),
        {
            "query_a": ["goal.population"],
            "query_b": ["goal.consumer"],
        },
    )

    assert result.require_atomic_multi_query is True
    assert [reason.code for reason in result.reasons] == [
        "ATOMIC_SNAPSHOT_CROSS_NODE_POPULATION"
    ]
    assert result.reasons[0].goal_kind == population_goal.kind
    assert result.reasons[0].query_ids == ["query_a", "query_b"]


def test_independent_goal_nodes_do_not_require_atomic_snapshot() -> None:
    result = _derive(
        _contract(_metric("goal.a"), _metric("goal.b")),
        {"query_a": ["goal.a"], "query_b": ["goal.b"]},
    )

    assert result.require_atomic_multi_query is False
    assert result.selected_query_ids == ["query_a", "query_b"]
    assert result.reasons == []


def test_colocated_structural_inputs_do_not_make_unrelated_node_atomic() -> None:
    derived = AnalysisQuestionGoal(
        goal_id="goal.derived",
        label="analysis",
        input_goal_ids=["goal.a", "goal.b"],
    )
    result = _derive(
        _contract(
            _metric("goal.a"),
            _metric("goal.b"),
            _metric("goal.independent"),
            derived,
        ),
        {
            "query_a": ["goal.a", "goal.b", "goal.derived"],
            "query_b": ["goal.independent"],
        },
    )

    assert result.require_atomic_multi_query is False
    assert result.reasons == []


def test_unselected_dependency_is_ignored_for_selected_portfolio() -> None:
    contract = _contract(
        _metric("goal.a"),
        _metric("goal.b"),
        _metric("goal.c"),
    )
    result = derive_multi_query_snapshot_requirement(
        ["query_a", "query_b"],
        receipt_node_ids={
            "node_a": "query_a",
            "node_b": "query_b",
            "node_c": "query_c",
        },
        graph_edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="node_a",
                target_client_key="node_c",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
        goal_contract=contract,
        goal_ids_by_query_id={
            "query_a": ["goal.a"],
            "query_b": ["goal.b"],
            "query_c": ["goal.c"],
        },
    )

    assert result.require_atomic_multi_query is False
    assert result.reasons == []


def test_single_query_never_requires_multi_query_snapshot() -> None:
    result = derive_multi_query_snapshot_requirement(
        ["query_a", "query_a"],
        receipt_node_ids={"node_a": "query_a", "node_b": "query_b"},
        graph_edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="node_a",
                target_client_key="node_b",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
        goal_contract=_contract(_metric("goal.a"), _metric("goal.b")),
        goal_ids_by_query_id={"query_a": ["goal.a"]},
    )

    assert result.require_atomic_multi_query is False
    assert result.selected_query_ids == ["query_a"]
    assert result.reasons == []
