from __future__ import annotations

from collections.abc import Iterable

import pytest

from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionEdgeSpec,
    GroundedExecutionGraphProposal,
    GroundedExecutionNodeSpec,
    build_grounded_execution_graph_receipt,
    discovery_evidence_snapshot_fingerprint,
    validate_grounded_execution_graph,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
)


def _goal_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "Show order details, then rank the applicable refunds",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "DETAIL",
                    "label": "order details",
                },
                {
                    "goalId": "ranking.refunds",
                    "kind": "RANKING",
                    "label": "refund ranking",
                    "metricGoalIds": ["metric.refund_amount"],
                    "limit": 3,
                    "populationScope": "ALL_MATCHING_ROWS",
                },
                {
                    "goalId": "metric.refund_amount",
                    "kind": "METRIC",
                    "label": "refund amount",
                },
            ],
        }
    )


def _population_goal_contract() -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": "Show scoped details and rank within that population",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "DETAIL",
                    "label": "scoped details",
                },
                {
                    "goalId": "metric.refund_amount",
                    "kind": "METRIC",
                    "label": "ranking measure",
                },
                {
                    "goalId": "ranking.refunds",
                    "kind": "RANKING",
                    "label": "ranking within the scoped details",
                    "metricGoalIds": ["metric.refund_amount"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                    "populationGoalIds": ["detail.orders"],
                },
            ],
        }
    )


def _evidence() -> list[dict[str, str]]:
    return [
        {
            "refId": "semantic:orders:table",
            "contentHash": "orders-table-hash",
            "topic": "orders",
        },
        {
            "refId": "semantic:refunds:metric",
            "contentHash": "refund-metric-hash",
            "topic": "refunds",
        },
    ]


def _node(
    client_key: str,
    goal_id: str | list[str],
    topic: str,
    evidence_ref_id: str,
) -> GroundedExecutionNodeSpec:
    return GroundedExecutionNodeSpec(
        client_key=client_key,
        objective=(goal_id if isinstance(goal_id, str) else ",".join(goal_id)),
        goal_ids=([goal_id] if isinstance(goal_id, str) else goal_id),
        topic_scope=[topic],
        evidence_ref_ids=[evidence_ref_id],
    )


def _proposal(
    *,
    contract: OriginalQuestionGoalContract | None = None,
    evidence: list[dict[str, str]] | None = None,
    nodes: list[GroundedExecutionNodeSpec] | None = None,
    edges: list[GroundedExecutionEdgeSpec] | None = None,
    base_version: int = 2,
) -> GroundedExecutionGraphProposal:
    active_contract = contract or _goal_contract()
    active_evidence = evidence or _evidence()
    return GroundedExecutionGraphProposal(
        base_version=base_version,
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(
            active_contract
        ),
        discovery_snapshot_fingerprint=discovery_evidence_snapshot_fingerprint(
            active_evidence
        ),
        nodes=nodes
        or [
            _node(
                "orders",
                "detail.orders",
                "orders",
                "semantic:orders:table",
            ),
            _node(
                "refunds",
                ["ranking.refunds", "metric.refund_amount"],
                "refunds",
                "semantic:refunds:metric",
            ),
        ],
        edges=edges or [],
    )


def _validate(
    proposal: GroundedExecutionGraphProposal,
    *,
    contract: OriginalQuestionGoalContract | None = None,
    evidence: list[dict[str, str]] | None = None,
    routed_topics: list[str] | None = None,
    current_version: int = 2,
):
    return validate_grounded_execution_graph(
        proposal,
        goal_contract=contract or _goal_contract(),
        discovery_evidence=evidence or _evidence(),
        routed_topics=routed_topics or ["orders", "refunds"],
        current_version=current_version,
    )


def _codes(result: object) -> set[str]:
    issues: Iterable[object] = getattr(result, "issues")
    return {str(getattr(issue, "code")) for issue in issues}


def test_valid_graph_is_accepted_and_receipt_has_stable_node_ids() -> None:
    proposal = _proposal()

    result = _validate(proposal)
    receipt = build_grounded_execution_graph_receipt(proposal, version=3)

    assert result.valid is True
    assert result.issues == []
    assert receipt.version == 3
    assert set(receipt.node_ids) == {"orders", "refunds"}
    assert set(receipt.parallel_frontier) == set(receipt.node_ids.values())
    assert receipt.waiting_artifact_nodes == []
    assert receipt.discovery_snapshot_fingerprint == (
        proposal.discovery_snapshot_fingerprint
    )


def test_goal_contract_fingerprint_mismatch_fails_closed() -> None:
    proposal = _proposal().model_copy(
        update={"goal_contract_fingerprint": "wrong-goal-fingerprint"}
    )

    result = _validate(proposal)

    assert result.valid is False
    assert "EXECUTION_GRAPH_GOAL_FINGERPRINT_MISMATCH" in _codes(result)


def test_discovery_snapshot_staleness_fails_closed() -> None:
    proposal = _proposal().model_copy(
        update={"discovery_snapshot_fingerprint": "stale-discovery-snapshot"}
    )

    result = _validate(proposal)

    assert result.valid is False
    assert "EXECUTION_GRAPH_DISCOVERY_SNAPSHOT_STALE" in _codes(result)


def test_compare_and_swap_version_conflict_fails_closed() -> None:
    result = _validate(_proposal(base_version=1), current_version=2)

    assert result.valid is False
    assert "EXECUTION_GRAPH_VERSION_CONFLICT" in _codes(result)


def test_duplicate_node_key_fails_closed() -> None:
    nodes = [
        _node(
            "duplicate",
            "detail.orders",
            "orders",
            "semantic:orders:table",
        ),
        _node(
            "duplicate",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert "EXECUTION_GRAPH_NODE_DUPLICATE" in _codes(result)


def test_unknown_goal_and_required_goal_omission_are_both_reported() -> None:
    nodes = [
        _node(
            "unknown-goal",
            "goal.not.published",
            "orders",
            "semantic:orders:table",
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert _codes(result) >= {
        "EXECUTION_GRAPH_GOAL_UNKNOWN",
        "EXECUTION_GRAPH_REQUIRED_GOALS_UNASSIGNED",
    }
    missing = next(
        issue
        for issue in result.issues
        if issue.code == "EXECUTION_GRAPH_REQUIRED_GOALS_UNASSIGNED"
    )
    assert missing.details["goalIds"] == ["detail.orders"]


def test_node_without_goal_assignment_fails_closed() -> None:
    nodes = [
        GroundedExecutionNodeSpec(
            client_key="orders",
            topic_scope=["orders"],
            evidence_ref_ids=["semantic:orders:table"],
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert _codes(result) >= {
        "EXECUTION_GRAPH_NODE_GOALS_REQUIRED",
        "EXECUTION_GRAPH_REQUIRED_GOALS_UNASSIGNED",
    }


def test_topic_scope_cannot_escape_routed_workspace() -> None:
    nodes = [
        _node(
            "orders",
            "detail.orders",
            "finance",
            "semantic:orders:table",
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert "EXECUTION_GRAPH_TOPIC_SCOPE_INVALID" in _codes(result)


def test_node_cannot_claim_evidence_that_was_not_read() -> None:
    nodes = [
        _node(
            "orders",
            "detail.orders",
            "orders",
            "semantic:orders:not-read",
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert "EXECUTION_GRAPH_EVIDENCE_NOT_READ" in _codes(result)


def test_every_query_node_must_bind_discovery_evidence() -> None:
    nodes = [
        GroundedExecutionNodeSpec(
            client_key="orders",
            goal_ids=["detail.orders"],
            topic_scope=["orders"],
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:refunds:metric",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert "EXECUTION_GRAPH_NODE_EVIDENCE_REQUIRED" in _codes(result)


def test_evidence_topic_must_be_inside_owning_node_scope() -> None:
    nodes = [
        _node(
            "orders",
            "detail.orders",
            "orders",
            "semantic:refunds:metric",
        ),
        _node(
            "refunds",
            ["ranking.refunds", "metric.refund_amount"],
            "refunds",
            "semantic:orders:table",
        ),
    ]

    result = _validate(_proposal(nodes=nodes))

    assert result.valid is False
    assert "EXECUTION_GRAPH_EVIDENCE_TOPIC_MISMATCH" in _codes(result)


def test_evidence_without_a_topic_cannot_bypass_node_scope() -> None:
    evidence = _evidence()
    evidence[0] = {**evidence[0], "topic": ""}
    proposal = _proposal(evidence=evidence)

    result = _validate(proposal, evidence=evidence)

    assert result.valid is False
    assert "EXECUTION_GRAPH_EVIDENCE_TOPIC_MISMATCH" in _codes(result)


def test_duplicate_discovery_ref_is_ambiguous_regardless_of_input_order() -> None:
    evidence = [
        *_evidence(),
        {
            "refId": "semantic:orders:table",
            "contentHash": "different-content",
            "topic": "refunds",
        },
    ]
    proposal = _proposal(evidence=evidence)

    first = _validate(proposal, evidence=evidence)
    second = _validate(proposal, evidence=list(reversed(evidence)))

    assert first.valid is False
    assert second.valid is False
    assert "EXECUTION_GRAPH_EVIDENCE_REF_AMBIGUOUS" in _codes(first)
    assert _codes(first) == _codes(second)


@pytest.mark.parametrize(
    "edge",
    [
        GroundedExecutionEdgeSpec(
            source_client_key="missing",
            target_client_key="refunds",
            dependency_mode="CONTRACT_SCOPE",
        ),
        GroundedExecutionEdgeSpec(
            source_client_key="orders",
            target_client_key="missing",
            dependency_mode="CONTRACT_SCOPE",
        ),
        GroundedExecutionEdgeSpec(
            source_client_key="orders",
            target_client_key="orders",
            dependency_mode="CONTRACT_SCOPE",
        ),
    ],
)
def test_unknown_or_self_edge_endpoint_fails_closed(
    edge: GroundedExecutionEdgeSpec,
) -> None:
    result = _validate(_proposal(edges=[edge]))

    assert result.valid is False
    assert "EXECUTION_GRAPH_EDGE_ENDPOINT_INVALID" in _codes(result)


def test_cycle_is_rejected() -> None:
    edges = [
        GroundedExecutionEdgeSpec(
            source_client_key="orders",
            target_client_key="refunds",
            dependency_mode="CONTRACT_SCOPE",
        ),
        GroundedExecutionEdgeSpec(
            source_client_key="refunds",
            target_client_key="orders",
            dependency_mode="CONTRACT_SCOPE",
        ),
    ]

    result = _validate(_proposal(edges=edges))

    assert result.valid is False
    assert "EXECUTION_GRAPH_CYCLE_FORBIDDEN" in _codes(result)


def test_contract_scope_edge_does_not_put_target_into_waiting_state() -> None:
    edge = GroundedExecutionEdgeSpec(
        source_client_key="orders",
        target_client_key="refunds",
        dependency_mode="CONTRACT_SCOPE",
    )
    proposal = _proposal(edges=[edge])

    result = _validate(proposal)
    receipt = build_grounded_execution_graph_receipt(proposal, version=3)

    assert result.valid is True
    assert receipt.waiting_artifact_nodes == []
    assert set(receipt.parallel_frontier) == set(receipt.node_ids.values())


def test_cross_node_population_requires_an_explicit_graph_relation() -> None:
    contract = _population_goal_contract()
    proposal = _proposal(contract=contract)

    missing = _validate(proposal, contract=contract)
    linked = _validate(
        proposal.model_copy(
            update={
                "edges": [
                    GroundedExecutionEdgeSpec(
                        source_client_key="orders",
                        target_client_key="refunds",
                        dependency_mode="CONTRACT_SCOPE",
                    )
                ]
            }
        ),
        contract=contract,
    )

    assert missing.valid is False
    assert "EXECUTION_GRAPH_REQUIRED_RELATION_MISSING" in _codes(missing)
    assert linked.valid is True


def test_population_goals_colocated_in_one_node_need_no_graph_edge() -> None:
    contract = _population_goal_contract()
    proposal = _proposal(
        contract=contract,
        nodes=[
            GroundedExecutionNodeSpec(
                client_key="combined",
                goal_ids=[
                    "detail.orders",
                    "metric.refund_amount",
                    "ranking.refunds",
                ],
                topic_scope=["orders", "refunds"],
                evidence_ref_ids=[
                    "semantic:orders:table",
                    "semantic:refunds:metric",
                ],
            )
        ],
    )

    result = _validate(proposal, contract=contract)

    assert result.valid is True


@pytest.mark.parametrize(
    ("artifact_kind", "target_binding_ref", "expected_code"),
    [
        (
            "UNVERIFIED_ROWS",
            "population.refunds",
            "EXECUTION_GRAPH_ARTIFACT_KIND_INVALID",
        ),
        (
            "VERIFIED_ENTITY_SET",
            "",
            "EXECUTION_GRAPH_ARTIFACT_TARGET_BINDING_REQUIRED",
        ),
        (
            "VERIFIED_RESULT_ARTIFACT",
            "   ",
            "EXECUTION_GRAPH_ARTIFACT_TARGET_BINDING_REQUIRED",
        ),
    ],
)
def test_verified_artifact_edge_requires_supported_kind_and_binding(
    artifact_kind: str,
    target_binding_ref: str,
    expected_code: str,
) -> None:
    edge = GroundedExecutionEdgeSpec(
        source_client_key="orders",
        target_client_key="refunds",
        dependency_mode="VERIFIED_ARTIFACT",
        artifact_kind=artifact_kind,
        target_binding_ref=target_binding_ref,
    )

    result = _validate(_proposal(edges=[edge]))

    assert result.valid is False
    assert expected_code in _codes(result)


@pytest.mark.parametrize(
    "artifact_kind",
    ["VERIFIED_ENTITY_SET", "VERIFIED_RESULT_ARTIFACT"],
)
def test_valid_verified_artifact_edge_puts_only_target_into_waiting_state(
    artifact_kind: str,
) -> None:
    edge = GroundedExecutionEdgeSpec(
        source_client_key="orders",
        target_client_key="refunds",
        dependency_mode="VERIFIED_ARTIFACT",
        artifact_kind=artifact_kind,
        target_binding_ref="population.refunds",
    )
    proposal = _proposal(edges=[edge])

    result = _validate(proposal)
    receipt = build_grounded_execution_graph_receipt(proposal, version=3)

    assert result.valid is True
    assert receipt.waiting_artifact_nodes == [receipt.node_ids["refunds"]]
    assert receipt.parallel_frontier == [receipt.node_ids["orders"]]
