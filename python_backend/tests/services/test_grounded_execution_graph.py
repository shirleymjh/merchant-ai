from __future__ import annotations

from collections.abc import Iterable

import pytest

from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionEdgeSpec,
    GroundedExecutionGraphProposal,
    GroundedExecutionGraphRevisionProposal,
    GroundedExecutionGraphNodeRuntimeState,
    GroundedExecutionNodeSpec,
    build_grounded_execution_graph_replan_evidence,
    build_grounded_execution_graph_receipt,
    discovery_evidence_snapshot_fingerprint,
    validate_grounded_execution_graph,
    validate_grounded_execution_graph_revision,
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
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(active_contract),
        discovery_snapshot_fingerprint=discovery_evidence_snapshot_fingerprint(active_evidence),
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
    assert receipt.discovery_snapshot_fingerprint == (proposal.discovery_snapshot_fingerprint)


def _revision_validation(
    *,
    active: GroundedExecutionGraphProposal,
    revised: GroundedExecutionGraphProposal,
    evidence: list[dict[str, str]],
    states: list[GroundedExecutionGraphNodeRuntimeState],
    trigger_query_key: str,
    trigger_kind: str = "DATA_GAP",
    replace_keys: list[str] | None = None,
    used_trigger_fingerprints: list[str] | None = None,
    completed_revision_count: int = 0,
    contract: OriginalQuestionGoalContract | None = None,
):
    active_receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    trigger = build_grounded_execution_graph_replan_evidence(
        trigger_kind=trigger_kind,
        source_stage=(
            "CONTRACT" if trigger_kind == "DATA_GAP" else "DATASOURCE" if trigger_kind == "TABLE_DELAY" else "EXECUTION"
        ),
        source_query_node_id=active_receipt.node_ids[trigger_query_key],
        code="STRUCTURED_TEST_TRIGGER",
        graph_receipt=active_receipt,
        details={"gapCodes": ["STRUCTURED_GAP"]},
    )
    revision = GroundedExecutionGraphRevisionProposal(
        base_graph_id=active_receipt.graph_id,
        base_version=active_receipt.version,
        base_fingerprint=active_receipt.fingerprint,
        trigger_evidence_id=trigger.evidence_id,
        trigger_evidence_fingerprint=trigger.evidence_fingerprint,
        replace_unexecuted_client_keys=replace_keys or [],
        graph=revised,
    )
    validation = validate_grounded_execution_graph_revision(
        revision,
        active_proposal=active,
        active_receipt=active_receipt,
        trigger_evidence=trigger,
        node_states=states,
        goal_contract=contract or _goal_contract(),
        discovery_evidence=evidence,
        routed_topics=["orders", "refunds"],
        used_trigger_fingerprints=(used_trigger_fingerprints or []),
        completed_revision_count=completed_revision_count,
        max_revision_count=2,
    )
    return validation, active_receipt, trigger, revision


def test_revision_appends_node_and_preserves_published_node_identity() -> None:
    active = _proposal(base_version=2)
    revised = _proposal(
        base_version=3,
        nodes=[
            *active.nodes,
            _node(
                "recovery",
                "metric.refund_amount",
                "refunds",
                "semantic:refunds:metric",
            ),
        ],
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key="orders",
            query_node_id=build_grounded_execution_graph_receipt(
                active,
                version=3,
            ).node_ids["orders"],
            lifecycle="PUBLISHED",
        ),
        GroundedExecutionGraphNodeRuntimeState(
            client_key="refunds",
            query_node_id=build_grounded_execution_graph_receipt(
                active,
                version=3,
            ).node_ids["refunds"],
            lifecycle="UNEXECUTED",
        ),
    ]

    validation, active_receipt, trigger, revision = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
    )

    assert validation.valid is True
    assert validation.added_client_keys == ["recovery"]
    receipt = build_grounded_execution_graph_receipt(
        revision.graph,
        version=4,
        parent_receipt=active_receipt,
        replan_evidence_fingerprint=trigger.evidence_fingerprint,
        preserved_node_ids={key: active_receipt.node_ids[key] for key in validation.carried_forward_client_keys},
    )
    assert receipt.node_ids["orders"] == active_receipt.node_ids["orders"]
    assert receipt.node_ids["refunds"] == active_receipt.node_ids["refunds"]
    assert receipt.node_ids["recovery"] not in set(active_receipt.node_ids.values())
    assert receipt.parent_version == 3
    assert receipt.replan_evidence_fingerprint == (trigger.evidence_fingerprint)
    assert receipt.revision_fingerprint


def test_revision_cannot_mutate_published_node_or_its_input_lineage() -> None:
    edge = GroundedExecutionEdgeSpec(
        source_client_key="orders",
        target_client_key="refunds",
        dependency_mode="CONTRACT_SCOPE",
    )
    active = _proposal(base_version=2, edges=[edge])
    mutated_orders = active.nodes[0].model_copy(update={"objective": "changed published objective"})
    revised = _proposal(
        base_version=3,
        nodes=[mutated_orders, active.nodes[1]],
        edges=[],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_id,
            lifecycle=("PUBLISHED" if key == "orders" else "UNEXECUTED"),
        )
        for key, query_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
    )

    assert validation.valid is False
    assert "EXECUTION_GRAPH_NODE_MUTATION_FORBIDDEN" in _codes(validation)
    assert "EXECUTION_GRAPH_INPUT_LINEAGE_MUTATION_FORBIDDEN" in _codes(validation)


def test_revision_cannot_add_unrelated_published_lineage_to_recovery_node() -> None:
    active = _proposal(base_version=2)
    recovery = _node(
        "refunds-recovery",
        ["ranking.refunds", "metric.refund_amount"],
        "refunds",
        "semantic:refunds:metric",
    )
    revised = _proposal(
        base_version=3,
        nodes=[*active.nodes, recovery],
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds-recovery",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_node_id,
            lifecycle=(
                "PUBLISHED"
                if key == "orders"
                else "UNEXECUTED"
            ),
        )
        for key, query_node_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
    )

    assert validation.valid is False
    assert (
        "EXECUTION_GRAPH_ADDED_NODE_LINEAGE_OUTSIDE_TRIGGER_SCOPE"
        in _codes(validation)
    )


def test_revision_rejects_mixed_authorized_and_unrelated_source_lineage() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": "Rank scoped orders without unrelated revenue lineage",
            "goals": [
                {
                    "goalId": "detail.orders",
                    "kind": "DETAIL",
                    "label": "scoped orders",
                },
                {
                    "goalId": "metric.refund_amount",
                    "kind": "METRIC",
                    "label": "refund amount",
                },
                {
                    "goalId": "metric.unrelated_revenue",
                    "kind": "METRIC",
                    "label": "unrelated revenue",
                },
                {
                    "goalId": "ranking.refunds",
                    "kind": "RANKING",
                    "label": "refund ranking within scoped orders",
                    "metricGoalIds": ["metric.refund_amount"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                    "populationGoalIds": ["detail.orders"],
                },
            ],
        }
    )
    active = _proposal(
        contract=contract,
        base_version=2,
        nodes=[
            _node(
                "orders",
                ["detail.orders", "metric.unrelated_revenue"],
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
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds",
                dependency_mode="VERIFIED_ARTIFACT",
                artifact_kind="VERIFIED_ENTITY_SET",
                target_binding_ref="semantic:refunds:population-binding",
            )
        ],
    )
    recovery = _node(
        "refunds-recovery",
        ["ranking.refunds", "metric.refund_amount"],
        "refunds",
        "semantic:refunds:metric",
    )
    revised = _proposal(
        contract=contract,
        base_version=3,
        nodes=[active.nodes[0], recovery],
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds-recovery",
                dependency_mode="VERIFIED_ARTIFACT",
                artifact_kind="VERIFIED_ENTITY_SET",
                target_binding_ref="semantic:refunds:population-binding",
            )
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_node_id,
            lifecycle=(
                "PUBLISHED"
                if key == "orders"
                else "EXECUTION_FAILED"
            ),
        )
        for key, query_node_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
        trigger_kind="EXECUTION_ERROR",
        contract=contract,
    )

    assert validation.valid is False
    assert (
        "EXECUTION_GRAPH_ADDED_NODE_LINEAGE_OUTSIDE_TRIGGER_SCOPE"
        in _codes(validation)
    )


def test_revision_can_reconnect_goal_authorized_population_lineage_to_recovery() -> None:
    contract = _population_goal_contract()
    active = _proposal(
        contract=contract,
        base_version=2,
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds",
                dependency_mode="VERIFIED_ARTIFACT",
                artifact_kind="VERIFIED_ENTITY_SET",
                target_binding_ref="semantic:refunds:population-binding",
            )
        ],
    )
    recovery = _node(
        "refunds-recovery",
        ["ranking.refunds", "metric.refund_amount"],
        "refunds",
        "semantic:refunds:metric",
    )
    revised = _proposal(
        contract=contract,
        base_version=3,
        nodes=[active.nodes[0], recovery],
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds-recovery",
                dependency_mode="VERIFIED_ARTIFACT",
                artifact_kind="VERIFIED_ENTITY_SET",
                target_binding_ref="semantic:refunds:population-binding",
            )
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_node_id,
            lifecycle=(
                "PUBLISHED"
                if key == "orders"
                else "EXECUTION_FAILED"
            ),
        )
        for key, query_node_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
        trigger_kind="EXECUTION_ERROR",
        contract=contract,
    )

    assert validation.valid is True
    assert validation.retired_failed_client_keys == ["refunds"]
    assert validation.added_client_keys == ["refunds-recovery"]


def test_revision_replaces_only_bound_unexecuted_downstream() -> None:
    extra_evidence = [
        *_evidence(),
        {
            "refId": "semantic:refunds:fallback",
            "contentHash": "refund-fallback-hash",
            "topic": "refunds",
        },
    ]
    active = _proposal(
        evidence=extra_evidence,
        base_version=2,
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
    )
    replacement = _node(
        "refunds",
        ["ranking.refunds", "metric.refund_amount"],
        "refunds",
        "semantic:refunds:fallback",
    )
    revised = _proposal(
        evidence=extra_evidence,
        base_version=3,
        nodes=[active.nodes[0], replacement],
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="orders",
                target_client_key="refunds",
                dependency_mode="CONTRACT_SCOPE",
            )
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_id,
            lifecycle=("PUBLISHED" if key == "orders" else "UNEXECUTED"),
        )
        for key, query_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=extra_evidence,
        states=states,
        trigger_query_key="refunds",
        replace_keys=["refunds"],
    )

    assert validation.valid is True
    assert validation.replaced_client_keys == ["refunds"]
    assert validation.carried_forward_client_keys == ["orders"]


def test_revision_rejects_trigger_replay_budget_and_executed_replacement() -> None:
    active = _proposal(base_version=2)
    revised = _proposal(
        base_version=3,
        nodes=[
            active.nodes[0],
            active.nodes[1].model_copy(update={"objective": "changed"}),
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_id,
            lifecycle="PUBLISHED",
        )
        for key, query_id in receipt.node_ids.items()
    ]
    _, _, trigger, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
        replace_keys=["refunds"],
    )

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
        replace_keys=["refunds"],
        used_trigger_fingerprints=[trigger.evidence_fingerprint],
        completed_revision_count=2,
    )

    assert validation.valid is False
    assert {
        "EXECUTION_GRAPH_EXECUTED_NODE_REPLACEMENT_FORBIDDEN",
        "EXECUTION_GRAPH_REPLAN_BUDGET_EXHAUSTED",
        "EXECUTION_GRAPH_REPLAN_TRIGGER_REPLAYED",
    }.issubset(_codes(validation))


@pytest.mark.parametrize(
    "trigger_kind",
    ["TABLE_DELAY", "EXECUTION_ERROR"],
)
def test_failed_executed_node_is_historical_and_requires_appended_recovery(
    trigger_kind: str,
) -> None:
    active = _proposal(base_version=2)
    revised = _proposal(
        base_version=3,
        nodes=[
            active.nodes[0],
            _node(
                "refunds-recovery",
                ["ranking.refunds", "metric.refund_amount"],
                "refunds",
                "semantic:refunds:metric",
            ),
        ],
    )
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_id,
            lifecycle=("EXECUTION_FAILED" if key == "refunds" else "PUBLISHED"),
        )
        for key, query_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
        trigger_kind=trigger_kind,
    )

    assert validation.valid is True
    assert validation.retired_failed_client_keys == ["refunds"]
    assert validation.added_client_keys == ["refunds-recovery"]


def test_revision_rejects_metadata_only_no_op() -> None:
    active = _proposal(base_version=2)
    revised = active.model_copy(update={"base_version": 3})
    receipt = build_grounded_execution_graph_receipt(
        active,
        version=3,
    )
    states = [
        GroundedExecutionGraphNodeRuntimeState(
            client_key=key,
            query_node_id=query_id,
            lifecycle="UNEXECUTED",
        )
        for key, query_id in receipt.node_ids.items()
    ]

    validation, _, _, _ = _revision_validation(
        active=active,
        revised=revised,
        evidence=_evidence(),
        states=states,
        trigger_query_key="refunds",
    )

    assert validation.valid is False
    assert "EXECUTION_GRAPH_REPLAN_NO_CHANGE" in _codes(validation)


def test_goal_contract_fingerprint_mismatch_fails_closed() -> None:
    proposal = _proposal().model_copy(update={"goal_contract_fingerprint": "wrong-goal-fingerprint"})

    result = _validate(proposal)

    assert result.valid is False
    assert "EXECUTION_GRAPH_GOAL_FINGERPRINT_MISMATCH" in _codes(result)


def test_discovery_snapshot_staleness_fails_closed() -> None:
    proposal = _proposal().model_copy(update={"discovery_snapshot_fingerprint": "stale-discovery-snapshot"})

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
    missing = next(issue for issue in result.issues if issue.code == "EXECUTION_GRAPH_REQUIRED_GOALS_UNASSIGNED")
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
