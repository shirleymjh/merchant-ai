from __future__ import annotations

from merchant_ai.models import (
    PlanningAssetEntry,
    PlanningAssetPack,
    RelationshipEntry,
    TaskRole,
)
from merchant_ai.services.planning import (
    best_metric_for_domain,
    compatible_group_by,
    normalize_query_graph_payload,
)


def graph_asset_pack() -> PlanningAssetPack:
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="runtime_measurements",
                table="runtime_measurements",
                columns=["tenant_key", "entity_key", "measure_value"],
            ),
            PlanningAssetEntry(
                key="runtime_entities",
                table="runtime_entities",
                columns=["tenant_key", "entity_key", "entity_label"],
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="measurement_entity",
                left_table="runtime_measurements",
                right_table="runtime_entities",
                join_keys=[{"leftColumn": "entity_key", "rightColumn": "entity_key"}],
            )
        ],
    )


def test_graph_without_explicit_anchor_uses_unique_dependency_root_not_node_order() -> None:
    pack = graph_asset_pack()
    nodes = [
        {"nodeId": "entity_lookup", "table": "runtime_entities"},
        {"nodeId": "measure_root", "table": "runtime_measurements"},
    ]
    payload = {
        "nodes": nodes,
        "edges": [{"source": "measure_root", "target": "entity_lookup"}],
    }

    forward = normalize_query_graph_payload("runtime graph", payload, pack)
    reversed_nodes = normalize_query_graph_payload(
        "runtime graph",
        {**payload, "nodes": list(reversed(nodes))},
        pack,
    )

    assert (
        next(intent for intent in forward.intents if intent.plan_task_id == "measure_root").task_role == TaskRole.ANCHOR
    )
    assert (
        next(intent for intent in reversed_nodes.intents if intent.plan_task_id == "measure_root").task_role
        == TaskRole.ANCHOR
    )
    assert forward.compiler_trace == ["QUERY_GRAPH_ANCHOR_INFERRED_FROM_DEPENDENCIES:measure_root"]
    assert reversed_nodes.compiler_trace == forward.compiler_trace


def test_graph_without_unique_anchor_fails_closed_instead_of_selecting_first_node() -> None:
    pack = graph_asset_pack()
    nodes = [
        {"nodeId": "measure_candidate", "table": "runtime_measurements"},
        {"nodeId": "entity_candidate", "table": "runtime_entities"},
    ]

    forward = normalize_query_graph_payload("ambiguous runtime graph", {"nodes": nodes}, pack)
    reversed_nodes = normalize_query_graph_payload(
        "ambiguous runtime graph",
        {"nodes": list(reversed(nodes))},
        pack,
    )

    assert forward.intents == []
    assert reversed_nodes.intents == []
    assert forward.compiler_trace == ["QUERY_GRAPH_ANCHOR_UNRESOLVED:rootCandidates=entity_candidate,measure_candidate"]
    assert reversed_nodes.compiler_trace == forward.compiler_trace


def test_single_node_graph_has_a_unique_implicit_anchor() -> None:
    plan = normalize_query_graph_payload(
        "single runtime node",
        {"nodes": [{"nodeId": "only_node", "table": "runtime_measurements"}]},
        graph_asset_pack(),
    )

    assert len(plan.intents) == 1
    assert plan.intents[0].task_role == TaskRole.ANCHOR
    assert plan.compiler_trace == ["QUERY_GRAPH_ANCHOR_INFERRED_FROM_DEPENDENCIES:only_node"]


def metric_asset_pack(metrics: list[PlanningAssetEntry]) -> PlanningAssetPack:
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="runtime_fact",
                table="runtime_fact",
                topic="domain_alpha",
                columns=["metric_a", "metric_b"],
                metadata={"semanticDomain": "domain_alpha"},
            )
        ],
        metrics=metrics,
    )


def runtime_metrics() -> list[PlanningAssetEntry]:
    return [
        PlanningAssetEntry(
            key="metric_a",
            table="runtime_fact",
            title="noise volume",
            source_ref_id="semantic:domain_alpha:runtime_fact:metric:metric_a",
        ),
        PlanningAssetEntry(
            key="metric_b",
            table="runtime_fact",
            title="target revenue",
            source_ref_id="semantic:domain_alpha:runtime_fact:metric:metric_b",
        ),
    ]


def test_domain_metric_mismatch_fails_closed_instead_of_using_first_table_metric() -> None:
    metrics = runtime_metrics()

    forward = best_metric_for_domain("domain_beta", "runtime_fact", metric_asset_pack(metrics), "target revenue")
    reversed_assets = best_metric_for_domain(
        "domain_beta",
        "runtime_fact",
        metric_asset_pack(list(reversed(metrics))),
        "target revenue",
    )

    assert forward is None
    assert reversed_assets is None


def test_domain_metric_can_use_a_unique_question_backed_winner() -> None:
    metrics = runtime_metrics()

    forward = best_metric_for_domain("domain_alpha", "runtime_fact", metric_asset_pack(metrics), "target revenue")
    reversed_assets = best_metric_for_domain(
        "domain_alpha",
        "runtime_fact",
        metric_asset_pack(list(reversed(metrics))),
        "target revenue",
    )

    assert forward is not None and forward.key == "metric_b"
    assert reversed_assets is not None and reversed_assets.key == "metric_b"


def grouping_asset_pack(field_keys: list[str], default_group: str = "") -> PlanningAssetPack:
    table_metadata = {"defaultGroupByColumn": default_group} if default_group else {}
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="runtime_dimensions",
                table="runtime_dimensions",
                columns=["dimension_a", "dimension_b"],
                metadata=table_metadata,
            )
        ],
        fields=[
            PlanningAssetEntry(
                key=key,
                table="runtime_dimensions",
                metadata={
                    "semantic": {
                        "role": "DIMENSION",
                        "defaultVisible": True,
                        "visibilityPolicy": {"level": "public"},
                    }
                },
            )
            for key in field_keys
        ],
    )


def test_unmapped_grain_with_multiple_generic_outputs_fails_closed() -> None:
    fields = ["dimension_a", "dimension_b"]
    columns = set(fields)

    forward = compatible_group_by(
        "unknown_grain",
        columns,
        grouping_asset_pack(fields),
        "runtime_dimensions",
    )
    reversed_assets = compatible_group_by(
        "unknown_grain",
        columns,
        grouping_asset_pack(list(reversed(fields))),
        "runtime_dimensions",
    )

    assert forward == ""
    assert reversed_assets == ""


def test_unmapped_grain_can_use_one_explicit_table_grouping_contract() -> None:
    selected = compatible_group_by(
        "unknown_grain",
        {"dimension_a", "dimension_b"},
        grouping_asset_pack(["dimension_a", "dimension_b"], default_group="dimension_b"),
        "runtime_dimensions",
    )

    assert selected == "dimension_b"
