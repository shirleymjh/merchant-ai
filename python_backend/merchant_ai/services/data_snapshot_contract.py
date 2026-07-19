from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import Field

from merchant_ai.models import APIModel, QueryBundle
from merchant_ai.services.grounded_execution_graph import GroundedExecutionEdgeSpec
from merchant_ai.services.grounded_goal_contract import OriginalQuestionGoalContract


class AtomicSnapshotRequirementReason(APIModel):
    """One structural reason why separately executed results need one snapshot."""

    code: str
    query_ids: list[str] = Field(default_factory=list)
    goal_ids: list[str] = Field(default_factory=list)
    goal_kind: str = ""
    edge_index: int = -1
    dependency_mode: str = ""


class MultiQuerySnapshotRequirement(APIModel):
    """Kernel decision derived only from the frozen graph and Goal Contract."""

    require_atomic_multi_query: bool = False
    selected_query_ids: list[str] = Field(default_factory=list)
    reasons: list[AtomicSnapshotRequirementReason] = Field(default_factory=list)


def derive_multi_query_snapshot_requirement(
    selected_query_ids: Iterable[str],
    *,
    receipt_node_ids: Mapping[str, str],
    graph_edges: Iterable[GroundedExecutionEdgeSpec],
    goal_contract: OriginalQuestionGoalContract,
    goal_ids_by_query_id: Mapping[str, Iterable[str]],
) -> MultiQuerySnapshotRequirement:
    """Decide whether a selected result portfolio needs an atomic snapshot.

    The policy is deliberately structural. It does not inspect question text,
    table names, metric names, SQL, or any business vocabulary. The Core still
    chooses query topology; this helper only evaluates the topology that was
    frozen and the Goal relationships that must be preserved when results are
    composed.
    """

    selected = list(
        dict.fromkeys(
            query_id
            for raw_query_id in selected_query_ids
            if (query_id := str(raw_query_id or "").strip())
        )
    )
    if len(selected) < 2:
        return MultiQuerySnapshotRequirement(selected_query_ids=selected)

    selected_set = set(selected)
    node_to_query = {
        str(node_key or "").strip(): str(query_id or "").strip()
        for node_key, query_id in receipt_node_ids.items()
        if str(node_key or "").strip() and str(query_id or "").strip()
    }
    query_ids_by_goal_id: dict[str, set[str]] = {}
    for raw_query_id, raw_goal_ids in goal_ids_by_query_id.items():
        query_id = str(raw_query_id or "").strip()
        if query_id not in selected_set:
            continue
        for raw_goal_id in raw_goal_ids:
            goal_id = str(raw_goal_id or "").strip()
            if goal_id:
                query_ids_by_goal_id.setdefault(goal_id, set()).add(query_id)

    reasons: list[AtomicSnapshotRequirementReason] = []
    reason_keys: set[tuple[object, ...]] = set()

    def add_reason(reason: AtomicSnapshotRequirementReason) -> None:
        normalized_query_ids = sorted(set(reason.query_ids) & selected_set)
        if len(normalized_query_ids) < 2:
            return
        normalized_goal_ids = sorted(set(reason.goal_ids))
        reason.query_ids = normalized_query_ids
        reason.goal_ids = normalized_goal_ids
        key = (
            reason.code,
            tuple(normalized_query_ids),
            tuple(normalized_goal_ids),
            reason.goal_kind,
            reason.edge_index,
            reason.dependency_mode,
        )
        if key not in reason_keys:
            reason_keys.add(key)
            reasons.append(reason)

    for edge_index, edge in enumerate(graph_edges):
        if edge.dependency_mode != "CONTRACT_SCOPE":
            continue
        source_query_id = node_to_query.get(edge.source_client_key, "")
        target_query_id = node_to_query.get(edge.target_client_key, "")
        add_reason(
            AtomicSnapshotRequirementReason(
                code="ATOMIC_SNAPSHOT_CONTRACT_SCOPE_EDGE",
                query_ids=[source_query_id, target_query_id],
                edge_index=edge_index,
                dependency_mode=edge.dependency_mode,
            )
        )

    for goal in goal_contract.goals:
        goal_kind = str(goal.kind or "").upper()
        structural_input_goal_ids: list[str] = []
        reason_code = ""
        if goal_kind == "COMPARISON":
            structural_input_goal_ids = [
                *list(getattr(goal, "left_goal_ids", ()) or ()),
                *list(getattr(goal, "right_goal_ids", ()) or ()),
            ]
            reason_code = "ATOMIC_SNAPSHOT_CROSS_NODE_COMPARISON"
        elif goal_kind == "ANALYSIS":
            structural_input_goal_ids = [
                *list(getattr(goal, "input_goal_ids", ()) or ()),
                *list(getattr(goal, "baseline_goal_ids", ()) or ()),
            ]
            reason_code = "ATOMIC_SNAPSHOT_CROSS_NODE_ANALYSIS"

        if structural_input_goal_ids:
            participating_query_ids = set(
                query_ids_by_goal_id.get(goal.goal_id, set())
            )
            for input_goal_id in structural_input_goal_ids:
                participating_query_ids.update(
                    query_ids_by_goal_id.get(input_goal_id, set())
                )
            add_reason(
                AtomicSnapshotRequirementReason(
                    code=reason_code,
                    query_ids=sorted(participating_query_ids),
                    goal_ids=[goal.goal_id, *structural_input_goal_ids],
                    goal_kind=goal_kind,
                )
            )

        if goal_kind not in {"RANKING", "DETAIL"}:
            continue
        population_goal_ids = list(
            getattr(goal, "population_goal_ids", ()) or ()
        )
        population_scope = str(
            getattr(goal, "population_scope", "") or ""
        ).upper()
        if not population_goal_ids or population_scope == "ALL_MATCHING_ROWS":
            continue
        participating_query_ids = set(
            query_ids_by_goal_id.get(goal.goal_id, set())
        )
        for population_goal_id in population_goal_ids:
            participating_query_ids.update(
                query_ids_by_goal_id.get(population_goal_id, set())
            )
        add_reason(
            AtomicSnapshotRequirementReason(
                code="ATOMIC_SNAPSHOT_CROSS_NODE_POPULATION",
                query_ids=sorted(participating_query_ids),
                goal_ids=[goal.goal_id, *population_goal_ids],
                goal_kind=goal_kind,
            )
        )

    return MultiQuerySnapshotRequirement(
        require_atomic_multi_query=bool(reasons),
        selected_query_ids=selected,
        reasons=reasons,
    )


def validate_query_bundle_snapshots(
    bundles: Iterable[QueryBundle],
    *,
    require_atomic_multi_query: bool,
) -> list[str]:
    """Return stable issue codes for a composed analytical result.

    This function makes portfolio composition fail-closed without guessing
    which business tables are important. A caller deciding to combine multiple
    Doris results must explicitly state whether atomic consistency is required.
    """

    snapshots = [bundle.data_snapshot for bundle in bundles if not bundle.failed]
    if not snapshots:
        return ["DATA_SNAPSHOT_MISSING"]
    identities = [snapshot.cache_identity() for snapshot in snapshots]
    if any(not identity for identity in identities):
        return ["DATA_SNAPSHOT_UNSUPPORTED"]
    first = identities[0]
    if any(identity != first for identity in identities[1:]):
        return ["DATA_SNAPSHOT_MISMATCH"]
    if (
        require_atomic_multi_query
        and len(snapshots) > 1
        and not all(snapshot.supports_atomic_multi_query() for snapshot in snapshots)
    ):
        return ["ATOMIC_MULTI_QUERY_SNAPSHOT_UNSUPPORTED"]
    return []
