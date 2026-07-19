from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import (
    DependencyQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    original_question_goal_contract_fingerprint,
)


_QUERY_GOAL_KINDS = {
    "METRIC",
    "DIMENSION",
    "TIME_WINDOW",
    "ENTITY",
    "DETAIL",
    "RANKING",
}
_ARTIFACT_KINDS = {
    "VERIFIED_ENTITY_SET",
    "VERIFIED_RESULT_ARTIFACT",
}


class GroundedExecutionNodeSpec(APIModel):
    client_key: str
    objective: str = ""
    goal_ids: list[str] = Field(default_factory=list)
    topic_scope: list[str] = Field(default_factory=list)
    evidence_ref_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_client_key(self) -> "GroundedExecutionNodeSpec":
        key = str(self.client_key or "").strip()
        if (
            not key
            or len(key) > 128
            or any(
                not (
                    character.isascii()
                    and (character.isalnum() or character in {"_", "-"})
                )
                for character in key
            )
        ):
            raise ValueError("clientKey must be a typed ASCII identifier")
        self.client_key = key
        return self


class GroundedExecutionEdgeSpec(APIModel):
    source_client_key: str
    target_client_key: str
    dependency_mode: Literal["CONTRACT_SCOPE", "VERIFIED_ARTIFACT"]
    artifact_kind: str = ""
    target_binding_ref: str = ""

    @model_validator(mode="after")
    def normalize_identifiers(self) -> "GroundedExecutionEdgeSpec":
        self.source_client_key = str(self.source_client_key or "").strip()
        self.target_client_key = str(self.target_client_key or "").strip()
        self.artifact_kind = str(self.artifact_kind or "").strip()
        self.target_binding_ref = str(self.target_binding_ref or "").strip()
        return self


class GroundedExecutionGraphProposal(APIModel):
    graph_version: str = "grounded_execution_graph.v1"
    base_version: int = 0
    goal_contract_fingerprint: str
    discovery_snapshot_fingerprint: str
    nodes: list[GroundedExecutionNodeSpec] = Field(default_factory=list)
    edges: list[GroundedExecutionEdgeSpec] = Field(default_factory=list)


class GroundedExecutionGraphIssue(APIModel):
    code: str
    message: str
    node_key: str = ""
    edge_index: int = -1
    details: dict[str, Any] = Field(default_factory=dict)


class GroundedExecutionGraphValidation(APIModel):
    valid: bool = False
    issues: list[GroundedExecutionGraphIssue] = Field(default_factory=list)


class GroundedExecutionGraphReceipt(APIModel):
    graph_id: str
    version: int
    fingerprint: str
    discovery_snapshot_fingerprint: str
    node_ids: dict[str, str] = Field(default_factory=dict)
    parallel_frontier: list[str] = Field(default_factory=list)
    waiting_artifact_nodes: list[str] = Field(default_factory=list)
    semantic_activation_fingerprint: str = ""
    semantic_activation_seal_fingerprint: str = ""
    semantic_activation_topics: list[str] = Field(default_factory=list)


def discovery_evidence_snapshot_fingerprint(
    evidence: list[dict[str, Any]],
) -> str:
    entries = sorted(
        (
            str(item.get("refId") or ""),
            str(item.get("contentHash") or ""),
            str(item.get("topic") or ""),
        )
        for item in evidence
        if str(item.get("refId") or "")
    )
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def grounded_execution_graph_fingerprint(
    proposal: GroundedExecutionGraphProposal,
) -> str:
    payload = proposal.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_grounded_execution_graph(
    proposal: GroundedExecutionGraphProposal,
    *,
    goal_contract: OriginalQuestionGoalContract,
    discovery_evidence: list[dict[str, Any]],
    routed_topics: list[str],
    current_version: int,
) -> GroundedExecutionGraphValidation:
    issues: list[GroundedExecutionGraphIssue] = []
    expected_goal_fingerprint = original_question_goal_contract_fingerprint(
        goal_contract
    )
    expected_snapshot = discovery_evidence_snapshot_fingerprint(
        discovery_evidence
    )
    if proposal.base_version != current_version:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_VERSION_CONFLICT",
                message="baseVersion does not match the active graph version",
                details={
                    "expected": current_version,
                    "actual": proposal.base_version,
                },
            )
        )
    if proposal.goal_contract_fingerprint != expected_goal_fingerprint:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_GOAL_FINGERPRINT_MISMATCH",
                message="Execution graph is not bound to the active Goal Contract",
            )
        )
    if proposal.discovery_snapshot_fingerprint != expected_snapshot:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_DISCOVERY_SNAPSHOT_STALE",
                message="Discovery evidence changed after this graph was proposed",
                details={"expected": expected_snapshot},
            )
        )
    if not proposal.nodes:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_NODE_REQUIRED",
                message="At least one execution node is required",
            )
        )
        return GroundedExecutionGraphValidation(valid=False, issues=issues)

    node_by_key: dict[str, GroundedExecutionNodeSpec] = {}
    for node in proposal.nodes:
        if node.client_key in node_by_key:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_DUPLICATE",
                    message="Node clientKey values must be unique",
                    node_key=node.client_key,
                )
            )
            continue
        node_by_key[node.client_key] = node

    goal_map = goal_contract.goal_map()
    routed = set(routed_topics)
    evidence_by_ref: dict[str, dict[str, Any]] = {}
    ambiguous_evidence_refs: set[str] = set()
    for item in discovery_evidence:
        ref_id = str(item.get("refId") or "").strip()
        if not ref_id:
            continue
        if ref_id in evidence_by_ref:
            ambiguous_evidence_refs.add(ref_id)
            continue
        evidence_by_ref[ref_id] = item
    for ref_id in sorted(ambiguous_evidence_refs):
        evidence_by_ref.pop(ref_id, None)
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_EVIDENCE_REF_AMBIGUOUS",
                message="Discovery evidence refId values must be unique",
                details={"refId": ref_id},
            )
        )
    assigned_goal_ids: set[str] = set()
    for node in proposal.nodes:
        if not node.goal_ids:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_GOALS_REQUIRED",
                    message="Every node must cover at least one Goal",
                    node_key=node.client_key,
                )
            )
        for goal_id in node.goal_ids:
            if goal_id not in goal_map:
                issues.append(
                    GroundedExecutionGraphIssue(
                        code="EXECUTION_GRAPH_GOAL_UNKNOWN",
                        message="Node references an unknown Goal",
                        node_key=node.client_key,
                        details={"goalId": goal_id},
                    )
                )
            else:
                assigned_goal_ids.add(goal_id)
        if not node.topic_scope or any(topic not in routed for topic in node.topic_scope):
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_TOPIC_SCOPE_INVALID",
                    message="Node Topic scope must stay inside the routed workspace",
                    node_key=node.client_key,
                )
            )
        if not node.evidence_ref_ids:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_EVIDENCE_REQUIRED",
                    message="Every query node must bind evidence from Discovery",
                    node_key=node.client_key,
                )
            )
        for ref_id in node.evidence_ref_ids:
            evidence = evidence_by_ref.get(ref_id)
            if evidence is None:
                issues.append(
                    GroundedExecutionGraphIssue(
                        code="EXECUTION_GRAPH_EVIDENCE_NOT_READ",
                        message="Node references evidence absent from Discovery ledger",
                        node_key=node.client_key,
                        details={"refId": ref_id},
                    )
                )
                continue
            topic = str(evidence.get("topic") or "")
            if not topic or topic not in set(node.topic_scope):
                issues.append(
                    GroundedExecutionGraphIssue(
                        code="EXECUTION_GRAPH_EVIDENCE_TOPIC_MISMATCH",
                        message="Node evidence lies outside its Topic scope",
                        node_key=node.client_key,
                        details={"refId": ref_id, "topic": topic},
                    )
                )

    required_query_goals = {
        goal.goal_id
        for goal in goal_contract.goals
        if goal.required and str(goal.kind or "").upper() in _QUERY_GOAL_KINDS
    }
    missing_goals = sorted(required_query_goals - assigned_goal_ids)
    if missing_goals:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REQUIRED_GOALS_UNASSIGNED",
                message="Required query Goals are absent from all graph nodes",
                details={"goalIds": missing_goals},
            )
        )

    adjacency: dict[str, set[str]] = {key: set() for key in node_by_key}
    indegree: dict[str, int] = {key: 0 for key in node_by_key}
    waiting_artifact_targets: set[str] = set()
    for index, edge in enumerate(proposal.edges):
        source = edge.source_client_key
        target = edge.target_client_key
        if source not in node_by_key or target not in node_by_key or source == target:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_EDGE_ENDPOINT_INVALID",
                    message="Edge endpoints must reference two distinct graph nodes",
                    edge_index=index,
                )
            )
            continue
        if edge.dependency_mode == "VERIFIED_ARTIFACT":
            if edge.artifact_kind not in _ARTIFACT_KINDS:
                issues.append(
                    GroundedExecutionGraphIssue(
                        code="EXECUTION_GRAPH_ARTIFACT_KIND_INVALID",
                        message="Artifact edges require a supported verified artifact kind",
                        edge_index=index,
                    )
                )
            if not edge.target_binding_ref:
                issues.append(
                    GroundedExecutionGraphIssue(
                        code="EXECUTION_GRAPH_ARTIFACT_TARGET_BINDING_REQUIRED",
                        message="Artifact edges require the downstream binding ref",
                        edge_index=index,
                    )
                )
            waiting_artifact_targets.add(target)
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1

    nodes_by_goal_id: dict[str, set[str]] = {}
    for node in proposal.nodes:
        for goal_id in node.goal_ids:
            nodes_by_goal_id.setdefault(goal_id, set()).add(node.client_key)

    def has_graph_path(source_key: str, target_key: str) -> bool:
        pending = [source_key]
        visited = {source_key}
        cursor = 0
        while cursor < len(pending):
            current = pending[cursor]
            cursor += 1
            for candidate in adjacency.get(current, set()):
                if candidate == target_key:
                    return True
                if candidate in visited:
                    continue
                visited.add(candidate)
                pending.append(candidate)
        return False

    required_relations: list[tuple[str, str, str]] = []
    for goal in goal_contract.goals:
        if isinstance(goal, RankingQuestionGoal) and (
            goal.population_scope != "ALL_MATCHING_ROWS"
        ):
            required_relations.extend(
                (
                    population_goal_id,
                    goal.goal_id,
                    "RANKING_POPULATION",
                )
                for population_goal_id in goal.population_goal_ids
            )
        if isinstance(goal, DependencyQuestionGoal):
            required_relations.extend(
                (
                    upstream_goal_id,
                    downstream_goal_id,
                    "DEPENDENCY_GOAL",
                )
                for upstream_goal_id in goal.upstream_goal_ids
                for downstream_goal_id in goal.downstream_goal_ids
            )

    missing_relation_keys: set[tuple[str, str, str, str]] = set()
    for source_goal_id, target_goal_id, relation_kind in required_relations:
        source_node_keys = nodes_by_goal_id.get(source_goal_id, set())
        target_node_keys = nodes_by_goal_id.get(target_goal_id, set())
        for target_node_key in target_node_keys:
            if target_node_key in source_node_keys:
                continue
            if any(
                has_graph_path(source_node_key, target_node_key)
                for source_node_key in source_node_keys
            ):
                continue
            issue_key = (
                source_goal_id,
                target_goal_id,
                target_node_key,
                relation_kind,
            )
            if issue_key in missing_relation_keys:
                continue
            missing_relation_keys.add(issue_key)
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REQUIRED_RELATION_MISSING",
                    message=(
                        "Goals assigned to separate query nodes require an "
                        "explicit directed execution-graph relation."
                    ),
                    node_key=target_node_key,
                    details={
                        "relationKind": relation_kind,
                        "sourceGoalId": source_goal_id,
                        "targetGoalId": target_goal_id,
                        "sourceNodeKeys": sorted(source_node_keys),
                        "targetNodeKey": target_node_key,
                    },
                )
            )

    ready = sorted(key for key, degree in indegree.items() if degree == 0)
    visited = 0
    cursor = 0
    while cursor < len(ready):
        current = ready[cursor]
        cursor += 1
        visited += 1
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(node_by_key):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_CYCLE_FORBIDDEN",
                message="Execution graph must be acyclic",
            )
        )

    return GroundedExecutionGraphValidation(valid=not issues, issues=issues)


def build_grounded_execution_graph_receipt(
    proposal: GroundedExecutionGraphProposal,
    *,
    version: int,
    semantic_activation_fingerprint: str = "",
    semantic_activation_seal_fingerprint: str = "",
    semantic_activation_topics: list[str] | None = None,
) -> GroundedExecutionGraphReceipt:
    fingerprint = grounded_execution_graph_fingerprint(proposal)
    node_ids = {
        node.client_key: "node_%s"
        % hashlib.sha256(
            (fingerprint + ":" + node.client_key).encode("utf-8")
        ).hexdigest()[:16]
        for node in proposal.nodes
    }
    waiting = {
        edge.target_client_key
        for edge in proposal.edges
        if edge.dependency_mode == "VERIFIED_ARTIFACT"
    }
    return GroundedExecutionGraphReceipt(
        graph_id="graph_%s" % fingerprint[:16],
        version=version,
        fingerprint=fingerprint,
        discovery_snapshot_fingerprint=proposal.discovery_snapshot_fingerprint,
        node_ids=node_ids,
        parallel_frontier=[
            node_ids[node.client_key]
            for node in proposal.nodes
            if node.client_key not in waiting
        ],
        waiting_artifact_nodes=[node_ids[key] for key in sorted(waiting)],
        semantic_activation_fingerprint=str(
            semantic_activation_fingerprint or ""
        ),
        semantic_activation_seal_fingerprint=str(
            semantic_activation_seal_fingerprint or ""
        ),
        semantic_activation_topics=list(
            semantic_activation_topics or []
        ),
    )
