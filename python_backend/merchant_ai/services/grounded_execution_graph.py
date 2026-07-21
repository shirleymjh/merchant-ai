from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

from pydantic import Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import (
    DependencyQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    goal_dependency_closure,
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
    "VERIFIED_SCALAR",
    "VERIFIED_BASELINE",
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
            or any(not (character.isascii() and (character.isalnum() or character in {"_", "-"})) for character in key)
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
    parent_graph_id: str = ""
    parent_version: int = 0
    parent_fingerprint: str = ""
    replan_evidence_fingerprint: str = ""
    replan_evidence_fingerprints: list[str] = Field(default_factory=list)
    carried_forward_node_ids: list[str] = Field(default_factory=list)
    retired_node_ids: list[str] = Field(default_factory=list)
    revision_fingerprint: str = ""


class GroundedExecutionGraphReplanEvidence(APIModel):
    evidence_version: Literal["grounded_execution_graph_replan_evidence.v1"] = (
        "grounded_execution_graph_replan_evidence.v1"
    )
    evidence_id: str
    trigger_kind: Literal[
        "DATA_GAP",
        "TABLE_DELAY",
        "EXECUTION_ERROR",
    ]
    source_stage: Literal[
        "CONTRACT",
        "DATASOURCE",
        "EXECUTION",
    ]
    source_query_node_id: str
    code: str
    graph_id: str
    graph_version: int = Field(ge=1)
    graph_fingerprint: str
    details: dict[str, Any] = Field(default_factory=dict)
    details_fingerprint: str
    evidence_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_identity(
        self,
    ) -> "GroundedExecutionGraphReplanEvidence":
        for field_name in (
            "evidence_id",
            "source_query_node_id",
            "code",
            "graph_id",
            "graph_fingerprint",
            "details_fingerprint",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError("%s must not be empty" % field_name)
        return self


class GroundedExecutionGraphNodeRuntimeState(APIModel):
    client_key: str
    query_node_id: str
    lifecycle: Literal[
        "UNEXECUTED",
        "PRE_AUTHORIZED",
        "PUBLISHED",
        "EXECUTION_FAILED",
    ]


class GroundedExecutionGraphTriggerBinding(APIModel):
    evidence_id: str
    evidence_fingerprint: str

    @model_validator(mode="after")
    def validate_identity(
        self,
    ) -> "GroundedExecutionGraphTriggerBinding":
        if not str(self.evidence_id or "").strip():
            raise ValueError("evidenceId must not be empty")
        if not str(self.evidence_fingerprint or "").strip():
            raise ValueError("evidenceFingerprint must not be empty")
        return self


class GroundedExecutionGraphRevisionProposal(APIModel):
    revision_version: Literal["grounded_execution_graph_revision.v1"] = "grounded_execution_graph_revision.v1"
    base_graph_id: str
    base_version: int = Field(ge=1)
    base_fingerprint: str
    trigger_evidence_id: str = ""
    trigger_evidence_fingerprint: str = ""
    trigger_evidence_set: list[
        GroundedExecutionGraphTriggerBinding
    ] = Field(default_factory=list)
    replace_unexecuted_client_keys: list[str] = Field(default_factory=list)
    graph: GroundedExecutionGraphProposal

    @model_validator(mode="after")
    def validate_trigger_binding(
        self,
    ) -> "GroundedExecutionGraphRevisionProposal":
        bindings = list(self.trigger_evidence_set)
        legacy_id = str(self.trigger_evidence_id or "").strip()
        legacy_fingerprint = str(
            self.trigger_evidence_fingerprint or ""
        ).strip()
        if not bindings:
            if not legacy_id or not legacy_fingerprint:
                raise ValueError("a trigger evidence binding is required")
            bindings = [
                GroundedExecutionGraphTriggerBinding(
                    evidence_id=legacy_id,
                    evidence_fingerprint=legacy_fingerprint,
                )
            ]
            self.trigger_evidence_set = bindings
        elif legacy_id or legacy_fingerprint:
            if (
                len(bindings) != 1
                or bindings[0].evidence_id != legacy_id
                or bindings[0].evidence_fingerprint
                != legacy_fingerprint
            ):
                raise ValueError(
                    "legacy trigger fields conflict with triggerEvidenceSet"
                )
        evidence_ids = [item.evidence_id for item in bindings]
        evidence_fingerprints = [
            item.evidence_fingerprint for item in bindings
        ]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("trigger evidence ids must be unique")
        if len(set(evidence_fingerprints)) != len(evidence_fingerprints):
            raise ValueError("trigger evidence fingerprints must be unique")
        return self


class GroundedExecutionGraphRevisionValidation(APIModel):
    valid: bool = False
    issues: list[GroundedExecutionGraphIssue] = Field(default_factory=list)
    carried_forward_client_keys: list[str] = Field(default_factory=list)
    replaced_client_keys: list[str] = Field(default_factory=list)
    added_client_keys: list[str] = Field(default_factory=list)
    retired_failed_client_keys: list[str] = Field(default_factory=list)


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


def _grounded_stable_fingerprint(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def grounded_execution_graph_replan_evidence_fingerprint(
    evidence: GroundedExecutionGraphReplanEvidence,
) -> str:
    payload = evidence.model_dump(by_alias=True, mode="json")
    payload["evidenceFingerprint"] = ""
    return _grounded_stable_fingerprint(payload)


def grounded_execution_graph_replan_evidence_set_fingerprint(
    evidences: Sequence[GroundedExecutionGraphReplanEvidence],
) -> str:
    """Seal an order-independent set of triggers for one graph revision."""

    normalized = sorted(
        (
            evidence.evidence_id,
            evidence.evidence_fingerprint,
            evidence.graph_id,
            evidence.graph_version,
            evidence.graph_fingerprint,
        )
        for evidence in evidences
    )
    if not normalized:
        raise ValueError("at least one replan evidence item is required")
    if len({item[0] for item in normalized}) != len(normalized):
        raise ValueError("replan evidence ids must be unique")
    if len({item[1] for item in normalized}) != len(normalized):
        raise ValueError("replan evidence fingerprints must be unique")
    return _grounded_stable_fingerprint(
        {
            "evidenceSetVersion": (
                "grounded_execution_graph_replan_evidence_set.v1"
            ),
            "evidences": normalized,
        }
    )


def seal_grounded_execution_graph_replan_evidence(
    evidence: GroundedExecutionGraphReplanEvidence,
) -> GroundedExecutionGraphReplanEvidence:
    return evidence.model_copy(
        update={"evidence_fingerprint": (grounded_execution_graph_replan_evidence_fingerprint(evidence))},
        deep=True,
    )


def build_grounded_execution_graph_replan_evidence(
    *,
    trigger_kind: Literal[
        "DATA_GAP",
        "TABLE_DELAY",
        "EXECUTION_ERROR",
    ],
    source_stage: Literal[
        "CONTRACT",
        "DATASOURCE",
        "EXECUTION",
    ],
    source_query_node_id: str,
    code: str,
    graph_receipt: GroundedExecutionGraphReceipt,
    details: dict[str, Any],
) -> GroundedExecutionGraphReplanEvidence:
    details_copy = json.loads(
        json.dumps(
            details,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    details_fingerprint = _grounded_stable_fingerprint(details_copy)
    evidence_id = (
        "graph_replan_evidence_%s"
        % _grounded_stable_fingerprint(
            {
                "triggerKind": trigger_kind,
                "sourceStage": source_stage,
                "sourceQueryNodeId": source_query_node_id,
                "code": code,
                "graphId": graph_receipt.graph_id,
                "graphVersion": graph_receipt.version,
                "graphFingerprint": graph_receipt.fingerprint,
            }
        )[:24]
    )
    return seal_grounded_execution_graph_replan_evidence(
        GroundedExecutionGraphReplanEvidence(
            evidence_id=evidence_id,
            trigger_kind=trigger_kind,
            source_stage=source_stage,
            source_query_node_id=source_query_node_id,
            code=str(code or "").strip(),
            graph_id=graph_receipt.graph_id,
            graph_version=graph_receipt.version,
            graph_fingerprint=graph_receipt.fingerprint,
            details=details_copy,
            details_fingerprint=details_fingerprint,
        )
    )


def _execution_node_spec_fingerprint(
    node: GroundedExecutionNodeSpec,
) -> str:
    return _grounded_stable_fingerprint(node.model_dump(by_alias=True, mode="json"))


def _execution_edge_signature(
    edge: GroundedExecutionEdgeSpec,
) -> tuple[str, str, str, str, str]:
    return (
        edge.source_client_key,
        edge.target_client_key,
        edge.dependency_mode,
        edge.artifact_kind,
        edge.target_binding_ref,
    )


def validate_grounded_execution_graph_revision(
    revision: GroundedExecutionGraphRevisionProposal,
    *,
    active_proposal: GroundedExecutionGraphProposal,
    active_receipt: GroundedExecutionGraphReceipt,
    trigger_evidence: GroundedExecutionGraphReplanEvidence
    | Sequence[GroundedExecutionGraphReplanEvidence],
    node_states: Sequence[GroundedExecutionGraphNodeRuntimeState],
    goal_contract: OriginalQuestionGoalContract,
    discovery_evidence: list[dict[str, Any]],
    routed_topics: list[str],
    used_trigger_fingerprints: Sequence[str] = (),
    completed_revision_count: int = 0,
    max_revision_count: int = 2,
) -> GroundedExecutionGraphRevisionValidation:
    issues: list[GroundedExecutionGraphIssue] = []
    trigger_evidences = (
        tuple(trigger_evidence)
        if isinstance(trigger_evidence, Sequence)
        else (trigger_evidence,)
    )
    if not trigger_evidences:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_EVIDENCE_REQUIRED",
                message="At least one structured trigger is required.",
            )
        )
    if completed_revision_count >= max(1, int(max_revision_count)):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_BUDGET_EXHAUSTED",
                message="The graph revision budget is exhausted.",
            )
        )
    evidence_ids = [item.evidence_id for item in trigger_evidences]
    evidence_fingerprints = [
        item.evidence_fingerprint for item in trigger_evidences
    ]
    if (
        len(set(evidence_ids)) != len(evidence_ids)
        or len(set(evidence_fingerprints)) != len(evidence_fingerprints)
    ):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_EVIDENCE_SET_DUPLICATE",
                message="A revision evidence set must contain unique items.",
            )
        )
    used_fingerprints = set(used_trigger_fingerprints)
    for evidence in trigger_evidences:
        if (
            not evidence.evidence_fingerprint
            or evidence.evidence_fingerprint
            != grounded_execution_graph_replan_evidence_fingerprint(
                evidence
            )
            or evidence.details_fingerprint
            != _grounded_stable_fingerprint(evidence.details)
        ):
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REPLAN_EVIDENCE_INVALID",
                    message="The structured replan evidence is not sealed.",
                    details={"evidenceId": evidence.evidence_id},
                )
            )
        if evidence.evidence_fingerprint in used_fingerprints:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REPLAN_TRIGGER_REPLAYED",
                    message=(
                        "One structured trigger can revise the graph once."
                    ),
                    details={"evidenceId": evidence.evidence_id},
                )
            )
    proposed_bindings = {
        item.evidence_id: item.evidence_fingerprint
        for item in revision.trigger_evidence_set
    }
    selected_bindings = {
        item.evidence_id: item.evidence_fingerprint
        for item in trigger_evidences
    }
    if proposed_bindings != selected_bindings:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_TRIGGER_MISMATCH",
                message=(
                    "The revision is not bound to the selected trigger set."
                ),
            )
        )
    base_identity = (
        revision.base_graph_id,
        revision.base_version,
        revision.base_fingerprint,
    )
    active_identity = (
        active_receipt.graph_id,
        active_receipt.version,
        active_receipt.fingerprint,
    )
    if base_identity != active_identity:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_CAS_CONFLICT",
                message="The revision base is not the active graph receipt.",
                details={
                    "activeGraphId": active_receipt.graph_id,
                    "activeVersion": active_receipt.version,
                },
            )
        )
    for evidence in trigger_evidences:
        evidence_identity = (
            evidence.graph_id,
            evidence.graph_version,
            evidence.graph_fingerprint,
        )
        if evidence_identity != active_identity:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REPLAN_EVIDENCE_STALE",
                    message=(
                        "The trigger belongs to another graph revision."
                    ),
                    details={"evidenceId": evidence.evidence_id},
                )
            )
    if grounded_execution_graph_fingerprint(active_proposal) != active_receipt.fingerprint:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_ACTIVE_PROPOSAL_INVALID",
                message="The active graph proposal no longer matches its receipt.",
            )
        )

    base_validation = validate_grounded_execution_graph(
        revision.graph,
        goal_contract=goal_contract,
        discovery_evidence=discovery_evidence,
        routed_topics=routed_topics,
        current_version=active_receipt.version,
    )
    issues.extend(base_validation.issues)

    old_nodes = {node.client_key: node for node in active_proposal.nodes}
    new_nodes = {node.client_key: node for node in revision.graph.nodes}
    states_by_key: dict[
        str,
        GroundedExecutionGraphNodeRuntimeState,
    ] = {}
    for state in node_states:
        if state.client_key in states_by_key:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_STATE_DUPLICATE",
                    message="A graph node has multiple runtime states.",
                    node_key=state.client_key,
                )
            )
            continue
        states_by_key[state.client_key] = state
        expected_query_id = active_receipt.node_ids.get(state.client_key)
        if expected_query_id != state.query_node_id:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_STATE_IDENTITY_MISMATCH",
                    message="A runtime state belongs to another graph node.",
                    node_key=state.client_key,
                )
            )
    missing_states = sorted(set(old_nodes) - set(states_by_key))
    if missing_states:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_NODE_STATE_INCOMPLETE",
                message="Every active graph node requires server runtime state.",
                details={"clientKeys": missing_states},
            )
        )

    replacement_keys = set(revision.replace_unexecuted_client_keys)
    if len(replacement_keys) != len(revision.replace_unexecuted_client_keys):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLACEMENT_DUPLICATE",
                message="Replacement client keys must be unique.",
            )
        )
    unknown_replacements = sorted(replacement_keys - set(old_nodes))
    if unknown_replacements:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLACEMENT_UNKNOWN",
                message="A replacement key is absent from the active graph.",
                details={"clientKeys": unknown_replacements},
            )
        )
    for key in sorted(replacement_keys.intersection(states_by_key)):
        if states_by_key[key].lifecycle != "UNEXECUTED":
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_EXECUTED_NODE_REPLACEMENT_FORBIDDEN",
                    message="Only an unexecuted node may be replaced.",
                    node_key=key,
                )
            )

    client_key_by_query_id = {
        query_node_id: key
        for key, query_node_id in active_receipt.node_ids.items()
    }
    source_client_keys: set[str] = set()
    trigger_by_source_key: dict[
        str,
        list[GroundedExecutionGraphReplanEvidence],
    ] = {}
    for evidence in trigger_evidences:
        source_client_key = client_key_by_query_id.get(
            evidence.source_query_node_id,
            "",
        )
        if not source_client_key:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REPLAN_SOURCE_NODE_UNKNOWN",
                    message=(
                        "A trigger source is absent from the active graph."
                    ),
                    details={"evidenceId": evidence.evidence_id},
                )
            )
            continue
        source_client_keys.add(source_client_key)
        trigger_by_source_key.setdefault(source_client_key, []).append(
            evidence
        )

    adjacency: dict[str, set[str]] = {key: set() for key in old_nodes}
    for edge in active_proposal.edges:
        adjacency.setdefault(edge.source_client_key, set()).add(edge.target_client_key)

    def is_downstream(source: str, target: str) -> bool:
        if source == target:
            return True
        pending = [source]
        visited = {source}
        cursor = 0
        while cursor < len(pending):
            current = pending[cursor]
            cursor += 1
            for candidate in adjacency.get(current, set()):
                if candidate == target:
                    return True
                if candidate in visited:
                    continue
                visited.add(candidate)
                pending.append(candidate)
        return False

    for key in sorted(replacement_keys):
        if source_client_keys and not any(
            is_downstream(source_client_key, key)
            for source_client_key in source_client_keys
        ):
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_REPLACEMENT_NOT_DOWNSTREAM",
                    message=(
                        "A replacement must be a selected trigger node or "
                        "one of its downstream nodes."
                    ),
                    node_key=key,
                )
            )

    removed_keys = set(old_nodes) - set(new_nodes)
    changed_keys = {
        key
        for key in set(old_nodes).intersection(new_nodes)
        if _execution_node_spec_fingerprint(old_nodes[key]) != _execution_node_spec_fingerprint(new_nodes[key])
    }
    failed_retirements: set[str] = set()
    for key in sorted(removed_keys):
        state = states_by_key.get(key)
        is_bound_failed_source = bool(
            state is not None
            and state.lifecycle == "EXECUTION_FAILED"
            and key in source_client_keys
            and trigger_by_source_key.get(key)
        )
        if is_bound_failed_source:
            failed_retirements.add(key)
            continue
        if key not in replacement_keys:
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_NODE_REMOVAL_FORBIDDEN",
                    message="Only an explicitly replaced unexecuted node may be removed.",
                    node_key=key,
                )
            )
    required_failed_retirements = {
        key
        for key in source_client_keys
        if states_by_key.get(key) is not None
        and states_by_key[key].lifecycle == "EXECUTION_FAILED"
    }
    missing_failed_retirements = sorted(
        required_failed_retirements - failed_retirements
    )
    if missing_failed_retirements:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_FAILED_TRIGGER_SET_RETIREMENT_REQUIRED",
                message=(
                    "Every selected failed source must be retired in the "
                    "same graph revision."
                ),
                details={"clientKeys": missing_failed_retirements},
            )
        )
    for key in sorted(changed_keys - replacement_keys):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_NODE_MUTATION_FORBIDDEN",
                message="An existing node cannot change unless it is an unexecuted replacement.",
                node_key=key,
            )
        )
    ineffective_replacements = sorted(replacement_keys - removed_keys - changed_keys)
    if ineffective_replacements:
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLACEMENT_NO_CHANGE",
                message="A declared replacement must change or remove its node.",
                details={"clientKeys": ineffective_replacements},
            )
        )

    retained_exact_keys = {key for key in set(old_nodes).intersection(new_nodes) if key not in changed_keys}
    old_incoming: dict[
        str,
        set[tuple[str, str, str, str, str]],
    ] = {key: set() for key in old_nodes}
    new_incoming: dict[
        str,
        set[tuple[str, str, str, str, str]],
    ] = {key: set() for key in new_nodes}
    for edge in active_proposal.edges:
        old_incoming.setdefault(edge.target_client_key, set()).add(_execution_edge_signature(edge))
    for edge in revision.graph.edges:
        new_incoming.setdefault(edge.target_client_key, set()).add(_execution_edge_signature(edge))
    for key in sorted(retained_exact_keys):
        if old_incoming.get(key, set()) != new_incoming.get(
            key,
            set(),
        ):
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_INPUT_LINEAGE_MUTATION_FORBIDDEN",
                    message="A retained node must preserve its complete input lineage.",
                    node_key=key,
                )
            )

    added_keys = set(new_nodes) - set(old_nodes)
    trigger_related_old_keys = {
        key
        for key in old_nodes
        if any(
            is_downstream(source_client_key, key)
            for source_client_key in source_client_keys
        )
    }
    trigger_related_goal_ids = {
        goal_id
        for key in trigger_related_old_keys
        for goal_id in old_nodes[key].goal_ids
    }
    for key in sorted(added_keys):
        unrelated_goal_ids = sorted(
            set(new_nodes[key].goal_ids) - trigger_related_goal_ids
        )
        if unrelated_goal_ids:
            issues.append(
                GroundedExecutionGraphIssue(
                    code=(
                        "EXECUTION_GRAPH_ADDED_NODE_TRIGGER_SCOPE_MISMATCH"
                    ),
                    message=(
                        "A revision may add nodes only for goals bound to a "
                        "selected trigger source or its existing downstream."
                    ),
                    node_key=key,
                    details={"unrelatedGoalIds": unrelated_goal_ids},
                )
            )

    old_edge_signatures = {
        _execution_edge_signature(edge)
        for edge in active_proposal.edges
    }
    for edge in revision.graph.edges:
        edge_signature = _execution_edge_signature(edge)
        if (
            edge_signature in old_edge_signatures
            or edge.target_client_key not in added_keys
            or edge.source_client_key not in retained_exact_keys
        ):
            continue
        source_goal_ids = set(
            old_nodes[edge.source_client_key].goal_ids
        )
        target_goal_ids = set(
            new_nodes[edge.target_client_key].goal_ids
        )
        formally_allowed_source_goal_ids = goal_dependency_closure(
            goal_contract,
            target_goal_ids,
        )
        if source_goal_ids.issubset(
            formally_allowed_source_goal_ids
        ):
            continue
        issues.append(
            GroundedExecutionGraphIssue(
                code=(
                    "EXECUTION_GRAPH_ADDED_NODE_LINEAGE_OUTSIDE_TRIGGER_SCOPE"
                ),
                message=(
                    "A recovery node cannot add lineage from an unrelated "
                    "retained node."
                ),
                node_key=edge.target_client_key,
                details={
                    "sourceClientKey": edge.source_client_key,
                    "sourceGoalIds": sorted(source_goal_ids),
                    "targetGoalIds": sorted(target_goal_ids),
                    "allowedSourceGoalIds": sorted(
                        formally_allowed_source_goal_ids
                    ),
                },
            )
        )
    for key in sorted(failed_retirements):
        old_goals = set(old_nodes[key].goal_ids)
        recovered_goals = {goal_id for added_key in added_keys for goal_id in new_nodes[added_key].goal_ids}
        if not old_goals.issubset(recovered_goals):
            issues.append(
                GroundedExecutionGraphIssue(
                    code="EXECUTION_GRAPH_FAILED_NODE_RECOVERY_REQUIRED",
                    message="A failed executed node requires appended recovery coverage.",
                    node_key=key,
                    details={"missingGoalIds": sorted(old_goals - recovered_goals)},
                )
            )
    new_edge_signatures = {_execution_edge_signature(edge) for edge in revision.graph.edges}
    if not (added_keys or removed_keys or changed_keys or old_edge_signatures != new_edge_signatures):
        issues.append(
            GroundedExecutionGraphIssue(
                code="EXECUTION_GRAPH_REPLAN_NO_CHANGE",
                message="A revision must materially change the graph.",
            )
        )

    return GroundedExecutionGraphRevisionValidation(
        valid=not issues,
        issues=issues,
        carried_forward_client_keys=sorted(retained_exact_keys),
        replaced_client_keys=sorted(replacement_keys),
        added_client_keys=sorted(added_keys),
        retired_failed_client_keys=sorted(failed_retirements),
    )


def validate_grounded_execution_graph(
    proposal: GroundedExecutionGraphProposal,
    *,
    goal_contract: OriginalQuestionGoalContract,
    discovery_evidence: list[dict[str, Any]],
    routed_topics: list[str],
    current_version: int,
) -> GroundedExecutionGraphValidation:
    issues: list[GroundedExecutionGraphIssue] = []
    expected_goal_fingerprint = original_question_goal_contract_fingerprint(goal_contract)
    expected_snapshot = discovery_evidence_snapshot_fingerprint(discovery_evidence)
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
        if isinstance(goal, RankingQuestionGoal) and (goal.population_scope != "ALL_MATCHING_ROWS"):
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
            if any(has_graph_path(source_node_key, target_node_key) for source_node_key in source_node_keys):
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
                        "Goals assigned to separate query nodes require an explicit directed execution-graph relation."
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
    parent_receipt: GroundedExecutionGraphReceipt | None = None,
    replan_evidence_fingerprint: str = "",
    replan_evidence_fingerprints: Sequence[str] = (),
    preserved_node_ids: dict[str, str] | None = None,
    retired_node_ids: Sequence[str] = (),
) -> GroundedExecutionGraphReceipt:
    fingerprint = grounded_execution_graph_fingerprint(proposal)
    preserved = dict(preserved_node_ids or {})
    node_ids = {
        node.client_key: (
            preserved[node.client_key]
            if node.client_key in preserved
            else "node_%s" % hashlib.sha256((fingerprint + ":" + node.client_key).encode("utf-8")).hexdigest()[:16]
        )
        for node in proposal.nodes
    }
    waiting = {edge.target_client_key for edge in proposal.edges if edge.dependency_mode == "VERIFIED_ARTIFACT"}
    carried_forward_node_ids = sorted(node_ids[key] for key in preserved if key in node_ids)
    revision_fingerprint = ""
    normalized_replan_fingerprints = sorted(
        {
            str(item or "").strip()
            for item in replan_evidence_fingerprints
            if str(item or "").strip()
        }
    )
    if parent_receipt is not None:
        revision_fingerprint = _grounded_stable_fingerprint(
            {
                "parentGraphId": parent_receipt.graph_id,
                "parentVersion": parent_receipt.version,
                "parentFingerprint": parent_receipt.fingerprint,
                "graphFingerprint": fingerprint,
                "graphVersion": version,
                "replanEvidenceFingerprint": (replan_evidence_fingerprint),
                "replanEvidenceFingerprints": (
                    normalized_replan_fingerprints
                ),
                "carriedForwardNodeIds": carried_forward_node_ids,
                "retiredNodeIds": sorted(
                    str(item or "").strip() for item in retired_node_ids if str(item or "").strip()
                ),
            }
        )
    return GroundedExecutionGraphReceipt(
        graph_id="graph_%s" % fingerprint[:16],
        version=version,
        fingerprint=fingerprint,
        discovery_snapshot_fingerprint=proposal.discovery_snapshot_fingerprint,
        node_ids=node_ids,
        parallel_frontier=[node_ids[node.client_key] for node in proposal.nodes if node.client_key not in waiting],
        waiting_artifact_nodes=[node_ids[key] for key in sorted(waiting)],
        semantic_activation_fingerprint=str(semantic_activation_fingerprint or ""),
        semantic_activation_seal_fingerprint=str(semantic_activation_seal_fingerprint or ""),
        semantic_activation_topics=list(semantic_activation_topics or []),
        parent_graph_id=(parent_receipt.graph_id if parent_receipt is not None else ""),
        parent_version=(parent_receipt.version if parent_receipt is not None else 0),
        parent_fingerprint=(parent_receipt.fingerprint if parent_receipt is not None else ""),
        replan_evidence_fingerprint=str(replan_evidence_fingerprint or ""),
        replan_evidence_fingerprints=normalized_replan_fingerprints,
        carried_forward_node_ids=carried_forward_node_ids,
        retired_node_ids=sorted(str(item or "").strip() for item in retired_node_ids if str(item or "").strip()),
        revision_fingerprint=revision_fingerprint,
    )
