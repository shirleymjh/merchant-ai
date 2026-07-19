from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
    GroundedContextWorkspaceError,
    _atomic_create_at,
    _atomic_write_at,
    _open_directory_beneath,
    _read_regular_file_at,
)
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionGraphProposal,
    GroundedExecutionGraphReceipt,
    discovery_evidence_snapshot_fingerprint,
    grounded_execution_graph_fingerprint,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphReceipt,
    population_dynamic_graph_receipt_fingerprint,
)


GroundedGraphRevisionJournalStatus = Literal[
    "PREPARED",
    "POPULATION_COMMITTED",
    "EXECUTION_COMMITTED",
]

_STATUS_ORDINAL = {
    "PREPARED": 1,
    "POPULATION_COMMITTED": 2,
    "EXECUTION_COMMITTED": 3,
}


class GroundedGraphRevisionJournalError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = str(code or "GRAPH_REVISION_JOURNAL_FAILED")
        super().__init__(self.code)


class _FrozenModel(APIModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=APIModel.model_config.get("alias_generator"),
        use_enum_values=True,
        extra="forbid",
        frozen=True,
    )


def _stable_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _required_text(value: Any, code: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(code)
    return normalized


def _valid_fingerprint(value: Any) -> bool:
    normalized = str(value or "").strip()
    return bool(len(normalized) == 64 and all(character in "0123456789abcdef" for character in normalized))


class GroundedGraphRevisionCandidateNodeDeclaration(_FrozenModel):
    client_key: str
    query_node_id: str
    objective: str = ""
    goal_ids: tuple[str, ...] = ()
    topic_scope: tuple[str, ...] = ()
    evidence_ref_ids: tuple[str, ...] = ()
    dependency_query_node_ids: tuple[str, ...] = ()
    contract_scope_query_node_ids: tuple[str, ...] = ()
    initial_status: Literal[
        "DECLARED",
        "WAITING_VERIFIED_ENTITY_SET",
    ] = "DECLARED"

    @model_validator(mode="after")
    def validate_declaration(
        self,
    ) -> "GroundedGraphRevisionCandidateNodeDeclaration":
        _required_text(self.client_key, "GRAPH_REVISION_NODE_CLIENT_KEY_REQUIRED")
        _required_text(self.query_node_id, "GRAPH_REVISION_NODE_QUERY_ID_REQUIRED")
        for values in (
            self.goal_ids,
            self.topic_scope,
            self.evidence_ref_ids,
            self.dependency_query_node_ids,
            self.contract_scope_query_node_ids,
        ):
            if len(set(values)) != len(values) or any(not str(item or "").strip() for item in values):
                raise ValueError("GRAPH_REVISION_NODE_DECLARATION_INVALID")
        return self


class GroundedGraphRevisionBaseBranchCheckpoint(_FrozenModel):
    client_key: str
    query_node_id: str
    objective: str = ""
    goal_ids: tuple[str, ...] = ()
    topic_scope: tuple[str, ...] = ()
    evidence_ref_ids: tuple[str, ...] = ()
    dependency_query_node_ids: tuple[str, ...] = ()
    contract_scope_query_node_ids: tuple[str, ...] = ()
    opened_topics: tuple[str, ...] = ()
    lifecycle: Literal[
        "UNEXECUTED",
        "PRE_AUTHORIZED",
        "PUBLISHED",
        "EXECUTION_FAILED",
    ]
    status: str
    verified_artifact_ids: tuple[str, ...] = ()
    last_gaps: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_branch(
        self,
    ) -> "GroundedGraphRevisionBaseBranchCheckpoint":
        _required_text(
            self.client_key,
            "GRAPH_REVISION_BASE_BRANCH_CLIENT_KEY_REQUIRED",
        )
        _required_text(
            self.query_node_id,
            "GRAPH_REVISION_BASE_BRANCH_QUERY_ID_REQUIRED",
        )
        _required_text(
            self.status,
            "GRAPH_REVISION_BASE_BRANCH_STATUS_REQUIRED",
        )
        for values in (
            self.goal_ids,
            self.topic_scope,
            self.evidence_ref_ids,
            self.dependency_query_node_ids,
            self.contract_scope_query_node_ids,
            self.opened_topics,
            self.verified_artifact_ids,
        ):
            if len(set(values)) != len(values) or any(
                not str(item or "").strip() for item in values
            ):
                raise ValueError(
                    "GRAPH_REVISION_BASE_BRANCH_DECLARATION_INVALID"
                )
        if self.lifecycle == "PUBLISHED" and (
            not self.verified_artifact_ids
            or self.status != "VERIFIED"
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_PUBLISHED_BRANCH_INVALID"
            )
        return self


class GroundedGraphRevisionBaseSessionCheckpoint(_FrozenModel):
    checkpoint_version: Literal[
        "grounded_graph_revision_base_session.v1"
    ] = "grounded_graph_revision_base_session.v1"
    question: str
    goal_contract: dict[str, Any]
    execution_proposal: dict[str, Any]
    execution_receipt: dict[str, Any]
    population_receipt: dict[str, Any]
    semantic_evidence: tuple[dict[str, Any], ...] = ()
    branches: tuple[
        GroundedGraphRevisionBaseBranchCheckpoint,
        ...,
    ] = ()
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    verified_query_artifacts: tuple[dict[str, Any], ...] = ()
    verified_entity_sets: tuple[dict[str, Any], ...] = ()
    verified_rule_artifacts: tuple[dict[str, Any], ...] = ()
    artifact_goal_ids: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    population_pre_execution_references: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)
    population_post_gate_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )
    population_artifact_query_node_ids: dict[str, str] = Field(
        default_factory=dict
    )
    population_goal_gate_id: str = ""
    population_goal_gate_result: dict[str, Any] = Field(
        default_factory=dict
    )
    population_goal_attestation: dict[str, Any] = Field(
        default_factory=dict
    )
    execution_graph_data_snapshot: dict[str, Any] = Field(
        default_factory=dict
    )
    execution_graph_revision_count: int = Field(ge=0, default=0)
    execution_graph_max_revision_count: int = Field(ge=1, default=1)
    opened_topics: tuple[str, ...] = ()
    checkpoint_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_checkpoint(
        self,
    ) -> "GroundedGraphRevisionBaseSessionCheckpoint":
        contract = OriginalQuestionGoalContract.model_validate(
            self.goal_contract
        )
        proposal = GroundedExecutionGraphProposal.model_validate(
            self.execution_proposal
        )
        receipt = GroundedExecutionGraphReceipt.model_validate(
            self.execution_receipt
        )
        population = PopulationDynamicGraphReceipt.model_validate(
            self.population_receipt
        )
        goal_fingerprint = (
            original_question_goal_contract_fingerprint(contract)
        )
        if (
            str(self.question or "").strip()
            != str(contract.question or "").strip()
            or proposal.goal_contract_fingerprint != goal_fingerprint
            or receipt.fingerprint
            != grounded_execution_graph_fingerprint(proposal)
            or receipt.discovery_snapshot_fingerprint
            != proposal.discovery_snapshot_fingerprint
            or proposal.discovery_snapshot_fingerprint
            != discovery_evidence_snapshot_fingerprint(
                list(self.semantic_evidence)
            )
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_EXECUTION_BINDING_INVALID"
            )
        if (
            population.receipt_fingerprint
            != population_dynamic_graph_receipt_fingerprint(population)
            or population.graph_id != receipt.graph_id
            or population.graph_version != receipt.version
            or population.graph_fingerprint != receipt.fingerprint
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_POPULATION_BINDING_INVALID"
            )

        proposal_nodes = {
            item.client_key: item for item in proposal.nodes
        }
        branches = {item.client_key: item for item in self.branches}
        if (
            len(branches) != len(self.branches)
            or set(branches) != set(proposal_nodes)
            or set(receipt.node_ids) != set(proposal_nodes)
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_BRANCH_SET_INVALID"
            )
        incoming_artifacts: dict[str, set[str]] = {
            key: set() for key in proposal_nodes
        }
        incoming_contracts: dict[str, set[str]] = {
            key: set() for key in proposal_nodes
        }
        for edge in proposal.edges:
            source_id = receipt.node_ids[edge.source_client_key]
            destination = (
                incoming_artifacts
                if edge.dependency_mode == "VERIFIED_ARTIFACT"
                else incoming_contracts
            )
            destination[edge.target_client_key].add(source_id)
        evidence_refs = {
            str(item.get("refId") or "").strip()
            for item in self.semantic_evidence
            if str(item.get("refId") or "").strip()
        }
        if len(evidence_refs) != len(self.semantic_evidence):
            raise ValueError(
                "GRAPH_REVISION_BASE_SEMANTIC_EVIDENCE_INVALID"
            )
        for client_key, node in proposal_nodes.items():
            branch = branches[client_key]
            if (
                branch.query_node_id != receipt.node_ids[client_key]
                or branch.objective != node.objective
                or branch.goal_ids != tuple(node.goal_ids)
                or branch.topic_scope != tuple(node.topic_scope)
                or branch.evidence_ref_ids
                != tuple(node.evidence_ref_ids)
                or not set(branch.evidence_ref_ids).issubset(
                    evidence_refs
                )
                or set(branch.dependency_query_node_ids)
                != incoming_artifacts[client_key]
                or set(branch.contract_scope_query_node_ids)
                != incoming_contracts[client_key]
            ):
                raise ValueError(
                    "GRAPH_REVISION_BASE_BRANCH_BINDING_INVALID"
                )

        artifact_ids = [
            str(item.get("artifactId") or "").strip()
            for item in self.verified_query_artifacts
        ]
        if (
            any(not item for item in artifact_ids)
            or len(set(artifact_ids)) != len(artifact_ids)
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_QUERY_ARTIFACT_SET_INVALID"
            )
        known_goal_ids = set(contract.goal_map())
        for artifact_id, goal_ids in self.artifact_goal_ids.items():
            if (
                artifact_id not in set(artifact_ids)
                or not goal_ids
                or not set(goal_ids).issubset(known_goal_ids)
            ):
                raise ValueError(
                    "GRAPH_REVISION_BASE_ARTIFACT_GOAL_BINDING_INVALID"
                )
        node_ids = set(receipt.node_ids.values())
        if any(
            artifact_id not in set(artifact_ids)
            or query_node_id not in node_ids
            for artifact_id, query_node_id in (
                self.population_artifact_query_node_ids.items()
            )
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_ARTIFACT_NODE_BINDING_INVALID"
            )
        if any(
            query_node_id not in node_ids
            for query_node_id in (
                self.population_pre_execution_references
            )
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_PRE_REFERENCE_SET_INVALID"
            )
        expected = (
            grounded_graph_revision_base_session_checkpoint_fingerprint(
                self
            )
        )
        if (
            self.checkpoint_fingerprint
            and self.checkpoint_fingerprint != expected
        ):
            raise ValueError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_UNSEALED"
            )
        return self


def grounded_graph_revision_base_session_checkpoint_fingerprint(
    checkpoint: GroundedGraphRevisionBaseSessionCheckpoint,
) -> str:
    value = checkpoint.model_dump(by_alias=True, mode="json")
    value["checkpointFingerprint"] = ""
    return _stable_fingerprint(value)


def seal_grounded_graph_revision_base_session_checkpoint(
    checkpoint: GroundedGraphRevisionBaseSessionCheckpoint,
) -> GroundedGraphRevisionBaseSessionCheckpoint:
    sealed = checkpoint.model_copy(
        update={
            "checkpoint_fingerprint": (
                grounded_graph_revision_base_session_checkpoint_fingerprint(
                    checkpoint
                )
            )
        },
        deep=True,
    )
    return GroundedGraphRevisionBaseSessionCheckpoint.model_validate(
        sealed.model_dump(by_alias=True, mode="json")
    )


class GroundedGraphRevisionBaseSessionCheckpointReference(
    _FrozenModel
):
    checkpoint_fingerprint: str
    content_sha256: str
    byte_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_reference(
        self,
    ) -> "GroundedGraphRevisionBaseSessionCheckpointReference":
        if not _valid_fingerprint(
            self.checkpoint_fingerprint
        ) or not _valid_fingerprint(self.content_sha256):
            raise ValueError(
                "GRAPH_REVISION_BASE_SESSION_REFERENCE_INVALID"
            )
        return self


class GroundedGraphRevisionRecoveryPayload(_FrozenModel):
    payload_version: Literal["grounded_graph_revision_recovery.v1"] = "grounded_graph_revision_recovery.v1"
    execution_proposal: dict[str, Any]
    execution_receipt: dict[str, Any]
    population_receipt: dict[str, Any]
    base_session_checkpoint: (
        GroundedGraphRevisionBaseSessionCheckpointReference
    )
    candidate_node_declarations: tuple[
        GroundedGraphRevisionCandidateNodeDeclaration,
        ...,
    ]
    payload_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_payload(self) -> "GroundedGraphRevisionRecoveryPayload":
        proposal = GroundedExecutionGraphProposal.model_validate(self.execution_proposal)
        execution_receipt = GroundedExecutionGraphReceipt.model_validate(self.execution_receipt)
        population_receipt = PopulationDynamicGraphReceipt.model_validate(self.population_receipt)
        execution_fingerprint = grounded_execution_graph_fingerprint(proposal)
        if execution_receipt.fingerprint != execution_fingerprint:
            raise ValueError("GRAPH_REVISION_EXECUTION_RECEIPT_INVALID")
        if (
            population_receipt.receipt_fingerprint != population_dynamic_graph_receipt_fingerprint(population_receipt)
            or population_receipt.graph_id != execution_receipt.graph_id
            or population_receipt.graph_version != execution_receipt.version
            or population_receipt.graph_fingerprint != execution_receipt.fingerprint
        ):
            raise ValueError("GRAPH_REVISION_POPULATION_RECEIPT_INVALID")

        declarations = {item.client_key: item for item in self.candidate_node_declarations}
        proposal_nodes = {item.client_key: item for item in proposal.nodes}
        population_goals_by_query_id = {
            item.query_node_id: tuple(item.consumer_goal_ids)
            for item in population_receipt.nodes
        }
        if (
            len(declarations) != len(self.candidate_node_declarations)
            or set(declarations) != set(proposal_nodes)
            or set(execution_receipt.node_ids) != set(proposal_nodes)
        ):
            raise ValueError("GRAPH_REVISION_NODE_DECLARATION_SET_INVALID")

        incoming_artifacts: dict[str, set[str]] = {key: set() for key in proposal_nodes}
        incoming_contracts: dict[str, set[str]] = {key: set() for key in proposal_nodes}
        for edge in proposal.edges:
            source_id = execution_receipt.node_ids[edge.source_client_key]
            target = edge.target_client_key
            if edge.dependency_mode == "VERIFIED_ARTIFACT":
                incoming_artifacts[target].add(source_id)
            else:
                incoming_contracts[target].add(source_id)
        for client_key, node in proposal_nodes.items():
            declaration = declarations[client_key]
            expected_status = "WAITING_VERIFIED_ENTITY_SET" if incoming_artifacts[client_key] else "DECLARED"
            if (
                declaration.query_node_id != execution_receipt.node_ids[client_key]
                or declaration.objective != node.objective
                or not set(node.goal_ids).issubset(
                    set(declaration.goal_ids)
                )
                or declaration.goal_ids
                != population_goals_by_query_id.get(
                    declaration.query_node_id,
                    (),
                )
                or declaration.topic_scope != tuple(node.topic_scope)
                or declaration.evidence_ref_ids != tuple(node.evidence_ref_ids)
                or set(declaration.dependency_query_node_ids) != incoming_artifacts[client_key]
                or set(declaration.contract_scope_query_node_ids) != incoming_contracts[client_key]
                or declaration.initial_status != expected_status
            ):
                raise ValueError("GRAPH_REVISION_NODE_DECLARATION_MISMATCH")

        expected = grounded_graph_revision_recovery_payload_fingerprint(self)
        if self.payload_fingerprint and self.payload_fingerprint != expected:
            raise ValueError("GRAPH_REVISION_RECOVERY_PAYLOAD_UNSEALED")
        return self


def grounded_graph_revision_recovery_payload_fingerprint(
    payload: GroundedGraphRevisionRecoveryPayload,
) -> str:
    value = payload.model_dump(by_alias=True, mode="json")
    value["payloadFingerprint"] = ""
    return _stable_fingerprint(value)


def seal_grounded_graph_revision_recovery_payload(
    payload: GroundedGraphRevisionRecoveryPayload,
) -> GroundedGraphRevisionRecoveryPayload:
    sealed = payload.model_copy(
        update={"payload_fingerprint": (grounded_graph_revision_recovery_payload_fingerprint(payload))},
        deep=True,
    )
    return GroundedGraphRevisionRecoveryPayload.model_validate(sealed.model_dump(by_alias=True, mode="json"))


def build_grounded_graph_revision_recovery_payload(
    *,
    execution_proposal: GroundedExecutionGraphProposal,
    execution_receipt: GroundedExecutionGraphReceipt,
    population_receipt: PopulationDynamicGraphReceipt,
    base_session_checkpoint: (
        GroundedGraphRevisionBaseSessionCheckpointReference
    ),
    assigned_goal_ids_by_client_key: Mapping[
        str,
        Sequence[str],
    ]
    | None = None,
) -> GroundedGraphRevisionRecoveryPayload:
    proposal = GroundedExecutionGraphProposal.model_validate(execution_proposal)
    receipt = GroundedExecutionGraphReceipt.model_validate(execution_receipt)
    incoming_artifacts: dict[str, list[str]] = {item.client_key: [] for item in proposal.nodes}
    incoming_contracts: dict[str, list[str]] = {item.client_key: [] for item in proposal.nodes}
    for edge in proposal.edges:
        source_id = receipt.node_ids[edge.source_client_key]
        target = edge.target_client_key
        destination = incoming_artifacts if edge.dependency_mode == "VERIFIED_ARTIFACT" else incoming_contracts
        if source_id not in destination[target]:
            destination[target].append(source_id)
    declarations = tuple(
        GroundedGraphRevisionCandidateNodeDeclaration(
            client_key=node.client_key,
            query_node_id=receipt.node_ids[node.client_key],
            objective=node.objective,
            goal_ids=tuple(
                (assigned_goal_ids_by_client_key or {}).get(
                    node.client_key,
                    node.goal_ids,
                )
            ),
            topic_scope=tuple(node.topic_scope),
            evidence_ref_ids=tuple(node.evidence_ref_ids),
            dependency_query_node_ids=tuple(incoming_artifacts[node.client_key]),
            contract_scope_query_node_ids=tuple(incoming_contracts[node.client_key]),
            initial_status=("WAITING_VERIFIED_ENTITY_SET" if incoming_artifacts[node.client_key] else "DECLARED"),
        )
        for node in proposal.nodes
    )
    payload = GroundedGraphRevisionRecoveryPayload(
        execution_proposal=proposal.model_dump(by_alias=True, mode="json"),
        execution_receipt=receipt.model_dump(by_alias=True, mode="json"),
        population_receipt=PopulationDynamicGraphReceipt.model_validate(population_receipt).model_dump(
            by_alias=True, mode="json"
        ),
        base_session_checkpoint=base_session_checkpoint,
        candidate_node_declarations=declarations,
    )
    return seal_grounded_graph_revision_recovery_payload(payload)


class GroundedGraphRevisionJournalRecord(_FrozenModel):
    record_version: Literal["grounded_graph_revision_journal_record.v1"] = "grounded_graph_revision_journal_record.v1"
    transaction_id: str
    revision: int = Field(ge=1)
    status: GroundedGraphRevisionJournalStatus
    owner_fingerprint: str
    run_authority_fingerprint: str
    base_execution_receipt_fingerprint: str
    new_execution_receipt_fingerprint: str
    base_population_receipt_fingerprint: str
    new_population_receipt_fingerprint: str
    evidence_set_fingerprint: str
    recovery_payload: GroundedGraphRevisionRecoveryPayload
    parent_record_fingerprint: str = ""
    record_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_record(self) -> "GroundedGraphRevisionJournalRecord":
        _required_text(self.transaction_id, "GRAPH_REVISION_TRANSACTION_ID_REQUIRED")
        for value in (
            self.owner_fingerprint,
            self.run_authority_fingerprint,
            self.base_execution_receipt_fingerprint,
            self.new_execution_receipt_fingerprint,
            self.base_population_receipt_fingerprint,
            self.new_population_receipt_fingerprint,
            self.evidence_set_fingerprint,
            self.recovery_payload.payload_fingerprint,
        ):
            if not _valid_fingerprint(value):
                raise ValueError("GRAPH_REVISION_FINGERPRINT_INVALID")
        if self.recovery_payload.execution_receipt.get("fingerprint") != (self.new_execution_receipt_fingerprint):
            raise ValueError("GRAPH_REVISION_EXECUTION_BINDING_MISMATCH")
        population_fingerprint = self.recovery_payload.population_receipt.get(
            "receiptFingerprint"
        ) or self.recovery_payload.population_receipt.get("receipt_fingerprint")
        if population_fingerprint != self.new_population_receipt_fingerprint:
            raise ValueError("GRAPH_REVISION_POPULATION_BINDING_MISMATCH")
        expected_transaction_id = grounded_graph_revision_transaction_id(
            owner_fingerprint=self.owner_fingerprint,
            run_authority_fingerprint=self.run_authority_fingerprint,
            base_execution_receipt_fingerprint=(self.base_execution_receipt_fingerprint),
            new_execution_receipt_fingerprint=(self.new_execution_receipt_fingerprint),
            base_population_receipt_fingerprint=(self.base_population_receipt_fingerprint),
            new_population_receipt_fingerprint=(self.new_population_receipt_fingerprint),
            evidence_set_fingerprint=self.evidence_set_fingerprint,
            recovery_payload_fingerprint=(self.recovery_payload.payload_fingerprint),
        )
        if self.transaction_id != expected_transaction_id:
            raise ValueError("GRAPH_REVISION_TRANSACTION_BINDING_MISMATCH")
        if self.revision == 1:
            if self.status != "PREPARED" or self.parent_record_fingerprint:
                raise ValueError("GRAPH_REVISION_INITIAL_RECORD_INVALID")
        elif not _valid_fingerprint(self.parent_record_fingerprint):
            raise ValueError("GRAPH_REVISION_PARENT_RECORD_REQUIRED")
        if self.record_fingerprint != grounded_graph_revision_record_fingerprint(self):
            raise ValueError("GRAPH_REVISION_RECORD_UNSEALED")
        return self


def grounded_graph_revision_record_fingerprint(
    record: GroundedGraphRevisionJournalRecord,
) -> str:
    value = record.model_dump(by_alias=True, mode="json")
    value["recordFingerprint"] = ""
    return _stable_fingerprint(value)


def seal_grounded_graph_revision_record(
    record: GroundedGraphRevisionJournalRecord,
) -> GroundedGraphRevisionJournalRecord:
    return record.model_copy(
        update={"record_fingerprint": (grounded_graph_revision_record_fingerprint(record))},
        deep=True,
    )


def grounded_graph_revision_transaction_id(
    *,
    owner_fingerprint: str,
    run_authority_fingerprint: str,
    base_execution_receipt_fingerprint: str,
    new_execution_receipt_fingerprint: str,
    base_population_receipt_fingerprint: str,
    new_population_receipt_fingerprint: str,
    evidence_set_fingerprint: str,
    recovery_payload_fingerprint: str,
) -> str:
    fingerprint = _stable_fingerprint(
        {
            "ownerFingerprint": owner_fingerprint,
            "runAuthorityFingerprint": run_authority_fingerprint,
            "baseExecutionReceiptFingerprint": (base_execution_receipt_fingerprint),
            "newExecutionReceiptFingerprint": (new_execution_receipt_fingerprint),
            "basePopulationReceiptFingerprint": (base_population_receipt_fingerprint),
            "newPopulationReceiptFingerprint": (new_population_receipt_fingerprint),
            "evidenceSetFingerprint": evidence_set_fingerprint,
            "recoveryPayloadFingerprint": recovery_payload_fingerprint,
        }
    )
    return "graph_revision_txn_%s" % fingerprint[:32]


class GroundedGraphRevisionJournalResult(_FrozenModel):
    record: GroundedGraphRevisionJournalRecord
    committed: bool = False
    idempotent: bool = False


class GroundedGraphRevisionRecovery(_FrozenModel):
    transaction_id: str
    phase: GroundedGraphRevisionJournalStatus
    next_action: Literal[
        "COMMIT_POPULATION",
        "COMMIT_EXECUTION",
        "COMPLETE",
    ]
    roll_forward_required: bool
    record: GroundedGraphRevisionJournalRecord
    recovery_payload: GroundedGraphRevisionRecoveryPayload


class _GroundedGraphRevisionJournalHead(_FrozenModel):
    head_version: Literal["grounded_graph_revision_journal_head.v1"] = "grounded_graph_revision_journal_head.v1"
    transaction_id: str
    owner_fingerprint: str
    run_authority_fingerprint: str
    revision: int = Field(ge=1)
    status: GroundedGraphRevisionJournalStatus
    record_fingerprint: str
    record_sha256: str
    head_fingerprint: str = ""


def _head_fingerprint(head: _GroundedGraphRevisionJournalHead) -> str:
    value = head.model_dump(by_alias=True, mode="json")
    value["headFingerprint"] = ""
    return _stable_fingerprint(value)


class GroundedGraphRevisionTransactionJournal:
    _BASE_COMPONENTS = ("checkpoints", "graph_revision_journal_v1")
    _BASE_SESSION_COMPONENTS = (
        "checkpoints",
        "graph_revision_base_session_v1",
    )
    _BASE_SESSION_PREFIX = "checkpoint_"
    _BASE_SESSION_SUFFIX = ".json"
    _BASE_CLAIM_PREFIX = "base_claim_"
    _BASE_CLAIM_SUFFIX = ".json"
    _HEAD_NAME = "head.json"
    _LOCK_NAME = ".journal.lock"
    _MAX_RECORD_BYTES = 4 * 1024 * 1024
    _MAX_HEAD_BYTES = 64 * 1024
    _MAX_BASE_SESSION_BYTES = 64 * 1024 * 1024

    def __init__(self, workspace: GroundedContextWorkspace) -> None:
        self.workspace = workspace
        self.workspace_root = Path(workspace.root).resolve(strict=True)

    def persist_base_session_checkpoint(
        self,
        checkpoint: GroundedGraphRevisionBaseSessionCheckpoint,
    ) -> GroundedGraphRevisionBaseSessionCheckpointReference:
        parsed = GroundedGraphRevisionBaseSessionCheckpoint.model_validate(
            checkpoint
        )
        if (
            not parsed.checkpoint_fingerprint
            or parsed.checkpoint_fingerprint
            != grounded_graph_revision_base_session_checkpoint_fingerprint(
                parsed
            )
        ):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_UNSEALED"
            )
        encoded = _stable_json_bytes(parsed)
        if len(encoded) > self._MAX_BASE_SESSION_BYTES:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_TOO_LARGE"
            )
        content_sha256 = hashlib.sha256(encoded).hexdigest()
        component = self._base_session_component(content_sha256)
        descriptor = -1
        try:
            descriptor = _open_directory_beneath(
                self.workspace_root,
                self._BASE_SESSION_COMPONENTS,
                create=True,
            )
            try:
                _atomic_create_at(
                    descriptor,
                    component,
                    encoded,
                    mode=0o400,
                    error_code=(
                        "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_WRITE_FAILED"
                    ),
                )
            except GroundedContextWorkspaceError:
                try:
                    existing = _read_regular_file_at(
                        descriptor,
                        component,
                    )
                except (
                    FileNotFoundError,
                    GroundedContextWorkspaceError,
                ) as exc:
                    raise GroundedGraphRevisionJournalError(
                        "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_WRITE_FAILED"
                    ) from exc
                if existing != encoded:
                    raise GroundedGraphRevisionJournalError(
                        "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_CONFLICT"
                    )
        except GroundedGraphRevisionJournalError:
            raise
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_WRITE_FAILED"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return GroundedGraphRevisionBaseSessionCheckpointReference(
            checkpoint_fingerprint=parsed.checkpoint_fingerprint,
            content_sha256=content_sha256,
            byte_count=len(encoded),
        )

    def load_base_session_checkpoint(
        self,
        reference: GroundedGraphRevisionBaseSessionCheckpointReference,
    ) -> GroundedGraphRevisionBaseSessionCheckpoint:
        parsed_reference = (
            GroundedGraphRevisionBaseSessionCheckpointReference.model_validate(
                reference
            )
        )
        if parsed_reference.byte_count > self._MAX_BASE_SESSION_BYTES:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_TOO_LARGE"
            )
        descriptor = -1
        try:
            descriptor = _open_directory_beneath(
                self.workspace_root,
                self._BASE_SESSION_COMPONENTS,
                create=False,
            )
            encoded = _read_regular_file_at(
                descriptor,
                self._base_session_component(
                    parsed_reference.content_sha256
                ),
            )
        except (
            FileNotFoundError,
            OSError,
            GroundedContextWorkspaceError,
        ) as exc:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_MISSING"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            len(encoded) != parsed_reference.byte_count
            or len(encoded) > self._MAX_BASE_SESSION_BYTES
            or hashlib.sha256(encoded).hexdigest()
            != parsed_reference.content_sha256
        ):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_INTEGRITY_FAILED"
            )
        try:
            checkpoint = (
                GroundedGraphRevisionBaseSessionCheckpoint.model_validate_json(
                    encoded
                )
            )
        except Exception as exc:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_INVALID"
            ) from exc
        if (
            checkpoint.checkpoint_fingerprint
            != parsed_reference.checkpoint_fingerprint
        ):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_CHECKPOINT_BINDING_INVALID"
            )
        return checkpoint

    def prepare(
        self,
        *,
        base_execution_receipt_fingerprint: str,
        new_execution_receipt_fingerprint: str,
        base_population_receipt_fingerprint: str,
        new_population_receipt_fingerprint: str,
        evidence_set_fingerprint: str,
        recovery_payload: GroundedGraphRevisionRecoveryPayload,
    ) -> GroundedGraphRevisionJournalResult:
        payload = GroundedGraphRevisionRecoveryPayload.model_validate(recovery_payload)
        if (
            not payload.payload_fingerprint
            or payload.payload_fingerprint != grounded_graph_revision_recovery_payload_fingerprint(payload)
        ):
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_RECOVERY_PAYLOAD_UNSEALED")
        transaction_id = grounded_graph_revision_transaction_id(
            owner_fingerprint=self.workspace.owner_fingerprint,
            run_authority_fingerprint=self.workspace.request_fingerprint,
            base_execution_receipt_fingerprint=(base_execution_receipt_fingerprint),
            new_execution_receipt_fingerprint=(new_execution_receipt_fingerprint),
            base_population_receipt_fingerprint=(base_population_receipt_fingerprint),
            new_population_receipt_fingerprint=(new_population_receipt_fingerprint),
            evidence_set_fingerprint=evidence_set_fingerprint,
            recovery_payload_fingerprint=payload.payload_fingerprint,
        )
        candidate = seal_grounded_graph_revision_record(
            GroundedGraphRevisionJournalRecord.model_construct(
                transaction_id=transaction_id,
                revision=1,
                status="PREPARED",
                owner_fingerprint=self.workspace.owner_fingerprint,
                run_authority_fingerprint=self.workspace.request_fingerprint,
                base_execution_receipt_fingerprint=(base_execution_receipt_fingerprint),
                new_execution_receipt_fingerprint=(new_execution_receipt_fingerprint),
                base_population_receipt_fingerprint=(base_population_receipt_fingerprint),
                new_population_receipt_fingerprint=(new_population_receipt_fingerprint),
                evidence_set_fingerprint=evidence_set_fingerprint,
                recovery_payload=payload,
            )
        )
        candidate = GroundedGraphRevisionJournalRecord.model_validate(candidate.model_dump(by_alias=True, mode="json"))
        self._claim_base(candidate)
        descriptor = self._open_transaction_directory(
            transaction_id,
            create=True,
        )
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock(descriptor)
            current = self._load_locked(descriptor, transaction_id)
            if current is not None:
                if not self._same_transaction(current, candidate):
                    raise GroundedGraphRevisionJournalError("GRAPH_REVISION_TRANSACTION_CONFLICT")
                return GroundedGraphRevisionJournalResult(
                    record=current,
                    idempotent=True,
                )
            self._publish_record_locked(descriptor, candidate)
            return GroundedGraphRevisionJournalResult(
                record=candidate,
                committed=True,
            )
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def advance(
        self,
        transaction_id: str,
        *,
        target_status: Literal[
            "POPULATION_COMMITTED",
            "EXECUTION_COMMITTED",
        ],
        expected_revision: int,
        expected_record_fingerprint: str,
    ) -> GroundedGraphRevisionJournalResult:
        descriptor = self._open_transaction_directory(
            transaction_id,
            create=False,
        )
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock(descriptor)
            current = self._load_locked(descriptor, transaction_id)
            if current is None:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_TRANSACTION_NOT_FOUND")
            current_ordinal = _STATUS_ORDINAL[current.status]
            target_ordinal = _STATUS_ORDINAL[target_status]
            if target_ordinal <= current_ordinal:
                committed_target = current
                while (
                    _STATUS_ORDINAL[committed_target.status]
                    > target_ordinal
                ):
                    committed_target = self._load_historical_record_locked(
                        descriptor,
                        transaction_id=transaction_id,
                        revision=committed_target.revision - 1,
                        record_fingerprint=(
                            committed_target.parent_record_fingerprint
                        ),
                    )
                if (
                    committed_target.status != target_status
                    or committed_target.revision
                    != int(expected_revision) + 1
                    or committed_target.parent_record_fingerprint
                    != str(expected_record_fingerprint or "").strip()
                ):
                    raise GroundedGraphRevisionJournalError(
                        "GRAPH_REVISION_JOURNAL_CAS_CONFLICT"
                    )
                return GroundedGraphRevisionJournalResult(
                    record=current,
                    idempotent=True,
                )
            if target_ordinal != current_ordinal + 1:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_STATUS_JUMP_FORBIDDEN")
            if (
                current.revision != int(expected_revision)
                or current.record_fingerprint != str(expected_record_fingerprint or "").strip()
            ):
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_CAS_CONFLICT")
            next_record = seal_grounded_graph_revision_record(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "status": target_status,
                        "parent_record_fingerprint": (current.record_fingerprint),
                        "record_fingerprint": "",
                    },
                    deep=True,
                )
            )
            next_record = GroundedGraphRevisionJournalRecord.model_validate(
                next_record.model_dump(by_alias=True, mode="json")
            )
            self._publish_record_locked(descriptor, next_record)
            return GroundedGraphRevisionJournalResult(
                record=next_record,
                committed=True,
            )
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def load(
        self,
        transaction_id: str,
    ) -> GroundedGraphRevisionJournalRecord | None:
        try:
            descriptor = self._open_transaction_directory(
                transaction_id,
                create=False,
            )
        except FileNotFoundError:
            return None
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock(descriptor)
            return self._load_locked(descriptor, transaction_id)
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def load_recovery(
        self,
        transaction_id: str,
    ) -> GroundedGraphRevisionRecovery | None:
        record = self.load(transaction_id)
        if record is None:
            return None
        next_action = {
            "PREPARED": "COMMIT_POPULATION",
            "POPULATION_COMMITTED": "COMMIT_EXECUTION",
            "EXECUTION_COMMITTED": "COMPLETE",
        }[record.status]
        return GroundedGraphRevisionRecovery(
            transaction_id=record.transaction_id,
            phase=record.status,
            next_action=next_action,
            roll_forward_required=record.status != "EXECUTION_COMMITTED",
            record=record,
            recovery_payload=record.recovery_payload,
        )

    def discover_pending(
        self,
    ) -> tuple[GroundedGraphRevisionRecovery, ...]:
        """Discover run-bound transactions that still require roll-forward."""

        try:
            base_descriptor = _open_directory_beneath(
                self.workspace_root,
                self._BASE_COMPONENTS,
                create=False,
            )
        except FileNotFoundError:
            return ()
        try:
            components = sorted(os.listdir(base_descriptor))
            recoveries: list[GroundedGraphRevisionRecovery] = []
            for component in components:
                if not component.startswith(self._BASE_CLAIM_PREFIX):
                    continue
                if not self._valid_base_claim_component(component):
                    raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_DIRECTORY_INVALID")
                try:
                    encoded_claim = _read_regular_file_at(
                        base_descriptor,
                        component,
                    )
                    claim = self._validate_base_claim(
                        component,
                        encoded_claim,
                    )
                    record = self._materialize_claim(claim)
                    if record.status == "EXECUTION_COMMITTED":
                        continue
                    next_action = {
                        "PREPARED": "COMMIT_POPULATION",
                        "POPULATION_COMMITTED": "COMMIT_EXECUTION",
                    }[record.status]
                    recoveries.append(
                        GroundedGraphRevisionRecovery(
                            transaction_id=record.transaction_id,
                            phase=record.status,
                            next_action=next_action,
                            roll_forward_required=True,
                            record=record,
                            recovery_payload=record.recovery_payload,
                        )
                    )
                except GroundedGraphRevisionJournalError:
                    raise
                except (OSError, GroundedContextWorkspaceError) as exc:
                    raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_DIRECTORY_INVALID") from exc
            return tuple(recoveries)
        finally:
            os.close(base_descriptor)

    def transaction_path(self, transaction_id: str) -> Path:
        return self.workspace_root.joinpath(
            *self._BASE_COMPONENTS,
            self._transaction_component(transaction_id),
        )

    @staticmethod
    def _same_transaction(
        left: GroundedGraphRevisionJournalRecord,
        right: GroundedGraphRevisionJournalRecord,
    ) -> bool:
        fields = (
            "transaction_id",
            "owner_fingerprint",
            "run_authority_fingerprint",
            "base_execution_receipt_fingerprint",
            "new_execution_receipt_fingerprint",
            "base_population_receipt_fingerprint",
            "new_population_receipt_fingerprint",
            "evidence_set_fingerprint",
        )
        return bool(
            all(getattr(left, field) == getattr(right, field) for field in fields)
            and left.recovery_payload.payload_fingerprint == right.recovery_payload.payload_fingerprint
        )

    @staticmethod
    def _transaction_component(transaction_id: str) -> str:
        normalized = _required_text(
            transaction_id,
            "GRAPH_REVISION_TRANSACTION_ID_REQUIRED",
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _base_session_component(cls, content_sha256: str) -> str:
        normalized = str(content_sha256 or "").strip()
        if not _valid_fingerprint(normalized):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_BASE_SESSION_REFERENCE_INVALID"
            )
        return "%s%s%s" % (
            cls._BASE_SESSION_PREFIX,
            normalized,
            cls._BASE_SESSION_SUFFIX,
        )

    @staticmethod
    def _valid_transaction_component(component: str) -> bool:
        return bool(len(component) == 64 and all(character in "0123456789abcdef" for character in component))

    @classmethod
    def _base_claim_name(
        cls,
        record: GroundedGraphRevisionJournalRecord,
    ) -> str:
        identity = {
            "ownerFingerprint": record.owner_fingerprint,
            "runAuthorityFingerprint": record.run_authority_fingerprint,
            "baseExecutionReceiptFingerprint": (record.base_execution_receipt_fingerprint),
            "basePopulationReceiptFingerprint": (record.base_population_receipt_fingerprint),
        }
        return "%s%s%s" % (
            cls._BASE_CLAIM_PREFIX,
            _stable_fingerprint(identity),
            cls._BASE_CLAIM_SUFFIX,
        )

    @classmethod
    def _valid_base_claim_component(cls, component: str) -> bool:
        if not (component.startswith(cls._BASE_CLAIM_PREFIX) and component.endswith(cls._BASE_CLAIM_SUFFIX)):
            return False
        fingerprint = component[len(cls._BASE_CLAIM_PREFIX) : -len(cls._BASE_CLAIM_SUFFIX)]
        return bool(len(fingerprint) == 64 and all(character in "0123456789abcdef" for character in fingerprint))

    def _claim_base(
        self,
        candidate: GroundedGraphRevisionJournalRecord,
    ) -> None:
        claim_name = self._base_claim_name(candidate)
        descriptor = -1
        try:
            descriptor = _open_directory_beneath(
                self.workspace_root,
                self._BASE_COMPONENTS,
                create=True,
            )
            try:
                _atomic_create_at(
                    descriptor,
                    claim_name,
                    _stable_json_bytes(candidate),
                    mode=0o400,
                    error_code="GRAPH_REVISION_BASE_CLAIM_FAILED",
                )
                return
            except GroundedContextWorkspaceError:
                try:
                    encoded_claim = _read_regular_file_at(
                        descriptor,
                        claim_name,
                    )
                except (
                    FileNotFoundError,
                    GroundedContextWorkspaceError,
                ) as exc:
                    raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_FAILED") from exc
            existing = self._validate_base_claim(
                claim_name,
                encoded_claim,
            )
            if existing.transaction_id != candidate.transaction_id:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_ALREADY_CLAIMED")
            if existing != candidate:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_CONFLICT")
        except GroundedGraphRevisionJournalError:
            raise
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_FAILED") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_base_claim(
        self,
        component: str,
        encoded_claim: bytes,
    ) -> GroundedGraphRevisionJournalRecord:
        if len(encoded_claim) > self._MAX_RECORD_BYTES:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_TOO_LARGE")
        try:
            record = GroundedGraphRevisionJournalRecord.model_validate_json(encoded_claim)
        except Exception as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_INVALID") from exc
        if (
            record.revision != 1
            or record.status != "PREPARED"
            or record.owner_fingerprint != self.workspace.owner_fingerprint
            or record.run_authority_fingerprint != self.workspace.request_fingerprint
            or component != self._base_claim_name(record)
        ):
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_BASE_CLAIM_BINDING_INVALID")
        return record

    def _materialize_claim(
        self,
        claim: GroundedGraphRevisionJournalRecord,
    ) -> GroundedGraphRevisionJournalRecord:
        descriptor = self._open_transaction_directory(
            claim.transaction_id,
            create=True,
        )
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock(descriptor)
            current = self._load_locked(
                descriptor,
                claim.transaction_id,
            )
            if current is None:
                self._publish_record_locked(descriptor, claim)
                return claim
            if not self._same_transaction(current, claim):
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_TRANSACTION_CONFLICT")
            return current
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def _open_transaction_directory(
        self,
        transaction_id: str,
        *,
        create: bool,
    ) -> int:
        try:
            return _open_directory_beneath(
                self.workspace_root,
                (
                    *self._BASE_COMPONENTS,
                    self._transaction_component(transaction_id),
                ),
                create=create,
            )
        except FileNotFoundError:
            raise
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_DIRECTORY_INVALID") from exc

    def _lock(self, descriptor: int) -> int:
        try:
            lock_descriptor = os.open(
                self._LOCK_NAME,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_LOCK_INVALID") from exc
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            os.close(lock_descriptor)
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_LOCK_INVALID")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        return lock_descriptor

    def _load_locked(
        self,
        descriptor: int,
        transaction_id: str,
    ) -> GroundedGraphRevisionJournalRecord | None:
        try:
            encoded_head = _read_regular_file_at(
                descriptor,
                self._HEAD_NAME,
            )
        except FileNotFoundError:
            return None
        if len(encoded_head) > self._MAX_HEAD_BYTES:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_HEAD_TOO_LARGE")
        try:
            head = _GroundedGraphRevisionJournalHead.model_validate_json(encoded_head)
        except Exception as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_HEAD_INVALID") from exc
        if (
            head.transaction_id != transaction_id
            or head.owner_fingerprint != self.workspace.owner_fingerprint
            or head.run_authority_fingerprint != self.workspace.request_fingerprint
            or head.head_fingerprint != _head_fingerprint(head)
            or not _valid_fingerprint(head.record_fingerprint)
            or not _valid_fingerprint(head.record_sha256)
        ):
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_HEAD_BINDING_INVALID")
        record_name = self._record_name(
            head.revision,
            head.record_fingerprint,
        )
        try:
            encoded_record = _read_regular_file_at(
                descriptor,
                record_name,
            )
        except FileNotFoundError as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_MISSING") from exc
        if (
            len(encoded_record) > self._MAX_RECORD_BYTES
            or hashlib.sha256(encoded_record).hexdigest() != head.record_sha256
        ):
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_INTEGRITY_FAILED")
        try:
            record = GroundedGraphRevisionJournalRecord.model_validate_json(encoded_record)
        except Exception as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_INVALID") from exc
        if (
            record.transaction_id != transaction_id
            or record.revision != head.revision
            or record.status != head.status
            or record.record_fingerprint != head.record_fingerprint
            or record.owner_fingerprint != self.workspace.owner_fingerprint
            or record.run_authority_fingerprint != self.workspace.request_fingerprint
        ):
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_BINDING_INVALID")
        return record

    def _load_historical_record_locked(
        self,
        descriptor: int,
        *,
        transaction_id: str,
        revision: int,
        record_fingerprint: str,
    ) -> GroundedGraphRevisionJournalRecord:
        normalized_fingerprint = str(
            record_fingerprint or ""
        ).strip()
        if revision < 1 or not _valid_fingerprint(
            normalized_fingerprint
        ):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_JOURNAL_HISTORY_INVALID"
            )
        try:
            encoded_record = _read_regular_file_at(
                descriptor,
                self._record_name(
                    revision,
                    normalized_fingerprint,
                ),
            )
        except FileNotFoundError as exc:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_JOURNAL_RECORD_MISSING"
            ) from exc
        if len(encoded_record) > self._MAX_RECORD_BYTES:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_JOURNAL_RECORD_TOO_LARGE"
            )
        try:
            record = (
                GroundedGraphRevisionJournalRecord.model_validate_json(
                    encoded_record
                )
            )
        except Exception as exc:
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_JOURNAL_RECORD_INVALID"
            ) from exc
        if (
            record.transaction_id != transaction_id
            or record.revision != revision
            or record.record_fingerprint != normalized_fingerprint
            or record.owner_fingerprint
            != self.workspace.owner_fingerprint
            or record.run_authority_fingerprint
            != self.workspace.request_fingerprint
        ):
            raise GroundedGraphRevisionJournalError(
                "GRAPH_REVISION_JOURNAL_RECORD_BINDING_INVALID"
            )
        return record

    def _publish_record_locked(
        self,
        descriptor: int,
        record: GroundedGraphRevisionJournalRecord,
    ) -> None:
        encoded = _stable_json_bytes(record)
        if len(encoded) > self._MAX_RECORD_BYTES:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_TOO_LARGE")
        name = self._record_name(
            record.revision,
            record.record_fingerprint,
        )
        try:
            existing = _read_regular_file_at(descriptor, name)
        except FileNotFoundError:
            existing = b""
        if existing:
            if existing != encoded:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_IMMUTABLE_RECORD_CONFLICT")
        else:
            try:
                _atomic_create_at(
                    descriptor,
                    name,
                    encoded,
                    mode=0o400,
                    error_code=("GRAPH_REVISION_JOURNAL_RECORD_CREATE_FAILED"),
                )
            except GroundedContextWorkspaceError as exc:
                raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_RECORD_CREATE_FAILED") from exc
        pending_head = _GroundedGraphRevisionJournalHead(
            transaction_id=record.transaction_id,
            owner_fingerprint=record.owner_fingerprint,
            run_authority_fingerprint=record.run_authority_fingerprint,
            revision=record.revision,
            status=record.status,
            record_fingerprint=record.record_fingerprint,
            record_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        head = pending_head.model_copy(update={"head_fingerprint": _head_fingerprint(pending_head)})
        try:
            _atomic_write_at(
                descriptor,
                self._HEAD_NAME,
                _stable_json_bytes(head),
                error_code="GRAPH_REVISION_JOURNAL_HEAD_COMMIT_FAILED",
            )
        except GroundedContextWorkspaceError as exc:
            raise GroundedGraphRevisionJournalError("GRAPH_REVISION_JOURNAL_HEAD_COMMIT_FAILED") from exc

    @staticmethod
    def _record_name(revision: int, fingerprint: str) -> str:
        return "record_%d_%s.json" % (int(revision), fingerprint)
