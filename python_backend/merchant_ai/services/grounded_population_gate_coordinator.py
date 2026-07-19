from __future__ import annotations

import hashlib
import json
from enum import Enum
from threading import RLock
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_population_verifier import (
    GoalDeclarationPopulationVerificationInput,
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationDeclaration,
    PopulationExecutionClaim,
    PopulationResultEvidence,
    PopulationSemanticReview,
    PopulationSemanticVerifier,
    PopulationVerificationAttestation,
    PopulationVerificationResult,
    PopulationVerificationStage,
    PostResultPopulationVerificationInput,
    PreExecutionPopulationVerificationInput,
    population_attestation_fingerprint,
)


class PopulationGatePhase(str, Enum):
    GOAL_DECLARATION = "GOAL_DECLARATION"
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_RESULT = "POST_RESULT"


class PopulationGateCode(str, Enum):
    COMMITTED = "COMMITTED"
    STATE_ALREADY_EXISTS = "STATE_ALREADY_EXISTS"
    STATE_NOT_FOUND = "STATE_NOT_FOUND"
    STATE_INVALID = "STATE_INVALID"
    STATE_STORE_FAILED = "STATE_STORE_FAILED"
    CAS_REVISION_MISMATCH = "CAS_REVISION_MISMATCH"
    BINDING_MISMATCH = "BINDING_MISMATCH"
    PHASE_MISMATCH = "PHASE_MISMATCH"
    ATTESTATION_CHAIN_INVALID = "ATTESTATION_CHAIN_INVALID"
    VERIFICATION_REJECTED = "VERIFICATION_REJECTED"
    GRAPH_BINDING_INVALID = "GRAPH_BINDING_INVALID"
    LEDGER_AUTHORITY_UNTRUSTED = "LEDGER_AUTHORITY_UNTRUSTED"
    LEDGER_READ_FAILED = "LEDGER_READ_FAILED"
    LEDGER_SNAPSHOT_INVALID = "LEDGER_SNAPSHOT_INVALID"
    LEDGER_BINDING_MISMATCH = "LEDGER_BINDING_MISMATCH"
    RESULT_SELECTION_DUPLICATE = "RESULT_SELECTION_DUPLICATE"
    RESULT_NOT_IN_LEDGER = "RESULT_NOT_IN_LEDGER"
    RESULT_RECEIPT_MISMATCH = "RESULT_RECEIPT_MISMATCH"
    RESULT_NOT_PUBLISHED = "RESULT_NOT_PUBLISHED"
    RESULT_NOT_VERIFIED = "RESULT_NOT_VERIFIED"
    RESULT_MUTABLE = "RESULT_MUTABLE"
    RESULT_COVERAGE_INCOMPLETE = "RESULT_COVERAGE_INCOMPLETE"
    RESULT_TRUNCATED = "RESULT_TRUNCATED"
    RESULT_COUNT_MISMATCH = "RESULT_COUNT_MISMATCH"
    RESULT_BINDING_MISMATCH = "RESULT_BINDING_MISMATCH"


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _require_text(value: Any, field_name: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


def _require_unique(values: Sequence[str], field_name: str) -> None:
    normalized = [_text(value) for value in values]
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicate values" % field_name)


def _stable_fingerprint(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PopulationExecutionNodeBinding(_StrictFrozenModel):
    query_node_id: str
    consumer_goal_ids: tuple[str, ...]
    generation: int = Field(ge=1)
    attempt_id: str
    query_contract_fingerprint: str
    sql_ast_fingerprint: str
    snapshot_fingerprint: str

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationExecutionNodeBinding":
        _require_text(self.query_node_id, "query_node_id")
        if not self.consumer_goal_ids:
            raise ValueError("consumer_goal_ids must not be empty")
        _require_unique(self.consumer_goal_ids, "consumer_goal_ids")
        _require_text(self.attempt_id, "attempt_id")
        _require_text(
            self.query_contract_fingerprint,
            "query_contract_fingerprint",
        )
        _require_text(self.sql_ast_fingerprint, "sql_ast_fingerprint")
        _require_text(self.snapshot_fingerprint, "snapshot_fingerprint")
        return self


class PopulationExecutionGraphBinding(_StrictFrozenModel):
    binding_version: Literal["population_execution_graph_binding.v1"] = "population_execution_graph_binding.v1"
    graph_id: str
    graph_version: int = Field(ge=1)
    graph_fingerprint: str
    nodes: tuple[PopulationExecutionNodeBinding, ...]
    binding_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationExecutionGraphBinding":
        _require_text(self.graph_id, "graph_id")
        _require_text(self.graph_fingerprint, "graph_fingerprint")
        if not self.nodes:
            raise ValueError("nodes must not be empty")
        _require_unique(
            tuple(item.query_node_id for item in self.nodes),
            "nodes.query_node_id",
        )
        return self


def population_execution_graph_binding_fingerprint(
    binding: PopulationExecutionGraphBinding,
) -> str:
    payload = binding.model_dump(by_alias=True, mode="json")
    payload["bindingFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_execution_graph_binding(
    binding: PopulationExecutionGraphBinding,
) -> PopulationExecutionGraphBinding:
    return binding.model_copy(update={"binding_fingerprint": population_execution_graph_binding_fingerprint(binding)})


class PopulationDynamicGraphNode(_StrictFrozenModel):
    query_node_id: str
    consumer_goal_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationDynamicGraphNode":
        _require_text(self.query_node_id, "query_node_id")
        if not self.consumer_goal_ids:
            raise ValueError("consumer_goal_ids must not be empty")
        _require_unique(self.consumer_goal_ids, "consumer_goal_ids")
        return self


class PopulationDynamicGraphEdge(_StrictFrozenModel):
    source_query_node_id: str
    target_query_node_id: str
    dependency_mode: Literal["CONTRACT_SCOPE", "VERIFIED_ARTIFACT"]
    artifact_kind: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationDynamicGraphEdge":
        _require_text(
            self.source_query_node_id,
            "source_query_node_id",
        )
        _require_text(
            self.target_query_node_id,
            "target_query_node_id",
        )
        if self.source_query_node_id == self.target_query_node_id:
            raise ValueError("a dynamic graph edge cannot self-reference")
        return self


class PopulationDynamicGraphReceipt(_StrictFrozenModel):
    receipt_version: Literal["population_dynamic_graph_receipt.v1"] = "population_dynamic_graph_receipt.v1"
    graph_id: str
    graph_version: int = Field(ge=1)
    graph_fingerprint: str
    nodes: tuple[PopulationDynamicGraphNode, ...]
    edges: tuple[PopulationDynamicGraphEdge, ...] = ()
    parent_receipt_fingerprint: str = ""
    revision_evidence_fingerprint: str = ""
    carried_forward_query_node_ids: tuple[str, ...] = ()
    retired_query_node_ids: tuple[str, ...] = ()
    receipt_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationDynamicGraphReceipt":
        _require_text(self.graph_id, "graph_id")
        _require_text(self.graph_fingerprint, "graph_fingerprint")
        if not self.nodes:
            raise ValueError("nodes must not be empty")
        _require_unique(
            tuple(item.query_node_id for item in self.nodes),
            "nodes.query_node_id",
        )
        node_ids = {item.query_node_id for item in self.nodes}
        _require_unique(
            self.carried_forward_query_node_ids,
            "carried_forward_query_node_ids",
        )
        _require_unique(
            self.retired_query_node_ids,
            "retired_query_node_ids",
        )
        if set(self.carried_forward_query_node_ids) - node_ids:
            raise ValueError("carried-forward query nodes must remain active")
        if set(self.retired_query_node_ids).intersection(node_ids):
            raise ValueError("retired query nodes cannot remain active")
        if self.parent_receipt_fingerprint and not (self.revision_evidence_fingerprint):
            raise ValueError("a revised graph requires structured evidence")
        for edge in self.edges:
            if edge.source_query_node_id not in node_ids or edge.target_query_node_id not in node_ids:
                raise ValueError("dynamic graph edge endpoint is unknown")
        return self


def population_dynamic_graph_receipt_fingerprint(
    receipt: PopulationDynamicGraphReceipt,
) -> str:
    payload = receipt.model_dump(by_alias=True, mode="json")
    payload["receiptFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_dynamic_graph_receipt(
    receipt: PopulationDynamicGraphReceipt,
) -> PopulationDynamicGraphReceipt:
    return receipt.model_copy(update={"receipt_fingerprint": (population_dynamic_graph_receipt_fingerprint(receipt))})


class PopulationNodeGateRecord(_StrictFrozenModel):
    query_node_id: str
    graph_receipt_fingerprint: str
    node_binding: PopulationExecutionNodeBinding
    required_consumer_goal_ids: tuple[str, ...] = ()
    pre_execution_attestation: PopulationVerificationAttestation
    post_result_attestation: PopulationVerificationAttestation | None = None
    ledger_snapshot_fingerprint: str = ""
    published_receipt_fingerprints: tuple[str, ...] = ()
    record_fingerprint: str = ""


def population_node_gate_record_fingerprint(
    record: PopulationNodeGateRecord,
) -> str:
    payload = record.model_dump(by_alias=True, mode="json")
    payload["recordFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_node_gate_record(
    record: PopulationNodeGateRecord,
) -> PopulationNodeGateRecord:
    return record.model_copy(update={"record_fingerprint": (population_node_gate_record_fingerprint(record))})


class PopulationPublishedArtifactReceipt(_StrictFrozenModel):
    receipt_version: Literal["population_published_artifact_receipt.v1"] = "population_published_artifact_receipt.v1"
    ledger_artifact_id: str
    source_query_artifact_id: str = ""
    publication_status: str
    generation: int = Field(ge=1)
    attempt_id: str
    goal_contract_fingerprint: str
    graph_fingerprint: str
    query_node_id: str
    covered_consumer_goal_ids: tuple[str, ...]
    result_is_truncated: bool
    stored_row_count: int = Field(ge=0)
    exact_result_row_count: int = Field(ge=0)
    evidence: PopulationArtifactEvidence
    receipt_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationPublishedArtifactReceipt":
        _require_text(self.ledger_artifact_id, "ledger_artifact_id")
        _require_text(self.publication_status, "publication_status")
        _require_text(self.attempt_id, "attempt_id")
        _require_text(
            self.goal_contract_fingerprint,
            "goal_contract_fingerprint",
        )
        _require_text(self.graph_fingerprint, "graph_fingerprint")
        _require_text(self.query_node_id, "query_node_id")
        if not self.covered_consumer_goal_ids:
            raise ValueError("covered_consumer_goal_ids must not be empty")
        _require_unique(
            self.covered_consumer_goal_ids,
            "covered_consumer_goal_ids",
        )
        return self


def population_published_artifact_receipt_fingerprint(
    receipt: PopulationPublishedArtifactReceipt,
) -> str:
    payload = receipt.model_dump(by_alias=True, mode="json")
    payload["receiptFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_published_artifact_receipt(
    receipt: PopulationPublishedArtifactReceipt,
) -> PopulationPublishedArtifactReceipt:
    return receipt.model_copy(
        update={"receipt_fingerprint": (population_published_artifact_receipt_fingerprint(receipt))}
    )


class PopulationArtifactLedgerEntry(_StrictFrozenModel):
    ledger_artifact_id: str
    publication_status: str
    receipt: PopulationPublishedArtifactReceipt
    entry_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationArtifactLedgerEntry":
        _require_text(self.ledger_artifact_id, "ledger_artifact_id")
        _require_text(self.publication_status, "publication_status")
        return self


def population_artifact_ledger_entry_fingerprint(
    entry: PopulationArtifactLedgerEntry,
) -> str:
    payload = entry.model_dump(by_alias=True, mode="json")
    payload["entryFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_artifact_ledger_entry(
    entry: PopulationArtifactLedgerEntry,
) -> PopulationArtifactLedgerEntry:
    return entry.model_copy(update={"entry_fingerprint": population_artifact_ledger_entry_fingerprint(entry)})


class PopulationArtifactLedgerSnapshot(_StrictFrozenModel):
    snapshot_version: Literal["population_artifact_ledger_snapshot.v1"] = "population_artifact_ledger_snapshot.v1"
    ledger_id: str
    ledger_authority_fingerprint: str
    ledger_revision: int = Field(ge=0)
    goal_contract_fingerprint: str
    graph_fingerprint: str
    entries: tuple[PopulationArtifactLedgerEntry, ...]
    snapshot_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationArtifactLedgerSnapshot":
        _require_text(self.ledger_id, "ledger_id")
        _require_text(
            self.ledger_authority_fingerprint,
            "ledger_authority_fingerprint",
        )
        _require_text(
            self.goal_contract_fingerprint,
            "goal_contract_fingerprint",
        )
        _require_text(self.graph_fingerprint, "graph_fingerprint")
        return self


def population_artifact_ledger_snapshot_fingerprint(
    snapshot: PopulationArtifactLedgerSnapshot,
) -> str:
    payload = snapshot.model_dump(by_alias=True, mode="json")
    payload["snapshotFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_artifact_ledger_snapshot(
    snapshot: PopulationArtifactLedgerSnapshot,
) -> PopulationArtifactLedgerSnapshot:
    return snapshot.model_copy(
        update={"snapshot_fingerprint": (population_artifact_ledger_snapshot_fingerprint(snapshot))}
    )


class PopulationArtifactLedgerReader(Protocol):
    @property
    def authority_fingerprint(self) -> str: ...

    def snapshot_population_artifacts(
        self,
        *,
        gate_id: str,
        goal_contract_fingerprint: str,
        graph_fingerprint: str,
    ) -> PopulationArtifactLedgerSnapshot: ...


class _PopulationGateCommand(_StrictFrozenModel):
    gate_id: str
    expected_revision: int = Field(ge=0)
    goal_contract_fingerprint: str

    @model_validator(mode="after")
    def validate_binding(self) -> "_PopulationGateCommand":
        _require_text(self.gate_id, "gate_id")
        _require_text(
            self.goal_contract_fingerprint,
            "goal_contract_fingerprint",
        )
        return self


class PopulationGoalDeclarationCommand(_PopulationGateCommand):
    question_fingerprint: str
    goal_skeleton_fingerprint: str = ""
    declaration_author_fingerprint: str
    semantic_review: PopulationSemanticReview
    declarations: tuple[PopulationDeclaration, ...]


class PopulationPreExecutionCommand(_PopulationGateCommand):
    graph_binding: PopulationExecutionGraphBinding
    claims: tuple[PopulationExecutionClaim, ...]


class PopulationNodePreExecutionCommand(_PopulationGateCommand):
    graph_receipt: PopulationDynamicGraphReceipt
    node_binding: PopulationExecutionNodeBinding
    required_consumer_goal_ids: tuple[str, ...] = ()
    claims: tuple[PopulationExecutionClaim, ...]


class PopulationGraphRevisionCommand(_PopulationGateCommand):
    previous_graph_receipt_fingerprint: str
    revised_graph_receipt: PopulationDynamicGraphReceipt
    revision_evidence_fingerprint: str
    revision_ordinal: int = Field(ge=1)
    maximum_revision_count: int = Field(ge=1)


class PopulationResultSelection(_StrictFrozenModel):
    consumer_goal_id: str
    query_node_id: str
    ledger_artifact_id: str
    receipt_fingerprint: str

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationResultSelection":
        _require_text(self.consumer_goal_id, "consumer_goal_id")
        _require_text(self.query_node_id, "query_node_id")
        _require_text(self.ledger_artifact_id, "ledger_artifact_id")
        _require_text(self.receipt_fingerprint, "receipt_fingerprint")
        return self


class PopulationPostResultCommand(_PopulationGateCommand):
    graph_fingerprint: str
    selections: tuple[PopulationResultSelection, ...]


class PopulationNodePostResultCommand(_PopulationGateCommand):
    graph_fingerprint: str
    query_node_id: str
    selections: tuple[PopulationResultSelection, ...]


class PopulationGateCoordinatorIssue(_StrictFrozenModel):
    code: PopulationGateCode
    message: str
    consumer_goal_id: str = ""
    query_node_id: str = ""
    ledger_artifact_id: str = ""
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    blocking: Literal[True] = True


class PopulationGateState(_StrictFrozenModel):
    state_version: Literal["population_gate_state.v1"] = "population_gate_state.v1"
    gate_id: str
    revision: int = Field(ge=1)
    phase: PopulationGatePhase
    goal_contract_fingerprint: str
    graph_fingerprint: str = ""
    graph_binding: PopulationExecutionGraphBinding | None = None
    graph_receipt: PopulationDynamicGraphReceipt | None = None
    graph_receipt_history: tuple[PopulationDynamicGraphReceipt, ...] = ()
    graph_revision_evidence_fingerprints: tuple[str, ...] = ()
    node_gate_records: tuple[PopulationNodeGateRecord, ...] = ()
    retired_node_gate_records: tuple[PopulationNodeGateRecord, ...] = ()
    goal_attestation: PopulationVerificationAttestation
    pre_execution_attestation: PopulationVerificationAttestation | None = None
    post_result_attestation: PopulationVerificationAttestation | None = None
    ledger_snapshot_fingerprint: str = ""
    published_receipt_fingerprints: tuple[str, ...] = ()
    state_fingerprint: str = ""


def population_gate_state_fingerprint(state: PopulationGateState) -> str:
    payload = state.model_dump(by_alias=True, mode="json")
    payload["stateFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_gate_state(state: PopulationGateState) -> PopulationGateState:
    return state.model_copy(update={"state_fingerprint": population_gate_state_fingerprint(state)})


class PopulationGateTransitionResult(_StrictFrozenModel):
    accepted: bool
    committed: bool
    code: PopulationGateCode
    message: str
    state: PopulationGateState | None = None
    verification: PopulationVerificationResult | None = None
    issues: tuple[PopulationGateCoordinatorIssue, ...] = ()


class PopulationGateStateStore(Protocol):
    """Persistence boundary; production adapters must provide atomic CAS."""

    def load_population_gate(self, gate_id: str) -> PopulationGateState | None: ...

    def create_population_gate(self, state: PopulationGateState) -> bool: ...

    def compare_and_swap_population_gate(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        expected_state_fingerprint: str,
        next_state: PopulationGateState,
    ) -> bool: ...


class InMemoryPopulationGateStateStore:
    """Process-local adapter for tests and single-process composition."""

    def __init__(self) -> None:
        self._states: dict[str, PopulationGateState] = {}
        self._lock = RLock()

    def load_population_gate(self, gate_id: str) -> PopulationGateState | None:
        with self._lock:
            state = self._states.get(_text(gate_id))
            return _copy_state(state) if state is not None else None

    def create_population_gate(self, state: PopulationGateState) -> bool:
        with self._lock:
            if state.gate_id in self._states:
                return False
            self._states[state.gate_id] = _copy_state(state)
            return True

    def compare_and_swap_population_gate(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        expected_state_fingerprint: str,
        next_state: PopulationGateState,
    ) -> bool:
        with self._lock:
            current = self._states.get(gate_id)
            if (
                current is None
                or current.revision != expected_revision
                or current.state_fingerprint != expected_state_fingerprint
            ):
                return False
            self._states[gate_id] = _copy_state(next_state)
            return True


_COMPLETE_COVERAGE = {
    PopulationArtifactCoverage.ALL_ROWS.value,
    PopulationArtifactCoverage.COMPLETE.value,
    PopulationArtifactCoverage.EXACT_ENTITY_SET.value,
    PopulationArtifactCoverage.TOP_N.value,
}


class PopulationGateCoordinator:
    """CAS-bound online coordinator over the pure population verifier."""

    def __init__(
        self,
        *,
        state_store: PopulationGateStateStore,
        ledger_reader: PopulationArtifactLedgerReader,
        trusted_semantic_verifier_fingerprints: Sequence[str],
        trusted_lineage_verifier_fingerprints: Sequence[str],
        trusted_artifact_verifier_fingerprints: Sequence[str],
        trusted_ledger_authority_fingerprints: Sequence[str],
        verifier: PopulationSemanticVerifier | None = None,
    ) -> None:
        self.state_store = state_store
        self.ledger_reader = ledger_reader
        self.trusted_semantic_verifier_fingerprints = _trusted_values(trusted_semantic_verifier_fingerprints)
        self.trusted_lineage_verifier_fingerprints = _trusted_values(trusted_lineage_verifier_fingerprints)
        self.trusted_artifact_verifier_fingerprints = _trusted_values(trusted_artifact_verifier_fingerprints)
        self.trusted_ledger_authority_fingerprints = _trusted_values(trusted_ledger_authority_fingerprints)
        self.verifier = verifier or PopulationSemanticVerifier()

    def get_state(self, gate_id: str) -> PopulationGateState | None:
        state = self.state_store.load_population_gate(_text(gate_id))
        return _copy_state(state) if state is not None else None

    def commit_goal_declaration(
        self,
        command: PopulationGoalDeclarationCommand,
    ) -> PopulationGateTransitionResult:
        if command.expected_revision != 0:
            return _transition_failure(
                PopulationGateCode.CAS_REVISION_MISMATCH,
                "A new population gate must use expected revision zero.",
            )
        try:
            existing = self.state_store.load_population_gate(command.gate_id)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.STATE_STORE_FAILED,
                "The population gate state store failed closed: %s" % _bounded_error(exc),
            )
        if existing is not None:
            return _transition_failure(
                PopulationGateCode.STATE_ALREADY_EXISTS,
                "A population gate already exists for this identifier.",
                state=existing,
            )
        verification = self.verifier.verify_goal_declaration(
            GoalDeclarationPopulationVerificationInput(
                question_fingerprint=command.question_fingerprint,
                goal_skeleton_fingerprint=(command.goal_skeleton_fingerprint),
                goal_contract_fingerprint=(command.goal_contract_fingerprint),
                declaration_author_fingerprint=(command.declaration_author_fingerprint),
                semantic_review=command.semantic_review,
                trusted_semantic_verifier_fingerprints=(self.trusted_semantic_verifier_fingerprints),
                declarations=command.declarations,
            )
        )
        if not verification.passed:
            return _transition_failure(
                PopulationGateCode.VERIFICATION_REJECTED,
                "Goal-declaration population verification was rejected.",
                verification=verification,
            )
        state = seal_population_gate_state(
            PopulationGateState(
                gate_id=command.gate_id,
                revision=1,
                phase=PopulationGatePhase.GOAL_DECLARATION,
                goal_contract_fingerprint=(command.goal_contract_fingerprint),
                goal_attestation=verification.attestation,
            )
        )
        try:
            created = self.state_store.create_population_gate(state)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.STATE_STORE_FAILED,
                "The population gate state store failed closed: %s" % _bounded_error(exc),
                verification=verification,
            )
        if not created:
            try:
                current = self.get_state(command.gate_id)
            except Exception as exc:
                return _transition_failure(
                    PopulationGateCode.STATE_STORE_FAILED,
                    "The population gate state store failed closed: %s" % _bounded_error(exc),
                    verification=verification,
                )
            return _transition_failure(
                PopulationGateCode.STATE_ALREADY_EXISTS,
                "A concurrent Goal declaration already committed this gate.",
                state=current,
                verification=verification,
            )
        return _transition_success(state, verification)

    def authorize_pre_execution(
        self,
        command: PopulationPreExecutionCommand,
    ) -> PopulationGateTransitionResult:
        loaded = self._load_transition_state(
            command,
            expected_phase=PopulationGatePhase.GOAL_DECLARATION,
        )
        if isinstance(loaded, PopulationGateTransitionResult):
            return loaded
        state = loaded
        chain_issues = _attestation_chain_issues(state, require_pre=False)
        if chain_issues:
            return _transition_failure(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved Goal attestation is invalid.",
                state=state,
                issues=chain_issues,
            )
        graph_issues = _graph_binding_issues(command, state)
        if graph_issues:
            return _transition_failure(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The structured execution graph does not bind the population claims.",
                state=state,
                issues=graph_issues,
            )
        verification = self.verifier.verify_pre_execution(
            PreExecutionPopulationVerificationInput(
                goal_contract_fingerprint=(command.goal_contract_fingerprint),
                graph_fingerprint=command.graph_binding.graph_fingerprint,
                declaration_attestation=state.goal_attestation,
                trusted_lineage_verifier_fingerprints=(self.trusted_lineage_verifier_fingerprints),
                trusted_artifact_verifier_fingerprints=(self.trusted_artifact_verifier_fingerprints),
                claims=command.claims,
            )
        )
        if not verification.passed:
            return _transition_failure(
                PopulationGateCode.VERIFICATION_REJECTED,
                "Pre-execution population verification was rejected.",
                state=state,
                verification=verification,
            )
        next_state = seal_population_gate_state(
            state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "phase": PopulationGatePhase.PRE_EXECUTION,
                    "graph_fingerprint": (command.graph_binding.graph_fingerprint),
                    "graph_binding": command.graph_binding.model_copy(deep=True),
                    "pre_execution_attestation": (verification.attestation.model_copy(deep=True)),
                    "state_fingerprint": "",
                }
            )
        )
        return self._commit_existing(state, next_state, verification)

    def authorize_node_pre_execution(
        self,
        command: PopulationNodePreExecutionCommand,
    ) -> PopulationGateTransitionResult:
        """CAS-append PRE for one ready node of a Core-frozen graph."""

        loaded = self._load_incremental_state(command)
        if isinstance(loaded, PopulationGateTransitionResult):
            return loaded
        state = loaded
        chain_issues = _attestation_chain_issues(
            state,
            require_pre=False,
        )
        if chain_issues:
            return _transition_failure(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved Goal attestation is invalid.",
                state=state,
                issues=chain_issues,
            )
        graph_issues = _dynamic_graph_node_issues(command, state)
        if graph_issues:
            return _transition_failure(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The dynamic graph/current-node binding is invalid.",
                state=state,
                issues=graph_issues,
            )
        dependency_issues = _node_dependency_issues(command, state)
        if dependency_issues:
            return _transition_failure(
                PopulationGateCode.PHASE_MISMATCH,
                "A verified-artifact predecessor has not completed POST.",
                state=state,
                issues=dependency_issues,
            )
        verification = self.verifier.verify_pre_execution(
            PreExecutionPopulationVerificationInput(
                goal_contract_fingerprint=(command.goal_contract_fingerprint),
                graph_fingerprint=(command.graph_receipt.graph_fingerprint),
                declaration_attestation=state.goal_attestation,
                trusted_lineage_verifier_fingerprints=(self.trusted_lineage_verifier_fingerprints),
                trusted_artifact_verifier_fingerprints=(self.trusted_artifact_verifier_fingerprints),
                required_consumer_goal_ids=(command.required_consumer_goal_ids),
                consumer_scope_selection_explicit=True,
                claims=command.claims,
            )
        )
        if not verification.passed:
            return _transition_failure(
                PopulationGateCode.VERIFICATION_REJECTED,
                "Current-node population PRE was rejected.",
                state=state,
                verification=verification,
            )
        record = seal_population_node_gate_record(
            PopulationNodeGateRecord(
                query_node_id=command.node_binding.query_node_id,
                graph_receipt_fingerprint=(command.graph_receipt.receipt_fingerprint),
                node_binding=command.node_binding.model_copy(deep=True),
                required_consumer_goal_ids=(command.required_consumer_goal_ids),
                pre_execution_attestation=(verification.attestation.model_copy(deep=True)),
            )
        )
        next_state = seal_population_gate_state(
            state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "phase": PopulationGatePhase.PRE_EXECUTION,
                    "graph_fingerprint": (command.graph_receipt.graph_fingerprint),
                    "graph_receipt": (command.graph_receipt.model_copy(deep=True)),
                    "node_gate_records": tuple([*state.node_gate_records, record]),
                    "state_fingerprint": "",
                }
            )
        )
        return self._commit_existing(state, next_state, verification)

    def revise_dynamic_graph(
        self,
        command: PopulationGraphRevisionCommand,
    ) -> PopulationGateTransitionResult:
        """CAS-install one evidence-bound revision without rewriting records."""

        loaded = self._load_incremental_state(command)
        if isinstance(loaded, PopulationGateTransitionResult):
            return loaded
        state = loaded
        issues = _dynamic_graph_revision_issues(command, state)
        if issues:
            return _transition_failure(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The population graph revision is invalid.",
                state=state,
                issues=issues,
            )
        current_receipt = state.graph_receipt
        assert current_receipt is not None
        revised_receipt = command.revised_graph_receipt
        active_node_ids = {item.query_node_id for item in revised_receipt.nodes}
        next_active_records = tuple(item for item in state.node_gate_records if item.query_node_id in active_node_ids)
        newly_retired_records = tuple(
            item for item in state.node_gate_records if item.query_node_id not in active_node_ids
        )
        history = tuple(
            [
                *state.graph_receipt_history,
                current_receipt.model_copy(deep=True),
            ]
        )
        next_state = seal_population_gate_state(
            state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "graph_fingerprint": (revised_receipt.graph_fingerprint),
                    "graph_receipt": revised_receipt.model_copy(deep=True),
                    "graph_receipt_history": history,
                    "graph_revision_evidence_fingerprints": tuple(
                        [
                            *state.graph_revision_evidence_fingerprints,
                            command.revision_evidence_fingerprint,
                        ]
                    ),
                    "node_gate_records": next_active_records,
                    "retired_node_gate_records": tuple(
                        [
                            *state.retired_node_gate_records,
                            *newly_retired_records,
                        ]
                    ),
                    "state_fingerprint": "",
                }
            )
        )
        return self._commit_existing(state, next_state, None)

    def commit_post_result(
        self,
        command: PopulationPostResultCommand,
    ) -> PopulationGateTransitionResult:
        loaded = self._load_transition_state(
            command,
            expected_phase=PopulationGatePhase.PRE_EXECUTION,
        )
        if isinstance(loaded, PopulationGateTransitionResult):
            return loaded
        state = loaded
        if command.graph_fingerprint != state.graph_fingerprint:
            return _transition_failure(
                PopulationGateCode.BINDING_MISMATCH,
                "The post-result command belongs to a different graph.",
                state=state,
            )
        chain_issues = _attestation_chain_issues(state, require_pre=True)
        if chain_issues:
            return _transition_failure(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved Goal/PRE attestation chain is invalid.",
                state=state,
                issues=chain_issues,
            )
        try:
            ledger_authority = _text(self.ledger_reader.authority_fingerprint)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.LEDGER_READ_FAILED,
                "The artifact ledger authority could not be read: %s" % _bounded_error(exc),
                state=state,
            )
        if ledger_authority not in set(self.trusted_ledger_authority_fingerprints):
            return _transition_failure(
                PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED,
                "The artifact ledger authority is not server-trusted.",
                state=state,
            )
        try:
            ledger_snapshot = PopulationArtifactLedgerSnapshot.model_validate(
                self.ledger_reader.snapshot_population_artifacts(
                    gate_id=state.gate_id,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    graph_fingerprint=state.graph_fingerprint,
                )
            )
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.LEDGER_READ_FAILED,
                "The artifact ledger snapshot failed closed: %s" % _bounded_error(exc),
                state=state,
            )
        ledger_issues = _ledger_snapshot_issues(
            ledger_snapshot,
            state,
            expected_authority=ledger_authority,
        )
        if ledger_issues:
            issue_codes = {str(issue.code) for issue in ledger_issues}
            if PopulationGateCode.LEDGER_BINDING_MISMATCH.value in issue_codes:
                code = PopulationGateCode.LEDGER_BINDING_MISMATCH
            elif PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED.value in issue_codes:
                code = PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED
            else:
                code = PopulationGateCode.LEDGER_SNAPSHOT_INVALID
            return _transition_failure(
                code,
                "The published artifact ledger snapshot is invalid.",
                state=state,
                issues=ledger_issues,
            )
        result_evidence, selection_issues = _result_evidence_from_ledger(
            command.selections,
            ledger_snapshot,
            state,
        )
        if selection_issues:
            return _transition_failure(
                selection_issues[0].code,
                "Published result selection failed closed.",
                state=state,
                issues=selection_issues,
            )
        verification = self.verifier.verify_post_result(
            PostResultPopulationVerificationInput(
                goal_contract_fingerprint=(state.goal_contract_fingerprint),
                graph_fingerprint=state.graph_fingerprint,
                pre_execution_attestation=(state.pre_execution_attestation),
                trusted_artifact_verifier_fingerprints=(self.trusted_artifact_verifier_fingerprints),
                results=result_evidence,
            )
        )
        if not verification.passed:
            return _transition_failure(
                PopulationGateCode.VERIFICATION_REJECTED,
                "Post-result population verification was rejected.",
                state=state,
                verification=verification,
            )
        receipt_fingerprints = tuple(sorted({selection.receipt_fingerprint for selection in command.selections}))
        next_state = seal_population_gate_state(
            state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "phase": PopulationGatePhase.POST_RESULT,
                    "post_result_attestation": (verification.attestation.model_copy(deep=True)),
                    "ledger_snapshot_fingerprint": (ledger_snapshot.snapshot_fingerprint),
                    "published_receipt_fingerprints": receipt_fingerprints,
                    "state_fingerprint": "",
                }
            )
        )
        return self._commit_existing(state, next_state, verification)

    def commit_node_post_result(
        self,
        command: PopulationNodePostResultCommand,
    ) -> PopulationGateTransitionResult:
        """CAS-append POST for one PRE-authorized dynamic graph node."""

        loaded = self._load_incremental_state(command)
        if isinstance(loaded, PopulationGateTransitionResult):
            return loaded
        state = loaded
        if command.graph_fingerprint != state.graph_fingerprint:
            return _transition_failure(
                PopulationGateCode.BINDING_MISMATCH,
                "The node POST belongs to a different graph.",
                state=state,
            )
        matching = [item for item in state.node_gate_records if item.query_node_id == command.query_node_id]
        if len(matching) != 1:
            return _transition_failure(
                PopulationGateCode.PHASE_MISMATCH,
                "The node has no unique PRE authorization.",
                state=state,
            )
        record = matching[0]
        if record.post_result_attestation is not None:
            return _transition_failure(
                PopulationGateCode.PHASE_MISMATCH,
                "The node POST is already committed.",
                state=state,
            )
        record_issues = _node_record_issues(record, state)
        if record_issues:
            return _transition_failure(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The node PRE record is invalid.",
                state=state,
                issues=record_issues,
            )
        try:
            ledger_authority = _text(self.ledger_reader.authority_fingerprint)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.LEDGER_READ_FAILED,
                "The artifact ledger authority could not be read: %s" % _bounded_error(exc),
                state=state,
            )
        if ledger_authority not in set(self.trusted_ledger_authority_fingerprints):
            return _transition_failure(
                PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED,
                "The artifact ledger authority is not server-trusted.",
                state=state,
            )
        try:
            ledger_snapshot = PopulationArtifactLedgerSnapshot.model_validate(
                self.ledger_reader.snapshot_population_artifacts(
                    gate_id=state.gate_id,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    graph_fingerprint=state.graph_fingerprint,
                )
            )
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.LEDGER_READ_FAILED,
                "The artifact ledger snapshot failed closed: %s" % _bounded_error(exc),
                state=state,
            )
        ledger_issues = _ledger_snapshot_issues(
            ledger_snapshot,
            state,
            expected_authority=ledger_authority,
        )
        if ledger_issues:
            return _transition_failure(
                PopulationGateCode.LEDGER_SNAPSHOT_INVALID,
                "The published artifact ledger snapshot is invalid.",
                state=state,
                issues=ledger_issues,
            )
        result_evidence, selection_issues = _result_evidence_from_ledger(
            command.selections,
            ledger_snapshot,
            state,
        )
        if selection_issues:
            return _transition_failure(
                selection_issues[0].code,
                "Published result selection failed closed.",
                state=state,
                issues=selection_issues,
            )
        verification = self.verifier.verify_post_result(
            PostResultPopulationVerificationInput(
                goal_contract_fingerprint=(state.goal_contract_fingerprint),
                graph_fingerprint=state.graph_fingerprint,
                pre_execution_attestation=(record.pre_execution_attestation),
                trusted_artifact_verifier_fingerprints=(self.trusted_artifact_verifier_fingerprints),
                required_consumer_goal_ids=(record.required_consumer_goal_ids),
                consumer_scope_selection_explicit=True,
                results=result_evidence,
            )
        )
        if not verification.passed:
            return _transition_failure(
                PopulationGateCode.VERIFICATION_REJECTED,
                "Current-node population POST was rejected.",
                state=state,
                verification=verification,
            )
        receipt_fingerprints = tuple(sorted({item.receipt_fingerprint for item in command.selections}))
        completed = seal_population_node_gate_record(
            record.model_copy(
                update={
                    "post_result_attestation": (verification.attestation.model_copy(deep=True)),
                    "ledger_snapshot_fingerprint": (ledger_snapshot.snapshot_fingerprint),
                    "published_receipt_fingerprints": (receipt_fingerprints),
                    "record_fingerprint": "",
                }
            )
        )
        records = tuple(
            completed if item.query_node_id == command.query_node_id else item for item in state.node_gate_records
        )
        next_state = seal_population_gate_state(
            state.model_copy(
                update={
                    "revision": state.revision + 1,
                    "phase": PopulationGatePhase.POST_RESULT,
                    "node_gate_records": records,
                    "ledger_snapshot_fingerprint": (ledger_snapshot.snapshot_fingerprint),
                    "published_receipt_fingerprints": tuple(
                        sorted(
                            {
                                *state.published_receipt_fingerprints,
                                *receipt_fingerprints,
                            }
                        )
                    ),
                    "state_fingerprint": "",
                }
            )
        )
        return self._commit_existing(state, next_state, verification)

    def _load_incremental_state(
        self,
        command: _PopulationGateCommand,
    ) -> PopulationGateState | PopulationGateTransitionResult:
        try:
            stored = self.state_store.load_population_gate(command.gate_id)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.STATE_STORE_FAILED,
                "The population gate state store failed closed: %s" % _bounded_error(exc),
            )
        if stored is None:
            return _transition_failure(
                PopulationGateCode.STATE_NOT_FOUND,
                "The population gate state does not exist.",
            )
        state = _copy_state(stored)
        if state.state_fingerprint != population_gate_state_fingerprint(state):
            return _transition_failure(
                PopulationGateCode.STATE_INVALID,
                "The saved population gate state fingerprint is invalid.",
            )
        if command.expected_revision != state.revision:
            return _transition_failure(
                PopulationGateCode.CAS_REVISION_MISMATCH,
                "The population gate revision changed before this node transition.",
                state=state,
            )
        if command.goal_contract_fingerprint != state.goal_contract_fingerprint:
            return _transition_failure(
                PopulationGateCode.BINDING_MISMATCH,
                "The active Goal Contract binding changed.",
                state=state,
            )
        return state

    def _load_transition_state(
        self,
        command: _PopulationGateCommand,
        *,
        expected_phase: PopulationGatePhase,
    ) -> PopulationGateState | PopulationGateTransitionResult:
        try:
            stored = self.state_store.load_population_gate(command.gate_id)
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.STATE_STORE_FAILED,
                "The population gate state store failed closed: %s" % _bounded_error(exc),
            )
        if stored is None:
            return _transition_failure(
                PopulationGateCode.STATE_NOT_FOUND,
                "The population gate state does not exist.",
            )
        state = _copy_state(stored)
        if state.state_fingerprint != population_gate_state_fingerprint(state):
            return _transition_failure(
                PopulationGateCode.STATE_INVALID,
                "The saved population gate state fingerprint is invalid.",
            )
        if command.expected_revision != state.revision:
            return _transition_failure(
                PopulationGateCode.CAS_REVISION_MISMATCH,
                "The population gate revision changed before this transition.",
                state=state,
            )
        if str(state.phase) != expected_phase.value:
            return _transition_failure(
                PopulationGateCode.PHASE_MISMATCH,
                "The population gate is not in the required lifecycle phase.",
                state=state,
            )
        if command.goal_contract_fingerprint != state.goal_contract_fingerprint:
            return _transition_failure(
                PopulationGateCode.BINDING_MISMATCH,
                "The active Goal Contract binding changed.",
                state=state,
            )
        return state

    def _commit_existing(
        self,
        previous: PopulationGateState,
        next_state: PopulationGateState,
        verification: PopulationVerificationResult | None,
    ) -> PopulationGateTransitionResult:
        try:
            committed = self.state_store.compare_and_swap_population_gate(
                gate_id=previous.gate_id,
                expected_revision=previous.revision,
                expected_state_fingerprint=previous.state_fingerprint,
                next_state=next_state,
            )
        except Exception as exc:
            return _transition_failure(
                PopulationGateCode.STATE_STORE_FAILED,
                "The population gate state store failed closed: %s" % _bounded_error(exc),
                state=previous,
                verification=verification,
            )
        if not committed:
            try:
                current = self.get_state(previous.gate_id)
            except Exception as exc:
                return _transition_failure(
                    PopulationGateCode.STATE_STORE_FAILED,
                    "The population gate state store failed closed: %s" % _bounded_error(exc),
                    state=previous,
                    verification=verification,
                )
            return _transition_failure(
                PopulationGateCode.CAS_REVISION_MISMATCH,
                "A concurrent population gate transition won the CAS.",
                state=current,
                verification=verification,
            )
        return _transition_success(next_state, verification)


def _dynamic_edge_signature(
    edge: PopulationDynamicGraphEdge,
) -> tuple[str, str, str, str]:
    return (
        edge.source_query_node_id,
        edge.target_query_node_id,
        edge.dependency_mode,
        edge.artifact_kind,
    )


def _dynamic_graph_revision_issues(
    command: PopulationGraphRevisionCommand,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    issues: list[PopulationGateCoordinatorIssue] = []
    current = state.graph_receipt
    revised = command.revised_graph_receipt
    if current is None:
        return [
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "A dynamic graph must be active before revision.",
            )
        ]
    if not revised.receipt_fingerprint or revised.receipt_fingerprint != population_dynamic_graph_receipt_fingerprint(
        revised
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The revised dynamic graph receipt is not sealed.",
            )
        )
    if (
        command.previous_graph_receipt_fingerprint != current.receipt_fingerprint
        or revised.parent_receipt_fingerprint != current.receipt_fingerprint
        or revised.graph_version != current.graph_version + 1
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.CAS_REVISION_MISMATCH,
                "The revised graph is not the next child of the active receipt.",
                details={
                    "activeGraphVersion": current.graph_version,
                    "revisedGraphVersion": revised.graph_version,
                },
            )
        )
    if (
        not command.revision_evidence_fingerprint
        or command.revision_evidence_fingerprint != revised.revision_evidence_fingerprint
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The population graph revision lacks bound structured evidence.",
            )
        )
    completed_revisions = len(state.graph_revision_evidence_fingerprints)
    if command.revision_ordinal != completed_revisions + 1 or completed_revisions >= command.maximum_revision_count:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.PHASE_MISMATCH,
                "The population graph revision budget is exhausted or out of order.",
            )
        )
    if command.revision_evidence_fingerprint in set(state.graph_revision_evidence_fingerprints):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.BINDING_MISMATCH,
                "A population graph revision trigger cannot be replayed.",
            )
        )

    current_nodes = {item.query_node_id: item for item in current.nodes}
    revised_nodes = {item.query_node_id: item for item in revised.nodes}
    carried_ids = set(revised.carried_forward_query_node_ids)
    retired_ids = set(revised.retired_query_node_ids)
    removed_ids = set(current_nodes) - set(revised_nodes)
    if removed_ids != retired_ids:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "Every removed node must be explicitly retired by the revision.",
                details={
                    "removedQueryNodeIds": sorted(removed_ids),
                    "retiredQueryNodeIds": sorted(retired_ids),
                },
            )
        )
    retained_ids = set(current_nodes).intersection(revised_nodes)
    if carried_ids != retained_ids:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "Every retained node must be explicitly carried forward.",
            )
        )
    current_incoming: dict[
        str,
        set[tuple[str, str, str, str]],
    ] = {node_id: set() for node_id in current_nodes}
    revised_incoming: dict[
        str,
        set[tuple[str, str, str, str]],
    ] = {node_id: set() for node_id in revised_nodes}
    for edge in current.edges:
        current_incoming[edge.target_query_node_id].add(_dynamic_edge_signature(edge))
    for edge in revised.edges:
        revised_incoming[edge.target_query_node_id].add(_dynamic_edge_signature(edge))
    for node_id in sorted(retained_ids):
        if (
            set(current_nodes[node_id].consumer_goal_ids) != set(revised_nodes[node_id].consumer_goal_ids)
            or current_incoming[node_id] != revised_incoming[node_id]
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.BINDING_MISMATCH,
                    "A carried node or its input population lineage changed.",
                    query_node_id=node_id,
                )
            )

    records_by_node = {item.query_node_id: item for item in state.node_gate_records}
    for node_id, record in records_by_node.items():
        if record.post_result_attestation is not None:
            if node_id not in carried_ids:
                issues.append(
                    _coordinator_issue(
                        PopulationGateCode.BINDING_MISMATCH,
                        "A published node must be carried into the next revision.",
                        query_node_id=node_id,
                    )
                )
        elif node_id not in retired_ids:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.BINDING_MISMATCH,
                    "A consumed PRE without POST must be retired before recovery.",
                    query_node_id=node_id,
                )
            )
        issues.extend(_node_record_issues(record, state))

    historical_ids = {
        item.query_node_id
        for receipt in (
            *state.graph_receipt_history,
            current,
        )
        for item in receipt.nodes
    }
    added_ids = set(revised_nodes) - set(current_nodes)
    replayed_node_ids = added_ids.intersection(historical_ids)
    if replayed_node_ids:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.BINDING_MISMATCH,
                "A new revision cannot reuse a historical query-node identity.",
                details={"queryNodeIds": sorted(replayed_node_ids)},
            )
        )
    issues.extend(_dynamic_graph_population_issues(revised, state))
    return _dedupe_issues(issues)


def _dynamic_graph_node_issues(
    command: PopulationNodePreExecutionCommand,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    receipt = command.graph_receipt
    node = command.node_binding
    issues: list[PopulationGateCoordinatorIssue] = []
    if not receipt.receipt_fingerprint or receipt.receipt_fingerprint != population_dynamic_graph_receipt_fingerprint(
        receipt
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The dynamic graph receipt fingerprint is invalid.",
                path="graphReceipt.receiptFingerprint",
            )
        )
    if state.graph_receipt is not None and (state.graph_receipt.receipt_fingerprint != receipt.receipt_fingerprint):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.BINDING_MISMATCH,
                "A node cannot switch the frozen dynamic graph receipt.",
                query_node_id=node.query_node_id,
            )
        )
    if state.graph_fingerprint and (state.graph_fingerprint != receipt.graph_fingerprint):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.BINDING_MISMATCH,
                "A node cannot switch the frozen graph fingerprint.",
                query_node_id=node.query_node_id,
            )
        )
    graph_nodes = {item.query_node_id: item for item in receipt.nodes}
    declared_node = graph_nodes.get(node.query_node_id)
    if declared_node is None:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The current node is absent from the dynamic graph receipt.",
                query_node_id=node.query_node_id,
            )
        )
    elif set(declared_node.consumer_goal_ids) != set(node.consumer_goal_ids):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The current node Goal assignment differs from the frozen graph.",
                query_node_id=node.query_node_id,
            )
        )
    if any(item.query_node_id == node.query_node_id for item in state.node_gate_records):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.BINDING_MISMATCH,
                "The graph node already consumed its PRE authorization.",
                query_node_id=node.query_node_id,
            )
        )
    attested_consumers = {item.consumer_goal_id for item in state.goal_attestation.accepted_scopes}
    expected_consumers = tuple(sorted(attested_consumers.intersection(node.consumer_goal_ids)))
    if set(command.required_consumer_goal_ids) != set(expected_consumers):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The node population consumer selection is incomplete.",
                query_node_id=node.query_node_id,
                details={
                    "expectedConsumerGoalIds": list(expected_consumers),
                    "actualConsumerGoalIds": list(command.required_consumer_goal_ids),
                },
            )
        )
    for claim in command.claims:
        if (
            claim.query_node_id != node.query_node_id
            or claim.consumer_goal_id not in set(command.required_consumer_goal_ids)
            or claim.generation != node.generation
            or claim.attempt_id != node.attempt_id
            or claim.query_contract_fingerprint != node.query_contract_fingerprint
            or claim.sql_ast_fingerprint != node.sql_ast_fingerprint
            or claim.effective_scope.snapshot_fingerprint != node.snapshot_fingerprint
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.GRAPH_BINDING_INVALID,
                    "A population claim does not match the current node execution identity.",
                    consumer_goal_id=claim.consumer_goal_id,
                    query_node_id=claim.query_node_id,
                )
            )
        for proof in claim.lineage_proofs:
            if (
                proof.graph_fingerprint != receipt.graph_fingerprint
                or proof.query_node_id != node.query_node_id
                or proof.generation != node.generation
                or proof.attempt_id != node.attempt_id
                or proof.query_contract_fingerprint != node.query_contract_fingerprint
                or proof.sql_ast_fingerprint != node.sql_ast_fingerprint
                or proof.result_snapshot_fingerprint != node.snapshot_fingerprint
            ):
                issues.append(
                    _coordinator_issue(
                        PopulationGateCode.GRAPH_BINDING_INVALID,
                        "A lineage proof does not match the current graph node.",
                        consumer_goal_id=claim.consumer_goal_id,
                        query_node_id=node.query_node_id,
                    )
                )
    issues.extend(_dynamic_graph_population_issues(receipt, state))
    return _dedupe_issues(issues)


def _dynamic_graph_population_issues(
    receipt: PopulationDynamicGraphReceipt,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    issues: list[PopulationGateCoordinatorIssue] = []
    node_ids_by_goal: dict[str, set[str]] = {}
    for node in receipt.nodes:
        for goal_id in node.consumer_goal_ids:
            node_ids_by_goal.setdefault(goal_id, set()).add(node.query_node_id)
    adjacency: dict[str, set[str]] = {node.query_node_id: set() for node in receipt.nodes}
    artifact_adjacency: dict[str, set[str]] = {node.query_node_id: set() for node in receipt.nodes}
    for edge in receipt.edges:
        adjacency[edge.source_query_node_id].add(edge.target_query_node_id)
        if edge.dependency_mode == "VERIFIED_ARTIFACT":
            artifact_adjacency[edge.source_query_node_id].add(edge.target_query_node_id)

    def has_path(
        source: str,
        target: str,
        *,
        artifact_only: bool,
    ) -> bool:
        if source == target:
            return True
        graph = artifact_adjacency if artifact_only else adjacency
        pending = [source]
        visited = {source}
        cursor = 0
        while cursor < len(pending):
            current = pending[cursor]
            cursor += 1
            for candidate in graph.get(current, set()):
                if candidate == target:
                    return True
                if candidate in visited:
                    continue
                visited.add(candidate)
                pending.append(candidate)
        return False

    for scope in state.goal_attestation.accepted_scopes:
        consumers = node_ids_by_goal.get(scope.consumer_goal_id, set())
        if len(consumers) != 1:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.GRAPH_BINDING_INVALID,
                    "A population consumer Goal must have one graph owner.",
                    consumer_goal_id=scope.consumer_goal_id,
                )
            )
            continue
        consumer_node = next(iter(consumers))
        artifact_only = _enum_text(scope.scope_kind) in {
            "VERIFIED_ENTITY_SET",
            "VERIFIED_RESULT_ARTIFACT",
        }
        for source_goal_id in scope.source_goal_ids:
            sources = node_ids_by_goal.get(source_goal_id, set())
            if not sources or not any(
                has_path(
                    source,
                    consumer_node,
                    artifact_only=artifact_only,
                )
                for source in sources
            ):
                issues.append(
                    _coordinator_issue(
                        PopulationGateCode.GRAPH_BINDING_INVALID,
                        "The dynamic graph omits required population lineage.",
                        consumer_goal_id=scope.consumer_goal_id,
                        query_node_id=consumer_node,
                        details={"sourceGoalId": source_goal_id},
                    )
                )
    return issues


def _node_dependency_issues(
    command: PopulationNodePreExecutionCommand,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    target = command.node_binding.query_node_id
    records = {item.query_node_id: item for item in state.node_gate_records}
    issues: list[PopulationGateCoordinatorIssue] = []
    for edge in command.graph_receipt.edges:
        if edge.target_query_node_id != target or edge.dependency_mode != "VERIFIED_ARTIFACT":
            continue
        source = records.get(edge.source_query_node_id)
        if source is None or source.post_result_attestation is None:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.PHASE_MISMATCH,
                    "The artifact dependency source has no committed POST.",
                    query_node_id=target,
                    details={
                        "sourceQueryNodeId": edge.source_query_node_id,
                        "artifactKind": edge.artifact_kind,
                    },
                )
            )
            continue
        issues.extend(_node_record_issues(source, state))
    return _dedupe_issues(issues)


def _node_record_issues(
    record: PopulationNodeGateRecord,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    issues: list[PopulationGateCoordinatorIssue] = []
    if (
        not record.record_fingerprint
        or record.record_fingerprint != population_node_gate_record_fingerprint(record)
        or record.query_node_id != record.node_binding.query_node_id
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved node gate record fingerprint is invalid.",
                query_node_id=record.query_node_id,
            )
        )
        return issues
    matching_receipts = tuple(
        receipt
        for receipt in (
            *state.graph_receipt_history,
            *((state.graph_receipt,) if state.graph_receipt is not None else ()),
        )
        if receipt.receipt_fingerprint == record.graph_receipt_fingerprint
    )
    if len(matching_receipts) != 1:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The node record graph receipt is absent or ambiguous.",
                query_node_id=record.query_node_id,
            )
        )
        return issues
    record_receipt = matching_receipts[0]
    receipt_nodes = {item.query_node_id: item for item in record_receipt.nodes}
    receipt_node = receipt_nodes.get(record.query_node_id)
    if receipt_node is None or set(receipt_node.consumer_goal_ids) != set(record.node_binding.consumer_goal_ids):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The node record does not match its immutable graph receipt.",
                query_node_id=record.query_node_id,
            )
        )
        return issues
    pre = record.pre_execution_attestation
    goal = state.goal_attestation
    if (
        not pre.passed
        or not pre.gate_open
        or _enum_text(pre.stage) != "PRE_EXECUTION"
        or pre.goal_contract_fingerprint != state.goal_contract_fingerprint
        or pre.graph_fingerprint != record_receipt.graph_fingerprint
        or pre.previous_attestation_fingerprint != goal.attestation_fingerprint
        or pre.attestation_fingerprint != population_attestation_fingerprint(pre)
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved node PRE attestation is invalid.",
                query_node_id=record.query_node_id,
            )
        )
    post = record.post_result_attestation
    if post is not None and (
        not post.passed
        or not post.gate_open
        or _enum_text(post.stage) != "POST_RESULT"
        or post.goal_contract_fingerprint != state.goal_contract_fingerprint
        or post.graph_fingerprint != record_receipt.graph_fingerprint
        or post.previous_attestation_fingerprint != pre.attestation_fingerprint
        or post.attestation_fingerprint != population_attestation_fingerprint(post)
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The saved node POST attestation is invalid.",
                query_node_id=record.query_node_id,
            )
        )
    return issues


def _enum_text(value: Any) -> str:
    return _text(getattr(value, "value", value))


def _graph_binding_issues(
    command: PopulationPreExecutionCommand,
    state: PopulationGateState,
) -> list[PopulationGateCoordinatorIssue]:
    binding = command.graph_binding
    issues: list[PopulationGateCoordinatorIssue] = []
    if not binding.binding_fingerprint or binding.binding_fingerprint != population_execution_graph_binding_fingerprint(
        binding
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.GRAPH_BINDING_INVALID,
                "The structured graph binding fingerprint is invalid.",
                path="graphBinding.bindingFingerprint",
            )
        )
    node_map = {node.query_node_id: node for node in binding.nodes}
    for claim in command.claims:
        node = node_map.get(claim.query_node_id)
        if node is None:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.GRAPH_BINDING_INVALID,
                    "A population claim references a query node absent from the graph binding.",
                    consumer_goal_id=claim.consumer_goal_id,
                    query_node_id=claim.query_node_id,
                    path="claims.queryNodeId",
                )
            )
            continue
        if claim.consumer_goal_id not in set(node.consumer_goal_ids):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.GRAPH_BINDING_INVALID,
                    "The graph node does not authorize this population consumer Goal.",
                    consumer_goal_id=claim.consumer_goal_id,
                    query_node_id=claim.query_node_id,
                )
            )
        comparisons = (
            (
                claim.generation,
                node.generation,
                "generation",
            ),
            (
                claim.attempt_id,
                node.attempt_id,
                "attemptId",
            ),
            (
                claim.query_contract_fingerprint,
                node.query_contract_fingerprint,
                "queryContractFingerprint",
            ),
            (
                claim.sql_ast_fingerprint,
                node.sql_ast_fingerprint,
                "sqlAstFingerprint",
            ),
            (
                claim.effective_scope.snapshot_fingerprint,
                node.snapshot_fingerprint,
                "snapshotFingerprint",
            ),
        )
        for actual, expected, field_name in comparisons:
            if actual != expected:
                issues.append(
                    _coordinator_issue(
                        PopulationGateCode.GRAPH_BINDING_INVALID,
                        "The population claim does not match the graph node %s." % field_name,
                        consumer_goal_id=claim.consumer_goal_id,
                        query_node_id=claim.query_node_id,
                        path="claims.%s" % field_name,
                    )
                )
        for proof in claim.lineage_proofs:
            proof_comparisons = (
                (
                    proof.graph_fingerprint,
                    binding.graph_fingerprint,
                    "graphFingerprint",
                ),
                (
                    proof.query_node_id,
                    node.query_node_id,
                    "queryNodeId",
                ),
                (
                    proof.generation,
                    node.generation,
                    "generation",
                ),
                (
                    proof.attempt_id,
                    node.attempt_id,
                    "attemptId",
                ),
                (
                    proof.query_contract_fingerprint,
                    node.query_contract_fingerprint,
                    "queryContractFingerprint",
                ),
                (
                    proof.sql_ast_fingerprint,
                    node.sql_ast_fingerprint,
                    "sqlAstFingerprint",
                ),
                (
                    proof.source_snapshot_fingerprint,
                    claim.required_scope.snapshot_fingerprint,
                    "sourceSnapshotFingerprint",
                ),
                (
                    proof.result_snapshot_fingerprint,
                    node.snapshot_fingerprint,
                    "resultSnapshotFingerprint",
                ),
            )
            for actual, expected, field_name in proof_comparisons:
                if not actual or actual != expected:
                    issues.append(
                        _coordinator_issue(
                            PopulationGateCode.GRAPH_BINDING_INVALID,
                            "A lineage proof does not match the frozen graph %s." % field_name,
                            consumer_goal_id=claim.consumer_goal_id,
                            query_node_id=claim.query_node_id,
                            path="claims.lineageProofs.%s" % field_name,
                        )
                    )
    return _dedupe_issues(issues)


def _attestation_chain_issues(
    state: PopulationGateState,
    *,
    require_pre: bool,
) -> list[PopulationGateCoordinatorIssue]:
    issues: list[PopulationGateCoordinatorIssue] = []
    goal = state.goal_attestation
    if (
        not goal.passed
        or not goal.gate_open
        or str(goal.stage) != PopulationVerificationStage.GOAL_DECLARATION.value
        or goal.goal_contract_fingerprint != state.goal_contract_fingerprint
        or goal.attestation_fingerprint != population_attestation_fingerprint(goal)
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                "The immutable Goal-declaration attestation is invalid.",
            )
        )
    pre = state.pre_execution_attestation
    if require_pre:
        if pre is None:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                    "The PRE_EXECUTION attestation is missing.",
                )
            )
        elif (
            not pre.passed
            or not pre.gate_open
            or str(pre.stage) != PopulationVerificationStage.PRE_EXECUTION.value
            or pre.goal_contract_fingerprint != state.goal_contract_fingerprint
            or pre.graph_fingerprint != state.graph_fingerprint
            or pre.previous_attestation_fingerprint != goal.attestation_fingerprint
            or pre.attestation_fingerprint != population_attestation_fingerprint(pre)
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.ATTESTATION_CHAIN_INVALID,
                    "The immutable PRE_EXECUTION attestation chain is invalid.",
                )
            )
    return issues


def _ledger_snapshot_issues(
    snapshot: PopulationArtifactLedgerSnapshot,
    state: PopulationGateState,
    *,
    expected_authority: str,
) -> list[PopulationGateCoordinatorIssue]:
    issues: list[PopulationGateCoordinatorIssue] = []
    if (
        not snapshot.snapshot_fingerprint
        or snapshot.snapshot_fingerprint != population_artifact_ledger_snapshot_fingerprint(snapshot)
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.LEDGER_SNAPSHOT_INVALID,
                "The artifact ledger snapshot fingerprint is invalid.",
                path="ledgerSnapshot.snapshotFingerprint",
            )
        )
    if snapshot.ledger_authority_fingerprint != expected_authority:
        issues.append(
            _coordinator_issue(
                PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED,
                "The ledger snapshot authority differs from the injected reader authority.",
            )
        )
    if (
        snapshot.goal_contract_fingerprint != state.goal_contract_fingerprint
        or snapshot.graph_fingerprint != state.graph_fingerprint
    ):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.LEDGER_BINDING_MISMATCH,
                "The artifact ledger snapshot belongs to another Goal or graph.",
            )
        )
    entry_ids = [entry.ledger_artifact_id for entry in snapshot.entries]
    if len(entry_ids) != len(set(entry_ids)):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.LEDGER_SNAPSHOT_INVALID,
                "The artifact ledger contains duplicate artifact identifiers.",
            )
        )
    for entry in snapshot.entries:
        if (
            not entry.entry_fingerprint
            or entry.entry_fingerprint != population_artifact_ledger_entry_fingerprint(entry)
            or entry.ledger_artifact_id != entry.receipt.ledger_artifact_id
            or entry.publication_status != entry.receipt.publication_status
            or not entry.receipt.receipt_fingerprint
            or entry.receipt.receipt_fingerprint != population_published_artifact_receipt_fingerprint(entry.receipt)
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.LEDGER_SNAPSHOT_INVALID,
                    "A ledger entry or published receipt fingerprint is invalid.",
                    ledger_artifact_id=entry.ledger_artifact_id,
                )
            )
    return _dedupe_issues(issues)


def _result_evidence_from_ledger(
    selections: Sequence[PopulationResultSelection],
    snapshot: PopulationArtifactLedgerSnapshot,
    state: PopulationGateState,
) -> tuple[
    tuple[PopulationResultEvidence, ...],
    list[PopulationGateCoordinatorIssue],
]:
    issues: list[PopulationGateCoordinatorIssue] = []
    identities = [(item.consumer_goal_id, item.query_node_id) for item in selections]
    if len(identities) != len(set(identities)):
        issues.append(
            _coordinator_issue(
                PopulationGateCode.RESULT_SELECTION_DUPLICATE,
                "A consumer Goal/query node has duplicate result selections.",
            )
        )
    entries = {entry.ledger_artifact_id: entry for entry in snapshot.entries}
    graph_binding = state.graph_binding
    nodes = (
        {item.query_node_id: item for item in graph_binding.nodes}
        if graph_binding is not None
        else {item.query_node_id: item.node_binding for item in state.node_gate_records}
    )
    results: list[PopulationResultEvidence] = []
    for selection in selections:
        entry = entries.get(selection.ledger_artifact_id)
        common = {
            "consumer_goal_id": selection.consumer_goal_id,
            "query_node_id": selection.query_node_id,
            "ledger_artifact_id": selection.ledger_artifact_id,
        }
        if entry is None:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_NOT_IN_LEDGER,
                    "The selected result artifact is absent from the verified ledger.",
                    **common,
                )
            )
            continue
        receipt = entry.receipt
        evidence = receipt.evidence
        node = nodes.get(selection.query_node_id)
        if selection.receipt_fingerprint != receipt.receipt_fingerprint:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_RECEIPT_MISMATCH,
                    "The selected receipt fingerprint does not match the ledger.",
                    **common,
                )
            )
        if entry.publication_status != "PUBLISHED" or receipt.publication_status != "PUBLISHED":
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_NOT_PUBLISHED,
                    "Only PUBLISHED ledger results may enter population finalization.",
                    **common,
                )
            )
        if not evidence.verified:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_NOT_VERIFIED,
                    "The ledger result has no passed verification evidence.",
                    **common,
                )
            )
        if not evidence.immutable:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_MUTABLE,
                    "A mutable result cannot prove population semantics.",
                    **common,
                )
            )
        if str(evidence.coverage) not in _COMPLETE_COVERAGE:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_COVERAGE_INCOMPLETE,
                    "PREVIEW, PARTIAL, or unknown coverage cannot prove a complete result.",
                    **common,
                )
            )
        if receipt.result_is_truncated:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_TRUNCATED,
                    "A truncated result cannot prove complete population membership.",
                    **common,
                )
            )
        if receipt.stored_row_count != receipt.exact_result_row_count:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_COUNT_MISMATCH,
                    "Stored and exact result row counts do not match.",
                    **common,
                )
            )
        if (
            receipt.goal_contract_fingerprint != state.goal_contract_fingerprint
            or receipt.graph_fingerprint != state.graph_fingerprint
            or receipt.query_node_id != selection.query_node_id
            or selection.consumer_goal_id not in set(receipt.covered_consumer_goal_ids)
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_BINDING_MISMATCH,
                    "The published receipt belongs to another population gate binding.",
                    **common,
                )
            )
        if node is None:
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_BINDING_MISMATCH,
                    "The selected query node is absent from the frozen graph.",
                    **common,
                )
            )
        elif (
            receipt.generation != node.generation
            or receipt.attempt_id != node.attempt_id
            or evidence.goal_contract_fingerprint != state.goal_contract_fingerprint
            or evidence.graph_fingerprint != state.graph_fingerprint
            or evidence.query_contract_fingerprint != node.query_contract_fingerprint
            or evidence.sql_ast_fingerprint != node.sql_ast_fingerprint
            or evidence.snapshot_fingerprint != node.snapshot_fingerprint
        ):
            issues.append(
                _coordinator_issue(
                    PopulationGateCode.RESULT_BINDING_MISMATCH,
                    "The published result does not match its query node execution identity.",
                    **common,
                )
            )
        results.append(
            PopulationResultEvidence(
                consumer_goal_id=selection.consumer_goal_id,
                query_node_id=selection.query_node_id,
                result_artifact=evidence,
                lineage_proof_fingerprints=(evidence.lineage_proof_fingerprints),
            )
        )
    return tuple(results), _dedupe_issues(issues)


def _copy_state(state: PopulationGateState) -> PopulationGateState:
    return PopulationGateState.model_validate(state.model_dump(by_alias=True, mode="json"))


def _trusted_values(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_text(value) for value in values if _text(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError("trusted authority fingerprints must be unique")
    return normalized


def _coordinator_issue(
    code: PopulationGateCode,
    message: str,
    *,
    consumer_goal_id: str = "",
    query_node_id: str = "",
    ledger_artifact_id: str = "",
    path: str = "",
    details: Mapping[str, Any] | None = None,
) -> PopulationGateCoordinatorIssue:
    return PopulationGateCoordinatorIssue(
        code=code,
        message=message,
        consumer_goal_id=consumer_goal_id,
        query_node_id=query_node_id,
        ledger_artifact_id=ledger_artifact_id,
        path=path,
        details=dict(details or {}),
    )


def _dedupe_issues(
    issues: Sequence[PopulationGateCoordinatorIssue],
) -> list[PopulationGateCoordinatorIssue]:
    retained: list[PopulationGateCoordinatorIssue] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for issue in issues:
        identity = (
            str(issue.code),
            issue.consumer_goal_id,
            issue.query_node_id,
            issue.ledger_artifact_id,
            issue.path,
        )
        if identity in seen:
            continue
        seen.add(identity)
        retained.append(issue)
    return retained


def _transition_success(
    state: PopulationGateState,
    verification: PopulationVerificationResult | None,
) -> PopulationGateTransitionResult:
    return PopulationGateTransitionResult(
        accepted=True,
        committed=True,
        code=PopulationGateCode.COMMITTED,
        message="Population gate transition committed.",
        state=_copy_state(state),
        verification=verification,
        issues=(),
    )


def _transition_failure(
    code: PopulationGateCode,
    message: str,
    *,
    state: PopulationGateState | None = None,
    verification: PopulationVerificationResult | None = None,
    issues: Sequence[PopulationGateCoordinatorIssue] = (),
) -> PopulationGateTransitionResult:
    return PopulationGateTransitionResult(
        accepted=False,
        committed=False,
        code=code,
        message=message,
        state=_copy_state(state) if state is not None else None,
        verification=verification,
        issues=tuple(issues),
    )


def _bounded_error(error: Exception) -> str:
    return ("%s: %s" % (type(error).__name__, str(error)))[:500]
