from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Optional, Sequence

from pydantic import ConfigDict, Field, model_validator
from sqlglot import exp, parse_one

from merchant_ai.models import APIModel, DataSnapshotContract
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_execution_identity import (
    grounded_data_snapshot_fingerprint,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationArtifactLedgerEntry,
    PopulationDynamicGraphReceipt,
    PopulationExecutionNodeBinding,
    PopulationGraphRevisionCommand,
    PopulationGateCoordinator,
    PopulationGateCode,
    PopulationNodePostResultCommand,
    PopulationNodePreExecutionCommand,
    PopulationResultSelection,
    population_dynamic_graph_receipt_fingerprint,
    population_node_gate_record_fingerprint,
)
from merchant_ai.services.grounded_population_online_gate import (
    GroundedWorkspacePopulationGateStateStore,
    PopulationOnlineGateCallResult,
    PopulationOnlineGateFacade,
    PublishedGroundedPopulationLedgerReader,
)
from merchant_ai.services.grounded_population_semantic_reviewer import (
    IndependentPopulationSemanticReviewer,
    PopulationSemanticReviewProvider,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationConstraintEvidence,
    PopulationConstraintKind,
    PopulationExecutionClaim,
    PopulationLineageMechanism,
    PopulationLineageProof,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _enum_value(value: Any) -> str:
    return _text(getattr(value, "value", value))


def _stable_fingerprint(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: Any, field_name: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


class PopulationPreExecutionNodeReference(_StrictFrozenModel):
    query_node_id: str
    consumer_goal_ids: tuple[str, ...]
    generation: int = Field(ge=1)
    attempt_id: str
    query_contract_fingerprint: str
    expected_sql_ast_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationPreExecutionNodeReference":
        for field_name in (
            "query_node_id",
            "attempt_id",
            "query_contract_fingerprint",
        ):
            _require_text(getattr(self, field_name), field_name)
        normalized = tuple(_require_text(item, "consumer_goal_ids") for item in self.consumer_goal_ids)
        if not normalized:
            raise ValueError("consumer_goal_ids must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("consumer_goal_ids must not contain duplicates")
        return self


class PopulationPreExecutionReference(_StrictFrozenModel):
    """Sealed server hand-off; the datasource snapshot is deliberately absent.

    The Core freezes the actual graph/node/Contract/AST identity.  QueryExecutor
    adds the live DataSnapshot only after SQL and ACL checks and immediately
    before the Doris side effect.
    """

    reference_version: str = "population_pre_execution_reference.v2"
    gate_id: str
    context_owner_fingerprint: str
    run_authority_fingerprint: str
    goal_contract_fingerprint: str
    graph_receipt: PopulationDynamicGraphReceipt
    node: PopulationPreExecutionNodeReference
    reference_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationPreExecutionReference":
        for field_name in (
            "gate_id",
            "context_owner_fingerprint",
            "run_authority_fingerprint",
            "goal_contract_fingerprint",
        ):
            _require_text(getattr(self, field_name), field_name)
        if (
            not self.graph_receipt.receipt_fingerprint
            or self.graph_receipt.receipt_fingerprint
            != population_dynamic_graph_receipt_fingerprint(self.graph_receipt)
        ):
            raise ValueError("graph_receipt must be sealed")
        matches = tuple(item for item in self.graph_receipt.nodes if item.query_node_id == self.node.query_node_id)
        if len(matches) != 1 or set(matches[0].consumer_goal_ids) != set(self.node.consumer_goal_ids):
            raise ValueError("node must match one Goal assignment in graph_receipt")
        return self


def population_pre_execution_reference_fingerprint(
    reference: PopulationPreExecutionReference,
) -> str:
    payload = reference.model_dump(by_alias=True, mode="json")
    payload["referenceFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_population_pre_execution_reference(
    reference: PopulationPreExecutionReference,
) -> PopulationPreExecutionReference:
    return reference.model_copy(
        update={"reference_fingerprint": (population_pre_execution_reference_fingerprint(reference))}
    )


def population_pre_execution_reference_valid(
    reference: PopulationPreExecutionReference,
) -> bool:
    declared = _text(reference.reference_fingerprint)
    return bool(declared and declared == population_pre_execution_reference_fingerprint(reference))


class GroundedPopulationRuntimeGateError(RuntimeError):
    def __init__(self, code: str, *, result: Any = None) -> None:
        self.code = _text(code) or "POPULATION_RUNTIME_GATE_FAILED"
        self.result = result
        super().__init__(self.code)


@dataclass(frozen=True)
class _PopulationRunAuthority:
    workspace: GroundedContextWorkspace
    ledger_provider: Callable[[], Sequence[Any]]


@dataclass(frozen=True)
class PopulationExecutorNodeEvidence:
    """Live executor evidence captured after SQL/ACL, before Doris."""

    query_node_id: str
    contract: GroundedQueryContract
    compilation: Any
    data_snapshot: DataSnapshotContract
    actual_sql_ast_fingerprint: str


@dataclass(frozen=True)
class _CrossNodePopulationLineage:
    required_scope: PopulationScopeDescriptor
    effective_scope: PopulationScopeDescriptor
    artifact_evidence: tuple[PopulationArtifactEvidence, ...]
    source_node_ids: tuple[str, ...]
    entity_mapping_fingerprint: str = ""
    relationship_ref_ids: tuple[str, ...] = ()
    grain_mapping_fingerprint: str = ""


class GroundedPopulationExecutionGate:
    """Server authority joining GOAL, executor PRE, and published-ledger POST.

    This class never chooses a query count, branch layout, or execution order.
    It accepts the dynamic graph receipt chosen by Core, then authorizes only
    the current live node with an atomic revision transition.
    """

    def __init__(
        self,
        *,
        settings: Any,
        semantic_provider: PopulationSemanticReviewProvider,
        declaration_author_fingerprint: str,
        semantic_authority_fingerprint: str,
        lineage_authority_fingerprint: str,
        artifact_authority_fingerprint: str,
        ledger_authority_fingerprint: str,
        semantic_timeout_seconds: float,
    ) -> None:
        if settings is None:
            raise ValueError("settings are required")
        self.settings = settings
        self.semantic_provider = semantic_provider
        self.declaration_author_fingerprint = _require_text(
            declaration_author_fingerprint,
            "declaration_author_fingerprint",
        )
        self.semantic_authority_fingerprint = _require_text(
            semantic_authority_fingerprint,
            "semantic_authority_fingerprint",
        )
        self.lineage_authority_fingerprint = _require_text(
            lineage_authority_fingerprint,
            "lineage_authority_fingerprint",
        )
        self.artifact_authority_fingerprint = _require_text(
            artifact_authority_fingerprint,
            "artifact_authority_fingerprint",
        )
        self.ledger_authority_fingerprint = _require_text(
            ledger_authority_fingerprint,
            "ledger_authority_fingerprint",
        )
        if self.declaration_author_fingerprint == self.semantic_authority_fingerprint:
            raise ValueError("Core declaration and semantic reviewer authorities must differ")
        self.semantic_timeout_seconds = float(semantic_timeout_seconds)
        if self.semantic_timeout_seconds <= 0:
            raise ValueError("semantic_timeout_seconds must be positive")
        self.maximum_graph_revision_count = max(
            1,
            int(
                getattr(
                    settings,
                    "grounded_execution_graph_max_revisions",
                    2,
                )
                or 2
            ),
        )
        self._runs: dict[str, _PopulationRunAuthority] = {}
        self._lock = RLock()

    @staticmethod
    def authority_fingerprint(role: str, deployment_identity: Any) -> str:
        return _stable_fingerprint(
            {
                "authorityRole": _require_text(role, "role"),
                "deploymentIdentity": deployment_identity,
                "protocol": "grounded_population_runtime_gate.v1",
            }
        )

    def register_run(
        self,
        *,
        workspace: GroundedContextWorkspace,
        ledger_provider: Callable[[], Sequence[Any]],
    ) -> None:
        if not isinstance(workspace, GroundedContextWorkspace):
            raise GroundedPopulationRuntimeGateError("POPULATION_WORKSPACE_REQUIRED")
        if not callable(ledger_provider):
            raise GroundedPopulationRuntimeGateError("POPULATION_LEDGER_PROVIDER_REQUIRED")
        owner = _require_text(
            workspace.owner_fingerprint,
            "workspace.owner_fingerprint",
        )
        run_authority = _require_text(
            workspace.request_fingerprint,
            "workspace.request_fingerprint",
        )
        authority = _PopulationRunAuthority(
            workspace=workspace,
            ledger_provider=ledger_provider,
        )
        with self._lock:
            existing = self._runs.get(run_authority)
            if existing is not None and (
                existing.workspace.root != workspace.root
                or existing.workspace.owner_fingerprint != owner
                or existing.workspace.request_fingerprint != run_authority
            ):
                raise GroundedPopulationRuntimeGateError("POPULATION_RUN_WORKSPACE_CONFLICT")
            self._runs[run_authority] = authority

    def gate_id(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
    ) -> str:
        return (
            "population_gate_%s"
            % _stable_fingerprint(
                {
                    "contextOwnerFingerprint": _require_text(
                        context_owner_fingerprint,
                        "context_owner_fingerprint",
                    ),
                    "runAuthorityFingerprint": _require_text(
                        run_authority_fingerprint,
                        "run_authority_fingerprint",
                    ),
                    "goalContractFingerprint": _require_text(
                        goal_contract_fingerprint,
                        "goal_contract_fingerprint",
                    ),
                }
            )[:32]
        )

    def commit_goal(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        exact_question: str,
        goal_contract: OriginalQuestionGoalContract,
    ) -> PopulationOnlineGateCallResult:
        authority = self._run_authority(
            context_owner_fingerprint,
            run_authority_fingerprint,
        )
        fingerprint = original_question_goal_contract_fingerprint(goal_contract)
        facade = self._facade(authority)
        return facade.commit_goal(
            gate_id=self.gate_id(
                context_owner_fingerprint=context_owner_fingerprint,
                run_authority_fingerprint=run_authority_fingerprint,
                goal_contract_fingerprint=fingerprint,
            ),
            expected_revision=0,
            exact_question=exact_question,
            goal_contract=goal_contract,
        )

    def revise_graph(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
        previous_graph_receipt_fingerprint: str,
        revised_graph_receipt: PopulationDynamicGraphReceipt,
        revision_evidence_fingerprint: str,
    ) -> PopulationOnlineGateCallResult:
        """CAS-install one sealed revision and invalidate its parent receipt."""

        authority = self._run_authority(
            context_owner_fingerprint,
            run_authority_fingerprint,
        )
        facade = self._facade(authority)
        gate_id = self.gate_id(
            context_owner_fingerprint=context_owner_fingerprint,
            run_authority_fingerprint=run_authority_fingerprint,
            goal_contract_fingerprint=goal_contract_fingerprint,
        )
        revised = (
            revised_graph_receipt
            if isinstance(
                revised_graph_receipt,
                PopulationDynamicGraphReceipt,
            )
            else PopulationDynamicGraphReceipt.model_validate(revised_graph_receipt)
        )
        attempts = self.maximum_graph_revision_count + 2
        for _attempt in range(attempts):
            state = facade.coordinator.get_state(gate_id)
            if state is None or state.graph_receipt is None:
                return self._failure(
                    PopulationVerificationStage.PRE_EXECUTION,
                    "POPULATION_GRAPH_STATE_NOT_FOUND",
                )
            if (
                state.graph_receipt.receipt_fingerprint == revised.receipt_fingerprint
                and revision_evidence_fingerprint in set(state.graph_revision_evidence_fingerprints)
            ):
                return PopulationOnlineGateCallResult(
                    stage=PopulationVerificationStage.PRE_EXECUTION,
                    accepted=True,
                    code="GRAPH_REVISION_ALREADY_COMMITTED",
                    message="The population graph revision is already active.",
                )
            if state.graph_receipt.receipt_fingerprint != previous_graph_receipt_fingerprint:
                return self._failure(
                    PopulationVerificationStage.PRE_EXECUTION,
                    "POPULATION_GRAPH_REVISION_STALE",
                )
            transition = facade.coordinator.revise_dynamic_graph(
                PopulationGraphRevisionCommand(
                    gate_id=gate_id,
                    expected_revision=state.revision,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    previous_graph_receipt_fingerprint=(previous_graph_receipt_fingerprint),
                    revised_graph_receipt=revised,
                    revision_evidence_fingerprint=(revision_evidence_fingerprint),
                    revision_ordinal=(len(state.graph_revision_evidence_fingerprints) + 1),
                    maximum_revision_count=(self.maximum_graph_revision_count),
                )
            )
            code = str(getattr(transition.code, "value", transition.code))
            if transition.accepted:
                return PopulationOnlineGateCallResult(
                    stage=PopulationVerificationStage.PRE_EXECUTION,
                    accepted=True,
                    code=code,
                    message=transition.message,
                    transition=transition,
                )
            if code != PopulationGateCode.CAS_REVISION_MISMATCH.value:
                return PopulationOnlineGateCallResult(
                    stage=PopulationVerificationStage.PRE_EXECUTION,
                    accepted=False,
                    code=code,
                    message=transition.message,
                    transition=transition,
                )
        return self._failure(
            PopulationVerificationStage.PRE_EXECUTION,
            "POPULATION_GRAPH_REVISION_CAS_CONTENTION",
        )

    def build_pre_execution_reference(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
        graph_receipt: PopulationDynamicGraphReceipt,
        node: PopulationPreExecutionNodeReference,
    ) -> PopulationPreExecutionReference:
        authority = self._run_authority(
            context_owner_fingerprint,
            run_authority_fingerprint,
        )
        facade = self._facade(authority)
        gate_id = self.gate_id(
            context_owner_fingerprint=context_owner_fingerprint,
            run_authority_fingerprint=run_authority_fingerprint,
            goal_contract_fingerprint=goal_contract_fingerprint,
        )
        state = facade.coordinator.get_state(gate_id)
        if state is None:
            raise GroundedPopulationRuntimeGateError("POPULATION_GOAL_STATE_NOT_FOUND")
        if state.goal_contract_fingerprint != _text(goal_contract_fingerprint):
            raise GroundedPopulationRuntimeGateError("POPULATION_GOAL_STATE_NOT_READY")
        normalized_receipt = (
            graph_receipt
            if isinstance(graph_receipt, PopulationDynamicGraphReceipt)
            else PopulationDynamicGraphReceipt.model_validate(graph_receipt)
        )
        normalized_node = (
            node
            if isinstance(node, PopulationPreExecutionNodeReference)
            else PopulationPreExecutionNodeReference.model_validate(node)
        )
        if (
            not normalized_receipt.receipt_fingerprint
            or normalized_receipt.receipt_fingerprint
            != population_dynamic_graph_receipt_fingerprint(normalized_receipt)
        ):
            raise GroundedPopulationRuntimeGateError("POPULATION_DYNAMIC_GRAPH_RECEIPT_INVALID")
        if state.graph_receipt is not None and (
            state.graph_receipt.receipt_fingerprint != normalized_receipt.receipt_fingerprint
        ):
            raise GroundedPopulationRuntimeGateError("POPULATION_DYNAMIC_GRAPH_RECEIPT_CHANGED")
        return seal_population_pre_execution_reference(
            PopulationPreExecutionReference(
                gate_id=gate_id,
                context_owner_fingerprint=(authority.workspace.owner_fingerprint),
                run_authority_fingerprint=(authority.workspace.request_fingerprint),
                goal_contract_fingerprint=state.goal_contract_fingerprint,
                graph_receipt=normalized_receipt,
                node=normalized_node,
            )
        )

    def authorize_node(
        self,
        *,
        reference: PopulationPreExecutionReference,
        execution: PopulationExecutorNodeEvidence,
    ) -> PopulationOnlineGateCallResult:
        """CAS-append one ready node's PRE immediately before Doris."""

        if not population_pre_execution_reference_valid(reference):
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "POPULATION_PRE_REFERENCE_INVALID",
            )
        authority = self._run_authority(
            reference.context_owner_fingerprint,
            reference.run_authority_fingerprint,
        )
        if not isinstance(execution, PopulationExecutorNodeEvidence):
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "POPULATION_PRE_NODE_EVIDENCE_INVALID",
            )
        node_reference = reference.node
        actual_contract_fingerprint = grounded_query_contract_fingerprint(execution.contract)
        ast_fingerprint = _text(execution.actual_sql_ast_fingerprint)
        if (
            _text(execution.query_node_id) != node_reference.query_node_id
            or actual_contract_fingerprint != node_reference.query_contract_fingerprint
            or not ast_fingerprint
            or (
                node_reference.expected_sql_ast_fingerprint
                and node_reference.expected_sql_ast_fingerprint != ast_fingerprint
            )
        ):
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "POPULATION_PRE_EXECUTION_IDENTITY_MISMATCH",
            )
        if not isinstance(execution.data_snapshot, DataSnapshotContract):
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "POPULATION_PRE_SNAPSHOT_REQUIRED",
            )
        snapshot_fingerprint = grounded_data_snapshot_fingerprint(execution.data_snapshot)
        node_binding = PopulationExecutionNodeBinding(
            query_node_id=node_reference.query_node_id,
            consumer_goal_ids=node_reference.consumer_goal_ids,
            generation=node_reference.generation,
            attempt_id=node_reference.attempt_id,
            query_contract_fingerprint=actual_contract_fingerprint,
            sql_ast_fingerprint=ast_fingerprint,
            snapshot_fingerprint=snapshot_fingerprint,
        )
        facade = self._facade(authority)
        attempts = len(reference.graph_receipt.nodes) + 1
        for _attempt in range(attempts):
            state = facade.coordinator.get_state(reference.gate_id)
            if state is None:
                return self._failure(
                    PopulationVerificationStage.PRE_EXECUTION,
                    "POPULATION_GOAL_STATE_NOT_FOUND",
                )
            required_consumers = tuple(
                sorted(
                    {item.consumer_goal_id for item in state.goal_attestation.accepted_scopes}.intersection(
                        node_reference.consumer_goal_ids
                    )
                )
            )
            try:
                source_entries: tuple[PopulationArtifactLedgerEntry, ...] = ()
                if self._requires_published_source_ledger(
                    reference,
                    execution.contract,
                ):
                    source_ledger = facade.coordinator.ledger_reader.snapshot_population_artifacts(
                        gate_id=state.gate_id,
                        goal_contract_fingerprint=(state.goal_contract_fingerprint),
                        graph_fingerprint=(reference.graph_receipt.graph_fingerprint),
                    )
                    source_entries = tuple(source_ledger.entries)
                claims = self._execution_claims(
                    state=state,
                    reference=reference,
                    execution=execution,
                    snapshot_fingerprint=snapshot_fingerprint,
                    sql_ast_fingerprint=ast_fingerprint,
                    required_consumer_goal_ids=required_consumers,
                    source_entries=source_entries,
                )
            except Exception:
                return self._failure(
                    PopulationVerificationStage.PRE_EXECUTION,
                    "POPULATION_PRE_LINEAGE_EVIDENCE_INVALID",
                )
            result = facade.authorize_node_pre_execution(
                PopulationNodePreExecutionCommand(
                    gate_id=reference.gate_id,
                    expected_revision=state.revision,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    graph_receipt=reference.graph_receipt,
                    node_binding=node_binding,
                    required_consumer_goal_ids=required_consumers,
                    claims=claims,
                )
            )
            if result.accepted or result.code != (PopulationGateCode.CAS_REVISION_MISMATCH.value):
                return result
        return self._failure(
            PopulationVerificationStage.PRE_EXECUTION,
            "POPULATION_PRE_CAS_CONTENTION",
        )

    def commit_node_post_result(
        self,
        *,
        reference: PopulationPreExecutionReference,
    ) -> PopulationOnlineGateCallResult:
        """CAS-append one node's POST from integrity-valid PUBLISHED receipts."""

        if not population_pre_execution_reference_valid(reference):
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_PRE_REFERENCE_INVALID",
            )
        authority = self._run_authority(
            reference.context_owner_fingerprint,
            reference.run_authority_fingerprint,
        )
        facade = self._facade(authority)
        attempts = len(reference.graph_receipt.nodes) + 1
        for _attempt in range(attempts):
            state = facade.coordinator.get_state(reference.gate_id)
            if state is None:
                return self._failure(
                    PopulationVerificationStage.POST_RESULT,
                    "POPULATION_PRE_STATE_NOT_FOUND",
                )
            matching_records = tuple(
                item for item in state.node_gate_records if item.query_node_id == reference.node.query_node_id
            )
            if len(matching_records) != 1:
                return self._failure(
                    PopulationVerificationStage.POST_RESULT,
                    "POPULATION_NODE_PRE_NOT_FOUND",
                )
            record = matching_records[0]
            if record.post_result_attestation is not None:
                return self._failure(
                    PopulationVerificationStage.POST_RESULT,
                    "POPULATION_NODE_POST_REPLAY_REJECTED",
                )
            try:
                ledger = facade.coordinator.ledger_reader.snapshot_population_artifacts(
                    gate_id=state.gate_id,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    graph_fingerprint=state.graph_fingerprint,
                )
            except Exception:
                return self._failure(
                    PopulationVerificationStage.POST_RESULT,
                    "POPULATION_PUBLISHED_LEDGER_UNAVAILABLE",
                )
            selections: list[PopulationResultSelection] = []
            for scope in record.pre_execution_attestation.accepted_scopes:
                matching_entries = tuple(
                    entry
                    for entry in ledger.entries
                    if entry.receipt.query_node_id == scope.query_node_id
                    and scope.consumer_goal_id in set(entry.receipt.covered_consumer_goal_ids)
                )
                if len(matching_entries) != 1:
                    return self._failure(
                        PopulationVerificationStage.POST_RESULT,
                        "POPULATION_PUBLISHED_RESULT_AMBIGUOUS",
                    )
                entry = matching_entries[0]
                selections.append(
                    PopulationResultSelection(
                        consumer_goal_id=scope.consumer_goal_id,
                        query_node_id=scope.query_node_id,
                        ledger_artifact_id=entry.ledger_artifact_id,
                        receipt_fingerprint=(entry.receipt.receipt_fingerprint),
                    )
                )
            result = facade.commit_node_post_result(
                PopulationNodePostResultCommand(
                    gate_id=state.gate_id,
                    expected_revision=state.revision,
                    goal_contract_fingerprint=(state.goal_contract_fingerprint),
                    graph_fingerprint=state.graph_fingerprint,
                    query_node_id=reference.node.query_node_id,
                    selections=tuple(selections),
                )
            )
            if result.accepted or result.code != (PopulationGateCode.CAS_REVISION_MISMATCH.value):
                return result
        return self._failure(
            PopulationVerificationStage.POST_RESULT,
            "POPULATION_POST_CAS_CONTENTION",
        )

    def require_graph_complete(
        self,
        *,
        reference: PopulationPreExecutionReference,
    ) -> PopulationOnlineGateCallResult:
        """Require every frozen graph node to have a valid PRE/POST chain."""

        if not population_pre_execution_reference_valid(reference):
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_PRE_REFERENCE_INVALID",
            )
        authority = self._run_authority(
            reference.context_owner_fingerprint,
            reference.run_authority_fingerprint,
        )
        state = self._facade(authority).coordinator.get_state(reference.gate_id)
        if state is None or state.graph_receipt is None:
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_GRAPH_STATE_NOT_FOUND",
            )
        receipt = reference.graph_receipt
        if (
            state.graph_receipt.receipt_fingerprint != receipt.receipt_fingerprint
            or state.graph_fingerprint != receipt.graph_fingerprint
        ):
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_GRAPH_BINDING_MISMATCH",
            )
        expected_nodes = {item.query_node_id: item for item in receipt.nodes}
        records = {item.query_node_id: item for item in state.node_gate_records}
        if (
            len(state.node_gate_records) != len(expected_nodes)
            or len(records) != len(expected_nodes)
            or set(records) != set(expected_nodes)
        ):
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_GRAPH_NODE_COVERAGE_INCOMPLETE",
            )
        post_consumers: set[str] = set()
        for node_id, node in expected_nodes.items():
            record = records[node_id]
            post = record.post_result_attestation
            record_receipts = tuple(
                candidate
                for candidate in (
                    *tuple(
                        getattr(
                            state,
                            "graph_receipt_history",
                            (),
                        )
                    ),
                    state.graph_receipt,
                )
                if candidate.receipt_fingerprint == record.graph_receipt_fingerprint
            )
            record_receipt = record_receipts[0] if len(record_receipts) == 1 else None
            record_receipt_node = next(
                (
                    item
                    for item in (record_receipt.nodes if record_receipt is not None else ())
                    if item.query_node_id == node_id
                ),
                None,
            )
            pre = record.pre_execution_attestation
            if (
                record.record_fingerprint != population_node_gate_record_fingerprint(record)
                or record_receipt is None
                or record_receipt_node is None
                or (
                    record.graph_receipt_fingerprint != receipt.receipt_fingerprint
                    and node_id not in set(receipt.carried_forward_query_node_ids)
                )
                or set(record.node_binding.consumer_goal_ids) != set(node.consumer_goal_ids)
                or set(record_receipt_node.consumer_goal_ids) != set(node.consumer_goal_ids)
                or not pre.passed
                or not pre.gate_open
                or pre.graph_fingerprint != record_receipt.graph_fingerprint
                or pre.attestation_fingerprint != population_attestation_fingerprint(pre)
                or post is None
                or not post.passed
                or not post.gate_open
                or post.graph_fingerprint != record_receipt.graph_fingerprint
                or post.attestation_fingerprint != population_attestation_fingerprint(post)
                or post.previous_attestation_fingerprint != record.pre_execution_attestation.attestation_fingerprint
            ):
                return self._failure(
                    PopulationVerificationStage.POST_RESULT,
                    "POPULATION_GRAPH_ATTESTATION_INCOMPLETE",
                )
            post_consumers.update(item.consumer_goal_id for item in post.accepted_scopes)
        expected_consumers = {item.consumer_goal_id for item in state.goal_attestation.accepted_scopes}
        if post_consumers != expected_consumers:
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "POPULATION_GRAPH_SCOPE_COVERAGE_INCOMPLETE",
            )
        return PopulationOnlineGateCallResult(
            stage=PopulationVerificationStage.POST_RESULT,
            accepted=True,
            code="GRAPH_COMPLETE",
            message="Every frozen graph node has a valid population POST.",
        )

    def _execution_claims(
        self,
        *,
        state: Any,
        reference: PopulationPreExecutionReference,
        execution: PopulationExecutorNodeEvidence,
        snapshot_fingerprint: str,
        sql_ast_fingerprint: str,
        required_consumer_goal_ids: Sequence[str],
        source_entries: Sequence[PopulationArtifactLedgerEntry] = (),
    ) -> tuple[PopulationExecutionClaim, ...]:
        claims: list[PopulationExecutionClaim] = []
        required_ids = set(required_consumer_goal_ids)
        node_reference = reference.node
        for attested in state.goal_attestation.accepted_scopes:
            if attested.consumer_goal_id not in required_ids:
                continue
            contract = execution.contract
            compilation = execution.compilation
            sql = _text(getattr(compilation, "sql", ""))
            required = self._resolved_scope(
                attested,
                contract=contract,
                snapshot_fingerprint=snapshot_fingerprint,
            )
            cross_node = self._cross_node_population_lineage(
                state=state,
                reference=reference,
                contract=contract,
                attested=attested,
                resolved_scope=required,
                current_snapshot_fingerprint=snapshot_fingerprint,
                source_entries=source_entries,
            )
            if cross_node is None:
                mechanism, preserved, complete = self._lineage_observation(
                    required,
                    contract=contract,
                    sql=sql,
                    consumer_goal_ids=(node_reference.consumer_goal_ids),
                )
                effective = required.model_copy(deep=True)
                artifact_evidence: tuple[PopulationArtifactEvidence, ...] = ()
                source_node_ids = (node_reference.query_node_id,)
                entity_mapping_fingerprint = ""
                relationship_ref_ids: tuple[str, ...] = ()
                grain_mapping_fingerprint = ""
            else:
                required = cross_node.required_scope
                effective = cross_node.effective_scope
                mechanism = PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT
                preserved = required.constraints
                complete = True
                artifact_evidence = cross_node.artifact_evidence
                source_node_ids = cross_node.source_node_ids
                entity_mapping_fingerprint = cross_node.entity_mapping_fingerprint
                relationship_ref_ids = cross_node.relationship_ref_ids
                grain_mapping_fingerprint = cross_node.grain_mapping_fingerprint
            proof = PopulationLineageProof(
                proof_id="population_proof_%s"
                % _stable_fingerprint(
                    {
                        "gateId": reference.gate_id,
                        "consumerGoalId": attested.consumer_goal_id,
                        "queryNodeId": node_reference.query_node_id,
                        "generation": node_reference.generation,
                        "attemptId": node_reference.attempt_id,
                        "sqlAstFingerprint": sql_ast_fingerprint,
                        "sourceArtifactFingerprints": sorted(item.artifact_fingerprint for item in artifact_evidence),
                    }
                )[:32],
                mechanism=mechanism,
                verifier_fingerprint=self.lineage_authority_fingerprint,
                verified=True,
                graph_fingerprint=(reference.graph_receipt.graph_fingerprint),
                query_node_id=node_reference.query_node_id,
                generation=node_reference.generation,
                attempt_id=node_reference.attempt_id,
                query_contract_fingerprint=(node_reference.query_contract_fingerprint),
                sql_ast_fingerprint=sql_ast_fingerprint,
                source_population_fingerprint=(required.population_fingerprint),
                result_population_fingerprint=(effective.population_fingerprint),
                source_goal_ids=required.source_goal_ids,
                source_node_ids=source_node_ids,
                preserved_constraints=preserved,
                artifact_evidence=artifact_evidence,
                source_entity_identity_ref=required.entity_identity_ref,
                result_entity_identity_ref=effective.entity_identity_ref,
                entity_mapping_fingerprint=entity_mapping_fingerprint,
                relationship_ref_ids=relationship_ref_ids,
                source_grain_fingerprint=required.grain_fingerprint,
                result_grain_fingerprint=effective.grain_fingerprint,
                grain_mapping_fingerprint=grain_mapping_fingerprint,
                source_snapshot_fingerprint=(required.snapshot_fingerprint),
                result_snapshot_fingerprint=(effective.snapshot_fingerprint),
                complete_membership=complete,
            )
            claims.append(
                PopulationExecutionClaim(
                    consumer_goal_id=attested.consumer_goal_id,
                    query_node_id=node_reference.query_node_id,
                    generation=node_reference.generation,
                    attempt_id=node_reference.attempt_id,
                    declaration_scope_fingerprint=(attested.declaration_scope_fingerprint),
                    required_scope=required,
                    effective_scope=effective,
                    query_contract_fingerprint=(node_reference.query_contract_fingerprint),
                    sql_ast_fingerprint=sql_ast_fingerprint,
                    lineage_proofs=(proof,),
                )
            )
        return tuple(claims)

    @staticmethod
    def _requires_published_source_ledger(
        reference: PopulationPreExecutionReference,
        contract: GroundedQueryContract,
    ) -> bool:
        target_node_id = reference.node.query_node_id
        return bool(
            contract.upstream_entity_bindings
            and any(
                edge.target_query_node_id == target_node_id and edge.dependency_mode == "VERIFIED_ARTIFACT"
                for edge in reference.graph_receipt.edges
            )
        )

    def _cross_node_population_lineage(
        self,
        *,
        state: Any,
        reference: PopulationPreExecutionReference,
        contract: GroundedQueryContract,
        attested: Any,
        resolved_scope: PopulationScopeDescriptor,
        current_snapshot_fingerprint: str,
        source_entries: Sequence[PopulationArtifactLedgerEntry],
    ) -> _CrossNodePopulationLineage | None:
        source_goal_ids = tuple(dict.fromkeys(_text(item) for item in attested.source_goal_ids))
        if not source_goal_ids or set(source_goal_ids).issubset(reference.node.consumer_goal_ids):
            return None
        if not contract.upstream_entity_bindings or not source_entries:
            return None

        target_node_id = reference.node.query_node_id
        incoming_source_nodes = {
            edge.source_query_node_id
            for edge in reference.graph_receipt.edges
            if edge.target_query_node_id == target_node_id and edge.dependency_mode == "VERIFIED_ARTIFACT"
        }
        graph_nodes = {item.query_node_id: item for item in reference.graph_receipt.nodes}
        records_by_node: dict[str, list[Any]] = {}
        for record in state.node_gate_records:
            records_by_node.setdefault(record.query_node_id, []).append(record)

        selected_evidence: dict[str, PopulationArtifactEvidence] = {}
        selected_bindings: list[Any] = []
        selected_source_nodes: list[str] = []
        selected_source_scopes: list[Any] = []
        for source_goal_id in source_goal_ids:
            matching_source_nodes = tuple(
                node
                for node in graph_nodes.values()
                if source_goal_id in set(node.consumer_goal_ids) and node.query_node_id in incoming_source_nodes
            )
            if len(matching_source_nodes) != 1:
                return None
            source_node = matching_source_nodes[0]
            source_records = tuple(records_by_node.get(source_node.query_node_id, ()))
            if len(source_records) != 1:
                return None
            source_record = source_records[0]
            source_post = source_record.post_result_attestation
            if source_post is None:
                return None
            source_scopes = tuple(
                scope
                for scope in source_post.accepted_scopes
                if scope.consumer_goal_id == source_goal_id and scope.query_node_id == source_node.query_node_id
            )
            if len(source_scopes) != 1:
                return None
            source_scope = source_scopes[0]

            matching_pairs: list[tuple[Any, PopulationArtifactLedgerEntry]] = []
            for binding in contract.upstream_entity_bindings:
                if not self._upstream_binding_targets_contract(
                    binding,
                    contract,
                ):
                    continue
                for entry in source_entries:
                    if self._published_source_entry_matches(
                        entry=entry,
                        binding=binding,
                        source_goal_id=source_goal_id,
                        source_node_id=source_node.query_node_id,
                        source_record=source_record,
                        source_scope=source_scope,
                    ):
                        matching_pairs.append((binding, entry))
            artifact_ids = {pair[1].receipt.evidence.artifact_id for pair in matching_pairs}
            if len(artifact_ids) != 1:
                return None
            artifact_id = next(iter(artifact_ids))
            artifact_pairs = tuple(
                pair for pair in matching_pairs if pair[1].receipt.evidence.artifact_id == artifact_id
            )
            evidence = artifact_pairs[0][1].receipt.evidence
            selected_evidence[artifact_id] = evidence.model_copy(deep=True)
            selected_bindings.extend(pair[0] for pair in artifact_pairs)
            selected_source_nodes.append(source_node.query_node_id)
            selected_source_scopes.append(source_scope)

        population_fingerprints = {
            scope.population_fingerprint for scope in selected_source_scopes if _text(scope.population_fingerprint)
        }
        source_snapshots = {
            scope.snapshot_fingerprint for scope in selected_source_scopes if _text(scope.snapshot_fingerprint)
        }
        source_grains = {scope.grain_fingerprint for scope in selected_source_scopes if _text(scope.grain_fingerprint)}
        source_identities = {
            scope.entity_identity_ref for scope in selected_source_scopes if _text(scope.entity_identity_ref)
        }
        if (
            len(population_fingerprints) != 1
            or len(source_snapshots) != 1
            or len(source_grains) != 1
            or len(source_identities) != 1
        ):
            return None
        source_snapshot = next(iter(source_snapshots))
        if source_snapshot != current_snapshot_fingerprint:
            return None
        source_population = next(iter(population_fingerprints))
        source_grain = next(iter(source_grains))
        source_identity = next(iter(source_identities))
        result_identity = _text(resolved_scope.entity_identity_ref)
        result_grain = _text(resolved_scope.grain_fingerprint)
        if not result_identity or not result_grain:
            return None

        relationship_ref_ids: tuple[str, ...] = ()
        entity_mapping_fingerprint = ""
        if source_identity != result_identity:
            relationship_ref_ids = self._binding_relationship_refs(
                contract,
                selected_bindings,
            )
            if not relationship_ref_ids:
                return None
            entity_mapping_fingerprint = _stable_fingerprint(
                {
                    "sourceEntityIdentity": source_identity,
                    "resultEntityIdentity": result_identity,
                    "bindings": selected_bindings,
                    "relationshipRefIds": relationship_ref_ids,
                }
            )
        grain_mapping_fingerprint = ""
        if source_grain != result_grain:
            grain_mapping_fingerprint = _stable_fingerprint(
                {
                    "sourceGrainFingerprint": source_grain,
                    "resultGrainFingerprint": result_grain,
                    "bindings": selected_bindings,
                    "relationshipRefIds": relationship_ref_ids,
                }
            )

        artifact_evidence = tuple(selected_evidence[key] for key in sorted(selected_evidence))
        required_scope = resolved_scope.model_copy(
            update={
                "source_artifact_ids": tuple(item.artifact_id for item in artifact_evidence),
                "population_fingerprint": source_population,
                "entity_identity_ref": source_identity,
                "grain_fingerprint": source_grain,
                "snapshot_fingerprint": source_snapshot,
            },
            deep=True,
        )
        effective_scope = resolved_scope.model_copy(
            update={
                "source_artifact_ids": tuple(item.artifact_id for item in artifact_evidence),
                "population_fingerprint": source_population,
                "entity_identity_ref": result_identity,
                "grain_fingerprint": result_grain,
                "snapshot_fingerprint": current_snapshot_fingerprint,
            },
            deep=True,
        )
        return _CrossNodePopulationLineage(
            required_scope=required_scope,
            effective_scope=effective_scope,
            artifact_evidence=artifact_evidence,
            source_node_ids=tuple(dict.fromkeys(selected_source_nodes)),
            entity_mapping_fingerprint=entity_mapping_fingerprint,
            relationship_ref_ids=relationship_ref_ids,
            grain_mapping_fingerprint=grain_mapping_fingerprint,
        )

    @staticmethod
    def _upstream_binding_targets_contract(
        binding: Any,
        contract: GroundedQueryContract,
    ) -> bool:
        if any(
            not _text(getattr(binding, field_name, ""))
            for field_name in (
                "entity_set_artifact_id",
                "source_query_artifact_id",
                "source_contract_fingerprint",
                "source_sql_fingerprint",
                "source_column",
                "source_entity_identity",
                "target_field_ref",
                "target_table",
                "target_column",
                "target_entity_identity",
                "values_hash",
            )
        ):
            return False
        if _text(binding.operator).upper() != "IN":
            return False
        matching_filters = tuple(
            item
            for item in contract.entity_filters
            if item.semantic_ref_id == binding.target_field_ref
            and item.table == binding.target_table
            and item.column == binding.target_column
            and _text(item.operator).upper() == "IN"
            and item.entity_identity == binding.target_entity_identity
        )
        if len(matching_filters) != 1:
            return False
        values = matching_filters[0].literal_value
        if not isinstance(values, (list, tuple)):
            return False
        return bool(
            int(binding.value_count or 0) == len(values) and binding.values_hash == _stable_fingerprint(list(values))
        )

    @staticmethod
    def _published_source_entry_matches(
        *,
        entry: PopulationArtifactLedgerEntry,
        binding: Any,
        source_goal_id: str,
        source_node_id: str,
        source_record: Any,
        source_scope: Any,
    ) -> bool:
        receipt = entry.receipt
        evidence = receipt.evidence
        complete_coverage = {
            PopulationArtifactCoverage.ALL_ROWS.value,
            PopulationArtifactCoverage.COMPLETE.value,
            PopulationArtifactCoverage.EXACT_ENTITY_SET.value,
            PopulationArtifactCoverage.TOP_N.value,
        }
        return bool(
            entry.publication_status == "PUBLISHED"
            and receipt.publication_status == "PUBLISHED"
            and receipt.source_query_artifact_id == binding.source_query_artifact_id
            and receipt.query_node_id == source_node_id
            and source_goal_id in set(receipt.covered_consumer_goal_ids)
            and receipt.generation == source_record.node_binding.generation
            and receipt.attempt_id == source_record.node_binding.attempt_id
            and evidence.query_contract_fingerprint
            == binding.source_contract_fingerprint
            == source_record.node_binding.query_contract_fingerprint
            and evidence.sql_ast_fingerprint
            == binding.source_sql_fingerprint
            == source_record.node_binding.sql_ast_fingerprint
            and evidence.snapshot_fingerprint
            == source_scope.snapshot_fingerprint
            == source_record.node_binding.snapshot_fingerprint
            and evidence.population_fingerprint == source_scope.population_fingerprint
            and binding.source_entity_identity == source_scope.entity_identity_ref
            and evidence.coverage in complete_coverage
            and evidence.verified
            and evidence.immutable
        )

    @staticmethod
    def _binding_relationship_refs(
        contract: GroundedQueryContract,
        bindings: Sequence[Any],
    ) -> tuple[str, ...]:
        refs: set[str] = set()
        for binding in bindings:
            for relationship in contract.relationships:
                if binding.target_table not in {
                    relationship.left_table,
                    relationship.right_table,
                }:
                    continue
                if any(
                    binding.source_column in set(key_pair) and binding.target_column in set(key_pair)
                    for key_pair in relationship.keys
                ):
                    refs.add(relationship.semantic_ref_id)
        return tuple(sorted(ref for ref in refs if _text(ref)))

    def _resolved_scope(
        self,
        attested: Any,
        *,
        contract: GroundedQueryContract,
        snapshot_fingerprint: str,
    ) -> PopulationScopeDescriptor:
        semantics = self._population_semantics(contract)
        population_fingerprint = _stable_fingerprint(
            {
                "scopeKind": _enum_value(attested.scope_kind),
                "sourceGoalIds": sorted(attested.source_goal_ids),
                "sourceArtifactIds": sorted(attested.source_artifact_ids),
                "semantics": semantics,
            }
        )
        grain_fingerprint = _stable_fingerprint(
            {
                "tableGrains": [
                    {
                        "tableRef": item.detail_ref_id,
                        "dataGrain": item.data_grain,
                    }
                    for item in contract.tables
                ],
                "groupEntityIdentities": sorted(
                    _text(item.entity_identity) for item in contract.dimensions if _text(item.entity_identity)
                ),
            }
        )
        identities = sorted(
            {
                _text(value)
                for value in [
                    *[item.entity_identity for item in contract.selected_fields],
                    *[item.entity_identity for item in contract.dimensions],
                    *[item.entity_identity for item in contract.entity_filters],
                    *[item.target_entity_identity for item in contract.upstream_entity_bindings],
                ]
                if _text(value)
            }
        )
        entity_identity_ref = (
            identities[0] if len(identities) == 1 else _stable_fingerprint(identities) if identities else ""
        )
        constraints = self._population_constraints(
            contract,
            population_fingerprint=population_fingerprint,
            dependent=(
                _enum_value(attested.scope_kind)
                not in {
                    PopulationScopeKind.INDEPENDENT.value,
                    PopulationScopeKind.UNIVERSE.value,
                }
            ),
        )
        return PopulationScopeDescriptor(
            scope_id="resolved_population_%s"
            % _stable_fingerprint(
                {
                    "consumerGoalId": attested.consumer_goal_id,
                    "populationFingerprint": population_fingerprint,
                }
            )[:32],
            kind=attested.scope_kind,
            source_goal_ids=tuple(attested.source_goal_ids),
            source_artifact_ids=tuple(attested.source_artifact_ids),
            population_fingerprint=population_fingerprint,
            entity_identity_ref=entity_identity_ref,
            grain_fingerprint=grain_fingerprint,
            snapshot_fingerprint=snapshot_fingerprint,
            constraints=constraints,
            complete_membership_required=(attested.complete_membership_required),
        )

    @staticmethod
    def _population_semantics(
        contract: GroundedQueryContract,
    ) -> dict[str, Any]:
        return {
            "tables": [
                {
                    "detailRefId": item.detail_ref_id,
                    "table": item.table,
                    "dataGrain": item.data_grain,
                }
                for item in contract.tables
            ],
            "timeRange": contract.time_range.model_dump(
                by_alias=True,
                mode="json",
            ),
            "timeField": contract.time_field.model_dump(
                by_alias=True,
                mode="json",
            ),
            "entityFilters": [item.model_dump(by_alias=True, mode="json") for item in contract.entity_filters],
            "upstreamEntityBindings": [
                item.model_dump(by_alias=True, mode="json") for item in contract.upstream_entity_bindings
            ],
            "relationships": [item.model_dump(by_alias=True, mode="json") for item in contract.relationships],
            "referenceScope": contract.reference_scope.model_dump(
                by_alias=True,
                mode="json",
            ),
        }

    def _population_constraints(
        self,
        contract: GroundedQueryContract,
        *,
        population_fingerprint: str,
        dependent: bool,
    ) -> tuple[PopulationConstraintEvidence, ...]:
        constraints: list[PopulationConstraintEvidence] = []
        if bool(contract.time_range.explicit):
            constraints.append(
                PopulationConstraintEvidence(
                    fingerprint=_stable_fingerprint(
                        {
                            "constraintKind": "TIME",
                            "timeRange": contract.time_range,
                            "timeField": contract.time_field,
                        }
                    ),
                    kind=PopulationConstraintKind.TIME,
                    semantic_ref_ids=tuple(item for item in (_text(contract.time_field.semantic_ref_id),) if item),
                )
            )
        for item in contract.entity_filters:
            constraints.append(
                PopulationConstraintEvidence(
                    fingerprint=_stable_fingerprint(
                        {
                            "constraintKind": "PREDICATE",
                            "binding": item,
                        }
                    ),
                    kind=PopulationConstraintKind.PREDICATE,
                    semantic_ref_ids=(item.semantic_ref_id,),
                )
            )
        for item in contract.relationships:
            constraints.append(
                PopulationConstraintEvidence(
                    fingerprint=_stable_fingerprint(
                        {
                            "constraintKind": "RELATION",
                            "binding": item,
                        }
                    ),
                    kind=PopulationConstraintKind.RELATION,
                    semantic_ref_ids=(item.semantic_ref_id,),
                )
            )
        if dependent:
            constraints.append(
                PopulationConstraintEvidence(
                    fingerprint=_stable_fingerprint(
                        {
                            "constraintKind": "ENTITY_MEMBERSHIP",
                            "populationFingerprint": population_fingerprint,
                        }
                    ),
                    kind=PopulationConstraintKind.ENTITY_MEMBERSHIP,
                )
            )
        constraints.append(
            PopulationConstraintEvidence(
                fingerprint=_stable_fingerprint(
                    {
                        "constraintKind": "GOVERNED_SCOPE",
                        "tableRefs": sorted(item.detail_ref_id for item in contract.tables),
                    }
                ),
                kind=PopulationConstraintKind.GOVERNED_SCOPE,
                semantic_ref_ids=tuple(sorted(item.detail_ref_id for item in contract.tables if item.detail_ref_id)),
            )
        )
        return tuple(constraints)

    def _lineage_observation(
        self,
        scope: PopulationScopeDescriptor,
        *,
        contract: GroundedQueryContract,
        sql: str,
        consumer_goal_ids: Sequence[str],
    ) -> tuple[
        PopulationLineageMechanism,
        tuple[PopulationConstraintEvidence, ...],
        bool,
    ]:
        kind = _enum_value(scope.kind)
        if kind in {
            PopulationScopeKind.INDEPENDENT.value,
            PopulationScopeKind.UNIVERSE.value,
        }:
            return (
                PopulationLineageMechanism.DIRECT_SCOPE,
                scope.constraints,
                True,
            )
        if kind == PopulationScopeKind.PREDICATE_SCOPE.value and (
            contract.reference_scope.executable and contract.reference_scope.referent_type == "PREDICATE_SCOPE"
        ):
            return (
                PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
                scope.constraints,
                True,
            )
        if kind not in {
            PopulationScopeKind.SAME_AS_GOAL.value,
            PopulationScopeKind.PREDICATE_SCOPE.value,
        }:
            return (
                PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
                self._time_only_constraints(scope.constraints),
                False,
            )
        if not set(scope.source_goal_ids).issubset(set(consumer_goal_ids)):
            return (
                PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
                self._time_only_constraints(scope.constraints),
                False,
            )
        mechanism = self._ast_lineage_mechanism(sql)
        if mechanism is None:
            return (
                PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
                self._time_only_constraints(scope.constraints),
                False,
            )
        return mechanism, scope.constraints, True

    @staticmethod
    def _time_only_constraints(
        constraints: Sequence[PopulationConstraintEvidence],
    ) -> tuple[PopulationConstraintEvidence, ...]:
        return tuple(item for item in constraints if _enum_value(item.kind) == PopulationConstraintKind.TIME.value)

    @staticmethod
    def _ast_lineage_mechanism(
        sql: str,
    ) -> Optional[PopulationLineageMechanism]:
        if not sql:
            return None
        try:
            root = parse_one(sql, read="mysql")
        except Exception:
            return None
        cte_aliases = {
            _text(item.alias_or_name).lower() for item in root.find_all(exp.CTE) if _text(item.alias_or_name)
        }
        if cte_aliases:
            referenced = {_text(item.name).lower() for item in root.find_all(exp.Table) if _text(item.name)}
            if cte_aliases.intersection(referenced):
                return PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE
        if any(True for _item in root.find_all(exp.Exists)):
            return PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE
        for item in root.find_all(exp.In):
            if item.args.get("query") is not None:
                return PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE
        for item in root.find_all(exp.Join):
            join_kind = _text(item.args.get("kind")).upper()
            if join_kind in {"SEMI", "ANTI"}:
                return PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE
        return None

    def _run_authority(
        self,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
    ) -> _PopulationRunAuthority:
        owner = _require_text(
            context_owner_fingerprint,
            "context_owner_fingerprint",
        )
        run_authority = _require_text(
            run_authority_fingerprint,
            "run_authority_fingerprint",
        )
        with self._lock:
            authority = self._runs.get(run_authority)
        if (
            authority is None
            or authority.workspace.owner_fingerprint != owner
            or authority.workspace.request_fingerprint != run_authority
        ):
            raise GroundedPopulationRuntimeGateError("POPULATION_RUN_AUTHORITY_NOT_REGISTERED")
        return authority

    def _facade(
        self,
        authority: _PopulationRunAuthority,
    ) -> PopulationOnlineGateFacade:
        state_store = GroundedWorkspacePopulationGateStateStore(authority.workspace)
        ledger_reader = PublishedGroundedPopulationLedgerReader(
            settings=self.settings,
            workspace=authority.workspace,
            state_store=state_store,
            ledger_provider=authority.ledger_provider,
            authority_fingerprint=self.ledger_authority_fingerprint,
        )
        reviewer = IndependentPopulationSemanticReviewer(
            self.semantic_provider,
            trusted_provider_authority_fingerprints=(self.semantic_authority_fingerprint,),
            timeout_seconds=self.semantic_timeout_seconds,
        )
        coordinator = PopulationGateCoordinator(
            state_store=state_store,
            ledger_reader=ledger_reader,
            trusted_semantic_verifier_fingerprints=(self.semantic_authority_fingerprint,),
            trusted_lineage_verifier_fingerprints=(self.lineage_authority_fingerprint,),
            trusted_artifact_verifier_fingerprints=(
                self.artifact_authority_fingerprint,
                self.ledger_authority_fingerprint,
            ),
            trusted_ledger_authority_fingerprints=(self.ledger_authority_fingerprint,),
        )
        return PopulationOnlineGateFacade(
            semantic_reviewer=reviewer,
            coordinator=coordinator,
            declaration_author_fingerprint=(self.declaration_author_fingerprint),
        )

    @staticmethod
    def _failure(
        stage: PopulationVerificationStage,
        code: str,
    ) -> PopulationOnlineGateCallResult:
        return PopulationOnlineGateCallResult(
            stage=stage,
            accepted=False,
            code=code,
            message=code,
        )
