from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from merchant_ai.config import Settings
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_population_online_gate import (
    GroundedWorkspacePopulationGateStateStore,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    InMemoryPopulationGateStateStore,
    PopulationDynamicGraphEdge,
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    PopulationExecutionNodeBinding,
    PopulationGateCode,
    PopulationGateCoordinator,
    PopulationGatePhase,
    PopulationGateState,
    PopulationGraphRevisionCommand,
    PopulationNodeGateRecord,
    PopulationNodePreExecutionCommand,
    seal_population_dynamic_graph_receipt,
    seal_population_gate_state,
    seal_population_node_gate_record,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    PopulationPreExecutionNodeReference,
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationScopeAttestation,
    PopulationScopeKind,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)


GOAL_FINGERPRINT = "goal-fingerprint"
OLD_GRAPH_FINGERPRINT = "old-graph-fingerprint"
NEW_GRAPH_FINGERPRINT = "new-graph-fingerprint"
PUBLISHED_NODE = "node-published"
NEW_NODE = "node-recovery"
PUBLISHED_GOAL = "goal-published"
NEW_GOAL = "goal-recovery"
REVISION_EVIDENCE = "revision-evidence-fingerprint"


class _UnusedLedger:
    authority_fingerprint = "ledger-authority"

    def snapshot_population_artifacts(self, **kwargs):
        del kwargs
        raise AssertionError("graph revision must not read the result ledger")


def _seal_attestation(
    attestation: PopulationVerificationAttestation,
) -> PopulationVerificationAttestation:
    return attestation.model_copy(update={"attestation_fingerprint": population_attestation_fingerprint(attestation)})


def _old_receipt() -> PopulationDynamicGraphReceipt:
    return seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id="population-graph-v1",
            graph_version=1,
            graph_fingerprint=OLD_GRAPH_FINGERPRINT,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id=PUBLISHED_NODE,
                    consumer_goal_ids=(PUBLISHED_GOAL,),
                ),
            ),
        )
    )


def _revised_receipt(
    old: PopulationDynamicGraphReceipt,
    *,
    mutate_carried_goal: bool = False,
) -> PopulationDynamicGraphReceipt:
    return seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id="population-graph-v2",
            graph_version=2,
            graph_fingerprint=NEW_GRAPH_FINGERPRINT,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id=PUBLISHED_NODE,
                    consumer_goal_ids=(NEW_GOAL if mutate_carried_goal else PUBLISHED_GOAL,),
                ),
                PopulationDynamicGraphNode(
                    query_node_id=NEW_NODE,
                    consumer_goal_ids=(NEW_GOAL,),
                ),
            ),
            edges=(
                PopulationDynamicGraphEdge(
                    source_query_node_id=PUBLISHED_NODE,
                    target_query_node_id=NEW_NODE,
                    dependency_mode="CONTRACT_SCOPE",
                ),
            ),
            parent_receipt_fingerprint=old.receipt_fingerprint,
            revision_evidence_fingerprint=REVISION_EVIDENCE,
            carried_forward_query_node_ids=(PUBLISHED_NODE,),
        )
    )


def _state() -> tuple[PopulationGateState, PopulationNodeGateRecord]:
    old = _old_receipt()
    goal = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            passed=True,
            gate_open=True,
            input_fingerprint="goal-input-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
        )
    )
    pre = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.PRE_EXECUTION,
            passed=True,
            gate_open=True,
            input_fingerprint="pre-input-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=OLD_GRAPH_FINGERPRINT,
            previous_attestation_fingerprint=(goal.attestation_fingerprint),
        )
    )
    post = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.POST_RESULT,
            passed=True,
            gate_open=True,
            input_fingerprint="post-input-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=OLD_GRAPH_FINGERPRINT,
            previous_attestation_fingerprint=(pre.attestation_fingerprint),
        )
    )
    record = seal_population_node_gate_record(
        PopulationNodeGateRecord(
            query_node_id=PUBLISHED_NODE,
            graph_receipt_fingerprint=old.receipt_fingerprint,
            node_binding=PopulationExecutionNodeBinding(
                query_node_id=PUBLISHED_NODE,
                consumer_goal_ids=(PUBLISHED_GOAL,),
                generation=1,
                attempt_id="published-attempt",
                query_contract_fingerprint="published-contract",
                sql_ast_fingerprint="published-sql-ast",
                snapshot_fingerprint="published-snapshot",
            ),
            pre_execution_attestation=pre,
            post_result_attestation=post,
        )
    )
    return (
        seal_population_gate_state(
            PopulationGateState(
                gate_id="population-gate",
                revision=3,
                phase=PopulationGatePhase.POST_RESULT,
                goal_contract_fingerprint=GOAL_FINGERPRINT,
                graph_fingerprint=OLD_GRAPH_FINGERPRINT,
                graph_receipt=old,
                node_gate_records=(record,),
                goal_attestation=goal,
            )
        ),
        record,
    )


def _coordinator(
    state: PopulationGateState,
) -> PopulationGateCoordinator:
    store = InMemoryPopulationGateStateStore()
    assert store.create_population_gate(state) is True
    return PopulationGateCoordinator(
        state_store=store,
        ledger_reader=_UnusedLedger(),
        trusted_semantic_verifier_fingerprints=("semantic-authority",),
        trusted_lineage_verifier_fingerprints=("lineage-authority",),
        trusted_artifact_verifier_fingerprints=("artifact-authority",),
        trusted_ledger_authority_fingerprints=("ledger-authority",),
    )


def _command(
    state: PopulationGateState,
    revised: PopulationDynamicGraphReceipt,
) -> PopulationGraphRevisionCommand:
    assert state.graph_receipt is not None
    return PopulationGraphRevisionCommand(
        gate_id=state.gate_id,
        expected_revision=state.revision,
        goal_contract_fingerprint=state.goal_contract_fingerprint,
        previous_graph_receipt_fingerprint=(state.graph_receipt.receipt_fingerprint),
        revised_graph_receipt=revised,
        revision_evidence_fingerprint=REVISION_EVIDENCE,
        revision_ordinal=1,
        maximum_revision_count=2,
    )


def test_population_revision_cas_preserves_published_record() -> None:
    state, record = _state()
    coordinator = _coordinator(state)
    revised = _revised_receipt(state.graph_receipt)

    result = coordinator.revise_dynamic_graph(_command(state, revised))

    assert result.accepted is True
    assert result.state is not None
    assert result.state.revision == state.revision + 1
    assert result.state.graph_receipt == revised
    assert result.state.graph_receipt_history == (state.graph_receipt,)
    assert result.state.node_gate_records == (record,)
    assert result.state.retired_node_gate_records == ()
    assert result.state.graph_revision_evidence_fingerprints == (REVISION_EVIDENCE,)


def test_population_revision_rejects_carried_node_mutation() -> None:
    state, _record = _state()
    coordinator = _coordinator(state)
    revised = _revised_receipt(
        state.graph_receipt,
        mutate_carried_goal=True,
    )

    result = coordinator.revise_dynamic_graph(_command(state, revised))

    assert result.accepted is False
    assert result.code == PopulationGateCode.GRAPH_BINDING_INVALID
    assert any("carried node" in issue.message.lower() for issue in result.issues)


def test_population_revision_has_single_cas_winner() -> None:
    state, _record = _state()
    coordinator = _coordinator(state)
    revised = _revised_receipt(state.graph_receipt)
    command = _command(state, revised)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: coordinator.revise_dynamic_graph(command),
                range(2),
            )
        )

    assert sum(item.accepted for item in results) == 1
    assert {item.code for item in results if not item.accepted} == {PopulationGateCode.CAS_REVISION_MISMATCH}


class _CompletionGate(GroundedPopulationExecutionGate):
    def __init__(self, state: PopulationGateState) -> None:
        self.state = state

    def _run_authority(
        self,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
    ) -> object:
        del context_owner_fingerprint, run_authority_fingerprint
        return object()

    def _facade(self, authority: object) -> SimpleNamespace:
        del authority
        return SimpleNamespace(coordinator=SimpleNamespace(get_state=lambda _gate_id: self.state))


def test_old_population_receipt_is_invalid_after_revision() -> None:
    state, _record = _state()
    coordinator = _coordinator(state)
    revised = _revised_receipt(state.graph_receipt)
    committed = coordinator.revise_dynamic_graph(_command(state, revised))
    assert committed.state is not None
    old_reference = seal_population_pre_execution_reference(
        PopulationPreExecutionReference(
            gate_id=state.gate_id,
            context_owner_fingerprint="owner-fingerprint",
            run_authority_fingerprint="run-authority-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_receipt=state.graph_receipt,
            node=PopulationPreExecutionNodeReference(
                query_node_id=PUBLISHED_NODE,
                consumer_goal_ids=(PUBLISHED_GOAL,),
                generation=1,
                attempt_id="published-attempt",
                query_contract_fingerprint="published-contract",
            ),
        )
    )

    result = _CompletionGate(committed.state).require_graph_complete(reference=old_reference)

    assert result.accepted is False
    assert result.code == "POPULATION_GRAPH_BINDING_MISMATCH"


def test_revised_node_requires_its_own_population_pre_post_chain() -> None:
    state, _record = _state()
    coordinator = _coordinator(state)
    revised = _revised_receipt(state.graph_receipt)
    committed = coordinator.revise_dynamic_graph(_command(state, revised))
    assert committed.state is not None
    current_reference = seal_population_pre_execution_reference(
        PopulationPreExecutionReference(
            gate_id=state.gate_id,
            context_owner_fingerprint="owner-fingerprint",
            run_authority_fingerprint="run-authority-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_receipt=revised,
            node=PopulationPreExecutionNodeReference(
                query_node_id=PUBLISHED_NODE,
                consumer_goal_ids=(PUBLISHED_GOAL,),
                generation=1,
                attempt_id="published-attempt",
                query_contract_fingerprint="published-contract",
            ),
        )
    )

    result = _CompletionGate(committed.state).require_graph_complete(reference=current_reference)

    assert result.accepted is False
    assert result.code == "POPULATION_GRAPH_NODE_COVERAGE_INCOMPLETE"


def test_revision_reruns_population_lineage_validation() -> None:
    state, _record = _state()
    scoped_goal = _seal_attestation(
        state.goal_attestation.model_copy(
            update={
                "accepted_scopes": (
                    PopulationScopeAttestation(
                        consumer_goal_id=NEW_GOAL,
                        scope_kind=PopulationScopeKind.PREDICATE_SCOPE,
                        source_goal_ids=(PUBLISHED_GOAL,),
                        declaration_scope_fingerprint=("declaration-scope-fingerprint"),
                    ),
                )
            }
        )
    )
    scoped_state = seal_population_gate_state(state.model_copy(update={"goal_attestation": scoped_goal}))
    coordinator = _coordinator(scoped_state)
    revised_without_lineage = seal_population_dynamic_graph_receipt(
        _revised_receipt(scoped_state.graph_receipt).model_copy(update={"edges": ()})
    )

    result = coordinator.revise_dynamic_graph(_command(scoped_state, revised_without_lineage))

    assert result.accepted is False
    assert result.code == PopulationGateCode.GRAPH_BINDING_INVALID
    assert any("required population lineage" in issue.message for issue in result.issues)


def test_workspace_state_store_persists_revision_cas_chain(
    tmp_path: Path,
) -> None:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="revision-thread",
        run_id="revision-run",
        merchant_id="revision-principal",
        access_role="revision-role",
        user_scope={},
        question="revision fixture question",
    )
    store = GroundedWorkspacePopulationGateStateStore(workspace)
    goal = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            passed=True,
            gate_open=True,
            input_fingerprint="goal-input-fingerprint",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
        )
    )
    initial = seal_population_gate_state(
        PopulationGateState(
            gate_id="workspace-revision-gate",
            revision=1,
            phase=PopulationGatePhase.GOAL_DECLARATION,
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            goal_attestation=goal,
        )
    )
    assert store.create_population_gate(initial) is True
    coordinator = PopulationGateCoordinator(
        state_store=store,
        ledger_reader=_UnusedLedger(),
        trusted_semantic_verifier_fingerprints=("semantic-authority",),
        trusted_lineage_verifier_fingerprints=("lineage-authority",),
        trusted_artifact_verifier_fingerprints=("artifact-authority",),
        trusted_ledger_authority_fingerprints=("ledger-authority",),
    )
    old = _old_receipt()
    pre = coordinator.authorize_node_pre_execution(
        PopulationNodePreExecutionCommand(
            gate_id=initial.gate_id,
            expected_revision=1,
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_receipt=old,
            node_binding=PopulationExecutionNodeBinding(
                query_node_id=PUBLISHED_NODE,
                consumer_goal_ids=(PUBLISHED_GOAL,),
                generation=1,
                attempt_id="failed-attempt",
                query_contract_fingerprint="failed-contract",
                sql_ast_fingerprint="failed-sql-ast",
                snapshot_fingerprint="failed-snapshot",
            ),
            required_consumer_goal_ids=(),
            claims=(),
        )
    )
    assert pre.accepted is True
    assert pre.state is not None
    revised = seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id="population-graph-v2",
            graph_version=2,
            graph_fingerprint=NEW_GRAPH_FINGERPRINT,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id=NEW_NODE,
                    consumer_goal_ids=(NEW_GOAL,),
                ),
            ),
            parent_receipt_fingerprint=old.receipt_fingerprint,
            revision_evidence_fingerprint=REVISION_EVIDENCE,
            retired_query_node_ids=(PUBLISHED_NODE,),
        )
    )
    revision = coordinator.revise_dynamic_graph(
        PopulationGraphRevisionCommand(
            gate_id=initial.gate_id,
            expected_revision=pre.state.revision,
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            previous_graph_receipt_fingerprint=(old.receipt_fingerprint),
            revised_graph_receipt=revised,
            revision_evidence_fingerprint=REVISION_EVIDENCE,
            revision_ordinal=1,
            maximum_revision_count=2,
        )
    )

    assert revision.accepted is True
    restarted = GroundedWorkspacePopulationGateStateStore(workspace)
    recovered = restarted.load_population_gate(initial.gate_id)
    assert recovered is not None
    assert recovered.graph_receipt == revised
    assert recovered.node_gate_records == ()
    assert len(recovered.retired_node_gate_records) == 1
    assert recovered.retired_node_gate_records[0].query_node_id == (PUBLISHED_NODE)
