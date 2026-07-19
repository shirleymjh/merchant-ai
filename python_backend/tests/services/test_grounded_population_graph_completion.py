from __future__ import annotations

from types import SimpleNamespace

from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    PopulationExecutionNodeBinding,
    PopulationNodeGateRecord,
    seal_population_dynamic_graph_receipt,
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


def _seal_attestation(
    attestation: PopulationVerificationAttestation,
) -> PopulationVerificationAttestation:
    return attestation.model_copy(
        update={
            "attestation_fingerprint": population_attestation_fingerprint(
                attestation
            )
        }
    )


def _fixture() -> tuple[
    PopulationPreExecutionReference,
    SimpleNamespace,
    PopulationNodeGateRecord,
]:
    goal_fingerprint = "goal-fingerprint"
    graph_fingerprint = "graph-fingerprint"
    node_id = "node-current"
    goal_id = "goal-current"
    receipt = seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id="graph-current",
            graph_version=1,
            graph_fingerprint=graph_fingerprint,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id=node_id,
                    consumer_goal_ids=(goal_id,),
                ),
            ),
        )
    )
    scope = PopulationScopeAttestation(
        consumer_goal_id=goal_id,
        scope_kind=PopulationScopeKind.INDEPENDENT,
        declaration_scope_fingerprint="declaration-fingerprint",
        query_node_id=node_id,
        generation=1,
        attempt_id="attempt-current",
        query_contract_fingerprint="contract-fingerprint",
        sql_ast_fingerprint="ast-fingerprint",
        snapshot_fingerprint="snapshot-fingerprint",
    )
    goal = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            passed=True,
            gate_open=True,
            input_fingerprint="goal-input",
            goal_contract_fingerprint=goal_fingerprint,
            accepted_scopes=(scope.model_copy(update={"query_node_id": ""}),),
        )
    )
    pre = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.PRE_EXECUTION,
            passed=True,
            gate_open=True,
            input_fingerprint="pre-input",
            goal_contract_fingerprint=goal_fingerprint,
            graph_fingerprint=graph_fingerprint,
            accepted_scopes=(scope,),
            previous_attestation_fingerprint=goal.attestation_fingerprint,
        )
    )
    post = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.POST_RESULT,
            passed=True,
            gate_open=True,
            input_fingerprint="post-input",
            goal_contract_fingerprint=goal_fingerprint,
            graph_fingerprint=graph_fingerprint,
            accepted_scopes=(scope,),
            previous_attestation_fingerprint=pre.attestation_fingerprint,
        )
    )
    binding = PopulationExecutionNodeBinding(
        query_node_id=node_id,
        consumer_goal_ids=(goal_id,),
        generation=1,
        attempt_id="attempt-current",
        query_contract_fingerprint="contract-fingerprint",
        sql_ast_fingerprint="ast-fingerprint",
        snapshot_fingerprint="snapshot-fingerprint",
    )
    record = seal_population_node_gate_record(
        PopulationNodeGateRecord(
            query_node_id=node_id,
            graph_receipt_fingerprint=receipt.receipt_fingerprint,
            node_binding=binding,
            required_consumer_goal_ids=(goal_id,),
            pre_execution_attestation=pre,
            post_result_attestation=post,
        )
    )
    state = SimpleNamespace(
        graph_receipt=receipt,
        graph_fingerprint=graph_fingerprint,
        node_gate_records=(record,),
        goal_attestation=goal,
    )
    reference = seal_population_pre_execution_reference(
        PopulationPreExecutionReference(
            gate_id="gate-current",
            context_owner_fingerprint="owner-current",
            run_authority_fingerprint="run-current",
            goal_contract_fingerprint=goal_fingerprint,
            graph_receipt=receipt,
            node=PopulationPreExecutionNodeReference(
                query_node_id=node_id,
                consumer_goal_ids=(goal_id,),
                generation=1,
                attempt_id="attempt-current",
                query_contract_fingerprint="contract-fingerprint",
            ),
        )
    )
    return reference, state, record


class _CompletionGate(GroundedPopulationExecutionGate):
    def __init__(self, state: SimpleNamespace) -> None:
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
        return SimpleNamespace(
            coordinator=SimpleNamespace(
                get_state=lambda _gate_id: self.state
            )
        )


def test_graph_completion_requires_one_valid_post_per_receipt_node() -> None:
    reference, state, _record = _fixture()

    result = _CompletionGate(state).require_graph_complete(
        reference=reference
    )

    assert result.accepted is True
    assert result.code == "GRAPH_COMPLETE"


def test_graph_completion_rejects_duplicate_node_records() -> None:
    reference, state, record = _fixture()
    state.node_gate_records = (record, record)

    result = _CompletionGate(state).require_graph_complete(
        reference=reference
    )

    assert result.accepted is False
    assert result.code == "POPULATION_GRAPH_NODE_COVERAGE_INCOMPLETE"
