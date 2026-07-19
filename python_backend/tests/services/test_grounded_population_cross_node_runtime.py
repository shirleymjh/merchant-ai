from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from merchant_ai.models import DataSnapshotContract
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationArtifactLedgerEntry,
    PopulationDynamicGraphEdge,
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    PopulationExecutionNodeBinding,
    PopulationNodeGateRecord,
    PopulationPublishedArtifactReceipt,
    seal_population_artifact_ledger_entry,
    seal_population_dynamic_graph_receipt,
    seal_population_node_gate_record,
    seal_population_published_artifact_receipt,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    PopulationExecutorNodeEvidence,
    PopulationPreExecutionNodeReference,
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationLineageMechanism,
    PopulationScopeAttestation,
    PopulationScopeKind,
    PopulationSemanticVerifier,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    PreExecutionPopulationVerificationInput,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedEntityFilterBinding,
    GroundedQueryContract,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
    GroundedUpstreamEntityBinding,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


SOURCE_GOAL = "goal-source"
CONSUMER_GOAL = "goal-consumer"
SOURCE_NODE = "node-source"
CONSUMER_NODE = "node-consumer"
GOAL_FINGERPRINT = hashlib.sha256(b"goal").hexdigest()
GRAPH_FINGERPRINT = hashlib.sha256(b"graph").hexdigest()
SNAPSHOT_FINGERPRINT = hashlib.sha256(b"snapshot").hexdigest()
SOURCE_CONTRACT_FINGERPRINT = hashlib.sha256(b"source-contract").hexdigest()
SOURCE_SQL_FINGERPRINT = hashlib.sha256(b"source-sql").hexdigest()
SOURCE_POPULATION_FINGERPRINT = hashlib.sha256(b"source-population").hexdigest()
SOURCE_GRAIN_FINGERPRINT = hashlib.sha256(b"source-grain").hexdigest()
LINEAGE_AUTHORITY = hashlib.sha256(b"lineage-authority").hexdigest()
ARTIFACT_AUTHORITY = hashlib.sha256(b"artifact-authority").hexdigest()
SOURCE_QUERY_ARTIFACT_ID = "query-artifact-source"
SOURCE_RESULT_ARTIFACT_ID = hashlib.sha256(b"published-result").hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


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


def _goal_attestation() -> PopulationVerificationAttestation:
    source = PopulationScopeAttestation(
        consumer_goal_id=SOURCE_GOAL,
        scope_kind=PopulationScopeKind.INDEPENDENT,
        declaration_scope_fingerprint="declaration-source",
    )
    consumer = PopulationScopeAttestation(
        consumer_goal_id=CONSUMER_GOAL,
        scope_kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(SOURCE_GOAL,),
        declaration_scope_fingerprint="declaration-consumer",
        complete_membership_required=True,
    )
    return _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            passed=True,
            gate_open=True,
            input_fingerprint="goal-input",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            accepted_scopes=(source, consumer),
        )
    )


def _source_record(
    goal: PopulationVerificationAttestation,
) -> PopulationNodeGateRecord:
    source_scope = PopulationScopeAttestation(
        consumer_goal_id=SOURCE_GOAL,
        scope_kind=PopulationScopeKind.INDEPENDENT,
        declaration_scope_fingerprint="declaration-source",
        population_fingerprint=SOURCE_POPULATION_FINGERPRINT,
        entity_identity_ref="entity-order",
        grain_fingerprint=SOURCE_GRAIN_FINGERPRINT,
        query_node_id=SOURCE_NODE,
        generation=1,
        attempt_id="attempt-source",
        query_contract_fingerprint=SOURCE_CONTRACT_FINGERPRINT,
        sql_ast_fingerprint=SOURCE_SQL_FINGERPRINT,
        snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        proof_fingerprints=(hashlib.sha256(b"source-proof").hexdigest(),),
    )
    pre = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.PRE_EXECUTION,
            passed=True,
            gate_open=True,
            input_fingerprint="pre-source",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            accepted_scopes=(source_scope,),
            previous_attestation_fingerprint=goal.attestation_fingerprint,
        )
    )
    post = _seal_attestation(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.POST_RESULT,
            passed=True,
            gate_open=True,
            input_fingerprint="post-source",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            accepted_scopes=(source_scope,),
            previous_attestation_fingerprint=pre.attestation_fingerprint,
        )
    )
    graph_receipt = _graph_receipt()
    return seal_population_node_gate_record(
        PopulationNodeGateRecord(
            query_node_id=SOURCE_NODE,
            graph_receipt_fingerprint=(
                graph_receipt.receipt_fingerprint
            ),
            node_binding=PopulationExecutionNodeBinding(
                query_node_id=SOURCE_NODE,
                consumer_goal_ids=(SOURCE_GOAL,),
                generation=1,
                attempt_id="attempt-source",
                query_contract_fingerprint=SOURCE_CONTRACT_FINGERPRINT,
                sql_ast_fingerprint=SOURCE_SQL_FINGERPRINT,
                snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
            ),
            required_consumer_goal_ids=(SOURCE_GOAL,),
            pre_execution_attestation=pre,
            post_result_attestation=post,
        )
    )


def _graph_receipt() -> PopulationDynamicGraphReceipt:
    return seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id="graph-two-node",
            graph_version=1,
            graph_fingerprint=GRAPH_FINGERPRINT,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id=SOURCE_NODE,
                    consumer_goal_ids=(SOURCE_GOAL,),
                ),
                PopulationDynamicGraphNode(
                    query_node_id=CONSUMER_NODE,
                    consumer_goal_ids=(CONSUMER_GOAL,),
                ),
            ),
            edges=(
                PopulationDynamicGraphEdge(
                    source_query_node_id=SOURCE_NODE,
                    target_query_node_id=CONSUMER_NODE,
                    dependency_mode="VERIFIED_ARTIFACT",
                    artifact_kind="VERIFIED_RESULT_ARTIFACT",
                ),
            ),
        )
    )


def _consumer_contract() -> GroundedQueryContract:
    values = ["order-1", "order-2"]
    target_ref = "semantic:order:id"
    return GroundedQueryContract(
        status="READY",
        question="rank the selected entities",
        query_shape="RANKED",
        execution_shape="ranked_list",
        primary_table="relation-current",
        tables=[
            GroundedTableBinding(
                topic="topic-current",
                table="relation-current",
                data_grain="one row per order",
                merchant_filter_column="principal-key",
                detail_ref_id="semantic:relation:current",
            )
        ],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id=target_ref,
                topic="topic-current",
                table="relation-current",
                column="order_id",
                output_alias="order_id",
                is_unique_key=True,
                entity_identity="entity-order",
            )
        ],
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id=target_ref,
                topic="topic-current",
                table="relation-current",
                column="order_id",
                operator="IN",
                literal_value=values,
                entity_identity="entity-order",
                allowed_operators=["IN"],
            )
        ],
        upstream_entity_bindings=[
            GroundedUpstreamEntityBinding(
                entity_set_artifact_id="entity-set-source",
                source_query_artifact_id=SOURCE_QUERY_ARTIFACT_ID,
                source_contract_fingerprint=SOURCE_CONTRACT_FINGERPRINT,
                source_sql_fingerprint=SOURCE_SQL_FINGERPRINT,
                source_column="order_id",
                source_semantic_ref_id="semantic:source:order-id",
                source_entity_identity="entity-order",
                target_field_ref=target_ref,
                target_table="relation-current",
                target_column="order_id",
                target_entity_identity="entity-order",
                operator="IN",
                value_count=len(values),
                values_hash=_fingerprint(values),
            )
        ],
    )


def _source_entry(
    *,
    source_query_artifact_id: str = SOURCE_QUERY_ARTIFACT_ID,
) -> PopulationArtifactLedgerEntry:
    evidence = PopulationArtifactEvidence(
        artifact_id=SOURCE_RESULT_ARTIFACT_ID,
        artifact_fingerprint=SOURCE_RESULT_ARTIFACT_ID,
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=PopulationArtifactCoverage.ALL_ROWS,
        population_fingerprint=SOURCE_POPULATION_FINGERPRINT,
        verifier_fingerprint=ARTIFACT_AUTHORITY,
        verified=True,
        immutable=True,
        goal_contract_fingerprint=GOAL_FINGERPRINT,
        graph_fingerprint=GRAPH_FINGERPRINT,
        query_contract_fingerprint=SOURCE_CONTRACT_FINGERPRINT,
        sql_ast_fingerprint=SOURCE_SQL_FINGERPRINT,
        snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
    )
    receipt = seal_population_published_artifact_receipt(
        PopulationPublishedArtifactReceipt(
            ledger_artifact_id="ledger-source",
            source_query_artifact_id=source_query_artifact_id,
            publication_status="PUBLISHED",
            generation=1,
            attempt_id="attempt-source",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            query_node_id=SOURCE_NODE,
            covered_consumer_goal_ids=(SOURCE_GOAL,),
            result_is_truncated=False,
            stored_row_count=2,
            exact_result_row_count=2,
            evidence=evidence,
        )
    )
    return seal_population_artifact_ledger_entry(
        PopulationArtifactLedgerEntry(
            ledger_artifact_id="ledger-source",
            publication_status="PUBLISHED",
            receipt=receipt,
        )
    )


def _claims(
    *,
    source_query_artifact_id: str = SOURCE_QUERY_ARTIFACT_ID,
) -> tuple[object, ...]:
    goal = _goal_attestation()
    contract = _consumer_contract()
    graph_receipt = _graph_receipt()
    reference = seal_population_pre_execution_reference(
        PopulationPreExecutionReference(
            gate_id="gate-two-node",
            context_owner_fingerprint="owner-two-node",
            run_authority_fingerprint="run-two-node",
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_receipt=graph_receipt,
            node=PopulationPreExecutionNodeReference(
                query_node_id=CONSUMER_NODE,
                consumer_goal_ids=(CONSUMER_GOAL,),
                generation=2,
                attempt_id="attempt-consumer",
                query_contract_fingerprint=(
                    grounded_query_contract_fingerprint(contract)
                ),
            ),
        )
    )
    gate = object.__new__(GroundedPopulationExecutionGate)
    gate.lineage_authority_fingerprint = LINEAGE_AUTHORITY
    return gate._execution_claims(
        state=SimpleNamespace(
            goal_attestation=goal,
            node_gate_records=(_source_record(goal),),
        ),
        reference=reference,
        execution=PopulationExecutorNodeEvidence(
            query_node_id=CONSUMER_NODE,
            contract=contract,
            compilation=SimpleNamespace(
                sql="SELECT order_id FROM relation-current"
            ),
            data_snapshot=DataSnapshotContract(),
            actual_sql_ast_fingerprint="consumer-ast",
        ),
        snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        sql_ast_fingerprint="consumer-ast",
        required_consumer_goal_ids=(CONSUMER_GOAL,),
        source_entries=(
            _source_entry(
                source_query_artifact_id=source_query_artifact_id
            ),
        ),
    )


def test_cross_node_claim_uses_published_parent_without_reassigning_source_goal() -> None:
    claims = _claims()

    assert len(claims) == 1
    claim = claims[0]
    proof = claim.lineage_proofs[0]
    assert claim.consumer_goal_id == CONSUMER_GOAL
    assert claim.query_node_id == CONSUMER_NODE
    assert proof.mechanism == (
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value
    )
    assert proof.source_goal_ids == (SOURCE_GOAL,)
    assert proof.source_node_ids == (SOURCE_NODE,)
    assert proof.artifact_evidence[0].artifact_id == (
        SOURCE_RESULT_ARTIFACT_ID
    )

    result = PopulationSemanticVerifier().verify_pre_execution(
        PreExecutionPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            declaration_attestation=_goal_attestation(),
            trusted_lineage_verifier_fingerprints=(LINEAGE_AUTHORITY,),
            trusted_artifact_verifier_fingerprints=(ARTIFACT_AUTHORITY,),
            required_consumer_goal_ids=(CONSUMER_GOAL,),
            consumer_scope_selection_explicit=True,
            claims=claims,
        )
    )

    assert result.passed is True
    assert result.gate_open is True


def test_cross_node_claim_rejects_a_different_source_query_artifact() -> None:
    claims = _claims(source_query_artifact_id="query-artifact-other")

    assert len(claims) == 1
    proof = claims[0].lineage_proofs[0]
    assert proof.mechanism != (
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value
    )
    assert proof.artifact_evidence == ()
