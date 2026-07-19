from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merchant_ai.config import Settings
from merchant_ai.models import DataSnapshotContract
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationConstraintEvidence,
    PopulationConstraintKind,
    PopulationExecutionClaim,
    PopulationGapCode,
    PopulationLineageMechanism,
    PopulationLineageProof,
    PopulationScopeAttestation,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationSemanticVerifier,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    PreExecutionPopulationVerificationInput,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    PopulationPreExecutionNodeReference,
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.grounded_query_executor import (
    GroundedQueryExecutionKernel,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


GOAL_FINGERPRINT = hashlib.sha256(b"goal-contract").hexdigest()
GRAPH_FINGERPRINT = hashlib.sha256(b"dynamic-graph").hexdigest()
QUERY_FINGERPRINT = hashlib.sha256(b"query-contract").hexdigest()
SQL_FINGERPRINT = hashlib.sha256(b"validated-ast").hexdigest()
SNAPSHOT_FINGERPRINT = hashlib.sha256(b"data-snapshot").hexdigest()
POPULATION_FINGERPRINT = hashlib.sha256(b"population").hexdigest()
GRAIN_FINGERPRINT = hashlib.sha256(b"grain").hexdigest()
DECLARATION_FINGERPRINT = hashlib.sha256(b"declaration").hexdigest()
LINEAGE_AUTHORITY = hashlib.sha256(b"lineage-authority").hexdigest()
ARTIFACT_AUTHORITY = hashlib.sha256(b"artifact-authority").hexdigest()
SOURCE_GOAL = "goal.source"
CONSUMER_GOAL = "goal.consumer"
SOURCE_NODE = "node.source"
CONSUMER_NODE = "node.consumer"
SOURCE_ARTIFACT = "artifact.source"


def _seal(
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
    return _seal(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            passed=True,
            gate_open=True,
            input_fingerprint=hashlib.sha256(b"goal-input").hexdigest(),
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            accepted_scopes=(
                PopulationScopeAttestation(
                    consumer_goal_id=CONSUMER_GOAL,
                    scope_kind=PopulationScopeKind.SAME_AS_GOAL,
                    source_goal_ids=(SOURCE_GOAL,),
                    declaration_scope_fingerprint=(
                        DECLARATION_FINGERPRINT
                    ),
                    complete_membership_required=True,
                ),
            ),
        )
    )


def _constraints() -> tuple[PopulationConstraintEvidence, ...]:
    return (
        PopulationConstraintEvidence(
            fingerprint=hashlib.sha256(b"time-constraint").hexdigest(),
            kind=PopulationConstraintKind.TIME,
            semantic_ref_ids=("semantic.time",),
        ),
        PopulationConstraintEvidence(
            fingerprint=hashlib.sha256(
                b"membership-constraint"
            ).hexdigest(),
            kind=PopulationConstraintKind.ENTITY_MEMBERSHIP,
        ),
        PopulationConstraintEvidence(
            fingerprint=hashlib.sha256(
                b"governed-scope-constraint"
            ).hexdigest(),
            kind=PopulationConstraintKind.GOVERNED_SCOPE,
            semantic_ref_ids=("semantic.relation",),
        ),
    )


def _scope() -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="scope.consumer",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(SOURCE_GOAL,),
        population_fingerprint=POPULATION_FINGERPRINT,
        entity_identity_ref="semantic.entity",
        grain_fingerprint=GRAIN_FINGERPRINT,
        snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        constraints=_constraints(),
        complete_membership_required=True,
    )


def _artifact() -> PopulationArtifactEvidence:
    return PopulationArtifactEvidence(
        artifact_id=SOURCE_ARTIFACT,
        artifact_fingerprint=hashlib.sha256(b"source-artifact").hexdigest(),
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=PopulationArtifactCoverage.ALL_ROWS,
        population_fingerprint=POPULATION_FINGERPRINT,
        verifier_fingerprint=ARTIFACT_AUTHORITY,
        verified=True,
        immutable=True,
    )


def _proof(
    mechanism: PopulationLineageMechanism,
    *,
    preserved_constraints: tuple[PopulationConstraintEvidence, ...] | None = None,
    complete_membership: bool = True,
    query_node_id: str = CONSUMER_NODE,
    generation: int = 5,
    attempt_id: str = "attempt-5",
) -> PopulationLineageProof:
    return PopulationLineageProof(
        proof_id="proof.consumer",
        mechanism=mechanism,
        verifier_fingerprint=LINEAGE_AUTHORITY,
        verified=True,
        graph_fingerprint=GRAPH_FINGERPRINT,
        query_node_id=query_node_id,
        generation=generation,
        attempt_id=attempt_id,
        query_contract_fingerprint=QUERY_FINGERPRINT,
        sql_ast_fingerprint=SQL_FINGERPRINT,
        source_population_fingerprint=POPULATION_FINGERPRINT,
        result_population_fingerprint=POPULATION_FINGERPRINT,
        source_goal_ids=(SOURCE_GOAL,),
        source_node_ids=(SOURCE_NODE,),
        preserved_constraints=(
            _constraints()
            if preserved_constraints is None
            else preserved_constraints
        ),
        artifact_evidence=(
            (_artifact(),)
            if mechanism
            == PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT
            else ()
        ),
        source_entity_identity_ref="semantic.entity",
        result_entity_identity_ref="semantic.entity",
        source_grain_fingerprint=GRAIN_FINGERPRINT,
        result_grain_fingerprint=GRAIN_FINGERPRINT,
        source_snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        result_snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        complete_membership=complete_membership,
    )


def _claim(proof: PopulationLineageProof) -> PopulationExecutionClaim:
    scope = _scope()
    if proof.mechanism == PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT:
        scope = scope.model_copy(
            update={"source_artifact_ids": (SOURCE_ARTIFACT,)}
        )
    return PopulationExecutionClaim(
        consumer_goal_id=CONSUMER_GOAL,
        query_node_id=CONSUMER_NODE,
        generation=5,
        attempt_id="attempt-5",
        declaration_scope_fingerprint=DECLARATION_FINGERPRINT,
        required_scope=scope,
        effective_scope=scope,
        query_contract_fingerprint=QUERY_FINGERPRINT,
        sql_ast_fingerprint=SQL_FINGERPRINT,
        lineage_proofs=(proof,),
    )


def _verify(proof: PopulationLineageProof) -> object:
    return PopulationSemanticVerifier().verify_pre_execution(
        PreExecutionPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            declaration_attestation=_goal_attestation(),
            trusted_lineage_verifier_fingerprints=(LINEAGE_AUTHORITY,),
            trusted_artifact_verifier_fingerprints=(ARTIFACT_AUTHORITY,),
            claims=(_claim(proof),),
        )
    )


def _gap_codes(result: object) -> set[str]:
    return {
        str(getattr(item.code, "value", item.code))
        for item in getattr(result, "gaps", ())
    }


def test_same_node_cte_can_preserve_the_declared_population() -> None:
    sql = (
        "WITH population_scope AS ("
        "SELECT entity_key, measure_value FROM fixture_relation"
        ") SELECT entity_key, measure_value FROM population_scope "
        "ORDER BY measure_value DESC LIMIT 3"
    )
    assert GroundedPopulationExecutionGate._ast_lineage_mechanism(sql) == (
        PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE
    )
    result = _verify(
        _proof(PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE)
    )

    assert result.passed is True
    assert result.gate_open is True
    assert result.attestation.accepted_scopes[0].query_node_id == (
        CONSUMER_NODE
    )


def test_dynamic_cross_node_result_artifact_can_preserve_population() -> None:
    result = _verify(
        _proof(PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT)
    )

    assert result.passed is True
    assert result.gate_open is True
    accepted = result.attestation.accepted_scopes[0]
    assert accepted.query_node_id == CONSUMER_NODE
    assert accepted.generation == 5
    assert accepted.attempt_id == "attempt-5"
    assert accepted.source_artifact_ids == (SOURCE_ARTIFACT,)


def test_shared_time_constraint_cannot_replace_population_lineage() -> None:
    time_only = tuple(
        item
        for item in _constraints()
        if item.kind == PopulationConstraintKind.TIME
    )
    result = _verify(
        _proof(
            PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
            preserved_constraints=time_only,
            complete_membership=False,
        )
    )

    assert result.passed is False
    assert result.gate_open is False
    assert PopulationGapCode.POPULATION_DEGRADED_TO_TIME_FILTER.value in (
        _gap_codes(result)
    )
    assert PopulationGapCode.MEMBERSHIP_INCOMPLETE.value in _gap_codes(
        result
    )


@pytest.mark.parametrize(
    "replay_updates",
    (
        {"query_node_id": SOURCE_NODE},
        {"generation": 6},
        {"attempt_id": "attempt-replayed"},
    ),
)
def test_cross_node_generation_or_attempt_replay_is_rejected(
    replay_updates: dict[str, object],
) -> None:
    original = _proof(
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT
    )
    replayed = original.model_copy(update=replay_updates)
    result = _verify(replayed)

    assert result.passed is False
    assert result.gate_open is False
    expected_codes = {
        "query_node_id": PopulationGapCode.LINEAGE_QUERY_NODE_MISMATCH.value,
        "generation": PopulationGapCode.LINEAGE_GENERATION_MISMATCH.value,
        "attempt_id": PopulationGapCode.LINEAGE_ATTEMPT_MISMATCH.value,
    }
    changed_field = next(iter(replay_updates))
    assert expected_codes[changed_field] in _gap_codes(result)


class _NoDorisRepository:
    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.query_calls = 0
        self.stream_calls = 0
        self.last_cache_hit = False
        self.last_cache_key = ""

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        self.snapshot_calls += 1
        return DataSnapshotContract(
            datasource_fingerprint=hashlib.sha256(
                b"fixture-datasource"
            ).hexdigest(),
            datasource_environment="test",
            consistency_mode="UNSUPPORTED",
            semantic_activation_fingerprint=(
                semantic_activation_fingerprint
            ),
            cache_generation="fixture-generation",
            unsupported_reason="TEST_SNAPSHOT_NON_ATOMIC",
        )

    def query(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.query_calls += 1
        raise AssertionError("Doris query must remain behind the PRE gate")

    def stream_query_batches(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        self.stream_calls += 1
        raise AssertionError("Doris stream must remain behind the PRE gate")


class _RejectingPreGate(GroundedPopulationExecutionGate):
    def __init__(self) -> None:
        self.calls = 0

    def authorize_node(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        return SimpleNamespace(
            accepted=False,
            code="POPULATION_PRE_REJECTED_BY_TEST_AUTHORITY",
        )


def _executor_contract() -> GroundedQueryContract:
    topic = "fixture_topic"
    relation = "fixture_relation"
    detail_ref = "semantic:fixture:detail"
    field_ref = "semantic:fixture:field:entity_key"
    return GroundedQueryContract(
        status="READY",
        question="return selected detail",
        topics=[topic],
        query_shape="DETAIL",
        execution_shape="detail_list",
        primary_table=relation,
        tables=[
            GroundedTableBinding(
                topic=topic,
                table=relation,
                data_grain="fixture_entity",
                merchant_filter_column="principal_key",
                detail_ref_id=detail_ref,
            )
        ],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id=field_ref,
                topic=topic,
                table=relation,
                column="entity_key",
                output_alias="entity_key",
                is_unique_key=True,
                entity_identity="fixture_entity",
            )
        ],
        evidence_refs=[detail_ref, field_ref],
    )


def _access_control(
    settings: Settings,
    root: Path,
    contract: GroundedQueryContract,
) -> AccessControlService:
    root.mkdir(parents=True, exist_ok=True)
    (root / "merchant_acl.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "defaultEffect": "DENY",
                "allowedMerchantIds": ["fixture-principal"],
                "tables": {
                    item.table: {"allowedRoles": ["fixture-role"]}
                    for item in contract.tables
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return AccessControlService(settings, root=root)


def test_pre_gate_rejection_has_zero_doris_side_effects(
    tmp_path: Path,
) -> None:
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace"),
        grounded_result_stream_max_rows=100,
        grounded_result_stream_max_bytes=1024 * 1024,
    )
    contract = _executor_contract()
    pack = materialize_grounded_asset_pack(
        contract,
        TopicAssetService(settings),
    )
    preparation = compile_grounded_query(contract, pack)
    repository = _NoDorisRepository()
    gate = _RejectingPreGate()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=_access_control(
            settings,
            tmp_path / "acl",
            contract,
        ),
        population_execution_gate=gate,
    )
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="fixture-thread",
        run_id="fixture-run",
        merchant_id="fixture-principal",
        access_role="fixture-role",
        user_scope={},
        question=contract.question,
    )
    node_id = preparation.plan.intents[0].plan_task_id
    reference = seal_population_pre_execution_reference(
        PopulationPreExecutionReference(
            gate_id="fixture-gate",
            context_owner_fingerprint=workspace.owner_fingerprint,
            run_authority_fingerprint=workspace.request_fingerprint,
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_receipt=seal_population_dynamic_graph_receipt(
                PopulationDynamicGraphReceipt(
                    graph_id="fixture-graph",
                    graph_version=1,
                    graph_fingerprint=GRAPH_FINGERPRINT,
                    nodes=(
                        PopulationDynamicGraphNode(
                            query_node_id=node_id,
                            consumer_goal_ids=(CONSUMER_GOAL,),
                        ),
                    ),
                )
            ),
            node=PopulationPreExecutionNodeReference(
                query_node_id=node_id,
                consumer_goal_ids=(CONSUMER_GOAL,),
                generation=1,
                attempt_id="fixture-attempt",
                query_contract_fingerprint=(
                    grounded_query_contract_fingerprint(contract)
                ),
            ),
        )
    )

    result = executor.execute_contract(
        "fixture-principal",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="fixture-run",
        artifact_root=str(workspace.artifacts_root),
        context_owner_fingerprint=workspace.owner_fingerprint,
        access_role="fixture-role",
        user_scope={},
        execution_reference_scope={},
        execution_goal_contract_fingerprint=GOAL_FINGERPRINT,
        expected_semantic_activation_fingerprint=(
            preparation.asset_pack_fingerprint
        ),
        population_pre_execution_reference=reference,
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="fixture-attempt",
    )

    assert result.merged_query_bundle.failed is True
    assert "POPULATION_PRE_EXECUTION_REJECTED" in (
        result.merged_query_bundle.error
    )
    assert gate.calls == 1
    assert repository.snapshot_calls == 1
    assert repository.query_calls == 0
    assert repository.stream_calls == 0
    assert not list(workspace.staging_root.rglob("rows.json"))
