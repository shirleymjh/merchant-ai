from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationGapCode,
    PopulationResultEvidence,
    PopulationScopeAttestation,
    PopulationSemanticVerifier,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    PostResultPopulationVerificationInput,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_result_streaming import (
    GroundedResultStreamCode,
    GroundedResultStreamLimits,
    GroundedResultStreamMaterializer,
    GroundedResultStreamingError,
)


GOAL_FINGERPRINT = hashlib.sha256(b"goal-contract").hexdigest()
GRAPH_FINGERPRINT = hashlib.sha256(b"execution-graph").hexdigest()
QUERY_FINGERPRINT = hashlib.sha256(b"query-contract").hexdigest()
SQL_FINGERPRINT = hashlib.sha256(b"validated-ast").hexdigest()
SNAPSHOT_FINGERPRINT = hashlib.sha256(b"data-snapshot").hexdigest()
POPULATION_FINGERPRINT = hashlib.sha256(b"population").hexdigest()
PROOF_FINGERPRINT = hashlib.sha256(b"lineage-proof").hexdigest()
ARTIFACT_AUTHORITY = hashlib.sha256(b"artifact-authority").hexdigest()
CONSUMER_GOAL = "goal.consumer"
QUERY_NODE = "node.consumer"


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


def _pre_attestation() -> PopulationVerificationAttestation:
    return _seal(
        PopulationVerificationAttestation(
            stage=PopulationVerificationStage.PRE_EXECUTION,
            passed=True,
            gate_open=True,
            input_fingerprint=hashlib.sha256(b"pre-input").hexdigest(),
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            accepted_scopes=(
                PopulationScopeAttestation(
                    consumer_goal_id=CONSUMER_GOAL,
                    scope_kind="SAME_AS_GOAL",
                    source_goal_ids=("goal.source",),
                    declaration_scope_fingerprint=hashlib.sha256(
                        b"declaration-scope"
                    ).hexdigest(),
                    population_fingerprint=POPULATION_FINGERPRINT,
                    grain_fingerprint=hashlib.sha256(b"grain").hexdigest(),
                    complete_membership_required=True,
                    query_node_id=QUERY_NODE,
                    generation=4,
                    attempt_id="attempt-4",
                    query_contract_fingerprint=QUERY_FINGERPRINT,
                    sql_ast_fingerprint=SQL_FINGERPRINT,
                    snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
                    proof_fingerprints=(PROOF_FINGERPRINT,),
                ),
            ),
            accepted_proof_fingerprints=(PROOF_FINGERPRINT,),
        )
    )


def _post_result(
    *,
    artifact_id: str,
    artifact_fingerprint: str,
    coverage: PopulationArtifactCoverage,
) -> object:
    evidence = PopulationArtifactEvidence(
        artifact_id=artifact_id,
        artifact_fingerprint=artifact_fingerprint,
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=coverage,
        population_fingerprint=POPULATION_FINGERPRINT,
        verifier_fingerprint=ARTIFACT_AUTHORITY,
        verified=True,
        immutable=True,
        goal_contract_fingerprint=GOAL_FINGERPRINT,
        graph_fingerprint=GRAPH_FINGERPRINT,
        query_contract_fingerprint=QUERY_FINGERPRINT,
        sql_ast_fingerprint=SQL_FINGERPRINT,
        snapshot_fingerprint=SNAPSHOT_FINGERPRINT,
        lineage_proof_fingerprints=(PROOF_FINGERPRINT,),
    )
    return PopulationSemanticVerifier().verify_post_result(
        PostResultPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FINGERPRINT,
            graph_fingerprint=GRAPH_FINGERPRINT,
            pre_execution_attestation=_pre_attestation(),
            trusted_artifact_verifier_fingerprints=(ARTIFACT_AUTHORITY,),
            results=(
                PopulationResultEvidence(
                    consumer_goal_id=CONSUMER_GOAL,
                    query_node_id=QUERY_NODE,
                    result_artifact=evidence,
                    lineage_proof_fingerprints=(PROOF_FINGERPRINT,),
                ),
            ),
        )
    )


def _limits(*, preview_rows: int = 3) -> GroundedResultStreamLimits:
    return GroundedResultStreamLimits(
        preview_rows=preview_rows,
        fetch_batch_rows=5,
        max_rows=10_000,
        max_bytes=8 * 1024 * 1024,
    )


def _batches(row_count: int) -> Iterable[list[Mapping[str, object]]]:
    for offset in range(0, row_count, 5):
        yield [
            {
                "entity_key": index,
                "measure_value": row_count - index,
            }
            for index in range(offset, min(offset + 5, row_count))
        ]


def _gap_codes(result: object) -> set[str]:
    return {
        str(getattr(item.code, "value", item.code))
        for item in getattr(result, "gaps", ())
    }


def test_only_full_immutable_rows_become_post_population_evidence(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    receipt = GroundedResultStreamMaterializer(
        artifact_root
    ).materialize_batches(
        _batches(23),
        artifact_id="result-artifact",
        limits=_limits(preview_rows=3),
    )

    assert len(receipt.preview_rows) == 3
    assert receipt.preview_is_truncated is True
    assert receipt.exact_row_count == 23
    assert receipt.complete is True
    assert receipt.active is True
    assert receipt.immutable is True
    assert receipt.coverage == PopulationArtifactCoverage.ALL_ROWS.value
    rows_path = artifact_root / receipt.rows_relative_path
    assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == (
        receipt.rows_canonical_sha256
    )

    full = _post_result(
        artifact_id=receipt.artifact_id,
        artifact_fingerprint=receipt.rows_canonical_sha256,
        coverage=PopulationArtifactCoverage(receipt.coverage),
    )
    preview = _post_result(
        artifact_id=receipt.artifact_id,
        artifact_fingerprint=receipt.rows_canonical_sha256,
        coverage=PopulationArtifactCoverage.PREVIEW,
    )

    assert full.passed is True
    assert full.gate_open is True
    assert preview.passed is False
    assert PopulationGapCode.ARTIFACT_COVERAGE_INCOMPLETE.value in (
        _gap_codes(preview)
    )


def test_partial_stream_never_produces_an_active_post_artifact(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    def interrupted_batches() -> Iterable[list[Mapping[str, object]]]:
        yield [{"entity_key": 1}, {"entity_key": 2}]
        raise RuntimeError("source interrupted")

    with pytest.raises(GroundedResultStreamingError) as raised:
        GroundedResultStreamMaterializer(
            artifact_root
        ).materialize_batches(
            interrupted_batches(),
            artifact_id="partial-artifact",
            limits=_limits(),
        )

    assert raised.value.code == GroundedResultStreamCode.SOURCE_FAILED
    assert raised.value.partial.row_count == 2
    assert not list(artifact_root.rglob("rows.json"))
    assert not list(artifact_root.rglob("*.sha256"))


def test_rows_tamper_breaks_immutable_authority_before_post(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    receipt = GroundedResultStreamMaterializer(
        artifact_root
    ).materialize_batches(
        _batches(9),
        artifact_id="tamper-artifact",
        limits=_limits(),
    )
    rows_path = artifact_root / receipt.rows_relative_path
    rows_path.chmod(0o600)
    rows_path.write_bytes(b"[]")

    read_result = WorkspaceArtifactStore(
        Settings(harness_workspace_path=str(tmp_path / "workspace")),
        artifact_root,
    ).read(
        receipt.rows_relative_path,
        offset=0,
        max_chars=1,
        require_immutable=True,
    )

    assert read_result["success"] is False
    assert read_result["error"] == "ARTIFACT_IMMUTABLE_STATE_INVALID"
    assert hashlib.sha256(rows_path.read_bytes()).hexdigest() != (
        receipt.rows_canonical_sha256
    )
