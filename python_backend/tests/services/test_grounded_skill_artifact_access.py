from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    DataSnapshotContract,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentSession,
    GroundedRunFilesystemBackend,
    GroundedSemanticBackend,
    _published_query_artifact_digests,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeSession,
    GroundedVerifiedQueryArtifact,
    verified_query_artifact_integrity_fingerprint,
)
from merchant_ai.services.grounded_skill_artifact_access import (
    GroundedSkillArtifactAccessError,
    build_grounded_skill_artifact_access,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.sandbox import MerchantAnalysisSandbox


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        harness_workspace_path=str(tmp_path / "runtime"),
        sandbox_backend="local",
        sandbox_unsafe_local_test_mode=True,
    )


def _workspace(
    settings: Settings,
) -> GroundedContextWorkspace:
    return GroundedContextWorkspace.open(
        settings,
        thread_id="thread-skill-artifacts",
        run_id="run-skill-artifacts",
        merchant_id="merchant-skill-artifacts",
        access_role="merchant_analyst",
        user_scope={"userId": "analyst-1"},
        question="analyze verified results",
    )


def _published_artifact(
    settings: Settings,
    workspace: GroundedContextWorkspace,
    *,
    tag: str,
    semantic_seed: str = "semantic-v1",
    coverage: str = "ALL_ROWS",
    truncated: bool = False,
    publication_status: str = "PUBLISHED",
    seal: bool = True,
) -> GroundedVerifiedQueryArtifact:
    semantic_fingerprint = hashlib.sha256(
        semantic_seed.encode("utf-8")
    ).hexdigest()
    datasource_fingerprint = hashlib.sha256(
        b"datasource-primary"
    ).hexdigest()
    snapshot = DataSnapshotContract(
        datasource_fingerprint=datasource_fingerprint,
        datasource_environment="test",
        consistency_mode="UNSUPPORTED",
        semantic_activation_fingerprint=semantic_fingerprint,
        cache_generation="cache-generation-7",
        captured_at="2026-07-19T00:00:00Z",
        unsupported_reason="TEST_SNAPSHOT_UNAVAILABLE",
    )
    rows = [
        {"entity_id": "%s-a" % tag, "value": 9},
        {"entity_id": "%s-b" % tag, "value": 4},
    ]
    store = WorkspaceArtifactStore(settings, workspace.artifacts_root)
    rows_artifact = store.write_json(
        "query_results",
        "%s.rows.json" % tag,
        rows,
        preview_chars=0,
        immutable=True,
    )
    sql_artifact = store.write_text(
        "query_results",
        "%s.sql" % tag,
        "SELECT governed_columns FROM governed_source",
        preview_chars=0,
        immutable=True,
    )
    contract = GroundedQueryContract(
        question="analyze verified results",
        status="READY",
        query_shape="SCALAR",
    )
    contract_fingerprint = grounded_query_contract_fingerprint(contract)
    sql_evidence_fingerprint = hashlib.sha256(
        ("sql-evidence:%s" % tag).encode("utf-8")
    ).hexdigest()
    verified = VerifiedEvidence(passed=True)
    verified_payload = verified.model_dump(by_alias=True, mode="json")
    verified_sha256 = _stable_hash(verified_payload)
    snapshot_payload = snapshot.model_dump(by_alias=True, mode="json")
    generation = 1
    attempt_id = "attempt-%s" % tag
    exact_count = len(rows) if not truncated else 0
    artifact_fingerprint = hashlib.sha256(
        ("published-artifact:%s" % tag).encode("utf-8")
    ).hexdigest()
    manifest_payload = {
        "schemaVersion": 2,
        "artifactKind": "GROUNDED_QUERY_RESULT",
        "publicationStatus": "VERIFIED",
        "artifactFingerprint": artifact_fingerprint,
        "executionGeneration": generation,
        "executionAttemptId": attempt_id,
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_evidence_fingerprint,
        "sqlSha256": sql_artifact["sha256"],
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": semantic_fingerprint,
        "dataSnapshot": snapshot_payload,
        "resultCoverage": coverage,
        "resultIsTruncated": truncated,
        "storedRowCount": len(rows),
        "exactResultRowCount": exact_count,
        "verifiedEvidence": verified_payload,
        "verifiedEvidenceSha256": verified_sha256,
        "rowsArtifact": {
            key: rows_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
        "sqlArtifact": {
            key: sql_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
    }
    manifest_artifact = store.write_json(
        "query_results",
        "%s.manifest.json" % tag,
        manifest_payload,
        preview_chars=0,
        immutable=True,
    )
    receipt = {
        "artifactFingerprint": artifact_fingerprint,
        "queryManifestSha256": manifest_artifact["sha256"],
        "rowsSha256": rows_artifact["sha256"],
        "sqlSha256": sql_artifact["sha256"],
        "manifestContentAddress": manifest_artifact["contentAddress"],
        "rowsContentAddress": rows_artifact["contentAddress"],
        "sqlContentAddress": sql_artifact["contentAddress"],
        "manifestRelativePath": manifest_artifact["relativePath"],
        "rowsRelativePath": rows_artifact["relativePath"],
        "sqlRelativePath": sql_artifact["relativePath"],
        "manifestRef": manifest_artifact["merchantUri"],
        "rowsRef": rows_artifact["merchantUri"],
        "sqlRef": sql_artifact["merchantUri"],
        "storedRowCount": len(rows),
        "exactResultRowCount": exact_count,
        "resultCoverage": coverage,
        "resultIsTruncated": truncated,
        "executionGeneration": generation,
        "attemptFingerprint": hashlib.sha256(
            attempt_id.encode("utf-8")
        ).hexdigest(),
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_evidence_fingerprint,
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": semantic_fingerprint,
        "dataSnapshotFingerprint": _stable_hash(snapshot_payload),
        "verifiedEvidenceSha256": verified_sha256,
    }
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            rows=rows,
            result_coverage=coverage,
            is_truncated=truncated,
            data_snapshot=snapshot,
        )
    )
    artifact = GroundedVerifiedQueryArtifact(
        artifact_id="query-%s" % tag,
        generation=generation,
        attempt_id=attempt_id,
        contract_fingerprint=contract_fingerprint,
        sql_fingerprint=sql_evidence_fingerprint,
        contract=contract,
        plan=QueryPlan(),
        run_result=run_result,
        verified_evidence=verified,
        publication_status=publication_status,
        result_artifact_receipts=[receipt],
        output_columns=["entity_id", "value"],
    )
    if seal:
        artifact.ledger_fingerprint = (
            verified_query_artifact_integrity_fingerprint(artifact)
        )
    return artifact


def _build_access(
    settings: Settings,
    workspace: GroundedContextWorkspace,
    artifacts: list[GroundedVerifiedQueryArtifact],
    selected_ids: list[str],
):
    return build_grounded_skill_artifact_access(
        settings=settings,
        trusted_workspace_root=workspace.root,
        artifact_root=workspace.artifacts_root,
        sandbox_staging_root=workspace.staging_root,
        owner_fingerprint=workspace.owner_fingerprint,
        verified_query_artifacts=artifacts,
        selected_artifact_ids=selected_ids,
        skill_run_id="skill-test-run",
    )


def test_skill_access_exposes_only_selected_published_manifest_and_rows(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    selected = _published_artifact(
        settings,
        workspace,
        tag="selected",
    )
    unselected = _published_artifact(
        settings,
        workspace,
        tag="unselected",
    )

    bundle = _build_access(
        settings,
        workspace,
        [selected, unselected],
        [selected.artifact_id],
    )

    receipt = selected.result_artifact_receipts[0]
    assert set(bundle.allowed_artifact_digests) == {
        receipt["manifestRelativePath"],
        receipt["rowsRelativePath"],
    }
    assert receipt["sqlRelativePath"] not in bundle.allowed_artifact_digests
    assert bundle.artifact_catalog[0]["populationUse"] == (
        "COMPLETE_POPULATION"
    )
    assert bundle.artifact_catalog[0]["completePopulation"] is True
    assert not any(
        key in bundle.artifact_catalog[0]
        for key in ("sqlRef", "sqlArtifact")
    )
    assert unselected.artifact_id not in bundle.selected_artifact_ids


def test_preview_is_an_observation_and_never_a_complete_population(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    preview = _published_artifact(
        settings,
        workspace,
        tag="preview",
        coverage="PREVIEW",
        truncated=True,
    )

    bundle = _build_access(
        settings,
        workspace,
        [preview],
        [preview.artifact_id],
    )

    assert bundle.artifact_catalog[0]["populationUse"] == "OBSERVATION"
    assert bundle.artifact_catalog[0]["completePopulation"] is False


def test_skill_access_rejects_unpublished_unsealed_and_mixed_activation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    unpublished = _published_artifact(
        settings,
        workspace,
        tag="pending",
        publication_status="VERIFIED_IN_MEMORY",
    )
    invalid_seal = _published_artifact(
        settings,
        workspace,
        tag="invalid-seal",
        seal=False,
    )
    first = _published_artifact(settings, workspace, tag="activation-one")
    second = _published_artifact(
        settings,
        workspace,
        tag="activation-two",
        semantic_seed="semantic-v2",
    )

    with pytest.raises(GroundedSkillArtifactAccessError) as unpublished_error:
        _build_access(
            settings,
            workspace,
            [unpublished],
            [unpublished.artifact_id],
        )
    assert "SKILL_SELECTED_ARTIFACT_NOT_PUBLISHED" in str(unpublished_error.value)
    with pytest.raises(GroundedSkillArtifactAccessError) as integrity_error:
        _build_access(
            settings,
            workspace,
            [invalid_seal],
            [invalid_seal.artifact_id],
        )
    assert "SKILL_SELECTED_ARTIFACT_LEDGER_INTEGRITY_INVALID" in str(
        integrity_error.value
    )
    with pytest.raises(GroundedSkillArtifactAccessError) as activation_error:
        _build_access(
            settings,
            workspace,
            [first, second],
            [first.artifact_id, second.artifact_id],
        )
    assert "SKILL_ARTIFACT_SEMANTIC_ACTIVATION_CONFLICT" in str(
        activation_error.value
    )


def test_core_dynamic_artifact_authority_denies_orphans_sql_and_stale_ledger(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    published = _published_artifact(
        settings,
        workspace,
        tag="core-visible",
    )
    orphan = WorkspaceArtifactStore(
        settings,
        workspace.artifacts_root,
    ).write_json(
        "query_results",
        "orphan.json",
        [{"secret": True}],
        preview_chars=0,
        immutable=True,
    )
    runtime = GroundedRuntimeSession(
        session_id="session-artifact-authority",
        question="analyze verified results",
        merchant_id="merchant-skill-artifacts",
        verified_query_ledger=[published],
    )
    session = GroundedDeepAgentSession(
        runtime=runtime,
        context_workspace=workspace,
    )
    backend = GroundedRunFilesystemBackend(
        root_kind="artifacts",
        read_only=True,
        settings=settings,
        allowed_artifact_digest_provider=(
            _published_query_artifact_digests
        ),
    )
    scope = GroundedSemanticBackend(object())
    receipt = published.result_artifact_receipts[0]

    with scope.scope(session):
        assert backend.read(
            "/artifacts/%s" % receipt["rowsRelativePath"]
        ).error is None
        assert backend.read(
            "/artifacts/%s" % receipt["manifestRelativePath"]
        ).error is None
        assert backend.read(
            "/artifacts/%s" % receipt["sqlRelativePath"]
        ).error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"
        assert backend.read(
            "/artifacts/%s" % orphan["relativePath"]
        ).error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"

        published.publication_status = "VERIFIED_IN_MEMORY"
        published.ledger_fingerprint = (
            verified_query_artifact_integrity_fingerprint(published)
        )
        assert backend.read(
            "/artifacts/%s" % receipt["rowsRelativePath"]
        ).error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"

        published.publication_status = "PUBLISHED"
        published.ledger_fingerprint = (
            verified_query_artifact_integrity_fingerprint(published)
        )
        published.result_artifact_receipts[0]["rowsSha256"] = (
            hashlib.sha256(b"ledger-tamper").hexdigest()
        )
        assert backend.read(
            "/artifacts/%s" % receipt["rowsRelativePath"]
        ).error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"


def test_v2_sandbox_rechecks_unsupported_snapshot_identity_and_omits_sql(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    published = _published_artifact(
        settings,
        workspace,
        tag="sandbox-v2",
    )
    bundle = _build_access(
        settings,
        workspace,
        [published],
        [published.artifact_id],
    )
    sandbox = MerchantAnalysisSandbox(settings)
    skill_root = tmp_path / "approved-skills"
    skill_root.mkdir()
    script = skill_root / "consume.py"
    script.write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "from pathlib import Path",
                "manifest_path = Path(os.environ['MERCHANT_ANALYSIS_INPUT_MANIFEST'])",
                "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))",
                "item = manifest['inputs'][0]",
                "rows = json.loads((manifest_path.parent / item['rowsPath']).read_text(encoding='utf-8'))",
                "sql_files = [str(path) for path in manifest_path.parent.rglob('*.sql')]",
                "print(json.dumps({'rows': len(rows), 'sqlFiles': sql_files}))",
            )
        ),
        encoding="utf-8",
    )
    sandbox.skill_root = skill_root.resolve()
    output = workspace.subagent_workspace("skill", "sandbox-v2")

    result = sandbox.run_python(
        script,
        [],
        output,
        5,
        artifact_access=bundle.access,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {"rows": 2, "sqlFiles": []}
    commit = bundle.access.verified_query_artifact_commits[0]
    assert commit.datasource_fingerprint
    assert commit.semantic_activation_fingerprint
    assert published.run_result.merged_query_bundle.data_snapshot.consistency_mode == (
        "UNSUPPORTED"
    )

    changed_access = replace(
        bundle.access,
        verified_query_artifact_commits=(
            replace(
                commit,
                datasource_fingerprint=hashlib.sha256(
                    b"different-datasource"
                ).hexdigest(),
            ),
        ),
    )
    rejected = sandbox.run_python(
        script,
        [],
        output,
        5,
        artifact_access=changed_access,
    )

    assert rejected.returncode == 126
    assert rejected.stderr == (
        "SANDBOX_QUERY_DATA_SNAPSHOT_IDENTITY_MISMATCH"
    )
