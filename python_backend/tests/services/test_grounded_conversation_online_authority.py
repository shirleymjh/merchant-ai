from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_conversation_online_authority import (
    GroundedConversationOnlineAuthorityFacade,
)
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationSemanticProviderOutput,
    ConversationSemanticResolverRequest,
)
from merchant_ai.services.grounded_conversation_state import (
    grounded_conversation_principal_fingerprint,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _SemanticProvider:
    def __init__(
        self,
        *,
        authority_fingerprint: str = "conversation-reviewer",
        reference_detected: bool = True,
        population_required: bool = True,
        complete_membership_required: bool = True,
    ) -> None:
        self._authority_fingerprint = authority_fingerprint
        self.reference_detected = reference_detected
        self.population_required = population_required
        self.complete_membership_required = complete_membership_required
        self.requests: list[ConversationSemanticResolverRequest] = []

    @property
    def authority_fingerprint(self) -> str:
        return self._authority_fingerprint

    def resolve_conversation_reference(
        self,
        request: ConversationSemanticResolverRequest,
        *,
        timeout_seconds: float,
    ) -> ConversationSemanticProviderOutput:
        assert timeout_seconds > 0
        self.requests.append(request)
        selected = (
            request.candidates[0].artifact_id
            if self.reference_detected and request.candidates
            else ""
        )
        return ConversationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            candidate_set_fingerprint=request.candidate_set_fingerprint,
            complete=True,
            reference_detected=self.reference_detected,
            ambiguous=False,
            selected_artifact_id=selected,
            referent_type=(
                ConversationReferenceType.RESULT_ARTIFACT
                if self.reference_detected
                else ConversationReferenceType.NONE
            ),
            downstream_operation=(
                ConversationDownstreamOperation.RANK
                if self.reference_detected
                else ConversationDownstreamOperation.UNSPECIFIED
            ),
            population_required=(
                self.population_required if self.reference_detected else False
            ),
            complete_membership_required=(
                self.complete_membership_required
                if self.reference_detected
                else False
            ),
            reference_phrases=("这里面",) if self.reference_detected else (),
        )


def _published_snapshot(
    tmp_path: Path,
    *,
    publication_status: str = "PUBLISHED",
    result_coverage: str = "ALL_ROWS",
    result_is_truncated: bool = False,
) -> dict[str, Any]:
    settings = Settings(
        harness_workspace_path=str(tmp_path / "runtime")
    )
    merchant_id = "merchant-1"
    access_role = "merchant_analyst"
    user_scope = {
        "userId": "user-1",
        "role": "merchant_operator",
        "storeIds": ["store-1"],
    }
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-1",
        run_id="run-1",
        merchant_id=merchant_id,
        access_role=access_role,
        user_scope=user_scope,
        question="最近7天订单明细",
    )
    store = WorkspaceArtifactStore(settings, workspace.artifacts_root)
    rows = [
        {"order_id": "order-1", "refund_amount": 30},
        {"order_id": "order-2", "refund_amount": 20},
    ]
    rows_encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rows_artifact = store.write_text(
        "query_results",
        "rows.json",
        rows_encoded,
        preview_chars=0,
        immutable=True,
    )
    sql_artifact = store.write_text(
        "query_results",
        "query.sql",
        "SELECT governed_columns FROM governed_orders",
        preview_chars=0,
        immutable=True,
    )
    contract = GroundedQueryContract(
        question="最近7天订单明细",
        topics=["订单管理"],
        status="READY",
        query_shape="DETAIL",
    )
    contract_fingerprint = grounded_query_contract_fingerprint(contract)
    sql_fingerprint = hashlib.sha256(b"sql-evidence").hexdigest()
    semantic_activation_fingerprint = hashlib.sha256(
        b"semantic-activation"
    ).hexdigest()
    data_snapshot = {
        "semanticActivationFingerprint": (
            semantic_activation_fingerprint
        ),
        "datasourceFingerprint": hashlib.sha256(
            b"datasource"
        ).hexdigest(),
    }
    verified_evidence = {"passed": True, "blockingGaps": []}
    attempt_id = "attempt-1"
    artifact_fingerprint = hashlib.sha256(b"artifact-1").hexdigest()
    manifest = {
        "schemaVersion": 3,
        "artifactKind": "GROUNDED_QUERY_RESULT",
        "publicationStatus": "VERIFIED",
        "artifactFingerprint": artifact_fingerprint,
        "executionGeneration": 1,
        "executionAttemptId": attempt_id,
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_fingerprint,
        "sqlSha256": sql_artifact["sha256"],
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": (
            semantic_activation_fingerprint
        ),
        "dataSnapshot": data_snapshot,
        "resultCoverage": result_coverage,
        "resultIsTruncated": result_is_truncated,
        "storedRowCount": len(rows),
        "artifactRowCount": len(rows),
        "artifactByteCount": rows_artifact["bytes"],
        "artifactCoverage": "ALL_ROWS",
        "artifactComplete": True,
        "exactResultRowCount": len(rows),
        "verifiedEvidence": verified_evidence,
        "verifiedEvidenceSha256": _stable_hash(verified_evidence),
        "rowsArtifact": {
            "relativePath": rows_artifact["relativePath"],
            "sha256": rows_artifact["sha256"],
            "contentAddress": rows_artifact["contentAddress"],
            "bytes": rows_artifact["bytes"],
            "immutable": True,
        },
        "sqlArtifact": {
            "relativePath": sql_artifact["relativePath"],
            "sha256": sql_artifact["sha256"],
            "contentAddress": sql_artifact["contentAddress"],
            "bytes": sql_artifact["bytes"],
            "immutable": True,
        },
    }
    manifest_encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_artifact = store.write_text(
        "query_results",
        "manifest.json",
        manifest_encoded,
        preview_chars=0,
        immutable=True,
    )
    receipt = {
        "artifactFingerprint": artifact_fingerprint,
        "manifestRelativePath": manifest_artifact["relativePath"],
        "rowsRelativePath": rows_artifact["relativePath"],
        "sqlRelativePath": sql_artifact["relativePath"],
        "queryManifestSha256": manifest_artifact["sha256"],
        "rowsSha256": rows_artifact["sha256"],
        "sqlSha256": sql_artifact["sha256"],
        "manifestContentAddress": manifest_artifact["contentAddress"],
        "rowsContentAddress": rows_artifact["contentAddress"],
        "sqlContentAddress": sql_artifact["contentAddress"],
        "storedRowCount": len(rows),
        "artifactRowCount": len(rows),
        "artifactByteCount": rows_artifact["bytes"],
        "artifactCoverage": "ALL_ROWS",
        "artifactComplete": True,
        "exactResultRowCount": len(rows),
        "resultCoverage": result_coverage,
        "resultIsTruncated": result_is_truncated,
        "executionGeneration": 1,
        "attemptFingerprint": hashlib.sha256(
            attempt_id.encode("utf-8")
        ).hexdigest(),
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_fingerprint,
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": (
            semantic_activation_fingerprint
        ),
        "dataSnapshotFingerprint": _stable_hash(data_snapshot),
        "verifiedEvidenceSha256": _stable_hash(verified_evidence),
    }
    artifact_id = "query-artifact-1"
    source_record = {
        "queryArtifactId": artifact_id,
        "publicationStatus": publication_status,
        "artifactRootRelativePath": str(
            workspace.artifacts_root.relative_to(
                settings.resolved_workspace_path
            )
        ),
        "contractFingerprint": contract_fingerprint,
        "sqlFingerprint": sql_fingerprint,
        "contract": contract.model_dump(by_alias=True, mode="json"),
        "publicationReceipt": receipt,
    }
    snapshot = {
        "principalFingerprint": (
            grounded_conversation_principal_fingerprint(
                merchant_id,
                user_scope,
            )
        ),
        "activeScope": {
            "artifactIds": [artifact_id],
            "sourceArtifacts": [source_record],
        },
    }
    return {
        "settings": settings,
        "workspace": workspace,
        "merchantId": merchant_id,
        "accessRole": access_role,
        "userScope": user_scope,
        "semanticActivationFingerprint": (
            semantic_activation_fingerprint
        ),
        "principalFingerprint": snapshot["principalFingerprint"],
        "snapshot": snapshot,
        "sourceRecord": source_record,
        "receipt": receipt,
    }


def _facade(
    values: dict[str, Any],
    provider: _SemanticProvider | None,
) -> GroundedConversationOnlineAuthorityFacade:
    return GroundedConversationOnlineAuthorityFacade(
        workspace_root=values["settings"].resolved_workspace_path,
        semantic_provider=provider,
        trusted_reviewer_authority_fingerprints=(
            "conversation-reviewer",
        ),
        core_authority_fingerprint="core-authority",
        review_timeout_seconds=1,
    )


def _resolve(
    values: dict[str, Any],
    facade: GroundedConversationOnlineAuthorityFacade,
    *,
    question: str = "这里面退款最多的三单",
):
    return facade.resolve(
        question,
        persisted_snapshot=values["snapshot"],
        persisted_revision=4,
        expected_principal_fingerprint=(
            values["principalFingerprint"]
        ),
        expected_context_owner_fingerprint=(
            values["workspace"].owner_fingerprint
        ),
        expected_semantic_activation_fingerprint=(
            values["semanticActivationFingerprint"]
        ),
    )


def test_facade_reopens_published_artifact_before_binding_population(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    provider = _SemanticProvider()

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.reference_contract.status == "BOUND"
    assert resolution.reference_contract.referent_type == "RESULT_ARTIFACT"
    assert resolution.reference_contract.population_required is True
    assert resolution.reference_contract.complete_membership_required is True
    assert resolution.reference_contract.membership_handle_type == (
        "PUBLISHED_RESULT_ROWS"
    )
    assert resolution.reference_contract.membership_handle_id == (
        values["receipt"]["rowsContentAddress"]
    )
    assert resolution.reference_contract.membership_values_hash == (
        values["receipt"]["rowsSha256"]
    )
    assert len(provider.requests) == 1
    candidate_payload = provider.requests[0].candidates[0].model_dump(
        by_alias=True,
        mode="json",
    )
    assert candidate_payload["label"] == "最近7天订单明细"
    assert "order-1" not in json.dumps(
        candidate_payload,
        ensure_ascii=False,
    )


def test_standalone_turn_without_retained_artifact_skips_semantic_review(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    values["snapshot"] = {}
    provider = _SemanticProvider(reference_detected=False)

    resolution = _resolve(
        values,
        _facade(values, provider),
        question="最近7天订单明细",
    )

    assert resolution.status == "STANDALONE"
    assert resolution.reference_detected is False
    assert len(provider.requests) == 0


@pytest.mark.parametrize(
    "publication_status",
    ["PENDING", "VERIFIED", "FAILED"],
)
def test_non_published_candidate_rejects_entire_batch_before_provider(
    tmp_path: Path,
    publication_status: str,
) -> None:
    values = _published_snapshot(
        tmp_path,
        publication_status=publication_status,
    )
    provider = _SemanticProvider()

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert resolution.clarification_type == (
        "CONVERSATION_PUBLISHED_CONTEXT_INVALID"
    )
    assert "CONVERSATION_AUTHORITY_PUBLICATION_REQUIRED" in (
        resolution.semantic_issue_codes
    )
    assert provider.requests == []


def test_rows_tamper_rejects_candidate_before_provider(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    rows_path = (
        values["workspace"].artifacts_root
        / values["receipt"]["rowsRelativePath"]
    )
    rows_path.chmod(0o600)
    rows_path.write_text("[]", encoding="utf-8")
    provider = _SemanticProvider()

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert any(
        code
        in {
            "CONVERSATION_AUTHORITY_ROWS_BINDING_MISMATCH",
            "CONVERSATION_AUTHORITY_ARTIFACT_BYTES_INVALID",
        }
        for code in resolution.semantic_issue_codes
    )
    assert provider.requests == []


def test_cross_principal_replay_rejects_candidate_before_provider(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    provider = _SemanticProvider()

    resolution = _facade(values, provider).resolve(
        "这里面退款最多的三单",
        persisted_snapshot=values["snapshot"],
        expected_principal_fingerprint=hashlib.sha256(
            b"another-principal"
        ).hexdigest(),
        expected_context_owner_fingerprint=(
            values["workspace"].owner_fingerprint
        ),
        expected_semantic_activation_fingerprint=(
            values["semanticActivationFingerprint"]
        ),
    )

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert "CONVERSATION_AUTHORITY_PRINCIPAL_MISMATCH" in (
        resolution.semantic_issue_codes
    )
    assert provider.requests == []


def test_missing_server_activation_binding_fails_closed_as_typed_resolution(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    provider = _SemanticProvider()

    resolution = _facade(values, provider).resolve(
        "这里面退款最多的三单",
        persisted_snapshot=values["snapshot"],
        expected_principal_fingerprint=values["principalFingerprint"],
        expected_context_owner_fingerprint=(
            values["workspace"].owner_fingerprint
        ),
        expected_semantic_activation_fingerprint="",
    )

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert "CONVERSATION_AUTHORITY_SNAPSHOT_INVALID" in (
        resolution.semantic_issue_codes
    )
    assert provider.requests == []


def test_preview_population_cannot_satisfy_complete_membership(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(
        tmp_path,
        result_coverage="PREVIEW",
        result_is_truncated=True,
    )
    provider = _SemanticProvider(
        complete_membership_required=True,
    )

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "UNSAFE_REFERENCE"
    assert resolution.needs_clarification is True
    assert resolution.reference_contract.complete_membership_required is True
    assert resolution.reference_contract.membership_handle_id == ""


def test_unavailable_semantic_reviewer_returns_typed_clarification(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)

    resolution = _resolve(values, _facade(values, None))

    assert resolution.status == "SEMANTIC_REVIEW_UNAVAILABLE"
    assert resolution.needs_clarification is True
    assert resolution.clarification_type == (
        "CONVERSATION_SEMANTIC_REVIEW_UNAVAILABLE"
    )


def test_orphan_source_record_rejects_entire_batch(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    values["snapshot"]["activeScope"]["artifactIds"].append(
        "query-artifact-orphan"
    )
    provider = _SemanticProvider()

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert "CONVERSATION_AUTHORITY_ARTIFACT_INDEX_INVALID" in (
        resolution.semantic_issue_codes
    )
    assert provider.requests == []


def test_path_escape_rejects_candidate_without_opening_outside_workspace(
    tmp_path: Path,
) -> None:
    values = _published_snapshot(tmp_path)
    values["sourceRecord"]["artifactRootRelativePath"] = "../outside"
    provider = _SemanticProvider()

    resolution = _resolve(values, _facade(values, provider))

    assert resolution.status == "PUBLISHED_CONTEXT_AUTHORITY_REJECTED"
    assert "CONVERSATION_AUTHORITY_PATH_INVALID" in (
        resolution.semantic_issue_codes
    )
    assert provider.requests == []
