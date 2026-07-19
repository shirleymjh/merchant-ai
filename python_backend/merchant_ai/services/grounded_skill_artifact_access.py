from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from merchant_ai.models import ResultCoverage
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_runtime_kernel import (
    verified_query_artifact_integrity_valid,
)
from merchant_ai.services.sandbox import (
    SANDBOX_INPUT_ALLOWLIST_VERSION,
    SandboxArtifactAccess,
    SandboxVerifiedArtifactCommit,
)


class GroundedSkillArtifactAccessError(RuntimeError):
    def __init__(self, code: str):
        self.code = str(code or "SKILL_ARTIFACT_ACCESS_FAILED")
        super().__init__(self.code)


@dataclass(frozen=True)
class GroundedSkillArtifactAccessBundle:
    access: SandboxArtifactAccess
    selected_artifact_ids: tuple[str, ...]
    artifact_catalog: tuple[dict[str, Any], ...]
    allowed_artifact_digests: Mapping[str, str]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _relative_artifact_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_RELATIVE_PATH_INVALID"
        )
    return path.as_posix()


def _merchant_ref(value: Any) -> str:
    ref = str(value or "").strip()
    if not ref.startswith("merchant://"):
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_OPAQUE_REF_INVALID"
        )
    return ref


def _object_value(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value.get(name)
        return None
    for name in names:
        observed = getattr(value, name, None)
        if observed is not None:
            return observed
    return None


def _population_use(receipt: Mapping[str, Any]) -> str:
    coverage = str(receipt.get("resultCoverage") or "")
    truncated = bool(receipt.get("resultIsTruncated"))
    stored = max(0, int(receipt.get("storedRowCount") or 0))
    exact = max(0, int(receipt.get("exactResultRowCount") or 0))
    if (
        coverage == ResultCoverage.ALL_ROWS.value
        and not truncated
        and exact == stored
    ):
        return "COMPLETE_POPULATION"
    if coverage == ResultCoverage.TOP_N.value and not truncated:
        return "RANKED_RESULT"
    return "OBSERVATION"


def _artifact_verified_evidence_hash(artifact: Any) -> str:
    verified = _object_value(
        artifact,
        "verified_evidence",
        "verifiedEvidence",
    )
    dump = getattr(verified, "model_dump", None)
    payload = (
        dump(by_alias=True, mode="json")
        if callable(dump)
        else verified
    )
    return _stable_hash(payload)


def _artifact_attempt_fingerprint(artifact: Any) -> str:
    attempt_id = str(
        _object_value(artifact, "attempt_id", "attemptId") or ""
    )
    return hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()


def _artifact_data_snapshot_identity(artifact: Any) -> dict[str, str]:
    run_result = _object_value(artifact, "run_result", "runResult")
    bundle = _object_value(
        run_result,
        "merged_query_bundle",
        "mergedQueryBundle",
    )
    snapshot = _object_value(bundle, "data_snapshot", "dataSnapshot")
    if snapshot is None:
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_DATA_SNAPSHOT_REQUIRED"
        )
    identity = {
        "datasourceFingerprint": str(
            _object_value(
                snapshot,
                "datasource_fingerprint",
                "datasourceFingerprint",
            )
            or ""
        ),
        "datasourceEnvironment": str(
            _object_value(
                snapshot,
                "datasource_environment",
                "datasourceEnvironment",
            )
            or ""
        ),
        "dataEpoch": str(
            _object_value(snapshot, "data_epoch", "dataEpoch") or ""
        ),
        "consistencyMode": str(
            _object_value(
                snapshot,
                "consistency_mode",
                "consistencyMode",
            )
            or "UNSUPPORTED"
        ),
        "semanticActivationFingerprint": str(
            _object_value(
                snapshot,
                "semantic_activation_fingerprint",
                "semanticActivationFingerprint",
            )
            or ""
        ),
        "cacheGeneration": str(
            _object_value(
                snapshot,
                "cache_generation",
                "cacheGeneration",
            )
            or ""
        ),
        "capturedAt": str(
            _object_value(snapshot, "captured_at", "capturedAt") or ""
        ),
        "unsupportedReason": str(
            _object_value(
                snapshot,
                "unsupported_reason",
                "unsupportedReason",
            )
            or ""
        ),
    }
    if (
        not _valid_sha256(identity["datasourceFingerprint"])
        or not _valid_sha256(
            identity["semanticActivationFingerprint"]
        )
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_DATA_SNAPSHOT_IDENTITY_INVALID"
        )
    return identity


def _validated_receipt(
    artifact: Any,
    raw_receipt: Any,
    *,
    expected_owner_fingerprint: str,
    expected_semantic_activation_fingerprint: str = "",
    expected_semantic_activation_seal_fingerprint: str = "",
) -> dict[str, Any]:
    if not isinstance(raw_receipt, Mapping):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_RECEIPT_INVALID"
        )
    receipt = dict(raw_receipt)
    digest_fields = (
        "artifactFingerprint",
        "queryManifestSha256",
        "rowsSha256",
        "sqlSha256",
        "verifiedEvidenceSha256",
        "dataSnapshotFingerprint",
        "attemptFingerprint",
        "contractFingerprint",
        "sqlEvidenceFingerprint",
    )
    if any(not _valid_sha256(receipt.get(key)) for key in digest_fields):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_DIGEST_INVALID"
        )
    content_address_pairs = (
        ("manifestContentAddress", "queryManifestSha256"),
        ("rowsContentAddress", "rowsSha256"),
        ("sqlContentAddress", "sqlSha256"),
    )
    for address_key, digest_key in content_address_pairs:
        if str(receipt.get(address_key) or "") != (
            "sha256:%s" % str(receipt.get(digest_key) or "")
        ):
            raise GroundedSkillArtifactAccessError(
                "SKILL_PUBLISHED_ARTIFACT_CONTENT_ADDRESS_INVALID"
            )
    for key in ("manifestRef", "rowsRef", "sqlRef"):
        _merchant_ref(receipt.get(key))
    for key in (
        "manifestRelativePath",
        "rowsRelativePath",
        "sqlRelativePath",
    ):
        receipt[key] = _relative_artifact_path(receipt.get(key))

    owner = str(receipt.get("contextOwnerFingerprint") or "")
    semantic = str(
        receipt.get("semanticActivationFingerprint") or ""
    )
    if owner != expected_owner_fingerprint or not _valid_sha256(owner):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_OWNER_MISMATCH"
        )
    if not _valid_sha256(semantic):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_SEMANTIC_ACTIVATION_INVALID"
        )
    if (
        expected_semantic_activation_fingerprint
        and semantic != expected_semantic_activation_fingerprint
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_ACTIVE_SEMANTIC_MISMATCH"
        )
    coverage = str(receipt.get("resultCoverage") or "")
    if coverage not in {item.value for item in ResultCoverage}:
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_COVERAGE_INVALID"
        )
    if not isinstance(receipt.get("resultIsTruncated"), bool):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_COVERAGE_RECEIPT_INVALID"
        )
    for key in (
        "storedRowCount",
        "exactResultRowCount",
        "executionGeneration",
    ):
        value = receipt.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GroundedSkillArtifactAccessError(
                "SKILL_PUBLISHED_ARTIFACT_COUNT_INVALID"
            )
    if int(receipt.get("executionGeneration") or 0) <= 0:
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_GENERATION_INVALID"
        )

    artifact_contract = str(
        _object_value(
            artifact,
            "contract_fingerprint",
            "contractFingerprint",
        )
        or ""
    )
    artifact_sql = str(
        _object_value(artifact, "sql_fingerprint", "sqlFingerprint")
        or ""
    )
    artifact_generation = int(
        _object_value(artifact, "generation") or 0
    )
    if (
        str(receipt.get("contractFingerprint") or "")
        != artifact_contract
        or str(receipt.get("sqlEvidenceFingerprint") or "")
        != artifact_sql
        or int(receipt.get("executionGeneration") or 0)
        != artifact_generation
        or str(receipt.get("attemptFingerprint") or "")
        != _artifact_attempt_fingerprint(artifact)
        or str(receipt.get("verifiedEvidenceSha256") or "")
        != _artifact_verified_evidence_hash(artifact)
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_LEDGER_BINDING_MISMATCH"
        )
    data_snapshot = _artifact_data_snapshot_identity(artifact)
    if (
        str(receipt.get("dataSnapshotFingerprint") or "")
        != _stable_hash(data_snapshot)
        or str(receipt.get("semanticActivationFingerprint") or "")
        != data_snapshot["semanticActivationFingerprint"]
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_DATA_SNAPSHOT_BINDING_MISMATCH"
        )
    artifact_semantic = str(
        _object_value(
            artifact,
            "semantic_activation_fingerprint",
            "semanticActivationFingerprint",
        )
        or ""
    )
    artifact_semantic_seal = str(
        _object_value(
            artifact,
            "semantic_activation_seal_fingerprint",
            "semanticActivationSealFingerprint",
        )
        or ""
    )
    if artifact_semantic and artifact_semantic != semantic:
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_LEDGER_SEMANTIC_MISMATCH"
        )
    if expected_semantic_activation_fingerprint and (
        artifact_semantic
        != expected_semantic_activation_fingerprint
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_ACTIVE_SEMANTIC_MISMATCH"
        )
    if expected_semantic_activation_seal_fingerprint and (
        artifact_semantic_seal
        != expected_semantic_activation_seal_fingerprint
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_PUBLISHED_ARTIFACT_ACTIVE_SEAL_MISMATCH"
        )
    receipt["_serverDataSnapshot"] = data_snapshot
    return receipt


def build_grounded_skill_artifact_access(
    *,
    settings: Any,
    trusted_workspace_root: Path,
    artifact_root: Path,
    sandbox_staging_root: Path,
    owner_fingerprint: str,
    verified_query_artifacts: Sequence[Any],
    selected_artifact_ids: Iterable[str],
    skill_run_id: str,
    expected_semantic_activation_fingerprint: str = "",
    expected_semantic_activation_seal_fingerprint: str = "",
) -> GroundedSkillArtifactAccessBundle:
    if settings is None:
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_SETTINGS_REQUIRED"
        )
    root = Path(artifact_root).resolve(strict=True)
    configured_root = Path(trusted_workspace_root).resolve(strict=True)
    try:
        root.relative_to(configured_root)
    except ValueError as exc:
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_ROOT_OUTSIDE_WORKSPACE"
        ) from exc
    staging_root = Path(sandbox_staging_root).resolve(strict=True)
    try:
        staging_root.relative_to(configured_root)
    except ValueError as exc:
        raise GroundedSkillArtifactAccessError(
            "SKILL_SANDBOX_STAGING_ROOT_OUTSIDE_WORKSPACE"
        ) from exc
    if not staging_root.is_dir() or staging_root == root:
        raise GroundedSkillArtifactAccessError(
            "SKILL_SANDBOX_STAGING_ROOT_INVALID"
        )
    selected = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in selected_artifact_ids
            if str(item).strip()
        )
    )
    if not selected:
        raise GroundedSkillArtifactAccessError(
            "SKILL_VERIFIED_ARTIFACT_SELECTION_REQUIRED"
        )
    ledger = {
        str(_object_value(item, "artifact_id", "artifactId") or ""): item
        for item in verified_query_artifacts
        if str(_object_value(item, "artifact_id", "artifactId") or "")
    }
    if any(artifact_id not in ledger for artifact_id in selected):
        raise GroundedSkillArtifactAccessError(
            "SKILL_SELECTED_ARTIFACT_NOT_IN_VERIFIED_LEDGER"
        )

    semantic_fingerprints: set[str] = set()
    inputs: list[dict[str, Any]] = []
    commits: list[SandboxVerifiedArtifactCommit] = []
    catalog: list[dict[str, Any]] = []
    allowed_digests: dict[str, str] = {}
    observed_fingerprints: set[str] = set()
    for artifact_id in selected:
        artifact = ledger[artifact_id]
        if not verified_query_artifact_integrity_valid(artifact):
            raise GroundedSkillArtifactAccessError(
                "SKILL_SELECTED_ARTIFACT_LEDGER_INTEGRITY_INVALID"
            )
        verified = _object_value(
            artifact,
            "verified_evidence",
            "verifiedEvidence",
        )
        if not bool(_object_value(verified, "passed")):
            raise GroundedSkillArtifactAccessError(
                "SKILL_SELECTED_ARTIFACT_NOT_VERIFIED"
            )
        if str(
            _object_value(
                artifact,
                "publication_status",
                "publicationStatus",
            )
            or ""
        ) != "PUBLISHED":
            raise GroundedSkillArtifactAccessError(
                "SKILL_SELECTED_ARTIFACT_NOT_PUBLISHED"
            )
        receipts = list(
            _object_value(
                artifact,
                "result_artifact_receipts",
                "resultArtifactReceipts",
            )
            or []
        )
        if not receipts:
            raise GroundedSkillArtifactAccessError(
                "SKILL_PUBLISHED_ARTIFACT_RECEIPT_REQUIRED"
            )
        for raw_receipt in receipts:
            receipt = _validated_receipt(
                artifact,
                raw_receipt,
                expected_owner_fingerprint=owner_fingerprint,
                expected_semantic_activation_fingerprint=(
                    expected_semantic_activation_fingerprint
                ),
                expected_semantic_activation_seal_fingerprint=(
                    expected_semantic_activation_seal_fingerprint
                ),
            )
            fingerprint = str(receipt["artifactFingerprint"])
            if fingerprint in observed_fingerprints:
                raise GroundedSkillArtifactAccessError(
                    "SKILL_PUBLISHED_ARTIFACT_FINGERPRINT_CONFLICT"
                )
            observed_fingerprints.add(fingerprint)
            semantic = str(receipt["semanticActivationFingerprint"])
            semantic_fingerprints.add(semantic)
            input_id = "query_%03d_%s" % (
                len(inputs) + 1,
                fingerprint[:16],
            )
            population_use = _population_use(receipt)
            inputs.append(
                {
                    "inputId": input_id,
                    "artifactFingerprint": fingerprint,
                    "queryManifest": {
                        "relativePath": receipt["manifestRelativePath"],
                        "sha256": receipt["queryManifestSha256"],
                        "contentAddress": receipt[
                            "manifestContentAddress"
                        ],
                    },
                    "requiredCoverage": receipt["resultCoverage"],
                    "populationUse": population_use,
                    "verifiedReceipt": {
                        key: receipt.get(key)
                        for key in (
                            "sqlSha256",
                            "sqlRelativePath",
                            "sqlContentAddress",
                            "executionGeneration",
                            "attemptFingerprint",
                            "contractFingerprint",
                            "sqlEvidenceFingerprint",
                            "dataSnapshotFingerprint",
                            "verifiedEvidenceSha256",
                        )
                    },
                }
            )
            commits.append(
                SandboxVerifiedArtifactCommit(
                    artifact_fingerprint=fingerprint,
                    query_manifest_sha256=str(
                        receipt["queryManifestSha256"]
                    ),
                    rows_sha256=str(receipt["rowsSha256"]),
                    result_coverage=str(receipt["resultCoverage"]),
                    data_snapshot_fingerprint=str(
                        receipt["dataSnapshotFingerprint"]
                    ),
                    datasource_fingerprint=str(
                        receipt["_serverDataSnapshot"][
                            "datasourceFingerprint"
                        ]
                    ),
                    datasource_environment=str(
                        receipt["_serverDataSnapshot"][
                            "datasourceEnvironment"
                        ]
                    ),
                    semantic_activation_fingerprint=str(
                        receipt["_serverDataSnapshot"][
                            "semanticActivationFingerprint"
                        ]
                    ),
                    cache_generation=str(
                        receipt["_serverDataSnapshot"][
                            "cacheGeneration"
                        ]
                    ),
                )
            )
            # SQL remains a server-side verification input only. It can carry
            # tenant predicates or row-policy literals, so analysis workers
            # receive the verified manifest and rows but never the SQL bytes.
            for path_key, digest_key in (
                ("manifestRelativePath", "queryManifestSha256"),
                ("rowsRelativePath", "rowsSha256"),
            ):
                allowed_digests[str(receipt[path_key])] = str(
                    receipt[digest_key]
                )
            complete_population = (
                population_use == "COMPLETE_POPULATION"
            )
            catalog.append(
                {
                    "inputId": input_id,
                    "queryArtifactId": artifact_id,
                    "artifactFingerprint": fingerprint,
                    "resultCoverage": str(receipt["resultCoverage"]),
                    "resultIsTruncated": bool(
                        receipt["resultIsTruncated"]
                    ),
                    "storedRowCount": int(receipt["storedRowCount"]),
                    "exactResultRowCount": int(
                        receipt["exactResultRowCount"]
                    ),
                    "populationUse": population_use,
                    "completePopulation": complete_population,
                    "manifestRef": _merchant_ref(
                        receipt["manifestRef"]
                    ),
                    "rowsRef": _merchant_ref(receipt["rowsRef"]),
                    "manifestArtifact": "/artifacts/%s"
                    % receipt["manifestRelativePath"],
                    "rowsArtifact": "/artifacts/%s"
                    % receipt["rowsRelativePath"],
                }
            )
    if len(semantic_fingerprints) != 1:
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_SEMANTIC_ACTIVATION_CONFLICT"
        )
    semantic_fingerprint = next(iter(semantic_fingerprints))
    if (
        expected_semantic_activation_fingerprint
        and semantic_fingerprint
        != expected_semantic_activation_fingerprint
    ):
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_ACTIVE_SEMANTIC_MISMATCH"
        )
    allowlist_payload = {
        "schemaVersion": SANDBOX_INPUT_ALLOWLIST_VERSION,
        "manifestKind": "SANDBOX_INPUT_ALLOWLIST",
        "contextOwnerFingerprint": owner_fingerprint,
        "semanticActivationFingerprint": semantic_fingerprint,
        "selectedQueryArtifactIds": list(selected),
        "inputs": inputs,
    }
    allowlist_fingerprint = _stable_hash(allowlist_payload)
    allowlist = WorkspaceArtifactStore(settings, root).write_json(
        "sandbox_inputs",
        "skill_%s_%s.allowlist.json"
        % (
            hashlib.sha256(
                str(skill_run_id or "").encode("utf-8")
            ).hexdigest()[:16],
            allowlist_fingerprint,
        ),
        allowlist_payload,
        preview_chars=0,
        immutable=True,
    )
    if not allowlist.get("success") or not allowlist.get("immutable"):
        raise GroundedSkillArtifactAccessError(
            "SKILL_ARTIFACT_ALLOWLIST_WRITE_FAILED"
        )
    access = SandboxArtifactAccess(
        run_artifact_root=root,
        allowlist_manifest_path=Path(str(allowlist.get("path") or "")),
        allowlist_sha256=str(allowlist.get("sha256") or ""),
        allowlist_content_address=str(
            allowlist.get("contentAddress") or ""
        ),
        expected_owner_fingerprint=owner_fingerprint,
        expected_semantic_activation_fingerprint=semantic_fingerprint,
        verified_query_artifact_commits=tuple(commits),
        trusted_workspace_root=configured_root,
        sandbox_staging_root=staging_root,
    )
    return GroundedSkillArtifactAccessBundle(
        access=access,
        selected_artifact_ids=selected,
        artifact_catalog=tuple(catalog),
        allowed_artifact_digests=dict(allowed_digests),
    )
