from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping

from merchant_ai.config import Settings
from merchant_ai.models import ResultCoverage


SANDBOX_INPUT_ALLOWLIST_VERSION = 1
SANDBOX_VERIFIED_INPUT_VERSION = 1


@dataclass
class SandboxResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class SandboxArtifactAccess:
    """Server-issued authority to consume artifacts from exactly one run.

    The expected identity values must come from the authenticated runtime, not
    from script arguments or from the allowlist document itself.  Both the
    allowlist and every listed query artifact remain content addressed.
    """

    run_artifact_root: Path
    allowlist_manifest_path: Path
    allowlist_sha256: str
    allowlist_content_address: str
    expected_owner_fingerprint: str
    expected_semantic_activation_fingerprint: str
    verified_query_artifact_commits: tuple["SandboxVerifiedArtifactCommit", ...]
    trusted_workspace_root: Path | None = None
    sandbox_staging_root: Path | None = None


@dataclass(frozen=True)
class SandboxVerifiedArtifactCommit:
    """Kernel-retained receipt for one query artifact eligible for analysis."""

    artifact_fingerprint: str
    query_manifest_sha256: str
    rows_sha256: str
    result_coverage: str
    data_snapshot_fingerprint: str = ""
    datasource_fingerprint: str = ""
    datasource_environment: str = ""
    semantic_activation_fingerprint: str = ""
    cache_generation: str = ""


@dataclass(frozen=True)
class _VerifiedQueryInput:
    input_id: str
    artifact_fingerprint: str
    query_manifest_path: Path
    query_manifest_sha256: str
    rows_path: Path
    rows_sha256: str
    result_coverage: str
    result_is_truncated: bool
    stored_row_count: int
    exact_result_row_count: int
    population_use: str


@dataclass(frozen=True)
class _PreparedSandboxInputs:
    root: Path
    manifest_path: Path
    manifest_sha256: str


class SandboxArtifactValidationError(RuntimeError):
    """Fail-closed validation error without exposing host paths to a script."""

    def __init__(self, code: str):
        self.code = str(code or "SANDBOX_ARTIFACT_VALIDATION_FAILED")
        super().__init__(self.code)


class MerchantAnalysisSandbox:
    """Run reviewed merchant-analysis scripts without exposing a general shell."""

    _ALLOWLIST_KIND = "SANDBOX_INPUT_ALLOWLIST"
    _QUERY_RESULT_KIND = "GROUNDED_QUERY_RESULT"
    _VERIFIED_QUERY_RESULT_KINDS = {
        "GROUNDED_QUERY_RESULT",
        "GROUNDED_QUERY_RESULT_VERIFIED",
    }
    _POPULATION_USES = {
        "OBSERVATION",
        "COMPLETE_POPULATION",
        "RANKED_RESULT",
    }
    _KNOWN_COVERAGES = {item.value for item in ResultCoverage}

    def __init__(self, settings: Settings):
        self.settings = settings
        configured_resources_root = getattr(
            settings,
            "resources_root",
            None,
        )
        resources_root = (
            Path(configured_resources_root)
            if configured_resources_root
            else Path(__file__).resolve().parents[2] / "resources"
        )
        self.skill_root = (
            resources_root / "runtime" / "agent_skills"
        ).resolve()

    def run_python(
        self,
        script: Path,
        args: List[str],
        workspace: Path,
        timeout_seconds: int,
        *,
        artifact_access: SandboxArtifactAccess | None = None,
    ) -> SandboxResult:
        script_supplied = Path(script)
        if script_supplied.is_symlink():
            return SandboxResult(126, stderr="SANDBOX_SCRIPT_INVALID")
        script_path = script_supplied.resolve()
        if not self._is_within(script_path, self.skill_root):
            return SandboxResult(126, stderr="SANDBOX_SCRIPT_NOT_APPROVED")
        if not script_path.is_file() or script_path.is_symlink() or script_path.suffix != ".py":
            return SandboxResult(126, stderr="SANDBOX_SCRIPT_INVALID")

        workspace_supplied = Path(workspace)
        if workspace_supplied.is_symlink():
            return SandboxResult(126, stderr="SANDBOX_WORKSPACE_INVALID")
        try:
            workspace_supplied.mkdir(parents=True, exist_ok=True)
            workspace_path = workspace_supplied.resolve(strict=True)
        except OSError:
            return SandboxResult(126, stderr="SANDBOX_WORKSPACE_INVALID")
        if not workspace_path.is_dir():
            return SandboxResult(126, stderr="SANDBOX_WORKSPACE_INVALID")

        prepared_inputs: _PreparedSandboxInputs | None = None
        try:
            backend = str(
                getattr(self.settings, "sandbox_backend", "") or ""
            ).lower()
            container_backend = backend in {
                "container",
                "docker",
                "podman",
            }
            local_test_backend = (
                backend == "local"
                and bool(
                    getattr(
                        self.settings,
                        "sandbox_unsafe_local_test_mode",
                        False,
                    )
                )
            )
            if not container_backend and not local_test_backend:
                return SandboxResult(
                    126,
                    stderr="SANDBOX_HARDENED_BACKEND_REQUIRED",
                )
            if artifact_access is not None:
                self._validate_artifact_workspace_boundary(
                    workspace_path,
                    artifact_access,
                )
                prepared_inputs = self._prepare_artifact_inputs(artifact_access)
            argument_error = self._validate_arguments(
                args,
                workspace_path,
                artifact_access,
            )
            if argument_error:
                return SandboxResult(126, stderr=argument_error)

            if container_backend:
                return self._run_container(
                    script_path,
                    args,
                    workspace_path,
                    timeout_seconds,
                    prepared_inputs,
                )
            return self._run_local(
                script_path,
                args,
                workspace_path,
                timeout_seconds,
                prepared_inputs,
            )
        except SandboxArtifactValidationError as exc:
            return SandboxResult(126, stderr=exc.code)
        finally:
            if prepared_inputs is not None:
                self._remove_prepared_inputs(prepared_inputs.root)

    def _validate_arguments(
        self,
        args: List[str],
        workspace_path: Path,
        artifact_access: SandboxArtifactAccess | None,
    ) -> str:
        artifact_root = (
            Path(artifact_access.run_artifact_root).resolve()
            if artifact_access is not None
            else None
        )
        for value in args:
            candidate_value = str(value or "")
            if candidate_value.startswith("-"):
                if "=" not in candidate_value:
                    continue
                candidate_value = candidate_value.split("=", 1)[1]
            if candidate_value.startswith("file://"):
                candidate_value = candidate_value[len("file://") :]
            if not candidate_value:
                continue
            candidate = Path(candidate_value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (workspace_path / candidate).resolve()
            )
            if artifact_root is not None and self._is_within(
                resolved,
                artifact_root,
            ):
                return "SANDBOX_ARTIFACT_PATH_MUST_USE_ALLOWLIST"
            if not self._is_within(resolved, workspace_path):
                return "SANDBOX_PATH_OUTSIDE_WORKSPACE"
        return ""

    def _validate_artifact_workspace_boundary(
        self,
        workspace_path: Path,
        access: SandboxArtifactAccess,
    ) -> None:
        trusted_root = self._trusted_workspace_root(access)
        workspace_verified, workspace_descriptor = (
            self._open_directory_beneath(
                trusted_root,
                workspace_path,
                "SANDBOX_OUTPUT_OUTSIDE_RUNTIME_WORKSPACE",
            )
        )
        os.close(workspace_descriptor)
        artifact_root, artifact_descriptor = self._open_directory_beneath(
            trusted_root,
            access.run_artifact_root,
            "SANDBOX_ARTIFACT_ROOT_INVALID",
        )
        os.close(artifact_descriptor)
        staging_root, staging_descriptor = self._open_directory_beneath(
            trusted_root,
            access.sandbox_staging_root,
            "SANDBOX_STAGING_ROOT_INVALID",
        )
        os.close(staging_descriptor)
        if self._is_within(workspace_verified, artifact_root) or self._is_within(
            artifact_root,
            workspace_verified,
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_INPUT_OUTPUT_BOUNDARY_OVERLAP"
            )
        if (
            self._is_within(workspace_verified, staging_root)
            or self._is_within(staging_root, workspace_verified)
            or self._is_within(artifact_root, staging_root)
            or self._is_within(staging_root, artifact_root)
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_INPUT_OUTPUT_BOUNDARY_OVERLAP"
            )

    def _prepare_artifact_inputs(
        self,
        access: SandboxArtifactAccess,
    ) -> _PreparedSandboxInputs:
        trusted_root = self._trusted_workspace_root(access)
        artifact_root, artifact_descriptor = self._open_directory_beneath(
            trusted_root,
            access.run_artifact_root,
            "SANDBOX_ARTIFACT_ROOT_INVALID",
        )
        os.close(artifact_descriptor)
        owner_fingerprint = str(access.expected_owner_fingerprint or "").strip()
        semantic_fingerprint = str(
            access.expected_semantic_activation_fingerprint or ""
        ).strip()
        if not owner_fingerprint:
            raise SandboxArtifactValidationError(
                "SANDBOX_CONTEXT_OWNER_REQUIRED"
            )
        if not semantic_fingerprint:
            raise SandboxArtifactValidationError(
                "SANDBOX_SEMANTIC_ACTIVATION_REQUIRED"
            )
        verified_registry: dict[str, SandboxVerifiedArtifactCommit] = {}
        supplied_commits = tuple(access.verified_query_artifact_commits or ())
        for commit in supplied_commits:
            fingerprint = str(commit.artifact_fingerprint or "").strip()
            manifest_sha256 = str(commit.query_manifest_sha256 or "").strip()
            rows_sha256 = str(commit.rows_sha256 or "").strip()
            coverage = str(commit.result_coverage or "").strip()
            if (
                not self._valid_sha256(fingerprint)
                or not self._valid_sha256(manifest_sha256)
                or not self._valid_sha256(rows_sha256)
                or coverage not in self._KNOWN_COVERAGES
                or fingerprint in verified_registry
            ):
                raise SandboxArtifactValidationError(
                    "SANDBOX_VERIFIED_ARTIFACT_REGISTRY_INVALID"
                )
            verified_registry[fingerprint] = commit
        if not verified_registry:
            raise SandboxArtifactValidationError(
                "SANDBOX_VERIFIED_ARTIFACT_REGISTRY_INVALID"
            )

        allowlist_path = self._resolve_artifact_file(
            artifact_root,
            access.allowlist_manifest_path,
            allow_absolute=True,
        )
        allowlist = self._read_verified_json_object(
            allowlist_path,
            expected_sha256=access.allowlist_sha256,
            expected_content_address=access.allowlist_content_address,
            max_bytes=max(
                1024,
                int(
                    getattr(
                        self.settings,
                        "sandbox_allowlist_max_bytes",
                        1024 * 1024,
                    )
                    or 1024 * 1024
                ),
            ),
            error_code="SANDBOX_ALLOWLIST_INVALID",
        )
        if allowlist.get("schemaVersion") != SANDBOX_INPUT_ALLOWLIST_VERSION:
            raise SandboxArtifactValidationError(
                "SANDBOX_ALLOWLIST_VERSION_UNSUPPORTED"
            )
        if str(allowlist.get("manifestKind") or "") != self._ALLOWLIST_KIND:
            raise SandboxArtifactValidationError(
                "SANDBOX_ALLOWLIST_KIND_INVALID"
            )
        if str(allowlist.get("contextOwnerFingerprint") or "") != owner_fingerprint:
            raise SandboxArtifactValidationError(
                "SANDBOX_CONTEXT_OWNER_MISMATCH"
            )
        if (
            str(allowlist.get("semanticActivationFingerprint") or "")
            != semantic_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_SEMANTIC_ACTIVATION_MISMATCH"
            )

        raw_inputs = allowlist.get("inputs")
        max_inputs = max(
            1,
            int(getattr(self.settings, "sandbox_max_artifact_inputs", 32) or 32),
        )
        if (
            not isinstance(raw_inputs, list)
            or not raw_inputs
            or len(raw_inputs) > max_inputs
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_ALLOWLIST_INPUTS_INVALID"
            )

        verified_inputs: list[_VerifiedQueryInput] = []
        observed_ids: set[str] = set()
        total_bytes = 0
        max_total_bytes = max(
            1,
            int(
                getattr(
                    self.settings,
                    "sandbox_artifact_input_max_bytes",
                    512 * 1024 * 1024,
                )
                or 512 * 1024 * 1024
            ),
        )
        for raw_input in raw_inputs:
            verified = self._verify_query_input(
                artifact_root=artifact_root,
                raw_input=raw_input,
                owner_fingerprint=owner_fingerprint,
                semantic_fingerprint=semantic_fingerprint,
                verified_registry=verified_registry,
            )
            if verified.input_id in observed_ids:
                raise SandboxArtifactValidationError(
                    "SANDBOX_INPUT_ID_CONFLICT"
                )
            observed_ids.add(verified.input_id)
            try:
                total_bytes += verified.rows_path.stat().st_size
                total_bytes += verified.query_manifest_path.stat().st_size
            except OSError as exc:
                raise SandboxArtifactValidationError(
                    "SANDBOX_ARTIFACT_READ_FAILED"
                ) from exc
            if total_bytes > max_total_bytes:
                raise SandboxArtifactValidationError(
                    "SANDBOX_ARTIFACT_INPUT_BUDGET_EXCEEDED"
                )
            verified_inputs.append(verified)

        staging_parent, staging_parent_descriptor = (
            self._open_directory_beneath(
                trusted_root,
                access.sandbox_staging_root,
                "SANDBOX_STAGING_ROOT_INVALID",
            )
        )
        try:
            staging_root = self._create_staging_directory(
                staging_parent,
                staging_parent_descriptor,
            )
        finally:
            os.close(staging_parent_descriptor)
        try:
            runtime_inputs: list[dict[str, Any]] = []
            for verified in verified_inputs:
                input_root = staging_root / "inputs" / verified.input_id
                input_root.mkdir(parents=True, exist_ok=False)
                staged_query_manifest = input_root / "query.manifest.json"
                staged_rows = input_root / "rows.json"
                self._copy_verified_file(
                    verified.query_manifest_path,
                    staged_query_manifest,
                    verified.query_manifest_sha256,
                )
                self._copy_verified_file(
                    verified.rows_path,
                    staged_rows,
                    verified.rows_sha256,
                )
                complete_population = (
                    verified.result_coverage == ResultCoverage.ALL_ROWS.value
                    and not verified.result_is_truncated
                    and verified.exact_result_row_count
                    == verified.stored_row_count
                )
                runtime_inputs.append(
                    {
                        "inputId": verified.input_id,
                        "artifactFingerprint": verified.artifact_fingerprint,
                        "queryManifestPath": (
                            "inputs/%s/query.manifest.json" % verified.input_id
                        ),
                        "queryManifestSha256": verified.query_manifest_sha256,
                        "queryManifestContentAddress": (
                            "sha256:%s" % verified.query_manifest_sha256
                        ),
                        "rowsPath": "inputs/%s/rows.json" % verified.input_id,
                        "rowsSha256": verified.rows_sha256,
                        "rowsContentAddress": "sha256:%s" % verified.rows_sha256,
                        "resultCoverage": verified.result_coverage,
                        "resultIsTruncated": verified.result_is_truncated,
                        "storedRowCount": verified.stored_row_count,
                        "exactResultRowCount": verified.exact_result_row_count,
                        "populationUse": verified.population_use,
                        "completePopulation": complete_population,
                    }
                )
            runtime_manifest = {
                "schemaVersion": SANDBOX_VERIFIED_INPUT_VERSION,
                "manifestKind": "SANDBOX_VERIFIED_INPUTS",
                "contextOwnerFingerprint": owner_fingerprint,
                "semanticActivationFingerprint": semantic_fingerprint,
                "inputs": runtime_inputs,
            }
            manifest_path = staging_root / "manifest.json"
            manifest_bytes = json.dumps(
                runtime_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self._atomic_create_file(manifest_path, manifest_bytes)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self._make_staging_read_only(staging_root)
            return _PreparedSandboxInputs(
                root=staging_root,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )
        except Exception:
            self._remove_prepared_inputs(staging_root)
            raise

    def _verify_query_input(
        self,
        *,
        artifact_root: Path,
        raw_input: Any,
        owner_fingerprint: str,
        semantic_fingerprint: str,
        verified_registry: Mapping[str, SandboxVerifiedArtifactCommit],
    ) -> _VerifiedQueryInput:
        if not isinstance(raw_input, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_ALLOWLIST_INPUT_INVALID"
            )
        input_id = str(raw_input.get("inputId") or "").strip()
        if not self._valid_path_component(input_id):
            raise SandboxArtifactValidationError("SANDBOX_INPUT_ID_INVALID")
        artifact_fingerprint = str(
            raw_input.get("artifactFingerprint") or ""
        ).strip()
        verified_commit = verified_registry.get(artifact_fingerprint)
        if not self._valid_sha256(artifact_fingerprint) or verified_commit is None:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_ARTIFACT_NOT_VERIFIED"
            )
        query_ref = raw_input.get("queryManifest")
        if not isinstance(query_ref, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_MANIFEST_REF_INVALID"
            )
        query_manifest_path = self._resolve_artifact_file(
            artifact_root,
            Path(str(query_ref.get("relativePath") or "")),
            allow_absolute=False,
        )
        query_manifest_sha256 = str(query_ref.get("sha256") or "")
        if query_manifest_sha256 != verified_commit.query_manifest_sha256:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_MANIFEST_NOT_VERIFIED"
            )
        query_manifest = self._read_verified_json_object(
            query_manifest_path,
            expected_sha256=query_manifest_sha256,
            expected_content_address=str(
                query_ref.get("contentAddress") or ""
            ),
            max_bytes=max(
                1024,
                int(
                    getattr(
                        self.settings,
                        "sandbox_query_manifest_max_bytes",
                        4 * 1024 * 1024,
                    )
                    or 4 * 1024 * 1024
                ),
            ),
            error_code="SANDBOX_QUERY_MANIFEST_INVALID",
        )
        schema_version = query_manifest.get("schemaVersion")
        artifact_kind = str(query_manifest.get("artifactKind") or "")
        legacy_manifest = (
            schema_version == 1
            and artifact_kind == self._QUERY_RESULT_KIND
        )
        verified_manifest = (
            schema_version == 2
            and artifact_kind in self._VERIFIED_QUERY_RESULT_KINDS
        )
        if schema_version not in {1, 2}:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_MANIFEST_VERSION_UNSUPPORTED"
            )
        if not legacy_manifest and not verified_manifest:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_ARTIFACT_KIND_INVALID"
            )
        if (
            str(query_manifest.get("artifactFingerprint") or "")
            != artifact_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_ARTIFACT_FINGERPRINT_MISMATCH"
            )
        if (
            str(query_manifest.get("contextOwnerFingerprint") or "")
            != owner_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_OWNER_MISMATCH"
            )
        if (
            str(query_manifest.get("semanticActivationFingerprint") or "")
            != semantic_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_SEMANTIC_ACTIVATION_MISMATCH"
            )
        if verified_manifest:
            self._verify_published_query_manifest(
                artifact_root=artifact_root,
                raw_input=raw_input,
                query_manifest=query_manifest,
                artifact_fingerprint=artifact_fingerprint,
                semantic_fingerprint=semantic_fingerprint,
                verified_commit=verified_commit,
            )

        required_coverage = str(raw_input.get("requiredCoverage") or "")
        result_coverage = str(query_manifest.get("resultCoverage") or "")
        if (
            required_coverage not in self._KNOWN_COVERAGES
            or result_coverage not in self._KNOWN_COVERAGES
            or required_coverage != result_coverage
            or result_coverage != verified_commit.result_coverage
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_COVERAGE_MISMATCH"
            )
        population_use = str(raw_input.get("populationUse") or "")
        if population_use not in self._POPULATION_USES:
            raise SandboxArtifactValidationError(
                "SANDBOX_POPULATION_USE_INVALID"
            )

        result_is_truncated = query_manifest.get("resultIsTruncated")
        stored_row_count = query_manifest.get("storedRowCount")
        exact_result_row_count = query_manifest.get("exactResultRowCount")
        if (
            not isinstance(result_is_truncated, bool)
            or not self._nonnegative_integer(stored_row_count)
            or not self._nonnegative_integer(exact_result_row_count)
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_COVERAGE_RECEIPT_INVALID"
            )
        if population_use == "COMPLETE_POPULATION" and not (
            result_coverage == ResultCoverage.ALL_ROWS.value
            and not result_is_truncated
            and exact_result_row_count == stored_row_count
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_COMPLETE_POPULATION_NOT_PROVED"
            )
        if population_use == "RANKED_RESULT" and not (
            result_coverage == ResultCoverage.TOP_N.value
            and not result_is_truncated
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_RANKED_RESULT_NOT_PROVED"
            )

        rows_ref = query_manifest.get("rowsArtifact")
        if not isinstance(rows_ref, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_ROWS_ARTIFACT_REF_INVALID"
            )
        rows_path = self._resolve_artifact_file(
            artifact_root,
            Path(str(rows_ref.get("relativePath") or "")),
            allow_absolute=False,
        )
        rows_sha256 = str(rows_ref.get("sha256") or "")
        if rows_sha256 != verified_commit.rows_sha256:
            raise SandboxArtifactValidationError(
                "SANDBOX_ROWS_ARTIFACT_NOT_VERIFIED"
            )
        self._verify_immutable_file(
            rows_path,
            rows_sha256,
            str(rows_ref.get("contentAddress") or ""),
        )
        if legacy_manifest and (
            str(query_manifest.get("rowsSha256") or "") != rows_sha256
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_ROWS_DIGEST_BINDING_INVALID"
            )
        return _VerifiedQueryInput(
            input_id=input_id,
            artifact_fingerprint=artifact_fingerprint,
            query_manifest_path=query_manifest_path,
            query_manifest_sha256=query_manifest_sha256,
            rows_path=rows_path,
            rows_sha256=rows_sha256,
            result_coverage=result_coverage,
            result_is_truncated=result_is_truncated,
            stored_row_count=stored_row_count,
            exact_result_row_count=exact_result_row_count,
            population_use=population_use,
        )

    def _verify_published_query_manifest(
        self,
        *,
        artifact_root: Path,
        raw_input: Mapping[str, Any],
        query_manifest: Mapping[str, Any],
        artifact_fingerprint: str,
        semantic_fingerprint: str,
        verified_commit: SandboxVerifiedArtifactCommit,
    ) -> None:
        """Reopen every v2 publication binding before staging row bytes.

        SQL is verified here as an immutable child of the publication but is
        intentionally not copied into the sandbox input mount.
        """

        if str(query_manifest.get("publicationStatus") or "") != "VERIFIED":
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_PUBLICATION_NOT_VERIFIED"
            )
        receipt = raw_input.get("verifiedReceipt")
        if not isinstance(receipt, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_VERIFIED_RECEIPT_REQUIRED"
            )
        verified_evidence = query_manifest.get("verifiedEvidence")
        if (
            not isinstance(verified_evidence, Mapping)
            or verified_evidence.get("passed") is not True
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_VERIFIED_EVIDENCE_INVALID"
            )
        verified_evidence_sha256 = str(
            query_manifest.get("verifiedEvidenceSha256") or ""
        )
        if (
            not self._valid_sha256(verified_evidence_sha256)
            or str(receipt.get("verifiedEvidenceSha256") or "")
            != verified_evidence_sha256
            or self._stable_hash(verified_evidence)
            != verified_evidence_sha256
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_VERIFIED_EVIDENCE_BINDING_INVALID"
            )

        scalar_bindings = (
            ("contractFingerprint", "contractFingerprint"),
            ("sqlEvidenceFingerprint", "sqlEvidenceFingerprint"),
            ("executionGeneration", "executionGeneration"),
        )
        for receipt_key, manifest_key in scalar_bindings:
            if receipt.get(receipt_key) != query_manifest.get(manifest_key):
                raise SandboxArtifactValidationError(
                    "SANDBOX_QUERY_PUBLICATION_BINDING_MISMATCH"
                )
        generation = query_manifest.get("executionGeneration")
        if not self._nonnegative_integer(generation) or int(generation) <= 0:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_EXECUTION_GENERATION_INVALID"
            )
        for key in ("contractFingerprint", "sqlEvidenceFingerprint"):
            if not self._valid_sha256(query_manifest.get(key)):
                raise SandboxArtifactValidationError(
                    "SANDBOX_QUERY_PUBLICATION_DIGEST_INVALID"
                )

        attempt_id = str(query_manifest.get("executionAttemptId") or "")
        attempt_fingerprint = hashlib.sha256(
            attempt_id.encode("utf-8")
        ).hexdigest()
        if (
            not attempt_id
            or str(receipt.get("attemptFingerprint") or "")
            != attempt_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_EXECUTION_ATTEMPT_MISMATCH"
            )
        data_snapshot = query_manifest.get("dataSnapshot")
        if not isinstance(data_snapshot, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_DATA_SNAPSHOT_INVALID"
            )
        if (
            str(receipt.get("dataSnapshotFingerprint") or "")
            != self._stable_hash(data_snapshot)
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_DATA_SNAPSHOT_BINDING_INVALID"
            )
        committed_snapshot_fingerprint = str(
            verified_commit.data_snapshot_fingerprint or ""
        )
        if (
            not self._valid_sha256(committed_snapshot_fingerprint)
            or committed_snapshot_fingerprint
            != str(receipt.get("dataSnapshotFingerprint") or "")
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_DATA_SNAPSHOT_NOT_VERIFIED"
            )
        snapshot_bindings = (
            (
                "datasourceFingerprint",
                verified_commit.datasource_fingerprint,
            ),
            (
                "datasourceEnvironment",
                verified_commit.datasource_environment,
            ),
            (
                "semanticActivationFingerprint",
                verified_commit.semantic_activation_fingerprint,
            ),
            (
                "cacheGeneration",
                verified_commit.cache_generation,
            ),
        )
        for field_name, expected_value in snapshot_bindings:
            if field_name not in data_snapshot or str(
                data_snapshot.get(field_name) or ""
            ) != str(expected_value or ""):
                raise SandboxArtifactValidationError(
                    "SANDBOX_QUERY_DATA_SNAPSHOT_IDENTITY_MISMATCH"
                )
        snapshot_activation = str(
            data_snapshot.get("semanticActivationFingerprint") or ""
        )
        if snapshot_activation != semantic_fingerprint:
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_SNAPSHOT_ACTIVATION_MISMATCH"
            )

        sql_ref = query_manifest.get("sqlArtifact")
        if not isinstance(sql_ref, Mapping):
            raise SandboxArtifactValidationError(
                "SANDBOX_SQL_ARTIFACT_REF_INVALID"
            )
        sql_relative_path = str(sql_ref.get("relativePath") or "")
        sql_sha256 = str(sql_ref.get("sha256") or "")
        sql_content_address = str(sql_ref.get("contentAddress") or "")
        if (
            str(receipt.get("sqlRelativePath") or "")
            != sql_relative_path
            or str(receipt.get("sqlSha256") or "") != sql_sha256
            or str(receipt.get("sqlContentAddress") or "")
            != sql_content_address
            or str(query_manifest.get("sqlSha256") or "") != sql_sha256
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_SQL_ARTIFACT_BINDING_INVALID"
            )
        sql_path = self._resolve_artifact_file(
            artifact_root,
            Path(sql_relative_path),
            allow_absolute=False,
        )
        self._verify_immutable_file(
            sql_path,
            sql_sha256,
            sql_content_address,
        )
        if (
            str(query_manifest.get("artifactFingerprint") or "")
            != artifact_fingerprint
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_QUERY_ARTIFACT_FINGERPRINT_MISMATCH"
            )

    @staticmethod
    def _stable_hash(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _resolve_artifact_file(
        self,
        root: Path,
        value: Path,
        *,
        allow_absolute: bool,
    ) -> Path:
        raw = Path(value)
        if not str(raw):
            raise SandboxArtifactValidationError("SANDBOX_ARTIFACT_PATH_INVALID")
        if raw.is_absolute() and not allow_absolute:
            raise SandboxArtifactValidationError("SANDBOX_ARTIFACT_PATH_INVALID")
        candidate = raw if raw.is_absolute() else root / raw
        try:
            lexical_relative = candidate.relative_to(root)
        except ValueError as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_PATH_OUTSIDE_RUN"
            ) from exc
        cursor = root
        for component in lexical_relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise SandboxArtifactValidationError(
                    "SANDBOX_ARTIFACT_SYMLINK_REJECTED"
                )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_PATH_OUTSIDE_RUN"
            ) from exc
        if not resolved.is_file():
            raise SandboxArtifactValidationError("SANDBOX_ARTIFACT_PATH_INVALID")
        return resolved

    def _verify_immutable_file(
        self,
        path: Path,
        expected_sha256: str,
        expected_content_address: str,
    ) -> None:
        digest = str(expected_sha256 or "")
        self._verify_immutable_declaration(
            path,
            digest,
            expected_content_address,
        )
        observed_digest = self._hash_file(path)
        if observed_digest != digest:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_HASH_MISMATCH"
            )

    def _read_verified_json_object(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_content_address: str,
        max_bytes: int,
        error_code: str,
    ) -> dict[str, Any]:
        self._verify_immutable_declaration(
            path,
            expected_sha256,
            expected_content_address,
        )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        chunks: list[bytes] = []
        observed_bytes = 0
        digest = hashlib.sha256()
        try:
            descriptor = os.open(path, flags)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed_bytes += len(chunk)
                if observed_bytes > max_bytes:
                    raise SandboxArtifactValidationError(error_code)
                chunks.append(chunk)
                digest.update(chunk)
        except SandboxArtifactValidationError:
            raise
        except OSError as exc:
            raise SandboxArtifactValidationError(error_code) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_HASH_MISMATCH"
            )
        try:
            payload = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SandboxArtifactValidationError(error_code) from exc
        if not isinstance(payload, dict):
            raise SandboxArtifactValidationError(error_code)
        return payload

    def _verify_immutable_declaration(
        self,
        path: Path,
        expected_sha256: str,
        expected_content_address: str,
    ) -> None:
        digest = str(expected_sha256 or "")
        if not self._valid_sha256(digest):
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_DIGEST_INVALID"
            )
        if str(expected_content_address or "") != "sha256:%s" % digest:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_CONTENT_ADDRESS_INVALID"
            )
        marker_identity = hashlib.sha256(path.name.encode("utf-8")).hexdigest()
        marker_path = path.parent / (
            ".artifact-immutable-%s.sha256" % marker_identity
        )
        if (
            not marker_path.exists()
            or marker_path.is_symlink()
            or not marker_path.is_file()
        ):
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_NOT_IMMUTABLE"
            )
        try:
            marker_digest = marker_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_MARKER_INVALID"
            ) from exc
        if marker_digest != digest:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_MARKER_MISMATCH"
            )

    def _copy_verified_file(
        self,
        source: Path,
        destination: Path,
        expected_sha256: str,
    ) -> None:
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        source_descriptor = -1
        destination_descriptor = -1
        digest = hashlib.sha256()
        try:
            source_descriptor = os.open(source, source_flags)
            destination_descriptor = os.open(destination, destination_flags, 0o400)
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                offset = 0
                while offset < len(chunk):
                    offset += os.write(destination_descriptor, chunk[offset:])
            os.fsync(destination_descriptor)
        except OSError as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_STAGE_FAILED"
            ) from exc
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
        if digest.hexdigest() != expected_sha256:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_HASH_CHANGED_DURING_STAGE"
            )

    def _run_local(
        self,
        script_path: Path,
        args: List[str],
        workspace_path: Path,
        timeout_seconds: int,
        prepared_inputs: _PreparedSandboxInputs | None,
    ) -> SandboxResult:
        command = [self.settings.python_executable, "-I", str(script_path), *args]
        env = self._sandbox_environment()
        if prepared_inputs is not None:
            env["MERCHANT_ANALYSIS_INPUT_MANIFEST"] = str(
                prepared_inputs.manifest_path
            )
            env["MERCHANT_ANALYSIS_INPUT_MANIFEST_SHA256"] = (
                prepared_inputs.manifest_sha256
            )
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_path),
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds or 1)),
                preexec_fn=self._local_resource_limiter(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                124,
                stdout=self._redact_host_paths(
                    str(exc.stdout or "")[-4000:],
                    script_path,
                    workspace_path,
                    prepared_inputs.root if prepared_inputs else None,
                ),
                stderr="SANDBOX_TIMEOUT",
            )
        except Exception as exc:
            return SandboxResult(
                125,
                stderr=self._redact_host_paths(
                    "SANDBOX_ERROR: %s" % str(exc)[:500],
                    script_path,
                    workspace_path,
                    prepared_inputs.root if prepared_inputs else None,
                ),
            )
        return SandboxResult(
            completed.returncode,
            stdout=self._redact_host_paths(
                str(completed.stdout or "")[-8000:],
                script_path,
                workspace_path,
                prepared_inputs.root if prepared_inputs else None,
            ),
            stderr=self._redact_host_paths(
                str(completed.stderr or "")[-8000:],
                script_path,
                workspace_path,
                prepared_inputs.root if prepared_inputs else None,
            ),
        )

    def _run_container(
        self,
        script_path: Path,
        args: List[str],
        workspace_path: Path,
        timeout_seconds: int,
        prepared_inputs: _PreparedSandboxInputs | None,
    ) -> SandboxResult:
        runtime = str(
            getattr(self.settings, "sandbox_container_runtime", "docker")
            or "docker"
        )
        if not shutil.which(runtime):
            return SandboxResult(
                125,
                stderr="SANDBOX_CONTAINER_RUNTIME_UNAVAILABLE",
            )
        relative_script = script_path.relative_to(self.skill_root)
        mapped_args = [self._container_arg(value, workspace_path) for value in args]
        memory = str(
            getattr(self.settings, "sandbox_container_memory", "512m") or "512m"
        )
        pids_limit = max(
            1,
            int(getattr(self.settings, "sandbox_container_pids", 128) or 128),
        )
        container_name = "merchant-analysis-%s" % hashlib.sha256(
            os.urandom(32)
        ).hexdigest()[:24]
        command = [
            runtime,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--ipc",
            "none",
            "--read-only",
            "--pids-limit",
            str(pids_limit),
            "--cpus",
            str(
                max(
                    0.1,
                    float(
                        getattr(self.settings, "sandbox_container_cpus", 1.0)
                        or 1.0
                    ),
                )
            ),
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--ulimit",
            "nofile=128:128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "-v",
            "%s:/opt/skills:ro" % self.skill_root,
            "-v",
            "%s:/output:rw" % workspace_path,
        ]
        for input_file in self._readonly_workspace_inputs(
            args,
            workspace_path,
        ):
            command.extend(
                [
                    "-v",
                    "%s:/output/%s:ro"
                    % (
                        input_file,
                        input_file.relative_to(workspace_path).as_posix(),
                    ),
                ]
            )
        if prepared_inputs is not None:
            command.extend(
                [
                    "-v",
                    "%s:/input:ro" % prepared_inputs.root,
                    "-e",
                    "MERCHANT_ANALYSIS_INPUT_MANIFEST=/input/manifest.json",
                    "-e",
                    "MERCHANT_ANALYSIS_INPUT_MANIFEST_SHA256=%s"
                    % prepared_inputs.manifest_sha256,
                ]
            )
        command.extend(
            [
                "-w",
                "/output",
                str(
                    getattr(
                        self.settings,
                        "sandbox_container_image",
                        "python:3.11-slim-bookworm",
                    )
                    or "python:3.11-slim-bookworm"
                ),
                "python",
                "-I",
                "/opt/skills/%s" % relative_script.as_posix(),
                *mapped_args,
            ]
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_path),
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds or 1)),
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_container(runtime, container_name)
            return SandboxResult(
                124,
                stdout=self._redact_host_paths(
                    str(exc.stdout or "")[-4000:],
                    script_path,
                    workspace_path,
                    prepared_inputs.root if prepared_inputs else None,
                ),
                stderr="SANDBOX_TIMEOUT",
            )
        except Exception as exc:
            return SandboxResult(
                125,
                stderr=self._redact_host_paths(
                    "SANDBOX_CONTAINER_ERROR: %s" % str(exc)[:500],
                    script_path,
                    workspace_path,
                    prepared_inputs.root if prepared_inputs else None,
                ),
            )
        return SandboxResult(
            completed.returncode,
            self._redact_host_paths(
                str(completed.stdout or "")[-8000:],
                script_path,
                workspace_path,
                prepared_inputs.root if prepared_inputs else None,
            ),
            self._redact_host_paths(
                str(completed.stderr or "")[-8000:],
                script_path,
                workspace_path,
                prepared_inputs.root if prepared_inputs else None,
            ),
        )

    def _redact_host_paths(
        self,
        value: str,
        *paths: Path | None,
    ) -> str:
        redacted = str(value or "")
        candidates = {
            str(path.resolve())
            for path in paths
            if path is not None
        }
        candidates.add(str(self.skill_root.resolve()))
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate:
                redacted = redacted.replace(
                    candidate,
                    "[SANDBOX_PATH_REDACTED]",
                )
        return redacted

    @staticmethod
    def _terminate_container(runtime: str, container_name: str) -> None:
        try:
            subprocess.run(
                [runtime, "rm", "--force", container_name],
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return

    def _container_arg(self, value: str, workspace_path: Path) -> str:
        candidate = Path(value)
        if candidate.is_absolute() and self._is_within(
            candidate.resolve(),
            workspace_path,
        ):
            return "/output/%s" % candidate.resolve().relative_to(
                workspace_path
            ).as_posix()
        return value

    def _readonly_workspace_inputs(
        self,
        args: List[str],
        workspace_path: Path,
    ) -> list[Path]:
        values: list[str] = []
        for index, raw_value in enumerate(args):
            value = str(raw_value or "")
            if index > 0 and str(args[index - 1] or "") == "--input":
                values.append(value)
            elif value.startswith("--input="):
                values.append(value.split("=", 1)[1])
        inputs: list[Path] = []
        for value in values:
            candidate = Path(value)
            target = (
                candidate
                if candidate.is_absolute()
                else workspace_path / candidate
            )
            try:
                resolved = target.resolve(strict=True)
            except OSError:
                continue
            if (
                resolved.is_file()
                and self._is_within(resolved, workspace_path)
                and resolved not in inputs
            ):
                inputs.append(resolved)
        return inputs

    def _sandbox_environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONIOENCODING": "utf-8",
            "MERCHANT_ANALYSIS_SANDBOX": "1",
        }

    def _local_resource_limiter(self, timeout_seconds: int):
        try:
            import resource
        except ImportError:
            return None

        cpu_seconds = max(1, int(math.ceil(float(timeout_seconds or 1))))
        memory_bytes = self._memory_bytes(
            str(
                getattr(self.settings, "sandbox_container_memory", "512m")
                or "512m"
            )
        )
        pids_limit = max(
            1,
            int(getattr(self.settings, "sandbox_container_pids", 128) or 128),
        )

        def apply_limits() -> None:
            try:
                os.setsid()
            except OSError:
                pass
            self._set_local_resource_limit(
                resource,
                resource.RLIMIT_CPU,
                cpu_seconds,
                cpu_seconds + 1,
            )
            if memory_bytes > 0 and hasattr(resource, "RLIMIT_AS"):
                self._set_local_resource_limit(
                    resource,
                    resource.RLIMIT_AS,
                    memory_bytes,
                    memory_bytes,
                )
            if hasattr(resource, "RLIMIT_NPROC"):
                self._set_local_resource_limit(
                    resource,
                    resource.RLIMIT_NPROC,
                    pids_limit,
                    pids_limit,
                )
            self._set_local_resource_limit(
                resource,
                resource.RLIMIT_NOFILE,
                128,
                128,
            )

        return apply_limits

    @staticmethod
    def _set_local_resource_limit(
        resource_module: Any,
        resource_kind: int,
        requested_soft: int,
        requested_hard: int,
    ) -> None:
        try:
            _, observed_hard = resource_module.getrlimit(resource_kind)
            infinity = resource_module.RLIM_INFINITY
            hard = (
                requested_hard
                if observed_hard == infinity
                else min(requested_hard, observed_hard)
            )
            soft = min(requested_soft, hard)
            resource_module.setrlimit(resource_kind, (soft, hard))
        except (OSError, ValueError):
            # Local execution is a developer fallback, not the production
            # containment boundary.  Artifact path and digest validation stay
            # mandatory even when a host does not expose one resource limit.
            return

    @staticmethod
    def _memory_bytes(value: str) -> int:
        text = str(value or "").strip().lower()
        if not text:
            return 0
        suffix = text[-1]
        multipliers = {
            "k": 1024,
            "m": 1024 * 1024,
            "g": 1024 * 1024 * 1024,
        }
        if suffix in multipliers:
            number = text[:-1]
            multiplier = multipliers[suffix]
        else:
            number = text
            multiplier = 1
        if not number or any(character not in "0123456789." for character in number):
            return 0
        try:
            return max(0, int(float(number) * multiplier))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _read_json_object(
        path: Path,
        *,
        max_bytes: int,
        error_code: str,
    ) -> dict[str, Any]:
        try:
            if path.stat().st_size > max_bytes:
                raise SandboxArtifactValidationError(error_code)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except SandboxArtifactValidationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SandboxArtifactValidationError(error_code) from exc
        if not isinstance(payload, dict):
            raise SandboxArtifactValidationError(error_code)
        return payload

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(path, flags)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        except OSError as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_ARTIFACT_READ_FAILED"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return digest.hexdigest()

    @staticmethod
    def _atomic_create_file(path: Path, content: bytes) -> None:
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_INPUT_MANIFEST_WRITE_FAILED"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _make_staging_read_only(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                path.chmod(0o500 if path.is_dir() else 0o400)
            except OSError as exc:
                raise SandboxArtifactValidationError(
                    "SANDBOX_ARTIFACT_STAGE_PERMISSION_FAILED"
                ) from exc
        root.chmod(0o500)

    @staticmethod
    def _remove_prepared_inputs(root: Path) -> None:
        try:
            for path in root.rglob("*"):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            root.chmod(0o700)
            shutil.rmtree(root, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _trusted_workspace_root(access: SandboxArtifactAccess) -> Path:
        supplied = access.trusted_workspace_root
        if supplied is None:
            raise SandboxArtifactValidationError(
                "SANDBOX_TRUSTED_WORKSPACE_ROOT_REQUIRED"
            )
        raw = Path(supplied)
        if raw.is_symlink() or not raw.is_absolute():
            raise SandboxArtifactValidationError(
                "SANDBOX_TRUSTED_WORKSPACE_ROOT_INVALID"
            )
        try:
            descriptor = os.open(
                raw,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise SandboxArtifactValidationError(
                        "SANDBOX_TRUSTED_WORKSPACE_ROOT_INVALID"
                    )
            finally:
                os.close(descriptor)
        except SandboxArtifactValidationError:
            raise
        except OSError as exc:
            raise SandboxArtifactValidationError(
                "SANDBOX_TRUSTED_WORKSPACE_ROOT_INVALID"
            ) from exc
        return Path(os.path.abspath(str(raw)))

    @classmethod
    def _open_directory_beneath(
        cls,
        trusted_root: Path,
        supplied_path: Path | None,
        error_code: str,
    ) -> tuple[Path, int]:
        if supplied_path is None:
            raise SandboxArtifactValidationError(error_code)
        raw = Path(supplied_path)
        if not raw.is_absolute():
            raise SandboxArtifactValidationError(error_code)
        lexical = Path(os.path.abspath(str(raw)))
        try:
            components = lexical.relative_to(trusted_root).parts
        except ValueError as exc:
            raise SandboxArtifactValidationError(error_code) from exc
        descriptor = -1
        try:
            descriptor = os.open(
                trusted_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            for component in components:
                if not cls._valid_path_component(component):
                    raise SandboxArtifactValidationError(error_code)
                child_descriptor = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                    os.close(child_descriptor)
                    raise SandboxArtifactValidationError(error_code)
                os.close(descriptor)
                descriptor = child_descriptor
            return trusted_root.joinpath(*components), descriptor
        except SandboxArtifactValidationError:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise SandboxArtifactValidationError(error_code) from exc

    @staticmethod
    def _create_staging_directory(
        parent: Path,
        parent_descriptor: int,
    ) -> Path:
        for _attempt in range(8):
            name = "verified_%s" % hashlib.sha256(
                os.urandom(32)
            ).hexdigest()[:24]
            try:
                os.mkdir(name, 0o700, dir_fd=parent_descriptor)
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor,
                )
                try:
                    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                        raise SandboxArtifactValidationError(
                            "SANDBOX_ARTIFACT_STAGE_FAILED"
                        )
                finally:
                    os.close(descriptor)
                return parent / name
            except FileExistsError:
                continue
            except SandboxArtifactValidationError:
                raise
            except OSError as exc:
                raise SandboxArtifactValidationError(
                    "SANDBOX_ARTIFACT_STAGE_FAILED"
                ) from exc
        raise SandboxArtifactValidationError(
            "SANDBOX_ARTIFACT_STAGE_FAILED"
        )

    @staticmethod
    def _valid_path_component(value: str) -> bool:
        text = str(value or "")
        return bool(text) and len(text) <= 128 and all(
            character.isascii()
            and (character.isalnum() or character in {"_", "-", "."})
            for character in text
        ) and text not in {".", ".."}

    @staticmethod
    def _valid_sha256(value: str) -> bool:
        text = str(value or "")
        return len(text) == 64 and all(
            character in "0123456789abcdef" for character in text
        )

    @staticmethod
    def _nonnegative_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
