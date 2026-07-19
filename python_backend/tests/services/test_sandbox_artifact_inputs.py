from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import merchant_ai.services.sandbox as sandbox_module
from merchant_ai.config import Settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.sandbox import (
    MerchantAnalysisSandbox,
    SandboxArtifactAccess,
    SandboxVerifiedArtifactCommit,
)


def _settings(tmp_path: Path, *, backend: str = "local") -> Settings:
    return Settings(
        harness_workspace_path=str(tmp_path / "runtime"),
        sandbox_backend=backend,
        sandbox_unsafe_local_test_mode=True,
        sandbox_container_runtime="docker",
    )


def _approved_sandbox(
    tmp_path: Path,
    *,
    backend: str = "local",
) -> tuple[MerchantAnalysisSandbox, Path, Settings]:
    settings = _settings(tmp_path, backend=backend)
    sandbox = MerchantAnalysisSandbox(settings)
    skill_root = tmp_path / "approved_skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    script = skill_root / "analyze.py"
    script.write_text(
        """
import hashlib
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["MERCHANT_ANALYSIS_INPUT_MANIFEST"])
manifest_bytes = manifest_path.read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == os.environ["MERCHANT_ANALYSIS_INPUT_MANIFEST_SHA256"]
manifest = json.loads(manifest_bytes)
item = manifest["inputs"][0]
rows = json.loads((manifest_path.parent / item["rowsPath"]).read_text(encoding="utf-8"))
output = {
    "coverage": item["resultCoverage"],
    "complete": item["completePopulation"],
    "rows": rows,
}
print(json.dumps(output, ensure_ascii=False))
""".strip(),
        encoding="utf-8",
    )
    sandbox.skill_root = skill_root.resolve()
    return sandbox, script, settings


def _artifact_access(
    settings: Settings,
    *,
    coverage: str,
    population_use: str,
    truncated: bool | None = None,
    owner: str = "owner-fingerprint",
    semantic: str = "semantic-activation-v1",
) -> tuple[SandboxArtifactAccess, Path, dict]:
    artifact_root = (
        settings.resolved_workspace_path
        / "threads"
        / "thread-1"
        / "runs"
        / "run-1"
        / "outputs"
        / "artifacts"
    )
    store = WorkspaceArtifactStore(settings, artifact_root)
    rows = [{"entity": "a", "value": 7}, {"entity": "b", "value": 3}]
    rows_artifact = store.write_json(
        "query_results",
        "rows.json",
        rows,
        preview_chars=0,
        immutable=True,
    )
    is_truncated = coverage == "PREVIEW" if truncated is None else truncated
    exact_count = len(rows) if coverage == "ALL_ROWS" and not is_truncated else 0
    artifact_fingerprint = hashlib.sha256(
        ("query-artifact:%s:%s" % (coverage, population_use)).encode("utf-8")
    ).hexdigest()
    query_manifest_payload = {
        "schemaVersion": 1,
        "artifactKind": "GROUNDED_QUERY_RESULT",
        "artifactFingerprint": artifact_fingerprint,
        "contextOwnerFingerprint": owner,
        "semanticActivationFingerprint": semantic,
        "resultCoverage": coverage,
        "resultIsTruncated": is_truncated,
        "storedRowCount": len(rows),
        "exactResultRowCount": exact_count,
        "rowsArtifact": {
            "relativePath": rows_artifact["relativePath"],
            "sha256": rows_artifact["sha256"],
            "contentAddress": rows_artifact["contentAddress"],
            "bytes": rows_artifact["bytes"],
        },
        "rowsSha256": rows_artifact["sha256"],
    }
    query_manifest = store.write_json(
        "query_results",
        "query.manifest.json",
        query_manifest_payload,
        preview_chars=0,
        immutable=True,
    )
    allowlist_payload = {
        "schemaVersion": 1,
        "manifestKind": "SANDBOX_INPUT_ALLOWLIST",
        "contextOwnerFingerprint": owner,
        "semanticActivationFingerprint": semantic,
        "inputs": [
            {
                "inputId": "query_1",
                "artifactFingerprint": artifact_fingerprint,
                "queryManifest": {
                    "relativePath": query_manifest["relativePath"],
                    "sha256": query_manifest["sha256"],
                    "contentAddress": query_manifest["contentAddress"],
                },
                "requiredCoverage": coverage,
                "populationUse": population_use,
            }
        ],
    }
    allowlist = store.write_json(
        "sandbox_inputs",
        "allowlist.json",
        allowlist_payload,
        preview_chars=0,
        immutable=True,
    )
    sandbox_staging_root = (
        settings.resolved_workspace_path / ".sandbox_inputs"
    )
    sandbox_staging_root.mkdir(parents=True, exist_ok=True)
    access = SandboxArtifactAccess(
        run_artifact_root=artifact_root,
        allowlist_manifest_path=Path(allowlist["path"]),
        allowlist_sha256=allowlist["sha256"],
        allowlist_content_address=allowlist["contentAddress"],
        expected_owner_fingerprint=owner,
        expected_semantic_activation_fingerprint=semantic,
        verified_query_artifact_commits=(
            SandboxVerifiedArtifactCommit(
                artifact_fingerprint=artifact_fingerprint,
                query_manifest_sha256=query_manifest["sha256"],
                rows_sha256=rows_artifact["sha256"],
                result_coverage=coverage,
            ),
        ),
        trusted_workspace_root=settings.resolved_workspace_path.resolve(),
        sandbox_staging_root=sandbox_staging_root.resolve(),
    )
    return access, Path(rows_artifact["path"]), {
        "rows": rows_artifact,
        "queryManifest": query_manifest,
        "allowlist": allowlist,
    }


def test_local_backend_consumes_only_staged_allowlisted_artifact_and_preserves_preview(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, source_rows_path, _ = _artifact_access(
        settings,
        coverage="PREVIEW",
        population_use="OBSERVATION",
    )
    output = settings.resolved_workspace_path / "scratch" / "analysis-1"

    result = sandbox.run_python(
        script,
        [],
        output,
        5,
        artifact_access=access,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "coverage": "PREVIEW",
        "complete": False,
        "rows": [{"entity": "a", "value": 7}, {"entity": "b", "value": 3}],
    }
    assert source_rows_path.is_file()
    assert list((settings.resolved_workspace_path / ".sandbox_inputs").iterdir()) == []


def test_preview_cannot_be_declared_as_complete_population(tmp_path: Path) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="PREVIEW",
        population_use="COMPLETE_POPULATION",
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_COMPLETE_POPULATION_NOT_PROVED"


@pytest.mark.parametrize(
    ("access_change", "expected_error"),
    [
        (
            {"expected_owner_fingerprint": "another-owner"},
            "SANDBOX_CONTEXT_OWNER_MISMATCH",
        ),
        (
            {"expected_semantic_activation_fingerprint": "semantic-v2"},
            "SANDBOX_SEMANTIC_ACTIVATION_MISMATCH",
        ),
    ],
)
def test_server_identity_must_match_allowlist_and_query_artifact(
    tmp_path: Path,
    access_change: dict,
    expected_error: str,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    changed = SandboxArtifactAccess(
        **{
            **access.__dict__,
            **access_change,
        }
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=changed,
    )

    assert result.returncode == 126
    assert result.stderr == expected_error


def test_local_backend_rechecks_immutable_marker_and_full_sha256(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, rows_path, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    rows_path.write_text('[{"tampered":true}]', encoding="utf-8")

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_ARTIFACT_HASH_MISMATCH"


def test_allowlist_must_also_be_committed_in_server_verified_registry(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    changed_access = SandboxArtifactAccess(
        **{
            **access.__dict__,
            "verified_query_artifact_commits": (
                SandboxVerifiedArtifactCommit(
                    artifact_fingerprint=hashlib.sha256(
                        b"another-artifact"
                    ).hexdigest(),
                    query_manifest_sha256=(
                        access.verified_query_artifact_commits[
                            0
                        ].query_manifest_sha256
                    ),
                    rows_sha256=access.verified_query_artifact_commits[
                        0
                    ].rows_sha256,
                    result_coverage="ALL_ROWS",
                ),
            ),
        }
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=changed_access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_QUERY_ARTIFACT_NOT_VERIFIED"


@pytest.mark.parametrize(
    ("commit_field", "expected_error"),
    [
        ("query_manifest_sha256", "SANDBOX_QUERY_MANIFEST_NOT_VERIFIED"),
        ("rows_sha256", "SANDBOX_ROWS_ARTIFACT_NOT_VERIFIED"),
    ],
)
def test_verified_commit_binds_exact_manifest_and_rows_content(
    tmp_path: Path,
    commit_field: str,
    expected_error: str,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    original_commit = access.verified_query_artifact_commits[0]
    commit_values = {
        **original_commit.__dict__,
        commit_field: hashlib.sha256(b"uncommitted-content").hexdigest(),
    }
    changed_access = SandboxArtifactAccess(
        **{
            **access.__dict__,
            "verified_query_artifact_commits": (
                SandboxVerifiedArtifactCommit(**commit_values),
            ),
        }
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=changed_access,
    )

    assert result.returncode == 126
    assert result.stderr == expected_error


def test_unlisted_source_path_cannot_be_passed_as_script_argument(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, rows_path, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )

    result = sandbox.run_python(
        script,
        [str(rows_path)],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_ARTIFACT_PATH_MUST_USE_ALLOWLIST"


@pytest.mark.parametrize("argument_form", ["flag", "relative"])
def test_artifact_path_cannot_hide_in_flag_or_relative_argument(
    tmp_path: Path,
    argument_form: str,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, rows_path, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    output = settings.resolved_workspace_path / "scratch" / "analysis-1"
    relative = os.path.relpath(rows_path, output)
    argument = "--input=%s" % rows_path if argument_form == "flag" else relative

    result = sandbox.run_python(
        script,
        [argument],
        output,
        5,
        artifact_access=access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_ARTIFACT_PATH_MUST_USE_ALLOWLIST"


def test_allowlist_cannot_reference_path_outside_current_run_root(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, artifacts = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    original_allowlist = json.loads(
        Path(artifacts["allowlist"]["path"]).read_text(encoding="utf-8")
    )
    original_allowlist["inputs"][0]["queryManifest"]["relativePath"] = (
        "../../outside.manifest.json"
    )
    store = WorkspaceArtifactStore(settings, access.run_artifact_root)
    changed_allowlist = store.write_json(
        "sandbox_inputs",
        "escape_allowlist.json",
        original_allowlist,
        preview_chars=0,
        immutable=True,
    )
    changed_access = SandboxArtifactAccess(
        **{
            **access.__dict__,
            "allowlist_manifest_path": Path(changed_allowlist["path"]),
            "allowlist_sha256": changed_allowlist["sha256"],
            "allowlist_content_address": changed_allowlist["contentAddress"],
        }
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=changed_access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_ARTIFACT_PATH_OUTSIDE_RUN"


def test_container_mounts_verified_inputs_read_only_and_output_separately(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path, backend="container")
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(sandbox_module.shutil, "which", lambda _runtime: "/bin/docker")

    def completed(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = dict(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", completed)
    output = settings.resolved_workspace_path / "scratch" / "analysis-1"

    result = sandbox.run_python(
        script,
        [],
        output,
        7,
        artifact_access=access,
    )

    assert result.returncode == 0
    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--ipc") + 1] == "none"
    assert "--read-only" in command
    assert "--pids-limit" in command
    assert "--cpus" in command
    assert "--memory" in command
    assert "--memory-swap" in command
    assert "--ulimit" in command
    mounts = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "-v"
    ]
    assert any(str(item).endswith(":/input:ro") for item in mounts)
    assert "%s:/output:rw" % output.resolve() in mounts
    assert not any(
        str(access.run_artifact_root.resolve()) in str(item)
        for item in mounts
    )
    assert "MERCHANT_ANALYSIS_INPUT_MANIFEST=/input/manifest.json" in command
    assert observed["kwargs"]["timeout"] == 7


def test_container_timeout_forcibly_removes_named_container(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path, backend="container")
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(sandbox_module.shutil, "which", lambda _runtime: "/bin/docker")

    def timeout_then_cleanup(command, **kwargs):
        calls.append(list(command))
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", timeout_then_cleanup)

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        2,
        artifact_access=access,
    )

    assert result.returncode == 124
    assert result.stderr == "SANDBOX_TIMEOUT"
    container_name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "--force", container_name]


def test_allowlist_content_address_is_required_even_when_sha_matches(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    changed_access = SandboxArtifactAccess(
        **{
            **access.__dict__,
            "allowlist_content_address": "sha256:%s"
            % hashlib.sha256(b"different").hexdigest(),
        }
    )

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-1",
        5,
        artifact_access=changed_access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_ARTIFACT_CONTENT_ADDRESS_INVALID"


def test_artifact_staging_parent_symlink_is_rejected_without_external_write(
    tmp_path: Path,
) -> None:
    sandbox, script, settings = _approved_sandbox(tmp_path)
    access, _, _ = _artifact_access(
        settings,
        coverage="ALL_ROWS",
        population_use="COMPLETE_POPULATION",
    )
    staging_root = Path(access.sandbox_staging_root)
    staging_root.rmdir()
    outside = tmp_path / "outside-staging"
    outside.mkdir()
    staging_root.symlink_to(outside, target_is_directory=True)

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis-link",
        5,
        artifact_access=access,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_STAGING_ROOT_INVALID"
    assert list(outside.iterdir()) == []


def test_local_backend_requires_explicit_test_only_capability(
    tmp_path: Path,
) -> None:
    settings = Settings(
        harness_workspace_path=str(tmp_path / "runtime"),
        sandbox_backend="local",
        sandbox_unsafe_local_test_mode=False,
    )
    sandbox = MerchantAnalysisSandbox(settings)
    skill_root = tmp_path / "approved"
    skill_root.mkdir()
    script = skill_root / "safe.py"
    script.write_text("print('ok')", encoding="utf-8")
    sandbox.skill_root = skill_root.resolve()

    result = sandbox.run_python(
        script,
        [],
        settings.resolved_workspace_path / "scratch" / "analysis",
        5,
    )

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_HARDENED_BACKEND_REQUIRED"


def test_local_test_backend_redacts_host_paths_from_output(
    tmp_path: Path,
) -> None:
    sandbox, _, settings = _approved_sandbox(tmp_path)
    script = sandbox.skill_root / "print_path.py"
    script.write_text(
        "from pathlib import Path\nprint(Path.cwd())",
        encoding="utf-8",
    )
    output = settings.resolved_workspace_path / "scratch" / "path-output"

    result = sandbox.run_python(script, [], output, 5)

    assert result.returncode == 0
    assert str(output.resolve()) not in result.stdout
    assert "[SANDBOX_PATH_REDACTED]" in result.stdout
