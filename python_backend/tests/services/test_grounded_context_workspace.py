from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentSession,
    GroundedRunFilesystemBackend,
    GroundedSemanticBackend,
)
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
    GroundedContextWorkspaceError,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession


def _settings(root: Path) -> Settings:
    return Settings(harness_workspace_path=str(root))


def test_context_workspace_is_run_scoped_and_identity_bound(tmp_path: Path) -> None:
    workspace = GroundedContextWorkspace.open(
        _settings(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={
            "tenantId": "tenant-1",
            "userId": "user-1",
            "storeIds": ["store-2", "store-1"],
            "permissions": ["export", "read"],
        },
        question="question",
    )

    assert workspace.root.is_relative_to(tmp_path.resolve())
    assert workspace.artifacts_root.is_dir()
    assert workspace.core_scratch_root.is_dir()
    assert workspace.subagents_root.is_dir()
    manifest = json.loads(
        (workspace.root / "context_workspace.json").read_text(encoding="utf-8")
    )
    assert manifest["ownerFingerprint"] == workspace.owner_fingerprint
    assert "merchant-1" not in json.dumps(manifest)
    assert "user-1" not in json.dumps(manifest)


def test_context_workspace_replay_is_idempotent_but_request_change_conflicts(
    tmp_path: Path,
) -> None:
    arguments = {
        "thread_id": "thread-1",
        "run_id": "run-1",
        "merchant_id": "merchant-1",
        "access_role": "merchant_analyst",
        "user_scope": {"tenantId": "tenant-1", "userId": "user-1"},
        "question": "same question",
    }
    first = GroundedContextWorkspace.open(_settings(tmp_path), **arguments)
    replay = GroundedContextWorkspace.open(_settings(tmp_path), **arguments)

    assert replay.request_fingerprint == first.request_fingerprint
    with pytest.raises(
        GroundedContextWorkspaceError,
        match="GROUNDED_CONTEXT_REQUEST_CONFLICT",
    ):
        GroundedContextWorkspace.open(
            _settings(tmp_path),
            **{**arguments, "question": "different question"},
        )
    with pytest.raises(
        GroundedContextWorkspaceError,
        match="GROUNDED_CONTEXT_REQUEST_CONFLICT",
    ):
        GroundedContextWorkspace.open(
            _settings(tmp_path),
            **{
                **arguments,
                "user_scope": {
                    **arguments["user_scope"],
                    "rowPolicy": {"version": "changed"},
                },
            },
        )


def test_context_workspace_manifest_create_is_concurrency_safe(tmp_path: Path) -> None:
    def open_workspace(_: int) -> str:
        return GroundedContextWorkspace.open(
            _settings(tmp_path),
            thread_id="thread-shared",
            run_id="run-shared",
            merchant_id="merchant-1",
            access_role="merchant_analyst",
            user_scope={"userId": "user-1"},
            question="question",
        ).request_fingerprint

    with ThreadPoolExecutor(max_workers=8) as pool:
        fingerprints = list(pool.map(open_workspace, range(24)))

    assert len(set(fingerprints)) == 1


def test_context_workspace_hashes_unsafe_path_identifiers(tmp_path: Path) -> None:
    workspace = GroundedContextWorkspace.open(
        _settings(tmp_path),
        thread_id="../../outside",
        run_id="/absolute/run",
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={},
        question="question",
    )

    assert workspace.root.is_relative_to(tmp_path.resolve())
    assert "outside" not in workspace.root.parts
    assert "absolute" not in workspace.root.parts


@pytest.mark.parametrize("identifier", [".", ".."])
def test_context_workspace_never_accepts_dot_path_components(
    tmp_path: Path,
    identifier: str,
) -> None:
    workspace = GroundedContextWorkspace.open(
        _settings(tmp_path),
        thread_id=identifier,
        run_id=identifier,
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={},
        question="question",
    )

    assert workspace.root.is_relative_to(tmp_path.resolve())
    assert workspace.root != tmp_path.resolve()


def test_context_workspace_rejects_symlinked_parent_before_external_creation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside"
    workspace_root.mkdir()
    outside_root.mkdir()
    (workspace_root / "threads").symlink_to(
        outside_root,
        target_is_directory=True,
    )

    with pytest.raises(
        GroundedContextWorkspaceError,
        match="GROUNDED_CONTEXT_WORKSPACE_OUTSIDE_ROOT",
    ):
        GroundedContextWorkspace.open(
            _settings(workspace_root),
            thread_id="thread-1",
            run_id="run-1",
            merchant_id="merchant-1",
            access_role="merchant_analyst",
            user_scope={"userId": "user-1"},
            question="question",
        )

    assert list(outside_root.iterdir()) == []


def test_context_workspace_rejects_symlinked_scratch_and_subagent_parents(
    tmp_path: Path,
) -> None:
    workspace = GroundedContextWorkspace.open(
        _settings(tmp_path / "workspace"),
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={"userId": "user-1"},
        question="question",
    )
    outside_scratch = tmp_path / "outside-scratch"
    outside_subagent = tmp_path / "outside-subagent"
    outside_scratch.mkdir()
    outside_subagent.mkdir()
    (workspace.core_scratch_root / "notes").symlink_to(
        outside_scratch,
        target_is_directory=True,
    )
    (workspace.subagents_root / "analysis").symlink_to(
        outside_subagent,
        target_is_directory=True,
    )

    with pytest.raises(
        GroundedContextWorkspaceError,
        match="GROUNDED_SCRATCH_WRITE_FAILED",
    ):
        workspace.write_core_scratch("notes/step.txt", "outside denied")
    with pytest.raises(
        GroundedContextWorkspaceError,
        match="GROUNDED_SUBAGENT_WORKSPACE_OUTSIDE_ROOT",
    ):
        workspace.subagent_workspace("analysis", "job-1")

    assert list(outside_scratch.iterdir()) == []
    assert list(outside_subagent.iterdir()) == []


def test_run_filesystem_mounts_immutable_artifacts_read_only_and_scratch_rw(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={"userId": "user-1"},
        question="question",
    )
    artifact = WorkspaceArtifactStore(settings, workspace.artifacts_root).write_json(
        "query_results",
        "result.json",
        [{"value": 1}],
        immutable=True,
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="session-1",
            question="question",
            merchant_id="merchant-1",
        ),
        context_workspace=workspace,
    )
    scope = GroundedSemanticBackend(object())
    artifacts = GroundedRunFilesystemBackend(
        root_kind="artifacts",
        read_only=True,
        settings=settings,
    )
    scratch = GroundedRunFilesystemBackend(
        root_kind="scratch",
        read_only=False,
        settings=settings,
    )

    with scope.scope(session):
        listing = artifacts.ls("/artifacts/query_results")
        assert [item.path for item in listing.entries] == ["/query_results/result.json"]
        read = artifacts.read("/artifacts/%s" % artifact["relativePath"])
        assert read.file_data and '"value": 1' in read.file_data["content"]
        assert artifacts.grep("value", "/artifacts").matches
        assert artifacts.write("/artifacts/result.json", "replace").error == (
            "GROUNDED_CONTEXT_FILESYSTEM_READ_ONLY"
        )
        assert artifacts.read("/artifacts/../../outside").error == (
            "GROUNDED_CONTEXT_PATH_OUTSIDE_ROOT"
        )

        assert scratch.write("/workspace/notes/step.txt", "first").error is None
        assert scratch.edit(
            "/workspace/notes/step.txt",
            "first",
            "second",
        ).error is None
        scratch_read = scratch.read("/workspace/notes/step.txt")
        assert scratch_read.file_data == {
            "content": "second",
            "encoding": "utf-8",
        }
        assert all(
            not item.path.startswith("/.context-")
            for item in scratch.ls("/workspace").entries
        )

        Path(artifact["path"]).write_text("tampered", encoding="utf-8")
        assert artifacts.read(
            "/artifacts/%s" % artifact["relativePath"]
        ).error == WorkspaceArtifactStore.IMMUTABLE_STATE_INVALID
