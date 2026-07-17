from __future__ import annotations

from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore


def _store(root: Path) -> WorkspaceArtifactStore:
    return WorkspaceArtifactStore(get_settings(), root)


def _assert_outside_root(result: dict) -> None:
    assert result["success"] is False
    assert result["error"] == WorkspaceArtifactStore.PATH_OUTSIDE_ROOT


def test_normal_round_trip_and_internal_absolute_paths_remain_supported(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")

    artifact = store.write_text("planner", "plan.txt", "safe content", preview_chars=0)

    assert artifact["success"] is True
    assert store.read(artifact["relativePath"])["content"] == "safe content"
    assert store.read(artifact["path"])["content"] == "safe content"


def test_read_rejects_parent_traversal_and_external_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP_SECRET", encoding="utf-8")
    store = _store(root)

    traversal_result = store.read("../secret.txt")
    absolute_result = store.read(str(outside))

    _assert_outside_root(traversal_result)
    _assert_outside_root(absolute_result)
    assert "TOP_SECRET" not in repr(traversal_result)
    assert "TOP_SECRET" not in repr(absolute_result)


def test_read_rejects_symlinked_file_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP_SECRET", encoding="utf-8")
    store = _store(root)
    (root / "leak.txt").symlink_to(outside)

    result = store.read("leak.txt")

    _assert_outside_root(result)
    assert "TOP_SECRET" not in repr(result)


def test_write_rejects_symlinked_directory_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = _store(root)
    (root / "planner").symlink_to(outside, target_is_directory=True)

    result = store.write_text("planner", "escaped.txt", "must stay inside")

    _assert_outside_root(result)
    assert not (outside / "escaped.txt").exists()


def test_grep_and_ls_do_not_expose_external_symlink_targets(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP_SECRET", encoding="utf-8")
    store = _store(root)
    store.write_text("planner", "safe.txt", "ordinary content")
    (root / "leak.txt").symlink_to(outside)

    assert store.grep("TOP_SECRET") == []
    assert all(item["relativePath"] != "leak.txt" for item in store.ls())


def test_switching_context_root_invalidates_paths_from_previous_root(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    store = _store(first_root)
    first_artifact = store.write_text("planner", "first.txt", "first")

    store.set_context_root(second_root)
    result = store.read(first_artifact["path"])

    _assert_outside_root(result)
    second_artifact = store.write_text("planner", "second.txt", "second")
    assert Path(second_artifact["path"]).is_relative_to(second_root.resolve())
