from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import merchant_ai.services.artifacts as artifact_module
from merchant_ai.config import get_settings
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.context_filesystem import merchant_uri_for_semantic_ref


def _store(root: Path) -> WorkspaceArtifactStore:
    return WorkspaceArtifactStore(get_settings(), root)


def _temporary_files_for(target: Path) -> list[Path]:
    return list(target.parent.glob(".artifact-write-*.tmp"))


def test_character_scanners_preserve_uri_and_search_grammar_without_regular_expressions(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")
    artifact = store.write_text("数据 分析/@north", "订单 @ detail.txt", "ORDER_detail_7 与中文订单")

    assert artifact["success"] is True
    assert Path(artifact["path"]).is_relative_to(store.root)
    assert artifact["relativePath"] == "数据_分析__north/订单_detail.txt"
    assert store.grep("123ORDER_detail_7")
    assert store.grep("中文订单")
    assert store.grep("单") == []
    assert merchant_uri_for_semantic_ref("semantic:销售///分析:manifest") == "merchant://topic/销售/分析/manifest"


def test_sanitized_paths_cannot_escape_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = _store(root)

    artifact = store.write_text("../../outside", "../escape.txt", "confined")

    assert artifact["success"] is True
    target = Path(artifact["path"])
    assert target.is_relative_to(root.resolve())
    assert target.read_text(encoding="utf-8") == "confined"
    assert not (tmp_path / "outside" / "escape.txt").exists()


def test_replace_failure_preserves_existing_artifact_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path / "artifacts")
    initial = store.write_text("planner", "plan.txt", "previous version")
    target = Path(initial["path"])
    observed_paths: list[tuple[Path, Path]] = []

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        observed_paths.append((Path(source), Path(destination)))
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifact_module.os, "replace", fail_replace)

    failed = store.write_text("planner", "plan.txt", "replacement version")

    assert failed == {
        "success": False,
        "error": "ARTIFACT_WRITE_FAILED",
        "path": "planner/plan.txt",
    }
    assert target.read_text(encoding="utf-8") == "previous version"
    assert observed_paths and observed_paths[0][0].parent == observed_paths[0][1].parent == target.parent
    assert _temporary_files_for(target) == []


def test_temporary_file_fsync_failure_preserves_existing_artifact_and_cleans_up(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path / "artifacts")
    initial = store.write_text("planner", "plan.txt", "durable version")
    target = Path(initial["path"])

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(artifact_module.os, "fsync", fail_fsync)

    failed = store.write_text("planner", "plan.txt", "partial version")

    assert failed["success"] is False
    assert failed["error"] == "ARTIFACT_WRITE_FAILED"
    assert target.read_text(encoding="utf-8") == "durable version"
    assert _temporary_files_for(target) == []


def test_concurrent_writers_publish_only_complete_artifacts(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")
    writer_count = 12
    barrier = threading.Barrier(writer_count)
    payloads = [json.dumps({"writer": index, "value": str(index) * 20_000}) for index in range(writer_count)]

    def publish(payload: str) -> dict:
        barrier.wait()
        return store.write_text("planner", "shared.json", payload, preview_chars=0)

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        results = list(executor.map(publish, payloads))

    assert all(result["success"] is True for result in results)
    target = store.root / "planner" / "shared.json"
    published = target.read_text(encoding="utf-8")
    assert published in payloads
    assert json.loads(published)["writer"] in range(writer_count)
    assert _temporary_files_for(target) == []


def test_immutable_write_is_content_addressed_idempotent_and_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")
    payload = {"rows": [{"值": 7}], "complete": True}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    expected_digest = hashlib.sha256(encoded).hexdigest()

    first = store.write_json("results", "snapshot.json", payload, preview_chars=0, immutable=True)
    repeated = store.write_json("results", "snapshot.json", payload, preview_chars=0, immutable=True)
    conflict = store.write_json("results", "snapshot.json", {"rows": []}, preview_chars=0, immutable=True)
    mutable_overwrite = store.write_text("results", "snapshot.json", "replacement")

    assert first["success"] is True
    assert first["immutable"] is True
    assert first["idempotent"] is False
    assert first["sha256"] == expected_digest
    assert len(first["sha256"]) == 64
    assert first["contentAddress"] == "sha256:%s" % expected_digest
    assert first["bytes"] == len(encoded)
    assert first["merchantUri"].startswith("merchant://artifact/results/")
    assert repeated["success"] is True
    assert repeated["idempotent"] is True
    assert repeated["sha256"] == first["sha256"]
    assert conflict["error"] == WorkspaceArtifactStore.IMMUTABLE_CONFLICT
    assert mutable_overwrite["error"] == WorkspaceArtifactStore.IMMUTABLE_CONFLICT
    assert Path(first["path"]).read_bytes() == encoded

    listed = store.ls("results")
    assert [item["relativePath"] for item in listed] == ["results/snapshot.json"]
    assert len(store.grep("complete")) == 1
    internal_paths = [path for path in Path(first["path"]).parent.iterdir() if path.name.startswith(".artifact-")]
    assert internal_paths
    assert all(store.read(str(path))["error"] == "ARTIFACT_NOT_FOUND" for path in internal_paths)


def test_concurrent_immutable_conflicts_have_one_winner_and_never_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    writer_count = 10
    stores = [_store(root) for _ in range(writer_count)]
    barrier = threading.Barrier(writer_count)
    payloads = ["immutable payload %d" % index for index in range(writer_count)]

    def publish(item: tuple[WorkspaceArtifactStore, str]) -> tuple[str, dict]:
        store, payload = item
        barrier.wait()
        return payload, store.write_text("results", "winner.txt", payload, immutable=True)

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        outcomes = list(executor.map(publish, zip(stores, payloads)))

    winners = [(payload, result) for payload, result in outcomes if result["success"] is True]
    conflicts = [result for _, result in outcomes if result["success"] is False]
    assert len(winners) == 1
    assert all(result["error"] == WorkspaceArtifactStore.IMMUTABLE_CONFLICT for result in conflicts)
    target = root / "results" / "winner.txt"
    assert target.read_text(encoding="utf-8") == winners[0][0]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == winners[0][1]["sha256"]
    assert _temporary_files_for(target) == []


def test_tampered_immutable_artifact_is_rejected_without_repairing_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")
    artifact = store.write_text("results", "evidence.txt", "verified", immutable=True)
    target = Path(artifact["path"])
    target.write_text("tampered", encoding="utf-8")

    read_result = store.read("results/evidence.txt")
    result = store.write_text("results", "evidence.txt", "verified", immutable=True)

    assert read_result["success"] is False
    assert read_result["error"] == WorkspaceArtifactStore.IMMUTABLE_STATE_INVALID
    assert result["success"] is False
    assert result["error"] == WorkspaceArtifactStore.IMMUTABLE_STATE_INVALID
    assert target.read_text(encoding="utf-8") == "tampered"


def test_missing_immutable_marker_is_rejected_by_trusted_artifact_reads(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "artifacts")
    artifact = store.write_text(
        "results",
        "evidence.txt",
        "verified population",
        immutable=True,
    )
    target = Path(artifact["path"])
    marker = next(
        path
        for path in target.parent.iterdir()
        if path.name.startswith(".artifact-immutable-")
    )
    marker.unlink()

    permissive_read = store.read("results/evidence.txt")
    trusted_read = store.read(
        "results/evidence.txt",
        require_immutable=True,
    )

    assert permissive_read["success"] is True
    assert permissive_read["immutable"] is False
    assert trusted_read["success"] is False
    assert (
        trusted_read["error"]
        == WorkspaceArtifactStore.IMMUTABLE_STATE_INVALID
    )
    assert store.grep(
        "population",
        require_immutable=True,
    ) == []
    assert store.ls(
        "results",
        require_immutable=True,
    ) == []


def test_immutable_intent_prevents_mutable_overwrite_after_data_publish_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = _store(tmp_path / "artifacts")
    original_create = artifact_module._atomic_create_text

    def fail_create(_target: Path, _content: bytes) -> None:
        raise OSError("injected data publish failure")

    monkeypatch.setattr(
        artifact_module,
        "_atomic_create_text",
        fail_create,
    )
    failed = store.write_text(
        "results",
        "evidence.txt",
        "verified",
        immutable=True,
    )
    target = store.root / "results" / "evidence.txt"

    assert failed["success"] is False
    assert failed["error"] == "ARTIFACT_WRITE_FAILED"
    assert not target.exists()
    assert store.write_text(
        "results",
        "evidence.txt",
        "mutable replacement",
    )["error"] == WorkspaceArtifactStore.IMMUTABLE_STATE_INVALID

    monkeypatch.setattr(
        artifact_module,
        "_atomic_create_text",
        original_create,
    )
    recovered = store.write_text(
        "results",
        "evidence.txt",
        "verified",
        immutable=True,
    )

    assert recovered["success"] is True
    assert recovered["immutable"] is True
    assert target.read_text(encoding="utf-8") == "verified"


def test_internal_artifact_paths_are_reserved_and_not_readable(tmp_path: Path) -> None:
    store = _store(tmp_path / "artifacts")

    result = store.write_text("results", ".artifact-write-user.tmp", "not allowed")

    assert result["success"] is False
    assert result["error"] == WorkspaceArtifactStore.PATH_RESERVED
