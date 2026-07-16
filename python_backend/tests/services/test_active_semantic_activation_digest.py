from __future__ import annotations

import json
from pathlib import Path

import pytest

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService


ACTIVE_FIXTURE_FILES = {
    "topic/manifest.json": [{"tableName": "events"}],
    "topic/relationships.json": [],
    "topic/tables/events/asset.json": {
        "topic": "topic",
        "tableName": "events",
        "timeColumn": "event_day",
    },
    "topic/tables/events/schema.json": [{"columnName": "event_day"}],
    "topic/tables/events/semantic_columns.json": [
        {"columnName": "event_day", "semanticRole": "TIME"}
    ],
    "topic/tables/events/metrics.json": [{"metricKey": "event_count", "formula": "COUNT(*)"}],
    "topic/tables/events/terms.json": [{"term": "event", "description": "event row"}],
    "topic/tables/events/knowledge_rules.json": [{"ruleId": "rule_1", "content": "use event day"}],
}


def _write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _build_service(root: Path, paths: list[str] | None = None) -> TopicAssetService:
    ordered_paths = paths or list(ACTIVE_FIXTURE_FILES)
    for relative in ordered_paths:
        _write_payload(root / relative, ACTIVE_FIXTURE_FILES[relative])
    settings = get_settings().model_copy(update={"topic_path": str(root)})
    return TopicAssetService(settings)


@pytest.mark.parametrize(
    ("sidecar_name", "replacement"),
    [
        ("metrics.json", [{"metricKey": "new_metric", "formula": "SUM(value_x)"}]),
        (
            "semantic_columns.json",
            [{"columnName": "value_x", "semanticRole": "MEASURE", "businessName": "Value"}],
        ),
        (
            "knowledge_rules.json",
            [{"ruleId": "rule_2", "content": "apply the reviewed runtime rule"}],
        ),
    ],
)
def test_active_source_hash_changes_for_every_runtime_semantic_sidecar(
    tmp_path: Path,
    sidecar_name: str,
    replacement: object,
) -> None:
    root = tmp_path / "topics"
    service = _build_service(root)
    topic_before = service.semantic_source_hash(["topic"])
    table_before = service.semantic_table_source_hash("topic", "events")

    _write_payload(root / "topic" / "tables" / "events" / sidecar_name, replacement)

    assert service.semantic_source_hash(["topic"]) != topic_before
    assert service.semantic_table_source_hash("topic", "events") != table_before


def test_active_semantic_digest_is_independent_of_creation_and_iteration_order(tmp_path: Path) -> None:
    paths = list(ACTIVE_FIXTURE_FILES)
    first = _build_service(tmp_path / "first", paths)
    second = _build_service(tmp_path / "second", list(reversed(paths)))

    first_files = [
        path.relative_to(first.root).as_posix()
        for path in first.canonical_semantic_files(["topic"])
    ]
    second_files = [
        path.relative_to(second.root).as_posix()
        for path in reversed(second.canonical_semantic_files(["topic"]))
    ]

    assert first_files == sorted(first_files)
    assert first.semantic_source_hash(["topic"]) == second.semantic_source_hash(["topic"])
    assert first.semantic_source_hash(["topic"]) == second.semantic_source_hash(["topic"])
    assert sorted(second_files) == first_files


def test_pending_temporary_and_generated_files_do_not_change_active_digest(tmp_path: Path) -> None:
    root = tmp_path / "topics"
    service = _build_service(root)
    before = service.semantic_source_hash(["topic"])

    noise = {
        "topic/pending/events/metrics.json": [{"metricKey": "not_active"}],
        "topic/tables/events/.metrics.json.activation.tmp": [{"metricKey": "temporary"}],
        "topic/tables/events/profile.json": {"generated": True},
        "topic/tables/events/semantic_version.json": {"generated": True},
        "topic/tables/events/metrics.md": "generated documentation",
        "topic/tables/events/nested/knowledge_rules.json": [{"ruleId": "nested"}],
        "topic/manifest.md": "generated documentation",
    }
    for relative, payload in noise.items():
        _write_payload(root / relative, payload)

    assert service.semantic_source_hash(["topic"]) == before
    assert TopicAssetService(service.settings).semantic_source_hash(["topic"]) == before
    assert all(
        path.relative_to(root).as_posix() not in noise
        for path in service.canonical_semantic_files(["topic"])
    )
