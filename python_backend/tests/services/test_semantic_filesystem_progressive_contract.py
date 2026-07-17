from __future__ import annotations

import json
from pathlib import Path

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import RecallItem
from merchant_ai.services.assets import (
    HybridRecallService,
    SemanticCatalogService,
    TopicAssetService,
    semantic_relationship_path,
    semantic_relationship_ref_id,
    semantic_table_detail_path,
    semantic_table_detail_ref_id,
    semantic_table_entry_keys,
    semantic_table_path,
    semantic_table_ref_id,
)


def _catalog() -> SemanticCatalogService:
    return SemanticCatalogService(TopicAssetService(get_settings()))


def test_every_progressively_advertised_ref_and_path_is_bidirectionally_readable() -> None:
    catalog = _catalog()
    checked: set[tuple[str, str]] = set()

    def assert_readable(ref_id: str, path: str) -> dict[str, object]:
        identity = (ref_id, path)
        if identity in checked:
            return catalog.read(ref_id=ref_id, path=path, max_chars=2_000_000)
        checked.add(identity)
        by_ref = catalog.read(ref_id=ref_id, max_chars=2_000_000)
        by_path = catalog.read(path=path, max_chars=2_000_000)
        by_both = catalog.read(ref_id=ref_id, path=path, max_chars=2_000_000)
        assert by_ref["success"], identity
        assert by_path["success"], identity
        assert by_both["success"], identity
        assert (by_ref["refId"], by_ref["path"], by_ref["kind"]) == (
            by_path["refId"],
            by_path["path"],
            by_path["kind"],
        )
        assert (by_both["refId"], by_both["path"]) == identity
        assert "#" not in path
        return by_path

    topic_index = catalog.topic_index_ref()
    assert_readable(str(topic_index["refId"]), str(topic_index["path"]))
    for topic in catalog.topic_assets.all_topic_names():
        manifest = catalog.manifest_ref(topic)
        manifest_read = assert_readable(str(manifest["refId"]), str(manifest["path"]))
        manifest_payload = json.loads(str(manifest_read["content"]))
        for table_item in manifest_payload["tables"]:
            detail_read = assert_readable(
                str(table_item["detailRefId"]),
                str(table_item["detailPath"]),
            )
            detail_payload = json.loads(str(detail_read["content"]))
            for child in detail_payload["children"]:
                child_read = assert_readable(str(child["refId"]), str(child["path"]))
                if str(child["path"]).endswith("/index.json"):
                    child_payload = json.loads(str(child_read["content"]))
                    for entry in child_payload.get("entries") or []:
                        assert_readable(str(entry["refId"]), str(entry["path"]))

    assert len(checked) > 1_000


def test_ref_and_path_conflict_and_full_asset_reads_fail_closed() -> None:
    catalog = _catalog()
    conflict = catalog.read(
        ref_id="semantic:经营画像:manifest",
        path="topics/客服工单/tables/dwm_cs_ticket_detail_di/detail.json",
    )
    assert conflict["success"] is False
    assert conflict["error"] == "SEMANTIC_REF_PATH_CONFLICT"

    full_asset = catalog.read(
        path=semantic_table_path("客服工单", "dwm_cs_ticket_detail_di"),
        max_chars=2_000_000,
    )
    assert full_asset["success"] is False
    assert full_asset["error"] == "FULL_TABLE_ASSET_DENIED"

    legacy_fragment = catalog.read(
        path="topics/客服工单/tables/dwm_cs_ticket_detail_di/asset.json#metric:ticket_cnt",
    )
    assert legacy_fragment["success"] is False
    assert legacy_fragment["error"] == "SEMANTIC_REF_NOT_FOUND"


def test_recall_navigation_has_no_host_paths_fragments_or_default_asset_coordinates() -> None:
    settings = get_settings()
    assets = TopicAssetService(settings)
    catalog = SemanticCatalogService(assets)
    documents = HybridRecallService(settings, assets)._load_documents()

    for item in documents:
        metadata = dict(item.metadata or {})
        if item.source_type == "GOVERNED_RULE":
            assert item.doc_id.startswith("semantic:rules:")
            assert not Path(str(metadata.get("sourcePath") or "")).is_absolute()
            continue
        ref_id = str(metadata.get("semanticRefId") or "")
        path = str(metadata.get("semanticPath") or "")
        assert ref_id.startswith("semantic:"), item.doc_id
        assert path.startswith("topics/"), item.doc_id
        assert "#metric:" not in path
        resolved = catalog.read(ref_id=ref_id, path=path, max_chars=2_000_000)
        assert resolved["success"], (item.doc_id, ref_id, path, resolved)
        if item.source_type == "SEMANTIC_TABLE_ASSET":
            assert resolved["kind"] == "TABLE_DETAIL"
            assert path.endswith("/detail.json")


def test_context_manifest_canonicalizes_and_deduplicates_before_limit() -> None:
    catalog = _catalog()
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    relationship_ref = semantic_relationship_ref_id(topic)
    relationship_path = semantic_relationship_path(topic)
    source_refs = {
        "legacy_table": RecallItem(
            doc_id=semantic_table_ref_id(topic, table),
            title="legacy table hit",
            content="internal recall payload",
            source_type="SEMANTIC_TABLE_ASSET",
            topic=topic,
            table=table,
            metadata={
                "semanticKind": "TABLE_ASSET",
                "semanticRefId": semantic_table_ref_id(topic, table),
                "semanticPath": semantic_table_path(topic, table),
            },
        ),
        "relationship_1": RecallItem(
            doc_id="semantic:客服工单:relationship:r1",
            title="relationship one",
            content="r1",
            source_type="SEMANTIC_RELATIONSHIP",
            topic=topic,
            metadata={
                "semanticKind": "RELATIONSHIP",
                "semanticRefId": relationship_ref,
                "semanticPath": relationship_path,
            },
        ),
        "relationship_2": RecallItem(
            doc_id="semantic:客服工单:relationship:r2",
            title="relationship two",
            content="r2",
            source_type="SEMANTIC_RELATIONSHIP",
            topic=topic,
            metadata={
                "semanticKind": "RELATIONSHIP",
                "semanticRefId": relationship_ref,
                "semanticPath": relationship_path,
            },
        ),
    }

    manifest = catalog.context_manifest(source_refs, limit=2)

    assert [(item["refId"], item["path"]) for item in manifest["refs"]] == [
        (semantic_table_detail_ref_id(topic, table), semantic_table_detail_path(topic, table)),
        (relationship_ref, relationship_path),
    ]
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "#metric:" not in serialized
    assert "/asset.json\"" not in serialized


def test_semantic_entry_keys_are_stable_across_reordering_and_fail_closed_on_duplicates() -> None:
    rules = [
        {"ruleId": "refund-policy", "title": "Refund policy", "content": "A"},
        {"ruleId": "ticket-policy", "title": "Ticket policy", "content": "B"},
    ]
    original = dict(zip((item["ruleId"] for item in rules), semantic_table_entry_keys("rules", rules)))
    reordered = list(reversed(rules))
    after = dict(zip((item["ruleId"] for item in reordered), semantic_table_entry_keys("rules", reordered)))
    assert original == after == {
        "refund-policy": "refund-policy",
        "ticket-policy": "ticket-policy",
    }

    colliding_columns = [
        {"columnName": "a/b", "description": "first"},
        {"columnName": "a\\b", "description": "second"},
    ]
    collision_keys = semantic_table_entry_keys("columns", colliding_columns)
    assert len(set(collision_keys)) == 2
    assert all(key.startswith("a_b-") for key in collision_keys)

    with pytest.raises(ValueError, match="SEMANTIC_ENTRY_KEY_COLLISION"):
        semantic_table_entry_keys("columns", [colliding_columns[0], dict(colliding_columns[0])])
