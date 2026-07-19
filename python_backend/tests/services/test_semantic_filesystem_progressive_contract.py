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
    semantic_relationship_index_path,
    semantic_relationship_index_ref_id,
    semantic_relationship_entry_path,
    semantic_relationship_entry_ref_id,
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


def test_trade_l1_exposes_compact_exact_leaf_navigation_without_binding_evidence() -> None:
    catalog = _catalog()
    detail = catalog.read(
        path="topics/电商交易/tables/dwm_trade_order_detail_di/detail.json",
        max_chars=20_000,
    )

    assert detail["success"] is True
    payload = json.loads(str(detail["content"]))
    navigation = payload["semanticNavigation"]
    metric = next(item for item in navigation["metricLeaves"] if item["key"] == "sku_cnt")
    brand = next(item for item in navigation["columnLeaves"] if item["key"] == "brand_name")
    article = next(item for item in navigation["columnLeaves"] if item["key"] == "article_id")

    assert "销量" in metric["aliases"]
    assert "品牌name" in brand["aliases"]
    assert metric["path"].endswith("/metrics/sku_cnt.json")
    assert brand["path"].endswith("/columns/brand_name.json")
    assert article["path"].endswith("/columns/article_id.json")
    assert "formula" not in json.dumps(navigation, ensure_ascii=False).lower()

    for leaf in (metric, brand, article):
        exact = catalog.read(
            ref_id=leaf["refId"],
            path=leaf["path"],
            max_chars=20_000,
        )
        assert exact["success"] is True
        assert exact["kind"] in {"METRIC", "COLUMN"}


def test_every_published_table_l1_has_bounded_asset_derived_exact_navigation() -> None:
    catalog = _catalog()
    checked_tables = 0

    for topic in catalog.topic_assets.all_topic_names():
        for manifest_item in catalog.topic_assets.load_manifest(topic):
            table = str(manifest_item.get("tableName") or "")
            if not table:
                continue
            checked_tables += 1
            asset = catalog.topic_assets.load_table_asset(topic, table)
            poisoned_manifest = {
                **manifest_item,
                "navigationHints": {
                    "metrics": [
                        {
                            "key": "manifest_only_metric",
                            "aliases": ["MANIFEST_ONLY_ALIAS"],
                        }
                    ],
                    "columns": ["manifest_only_column"],
                },
            }
            ref = catalog.table_detail_ref(
                topic,
                table,
                manifest_item=poisoned_manifest,
            )
            assert int(ref["estimatedChars"]) <= catalog.L1_DETAIL_MAX_CHARS
            assert int(ref["estimatedChars"]) == len(str(ref["content"]))

            payload = json.loads(str(ref["content"]))
            navigation = payload["semanticNavigation"]
            assert navigation["source"] == "published_asset"
            assert navigation["questionIndependent"] is True
            assert navigation["bindingEvidence"] is False
            assert navigation["publishedCounts"] == {
                "metrics": len(asset.get("metrics") or []),
                "columns": len(asset.get("semanticColumns") or []),
            }
            assert navigation["advertisedCounts"] == {
                "metrics": len(navigation["metricLeaves"]),
                "columns": len(navigation["columnLeaves"]),
            }

            serialized_navigation = json.dumps(navigation, ensure_ascii=False)
            assert "manifest_only" not in serialized_navigation
            assert "MANIFEST_ONLY_ALIAS" not in serialized_navigation
            for forbidden_definition_field in (
                "formula",
                "sourceColumns",
                "enumValues",
                "description",
            ):
                assert forbidden_definition_field not in serialized_navigation

            for section, field, leaf_field in (
                ("metrics", "metrics", "metricLeaves"),
                ("columns", "semanticColumns", "columnLeaves"),
            ):
                values = [item for item in asset.get(field) or [] if isinstance(item, dict)]
                published_keys = set(semantic_table_entry_keys(section, values))
                leaves = navigation[leaf_field]
                if values:
                    assert leaves
                for leaf in leaves:
                    assert set(leaf) == {"key", "aliases", "refId", "path"}
                    assert leaf["key"] in published_keys
                    assert isinstance(leaf["aliases"], list)
                    assert len(leaf["aliases"]) <= (catalog.L1_NAVIGATION_MAX_ALIASES_PER_LEAF)
                    exact = catalog.read(
                        ref_id=str(leaf["refId"]),
                        path=str(leaf["path"]),
                        max_chars=2_000_000,
                    )
                    assert exact["success"], (topic, table, leaf, exact)
                    assert exact["kind"] in {"METRIC", "COLUMN"}

    assert checked_tables >= 10


def test_all_topic_l0_manifests_do_not_disclose_or_load_table_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()

    for topic in catalog.topic_assets.all_topic_names():
        manifest = catalog.manifest_ref(topic)
        payload = json.loads(str(manifest["content"]))
        assert payload["layer"] == "manifest"
        assert "semanticNavigation" not in payload
        l0_string_values = {value for table in payload["tables"] for value in table.values() if isinstance(value, str)}
        for table in payload["tables"]:
            assert set(table) <= {
                "topic",
                "table",
                "title",
                "detailRefId",
                "detailPath",
                "businessSummary",
            }
            asset = catalog.topic_assets.load_table_asset(topic, str(table["table"]))
            for section, values in (
                ("metrics", asset.get("metrics") or []),
                ("columns", asset.get("semanticColumns") or []),
            ):
                for key in semantic_table_entry_keys(section, values):
                    assert key not in l0_string_values

    def fail_if_l0_loads_asset(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("L0 navigation must not load asset.json")

    monkeypatch.setattr(
        catalog.topic_assets,
        "load_table_asset",
        fail_if_l0_loads_asset,
    )
    assert catalog.ls(topic="电商交易")
    catalog.grep(query="订单", topic="电商交易")


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
        (
            semantic_relationship_index_ref_id(topic),
            semantic_relationship_index_path(topic),
        ),
    ]
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "#metric:" not in serialized
    assert "/asset.json\"" not in serialized


def test_entity_field_discloses_typed_lookup_capabilities_only_at_exact_field_read() -> None:
    catalog = _catalog()
    result = catalog.read(
        path=(
            "topics/电商交易/tables/dwm_trade_order_detail_di/"
            "columns/order_id.json"
        ),
        max_chars=2_000_000,
    )

    assert result["success"] is True
    definition = json.loads(str(result["content"]))["definition"]
    assert definition["entityRole"] == "KEY"
    assert definition["canonicalEntityRef"] == "entity:order"
    assert definition["isUniqueEntityKey"] is True
    assert definition["filterOperators"] == ["EQ", "IN"]
    assert definition["lookupTimePolicy"] == {
        "mode": "unbounded",
        "timeRequired": False,
        "timeColumn": "pt",
        "policySource": "field",
    }
    assert definition["schemaContract"]["dataType"] == "varchar(128)"


def test_relationships_are_progressively_disclosed_as_index_then_one_exact_edge() -> None:
    catalog = _catalog()
    topic = "商品管理"
    index = catalog.read(
        ref_id=semantic_relationship_index_ref_id(topic),
        path=semantic_relationship_index_path(topic),
        max_chars=2_000_000,
    )

    assert index["success"] is True
    index_payload = json.loads(str(index["content"]))
    selected = next(
        item
        for item in index_payload["entries"]
        if item["name"] == "order_goods_by_spu_id"
    )
    assert selected["refId"] == semantic_relationship_entry_ref_id(
        topic,
        "order_goods_by_spu_id",
    )
    assert selected["path"] == semantic_relationship_entry_path(
        topic,
        "order_goods_by_spu_id",
    )

    edge = catalog.read(
        ref_id=str(selected["refId"]),
        path=str(selected["path"]),
        max_chars=2_000_000,
    )
    assert edge["success"] is True
    edge_payload = json.loads(str(edge["content"]))
    assert len(edge_payload["relationships"]) == 1
    assert edge_payload["relationships"][0]["name"] == "order_goods_by_spu_id"
    assert "goods_refund_by_spu_name" not in str(edge["content"])


def test_all_published_relationships_and_transfer_keys_are_governed() -> None:
    catalog = _catalog()
    table_details: dict[str, dict[str, object]] = {}
    field_entries: dict[tuple[str, str], dict[str, object]] = {}

    for topic in catalog.topic_assets.all_topic_names():
        manifest = catalog.read(
            path="topics/%s/manifest.json" % topic,
            max_chars=2_000_000,
        )
        assert manifest["success"], topic
        for table_item in json.loads(str(manifest["content"]))["tables"]:
            table = str(table_item["table"])
            detail_read = catalog.read(
                path=str(table_item["detailPath"]),
                max_chars=2_000_000,
            )
            detail = json.loads(str(detail_read["content"]))
            table_details[table] = detail
            columns_child = next(
                item
                for item in detail["children"]
                if str(item["path"]).endswith("/columns/index.json")
            )
            columns_index = catalog.read(
                path=str(columns_child["path"]),
                max_chars=2_000_000,
            )
            for entry in json.loads(str(columns_index["content"]))["entries"]:
                field_entries[(table, str(entry["key"]))] = entry

    checked_relationships: set[tuple[str, str, str]] = set()
    for topic in catalog.topic_assets.all_topic_names():
        relationship_index = catalog.read(
            path=semantic_relationship_index_path(topic),
            max_chars=2_000_000,
        )
        if not relationship_index.get("success"):
            continue
        for entry in json.loads(str(relationship_index["content"])).get(
            "entries", []
        ):
            edge_read = catalog.read(
                path=str(entry["path"]),
                max_chars=2_000_000,
            )
            relationship = json.loads(str(edge_read["content"]))[
                "relationships"
            ][0]
            identity = (
                str(relationship["name"]),
                str(relationship["leftTable"]),
                str(relationship["rightTable"]),
            )
            if identity in checked_relationships:
                continue
            checked_relationships.add(identity)

            assert relationship.get("keys"), identity
            assert relationship.get("grain"), identity
            assert str(relationship.get("cardinality") or "").lower() in {
                "one_to_one",
                "one_to_many",
                "many_to_one",
                "many_to_many",
            }, identity
            assert relationship.get("fanoutPolicy"), identity
            assert "dedupKeys" in relationship, identity
            assert relationship.get("rowIdentityPreserved"), identity

            left_table = str(relationship["leftTable"])
            right_table = str(relationship["rightTable"])
            left_detail = table_details[left_table]
            right_detail = table_details[right_table]
            for left_key, right_key in relationship["keys"]:
                is_scope_key = (
                    left_key == left_detail.get("merchantFilterColumn")
                    and right_key == right_detail.get("merchantFilterColumn")
                )
                is_time_key = (
                    left_key == left_detail.get("timeColumn")
                    and right_key == right_detail.get("timeColumn")
                )
                if is_scope_key or is_time_key:
                    continue
                left_entry = field_entries[(left_table, str(left_key))]
                right_entry = field_entries[(right_table, str(right_key))]
                left_definition = json.loads(
                    str(
                        catalog.read(
                            path=str(left_entry["path"]),
                            max_chars=2_000_000,
                        )["content"]
                    )
                )["definition"]
                right_definition = json.loads(
                    str(
                        catalog.read(
                            path=str(right_entry["path"]),
                            max_chars=2_000_000,
                        )["content"]
                    )
                )["definition"]
                left_entity = str(
                    left_definition.get("canonicalEntityRef") or ""
                )
                right_entity = str(
                    right_definition.get("canonicalEntityRef") or ""
                )
                assert left_entity and left_entity == right_entity, (
                    identity,
                    left_key,
                    right_key,
                    left_entity,
                    right_entity,
                )
                assert "IN" in left_definition.get("filterOperators", []), (
                    identity,
                    left_key,
                )
                assert "IN" in right_definition.get("filterOperators", []), (
                    identity,
                    right_key,
                )

    assert len(checked_relationships) >= 16


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
