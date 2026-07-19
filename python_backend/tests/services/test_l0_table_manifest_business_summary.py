from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.services.assets import (
    SemanticCatalogService,
    TopicAssetService,
    build_stable_topic_table_manifest,
)
from merchant_ai.services.planning import compact_table_manifest_for_prompt


TOPIC = "客服工单"
TABLE = "dwm_cs_ticket_detail_di"


def test_ticket_table_l0_is_discoverable_by_product_without_claiming_product_grain() -> None:
    topic_assets = TopicAssetService(get_settings())
    stable_manifest = build_stable_topic_table_manifest(topic_assets, [TOPIC])
    planner_manifest = compact_table_manifest_for_prompt(stable_manifest)

    assert planner_manifest["tableCount"] == 1
    table = planner_manifest["tables"][0]
    assert table["table"] == TABLE
    assert set(table) == {
        "topic",
        "table",
        "title",
        "businessSummary",
        "detailRefId",
        "detailPath",
    }
    summary = table["businessSummary"]
    assert all(term in summary for term in ("客服工单", "订单/子订单", "商品", "客服服务过程"))
    assert "按表内商品属性" in summary
    assert "商品粒度" not in summary
    assert "商品级" not in summary

    serialized_l0 = json.dumps(planner_manifest, ensure_ascii=False)
    for forbidden in (
        "dataGrain",
        "timeColumn",
        "merchantFilterColumn",
        "metricCount",
        "ruleCount",
        "relationships",
        "spu_id",
        "spu_name",
        "客服工单明细量",
    ):
        assert forbidden not in serialized_l0


def test_ticket_table_details_expose_navigation_but_require_exact_leaf_reads() -> None:
    topic_assets = TopicAssetService(get_settings())
    catalog = SemanticCatalogService(topic_assets)

    manifest_read = catalog.read(ref_id=f"semantic:{TOPIC}:manifest")
    assert manifest_read["success"] is True
    l0 = json.loads(manifest_read["content"])
    assert l0["tables"][0]["table"] == TABLE
    assert "商品" in l0["tables"][0]["businessSummary"]
    assert "dataGrain" not in manifest_read["content"]
    assert "spu_id" not in manifest_read["content"]

    detail_read = catalog.read(ref_id=l0["tables"][0]["detailRefId"])
    assert detail_read["success"] is True
    detail = json.loads(detail_read["content"])
    assert detail["dataGrain"] == "订单/子订单明细粒度"
    assert detail["dataGrain"] != "商品粒度"

    navigation = detail["semanticNavigation"]
    assert navigation["source"] == "published_asset"
    assert navigation["questionIndependent"] is True
    assert navigation["bindingEvidence"] is False
    published_asset = topic_assets.load_table_asset(TOPIC, TABLE)
    published_columns = {
        str(item.get("columnName") or item.get("key") or ""): item
        for item in published_asset.get("semanticColumns") or []
        if isinstance(item, dict) and str(item.get("columnName") or item.get("key") or "")
    }
    advertised_columns = {
        str(item.get("key") or ""): item
        for item in navigation["columnLeaves"]
        if isinstance(item, dict) and str(item.get("key") or "")
    }
    assert advertised_columns
    assert set(advertised_columns).issubset(published_columns)
    assert navigation["publishedCounts"]["columns"] == len(published_columns)
    assert navigation["advertisedCounts"]["columns"] == len(advertised_columns)
    for key, leaf in advertised_columns.items():
        assert set(leaf).issubset(
            {"key", "aliases", "refId", "path", "semanticRole", "timeRole"}
        )
        assert leaf["refId"]
        assert leaf["path"]
        assert "definition" not in leaf
        assert "schemaContract" not in leaf
        assert key in published_columns

    exact_leaf = next(iter(advertised_columns.values()))
    exact_read = catalog.read(ref_id=exact_leaf["refId"], max_chars=2_000_000)
    assert exact_read["success"] is True
    exact_payload = json.loads(exact_read["content"])
    assert exact_payload["key"] == exact_leaf["key"]
    assert exact_payload["definition"]["columnName"] == exact_leaf["key"]
    assert exact_payload["definition"]["schemaContract"]

    schema_ref = next(child["refId"] for child in detail["children"] if child["kind"] == "SCHEMA")
    schema_read = catalog.read(ref_id=schema_ref, max_chars=2_000_000)
    assert schema_read["success"] is True
    schema = json.loads(schema_read["content"])
    column_names = {item["columnName"] for item in schema["schemaColumns"]}
    assert {"spu_id", "spu_name"}.issubset(column_names)


def test_every_published_table_has_a_business_summary_for_core_selection() -> None:
    topic_assets = TopicAssetService(get_settings())
    manifest = build_stable_topic_table_manifest(
        topic_assets,
        topic_assets.all_topic_names(),
    )

    assert manifest["tables"]
    assert all(str(item.get("businessSummary") or "").strip() for item in manifest["tables"])
    serialized = json.dumps(compact_table_manifest_for_prompt(manifest), ensure_ascii=False)
    for forbidden in (
        "dataGrain",
        "timeColumn",
        "merchantFilterColumn",
        "metricCount",
        "ruleCount",
        "relationships",
    ):
        assert forbidden not in serialized
