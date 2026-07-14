import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, semantic_catalog_conflict_detection


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def topic_assets(tmp_path: Path) -> TopicAssetService:
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    return TopicAssetService(settings)


def write_asset(root: Path, topic: str, table: str, asset: dict) -> None:
    directory = root / "topics" / topic / "tables" / table
    write_json(directory / "asset.json", {"topic": topic, "tableName": table, "status": "PUBLISHED", **asset})


def conflict_types(report: dict) -> set[str]:
    return {str(item.get("type") or "") for item in report.get("conflicts") or []}


def test_catalog_blocks_same_table_in_multiple_topics(tmp_path):
    service = topic_assets(tmp_path)
    write_asset(tmp_path, "trade", "shared_table", {})
    write_asset(tmp_path, "refund", "shared_table", {})

    report = semantic_catalog_conflict_detection(service)

    assert "duplicate_table_owner" in conflict_types(report)


def test_catalog_blocks_multiple_formulas_for_one_canonical_metric(tmp_path):
    service = topic_assets(tmp_path)
    write_asset(
        tmp_path,
        "profile",
        "daily_profile",
        {"metrics": [{"metricKey": "gmv", "canonicalMetricKey": "gmv", "formula": "SUM(order_gmv)"}]},
    )
    write_asset(
        tmp_path,
        "trade",
        "trade_detail",
        {"metrics": [{"metricKey": "paid_gmv", "canonicalMetricKey": "gmv", "formula": "SUM(pay_amt)"}]},
    )

    report = semantic_catalog_conflict_detection(service)

    assert "catalog_metric_formula_conflict" in conflict_types(report)


def test_catalog_blocks_two_labels_for_same_enum_value(tmp_path):
    service = topic_assets(tmp_path)
    write_asset(
        tmp_path,
        "trade",
        "order_detail",
        {"semanticColumns": [{"columnName": "status", "enumMappings": {"1": "paid"}}]},
    )
    candidate = {
        "topic": "trade_v2",
        "tableName": "order_detail",
        "status": "PUBLISHED",
        "semanticColumns": [{"columnName": "status", "enumMappings": {"1": "created"}}],
    }

    report = semantic_catalog_conflict_detection(service, "trade_v2", "order_detail", candidate)

    assert "duplicate_table_owner" in conflict_types(report)
    assert "enum_interpretation_conflict" in conflict_types(report)


def test_sidecar_is_authoritative_over_stale_denormalized_asset(tmp_path):
    service = topic_assets(tmp_path)
    directory = tmp_path / "topics" / "trade" / "tables" / "orders"
    write_json(
        directory / "asset.json",
        {
            "topic": "trade",
            "tableName": "orders",
            "metrics": [{"metricKey": "gmv", "formula": "SUM(stale_amt)", "aliases": ["stale"]}],
        },
    )
    write_json(directory / "metrics.json", [{"metricKey": "gmv", "formula": "SUM(gmv_amt)", "aliases": ["GMV"]}])

    asset = service.load_table_asset("trade", "orders")

    assert asset["metrics"] == [{"metricKey": "gmv", "formula": "SUM(gmv_amt)", "aliases": ["GMV"]}]


def test_checked_in_catalog_has_no_blocking_conflicts():
    service = TopicAssetService(get_settings())

    report = semantic_catalog_conflict_detection(service)

    assert report == {"status": "passed", "conflictCount": 0, "conflicts": []}
