from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, validate_semantic_asset
from merchant_ai.services.semantic_asset_migrations import migrate_published_semantic_asset


def test_migration_materializes_policy_driven_temporal_contracts_without_business_names() -> None:
    asset = {
        "tableName": "fact_metrics",
        "timeColumn": "event_day",
        "semanticColumns": [{"columnName": "entity_id", "role": "ENTITY"}],
        "metrics": [
            {
                "metricKey": "count_metric",
                "formula": "SUM(count_value)",
                "aggregationPolicy": "period_rollup",
            },
            {
                "metricKey": "snapshot_metric",
                "formula": "MAX(snapshot_value)",
                "aggregationPolicy": "latest_value_only",
            },
            {
                "metricKey": "daily_metric",
                "formula": "MAX(daily_value)",
                "aggregationPolicy": "daily_value_only",
            },
            {
                "metricKey": "ratio_metric",
                "formula": "event_count / NULLIF(base_count, 0)",
                "aggregationPolicy": "ratio_of_sums",
            },
        ],
    }

    migrated, changes, errors = migrate_published_semantic_asset(asset)

    assert not errors
    assert changes
    assert migrated["entityLookupPolicy"] == {"mode": "clarify", "timeColumn": "event_day"}
    assert migrated["metrics"][0]["applicableTimeGrain"] == "period"
    assert migrated["metrics"][0]["timeSemantics"]["selectionPolicy"] == "period_window"
    assert migrated["metrics"][1]["applicableTimeGrain"] == "day"
    assert migrated["metrics"][1]["timeSemantics"]["selectionPolicy"] == "latest_as_of"
    assert migrated["metrics"][2]["timeSemantics"]["selectionPolicy"] == "per_time_grain"
    assert migrated["metrics"][3]["formula"] == "SUM(event_count) / NULLIF(SUM(base_count), 0)"
    assert all(metric["timeColumn"] == "event_day" for metric in migrated["metrics"])


def test_migration_preserves_existing_temporal_declarations() -> None:
    asset = {
        "tableName": "fact_metrics",
        "timeColumn": "event_day",
        "metrics": [
            {
                "metricKey": "count_metric",
                "formula": "SUM(count_value)",
                "aggregationPolicy": "period_rollup",
                "applicableTimeGrain": "custom_period",
                "timeColumn": "business_day",
                "timeSemantics": {
                    "selectionPolicy": "custom_selection",
                    "asOfPolicy": "calendar",
                    "missingDataPolicy": "fail_closed",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    migrated, _, errors = migrate_published_semantic_asset(asset)

    assert not errors
    metric = migrated["metrics"][0]
    assert metric["applicableTimeGrain"] == "custom_period"
    assert metric["timeColumn"] == "business_day"
    assert metric["timeSemantics"]["selectionPolicy"] == "custom_selection"
    assert metric["timeSemantics"]["asOfPolicy"] == "calendar"


def test_checked_in_published_catalog_passes_the_complete_semantic_asset_contract() -> None:
    assets = TopicAssetService(get_settings())
    violations = []

    for topic in assets.all_topic_names():
        relationships = assets.load_relationships(topic)
        for manifest_entry in assets.load_manifest(topic):
            table = str(manifest_entry.get("tableName") or "")
            if not table:
                continue
            validation = validate_semantic_asset(assets.load_table_asset(topic, table), relationships)
            for error in validation.get("errors") or []:
                violations.append({"topic": topic, "table": table, **error})

    assert violations == []
