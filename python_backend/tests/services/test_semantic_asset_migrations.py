from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, validate_semantic_asset
from merchant_ai.services.semantic_asset_migrations import migrate_published_semantic_asset


def test_migration_materializes_policy_driven_temporal_contracts_without_business_names() -> None:
    asset = {
        "tableName": "fact_metrics",
        "timeColumn": "event_day",
        "semanticColumns": [
            {
                "columnName": "entity_id",
                "canonicalEntityRef": "entity:generic",
                "isUniqueEntityKey": True,
            }
        ],
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


def test_migration_separates_derived_metric_lineage_from_physical_source_columns() -> None:
    asset = {
        "tableName": "fact_events",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "event_day"},
            {"columnName": "accepted_flag"},
            {"columnName": "entity_id"},
        ],
        "metrics": [
            {
                "metricKey": "accepted_measure",
                "formula": "SUM(accepted_flag)",
                "sourceColumns": ["accepted_flag"],
                "aggregationPolicy": "period_rollup",
            },
            {
                "metricKey": "total_measure",
                "formula": "COUNT(DISTINCT entity_id)",
                "sourceColumns": ["entity_id"],
                "aggregationPolicy": "period_rollup",
            },
            {
                "metricKey": "acceptance_ratio",
                "formula": "SUM(accepted_measure) / NULLIF(SUM(total_measure), 0)",
                "sourceColumns": [],
                "semanticCleanup": {
                    "droppedNonSchemaSourceColumns": [
                        "accepted_measure",
                        "total_measure",
                    ]
                },
                "aggregationPolicy": "ratio_of_sums",
            },
        ],
    }

    migrated, changes, errors = migrate_published_semantic_asset(asset)

    assert not errors
    metrics = {metric["metricKey"]: metric for metric in migrated["metrics"]}
    assert metrics["acceptance_ratio"]["requiresMetrics"] == [
        "accepted_measure",
        "total_measure",
    ]
    assert metrics["acceptance_ratio"]["sourceColumns"] == []
    assert metrics["accepted_measure"]["sourceColumns"] == ["accepted_flag"]
    assert metrics["total_measure"]["sourceColumns"] == ["entity_id"]
    assert "fact_events.acceptance_ratio.requiresMetrics" in changes


def test_migration_does_not_infer_metric_lineage_from_a_physical_name_collision() -> None:
    asset = {
        "tableName": "fact_events",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "event_day"},
            {"columnName": "measure_value"},
        ],
        "metrics": [
            {
                "metricKey": "measure_value",
                "formula": "SUM(measure_value)",
                "sourceColumns": ["measure_value"],
                "aggregationPolicy": "period_rollup",
            },
            {
                "metricKey": "physical_rollup",
                "formula": "SUM(measure_value)",
                "sourceColumns": ["measure_value"],
                "aggregationPolicy": "period_rollup",
            },
        ],
    }

    migrated, _, errors = migrate_published_semantic_asset(asset)

    assert not errors
    metrics = {metric["metricKey"]: metric for metric in migrated["metrics"]}
    assert metrics["physical_rollup"].get("requiresMetrics") is None
    assert metrics["physical_rollup"]["sourceColumns"] == ["measure_value"]


def test_migration_preserves_formally_declared_external_metric_lineage() -> None:
    asset = {
        "tableName": "fact_events",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "event_day"},
            {"columnName": "local_value"},
        ],
        "metrics": [
            {
                "metricKey": "local_measure",
                "formula": "SUM(local_value)",
                "sourceColumns": ["local_value"],
                "aggregationPolicy": "period_rollup",
            },
            {
                "metricKey": "cross_source_ratio",
                "formula": "SUM(local_measure) / NULLIF(SUM(external_measure), 0)",
                "sourceColumns": ["local_measure", "external_measure"],
                "metricDependencies": ["local_measure", "external_measure"],
                "aggregationPolicy": "ratio_of_sums",
            },
        ],
    }

    migrated, _, errors = migrate_published_semantic_asset(asset)

    assert not errors
    metrics = {metric["metricKey"]: metric for metric in migrated["metrics"]}
    assert metrics["cross_source_ratio"]["requiresMetrics"] == [
        "local_measure",
        "external_measure",
    ]
    assert metrics["cross_source_ratio"]["sourceColumns"] == []


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
