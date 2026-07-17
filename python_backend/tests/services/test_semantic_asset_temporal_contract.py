import json

from merchant_ai.config import get_settings
from merchant_ai.services.assets import (
    TopicAssetService,
    semantic_asset_builder_tool,
    validate_semantic_asset,
)


def validation_codes(asset: dict) -> set[str]:
    result = validate_semantic_asset(asset, [])
    return {str(item.get("code") or "") for item in result["errors"]}


def test_non_metric_semantic_column_cannot_publish_a_metric_formula() -> None:
    asset = {
        "tableName": "daily_fact",
        "schemaColumns": [{"columnName": "status"}],
        "semanticColumns": [
            {
                "columnName": "status",
                "role": "DIMENSION",
                "metricFormula": "AVG(status)",
            }
        ],
    }

    assert "NON_METRIC_COLUMN_HAS_METRIC_FORMULA" in validation_codes(asset)


def test_daily_value_only_requires_grain_and_rejects_cross_day_rollup_formula() -> None:
    asset = {
        "tableName": "daily_profile",
        "schemaColumns": [{"columnName": "daily_rate"}],
        "metrics": [
            {
                "metricKey": "daily_rate",
                "formula": "AVG(daily_rate)",
                "sourceColumns": ["daily_rate"],
                "aggregationPolicy": "daily_value_only",
            }
        ],
    }

    assert {
        "DAILY_VALUE_TIME_GRAIN_UNDECLARED",
        "DAILY_VALUE_CROSS_DAY_AGGREGATION_FORBIDDEN",
    } <= validation_codes(asset)


def test_daily_value_only_accepts_safe_daily_selection_contract() -> None:
    asset = {
        "tableName": "daily_profile",
        "timeColumn": "event_day",
        "schemaColumns": [{"columnName": "daily_rate"}, {"columnName": "event_day"}],
        "metrics": [
            {
                "metricKey": "daily_rate",
                "formula": "MAX(daily_rate)",
                "sourceColumns": ["daily_rate"],
                "aggregationPolicy": "daily_value_only",
                "metricGrain": "tenant_day",
                "applicableTimeGrain": "day",
                "timeSemantics": {
                    "selectionPolicy": "per_time_grain",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert not validation_codes(asset)


def test_ratio_metric_requires_an_explicit_rollup_contract() -> None:
    asset = {
        "tableName": "daily_profile",
        "schemaColumns": [
            {"columnName": "numerator"},
            {"columnName": "denominator"},
        ],
        "metrics": [
            {
                "metricKey": "ratio",
                "formula": "numerator / NULLIF(denominator, 0)",
                "sourceColumns": ["numerator", "denominator"],
            }
        ],
    }

    assert "RATIO_AGGREGATION_POLICY_UNDECLARED" in validation_codes(asset)


def test_semantic_builder_protocol_requires_metric_calculation_semantics() -> None:
    schema = semantic_asset_builder_tool().openai_schema()["function"]["parameters"]
    metric_schema = schema["properties"]["metrics"]["items"]
    column_schema = schema["properties"]["semanticColumns"]["items"]

    assert "calculationSemantics" in metric_schema["required"]
    assert "calculationSemantics" in metric_schema["properties"]
    assert "calculationSemantics" in column_schema["properties"]


def test_non_composable_semantics_requires_native_window_and_alternative() -> None:
    asset = {
        "tableName": "daily_profile",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "daily_unique"},
            {"columnName": "event_day"},
        ],
        "metrics": [
            {
                "metricKey": "daily_unique",
                "formula": "SUM(daily_unique)",
                "sourceColumns": ["daily_unique"],
                "aggregationPolicy": "period_rollup",
                "metricGrain": "tenant_day",
                "applicableTimeGrain": "period",
                "timeSemantics": {
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
                "calculationSemantics": {
                    "timeRollupPolicy": "NOT_COMPOSABLE",
                },
            }
        ],
    }

    result = validate_semantic_asset(asset, [])
    errors = {str(item.get("code") or "") for item in result["errors"]}
    warnings = {str(item.get("code") or "") for item in result["warnings"]}

    assert "NON_COMPOSABLE_NATIVE_WINDOW_REQUIRED" in errors
    assert "NON_COMPOSABLE_ALTERNATIVE_UNDECLARED" in warnings


def test_weighted_average_semantics_requires_governed_components_and_weight() -> None:
    asset = {
        "tableName": "daily_profile",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "value_sum"},
            {"columnName": "event_day"},
        ],
        "metrics": [
            {
                "metricKey": "weighted_value",
                "formula": "SUM(value_sum)",
                "sourceColumns": ["value_sum"],
                "aggregationPolicy": "period_recompute",
                "metricGrain": "tenant_day",
                "applicableTimeGrain": "period",
                "timeSemantics": {
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
                "calculationSemantics": {
                    "timeRollupPolicy": "WEIGHTED_AVERAGE",
                    "requiredComponents": ["value_sum", "weight_sum"],
                },
            }
        ],
    }

    errors = validation_codes(asset)

    assert "WEIGHTED_AVERAGE_WEIGHT_UNDECLARED" in errors
    assert "CALCULATION_COMPONENT_REF_MISSING" in errors


def test_every_metric_requires_a_supported_aggregation_policy() -> None:
    missing = {
        "tableName": "metric_table",
        "schemaColumns": [{"columnName": "metric_value"}],
        "metrics": [
            {
                "metricKey": "metric_value",
                "formula": "SUM(metric_value)",
                "sourceColumns": ["metric_value"],
            }
        ],
    }
    invalid = {
        **missing,
        "metrics": [
            {
                **missing["metrics"][0],
                "aggregationPolicy": "guess_from_metric_name",
            }
        ],
    }

    assert "METRIC_AGGREGATION_POLICY_UNDECLARED" in validation_codes(missing)
    assert "METRIC_AGGREGATION_POLICY_INVALID" in validation_codes(invalid)


def test_additive_period_rollup_cannot_publish_an_average_formula() -> None:
    asset = {
        "tableName": "metric_table",
        "schemaColumns": [{"columnName": "metric_value"}],
        "metrics": [
            {
                "metricKey": "metric_value",
                "formula": "AVG(metric_value)",
                "sourceColumns": ["metric_value"],
                "aggregationPolicy": "period_rollup",
            }
        ],
    }

    assert "PERIOD_ROLLUP_NON_ADDITIVE_FORMULA" in validation_codes(asset)


def test_time_semantics_must_be_complete_and_cannot_hide_unknown_as_zero() -> None:
    undeclared = {
        "tableName": "snapshot_fact",
        "schemaColumns": [{"columnName": "snapshot_value"}],
        "metrics": [
            {
                "metricKey": "snapshot_value",
                "formula": "SUM(snapshot_value)",
                "sourceColumns": ["snapshot_value"],
                "aggregationPolicy": "latest_value_only",
                "timeSemantics": {
                    "selectionPolicy": "undeclared",
                    "asOfPolicy": "undeclared",
                    "missingDataPolicy": "undeclared",
                    "zeroValuePolicy": "undeclared",
                },
            }
        ],
    }
    conflicting = {
        **undeclared,
        "metrics": [
            {
                **undeclared["metrics"][0],
                "timeSemantics": {
                    "selectionPolicy": "latest_as_of",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "zero_fill",
                    "zeroValuePolicy": "treat_as_missing",
                },
            }
        ],
    }

    assert {
        "METRIC_TIME_SEMANTICS_UNDECLARED",
        "LATEST_VALUE_AS_OF_POLICY_UNDECLARED",
    } <= validation_codes(undeclared)
    assert "METRIC_MISSING_ZERO_POLICY_CONFLICT" in validation_codes(conflicting)


def test_latest_value_time_semantics_accepts_explicit_as_of_and_unknown_policy() -> None:
    asset = {
        "tableName": "snapshot_fact",
        "timeColumn": "event_day",
        "schemaColumns": [{"columnName": "snapshot_value"}, {"columnName": "event_day"}],
        "metrics": [
            {
                "metricKey": "snapshot_value",
                "formula": "SUM(snapshot_value)",
                "sourceColumns": ["snapshot_value"],
                "aggregationPolicy": "latest_value_only",
                "metricGrain": "tenant_snapshot",
                "applicableTimeGrain": "day",
                "timeSemantics": {
                    "selectionPolicy": "latest_as_of",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert not validation_codes(asset)


def test_temporal_policy_matrix_rejects_contracts_without_an_executable_axis_or_grain() -> None:
    asset = {
        "tableName": "snapshot_fact",
        "schemaColumns": [{"columnName": "snapshot_value"}],
        "metrics": [
            {
                "metricKey": "snapshot_value",
                "formula": "SUM(snapshot_value)",
                "sourceColumns": ["snapshot_value"],
                "aggregationPolicy": "latest_value_only",
                "timeSemantics": {
                    "selectionPolicy": "latest_as_of",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert {
        "METRIC_TIME_COLUMN_UNDECLARED",
        "METRIC_GRAIN_UNDECLARED",
        "METRIC_APPLICABLE_TIME_GRAIN_UNDECLARED",
    } <= validation_codes(asset)


def test_ratio_of_sums_requires_a_real_ratio_of_aggregate_sums() -> None:
    asset = {
        "tableName": "ratio_fact",
        "timeColumn": "event_day",
        "schemaColumns": [
            {"columnName": "numerator"},
            {"columnName": "denominator"},
            {"columnName": "event_day"},
        ],
        "metrics": [
            {
                "metricKey": "ratio",
                "formula": "SUM(numerator)",
                "sourceColumns": ["numerator", "denominator"],
                "aggregationPolicy": "ratio_of_sums",
                "metricGrain": "tenant_event",
                "applicableTimeGrain": "period",
                "timeSemantics": {
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert "RATIO_OF_SUMS_FORMULA_INVALID" in validation_codes(asset)


def test_daily_value_and_period_window_reject_unimplemented_time_combinations() -> None:
    daily = {
        "tableName": "daily_fact",
        "timeColumn": "event_day",
        "schemaColumns": [{"columnName": "value"}, {"columnName": "event_day"}],
        "metrics": [
            {
                "metricKey": "value",
                "formula": "MAX(value)",
                "sourceColumns": ["value"],
                "aggregationPolicy": "daily_value_only",
                "metricGrain": "tenant_day",
                "applicableTimeGrain": "month",
                "timeSemantics": {
                    "selectionPolicy": "per_time_grain",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }
    period = {
        **daily,
        "metrics": [
            {
                **daily["metrics"][0],
                "aggregationPolicy": "period_rollup",
                "applicableTimeGrain": "period",
                "timeSemantics": {
                    **daily["metrics"][0]["timeSemantics"],
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_observation",
                },
            }
        ],
    }

    assert "DAILY_VALUE_TIME_GRAIN_INVALID" in validation_codes(daily)
    assert "METRIC_AS_OF_POLICY_NOT_EXECUTABLE" in validation_codes(period)


def test_metric_time_axis_override_is_supported_when_both_paths_use_the_sealed_axis() -> None:
    asset = {
        "tableName": "event_fact",
        "timeColumn": "partition_day",
        "schemaColumns": [
            {"columnName": "value"},
            {"columnName": "partition_day"},
            {"columnName": "business_day"},
        ],
        "metrics": [
            {
                "metricKey": "value",
                "formula": "SUM(value)",
                "sourceColumns": ["value"],
                "aggregationPolicy": "period_rollup",
                "metricGrain": "tenant_event",
                "applicableTimeGrain": "period",
                "timeColumn": "business_day",
                "timeSemantics": {
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert not validation_codes(asset)


def test_time_selection_policy_cannot_conflict_with_aggregation_policy() -> None:
    asset = {
        "tableName": "snapshot_fact",
        "schemaColumns": [{"columnName": "snapshot_value"}],
        "metrics": [
            {
                "metricKey": "snapshot_value",
                "formula": "SUM(snapshot_value)",
                "sourceColumns": ["snapshot_value"],
                "aggregationPolicy": "latest_value_only",
                "timeSemantics": {
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
            }
        ],
    }

    assert "METRIC_TIME_SELECTION_POLICY_CONFLICT" in validation_codes(asset)


def test_checked_in_catalog_satisfies_role_and_daily_value_contracts() -> None:
    assets = TopicAssetService(get_settings())
    governed_codes = {
        "NON_METRIC_COLUMN_HAS_METRIC_FORMULA",
        "DAILY_VALUE_TIME_GRAIN_UNDECLARED",
        "DAILY_VALUE_CROSS_DAY_AGGREGATION_FORBIDDEN",
        "RATIO_AGGREGATION_POLICY_UNDECLARED",
        "METRIC_AGGREGATION_POLICY_UNDECLARED",
        "METRIC_AGGREGATION_POLICY_INVALID",
        "PERIOD_ROLLUP_NON_ADDITIVE_FORMULA",
    }

    violations = []
    for topic in assets.all_topic_names():
        relationships = assets.load_relationships(topic)
        for item in assets.load_manifest(topic):
            table = str(item.get("tableName") or "")
            result = validate_semantic_asset(assets.load_table_asset(topic, table), relationships)
            for error in result["errors"]:
                if error.get("code") in governed_codes:
                    violations.append((topic, table, error))

    assert violations == []


def test_checked_in_asset_json_has_no_duplicate_contract_keys() -> None:
    assets = TopicAssetService(get_settings())
    violations: list[tuple[str, str]] = []

    for path in assets.root.glob("*/tables/*/asset.json"):
        duplicate_keys: list[str] = []

        def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    duplicate_keys.append(key)
                result[key] = value
            return result

        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
        violations.extend((str(path.relative_to(assets.root)), key) for key in duplicate_keys)

    assert violations == []
