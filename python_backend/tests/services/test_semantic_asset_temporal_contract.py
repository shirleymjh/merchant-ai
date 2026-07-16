import json

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, validate_semantic_asset


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
        "schemaColumns": [{"columnName": "daily_rate"}],
        "metrics": [
            {
                "metricKey": "daily_rate",
                "formula": "MAX(daily_rate)",
                "sourceColumns": ["daily_rate"],
                "aggregationPolicy": "daily_value_only",
                "applicableTimeGrain": "day",
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


def test_checked_in_catalog_satisfies_role_and_daily_value_contracts() -> None:
    assets = TopicAssetService(get_settings())
    governed_codes = {
        "NON_METRIC_COLUMN_HAS_METRIC_FORMULA",
        "DAILY_VALUE_TIME_GRAIN_UNDECLARED",
        "DAILY_VALUE_CROSS_DAY_AGGREGATION_FORBIDDEN",
        "RATIO_AGGREGATION_POLICY_UNDECLARED",
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
