from __future__ import annotations

import ast
from pathlib import Path

from merchant_ai.services.assets import (
    enforce_sample_evidence_governance,
    normalize_column_type,
    normalize_for_match,
    parse_doris_create_table_metadata,
    parse_identifier_list,
    parse_semantic_file_identity,
    parse_semantic_metric_identity,
    parse_semantic_relationship_entry_identity,
    parse_semantic_table_entry_identity,
    question_match_terms,
    sanitize_asset_path_part,
    sanitize_semantic_file_name,
    semantic_metric_source_columns,
)


def test_assets_module_has_no_pattern_engine_dependency() -> None:
    source_path = Path(__file__).parents[2] / "merchant_ai" / "services" / "assets.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    forbidden_imports = []
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(alias.name for alias in node.names if alias.name == "re")
        if isinstance(node, ast.ImportFrom) and node.module == "re":
            forbidden_imports.append(node.module)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "re"
        ):
            forbidden_calls.append(node.func.attr)

    assert forbidden_imports == []
    assert forbidden_calls == []


def test_semantic_virtual_identities_are_structural_and_fail_closed() -> None:
    assert parse_semantic_relationship_entry_identity(
        "semantic:trade:relationship:orders_to_lines",
        "",
    ) == ("entry", "trade", "orders_to_lines")
    assert parse_semantic_relationship_entry_identity(
        "",
        "topics/trade/relationships/index.json",
    ) == ("index", "trade", "")
    assert parse_semantic_metric_identity(
        "",
        "topics/trade/tables/fact/metrics/net_amount.json",
    ) == ("trade", "fact", "net_amount")
    assert parse_semantic_table_entry_identity(
        "semantic:trade:fact:field:merchant_id",
        "",
    ) == ("trade", "fact", "columns", "merchant_id")
    assert parse_semantic_file_identity(
        "",
        "topics/trade/tables/fact/metrics/index.json",
    ) == ("section", "trade", "fact", "metrics")

    assert parse_semantic_metric_identity("", "topics/trade/../metrics/x.json") is None
    assert parse_semantic_metric_identity("", "topics/trade/tables/fact/metrics/index.json") is None
    assert parse_semantic_table_entry_identity("semantic:trade::field:id", "") is None
    assert parse_semantic_file_identity("semantic:trade:fact:unknown", "") is None


def test_doris_ddl_metadata_uses_balanced_structure() -> None:
    ddl = """
        CREATE TABLE `fact` (
            `merchant_id` BIGINT,
            `event_time` DATETIME,
            `amount` DECIMAL(18, 2)
        )
        UNIQUE KEY (`merchant_id`, `event_time`)
        AUTO PARTITION BY RANGE (date_trunc('day', `event_time`)) ()
        DISTRIBUTED BY HASH (`merchant_id`) BUCKETS 16
    """

    assert parse_doris_create_table_metadata(ddl) == {
        "keyModel": "UNIQUE KEY",
        "primaryKeyColumns": ["merchant_id", "event_time"],
        "partitionColumns": ["event_time"],
        "bucketColumns": ["merchant_id"],
        "source": "show_create_table",
    }
    assert parse_identifier_list("date_trunc('day', event_time), merchant_id") == [
        "event_time",
        "merchant_id",
    ]
    assert parse_identifier_list("date_trunc('day, event_time)") == []
    assert parse_identifier_list("`unterminated") == []


def test_formula_scanning_respects_identifiers_and_ignores_sql_syntax() -> None:
    metric = {"formula": "SUM(net_amount) / NULLIF(SUM(order_count), 0)"}
    assert semantic_metric_source_columns(metric) == ["net_amount", "order_count"]
    assert semantic_metric_source_columns(
        {"sourceColumns": ["governed_amount"], "formula": "SUM(untrusted_fallback)"}
    ) == ["governed_amount"]


def test_text_normalization_and_sample_quarantine_preserve_contract_behavior() -> None:
    asset = {
        "sampleEvidenceGovernance": {"usableForSemanticDecisions": False},
        "semanticColumns": [
            {
                "columnName": "status",
                "evidence": "profiled; samples=[A, B]; reviewed",
                "sampleValues": ["A", "B"],
            }
        ],
    }
    sanitized = enforce_sample_evidence_governance(asset)

    assert sanitized["semanticColumns"][0]["evidence"] == "profiled; reviewed"
    assert "sampleValues" not in sanitized["semanticColumns"][0]
    assert normalize_column_type({"type": " DECIMAL (18, 2) "}) == "decimal"
    assert normalize_for_match(" Net\tAmount \n") == "netamount"
    assert sanitize_semantic_file_name("经营 说明?.md") == "经营_说明_.md"
    assert sanitize_asset_path_part("../tenant scope") == "tenant_scope"
    assert question_match_terms("GMV_7d 最近7天订单")[:2] == ["gmv_7d", "7"]
