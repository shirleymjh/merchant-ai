import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import PlanningAssetPack, TopicBuildRequest
from merchant_ai.services.assets import (
    PlanningAssetPackBuilder,
    TopicAssetService,
    TopicBuilderWorkflow,
    normalize_table_usage_profile,
    semantic_catalog_conflict_detection,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class NeutralDoris:
    def show_full_columns(self, table):
        return [
            {"Field": "tenant_key", "Type": "varchar", "Comment": "tenant"},
            {"Field": "event_day", "Type": "date", "Comment": "event date"},
            {"Field": "entity_key", "Type": "varchar", "Comment": "entity"},
            {"Field": "value_x", "Type": "decimal(18,2)", "Comment": "value"},
        ]

    def show_create_table(self, table):
        return [
            {
                "Table": table,
                "Create Table": """
CREATE TABLE `opaque_source` (
  `tenant_key` varchar(64),
  `event_day` date,
  `entity_key` varchar(64),
  `value_x` decimal(18,2)
)
DUPLICATE KEY(`tenant_key`, `entity_key`)
PARTITION BY RANGE(`event_day`)()
DISTRIBUTED BY HASH(`tenant_key`) BUCKETS 4
""",
            }
        ]

    def sample_rows(self, table, merchant_id, limit=20):
        return [{"tenant_key": "t-1", "event_day": "2026-07-01", "entity_key": "e-1", "value_x": 3.0}]


def test_builder_uses_physical_metadata_and_leaves_business_contracts_undeclared(tmp_path):
    settings = get_settings().model_copy(
        update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")}
    )
    workflow = TopicBuilderWorkflow(
        settings,
        NeutralDoris(),
        TopicAssetService(settings),
        llm=type("DisabledLlm", (), {"configured": False})(),
    )

    result = workflow.build(TopicBuildRequest(topic="neutral", table_name="opaque_source", merchant_id="t-1"))

    asset = json.loads(Path(result["path"], "asset.json").read_text(encoding="utf-8"))
    columns = {item["columnName"]: item for item in asset["semanticColumns"]}
    assert asset["timeColumn"] == "event_day"
    assert asset["merchantFilterColumn"] == ""
    assert asset["rowAccessPolicy"] == {}
    assert asset["dataGrain"] == "UNDECLARED"
    assert asset["tableUsageProfile"]["queryableByAgent"] is False
    assert asset["tableUsageProfile"]["businessLayer"] == "UNDECLARED"
    assert columns["tenant_key"]["semanticRole"] == "KEY"
    assert columns["event_day"]["semanticRole"] == "TIME"
    assert columns["value_x"]["semanticRole"] == "UNDECLARED"
    assert columns["entity_key"]["defaultVisible"] is False


def test_asset_pack_entity_keys_come_from_declared_semantic_roles(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    table_dir = tmp_path / "topics" / "neutral" / "tables" / "opaque_source"
    write_json(tmp_path / "topics" / "neutral" / "manifest.json", [{"tableName": "opaque_source"}])
    write_json(
        table_dir / "asset.json",
        {
            "topic": "neutral",
            "tableName": "opaque_source",
            "tableUsageProfile": {
                "contractStatus": "APPROVED",
                "businessLayer": "CURATED",
                "queryableByAgent": True,
                "authorityLevel": 80,
                "topicRole": "DETAIL",
            },
            "semanticColumns": [
                {"columnName": "tenant_key", "semanticRole": "TENANT_KEY"},
                {"columnName": "event_day", "semanticRole": "TIME"},
                {"columnName": "entity_key", "semanticRole": "ENTITY_KEY"},
            ],
            "schemaColumns": [
                {"columnName": "tenant_key"},
                {"columnName": "event_day"},
                {"columnName": "entity_key"},
            ],
        },
    )
    builder = PlanningAssetPackBuilder(TopicAssetService(settings))
    pack = PlanningAssetPack()

    builder._append_table_assets(pack, "neutral", "opaque_source")

    assert [item.key for item in pack.entity_keys] == ["entity_key"]


def test_table_usage_and_global_alias_ownership_fail_closed_without_metadata(tmp_path):
    assert normalize_table_usage_profile({}, "ads_name_that_looks_curated")["queryableByAgent"] is False
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    for topic, table, metric_key in (("a", "x", "m1"), ("b", "y", "m2")):
        write_json(
            tmp_path / "topics" / topic / "tables" / table / "asset.json",
            {
                "topic": topic,
                "tableName": table,
                "metrics": [{"metricKey": metric_key, "formula": "1", "aliases": ["shared label"]}],
            },
        )
    assets = TopicAssetService(settings)
    assert semantic_catalog_conflict_detection(assets)["status"] == "passed"

    candidate = {
        "topic": "b",
        "tableName": "y",
        "metrics": [
            {
                "metricKey": "m2",
                "formula": "1",
                "aliases": ["shared label"],
                "aliasConflictScope": "GLOBAL",
            }
        ],
    }
    first = assets.load_table_asset("a", "x")
    first["metrics"][0]["aliasConflictScope"] = "GLOBAL"
    write_json(tmp_path / "topics" / "a" / "tables" / "x" / "asset.json", first)
    assets = TopicAssetService(settings)

    report = semantic_catalog_conflict_detection(assets, "b", "y", candidate)

    assert any(item["type"] == "global_ratio_alias_conflict" for item in report["conflicts"])
