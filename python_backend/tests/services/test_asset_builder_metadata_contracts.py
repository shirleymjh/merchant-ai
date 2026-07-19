import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack, TopicBuildRequest
from merchant_ai.services.assets import (
    PlanningAssetPackBuilder,
    TopicAssetService,
    TopicBuilderWorkflow,
    normalize_table_usage_profile,
    semantic_asset_builder_tool,
    semantic_catalog_conflict_detection,
)
from merchant_ai.services.query import semantic_table_access_hint


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class NeutralDoris:
    def __init__(self):
        self.sample_calls = 0

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

    def show_indexes(self, table):
        return [
            {
                "Key_name": "idx_entity_key",
                "Column_name": "entity_key",
                "Index_type": "INVERTED",
                "Properties": "parser=none",
            }
        ]

    def show_partitions(self, table):
        return [
            {"PartitionName": "p20260701", "Buckets": 4},
            {"PartitionName": "p20260702", "Buckets": 4},
        ]

    def sample_rows(self, table, merchant_id, merchant_filter_column, limit=20):
        self.sample_calls += 1
        rows = [{"tenant_key": "t-1", "event_day": "2026-07-01", "entity_key": "e-1", "value_x": 3.0}]
        return [row for row in rows if str(row.get(merchant_filter_column)) == str(merchant_id)][:limit]


def test_topic_builder_metric_schema_requires_generic_execution_semantics() -> None:
    parameters = semantic_asset_builder_tool().parameters
    metric_schema = parameters["properties"]["metrics"]["items"]

    assert metric_schema["additionalProperties"] is False
    assert {
        "aggregationPolicy",
        "metricGrain",
        "applicableTimeGrain",
        "timeSemantics",
    } <= set(metric_schema["required"])
    assert set(metric_schema["properties"]["aggregationPolicy"]["enum"]) == {
        "period_rollup",
        "period_recompute",
        "latest_value_only",
        "daily_value_only",
        "ratio_of_sums",
    }

    time_semantics = metric_schema["properties"]["timeSemantics"]
    assert time_semantics["additionalProperties"] is False
    assert set(time_semantics["required"]) == {
        "selectionPolicy",
        "asOfPolicy",
        "missingDataPolicy",
        "zeroValuePolicy",
    }
    assert "latest_as_of" in time_semantics["properties"]["selectionPolicy"]["enum"]
    assert "latest_available_partition" in time_semantics["properties"]["asOfPolicy"]["enum"]
    assert "disclose_unknown" in time_semantics["properties"]["missingDataPolicy"]["enum"]
    assert "preserve_observed_zero" in time_semantics["properties"]["zeroValuePolicy"]["enum"]


def test_access_hint_reads_published_physical_metadata_without_semantic_inference() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="fact_x",
                table="fact_x",
                columns=["tenant_key", "entity_key", "event_day"],
                metadata={
                    "merchantFilterColumn": "tenant_key",
                    "physicalMetadata": {
                        "bucketColumns": ["tenant_key"],
                        "partitionColumns": ["event_day"],
                        "invertedIndexColumns": ["entity_key"],
                    },
                },
            )
        ]
    )

    hint = semantic_table_access_hint(
        pack,
        "fact_x",
        {"tenant_key", "entity_key", "event_day"},
    )

    assert hint["distributionKeys"] == ["tenant_key"]
    assert hint["invertedIndexes"] == ["entity_key"]
    assert hint["fallbackFilters"] == ["tenant_key", "event_day"]


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

    result = workflow.build(
        TopicBuildRequest(
            topic="neutral",
            table_name="opaque_source",
            merchant_id="t-1",
            merchant_filter_column="tenant_key",
        )
    )

    asset = json.loads(Path(result["path"], "asset.json").read_text(encoding="utf-8"))
    columns = {item["columnName"]: item for item in asset["semanticColumns"]}
    assert asset["timeColumn"] == ""
    assert asset["merchantFilterColumn"] == "tenant_key"
    assert asset["rowAccessPolicy"]["filterColumn"] == "tenant_key"
    assert asset["samplingGovernance"]["merchantFilterColumnSource"] == "ADMIN_REQUEST"
    assert asset["dataGrain"] == "UNDECLARED"
    assert asset["tableUsageProfile"]["queryableByAgent"] is False
    assert asset["tableUsageProfile"]["businessLayer"] == "UNDECLARED"
    assert columns["tenant_key"]["semanticRole"] == "UNDECLARED"
    assert columns["event_day"]["semanticRole"] == "UNDECLARED"
    assert columns["value_x"]["semanticRole"] == "UNDECLARED"
    assert columns["entity_key"]["defaultVisible"] is False
    assert asset["physicalMetadata"]["partitionColumns"] == ["event_day"]
    assert asset["physicalMetadata"]["bucketCount"] == 4
    assert asset["physicalMetadata"]["invertedIndexColumns"] == ["entity_key"]
    assert asset["physicalMetadata"]["ddlHash"]


def test_topic_builder_fails_before_sampling_without_governed_tenant_column(tmp_path):
    settings = get_settings().model_copy(
        update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")}
    )
    doris = NeutralDoris()

    class CountingLlm:
        configured = True

        def __init__(self):
            self.calls = 0

        def tool_json_chat(self, *args, **kwargs):
            self.calls += 1
            return {}

    llm = CountingLlm()
    workflow = TopicBuilderWorkflow(settings, doris, TopicAssetService(settings), llm=llm)

    result = workflow.build(TopicBuildRequest(topic="neutral", table_name="opaque_source", merchant_id="t-1"))

    assert result["success"] is False
    assert result["code"] == "TENANT_FILTER_COLUMN_REQUIRED"
    assert doris.sample_calls == 0
    assert llm.calls == 0


def test_asset_pack_entity_keys_come_from_declared_semantic_roles(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    table_dir = tmp_path / "topics" / "neutral" / "tables" / "opaque_source"
    write_json(tmp_path / "topics" / "neutral" / "manifest.json", [{"tableName": "opaque_source"}])
    write_json(
        table_dir / "asset.json",
        {
            "topic": "neutral",
            "tableName": "opaque_source",
            "status": "PUBLISHED",
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
                "status": "PUBLISHED",
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
