import json

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import TopicBuildRequest
from merchant_ai.services.assets import TopicAssetService, TopicBuilderWorkflow
from merchant_ai.services.repositories import DorisRepository, write_json


class CapturingDb:
    def __init__(self):
        self.calls = []

    def query(self, sql, params=None):
        bound = list(params or [])
        self.calls.append((sql, bound))
        merchant_id = str(bound[0]) if bound else ""
        if "distinct_count" in sql:
            return [{"scanned_rows": 8, "distinct_count": 2}]
        if "enum_value" in sql:
            values = ["paid", "refund"] if merchant_id == "merchant-a" else ["created", "closed"]
            return [
                {"enum_value": value, "value_count": 4}
                for value in values
            ]
        return [{"tenant_key": merchant_id, "order_status": "paid"}]


def test_doris_topic_builder_reads_are_tenant_bound_and_cache_keys_do_not_cross(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_doris_select_ttl_seconds": 60,
        }
    )
    repository = DorisRepository(settings)
    database = CapturingDb()
    repository.db = database

    first = repository.sample_rows("orders", "merchant-a", "tenant_key", limit=10)
    second = repository.sample_rows("orders", "merchant-b", "tenant_key", limit=10)
    profile = repository.profile_enum_candidates(
        "orders",
        "merchant-a",
        "tenant_key",
        ["tenant_key", "order_status"],
        limit=10,
    )

    assert first == [{"tenant_key": "merchant-a", "order_status": "paid"}]
    assert second == [{"tenant_key": "merchant-b", "order_status": "paid"}]
    assert profile["order_status"]["values"] == ["paid", "refund"]
    assert "tenant_key" not in profile
    assert all("WHERE `tenant_key` = %s" in sql for sql, _params in database.calls)
    assert {tuple(params) for _sql, params in database.calls} == {("merchant-a",), ("merchant-b",)}


def test_doris_topic_builder_sampling_rejects_missing_or_unsafe_tenant_authority(tmp_path):
    repository = DorisRepository(
        get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "cache_enabled": False})
    )
    database = CapturingDb()
    repository.db = database

    with pytest.raises(ValueError):
        repository.sample_rows("orders", "merchant-a", "", limit=10)
    with pytest.raises(ValueError):
        repository.sample_rows("orders", "merchant-a", "tenant_key` OR 1=1", limit=10)
    with pytest.raises(ValueError):
        repository.profile_enum_candidates("orders", "", "tenant_key", ["order_status"], limit=10)

    assert database.calls == []


class TenantAwareBuilderDoris:
    def __init__(self):
        self.sample_calls = []
        self.profile_calls = []
        self.rows = [
            {
                "tenant_key": "merchant-a",
                "buyer_phone": "13800000001",
                "order_status": "paid",
            },
            {
                "tenant_key": "merchant-b",
                "buyer_phone": "13900000002",
                "order_status": "closed",
            },
        ]

    def show_full_columns(self, table):
        return [
            {
                "Field": "tenant_key",
                "Type": "varchar",
                "Comment": "tenant",
                "visibilityPolicy": {"level": "hidden"},
                "maskingPolicy": {"strategy": "full"},
            },
            {
                "Field": "buyer_phone",
                "Type": "varchar",
                "Comment": "phone",
                "visibilityPolicy": {"level": "restricted"},
                "maskingPolicy": {"strategy": "full"},
            },
            {"Field": "order_status", "Type": "varchar", "Comment": "status"},
        ]

    def show_create_table(self, table):
        return []

    def sample_rows(self, table, merchant_id, merchant_filter_column, limit=20):
        self.sample_calls.append((table, merchant_id, merchant_filter_column, limit))
        return [
            row
            for row in self.rows
            if str(row.get(merchant_filter_column)) == str(merchant_id)
        ][:limit]

    def profile_enum_candidates(
        self,
        table,
        merchant_id,
        merchant_filter_column,
        columns,
        limit=20,
    ):
        self.profile_calls.append((table, merchant_id, merchant_filter_column, list(columns), limit))
        values = [
            row["order_status"]
            for row in self.rows
            if row[merchant_filter_column] == merchant_id
        ]
        return {
            "order_status": {
                "values": values,
                "scannedRows": len(values),
                "distinctCount": len(set(values)),
                "reviewStatus": "UNREVIEWED",
            }
        }


class PromptCaptureLlm:
    configured = True

    def __init__(self):
        self.payloads = []

    def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, timeout_seconds=None):
        self.payloads.append(json.loads(user_prompt))
        return {}


def test_topic_builder_never_samples_other_tenants_or_prompts_tenant_and_sensitive_fields(tmp_path):
    settings = get_settings().model_copy(
        update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")}
    )
    doris = TenantAwareBuilderDoris()
    llm = PromptCaptureLlm()
    workflow = TopicBuilderWorkflow(settings, doris, TopicAssetService(settings), llm=llm)

    result = workflow.build(
        TopicBuildRequest(
            topic="orders",
            table_name="order_detail",
            merchant_id="merchant-a",
            merchant_filter_column="tenant_key",
        )
    )

    assert result["success"] is True
    sample_rows = json.loads(
        (settings.resolved_topic_path / "orders" / "pending" / "order_detail" / "sample_rows.json").read_text(
            encoding="utf-8"
        )
    )
    assert sample_rows == [
        {"tenant_key": "merchant-a", "buyer_phone": "13800000001", "order_status": "paid"}
    ]
    assert doris.sample_calls == [("order_detail", "merchant-a", "tenant_key", 20)]
    assert doris.profile_calls[0][1:3] == ("merchant-a", "tenant_key")
    assert "tenant_key" not in doris.profile_calls[0][3]

    prompt = llm.payloads[0]
    serialized_prompt = json.dumps(prompt, ensure_ascii=False)
    assert "merchant-b" not in serialized_prompt
    assert "13900000002" not in serialized_prompt
    assert "merchant-a" not in serialized_prompt
    assert "13800000001" not in serialized_prompt
    assert "tenant_key" not in serialized_prompt
    assert "buyer_phone" not in serialized_prompt
    assert prompt["sampleRows"] == [{"order_status": "paid"}]


def test_topic_builder_can_reuse_published_tenant_column_without_redeclaring_it(tmp_path):
    settings = get_settings().model_copy(
        update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")}
    )
    assets = TopicAssetService(settings)
    published_dir = assets.table_asset_dir("orders", "order_detail")
    write_json(
        published_dir / "asset.json",
        {
            "topic": "orders",
            "tableName": "order_detail",
            "merchantFilterColumn": "tenant_key",
            "rowAccessPolicy": {"filterColumn": "tenant_key", "required": True},
        },
    )
    doris = TenantAwareBuilderDoris()
    workflow = TopicBuilderWorkflow(
        settings,
        doris,
        assets,
        llm=type("DisabledLlm", (), {"configured": False})(),
    )

    result = workflow.build(
        TopicBuildRequest(topic="orders", table_name="order_detail", merchant_id="merchant-a")
    )

    assert result["success"] is True
    asset = json.loads(
        (settings.resolved_topic_path / "orders" / "pending" / "order_detail" / "asset.json").read_text(
            encoding="utf-8"
        )
    )
    assert asset["merchantFilterColumn"] == "tenant_key"
    assert asset["samplingGovernance"]["merchantFilterColumnSource"] == "PUBLISHED_SEMANTIC_ASSET"
    assert doris.sample_calls[0][2] == "tenant_key"
