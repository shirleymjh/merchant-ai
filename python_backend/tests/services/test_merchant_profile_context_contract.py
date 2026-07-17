from __future__ import annotations

import json

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, build_stable_topic_table_manifest
from merchant_ai.services.repositories import MerchantService
from merchant_ai.services.runtime_bindings import SemanticRuntimeBindingRegistry


def test_principal_profile_projects_only_declared_stable_context_columns():
    class Repository:
        def __init__(self):
            self.calls = []

        def query(self, sql, params):
            self.calls.append((sql, params))
            # A repository double may return extra data.  MerchantService must
            # still apply the binding allowlist before data reaches a prompt.
            return [
                {
                    "principal_key": "p-1",
                    "display_value": "Profile Alpha",
                    "merchant_type_name": "企业商户",
                    "brand_type_name": "品牌方",
                    "mobile": "13800000000",
                    "bank_account": "sensitive-account",
                    "order_gmv_amt_1d": 999,
                }
            ]

    repository = Repository()
    service = MerchantService(
        get_settings(),
        repository,
        {
            "table": "table_alpha",
            "lookupColumn": "principal_key",
            "idColumn": "principal_key",
            "displayColumns": ["display_value"],
            "contextColumns": ["merchant_type_name", "brand_type_name"],
        },
    )

    merchant = service.current_merchant("p-1")

    assert repository.calls == [
        (
            "SELECT `principal_key`, `display_value`, `merchant_type_name`, `brand_type_name` "
            "FROM `table_alpha` WHERE `principal_key` = %s LIMIT 1",
            ["p-1"],
        )
    ]
    assert merchant.rows == {
        "merchant_type_name": "企业商户",
        "brand_type_name": "品牌方",
    }
    rendered = merchant.profile_markdown()
    assert "企业商户" in rendered
    assert "品牌方" in rendered
    assert "13800000000" not in rendered
    assert "sensitive-account" not in rendered
    assert "999" not in rendered


def test_principal_profile_drops_declared_columns_missing_from_live_schema():
    class Repository:
        def __init__(self):
            self.calls = []

        @staticmethod
        def show_full_columns(_table):
            return [
                {"Field": "principal_key"},
                {"Field": "display_value"},
                {"Field": "merchant_type_name"},
            ]

        def query(self, sql, params):
            self.calls.append((sql, params))
            return [
                {
                    "principal_key": "p-1",
                    "display_value": "Profile Alpha",
                    "merchant_type_name": "企业商户",
                }
            ]

    repository = Repository()
    service = MerchantService(
        get_settings(),
        repository,
        {
            "table": "table_alpha",
            "lookupColumn": "principal_key",
            "idColumn": "principal_key",
            "displayColumns": ["display_value"],
            "contextColumns": ["merchant_type_name", "removed_column"],
        },
    )

    merchant = service.current_merchant("p-1")

    assert repository.calls == [
        (
            "SELECT `principal_key`, `display_value`, `merchant_type_name` "
            "FROM `table_alpha` WHERE `principal_key` = %s LIMIT 1",
            ["p-1"],
        )
    ]
    assert merchant.rows == {"merchant_type_name": "企业商户"}


def test_principal_profile_binding_rejects_unsafe_context_column(tmp_path):
    asset_path = tmp_path / "身份信息" / "tables" / "dim_profile" / "asset.json"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text(
        json.dumps(
            {
                "status": "PUBLISHED",
                "tableName": "dim_profile",
                "runtimeBindings": [
                    {
                        "role": "principal_profile",
                        "lookupColumn": "merchant_id",
                        "idColumn": "merchant_id",
                        "displayColumns": ["company_name"],
                        "contextColumns": ["merchant_type_name", "mobile, bank_account"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path)})

    assert SemanticRuntimeBindingRegistry(settings).resolve("principal_profile") == {}


def test_published_principal_profile_declares_a_small_non_sensitive_stable_tag_allowlist():
    binding = SemanticRuntimeBindingRegistry(get_settings()).resolve("principal_profile")
    context_columns = [str(item) for item in binding.get("contextColumns") or []]
    forbidden = {
        "mobile",
        "refund_mobile",
        "contact_mobile",
        "license_id",
        "corporation_idcard",
        "contact_idcard",
        "bank_account",
        "company_address",
        "business_address",
        "send_address",
        "refnd_address",
    }

    assert 1 <= len(context_columns) <= 12
    assert forbidden.isdisjoint(context_columns)
    assert {"merchant_type_name", "brand_type_name"}.issubset(context_columns)


def test_operating_profile_is_a_cross_topic_navigation_hub_with_metric_lineage():
    assets = TopicAssetService(get_settings())
    profile_asset = assets.load_table_asset("经营画像", "ads_merchant_profile")
    configured = [item for item in (profile_asset.get("dailyReportProfile") or {}).get("metrics") or [] if isinstance(item, dict)]
    definitions = {
        str(item.get("metricKey") or ""): item
        for item in profile_asset.get("metrics") or []
        if isinstance(item, dict) and item.get("metricKey")
    }

    assert (profile_asset.get("tableUsageProfile") or {}).get("topicRole") == "PROFILE"
    assert profile_asset.get("questionCategory") == "UNKNOWN"
    assert configured
    assert all(str(item.get("metricRef") or "") in definitions for item in configured)

    # PROFILE is an overview/navigation layer.  Every curated card must name
    # its owning business Topic; the profile table itself must not become the
    # semantic owner merely because it materializes several domains together.
    owner_topics = {str(item.get("ownerTopic") or "") for item in configured}
    assert "" not in owner_topics
    assert len(owner_topics) >= 2


def test_business_topic_l0_does_not_implicitly_expand_the_operating_profile_hub():
    assets = TopicAssetService(get_settings())

    manifest = build_stable_topic_table_manifest(assets, ["客服工单"])

    assert [item["topic"] for item in manifest["topics"]] == ["客服工单"]
    assert "dwm_cs_ticket_detail_di" in {item["table"] for item in manifest["tables"]}
    assert "ads_merchant_profile" not in {item["table"] for item in manifest["tables"]}
