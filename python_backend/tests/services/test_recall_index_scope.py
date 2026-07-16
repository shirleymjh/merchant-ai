import json

from merchant_ai.config import get_settings
from merchant_ai.models import RecallItem
from merchant_ai.services.recall_index import RecallIndexManager


class MutableDocumentProvider:
    def __init__(self, settings):
        self.settings = settings
        self.version = "v1"

    def clear_cache(self):
        return None

    def _load_documents(self):
        return [
            RecallItem(
                doc_id="metric:profile:return_rate",
                title="return rate",
                content=self.version,
                source_type="SEMANTIC_METRIC",
                topic="经营画像",
                table="ads_merchant_profile",
                metadata={
                    "semanticRefId": "metric:profile:return_rate",
                    "semanticPath": "topics/经营画像/tables/ads_merchant_profile/asset.json#metric:return_rate",
                },
            ),
            RecallItem(
                doc_id="metric:trade:gmv",
                title="gmv",
                content="stable",
                source_type="SEMANTIC_METRIC",
                topic="电商交易",
                table="dwm_trade_order_detail_di",
                metadata={
                    "semanticRefId": "metric:trade:gmv",
                    "semanticPath": "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json#metric:gmv",
                },
            ),
        ]


def test_scoped_rebuild_preserves_global_manifest(tmp_path):
    settings = get_settings().model_copy(
        update={
            "es_enabled": False,
            "harness_workspace_path": str(tmp_path / "workspace"),
            "topic_path": str(tmp_path / "topics"),
        }
    )
    provider = MutableDocumentProvider(settings)
    manager = RecallIndexManager(settings, provider)
    manager.rebuild(changed_only=True)

    provider.version = "v2"
    result = manager.rebuild(changed_only=True, topic="经营画像", table_name="ads_merchant_profile")
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))

    assert result["updatedRefs"] == ["经营画像/tables/ads_merchant_profile/asset.json#metric:return_rate"]
    assert manifest["docCount"] == 2
    assert {item["docId"] for item in manifest["docs"]} == {
        "metric:profile:return_rate",
        "metric:trade:gmv",
    }
