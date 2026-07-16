import json

from merchant_ai.config import get_settings
from merchant_ai.services.assets import HybridRecallService, TopicAssetService
from merchant_ai.services.recall_index import RecallIndexManager
from merchant_ai.services.semantic_publish import SemanticPublishCoordinator


class FailingStageAdapter:
    def stage(self, docs, manifest):
        return {
            "success": False,
            "mode": "es",
            "transactionMode": "physical_index_alias_swap",
            "errorCode": "ES_STAGE_DOC_COUNT_MISMATCH",
        }


class FailingCommitAdapter:
    def __init__(self):
        self.aborted = False

    def stage(self, docs, manifest):
        return {
            "success": True,
            "mode": "es",
            "transactionMode": "physical_index_alias_swap",
            "alias": "recall",
            "physicalIndex": "recall__candidate",
            "previousState": {"kind": "alias", "indices": ["recall__old"]},
            "committed": False,
        }

    def commit_stage(self, stage):
        return {"success": False, "errorCode": "ES_ALIAS_SWAP_FAILED"}

    def abort_stage(self, stage):
        self.aborted = True
        return {"success": True, "aborted": True}


class SuccessfulCommitAdapter(FailingCommitAdapter):
    def commit_stage(self, stage):
        stage["committed"] = True
        return {"success": True, "alias": "recall", "physicalIndex": "recall__candidate"}

    def finalize_stage(self, stage):
        return {"success": True, "removed": ["recall__old"], "retained": []}


class WritingGovernance:
    def __init__(self, settings, topic_assets):
        self.settings = settings
        self.topic_assets = topic_assets

    def after_publish(self, topic, table_name, reviewer, review_note):
        target = self.topic_assets.table_asset_dir(topic, table_name)
        (target / "semantic_version_manifest.json").write_text(
            json.dumps({"activeVersion": "new"}),
            encoding="utf-8",
        )
        governance_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table_name
        governance_dir.mkdir(parents=True, exist_ok=True)
        (governance_dir / "publish-new.json").write_text("{}", encoding="utf-8")
        return {"success": True, "status": "GOVERNED_PUBLISHED"}


def _setup_runtime(tmp_path):
    settings = get_settings().model_copy(
        update={
            "es_enabled": False,
            "es_index": "recall",
            "topic_path": str(tmp_path / "topics"),
            "rule_knowledge_path": str(tmp_path / "rules"),
            "harness_workspace_path": str(tmp_path / "workspace"),
        }
    )
    settings.resolved_rule_knowledge_path.mkdir(parents=True, exist_ok=True)
    topic = "经营画像"
    table = "ads_merchant_profile"
    topic_dir = settings.resolved_topic_path / topic
    target = topic_dir / "tables" / table
    pending = topic_dir / "pending" / table
    target.mkdir(parents=True, exist_ok=True)
    pending.mkdir(parents=True, exist_ok=True)
    (topic_dir / "manifest.json").write_text(
        json.dumps([{"tableName": table, "status": "PUBLISHED"}], ensure_ascii=False),
        encoding="utf-8",
    )
    old_asset = {
        "topic": topic,
        "tableName": table,
        "businessName": "old",
        "metrics": [{"metricKey": "return_rate", "businessName": "old return rate", "formula": "old_formula"}],
    }
    new_asset = {
        "topic": topic,
        "tableName": table,
        "businessName": "new",
        "metrics": [{"metricKey": "return_rate", "businessName": "new return rate", "formula": "new_formula"}],
    }
    (target / "asset.json").write_text(json.dumps(old_asset, ensure_ascii=False), encoding="utf-8")
    (pending / "asset.json").write_text(json.dumps(new_asset, ensure_ascii=False), encoding="utf-8")
    (target / "semantic_version_manifest.json").write_text(
        json.dumps({"activeVersion": "old"}),
        encoding="utf-8",
    )
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    governance_dir.mkdir(parents=True, exist_ok=True)
    (governance_dir / "preflight.json").write_text("{}", encoding="utf-8")
    topic_assets = TopicAssetService(settings)
    provider = HybridRecallService(settings, topic_assets)
    manager = RecallIndexManager(settings, provider)
    manager.rebuild(documents=provider._load_documents())
    return settings, topic_assets, provider, manager, topic, table


def test_publish_does_not_advance_filesystem_or_manifest_when_es_stage_fails(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    manifest_path = old_manager.manifest_path
    old_manifest = manifest_path.read_text(encoding="utf-8")
    old_asset = (topic_assets.table_asset_dir(topic, table) / "asset.json").read_text(encoding="utf-8")
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider, es_adapter=FailingStageAdapter())
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", {"publishable": True})

    assert result["success"] is False
    assert result["status"] == "PUBLISH_PENDING_INDEX_RETRY"
    assert result["recallIndex"]["activeManifestAdvanced"] is False
    assert manifest_path.read_text(encoding="utf-8") == old_manifest
    assert (topic_assets.table_asset_dir(topic, table) / "asset.json").read_text(encoding="utf-8") == old_asset


def test_publish_restores_old_filesystem_when_alias_swap_fails(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    manifest_path = old_manager.manifest_path
    old_manifest = manifest_path.read_text(encoding="utf-8")
    target = topic_assets.table_asset_dir(topic, table)
    old_asset = json.loads((target / "asset.json").read_text(encoding="utf-8"))
    settings.es_enabled = True
    adapter = FailingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", {"publishable": True})

    assert result["success"] is False
    assert result["status"] == "PUBLISH_PENDING_INDEX_RETRY"
    assert result["compensation"] == {"success": True, "status": "OLD_ACTIVE_RESTORED"}
    assert adapter.aborted is True
    assert json.loads((target / "asset.json").read_text(encoding="utf-8")) == old_asset
    assert json.loads((target / "semantic_version_manifest.json").read_text(encoding="utf-8"))["activeVersion"] == "old"
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    assert (governance_dir / "preflight.json").exists()
    assert not (governance_dir / "publish-new.json").exists()
    assert manifest_path.read_text(encoding="utf-8") == old_manifest


def test_publish_advances_filesystem_and_manifest_only_after_index_commit(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    old_version = json.loads(old_manager.manifest_path.read_text(encoding="utf-8"))["indexVersion"]
    settings.es_enabled = True
    manager = RecallIndexManager(settings, provider, es_adapter=SuccessfulCommitAdapter())
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", {"publishable": True})

    assert result["success"] is True
    assert result["status"] == "PUBLISHED"
    assert result["publishState"] == "ACTIVE"
    assert result["recallIndex"]["activeManifestAdvanced"] is True
    assert result["recallIndex"]["indexVersion"] != old_version
    target = topic_assets.table_asset_dir(topic, table)
    assert json.loads((target / "asset.json").read_text(encoding="utf-8"))["businessName"] == "new"
    assert json.loads(manager.manifest_path.read_text(encoding="utf-8"))["indexVersion"] == result["recallIndex"]["indexVersion"]
