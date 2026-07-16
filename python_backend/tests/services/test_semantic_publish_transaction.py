import json

import pytest

from merchant_ai.config import get_settings
from merchant_ai.services.assets import HybridRecallService, TopicAssetService, semantic_candidate_source_hash
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


class CountingCommitAdapter(SuccessfulCommitAdapter):
    def __init__(self):
        super().__init__()
        self.stage_calls = 0

    def stage(self, docs, manifest):
        self.stage_calls += 1
        return super().stage(docs, manifest)


class MutatingStageAdapter(CountingCommitAdapter):
    def __init__(self, pending_asset):
        super().__init__()
        self.pending_asset = pending_asset

    def stage(self, docs, manifest):
        result = super().stage(docs, manifest)
        payload = json.loads(self.pending_asset.read_text(encoding="utf-8"))
        payload["businessName"] = "changed-during-index-stage"
        self.pending_asset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result


class WritingGovernance:
    def __init__(self, settings, topic_assets):
        self.settings = settings
        self.topic_assets = topic_assets

    def preflight_publish(self, topic, table_name):
        return _valid_preflight(self.topic_assets, topic, table_name)

    def after_publish(self, topic, table_name, reviewer, review_note):
        target = self.topic_assets.table_asset_dir(topic, table_name)
        (target / "semantic_version_manifest.json").write_text(
            json.dumps({"activeVersion": "new"}),
            encoding="utf-8",
        )
        governance_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table_name
        governance_dir.mkdir(parents=True, exist_ok=True)
        (governance_dir / "publish-new.json").write_text("{}", encoding="utf-8")
        return {
            "success": True,
            "status": "GOVERNED_PUBLISHED",
            "topic": topic,
            "tableName": table_name,
            "releaseGate": {"publishable": True},
        }


class RejectingPostPublishGovernance(WritingGovernance):
    def after_publish(self, topic, table_name, reviewer, review_note):
        result = super().after_publish(topic, table_name, reviewer, review_note)
        result["status"] = "POST_PUBLISH_RELEASE_GATE_FAILED"
        result["releaseGate"] = {"publishable": False, "blockingReasons": ["fault-injected"]}
        return result


class RejectingAuthoritativePreflightGovernance(WritingGovernance):
    def preflight_publish(self, topic, table_name):
        result = super().preflight_publish(topic, table_name)
        result.update({"publishable": False, "status": "PREFLIGHT_FAILED"})
        return result


class MutatingCandidateCoordinator(SemanticPublishCoordinator):
    def _candidate_documents(self, work_dir, topic, table_name, reviewer, review_note):
        result = super()._candidate_documents(work_dir, topic, table_name, reviewer, review_note)
        pending = self.settings.resolved_topic_path / topic / "pending" / table_name / "asset.json"
        payload = json.loads(pending.read_text(encoding="utf-8"))
        payload["businessName"] = "changed-after-candidate-copy"
        pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result


def _valid_preflight(topic_assets, topic, table_name):
    pending = topic_assets.root / topic / "pending" / table_name
    source_hash = semantic_candidate_source_hash(pending)
    return {
        "success": True,
        "publishable": True,
        "status": "PREFLIGHT_PASSED",
        "topic": topic,
        "tableName": table_name,
        "pendingSourceHash": source_hash,
        "semanticCatalogVersion": {"sourceHash": source_hash},
    }


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

    result = coordinator.publish_approved(topic, table, "reviewer", "note", _valid_preflight(topic_assets, topic, table))

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
    pending_path = settings.resolved_topic_path / topic / "pending" / table / "asset.json"
    old_pending = pending_path.read_text(encoding="utf-8")
    topic_manifest_path = settings.resolved_topic_path / topic / "manifest.json"
    old_topic_manifest = topic_manifest_path.read_text(encoding="utf-8")
    settings.es_enabled = True
    adapter = FailingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", _valid_preflight(topic_assets, topic, table))

    assert result["success"] is False
    assert result["status"] == "PUBLISH_PENDING_INDEX_RETRY"
    assert result["compensation"] == {"success": True, "status": "OLD_ACTIVE_RESTORED"}
    assert adapter.aborted is True
    assert json.loads((target / "asset.json").read_text(encoding="utf-8")) == old_asset
    assert pending_path.read_text(encoding="utf-8") == old_pending
    assert topic_manifest_path.read_text(encoding="utf-8") == old_topic_manifest
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

    result = coordinator.publish_approved(topic, table, "reviewer", "note", _valid_preflight(topic_assets, topic, table))

    assert result["success"] is True
    assert result["status"] == "PUBLISHED"
    assert result["publishState"] == "ACTIVE"
    assert result["recallIndex"]["activeManifestAdvanced"] is True
    assert result["recallIndex"]["indexVersion"] != old_version
    target = topic_assets.table_asset_dir(topic, table)
    assert json.loads((target / "asset.json").read_text(encoding="utf-8"))["businessName"] == "new"
    assert json.loads((target / "asset.json").read_text(encoding="utf-8"))["status"] == "PUBLISHED"
    pending = settings.resolved_topic_path / topic / "pending" / table / "asset.json"
    assert json.loads(pending.read_text(encoding="utf-8")).get("status") != "PUBLISHED"
    assert json.loads(manager.manifest_path.read_text(encoding="utf-8"))["indexVersion"] == result["recallIndex"]["indexVersion"]


def test_direct_publish_materializes_active_asset_and_registers_new_table_without_mutating_candidate(tmp_path):
    settings = get_settings().model_copy(
        update={
            "topic_path": str(tmp_path / "topics"),
            "rule_knowledge_path": str(tmp_path / "rules"),
            "harness_workspace_path": str(tmp_path / "workspace"),
        }
    )
    topic = "new-topic"
    table = "new_fact"
    pending = settings.resolved_topic_path / topic / "pending" / table
    pending.mkdir(parents=True, exist_ok=True)
    candidate = {
        "topic": topic,
        "tableName": table,
        "tableComment": "new fact",
        "timeColumn": "event_day",
        "merchantFilterColumn": "tenant_id",
        "schemaColumns": [{"columnName": "amount"}],
        "metrics": [
            {
                "metricKey": "amount_total",
                "businessName": "amount total",
                "formula": "SUM(amount)",
                "sourceColumns": ["amount"],
            }
        ],
    }
    (pending / "asset.json").write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
    assets = TopicAssetService(settings)

    result = assets.publish(topic, table, True, "reviewer", "approved")

    assert result["success"] is True
    assert result["manifestChanged"] is True
    assert json.loads((pending / "asset.json").read_text(encoding="utf-8")) == candidate
    active = json.loads((assets.table_asset_dir(topic, table) / "asset.json").read_text(encoding="utf-8"))
    assert active["status"] == "PUBLISHED"
    manifest = assets.load_manifest(topic)
    assert manifest == [
        {
            "tableName": table,
            "tableComment": "new fact",
            "dataGrain": "",
            "timeColumn": "event_day",
            "merchantFilterColumn": "tenant_id",
            "freshnessType": "",
            "supportsDetail": True,
            "supportsMetrics": True,
            "preferredFor": ["DETAIL", "METRIC", "TOPN", "GROUP_AGG"],
            "metricCount": 1,
            "ruleCount": 0,
            "status": "PUBLISHED",
        }
    ]
    metric_docs = [
        item
        for item in HybridRecallService(settings, assets)._load_documents()
        if item.source_type == "SEMANTIC_METRIC" and item.table == table
    ]
    assert len(metric_docs) == 1
    assert metric_docs[0].metadata["status"] == "PUBLISHED"


@pytest.mark.parametrize(
    ("invalid_kind", "expected_error"),
    [
        ("failed", "PREFLIGHT_FAILED"),
        ("cross_table", "PREFLIGHT_SCOPE_MISMATCH"),
        ("missing_binding", "PREFLIGHT_BINDING_INVALID"),
        ("stale_hash", "PREFLIGHT_STALE"),
    ],
)
def test_untrusted_preflight_never_stages_or_activates_semantics(tmp_path, invalid_kind, expected_error):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    old_index_manifest = old_manager.manifest_path.read_text(encoding="utf-8")
    target_asset = topic_assets.table_asset_dir(topic, table) / "asset.json"
    old_active = target_asset.read_text(encoding="utf-8")
    preflight = _valid_preflight(topic_assets, topic, table)
    if invalid_kind == "failed":
        preflight["publishable"] = False
        preflight["status"] = "PREFLIGHT_FAILED"
    elif invalid_kind == "cross_table":
        preflight["tableName"] = "another_table"
    elif invalid_kind == "missing_binding":
        preflight.pop("pendingSourceHash")
        preflight.pop("semanticCatalogVersion")
    else:
        preflight["pendingSourceHash"] = "0" * 64
        preflight["semanticCatalogVersion"]["sourceHash"] = "0" * 64
    settings.es_enabled = True
    adapter = CountingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", preflight)

    assert result["success"] is False
    assert result["status"] == expected_error
    assert result["errorCode"] == expected_error
    assert adapter.stage_calls == 0
    assert target_asset.read_text(encoding="utf-8") == old_active
    assert old_manager.manifest_path.read_text(encoding="utf-8") == old_index_manifest
    assert not (coordinator.transaction_root / "transactions").exists()


def test_authoritative_preflight_cannot_be_forged_by_a_publishable_input(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    old_index_manifest = old_manager.manifest_path.read_text(encoding="utf-8")
    target_asset = topic_assets.table_asset_dir(topic, table) / "asset.json"
    old_active = target_asset.read_text(encoding="utf-8")
    settings.es_enabled = True
    adapter = CountingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    governance = RejectingAuthoritativePreflightGovernance(settings, topic_assets)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, governance, manager)

    result = coordinator.publish_approved(
        topic,
        table,
        "reviewer",
        "note",
        _valid_preflight(topic_assets, topic, table),
    )

    assert result["success"] is False
    assert result["status"] == "PREFLIGHT_FAILED"
    assert adapter.stage_calls == 0
    assert target_asset.read_text(encoding="utf-8") == old_active
    assert old_manager.manifest_path.read_text(encoding="utf-8") == old_index_manifest


def test_candidate_change_after_preflight_returns_stale_before_index_staging(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    preflight = _valid_preflight(topic_assets, topic, table)
    old_index_manifest = old_manager.manifest_path.read_text(encoding="utf-8")
    target_asset = topic_assets.table_asset_dir(topic, table) / "asset.json"
    old_active = target_asset.read_text(encoding="utf-8")
    settings.es_enabled = True
    adapter = CountingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    coordinator = MutatingCandidateCoordinator(
        settings,
        topic_assets,
        WritingGovernance(settings, topic_assets),
        manager,
    )

    result = coordinator.publish_approved(topic, table, "reviewer", "note", preflight)

    assert result["success"] is False
    assert result["status"] == "PREFLIGHT_STALE"
    assert result["errorCode"] == "PREFLIGHT_STALE"
    assert adapter.stage_calls == 0
    assert target_asset.read_text(encoding="utf-8") == old_active
    assert old_manager.manifest_path.read_text(encoding="utf-8") == old_index_manifest


def test_candidate_change_during_index_staging_aborts_staged_index_without_advancing_active(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    preflight = _valid_preflight(topic_assets, topic, table)
    old_index_manifest = old_manager.manifest_path.read_text(encoding="utf-8")
    target_asset = topic_assets.table_asset_dir(topic, table) / "asset.json"
    old_active = target_asset.read_text(encoding="utf-8")
    pending_asset = settings.resolved_topic_path / topic / "pending" / table / "asset.json"
    settings.es_enabled = True
    adapter = MutatingStageAdapter(pending_asset)
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, WritingGovernance(settings, topic_assets), manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", preflight)

    assert result["success"] is False
    assert result["status"] == "PREFLIGHT_STALE"
    assert result["errorCode"] == "PREFLIGHT_STALE"
    assert result["recallIndex"]["activeManifestAdvanced"] is False
    assert adapter.stage_calls == 1
    assert adapter.aborted is True
    assert target_asset.read_text(encoding="utf-8") == old_active
    assert old_manager.manifest_path.read_text(encoding="utf-8") == old_index_manifest


def test_post_publish_release_gate_failure_restores_filesystem_manifest_governance_and_index(tmp_path):
    settings, topic_assets, provider, old_manager, topic, table = _setup_runtime(tmp_path)
    preflight = _valid_preflight(topic_assets, topic, table)
    old_index_manifest = old_manager.manifest_path.read_text(encoding="utf-8")
    target = topic_assets.table_asset_dir(topic, table)
    old_active = (target / "asset.json").read_text(encoding="utf-8")
    topic_manifest = settings.resolved_topic_path / topic / "manifest.json"
    old_topic_manifest = topic_manifest.read_text(encoding="utf-8")
    pending = settings.resolved_topic_path / topic / "pending" / table / "asset.json"
    old_pending = pending.read_text(encoding="utf-8")
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    settings.es_enabled = True
    adapter = CountingCommitAdapter()
    manager = RecallIndexManager(settings, provider, es_adapter=adapter)
    governance = RejectingPostPublishGovernance(settings, topic_assets)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, governance, manager)

    result = coordinator.publish_approved(topic, table, "reviewer", "note", preflight)

    assert result["success"] is False
    assert result["status"] == "POST_PUBLISH_RELEASE_GATE_FAILED"
    assert result["errorCode"] == "POST_PUBLISH_RELEASE_GATE_FAILED"
    assert result["compensation"] == {"success": True, "status": "OLD_ACTIVE_RESTORED"}
    assert adapter.stage_calls == 1
    assert adapter.aborted is True
    assert (target / "asset.json").read_text(encoding="utf-8") == old_active
    assert topic_manifest.read_text(encoding="utf-8") == old_topic_manifest
    assert pending.read_text(encoding="utf-8") == old_pending
    assert (governance_dir / "preflight.json").exists()
    assert not (governance_dir / "publish-new.json").exists()
    assert old_manager.manifest_path.read_text(encoding="utf-8") == old_index_manifest
