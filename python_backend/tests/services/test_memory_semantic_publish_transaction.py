import json

from merchant_ai.config import get_settings
from merchant_ai.models import ExtractedKeywords, QuestionCategory
from merchant_ai.services.assets import HybridRecallService, TopicAssetService, semantic_candidate_source_hash
from merchant_ai.services.memory import KnowledgeSuggestionGovernanceService, StructuredMemoryStore
from merchant_ai.services.recall_index import RecallIndexManager
from merchant_ai.services.semantic_publish import SemanticPublishCoordinator


class BoundGovernance:
    def __init__(self, settings, topic_assets, post_publishable=True):
        self.settings = settings
        self.topic_assets = topic_assets
        self.post_publishable = post_publishable

    def preflight_publish(self, topic, table_name):
        pending = self.topic_assets.root / topic / "pending" / table_name
        source_hash = semantic_candidate_source_hash(pending)
        return {
            "success": True,
            "publishable": True,
            "status": "PREFLIGHT_PASSED",
            "topic": topic,
            "tableName": table_name,
            "pendingSourceHash": source_hash,
            "semanticCatalogVersion": {
                "topic": topic,
                "table": table_name,
                "sourceHash": source_hash,
            },
        }

    def after_publish(self, topic, table_name, reviewer, review_note):
        target = self.topic_assets.table_asset_dir(topic, table_name)
        (target / "semantic_version.json").write_text(
            json.dumps({"activeVersion": "candidate"}),
            encoding="utf-8",
        )
        governance_dir = self.settings.resolved_workspace_path / "semantic_governance" / topic / table_name
        governance_dir.mkdir(parents=True, exist_ok=True)
        (governance_dir / "publish-new.json").write_text("{}", encoding="utf-8")
        return {
            "success": True,
            "status": "GOVERNED_PUBLISHED" if self.post_publishable else "POST_PUBLISH_RELEASE_GATE_FAILED",
            "topic": topic,
            "tableName": table_name,
            "releaseGate": {"publishable": self.post_publishable},
        }


class ExplodingReadbackTopicAssets(TopicAssetService):
    def verify_published_suggestion(self, topic, table, suggestion_id):
        raise RuntimeError("fault-injected readback failure")


class FailingStageAdapter:
    def stage(self, docs, manifest):
        return {"success": False, "errorCode": "FAULT_INJECTED_INDEX_STAGE_FAILURE"}


def _setup_runtime(tmp_path, asset_type=TopicAssetService, post_publishable=True):
    settings = get_settings().model_copy(
        update={
            "es_enabled": False,
            "topic_path": str(tmp_path / "topics"),
            "rule_knowledge_path": str(tmp_path / "rules"),
            "harness_workspace_path": str(tmp_path / "workspace"),
            "memory_backend": "file",
        }
    )
    settings.resolved_rule_knowledge_path.mkdir(parents=True, exist_ok=True)
    topic = "开放经营主题"
    table = "fact_open_business"
    topic_dir = settings.resolved_topic_path / topic
    target = topic_dir / "tables" / table
    target.mkdir(parents=True, exist_ok=True)
    (topic_dir / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "tableName": table,
                    "tableComment": "open business fact",
                    "supportsDetail": True,
                    "supportsMetrics": False,
                    "preferredFor": ["DETAIL", "RULE_REFERENCE"],
                    "status": "PUBLISHED",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (target / "asset.json").write_text(
        json.dumps(
            {
                "topic": topic,
                "tableName": table,
                "tableComment": "open business fact",
                "status": "PUBLISHED",
                "schemaColumns": [{"columnName": "seller_id"}],
                "semanticColumns": [],
                "metrics": [],
                "terms": [],
                "knowledgeRules": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    governance_dir.mkdir(parents=True, exist_ok=True)
    (governance_dir / "preflight.json").write_text("{}", encoding="utf-8")
    topic_assets = asset_type(settings)
    recall_provider = HybridRecallService(settings, topic_assets)
    recall_manager = RecallIndexManager(settings, recall_provider)
    initial_index = recall_manager.rebuild(documents=recall_provider._load_documents())
    assert initial_index["success"] is True
    governance = BoundGovernance(settings, topic_assets, post_publishable=post_publishable)
    coordinator = SemanticPublishCoordinator(settings, topic_assets, governance, recall_manager)
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["knowledgeSuggestions"] = [
        {
            "suggestionId": "ks_open_rule",
            "suggestionType": "business_rule",
            "status": "approved",
            "scopeType": "platform",
            "topic": topic,
            "metricName": "异常成交规则",
            "sourceTable": table,
            "payload": {
                "correctionText": "异常成交需要人工复核。",
                "alwaysApply": True,
                "proposedScope": "platform",
            },
            "createdAt": "2026-07-01T00:00:00",
        }
    ]
    store.save("seller_100", memory)
    service = KnowledgeSuggestionGovernanceService(
        settings,
        memory_store=store,
        topic_assets=topic_assets,
        governance_service=governance,
        semantic_publish_coordinator=coordinator,
    )
    return {
        "settings": settings,
        "topic": topic,
        "table": table,
        "topicAssets": topic_assets,
        "recallProvider": recall_provider,
        "recallManager": recall_manager,
        "governance": governance,
        "coordinator": coordinator,
        "store": store,
        "service": service,
    }


def _active_snapshot(runtime):
    settings = runtime["settings"]
    topic = runtime["topic"]
    table = runtime["table"]
    target = runtime["topicAssets"].table_asset_dir(topic, table)
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    pending_dir = settings.resolved_topic_path / topic / "pending" / table
    return {
        "target": {
            path.relative_to(target).as_posix(): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        },
        "topicManifest": (settings.resolved_topic_path / topic / "manifest.json").read_bytes(),
        "governance": {
            path.relative_to(governance_dir).as_posix(): path.read_bytes()
            for path in governance_dir.rglob("*")
            if path.is_file()
        },
        "indexManifest": runtime["recallManager"].manifest_path.read_bytes(),
        "pendingExisted": pending_dir.exists(),
        "pending": {
            path.relative_to(pending_dir).as_posix(): path.read_bytes()
            for path in pending_dir.rglob("*")
            if path.is_file()
        }
        if pending_dir.exists()
        else {},
    }


def _assert_active_snapshot(runtime, snapshot):
    settings = runtime["settings"]
    topic = runtime["topic"]
    table = runtime["table"]
    target = runtime["topicAssets"].table_asset_dir(topic, table)
    governance_dir = settings.resolved_workspace_path / "semantic_governance" / topic / table
    pending_dir = settings.resolved_topic_path / topic / "pending" / table
    current_governance = {
        path.relative_to(governance_dir).as_posix(): path.read_bytes()
        for path in governance_dir.rglob("*")
        if path.is_file()
    }
    current_target = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    assert current_target == snapshot["target"]
    assert (settings.resolved_topic_path / topic / "manifest.json").read_bytes() == snapshot["topicManifest"]
    assert current_governance == snapshot["governance"]
    assert runtime["recallManager"].manifest_path.read_bytes() == snapshot["indexManifest"]
    assert pending_dir.exists() is snapshot["pendingExisted"]
    current_pending = (
        {
            path.relative_to(pending_dir).as_posix(): path.read_bytes()
            for path in pending_dir.rglob("*")
            if path.is_file()
        }
        if pending_dir.exists()
        else {}
    )
    assert current_pending == snapshot["pending"]
    assert runtime["store"].load("seller_100")["knowledgeSuggestions"][0]["status"] == "approved"


def test_memory_suggestion_publish_uses_atomic_authority_and_is_immediately_recallable(tmp_path):
    runtime = _setup_runtime(tmp_path)
    previous_version = json.loads(runtime["recallManager"].manifest_path.read_text(encoding="utf-8"))["indexVersion"]

    result = runtime["service"].publish_suggestion(
        "seller_100",
        "ks_open_rule",
        reviewer="reviewer",
        review_note="approved",
    )

    assert result["success"] is True
    assert result["status"] == "PUBLISHED"
    assert result["published"]["publishState"] == "ACTIVE"
    assert result["recallIndex"]["activeManifestAdvanced"] is True
    assert result["recallIndex"]["indexVersion"] != previous_version
    assert result["recallReadback"]["indexChanged"] is True
    assert result["recallReadback"]["missingRefs"] == []
    assert result["suggestion"]["status"] == "published"
    assert result["suggestion"]["publishedRefId"] == result["recallReadback"]["expectedRefs"][0]
    active = runtime["topicAssets"].load_table_asset(runtime["topic"], runtime["table"])
    assert any(item.get("sourceSuggestionId") == "ks_open_rule" for item in active["knowledgeRules"])
    recall_docs = runtime["recallProvider"]._load_documents()
    table_doc = next(item for item in recall_docs if item.topic == runtime["topic"] and item.table == runtime["table"])
    assert "异常成交需要人工复核" in table_doc.content
    recalled = runtime["recallProvider"].recall(
        "异常成交需要人工复核",
        ExtractedKeywords(keywords=["异常成交", "人工复核"]),
        [],
        "",
        "seller_100",
        [QuestionCategory(runtime["topic"])],
    )
    assert any(item.topic == runtime["topic"] and item.table == runtime["table"] for item in recalled.items)


def test_memory_suggestion_post_publish_gate_failure_restores_every_surface_and_pending(tmp_path):
    runtime = _setup_runtime(tmp_path, post_publishable=False)
    snapshot = _active_snapshot(runtime)

    result = runtime["service"].publish_suggestion("seller_100", "ks_open_rule", reviewer="reviewer")

    assert result["success"] is False
    assert result["status"] == "POST_PUBLISH_RELEASE_GATE_FAILED"
    assert result["pendingCompensation"] == {"success": True, "status": "PENDING_CANDIDATE_RESTORED"}
    assert result["published"]["compensation"] == {"success": True, "status": "OLD_ACTIVE_RESTORED"}
    _assert_active_snapshot(runtime, snapshot)


def test_memory_suggestion_verifier_exception_restores_every_surface_and_pending(tmp_path):
    runtime = _setup_runtime(tmp_path, asset_type=ExplodingReadbackTopicAssets)
    pending = runtime["settings"].resolved_topic_path / runtime["topic"] / "pending" / runtime["table"]
    pending.mkdir(parents=True, exist_ok=True)
    active_asset = runtime["topicAssets"].table_asset_dir(runtime["topic"], runtime["table"]) / "asset.json"
    (pending / "asset.json").write_bytes(active_asset.read_bytes())
    (pending / "review-note.md").write_text("existing pending review", encoding="utf-8")
    snapshot = _active_snapshot(runtime)

    result = runtime["service"].publish_suggestion("seller_100", "ks_open_rule", reviewer="reviewer")

    assert result["success"] is False
    assert result["status"] == "ACTIVATION_VERIFICATION_FAILED"
    assert result["pendingCompensation"] == {"success": True, "status": "PENDING_CANDIDATE_RESTORED"}
    assert result["published"]["compensation"] == {"success": True, "status": "OLD_ACTIVE_RESTORED"}
    _assert_active_snapshot(runtime, snapshot)


def test_memory_suggestion_index_stage_failure_leaves_all_active_surfaces_unchanged(tmp_path):
    runtime = _setup_runtime(tmp_path)
    snapshot = _active_snapshot(runtime)
    runtime["settings"].es_enabled = True
    failing_manager = RecallIndexManager(
        runtime["settings"],
        runtime["recallProvider"],
        es_adapter=FailingStageAdapter(),
    )
    runtime["recallManager"] = failing_manager
    coordinator = SemanticPublishCoordinator(
        runtime["settings"],
        runtime["topicAssets"],
        runtime["governance"],
        failing_manager,
    )
    runtime["service"]._semantic_publish_coordinator = coordinator

    result = runtime["service"].publish_suggestion("seller_100", "ks_open_rule", reviewer="reviewer")

    assert result["success"] is False
    assert result["status"] == "PUBLISH_PENDING_INDEX_RETRY"
    assert result["pendingCompensation"] == {"success": True, "status": "PENDING_CANDIDATE_RESTORED"}
    _assert_active_snapshot(runtime, snapshot)
