from types import SimpleNamespace

from merchant_ai.config import Settings
from merchant_ai.services.assets import (
    SemanticAssetGovernanceService,
    TopicAssetService,
    semantic_candidate_source_hash,
)
from merchant_ai.services.repositories import write_json
from merchant_ai.services.semantic_publish import SemanticPublishCoordinator


def review_runtime(tmp_path):
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace"),
        topic_path=str(tmp_path / "topics"),
    )
    topic_assets = TopicAssetService(settings)
    pending = settings.resolved_topic_path / "交易" / "pending" / "orders"
    pending.mkdir(parents=True, exist_ok=True)
    write_json(
        pending / "asset.json",
        {
            "topic": "交易",
            "tableName": "orders",
            "description": "订单语义资产",
            "metrics": [],
        },
    )
    governance = SemanticAssetGovernanceService(
        settings,
        SimpleNamespace(show_full_columns=lambda _table: []),
        topic_assets,
    )
    return governance, topic_assets, pending


def test_semantic_review_requires_different_submitter_and_reviewer(
    tmp_path,
    monkeypatch,
) -> None:
    governance, _topic_assets, pending = review_runtime(tmp_path)
    draft = governance.stage_review_draft("交易", "orders", "alice")
    source_hash = semantic_candidate_source_hash(pending)
    monkeypatch.setattr(
        governance,
        "preflight_publish",
        lambda _topic, _table: {
            "success": True,
            "publishable": True,
            "status": "PREFLIGHT_PASSED",
            "pendingSourceHash": source_hash,
        },
    )

    submitted = governance.submit_review("交易", "orders", "alice", "新增订单口径")
    self_review = governance.review_submission(
        "交易",
        "orders",
        "alice",
        True,
        "自审",
    )
    reviewed = governance.review_submission(
        "交易",
        "orders",
        "bob",
        True,
        "已核对",
    )

    assert draft["status"] == "DRAFT"
    assert submitted["status"] == "PENDING_REVIEW"
    assert self_review["status"] == "SELF_REVIEW_FORBIDDEN"
    assert reviewed["status"] == "APPROVED"
    assert reviewed["submittedBy"] == "alice"
    assert reviewed["reviewedBy"] == "bob"
    assert governance.authorize_publish("交易", "orders", "alice")["status"] == "SELF_PUBLISH_FORBIDDEN"
    assert governance.authorize_publish("交易", "orders", "bob")["status"] == "PUBLISH_AUTHORIZED"


def test_semantic_review_is_bound_to_exact_draft_hash(
    tmp_path,
    monkeypatch,
) -> None:
    governance, _topic_assets, pending = review_runtime(tmp_path)
    governance.stage_review_draft("交易", "orders", "alice")
    source_hash = semantic_candidate_source_hash(pending)
    monkeypatch.setattr(
        governance,
        "preflight_publish",
        lambda _topic, _table: {
            "success": True,
            "publishable": True,
            "status": "PREFLIGHT_PASSED",
            "pendingSourceHash": source_hash,
        },
    )
    governance.submit_review("交易", "orders", "alice")

    write_json(
        pending / "asset.json",
        {
            "topic": "交易",
            "tableName": "orders",
            "description": "提交后被修改",
            "metrics": [],
        },
    )

    result = governance.review_submission("交易", "orders", "bob", True)

    assert result["status"] == "DRAFT_CHANGED_AFTER_SUBMISSION"
    assert result["expectedSourceHash"] == source_hash
    assert result["actualSourceHash"] != source_hash


def test_mark_published_keeps_audit_workflow_in_active_asset(tmp_path) -> None:
    governance, topic_assets, pending = review_runtime(tmp_path)
    governance.stage_review_draft("交易", "orders", "alice")
    workflow = governance.review_workflow("交易", "orders")
    workflow.update(
        {
            "status": "APPROVED",
            "submittedBy": "alice",
            "reviewedBy": "bob",
            "approvedSourceHash": semantic_candidate_source_hash(pending),
        }
    )
    write_json(
        pending / TopicAssetService.REVIEW_WORKFLOW_FILE,
        workflow,
    )
    topic_assets.table_asset_dir("交易", "orders").mkdir(
        parents=True,
        exist_ok=True,
    )

    published = governance.mark_published("交易", "orders", "bob")

    active_workflow = topic_assets.table_asset_dir(
        "交易",
        "orders",
    ) / TopicAssetService.REVIEW_WORKFLOW_FILE
    assert published["status"] == "PUBLISHED"
    assert published["publishedBy"] == "bob"
    assert active_workflow.exists()


def test_relationship_draft_publishes_only_selected_table_scope(tmp_path) -> None:
    _governance, topic_assets, pending = review_runtime(tmp_path)
    write_json(
        topic_assets.root / "交易" / "relationships.json",
        [
            {
                "name": "orders_refunds",
                "leftTable": "orders",
                "rightTable": "refunds",
                "keys": [["order_id", "order_id"]],
            },
            {
                "name": "goods_categories",
                "leftTable": "goods",
                "rightTable": "categories",
                "keys": [["category_id", "category_id"]],
            },
        ],
    )
    write_json(
        pending / TopicAssetService.PENDING_RELATIONSHIPS_FILE,
        [
            {
                "name": "orders_refunds",
                "leftTable": "orders",
                "rightTable": "refunds",
                "joinType": "LEFT",
                "keys": [["sub_order_id", "sub_order_id"]],
            }
        ],
    )

    result = topic_assets.publish(
        "交易",
        "orders",
        approved=True,
        reviewer="bob",
        review_note="关系已核对",
    )
    relationships = topic_assets.load_relationships("交易")

    assert result["relationshipsChanged"] is True
    assert [item["name"] for item in relationships] == [
        "goods_categories",
        "orders_refunds",
    ]
    assert relationships[0]["keys"] == [["category_id", "category_id"]]
    assert relationships[1]["keys"] == [["sub_order_id", "sub_order_id"]]


def test_relationships_are_included_in_semantic_rollback_snapshot(tmp_path) -> None:
    governance, topic_assets, _pending = review_runtime(tmp_path)
    target = topic_assets.table_asset_dir("交易", "orders")
    target.mkdir(parents=True, exist_ok=True)
    write_json(
        target / "asset.json",
        {
            "topic": "交易",
            "tableName": "orders",
            "status": "PUBLISHED",
            "metrics": [],
        },
    )
    write_json(
        topic_assets.root / "交易" / "manifest.json",
        [{"tableName": "orders", "status": "PUBLISHED"}],
    )
    relationships_path = topic_assets.root / "交易" / "relationships.json"
    original = [
        {
            "name": "orders_refunds",
            "leftTable": "orders",
            "rightTable": "refunds",
            "keys": [["order_id", "order_id"]],
        }
    ]
    write_json(relationships_path, original)
    snapshot = governance._create_rollback_snapshot("交易", "orders")
    write_json(relationships_path, [])

    result = governance.rollback(
        "交易",
        "orders",
        version=snapshot["semanticVersion"],
        reviewer="bob",
        reason="恢复旧关系",
    )

    assert result["success"] is True
    assert topic_assets.load_relationships("交易") == original
    assert "relationships.json" in result["restoredFiles"]


def test_relationship_draft_expands_recall_rebuild_to_topic_scope(tmp_path) -> None:
    _governance, topic_assets, pending = review_runtime(tmp_path)
    coordinator = SemanticPublishCoordinator(
        topic_assets.settings,
        topic_assets,
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert coordinator._recall_scope_table("交易", "orders") == "orders"

    write_json(
        pending / TopicAssetService.PENDING_RELATIONSHIPS_FILE,
        [],
    )

    assert coordinator._recall_scope_table("交易", "orders") == ""
