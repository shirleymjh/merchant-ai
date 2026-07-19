import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import KnowledgeRetrievalRequest, RecallItem
from merchant_ai.services.assets import SemanticCatalogService, TopicAssetService
from merchant_ai.services.retrieval import filter_recall_items_by_governance


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def service_for(tmp_path: Path) -> TopicAssetService:
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    return TopicAssetService(settings)


def write_manifest(tmp_path: Path, topic: str, table: str, **metadata: object) -> None:
    write_json(
        tmp_path / "topics" / topic / "manifest.json",
        [{"tableName": table, "title": "Navigation title", **metadata}],
    )


def write_asset(tmp_path: Path, topic: str, table: str, status: str) -> None:
    write_json(
        tmp_path / "topics" / topic / "tables" / table / "asset.json",
        {
            "topic": topic,
            "tableName": table,
            "status": status,
            "questionCategory": "GOVERNED_CATEGORY",
            "metrics": [{"metricKey": "governed_metric", "formula": "SUM(value)"}],
        },
    )


def test_directory_and_manifest_do_not_grant_topic_or_table_authority(tmp_path: Path) -> None:
    write_manifest(tmp_path, "trade", "orders", questionCategory="UNTRUSTED_CATEGORY")
    write_json(
        tmp_path / "topics" / "trade" / "relationships.json",
        [{"name": "untrusted_edge", "fromTable": "orders", "toTable": "other"}],
    )
    service = service_for(tmp_path)

    assert service.all_navigation_topic_names() == ["trade"]
    assert service.all_topic_names() == []
    assert service.load_manifest("trade") == []
    assert service.load_relationships("trade") == []
    topic_contract = service.load_topic_contract("trade")
    assert topic_contract["authority"] == "DENIED"
    assert topic_contract["activation"]["code"] == "TOPIC_HAS_NO_ACTIVE_SEMANTIC_ASSETS"

    denied = service.load_table_asset("trade", "orders")
    assert denied["authority"] == "DENIED"
    assert denied["activation"]["code"] == "SEMANTIC_ASSET_FILE_REQUIRED"
    assert denied["metrics"] == []


def test_navigation_coordinate_is_visible_but_read_cannot_become_binding_evidence(tmp_path: Path) -> None:
    write_manifest(tmp_path, "trade", "orders")
    catalog = SemanticCatalogService(service_for(tmp_path))

    refs = catalog.ls(topic="trade")
    assert any(item.get("table") == "orders" for item in refs)

    result = catalog.read(path="topics/trade/tables/orders/detail.json")
    assert result["success"] is False
    assert result["error"] == "SEMANTIC_BINDING_AUTHORITY_DENIED"
    assert result["activation"]["code"] == "SEMANTIC_ASSET_FILE_REQUIRED"


def test_missing_unknown_and_non_active_statuses_fail_closed(tmp_path: Path) -> None:
    statuses = ["", "UNKNOWN", "PENDING_REVIEW", "DISABLED"]
    for index, status in enumerate(statuses):
        topic = "topic_%s" % index
        table = "table_%s" % index
        write_manifest(tmp_path, topic, table)
        write_asset(tmp_path, topic, table, status)

    service = service_for(tmp_path)

    assert service.all_topic_names() == []
    for index, status in enumerate(statuses):
        decision = service.table_activation_decision("topic_%s" % index, "table_%s" % index)
        assert decision["allowed"] is False
        assert decision["code"] == "SEMANTIC_ASSET_NOT_ACTIVE"
        assert decision["lifecycleStatus"] == status


def test_published_or_active_asset_is_executable_and_manifest_cannot_override_contract(tmp_path: Path) -> None:
    for status in ("PUBLISHED", "ACTIVE"):
        topic = status.lower()
        table = "orders"
        write_manifest(tmp_path, topic, table, questionCategory="UNTRUSTED_CATEGORY")
        write_asset(tmp_path, topic, table, status)

    service = service_for(tmp_path)

    assert service.all_topic_names() == ["active", "published"]
    for topic in service.all_topic_names():
        asset = service.load_table_asset(topic, "orders")
        assert asset["authority"] == "EXECUTABLE"
        assert asset["activation"]["allowed"] is True
        assert service.load_topic_contract(topic)["declaredCategoryId"] == "GOVERNED_CATEGORY"


def test_semantic_recall_without_live_asset_status_is_rejected_structurally() -> None:
    request = KnowledgeRetrievalRequest(query="metric", merchant_id="merchant-1")
    items = [
        RecallItem(
            doc_id="semantic:trade:orders:metric:amount",
            source_type="SEMANTIC_METRIC",
            table="orders",
            metadata={"semanticKind": "METRIC"},
        ),
        RecallItem(doc_id="navigation-note", source_type="TEXT", metadata={}),
    ]

    kept, filtered = filter_recall_items_by_governance(items, request)

    assert [item.doc_id for item in kept] == ["navigation-note"]
    assert filtered == {"semantic_activation": 1}
