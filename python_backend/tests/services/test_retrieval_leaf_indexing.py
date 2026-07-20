import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import QuestionCategory
from merchant_ai.services.assets import HybridRecallService, TopicAssetService


TOPIC = "leaf_topic"
TABLE = "fact_orders"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def leaf_recall_provider(tmp_path: Path) -> tuple[HybridRecallService, TopicAssetService]:
    topics = tmp_path / "topics"
    rules = tmp_path / "rules"
    rules.mkdir()
    write_json(
        topics / TOPIC / "manifest.json",
        [{"tableName": TABLE, "tableComment": "订单事实表", "status": "PUBLISHED"}],
    )
    write_json(
        topics / TOPIC / "tables" / TABLE / "asset.json",
        {
            "topic": TOPIC,
            "questionCategory": TOPIC,
            "tableName": TABLE,
            "status": "PUBLISHED",
            "version": "asset-v7",
            "merchantId": "merchant-asset",
            "allowedRoles": ["asset-role"],
            "requiredPermissions": ["asset.read"],
            "visibilityPolicy": {"level": "restricted", "allowedRoles": ["asset-role"]},
            "expiresAt": "2099-12-31T00:00:00Z",
            "semanticColumns": [
                {
                    "columnName": "field_%03d" % index,
                    "businessName": "业务字段 %03d" % index,
                    "role": "DIMENSION",
                    "aliases": ["字段别名 %03d" % index],
                    **(
                        {
                            "version": "field-v1",
                            "merchantId": "merchant-field",
                            "allowedRoles": ["field-role"],
                            "requiredPermissions": ["field.read"],
                            "visibilityPolicy": {"level": "private", "allowedRoles": ["field-role"]},
                            "expiresAt": "2098-12-31T00:00:00Z",
                        }
                        if index == 0
                        else {}
                    ),
                }
                for index in range(61)
            ],
            "terms": [
                {
                    "term": "业务术语 %03d" % index,
                    "description": "术语定义 %03d" % index,
                    "aliases": ["术语别名 %03d" % index],
                }
                for index in range(121)
            ],
            "knowledgeRules": [
                {
                    "ruleId": "rule_%03d" % index,
                    "title": "业务规则 %03d" % index,
                    "content": "规则正文 %03d" % index,
                    "alwaysApply": False,
                    "priority": index,
                }
                for index in range(41)
            ],
        },
    )
    settings = get_settings().model_copy(
        update={
            "topic_path": str(topics),
            "rule_knowledge_path": str(rules),
        }
    )
    assets = TopicAssetService(settings)
    return HybridRecallService(settings, assets), assets


def test_all_published_table_leaves_are_independent_stable_l2_documents(tmp_path: Path) -> None:
    provider, _assets = leaf_recall_provider(tmp_path)

    documents = provider._load_documents()
    by_type = {
        source_type: [item for item in documents if item.source_type == source_type]
        for source_type in {
            "SEMANTIC_COLUMN",
            "SEMANTIC_TERM",
            "SEMANTIC_BUSINESS_RULE",
        }
    }

    # These counts deliberately exceed the old table-summary truncation limits
    # (60 fields, 120 terms, 40 rules).
    assert len(by_type["SEMANTIC_COLUMN"]) == 61
    assert len(by_type["SEMANTIC_TERM"]) == 121
    assert len(by_type["SEMANTIC_BUSINESS_RULE"]) == 41

    expected = {
        "semantic:leaf_topic:fact_orders:field:field_060": (
            "SEMANTIC_COLUMN",
            "COLUMN",
            "topics/leaf_topic/tables/fact_orders/columns/field_060.json",
        ),
        "semantic:leaf_topic:fact_orders:term:业务术语_120": (
            "SEMANTIC_TERM",
            "TERM",
            "topics/leaf_topic/tables/fact_orders/terms/业务术语_120.json",
        ),
        "semantic:leaf_topic:fact_orders:rule:rule_040": (
            "SEMANTIC_BUSINESS_RULE",
            "BUSINESS_RULE",
            "topics/leaf_topic/tables/fact_orders/rules/rule_040.json",
        ),
    }
    by_id = {item.doc_id: item for item in documents}
    for ref_id, (source_type, semantic_kind, semantic_path) in expected.items():
        item = by_id[ref_id]
        assert item.source_type == source_type
        assert item.metadata["semanticRefId"] == ref_id
        assert item.metadata["semanticPath"] == semantic_path
        assert item.metadata["semanticKind"] == semantic_kind
        assert item.metadata["contextLayer"] == "L2"
        assert item.metadata["retrievalLevel"] == "LEAF"
        assert item.metadata["parentDirectoryId"] == "semantic:leaf_topic:fact_orders:directory"
        assert item.metadata["directoryId"] == "semantic:leaf_topic:fact_orders:directory"
        resolved = provider.semantic_catalog.read(ref_id=ref_id, path=semantic_path, max_chars=100_000)
        assert resolved["success"] is True
        assert resolved["kind"] == semantic_kind

    # A fresh provider produces the same identities, independent of process cache.
    fresh_documents = HybridRecallService(provider.settings, TopicAssetService(provider.settings))._load_documents()
    assert {
        item.doc_id
        for item in documents
        if item.source_type in by_type
    } == {
        item.doc_id
        for item in fresh_documents
        if item.source_type in by_type
    }


def test_leaf_documents_preserve_governance_without_enabling_always_apply(tmp_path: Path) -> None:
    provider, assets = leaf_recall_provider(tmp_path)
    documents = provider._load_documents()
    by_id = {item.doc_id: item for item in documents}

    overridden = by_id["semantic:leaf_topic:fact_orders:field:field_000"].metadata
    assert overridden["status"] == "PUBLISHED"
    assert overridden["assetStatus"] == "PUBLISHED"
    assert overridden["version"] == "field-v1"
    assert overridden["assetVersion"] == "asset-v7"
    assert overridden["merchantId"] == "merchant-field"
    assert overridden["assetMerchantId"] == "merchant-asset"
    assert overridden["allowedRoles"] == ["field-role"]
    assert overridden["requiredPermissions"] == ["field.read"]
    assert overridden["visibilityPolicy"]["level"] == "private"
    assert overridden["expiresAt"] == "2098-12-31T00:00:00Z"

    inherited = by_id["semantic:leaf_topic:fact_orders:term:业务术语_000"].metadata
    assert inherited["version"] == "asset-v7"
    assert inherited["merchantId"] == "merchant-asset"
    assert inherited["allowedRoles"] == ["asset-role"]
    assert inherited["requiredPermissions"] == ["asset.read"]
    assert inherited["visibilityPolicy"]["level"] == "restricted"
    assert inherited["expiresAt"] == "2099-12-31T00:00:00Z"

    rule_documents = [item for item in documents if item.source_type == "SEMANTIC_BUSINESS_RULE"]
    assert rule_documents
    assert all(item.source_type != "GOVERNED_RULE" for item in rule_documents)
    assert all(item.metadata.get("alwaysApply") is not True for item in rule_documents)
    assert assets.always_apply_rules([QuestionCategory(TOPIC)]) == []
