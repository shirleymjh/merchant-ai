from __future__ import annotations

import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import diagnostic_seed_topics
from merchant_ai.models import QuestionCategory, TopicRoutingDecision, category_display
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.routing import KeywordExtractService, RouteSlotExtractor, TopicRouterService


def write_topic_asset(
    root: Path,
    *,
    topic: str = "履约实验域",
    category_id: str = "FULFILLMENT_EXPERIMENT",
    linked_topics: list[str] | None = None,
    open_diagnostic: bool = False,
) -> TopicAssetService:
    table = "fact_fulfillment_experiment"
    table_dir = root / topic / "tables" / table
    table_dir.mkdir(parents=True)
    manifest_item = {
        "tableName": table,
        "topicContract": {
            "categoryId": category_id,
            "displayName": "履约实验",
            "aliases": ["履约观测"],
            "openDiagnostic": open_diagnostic,
            "diagnosticProfiles": ["OPERATIONS_OVERVIEW"] if open_diagnostic else [],
        },
    }
    (root / topic / "manifest.json").write_text(
        json.dumps([manifest_item], ensure_ascii=False),
        encoding="utf-8",
    )
    asset = {
        **manifest_item,
        "topic": topic,
        "status": "PUBLISHED",
        "questionCategory": category_id,
        "semanticColumns": [
            {
                "columnName": "event_ref",
                "businessName": "事件引用",
                "role": "KEY",
                "canonicalEntityRef": "entity:event",
                "isUniqueEntityKey": True,
                "filterOperators": ["EQ", "IN"],
                "aliases": ["event_ref"],
            }
        ],
        "metrics": [
            {
                "metricKey": "processing_latency",
                "businessName": "处理时延",
                "aliases": ["处理时延"],
                "formula": "AVG(latency_value)",
                "linkedTopics": linked_topics or [],
            }
        ],
        "terms": [],
        "knowledgeRules": [],
        "schemaColumns": [],
    }
    (table_dir / "asset.json").write_text(json.dumps(asset, ensure_ascii=False), encoding="utf-8")
    settings = get_settings().model_copy(update={"topic_path": str(root)})
    return TopicAssetService(settings)


def test_open_category_identifier_preserves_unpublished_values() -> None:
    category = QuestionCategory("FUTURE_ASSET_CATEGORY")
    decision = TopicRoutingDecision(candidate_topics=[category])

    assert category.value == "FUTURE_ASSET_CATEGORY"
    assert decision.model_dump(by_alias=True)["candidateTopics"] == ["FUTURE_ASSET_CATEGORY"]


def test_topic_category_display_and_alias_are_loaded_from_asset(tmp_path: Path) -> None:
    assets = write_topic_asset(tmp_path / "topics")

    contract = assets.load_topic_contract("履约实验域")
    resolved = assets.resolve_topic_category("履约观测")

    assert resolved == QuestionCategory("FULFILLMENT_EXPERIMENT")
    assert contract["displayName"] == "履约实验"
    assert category_display(resolved) == "履约实验"
    assert assets.topic_names_for_categories([resolved]) == ["履约实验域"]


def test_metric_owner_and_linked_topics_become_dynamic_route_candidates(tmp_path: Path) -> None:
    assets = write_topic_asset(
        tmp_path / "topics",
        linked_topics=["CAPACITY_FORECAST"],
    )
    keywords = KeywordExtractService(assets).extract("查看处理时延")
    decision = TopicRouterService(assets).route("查看处理时延", keywords)

    assert set(decision.candidate_topics) == {
        QuestionCategory("FULFILLMENT_EXPERIMENT"),
        QuestionCategory("CAPACITY_FORECAST"),
    }


def test_key_object_reference_topics_come_from_asset_contract(tmp_path: Path) -> None:
    assets = write_topic_asset(tmp_path / "topics")
    keywords = KeywordExtractService(assets).extract("查看 event_ref=A-42 的处理时延")
    slots = RouteSlotExtractor(assets).extract("查看 event_ref=A-42 的处理时延", keywords)

    assert [(item.ref_type, item.value) for item in slots.object_refs] == [("event_ref", "event_ref_A_42")]
    assert QuestionCategory("FULFILLMENT_EXPERIMENT") in {item.topic for item in slots.topic_candidates}


def test_missing_topic_keeps_open_discovery_instead_of_guessing_category(tmp_path: Path) -> None:
    assets = write_topic_asset(tmp_path / "topics")
    decision = TopicRouterService(assets).route("一个资产尚未覆盖的新问题", KeywordExtractService(assets).extract("一个资产尚未覆盖的新问题"))

    assert decision.candidate_topics == []
    assert decision.primary_topic == QuestionCategory.UNKNOWN
    assert decision.routing_mode == "open_discovery"
    assert decision.clarification_required is False


def test_diagnostic_seed_requires_explicit_asset_opt_in(tmp_path: Path) -> None:
    disabled = write_topic_asset(tmp_path / "disabled")
    enabled = write_topic_asset(tmp_path / "enabled", open_diagnostic=True)

    assert diagnostic_seed_topics("OPERATIONS_OVERVIEW", disabled) == []
    assert diagnostic_seed_topics("OPERATIONS_OVERVIEW", enabled) == [
        QuestionCategory("FULFILLMENT_EXPERIMENT")
    ]
