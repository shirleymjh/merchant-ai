from __future__ import annotations

from pathlib import Path
from typing import Any

from merchant_ai.config import get_settings
from merchant_ai.models import (
    FastUnderstandingResult,
    FreshnessCheckResult,
    MerchantInfo,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionCategory,
    RouteSlots,
    SkillEvaluationCase,
)
from merchant_ai.services.merchant_profile import MerchantProfileStore, MerchantProfileSummaryService
from merchant_ai.services.planning import category_for_rule_evidence
from merchant_ai.services.retrieval import topic_categories_support_knowledge_capability
from merchant_ai.services.skill_evaluation import SkillEvaluationService


class _TopicContracts:
    def __init__(self, contract: dict[str, Any]):
        self.contract = contract

    def topic_names_for_categories(self, categories: list[QuestionCategory]) -> list[str]:
        return ["runtime-topic"] if categories else []

    def load_topic_contract(self, topic: str) -> dict[str, Any]:
        assert topic == "runtime-topic"
        return self.contract


def test_freshness_contract_has_no_physical_partition_default_and_keeps_legacy_aliases() -> None:
    empty = FreshnessCheckResult()
    result = FreshnessCheckResult(pt_column="event_day", min_pt="2026-01-01", max_pt="2026-01-02")

    assert empty.time_column == ""
    assert empty.pt_column == ""
    assert result.time_column == "event_day"
    assert result.max_time_value == "2026-01-02"
    assert result.pt_column == "event_day"
    assert result.max_pt == "2026-01-02"
    assert result.model_dump(by_alias=True)["timeColumn"] == "event_day"
    assert result.model_dump(by_alias=True)["ptColumn"] == "event_day"


def test_rule_retrieval_lane_comes_from_topic_capability_not_category_name() -> None:
    assets = _TopicContracts({"metadata": {"knowledgeCapabilities": ["governed_rule"]}})

    assert topic_categories_support_knowledge_capability(
        assets,  # type: ignore[arg-type]
        [QuestionCategory("FULFILLMENT_POLICY")],
        "rule",
    )
    assert not topic_categories_support_knowledge_capability(
        _TopicContracts({"metadata": {"knowledgeCapabilities": ["metrics"]}}),  # type: ignore[arg-type]
        [QuestionCategory("FULFILLMENT_POLICY")],
        "rule",
    )


def test_rule_plan_category_is_declared_by_asset_and_ambiguous_assets_fail_open() -> None:
    single = PlanningAssetPack(
        rules=[PlanningAssetEntry(key="late_policy", topic="FULFILLMENT_POLICY")]
    )
    ambiguous = PlanningAssetPack(
        rules=[
            PlanningAssetEntry(key="late_policy", topic="FULFILLMENT_POLICY"),
            PlanningAssetEntry(key="quality_policy", topic="QUALITY_POLICY"),
        ]
    )

    assert category_for_rule_evidence(single) == QuestionCategory("FULFILLMENT_POLICY")
    assert category_for_rule_evidence(ambiguous) == QuestionCategory.UNKNOWN


def test_skill_evaluation_defaults_to_unknown_but_preserves_open_category() -> None:
    service = SkillEvaluationService.__new__(SkillEvaluationService)
    default_plan = service._plan_from_case(SkillEvaluationCase(question="evaluate this"))
    open_plan = service._plan_from_case(
        SkillEvaluationCase(
            question="evaluate this",
            planned_evidence=[{"category": "FULFILLMENT_EXPERIMENT"}],
        )
    )

    assert default_plan.intents[0].category == QuestionCategory.UNKNOWN
    assert open_plan.intents[0].category == QuestionCategory("FULFILLMENT_EXPERIMENT")


def test_merchant_profile_time_window_uses_runtime_contract_without_seven_day_guess(tmp_path: Path) -> None:
    service = MerchantProfileSummaryService()
    merchant = MerchantInfo(merchant_id="m-1")
    empty = service.summarize(
        merchant=merchant,
        memory_injection={},
        memory_constraints=[],
        route_slots=RouteSlots(),
        fast_understanding=FastUnderstandingResult(),
    )
    contracted = service.summarize(
        merchant=merchant,
        memory_injection={},
        memory_constraints=[],
        route_slots=RouteSlots(),
        fast_understanding=FastUnderstandingResult(time_window_days=13),
    )
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path / "workspace")})

    assert empty["defaultTimeWindow"] == 0
    assert contracted["defaultTimeWindow"] == 13
    assert MerchantProfileStore(settings).get_profile("m-1")["defaultTimeWindow"] == 0
