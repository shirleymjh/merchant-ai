from __future__ import annotations

import json
from typing import Any

from merchant_ai.config import get_settings
from merchant_ai.models import PlanningAssetPack
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    compact_asset_planning_contract,
    planner_question_understanding_output_validation_errors,
)
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.time_semantics import resolve_time_window_contract
from merchant_ai.services.tools import question_understanding_tool


def empty_semantic_query() -> dict[str, Any]:
    return {
        "resultMode": "metric",
        "filterNodes": [],
        "rootFilterNodeId": "",
        "selectRefIds": [],
        "measureRefIds": [],
        "dimensionRefIds": [],
        "sourceRefIds": [],
        "relationshipRefIds": [],
        "joinStrategy": "auto",
        "orderBy": [],
        "limit": 0,
        "bindingStatus": "unresolved",
    }


def understanding_payload(
    analysis_intent: str,
    *,
    calculations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    analytical = analysis_intent != "none"
    return {
        "status": "UNDERSTOOD",
        "reason": "test understanding",
        "questionUnderstanding": {
            "analysisGrain": "merchant",
            "analysisIntent": analysis_intent,
            "requiresExplanation": analytical,
            "requiredEvidenceIntents": (
                [
                    {
                        "semanticLabel": "comparison_baseline",
                        "reason": "comparison needs both governed measures",
                        "requiredLevel": "required",
                        "evidenceMode": "metric",
                        "suggestedMetricRefs": ["metric_a", "metric_b"],
                        "suggestedDomains": ["test"],
                    }
                ]
                if analytical
                else []
            ),
            "anchorMetric": {
                "metricRef": "metric_a",
                "sourcePhrase": "指标甲",
                "ownerTable": "fact_a",
                "objectiveType": "metric_total",
                "groupByColumn": "merchant_id",
                "order": "desc",
                "limit": 1,
            },
            "supportMetrics": [
                {
                    "metricRef": "metric_b",
                    "sourcePhrase": "指标乙",
                    "ownerTable": "fact_b",
                    "resultMode": "metric",
                }
            ],
            "metricCandidateDecisions": [],
            "calculationIntents": calculations or [],
            "scopeConstraints": [],
            "filters": [],
            "semanticQuery": empty_semantic_query(),
            "timeWindowDays": 7,
        },
    }


def validation_errors(
    question: str,
    payload: dict[str, Any],
    planning_contract: dict[str, Any] | None = None,
) -> list[str]:
    return planner_question_understanding_output_validation_errors(
        question,
        payload,
        question_understanding_tool(False).parameters,
        planning_contract=planning_contract or {},
    )


def test_parallel_metric_values_are_not_a_comparison_contract() -> None:
    question = "最近7天指标甲、指标乙各自是多少？"

    invalid = validation_errors(question, understanding_payload("comparison"))
    direct_lookup = validation_errors(question, understanding_payload("none"))

    assert any("comparison requires an explicit structured relation" in error for error in invalid)
    assert direct_lookup == []


def test_explicit_metric_operands_support_comparison_intent() -> None:
    question = "最近7天指标甲与指标乙的差值是多少？"
    payload = understanding_payload(
        "comparison",
        calculations=[
            {
                "operation": "difference",
                "sourcePhrase": "指标甲与指标乙的差值",
                "basePopulationPhrase": "",
                "eventPopulationPhrase": "",
                "numeratorMetricRef": "metric_a",
                "denominatorMetricRef": "metric_b",
                "groupByColumn": "merchant_id",
            }
        ],
    )

    assert validation_errors(question, payload) == []


def test_unbound_phrase_operands_and_empty_evidence_fail_closed() -> None:
    question = "最近7天指标甲与指标乙的差值是多少？"
    phrase_only = understanding_payload(
        "comparison",
        calculations=[
            {
                "operation": "difference",
                "sourcePhrase": "指标甲与指标乙的差值",
                "basePopulationPhrase": "指标甲",
                "eventPopulationPhrase": "指标乙",
                "numeratorMetricRef": "",
                "denominatorMetricRef": "",
                "groupByColumn": "merchant_id",
            }
        ],
    )
    missing_evidence = understanding_payload(
        "comparison",
        calculations=[
            {
                "operation": "difference",
                "sourcePhrase": "指标甲与指标乙的差值",
                "basePopulationPhrase": "",
                "eventPopulationPhrase": "",
                "numeratorMetricRef": "metric_a",
                "denominatorMetricRef": "metric_b",
                "groupByColumn": "merchant_id",
            }
        ],
    )
    missing_evidence["questionUnderstanding"]["requiredEvidenceIntents"] = []

    assert any("comparison requires an explicit structured relation" in error for error in validation_errors(question, phrase_only))
    assert any("comparison requires at least one evidence intent" in error for error in validation_errors(question, missing_evidence))


def test_only_explicit_time_relation_supports_comparison_intent() -> None:
    comparison_question = "最近30天指标甲与前30天相比有什么变化？"
    comparison_contract = compact_asset_planning_contract(
        PlanningAssetPack(),
        time_window_contract=resolve_time_window_contract(comparison_question),
    )
    conjunction_question = "近三天和今天指标甲各是多少？"
    conjunction_contract = compact_asset_planning_contract(
        PlanningAssetPack(),
        time_window_contract=resolve_time_window_contract(conjunction_question),
    )

    assert comparison_contract["windowRelation"] == "comparison"
    assert validation_errors(
        comparison_question,
        understanding_payload("comparison"),
        comparison_contract,
    ) == []
    assert conjunction_contract["windowRelation"] == "explicit_conjunction"
    assert any(
        "comparison requires an explicit structured relation" in error
        for error in validation_errors(
            conjunction_question,
            understanding_payload("comparison"),
            conjunction_contract,
        )
    )
    assert validation_errors(
        conjunction_question,
        understanding_payload("none"),
        conjunction_contract,
    ) == []


def test_invalid_comparison_output_is_retried_as_direct_lookup() -> None:
    class RetryModel:
        configured = True
        last_error = ""
        error_events: list[Any] = []

        def __init__(self) -> None:
            self.user_payloads: list[dict[str, Any]] = []

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, **kwargs):
            del system_prompt, tool_schema, fallback, kwargs
            self.user_payloads.append(json.loads(user_prompt))
            return understanding_payload("comparison" if len(self.user_payloads) == 1 else "none")

    model = RetryModel()
    planner = QueryGraphPlanner(
        model,
        settings=get_settings().model_copy(
            update={
                "agent_planner_invalid_output_retries": 1,
                "agent_planner_prompt_budget_chars": 50_000,
            }
        ),
    )

    payload = planner._llm_understand(
        "最近7天指标甲、指标乙各自是多少？",
        PlanningAssetPack(),
        [],
        [],
        use_tool_loop=False,
    )

    assert payload["status"] == "UNDERSTOOD"
    assert payload["questionUnderstanding"]["analysisIntent"] == "none"
    assert len(model.user_payloads) == 2
    assert any(
        "comparison requires an explicit structured relation" in error
        for error in model.user_payloads[1]["structuredOutputFeedback"]["validationErrors"]
    )


def test_planner_prompt_describes_relation_not_metric_count() -> None:
    prompt = PromptAssembler().render(
        "planner.question_understanding",
        variables={
            "filesystem_authority_instruction": "no filesystem",
            "force_catalog_instruction": "",
        },
    ).system_prompt

    assert "comparison 是两项已绑定指标或明确时间窗的关系契约，不是多指标列表" in prompt
    assert "explicit_conjunction 只是逐窗取值" in prompt
    assert "不得按指标数量或并列措辞推断" in prompt
