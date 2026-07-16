from __future__ import annotations

from merchant_ai.graph.query_graph_contract import query_graph_structure_fingerprint
from merchant_ai.models import (
    AnswerMode,
    DisplayPolicy,
    EntityReference,
    IntentType,
    KnowledgeRef,
    QueryPlan,
    QuestionIntent,
    ResolvedTimeRange,
)


def _contract_plan() -> QueryPlan:
    return QueryPlan(
        intents=[
            QuestionIntent(
                question="查询受治理指标",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_task",
                preferred_table="governed_table",
                metric_name="governed_metric",
                filter_column="merchant_id",
                filter_value="M-1",
                entity_reference=EntityReference(
                    semantic_ref_id="semantic:merchant_id",
                    field="merchant_id",
                    table="governed_table",
                    raw_label="商家",
                    raw_value="M-1",
                    values=["M-1"],
                    status="resolved",
                    confidence=0.99,
                    source="planner",
                ),
                required_evidence=["governed_metric"],
                output_keys=["merchant_id", "governed_metric"],
                knowledge_ref_ids=["semantic:metric:governed_metric"],
                knowledge_refs=[
                    KnowledgeRef(
                        ref_id="semantic:metric:governed_metric",
                        ref_type="metric",
                        table="governed_table",
                        title="受治理指标",
                        reason="planner selected it",
                        score=0.99,
                    )
                ],
                metric_resolution={
                    "semanticRefId": "semantic:metric:governed_metric",
                    "metricKey": "governed_metric",
                    "ownerTable": "governed_table",
                    "formula": "SUM(metric_value)",
                    "sourceColumns": ["metric_value"],
                    "displayName": "受治理指标",
                    "description": "展示说明",
                },
                time_range=ResolvedTimeRange(
                    start_date="2026-07-01",
                    end_date="2026-07-07",
                    days=7,
                    label="最近 7 天",
                    source="planner",
                ),
            )
        ],
        evidence_contracts=[{"label": "governed_metric", "taskId": "metric_task"}],
        final_required_evidence=["governed_metric"],
        question_understanding={
            "analysisGrain": "merchant",
            "analysisIntent": "none",
            "filters": [{"field": "merchant_id", "value": "M-1"}],
            "source": "planner",
        },
    )


def test_display_only_changes_do_not_change_executable_structure_fingerprint():
    original = _contract_plan()
    displayed = original.model_copy(deep=True)
    displayed.display_title = "新的展示标题"
    displayed.display_policy = DisplayPolicy.SHOW_ALL
    displayed.intents[0].display_policy = DisplayPolicy.HIDE_INTERMEDIATE
    displayed.intents[0].question = "相同契约的展示问句"
    displayed.intents[0].group_by_name = "新的展示名称"
    displayed.intents[0].analysis_source = "different_provider"
    displayed.intents[0].analysis_note = "rewritten explanation only"
    displayed.intents[0].metric_resolution["displayName"] = "新的指标展示名"
    displayed.intents[0].metric_resolution["description"] = "新的展示说明"
    displayed.intents[0].knowledge_refs[0].title = "新的知识标题"
    displayed.intents[0].knowledge_refs[0].reason = "rewritten retrieval prose"
    displayed.intents[0].knowledge_refs[0].score = 0.2
    displayed.intents[0].entity_reference.raw_label = "新的实体展示名"
    displayed.intents[0].entity_reference.source = "repair_provider"
    displayed.intents[0].entity_reference.confidence = 0.8
    displayed.intents[0].time_range.label = "改写后的时间标题"
    displayed.intents[0].time_range.source = "repair_provider"
    displayed.question_understanding["source"] = "repair_provider"
    displayed.question_understanding["displayTitle"] = "理解层展示标题"

    assert query_graph_structure_fingerprint(displayed) == query_graph_structure_fingerprint(original)


def test_clarification_only_change_does_not_change_executable_structure_fingerprint():
    original = _contract_plan()
    clarified = original.model_copy(deep=True)
    clarified.clarification_needs = ["请选择指标展示名称"]

    assert query_graph_structure_fingerprint(clarified) == query_graph_structure_fingerprint(original)


def test_execution_and_evidence_contract_changes_change_structure_fingerprint():
    original = _contract_plan()
    original_fingerprint = query_graph_structure_fingerprint(original)

    changed_filter = original.model_copy(deep=True)
    changed_filter.intents[0].filter_value = "M-2"
    changed_filter.intents[0].entity_reference.values = ["M-2"]

    changed_formula = original.model_copy(deep=True)
    changed_formula.intents[0].metric_resolution["formula"] = "COUNT(metric_value)"

    changed_evidence = original.model_copy(deep=True)
    changed_evidence.final_required_evidence.append("merchant_id")

    for changed in (changed_filter, changed_formula, changed_evidence):
        assert query_graph_structure_fingerprint(changed) != original_fingerprint
