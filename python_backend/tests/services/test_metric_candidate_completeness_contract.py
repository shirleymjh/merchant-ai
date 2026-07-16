from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.planning import metric_candidate_decision_gaps


def _metric_entry(key: str) -> PlanningAssetEntry:
    return PlanningAssetEntry(
        key=key,
        table="table_alpha",
        source_ref_id="semantic:domain_alpha:table_alpha:metric:%s" % key,
        metadata={
            "metricKey": key,
            "formula": "SUM(%s)" % key,
            "sourceColumns": [key],
        },
    )


def _merged_metric_plan() -> QueryPlan:
    return QueryPlan(
        intents=[
            QuestionIntent(
                question="compare governed metrics",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="task_metric_alpha",
                preferred_table="table_alpha",
                metric_name="metric_alpha",
                metric_column="metric_alpha",
                metric_specs=[
                    {
                        "metricName": "metric_alpha",
                        "metricColumn": "metric_alpha",
                        "sourceColumns": ["metric_alpha"],
                        "sourceTaskId": "task_metric_alpha",
                    },
                    {
                        "metricName": "metric_beta",
                        "metricColumn": "metric_beta",
                        "sourceColumns": ["metric_beta"],
                        "sourceTaskId": "task_metric_beta",
                    },
                ],
            )
        ]
    )


def test_selected_candidate_is_covered_by_metric_spec_in_merged_intent():
    metric_beta_ref = "semantic:domain_alpha:table_alpha:metric:metric_beta"
    plan = _merged_metric_plan()
    plan.question_understanding = {
        "metricCandidateDecisions": [
            {
                "phrase": "metric beta",
                "decision": "selected_one",
                "selectedCandidateId": metric_beta_ref,
                "selectedMetricRef": "metric_beta",
                "selectedOwnerTable": "table_alpha",
            }
        ]
    }
    pack = PlanningAssetPack(metrics=[_metric_entry("metric_alpha"), _metric_entry("metric_beta")])

    gaps = metric_candidate_decision_gaps(plan, pack)

    assert "SELECTED_METRIC_CANDIDATE_NOT_PLANNED" not in {gap.code for gap in gaps}


def test_selected_source_task_id_is_covered_by_metric_spec_contract():
    plan = _merged_metric_plan()
    plan.question_understanding = {
        "metricCandidateDecisions": [
            {
                "phrase": "metric beta",
                "decision": "selected_one",
                "selectedCandidateId": "task_metric_beta",
            }
        ]
    }

    gaps = metric_candidate_decision_gaps(plan, PlanningAssetPack())

    assert "SELECTED_METRIC_CANDIDATE_NOT_PLANNED" not in {gap.code for gap in gaps}


def test_unplanned_selected_candidate_still_fails_closed():
    metric_gamma_ref = "semantic:domain_alpha:table_alpha:metric:metric_gamma"
    plan = _merged_metric_plan()
    plan.question_understanding = {
        "metricCandidateDecisions": [
            {
                "phrase": "metric gamma",
                "decision": "selected_one",
                "selectedCandidateId": metric_gamma_ref,
                "selectedMetricRef": "metric_gamma",
                "selectedOwnerTable": "table_alpha",
            }
        ]
    }
    pack = PlanningAssetPack(
        metrics=[
            _metric_entry("metric_alpha"),
            _metric_entry("metric_beta"),
            _metric_entry("metric_gamma"),
        ]
    )

    gaps = metric_candidate_decision_gaps(plan, pack)

    assert "SELECTED_METRIC_CANDIDATE_NOT_PLANNED" in {gap.code for gap in gaps}
