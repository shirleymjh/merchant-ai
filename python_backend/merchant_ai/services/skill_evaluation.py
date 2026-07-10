from __future__ import annotations

from typing import Any, Dict, List

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    SkillEvaluationCase,
    SkillEvaluationRequest,
)
from merchant_ai.services.answer import AnswerComposeService
from merchant_ai.services.llm import LlmClient


class SkillEvaluationService:
    """Offline cases for whether LeadAgent should trigger an analysis Skill."""

    def __init__(self, settings: Settings, answer_service: AnswerComposeService | None = None):
        self.settings = settings
        self.answer_service = answer_service or AnswerComposeService(LlmClient(settings))

    def evaluate(self, request: SkillEvaluationRequest) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for case in request.cases:
            result = self.evaluate_case(case)
            results.append(result)
        total = len(results)
        passed = sum(1 for item in results if item.get("passed"))
        false_positive = sum(1 for item in results if item.get("falsePositive"))
        false_negative = sum(1 for item in results if item.get("falseNegative"))
        wrong_skill = sum(1 for item in results if item.get("wrongSkill"))
        return {
            "success": True,
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total, 4) if total else 0.0,
            "falsePositive": false_positive,
            "falseNegative": false_negative,
            "wrongSkill": wrong_skill,
            "results": results,
        }

    def evaluate_case(self, case: SkillEvaluationCase) -> Dict[str, Any]:
        plan = self._plan_from_case(case)
        rows = list(case.evidence_rows or [])
        bundle = QueryBundle(rows=rows, original_row_count=len(rows))
        run_result = AgentRunResult(
            task_results=[AgentTaskResult(task_id="eval_task", success=True, query_bundle=bundle)] if rows else [],
            query_bundles=[bundle] if rows else [],
            merged_query_bundle=bundle,
        )
        selected = self.answer_service.propose_answer_skill(case.question, plan, run_result, case.has_rule_context)
        trace = dict(getattr(self.answer_service, "last_analysis_skill_trace", {}) or {})
        triggered = bool(selected)
        expected = str(case.expected_skill or "")
        false_positive = triggered and not case.expect_trigger
        false_negative = (not triggered) and case.expect_trigger
        wrong_skill = bool(case.expect_trigger and expected and selected and selected != expected)
        passed = not false_positive and not false_negative and not wrong_skill
        return {
            "caseId": case.case_id,
            "question": case.question,
            "expectedSkill": expected,
            "expectTrigger": case.expect_trigger,
            "selectedSkill": selected,
            "triggered": triggered,
            "passed": passed,
            "falsePositive": false_positive,
            "falseNegative": false_negative,
            "wrongSkill": wrong_skill,
            "matchTrace": {
                "matchedBy": trace.get("matchedBy"),
                "confidence": trace.get("confidence"),
                "reason": trace.get("reason"),
                "fallbackSkill": trace.get("fallbackSkill"),
                "candidateSkills": trace.get("candidateSkills"),
            },
        }

    def _plan_from_case(self, case: SkillEvaluationCase) -> QueryPlan:
        intents: List[QuestionIntent] = []
        for index, item in enumerate(case.planned_evidence or []):
            category = self._category(str(item.get("category") or "TRADE"))
            answer_mode = self._answer_mode(str(item.get("answerMode") or "GROUP_AGG"))
            intents.append(
                QuestionIntent(
                    question=case.question,
                    intent_type="VALID",
                    category=category,
                    answer_mode=answer_mode,
                    plan_task_id=str(item.get("taskId") or "eval_%d" % index),
                    metric_name=str(item.get("metric") or item.get("metricName") or ""),
                    preferred_table=str(item.get("table") or ""),
                    group_by_column=str(item.get("groupBy") or item.get("groupByColumn") or ""),
                    metric_resolution=dict(item.get("resolution") or {}),
                )
            )
        if not intents:
            intents.append(
                QuestionIntent(
                    question=case.question,
                    intent_type="VALID",
                    category=QuestionCategory.TRADE,
                    answer_mode=AnswerMode.GROUP_AGG,
                    plan_task_id="eval_task",
                    metric_name=str((case.question_understanding or {}).get("metric") or ""),
                )
            )
        return QueryPlan(question_understanding=dict(case.question_understanding or {}), intents=intents)

    def _category(self, value: str) -> QuestionCategory:
        try:
            return QuestionCategory(value)
        except Exception:
            for item in QuestionCategory:
                if item.name == value:
                    return item
            return QuestionCategory.TRADE

    def _answer_mode(self, value: str) -> AnswerMode:
        try:
            return AnswerMode(value)
        except Exception:
            for item in AnswerMode:
                if item.name == value:
                    return item
            return AnswerMode.GROUP_AGG
