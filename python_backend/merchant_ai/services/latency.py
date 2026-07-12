from __future__ import annotations

import re
from typing import Any, Dict

from merchant_ai.models import AnswerMode, FastUnderstandingResult, QueryPlan


class LatencyOptimizer:
    """Classifies requests that can use a shorter, still verified BI path."""

    FAST_INTENTS = {"metric_query", "detail_lookup"}
    FAST_ANSWER_MODES = {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DETAIL}

    def initial_policy(self, fast: FastUnderstandingResult) -> Dict[str, Any]:
        simple = (
            fast.intent_kind == "metric_query" and fast.complexity in {"simple", "medium"}
        ) or (
            fast.intent_kind == "detail_lookup" and fast.complexity == "simple"
        )
        return {
            "mode": "fast_path" if simple else "standard_path",
            "eligible": bool(simple),
            "reason": "simple metric/detail query can skip planner reflection, skill worker, and answer LLM"
            if simple
            else "complex or unknown request keeps full controlled ReAct path",
            "skipNodes": ["reflect_query_graph", "run_analysis_skill", "answer_llm"] if simple else [],
            "preservedGuardrails": ["semantic_recall", "query_graph_validation", "readonly_sql", "evidence_verification"],
        }

    def update_after_plan(self, policy: Dict[str, Any], plan: QueryPlan) -> Dict[str, Any]:
        if not policy.get("eligible"):
            return policy
        executable = [
            intent
            for intent in (plan.intents or [])
            if intent.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT, AnswerMode.INVALID}
        ]
        tables = {intent.preferred_table for intent in executable if intent.preferred_table}
        simple_graph = (
            0 < len(executable) <= 3
            and not plan.dependencies
            and not plan.knowledge_requests
            and len(tables) <= 1
            and all(intent.answer_mode in self.FAST_ANSWER_MODES for intent in executable)
        )
        next_policy = dict(policy)
        next_policy["simpleGraph"] = bool(simple_graph)
        if simple_graph:
            next_policy["mode"] = "fast_path_verified_graph"
            next_policy["reason"] = "small same-table QueryGraph with no dependencies can use fast verified path"
        else:
            next_policy["eligible"] = False
            next_policy["mode"] = "standard_path"
            next_policy["reason"] = "planned graph needs full reflection/skill/answer path"
            next_policy["skipNodes"] = []
        return next_policy

    def answer_allows_llm(self, policy: Dict[str, Any]) -> bool:
        return str(policy.get("mode") or "") not in {"fast_path", "fast_path_verified_graph"}

    def response_payload(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mode": policy.get("mode") or "standard_path",
            "eligible": bool(policy.get("eligible")),
            "reason": policy.get("reason", ""),
            "skipNodes": list(policy.get("skipNodes") or []),
            "preservedGuardrails": list(policy.get("preservedGuardrails") or []),
        }

    def execution_tier_policy(
        self,
        question: str,
        plan: QueryPlan,
        fast: FastUnderstandingResult,
        remaining_seconds: float,
        prior_failure_count: int = 0,
        has_attachments: bool = False,
    ) -> Dict[str, Any]:
        executable = [intent for intent in plan.intents if intent.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT, AnswerMode.INVALID}]
        tables = {intent.preferred_table for intent in executable if intent.preferred_table}
        score = 0
        reasons = []
        if fast.intent_kind == "metric_query" and fast.complexity == "simple":
            score -= 2
            reasons.append("simple_metric_prefers_direct")
        if len(tables) > 1 or plan.dependencies:
            score += 2
            reasons.append("cross_table_or_dependency")
        if plan.knowledge_requests:
            score += 1
            reasons.append("semantic_gap")
        if prior_failure_count:
            score += 2
            reasons.append("prior_execution_failure")
        if has_attachments:
            score += 2
            reasons.append("attachment_context")
        if re.search(r"原因|归因|为什么|诊断|异常|建议|分析|下钻|关联", question or "", re.I):
            score += 2
            reasons.append("analysis_or_attribution")
        if len(executable) > 3:
            score += 1
            reasons.append("large_query_graph")
        if remaining_seconds <= 18:
            return {
                "defaultMode": "direct",
                "allowedModes": ["direct"],
                "score": score,
                "reasons": [*reasons, "remaining_budget_low"],
            }
        default_mode = "subagent" if score >= 2 else "direct"
        return {
            "defaultMode": default_mode,
            "allowedModes": [default_mode, "direct" if default_mode == "subagent" else "subagent"],
            "score": score,
            "reasons": reasons or ["standard_query"],
        }
