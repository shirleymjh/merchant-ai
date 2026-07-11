from __future__ import annotations

from typing import Any, Dict, List

from merchant_ai.models import AnswerMode, FastUnderstandingResult, QueryPlan


class LatencyOptimizer:
    """Classifies requests that can use a shorter, still verified BI path."""

    FAST_INTENTS = {"metric_query", "detail_lookup"}
    FAST_ANSWER_MODES = {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DETAIL}

    def initial_policy(self, fast: FastUnderstandingResult) -> Dict[str, Any]:
        simple = fast.complexity == "simple" and fast.intent_kind in self.FAST_INTENTS
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
        simple_graph = (
            0 < len(executable) <= 1
            and not plan.dependencies
            and not plan.knowledge_requests
            and all(intent.answer_mode in self.FAST_ANSWER_MODES for intent in executable)
        )
        next_policy = dict(policy)
        next_policy["simpleGraph"] = bool(simple_graph)
        if simple_graph:
            next_policy["mode"] = "fast_path_verified_graph"
            next_policy["reason"] = "single-node QueryGraph with no dependencies can use fast verified path"
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
