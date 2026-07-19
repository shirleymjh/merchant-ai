from __future__ import annotations

from typing import Any, Dict

from merchant_ai.models import AnswerMode, FastUnderstandingResult, QueryPlan


class LatencyOptimizer:
    """Classifies requests that can use a shorter, still verified BI path."""

    FAST_INTENTS = {"metric_query", "detail_lookup"}
    FAST_ANSWER_MODES = {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DETAIL}
    FAST_CANDIDATE = "fast_candidate"
    FAST_VERIFIED = "fast_verified"
    STANDARD = "standard_path"

    def initial_policy(self, fast: FastUnderstandingResult) -> Dict[str, Any]:
        metric_count = len({str(item) for item in fast.metric_phrases if str(item)})
        domain_count = len({str(item) for item in fast.topics if str(item)})
        simple = bool(
            float(fast.confidence or 0.0) >= 0.8
            and not fast.needs_planner
            and (
                (
                    fast.intent_kind == "metric_query"
                    and fast.complexity == "simple"
                    and metric_count == 1
                    and domain_count == 1
                    and str(fast.analysis_intent or "lookup") in {"", "lookup", "metric", "trend"}
                )
                or (
                    fast.intent_kind == "detail_lookup"
                    and fast.complexity == "simple"
                    and domain_count <= 2
                    and str(fast.analysis_intent or "lookup") in {"", "lookup", "metric"}
                )
            )
        )
        return {
            "state": self.FAST_CANDIDATE if simple else self.STANDARD,
            "mode": "fast_path" if simple else "standard_path",
            "eligible": bool(simple),
            "reason": "high-confidence simple request is a fast candidate pending semantic and graph verification"
            if simple
            else "complex or unknown request keeps full controlled ReAct path",
            "skipNodes": ["reflect_query_graph", "run_analysis_skill", "answer_llm"] if simple else [],
            "preservedGuardrails": ["semantic_recall", "query_graph_validation", "readonly_sql", "evidence_verification"],
            "transitionHistory": [self.FAST_CANDIDATE if simple else self.STANDARD],
        }

    def update_after_plan(self, policy: Dict[str, Any], plan: QueryPlan) -> Dict[str, Any]:
        if self.is_standard(policy):
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
            and not bool((plan.question_understanding or {}).get("requiresExplanation"))
            and not bool((plan.question_understanding or {}).get("requires_explanation"))
            and all(self._intent_has_fast_semantic_contract(intent) for intent in executable)
        )
        next_policy = dict(policy)
        next_policy["simpleGraph"] = bool(simple_graph)
        if simple_graph:
            next_policy["state"] = self.FAST_CANDIDATE
            next_policy["mode"] = "fast_path_candidate_graph"
            next_policy["reason"] = "small governed same-table graph remains a fast candidate pending validator approval"
            return next_policy
        return self.upgrade_to_standard(next_policy, "planned graph failed the fast semantic contract")

    def update_after_validation(self, policy: Dict[str, Any], validation: Any) -> Dict[str, Any]:
        if not self.is_fast_candidate(policy):
            return policy
        if not bool(getattr(validation, "valid", False)):
            codes = ",".join(str(getattr(gap, "code", "")) for gap in list(getattr(validation, "gaps", []) or [])[:4])
            return self.upgrade_to_standard(policy, "fast graph validation failed%s" % (":%s" % codes if codes else ""))
        return self.mark_verified(policy, "semantic contract, graph structure, and question coverage passed deterministic validation")

    def mark_verified(self, policy: Dict[str, Any], reason: str) -> Dict[str, Any]:
        if self.is_standard(policy):
            return policy
        next_policy = dict(policy)
        next_policy["state"] = self.FAST_VERIFIED
        next_policy["mode"] = "fast_path_verified_graph"
        next_policy["eligible"] = True
        next_policy["reason"] = reason or "fast request passed deterministic verification"
        next_policy["transitionHistory"] = self._append_transition(policy, self.FAST_VERIFIED)
        return next_policy

    def upgrade_to_standard(self, policy: Dict[str, Any], reason: str) -> Dict[str, Any]:
        if self.is_standard(policy):
            return policy
        next_policy = dict(policy)
        next_policy.update(
            {
                "state": self.STANDARD,
                "mode": "standard_path",
                "eligible": False,
                "reason": reason or "fast request escalated to the standard path",
                "skipNodes": [],
                "escalated": True,
                "escalationReason": reason or "fast request escalated to the standard path",
                "transitionHistory": self._append_transition(policy, self.STANDARD),
            }
        )
        return next_policy

    def is_fast_candidate(self, policy: Dict[str, Any]) -> bool:
        return str((policy or {}).get("state") or "") == self.FAST_CANDIDATE

    def is_fast_verified(self, policy: Dict[str, Any]) -> bool:
        return str((policy or {}).get("state") or "") == self.FAST_VERIFIED

    def is_fast_active(self, policy: Dict[str, Any]) -> bool:
        return self.is_fast_candidate(policy) or self.is_fast_verified(policy)

    def is_standard(self, policy: Dict[str, Any]) -> bool:
        return str((policy or {}).get("state") or "") == self.STANDARD or not bool((policy or {}).get("eligible"))

    def blocks_expensive_agents(self, policy: Dict[str, Any]) -> bool:
        return self.is_fast_active(policy)

    def _append_transition(self, policy: Dict[str, Any], state: str) -> list[str]:
        history = [str(item) for item in list((policy or {}).get("transitionHistory") or []) if str(item)]
        if not history or history[-1] != state:
            history.append(state)
        return history

    def _intent_has_fast_semantic_contract(self, intent: Any) -> bool:
        if intent.answer_mode == AnswerMode.DETAIL and not (intent.metric_name or intent.metric_column or intent.metric_resolution):
            return True
        if intent.answer_mode not in {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN}:
            return False
        resolution = dict(intent.metric_resolution or {})
        semantic_ref = str(resolution.get("semanticRefId") or "")
        governance = str(resolution.get("metricGovernanceMode") or "")
        if governance == "compiled_local" or semantic_ref.startswith("semantic:compiled_local:"):
            return False
        published = governance == "published_semantic" or (semantic_ref.startswith("semantic:") and not semantic_ref.startswith("semantic:compiled_local:"))
        confidence = resolution.get("confidence")
        return bool(published and (confidence is None or float(confidence or 0.0) >= 0.7))

    def answer_allows_llm(self, policy: Dict[str, Any]) -> bool:
        return not self.blocks_expensive_agents(policy)

    def response_payload(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "state": policy.get("state") or self.STANDARD,
            "mode": policy.get("mode") or "standard_path",
            "eligible": bool(policy.get("eligible")),
            "reason": policy.get("reason", ""),
            "skipNodes": list(policy.get("skipNodes") or []),
            "preservedGuardrails": list(policy.get("preservedGuardrails") or []),
            "escalated": bool(policy.get("escalated")),
            "escalationReason": str(policy.get("escalationReason") or ""),
            "transitionHistory": list(policy.get("transitionHistory") or []),
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
        understanding = plan.question_understanding if isinstance(plan.question_understanding, dict) else {}
        analysis_intent = str(fast.analysis_intent or "").strip().lower()
        requires_explanation = bool(
            understanding.get("requiresExplanation")
            or understanding.get("requires_explanation")
            or analysis_intent not in {"", "lookup", "metric", "trend"}
        )
        if requires_explanation:
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
