from __future__ import annotations

from typing import Any, Dict, List, Tuple

from merchant_ai.graph.message_history import append_context_section
from merchant_ai.models import ChatContext, RouteSlots, RouteTimeWindow
from merchant_ai.services.time_semantics import has_explicit_time_expression, resolve_time_range


class ClarificationResolutionService:
    """Resolve user clarification answers into structured runtime slots."""

    def resolve_context(self, context: ChatContext, answer_text: str) -> Dict[str, Any]:
        raw_answer = str(answer_text or "").strip()
        pending_type = str(context.pending_clarification_type or "")
        if not raw_answer or not pending_type:
            return {}
        answer, selected_index = self.resolve_option(context.pending_clarification_options, raw_answer)
        resolution: Dict[str, Any] = {
            "resolved": True,
            "stage": context.pending_clarification_stage,
            "type": pending_type,
            "rawAnswer": raw_answer,
            "normalizedAnswer": answer,
            "pendingQuestion": context.pending_question,
        }
        if selected_index >= 0:
            resolution.update({"selectedOption": answer, "selectedOptionIndex": selected_index})
        if pending_type == "time_window":
            days, label = self.parse_time_window(answer)
            if days:
                resolution.update({"timeWindowDays": days, "timeExpression": label, "clarificationResolved": True})
        elif pending_type == "metric_focus":
            metric_focus = answer[:120]
            if metric_focus:
                resolution.update({"metricFocus": metric_focus, "clarificationResolved": True})
        elif pending_type == "priority_goal":
            resolution.update({"priorityGoal": answer[:80], "clarificationResolved": True})
        elif pending_type == "topic_required":
            resolution.update(
                {
                    "topicFocus": answer[:80],
                    "clarificationResolved": True,
                }
            )
        elif pending_type == "business_scope":
            days, label = self.parse_time_window(answer)
            if days:
                resolution.update({"timeWindowDays": days, "timeExpression": label})
            metric_focus = answer[:120] if not days else ""
            if metric_focus:
                resolution["metricFocus"] = metric_focus
            resolution["clarificationResolved"] = bool(days or metric_focus)
        elif pending_type == "planner_clarification":
            # PlannerAgent already selected the blocking question. The harness
            # records the user's answer for the resumed planning turn without
            # reinterpreting that business choice.
            resolution["clarificationResolved"] = True
        elif pending_type == "skill_confirm" and selected_index >= 0:
            resolution.update(
                {
                    "confirmationDecision": "accepted" if selected_index == 0 else "declined",
                    "clarificationResolved": True,
                }
            )
        if not resolution.get("clarificationResolved"):
            return {}
        self.apply_to_context(context, resolution)
        return resolution

    def apply_to_context(self, context: ChatContext, resolution: Dict[str, Any]) -> None:
        context.clarification_resolved = True
        if int(resolution.get("timeWindowDays") or 0) > 0:
            context.resolved_time_window_days = int(resolution.get("timeWindowDays") or 0)
            context.days = context.resolved_time_window_days
            context.time_expression = str(resolution.get("timeExpression") or context.time_expression or "")
        if resolution.get("metricFocus"):
            context.metric_focus = str(resolution.get("metricFocus") or "")
            context.user_preference = append_context_section(
                context.user_preference,
                "clarified_metric_focus:%s" % context.metric_focus,
                max_chars=800,
            )
        if resolution.get("priorityGoal"):
            context.priority_goal = str(resolution.get("priorityGoal") or "")
            context.user_preference = append_context_section(
                context.user_preference,
                "clarified_priority_goal:%s" % context.priority_goal,
                max_chars=800,
            )
        if resolution.get("topicFocus"):
            context.topic = str(resolution.get("topicFocus") or "")

    def apply_to_route_slots(self, route_slots: RouteSlots, resolution: Dict[str, Any]) -> Tuple[RouteSlots, List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        days = int((resolution or {}).get("timeWindowDays") or 0)
        if days:
            route_slots.time_window = RouteTimeWindow(days=days, raw=str(resolution.get("timeExpression") or "clarified_%sd" % days))
            route_slots.route_warnings = [warning for warning in route_slots.route_warnings if warning != "NO_TIME_WINDOW"]
            trace.append(
                {
                    "stage": "clarification_resolution_applied",
                    "type": "time_window",
                    "timeWindow": route_slots.time_window.model_dump(by_alias=True),
                }
            )
        metric_focus = str((resolution or {}).get("metricFocus") or "")
        if metric_focus and metric_focus not in route_slots.analysis_signals:
            route_slots.analysis_signals.append(metric_focus)
            trace.append(
                {
                    "stage": "clarification_resolution_applied",
                    "type": "metric_focus",
                    "metricFocus": metric_focus,
                }
            )
        return route_slots, trace

    def parse_time_window(self, answer: str) -> tuple[int, str]:
        text = str(answer or "")
        if not has_explicit_time_expression(text):
            return 0, ""
        resolved = resolve_time_range(text)
        days = int(resolved.days or 0)
        if days <= 0:
            return 0, ""
        return days, str(resolved.label or text).strip()

    def resolve_option(self, options: List[str], answer: str) -> tuple[str, int]:
        text = str(answer or "").strip()
        normalized_options = [str(item or "").strip() for item in options or []]
        if text in normalized_options:
            return text, normalized_options.index(text)
        candidate = "".join(text.split())
        if candidate.startswith(("选", "第")):
            candidate = candidate[1:]
        if candidate.endswith(("个", "项")):
            candidate = candidate[:-1]
        if not (1 <= len(candidate) <= 2 and candidate.isascii() and candidate.isdigit()):
            return text, -1
        index = int(candidate) - 1
        if index < 0 or index >= len(normalized_options) or not normalized_options[index]:
            return text, -1
        return normalized_options[index], index
