from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from merchant_ai.graph.message_history import append_context_section
from merchant_ai.models import ChatContext, RouteSlots, RouteTimeWindow


class ClarificationResolutionService:
    """Resolve merchant clarification answers into structured runtime slots."""

    def resolve_context(self, context: ChatContext, answer_text: str) -> Dict[str, Any]:
        answer = str(answer_text or "").strip()
        pending_type = str(context.pending_clarification_type or "")
        if not answer or not pending_type:
            return {}
        resolution: Dict[str, Any] = {
            "resolved": True,
            "stage": context.pending_clarification_stage,
            "type": pending_type,
            "rawAnswer": answer,
            "pendingQuestion": context.pending_question,
        }
        if pending_type == "time_window":
            days, label = self.parse_time_window(answer)
            if days:
                resolution.update({"timeWindowDays": days, "timeExpression": label, "clarificationResolved": True})
        elif pending_type == "metric_focus":
            metric_focus = self.parse_metric_focus(answer)
            if metric_focus:
                resolution.update({"metricFocus": metric_focus, "clarificationResolved": True})
        elif pending_type == "priority_goal":
            resolution.update({"priorityGoal": answer[:80], "clarificationResolved": True})
        elif pending_type == "topic_required":
            resolution.update({"topicFocus": answer[:80], "clarificationResolved": True})
        elif pending_type == "business_scope":
            days, label = self.parse_time_window(answer)
            if days:
                resolution.update({"timeWindowDays": days, "timeExpression": label})
            metric_focus = self.parse_metric_focus(answer)
            if metric_focus:
                resolution["metricFocus"] = metric_focus
            resolution["clarificationResolved"] = bool(days or metric_focus)
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
        explicit = re.search(r"(?:近|最近)?\s*(\d{1,3})\s*天", text)
        if explicit:
            days = max(1, min(365, int(explicit.group(1))))
            return days, "近%d天" % days
        if re.search(r"昨天|昨日", text):
            return 1, "昨天"
        if "近7天" in text or "最近7天" in text:
            return 7, "近7天"
        if "近30天" in text or "最近30天" in text:
            return 30, "近30天"
        if "本周" in text:
            return 7, "本周"
        if "本月" in text:
            return 30, "本月"
        return 0, ""

    def parse_metric_focus(self, answer: str) -> str:
        text = str(answer or "").lower()
        patterns = [
            (r"综合经营|综合风险|经营风险|整体", "综合经营风险"),
            (r"gmv|销售额|成交额", "GMV/销售额"),
            (r"订单量|订单|下单|支付订单", "订单量"),
            (r"退款率|退款|退货", "退款率"),
            (r"客诉|工单|客服", "客诉/工单"),
            (r"赔付|理赔|补偿", "赔付/理赔"),
            (r"商品|动销|新品", "商品动销"),
            (r"履约|发货|供应链", "履约/供应链"),
        ]
        for pattern, value in patterns:
            if re.search(pattern, text):
                return value
        return ""
