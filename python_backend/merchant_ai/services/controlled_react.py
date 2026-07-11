from __future__ import annotations

import re
from typing import Any, Dict, List

from merchant_ai.models import AgentRunResult, GraphValidationResult, PlanningAssetPack, QueryPlan


class ControlledReactExplorer:
    """Builds Diana-style exploration traces without bypassing BI guardrails."""

    MAX_HYPOTHESES = 3
    MAX_CANDIDATES = 3

    def build_hypotheses(self, question: str, pack: PlanningAssetPack, fast_understanding: Dict[str, Any] | None = None) -> Dict[str, Any]:
        text = question or ""
        metric_names = [str(item.key or item.title or "") for item in pack.metrics if str(item.key or item.title or "")]
        table_names = pack.known_tables()
        hypotheses: List[Dict[str, Any]] = []
        templates = self._hypothesis_templates(text, metric_names)
        for index, template in enumerate(templates[: self.MAX_HYPOTHESES], start=1):
            hypotheses.append(
                {
                    "hypothesisId": "hyp_%d" % index,
                    "title": template["title"],
                    "reason": template["reason"],
                    "metricHints": template["metricHints"][:4],
                    "requiredEvidence": template["requiredEvidence"][:4],
                    "status": "candidate",
                }
            )
        return {
            "mode": "controlled_hypothesis_exploration",
            "budget": {"maxHypotheses": self.MAX_HYPOTHESES, "maxCandidateGraphs": self.MAX_CANDIDATES},
            "questionSignals": {
                "mentionsDrop": bool(re.search(r"下降|下跌|降低|drop", text, re.I)),
                "mentionsRefund": bool(re.search(r"退款|退货|refund", text, re.I)),
                "mentionsAttribution": bool(re.search(r"原因|归因|why|attribution", text, re.I)),
                "intentKind": str((fast_understanding or {}).get("intentKind") or (fast_understanding or {}).get("intent_kind") or ""),
            },
            "assetCoverage": {
                "tables": len(table_names),
                "metrics": len(metric_names),
                "relationships": len(pack.relationships),
            },
            "hypotheses": hypotheses,
            "guardrails": ["semantic_assets_only", "candidate_graph_scoring_only", "no_direct_sql_execution"],
        }

    def evaluate_candidates(self, hypotheses: Dict[str, Any], pack: PlanningAssetPack, plan: QueryPlan) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        plan_tables = {str(getattr(intent, "preferred_table", "") or "") for intent in plan.intents if getattr(intent, "preferred_table", "")}
        plan_metrics = {str(getattr(intent, "metric_name", "") or "") for intent in plan.intents if getattr(intent, "metric_name", "")}
        known_tables = set(pack.known_tables())
        known_metrics = {str(item.key or "") for item in pack.metrics if str(item.key or "")}
        relationship_bonus = min(20, len(pack.relationships) * 4)
        for item in (hypotheses or {}).get("hypotheses", [])[: self.MAX_CANDIDATES]:
            metric_hits = len(plan_metrics & set(item.get("metricHints") or []))
            table_score = 25 if plan_tables and plan_tables <= known_tables else 10 if plan_tables else 0
            metric_score = min(35, metric_hits * 12 + (10 if plan_metrics & known_metrics else 0))
            evidence_score = min(25, len(item.get("requiredEvidence") or []) * 6)
            score = table_score + metric_score + evidence_score + relationship_bonus
            candidates.append(
                {
                    "candidateId": "cand_%s" % str(item.get("hypothesisId") or len(candidates) + 1),
                    "hypothesisId": item.get("hypothesisId", ""),
                    "title": item.get("title", ""),
                    "score": min(100, score),
                    "status": "selected" if not candidates else "scored",
                    "scoreBreakdown": {
                        "semanticTableCoverage": table_score,
                        "metricCoverage": metric_score,
                        "evidencePotential": evidence_score,
                        "relationshipSupport": relationship_bonus,
                    },
                    "guardrailResult": {
                        "readOnly": True,
                        "requiresQueryGraphValidation": True,
                        "directSqlAllowed": False,
                    },
                }
            )
        candidates.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
        for index, item in enumerate(candidates):
            item["status"] = "selected" if index == 0 else "scored"
        return {
            "mode": "candidate_query_graph_sandbox",
            "selectedCandidateId": candidates[0]["candidateId"] if candidates else "",
            "candidates": candidates,
            "selectionPolicy": "highest semantic/evidence score that still requires QueryGraph validation",
        }

    def strategy_switch_trace(
        self,
        state_payload: Dict[str, Any],
        validation: GraphValidationResult,
        run_result: AgentRunResult,
    ) -> List[Dict[str, Any]]:
        switches: List[Dict[str, Any]] = []
        if state_payload.get("query_graph_repair_attempts"):
            switches.append({"from": "initial_query_graph", "to": "graph_repair", "reason": "planner validation or execution gap"})
        if getattr(validation, "gaps", None):
            switches.append({"from": "candidate_graph", "to": "repair_or_retrieve_more_context", "reason": "QueryGraph validation produced gaps"})
        if getattr(run_result, "evidence_gaps", None):
            switches.append({"from": "full_answer", "to": "partial_answer_with_disclosure", "reason": "evidence gaps require guarded answer"})
        if state_payload.get("analysis_skill_trace"):
            switches.append({"from": "plain_answer", "to": "skill_workflow", "reason": "reusable merchant SOP matched"})
        if state_payload.get("freshness_reports"):
            switches.append({"from": "offline_first", "to": "freshness_aware_execution", "reason": "table freshness was checked before answer"})
        return switches

    def _hypothesis_templates(self, question: str, metric_names: List[str]) -> List[Dict[str, Any]]:
        text = question or ""
        wanted = set(metric_names)
        def metrics(*names: str) -> List[str]:
            selected = [name for name in names if name in wanted]
            return selected or metric_names[:3]

        templates = [
            {
                "title": "核心经营指标变化来自交易规模或转化变化",
                "reason": "优先验证主指标是否真的异常，再拆成交、订单、商品或时间维度。",
                "metricHints": metrics("order_gmv_amt_1d", "pay_amt", "order_detail_cnt"),
                "requiredEvidence": ["trend", "baseline", "dimension_breakdown"],
            },
            {
                "title": "售后或退款因素影响经营结果",
                "reason": "问题包含退款/售后/风险信号时，需要把退款率、退款金额或工单证据接入主图。",
                "metricHints": metrics("refund_rate", "refund_amt", "refund_cnt"),
                "requiredEvidence": ["refund_trend", "refund_reason", "related_orders"],
            },
            {
                "title": "数据口径或新鲜度导致观察偏差",
                "reason": "BI Agent 需要确认离线表延迟、实时 fallback 和口径规则，避免把缺数据当业务为 0。",
                "metricHints": metric_names[:3],
                "requiredEvidence": ["freshness_check", "semantic_rule", "data_update_time"],
            },
        ]
        if not re.search(r"退款|退货|售后|refund", text, re.I):
            templates = [templates[0], templates[2], templates[1]]
        return templates
