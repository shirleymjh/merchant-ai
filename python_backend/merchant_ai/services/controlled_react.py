from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from merchant_ai.models import AgentRunResult, AnswerMode, GraphValidationResult, IntentType, PlanningAssetPack, QueryPlan, QuestionIntent


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
                "mentionsAttribution": bool(re.search(r"原因|归因|why|attribution", text, re.I)),
                "intentKind": str((fast_understanding or {}).get("intentKind") or (fast_understanding or {}).get("intent_kind") or ""),
                "recalledMetricCount": len(metric_names),
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
            candidate_plan = self._independent_candidate_plan(plan, item)
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
                    "queryGraph": candidate_plan.model_dump(by_alias=True),
                    "independentNodeCount": len(candidate_plan.intents),
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

    def _independent_candidate_plan(self, plan: QueryPlan, hypothesis: Dict[str, Any]) -> QueryPlan:
        hints = [str(item or "").lower() for item in hypothesis.get("metricHints") or [] if str(item or "").strip()]
        selected = []
        for intent in plan.intents:
            searchable = " ".join(
                [
                    str(getattr(intent, "metric_name", "") or ""),
                    str(getattr(intent, "metric_column", "") or ""),
                    str(getattr(intent, "metric_formula", "") or ""),
                    str(getattr(intent, "question", "") or ""),
                ]
            ).lower()
            if not hints or any(hint in searchable or searchable in hint for hint in hints):
                selected.append(intent.model_copy(deep=True))
        if not selected and plan.intents:
            selected = [plan.intents[0].model_copy(deep=True)]
        selected_ids = {intent.plan_task_id for intent in selected}
        dependencies = [
            dependency.model_copy(deep=True)
            for dependency in plan.dependencies
            if dependency.anchor_task_id in selected_ids and dependency.dependent_task_id in selected_ids
        ]
        for intent in selected:
            intent.analysis_note = "; ".join(
                part for part in [intent.analysis_note, "hypothesis=%s" % str(hypothesis.get("title") or "")] if part
            )
        return plan.model_copy(
            deep=True,
            update={
                "intents": selected,
                "dependencies": dependencies,
                "agent_trace": list(plan.agent_trace or []) + ["independent_hypothesis_graph=%s" % hypothesis.get("hypothesisId")],
            },
        )

    def run_parallel_evidence_reviews(self, hypotheses: Dict[str, Any], run_result: AgentRunResult) -> List[Dict[str, Any]]:
        rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
        candidates = list((hypotheses or {}).get("hypotheses") or [])[: self.MAX_HYPOTHESES]
        if not candidates:
            return []
        with ThreadPoolExecutor(max_workers=min(self.MAX_HYPOTHESES, len(candidates))) as executor:
            futures = [executor.submit(self._review_hypothesis, item, rows) for item in candidates]
            return [future.result() for future in futures]

    def compare_independent_executions(
        self,
        executions: List[Dict[str, Any]],
        min_score: int = 45,
        max_survivors: int = 2,
    ) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = []
        for execution in executions:
            run_result = execution.get("runResult")
            validation = execution.get("validation")
            verified = getattr(run_result, "verified_evidence", None)
            task_results = list(getattr(run_result, "task_results", []) or [])
            successful = [item for item in task_results if not item.query_bundle.failed]
            failed = [item for item in task_results if item.query_bundle.failed]
            merged_bundle = getattr(run_result, "merged_query_bundle", None)
            rows = int(merged_bundle.effective_row_count() if merged_bundle is not None else 0)
            covered = len(getattr(verified, "covered_evidence", []) or [])
            gaps = len(getattr(verified, "gaps", []) or [])
            score = int(execution.get("semanticScore") or 0)
            score += 20 if getattr(validation, "valid", False) else -40
            score += 25 if getattr(verified, "passed", False) else -min(25, gaps * 6)
            score += min(20, len(successful) * 7)
            score += min(15, covered * 2)
            score += min(10, rows)
            score -= min(30, len(failed) * 10)
            if rows <= 0:
                score -= 20
            ranked.append(
                {
                    **execution,
                    "evidenceScore": max(0, min(100, score)),
                    "rowCount": rows,
                    "successfulTasks": len(successful),
                    "failedTasks": len(failed),
                    "coveredEvidenceCount": covered,
                    "gapCount": gaps,
                    "verifiedPassed": bool(getattr(verified, "passed", False)),
                }
            )
        ranked.sort(key=lambda item: (int(item.get("evidenceScore") or 0), int(item.get("rowCount") or 0)), reverse=True)
        survivor_limit = max(1, int(max_survivors or 1))
        survivor_ids: List[str] = []
        for index, item in enumerate(ranked):
            eligible = bool(
                getattr(item.get("validation"), "valid", False)
                and item.get("verifiedPassed")
                and int(item.get("successfulTasks") or 0) > 0
                and int(item.get("rowCount") or 0) > 0
            )
            survives = bool(
                eligible
                and len(survivor_ids) < survivor_limit
                and int(item.get("evidenceScore") or 0) >= int(min_score or 0)
            )
            item["decision"] = "survive" if survives else "pruned"
            item["rank"] = index + 1
            if not survives:
                if not eligible:
                    item["eliminationReason"] = "hypothesis evidence did not pass validation and verification gates"
                elif int(item.get("evidenceScore") or 0) < int(min_score or 0):
                    item["eliminationReason"] = "hypothesis evidence score is below the survivor threshold"
            if survives:
                survivor_ids.append(str(item.get("hypothesisId") or ""))
        return {
            "ranked": ranked,
            "survivorIds": survivor_ids,
            "prunedIds": [str(item.get("hypothesisId") or "") for item in ranked if item.get("decision") == "pruned"],
            "winnerId": survivor_ids[0] if survivor_ids else "",
            "comparisonPolicy": "validation.valid && verified.passed && non-empty rows && score>=threshold",
        }

    def followup_decision(
        self,
        execution: Dict[str, Any],
        remaining_seconds: float = 60.0,
        minimum_information_gain: float = 0.35,
        answer_reserve_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        covered = int(execution.get("coveredEvidenceCount") or 0)
        gaps = int(execution.get("gapCount") or 0)
        successful_tasks = int(execution.get("successfulTasks") or 0)
        if int(execution.get("rowCount") or 0) <= 0:
            action = "switch_table"
            gain = 0.7
            cost_seconds = 20
            reason = "当前假设独立查询无数据，尝试同指标的其他语义表"
        elif not bool(execution.get("verifiedPassed")) or gaps > 0:
            action = "expand_evidence"
            gain = min(0.8, 0.45 + gaps * 0.08)
            cost_seconds = 18
            reason = "当前假设仍有证据缺口，追加独立证据节点"
        elif covered < 2 and successful_tasks < 2:
            action = "drill_dimension"
            gain = 0.4
            cost_seconds = 15
            reason = "当前只有单一证据来源，增加业务维度以确认驱动因素"
        else:
            return {
                "action": "stop",
                "reason": "当前假设已经有足够的独立证据，继续查询的边际收益较低",
                "estimatedInformationGain": 0.1,
                "estimatedCostSeconds": 0,
                "approved": False,
            }
        budget_ok = float(remaining_seconds or 0) >= float(cost_seconds + answer_reserve_seconds)
        gain_ok = gain >= float(minimum_information_gain or 0)
        approved = bool(budget_ok and gain_ok)
        return {
            "action": action if approved else "stop",
            "proposedAction": action,
            "reason": reason if approved else "预计新增信息不足或剩余时间不足，停止继续探索",
            "estimatedInformationGain": round(gain, 3),
            "estimatedCostSeconds": cost_seconds,
            "remainingSeconds": round(float(remaining_seconds or 0), 2),
            "approved": approved,
            "budgetApproved": budget_ok,
            "informationGainApproved": gain_ok,
        }

    def fallback_followup_plan(
        self,
        source_plan: QueryPlan,
        hypothesis: Dict[str, Any],
        pack: PlanningAssetPack,
        action: str,
        namespace: str,
    ) -> QueryPlan:
        if not source_plan.intents:
            return QueryPlan()
        source = source_plan.intents[0]
        hints = [str(item or "").lower() for item in hypothesis.get("metricHints") or []]
        current_tables = {str(intent.preferred_table or "") for intent in source_plan.intents}
        if action == "switch_table":
            alternatives = [
                metric
                for metric in pack.metrics
                if metric.table
                and metric.table not in current_tables
                and any(hint and (hint in str(metric.key).lower() or hint in str(metric.title).lower()) for hint in hints)
            ]
            if alternatives:
                metric = alternatives[0]
                columns = set(pack.known_columns(metric.table))
                source_columns = [str(item) for item in (metric.metadata or {}).get("sourceColumns") or metric.columns or []]
                metric_column = next((item for item in source_columns if item in columns), "") or (metric.key if metric.key in columns else "")
                intent = source.model_copy(
                    deep=True,
                    update={
                        "plan_task_id": "%s_switch" % namespace,
                        "preferred_table": metric.table,
                        "metric_name": metric.key or metric.title,
                        "metric_column": metric_column,
                        "metric_formula": str((metric.metadata or {}).get("formula") or ""),
                        "metric_resolution": resource_metric_resolution(metric),
                        "depends_on_task_ids": [],
                        "analysis_source": "hypothesis_table_switch",
                        "analysis_note": "独立假设无数据后切换语义表验证",
                    },
                )
                return QueryPlan(intents=[intent], final_required_evidence=list(source_plan.final_required_evidence), agent_trace=["hypothesis.fallback=table_switch"])
        table = source.preferred_table
        dimensions = [
            field.key
            for field in pack.fields
            if field.table == table
            and field.key in set(pack.known_columns(table))
            and field.key not in {source.metric_column, source.group_by_column}
            and semantic_asset_role(field) == "DIMENSION"
        ]
        if dimensions:
            dimension = dimensions[0]
            intent = source.model_copy(
                deep=True,
                update={
                    "plan_task_id": "%s_drill" % namespace,
                    "answer_mode": AnswerMode.GROUP_AGG,
                    "group_by_column": dimension,
                    "group_by_name": dimension,
                    "output_keys": list(dict.fromkeys([dimension, *list(source.output_keys or [])])),
                    "limit": min(50, max(10, int(source.limit or 20))),
                    "depends_on_task_ids": [],
                    "analysis_source": "hypothesis_dimension_drilldown",
                    "analysis_note": "独立假设证据下钻业务维度 %s" % dimension,
                },
            )
            return QueryPlan(intents=[intent], final_required_evidence=list(source_plan.final_required_evidence), agent_trace=["hypothesis.fallback=dimension_drilldown"])
        return QueryPlan()

    def fallback_hypothesis_seed_plan(
        self,
        hypothesis: Dict[str, Any],
        pack: PlanningAssetPack,
        question: str,
        namespace: str,
        days: int = 7,
    ) -> QueryPlan:
        hints = [str(item or "").lower() for item in hypothesis.get("metricHints") or [] if str(item or "").strip()]
        candidates = [
            metric
            for metric in pack.metrics
            if metric.table
            and (
                not hints
                or any(
                    hint in str(metric.key or "").lower()
                    or hint in str(metric.title or "").lower()
                    or str(metric.key or "").lower() in hint
                    for hint in hints
                )
            )
        ]
        if not candidates:
            candidates = [metric for metric in pack.metrics if metric.table]
        if not candidates:
            return QueryPlan()
        metric = candidates[0]
        columns = set(pack.known_columns(metric.table))
        metadata = metric.metadata or {}
        source_columns = [str(item) for item in metadata.get("sourceColumns") or metric.columns or [] if str(item)]
        metric_key = str(metric.key or metric.title or "")
        metric_column = next((item for item in source_columns if item in columns), "") or (metric_key if metric_key in columns else "")
        formula = str(metadata.get("formula") or metadata.get("metricFormula") or "")
        group_by = resource_time_column(pack, metric)
        group_by_name = resource_field_label(pack, metric.table, group_by) if group_by else ""
        answer_mode = AnswerMode.GROUP_AGG if group_by else AnswerMode.METRIC
        intent = QuestionIntent(
            question=question,
            intent_type=IntentType.VALID,
            answer_mode=answer_mode,
            plan_task_id="%s_seed" % namespace,
            preferred_table=metric.table,
            metric_name=metric_key,
            metric_column=metric_column,
            metric_formula=formula,
            metric_resolution=resource_metric_resolution(metric),
            group_by_column=group_by,
            group_by_name=group_by_name,
            days=max(1, int(days or 7)),
            limit=30,
            required_evidence=list(hypothesis.get("requiredEvidence") or []),
            output_keys=[group_by] if group_by else [],
            knowledge_ref_ids=[str(metric.source_ref_id)] if metric.source_ref_id else [],
            analysis_source="hypothesis_semantic_seed",
            analysis_note="Planner 不可用时，使用语义指标为独立假设生成受控 QueryGraph",
        )
        return QueryPlan(
            intents=[intent],
            final_required_evidence=list(hypothesis.get("requiredEvidence") or []),
            agent_trace=["hypothesis.fallback=semantic_seed", "hypothesis.id=%s" % hypothesis.get("hypothesisId")],
        )

    def _review_hypothesis(self, hypothesis: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_hints = [str(item or "").lower() for item in hypothesis.get("metricHints") or []]
        available_columns = sorted({str(key) for row in rows[:200] for key in row.keys()})
        matched_columns = [
            column
            for column in available_columns
            if any(hint and (hint in column.lower() or column.lower() in hint) for hint in metric_hints)
        ]
        numeric_evidence: List[Dict[str, Any]] = []
        for column in matched_columns[:4]:
            values = [float(row[column]) for row in rows if isinstance(row.get(column), (int, float))]
            if values:
                numeric_evidence.append(
                    {
                        "column": column,
                        "min": min(values),
                        "max": max(values),
                        "first": values[0],
                        "last": values[-1],
                        "samples": len(values),
                    }
                )
        supported = bool(rows and (matched_columns or numeric_evidence))
        return {
            "hypothesisId": hypothesis.get("hypothesisId", ""),
            "title": hypothesis.get("title", ""),
            "status": "supported_by_available_evidence" if supported else "insufficient_evidence",
            "rowCount": len(rows),
            "matchedColumns": matched_columns[:8],
            "evidence": numeric_evidence,
            "workerMode": "parallel_isolated_evidence_review",
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
        del question
        primary = metric_names[:1]
        related = metric_names[:4]
        return [
            {
                "title": "验证首个已召回指标的时间变化",
                "reason": "先验证指标变化是否由完整的时间序列和可比基线支持。",
                "metricHints": primary,
                "requiredEvidence": ["trend", "baseline", "dimension_breakdown"],
            },
            {
                "title": "比较已召回指标是否同步变化",
                "reason": "只在多个已召回指标具有可比时间覆盖时检验同步性。",
                "metricHints": related,
                "requiredEvidence": ["aligned_series", "comparable_windows", "coverage_check"],
            },
            {
                "title": "核对语义口径与数据覆盖",
                "reason": "确认指标元数据、数据更新时间和覆盖范围，避免把缺失值解释为业务事实。",
                "metricHints": related,
                "requiredEvidence": ["freshness_check", "semantic_rule", "data_update_time"],
            },
        ]


def semantic_asset_role(entry: Any) -> str:
    metadata = getattr(entry, "metadata", {}) or {}
    semantic = metadata.get("semantic") if isinstance(metadata.get("semantic"), dict) else {}
    return str(semantic.get("role") or metadata.get("semanticRole") or metadata.get("role") or "").strip().upper()


def resource_time_column(pack: PlanningAssetPack, metric: Any) -> str:
    table = str(getattr(metric, "table", "") or "")
    known = set(pack.known_columns(table))
    metadata_sources = [getattr(metric, "metadata", {}) or {}]
    metadata_sources.extend(
        getattr(entry, "metadata", {}) or {}
        for entry in pack.tables
        if str(getattr(entry, "table", "") or getattr(entry, "key", "") or "") == table
    )
    for metadata in metadata_sources:
        for key in ["timeColumn", "time_column", "timeDimension", "time_dimension", "defaultGroupBy", "default_group_by"]:
            value = metadata.get(key)
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                column = str(candidate or "").strip()
                if column and column in known:
                    return column
    for field in pack.fields:
        if field.table == table and field.key in known and semantic_asset_role(field) in {"TIME", "PARTITION"}:
            return str(field.key)
    return ""


def resource_field_label(pack: PlanningAssetPack, table: str, column: str) -> str:
    for field in pack.fields:
        if field.table != table or field.key != column:
            continue
        metadata = field.metadata or {}
        semantic = metadata.get("semantic") if isinstance(metadata.get("semantic"), dict) else {}
        return str(field.title or semantic.get("businessName") or semantic.get("description") or "").strip()
    return ""


def resource_metric_resolution(metric: Any) -> Dict[str, Any]:
    metadata = getattr(metric, "metadata", {}) or {}
    resolution: Dict[str, Any] = {"metricKey": str(getattr(metric, "key", "") or "").strip()}
    display_name = str(metadata.get("displayName") or metadata.get("display_name") or getattr(metric, "title", "") or "").strip()
    if display_name:
        resolution["displayName"] = display_name
    for target, aliases in {
        "description": ["description"],
        "unit": ["unit"],
        "valueFormat": ["valueFormat", "value_format"],
        "sourceColumnLabels": ["sourceColumnLabels", "source_column_labels"],
        "sourceColumns": ["sourceColumns", "source_columns"],
        "entityColumns": ["entityColumns", "entity_columns"],
        "scopeColumns": ["scopeColumns", "scope_columns"],
    }.items():
        value = next((metadata.get(alias) for alias in aliases if metadata.get(alias) not in (None, "", [], {})), None)
        if value not in (None, "", [], {}):
            resolution[target] = value
    source_ref = str(getattr(metric, "source_ref_id", "") or "").strip()
    if source_ref:
        resolution["semanticRefId"] = source_ref
    return resolution
