from __future__ import annotations

import json
import math
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from merchant_ai.config import Settings
from merchant_ai.models import ChatResponse, GoldenEvaluationRequest
from merchant_ai.services.repositories import write_json


GOLDEN_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "scm_inbound_7d",
        "question": "最近 7 天入库量怎么样",
        "expectedIntent": "metric_query",
        "expectedTopics": ["SCM"],
        "expectedSourceTypes": ["SEMANTIC_METRIC"],
        "expectedMetrics": ["inbound_cnt"],
    },
    {
        "id": "trade_order_count_7d",
        "question": "最近7天订单量是多少？",
        "expectedIntent": "metric_query",
        "expectedTopics": ["TRADE"],
        "expectedSourceTypes": ["SEMANTIC_METRIC"],
        "expectedMetrics": ["order_detail_cnt"],
    },
    {
        "id": "refund_metric_dispute",
        "question": "退款率口径是不是退款单数除以下单订单数？",
        "expectedIntent": "rule_data_mixed",
        "expectedTopics": ["REFUND"],
        "expectedDisclosure": ["metric_dispute"],
    },
    {
        "id": "cross_table_refund_goods",
        "question": "最近 7 天有退款的订单，关联看一下对应商品发布时间。",
        "expectedIntent": "multi_hop",
        "expectedTopics": ["REFUND", "GOODS"],
        "expectedSourceTypes": ["SEMANTIC_RELATIONSHIP", "SEMANTIC_METRIC"],
    },
    {
        "id": "platform_rule_only",
        "question": "商家入驻资质规则是什么？",
        "expectedIntent": "rule_only",
        "expectedTopics": ["PLATFORM_RULE"],
        "expectedSourceTypes": ["GOVERNED_RULE"],
    },
]


def evaluation_observability_record(debug_trace: Dict[str, Any]) -> Dict[str, Any]:
    harness = (debug_trace or {}).get("harness") or {}
    observability = harness.get("observability") or (debug_trace or {}).get("observability") or {}
    knowledge = harness.get("knowledgeRetrieval") or {}
    return {
        "selectedMemoryIds": observability.get("selectedMemoryIds") or [],
        "semanticRefIds": observability.get("semanticRefIds") or [],
        "contextHash": observability.get("contextHash") or "",
        "validationGapCount": len(observability.get("validationGaps") or []),
        "evidenceGapCount": len(observability.get("evidenceGaps") or []),
        "repairCount": int(observability.get("repairCount") or 0),
        "recallSourceRefs": knowledge.get("sourceRefs") or [],
        "recallRounds": knowledge.get("rounds") or [],
    }


class GoldenCaseLoader:
    """Loads production-style golden cases from jsonl with legacy fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def default_path(self) -> Path:
        return self.settings.resources_root / "evaluation" / "golden_cases.jsonl"

    def load(self, path: str = "") -> List[Dict[str, Any]]:
        source = Path(path) if path else self.default_path()
        cases: List[Dict[str, Any]] = []
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    item = json.loads(text)
                except Exception:
                    continue
                if item.get("id") and item.get("question"):
                    cases.append(item)
        if cases:
            return cases
        return [dict(item) for item in GOLDEN_QUESTIONS]


class GoldenEvaluationService:
    """Run end-to-end golden cases and score recall, graph, SQL, evidence and answer layers."""

    LAYERS = ["recall", "queryGraph", "sql", "execution", "evidence", "answer"]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = GoldenCaseLoader(settings)

    def evaluate(
        self,
        request: GoldenEvaluationRequest,
        runner: Callable[..., Any],
        query_executor: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        merchant_id = request.merchant_id or self.settings.merchant_id
        cases = self._select_cases(self.loader.load(request.cases_path), request.case_ids, request.limit)
        results = [self.evaluate_case(case, merchant_id, runner, query_executor=query_executor) for case in cases]
        summary = self._summary(results)
        report = {
            "success": True,
            "merchantId": merchant_id,
            "generatedAt": datetime.now().isoformat(),
            "caseCount": len(results),
            **summary,
            "results": results,
        }
        if request.persist_report:
            report["reportPath"] = str(self._persist_report(report))
        failed_items = self._governance_items(results, merchant_id)
        report["governanceItems"] = failed_items
        if request.persist_governance_items and failed_items:
            report["governancePath"] = str(self._persist_governance_items(failed_items))
        return report

    def evaluate_case(
        self,
        case: Dict[str, Any],
        merchant_id: str,
        runner: Callable[..., Any],
        query_executor: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        started = datetime.now()
        response: Optional[ChatResponse] = None
        error = ""
        try:
            raw = self._run_case(case, merchant_id, runner)
            response = raw if isinstance(raw, ChatResponse) else ChatResponse.model_validate(raw)
        except Exception as exc:
            error = str(exc)[:1000]
        duration_ms = int((datetime.now() - started).total_seconds() * 1000)
        if response is None:
            layers = {layer: self._layer(False, ["workflow_error:%s" % error]) for layer in self.LAYERS}
            return self._case_result(case, duration_ms, "", {}, layers, error)
        trace = response.debug_trace or {}
        layers = {
            "recall": self._score_recall(case, trace),
            "queryGraph": self._score_query_graph(case, trace),
            "sql": self._score_sql(case, trace),
            "execution": self._score_execution(case, trace, query_executor, merchant_id),
            "evidence": self._score_evidence(case, trace),
            "answer": self._score_answer(case, response.answer or "", trace),
        }
        return self._case_result(case, duration_ms, response.id, trace, layers, "")

    def _run_case(self, case: Dict[str, Any], merchant_id: str, runner: Callable[..., Any]) -> Any:
        thread_id = "golden_%s" % uuid.uuid4().hex[:12]
        run_id = "run_%s" % uuid.uuid4().hex[:12]

        def listener(event_type: str, node: str, payload: Dict[str, Any]) -> None:
            return None

        kwargs = {
            "context": None,
            "listener": listener,
            "thread_id": thread_id,
            "run_id": run_id,
            "message_history": case.get("messageHistory") or case.get("message_history") or [],
        }
        try:
            return runner(case["question"], merchant_id, **kwargs)
        except TypeError:
            return runner(case["question"], merchant_id, None, listener, thread_id, run_id, kwargs["message_history"])

    def _score_recall(self, case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        blob = self._trace_blob(
            [
                (trace.get("harness") or {}).get("knowledgeRetrieval") or {},
                trace.get("planningAssetPack") or {},
                trace.get("plannerLoadedRefs") or [],
                trace.get("metricResolution") or {},
                trace.get("semanticMetric") or {},
                trace.get("semanticMetrics") or [],
                trace.get("metricTerms") or [],
                {"quickMetricPath": trace.get("quickMetricPath"), "metric": trace.get("metric"), "days": trace.get("days")},
            ]
        )
        reasons = []
        reasons.extend(self._missing_terms(case.get("expectedSourceTypes") or [], blob, "source_type"))
        reasons.extend(self._missing_terms(case.get("expectedTopics") or [], blob, "topic"))
        reasons.extend(self._missing_terms(case.get("expectedMetrics") or [], blob, "metric"))
        return self._layer(not reasons, reasons)

    def _score_query_graph(self, case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        intents = self._plan_intents(trace)
        deps = self._dependencies(trace)
        understanding = self._question_understanding(trace)
        quick_context = [
            trace.get("semanticMetric") or {},
            trace.get("semanticMetrics") or [],
            trace.get("metricTerms") or [],
            {"quickMetricPath": trace.get("quickMetricPath"), "metric": trace.get("metric"), "days": trace.get("days")},
        ]
        blob = self._trace_blob([intents, deps, understanding, *quick_context])
        reasons = []
        reasons.extend(self._missing_terms(case.get("expectedTables") or [], blob, "table"))
        reasons.extend(self._missing_terms(case.get("expectedMetrics") or [], blob, "metric"))
        expected_days = int(case.get("expectedTimeWindowDays") or 0)
        if expected_days and str(expected_days) not in blob:
            reasons.append("missing_time_window:%s" % expected_days)
        for item in case.get("expectedFilters") or []:
            if self._normalize(item) not in blob and not self._expected_filter_present(item, [intents, deps, understanding, self._task_results(trace)]):
                reasons.append("missing_filter:%s" % item)
        expected_graph = case.get("expectedGraph") or {}
        if expected_graph.get("hasAnchor") and not intents:
            reasons.append("missing_anchor_node")
        if expected_graph.get("hasDependent") and not deps:
            reasons.append("missing_dependent_node")
        for key in expected_graph.get("joinKeys") or []:
            if self._normalize(key) not in blob:
                reasons.append("missing_join_key:%s" % key)
        return self._layer(not reasons, reasons)

    def _score_sql(self, case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        task_results = self._task_results(trace)
        expected_tables = case.get("expectedTables") or []
        expected_intent = str(case.get("expectedIntent") or "").lower()
        expects_data_execution = bool(
            expected_tables
            or case.get("expectedMetrics")
            or case.get("expectedGraph")
            or expected_intent in {"metric_query", "detail_lookup", "ranking", "multi_hop", "analysis", "rule_data_mixed"}
        )
        if not task_results and not expects_data_execution:
            return self._layer(True, [])
        reasons = []
        if expects_data_execution and not task_results and bool(trace.get("quickMetricPath")):
            blob = self._trace_blob(
                [
                    trace.get("semanticMetric") or {},
                    trace.get("semanticMetrics") or [],
                    trace.get("metricTerms") or [],
                    {"quickMetricPath": True, "metric": trace.get("metric"), "days": trace.get("days")},
                ]
            )
            reasons.extend(self._missing_terms(expected_tables, blob, "table"))
            reasons.extend(self._missing_terms(case.get("expectedMetrics") or [], blob, "metric"))
            return self._layer(not reasons, reasons)
        if expects_data_execution and not task_results:
            reasons.append("missing_task_results")
        for item in task_results:
            if not bool(item.get("success", False)):
                reasons.append("task_failed:%s" % (item.get("taskId") or item.get("task_id") or "unknown"))
            bundle = item.get("queryBundle") or item.get("query_bundle") or {}
            if bundle.get("failed"):
                reasons.append("sql_failed:%s" % str(bundle.get("error") or "")[:120])
        blob = self._trace_blob([task_results])
        reasons.extend(self._missing_terms(expected_tables, blob, "table"))
        return self._layer(not reasons, reasons)

    def _score_execution(
        self,
        case: Dict[str, Any],
        trace: Dict[str, Any],
        query_executor: Optional[Callable[..., List[Dict[str, Any]]]],
        merchant_id: str,
    ) -> Dict[str, Any]:
        contract = case.get("expectedResult") or case.get("expected_result") or {}
        if not isinstance(contract, dict):
            contract = {}
        reference_sql = str(contract.get("sql") or case.get("referenceSql") or case.get("expectedSql") or "").strip()
        expected_rows = contract.get("rows")
        if expected_rows is None:
            expected_rows = case.get("expectedRows")
        applicable = bool(reference_sql or expected_rows is not None)
        if not applicable:
            return self._layer(True, [], applicable=False, details={"status": "not_configured"})

        if reference_sql:
            if query_executor is None:
                return self._layer(
                    False,
                    ["reference_query_executor_unavailable"],
                    applicable=True,
                    details={"status": "reference_not_executed"},
                )
            if not self._safe_reference_sql(reference_sql):
                return self._layer(False, ["unsafe_reference_sql"], applicable=True)
            try:
                params = [
                    merchant_id if str(item) == "$merchant_id" else item
                    for item in (contract.get("params") or case.get("referenceParams") or [])
                ]
                expected_rows = query_executor(reference_sql, params or None)
            except TypeError:
                expected_rows = query_executor(reference_sql)
            except Exception as exc:
                return self._layer(
                    False,
                    ["reference_sql_failed:%s" % str(exc)[:180]],
                    applicable=True,
                    details={"status": "reference_failed"},
                )

        actual_rows = self._actual_execution_rows(trace, contract)
        if actual_rows is None:
            return self._layer(False, ["agent_result_rows_missing"], applicable=True)
        comparison = compare_execution_rows(
            expected_rows or [],
            actual_rows,
            columns=contract.get("columns") or case.get("resultColumns") or [],
            key_columns=contract.get("keyColumns") or case.get("resultKeyColumns") or [],
            order_sensitive=bool(contract.get("orderSensitive", case.get("resultOrderSensitive", False))),
            numeric_tolerance=float(contract.get("numericTolerance", case.get("numericTolerance", 0.0)) or 0.0),
        )
        return self._layer(
            bool(comparison.get("matched")),
            list(comparison.get("reasons") or []),
            applicable=True,
            details=comparison,
        )

    def _actual_execution_rows(self, trace: Dict[str, Any], contract: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        task_results = self._task_results(trace)
        task_id = str(contract.get("taskId") or "")
        selected = [
            item
            for item in task_results
            if isinstance(item, dict)
            and (not task_id or str(item.get("taskId") or item.get("task_id") or "") == task_id)
        ]
        if task_id and not selected:
            return None
        rows: List[Dict[str, Any]] = []
        for item in selected:
            bundle = item.get("queryBundle") or item.get("query_bundle") or {}
            bundle_rows = bundle.get("rows") if isinstance(bundle, dict) else None
            if isinstance(bundle_rows, list):
                rows.extend(row for row in bundle_rows if isinstance(row, dict))
        return rows if selected else None

    def _safe_reference_sql(self, sql: str) -> bool:
        normalized = " ".join(str(sql or "").strip().lower().split())
        if not (normalized.startswith("select ") or normalized.startswith("with ")):
            return False
        forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " truncate ", " create ", " grant "]
        padded = " %s " % normalized
        return ";" not in normalized.rstrip(";") and not any(token in padded for token in forbidden)

    def _score_evidence(self, case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        verified = trace.get("verifiedEvidence") or trace.get("verified_evidence") or {}
        gaps = trace.get("evidenceGaps") or trace.get("evidence_gaps") or []
        validation = trace.get("validation") or trace.get("queryGraphValidation") or {}
        blob = self._trace_blob([verified, gaps, validation, self._question_understanding(trace)])
        reasons = []
        if verified and verified.get("passed") is False and verified.get("blockingGaps"):
            reasons.append("blocking_evidence_gaps")
        if any(str(gap.get("severity") or "").lower() == "blocking" for gap in gaps if isinstance(gap, dict)):
            reasons.append("blocking_gap_in_trace")
        validation_gaps = validation.get("gaps") if isinstance(validation, dict) else []
        for gap in validation_gaps or []:
            code = str((gap or {}).get("code") or "")
            if code in {"PLANNER_LLM_TIMEOUT", "PLANNER_CONTEXT_OVER_BUDGET"}:
                reasons.append("planner_validation_gap:%s" % code)
        reasons.extend(self._missing_terms(case.get("expectedEvidence") or [], blob, "evidence"))
        for item in case.get("expectedDisclosure") or []:
            if self._normalize(item) not in blob:
                reasons.append("missing_disclosure:%s" % item)
        return self._layer(not reasons, reasons)

    def _score_answer(self, case: Dict[str, Any], answer: str, trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self._normalize(answer)
        reasons = []
        if not text:
            reasons.append("empty_answer")
        for term in case.get("answerMustMention") or []:
            if self._normalize(term) not in text:
                reasons.append("answer_missing:%s" % term)
        for term in case.get("answerMustNotMention") or []:
            if self._normalize(term) in text:
                reasons.append("answer_forbidden:%s" % term)
        claim_verification = self._answer_claim_verification(trace or {})
        if claim_verification and claim_verification.get("passed") is False:
            reasons.append("answer_claim_verification_failed")
        return self._layer(
            not reasons,
            reasons,
            details={
                "claimVerificationConfigured": bool(claim_verification),
                "claimVerificationPassed": claim_verification.get("passed") if claim_verification else None,
                "claimFallbackUsed": bool(claim_verification.get("fallbackUsed")) if claim_verification else False,
                "rejectedClaimCount": len(claim_verification.get("rejectedClaims") or []) if claim_verification else 0,
            },
        )

    def _answer_claim_verification(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        direct = trace.get("answerClaimVerification") or trace.get("answer_claim_verification") or {}
        if isinstance(direct, dict) and direct:
            return direct
        guard = trace.get("answerGuard") or trace.get("answer_guard") or {}
        if not isinstance(guard, dict):
            return {}
        verification = guard.get("claimVerification") or guard.get("claim_verification") or {}
        return verification if isinstance(verification, dict) else {}

    def _case_result(
        self,
        case: Dict[str, Any],
        duration_ms: int,
        response_id: str,
        trace: Dict[str, Any],
        layers: Dict[str, Dict[str, Any]],
        error: str,
    ) -> Dict[str, Any]:
        passed = all(item.get("passed") for item in layers.values())
        failed_layers = [layer for layer, item in layers.items() if not item.get("passed")]
        first_failed_layer = failed_layers[0] if failed_layers else ""
        return {
            "caseId": case.get("id", ""),
            "question": case.get("question", ""),
            "responseId": response_id,
            "durationMs": duration_ms,
            "passed": passed,
            "failedLayers": failed_layers,
            "firstFailedLayer": first_failed_layer,
            "layers": layers,
            "error": error,
            "observability": evaluation_observability_record(trace),
            "traceExcerpt": self._trace_excerpt(trace),
        }

    def _summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for item in results if item.get("passed"))
        layer_scores = {}
        for layer in self.LAYERS:
            applicable_items = [
                item for item in results if (item.get("layers") or {}).get(layer, {}).get("applicable", True)
            ]
            layer_passed = sum(1 for item in applicable_items if (item.get("layers") or {}).get(layer, {}).get("passed"))
            layer_scores[layer] = round(layer_passed / len(applicable_items), 4) if applicable_items else None
        first_failure_breakdown = {
            layer: sum(1 for item in results if item.get("firstFailedLayer") == layer) for layer in self.LAYERS
        }
        return {
            "passed": passed,
            "failed": total - passed,
            "accuracy": round(passed / total, 4) if total else 0.0,
            "recallAccuracy": layer_scores["recall"],
            "queryGraphAccuracy": layer_scores["queryGraph"],
            "sqlSuccessRate": layer_scores["sql"],
            "executionAccuracy": layer_scores["execution"],
            "executionCaseCount": sum(
                1 for item in results if (item.get("layers") or {}).get("execution", {}).get("applicable", True)
            ),
            "evidenceCoverageRate": layer_scores["evidence"],
            "answerAccuracy": layer_scores["answer"],
            "answerClaimFallbackCount": sum(
                1
                for item in results
                if bool((((item.get("layers") or {}).get("answer") or {}).get("details") or {}).get("claimFallbackUsed"))
            ),
            "firstFailureBreakdown": first_failure_breakdown,
        }

    def _governance_items(self, results: List[Dict[str, Any]], merchant_id: str) -> List[Dict[str, Any]]:
        items = []
        for result in results:
            for layer in result.get("failedLayers") or []:
                reasons = ((result.get("layers") or {}).get(layer) or {}).get("reasons") or []
                items.append(
                    {
                        "id": "eval_%s_%s_%s" % (result.get("caseId", ""), layer, uuid.uuid4().hex[:8]),
                        "type": self._governance_type(layer),
                        "status": "pending_review",
                        "merchantId": merchant_id,
                        "caseId": result.get("caseId", ""),
                        "question": result.get("question", ""),
                        "failedLayer": layer,
                        "reasons": reasons,
                        "suggestedAction": self._suggested_action(layer),
                        "createdAt": datetime.now().isoformat(),
                    }
                )
        return items

    def _persist_report(self, report: Dict[str, Any]) -> Path:
        root = self.settings.resolved_ops_path / "golden_evaluation_reports"
        root.mkdir(parents=True, exist_ok=True)
        path = root / ("golden_eval_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        write_json(path, report)
        return path

    def _persist_governance_items(self, items: List[Dict[str, Any]]) -> Path:
        path = self.settings.resolved_ops_path / "evaluation_governance_items.json"
        existing: List[Dict[str, Any]] = []
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                existing = data.get("items", data if isinstance(data, list) else []) or []
        except Exception:
            existing = []
        merged = (existing + items)[-500:]
        write_json(path, {"items": merged})
        return path

    def _select_cases(self, cases: List[Dict[str, Any]], case_ids: List[str], limit: int) -> List[Dict[str, Any]]:
        selected = cases
        if case_ids:
            wanted = set(case_ids)
            selected = [case for case in selected if str(case.get("id") or "") in wanted]
        if limit and limit > 0:
            selected = selected[:limit]
        return selected

    def _layer(
        self,
        passed: bool,
        reasons: List[str],
        applicable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "passed": bool(passed),
            "score": 1.0 if passed else 0.0,
            "applicable": bool(applicable),
            "reasons": reasons,
        }
        if details is not None:
            payload["details"] = details
        return payload

    def _plan_intents(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        top_level = trace.get("planIntents") or trace.get("plan_intents") or []
        if top_level:
            return top_level
        plan = trace.get("plan") or {}
        if isinstance(plan, dict):
            return plan.get("intents") or []
        return []

    def _dependencies(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        top_level = trace.get("dependencies") or []
        if top_level:
            return top_level
        plan = trace.get("plan") or {}
        if isinstance(plan, dict):
            return plan.get("dependencies") or []
        return []

    def _task_results(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        top_level = trace.get("taskResults") or trace.get("task_results") or []
        if top_level:
            return top_level
        tasks = trace.get("tasks") or []
        if tasks and any(isinstance(item, dict) and ("queryBundle" in item or "query_bundle" in item) for item in tasks):
            return tasks
        run_result = trace.get("agentRunResult") or trace.get("agent_run_result") or {}
        if isinstance(run_result, dict):
            return run_result.get("taskResults") or run_result.get("task_results") or []
        return []

    def _question_understanding(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        understanding = trace.get("questionUnderstanding") or trace.get("question_understanding") or {}
        if understanding:
            return understanding
        plan = trace.get("plan") or {}
        if isinstance(plan, dict):
            return plan.get("questionUnderstanding") or plan.get("question_understanding") or {}
        return {}

    def _trace_excerpt(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        plan = trace.get("plan") or {}
        validation = trace.get("validation") or trace.get("queryGraphValidation") or {}
        task_results = self._task_results(trace)
        return {
            "answerPreview": str(trace.get("answer") or "")[:600],
            "questionUnderstanding": self._question_understanding(trace),
            "planIntents": self._plan_intents(trace)[:8],
            "dependencies": self._dependencies(trace)[:8],
            "validation": validation,
            "agentTrace": (plan.get("agentTrace") or trace.get("agentTrace") or [])[:12] if isinstance(plan, dict) else trace.get("agentTrace", [])[:12],
            "compilerTrace": (plan.get("compilerTrace") or trace.get("compilerTrace") or [])[:20] if isinstance(plan, dict) else trace.get("compilerTrace", [])[:20],
            "taskResults": [
                {
                    "taskId": item.get("taskId") or item.get("task_id"),
                    "success": item.get("success"),
                    "summary": str(item.get("summary") or "")[:300],
                    "queryBundle": self._compact_query_bundle(item.get("queryBundle") or item.get("query_bundle") or {}),
                }
                for item in task_results[:8]
                if isinstance(item, dict)
            ],
            "evidenceGaps": (trace.get("evidenceGaps") or trace.get("evidence_gaps") or [])[:12],
            "verifiedEvidence": trace.get("verifiedEvidence") or trace.get("verified_evidence") or {},
        }

    def _compact_query_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(bundle, dict):
            return {}
        return {
            "failed": bool(bundle.get("failed")),
            "error": str(bundle.get("error") or "")[:300],
            "summary": str(bundle.get("summary") or "")[:300],
            "tables": bundle.get("tables") or [],
            "rowCount": bundle.get("originalRowCount") or bundle.get("original_row_count") or len(bundle.get("rows") or []),
            "sqlPreview": str(bundle.get("sql") or "")[:500],
        }

    def _missing_terms(self, terms: Iterable[Any], blob: str, label: str) -> List[str]:
        return ["missing_%s:%s" % (label, term) for term in terms if self._normalize(term) not in blob]

    def _expected_filter_present(self, expected_filter: Any, values: List[Any]) -> bool:
        text = str(expected_filter or "")
        if "=" not in text:
            return False
        column, value = [self._normalize(item) for item in text.split("=", 1)]
        if not column or not value:
            return False
        blob = self._trace_blob(values)
        return column in blob and value in blob

    def _trace_blob(self, values: List[Any]) -> str:
        return self._normalize(json.dumps(values, ensure_ascii=False, default=str, sort_keys=True))

    def _normalize(self, value: Any) -> str:
        return str(value or "").lower().replace("_", "").replace("-", "").replace(" ", "")

    def _governance_type(self, layer: str) -> str:
        return {
            "recall": "recall_index_or_semantic_asset_gap",
            "queryGraph": "query_graph_or_semantic_relationship_gap",
            "sql": "sql_execution_or_schema_gap",
            "execution": "sql_execution_accuracy_gap",
            "evidence": "evidence_contract_gap",
            "answer": "answer_generation_gap",
        }.get(layer, "evaluation_gap")

    def _suggested_action(self, layer: str) -> str:
        return {
            "recall": "检查 ES 召回索引、术语别名、指标定义和 source type 覆盖",
            "queryGraph": "检查 QueryGraph anchor/dependent、join key、表关系和指标编译",
            "sql": "检查 SQL 字段白名单、商家过滤、时间窗、Doris 执行和修复策略",
            "execution": "比较标准 SQL 与 Agent 结果，检查指标口径、聚合粒度、时间窗和过滤条件",
            "evidence": "检查 EvidenceVerifier 契约、requiredEvidenceIntents 和 blocking gap",
            "answer": "检查 AnswerAgent 是否基于证据完整表达且没有遗漏 gap",
        }.get(layer, "人工复核该评测失败原因")


def compare_execution_rows(
    expected_rows: Iterable[Dict[str, Any]],
    actual_rows: Iterable[Dict[str, Any]],
    columns: Iterable[str] = (),
    key_columns: Iterable[str] = (),
    order_sensitive: bool = False,
    numeric_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compare result sets by values rather than SQL text."""
    selected_columns = [str(item) for item in columns if str(item)]
    keys = [str(item) for item in key_columns if str(item)]
    expected = [_project_row(row, selected_columns) for row in expected_rows if isinstance(row, dict)]
    actual = [_project_row(row, selected_columns) for row in actual_rows if isinstance(row, dict)]
    reasons: List[str] = []
    if len(expected) != len(actual):
        reasons.append("row_count_mismatch:expected=%d,actual=%d" % (len(expected), len(actual)))
    if keys:
        expected = sorted(expected, key=lambda row: _row_key(row, keys))
        actual = sorted(actual, key=lambda row: _row_key(row, keys))
    elif not order_sensitive:
        expected = sorted(expected, key=_stable_row_text)
        actual = sorted(actual, key=_stable_row_text)
    mismatch_count = 0
    mismatch_samples: List[Dict[str, Any]] = []
    for index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        row_reasons = _compare_row(expected_row, actual_row, numeric_tolerance)
        if row_reasons:
            mismatch_count += 1
            if len(mismatch_samples) < 5:
                mismatch_samples.append(
                    {
                        "index": index,
                        "reasons": row_reasons,
                        "expected": expected_row,
                        "actual": actual_row,
                    }
                )
    if mismatch_count:
        reasons.append("row_value_mismatch:%d" % mismatch_count)
    return {
        "matched": not reasons,
        "expectedRowCount": len(expected),
        "actualRowCount": len(actual),
        "mismatchCount": mismatch_count,
        "numericTolerance": numeric_tolerance,
        "orderSensitive": order_sensitive,
        "keyColumns": keys,
        "reasons": reasons,
        "mismatchSamples": mismatch_samples,
    }


def _project_row(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    if columns:
        return {column: _json_safe_value(row.get(column)) for column in columns}
    return {
        str(key): _json_safe_value(value)
        for key, value in sorted(row.items(), key=lambda item: str(item[0]))
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _row_key(row: Dict[str, Any], keys: List[str]) -> str:
    return json.dumps([row.get(key) for key in keys], ensure_ascii=False, default=str)


def _stable_row_text(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, default=str, sort_keys=True)


def _compare_row(expected: Dict[str, Any], actual: Dict[str, Any], tolerance: float) -> List[str]:
    reasons: List[str] = []
    if set(expected) != set(actual):
        reasons.append("columns_mismatch")
    for column in sorted(set(expected) | set(actual)):
        if not _values_equal(expected.get(column), actual.get(column), tolerance):
            reasons.append("value_mismatch:%s" % column)
    return reasons


def _values_equal(expected: Any, actual: Any, tolerance: float) -> bool:
    if expected is None or actual is None:
        return expected is actual
    expected_number = _decimal_value(expected)
    actual_number = _decimal_value(actual)
    if expected_number is not None and actual_number is not None:
        difference = abs(expected_number - actual_number)
        return difference <= Decimal(str(max(0.0, tolerance))) or math.isclose(
            float(expected_number), float(actual_number), abs_tol=max(0.0, tolerance), rel_tol=0.0
        )
    return str(expected).strip() == str(actual).strip()


def _decimal_value(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None
