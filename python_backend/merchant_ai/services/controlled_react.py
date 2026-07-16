from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from merchant_ai.models import AgentRunResult, PlanningAssetPack


class ControlledReactExplorer:
    """Normalizes model hypotheses and records evidence without choosing a path."""

    MAX_HYPOTHESES = 3

    def build_hypotheses(
        self,
        question: str,
        pack: PlanningAssetPack,
        fast_understanding: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return only hypotheses explicitly supplied by a model-produced contract."""

        del question
        structured = fast_understanding or {}
        supplied = model_supplied_hypotheses(structured)
        hypotheses: List[Dict[str, Any]] = []
        for index, item in enumerate(supplied[: self.MAX_HYPOTHESES], start=1):
            title = str(item.get("title") or item.get("statement") or item.get("objective") or "").strip()
            if not title:
                continue
            hypotheses.append(
                {
                    "hypothesisId": str(item.get("hypothesisId") or item.get("hypothesis_id") or f"hyp_{index}"),
                    "title": title,
                    "reason": str(item.get("reason") or item.get("rationale") or "").strip(),
                    "metricHints": string_list(item.get("metricHints") or item.get("metric_hints"))[:4],
                    "requiredEvidence": string_list(item.get("requiredEvidence") or item.get("required_evidence"))[:4],
                    "status": "candidate",
                    "source": str(item.get("source") or "lead_or_planner_model"),
                }
            )
        return {
            "mode": "model_supplied_hypothesis_exploration",
            "budget": {"maxHypotheses": self.MAX_HYPOTHESES},
            "questionSignals": {
                "analysisIntent": str(structured.get("analysisIntent") or structured.get("analysis_intent") or ""),
                "requiresExplanation": structured_bool(
                    structured.get("requiresExplanation", structured.get("requires_explanation"))
                ),
                "requiredEvidenceIntentCount": len(
                    list_value(
                        structured.get("requiredEvidenceIntents")
                        or structured.get("required_evidence_intents")
                    )
                ),
                "intentKind": str(structured.get("intentKind") or structured.get("intent_kind") or ""),
            },
            "assetCoverage": {
                "tables": len(pack.known_tables()),
                "metrics": len(pack.metrics),
                "relationships": len(pack.relationships),
            },
            "hypotheses": hypotheses,
            "guardrails": ["model_supplied_only", "observation_only", "no_automatic_selection"],
        }

    def run_parallel_evidence_reviews(
        self,
        hypotheses: Dict[str, Any],
        run_result: AgentRunResult,
    ) -> List[Dict[str, Any]]:
        """Review model candidates independently; do not rank or advance them."""

        rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
        candidates = list((hypotheses or {}).get("hypotheses") or [])[: self.MAX_HYPOTHESES]
        candidates = [item for item in candidates if isinstance(item, dict)]
        if not candidates:
            return []
        with ThreadPoolExecutor(max_workers=min(self.MAX_HYPOTHESES, len(candidates))) as executor:
            futures = [executor.submit(self._review_hypothesis, item, rows) for item in candidates]
            return [future.result() for future in futures]

    def build_execution_gate_ledger(self, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Expose validation and evidence facts to the Lead Agent without deciding."""

        observations: List[Dict[str, Any]] = []
        for execution in executions:
            run_result = execution.get("runResult")
            validation = execution.get("validation")
            verified = getattr(run_result, "verified_evidence", None)
            task_results = list(getattr(run_result, "task_results", []) or [])
            successful = [item for item in task_results if not item.query_bundle.failed]
            failed = [item for item in task_results if item.query_bundle.failed]
            merged_bundle = getattr(run_result, "merged_query_bundle", None)
            row_count = int(merged_bundle.effective_row_count() if merged_bundle is not None else 0)
            validation_gaps = list(getattr(validation, "gaps", []) or [])
            evidence_gaps = list(getattr(verified, "gaps", []) or [])
            observations.append(
                {
                    "hypothesisId": str(execution.get("hypothesisId") or ""),
                    "checks": {
                        "queryGraphValid": bool(getattr(validation, "valid", False)),
                        "evidenceVerified": bool(getattr(verified, "passed", False)),
                        "resultRowsPresent": row_count > 0,
                    },
                    "rowCount": row_count,
                    "successfulTaskCount": len(successful),
                    "failedTaskCount": len(failed),
                    "coveredEvidenceCount": len(getattr(verified, "covered_evidence", []) or []),
                    "validationGapCount": len(validation_gaps),
                    "evidenceGapCount": len(evidence_gaps),
                }
            )
        return {
            "mode": "validation_evidence_gate_ledger",
            "observations": observations,
            "automaticSelection": False,
            "decisionOwner": "lead_react_model",
        }

    def _review_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metric_hints = [item.lower() for item in string_list(hypothesis.get("metricHints"))]
        available_columns = sorted({str(key) for row in rows[:200] for key in row.keys()})
        matched_columns = [
            column
            for column in available_columns
            if any(hint in column.lower() or column.lower() in hint for hint in metric_hints)
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
        return {
            "hypothesisId": str(hypothesis.get("hypothesisId") or ""),
            "title": str(hypothesis.get("title") or ""),
            "checks": {
                "rowsPresent": bool(rows),
                "hintMatched": bool(matched_columns),
                "numericEvidencePresent": bool(numeric_evidence),
            },
            "rowCount": len(rows),
            "matchedColumns": matched_columns[:8],
            "evidence": numeric_evidence,
            "workerMode": "parallel_isolated_evidence_review",
            "automaticSelection": False,
        }


def structured_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def model_supplied_hypotheses(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    nested = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    candidates = (
        payload.get("hypotheses")
        or payload.get("analysisHypotheses")
        or payload.get("analysis_hypotheses")
        or (nested.get("hypotheses") if isinstance(nested, dict) else None)
        or (nested.get("analysisHypotheses") if isinstance(nested, dict) else None)
        or (nested.get("analysis_hypotheses") if isinstance(nested, dict) else None)
        or []
    )
    return [dict(item) for item in candidates if isinstance(item, dict)]


def list_value(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]
