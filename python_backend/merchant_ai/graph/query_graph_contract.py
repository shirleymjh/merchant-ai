from __future__ import annotations

import hashlib
import json
from typing import Any, MutableMapping

from merchant_ai.models import GraphValidationResult, QueryPlan


VALIDATION_NOT_RUN = "not_run"
VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"

_NON_EXECUTION_PLAN_FIELDS = {
    "agent_trace",
    "compiler_trace",
    "planner_tool_calls",
    "planner_tool_results",
    "planner_loaded_refs",
    "planner_context_files",
    "planner_prompt_stats",
}


def query_graph_fingerprint(plan: QueryPlan | None) -> str:
    """Return the identity of the executable and evidence-bearing graph contract."""

    graph = plan or QueryPlan()
    payload = graph.model_dump(
        by_alias=True,
        mode="json",
        exclude=_NON_EXECUTION_PLAN_FIELDS,
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_graph_validation(
    state: MutableMapping[str, Any],
    result: GraphValidationResult,
    plan: QueryPlan | None = None,
) -> None:
    """Record one deterministic validation attempt for exactly one graph version."""

    state["query_graph_validation_result"] = result
    state["query_graph_validation_status"] = VALIDATION_PASSED if result.valid else VALIDATION_FAILED
    state["validated_query_graph_fingerprint"] = query_graph_fingerprint(plan or state.get("plan"))
    state["query_graph_validation_attempted"] = True
    state["query_graph_validation_passed"] = bool(result.valid)
    # Compatibility flag is deliberately derived from *passed*, never attempted.
    state["query_graph_validated"] = bool(result.valid)


def invalidate_graph_validation(state: MutableMapping[str, Any]) -> None:
    state["query_graph_validation_result"] = GraphValidationResult()
    state["query_graph_validation_status"] = VALIDATION_NOT_RUN
    state["validated_query_graph_fingerprint"] = ""
    state["query_graph_validation_attempted"] = False
    state["query_graph_validation_passed"] = False
    state["query_graph_validated"] = False


def mark_graph_validation_stale(state: MutableMapping[str, Any]) -> None:
    """Detach an old validation from a graph that has since changed, preserving its audit payload."""

    state["query_graph_validation_status"] = VALIDATION_NOT_RUN
    state["query_graph_validation_attempted"] = False
    state["query_graph_validation_passed"] = False
    state["query_graph_validated"] = False


def graph_validation_matches_current_plan(state: MutableMapping[str, Any]) -> bool:
    expected = str(state.get("validated_query_graph_fingerprint") or "")
    return bool(expected and expected == query_graph_fingerprint(state.get("plan")))


def graph_validation_attempted(state: MutableMapping[str, Any]) -> bool:
    status = str(state.get("query_graph_validation_status") or VALIDATION_NOT_RUN)
    return status in {VALIDATION_PASSED, VALIDATION_FAILED} and graph_validation_matches_current_plan(state)


def graph_validation_passed(state: MutableMapping[str, Any]) -> bool:
    result = state.get("query_graph_validation_result")
    return bool(
        str(state.get("query_graph_validation_status") or "") == VALIDATION_PASSED
        and graph_validation_matches_current_plan(state)
        and isinstance(result, GraphValidationResult)
        and result.valid
    )


def graph_validation_failure_reason(state: MutableMapping[str, Any]) -> str:
    status = str(state.get("query_graph_validation_status") or VALIDATION_NOT_RUN)
    if status == VALIDATION_NOT_RUN:
        return "QUERY_GRAPH_VALIDATION_NOT_RUN"
    if not graph_validation_matches_current_plan(state):
        return "QUERY_GRAPH_CHANGED_AFTER_VALIDATION"
    result = state.get("query_graph_validation_result")
    if status != VALIDATION_PASSED or not isinstance(result, GraphValidationResult) or not result.valid:
        return "QUERY_GRAPH_VALIDATION_FAILED"
    return ""
