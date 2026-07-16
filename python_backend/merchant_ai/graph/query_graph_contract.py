from __future__ import annotations

import hashlib
import json
from typing import Any, MutableMapping

from merchant_ai.models import AgentRunResult, EvidenceGap, GraphValidationGap, GraphValidationResult, QueryPlan, VerifiedEvidence


VALIDATION_NOT_RUN = "not_run"
VALIDATION_PASSED = "passed"
VALIDATION_FAILED = "failed"

_VALIDATION_EVIDENCE_SOURCE = "query_graph_validator"
_VALIDATION_PARTIAL_PREFIX = "QueryGraph 完整性校验未通过"

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
    project_graph_validation_evidence(state, result)


def graph_validation_evidence_gaps(result: GraphValidationResult) -> list[EvidenceGap]:
    """Translate graph-contract failures into the answer/evidence contract."""

    validation_gaps = list(result.gaps) or [
        GraphValidationGap(
            code="QUERY_GRAPH_VALIDATION_FAILED",
            reason="QueryGraph validation failed without a more specific gap contract",
        )
    ]
    return [
        EvidenceGap(
            code=gap.code,
            task_id=gap.task_id,
            evidence=gap.evidence,
            reason=gap.reason or "QueryGraph validation failed before SQL execution",
            severity="blocking",
            disclosure_required=True,
            source=_VALIDATION_EVIDENCE_SOURCE,
            answer_instruction=(
                "QueryGraph 校验缺口 %s 尚未修复，不能执行查询或输出完整业务结论。%s"
                % (gap.code, (" " + gap.reason) if gap.reason else "")
            ).strip(),
            details={
                "gapCode": gap.code,
                "taskId": gap.task_id,
                "evidence": gap.evidence,
                "source": _VALIDATION_EVIDENCE_SOURCE,
                "severity": "blocking",
                "validationStage": "query_graph",
            },
        )
        for gap in validation_gaps
    ]


def project_graph_validation_evidence(
    state: MutableMapping[str, Any],
    result: GraphValidationResult,
) -> None:
    """Keep validation failures visible to verification, answering, and final audit."""

    raw_run_result = state.get("agent_run_result")
    if isinstance(raw_run_result, AgentRunResult):
        run_result = raw_run_result
    elif isinstance(raw_run_result, dict):
        run_result = AgentRunResult.model_validate(raw_run_result)
    else:
        run_result = AgentRunResult()

    retained = [
        gap
        for gap in run_result.evidence_gaps
        if str(getattr(gap, "source", "") or "") != _VALIDATION_EVIDENCE_SOURCE
    ]
    projected = [] if result.valid else graph_validation_evidence_gaps(result)
    run_result.evidence_gaps = _dedupe_evidence_gaps([*retained, *projected])

    verified = run_result.verified_evidence or VerifiedEvidence()
    retained_verified = [
        gap
        for gap in verified.gaps
        if str(getattr(gap, "source", "") or "") != _VALIDATION_EVIDENCE_SOURCE
    ]
    retained_blocking = [
        gap
        for gap in verified.blocking_gaps
        if str(getattr(gap, "source", "") or "") != _VALIDATION_EVIDENCE_SOURCE
    ]
    verified.gaps = _dedupe_evidence_gaps([*retained_verified, *projected])
    verified.blocking_gaps = _dedupe_evidence_gaps([*retained_blocking, *projected])

    if result.valid:
        if run_result.partial_answer_reason.startswith(_VALIDATION_PARTIAL_PREFIX):
            run_result.partial_answer_reason = ""
        if verified.partial_answer_reason.startswith(_VALIDATION_PARTIAL_PREFIX):
            verified.partial_answer_reason = ""
        verified.answer_guard_required = bool(verified.blocking_gaps)
    else:
        reason = _graph_validation_partial_reason(result)
        run_result.partial_answer_reason = reason
        verified.passed = False
        verified.answer_guard_required = True
        verified.partial_answer_reason = reason

    run_result.verified_evidence = verified
    state["agent_run_result"] = run_result


def _graph_validation_partial_reason(result: GraphValidationResult) -> str:
    summaries = []
    for gap in graph_validation_evidence_gaps(result)[:4]:
        detail = str(gap.reason or gap.evidence or "").strip()
        summary = "%s%s" % (gap.code, (": " + detail) if detail else "")
        if summary not in summaries:
            summaries.append(summary)
    return "%s：%s" % (_VALIDATION_PARTIAL_PREFIX, "；".join(summaries))


def _dedupe_evidence_gaps(gaps: list[EvidenceGap]) -> list[EvidenceGap]:
    deduped: list[EvidenceGap] = []
    seen: set[tuple[str, str, str, str]] = set()
    for gap in gaps:
        key = (
            str(gap.code or gap.gap_code or ""),
            str(gap.task_id or gap.source_node_id or ""),
            str(gap.evidence or ""),
            str(gap.source or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gap)
    return deduped


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
