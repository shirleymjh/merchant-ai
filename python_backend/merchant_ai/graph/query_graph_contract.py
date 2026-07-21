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

_EXECUTABLE_INTENT_FIELDS = (
    "intentType",
    "category",
    "answerMode",
    "planTaskId",
    "taskRole",
    "preferredTable",
    "metricColumn",
    "metricName",
    "metricFormula",
    "metricSpecs",
    "groupByColumn",
    "filterColumn",
    "filterValue",
    "days",
    "limit",
    "requiredEvidence",
    "outputKeys",
    "dependsOnTaskIds",
    "sqlStrategy",
    "sql",
)

# ``question_understanding`` is intentionally an open dictionary because its
# vocabulary is published by semantic assets.  The graph identity nevertheless
# needs an explicit boundary: only values consumed as execution/evidence
# contracts belong here.  Provider traces, recovery candidates and display
# prose must not turn a no-op Repair into structural progress.
_UNDERSTANDING_CONTRACT_FIELDS = {
    "analysisGrain": ("analysisGrain", "analysis_grain"),
    "analysisIntent": ("analysisIntent", "analysis_intent"),
    "requiresExplanation": ("requiresExplanation", "requires_explanation"),
    "requiredEvidenceIntents": ("requiredEvidenceIntents", "required_evidence_intents"),
    "anchorMetric": ("anchorMetric", "anchor_metric"),
    "rankingObjective": ("rankingObjective", "ranking_objective"),
    "supportMetrics": ("supportMetrics", "support_metrics"),
    "requestedMeasures": ("requestedMeasures", "requested_measures"),
    "metricCandidateDecisions": ("metricCandidateDecisions", "metric_candidate_decisions"),
    "metricPhrases": ("metricPhrases", "metric_phrases"),
    "originalMetricPhrases": ("originalMetricPhrases", "original_metric_phrases"),
    "metricObligations": ("metricObligations", "metric_obligations"),
    "calculationIntents": ("calculationIntents", "calculation_intents"),
    "scopeConstraints": ("scopeConstraints", "scope_constraints"),
    "filters": ("filters", "filter"),
    "semanticQuery": ("semanticQuery", "semantic_query"),
    "sourceConditionLedger": ("sourceConditionLedger", "source_condition_ledger"),
    "timeWindowDays": ("timeWindowDays", "time_window_days"),
    "timeRange": ("timeRange", "time_range"),
    "timeWindowContract": ("timeWindowContract", "time_window_contract"),
    "selectedMetrics": ("selectedMetrics", "selected_metrics"),
    "selectedRefs": ("selectedRefs", "selected_refs", "semanticSelectionRefs"),
    "diagnosticDriverContracts": ("diagnosticDriverContracts", "diagnostic_driver_contracts"),
    "allowDegradedHypothesisExploration": (
        "allowDegradedHypothesisExploration",
        "allow_degraded_hypothesis_exploration",
    ),
    "suppressDefaultTrendContext": ("suppressDefaultTrendContext", "suppress_default_trend_context"),
    "skillWorkflow": ("skillWorkflow", "skill_workflow"),
    "planningContract": ("planningContract", "planning_contract"),
    "queryContract": ("queryContract", "query_contract"),
}

_ENTITY_REFERENCE_CONTRACT_FIELDS = (
    "semanticRefId",
    "field",
    "table",
    "rawValue",
    "values",
    "valueType",
    "comparisonPolicy",
    "candidateRefIds",
    "status",
    "placeholder",
    "timeScopeExplicit",
    "lookupTimePolicy",
)

_TIME_RANGE_CONTRACT_FIELDS = (
    "kind",
    "startDate",
    "endDate",
    "days",
    "timezone",
    "calendarAnchorPolicy",
    "dataAsOfPolicy",
    "explicit",
    "windowRole",
    "offsetDays",
    "comparisonType",
    "executionStartDate",
    "executionEndDate",
    "executionStartValue",
    "executionEndValue",
    "executionBoundaryPolicy",
)

_METRIC_RESOLUTION_CONTRACT_FIELDS = {
    "requestedMetricRef": ("requestedMetricRef", "requested_metric_ref"),
    "metricKey": ("metricKey", "metric_key"),
    "ownerTable": ("ownerTable", "owner_table"),
    "formula": ("formula",),
    "originalFormula": ("originalFormula", "original_formula"),
    "sourceColumns": ("sourceColumns", "source_columns"),
    "droppedSourceColumns": ("droppedSourceColumns", "dropped_source_columns"),
    "sourceMetricRefs": ("sourceMetricRefs", "source_metric_refs"),
    "unit": ("unit",),
    "metricGrain": ("metricGrain", "metric_grain"),
    "metricIntent": ("metricIntent", "metric_intent"),
    "timeColumn": ("timeColumn", "time_column"),
    "aggregationPolicy": ("aggregationPolicy", "aggregation_policy"),
    "applicableTimeGrain": ("applicableTimeGrain", "applicable_time_grain"),
    "timeSemantics": ("timeSemantics", "time_semantics"),
    "missingValuePolicy": ("missingValuePolicy", "missing_value_policy"),
    "zeroValueMeaning": ("zeroValueMeaning", "zero_value_meaning"),
    "semanticRefId": ("semanticRefId", "semantic_ref_id"),
    "semanticContract": ("semanticContract", "semantic_contract"),
    "semanticContractHash": ("semanticContractHash", "semantic_contract_hash"),
    "metricGovernanceMode": ("metricGovernanceMode", "metric_governance_mode"),
    "assetRefId": ("assetRefId", "asset_ref_id"),
    "contractProvenance": ("contractProvenance", "contract_provenance"),
    "groupByColumn": ("groupByColumn", "group_by_column"),
    "computeStrategy": ("computeStrategy", "compute_strategy"),
    "derivedMetric": ("derivedMetric", "derived_metric"),
    "componentMetricKeys": ("componentMetricKeys", "component_metric_keys"),
    "sourceMetricTaskId": ("sourceMetricTaskId", "source_metric_task_id"),
    "bridgeTaskId": ("bridgeTaskId", "bridge_task_id"),
    "projectionDimensions": ("projectionDimensions", "projection_dimensions"),
    "supportOnly": ("supportOnly", "support_only"),
    "internalOnly": ("internalOnly", "internal_only"),
    "localCompilationPolicy": ("localCompilationPolicy", "local_compilation_policy"),
    "timeWindowRole": ("timeWindowRole", "time_window_role"),
    "timeWindowGrain": ("timeWindowGrain", "time_window_grain"),
    "timeWindowContract": ("timeWindowContract", "time_window_contract"),
    "entityColumns": ("entityColumns", "entity_columns"),
    "filterConditions": ("filterConditions", "filter_conditions"),
    "dependencyFields": ("dependencyFields", "dependency_fields"),
    # These fields participate in validation/evidence gating even though they
    # do not directly alter SQL text.
    "confidence": ("confidence",),
    "resolutionSource": ("resolutionSource", "resolution_source"),
    "fieldWarning": ("fieldWarning", "field_warning"),
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


def query_graph_structure_fingerprint(plan: QueryPlan | None) -> str:
    """Return only executable/evidence-bearing QueryGraph identity.

    This deliberately differs from :func:`query_graph_fingerprint`: adding a
    pending knowledge request must invalidate an old validation snapshot, but
    must never be reported as a successful structural graph repair.
    """

    payload = query_graph_executable_contract(plan)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def query_graph_executable_contract(plan: QueryPlan | None) -> dict[str, Any]:
    """Project a QueryPlan onto fields that can change execution or proof.

    This is an allow-list contract rather than a growing deny-list.  New
    display, trace or orchestration fields therefore remain non-structural by
    default; a new executable field must be deliberately admitted here with a
    contract test.
    """

    graph = plan or QueryPlan()
    raw = graph.model_dump(by_alias=True, mode="json")
    understanding = _project_alias_contract(
        raw.get("questionUnderstanding") or {},
        _UNDERSTANDING_CONTRACT_FIELDS,
    )
    if "semanticQuery" in understanding:
        understanding["semanticQuery"] = semantic_query_execution_contract(understanding["semanticQuery"])
    if "sourceConditionLedger" in understanding:
        understanding["sourceConditionLedger"] = _project_source_condition_ledger(
            understanding["sourceConditionLedger"]
        )
    return {
        "intents": [_project_executable_intent(item) for item in raw.get("intents") or []],
        "dependencies": raw.get("dependencies") or [],
        "evidenceContracts": raw.get("evidenceContracts") or [],
        "finalRequiredEvidence": raw.get("finalRequiredEvidence") or [],
        "finalEvidenceColumnHints": raw.get("finalEvidenceColumnHints") or {},
        "entityFilterObligations": [
            _project_entity_filter_obligation(item)
            for item in raw.get("entityFilterObligations") or []
        ],
        "semanticFilterObligations": sorted(
            [
                _project_semantic_filter_obligation(item)
                for item in raw.get("semanticFilterObligations") or []
            ],
            key=_canonical_payload_sort_key,
        ),
        "questionUnderstanding": understanding,
    }


def _project_executable_intent(intent: Any) -> dict[str, Any]:
    source = intent if isinstance(intent, dict) else {}
    projected = {field: source.get(field) for field in _EXECUTABLE_INTENT_FIELDS}
    projected["entityReference"] = _project_fields(
        source.get("entityReference") or {},
        _ENTITY_REFERENCE_CONTRACT_FIELDS,
    )
    projected["timeRange"] = _project_fields(
        source.get("timeRange") or {},
        _TIME_RANGE_CONTRACT_FIELDS,
    )
    projected["metricResolution"] = _project_metric_resolution(source.get("metricResolution") or {})
    projected["semanticQuery"] = semantic_query_execution_contract(source.get("semanticQuery") or {})
    # A KnowledgeRef's label, reason and retrieval score are context metadata;
    # the governed reference identity is the evidence contract.
    projected["knowledgeRefs"] = [
        _project_fields(item, ("refId", "refType", "table", "column", "relationshipId"))
        for item in source.get("knowledgeRefs") or []
        if isinstance(item, dict)
    ]
    return projected


def _project_entity_filter_obligation(obligation: Any) -> dict[str, Any]:
    source = obligation if isinstance(obligation, dict) else {}
    return {
        "obligationId": source.get("obligationId"),
        "taskId": source.get("taskId"),
        "required": source.get("required"),
        "reference": _project_fields(
            source.get("reference") or {},
            _ENTITY_REFERENCE_CONTRACT_FIELDS,
        ),
        "status": source.get("status"),
    }


def _project_semantic_filter_obligation(obligation: Any) -> dict[str, Any]:
    source = obligation if isinstance(obligation, dict) else {}
    projected = _project_fields(
        source,
        (
            "taskId",
            "semanticRefId",
            "operator",
            "rawValues",
            "resolvedValues",
            "boundTable",
            "boundField",
            "memberKind",
            "dataType",
            "required",
            "status",
        ),
    )
    if str(projected.get("operator") or "").lower() in {"in", "not_in"}:
        projected["rawValues"] = _canonical_unordered_values(projected.get("rawValues"))
        projected["resolvedValues"] = _canonical_unordered_values(projected.get("resolvedValues"))
    return projected


def semantic_query_execution_contract(query: Any) -> dict[str, Any]:
    """Canonical SQL-bearing subset of a SemanticQuery payload.

    Node ids, source prose, knowledge refs, selections, relationships and
    ordering are deliberately absent.  The latter fields are not consumed by
    the current compiler and are rejected by validation instead of being
    allowed to manufacture Repair progress.
    """

    source = query if isinstance(query, dict) else {}
    projected = _project_fields(
        source,
        (
            "resultMode",
            "limit",
        ),
    )
    projected["filterExpression"] = _canonical_filter_expression(
        source.get("filterNodes") or source.get("filter_nodes") or [],
        source.get("rootFilterNodeId") or source.get("root_filter_node_id") or "",
        include_binding=True,
    )
    return projected


def _project_source_condition_ledger(ledger: Any) -> dict[str, Any]:
    source = ledger if isinstance(ledger, dict) else {}
    return {
        "auditorStatus": source.get("auditorStatus") or source.get("auditor_status"),
        "questionHash": source.get("questionHash") or source.get("question_hash"),
        "conditionExpression": _canonical_filter_expression(
            source.get("conditionNodes") or source.get("condition_nodes") or [],
            source.get("rootConditionNodeId") or source.get("root_condition_node_id") or "",
            include_binding=False,
        ),
    }


def _canonical_filter_expression(
    raw_nodes: Any,
    root_id: Any,
    *,
    include_binding: bool,
) -> Any:
    nodes = [item for item in raw_nodes if isinstance(item, dict)] if isinstance(raw_nodes, list) else []
    by_id: dict[str, dict[str, Any]] = {}
    for item in nodes:
        node_id = str(item.get("nodeId") or item.get("node_id") or "")
        if node_id and node_id not in by_id:
            by_id[node_id] = item

    visiting: set[str] = set()

    def visit(node_id: str) -> Any:
        if node_id in visiting:
            return {"invalid": "cycle"}
        item = by_id.get(node_id)
        if item is None:
            return {"invalid": "unknown_child"}
        node_type = str(item.get("nodeType") or item.get("node_type") or "predicate").lower()
        if node_type == "predicate":
            operator = str(item.get("operator") or "").lower()
            raw_values = item.get("rawValues") or item.get("raw_values") or []
            resolved_values = item.get("resolvedValues") or item.get("resolved_values") or []
            payload: dict[str, Any] = {
                "type": "predicate",
                "operator": operator,
                "rawValues": (
                    _canonical_unordered_values(raw_values)
                    if operator in {"in", "not_in"}
                    else list(raw_values) if isinstance(raw_values, list) else []
                ),
            }
            if include_binding:
                payload.update(
                    {
                        "semanticRefId": item.get("semanticRefId") or item.get("semantic_ref_id"),
                        "resolvedValues": (
                            _canonical_unordered_values(resolved_values)
                            if operator in {"in", "not_in"}
                            else list(resolved_values) if isinstance(resolved_values, list) else []
                        ),
                        "boundTable": item.get("boundTable") or item.get("bound_table"),
                        "boundField": item.get("boundField") or item.get("bound_field"),
                        "memberKind": item.get("memberKind") or item.get("member_kind"),
                        "dataType": item.get("dataType") or item.get("data_type"),
                    }
                )
            return payload
        if node_type != "group":
            return {"invalid": "node_type"}
        logical = str(item.get("logicalOperator") or item.get("logical_operator") or "").lower()
        children = item.get("childNodeIds") or item.get("child_node_ids") or []
        visiting.add(node_id)
        expressions = [visit(str(child)) for child in children if str(child or "")]
        visiting.discard(node_id)
        if logical in {"and", "or"}:
            expressions = sorted(expressions, key=_canonical_expression_sort_key)
        return {"type": "group", "logicalOperator": logical, "children": expressions}

    root = str(root_id or "")
    if root:
        return visit(root)
    # Invalid/incomplete contracts still get a deterministic identity without
    # making list order look like an executable change.
    expressions = [visit(node_id) for node_id in sorted(by_id)]
    return sorted(expressions, key=_canonical_expression_sort_key)


def _canonical_expression_sort_key(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _legacy_project_semantic_query_nodes(query: Any) -> list[dict[str, Any]]:
    """Compatibility helper retained for older callers during migration."""

    source = query if isinstance(query, dict) else {}
    nodes: list[dict[str, Any]] = []
    for item in source.get("filterNodes") or []:
        if not isinstance(item, dict):
            continue
        node = _project_fields(
            item,
            (
                "nodeId",
                "nodeType",
                "semanticRefId",
                "sourcePhrase",
                "operator",
                "rawValues",
                "resolvedValues",
                "boundTable",
                "boundField",
                "memberKind",
                "dataType",
                "resolutionStatus",
                "logicalOperator",
                "childNodeIds",
                "knowledgeRefIds",
            ),
        )
        operator = str(node.get("operator") or "").lower()
        logical = str(node.get("logicalOperator") or "").lower()
        if operator in {"in", "not_in"}:
            node["rawValues"] = _canonical_unordered_values(node.get("rawValues"))
            node["resolvedValues"] = _canonical_unordered_values(node.get("resolvedValues"))
        if logical in {"and", "or"}:
            node["childNodeIds"] = sorted(set(node.get("childNodeIds") or []))
        node["knowledgeRefIds"] = sorted(set(node.get("knowledgeRefIds") or []))
        nodes.append(node)
    return sorted(nodes, key=_canonical_payload_sort_key)


def _canonical_unordered_values(values: Any) -> list[Any]:
    payload = list(values) if isinstance(values, list) else []
    deduped: dict[str, Any] = {}
    for value in payload:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        deduped.setdefault(encoded, value)
    return [deduped[key] for key in sorted(deduped)]


def _canonical_payload_sort_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _project_metric_resolution(resolution: Any) -> dict[str, Any]:
    source = resolution if isinstance(resolution, dict) else {}
    projected = _project_alias_contract(source, _METRIC_RESOLUTION_CONTRACT_FIELDS)
    components = source.get("componentMetrics") or source.get("component_metrics")
    if isinstance(components, list):
        projected["componentMetrics"] = [
            _project_metric_resolution(item) for item in components if isinstance(item, dict)
        ]
    return projected


def _project_alias_contract(
    source: Any,
    fields: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    payload = source if isinstance(source, dict) else {}
    projected: dict[str, Any] = {}
    for canonical, aliases in fields.items():
        for alias in aliases:
            if alias in payload:
                projected[canonical] = payload[alias]
                break
    return projected


def _project_fields(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    payload = source if isinstance(source, dict) else {}
    return {field: payload.get(field) for field in fields}


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
