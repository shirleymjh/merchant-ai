from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class AgentToolDefinition:
    """Runtime tool schema exposed to tool-calling LLMs."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def trace_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolCapability:
    """Operational contract for a runtime tool."""

    name: str
    description: str = ""
    permission: str = "agent.tool.execute"
    side_effect_level: str = "none"
    sandbox_required: bool = False
    cache_policy: str = "disabled"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_required_keys: List[str] = field(default_factory=list)
    fail_closed: bool = False
    failure_modes: List[str] = field(default_factory=list)
    cost_hint: str = "low"

    def trace(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "sideEffectLevel": self.side_effect_level,
            "sandboxRequired": self.sandbox_required,
            "cachePolicy": self.cache_policy,
            "outputRequiredKeys": list(self.output_required_keys),
            "failClosed": self.fail_closed,
            "failureModes": list(self.failure_modes),
            "costHint": self.cost_hint,
        }


class ToolRegistry:
    """Capability-card registry for deferred tool discovery and runtime validation."""

    def __init__(self, capabilities: Iterable[ToolCapability] | None = None):
        self._capabilities: Dict[str, ToolCapability] = {}
        for capability in capabilities or []:
            self.register(capability)

    def register(self, capability: ToolCapability) -> None:
        self._capabilities[capability.name] = capability

    def capability(self, name: str) -> ToolCapability:
        return self._capabilities.get(str(name or ""), default_tool_capability(str(name or "")))

    def catalog(self, names: Iterable[str] | None = None) -> List[Dict[str, Any]]:
        selected = set(names or self._capabilities.keys())
        return [self.capability(name).trace() for name in sorted(selected) if name]

    def names(self) -> List[str]:
        return sorted(self._capabilities.keys())


def default_tool_capability(name: str, description: str = "") -> ToolCapability:
    tool_name = str(name or "")
    # An unregistered tool has unknown effects.  Treating it as a read would
    # permit unsafe lease takeover and result caching for newly introduced
    # writes before their capability card is reviewed.
    side_effect = "unknown"
    permission = "agent.tool.execute"
    cache_policy = "disabled"
    sandbox_required = False
    output_required: List[str] = []
    fail_closed = True
    failure_modes = [
        "TIMEOUT",
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "UNCLASSIFIED_SIDE_EFFECT",
    ]
    if tool_name in {"execute_sql", "doris_query"}:
        side_effect = "external_read"
        permission = "agent.sql.execute"
        cache_policy = "ttl"
        output_required = ["rows"]
        fail_closed = True
        failure_modes += ["UNKNOWN_COLUMN", "MEM_ALLOC_FAILED", "UNSAFE_SQL"]
    elif tool_name.startswith("artifact_"):
        side_effect = "read"
        permission = "agent.artifact.read"
        cache_policy = "ttl"
        sandbox_required = True
        if tool_name == "artifact_write":
            side_effect = "workspace_write"
            permission = "agent.artifact.write"
            output_required = ["path"]
            fail_closed = True
    elif tool_name.startswith("semantic_"):
        side_effect = "read"
        permission = "agent.semantic.read"
        cache_policy = "versioned"
        if tool_name == "semantic_write":
            side_effect = "governed_write"
            permission = "agent.semantic.propose"
            output_required = ["path"]
            fail_closed = True
    elif tool_name.startswith("draft_") or tool_name in {
        "repair_sql",
        "summarize_node_result",
        "execution_contract_validation",
    }:
        side_effect = "none"
        permission = "agent.reasoning"
        cache_policy = "disabled"
    elif tool_name in {"inspect_schema", "resolve_columns", "check_freshness"}:
        side_effect = "read"
        permission = "agent.tool.read"
        cache_policy = "versioned"
    elif tool_name in {
        "choose_sql_strategy",
        "validate_sql",
    }:
        side_effect = "none"
        permission = "agent.reasoning"
        cache_policy = "disabled"
    return ToolCapability(
        name=tool_name,
        description=description or tool_name,
        permission=permission,
        side_effect_level=side_effect,
        sandbox_required=sandbox_required,
        cache_policy=cache_policy,
        output_required_keys=output_required,
        fail_closed=fail_closed,
        failure_modes=sorted(set(failure_modes)),
    )


def tool_registry_from_descriptions(tool_registry: Mapping[str, str]) -> ToolRegistry:
    registry = ToolRegistry()
    for name, description in tool_registry.items():
        registry.register(default_tool_capability(str(name), str(description)))
    return registry


RUNTIME_NODE_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "inspect_schema": "inspect asset/live schema available for this node",
    "resolve_columns": "resolve required columns and output keys",
    "execution_contract_validation": "validate the grounded node execution contract before SQL draft",
    "check_freshness": "check the semantic contract's declared time column and fallback risk",
    "choose_sql_strategy": "choose plan-bound LLM SQL or structured fallback",
    "draft_structured_sql": "draft safe one-table structured SQL",
    "draft_llm_sql": "draft one-table SQL with LLM bound to node plan contract",
    "validate_sql": "validate SQL with sqlglot and node scope",
    "execute_sql": "execute SQL in Doris",
    "repair_sql": "repair SQL only, never QueryGraph",
    "summarize_node_result": "summarize rows, entity set, and gaps",
}


def canonical_tool_registry(extra_descriptions: Mapping[str, str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    definitions = (
        semantic_file_tool_definitions()
        + artifact_file_tool_definitions()
        + [sql_draft_tool(), sql_repair_tool(), lead_action_selection_tool([]), delegate_subagent_tool([])]
    )
    for definition in definitions:
        registry.register(default_tool_capability(definition.name, definition.description))
    for name, description in RUNTIME_NODE_TOOL_DESCRIPTIONS.items():
        registry.register(default_tool_capability(name, description))
    for name, description in (extra_descriptions or {}).items():
        registry.register(default_tool_capability(str(name), str(description)))
    return registry


def validate_tool_result_contract(tool_name: str, result: Any, registry: ToolRegistry | None = None) -> Dict[str, Any]:
    capability = (registry or ToolRegistry()).capability(tool_name)
    payload = result if isinstance(result, dict) else {"value": result}
    missing = [key for key in capability.output_required_keys if key not in payload]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "toolName": tool_name,
        "valid": not missing,
        "missingKeys": missing,
        "capability": capability.trace(),
        "enforced": bool(capability.fail_closed),
        "resultHash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
    }


def tool_schema_catalog(tools: Iterable[AgentToolDefinition]) -> List[Dict[str, str]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in tools
    ]


def select_tool_schemas(tools: Iterable[AgentToolDefinition], names: Iterable[str]) -> List[Dict[str, Any]]:
    by_name = {tool.name: tool for tool in tools}
    selected = []
    for name in names:
        tool = by_name.get(str(name or ""))
        if tool is not None:
            selected.append(tool.openai_schema())
    return selected


def deferred_tool_schema_loader_tool(available_names: Iterable[str]) -> AgentToolDefinition:
    names = sorted({str(name) for name in available_names if str(name or "").strip()})
    return AgentToolDefinition(
        name="load_tool_schemas",
        description="Load full schemas for deferred runtime tools by name after inspecting the tool catalog.",
        parameters=object_schema(
            {
                "toolNames": array_property(
                    "tool names to load from the deferred catalog",
                    string_property("tool name", names),
                ),
                "reason": string_property("why these tools are needed for the current step"),
            },
            required=["toolNames", "reason"],
        ),
    )


def object_schema(properties: Mapping[str, Any], required: Iterable[str] | None = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required or []),
        "additionalProperties": False,
    }


def string_property(description: str, enum: Iterable[str] | None = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = list(enum)
    return schema


def integer_property(description: str, minimum: int | None = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def boolean_property(description: str) -> Dict[str, Any]:
    return {"type": "boolean", "description": description}


def array_property(description: str, item_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "array", "description": description, "items": item_schema}


def semantic_asset_selection_tool() -> AgentToolDefinition:
    selected_asset_schema = object_schema(
        {
            "sourcePhrase": string_property("exact metric/business phrase copied from the user question"),
            "semanticRefId": string_property("selected semanticCatalog.candidateMetrics.sourceRefId"),
            "metricRef": string_property("selected semanticCatalog.candidateMetrics.key"),
            "ownerTable": string_property("selected semanticCatalog.candidateMetrics.table"),
            "confidence": string_property("0.0-1.0 confidence as a string"),
            "reason": string_property("brief semantic evidence for the selection"),
        },
        required=["sourcePhrase", "semanticRefId", "metricRef", "ownerTable", "confidence", "reason"],
    )
    gap_schema = object_schema(
        {
            "code": string_property(
                "why selection cannot be completed from current workspace",
                [
                    "AMBIGUOUS_METRIC",
                    "MISSING_SEMANTIC_ASSET",
                    "NEED_SEMANTIC_READ",
                    "UNSUPPORTED_GRAPH_CONTRACT",
                    "LOW_CONFIDENCE",
                    "INVALID",
                ],
            ),
            "sourcePhrase": string_property("metric/business phrase that has the gap"),
            "reason": string_property("brief reason"),
            "candidateIds": array_property("related candidate semanticRefIds", string_property("semanticRefId")),
            "clarificationQuestion": string_property("ask_human question when the candidates are ambiguous"),
        },
        required=["code", "sourcePhrase", "reason", "candidateIds"],
    )
    query_contract_schema = object_schema(
        {
            "contractType": string_property(
                "minimal build contract supported by selected assets",
                ["independent_metrics", "time_series_metrics", "requires_planner"],
            ),
            "timeWindowDays": integer_property("requested time window in days", 1),
            "analysisIntent": string_property("structured analysis intent inherited from planningContract"),
            "timeGrain": string_property("structured time grain such as day when the contract is a time series"),
            "timeSeries": boolean_property("whether the selected metrics must be grouped by their declared time column"),
            "reason": string_property("brief reason for this contract"),
        },
        required=["contractType", "timeWindowDays", "reason"],
    )
    return AgentToolDefinition(
        name="emit_semantic_asset_selection",
        description=(
            "Select semantic metric assets from the current Topic workspace candidates. "
            "Do not emit SQL, QueryGraph, tables outside the candidates, or questionUnderstanding."
        ),
        parameters=object_schema(
            {
                "status": string_property(
                    "selection status",
                    ["SELECTED", "AMBIGUOUS", "NEED_MORE_KNOWLEDGE", "UNSUPPORTED", "INVALID"],
                ),
                "selectedAssets": array_property("one selected semantic asset per explicit user metric phrase", selected_asset_schema),
                "gaps": array_property("selection gaps that require retrieve_knowledge, semantic_read, planner, or ask_human", gap_schema),
                "queryContract": query_contract_schema,
                "reason": string_property("brief summary"),
            },
            required=["status", "selectedAssets", "gaps", "queryContract", "reason"],
        ),
    )


def question_understanding_tool(force_catalog: bool = False) -> AgentToolDefinition:
    status_values = ["UNDERSTOOD", "INVALID"] if force_catalog else ["UNDERSTOOD", "NEED_MORE_KNOWLEDGE", "INVALID"]
    ranking_schema = object_schema(
        {
            "metricRef": string_property("candidate metric key used for sorting; empty for detail lookup"),
            "sourcePhrase": string_property("exact source phrase from the user question"),
            "ownerTable": string_property("metric owner table from semanticCatalog"),
            "objectiveType": string_property("metric objective type", ["metric_total", "ranking", "trend_anchor", "detail_anchor"]),
            "groupByColumn": string_property("grain column selected from semanticCatalog"),
            "order": string_property("sort direction", ["desc", "asc"]),
            "limit": integer_property("top N limit", 1),
        },
        required=["metricRef", "sourcePhrase", "ownerTable"],
    )
    measure_schema = object_schema(
        {
            "metricRef": string_property("candidate metric key"),
            "sourcePhrase": string_property("exact source phrase from the user question"),
            "ownerTable": string_property("metric owner table from semanticCatalog"),
            "resultMode": string_property(
                "typed result contract: metric computes the governed metric; detail requests row-level evidence",
                ["metric", "detail"],
            ),
        },
        required=["metricRef", "sourcePhrase", "ownerTable", "resultMode"],
    )
    metric_candidate_decision_schema = object_schema(
        {
            "phrase": string_property("metric phrase from user"),
            "decision": string_property("candidate arbitration result", ["selected_one", "need_clarification", "need_more_context"]),
            "selectedCandidateId": string_property("selected semanticRefId or ownerTable.metricRef"),
            "selectedMetricRef": string_property("selected metric key"),
            "selectedOwnerTable": string_property("selected owner table"),
            "rejectedCandidateIds": array_property(
                "rejected semanticRefIds or ownerTable.metricRefs",
                string_property("rejected candidate id"),
            ),
            "confidence": string_property("0.0-1.0 confidence as a string"),
            "reason": string_property("brief reason"),
            "clarificationQuestion": string_property("question to ask the user when decision is need_clarification"),
        },
        required=["phrase", "decision", "selectedCandidateId", "rejectedCandidateIds", "reason"],
    )
    scope_schema = object_schema(
        {
            "scopeId": string_property("stable id for this bounded entity set"),
            "sourcePhrase": string_property("exact source phrase that limits the population"),
            "ownerTable": string_property("semanticCatalog table that defines the bounded population"),
            "metricRef": string_property("optional candidate metric key that describes the scope source"),
            "entityGrain": string_property("entity grain declared by the selected semantic asset"),
            "targetDomain": string_property("optional semantic topic that the scope should constrain"),
            "required": boolean_property("true when this scope must be applied before computing anchor/support metrics"),
        },
        required=["sourcePhrase", "ownerTable", "entityGrain"],
    )
    filter_schema = object_schema(
        {
            "field": string_property("legacy equality-filter field from semanticCatalog/live schema"),
            "value": string_property("legacy raw value; use semanticQuery for new multi-condition plans"),
        },
        required=["field", "value"],
    )
    semantic_filter_node_schema = object_schema(
        {
            "nodeId": string_property("unique id within semanticQuery.filterNodes"),
            "nodeType": string_property("predicate or boolean group node", ["predicate", "group"]),
            "semanticRefId": string_property(
                "recalled semantic member ref for a predicate; empty for a group; never emit a physical column guess"
            ),
            "sourcePhrase": string_property("exact user phrase represented by this predicate or group"),
            "operator": string_property(
                "typed predicate operator; empty for a group",
                [
                    "",
                    "eq",
                    "neq",
                    "in",
                    "not_in",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                    "between",
                    "is_null",
                    "is_not_null",
                    "contains",
                    "not_contains",
                    "starts_with",
                    "ends_with",
                ],
            ),
            "rawValues": array_property(
                "unmodified value phrases copied from the user; empty for groups and null checks",
                string_property("one raw user value"),
            ),
            "logicalOperator": string_property("boolean operator for a group; empty for a predicate", ["", "and", "or", "not"]),
            "childNodeIds": array_property(
                "ids of child nodes for a group; empty for a predicate",
                string_property("child node id"),
            ),
            "knowledgeRefIds": array_property(
                "recalled knowledge refs supporting this semantic choice",
                string_property("knowledge ref id"),
            ),
            "reason": string_property("brief reason for this semantic node"),
        },
        required=[
            "nodeId",
            "nodeType",
            "semanticRefId",
            "sourcePhrase",
            "operator",
            "rawValues",
            "logicalOperator",
            "childNodeIds",
            "knowledgeRefIds",
        ],
    )
    semantic_order_schema = object_schema(
        {
            "semanticRefId": string_property("recalled semantic member or metric ref used for ordering"),
            "direction": string_property("sort direction", ["asc", "desc"]),
            "nulls": string_property("null ordering policy", ["auto", "first", "last"]),
        },
        required=["semanticRefId", "direction"],
    )
    semantic_query_schema = object_schema(
        {
            "resultMode": string_property(
                "semantic result shape",
                ["metric", "detail", "topn", "group_agg", "derived"],
            ),
            "filterNodes": array_property(
                "flat predicate/group node graph; express nesting only through childNodeIds",
                semantic_filter_node_schema,
            ),
            "rootFilterNodeId": string_property("root filter node id; empty only when no user filter exists"),
            "selectRefIds": array_property(
                "reserved result projection contract; emit [] until the compiler advertises projection support",
                string_property("semantic ref id"),
            ),
            "measureRefIds": array_property(
                "reserved measure projection contract; emit [] and use anchor/support metrics today",
                string_property("measure semantic ref id"),
            ),
            "dimensionRefIds": array_property(
                "reserved dimension projection contract; emit [] until the compiler advertises support",
                string_property("dimension semantic ref id"),
            ),
            "sourceRefIds": array_property(
                "reserved explicit source contract; emit [] because sources are derived from bound semantic refs",
                string_property("source semantic ref id"),
            ),
            "relationshipRefIds": array_property(
                "reserved governed relationship refs; emit [] until governed JOIN execution is available",
                string_property("relationship semantic ref id"),
            ),
            "joinStrategy": string_property(
                "logical join strategy; physical join keys are resolved from relationship refs",
                ["auto", "single_source"],
            ),
            "orderBy": array_property(
                "reserved semantic ordering contract; emit [] and use rankingObjective today",
                semantic_order_schema,
            ),
            "limit": integer_property("requested row limit; zero means no explicit limit", 0),
            "bindingStatus": string_property(
                "Planner must emit unresolved; governed binding stages update this field later",
                ["unresolved"],
            ),
        },
        required=[
            "resultMode",
            "filterNodes",
            "rootFilterNodeId",
            "selectRefIds",
            "measureRefIds",
            "dimensionRefIds",
            "sourceRefIds",
            "relationshipRefIds",
            "joinStrategy",
            "orderBy",
            "limit",
        ],
    )
    evidence_intent_schema = object_schema(
        {
            "semanticLabel": string_property("semantic evidence label, e.g. explanation_context, risk_driver, comparison_baseline"),
            "reason": string_property("why this evidence is needed for the user's requested analysis"),
            "requiredLevel": string_property("whether the answer needs this evidence", ["required", "optional"]),
            "evidenceMode": string_property(
                "typed evidence contract; detail is the only mode that permits a row-level DETAIL branch",
                ["metric", "detail", "rule", "context"],
            ),
            "suggestedMetricRefs": array_property("candidate semantic metric keys that could satisfy this evidence", string_property("metric key")),
            "suggestedDomains": array_property("candidate semantic domains needed for this evidence", string_property("domain name")),
        },
        required=["semanticLabel", "reason", "requiredLevel", "evidenceMode"],
    )
    understanding_schema = object_schema(
        {
            "analysisGrain": string_property("analysis grain declared by the selected semantic asset or user request"),
            "analysisIntent": string_property(
                "analysis intent declared by the LLM understanding stage",
                ["none", "diagnosis", "trend_check", "risk_ranking", "overview", "comparison", "anomaly_check"],
            ),
            "requiresExplanation": boolean_property("true when the user asks for diagnosis, reasons, anomaly interpretation, risk judgement or summary insight"),
            "requiredEvidenceIntents": array_property(
                "evidence intents required to support the declared analysis intent; must be non-empty when analysisIntent is not none; empty only for simple lookup/ranking questions",
                evidence_intent_schema,
            ),
            "anchorMetric": ranking_schema,
            "supportMetrics": array_property("additional requested metrics", measure_schema),
            "metricCandidateDecisions": array_property(
                "same/near-name metric candidate arbitration before planning",
                metric_candidate_decision_schema,
            ),
            "calculationIntents": array_property(
                "explicit derived calculations requested by the user, such as percentage/proportion/ratio between a scoped denominator and an event numerator",
                object_schema(
                    {
                        "operation": string_property("calculation operation", ["ratio", "percentage", "difference", "comparison"]),
                        "sourcePhrase": string_property("exact phrase from the user expressing this calculation"),
                        "basePopulationPhrase": string_property("exact user phrase naming the base population for a ratio/percentage"),
                        "eventPopulationPhrase": string_property("exact user phrase naming the event/subset numerator population"),
                        "numeratorMetricRef": string_property(
                            "semantic metric key for the event/subset numerator; for ratio/percentage it must not equal denominatorMetricRef"
                        ),
                        "denominatorMetricRef": string_property(
                            "semantic metric key for the base population denominator; for ratio/percentage it must not equal numeratorMetricRef"
                        ),
                        "groupByColumn": string_property("grain/group key for the calculation"),
                    },
                    required=["operation", "sourcePhrase"],
                ),
            ),
            "scopeConstraints": array_property(
                "bounded entity sets that must constrain the graph before computing anchor/support metrics",
                scope_schema,
            ),
            "filters": array_property("explicit entity filters", filter_schema),
            "semanticQuery": semantic_query_schema,
            "timeWindowDays": integer_property("requested time window in days", 1),
        },
        required=[
            "analysisGrain",
            "analysisIntent",
            "requiresExplanation",
            "requiredEvidenceIntents",
            "anchorMetric",
            "supportMetrics",
            "metricCandidateDecisions",
            "calculationIntents",
            "scopeConstraints",
            "semanticQuery",
            "timeWindowDays",
        ],
    )
    request_schema = object_schema(
        {
            "type": string_property("knowledge request type", ["TABLE", "FIELD", "METRIC", "RELATIONSHIP", "BUSINESS_RULE", "FRESHNESS", "REALTIME_FALLBACK"]),
            "query": string_property("knowledge query"),
            "reason": string_property("why more knowledge is needed"),
        },
        required=["type", "query", "reason"],
    )
    return AgentToolDefinition(
        name="emit_question_understanding",
        description="Return semantic-layer-bounded understanding for a BI question. Do not return SQL or QueryGraph.",
        parameters=object_schema(
            {
                "status": string_property("understanding status", status_values),
                "questionUnderstanding": understanding_schema,
                "knowledgeRequests": array_property("knowledge requests when status is NEED_MORE_KNOWLEDGE", request_schema),
                "reason": string_property("brief reasoning summary"),
            },
            required=["status", "questionUnderstanding", "reason"],
        ),
    )


def source_condition_coverage_tool() -> AgentToolDefinition:
    """Independent, semantic-free extraction contract for user predicates."""

    condition_node = object_schema(
        {
            "nodeId": string_property("unique id within this independent condition AST"),
            "nodeType": string_property("atomic predicate or boolean group", ["predicate", "group"]),
            "sourcePhrase": string_property(
                "exact verbatim question span for a predicate; empty for a synthetic boolean group"
            ),
            "startOffset": integer_property("zero-based inclusive sourcePhrase offset", 0),
            "endOffset": integer_property("zero-based exclusive sourcePhrase offset", 0),
            "operator": string_property(
                "operator expressed by the user; empty for a group",
                [
                    "",
                    "eq",
                    "neq",
                    "in",
                    "not_in",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                    "between",
                    "is_null",
                    "is_not_null",
                    "contains",
                    "not_contains",
                    "starts_with",
                    "ends_with",
                ],
            ),
            "rawValues": array_property(
                "unmodified value phrases copied from the same question span",
                string_property("one raw user value"),
            ),
            "logicalOperator": string_property(
                "boolean operator for a group; empty for a predicate",
                ["", "and", "or", "not"],
            ),
            "childNodeIds": array_property(
                "child node ids for a group; empty for a predicate",
                string_property("child node id"),
            ),
        },
        required=[
            "nodeId",
            "nodeType",
            "sourcePhrase",
            "startOffset",
            "endOffset",
            "operator",
            "rawValues",
            "logicalOperator",
            "childNodeIds",
        ],
    )
    excluded_constraint = object_schema(
        {
            "constraintType": string_property(
                "constraint handled by another governed contract",
                ["time_window", "result_limit", "result_order", "projection"],
            ),
            "sourcePhrase": string_property("exact verbatim question span"),
            "startOffset": integer_property("zero-based inclusive sourcePhrase offset", 0),
            "endOffset": integer_property("zero-based exclusive sourcePhrase offset", 0),
            "reason": string_property("why this is not a row predicate"),
        },
        required=["constraintType", "sourcePhrase", "startOffset", "endOffset", "reason"],
    )
    return AgentToolDefinition(
        name="emit_source_condition_coverage",
        description=(
            "Independently enumerate every explicit row/filter condition in the original BI question. "
            "Do not use or infer semantic refs, tables, columns, SQL, or a prior Planner AST. "
            "conditionNodes contains only row/population predicates. Exclude relative/absolute time windows, Top-N/limit, "
            "ordering and requested output fields from that boolean AST and list them in excludedConstraints so their "
            "separate time/result contracts can verify them. Return VERIFIED only when this classification is complete."
        ),
        parameters=object_schema(
            {
                "status": string_property(
                    "independent extraction result",
                    ["VERIFIED", "AMBIGUOUS", "INVALID"],
                ),
                "conditionNodes": array_property(
                    "flat non-recursive predicate/group graph; empty when the question has no explicit filter",
                    condition_node,
                ),
                "rootConditionNodeId": string_property(
                    "root node id; empty only when conditionNodes is empty"
                ),
                "excludedConstraints": array_property(
                    "time and result-shape constraints intentionally governed outside semanticQuery filters",
                    excluded_constraint,
                ),
                "clarificationQuestion": string_property(
                    "question for the user when condition boundaries or boolean meaning are ambiguous"
                ),
                "reason": string_property("brief coverage decision"),
            },
            required=[
                "status",
                "conditionNodes",
                "rootConditionNodeId",
                "excludedConstraints",
                "clarificationQuestion",
                "reason",
            ],
        ),
    )


def sql_draft_tool() -> AgentToolDefinition:
    return AgentToolDefinition(
        name="draft_sql",
        description="Return safe SQL for the current single QueryGraph node, strictly bound to nodePlanContract.",
        parameters=object_schema(
            {
                "sql": string_property("single SELECT/WITH SQL statement scoped to nodePlanContract.preferredTable and allowedColumns"),
                "reason": string_property("brief reason for selected filters, fields and grouping"),
            },
            required=["sql"],
        ),
    )


def sql_repair_tool() -> AgentToolDefinition:
    return AgentToolDefinition(
        name="repair_sql",
        description="Return repaired SQL without changing QueryGraph semantics or nodePlanContract.",
        parameters=object_schema(
            {
                "sql": string_property("repaired single SELECT/WITH SQL statement scoped to nodePlanContract"),
                "reason": string_property("brief repair rationale"),
            },
            required=["sql"],
        ),
    )


def lead_action_selection_tool(action_ids: Iterable[str]) -> AgentToolDefinition:
    actions = list(action_ids)
    return AgentToolDefinition(
        name="select_agent_action",
        description="Select exactly one next Lead Agent action from the runtime action registry.",
        parameters=object_schema(
            {
                "actionId": string_property("selected action id", actions),
                "reason": string_property("short decision reason"),
            },
            required=["actionId", "reason"],
        ),
    )


def delegate_subagent_tool(task_kinds: Iterable[str]) -> AgentToolDefinition:
    """Formal Lead Agent tool for bounded, isolated worker delegation."""
    kinds = sorted({str(item) for item in task_kinds if str(item or "").strip()})
    task = object_schema(
        {
            "taskKind": string_property("worker capability selected for this task", kinds),
            "objective": string_property("self-contained objective for the isolated Sub-Agent"),
            "inputs": {"type": "object", "description": "bounded task inputs; runtime validates and enriches these", "additionalProperties": True},
            "expectedOutputs": array_property("outputs required by the Lead Agent", string_property("output name or acceptance criterion")),
            "timeout": integer_property("hard task timeout in seconds", 1),
        },
        required=["taskKind", "objective", "inputs", "expectedOutputs", "timeout"],
    )
    return AgentToolDefinition(
        name="delegate_subagent",
        description="Delegate one or more independent, bounded tasks to isolated workers and return a uniform result contract.",
        parameters=object_schema(
            {
                "tasks": array_property("bounded Sub-Agent tasks", task),
                "parallel": boolean_property("run independent tasks concurrently"),
                "isolationMode": string_property("execution isolation", ["worker"]),
                "readArtifactPolicy": string_property("when the Lead Agent should read result artifacts", ["on_completion", "summary_first"]),
                "failureStrategy": string_property("strategy when a task fails", ["retry", "fallback", "repair", "continue_partial"]),
                "reason": string_property("why delegation is preferable to continuing in the Lead Agent context"),
            },
            required=["tasks", "parallel", "isolationMode", "readArtifactPolicy", "failureStrategy", "reason"],
        ),
    )


def semantic_file_tool_definitions() -> List[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name="semantic_ls",
            description="List semantic-layer file refs before reading large table or relationship assets.",
            parameters=object_schema(
                {
                    "path": string_property("optional semantic directory such as topics/<topic>/tables/<table>"),
                    "topic": string_property("optional topic id or display name from the runtime topic catalog"),
                    "query": string_property("optional search phrase for narrowing refs"),
                    "limit": integer_property("maximum refs to return", 1),
                    "reason": string_property("why these semantic refs are needed now"),
                },
            ),
        ),
        AgentToolDefinition(
            name="semantic_read",
            description="Read one semantic-layer file ref by refId or path. Use only after semantic_ls/grep indicates it is needed.",
            parameters=object_schema(
                {
                    "refId": string_property("semantic ref id returned by semantic_ls or semantic_grep"),
                    "path": string_property("semantic file path returned by semantic_ls or semantic_grep"),
                    "maxChars": integer_property("maximum characters to read", 1),
                    "offset": integer_property("character offset for progressive reads", 0),
                    "reason": string_property("why this file content is required"),
                },
            ),
        ),
        AgentToolDefinition(
            name="semantic_grep",
            description="Search semantic-layer files and return refs plus small snippets, without loading whole assets.",
            parameters=object_schema(
                {
                    "query": string_property("search phrase or metric/table/field term"),
                    "topic": string_property("optional topic display name"),
                    "limit": integer_property("maximum hits to return", 1),
                    "reason": string_property("why this search is needed"),
                },
                required=["query"],
            ),
        ),
        AgentToolDefinition(
            name="semantic_write",
            description="Write a proposal/offloaded artifact for semantic-layer review. Never overwrite canonical asset.json directly.",
            parameters=object_schema(
                {
                    "topic": string_property("topic display name"),
                    "table": string_property("optional table name"),
                    "fileName": string_property("proposal file name"),
                    "content": string_property("proposal or offloaded artifact content"),
                    "reason": string_property("why this write is needed"),
                },
                required=["topic", "fileName", "content"],
            ),
        ),
    ]


def semantic_file_tool_schemas() -> List[Dict[str, Any]]:
    return [tool.trace_schema() for tool in semantic_file_tool_definitions()]


def artifact_file_tool_definitions() -> List[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name="artifact_ls",
            description="List workspace artifacts generated during this agent run.",
            parameters=object_schema(
                {
                    "namespace": string_property("optional artifact namespace, e.g. planner, sql, sql_results"),
                    "limit": integer_property("maximum files to return", 1),
                    "reason": string_property("why these artifacts are needed"),
                },
            ),
        ),
        AgentToolDefinition(
            name="artifact_read",
            description="Read a workspace artifact by path with offset/limit for progressive disclosure.",
            parameters=object_schema(
                {
                    "path": string_property("artifact path or relativePath"),
                    "maxChars": integer_property("maximum characters to read", 1),
                    "offset": integer_property("character offset", 0),
                    "reason": string_property("why this artifact content is needed"),
                },
                required=["path"],
            ),
        ),
        AgentToolDefinition(
            name="artifact_grep",
            description="Search workspace artifacts and return paths plus snippets.",
            parameters=object_schema(
                {
                    "query": string_property("search phrase"),
                    "limit": integer_property("maximum hits to return", 1),
                    "reason": string_property("why this search is needed"),
                },
                required=["query"],
            ),
        ),
        AgentToolDefinition(
            name="artifact_write",
            description="Write an intermediate artifact into the current run workspace.",
            parameters=object_schema(
                {
                    "namespace": string_property("artifact namespace"),
                    "fileName": string_property("file name"),
                    "content": string_property("artifact content"),
                    "reason": string_property("why this write is needed"),
                },
                required=["namespace", "fileName", "content"],
            ),
        ),
    ]


def artifact_file_tool_schemas() -> List[Dict[str, Any]]:
    return [tool.trace_schema() for tool in artifact_file_tool_definitions()]


def node_runtime_tool_schemas(tool_registry: Mapping[str, str], selected_tools: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    selected = set(selected_tools or tool_registry.keys())
    capability_registry = tool_registry_from_descriptions(tool_registry)
    schemas: List[Dict[str, Any]] = []
    for name, description in tool_registry.items():
        if name not in selected:
            continue
        schema = AgentToolDefinition(
            name=name,
            description=description,
            parameters=object_schema(
                {
                    "taskId": string_property("QueryGraph node task id"),
                    "reason": string_property("why this tool should run now"),
                },
                required=["taskId"],
            ),
        ).trace_schema()
        schema["capability"] = capability_registry.capability(name).trace()
        schemas.append(schema)
    return schemas
