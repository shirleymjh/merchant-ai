from __future__ import annotations

from dataclasses import dataclass
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


def question_understanding_tool(force_catalog: bool = False) -> AgentToolDefinition:
    status_values = ["UNDERSTOOD", "INVALID"] if force_catalog else ["UNDERSTOOD", "NEED_MORE_KNOWLEDGE", "INVALID"]
    ranking_schema = object_schema(
        {
            "metricRef": string_property("candidate metric key used for sorting; empty for detail lookup"),
            "sourcePhrase": string_property("exact source phrase from the user question"),
            "ownerTable": string_property("metric owner table from semanticCatalog"),
            "objectiveType": string_property("metric objective type", ["metric_total", "ranking", "trend_anchor", "detail_anchor"]),
            "groupByColumn": string_property("grain column such as spu_id, sub_order_id, order_id or pt"),
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
        },
        required=["metricRef", "ownerTable"],
    )
    scope_schema = object_schema(
        {
            "scopeId": string_property("stable id for this bounded entity set, e.g. coupon_order_scope"),
            "sourcePhrase": string_property("exact source phrase that limits the population, e.g. orders brought by a campaign"),
            "ownerTable": string_property("semanticCatalog table that defines the bounded population"),
            "metricRef": string_property("optional candidate metric key that describes the scope source"),
            "entityGrain": string_property("entity grain produced by the scope", ["merchant", "product", "order", "day", "ticket", "refund", "coupon", "unknown"]),
            "targetDomain": string_property("optional semantic domain that the scope should constrain, e.g. order/refund/product/coupon"),
            "required": boolean_property("true when this scope must be applied before computing rankingObjective/requestedMeasures"),
        },
        required=["sourcePhrase", "ownerTable", "entityGrain"],
    )
    filter_schema = object_schema(
        {
            "field": string_property(
                "filter field from semanticCatalog/live schema",
                [
                    "order_id",
                    "sub_order_id",
                    "spu_id",
                    "refund_id",
                    "ticket_id",
                    "bill_id",
                    "coupon_id",
                    "refund_status_name",
                    "refund_amt_status_name",
                    "process_status_name",
                    "pay_status_name",
                    "sub_order_status_name",
                    "spu_status_name",
                    "ticket_status_name",
                    "repay_status_name",
                ],
            ),
            "value": string_property("entity or status value from the user question; use comma-separated values for OR/IN filters"),
        },
        required=["field", "value"],
    )
    evidence_intent_schema = object_schema(
        {
            "semanticLabel": string_property("semantic evidence label, e.g. explanation_context, risk_driver, comparison_baseline"),
            "reason": string_property("why this evidence is needed for the user's requested analysis"),
            "requiredLevel": string_property("whether the answer needs this evidence", ["required", "optional"]),
            "suggestedMetricRefs": array_property("candidate semantic metric keys that could satisfy this evidence", string_property("metric key")),
            "suggestedDomains": array_property("candidate semantic domains needed for this evidence", string_property("domain name")),
        },
        required=["semanticLabel", "reason", "requiredLevel"],
    )
    understanding_schema = object_schema(
        {
            "analysisGrain": string_property("analysis grain", ["merchant", "product", "order", "day", "ticket", "refund", "coupon", "unknown"]),
            "analysisIntent": string_property(
                "analysis intent declared by the LLM understanding stage",
                ["none", "diagnosis", "trend_check", "risk_ranking", "overview", "comparison", "anomaly_check"],
            ),
            "requiresExplanation": boolean_property("true when the user asks for diagnosis, reasons, anomaly interpretation, risk judgement or summary insight"),
            "requiredEvidenceIntents": array_property(
                "evidence intents required to support the declared analysis intent; must be non-empty when analysisIntent is not none; empty only for simple lookup/ranking questions",
                evidence_intent_schema,
            ),
            "rankingObjective": ranking_schema,
            "requestedMeasures": array_property("additional requested metrics", measure_schema),
            "calculationIntents": array_property(
                "explicit derived calculations requested by the user, such as percentage/proportion/ratio between a scoped denominator and an event numerator",
                object_schema(
                    {
                        "operation": string_property("calculation operation", ["ratio", "percentage", "difference", "comparison"]),
                        "sourcePhrase": string_property("exact phrase from the user expressing this calculation"),
                        "basePopulationPhrase": string_property(
                            "base population phrase for ratio/percentage, e.g. '使用优惠券的订单' in '使用优惠券的订单中，有退货的订单占多少'"
                        ),
                        "eventPopulationPhrase": string_property(
                            "event/subset population phrase for ratio/percentage numerator, e.g. '有退货的订单' in '使用优惠券的订单中，有退货的订单占多少'"
                        ),
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
                "bounded entity sets that must constrain the graph before ranking/measures, e.g. orders from a campaign, refunded orders, newly published products",
                scope_schema,
            ),
            "filters": array_property("explicit entity filters", filter_schema),
            "timeWindowDays": integer_property("requested time window in days", 1),
        },
        required=[
            "analysisGrain",
            "analysisIntent",
            "requiresExplanation",
            "requiredEvidenceIntents",
            "rankingObjective",
            "requestedMeasures",
            "calculationIntents",
            "scopeConstraints",
            "filters",
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


def semantic_file_tool_definitions() -> List[AgentToolDefinition]:
    return [
        AgentToolDefinition(
            name="semantic_ls",
            description="List semantic-layer file refs before reading large table or relationship assets.",
            parameters=object_schema(
                {
                    "topic": string_property("optional topic display name, e.g. 电商交易"),
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
                    "refId": string_property("semantic ref id, e.g. semantic:电商交易:dwm_trade_order_detail_di:asset"),
                    "path": string_property("semantic file path, e.g. topics/电商交易/tables/dwm_trade_order_detail_di/asset.json"),
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
    schemas: List[Dict[str, Any]] = []
    for name, description in tool_registry.items():
        if name not in selected:
            continue
        schemas.append(
            AgentToolDefinition(
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
        )
    return schemas
