from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Dict, List

from merchant_ai.models import GraphValidationGap, QueryPlan, ToolCallRequest


def compact_openai_tool_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Keep function-calling structure while removing verbose descriptions."""

    compact = deepcopy(schema)

    def visit(value: Any, depth: int = 0) -> None:
        if isinstance(value, dict):
            if "description" in value:
                description = str(value.get("description") or "")
                value["description"] = description[:80] if depth <= 2 else ""
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth)

    visit(compact)
    return compact


def planner_prompt_stats(system_prompt: str, user_prompt: str, tool_schema: Any) -> Dict[str, Any]:
    tool_chars = len(json.dumps(tool_schema, ensure_ascii=False, sort_keys=True, default=str))
    schema_count = len(tool_schema) if isinstance(tool_schema, list) else (1 if tool_schema else 0)
    return {
        "systemPromptChars": len(system_prompt or ""),
        "userPromptChars": len(user_prompt or ""),
        "toolSchemaChars": tool_chars,
        "totalChars": len(system_prompt or "") + len(user_prompt or "") + tool_chars,
        "toolSchemaCount": schema_count,
        "schemaMode": "runtime_tool_bundle" if isinstance(tool_schema, list) else "compact_tool_schema",
    }


def compact_planner_trace(trace: List[str], gaps: List[GraphValidationGap], compact_retry: bool) -> List[str]:
    if not trace:
        return []
    if not gaps and not compact_retry:
        return []
    markers = ("gap", "error", "invalid", "critic", "repair", "planner", "validation", "timeout", "provider", "calculation")
    selected = [item for item in trace if any(marker in str(item).lower() for marker in markers)]
    return selected[-3:]


def planner_repair_feedback_for_understanding(gaps: List[GraphValidationGap], previous_understanding: Dict[str, Any]) -> Dict[str, Any]:
    calculation_gaps = [
        gap
        for gap in gaps
        if gap.code in {"CALCULATION_NUMERATOR_MISSING", "CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR", "CALCULATION_NUMERATOR_NOT_EVENT_METRIC"}
    ]
    if not calculation_gaps:
        return {}
    previous_calculations = [
        item
        for item in previous_understanding.get("calculationIntents") or previous_understanding.get("calculation_intents") or []
        if isinstance(item, dict)
    ]
    previous_ranking = previous_understanding.get("rankingObjective") or previous_understanding.get("ranking_objective") or {}
    denominator_ref = ""
    if isinstance(previous_ranking, dict):
        denominator_ref = str(
            previous_ranking.get("resolvedMetricRef")
            or previous_ranking.get("metricRef")
            or previous_ranking.get("metric_ref")
            or ""
        )
    feedback_items: List[Dict[str, Any]] = []
    for gap in calculation_gaps:
        invalid = next(
            (
                item
                for item in previous_calculations
                if str(item.get("sourcePhrase") or item.get("source_phrase") or gap.evidence) == str(gap.evidence or "")
            ),
            previous_calculations[0] if previous_calculations else {},
        )
        numerator_ref = str(invalid.get("numeratorMetricRef") or invalid.get("numerator_metric_ref") or "")
        invalid_denominator_ref = str(invalid.get("denominatorMetricRef") or invalid.get("denominator_metric_ref") or denominator_ref)
        feedback_items.append(
            {
                "code": gap.code,
                "sourcePhrase": gap.evidence,
                "reason": gap.reason,
                "invalidNumeratorMetricRef": numerator_ref,
                "invalidDenominatorMetricRef": invalid_denominator_ref,
                "instruction": (
                    "Re-understand the ratio/proportion. numeratorMetricRef must be the event/subset being counted; "
                    "denominatorMetricRef must be the base population. They must not resolve to the same canonical metric. "
                    "Do not use an already-derived rate/ratio metric as the numerator. "
                    "If semanticCatalog lacks the numerator metric, return NEED_MORE_KNOWLEDGE with a METRIC knowledge request instead of repeating the same pair."
                ),
            }
        )
    return {
        "calculation": feedback_items,
        "mustFixBeforePlanning": True,
    }


def payload_has_understanding(payload: Dict[str, Any]) -> bool:
    understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    return isinstance(understanding, dict) and bool(understanding)


def normalize_llm_tool_calls(calls: List[Dict[str, Any]], round_index: int) -> List[ToolCallRequest]:
    normalized: List[ToolCallRequest] = []
    for index, call in enumerate(calls):
        name = str(call.get("name") or "")
        if not name:
            continue
        args = call.get("args") or {}
        if isinstance(args, str):
            args = parse_json_object(args)
        normalized.append(
            ToolCallRequest(
                id=str(call.get("id") or "planner_round_%d_call_%d" % (round_index + 1, index + 1)),
                name=name,
                args=args if isinstance(args, dict) else {},
            )
        )
    return normalized


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def compact_tool_result_for_prompt(result: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    limit = max(1000, int(max_chars or 12000))
    payload = json.dumps(result or {}, ensure_ascii=False, default=str)
    if len(payload) <= limit:
        return result or {}
    compact = dict(result or {})
    if "content" in compact:
        content = str(compact.get("content") or "")
        compact["content"] = content[:limit]
        compact["truncated"] = True
        compact["nextContentOffsetChars"] = min(len(content), limit)
    elif "items" in compact:
        compact["items"] = compact.get("items", [])[:20]
        compact["truncated"] = True
    elif "hits" in compact:
        compact["hits"] = compact.get("hits", [])[:10]
        compact["truncated"] = True
    else:
        compact = {"preview": payload[:limit], "truncated": True}
    return compact


def planner_tool_results_for_prompt(results: List[Dict[str, Any]], max_items: int = 4, max_chars: int = 12000) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    budget = max(1000, int(max_chars or 12000))
    used = 0
    for item in list(results or [])[-max(1, max_items) :]:
        compact = dict(item or {})
        if "result" in compact:
            compact["result"] = compact_tool_result_for_prompt(compact.get("result") or {}, max(1000, int(budget / max(1, max_items))))
        for key in ["promptArtifact", "artifact"]:
            if key in compact and isinstance(compact[key], dict):
                compact[key] = {
                    "relativePath": compact[key].get("relativePath") or compact[key].get("path", ""),
                    "sha256": compact[key].get("sha256", ""),
                    "estimatedChars": compact[key].get("estimatedChars", 0),
                }
        raw = json.dumps(compact, ensure_ascii=False, default=str)
        if used + len(raw) > budget and selected:
            selected.append(
                {
                    "offloaded": True,
                    "reason": "planner tool results exceeded prompt budget",
                    "omittedCount": max(0, len(results or []) - len(selected)),
                }
            )
            break
        selected.append(compact)
        used += len(raw)
    return selected


def compact_previous_understanding(payload: Dict[str, Any], max_items: int = 3) -> Dict[str, Any]:
    understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    if not isinstance(understanding, dict):
        understanding = {}
    compact_understanding: Dict[str, Any] = {}
    for key, value in understanding.items():
        if isinstance(value, list):
            compact_understanding[key] = value[:max(1, max_items)]
        elif isinstance(value, dict):
            compact_understanding[key] = value
        elif key in {"analysisIntent", "analysis_intent", "requiresExplanation", "requires_explanation", "analysisGrain", "analysis_grain"}:
            compact_understanding[key] = value
    return {
        "status": str(payload.get("status") or ""),
        "reason": str(payload.get("reason") or "")[:500],
        "questionUnderstanding": compact_understanding,
    }


def artifact_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": artifact.get("path", ""),
        "relativePath": artifact.get("relativePath", ""),
        "estimatedChars": artifact.get("estimatedChars", 0),
        "sha256": artifact.get("sha256", ""),
        "truncated": artifact.get("truncated", False),
    }


def compact_planner_context(planner_context: Dict[str, Any] | None, budget_level: int = 0) -> Dict[str, Any]:
    if not isinstance(planner_context, dict):
        return {}
    result: Dict[str, Any] = {}
    diagnostic = planner_context.get("openDiagnostic") or planner_context.get("open_diagnostic") or {}
    if isinstance(diagnostic, dict) and diagnostic.get("scope"):
        result.update(
            {
                "scope": str(diagnostic.get("scope") or ""),
                "intent": str(diagnostic.get("intent") or ""),
                "goal": str(diagnostic.get("goal") or ""),
                "seedTopics": [
                    str(item)
                    for item in diagnostic.get("seedTopics") or diagnostic.get("seed_topics") or []
                    if item
                ][:8],
            }
        )
    conversation = planner_context.get("conversationContext") or planner_context.get("conversation_context") or {}
    if isinstance(conversation, dict) and conversation:
        recent_limit = 2 if budget_level >= 2 else 6
        result["conversationContext"] = {
            "trust": "untrusted_conversation_data",
            "previousQuestion": str(conversation.get("previousQuestion") or "")[: 300 if budget_level >= 2 else 600],
            "previousAnswerPreview": str(conversation.get("previousAnswerPreview") or "")[: 400 if budget_level >= 2 else 800],
            "previousSummary": "" if budget_level >= 2 else str(conversation.get("previousSummary") or "")[:1000],
            "recentMessages": [
                {
                    "role": str(item.get("role") or ""),
                    "text": str(item.get("text") or "")[:800],
                }
                for item in conversation.get("recentMessages") or []
                if isinstance(item, dict) and str(item.get("role") or "") in {"user", "assistant"}
            ][-recent_limit:],
        }
    return result


def compact_memory_constraints(planner_context: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(planner_context, dict):
        return []
    constraints = planner_context.get("memoryConstraints") or planner_context.get("memory_constraints") or []
    if not isinstance(constraints, list):
        return []
    compacted: List[Dict[str, Any]] = []
    for item in constraints[:12]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                key: item.get(key)
                for key in [
                    "id",
                    "type",
                    "enforcement",
                    "instruction",
                    "targetMetrics",
                    "topics",
                    "timeWindows",
                    "confidence",
                    "governanceInstruction",
                ]
                if item.get(key) not in (None, "", [], {})
            }
        )
    return compacted


def planner_failure_trace_reason(configured: bool, last_error: str) -> str:
    if not configured:
        return "planner.no_llm_configured"
    error = str(last_error or "")
    if error.startswith("context_over_budget:"):
        return "PLANNER_CONTEXT_OVER_BUDGET: %s" % error
    if error.startswith("timeout:"):
        return "PLANNER_LLM_TIMEOUT: %s" % error
    if error.startswith("provider_error:"):
        return "PLANNER_PROVIDER_ERROR: %s" % error
    if error.startswith("json_parse_error:"):
        return "PLANNER_JSON_PARSE_ERROR: %s" % error
    if error.startswith("empty_response:"):
        return "PLANNER_EMPTY_RESPONSE: %s" % error
    return error or "planner.no_valid_llm_understanding"


def planner_llm_terminal_error(error: str) -> bool:
    lowered = str(error or "").lower()
    return any(marker in lowered for marker in ["timeout:", "provider_error:", "empty_response:"])


def planner_failure_gap_code(plan: QueryPlan) -> str:
    trace = "\n".join(plan.agent_trace or [])
    trace_lower = trace.lower()
    if "planner_context_over_budget" in trace_lower or "context_over_budget:" in trace_lower:
        return "PLANNER_CONTEXT_OVER_BUDGET"
    if "planner.no_llm_configured" in trace:
        return "PLANNER_LLM_NOT_CONFIGURED"
    if "planner_llm_timeout" in trace_lower or "timeout:" in trace_lower:
        return "PLANNER_LLM_TIMEOUT"
    if "planner_provider_error" in trace_lower or "provider_error:" in trace_lower:
        return "PLANNER_PROVIDER_ERROR"
    if "planner_json_parse_error" in trace_lower or "json_parse_error:" in trace_lower:
        return "PLANNER_JSON_PARSE_ERROR"
    if "planner_empty_response" in trace_lower or "empty_response:" in trace_lower:
        return "PLANNER_EMPTY_RESPONSE"
    return ""


def planner_failure_reason(plan: QueryPlan, code: str) -> str:
    trace = "；".join(plan.agent_trace[-3:]) if plan.agent_trace else ""
    if code == "PLANNER_LLM_TIMEOUT":
        return "Planner LLM 调用超时，questionUnderstanding 未返回；不能伪装成业务无数据。%s" % trace
    if code == "PLANNER_CONTEXT_OVER_BUDGET":
        return "Planner questionUnderstanding 上下文超过预算，未调用 LLM；需要缩小 semantic catalog 或按需读取文件上下文。%s" % trace
    if code == "PLANNER_LLM_NOT_CONFIGURED":
        return "当前未配置可用 LLM，questionUnderstanding 未生成。"
    if code == "PLANNER_PROVIDER_ERROR":
        return "Planner LLM provider 调用失败，questionUnderstanding 未生成。%s" % trace
    if code == "PLANNER_JSON_PARSE_ERROR":
        return "Planner LLM 返回内容无法解析为 questionUnderstanding。%s" % trace
    if code == "PLANNER_EMPTY_RESPONSE":
        return "Planner LLM 返回空内容，questionUnderstanding 未生成。%s" % trace
    return trace or "Planner 未能生成 QueryGraph。"
