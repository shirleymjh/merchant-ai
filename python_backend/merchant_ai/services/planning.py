from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import (
    AnswerMode,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    KnowledgeRef,
    KnowledgeRequest,
    KnowledgeRequestType,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    PlannerReflectionResult,
    QuestionCategory,
    QuestionIntent,
    QueryPlan,
    RecallBundle,
    TaskRole,
    ToolCallRequest,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.assets import PlanningAssetPackBuilder, SemanticCatalogService, SemanticMetricIndex
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.routing import extract_days
from merchant_ai.services.tool_runtime import ToolCallExecutor, ToolFailureRegistry, ToolRuntimePolicyRegistry
from merchant_ai.services.tools import artifact_file_tool_definitions, question_understanding_tool, semantic_file_tool_definitions


class QueryGraphPlanner:
    def __init__(
        self,
        llm: LlmClient,
        semantic_catalog: SemanticCatalogService | None = None,
        artifact_store: WorkspaceArtifactStore | None = None,
        settings: Settings | None = None,
    ):
        self.llm = llm
        self.settings = settings or get_settings()
        self.semantic_catalog = semantic_catalog
        self.artifact_store = artifact_store or WorkspaceArtifactStore(self.settings)
        self.compiler = QuestionUnderstandingCompiler()
        self.prompt_assembler = PromptAssembler()
        self.tool_failure_registry = ToolFailureRegistry()
        self.tool_runtime_policies = ToolRuntimePolicyRegistry(self.settings)
        self.tool_executor = ToolCallExecutor(
            self.tool_runtime_policies,
            self.tool_failure_registry,
            max_concurrency=max(1, self.settings.max_concurrent_sub_agents),
        )

    def with_artifact_root(self, root: str) -> None:
        self.artifact_store = self.artifact_store.with_root(root)

    def plan(
        self,
        question: str,
        history_rows: List[Dict[str, Any]],
        knowledge_context: str,
        recall_bundle: RecallBundle,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        planner_context: Dict[str, Any] | None = None,
    ) -> Tuple[QueryPlan, List[KnowledgeRequest], str]:
        if self.llm.configured:
            payload = self._llm_understand(question, asset_pack, gaps, trace, planner_context=planner_context)
            payload = self._retry_compact_understanding_if_needed(question, asset_pack, gaps, trace, payload, planner_context=planner_context)
            plan = self._plan_from_payload(question, payload, asset_pack)
            if plan.intents:
                if self._should_refine_successful_plan_with_semantic_tools(payload, plan):
                    refined_payload = self._llm_understand(
                        question,
                        asset_pack,
                        gaps,
                        trace,
                        planner_context=planner_context,
                        use_tool_loop=True,
                        prior_understanding=payload,
                    )
                    refined_payload = self._retry_compact_understanding_if_needed(
                        question,
                        asset_pack,
                        gaps,
                        trace,
                        refined_payload,
                        planner_context=planner_context,
                    )
                    refined_plan = self._plan_from_payload(question, refined_payload, asset_pack)
                    if refined_plan.intents:
                        refined_plan.agent_trace.append("planner.semantic_tool_loop=refined_understanding")
                        return refined_plan, [], refined_payload.get("reason", "")
                return plan, [], payload.get("reason", "")

            if self._should_enter_semantic_tool_loop(payload, plan):
                tool_payload = self._llm_understand(
                    question,
                    asset_pack,
                    gaps,
                    trace,
                    planner_context=planner_context,
                    use_tool_loop=True,
                )
                tool_payload = self._retry_compact_understanding_if_needed(
                    question,
                    asset_pack,
                    gaps,
                    trace,
                    tool_payload,
                    planner_context=planner_context,
                )
                tool_plan = self._plan_from_payload(question, tool_payload, asset_pack)
                if tool_plan.intents:
                    tool_plan.agent_trace.append("planner.semantic_tool_loop=on_demand")
                    return tool_plan, [], tool_payload.get("reason", "")
                if tool_payload.get("status") == "NEED_MORE_KNOWLEDGE" or tool_payload.get("knowledgeRequests"):
                    payload = tool_payload
                    plan = tool_plan

            status = payload.get("status")
            if status == "NEED_MORE_KNOWLEDGE":
                if asset_pack.known_tables():
                    forced_payload = self._llm_understand(
                        question,
                        asset_pack,
                        gaps,
                        trace,
                        force_catalog=True,
                        planner_context=planner_context,
                    )
                    forced_payload = self._retry_compact_understanding_if_needed(
                        question,
                        asset_pack,
                        gaps,
                        trace,
                        forced_payload,
                        force_catalog=True,
                        planner_context=planner_context,
                    )
                    forced_plan = self._plan_from_payload(question, forced_payload, asset_pack)
                    if forced_plan.intents:
                        forced_plan.agent_trace.append("planner.need_more_overridden_by_semantic_catalog")
                        return forced_plan, [], forced_payload.get("reason", "")
                return QueryPlan(agent_trace=["planner.status=NEED_MORE_KNOWLEDGE"]), parse_knowledge_requests(
                    payload.get("knowledgeRequests", [])
                ), payload.get("reason", "")
        trace_reason = planner_failure_trace_reason(self.llm.configured, self.llm.last_error)
        return QueryPlan(agent_trace=[trace_reason]), [], trace_reason

    def _retry_compact_understanding_if_needed(
        self,
        question: str,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        payload: Dict[str, Any],
        force_catalog: bool = False,
        planner_context: Dict[str, Any] | None = None,
        use_tool_loop: bool = False,
    ) -> Dict[str, Any]:
        if payload_has_understanding(payload) or payload.get("status") == "NEED_MORE_KNOWLEDGE":
            return payload
        last_error = self.llm.last_error or ""
        if "timeout:" in last_error:
            return payload
        if not any(marker in last_error for marker in ["timeout:", "provider_error:", "empty_response:"]):
            return payload
        retry_payload = self._llm_understand(
            question,
            asset_pack,
            gaps,
            trace,
            force_catalog=force_catalog,
            compact_retry=True,
            planner_context=planner_context,
            use_tool_loop=use_tool_loop,
        )
        if payload_has_understanding(retry_payload) or retry_payload.get("status") == "NEED_MORE_KNOWLEDGE":
            retry_payload["_plannerRetry"] = {
                "reason": last_error,
                "strategy": "compact_semantic_catalog",
            }
            return retry_payload
        if last_error and not retry_payload.get("_firstError"):
            retry_payload["_firstError"] = last_error
        return retry_payload or payload

    def _plan_from_payload(self, question: str, payload: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
        understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
        if understanding:
            expansion_trace = self._expand_asset_pack_from_understanding(asset_pack, understanding)
            plan = self.compiler.compile(question, understanding, asset_pack)
            if expansion_trace:
                plan.compiler_trace.extend(expansion_trace)
            if plan.intents:
                plan = enrich_llm_plan(question, plan, asset_pack, payload)
                append_prompt_trace(plan, payload)
                attach_planner_tool_trace(plan, payload)
                plan.agent_trace.append("planner=llm_understanding_compiled")
                return plan
        if payload.get("queryPlan"):
            plan = QueryPlan(agent_trace=["planner.query_plan_payload_ignored"])
            append_prompt_trace(plan, payload)
            attach_planner_tool_trace(plan, payload)
            return plan
        plan = QueryPlan()
        append_prompt_trace(plan, payload)
        attach_planner_tool_trace(plan, payload)
        return plan

    def _expand_asset_pack_from_understanding(self, asset_pack: PlanningAssetPack, understanding: Dict[str, Any]) -> List[str]:
        if not self.semantic_catalog:
            return []
        return PlanningAssetPackBuilder(self.semantic_catalog.topic_assets).expand_for_question_understanding(asset_pack, understanding)

    def repair(
        self,
        question: str,
        plan: QueryPlan,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        history_rows: List[Dict[str, Any]],
        knowledge_context: str,
        recall_bundle: RecallBundle,
    ) -> QueryPlan:
        if not plan.intents and self.llm.last_error.startswith("provider_error"):
            plan.agent_trace.append("planner.repair.skipped_provider_error")
            return plan
        semantic_repaired = repair_missing_domain_dependencies(question, plan, asset_pack)
        if len(semantic_repaired.intents) > len(plan.intents):
            semantic_repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=semantic_missing_domains"])
            return semantic_repaired
        if self.llm.configured and gaps:
            payload = self._llm_repair(question, plan, asset_pack, gaps)
            understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
            if understanding:
                repaired = self.compiler.compile(question, understanding, asset_pack)
                if repaired.intents:
                    repaired = enrich_llm_plan(question, repaired, asset_pack, payload)
                    repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=llm_reunderstanding"])
                    return repaired
        plan.agent_trace.append("planner.repair.unavailable")
        return plan

    def _semantic_repair_applicable(self, gaps: List[GraphValidationGap]) -> bool:
        repairable_codes = {
            "DEPENDENCY_KEY_NOT_IN_SCHEMA",
            "DEPENDENCY_KEY_NOT_PRODUCED",
            "JOIN_KEY_NOT_PRODUCED",
            "MISSING_RELATIONSHIP",
            "INVALID_EDGE",
            "MISSING_DEPENDENCY_KEY",
            "BROKEN_DEPENDENCY_ENDPOINT",
        }
        return any(gap.code in repairable_codes for gap in gaps)

    def _llm_understand(
        self,
        question: str,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        force_catalog: bool = False,
        compact_retry: bool = False,
        planner_context: Dict[str, Any] | None = None,
        use_tool_loop: bool = False,
        prior_understanding: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        prompt = self.prompt_assembler.render(
            "planner.question_understanding",
            variables={
                "force_catalog_instruction": (
                    "当前 semanticCatalog 已有可用指标和表；除非问题完全无法映射，否则不要返回 NEED_MORE_KNOWLEDGE，必须从候选指标里选择最贴近的问题理解。"
                    if force_catalog
                    else ""
                )
            },
            sections={
                "context_policy": (
                    "Planner compact retry：只使用最相关的表/指标/关系，优先返回 questionUnderstanding，禁止输出 QueryGraph/SQL。"
                    if compact_retry
                    else (
                        "Planner semantic tool loop：当 semanticCatalog 清单不足以判断字段/口径/关系时，按需调用 semantic_read/grep 或 artifact_read/grep；准备好后调用 emit_question_understanding。"
                        if use_tool_loop
                        else "Planner fast path：只使用 semanticManifest、ultra compact semanticCatalog、semanticFileContext refs、validationGaps 和最近 trace；优先直接调用 emit_question_understanding。缺关键知识时返回 NEED_MORE_KNOWLEDGE，不要猜表字段。"
                    )
                ),
            },
        )
        user_payload = self._understanding_payload(question, asset_pack, gaps, trace, force_catalog, compact_retry, planner_context)
        if prior_understanding:
            user_payload["previousUnderstanding"] = compact_previous_understanding(prior_understanding)
        tool = question_understanding_tool(force_catalog)
        payload = (
            self._llm_understand_with_semantic_tools(prompt.system_prompt, user_payload, tool, force_catalog)
            if use_tool_loop
            else {}
        )
        if not payload:
            user = json.dumps(user_payload, ensure_ascii=False)
            if hasattr(self.llm, "tool_json_chat"):
                payload = self.llm.tool_json_chat(prompt.system_prompt, user, tool.openai_schema(), {})
            else:
                payload = self.llm.json_chat(prompt.system_prompt, user, {})
        else:
            payload["_usedSemanticToolLoop"] = True
        payload["_promptTrace"] = prompt.trace()
        payload["_toolSchema"] = tool.trace_schema()
        if compact_retry:
            payload["_compactRetry"] = True
        return payload

    def _should_enter_semantic_tool_loop(self, payload: Dict[str, Any], plan: QueryPlan) -> bool:
        if not self.semantic_catalog or not hasattr(self.llm, "tool_chat") or self.settings.agent_planner_tool_rounds <= 0:
            return False
        if payload.get("_usedSemanticToolLoop"):
            return False
        last_error = str(self.llm.last_error or "")
        if last_error.startswith("timeout:") or last_error.startswith("provider_error:"):
            return False
        if payload.get("status") == "NEED_MORE_KNOWLEDGE" or payload.get("knowledgeRequests"):
            return True
        return bool(payload) and not plan.intents

    def _should_refine_successful_plan_with_semantic_tools(self, payload: Dict[str, Any], plan: QueryPlan) -> bool:
        if not self._should_enter_semantic_tool_loop(payload, QueryPlan()):
            return False
        if not plan.intents:
            return False
        understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
        if not isinstance(understanding, dict):
            return False
        analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "").lower()
        if analysis_intent in {"trend_check", "anomaly_check", "overview", "comparison"}:
            return True
        evidence_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
        if not isinstance(evidence_items, list):
            return False
        labels = {
            str((item or {}).get("semanticLabel") or (item or {}).get("semantic_label") or "").lower()
            for item in evidence_items
            if isinstance(item, dict)
        }
        return bool(labels & {"comparison_baseline", "trend_context"})

    def _understanding_payload(
        self,
        question: str,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        force_catalog: bool,
        compact_retry: bool,
        planner_context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "semanticManifest": semantic_manifest_from_asset_pack(asset_pack),
            "semanticCatalog": ultra_compact_understanding_catalog(asset_pack, question, planner_context),
            "semanticFileContext": semantic_file_context_from_asset_pack(asset_pack),
            "diagnosticContext": compact_planner_context(planner_context),
            "validationGaps": [gap.model_dump(by_alias=True) for gap in gaps],
            "trace": trace[-3:] if compact_retry else trace[-8:],
            "plannerToolResults": [],
            "requiredSchema": {
                "status": "UNDERSTOOD | INVALID" if force_catalog else "UNDERSTOOD | NEED_MORE_KNOWLEDGE | INVALID",
                "questionUnderstanding": {
                    "analysisGrain": "product|order|day|ticket|refund|coupon|unknown",
                    "analysisIntent": "none|diagnosis|trend_check|risk_ranking|overview|comparison|anomaly_check",
                    "requiresExplanation": True,
                    "requiredEvidenceIntents": [
                        {
                            "semanticLabel": "explanation_context|risk_driver|comparison_baseline|trend_context",
                            "reason": "why this evidence is needed",
                            "requiredLevel": "required|optional",
                            "suggestedMetricRefs": ["candidate metric keys"],
                            "suggestedDomains": ["trade|refund|goods|ticket|compensation|coupon|scm"],
                        }
                    ],
                    "rankingObjective": {
                        "metricRef": "candidate metric key used for sorting, empty if no ranking",
                        "sourcePhrase": "exact phrase from question",
                        "ownerTable": "metric owner table",
                        "groupByColumn": "grain column such as spu_id/order_id/pt",
                        "order": "desc|asc",
                        "limit": 10,
                    },
                    "requestedMeasures": [
                        {
                            "metricRef": "candidate metric key",
                            "sourcePhrase": "exact phrase from question",
                            "ownerTable": "metric owner table",
                        }
                    ],
                    "filters": [{"field": "order_id|sub_order_id|spu_id|refund_id|ticket_id|bill_id|coupon_id", "value": "entity id from question"}],
                    "timeWindowDays": 30,
                },
                "knowledgeRequests": "KnowledgeRequest[]",
                "reason": "string",
            },
        }

    def _llm_understand_with_semantic_tools(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        output_tool: Any,
        force_catalog: bool,
    ) -> Dict[str, Any]:
        if not self.semantic_catalog or not hasattr(self.llm, "tool_chat") or self.settings.agent_planner_tool_rounds <= 0:
            return {}
        tools = [output_tool.openai_schema()] + [
            tool.openai_schema() for tool in semantic_file_tool_definitions() + artifact_file_tool_definitions()
        ]
        planner_tool_results: List[Dict[str, Any]] = []
        planner_tool_calls: List[Dict[str, Any]] = []
        loaded_refs: List[str] = []
        final_payload: Dict[str, Any] = {}
        for round_index in range(max(1, self.settings.agent_planner_tool_rounds)):
            round_payload = dict(user_payload)
            round_payload["plannerToolResults"] = planner_tool_results[-8:]
            round_payload["plannerToolPolicy"] = {
                "round": round_index + 1,
                "maxRounds": self.settings.agent_planner_tool_rounds,
                "instruction": (
                    "Call semantic_read/grep only when semanticCatalog lacks needed detail; call emit_question_understanding when ready. "
                    "If previousUnderstanding declares comparison_baseline or trend_context, inspect semantic files for the best metric owner table before emitting."
                ),
                "forceCatalog": force_catalog,
            }
            prompt_artifact = self.artifact_store.write_json("planner", "planner_round_%d_prompt.json" % (round_index + 1), round_payload, preview_chars=0)
            result = self.llm.tool_chat(
                system_prompt,
                json.dumps(round_payload, ensure_ascii=False),
                tools,
                {"content": "", "toolCalls": []},
            )
            calls = normalize_llm_tool_calls(result.get("toolCalls") or [], round_index)
            planner_tool_calls.extend([call.model_dump(by_alias=True) for call in calls])
            emit_call = next((call for call in calls if call.name == output_tool.name), None)
            if emit_call:
                final_payload = dict(emit_call.args)
                break
            semantic_calls = [call for call in calls if call.name.startswith("semantic_") or call.name.startswith("artifact_")]
            if semantic_calls:
                results = self.tool_executor.execute(semantic_calls, self._semantic_tool_handlers())
                serialized_results = []
                for item in results:
                    payload = item.model_dump(by_alias=True)
                    payload["round"] = round_index + 1
                    if item.result.get("refId"):
                        loaded_refs.append(str(item.result.get("refId")))
                    result_artifact = self.artifact_store.write_json(
                        "planner/tool_results",
                        "%s_%s_round_%d.json" % (item.name, item.id or "call", round_index + 1),
                        payload,
                        preview_chars=0,
                    )
                    payload["artifact"] = artifact_summary(result_artifact)
                    payload["promptArtifact"] = artifact_summary(prompt_artifact)
                    payload["result"] = compact_tool_result_for_prompt(item.result, self.settings.context_file_inline_max_chars)
                    serialized_results.append(payload)
                planner_tool_results.extend(serialized_results)
                continue
            parsed = parse_json_object(str(result.get("content") or ""))
            if parsed:
                final_payload = parsed
                break
            if round_index == 0:
                return {}
        if not final_payload:
            return {}
        final_payload["_plannerToolCalls"] = planner_tool_calls
        final_payload["_plannerToolResults"] = planner_tool_results
        final_payload["_plannerLoadedRefs"] = sorted(set(loaded_refs))
        final_payload["_plannerContextFiles"] = self.artifact_store.ls("planner", limit=50)
        return final_payload

    def _semantic_tool_handlers(self) -> Dict[str, Any]:
        return {
            "semantic_ls": self._handle_semantic_ls,
            "semantic_read": self._handle_semantic_read,
            "semantic_grep": self._handle_semantic_grep,
            "semantic_write": self._handle_semantic_write,
            "artifact_ls": self._handle_artifact_ls,
            "artifact_read": self._handle_artifact_read,
            "artifact_grep": self._handle_artifact_grep,
            "artifact_write": self._handle_artifact_write,
        }

    def _handle_semantic_ls(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "items": self.semantic_catalog.ls(
                topic=str(args.get("topic") or ""),
                query=str(args.get("query") or ""),
                limit=int(args.get("limit") or 20),
            )
        }

    def _handle_semantic_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.semantic_catalog.read(
            ref_id=str(args.get("refId") or ""),
            path=str(args.get("path") or ""),
            max_chars=min(int(args.get("maxChars") or self.settings.context_file_inline_max_chars), self.settings.context_file_inline_max_chars),
            offset=int(args.get("offset") or 0),
        )

    def _handle_semantic_grep(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hits": self.semantic_catalog.grep(
                query=str(args.get("query") or ""),
                topic=str(args.get("topic") or ""),
                limit=int(args.get("limit") or 20),
            )
        }

    def _handle_semantic_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.semantic_catalog.write_proposal(
            topic=str(args.get("topic") or ""),
            table=str(args.get("table") or ""),
            file_name=str(args.get("fileName") or "proposal.md"),
            content=str(args.get("content") or ""),
        )

    def _handle_artifact_ls(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "items": self.artifact_store.ls(
                namespace=str(args.get("namespace") or ""),
                limit=int(args.get("limit") or 100),
            )
        }

    def _handle_artifact_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.artifact_store.read(
            path=str(args.get("path") or ""),
            offset=int(args.get("offset") or 0),
            max_chars=min(int(args.get("maxChars") or self.settings.context_file_inline_max_chars), self.settings.context_file_inline_max_chars),
        )

    def _handle_artifact_grep(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hits": self.artifact_store.grep(
                query=str(args.get("query") or ""),
                limit=int(args.get("limit") or 20),
            )
        }

    def _handle_artifact_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.artifact_store.write_text(
            namespace=str(args.get("namespace") or "planner"),
            name=str(args.get("fileName") or "artifact.txt"),
            content=str(args.get("content") or ""),
            preview_chars=0,
        )

    def _llm_repair(self, question: str, plan: QueryPlan, asset_pack: PlanningAssetPack, gaps: List[GraphValidationGap]) -> Dict[str, Any]:
        prompt = self.prompt_assembler.render(
            "planner.repair_understanding",
            sections={
                "repair_policy": "只修正 questionUnderstanding；anchor 错就重新选择 rankingObjective，不生成 SQL 或 QueryGraph。",
            },
        )
        user = json.dumps(
            {
                "question": question,
                "previousUnderstanding": plan.question_understanding,
                "semanticCatalog": ultra_compact_understanding_catalog(asset_pack, question),
                "semanticFileContext": semantic_file_context_from_asset_pack(asset_pack),
                "gaps": [gap.model_dump(by_alias=True) for gap in gaps],
                "requiredSchema": {
                    "status": "UNDERSTOOD | NEED_MORE_KNOWLEDGE | INVALID",
                    "questionUnderstanding": {
                        "analysisGrain": "product|order|day|ticket|refund|coupon|unknown",
                        "analysisIntent": "none|diagnosis|trend_check|risk_ranking|overview|comparison|anomaly_check",
                        "requiresExplanation": True,
                        "requiredEvidenceIntents": [
                            {
                                "semanticLabel": "explanation_context|risk_driver|comparison_baseline|trend_context",
                                "reason": "why this evidence is needed",
                                "requiredLevel": "required|optional",
                                "suggestedMetricRefs": ["candidate metric keys"],
                                "suggestedDomains": ["trade|refund|goods|ticket|compensation|coupon|scm"],
                            }
                        ],
                        "rankingObjective": {
                            "metricRef": "candidate metric key used for sorting",
                            "sourcePhrase": "exact phrase from question",
                            "ownerTable": "metric owner table",
                            "groupByColumn": "entity grain column",
                            "order": "desc|asc",
                            "limit": 10,
                        },
                        "requestedMeasures": [
                            {
                                "metricRef": "candidate metric key",
                                "sourcePhrase": "exact phrase from question",
                                "ownerTable": "metric owner table",
                            }
                        ],
                        "filters": [],
                        "timeWindowDays": 30,
                    },
                    "reason": "string",
                },
            },
            ensure_ascii=False,
        )
        tool = question_understanding_tool(False)
        if hasattr(self.llm, "tool_json_chat"):
            payload = self.llm.tool_json_chat(prompt.system_prompt, user, tool.openai_schema(), {})
        else:
            payload = self.llm.json_chat(prompt.system_prompt, user, {})
        payload["_promptTrace"] = prompt.trace()
        payload["_toolSchema"] = tool.trace_schema()
        return payload


class PlannerReflectionAgent:
    """Critic agent that reviews a QueryGraph before validation/execution."""

    def reflect(self, question: str, plan: QueryPlan, asset_pack: PlanningAssetPack) -> PlannerReflectionResult:
        issues: List[Dict[str, Any]] = []
        suggested_actions: List[str] = []
        suggested_requests: List[KnowledgeRequest] = []
        repair_hints: List[str] = []
        if not plan or not plan.intents:
            issues.append(
                {
                    "code": "MISSING_QUERY_GRAPH",
                    "severity": "error",
                    "reason": "Planner did not produce executable QueryGraph nodes",
                }
            )
            suggested_actions.extend(["retrieve_knowledge", "plan_graph"])
            suggested_requests.append(
                KnowledgeRequest(
                    type=KnowledgeRequestType.BUSINESS_RULE,
                    query=question,
                    reason="planner reflection needs semantic rules to build QueryGraph",
                )
            )
        known_tables = set(asset_pack.known_tables())
        requested_domains = requested_semantic_domains_for_plan(question, plan, asset_pack)
        covered_domains = {semantic_domain_for_table(intent.preferred_table) for intent in plan.intents if intent.preferred_table}
        for intent in plan.intents:
            metric_domain = metric_domain_for_intent(intent, asset_pack)
            if metric_domain:
                covered_domains.add(metric_domain)
        missing_domains = sorted(domain for domain in requested_domains if domain not in covered_domains)
        if missing_domains:
            issues.append(
                {
                    "code": "DOMAIN_COVERAGE_GAP",
                    "severity": "error",
                    "domains": missing_domains,
                    "reason": "QueryGraph does not cover all requested semantic domains",
                }
            )
            suggested_actions.extend(["retrieve_knowledge", "repair_graph"])
            for domain in missing_domains[:4]:
                suggested_requests.append(
                    KnowledgeRequest(
                        type=KnowledgeRequestType.RELATIONSHIP,
                        query="%s %s relationships and table assets" % (question, domain),
                        reason="missing requested semantic domain in QueryGraph: %s" % domain,
                    )
                )
        for measure_issue in unplanned_requested_measure_issues(plan):
            issues.append(measure_issue)
            suggested_actions.extend(["retrieve_knowledge", "plan_graph"])
            repair_hints.append("compile requested measure %s into QueryGraph or retrieve its semantic metric definition" % measure_issue.get("metricRef"))
            suggested_requests.append(
                KnowledgeRequest(
                    type=KnowledgeRequestType.METRIC,
                    query="%s %s metric definition owner table relationship"
                    % (measure_issue.get("metricRef") or "", measure_issue.get("ownerTable") or ""),
                    reason="questionUnderstanding requested measure was not compiled into QueryGraph",
                )
            )
        for intent in plan.intents:
            task_id = intent.plan_task_id or intent.preferred_table
            if intent.preferred_table and intent.preferred_table not in known_tables:
                issues.append(
                    {
                        "code": "UNKNOWN_PLAN_TABLE",
                        "severity": "error",
                        "taskId": task_id,
                        "table": intent.preferred_table,
                        "reason": "planned table is not present in PlanningAssetPack",
                    }
                )
                suggested_actions.append("repair_graph")
            if not intent.knowledge_refs:
                issues.append(
                    {
                        "code": "MISSING_KNOWLEDGE_REF",
                        "severity": "error",
                        "taskId": task_id,
                        "table": intent.preferred_table,
                        "reason": "node has no KnowledgeRef citation from semantic layer or recalled knowledge",
                    }
                )
                suggested_actions.extend(["retrieve_knowledge", "repair_graph"])
                repair_hints.append("attach table/field/metric/relationship KnowledgeRef to %s" % task_id)
            metric_issue = metric_resolution_issue(intent)
            if metric_issue:
                issues.append(metric_issue)
                suggested_actions.extend(["retrieve_knowledge", "plan_graph"])
                repair_hints.append("resolve metricRef/sourceColumns for %s against semantic layer before SQL planning" % task_id)
                suggested_requests.append(
                    KnowledgeRequest(
                        type=KnowledgeRequestType.METRIC,
                        query="%s %s metric definition aliases formula source columns"
                        % (intent.metric_resolution.get("requestedMetricRef") or intent.metric_name or intent.metric_column, intent.preferred_table),
                        reason="planner reflection found metric resolution gap on %s" % task_id,
                    )
                )
            if intent.task_role == TaskRole.DEPENDENT and not intent.depends_on_task_ids:
                issues.append(
                    {
                        "code": "DEPENDENT_WITHOUT_UPSTREAM",
                        "severity": "error",
                        "taskId": task_id,
                        "reason": "dependent node must declare upstream task ids",
                    }
                )
                suggested_actions.append("repair_graph")
            if intent.preferred_table and "pt" in asset_pack.known_columns(intent.preferred_table) and int(intent.days or 0) <= 2:
                issues.append(
                    {
                        "code": "FRESHNESS_RISK",
                        "severity": "warning",
                        "taskId": task_id,
                        "table": intent.preferred_table,
                        "reason": "recent time window may require freshness check or realtime fallback",
                    }
                )
        if plan.dependencies:
            task_ids = {intent.plan_task_id for intent in plan.intents}
            for dep in plan.dependencies:
                if dep.anchor_task_id not in task_ids or dep.dependent_task_id not in task_ids:
                    issues.append(
                        {
                            "code": "BROKEN_DEPENDENCY_ENDPOINT",
                            "severity": "error",
                            "taskId": dep.dependent_task_id,
                            "reason": "dependency references a missing QueryGraph node",
                        }
                    )
                    suggested_actions.append("repair_graph")
                if not (dep.join_key or dep.anchor_column or dep.dependent_column):
                    issues.append(
                        {
                            "code": "MISSING_DEPENDENCY_KEY",
                            "severity": "error",
                            "taskId": dep.dependent_task_id,
                            "reason": "dependency has no entity transfer key",
                        }
                    )
                    suggested_actions.append("repair_graph")
        if not plan.evidence_contracts and not plan.final_required_evidence:
            issues.append(
                {
                    "code": "MISSING_EVIDENCE_CONTRACT",
                    "severity": "error",
                    "reason": "plan has no structured or final evidence contract",
                }
            )
            suggested_actions.append("repair_graph")
            repair_hints.append("generate evidenceContracts from planned nodes before execution")
        contract_issue = analysis_contract_issue(plan)
        if contract_issue:
            issues.append(contract_issue)
            if contract_issue["code"] == "MISSING_ANALYSIS_EVIDENCE_CONTRACT":
                suggested_actions.append("plan_graph")
                repair_hints.append("rerun LLM question understanding and require requiredEvidenceIntents")
            else:
                suggested_actions.extend(["retrieve_knowledge", "plan_graph"])
                requested_evidence = analysis_required_evidence_intents(plan)
                query = " ".join(
                    dedupe_strings(
                        [
                            str(item.get("semanticLabel") or item.get("semantic_label") or "")
                            for item in requested_evidence
                            if isinstance(item, dict)
                        ]
                        + [
                            str(item.get("reason") or "")
                            for item in requested_evidence
                            if isinstance(item, dict)
                        ]
                    )
                ).strip()
                suggested_requests.append(
                    KnowledgeRequest(
                        type=KnowledgeRequestType.BUSINESS_RULE,
                        query=query or "analysis evidence requirements",
                        reason="questionUnderstanding.requiredEvidenceIntents are not covered by current QueryGraph",
                    )
                )
        anchor_mismatch = anchor_mismatch_issue(plan)
        if anchor_mismatch:
            issues.insert(0, anchor_mismatch)
            suggested_actions.insert(0, "plan_graph")
            repair_hints.insert(0, "rerun LLM question understanding with anchor mismatch feedback")
        blocking = [issue for issue in issues if str(issue.get("severity")) == "error"]
        repair_reason = reflection_repair_reason(issues)
        return PlannerReflectionResult(
            passed=not blocking,
            issues=issues,
            suggested_actions=dedupe_strings(suggested_actions),
            suggested_knowledge_requests=suggested_requests[:6],
            repair_hints=dedupe_strings(repair_hints),
            repair_reason=repair_reason,
        )


def append_prompt_trace(plan: QueryPlan, payload: Dict[str, Any]) -> None:
    trace = payload.get("_promptTrace") if isinstance(payload, dict) else None
    if not isinstance(trace, dict):
        return
    marker = "prompt=%s@%s" % (trace.get("promptId") or "", trace.get("version") or "")
    if marker not in plan.agent_trace:
        plan.agent_trace.append(marker)
    sections = ",".join(str(item) for item in trace.get("sections") or [] if item)
    if sections:
        section_marker = "prompt.sections=%s:%s" % (trace.get("promptId") or "", sections)
        if section_marker not in plan.agent_trace:
            plan.agent_trace.append(section_marker)
    schema = payload.get("_toolSchema") if isinstance(payload, dict) else None
    if isinstance(schema, dict) and schema.get("name"):
        tool_marker = "tool_schema=%s" % schema.get("name")
        if tool_marker not in plan.agent_trace:
            plan.agent_trace.append(tool_marker)
    if payload.get("_compactRetry"):
        reason = str((payload.get("_plannerRetry") or {}).get("reason") or payload.get("_firstError") or "")
        marker = "planner.retry=compact_semantic_catalog"
        if reason:
            marker = "%s:%s" % (marker, reason[:120])
        if marker not in plan.agent_trace:
            plan.agent_trace.append(marker)
    if payload.get("_usedSemanticToolLoop"):
        plan.agent_trace.append("planner.semantic_tool_loop=enabled")
    if payload.get("_plannerLoadedRefs"):
        plan.agent_trace.append("planner.loaded_refs=%s" % ",".join(str(item) for item in payload.get("_plannerLoadedRefs") or []))


def attach_planner_tool_trace(plan: QueryPlan, payload: Dict[str, Any]) -> None:
    plan.planner_tool_calls = list(payload.get("_plannerToolCalls") or [])
    plan.planner_tool_results = list(payload.get("_plannerToolResults") or [])
    plan.planner_loaded_refs = [str(item) for item in payload.get("_plannerLoadedRefs") or []]
    plan.planner_context_files = list(payload.get("_plannerContextFiles") or [])


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


def compact_previous_understanding(payload: Dict[str, Any]) -> Dict[str, Any]:
    understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    if not isinstance(understanding, dict):
        understanding = {}
    return {
        "status": str(payload.get("status") or ""),
        "reason": str(payload.get("reason") or "")[:500],
        "questionUnderstanding": understanding,
    }


def artifact_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": artifact.get("path", ""),
        "relativePath": artifact.get("relativePath", ""),
        "estimatedChars": artifact.get("estimatedChars", 0),
        "sha256": artifact.get("sha256", ""),
        "truncated": artifact.get("truncated", False),
    }


def compact_planner_context(planner_context: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(planner_context, dict):
        return {}
    diagnostic = planner_context.get("openDiagnostic") or planner_context.get("open_diagnostic") or {}
    if not isinstance(diagnostic, dict) or not diagnostic.get("scope"):
        return {}
    return {
        "scope": str(diagnostic.get("scope") or ""),
        "intent": str(diagnostic.get("intent") or ""),
        "goal": str(diagnostic.get("goal") or ""),
        "seedTopics": [str(item) for item in diagnostic.get("seedTopics") or diagnostic.get("seed_topics") or [] if item][:8],
    }


def planner_failure_trace_reason(configured: bool, last_error: str) -> str:
    if not configured:
        return "planner.no_llm_configured"
    error = str(last_error or "")
    if error.startswith("timeout:"):
        return "PLANNER_LLM_TIMEOUT: %s" % error
    if error.startswith("provider_error:"):
        return "PLANNER_PROVIDER_ERROR: %s" % error
    if error.startswith("json_parse_error:"):
        return "PLANNER_JSON_PARSE_ERROR: %s" % error
    if error.startswith("empty_response:"):
        return "PLANNER_EMPTY_RESPONSE: %s" % error
    return error or "planner.no_valid_llm_understanding"


def planner_failure_gap_code(plan: QueryPlan) -> str:
    trace = "\n".join(plan.agent_trace or [])
    trace_lower = trace.lower()
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
    if code == "PLANNER_LLM_NOT_CONFIGURED":
        return "当前未配置可用 LLM，questionUnderstanding 未生成。"
    if code == "PLANNER_PROVIDER_ERROR":
        return "Planner LLM provider 调用失败，questionUnderstanding 未生成。%s" % trace
    if code == "PLANNER_JSON_PARSE_ERROR":
        return "Planner LLM 返回内容无法解析为 questionUnderstanding。%s" % trace
    if code == "PLANNER_EMPTY_RESPONSE":
        return "Planner LLM 返回空内容，questionUnderstanding 未生成。%s" % trace
    return trace or "Planner 未能生成 QueryGraph。"


class SemanticLayerIndex:
    def __init__(self, question: str, recall_bundle: RecallBundle, asset_pack: PlanningAssetPack):
        self.question = question or ""
        self.text = normalize_text(question)
        self.terms = extract_question_terms(question)
        self.recall_bundle = recall_bundle
        self.asset_pack = asset_pack
        self.known_tables = asset_pack.known_tables()
        self.tables = {item.table or item.key: item for item in asset_pack.tables if item.table or item.key}
        self.fields_by_table = group_entries_by_table(asset_pack.fields)
        self.metrics_by_table = group_entries_by_table(asset_pack.metrics)
        self.terms_by_table = group_entries_by_table(asset_pack.terms)
        self.relationships_by_table: Dict[str, List[Any]] = {}
        for rel in asset_pack.relationships:
            self.relationships_by_table.setdefault(rel.left_table, []).append(rel)
            self.relationships_by_table.setdefault(rel.right_table, []).append(rel)

    def relationship_path(self, start_table: str, target_table: str) -> List[Any]:
        if start_table == target_table:
            return []
        queue: List[Tuple[str, List[Any]]] = [(start_table, [])]
        seen = {start_table}
        while queue:
            table, path = queue.pop(0)
            if len(path) >= 3:
                continue
            for rel in self.relationships_by_table.get(table, []):
                next_table = rel.right_table if rel.left_table == table else rel.left_table
                if not next_table or next_table in seen:
                    continue
                if not self._relationship_columns_available(rel):
                    continue
                next_path = path + [rel]
                if next_table == target_table:
                    return next_path
                seen.add(next_table)
                queue.append((next_table, next_path))
        return []

    def neighbor_tables(self, table: str) -> List[str]:
        neighbors: List[str] = []
        for rel in self.relationships_by_table.get(table, []):
            if not self._relationship_columns_available(rel):
                continue
            other = rel.right_table if rel.left_table == table else rel.left_table
            if other and other not in neighbors:
                neighbors.append(other)
        return neighbors

    def _relationship_columns_available(self, rel: Any) -> bool:
        left_columns = set(self.asset_pack.known_columns(rel.left_table))
        right_columns = set(self.asset_pack.known_columns(rel.right_table))
        if not left_columns or not right_columns:
            return False
        for key in rel.join_keys:
            left = str(key.get("leftColumn") or "")
            right = str(key.get("rightColumn") or "")
            if left and left not in left_columns:
                return False
            if right and right not in right_columns:
                return False
        return bool(rel.join_keys)

    def knowledge_refs_for_table(self, table: str, columns: List[str], reason: str) -> List[KnowledgeRef]:
        refs: List[KnowledgeRef] = []
        table_entry = self.tables.get(table)
        if table_entry:
            refs.append(
                KnowledgeRef(
                    ref_id=table_entry.source_ref_id,
                    ref_type="TABLE",
                    table=table,
                    title=table_entry.title or table,
                    reason=reason,
                    score=1.0,
                )
            )
        field_index = {field.key: field for field in self.fields_by_table.get(table, [])}
        for column in columns:
            field = field_index.get(column)
            if field:
                refs.append(
                    KnowledgeRef(
                        ref_id=field.source_ref_id,
                        ref_type="FIELD",
                        table=table,
                        column=column,
                        title=field.title or column,
                        reason="node selected column",
                        score=1.0,
                    )
                )
        for metric in rank_asset_entries(self.metrics_by_table.get(table, []), self.question)[:3]:
            refs.append(
                KnowledgeRef(
                    ref_id=metric.source_ref_id,
                    ref_type="METRIC",
                    table=table,
                    column=",".join(metric.columns),
                    title=metric.title or metric.key,
                    reason="metric recalled for node",
                    score=1.0,
                )
            )
        for rel in self.relationships_by_table.get(table, [])[:4]:
            refs.append(
                KnowledgeRef(
                    ref_id=rel.source_ref_id,
                    ref_type="RELATIONSHIP",
                    table=table,
                    relationship_id=rel.relationship_id,
                    title=rel.relationship_id,
                    reason="relationship candidate for dependency graph",
                    score=1.0,
                )
            )
        return dedupe_knowledge_refs(refs)[:16]

    def ref_summary(self, plan: QueryPlan) -> str:
        refs: List[str] = []
        for intent in plan.intents:
            for ref_id in intent.knowledge_ref_ids[:4]:
                if ref_id and ref_id not in refs:
                    refs.append(ref_id)
        return ",".join(refs[:10])

class EvidenceContractBuilder:
    """Build structured evidence contracts without choosing graph topology."""

    def contracts_from_intents(self, intents: List[QuestionIntent]) -> List[Dict[str, Any]]:
        contracts: List[Dict[str, Any]] = []
        for intent in intents:
            contract = {
                "taskId": intent.plan_task_id,
                "table": intent.preferred_table,
                "semanticLabel": self._semantic_label(intent),
                "requiredLevel": "required",
            }
            if intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
                columns: List[str] = []
                for column in [intent.group_by_column, intent.filter_column]:
                    if column and column not in columns:
                        columns.append(column)
                metric = self._metric_contract_column(intent)
                if metric and metric not in columns:
                    columns.append(metric)
                if not metric:
                    count_alias = self._count_alias_for_table(intent.preferred_table)
                    if count_alias not in columns:
                        columns.append(count_alias)
                any_of = self._contract_any_of_groups(intent)
                if columns:
                    contract["columns"] = columns[:8]
                if any_of:
                    contract["columnsAnyOf"] = any_of[:4]
                aliases = self._semantic_aliases_for_contract(intent)
                if aliases:
                    contract["semanticAliases"] = aliases
                if intent.metric_resolution:
                    contract["metricResolution"] = intent.metric_resolution
            else:
                columns = []
                for column in intent.output_keys + intent.required_evidence + [intent.metric_column, intent.group_by_column, intent.filter_column]:
                    if column and column not in columns:
                        columns.append(column)
                contract["columns"] = columns[:16]
                aliases = self._semantic_aliases_for_contract(intent)
                if aliases:
                    contract["semanticAliases"] = aliases
                if intent.metric_resolution:
                    contract["metricResolution"] = intent.metric_resolution
            contracts.append(contract)
        return contracts

    def _contract_any_of_groups(self, intent: QuestionIntent) -> List[List[str]]:
        fields = set(intent.output_keys + [intent.group_by_column, intent.filter_column])
        groups: List[List[str]] = []
        if fields & {"spu_id", "spu_name"}:
            groups.append(["spu_id", "spu_name"])
        if fields & {"sub_order_id", "order_id"}:
            groups.append(["sub_order_id", "order_id"])
        if fields & {"ticket_id"} or "ticket" in intent.preferred_table:
            groups.append(["ticket_id", "sub_order_id", "order_id"])
        if fields & {"bill_id"} or "repay" in intent.preferred_table:
            groups.append(["bill_id", "ticket_id", "sub_order_id", "order_id"])
        if fields & {"coupon_id", "discount_rel_id"} or "coupon" in intent.preferred_table:
            groups.append(["coupon_id", "discount_rel_id"])
        if fields & {"pt"} and not groups:
            groups.append(["pt"])
        deduped: List[List[str]] = []
        for group in groups:
            compact = [column for column in group if column]
            if compact and compact not in deduped:
                deduped.append(compact)
        return deduped

    def _metric_contract_column(self, intent: QuestionIntent) -> str:
        if intent.metric_column == "pay_amt" and "refund" in intent.preferred_table:
            return "refund_related_pay_amt"
        if intent.metric_name == "pay_amt" and "refund" in intent.preferred_table:
            return "refund_related_pay_amt"
        if intent.metric_name:
            return intent.metric_name
        if intent.metric_column == "pay_amt":
            return "order_pay_amt"
        if intent.metric_column == "repay_amt":
            return "repay_amt"
        if intent.metric_column:
            return "sum_%s" % intent.metric_column
        return ""

    def _semantic_aliases_for_contract(self, intent: QuestionIntent) -> Dict[str, List[str]]:
        aliases: Dict[str, List[str]] = {
            "refund_related_pay_amt": ["refund_related_pay_amt", "refund_related_pay_amt_raw", "pay_amt", "sum_pay_amt"],
            "order_pay_amt": ["order_pay_amt", "pay_amt", "sum_pay_amt"],
            "repay_amt": ["repay_amt", "sum_repay_amt"],
            "order_cnt": ["order_cnt", "cnt", "count", "sub_order_cnt"],
            "refund_cnt": ["refund_cnt", "cnt", "count", "refund_bill_cnt"],
            "ticket_cnt": ["ticket_cnt", "cnt", "count", "ticket_bill_cnt"],
            "repay_cnt": ["repay_cnt", "cnt", "count", "repay_bill_cnt"],
            "coupon_cnt": ["coupon_cnt", "cnt", "count"],
            "scm_cnt": ["scm_cnt", "cnt", "count"],
            "goods_cnt": ["goods_cnt", "cnt", "count"],
        }
        wanted = {self._metric_contract_column(intent), self._count_alias_for_table(intent.preferred_table), intent.metric_name}
        return {key: value for key, value in aliases.items() if key in wanted}

    def _count_alias_for_table(self, table: str) -> str:
        if "refund" in table:
            return "refund_cnt"
        if "ticket" in table:
            return "ticket_cnt"
        if "repay" in table:
            return "repay_cnt"
        if "coupon" in table:
            return "coupon_cnt"
        if "scm" in table:
            return "scm_cnt"
        if "goods" in table:
            return "goods_cnt"
        return "order_cnt"

    def _semantic_label(self, intent: QuestionIntent) -> str:
        category = str(intent.category)
        if intent.metric_name:
            return intent.metric_name
        if category == QuestionCategory.GOODS.value:
            return "goods_publish_or_audit_evidence"
        if category == QuestionCategory.REFUND.value:
            return "refund_evidence"
        if category == QuestionCategory.COMPENSATION.value:
            return "repay_evidence"
        if category == QuestionCategory.CS_TICKET.value:
            return "ticket_evidence"
        return intent.preferred_table

    def final_evidence_labels(self, intents: List[QuestionIntent]) -> List[str]:
        labels: List[str] = []
        for intent in intents:
            label = self._semantic_label(intent)
            if label not in labels:
                labels.append(label)
        return labels


class QueryGraphValidator:
    def validate(self, question: str, plan: QueryPlan, asset_pack: PlanningAssetPack) -> GraphValidationResult:
        gaps: List[GraphValidationGap] = []
        planner_failure_code = planner_failure_gap_code(plan)
        planner_failed = bool(planner_failure_code)
        if not plan.intents:
            if planner_failure_code:
                gaps.append(
                    GraphValidationGap(
                        code=planner_failure_code,
                        reason=planner_failure_reason(plan, planner_failure_code),
                    )
                )
            else:
                gaps.append(GraphValidationGap(code="MISSING_QUERY_GRAPH", reason="QueryGraph 没有可执行节点"))
        table_names = set(asset_pack.known_tables())
        planned_metric_names = {intent.metric_name for intent in plan.intents if intent.metric_name}
        for intent in plan.intents:
            if intent.intent_type != IntentType.VALID or intent.answer_mode == AnswerMode.RULE:
                continue
            if not intent.preferred_table:
                gaps.append(GraphValidationGap(code="MISSING_TABLE", task_id=intent.plan_task_id, reason="缺少执行表"))
                continue
            if table_names and intent.preferred_table not in table_names:
                gaps.append(
                    GraphValidationGap(
                        code="MISSING_TABLE",
                        evidence=intent.preferred_table,
                        task_id=intent.plan_task_id,
                        reason="表不在 PlanningAssetPack 中",
                    )
                )
            columns = set(asset_pack.known_columns(intent.preferred_table))
            for col in [intent.metric_column, intent.group_by_column, intent.filter_column]:
                if col and columns and col not in columns and not is_formula(col):
                    gaps.append(
                        GraphValidationGap(
                            code="MISSING_FIELD",
                            evidence=col,
                            task_id=intent.plan_task_id,
                            reason="字段不在 PlanningAssetPack schema 中",
                        )
                    )
            for dependency in missing_metric_dependencies(intent, asset_pack, planned_metric_names):
                gaps.append(
                    GraphValidationGap(
                        code="MISSING_METRIC_DEPENDENCY",
                        evidence=dependency,
                        task_id=intent.plan_task_id,
                        reason="派生指标依赖的指标/字段未在当前 QueryGraph 或 PlanningAssetPack 中覆盖",
                    )
                )
        dependency_pairs = {(dep.anchor_task_id, dep.dependent_task_id) for dep in plan.dependencies}
        intent_by_task = {intent.plan_task_id: intent for intent in plan.intents if intent.plan_task_id}
        for intent in plan.intents:
            if intent.task_role == TaskRole.DEPENDENT:
                for parent in intent.depends_on_task_ids:
                    if (parent, intent.plan_task_id) not in dependency_pairs:
                        gaps.append(
                            GraphValidationGap(
                                code="INVALID_EDGE",
                                task_id=intent.plan_task_id,
                                reason="dependent 节点缺少 QueryGraph edge",
                            )
                        )
        for dep in plan.dependencies:
            if dep.anchor_task_id == dep.dependent_task_id:
                gaps.append(
                    GraphValidationGap(
                        code="SELF_DEPENDENCY_EDGE",
                        task_id=dep.dependent_task_id,
                        evidence="%s->%s" % (dep.anchor_task_id, dep.dependent_task_id),
                        reason="QueryGraph dependency cannot point a node to itself",
                    )
                )
                continue
            if dep.anchor_task_id not in intent_by_task or dep.dependent_task_id not in intent_by_task:
                gaps.append(
                    GraphValidationGap(
                        code="INVALID_EDGE",
                        task_id=dep.dependent_task_id,
                        evidence="%s->%s" % (dep.anchor_task_id, dep.dependent_task_id),
                        reason="QueryGraph edge references a missing node",
                    )
                )
                continue
            gaps.extend(self._dependency_schema_gaps(dep, intent_by_task, asset_pack))
            gaps.extend(self._dependency_production_gaps(dep, intent_by_task))
            if not self._relationship_supports(dep, asset_pack, intent_by_task):
                gaps.append(
                    GraphValidationGap(
                        code="MISSING_RELATIONSHIP",
                        evidence=dep.join_key,
                        task_id=dep.dependent_task_id,
                        reason="join key 未命中 relationships",
                    )
                )
        cycle = dependency_cycle(plan.dependencies)
        if cycle:
            gaps.append(
                GraphValidationGap(
                    code="CYCLIC_DEPENDENCY_EDGE",
                    evidence="->".join(cycle),
                    reason="QueryGraph dependencies must form a DAG",
                )
            )
        repairable = bool(gaps) and not planner_failed and any(gap.code != "MISSING_QUERY_GRAPH" for gap in gaps)
        requests = []
        if repairable:
            requests = [
                KnowledgeRequest(
                    type=knowledge_request_type_for_gap(gap),
                    query="%s %s %s" % (question, gap.code, gap.evidence),
                    needed_for_task_id=gap.task_id,
                    reason=gap.reason,
                )
                for gap in gaps[:8]
            ]
        return GraphValidationResult(valid=not gaps, gaps=gaps, repairable=repairable, recommended_knowledge_requests=requests)

    def _dependency_schema_gaps(
        self,
        dep: PlanDependency,
        intent_by_task: Dict[str, QuestionIntent],
        asset_pack: PlanningAssetPack,
    ) -> List[GraphValidationGap]:
        gaps: List[GraphValidationGap] = []
        anchor = intent_by_task[dep.anchor_task_id]
        dependent = intent_by_task[dep.dependent_task_id]
        checks = [
            (dep.anchor_column or dep.join_key, anchor.preferred_table, dep.anchor_task_id, "anchor"),
            (dep.dependent_column or dep.join_key, dependent.preferred_table, dep.dependent_task_id, "dependent"),
        ]
        for raw_tokens, table, task_id, side in checks:
            columns = set(asset_pack.known_columns(table))
            if not columns:
                continue
            missing = [token for token in split_join_tokens(raw_tokens) if token and token not in columns]
            if missing:
                gaps.append(
                    GraphValidationGap(
                        code="DEPENDENCY_KEY_NOT_IN_SCHEMA",
                        task_id=task_id,
                        evidence="%s.%s" % (table, ",".join(missing)),
                        reason="%s dependency key is not present in table schema" % side,
                    )
                )
        return gaps

    def _dependency_production_gaps(
        self,
        dep: PlanDependency,
        intent_by_task: Dict[str, QuestionIntent],
    ) -> List[GraphValidationGap]:
        anchor = intent_by_task[dep.anchor_task_id]
        if anchor.answer_mode not in {AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.METRIC}:
            return []
        join_tokens = [
            token
            for token in split_join_tokens(dep.anchor_column or dep.join_key)
            if token not in {"seller_id", "merchant_id"}
        ]
        if not join_tokens:
            return []
        produced = set(anchor.output_keys)
        produced.update(column for column in [anchor.group_by_column, anchor.filter_column] if column)
        produced.update(column for column in anchor.required_evidence if column)
        missing = [token for token in join_tokens if token not in produced]
        if not missing:
            return []
        return [
            GraphValidationGap(
                code="DEPENDENCY_KEY_NOT_PRODUCED",
                task_id=dep.anchor_task_id,
                evidence=",".join(missing),
                reason="anchor aggregate node does not declare the dependency key in outputKeys/requiredEvidence",
            )
        ]

    def _relationship_supports(self, dep: PlanDependency, asset_pack: PlanningAssetPack, intent_by_task: Dict[str, QuestionIntent] | None = None) -> bool:
        if not dep.join_key:
            return True
        if dep.join_key in {"pt", "merchant_id", "seller_id"}:
            return True
        if intent_by_task and dep.anchor_task_id in intent_by_task and dep.dependent_task_id in intent_by_task:
            anchor = intent_by_task[dep.anchor_task_id]
            dependent = intent_by_task[dep.dependent_task_id]
            if anchor.preferred_table == dependent.preferred_table:
                columns = set(asset_pack.known_columns(anchor.preferred_table))
                tokens = split_join_tokens(dep.anchor_column or dep.join_key) + split_join_tokens(dep.dependent_column or dep.join_key)
                return all(token in columns for token in tokens if token not in {"seller_id", "merchant_id"})
        wanted_tokens = split_join_tokens(dep.join_key) + split_join_tokens(dep.anchor_column) + split_join_tokens(dep.dependent_column)
        for rel in asset_pack.relationships:
            relationship_tokens: List[str] = []
            for key in rel.join_keys:
                relationship_tokens.extend(str(value) for value in key.values() if value)
                if dep.anchor_column in key.values() and dep.dependent_column in key.values():
                    return True
                if dep.join_key in key.values():
                    return True
            if wanted_tokens and all(token in relationship_tokens for token in wanted_tokens if token not in {"seller_id", "merchant_id"}):
                return True
        return not asset_pack.relationships


def dependency_cycle(dependencies: List[PlanDependency]) -> List[str]:
    adjacency: Dict[str, List[str]] = {}
    for dep in dependencies:
        if dep.anchor_task_id and dep.dependent_task_id:
            adjacency.setdefault(dep.anchor_task_id, []).append(dep.dependent_task_id)
    visited: set[str] = set()
    visiting: set[str] = set()
    path: List[str] = []

    def visit(node: str) -> List[str]:
        if node in visiting:
            if node in path:
                return path[path.index(node) :] + [node]
            return [node, node]
        if node in visited:
            return []
        visiting.add(node)
        path.append(node)
        for child in adjacency.get(node, []):
            found = visit(child)
            if found:
                return found
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return []

    for node in list(adjacency):
        found = visit(node)
        if found:
            return found
    return []


def compact_asset_pack_for_prompt(asset_pack: PlanningAssetPack, question: str = "") -> Dict[str, Any]:
    metrics = prompt_metric_entries(asset_pack, question, 14)
    table_entries = planner_catalog_table_entries(asset_pack, question, metrics, 3)
    prompt_tables = {item.table or item.key for item in table_entries if item.table or item.key}
    metrics = prompt_metric_entries_for_tables(asset_pack, question, prompt_tables, 12)
    relationships = [
        rel
        for rel in asset_pack.relationships
        if not prompt_tables or rel.left_table in prompt_tables and rel.right_table in prompt_tables
    ][:10]
    return {
        "designRule": (
            "Lead planner 只生成 QueryGraph，不生成 SQL。preferredTable 必须来自 tables；"
            "字段可以留空，NodeWorker 会基于节点局部 schema 生成 SQL。"
        ),
        "tables": [compact_table_entry(item, question) for item in table_entries],
        "candidateMetrics": [compact_metric_entry(item) for item in metrics],
        "relationships": [compact_relationship_entry(item) for item in relationships],
    }


def compact_understanding_catalog(asset_pack: PlanningAssetPack, question: str = "") -> Dict[str, Any]:
    metrics = prompt_metric_entries(asset_pack, question, 18)
    table_entries = planner_catalog_table_entries(asset_pack, question, metrics, 3)
    prompt_tables = {item.table or item.key for item in table_entries if item.table or item.key}
    metrics = prompt_metric_entries_for_tables(asset_pack, question, prompt_tables, 12)
    relationships = [
        rel
        for rel in asset_pack.relationships
        if not prompt_tables or rel.left_table in prompt_tables and rel.right_table in prompt_tables
    ][:12]
    return {
        "tables": [
            {
                "table": item.table or item.key,
                "domain": semantic_domain_for_table(item.table or item.key),
                "keyColumns": select_planner_columns(item.columns, question)[:12],
            }
            for item in table_entries
        ],
        "candidateMetrics": [compact_metric_entry(item) for item in metrics],
        "relationships": [compact_relationship_entry(item) for item in relationships],
    }


def ultra_compact_understanding_catalog(
    asset_pack: PlanningAssetPack,
    question: str = "",
    planner_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = prompt_metric_entries(asset_pack, question, 8)
    diagnostic_context = compact_planner_context(planner_context)
    if diagnostic_context:
        table_entries = diagnostic_catalog_table_entries(asset_pack, question, metrics, 3)
    else:
        table_entries = planner_catalog_table_entries(asset_pack, question, metrics, 3)
    prompt_tables = {item.table or item.key for item in table_entries if item.table or item.key}
    metrics = prompt_metric_entries_for_tables(asset_pack, question, prompt_tables, 8)
    relationships = [
        rel
        for rel in asset_pack.relationships
        if rel.left_table in prompt_tables and rel.right_table in prompt_tables
    ][:6]
    return {
        "tables": [
            {
                "table": item.table or item.key,
                "domain": semantic_domain_for_table(item.table or item.key),
                "keyColumns": select_planner_columns(item.columns, question)[:8],
                "sourceRefId": item.source_ref_id,
            }
            for item in table_entries
        ],
        "candidateMetrics": [
            {
                "key": item.key,
                "table": item.table,
                "title": item.title,
                "columns": item.columns[:2],
                "sourceRefId": item.source_ref_id,
            }
            for item in metrics
        ],
        "relationships": [
            {
                "relationshipId": item.relationship_id,
                "leftTable": item.left_table,
                "rightTable": item.right_table,
                "joinKeys": item.join_keys[:2],
                "sourceRefId": item.source_ref_id,
            }
            for item in relationships
        ],
        "catalogPolicy": "ultra compact semantic candidates for questionUnderstanding; use only these tables, metrics and relationships",
    }


def semantic_manifest_from_asset_pack(asset_pack: PlanningAssetPack, limit: int = 12) -> Dict[str, Any]:
    return {
        "mode": "table_manifest_first",
        "policy": "This is the first layer of semantic context. Read table/detail refs only when needed.",
        "tables": [
            {
                "table": item.table or item.key,
                "topic": item.topic,
                "title": item.title,
                "dataGrain": (item.metadata or {}).get("dataGrain", ""),
                "timeColumn": (item.metadata or {}).get("timeColumn", ""),
                "merchantFilterColumn": (item.metadata or {}).get("merchantFilterColumn", ""),
                "sourceRefId": item.source_ref_id,
                "path": "topics/%s/tables/%s/asset.json" % (item.topic or "unknown", item.table or item.key),
            }
            for item in asset_pack.tables[:limit]
        ],
        "relationshipsPathHints": sorted(
            {
                "topics/%s/relationships.json" % ref.split(":")[1]
                for ref in [relationship.source_ref_id for relationship in asset_pack.relationships]
                if isinstance(ref, str) and ref.startswith("semantic:") and len(ref.split(":")) >= 3
            }
        ),
    }


def semantic_file_context_from_asset_pack(asset_pack: PlanningAssetPack, limit: int = 12) -> Dict[str, Any]:
    refs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    pack_tables = set(asset_pack.known_tables())
    relationship_topics = relationship_topics_from_asset_pack(asset_pack)
    for item in list(asset_pack.source_refs.values()):
        metadata = item.metadata or {}
        ref_id = str(metadata.get("semanticRefId") or item.doc_id or "")
        path = str(metadata.get("semanticPath") or "")
        if not ref_id.startswith("semantic:") or not path or ref_id in seen:
            continue
        if item.table and pack_tables and item.table not in pack_tables:
            continue
        if not item.table and item.source_type == "SEMANTIC_RELATIONSHIP" and relationship_topics and item.topic not in relationship_topics:
            continue
        seen.add(ref_id)
        refs.append(
            {
                "refId": ref_id,
                "path": path,
                "kind": metadata.get("semanticKind") or item.source_type,
                "topic": item.topic,
                "table": item.table,
                "title": item.title,
                "layers": metadata.get("layers") or {},
                "estimatedChars": metadata.get("estimatedChars", len(item.content or "")),
                "offloadRecommended": bool(metadata.get("offloadRecommended")),
            }
        )
    if not refs:
        for table in asset_pack.tables[:limit]:
            topic = table.topic or "unknown"
            table_name = table.table or table.key
            if not table_name:
                continue
            refs.append(
                {
                    "refId": (
                        table.source_ref_id.rsplit(":", 1)[0] + ":asset"
                        if table.source_ref_id.endswith(":table")
                        else table.source_ref_id or "semantic:%s:%s:asset" % (topic, table_name)
                    ),
                    "path": "topics/%s/tables/%s/asset.json" % (topic, table_name),
                    "kind": "TABLE_ASSET",
                    "topic": topic,
                    "table": table_name,
                    "title": table.title or table_name,
                    "layers": {},
                    "estimatedChars": len(table.description or ""),
                    "offloadRecommended": False,
                }
            )
    return {
        "mode": "filesystem_as_context",
        "policy": "Use semanticCatalog for first-pass understanding; when evidence is missing, request semantic_ls/read/grep instead of guessing fields.",
        "tools": ["semantic_ls", "semantic_read", "semantic_grep", "semantic_write"],
        "refs": refs[:limit],
    }


def relationship_topics_from_asset_pack(asset_pack: PlanningAssetPack) -> set[str]:
    topics: set[str] = set()
    for relationship in asset_pack.relationships:
        ref_id = str(relationship.source_ref_id or "")
        if not ref_id.startswith("semantic:"):
            continue
        parts = ref_id.split(":")
        if len(parts) >= 3 and parts[2] == "relationship":
            topics.add(parts[1])
    return topics


def diagnostic_catalog_table_entries(asset_pack: PlanningAssetPack, question: str, metrics: List[Any], limit: int) -> List[Any]:
    table_by_name = {item.table or item.key: item for item in asset_pack.tables if item.table or item.key}
    selected: List[Any] = []
    selected_tables: set[str] = set()
    for table in asset_pack.known_tables():
        if semantic_domain_for_table(table) != "profile":
            continue
        entry = table_by_name.get(table)
        if entry:
            selected.append(entry)
            selected_tables.add(table)
            return selected[:limit]
    for metric in metrics:
        table = str(getattr(metric, "table", "") or "")
        if table and table not in selected_tables and table in table_by_name:
            selected.append(table_by_name[table])
            selected_tables.add(table)
        if len(selected) >= limit:
            return selected
    for entry in planner_catalog_table_entries(asset_pack, question, metrics, limit):
        table = entry.table or entry.key
        if table and table not in selected_tables:
            selected.append(entry)
            selected_tables.add(table)
        if len(selected) >= limit:
            return selected
    return selected


def prompt_table_entries(asset_pack: PlanningAssetPack, question: str) -> List[Any]:
    requested_domains = set(requested_semantic_domains(question, asset_pack))
    ranked = rank_asset_entries(asset_pack.tables, question)
    if not requested_domains:
        return ranked[:8]
    selected: List[Any] = []
    for table in ranked:
        table_name = table.table or table.key
        domain = semantic_domain_for_table(table_name)
        if domain in requested_domains or domain == "order":
            selected.append(table)
    if not selected:
        selected = ranked[:8]
    return selected[:8]


def planner_catalog_table_entries(asset_pack: PlanningAssetPack, question: str, metrics: List[Any], limit: int = 3) -> List[Any]:
    table_by_name = {item.table or item.key: item for item in asset_pack.tables if item.table or item.key}
    selected: List[Any] = []
    selected_tables: set[str] = set()

    for table_name in planner_catalog_seed_tables(asset_pack, question, limit):
        if table_name in selected_tables or table_name not in table_by_name:
            continue
        selected.append(table_by_name[table_name])
        selected_tables.add(table_name)
        if len(selected) >= limit:
            return selected

    for table in prompt_table_entries(asset_pack, question):
        table_name = table.table or table.key
        if not table_name or table_name in selected_tables:
            continue
        selected.append(table)
        selected_tables.add(table_name)
        if len(selected) >= limit:
            return selected

    for metric in metrics:
        table_name = str(getattr(metric, "table", "") or "")
        if not table_name or table_name in selected_tables or table_name not in table_by_name:
            continue
        selected.append(table_by_name[table_name])
        selected_tables.add(table_name)
        if len(selected) >= limit:
            return selected
    return selected


def planner_catalog_seed_tables(asset_pack: PlanningAssetPack, question: str, limit: int) -> List[str]:
    table_by_domain: Dict[str, List[str]] = {}
    for table in asset_pack.known_tables():
        table_by_domain.setdefault(semantic_domain_for_table(table), []).append(table)
    selected: List[str] = []
    for table in question_understanding_expanded_tables(asset_pack):
        if table in asset_pack.known_tables() and table not in selected:
            selected.append(table)
        if len(selected) >= limit:
            return selected
    seed_domains = planner_catalog_seed_domains(question, asset_pack)
    for domain in seed_domains:
        table = best_catalog_table_for_domain(domain, table_by_domain.get(domain, []), question)
        if table and table not in selected:
            selected.append(table)
        if len(selected) >= limit:
            break
    return selected


def question_understanding_expanded_tables(asset_pack: PlanningAssetPack) -> List[str]:
    traces: List[str] = []
    expansion = asset_pack.metric_compaction.get("questionUnderstandingExpansion") if asset_pack.metric_compaction else []
    if isinstance(expansion, list):
        traces.extend(str(item) for item in expansion)
    traces.extend(str(item) for item in asset_pack.relationship_closure)
    tables: List[str] = []
    for trace in traces:
        if not trace.startswith("metric_request_table:") or "->" not in trace:
            continue
        table = trace.split("->", 1)[1].split(":", 1)[0].strip()
        if table and table not in tables:
            tables.append(table)
    return tables


def planner_catalog_seed_domains(question: str, asset_pack: PlanningAssetPack) -> List[str]:
    return requested_semantic_domains(question, asset_pack)


def best_catalog_table_for_domain(domain: str, tables: List[str], question: str) -> str:
    if not tables:
        return ""
    ranked = sorted(tables, key=lambda table: catalog_table_score(table, question), reverse=True)
    return ranked[0]


def catalog_table_score(table: str, question: str) -> int:
    lower = (table or "").lower()
    score = asset_entry_score(type("CatalogTable", (), {"key": table, "title": table, "aliases": [], "description": "", "metadata": {}})(), extract_question_terms(question))
    if "detail" in lower or lower.startswith("dwm_") or lower.startswith("dwd_"):
        score += 3
    return score


def prompt_metric_entries(asset_pack: PlanningAssetPack, question: str, limit: int) -> List[Any]:
    terms = extract_question_terms(question)
    ranked = rank_asset_entries(asset_pack.metrics, question)
    matched = [item for item in ranked if asset_entry_score(item, terms) > 0]
    return (matched or ranked)[:limit]


def prompt_metric_entries_for_tables(asset_pack: PlanningAssetPack, question: str, prompt_tables: set[str], limit: int) -> List[Any]:
    scoped = [item for item in asset_pack.metrics if item.table in prompt_tables]
    if not scoped:
        return []
    terms = extract_question_terms(question)
    ranked = sorted(scoped, key=lambda item: prompt_metric_score(item, question, terms), reverse=True)
    matched = [item for item in ranked if prompt_metric_score(item, question, terms) > 0]
    return (matched or ranked)[:limit]


def prompt_metric_score(item: Any, question: str, terms: List[str]) -> int:
    score = asset_entry_score(item, terms)
    metadata = getattr(item, "metadata", {}) or {}
    confidence = metadata.get("confidence")
    if isinstance(confidence, (int, float)):
        score += int(confidence * 2)
    return score


def include_metric_tables(table_entries: List[Any], metrics: List[Any], asset_pack: PlanningAssetPack, limit: int) -> List[Any]:
    selected = list(table_entries)
    selected_tables = {item.table or item.key for item in selected if item.table or item.key}
    table_by_name = {item.table or item.key: item for item in asset_pack.tables if item.table or item.key}
    for metric in metrics:
        table = str(getattr(metric, "table", "") or "")
        if not table or table in selected_tables or table not in table_by_name:
            continue
        selected.append(table_by_name[table])
        selected_tables.add(table)
    return selected[:limit]


def semantic_catalog_sufficient(asset_pack: PlanningAssetPack, question: str) -> bool:
    terms = extract_question_terms(question)
    if not terms:
        return bool(asset_pack.known_tables())
    matched_metrics = [
        metric
        for metric in asset_pack.metrics
        if asset_entry_score(metric, terms) > 0 and metric.table in set(asset_pack.known_tables())
    ]
    if matched_metrics:
        return True
    matched_tables = [
        table
        for table in asset_pack.tables
        if asset_entry_score(table, terms) > 0 and (table.table or table.key)
    ]
    return bool(matched_tables)


class QuestionUnderstandingCompiler:
    """Compile LLM questionUnderstanding into a semantic-layer-bounded QueryGraph."""

    def compile(self, question: str, understanding: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
        return compile_query_graph_from_understanding(question, understanding, asset_pack)


def compile_query_graph_from_understanding(question: str, understanding: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
    if not isinstance(understanding, dict):
        return QueryPlan(agent_trace=["planner.understanding_compile.invalid_payload"], compiler_trace=["INVALID_UNDERSTANDING"])
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if not isinstance(ranking, dict):
        return QueryPlan(
            agent_trace=["planner.understanding_compile.missing_ranking_objective"],
            compiler_trace=["MISSING_RANKING_OBJECTIVE"],
            question_understanding=understanding,
        )
    if not str(ranking.get("metricRef") or ranking.get("metric_ref") or ""):
        detail_plan = compile_entity_detail_graph_from_understanding(question, understanding, asset_pack)
        if detail_plan.intents:
            return detail_plan
        return QueryPlan(
            agent_trace=["planner.understanding_compile.missing_ranking_objective"],
            compiler_trace=["MISSING_RANKING_OBJECTIVE"],
            question_understanding=understanding,
        )
    resolver = SemanticMetricResolver(asset_pack)
    ranking_resolution = resolver.resolve(
        question=question,
        metric_ref=str(ranking.get("metricRef") or ranking.get("metric_ref") or ""),
        owner_table=str(ranking.get("ownerTable") or ranking.get("owner_table") or ""),
        source_phrase=str(ranking.get("sourcePhrase") or ranking.get("source_phrase") or ""),
    )
    ranking_metric = ranking_resolution.metric
    if not ranking_metric:
        missing_ref = str(ranking.get("metricRef") or ranking.get("metric_ref") or "")
        return QueryPlan(
            agent_trace=["planner.understanding_compile.missing_ranking_metric"],
            compiler_trace=["UNKNOWN_METRIC_REF:%s" % missing_ref, "METRIC_RESOLUTION_LOW_CONFIDENCE:%s" % missing_ref],
            question_understanding=understanding,
        )
    grain = str(understanding.get("analysisGrain") or understanding.get("analysis_grain") or "")
    anchor = compiled_metric_intent(
        question=question,
        metric=ranking_metric,
        task_id="anchor_%s" % (semantic_domain_for_metric(ranking_metric) or semantic_domain_for_table(ranking_metric.table)),
        role=TaskRole.ANCHOR,
        mode=AnswerMode.TOPN,
        grain=grain,
        group_by=str(ranking.get("groupByColumn") or ranking.get("group_by_column") or ""),
        depends_on=[],
        limit=int(ranking.get("limit") or infer_limit(question)),
        asset_pack=asset_pack,
        metric_resolution=ranking_resolution.payload(),
    )
    if not anchor:
        return QueryPlan(
            agent_trace=["planner.understanding_compile.anchor_unavailable"],
            compiler_trace=["ANCHOR_UNAVAILABLE:%s" % ranking_metric.key],
            question_understanding=understanding,
        )
    intents: List[QuestionIntent] = [anchor]
    dependencies: List[PlanDependency] = []
    task_by_table: Dict[str, str] = {anchor.preferred_table: anchor.plan_task_id}
    expansion_task_id = ""
    index = SemanticLayerIndex(question, RecallBundle(), asset_pack)
    measures = understanding.get("requestedMeasures") or understanding.get("requested_measures") or []
    measure_items = [item for item in measures if isinstance(item, dict)]
    measure_items, formula_dependency_refs = expand_measure_items_with_metric_dependencies(ranking_metric, measure_items, asset_pack)
    measure_refs = [str(item.get("metricRef") or item.get("metric_ref") or "") for item in measure_items]
    unplanned_measure_refs: List[str] = []
    for measure in measure_items:
        metric_ref = str(measure.get("metricRef") or measure.get("metric_ref") or "")
        owner_table = str(measure.get("ownerTable") or measure.get("owner_table") or "")
        metric_resolution = resolver.resolve(
            question=question,
            metric_ref=metric_ref,
            owner_table=owner_table,
            source_phrase=str(measure.get("sourcePhrase") or measure.get("source_phrase") or ""),
        )
        metric = metric_resolution.metric
        if not metric:
            unplanned_measure_refs.append("UNRESOLVED_REQUESTED_MEASURE:%s:%s" % (metric_ref, owner_table))
            continue
        if metric.key == ranking_metric.key and metric.table == ranking_metric.table:
            continue
        parent_table = anchor.preferred_table
        parent_task = anchor.plan_task_id
        if parent_table == metric.table:
            group_by = anchor.group_by_column or grain_column_for_table(grain, set(asset_pack.known_columns(metric.table)))
            intent = compiled_metric_intent(
                question=question,
                metric=metric,
                task_id="%s_%s_lookup" % (semantic_domain_for_metric(metric), metric.key),
                role=TaskRole.DEPENDENT,
                mode=AnswerMode.GROUP_AGG,
                grain=grain,
                group_by=group_by,
                depends_on=[parent_task],
                limit=20,
                asset_pack=asset_pack,
                metric_resolution=metric_resolution.payload(),
            )
            if intent:
                intent = intent.model_copy(update={"plan_task_id": unique_task_id(intent.plan_task_id, [item.plan_task_id for item in intents])})
                intents.append(intent)
                if group_by:
                    add_dependency_if_valid(
                        dependencies,
                        PlanDependency(
                            anchor_task_id=parent_task,
                            dependent_task_id=intent.plan_task_id,
                            join_key=group_by,
                            anchor_column=group_by,
                            dependent_column=group_by,
                            relation_type="LOOKUP",
                        ),
                    )
            continue
        path = index.relationship_path(parent_table, metric.table)
        if not path and parent_table != metric.table:
            group_by = str(measure.get("groupByColumn") or measure.get("group_by_column") or "")
            independent_grain = "day" if "pt" in asset_pack.known_columns(metric.table) else grain
            intent = compiled_metric_intent(
                question=question,
                metric=metric,
                task_id="%s_%s_context" % (semantic_domain_for_metric(metric), metric.key),
                role=TaskRole.ANCHOR,
                mode=AnswerMode.GROUP_AGG,
                grain=independent_grain,
                group_by=group_by,
                depends_on=[],
                limit=20,
                asset_pack=asset_pack,
                metric_resolution=metric_resolution.payload(),
            )
            if intent:
                intent = intent.model_copy(update={"plan_task_id": unique_task_id(intent.plan_task_id, [item.plan_task_id for item in intents])})
                intents.append(intent)
                task_by_table[metric.table] = intent.plan_task_id
            else:
                unplanned_measure_refs.append("UNPLANNED_REQUESTED_MEASURE:%s:%s:no_relationship_from:%s" % (metric_ref, metric.table, parent_table))
            continue
        if path and dependency_requires_unproduced_key(path[0], parent_table, anchor):
            expansion = compiled_entity_expansion_intent(question, anchor, asset_pack)
            if expansion and not expansion_task_id:
                intents.append(expansion)
                expansion_task_id = expansion.plan_task_id
                add_dependency_if_valid(
                    dependencies,
                    PlanDependency(
                        anchor_task_id=anchor.plan_task_id,
                        dependent_task_id=expansion.plan_task_id,
                        join_key=anchor.group_by_column,
                        anchor_column=anchor.group_by_column,
                        dependent_column=anchor.group_by_column,
                        relation_type="LOOKUP",
                    ),
                )
            if expansion_task_id:
                parent_task = expansion_task_id
        for rel in path:
            next_table = rel.right_table if rel.left_table == parent_table else rel.left_table
            next_is_target = next_table == metric.table
            if next_is_target and metric_intent_missing(intents, metric.table, metric.key):
                intent = compiled_metric_intent(
                    question=question,
                    metric=metric,
                    task_id="%s_lookup" % semantic_domain_for_table(next_table),
                    role=TaskRole.DEPENDENT,
                    mode=AnswerMode.GROUP_AGG,
                    grain=grain,
                    group_by="",
                    depends_on=[parent_task],
                    limit=20,
                    asset_pack=asset_pack,
                    metric_resolution=metric_resolution.payload(),
                )
                if intent:
                    intent = intent.model_copy(update={"plan_task_id": unique_task_id(intent.plan_task_id, [item.plan_task_id for item in intents])})
                    intents.append(intent)
                    if next_table not in task_by_table:
                        task_by_table[next_table] = intent.plan_task_id
                    dependent_task = intent.plan_task_id
                else:
                    dependent_task = task_by_table.get(next_table, "")
            else:
                if next_table not in task_by_table:
                    if next_is_target:
                        intent = compiled_metric_intent(
                            question=question,
                            metric=metric,
                            task_id="%s_lookup" % semantic_domain_for_table(next_table),
                            role=TaskRole.DEPENDENT,
                            mode=AnswerMode.GROUP_AGG,
                            grain=grain,
                            group_by="",
                            depends_on=[parent_task],
                            limit=20,
                            asset_pack=asset_pack,
                            metric_resolution=metric_resolution.payload(),
                        )
                    else:
                        intent = compiled_bridge_intent(question, next_table, asset_pack, parent_task)
                    if intent:
                        intent = intent.model_copy(update={"plan_task_id": unique_task_id(intent.plan_task_id, [item.plan_task_id for item in intents])})
                        intents.append(intent)
                        task_by_table[next_table] = intent.plan_task_id
                dependent_task = task_by_table.get(next_table, "")
            dep = dependency_from_relationship(parent_task, dependent_task, parent_table, next_table, rel)
            if dep:
                add_dependency_if_valid(dependencies, dep)
            parent_table = next_table
            parent_task = dependent_task
        if metric.table not in task_by_table:
            unplanned_measure_refs.append("UNPLANNED_REQUESTED_MEASURE:%s:%s:no_target_node" % (metric_ref, metric.table))
    structured_domains = requested_semantic_domains_from_understanding(understanding, asset_pack)
    if grain == "product" or "goods" in structured_domains:
        add_product_dimension_lookup(question, intents, dependencies, task_by_table, index, asset_pack)
    compiled = sync_intent_dependencies(
        QueryPlan(
            intents=intents,
            dependencies=dependencies,
            agent_trace=["planner=llm_understanding_compiler"],
            question_understanding=understanding,
            compiler_trace=[
                "ANCHOR_METRIC:%s:%s" % (ranking_metric.key, ranking_metric.table),
                "METRIC_RESOLUTION:%s->%s:%s:%s"
                % (
                    ranking_resolution.requested_metric_ref,
                    ranking_metric.table,
                    ranking_metric.key,
                    ranking_resolution.resolution_source,
                ),
                *metric_resolution_trace_markers(ranking_resolution),
                "MEASURE_METRICS:%s" % ",".join(ref for ref in measure_refs if ref),
                "FORMULA_DEP_METRICS:%s" % ",".join(formula_dependency_refs),
                *dedupe_strings(unplanned_measure_refs),
            ],
        )
    )
    compiled.evidence_contracts = EvidenceContractBuilder().contracts_from_intents(compiled.intents)
    compiled.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(compiled.intents)
    return compiled


def compile_entity_detail_graph_from_understanding(question: str, understanding: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
    filter_column, filter_value = detail_filter_from_understanding(question, understanding)
    if not filter_column or not filter_value:
        return QueryPlan(
            agent_trace=["planner.understanding_compile.missing_detail_filter"],
            compiler_trace=["MISSING_DETAIL_FILTER"],
            question_understanding=understanding,
        )
    anchor_table = best_detail_anchor_table(filter_column, question, asset_pack)
    if not anchor_table:
        return QueryPlan(
            agent_trace=["planner.understanding_compile.detail_anchor_unavailable"],
            compiler_trace=["DETAIL_ANCHOR_UNAVAILABLE:%s" % filter_column],
            question_understanding=understanding,
        )
    columns = set(asset_pack.known_columns(anchor_table))
    index = SemanticLayerIndex(question, RecallBundle(), asset_pack)
    output_keys = generic_output_keys(QuestionIntent(filter_column=filter_column), columns)
    required = dedupe_strings(output_keys + domain_evidence_columns(semantic_domain_for_table(anchor_table), columns))
    anchor = QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_table(anchor_table),
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="anchor_%s" % semantic_domain_for_table(anchor_table),
        task_role=TaskRole.ANCHOR,
        preferred_table=anchor_table,
        filter_column=filter_column,
        filter_value=filter_value,
        days=extract_days(question, 30),
        limit=20,
        required_evidence=required[:18],
        output_keys=output_keys[:18],
        knowledge_refs=index.knowledge_refs_for_table(anchor_table, required or output_keys, reason="llm detail understanding selected anchor"),
        analysis_source="llm_question_understanding_compiler",
        analysis_note="detailFilter=%s" % filter_column,
        sql_strategy="llm_plan_bound_first",
    )
    anchor = anchor.model_copy(update={"knowledge_ref_ids": [ref.ref_id for ref in anchor.knowledge_refs if ref.ref_id]})
    plan = QueryPlan(
        intents=[anchor],
        question_understanding=understanding,
        compiler_trace=["DETAIL_ANCHOR:%s:%s=%s" % (anchor_table, filter_column, filter_value)],
        agent_trace=["planner=llm_detail_understanding_compiler"],
    )
    plan = repair_missing_domain_dependencies(question, plan, asset_pack)
    plan = attach_metric_resolutions_from_understanding(question, plan, understanding, asset_pack)
    if not plan.evidence_contracts:
        plan.evidence_contracts = EvidenceContractBuilder().contracts_from_intents(plan.intents)
    if not plan.final_required_evidence:
        plan.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(plan.intents)
    return plan


def detail_filter_from_understanding(question: str, understanding: Dict[str, Any]) -> Tuple[str, str]:
    filters = understanding.get("filters") or []
    if isinstance(filters, list):
        for item in filters:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or item.get("column") or "")
            value = str(item.get("value") or "")
            if field and value:
                return field, value
    for column in ["sub_order_id", "order_id", "refund_id", "ticket_id", "bill_id", "coupon_id", "spu_id"]:
        value = extract_entity_value(question, column)
        if value:
            return column, value
    return "", ""


def attach_metric_resolutions_from_understanding(
    question: str,
    plan: QueryPlan,
    understanding: Dict[str, Any],
    asset_pack: PlanningAssetPack,
) -> QueryPlan:
    if not plan.intents:
        return plan
    resolver = SemanticMetricResolver(asset_pack)
    measure_items = []
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if isinstance(ranking, dict) and str(ranking.get("metricRef") or ranking.get("metric_ref") or ""):
        measure_items.append(ranking)
    raw_measures = understanding.get("requestedMeasures") or understanding.get("requested_measures") or []
    if isinstance(raw_measures, list):
        measure_items.extend(item for item in raw_measures if isinstance(item, dict))
    resolutions = []
    for item in measure_items:
        resolution = resolver.resolve(
            question=question,
            metric_ref=str(item.get("metricRef") or item.get("metric_ref") or ""),
            owner_table=str(item.get("ownerTable") or item.get("owner_table") or ""),
            source_phrase=str(item.get("sourcePhrase") or item.get("source_phrase") or ""),
        )
        if resolution.metric:
            resolutions.append(resolution)
    if not resolutions:
        return plan
    updated_intents: List[QuestionIntent] = []
    changed = False
    for intent in plan.intents:
        if intent.metric_resolution:
            updated_intents.append(intent)
            continue
        resolution = metric_resolution_for_intent(intent, resolutions)
        if not resolution:
            updated_intents.append(intent)
            continue
        metric = resolution.metric
        metadata = getattr(metric, "metadata", {}) or {}
        source_columns = [str(item) for item in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns if item]
        updates: Dict[str, Any] = {"metric_resolution": resolution.payload()}
        if not intent.metric_name:
            updates["metric_name"] = metric.key
        if not intent.metric_column and source_columns:
            updates["metric_column"] = source_columns[0]
        if not intent.metric_formula:
            updates["metric_formula"] = metric_formula_for_entry(metric)
        updated_intents.append(intent.model_copy(update=updates))
        changed = True
    if not changed:
        return plan
    updated = plan.model_copy(update={"intents": updated_intents})
    updated.evidence_contracts = EvidenceContractBuilder().contracts_from_intents(updated.intents)
    updated.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(updated.intents)
    return updated


def metric_resolution_for_intent(intent: QuestionIntent, resolutions: List[SemanticMetricResolution]) -> SemanticMetricResolution | None:
    for resolution in resolutions:
        metric = resolution.metric
        if not metric or metric.table != intent.preferred_table:
            continue
        metadata = getattr(metric, "metadata", {}) or {}
        source_columns = {str(item) for item in metadata.get("sourceColumns") or metadata.get("source_columns") or metric.columns if item}
        if intent.metric_name and intent.metric_name == metric.key:
            return resolution
        if intent.metric_column and intent.metric_column in source_columns:
            return resolution
        if intent.metric_name or intent.metric_column:
            continue
        if source_columns & set(intent.required_evidence + intent.output_keys):
            return resolution
    return None


def best_detail_anchor_table(filter_column: str, question: str, asset_pack: PlanningAssetPack) -> str:
    candidates = [table for table in asset_pack.known_tables() if filter_column in asset_pack.known_columns(table)]
    if not candidates:
        return ""
    requested_domains = requested_semantic_domains(question, asset_pack)
    domain_priority = {
        "sub_order_id": ["order", "refund", "ticket", "repay"],
        "order_id": ["order", "refund", "ticket", "repay"],
        "refund_id": ["refund", "order"],
        "ticket_id": ["ticket", "repay", "order"],
        "bill_id": ["repay", "ticket", "order"],
        "coupon_id": ["coupon", "order"],
        "spu_id": ["goods", "order", "scm", "refund"],
    }.get(filter_column, requested_domains)
    for domain in requested_domains + domain_priority:
        for table in candidates:
            if semantic_domain_for_table(table) == domain:
                return table
    return candidates[0]


def add_product_dimension_lookup(
    question: str,
    intents: List[QuestionIntent],
    dependencies: List[PlanDependency],
    task_by_table: Dict[str, str],
    index: SemanticLayerIndex,
    asset_pack: PlanningAssetPack,
) -> None:
    goods_table = best_table_for_domain("goods", asset_pack)
    if not goods_table or goods_table in task_by_table:
        return
    intent_by_task = {intent.plan_task_id: intent for intent in intents if intent.plan_task_id}
    candidates = []
    existing = [
        (position, intent.preferred_table, intent.plan_task_id)
        for position, intent in enumerate(intents)
        if intent.preferred_table and intent.plan_task_id
    ]
    for position, parent_table, parent_task in existing:
        path = index.relationship_path(parent_table, goods_table)
        if not path and parent_table != goods_table:
            continue
        parent_intent = intent_by_task.get(parent_task)
        if path and parent_intent and dependency_requires_unproduced_key(path[0], parent_table, parent_intent):
            continue
        candidates.append((len(path), position, parent_table, parent_task, path))
    for _, _, parent_table, parent_task, path in sorted(candidates):
        current_table = parent_table
        current_task = parent_task
        for rel in path:
            next_table = rel.right_table if rel.left_table == current_table else rel.left_table
            next_is_goods = next_table == goods_table
            if next_table not in task_by_table:
                intent = compiled_goods_lookup_intent(question, goods_table, asset_pack, current_task) if next_is_goods else compiled_bridge_intent(question, next_table, asset_pack, current_task)
                if intent:
                    intent = intent.model_copy(update={"plan_task_id": unique_task_id(intent.plan_task_id, [item.plan_task_id for item in intents])})
                    intents.append(intent)
                    task_by_table[next_table] = intent.plan_task_id
            dependent_task = task_by_table.get(next_table, "")
            dep = dependency_from_relationship(current_task, dependent_task, current_table, next_table, rel)
            if dep:
                add_dependency_if_valid(dependencies, dep)
            current_table = next_table
            current_task = dependent_task
        if goods_table in task_by_table:
            return


def best_table_for_domain(domain: str, asset_pack: PlanningAssetPack) -> str:
    for table in asset_pack.known_tables():
        if semantic_domain_for_table(table) == domain:
            return table
    return ""


def metric_intent_missing(intents: List[QuestionIntent], table: str, metric_key: str) -> bool:
    for intent in intents:
        if intent.preferred_table != table:
            continue
        if intent.metric_name == metric_key:
            return False
        resolution = intent.metric_resolution or {}
        if str(resolution.get("metricKey") or resolution.get("metric_key") or "") == metric_key:
            return False
    return True


def metric_entry_by_ref(metric_ref: str, asset_pack: PlanningAssetPack, owner_table: str = "") -> Any:
    if not metric_ref:
        return None
    normalized = metric_ref.strip()
    metrics = [metric for metric in asset_pack.metrics if not owner_table or metric.table == owner_table]
    for metric in metrics:
        names = {metric.key, metric.title, metric.source_ref_id}
        names.update(metric.aliases)
        if normalized in {str(item) for item in names if item}:
            return metric
    return None


class SemanticMetricResolution:
    def __init__(
        self,
        requested_metric_ref: str = "",
        source_phrase: str = "",
        metric: Any = None,
        confidence: float = 0.0,
        resolution_source: str = "",
        field_warning: str = "",
        candidate_scores: List[Dict[str, Any]] | None = None,
    ):
        self.requested_metric_ref = requested_metric_ref
        self.source_phrase = source_phrase
        self.metric = metric
        self.confidence = confidence
        self.resolution_source = resolution_source
        self.field_warning = field_warning
        self.candidate_scores = candidate_scores or []

    def payload(self) -> Dict[str, Any]:
        metric = self.metric
        if not metric:
            return {
                "requestedMetricRef": self.requested_metric_ref,
                "sourcePhrase": self.source_phrase,
                "confidence": self.confidence,
                "resolutionSource": self.resolution_source or "unresolved",
                "fieldWarning": self.field_warning,
            }
        metadata = getattr(metric, "metadata", {}) or {}
        source_columns = [str(item) for item in metadata.get("sourceColumns") or metadata.get("source_columns") or getattr(metric, "columns", []) or [] if item]
        formula = str(metadata.get("formula") or metadata.get("metricFormula") or metric_formula_for_entry(metric) or "")
        return {
            "requestedMetricRef": self.requested_metric_ref,
            "sourcePhrase": self.source_phrase,
            "metricKey": metric.key,
            "ownerTable": metric.table,
            "sourceColumns": source_columns,
            "formula": formula,
            "displayName": metric.title or metadata.get("businessName") or metric.key,
            "confidence": self.confidence,
            "resolutionSource": self.resolution_source,
            "fieldWarning": self.field_warning,
            "semanticRefId": metric.source_ref_id,
            "candidateScores": self.candidate_scores[:5],
        }


class SemanticMetricResolver:
    def __init__(self, asset_pack: PlanningAssetPack):
        self.asset_pack = asset_pack

    def resolve(self, question: str, metric_ref: str, owner_table: str = "", source_phrase: str = "") -> SemanticMetricResolution:
        requested = str(metric_ref or "").strip()
        phrase = str(source_phrase or "").strip()
        if owner_table and owner_table not in self.asset_pack.known_tables():
            return SemanticMetricResolution(requested, phrase, None, 0.0, "owner_table_not_loaded", "")
        index = SemanticMetricIndex(self._candidate_metrics(""))
        candidate_scores = index.candidates(requested, owner_table, phrase)
        resolved = index.resolve(requested, owner_table, phrase)
        if not resolved:
            return SemanticMetricResolution(requested, phrase, None, 0.0, "unresolved", "")
        source = resolved.resolution_reason or "semantic_alias"
        confidence = semantic_metric_confidence(source, resolved.rank_score, resolved.phrase_score)
        warning = semantic_metric_field_warning(resolved.metric) if confidence >= 0.7 else "指标由弱匹配得到，口径需要人工确认"
        ordered_candidates = [resolved] + [
            item
            for item in candidate_scores
            if item.metric.table != resolved.metric.table or item.metric.key != resolved.metric.key
        ]
        return SemanticMetricResolution(
            requested,
            phrase,
            resolved.metric,
            confidence,
            source,
            warning,
            [item.payload() for item in ordered_candidates[:5]],
        )

    def _candidate_metrics(self, owner_table: str) -> List[PlanningAssetEntry]:
        candidates = [metric for metric in self.asset_pack.metrics if not owner_table or metric.table == owner_table]
        tables = [owner_table] if owner_table else self.asset_pack.known_tables()
        for table in tables:
            for metric in self._metadata_metrics_for_table(table):
                if not any(item.table == metric.table and item.key == metric.key for item in candidates):
                    candidates.append(metric)
        return candidates

    def _metadata_metrics_for_table(self, table: str) -> List[PlanningAssetEntry]:
        if not table:
            return []
        table_entry = next((entry for entry in self.asset_pack.tables if entry.table == table), None)
        if not table_entry:
            return []
        metrics = (table_entry.metadata or {}).get("metrics") or []
        entries: List[PlanningAssetEntry] = []
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            key = str(metric.get("metricKey") or "")
            if not key:
                continue
            entries.append(
                PlanningAssetEntry(
                    key=key,
                    table=table,
                    topic=table_entry.topic,
                    title=str(metric.get("businessName") or key),
                    columns=[str(column) for column in metric.get("sourceColumns") or []],
                    aliases=[str(alias) for alias in metric.get("aliases") or []],
                    description=json.dumps(metric, ensure_ascii=False),
                    source_ref_id="semantic:%s:%s:metric:%s" % (table_entry.topic, table, key),
                    metadata=metric,
                )
            )
        return entries


def normalize_metric_match_text(value: Any) -> str:
    return str(value or "").lower().replace(" ", "").replace("_", "")


def semantic_metric_confidence(resolution_source: str, rank_score: int, phrase_score: int) -> float:
    if resolution_source == "semantic_metric_ref":
        return 1.0
    if resolution_source == "semantic_phrase_override":
        return 0.95
    if resolution_source == "semantic_phrase_match":
        return 0.9 if phrase_score >= 18 else 0.75
    if resolution_source == "semantic_alias":
        return min(0.9, max(0.7, rank_score / 60.0))
    return min(0.69, rank_score / 60.0)


def metric_resolution_trace_markers(resolution: SemanticMetricResolution) -> List[str]:
    if resolution.resolution_source != "semantic_phrase_override" or not resolution.metric:
        return []
    markers = [
        "METRIC_SEMANTIC_MISMATCH:%s:%s->%s:%s"
        % (
            resolution.source_phrase,
            resolution.requested_metric_ref,
            resolution.metric.table,
            resolution.metric.key,
        )
    ]
    candidates = []
    for item in resolution.candidate_scores[:3]:
        candidates.append(
            "%s.%s(ref=%s,phrase=%s,total=%s)"
            % (
                item.get("ownerTable") or "",
                item.get("metricKey") or "",
                item.get("refScore") or 0,
                item.get("phraseScore") or 0,
                item.get("rankScore") or 0,
            )
        )
    if candidates:
        markers.append("METRIC_CANDIDATES:%s" % "|".join(candidates))
    return markers


def semantic_metric_field_warning(metric: Any) -> str:
    if semantic_domain_for_table(getattr(metric, "table", "")) == "refund" and getattr(metric, "key", "") == "pay_amt":
        return "退款金额按 dwm_trade_refund_detail_di.pay_amt 统计，表示退款明细关联订单的支付金额口径。"
    return ""


def expand_measure_items_with_metric_dependencies(
    ranking_metric: Any,
    measure_items: List[Dict[str, Any]],
    asset_pack: PlanningAssetPack,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    expanded = list(measure_items)
    existing: set[Tuple[str, str]] = set()
    seed_metrics = [ranking_metric]
    if ranking_metric:
        existing.add((ranking_metric.table, ranking_metric.key))
    for item in measure_items:
        metric = metric_entry_by_ref(
            str(item.get("metricRef") or item.get("metric_ref") or ""),
            asset_pack,
            str(item.get("ownerTable") or item.get("owner_table") or ""),
        )
        if not metric:
            continue
        seed_metrics.append(metric)
        existing.add((metric.table, metric.key))

    added_refs: List[str] = []
    for metric in seed_metrics:
        if not metric:
            continue
        same_table_columns = set(asset_pack.known_columns(metric.table))
        same_table_metric_refs = {item.key for item in asset_pack.metrics if item.table == metric.table}
        for dep_ref in metric_dependency_refs(metric):
            if dep_ref in same_table_columns or dep_ref in same_table_metric_refs:
                continue
            dep_metric = metric_entry_by_ref(dep_ref, asset_pack)
            if not dep_metric:
                continue
            identity = (dep_metric.table, dep_metric.key)
            if identity in existing:
                continue
            expanded.append(
                {
                    "metricRef": dep_metric.key,
                    "ownerTable": dep_metric.table,
                    "sourcePhrase": "semantic formula dependency for %s" % metric.key,
                }
            )
            existing.add(identity)
            added_refs.append(dep_metric.key)
    return expanded, dedupe_strings(added_refs)


def compiled_metric_intent(
    question: str,
    metric: Any,
    task_id: str,
    role: TaskRole,
    mode: AnswerMode,
    grain: str,
    group_by: str,
    depends_on: List[str],
    limit: int,
    asset_pack: PlanningAssetPack,
    metric_resolution: Dict[str, Any] | None = None,
) -> QuestionIntent | None:
    table = metric.table
    columns = set(asset_pack.known_columns(table))
    if not columns:
        return None
    source_column = metric.columns[0] if metric.columns else ""
    metric_formula = metric_formula_for_entry(metric)
    group_column = group_by if group_by in columns else grain_column_for_table(grain, columns)
    output_keys = [column for column in ["seller_id", "merchant_id", group_column, "spu_name"] if column and column in columns]
    required = [column for column in [group_column] + metric_formula_columns(metric_formula, columns) + [source_column] if column and column in columns]
    knowledge_refs = SemanticLayerIndex(question, RecallBundle(), asset_pack).knowledge_refs_for_table(
        table,
        required or output_keys or [source_column],
        reason="llm metric understanding selected node",
    )
    return QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_metric(metric, table),
        answer_mode=mode,
        plan_task_id=task_id,
        task_role=role,
        preferred_table=table,
        metric_column=source_column,
        metric_name=metric.key,
        metric_formula=metric_formula,
        group_by_column=group_column,
        days=extract_days(question, 30),
        limit=limit,
        required_evidence=dedupe_strings(required),
        output_keys=output_keys,
        depends_on_task_ids=depends_on,
        knowledge_refs=knowledge_refs,
        knowledge_ref_ids=[ref.ref_id for ref in knowledge_refs if ref.ref_id],
        analysis_source="llm_question_understanding_compiler",
        analysis_note="metricRef=%s" % metric.key,
        sql_strategy="llm_plan_bound_first",
        metric_resolution=metric_resolution or {},
    )


def grain_column_for_table(grain: str, columns: set) -> str:
    grain_map = {
        "product": ["spu_id", "spu_name"],
        "order": ["sub_order_id", "order_id"],
        "day": ["pt"],
        "ticket": ["ticket_id", "sub_order_id"],
        "refund": ["refund_id", "sub_order_id"],
        "coupon": ["coupon_id", "discount_rel_id"],
    }
    for column in grain_map.get(grain, []):
        if column in columns:
            return column
    return compatible_group_by("", columns)


def compiled_entity_expansion_intent(question: str, anchor: QuestionIntent, asset_pack: PlanningAssetPack) -> QuestionIntent | None:
    columns = set(asset_pack.known_columns(anchor.preferred_table))
    if not anchor.group_by_column or anchor.group_by_column not in columns:
        return None
    output_keys = [
        column
        for column in ["seller_id", "merchant_id", anchor.group_by_column, "sub_order_id", "order_id", "ticket_id", "bill_id", "refund_id", "spu_id", "spu_name"]
        if column in columns
    ]
    knowledge_refs = SemanticLayerIndex(question, RecallBundle(), asset_pack).knowledge_refs_for_table(
        anchor.preferred_table,
        output_keys,
        reason="entity expansion for downstream dependencies",
    )
    return QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_table(anchor.preferred_table),
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="%s_entity_expand" % semantic_domain_for_table(anchor.preferred_table),
        task_role=TaskRole.DEPENDENT,
        preferred_table=anchor.preferred_table,
        days=anchor.days,
        limit=200,
        required_evidence=output_keys,
        output_keys=output_keys,
        depends_on_task_ids=[anchor.plan_task_id],
        knowledge_refs=knowledge_refs,
        knowledge_ref_ids=[ref.ref_id for ref in knowledge_refs if ref.ref_id],
        analysis_source="llm_question_understanding_compiler",
        analysis_note="entity expansion for downstream dependencies",
        sql_strategy="llm_plan_bound_first",
    )


def compiled_bridge_intent(question: str, table: str, asset_pack: PlanningAssetPack, parent_task: str) -> QuestionIntent | None:
    columns = set(asset_pack.known_columns(table))
    if not columns:
        return None
    output_keys = [
        column
        for column in ["seller_id", "merchant_id", "sub_order_id", "order_id", "ticket_id", "bill_id", "refund_id", "spu_id", "spu_name", "pt"]
        if column in columns
    ]
    knowledge_refs = SemanticLayerIndex(question, RecallBundle(), asset_pack).knowledge_refs_for_table(
        table,
        output_keys,
        reason="relationship bridge",
    )
    return QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_table(table),
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="%s_bridge" % semantic_domain_for_table(table),
        task_role=TaskRole.DEPENDENT,
        preferred_table=table,
        days=extract_days(question, 30),
        limit=200,
        required_evidence=output_keys,
        output_keys=output_keys,
        depends_on_task_ids=[parent_task],
        knowledge_refs=knowledge_refs,
        knowledge_ref_ids=[ref.ref_id for ref in knowledge_refs if ref.ref_id],
        analysis_source="llm_question_understanding_compiler",
        analysis_note="relationship bridge",
        sql_strategy="llm_plan_bound_first",
    )


def repair_missing_domain_dependencies(question: str, plan: QueryPlan, asset_pack: PlanningAssetPack) -> QueryPlan:
    if not plan.intents:
        return plan
    requested_domains = requested_semantic_domains_for_plan(question, plan, asset_pack)
    if not requested_domains:
        return plan
    covered_domains = {semantic_domain_for_table(intent.preferred_table) for intent in plan.intents if intent.preferred_table}
    for intent in plan.intents:
        metric_domain = metric_domain_for_intent(intent, asset_pack)
        if metric_domain:
            covered_domains.add(metric_domain)
    missing_domains = [domain for domain in requested_domains if domain not in covered_domains]
    if not missing_domains:
        return plan
    index = SemanticLayerIndex(question, RecallBundle(), asset_pack)
    intents = list(plan.intents)
    dependencies = list(plan.dependencies)
    task_by_table = {intent.preferred_table: intent.plan_task_id for intent in intents if intent.preferred_table and intent.plan_task_id}
    table_by_task = {intent.plan_task_id: intent.preferred_table for intent in intents if intent.preferred_table and intent.plan_task_id}
    added = False
    for domain in missing_domains:
        target_table = best_table_for_domain(domain, asset_pack)
        if not target_table or target_table in task_by_table:
            continue
        path_info = best_existing_path_to_table(task_by_table, target_table, index)
        if not path_info:
            continue
        parent_task, parent_table, path = path_info
        current_task = parent_task
        current_table = parent_table
        for rel in path:
            next_table = rel.right_table if rel.left_table == current_table else rel.left_table
            if next_table == target_table:
                final_intent = compiled_domain_lookup_intent(question, domain, target_table, asset_pack, current_task, next_task_id(domain, intents))
                if not final_intent:
                    break
                intents.append(final_intent)
                task_by_table[target_table] = final_intent.plan_task_id
                table_by_task[final_intent.plan_task_id] = target_table
                dep = dependency_from_relationship(current_task, final_intent.plan_task_id, current_table, target_table, rel)
                add_dependency_if_valid(dependencies, dep)
                added = True
                break
            existing_task = task_by_table.get(next_table)
            if existing_task:
                current_task = existing_task
                current_table = next_table
                continue
            bridge_intent = compiled_bridge_intent(question, next_table, asset_pack, current_task)
            if not bridge_intent:
                break
            bridge_intent = bridge_intent.model_copy(update={"plan_task_id": next_task_id(semantic_domain_for_table(next_table) or "bridge", intents)})
            intents.append(bridge_intent)
            task_by_table[next_table] = bridge_intent.plan_task_id
            table_by_task[bridge_intent.plan_task_id] = next_table
            dep = dependency_from_relationship(current_task, bridge_intent.plan_task_id, current_table, next_table, rel)
            add_dependency_if_valid(dependencies, dep)
            current_task = bridge_intent.plan_task_id
            current_table = next_table
            added = True
    if not added:
        return plan
    repaired = plan.model_copy(update={"intents": intents, "dependencies": dependencies})
    repaired = sync_intent_dependencies(repaired)
    repaired = repaired.model_copy(update={"evidence_contracts": EvidenceContractBuilder().contracts_from_intents(repaired.intents)})
    repaired.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(repaired.intents)
    return repaired


def best_existing_path_to_table(task_by_table: Dict[str, str], target_table: str, index: "SemanticLayerIndex") -> Tuple[str, str, List[Any]] | None:
    best: Tuple[str, str, List[Any]] | None = None
    for table, task_id in task_by_table.items():
        if table == target_table:
            return task_id, table, []
        path = index.relationship_path(table, target_table)
        if not path:
            continue
        if best is None or len(path) < len(best[2]):
            best = (task_id, table, path)
    return best


def compiled_domain_lookup_intent(
    question: str,
    domain: str,
    table: str,
    asset_pack: PlanningAssetPack,
    parent_task: str,
    task_id: str,
) -> QuestionIntent | None:
    columns = set(asset_pack.known_columns(table))
    if not columns:
        return None
    metric = best_metric_for_domain(domain, table, asset_pack, question)
    metric_column = metric.columns[0] if metric and metric.columns else ""
    metric_formula = metric_formula_for_entry(metric) if metric else ""
    group_column = preferred_dependent_group_column(columns)
    evidence_columns = domain_evidence_columns(domain, columns)
    formula_columns = metric_formula_columns(metric_formula, columns)
    required = dedupe_strings([group_column] + evidence_columns + formula_columns + ([metric_column] if metric_column else []))
    output_keys = dedupe_strings(generic_output_keys(QuestionIntent(group_by_column=group_column), columns) + evidence_columns)
    return QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_metric(metric, table) if metric else category_for_table(table),
        answer_mode=AnswerMode.GROUP_AGG if metric else AnswerMode.DETAIL,
        plan_task_id=task_id,
        task_role=TaskRole.DEPENDENT,
        preferred_table=table,
        metric_column=metric_column,
        metric_name=metric.key if metric else "",
        metric_formula=metric_formula,
        group_by_column=group_column,
        days=extract_days(question, 30),
        limit=infer_limit(question),
        required_evidence=required[:18],
        output_keys=output_keys[:18],
        depends_on_task_ids=[parent_task],
        knowledge_refs=SemanticLayerIndex(question, RecallBundle(), asset_pack).knowledge_refs_for_table(table, required or output_keys, reason="semantic repair added missing domain"),
        analysis_source="semantic_missing_domain_repair",
        analysis_note="missingDomain=%s" % domain,
        sql_strategy="llm_plan_bound_first",
    )


def best_metric_for_domain(domain: str, table: str, asset_pack: PlanningAssetPack, question: str) -> Any:
    metrics = [metric for metric in rank_asset_entries(asset_pack.metrics, question) if metric.table == table]
    for metric in metrics:
        if semantic_domain_for_metric(metric) == domain:
            return metric
    return metrics[0] if metrics else None


def preferred_dependent_group_column(columns: set) -> str:
    for column in ["sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id", "refund_id", "bill_id", "pt"]:
        if column in columns:
            return column
    return compatible_group_by("", columns)


def domain_evidence_columns(domain: str, columns: set) -> List[str]:
    candidates = {
        "refund": ["sub_order_id", "order_id", "refund_id", "refund_status_name", "refund_create_time", "pay_amt", "spu_id", "spu_name", "pt"],
        "ticket": ["sub_order_id", "order_id", "ticket_id", "ticket_status_name", "ticket_create_time", "spu_id", "spu_name", "pt"],
        "repay": ["sub_order_id", "order_id", "ticket_id", "bill_id", "repay_amt", "pay_status_name", "pt"],
        "order": ["sub_order_id", "order_id", "pay_amt", "spu_id", "spu_name", "pt"],
        "goods": ["spu_id", "spu_name", "spu_apply_create_time", "spu_status_name", "pt"],
        "coupon": ["coupon_id", "discount_rel_id", "coupon_amt", "pt"],
        "scm": ["spu_id", "spu_name", "inbound_cnt", "pt"],
    }
    return [column for column in candidates.get(domain, []) if column in columns]


def next_task_id(prefix: str, intents: List[QuestionIntent]) -> str:
    base = "%s_lookup" % (prefix or "domain")
    existing = {intent.plan_task_id for intent in intents}
    if base not in existing:
        return base
    index = 2
    while "%s_%d" % (base, index) in existing:
        index += 1
    return "%s_%d" % (base, index)


def compiled_goods_lookup_intent(question: str, table: str, asset_pack: PlanningAssetPack, parent_task: str) -> QuestionIntent | None:
    columns = set(asset_pack.known_columns(table))
    if not columns:
        return None
    output_keys = [
        column
        for column in ["seller_id", "merchant_id", "spu_id", "spu_name", "spu_apply_create_time", "spu_status_name", "pt"]
        if column in columns
    ]
    knowledge_refs = SemanticLayerIndex(question, RecallBundle(), asset_pack).knowledge_refs_for_table(
        table,
        output_keys,
        reason="product dimension lookup",
    )
    return QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        category=category_for_table(table),
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="goods_lookup",
        task_role=TaskRole.DEPENDENT,
        preferred_table=table,
        days=extract_days(question, 30),
        limit=200,
        required_evidence=output_keys,
        output_keys=output_keys,
        depends_on_task_ids=[parent_task],
        knowledge_refs=knowledge_refs,
        knowledge_ref_ids=[ref.ref_id for ref in knowledge_refs if ref.ref_id],
        analysis_source="llm_question_understanding_compiler",
        analysis_note="product dimension lookup",
        sql_strategy="llm_plan_bound_first",
    )


def dependency_requires_unproduced_key(rel: Any, parent_table: str, parent: QuestionIntent) -> bool:
    next_table = rel.right_table if rel.left_table == parent_table else rel.left_table
    dep = dependency_from_relationship(parent.plan_task_id, "probe", parent_table, next_table, rel)
    if not dep:
        return False
    produced = set(parent.output_keys + parent.required_evidence + [parent.group_by_column, parent.filter_column])
    needed = [token for token in split_join_tokens(dep.anchor_column or dep.join_key) if token not in {"seller_id", "merchant_id"}]
    return any(token not in produced for token in needed)


def enrich_llm_plan(question: str, plan: QueryPlan, asset_pack: PlanningAssetPack, payload: Dict[str, Any]) -> QueryPlan:
    """Attach semantic evidence to an LLM-understood graph without choosing the graph for it."""
    known_tables = set(asset_pack.known_tables())
    index = SemanticLayerIndex(question, RecallBundle(), asset_pack)
    enriched_intents: List[QuestionIntent] = []
    for intent in plan.intents:
        metric = metric_entry_for_intent(intent, asset_pack)
        updates: Dict[str, Any] = {}
        if metric:
            updates["metric_name"] = metric.key
            updates["metric_formula"] = metric_formula_for_entry(metric)
            if metric.columns:
                updates["metric_column"] = metric.columns[0]
            if metric.table and metric.table != intent.preferred_table:
                updates["preferred_table"] = metric.table
            updates["category"] = category_for_metric(metric, metric.table or intent.preferred_table)
        if updates:
            intent = intent.model_copy(update=updates)
        if intent.preferred_table and known_tables and intent.preferred_table not in known_tables:
            enriched_intents.append(intent)
            continue
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        if intent.group_by_column and intent.group_by_column not in columns:
            intent = intent.model_copy(update={"group_by_column": compatible_group_by(intent.group_by_column, columns)})
        output_keys = known_columns_only(intent.output_keys, columns)
        for column in generic_output_keys(intent, columns):
            if column not in output_keys:
                output_keys.append(column)
        required = known_columns_only(intent.required_evidence, columns)
        for column in [intent.group_by_column, intent.filter_column, intent.metric_column] + output_keys:
            if column and column in columns and column not in required:
                required.append(column)
        refs = index.knowledge_refs_for_table(table, required or output_keys, reason="llm planner selected node") if table else []
        enriched_intents.append(
            intent.model_copy(
                update={
                    "question": intent.question or question,
                    "intent_type": intent.intent_type or IntentType.VALID,
                    "category": updates.get("category") or category_for_table(table),
                    "days": int(intent.days or extract_days(question, 30)),
                    "limit": int(intent.limit or infer_limit(question)),
                    "required_evidence": required[:18],
                    "output_keys": output_keys[:18],
                    "knowledge_refs": refs,
                    "knowledge_ref_ids": [ref.ref_id for ref in refs if ref.ref_id],
                    "analysis_source": "llm_question_understanding",
                    "analysis_note": intent.analysis_note or understanding_note(payload),
                    "sql_strategy": intent.sql_strategy
                    if intent.sql_strategy in {"structured_first", "llm_first", "llm_plan_bound_first", "llm_first_debug"}
                    else "llm_plan_bound_first",
                }
            )
        )
    plan = plan.model_copy(update={"intents": enriched_intents})
    plan = reconcile_dependencies_with_schema(plan, asset_pack)
    if not plan.evidence_contracts:
        plan.evidence_contracts = EvidenceContractBuilder().contracts_from_intents(plan.intents)
    if not plan.final_required_evidence:
        plan.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(plan.intents)
    understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    if understanding:
        plan.agent_trace.append("llm.question_understanding=%s" % json.dumps(understanding, ensure_ascii=False, default=str)[:600])
    return plan


def metric_entry_for_intent(intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Any:
    candidate_groups: List[set[str]] = []
    if intent.metric_name:
        candidate_groups.append({str(intent.metric_name)})
    if intent.metric_column:
        candidate_groups.append({str(intent.metric_column)})
    if not candidate_groups and intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
        evidence_candidates = {str(item) for item in intent.required_evidence if item}
        if evidence_candidates:
            candidate_groups.append(evidence_candidates)
    if not candidate_groups:
        return None
    preferred_metrics = [metric for metric in asset_pack.metrics if not intent.preferred_table or metric.table == intent.preferred_table]
    for candidates in candidate_groups:
        for metric in preferred_metrics:
            names = {metric.key, metric.title}
            names.update(metric.aliases)
            if candidates & {str(item) for item in names if item}:
                return metric
    for candidates in candidate_groups:
        for metric in preferred_metrics:
            if candidates & {str(item) for item in metric.columns if item}:
                return metric
    if intent.preferred_table:
        fallback_intent = intent.model_copy(update={"preferred_table": ""})
        return metric_entry_for_intent(fallback_intent, asset_pack)
    return None


def metric_domain_for_intent(intent: QuestionIntent, asset_pack: PlanningAssetPack) -> str:
    resolution_domain = semantic_domain_for_metric_resolution(intent.metric_resolution)
    if resolution_domain:
        return resolution_domain
    metric = metric_entry_for_intent(intent, asset_pack)
    return semantic_domain_for_metric(metric) if metric else ""


def anchor_mismatch_issue(plan: QueryPlan) -> Dict[str, Any]:
    understanding = plan.question_understanding or {}
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if not isinstance(ranking, dict) or not plan.intents:
        return {}
    expected_table = str(ranking.get("ownerTable") or ranking.get("owner_table") or "")
    expected_metric = str(ranking.get("metricRef") or ranking.get("metric_ref") or "")
    anchor = plan.intents[0]
    metric_names = {anchor.metric_name, anchor.metric_column}
    if expected_table and anchor.preferred_table != expected_table:
        return {
            "code": "ANCHOR_MISMATCH",
            "severity": "error",
            "taskId": anchor.plan_task_id,
            "reason": "anchor table does not match rankingObjective.ownerTable",
            "expectedTable": expected_table,
            "actualTable": anchor.preferred_table,
        }
    if expected_metric and expected_metric not in {str(item) for item in metric_names if item}:
        resolution = anchor.metric_resolution or {}
        if (
            expected_metric == str(resolution.get("requestedMetricRef") or resolution.get("requested_metric_ref") or "")
            and str(resolution.get("metricKey") or resolution.get("metric_key") or "") in {str(item) for item in metric_names if item}
        ):
            return {}
        return {
            "code": "ANCHOR_MISMATCH",
            "severity": "error",
            "taskId": anchor.plan_task_id,
            "reason": "anchor metric does not match rankingObjective.metricRef",
            "expectedMetric": expected_metric,
            "actualMetric": anchor.metric_name or anchor.metric_column,
        }
    return {}


def analysis_contract_issue(plan: QueryPlan) -> Dict[str, Any]:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    required_evidence = analysis_required_evidence_intents(plan)
    analysis_declared = requires_explanation or (analysis_intent and analysis_intent != "none") or bool(required_evidence)
    if not analysis_declared:
        return {}
    if (requires_explanation or analysis_intent != "none") and not required_evidence:
        return {
            "code": "MISSING_ANALYSIS_EVIDENCE_CONTRACT",
            "severity": "error",
            "analysisIntent": analysis_intent or "none",
            "reason": "questionUnderstanding declares analysis intent but does not declare requiredEvidenceIntents",
        }
    if required_evidence and not analysis_evidence_contract_covered(plan):
        return {
            "code": "ANALYSIS_EVIDENCE_NOT_COVERED",
            "severity": "error",
            "analysisIntent": analysis_intent or "none",
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": item.get("semanticLabel") or item.get("semantic_label") or "",
                    "requiredLevel": item.get("requiredLevel") or item.get("required_level") or "required",
                }
                for item in required_evidence
                if isinstance(item, dict)
            ],
            "reason": "questionUnderstanding.requiredEvidenceIntents are not covered by QueryGraph nodes/dependencies",
        }
    return {}


def analysis_required_evidence_intents(plan: QueryPlan) -> List[Dict[str, Any]]:
    understanding = plan.question_understanding or {}
    raw_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
    if not isinstance(raw_items, list):
        return []
    items = [item for item in raw_items if isinstance(item, dict)]
    return [
        item
        for item in items
        if str(item.get("requiredLevel") or item.get("required_level") or "required").strip().lower() != "optional"
    ]


def analysis_evidence_contract_covered(plan: QueryPlan) -> bool:
    executable = [intent for intent in plan.intents if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE]
    if len(executable) > 1 or plan.dependencies:
        return True
    planned_refs = {
        str(value)
        for intent in executable
        for value in [intent.metric_name, intent.metric_column, intent.preferred_table, intent.group_by_column]
        if value
    }
    required_metric_refs = {
        str(metric_ref)
        for item in analysis_required_evidence_intents(plan)
        if isinstance(item, dict)
        for metric_ref in (item.get("suggestedMetricRefs") or item.get("suggested_metric_refs") or [])
        if metric_ref
    }
    return bool(required_metric_refs and required_metric_refs.issubset(planned_refs))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def metric_resolution_issue(intent: QuestionIntent) -> Dict[str, Any]:
    resolution = intent.metric_resolution or {}
    if not resolution:
        return {}
    task_id = intent.plan_task_id or intent.preferred_table
    requested = str(resolution.get("requestedMetricRef") or resolution.get("requested_metric_ref") or "")
    metric_key = str(resolution.get("metricKey") or resolution.get("metric_key") or "")
    confidence = float(resolution.get("confidence") or 0)
    if requested and not metric_key:
        return {
            "code": "METRIC_RESOLUTION_NEEDED",
            "severity": "error",
            "taskId": task_id,
            "table": intent.preferred_table,
            "metricRef": requested,
            "reason": "metricRef could not be resolved to a semantic metric/source column",
        }
    if requested and 0 < confidence < 0.7:
        return {
            "code": "METRIC_RESOLUTION_LOW_CONFIDENCE",
            "severity": "warning",
            "taskId": task_id,
            "table": intent.preferred_table,
            "metricRef": requested,
            "resolvedMetric": metric_key,
            "confidence": confidence,
            "reason": "metricRef was resolved only by weak semantic matching; planner should read metric definition or re-understand",
        }
    return {}


def unplanned_requested_measure_issues(plan: QueryPlan) -> List[Dict[str, Any]]:
    understanding = plan.question_understanding or {}
    requested_items = requested_metric_items_from_understanding(understanding, include_ranking=False)
    if not requested_items:
        return []
    covered = planned_metric_refs(plan)
    issues: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for item in requested_items:
        metric_ref = str(item.get("metricRef") or item.get("metric_ref") or "")
        owner_table = str(item.get("ownerTable") or item.get("owner_table") or "")
        if not metric_ref:
            continue
        identity = (metric_ref, owner_table)
        if identity in seen:
            continue
        seen.add(identity)
        if owner_table:
            if (owner_table, metric_ref) in covered:
                continue
        elif metric_ref in covered:
            continue
        issues.append(
            {
                "code": "REQUESTED_MEASURE_NOT_PLANNED",
                "severity": "error",
                "metricRef": metric_ref,
                "ownerTable": owner_table,
                "sourcePhrase": str(item.get("sourcePhrase") or item.get("source_phrase") or ""),
                "reason": "LLM questionUnderstanding requested this measure, but QueryGraph has no matching metric node/resolution",
            }
        )
    return issues


def planned_metric_refs(plan: QueryPlan) -> set[Any]:
    refs: set[Any] = set()
    for intent in plan.intents:
        for value in [intent.metric_name, intent.metric_column]:
            if value:
                refs.add(str(value))
                if intent.preferred_table:
                    refs.add((intent.preferred_table, str(value)))
        resolution = intent.metric_resolution or {}
        for key in ["requestedMetricRef", "requested_metric_ref", "metricKey", "metric_key"]:
            value = str(resolution.get(key) or "")
            if value:
                refs.add(value)
                owner_table = str(resolution.get("ownerTable") or resolution.get("owner_table") or intent.preferred_table or "")
                if owner_table:
                    refs.add((owner_table, value))
        for column in resolution.get("sourceColumns") or resolution.get("source_columns") or []:
            value = str(column or "")
            if value:
                refs.add(value)
                if intent.preferred_table:
                    refs.add((intent.preferred_table, value))
    return refs


def reflection_repair_reason(issues: List[Dict[str, Any]]) -> str:
    codes = [str(issue.get("code") or "") for issue in issues]
    if "ANCHOR_MISMATCH" in codes:
        return "ANCHOR_MISMATCH"
    if "METRIC_RESOLUTION_NEEDED" in codes:
        return "METRIC_RESOLUTION_NEEDED"
    if "REQUESTED_MEASURE_NOT_PLANNED" in codes:
        return "METRIC_RESOLUTION_NEEDED"
    if "METRIC_RESOLUTION_LOW_CONFIDENCE" in codes:
        return "METRIC_RESOLUTION_LOW_CONFIDENCE"
    if "DOMAIN_COVERAGE_GAP" in codes:
        return "MISSING_DOMAIN"
    if any(code in codes for code in ["INVALID_EDGE", "BROKEN_DEPENDENCY_ENDPOINT", "MISSING_DEPENDENCY_KEY", "DEPENDENT_WITHOUT_UPSTREAM"]):
        return "MISSING_EDGE"
    if "MISSING_KNOWLEDGE_REF" in codes:
        return "MISSING_KNOWLEDGE_REF"
    if "FRESHNESS_RISK" in codes:
        return "FRESHNESS_RISK"
    if "MISSING_EVIDENCE_CONTRACT" in codes:
        return "MISSING_EVIDENCE_CONTRACT"
    if "MISSING_ANALYSIS_EVIDENCE_CONTRACT" in codes:
        return "ANALYSIS_CONTRACT_MISSING"
    if "ANALYSIS_EVIDENCE_NOT_COVERED" in codes:
        return "MISSING_REQUIRED_EVIDENCE"
    return ""


def semantic_domain_for_metric(metric: Any) -> str:
    if not metric:
        return ""
    text = " ".join(
        [
            str(getattr(metric, "key", "")),
            str(getattr(metric, "title", "")),
            str(getattr(metric, "table", "")),
            " ".join(getattr(metric, "aliases", []) or []),
            str(getattr(metric, "description", "")),
        ]
    ).lower()
    if any(token in text for token in ["refund", "return", "退款", "退货", "售后"]):
        return "refund"
    if any(token in text for token in ["repay", "compensation", "赔付", "理赔", "补偿"]):
        return "repay"
    if any(token in text for token in ["ticket", "cs_", "工单", "客服"]):
        return "ticket"
    if any(token in text for token in ["coupon", "优惠券", "券"]):
        return "coupon"
    if any(token in text for token in ["scm", "inbound", "供应链", "入库"]):
        return "scm"
    if any(token in text for token in ["goods", "spu", "sku", "商品", "上架", "审核"]):
        return "goods"
    if any(token in text for token in ["gmv", "order", "pay", "trade", "订单", "支付", "成交"]):
        return "order"
    return semantic_domain_for_table(str(getattr(metric, "table", "") or ""))


def semantic_domain_for_metric_resolution(resolution: Dict[str, Any]) -> str:
    if not resolution or not resolution.get("metricKey"):
        return ""
    text = " ".join(
        [
            str(resolution.get("requestedMetricRef") or ""),
            str(resolution.get("sourcePhrase") or ""),
            str(resolution.get("metricKey") or ""),
            str(resolution.get("ownerTable") or ""),
            str(resolution.get("displayName") or ""),
            " ".join(str(item) for item in resolution.get("sourceColumns") or []),
        ]
    ).lower()
    if any(token in text for token in ["refund", "return", "退款", "退货", "售后"]):
        return "refund"
    if any(token in text for token in ["repay", "compensation", "赔付", "理赔", "补偿"]):
        return "repay"
    if any(token in text for token in ["ticket", "cs_", "工单", "客服"]):
        return "ticket"
    if any(token in text for token in ["coupon", "优惠券", "券"]):
        return "coupon"
    if any(token in text for token in ["scm", "inbound", "供应链", "入库"]):
        return "scm"
    if any(token in text for token in ["goods", "spu", "sku", "商品", "上架", "审核"]):
        return "goods"
    if any(token in text for token in ["gmv", "order", "pay", "trade", "订单", "支付", "成交"]):
        return "order"
    return semantic_domain_for_table(str(resolution.get("ownerTable") or ""))


def category_for_metric(metric: Any, fallback_table: str = "") -> QuestionCategory:
    domain = semantic_domain_for_metric(metric)
    mapping = {
        "refund": QuestionCategory.REFUND,
        "goods": QuestionCategory.GOODS,
        "ticket": QuestionCategory.CS_TICKET,
        "repay": QuestionCategory.COMPENSATION,
        "coupon": QuestionCategory.COUPON,
        "scm": QuestionCategory.SCM,
        "order": QuestionCategory.TRADE,
        "profile": QuestionCategory.TRADE,
    }
    return mapping.get(domain) or category_for_table(fallback_table)


def metric_formula_for_entry(metric: Any) -> str:
    metadata = getattr(metric, "metadata", {}) or {}
    return str(metadata.get("formula") or metadata.get("metricFormula") or "").strip()


def metric_dependency_refs(metric: Any) -> List[str]:
    metadata = getattr(metric, "metadata", {}) or {}
    refs = [str(item) for item in metadata.get("sourceColumns") or metadata.get("source_columns") or [] if item]
    return dedupe_strings(refs)


def missing_metric_dependencies(intent: QuestionIntent, asset_pack: PlanningAssetPack, planned_metric_names: set[str]) -> List[str]:
    metric = metric_entry_for_intent(intent, asset_pack)
    if not metric:
        return []
    table_columns = set(asset_pack.known_columns(intent.preferred_table))
    same_table_metric_refs = {item.key for item in asset_pack.metrics if item.table == intent.preferred_table}
    missing: List[str] = []
    for ref in metric_dependency_refs(metric):
        if ref in table_columns or ref in same_table_metric_refs:
            continue
        candidate_tables = [item.table for item in asset_pack.metrics if item.key == ref]
        if not candidate_tables or ref not in planned_metric_names:
            missing.append(ref)
    return dedupe_strings(missing)


def knowledge_request_type_for_gap(gap: GraphValidationGap) -> KnowledgeRequestType:
    if gap.code == "MISSING_RELATIONSHIP":
        return KnowledgeRequestType.RELATIONSHIP
    if gap.code == "MISSING_METRIC_DEPENDENCY":
        return KnowledgeRequestType.METRIC
    return KnowledgeRequestType.FIELD


def metric_formula_columns(formula: str, available_columns: set) -> List[str]:
    if not formula:
        return []
    columns: List[str] = []
    for token in re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", formula):
        if token in available_columns and token not in columns:
            columns.append(token)
    return columns


def compatible_group_by(requested: str, columns: set) -> str:
    if requested in columns:
        return requested
    if requested in {"spu_id", "spu_name"}:
        for column in ["spu_id", "spu_name", "sub_order_id", "order_id", "ticket_id", "bill_id"]:
            if column in columns:
                return column
    for column in ["sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id", "bill_id", "refund_id", "pt"]:
        if column in columns:
            return column
    return ""


def reconcile_dependencies_with_schema(plan: QueryPlan, asset_pack: PlanningAssetPack) -> QueryPlan:
    node_by_id = {
        intent.plan_task_id: {"preferredTable": intent.preferred_table}
        for intent in plan.intents
        if intent.plan_task_id and intent.preferred_table
    }
    dependencies: List[PlanDependency] = []
    for dep in plan.dependencies:
        if dep.anchor_task_id == dep.dependent_task_id:
            continue
        if dep.anchor_task_id not in node_by_id or dep.dependent_task_id not in node_by_id:
            add_dependency_if_valid(dependencies, dep)
            continue
        if node_by_id[dep.anchor_task_id]["preferredTable"] == node_by_id[dep.dependent_task_id]["preferredTable"]:
            add_dependency_if_valid(dependencies, dep)
            continue
        reconciled = make_dependency(dep.anchor_task_id, dep.dependent_task_id, node_by_id, asset_pack)
        add_dependency_if_valid(dependencies, reconciled if reconciled.join_key else dep)
    if not dependencies:
        return sync_intent_dependencies(plan.model_copy(update={"dependencies": []}))
    return sync_intent_dependencies(plan.model_copy(update={"dependencies": dependencies}))


def add_dependency_if_valid(dependencies: List[PlanDependency], dependency: PlanDependency) -> None:
    if not dependency.anchor_task_id or not dependency.dependent_task_id:
        return
    if dependency.anchor_task_id == dependency.dependent_task_id:
        return
    if dependency_creates_cycle(dependencies, dependency):
        return
    key = dependency_key(dependency)
    if any(dependency_key(existing) == key for existing in dependencies):
        return
    dependencies.append(dependency)


def dependency_creates_cycle(dependencies: List[PlanDependency], dependency: PlanDependency) -> bool:
    adjacency: Dict[str, List[str]] = {}
    for existing in dependencies:
        if existing.anchor_task_id and existing.dependent_task_id and existing.anchor_task_id != existing.dependent_task_id:
            adjacency.setdefault(existing.anchor_task_id, []).append(existing.dependent_task_id)
    target = dependency.anchor_task_id
    stack = [dependency.dependent_task_id]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def dependency_key(dependency: PlanDependency) -> Tuple[str, str, str, str, str]:
    return (
        dependency.anchor_task_id,
        dependency.dependent_task_id,
        dependency.join_key,
        dependency.anchor_column,
        dependency.dependent_column,
    )


def sync_intent_dependencies(plan: QueryPlan) -> QueryPlan:
    depends_by_task: Dict[str, List[str]] = {}
    for dep in plan.dependencies:
        if dep.anchor_task_id == dep.dependent_task_id:
            continue
        depends_by_task.setdefault(dep.dependent_task_id, [])
        if dep.anchor_task_id not in depends_by_task[dep.dependent_task_id]:
            depends_by_task[dep.dependent_task_id].append(dep.anchor_task_id)
    intents: List[QuestionIntent] = []
    for intent in plan.intents:
        if intent.task_role == TaskRole.DEPENDENT:
            intents.append(intent.model_copy(update={"depends_on_task_ids": depends_by_task.get(intent.plan_task_id, [])}))
        else:
            intents.append(intent.model_copy(update={"depends_on_task_ids": []}))
    return plan.model_copy(update={"intents": intents})


def generic_output_keys(intent: QuestionIntent, columns: set) -> List[str]:
    candidates = [
        "seller_id",
        "merchant_id",
        intent.filter_column,
        intent.group_by_column,
        "sub_order_id",
        "order_id",
        "spu_id",
        "spu_name",
        "refund_id",
        "ticket_id",
        "bill_id",
        "coupon_id",
        "discount_rel_id",
        "pt",
    ]
    return [column for column in candidates if column and column in columns]


def known_columns_only(values: List[str], columns: set) -> List[str]:
    selected: List[str] = []
    for value in values:
        column = str(value or "")
        if column and column in columns and column not in selected:
            selected.append(column)
    return selected


def understanding_note(payload: Dict[str, Any]) -> str:
    understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
    if not isinstance(understanding, dict):
        return ""
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if isinstance(ranking, dict):
        metric = ranking.get("metricRef") or ranking.get("metric_ref") or ranking.get("sourcePhrase") or ""
        if metric:
            return "rankingObjective=%s" % metric
    return "llm_question_understanding"


def normalize_query_graph_payload(question: str, payload: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return QueryPlan()
    known_tables = set(asset_pack.known_tables())
    node_by_id: Dict[str, Dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("nodeId") or node.get("id") or node.get("taskId") or "node_%s" % (index + 1))
        table = str(node.get("preferredTable") or node.get("table") or "")
        if table and (not known_tables or table in known_tables):
            node_by_id[node_id] = {**node, "nodeId": node_id, "preferredTable": table}
    if not node_by_id:
        return QueryPlan()

    anchor_ids = [
        node_id
        for node_id, node in node_by_id.items()
        if "anchor" in str(node.get("role") or node.get("taskRole") or "").lower()
    ]
    if not anchor_ids:
        anchor_ids = [next(iter(node_by_id))]

    dependencies = normalize_dependencies(payload.get("edges") or payload.get("dependencies") or [], node_by_id, anchor_ids, asset_pack)
    depends_by_node: Dict[str, List[str]] = {}
    for dep in dependencies:
        depends_by_node.setdefault(dep.dependent_task_id, [])
        if dep.anchor_task_id not in depends_by_node[dep.dependent_task_id]:
            depends_by_node[dep.dependent_task_id].append(dep.anchor_task_id)
    for node_id in node_by_id:
        if node_id not in anchor_ids and node_id not in depends_by_node:
            anchor = anchor_ids[0]
            if anchor != node_id:
                dependency = make_dependency(anchor, node_id, node_by_id, asset_pack)
                if dependency.join_key:
                    add_dependency_if_valid(dependencies, dependency)
                    depends_by_node.setdefault(node_id, []).append(anchor)

    intents: List[QuestionIntent] = []
    for index, (node_id, node) in enumerate(node_by_id.items()):
        table = str(node.get("preferredTable") or "")
        filter_column, filter_value = first_filter(node)
        role = TaskRole.ANCHOR if node_id in anchor_ids else TaskRole.DEPENDENT
        fields = [str(item) for item in (node.get("fields") or node.get("outputFields") or []) if item]
        intents.append(
            QuestionIntent(
                question=str(node.get("question") or question),
                intent_type=IntentType.VALID,
                category=category_for_table(table),
                answer_mode=answer_mode_for_node(node, question),
                plan_task_id=node_id,
                task_role=role,
                preferred_table=table,
                filter_column=filter_column,
                filter_value=filter_value,
                days=extract_days(question, 30),
                limit=infer_limit(question),
                required_evidence=fields[:12],
                output_keys=fields[:20],
                depends_on_task_ids=depends_by_node.get(node_id, []),
                analysis_source="llm_graph_normalizer",
                analysis_note=str(node.get("role") or node.get("note") or ""),
            )
        )
    return QueryPlan(
        intents=intents,
        dependencies=dependencies,
        final_required_evidence=normalize_output_evidence(payload),
        display_title=str(payload.get("intent") or payload.get("title") or ""),
    )


def normalize_dependencies(
    edges: Any,
    node_by_id: Dict[str, Dict[str, Any]],
    anchor_ids: List[str],
    asset_pack: PlanningAssetPack,
) -> List[PlanDependency]:
    dependencies: List[PlanDependency] = []
    if not isinstance(edges, list):
        return dependencies
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        left = str(edge.get("leftNodeId") or edge.get("source") or edge.get("anchorTaskId") or "")
        right = str(edge.get("rightNodeId") or edge.get("target") or edge.get("dependentTaskId") or "")
        if left not in node_by_id or right not in node_by_id:
            continue
        if left in anchor_ids and right not in anchor_ids:
            anchor_id, dependent_id = left, right
        elif right in anchor_ids and left not in anchor_ids:
            anchor_id, dependent_id = right, left
        else:
            anchor_id, dependent_id = left, right
        if anchor_id == dependent_id:
            continue
        dependency = make_dependency(anchor_id, dependent_id, node_by_id, asset_pack, str(edge.get("relationshipId") or edge.get("relationship") or ""))
        if dependency.join_key:
            add_dependency_if_valid(dependencies, dependency)
    return dependencies


def make_dependency(
    anchor_id: str,
    dependent_id: str,
    node_by_id: Dict[str, Dict[str, Any]],
    asset_pack: PlanningAssetPack,
    relationship_id: str = "",
) -> PlanDependency:
    if anchor_id == dependent_id:
        return PlanDependency(anchor_task_id=anchor_id, dependent_task_id=dependent_id)
    anchor_table = str(node_by_id.get(anchor_id, {}).get("preferredTable") or "")
    dependent_table = str(node_by_id.get(dependent_id, {}).get("preferredTable") or "")
    rel = find_relationship(asset_pack, relationship_id, anchor_table, dependent_table)
    anchor_column, dependent_column, join_key = relationship_columns(rel, anchor_table, dependent_table)
    if not join_key:
        join_key = infer_join_key(anchor_table, dependent_table, asset_pack)
        anchor_column = anchor_column or join_key
        dependent_column = dependent_column or join_key
    return PlanDependency(
        anchor_task_id=anchor_id,
        dependent_task_id=dependent_id,
        join_key=join_key,
        anchor_column=anchor_column,
        dependent_column=dependent_column,
        relation_type="LOOKUP",
    )


def find_relationship(asset_pack: PlanningAssetPack, relationship_id: str, left_table: str, right_table: str) -> Any:
    for rel in asset_pack.relationships:
        if relationship_id and rel.relationship_id == relationship_id:
            return rel
    for rel in asset_pack.relationships:
        if {rel.left_table, rel.right_table} == {left_table, right_table}:
            return rel
    return None


def relationship_columns(rel: Any, anchor_table: str, dependent_table: str) -> Tuple[str, str, str]:
    if not rel:
        return "", "", ""
    left_cols = [str(key.get("leftColumn") or "") for key in rel.join_keys if key.get("leftColumn")]
    right_cols = [str(key.get("rightColumn") or "") for key in rel.join_keys if key.get("rightColumn")]
    if rel.left_table == anchor_table and rel.right_table == dependent_table:
        anchor_cols, dependent_cols = left_cols, right_cols
    elif rel.right_table == anchor_table and rel.left_table == dependent_table:
        anchor_cols, dependent_cols = right_cols, left_cols
    else:
        anchor_cols, dependent_cols = left_cols, right_cols
    anchor_column = "+".join(column for column in anchor_cols if column)
    dependent_column = "+".join(column for column in dependent_cols if column)
    business_keys = [column for column in dependent_cols if column not in {"seller_id", "merchant_id"}]
    join_key = "+".join(business_keys) or dependent_column or anchor_column
    if rel.left_table == anchor_table and rel.right_table == dependent_table:
        return anchor_column, dependent_column, join_key
    if rel.right_table == anchor_table and rel.left_table == dependent_table:
        return anchor_column, dependent_column, join_key
    return anchor_column, dependent_column, join_key


def infer_join_key(anchor_table: str, dependent_table: str, asset_pack: PlanningAssetPack) -> str:
    anchor_cols = set(asset_pack.known_columns(anchor_table))
    dependent_cols = set(asset_pack.known_columns(dependent_table))
    for candidate in ["sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id", "bill_id", "refund_id", "seller_id", "merchant_id"]:
        if candidate in anchor_cols and candidate in dependent_cols:
            return candidate
    return ""


def first_filter(node: Dict[str, Any]) -> Tuple[str, str]:
    filters = node.get("filters") or []
    if not isinstance(filters, list):
        return "", ""
    for item in filters:
        if not isinstance(item, dict):
            continue
        if str(item.get("operator") or "=").strip() in {"=", "==", "IN", "in"}:
            return str(item.get("field") or item.get("column") or ""), str(item.get("value") or "")
    return "", ""


def category_for_table(table: str) -> QuestionCategory:
    lower = table.lower()
    if "profile" in lower:
        return QuestionCategory.TRADE
    if "refund" in lower:
        return QuestionCategory.REFUND
    if "goods" in lower:
        return QuestionCategory.GOODS
    if "ticket" in lower:
        return QuestionCategory.CS_TICKET
    if "repay" in lower or "compensation" in lower:
        return QuestionCategory.COMPENSATION
    if "coupon" in lower:
        return QuestionCategory.COUPON
    if "scm" in lower:
        return QuestionCategory.SCM
    if "deposit" in lower or "appeal" in lower:
        return QuestionCategory.MERCHANT_OTHER
    if "order" in lower or "trade" in lower:
        return QuestionCategory.TRADE
    return QuestionCategory.UNKNOWN


def answer_mode_for_node(node: Dict[str, Any], question: str) -> AnswerMode:
    text = ("%s %s" % (node.get("role") or "", question)).lower()
    if any(word in text for word in ["top", "最多", "最高", "前"]):
        return AnswerMode.TOPN
    if any(word in text for word in ["量", "金额", "cnt", "amt", "count", "sum"]) and "明细" not in text:
        return AnswerMode.GROUP_AGG
    return AnswerMode.DETAIL


def infer_limit(question: str) -> int:
    text = question or ""
    for marker in ["前", "top", "Top", "TOP"]:
        if marker not in text:
            continue
        for size in [20, 10, 5, 3]:
            if str(size) in text:
                return size
    return 20


def normalize_text(value: Any) -> str:
    return str(value or "").lower().replace(" ", "").replace("_", "_")


def group_entries_by_table(entries: List[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for item in entries:
        table = str(getattr(item, "table", "") or "")
        if table:
            grouped.setdefault(table, []).append(item)
    return grouped


def semantic_domain_for_table(table: str) -> str:
    lower = (table or "").lower()
    if "profile" in lower:
        return "profile"
    if "refund" in lower:
        return "refund"
    if "goods" in lower:
        return "goods"
    if "ticket" in lower:
        return "ticket"
    if "repay" in lower or "compensation" in lower:
        return "repay"
    if "coupon" in lower:
        return "coupon"
    if "scm" in lower:
        return "scm"
    if "deposit" in lower or "appeal" in lower:
        return "merchant"
    if "order" in lower or "trade" in lower:
        return "order"
    return "unknown"


def requested_semantic_domains(question: str, asset_pack: PlanningAssetPack) -> List[str]:
    text = normalize_text(question)
    domain_terms = {
        "order": ["订单", "子订单", "下单", "gmv", "成交", "支付"],
        "refund": ["退款", "退货", "售后"],
        "goods": ["商品", "spu", "新品", "审核", "发布"],
        "ticket": ["工单", "客服"],
        "repay": ["赔付", "理赔", "补偿"],
        "coupon": ["优惠券", "券", "补贴"],
        "scm": ["供应链", "入库", "出库"],
        "merchant": ["保证金", "申诉", "处罚", "结算"],
    }
    available = {semantic_domain_for_table(table) for table in asset_pack.known_tables()}
    domains: List[str] = []
    for domain, terms in domain_terms.items():
        if domain in available and any(term in text for term in terms):
            domains.append(domain)
    return domains


def requested_semantic_domains_for_plan(question: str, plan: QueryPlan, asset_pack: PlanningAssetPack) -> List[str]:
    structured = requested_semantic_domains_from_understanding(plan.question_understanding or {}, asset_pack)
    if structured:
        return structured
    return requested_semantic_domains(question, asset_pack)


def requested_semantic_domains_from_understanding(understanding: Dict[str, Any], asset_pack: PlanningAssetPack) -> List[str]:
    if not isinstance(understanding, dict):
        return []
    available = {semantic_domain_for_table(table) for table in asset_pack.known_tables()}
    domains: List[str] = []
    for item in requested_metric_items_from_understanding(understanding, include_ranking=True):
        owner_table = str(item.get("ownerTable") or item.get("owner_table") or "")
        domain = semantic_domain_for_table(owner_table)
        if domain in available:
            domains.append(domain)
    evidence_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            for raw_domain in item.get("suggestedDomains") or item.get("suggested_domains") or []:
                domain = normalize_semantic_domain(str(raw_domain))
                if domain in available:
                    domains.append(domain)
    return dedupe_strings(domains)


def normalize_semantic_domain(value: str) -> str:
    text = (value or "").strip().lower()
    aliases = {
        "trade": "order",
        "order": "order",
        "refund": "refund",
        "goods": "goods",
        "product": "goods",
        "ticket": "ticket",
        "cs_ticket": "ticket",
        "compensation": "repay",
        "repay": "repay",
        "coupon": "coupon",
        "scm": "scm",
        "merchant": "merchant",
        "merchant_other": "merchant",
        "profile": "profile",
    }
    return aliases.get(text, text)


def requested_metric_items_from_understanding(understanding: Dict[str, Any], include_ranking: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(understanding, dict):
        return []
    items: List[Dict[str, Any]] = []
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if include_ranking and isinstance(ranking, dict) and str(ranking.get("metricRef") or ranking.get("metric_ref") or ""):
        items.append(ranking)
    measures = understanding.get("requestedMeasures") or understanding.get("requested_measures") or []
    if isinstance(measures, list):
        items.extend(item for item in measures if isinstance(item, dict))
    return items


def dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def unique_task_id(base: str, existing: Any) -> str:
    taken = {str(item) for item in existing}
    if base not in taken:
        return base
    index = 2
    while "%s_%s" % (base, index) in taken:
        index += 1
    return "%s_%s" % (base, index)


def dependency_from_relationship(anchor_task_id: str, dependent_task_id: str, anchor_table: str, dependent_table: str, rel: Any) -> PlanDependency | None:
    if not rel:
        return None
    anchor_columns: List[str] = []
    dependent_columns: List[str] = []
    for key in rel.join_keys:
        left = str(key.get("leftColumn") or "")
        right = str(key.get("rightColumn") or "")
        if rel.left_table == anchor_table and rel.right_table == dependent_table:
            anchor_columns.append(left)
            dependent_columns.append(right)
        elif rel.right_table == anchor_table and rel.left_table == dependent_table:
            anchor_columns.append(right)
            dependent_columns.append(left)
    anchor_column = "+".join(column for column in anchor_columns if column)
    dependent_column = "+".join(column for column in dependent_columns if column)
    non_partition = [column for column in dependent_columns if column not in {"seller_id", "merchant_id"}]
    join_key = "+".join(non_partition) or dependent_column or anchor_column
    return PlanDependency(
        anchor_task_id=anchor_task_id,
        dependent_task_id=dependent_task_id,
        join_key=join_key,
        anchor_column=anchor_column,
        dependent_column=dependent_column,
        relation_type="LOOKUP",
    )


def dedupe_knowledge_refs(refs: List[KnowledgeRef]) -> List[KnowledgeRef]:
    deduped: List[KnowledgeRef] = []
    seen = set()
    for ref in refs:
        key = ref.ref_id or "%s:%s:%s:%s" % (ref.ref_type, ref.table, ref.column, ref.relationship_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def extract_entity_value(question: str, column: str) -> str:
    text = question or ""
    normalized_column = (column or "").lower()
    if normalized_column == "sub_order_id":
        match = re.search(r"\bsub_order_id[_:=：-]*[A-Za-z0-9_-]*\b", text, re.IGNORECASE)
        return match.group(0).strip(":=：-") if match else ""
    if normalized_column == "order_id":
        for match in re.finditer(r"\border_id[_:=：-]*[A-Za-z0-9_-]*\b", text, re.IGNORECASE):
            value = match.group(0).strip(":=：-")
            if not value.lower().startswith("sub_order_id"):
                return value
        return ""
    match = re.search(r"\b%s[_:=：-]*[A-Za-z0-9_-]*\b" % re.escape(normalized_column), text, re.IGNORECASE)
    return match.group(0).strip(":=：-") if match else ""


def normalize_output_evidence(payload: Dict[str, Any]) -> List[str]:
    output = payload.get("output") or {}
    fields = output.get("fields") if isinstance(output, dict) else []
    if not isinstance(fields, list):
        return []
    evidence: List[str] = []
    for item in fields:
        if isinstance(item, dict):
            value = str(item.get("alias") or item.get("field") or "")
        else:
            value = str(item or "")
        if value:
            evidence.append(value)
    return evidence[:24]


def compact_table_entry(item: Any, question: str = "") -> Dict[str, Any]:
    return {
        "table": item.table or item.key,
        "topic": item.topic,
        "title": item.title,
        "keyColumns": select_planner_columns(item.columns, question),
        "description": trim_text(item.description, 80),
    }


def compact_metric_entry(item: Any) -> Dict[str, Any]:
    payload = {
        "key": item.key,
        "table": item.table,
        "title": item.title,
        "columns": item.columns[:4],
    }
    metadata = compact_metadata(item.metadata)
    if metadata:
        payload["metadata"] = metadata
    return payload


def compact_field_entry(item: Any) -> Dict[str, Any]:
    return {
        "key": item.key,
        "table": item.table,
        "title": item.title,
    }


def compact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}
    allowed = {
        "aggregation",
        "agg",
        "formula",
        "metricFormula",
        "businessMeaning",
        "unit",
        "warning",
        "joinKey",
        "semanticType",
        "dataType",
    }
    compacted: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in allowed:
            continue
        if isinstance(value, str):
            compacted[key] = trim_text(value, 80)
        elif isinstance(value, (int, float, bool)):
            compacted[key] = value
        elif isinstance(value, list):
            compacted[key] = [trim_text(str(item), 50) for item in value[:4]]
        elif isinstance(value, dict):
            compacted[key] = {str(k): trim_text(str(v), 50) for k, v in list(value.items())[:4]}
    return compacted


def compact_relationship_entry(item: Any) -> Dict[str, Any]:
    return {
        "relationshipId": item.relationship_id,
        "leftTable": item.left_table,
        "rightTable": item.right_table,
        "joinKeys": item.join_keys,
        "description": trim_text(item.description, 80),
    }


def compact_skill(skill: Any) -> Dict[str, Any]:
    return {
        "domain": skill.domain,
        "displayName": skill.display_name,
        "retrievalHints": [trim_text(item, 70) for item in getattr(skill, "retrieval_hints", [])[:3]],
        "fieldWarnings": [trim_text(item, 70) for item in skill.field_warnings[:3]],
        "answerGuidelines": [trim_text(item, 70) for item in skill.answer_guidelines[:3]],
    }


def compact_missing_live_columns(missing: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        table: {"count": len(columns), "sample": columns[:12]}
        for table, columns in missing.items()
        if columns
    }


def rank_asset_entries(entries: List[Any], question: str) -> List[Any]:
    terms = extract_question_terms(question)
    if not terms:
        return list(entries)

    return sorted(entries, key=lambda item: asset_entry_score(item, terms), reverse=True)


def asset_entry_score(item: Any, terms: List[str]) -> int:
    metadata = getattr(item, "metadata", {}) or {}
    strong_text = " ".join(
        [
            str(getattr(item, "key", "")),
            str(getattr(item, "title", "")),
            str(getattr(item, "business_name", "")),
            str(getattr(item, "businessName", "")),
            " ".join(getattr(item, "aliases", []) or []),
        ]
    ).lower()
    description = str(getattr(item, "description", "")).lower()
    score = sum(3 for term in terms if term and term in strong_text)
    score += sum(1 for term in terms if term and term in description)
    term_set = set(terms)
    if not (term_set & {"rate", "ratio", "率", "比例", "占比"}) and any(token in strong_text for token in ["rate", "ratio", "比例", "占比"]):
        score -= 2
    if not (term_set & {"优惠", "优惠券", "coupon", "券", "补贴", "discount"}) and any(
        token in strong_text for token in ["coupon", "discount", "优惠", "券", "补贴"]
    ):
        score -= 1
    source_columns = metadata.get("sourceColumns") or metadata.get("source_columns") or []
    formula = str(metadata.get("formula") or metadata.get("metricFormula") or "").lower()
    if (len(source_columns) > 1 or "/" in formula or "-" in formula) and not (
        term_set & {"净", "扣", "扣除", "after", "综合", "派生"}
    ):
        score -= 4
    return score


def select_planner_columns(columns: List[str], question: str) -> List[str]:
    terms = extract_question_terms(question)
    priority_fragments = [
        "merchant_id",
        "seller_id",
        "order_id",
        "sub_order_id",
        "refund_id",
        "ticket_id",
        "compensation",
        "repay",
        "spu",
        "sku",
        "goods",
        "coupon",
        "warehouse",
        "pt",
        "date",
        "time",
        "create",
        "publish",
        "audit",
        "status",
        "amt",
        "amount",
        "gmv",
        "cnt",
        "user",
    ]
    selected: List[str] = []
    for column in columns:
        lowered = column.lower()
        if any(fragment in lowered for fragment in priority_fragments) or any(term in lowered for term in terms):
            selected.append(column)
        if len(selected) >= 16:
            break
    if len(selected) < 12:
        for column in columns:
            if column not in selected:
                selected.append(column)
            if len(selected) >= 12:
                break
    return selected


def extract_question_terms(question: str) -> List[str]:
    text = (question or "").lower()
    raw_terms = re_split_terms(text)
    return [term for term in raw_terms if len(term) >= 2][:24]


def re_split_terms(text: str) -> List[str]:
    normalized = text.replace("_", " ")
    terms: List[str] = []
    for chunk in normalized.replace("，", " ").replace("。", " ").replace(",", " ").split():
        chunk = chunk.strip()
        if chunk and chunk not in terms:
            terms.append(chunk)
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]*|\d+", text):
        normalized_token = token.lower()
        if normalized_token and normalized_token not in terms:
            terms.append(normalized_token)
    for keyword in [
        "订单",
        "子订单",
        "下单",
        "下单数",
        "下单量",
        "订单数",
        "订单量",
        "销量",
        "退款",
        "退款量",
        "退款金额",
        "退款率",
        "退货",
        "商品",
        "spu",
        "sku",
        "工单",
        "工单量",
        "赔付",
        "赔付金额",
        "赔付单量",
        "优惠券",
        "供应链",
        "入库",
        "入库量",
        "审核",
        "发布",
        "发布成功",
        "金额",
        "最多",
        "最高",
        "净",
        "扣",
        "扣除",
    ]:
        if keyword in text and keyword not in terms:
            terms.append(keyword)
    return terms


def split_join_tokens(value: str) -> List[str]:
    if not value:
        return []
    tokens: List[str] = []
    for piece in str(value).replace("+", ",").split(","):
        token = piece.strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def trim_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def parse_knowledge_requests(items: Any) -> List[KnowledgeRequest]:
    requests: List[KnowledgeRequest] = []
    if not isinstance(items, list):
        return requests
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["type"] = normalize_knowledge_request_type(normalized.get("type"))
        try:
            requests.append(KnowledgeRequest(**normalized))
        except Exception:
            requests.append(
                KnowledgeRequest(
                    type=KnowledgeRequestType.FIELD,
                    query=str(normalized.get("query") or normalized.get("reason") or ""),
                    needed_for_task_id=str(normalized.get("neededForTaskId") or normalized.get("needed_for_task_id") or ""),
                    reason=str(normalized.get("reason") or "LLM 返回了不完整 knowledge request"),
                )
            )
    return requests


def normalize_knowledge_request_type(value: Any) -> str:
    raw = str(value or "").upper().strip()
    aliases = {
        "FIELD_OR_METRIC": "FIELD",
        "METRIC_OR_FIELD": "METRIC",
        "TABLE_OR_FIELD": "TABLE",
        "SCHEMA": "FIELD",
        "JOIN": "RELATIONSHIP",
        "JOIN_KEY": "RELATIONSHIP",
        "RELATION": "RELATIONSHIP",
        "RULE": "BUSINESS_RULE",
        "REALTIME": "REALTIME_FALLBACK",
    }
    raw = aliases.get(raw, raw)
    allowed = {item.value for item in KnowledgeRequestType}
    return raw if raw in allowed else KnowledgeRequestType.FIELD.value


def is_formula(value: str) -> bool:
    return "(" in value and ")" in value
