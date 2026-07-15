from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from merchant_ai.models import GraphValidationGap, KnowledgeRequest, PlanningAssetPack, QueryPlan, RecallBundle


PlanFromPayload = Callable[[str, Dict[str, Any], PlanningAssetPack], QueryPlan]
LlmUnderstand = Callable[..., Dict[str, Any]]
LlmRepair = Callable[[str, QueryPlan, PlanningAssetPack, List[GraphValidationGap]], Dict[str, Any]]


@dataclass
class UnderstandingExtractor:
    """Owns question-understanding acquisition policy, without compiling QueryGraph nodes."""

    planner: Any

    def prior_understanding(self, planner_context: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if isinstance(planner_context, dict) and isinstance(planner_context.get("previousUnderstanding"), dict):
            return planner_context.get("previousUnderstanding")
        return None

    def semantic_fast_path(self, question: str, asset_pack: PlanningAssetPack) -> QueryPlan:
        return self.planner._semantic_fast_path(question, asset_pack)

    def initial_payload(
        self,
        question: str,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        planner_context: Dict[str, Any] | None,
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any] | None, str]:
        prior = self.prior_understanding(planner_context)
        initial_tool_entry = self.planner._initial_semantic_tool_entry(question, asset_pack, gaps, planner_context)
        payload = self.planner._llm_understand(
            question,
            asset_pack,
            gaps,
            trace,
            planner_context=planner_context,
            prior_understanding=prior,
            use_tool_loop=bool(initial_tool_entry),
            filesystem_context_entry=initial_tool_entry or "fast_path",
        )
        return payload, bool(initial_tool_entry), prior, initial_tool_entry or ""

    def recovery_payload(
        self,
        question: str,
        asset_pack: PlanningAssetPack,
        gaps: List[GraphValidationGap],
        trace: List[str],
        planner_context: Dict[str, Any] | None,
        prior_understanding: Dict[str, Any] | None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.planner._llm_understand(
            question,
            asset_pack,
            gaps,
            trace,
            planner_context=planner_context,
            prior_understanding=prior_understanding,
            **kwargs,
        )

    def failure_fallback_plan(self, question: str, asset_pack: PlanningAssetPack, trace_reason: str) -> Tuple[QueryPlan, List[KnowledgeRequest], str]:
        if has_knowledge_request_gaps(asset_pack):
            return QueryPlan(agent_trace=[trace_reason, "planner.failure_fallback=blocked_by_knowledge_request_gaps"]), [], trace_reason
        rejected_trace: List[str] = []
        entity_plan = self.planner._entity_detail_fallback(question, asset_pack)
        if entity_plan.intents:
            coverage_gaps = self.planner._failure_candidate_coverage_gaps(question, entity_plan, asset_pack)
            if not coverage_gaps:
                entity_plan.agent_trace.extend(
                    [trace_reason, "planner.entity_id_semantic_fallback_after_llm_failure", "planner.semantic_fast_path=entity_detail_after_failure"]
                )
                return entity_plan, [], "SEMANTIC_FAST_PATH"
            rejected_trace.extend(fallback_coverage_rejection_trace("entity_detail", coverage_gaps))
        recalled_metric_plan = self.planner._recalled_metric_diagnostic_fallback(question, asset_pack)
        if recalled_metric_plan.intents:
            coverage_gaps = self.planner._failure_candidate_coverage_gaps(question, recalled_metric_plan, asset_pack)
            if not coverage_gaps:
                recalled_metric_plan.agent_trace.extend([trace_reason, "planner.recalled_metric_diagnostic_fallback_after_llm_failure"])
                return recalled_metric_plan, [], "SEMANTIC_FAST_PATH"
            rejected_trace.extend(fallback_coverage_rejection_trace("recalled_metric_diagnostic", coverage_gaps))
        if trace_reason == "planner.no_llm_configured":
            metric_plan = self.planner._semantic_metric_fallback(question, asset_pack)
            if metric_plan.intents:
                coverage_gaps = self.planner._failure_candidate_coverage_gaps(question, metric_plan, asset_pack)
                if not coverage_gaps:
                    metric_plan.agent_trace.extend([trace_reason, "planner.semantic_metric_fallback_after_llm_failure"])
                    return metric_plan, [], "SEMANTIC_FAST_PATH"
                rejected_trace.extend(fallback_coverage_rejection_trace("semantic_metric", coverage_gaps))
        else:
            rejected_trace.append("semantic_metric_fallback_skipped_after_llm_failure")
        trend_plan = self.planner._multi_metric_trend_fallback(question, asset_pack)
        if trend_plan.intents:
            coverage_gaps = self.planner._failure_candidate_coverage_gaps(question, trend_plan, asset_pack)
            if not coverage_gaps:
                trend_plan.agent_trace.extend([trace_reason, "planner.multi_metric_trend_fallback_after_llm_failure"])
                return trend_plan, [], "SEMANTIC_FAST_PATH"
            rejected_trace.extend(fallback_coverage_rejection_trace("multi_metric_trend", coverage_gaps))
        return QueryPlan(agent_trace=[trace_reason, *rejected_trace, "planner.failure_fallback=fail_closed_coverage"]), [], trace_reason


def has_knowledge_request_gaps(asset_pack: PlanningAssetPack) -> bool:
    return bool((asset_pack.metric_compaction or {}).get("knowledgeRequestGaps"))


def repair_payload_knowledge_requests(payload: Dict[str, Any]) -> List[KnowledgeRequest]:
    raw_items = payload.get("knowledgeRequests") or payload.get("knowledge_requests") or []
    requests: List[KnowledgeRequest] = []
    for item in raw_items or []:
        if isinstance(item, KnowledgeRequest):
            requests.append(item)
        elif isinstance(item, dict):
            try:
                requests.append(KnowledgeRequest.model_validate(item))
            except Exception:
                continue
    return requests


@dataclass
class PlanCompiler:
    """Compiles an extracted understanding payload into a QueryPlan."""

    compiler: Any
    coverage_critic: Any
    expand_asset_pack: Callable[[PlanningAssetPack, Dict[str, Any]], List[str]]
    enrich_plan: Callable[[str, QueryPlan, PlanningAssetPack, Dict[str, Any]], QueryPlan]
    append_prompt_trace: Callable[[QueryPlan, Dict[str, Any]], None]
    attach_tool_trace: Callable[[QueryPlan, Dict[str, Any]], None]

    def compile(self, question: str, payload: Dict[str, Any], asset_pack: PlanningAssetPack) -> QueryPlan:
        understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
        if understanding:
            expansion_trace = self.expand_asset_pack(asset_pack, understanding)
            coverage = self.coverage_critic.complete(question, understanding, asset_pack)
            understanding = coverage.understanding
            expansion_trace.extend(self.expand_asset_pack(asset_pack, understanding))
            plan = self.compiler.compile(question, understanding, asset_pack)
            if coverage.trace:
                plan.compiler_trace = coverage.trace + plan.compiler_trace
            if coverage.added_measures:
                plan.agent_trace.append("planner.understanding_coverage_critic=semantic_metric_completion")
            if expansion_trace:
                plan.compiler_trace.extend(expansion_trace)
            if plan.intents:
                plan = self.enrich_plan(question, plan, asset_pack, payload)
                self.append_prompt_trace(plan, payload)
                self.attach_tool_trace(plan, payload)
                plan.agent_trace.append("planner=llm_understanding_compiled")
                return plan
            if plan.knowledge_requests:
                self.append_prompt_trace(plan, payload)
                self.attach_tool_trace(plan, payload)
                plan.agent_trace.append("planner=llm_understanding_needs_semantic_metric_evidence")
                return plan
        if payload.get("queryPlan"):
            plan = QueryPlan(agent_trace=["planner.query_plan_payload_ignored"])
            self.append_prompt_trace(plan, payload)
            self.attach_tool_trace(plan, payload)
            return plan
        plan = QueryPlan()
        self.append_prompt_trace(plan, payload)
        self.attach_tool_trace(plan, payload)
        return plan


@dataclass
class PlanRepairer:
    """Owns planner repair policy."""

    llm: Any
    compiler: Any
    root_metric_repair: Callable[[str, QueryPlan, PlanningAssetPack, Any], QueryPlan]
    dependency_key_repair: Callable[[str, QueryPlan, PlanningAssetPack, List[GraphValidationGap]], QueryPlan]
    missing_domain_repair: Callable[[str, QueryPlan, PlanningAssetPack], QueryPlan]
    llm_repair: Callable[[str, QueryPlan, PlanningAssetPack, List[GraphValidationGap]], Dict[str, Any]]
    enrich_plan: Callable[[str, QueryPlan, PlanningAssetPack, Dict[str, Any]], QueryPlan]

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
        root_repaired = self.root_metric_repair(question, plan, asset_pack, self.compiler)
        if root_repaired.compiler_trace != plan.compiler_trace or root_repaired.question_understanding != plan.question_understanding:
            root_repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=promote_more_specific_root_metric"])
            return root_repaired
        semantic_repaired = self.dependency_key_repair(question, plan, asset_pack, gaps)
        if semantic_repaired.compiler_trace != plan.compiler_trace or len(semantic_repaired.dependencies) != len(plan.dependencies):
            semantic_repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=semantic_relationship_graph_bridge"])
            return semantic_repaired
        semantic_repaired = self.missing_domain_repair(question, plan, asset_pack)
        if len(semantic_repaired.intents) > len(plan.intents):
            semantic_repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=semantic_missing_domains"])
            return semantic_repaired
        if self.llm.configured and gaps:
            payload = self.llm_repair(question, plan, asset_pack, gaps)
            understanding = payload.get("questionUnderstanding") or payload.get("question_understanding") or {}
            if understanding:
                repaired = self.compiler.compile(question, understanding, asset_pack)
                if repaired.intents:
                    repaired = self.enrich_plan(question, repaired, asset_pack, payload)
                    repaired.agent_trace.extend(plan.agent_trace + ["planner.repair=llm_reunderstanding"])
                    return repaired
            repair_requests = repair_payload_knowledge_requests(payload)
            if repair_requests:
                return QueryPlan(
                    knowledge_requests=repair_requests,
                    agent_trace=plan.agent_trace + ["planner.repair=llm_requested_knowledge"],
                )
        plan.agent_trace.append("planner.repair.unavailable")
        return plan


@dataclass
class GraphContractValidator:
    """Stable façade for graph contract validation."""

    validator: Any

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        return self.validator.validate(*args, **kwargs)


def fallback_coverage_rejection_trace(candidate: str, gaps: List[GraphValidationGap]) -> List[str]:
    return [
        "planner.failure_fallback.coverage_rejected=%s:%s:%s"
        % (candidate, gap.code, gap.evidence)
        for gap in gaps[:8]
    ]
