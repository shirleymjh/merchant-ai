from __future__ import annotations

import json
import re
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from merchant_ai.models import (
    AgentRunResult,
    AnswerMode,
    AnswerClaim,
    AnswerClaimVerification,
    ChatContext,
    ChatDataSection,
    DailyReportResponse,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    VerifiedAnswerContext,
    category_display,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.memory import MemoryStore
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, PendingAnswerStore
from merchant_ai.services.security import identity_scope_hash
from merchant_ai.services.answer_formatting import (
    answer_numeric_value,
    extract_question_time_phrase,
    format_cell,
    format_metric_value_for_answer,
    humanize_column_name,
    identifier_like_column,
)
from merchant_ai.services.answer_claims import AnswerClaimVerifier, build_verified_facts
from merchant_ai.services.time_semantics import declared_time_column_for_intent


TIME_DIMENSION_KEY = "time_dimension"


def answer_context_policy() -> str:
    return (
        "AnswerAgent 只读取 VerifiedAnswerContext 中的 question、businessContext、verifiedFacts、dataRows、dataSections、metricDisclosures、evidenceGaps、degradedReasons、analysisDraft、mandatoryAnswerSkeleton；不要读取或推断 QueryGraph。"
        "metricFacts 是本轮必须覆盖的指标事实清单，最终回答必须自然覆盖其中每个指标名称、数值和时间范围；不要遗漏，也不要机械照抄模板。"
        "metricComparisonFacts 是已验证的同指标跨时间窗口对比事实；用户问相比、变化、环比时必须覆盖当前值、对比窗口值和变化方向。"
        "verifiedFacts 是数字、日期、实体和排序结论的唯一事实来源；任何事实陈述都必须能直接绑定其中的 factId，不能补充未出现的数值或业务实体。"
        "mandatoryAnswerSkeleton 是必须覆盖的答案事实骨架；可以改写成自然语言，但不能遗漏其中的指标、数值、时间范围和缺口说明。"
        "你的输出面向商家，不面向研发或分析师；语气要像经营助手，先直接回答用户问题，再给必要说明和建议。"
        "不要使用“分析结论”“关键证据”“限制”“证据门禁”“当前证据显示”“已看到的点位显示”这类报告或内部调试话术。"
        "不要说“查到几行”“使用表”“SQL”“字段名”“Doris”；不要输出 markdown 表格，表格和图表由前端结构化区域渲染。"
        "核心经营指标可以默认保留一句业务口径说明，只说统计对象、时间和店铺范围；用户没有问口径时，不要展开字段、来源表和计算公式。"
        "同一指标存在多个候选口径时，只回答语义层确认的主口径，不要把多个相似口径并列解释。"
        "用户提到和后台/看板数据不一致时，进入口径对账思路，优先说明时间范围、统计对象、过滤条件、聚合粒度和数据更新时间。"
        "如果是趋势问题，第一段直接写“最近N天，指标从 A 变化到 B，整体上升/下降 C。”，不要写“趋势里”“点位显示”；有峰值和低点时用一句话说明。"
        "dataRows 或 dataSections 中 resultRole=summary 的行是已验证汇总结果，优先用于回答总量；resultRole=trend_context 的行只用于解释趋势。"
        "不要因为趋势只有部分日期有点位，就否定 summary 汇总；不要说“其余日期没有看到明细”。"
        "如果用户基于已返回明细继续追问分析或建议，直接归纳原因、风险和行动建议；不要再次逐行复述明细，也不要重新输出同一批明细。"
        "如果用户询问指标但没有显式说“给建议”，仍要基于该指标的变化、峰谷或维度差异给出简短经营判断，并给至少 2 条与本轮数据直接相关的可执行建议。"
        "如果 evidenceGaps 存在，用“说明：”简短提示，不要扩大成失败结论。"
        "最后输出“建议：”，用短横线列出最多 2 条；建议必须结合 businessContext 的商家画像、长期记忆/近期关注和本轮数据，避免泛泛说继续追问。"
    )


class AnswerComposeService:
    def __init__(self, llm: LlmClient):
        self.llm = llm
        self.prompt_assembler = PromptAssembler()
        self._last_prompt_chars: ContextVar[int] = ContextVar("answer_prompt_chars_%x" % id(self), default=0)
        self._last_analysis_skill_trace: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            "answer_skill_trace_%x" % id(self),
            default=None,
        )
        self._last_compose_llm_attempted: ContextVar[bool] = ContextVar(
            "answer_compose_attempted_%x" % id(self),
            default=False,
        )
        self._last_compose_used_llm: ContextVar[bool] = ContextVar(
            "answer_compose_used_llm_%x" % id(self),
            default=False,
        )
        self._last_answer_claim_trace: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            "answer_claim_trace_%x" % id(self),
            default=None,
        )

    @property
    def last_prompt_chars(self) -> int:
        return int(self._last_prompt_chars.get())

    @last_prompt_chars.setter
    def last_prompt_chars(self, value: int) -> None:
        self._last_prompt_chars.set(int(value or 0))

    @property
    def last_analysis_skill_trace(self) -> Dict[str, Any]:
        trace = self._last_analysis_skill_trace.get()
        if trace is None:
            trace = {}
            self._last_analysis_skill_trace.set(trace)
        return trace

    @last_analysis_skill_trace.setter
    def last_analysis_skill_trace(self, value: Dict[str, Any]) -> None:
        self._last_analysis_skill_trace.set(dict(value or {}))

    @property
    def last_compose_llm_attempted(self) -> bool:
        return bool(self._last_compose_llm_attempted.get())

    @last_compose_llm_attempted.setter
    def last_compose_llm_attempted(self, value: bool) -> None:
        self._last_compose_llm_attempted.set(bool(value))

    @property
    def last_compose_used_llm(self) -> bool:
        return bool(self._last_compose_used_llm.get())

    @last_compose_used_llm.setter
    def last_compose_used_llm(self, value: bool) -> None:
        self._last_compose_used_llm.set(bool(value))

    @property
    def last_answer_claim_trace(self) -> Dict[str, Any]:
        trace = self._last_answer_claim_trace.get()
        if trace is None:
            trace = {}
            self._last_answer_claim_trace.set(trace)
        return trace

    @last_answer_claim_trace.setter
    def last_answer_claim_trace(self, value: Dict[str, Any]) -> None:
        self._last_answer_claim_trace.set(dict(value or {}))

    def compose(
        self,
        question: str,
        merchant: MerchantInfo,
        plan: QueryPlan,
        run_result: AgentRunResult,
        knowledge_context: str,
        analysis_summary: str = "",
        allow_llm: bool = True,
        rule_context: str = "",
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.last_compose_llm_attempted = False
        self.last_compose_used_llm = False
        self.last_prompt_chars = 0
        self.last_answer_claim_trace = {}
        if not plan.intents:
            return self._no_execution_answer(plan)
        primary = plan.intents[0] if plan.intents else QuestionIntent()
        if primary.answer_mode == AnswerMode.CHAT:
            return "您好，我是 yshopping 商家 AI 助手，可以查询已接入的经营指标、明细和规则信息。"
        if primary.answer_mode == AnswerMode.INVALID:
            return "问题还缺少业务对象或查询范围，请补充要看的指标、时间范围或业务域。"
        if primary.answer_mode == AnswerMode.RULE:
            return self._compose_rule_answer(question, knowledge_context)
        effective_rule_context = rule_context if plan_requires_rule_evidence(plan) else ""
        bundle = run_result.merged_query_bundle if run_result else QueryBundle()
        mandatory_skeleton = self._mandatory_answer_skeleton(question, plan, run_result)
        if analysis_summary:
            if answer_metric_comparison_facts(question, plan, run_result):
                grounded = deterministic_structured_answer(question, plan, run_result)
                if grounded:
                    return self._compose_final_answer(
                        grounded,
                        mandatory_skeleton,
                        question,
                        plan,
                        run_result,
                        bundle,
                        effective_rule_context,
                        merchant,
                        personalization_context,
                    )
            if deterministic_single_semantic_metric_answer(plan) or not question_requests_diagnosis(question):
                grounded = deterministic_structured_answer(question, plan, run_result)
                if grounded:
                    return self._compose_final_answer(
                        grounded,
                        mandatory_skeleton,
                        question,
                        plan,
                        run_result,
                        bundle,
                        effective_rule_context,
                        merchant,
                        personalization_context,
                    )
            cleaned_summary = sanitize_business_answer_text(analysis_summary, question, plan, run_result)
            cleaned_summary = self._ensure_multi_trend_answer_coverage(cleaned_summary, question, plan, run_result)
            cleaned_summary = self._ensure_multi_metric_summary_coverage(cleaned_summary, question, plan, run_result)
            answer = ensure_required_field_answer_coverage(cleaned_summary, plan, run_result)
            return self._compose_final_answer(
                answer,
                mandatory_skeleton,
                question,
                plan,
                run_result,
                bundle,
                effective_rule_context,
                merchant,
                personalization_context,
            )
        if primary.intent_type == "VALID" and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT} and (not run_result or not run_result.task_results):
            return self._no_execution_answer(plan)
        if run_result and run_result.task_results and all(result.query_bundle.failed for result in run_result.task_results):
            return self._finalize_answer(
                self._append_rule_evidence(
                    self.append_business_advice(
                        self._execution_failure_answer(run_result),
                        plan.intents,
                        bundle,
                        question=question,
                        plan=plan,
                        run_result=run_result,
                        merchant=merchant,
                        personalization_context=personalization_context,
                        allow_llm=False,
                    ),
                    question,
                    effective_rule_context,
                ),
                question,
                plan,
                run_result,
            )
        partial_blocking_answer = blocking_evidence_partial_answer(question, plan, run_result)
        if partial_blocking_answer:
            return self._compose_final_answer(
                partial_blocking_answer,
                mandatory_skeleton,
                question,
                plan,
                run_result,
                bundle,
                effective_rule_context,
                merchant,
                personalization_context,
                append_advice=False,
            )
        if question_asks_metric_reconciliation(question):
            reconciliation_answer = self._metric_reconciliation_answer(question, plan, run_result)
            if reconciliation_answer:
                return self._compose_final_answer(
                    reconciliation_answer,
                    mandatory_skeleton,
                    question,
                    plan,
                    run_result,
                    bundle,
                    effective_rule_context,
                    merchant,
                    personalization_context,
                )
        ranking_answer = deterministic_ranking_answer(question, plan, run_result)
        if answer_metric_comparison_facts(question, plan, run_result):
            structured_answer = deterministic_structured_answer(question, plan, run_result)
            if structured_answer:
                structured_answer = ensure_required_field_answer_coverage(structured_answer, plan, run_result)
                return self._compose_final_answer(
                    structured_answer,
                    mandatory_skeleton,
                    question,
                    plan,
                    run_result,
                    bundle,
                    effective_rule_context,
                    merchant,
                    personalization_context,
                )
        if not question_requests_diagnosis(question):
            structured_answer = ranking_answer or deterministic_structured_answer(question, plan, run_result)
            if structured_answer:
                structured_answer = ensure_required_field_answer_coverage(structured_answer, plan, run_result)
                return self._compose_final_answer(
                    structured_answer,
                    mandatory_skeleton,
                    question,
                    plan,
                    run_result,
                    bundle,
                    effective_rule_context,
                    merchant,
                    personalization_context,
                )
        llm_first_attempted = False
        if allow_llm and self.llm.configured and (bundle.rows or run_result.evidence_gaps):
            llm_first_attempted = True
            answer = self._compose_llm_business_answer(
                question,
                plan,
                run_result,
                rule_context,
                merchant,
                personalization_context,
                mandatory_skeleton=mandatory_skeleton,
            )
            if answer:
                hybrid_ranking_analysis = bool(ranking_answer and question_requests_diagnosis(question))
                if hybrid_ranking_analysis:
                    answer = merge_deterministic_ranking_with_llm_answer(ranking_answer, answer)
                coverage_answer = answer_coverage_partial_answer(question, plan, run_result)
                if coverage_answer and not answer_acknowledges_incomplete_evidence(answer):
                    answer = coverage_answer
                answer = ensure_required_field_answer_coverage(answer, plan, run_result)
                return self._compose_final_answer(
                    answer,
                    mandatory_skeleton,
                    question,
                    plan,
                    run_result,
                    bundle,
                    effective_rule_context,
                    merchant,
                    personalization_context,
                )
        structured_answer = ""
        trusted_structured = False
        if ranking_answer and (
            deterministic_ranking_preferred_before_llm(question)
            and (not question_requests_diagnosis(question) or not (allow_llm and self.llm.configured))
        ):
            structured_answer = ranking_answer
        elif deterministic_single_semantic_metric_answer(plan):
            # Keep the factual metric value/trend deterministic.  LLM prose is
            # never allowed to replace the one metric contract or its values.
            structured_answer = deterministic_structured_answer(question, plan, run_result)
            trusted_structured = trusted_single_metric_verified_answer(plan, run_result)
        elif not question_requests_diagnosis(question):
            # Verified data lookups, multi-metric trends and detail queries do
            # not need generative prose in the response hot path.  Rendering
            # from task-bound evidence is both faster and safer: an LLM cannot
            # replace a value that was already resolved by the semantic layer.
            structured_answer = deterministic_structured_answer(question, plan, run_result)
        elif not (allow_llm and self.llm.configured):
            structured_answer = deterministic_structured_answer(question, plan, run_result)
        if structured_answer:
            structured_answer = ensure_required_field_answer_coverage(structured_answer, plan, run_result)
            return self._compose_final_answer(
                structured_answer,
                mandatory_skeleton,
                question,
                plan,
                run_result,
                bundle,
                effective_rule_context,
                merchant,
                personalization_context,
                trusted_structured=trusted_structured,
            )
        if not llm_first_attempted and allow_llm and self.llm.configured and (bundle.rows or run_result.evidence_gaps):
            answer = self._compose_llm_business_answer(
                question,
                plan,
                run_result,
                rule_context,
                merchant,
                personalization_context,
                mandatory_skeleton=mandatory_skeleton,
            )
            if answer:
                hybrid_ranking_analysis = bool(ranking_answer and question_requests_diagnosis(question))
                if hybrid_ranking_analysis:
                    answer = merge_deterministic_ranking_with_llm_answer(ranking_answer, answer)
                coverage_answer = answer_coverage_partial_answer(question, plan, run_result)
                if coverage_answer and not answer_acknowledges_incomplete_evidence(answer):
                    answer = coverage_answer
                answer = ensure_required_field_answer_coverage(answer, plan, run_result)
                return self._compose_final_answer(
                    answer,
                    mandatory_skeleton,
                    question,
                    plan,
                    run_result,
                    bundle,
                    effective_rule_context,
                    merchant,
                    personalization_context,
                )
        fallback_answer = ensure_required_field_answer_coverage(
            mandatory_skeleton or self._fallback_data_answer(question, plan, bundle, run_result),
            plan,
            run_result,
        )
        return self._compose_final_answer(
            fallback_answer,
            mandatory_skeleton,
            question,
            plan,
            run_result,
            bundle,
            effective_rule_context,
            merchant,
            personalization_context,
        )

    def summarize_analysis(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
        allow_skill: bool = True,
    ) -> str:
        self.last_analysis_skill_trace = {"skillName": "", "matchStatus": "pending"}
        if not run_result or not run_result.merged_query_bundle.rows:
            return ""
        analysis_requested = analysis_summary_required(plan)
        declared_skill = allow_skill and answer_skill_required(plan, run_result, bool(rule_context))
        if not analysis_requested and not declared_skill:
            return ""
        skill_name = self.propose_answer_skill(question, plan, run_result, bool(rule_context)) if allow_skill else ""
        if not skill_name and not analysis_requested:
            return ""
        skill_answer = ""
        if skill_name:
            skill_answer = self.run_analysis_skill(
                question,
                plan,
                run_result,
                outputs_path,
                rule_context,
                skill_name=skill_name,
                merchant=merchant,
                personalization_context=personalization_context,
            )
        else:
            self.last_analysis_skill_trace["skillExecutionSkipped"] = True
        if skill_answer:
            return skill_answer
        deterministic = deterministic_analysis_summary(question, plan, run_result)
        if deterministic:
            self.last_analysis_skill_trace["deterministicAnalysisSummary"] = True
            return deterministic
        if not self.llm.configured:
            return ""
        analysis_prompt = self.prompt_assembler.render(
            "answer.analysis",
            sections={
                "analysis_policy": (
                    "只能基于 compact evidence 判断趋势、异常和原因假设；不能把缺失证据当事实。"
                    "输出给最终 AnswerAgent 继续润色，所以只保留业务判断和必要说明；"
                    "不要输出“分析结论/关键证据/限制/口径”等标题。"
                )
            },
        )
        prompt = json.dumps(
            answer_data_package(
                question,
                plan,
                run_result,
                rule_context,
                merchant=merchant,
                personalization_context=personalization_context,
            ),
            ensure_ascii=False,
            default=str,
        )
        self.last_prompt_chars = len(analysis_prompt.system_prompt) + len(prompt)
        self.last_analysis_skill_trace["llmFallbackAttempted"] = True
        answer = self.llm.chat(
            analysis_prompt.system_prompt,
            prompt,
            "",
            timeout_seconds=self.llm.settings.llm_analysis_timeout_seconds,
        )
        self.last_analysis_skill_trace["llmFallbackUsed"] = bool(answer)
        return answer

    def _compose_llm_business_answer(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        rule_context: str,
        merchant: MerchantInfo | None,
        personalization_context: Optional[Dict[str, Any]],
        analysis_summary: str = "",
        mandatory_skeleton: str = "",
    ) -> str:
        self.last_compose_llm_attempted = True
        package = answer_data_package(
            question,
            plan,
            run_result,
            rule_context,
            merchant=merchant,
            personalization_context=personalization_context,
        )
        if analysis_summary:
            package["analysisDraft"] = analysis_summary[:1800]
        if include_mandatory_skeleton_in_answer_prompt(question, plan, run_result, mandatory_skeleton):
            package["mandatoryAnswerSkeleton"] = mandatory_skeleton[:2400]
        prompt = json.dumps(package, ensure_ascii=False, default=str)
        answer_prompt = self.prompt_assembler.render(
            "answer.bi",
            sections={"answer_context_policy": answer_context_policy()},
        )
        self.last_prompt_chars += len(prompt) + len(answer_prompt.system_prompt)
        answer = self.llm.chat(
            answer_prompt.system_prompt,
            prompt,
            "",
            timeout_seconds=self.llm.settings.llm_answer_timeout_seconds,
        )
        if not answer:
            return ""
        self.last_compose_used_llm = True
        answer = sanitize_business_answer_text(answer, question, plan, run_result)
        answer = self._ensure_multi_trend_answer_coverage(answer, question, plan, run_result)
        if should_proactively_patch_metric_summary(answer, question, plan, run_result):
            answer = self._ensure_multi_metric_summary_coverage(answer, question, plan, run_result)
        answer = self._correct_metric_total_misread(answer, question, plan, run_result)
        answer = self._clean_summary_trend_misphrasing(answer, plan, run_result)
        return sanitize_business_answer_text(answer, question, plan, run_result)

    def _mandatory_answer_skeleton(self, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not run_result:
            return ""
        partial = (
            blocking_evidence_partial_answer(question, plan, run_result)
            or answer_coverage_partial_answer(question, plan, run_result)
            or gap_aware_partial_answer(question, plan, run_result)
        )
        if partial:
            return partial
        structured = deterministic_structured_answer(question, plan, run_result)
        if structured:
            return structured
        trend = multi_trend_metric_sentence(question, plan, run_result)
        if trend:
            return trend + chart_hint_sentence(plan, run_result)
        summary = summary_metric_sentence(question, plan, run_result)
        if summary:
            return summary + chart_hint_sentence(plan, run_result)
        if run_result and len(visible_successful_tasks(plan, run_result)) > 1:
            lines = ["当前已拿到多组可核对数据。"]
            table = business_summary_table(plan, run_result)
            if table:
                lines.append("")
                lines.append("结果摘要：")
                lines.append(table)
            evidence = task_evidence_sections(plan, run_result)
            if evidence:
                lines.append("")
                lines.append(evidence)
            return "\n".join(lines)
        return ""

    def _compose_final_answer(
        self,
        draft: str,
        mandatory_skeleton: str,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        bundle: QueryBundle,
        effective_rule_context: str,
        merchant: MerchantInfo | None,
        personalization_context: Optional[Dict[str, Any]],
        append_advice: bool = True,
        trusted_structured: bool = False,
    ) -> str:
        answer = sanitize_business_answer_text(draft or mandatory_skeleton, question, plan, run_result)
        if should_apply_mandatory_skeleton_to_draft(answer, mandatory_skeleton, question, plan, run_result):
            answer = self._ensure_mandatory_answer_skeleton(answer, mandatory_skeleton, question, plan, run_result)
        answer = ensure_required_field_answer_coverage(answer, plan, run_result)
        if append_advice:
            answer = self.append_business_advice(
                answer,
                plan.intents,
                bundle,
                question=question,
                plan=plan,
                run_result=run_result,
                merchant=merchant,
                personalization_context=personalization_context,
                allow_llm=False,
            )
        answer = self._append_rule_evidence(answer, question, effective_rule_context)
        answer = self._append_lightweight_metric_disclosure(answer, question, plan, run_result)
        skeleton_fallback = self._append_lightweight_metric_disclosure(
            self._append_rule_evidence(mandatory_skeleton, question, effective_rule_context),
            question,
            plan,
            run_result,
        )
        return self._finalize_answer(answer, question, plan, run_result, fallback_answer=skeleton_fallback)

    def _ensure_mandatory_answer_skeleton(
        self,
        answer: str,
        skeleton: str,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
    ) -> str:
        if not skeleton:
            return answer
        if not answer:
            return skeleton
        answer = self._ensure_multi_trend_answer_coverage(answer, question, plan, run_result)
        answer = self._ensure_multi_metric_summary_coverage(answer, question, plan, run_result)
        coverage = answer_requirement_coverage(question, plan, run_result)
        if coverage.get("shouldBlockDirectAnswer") and not answer_acknowledges_incomplete_evidence(answer):
            metric_facts = answer_metric_facts(question, plan, run_result)
            if metric_facts and all(answer_contains_metric_fact_value(answer, fact) for fact in metric_facts):
                return skeleton.rstrip() + "\n\n" + answer.strip()
            return skeleton
        if self._mandatory_skeleton_is_covered(answer, skeleton, plan, run_result):
            return answer
        return skeleton.rstrip() + "\n\n" + answer.strip()

    def _mandatory_skeleton_is_covered(
        self,
        answer: str,
        skeleton: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
    ) -> bool:
        if not answer or not skeleton:
            return False
        compact_answer = re.sub(r"\s+", "", answer)
        compact_skeleton = re.sub(r"\s+", "", skeleton)
        if compact_skeleton and compact_skeleton in compact_answer:
            return True
        if run_result:
            for item in summary_metric_values(plan, run_result):
                if not answer_contains_summary_metric_value(answer, item):
                    return False
            intent_map = intent_by_task_id(plan)
            trend_tasks = []
            for task in visible_successful_tasks(plan, run_result):
                intent = intent_map.get(task.task_id)
                if answer_result_role(intent) != "trend_context" or not task.query_bundle.rows:
                    continue
                trend_rows, complete = query_bundle_rows_for_trend(task.query_bundle)
                points = metric_series_rows_for_intent(plan, intent, trend_rows) if intent and complete else []
                if len(points) >= 2:
                    trend_tasks.append((intent, points))
            for intent, points in trend_tasks:
                resolution = intent.metric_resolution or {}
                metric_key = str(resolution.get("metricKey") or intent.metric_name or "")
                label = str(resolution.get("displayName") or friendly_column_label(plan, metric_key) or metric_key)
                first_value = format_metric_value_for_answer(points[0].get("value"), metric_key, label, resolution)
                last_value = format_metric_value_for_answer(points[-1].get("value"), metric_key, label, resolution)
                if label not in skeleton or (first_value not in skeleton and last_value not in skeleton):
                    continue
                if label and label not in answer:
                    return False
                if first_value not in answer and last_value not in answer:
                    return False
            if trend_tasks:
                return True
        skeleton_numbers = [item for item in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", skeleton) if item not in {"0", "1"}]
        return all(item in answer for item in skeleton_numbers[:8])

    def propose_answer_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        has_rule_context: bool = False,
    ) -> str:
        candidates = answer_skill_headers(self.llm.settings.resources_root / "runtime" / "agent_skills")
        declared = select_answer_skill(plan, run_result, has_rule_context, skill_headers=candidates)
        trace: Dict[str, Any] = {
            "lifecycle": ["match", "confirm", "isolated_execute", "progress", "output"],
            "matchMode": self.llm.settings.answer_skill_match_mode,
            "candidateSkills": [item.get("name") for item in candidates],
            "declaredSkill": declared,
            "skillName": declared,
        }
        self.last_analysis_skill_trace = trace
        if skill_route_explicit_no_match(plan.question_understanding or {}):
            trace.update(
                {
                    "matchedBy": "semantic_explicit_no_match",
                    "matchStatus": "explicit_no_match",
                    "skillName": "",
                    "fallbackSuppressed": True,
                    "fallbackSuppressedReason": "AUTHORITATIVE_SEMANTIC_NO_MATCH",
                }
            )
            self.last_analysis_skill_trace = trace
            return ""
        if declared:
            trace.update(
                {
                    "matchedBy": "structured_plan_declaration",
                    "matchStatus": "matched",
                    "skillName": declared,
                }
            )
            self.last_analysis_skill_trace = trace
            return declared
        if not candidates or self.llm.settings.answer_skill_match_mode == "off":
            trace["matchedBy"] = "runtime_skill_headers"
            trace["matchStatus"] = "no_match"
            trace["skillName"] = ""
            self.last_analysis_skill_trace = trace
            return ""
        if not self.llm.configured:
            trace["matchedBy"] = "runtime_skill_headers_no_llm"
            trace["matchStatus"] = "no_match"
            trace["skillName"] = ""
            self.last_analysis_skill_trace = trace
            return ""
        prompt_payload = {
            "question": question,
            "questionUnderstanding": plan.question_understanding,
            "plannedEvidence": [
                {
                    "taskId": intent.plan_task_id,
                    "category": category_display(intent.category),
                    "answerMode": str(intent.answer_mode),
                    "metric": intent.metric_name,
                    "table": intent.preferred_table,
                    "resolution": intent.metric_resolution,
                }
                for intent in plan.intents[:10]
            ],
            "hasRuleContext": has_rule_context,
            "evidenceRows": run_result.merged_query_bundle.rows[:5] if run_result else [],
            "evidenceGaps": [gap.model_dump(by_alias=True) for gap in (run_result.evidence_gaps if run_result else [])[:5]],
            "skills": candidates,
        }
        system = (
            "You are an Answer Skill matcher. Choose at most one skill from the provided skill headers. "
            "Do not invent skill names. Return JSON only: {\"skillName\":\"...\", \"confidence\":0-1, \"reason\":\"...\"}. "
            "Return empty skillName when no skill should run. An empty skillName is an authoritative no-match decision."
        )
        raw = self.llm.chat(system, json.dumps(prompt_payload, ensure_ascii=False, default=str), "", timeout_seconds=8)
        trace["matchedBy"] = "llm_skill_header_match"
        trace["llmRaw"] = raw[:800] if raw else ""
        payload = parse_skill_match_payload(raw)
        allowed = {str(item.get("name") or "") for item in candidates}
        selected = skill_match_payload_name(payload)
        trace["llmMatchStatus"] = skill_match_response_status(raw, payload, selected)
        trace["confidence"] = payload.get("confidence")
        trace["reason"] = payload.get("reason")
        if skill_match_payload_explicit_no_match(payload):
            trace.update(
                {
                    "matchedBy": "llm_explicit_no_match",
                    "matchStatus": "explicit_no_match",
                    "skillName": "",
                }
            )
            self.last_analysis_skill_trace = trace
            return ""
        if selected and selected not in allowed:
            selected = ""
            trace["matchWarning"] = "LLM_SELECTED_UNKNOWN_SKILL"
            trace["llmMatchStatus"] = "unknown_skill"
        trace["matchStatus"] = "matched" if selected else "no_match"
        trace["skillName"] = selected
        self.last_analysis_skill_trace = trace
        return selected

    def run_analysis_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        skill_name: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        selected_skill = str(skill_name or "").strip()
        if not selected_skill:
            trace = dict(self.last_analysis_skill_trace or {})
            trace.update(
                {
                    "activated": False,
                    "skillExecutionSkipped": True,
                    "lifecycleStage": "skipped",
                    "matchStatus": trace.get("matchStatus") or "no_match",
                    "progress": ["matched", "no_match", "skipped"],
                }
            )
            self.last_analysis_skill_trace = trace
            return ""

        headers = answer_skill_headers(self.llm.settings.resources_root / "runtime" / "agent_skills")
        metadata = next((item for item in headers if str(item.get("name") or "") == selected_skill), None)
        if metadata is None:
            self.last_analysis_skill_trace = {
                **dict(self.last_analysis_skill_trace or {}),
                "skillName": selected_skill,
                "activated": False,
                "lifecycleStage": "failed",
                "error": "skill is not present in runtime resources",
                "progress": ["matched", "failed:unknown runtime skill"],
            }
            return ""

        from merchant_ai.services.skill_worker import SkillWorkerExecutor

        result = SkillWorkerExecutor(self.llm).execute_answer_skill(
            question,
            plan,
            run_result,
            outputs_path,
            rule_context,
            selected_skill,
            merchant=merchant,
            personalization_context=personalization_context,
            initial_trace=dict(self.last_analysis_skill_trace or {}),
        )
        self.last_analysis_skill_trace = result.trace
        return result.answer

    def run_parallel_analysis_skills(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        skill_names: List[str],
        outputs_path: str = "",
        rule_context: str = "",
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        from merchant_ai.services.skill_worker import SkillWorkerExecutor

        executor = SkillWorkerExecutor(self.llm)
        results = executor.execute_answer_skills(
            question,
            plan,
            run_result,
            skill_names,
            outputs_path,
            rule_context,
            merchant=merchant,
            personalization_context=personalization_context,
            initial_trace=dict(self.last_analysis_skill_trace or {}),
        )
        successful = [result for result in results if result.answer and not result.trace.get("error")]
        sections = []
        for result in successful:
            title = str(result.trace.get("skillName") or "analysis_skill")
            sections.append("### %s\n%s" % (title, result.answer.strip()))
        summary = "\n\n".join(sections)
        requires_confirmation = bool(
            self.last_analysis_skill_trace.get("requiresConfirmation", self.llm.settings.skill_confirmation_required)
        )
        confirmed = bool(self.last_analysis_skill_trace.get("confirmed", not requires_confirmation))
        self.last_analysis_skill_trace = {
            "skillName": "parallel_skill_batch",
            "skillNames": [str(name) for name in skill_names or [] if str(name).strip()],
            "activated": bool(results),
            "executionMode": "parallel_isolated_skill_workers",
            "workerType": "SKILL_WORKER_BATCH",
            "subAgentType": "SKILL_WORKER_BATCH",
            "isolatedExecution": True,
            "parallelExecution": True,
            "requiresConfirmation": requires_confirmation,
            "confirmed": confirmed,
            "confirmationStatus": "confirmed" if requires_confirmation else "not_required",
            "lifecycleStage": "completed" if successful else "failed",
            "progress": [
                "matched",
                "confirmed" if requires_confirmation else "ready",
                "parallel_isolated_execute",
                "progress_synced",
                "completed" if successful else "failed",
            ],
            "skillBatchResults": [result.trace for result in results],
            "completedCount": len(successful),
            "failedCount": len(results) - len(successful),
            "summaryChars": len(summary),
            "reuseCandidate": any(bool(result.trace.get("reuseCandidate")) for result in results),
        }
        if not successful:
            self.last_analysis_skill_trace["error"] = "all parallel skill workers failed"
        return summary

    def append_business_advice(
        self,
        answer: str,
        intents: List[QuestionIntent],
        bundle: QueryBundle,
        question: str = "",
        plan: QueryPlan | None = None,
        run_result: AgentRunResult | None = None,
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
        allow_llm: bool = False,
    ) -> str:
        if not answer:
            answer = "当前没有足够数据形成结论。"
        answer = normalize_inline_business_advice(answer.rstrip())
        if has_business_advice_section(answer):
            return answer
        if allow_llm and self.llm.configured and plan is not None:
            items = self._llm_business_advice(question, answer, plan, run_result, merchant, personalization_context)
            if items:
                return answer.rstrip() + "\n\n建议：\n" + "\n".join("- %s" % item for item in items[:2])
        return answer.rstrip()

    def _llm_business_advice(
        self,
        question: str,
        answer: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        merchant: MerchantInfo | None,
        personalization_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        package = answer_data_package(
            question,
            plan,
            run_result,
            merchant=merchant,
            personalization_context=personalization_context,
        )
        payload = {
            "question": question,
            "answerDraft": answer[:1200],
            "businessContext": package.get("businessContext", {}),
            "currentDataSignals": (package.get("businessContext") or {}).get("currentDataSignals", []),
            "evidenceGaps": package.get("evidenceGaps", []),
        }
        system = (
            "你是商家经营助手的建议生成器。只基于输入中的商家画像、长期记忆/近期关注和本轮已验证数据给建议。"
            "输出 JSON：{\"suggestions\":[\"建议1\",\"建议2\"]}。最多 2 条，每条不超过 45 个中文字符。"
            "不要建议用户继续追问，不要暴露表名、字段名、SQL 或内部证据口径。"
        )
        raw = self.llm.chat(
            system,
            json.dumps(payload, ensure_ascii=False, default=str),
            "",
            timeout_seconds=min(int(self.llm.settings.llm_answer_timeout_seconds or 12), 8),
        )
        self.last_prompt_chars += len(system) + len(json.dumps(payload, ensure_ascii=False, default=str))
        return parse_llm_suggestions(raw)

    def _correct_metric_total_misread(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        if not re.search(r"(不能|无法|不(?:能|可)直接|不能准确).{0,24}(确认|判断|得到)", answer):
            return answer
        sentence = multi_summary_metric_sentence(question, plan, run_result)
        if sentence:
            return sentence + chart_hint_sentence(plan, run_result)
        summary = primary_summary_metric_value(plan, run_result)
        if not summary:
            return answer
        label = summary.get("label") or "指标"
        value = summary.get("value")
        time_phrase = extract_question_time_phrase(question)
        prefix = "%s，" % time_phrase if time_phrase else "当前查询范围内，"
        trend = primary_trend_points(plan, run_result, summary.get("metricKey") or "")
        lines = ["%s%s为 %s。" % (prefix, label, format_cell(value))]
        if trend:
            lines.append("")
            lines.append("按日趋势明细已同步到图表区域。")
        return "\n".join(lines)

    def _ensure_multi_metric_summary_coverage(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        summaries = summary_metric_values(plan, run_result)
        if len(summaries) <= 1:
            return answer
        missing = [
            item for item in summaries
            if not answer_contains_summary_metric_value(answer, item)
        ]
        if not missing:
            return answer
        sentence = multi_summary_metric_sentence(question, plan, run_result)
        if not sentence:
            return answer
        stripped = answer.strip()
        if re.match(r"^分析结论[:：]\s*$", stripped.splitlines()[0].strip() if stripped.splitlines() else ""):
            return sentence + "\n\n" + stripped
        return sentence + "\n\n" + stripped

    def _ensure_multi_trend_answer_coverage(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        trend_tasks = []
        intent_map = intent_by_task_id(plan)
        for task in visible_successful_tasks(plan, run_result):
            intent = intent_map.get(task.task_id)
            if answer_result_role(intent) != "trend_context" or not task.query_bundle.rows:
                continue
            trend_rows, complete = query_bundle_rows_for_trend(task.query_bundle)
            points = metric_series_rows_for_intent(plan, intent, trend_rows) if intent and complete else []
            if len(points) >= 2:
                trend_tasks.append((intent, points))
        if len(trend_tasks) <= 1:
            return answer
        missing = []
        for intent, points in trend_tasks:
            resolution = intent.metric_resolution or {}
            metric_key = str(resolution.get("metricKey") or intent.metric_name or "")
            label = str(resolution.get("displayName") or friendly_column_label(plan, metric_key) or metric_key)
            first_value = format_metric_value_for_answer(points[0].get("value"), metric_key, label, resolution)
            last_value = format_metric_value_for_answer(points[-1].get("value"), metric_key, label, resolution)
            if label not in answer or (first_value not in answer and last_value not in answer):
                missing.append(metric_key or label)
        if not missing:
            return answer
        sentence = multi_trend_metric_sentence(question, plan, run_result)
        if not sentence:
            return answer
        stripped = answer.strip()
        if sentence in stripped:
            return answer
        return sentence + "\n\n" + stripped

    def _clean_summary_trend_misphrasing(self, answer: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        summary = primary_summary_metric_value(plan, run_result)
        if not summary:
            return answer
        trend = primary_trend_points(plan, run_result, summary.get("metricKey") or "")
        if not trend:
            return answer
        cleaned_lines: List[str] = []
        for line in answer.splitlines():
            text = line.strip()
            if re.search(r"(其余|其他).{0,8}(日期|天).{0,16}(没有|未|暂无).{0,12}(看到|分日|明细|数据)", text):
                continue
            if re.search(r"(只|仅).{0,4}覆盖.{0,12}\d+.{0,4}(个)?(自然日|天)", text):
                continue
            if re.search(r"未带日期.{0,16}(记录|数据)", text):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or answer

    def _append_lightweight_metric_disclosure(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or not run_result.merged_query_bundle.rows:
            return answer
        if not question_asks_metric_disclosure(question):
            return answer
        if question_asks_metric_reconciliation(question):
            return answer
        note = lightweight_metric_disclosure_note(question, plan, run_result)
        if not note or note in answer or "统计说明：" in answer:
            return answer
        return answer.rstrip() + "\n\n" + note

    def _metric_reconciliation_answer(self, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not run_result:
            return ""
        lines: List[str] = []
        summary = multi_summary_metric_sentence(question, plan, run_result)
        if not summary:
            primary = primary_summary_metric_value(plan, run_result)
            if primary:
                label = primary.get("label") or "指标"
                value = format_metric_value_for_answer(
                    primary.get("value"),
                    primary.get("metricKey") or "",
                    str(label),
                    primary.get("displayMetadata"),
                )
                time_phrase = extract_question_time_phrase(question)
                prefix = "%s，" % time_phrase if time_phrase else "本次查询范围内，"
                summary = "%s%s为 %s。" % (prefix, label, value)
        lines.append("口径对账：我先按本次查询口径复核。")
        if summary:
            lines.append(summary)
        note = lightweight_metric_disclosure_note(question, plan, run_result)
        if note:
            lines.append(note)
        lines.append("如果该数值和其他看板不一致，请按当前已发布语义契约逐项核对：")
        for disclosure in metric_disclosures(plan, run_result.verified_evidence)[:3]:
            name = str(disclosure.get("displayName") or disclosure.get("metricKey") or disclosure.get("metric") or "指标")
            description = str(disclosure.get("description") or disclosure.get("fieldWarning") or "")
            detail = description or lightweight_metric_description(disclosure, include_formula=True)
            lines.append("- %s：%s" % (name, detail or "以已发布语义层定义为准"))
        lines.extend(
            [
                "- 时间口径：核对起止时间、自然日边界和时区是否一致。",
                "- 统计对象：核对纳入范围、状态过滤和排除条件是否一致。",
                "- 聚合方式：核对求和、去重、派生计算及调整项是否一致。",
                "- 数据粒度：核对分组维度与汇总层级是否一致。",
                "- 数据更新：核对数据分区、刷新时间和实时/离线来源是否一致。",
                "- 查询范围：核对过滤条件和租户范围是否一致。",
            ]
        )
        lines.append("建议提供对方看板的语义指标 ID 与查询范围，再按同一契约复算。")
        return "\n".join(lines)

    def _apply_answer_guard(self, answer: str, run_result: AgentRunResult | None) -> str:
        verified = run_result.verified_evidence if run_result else None
        if not verified or not verified.answer_guard_required:
            return answer
        additions: List[str] = []
        for disclosure in verified.required_disclosures:
            disclosure_text = str(disclosure or "").strip()
            if disclosure_text and disclosure_text not in answer:
                additions.append(disclosure_text)
        for gap in verified.gaps:
            if not gap.disclosure_required:
                continue
            detail = gap.answer_instruction or gap.reason
            if not detail:
                continue
            if gap.code and gap.code in answer:
                continue
            if detail in answer:
                continue
            additions.append("%s：%s" % (gap.code or "EVIDENCE_GAP", detail))
        additions = dedupe_strings(additions)
        if not additions:
            return answer
        merchant_notes = [merchant_facing_gap_note(item) for item in additions]
        merchant_notes = [item for item in dedupe_strings(merchant_notes) if item and item not in answer]
        if not merchant_notes:
            return answer
        return answer.rstrip() + "\n\n说明：\n" + "\n".join("- %s" % item for item in merchant_notes[:3])

    def _finalize_answer(
        self,
        answer: str,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        fallback_answer: str = "",
    ) -> str:
        if run_result is not None:
            run_result.verified_facts = build_verified_facts(plan, run_result)
        guarded = self._apply_answer_guard(answer or fallback_answer, run_result)
        verification = lightweight_answer_contract_verification(question, plan, run_result, guarded)
        if verification is None:
            verification = AnswerClaimVerifier().verify(question, plan, run_result, guarded)
        if verification is None or verification.passed:
            self._record_lightweight_answer_verification(run_result, verification)
            return guarded
        failure_reason = lightweight_answer_failure_reason(verification)
        deterministic_fallback = deterministic_structured_answer(question, plan, run_result)
        fact_fallback = verified_metric_facts_fallback_answer(question, plan, run_result)
        if fallback_answer and answer_requirement_coverage(question, plan, run_result).get("shouldBlockDirectAnswer"):
            deterministic_fallback = self._ensure_mandatory_answer_skeleton(
                deterministic_fallback, fallback_answer, question, plan, run_result
            )
            fact_fallback = self._ensure_mandatory_answer_skeleton(
                fact_fallback, fallback_answer, question, plan, run_result
            )
        fallback_candidates = [fallback_answer, deterministic_fallback, fact_fallback]
        seen: set[str] = {guarded}
        last_verification: AnswerClaimVerification | None = None
        for candidate in fallback_candidates:
            candidate = self._apply_answer_guard(str(candidate or "").strip(), run_result)
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            candidate_verification = lightweight_answer_contract_verification(question, plan, run_result, candidate)
            if candidate_verification is None:
                candidate_verification = AnswerClaimVerifier().verify(question, plan, run_result, candidate)
            candidate_verification = candidate_verification.model_copy(
                update={
                    "fallback_used": True,
                    "fallback_reason": failure_reason,
                    "rejected_claims": verification.unsupported_claims,
                }
            )
            last_verification = candidate_verification
            if candidate_verification.passed:
                self._record_lightweight_answer_verification(run_result, candidate_verification)
                return candidate

        # Never return a candidate that the same contract has just rejected.
        # When even the fact-only rendering cannot pass, suppress all numeric
        # claims and preserve the failed verification in the runtime trace.
        safe_answer = "已拿到查询结果，但回答完整性校验未通过；为避免展示未经校验的数值，请稍后重试。"
        safe_verification = lightweight_answer_contract_verification(question, plan, run_result, safe_answer)
        if safe_verification is None:
            safe_verification = AnswerClaimVerifier().verify(question, plan, run_result, safe_answer)
        final_verification = safe_verification or last_verification or verification
        final_verification = final_verification.model_copy(
            update={
                "fallback_used": True,
                "fallback_reason": "%s;safe_fact_fallback_failed" % failure_reason,
                "rejected_claims": verification.unsupported_claims,
            }
        )
        self._record_lightweight_answer_verification(run_result, final_verification)
        return safe_answer

    def _record_lightweight_answer_verification(
        self,
        run_result: AgentRunResult | None,
        verification: AnswerClaimVerification | None,
    ) -> None:
        if verification is None:
            self.last_answer_claim_trace = {}
            return
        if run_result is not None:
            run_result.answer_claim_verification = verification
        self.last_answer_claim_trace = verification.model_dump(by_alias=True)

    def contextual_suggestions(
        self,
        question: str,
        intents: List[QuestionIntent],
        run_result: AgentRunResult | None = None,
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        return contextual_business_suggestions(
            question,
            intents,
            run_result=run_result,
            merchant=merchant,
            personalization_context=personalization_context,
        )

    def merchant_experience(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        merchant: MerchantInfo | None = None,
        sections: Optional[List[ChatDataSection]] = None,
        suggestions: Optional[List[str]] = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_merchant_experience_package(
            question,
            plan,
            run_result,
            merchant=merchant,
            sections=sections or self.build_sections(plan, run_result or AgentRunResult()),
            suggestions=suggestions or self.contextual_suggestions(
                question,
                plan.intents,
                run_result=run_result,
                merchant=merchant,
                personalization_context=personalization_context,
            ),
            personalization_context=personalization_context,
        )

    def build_sections(self, plan: QueryPlan, run_result: AgentRunResult) -> List[ChatDataSection]:
        sections: List[ChatDataSection] = []
        if not run_result:
            return sections
        intent_map = intent_by_task_id(plan)
        sources: List[Any] = []
        if run_result.task_results:
            sources = [(intent_map.get(item.task_id), item.query_bundle) for item in visible_successful_tasks(plan, run_result)]
        else:
            sources = [
                (plan.intents[index] if index < len(plan.intents) else None, bundle)
                for index, bundle in enumerate(run_result.query_bundles)
            ]
        for intent, bundle in sources:
            if bundle.failed or not bundle.rows:
                continue
            title = "查询结果"
            data_rows = bundle.rows[:50]
            if intent:
                title = section_title_for_intent(plan, intent, title)
                if answer_result_role(intent) == "trend_context":
                    trend_rows, _ = query_bundle_rows_for_trend(bundle)
                    series_rows = metric_series_rows_for_intent(plan, intent, trend_rows)
                    if series_rows:
                        data_rows = series_rows
            sections.append(
                ChatDataSection(
                    title=title,
                    result_role=answer_result_role(intent),
                    doris_tables=bundle.tables,
                    data_rows=data_rows,
                    original_row_count=bundle.effective_row_count(),
                    result_summary=bundle.summary,
                )
            )
        return sections

    def _compose_rule_answer(self, question: str, knowledge_context: str) -> str:
        if self.llm.configured and knowledge_context:
            rule_prompt = self.prompt_assembler.render(
                "answer.rule",
                sections={"rule_context_policy": "只基于给定 knowledge 片段回答，缺知识时明确要求运营补充。"},
            )
            answer = self.llm.chat(
                rule_prompt.system_prompt,
                json.dumps({"question": question, "knowledge": knowledge_context[:12000]}, ensure_ascii=False, default=str),
                "",
                timeout_seconds=self.llm.settings.llm_answer_timeout_seconds,
            )
            if answer:
                return answer
        if knowledge_context:
            return "我找到了相关规则片段，但当前未配置 LLM，先给出可核对的知识来源：\n\n%s" % knowledge_context[:1200]
        return "当前没有命中足够的规则知识，请补充具体规则场景或让运营完善知识库。"

    def _append_rule_evidence(self, answer: str, question: str, rule_context: str) -> str:
        evidence = compact_rule_evidence(question, rule_context)
        if not evidence:
            return answer
        if "规则依据" in answer:
            return answer
        lines = ["规则依据："]
        lines.extend("- %s" % item for item in evidence[:5])
        return answer.rstrip() + "\n\n" + "\n".join(lines)

    def _fallback_data_answer(self, question: str, plan: QueryPlan, bundle: QueryBundle, run_result: AgentRunResult | None = None) -> str:
        if bundle.failed:
            return "这次没有拿到可靠的数据结果，暂时不能给出具体数值。可以稍后重试，或缩小时间范围后再查。"
        if not bundle.rows:
            return "当前查询范围内没有查到符合条件的数据。"
        gap_answer = (
            blocking_evidence_partial_answer(question, plan, run_result)
            or answer_coverage_partial_answer(question, plan, run_result)
            or gap_aware_partial_answer(question, plan, run_result)
        )
        if gap_answer:
            return gap_answer
        friendly = merchant_friendly_data_answer(question, plan, bundle, run_result)
        if friendly:
            return friendly
        overview = generic_result_overview_sentence(question, plan, bundle.rows, run_result)
        lines = [overview or "当前查询范围内，已返回可核对的数据结果。"]
        gaps = run_result.evidence_gaps if run_result else []
        if gaps:
            lines.append("其中有部分关联信息暂时未补齐，结论需要结合下方明细谨慎判断。")
        multi_node_success = bool(run_result and len([item for item in run_result.task_results if not item.query_bundle.failed]) > 1)
        derived = run_result.verified_evidence.derived_evidence if run_result and run_result.verified_evidence else []
        if derived:
            formulas = ["%s=%s" % (item.get("metric"), item.get("formula")) for item in derived[:6] if item.get("metric") and item.get("formula")]
            if formulas:
                lines.append("")
                lines.append("计算说明：%s" % "；".join(formulas))
        if multi_node_success and run_result:
            summary_table = business_summary_table(plan, run_result)
            if summary_table:
                lines.append("")
                lines.append("结果摘要：")
                lines.append(summary_table)
            section = task_evidence_sections(plan, run_result)
            if section:
                lines.append("")
                lines.append(section)
        return "\n".join(lines)

    def _execution_failure_answer(self, run_result: AgentRunResult) -> str:
        failed = [item for item in run_result.task_results if item.query_bundle.failed]
        succeeded = [item for item in run_result.task_results if not item.query_bundle.failed]
        lines = ["本轮查询没有形成完整证据，不能把失败或未执行解释成业务为 0。"]
        for item in failed[:3]:
            error = item.query_bundle.error or item.summary
            lines.append("- 节点 %s 执行失败：%s" % (item.task_id, str(error)[:220]))
        if succeeded:
            lines.append("- 另有 %d 个节点执行完成，但由于依赖证据不完整，只能作为部分证据。" % len(succeeded))
        if run_result.partial_answer_reason:
            lines.append("- 证据门禁：%s" % run_result.partial_answer_reason)
        return "\n".join(lines)

    def _no_execution_answer(self, plan: QueryPlan) -> str:
        trace_lower = "，".join(plan.agent_trace).lower() if plan.agent_trace else ""
        if "timeout" in trace_lower or "provider_error" in trace_lower or "planner_provider_error" in trace_lower:
            return "这次没有拿到可验证的数据结果，可能是规划步骤超时或模型服务暂时异常；不能把它解释成业务为 0。建议稍后重试，或把问题拆成更明确的指标和时间范围。"
        if "json_parse_error" in trace_lower or "planner_json_parse_error" in trace_lower:
            return "这次没有形成可执行的数据查询计划，因此没有可验证结果；不能把它解释成业务为 0。建议换一种更明确的问法后重试。"
        if any("planner.no_llm_configured" in item for item in plan.agent_trace):
            return "当前模型服务未配置完成，所以没有拿到可验证的数据结果；不能把它解释成业务为 0。"
        if any("planner.no_valid_llm_understanding" in item for item in plan.agent_trace):
            return "这次没有稳定理解出要查询的指标、范围或条件，因此没有可验证结果；不能把它解释成业务为 0。"
        if not plan.intents:
            return "这次没有形成可执行的数据查询计划，因此没有可验证结果；不能把它解释成业务为 0。"
        return "这次查询没有进入实际取数阶段，因此没有可验证结果；不能把它解释成业务为 0。"


def merchant_friendly_data_answer(question: str, plan: QueryPlan, bundle: QueryBundle, run_result: AgentRunResult | None = None) -> str:
    rows = bundle.rows or []
    if not rows:
        return ""
    structured = deterministic_structured_answer(question, plan, run_result, fallback_rows=rows)
    if structured:
        return structured
    prefix = answer_time_prefix(question)
    row = rows[0]
    metric_column = primary_answer_metric_column(plan, row)
    if metric_column:
        metric_label = friendly_column_label(plan, metric_column)
        metric_value = format_answer_cell(
            metric_column,
            row.get(metric_column),
            metric_label,
            answer_column_display_contracts(plan).get(metric_column),
        )
        entity_column = primary_entity_column(plan, row)
        if is_ranking_plan(plan) and entity_column:
            entity_label = friendly_column_label(plan, entity_column)
            entity_value = format_cell(row.get(entity_column))
            return "%s%s %s 的%s为 %s。" % (prefix, entity_label, entity_value, metric_label, metric_value)
        if len(rows) == 1:
            if entity_column:
                entity_label = friendly_column_label(plan, entity_column)
                entity_value = format_cell(row.get(entity_column))
                return "%s%s %s 的%s为 %s。" % (prefix, entity_label, entity_value, metric_label, metric_value)
            return "%s%s为 %s。" % (prefix, metric_label, metric_value)
        sample = row_sample_sentence(question, plan, rows)
        if sample:
            return sample
        return "%s%s已返回 %d 条结果。" % (prefix, metric_label, len(rows))
    if len(rows) == 1:
        detail = single_row_detail_sentence(question, plan, row)
        if detail:
            return detail
        return "%s已返回 1 条匹配结果。" % prefix
    overview = generic_result_overview_sentence(question, plan, rows, run_result)
    return overview or "%s已返回 %d 条匹配结果。" % (prefix, len(rows))


def gap_aware_partial_answer(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result or not run_result.evidence_gaps:
        return ""
    blocking = [gap for gap in run_result.evidence_gaps if str(getattr(gap, "severity", "") or "") == "blocking"]
    if not blocking:
        return ""
    gap_text = " ".join("%s %s %s" % (getattr(gap, "code", ""), getattr(gap, "evidence", ""), getattr(gap, "reason", "")) for gap in blocking)
    asks_ratio = bool(question_requested_ratio_phrases(question))
    derived_gap = bool(re.search(r"(DERIVED|share|ratio|rate|占比|比例|率)", gap_text, flags=re.I))
    if not asks_ratio and not derived_gap:
        return ""
    lines = ["这题目前不能给出完整占比结论。"]
    missing_metrics = dedupe_strings(
        [
            friendly_column_label(plan, str(getattr(gap, "evidence", "") or ""))
            for gap in blocking
            if str(getattr(gap, "evidence", "") or "").strip()
        ]
    )
    if missing_metrics:
        lines.append("缺口主要在：%s。" % "、".join(missing_metrics[:3]))
    zero_rows = [
        gap for gap in blocking
        if str(getattr(gap, "code", "") or "") == "ZERO_ROWS" or "返回 0 行" in str(getattr(gap, "reason", "") or "")
    ]
    if zero_rows:
        zero_labels = dedupe_strings(
            [
                friendly_column_label(plan, str(getattr(gap, "evidence", "") or ""))
                for gap in zero_rows
                if str(getattr(gap, "evidence", "") or "").strip()
            ]
        )
        if zero_labels:
            lines.append("%s相关查询返回 0 行，不能直接当作占比为 0。" % "、".join(zero_labels[:3]))
    evidence_lines = partial_evidence_summary_lines(plan, run_result)
    if evidence_lines:
        lines.append("")
        lines.append("已拿到的证据：")
        lines.extend(re.sub(r"：返回\s*\d+\s*条。?$", "：已有返回结果。", item) for item in evidence_lines[:5])
    lines.append("")
    lines.append("建议先补齐上述缺失指标及其关联证据，再按已发布公式计算目标比例。")
    return "\n".join(lines)


def blocking_evidence_partial_answer(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    verified = getattr(run_result, "verified_evidence", None)
    blocking = list(getattr(verified, "blocking_gaps", []) or [])
    if not blocking:
        blocking = [
            gap for gap in (getattr(run_result, "evidence_gaps", []) or [])
            if str(getattr(gap, "severity", "") or "") == "blocking"
        ]
    if not blocking:
        return ""
    lines = ["这题目前不能给出完整结论。"]
    reason = str(getattr(verified, "partial_answer_reason", "") or getattr(run_result, "partial_answer_reason", "") or "").strip()
    if reason:
        lines.append("主要原因：%s。" % reason.rstrip("。"))
    else:
        reasons = dedupe_strings(
            [
                str(getattr(gap, "answer_instruction", "") or getattr(gap, "reason", "") or getattr(gap, "code", "") or "").strip()
                for gap in blocking
                if str(getattr(gap, "answer_instruction", "") or getattr(gap, "reason", "") or getattr(gap, "code", "") or "").strip()
            ]
        )
        if reasons:
            lines.append("主要缺口：%s。" % "；".join(reasons[:3]).rstrip("。"))
    evidence_lines = partial_evidence_summary_lines(plan, run_result)
    if evidence_lines:
        lines.append("")
        lines.append("已拿到的证据：")
        lines.extend(re.sub(r"：返回\s*\d+\s*条。?$", "：已有返回结果。", item) for item in evidence_lines[:5])
    lines.append("")
    lines.append("上面缺口补齐前，只能把本轮结果作为部分证据，不能当作完整业务判断。")
    return "\n".join(lines)


def answer_coverage_partial_answer(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    coverage = answer_requirement_coverage(question, plan, run_result)
    if not coverage.get("shouldBlockDirectAnswer"):
        return ""
    missing = coverage.get("missing") or []
    complete = coverage.get("complete") or []
    lines = ["这题目前不能直接给出完整结论。"]
    if missing:
        lines.append("缺少的关键结果：%s。" % "、".join(item["label"] for item in missing[:4]))
    if complete:
        lines.append("已覆盖的结果：%s。" % "、".join(item["label"] for item in complete[:4]))
    evidence_lines = partial_evidence_summary_lines(plan, run_result) if run_result else []
    if evidence_lines:
        lines.append("")
        lines.append("已拿到的证据：")
        lines.extend(evidence_lines[:5])
    lines.append("")
    lines.append("建议补齐缺失结果后再做占比、排序或风险判断，避免把部分明细误当成最终结论。")
    return "\n".join(lines)


def answer_acknowledges_incomplete_evidence(answer: str) -> bool:
    return bool(re.search(r"(不能|无法|暂不能|缺少|未完整|未补齐|证据不足|不完整|不能直接)", str(answer or "")))


def answer_requirement_coverage(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> Dict[str, Any]:
    requirements = answer_requirements(question, plan)
    available = available_answer_columns(run_result)
    complete: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []
    for requirement in requirements:
        aliases = {str(item).strip() for item in requirement.get("aliases", []) if str(item or "").strip()}
        if aliases & available:
            complete.append(requirement)
        else:
            missing.append(requirement)
    if missing and complete:
        complete_labels = {normalized_metric_phrase(item.get("label")) for item in complete if normalized_metric_phrase(item.get("label"))}
        complete_keys = {normalized_metric_phrase(item.get("key")) for item in complete if normalized_metric_phrase(item.get("key"))}
        still_missing: List[Dict[str, str]] = []
        for requirement in missing:
            label = normalized_metric_phrase(requirement.get("label"))
            key = normalized_metric_phrase(requirement.get("key"))
            if (label and label in complete_labels) or (key and key in complete_labels) or (label and label in complete_keys):
                complete.append(requirement)
                continue
            still_missing.append(requirement)
        missing = still_missing
    asks_ratio = bool(question_requested_ratio_phrases(question))
    has_ratio_result = any(answer_requirement_is_ratio(item) for item in complete)
    missing_ratio_result = any(answer_requirement_is_ratio(item) for item in missing)
    should_block = bool(asks_ratio and (not has_ratio_result or missing_ratio_result))
    if missing and question_requests_diagnosis(question):
        should_block = True
    return {"requirements": requirements, "complete": complete, "missing": missing, "shouldBlockDirectAnswer": should_block}


def ensure_required_field_answer_coverage(
    answer: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
) -> str:
    """Append verified, explicitly requested semantic fields omitted by prose."""
    if not answer or not run_result:
        return answer
    intent_map = intent_by_task_id(plan)
    task_by_table: Dict[str, List[Any]] = {}
    for task in visible_successful_tasks(plan, run_result):
        intent = intent_map.get(task.task_id)
        if intent and intent.preferred_table:
            task_by_table.setdefault(intent.preferred_table, []).append(task)
    additions: List[str] = []
    understanding = plan.question_understanding or {}
    evidence_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        fields = item.get("suggestedFields") or item.get("suggested_fields") or []
        tables = item.get("suggestedTables") or item.get("suggested_tables") or []
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(tables, str):
            tables = [tables]
        if not fields or not tables:
            continue
        label = str(
            item.get("sourcePhrase")
            or item.get("source_phrase")
            or item.get("semanticLabel")
            or item.get("semantic_label")
            or fields[0]
        ).strip()
        values: List[str] = []
        for table in tables:
            for task in task_by_table.get(table, []):
                for row in (task.query_bundle.rows or [])[:10]:
                    for field in fields:
                        value = row.get(field)
                        if value in (None, ""):
                            continue
                        rendered = format_answer_cell(
                            field,
                            value,
                            label,
                            answer_column_display_contracts(plan).get(str(field)),
                        )
                        if rendered and rendered not in values:
                            values.append(rendered)
        if not values or any(value in answer for value in values):
            continue
        additions.append("- %s：%s" % (label, "、".join(values[:3])))
    if not additions:
        return answer
    return answer.rstrip() + "\n\n补充明细：\n" + "\n".join(additions[:6])


def answer_requirements(question: str, plan: QueryPlan) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    understanding = plan.question_understanding or {}
    for metric in understanding.get("selectedMetrics") or understanding.get("selected_metrics") or []:
        if isinstance(metric, dict):
            add_answer_requirement(items, metric, "selected_metric")
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if isinstance(ranking, dict):
        add_answer_requirement(items, ranking, "ranking")
    for measure in understanding.get("requestedMeasures") or understanding.get("requested_measures") or []:
        if isinstance(measure, dict):
            add_answer_requirement(items, measure, "measure")
    for evidence in understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []:
        if not isinstance(evidence, dict):
            continue
        for metric in evidence.get("suggestedMetricRefs") or evidence.get("suggested_metric_refs") or []:
            add_answer_requirement(items, {"metricRef": metric, "sourcePhrase": evidence.get("semanticLabel") or metric}, "evidence")
        for field in evidence.get("suggestedFields") or evidence.get("suggested_fields") or []:
            add_answer_requirement(items, {"metricRef": field, "sourcePhrase": evidence.get("semanticLabel") or field}, "evidence")
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        if intent.answer_mode in {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DERIVED}:
            for spec in intent.metric_specs:
                if isinstance(spec, dict):
                    add_answer_requirement(
                        items,
                        {
                            "metricRef": spec.get("metricName") or spec.get("metric_name") or spec.get("metricColumn") or spec.get("metric_column"),
                            "sourcePhrase": spec.get("displayName") or spec.get("display_name") or spec.get("metricName") or spec.get("metric_name"),
                            "sourceColumns": spec.get("sourceColumns") or spec.get("source_columns") or [],
                        },
                        "metric_spec",
                    )
            add_answer_requirement(
                items,
                {
                    "metricRef": resolution.get("metricKey") or intent.metric_name or intent.metric_column,
                    "sourcePhrase": resolution.get("displayName") or resolution.get("sourcePhrase") or intent.metric_name,
                },
                "plan",
            )
    for phrase in question_requested_ratio_phrases(question):
        if any(answer_requirement_matches_ratio_phrase(item, phrase) for item in items):
            continue
        add_answer_requirement(items, {"metricRef": phrase, "sourcePhrase": phrase}, "question_ratio")
    return dedupe_requirement_items(items)


def question_requested_ratio_phrases(question: str) -> List[str]:
    text = str(question or "").strip()
    if not text:
        return []
    phrases: List[str] = []
    segments = re.split(r"[、,，；;。！？?]|[和及与]", text)
    for segment in segments:
        cleaned = re.sub(r"(?:最近|近|过去|前)\s*\d+\s*(?:天|日|周|个月|月|年)", "", segment)
        for suffix in re.finditer(r"占比|比例|率", cleaned):
            prefix = cleaned[: suffix.end()]
            match = re.search(r"([A-Za-z0-9_\u4e00-\u9fff]{1,12}(?:占比|比例|率))$", prefix)
            phrase = str(match.group(1) if match else suffix.group(0)).strip()
            phrase = re.sub(r"^(?:请|帮我|帮忙|查询|查看|看看|看下|统计|分析|最近|当前|本期|同期)+", "", phrase)
            if phrase:
                phrases.append(phrase)
        phrases.extend(match.group(0) for match in re.finditer(r"\b[A-Za-z0-9_]*(?:rate|ratio|share)\b", cleaned, flags=re.I))
    if not phrases and re.search(r"(占多少|分别占多少)", text):
        phrases.append("占比结果")
    return dedupe_strings(phrases)


def answer_requirement_is_ratio(requirement: Dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(requirement.get("key") or ""),
            str(requirement.get("label") or ""),
            *[str(item) for item in requirement.get("aliases") or []],
        ]
    )
    return bool(re.search(r"(rate|ratio|share|占比|比例|率)", text, flags=re.I))


def answer_requirement_matches_ratio_phrase(requirement: Dict[str, Any], phrase: str) -> bool:
    requested = normalized_metric_phrase(phrase)
    if not requested:
        return False
    candidates = [
        requirement.get("key"),
        requirement.get("label"),
        *(requirement.get("aliases") or []),
    ]
    for candidate in candidates:
        normalized = normalized_metric_phrase(candidate)
        if not normalized:
            continue
        if normalized == requested:
            return True
        if len(requested) >= 2 and (requested in normalized or normalized in requested):
            return True
    return False


def add_answer_requirement(items: List[Dict[str, Any]], raw: Dict[str, Any], source: str) -> None:
    metric = str(raw.get("metricRef") or raw.get("metric_ref") or raw.get("resolvedMetricRef") or raw.get("metricKey") or "").strip()
    phrase = str(
        raw.get("displayName")
        or raw.get("sourcePhrase")
        or raw.get("source_phrase")
        or raw.get("semanticLabel")
        or raw.get("semantic_label")
        or metric
    ).strip()
    if not metric and not phrase:
        return
    label = humanize_column_name(metric) if metric else phrase
    aliases = {
        metric,
        str(raw.get("resolvedMetricRef") or "").strip(),
        str(raw.get("semanticRefId") or raw.get("semantic_ref_id") or "").strip(),
        *[str(item).strip() for item in raw.get("sourceColumns") or []],
    }
    items.append({"key": metric or phrase, "label": phrase or label, "aliases": sorted(item for item in aliases if item), "source": source})


def dedupe_requirement_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("key") or item.get("label") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def available_answer_columns(run_result: AgentRunResult | None) -> set[str]:
    columns: set[str] = set()
    if not run_result:
        return columns
    for item in visible_successful_tasks(QueryPlan(), run_result):
        for row in (item.query_bundle.rows or [])[:5]:
            columns.update(str(key) for key in row.keys())
    for row in (run_result.merged_query_bundle.rows or [])[:10]:
        columns.update(str(key) for key in row.keys())
    return columns


def partial_evidence_summary_lines(plan: QueryPlan, run_result: AgentRunResult) -> List[str]:
    lines: List[str] = []
    intent_map = intent_by_task_id(plan)
    for item in visible_successful_tasks(plan, run_result):
        rows = item.query_bundle.rows or []
        if not rows:
            continue
        intent = intent_map.get(item.task_id)
        title = task_evidence_title(intent, item, plan)
        lines.append("- %s：返回 %d 条。" % (title, item.query_bundle.effective_row_count()))
    return dedupe_strings(lines)


def section_title_for_intent(plan: QueryPlan, intent: QuestionIntent, default: str = "查询结果") -> str:
    resolution = intent.metric_resolution or {}
    metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
    metric_label = str(resolution.get("displayName") or "").strip() or (friendly_column_label(plan, metric_key) if metric_key else "")
    if is_time_series_intent(plan, intent) and metric_label:
        return "%s趋势" % metric_label
    if metric_label:
        return metric_label
    return intent.preferred_table or default


def metric_series_rows_for_intent(plan: QueryPlan, intent: QuestionIntent, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = metric_series_groups_for_intent(plan, intent, rows)
    return [point for group in groups for point in group.get("points", [])]


def query_bundle_rows_for_trend(bundle: QueryBundle | None) -> Tuple[List[Dict[str, Any]], bool]:
    """Load complete deterministic trend rows without expanding prompt previews."""

    if not bundle:
        return [], False
    preview_rows = [row for row in (bundle.rows or []) if isinstance(row, dict)]
    expected_rows = max(0, int(bundle.original_row_count or 0))
    best_rows = preview_rows
    for raw_path in bundle.offloaded_files or []:
        path = Path(str(raw_path or ""))
        if not path.name.endswith("_rows.json"):
            continue
        try:
            if not path.is_file() or path.stat().st_size > 20_000_000:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        artifact_rows = [row for row in payload if isinstance(row, dict)]
        if len(artifact_rows) > len(best_rows):
            best_rows = artifact_rows
        if not expected_rows or len(artifact_rows) >= expected_rows:
            return artifact_rows, True
    complete = not expected_rows or len(best_rows) >= expected_rows
    return best_rows, complete


def metric_series_groups_for_intent(plan: QueryPlan, intent: QuestionIntent, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    time_column = plan_time_column_for_intent(plan, intent)
    if (
        not time_column
        or intent.group_by_column != time_column
        or intent.answer_mode not in {AnswerMode.GROUP_AGG, AnswerMode.DERIVED}
        or not rows
    ):
        return []
    if intent.metric_specs:
        return metric_spec_series_groups_for_intent(plan, intent, rows, time_column)
    value_column = metric_value_column_for_rows(plan, intent, rows)
    if not value_column:
        return []
    resolution = intent.metric_resolution or {}
    metric_key = str(resolution.get("metricKey") or intent.metric_name or value_column)
    metric_label = intent_metric_label(plan, intent, metric_key, value_column)
    series = []
    for row in sorted(rows, key=lambda item: str(item.get(time_column) or "")):
        if row.get(time_column) in (None, ""):
            continue
        value = answer_numeric_value(row.get(value_column))
        if value is None:
            continue
        series.append(
            {
                "metric_name": metric_label,
                "metric_key": metric_key,
                TIME_DIMENSION_KEY: row.get(time_column),
                "value": value,
            }
        )
    return [
        {
            "metricKey": metric_key,
            "label": metric_label,
            "timeColumn": time_column,
            "displayMetadata": metric_display_contract(intent),
            "points": series,
        }
    ] if series else []


def metric_spec_series_groups_for_intent(
    plan: QueryPlan,
    intent: QuestionIntent,
    rows: List[Dict[str, Any]],
    time_column: str,
) -> List[Dict[str, Any]]:
    available = set(str(key) for row in rows[:8] for key in row.keys())
    groups: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for spec in intent.metric_specs:
        if not isinstance(spec, dict):
            continue
        metric_key = str(spec.get("metricName") or spec.get("metric_name") or spec.get("metricColumn") or spec.get("metric_column") or "").strip()
        metric_column = str(spec.get("metricColumn") or spec.get("metric_column") or metric_key).strip()
        candidates = dedupe_strings(
            [
                metric_key,
                metric_column,
                *[str(item) for item in (spec.get("sourceColumns") or spec.get("source_columns") or []) if item],
            ]
        )
        value_column = next(
            (
                candidate
                for candidate in candidates
                if candidate in available and any(answer_numeric_value(row.get(candidate)) is not None for row in rows)
            ),
            "",
        )
        if not value_column:
            continue
        key = metric_key or value_column
        if key in seen:
            continue
        seen.add(key)
        label = metric_spec_label(plan, intent, spec, key, value_column)
        points = []
        for row in sorted(rows, key=lambda item: str(item.get(time_column) or "")):
            if row.get(time_column) in (None, ""):
                continue
            value = answer_numeric_value(row.get(value_column))
            if value is None:
                continue
            points.append(
                {
                    "metric_name": label,
                    "metric_key": key,
                    TIME_DIMENSION_KEY: row.get(time_column),
                    "value": value,
                }
            )
        if points:
            groups.append(
                {
                    "metricKey": key,
                    "label": label,
                    "timeColumn": time_column,
                    "displayMetadata": metric_display_contract(intent, spec),
                    "points": points,
                }
            )
    return groups


def plan_time_column_for_intent(plan: QueryPlan, intent: QuestionIntent) -> str:
    understanding = plan.question_understanding or {}
    contract = understanding.get("timeWindowContract") or understanding.get("time_window_contract") or {}
    return declared_time_column_for_intent(intent, contract)


def is_time_series_intent(plan: QueryPlan, intent: QuestionIntent | None) -> bool:
    if intent is None:
        return False
    time_column = plan_time_column_for_intent(plan, intent)
    return bool(
        time_column
        and intent.group_by_column == time_column
        and intent.answer_mode in {AnswerMode.GROUP_AGG, AnswerMode.DERIVED}
    )


def metric_value_column_for_rows(plan: QueryPlan, intent: QuestionIntent, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    resolution = intent.metric_resolution or {}
    candidates = [
        resolution.get("metricKey"),
        intent.metric_name,
        intent.metric_column,
        *(resolution.get("sourceColumns") or []),
    ]
    available = set(str(key) for row in rows[:5] for key in row.keys())
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text in available and any(answer_numeric_value(row.get(text)) is not None for row in rows):
            return text
    if resolution.get("semanticRefId") or resolution.get("semantic_ref_id"):
        return ""
    dimension_columns = {
        str(column or "").strip()
        for column in [
            intent.group_by_column,
            intent.filter_column,
            *(resolution.get("entityColumns") or resolution.get("entity_columns") or []),
        ]
        if str(column or "").strip()
    }
    for column in rows[0].keys():
        text = str(column or "")
        if text in dimension_columns or identifier_like_column(text):
            continue
        if any(answer_numeric_value(row.get(text)) is not None for row in rows):
            return text
    return ""


def metric_display_contract(intent: QuestionIntent, spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return only runtime metric presentation metadata, without guessing from identifiers."""
    contract: Dict[str, Any] = {}
    sources = [intent.metric_resolution or {}, spec or {}]
    aliases = {
        "displayName": ["displayName", "display_name"],
        "description": ["description"],
        "unit": ["unit"],
        "valueFormat": ["valueFormat", "value_format"],
        "sourceColumnLabels": ["sourceColumnLabels", "source_column_labels"],
    }
    for target, keys in aliases.items():
        for source in sources:
            value = next((source.get(key) for key in keys if source.get(key) not in (None, "", [], {})), None)
            if value not in (None, "", [], {}):
                contract[target] = value
                break
    return contract


def summary_metric_values(plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    values: List[Dict[str, Any]] = []
    seen: set[str] = set()
    comparison_plan = plan_has_time_window_comparison(plan)
    for intent in plan.intents:
        if not intent:
            continue
        role = answer_result_role(intent)
        if role not in {"summary", "group_summary", "derived"}:
            continue
        rows = rows_for_metric_intent(plan, run_result, intent)
        if not rows:
            continue
        # A grouped metric with several entities is not a scalar summary. A
        # single grouped row, however, is the governed metric result for that
        # explicitly filtered entity and must outrank same-named columns from
        # other task/table rows.
        if role == "group_summary" and len(rows) != 1:
            continue
        metric_spec_values = summary_metric_spec_values(plan, intent, rows)
        if metric_spec_values:
            for item in metric_spec_values:
                metric_key = str(item.get("metricKey") or "")
                seen_key = summary_metric_seen_key(metric_key, str(item.get("timeWindowRole") or ""), comparison_plan)
                if not metric_key or seen_key in seen:
                    continue
                seen.add(seen_key)
                values.append(item)
            continue
        value_column = metric_value_column_for_rows(plan, intent, rows)
        if not value_column:
            continue
        value = rows[0].get(value_column)
        if value in (None, ""):
            continue
        resolution = intent.metric_resolution or {}
        metric_key = str(resolution.get("metricKey") or intent.metric_name or value_column)
        time_role = answer_time_window_role(intent, rows[0])
        seen_key = summary_metric_seen_key(metric_key, time_role, comparison_plan)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        values.append(
            {
                "metricKey": metric_key,
                "label": intent_metric_label(plan, intent, metric_key, value_column),
                "value": value,
                "displayMetadata": metric_display_contract(intent),
                "taskId": intent.plan_task_id,
                "timeWindowRole": time_role,
                "timeWindowLabel": answer_time_window_label(intent, rows[0]),
            }
        )
    return values


def summary_metric_spec_values(plan: QueryPlan, intent: QuestionIntent, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows or not intent.metric_specs:
        return []
    available = set(str(key) for row in rows[:5] for key in row.keys())
    values: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for spec in intent.metric_specs:
        if not isinstance(spec, dict):
            continue
        metric_key = str(spec.get("metricName") or spec.get("metric_name") or spec.get("metricColumn") or spec.get("metric_column") or "").strip()
        metric_column = str(spec.get("metricColumn") or spec.get("metric_column") or metric_key).strip()
        candidates = dedupe_strings(
            [
                metric_key,
                metric_column,
                *[str(item) for item in (spec.get("sourceColumns") or spec.get("source_columns") or []) if item],
            ]
        )
        value_column = next(
            (
                candidate
                for candidate in candidates
                if candidate in available and any(answer_numeric_value(row.get(candidate)) is not None for row in rows)
            ),
            "",
        )
        if not value_column:
            continue
        value = rows[0].get(value_column)
        if value in (None, ""):
            continue
        key = metric_key or value_column
        if key in seen:
            continue
        seen.add(key)
        values.append(
            {
                "metricKey": key,
                "label": metric_spec_label(plan, intent, spec, key, value_column),
                "value": value,
                "displayMetadata": metric_display_contract(intent, spec),
                "taskId": intent.plan_task_id,
                "timeWindowRole": answer_time_window_role(intent, rows[0]),
                "timeWindowLabel": answer_time_window_label(intent, rows[0]),
            }
        )
    return values


def plan_has_time_window_comparison(plan: QueryPlan) -> bool:
    contract = (plan.question_understanding or {}).get("timeWindowContract") or {}
    if isinstance(contract, dict) and contract.get("requiresComparison"):
        return True
    for intent in plan.intents:
        if answer_time_window_role(intent, {}) == "comparison":
            return True
    return False


def summary_metric_seen_key(metric_key: str, time_role: str, comparison_plan: bool) -> str:
    if comparison_plan:
        return "%s|%s" % (metric_key, time_role or "primary")
    return metric_key


def answer_time_window_role(intent: QuestionIntent | None, row: Dict[str, Any]) -> str:
    row_role = str((row or {}).get("__timeWindowRole") or "").strip()
    if row_role:
        return row_role
    if intent:
        role = str(getattr(intent.time_range, "window_role", "") or (intent.metric_resolution or {}).get("timeWindowRole") or "").strip()
        if role:
            return role
    return ""


def answer_time_window_label(intent: QuestionIntent | None, row: Dict[str, Any]) -> str:
    row_label = str((row or {}).get("__timeWindowLabel") or "").strip()
    if row_label:
        return row_label
    if intent:
        label = str(getattr(intent.time_range, "label", "") or "").strip()
        if label:
            return label
    return ""


def metric_spec_label(
    plan: QueryPlan,
    intent: QuestionIntent,
    spec: Dict[str, Any],
    metric_key: str,
    value_column: str,
) -> str:
    del plan, metric_key, value_column
    for key in ["displayName", "display_name", "naturalName", "natural_name", "label", "metricLabel", "metric_label"]:
        label = str(spec.get(key) or "").strip()
        if label:
            return label
    resolution = intent.metric_resolution or {}
    return str(resolution.get("displayName") or "指标").strip()


def intent_metric_label(plan: QueryPlan, intent: QuestionIntent, metric_key: str, value_column: str) -> str:
    del plan, metric_key, value_column
    resolution = intent.metric_resolution or {}
    return str(resolution.get("displayName") or "指标").strip()


def rows_for_metric_intent(plan: QueryPlan, run_result: AgentRunResult, intent: QuestionIntent) -> List[Dict[str, Any]]:
    """Find verified rows for a metric intent, even when execution coalesced it."""

    if not run_result:
        return []
    candidate_groups: List[List[Dict[str, Any]]] = []
    intent_map = intent_by_task_id(plan)
    for item in visible_successful_tasks(plan, run_result):
        rows = list(getattr(getattr(item, "query_bundle", None), "rows", []) or [])
        if not rows:
            continue
        if item.task_id == intent.plan_task_id:
            candidate_groups.insert(0, rows)
            continue
        task_intent = intent_map.get(item.task_id)
        if task_intent and intent.preferred_table and task_intent.preferred_table != intent.preferred_table:
            continue
        candidate_groups.append(rows)
    merged_rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
    if merged_rows:
        candidate_groups.append(merged_rows)
    for rows in candidate_groups:
        if metric_value_column_for_rows(plan, intent, rows) or metric_spec_series_or_summary_available(intent, rows):
            return rows
    return []


def metric_spec_series_or_summary_available(intent: QuestionIntent, rows: List[Dict[str, Any]]) -> bool:
    if not rows or not intent.metric_specs:
        return False
    available = set(str(key) for row in rows[:8] for key in row.keys())
    for spec in intent.metric_specs:
        if not isinstance(spec, dict):
            continue
        candidates = dedupe_strings(
            [
                str(spec.get("metricName") or spec.get("metric_name") or ""),
                str(spec.get("metricColumn") or spec.get("metric_column") or ""),
                *[str(item) for item in (spec.get("sourceColumns") or spec.get("source_columns") or []) if item],
            ]
        )
        if any(candidate in available and any(answer_numeric_value(row.get(candidate)) is not None for row in rows) for candidate in candidates):
            return True
    return False


def answer_metric_facts(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    summaries = summary_metric_values(plan, run_result) if run_result else []
    if not summaries:
        return []
    time_phrase = extract_question_time_phrase(question) or "本次查询范围"
    comparison_by_metric = {
        str(item.get("metricKey") or ""): item
        for item in summaries
        if str(item.get("timeWindowRole") or "") == "comparison"
    }
    facts: List[Dict[str, Any]] = []
    for item in summaries[:8]:
        label = str(item.get("label") or item.get("metricKey") or "指标")
        metric_key = str(item.get("metricKey") or "")
        time_label = str(item.get("timeWindowLabel") or "").strip() or time_phrase
        fact = {
            "metricKey": metric_key,
            "displayName": label,
            "value": item.get("value"),
            "formattedValue": format_metric_value_for_answer(
                item.get("value"), metric_key, label, item.get("displayMetadata")
            ),
            "timeRange": time_label,
            "timeWindowRole": item.get("timeWindowRole") or "",
            "taskId": item.get("taskId") or "",
        }
        if str(item.get("timeWindowRole") or "") == "primary":
            comparison = comparison_by_metric.get(metric_key)
            if comparison:
                fact.update(metric_comparison_payload(item, comparison))
        facts.append(fact)
    return facts


def answer_metric_comparison_facts(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    return [fact for fact in answer_metric_facts(question, plan, run_result) if fact.get("comparisonValue") not in (None, "")]


def metric_comparison_payload(primary: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    label = str(primary.get("label") or primary.get("metricKey") or "指标")
    metric_key = str(primary.get("metricKey") or "")
    primary_value = answer_numeric_value(primary.get("value"))
    comparison_value = answer_numeric_value(comparison.get("value"))
    payload: Dict[str, Any] = {
        "comparisonTimeRange": comparison.get("timeWindowLabel") or "对比窗口",
        "comparisonValue": comparison.get("value"),
        "formattedComparisonValue": format_metric_value_for_answer(
            comparison.get("value"), metric_key, label, primary.get("displayMetadata")
        ),
    }
    if primary_value is None or comparison_value is None:
        return payload
    change_value = primary_value - comparison_value
    change_rate = (change_value / abs(comparison_value)) if comparison_value else None
    payload.update(
        {
            "changeValue": change_value,
            "formattedChangeValue": format_metric_change_value_for_answer(
                change_value, metric_key, label, primary.get("displayMetadata")
            ),
            "changeRate": change_rate,
            "formattedChangeRate": format_signed_percent(change_rate) if change_rate is not None else "",
            "direction": metric_change_direction(change_value),
        }
    )
    return payload


def metric_change_direction(change_value: float) -> str:
    if abs(change_value) <= 1e-12:
        return "持平"
    return "上升" if change_value > 0 else "下降"


def format_metric_change_value_for_answer(
    value: float,
    metric_key: str,
    label: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    contract = metadata or {}
    value_format = str(contract.get("valueFormat") or contract.get("value_format") or "").strip().lower()
    if value_format in {"percent", "percentage", "ratio"} or str(contract.get("unit") or "").strip() == "%":
        points = abs(float(value or 0)) * 100
        if float(points).is_integer():
            return "%s个百分点" % int(points)
        return ("%.2f个百分点" % points).replace(".00个百分点", "个百分点")
    return format_metric_value_for_answer(abs(value), metric_key, label, contract)


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return ""
    percent = float(value) * 100
    sign = "+" if percent > 0 else ""
    if float(percent).is_integer():
        return "%s%d%%" % (sign, int(percent))
    return ("%s%.2f%%" % (sign, percent)).replace(".00%", "%")


def include_mandatory_skeleton_in_answer_prompt(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    skeleton: str,
) -> bool:
    if not skeleton:
        return False
    if not run_result:
        return True
    if getattr(run_result, "evidence_gaps", None):
        return True
    coverage = answer_requirement_coverage(question, plan, run_result)
    if coverage.get("shouldBlockDirectAnswer"):
        return True
    if answer_metric_comparison_facts(question, plan, run_result):
        return True
    return not bool(answer_metric_facts(question, plan, run_result))


def should_proactively_patch_metric_summary(
    answer: str,
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
) -> bool:
    if not run_result:
        return True
    if getattr(run_result, "evidence_gaps", None):
        return True
    metric_facts = answer_metric_facts(question, plan, run_result)
    if not metric_facts:
        return True
    return any(not answer_contains_metric_fact_value(answer, fact) for fact in metric_facts)


def should_apply_mandatory_skeleton_to_draft(
    answer: str,
    skeleton: str,
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
) -> bool:
    if not skeleton:
        return False
    if not run_result:
        return True
    if getattr(run_result, "evidence_gaps", None):
        return True
    if answer_requirement_coverage(question, plan, run_result).get("shouldBlockDirectAnswer"):
        return True
    metric_facts = answer_metric_facts(question, plan, run_result)
    if not metric_facts:
        return True
    return any(not answer_contains_metric_fact_value(answer, fact) for fact in metric_facts)


def lightweight_answer_contract_verification(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    answer: str,
) -> AnswerClaimVerification | None:
    metric_facts = answer_metric_facts(question, plan, run_result)
    if not metric_facts:
        return None
    claims: List[AnswerClaim] = []
    unsupported: List[AnswerClaim] = []
    missing = [fact for fact in metric_facts if not answer_contains_metric_fact_value(answer, fact)]
    if missing:
        claim = AnswerClaim(
            text="metricFacts coverage",
            numeric_values=[str(item.get("formattedValue") or item.get("value") or "") for item in missing],
            supported=False,
            reasons=["missing_metric_fact:%s" % (item.get("metricKey") or item.get("displayName") or "") for item in missing],
        )
        claims.append(claim)
        unsupported.append(claim)
    extra_numbers = unsupported_answer_numbers(answer, question, plan, run_result, metric_facts)
    if extra_numbers:
        claim = AnswerClaim(
            text="extra numeric values",
            numeric_values=extra_numbers,
            supported=False,
            reasons=["unsupported_extra_value:%s" % value for value in extra_numbers],
        )
        claims.append(claim)
        unsupported.append(claim)
    if not claims:
        claims.append(
            AnswerClaim(
                text="metricFacts coverage",
                numeric_values=[str(item.get("formattedValue") or item.get("value") or "") for item in metric_facts],
                supported=True,
            )
        )
    return AnswerClaimVerification(
        passed=not unsupported,
        fact_count=len(metric_facts),
        claims=claims,
        unsupported_claims=unsupported,
    )


def lightweight_answer_failure_reason(verification: AnswerClaimVerification | None) -> str:
    if verification is None:
        return ""
    reasons: List[str] = []
    for claim in verification.unsupported_claims[:5]:
        reasons.extend(str(item) for item in claim.reasons if str(item))
    return ";".join(dedupe_strings(reasons)[:8]) or "lightweight_answer_contract_failed"


def verified_metric_facts_fallback_answer(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
) -> str:
    """Render only required metric facts, without unverified prose or deltas."""

    facts = answer_metric_facts(question, plan, run_result)
    if not facts:
        return ""
    lines = ["已验证结果："]
    seen: set[Tuple[str, str, str]] = set()
    for fact in facts:
        label = str(fact.get("displayName") or fact.get("metricKey") or "指标").strip()
        time_range = str(fact.get("timeRange") or "").strip()
        value = str(fact.get("formattedValue") or fact.get("value") or "").strip()
        key = (label, time_range, value)
        if not value or key in seen:
            continue
        seen.add(key)
        subject = "%s%s" % (("%s " % time_range) if time_range else "", label)
        lines.append("- %s：%s" % (subject, value))
    return "\n".join(lines) if len(lines) > 1 else ""


def answer_contains_metric_fact_value(answer: str, fact: Dict[str, Any]) -> bool:
    answer_numbers = answer_numeric_tokens(answer)
    expected = numeric_token_value(fact.get("value"))
    if expected is not None and any(float_close(expected, item) for item in answer_numbers):
        return True
    formatted = normalize_answer_value_text(fact.get("formattedValue"))
    raw = normalize_answer_value_text(fact.get("value"))
    text = normalize_answer_value_text(answer)
    return bool((formatted and formatted in text) or (raw and raw in text))


def answer_contains_summary_metric_value(answer: str, item: Dict[str, Any]) -> bool:
    label = str(item.get("label") or "")
    metric_key = str(item.get("metricKey") or "")
    return answer_contains_metric_fact_value(
        answer,
        {
            "value": item.get("value"),
            "formattedValue": format_metric_value_for_answer(
                item.get("value"), metric_key, label, item.get("displayMetadata")
            ),
        },
    )


def unsupported_answer_numbers(
    answer: str,
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    metric_facts: List[Dict[str, Any]],
) -> List[str]:
    allowed = allowed_answer_numbers(question, plan, run_result, metric_facts)
    unsupported: List[str] = []
    for raw, value in answer_numeric_token_pairs(answer):
        if any(float_close(value, candidate) for candidate in allowed):
            continue
        # Avoid treating list counts and prose scaffolding as business facts.
        if abs(value) < 10 and not raw.endswith("%") and "." not in raw:
            continue
        unsupported.append(raw)
    return dedupe_strings(unsupported)


def allowed_answer_numbers(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    metric_facts: List[Dict[str, Any]],
) -> List[float]:
    values: List[float] = []
    for fact in metric_facts:
        value = numeric_token_value(fact.get("value"))
        if value is not None:
            values.append(value)
        for key in ["comparisonValue", "changeValue", "changeRate"]:
            numeric = numeric_token_value(fact.get(key))
            if numeric is not None:
                values.append(numeric)
                if key == "changeValue":
                    values.append(abs(numeric))
        # The deterministic renderer intentionally presents decreases as an
        # absolute magnitude after the direction word (and rate deltas as
        # percentage points).  Admit those exact rendered values as well as
        # the signed raw facts used to calculate them.
        for key in [
            "formattedValue",
            "formattedComparisonValue",
            "formattedChangeValue",
            "formattedChangeRate",
        ]:
            values.extend(numeric for _, numeric in answer_numeric_token_pairs(str(fact.get(key) or "")))
    if run_result:
        for row in list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])[:20]:
            for value in row.values():
                numeric = numeric_token_value(value)
                if numeric is not None:
                    values.append(numeric)
        for task in getattr(run_result, "task_results", []) or []:
            task_rows, _ = query_bundle_rows_for_trend(getattr(task, "query_bundle", None))
            for row in task_rows[:2000]:
                for value in row.values():
                    numeric = numeric_token_value(value)
                    if numeric is not None:
                        values.append(numeric)
        deterministic_trend = multi_trend_metric_sentence(question, plan, run_result)
        values.extend(value for _, value in answer_numeric_token_pairs(deterministic_trend))
    values.extend(value for _, value in answer_numeric_token_pairs(question))
    return values


def answer_numeric_tokens(text: str) -> List[float]:
    return [value for _, value in answer_numeric_token_pairs(text)]


def answer_numeric_token_pairs(text: str) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    scrubbed = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", str(text or ""))
    for match in re.finditer(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?%?", scrubbed):
        raw = match.group(0)
        value = numeric_token_value(raw)
        if value is None:
            continue
        pairs.append((raw, value))
    return pairs


def numeric_token_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    percent = text.endswith("%")
    text = text.rstrip("%")
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if percent else number


def float_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(0.005, abs(right) * 0.000001)


def normalize_answer_value_text(value: Any) -> str:
    return re.sub(r"[\s,，元%]+", "", str(value or "").strip().lower())


def multi_summary_metric_sentence(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    summaries = summary_metric_values(plan, run_result) if run_result else []
    return summary_metric_sentence_from_items(question, summaries) if len(summaries) > 1 else ""


def summary_metric_sentence(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    summaries = summary_metric_values(plan, run_result)
    return summary_metric_sentence_from_items(question, summaries)


def summary_metric_sentence_from_items(question: str, summaries: List[Dict[str, Any]]) -> str:
    if not summaries:
        return ""
    comparison_sentence = comparison_metric_sentence_from_items(question, summaries)
    if comparison_sentence:
        return comparison_sentence
    time_phrase = extract_question_time_phrase(question)
    prefix = "%s，" % time_phrase if time_phrase else "当前查询范围内，"
    parts = []
    for item in summaries[:5]:
        label = item.get("label") or "指标"
        value = format_metric_value_for_answer(
            item.get("value"), item.get("metricKey") or "", str(label), item.get("displayMetadata")
        )
        parts.append("%s为 %s" % (label, value))
    return prefix + "，".join(parts) + "。"


def comparison_metric_sentence_from_items(question: str, summaries: List[Dict[str, Any]]) -> str:
    if not summaries:
        return ""
    primary_items = [item for item in summaries if str(item.get("timeWindowRole") or "") == "primary"]
    comparison_by_metric = {
        str(item.get("metricKey") or ""): item
        for item in summaries
        if str(item.get("timeWindowRole") or "") == "comparison"
    }
    if not primary_items or not comparison_by_metric:
        return ""
    time_phrase = primary_items[0].get("timeWindowLabel") or extract_question_time_phrase(question) or "当前查询范围"
    parts: List[str] = []
    for item in primary_items[:6]:
        metric_key = str(item.get("metricKey") or "")
        comparison = comparison_by_metric.get(metric_key)
        if not comparison:
            continue
        label = str(item.get("label") or metric_key or "指标")
        current_value = format_metric_value_for_answer(
            item.get("value"), metric_key, label, item.get("displayMetadata")
        )
        comparison_label = str(comparison.get("timeWindowLabel") or "对比窗口")
        comparison_value = format_metric_value_for_answer(
            comparison.get("value"), metric_key, label, item.get("displayMetadata")
        )
        payload = metric_comparison_payload(item, comparison)
        direction = str(payload.get("direction") or "")
        if direction == "持平":
            change_text = "持平"
        elif direction:
            rate_text = str(payload.get("formattedChangeRate") or "").strip()
            suffix = "，%s" % rate_text if rate_text else ""
            change_text = "%s %s%s" % (direction, payload.get("formattedChangeValue") or "", suffix)
        else:
            change_text = "已完成对比"
        parts.append("%s为 %s（%s为 %s，%s）" % (label, current_value, comparison_label, comparison_value, change_text))
    if not parts:
        return ""
    return "%s，%s。" % (time_phrase, "；".join(parts))


def chart_hint_sentence(plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    has_trend = any(answer_result_role(intent_by_task_id(plan).get(item.task_id)) == "trend_context" for item in visible_successful_tasks(plan, run_result))
    return "\n\n按日趋势明细已同步到图表区域。" if has_trend else ""


def multi_trend_metric_sentence(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    parts: List[str] = []
    single_point_parts: List[str] = []
    trend_groups: List[Dict[str, Any]] = []
    incomplete_labels: List[str] = []
    for task in visible_successful_tasks(plan, run_result):
        intent = intent_by_task_id(plan).get(task.task_id)
        if answer_result_role(intent) != "trend_context" or not task.query_bundle.rows:
            continue
        trend_rows, complete = query_bundle_rows_for_trend(task.query_bundle)
        for group in metric_series_groups_for_intent(plan, intent, trend_rows) if intent else []:
            points = list(group.get("points") or [])
            group = {**group, "points": points, "seriesComplete": complete}
            trend_groups.append(group)
            metric_key = str(group.get("metricKey") or "")
            label = str(group.get("label") or friendly_column_label(plan, metric_key) or "指标")
            display_metadata = group.get("displayMetadata") if isinstance(group.get("displayMetadata"), dict) else {}
            if not complete:
                incomplete_labels.append(label)
                continue
            if len(points) < 2:
                if len(points) == 1:
                    point = points[0]
                    single_point_parts.append(
                        "%s在 %s 的值为 %s"
                        % (
                            label,
                            format_cell(point.get(TIME_DIMENSION_KEY)),
                            format_metric_value_for_answer(point.get("value"), metric_key, label, display_metadata),
                        )
                    )
                continue
            first, last = points[0], points[-1]
            first_value = answer_numeric_value(first.get("value"))
            last_value = answer_numeric_value(last.get("value"))
            if first_value is None or last_value is None:
                continue
            delta = last_value - first_value
            direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
            if delta == 0:
                change_text = "整体持平"
            else:
                change_text = "整体%s %s" % (
                    direction,
                    format_metric_value_for_answer(abs(delta), metric_key, label, display_metadata),
                )
            parts.append(
                "%s从 %s 的 %s 变化到 %s 的 %s，%s"
                % (
                    label,
                    format_cell(first.get(TIME_DIMENSION_KEY)),
                    format_metric_value_for_answer(first.get("value"), metric_key, label, display_metadata),
                    format_cell(last.get(TIME_DIMENSION_KEY)),
                    format_metric_value_for_answer(last.get("value"), metric_key, label, display_metadata),
                    change_text,
                )
            )
    if not parts and not single_point_parts and not incomplete_labels:
        return ""
    time_phrase = extract_question_time_phrase(question)
    prefix = "%s，" % time_phrase if time_phrase else "当前查询范围内，"
    sync = trend_sync_judgement(question, trend_groups)
    sync_prefix = "%s：" % sync if sync else ""
    complete_groups = [group for group in trend_groups if group.get("seriesComplete")]
    extreme = trend_extreme_change_phrase(complete_groups)
    suffix = "；%s" % extreme if extreme else ""
    body_parts = [*parts[:6]]
    body_parts.extend(single_point_parts[: max(0, 6 - len(body_parts))])
    if incomplete_labels:
        body_parts.append("%s按日数据未加载完整，未使用预览片段判断首尾变化或同步性" % "、".join(dedupe_strings(incomplete_labels)))
    return prefix + sync_prefix + "；".join(body_parts) + suffix + "。"


def trend_extreme_change_phrase(groups: List[Dict[str, Any]]) -> str:
    extremes: List[Dict[str, Any]] = []
    for group in groups:
        metric_key = str(group.get("metricKey") or "")
        label = str(group.get("label") or "指标")
        points = list(group.get("points") or [])
        best: Dict[str, Any] = {}
        previous: Dict[str, Any] | None = None
        for point in points:
            if previous is None:
                previous = point
                continue
            prev_value = answer_numeric_value(previous.get("value"))
            value = answer_numeric_value(point.get("value"))
            if prev_value is None or value is None:
                previous = point
                continue
            delta = value - prev_value
            score = abs(delta)
            if not best or score > float(best.get("score") or 0):
                best = {
                    "score": score,
                    "label": label,
                    "metricKey": metric_key,
                    "displayMetadata": group.get("displayMetadata") or {},
                    TIME_DIMENSION_KEY: point.get(TIME_DIMENSION_KEY),
                    "delta": delta,
                }
            previous = point
        if best:
            extremes.append(best)
    if not extremes:
        return ""

    phrases: List[str] = []
    for best in extremes[:6]:
        delta = float(best.get("delta") or 0)
        direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
        metric_key = str(best.get("metricKey") or "")
        label = str(best.get("label") or "指标")
        phrases.append(
            "%s 在 %s，较前一日%s %s"
            % (
                label,
                format_cell(best.get(TIME_DIMENSION_KEY)),
                direction,
                format_metric_value_for_answer(abs(delta), metric_key, label, best.get("displayMetadata")),
            )
        )
    if len(phrases) == 1:
        return "最大单日变化点是 %s" % phrases[0]
    return "各指标最大单日变化分别是：%s；不同量纲不做横向大小比较" % "；".join(phrases)


def trend_sync_judgement(question: str, groups: List[Dict[str, Any]]) -> str:
    text = str(question or "")
    if "同步" not in text or len(groups) < 2:
        return ""
    target = "any"
    if re.search(r"同步(上升|增长|上涨)", text):
        target = "up"
    elif re.search(r"同步(下降|减少|下滑)", text):
        target = "down"
    analysis = aligned_trend_sync_analysis(groups, target)
    target_label = "同步上升" if target == "up" else "同步下降" if target == "down" else "同步变化"
    expected = int(analysis.get("expectedIntervals") or 0)
    comparable = int(analysis.get("comparableIntervals") or 0)
    if analysis.get("incompleteSeries"):
        return "按日数据未加载完整，暂不能判断是否%s" % target_label
    alignment_coverage = float(analysis.get("alignmentCoverage") or 0)
    if comparable < 2 or expected <= 0 or alignment_coverage < 0.5:
        return "共同日期不足，暂不能判断是否%s（日期对齐覆盖率 %s，%d/%d 个相邻日区间可比；缺失日期未按 0 处理）" % (
            target_label,
            format_trend_coverage(alignment_coverage),
            comparable,
            expected,
        )
    matching = int(analysis.get("matchingIntervals") or 0)
    consistency = float(analysis.get("consistencyRate") or 0)
    synchronized = consistency >= 0.8
    if target == "up":
        outcome = "同步上升" if synchronized else "没有同步上升"
    elif target == "down":
        outcome = "同步下降" if synchronized else "没有同步下降"
    else:
        outcome = "走势基本同步" if synchronized else "走势不同步"
    return "%s，同步覆盖率 %s（%d/%d 个共同可比日变化），日期对齐覆盖率 %s（%d/%d）" % (
        outcome,
        format_trend_coverage(consistency),
        matching,
        comparable,
        format_trend_coverage(alignment_coverage),
        comparable,
        expected,
    )


def aligned_trend_sync_analysis(groups: List[Dict[str, Any]], target: str = "any") -> Dict[str, Any]:
    series_maps: List[Dict[date, float]] = []
    incomplete = False
    for group in groups:
        if not bool(group.get("seriesComplete", True)):
            incomplete = True
        values: Dict[date, float] = {}
        for point in group.get("points") or []:
            point_date = trend_point_date(point.get(TIME_DIMENSION_KEY))
            value = answer_numeric_value(point.get("value"))
            if point_date is None or value is None:
                continue
            values[point_date] = value
        series_maps.append(values)
    available_dates = sorted({item for series in series_maps for item in series})
    if len(series_maps) < 2 or len(available_dates) < 2:
        return {
            "incompleteSeries": incomplete,
            "expectedIntervals": 0,
            "comparableIntervals": 0,
            "matchingIntervals": 0,
            "alignmentCoverage": 0.0,
            "consistencyRate": 0.0,
        }
    start, end = available_dates[0], available_dates[-1]
    expected = max(0, (end - start).days)
    comparable = 0
    matching = 0
    for ordinal in range(start.toordinal() + 1, end.toordinal() + 1):
        current = date.fromordinal(ordinal)
        previous = date.fromordinal(ordinal - 1)
        if any(previous not in series or current not in series for series in series_maps):
            continue
        comparable += 1
        directions = [
            1 if series[current] > series[previous] else -1 if series[current] < series[previous] else 0
            for series in series_maps
        ]
        if target == "up":
            matched = all(item > 0 for item in directions)
        elif target == "down":
            matched = all(item < 0 for item in directions)
        else:
            matched = len(set(directions)) == 1
        if matched:
            matching += 1
    return {
        "incompleteSeries": incomplete,
        "expectedIntervals": expected,
        "comparableIntervals": comparable,
        "matchingIntervals": matching,
        "alignmentCoverage": (comparable / expected) if expected else 0.0,
        "consistencyRate": (matching / comparable) if comparable else 0.0,
    }


def trend_point_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def format_trend_coverage(value: float) -> str:
    return ("%.2f%%" % (float(value or 0) * 100)).replace(".00%", "%")


def primary_summary_metric_value(plan: QueryPlan, run_result: AgentRunResult) -> Dict[str, Any]:
    if not run_result:
        return {}
    values = summary_metric_values(plan, run_result)
    if values:
        return values[0]
    return {}


def primary_trend_points(plan: QueryPlan, run_result: AgentRunResult, metric_key: str) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    for item in visible_successful_tasks(plan, run_result):
        intent = intent_by_task_id(plan).get(item.task_id)
        if answer_result_role(intent) != "trend_context" or not item.query_bundle.rows:
            continue
        resolution = intent.metric_resolution or {}
        current_key = str(resolution.get("metricKey") or intent.metric_name or "").strip() if intent else ""
        if metric_key and current_key and metric_key != current_key:
            continue
        trend_rows, complete = query_bundle_rows_for_trend(item.query_bundle)
        return metric_series_rows_for_intent(plan, intent, trend_rows) if intent and complete else []
    return []


def primary_answer_metric_column(plan: QueryPlan, row: Dict[str, Any]) -> str:
    available = set(str(key) for key in (row or {}).keys())
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        candidates = [
            resolution.get("metricKey"),
            intent.metric_name,
            intent.metric_column,
            *(resolution.get("sourceColumns") or []),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text and text in available and not identifier_like_column(text):
                return text
    if any((intent.metric_resolution or {}).get("semanticRefId") for intent in plan.intents):
        return ""
    for column, value in (row or {}).items():
        text = str(column or "")
        if identifier_like_column(text):
            continue
        if isinstance(value, (int, float)) or re.fullmatch(r"-?\d+(\.\d+)?", str(value or "")):
            return text
    return ""


def primary_entity_column(plan: QueryPlan, row: Dict[str, Any]) -> str:
    available = set(str(key) for key in (row or {}).keys())
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        candidates = [
            intent.filter_column,
            intent.group_by_column,
            *(resolution.get("entityColumns") or resolution.get("entity_columns") or []),
            *intent.output_keys,
        ]
        for candidate in candidates:
            column = str(candidate or "").strip()
            if entity_like_column(column) and column in available and row.get(column) not in (None, ""):
                return column
    for column in primary_summary_entity_columns(plan):
        if column in available and row.get(column) not in (None, ""):
            return column
    return ""


def friendly_column_label(plan: QueryPlan, column: str) -> str:
    labels = answer_column_labels(plan)
    if labels.get(column):
        return labels[column]
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        if column == str(resolution.get("metricKey") or "") and resolution.get("displayName"):
            return str(resolution.get("displayName"))
    return humanize_column_name(column)


def default_answer_column_label(column: str) -> str:
    del column
    return ""


def fallback_metric_label(intent: QuestionIntent, metric: str) -> str:
    del intent, metric
    return ""


def is_ranking_plan(plan: QueryPlan) -> bool:
    return any(intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG} for intent in plan.intents)


def fallback_display_columns(plan: QueryPlan, rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    available = set()
    for row in rows[:8]:
        available.update(str(key) for key in row.keys())
    preferred: List[str] = []
    for intent in plan.intents:
        for column in [intent.group_by_column, intent.filter_column, intent.metric_name, intent.metric_column] + intent.output_keys[:8]:
            if column and column in available and column not in preferred:
                preferred.append(column)
    for column in rows[0].keys():
        if column not in preferred:
            preferred.append(str(column))
        if len(preferred) >= 8:
            break
    return preferred[:8]


def dedupe_strings(items: List[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def answer_time_prefix(question: str) -> str:
    time_prefix = extract_question_time_phrase(question)
    return "%s，" % time_prefix if time_prefix else "当前查询范围内，"


def question_requests_diagnosis(question: str) -> bool:
    return bool(re.search(r"(为什么|原因|归因|怎么回事|分析|建议|怎么办|策略|优化|风险|异常)", str(question or "")))


def deterministic_structured_answer(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    fallback_rows: List[Dict[str, Any]] | None = None,
) -> str:
    if not run_result and not fallback_rows:
        return ""
    detail = deterministic_cross_task_detail_answer(question, plan, run_result)
    if detail:
        return detail
    if run_result and answer_metric_comparison_facts(question, plan, run_result):
        summary_sentence = summary_metric_sentence(question, plan, run_result)
        if summary_sentence:
            return summary_sentence + chart_hint_sentence(plan, run_result)
    ranking = deterministic_ranking_answer(question, plan, run_result)
    if ranking:
        return ranking
    if run_result:
        summary_sentence = summary_metric_sentence(question, plan, run_result)
        if summary_sentence:
            return summary_sentence + chart_hint_sentence(plan, run_result)
        trend_sentence = multi_trend_metric_sentence(question, plan, run_result)
        if question_requests_diagnosis(question):
            if trend_sentence:
                return trend_sentence + chart_hint_sentence(plan, run_result)
            return ""
        if trend_sentence:
            return trend_sentence + chart_hint_sentence(plan, run_result)
        derived_sentence = derived_metric_sentence(question, plan, run_result)
        if derived_sentence:
            return derived_sentence + chart_hint_sentence(plan, run_result)
    rows = fallback_rows or (run_result.merged_query_bundle.rows if run_result else [])
    if not rows:
        return ""
    if len(rows) == 1:
        row = rows[0]
        metric_column = primary_answer_metric_column(plan, row)
        entity_column = primary_entity_column(plan, row)
        if metric_column:
            metric_label = friendly_column_label(plan, metric_column)
            metric_value = format_answer_cell(
                metric_column,
                row.get(metric_column),
                metric_label,
                answer_column_display_contracts(plan).get(metric_column),
            )
            if entity_column:
                entity_label = friendly_column_label(plan, entity_column)
                return "%s%s %s 的%s为 %s。" % (
                    answer_time_prefix(question),
                    entity_label,
                    format_cell(row.get(entity_column)),
                    metric_label,
                    metric_value,
                )
            return "%s%s为 %s。" % (answer_time_prefix(question), metric_label, metric_value)
        return single_row_detail_sentence(question, plan, row)
    sample = row_sample_sentence(question, plan, rows)
    if sample:
        return sample
    return generic_result_overview_sentence(question, plan, rows, run_result)


def derived_metric_sentence(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    intent_map = intent_by_task_id(plan)
    for task in visible_successful_tasks(plan, run_result):
        intent = intent_map.get(task.task_id)
        if not intent or intent.answer_mode != AnswerMode.DERIVED or not task.query_bundle.rows:
            continue
        rows = task.query_bundle.rows
        if len(rows) != 1:
            continue
        value_column = metric_value_column_for_rows(plan, intent, rows)
        if not value_column:
            continue
        row = rows[0]
        value = row.get(value_column)
        if value in (None, ""):
            continue
        resolution = intent.metric_resolution or {}
        metric_key = str(resolution.get("metricKey") or intent.metric_name or value_column)
        label = str(resolution.get("displayName") or friendly_column_label(plan, value_column))
        return "%s%s为 %s。" % (
            answer_time_prefix(question),
            label,
            format_metric_value_for_answer(value, metric_key, label, resolution),
        )
    return ""


def deterministic_cross_task_detail_answer(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
) -> str:
    if not run_result or len(plan.intents) <= 1 or not any(intent.answer_mode == AnswerMode.DETAIL for intent in plan.intents):
        return ""
    task_map = {item.task_id: item for item in run_result.task_results if not item.query_bundle.failed and item.query_bundle.rows}
    sections: List[str] = []
    for intent in plan.intents:
        task = task_map.get(intent.plan_task_id)
        if not task:
            continue
        row = task.query_bundle.rows[0]
        section_plan = QueryPlan(intents=[intent])
        labels = answer_column_labels(section_plan)
        requested = dedupe_strings(list(intent.output_keys or []) + list(intent.required_evidence or []))
        display_columns: List[str] = []
        for column in requested:
            if column not in row or row.get(column) in (None, ""):
                continue
            if identifier_like_column(column) and not labels.get(column):
                continue
            if column.endswith("_id") and column.replace("_id", "_name") in row and row.get(column.replace("_id", "_name")) not in (None, ""):
                continue
            display_columns.append(column)
        if not display_columns:
            continue
        contracts = answer_column_display_contracts(section_plan)
        resolution = intent.metric_resolution or {}
        title = str(resolution.get("displayName") or "").strip()
        if not title:
            title = "%s明细" % category_display(intent.category)
        lines = ["%s：" % title]
        for column in display_columns[:10]:
            label = friendly_column_label(section_plan, column)
            lines.append(
                "- %s：%s"
                % (label, format_answer_cell(column, row.get(column), label, contracts.get(column)))
            )
        sections.append("\n".join(lines))
    if not sections:
        return ""
    prefix = extract_question_time_phrase(question)
    heading = "%s查询结果：" % prefix if prefix else "查询结果："
    return heading + "\n\n" + "\n\n".join(sections)


def deterministic_ranking_preferred_before_llm(question: str) -> bool:
    text = str(question or "")
    return bool(re.search(r"(top|前\s*\d+|排行|排名|最高.{0,8}(哪些|几个|列表)|最多.{0,8}(哪些|几个|列表))", text, flags=re.I))


def merge_deterministic_ranking_with_llm_answer(ranking_answer: str, llm_answer: str) -> str:
    ranking = (ranking_answer or "").strip()
    analysis = (llm_answer or "").strip()
    if not ranking:
        return analysis
    if not analysis:
        return ranking
    ranking_first_value = ""
    table_rows = [line for line in ranking.splitlines() if line.startswith("|") and "---" not in line]
    if len(table_rows) >= 2:
        cells = [cell.strip() for cell in table_rows[1].strip("|").split("|")]
        ranking_first_value = " ".join(cell for cell in cells[:2] if cell)
    if ranking_first_value and ranking_first_value in analysis:
        return analysis
    if analysis.startswith("数据结果：") or analysis.startswith("结果："):
        return analysis
    return "数据结果：\n%s\n\n分析判断：\n%s" % (ranking, analysis)


def format_answer_cell(
    column: str,
    value: Any,
    label: str = "",
    metadata: Dict[str, Any] | None = None,
) -> str:
    text = str(column or "").strip().lower()
    if identifier_like_column(text) or re.search(r"(name|title|time|date|日期|时间|名称)", text):
        return format_cell(value)
    if answer_numeric_value(value) is not None:
        return format_metric_value_for_answer(value, column, label, metadata)
    return format_cell(value)


def display_entity_column(plan: QueryPlan, rows: List[Dict[str, Any]], columns: List[str] | None = None) -> str:
    if not rows:
        return ""
    available = columns or list(rows[0].keys())
    for column in available:
        text = str(column or "").lower()
        if text.endswith("_name") or text in {"name", "title"}:
            if any(row.get(column) not in (None, "") for row in rows[:8]):
                return column
    primary = primary_entity_column(plan, rows[0])
    if primary:
        return primary
    for column in available:
        if entity_like_column(column) and any(row.get(column) not in (None, "") for row in rows[:8]):
            return column
    return ""


def entity_kind_label(plan: QueryPlan, column: str) -> str:
    return answer_column_labels(plan).get(str(column or ""), "") or "对象"


def metric_columns_for_rows(plan: QueryPlan, rows: List[Dict[str, Any]], columns: List[str] | None = None) -> List[str]:
    if not rows:
        return []
    available = columns or list(rows[0].keys())
    dimension_columns = {
        str(column or "").strip()
        for intent in plan.intents
        for column in [
            intent.group_by_column,
            intent.filter_column,
            *((intent.metric_resolution or {}).get("entityColumns") or (intent.metric_resolution or {}).get("entity_columns") or []),
        ]
        if str(column or "").strip()
    }
    result: List[str] = []
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        for candidate in [resolution.get("metricKey"), intent.metric_name, intent.metric_column, *(resolution.get("sourceColumns") or [])]:
            text = str(candidate or "").strip()
            if text and text in available and text not in result:
                result.append(text)
    for column in available:
        text = str(column or "").strip()
        if not text or text in result:
            continue
        lower = text.lower()
        if identifier_like_column(lower) or text in dimension_columns:
            continue
        if re.search(r"(name|title|time|date|日期|时间|名称)", lower):
            continue
        if any(answer_numeric_value(row.get(text)) is not None for row in rows[:8]):
            result.append(text)
    return result


def ranking_objective_metric_column(question: str, plan: QueryPlan, columns: List[str]) -> str:
    del question
    available = {str(column) for column in columns or []}
    for candidate in ranking_objective_metric_refs(plan):
        if candidate and candidate in available:
            return candidate
    return ""


def ranking_objective_metric_refs(plan: QueryPlan) -> List[str]:
    candidates: List[str] = []
    understanding = plan.question_understanding or {}
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if isinstance(ranking, dict):
        candidates.extend(
            [
                str(ranking.get("metricRef") or "").strip(),
                str(ranking.get("metric_ref") or "").strip(),
                str(ranking.get("resolvedMetricRef") or "").strip(),
                str(ranking.get("resolved_metric_ref") or "").strip(),
                str(ranking.get("metricKey") or "").strip(),
                str(ranking.get("metric_key") or "").strip(),
                str(ranking.get("metricColumn") or "").strip(),
                str(ranking.get("metric_column") or "").strip(),
            ]
        )
    for intent in plan.intents:
        if intent.answer_mode in {AnswerMode.DERIVED, AnswerMode.TOPN}:
            resolution = intent.metric_resolution or {}
            candidates.extend(
                [
                    str(resolution.get("metricKey") or "").strip(),
                    str(intent.metric_name or "").strip(),
                    str(intent.metric_column or "").strip(),
                ]
            )
    return dedupe_strings(candidates)


def requested_metric_columns_for_rows(plan: QueryPlan, rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    available = {str(column) for column in columns or []}
    requested: List[str] = []
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        for spec in intent.metric_specs:
            if not isinstance(spec, dict):
                continue
            for candidate in metric_spec_candidate_columns(spec):
                text = str(candidate or "").strip()
                if text and text in available and text not in requested:
                    requested.append(text)
        for candidate in [
            resolution.get("metricKey"),
            intent.metric_name,
            intent.metric_column,
            *(resolution.get("sourceColumns") or []),
        ]:
            text = str(candidate or "").strip()
            if text and text in available and text not in requested:
                requested.append(text)
            for alias in answer_metric_aliases_for_intent(intent, text):
                if alias in available and alias not in requested:
                    requested.append(alias)
    if requested:
        return [column for column in requested if column in metric_columns_for_rows(plan, rows, columns)]
    return metric_columns_for_rows(plan, rows, columns)


def answer_metric_aliases_for_intent(intent: QuestionIntent, metric_key: str) -> List[str]:
    resolution = intent.metric_resolution or {}
    alias_map = resolution.get("columnAliases") or resolution.get("column_aliases") or {}
    if isinstance(alias_map, dict):
        values = alias_map.get(metric_key) or []
        if isinstance(values, str):
            return [values]
        if isinstance(values, list):
            return [str(value) for value in values if str(value).strip()]
    return []


def metric_spec_candidate_columns(spec: Dict[str, Any]) -> List[str]:
    source_columns = spec.get("sourceColumns") or spec.get("source_columns") or []
    return [
        str(item)
        for item in [
            spec.get("metricName"),
            spec.get("metric_name"),
            spec.get("metricColumn"),
            spec.get("metric_column"),
            *source_columns,
        ]
        if str(item or "").strip()
    ]


def ranking_metric_columns(question: str, plan: QueryPlan, rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    metrics = requested_metric_columns_for_rows(plan, rows, columns)
    objective = ranking_objective_metric_column(question, plan, columns)
    if objective:
        metrics = [objective] + [column for column in metrics if column != objective]
    return metrics


def ranking_display_columns(question: str, plan: QueryPlan, rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    entity_columns: List[str] = []
    for column in primary_summary_entity_columns(plan):
        if column in columns and column not in entity_columns:
            entity_columns.append(column)
    display_entity = display_entity_column(plan, rows, columns)
    if display_entity and display_entity not in entity_columns:
        entity_columns.append(display_entity)
    metrics = ranking_metric_columns(question, plan, rows, columns)
    selected = entity_columns + [column for column in metrics if column not in entity_columns]
    return selected or columns


def requested_primary_metric_items(plan: QueryPlan) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for intent in plan.intents:
        if intent.answer_mode not in {AnswerMode.METRIC, AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DERIVED}:
            continue
        resolution = intent.metric_resolution or {}
        column = str(resolution.get("metricKey") or intent.metric_name or intent.metric_column or "").strip()
        if not column:
            continue
        label = str(resolution.get("displayName") or "").strip() or friendly_column_label(plan, column)
        if not any(item["column"] == column for item in items):
            items.append(
                {
                    "column": column,
                    "label": label,
                    "taskId": intent.plan_task_id or "",
                    "sourcePhrase": str(resolution.get("sourcePhrase") or ""),
                    "aliases": answer_metric_aliases_for_intent(intent, column),
                }
            )
    return items


def ranking_missing_metric_note(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    displayed_columns: List[str],
) -> str:
    if not run_result:
        return ""
    displayed = set(displayed_columns or [])
    missing = [
        item for item in requested_primary_metric_items(plan)
        if item["column"] not in displayed and not (set(item.get("aliases") or []) & displayed)
    ]
    notes: List[str] = []
    intent_map = intent_by_task_id(plan)
    entity_ranking = any(
        intent.answer_mode in {AnswerMode.TOPN, AnswerMode.DERIVED, AnswerMode.GROUP_AGG}
        and entity_like_column(intent.group_by_column)
        for intent in plan.intents
    )
    available_rows_by_task = {
        item.task_id: item.query_bundle.rows
        for item in visible_successful_tasks(plan, run_result)
        if item.query_bundle.rows
    }
    for item in missing[:3]:
        intent = intent_map.get(item["taskId"])
        if entity_ranking and intent and is_time_series_intent(plan, intent) and not intent.depends_on_task_ids:
            continue
        column = item["column"]
        label = item["label"]
        task_rows = available_rows_by_task.get(item["taskId"], [])
        has_metric = any(column in row and row.get(column) not in (None, "") for row in task_rows[:20])
        if has_metric:
            notes.append("%s已单独返回，但没有按当前排行对象成功合并进主表" % label)
        else:
            notes.append("%s本轮没有返回可合并结果" % label)
    refs = required_evidence_refs(plan.question_understanding or {})
    for field in list(refs.get("fields") or [])[:3]:
        if field in displayed:
            continue
        label = friendly_column_label(plan, field)
        notes.append("%s没有按当前排行对象成功合并进主表" % label)
    coverage = answer_requirement_coverage(question, plan, run_result)
    for item in (coverage.get("missing") or [])[:3]:
        label = str(item.get("label") or "")
        if label and not any(label in note for note in notes):
            notes.append("%s没有返回可合并结果" % label)
    return "；".join(notes) + "。" if notes else ""


def row_sample_sentence(question: str, plan: QueryPlan, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = fallback_display_columns(plan, rows)
    entity_column = display_entity_column(plan, rows, columns)
    metric_columns = metric_columns_for_rows(plan, rows, columns)
    metric_column = metric_columns[0] if metric_columns else primary_answer_metric_column(plan, rows[0])
    if not metric_column:
        return ""
    metric_label = friendly_column_label(plan, metric_column)
    metric_contract = answer_column_display_contracts(plan).get(metric_column)
    if not entity_column:
        values = [
            format_answer_cell(metric_column, row.get(metric_column), metric_label, metric_contract)
            for row in rows[:5]
            if row.get(metric_column) not in (None, "")
        ]
        if not values:
            return ""
        return "%s%s返回 %d 条结果，前几项为：%s。" % (
            answer_time_prefix(question),
            metric_label,
            len(rows),
            "、".join(values[:5]),
        )
    entity_label = friendly_column_label(plan, entity_column)
    parts: List[str] = []
    for row in rows[:5]:
        entity_value = row.get(entity_column)
        metric_value = row.get(metric_column)
        if entity_value in (None, "") or metric_value in (None, ""):
            continue
        parts.append(
            "%s %s：%s"
            % (
                entity_label,
                format_cell(entity_value),
                format_answer_cell(metric_column, metric_value, metric_label, metric_contract),
            )
        )
    if not parts:
        return ""
    return "%s%s按%s返回 %d 条结果，前几项为：%s。" % (
        answer_time_prefix(question),
        metric_label,
        entity_kind_label(plan, entity_column),
        len(rows),
        "；".join(parts),
    )


def single_row_detail_sentence(question: str, plan: QueryPlan, row: Dict[str, Any]) -> str:
    if not row:
        return ""
    entity_column = primary_entity_column(plan, row)
    labels = answer_column_labels(plan)
    contracts = answer_column_display_contracts(plan)
    parts: List[str] = []
    for column, value in row.items():
        if value in (None, "") or column == entity_column:
            continue
        text = str(column or "").strip()
        if identifier_like_column(text) and not labels.get(text):
            continue
        label = labels.get(text) or friendly_column_label(plan, text)
        parts.append("%s为 %s" % (label, format_answer_cell(text, value, label, contracts.get(text))))
        if len(parts) >= 4:
            break
    if not parts:
        return ""
    if entity_column:
        return "%s%s %s，%s。" % (
            answer_time_prefix(question),
            friendly_column_label(plan, entity_column),
            format_cell(row.get(entity_column)),
            "，".join(parts),
        )
    return "%s%s。" % (answer_time_prefix(question), "，".join(parts))


def generic_result_overview_sentence(
    question: str,
    plan: QueryPlan,
    rows: List[Dict[str, Any]],
    run_result: AgentRunResult | None,
) -> str:
    if not rows:
        return ""
    metric_labels = dedupe_strings(
        [
            str((intent.metric_resolution or {}).get("displayName") or friendly_column_label(plan, (intent.metric_resolution or {}).get("metricKey") or intent.metric_name or intent.metric_column))
            for intent in plan.intents
            if ((intent.metric_resolution or {}).get("metricKey") or intent.metric_name or intent.metric_column)
        ]
    )
    entity_labels = dedupe_strings(
        [
            entity_kind_label(plan, column)
            for column in [intent.group_by_column or intent.filter_column for intent in plan.intents]
            if column
        ]
    )
    if not metric_labels:
        metric_labels = [friendly_column_label(plan, column) for column in metric_columns_for_rows(plan, rows, fallback_display_columns(plan, rows))[:3]]
    metric_text = "、".join(metric_labels[:3]) if metric_labels else "数据"
    entity_text = "，按%s维度" % "、".join(entity_labels[:2]) if entity_labels else ""
    task_count = len(visible_successful_tasks(plan, run_result)) if run_result else 0
    task_text = "，覆盖 %d 个证据节点" % task_count if task_count > 1 else ""
    return "%s%s已返回 %d 条结果%s%s。" % (
        answer_time_prefix(question),
        metric_text,
        len(rows),
        entity_text,
        task_text,
    )


def markdown_table(
    rows: List[Dict[str, Any]],
    columns: List[str],
    labels: Dict[str, str] | None = None,
    contracts: Dict[str, Dict[str, Any]] | None = None,
) -> str:
    labels = labels or {}
    contracts = contracts or {}
    header = "| %s |" % " | ".join(str(labels.get(column) or humanize_column_name(column)) for column in columns)
    divider = "| %s |" % " | ".join("---" for _ in columns)
    body = []
    for row in rows:
        body.append(
            "| %s |"
            % " | ".join(
                format_answer_cell(
                    column,
                    row.get(column, ""),
                    labels.get(column) or humanize_column_name(column),
                    contracts.get(column),
                )
                for column in columns
            )
        )
    return "\n".join([header, divider] + body)


def business_summary_snapshot(plan: QueryPlan, run_result: AgentRunResult) -> tuple[List[Dict[str, Any]], List[str]]:
    succeeded = [item for item in run_result.task_results if not item.query_bundle.failed and item.query_bundle.rows]
    if len(succeeded) <= 1:
        return [], []
    intent_by_task = {intent.plan_task_id: intent for intent in plan.intents}
    ordered = sorted(succeeded, key=lambda item: task_evidence_priority(intent_by_task.get(item.task_id), item, plan))
    support_ids = support_task_ids_for_answer(plan)
    visible = [item for item in ordered if answer_visible_task(intent_by_task.get(item.task_id), item, plan, support_ids)]
    if len(visible) <= 1:
        return [], []
    base = first_entity_summary_task(visible, intent_by_task, plan)
    if not base:
        return [], []
    merged_rows = merge_visible_task_rows(base, [item for item in visible if item.task_id != base.task_id])
    if not merged_rows:
        return [], []
    columns = business_summary_columns(plan, merged_rows)
    if not columns:
        return [], []
    return merged_rows, columns


def business_summary_table(plan: QueryPlan, run_result: AgentRunResult) -> str:
    rows, columns = business_summary_snapshot(plan, run_result)
    if not rows or not columns:
        return ""
    labels = answer_column_labels(plan)
    return markdown_table(rows[:8], columns, labels, answer_column_display_contracts(plan))


def deterministic_ranking_answer(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result or not re.search(r"top|前\s*\d+|最高|最多|排行|排名", str(question or ""), flags=re.I):
        return ""
    if not any(intent.answer_mode in {AnswerMode.TOPN, AnswerMode.DERIVED, AnswerMode.GROUP_AGG} for intent in plan.intents):
        return ""
    rows, columns = business_summary_snapshot(plan, run_result)
    if not rows or not columns:
        intent_map = intent_by_task_id(plan)
        for task in visible_successful_tasks(plan, run_result):
            intent = intent_map.get(task.task_id)
            candidate_rows = task.query_bundle.rows or []
            if not candidate_rows or not intent or intent.answer_mode not in {AnswerMode.TOPN, AnswerMode.DERIVED, AnswerMode.GROUP_AGG}:
                continue
            if not (entity_like_column(intent.group_by_column) or any(entity_like_column(key) for key in candidate_rows[0].keys())):
                continue
            rows = candidate_rows
            columns = business_summary_columns(plan, rows) or fallback_display_columns(plan, rows)
            break
    if not rows or not columns:
        return ""
    columns = ranking_display_columns(question, plan, rows, columns)
    table = markdown_table(
        rows[:8],
        columns,
        answer_column_labels(plan),
        answer_column_display_contracts(plan),
    )
    entity_column = display_entity_column(plan, rows, columns)
    metrics = ranking_metric_columns(question, plan, rows, columns)
    metric_labels = [friendly_column_label(plan, column) for column in metrics[:3]]
    missing_note = ranking_missing_metric_note(question, plan, run_result, columns)
    note_text = "\n\n说明：%s" % missing_note if missing_note else ""
    if entity_column and metric_labels:
        intro = "%s按%s排序的%s如下" % (
            answer_time_prefix(question),
            metric_labels[0],
            entity_kind_label(plan, entity_column),
        )
        if len(metric_labels) > 1:
            intro += "，并补充%s" % "、".join(metric_labels[1:3])
        return "%s：\n\n%s%s" % (intro, table, note_text)
    return "%s结果如下：\n\n%s%s" % (answer_time_prefix(question), table, note_text)


def first_entity_summary_task(items: List[Any], intent_by_task: Dict[str, QuestionIntent], plan: QueryPlan | None = None) -> Any | None:
    objective_refs = ranking_objective_metric_refs(plan) if plan else []
    for metric_ref in objective_refs:
        for item in items:
            intent = intent_by_task.get(item.task_id)
            rows = item.query_bundle.rows if item.query_bundle else []
            if not rows or metric_ref not in rows[0]:
                continue
            if intent and intent.answer_mode in {AnswerMode.DERIVED, AnswerMode.TOPN, AnswerMode.GROUP_AGG} and (
                entity_like_column(intent.group_by_column) or any(entity_like_column(key) for key in rows[0].keys())
            ):
                return item
    for item in items:
        intent = intent_by_task.get(item.task_id)
        if intent and intent.answer_mode in {AnswerMode.DERIVED, AnswerMode.TOPN, AnswerMode.GROUP_AGG} and entity_like_column(intent.group_by_column):
            return item
    for item in items:
        if any(entity_like_column(key) for key in (item.query_bundle.rows[0].keys() if item.query_bundle.rows else [])):
            return item
    return items[0] if items else None


def merge_visible_task_rows(base_item: Any, other_items: List[Any]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = [dict(row) for row in base_item.query_bundle.rows[:20]]
    for _ in range(2):
        changed = False
        for item in other_items:
            rows = item.query_bundle.rows or []
            for target in merged:
                match = first_matching_row(target, rows)
                if not match:
                    continue
                for key, value in match.items():
                    if key not in target or target.get(key) in (None, ""):
                        target[key] = value
                        changed = True
                        continue
                    if target.get(key) != value:
                        base_alias = merge_conflict_column_name(base_item, key)
                        alias = merge_conflict_column_name(item, key)
                        if base_alias and key in target and base_alias not in target:
                            target[base_alias] = target.pop(key)
                            changed = True
                        if alias and alias not in target:
                            target[alias] = value
                            changed = True
        if not changed:
            break
    return merged


def merge_conflict_column_name(item: Any, column: str) -> str:
    text = str(column or "").strip()
    tables = {str(table or "") for table in getattr(item.query_bundle, "tables", [])}
    table = next(iter(sorted(tables)), "")
    prefix = re.sub(r"[^A-Za-z0-9_]+", "_", table).strip("_")
    return "%s__%s" % (prefix, text) if prefix and text else ""


def first_matching_row(base: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    base_has_entity = any(summary_business_entity_key(key) for key in base)
    entity_keys = dedupe_strings(
        [
            str(column or "").rsplit("__", 1)[-1]
            for column in base
            if summary_business_entity_key(str(column or "").rsplit("__", 1)[-1])
        ]
    )
    for key in entity_keys:
        base_values = row_entity_values(base, key)
        if not base_values:
            continue
        for row in rows:
            row_values = row_entity_values(row, key)
            if row_values and set(base_values) & set(row_values):
                return row
    if not base_has_entity and len(rows) == 1:
        return rows[0]
    return None


def row_entity_values(row: Dict[str, Any], key: str) -> List[str]:
    suffix = "__%s" % key
    values: List[str] = []
    for column, value in row.items():
        column_text = str(column or "")
        if column_text != key and not column_text.endswith(suffix):
            continue
        normalized = normalized_cell(value)
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def summary_business_entity_key(column: str) -> bool:
    return entity_like_column(column)


def normalized_cell(value: Any) -> str:
    return str(value or "").strip()


def business_summary_columns(plan: QueryPlan, rows: List[Dict[str, Any]]) -> List[str]:
    available = set()
    for row in rows[:8]:
        available.update(str(key) for key in row.keys())
    preferred: List[str] = []
    entity_column_list = primary_summary_entity_columns(plan)
    entity_columns = set(entity_column_list)
    for column in entity_column_list:
        if column in available:
            preferred.append(column)
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        for column in [resolution.get("metricKey"), intent.metric_name]:
            if column and column in available and column not in preferred:
                preferred.append(column)
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        for column in [resolution.get("metricKey"), *(resolution.get("sourceColumns") or [])]:
            text = str(column or "").strip()
            if text and text in available and text not in preferred and summary_column_allowed(text, entity_columns):
                preferred.append(text)
    for intent in plan.intents:
        for column in [intent.group_by_column] + intent.output_keys:
            if column and column in available and column not in preferred and summary_column_allowed(column, entity_columns):
                preferred.append(column)
    for column in rows[0].keys():
        text = str(column)
        if text not in preferred and summary_column_allowed(text, entity_columns):
            preferred.append(text)
        if len(preferred) >= 10:
            break
    return preferred[:10]


def summary_column_allowed(column: str, entity_columns: set[str]) -> bool:
    text = str(column or "").strip()
    if not text:
        return False
    if entity_like_column(text) and text not in entity_columns:
        return False
    return True


def primary_summary_entity_columns(plan: QueryPlan) -> List[str]:
    understanding = plan.question_understanding or {}
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    group_by = str(ranking.get("groupByColumn") or ranking.get("group_by_column") or "").strip() if isinstance(ranking, dict) else ""
    if not group_by:
        for intent in plan.intents:
            resolution = intent.metric_resolution or {}
            if str(resolution.get("computeStrategy") or "") == "projection_group_aggregate" and entity_like_column(intent.group_by_column):
                group_by = intent.group_by_column
                break
    for intent in plan.intents:
        if not group_by and intent.answer_mode in {AnswerMode.DERIVED, AnswerMode.TOPN, AnswerMode.GROUP_AGG} and entity_like_column(intent.group_by_column):
            group_by = intent.group_by_column
            break
    candidates: List[str] = [group_by] if group_by else []
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        candidates.extend(
            str(column)
            for column in resolution.get("entityColumns") or resolution.get("entity_columns") or []
            if str(column).strip()
        )
        candidates.extend(
            str(column)
            for column in [intent.group_by_column, intent.filter_column, *intent.output_keys]
            if entity_like_column(column)
        )
    return dedupe_strings([column for column in candidates if entity_like_column(column)])


def answer_column_labels(plan: QueryPlan) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        metric = str(resolution.get("metricKey") or intent.metric_name or intent.metric_column or "").strip()
        display = str(resolution.get("displayName") or "").strip()
        if metric and display:
            labels[metric] = display
            table = re.sub(r"[^A-Za-z0-9_]+", "_", str(intent.preferred_table or "")).strip("_")
            if table:
                labels["%s__%s" % (table, metric)] = display
        source_labels = resolution.get("sourceColumnLabels") or resolution.get("source_column_labels") or {}
        if isinstance(source_labels, dict):
            for column, label in source_labels.items():
                column_text = str(column or "").strip()
                label_text = str(label or "").strip()
                if column_text and label_text:
                    labels[column_text] = label_text
        for spec in intent.metric_specs:
            if not isinstance(spec, dict):
                continue
            spec_label = direct_metric_spec_label(spec) or str(resolution.get("displayName") or "").strip()
            for candidate in metric_spec_candidate_columns(spec):
                text = str(candidate or "").strip()
                if text and spec_label:
                    labels[text] = spec_label
                    table = re.sub(r"[^A-Za-z0-9_]+", "_", str(intent.preferred_table or "")).strip("_")
                    if table:
                        labels["%s__%s" % (table, text)] = spec_label
        if intent.group_by_column and intent.group_by_name:
            labels[intent.group_by_column] = intent.group_by_name
        for column in [intent.group_by_column, intent.filter_column, *intent.output_keys, *intent.required_evidence]:
            text = str(column or "").strip()
            default = default_answer_column_label(text)
            if text and default:
                labels.setdefault(text, default)
    for evidence in (plan.question_understanding or {}).get("requiredEvidenceIntents") or (plan.question_understanding or {}).get("required_evidence_intents") or []:
        if not isinstance(evidence, dict):
            continue
        for field in evidence.get("suggestedFields") or evidence.get("suggested_fields") or []:
            text = str(field or "").strip()
            default = default_answer_column_label(text)
            if text and default:
                labels.setdefault(text, default)
    return labels


def answer_column_display_contracts(plan: QueryPlan) -> Dict[str, Dict[str, Any]]:
    contracts: Dict[str, Dict[str, Any]] = {}
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        contract = metric_display_contract(intent)
        candidates = [
            resolution.get("metricKey"),
            intent.metric_name,
            intent.metric_column,
            *(resolution.get("sourceColumns") or []),
        ]
        source_labels = resolution.get("sourceColumnLabels") or resolution.get("source_column_labels") or {}
        if isinstance(source_labels, dict):
            candidates.extend(source_labels.keys())
        for candidate in candidates:
            column = str(candidate or "").strip()
            if column and contract:
                contracts[column] = contract
        for spec in intent.metric_specs:
            if not isinstance(spec, dict):
                continue
            spec_contract = metric_display_contract(intent, spec)
            for candidate in metric_spec_candidate_columns(spec):
                column = str(candidate or "").strip()
                if column and spec_contract:
                    contracts[column] = spec_contract
    return contracts


def direct_metric_spec_label(spec: Dict[str, Any]) -> str:
    for key in ["displayName", "display_name", "naturalName", "natural_name", "label", "metricLabel", "metric_label", "sourcePhrase", "source_phrase"]:
        label = str(spec.get(key) or "").strip()
        if label:
            return label
    return ""


def task_evidence_sections(plan: QueryPlan, run_result: AgentRunResult) -> str:
    succeeded = [item for item in run_result.task_results if not item.query_bundle.failed]
    if len(succeeded) <= 1:
        return ""
    intent_by_task = {intent.plan_task_id: intent for intent in plan.intents}
    ordered = sorted(succeeded, key=lambda item: task_evidence_priority(intent_by_task.get(item.task_id), item, plan))
    support_ids = support_task_ids_for_answer(plan)
    visible = [item for item in ordered if answer_visible_task(intent_by_task.get(item.task_id), item, plan, support_ids)]
    if not visible:
        visible = ordered
    lines = ["分节点证据："]
    for item in visible[:6]:
        bundle = item.query_bundle
        if not bundle.rows:
            lines.append("- %s：执行成功但返回 0 行。" % item.task_id)
            continue
        intent = intent_by_task.get(item.task_id)
        title = task_evidence_title(intent, item, plan)
        tables = "、".join(bundle.tables) if bundle.tables else (intent.preferred_table if intent else "")
        location = tables or ("派生计算" if intent and intent.answer_mode == AnswerMode.DERIVED else "unknown_table")
        lines.append("- %s（%s）：%s 行。" % (title, location, bundle.effective_row_count()))
        section_plan = QueryPlan(intents=[intent]) if intent else plan
        columns = fallback_display_columns(section_plan, bundle.rows)
        if columns:
            lines.append(
                markdown_table(
                    bundle.rows[:4],
                    columns,
                    answer_column_labels(section_plan),
                    answer_column_display_contracts(section_plan),
                )
            )
    failed = [item for item in run_result.task_results if item.query_bundle.failed]
    for item in failed[:3]:
        lines.append("- %s：执行失败，%s" % (item.task_id, (item.query_bundle.error or item.summary)[:160]))
    return "\n".join(lines)


def task_evidence_title(intent: QuestionIntent | None, item: Any, plan: QueryPlan | None = None) -> str:
    if not intent:
        return str(getattr(item, "task_id", "") or "查询结果")
    resolution = intent.metric_resolution or {}
    display = str(resolution.get("displayName") or "").strip()
    metric = str(resolution.get("metricKey") or intent.metric_name or "").strip()
    if str(resolution.get("computeStrategy") or "") == "projection_group_aggregate":
        group = intent.group_by_column or "entity"
        return "%s（按 %s 汇总）" % (display or metric or "派生指标", group)
    if display:
        return display
    if metric:
        return metric
    if plan:
        required = required_evidence_refs(plan.question_understanding or {})
        matched_fields = [field for field in intent.output_keys if field in required["fields"]]
        if matched_fields:
            return friendly_column_label(plan, matched_fields[0])
    if intent.preferred_table:
        return intent.preferred_table
    return str(getattr(item, "task_id", "") or "查询结果")


def support_task_ids_for_answer(plan: QueryPlan) -> set[str]:
    support: set[str] = set()
    for intent in plan.intents:
        task_id = intent.plan_task_id
        if not task_id:
            continue
        resolution = intent.metric_resolution or {}
        if task_id.startswith("component_") or str(resolution.get("sourcePhrase") or "").startswith("semantic formula dependency"):
            support.add(task_id)
        if str(resolution.get("computeStrategy") or "") == "projection_group_aggregate":
            for key in ["sourceMetricTaskId", "bridgeTaskId"]:
                value = str(resolution.get(key) or "").strip()
                if value:
                    support.add(value)
    return support


def answer_visible_task(intent: QuestionIntent | None, item: Any, plan: QueryPlan, support_ids: set[str]) -> bool:
    task_id = str(getattr(item, "task_id", "") or "")
    if task_id in support_ids:
        return False
    if not intent:
        return True
    resolution = intent.metric_resolution or {}
    source_phrase = str(resolution.get("sourcePhrase") or "").strip()
    metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
    if should_hide_alternate_metric(plan, intent):
        return False
    if metric_key and source_phrase and not source_phrase_in_question(source_phrase, intent.question):
        refs = required_evidence_refs(plan.question_understanding or {})
        ranking_refs = ranking_metric_refs(plan.question_understanding or {})
        if metric_key not in refs["metrics"] and metric_key not in ranking_refs:
            return False
    return True


def should_hide_alternate_metric(plan: QueryPlan, intent: QuestionIntent | None) -> bool:
    if not intent:
        return False
    resolution = intent.metric_resolution or {}
    if bool(
        resolution.get("internalOnly")
        or resolution.get("supportOnly")
        or str(resolution.get("displayRole") or "") == "support"
        or str(resolution.get("sourcePhrase") or "").startswith("semantic formula dependency")
    ):
        return True
    requested = question_understanding_metric_refs(plan.question_understanding or {})
    metric_key = intent_metric_key(intent)
    if not requested or not metric_key or metric_key in requested:
        return False
    requested_phrases: set[str] = set()
    for candidate in plan.intents:
        if intent_metric_key(candidate) not in requested:
            continue
        candidate_resolution = candidate.metric_resolution or {}
        requested_phrases.update(
            normalized_metric_phrase(value)
            for value in [candidate_resolution.get("displayName"), candidate_resolution.get("sourcePhrase")]
            if normalized_metric_phrase(value)
        )
    current_phrases = {
        normalized_metric_phrase(value)
        for value in [resolution.get("displayName"), resolution.get("sourcePhrase")]
        if normalized_metric_phrase(value)
    }
    return bool(requested_phrases & current_phrases)


def normalized_metric_phrase(value: Any) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "")).strip().lower()


def intent_metric_key(intent: QuestionIntent | None) -> str:
    if not intent:
        return ""
    resolution = intent.metric_resolution or {}
    return str(resolution.get("metricKey") or intent.metric_name or intent.metric_column or "").strip()


def source_phrase_in_question(source_phrase: str, question: str) -> bool:
    phrase = str(source_phrase or "").strip().lower()
    text = str(question or "").strip().lower()
    if not phrase or not text:
        return True
    return phrase in text


def task_evidence_priority(intent: QuestionIntent | None, item: Any, plan: QueryPlan) -> int:
    score = 100
    task_id = str(getattr(item, "task_id", "") or "")
    if not intent:
        return score
    resolution = intent.metric_resolution or {}
    source_phrase = str(resolution.get("sourcePhrase") or "").strip().lower()
    requested_metrics = question_understanding_metric_refs(plan.question_understanding or {})
    required_refs = required_evidence_refs(plan.question_understanding or {})
    metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
    if str(resolution.get("computeStrategy") or "") == "projection_group_aggregate":
        score -= 60
    if metric_key and metric_key in requested_metrics:
        score -= 45
    if metric_key and metric_key in required_refs["metrics"]:
        score -= 35
    if intent.preferred_table and intent.preferred_table in required_refs["tables"]:
        score -= 30
    if set(intent.output_keys or []) & required_refs["fields"]:
        score -= 30
    if intent.answer_mode == AnswerMode.DERIVED:
        score -= 15
    if task_id.startswith("component_") or source_phrase.startswith("semantic formula dependency"):
        score += 70
    if "bridge" in task_id and not (set(intent.output_keys or []) & required_refs["fields"]) and not metric_key:
        score += 45
    return score


def ranking_metric_refs(understanding: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    if not isinstance(ranking, dict):
        return refs
    for key in ["resolvedMetricRef", "metricRef"]:
        value = str(ranking.get(key) or "").strip()
        if value:
            refs.add(value)
    return refs


def question_understanding_metric_refs(understanding: Dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    refs.update(ranking_metric_refs(understanding))
    for item in understanding.get("requestedMeasures") or understanding.get("requested_measures") or []:
        if not isinstance(item, dict):
            continue
        for key in ["resolvedMetricRef", "metricRef"]:
            value = str(item.get(key) or "").strip()
            if value:
                refs.add(value)
    return refs


def required_evidence_refs(understanding: Dict[str, Any]) -> Dict[str, set[str]]:
    refs = {"metrics": set(), "tables": set(), "fields": set()}
    for item in understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []:
        if not isinstance(item, dict):
            continue
        for key, target in [
            ("suggestedMetricRefs", "metrics"),
            ("suggested_metric_refs", "metrics"),
            ("suggestedTables", "tables"),
            ("suggested_tables", "tables"),
            ("suggestedFields", "fields"),
            ("suggested_fields", "fields"),
        ]:
            values = item.get(key) or []
            if isinstance(values, str):
                values = [values]
            refs[target].update(str(value).strip() for value in values if str(value or "").strip())
    return refs


def analysis_summary_required(plan: QueryPlan) -> bool:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip().lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    if rule_evidence_only_analysis(understanding, analysis_intent):
        return False
    if entity_ranking_answer_only(plan, analysis_intent):
        return False
    if analysis_intent == "overview" and single_metric_overview(plan):
        return False
    grain = str(understanding.get("analysisGrain") or understanding.get("analysis_grain") or "").strip().lower()
    if analysis_intent == "overview":
        return requires_explanation and grain in {"merchant", "day"}
    return requires_explanation or (analysis_intent and analysis_intent != "none")


def deterministic_single_semantic_metric_answer(plan: QueryPlan) -> bool:
    # A cross-domain/detail graph is not a single-metric answer merely because
    # it contains only one metric node. Returning the metric spine alone would
    # discard explicitly requested lifecycle/detail evidence.
    if any(
        intent.intent_type == "VALID" and intent.answer_mode == AnswerMode.DETAIL
        for intent in plan.intents
    ):
        return False
    metric_intents = [
        intent
        for intent in plan.intents
        if intent.answer_mode in {AnswerMode.METRIC, AnswerMode.GROUP_AGG}
        and not should_hide_alternate_metric(plan, intent)
    ]
    if not metric_intents:
        return False
    semantic_refs = {
        str((intent.metric_resolution or {}).get("semanticRefId") or "")
        for intent in metric_intents
        if str((intent.metric_resolution or {}).get("semanticRefId") or "").startswith("semantic:")
    }
    return len(semantic_refs) == 1 and all(
        str((intent.metric_resolution or {}).get("semanticRefId") or "") in semantic_refs
        for intent in metric_intents
    )


def trusted_single_metric_verified_answer(plan: QueryPlan, run_result: AgentRunResult | None) -> bool:
    if not run_result or not deterministic_single_semantic_metric_answer(plan):
        return False
    verified = run_result.verified_evidence
    if not verified or not verified.passed or verified.blocking_gaps or run_result.evidence_gaps:
        return False
    tasks = visible_successful_tasks(plan, run_result)
    if len(tasks) != 1:
        return False
    rows = tasks[0].query_bundle.rows
    if len(rows) != 1:
        return False
    return bool(summary_metric_values(plan, run_result))


def answer_skill_required(
    plan: QueryPlan,
    run_result: AgentRunResult | None = None,
    has_rule_context: bool = False,
    skill_headers: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    return bool(select_answer_skill(plan, run_result, has_rule_context, skill_headers=skill_headers))


def deterministic_analysis_skill_fallback(
    plan: QueryPlan,
    run_result: AgentRunResult | None = None,
    has_rule_context: bool = False,
    skill_headers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Return only an explicitly declared runtime skill; never infer a business workflow."""
    return select_answer_skill(plan, run_result, has_rule_context, skill_headers=skill_headers)


def select_answer_skill(
    plan: QueryPlan,
    run_result: AgentRunResult | None = None,
    has_rule_context: bool = False,
    skill_headers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    understanding = plan.question_understanding or {}
    if skill_route_explicit_no_match(understanding):
        return ""
    del run_result, has_rule_context
    headers = skill_headers if skill_headers is not None else configured_answer_skill_headers()
    allowed = {str(item.get("name") or "").strip() for item in headers if str(item.get("name") or "").strip()}
    return declared_skill_workflow_name(understanding, allowed)


def deterministic_analysis_summary(question: str, plan: QueryPlan, run_result: AgentRunResult) -> str:
    if not analysis_summary_required(plan):
        return ""
    rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
    if not rows:
        return ""
    metric_labels: List[str] = []
    for intent in plan.intents:
        label = str((intent.metric_resolution or {}).get("displayName") or "").strip()
        if label and label not in metric_labels:
            metric_labels.append(label)
    subject = "、".join(metric_labels[:4]) or "当前指标"
    row_count = len(rows)
    return "%s已有 %d 行已验证数据，可基于当前结果做趋势或异常判断；证据不足的原因不要扩展为确定结论。" % (subject, row_count)


def declared_skill_workflow_name(understanding: Dict[str, Any], allowed_skill_names: set[str]) -> str:
    for key in [
        "skillWorkflow",
        "skill_workflow",
        "recommendedSkill",
        "recommended_skill",
        "analysisSkill",
        "analysis_skill",
        "skillMatch",
        "skill_match",
        "skillRoute",
        "skill_route",
    ]:
        value = understanding.get(key)
        if isinstance(value, dict):
            if skill_route_payload_explicit_no_match(value):
                continue
            name = str(value.get("skillName") or value.get("skill_name") or value.get("name") or "").strip()
        else:
            name = str(value or "").strip()
        if name in allowed_skill_names:
            return name
    return ""


def skill_route_explicit_no_match(understanding: Dict[str, Any]) -> bool:
    route_keys = [
        "skillWorkflow",
        "skill_workflow",
        "recommendedSkill",
        "recommended_skill",
        "analysisSkill",
        "analysis_skill",
        "skillMatch",
        "skill_match",
        "skillRoute",
        "skill_route",
    ]
    return any(
        skill_route_payload_explicit_no_match(understanding.get(key))
        for key in route_keys
        if isinstance(understanding.get(key), dict)
    )


def skill_route_payload_explicit_no_match(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    name_keys = ["skillName", "skill_name", "name"]
    if any(key in payload for key in name_keys):
        name = str(next((payload.get(key) for key in name_keys if key in payload), "") or "").strip()
        if not name or normalize_skill_match_status(name) in SKILL_NO_MATCH_STATUSES:
            return True
    for key in ["status", "decision", "outcome", "matchStatus", "match_status"]:
        if normalize_skill_match_status(payload.get(key)) in SKILL_NO_MATCH_STATUSES:
            return True
    for key in ["applicable", "matched", "enabled"]:
        if key in payload and not boolish(payload.get(key)):
            return True
    return False


def plan_has_ratio_calculation(plan: QueryPlan) -> bool:
    understanding = plan.question_understanding or {}
    for item in understanding.get("calculationIntents") or understanding.get("calculation_intents") or []:
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation") or "").lower()
        if operation in {"ratio", "percentage", "share", "rate"}:
            return True
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        if resolution.get("computeStrategy") == "scope_event_ratio":
            return True
        formula = str(resolution.get("formula") or intent.metric_formula or "")
        if "/" in formula and intent.answer_mode == AnswerMode.DERIVED:
            return True
    return False


def plan_requires_rule_evidence(plan: QueryPlan) -> bool:
    if any(intent.answer_mode == AnswerMode.RULE for intent in plan.intents):
        return True
    understanding = plan.question_understanding or {}
    for item in understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("semanticLabel") or item.get("semantic_label") or "").lower()
        domains = {str(domain or "").lower() for domain in item.get("suggestedDomains") or item.get("suggested_domains") or []}
        if "rule" in label or "规则" in label or domains & {"rule", "rules", "governance", "platform_rule"}:
            return True
    return False


def single_metric_overview(plan: QueryPlan) -> bool:
    executable = [intent for intent in plan.intents if intent.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT, AnswerMode.INVALID}]
    if len(executable) != 1:
        return False
    intent = executable[0]
    if intent.answer_mode != AnswerMode.METRIC:
        return False
    group_by = (intent.group_by_column or "").strip()
    if not group_by:
        return True
    resolution = intent.metric_resolution or {}
    scope_columns = {
        str(column or "").strip()
        for column in resolution.get("scopeColumns") or resolution.get("scope_columns") or []
        if str(column or "").strip()
    }
    return group_by in scope_columns


def entity_ranking_answer_only(plan: QueryPlan, analysis_intent: str) -> bool:
    if analysis_intent not in {"risk_ranking", "ranking", "topn"}:
        return False
    executable = [intent for intent in plan.intents if intent.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT, AnswerMode.INVALID}]
    if not executable:
        return False
    if any(intent.answer_mode in {AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DERIVED} and entity_like_column(intent.group_by_column) for intent in executable):
        return True
    return any(intent.answer_mode == AnswerMode.DETAIL and entity_like_output_keys(intent.output_keys) for intent in executable)


def entity_like_output_keys(output_keys: List[str]) -> bool:
    return any(entity_like_column(key) for key in output_keys or [])


def entity_like_column(column: str | None) -> bool:
    text = str(column or "").strip().lower()
    if not text:
        return False
    return identifier_like_column(text) or text == "id" or text.endswith("_name")


def rule_evidence_only_analysis(understanding: Dict[str, Any], analysis_intent: str) -> bool:
    evidence_items = understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []
    if not evidence_items:
        return False
    rule_tokens = {"rule", "rules", "policy", "governance", "platform", "规则", "治理", "平台"}
    has_rule_item = False
    non_rule_items = 0
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("semanticLabel") or item.get("semantic_label") or "").lower()
        domains = [str(domain).lower() for domain in (item.get("suggestedDomains") or item.get("suggested_domains") or [])]
        text = " ".join([label, *domains])
        is_rule = any(token in text for token in rule_tokens)
        has_rule_item = has_rule_item or is_rule
        if not is_rule:
            non_rule_items += 1
    return has_rule_item and non_rule_items == 0 and analysis_intent in {"none", "overview", "diagnosis"}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def answer_data_package(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    rule_context: str = "",
    merchant: MerchantInfo | None = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = verified_answer_context(
        question,
        plan,
        run_result,
        rule_context=rule_context,
        merchant=merchant,
        personalization_context=personalization_context,
    ).prompt_payload()
    metric_facts = answer_metric_facts(question, plan, run_result)
    if metric_facts:
        payload["metricFacts"] = metric_facts
    comparison_facts = [fact for fact in metric_facts if fact.get("comparisonValue") not in (None, "")]
    if comparison_facts:
        payload["metricComparisonFacts"] = comparison_facts
    return payload


def verified_answer_context(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    rule_context: str = "",
    merchant: MerchantInfo | None = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> VerifiedAnswerContext:
    if not run_result:
        return VerifiedAnswerContext(
            question=question,
            business_context=answer_business_context(question, plan, run_result, merchant, personalization_context),
            rule_evidence=compact_rule_evidence(question, rule_context),
        )
    verified = run_result.verified_evidence
    facts = build_verified_facts(plan, run_result)
    run_result.verified_facts = facts
    return VerifiedAnswerContext(
        question=question,
        business_context=answer_business_context(question, plan, run_result, merchant, personalization_context),
        tables=run_result.merged_query_bundle.tables,
        row_count=run_result.merged_query_bundle.effective_row_count(),
        data_rows=answer_data_rows(plan, run_result),
        data_sections=answer_prompt_sections(plan, run_result),
        metric_disclosures=metric_disclosures(plan, verified),
        lightweight_metric_disclosures=lightweight_metric_disclosures(question, plan, run_result),
        evidence_gaps=compact_evidence_gaps(run_result.evidence_gaps),
        degraded_reasons=list(run_result.degraded_reasons or [])[:12],
        rule_evidence=compact_rule_evidence(question, rule_context),
        verified_passed=bool(verified.passed),
        partial_answer_reason=run_result.partial_answer_reason or verified.partial_answer_reason,
        verified_facts=facts,
    )


def answer_data_rows(plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    if task_metric_rows_better_for_answer(plan, run_result):
        rows = prioritized_task_metric_rows(plan, run_result)
        if rows:
            return rows[:40]
    return run_result.merged_query_bundle.rows[:40]


def task_metric_rows_better_for_answer(plan: QueryPlan, run_result: AgentRunResult) -> bool:
    visible = visible_successful_tasks(plan, run_result)
    metric_tasks = []
    for item in visible:
        intent = intent_by_task_id(plan).get(item.task_id)
        if not intent:
            continue
        resolution = intent.metric_resolution or {}
        metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
        if metric_key and intent.answer_mode in {AnswerMode.GROUP_AGG, AnswerMode.TOPN, AnswerMode.DERIVED}:
            metric_tasks.append(intent)
    if len(metric_tasks) < 2:
        return False
    group_columns = {str(intent.group_by_column or "").strip() for intent in metric_tasks if str(intent.group_by_column or "").strip()}
    return len(group_columns) <= 1


def prioritized_task_metric_rows(plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in visible_successful_tasks(plan, run_result):
        intent = intent_by_task_id(plan).get(item.task_id)
        if not intent or not item.query_bundle.rows:
            continue
        resolution = intent.metric_resolution or {}
        metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
        display = str(resolution.get("displayName") or metric_key or item.task_id).strip()
        if not metric_key and intent.answer_mode != AnswerMode.DERIVED:
            continue
        for row in item.query_bundle.rows[:8]:
            enriched = dict(row)
            enriched.setdefault("__taskId", item.task_id)
            enriched.setdefault("__metricKey", metric_key)
            enriched.setdefault("__metricName", display)
            enriched.setdefault("__resultRole", answer_result_role(intent))
            enriched.setdefault("__groupByColumn", intent.group_by_column or "")
            rows.append(enriched)
    return rows


def answer_prompt_sections(plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for item in visible_successful_tasks(plan, run_result):
        intent = intent_by_task_id(plan).get(item.task_id)
        if not intent or not item.query_bundle.rows:
            continue
        role = answer_result_role(intent)
        sections.append(
            {
                "taskId": item.task_id,
                "resultRole": role,
                "title": section_title_for_intent(plan, intent, item.task_id),
                "metricKey": (intent.metric_resolution or {}).get("metricKey") or intent.metric_name,
                "metricName": (intent.metric_resolution or {}).get("displayName") or intent.metric_name,
                "groupByColumn": intent.group_by_column or "",
                "tables": item.query_bundle.tables,
                "rowCount": item.query_bundle.effective_row_count(),
                "rows": item.query_bundle.rows[:10],
            }
        )
    return sections


def answer_result_role(intent: QuestionIntent | None) -> str:
    if not intent:
        return "result"
    resolution = intent.metric_resolution or {}
    if resolution.get("displayRole") == "trend_context":
        return "trend_context"
    time_column = declared_time_column_for_intent(intent)
    if time_column and intent.group_by_column == time_column and intent.answer_mode == AnswerMode.GROUP_AGG:
        return "trend_context"
    if intent.answer_mode == AnswerMode.METRIC:
        return "summary"
    if intent.answer_mode == AnswerMode.TOPN:
        return "ranking"
    if intent.answer_mode == AnswerMode.GROUP_AGG:
        return "group_summary"
    if intent.answer_mode == AnswerMode.DETAIL:
        return "detail"
    if intent.answer_mode == AnswerMode.DERIVED:
        return "derived"
    return "result"


def visible_successful_tasks(plan: QueryPlan, run_result: AgentRunResult) -> List[Any]:
    intent_map = intent_by_task_id(plan)
    succeeded = [item for item in run_result.task_results if not item.query_bundle.failed and item.query_bundle.rows]
    ordered = sorted(succeeded, key=lambda item: task_evidence_priority(intent_map.get(item.task_id), item, plan))
    support_ids = support_task_ids_for_answer(plan)
    visible = [item for item in ordered if answer_visible_task(intent_map.get(item.task_id), item, plan, support_ids)]
    return visible or ordered


def intent_by_task_id(plan: QueryPlan) -> Dict[str, QuestionIntent]:
    return {intent.plan_task_id: intent for intent in plan.intents if intent.plan_task_id}


def compact_rule_evidence(question: str, rule_context: str, max_items: int = 5) -> List[str]:
    context = (rule_context or "").strip()
    if not context:
        return []
    candidates: List[str] = []
    for raw_line in context.splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue
        if line.startswith(("{", "[", "}")):
            continue
        if '"topic"' in line or '"tableName"' in line or '"columnName"' in line:
            continue
        if line.startswith("#") or line.startswith("召回规则片段"):
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        if len(line) < 8 or "常见测试问法" in line:
            continue
        candidates.append(line)
    if not candidates:
        return [context[:240]]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (_rule_line_score(question, item[1]), -item[0]),
        reverse=True,
    )
    selected: List[str] = []
    for _, line in ranked:
        if line in selected:
            continue
        selected.append(line[:180])
        if len(selected) >= max_items:
            break
    return selected


def _rule_line_score(question: str, line: str) -> int:
    normalized = re.sub(r"\s+", "", question or "")
    grams: set[str] = set()
    for size in (2, 3, 4):
        grams.update(normalized[index : index + size] for index in range(max(0, len(normalized) - size + 1)))
    score = sum(size for gram in grams if gram and gram in line for size in [len(gram)])
    if any(char in line for char in normalized):
        score += 1
    return score


def metric_disclosures(plan: QueryPlan, verified: Any) -> List[Dict[str, Any]]:
    disclosures: List[Dict[str, Any]] = []
    for intent in plan.intents:
        if should_hide_alternate_metric(plan, intent):
            continue
        if intent.metric_specs:
            for spec in intent.metric_specs:
                if not isinstance(spec, dict):
                    continue
                disclosure = {
                    "metricKey": spec.get("metricName") or spec.get("metric_key") or spec.get("metricColumn"),
                    "ownerTable": intent.preferred_table,
                    "formula": spec.get("metricFormula") or spec.get("formula"),
                    "sourceColumns": spec.get("sourceColumns")
                    or ([spec.get("metricColumn")] if spec.get("metricColumn") else []),
                    "semanticRefId": (intent.metric_resolution or {}).get("semanticRefId"),
                    **metric_display_contract(intent, spec),
                }
                disclosures.append({key: value for key, value in disclosure.items() if value not in (None, "", [], {})})
            continue
        resolution = intent.metric_resolution or {}
        if resolution:
            disclosure = {
                    key: resolution.get(key)
                    for key in [
                        "requestedMetricRef",
                        "metricKey",
                        "ownerTable",
                        "sourceColumns",
                        "formula",
                        "displayName",
                        "description",
                        "unit",
                        "valueFormat",
                        "sourceColumnLabels",
                        "fieldWarning",
                        "semanticRefId",
                        "semanticContractHash",
                    ]
                    if resolution.get(key) not in (None, "", [])
                }
            if intent.preferred_table:
                disclosure.setdefault("ownerTable", intent.preferred_table)
            disclosures.append(disclosure)
    for item in getattr(verified, "derived_evidence", [])[:8]:
        if isinstance(item, dict):
            disclosures.append(
                {
                    key: item.get(key)
                    for key in [
                        "metric",
                        "formula",
                        "sourceColumns",
                        "sourceColumnLabels",
                        "displayName",
                        "description",
                        "unit",
                        "valueFormat",
                        "fieldWarning",
                    ]
                    if item.get(key) not in (None, "", [])
                }
            )
    return [item for item in dedupe_dicts(disclosures) if item]


def lightweight_metric_disclosures(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    items: List[Dict[str, Any]] = []
    for item in metric_disclosures(plan, run_result.verified_evidence):
        description = lightweight_metric_description(item, include_formula=question_asks_metric_disclosure(question))
        if not description:
            continue
        items.append(
            {
                "metricKey": item.get("metricKey") or item.get("metric"),
                "displayName": item.get("displayName") or "指标",
                "description": description,
                "unit": item.get("unit") or "",
                "valueFormat": item.get("valueFormat") or "",
                "sourceColumnLabels": item.get("sourceColumnLabels") or {},
            }
        )
    return [item for item in dedupe_dicts(items) if item][:4]


def lightweight_metric_disclosure_note(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    disclosures = lightweight_metric_disclosures(question, plan, run_result)
    if not disclosures:
        return ""
    descriptions = dedupe_strings([str(item.get("description") or "").strip() for item in disclosures])
    descriptions = [merchant_friendly_note_phrase(item) for item in descriptions if item]
    descriptions = [item for item in dedupe_strings(descriptions) if item]
    if not descriptions:
        return ""
    time_phrase = extract_question_time_phrase(question) or "本次查询时间范围"
    body = "；".join(description.rstrip("。；") for description in descriptions[:3] if description).strip("。；")
    if not body:
        return ""
    return "统计说明：%s。时间是%s，范围是当前店铺。" % (body, time_phrase)


def lightweight_metric_description(item: Dict[str, Any], include_formula: bool = False) -> str:
    display_name = str(item.get("displayName") or "指标")
    description = str(item.get("description") or item.get("fieldWarning") or "").strip()
    if description:
        friendly = merchant_friendly_metric_description(description)
        if friendly:
            return "%s：%s" % (display_name, friendly)
    formula = str(item.get("formula") or "").strip()
    if include_formula and formula:
        formula_phrase = merchant_friendly_formula_phrase(item)
        if formula_phrase:
            return "%s：%s" % (display_name, formula_phrase)
    return ""


def merchant_friendly_metric_description(description: str) -> str:
    text = str(description or "").strip()
    if not text:
        return ""
    return strip_internal_metric_description(text)


def merchant_friendly_note_phrase(description: str) -> str:
    text = str(description or "").strip(" 。；")
    if "：" in text:
        text = text.split("：", 1)[1].strip(" 。；")
    if ":" in text:
        text = text.split(":", 1)[1].strip(" 。；")
    if text.startswith("按") and text.endswith("的比例"):
        return "%s统计" % text
    if text and not text.startswith(("按", "以", "基于")) and re.search(r"(占|按).{0,20}(统计|比例|占比)", text):
        if text.startswith(("店铺", "跨天", "指定周期")) and "按" in text:
            text = text[text.index("按") :]
        elif text.endswith("的比例"):
            text = "按%s统计" % text
    return text


def merchant_friendly_formula_phrase(item: Dict[str, Any]) -> str:
    text = str(item.get("formula") or "")
    columns = [column.lower() for column in re.findall(r"`?([a-z][a-z0-9]+(?:_[a-z0-9]+)+)`?", text, flags=re.I)]
    source_labels = item.get("sourceColumnLabels") or {}
    column_labels = (
        {str(key).lower(): str(value) for key, value in source_labels.items() if str(key) and str(value)}
        if isinstance(source_labels, dict)
        else {}
    )
    labels = [column_labels[column] for column in columns if column in column_labels]
    if "/" in text and len(labels) >= 2:
        return "%s占%s的比例" % (labels[0], labels[1])
    if labels:
        return "按%s统计" % labels[0]
    return "按已配置语义公式统计" if text else ""


def strip_internal_metric_description(description: str) -> str:
    text = str(description or "").strip()
    if not text:
        return ""
    text = text.replace("`", "")
    text = re.sub(r"(?:^|[。；;，,]\s*)(?:公式|计算公式|语义公式)(?:为|是|[:：])\s*[^。；;]+[。；;]?", "。", text, flags=re.I)
    text = re.sub(r"\b(?:SUM|COUNT|AVG|MAX|MIN|NULLIF|CASE|WHEN|THEN|ELSE|END)\s*\([^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){2,}\b", "", text, flags=re.I)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,；;]\s*(?:和|与)?\s*(?:不同|相同|一致)?\s*$", "", text)
    text = re.sub(r"。{2,}", "。", text)
    parts = []
    for raw_part in re.split(r"[。；;]", text):
        part = raw_part.strip(" ：:，,。")
        if not part:
            continue
        if is_internal_answer_line(part):
            continue
        if re.search(r"使用\s*作为(?:分子|分母)", part):
            continue
        if re.search(r"\b[A-Za-z][A-Za-z0-9_]*\b", part) and "_" in part:
            continue
        parts.append(part)
    return "；".join(dedupe_strings(parts[:2])).strip("。；")


def metric_disclosure_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts: List[str] = []
    for key in ["displayName", "metricKey", "metric", "requestedMetricRef", "formula", "fieldWarning"]:
        value = item.get(key)
        if value not in (None, "", []):
            parts.append(str(value))
    source_columns = item.get("sourceColumns") or []
    if isinstance(source_columns, list):
        parts.extend(str(column) for column in source_columns if column)
    return " ".join(parts).lower()


def compact_evidence_gaps(gaps: List[Any]) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for gap in gaps[:8]:
        compacted.append(
            {
                "code": gap.code,
                "taskId": gap.task_id,
                "evidence": gap.evidence,
                "reason": gap.reason,
                "answerInstruction": gap.answer_instruction,
            }
        )
    return compacted


def dedupe_dicts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def sanitize_business_answer_text(answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None = None) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    labels = answer_column_labels(plan)
    for raw, label in sorted(labels.items(), key=lambda item: len(item[0]), reverse=True):
        if not raw or not label or raw == label:
            continue
        text = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(raw), label, text)
    text = text.replace("verified evidence", "当前数据").replace("Verified evidence", "当前数据")
    if not question_asks_metric_disclosure(question):
        text = remove_metric_disclosure_block(text)
    text = re.sub(r"(?m)^\s*分析结论[:：]\s*$\n?", "", text)
    text = re.sub(r"(?m)^\s*关键证据[:：]\s*$\n?", "", text)
    text = re.sub(r"(?m)^\s*(限制|证据缺口|证据门禁)[:：]", "说明：", text)
    text = re.sub(r"(?m)^\s*证据[:：]\s*$\n?", "", text)
    text = re.sub(r"当前证据显示存在可解释的波动点，不能简单判断为业务为 0 或无异常。", "这几天有波动，建议结合下方趋势和关联指标一起看。", text)
    text = re.sub(r"当前数据看存在可解释的波动点，不能简单判断为业务为 0 或无异常。", "这几天有波动，建议结合下方趋势和关联指标一起看。", text)
    text = re.sub(r"已看到的点位显示[:：]?", "", text)
    text = re.sub(r"当前证据显示", "当前数据看", text)
    text = re.sub(r"证据显示", "数据看", text)
    text = re.sub(r"已验证查询结果", "当前查询结果", text)
    text = re.sub(r"语义层指标口径", "当前指标口径", text)
    text = re.sub(r"业务为\s*0\s*或无异常", "没有异常", text)
    text = re.sub(r"(?m)^\s*-\s*(当前)?可用行数较少，异常判断可信度有限。", "- 可用数据点较少，异常判断可信度有限。", text)
    text = remove_hidden_alternate_metric_lines(text, plan)
    if not question_asks_metric_disclosure(question):
        text = remove_internal_answer_lines(text)
    text = normalize_answer_headings(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_answer_headings(text: str) -> str:
    lines: List[str] = []
    previous_blank = False
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        if re.match(r"^(分析结论|关键证据)[:：]?$", stripped):
            continue
        if re.match(r"^(限制|证据缺口|证据门禁)[:：]?$", stripped):
            lines.append("说明：")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def remove_internal_answer_lines(text: str) -> str:
    kept: List[str] = []
    skip_until_blank = False
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            skip_until_blank = False
            kept.append(raw_line)
            continue
        if re.match(r"^(口径|指标口径|字段口径|计算口径|使用表|来源表|SQL|字段名)[:：]", stripped, flags=re.I):
            skip_until_blank = True
            continue
        if skip_until_blank:
            if re.match(r"^(建议|说明)[:：]", stripped):
                skip_until_blank = False
                kept.append(raw_line)
            continue
        if is_internal_answer_line(stripped):
            continue
        kept.append(raw_line)
    return "\n".join(kept).strip()


def is_internal_answer_line(line: str) -> bool:
    if re.search(r"\b(SQL|Doris|QueryGraph|SELECT|FROM|WHERE|GROUP BY|HAVING|JOIN)\b", line, flags=re.I):
        return True
    if re.search(r"\b(?:ads|dwm|dwd|dim)_[a-z0-9_]+\b", line, flags=re.I):
        return True
    if re.search(r"\b(?:SUM|COUNT|AVG|MAX|MIN)\s*\(", line, flags=re.I):
        return True
    if re.search(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){2,}\b", line) and re.search(r"[:：=(),]", line):
        return True
    if re.search(r"(查到|查询到)\s*\d+\s*行|使用表|字段名|表名|执行失败|节点|EVIDENCE_GAP", line, flags=re.I):
        return True
    return False


def merchant_facing_gap_note(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[A-Z_]+[:：]\s*", "", text)
    if "SQL 成功但返回 0 行" in text or "执行成功但返回 0 行" in text:
        return "相关查询返回 0 行，不能直接解释为业务为 0。"
    if "派生指标缺少" in text or "分子/分母" in text:
        return "占比或派生指标缺少完整分子/分母，暂不能计算。"
    if "说明该证据缺口对回答范围的影响" in text or "说明该说明对回答范围的影响" in text:
        return "部分关联证据未完整覆盖，结论只能基于已返回数据。"
    replacements = {
        "证据门禁": "说明",
        "证据缺口": "说明",
        "证据": "数据",
        "Doris": "数据源",
        "SQL": "查询",
        "QueryGraph": "查询计划",
        "节点": "部分数据",
        "字段": "指标",
        "表关系": "数据关系",
        "语义层": "指标口径",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\b(?:ads|dwm|dwd|dim)_[a-z0-9_]+\b", "相关数据", text, flags=re.I)
    text = re.sub(r"相关数据\.[A-Za-z0-9_]+", "当前已确认口径", text)
    text = re.sub(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){2,}\b", "相关指标", text)
    text = re.sub(r"\s+", " ", text).strip(" ：:-")
    if not text:
        return ""
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return text


def hidden_alternate_metric_terms(plan: QueryPlan) -> List[str]:
    labels = answer_column_labels(plan)
    visible_business_terms: set[str] = set()
    for intent in plan.intents:
        if should_hide_alternate_metric(plan, intent):
            continue
        resolution = intent.metric_resolution or {}
        metric_key = intent_metric_key(intent)
        for item in [
            resolution.get("displayName"),
            resolution.get("sourcePhrase"),
            labels.get(metric_key),
            labels.get(str(intent.metric_name or "")),
        ]:
            term = str(item or "").strip()
            if len(term) >= 2:
                visible_business_terms.add(term)
    terms: List[str] = []
    for intent in plan.intents:
        if not should_hide_alternate_metric(plan, intent):
            continue
        resolution = intent.metric_resolution or {}
        metric_key = intent_metric_key(intent)
        candidates = [
            metric_key,
            intent.metric_name,
            intent.metric_column,
            labels.get(str(intent.metric_name or "")),
        ]
        for item in candidates:
            term = str(item or "").strip()
            if len(term) >= 2 and term not in visible_business_terms and term not in terms:
                terms.append(term)
    return terms


def remove_hidden_alternate_metric_lines(text: str, plan: QueryPlan) -> str:
    terms = hidden_alternate_metric_terms(plan)
    if not terms:
        return text
    kept: List[str] = []
    for line in str(text or "").splitlines():
        if line.strip().startswith("|"):
            kept.append(line)
            continue
        if any(term and term in line for term in terms):
            continue
        kept.append(line)
    return "\n".join(kept)


def question_asks_metric_disclosure(question: str) -> bool:
    return bool(re.search(r"(口径|怎么算|计算方式|字段|来源表|SQL|sql)", str(question or ""), flags=re.I))


def question_asks_metric_reconciliation(question: str) -> bool:
    return bool(
        re.search(
            r"(后台|看板|生意参谋|数据中心|报表).{0,12}(不一致|不一样|对不上|不对|差|少|多)"
            r"|(?:不一致|不一样|对不上|数不对|差很多|少了|多了).{0,12}(后台|看板|报表)"
            r"|(?:核对|对账).{0,12}(口径|数据|指标)",
            str(question or ""),
            flags=re.I,
        )
    )


def remove_metric_disclosure_block(text: str) -> str:
    lines = str(text or "").splitlines()
    cleaned: List[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(口径|指标口径|字段口径|计算口径)[:：]\s*$", stripped):
            skipping = True
            continue
        if skipping:
            if not stripped:
                skipping = False
                continue
            if re.match(r"^(建议|说明|限制|分析结论|关键证据)[:：]", stripped):
                skipping = False
                cleaned.append(line)
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def render_structured_skill_answer(renderer_name: str, payload: Dict[str, Any]) -> str:
    rows = payload.get("dataRows") or []
    disclosures = payload.get("metricDisclosures") or []
    gaps = payload.get("evidenceGaps") or []
    rule_evidence = payload.get("ruleEvidence") or []
    metadata = payload.get("skillMetadata") if isinstance(payload.get("skillMetadata"), dict) else {}
    configured_renderer = str(metadata.get("renderer") or "").strip()
    configured_name = str(metadata.get("name") or "").strip()
    if configured_renderer != "verified_evidence" or renderer_name not in {configured_renderer, configured_name}:
        return ""
    title = str(metadata.get("title") or metadata.get("description") or "").strip()
    if not title:
        return ""
    lines = ["%s：" % title]
    if disclosures:
        lines.append("- 已验证指标：%s" % "；".join(compact_disclosure(item) for item in disclosures[:6] if compact_disclosure(item)))
    if rows:
        lines.append("- 已验证结果：")
        for index, row in enumerate(rows[:6], 1):
            preview = compact_row_preview(row, disclosures)
            if preview:
                lines.append("  %d. %s" % (index, preview))
    else:
        lines.append("- 当前资源没有收到可展示的已验证结果。")
    if rule_evidence:
        lines.append("- 已验证规则依据：%s" % "；".join(str(item) for item in rule_evidence[:3] if str(item).strip()))
    if gaps:
        lines.append("")
        lines.append("说明：")
        for gap in gaps[:5]:
            if not isinstance(gap, dict):
                continue
            note = merchant_facing_gap_note(gap.get("reason") or gap.get("answerInstruction") or "当前信息还不完整")
            if note:
                lines.append("- %s" % note)
    lines.append("")
    lines.append("说明：以上判断基于本轮已查询到的数据。")
    return "\n".join(line for line in lines if line is not None).strip()


def has_business_advice_section(answer: str) -> bool:
    return bool(re.search(r"(^|\n)\s*建议[:：]", str(answer or "")))


def normalize_inline_business_advice(answer: str) -> str:
    body_lines: List[str] = []
    advice_items: List[str] = []
    in_advice = False
    for line in str(answer or "").splitlines():
        stripped = line.strip()
        if re.match(r"^建议[:：]\s*$", stripped):
            in_advice = True
            continue
        if re.match(r"^建议(?:[:：]|\S)", stripped):
            in_advice = True
            item = re.sub(r"^建议[:：]?\s*", "", stripped).strip()
            if item:
                advice_items.append(re.sub(r"^[-*\d.、\s]+", "", item).strip())
            continue
        if in_advice:
            if not stripped:
                continue
            if re.match(r"^(说明|参考|备注|数据|结论)[:：]\s*$", stripped):
                in_advice = False
                body_lines.append(line)
                continue
            item = re.sub(r"^[-*]?\s*\d+[.、]\s*", "", stripped).strip()
            item = re.sub(r"^[-*]\s*", "", item).strip()
            if item:
                advice_items.append(item)
            continue
        body_lines.append(line)
    if not advice_items:
        return answer
    cleaned_items = [re.sub(r"^[-*\d.、\s]+", "", item).strip() for item in advice_items if item.strip()]
    body = "\n".join(body_lines).rstrip()
    return body + "\n\n建议：\n" + "\n".join("- %s" % item for item in cleaned_items[:2])


def parse_llm_suggestions(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    payload: Any = None
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    suggestions: List[str] = []
    if isinstance(payload, dict):
        raw_items = payload.get("suggestions") or payload.get("建议") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []
    for item in raw_items:
        value = re.sub(r"^[-*\d.、\s]+", "", str(item or "").strip())
        if not value:
            continue
        if re.search(r"(继续追问|如需|如果需要|可以再问|我可以)", value):
            continue
        if re.search(r"(SQL|sql|字段|表名|QueryGraph|Doris)", value):
            continue
        if value not in suggestions:
            suggestions.append(value[:90])
        if len(suggestions) >= 2:
            break
    return suggestions


def contextual_business_suggestions(
    question: str,
    intents: List[QuestionIntent],
    run_result: AgentRunResult | None = None,
    merchant: MerchantInfo | None = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    del merchant, personalization_context
    if run_result is None or not answer_evidence_passed(run_result):
        return []
    plan = QueryPlan(intents=intents or [])
    labels = dedupe_strings(
        [
            str(signal.get("label") or "").strip()
            for signal in answer_current_data_signals(question, plan, run_result)
            if isinstance(signal, dict) and str(signal.get("label") or "").strip()
        ]
    )
    if not labels:
        return []
    question_norm = normalize_suggestion_text(question)
    suggestions: List[str] = []
    templates = ["查看%s按时间维度的变化", "按已验证维度拆解%s", "核对%s波动区间对应的明细"]
    for label in labels[:3]:
        for template in templates:
            value = template % label
            if normalize_suggestion_text(value) != question_norm and value not in suggestions:
                suggestions.append(value)
    return suggestions[:6]

def build_merchant_experience_package(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    merchant: MerchantInfo | None = None,
    sections: Optional[List[ChatDataSection]] = None,
    suggestions: Optional[List[str]] = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_passed = run_result is None or answer_evidence_passed(run_result)
    suggestion_items = (
        dedupe_strings([str(item) for item in (suggestions or []) if str(item).strip()])[:8]
        if evidence_passed
        else []
    )
    anomaly_alerts = merchant_anomaly_alerts(question, plan, run_result) if evidence_passed else []
    traceability = merchant_traceability(question, plan, run_result, merchant, sections or [])
    drill_actions = merchant_drill_down_actions(question, plan, run_result, suggestion_items)
    metric_notes = lightweight_metric_disclosures(question, plan, run_result)
    return {
        "version": "v1",
        "businessAdvice": (
            merchant_business_advice(question, plan, run_result, anomaly_alerts, personalization_context)[:2]
            if evidence_passed
            else []
        ),
        "suggestedQuestions": suggestion_items[:6],
        "anomalyAlerts": anomaly_alerts[:4],
        "metricDisclosures": metric_notes,
        "evidenceGaps": compact_evidence_gaps(getattr(run_result, "evidence_gaps", []) or []) if run_result else [],
        "traceability": traceability,
        "drillDownActions": drill_actions[:5] if evidence_passed else [],
        "reportSubscriptionHint": merchant_report_subscription_hint(plan, run_result) if evidence_passed else {},
        "clarificationHints": merchant_clarification_hints(plan),
    }


def answer_evidence_passed(run_result: AgentRunResult | None) -> bool:
    verified = getattr(run_result, "verified_evidence", None) if run_result is not None else None
    has_rows = bool(getattr(getattr(run_result, "merged_query_bundle", None), "rows", None)) or any(
        getattr(getattr(result, "query_bundle", None), "rows", None) for result in getattr(run_result, "task_results", []) or []
    )
    if verified is not None:
        if getattr(verified, "passed", False):
            return True
        if getattr(verified, "gaps", None) or getattr(verified, "blocking_gaps", None) or getattr(verified, "answer_guard_required", False):
            return False
        return has_rows
    if run_result is None or getattr(run_result, "evidence_gaps", None):
        return False
    return has_rows


def merchant_business_advice(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    anomaly_alerts: List[Dict[str, Any]],
    personalization_context: Optional[Dict[str, Any]],
) -> List[str]:
    del personalization_context
    if run_result is None or not answer_evidence_passed(run_result):
        return []
    labels = dedupe_strings(
        [str(item.get("metric") or "").strip() for item in anomaly_alerts if isinstance(item, dict)]
        + [
            str(item.get("label") or "").strip()
            for item in answer_current_data_signals(question, plan, run_result)
            if isinstance(item, dict)
        ]
    )
    advice: List[str] = []
    for label in labels[:2]:
        if not label:
            continue
        advice.extend(
            [
                "优先核对%s波动区间及关联明细。" % label,
                "按已验证维度拆解%s，优先检查变化最大的对象。" % label,
            ]
        )
    return dedupe_strings(advice)[:2]


def merchant_anomaly_alerts(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    alerts: List[Dict[str, Any]] = []
    intent_map = intent_by_task_id(plan)
    for task in visible_successful_tasks(plan, run_result):
        intent = intent_map.get(task.task_id)
        if not intent or not task.query_bundle.rows:
            continue
        trend_rows, complete = query_bundle_rows_for_trend(task.query_bundle)
        points = metric_series_rows_for_intent(plan, intent, trend_rows) if complete else []
        if len(points) < 2:
            continue
        first = answer_numeric_value(points[0].get("value"))
        last = answer_numeric_value(points[-1].get("value"))
        if first is None or last is None:
            continue
        base = abs(first) if abs(first) > 0.000001 else 1.0
        change_rate = (last - first) / base
        if abs(change_rate) < 0.3:
            continue
        metric_key = str((intent.metric_resolution or {}).get("metricKey") or intent.metric_name or "")
        metric_label = str((intent.metric_resolution or {}).get("displayName") or friendly_column_label(plan, metric_key) or intent.metric_name or "指标")
        direction = "上升" if change_rate > 0 else "下降"
        alerts.append(
            {
                "type": "trend_change",
                "severity": "warning" if abs(change_rate) < 0.8 else "high",
                "metric": metric_label,
                "direction": direction,
                "changeRate": round(change_rate, 4),
                "message": "%s从 %s 到 %s，%s约 %.1f%%。" % (
                    metric_label,
                    format_cell(points[0].get("value")),
                    format_cell(points[-1].get("value")),
                    direction,
                    abs(change_rate) * 100,
                ),
                "drillDownQuestion": "%s波动最大的时间点对应哪些已验证明细？" % metric_label,
            }
        )
    return alerts


def merchant_traceability(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    merchant: MerchantInfo | None,
    sections: List[ChatDataSection],
) -> Dict[str, Any]:
    tables = dedupe_strings(
        [table for section in sections for table in section.doris_tables]
        + [table for table in getattr(getattr(run_result, "merged_query_bundle", None), "tables", []) or []]
    )
    rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
    dates = declared_time_values(plan, run_result)
    return {
        "sourceSummary": "基于语义层口径、Doris 查询结果和证据校验生成",
        "merchantId": getattr(merchant, "merchant_id", "") if merchant else "",
        "merchantName": getattr(merchant, "merchant_name", "") if merchant else "",
        "timeRange": extract_question_time_phrase(question) or "按本次问题解析的时间范围",
        "dataUpdatedAt": max(dates) if dates else "",
        "rowCount": len(rows),
        "sectionCount": len(sections),
        "sourceTables": tables[:8],
        "evidenceStatus": "verified" if run_result and getattr(run_result.verified_evidence, "passed", False) else "partial",
    }


def declared_time_values(plan: QueryPlan, run_result: AgentRunResult | None) -> List[str]:
    if not run_result:
        return []
    values: List[str] = []
    intent_map = intent_by_task_id(plan)
    for task in run_result.task_results:
        intent = intent_map.get(task.task_id)
        contract = getattr(task, "node_plan_contract", None)
        time_contract = getattr(contract, "time_window_contract", None) or {}
        column = declared_time_column_from_contract(time_contract) or (
            plan_time_column_for_intent(plan, intent) if intent else ""
        )
        if not column:
            continue
        values.extend(
            str(row.get(column))
            for row in task.query_bundle.rows or []
            if isinstance(row, dict) and row.get(column) not in (None, "")
        )
    if values:
        return values
    columns = {
        plan_time_column_for_intent(plan, intent)
        for intent in plan.intents
        if plan_time_column_for_intent(plan, intent)
    }
    for row in run_result.merged_query_bundle.rows or []:
        if not isinstance(row, dict):
            continue
        values.extend(str(row.get(column)) for column in columns if row.get(column) not in (None, ""))
    return values


def declared_time_column_from_contract(contract: Any) -> str:
    if not isinstance(contract, dict):
        return ""
    for key in ["partitionColumn", "partition_column", "timeColumn", "time_column"]:
        value = str(contract.get(key) or "").strip()
        if value:
            return value
    return ""


def merchant_drill_down_actions(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    suggestions: List[str],
) -> List[Dict[str, Any]]:
    if run_result is None or not answer_evidence_passed(run_result):
        return []
    actions: List[Dict[str, Any]] = []

    def add(label: str, follow_up: str, action_type: str = "follow_up_question") -> None:
        if any(item.get("label") == label for item in actions):
            return
        actions.append({"label": label, "question": follow_up, "actionType": action_type})

    labels = dedupe_strings(
        [
            str(item.get("label") or "").strip()
            for item in answer_current_data_signals(question, plan, run_result)
            if isinstance(item, dict)
        ]
    )
    for label in labels[:3]:
        add("查看%s明细" % label, "按已验证维度拆解%s" % label)
    for item in suggestions[:2]:
        add("继续分析", item)
    return actions


def merchant_report_subscription_hint(plan: QueryPlan, run_result: AgentRunResult | None) -> Dict[str, Any]:
    categories = {normalize_question_category(intent.category) for intent in plan.intents or []}
    metrics = dedupe_strings(
        [
            str((intent.metric_resolution or {}).get("displayName") or "")
            for intent in plan.intents
            if str((intent.metric_resolution or {}).get("displayName") or "").strip()
        ]
    )
    if not categories and not metrics:
        return {}
    return {
        "enabled": True,
        "title": "加入经营日报关注",
        "description": "可把本次关注的业务域和指标加入日报，后续自动推送异常和重点变化。",
        "topics": [category_display(category) for category in categories if category != QuestionCategory.UNKNOWN][:5],
        "metrics": metrics[:6],
    }


def merchant_clarification_hints(plan: QueryPlan) -> List[str]:
    hints: List[str] = []
    if not plan.intents:
        return ["请补充查询时间范围。", "也可以补充要看的业务对象或指标。"]
    primary = plan.intents[0]
    if not primary.days:
        hints.append("如果没有指定时间，系统会优先按近期常用时间窗或最近7天理解。")
    if primary.category == QuestionCategory.UNKNOWN:
        hints.append("业务范围不明确时，需要先确认查询对象。")
    if not primary.metric_name and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT}:
        hints.append("指标不明确时，需要先确认指标定义或明细范围。")
    return hints[:3]


def normalize_question_category(value: Any) -> QuestionCategory:
    if isinstance(value, QuestionCategory):
        return value
    try:
        return QuestionCategory(str(value))
    except Exception:
        return QuestionCategory.UNKNOWN


def normalize_suggestion_text(value: str) -> str:
    return re.sub(r"[\s？?。！!，,、]+", "", str(value or "").strip().lower())


def answer_business_context(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    merchant: MerchantInfo | None = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = personalization_context or {}
    memory_injection = context.get("memoryInjection") or context.get("memory_injection") or {}
    recent_focus = context.get("recentFocus") or memory_injection.get("recentFocus") or {}
    merchant_profile = str(context.get("merchantProfileContext") or "")
    if not merchant_profile and merchant is not None:
        merchant_profile = merchant.profile_markdown()
    return {
        "merchantProfile": compact_answer_context_text(merchant_profile, 1200),
        "sessionSummary": compact_answer_context_text(str(context.get("sessionContext") or ""), 1000),
        "memorySummary": compact_answer_context_text(str(context.get("memoryContext") or ""), 1000),
        "recentFocus": compact_recent_focus(recent_focus),
        "relevantPreferences": compact_memory_payloads(memory_injection.get("relevantPreferences") or memory_injection.get("preferences") or [], 4),
        "relevantCorrections": compact_memory_payloads(memory_injection.get("relevantCorrections") or [], 3),
        "relevantFacts": compact_memory_payloads(memory_injection.get("relevantFacts") or memory_injection.get("facts") or [], 3),
        "questionCategories": [category_display(intent.category) for intent in plan.intents[:6]],
        "currentDataSignals": answer_current_data_signals(question, plan, run_result),
    }


def compact_answer_context_text(value: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(value or "").strip())
    return text[:max_chars]


def compact_recent_focus(focus: Any) -> Dict[str, Any]:
    if not isinstance(focus, dict):
        return {}
    return {
        "summary": focus.get("summary") or focus.get("focusPattern") or "",
        "topTopics": focus.get("topTopics") or focus.get("top_categories") or [],
        "topMetrics": focus.get("topMetrics") or focus.get("top_metrics") or [],
        "commonTimeWindows": focus.get("commonTimeWindows") or focus.get("common_time_ranges") or [],
    }


def compact_memory_payloads(items: Any, max_items: int) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for item in list(items or [])[:max_items]:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                key: item.get(key)
                for key in ["memoryType", "summary", "value", "topics", "metrics", "timeWindows", "confidence"]
                if item.get(key) not in (None, "", [])
            }
        )
    return compacted


def answer_current_data_signals(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    signals: List[Dict[str, Any]] = []
    for item in summary_metric_values(plan, run_result)[:6]:
        label = str(item.get("label") or item.get("metricKey") or "指标")
        signals.append(
            {
                "type": "summary",
                "label": label,
                "value": format_metric_value_for_answer(
                    item.get("value"),
                    str(item.get("metricKey") or ""),
                    label,
                    item.get("displayMetadata"),
                ),
                "taskId": item.get("taskId"),
            }
        )
    for task in visible_successful_tasks(plan, run_result)[:8]:
        intent = intent_by_task_id(plan).get(task.task_id)
        if answer_result_role(intent) != "trend_context" or not task.query_bundle.rows:
            continue
        resolution = (intent.metric_resolution if intent else {}) or {}
        metric_key = str(resolution.get("metricKey") or getattr(intent, "metric_name", "") or "")
        label = str(resolution.get("displayName") or friendly_column_label(plan, metric_key) or "指标")
        trend_rows, complete = query_bundle_rows_for_trend(task.query_bundle)
        points = metric_series_rows_for_intent(plan, intent, trend_rows) if intent and complete else []
        if not points:
            continue
        values = [answer_numeric_value(point.get("value")) for point in points]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        peak = max(points, key=lambda point: answer_numeric_value(point.get("value")) or float("-inf"))
        signals.append(
            {
                "type": "trend",
                "label": label,
                "dateRange": extract_question_time_phrase(question),
                "pointCount": len(points),
                "peakDate": peak.get(TIME_DIMENSION_KEY),
                "peakValue": format_metric_value_for_answer(peak.get("value"), metric_key, label, resolution),
            }
        )
    return signals[:10]


def compact_disclosure(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    name = str(item.get("displayName") or "").strip()
    if not name:
        return ""
    description = str(item.get("description") or "").strip()
    unit = str(item.get("unit") or "").strip()
    details = "；".join(value for value in [description, "单位：%s" % unit if unit else ""] if value)
    return ("%s（%s）" % (name, details))[:160] if details else name[:100]


def compact_row_preview(row: Dict[str, Any], disclosures: List[Dict[str, Any]]) -> str:
    if not isinstance(row, dict):
        return ""
    labels: Dict[str, str] = {}
    contracts: Dict[str, Dict[str, Any]] = {}
    for item in disclosures:
        if not isinstance(item, dict):
            continue
        display_name = str(item.get("displayName") or "").strip()
        if not display_name:
            continue
        for key in [item.get("metricKey"), item.get("sourceColumn")]:
            if str(key or "").strip():
                labels[str(key)] = display_name
                contracts[str(key)] = item
        source_labels = item.get("sourceColumnLabels")
        if isinstance(source_labels, dict):
            labels.update({str(key): str(value) for key, value in source_labels.items() if str(key) and str(value)})
            contracts.update({str(key): item for key in source_labels if str(key)})
        elif isinstance(source_labels, list):
            for entry in source_labels:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("column") or entry.get("sourceColumn") or "").strip()
                label = str(entry.get("label") or entry.get("displayName") or "").strip()
                if key and label:
                    labels[key] = label
                    contracts[key] = item
    parts: List[str] = []
    for key, value in row.items():
        label = labels.get(str(key))
        if label:
            parts.append(
                "%s=%s"
                % (label, format_metric_value_for_answer(value, str(key), label, contracts.get(str(key))))
            )
        if len(parts) >= 8:
            break
    return "，".join(parts)[:220]


def answer_skill_headers(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    headers: List[Dict[str, Any]] = []
    for skill_file in sorted(root.glob("*/SKILL.md")):
        meta = load_skill_frontmatter(skill_file)
        body = skill_file.read_text(encoding="utf-8")
        name = str(meta.get("name") or skill_file.parent.name)
        if not name:
            continue
        when_to_use = _skill_header_field(meta, body, "whenToUse", "Activation Contract", 700)
        required_inputs = _skill_section_lines(meta.get("requiredInputs"), body, "Required Inputs", 6)
        constraints = _skill_section_lines(meta.get("constraints"), body, "Evidence Rules", 8)
        if not constraints:
            constraints = _skill_section_lines("", body, "Constraints", 8)
        headers.append(
            {
                "name": name,
                "description": str(meta.get("description") or "")[:500],
                "title": str(meta.get("title") or "")[:160],
                "executionMode": str(meta.get("executionMode") or meta.get("execution_mode") or "").strip(),
                "renderer": str(meta.get("renderer") or "").strip(),
                "script": str(meta.get("script") or meta.get("scriptPath") or meta.get("script_path") or "").strip(),
                "whenToUse": when_to_use,
                "when_to_use": when_to_use,
                "constraints": constraints,
                "requiredInputs": required_inputs,
                "required_inputs": required_inputs,
                "path": str(skill_file.relative_to(root.parent.parent) if root.parent.parent in skill_file.parents else skill_file),
            }
        )
    return headers


def configured_answer_skill_headers() -> List[Dict[str, Any]]:
    """Scan the active runtime resource directory without embedding skill identities."""
    try:
        from merchant_ai.config import get_settings

        root = get_settings().resources_root / "runtime" / "agent_skills"
    except Exception:
        root = Path(__file__).resolve().parents[2] / "resources" / "runtime" / "agent_skills"
    return answer_skill_headers(root)


def _skill_header_field(meta: Dict[str, Any], body: str, key: str, section: str, limit: int) -> str:
    value = str(meta.get(key) or meta.get(key[:1].lower() + key[1:]) or "")
    if value:
        return value[:limit]
    lines = _markdown_section_lines(body, section, limit_lines=6)
    return " ".join(lines)[:limit]


def _skill_section_lines(meta_value: Any, body: str, section: str, limit_lines: int) -> List[str]:
    if isinstance(meta_value, list):
        return [str(item).strip() for item in meta_value if str(item).strip()][:limit_lines]
    if str(meta_value or "").strip():
        return [item.strip() for item in str(meta_value).split(";") if item.strip()][:limit_lines]
    return _markdown_section_lines(body, section, limit_lines=limit_lines)


def _markdown_section_lines(body: str, section: str, limit_lines: int) -> List[str]:
    lines = body.splitlines()
    capture = False
    collected: List[str] = []
    target = section.strip().lower()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip().lower()
            if capture and heading != target:
                break
            capture = heading == target
            continue
        if not capture:
            continue
        if not stripped or stripped.startswith("```"):
            continue
        cleaned = stripped.lstrip("-").strip()
        if cleaned:
            collected.append(cleaned)
        if len(collected) >= limit_lines:
            break
    return collected


def parse_skill_match_payload(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


SKILL_NO_MATCH_STATUSES = {
    "no_match",
    "no-match",
    "not_applicable",
    "not-applicable",
    "inapplicable",
    "none",
    "no_skill",
    "no-skill",
    "rejected",
    "skip",
    "skipped",
}


def normalize_skill_match_status(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def skill_match_payload_name(payload: Dict[str, Any]) -> str:
    return str(payload.get("skillName") or payload.get("skill_name") or "").strip()


def skill_match_payload_explicit_no_match(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    return skill_route_payload_explicit_no_match(payload)


def skill_match_response_status(raw: str, payload: Dict[str, Any], selected: str) -> str:
    if not str(raw or "").strip():
        return "empty_response"
    if not payload:
        return "invalid_payload"
    if skill_match_payload_explicit_no_match(payload):
        return "explicit_no_match"
    return "matched" if selected else "invalid_payload"


def analysis_skill_match_is_explicit_no_match(trace: Dict[str, Any]) -> bool:
    return bool(
        isinstance(trace, dict)
        and (
            trace.get("matchStatus") == "explicit_no_match"
            or trace.get("matchedBy") in {"llm_explicit_no_match", "semantic_explicit_no_match"}
        )
    )


def answer_skill_reuse_candidate(skill_name: str, result: Dict[str, Any]) -> bool:
    if not skill_name or not isinstance(result, dict):
        return False
    row_count = int(result.get("rowCount") or 0)
    findings = result.get("findings") or []
    answer = str(result.get("answerMarkdown") or "")
    return row_count > 0 and (bool(findings) or len(answer) >= 20)


def load_skill_frontmatter(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta: Dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta


class DailyReportService:
    def __init__(self, doris_repository: DorisRepository, default_merchant_id: str = "", topic_assets: Any = None):
        self.doris_repository = doris_repository
        self.default_merchant_id = str(default_merchant_id or "").strip()
        self.topic_assets = topic_assets

    def report(self, merchant_id: str) -> DailyReportResponse:
        target = str(merchant_id or self.default_merchant_id).strip()
        metrics: Dict[str, Any] = {}
        role_values: Dict[str, Any] = {}
        merchant_name = "yshopping商家%s" % target
        profile, topic, table, asset = self._semantic_profile()
        metric_definitions = {
            str(item.get("metricKey") or ""): item
            for item in (asset.get("metrics") or [])
            if isinstance(item, dict) and item.get("metricKey")
        }
        schema_columns = {
            str(item.get("columnName") or item.get("Field") or "")
            for item in (asset.get("schemaColumns") or [])
            if item.get("columnName") or item.get("Field")
        }
        configured_metrics: List[Dict[str, Any]] = []
        select_parts: List[str] = []
        for item in profile.get("metrics") or []:
            if not isinstance(item, dict):
                continue
            metric_ref = str(item.get("metricRef") or "")
            definition = metric_definitions.get(metric_ref) or {}
            formula = compile_metric_formula(str(definition.get("formula") or ""), schema_columns)
            source_columns = [str(value) for value in definition.get("sourceColumns") or [] if str(value)]
            if not formula or not source_columns or any(column not in schema_columns for column in source_columns):
                continue
            alias = "metric_%d" % len(configured_metrics)
            configured_metrics.append({**item, "definition": definition, "alias": alias})
            select_parts.append("%s AS `%s`" % (formula, alias))
        tenant_column = str(asset.get("merchantFilterColumn") or "")
        time_column = str(asset.get("timeColumn") or "")
        try:
            row = None
            if select_parts and safe_report_identifier(table) and safe_report_identifier(tenant_column) and safe_report_identifier(time_column):
                sql = (
                    "SELECT %s FROM `%s` WHERE `%s`=%%s AND `%s`=(SELECT MAX(`%s`) FROM `%s` WHERE `%s`=%%s)"
                    % (", ".join(select_parts), table, tenant_column, time_column, time_column, table, tenant_column)
                )
                row = self.doris_repository.query_one(sql, [target, target])
            if row:
                for item in configured_metrics:
                    label = str(item.get("displayName") or (item.get("definition") or {}).get("businessName") or item.get("metricRef") or "")
                    role = str(item.get("role") or item.get("metricRef") or "")
                    value = row.get(item["alias"], 0)
                    if label:
                        metrics[label] = value
                    if role:
                        role_values[role] = value
        except Exception:
            pass
        alerts = daily_report_alerts(profile, configured_metrics, role_values)
        alert_suggestions = [
            str(item.get("drillDownQuestion") or "")
            for item in alerts
            if str(item.get("drillDownQuestion") or "").strip()
        ]
        configured_suggestions = [str(item) for item in profile.get("suggestions") or [] if str(item)]
        return DailyReportResponse(
            merchant_id=target,
            merchant_name=merchant_name,
            date=date.today().isoformat(),
            metrics=metrics,
            anomaly_alerts=alerts,
            drill_down_actions=[dict(item) for item in profile.get("drillDownActions") or [] if isinstance(item, dict)],
            traceability={
                "sourceSummary": "基于已发布语义日报配置生成",
                "merchantId": target,
                "merchantName": merchant_name,
                "timeRange": "昨日",
                "sourceTables": [table] if table else [],
                "semanticTopic": topic,
                "semanticProfile": str(profile.get("profileKey") or ""),
                "semanticMetricRefs": [str(item.get("metricRef") or "") for item in configured_metrics],
            },
            suggestions=dedupe_strings(alert_suggestions + configured_suggestions)[:3],
        )

    def _semantic_profile(self) -> tuple[Dict[str, Any], str, str, Dict[str, Any]]:
        if self.topic_assets is None:
            return {}, "", "", {}
        for topic in self.topic_assets.all_topic_names():
            for manifest in self.topic_assets.load_manifest(topic):
                table = str(manifest.get("tableName") or "")
                if not table:
                    continue
                asset = self.topic_assets.load_table_asset(topic, table)
                profile = asset.get("dailyReportProfile") or {}
                if isinstance(profile, dict) and profile.get("metrics"):
                    return profile, topic, table, asset
        return {}, "", "", {}


def safe_report_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value or "")))


def daily_report_alerts(profile: Dict[str, Any], metrics: List[Dict[str, Any]], role_values: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    by_role = {str(item.get("role") or ""): item for item in metrics if item.get("role")}
    for rule in profile.get("alerts") or []:
        if not isinstance(rule, dict):
            continue
        conditions = [item for item in rule.get("conditions") or [] if isinstance(item, dict)]
        if not conditions or not all(daily_report_condition_matches(item, role_values) for item in conditions):
            continue
        role = str(rule.get("metricRole") or "")
        metric = by_role.get(role) or {}
        display_name = str(metric.get("displayName") or metric.get("metricRef") or role)
        value = role_values.get(role)
        template = str(rule.get("messageTemplate") or "{displayName}: {formattedValue}")
        alerts.append(
            {
                "type": str(rule.get("type") or "semantic_alert"),
                "severity": str(rule.get("severity") or "warning"),
                "metric": display_name,
                "message": template.format(displayName=display_name, formattedValue=format_cell(value), value=value),
                "drillDownQuestion": str(rule.get("drillDownQuestion") or ""),
            }
        )
    return alerts[:3]


def daily_report_condition_matches(condition: Dict[str, Any], role_values: Dict[str, Any]) -> bool:
    actual = answer_numeric_value(role_values.get(str(condition.get("role") or "")))
    expected = answer_numeric_value(condition.get("value"))
    if actual is None or expected is None:
        return False
    operator = str(condition.get("operator") or "==")
    return {
        "==": actual == expected,
        "!=": actual != expected,
        ">": actual > expected,
        ">=": actual >= expected,
        "<": actual < expected,
        "<=": actual <= expected,
    }.get(operator, False)


class FeedbackService:
    def __init__(
        self,
        answer_repository: AnswerRepository,
        pending_store: PendingAnswerStore,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.answer_repository = answer_repository
        self.pending_store = pending_store
        self.memory_store = memory_store

    def apply_feedback(
        self,
        answer_id: str,
        adopted: Any,
        liked: Any,
        disliked: Any,
        identity: Any = None,
    ) -> bool:
        pending = self.pending_store.get(answer_id)
        persisted = False
        if pending:
            persisted = self.answer_repository.insert_answer(
                pending,
                adopted=bool(adopted) if adopted is not None else False,
                liked=bool(liked) if liked is not None else False,
                disliked=bool(disliked) if disliked is not None else False,
            )
        self.answer_repository.update_feedback(answer_id, adopted, liked, disliked)
        if self.memory_store is not None and pending and self._memory_feedback_authorized(pending, identity):
            self.memory_store.update_from_feedback(pending, adopted=adopted, liked=liked, disliked=disliked)
        return persisted

    def _memory_feedback_authorized(self, pending: Any, identity: Any) -> bool:
        if identity is None:
            return False
        payload = identity.model_dump(by_alias=True) if hasattr(identity, "model_dump") else identity
        if not isinstance(payload, dict):
            return False
        merchant_id = str(payload.get("merchantId") or payload.get("merchant_id") or "").strip()
        if not merchant_id or merchant_id != str(getattr(pending, "merchant_id", "") or "").strip():
            return False
        expected_hash = str(getattr(pending, "identity_scope_hash", "") or "")
        if not expected_hash or identity_scope_hash(identity, merchant_id) != expected_hash:
            return False
        pending_user = str(getattr(pending, "user_id", "") or "")
        current_user = str(payload.get("userId") or payload.get("user_id") or "")
        return bool(pending_user and current_user and current_user == pending_user)


def joined_categories(plan: QueryPlan) -> str:
    categories = plan.categories()
    if not categories:
        return "未知"
    return "、".join(category_display(category) for category in categories)


def build_response_context(
    question: str,
    plan: QueryPlan,
    merchant: MerchantInfo,
    sections: List[ChatDataSection],
    pending_stage: str = "",
    pending_type: str = "",
    pending_options: List[str] = None,
    pending_question: str = "",
) -> ChatContext:
    primary = plan.intents[0] if plan.intents else QuestionIntent()
    topics = plan.categories()
    metric_keys = dedupe_strings(
        [
            str(intent.metric_resolution.get("metricKey") or intent.metric_name or intent.metric_column or "")
            for intent in plan.intents
        ]
    )
    dimension_keys = dedupe_strings(
        [str(intent.group_by_column or "") for intent in plan.intents]
    )
    return ChatContext(
        question=question,
        days=primary.days,
        category=category_display(primary.category),
        answer_mode=primary.answer_mode.value if hasattr(primary.answer_mode, "value") else str(primary.answer_mode),
        topic=joined_categories(plan),
        topics=topics,
        metric_keys=metric_keys,
        dimension_keys=dimension_keys,
        data_catalog=",".join(table for section in sections for table in section.doris_tables),
        merchant_profile=merchant.profile_markdown(),
        context_summary="QueryGraph nodes=%s, sections=%s" % (len(plan.intents), len(sections)),
        pending_clarification_stage=pending_stage,
        pending_clarification_type=pending_type,
        pending_question=(pending_question or question) if pending_stage else "",
        pending_clarification_options=pending_options or [],
    )
