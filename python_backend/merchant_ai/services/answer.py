from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from merchant_ai.models import (
    AgentRunResult,
    AnswerMode,
    ChatContext,
    ChatDataSection,
    DailyReportResponse,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    category_display,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.memory import StructuredMemoryStore
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, PendingAnswerStore


class AnswerComposeService:
    def __init__(self, llm: LlmClient):
        self.llm = llm
        self.prompt_assembler = PromptAssembler()
        self.last_prompt_chars = 0
        self.last_analysis_skill_trace: Dict[str, Any] = {}
        self.last_compose_llm_attempted = False
        self.last_compose_used_llm = False

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
    ) -> str:
        self.last_compose_llm_attempted = False
        self.last_compose_used_llm = False
        self.last_prompt_chars = 0
        if not plan.intents:
            return self._no_execution_answer(plan)
        primary = plan.intents[0] if plan.intents else QuestionIntent()
        if primary.answer_mode == AnswerMode.CHAT:
            return "您好，我是 yshopping 商家 AI 助手，可以帮您查询经营、订单、退款、客服、赔付、优惠券、商品和商家资料。"
        if primary.answer_mode == AnswerMode.INVALID:
            return "这个问题还缺少业务对象或查询范围，请补充要看的指标、时间范围或业务域。"
        if primary.answer_mode == AnswerMode.RULE:
            return self._compose_rule_answer(question, knowledge_context)
        effective_rule_context = rule_context if plan_requires_rule_evidence(plan) else ""
        if analysis_summary:
            return self._apply_answer_guard(self._append_rule_evidence(analysis_summary, question, effective_rule_context), run_result)
        bundle = run_result.merged_query_bundle if run_result else QueryBundle()
        if primary.intent_type == "VALID" and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT} and (not run_result or not run_result.task_results):
            return self._no_execution_answer(plan)
        if run_result and run_result.task_results and all(result.query_bundle.failed for result in run_result.task_results):
            return self._apply_answer_guard(
                self._append_rule_evidence(self.append_business_advice(self._execution_failure_answer(run_result), plan.intents, bundle), question, effective_rule_context),
                run_result,
            )
        if allow_llm and self.llm.configured and (bundle.rows or run_result.evidence_gaps):
            self.last_compose_llm_attempted = True
            prompt = json.dumps(answer_data_package(question, plan, run_result, rule_context), ensure_ascii=False, default=str)
            self.last_prompt_chars = len(prompt)
            answer_prompt = self.prompt_assembler.render(
                "answer.bi",
                sections={
                    "answer_context_policy": (
                        "AnswerAgent 只读取 question、tables、rowCount、dataRows、dataSections、metricDisclosures、evidenceGaps；不要读取或推断 QueryGraph。"
                        "用商家能理解的自然语言先给结论；不要说“查到几行”“使用表”“SQL”“字段名”。"
                        "不要输出 markdown 表格，表格由前端结构化区域渲染。只有用户追问口径时才轻量解释指标口径。"
                        "dataRows 或 dataSections 中 resultRole=summary 的行是已验证汇总结果，优先用于回答总量；"
                        "resultRole=trend_context 的行只是趋势辅助，不能因为趋势只有部分有数日期，就否定 summary 汇总。"
                        "如果 summary 和 trend_context 是同一个指标，不要把 summary 行说成“未带日期的记录”。"
                        "趋势只用于说明有波动的日期，不要说“其余日期没有看到明细”这类会让商家误解为总量不可信的话。"
                        "最后最多给 2 条和当前问题强相关的经营建议，不要泛泛说继续追问。"
                    ),
                },
            )
            self.last_prompt_chars += len(answer_prompt.system_prompt)
            answer = self.llm.chat(
                answer_prompt.system_prompt,
                prompt,
                "",
                timeout_seconds=self.llm.settings.llm_answer_timeout_seconds,
            )
            if answer:
                self.last_compose_used_llm = True
                answer = self._correct_metric_total_misread(answer, question, plan, run_result)
                answer = self._clean_summary_trend_misphrasing(answer, plan, run_result)
                return self._apply_answer_guard(
                    self._append_rule_evidence(self.append_business_advice(answer, plan.intents, bundle), question, effective_rule_context),
                    run_result,
                )
        return self._apply_answer_guard(
            self._append_rule_evidence(
                self.append_business_advice(self._fallback_data_answer(question, plan, bundle, run_result), plan.intents, bundle),
                question,
                effective_rule_context,
            ),
            run_result,
        )

    def summarize_analysis(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
    ) -> str:
        self.last_analysis_skill_trace = {}
        if not run_result or not run_result.merged_query_bundle.rows:
            return ""
        skill_name = self.propose_answer_skill(question, plan, run_result, bool(rule_context))
        if not skill_name and not analysis_summary_required(plan):
            return ""
        skill_answer = self.run_analysis_skill(question, plan, run_result, outputs_path, rule_context, skill_name=skill_name)
        if skill_answer:
            return skill_answer
        if not self.llm.configured:
            return ""
        analysis_prompt = self.prompt_assembler.render(
            "answer.analysis",
            sections={"analysis_policy": "只能基于 compact evidence 判断趋势、异常和原因假设；不能把缺失证据当事实。"},
        )
        prompt = json.dumps(
            answer_data_package(question, plan, run_result, rule_context),
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

    def propose_answer_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        has_rule_context: bool = False,
    ) -> str:
        candidates = answer_skill_headers(self.llm.settings.resources_root / "runtime" / "agent_skills")
        fallback = select_answer_skill(plan, run_result, has_rule_context)
        trace: Dict[str, Any] = {
            "lifecycle": ["match", "confirm", "isolated_execute", "progress", "output"],
            "matchMode": self.llm.settings.answer_skill_match_mode,
            "candidateSkills": [item.get("name") for item in candidates],
            "fallbackSkill": fallback,
        }
        self.last_analysis_skill_trace = trace
        if not candidates or self.llm.settings.answer_skill_match_mode == "off":
            trace["matchedBy"] = "deterministic_fallback"
            trace["skillName"] = fallback
            return fallback
        if not self.llm.configured and self.llm.settings.answer_skill_match_mode != "always":
            trace["matchedBy"] = "deterministic_fallback_no_llm"
            trace["skillName"] = fallback
            return fallback
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
            "Return empty skillName when no skill should run."
        )
        raw = self.llm.chat(system, json.dumps(prompt_payload, ensure_ascii=False, default=str), "", timeout_seconds=8)
        trace["matchedBy"] = "llm_skill_header_match"
        trace["llmRaw"] = raw[:800] if raw else ""
        payload = parse_skill_match_payload(raw)
        allowed = {str(item.get("name") or "") for item in candidates}
        selected = str(payload.get("skillName") or "")
        if selected and selected not in allowed:
            selected = ""
            trace["matchWarning"] = "LLM_SELECTED_UNKNOWN_SKILL"
        if not selected:
            selected = fallback
            trace["matchedBy"] = "deterministic_fallback_after_llm"
        trace["skillName"] = selected
        trace["confidence"] = payload.get("confidence")
        trace["reason"] = payload.get("reason")
        return selected

    def run_analysis_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        skill_name: str = "",
    ) -> str:
        selected_skill = skill_name or select_answer_skill(plan, run_result, bool(rule_context)) or "bi_trend_attribution"
        skill_dir = self.llm.settings.resources_root / "runtime" / "agent_skills" / selected_skill
        skill_file = skill_dir / "SKILL.md"
        script = skill_dir / "scripts" / "profile_timeseries.py"
        trace: Dict[str, Any] = {
            "skillName": selected_skill,
            "matchedBy": self.last_analysis_skill_trace.get("matchedBy") or "questionUnderstanding+verifiedEvidence",
            "matchTrace": dict(self.last_analysis_skill_trace or {}),
            "activated": False,
            "skillPath": str(skill_file),
            "scriptPath": str(script),
        }
        self.last_analysis_skill_trace = trace
        if not skill_file.exists():
            trace["error"] = "skill package missing"
            return ""
        skill_meta = load_skill_frontmatter(skill_file)
        trace["metadata"] = skill_meta
        if selected_skill != "bi_trend_attribution":
            return self.run_structured_answer_skill(
                selected_skill,
                skill_meta,
                question,
                plan,
                run_result,
                outputs_path,
                rule_context,
                trace,
            )
        if not script.exists():
            trace["error"] = "skill script missing"
            return ""
        artifact_root = Path(outputs_path) if outputs_path else self.llm.settings.resolved_workspace_path / "analysis_skills"
        target = artifact_root / "artifacts" / "analysis_skills" / selected_skill
        target.mkdir(parents=True, exist_ok=True)
        input_path = target / "skill_input.json"
        output_path = target / "skill_output.json"
        payload = answer_data_package(question, plan, run_result, rule_context)
        payload["questionUnderstanding"] = plan.question_understanding
        payload["skillMetadata"] = skill_meta
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        trace.update(
            {
                "activated": True,
                "inputArtifact": str(input_path),
                "outputArtifact": str(output_path),
                "inputRows": len(payload.get("dataRows") or []),
            }
        )
        try:
            completed = subprocess.run(
                [self.llm.settings.python_executable, str(script), "--input", str(input_path), "--output", str(output_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            trace["error"] = str(exc)
            return ""
        trace["returnCode"] = completed.returncode
        trace["stderr"] = completed.stderr[-1000:]
        if completed.returncode != 0 or not output_path.exists():
            trace["error"] = completed.stderr[-1000:] or "skill script failed"
            return ""
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            trace["error"] = "invalid skill output: %s" % exc
            return ""
        trace["outputRows"] = result.get("rowCount", 0)
        trace["findings"] = result.get("findings", [])[:6]
        trace["caveats"] = result.get("caveats", [])[:6]
        answer = str(result.get("answerMarkdown") or "").strip()
        if not answer:
            trace["error"] = "empty skill answer"
        return answer

    def run_structured_answer_skill(
        self,
        skill_name: str,
        skill_meta: Dict[str, Any],
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
        trace: Dict[str, Any] | None = None,
    ) -> str:
        trace = trace if trace is not None else {}
        artifact_root = Path(outputs_path) if outputs_path else self.llm.settings.resolved_workspace_path / "analysis_skills"
        target = artifact_root / "artifacts" / "analysis_skills" / skill_name
        target.mkdir(parents=True, exist_ok=True)
        input_path = target / "skill_input.json"
        output_path = target / "skill_output.json"
        payload = answer_data_package(question, plan, run_result, rule_context)
        payload["questionUnderstanding"] = plan.question_understanding
        payload["skillMetadata"] = skill_meta
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        answer = render_structured_skill_answer(skill_name, payload)
        output = {
            "skillName": skill_name,
            "rowCount": len(payload.get("dataRows") or []),
            "answerMarkdown": answer,
            "caveats": [gap.get("code") for gap in payload.get("evidenceGaps") or [] if isinstance(gap, dict)],
        }
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        trace.update(
            {
                "activated": True,
                "inputArtifact": str(input_path),
                "outputArtifact": str(output_path),
                "inputRows": len(payload.get("dataRows") or []),
                "outputRows": output["rowCount"],
                "deterministicRenderer": True,
            }
        )
        return answer

    def append_business_advice(self, answer: str, intents: List[QuestionIntent], bundle: QueryBundle) -> str:
        if not answer:
            answer = "当前没有足够数据形成结论。"
        answer = re.sub(r"\n\n建议[:：].*$", "", answer.rstrip(), flags=re.S)
        answer = strip_model_advice_lines(answer)
        items = business_advice_items(intents, bundle)
        return answer.rstrip() + "\n\n建议：\n" + "\n".join("- %s" % item for item in items[:2])

    def _correct_metric_total_misread(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        if not re.search(r"(不能|无法|不(?:能|可)直接|不能准确).{0,24}(确认|判断|得到)", answer):
            return answer
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
            lines.append("按日趋势已整理在下方图表中。")
        return "\n".join(lines)

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
        return answer.rstrip() + "\n\n证据门禁：\n" + "\n".join("- %s" % item for item in additions[:8])

    def contextual_suggestions(self, question: str, intents: List[QuestionIntent]) -> List[str]:
        categories = {intent.category for intent in intents}
        if QuestionCategory.TRADE in categories:
            return ["最近7天GMV趋势", "最近30天订单量Top5日期", "昨天支付订单明细"]
        if QuestionCategory.REFUND in categories:
            return ["最近30天退款金额Top5订单", "昨天退款明细", "退款订单对应商品情况"]
        if QuestionCategory.CS_TICKET in categories:
            return ["最近7天客服工单量", "催单工单对应订单状态", "二次开启工单明细"]
        if QuestionCategory.COMPENSATION in categories:
            return ["最近30天赔付金额Top5订单", "赔付订单关联退款状态", "赔付金额趋势"]
        if QuestionCategory.GOODS in categories:
            return ["最近15天新发布商品表现", "商品审核拒绝原因", "上架商品成交情况"]
        return ["最近7天经营概况", "昨天退款明细", "最近30天GMV最高的前5天"]

    def build_sections(self, plan: QueryPlan, run_result: AgentRunResult) -> List[ChatDataSection]:
        sections: List[ChatDataSection] = []
        if not run_result:
            return sections
        intent_map = intent_by_task_id(plan)
        sources: List[Any] = []
        if run_result.task_results:
            sources = [(intent_map.get(item.task_id), item.query_bundle) for item in run_result.task_results]
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
                series_rows = metric_series_rows_for_intent(plan, intent, bundle.rows)
                if series_rows:
                    data_rows = series_rows
            sections.append(
                ChatDataSection(
                    title=title,
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
        friendly = merchant_friendly_data_answer(question, plan, bundle, run_result)
        if friendly:
            return friendly
        successful_task_count = len([item for item in run_result.task_results if not item.query_bundle.failed]) if run_result else 0
        if successful_task_count > 1:
            lines = ["已完成本轮关联指标查询，结果已整理在下方表格中。"]
        else:
            lines = ["已拿到当前查询范围内的结果，明细见下方表格。"]
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
        trace = "，".join(plan.agent_trace[-4:]) if plan.agent_trace else "无 planner trace"
        trace_lower = "，".join(plan.agent_trace).lower() if plan.agent_trace else ""
        if "timeout" in trace_lower or "provider_error" in trace_lower or "planner_provider_error" in trace_lower:
            return "本轮没有实际执行数据查询：Planner LLM 调用超时或失败，QueryGraph 没有生成，因此不能判断为业务为 0。Planner 轨迹：%s。" % trace
        if "json_parse_error" in trace_lower or "planner_json_parse_error" in trace_lower:
            return "本轮没有实际执行数据查询：Planner LLM 返回内容无法解析为 QueryGraph JSON，因此不能判断为业务为 0。Planner 轨迹：%s。" % trace
        if any("planner.no_llm_configured" in item for item in plan.agent_trace):
            return "本轮没有实际执行数据查询：当前未配置可用 LLM，QueryGraph 没有生成，因此不能判断为业务为 0。"
        if any("planner.no_valid_llm_understanding" in item for item in plan.agent_trace):
            return "本轮没有实际执行数据查询：Planner LLM 没有返回可编译的问题理解，因此 QueryGraph 没有生成；这不是业务为 0。Planner 轨迹：%s。" % trace
        if not plan.intents:
            return "本轮没有实际执行数据查询：QueryGraph 没有形成可执行节点，因此不能判断为业务为 0。"
        return "本轮没有实际执行数据查询：QueryGraph 尚未进入 SQL 执行阶段，因此不能判断为业务为 0。Planner 轨迹：%s。" % trace


def merchant_friendly_data_answer(question: str, plan: QueryPlan, bundle: QueryBundle, run_result: AgentRunResult | None = None) -> str:
    rows = bundle.rows or []
    if not rows:
        return ""
    successful_task_count = len([item for item in run_result.task_results if not item.query_bundle.failed]) if run_result else 0
    time_prefix = extract_question_time_phrase(question)
    prefix = "%s，" % time_prefix if time_prefix else "当前查询范围内，"
    if successful_task_count > 1:
        if is_ranking_plan(plan):
            return "%s已整理出排名结果和关联指标，完整明细见下方表格。" % prefix
        return "%s已完成关联指标查询，完整结果见下方表格。" % prefix
    row = rows[0]
    metric_column = primary_answer_metric_column(plan, row)
    if metric_column:
        metric_label = friendly_column_label(plan, metric_column)
        metric_value = format_cell(row.get(metric_column))
        entity_column = primary_entity_column(plan, row)
        if is_ranking_plan(plan) and entity_column:
            entity_label = friendly_column_label(plan, entity_column)
            entity_value = format_cell(row.get(entity_column))
            return "%s%s %s 的%s为 %s，下方表格里有完整排名。" % (prefix, entity_label, entity_value, metric_label, metric_value)
        if len(rows) == 1:
            return "%s%s为 %s。" % (prefix, metric_label, metric_value)
        return "%s%s结果已整理出来，明细见下方表格。" % (prefix, metric_label)
    if len(rows) == 1:
        return "%s已查询到对应结果，详情见下方表格。" % prefix
    return "%s已整理出相关明细，详情见下方表格。" % prefix


def business_advice_items(intents: List[QuestionIntent], bundle: QueryBundle) -> List[str]:
    categories = {intent.category for intent in intents}
    metric_text = " ".join(
        str(item or "")
        for intent in intents
        for item in [
            intent.metric_name,
            intent.metric_column,
            intent.group_by_column,
            (intent.metric_resolution or {}).get("displayName"),
            (intent.metric_resolution or {}).get("metricKey"),
        ]
    ).lower()
    if QuestionCategory.REFUND in categories or QuestionCategory.COMPENSATION in categories:
        return [
            "优先拆到退款原因、商品和订单状态，定位是商品描述、履约还是售后处理导致的退款/赔付。",
            "如果金额或单量集中在少数商品，可以继续看这些商品的下单量、退款率和客服工单，判断是否需要调整商品说明或售后策略。",
        ]
    if QuestionCategory.CS_TICKET in categories:
        return [
            "建议按工单类型、催单/物流/退款场景拆开看，先定位是否集中在少数问题类型。",
            "如果工单量升高，可以联动发货超时订单和退款订单，判断是否是履约异常带来的客服压力。",
        ]
    if QuestionCategory.GOODS in categories:
        return [
            "建议结合商品发布时间、审核状态和近期开单表现，优先处理新上架但转化或售后异常的商品。",
            "如果审核拒绝或质检异常集中在少数类目，可以继续拆拒绝原因、图片资质和标题类目规范。",
        ]
    if QuestionCategory.SCM in categories:
        return [
            "建议把履约量和发货超时、签收超时一起看，先判断问题集中在出库、物流还是签收环节。",
            "如果异常集中在某个仓或某批商品，可以继续拆仓库、商品和订单明细做定位。",
        ]
    if QuestionCategory.TRADE in categories:
        if "gmv" in metric_text or "amt" in metric_text or "金额" in metric_text:
            return [
                "建议按日期趋势和商品/类目拆分 GMV，先判断是整体下滑还是少数商品贡献变化。",
                "可以联动订单量、支付用户数和客单价，判断变化来自流量、转化还是客单价。",
            ]
        return [
            "建议先看按日趋势，确认订单量是否集中在某几天波动，再拆到商品或类目定位来源。",
            "可以联动 GMV、支付用户数和客单价，判断是订单规模变化还是客单价变化。",
        ]
    if bundle.rows:
        return [
            "建议先看时间趋势，再按商品、类目或订单维度拆分，确认异常是否集中。",
            "可以结合退款、工单和履约指标一起看，避免只看单一指标造成误判。",
        ]
    return [
        "建议补充更明确的时间范围或业务对象，我可以继续按同一口径查询。",
        "如果要做分析，可以同时给出主指标和想对比的关联指标。",
    ]


def section_title_for_intent(plan: QueryPlan, intent: QuestionIntent, default: str = "查询结果") -> str:
    resolution = intent.metric_resolution or {}
    metric_key = str(resolution.get("metricKey") or intent.metric_name or "").strip()
    metric_label = str(resolution.get("displayName") or "").strip() or (friendly_column_label(plan, metric_key) if metric_key else "")
    if intent.group_by_column == "pt" and metric_label:
        return "%s趋势" % metric_label
    if metric_label:
        return metric_label
    return intent.preferred_table or default


def metric_series_rows_for_intent(plan: QueryPlan, intent: QuestionIntent, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if intent.group_by_column != "pt" or intent.answer_mode != AnswerMode.GROUP_AGG or not rows:
        return []
    value_column = metric_value_column_for_rows(plan, intent, rows)
    if not value_column:
        return []
    resolution = intent.metric_resolution or {}
    metric_label = str(resolution.get("displayName") or "").strip() or friendly_column_label(plan, value_column)
    series = []
    for row in sorted(rows, key=lambda item: str(item.get("pt") or "")):
        if row.get("pt") in (None, ""):
            continue
        value = answer_numeric_value(row.get(value_column))
        if value is None:
            continue
        series.append({"metric_name": metric_label, "pt": row.get("pt"), "value": value})
    return series


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
    for column in rows[0].keys():
        text = str(column or "")
        if text in {"pt", "seller_id", "merchant_id"} or identifier_like_column(text):
            continue
        if any(answer_numeric_value(row.get(text)) is not None for row in rows):
            return text
    return ""


def answer_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def strip_model_advice_lines(answer: str) -> str:
    cleaned: List[str] = []
    for line in str(answer or "").splitlines():
        text = line.strip()
        if re.match(r"^(建议|建议优先|可以继续追问|如需|如果需要)", text):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def primary_summary_metric_value(plan: QueryPlan, run_result: AgentRunResult) -> Dict[str, Any]:
    if not run_result:
        return {}
    for item in visible_successful_tasks(plan, run_result):
        intent = intent_by_task_id(plan).get(item.task_id)
        if answer_result_role(intent) != "summary" or not item.query_bundle.rows:
            continue
        rows = item.query_bundle.rows
        value_column = metric_value_column_for_rows(plan, intent, rows) if intent else ""
        if not value_column:
            continue
        value = rows[0].get(value_column)
        if value in (None, ""):
            continue
        resolution = intent.metric_resolution or {}
        return {
            "metricKey": str(resolution.get("metricKey") or intent.metric_name or value_column),
            "label": str(resolution.get("displayName") or friendly_column_label(plan, value_column)),
            "value": value,
            "taskId": item.task_id,
        }
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
        return metric_series_rows_for_intent(plan, intent, item.query_bundle.rows) if intent else []
    return []


def extract_question_time_phrase(question: str) -> str:
    text = str(question or "")
    for pattern in [
        r"最近\s*\d+\s*[天日周月]",
        r"近\s*\d+\s*[天日周月]",
        r"过去\s*\d+\s*[天日周月]",
        r"昨天",
        r"今日",
        r"今天",
        r"本周",
        r"本月",
    ]:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", "", match.group(0))
    return ""


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
    for column, value in (row or {}).items():
        text = str(column or "")
        if identifier_like_column(text):
            continue
        if isinstance(value, (int, float)) or re.fullmatch(r"-?\d+(\.\d+)?", str(value or "")):
            return text
    return ""


def primary_entity_column(plan: QueryPlan, row: Dict[str, Any]) -> str:
    available = set(str(key) for key in (row or {}).keys())
    for column in primary_summary_entity_columns(plan):
        if column in available and row.get(column) not in (None, ""):
            return column
    for column in ["spu_name", "spu_id", "sub_order_id", "order_id", "refund_id", "ticket_id", "bill_id"]:
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


def humanize_column_name(column: str) -> str:
    text = str(column or "").strip()
    dictionary = {
        "order": "订单",
        "detail": "明细",
        "cnt": "数量",
        "amt": "金额",
        "gmv": "GMV",
        "refund": "退款",
        "return": "退货",
        "rate": "比例",
        "pay": "支付",
        "user": "用户",
        "ticket": "工单",
        "goods": "商品",
        "spu": "商品",
        "create": "创建",
        "time": "时间",
    }
    parts = [dictionary.get(part, "") for part in re.split(r"[_\s]+", text.lower())]
    label = "".join(part for part in parts if part)
    return label or text


def identifier_like_column(column: str) -> bool:
    text = str(column or "").strip().lower()
    return text in {"seller_id", "merchant_id", "user_id", "pt"} or text.endswith("_id") or text.endswith("_no")


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
    for column in ["pt", "order_id", "sub_order_id", "spu_id", "spu_name", "refund_id", "ticket_id", "bill_id"]:
        if column in available and column not in preferred:
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


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], labels: Dict[str, str] | None = None) -> str:
    labels = labels or {}
    header = "| %s |" % " | ".join(labels.get(column, column) for column in columns)
    divider = "| %s |" % " | ".join("---" for _ in columns)
    body = []
    for row in rows:
        body.append("| %s |" % " | ".join(format_cell(row.get(column, "")) for column in columns))
    return "\n".join([header, divider] + body)


def business_summary_table(plan: QueryPlan, run_result: AgentRunResult) -> str:
    succeeded = [item for item in run_result.task_results if not item.query_bundle.failed and item.query_bundle.rows]
    if len(succeeded) <= 1:
        return ""
    intent_by_task = {intent.plan_task_id: intent for intent in plan.intents}
    ordered = sorted(succeeded, key=lambda item: task_evidence_priority(intent_by_task.get(item.task_id), item, plan))
    support_ids = support_task_ids_for_answer(plan)
    visible = [item for item in ordered if answer_visible_task(intent_by_task.get(item.task_id), item, plan, support_ids)]
    if len(visible) <= 1:
        return ""
    base = first_entity_summary_task(visible, intent_by_task)
    if not base:
        return ""
    merged_rows = merge_visible_task_rows(base, [item for item in visible if item.task_id != base.task_id])
    if not merged_rows:
        return ""
    columns = business_summary_columns(plan, merged_rows)
    if not columns:
        return ""
    labels = answer_column_labels(plan)
    return markdown_table(merged_rows[:8], columns, labels)


def first_entity_summary_task(items: List[Any], intent_by_task: Dict[str, QuestionIntent]) -> Any | None:
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
    for item in other_items:
        rows = item.query_bundle.rows or []
        for target in merged:
            match = first_matching_row(target, rows)
            if not match:
                continue
            for key, value in match.items():
                if key not in target or target.get(key) in (None, ""):
                    target[key] = value
    return merged


def first_matching_row(base: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    base_has_entity = any(summary_business_entity_key(key) for key in base)
    for key in ["spu_id", "spu_name", "order_id", "sub_order_id", "refund_id", "ticket_id", "bill_id"]:
        base_value = normalized_cell(base.get(key))
        if not base_value:
            continue
        for row in rows:
            if normalized_cell(row.get(key)) == base_value:
                return row
    if not base_has_entity and len(rows) == 1:
        return rows[0]
    return None


def summary_business_entity_key(column: str) -> bool:
    text = str(column or "").strip().lower()
    return text in {"spu_id", "spu_name", "order_id", "sub_order_id", "refund_id", "ticket_id", "bill_id"}


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
    for column in ["refund_bill_cnt", "pay_amt", "repay_bill_cnt", "order_detail_cnt"]:
        if column in available and column not in preferred:
            preferred.append(column)
    for intent in plan.intents:
        for column in [intent.group_by_column] + intent.output_keys:
            if column and column in available and column not in preferred and summary_column_allowed(column, entity_columns):
                preferred.append(column)
    for column in ["spu_apply_create_time"]:
        if column in available and column not in preferred:
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
    if text in {"seller_id", "merchant_id"}:
        return False
    if text == "pt" and "pt" not in entity_columns:
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
    if group_by in {"spu_id", "spu_name"}:
        return ["spu_id", "spu_name"]
    if group_by == "order_id":
        return ["order_id", "sub_order_id"]
    if group_by == "sub_order_id":
        return ["sub_order_id", "order_id"]
    if group_by in {"refund_id", "ticket_id", "bill_id"}:
        return [group_by, "order_id", "sub_order_id"]
    return ["spu_id", "spu_name", "order_id", "sub_order_id", "refund_id", "ticket_id", "bill_id", "pt"]


def answer_column_labels(plan: QueryPlan) -> Dict[str, str]:
    labels: Dict[str, str] = {
        "spu_id": "SPU ID",
        "spu_name": "商品",
        "order_id": "订单号",
        "sub_order_id": "子订单号",
        "refund_id": "退款单号",
        "ticket_id": "工单号",
        "bill_id": "赔付单号",
        "pt": "日期",
        "spu_apply_create_time": "商品发布时间",
        "pay_amt": "退款金额",
        "refund_bill_cnt": "退款单量",
        "repay_bill_cnt": "赔付单量",
        "order_detail_cnt": "下单数",
    }
    for intent in plan.intents:
        resolution = intent.metric_resolution or {}
        metric = str(resolution.get("metricKey") or intent.metric_name or "").strip()
        display = str(resolution.get("displayName") or "").strip()
        if metric and display:
            labels[metric] = display
    return labels


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
        title = task_evidence_title(intent, item)
        tables = "、".join(bundle.tables) if bundle.tables else (intent.preferred_table if intent else "")
        location = tables or ("派生计算" if intent and intent.answer_mode == AnswerMode.DERIVED else "unknown_table")
        lines.append("- %s（%s）：%s 行。" % (title, location, bundle.effective_row_count()))
        section_plan = QueryPlan(intents=[intent]) if intent else plan
        columns = fallback_display_columns(section_plan, bundle.rows)
        if columns:
            lines.append(markdown_table(bundle.rows[:4], columns, answer_column_labels(section_plan)))
    failed = [item for item in run_result.task_results if item.query_bundle.failed]
    for item in failed[:3]:
        lines.append("- %s：执行失败，%s" % (item.task_id, (item.query_bundle.error or item.summary)[:160]))
    return "\n".join(lines)


def task_evidence_title(intent: QuestionIntent | None, item: Any) -> str:
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
    if intent.preferred_table == "dwm_goods_detail_df" and "spu_apply_create_time" in (intent.output_keys or []):
        return "商品发布时间"
    if metric:
        return metric
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
    if metric_key and source_phrase and not source_phrase_in_question(source_phrase, intent.question):
        refs = required_evidence_refs(plan.question_understanding or {})
        ranking_refs = ranking_metric_refs(plan.question_understanding or {})
        if metric_key not in refs["metrics"] and metric_key not in ranking_refs:
            return False
    return True


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


def format_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ")[:80]


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


def answer_skill_required(plan: QueryPlan, run_result: AgentRunResult | None = None, has_rule_context: bool = False) -> bool:
    return bool(select_answer_skill(plan, run_result, has_rule_context))


def select_answer_skill(plan: QueryPlan, run_result: AgentRunResult | None = None, has_rule_context: bool = False) -> str:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip().lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    categories = {intent.category for intent in plan.intents}
    if plan_requires_rule_evidence(plan) and has_rule_context:
        return "rule_compliance"
    if plan_has_ratio_calculation(plan):
        return "ratio_analysis"
    if plan_has_new_product_risk_shape(plan):
        return "new_product_risk"
    if analysis_intent in {"trend_check", "anomaly_check", "diagnosis", "comparison", "overview"} and requires_explanation:
        return "bi_trend_attribution"
    risk_domains = {QuestionCategory.REFUND, QuestionCategory.COMPENSATION, QuestionCategory.CS_TICKET, QuestionCategory.GOODS, QuestionCategory.SCM}
    if analysis_intent in {"risk_ranking", "diagnosis", "anomaly_check"} and categories & risk_domains:
        return "risk_analysis"
    if run_result and run_result.evidence_gaps and analysis_intent not in {"none", ""}:
        return "risk_analysis"
    return ""


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


def plan_has_new_product_risk_shape(plan: QueryPlan) -> bool:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "").lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    if analysis_intent not in {"risk_ranking", "diagnosis", "anomaly_check"} and not requires_explanation:
        return False
    categories = {intent.category for intent in plan.intents}
    if QuestionCategory.GOODS not in categories:
        return False
    lifecycle_terms = {"publish", "published", "online", "spu_apply_create_time", "goods_online", "商品发布时间", "发布时间", "上架"}
    for intent in plan.intents:
        payload = json.dumps(
            {
                "metric": intent.metric_name,
                "columns": intent.output_keys + intent.required_evidence,
                "resolution": intent.metric_resolution,
            },
            ensure_ascii=False,
            default=str,
        )
        if any(term in payload for term in lifecycle_terms):
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
    return group_by in {"", "merchant_id", "seller_id"}


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
    return text == "spu_name" or text.endswith("_id") or text in {"id", "order_no", "bill_no"}


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
) -> Dict[str, Any]:
    if not run_result:
        return {
            "question": question,
            "dataRows": [],
            "metricDisclosures": [],
            "evidenceGaps": [],
            "ruleEvidence": compact_rule_evidence(question, rule_context),
        }
    verified = run_result.verified_evidence
    return {
        "question": question,
        "tables": run_result.merged_query_bundle.tables,
        "rowCount": run_result.merged_query_bundle.effective_row_count(),
        "dataRows": answer_data_rows(plan, run_result),
        "dataSections": answer_prompt_sections(plan, run_result),
        "metricDisclosures": metric_disclosures(plan, verified),
        "evidenceGaps": compact_evidence_gaps(run_result.evidence_gaps),
        "ruleEvidence": compact_rule_evidence(question, rule_context),
    }


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
    group_columns = {str(intent.group_by_column or "").strip() for intent in metric_tasks}
    return group_columns <= {"", "pt", "merchant_id", "seller_id"} or "pt" in group_columns


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
    if intent.group_by_column == "pt" and intent.answer_mode == AnswerMode.GROUP_AGG:
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
        if intent.metric_specs:
            for spec in intent.metric_specs:
                if not isinstance(spec, dict):
                    continue
                disclosures.append(
                    {
                        "metricKey": spec.get("metricName") or spec.get("metric_key") or spec.get("metricColumn"),
                        "ownerTable": intent.preferred_table,
                        "formula": spec.get("metricFormula") or spec.get("formula"),
                        "sourceColumns": spec.get("sourceColumns") or ([spec.get("metricColumn")] if spec.get("metricColumn") else []),
                    }
                )
            continue
        resolution = intent.metric_resolution or {}
        if resolution:
            disclosures.append(
                {
                    key: resolution.get(key)
                    for key in [
                        "requestedMetricRef",
                        "metricKey",
                        "ownerTable",
                        "sourceColumns",
                        "formula",
                        "displayName",
                        "fieldWarning",
                    ]
                    if resolution.get(key) not in (None, "", [])
                }
            )
        elif intent.metric_formula or intent.metric_name:
            disclosures.append(
                {
                    "metricKey": intent.metric_name or intent.metric_column,
                    "ownerTable": intent.preferred_table,
                    "formula": intent.metric_formula,
                    "sourceColumns": [intent.metric_column] if intent.metric_column else [],
                }
            )
    for item in getattr(verified, "derived_evidence", [])[:8]:
        if isinstance(item, dict):
            disclosures.append(
                {
                    key: item.get(key)
                    for key in ["metric", "formula", "sourceColumns", "fieldWarning"]
                    if item.get(key) not in (None, "", [])
                }
            )
    return [item for item in dedupe_dicts(disclosures) if item]


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


def render_structured_skill_answer(skill_name: str, payload: Dict[str, Any]) -> str:
    rows = payload.get("dataRows") or []
    disclosures = payload.get("metricDisclosures") or []
    gaps = payload.get("evidenceGaps") or []
    rule_evidence = payload.get("ruleEvidence") or []
    question = str(payload.get("question") or "")
    title = {
        "risk_analysis": "风险分析",
        "rule_compliance": "规则合规分析",
        "ratio_analysis": "占比分析",
        "new_product_risk": "新品风险分析",
    }.get(skill_name, "分析结果")
    lines = ["%s：" % title]
    if skill_name == "ratio_analysis":
        lines.extend(ratio_skill_lines(disclosures, rows))
    elif skill_name == "rule_compliance":
        lines.extend(rule_compliance_skill_lines(rule_evidence, rows))
    elif skill_name == "new_product_risk":
        lines.extend(new_product_risk_skill_lines(rows, disclosures))
    else:
        lines.extend(risk_skill_lines(rows, disclosures, question))
    if gaps:
        lines.append("")
        lines.append("证据缺口：")
        for gap in gaps[:5]:
            if not isinstance(gap, dict):
                continue
            lines.append("- %s：%s" % (gap.get("code") or "EVIDENCE_GAP", gap.get("reason") or gap.get("answerInstruction") or "证据未完全覆盖"))
    lines.append("")
    lines.append("说明：以上结论只基于已验证查询结果和语义层指标口径。")
    return "\n".join(line for line in lines if line is not None).strip()


def ratio_skill_lines(disclosures: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[str]:
    derived = [
        item
        for item in disclosures
        if isinstance(item, dict) and ("/" in str(item.get("formula") or "") or "share" in str(item.get("metricKey") or ""))
    ]
    lines: List[str] = []
    if derived:
        item = derived[0]
        lines.append("- 公式：%s = %s" % (item.get("metricKey") or item.get("displayName") or "派生占比", item.get("formula") or "分子 / 分母"))
    else:
        lines.append("- 公式：按语义层解析出的分子 / 分母计算，占比必须同时具备两侧证据。")
    if rows:
        lines.append("- 证据样例：%s" % compact_row_preview(rows[0]))
    else:
        lines.append("- 当前没有可用于展示的结果行。")
    lines.append("- 判断：若分子、分母任一缺失，不把缺失值解释为 0。")
    return lines


def rule_compliance_skill_lines(rule_evidence: List[str], rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if rule_evidence:
        lines.append("- 规则依据：%s" % "；".join(rule_evidence[:3]))
    else:
        lines.append("- 规则依据：当前召回规则证据不足，不能形成强合规判断。")
    if rows:
        lines.append("- 数据证据：%s" % compact_row_preview(rows[0]))
    else:
        lines.append("- 数据证据：当前没有可用于对照规则的结果行。")
    lines.append("- 判断：只在规则证据给出明确口径/条件时判断风险，不用高指标值直接推断违规。")
    return lines


def new_product_risk_skill_lines(rows: List[Dict[str, Any]], disclosures: List[Dict[str, Any]]) -> List[str]:
    lines = ["- 识别口径：只有存在商品发布/上架/审核生命周期证据时，才标记为新品风险。"]
    if rows:
        for index, row in enumerate(rows[:5], 1):
            lines.append("- 候选 %d：%s" % (index, compact_row_preview(row)))
    else:
        lines.append("- 当前没有可排序的商品结果行。")
    if disclosures:
        lines.append("- 指标口径：%s" % "; ".join(compact_disclosure(item) for item in disclosures[:4]))
    lines.append("- 判断：优先关注同时具备新品生命周期证据和退款/赔付/工单压力的商品。")
    return lines


def risk_skill_lines(rows: List[Dict[str, Any]], disclosures: List[Dict[str, Any]], question: str) -> List[str]:
    lines: List[str] = []
    if rows:
        lines.append("- 优先项：")
        for index, row in enumerate(rows[:5], 1):
            lines.append("  %d. %s" % (index, compact_row_preview(row)))
    else:
        lines.append("- 当前没有可排序的风险结果行。")
    if disclosures:
        lines.append("- 使用指标：%s" % "; ".join(compact_disclosure(item) for item in disclosures[:5]))
    lines.append("- 判断：把金额、单量、比例、工单/赔付证据共同出现的对象作为高优先级，缺证据时只给风险提示。")
    return lines


def compact_disclosure(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    name = item.get("displayName") or item.get("metricKey") or item.get("metric") or "指标"
    formula = item.get("formula") or ""
    return ("%s=%s" % (name, formula))[:120] if formula else str(name)[:80]


def compact_row_preview(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return str(row)[:180]
    parts: List[str] = []
    for key, value in list(row.items())[:8]:
        parts.append("%s=%s" % (key, format_cell(value)))
    return "，".join(parts)[:220]


def answer_skill_headers(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    headers: List[Dict[str, Any]] = []
    for skill_file in sorted(root.glob("*/SKILL.md")):
        meta = load_skill_frontmatter(skill_file)
        name = str(meta.get("name") or skill_file.parent.name)
        if not name:
            continue
        headers.append(
            {
                "name": name,
                "description": str(meta.get("description") or "")[:500],
                "path": str(skill_file.relative_to(root.parent.parent) if root.parent.parent in skill_file.parents else skill_file),
            }
        )
    return headers


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
    def __init__(self, doris_repository: DorisRepository):
        self.doris_repository = doris_repository

    def report(self, merchant_id: str) -> DailyReportResponse:
        target = merchant_id or "100"
        metrics: Dict[str, Any] = {}
        merchant_name = "yshopping商家%s" % target
        try:
            row = self.doris_repository.query_one(
                """
                SELECT *
                FROM ads_merchant_profile
                WHERE merchant_id = %s
                ORDER BY pt DESC
                LIMIT 1
                """,
                [target],
            )
            if row:
                merchant_name = str(row.get("merchant_name") or merchant_name)
                mapping = {
                    "昨日总gmv金额": "order_gmv_amt_1d",
                    "昨日下单用户量": "order_user_cnt_1d",
                    "昨日总订单量": "order_cnt_1d",
                    "昨日交易成功订单量": "trade_success_order_cnt_1d",
                    "昨日退货量": "refund_order_cnt_1d",
                    "昨日退款金额": "refund_amt_1d",
                }
                metrics = {label: row.get(column, 0) for label, column in mapping.items()}
        except Exception:
            pass
        if not metrics:
            metrics = {
                "昨日总gmv金额": 0,
                "昨日下单用户量": 0,
                "昨日总订单量": 0,
                "昨日交易成功订单量": 0,
                "昨日退货量": 0,
                "昨日退款金额": 0,
            }
        return DailyReportResponse(
            merchant_id=target,
            merchant_name=merchant_name,
            date=date.today().isoformat(),
            metrics=metrics,
            suggestions=[
                "优先关注订单、退款和客服工单的异常波动。",
                "可继续追问具体日期、Top 订单或明细，我会基于当前经营口径展开。",
            ],
        )


class FeedbackService:
    def __init__(
        self,
        answer_repository: AnswerRepository,
        pending_store: PendingAnswerStore,
        memory_store: Optional[StructuredMemoryStore] = None,
    ):
        self.answer_repository = answer_repository
        self.pending_store = pending_store
        self.memory_store = memory_store

    def apply_feedback(self, answer_id: str, adopted: Any, liked: Any, disliked: Any) -> bool:
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
        if self.memory_store is not None and pending:
            try:
                self.memory_store.update_from_feedback(pending, adopted=adopted, liked=liked, disliked=disliked)
            except Exception:
                pass
        return persisted


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
) -> ChatContext:
    primary = plan.intents[0] if plan.intents else QuestionIntent()
    return ChatContext(
        question=question,
        days=primary.days,
        category=category_display(primary.category),
        answer_mode=primary.answer_mode.value if hasattr(primary.answer_mode, "value") else str(primary.answer_mode),
        topic=joined_categories(plan),
        data_catalog=",".join(table for section in sections for table in section.doris_tables),
        merchant_profile=merchant.profile_markdown(),
        context_summary="QueryGraph nodes=%s, sections=%s" % (len(plan.intents), len(sections)),
        pending_clarification_stage=pending_stage,
        pending_clarification_type=pending_type,
        pending_question=question if pending_stage else "",
        pending_clarification_options=pending_options or [],
    )
