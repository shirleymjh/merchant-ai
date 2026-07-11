from __future__ import annotations

import json
import re
import subprocess
import uuid
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
    VerifiedAnswerContext,
    category_display,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.memory import MemoryStore
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, PendingAnswerStore
from merchant_ai.services.answer_formatting import (
    answer_numeric_value,
    extract_question_time_phrase,
    format_cell,
    format_metric_value_for_answer,
    humanize_column_name,
    identifier_like_column,
)


def answer_context_policy() -> str:
    return (
        "AnswerAgent 只读取 VerifiedAnswerContext 中的 question、businessContext、dataRows、dataSections、metricDisclosures、evidenceGaps、degradedReasons、analysisDraft；不要读取或推断 QueryGraph。"
        "你的输出面向商家，不面向研发或分析师；语气要像经营助手，先直接回答用户问题，再给必要说明和建议。"
        "不要使用“分析结论”“关键证据”“限制”“证据门禁”“当前证据显示”“已看到的点位显示”这类报告或内部调试话术。"
        "不要说“查到几行”“使用表”“SQL”“字段名”“Doris”；不要输出 markdown 表格，表格和图表由前端结构化区域渲染。"
        "核心经营指标可以默认保留一句业务口径说明，只说统计对象、时间和店铺范围；用户没有问口径时，不要展开字段、来源表和计算公式。"
        "同一指标存在多个候选口径时，只回答语义层确认的主口径，不要把多个相似口径并列解释。"
        "用户提到和后台/看板数据不一致时，进入口径对账思路，优先说明时间口径、订单状态、是否扣退款、商品粒度和数据更新时间。"
        "如果是趋势问题，第一段直接写“最近N天，指标从 A 变化到 B，整体上升/下降 C。”，不要写“趋势里”“点位显示”；有峰值和低点时用一句话说明。"
        "dataRows 或 dataSections 中 resultRole=summary 的行是已验证汇总结果，优先用于回答总量；resultRole=trend_context 的行只用于解释趋势。"
        "不要因为趋势只有部分日期有点位，就否定 summary 汇总；不要说“其余日期没有看到明细”。"
        "如果 evidenceGaps 存在，用“说明：”简短提示，不要扩大成失败结论。"
        "最后输出“建议：”，用短横线列出最多 2 条；建议必须结合 businessContext 的商家画像、长期记忆/近期关注和本轮数据，避免泛泛说继续追问。"
    )


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
        personalization_context: Optional[Dict[str, Any]] = None,
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
        bundle = run_result.merged_query_bundle if run_result else QueryBundle()
        if analysis_summary:
            cleaned_summary = sanitize_business_answer_text(analysis_summary, question, plan, run_result)
            answer = cleaned_summary
            return self._apply_answer_guard(
                self._append_lightweight_metric_disclosure(
                    self._append_rule_evidence(
                        self.append_business_advice(
                            answer,
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
                ),
                run_result,
            )
        if primary.intent_type == "VALID" and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT} and (not run_result or not run_result.task_results):
            return self._no_execution_answer(plan)
        if run_result and run_result.task_results and all(result.query_bundle.failed for result in run_result.task_results):
            return self._apply_answer_guard(
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
                        allow_llm=allow_llm,
                    ),
                    question,
                    effective_rule_context,
                ),
                run_result,
            )
        if question_asks_metric_reconciliation(question):
            reconciliation_answer = self._metric_reconciliation_answer(question, plan, run_result)
            if reconciliation_answer:
                return self._apply_answer_guard(
                    self._append_rule_evidence(
                        self.append_business_advice(
                            reconciliation_answer,
                            plan.intents,
                            bundle,
                            question=question,
                            plan=plan,
                            run_result=run_result,
                            merchant=merchant,
                            personalization_context=personalization_context,
                            allow_llm=allow_llm,
                        ),
                        question,
                        effective_rule_context,
                    ),
                    run_result,
                )
        if allow_llm and self.llm.configured and (bundle.rows or run_result.evidence_gaps):
            answer = self._compose_llm_business_answer(
                question,
                plan,
                run_result,
                rule_context,
                merchant,
                personalization_context,
            )
            if answer:
                return self._apply_answer_guard(
                    self._append_lightweight_metric_disclosure(
                        self._append_rule_evidence(
                            self.append_business_advice(
                                answer,
                                plan.intents,
                                bundle,
                                question=question,
                                plan=plan,
                                run_result=run_result,
                                merchant=merchant,
                                personalization_context=personalization_context,
                                allow_llm=allow_llm,
                            ),
                            question,
                            effective_rule_context,
                        ),
                        question,
                        plan,
                        run_result,
                    ),
                    run_result,
                )
        return self._apply_answer_guard(
            self._append_lightweight_metric_disclosure(
                self._append_rule_evidence(
                    self.append_business_advice(
                        self._fallback_data_answer(question, plan, bundle, run_result),
                        plan.intents,
                        bundle,
                        question=question,
                        plan=plan,
                        run_result=run_result,
                        merchant=merchant,
                        personalization_context=personalization_context,
                        allow_llm=allow_llm,
                    ),
                    question,
                    effective_rule_context,
                ),
                question,
                plan,
                run_result,
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
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.last_analysis_skill_trace = {}
        if not run_result or not run_result.merged_query_bundle.rows:
            return ""
        if not analysis_summary_required(plan) and not answer_skill_required(plan, run_result, bool(rule_context)):
            return ""
        skill_name = self.propose_answer_skill(question, plan, run_result, bool(rule_context))
        if not skill_name and not analysis_summary_required(plan):
            return ""
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
        answer = self._ensure_multi_metric_summary_coverage(answer, question, plan, run_result)
        answer = self._correct_metric_total_misread(answer, question, plan, run_result)
        answer = self._clean_summary_trend_misphrasing(answer, plan, run_result)
        return sanitize_business_answer_text(answer, question, plan, run_result)

    def propose_answer_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        has_rule_context: bool = False,
    ) -> str:
        candidates = answer_skill_headers(self.llm.settings.resources_root / "runtime" / "agent_skills")
        fallback = select_answer_skill(plan, run_result, has_rule_context)
        match_mode = str(self.llm.settings.answer_skill_match_mode or "").lower()
        if not fallback and (match_mode == "always" or bool(getattr(self.llm.settings, "skill_confirmation_required", False))):
            fallback = deterministic_analysis_skill_fallback(plan, run_result, has_rule_context)
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
        if fallback and (match_mode in {"deterministic_first", "header"} or (match_mode == "always" and fallback == "bi_trend_attribution")):
            trace["matchedBy"] = "deterministic_fallback_before_llm"
            trace["skillName"] = fallback
            return fallback
        if not self.llm.configured:
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
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        selected_skill = skill_name or select_answer_skill(plan, run_result, bool(rule_context)) or "bi_trend_attribution"
        if self._should_use_local_fast_structured_skill(selected_skill, plan, run_result):
            trace = {
                "skillName": selected_skill,
                "matchedBy": self.last_analysis_skill_trace.get("matchedBy") or "semantic_fast_path",
                "matchTrace": dict(self.last_analysis_skill_trace or {}),
                "activated": True,
                "workerType": "LOCAL_STRUCTURED_RENDERER",
                "subAgentType": "local_fast_summary",
                "isolatedExecution": False,
                "deterministicRenderer": True,
                "lifecycleStage": "completed",
                "progress": ["matched", "local_structured_render", "completed"],
                "reuseCandidate": False,
            }
            skill_dir = self.llm.settings.resources_root / "runtime" / "agent_skills" / selected_skill
            skill_file = skill_dir / "SKILL.md"
            skill_meta = load_skill_frontmatter(skill_file) if skill_file.exists() else {}
            payload = answer_data_package(
                question,
                plan,
                run_result,
                rule_context,
                merchant=merchant,
                personalization_context=personalization_context,
            )
            payload["questionUnderstanding"] = plan.question_understanding
            payload["skillMetadata"] = skill_meta
            answer = render_structured_skill_answer(selected_skill, payload)
            trace["inputRows"] = len(payload.get("dataRows") or [])
            trace["outputRows"] = len(payload.get("dataRows") or [])
            trace["summaryChars"] = len(answer)
            self.last_analysis_skill_trace = trace
            return answer
        if self._should_run_isolated_skill_worker(selected_skill):
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
        isolated_run_id = "skill_%s_%s" % (selected_skill, uuid.uuid4().hex[:10])
        skill_dir = self.llm.settings.resources_root / "runtime" / "agent_skills" / selected_skill
        skill_file = skill_dir / "SKILL.md"
        script = skill_dir / "scripts" / "profile_timeseries.py"
        artifact_root = Path(outputs_path) if outputs_path else self.llm.settings.resolved_workspace_path / "analysis_skills"
        target = artifact_root / "artifacts" / "analysis_skills" / selected_skill / "runs" / isolated_run_id
        checkpoint_path = target / "skill_checkpoint.json"
        trace: Dict[str, Any] = {
            "skillName": selected_skill,
            "matchedBy": self.last_analysis_skill_trace.get("matchedBy") or "questionUnderstanding+verifiedEvidence",
            "matchTrace": dict(self.last_analysis_skill_trace or {}),
            "activated": False,
            "skillPath": str(skill_file),
            "scriptPath": str(script),
            "lifecycleStage": "matched",
            "requiresConfirmation": bool(self.llm.settings.skill_confirmation_required),
            "confirmed": not bool(self.llm.settings.skill_confirmation_required),
            "isolatedRunId": isolated_run_id,
            "workspacePath": str(target),
            "checkpointPath": str(checkpoint_path),
            "progress": ["matched"],
            "reuseCandidate": False,
        }
        self.last_analysis_skill_trace = trace
        if not skill_file.exists():
            trace["error"] = "skill package missing"
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:skill package missing")
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
                merchant=merchant,
                personalization_context=personalization_context,
            )
        if not script.exists():
            trace["error"] = "skill script missing"
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:skill script missing")
            return ""
        target.mkdir(parents=True, exist_ok=True)
        input_path = target / "skill_input.json"
        output_path = target / "skill_output.json"
        payload = answer_data_package(
            question,
            plan,
            run_result,
            rule_context,
            merchant=merchant,
            personalization_context=personalization_context,
        )
        payload["questionUnderstanding"] = plan.question_understanding
        payload["skillMetadata"] = skill_meta
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        trace.update(
            {
                "activated": True,
                "inputArtifact": str(input_path),
                "outputArtifact": str(output_path),
                "inputRows": len(payload.get("dataRows") or []),
                "lifecycleStage": "isolated_execute",
                "progress": trace["progress"] + ["confirmed" if trace["confirmed"] else "awaiting_confirmation", "isolated_execute"],
            }
        )
        self._write_skill_checkpoint(checkpoint_path, trace, status="running")
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
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:%s" % str(exc)[:80])
            self._write_skill_checkpoint(checkpoint_path, trace, status="failed")
            return ""
        trace["returnCode"] = completed.returncode
        trace["stderr"] = completed.stderr[-1000:]
        if completed.returncode != 0 or not output_path.exists():
            trace["error"] = completed.stderr[-1000:] or "skill script failed"
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:skill script failed")
            self._write_skill_checkpoint(checkpoint_path, trace, status="failed")
            return ""
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            trace["error"] = "invalid skill output: %s" % exc
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:invalid output")
            self._write_skill_checkpoint(checkpoint_path, trace, status="failed")
            return ""
        trace["outputRows"] = result.get("rowCount", 0)
        trace["findings"] = result.get("findings", [])[:6]
        trace["caveats"] = result.get("caveats", [])[:6]
        trace["lifecycleStage"] = "completed"
        trace["progress"].extend(["progress_synced", "completed"])
        trace["reuseCandidate"] = bool(self.llm.settings.skill_reuse_suggestion_enabled and answer_skill_reuse_candidate(selected_skill, result))
        self._write_skill_checkpoint(checkpoint_path, trace, status="completed")
        answer = str(result.get("answerMarkdown") or "").strip()
        if not answer:
            trace["error"] = "empty skill answer"
            trace["lifecycleStage"] = "failed"
            trace["progress"].append("failed:empty answer")
            self._write_skill_checkpoint(checkpoint_path, trace, status="failed")
        return answer

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
        self.last_analysis_skill_trace = {
            "skillName": "parallel_skill_batch",
            "skillNames": [str(name) for name in skill_names or [] if str(name).strip()],
            "activated": bool(results),
            "executionMode": "parallel_isolated_skill_workers",
            "workerType": "SKILL_WORKER_BATCH",
            "subAgentType": "SKILL_WORKER_BATCH",
            "isolatedExecution": True,
            "parallelExecution": True,
            "lifecycleStage": "completed" if successful else "failed",
            "progress": ["matched", "parallel_isolated_execute", "progress_synced", "completed" if successful else "failed"],
            "skillBatchResults": [result.trace for result in results],
            "completedCount": len(successful),
            "failedCount": len(results) - len(successful),
            "summaryChars": len(summary),
            "reuseCandidate": any(bool(result.trace.get("reuseCandidate")) for result in results),
        }
        if not successful:
            self.last_analysis_skill_trace["error"] = "all parallel skill workers failed"
        return summary

    def _should_run_isolated_skill_worker(self, skill_name: str) -> bool:
        if not bool(getattr(self.llm.settings, "skill_worker_enabled", True)):
            return False
        configured = str(getattr(self.llm.settings, "skill_worker_complex_names", "") or "")
        names = {item.strip() for item in configured.split(",") if item.strip()}
        return not names or skill_name in names

    def _should_use_local_fast_structured_skill(
        self,
        skill_name: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
    ) -> bool:
        if skill_name not in {"risk_analysis", "new_product_risk", "ratio_analysis"}:
            return False
        if not run_result or not run_result.merged_query_bundle.rows:
            return False
        trace = " ".join(str(item or "") for item in (plan.agent_trace or []))
        source = str((plan.question_understanding or {}).get("source") or "")
        if "semantic_topn_metric_fast_path" not in source and "planner.semantic_fast_path" not in trace:
            return False
        if run_result.evidence_gaps:
            return False
        return True

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
        merchant: MerchantInfo | None = None,
        personalization_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        trace = trace if trace is not None else {}
        artifact_root = Path(outputs_path) if outputs_path else self.llm.settings.resolved_workspace_path / "analysis_skills"
        isolated_run_id = str(trace.get("isolatedRunId") or "skill_%s_%s" % (skill_name, uuid.uuid4().hex[:10]))
        target = artifact_root / "artifacts" / "analysis_skills" / skill_name / "runs" / isolated_run_id
        checkpoint_path = Path(str(trace.get("checkpointPath") or target / "skill_checkpoint.json"))
        target.mkdir(parents=True, exist_ok=True)
        input_path = target / "skill_input.json"
        output_path = target / "skill_output.json"
        payload = answer_data_package(
            question,
            plan,
            run_result,
            rule_context,
            merchant=merchant,
            personalization_context=personalization_context,
        )
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
                "lifecycleStage": "completed",
                "isolatedRunId": isolated_run_id,
                "workspacePath": str(target),
                "checkpointPath": str(checkpoint_path),
                "progress": list(dict.fromkeys((trace.get("progress") or ["matched"]) + ["confirmed", "isolated_execute", "progress_synced", "completed"])),
                "reuseCandidate": bool(self.llm.settings.skill_reuse_suggestion_enabled and answer_skill_reuse_candidate(skill_name, output)),
            }
        )
        self._write_skill_checkpoint(checkpoint_path, trace, status="completed")
        return answer

    def _write_skill_checkpoint(self, path: Path, trace: Dict[str, Any], status: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "skillName": trace.get("skillName"),
                        "isolatedRunId": trace.get("isolatedRunId"),
                        "status": status,
                        "stage": trace.get("lifecycleStage"),
                        "progress": trace.get("progress") or [],
                        "inputArtifact": trace.get("inputArtifact"),
                        "outputArtifact": trace.get("outputArtifact"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return

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
            lines.append("按日趋势已整理在下方图表中。")
        return "\n".join(lines)

    def _ensure_multi_metric_summary_coverage(self, answer: str, question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
        if not answer or not run_result or run_result.evidence_gaps:
            return answer
        summaries = summary_metric_values(plan, run_result)
        if len(summaries) <= 1:
            return answer
        missing = [
            item for item in summaries
            if str(item.get("label") or "") and str(item.get("label") or "") not in answer
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
                value = format_metric_value_for_answer(primary.get("value"), primary.get("metricKey") or "", str(label))
                time_phrase = extract_question_time_phrase(question)
                prefix = "%s，" % time_phrase if time_phrase else "本次查询范围内，"
                summary = "%s%s为 %s。" % (prefix, label, value)
        lines.append("口径对账：我先按本次查询口径复核。")
        if summary:
            lines.append(summary)
        note = lightweight_metric_disclosure_note(question, plan, run_result)
        if note:
            lines.append(note)
        lines.append("如果这个数和后台看板不一致，通常优先核对：")
        lines.append("- 时间口径：后台按下单时间、支付时间还是退款成功时间统计。")
        lines.append("- 状态口径：是否只算支付成功、交易成功，是否排除关闭或异常订单。")
        lines.append("- 退款口径：GMV 是否扣退款，退款金额算申请退款还是退款成功。")
        lines.append("- 商品粒度：按 SPU、SKU、子订单还是商品维度汇总。")
        lines.append("- 数据更新：离线数仓和后台实时看板是否存在更新时间差。")
        lines.append("建议把后台看板的指标名称和时间范围发来，我可以按同一口径重算。")
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
                series_rows = metric_series_rows_for_intent(plan, intent, bundle.rows)
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
        summary_sentence = multi_summary_metric_sentence(question, plan, run_result)
        if summary_sentence:
            return summary_sentence + chart_hint_sentence(plan, run_result)
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


def summary_metric_values(plan: QueryPlan, run_result: AgentRunResult) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    values: List[Dict[str, Any]] = []
    seen: set[str] = set()
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
        metric_key = str(resolution.get("metricKey") or intent.metric_name or value_column)
        if metric_key in seen:
            continue
        seen.add(metric_key)
        values.append(
            {
                "metricKey": metric_key,
                "label": str(resolution.get("displayName") or friendly_column_label(plan, value_column)),
                "value": value,
                "taskId": item.task_id,
            }
        )
    return values


def multi_summary_metric_sentence(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    summaries = summary_metric_values(plan, run_result)
    if len(summaries) <= 1:
        return ""
    time_phrase = extract_question_time_phrase(question)
    prefix = "%s，" % time_phrase if time_phrase else "当前查询范围内，"
    parts = []
    for item in summaries[:5]:
        label = item.get("label") or "指标"
        value = format_metric_value_for_answer(item.get("value"), item.get("metricKey") or "", str(label))
        parts.append("%s为 %s" % (label, value))
    return prefix + "，".join(parts) + "。"


def chart_hint_sentence(plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    if not run_result:
        return ""
    has_trend = any(answer_result_role(intent_by_task_id(plan).get(item.task_id)) == "trend_context" for item in visible_successful_tasks(plan, run_result))
    return "\n\n按日趋势已整理在下方图表中。" if has_trend else ""


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
        return metric_series_rows_for_intent(plan, intent, item.query_bundle.rows) if intent else []
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
        "order_detail_cnt": "订单量",
        "order_gmv_amt_1d": "GMV",
        "pay_gmv_amt_1d": "支付GMV",
        "trade_success_gmv_amt_1d": "交易成功GMV",
        "refund_amt_1d": "退款金额",
        "return_success_amt_1d": "退货成功金额",
        "seller_repay_amt_1d": "赔付金额",
        "cs_ticket_cnt_1d": "咨询工单量",
        "pay_order_cnt_1d": "支付订单量",
        "order_cnt_1d": "订单量",
        "pay_user_cnt_1d": "支付用户量",
        "avg_pay_order_amt_1d": "客单价",
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
    metric_key = intent_metric_key(intent)
    family = semantic_metric_family(metric_key)
    if not family:
        return False
    question = plan_question_text(plan)
    if question_requests_multiple_metric_family(question, family):
        return False
    primary = primary_metric_for_family(plan, family)
    return bool(primary and metric_key and metric_key != primary)


def intent_metric_key(intent: QuestionIntent | None) -> str:
    if not intent:
        return ""
    resolution = intent.metric_resolution or {}
    return str(resolution.get("metricKey") or intent.metric_name or intent.metric_column or "").strip()


def semantic_metric_family(metric_key: str) -> str:
    text = str(metric_key or "").strip().lower()
    if "gmv" in text:
        return "gmv"
    return ""


def plan_question_text(plan: QueryPlan) -> str:
    questions = [str(intent.question or "").strip() for intent in plan.intents if str(intent.question or "").strip()]
    return " ".join(questions)


def question_requests_multiple_metric_family(question: str, family: str) -> bool:
    text = re.sub(r"\s+", "", str(question or "").lower())
    if family == "gmv":
        if re.search(r"(不同口径|各口径|分别|对比|比较|口径)", text):
            return True
    return False


def primary_metric_for_family(plan: QueryPlan, family: str) -> str:
    metrics: List[str] = []
    metrics.extend(ranking_metric_refs(plan.question_understanding or {}))
    metrics.extend(question_understanding_metric_refs(plan.question_understanding or {}))
    for intent in plan.intents:
        key = intent_metric_key(intent)
        if key:
            metrics.append(key)
    preferred_by_family = {
        "gmv": ["order_gmv_amt_1d", "pay_gmv_amt_1d", "trade_success_gmv_amt_1d"],
    }
    candidates = [metric for metric in metrics if semantic_metric_family(metric) == family]
    question = re.sub(r"\s+", "", plan_question_text(plan).lower())
    if family == "gmv":
        explicit = []
        if "支付gmv" in question:
            explicit.append("pay_gmv_amt_1d")
        if "交易成功gmv" in question or "成交gmv" in question:
            explicit.append("trade_success_gmv_amt_1d")
        for metric in explicit:
            if metric in candidates:
                return metric
    for preferred in preferred_by_family.get(family, []):
        if preferred in candidates:
            return preferred
    return candidates[0] if candidates else ""


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


def answer_skill_required(plan: QueryPlan, run_result: AgentRunResult | None = None, has_rule_context: bool = False) -> bool:
    return bool(select_answer_skill(plan, run_result, has_rule_context))


def deterministic_analysis_skill_fallback(plan: QueryPlan, run_result: AgentRunResult | None = None, has_rule_context: bool = False) -> str:
    selected = select_answer_skill(plan, run_result, has_rule_context)
    if selected:
        return selected
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip().lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    if analysis_intent in {"trend_check", "anomaly_check", "diagnosis", "comparison", "overview"} and requires_explanation:
        return "bi_trend_attribution"
    return ""


def select_answer_skill(plan: QueryPlan, run_result: AgentRunResult | None = None, has_rule_context: bool = False) -> str:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip().lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    categories = {intent.category for intent in plan.intents}
    if plan_requires_rule_evidence(plan) and has_rule_context:
        return "rule_compliance"
    declared_skill = declared_skill_workflow_name(understanding)
    if declared_skill:
        return declared_skill
    if not reusable_analysis_workflow_requested(understanding):
        return ""
    if plan_has_new_product_risk_shape(plan):
        return "new_product_risk"
    question_text = " ".join([str(understanding.get("originalQuestion") or understanding.get("question") or "")] + [str(intent.question or "") for intent in plan.intents]).lower()
    if analysis_intent in {"overview", "diagnosis", "store_health", "health_check"} and looks_like_merchant_daily_briefing(question_text, categories):
        return "merchant_daily_briefing"
    if analysis_intent in {"trend_check", "anomaly_check", "diagnosis", "comparison"} and QuestionCategory.REFUND in categories and looks_like_refund_diagnosis(question_text):
        return "refund_rate_diagnosis"
    if analysis_intent in {"trend_check", "anomaly_check", "diagnosis", "comparison"} and QuestionCategory.TRADE in categories and looks_like_gmv_or_order_drop(question_text):
        return "gmv_drop_diagnosis"
    risk_domains = {QuestionCategory.REFUND, QuestionCategory.COMPENSATION, QuestionCategory.CS_TICKET, QuestionCategory.GOODS, QuestionCategory.SCM}
    if analysis_intent in {"risk_ranking", "diagnosis", "anomaly_check"} and categories & risk_domains:
        return "risk_analysis"
    if plan_has_ratio_calculation(plan):
        return "ratio_analysis"
    if analysis_intent in {"trend_check", "anomaly_check", "diagnosis", "comparison", "overview"} and requires_explanation:
        return "bi_trend_attribution"
    if run_result and run_result.evidence_gaps and analysis_intent not in {"none", ""}:
        return "risk_analysis"
    return ""


def deterministic_analysis_summary(question: str, plan: QueryPlan, run_result: AgentRunResult) -> str:
    if not analysis_summary_required(plan):
        return ""
    rows = list(getattr(getattr(run_result, "merged_query_bundle", None), "rows", []) or [])
    if not rows:
        return ""
    metric_labels: List[str] = []
    for intent in plan.intents:
        label = friendly_column_label(plan, intent.metric_name or intent.metric_column or "")
        if label and label not in metric_labels:
            metric_labels.append(label)
    if not metric_labels:
        for key in rows[0].keys():
            label = friendly_column_label(plan, str(key))
            if label and label not in metric_labels and key not in {"seller_id", "merchant_id", "pt"}:
                metric_labels.append(label)
    subject = "、".join(metric_labels[:4]) or "当前指标"
    row_count = len(rows)
    return "%s已有 %d 行已验证数据，可基于这些结果做趋势或异常判断；证据不足的原因不要扩展为确定结论。" % (subject, row_count)


def declared_skill_workflow_name(understanding: Dict[str, Any]) -> str:
    allowed = {
        "bi_trend_attribution",
        "risk_analysis",
        "rule_compliance",
        "ratio_analysis",
        "new_product_risk",
        "gmv_drop_diagnosis",
        "refund_rate_diagnosis",
        "merchant_daily_briefing",
    }
    for key in ["skillWorkflow", "skill_workflow", "recommendedSkill", "recommended_skill", "analysisSkill", "analysis_skill"]:
        value = understanding.get(key)
        if isinstance(value, dict):
            name = str(value.get("skillName") or value.get("skill_name") or value.get("name") or "").strip()
        else:
            name = str(value or "").strip()
        if name in allowed:
            return name
    return ""


def looks_like_gmv_or_order_drop(text: str) -> bool:
    return bool(re.search(r"gmv|销售额|成交额|订单|下单|支付|下降|下滑|变少|异常|波动|为什么|原因|归因", text or ""))


def looks_like_refund_diagnosis(text: str) -> bool:
    return bool(re.search(r"退款率|退款金额|退款|退货|售后|上升|升高|变高|异常|为什么|原因|归因", text or ""))


def looks_like_merchant_daily_briefing(text: str, categories: set[QuestionCategory]) -> bool:
    if re.search(r"日报|简报|经营情况|店铺情况|经营健康|整体表现|今天怎么样|昨天怎么样", text or ""):
        return True
    return len({item for item in categories if item != QuestionCategory.UNKNOWN}) >= 3


def reusable_analysis_workflow_requested(understanding: Dict[str, Any]) -> bool:
    workflow = understanding.get("skillWorkflow") or understanding.get("skill_workflow") or {}
    if isinstance(workflow, dict) and boolish(workflow.get("enabled", workflow.get("required", False))):
        return True
    for key in ["reusableAnalysis", "reusable_analysis", "fixedAnalysisWorkflow", "fixed_analysis_workflow"]:
        if boolish(understanding.get(key)):
            return True
    for item in understanding.get("requiredEvidenceIntents") or understanding.get("required_evidence_intents") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("semanticLabel") or item.get("semantic_label") or "").strip().lower()
        if label in {"reusable_analysis_workflow", "skill_workflow", "fixed_analysis_workflow"}:
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
    merchant: MerchantInfo | None = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return verified_answer_context(
        question,
        plan,
        run_result,
        rule_context=rule_context,
        merchant=merchant,
        personalization_context=personalization_context,
    ).prompt_payload()


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
        if should_hide_alternate_metric(plan, intent):
            continue
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
            metric_key = str(item.get("metric") or item.get("metricKey") or "").strip()
            if metric_key and semantic_metric_family(metric_key):
                primary = primary_metric_for_family(plan, semantic_metric_family(metric_key))
                if primary and metric_key != primary and not question_requests_multiple_metric_family(plan_question_text(plan), semantic_metric_family(metric_key)):
                    continue
            disclosures.append(
                {
                    key: item.get(key)
                    for key in ["metric", "formula", "sourceColumns", "fieldWarning"]
                    if item.get(key) not in (None, "", [])
                }
            )
    return [item for item in dedupe_dicts(disclosures) if item]


def lightweight_metric_disclosures(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    items: List[Dict[str, Any]] = []
    for item in metric_disclosures(plan, run_result.verified_evidence):
        description = lightweight_metric_description(item)
        if not description:
            continue
        items.append(
            {
                "metricKey": item.get("metricKey") or item.get("metric"),
                "displayName": item.get("displayName") or item.get("metricKey") or item.get("metric"),
                "description": description,
            }
        )
    return [item for item in dedupe_dicts(items) if item][:4]


def lightweight_metric_disclosure_note(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> str:
    disclosures = lightweight_metric_disclosures(question, plan, run_result)
    if not disclosures:
        return ""
    descriptions = dedupe_strings([str(item.get("description") or "").strip() for item in disclosures])
    descriptions = [item for item in descriptions if item]
    if not descriptions:
        return ""
    time_phrase = extract_question_time_phrase(question) or "本次查询时间范围"
    return "统计说明：%s；时间为%s；范围为当前店铺。" % ("；".join(descriptions[:3]), time_phrase)


def lightweight_metric_description(item: Dict[str, Any]) -> str:
    family = lightweight_metric_family(item)
    if not family:
        return ""
    text = metric_disclosure_text(item)
    display_name = str(item.get("displayName") or item.get("metricKey") or item.get("metric") or "指标")
    if family == "gmv":
        if re.search(r"(扣|减|净|refund|退款|-)", text, flags=re.I):
            return "%s按扣除退款后的净成交金额统计" % display_name
        if "trade_success" in text or "交易成功" in text:
            return "%s按交易成功金额统计" % display_name
        return "%s按支付成功订单金额统计，未主动扣除后续退款" % display_name
    if family == "refund_amount":
        if "success" in text or "成功" in text:
            return "%s按退款成功金额统计" % display_name
        return "%s按已确认的退款金额统计" % display_name
    if family == "refund_rate":
        return "%s按已确认的退款分子除以对应订单基数计算" % display_name
    if family == "order_count":
        if "distinct" in text or "去重" in text:
            return "%s按订单或子订单去重统计" % display_name
        return "%s按当前查询范围内的订单记录统计" % display_name
    if family == "ticket_count":
        return "%s按客服工单记录数统计" % display_name
    return ""


def lightweight_metric_family(item: Dict[str, Any]) -> str:
    text = metric_disclosure_text(item)
    if not text:
        return ""
    if "gmv" in text:
        return "gmv"
    if re.search(r"(退款|售后|refund)", text, flags=re.I) and re.search(r"(率|rate|ratio|占比)", text, flags=re.I):
        return "refund_rate"
    if re.search(r"(退款|售后|refund)", text, flags=re.I) and re.search(r"(金额|amt|amount|pay)", text, flags=re.I):
        return "refund_amount"
    if re.search(r"(下单|订单量|订单数|order.*cnt|order_detail_cnt|order_cnt)", text, flags=re.I):
        return "order_count"
    if re.search(r"(工单|咨询|ticket|workorder)", text, flags=re.I) and re.search(r"(量|数|cnt|count)", text, flags=re.I):
        return "ticket_count"
    return ""


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
            resolution.get("displayName"),
            labels.get(metric_key),
            labels.get(str(intent.metric_name or "")),
        ]
        for item in candidates:
            term = str(item or "").strip()
            if len(term) >= 2 and term not in terms:
                terms.append(term)
    return terms


def remove_hidden_alternate_metric_lines(text: str, plan: QueryPlan) -> str:
    terms = hidden_alternate_metric_terms(plan)
    if not terms:
        return text
    kept: List[str] = []
    for line in str(text or "").splitlines():
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
        "gmv_drop_diagnosis": "GMV/订单下跌归因",
        "refund_rate_diagnosis": "退款率升高归因",
        "merchant_daily_briefing": "商家经营简报",
    }.get(skill_name, "分析结果")
    lines = ["%s：" % title]
    if skill_name == "ratio_analysis":
        lines.extend(ratio_skill_lines(disclosures, rows))
    elif skill_name == "rule_compliance":
        lines.extend(rule_compliance_skill_lines(rule_evidence, rows))
    elif skill_name == "new_product_risk":
        lines.extend(new_product_risk_skill_lines(rows, disclosures))
    elif skill_name == "merchant_daily_briefing":
        lines.extend(merchant_daily_briefing_skill_lines(rows, disclosures))
    elif skill_name == "gmv_drop_diagnosis":
        lines.extend(gmv_drop_diagnosis_skill_lines(rows, disclosures))
    elif skill_name == "refund_rate_diagnosis":
        lines.extend(refund_rate_diagnosis_skill_lines(rows, disclosures))
    else:
        lines.extend(risk_skill_lines(rows, disclosures, question))
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
    plan = QueryPlan(intents=intents or [])
    business_context = answer_business_context(question, plan, run_result, merchant, personalization_context)
    intent_signal_text = contextual_question_intent_signal_text(question, intents)
    context_signal_text = contextual_suggestion_signal_text(question, intents, business_context, merchant, personalization_context)
    categories = {normalize_question_category(intent.category) for intent in intents or []}
    ranked: Dict[str, tuple[float, int]] = {}
    order = 0
    question_norm = normalize_suggestion_text(question)

    def add(text: str, score: float) -> None:
        nonlocal order
        value = re.sub(r"\s+", "", str(text or "").strip())
        if not value:
            return
        normalized = normalize_suggestion_text(value)
        if not normalized or normalized == question_norm:
            return
        if re.search(r"(SQL|sql|字段名|表名|Doris|QueryGraph|继续追问|我可以)", value):
            return
        current = ranked.get(value)
        if current is None:
            ranked[value] = (score, order)
            order += 1
        elif score > current[0]:
            ranked[value] = (score, current[1])

    has_trade = QuestionCategory.TRADE in categories or suggestion_has(intent_signal_text, r"gmv|成交|销售额|订单|下单|客单价")
    has_refund = QuestionCategory.REFUND in categories or suggestion_has(intent_signal_text, r"退款|退货|售后|退款率")
    has_ticket = QuestionCategory.CS_TICKET in categories or suggestion_has(intent_signal_text, r"工单|客服|咨询|催单")
    has_goods = QuestionCategory.GOODS in categories or suggestion_has(intent_signal_text, r"商品|货品|上架|审核|新品|spu")
    has_compensation = QuestionCategory.COMPENSATION in categories or suggestion_has(intent_signal_text, r"赔付|理赔|补偿")
    has_coupon = QuestionCategory.COUPON in categories or suggestion_has(intent_signal_text, r"优惠券|券|补贴|优惠金额")
    has_scm = QuestionCategory.SCM in categories or suggestion_has(intent_signal_text, r"供应链|履约|发货|超时|入库|签收")
    has_rule = QuestionCategory.PLATFORM_RULE in categories or suggestion_has(intent_signal_text, r"规则|资质|规范|处罚|申诉")
    has_deposit = QuestionCategory.MERCHANT_OTHER in categories or suggestion_has(intent_signal_text, r"保证金|deposit|冻结|充值")

    if has_trade and has_refund and has_ticket:
        add("按商品拆解订单量、退款金额和工单量", 18)
        add("退款金额最高的商品是否也带来更多工单？", 17)
        add("工单最多的问题类型和订单状态是什么？", 16)
    if has_trade and has_refund:
        add("订单量变化和退款金额是否同步波动？", 15)
        add("退款金额最高的商品有哪些？", 14)
        add("下单多但退款也高的商品有哪些？", 13)
    if has_refund and has_ticket:
        add("退款相关工单主要集中在哪些问题？", 15)
        add("退款率高的商品是否也带来较多工单？", 14)
    if has_goods and has_refund:
        add("新品退款率是否偏高？", 15)
        add("退款高的商品发布时间是什么时候？", 14)
    if has_goods and has_ticket:
        add("商品审核问题是否带来客服咨询？", 13)

    if has_trade:
        if suggestion_has(intent_signal_text, r"gmv|生意流水|成交额|销售额"):
            add("GMV变化主要来自订单量还是客单价？", 17)
            add("GMV下降商品主要集中在哪些类目？", 16)
            add("最近7天GMV和退款金额一起看", 12)
        if suggestion_has(intent_signal_text, r"订单|下单|支付订单|成交订单"):
            add("最近7天下单量按日趋势如何？", 13)
            add("下单量最高的商品有哪些？", 12)
            add("支付订单量和退款率是否同步变化？", 11)
        add("最近7天店铺整体经营情况怎么样？", 7)
    if has_refund:
        add("退款原因占比最近是否变化？", 13)
        add("最近7天退款金额按日趋势如何？", 12)
        add("直接退款和退货退款分别占多少？", 11)
    if has_ticket:
        add("工单最多的问题类型有哪些？", 13)
        add("催单工单对应订单状态是什么？", 12)
        add("客服工单量按天趋势如何？", 11)
    if has_goods:
        add("商品审核拒绝原因主要是什么？", 13)
        add("最近上架失败商品明细有哪些？", 12)
        add("最近15天新发布商品表现如何？", 11)
    if has_compensation:
        add("赔付金额最高的订单有哪些？", 13)
        add("赔付订单是否关联退款或工单？", 12)
        add("赔付金额按天趋势如何？", 11)
    if has_coupon:
        add("优惠券投入带来了多少订单？", 13)
        add("优惠券订单退款率是否偏高？", 12)
        add("最近7天优惠金额和GMV表现如何？", 11)
    if has_scm:
        add("最近7天发货超时订单量是多少？", 13)
        add("履约异常集中在哪些商品？", 12)
        add("供应链履约量按天趋势如何？", 11)
    if has_rule:
        add("商品上架需要补哪些资质？", 12)
        add("商品审核被拒后优先排查什么？", 11)
        add("平台规则里哪些处罚会影响经营？", 8)
    if has_deposit:
        add("保证金余额和冻结金额是否异常？", 12)
        add("最近30天保证金充值流水", 11)
        add("保证金冻结原因是什么？", 10)

    if suggestion_has(intent_signal_text, r"趋势|变化|下降|上升|波动|异常"):
        add("波动最大的日期对应哪些订单或商品？", 12)
        if has_trade:
            add("近期流量和转化是否影响成交？", 10)
        if has_refund:
            add("波动当天退款原因和商品分布是什么？", 11)
    if suggestion_has(intent_signal_text, r"top|前\d+|最高|最多|排行"):
        add("这些对象的明细和关联指标一起看", 12)
    if suggestion_has(context_signal_text, r"七天无理由|7天无理由"):
        add("七天无理由退货占比最近是否升高？", 10)
    if suggestion_has(context_signal_text, r"自发货|发货模式|履约模式"):
        add("自发货订单是否存在发货超时风险？", 10)

    for signal in business_context.get("currentDataSignals") or []:
        label = str(signal.get("label") or "")
        if "退款" in label:
            add("退款金额高的商品和订单明细", 10)
        if "工单" in label or "咨询" in label:
            add("工单上升那几天主要是什么问题？", 10)
        if "订单" in label or "GMV" in label.upper():
            add("订单量变化主要来自哪些商品？", 10)

    if suggestion_has(context_signal_text, r"退款|售后|退货"):
        add("退款高发商品优先排查哪些原因？", 9)
    if suggestion_has(context_signal_text, r"工单|客服|咨询|催单"):
        add("咨询工单最多的问题怎么处理？", 9)
    if suggestion_has(context_signal_text, r"商品审核|上架|新品"):
        add("商品审核拒绝后下一步怎么改？", 9)
    if suggestion_has(context_signal_text, r"发货|履约|供应链|超时"):
        add("发货超时订单集中在哪些商品？", 9)
    if suggestion_has(context_signal_text, r"保证金|冻结|充值"):
        add("保证金冻结和充值流水一起看", 9)

    if len(ranked) < 3:
        add("最近7天店铺整体经营情况怎么样？", 1)
        add("最近7天订单量和退款金额有什么变化？", 1)
        add("商品审核被拒怎么办？", 1)

    sorted_items = sorted(ranked.items(), key=lambda item: (-item[1][0], item[1][1]))
    return [item[0] for item in sorted_items[:9]]


def build_merchant_experience_package(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    merchant: MerchantInfo | None = None,
    sections: Optional[List[ChatDataSection]] = None,
    suggestions: Optional[List[str]] = None,
    personalization_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    suggestion_items = dedupe_strings([str(item) for item in (suggestions or []) if str(item).strip()])[:8]
    anomaly_alerts = merchant_anomaly_alerts(question, plan, run_result)
    traceability = merchant_traceability(question, plan, run_result, merchant, sections or [])
    drill_actions = merchant_drill_down_actions(question, plan, run_result, suggestion_items)
    metric_notes = lightweight_metric_disclosures(question, plan, run_result)
    return {
        "version": "v1",
        "businessAdvice": merchant_business_advice(question, plan, run_result, anomaly_alerts, personalization_context)[:2],
        "suggestedQuestions": suggestion_items[:6],
        "anomalyAlerts": anomaly_alerts[:4],
        "metricDisclosures": metric_notes,
        "traceability": traceability,
        "drillDownActions": drill_actions[:5],
        "reportSubscriptionHint": merchant_report_subscription_hint(plan, run_result),
        "clarificationHints": merchant_clarification_hints(plan),
    }


def merchant_business_advice(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    anomaly_alerts: List[Dict[str, Any]],
    personalization_context: Optional[Dict[str, Any]],
) -> List[str]:
    categories = {normalize_question_category(intent.category) for intent in plan.intents or []}
    text = contextual_question_intent_signal_text(question, plan.intents)
    context_text = json.dumps(personalization_context or {}, ensure_ascii=False, default=str)
    advice: List[str] = []
    if anomaly_alerts:
        label = str(anomaly_alerts[0].get("metric") or "异常指标")
        advice.append("优先排查%s波动对应的商品、订单或渠道。" % label)
    if QuestionCategory.REFUND in categories or suggestion_has(text + context_text, r"退款|退货|售后"):
        advice.append("优先查看退款高的商品和退款原因，判断是否集中在质量、描述或履约问题。")
    if QuestionCategory.CS_TICKET in categories or suggestion_has(text + context_text, r"工单|客服|咨询|催单"):
        advice.append("同步查看客服工单类型，先处理催单和售后咨询集中的问题。")
    if QuestionCategory.GOODS in categories or suggestion_has(text + context_text, r"商品|新品|上架|审核"):
        advice.append("对新上架或审核异常商品做单独下钻，避免问题继续放大。")
    if QuestionCategory.SCM in categories or suggestion_has(text + context_text, r"发货|履约|超时"):
        advice.append("优先定位发货或签收超时订单，按商品和仓配链路拆解原因。")
    if QuestionCategory.TRADE in categories and not advice:
        advice.append("先拆成订单量、客单价和退款影响三部分看经营变化。")
    if not advice:
        advice.append("先从金额、单量和最近波动最大的对象下钻定位原因。")
    return dedupe_strings(advice)


def merchant_anomaly_alerts(question: str, plan: QueryPlan, run_result: AgentRunResult | None) -> List[Dict[str, Any]]:
    if not run_result:
        return []
    alerts: List[Dict[str, Any]] = []
    intent_map = intent_by_task_id(plan)
    for task in visible_successful_tasks(plan, run_result):
        intent = intent_map.get(task.task_id)
        if not intent or not task.query_bundle.rows:
            continue
        points = metric_series_rows_for_intent(plan, intent, task.query_bundle.rows)
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
                "drillDownQuestion": "%s波动最大的日期对应哪些商品或订单？" % metric_label,
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
    dates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ["pt", "date", "biz_date", "stat_date", "dt"]:
            if row.get(key):
                dates.append(str(row.get(key)))
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


def merchant_drill_down_actions(
    question: str,
    plan: QueryPlan,
    run_result: AgentRunResult | None,
    suggestions: List[str],
) -> List[Dict[str, Any]]:
    categories = {normalize_question_category(intent.category) for intent in plan.intents or []}
    text = contextual_question_intent_signal_text(question, plan.intents)
    actions: List[Dict[str, Any]] = []

    def add(label: str, follow_up: str, action_type: str = "follow_up_question") -> None:
        if any(item.get("label") == label for item in actions):
            return
        actions.append({"label": label, "question": follow_up, "actionType": action_type})

    if QuestionCategory.REFUND in categories or suggestion_has(text, r"退款|退货|售后"):
        add("查看异常商品", "退款金额或退款率最高的商品有哪些？")
        add("查看退款原因", "退款原因占比最近是否变化？")
    if QuestionCategory.CS_TICKET in categories or suggestion_has(text, r"工单|客服|咨询|催单"):
        add("查看相关工单", "工单最多的问题类型和订单状态是什么？")
    if QuestionCategory.TRADE in categories or suggestion_has(text, r"gmv|订单|成交|销售额"):
        add("拆解成交变化", "GMV变化主要来自订单量还是客单价？")
    if QuestionCategory.GOODS in categories or suggestion_has(text, r"商品|新品|上架|审核"):
        add("查看商品明细", "这些商品的上架、审核和售后表现一起看")
    if QuestionCategory.SCM in categories or suggestion_has(text, r"发货|履约|超时"):
        add("查看履约异常", "发货超时订单集中在哪些商品？")
    for item in suggestions[:2]:
        add("继续分析", item)
    return actions


def merchant_report_subscription_hint(plan: QueryPlan, run_result: AgentRunResult | None) -> Dict[str, Any]:
    categories = {normalize_question_category(intent.category) for intent in plan.intents or []}
    metrics = dedupe_strings([str(intent.metric_name or (intent.metric_resolution or {}).get("metricKey") or "") for intent in plan.intents if intent.metric_name or (intent.metric_resolution or {}).get("metricKey")])
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
        return ["请选择时间范围，例如最近7天、昨天或最近30天。", "也可以补充要看的业务域，例如交易、退款、客服或商品。"]
    primary = plan.intents[0]
    if not primary.days:
        hints.append("如果没有指定时间，系统会优先按近期常用时间窗或最近7天理解。")
    if primary.category == QuestionCategory.UNKNOWN:
        hints.append("业务域不明确时，会先让商家在交易、退款、客服、商品等范围里选择。")
    if not primary.metric_name and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT}:
        hints.append("指标不明确时，会先确认要看金额、单量、比例还是明细。")
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


def suggestion_has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


def contextual_question_intent_signal_text(question: str, intents: List[QuestionIntent]) -> str:
    parts: List[str] = [str(question or "")]
    for intent in intents or []:
        resolution = intent.metric_resolution or {}
        parts.extend(
            [
                category_display(normalize_question_category(intent.category)),
                str(intent.metric_name or ""),
                str(intent.metric_column or ""),
                str(intent.group_by_name or ""),
                str(intent.group_by_column or ""),
                str(intent.preferred_table or ""),
                str(resolution.get("metricKey") or ""),
                str(resolution.get("displayName") or ""),
                str(resolution.get("semanticLabel") or ""),
            ]
        )
    return "\n".join(part for part in parts if part)


def contextual_suggestion_signal_text(
    question: str,
    intents: List[QuestionIntent],
    business_context: Dict[str, Any],
    merchant: MerchantInfo | None,
    personalization_context: Optional[Dict[str, Any]],
) -> str:
    parts: List[str] = [contextual_question_intent_signal_text(question, intents)]
    context = personalization_context or {}
    parts.append(str((business_context or {}).get("merchantProfile") or ""))
    parts.append(str((business_context or {}).get("sessionSummary") or ""))
    parts.append(str((business_context or {}).get("memorySummary") or ""))
    parts.append(json.dumps((business_context or {}).get("recentFocus") or {}, ensure_ascii=False, default=str))
    parts.append(json.dumps((business_context or {}).get("relevantPreferences") or [], ensure_ascii=False, default=str))
    parts.append(json.dumps((business_context or {}).get("currentDataSignals") or [], ensure_ascii=False, default=str))
    parts.append(str(context.get("runtimeContext") or ""))
    if merchant is not None:
        parts.append(merchant.profile_markdown())
    return "\n".join(part for part in parts if part)


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
                "value": format_metric_value_for_answer(item.get("value"), str(item.get("metricKey") or ""), label),
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
        points = metric_series_rows_for_intent(plan, intent, task.query_bundle.rows) if intent else []
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
                "peakDate": peak.get("pt"),
                "peakValue": format_metric_value_for_answer(peak.get("value"), metric_key, label),
            }
        )
    return signals[:10]


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


def gmv_drop_diagnosis_skill_lines(rows: List[Dict[str, Any]], disclosures: List[Dict[str, Any]]) -> List[str]:
    lines = ["- 流程：先判断 GMV/订单变化，再定位商品、渠道或日期集中点，最后只基于证据给出原因假设。"]
    if rows:
        lines.append("- 关键证据：")
        for index, row in enumerate(rows[:6], 1):
            lines.append("  %d. %s" % (index, compact_row_preview(row)))
    else:
        lines.append("- 当前没有可用于归因的 GMV/订单结果行。")
    if disclosures:
        lines.append("- 指标口径：%s" % "; ".join(compact_disclosure(item) for item in disclosures[:5]))
    lines.append("- 判断：如果只有结果指标、没有结构拆解证据，只能说明下跌现象，不能断言具体原因。")
    return lines


def refund_rate_diagnosis_skill_lines(rows: List[Dict[str, Any]], disclosures: List[Dict[str, Any]]) -> List[str]:
    lines = ["- 流程：先确认退款率/退款金额变化，再看商品、订单、客服/赔付是否集中，避免把单量下降误判为退款问题。"]
    if rows:
        lines.append("- 关键证据：")
        for index, row in enumerate(rows[:6], 1):
            lines.append("  %d. %s" % (index, compact_row_preview(row)))
    else:
        lines.append("- 当前没有可用于退款归因的结果行。")
    if disclosures:
        lines.append("- 指标口径：%s" % "; ".join(compact_disclosure(item) for item in disclosures[:5]))
    lines.append("- 判断：退款率升高必须同时披露分子和分母，缺任一侧证据时只提示排查方向。")
    return lines


def merchant_daily_briefing_skill_lines(rows: List[Dict[str, Any]], disclosures: List[Dict[str, Any]]) -> List[str]:
    lines = ["- 流程：按交易、退款售后、客服/赔付、商品/履约顺序汇总，只突出需要商家行动的变化。"]
    if rows:
        lines.append("- 今日/近期经营信号：")
        for index, row in enumerate(rows[:8], 1):
            lines.append("  %d. %s" % (index, compact_row_preview(row)))
    else:
        lines.append("- 当前没有可用于生成经营简报的结果行。")
    if disclosures:
        lines.append("- 使用指标：%s" % "; ".join(compact_disclosure(item) for item in disclosures[:6]))
    lines.append("- 判断：简报只做优先级排序和行动建议，不把缺失 topic 解读为业务正常。")
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
                "whenToUse": when_to_use,
                "when_to_use": when_to_use,
                "constraints": constraints,
                "requiredInputs": required_inputs,
                "required_inputs": required_inputs,
                "path": str(skill_file.relative_to(root.parent.parent) if root.parent.parent in skill_file.parents else skill_file),
            }
        )
    return headers


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
        alerts = daily_report_alerts(metrics)
        return DailyReportResponse(
            merchant_id=target,
            merchant_name=merchant_name,
            date=date.today().isoformat(),
            metrics=metrics,
            anomaly_alerts=alerts,
            drill_down_actions=[
                {"label": "查看退款商品", "question": "昨日退款金额最高的商品有哪些？", "actionType": "follow_up_question"},
                {"label": "查看订单趋势", "question": "最近7天订单量和GMV按日趋势如何？", "actionType": "follow_up_question"},
                {"label": "查看客服工单", "question": "昨日客服工单最多的问题类型有哪些？", "actionType": "follow_up_question"},
            ],
            traceability={
                "sourceSummary": "基于商家画像宽表最新分区生成",
                "merchantId": target,
                "merchantName": merchant_name,
                "timeRange": "昨日",
                "sourceTables": ["ads_merchant_profile"],
            },
            suggestions=daily_report_suggestions(metrics, alerts),
        )


def daily_report_alerts(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    refund_amt = answer_numeric_value(metrics.get("昨日退款金额"))
    order_cnt = answer_numeric_value(metrics.get("昨日总订单量"))
    gmv = answer_numeric_value(metrics.get("昨日总gmv金额"))
    if refund_amt is not None and refund_amt > 0:
        alerts.append(
            {
                "type": "refund_attention",
                "severity": "warning",
                "metric": "昨日退款金额",
                "message": "昨日退款金额为 %s，建议下钻退款商品和退款原因。" % format_cell(metrics.get("昨日退款金额")),
                "drillDownQuestion": "昨日退款金额最高的商品有哪些？",
            }
        )
    if order_cnt is not None and order_cnt == 0 and gmv is not None and gmv == 0:
        alerts.append(
            {
                "type": "trade_flat",
                "severity": "warning",
                "metric": "订单量",
                "message": "昨日订单和GMV均为 0，建议确认是否为休店、流量异常或数据未产出。",
                "drillDownQuestion": "最近7天订单量和GMV按日趋势如何？",
            }
        )
    return alerts[:3]


def daily_report_suggestions(metrics: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[str]:
    suggestions: List[str] = []
    if alerts:
        suggestions.append("优先处理日报里的异常提醒，并下钻到商品、订单或原因。")
    if answer_numeric_value(metrics.get("昨日退款金额")) not in (None, 0):
        suggestions.append("重点查看退款金额最高商品，判断是否集中在质量、描述或履约问题。")
    suggestions.extend(
        [
            "关注订单、退款和客服工单是否同步波动。",
            "可把重点指标加入经营日报，持续跟踪异常变化。",
        ]
    )
    return dedupe_strings(suggestions)[:3]


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
