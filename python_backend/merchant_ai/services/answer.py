from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

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
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.repositories import AnswerRepository, DorisRepository, PendingAnswerStore


class AnswerComposeService:
    def __init__(self, llm: LlmClient):
        self.llm = llm
        self.prompt_assembler = PromptAssembler()
        self.last_prompt_chars = 0
        self.last_analysis_skill_trace: Dict[str, Any] = {}

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
        if allow_llm and self.llm.configured and bundle.rows and analysis_summary_required(plan):
            prompt = json.dumps(answer_data_package(question, plan, run_result, rule_context), ensure_ascii=False, default=str)
            self.last_prompt_chars = len(prompt)
            answer_prompt = self.prompt_assembler.render(
                "answer.bi",
                sections={
                    "answer_context_policy": "AnswerAgent 只读取 question、dataRows、metricDisclosures、evidenceGaps；不要读取或推断 QueryGraph。",
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
        if not analysis_summary_required(plan):
            return ""
        skill_answer = self.run_analysis_skill(question, plan, run_result, outputs_path, rule_context)
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

    def run_analysis_skill(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
        outputs_path: str = "",
        rule_context: str = "",
    ) -> str:
        skill_dir = self.llm.settings.resources_root / "runtime" / "agent_skills" / "bi_trend_attribution"
        skill_file = skill_dir / "SKILL.md"
        script = skill_dir / "scripts" / "profile_timeseries.py"
        trace: Dict[str, Any] = {
            "skillName": "bi_trend_attribution",
            "matchedBy": "questionUnderstanding.analysisIntent",
            "activated": False,
            "skillPath": str(skill_file),
            "scriptPath": str(script),
        }
        self.last_analysis_skill_trace = trace
        if not skill_file.exists() or not script.exists():
            trace["error"] = "skill package missing"
            return ""
        skill_meta = load_skill_frontmatter(skill_file)
        trace["metadata"] = skill_meta
        artifact_root = Path(outputs_path) if outputs_path else self.llm.settings.resolved_workspace_path / "analysis_skills"
        target = artifact_root / "artifacts" / "analysis_skills" / "bi_trend_attribution"
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

    def append_business_advice(self, answer: str, intents: List[QuestionIntent], bundle: QueryBundle) -> str:
        if not answer:
            answer = "当前没有足够数据形成结论。"
        if "建议" in answer:
            return answer
        categories = {intent.category for intent in intents}
        if QuestionCategory.REFUND in categories or QuestionCategory.COMPENSATION in categories:
            return answer + "\n\n建议：优先核对高金额订单的售后原因、赔付责任和客服触达记录，避免重复赔付或异常退款扩大。"
        if QuestionCategory.CS_TICKET in categories:
            return answer + "\n\n建议：关注工单高发问题和二次开启场景，把催单、退款、物流异常拆开跟进。"
        if QuestionCategory.GOODS in categories:
            return answer + "\n\n建议：结合商品审核状态、最近发布商品和成交表现，优先处理影响转化的商品资料问题。"
        return answer + "\n\n建议：可以继续追问明细、Top 排名或按日期趋势，我会基于同一业务口径继续展开。"

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
        for index, bundle in enumerate(run_result.query_bundles):
            if bundle.failed or not bundle.rows:
                continue
            title = "查询结果"
            if index < len(plan.intents):
                intent = plan.intents[index]
                title = intent.metric_name or intent.preferred_table or title
            sections.append(
                ChatDataSection(
                    title=title,
                    doris_tables=bundle.tables,
                    data_rows=bundle.rows[:50],
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
            return "这次查询没有成功执行：%s。已保留执行轨迹，建议检查数据表、字段或 SQL 口径。" % bundle.error
        if not bundle.rows:
            return "本轮查询执行成功但返回 0 行；这只能说明当前 SQL 口径下没有返回记录，不能解释为业务指标为 0。若这是近期数据问题，可能需要检查离线分区或实时 fallback。"
        lines = ["已按当前口径查询到 %s 行数据。" % bundle.effective_row_count()]
        for table in bundle.tables:
            lines.append("- 使用表：%s" % table)
        gaps = run_result.evidence_gaps if run_result else []
        if gaps:
            lines.append("- 证据状态：%s" % "；".join("%s:%s" % (gap.code, gap.reason[:80]) for gap in gaps[:4]))
        multi_node_success = bool(run_result and len([item for item in run_result.task_results if not item.query_bundle.failed]) > 1)
        columns = fallback_display_columns(plan, bundle.rows)
        if columns and not multi_node_success:
            lines.append("")
            lines.append(markdown_table(bundle.rows[:8], columns))
        derived = run_result.verified_evidence.derived_evidence if run_result and run_result.verified_evidence else []
        if derived:
            formulas = ["%s=%s" % (item.get("metric"), item.get("formula")) for item in derived[:6] if item.get("metric") and item.get("formula")]
            if formulas:
                lines.append("")
                lines.append("口径：%s" % "；".join(formulas))
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
    for key in ["spu_id", "spu_name", "order_id", "sub_order_id", "refund_id", "ticket_id", "bill_id"]:
        base_value = normalized_cell(base.get(key))
        if not base_value:
            continue
        for row in rows:
            if normalized_cell(row.get(key)) == base_value:
                return row
    return None


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
        "dataRows": run_result.merged_query_bundle.rows[:40],
        "metricDisclosures": metric_disclosures(plan, verified),
        "evidenceGaps": compact_evidence_gaps(run_result.evidence_gaps),
        "ruleEvidence": compact_rule_evidence(question, rule_context),
    }


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
    def __init__(self, answer_repository: AnswerRepository, pending_store: PendingAnswerStore):
        self.answer_repository = answer_repository
        self.pending_store = pending_store

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
