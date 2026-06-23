from __future__ import annotations

import json
from datetime import date
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

    def compose(
        self,
        question: str,
        merchant: MerchantInfo,
        plan: QueryPlan,
        run_result: AgentRunResult,
        knowledge_context: str,
        analysis_summary: str = "",
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
        if analysis_summary:
            return analysis_summary
        bundle = run_result.merged_query_bundle if run_result else QueryBundle()
        if primary.intent_type == "VALID" and primary.answer_mode not in {AnswerMode.RULE, AnswerMode.CHAT} and (not run_result or not run_result.task_results):
            return self._no_execution_answer(plan)
        if run_result and run_result.task_results and all(result.query_bundle.failed for result in run_result.task_results):
            return self._apply_answer_guard(
                self.append_business_advice(self._execution_failure_answer(run_result), plan.intents, bundle),
                run_result,
            )
        if self.llm.configured and bundle.rows:
            prompt = json.dumps(
                {
                    "question": question,
                    "merchant": merchant.model_dump(by_alias=True),
                    "plan": plan.model_dump(by_alias=True),
                    "tables": bundle.tables,
                    "rows": bundle.rows[:80],
                    "evidenceCheck": run_result.evidence_check.model_dump(by_alias=True) if run_result else {},
                    "verifiedEvidence": run_result.verified_evidence.model_dump(by_alias=True) if run_result else {},
                    "evidenceGaps": [gap.model_dump(by_alias=True) for gap in run_result.evidence_gaps] if run_result else [],
                },
                ensure_ascii=False,
                default=str,
            )
            answer_prompt = self.prompt_assembler.render(
                "answer.bi",
                sections={
                    "answer_context_policy": "AnswerAgent 只读取 verified evidence、rows preview、evidence gaps 和 plan；不要补造未执行事实。",
                },
            )
            answer = self.llm.chat(
                answer_prompt.system_prompt,
                prompt,
                "",
                timeout_seconds=self.llm.settings.llm_answer_timeout_seconds,
            )
            if answer:
                return self._apply_answer_guard(self.append_business_advice(answer, plan.intents, bundle), run_result)
        return self._apply_answer_guard(
            self.append_business_advice(self._fallback_data_answer(question, plan, bundle, run_result), plan.intents, bundle),
            run_result,
        )

    def summarize_analysis(self, question: str, plan: QueryPlan, run_result: AgentRunResult) -> str:
        if not self.llm.configured or not run_result or not run_result.merged_query_bundle.rows:
            return ""
        if not analysis_summary_required(plan):
            return ""
        analysis_prompt = self.prompt_assembler.render(
            "answer.analysis",
            sections={"analysis_policy": "只能基于 evidence 判断趋势、异常和原因假设；不能把缺失证据当事实。"},
        )
        return self.llm.chat(
            analysis_prompt.system_prompt,
            json.dumps(
                {
                    "question": question,
                    "plan": plan.model_dump(by_alias=True),
                    "rows": run_result.merged_query_bundle.rows[:80],
                    "evidenceCheck": run_result.evidence_check.model_dump(by_alias=True),
                },
                ensure_ascii=False,
                default=str,
            ),
            "",
            timeout_seconds=self.llm.settings.llm_answer_timeout_seconds,
        )

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
        columns = fallback_display_columns(plan, bundle.rows)
        if columns:
            lines.append("")
            lines.append(markdown_table(bundle.rows[:8], columns))
        derived = run_result.verified_evidence.derived_evidence if run_result and run_result.verified_evidence else []
        if derived:
            formulas = ["%s=%s" % (item.get("metric"), item.get("formula")) for item in derived[:6] if item.get("metric") and item.get("formula")]
            if formulas:
                lines.append("")
                lines.append("口径：%s" % "；".join(formulas))
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


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| %s |" % " | ".join(columns)
    divider = "| %s |" % " | ".join("---" for _ in columns)
    body = []
    for row in rows:
        body.append("| %s |" % " | ".join(format_cell(row.get(column, "")) for column in columns))
    return "\n".join([header, divider] + body)


def format_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ")[:80]


def analysis_summary_required(plan: QueryPlan) -> bool:
    understanding = plan.question_understanding or {}
    analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "none").strip().lower()
    requires_explanation = boolish(understanding.get("requiresExplanation", understanding.get("requires_explanation")))
    return requires_explanation or (analysis_intent and analysis_intent != "none")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


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
