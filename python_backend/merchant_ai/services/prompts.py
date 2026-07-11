from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class PromptSectionSpec:
    section_id: str
    version: str
    title: str
    content: str


@dataclass(frozen=True)
class PromptTemplateSpec:
    prompt_id: str
    version: str
    agent: str
    description: str
    template: str
    section_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptRender:
    prompt_id: str
    version: str
    agent: str
    system_prompt: str
    section_ids: List[str] = field(default_factory=list)
    static_section_ids: List[str] = field(default_factory=list)
    template_fingerprint: str = ""
    render_fingerprint: str = ""

    def trace(self) -> Dict[str, Any]:
        return {
            "promptId": self.prompt_id,
            "version": self.version,
            "agent": self.agent,
            "sections": self.section_ids,
            "staticSections": self.static_section_ids,
            "templateFingerprint": self.template_fingerprint,
            "renderFingerprint": self.render_fingerprint,
        }

    def marker(self) -> str:
        return "prompt=%s@%s" % (self.prompt_id, self.version)


class PromptRegistry:
    """Central prompt registry inspired by DeerFlow's prompt assembly boundary."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplateSpec] = {}
        self._sections: Dict[str, PromptSectionSpec] = {}

    def register(self, spec: PromptTemplateSpec) -> None:
        self._templates[spec.prompt_id] = spec

    def register_section(self, spec: PromptSectionSpec) -> None:
        self._sections[spec.section_id] = spec

    def get(self, prompt_id: str) -> PromptTemplateSpec:
        if prompt_id not in self._templates:
            raise KeyError("Unknown prompt template: %s" % prompt_id)
        return self._templates[prompt_id]

    def get_section(self, section_id: str) -> PromptSectionSpec:
        if section_id not in self._sections:
            raise KeyError("Unknown prompt section: %s" % section_id)
        return self._sections[section_id]

    def list_specs(self) -> List[PromptTemplateSpec]:
        return list(self._templates.values())

    def list_sections(self) -> List[PromptSectionSpec]:
        return list(self._sections.values())


class PromptAssembler:
    """Render scoped system prompts from registered templates and runtime sections."""

    def __init__(self, registry: PromptRegistry | None = None) -> None:
        self.registry = registry or default_prompt_registry()

    def render(
        self,
        prompt_id: str,
        variables: Mapping[str, Any] | None = None,
        sections: Mapping[str, Any] | None = None,
    ) -> PromptRender:
        spec = self.registry.get(prompt_id)
        rendered = spec.template.format_map(SafeFormatDict({key: self._stringify(value) for key, value in (variables or {}).items()}))
        static_section_ids: List[str] = []
        static_section_texts: List[str] = []
        for section_id in spec.section_ids:
            section = self.registry.get_section(section_id)
            static_section_ids.append(section.section_id)
            static_section_texts.append(
                '<static-section name="%s" version="%s" title="%s">\n%s\n</static-section>'
                % (section.section_id, section.version, section.title, section.content.strip())
            )
        section_ids: List[str] = []
        section_texts: List[str] = []
        for key, value in (sections or {}).items():
            text = self._stringify(value).strip()
            if not text:
                continue
            section_ids.append(str(key))
            section_texts.append('<runtime-section name="%s">\n%s\n</runtime-section>' % (key, text))
        template_body = "\n\n".join([*static_section_texts, rendered.strip()]).strip()
        template_fingerprint = hashlib.sha256(template_body.encode("utf-8")).hexdigest()[:16]
        body = "\n\n".join([template_body, *section_texts]).strip()
        body = '<prompt id="%s" version="%s" agent="%s" templateFingerprint="%s">\n%s\n</prompt>' % (
            spec.prompt_id,
            spec.version,
            spec.agent,
            template_fingerprint,
            body,
        )
        render_fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        return PromptRender(
            prompt_id=spec.prompt_id,
            version=spec.version,
            agent=spec.agent,
            system_prompt=body,
            section_ids=section_ids,
            static_section_ids=static_section_ids,
            template_fingerprint=template_fingerprint,
            render_fingerprint=render_fingerprint,
        )

    def catalog_summary(self) -> List[Dict[str, str]]:
        return [
            {
                "promptId": spec.prompt_id,
                "version": spec.version,
                "agent": spec.agent,
                "description": spec.description,
                "staticSections": ",".join(spec.section_ids),
            }
            for spec in self.registry.list_specs()
        ]

    def lead_prompt_summary(self, action_ids: Iterable[str], loaded_skills: Iterable[str], max_concurrent_sub_agents: int) -> Dict[str, Any]:
        render = self.render(
            "lead.system",
            variables={
                "agent_name": "MerchantBILeadAgent",
                "max_concurrent_sub_agents": max_concurrent_sub_agents,
            },
            sections={
                "available_actions": "\n".join("- %s" % action for action in action_ids),
                "loaded_skills": "\n".join("- %s" % skill for skill in loaded_skills),
            },
        )
        trace = render.trace()
        trace["preview"] = render.system_prompt[:800]
        return trace

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return "\n".join(str(item) for item in value)
        return str(value)


def default_prompt_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register_section(
        PromptSectionSpec(
            section_id="common.stable_boundary",
            version="v1",
            title="稳定规则和动态上下文边界",
            content=(
                "系统提示词只承载稳定角色、职责边界、工具边界和输出要求。"
                "当前问题、商家、召回、记忆、节点合约、工具结果、证据缺口等运行时事实必须来自用户提示词或 runtime-section，"
                "不得把过期的动态信息当成静态规则。"
            ),
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="common.artifact_references",
            version="v1",
            title="大对象引用策略",
            content=(
                "完整 QueryGraph、SQL、查询结果、工具输出、证据报告和 trace 优先通过 artifact/source ref 追溯。"
                "提示词里只使用摘要、状态、行数、证据缺口和引用，不要求系统预加载全量大对象。"
            ),
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="lead.action_registry",
            version="v1",
            title="主智能体动作边界",
            content="主智能体只能从动作注册表选择下一步；不能临时发明流程、跳过证据校验或让子任务共享完整全局状态。",
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="planner.semantic_boundary",
            version="v1",
            title="规划语义边界",
            content=(
                "规划阶段只做问题结构理解、语义资产选择、查询图和证据需求声明。"
                "不直接写 SQL，不根据记忆或字段名猜指标口径；指标、字段、关系以语义层为准。"
            ),
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="node.contract_boundary",
            version="v1",
            title="节点合约边界",
            content="节点执行只能使用当前 nodePlanContract 中声明的表、字段、指标、商家过滤、时间范围和上游实体集合。",
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="answer.verified_evidence",
            version="v1",
            title="回答证据边界",
            content="回答只能基于已验证证据、结果摘要和证据缺口；缺证据要说明，不得把缺失、失败或未知解释成业务为 0。",
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="lead.system",
            version="v1",
            agent="LeadAgent",
            description="Main harness prompt assembled with actions, skills and budgets.",
            section_ids=["common.stable_boundary", "lead.action_registry", "common.artifact_references"],
            template=(
                "你是 {agent_name}，负责商家 BI Agent Harness 的全局调度。\n"
                "你保存全局目标、用户约束、任务进度和最终汇总；子 Agent 只接收与自身任务相关的局部上下文。\n"
                "根据 registry 中的 action 选择下一步，不编造不存在的 action。最多并发子 Agent 数：{max_concurrent_sub_agents}。\n"
                "遇到失败时区分 LLM 失败、规划失败、SQL 失败、0 行、证据缺失，不把失败说成业务为 0。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="planner.question_understanding",
            version="v1",
            agent="PlannerAgent",
            description="Understand a BI question into semantic-layer bounded questionUnderstanding.",
            section_ids=["common.stable_boundary", "planner.semantic_boundary", "common.artifact_references"],
            template=(
                "你是商家 BI 问题理解器。只输出 JSON。\n"
                "你的任务不是生成 SQL，也不是自由选表，而是从用户问题中识别 analysisGrain、rankingObjective、requestedMeasures、scopeConstraints、filters、timeWindowDays。\n"
                "同时必须声明 analysisIntent、requiresExplanation、requiredEvidenceIntents：简单查询/排行用 none/false/[]；需要诊断、原因解释、异常判断、风险判断、经营总结时，由你声明所需证据意图。\n"
                "只要 analysisIntent 不是 none，requiredEvidenceIntents 必须至少 1 条；comparison/trend_check/anomaly_check/risk_ranking/overview/diagnosis 都不能返回空 evidence intents。\n"
                "如果问题明显需要固定、可复用的商家经营 SOP，必须显式声明 skillWorkflow/reusableAnalysis/fixedAnalysisWorkflow 或 recommendedSkill；可选 Skill 仅限 gmv_drop_diagnosis、refund_rate_diagnosis、merchant_daily_briefing、bi_trend_attribution、risk_analysis、ratio_analysis、rule_compliance、new_product_risk。普通查数、排行、明细不要声明 Skill。\n"
                "不要依赖代码关键词补规则；如果需要解释型证据，把证据需求写进 requiredEvidenceIntents，再由语义层编译和 Critic 校验。\n"
                "metricRef 必须来自 semanticCatalog.candidateMetrics.key；ownerTable 必须使用对应 metric 的 table。\n"
                "memoryConstraints 只能作为本轮解释偏好、历史纠错或口径争议信号；不得用 memory 改写 semanticCatalog、指标公式、表关系或字段定义。\n"
                "如果 memoryConstraints 与 semanticCatalog 冲突，必须以 semanticCatalog 为准，并通过 validationGaps/clarification 表达未应用原因。\n"
                "sourcePhrase 必须只填写用户原话中的指标/业务对象原词，不要包含排序词、Top/前N、最高/最低、时间窗或分析动作。例如“GMV最高的前5天”的 sourcePhrase 只写“GMV”。\n"
                "rankingObjective.objectiveType 用来表达主指标用途：求一个商家总量/指标值用 metric_total；Top/最高/最多/前N 用 ranking；走势用 trend_anchor；明细实体过滤用 detail_anchor。\n"
                "如果用户问 Top/最高/最多/前N，rankingObjective 必须是被排序的主指标；其他指标放 requestedMeasures。\n"
                "如果用户只问“某指标是多少/怎么样/当前值”，不要伪造成 Top 排名；选择对应 metricRef，objectiveType=metric_total，groupByColumn 使用 seller_id/merchant_id 这类商家粒度字段。\n"
                "如果用户问具体订单/子订单/商品/退款/工单明细，rankingObjective 可以为空，但 filters 必须写出实体字段和值。\n"
                "如果用户问题包含状态、阶段、处理进度、成功/失败/异常/处理中等限定，filters 必须写出对应 semanticCatalog/live schema 里的状态字段和值；多个状态值用逗号分隔。\n"
                "如果用户表达“在某业务集合中/某业务对象带来的/使用某业务对象的/基于某集合”的限定，必须写入 scopeConstraints；scopeConstraints 表示后续 rankingObjective 和 requestedMeasures 都必须先受这个实体集合约束，不能只把它当作普通 requestedMeasure。scopeConstraints.ownerTable 必须是产生限定集合的来源业务对象表；如果目标集合是订单但来源是活动/券/商品/退款等，不要把 ownerTable 简单重复成订单表，除非你同时给出真实 filter。\n"
                "如果用户要从一个业务集合关联查看另一个业务域的证据，例如订单集合回填退款/商品/工单/赔付，且 semanticCatalog 已提供相关表、指标或 relationships，不要空泛返回 NEED_MORE_KNOWLEDGE；应输出 UNDERSTOOD。没有显式排序时，rankingObjective 可以选择定义 anchor 集合的最小可执行指标或留空，requestedMeasures 放需要补证据的指标，filters 只写用户明确给出的实体值。\n"
                "关系链的 join key 不需要你猜，后续编译器会从 semanticCatalog.relationships 选择；你只需要准确声明分析粒度、主集合和需要补充的业务域/指标。\n"
                "选择 rankingObjective 时要贴合用户排序短语，问题只说 GMV 时优先选直接 GMV 指标，不要选优惠率、占比或扣退款后派生指标。\n"
                "如果输入含 diagnosticContext 且 semanticCatalog 有候选指标，优先围绕 diagnosticContext.intent/goal 选择可执行的 overview/risk_ranking/comparison 理解，不要空泛返回 NEED_MORE_KNOWLEDGE。\n"
                "如果用户问走势、相关、匹配、同步上升或异常波动，analysisGrain 通常为 day；选择一个主时间序列指标做 rankingObjective，其余序列指标放 requestedMeasures。\n"
                "{force_catalog_instruction}"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="planner.repair_understanding",
            version="v1",
            agent="PlannerAgent",
            description="Re-understand a question after critic or validation feedback.",
            section_ids=["common.stable_boundary", "planner.semantic_boundary", "common.artifact_references"],
            template=(
                "你是商家 BI 问题重新理解 agent。只输出 JSON。\n"
                "不要直接输出 QueryGraph 或 SQL，只输出 questionUnderstanding。\n"
                "如果 critic 指出 scope 未落地、分析证据契约缺失或未覆盖，必须重新声明 scopeConstraints、analysisIntent、requiresExplanation、requiredEvidenceIntents，并把所需指标放入 requestedMeasures 或 knowledgeRequests。\n"
                "如果 critic 指出 MEMORY_CONSTRAINT_UNAPPLIED，必须只在 semanticCatalog 可支持时选择对应 metricRef；不支持时输出 clarification/knowledge gap，不得修改语义层定义。\n"
                "修复必须限制在 semanticCatalog 内，metricRef 必须来自 candidateMetrics.key。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="node.sql_draft",
            version="v2",
            agent="NodeAgent",
            description="Draft safe one-table SQL for a single QueryGraph node.",
            section_ids=["common.stable_boundary", "node.contract_boundary", "common.artifact_references"],
            template=(
                "你是 SQL NodeWorker。只输出 JSON: {{\"sql\":\"...\"}}。\n"
                "只能基于 nodePlanContract 写 SQL；只能查询 preferredTable；只能使用 allowedColumns；不要 join 其他表，不要修改 QueryGraph。\n"
                "SELECT 必须原样包含 nodePlanContract.outputKeys 的每个字段，以及 nodePlanContract.groupByColumn；这些字段即使只是用于 dependent 传递，也必须出现在 SELECT 结果中，不能只放在 WHERE 或 GROUP BY。\n"
                "如果 nodePlanContract.metricSpecs 不为空，SELECT 必须输出每个 metricSpec.metricName；这些指标已经由 Planner/Compiler 确定，不能少查、不能改名、不能自行替换口径。\n"
                "GROUP_AGG/TOPN 查询必须让所有非聚合 SELECT 字段同时出现在 GROUP BY 中，尤其不能丢 seller_id、pt、spu_id、spu_name、sub_order_id、order_id、ticket_id、bill_id、coupon_id。\n"
                "当 GROUP_AGG/TOPN 的 groupByColumn 是实体键（如 spu_id、spu_name、sub_order_id、order_id、ticket_id、bill_id、refund_id、coupon_id）时，必须在 WHERE 里过滤 NULL 和空字符串，避免产生空实体桶。\n"
                "dependent node 用 upstreamEntitySets 做 IN 过滤。必须按 merchant_id/seller_id 过滤商家。\n"
                "pt 是 Doris DATE 分区列，时间窗必须写成 `pt` >= DATE_SUB(CURDATE(), INTERVAL N DAY)，不要使用 DATE_FORMAT('%Y%m%d')。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="node.sql_repair",
            version="v1",
            agent="NodeAgent",
            description="Repair SQL without changing QueryGraph semantics.",
            section_ids=["common.stable_boundary", "node.contract_boundary", "common.artifact_references"],
            template=(
                "你是 SQL repair agent。只输出 JSON: {{\"sql\":\"...\"}}。\n"
                "只能基于 nodePlanContract 修 SQL，不能修改 QueryGraph 语义，不能新增 preferredTable 或 allowedColumns 外内容。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="answer.bi",
            version="v2",
            agent="AnswerAgent",
            description="Compose BI answer only from verified evidence.",
            section_ids=["common.stable_boundary", "answer.verified_evidence", "common.artifact_references"],
            template=(
                "你是商家经营分析助手。只基于输入的已验证数据回答，缺失证据要明确说明，不要把缺失解释成 0。\n"
                "回答要自然、简洁、先说结论，避免研发调试口吻。\n"
                "字段名带 raw 表示原始字段值；不要把 refund_related_pay_amt_raw 或 pay_amt 说成已确认的独立退款金额。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="answer.analysis",
            version="v2",
            agent="AnalysisAgent",
            description="Produce business interpretation from evidence.",
            section_ids=["common.stable_boundary", "answer.verified_evidence", "common.artifact_references"],
            template=(
                "你是经营分析助手。基于当前数据给出商家能读懂的经营判断，不在 SQL 阶段硬编码业务假设。\n"
                "不要输出“分析结论/关键证据/限制/口径”这类固定报告标题，不要暴露字段名、表名或 SQL。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="answer.rule",
            version="v1",
            agent="AnswerAgent",
            description="Answer platform rule questions from retrieved knowledge only.",
            section_ids=["common.stable_boundary", "answer.verified_evidence", "common.artifact_references"],
            template="你是平台规则助手。只基于给定知识回答；没有依据时说需要运营补充规则。",
        )
    )
    return registry
