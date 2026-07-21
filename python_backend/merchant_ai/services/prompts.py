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
                "当前问题、授权主体、召回、记忆、节点合约、工具结果、证据缺口等运行时事实必须来自用户提示词或 runtime-section，"
                "不得把过期的动态信息当成静态规则。"
                "当当前用户本轮输入或当前会话短期记忆中的明确纠正、限定条件、时间窗、对象集合与历史线程摘要或长期记忆冲突时，"
                "优先采用当前用户本轮输入和当前会话短期记忆；长期记忆只能作为历史偏好、纠错或争议信号。"
                "指标公式、字段定义、权限边界和已验证工具证据仍以 semanticCatalog、节点合约和工具结果为准。"
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
            version="v2",
            title="规划语义边界",
            content=(
                "Planner 是 Core Agent 调用的受治理 QueryGraph 编译工具，不是 subagent，也不是第二个 ReAct Agent。"
                "Core Agent 拥有动作与文件工具的选择权；Planner 只做问题结构理解、已读语义证据校验、查询图和证据需求声明。"
                "当 runtime-section 声明 core_managed_filesystem 时，只能消费 Core 已读取的 coreSemanticEvidence，"
                "不得自行调用 ls、grep、read 或扩展到未读 ref；证据不足必须返回 NEED_MORE_KNOWLEDGE。"
                "仅显式 legacy runtime-section 可以授权旧 semantic tool loop。"
                "Planner 不直接写 SQL，不根据记忆、L0 摘要、表名或字段名猜指标口径；指标、字段、关系以精确语义定义为准。"
            ),
        )
    )
    registry.register_section(
        PromptSectionSpec(
            section_id="node.contract_boundary",
            version="v1",
            title="节点合约边界",
            content="节点执行只能使用当前 nodePlanContract 中声明的表、字段、指标、授权主体过滤、时间范围和上游实体集合。",
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
                "你是 {agent_name}，负责受治理 BI Agent Harness 的全局调度。\n"
                "你保存全局目标、用户约束、任务进度和最终汇总；子 Agent 只接收与自身任务相关的局部上下文。\n"
                "根据 registry 中的 action 选择下一步，不编造不存在的 action。最多并发子 Agent 数：{max_concurrent_sub_agents}。\n"
                "遇到失败时区分 LLM 失败、规划失败、SQL 失败、0 行、证据缺失，不把失败说成业务为 0。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="planner.semantic_asset_selection",
            version="v2",
            agent="PlannerAgent",
            description="Select semantic metric refs before QueryGraph compilation.",
            section_ids=["common.stable_boundary", "planner.semantic_boundary", "common.artifact_references"],
            template=(
                "你是商家 BI 语义资产选择器。只输出 JSON，不要输出 SQL、QueryGraph 或解释长文。\n"
                "输入包含用户问题、metricPhrases、候选短卡 retrievedCandidates、候选分组 candidateGroups，可能包含 semanticReadResults。\n"
                "你的任务是在候选 ref 中选择能回答用户指标词的语义资产；短卡不够判断时，可以请求 semantic_read 读取少量 ref 的完整定义。\n"
                "第一轮可输出 action=semantic_read，readRefs 最多 3 个，ref 必须来自候选或候选的 readableRefs；第二轮已有 semanticReadResults 时不能再次 read。\n"
                "如果读完仍无法安全判断业务口径，输出 action=ask_human 并给出需要用户确认的问题；不要自己猜不存在的口径。\n"
                "如果问题需要排行、明细列表、原因分析、建议或跨节点依赖，当前选择器只负责选择已明确的指标；不能形成简单指标查询时输出 action=unsupported。\n"
                "不要选择候选之外的 ref，不要补造表名、字段名或指标名。\n"
                "输出格式：{{\"action\":\"select|semantic_read|ask_human|unsupported\",\"selectedRefs\":[\"semantic:...\"],\"readRefs\":[\"semantic:...\"],\"clarifications\":[{{\"phrase\":\"\",\"question\":\"\",\"options\":[{{\"ref\":\"\",\"label\":\"\"}}]}}],\"reason\":\"\"}}。\n"
                "通用示例：用户明确使用候选短卡中的限定指标名且定义充分 => action=select。\n"
                "通用示例：同一原词存在多个不同口径且短卡不足 => action=semantic_read；读完仍无法唯一判断 => action=ask_human。\n"
                "通用示例：用户一次询问多个彼此独立且定义充分的指标 => selectedRefs 必须覆盖每个指标分组。\n"
                "通用示例：问题要求原因分析、排行、明细或依赖链 => 可保留已明确指标 ref，但 action=unsupported 交给完整 Planner。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="planner.question_understanding",
            version="v2",
            agent="PlannerAgent",
            description="Understand a BI question into semantic-layer bounded questionUnderstanding.",
            section_ids=["common.stable_boundary", "planner.semantic_boundary", "common.artifact_references"],
            template=(
                "你是商家 BI 的 QueryGraph Planner。只通过 emit_question_understanding 输出结构化结果，不输出 SQL、QueryGraph 文本或解释长文。\n"
                "你不是 Core Agent，也不是 subagent；文件工具权限严格服从本次 runtime-section。{filesystem_authority_instruction}\n"
                "你必须依据本次已授权的精确语义读取证据，独立验证哪些已读表能同时覆盖用户要求的事实、维度、过滤与时间语义，"
                "再识别 analysisGrain、anchorMetric、supportMetrics、scopeConstraints、filters、timeWindowDays。候选排序和 L0 摘要都不是最终选表结论。\n"
                "画像汇总表只是可选的聚合资产：简单汇总可直接使用；排行、拆分、明细或原因分析可以直接选择已读的明细事实表，"
                "不得强制先查画像再沿 detailMetricRef 下钻。\n"
                "已授权精确读取证据中未出现的指标定义、字段、schema、关系或规则一律视为未读；缺少关键证据时返回 NEED_MORE_KNOWLEDGE，"
                "在 knowledgeRequests 中说明 Core 下一步应补读的 TABLE/FIELD/METRIC/RELATIONSHIP/BUSINESS_RULE，不得猜测或静默丢失条件。\n"
                "同时必须声明 analysisIntent、requiresExplanation、requiredEvidenceIntents：简单查询/排行用 none/false/[]；需要诊断、原因解释、异常判断、风险判断、经营总结时，由你声明所需证据意图。\n"
                "comparison 是两项已绑定指标或明确时间窗的关系契约，不是多指标列表；仅在用户明确要求差值、比例、相对高低、变化或基线判断时使用。"
                "逐项取值一律用 none/false/[]，不得按指标数量或并列措辞推断。\n"
                "comparison 必须有结构依据：calculationIntents 提供用户原话 sourcePhrase 及两个已绑定的不同 metricRef；"
                "时间比较须有两个窗口且 windowRelation=comparison/explicit_comparison；explicit_conjunction 只是逐窗取值。\n"
                "只要 analysisIntent 不是 none，requiredEvidenceIntents 必须至少 1 条；comparison/trend_check/anomaly_check/risk_ranking/overview/diagnosis 都不能返回空 evidence intents。\n"
                "不要依赖代码关键词补规则；如果需要解释型证据，把证据需求写进 requiredEvidenceIntents，再由语义层编译和 Critic 校验。\n"
                "metricRef 必须来自精确 METRIC 读取证据中的 metricKey；ownerTable 必须等于该定义的 table。\n"
                "groupBy、过滤、选择字段必须来自已读 COLUMN 定义或已读 SCHEMA；跨表依赖必须来自已读 RELATIONSHIPS。"
                "只读过 manifest、table detail 或 section index 只能用于导航，不能授权具体绑定。\n"
                "Metric candidate selection：同一用户指标 phrase 下多个同名/近义候选默认互斥；先看 title/tableKind/grainHint/formula/description，在 metricCandidateDecisions 中 selected_one 或 need_clarification。除非用户明确要求口径对账/差异排查，不要同时查询多个同名候选；rejectedCandidateIds 不能进入 anchorMetric/supportMetrics/QueryGraph。\n"
                "通用候选规则：整体汇总请求选择 metadata 标注为汇总粒度的候选；明细请求选择明细粒度候选；用户未要求口径对账时，同一 sourcePhrase 只能 selected_one，不要双查互斥口径；多个候选都合理时 need_clarification。\n"
                "如果输入包含 knowledgeRequestGaps，表示这些补知识请求已经失败或无新增证据；不要重复请求同一个知识，必须基于现有 semanticCatalog 规划可回答部分，或把不支持部分留成结构化缺口。\n"
                "memoryConstraints 只能作为本轮解释偏好、历史纠错或口径争议信号；不得用 memory 改写 semanticCatalog、指标公式、表关系或字段定义。\n"
                "如果 memoryConstraints 与 semanticCatalog 冲突，必须以 semanticCatalog 为准，并通过 validationGaps/clarification 表达未应用原因。\n"
                "sourcePhrase 必须只填写用户原话中的指标或业务对象原词，不要包含排序词、Top/前N、最高/最低、时间窗或分析动作。\n"
                "anchorMetric.objectiveType 用来表达主指标用途：求一个授权主体总量/指标值用 metric_total；Top/最高/最多/前N 用 ranking；走势用 trend_anchor；明细实体过滤用 detail_anchor。\n"
                "如果用户问 Top/最高/最多/前N，anchorMetric 必须是被排序的主指标；其他指标放 supportMetrics。\n"
                "如果用户只问“某指标是多少/怎么样/当前值”，不要伪造成 Top 排名；选择对应 metricRef，objectiveType=metric_total，groupByColumn 使用所选资产声明的主体粒度字段。\n"
                "如果用户问具体实体明细，anchorMetric 可以为空，但 filters 必须写出语义目录中存在的实体字段和值。\n"
                "如果用户问题包含状态、阶段、处理进度、成功/失败/异常/处理中等限定，filters 必须写出对应 semanticCatalog/live schema 里的状态字段和值；多个状态值用逗号分隔。\n"
                "如果用户表达“在某业务集合中/某业务对象带来的/使用某业务对象的/基于某集合”的限定，必须写入 scopeConstraints；后续 anchorMetric 和 supportMetrics 必须先受这个实体集合约束。scopeConstraints.ownerTable 必须来自产生该集合的语义资产，不能按目标表猜测。\n"
                "如果用户要从一个业务集合关联查看另一个业务域的证据，且 semanticCatalog 已提供相关资产或 relationships，应输出 UNDERSTOOD。没有显式排序时，anchorMetric 可选择定义主集合的最小可执行指标或留空，supportMetrics 放需要补证据的指标。\n"
                "关系链的 join key 不需要你猜，后续编译器会从 semanticCatalog.relationships 选择；你只需要准确声明分析粒度、主集合和需要补充的业务域/指标。\n"
                "选择 anchorMetric 时要贴合用户原始排序短语；用户只说某指标时优先选择该原词的直接语义定义，不要自动替换成其他派生指标。\n"
                "如果输入含 diagnosticContext 且 semanticCatalog 有候选指标，优先围绕 diagnosticContext.intent/goal 选择可执行的 overview/risk_ranking/comparison 理解，不要空泛返回 NEED_MORE_KNOWLEDGE。\n"
                "如果用户问走势、相关、匹配、同步上升或异常波动，analysisGrain 通常为 day；选择一个主时间序列指标做 anchorMetric，其余序列指标放 supportMetrics。\n"
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
                "如果 critic 指出 scope 未落地、分析证据契约缺失或未覆盖，必须重新声明 scopeConstraints、analysisIntent、requiresExplanation、requiredEvidenceIntents，并把所需指标放入 supportMetrics 或 knowledgeRequests。\n"
                "修复 comparison 时不能仅保留多个并列指标：必须绑定两个指标 operand，或使用 planningContract 中明确的比较时间窗；"
                "若用户只要求逐项取值，应修复为 analysisIntent=none、requiresExplanation=false、requiredEvidenceIntents=[]。\n"
                "如果 critic 指出 MEMORY_CONSTRAINT_UNAPPLIED，必须只在 semanticCatalog 可支持时选择对应 metricRef；不支持时输出 clarification/knowledge gap，不得修改语义层定义。\n"
                "如果 critic 指出同名/近义指标冲突，必须先修复 metricCandidateDecisions：同一用户指标 phrase 下多个候选默认互斥，除非用户明确要求口径对账/差异排查，不要同时查询多个同名候选。\n"
                "通用修复规则：同一 sourcePhrase 被多个互斥口径同时选中时，按用户要求的粒度和资产 selectionGuidance 修复为 selected_one；仍无法唯一判断时返回 need_clarification。\n"
                "如果输入包含 knowledgeRequestGaps，不要重复请求这些已失败的补知识项；只能基于现有 semanticCatalog 修复，或保留结构化缺口。\n"
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
                "GROUP_AGG/TOPN 查询必须让 nodePlanContract 中所有非聚合 SELECT 字段同时出现在 GROUP BY 中，不能丢 outputKeys 或 groupByColumn。\n"
                "当 groupByColumn 的语义角色是实体键时，必须按 nodePlanContract 的空值策略过滤 NULL 和空字符串，避免产生空实体桶。\n"
                "dependent node 用 upstreamEntitySets 做 IN 过滤。主体过滤必须使用 nodePlanContract.merchantFilterColumn 和授权值。\n"
                "TimeWindowContract 必须落地：相对时间窗必须锚定 preferredTable 在当前授权主体过滤后的 MAX(timeColumn)，不要用 CURDATE()/CURRENT_DATE。"
                "通用写法：`timeColumn` BETWEEN DATE_SUB((SELECT MAX(`timeColumn`) FROM `preferredTable` WHERE `merchantFilterColumn` = <authorized value>), INTERVAL N-1 DAY) AND (SELECT MAX(`timeColumn`) FROM `preferredTable` WHERE `merchantFilterColumn` = <authorized value>)。"
                "显式日期/明确 startDate-endDate 且 calendarAnchorPolicy=explicit_date_range 时才用固定日期 BETWEEN；不要使用 DATE_FORMAT('%Y%m%d')。"
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
                "你是受治理的业务分析助手。只基于输入的已验证数据回答，缺失证据要明确说明，不要把缺失解释成 0。\n"
                "回答要自然、简洁、先说结论，避免研发调试口吻。\n"
                "如果 verified evidence 同时出现同一 sourcePhrase 的多个口径，但 metricCandidateDecisions 已 selected_one，最终答案只能使用 selectedCandidateId 对应证据；rejectedCandidateIds 不能作为并列结论。只有用户明确要求口径对账时才可并列比较。\n"
                "指标名称和口径以 verified evidence 中的 metric_resolution / metricDisclosures 为准；字段名带 raw 表示原始字段值，不要把 raw 字段当作正式指标口径。"
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
                "你是经营分析助手。基于当前数据给出业务用户能读懂的判断，不在 SQL 阶段硬编码业务假设。\n"
                "不要输出“分析结论/关键证据/限制/口径”这类固定报告标题，不要暴露字段名、表名或 SQL。"
            ),
        )
    )
    registry.register(
        PromptTemplateSpec(
            prompt_id="answer.rule",
            version="v1",
            agent="AnswerAgent",
            description="Answer governed knowledge questions from retrieved evidence only.",
            section_ids=["common.stable_boundary", "answer.verified_evidence", "common.artifact_references"],
            template="你是治理知识问答助手。只基于给定知识回答；没有依据时明确说明需要补充权威知识。",
        )
    )
    return registry
