import asyncio
import json
import time

import pytest

from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    ChatContext,
    EntitySet,
    EvidenceGap,
    GraphValidationResult,
    IntentType,
    KnowledgeRequest,
    KnowledgeRequestType,
    KnowledgeRef,
    MerchantInfo,
    NodeExecutionContext,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    QuestionRoute,
    RelationshipEntry,
    RoutingDecision,
    SqlRepairAttempt,
    TaskRole,
    ThreadData,
    ToolCallRequest,
    VerifiedEvidence,
)
from merchant_ai.services.assets import (
    HybridRecallService,
    PlanningAssetPackBuilder,
    SemanticCatalogService,
    SemanticMetricIndex,
    SkillLoader,
    TopicAssetService,
    WikiMemoryService,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.answer import AnswerComposeService
from merchant_ai.services.context import ContextManager
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.planning import (
    PlannerReflectionAgent,
    QuestionUnderstandingCompiler,
    QueryGraphPlanner,
    QueryGraphValidator,
    SemanticMetricResolver,
    compact_understanding_catalog,
    ultra_compact_understanding_catalog,
)
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService, choose_entity_transfer_key, table_access_hint
from merchant_ai.services.routing import KeywordExtractService, QuestionRoutingService, TopicRouterService
from merchant_ai.services.tool_runtime import ToolCallExecutor, ToolFailureRegistry, ToolRuntimePolicyRegistry
from merchant_ai.services.tools import (
    artifact_file_tool_schemas,
    lead_action_selection_tool,
    node_runtime_tool_schemas,
    question_understanding_tool,
    semantic_file_tool_schemas,
    sql_draft_tool,
)


REGRESSION_CASES_7D = [
    "最近 7 天查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。",
    "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。",
    "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。",
    "最近 7 天退款金额最高的商品，看看对应下单量是否也高。",
    "最近 7 天有退款的订单，关联看一下对应商品发布时间。",
    "最近 7 天客服工单里涉及退款的订单，同时看这些订单是否发生了赔付。",
    "最近 7 天赔付金额较高的订单，关联看一下订单金额和退款金额。",
    "最近 7 天优惠券相关订单表现怎么样，是否带来了更多下单？",
    "最近 7 天供应链入库较多的商品，同时看下这些商品的下单情况。",
    "最近 7 天商品审核被拒的 SPU，后续有没有产生订单或退款？",
]

EXTENDED_REGRESSION_CASES = [
    "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间，帮我判断哪些是高风险新品。",
    "最近60天赔付金额最高的前5个订单，关联看订单金额、退款金额、退款状态和对应客服工单情况。",
    "最近30天有客服工单的订单里，哪些后来发生了退款或赔付？分别占多少。",
    "最近90天下单量前20的SPU里，哪些退款率明显高于店铺平均水平？",
    "最近30天优惠券带来的订单里，退款率最高的商品有哪些，同时看券金额投入是否过高。",
    "最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。",
    "最近30天审核被拒后又重新发布成功的商品，后续有没有产生订单、退款或赔付。",
    "最近30天退款金额最高的几天，对应主要是哪几个商品、哪些订单、有没有赔付。",
    "最近60天赔付单量较高的商品，关联看退款量、退款金额和商品发布时间。",
    "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。",
    "最近90天哪些商品下单数不高，但退款率和赔付率都偏高。",
    "最近30天有退款的订单里，哪些对应的是最近15天新发布商品。",
    "最近60天客服工单量最高的前10个商品，同时看这些商品的退款量和赔付金额。",
    "最近30天退款状态为处理中或异常的订单，关联看商品发布时间和订单金额。",
    "最近90天高销量商品里，哪些是“下单多、退款多、赔付也多”的三高商品。",
    "最近30天优惠券活动覆盖的商品中，哪些商品虽然下单多，但退款金额也高。",
    "最近60天审核被拒商品中，哪些后来仍然产生了较多退款订单。",
    "最近30天赔付金额高的订单，对应商品是否也是退款高发商品。",
    "最近90天店铺整体退款率、赔付率、工单率走势是否同步上升，帮我分析可能原因。",
    "最近30天哪些商品最值得优先处理？请结合下单量、退款率、赔付金额、工单量一起判断。",
]


def test_prompt_assembler_builds_deerflow_style_runtime_sections():
    prompt = PromptAssembler().render(
        "lead.system",
        variables={"agent_name": "MerchantBILeadAgent", "max_concurrent_sub_agents": 3},
        sections={
            "available_actions": "- route_topic\n- plan_graph\n- execute_graph",
            "loaded_skills": "- trade\n- refund",
        },
    )
    assert '<prompt id="lead.system" version="v1" agent="LeadAgent">' in prompt.system_prompt
    assert '<runtime-section name="available_actions">' in prompt.system_prompt
    assert "plan_graph" in prompt.system_prompt
    assert "trade" in prompt.system_prompt
    assert prompt.trace()["promptId"] == "lead.system"
    assert prompt.trace()["sections"] == ["available_actions", "loaded_skills"]


def test_tool_schemas_define_runtime_call_format():
    lead_schema = lead_action_selection_tool(["route_topic", "plan_graph"]).trace_schema()
    assert lead_schema["name"] == "select_agent_action"
    assert lead_schema["parameters"]["properties"]["actionId"]["enum"] == ["route_topic", "plan_graph"]
    node_schemas = node_runtime_tool_schemas({"inspect_schema": "inspect schema", "execute_sql": "execute SQL"}, ["execute_sql"])
    assert [item["name"] for item in node_schemas] == ["execute_sql"]
    sql_schema = sql_draft_tool().openai_schema()
    assert sql_schema["function"]["name"] == "draft_sql"
    assert "sql" in sql_schema["function"]["parameters"]["required"]
    understanding_schema = question_understanding_tool().trace_schema()["parameters"]["properties"]["questionUnderstanding"]
    assert "analysisIntent" in understanding_schema["required"]
    assert "requiresExplanation" in understanding_schema["required"]
    assert "requiredEvidenceIntents" in understanding_schema["required"]


def test_skill_loader_selects_domain_skills():
    settings = get_settings()
    skills = SkillLoader(settings).select("最近 90 天有退款的订单，关联看一下对应商品发布时间。", [QuestionCategory.REFUND])
    domains = {skill.domain for skill in skills}
    assert "refund" in domains
    assert "goods" in domains
    for skill in skills:
        assert skill.retrieval_hints
        assert skill.tables == []
        assert skill.metrics == []
        assert skill.entity_keys == []
        assert skill.relationships == []
        assert skill.graph_patterns == []


def test_semantic_table_asset_is_the_canonical_runtime_source():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    asset = topic_assets.load_table_asset("电商交易", "dwm_trade_order_detail_di")
    assert asset["tableName"] == "dwm_trade_order_detail_di"
    assert topic_assets.load_table_schema("电商交易", "dwm_trade_order_detail_di") == asset["schemaColumns"]
    assert topic_assets.load_table_metrics("电商交易", "dwm_trade_order_detail_di") == asset["metrics"]
    assert topic_assets.load_table_terms("电商交易", "dwm_trade_order_detail_di") == asset["terms"]
    assert topic_assets.load_table_semantic_columns("电商交易", "dwm_trade_order_detail_di") == asset["semanticColumns"]
    assert topic_assets.load_table_knowledge_rules("电商交易", "dwm_trade_order_detail_di") == asset["knowledgeRules"]


def test_semantic_catalog_exposes_filesystem_context_tools():
    settings = get_settings()
    catalog = SemanticCatalogService(TopicAssetService(settings))
    refs = catalog.ls(topic="电商交易", query="dwm_trade_order_detail_di", limit=5)
    order_ref = next(ref for ref in refs if ref["table"] == "dwm_trade_order_detail_di")
    assert order_ref["path"] == "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json"
    assert order_ref["layers"]["metrics"] > 0
    content = catalog.read(ref_id=order_ref["refId"], max_chars=800)
    assert content["success"]
    assert "dwm_trade_order_detail_di" in content["content"]
    hits = catalog.grep("order_detail_cnt", topic="电商交易", limit=5)
    assert any(hit["refId"] == order_ref["refId"] for hit in hits)


def test_semantic_file_tool_schemas_are_runtime_injected():
    schemas = semantic_file_tool_schemas()
    names = {schema["name"] for schema in schemas}
    assert {"semantic_ls", "semantic_read", "semantic_grep", "semantic_write"} <= names
    read_schema = next(schema for schema in schemas if schema["name"] == "semantic_read")
    assert {"refId", "path", "maxChars", "offset"} <= set(read_schema["parameters"]["properties"])
    artifact_names = {schema["name"] for schema in artifact_file_tool_schemas()}
    assert {"artifact_ls", "artifact_read", "artifact_grep", "artifact_write"} <= artifact_names


def test_planner_semantic_tool_loop_loads_ref_then_emits_understanding(tmp_path):
    settings = get_settings()
    settings.agent_planner_tool_rounds = 3
    topic_assets = TopicAssetService(settings)
    catalog = SemanticCatalogService(topic_assets)
    question = "最近 90 天下单最多的前 5 个 SPU，同时看它们的退款量。"
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    planner = QueryGraphPlanner(
        SemanticToolLoopPlannerLlm(),
        semantic_catalog=catalog,
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )
    plan, requests, _ = planner.plan(question, [], "", recall, pack, [], [])
    assert not requests
    assert plan.intents
    assert any(call["name"] == "semantic_read" for call in plan.planner_tool_calls)
    assert "semantic:电商交易:dwm_trade_order_detail_di:asset" in plan.planner_loaded_refs
    assert plan.planner_context_files
    assert any("planner.semantic_tool_loop=enabled" == item for item in plan.agent_trace)
    assert any("planner.semantic_tool_loop=on_demand" == item for item in plan.agent_trace)


def test_planner_fast_path_skips_semantic_tool_loop_when_understood(tmp_path):
    settings = get_settings()
    settings.agent_planner_tool_rounds = 3
    topic_assets = TopicAssetService(settings)
    catalog = SemanticCatalogService(topic_assets)
    question = "最近 90 天下单最多的前 5 个 SPU，同时看它们的退款量。"
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    llm = FastPathPlannerLlm()
    planner = QueryGraphPlanner(
        llm,
        semantic_catalog=catalog,
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )
    plan, requests, _ = planner.plan(question, [], "", recall, pack, [], [])
    assert not requests
    assert plan.intents
    assert llm.tool_chat_calls == 0
    assert plan.planner_tool_calls == []
    assert not any("planner.semantic_tool_loop" in item for item in plan.agent_trace)


def test_planner_refines_analysis_understanding_with_semantic_tool_loop(tmp_path):
    settings = get_settings()
    settings.agent_planner_tool_rounds = 3
    topic_assets = TopicAssetService(settings)
    catalog = SemanticCatalogService(topic_assets)
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    recall = recall_bundle_empty()
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall,
        [
            QuestionCategory.TRADE,
            QuestionCategory.REFUND,
            QuestionCategory.CS_TICKET,
            QuestionCategory.COMPENSATION,
        ],
    )
    assert "ads_merchant_profile" not in pack.known_tables()
    llm = RefiningAnalysisPlannerLlm()
    planner = QueryGraphPlanner(
        llm,
        semantic_catalog=catalog,
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )
    plan, requests, _ = planner.plan(question, [], "", recall, pack, [], [])
    assert not requests
    assert plan.intents
    assert plan.intents[0].preferred_table == "ads_merchant_profile"
    assert plan.question_understanding["rankingObjective"]["metricRef"] == "order_gmv_amt_1d"
    assert "ads_merchant_profile" in pack.known_tables()
    assert any(call["name"] == "semantic_grep" for call in plan.planner_tool_calls)
    assert any("previousUnderstanding" in payload for payload in llm.tool_chat_payloads)
    assert any("planner.semantic_tool_loop=refined_understanding" == item for item in plan.agent_trace)


def test_recall_uses_single_semantic_asset_document_per_table():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    docs = recall._load_documents()
    order_docs = [doc for doc in docs if doc.table == "dwm_trade_order_detail_di" and doc.topic == "电商交易"]
    assert len(order_docs) == 1
    assert order_docs[0].source_type == "SEMANTIC_TABLE_ASSET"
    assert order_docs[0].doc_id == "semantic:电商交易:dwm_trade_order_detail_di:asset"
    assert order_docs[0].metadata["semanticPath"] == "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json"
    assert "order_detail_cnt" in order_docs[0].content
    assert not any(doc.source_type in {"TOPIC_TABLE", "TOPIC_ASSET", "RELATIONSHIP"} for doc in docs)
    assert not any(any(name in doc.doc_id for name in ["metrics.json", "terms.json", "semantic_columns.json", "knowledge_rules.json"]) for doc in docs)


def test_planning_asset_pack_refs_point_to_unified_semantic_catalog():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        "最近 90 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。",
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    asset_entries = pack.tables + pack.fields + pack.metrics + pack.entity_keys + pack.terms + pack.relationships
    assert asset_entries
    assert all((entry.source_ref_id or "").startswith("semantic:") for entry in asset_entries)


def test_lead_policy_has_no_standalone_load_skills_action():
    policy = V2AgentPolicy(get_settings())
    assert "load_skills" not in policy.registry.public_action_ids()
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": False,
        "react_round": 1,
        "plan": QueryPlan(),
    }
    decision = policy.decide(state)
    assert decision.selected_action == "retrieve_knowledge"
    assert decision.selected_node == "retrieve_knowledge"


def test_asset_pack_does_not_auto_expand_tables_from_relationship_closure():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact("最近 90 天有退款的商品发布时间。", recall_bundle_empty(), [QuestionCategory.REFUND, QuestionCategory.GOODS])
    tables = set(pack.known_tables())
    assert "dwm_trade_refund_detail_di" in tables
    assert "dwm_goods_detail_df" in tables
    assert "dwm_trade_order_detail_di" not in tables
    assert pack.relationship_closure == []


def test_asset_pack_expands_metric_formula_dependency_tables_when_metric_requested():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        "最近45天优惠券金额投入最高的商品，退款率是否偏高？",
        recall_bundle_empty(),
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    tables = set(pack.known_tables())
    metric_keys = {metric.key for metric in pack.metrics}
    relationships = {rel.relationship_id for rel in pack.relationships}
    assert "dwm_trade_order_detail_di" in tables
    assert "order_detail_cnt" in metric_keys
    assert any(item == "metric_dependency:refund_rate->order_detail_cnt:dwm_trade_order_detail_di" for item in pack.relationship_closure)
    assert {"coupon_order_by_coupon_id", "order_refund_by_sub_order"} <= relationships


def test_sql_validator_allows_cte_and_rejects_unknown_real_table():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "spu_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "pay_amt", "pt"]),
        ]
    )
    validator = SqlValidationService()
    sql = """
    WITH top_spu AS (
      SELECT seller_id, spu_id, COUNT(*) AS order_cnt
      FROM dwm_trade_order_detail_di
      GROUP BY seller_id, spu_id
    )
    SELECT *
    FROM top_spu t
    JOIN dwm_trade_refund_detail_di r ON t.seller_id = r.seller_id
    """
    result = validator.validate(sql, pack)
    assert result.valid
    assert "top_spu" in result.cte_names
    assert set(result.base_tables) == {"dwm_trade_order_detail_di", "dwm_trade_refund_detail_di"}
    doris_date = validator.validate(
        "SELECT seller_id FROM dwm_trade_order_detail_di WHERE pt >= DATE_SUB(CURRENT_DATE(), 6)",
        pack,
    )
    assert doris_date.valid
    rejected = validator.validate("SELECT * FROM not_in_pack", pack)
    assert not rejected.valid
    assert rejected.error_code == "UNKNOWN_BASE_TABLE"


@pytest.mark.slow
def test_llm_understanding_compiler_builds_graph_for_regression_cases():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    categories = [
        QuestionCategory.TRADE,
        QuestionCategory.REFUND,
        QuestionCategory.GOODS,
        QuestionCategory.CS_TICKET,
        QuestionCategory.COMPENSATION,
        QuestionCategory.COUPON,
        QuestionCategory.SCM,
    ]
    topic_assets = TopicAssetService(settings)
    planner = QueryGraphPlanner(FakeCaseUnderstandingLlm(), SemanticCatalogService(topic_assets))
    for question in REGRESSION_CASES_7D:
        pack = builder.compact(question, recall_bundle_empty(), categories)
        plan, requests, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
        assert not requests, question
        assert plan.intents, question
        assert plan.evidence_contracts, question
        assert plan.question_understanding, question
        assert all(intent.knowledge_refs for intent in plan.intents), question
        assert not any("knowledge_grounded" in item or "pattern:" in item for item in plan.agent_trace)


@pytest.mark.slow
def test_llm_understanding_compiler_builds_graph_for_extended_risk_cases():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    categories = [
        QuestionCategory.TRADE,
        QuestionCategory.REFUND,
        QuestionCategory.GOODS,
        QuestionCategory.CS_TICKET,
        QuestionCategory.COMPENSATION,
        QuestionCategory.COUPON,
        QuestionCategory.SCM,
    ]
    topic_assets = TopicAssetService(settings)
    planner = QueryGraphPlanner(FakeCaseUnderstandingLlm(), SemanticCatalogService(topic_assets))
    for question in EXTENDED_REGRESSION_CASES:
        pack = builder.compact(question, recall_bundle_empty(), categories)
        plan, requests, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
        assert not requests, question
        assert plan.intents, question
        assert plan.evidence_contracts, question
        ranking = plan.question_understanding["rankingObjective"]
        resolution = plan.intents[0].metric_resolution or {}
        assert plan.intents[0].preferred_table == (resolution.get("ownerTable") or ranking["ownerTable"]), question
        assert plan.intents[0].metric_name == (resolution.get("metricKey") or ranking["metricRef"]), question
        assert not any("knowledge_grounded" in item or "pattern:" in item for item in plan.agent_trace)


def test_planner_uses_ultra_catalog_and_surfaces_timeout_gap():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    llm = FakeTimeoutThenCompactUnderstandingLlm()
    plan, requests, _ = QueryGraphPlanner(llm).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert not plan.intents
    assert llm.calls == 1
    assert llm.catalog_sizes[0] == len(json.dumps(ultra_compact_understanding_catalog(pack, question), ensure_ascii=False))
    assert any("PLANNER_LLM_TIMEOUT" in item for item in plan.agent_trace)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert [gap.code for gap in validation.gaps] == ["PLANNER_LLM_TIMEOUT"]


def test_llm_understanding_nodes_carry_refs_and_no_declared_extended_patterns():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    categories = [
        QuestionCategory.TRADE,
        QuestionCategory.REFUND,
        QuestionCategory.GOODS,
        QuestionCategory.CS_TICKET,
        QuestionCategory.COMPENSATION,
        QuestionCategory.COUPON,
        QuestionCategory.SCM,
    ]
    questions = [
        EXTENDED_REGRESSION_CASES[1],
        EXTENDED_REGRESSION_CASES[2],
        EXTENDED_REGRESSION_CASES[4],
        EXTENDED_REGRESSION_CASES[9],
        EXTENDED_REGRESSION_CASES[19],
    ]
    topic_assets = TopicAssetService(settings)
    planner = QueryGraphPlanner(FakeCaseUnderstandingLlm(), SemanticCatalogService(topic_assets))
    for question in questions:
        pack = builder.compact(question, recall_bundle_empty(), categories)
        assert all(not skill.graph_patterns for skill in pack.skills)
        plan, _, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
        assert plan.intents, question
        assert all(intent.knowledge_refs for intent in plan.intents), question
        assert not any("pattern.intent=" in item or "knowledge_grounded" in item for item in plan.agent_trace)


def test_llm_understanding_prunes_unrequested_neighbor_domains():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近60天赔付金额最高的前5个订单，关联看订单金额、退款金额、退款状态和对应客服工单情况。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.COMPENSATION, QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET],
    )
    plan, _, _ = QueryGraphPlanner(FakeCaseUnderstandingLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    tables = {intent.preferred_table for intent in plan.intents}
    assert {"dwm_cs_repay_detail_df", "dwm_trade_order_detail_di", "dwm_trade_refund_detail_di", "dwm_cs_ticket_detail_di"} <= tables
    assert "dwm_goods_detail_df" not in tables
    assert "dwm_coupon_detail_di" not in tables
    assert "dwm_scm_detail_di" not in tables
    assert "ads_merchant_profile" not in tables


def test_llm_understanding_compiler_skips_relationship_missing_live_join_key():
    question = "最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_scm_detail_di",
                columns=["seller_id", "spu_id", "inbound_cnt", "pt"],
                title="供应链入库明细",
            ),
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "spu_id", "pay_amt", "pt"],
                title="订单明细",
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "sub_order_id", "spu_name", "pay_amt", "pt"],
                title="退款明细",
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="scm_refund_by_spu_name",
                left_table="dwm_scm_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_name", "rightColumn": "spu_name"},
                ],
            ),
            RelationshipEntry(
                relationship_id="scm_order_by_spu_id",
                left_table="dwm_scm_detail_di",
                right_table="dwm_trade_order_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            ),
        ],
    )
    pack.metrics = [
        PlanningAssetEntry(key="inbound_cnt", table="dwm_scm_detail_di", columns=["inbound_cnt"], title="入库数量"),
        PlanningAssetEntry(key="order_detail_cnt", table="dwm_trade_order_detail_di", columns=["sub_order_id"], title="下单数"),
        PlanningAssetEntry(key="refund_bill_cnt", table="dwm_trade_refund_detail_di", columns=["pay_amt"], title="退款量"),
    ]
    plan, _, _ = QueryGraphPlanner(FakeCaseUnderstandingLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    task_table = {intent.plan_task_id: intent.preferred_table for intent in plan.intents}
    refund_task_ids = {task_id for task_id, table in task_table.items() if table == "dwm_trade_refund_detail_di"}
    assert refund_task_ids
    for dep in plan.dependencies:
        if dep.dependent_task_id in refund_task_ids:
            assert task_table[dep.anchor_task_id] == "dwm_trade_order_detail_di"
            assert dep.anchor_column == "seller_id+sub_order_id"


def test_query_graph_planner_uses_llm_understanding_not_knowledge_grounded_anchor():
    question = "最近60天客服工单量最高的前10个商品，同时看这些商品的退款量和赔付金额。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_ticket_detail_di", columns=["seller_id", "ticket_id", "sub_order_id", "spu_id", "spu_name", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "refund_id", "sub_order_id", "spu_name", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "bill_id", "sub_order_id", "repay_amt", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="ticket_cnt",
                table="dwm_cs_ticket_detail_di",
                columns=["ticket_id"],
                title="工单量",
                source_ref_id="metric:ticket_cnt",
            ),
            PlanningAssetEntry(
                key="refund_cnt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_id"],
                title="退款量",
                source_ref_id="metric:refund_cnt",
            ),
            PlanningAssetEntry(
                key="repay_amt",
                table="dwm_cs_repay_detail_df",
                columns=["repay_amt"],
                title="赔付金额",
                source_ref_id="metric:repay_amt",
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="ticket_refund_by_sub_order",
                left_table="dwm_cs_ticket_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="ticket_repay_by_sub_order",
                left_table="dwm_cs_ticket_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
        ],
    )
    planner = QueryGraphPlanner(FakePlannerLlm())
    plan, requests, reason = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert reason == "llm understood ranking objective"
    assert plan.intents[0].preferred_table == "dwm_cs_ticket_detail_di"
    assert plan.intents[0].plan_task_id == "anchor_ticket"
    assert "prompt=planner.question_understanding@v1" in plan.agent_trace
    assert "tool_schema=emit_question_understanding" in plan.agent_trace
    assert any("planner=llm_understanding" in item for item in plan.agent_trace)
    assert not any("knowledge_grounded" in item for item in plan.agent_trace)
    assert all(intent.knowledge_refs for intent in plan.intents)


def test_llm_understanding_product_graph_adds_goods_lookup_without_cycles():
    question = "最近60天客服工单量最高的前10个商品，同时看这些商品的退款量和赔付金额。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_ticket_detail_di", columns=["seller_id", "ticket_id", "sub_order_id", "spu_id", "spu_name", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "order_id", "spu_id", "spu_name", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "refund_id", "sub_order_id", "spu_name", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "bill_id", "sub_order_id", "repay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_goods_detail_df", columns=["seller_id", "spu_id", "spu_name", "spu_apply_create_time", "spu_status_name", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="ticket_cnt", table="dwm_cs_ticket_detail_di", columns=["ticket_id"], title="工单量"),
            PlanningAssetEntry(key="refund_cnt", table="dwm_trade_refund_detail_di", columns=["refund_id"], title="退款量"),
            PlanningAssetEntry(key="repay_amt", table="dwm_cs_repay_detail_df", columns=["repay_amt"], title="赔付金额"),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="ticket_order_by_sub_order",
                left_table="dwm_cs_ticket_detail_di",
                right_table="dwm_trade_order_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_repay_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_goods_by_spu",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_goods_detail_df",
                join_keys=[{"leftColumn": "spu_id", "rightColumn": "spu_id"}],
            ),
        ],
    )
    plan, _, _ = QueryGraphPlanner(FakePlannerLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    task_tables = {intent.plan_task_id: intent.preferred_table for intent in plan.intents}
    assert task_tables["anchor_ticket"] == "dwm_cs_ticket_detail_di"
    assert task_tables["goods_lookup"] == "dwm_goods_detail_df"
    assert not any(dep.anchor_task_id == "repay_lookup" and dep.dependent_task_id == "order_bridge" for dep in plan.dependencies)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [gap.code for gap in validation.gaps]
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)
    assert reflection.passed, reflection.issues


def test_understanding_catalog_keeps_profile_daily_metrics_for_gmv_trend():
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    pack = profile_daily_pack()
    catalog = compact_understanding_catalog(pack, question)
    tables = {item["table"] for item in catalog["tables"]}
    assert "ads_merchant_profile" not in tables

    pack.metric_compaction.setdefault("questionUnderstandingExpansion", []).append(
        "metric_request_table:order_gmv_amt_1d->ads_merchant_profile:ownerTable"
    )
    catalog = compact_understanding_catalog(pack, question)
    tables = {item["table"] for item in catalog["tables"]}
    metric_keys = [item["key"] for item in catalog["candidateMetrics"]]
    assert "ads_merchant_profile" in tables
    assert "order_gmv_amt_1d" in metric_keys
    assert {"refund_amt_1d", "seller_repay_amt_1d", "cs_ticket_cnt_1d"} <= set(metric_keys)


def test_understanding_catalog_limits_initial_tables_after_topic_routing():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    catalog = compact_understanding_catalog(pack, question)
    tables = {item["table"] for item in catalog["tables"]}
    metric_keys = [item["key"] for item in catalog["candidateMetrics"]]
    metric_tables = {item["table"] for item in catalog["candidateMetrics"]}
    relationship_tables = {
        table
        for item in catalog["relationships"]
        for table in [item["leftTable"], item["rightTable"]]
    }
    assert 1 <= len(tables) <= 3
    assert tables == {"dwm_trade_order_detail_di", "dwm_trade_refund_detail_di", "dwm_goods_detail_df"}
    assert "ads_merchant_profile" not in tables
    assert "order_detail_cnt" in metric_keys
    assert "refund_bill_cnt" in metric_keys
    assert any(item["key"] == "pay_amt" and item["table"] == "dwm_trade_refund_detail_di" for item in catalog["candidateMetrics"])
    assert metric_tables <= tables
    assert relationship_tables <= tables


def test_asset_pack_trims_metrics_before_planner_prompt():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近7天店铺GMV、退款率、赔付率、工单率趋势是否一起变差？"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )
    metric_keys = {metric.key for metric in pack.metrics}
    assert len(pack.metrics) <= 24
    assert pack.metric_compaction["before"] > pack.metric_compaction["after"]
    assert "ads_merchant_profile" not in pack.known_tables()
    assert "refund_rate_1d" not in metric_keys
    builder.expand_for_question_understanding(
        pack,
        {
            "rankingObjective": {
                "metricRef": "order_gmv_amt_1d",
                "ownerTable": "ads_merchant_profile",
                "sourcePhrase": "GMV",
            },
            "requestedMeasures": [
                {"metricRef": "refund_rate_1d", "ownerTable": "ads_merchant_profile", "sourcePhrase": "退款率"},
                {"metricRef": "seller_repay_amt_1d", "ownerTable": "ads_merchant_profile", "sourcePhrase": "赔付金额"},
                {"metricRef": "cs_ticket_cnt_1d", "ownerTable": "ads_merchant_profile", "sourcePhrase": "工单量"},
            ],
        },
    )
    expanded_metric_keys = {metric.key for metric in pack.metrics}
    assert "ads_merchant_profile" in pack.known_tables()
    assert {"order_gmv_amt_1d", "refund_rate_1d", "seller_repay_amt_1d", "cs_ticket_cnt_1d"} <= expanded_metric_keys


def test_first_step_topic_router_covers_trade_refund_goods_for_top_spu_refund_question():
    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    keywords = KeywordExtractService().extract(question)
    route = QuestionRoutingService().route(question, keywords, recall_bundle_empty())
    topic = TopicRouterService().route(question, keywords)
    assert route.route == QuestionRoute.BUSINESS
    assert {QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS} <= set(topic.candidate_topics)
    assert "不表示 anchor" in topic.reason


def test_first_step_topic_router_does_not_invent_topics_for_store_diagnosis():
    question = "当前店铺最近 90 天整体经营情况怎么样，帮我总结风险和机会。"
    keywords = KeywordExtractService().extract(question)
    route = QuestionRoutingService().route(question, keywords, recall_bundle_empty())
    topic = TopicRouterService().route(question, keywords)
    assert route.route == QuestionRoute.BUSINESS
    assert topic.candidate_topics == []
    assert not topic.clarification_required
    assert "开放 scope" in topic.reason
    assert QuestionCategory.PLATFORM_RULE not in topic.candidate_topics


def test_first_step_rejects_bare_analysis_without_business_subject():
    question = "分析问题"
    keywords = KeywordExtractService().extract(question)
    route = QuestionRoutingService().route(question, keywords, recall_bundle_empty())
    assert route.route == QuestionRoute.INVALID


def test_open_store_diagnostic_expands_discovery_topics_without_clarification():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "当前店铺最近 90 天整体经营情况怎么样，帮我总结风险和机会。",
        "merchant_001",
        ChatContext(),
        None,
        "test_open_store_diagnostic",
        "run_open_store_diagnostic",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert not state["human_clarification_required"]
    assert state["open_diagnostic_scope"] == "OPEN_DIAGNOSTIC"
    assert state["open_diagnostic_intent"] == "STORE_HEALTH_DIAGNOSIS"
    assert {QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COMPENSATION, QuestionCategory.CS_TICKET} <= set(
        state["open_diagnostic_seed_topics"]
    )
    assert state["topic_routing_decision"].recall_topics() == []
    state = workflow.retrieve_knowledge(state)
    state = workflow.compact_assets(state)
    assert {"trade", "refund", "goods", "ticket_compensation"} <= set(state["loaded_skills"])
    assert state["planning_asset_pack"].known_tables()
    assert state["planning_asset_pack"].metrics


def test_open_store_diagnostic_expands_even_with_explicit_after_sales_topics():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "最近90天店铺整体退款率、赔付率、工单率走势是否同步上升，帮我分析可能原因。",
        "merchant_001",
        ChatContext(),
        None,
        "test_open_store_trend_with_topics",
        "run_open_store_trend_with_topics",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert not state["human_clarification_required"]
    assert state["open_diagnostic_intent"] == "STORE_HEALTH_DIAGNOSIS"
    assert {QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COMPENSATION, QuestionCategory.CS_TICKET} <= set(
        state["open_diagnostic_seed_topics"]
    )
    assert QuestionCategory.REFUND in state["topic_routing_decision"].recall_topics()
    assert QuestionCategory.TRADE in workflow._effective_topic_categories(state)


def test_open_priority_recommendation_asks_goal_before_discovery():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "如果我只优先处理 3 个问题，最近 90 天数据建议我先处理什么？",
        "merchant_001",
        ChatContext(),
        None,
        "test_open_priority",
        "run_open_priority",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert state["human_clarification_required"]
    assert state["human_clarification_stage"] == "OPEN_DIAGNOSTIC"
    assert state["human_clarification_type"] == "priority_goal"
    assert "综合经营风险" in state["human_clarification_options"]
    assert state["open_diagnostic_intent"] == "PRIORITY_RECOMMENDATION"


def test_open_priority_recommendation_continues_after_goal_clarification():
    workflow = create_workflow(get_settings())
    pending_question = "如果我只优先处理 3 个问题，最近 90 天数据建议我先处理什么？"
    state = workflow._initial_state(
        "综合经营风险",
        "merchant_001",
        ChatContext(
            pending_clarification_stage="OPEN_DIAGNOSTIC",
            pending_clarification_type="priority_goal",
            pending_question=pending_question,
            pending_clarification_options=["综合经营风险", "降低退款/赔付损失"],
        ),
        None,
        "test_open_priority_continue",
        "run_open_priority_continue",
    )
    state = workflow.inherit_context(state)
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert not state["human_clarification_required"]
    assert state["open_diagnostic_intent"] == "PRIORITY_RECOMMENDATION"
    assert state["open_diagnostic_goal"] == "综合经营风险"
    assert QuestionCategory.TRADE in state["open_diagnostic_seed_topics"]
    assert QuestionCategory.SCM in state["open_diagnostic_seed_topics"]
    state = workflow.retrieve_knowledge(state)
    assert {"trade", "refund", "goods", "ticket_compensation", "coupon", "scm"} <= set(state["loaded_skills"])


def test_open_priority_recommendation_keeps_scope_for_domain_specific_goal():
    workflow = create_workflow(get_settings())
    pending_question = "我这个月只能处理两个问题，建议先处理什么最划算？"
    state = workflow._initial_state(
        "降低退款/赔付损失",
        "merchant_001",
        ChatContext(
            pending_clarification_stage="OPEN_DIAGNOSTIC",
            pending_clarification_type="priority_goal",
            pending_question=pending_question,
            pending_clarification_options=["综合经营风险", "降低退款/赔付损失"],
        ),
        None,
        "test_open_priority_domain_goal",
        "run_open_priority_domain_goal",
    )
    state = workflow.inherit_context(state)
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert not state["human_clarification_required"]
    assert state["open_diagnostic_intent"] == "PRIORITY_RECOMMENDATION"
    assert state["open_diagnostic_goal"] == "降低退款/赔付损失"
    assert QuestionCategory.COUPON in state["open_diagnostic_seed_topics"]
    state = workflow.retrieve_knowledge(state)
    assert {"refund", "ticket_compensation", "trade"} <= set(state["loaded_skills"])
    state = workflow.compact_assets(state)
    assert "ads_merchant_profile" in state["planning_asset_pack"].known_tables()


def test_validator_requests_missing_metric_dependency_for_refund_rate_denominator():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        "最近 90 天退款率偏高的商品有哪些？",
        recall_bundle_empty(),
        [QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近 90 天退款率偏高的商品有哪些？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                task_role=TaskRole.ANCHOR,
                plan_task_id="anchor_refund",
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="refund_rate",
                metric_column="refund_bill_cnt",
                group_by_column="spu_name",
            )
        ]
    )
    validation = QueryGraphValidator().validate(plan.intents[0].question, plan, pack)
    assert not validation.valid
    assert any(gap.code == "MISSING_METRIC_DEPENDENCY" and gap.evidence == "order_detail_cnt" for gap in validation.gaps)
    assert any(request.type == KnowledgeRequestType.METRIC and "order_detail_cnt" in request.query for request in validation.recommended_knowledge_requests)


def test_knowledge_request_relationship_scope_expands_by_table_path():
    workflow = create_workflow(get_settings())
    request = KnowledgeRequest(
        type=KnowledgeRequestType.RELATIONSHIP,
        query=(
            "是否存在 dwm_cs_ticket_detail_di 与 dwm_trade_refund_detail_di 之间可用于关联分析的关系"
        ),
    )
    topics = workflow._knowledge_request_topics(request, [QuestionCategory.REFUND, QuestionCategory.CS_TICKET])
    assert QuestionCategory.TRADE in topics
    assert QuestionCategory.REFUND in topics
    assert QuestionCategory.CS_TICKET in topics


def test_llm_understanding_compiles_same_table_daily_metric_dependents():
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    pack = profile_daily_pack()
    plan, requests, _ = QueryGraphPlanner(FakeDailyProfilePlannerLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    metrics_by_task = {intent.plan_task_id: intent.metric_name for intent in plan.intents}
    assert metrics_by_task["anchor_order"] == "order_gmv_amt_1d"
    assert "refund_amt_1d" in metrics_by_task.values()
    assert "seller_repay_amt_1d" in metrics_by_task.values()
    assert "cs_ticket_cnt_1d" in metrics_by_task.values()
    assert all(dep.anchor_task_id == "anchor_order" and dep.join_key == "pt" for dep in plan.dependencies)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [gap.code for gap in validation.gaps]
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)
    assert reflection.passed, reflection.issues


def test_llm_understanding_compiler_adds_formula_dependency_measure_nodes():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近45天优惠券金额投入最高的商品，退款率是否偏高？"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    plan, requests, _ = QueryGraphPlanner(FakeCouponRefundRatePlannerLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    metrics_by_task = {intent.plan_task_id: intent.metric_name for intent in plan.intents}
    assert metrics_by_task["anchor_coupon"] == "coupon_amt"
    assert "refund_rate" in metrics_by_task.values()
    assert "order_detail_cnt" in metrics_by_task.values()
    assert any("FORMULA_DEP_METRICS:order_detail_cnt" == item for item in plan.compiler_trace)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [(gap.code, gap.evidence) for gap in validation.gaps]


def test_llm_understanding_compiles_deposit_gmv_day_relation_from_semantic_layer():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    question = "最近180天保证金充值金额变化和GMV变化是否相关？"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.MERCHANT_OTHER],
    )
    assert "ads_merchant_profile" not in pack.known_tables()
    plan, requests, _ = QueryGraphPlanner(FakeDepositGmvPlannerLlm(), SemanticCatalogService(topic_assets)).plan(
        question,
        [],
        "",
        recall_bundle_empty(),
        pack,
        [],
        [],
    )
    assert not requests
    tables = {intent.preferred_table for intent in plan.intents}
    assert {"dwd_merchant_deposit_recharge_df", "ads_merchant_profile"} <= tables
    assert any(dep.join_key == "pt" for dep in plan.dependencies)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [(gap.code, gap.evidence) for gap in validation.gaps]
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)
    assert reflection.passed, reflection.issues


def test_planner_passes_compact_diagnostic_context_to_llm():
    llm = FakeDiagnosticContextPlannerLlm()
    question = "当前店铺最近 90 天整体经营情况怎么样，帮我总结风险和机会。"
    plan, requests, _ = QueryGraphPlanner(llm).plan(
        question,
        [],
        "",
        recall_bundle_empty(),
        profile_daily_pack(),
        [],
        [],
        {
            "openDiagnostic": {
                "scope": "OPEN_DIAGNOSTIC",
                "intent": "STORE_HEALTH_DIAGNOSIS",
                "goal": "综合经营健康度",
                "seedTopics": ["TRADE", "REFUND", "CS_TICKET", "COMPENSATION", "GOODS"],
            }
        },
    )
    assert not requests
    assert plan.intents
    assert llm.payloads[0]["diagnosticContext"]["intent"] == "STORE_HEALTH_DIAGNOSIS"
    assert llm.payloads[0]["diagnosticContext"]["goal"] == "综合经营健康度"
    file_context = llm.payloads[0]["semanticFileContext"]
    assert file_context["mode"] == "filesystem_as_context"
    assert "semantic_read" in file_context["tools"]
    assert "semantic_write" in file_context["tools"]
    assert any(str(ref["path"]).endswith("/asset.json") for ref in file_context["refs"])


def test_llm_understanding_compiles_detail_lookup_without_ranking_objective():
    question = "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "order_id", "sub_order_id", "spu_id", "spu_name", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "order_id", "sub_order_id", "refund_id", "spu_name", "pay_amt", "buyer_id", "refund_create_time", "pt"]),
            PlanningAssetEntry(table="dwm_goods_detail_df", columns=["seller_id", "spu_id", "spu_name", "spu_apply_create_time", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="pay_amt", table="dwm_trade_refund_detail_di", columns=["pay_amt"], title="退款关联支付金额"),
            PlanningAssetEntry(key="goods_cnt", table="dwm_goods_detail_df", columns=["spu_id"], title="商品数"),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "seller_id", "rightColumn": "seller_id"}, {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_goods_by_spu",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_goods_detail_df",
                join_keys=[{"leftColumn": "seller_id", "rightColumn": "seller_id"}, {"leftColumn": "spu_id", "rightColumn": "spu_id"}],
            ),
        ],
    )
    plan, requests, _ = QueryGraphPlanner(FakeDetailPlannerLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    tables = {intent.preferred_table for intent in plan.intents}
    assert {"dwm_trade_order_detail_di", "dwm_trade_refund_detail_di", "dwm_goods_detail_df"} <= tables
    assert plan.intents[0].filter_column == "order_id"
    assert plan.intents[0].filter_value == "order_id_100"
    assert plan.intents[0].answer_mode == "DETAIL"
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_planner_semantic_repair_adds_missing_requested_domain_dependencies():
    question = "最近60天赔付金额最高的前5个订单，关联看订单金额、退款金额、退款状态和对应客服工单情况。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "order_id", "sub_order_id", "bill_id", "repay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "order_id", "sub_order_id", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "refund_id", "pay_amt", "refund_status_name", "pt"]),
            PlanningAssetEntry(table="dwm_cs_ticket_detail_di", columns=["seller_id", "sub_order_id", "ticket_id", "ticket_status_name", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="repay_amt", table="dwm_cs_repay_detail_df", columns=["repay_amt"], title="赔付金额", metadata={"formula": "SUM(repay_amt)"}),
            PlanningAssetEntry(key="pay_amt", table="dwm_trade_order_detail_di", columns=["pay_amt"], title="订单金额", metadata={"formula": "SUM(pay_amt)"}),
            PlanningAssetEntry(key="refund_related_pay_amt", table="dwm_trade_refund_detail_di", columns=["pay_amt"], title="退款金额", metadata={"formula": "SUM(pay_amt)"}),
            PlanningAssetEntry(key="ticket_cnt", table="dwm_cs_ticket_detail_di", columns=["ticket_id"], title="工单量", metadata={"formula": "COUNT(DISTINCT ticket_id)"}),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_repay_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="order_ticket_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_ticket_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_repay",
                preferred_table="dwm_cs_repay_detail_df",
                metric_column="repay_amt",
                metric_name="repay_amt",
                group_by_column="order_id",
                output_keys=["seller_id", "order_id", "sub_order_id"],
            ),
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="order_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_order_detail_di",
                metric_column="pay_amt",
                metric_name="pay_amt",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                depends_on_task_ids=["anchor_repay"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_repay",
                dependent_task_id="order_lookup",
                join_key="sub_order_id",
                anchor_column="sub_order_id",
                dependent_column="sub_order_id",
            )
        ],
    )
    repaired = QueryGraphPlanner(FakePlannerLlm()).repair(question, plan, pack, [], [], "", recall_bundle_empty())
    tables = {intent.preferred_table for intent in repaired.intents}
    assert "dwm_trade_refund_detail_di" in tables
    assert "dwm_cs_ticket_detail_di" in tables
    assert QueryGraphValidator().validate(question, repaired, pack).valid


def test_llm_client_timeout_cancels_async_provider_call():
    settings = get_settings()
    old_timeout = settings.llm_request_timeout_seconds
    settings.llm_request_timeout_seconds = 1
    client = LlmClient(settings)
    model = SlowAsyncModel()
    client._model = model
    try:
        result = client.chat("system", "user", "fallback")
    finally:
        settings.llm_request_timeout_seconds = old_timeout
    assert result == "fallback"
    assert client.last_error.startswith("timeout:")
    assert model.cancelled


def test_llm_client_tool_json_chat_parses_native_tool_call():
    settings = get_settings()
    client = LlmClient(settings)
    model = ToolCallingAsyncModel()
    client._model = model
    result = client.tool_json_chat("system", "user", sql_draft_tool().openai_schema(), {})
    assert result["sql"] == "SELECT 1"
    assert model.bound_tools[0]["function"]["name"] == "draft_sql"
    assert model.tool_choice == "draft_sql"


def test_context_manager_snapshots_protected_facts(tmp_path):
    settings = get_settings()
    manager = ContextManager(settings)
    state = {
        "question": "最近30天GMV最高的前5天，同时看退款金额。",
        "requested_merchant_id": "seller_100",
        "thread_data": ThreadData(outputs_path=str(tmp_path)),
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="GMV top days",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_gmv",
                    preferred_table="ads_merchant_profile",
                )
            ]
        ),
        "query_graph_validation_result": GraphValidationResult(),
    }
    snapshot = manager.refresh_state(state, "plan_query_graph")
    keys = {fact.key for fact in snapshot.protected_facts}
    assert {"question", "merchant_id", "plan_tables", "plan_tasks"} <= keys
    assert state["summary_context"]
    assert (tmp_path / "context_snapshot.json").exists()
    dumped = state["context_snapshots"][-1]
    assert dumped["protectedFacts"][0]["sourceRefs"]


def test_tool_failure_registry_blocks_repeated_identical_failures():
    registry = ToolFailureRegistry(repeat_threshold=2, circuit_threshold=4)
    args = {"taskId": "anchor", "sql": "SELECT * FROM x"}
    assert registry.should_block("execute_sql", args) is None
    registry.record_failure("execute_sql", args, "MEM_ALLOC_FAILED", "memory pressure")
    assert registry.should_block("execute_sql", args) is None
    registry.record_failure("execute_sql", args, "MEM_ALLOC_FAILED", "memory pressure")
    blocked = registry.should_block("execute_sql", args)
    assert blocked is not None
    assert blocked.open
    assert "repeated identical failure" in blocked.reason


def test_tool_call_executor_pairs_parallel_results_and_isolates_failures():
    settings = get_settings()
    registry = ToolFailureRegistry()
    executor = ToolCallExecutor(ToolRuntimePolicyRegistry(settings), registry, max_concurrency=2)

    def ok(args):
        return {"value": args["value"]}

    def boom(args):
        raise RuntimeError("bad argument")

    results = executor.execute(
        [
            ToolCallRequest(id="call_1", name="ok_tool", args={"value": 7}),
            ToolCallRequest(id="call_2", name="bad_tool", args={}),
        ],
        {"ok_tool": ok, "bad_tool": boom},
    )
    assert [item.id for item in results] == ["call_1", "call_2"]
    assert results[0].status == "success"
    assert results[0].result == {"value": 7}
    assert results[1].status == "failed"
    assert results[1].error_type == "ERROR"
    assert registry.trace()["failures"]


def test_execute_sql_runtime_policy_leaves_repair_to_node_worker():
    settings = get_settings()
    policy = ToolRuntimePolicyRegistry(settings).policy_for("execute_sql")
    assert policy.max_retries == 0
    assert "MEM_ALLOC_FAILED" in policy.non_retryable_errors
    assert not policy.retryable_errors


def test_planner_overrides_need_more_when_semantic_catalog_is_sufficient():
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量。"
    pack = profile_daily_pack()
    llm = FakeNeedMoreThenProfilePlannerLlm()
    plan, requests, _ = QueryGraphPlanner(llm).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert plan.intents
    assert llm.calls == 2
    assert any("planner.need_more_overridden_by_semantic_catalog" in item for item in plan.agent_trace)


def test_structured_formula_uses_semantic_metric_formula_not_sum_first_column():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    intent = QuestionIntent(
        question="退款率走势",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="refund_rate",
        preferred_table="ads_merchant_profile",
        sql_strategy="structured_first",
        group_by_column="pt",
        metric_column="refund_rate_1d",
        metric_name="refund_rate_1d",
        metric_formula="AVG(refund_rate_1d)",
        output_keys=["seller_id", "pt"],
    )
    pack.tables[0].columns.append("refund_rate_1d")
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "AVG(`refund_rate_1d`) AS `refund_rate_1d`" in sql
    assert "SUM(`refund_rate_1d`)" not in sql
    assert SqlValidationService().validate(sql, pack).valid


def test_node_agent_llm_writes_plan_bound_sql_from_contract():
    settings = get_settings()
    worker = NodeWorkerExecutor(GoodPlanBoundSqlLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前5天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天")
    assert result.task_results[0].success
    sql = result.task_results[0].query_bundle.sql
    assert "order_gmv_amt_1d" in sql
    assert "ads_merchant_profile" in sql
    assert result.sql_draft_decisions[0].source == "llm_plan_bound"
    assert result.node_plan_contracts[0].allowed_columns
    assert result.node_plan_critiques[0].valid


def test_doris_mem_error_uses_resource_safe_fallback_without_llm_repair():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 2
    llm = GoodPlanBoundSqlLlm()
    worker = NodeWorkerExecutor(llm, MemoryFailThenOkDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前80天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=80,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前80天")
    assert result.task_results[0].success
    assert result.sql_repairs[0].error_code == "MEM_ALLOC_FAILED"
    assert "LIMIT 20" in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_resource_safe_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_doris_mem_error_does_not_call_llm_repair_when_no_safe_fallback():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 2
    llm = GoodPlanBoundSqlLlm()
    worker = NodeWorkerExecutor(llm, AlwaysMemoryFailDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前20天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=20,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前20天")
    assert not result.task_results[0].success
    assert "MEM_ALLOC_FAILED" in result.task_results[0].query_bundle.error
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_node_agent_rejects_contract_external_field_and_uses_structured_fallback():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    llm = BadContractColumnSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前5天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天")
    assert result.task_results[0].success
    assert result.sql_repairs
    assert result.sql_repairs[0].success
    assert result.sql_repairs[0].error_code == "UNKNOWN_CONTRACT_COLUMN"
    assert "SUM(`order_gmv_amt_1d`)" in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_structured_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_node_agent_missing_output_key_uses_structured_fallback_without_llm_repair():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    llm = MissingOutputKeySqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "spu_id", "spu_name", "pay_amt"],
            )
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天下单最多的前5个SPU",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                group_by_column="spu_id",
                metric_column="pay_amt",
                metric_name="order_gmv",
                output_keys=["seller_id", "spu_id", "spu_name"],
                days=30,
                limit=5,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天下单最多的前5个SPU")
    assert result.task_results[0].success
    assert result.sql_repairs[0].error_code == "MISSING_OUTPUT_KEY"
    assert "`spu_name`" in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_structured_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_node_agent_invalid_partition_filter_uses_structured_fallback_without_llm_repair():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    llm = InvalidPartitionSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前5天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天")
    assert result.task_results[0].success
    assert result.sql_repairs[0].error_code == "INVALID_PARTITION_FILTER"
    assert "DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 30 DAY), '%Y%m%d')" in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_structured_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_node_worker_merges_same_table_metric_nodes_before_execution():
    settings = get_settings()
    worker = NodeWorkerExecutor(EmptySqlLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="GMV最高日期",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
                sql_strategy="structured_first",
            ),
            QuestionIntent(
                question="看退款金额",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="refund_metric",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="refund_amt_1d",
                metric_name="refund_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
                sql_strategy="structured_first",
            ),
            QuestionIntent(
                question="看工单量",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="ticket_metric",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="cs_ticket_cnt_1d",
                metric_name="cs_ticket_cnt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
                sql_strategy="structured_first",
            ),
        ],
        evidence_contracts=[
            {"taskId": "refund_metric", "table": "ads_merchant_profile", "columns": ["refund_amt_1d"], "semanticLabel": "退款金额"}
        ],
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天，同时看退款金额和工单量")
    assert len(plan.intents) == 1
    assert len(result.task_results) == 1
    assert len(plan.intents[0].metric_specs) == 3
    assert plan.evidence_contracts[0]["taskId"] == "anchor_gmv"
    sql = result.task_results[0].query_bundle.sql
    assert "SUM(`order_gmv_amt_1d`) AS `order_gmv_amt_1d`" in sql
    assert "SUM(`refund_amt_1d`) AS `refund_amt_1d`" in sql
    assert "SUM(`cs_ticket_cnt_1d`) AS `cs_ticket_cnt_1d`" in sql
    assert "COUNT(*) AS `order_cnt`" not in sql
    assert "LIMIT 5" in sql
    assert any("same_table_metric_merge" in item for item in plan.agent_trace)


def test_node_worker_same_table_merge_keeps_topn_anchor_when_not_first():
    settings = get_settings()
    worker = NodeWorkerExecutor(EmptySqlLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="看退款金额",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="refund_metric",
                task_role="DEPENDENT",
                depends_on_task_ids=["anchor_gmv"],
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="refund_amt_1d",
                metric_name="refund_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=20,
                sql_strategy="structured_first",
            ),
            QuestionIntent(
                question="GMV最高日期",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
                sql_strategy="structured_first",
            ),
        ],
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天，同时看退款金额")
    assert len(plan.intents) == 1
    assert plan.intents[0].plan_task_id == "anchor_gmv"
    sql = result.task_results[0].query_bundle.sql
    assert "ORDER BY `order_gmv_amt_1d` DESC LIMIT 5" in sql


def test_node_worker_executes_independent_ready_nodes_concurrently():
    settings = get_settings()
    settings.max_concurrent_sub_agents = 2
    settings.agent_node_timeout_seconds = 5
    worker = NodeWorkerExecutor(EmptySqlLlm(), SlowDoris(delay_seconds=0.25), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="table_a", columns=["seller_id"]),
            PlanningAssetEntry(table="table_b", columns=["seller_id"]),
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="A",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="node_a",
                preferred_table="table_a",
                output_keys=["seller_id"],
                sql="SELECT `seller_id` FROM `table_a` WHERE `seller_id` = '100' LIMIT 1",
            ),
            QuestionIntent(
                question="B",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="node_b",
                preferred_table="table_b",
                output_keys=["seller_id"],
                sql="SELECT `seller_id` FROM `table_b` WHERE `seller_id` = '100' LIMIT 1",
            ),
        ]
    )
    started = time.monotonic()
    result = worker.execute_plan("100", plan, pack, "", "并发测试")
    elapsed = time.monotonic() - started
    assert [item.success for item in result.task_results] == [True, True]
    assert elapsed < 0.45


def test_asset_builder_caches_live_schema_lookup():
    class CountingSchemaDoris:
        def __init__(self):
            self.calls = 0

        def show_full_columns(self, table):
            self.calls += 1
            return [{"Field": "seller_id"}, {"Field": "pt"}]

    repo = CountingSchemaDoris()
    builder = PlanningAssetPackBuilder(TopicAssetService(get_settings()), doris_repository=repo)
    assert builder._live_schema("some_table")
    assert builder._live_schema("some_table")
    assert repo.calls == 1


def test_node_contract_critic_blocks_missing_metric_before_llm_sql():
    settings = get_settings()
    llm = GoodPlanBoundSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="ads_merchant_profile", columns=["seller_id", "pt", "order_gmv_amt_1d"])]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="GMV",
                intent_type="VALID",
                answer_mode="METRIC",
                plan_task_id="metric_gmv",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id"],
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "GMV")
    assert not result.task_results[0].success
    assert result.node_plan_critiques[0].code == "MISSING_METRIC_COLUMN"
    assert llm.calls == 0


def test_node_contract_critic_blocks_dependent_without_upstream_entity():
    settings = get_settings()
    llm = GoodPlanBoundSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pay_amt", "pt"])]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="退款明细",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="refund_lookup",
                task_role="DEPENDENT",
                depends_on_task_ids=["order_anchor"],
                preferred_table="dwm_trade_refund_detail_di",
                required_evidence=["sub_order_id", "pay_amt"],
                output_keys=["seller_id", "sub_order_id"],
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "退款明细")
    assert not result.task_results[0].success
    assert result.node_plan_critiques[0].code == "MISSING_UPSTREAM_ENTITY"
    assert any(gap.code == "MISSING_UPSTREAM_ENTITY" for gap in result.evidence_gaps)
    assert llm.calls == 0


def test_node_agent_empty_llm_sql_falls_back_to_structured_sql():
    settings = get_settings()
    worker = NodeWorkerExecutor(EmptySqlLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = profile_daily_pack()
    intent = QuestionIntent(
        question="GMV趋势",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="gmv_trend",
        preferred_table="ads_merchant_profile",
        group_by_column="pt",
        metric_column="order_gmv_amt_1d",
        metric_name="order_gmv_amt_1d",
        output_keys=["seller_id", "pt"],
        days=30,
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100", question="GMV趋势"))
    decision = worker._last_sql_draft_decisions[intent.plan_task_id]
    assert "SUM(`order_gmv_amt_1d`)" in sql
    assert decision.source == "structured_fallback"
    assert decision.structured_fallback_used


def test_query_graph_enrichment_removes_self_loop_dependencies_and_syncs_intents():
    question = "最近60天客服工单量最高的前10个商品，同时看这些商品的退款量和赔付金额。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_ticket_detail_di", columns=["seller_id", "ticket_id", "sub_order_id", "spu_id", "spu_name", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "refund_id", "sub_order_id", "spu_name", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "bill_id", "ticket_id", "sub_order_id", "repay_amt", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="ticket_cnt", table="dwm_cs_ticket_detail_di", columns=["ticket_id"], title="工单量"),
            PlanningAssetEntry(key="refund_bill_cnt", table="dwm_trade_refund_detail_di", columns=["refund_id"], title="退款量"),
            PlanningAssetEntry(key="repay_amt", table="dwm_cs_repay_detail_df", columns=["repay_amt"], title="赔付金额"),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="ticket_refund_by_sub_order",
                left_table="dwm_cs_ticket_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            ),
            RelationshipEntry(
                relationship_id="ticket_repay_by_ticket",
                left_table="dwm_cs_ticket_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[{"leftColumn": "ticket_id", "rightColumn": "ticket_id"}],
            ),
        ],
    )
    planner = QueryGraphPlanner(FakeSelfLoopPlannerLlm())
    plan, _, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert plan.dependencies
    assert not any(dep.anchor_task_id == dep.dependent_task_id for dep in plan.dependencies)
    dependency_pairs = {(dep.anchor_task_id, dep.dependent_task_id) for dep in plan.dependencies}
    assert ("anchor_ticket", "ticket_entity_expand") in dependency_pairs
    assert ("ticket_entity_expand", "refund_lookup") in dependency_pairs
    assert ("anchor_ticket", "repay_lookup") in dependency_pairs
    depends_by_task = {intent.plan_task_id: intent.depends_on_task_ids for intent in plan.intents}
    assert depends_by_task["refund_lookup"] == ["ticket_entity_expand"]
    assert depends_by_task["repay_lookup"] == ["anchor_ticket"]


def test_query_graph_validator_rejects_dependency_key_missing_from_node_schema():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_scm_detail_di", columns=["seller_id", "spu_id", "inbound_cnt", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "pay_amt", "pt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="scm_refund_by_spu_name",
                left_table="dwm_scm_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_name", "rightColumn": "spu_name"},
                ],
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="供应链退款",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_scm",
                task_role="ANCHOR",
                preferred_table="dwm_scm_detail_di",
                group_by_column="spu_id",
                output_keys=["seller_id", "spu_id"],
            ),
            QuestionIntent(
                question="供应链退款",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="refund_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_refund_detail_di",
                depends_on_task_ids=["anchor_scm"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_scm",
                dependent_task_id="refund_lookup",
                join_key="seller_id+spu_name",
                anchor_column="seller_id+spu_name",
                dependent_column="seller_id+spu_name",
            )
        ],
    )
    result = QueryGraphValidator().validate("供应链退款", plan, pack)
    assert any(gap.code == "DEPENDENCY_KEY_NOT_IN_SCHEMA" for gap in result.gaps)
    assert result.repairable


def test_query_graph_validator_rejects_dependency_cycles():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[{"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"}],
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="订单退款",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="order_lookup",
                preferred_table="dwm_trade_order_detail_di",
                output_keys=["sub_order_id"],
            ),
            QuestionIntent(
                question="订单退款",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="refund_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_refund_detail_di",
                output_keys=["sub_order_id"],
                depends_on_task_ids=["order_lookup"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="order_lookup",
                dependent_task_id="refund_lookup",
                join_key="sub_order_id",
                anchor_column="sub_order_id",
                dependent_column="sub_order_id",
            ),
            PlanDependency(
                anchor_task_id="refund_lookup",
                dependent_task_id="order_lookup",
                join_key="sub_order_id",
                anchor_column="sub_order_id",
                dependent_column="sub_order_id",
            ),
        ],
    )
    result = QueryGraphValidator().validate("最近7天订单退款情况", plan, pack)
    assert not result.valid
    assert "CYCLIC_DEPENDENCY_EDGE" in {gap.code for gap in result.gaps}


def test_query_graph_validator_rejects_self_dependency_edge():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pt"])],
        relationships=[],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="退款",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="refund_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_refund_detail_di",
                depends_on_task_ids=["refund_lookup"],
            )
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="refund_lookup",
                dependent_task_id="refund_lookup",
                join_key="sub_order_id",
                anchor_column="sub_order_id",
                dependent_column="sub_order_id",
            )
        ],
    )
    result = QueryGraphValidator().validate("退款", plan, pack)
    assert not result.valid
    assert any(gap.code == "SELF_DEPENDENCY_EDGE" for gap in result.gaps)


def test_lead_policy_registry_selects_reflection_before_validation():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": False,
        "query_graph_validated": False,
        "react_round": 4,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="订单退款",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="anchor_order",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ],
            evidence_contracts=[{"taskId": "anchor_order", "table": "dwm_trade_order_detail_di", "columns": ["sub_order_id"]}],
        ),
    }
    policy = V2AgentPolicy(get_settings())
    decision = policy.decide(state)
    assert decision.selected_action == "reflect_plan"
    assert decision.selected_node == "reflect_query_graph"
    assert "validate_graph" in decision.available_actions


def test_lead_policy_does_not_retry_plan_after_provider_timeout():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": False,
        "query_graph_validated": False,
        "react_round": 5,
        "query_graph_plan_attempts": 1,
        "planner_provider_error": "timeout: provider call exceeded 15 seconds",
        "plan": QueryPlan(),
    }
    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "validate_graph"
    assert "plan_graph" not in decision.available_actions
    assert decision.budget_exhausted


def test_lead_policy_prioritizes_existing_plan_over_pending_requests():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": False,
        "query_graph_validated": False,
        "react_round": 5,
        "query_graph_retrieve_count": 0,
        "pending_knowledge_requests": [{"type": "FIELD", "query": "optional context"}],
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="赔付订单",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_repay",
                    preferred_table="dwm_cs_repay_detail_df",
                )
            ]
        ),
    }
    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "reflect_plan"
    assert "retrieve_knowledge" not in decision.available_actions


def test_lead_policy_does_not_retrieve_loop_after_reflection_when_plan_exists():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": False,
        "react_round": 7,
        "query_graph_retrieve_count": 1,
        "query_graph_repair_attempts": 0,
        "planner_reflection": PlannerReflectionResult(
            passed=False,
            issues=[{"code": "MISSING_OPTIONAL_KNOWLEDGE", "reason": "missing optional knowledge"}],
            suggested_knowledge_requests=[{"type": "FIELD", "query": "optional context"}],
        ),
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="赔付订单",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_repay",
                    preferred_table="dwm_cs_repay_detail_df",
                )
            ]
        ),
    }
    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "repair_graph"
    assert "retrieve_knowledge" not in decision.available_actions


def test_lead_policy_repairs_graph_after_repairable_evidence_gap():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "sql_generated": True,
        "sql_repair_reviewed": True,
        "evidence_graph_verified": True,
        "chat_bi_completed": False,
        "react_round": 8,
        "query_graph_repair_attempts": 0,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="供应链退款",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_scm",
                    preferred_table="dwm_scm_detail_di",
                )
            ],
            evidence_contracts=[{"taskId": "anchor_scm", "table": "dwm_scm_detail_di", "columns": ["spu_id"]}],
        ),
        "query_graph_validation_result": GraphValidationResult(valid=True),
        "agent_run_result": AgentRunResult(
            evidence_gaps=[
                EvidenceGap(
                    code="JOIN_KEY_NOT_PRODUCED",
                    task_id="refund_lookup",
                    reason="upstream node did not produce dependency key",
                )
            ]
        ),
    }
    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "repair_graph"
    assert decision.selected_node == "repair_query_graph"


def test_lead_policy_repairs_graph_before_sql_repair_for_contract_gap():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "sql_generated": True,
        "sql_repair_reviewed": False,
        "evidence_graph_verified": False,
        "chat_bi_completed": False,
        "react_round": 8,
        "query_graph_repair_attempts": 0,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="GMV",
                    intent_type="VALID",
                    answer_mode="METRIC",
                    plan_task_id="metric_gmv",
                    preferred_table="ads_merchant_profile",
                )
            ],
            evidence_contracts=[{"taskId": "metric_gmv", "table": "ads_merchant_profile", "columns": ["order_gmv_amt_1d"]}],
        ),
        "query_graph_validation_result": GraphValidationResult(valid=True),
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="metric_gmv",
                    success=False,
                    query_bundle=QueryBundle(failed=True, error="MISSING_METRIC_COLUMN：metric node has no metricColumn"),
                )
            ],
            evidence_gaps=[
                EvidenceGap(
                    code="MISSING_METRIC_COLUMN",
                    task_id="metric_gmv",
                    reason="metric node has no metricColumn",
                )
            ],
        ),
    }
    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "repair_graph"
    assert decision.selected_node == "repair_query_graph"


def test_planner_reflection_detects_missing_domain_and_missing_refs():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pay_amt", "pt"]),
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="订单退款",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                output_keys=["sub_order_id"],
            )
        ],
        evidence_contracts=[{"taskId": "anchor_order", "table": "dwm_trade_order_detail_di", "columns": ["sub_order_id"]}],
    )
    reflection = PlannerReflectionAgent().reflect("最近7天订单退款情况", plan, pack)
    codes = {issue["code"] for issue in reflection.issues}
    assert not reflection.passed
    assert "DOMAIN_COVERAGE_GAP" in codes
    assert "MISSING_KNOWLEDGE_REF" in codes
    assert "repair_graph" in reflection.suggested_actions


def test_planner_reflection_passes_when_domains_refs_and_evidence_present():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pay_amt", "pt"]),
        ]
    )
    ref = KnowledgeRef(ref_id="trade:order", ref_type="TABLE", table="dwm_trade_order_detail_di")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="订单退款",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                output_keys=["sub_order_id"],
                knowledge_refs=[ref],
            ),
            QuestionIntent(
                question="订单退款",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="refund_lookup",
                preferred_table="dwm_trade_refund_detail_di",
                depends_on_task_ids=["anchor_order"],
                knowledge_refs=[ref.model_copy(update={"table": "dwm_trade_refund_detail_di"})],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="refund_lookup",
                join_key="sub_order_id",
                anchor_column="sub_order_id",
                dependent_column="sub_order_id",
            )
        ],
        evidence_contracts=[
            {"taskId": "anchor_order", "table": "dwm_trade_order_detail_di", "columns": ["sub_order_id"]},
            {"taskId": "refund_lookup", "table": "dwm_trade_refund_detail_di", "columns": ["sub_order_id"]},
        ],
    )
    reflection = PlannerReflectionAgent().reflect("最近7天订单退款情况", plan, pack)
    assert reflection.passed


def test_planner_reflection_flags_low_confidence_metric_resolution():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "pay_amt"])],
    )
    ref = KnowledgeRef(ref_id="refund:metric:pay_amt", ref_type="METRIC", table="dwm_trade_refund_detail_di")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="售后金额",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_refund",
                task_role="ANCHOR",
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="pt",
                output_keys=["pt"],
                knowledge_refs=[ref],
                metric_resolution={
                    "requestedMetricRef": "unknown_amt",
                    "metricKey": "pay_amt",
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "confidence": 0.4,
                    "resolutionSource": "semantic_weak_match",
                },
            )
        ],
        evidence_contracts=[{"taskId": "anchor_refund", "table": "dwm_trade_refund_detail_di", "columns": ["pt", "pay_amt"]}],
    )
    reflection = PlannerReflectionAgent().reflect("最近30天售后金额", plan, pack)
    assert reflection.passed
    assert any(issue["code"] == "METRIC_RESOLUTION_LOW_CONFIDENCE" for issue in reflection.issues)
    assert reflection.repair_reason == "METRIC_RESOLUTION_LOW_CONFIDENCE"


def test_planner_reflection_uses_declared_analysis_contract():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "pay_amt"])]
    )
    ref = KnowledgeRef(ref_id="refund:table", ref_type="TABLE", table="dwm_trade_refund_detail_di")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="退款趋势",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_refund",
                task_role="ANCHOR",
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="pt",
                output_keys=["pt"],
                knowledge_refs=[ref],
            )
        ],
        evidence_contracts=[{"taskId": "anchor_refund", "table": "dwm_trade_refund_detail_di", "columns": ["pt", "pay_amt"]}],
        question_understanding={
            "analysisGrain": "day",
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [],
            "rankingObjective": {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di"},
            "requestedMeasures": [],
            "filters": [],
            "timeWindowDays": 30,
        },
    )
    reflection = PlannerReflectionAgent().reflect("最近30天指标情况", plan, pack)
    codes = {issue["code"] for issue in reflection.issues}
    assert "MISSING_ANALYSIS_EVIDENCE_CONTRACT" in codes
    assert reflection.repair_reason == "ANALYSIS_CONTRACT_MISSING"
    assert "plan_graph" in reflection.suggested_actions


def test_planner_reflection_checks_required_evidence_intents_not_question_terms():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "pay_amt"])]
    )
    ref = KnowledgeRef(ref_id="refund:table", ref_type="TABLE", table="dwm_trade_refund_detail_di")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="退款金额",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_refund",
                task_role="ANCHOR",
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="pt",
                output_keys=["pt"],
                knowledge_refs=[ref],
            )
        ],
        evidence_contracts=[{"taskId": "anchor_refund", "table": "dwm_trade_refund_detail_di", "columns": ["pt", "pay_amt"]}],
        question_understanding={
            "analysisGrain": "day",
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "explanation_context",
                    "reason": "需要关联订单、商品或客服证据解释波动",
                    "requiredLevel": "required",
                    "suggestedMetricRefs": ["order_detail_cnt", "ticket_cnt"],
                    "suggestedDomains": ["trade", "ticket"],
                }
            ],
            "rankingObjective": {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di"},
            "requestedMeasures": [],
            "filters": [],
            "timeWindowDays": 30,
        },
    )
    reflection = PlannerReflectionAgent().reflect("最近30天指标情况", plan, pack)
    codes = {issue["code"] for issue in reflection.issues}
    assert "ANALYSIS_EVIDENCE_NOT_COVERED" in codes
    assert reflection.repair_reason == "MISSING_REQUIRED_EVIDENCE"
    assert reflection.suggested_knowledge_requests


def test_planner_reflection_does_not_scan_analysis_phrases_without_llm_contract():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "pay_amt"])]
    )
    ref = KnowledgeRef(ref_id="refund:table", ref_type="TABLE", table="dwm_trade_refund_detail_di")
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="退款金额",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_refund",
                task_role="ANCHOR",
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="pt",
                output_keys=["pt"],
                knowledge_refs=[ref],
            )
        ],
        evidence_contracts=[{"taskId": "anchor_refund", "table": "dwm_trade_refund_detail_di", "columns": ["pt", "pay_amt"]}],
        question_understanding={
            "analysisGrain": "day",
            "analysisIntent": "none",
            "requiresExplanation": False,
            "requiredEvidenceIntents": [],
            "rankingObjective": {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di"},
            "requestedMeasures": [],
            "filters": [],
            "timeWindowDays": 30,
        },
    )
    reflection = PlannerReflectionAgent().reflect("最近30天退款金额最高的几天分别发生了什么", plan, pack)
    codes = {issue["code"] for issue in reflection.issues}
    assert "ANALYSIS_EVIDENCE_NOT_COVERED" not in codes
    assert "MISSING_ANALYSIS_EVIDENCE_CONTRACT" not in codes


def test_sql_validator_rejects_unqualified_unknown_single_table_column():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_coupon_detail_di",
                columns=["seller_id", "coupon_id", "coupon_amt", "pt"],
            )
        ]
    )
    result = SqlValidationService().validate(
        "SELECT coupon_refund_cnt FROM dwm_coupon_detail_di WHERE seller_id = '100'",
        pack,
    )
    assert not result.valid
    assert result.error_code == "UNKNOWN_COLUMN"
    assert result.unknown_columns == ["dwm_coupon_detail_di.coupon_refund_cnt"]


def test_sql_validator_does_not_hide_unknown_column_behind_same_name_alias():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_goods_detail_df",
                columns=["seller_id", "spu_id", "spu_name", "pt"],
            )
        ]
    )
    result = SqlValidationService().validate(
        """
        SELECT seller_id, AVG(spu_auth_price) AS spu_auth_price
        FROM dwm_goods_detail_df
        WHERE seller_id = '100'
        GROUP BY seller_id
        ORDER BY spu_auth_price DESC
        """,
        pack,
    )
    assert not result.valid
    assert result.error_code == "UNKNOWN_COLUMN"
    assert result.unknown_columns == ["dwm_goods_detail_df.spu_auth_price"]


def test_structured_first_uses_safe_single_table_sql_before_llm():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_coupon_detail_di",
                columns=["seller_id", "coupon_id", "coupon_amt", "pt"],
            )
        ]
    )
    intent = QuestionIntent(
        question="优惠券退款率最高商品",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="coupon_activity",
        preferred_table="dwm_coupon_detail_di",
        sql_strategy="structured_first",
        group_by_column="coupon_id",
        metric_column="coupon_amt",
        metric_name="coupon_amt",
        output_keys=["seller_id", "coupon_id"],
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "coupon_refund_cnt" not in sql
    assert "`seller_id` = '100'" in sql
    assert "`pt` >=" in sql
    assert "GROUP BY" in sql
    assert worker.llm.calls == 0


def test_structured_count_metric_uses_count_distinct_not_sum_identifier():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=["seller_id", "spu_id", "ticket_id", "pt"],
            )
        ]
    )
    intent = QuestionIntent(
        question="工单量最高商品",
        intent_type="VALID",
        answer_mode="TOPN",
        plan_task_id="anchor_ticket",
        preferred_table="dwm_cs_ticket_detail_di",
        sql_strategy="structured_first",
        group_by_column="spu_id",
        metric_column="ticket_id",
        metric_name="ticket_cnt",
        output_keys=["seller_id", "spu_id", "ticket_id"],
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "COUNT(DISTINCT `ticket_id`) AS `ticket_cnt`" in sql
    assert "SUM(`ticket_id`)" not in sql


def test_structured_topn_keeps_aggregate_grain_when_output_keys_include_detail_ids():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "spu_id", "spu_name", "sub_order_id", "pt"],
            )
        ]
    )
    intent = QuestionIntent(
        question="Top SPU 下单量",
        intent_type="VALID",
        answer_mode="TOPN",
        plan_task_id="top_spu",
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="structured_first",
        group_by_column="spu_id",
        metric_column="sub_order_id",
        metric_name="order_detail_cnt",
        output_keys=["seller_id", "spu_id", "spu_name", "sub_order_id"],
        required_evidence=["seller_id", "spu_id", "sub_order_id"],
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "COUNT(DISTINCT `sub_order_id`) AS `order_detail_cnt`" in sql
    assert "GROUP BY `seller_id`, `spu_id`, `spu_name`" in sql
    assert "GROUP BY `seller_id`, `spu_id`, `spu_name`, `sub_order_id`" not in sql


def test_node_agent_records_tool_traces_and_freshness_report():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDorisWithFreshness(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "pt"],
            )
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天下单量",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                sql_strategy="structured_first",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                days=1,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近7天下单量")
    assert result.task_results[0].success
    tools = [item.tool_name for item in result.node_tool_traces]
    assert "inspect_schema" in tools
    assert "check_freshness" in tools
    assert "draft_structured_sql" in tools
    assert "validate_sql" in tools
    assert "execute_sql" in tools
    assert result.freshness_reports[0].status == "AVAILABLE"
    assert result.freshness_reports[0].max_pt == "20260622"


def test_node_worker_writes_sql_and_result_artifacts(tmp_path):
    settings = get_settings()
    old_rows = settings.context_artifact_inline_max_rows
    settings.context_artifact_inline_max_rows = 1
    worker = NodeWorkerExecutor(FakeLlm(), ManyRowsDoris(), SqlValidationService(), settings)
    worker.with_artifact_root(str(tmp_path))
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "pt"],
            )
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天下单量",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                sql_strategy="structured_first",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                days=1,
            )
        ]
    )
    try:
        result = worker.execute_plan("100", plan, pack, "", "最近7天下单量")
    finally:
        settings.context_artifact_inline_max_rows = old_rows
    bundle = result.task_results[0].query_bundle
    assert bundle.original_row_count == 3
    assert len(bundle.rows) == 1
    assert any(path.endswith(".sql") for path in bundle.offloaded_files)
    assert any(path.endswith("_rows.json") for path in bundle.offloaded_files)
    assert all(tmp_path.as_posix() in path for path in bundle.offloaded_files)


def test_freshness_check_without_pt_is_explicitly_classified():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id"])]
    )
    intent = QuestionIntent(
        question="订单明细",
        intent_type="VALID",
        answer_mode="DETAIL",
        plan_task_id="anchor_order",
        preferred_table="dwm_trade_order_detail_di",
        days=1,
    )
    report = worker._check_freshness(intent, pack, NodeExecutionContext(merchant_id="100"))
    assert report.status == "NO_PT_COLUMN"
    assert not report.checked


def test_dependent_without_upstream_skips_freshness_query():
    settings = get_settings()
    doris = CountingDoris()
    worker = NodeWorkerExecutor(FakeLlm(), doris, SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "pt"])]
    )
    intent = QuestionIntent(
        question="退款 dependent",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="refund_lookup",
        task_role="DEPENDENT",
        preferred_table="dwm_trade_refund_detail_di",
        depends_on_task_ids=["order_anchor"],
        days=1,
    )
    result = worker.execute_node(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert not result.success
    assert doris.calls == 0
    assert result.freshness_reports[0].status == "SKIPPED"


def test_dependent_aggregate_keeps_status_context_columns():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "sub_order_id", "order_id", "spu_name", "pay_amt", "refund_status_name", "refund_create_time", "pt"],
            )
        ]
    )
    intent = QuestionIntent(
        question="退款状态",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="refund_lookup",
        task_role="DEPENDENT",
        preferred_table="dwm_trade_refund_detail_di",
        metric_column="pay_amt",
        metric_name="refund_related_pay_amt",
        group_by_column="sub_order_id",
        required_evidence=["sub_order_id", "pay_amt", "refund_status_name", "refund_create_time"],
        output_keys=["seller_id", "sub_order_id", "order_id", "spu_name"],
        sql_strategy="structured_first",
    )
    sql = worker._draft_sql(
        intent,
        pack,
        "",
        NodeExecutionContext(
            merchant_id="100",
            upstream_entity_sets=[EntitySet(task_id="order_anchor", join_key="sub_order_id", values=["sub_order_id_1"])],
        ),
    )
    assert "`refund_status_name`" in sql
    assert "`refund_create_time`" in sql
    assert "`spu_name`" in sql


def test_dependent_ticket_aggregate_keeps_ticket_status_context():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=[
                    "seller_id",
                    "sub_order_id",
                    "order_id",
                    "spu_id",
                    "spu_name",
                    "ticket_id",
                    "ticket_status_name",
                    "ticket_create_time",
                    "pt",
                ],
            )
        ]
    )
    intent = QuestionIntent(
        question="工单情况",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="ticket_lookup",
        task_role="DEPENDENT",
        preferred_table="dwm_cs_ticket_detail_di",
        group_by_column="sub_order_id",
        required_evidence=["sub_order_id", "ticket_id", "ticket_status_name", "ticket_create_time"],
        output_keys=["seller_id", "sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id"],
        sql_strategy="structured_first",
    )
    sql = worker._draft_sql(
        intent,
        pack,
        "",
        NodeExecutionContext(
            merchant_id="100",
            upstream_entity_sets=[EntitySet(task_id="repay_anchor", join_key="sub_order_id", values=["sub_order_id_1"])],
        ),
    )
    assert "`ticket_status_name`" in sql
    assert "`ticket_create_time`" in sql


def test_evidence_verifier_supports_columns_any_of_and_aliases():
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "high_compensation_orders",
                "table": "dwm_cs_repay_detail_df",
                "columns": ["repay_amt"],
                "columnsAnyOf": [["ticket_id", "sub_order_id", "order_id"]],
                "semanticAliases": {"repay_amt": ["repay_amt", "sum_repay_amt"]},
                "semanticLabel": "high_compensation_orders",
                "requiredLevel": "required",
            }
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="high_compensation_orders",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_cs_repay_detail_df"],
                    rows=[{"sub_order_id": "sub_order_id_1", "sum_repay_amt": "20.50"}],
                ),
            )
        ],
        merged_query_bundle=QueryBundle(rows=[{"sub_order_id": "sub_order_id_1", "sum_repay_amt": "20.50"}]),
    )
    verified = EvidenceVerifier().verify("赔付金额最高订单", plan, run)
    assert verified.passed
    assert not verified.gaps


def test_top_spu_entity_transfer_uses_business_key_not_seller_id():
    dep = PlanDependency(
        anchor_task_id="top_spu_orders",
        dependent_task_id="refund_for_top_spu",
        join_key="seller_id+spu_name",
        anchor_column="seller_id+spu_name",
        dependent_column="seller_id+spu_name",
    )
    key, dependent_key = choose_entity_transfer_key(
        dep,
        [{"seller_id": "100", "spu_name": "spu_name_1"}],
        AgentTaskResult(),
        {"seller_id", "spu_name", "pay_amt"},
    )
    assert key == "spu_name"
    assert dependent_key == "spu_name"


def test_entity_transfer_never_falls_back_to_partition_key_only():
    dep = PlanDependency(
        anchor_task_id="repay_anchor",
        dependent_task_id="ticket_lookup",
        join_key="seller_id+ticket_id",
        anchor_column="seller_id+ticket_id",
        dependent_column="seller_id+ticket_id",
    )
    key, dependent_key = choose_entity_transfer_key(
        dep,
        [{"seller_id": "100", "sub_order_id": "sub_order_id_155"}],
        AgentTaskResult(),
        {"seller_id", "ticket_id", "sub_order_id"},
    )
    assert key == "sub_order_id"
    assert dependent_key == "sub_order_id"


def test_dependent_where_uses_multiple_upstream_entity_columns():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "sub_order_id", "spu_id", "pt", "pay_amt"],
            )
        ]
    )
    intent = QuestionIntent(
        question="看上游商品和日期的退款",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="refund_lookup",
        task_role="DEPENDENT",
        preferred_table="dwm_trade_refund_detail_di",
        group_by_column="spu_id",
        metric_column="pay_amt",
        metric_name="refund_related_pay_amt",
        output_keys=["seller_id", "spu_id", "pt"],
        sql_strategy="structured_first",
    )
    sql = worker._draft_sql(
        intent,
        pack,
        "",
        NodeExecutionContext(
            merchant_id="100",
            upstream_entity_sets=[
                EntitySet(
                    task_id="anchor",
                    join_key="spu_id",
                    values=["spu_id_1"],
                    column_values={"spu_id": ["spu_id_1"], "pt": ["20260620"]},
                )
            ],
        ),
    )
    assert "`spu_id` IN ('spu_id_1')" in sql
    assert "`pt` IN ('20260620')" in sql


def test_node_context_builds_multi_entity_sets_from_parent_rows():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="order_table", columns=["seller_id", "sub_order_id", "spu_id", "pt"]),
            PlanningAssetEntry(table="refund_table", columns=["seller_id", "sub_order_id", "spu_id", "pt"]),
        ]
    )
    plan = QueryPlan(
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="refund_lookup",
                join_key="seller_id+sub_order_id+spu_id+pt",
                anchor_column="seller_id+sub_order_id+spu_id+pt",
                dependent_column="seller_id+sub_order_id+spu_id+pt",
            )
        ]
    )
    completed = {
        "anchor_order": AgentTaskResult(
            task_id="anchor_order",
            success=True,
            query_bundle=QueryBundle(rows=[{"seller_id": "100", "sub_order_id": "sub_1", "spu_id": "spu_1", "pt": "20260620"}]),
        )
    }
    context = worker._node_context(
        "refund_lookup",
        QuestionIntent(plan_task_id="refund_lookup", preferred_table="refund_table"),
        completed,
        plan,
        "100",
        "退款",
        pack,
    )
    assert context.upstream_entity_sets[0].column_values["sub_order_id"] == ["sub_1"]
    assert context.upstream_entity_sets[0].column_values["spu_id"] == ["spu_1"]
    assert context.upstream_entity_sets[0].column_values["pt"] == ["20260620"]


def test_multi_column_relationship_validation_accepts_token_set():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_goods_detail_df", columns=["seller_id", "spu_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "spu_id", "pt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="goods_order_by_spu_id",
                left_table="dwm_goods_detail_df",
                right_table="dwm_trade_order_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="审核拒绝后续订单",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="goods",
                task_role="ANCHOR",
                preferred_table="dwm_goods_detail_df",
            ),
            QuestionIntent(
                question="审核拒绝后续订单",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="orders",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_order_detail_di",
                depends_on_task_ids=["goods"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="goods",
                dependent_task_id="orders",
                join_key="seller_id+spu_id",
                anchor_column="seller_id+spu_id",
                dependent_column="seller_id+spu_id",
            )
        ],
    )
    result = QueryGraphValidator().validate("审核拒绝后续订单", plan, pack)
    assert not any(gap.code == "MISSING_RELATIONSHIP" for gap in result.gaps)


def test_goods_snapshot_lookup_does_not_inherit_question_window():
    hint = table_access_hint("dwm_goods_detail_df", {"seller_id", "spu_id", "pt", "spu_apply_create_time"})
    assert hint["timeWindowPolicy"].startswith("do_not_apply_question_window")


def test_node_worker_repairs_validation_failure():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(tables=[PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id"])])
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="测试",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="node_1",
                preferred_table="dwm_trade_order_detail_di",
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "测试")
    assert result.task_results[0].success
    assert result.sql_repairs
    assert result.sql_repairs[0].success


def test_evidence_verifier_detects_ambiguous_refund_amount():
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["dwm_trade_refund_detail_di"],
            rows=[{"refund_related_pay_amt_raw": "121.50"}],
            original_row_count=1,
        )
    )
    verified = EvidenceVerifier().verify("看退款金额", QueryPlan(final_required_evidence=["refund_related_pay_amt_raw"]), run)
    assert verified.passed
    assert any(gap.code == "FIELD_AMBIGUOUS" for gap in verified.gaps)


def test_evidence_verifier_warns_on_resource_safe_sql_fallback():
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="anchor_gmv",
                success=True,
                query_bundle=QueryBundle(
                    sql="SELECT `seller_id`, `pt`, SUM(`order_gmv_amt_1d`) AS `order_gmv_amt_1d` FROM `ads_merchant_profile` LIMIT 20",
                    tables=["ads_merchant_profile"],
                    rows=[{"seller_id": "100", "pt": "20260620", "order_gmv_amt_1d": 100}],
                ),
                sql_repairs=[
                    SqlRepairAttempt(
                        task_id="anchor_gmv",
                        error_code="MEM_ALLOC_FAILED",
                        error_message="Doris resource error; switched to resource-safe SQL",
                        repaired_sql="SELECT `seller_id`, `pt`, SUM(`order_gmv_amt_1d`) AS `order_gmv_amt_1d` FROM `ads_merchant_profile` LIMIT 20",
                        success=True,
                    )
                ],
            )
        ],
        merged_query_bundle=QueryBundle(rows=[{"seller_id": "100", "pt": "20260620", "order_gmv_amt_1d": 100}]),
    )
    verified = EvidenceVerifier().verify("GMV Top 日期", QueryPlan(), run)
    assert verified.passed
    assert any(gap.code == "RESOURCE_DEGRADED_QUERY" and gap.severity == "warning" for gap in verified.gaps)


def test_evidence_verifier_uses_structured_contracts_without_false_missing():
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "order_detail",
                "table": "dwm_trade_order_detail_di",
                "columns": ["order_id", "sub_order_id", "spu_id"],
                "semanticLabel": "order_detail",
                "requiredLevel": "required",
            },
            {
                "taskId": "refund_for_order",
                "table": "dwm_trade_refund_detail_di",
                "columns": ["sub_order_id", "pay_amt"],
                "semanticLabel": "refund_related_pay_amt",
                "requiredLevel": "required",
            },
            {
                "taskId": "goods_publish_for_order",
                "table": "dwm_goods_detail_df",
                "columns": ["spu_id", "spu_apply_create_time"],
                "semanticLabel": "goods_publish_time",
                "requiredLevel": "required",
            },
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="order_detail",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[{"order_id": "order_id_100", "sub_order_id": "sub_order_id_100", "spu_id": "spu_id_001"}],
                ),
            ),
            AgentTaskResult(
                task_id="refund_for_order",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[{"sub_order_id": "sub_order_id_100", "pay_amt": "121.50"}],
                ),
            ),
            AgentTaskResult(
                task_id="goods_publish_for_order",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[{"spu_id": "spu_id_001", "spu_apply_create_time": "2026-05-05 17:21:00"}],
                ),
            ),
        ],
        merged_query_bundle=QueryBundle(rows=[{"order_id": "order_id_100"}]),
    )
    verified = EvidenceVerifier().verify("看订单、退款和商品发布", plan, run)
    assert verified.passed
    assert not any(gap.code.startswith("MISSING_REQUIRED") for gap in verified.gaps)


def test_evidence_verifier_records_derived_metric_formula_trace():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="refund_rate",
                preferred_table="ads_merchant_profile",
                metric_name="refund_rate_1d",
                metric_column="refund_rate_1d",
                metric_formula="AVG(refund_rate_1d)",
            )
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refund_rate",
                success=True,
                query_bundle=QueryBundle(tables=["ads_merchant_profile"], rows=[{"pt": "20260620", "refund_rate_1d": 0.12}]),
            )
        ],
        merged_query_bundle=QueryBundle(rows=[{"pt": "20260620", "refund_rate_1d": 0.12}]),
    )
    verified = EvidenceVerifier().verify("退款率走势", plan, run)
    assert verified.derived_evidence
    assert verified.derived_evidence[0]["formula"] == "AVG(refund_rate_1d)"
    assert verified.derived_evidence[0]["covered"]


def test_semantic_metric_resolver_maps_refund_amt_alias_to_pay_amt():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "pay_amt", "pt"])],
        metrics=[
            PlanningAssetEntry(
                key="pay_amt",
                table="dwm_trade_refund_detail_di",
                columns=["pay_amt"],
                title="退款金额",
                aliases=["pay_amt", "refund_amt", "退款金额"],
                metadata={"sourceColumns": ["pay_amt"], "formula": "SUM(pay_amt)", "aliases": ["refund_amt", "退款金额"]},
                source_ref_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:pay_amt",
            )
        ],
    )
    resolution = SemanticMetricResolver(pack).resolve(
        "最近30天退款金额最高的商品",
        "refund_amt",
        "dwm_trade_refund_detail_di",
        "退款金额",
    )
    assert resolution.metric
    assert resolution.metric.key == "pay_amt"
    assert resolution.confidence >= 0.7
    assert resolution.payload()["sourceColumns"] == ["pay_amt"]
    assert "pay_amt" in resolution.field_warning


def test_semantic_metric_resolver_uses_runtime_semantic_asset_aliases():
    settings = get_settings()
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        "最近30天退款金额最高的商品",
        recall_bundle_empty(),
        [QuestionCategory.REFUND],
    )
    resolution = SemanticMetricResolver(pack).resolve(
        "最近30天退款金额最高的商品",
        "refund_amt",
        "dwm_trade_refund_detail_di",
        "退款金额",
    )
    assert resolution.metric
    assert resolution.metric.key == "pay_amt"
    assert resolution.confidence >= 0.7


def test_semantic_metric_index_prefers_source_phrase_over_bad_llm_metric_ref():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    index = SemanticMetricIndex(builder._all_metric_entries())
    gmv = index.resolve("pay_amt", "dwm_trade_order_detail_di", "GMV最高的前5天")
    assert gmv
    assert gmv.metric.table == "ads_merchant_profile"
    assert gmv.metric.key == "order_gmv_amt_1d"
    assert gmv.resolution_reason == "semantic_phrase_override"

    refund = index.resolve("pay_amt", "dwm_trade_refund_detail_di", "退款金额")
    assert refund
    assert refund.metric.table == "dwm_trade_refund_detail_di"
    assert refund.metric.key == "pay_amt"
    assert refund.resolution_reason == "semantic_metric_ref"


def test_compiler_corrects_metric_semantic_mismatch_from_source_phrase():
    settings = get_settings()
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )
    understanding = {
        "analysisGrain": "day",
        "analysisIntent": "anomaly_check",
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "GMV最高的前5天",
            "groupByColumn": "pt",
            "order": "desc",
            "limit": 5,
        },
        "requestedMeasures": [
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
            {"metricRef": "ticket_cnt", "ownerTable": "dwm_cs_ticket_detail_di", "sourcePhrase": "工单量"},
        ],
        "timeWindowDays": 30,
    }
    traces = builder.expand_for_question_understanding(pack, understanding)
    assert any("metric_request_table:pay_amt->ads_merchant_profile" in item for item in traces)

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    assert plan.intents
    assert plan.intents[0].preferred_table == "ads_merchant_profile"
    assert plan.intents[0].metric_name == "order_gmv_amt_1d"
    assert plan.intents[0].metric_resolution["resolutionSource"] == "semantic_phrase_override"
    assert plan.intents[0].metric_resolution["candidateScores"][0]["metricKey"] == "order_gmv_amt_1d"
    assert any(item.startswith("METRIC_SEMANTIC_MISMATCH:") for item in plan.compiler_trace)


def test_semantic_metric_resolver_reads_compacted_out_table_metadata_metric():
    settings = get_settings()
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )
    before = SemanticMetricResolver(pack).resolve(
        question,
        "cs_ticket_cnt_1d",
        "ads_merchant_profile",
        "工单量",
    )
    assert not before.metric
    traces = builder.expand_for_question_understanding(
        pack,
        {
            "requestedMeasures": [
                {
                    "metricRef": "cs_ticket_cnt_1d",
                    "ownerTable": "ads_merchant_profile",
                    "sourcePhrase": "工单量",
                }
            ]
        },
    )
    assert any("metric_request_table:cs_ticket_cnt_1d->ads_merchant_profile" in item for item in traces)
    resolution = SemanticMetricResolver(pack).resolve(
        question,
        "cs_ticket_cnt_1d",
        "ads_merchant_profile",
        "工单量",
    )
    assert resolution.metric
    assert resolution.metric.key == "cs_ticket_cnt_1d"
    assert resolution.metric.table == "ads_merchant_profile"
    assert resolution.confidence >= 0.7


def test_planner_expands_asset_pack_from_llm_requested_metric_owner_table():
    class FakeBaselineUnderstandingLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "reason": "product ranking needs store refund-rate baseline",
                "questionUnderstanding": {
                    "analysisGrain": "product",
                    "analysisIntent": "comparison",
                    "requiresExplanation": True,
                    "requiredEvidenceIntents": [
                        {
                            "semanticLabel": "comparison_baseline",
                            "reason": "需要店铺平均退款率作为对比基线",
                            "requiredLevel": "required",
                            "suggestedMetricRefs": ["refund_rate_1d"],
                            "suggestedDomains": ["profile"],
                        }
                    ],
                    "rankingObjective": {
                        "metricRef": "order_detail_cnt",
                        "sourcePhrase": "下单量前20",
                        "ownerTable": "dwm_trade_order_detail_di",
                        "groupByColumn": "spu_id",
                        "order": "desc",
                        "limit": 20,
                    },
                    "requestedMeasures": [
                        {
                            "metricRef": "refund_rate_1d",
                            "sourcePhrase": "店铺平均退款率",
                            "ownerTable": "ads_merchant_profile",
                        }
                    ],
                    "filters": [],
                    "timeWindowDays": 90,
                },
            }

    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    question = "最近90天下单量前20的SPU里，哪些退款率明显高于店铺平均水平？"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    assert "ads_merchant_profile" not in pack.known_tables()
    planner = QueryGraphPlanner(FakeBaselineUnderstandingLlm(), SemanticCatalogService(topic_assets))
    plan, requests, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert "ads_merchant_profile" in pack.known_tables()
    assert any(intent.preferred_table == "ads_merchant_profile" and intent.metric_name == "refund_rate_1d" for intent in plan.intents)
    assert any("metric_request_table:refund_rate_1d->ads_merchant_profile" in item for item in plan.compiler_trace)
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)
    assert not any(issue["code"] == "REQUESTED_MEASURE_NOT_PLANNED" for issue in reflection.issues)


def test_planner_reflection_flags_requested_measure_not_planned():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "spu_id", "pt", "order_id"]),
            PlanningAssetEntry(table="ads_merchant_profile", columns=["seller_id", "pt", "refund_rate_1d"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["order_id"],
                title="下单数",
                source_ref_id="semantic:trade:dwm_trade_order_detail_di:metric:order_detail_cnt",
            ),
            PlanningAssetEntry(
                key="refund_rate_1d",
                table="ads_merchant_profile",
                columns=["refund_rate_1d"],
                title="店铺退款率",
                source_ref_id="semantic:profile:ads_merchant_profile:metric:refund_rate_1d",
            ),
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                metric_column="order_id",
                group_by_column="spu_id",
                output_keys=["seller_id", "spu_id"],
                required_evidence=["spu_id", "order_id"],
                knowledge_refs=[
                    KnowledgeRef(
                        ref_id="semantic:trade:dwm_trade_order_detail_di:metric:order_detail_cnt",
                        ref_type="metric",
                        table="dwm_trade_order_detail_di",
                    )
                ],
            )
        ],
        evidence_contracts=[
            {
                "taskId": "anchor_order",
                "table": "dwm_trade_order_detail_di",
                "columns": ["spu_id", "order_detail_cnt"],
                "semanticLabel": "下单数",
                "requiredLevel": "required",
            }
        ],
        question_understanding={
            "analysisGrain": "product",
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
            },
            "requestedMeasures": [
                {
                    "metricRef": "refund_rate_1d",
                    "ownerTable": "ads_merchant_profile",
                    "sourcePhrase": "店铺平均退款率",
                }
            ],
        },
    )
    reflection = PlannerReflectionAgent().reflect("商品退款率和店铺平均水平对比", plan, pack)
    assert any(issue["code"] == "REQUESTED_MEASURE_NOT_PLANNED" for issue in reflection.issues)
    assert reflection.repair_reason == "METRIC_RESOLUTION_NEEDED"


def test_compiler_repairs_missing_refund_amt_metric_with_semantic_resolution():
    question = "最近30天退款金额最高的商品"
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "pay_amt", "pt"])],
        metrics=[
            PlanningAssetEntry(
                key="pay_amt",
                table="dwm_trade_refund_detail_di",
                columns=["pay_amt"],
                title="退款金额",
                aliases=["pay_amt", "refund_amt", "退款金额"],
                metadata={"sourceColumns": ["pay_amt"], "formula": "SUM(pay_amt)", "aliases": ["refund_amt", "退款金额"]},
                source_ref_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:pay_amt",
            )
        ],
    )
    plan = QuestionUnderstandingCompiler().compile(
        question,
        {
            "analysisGrain": "product",
            "rankingObjective": {
                "metricRef": "refund_amt",
                "sourcePhrase": "退款金额",
                "ownerTable": "dwm_trade_refund_detail_di",
                "groupByColumn": "spu_name",
                "limit": 10,
            },
            "requestedMeasures": [],
            "timeWindowDays": 30,
        },
        pack,
    )
    assert plan.intents
    assert plan.intents[0].metric_name == "pay_amt"
    assert plan.intents[0].metric_column == "pay_amt"
    assert plan.intents[0].metric_resolution["metricKey"] == "pay_amt"
    assert plan.intents[0].metric_resolution["requestedMetricRef"] == "refund_amt"
    assert "UNKNOWN_METRIC_REF" not in ",".join(plan.compiler_trace)
    assert plan.evidence_contracts[0]["columns"] == ["spu_name", "refund_related_pay_amt"]
    assert plan.evidence_contracts[0]["metricResolution"]["sourceColumns"] == ["pay_amt"]


def test_evidence_verifier_accepts_resolved_refund_metric_without_ambiguous_gap():
    field_warning = "退款金额按 dwm_trade_refund_detail_di.pay_amt 统计，表示退款明细关联订单的支付金额口径。"
    metric_resolution = {
        "requestedMetricRef": "refund_amt",
        "metricKey": "pay_amt",
        "ownerTable": "dwm_trade_refund_detail_di",
        "sourceColumns": ["pay_amt"],
        "formula": "SUM(pay_amt)",
        "displayName": "退款金额",
        "confidence": 0.95,
        "resolutionSource": "semantic_alias",
        "fieldWarning": field_warning,
    }
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "refund_top",
                "table": "dwm_trade_refund_detail_di",
                "columns": ["spu_name", "refund_related_pay_amt"],
                "semanticLabel": "refund_related_pay_amt",
                "requiredLevel": "required",
                "semanticAliases": {"refund_related_pay_amt": ["pay_amt", "sum_pay_amt"]},
                "metricResolution": metric_resolution,
            }
        ],
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"spu_name": "A", "pay_amt": "121.50"}]),
            )
        ],
        merged_query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"spu_name": "A", "pay_amt": "121.50"}]),
    )
    verified = EvidenceVerifier().verify("看退款金额", plan, run)
    assert verified.passed
    assert not any(gap.code == "FIELD_AMBIGUOUS" for gap in verified.gaps)
    assert verified.required_disclosures == [field_warning]
    assert verified.answer_guard_required


def test_evidence_verifier_marks_low_confidence_metric_resolution_as_warning():
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "refund_top",
                "table": "dwm_trade_refund_detail_di",
                "columns": ["pay_amt"],
                "semanticLabel": "refund_metric",
                "requiredLevel": "required",
                "metricResolution": {
                    "requestedMetricRef": "unknown_amt",
                    "metricKey": "pay_amt",
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "confidence": 0.4,
                    "resolutionSource": "semantic_weak_match",
                    "fieldWarning": "指标由弱匹配得到，口径需要人工确认",
                },
            }
        ],
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"pay_amt": "121.50"}]),
            )
        ],
        merged_query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"pay_amt": "121.50"}]),
    )
    verified = EvidenceVerifier().verify("看售后金额", plan, run)
    assert verified.passed
    assert any(gap.code == "FIELD_AMBIGUOUS" and gap.severity == "warning" for gap in verified.gaps)


def test_answer_guard_appends_required_metric_resolution_disclosure():
    class FakeAnswerLlm:
        configured = True
        settings = get_settings()

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            return "已查询到退款商品数据。"

    disclosure = "退款金额按 dwm_trade_refund_detail_di.pay_amt 统计，表示退款明细关联订单的支付金额口径。"
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"spu_name": "A", "pay_amt": "121.50"}]),
        task_results=[
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"spu_name": "A", "pay_amt": "121.50"}]),
            )
        ],
        verified_evidence=VerifiedEvidence(
            passed=True,
            answer_guard_required=True,
            required_disclosures=[disclosure],
        ),
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="看退款金额",
                intent_type="VALID",
                answer_mode="TOPN",
                category=QuestionCategory.REFUND,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
            )
        ]
    )
    answer = AnswerComposeService(FakeAnswerLlm()).compose("看退款金额", MerchantInfo(merchant_id="100"), plan, run, "")
    assert "证据门禁" in answer
    assert disclosure in answer


def test_answer_analysis_summary_uses_structured_analysis_intent_not_question_terms():
    class FakeAnalysisLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            return "分析摘要"

    llm = FakeAnalysisLlm()
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["ads_merchant_profile"],
            rows=[{"seller_id": "100", "pt": "20260620", "order_gmv_amt_1d": 100}],
        )
    )
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "anomaly_check",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [{"semanticLabel": "gmv_anomaly", "requiredLevel": "required"}],
        }
    )
    summary = AnswerComposeService(llm).summarize_analysis("最近30天GMV是否正常？", plan, run)
    assert summary == "分析摘要"
    assert llm.calls == 1


def test_answer_analysis_summary_skips_when_structured_intent_is_none():
    class FakeAnalysisLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            return "不应该调用"

    llm = FakeAnalysisLlm()
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["ads_merchant_profile"],
            rows=[{"seller_id": "100", "pt": "20260620", "order_gmv_amt_1d": 100}],
        )
    )
    plan = QueryPlan(question_understanding={"analysisIntent": "none", "requiresExplanation": False})
    summary = AnswerComposeService(llm).summarize_analysis("帮我分析一下GMV Top5", plan, run)
    assert summary == ""
    assert llm.calls == 0


def recall_bundle_empty():
    from merchant_ai.models import RecallBundle

    return RecallBundle()


def profile_daily_pack():
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="ads_merchant_profile",
                columns=[
                    "seller_id",
                    "pt",
                    "order_gmv_amt_1d",
                    "refund_amt_1d",
                    "seller_repay_amt_1d",
                    "cs_ticket_cnt_1d",
                ],
            ),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pt", "sub_order_id", "pay_amt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pt", "refund_id", "pay_amt"]),
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "pt", "bill_id", "repay_amt"]),
            PlanningAssetEntry(table="dwm_cs_ticket_detail_di", columns=["seller_id", "pt", "ticket_id"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_gmv_amt_1d",
                table="ads_merchant_profile",
                columns=["order_gmv_amt_1d"],
                title="总gmv金额元",
                aliases=["GMV", "总GMV", "成交额"],
                metadata={"sourceColumns": ["order_gmv_amt_1d"], "formula": "SUM(order_gmv_amt_1d)"},
            ),
            PlanningAssetEntry(
                key="net_gmv_after_refund",
                table="ads_merchant_profile",
                columns=["order_gmv_amt_1d", "refund_amt_1d"],
                title="扣退款后 GMV",
                aliases=["扣退款后 GMV"],
                metadata={"sourceColumns": ["order_gmv_amt_1d", "refund_amt_1d"], "formula": "SUM(order_gmv_amt_1d) - SUM(refund_amt_1d)"},
            ),
            PlanningAssetEntry(
                key="refund_amt_1d",
                table="ads_merchant_profile",
                columns=["refund_amt_1d"],
                title="退款金额元",
                aliases=["退款金额"],
                metadata={"sourceColumns": ["refund_amt_1d"], "formula": "SUM(refund_amt_1d)"},
            ),
            PlanningAssetEntry(
                key="seller_repay_amt_1d",
                table="ads_merchant_profile",
                columns=["seller_repay_amt_1d"],
                title="卖家赔付金额元",
                aliases=["赔付金额"],
                metadata={"sourceColumns": ["seller_repay_amt_1d"], "formula": "SUM(seller_repay_amt_1d)"},
            ),
            PlanningAssetEntry(
                key="cs_ticket_cnt_1d",
                table="ads_merchant_profile",
                columns=["cs_ticket_cnt_1d"],
                title="咨询工单量",
                aliases=["工单量"],
                metadata={"sourceColumns": ["cs_ticket_cnt_1d"], "formula": "SUM(cs_ticket_cnt_1d)"},
            ),
        ],
    )


class FakeLlm:
    configured = True

    def __init__(self):
        self.calls = 0
        self.last_error = ""

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        if self.calls == 1:
            return {"sql": "SELECT * FROM outside_table"}
        return {"sql": "SELECT `seller_id` FROM `dwm_trade_order_detail_di` LIMIT 1"}


class GoodPlanBoundSqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        payload = json.loads(user_prompt)
        contract = payload["nodePlanContract"]
        table = contract["preferredTable"]
        metric = contract.get("metricColumn") or "order_gmv_amt_1d"
        group_by = contract.get("groupByColumn") or "seller_id"
        merchant_column = contract.get("merchantFilterColumn") or "seller_id"
        days = int(contract.get("days") or 7)
        limit = int(contract.get("limit") or 20)
        output_keys = [column for column in contract.get("outputKeys", []) if column]
        select_keys = output_keys or [merchant_column, group_by]
        select_keys = [column for index, column in enumerate(select_keys) if column and column not in select_keys[:index]]
        sql = (
            "SELECT %s, SUM(`%s`) AS `%s` FROM `%s` "
            "WHERE `%s` = '100' AND `pt` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL %d DAY), '%%Y%%m%%d') "
            "GROUP BY %s ORDER BY `%s` DESC LIMIT %d"
            % (
                ", ".join("`%s`" % column for column in select_keys),
                metric,
                metric,
                table,
                merchant_column,
                days,
                ", ".join("`%s`" % column for column in select_keys),
                metric,
                limit,
            )
        )
        return {"sql": sql, "reason": "plan-bound test SQL"}


class BadContractColumnSqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {
            "sql": (
                "SELECT `seller_id`, `pt`, SUM(`refund_amt_1d`) AS `refund_amt_1d` FROM `ads_merchant_profile` "
                "WHERE `seller_id` = '100' AND `pt` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 30 DAY), '%Y%m%d') "
                "GROUP BY `seller_id`, `pt` ORDER BY `refund_amt_1d` DESC LIMIT 5"
            )
        }


class MissingOutputKeySqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {
            "sql": (
                "SELECT `seller_id`, `spu_id`, SUM(`pay_amt`) AS `order_gmv` FROM `dwm_trade_order_detail_di` "
                "WHERE `seller_id` = '100' AND `pt` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 30 DAY), '%Y%m%d') "
                "GROUP BY `seller_id`, `spu_id` ORDER BY `order_gmv` DESC LIMIT 5"
            )
        }


class InvalidPartitionSqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {
            "sql": (
                "SELECT `seller_id`, `pt`, SUM(`order_gmv_amt_1d`) AS `order_gmv_amt_1d` FROM `ads_merchant_profile` "
                "WHERE `seller_id` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
                "GROUP BY `seller_id`, `pt` ORDER BY `order_gmv_amt_1d` DESC LIMIT 5"
            )
        }


class EmptySqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {}


class FakePlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "PLAN_READY",
            "reason": "llm understood ranking objective",
            "questionUnderstanding": {
                "analysisGrain": "product",
                "rankingObjective": {
                    "metricRef": "ticket_cnt",
                    "sourcePhrase": "客服工单量最高的前10个商品",
                    "ownerTable": "dwm_cs_ticket_detail_di",
                    "groupByColumn": "spu_id",
                    "order": "desc",
                    "limit": 10,
                },
                "requestedMeasures": [
                    {"metricRef": "refund_cnt", "sourcePhrase": "退款量", "ownerTable": "dwm_trade_refund_detail_di"},
                    {"metricRef": "repay_amt", "sourcePhrase": "赔付金额", "ownerTable": "dwm_cs_repay_detail_df"},
                ],
                "timeWindowDays": 60,
            },
            "queryPlan": {
                "intents": [
                    {
                        "question": "客服工单量最高商品",
                        "intentType": "VALID",
                        "category": "CS_TICKET",
                        "answerMode": "TOPN",
                        "planTaskId": "anchor_ticket",
                        "taskRole": "ANCHOR",
                        "preferredTable": "dwm_cs_ticket_detail_di",
                        "groupByColumn": "spu_id",
                        "days": 60,
                        "limit": 10,
                        "requiredEvidence": ["spu_id", "ticket_id"],
                        "outputKeys": ["seller_id", "spu_id", "sub_order_id"],
                    },
                    {
                        "question": "退款量",
                        "intentType": "VALID",
                        "category": "REFUND",
                        "answerMode": "GROUP_AGG",
                        "planTaskId": "refund_lookup",
                        "taskRole": "DEPENDENT",
                        "preferredTable": "dwm_trade_refund_detail_di",
                        "groupByColumn": "spu_name",
                        "days": 60,
                        "limit": 10,
                        "requiredEvidence": ["spu_name", "refund_id", "pay_amt"],
                        "outputKeys": ["seller_id", "sub_order_id", "spu_name"],
                        "dependsOnTaskIds": ["anchor_ticket"],
                    },
                    {
                        "question": "赔付金额",
                        "intentType": "VALID",
                        "category": "COMPENSATION",
                        "answerMode": "GROUP_AGG",
                        "planTaskId": "repay_lookup",
                        "taskRole": "DEPENDENT",
                        "preferredTable": "dwm_cs_repay_detail_df",
                        "metricColumn": "repay_amt",
                        "metricName": "repay_amt",
                        "groupByColumn": "sub_order_id",
                        "days": 60,
                        "limit": 10,
                        "requiredEvidence": ["sub_order_id", "repay_amt"],
                        "outputKeys": ["seller_id", "sub_order_id"],
                        "dependsOnTaskIds": ["anchor_ticket"],
                    },
                ],
                "dependencies": [
                    {
                        "anchorTaskId": "anchor_ticket",
                        "dependentTaskId": "refund_lookup",
                        "joinKey": "sub_order_id",
                        "anchorColumn": "sub_order_id",
                        "dependentColumn": "sub_order_id",
                    },
                    {
                        "anchorTaskId": "anchor_ticket",
                        "dependentTaskId": "repay_lookup",
                        "joinKey": "sub_order_id",
                        "anchorColumn": "sub_order_id",
                        "dependentColumn": "sub_order_id",
                    },
                ],
                "finalRequiredEvidence": ["ticket_cnt", "refund_cnt", "repay_amt"],
            },
        }


class SemanticToolLoopPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = []
        self.fast_calls = []

    def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None):
        payload = json.loads(user_prompt)
        self.fast_calls.append(payload)
        return {
            "status": "NEED_MORE_KNOWLEDGE",
            "reason": "need table asset detail before understanding",
            "knowledgeRequests": [
                {
                    "requestType": "TABLE",
                    "query": "dwm_trade_order_detail_di semantic asset",
                    "reason": "need order metric details",
                }
            ],
        }

    def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, timeout_seconds=None, tool_choice=None):
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        if len(self.calls) == 1:
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_read_order_asset",
                        "name": "semantic_read",
                        "args": {
                            "refId": "semantic:电商交易:dwm_trade_order_detail_di:asset",
                            "maxChars": 2000,
                            "reason": "need order metric details",
                        },
                    }
                ],
            }
        return {
            "content": "",
            "toolCalls": [
                {
                    "id": "call_emit_understanding",
                    "name": "emit_question_understanding",
                    "args": {
                        "status": "UNDERSTOOD",
                        "reason": "understood after reading semantic asset",
                        "questionUnderstanding": {
                            "analysisGrain": "product",
                            "analysisIntent": "none",
                            "requiresExplanation": False,
                            "requiredEvidenceIntents": [],
                            "rankingObjective": {
                                "metricRef": "order_detail_cnt",
                                "sourcePhrase": "下单最多",
                                "ownerTable": "dwm_trade_order_detail_di",
                                "groupByColumn": "spu_id",
                                "order": "desc",
                                "limit": 5,
                            },
                            "requestedMeasures": [
                                {
                                    "metricRef": "refund_bill_cnt",
                                    "sourcePhrase": "退款量",
                                    "ownerTable": "dwm_trade_refund_detail_di",
                                }
                            ],
                            "filters": [],
                            "timeWindowDays": 90,
                        },
                    },
                }
            ],
        }


class FastPathPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.tool_json_calls = []
        self.tool_chat_calls = 0

    def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None):
        payload = json.loads(user_prompt)
        self.tool_json_calls.append(payload)
        return {
            "status": "UNDERSTOOD",
            "reason": "understood from compact catalog",
            "questionUnderstanding": {
                "analysisGrain": "product",
                "analysisIntent": "none",
                "requiresExplanation": False,
                "requiredEvidenceIntents": [],
                "rankingObjective": {
                    "metricRef": "order_detail_cnt",
                    "sourcePhrase": "下单最多",
                    "ownerTable": "dwm_trade_order_detail_di",
                    "groupByColumn": "spu_id",
                    "order": "desc",
                    "limit": 5,
                },
                "requestedMeasures": [
                    {
                        "metricRef": "refund_bill_cnt",
                        "sourcePhrase": "退款量",
                        "ownerTable": "dwm_trade_refund_detail_di",
                    }
                ],
                "filters": [],
                "timeWindowDays": 90,
            },
        }

    def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, timeout_seconds=None, tool_choice=None):
        self.tool_chat_calls += 1
        return {"content": "", "toolCalls": []}


class RefiningAnalysisPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.tool_json_payloads = []
        self.tool_chat_payloads = []

    def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None):
        payload = json.loads(user_prompt)
        self.tool_json_payloads.append(payload)
        return {
            "status": "UNDERSTOOD",
            "reason": "fast path used compact catalog before reading daily profile metrics",
            "questionUnderstanding": {
                "analysisGrain": "day",
                "analysisIntent": "anomaly_check",
                "requiresExplanation": True,
                "requiredEvidenceIntents": [
                    {
                        "semanticLabel": "trend_context",
                        "reason": "need daily trend context before judging anomaly",
                        "requiredLevel": "required",
                        "suggestedMetricRefs": ["pay_amt"],
                        "suggestedDomains": ["trade", "refund", "ticket", "compensation"],
                    }
                ],
                "rankingObjective": {
                    "metricRef": "pay_amt",
                    "sourcePhrase": "GMV最高的前5天",
                    "ownerTable": "dwm_trade_order_detail_di",
                    "groupByColumn": "pt",
                    "order": "desc",
                    "limit": 5,
                },
                "requestedMeasures": [
                    {"metricRef": "pay_amt", "sourcePhrase": "退款金额", "ownerTable": "dwm_trade_refund_detail_di"},
                    {"metricRef": "ticket_cnt", "sourcePhrase": "工单量", "ownerTable": "dwm_cs_ticket_detail_di"},
                ],
                "filters": [],
                "timeWindowDays": 30,
            },
        }

    def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, timeout_seconds=None, tool_choice=None):
        payload = json.loads(user_prompt)
        self.tool_chat_payloads.append(payload)
        if len(self.tool_chat_payloads) == 1:
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_grep_profile_metrics",
                        "name": "semantic_grep",
                        "args": {"query": "GMV 退款金额 赔付金额 工单量 日指标", "limit": 10},
                    }
                ],
            }
        return {
            "content": "",
            "toolCalls": [
                {
                    "id": "call_emit_refined_understanding",
                    "name": "emit_question_understanding",
                    "args": {
                        "status": "UNDERSTOOD",
                        "reason": "refined after reading semantic catalog",
                        "questionUnderstanding": {
                            "analysisGrain": "day",
                            "analysisIntent": "anomaly_check",
                            "requiresExplanation": True,
                            "requiredEvidenceIntents": [
                                {
                                    "semanticLabel": "trend_context",
                                    "reason": "daily KPI trend context is required for anomaly analysis",
                                    "requiredLevel": "required",
                                    "suggestedMetricRefs": [
                                        "order_gmv_amt_1d",
                                        "refund_amt_1d",
                                        "seller_repay_amt_1d",
                                        "cs_ticket_cnt_1d",
                                    ],
                                    "suggestedDomains": ["profile"],
                                }
                            ],
                            "rankingObjective": {
                                "metricRef": "order_gmv_amt_1d",
                                "sourcePhrase": "GMV最高的前5天",
                                "ownerTable": "ads_merchant_profile",
                                "groupByColumn": "pt",
                                "order": "desc",
                                "limit": 5,
                            },
                            "requestedMeasures": [
                                {
                                    "metricRef": "refund_amt_1d",
                                    "sourcePhrase": "退款金额",
                                    "ownerTable": "ads_merchant_profile",
                                },
                                {
                                    "metricRef": "seller_repay_amt_1d",
                                    "sourcePhrase": "赔付金额",
                                    "ownerTable": "ads_merchant_profile",
                                },
                                {
                                    "metricRef": "cs_ticket_cnt_1d",
                                    "sourcePhrase": "工单量",
                                    "ownerTable": "ads_merchant_profile",
                                },
                            ],
                            "filters": [],
                            "timeWindowDays": 30,
                        },
                    },
                }
            ],
        }


class FakeCaseUnderstandingLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        payload = json.loads(user_prompt)
        question = payload.get("question", "")
        return {
            "status": "PLAN_READY",
            "reason": "fake LLM supplied questionUnderstanding",
            "questionUnderstanding": fake_question_understanding(question, payload.get("semanticCatalog") or {}),
        }


class FakeTimeoutThenCompactUnderstandingLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0
        self.catalog_sizes = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        payload = json.loads(user_prompt)
        catalog = payload.get("semanticCatalog") or {}
        self.catalog_sizes.append(len(json.dumps(catalog, ensure_ascii=False)))
        if self.calls == 1:
            self.last_error = "timeout: provider call exceeded 20 seconds"
            return {}
        self.last_error = ""
        metrics = catalog.get("candidateMetrics") or []
        tables = catalog.get("tables") or []
        order_metric = next((item for item in metrics if item.get("table") == "dwm_trade_order_detail_di"), metrics[0])
        refund_metric = next((item for item in metrics if item.get("table") == "dwm_trade_refund_detail_di"), None)
        table_columns = {item.get("table"): item.get("keyColumns") or [] for item in tables}
        group_by = "spu_id" if "spu_id" in table_columns.get(order_metric.get("table"), []) else "spu_name"
        return {
            "status": "UNDERSTOOD",
            "reason": "compact retry understood top spu with refund measures",
            "questionUnderstanding": {
                "analysisGrain": "product",
                "rankingObjective": {
                    "metricRef": order_metric.get("key"),
                    "sourcePhrase": "下单最多的前 5 个 SPU",
                    "ownerTable": order_metric.get("table"),
                    "groupByColumn": group_by,
                    "order": "desc",
                    "limit": 5,
                },
                "requestedMeasures": [
                    {
                        "metricRef": refund_metric.get("key"),
                        "sourcePhrase": "退款量和退款金额",
                        "ownerTable": refund_metric.get("table"),
                    }
                ]
                if refund_metric
                else [],
                "timeWindowDays": 7,
            },
        }


def fake_question_understanding(question, catalog=None):
    catalog = catalog or {}
    if "GMV" in question or "gmv" in question or "走势" in question or "波动" in question:
        profile_table = "ads_merchant_profile"
        if "GMV" in question or "gmv" in question:
            ranking = catalog_metric_ref(
                catalog,
                ["order_gmv_amt_1d", "pay_gmv_amt_1d", "trade_success_gmv_amt_1d"],
                profile_table,
                "GMV最高",
                "pt",
                5,
                fallback=("order_gmv_amt_1d", profile_table),
            )
            measures = catalog_measure_refs(
                catalog,
                [
                    ("refund_amt_1d", profile_table, "退款金额"),
                    ("seller_repay_amt_1d", profile_table, "赔付金额"),
                    ("cs_ticket_cnt_1d", profile_table, "工单量"),
                ],
            )
        else:
            ranking = catalog_metric_ref(
                catalog,
                ["refund_rate_1d", "seller_repay_rate_1d", "cs_ticket_rate_1d", "refund_amt_1d", "seller_repay_amt_1d", "cs_ticket_cnt_1d"],
                profile_table,
                "趋势指标",
                "pt",
                5,
                fallback=("refund_rate_1d", profile_table),
            )
            measures = catalog_measure_refs(
                catalog,
                [
                    ("refund_rate_1d", profile_table, "退款率"),
                    ("seller_repay_rate_1d", profile_table, "赔付率"),
                    ("cs_ticket_rate_1d", profile_table, "工单率"),
                    ("refund_amt_1d", profile_table, "退款金额"),
                    ("seller_repay_amt_1d", profile_table, "赔付金额"),
                    ("cs_ticket_cnt_1d", profile_table, "工单量"),
                ],
                exclude={(ranking.get("metricRef"), ranking.get("ownerTable"))},
            )
        return {
            "analysisGrain": "day",
            "rankingObjective": ranking,
            "requestedMeasures": measures,
            "timeWindowDays": days_from_question(question, 30),
        }
    if "客服工单量" in question or "工单量最高" in question or "有客服工单" in question:
        return {
            "analysisGrain": "product" if "商品" in question or "SPU" in question else "order",
            "rankingObjective": metric_ref("ticket_cnt", "dwm_cs_ticket_detail_di", "工单量", "spu_id" if "商品" in question or "SPU" in question else "sub_order_id", 10),
            "requestedMeasures": [
                measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款量"),
                measure_ref("repay_amt", "dwm_cs_repay_detail_df", "赔付金额"),
            ],
            "timeWindowDays": days_from_question(question, 30),
        }
    if "赔付单量" in question:
        return {
            "analysisGrain": "product" if "商品" in question else "order",
            "rankingObjective": metric_ref("repay_bill_cnt", "dwm_cs_repay_detail_df", "赔付单量", "spu_id" if "商品" in question else "sub_order_id", 5),
            "requestedMeasures": [
                measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款量"),
                measure_ref("pay_amt", "dwm_trade_refund_detail_di", "退款金额"),
                measure_ref("goods_cnt", "dwm_goods_detail_df", "商品发布时间"),
            ],
            "timeWindowDays": days_from_question(question, 60),
        }
    if "赔付" in question:
        return {
            "analysisGrain": "order",
            "rankingObjective": metric_ref("repay_amt", "dwm_cs_repay_detail_df", "赔付金额", "sub_order_id", 5),
            "requestedMeasures": [
                measure_ref("pay_amt", "dwm_trade_order_detail_di", "订单金额"),
                measure_ref("pay_amt", "dwm_trade_refund_detail_di", "退款金额"),
                measure_ref("ticket_cnt", "dwm_cs_ticket_detail_di", "客服工单"),
            ],
            "timeWindowDays": days_from_question(question, 60),
        }
    if "供应链" in question or "入库" in question:
        return {
            "analysisGrain": "product",
            "rankingObjective": metric_ref("inbound_cnt", "dwm_scm_detail_di", "入库量", "spu_id", 10),
            "requestedMeasures": [
                measure_ref("order_detail_cnt", "dwm_trade_order_detail_di", "下单表现"),
                measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款表现"),
            ],
            "timeWindowDays": days_from_question(question, 45),
        }
    if "优惠券" in question or "券" in question:
        return {
            "analysisGrain": "product" if "商品" in question else "coupon",
            "rankingObjective": metric_ref("coupon_amt", "dwm_coupon_detail_di", "券金额投入", "coupon_id", 10),
            "requestedMeasures": [
                measure_ref("order_detail_cnt", "dwm_trade_order_detail_di", "下单"),
                measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款"),
            ],
            "timeWindowDays": days_from_question(question, 30),
        }
    if "审核" in question or "发布" in question or "新品" in question:
        return {
            "analysisGrain": "product",
            "rankingObjective": metric_ref("goods_audit_reject_detail_cnt", "dwm_goods_detail_df", "审核拒绝商品", "spu_id", 10),
            "requestedMeasures": [
                measure_ref("order_detail_cnt", "dwm_trade_order_detail_di", "订单"),
                measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款"),
                measure_ref("repay_amt", "dwm_cs_repay_detail_df", "赔付"),
            ],
            "timeWindowDays": days_from_question(question, 30),
        }
    if "退款金额" in question:
        return {
            "analysisGrain": "product" if "商品" in question or "SPU" in question else "order",
            "rankingObjective": metric_ref("pay_amt", "dwm_trade_refund_detail_di", "退款金额", "spu_name" if "商品" in question else "sub_order_id", 10),
            "requestedMeasures": [
                measure_ref("order_detail_cnt", "dwm_trade_order_detail_di", "下单量"),
                measure_ref("goods_cnt", "dwm_goods_detail_df", "商品发布时间"),
                measure_ref("repay_amt", "dwm_cs_repay_detail_df", "赔付"),
            ],
            "timeWindowDays": days_from_question(question, 30),
        }
    if "退款" in question or "退货" in question:
        return {
            "analysisGrain": "product" if "商品" in question or "SPU" in question else "order",
            "rankingObjective": metric_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款量", "spu_name" if "商品" in question else "sub_order_id", 10),
            "requestedMeasures": [
                measure_ref("order_detail_cnt", "dwm_trade_order_detail_di", "下单"),
                measure_ref("pay_amt", "dwm_trade_refund_detail_di", "退款金额"),
                measure_ref("goods_cnt", "dwm_goods_detail_df", "商品发布时间"),
            ],
            "timeWindowDays": days_from_question(question, 30),
        }
    return {
        "analysisGrain": "product" if "商品" in question or "SPU" in question else "order",
        "rankingObjective": metric_ref("order_detail_cnt", "dwm_trade_order_detail_di", "下单量", "spu_id" if "商品" in question or "SPU" in question else "sub_order_id", 10),
        "requestedMeasures": [
            measure_ref("refund_bill_cnt", "dwm_trade_refund_detail_di", "退款"),
            measure_ref("repay_amt", "dwm_cs_repay_detail_df", "赔付"),
            measure_ref("ticket_cnt", "dwm_cs_ticket_detail_di", "工单"),
        ],
        "timeWindowDays": days_from_question(question, 30),
    }


def metric_ref(metric, table, phrase, group_by, limit):
    return {
        "metricRef": metric,
        "sourcePhrase": phrase,
        "ownerTable": table,
        "groupByColumn": group_by,
        "order": "desc",
        "limit": limit,
    }


def measure_ref(metric, table, phrase):
    return {"metricRef": metric, "ownerTable": table, "sourcePhrase": phrase}


def catalog_metric_ref(catalog, keys, table, phrase, group_by, limit, fallback):
    item = catalog_metric_item(catalog, keys, table)
    metric = item.get("key") if item else fallback[0]
    owner_table = item.get("table") if item else fallback[1]
    return metric_ref(metric, owner_table, phrase, group_by, limit)


def catalog_measure_refs(catalog, candidates, exclude=None):
    exclude = exclude or set()
    measures = []
    for metric, table, phrase in candidates:
        item = catalog_metric_item(catalog, [metric], table)
        if not item:
            continue
        identity = (item.get("key"), item.get("table"))
        if identity in exclude:
            continue
        measures.append(measure_ref(str(item.get("key") or ""), str(item.get("table") or ""), phrase))
    return measures


def catalog_metric_item(catalog, keys, table):
    metrics = catalog.get("candidateMetrics") or []
    for key in keys:
        for item in metrics:
            if item.get("key") == key and item.get("table") == table:
                return item
    return None


def days_from_question(question, default):
    for days in [90, 60, 45, 30, 15, 7]:
        if str(days) in question:
            return days
    return default


class FakeDailyProfilePlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "PLAN_READY",
            "reason": "llm understood daily KPI ranking objective",
            "questionUnderstanding": {
                "analysisGrain": "day",
                "rankingObjective": {
                    "metricRef": "order_gmv_amt_1d",
                    "sourcePhrase": "GMV最高的前5天",
                    "ownerTable": "ads_merchant_profile",
                    "groupByColumn": "pt",
                    "order": "desc",
                    "limit": 5,
                },
                "requestedMeasures": [
                    {"metricRef": "refund_amt_1d", "sourcePhrase": "退款金额", "ownerTable": "ads_merchant_profile"},
                    {"metricRef": "seller_repay_amt_1d", "sourcePhrase": "赔付金额", "ownerTable": "ads_merchant_profile"},
                    {"metricRef": "cs_ticket_cnt_1d", "sourcePhrase": "工单量", "ownerTable": "ads_merchant_profile"},
                ],
                "timeWindowDays": 30,
            },
        }


class FakeCouponRefundRatePlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "UNDERSTOOD",
            "reason": "coupon investment with refund rate understanding",
            "questionUnderstanding": {
                "analysisGrain": "product",
                "analysisIntent": "comparison",
                "requiresExplanation": True,
                "requiredEvidenceIntents": [
                    {
                        "semanticLabel": "comparison_baseline",
                        "reason": "退款率判断需要订单量分母和优惠券投入指标",
                        "requiredLevel": "required",
                        "suggestedMetricRefs": ["refund_rate", "order_detail_cnt", "coupon_amt"],
                        "suggestedDomains": ["coupon", "refund", "trade"],
                    }
                ],
                "rankingObjective": {
                    "metricRef": "coupon_amt",
                    "sourcePhrase": "优惠券金额投入最高",
                    "ownerTable": "dwm_coupon_detail_di",
                    "groupByColumn": "coupon_id",
                    "order": "desc",
                    "limit": 10,
                },
                "requestedMeasures": [
                    {"metricRef": "refund_rate", "sourcePhrase": "退款率", "ownerTable": "dwm_trade_refund_detail_di"}
                ],
                "filters": [],
                "timeWindowDays": 45,
            },
        }


class FakeDiagnosticContextPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.payloads = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        payload = json.loads(user_prompt)
        self.payloads.append(payload)
        return {
            "status": "UNDERSTOOD",
            "reason": "diagnostic context guided overview understanding",
            "questionUnderstanding": {
                "analysisGrain": "day",
                "analysisIntent": "overview",
                "requiresExplanation": True,
                "requiredEvidenceIntents": [
                    {
                        "semanticLabel": "risk_driver",
                        "reason": "经营健康度需要 GMV、退款、赔付和工单趋势证据",
                        "requiredLevel": "required",
                        "suggestedMetricRefs": ["order_gmv_amt_1d", "refund_amt_1d", "seller_repay_amt_1d", "cs_ticket_cnt_1d"],
                        "suggestedDomains": ["trade", "refund", "compensation", "ticket"],
                    }
                ],
                "rankingObjective": {
                    "metricRef": "order_gmv_amt_1d",
                    "sourcePhrase": "整体经营情况",
                    "ownerTable": "ads_merchant_profile",
                    "groupByColumn": "pt",
                    "order": "desc",
                    "limit": 30,
                },
                "requestedMeasures": [
                    {"metricRef": "refund_amt_1d", "sourcePhrase": "退款风险", "ownerTable": "ads_merchant_profile"},
                    {"metricRef": "seller_repay_amt_1d", "sourcePhrase": "赔付风险", "ownerTable": "ads_merchant_profile"},
                    {"metricRef": "cs_ticket_cnt_1d", "sourcePhrase": "工单风险", "ownerTable": "ads_merchant_profile"},
                ],
                "filters": [],
                "timeWindowDays": 90,
            },
        }


class FakeDepositGmvPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "UNDERSTOOD",
            "reason": "deposit and GMV comparison understanding",
            "questionUnderstanding": {
                "analysisGrain": "day",
                "analysisIntent": "comparison",
                "requiresExplanation": True,
                "requiredEvidenceIntents": [
                    {
                        "semanticLabel": "comparison_baseline",
                        "reason": "相关性判断需要保证金充值金额与 GMV 日趋势同时覆盖",
                        "requiredLevel": "required",
                        "suggestedMetricRefs": ["deposit_recharge_amt", "order_gmv_amt_1d"],
                        "suggestedDomains": ["merchant_other", "trade"],
                    }
                ],
                "rankingObjective": {
                    "metricRef": "deposit_recharge_amt",
                    "sourcePhrase": "保证金充值金额变化",
                    "ownerTable": "dwd_merchant_deposit_recharge_df",
                    "groupByColumn": "pt",
                    "order": "desc",
                    "limit": 180,
                },
                "requestedMeasures": [
                    {"metricRef": "order_gmv_amt_1d", "sourcePhrase": "GMV变化", "ownerTable": "ads_merchant_profile"}
                ],
                "filters": [],
                "timeWindowDays": 180,
            },
        }


class FakeNeedMoreThenProfilePlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "status": "NEED_MORE_KNOWLEDGE",
                "reason": "need more knowledge even though catalog is enough",
                "knowledgeRequests": [{"type": "METRIC", "query": "GMV 退款 赔付 工单"}],
            }
        return FakeDailyProfilePlannerLlm().json_chat(system_prompt, user_prompt, fallback)


class FakeDetailPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "PLAN_READY",
            "reason": "llm understood detail lookup",
            "questionUnderstanding": {
                "analysisGrain": "order",
                "rankingObjective": {},
                "requestedMeasures": [
                    {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
                    {"metricRef": "goods_cnt", "ownerTable": "dwm_goods_detail_df", "sourcePhrase": "商品发布时间"},
                ],
                "filters": [{"field": "order_id", "value": "order_id_100"}],
                "timeWindowDays": 30,
            },
        }


class SlowAsyncModel:
    def __init__(self):
        self.cancelled = False

    async def ainvoke(self, messages):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class ToolCallingAsyncModel:
    def __init__(self):
        self.bound_tools = []
        self.tool_choice = ""

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tools = tools
        self.tool_choice = tool_choice or ""
        return self

    async def ainvoke(self, messages):
        return FakeToolMessage()


class FakeToolMessage:
    content = ""
    tool_calls = [{"id": "call_1", "name": "draft_sql", "args": {"sql": "SELECT 1", "reason": "unit test"}}]
    additional_kwargs = {}


class FakeSelfLoopPlannerLlm:
    configured = True
    last_error = ""
    error_events = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        return {
            "status": "PLAN_READY",
            "reason": "llm graph included self loops",
            "questionUnderstanding": {
                "analysisGrain": "product",
                "rankingObjective": {
                    "metricRef": "ticket_cnt",
                    "sourcePhrase": "客服工单量最高的前10个商品",
                    "ownerTable": "dwm_cs_ticket_detail_di",
                    "groupByColumn": "spu_id",
                    "order": "desc",
                    "limit": 10,
                },
                "requestedMeasures": [
                    {"metricRef": "refund_bill_cnt", "sourcePhrase": "退款量", "ownerTable": "dwm_trade_refund_detail_di"},
                    {"metricRef": "repay_amt", "sourcePhrase": "赔付金额", "ownerTable": "dwm_cs_repay_detail_df"},
                ],
            },
            "queryPlan": {
                "intents": [
                    {
                        "question": "工单量最高商品",
                        "intentType": "VALID",
                        "category": "CS_TICKET",
                        "answerMode": "TOPN",
                        "planTaskId": "anchor_ticket",
                        "taskRole": "ANCHOR",
                        "preferredTable": "dwm_cs_ticket_detail_di",
                        "groupByColumn": "spu_id",
                        "metricColumn": "ticket_id",
                        "metricName": "ticket_cnt",
                        "outputKeys": ["seller_id", "spu_id", "sub_order_id", "ticket_id"],
                    },
                    {
                        "question": "退款量",
                        "intentType": "VALID",
                        "category": "REFUND",
                        "answerMode": "GROUP_AGG",
                        "planTaskId": "refund_lookup",
                        "taskRole": "DEPENDENT",
                        "preferredTable": "dwm_trade_refund_detail_di",
                        "groupByColumn": "spu_name",
                        "metricColumn": "refund_id",
                        "metricName": "refund_bill_cnt",
                        "outputKeys": ["seller_id", "sub_order_id", "spu_name", "refund_id"],
                        "dependsOnTaskIds": ["anchor_ticket", "refund_lookup"],
                    },
                    {
                        "question": "赔付金额",
                        "intentType": "VALID",
                        "category": "COMPENSATION",
                        "answerMode": "GROUP_AGG",
                        "planTaskId": "repay_lookup",
                        "taskRole": "DEPENDENT",
                        "preferredTable": "dwm_cs_repay_detail_df",
                        "groupByColumn": "sub_order_id",
                        "metricColumn": "repay_amt",
                        "metricName": "repay_amt",
                        "outputKeys": ["seller_id", "sub_order_id", "ticket_id", "repay_amt"],
                        "dependsOnTaskIds": ["anchor_ticket", "repay_lookup"],
                    },
                ],
                "dependencies": [
                    {
                        "anchorTaskId": "anchor_ticket",
                        "dependentTaskId": "refund_lookup",
                        "joinKey": "sub_order_id",
                        "anchorColumn": "sub_order_id",
                        "dependentColumn": "sub_order_id",
                    },
                    {
                        "anchorTaskId": "refund_lookup",
                        "dependentTaskId": "refund_lookup",
                        "joinKey": "sub_order_id",
                        "anchorColumn": "sub_order_id",
                        "dependentColumn": "sub_order_id",
                    },
                    {
                        "anchorTaskId": "anchor_ticket",
                        "dependentTaskId": "repay_lookup",
                        "joinKey": "ticket_id",
                        "anchorColumn": "ticket_id",
                        "dependentColumn": "ticket_id",
                    },
                    {
                        "anchorTaskId": "repay_lookup",
                        "dependentTaskId": "repay_lookup",
                        "joinKey": "sub_order_id",
                        "anchorColumn": "sub_order_id",
                        "dependentColumn": "sub_order_id",
                    },
                ],
                "finalRequiredEvidence": ["ticket_cnt", "refund_bill_cnt", "repay_amt"],
            },
        }


class FakeDoris:
    def query(self, sql, params=None):
        return [{"seller_id": "100"}]


class SlowDoris:
    def __init__(self, delay_seconds=0.1):
        self.delay_seconds = delay_seconds

    def query(self, sql, params=None):
        time.sleep(self.delay_seconds)
        return [{"seller_id": "100"}]


class MemoryFailThenOkDoris:
    def __init__(self):
        self.calls = 0
        self.sqls = []

    def query(self, sql, params=None):
        self.calls += 1
        self.sqls.append(sql)
        if self.calls == 1:
            raise RuntimeError("MEM_ALLOC_FAILED: query memory limit exceeded")
        return [{"seller_id": "100", "pt": "20260620", "order_gmv_amt_1d": 100}]


class AlwaysMemoryFailDoris:
    def query(self, sql, params=None):
        raise RuntimeError("MEM_ALLOC_FAILED: query memory limit exceeded")


class FakeDorisWithFreshness:
    def query(self, sql, params=None):
        if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            return [{"min_pt": "20260601", "max_pt": "20260622"}]
        return [{"seller_id": "100", "sub_order_id": "sub_order_id_1", "order_cnt": 1}]


class ManyRowsDoris:
    def query(self, sql, params=None):
        if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            return [{"min_pt": "20260601", "max_pt": "20260622"}]
        return [
            {"seller_id": "100", "sub_order_id": "sub_order_id_1", "order_cnt": 1},
            {"seller_id": "100", "sub_order_id": "sub_order_id_2", "order_cnt": 1},
            {"seller_id": "100", "sub_order_id": "sub_order_id_3", "order_cnt": 1},
        ]


class CountingDoris:
    def __init__(self):
        self.calls = 0

    def query(self, sql, params=None):
        self.calls += 1
        return []
