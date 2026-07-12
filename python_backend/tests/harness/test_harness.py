import asyncio
import json
import time
from pathlib import Path
from threading import Event, Lock

import pytest

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.state import emit, merge_agent_state_update
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.workflow import (
    append_knowledge_request_gaps,
    answer_safe_memory_injection,
    create_workflow,
    dedupe_workflow_knowledge_requests,
    knowledge_request_key,
    merchant_access_role,
    observability_summary,
)
from merchant_ai.models import (
    AgentDecision,
    AgentActionTrace,
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    ChatContext,
    ChatResponse,
    ChatRequest,
    EntitySet,
    EvidenceGap,
    FastUnderstandingResult,
    FreshnessCheckResult,
    GoldenEvaluationRequest,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    KnowledgeRequest,
    KnowledgeRequestType,
    KnowledgeRef,
    KnowledgeSuggestionReviewRequest,
    MerchantInfo,
    NodeExecutionContext,
    NodePlanContract,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    PlannerRepairRequest,
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    QuestionRoute,
    PendingAnswer,
    RunCreateRequest,
    RelationshipEntry,
    RecallItem,
    RecallBundle,
    RouteSlots,
    RouteTimeWindow,
    RoutingDecision,
    SkillLifecycleRecord,
    SkillMatchState,
    SqlValidationResult,
    SqlRepairAttempt,
    SkillDraftReviewRequest,
    SkillEvaluationCase,
    SkillEvaluationRequest,
    TaskRole,
    ThreadData,
    TopicBuildRequest,
    ToolCallExecutionResult,
    ToolCallRequest,
    ToolCachePolicy,
    TraceSpan,
    LoadBalancerTarget,
    MemoryRetrievalCandidate,
    VerifiedEvidence,
    WorkspaceManifest,
    UserIdentity,
)
from merchant_ai.services.assets import (
    HybridRecallService,
    PlanningAssetPackBuilder,
    SemanticAssetGovernanceService,
    SemanticCatalogService,
    SemanticMetricIndex,
    SkillLoader,
    TopicBuilderWorkflow,
    TopicAssetService,
    WikiMemoryService,
)
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.cache import build_ttl_cache
from merchant_ai.services.clarification import ClarificationResolutionService
from merchant_ai.services.answer import (
    AnswerComposeService,
    DailyReportService,
    FeedbackService,
    analysis_summary_required,
    answer_skill_required,
    answer_data_package,
    compact_rule_evidence,
    plan_requires_rule_evidence,
    plan_has_ratio_calculation,
    select_answer_skill,
    sanitize_business_answer_text,
    business_summary_table,
    task_evidence_sections,
    answer_skill_headers,
    render_structured_skill_answer,
)
from merchant_ai.services.context import ContextManager
from merchant_ai.services.controlled_react import ControlledReactExplorer
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.latency import LatencyOptimizer
from merchant_ai.services.formulas import compile_metric_formula, reconcile_metric_formula_for_schema
from merchant_ai.services.llm import LlmClient, prompt_cache_metadata
from merchant_ai.services.middleware import (
    ActionContractMiddleware,
    CancellationMiddleware,
    ClarificationMiddleware,
    ContextBudgetMiddleware,
    DynamicContextMiddleware,
    FileSystemContextMiddleware,
    LoopGuardMiddleware,
    MemoryMiddleware,
    MiddlewareChain,
    RunBudgetMiddleware,
    SafetyFinishReasonMiddleware,
    SummarizeMiddleware,
    TokenUsageMiddleware,
    ToolCallRecoveryMiddleware,
    ToolOutputBudgetMiddleware,
    append_middleware_event,
    default_harness_middlewares,
    estimate_text_tokens,
)
from merchant_ai.services.context_assembly import ContextAssembler, ThreadContextService
from merchant_ai.services.runs import AgentAsyncRunService, AgentRunManager, AgentRunStreamService, answer_chunks, run_duration_ms
from merchant_ai.services.memory import (
    EnterpriseMemoryStore,
    MemoryEsRepository,
    MemoryKnowledgeGovernanceService,
    MemoryManagementService,
    StructuredMemoryStore,
    create_memory_store,
    estimate_memory_tokens,
    boost_vector_candidates,
    memory_budget_tokens,
    memory_query_hash,
    truncate_memory_text_by_tokens,
)
from merchant_ai.services.memory_constraints import build_memory_constraints
from merchant_ai.services.merchant_profile import MerchantProfileStore, MerchantProfileSummaryService
from merchant_ai.services.evaluation import GOLDEN_QUESTIONS, GoldenCaseLoader, GoldenEvaluationService, evaluation_observability_record
from merchant_ai.services.repositories import PendingAnswerStore, write_json
from merchant_ai.services.retrieval import (
    EsKnowledgeRetrievalService,
    HybridKnowledgeRetrievalService,
    build_retrieval_profile,
    limit_recall_items_by_source_type,
    rrf_fuse_recall_items,
    source_type_top_k_policy,
)
from merchant_ai.services.recall_index import EsRecallIndexAdapter, RecallIndexManager
from merchant_ai.services.planning import (
    PlannerReflectionAgent,
    QuestionUnderstandingCompiler,
    QueryGraphPlanner,
    QueryGraphValidator,
    SemanticRelationshipGraphIndex,
    SemanticMetricResolver,
    UnderstandingCoverageCritic,
    anchor_mismatch_issue,
    compact_understanding_catalog,
    compact_openai_tool_schema,
    compile_semantic_entity_chain_graph,
    compile_semantic_metric_fallback_graph,
    compile_query_graph_from_understanding,
    planner_prompt_stats,
    planner_repair_feedback_for_understanding,
    repair_dependency_key_production_gaps,
    repair_more_specific_root_metric,
    scope_ratio_numerator_knowledge_request,
    semantic_workspace_manifest_from_asset_pack,
    ultra_compact_understanding_catalog,
)
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.quick_metrics import published_semantic_quick_metrics, quick_metric_response
from merchant_ai.services.query import (
    NodeWorkerExecutor,
    SqlValidationService,
    bind_node_sql_parameters,
    choose_entity_transfer_key,
    compact_tool_failure_for_prompt,
    entity_set_from_rows,
    merge_task_result_bundles,
    multi_entity_transfer_values,
    serialize_tool_execution_result,
    table_access_hint,
)
from merchant_ai.services.skill_worker import SkillWorkerExecutor
from merchant_ai.services.sandbox import MerchantAnalysisSandbox
from merchant_ai.services.skill_drafts import SkillDraftService
from merchant_ai.services.skill_evaluation import SkillEvaluationService
from merchant_ai.services.repositories import DorisRepository
from merchant_ai.services.routing import KeywordExtractService, QuestionRoutingService, RouteSlotExtractor, TopicRouterService
from merchant_ai.services.tool_runtime import (
    RoundRobinLoadBalancer,
    ToolCallExecutor,
    ToolFailureRegistry,
    ToolRuntimePolicyRegistry,
    ToolRuntimeService,
    tool_error_details,
)
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
    assert '<prompt id="lead.system" version="v1" agent="LeadAgent" templateFingerprint="' in prompt.system_prompt
    assert '<static-section name="common.stable_boundary" version="v1"' in prompt.system_prompt
    assert '<runtime-section name="available_actions">' in prompt.system_prompt
    assert "plan_graph" in prompt.system_prompt
    assert "trade" in prompt.system_prompt
    assert prompt.trace()["promptId"] == "lead.system"
    assert prompt.trace()["sections"] == ["available_actions", "loaded_skills"]
    assert "lead.action_registry" in prompt.trace()["staticSections"]
    assert prompt.trace()["templateFingerprint"]
    assert prompt.trace()["renderFingerprint"]
    cache_meta = prompt_cache_metadata(prompt.system_prompt)
    assert cache_meta["promptId"] == "lead.system"
    assert cache_meta["templateFingerprint"] == prompt.trace()["templateFingerprint"]


def test_harness_middlewares_are_configurable_by_registry():
    settings = get_settings().model_copy(
        update={
            "harness_middleware_disabled": "memory,skill",
            "harness_middleware_order": "dynamic_context,run_budget",
        }
    )
    middlewares = default_harness_middlewares(settings, ContextManager(settings))
    names = [item.name for item in middlewares]

    assert names[:2] == ["dynamic_context", "run_budget"]
    assert "memory" not in names
    assert "skill" not in names
    assert "context_snapshot" in names


def test_agent_state_merge_dedupes_shared_runtime_lists():
    existing = {
        "tool_call_ledger": [{"toolCallId": "call_1", "status": "running"}],
        "runtime_injection": {"a": 1},
    }
    update = {
        "tool_call_ledger": [
            {"toolCallId": "call_1", "status": "success"},
            {"toolCallId": "call_2", "status": "failed"},
        ],
        "runtime_injection": {"b": 2},
    }

    merged = merge_agent_state_update(existing, update)

    assert merged["tool_call_ledger"] == [
        {"toolCallId": "call_1", "status": "success"},
        {"toolCallId": "call_2", "status": "failed"},
    ]
    assert merged["runtime_injection"] == {"a": 1, "b": 2}


def test_middleware_chain_merges_partial_state_updates_and_records_trace():
    class PartialUpdateMiddleware:
        name = "partial_update"

        def before_policy(self, state):
            return {
                "tool_call_ledger": [
                    {"toolCallId": "call_1", "status": "success"},
                    {"toolCallId": "call_2", "status": "failed"},
                ],
                "runtime_injection": {"b": 2},
            }

        def before_action(self, state, decision):
            return state

    state = {
        "tool_call_ledger": [{"toolCallId": "call_1", "status": "running"}],
        "runtime_injection": {"a": 1},
        "middleware_events": [],
    }

    result = MiddlewareChain([PartialUpdateMiddleware()]).before_policy(state)

    assert result["tool_call_ledger"] == [
        {"toolCallId": "call_1", "status": "success"},
        {"toolCallId": "call_2", "status": "failed"},
    ]
    assert result["runtime_injection"] == {"a": 1, "b": 2}
    assert any(event.code == "MIDDLEWARE_CHAIN_ORDER" for event in result["middleware_events"])
    assert any(event.code == "MIDDLEWARE_STATE_DELTA" for event in result["middleware_events"])


def test_controlled_react_explorer_scores_candidate_graphs_with_guardrails():
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pt", "pay_amt"])],
        metrics=[
            PlanningAssetEntry(key="order_gmv_amt_1d", table="dwm_trade_order_detail_di"),
            PlanningAssetEntry(key="refund_rate", table="dwm_trade_refund_detail_di"),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="gmv",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_gmv_amt_1d",
            )
        ]
    )

    explorer = ControlledReactExplorer()
    hypotheses = explorer.build_hypotheses("最近7天GMV下降原因", pack, {"intentKind": "analysis"})
    candidates = explorer.evaluate_candidates(hypotheses, pack, plan)

    assert hypotheses["mode"] == "controlled_hypothesis_exploration"
    assert len(hypotheses["hypotheses"]) == 3
    assert candidates["mode"] == "candidate_query_graph_sandbox"
    assert candidates["candidates"][0]["status"] == "selected"
    assert candidates["candidates"][0]["guardrailResult"]["directSqlAllowed"] is False


def test_controlled_react_runs_parallel_hypothesis_evidence_reviews():
    explorer = ControlledReactExplorer()
    hypotheses = {
        "hypotheses": [
            {"hypothesisId": "h1", "title": "交易变化", "metricHints": ["order_gmv_amt_1d"]},
            {"hypothesisId": "h2", "title": "退款变化", "metricHints": ["refund_amt_1d"]},
        ]
    }
    bundle = QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100, "refund_amt_1d": 10}])
    result = explorer.run_parallel_evidence_reviews(hypotheses, AgentRunResult(merged_query_bundle=bundle))

    assert len(result) == 2
    assert all(item["workerMode"] == "parallel_isolated_evidence_review" for item in result)
    assert all(item["status"] == "supported_by_available_evidence" for item in result)


def test_controlled_react_ranks_independent_query_evidence_and_prunes_weak_hypothesis():
    explorer = ControlledReactExplorer()
    strong_result = AgentRunResult(
        task_results=[AgentTaskResult(task_id="h1_node", success=True, query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "gmv": 100}], tables=["trade"]))],
        merged_query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "gmv": 100}], tables=["trade"]),
        verified_evidence=VerifiedEvidence(passed=True, covered_evidence=["gmv", "pt"]),
    )
    weak_result = AgentRunResult(
        task_results=[AgentTaskResult(task_id="h2_node", success=False, query_bundle=QueryBundle(failed=True, error="no rows"))],
        merged_query_bundle=QueryBundle(failed=True),
        verified_evidence=VerifiedEvidence(passed=False, gaps=[EvidenceGap(code="ZERO_ROWS", reason="no rows")]),
    )

    comparison = explorer.compare_independent_executions(
        [
            {"hypothesisId": "h1", "hypothesis": {"title": "交易下降"}, "semanticScore": 40, "validation": GraphValidationResult(valid=True), "runResult": strong_result},
            {"hypothesisId": "h2", "hypothesis": {"title": "退款影响"}, "semanticScore": 20, "validation": GraphValidationResult(valid=True), "runResult": weak_result},
        ],
        min_score=45,
        max_survivors=1,
    )

    assert comparison["winnerId"] == "h1"
    assert comparison["survivorIds"] == ["h1"]
    assert comparison["prunedIds"] == ["h2"]
    decision = explorer.followup_decision(comparison["ranked"][0])
    assert decision["action"] == "stop"
    assert "足够" in decision["reason"]


def test_main_agent_exposes_independent_hypothesis_query_tool_for_attribution_question():
    settings = get_settings().model_copy(update={"hypothesis_query_exploration_enabled": True, "lead_agent_autonomous_enabled": True})
    policy = V2AgentPolicy(settings)
    state = {
        "question": "最近7天GMV下降原因分析",
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "query_graph_validation_result": GraphValidationResult(valid=True),
        "sql_generated": True,
        "sql_repair_reviewed": True,
        "evidence_graph_verified": True,
        "hypothesis_exploration_completed": False,
        "hypothesis_exploration": {
            "questionSignals": {"mentionsAttribution": True, "mentionsDrop": True},
            "hypotheses": [{"hypothesisId": "h1"}, {"hypothesisId": "h2"}, {"hypothesisId": "h3"}],
        },
        "plan": QueryPlan(intents=[QuestionIntent(plan_task_id="main", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC, preferred_table="trade")]),
        "agent_run_result": AgentRunResult(task_results=[AgentTaskResult(task_id="main", success=True, query_bundle=QueryBundle(rows=[{"gmv": 1}]))]),
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "react_round": 5,
    }

    decision = policy.decide(state)

    assert decision.selected_action == "explore_hypotheses"
    assert "explore_hypotheses" in decision.available_actions


def test_main_agent_recovers_planner_failure_with_semantic_hypothesis_queries():
    settings = get_settings().model_copy(update={"hypothesis_query_exploration_enabled": True, "lead_agent_autonomous_enabled": True})
    policy = V2AgentPolicy(settings)
    state = {
        "question": "GMV下降原因分析",
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "planner_provider_error": "PLANNER_LLM_TIMEOUT",
        "plan": QueryPlan(),
        "planning_asset_pack": PlanningAssetPack(metrics=[PlanningAssetEntry(key="gmv", table="trade")]),
        "hypothesis_exploration_completed": False,
        "hypothesis_exploration": {"hypotheses": [{"hypothesisId": "h1"}, {"hypothesisId": "h2"}]},
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "agent_run_result": AgentRunResult(),
        "react_round": 4,
    }

    decision = policy.decide(state)

    assert decision.selected_action == "explore_hypotheses"


def test_execution_tier_prefers_direct_for_simple_metric_but_is_not_hard_coded():
    optimizer = LatencyOptimizer()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="gmv",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                preferred_table="trade",
            )
        ]
    )
    fast = FastUnderstandingResult(intent_kind="metric_query", complexity="simple")

    normal = optimizer.execution_tier_policy("最近7天GMV", plan, fast, remaining_seconds=60)
    escalated = optimizer.execution_tier_policy(
        "最近7天GMV",
        plan,
        fast,
        remaining_seconds=60,
        prior_failure_count=1,
        has_attachments=True,
    )

    assert normal["defaultMode"] == "direct"
    assert normal["allowedModes"] == ["direct", "subagent"]
    assert escalated["defaultMode"] == "subagent"
    assert "prior_execution_failure" in escalated["reasons"]
    assert "attachment_context" in escalated["reasons"]


def test_controlled_react_builds_safe_semantic_seed_query_graph_when_planner_is_unavailable():
    explorer = ControlledReactExplorer()
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="trade", columns=["seller_id", "pt", "pay_amt"])],
        metrics=[PlanningAssetEntry(key="gmv", title="GMV", table="trade", metadata={"sourceColumns": ["pay_amt"]})],
    )

    plan = explorer.fallback_hypothesis_seed_plan(
        {"hypothesisId": "h1", "metricHints": ["gmv"], "requiredEvidence": ["trend"]},
        pack,
        "GMV下降原因",
        "hyp_h1",
        7,
    )

    assert plan.intents[0].preferred_table == "trade"
    assert plan.intents[0].metric_column == "pay_amt"
    assert plan.intents[0].group_by_column == "pt"
    assert plan.intents[0].analysis_source == "hypothesis_semantic_seed"


def test_workflow_keeps_independent_hypothesis_evidence_in_ledger(monkeypatch, tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "agent_checkpointer_backend": "memory",
            "hypothesis_max_rounds": 1,
            "hypothesis_max_survivors": 1,
            "hypothesis_min_survivor_score": 45,
            "llm_api_key": "",
            "embedding_api_key": "",
            "es_enabled": False,
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state("GMV下降原因分析", "100", ChatContext(), None, "thread_hyp", "run_hyp", [])
    state["merchant"] = MerchantInfo(merchant_id="100", merchant_name="test")
    state["hypothesis_exploration"] = {
        "hypotheses": [
            {"hypothesisId": "h1", "title": "交易规模下降", "metricHints": ["gmv"]},
            {"hypothesisId": "h2", "title": "退款压力上升", "metricHints": ["refund"]},
        ]
    }
    state["candidate_query_graphs"] = {
        "candidates": [
            {"hypothesisId": "h1", "score": 45},
            {"hypothesisId": "h2", "score": 20},
        ]
    }
    state["plan"] = QueryPlan(
        intents=[QuestionIntent(plan_task_id="baseline", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC, preferred_table="trade")]
    )
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="baseline", success=True, query_bundle=QueryBundle(rows=[{"gmv": 90}], tables=["trade"]))],
        merged_query_bundle=QueryBundle(rows=[{"gmv": 90}], tables=["trade"]),
        verified_evidence=VerifiedEvidence(passed=True, covered_evidence=["gmv"]),
    )

    def fake_plan(_state, hypothesis, round_index, _previous, _followup):
        hypothesis_id = hypothesis["hypothesisId"]
        plan = QueryPlan(
            intents=[
                QuestionIntent(
                    plan_task_id="%s_query" % hypothesis_id,
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.METRIC,
                    preferred_table="trade" if hypothesis_id == "h1" else "refund",
                )
            ]
        )
        return {
            "hypothesis": hypothesis,
            "hypothesisId": hypothesis_id,
            "plan": plan,
            "validation": GraphValidationResult(valid=True),
            "round": round_index,
            "planningMode": "independent_planner",
            "semanticScore": 45 if hypothesis_id == "h1" else 20,
        }

    def fake_execute(_state, planned, _parallel):
        executions = []
        for item in planned:
            hypothesis_id = item["hypothesisId"]
            if hypothesis_id == "h1":
                result = AgentRunResult(
                    task_results=[AgentTaskResult(task_id="h1_query", success=True, query_bundle=QueryBundle(rows=[{"gmv": 80}], tables=["trade"]))],
                    merged_query_bundle=QueryBundle(rows=[{"gmv": 80}], tables=["trade"]),
                    verified_evidence=VerifiedEvidence(passed=True, covered_evidence=["gmv"]),
                )
            else:
                result = AgentRunResult(
                    task_results=[AgentTaskResult(task_id="h2_query", success=False, query_bundle=QueryBundle(failed=True, error="no evidence"))],
                    merged_query_bundle=QueryBundle(failed=True),
                    verified_evidence=VerifiedEvidence(passed=False, gaps=[EvidenceGap(code="ZERO_ROWS", reason="no evidence")]),
                )
            executions.append({**item, "runResult": result})
        return executions

    monkeypatch.setattr(workflow, "_generate_independent_hypothesis_plan", fake_plan)
    monkeypatch.setattr(workflow, "_execute_hypothesis_plans_parallel", fake_execute)

    result = workflow.explore_hypotheses(state)

    assert result["hypothesis_exploration_completed"] is True
    assert result["hypothesis_selected_ids"] == ["h1"]
    assert [item["decision"] for item in result["hypothesis_results"]] == ["survive", "pruned"]
    assert {item.task_id for item in result["agent_run_result"].task_results} == {"baseline"}
    ledger = result["hypothesis_evidence_ledger"]
    assert ledger.winner_id == "h1"
    assert ledger.survivor_ids == ["h1"]
    assert ledger.pruned_ids == ["h2"]
    assert ledger.entries[0].supporting_evidence_ids
    assert ledger.entries[1].failed_evidence_ids
    assert ledger.entries[1].elimination_reason
    workflow.checkpoint_manager.close()


def test_always_apply_rules_bypass_normal_recall():
    rules = TopicAssetService(get_settings()).always_apply_rules([QuestionCategory.REFUND])

    assert rules
    assert all(item.get("alwaysApply") is True for item in rules)
    assert all(item.get("topic") and item.get("tableName") for item in rules)


def test_merchant_identity_maps_business_roles_to_access_roles():
    identity = UserIdentity(user_id="u_finance", role="merchant_finance", region="AU")

    assert merchant_access_role(identity.role) == "merchant_finance"
    assert "Region：AU" in identity.prompt_markdown()


def test_analysis_sandbox_rejects_unapproved_script(tmp_path):
    result = MerchantAnalysisSandbox(get_settings()).run_python(tmp_path / "unknown.py", [], tmp_path, 1)

    assert result.returncode == 126
    assert result.stderr == "SANDBOX_SCRIPT_NOT_APPROVED"


def test_latency_optimizer_marks_simple_single_node_graph_as_fast_path():
    optimizer = LatencyOptimizer()
    policy = optimizer.initial_policy(
        FastUnderstandingResult(intent_kind="metric_query", complexity="simple", needs_planner=False)
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="gmv_metric",
                preferred_table="ads_merchant_profile",
            )
        ]
    )

    optimized = optimizer.update_after_plan(policy, plan)

    assert optimized["mode"] == "fast_path_verified_graph"
    assert "reflect_query_graph" in optimized["skipNodes"]
    assert optimizer.answer_allows_llm(optimized) is False


def test_middleware_chain_fail_closed_blocks_on_critical_middleware_error():
    class BrokenPermissionMiddleware:
        name = "permission"
        failure_policy = "closed"

        def before_policy(self, state):
            raise RuntimeError("permission backend unavailable")

        def before_action(self, state, decision):
            return state

    state = {"middleware_events": [], "safety_finish_reasons": [], "chat_bi_completed": False, "answer": ""}

    result = MiddlewareChain([BrokenPermissionMiddleware()]).before_policy(state)

    assert result["middleware_blocked"] is True
    assert result["chat_bi_completed"] is True
    assert result["safety_finish_reasons"][0]["finishReason"] == "middleware_fail_closed"
    assert any(event.code == "MIDDLEWARE_FAIL_CLOSED" and event.channel == "audit" for event in result["middleware_events"])


def test_middleware_events_preserve_audit_entries_when_trace_is_capped():
    state = {"middleware_events": []}
    append_middleware_event(state, "permission", "before_policy", status="blocked", code="WRITE_OPERATION_REQUIRES_HUMAN", message="blocked")
    for index in range(240):
        append_middleware_event(state, "token_usage", "before_policy", status="observed", code="TOKEN_USAGE_ESTIMATED", message=str(index))

    assert len(state["middleware_events"]) == 200
    assert any(event.code == "WRITE_OPERATION_REQUIRES_HUMAN" and event.channel == "audit" for event in state["middleware_events"])


def test_deferred_tool_schema_loader_selects_requested_schemas_only():
    from merchant_ai.services.tools import deferred_tool_schema_loader_tool, select_tool_schemas, semantic_file_tool_definitions, tool_schema_catalog

    tools = semantic_file_tool_definitions()
    loader = deferred_tool_schema_loader_tool([tool.name for tool in tools])
    selected = select_tool_schemas(tools, ["semantic_read"])

    assert loader.name == "load_tool_schemas"
    assert any(item["name"] == "semantic_read" for item in tool_schema_catalog(tools))
    assert [item["function"]["name"] for item in selected] == ["semantic_read"]


def test_evidence_gap_has_standard_recovery_fields():
    verified = EvidenceVerifier().verify(
        "最近 7 天 GMV 是多少",
        QueryPlan(),
        AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="gmv_node",
                    success=False,
                    query_bundle=QueryBundle(failed=True, error="Unknown column gmv_bad", sql="select gmv_bad from dwd_order"),
                )
            ]
        ),
    )

    gap = verified.gaps[0]

    assert gap.gap_code == "UNKNOWN_COLUMN"
    assert gap.source_node_id == "gmv_node"
    assert gap.suggested_action == "retry_repair_or_answer_with_gap"
    assert gap.details["gapCode"] == "UNKNOWN_COLUMN"


def test_skill_lifecycle_record_has_standard_engineering_fields(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "llm_api_key": ""})
    workflow = create_workflow(settings)
    state = {
        "skill_lifecycle_records": [],
        "agent_run_result": AgentRunResult(),
        "thinking_steps": [],
    }
    checkpoint = tmp_path / "skill_checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")

    workflow.record_skill_lifecycle(
        state,
        {
            "skillName": "risk_analysis",
            "lifecycleStage": "completed",
            "isolatedRunId": "skill_risk_1",
            "matchedBy": "unit",
            "checkpointPath": str(checkpoint),
            "contextPackage": {"contextHash": "ctx_hash"},
            "startedAt": "2026-07-11T00:00:00",
            "completedAt": "2026-07-11T00:00:01",
            "durationMs": 1000,
            "progress": ["matched", "completed"],
        },
    )

    record = state["skill_lifecycle_records"][0]
    assert record.record_id
    assert record.status == "success"
    assert record.context_hash == "ctx_hash"
    assert record.duration_ms == 1000
    assert record.artifact_refs


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
    assert "scopeConstraints" in understanding_schema["required"]


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
    pay_amt_metric = next(item for item in asset["metrics"] if item["metricKey"] == "pay_amt")
    assert {"销售额", "成交额", "交易成交金额"} <= set(pay_amt_metric["aliases"])


def test_semantic_catalog_exposes_filesystem_context_tools():
    settings = get_settings()
    catalog = SemanticCatalogService(TopicAssetService(settings))
    refs = catalog.ls(topic="电商交易", query="dwm_trade_order_detail_di", limit=5)
    order_ref = next(ref for ref in refs if ref["table"] == "dwm_trade_order_detail_di")
    assert order_ref["path"] == "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json"
    assert order_ref["merchantUri"].startswith("merchant://topic/电商交易/table/dwm_trade_order_detail_di")
    assert order_ref["contextLayer"] == "L2"
    assert order_ref["layers"]["metrics"] > 0
    content = catalog.read(ref_id=order_ref["refId"], max_chars=800)
    assert content["success"]
    assert content["merchantUri"] == order_ref["merchantUri"]
    assert content["contextLayer"] == "L2"
    assert "dwm_trade_order_detail_di" in content["content"]
    hits = catalog.grep("order_detail_cnt", topic="电商交易", limit=5)
    assert any(hit["refId"] == order_ref["refId"] for hit in hits)


def test_semantic_workspace_manifest_uses_layered_context_uris():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        "最近7天订单量是多少",
        recall_bundle_empty(),
        [QuestionCategory.TRADE],
    )

    workspace = semantic_workspace_manifest_from_asset_pack(pack, limit=4)

    assert workspace["uriScheme"] == "merchant://"
    assert {"L0", "L1", "L2"} <= set(workspace["layers"])
    assert workspace["manifestRefs"]
    assert workspace["manifestRefs"][0]["merchantUri"].startswith("merchant://topic/")
    assert workspace["tableRefs"]
    assert workspace["tableRefs"][0]["merchantUri"].startswith("merchant://topic/")


def test_artifact_store_returns_context_uri(tmp_path):
    settings = get_settings()
    store = WorkspaceArtifactStore(settings, tmp_path)

    artifact = store.write_json("planner", "query_graph.json", {"ok": True}, preview_chars=0)
    listed = store.ls("planner", limit=5)
    read_back = store.read(artifact["relativePath"])

    assert artifact["merchantUri"].startswith("merchant://artifact/planner/")
    assert listed[0]["merchantUri"].startswith("merchant://artifact/")
    assert read_back["merchantUri"].startswith("merchant://artifact/")


def test_semantic_file_tool_schemas_are_runtime_injected():
    schemas = semantic_file_tool_schemas()
    names = {schema["name"] for schema in schemas}
    assert {"semantic_ls", "semantic_read", "semantic_grep", "semantic_write"} <= names
    read_schema = next(schema for schema in schemas if schema["name"] == "semantic_read")
    assert {"refId", "path", "maxChars", "offset"} <= set(read_schema["parameters"]["properties"])
    artifact_names = {schema["name"] for schema in artifact_file_tool_schemas()}
    assert {"artifact_ls", "artifact_read", "artifact_grep", "artifact_write"} <= artifact_names


def test_planner_semantic_tool_loop_loads_ref_then_emits_understanding(tmp_path):
    settings = get_settings().model_copy(
        update={"agent_planner_tool_rounds": 3, "planner_filesystem_context_mode": "off"}
    )
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
    assert any(call["name"] == "semantic_ls" for call in plan.planner_tool_calls)
    assert any(call["name"] == "semantic_read" for call in plan.planner_tool_calls)
    assert "semantic:电商交易:manifest" in plan.planner_loaded_refs
    assert "semantic:电商交易:dwm_trade_order_detail_di:asset" in plan.planner_loaded_refs
    assert plan.planner_context_files
    assert len(planner.llm.fast_calls) == 1
    assert planner.llm.calls[0]["semanticWorkspace"]["mode"] == "filesystem_as_context"
    assert "semanticManifest" not in planner.llm.calls[0]
    assert any("planner.semantic_tool_loop=enabled" == item for item in plan.agent_trace)
    assert any("planner.semantic_tool_loop=on_demand" == item for item in plan.agent_trace)


def test_planner_auto_mode_lets_model_read_semantic_assets_when_needed(tmp_path):
    settings = get_settings().model_copy(
        update={"agent_planner_tool_rounds": 3, "planner_filesystem_context_mode": "auto"}
    )
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
    llm = SemanticToolLoopPlannerLlm()
    planner = QueryGraphPlanner(
        llm,
        semantic_catalog=catalog,
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )

    plan, requests, _ = planner.plan(question, [], "", recall, pack, [], [])

    assert not requests
    assert plan.intents
    assert llm.fast_calls == []
    first_payload = llm.calls[0]
    assert first_payload["filesystemContextPolicy"]["entry"] == "adaptive"
    assert first_payload["filesystemContextPolicy"]["mustReadBeforeEmit"] is False
    assert first_payload["semanticWorkspace"]["mode"] == "filesystem_as_context"
    assert any(call["name"] == "semantic_read" for call in plan.planner_tool_calls)
    assert any("planner.semantic_tool_loop=adaptive" == item for item in plan.agent_trace)
    assert any("planner.filesystem_context_mode=auto" == item for item in plan.agent_trace)


def test_planner_auto_mode_can_emit_directly_when_asset_pack_is_enough(tmp_path):
    class AdaptiveDirectPlannerLlm:
        configured = True
        last_error = ""
        error_events = []

        def __init__(self):
            self.calls = []
            self.fast_calls = []

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None):
            self.fast_calls.append(json.loads(user_prompt))
            return {}

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, timeout_seconds=None, tool_choice=None):
            payload = json.loads(user_prompt)
            self.calls.append(payload)
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_emit_understanding",
                        "name": "emit_question_understanding",
                        "args": {
                            "status": "UNDERSTOOD",
                            "reason": "compact asset pack was enough",
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

    settings = get_settings().model_copy(
        update={"agent_planner_tool_rounds": 3, "planner_filesystem_context_mode": "auto"}
    )
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
    llm = AdaptiveDirectPlannerLlm()
    planner = QueryGraphPlanner(
        llm,
        semantic_catalog=catalog,
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )

    plan, requests, _ = planner.plan(question, [], "", recall, pack, [], [])

    assert not requests
    assert plan.intents
    assert llm.fast_calls == []
    assert llm.calls[0]["filesystemContextPolicy"]["entry"] == "adaptive"
    assert not any(call["name"].startswith("semantic_") for call in plan.planner_tool_calls)
    assert any(call["name"] == "emit_question_understanding" for call in plan.planner_tool_calls)
    assert any("planner.semantic_tool_loop=adaptive" == item for item in plan.agent_trace)


def test_semantic_catalog_lists_topic_manifest_before_detail_refs():
    settings = get_settings()
    catalog = SemanticCatalogService(TopicAssetService(settings))
    refs = catalog.ls(topic="电商交易", limit=5)
    assert refs
    assert refs[0]["kind"] == "TOPIC_MANIFEST"
    assert refs[0]["refId"] == "semantic:电商交易:manifest"
    read = catalog.read(ref_id="semantic:电商交易:manifest", max_chars=2000)
    assert read["success"]
    assert read["kind"] == "TOPIC_MANIFEST"
    assert "tables" in read["content"]


def test_planner_limits_need_more_to_one_recovery_call():
    class NeedMoreTwiceLlm:
        configured = True
        last_error = ""
        error_events = []

        def __init__(self):
            self.calls = 0

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            self.calls += 1
            return {
                "status": "NEED_MORE_KNOWLEDGE",
                "reason": "need more despite loaded semantic assets",
                "knowledgeRequests": [{"type": "TABLE", "query": "refund goods"}],
            }

    question = "最近 7 天有退款的订单，关联看一下对应商品发布时间。"
    pack = trade_refund_goods_pack()
    llm = NeedMoreTwiceLlm()
    plan, requests, _ = QueryGraphPlanner(llm).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert llm.calls == 2
    assert requests
    assert not plan.intents
    assert any("planner.need_more_fail_closed" == item for item in plan.agent_trace)
    assert any("planner.llm_call_budget=recovery_exhausted" == item for item in plan.agent_trace)


def test_planner_fast_path_skips_semantic_tool_loop_when_understood(tmp_path):
    settings = get_settings().model_copy(
        update={"agent_planner_tool_rounds": 3, "planner_filesystem_context_mode": "off"}
    )
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


def test_planner_success_path_does_not_refine_with_semantic_tool_loop(tmp_path):
    settings = get_settings().model_copy(
        update={"agent_planner_tool_rounds": 3, "planner_filesystem_context_mode": "off"}
    )
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
    assert requests
    assert not plan.intents
    assert any(request.type == KnowledgeRequestType.METRIC and "GMV" in request.query for request in requests)
    assert llm.tool_chat_payloads == []
    assert plan.planner_tool_calls == []
    assert any("planner=llm_understanding_needs_semantic_metric_evidence" == item for item in plan.agent_trace)
    assert not any("planner.semantic_tool_loop=refined_understanding" == item for item in plan.agent_trace)


def test_recall_uses_single_semantic_asset_document_per_table():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    docs = recall._load_documents()
    order_docs = [doc for doc in docs if doc.table == "dwm_trade_order_detail_di" and doc.topic == "电商交易"]
    table_docs = [doc for doc in order_docs if doc.source_type == "SEMANTIC_TABLE_ASSET"]
    metric_docs = [doc for doc in order_docs if doc.source_type == "SEMANTIC_METRIC"]
    assert len(table_docs) == 1
    assert table_docs[0].doc_id == "semantic:电商交易:dwm_trade_order_detail_di:asset"
    assert table_docs[0].metadata["semanticPath"] == "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json"
    assert "order_detail_cnt" in table_docs[0].content
    assert any(doc.metadata.get("metricKey") == "order_detail_cnt" for doc in metric_docs)
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


def test_compact_assets_does_not_seed_profile_from_recalled_table_asset_for_detail_question():
    settings = get_settings()
    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:电商交易:ads_merchant_profile:asset",
                title="店铺经营画像",
                content="ads_merchant_profile order_gmv_amt_1d refund_rate_1d",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="电商交易",
                table="ads_merchant_profile",
                fusion_score=9.9,
                metadata={"tableName": "ads_merchant_profile"},
            )
        ]
    )
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))

    pack = builder.compact(
        "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额。",
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND],
    )

    assert "ads_merchant_profile" not in pack.known_tables()
    catalog = ultra_compact_understanding_catalog(pack, "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额。")
    assert "ads_merchant_profile" not in {item["table"] for item in catalog["tables"]}


def test_asset_pack_filters_metrics_referencing_missing_live_columns():
    class DriftedGoodsDoris:
        def show_full_columns(self, table):
            if table == "dwm_goods_detail_df":
                return [
                    {"Field": "seller_id"},
                    {"Field": "spu_id"},
                    {"Field": "spu_name"},
                    {"Field": "spu_create_time"},
                    {"Field": "pt"},
                ]
            return []

    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings), doris_repository=DriftedGoodsDoris())

    pack = builder.compact("最近30天上架商品量和商品发布时间", recall_bundle_empty(), [QuestionCategory.GOODS])

    assert "spu_status_code" in pack.missing_live_columns.get("dwm_goods_detail_df", [])
    assert "goods_online_detail_cnt" not in {metric.key for metric in pack.metrics}
    assert pack.metric_compaction.get("schemaFilteredMetricCounts", {}).get("dwm_goods_detail_df", 0) >= 1


def test_relationship_graph_scores_entity_path_above_merchant_hub():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "sub_order_id", "bill_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "order_id", "pt"]),
            PlanningAssetEntry(table="ads_merchant_profile", columns=["seller_id", "pt"]),
            PlanningAssetEntry(table="dwd_merchant_deposit_recharge_df", columns=["seller_id", "pt", "pay_amt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="repay_profile_by_seller",
                left_table="dwm_cs_repay_detail_df",
                right_table="ads_merchant_profile",
                join_keys=[{"leftColumn": "seller_id", "rightColumn": "seller_id"}],
                path_semantics=["tenant_context"],
            ),
            RelationshipEntry(
                relationship_id="profile_deposit_by_seller",
                left_table="ads_merchant_profile",
                right_table="dwd_merchant_deposit_recharge_df",
                join_keys=[{"leftColumn": "seller_id", "rightColumn": "seller_id"}],
                path_semantics=["tenant_context"],
            ),
            RelationshipEntry(
                relationship_id="repay_order_by_sub_order",
                left_table="dwm_cs_repay_detail_df",
                right_table="dwm_trade_order_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
                grain="sub_order_id_bill_id",
                path_semantics=["order_entity", "compensation_entity", "entity_filter"],
            ),
        ],
    )

    path = SemanticRelationshipGraphIndex(pack).edge_path(
        "dwm_cs_repay_detail_df",
        "dwm_trade_order_detail_di",
        analysis_grain="order",
    )

    assert [edge.relationship_id for edge in path] == ["repay_order_by_sub_order"]


def test_relationship_entry_loads_grain_and_path_semantics_from_assets():
    settings = get_settings()
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        "最近30天有客服工单的订单里，哪些后来发生了退款或赔付？",
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )

    rel = next(item for item in pack.relationships if item.relationship_id == "order_ticket_by_sub_order")

    assert rel.grain == "sub_order_id_ticket_id"
    assert {"order_entity", "ticket_entity", "entity_filter"} <= set(rel.path_semantics)


def test_metric_resolver_treats_owner_table_as_scope_not_bonus():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pay_amt", "pt"]),
            PlanningAssetEntry(table="dwd_merchant_deposit_recharge_df", columns=["seller_id", "pay_amt", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_pay_amt",
                table="dwm_trade_order_detail_di",
                title="订单金额",
                aliases=["pay_amt", "订单金额"],
                columns=["pay_amt"],
            ),
            PlanningAssetEntry(
                key="deposit_recharge_amt",
                table="dwd_merchant_deposit_recharge_df",
                title="保证金充值金额",
                aliases=["pay_amt", "充值金额"],
                columns=["pay_amt"],
            ),
        ],
    )

    resolved = SemanticMetricResolver(pack).resolve(
        question="最近60天赔付金额最高订单，关联看订单金额",
        metric_ref="pay_amt",
        owner_table="dwm_trade_order_detail_di",
        source_phrase="订单金额",
    )

    assert resolved.metric is not None
    assert resolved.metric.table == "dwm_trade_order_detail_di"
    assert resolved.metric.key == "order_pay_amt"


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
    assert decision.selected_action == "fast_understand"
    assert decision.selected_node == "fast_understand"
    assert decision.available_actions == ["fast_understand"]


def test_lead_policy_inserts_fast_understand_after_route_before_retrieval():
    policy = V2AgentPolicy(get_settings())
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "route_slots": RouteSlots(),
        "fast_understood": False,
        "data_discovered": False,
        "react_round": 1,
        "plan": QueryPlan(),
    }

    decision = policy.decide(state)

    assert decision.selected_action == "fast_understand"
    assert decision.selected_node == "fast_understand"


def test_action_contract_reroutes_execution_before_graph_validation():
    state = {
        "query_graph_validated": False,
        "middleware_events": [],
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    metric_name="order_count",
                    category=QuestionCategory.TRADE,
                    days=7,
                )
            ]
        ),
        "query_graph_validation_result": GraphValidationResult(),
    }
    decision = AgentDecision(
        selected_action="execute_graph",
        selected_node="execute_query_graph",
        available_actions=["execute_graph", "validate_graph", "answer_data"],
        reason="llm selected execution directly",
    )

    ActionContractMiddleware().before_action(state, decision)

    assert decision.selected_action == "validate_graph"
    assert decision.selected_node == "validate_query_graph"
    assert decision.source == "contract"
    assert state["middleware_events"][-1].code == "ACTION_CONTRACT_MISSING_PREREQUISITE"
    assert state["middleware_events"][-1].metadata["missingStateFlags"] == ["query_graph_validated"]


def test_emit_adds_standard_event_envelope_without_hiding_payload():
    captured = []

    def listener(event_type, node, payload):
        captured.append((event_type, node, payload))

    state = {
        "run_id": "run_evt",
        "thread_id": "thread_evt",
        "event_listener": listener,
        "_active_step_id": "step_evt",
    }

    emit(state, "node.started", "PLAN_QUERY_GRAPH", {"custom": "value"})

    assert captured[0][0] == "node.started"
    assert captured[0][1] == "PLAN_QUERY_GRAPH"
    payload = captured[0][2]
    assert payload["custom"] == "value"
    assert payload["eventEnvelopeVersion"] == "v1"
    assert payload["eventType"] == "node.started"
    assert payload["node"] == "PLAN_QUERY_GRAPH"
    assert payload["runId"] == "run_evt"
    assert payload["threadId"] == "thread_evt"
    assert payload["correlationId"] == "step_evt"
    assert payload["eventId"].startswith("evt_")


def test_lead_agent_defaults_to_adaptive_bounded_llm_sixteen_round_loop():
    assert Settings.model_fields["agent_main_rounds"].default == 16
    assert Settings.model_fields["lead_action_llm_mode"].default == "adaptive"
    assert V2AgentPolicy().max_main_actions == 16


def test_default_lead_llm_only_judges_fast_gate():
    class FastGateLeadActionLlm:
        configured = True

        def __init__(self):
            self.calls = 0

        def tool_json_chat(self, _system_prompt, user_prompt, _tool_schema, fallback=None, timeout_seconds=None):
            self.calls += 1
            payload = json.loads(user_prompt)
            assert payload["fastUnderstanding"]["analysisIntent"] == "ratio"
            return {"actionId": "retrieve_knowledge", "reason": "ratio needs Planner"}

    workflow = create_workflow(get_settings().model_copy(update={"lead_action_llm_mode": "fast_gate"}))
    lead_llm = FastGateLeadActionLlm()
    workflow.planner.llm = lead_llm
    state = workflow._initial_state("最近7天退款金额占GMV比例", "100", ChatContext(), None, "", "")
    state["fast_understanding"] = FastUnderstandingResult(analysis_intent="ratio")
    state["main_agent_observations"] = [{"summary": "ratio request"}]
    fast_decision = AgentDecision(
        selected_action="try_fast_metric",
        selected_node="try_fast_metric",
        available_actions=["try_fast_metric", "retrieve_knowledge"],
        reason="fast gate",
    )

    selected = workflow.apply_bounded_lead_llm_decision(state, fast_decision)

    assert selected.selected_action == "retrieve_knowledge"
    assert selected.source == "lead_llm_tool"
    assert lead_llm.calls == 1

    later_decision = AgentDecision(
        selected_action="retrieve_knowledge",
        selected_node="retrieve_knowledge",
        available_actions=["retrieve_knowledge", "compact_assets"],
        reason="continue workflow",
    )
    unchanged = workflow.apply_bounded_lead_llm_decision(state, later_decision)
    assert unchanged.selected_action == "retrieve_knowledge"
    assert lead_llm.calls == 1


def test_bounded_lead_llm_cannot_select_action_outside_registry_candidates():
    class InvalidLeadActionLlm:
        configured = True

        def json_chat(self, *_args, **_kwargs):
            return {"selectedAction": "invent_new_action", "reason": "try to bypass registry"}

    settings = get_settings().model_copy(update={"lead_action_llm_mode": "always"})
    workflow = create_workflow(settings)
    workflow.planner.llm = InvalidLeadActionLlm()
    state = workflow._initial_state("最近7天总订单量是多少？", "100", ChatContext(), None, "", "")
    decision = AgentDecision(
        selected_action="retrieve_knowledge",
        selected_node="retrieve_knowledge",
        available_actions=["retrieve_knowledge", "compact_assets"],
        reason="deterministic",
    )

    selected = workflow.apply_bounded_lead_llm_decision(state, decision)

    assert selected.selected_action == "retrieve_knowledge"
    assert state["bounded_lead_llm_trace"]["status"] == "ignored"
    assert state["bounded_lead_llm_trace"]["reason"] == "llm_selected_action_not_allowed"


def test_bounded_lead_llm_uses_action_tool_schema():
    class ToolLeadActionLlm:
        configured = True

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, timeout_seconds=None):
            assert ((tool_schema.get("function") or {}).get("name")) == "select_agent_action"
            payload = json.loads(user_prompt)
            assert "observation" in payload
            return {"actionId": "compact_assets", "reason": "assets are next"}

    settings = get_settings().model_copy(update={"lead_action_llm_mode": "always"})
    workflow = create_workflow(settings)
    workflow.planner.llm = ToolLeadActionLlm()
    state = workflow._initial_state("最近7天总订单量是多少？", "100", ChatContext(), None, "", "")
    state["main_agent_observations"] = [{"summary": "retrieved=true; assets=false"}]
    decision = AgentDecision(
        selected_action="retrieve_knowledge",
        selected_node="retrieve_knowledge",
        available_actions=["retrieve_knowledge", "compact_assets"],
        reason="deterministic",
    )

    selected = workflow.apply_bounded_lead_llm_decision(state, decision)

    assert selected.selected_action == "compact_assets"
    assert selected.selected_node == "compact_assets"
    assert selected.source == "lead_llm_tool"
    assert state["bounded_lead_llm_trace"]["tool"]["name"] == "select_agent_action"


def test_bounded_lead_llm_payload_includes_structured_decision_context():
    class ContextAwareLeadActionLlm:
        configured = True

        def tool_json_chat(self, _system_prompt, user_prompt, _tool_schema, fallback=None, timeout_seconds=None):
            payload = json.loads(user_prompt)
            assert payload["decisionContext"]["progress"]["knowledgeRecallStalled"] is True
            assert payload["decisionContext"]["gaps"]["graph"][0]["code"] == "MISSING_RELATIONSHIP"
            return {"actionId": "repair_graph", "reason": "recall stalled; repair with current assets"}

    settings = get_settings().model_copy(update={"lead_action_llm_mode": "always"})
    workflow = create_workflow(settings)
    workflow.planner.llm = ContextAwareLeadActionLlm()
    state = workflow._initial_state("最近7天退款率最高商品关联工单量", "100", ChatContext(), None, "", "")
    state["lead_decision_context"] = {
        "progress": {"knowledgeRecallStalled": True},
        "gaps": {"graph": [{"code": "MISSING_RELATIONSHIP"}]},
    }
    decision = AgentDecision(
        selected_action="retrieve_knowledge",
        selected_node="retrieve_knowledge",
        available_actions=["retrieve_knowledge", "repair_graph", "answer_data"],
        reason="pending knowledge",
    )

    selected = workflow.apply_bounded_lead_llm_decision(state, decision)

    assert selected.selected_action == "repair_graph"
    assert selected.source == "lead_llm_tool"


def test_lead_decision_context_tracks_recall_delta_gaps_and_sql_failures():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("最近7天客服工单异常原因", "100", ChatContext(), None, "", "")
    state["knowledge_bundle"] = KnowledgeBundle(source_refs=["semantic:metric:ticket_cnt"])
    state["recall_bundle"] = RecallBundle(items=[RecallItem(doc_id="semantic:relationship:ticket_order")])
    state["query_graph_validation_result"] = GraphValidationResult(
        valid=False,
        gaps=[GraphValidationGap(code="MISSING_RELATIONSHIP", task_id="ticket_lookup", reason="missing ticket/order edge")],
    )
    state["agent_run_result"] = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="ticket_lookup",
                success=False,
                summary="SQL failed",
                query_bundle=QueryBundle(failed=True, error="Unknown column ticket_state"),
                validation_results=[SqlValidationResult(valid=False, error_code="UNKNOWN_COLUMN", message="ticket_state not found")],
            )
        ],
        evidence_gaps=[EvidenceGap(code="SQL_EXECUTION_FAILED", task_id="ticket_lookup", reason="Unknown column")],
    )
    state["action_history"] = [
        AgentActionTrace(round=1, action="retrieve_knowledge", node="retrieve_knowledge", agent="KnowledgeAgent", status="selected")
    ]
    state["query_graph_retrieve_count"] = 1
    state["recall_strategy"] = {"profileKinds": ["broad"], "queryTypes": ["multi_hop_analysis"]}
    state["worker_dispatch_context"] = {"workerType": "NodeWorker", "shouldDispatch": True, "reason": "complex_or_multi_node_query_graph"}

    first = workflow.build_lead_decision_context(state, {"summary": "first"})
    second = workflow.build_lead_decision_context(state, {"summary": "second"})

    assert first["progress"]["newRecallRefsCount"] == 2
    assert second["progress"]["knowledgeRecallStalled"] is True
    assert second["gaps"]["graph"][0]["code"] == "MISSING_RELATIONSHIP"
    assert second["gaps"]["evidence"][0]["code"] == "SQL_EXECUTION_FAILED"
    assert second["executionFailures"]["sql"][0]["validationErrorCode"] == "UNKNOWN_COLUMN"
    assert second["retrievalStrategy"]["profileKinds"] == ["broad"]
    assert second["workerDispatch"]["workerType"] == "NodeWorker"


def test_asset_pack_does_not_auto_expand_tables_from_relationship_closure():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact("最近 90 天有退款的商品发布时间。", recall_bundle_empty(), [QuestionCategory.REFUND, QuestionCategory.GOODS])
    tables = set(pack.known_tables())
    assert "dwm_trade_refund_detail_di" in tables
    assert "dwm_goods_detail_df" in tables
    assert "dwm_trade_order_detail_di" not in tables
    assert not any("relationship_bridge_table" in item for item in pack.relationship_closure)


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


def test_appeal_success_rate_is_semantic_derived_metric():
    settings = get_settings()
    question = "最近60天申诉成功率和退款率是否有关"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.MERCHANT_OTHER, QuestionCategory.REFUND],
    )
    metric_keys = {metric.key for metric in pack.metrics}
    assert {"appeal_success_rate", "appeal_pass_cnt", "appeal_cnt"} <= metric_keys

    resolution = SemanticMetricResolver(pack).resolve(
        question,
        "appeal_success_rate",
        "dwd_merchant_appeal_detail_df",
        "申诉成功率",
    )
    assert resolution.metric
    assert resolution.metric.key == "appeal_success_rate"
    assert resolution.metric.columns == ["appeal_pass_cnt", "appeal_cnt"]

    plan = compile_query_graph_from_understanding(
        question,
        {
            "analysisGrain": "day",
            "analysisIntent": "trend_check",
            "rankingObjective": {
                "metricRef": "appeal_success_rate",
                "ownerTable": "dwd_merchant_appeal_detail_df",
                "sourcePhrase": "申诉成功率",
                "groupByColumn": "pt",
                "objectiveType": "trend_anchor",
                "limit": 60,
            },
            "timeWindowDays": 60,
        },
        pack,
    )
    compiled_metrics = {intent.metric_name for intent in plan.intents}
    assert {"appeal_success_rate", "appeal_pass_cnt", "appeal_cnt"} <= compiled_metrics
    assert any("DERIVED_METRIC:appeal_success_rate" in item for item in plan.compiler_trace)


def test_understanding_coverage_critic_completes_missing_parallel_metric():
    settings = get_settings()
    question = "最近60天申诉成功率和退款率是否有关"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.MERCHANT_OTHER, QuestionCategory.REFUND],
    )
    understanding = {
        "analysisGrain": "day",
        "analysisIntent": "comparison",
        "rankingObjective": {
            "metricRef": "refund_rate",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款率",
            "groupByColumn": "pt",
            "objectiveType": "trend_anchor",
            "limit": 60,
        },
        "requestedMeasures": [
            {"metricRef": "refund_rate", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款率"}
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "correlation_timeseries",
                "reason": "需要按天对齐最近60天的申诉成功率与退款率时间序列，以判断二者是否同向变化或存在相关性",
                "requiredLevel": "required",
                "suggestedMetricRefs": ["refund_rate"],
            }
        ],
        "timeWindowDays": 60,
    }
    result = UnderstandingCoverageCritic().complete(question, understanding, pack)
    added = {(item["ownerTable"], item["metricRef"]) for item in result.added_measures}
    assert ("dwd_merchant_appeal_detail_df", "appeal_success_rate") in added

    plan = compile_query_graph_from_understanding(question, result.understanding, pack)
    compiled_metrics = {intent.metric_name for intent in plan.intents}
    assert {"refund_rate", "appeal_success_rate", "appeal_pass_cnt", "appeal_cnt"} <= compiled_metrics
    assert any("REQUESTED_MEASURE_APPENDED:dwd_merchant_appeal_detail_df:appeal_success_rate" in item for item in plan.compiler_trace)


def test_understanding_coverage_critic_does_not_add_implicit_refund_rate():
    settings = get_settings()
    question = "最近 30 天退款金额最高的商品，看看对应下单量是否也高。"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    understanding = {
        "analysisGrain": "product",
        "analysisIntent": "comparison",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款金额",
            "groupByColumn": "spu_id",
            "objectiveType": "ranking",
            "order": "desc",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "下单量"}
        ],
        "requiredEvidenceIntents": [],
        "timeWindowDays": 30,
    }

    result = UnderstandingCoverageCritic().complete(question, understanding, pack)

    requested = {
        (item.get("ownerTable"), item.get("metricRef"))
        for item in result.understanding.get("requestedMeasures", [])
        if isinstance(item, dict)
    }
    assert ("dwm_trade_refund_detail_di", "refund_rate") not in requested
    assert any("UNDERSTANDING_COVERAGE_SKIP_DERIVED_IMPLICIT:dwm_trade_refund_detail_di:refund_rate" in item for item in result.trace)


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
        plan, requests, _ = planner.plan(question, [], "", recall_bundle_empty(), pack, [], [])
        if requests:
            assert all(request.type == KnowledgeRequestType.METRIC for request in requests), question
            assert any("planner=llm_understanding_needs_semantic_metric_evidence" == item for item in plan.agent_trace), question
            continue
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
    plan = compile_query_graph_from_understanding(
        question,
        {
            "analysisGrain": "product",
            "rankingObjective": {
                "metricRef": "inbound_cnt",
                "ownerTable": "dwm_scm_detail_di",
                "sourcePhrase": "入库量",
                "groupByColumn": "spu_id",
                "limit": 10,
                "order": "desc",
            },
            "requestedMeasures": [
                {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "下单表现"},
                {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款表现"},
            ],
            "timeWindowDays": 45,
        },
        pack,
    )
    task_table = {intent.plan_task_id: intent.preferred_table for intent in plan.intents}
    refund_task_ids = {task_id for task_id, table in task_table.items() if table == "dwm_trade_refund_detail_di"}
    assert refund_task_ids
    bridge = next(intent for intent in plan.intents if intent.preferred_table == "dwm_trade_order_detail_di" and intent.answer_mode == AnswerMode.DETAIL)
    assert "sub_order_id" in bridge.output_keys
    for dep in plan.dependencies:
        if dep.dependent_task_id in refund_task_ids:
            assert dep.anchor_task_id == bridge.plan_task_id
            assert task_table[dep.anchor_task_id] == "dwm_trade_order_detail_di"
            assert dep.anchor_column == "seller_id+sub_order_id"
    assert any("INSERT_ENTITY_BRIDGE" in item for item in plan.compiler_trace)
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_compiler_ignores_metric_objective_declared_as_scope_constraint():
    question = "上个月销售额最高的前3个商品，以及这些商品最近7天退款量是多少？"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "spu_id", "product_amt", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "spu_name", "refund_id", "pt"]),
            PlanningAssetEntry(table="dwm_goods_detail_df", columns=["seller_id", "spu_id", "spu_apply_create_time", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="product_amt", table="dwm_trade_order_detail_di", columns=["product_amt"], title="销售额", aliases=["销售额", "GMV"]),
            PlanningAssetEntry(key="refund_bill_cnt", table="dwm_trade_refund_detail_di", columns=["refund_id"], title="退款量"),
            PlanningAssetEntry(key="goods_cnt", table="dwm_goods_detail_df", columns=["spu_id"], title="商品数"),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="order_goods_by_spu",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_goods_detail_df",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            ),
        ],
    )
    plan = compile_query_graph_from_understanding(
        question,
        {
            "analysisGrain": "product",
            "rankingObjective": {
                "metricRef": "product_amt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "上个月销售额最高的前3个商品",
                "objectiveType": "ranking",
                "groupByColumn": "spu_id",
                "limit": 3,
                "order": "desc",
            },
            "requestedMeasures": [
                {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "这些商品最近7天退款量"}
            ],
            "scopeConstraints": [
                {
                    "sourcePhrase": "上个月销售额最高的前3个商品",
                    "ownerTable": "dwm_trade_order_detail_di",
                    "metricRef": "product_amt",
                    "entityGrain": "product",
                    "targetDomain": "refund",
                    "required": True,
                },
                {
                    "sourcePhrase": "这些商品最近7天退款量",
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "metricRef": "refund_bill_cnt",
                    "entityGrain": "product",
                    "targetDomain": "refund",
                    "required": True,
                },
            ],
            "timeWindowDays": 7,
        },
        pack,
    )
    result = QueryGraphValidator().validate(question, plan, pack)
    assert result.valid
    assert not any(intent.plan_task_id.endswith("_scope") for intent in plan.intents)
    assert any(item.startswith("SCOPE_SKIPPED_NOT_POPULATION:ranking_objective") for item in plan.compiler_trace)
    assert any(item.startswith("SCOPE_SKIPPED_NOT_POPULATION:requested_measure") for item in plan.compiler_trace)
    bridge = next(intent for intent in plan.intents if intent.preferred_table == "dwm_trade_order_detail_di" and intent.answer_mode == AnswerMode.DETAIL)
    assert "sub_order_id" in bridge.output_keys


def test_compiler_canonicalizes_day_grain_to_pt():
    question = "最近30天保证金充值流水、申诉次数和处罚次数分别是多少，有没有异常？"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwd_merchant_deposit_recharge_df", columns=["merchant_id", "deposit_recharge_id", "deposit_recharge_amt", "create_time", "pt"]),
            PlanningAssetEntry(table="dwd_merchant_appeal_detail_df", columns=["merchant_id", "appeal_id", "create_time", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="deposit_recharge_cnt", table="dwd_merchant_deposit_recharge_df", columns=["deposit_recharge_id"], title="保证金充值流水"),
            PlanningAssetEntry(key="deposit_recharge_amt", table="dwd_merchant_deposit_recharge_df", columns=["deposit_recharge_amt"], title="保证金充值金额"),
            PlanningAssetEntry(key="appeal_cnt", table="dwd_merchant_appeal_detail_df", columns=["appeal_id"], title="申诉次数"),
        ],
    )
    plan = compile_query_graph_from_understanding(
        question,
        {
            "analysisGrain": "day",
            "analysisIntent": "anomaly_check",
            "rankingObjective": {
                "metricRef": "deposit_recharge_cnt",
                "ownerTable": "dwd_merchant_deposit_recharge_df",
                "sourcePhrase": "保证金充值流水",
                "objectiveType": "trend_anchor",
                "groupByColumn": "create_time",
                "limit": 30,
            },
            "requestedMeasures": [
                {"metricRef": "appeal_cnt", "ownerTable": "dwd_merchant_appeal_detail_df", "sourcePhrase": "申诉次数", "groupByColumn": "create_time"},
                {"metricRef": "deposit_recharge_amt", "ownerTable": "dwd_merchant_deposit_recharge_df", "sourcePhrase": "保证金充值金额", "groupByColumn": "create_time"},
            ],
            "timeWindowDays": 30,
        },
        pack,
    )
    assert QueryGraphValidator().validate(question, plan, pack).valid
    assert {intent.group_by_column for intent in plan.intents if intent.answer_mode == AnswerMode.GROUP_AGG} == {"pt"}


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


def test_asset_pack_uses_keyword_targeted_topk_tables_for_multi_domain_question():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近60天赔付金额最高的前5个订单，关联看订单金额、退款金额、退款状态和对应客服工单情况。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [
            QuestionCategory.COMPENSATION,
            QuestionCategory.TRADE,
            QuestionCategory.REFUND,
            QuestionCategory.CS_TICKET,
            QuestionCategory.GOODS,
            QuestionCategory.SCM,
        ],
    )
    tables = set(pack.known_tables())
    assert tables == {
        "dwm_cs_repay_detail_df",
        "dwm_trade_order_detail_di",
        "dwm_trade_refund_detail_di",
        "dwm_cs_ticket_detail_di",
    }
    assert "ads_merchant_profile" not in tables
    assert "dwm_goods_detail_df" not in tables
    assert any("targeted_seed_tables:" in item for item in pack.relationship_closure)

    catalog = ultra_compact_understanding_catalog(pack, question)
    metrics = {(item["table"], item["key"]): item for item in catalog["candidateMetrics"]}
    assert ("dwm_cs_repay_detail_df", "repay_amt") in metrics
    assert "赔付金额" in metrics[("dwm_cs_repay_detail_df", "repay_amt")]["matchedPhrases"]


def test_ultra_catalog_keeps_candidate_metric_coverage_per_involved_table():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [
            QuestionCategory.SCM,
            QuestionCategory.TRADE,
            QuestionCategory.REFUND,
            QuestionCategory.GOODS,
        ],
    )

    catalog = ultra_compact_understanding_catalog(pack, question)
    metrics = {(item["table"], item["key"]) for item in catalog["candidateMetrics"]}

    assert ("dwm_scm_detail_di", "scm_inbound_total_cnt") in metrics
    assert ("dwm_trade_order_detail_di", "order_detail_cnt") in metrics
    assert ("dwm_trade_refund_detail_di", "refund_bill_cnt") in metrics
    assert len(catalog["candidateMetrics"]) <= min(
        settings.agent_planner_seed_metric_limit,
        max(6, len(catalog["tables"]) * 2 + 2),
    )


def test_ultra_catalog_preserves_recalled_metric_without_local_rescoring():
    settings = get_settings()
    question = "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间。"
    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate",
                title="电商退货/dwm_trade_refund_detail_di/refund_rate metric",
                content="商品退款率 退款率 退货率 refund_rate",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=151.0,
                metadata={
                    "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate",
                    "metricKey": "refund_rate",
                    "tableName": "dwm_trade_refund_detail_di",
                    "businessName": "商品退货率",
                    "formula": "退货量 / 订单量",
                    "sourceColumns": ["refund_bill_cnt", "order_detail_cnt"],
                    "aliases": ["商品退款率", "退款率", "退货率", "售后率"],
                    "recallQuery": question,
                },
            ),
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:pay_amt",
                title="电商退货/dwm_trade_refund_detail_di/pay_amt metric",
                content="退款金额 pay_amt",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=100.0,
                metadata={
                    "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:pay_amt",
                    "metricKey": "pay_amt",
                    "tableName": "dwm_trade_refund_detail_di",
                    "businessName": "退款金额",
                    "formula": "SUM(pay_amt)",
                    "sourceColumns": ["pay_amt"],
                    "aliases": ["退款金额", "refund_amt"],
                    "recallQuery": question,
                },
            ),
        ]
    )
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(question, recall, [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS])

    catalog = ultra_compact_understanding_catalog(pack, question)
    metrics = {(item["table"], item["key"]): item for item in catalog["candidateMetrics"]}

    assert ("dwm_trade_refund_detail_di", "refund_rate") in metrics
    assert "退款率" in metrics[("dwm_trade_refund_detail_di", "refund_rate")]["matchedPhrases"]


def test_planner_fast_path_uses_small_context_package():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    planner = QueryGraphPlanner(FakeCaseUnderstandingLlm(), settings=settings)
    payload = planner._understanding_payload(question, pack, [], [], False, False, None)
    user_prompt = json.dumps(payload, ensure_ascii=False)
    compact_schema = compact_openai_tool_schema(question_understanding_tool(False).openai_schema())
    full_schema = question_understanding_tool(False).openai_schema()
    stats = planner_prompt_stats("system", user_prompt, compact_schema)
    assert "semanticFileContext" not in payload
    assert "semanticManifest" not in payload
    assert len(payload["semanticCatalog"]["tables"]) <= settings.agent_planner_seed_table_limit
    assert len(user_prompt) < settings.agent_planner_prompt_budget_chars
    assert stats["toolSchemaChars"] < len(json.dumps(full_schema, ensure_ascii=False))
    tool_payload = planner._understanding_payload(question, pack, [], [], False, False, None, include_full_file_context=True)
    workspace = tool_payload["semanticWorkspace"]
    assert workspace["mode"] == "filesystem_as_context"
    assert "semanticManifest" not in tool_payload
    assert workspace["manifestRefs"]
    assert len(workspace["tableRefs"]) <= settings.agent_planner_seed_table_limit
    assert tool_payload["semanticFileContext"] == workspace
    generated_workspace = semantic_workspace_manifest_from_asset_pack(pack, limit=settings.agent_planner_seed_table_limit)
    assert generated_workspace["tools"]


def test_ultra_catalog_keeps_cross_domain_metric_refs_without_expanding_tables():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [
            QuestionCategory.TRADE,
            QuestionCategory.REFUND,
            QuestionCategory.CS_TICKET,
            QuestionCategory.COMPENSATION,
        ],
    )
    catalog = ultra_compact_understanding_catalog(pack, question)
    table_names = {item["table"] for item in catalog["tables"]}
    metric_tables = {item["table"] for item in catalog["candidateMetrics"]}
    metric_keys = {item["key"] for item in catalog["candidateMetrics"]}
    assert 1 <= len(table_names) <= settings.agent_planner_seed_table_limit
    assert "dwm_cs_repay_detail_df" in metric_tables
    assert {"repay_amt", "ticket_cnt", "pay_amt"} & metric_keys


def test_successful_planner_fast_path_does_not_call_recovery_llm():
    class CountingFastPlannerLlm(FakeCaseUnderstandingLlm):
        def __init__(self):
            self.calls = 0

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            self.calls += 1
            return super().json_chat(system_prompt, user_prompt, fallback)

    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    llm = CountingFastPlannerLlm()
    plan, requests, _ = QueryGraphPlanner(llm, settings=settings).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert plan.intents
    assert not requests
    assert llm.calls == 1
    assert plan.planner_prompt_stats["schemaMode"] == "compact_tool_schema"


def test_planner_auto_compacts_catalog_before_context_over_budget():
    base_settings = get_settings()
    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    builder = PlanningAssetPackBuilder(TopicAssetService(base_settings), SkillLoader(base_settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [
            QuestionCategory.TRADE,
            QuestionCategory.REFUND,
            QuestionCategory.GOODS,
        ],
    )
    planner = QueryGraphPlanner(FakeCaseUnderstandingLlm(), settings=base_settings.model_copy(update={"agent_planner_prompt_budget_chars": 100000}))
    prompt = planner.prompt_assembler.render("planner.question_understanding", sections={"context_policy": "Planner fast path"})
    tool_schema = compact_openai_tool_schema(question_understanding_tool(False).openai_schema())
    level0_payload = planner._understanding_payload(question, pack, [], [], False, False, None, budget_level=0)
    level2_payload = planner._understanding_payload(question, pack, [], [], False, True, None, budget_level=2)
    level0_total = planner_prompt_stats(prompt.system_prompt, json.dumps(level0_payload, ensure_ascii=False), tool_schema)["totalChars"]
    level2_total = planner_prompt_stats(prompt.system_prompt, json.dumps(level2_payload, ensure_ascii=False), tool_schema)["totalChars"]
    assert level0_total > level2_total

    budget = min(level0_total - 100, level2_total + 1200)
    assert level2_total < budget < level0_total
    settings = base_settings.model_copy(update={"agent_planner_prompt_budget_chars": budget})
    plan, requests, reason = QueryGraphPlanner(FakeCaseUnderstandingLlm(), settings=settings).plan(
        question,
        [],
        "",
        recall_bundle_empty(),
        pack,
        [],
        [],
    )

    assert plan.intents
    assert not requests
    assert reason != "PLANNER_CONTEXT_OVER_BUDGET"
    assert plan.planner_prompt_stats["budgetLevel"] >= 1
    assert plan.planner_prompt_stats["totalChars"] <= budget
    assert plan.planner_prompt_stats["budgetTrace"][0]["totalChars"] > budget


def test_planner_reflection_uses_resolved_metric_for_anchor_check():
    question = "最近30天GMV最高的前5天。"
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "pay_amt",
                "ownerTable": "dwm_trade_order_detail_di",
            }
        },
        intents=[
            QuestionIntent(
                question=question,
                answer_mode="TOPN",
                plan_task_id="anchor_profile",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                group_by_column="pt",
                output_keys=["merchant_id", "pt"],
                metric_resolution={
                    "requestedMetricRef": "pay_amt",
                    "metricKey": "order_gmv_amt_1d",
                    "ownerTable": "ads_merchant_profile",
                },
            )
        ],
    )
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="ads_merchant_profile", columns=["merchant_id", "pt", "order_gmv_amt_1d"])],
        metrics=[
            PlanningAssetEntry(
                key="order_gmv_amt_1d",
                table="ads_merchant_profile",
                columns=["order_gmv_amt_1d"],
                title="GMV",
            )
        ],
    )

    reflection = PlannerReflectionAgent().reflect(question, plan, pack)

    assert not any(issue.get("code") == "ANCHOR_MISMATCH" for issue in reflection.issues)


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
    assert len(pack.metrics) <= settings.agent_planner_seed_metric_limit
    assert pack.metric_compaction["targetedSeed"]["tables"]
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


def test_keyword_extractor_structures_metrics_dimensions_ranking_and_weighted_topics():
    settings = get_settings()
    service = KeywordExtractService(TopicAssetService(settings))

    keywords = service.extract("最近7天退款金额最高的5个商品，同时看订单量")

    assert keywords.metric_keywords == ["退款金额", "订单量"]
    assert keywords.dimension_keywords == ["商品"]
    assert keywords.ranking_keywords == ["最高"]
    assert keywords.analysis_intent == "ranking"
    assert keywords.confidence >= 0.8
    assert keywords.topic_scores[QuestionCategory.REFUND.value] > 0
    assert keywords.topic_scores[QuestionCategory.TRADE.value] > 0
    assert keywords.topic_scores[QuestionCategory.GOODS.value] > 0
    mention_keys = {(item.kind, item.canonical_key) for item in keywords.mentions}
    assert ("metric", "pay_amt") in mention_keys
    assert ("metric", "order_detail_cnt") in mention_keys
    assert ("dimension", "spu_id") in mention_keys


def test_keyword_extractor_uses_longest_metric_match_and_keeps_domain_nouns_out_of_metrics():
    service = KeywordExtractService(TopicAssetService(get_settings()))

    amount = service.extract("最近7天退款金额趋势")
    detail = service.extract("查询这个子订单的退款情况")
    unrelated = service.extract("证券行情怎么样")

    assert amount.metric_keywords == ["退款金额"]
    assert "退款" not in amount.metric_keywords
    assert detail.metric_keywords == []
    assert QuestionCategory.REFUND.value in detail.topic_scores
    assert QuestionCategory.COUPON.value not in unrelated.topic_scores


def test_keyword_extractor_marks_context_references_and_stays_fast():
    service = KeywordExtractService(TopicAssetService(get_settings()))
    started = time.perf_counter()

    keywords = service.extract("结合上述明细分析原因并给建议")
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert keywords.analysis_intent == "attribution"
    assert "上述" in keywords.unresolved_phrases
    assert keywords.confidence < 0.5
    assert elapsed_ms < 30


def test_structured_keywords_prevent_dimension_ranking_question_from_using_quick_metric_path():
    service = KeywordExtractService(TopicAssetService(get_settings()))
    question = "最近7天退款金额最高的5个商品，同时看订单量"
    keywords = service.extract(question)

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("dimension/ranking question must be handled by Planner")

    assert quick_metric_response(question, "100", UnexpectedRepository(), keywords) is None


def test_ratio_and_partial_metric_coverage_fall_back_to_planner():
    service = KeywordExtractService(TopicAssetService(get_settings()))

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("unsupported fast request must fall back before querying")

    ratio = "最近7天退款金额占GMV比例"
    coupon = "最近7天优惠券活动投入和支付订单数"

    ratio_keywords = service.extract(ratio)
    coupon_keywords = service.extract(coupon)
    assert ratio_keywords.analysis_intent == "ratio"
    assert quick_metric_response(ratio, "100", UnexpectedRepository(), ratio_keywords) is None
    assert quick_metric_response(coupon, "100", UnexpectedRepository(), coupon_keywords) is None


def test_attribution_question_never_uses_quick_metric_path():
    service = KeywordExtractService(TopicAssetService(get_settings()))
    question = "最近7天GMV为什么下降？"
    keywords = service.extract(question)

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("attribution must use semantic retrieval and Planner")

    assert quick_metric_response(question, "100", UnexpectedRepository(), keywords) is None


def test_quick_metric_falls_back_when_published_semantics_are_ambiguous():
    assets = TopicAssetService(get_settings())
    semantic_metrics = published_semantic_quick_metrics(assets)
    keywords = KeywordExtractService(assets).extract("最近7天GMV趋势")

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("ambiguous GMV semantics must fall back to Planner")

    assert quick_metric_response("最近7天GMV趋势", "100", UnexpectedRepository(), keywords, semantic_metrics) is None


def test_quick_metric_uses_unique_published_semantic_formula():
    assets = TopicAssetService(get_settings())
    semantic_metrics = published_semantic_quick_metrics(assets)
    keywords = KeywordExtractService(assets).extract("最近7天总GMV趋势")
    calls = []

    class Repository:
        def query(self, sql, params=None):
            calls.append((sql, params))
            return [{"pt": "2026-07-10", "value": 100.0}, {"pt": "2026-07-11", "value": 120.0}]

    response = quick_metric_response("最近7天总GMV趋势", "100", Repository(), keywords, semantic_metrics)

    assert response is not None
    disclosure = response.merchant_experience["metricDisclosures"][0]
    assert disclosure["metricKey"] == "order_gmv_amt_1d"
    assert disclosure["formula"] == "SUM(order_gmv_amt_1d)"
    assert "order_gmv_amt_1d" in calls[0][0]
    assert response.debug_trace["semanticMetric"]["topic"] == "经营画像"


def test_negation_ambiguity_and_false_substring_topic_are_structured():
    service = KeywordExtractService(TopicAssetService(get_settings()))

    unrelated = service.extract("证券行情怎么样")
    negated = service.extract("最近7天订单量，不看退款")
    ambiguous = service.extract("最近7天支付金额趋势")
    ambiguous_slots = RouteSlotExtractor().extract("最近7天支付金额趋势", ambiguous)

    assert QuestionCategory.COUPON.value not in unrelated.topic_scores
    assert QuestionCategory.REFUND in negated.excluded_topics
    assert QuestionCategory.REFUND.value not in negated.topic_scores
    assert ambiguous.ambiguous_metric_keywords == ["支付金额"]
    assert ambiguous.confidence < 0.8
    assert "AMBIGUOUS_METRIC" in ambiguous_slots.route_warnings


def test_topic_router_inherits_multiple_context_topics_without_joining_them():
    topics = [QuestionCategory.TRADE, QuestionCategory.REFUND]
    decision = TopicRouterService().route(
        "那最近30天呢",
        KeywordExtractService().extract("那最近30天呢"),
        context_topics=topics,
    )

    assert decision.candidate_topics == topics
    assert decision.primary_topic == QuestionCategory.UNKNOWN


def test_fast_metric_is_a_main_agent_action_and_preserves_response_sections(monkeypatch):
    workflow = create_workflow(get_settings().model_copy(update={"lead_action_llm_mode": "off"}))
    state = workflow._initial_state("最近7天总GMV趋势", "100", ChatContext(), None, "fast_thread", "fast_run")
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.fast_understand(state)

    monkeypatch.setattr(
        workflow.node_worker.doris_repository,
        "query",
        lambda *_args, **_kwargs: [
            {"pt": "2026-07-10", "value": 100.0},
            {"pt": "2026-07-11", "value": 120.0},
        ],
    )
    decision = workflow.policy.decide(state)
    assert decision.selected_action == "try_fast_metric"

    state = workflow.policy_node(state)
    assert state["action_history"][-1].action == "try_fast_metric"
    state = workflow.try_fast_metric(state)
    assert state["fast_metric_completed"]
    assert state["fast_metric_response"].data_sections

    state = workflow.cache_answer(state)
    response = workflow.to_response(state)
    assert response.id == state["qa_id"]
    assert response.data_sections
    assert response.context.metric_keys == ["order_gmv_amt_1d"]
    assert response.context.dimension_keys == ["pt"]
    assert response.debug_trace["leadAgentFastDecision"] is True


def test_multi_topic_route_does_not_assign_fixed_order_primary_topic():
    question = "最近30天退款金额最高的前5个商品，同时看这些商品的下单量和商品发布时间，帮我判断哪些是高风险新品。"
    keywords = KeywordExtractService().extract(question)
    slots = RouteSlotExtractor().extract(question, keywords)
    topic = TopicRouterService().route(question, keywords, route_slots=slots)

    assert topic.primary_topic == QuestionCategory.UNKNOWN
    assert topic.recall_topics() == [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS]
    assert "primaryTopic 保持 UNKNOWN" in topic.reason


def test_first_step_topic_router_covers_coupon_activity_synonym():
    question = "最近30天券活动投入最高的活动，带来的支付订单数怎么样？"
    keywords = KeywordExtractService().extract(question)
    topic = TopicRouterService().route(question, keywords)

    assert QuestionCategory.COUPON in set(topic.candidate_topics)
    assert QuestionCategory.TRADE in set(topic.candidate_topics)


def test_route_slots_extract_object_refs_without_entity_pollution():
    question = "查询子订单 sub_order_id_100 和订单 order_id_200 的退款情况，并看 spu_id_300 商品。"
    keywords = KeywordExtractService().extract(question)
    slots = RouteSlotExtractor().extract(question, keywords)

    refs = {(item.ref_type, item.value) for item in slots.object_refs}
    assert ("sub_order_id", "sub_order_id_100") in refs
    assert ("order_id", "order_id_200") in refs
    assert ("spu_id", "spu_id_300") in refs
    assert ("order_id", "order_id_100") not in refs
    assert {item.topic for item in slots.topic_candidates} >= {QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS}


def test_route_slots_extract_time_operation_risk_and_weak_analysis_hint():
    read_slots = RouteSlotExtractor().extract(
        "昨天平台规则里处罚和保证金有什么风险？看下趋势。",
        KeywordExtractService().extract("昨天平台规则里处罚和保证金有什么风险？看下趋势。"),
    )
    assert read_slots.time_window.days == 1
    assert read_slots.time_window.needs_freshness_check
    assert read_slots.operation == "read"
    assert read_slots.risk_level == "rule_sensitive"
    assert read_slots.analysis_signals == ["weak_analysis_hint"]

    write_slots = RouteSlotExtractor().extract("删除最近30天订单数据", KeywordExtractService().extract("删除最近30天订单数据"))
    assert write_slots.operation == "write_requested"
    assert write_slots.risk_level == "high_risk"

    compensation_slots = RouteSlotExtractor().extract(
        "最近30天赔付金额最高的订单有哪些？",
        KeywordExtractService().extract("最近30天赔付金额最高的订单有哪些？"),
    )
    assert compensation_slots.risk_level == "normal"
    assert QuestionCategory.COMPENSATION in {item.topic for item in compensation_slots.topic_candidates}


def test_route_topic_blocks_write_operation_before_retrieval():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("删除最近30天订单数据", "100", ChatContext(), None, "write_thread", "write_run")
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)

    assert state["route_slots"].operation == "write_requested"
    assert state["human_clarification_required"]
    assert state["human_clarification_type"] == "write_operation"
    assert workflow.policy.decide(state).selected_action == "ask_human"
    state = workflow.policy_node(state)
    assert state["_next_action"] == "human_in_loop"
    assert state["clarification_tool_message"]["toolName"] == "ask_clarification"
    assert state["clarification_command"]["goto"] == "END"
    assert state["tool_call_results"][-1].name == "ask_clarification"


def test_bounded_route_llm_can_only_filter_allowed_topics(tmp_path):
    class FilteringRouteLlm:
        configured = True
        last_error = ""

        def json_chat(self, system_prompt, user_prompt, fallback=None, timeout_seconds=None):
            return {"topics": ["TRADE", "UNKNOWN", "COUPON"], "confidence": 0.91, "reason": "keep trade only"}

    settings = get_settings().model_copy(update={"route_llm_mode": "always", "harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    workflow.planner.llm = FilteringRouteLlm()
    state = workflow._initial_state("最近7天订单和退款情况", "100", ChatContext(), None, "route_llm_thread", "route_llm_run")
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)

    assert state["bounded_route_llm_trace"]["status"] == "applied"
    assert state["topic_routing_decision"].candidate_topics == [QuestionCategory.TRADE]
    assert "COUPON" not in [getattr(item, "value", str(item)) for item in state["topic_routing_decision"].candidate_topics]


def test_bounded_route_llm_timeout_keeps_deterministic_route(tmp_path):
    class TimeoutRouteLlm:
        configured = True
        last_error = "timeout: provider call exceeded 8 seconds"

        def json_chat(self, system_prompt, user_prompt, fallback=None, timeout_seconds=None):
            return {}

    settings = get_settings().model_copy(update={"route_llm_mode": "always", "harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    workflow.planner.llm = TimeoutRouteLlm()
    state = workflow._initial_state("最近7天订单和退款情况", "100", ChatContext(), None, "route_timeout_thread", "route_timeout_run")
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)

    assert state["bounded_route_llm_trace"]["errorCode"] == "ROUTE_LLM_TIMEOUT"
    assert "ROUTE_LLM_TIMEOUT" in state["route_slots"].route_warnings
    assert {QuestionCategory.TRADE, QuestionCategory.REFUND} <= set(state["topic_routing_decision"].candidate_topics)


def test_compact_assets_filters_weak_tables_inside_broad_merchant_topic():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))

    deposit_pack = builder.compact(
        "最近60天保证金充值金额变化和GMV变化是否同步？",
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.MERCHANT_OTHER],
    )
    assert "dwd_merchant_deposit_recharge_df" in deposit_pack.known_tables()
    assert "dwd_merchant_appeal_detail_df" not in deposit_pack.known_tables()

    appeal_pack = builder.compact(
        "最近30天申诉成功的订单多吗？关联看有没有退款。",
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.MERCHANT_OTHER],
    )
    assert "dwd_merchant_appeal_detail_df" in appeal_pack.known_tables()
    assert "dwd_merchant_deposit_recharge_df" not in appeal_pack.known_tables()


def test_route_analysis_hint_does_not_become_final_analysis_intent():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "最近30天签收订单数和订单金额走势怎么样？",
        "100",
        ChatContext(),
        None,
        "intent_signal_thread",
        "intent_signal_run",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.retrieve_knowledge(state)

    assert not state["intent_signals"].has_analysis_intent
    assert "route_analysis_hint_present" in state["intent_signals"].observations
    assert "analysis_intent_signal_present" not in state["intent_signals"].observations


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


def test_platform_rule_question_uses_recall_then_rule_answer_plan(monkeypatch):
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("平台规则里商家发货超时会怎么处罚？", "100", ChatContext(), None, "", "")
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    assert QuestionCategory.PLATFORM_RULE in state["topic_routing_decision"].recall_topics()

    state = workflow.retrieve_knowledge(state)

    assert state["data_discovered"]
    assert state["recall_bundle"].items
    assert any("rule" in (item.answer_mode or "").lower() for item in state["recall_bundle"].items[:6])
    assert state["intent_signals"].has_rule_evidence
    assert not state["intent_signals"].has_data_intent
    assert state["intent_signals"].suggested_actions == ["answer_rule"]
    assert state["rule_recall_ready"]
    assert state["rule_recall_refs"]
    assert state["rule_recall_context"]
    assert not state["plan"].intents
    decision = workflow.policy.decide(state)
    assert decision.selected_action == "answer_rule"
    assert decision.selected_node == "answer_rule"

    monkeypatch.setattr(workflow.answer_service, "compose", lambda *args, **kwargs: "规则召回答案")
    state = workflow.answer_rule(state)
    assert state["plan"].intents
    assert state["plan"].intents[0].answer_mode == AnswerMode.RULE
    assert state["plan"].intents[0].analysis_source == "rule_recall"
    assert state["plan"].intents[0].knowledge_ref_ids
    assert state["query_graph_validation_result"].valid
    assert not state["query_graph_validation_result"].repairable
    assert state["chat_bi_completed"]


def test_rule_recall_does_not_short_circuit_bi_entity_lookup():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。",
        "100",
        ChatContext(),
        None,
        "",
        "",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.retrieve_knowledge(state)

    assert state["recall_bundle"].items
    assert not state["intent_signals"].has_rule_evidence
    assert state["intent_signals"].has_data_intent
    assert not state["rule_recall_ready"]
    decision = workflow.policy.decide(state)
    assert decision.selected_action != "answer_rule"


def test_rule_recall_is_retained_as_evidence_for_mixed_rule_bi_question():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
        "100",
        ChatContext(),
        None,
        "",
        "",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.retrieve_knowledge(state)

    assert state["recall_bundle"].items
    assert state["intent_signals"].has_rule_evidence
    assert state["intent_signals"].has_data_intent
    assert "compact_assets" in state["intent_signals"].suggested_actions
    assert state["rule_recall_refs"]
    assert state["rule_recall_context"]
    assert not state["rule_recall_ready"]
    decision = workflow.policy.decide(state)
    assert decision.selected_action != "answer_rule"
    assert decision.selected_action == "compact_assets"


def test_intent_signals_support_implicit_rule_question_after_recall():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state(
        "商家发货超时会怎么处罚？",
        "100",
        ChatContext(),
        None,
        "",
        "",
    )
    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.retrieve_knowledge(state)

    assert state["intent_signals"].has_rule_evidence
    assert not state["intent_signals"].has_data_intent
    assert state["rule_recall_ready"]
    assert workflow.policy.decide(state).selected_action == "answer_rule"


def test_answer_compose_merges_rule_evidence_with_bi_data():
    settings = get_settings().model_copy(update={"llm_api_key": ""})
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "none",
            "requiresExplanation": False,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "platform_rule_shipping_timeout",
                    "requiredLevel": "required",
                    "suggestedDomains": ["rule", "governance"],
                }
            ],
        },
        intents=[
            QuestionIntent(
                question="平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"pt": "2026-06-20", "ship_timeout_order_cnt_1d": 3, "merchant_id": "100"}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="ship_timeout_metric", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )
    rule_context = """
召回规则片段 [BASE_WIKI] yshopping 平台商家规则
- 商家发货时应确保订单状态、发货时效、物流单号和承运信息一致，避免虚假发货、超时发货或单号无轨迹。
- 若用户咨询“发货规则”“超时发货怎么办”，应重点回答发货时效、物流信息准确性和异常订单排查路径。
"""

    answer = service.compose("平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。", MerchantInfo(merchant_id="100"), plan, run, "", rule_context=rule_context)

    assert "发货超时" in answer or "订单" in answer
    assert "已按当前口径" not in answer
    assert "规则依据" in answer
    assert "发货时效" in answer
    assert compact_rule_evidence("发货超时要注意什么", rule_context)


def test_answer_skill_does_not_use_skill_for_ordinary_scope_event_ratio(tmp_path):
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "comparison",
            "requiresExplanation": True,
            "calculationIntents": [
                {
                    "operation": "ratio",
                    "basePopulationPhrase": "使用优惠券的订单",
                    "eventPopulationPhrase": "有退货的订单",
                    "denominatorMetricRef": "order_detail_cnt",
                    "numeratorMetricRef": "refund_bill_cnt",
                }
            ],
        },
        intents=[
            QuestionIntent(
                question="最近10天使用优惠券的订单中，有退货的订单占多少？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                plan_task_id="derived_refund_share",
                metric_name="refund_bill_cnt_share_of_order_detail_cnt",
                metric_formula="refund_bill_cnt / order_detail_cnt",
                metric_resolution={"computeStrategy": "scope_event_ratio", "formula": "refund_bill_cnt / order_detail_cnt"},
            )
        ],
    )
    bundle = QueryBundle(
        rows=[{"order_detail_cnt": 10, "refund_bill_cnt": 2, "refund_bill_cnt_share_of_order_detail_cnt": 0.2}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="derived_refund_share", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    assert select_answer_skill(plan, run, False) == ""


def test_answer_skill_uses_declared_reusable_ratio_workflow(tmp_path):
    settings = get_settings().model_copy(update={"llm_api_key": "", "harness_workspace_path": str(tmp_path)})
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "comparison",
            "requiresExplanation": True,
            "skillWorkflow": {"skillName": "ratio_analysis", "required": True},
            "calculationIntents": [{"operation": "ratio"}],
        },
        intents=[
            QuestionIntent(
                question="固定占比分析流程",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                plan_task_id="derived_refund_share",
                metric_name="refund_bill_cnt_share_of_order_detail_cnt",
                metric_formula="refund_bill_cnt / order_detail_cnt",
                metric_resolution={"computeStrategy": "scope_event_ratio", "formula": "refund_bill_cnt / order_detail_cnt"},
            )
        ],
    )
    bundle = QueryBundle(
        rows=[{"order_detail_cnt": 10, "refund_bill_cnt": 2, "refund_bill_cnt_share_of_order_detail_cnt": 0.2}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="derived_refund_share", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    assert select_answer_skill(plan, run, False) == "ratio_analysis"
    answer = service.summarize_analysis("固定占比分析流程", plan, run, str(tmp_path))

    assert "占比分析" in answer
    assert "refund_bill_cnt / order_detail_cnt" in answer
    assert service.last_analysis_skill_trace["skillName"] == "ratio_analysis"


def test_answer_skill_selects_merchant_sop_skills_for_declared_workflows():
    gmv_plan = QueryPlan(
        question_understanding={
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "skillWorkflow": {"enabled": True},
            "question": "最近7天GMV为什么下降？",
        },
        intents=[
            QuestionIntent(
                question="最近7天GMV为什么下降？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv",
                metric_name="gmv",
            )
        ],
    )
    refund_plan = QueryPlan(
        question_understanding={
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "fixedAnalysisWorkflow": True,
            "question": "最近退款率为什么升高？",
        },
        intents=[
            QuestionIntent(
                question="最近退款率为什么升高？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_rate",
                metric_name="refund_rate",
            )
        ],
    )
    briefing_plan = QueryPlan(
        question_understanding={
            "analysisIntent": "overview",
            "requiresExplanation": True,
            "reusableAnalysis": True,
            "question": "生成今天的店铺经营简报",
        },
        intents=[
            QuestionIntent(question="交易", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC, category=QuestionCategory.TRADE),
            QuestionIntent(question="退款", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC, category=QuestionCategory.REFUND),
            QuestionIntent(question="客服", intent_type=IntentType.VALID, answer_mode=AnswerMode.METRIC, category=QuestionCategory.CS_TICKET),
        ],
    )

    assert select_answer_skill(gmv_plan, AgentRunResult(), False) == "gmv_drop_diagnosis"
    assert select_answer_skill(refund_plan, AgentRunResult(), False) == "refund_rate_diagnosis"
    assert select_answer_skill(briefing_plan, AgentRunResult(), False) == "merchant_daily_briefing"


def test_answer_skill_does_not_trigger_sop_for_plain_merchant_lookup():
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "none",
            "requiresExplanation": False,
            "question": "最近7天GMV是多少？",
        },
        intents=[
            QuestionIntent(
                question="最近7天GMV是多少？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv",
                metric_name="gmv",
            )
        ],
    )

    assert select_answer_skill(plan, AgentRunResult(), False) == ""


def test_structured_merchant_sop_skill_answers_bind_to_evidence():
    answer = render_structured_skill_answer(
        "refund_rate_diagnosis",
        {
            "dataRows": [{"pt": "2026-07-10", "refund_bill_cnt": 12, "order_detail_cnt": 100, "refund_rate": 0.12}],
            "metricDisclosures": [{"displayName": "退款率", "formula": "refund_bill_cnt / order_detail_cnt"}],
            "evidenceGaps": [{"code": "MISSING_REASON_BREAKDOWN", "reason": "缺少退款原因维度"}],
        },
    )

    assert "退款率升高归因" in answer
    assert "refund_bill_cnt / order_detail_cnt" in answer
    assert "缺少退款原因维度" in answer
    assert "缺任一侧证据时只提示排查方向" in answer


def test_answer_skill_does_not_use_skill_for_ordinary_risk_ratio_metric():
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "risk_ranking",
            "requiresExplanation": True,
            "analysisGrain": "product",
        },
        intents=[
            QuestionIntent(
                question="按风险排名",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                plan_task_id="derived_risk_metric",
                metric_name="refund_rate",
                metric_formula="refund_bill_cnt / order_detail_cnt",
                metric_resolution={"computeStrategy": "component_metric_ratio", "formula": "refund_bill_cnt / order_detail_cnt"},
                group_by_column="spu_id",
                output_keys=["seller_id", "spu_id", "refund_rate"],
            )
        ],
    )
    bundle = QueryBundle(rows=[{"spu_id": "spu_1", "refund_rate": 0.4}], original_row_count=1)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="derived_risk_metric", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    assert plan_has_ratio_calculation(plan)
    assert select_answer_skill(plan, run, False) == ""


def test_answer_skill_match_can_use_llm_header_selection(tmp_path):
    class HeaderMatchLlm:
        configured = True
        settings = get_settings().model_copy(
            update={
                "harness_workspace_path": str(tmp_path),
                "answer_skill_match_mode": "always",
            }
        )

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            assert "Answer Skill matcher" in system_prompt
            assert "ratio_analysis" in user_prompt
            return '{"skillName":"ratio_analysis","confidence":0.91,"reason":"ratio calculation"}'

    service = AnswerComposeService(HeaderMatchLlm())
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "comparison",
            "requiresExplanation": True,
            "skillWorkflow": {"enabled": True},
            "calculationIntents": [{"operation": "ratio", "numeratorMetricRef": "refund_bill_cnt", "denominatorMetricRef": "order_detail_cnt"}],
        },
        intents=[
            QuestionIntent(
                question="占比",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                metric_resolution={"computeStrategy": "scope_event_ratio", "formula": "refund_bill_cnt / order_detail_cnt"},
            )
        ],
    )
    bundle = QueryBundle(rows=[{"order_detail_cnt": 10, "refund_bill_cnt": 2}], original_row_count=1)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="ratio", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    answer = service.summarize_analysis("占比", plan, run, str(tmp_path))

    assert "占比分析" in answer
    assert service.last_analysis_skill_trace["matchedBy"] == "llm_skill_header_match"
    assert service.last_analysis_skill_trace["skillName"] == "ratio_analysis"


def test_semantic_fast_path_risk_skill_uses_local_renderer(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "skill_worker_enabled": True})
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "product",
            "analysisIntent": "risk_ranking",
            "requiresExplanation": True,
            "source": "semantic_topn_metric_fast_path",
        },
        agent_trace=["planner=semantic_topn_metric_fast_path", "planner.semantic_fast_path=topn_metric"],
        intents=[
            QuestionIntent(
                question="退款率最高商品",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                plan_task_id="derived_refund_rate",
                metric_name="refund_rate",
                group_by_column="sku_id",
                output_keys=["seller_id", "sku_id", "spu_name", "refund_rate"],
            ),
            QuestionIntent(
                question="商品发布时间",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                plan_task_id="goods_lookup",
                preferred_table="dwm_goods_detail_df",
                output_keys=["seller_id", "spu_id", "spu_name", "spu_apply_create_time"],
            ),
        ],
    )
    bundle = QueryBundle(
        rows=[
            {
                "seller_id": "100",
                "sku_id": "5",
                "spu_name": "Commuter Backpack Grey",
                "refund_rate": 62.5,
                "order_detail_cnt": 8,
                "pay_amt": 418.65,
                "spu_apply_create_time": "2026-03-01T10:00:00",
            }
        ],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="derived_refund_rate", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = service.run_analysis_skill(
        "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间，帮我判断哪些是高风险新品。",
        plan,
        run,
        str(tmp_path),
        skill_name="risk_analysis",
    )

    assert "风险分析" in answer
    assert "Commuter Backpack Grey" in answer
    assert service.last_analysis_skill_trace["workerType"] == "LOCAL_STRUCTURED_RENDERER"
    assert service.last_analysis_skill_trace["isolatedExecution"] is False
    assert service.last_analysis_skill_trace["deterministicRenderer"] is True


def test_answer_skill_selects_rule_compliance_for_rule_plus_data(tmp_path):
    settings = get_settings().model_copy(update={"llm_api_key": "", "harness_workspace_path": str(tmp_path)})
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [{"semanticLabel": "platform_rule", "suggestedDomains": ["rule"]}],
        },
        intents=[
            QuestionIntent(
                question="平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                metric_name="ship_timeout_order_cnt_1d",
            )
        ],
    )
    bundle = QueryBundle(rows=[{"pt": "2026-06-20", "ship_timeout_order_cnt_1d": 3}], original_row_count=1)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="ship_timeout", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    assert answer_skill_required(plan, run, True)
    answer = service.summarize_analysis("平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。", plan, run, str(tmp_path), "发货超时需关注发货时效。")

    assert "规则合规分析" in answer
    assert "规则依据" in answer
    assert service.last_analysis_skill_trace["skillName"] == "rule_compliance"


def test_answer_skill_selects_new_product_risk_from_structured_lifecycle_evidence(tmp_path):
    settings = get_settings().model_copy(update={"llm_api_key": "", "harness_workspace_path": str(tmp_path)})
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "risk_ranking",
            "requiresExplanation": True,
            "skillWorkflow": {"skillName": "new_product_risk", "required": True},
        },
        intents=[
            QuestionIntent(
                question="最近30天退款率最高的商品，判断高风险新品。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                preferred_table="dwm_goods_detail_df",
                output_keys=["spu_id", "spu_apply_create_time"],
                required_evidence=["spu_apply_create_time"],
            ),
            QuestionIntent(
                question="最近30天退款率最高的商品，判断高风险新品。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.REFUND,
                metric_name="refund_bill_cnt",
            ),
        ],
    )
    bundle = QueryBundle(rows=[{"spu_id": "spu_1", "spu_apply_create_time": "2026-06-01", "refund_bill_cnt": 5}], original_row_count=1)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="risk", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )

    assert select_answer_skill(plan, run, False) == "new_product_risk"
    answer = service.summarize_analysis("最近30天退款率最高的商品，判断高风险新品。", plan, run, str(tmp_path))

    assert "新品风险分析" in answer
    assert "spu_apply_create_time" in answer


def test_rule_only_evidence_intent_does_not_trigger_bi_analysis_summary():
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "overview",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "platform_rule_shipping_timeout",
                    "requiredLevel": "required",
                    "suggestedDomains": ["rule", "governance"],
                }
            ],
        }
    )

    assert not analysis_summary_required(plan)


def test_single_metric_rule_overview_does_not_trigger_bi_analysis_summary():
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "overview",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "business_rule_context",
                    "requiredLevel": "required",
                    "suggestedDomains": ["profile", "order"],
                },
                {
                    "semanticLabel": "recent_timeout_volume",
                    "requiredLevel": "required",
                    "suggestedDomains": ["profile"],
                    "suggestedMetricRefs": ["ship_timeout_order_cnt_1d"],
                },
            ],
        },
        intents=[
            QuestionIntent(
                question="平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
                group_by_column="merchant_id",
            )
        ],
    )

    assert not analysis_summary_required(plan)


def test_rule_data_anomaly_question_still_triggers_bi_analysis_summary():
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "anomaly_check",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "business_rule_context",
                    "requiredLevel": "required",
                    "suggestedDomains": ["rule", "governance"],
                },
                {
                    "semanticLabel": "metric_anomaly_signal",
                    "requiredLevel": "required",
                    "suggestedDomains": ["profile"],
                    "suggestedMetricRefs": ["ship_timeout_order_cnt_1d"],
                },
            ],
        },
        intents=[
            QuestionIntent(
                question="平台规则说发货超时要注意什么，同时分析最近7天发货超时订单量是否异常。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
                group_by_column="merchant_id",
            )
        ],
    )

    assert analysis_summary_required(plan)


def test_product_overview_does_not_trigger_trend_analysis_summary():
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "product",
            "analysisIntent": "overview",
            "requiresExplanation": True,
        },
        intents=[
            QuestionIntent(
                question="最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                category=QuestionCategory.SCM,
                preferred_table="dwm_scm_detail_di",
                metric_name="scm_inbound_total_cnt",
                group_by_column="spu_id",
            )
        ],
    )

    assert not analysis_summary_required(plan)


def test_product_risk_ranking_lookup_does_not_trigger_trend_analysis_summary():
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "product",
            "analysisIntent": "risk_ranking",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {"semanticLabel": "refund_evidence", "suggestedMetricRefs": ["refund_bill_cnt", "pay_amt"]},
                {"semanticLabel": "goods_publish_time", "suggestedFields": ["spu_apply_create_time"]},
            ],
        },
        intents=[
            QuestionIntent(
                question="找到最近60天赔付单量较高的商品，看下对应的退款量、退款金额和商品发布时间。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.COMPENSATION,
                metric_name="repay_bill_cnt",
                group_by_column="spu_id",
                output_keys=["seller_id", "spu_id", "spu_name"],
            ),
            QuestionIntent(
                question="找到最近60天赔付单量较高的商品，看下对应的退款量、退款金额和商品发布时间。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.REFUND,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                group_by_column="spu_name",
            ),
            QuestionIntent(
                question="找到最近60天赔付单量较高的商品，看下对应的退款量、退款金额和商品发布时间。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                preferred_table="dwm_goods_detail_df",
                output_keys=["seller_id", "spu_id", "spu_name", "spu_apply_create_time"],
            ),
        ],
    )

    assert not analysis_summary_required(plan)


def test_trend_skill_ignores_entity_identifier_columns():
    import importlib.util

    script = get_settings().resources_root / "runtime" / "agent_skills" / "bi_trend_attribution" / "scripts" / "profile_timeseries.py"
    spec = importlib.util.spec_from_file_location("profile_timeseries_for_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    profile = module.build_profile(
        {
            "dataRows": [
                {"spu_id": "1", "spu_name": "A", "repay_bill_cnt": 2, "pay_amt": "10.00"},
                {"spu_id": "2", "spu_name": "B", "repay_bill_cnt": 1, "pay_amt": "5.00"},
            ],
            "metricDisclosures": [
                {"metricKey": "repay_bill_cnt"},
                {"metricKey": "pay_amt"},
            ],
        }
    )

    assert "spu_id" not in profile["metricKeys"]
    assert set(profile["metricKeys"]) == {"repay_bill_cnt", "pay_amt"}
    assert not any("spu_id" in (finding.get("title") or "") for finding in profile["findings"])


def test_trend_skill_only_uses_disclosed_metric_not_alternate_gmv_candidates():
    import importlib.util

    script = get_settings().resources_root / "runtime" / "agent_skills" / "bi_trend_attribution" / "scripts" / "profile_timeseries.py"
    spec = importlib.util.spec_from_file_location("profile_timeseries_disclosed_metric_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    profile = module.build_profile(
        {
            "question": "最近7天GMV趋势",
            "dataRows": [
                {"pt": "2026-06-22", "pay_gmv_amt_1d": "990.00", "trade_success_gmv_amt_1d": "507.00"},
                {"pt": "2026-06-23", "pay_gmv_amt_1d": "782.50", "trade_success_gmv_amt_1d": "563.50"},
            ],
            "metricDisclosures": [
                {"metricKey": "pay_gmv_amt_1d", "displayName": "支付GMV"},
            ],
        }
    )

    assert profile["metricKeys"] == ["pay_gmv_amt_1d"]
    assert "交易成功GMV" not in profile["answerMarkdown"]
    assert "关键证据" not in profile["answerMarkdown"]
    assert "限制" not in profile["answerMarkdown"]
    assert "建议" not in profile["answerMarkdown"]


def test_task_evidence_sections_prioritize_user_facing_nodes():
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {"metricRef": "repay_bill_cnt", "resolvedMetricRef": "repay_bill_cnt", "groupByColumn": "spu_id"},
            "requestedMeasures": [
                {"metricRef": "refund_bill_cnt", "resolvedMetricRef": "refund_bill_cnt"},
                {"metricRef": "pay_amt", "resolvedMetricRef": "pay_amt"},
            ],
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "goods_publish_time",
                    "suggestedTables": ["dwm_goods_detail_df"],
                    "suggestedFields": ["spu_apply_create_time"],
                }
            ],
        },
        intents=[
            QuestionIntent(
                plan_task_id="anchor_repay",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                category=QuestionCategory.COMPENSATION,
                preferred_table="dwm_cs_repay_detail_df",
                metric_name="repay_bill_cnt",
                group_by_column="sub_order_id",
                metric_resolution={"metricKey": "repay_bill_cnt", "sourcePhrase": "赔付单量", "displayName": "赔付单量"},
            ),
            QuestionIntent(
                plan_task_id="component_order_order_detail_cnt",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                group_by_column="sub_order_id",
                metric_resolution={"metricKey": "order_detail_cnt", "sourcePhrase": "semantic formula dependency for compensation_rate"},
            ),
            QuestionIntent(
                plan_task_id="order_bridge",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.TRADE,
                preferred_table="dwm_trade_order_detail_di",
                output_keys=["seller_id", "sub_order_id", "order_id", "spu_id", "spu_name"],
            ),
            QuestionIntent(
                plan_task_id="projected_repay_bill_cnt_by_spu_id",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.COMPENSATION,
                metric_name="repay_bill_cnt",
                group_by_column="spu_id",
                output_keys=["spu_id", "repay_bill_cnt", "spu_name"],
                metric_resolution={
                    "metricKey": "repay_bill_cnt",
                    "sourcePhrase": "赔付单量",
                    "displayName": "赔付单量",
                    "computeStrategy": "projection_group_aggregate",
                    "sourceMetricTaskId": "anchor_repay",
                    "bridgeTaskId": "order_bridge",
                },
                depends_on_task_ids=["anchor_repay", "order_bridge"],
            ),
            QuestionIntent(
                plan_task_id="goods_bridge",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                preferred_table="dwm_goods_detail_df",
                output_keys=["seller_id", "spu_id", "spu_name", "spu_apply_create_time"],
            ),
            QuestionIntent(
                plan_task_id="refund_lookup",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.REFUND,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="refund_bill_cnt",
                group_by_column="spu_name",
                metric_resolution={"metricKey": "refund_bill_cnt", "sourcePhrase": "退款量", "displayName": "退款量"},
            ),
            QuestionIntent(
                plan_task_id="refund_lookup_2",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.REFUND,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                group_by_column="spu_name",
                metric_resolution={"metricKey": "pay_amt", "sourcePhrase": "退款金额", "displayName": "退款金额"},
            ),
        ],
    )

    def result(task_id: str, table: str, row: dict) -> AgentTaskResult:
        return AgentTaskResult(
            task_id=task_id,
            success=True,
            query_bundle=QueryBundle(tables=[table] if table else [], rows=[row], original_row_count=1),
        )

    run = AgentRunResult(
        task_results=[
            result("anchor_repay", "dwm_cs_repay_detail_df", {"sub_order_id": "sub_1", "repay_bill_cnt": 1}),
            result("component_order_order_detail_cnt", "dwm_trade_order_detail_di", {"sub_order_id": "sub_1", "order_detail_cnt": 1}),
            result("order_bridge", "dwm_trade_order_detail_di", {"sub_order_id": "sub_1", "spu_id": "spu_1", "spu_name": "A"}),
            result("projected_repay_bill_cnt_by_spu_id", "", {"spu_id": "spu_1", "spu_name": "A", "repay_bill_cnt": 1}),
            result("goods_bridge", "dwm_goods_detail_df", {"spu_id": "spu_1", "spu_name": "A", "spu_apply_create_time": "2026-05-01"}),
            result("refund_lookup", "dwm_trade_refund_detail_di", {"spu_name": "A", "refund_bill_cnt": 2}),
            result("refund_lookup_2", "dwm_trade_refund_detail_di", {"spu_name": "A", "pay_amt": "88.00"}),
        ],
    )

    section = task_evidence_sections(plan, run)
    summary = business_summary_table(plan, run)

    assert "赔付单量（按 spu_id 汇总）" in section
    assert "商品发布时间" in section
    assert "退款量" in section
    assert "退款金额" in section
    assert "anchor_repay" not in section
    assert "order_bridge" not in section
    assert "component_order_order_detail_cnt" not in section
    assert "| SPU ID | 商品 | 赔付单量 | 退款量 | 退款金额 | 商品发布时间 |" in summary
    assert "| spu_1 | A | 1 | 2 | 88.00 | 2026-05-01 |" in summary


def test_business_summary_table_merges_sibling_merchant_metrics():
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {"metricRef": "order_gmv_amt_1d", "resolvedMetricRef": "order_gmv_amt_1d"},
            "requestedMeasures": [
                {"metricRef": "pay_amt", "resolvedMetricRef": "pay_amt"},
                {"metricRef": "ticket_cnt", "resolvedMetricRef": "ticket_cnt"},
            ],
        },
        intents=[
            QuestionIntent(
                plan_task_id="gmv_metric",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_resolution={"metricKey": "order_gmv_amt_1d", "displayName": "GMV"},
            ),
            QuestionIntent(
                plan_task_id="refund_metric",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_resolution={"metricKey": "pay_amt", "displayName": "退款金额"},
            ),
            QuestionIntent(
                plan_task_id="ticket_metric",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                preferred_table="dwm_cs_ticket_detail_di",
                metric_name="ticket_cnt",
                metric_resolution={"metricKey": "ticket_cnt", "displayName": "工单量"},
            ),
        ],
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="refund_metric", success=True, query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"seller_id": "100", "pay_amt": "88.00"}])),
            AgentTaskResult(task_id="gmv_metric", success=True, query_bundle=QueryBundle(tables=["ads_merchant_profile"], rows=[{"merchant_id": "100", "order_gmv_amt_1d": "188.00"}])),
            AgentTaskResult(task_id="ticket_metric", success=True, query_bundle=QueryBundle(tables=["dwm_cs_ticket_detail_di"], rows=[{"seller_id": "100", "ticket_cnt": 3}])),
        ]
    )

    summary = business_summary_table(plan, run)

    assert "| GMV | 退款金额 | 工单量 |" in summary
    assert "| 188.00 | 88.00 | 3 |" in summary


def test_rule_evidence_only_appends_when_plan_requires_rules():
    product_plan = QueryPlan(
        question_understanding={
            "analysisGrain": "product",
            "analysisIntent": "overview",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "top_inbound_goods_order_performance",
                    "requiredLevel": "required",
                    "suggestedDomains": ["scm", "trade", "refund"],
                }
            ],
        }
    )
    rule_plan = QueryPlan(
        question_understanding={
            "analysisGrain": "merchant",
            "analysisIntent": "overview",
            "requiresExplanation": True,
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "platform_rule_shipping_timeout",
                    "requiredLevel": "required",
                    "suggestedDomains": ["rule"],
                }
            ],
        }
    )

    assert not plan_requires_rule_evidence(product_plan)
    assert plan_requires_rule_evidence(rule_plan)


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


def test_time_window_clarification_structurally_updates_route_slots():
    workflow = create_workflow(get_settings())
    pending_question = "帮我看一下订单情况"
    state = workflow._initial_state(
        "近30天",
        "merchant_001",
        ChatContext(
            pending_clarification_stage="BUSINESS_SCOPE",
            pending_clarification_type="time_window",
            pending_question=pending_question,
            pending_clarification_options=["近7天", "昨天", "近30天"],
        ),
        None,
        "test_time_clarification",
        "run_time_clarification",
    )
    state = workflow.inherit_context(state)
    assert state["clarification_resolution"]["timeWindowDays"] == 30
    assert state["request_context"].clarification_resolved is True
    assert state["request_context"].resolved_time_window_days == 30

    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)

    assert state["route_slots"].time_window.days == 30
    assert state["route_slots"].time_window.raw == "近30天"
    assert any(item.get("type") == "time_window" for item in state["route_decision_trace"])


def test_clarification_resolution_service_resolves_context_and_route_slots():
    service = ClarificationResolutionService()
    context = ChatContext(
        pending_clarification_stage="BUSINESS_SCOPE",
        pending_clarification_type="metric_focus",
        pending_question="帮我分析店铺表现为什么变差",
    )

    resolution = service.resolve_context(context, "重点看退款率")
    slots, trace = service.apply_to_route_slots(RouteSlots(), resolution)

    assert resolution["metricFocus"] == "退款率"
    assert context.clarification_resolved is True
    assert context.metric_focus == "退款率"
    assert "退款率" in slots.analysis_signals
    assert trace[0]["type"] == "metric_focus"


def test_metric_focus_clarification_structurally_updates_fast_understanding():
    workflow = create_workflow(get_settings())
    pending_question = "帮我分析店铺表现为什么变差"
    state = workflow._initial_state(
        "重点看退款率",
        "merchant_001",
        ChatContext(
            pending_clarification_stage="BUSINESS_SCOPE",
            pending_clarification_type="metric_focus",
            pending_question=pending_question,
            pending_clarification_options=["综合经营风险", "GMV/销售额", "退款率"],
        ),
        None,
        "test_metric_clarification",
        "run_metric_clarification",
    )
    state = workflow.inherit_context(state)
    assert state["clarification_resolution"]["metricFocus"] == "退款率"

    state = workflow.runtime_bootstrap(state)
    state = workflow.route_topic(state)
    state = workflow.fast_understand(state)

    assert "退款率" in state["route_slots"].analysis_signals
    assert "退款率" in state["fast_understanding"].metric_phrases
    assert not state["human_clarification_required"]


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


def test_semantic_metric_resolver_accepts_exact_detail_metric_without_recall_loop():
    settings = get_settings()
    question = "最近7天订单量是多少？"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE],
    )

    resolution = SemanticMetricResolver(pack).resolve(
        question,
        "order_detail_cnt",
        "dwm_trade_order_detail_di",
        "订单量",
    )

    assert resolution.metric
    assert resolution.metric.key == "order_detail_cnt"
    assert resolution.metric.table == "dwm_trade_order_detail_di"
    assert resolution.resolution_source == "semantic_metric_ref"
    assert not resolution.knowledge_requests


def test_knowledge_requests_dedupe_by_stable_key_and_preserve_task_scope():
    first = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="订单量 语义指标口径",
        needed_for_task_id="anchor_order",
        reason="Resolver needs scoped semantic metric evidence: metric_evidence_unscoped requested=order_detail_cnt",
    )
    duplicate = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="订单量   语义指标口径",
        needed_for_task_id="anchor_order",
        reason="Resolver needs scoped semantic metric evidence: metric_evidence_unscoped requested=other",
    )
    other_task = duplicate.model_copy(update={"needed_for_task_id": "dependent_order"})

    deduped = dedupe_workflow_knowledge_requests([first, duplicate, other_task])

    assert len(deduped) == 2
    assert knowledge_request_key(first) == knowledge_request_key(duplicate)
    assert knowledge_request_key(first) != knowledge_request_key(other_task)


def test_knowledge_request_gap_append_is_stable():
    request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="GMV 语义指标口径",
        reason="Resolver needs scoped semantic metric evidence: metric_evidence_unscoped requested=pay_amt",
    )

    first = append_knowledge_request_gaps([], [request], "METRIC_EVIDENCE_UNCHANGED")
    second = append_knowledge_request_gaps(first, [request], "METRIC_EVIDENCE_UNCHANGED")

    assert len(second) == 1
    assert second[0]["code"] == "METRIC_EVIDENCE_UNCHANGED"


def test_knowledge_retrieval_service_returns_unified_bundle():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    recall_service = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    service = HybridKnowledgeRetrievalService(recall_service)

    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="最近7天订单量是多少？",
            keywords=["订单量"],
            merchant_id=settings.merchant_id,
            topic_categories=[QuestionCategory.TRADE],
        )
    )

    assert bundle.backend == "hybrid"
    assert bundle.recall_bundle.items
    assert bundle.source_refs
    assert bundle.recall_rounds[0].query == "最近7天订单量是多少？"
    assert bundle.recall_rounds[0].source_refs == bundle.source_refs


def test_retrieve_knowledge_uses_pending_request_as_second_round_query():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("最近7天订单量是多少？", "100", None, None, "test_thread_recall", "test_run_recall")
    state = workflow.route_topic(state)
    state["pending_knowledge_requests"] = [
        KnowledgeRequest(
            type=KnowledgeRequestType.METRIC,
            query="订单量 语义指标口径",
            needed_for_task_id="anchor_order",
            reason="Resolver needs scoped semantic metric evidence: metric_evidence_unscoped requested=order_detail_cnt",
        )
    ]

    state = workflow.retrieve_knowledge(state)

    rounds = state.get("recall_rounds", [])
    assert rounds
    assert any(item.get("requestKey") for item in rounds)
    assert any("订单量 语义指标口径" == item.get("query") for item in rounds)
    assert state.get("knowledge_bundle").backend in {
        workflow.knowledge_retriever.backend_name,
        "%s_fallback_hybrid" % workflow.knowledge_retriever.backend_name,
    }
    assert state.get("pending_knowledge_requests") == []


def test_repeated_pending_knowledge_request_without_new_refs_is_blocked():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("最近7天订单量是多少？", "100", None, None, "test_thread_block", "test_run_block")
    state = workflow.route_topic(state)
    request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="订单量 语义指标口径",
        needed_for_task_id="anchor_order",
        reason="Resolver needs scoped semantic metric evidence: metric_evidence_unscoped requested=order_detail_cnt",
    )

    state["pending_knowledge_requests"] = [request]
    state = workflow.retrieve_knowledge(state)
    state["pending_knowledge_requests"] = [request]
    state = workflow.retrieve_knowledge(state)

    key = knowledge_request_key(request)
    assert key in set(state.get("blocked_knowledge_request_keys") or [])
    assert any(item.get("code") == "METRIC_EVIDENCE_UNCHANGED" for item in state.get("knowledge_request_gaps", []))


def test_compact_assets_records_recall_lineage_and_loaded_refs():
    workflow = create_workflow(get_settings())
    state = workflow._initial_state("最近7天订单量是多少？", "100", None, None, "test_thread_compact", "test_run_compact")
    state = workflow.route_topic(state)
    state = workflow.retrieve_knowledge(state)
    state = workflow.compact_assets(state)

    compaction = state["planning_asset_pack"].metric_compaction
    assert compaction["recallBackend"] in {
        workflow.knowledge_retriever.backend_name,
        "%s_fallback_hybrid" % workflow.knowledge_retriever.backend_name,
    }
    assert compaction["recallLineage"]
    assert compaction["loadedSourceRefs"]
    assert set(compaction["loadedSourceRefs"]).issuperset(state["knowledge_bundle"].source_refs[:1])


class FakeRecallDocumentProvider:
    def __init__(self, settings):
        self.settings = settings
        self.cleared = 0

    def _load_documents(self):
        return [
            RecallItem(
                doc_id="semantic:电商交易:dwm_trade_order_detail_di:asset",
                title="电商交易/dwm_trade_order_detail_di semantic asset",
                content="订单明细语义资产",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="电商交易",
                table="dwm_trade_order_detail_di",
                metadata={
                    "semanticRefId": "semantic:电商交易:dwm_trade_order_detail_di:asset",
                    "semanticPath": "topics/电商交易/tables/dwm_trade_order_detail_di/asset.json",
                },
            )
        ]

    def clear_cache(self):
        self.cleared += 1


class FakeEsRecallAdapter:
    def __init__(self):
        self.calls = []

    def sync(self, docs, deleted_refs, replace_all=False):
        self.calls.append((docs, deleted_refs, replace_all))
        return {"success": True, "mode": "es", "upserted": len(docs), "deleted": len(deleted_refs), "replaceAll": replace_all}


def write_test_semantic_asset(root: Path) -> None:
    table_dir = root / "电商交易" / "tables" / "dwm_trade_order_detail_di"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "asset.json").write_text('{"tableName":"dwm_trade_order_detail_di"}', encoding="utf-8")


class FakeGovernanceDoris:
    def show_full_columns(self, table):
        return [
            {"Field": "seller_id", "Type": "varchar"},
            {"Field": "pay_amt", "Type": "decimal(18,2)"},
            {"Field": "refund_id", "Type": "varchar"},
        ]


class TopicBuilderDoris:
    def __init__(self, schema_rows=None, sample_rows=None):
        self.schema_rows = list(schema_rows or [])
        self.sample_rows_payload = list(sample_rows or [])

    def show_full_columns(self, table):
        return list(self.schema_rows)

    def sample_rows(self, table, merchant_id, limit=20):
        return list(self.sample_rows_payload)[:limit]


class TopicBuilderLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, timeout_seconds=None):
        self.calls += 1
        payload = json.loads(user_prompt)
        assert payload["sampleRows"]
        return {
            "tableComment": "退款明细候选表",
            "dataGrain": "退款/售后明细粒度",
            "timeColumn": "pt",
            "merchantFilterColumn": "seller_id",
            "semanticColumns": [
                {
                    "columnName": "seller_id",
                    "businessName": "商家ID",
                    "role": "KEY",
                    "description": "商家唯一标识",
                    "aliases": ["seller_id", "商家ID"],
                    "enumValues": [],
                    "sampleValues": ["100"],
                    "confidence": 0.91,
                    "evidence": "llm",
                },
                {
                    "columnName": "pt",
                    "businessName": "业务日期",
                    "role": "TIME",
                    "description": "按天分区字段",
                    "aliases": ["pt", "业务日期"],
                    "enumValues": [],
                    "sampleValues": ["2026-07-01"],
                    "confidence": 0.9,
                    "evidence": "llm",
                },
                {
                    "columnName": "pay_amt",
                    "businessName": "退款金额字段",
                    "role": "ATTRIBUTE",
                    "description": "退款金额原始字段",
                    "aliases": ["pay_amt", "退款金额字段"],
                    "enumValues": [],
                    "sampleValues": ["18.3"],
                    "confidence": 0.88,
                    "evidence": "llm",
                },
            ],
            "metrics": [
                {
                    "metricKey": "pay_amt",
                    "canonicalMetricKey": "pay_amt",
                    "businessName": "退款金额",
                    "formula": "SUM(pay_amt)",
                    "unit": "元",
                    "description": "退款金额",
                    "sourceColumns": ["pay_amt"],
                    "aliases": ["退款金额", "refund_amt"],
                    "confidence": 0.92,
                    "evidence": "llm",
                }
            ],
            "terms": [
                {
                    "term": "退款金额",
                    "businessName": "退款金额",
                    "description": "退款金额标准术语",
                    "aliases": ["退款金额", "refund_amt"],
                    "canonicalMetricKey": "pay_amt",
                }
            ],
            "knowledgeRules": [
                {
                    "ruleId": "refund_time_rule",
                    "title": "退款按天统计",
                    "description": "优先使用 pt 做退款时间范围过滤",
                    "aliases": ["退款时间过滤"],
                    "appliesToColumns": ["pt"],
                    "appliesToMetrics": ["pay_amt"],
                }
            ],
        }


def write_pending_governance_asset(root: Path, metric_source: str = "pay_amt") -> Path:
    pending_dir = root / "电商退货" / "pending" / "dwm_trade_refund_detail_di"
    pending_dir.mkdir(parents=True, exist_ok=True)
    (root / "电商退货").mkdir(parents=True, exist_ok=True)
    write_json(pending_dir / "asset.json", {"topic": "电商退货", "tableName": "dwm_trade_refund_detail_di"})
    write_json(
        pending_dir / "schema.json",
        [
            {"columnName": "seller_id", "dataType": "varchar"},
            {"columnName": "pay_amt", "dataType": "decimal(18,2)"},
            {"columnName": "refund_id", "dataType": "varchar"},
        ],
    )
    write_json(
        pending_dir / "metrics.json",
        [
            {
                "metricKey": "pay_amt",
                "canonicalMetricKey": "pay_amt",
                "businessName": "退款金额",
                "formula": "SUM(%s)" % metric_source,
                "sourceColumns": [metric_source],
            }
        ],
    )
    write_json(root / "电商退货" / "relationships.json", [])
    return pending_dir


def test_semantic_governance_preflight_rejects_missing_metric_column(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    write_pending_governance_asset(settings.resolved_topic_path, metric_source="missing_amt")
    service = SemanticAssetGovernanceService(settings, FakeGovernanceDoris(), TopicAssetService(settings))

    result = service.preflight_publish("电商退货", "dwm_trade_refund_detail_di")

    assert not result["publishable"]
    assert result["status"] == "PREFLIGHT_FAILED"
    assert result["releaseGate"]["severity"] == "blocking"
    assert "SEMANTIC_VALIDATION_ERRORS" in result["releaseGate"]["blockingReasons"]
    assert result["validation"]["errors"][0]["code"] == "METRIC_SOURCE_COLUMN_MISSING"
    assert "impactTestPlan" in result
    assert Path(result["reviewArtifact"]).exists()


def test_semantic_governance_publish_writes_version_and_drift_artifact(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    write_pending_governance_asset(settings.resolved_topic_path)
    topic_assets = TopicAssetService(settings)
    service = SemanticAssetGovernanceService(settings, FakeGovernanceDoris(), topic_assets)

    preflight = service.preflight_publish("电商退货", "dwm_trade_refund_detail_di")
    published = topic_assets.publish("电商退货", "dwm_trade_refund_detail_di", True, "tester", "ok")
    governed = service.after_publish("电商退货", "dwm_trade_refund_detail_di", "tester", "ok")

    assert preflight["publishable"]
    assert published["status"] == "PUBLISHED"
    assert published["publishMode"] == "scoped_incremental"
    assert published["publishScope"]["table"] == "dwm_trade_refund_detail_di"
    version_path = settings.resolved_topic_path / "电商退货" / "tables" / "dwm_trade_refund_detail_di" / "semantic_version.json"
    assert version_path.exists()
    version = json.loads(version_path.read_text(encoding="utf-8"))
    assert version["semanticVersion"].startswith("semantic-")
    assert governed["schemaDriftReport"]["extraLiveColumns"] == []
    assert governed["releaseGate"]["publishable"]
    assert governed["semanticGovernance"]["owner"]
    assert governed["approvalWorkflow"]["stage"] == "published"
    assert governed["grayReleasePlan"]["strategy"] == "scoped_incremental"
    assert governed["grayReleaseMonitor"]["status"] == "monitoring"
    assert governed["conflictDetection"]["status"] == "passed"
    assert governed["conflictRepairPlan"]["status"] == "no_conflict"
    assert governed["evaluationGate"]["goldenEval"]["requiredBeforePublish"] is True
    assert governed["semanticLineage"]["metrics"][0]["metricKey"] == "pay_amt"
    assert governed["driftGovernance"]["severity"] == "passed"
    assert governed["impactTestPlan"]["impactedMetrics"] == ["pay_amt"]
    assert governed["publishMode"] == "scoped_incremental"
    assert governed["publishScope"]["table"] == "dwm_trade_refund_detail_di"
    assert Path(governed["reviewArtifact"]).exists()
    assert Path(governed["publishHistoryPath"]).exists()


def test_topic_builder_batch_build_reports_factory_status(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    topic_assets = TopicAssetService(settings)
    doris = TopicBuilderDoris(
        schema_rows=[
            {"Field": "seller_id", "Type": "varchar", "Comment": "商家ID"},
            {"Field": "pt", "Type": "date", "Comment": "业务日期"},
            {"Field": "pay_amt", "Type": "decimal(18,2)", "Comment": "金额"},
        ],
        sample_rows=[{"seller_id": "100", "pt": "2026-07-01", "pay_amt": 18.3}],
    )
    workflow = TopicBuilderWorkflow(settings, doris, topic_assets, llm=type("DisabledBuilderLlm", (), {"configured": False})())

    result = workflow.build_batch(
        [
            TopicBuildRequest(topic="电商退货", table_name="dwm_trade_refund_detail_di", merchant_id="100"),
            TopicBuildRequest(topic="电商交易", table_name="dwm_trade_order_detail_di", merchant_id="100"),
        ]
    )

    assert result["status"] == "BATCH_BUILT"
    assert result["factoryReport"]["mode"] == "topic_asset_factory"
    assert result["successCount"] == 2
    assert Path(result["reportPath"]).exists()


def test_topic_asset_publish_deletes_stale_managed_files_within_table_scope(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    topic_assets = TopicAssetService(settings)
    target_dir = settings.resolved_topic_path / "电商退货" / "tables" / "dwm_trade_refund_detail_di"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "asset.json", {"topic": "电商退货", "tableName": "dwm_trade_refund_detail_di"})
    write_json(target_dir / "semantic_columns.json", [{"columnName": "seller_id"}])
    write_json(target_dir / "sample_rows.json", [{"seller_id": "100"}])
    write_json(target_dir / "semantic_version.json", {"semanticVersion": "semantic-old"})
    write_json(target_dir / "semantic_publish_history.json", [{"semanticVersion": "semantic-old"}])

    pending_dir = settings.resolved_topic_path / "电商退货" / "pending" / "dwm_trade_refund_detail_di"
    pending_dir.mkdir(parents=True, exist_ok=True)
    write_json(pending_dir / "asset.json", {"topic": "电商退货", "tableName": "dwm_trade_refund_detail_di"})
    write_json(pending_dir / "semantic_columns.json", [{"columnName": "seller_id"}, {"columnName": "pay_amt"}])

    result = topic_assets.publish("电商退货", "dwm_trade_refund_detail_di", True, "tester", "cleanup stale files")

    assert result["publishMode"] == "scoped_incremental"
    assert "sample_rows.json" in result["deletedFiles"]
    assert not (target_dir / "sample_rows.json").exists()
    assert (target_dir / "semantic_version.json").exists()
    assert (target_dir / "semantic_publish_history.json").exists()


def test_topic_builder_build_generates_pending_assets_with_llm_and_samples(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    topic_assets = TopicAssetService(settings)
    doris = TopicBuilderDoris(
        schema_rows=[
            {"Field": "seller_id", "Type": "varchar", "Comment": "商家ID"},
            {"Field": "pt", "Type": "date", "Comment": "业务日期"},
            {"Field": "refund_id", "Type": "varchar", "Comment": "退款单号"},
            {"Field": "pay_amt", "Type": "decimal(18,2)", "Comment": "退款金额"},
        ],
        sample_rows=[
            {"seller_id": "100", "pt": "2026-07-01", "refund_id": "r1", "pay_amt": 18.3},
            {"seller_id": "100", "pt": "2026-07-02", "refund_id": "r2", "pay_amt": 21.5},
        ],
    )
    llm = TopicBuilderLlm()
    workflow = TopicBuilderWorkflow(settings, doris, topic_assets, llm=llm)

    result = workflow.build(
        TopicBuildRequest(
            topic="电商退货",
            table_name="dwm_trade_refund_detail_di",
            merchant_id="100",
            manual_notes="退款表",
            business_knowledge="退款金额按天看",
            sample_sqls=["SELECT seller_id, pt, pay_amt FROM dwm_trade_refund_detail_di LIMIT 10"],
        )
    )

    pending_dir = settings.resolved_topic_path / "电商退货" / "pending" / "dwm_trade_refund_detail_di"
    assert result["success"]
    assert result["generationMode"] == "llm"
    assert llm.calls == 1
    assert (pending_dir / "schema.json").exists()
    assert (pending_dir / "sample_rows.json").exists()
    assert (pending_dir / "sample_profile.json").exists()
    assert (pending_dir / "semantic_columns.json").exists()
    assert (pending_dir / "metrics.json").exists()
    assert (pending_dir / "terms.json").exists()
    assert (pending_dir / "knowledge_rules.json").exists()
    assert (pending_dir / "asset_production_report.json").exists()
    asset = json.loads((pending_dir / "asset.json").read_text(encoding="utf-8"))
    production = json.loads((pending_dir / "asset_production_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((pending_dir / "metrics.json").read_text(encoding="utf-8"))
    fields = json.loads((pending_dir / "semantic_columns.json").read_text(encoding="utf-8"))
    assert asset["generationMode"] == "llm"
    assert result["builderPhases"]["schemaDiscovery"]["status"] == "completed"
    assert result["builderPhases"]["semanticAnalysis"]["mode"] == "llm"
    assert asset["builderPhases"]["humanReviewPublish"]["status"] == "pending_review"
    assert asset["semanticGovernance"]["approval"]["required"] is True
    assert asset["approvalWorkflow"]["stage"] == "pending_review"
    assert production["semanticDraft"]["metricCount"] >= 1
    assert result["assetProductionReport"]["status"] == "ready_for_human_review"
    assert asset["timeColumn"] == "pt"
    assert asset["merchantFilterColumn"] == "seller_id"
    assert any(item["metricKey"] == "pay_amt" for item in metrics)
    assert any(item["columnName"] == "refund_id" for item in fields)


def test_topic_builder_refresh_incremental_rebuilds_pending_assets_after_schema_change(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    topic_assets = TopicAssetService(settings)
    doris = TopicBuilderDoris(
        schema_rows=[
            {"Field": "seller_id", "Type": "varchar", "Comment": "商家ID"},
            {"Field": "pt", "Type": "date", "Comment": "业务日期"},
            {"Field": "pay_amt", "Type": "decimal(18,2)", "Comment": "退款金额"},
        ],
        sample_rows=[{"seller_id": "100", "pt": "2026-07-01", "pay_amt": 18.3}],
    )

    workflow = TopicBuilderWorkflow(settings, doris, topic_assets, llm=type("DisabledBuilderLlm", (), {"configured": False})())
    request = TopicBuildRequest(topic="电商退货", table_name="dwm_trade_refund_detail_di", merchant_id="100")
    workflow.build(request)

    topic_assets.publish("电商退货", "dwm_trade_refund_detail_di", True, "tester", "seed")
    doris.schema_rows = [
        {"Field": "seller_id", "Type": "varchar", "Comment": "商家ID"},
        {"Field": "pt", "Type": "date", "Comment": "业务日期"},
        {"Field": "pay_amt", "Type": "decimal(18,2)", "Comment": "退款金额"},
        {"Field": "refund_reason", "Type": "varchar", "Comment": "退款原因"},
    ]
    doris.sample_rows_payload = [
        {"seller_id": "100", "pt": "2026-07-01", "pay_amt": 18.3, "refund_reason": "缺货"},
        {"seller_id": "100", "pt": "2026-07-02", "pay_amt": 21.5, "refund_reason": "质量问题"},
    ]

    result = workflow.refresh_incremental(request)

    pending_dir = settings.resolved_topic_path / "电商退货" / "pending" / "dwm_trade_refund_detail_di"
    fields = json.loads((pending_dir / "semantic_columns.json").read_text(encoding="utf-8"))
    profile = json.loads((pending_dir / "sample_profile.json").read_text(encoding="utf-8"))
    assert result["success"]
    assert "refund_reason" in result["schemaDiff"]["added"]
    assert any(item["columnName"] == "refund_reason" for item in fields)
    assert "refund_reason" in profile["enumCandidates"]


def test_topic_builder_heuristics_add_row_access_and_sensitive_column_policies(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    topic_assets = TopicAssetService(settings)
    doris = TopicBuilderDoris(
        schema_rows=[
            {"Field": "seller_id", "Type": "varchar", "Comment": "商家ID"},
            {"Field": "buyer_phone", "Type": "varchar", "Comment": "买家手机号"},
            {"Field": "pt", "Type": "date", "Comment": "业务日期"},
        ],
        sample_rows=[{"seller_id": "100", "buyer_phone": "13812345678", "pt": "2026-07-01"}],
    )
    workflow = TopicBuilderWorkflow(settings, doris, topic_assets, llm=type("DisabledBuilderLlm", (), {"configured": False})())

    result = workflow.build(TopicBuildRequest(topic="电商交易", table_name="dwm_trade_order_detail_di", merchant_id="100"))

    pending_dir = settings.resolved_topic_path / "电商交易" / "pending" / "dwm_trade_order_detail_di"
    asset = json.loads((pending_dir / "asset.json").read_text(encoding="utf-8"))
    fields = json.loads((pending_dir / "semantic_columns.json").read_text(encoding="utf-8"))
    buyer_phone = next(item for item in fields if item["columnName"] == "buyer_phone")
    assert result["success"]
    assert asset["rowAccessPolicy"]["filterColumn"] == "seller_id"
    assert buyer_phone["visibilityPolicy"]["level"] == "restricted"
    assert buyer_phone["maskingPolicy"]["strategy"] == "full"


def test_semantic_governance_blocks_missing_live_columns(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics"), "harness_workspace_path": str(tmp_path / "workspace")})
    write_pending_governance_asset(settings.resolved_topic_path)

    class MissingLiveColumnDoris:
        def show_full_columns(self, table):
            return [{"Field": "seller_id", "Type": "varchar"}]

    service = SemanticAssetGovernanceService(settings, MissingLiveColumnDoris(), TopicAssetService(settings))

    result = service.preflight_publish("电商退货", "dwm_trade_refund_detail_di")

    assert not result["publishable"]
    assert result["driftGovernance"]["severity"] == "blocking"
    assert "MISSING_LIVE_COLUMNS" in result["releaseGate"]["blockingReasons"]


def test_recall_index_manager_writes_manifest_and_clears_cache(tmp_path):
    settings = get_settings()
    old_topic_path = settings.topic_path
    old_workspace = settings.harness_workspace_path
    old_es_enabled = settings.es_enabled
    try:
        settings.topic_path = str(tmp_path / "topics")
        settings.harness_workspace_path = str(tmp_path / "workspace")
        settings.es_enabled = False
        write_test_semantic_asset(settings.resolved_topic_path)
        provider = FakeRecallDocumentProvider(settings)
        cleared = {"asset": 0}
        manager = RecallIndexManager(settings, provider, cache_clearers=[lambda: cleared.__setitem__("asset", cleared["asset"] + 1)])

        result = manager.rebuild(changed_only=True, topic="电商交易", table_name="dwm_trade_order_detail_di")

        assert result["success"]
        assert result["mode"] == "local_recall"
        assert result["rebuildMode"] == "scoped_incremental"
        assert result["rebuildScope"] == {"topic": "电商交易", "table": "dwm_trade_order_detail_di"}
        assert result["updatedRefCount"] == 1
        assert result["updatedRefs"] == ["电商交易/tables/dwm_trade_order_detail_di/asset.json"]
        assert Path(result["manifestPath"]).exists()
        assert provider.cleared == 1
        assert cleared["asset"] == 1
        assert result["es"]["mode"] == "disabled"
    finally:
        settings.topic_path = old_topic_path
        settings.harness_workspace_path = old_workspace
        settings.es_enabled = old_es_enabled


def test_recall_index_manager_calls_es_adapter_when_enabled(tmp_path):
    settings = get_settings()
    old_topic_path = settings.topic_path
    old_workspace = settings.harness_workspace_path
    old_es_enabled = settings.es_enabled
    try:
        settings.topic_path = str(tmp_path / "topics")
        settings.harness_workspace_path = str(tmp_path / "workspace")
        settings.es_enabled = True
        write_test_semantic_asset(settings.resolved_topic_path)
        provider = FakeRecallDocumentProvider(settings)
        adapter = FakeEsRecallAdapter()
        manager = RecallIndexManager(settings, provider, es_adapter=adapter)

        result = manager.rebuild(changed_only=True, topic="电商交易", table_name="dwm_trade_order_detail_di")

        assert result["success"]
        assert result["mode"] == "es"
        assert result["rebuildMode"] == "scoped_incremental"
        assert result["es"]["upserted"] == 1
        assert len(adapter.calls) == 1
        docs, deleted_refs, replace_all = adapter.calls[0]
        assert docs[0].doc_id == "semantic:电商交易:dwm_trade_order_detail_di:asset"
        assert deleted_refs == []
        assert replace_all is False
    finally:
        settings.topic_path = old_topic_path
        settings.harness_workspace_path = old_workspace
        settings.es_enabled = old_es_enabled


def test_create_workflow_uses_es_retriever_when_enabled():
    settings = get_settings()
    old_es_enabled = settings.es_enabled
    try:
        settings.es_enabled = True
        workflow = create_workflow(settings)
        assert isinstance(workflow.knowledge_retriever, EsKnowledgeRetrievalService)
    finally:
        settings.es_enabled = old_es_enabled


def test_es_knowledge_retrieval_returns_source_refs(monkeypatch):
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    service = EsKnowledgeRetrievalService(settings, topic_assets)

    def fake_search(query_text, topics, include_base_wiki=False):
        assert "退款金额" in query_text
        assert topics
        assert include_base_wiki is False
        return [
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_related_pay_amt",
                title="refund metric",
                content="退款金额 pay_amt",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=9.0,
                metadata={"semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_related_pay_amt"},
            )
        ]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(query="最近30天退款金额最高的商品", topic_categories=[QuestionCategory.REFUND])
    )

    assert bundle.backend == "es"
    assert "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_related_pay_amt" in bundle.source_refs
    assert bundle.recall_rounds[0].backend == "es"
    assert bundle.recall_bundle.items[0].table == "dwm_trade_refund_detail_di"


def test_es_retrieval_supplements_exact_metric_evidence_when_topk_misses_it(monkeypatch):
    settings = get_settings().model_copy(update={"embedding_api_key": "", "llm_api_key": "", "cache_enabled": False})
    topic_assets = TopicAssetService(settings)
    service = EsKnowledgeRetrievalService(settings, topic_assets)

    def fake_search(query_text, topics, include_base_wiki=False):
        return [
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate",
                title="refund rate",
                content="退款率",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=99.0,
                metadata={
                    "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate",
                    "metricKey": "refund_rate",
                    "tableName": "dwm_trade_refund_detail_di",
                    "topic": "电商退货",
                    "businessName": "商品退货率",
                },
            )
        ]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="最近30天退款金额最高的前5个商品",
            topic_categories=[QuestionCategory.REFUND, QuestionCategory.GOODS],
        )
    )

    metric_ids = {item.doc_id for item in bundle.recall_bundle.items if str(item.source_type).upper() == "SEMANTIC_METRIC"}
    assert "semantic:电商退货:dwm_trade_refund_detail_di:metric:pay_amt" in metric_ids
    pay_item = next(item for item in bundle.recall_bundle.items if item.doc_id.endswith(":pay_amt"))
    assert pay_item.metadata["businessName"] == "退款金额"
    assert pay_item.metadata["matchedMetricLabel"] == "退款金额"
    assert pay_item.metadata["recallSupplement"] == "metric_candidate_resolution"
    assert pay_item.metadata["recallChannel"] == "metric_resolver"
    assert pay_item.metadata["metricResolutionType"] == "exact_business_name"
    assert pay_item.metadata["metricResolutionConfidence"] >= 0.95
    assert "metric_resolver" in bundle.recall_rounds[0].recall_channels
    assert bundle.recall_rounds[0].source_type_top_k["SEMANTIC_METRIC"] >= 1
    assert bundle.recall_rounds[0].vector_disabled is True
    assert bundle.recall_rounds[0].metric_candidates
    assert bundle.recall_rounds[0].metric_candidates[0]["metricKey"] == "pay_amt"


def test_es_retrieval_resolves_metric_candidates_before_semantic_context(monkeypatch):
    settings = get_settings().model_copy(update={"embedding_api_key": "", "llm_api_key": "", "cache_enabled": False})
    topic_assets = TopicAssetService(settings)
    service = EsKnowledgeRetrievalService(settings, topic_assets)

    def fake_search(query_text, topics, include_base_wiki=False):
        return [
            RecallItem(
                doc_id="semantic:商品管理:dwm_goods_detail_df:table",
                title="goods table",
                content="商品表说明",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="商品管理",
                table="dwm_goods_detail_df",
                fusion_score=8.0,
                metadata={"semanticRefId": "semantic:商品管理:dwm_goods_detail_df:table"},
            )
        ]

    monkeypatch.setattr(service, "_search", fake_search)
    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="最近30天退款金额最高的前5个商品",
            topic_categories=[QuestionCategory.REFUND, QuestionCategory.GOODS],
        )
    )

    trace = bundle.recall_rounds[0]
    assert trace.metric_candidates
    pay_candidate = next(item for item in trace.metric_candidates if item["metricKey"] == "pay_amt")
    assert pay_candidate["matchedMetricLabel"] == "退款金额"
    assert pay_candidate["metricResolutionType"].startswith("exact")
    evidence = next(item for item in bundle.recall_bundle.items if item.doc_id.endswith(":pay_amt"))
    assert evidence.metadata["metricResolutionReason"].startswith("matched_")


def test_es_retrieval_limits_items_by_source_type_policy():
    items = [
        RecallItem(doc_id="metric_%d" % index, source_type="SEMANTIC_METRIC", fusion_score=100 - index)
        for index in range(20)
    ] + [
        RecallItem(doc_id="wiki_%d" % index, source_type="BASE_WIKI", fusion_score=90 - index)
        for index in range(10)
    ]
    limited = limit_recall_items_by_source_type(items, source_type_top_k_policy(include_base_wiki=False), limit=18)

    metric_count = sum(1 for item in limited if item.source_type == "SEMANTIC_METRIC")
    wiki_count = sum(1 for item in limited if item.source_type == "BASE_WIKI")
    assert metric_count <= 12
    assert wiki_count <= 3
    assert len(limited) == 15


def test_retrieval_profile_uses_dynamic_topk_for_focused_and_broad_queries():
    settings = get_settings()

    focused = build_retrieval_profile(
        query_text="最近7天退款金额",
        topics=["电商退货"],
        include_base_wiki=False,
        metric_candidates=[],
        settings=settings,
    )
    broad = build_retrieval_profile(
        query_text="最近30天退款金额最高的前5个商品，同时看下单量、商品发布时间，并分析是否存在异常波动",
        topics=["电商退货", "交易履约", "商品管理"],
        include_base_wiki=False,
        metric_candidates=[{"metricKey": "pay_amt"}, {"metricKey": "order_cnt"}],
        settings=settings,
    )

    assert focused["profileKind"] == "focused"
    assert focused["queryType"] == "simple_metric"
    assert focused["textTopK"] < int(settings.es_text_top_k or 12)
    assert focused["hybridTopK"] <= 16
    assert broad["profileKind"] == "broad"
    assert broad["queryType"] == "multi_hop_analysis"
    assert broad["textTopK"] > int(settings.es_text_top_k or 12)
    assert broad["hybridTopK"] > int(settings.es_hybrid_top_k or 24)


def test_retrieval_profile_uses_fast_understanding_complexity_override():
    settings = get_settings()

    profile = build_retrieval_profile(
        query_text="退款率",
        topics=["电商退货"],
        include_base_wiki=False,
        metric_candidates=[],
        settings=settings,
        intent_kind="analysis",
        complexity="complex",
    )

    assert profile["queryType"] == "multi_hop_analysis"
    assert profile["profileKind"] == "broad"
    assert profile["fastComplexity"] == "complex"
    assert any("fast_understanding" in reason for reason in profile["reasons"])


def test_retrieval_profile_can_be_overridden_by_config():
    settings = get_settings().model_copy(
        update={
            "es_retrieval_profiles_json": json.dumps(
                {
                    "simple_metric": {
                        "textTopK": 5,
                        "vectorTopK": 4,
                        "hybridTopK": 9,
                        "sourceTypeCaps": {"SEMANTIC_METRIC": 7, "BASE_WIKI": 1},
                    }
                },
                ensure_ascii=False,
            )
        }
    )
    profile = build_retrieval_profile(
        query_text="最近7天退款金额",
        topics=["电商退货"],
        include_base_wiki=False,
        metric_candidates=[],
        settings=settings,
    )
    policy = source_type_top_k_policy(
        include_base_wiki=False,
        query_text="最近7天退款金额",
        topics=["电商退货"],
        metric_candidates=[],
        retrieval_profile=profile,
    )

    assert profile["textTopK"] == 5
    assert profile["vectorTopK"] == 4
    assert profile["hybridTopK"] == 9
    assert policy["SEMANTIC_METRIC"] >= 7


def test_source_type_topk_policy_expands_relationships_for_multi_hop_queries():
    policy = source_type_top_k_policy(
        include_base_wiki=False,
        query_text="最近30天退款金额最高的商品，同时看商品发布时间并关联订单情况",
        topics=["电商退货", "商品管理", "交易履约"],
        metric_candidates=[{"metricKey": "pay_amt"}],
        retrieval_profile={"profileKind": "broad"},
    )

    assert policy["SEMANTIC_RELATIONSHIP"] >= 10
    assert policy["SEMANTIC_METRIC"] >= 14
    assert policy["SEMANTIC_TABLE_ASSET"] >= 8


def test_es_retrieval_trace_includes_query_type_and_lanes(monkeypatch):
    settings = get_settings().model_copy(update={"embedding_api_key": "", "llm_api_key": "", "cache_enabled": False})
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))

    monkeypatch.setattr(
        service,
        "_search",
        lambda query_text, topics, include_base_wiki=False: [
            RecallItem(
                doc_id="semantic:test",
                title="退款金额",
                content="退款金额指标",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=10,
            )
        ],
    )

    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="最近30天退款金额最高的商品，同时看商品发布时间",
            topic_categories=[QuestionCategory.REFUND, QuestionCategory.GOODS],
        )
    )
    trace = bundle.recall_rounds[0]
    assert trace.query_type == "multi_hop_analysis"
    assert any(item["lane"] == "bm25_lane" for item in trace.retrieval_lanes)
    assert any(item["lane"] == "metric_candidate_lane" for item in trace.retrieval_lanes)


def test_rrf_fusion_prefers_items_ranked_by_both_channels():
    text_items = [
        RecallItem(doc_id="semantic:a", title="A", fusion_score=100, metadata={"recallChannel": "bm25"}),
        RecallItem(doc_id="semantic:b", title="B", fusion_score=80, metadata={"recallChannel": "bm25"}),
    ]
    vector_items = [
        RecallItem(doc_id="semantic:b", title="B", fusion_score=0.91, metadata={"recallChannel": "vector"}),
        RecallItem(doc_id="semantic:c", title="C", fusion_score=0.89, metadata={"recallChannel": "vector"}),
    ]

    fused = rrf_fuse_recall_items([("bm25", text_items), ("vector", vector_items)], rrf_k=60, score_scale=1000, limit=10)

    assert [item.doc_id for item in fused][:3] == ["semantic:b", "semantic:a", "semantic:c"]
    assert fused[0].metadata["recallFusion"] == "rrf"
    assert fused[0].metadata["rrfRanks"] == {"bm25": 2, "vector": 1}
    assert set(fused[0].metadata["recallChannels"]) == {"bm25", "vector"}


def test_es_hybrid_search_uses_vector_and_rrf_when_embedding_configured(monkeypatch):
    settings = get_settings().model_copy(
        update={
            "embedding_api_key": "test-key",
            "es_vector_enabled": True,
            "es_hybrid_top_k": 10,
            "cache_enabled": False,
        }
    )
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    monkeypatch.setattr(service, "_embed_text", lambda text: [0.1, 0.2])
    monkeypatch.setattr(
        service,
        "_text_search",
        lambda query_text, topics, include_base_wiki=False: [
            RecallItem(doc_id="semantic:a", title="A", fusion_score=100),
            RecallItem(doc_id="semantic:b", title="B", fusion_score=80),
        ],
    )
    monkeypatch.setattr(
        service,
        "_vector_search",
        lambda query_text, vector, topics, include_base_wiki=False: [
            RecallItem(doc_id="semantic:b", title="B", fusion_score=0.91),
            RecallItem(doc_id="semantic:c", title="C", fusion_score=0.89),
        ],
    )

    items = service._search("退款金额 商品发布时间", ["电商退货", "商品管理"])

    assert [item.doc_id for item in items][:3] == ["semantic:b", "semantic:a", "semantic:c"]
    assert items[0].metadata["recallFusion"] == "rrf"
    assert items[0].metadata["rrfRanks"] == {"bm25": 2, "vector": 1}


def test_es_hybrid_search_falls_back_to_text_without_embedding_key(monkeypatch):
    settings = get_settings().model_copy(update={"embedding_api_key": "", "llm_api_key": "", "es_vector_enabled": True})
    service = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    text_items = [RecallItem(doc_id="semantic:a", title="A", fusion_score=100)]
    monkeypatch.setattr(service, "_text_search", lambda query_text, topics, include_base_wiki=False: text_items)

    items = service._search("退款金额", ["电商退货"])

    assert items == text_items


def test_es_index_adapter_adds_content_vector_when_embedding_configured(monkeypatch):
    settings = get_settings().model_copy(update={"embedding_api_key": "test-key", "embedding_dims": 2, "es_vector_enabled": True})
    adapter = EsRecallIndexAdapter(settings)
    monkeypatch.setattr(adapter, "_embed_recall_item", lambda item: [0.1, 0.2])

    payload = adapter._recall_item_to_es_doc(
        RecallItem(doc_id="semantic:test", title="退款金额", content="退款金额指标定义", metadata={"semanticRefId": "semantic:test"})
    )

    assert payload["content_vector"] == [0.1, 0.2]
    assert payload["metadata"]["embeddingModel"] == settings.embedding_model
    assert payload["metadata"]["embeddingDims"] == 2


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
    topic_assets = TopicAssetService(settings)
    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    question = "最近45天优惠券金额投入最高的商品，退款率是否偏高？"
    recall_service = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    recall = recall_service.recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    pack = builder.compact(
        question,
        recall,
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    plan, requests, _ = QueryGraphPlanner(FakeCouponRefundRatePlannerLlm()).plan(question, [], "", recall, pack, [], [])
    if requests:
        scoped_recall = recall_service.recall(
            requests[0].query,
            KeywordExtractService().extract(requests[0].query),
            [],
            "",
            "100",
            [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
        )
        merged_recall = RecallBundle(items=merge_recall_items_for_test(recall.items, scoped_recall.items))
        pack = builder.compact(
            question,
            merged_recall,
            [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
        )
        plan, requests, _ = QueryGraphPlanner(FakeCouponRefundRatePlannerLlm()).plan(question, [], "", merged_recall, pack, [], [])
    assert not requests
    metrics_by_task = {intent.plan_task_id: intent.metric_name for intent in plan.intents}
    assert metrics_by_task["anchor_coupon"] == "coupon_total_amt"
    assert "refund_rate" in metrics_by_task.values()
    assert "order_detail_cnt" in metrics_by_task.values()
    derived = next(intent for intent in plan.intents if intent.metric_name == "refund_rate")
    assert derived.answer_mode == AnswerMode.DERIVED
    assert derived.preferred_table == ""
    assert any("DERIVED_REQUESTED_METRIC:refund_rate" in item for item in plan.compiler_trace)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [(gap.code, gap.evidence) for gap in validation.gaps]


def test_llm_understanding_compiler_applies_scope_constraints_before_metrics():
    class CouponOrderRefundRatioLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "reason": "coupon-used order scope with refund numerator",
                "questionUnderstanding": {
                    "analysisGrain": "order",
                    "analysisIntent": "none",
                    "requiresExplanation": False,
                    "requiredEvidenceIntents": [],
                    "rankingObjective": {
                        "metricRef": "order_detail_cnt",
                        "sourcePhrase": "有使用优惠券的订单",
                        "ownerTable": "dwm_trade_order_detail_di",
                        "objectiveType": "metric_total",
                        "groupByColumn": "seller_id",
                        "order": "desc",
                        "limit": 1,
                    },
                    "requestedMeasures": [
                        {"metricRef": "refund_bill_cnt", "sourcePhrase": "有退货的订单", "ownerTable": "dwm_trade_refund_detail_di"}
                    ],
                    "scopeConstraints": [
                        {
                            "scopeId": "coupon_used_orders",
                            "sourcePhrase": "有使用优惠券的订单",
                            "ownerTable": "dwm_coupon_detail_di",
                            "metricRef": "coupon_amt",
                            "entityGrain": "order",
                            "targetDomain": "order",
                            "required": True,
                        }
                    ],
                    "filters": [],
                    "timeWindowDays": 10,
                },
            }

    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    question = "最近10天有使用优惠券的订单有退货的订单占多少。"
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COUPON],
    )
    plan, requests, _ = QueryGraphPlanner(CouponOrderRefundRatioLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])

    assert not requests
    task_tables = {intent.plan_task_id: intent.preferred_table for intent in plan.intents}
    assert task_tables["anchor_coupon_scope"] == "dwm_coupon_detail_di"
    assert task_tables["order_scope"] == "dwm_trade_order_detail_di"
    assert "order_lookup" in task_tables
    assert "refund_lookup" in task_tables
    order_scope = next(intent for intent in plan.intents if intent.plan_task_id == "order_scope")
    assert {"discount_rel_id", "sub_order_id"} <= set(order_scope.output_keys)
    assert any(dep.anchor_task_id == "anchor_coupon_scope" and dep.dependent_task_id == "order_scope" for dep in plan.dependencies)
    assert any(dep.anchor_task_id == "order_scope" and dep.dependent_task_id == "order_lookup" for dep in plan.dependencies)
    assert any(dep.anchor_task_id == "order_scope" and dep.dependent_task_id == "refund_lookup" for dep in plan.dependencies)
    ratio = next((intent for intent in plan.intents if intent.answer_mode == AnswerMode.DERIVED and str((intent.metric_resolution or {}).get("computeStrategy") or "") == "scope_event_ratio"), None)
    assert ratio is not None
    assert "refund_bill_cnt" in ratio.metric_formula
    assert "order_detail_cnt" in ratio.metric_formula
    assert any(dep.dependent_task_id == ratio.plan_task_id and dep.relation_type == "DERIVED_COMPONENT" for dep in plan.dependencies)
    assert any(item.startswith("SCOPE_CONSTRAINT:") for item in plan.compiler_trace)
    assert any(item.startswith("SCOPE_EVENT_RATIO:") for item in plan.compiler_trace)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert validation.valid, [(gap.code, gap.evidence) for gap in validation.gaps]


def test_scope_ratio_rejects_same_numerator_and_denominator_metric():
    settings = get_settings()
    question = "最近10天使用优惠券的订单中，有退货的订单占多少"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COUPON],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "requiredEvidenceIntents": [],
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "sourcePhrase": "订单",
            "ownerTable": "dwm_trade_order_detail_di",
            "objectiveType": "metric_total",
            "groupByColumn": "seller_id",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "sourcePhrase": "订单", "ownerTable": "dwm_trade_order_detail_di"}
        ],
        "calculationIntents": [
            {
                "operation": "percentage",
                "sourcePhrase": "占多少",
                "numeratorMetricRef": "order_detail_cnt",
                "denominatorMetricRef": "order_detail_cnt",
                "groupByColumn": "seller_id",
            }
        ],
        "scopeConstraints": [
            {
                "scopeId": "coupon_orders",
                "sourcePhrase": "优惠券",
                "ownerTable": "dwm_coupon_detail_di",
                "metricRef": "coupon_amt",
                "entityGrain": "order",
                "targetDomain": "order",
                "required": True,
            }
        ],
        "filters": [],
        "timeWindowDays": 10,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    validation = QueryGraphValidator().validate(question, plan, pack)

    assert not any(
        intent.answer_mode == AnswerMode.DERIVED
        and str((intent.metric_resolution or {}).get("computeStrategy") or "") == "scope_event_ratio"
        for intent in plan.intents
    )
    assert any(item.startswith("CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR:") for item in plan.compiler_trace)
    assert any(gap.code == "CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR" for gap in validation.gaps)
    assert not validation.recommended_knowledge_requests


def test_planner_payload_tells_llm_to_fix_invalid_ratio_numerator():
    settings = get_settings()
    question = "最近10天使用优惠券的订单中，有退货的订单占多少"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COUPON],
    )
    previous_understanding = {
        "rankingObjective": {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di"},
        "calculationIntents": [
            {
                "operation": "percentage",
                "sourcePhrase": "占多少",
                "basePopulationPhrase": "使用优惠券的订单",
                "eventPopulationPhrase": "有退货的订单",
                "numeratorMetricRef": "order_detail_cnt",
                "denominatorMetricRef": "order_detail_cnt",
            }
        ],
    }
    gaps = [
        GraphValidationGap(
            code="CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR",
            evidence="占多少",
            task_id="scope_event_ratio",
            reason="calculation numerator and denominator resolve to the same canonical metric",
        )
    ]

    feedback = planner_repair_feedback_for_understanding(gaps, previous_understanding)
    payload = QueryGraphPlanner(FakeCaseUnderstandingLlm(), settings=settings)._understanding_payload(
        question,
        pack,
        gaps,
        ["CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR:占多少"],
        False,
        False,
        None,
        prior_understanding=previous_understanding,
    )

    assert feedback["mustFixBeforePlanning"] is True
    assert feedback["calculation"][0]["invalidNumeratorMetricRef"] == "order_detail_cnt"
    assert feedback["calculation"][0]["invalidDenominatorMetricRef"] == "order_detail_cnt"
    assert "must not resolve to the same canonical metric" in feedback["calculation"][0]["instruction"]
    assert payload["repairFeedback"] == feedback
    assert payload["outputContract"]["repairRule"]
    assert "event=有退货的订单" in payload["outputContract"]["populationRatioExamples"][0]


def test_scope_ratio_numerator_knowledge_request_keeps_event_context():
    question = "最近10天使用优惠券的订单中，有退货的订单占多少"
    understanding = {
        "rankingObjective": {"metricRef": "order_detail_cnt"},
        "requestedMeasures": [{"metricRef": "order_detail_cnt", "sourcePhrase": "订单"}],
        "calculationIntents": [
            {
                "operation": "percentage",
                "sourcePhrase": "占多少",
                "basePopulationPhrase": "使用优惠券的订单",
                "eventPopulationPhrase": "有退货的订单",
                "numeratorMetricRef": "order_detail_cnt",
                "denominatorMetricRef": "order_detail_cnt",
            }
        ],
        "scopeConstraints": [
            {"scopeId": "coupon_orders", "sourcePhrase": "使用优惠券", "targetDomain": "order"},
            {"scopeId": "refund_orders", "sourcePhrase": "有退货的订单", "targetDomain": "order"},
        ],
    }

    request = scope_ratio_numerator_knowledge_request(question, understanding, "CALCULATION_NUMERATOR_MISSING")

    assert "有退货的订单" in request.query
    assert request.source_phrase == "有退货的订单"
    assert "使用优惠券的订单" in request.query
    assert "使用优惠券" in request.query
    assert "invalid numerator=order_detail_cnt" in request.query
    assert "denominator=order_detail_cnt" in request.query
    assert "event subset metric" in request.query


def test_compiler_repairs_scope_source_domain_from_semantic_target_hint():
    question = "最近10天使用优惠券的订单中，有退货的订单占多少"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_coupon_detail_di", columns=["seller_id", "coupon_id", "pt"]),
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "order_id", "sub_order_id", "discount_rel_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "order_id", "sub_order_id", "refund_id", "discount_id", "pt"],
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                metadata={"metricKey": "order_detail_cnt", "formula": "COUNT(DISTINCT sub_order_id)"},
            ),
            PlanningAssetEntry(
                key="refund_bill_cnt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_id"],
                metadata={"metricKey": "refund_bill_cnt", "formula": "COUNT(DISTINCT refund_id)"},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="coupon_order_by_coupon_id",
                left_table="dwm_coupon_detail_di",
                right_table="dwm_trade_order_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "coupon_id", "rightColumn": "discount_rel_id"},
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
    understanding = {
        "analysisGrain": "order",
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "groupByColumn": "order_id",
            "sourcePhrase": "使用优惠券的订单中有退货的订单占多少",
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单"},
            {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "有退货的订单"},
        ],
        "scopeConstraints": [
            {
                "scopeId": "coupon_used_orders",
                "sourcePhrase": "使用优惠券的订单",
                "ownerTable": "dwm_trade_order_detail_di",
                "metricRef": "order_detail_cnt",
                "entityGrain": "order",
                "targetDomain": "coupon",
                "required": True,
            }
        ],
        "calculationIntents": [
            {
                "operation": "ratio",
                "sourcePhrase": "使用优惠券的订单中有退货的订单占多少",
                "basePopulationPhrase": "使用优惠券的订单",
                "eventPopulationPhrase": "有退货的订单",
                "denominatorMetricRef": "order_detail_cnt",
                "numeratorMetricRef": "refund_bill_cnt",
                "groupByColumn": "order_id",
            }
        ],
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    validation = QueryGraphValidator().validate(question, plan, pack)

    assert any("SCOPE_SOURCE_DOMAIN_REPAIRED" in item for item in plan.compiler_trace)
    assert any(intent.preferred_table == "dwm_coupon_detail_di" for intent in plan.intents)
    assert any(
        dep.anchor_column == "seller_id+coupon_id"
        and dep.dependent_column == "seller_id+discount_rel_id"
        for dep in plan.dependencies
    )
    assert not any(gap.code.startswith("SCOPE_") for gap in validation.gaps)


def test_semantic_metric_resolver_uses_event_population_phrase_for_ratio_numerator():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "refund_id", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                title="订单量",
                columns=["order_id"],
                metadata={"metricKey": "order_detail_cnt", "formula": "COUNT(DISTINCT order_id)"},
            ),
            PlanningAssetEntry(
                key="refund_bill_cnt",
                table="dwm_trade_refund_detail_di",
                title="退款单量",
                columns=["refund_id"],
                aliases=["退款量", "退款订单量", "有退货的订单"],
                metadata={"metricKey": "refund_bill_cnt", "formula": "COUNT(DISTINCT refund_id)"},
            ),
        ],
        metric_compaction={
            "recalledMetricEvidence": [
                {
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "metricKey": "refund_bill_cnt",
                    "businessName": "退款单量",
                    "aliases": ["退款量", "退款订单量", "有退货的订单"],
                    "recallQueries": ["有退货的订单 语义指标口径"],
                    "fusionScore": 9.5,
                }
            ]
        },
    )
    understanding = {
        "calculationIntents": [
            {
                "operation": "percentage",
                "sourcePhrase": "占多少",
                "basePopulationPhrase": "使用优惠券的订单",
                "eventPopulationPhrase": "有退货的订单",
                "numeratorMetricRef": "order_detail_cnt",
                "denominatorMetricRef": "order_detail_cnt",
            }
        ]
    }

    resolution, gap = SemanticMetricResolver(pack).resolve_event_population_metric(
        question="最近10天使用优惠券的订单中，有退货的订单占多少",
        understanding=understanding,
        numerator_metric_ref="order_detail_cnt",
        denominator_table="dwm_trade_order_detail_di",
        denominator_metric_ref="order_detail_cnt",
    )

    assert gap == ""
    assert resolution.metric.key == "refund_bill_cnt"
    assert resolution.source_phrase == "有退货的订单"
    assert resolution.resolution_source == "semantic_recall_evidence"


def test_scope_ratio_resolves_rate_metric_to_event_component_numerator():
    settings = get_settings()
    question = "最近10天使用优惠券的订单中，有退货的订单占多少"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.COUPON],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "requiredEvidenceIntents": [],
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "sourcePhrase": "订单",
            "ownerTable": "dwm_trade_order_detail_di",
            "objectiveType": "metric_total",
            "groupByColumn": "seller_id",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "refund_rate", "sourcePhrase": "退货", "ownerTable": "dwm_trade_refund_detail_di"}
        ],
        "calculationIntents": [
            {
                "operation": "percentage",
                "sourcePhrase": "占多少",
                "basePopulationPhrase": "使用优惠券的订单",
                "eventPopulationPhrase": "有退货的订单",
                "numeratorMetricRef": "refund_rate",
                "denominatorMetricRef": "order_detail_cnt",
                "groupByColumn": "seller_id",
            }
        ],
        "scopeConstraints": [
            {
                "scopeId": "coupon_orders",
                "sourcePhrase": "使用优惠券",
                "ownerTable": "dwm_coupon_detail_di",
                "metricRef": "coupon_amt",
                "entityGrain": "order",
                "targetDomain": "order",
                "required": True,
            }
        ],
        "filters": [],
        "timeWindowDays": 10,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    validation = QueryGraphValidator().validate(question, plan, pack)

    ratio = next(
        (
            intent
            for intent in plan.intents
            if intent.answer_mode == AnswerMode.DERIVED
            and str((intent.metric_resolution or {}).get("computeStrategy") or "") == "scope_event_ratio"
        ),
        None,
    )
    assert ratio is not None
    assert "refund_bill_cnt" in ratio.metric_formula
    assert "order_detail_cnt" in ratio.metric_formula
    assert any(item.startswith("SCOPE_EVENT_RATIO:") and "refund_bill_cnt/order_detail_cnt" in item for item in plan.compiler_trace)
    assert not any(gap.code == "CALCULATION_NUMERATOR_NOT_EVENT_METRIC" for gap in validation.gaps)


def test_planner_reflection_rejects_same_table_scope_without_filter_or_subset_metric():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "order_id", "pt"],
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                metadata={"metricKey": "order_detail_cnt", "formula": "COUNT(DISTINCT sub_order_id)"},
            )
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "order",
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "某业务集合里的订单",
            },
            "requestedMeasures": [],
            "scopeConstraints": [
                {
                    "scopeId": "same_table_fake_scope",
                    "sourcePhrase": "某业务集合里的订单",
                    "ownerTable": "dwm_trade_order_detail_di",
                    "metricRef": "order_detail_cnt",
                    "entityGrain": "order",
                    "targetDomain": "order",
                    "required": True,
                }
            ],
            "filters": [],
        },
        intents=[
            QuestionIntent(
                question="某业务集合里的订单占比",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                task_role=TaskRole.ANCHOR,
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                metric_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                metric_resolution={
                    "requestedMetricRef": "order_detail_cnt",
                    "metricKey": "order_detail_cnt",
                    "ownerTable": "dwm_trade_order_detail_di",
                },
            )
        ],
    )

    validation = QueryGraphValidator().validate("某业务集合里的订单占比", plan, pack)
    assert validation.valid
    reflection = PlannerReflectionAgent().reflect("某业务集合里的订单占比", plan, pack)
    assert "SCOPE_NOT_NARROWING" in {issue["code"] for issue in reflection.issues}


def test_compiler_ignores_scope_that_duplicates_ranking_objective():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_scm_detail_di",
                columns=["seller_id", "spu_id", "inbound_cnt", "pt"],
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="scm_inbound_total_cnt",
                table="dwm_scm_detail_di",
                columns=["inbound_cnt"],
                title="入库数量",
                metadata={"metricKey": "scm_inbound_total_cnt", "formula": "SUM(inbound_cnt)", "sourceColumns": ["inbound_cnt"]},
            )
        ],
    )
    understanding = {
        "analysisGrain": "product",
        "rankingObjective": {
            "metricRef": "scm_inbound_total_cnt",
            "ownerTable": "dwm_scm_detail_di",
            "groupByColumn": "spu_id",
            "limit": 10,
        },
        "requestedMeasures": [],
        "scopeConstraints": [
            {
                "scopeId": "top10_inbound_products",
                "sourcePhrase": "供应链入库量前10商品",
                "ownerTable": "dwm_scm_detail_di",
                "metricRef": "scm_inbound_total_cnt",
                "entityGrain": "product",
                "targetDomain": "scm",
                "required": True,
            }
        ],
        "filters": [],
    }

    plan = compile_query_graph_from_understanding("供应链入库量前10商品", understanding, pack)
    task_ids = {intent.plan_task_id for intent in plan.intents}
    assert "anchor_scm_scope" not in task_ids
    assert "anchor_scm" in task_ids
    validation = QueryGraphValidator().validate("供应链入库量前10商品", plan, pack)
    assert not any(gap.code == "SCOPE_NOT_NARROWING" for gap in validation.gaps)


def test_planner_critic_accepts_scope_anchor_before_ranking_metric_node():
    plan = QueryPlan(
        question_understanding={
            "analysisGrain": "order",
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "优惠券订单数",
            },
            "requestedMeasures": [],
            "scopeConstraints": [
                {
                    "scopeId": "coupon_used_orders",
                    "sourcePhrase": "使用优惠券的订单中",
                    "ownerTable": "dwm_coupon_detail_di",
                    "metricRef": "coupon_total_amt",
                    "entityGrain": "order",
                    "targetDomain": "order",
                    "required": True,
                }
            ],
        },
        intents=[
            QuestionIntent(
                question="最近10天使用优惠券的订单中，有退货的订单占多少",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                task_role=TaskRole.ANCHOR,
                plan_task_id="anchor_coupon_scope",
                preferred_table="dwm_coupon_detail_di",
                output_keys=["seller_id", "coupon_id"],
            ),
            QuestionIntent(
                question="最近10天使用优惠券的订单中，有退货的订单占多少",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                task_role=TaskRole.DEPENDENT,
                plan_task_id="order_lookup",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                metric_column="sub_order_id",
                metric_resolution={
                    "requestedMetricRef": "order_detail_cnt",
                    "metricKey": "order_detail_cnt",
                    "ownerTable": "dwm_trade_order_detail_di",
                },
            ),
        ],
    )

    assert anchor_mismatch_issue(plan) == {}


def test_query_graph_validator_rejects_dependency_key_not_produced_by_detail_node():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "refund_id", "sub_order_id", "discount_id", "pt"]),
            PlanningAssetEntry(table="dwm_coupon_detail_di", columns=["seller_id", "coupon_id", "coupon_amt", "pt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="coupon_refund_by_coupon_id",
                left_table="dwm_coupon_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "coupon_id", "rightColumn": "discount_id"},
                ],
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="优惠券退款",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                task_role=TaskRole.ANCHOR,
                plan_task_id="refund_entity_expand",
                preferred_table="dwm_trade_refund_detail_di",
                output_keys=["seller_id", "refund_id", "sub_order_id"],
            ),
            QuestionIntent(
                question="优惠券退款",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                task_role=TaskRole.DEPENDENT,
                plan_task_id="coupon_lookup",
                preferred_table="dwm_coupon_detail_di",
                metric_name="coupon_amt",
                metric_column="coupon_amt",
                group_by_column="coupon_id",
                output_keys=["seller_id", "coupon_id"],
                depends_on_task_ids=["refund_entity_expand"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="refund_entity_expand",
                dependent_task_id="coupon_lookup",
                join_key="coupon_id",
                anchor_column="seller_id+discount_id",
                dependent_column="seller_id+coupon_id",
            )
        ],
    )

    validation = QueryGraphValidator().validate("优惠券退款", plan, pack)
    assert not validation.valid
    assert any(gap.code == "DEPENDENCY_KEY_NOT_PRODUCED" and gap.evidence == "discount_id" for gap in validation.gaps)


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
    assert "semanticFileContext" not in llm.payloads[0]
    assert "semanticManifest" not in llm.payloads[0]
    assert llm.payloads[0]["semanticCatalog"]["tables"]


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
            PlanningAssetEntry(
                key="goods_publish_time",
                table="dwm_goods_detail_df",
                columns=["spu_apply_create_time"],
                title="商品发布时间",
                metadata={"sourceColumns": ["spu_apply_create_time"]},
            ),
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


def test_detail_goods_lookup_does_not_require_missing_status_code_metric():
    class DetailSubOrderLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "questionUnderstanding": {
                        "analysisGrain": "order",
                        "rankingObjective": {},
                        "requestedMeasures": [
                            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
                            {
                                "metricRef": "goods_publish_time",
                                "ownerTable": "dwm_goods_detail_df",
                                "sourcePhrase": "商品发布信息",
                            },
                        ],
                        "filters": [{"field": "sub_order_id", "value": "sub_order_id_100"}],
                        "timeWindowDays": 7,
                    },
            }

    question = "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。"
    pack = trade_refund_goods_pack(include_missing_goods_metric=True)
    plan, requests, _ = QueryGraphPlanner(DetailSubOrderLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    goods = next(intent for intent in plan.intents if intent.preferred_table == "dwm_goods_detail_df")
    assert goods.answer_mode == AnswerMode.DETAIL
    assert not goods.metric_name
    assert "spu_status_code" not in goods.required_evidence
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_topn_anchor_uses_entity_expansion_for_unproduced_refund_key():
    class TopSpuLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "questionUnderstanding": {
                    "analysisGrain": "product",
                    "rankingObjective": {
                        "metricRef": "order_detail_cnt",
                        "ownerTable": "dwm_trade_order_detail_di",
                        "sourcePhrase": "下单最多的前 5 个 SPU",
                        "groupByColumn": "spu_id",
                        "limit": 5,
                        "order": "desc",
                    },
                    "requestedMeasures": [
                        {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款量"},
                        {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
                    ],
                    "timeWindowDays": 7,
                },
            }

    question = "最近 7 天下单最多的前 5 个 SPU，同时看它们的退款量和退款金额。"
    pack = trade_refund_goods_pack()
    plan, requests, _ = QueryGraphPlanner(TopSpuLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    anchor = next(intent for intent in plan.intents if intent.plan_task_id == "anchor_order")
    assert "sub_order_id" not in anchor.output_keys
    assert any(intent.plan_task_id == "order_entity_expand" for intent in plan.intents)
    assert any(dep.anchor_task_id == "order_entity_expand" and dep.dependent_task_id.startswith("refund_lookup") for dep in plan.dependencies)
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_need_more_planner_fails_closed_instead_of_semantic_entity_chain():
    class NeedMoreLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "NEED_MORE_KNOWLEDGE",
                "reason": "need more despite loaded semantic assets",
                "knowledgeRequests": [{"type": "TABLE", "query": "refund goods"}],
            }

    question = "最近 7 天有退款的订单，关联看一下对应商品发布时间。"
    pack = trade_refund_goods_pack()
    plan, requests, _ = QueryGraphPlanner(NeedMoreLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert requests
    assert not plan.intents
    assert any("planner.need_more_fail_closed" in item for item in plan.agent_trace)
    assert not any("semantic_entity_chain_fallback" in item for item in plan.agent_trace)
    validation = QueryGraphValidator().validate(question, plan, pack)
    assert not validation.valid
    assert any(gap.code == "MISSING_QUERY_GRAPH" for gap in validation.gaps)


def test_detail_chain_contract_does_not_require_dependent_product_key_when_goods_node_covers_it():
    class DetailSubOrderLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "questionUnderstanding": {
                    "analysisGrain": "order",
                    "analysisIntent": "none",
                        "requiresExplanation": False,
                            "requiredEvidenceIntents": [],
                            "rankingObjective": {},
                            "requestedMeasures": [
                                {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款"},
                                {
                                    "metricRef": "goods_publish_time",
                                    "ownerTable": "dwm_goods_detail_df",
                                    "sourcePhrase": "商品发布信息",
                            },
                        ],
                        "filters": [{"field": "sub_order_id", "value": "sub_order_id_100"}],
                        "timeWindowDays": 7,
                    },
            }

    question = "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。"
    pack = trade_refund_goods_pack()
    plan, requests, _ = QueryGraphPlanner(DetailSubOrderLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    refund_contract = next(contract for contract in plan.evidence_contracts if contract["taskId"] == "refund_lookup")
    assert ["spu_id", "spu_name"] not in refund_contract.get("columnsAnyOf", [])
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="anchor_order",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[
                        {
                            "seller_id": "100",
                            "sub_order_id": "sub_order_id_100",
                            "order_id": "order_id_100",
                            "spu_id": "spu_id_100",
                            "spu_name": "商品A",
                            "discount_rel_id": "",
                            "pt": "20260620",
                            "pay_amt": 100,
                        }
                    ],
                ),
            ),
            AgentTaskResult(
                task_id="refund_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[{"seller_id": "100", "sub_order_id": "sub_order_id_100", "order_id": "order_id_100", "refund_bill_cnt": 1}],
                ),
            ),
            AgentTaskResult(
                task_id="goods_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[
                        {
                            "seller_id": "100",
                            "spu_id": "spu_id_100",
                            "spu_name": "商品A",
                            "spu_apply_create_time": "2026-06-01 10:00:00",
                            "spu_status_name": "已发布",
                            "pt": "20260620",
                        }
                    ],
                ),
            ),
        ],
        merged_query_bundle=QueryBundle(
            tables=["dwm_trade_order_detail_di", "dwm_trade_refund_detail_di", "dwm_goods_detail_df"],
            rows=[
                {"seller_id": "100", "sub_order_id": "sub_order_id_100", "spu_id": "spu_id_100", "spu_apply_create_time": "2026-06-01 10:00:00"},
                {"seller_id": "100", "sub_order_id": "sub_order_id_100", "refund_bill_cnt": 1},
            ],
        ),
    )
    verified = EvidenceVerifier().verify(question, plan, run)
    assert verified.passed
    assert not any(gap.code == "MISSING_ENTITY_KEY" for gap in verified.gaps)


def test_detail_lookup_metric_resolution_does_not_bind_missing_formula_columns():
    class DetailSubOrderWithGoodsMetricLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "questionUnderstanding": {
                    "analysisGrain": "order",
                    "analysisIntent": "none",
                    "requiresExplanation": False,
                    "requiredEvidenceIntents": [],
                    "rankingObjective": {},
                    "requestedMeasures": [
                        {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单"},
                        {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款"},
                        {"metricRef": "goods_online_detail_cnt", "ownerTable": "dwm_goods_detail_df", "sourcePhrase": "商品发布信息"},
                    ],
                    "filters": [{"field": "sub_order_id", "value": "sub_order_id_100"}],
                    "timeWindowDays": 7,
                },
            }

    question = "最近 7 天查询子订单 sub_order_id_100 的订单、退款和商品发布信息。"
    pack = trade_refund_goods_pack(include_missing_goods_metric=True)
    plan, requests, _ = QueryGraphPlanner(DetailSubOrderWithGoodsMetricLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    goods = next(intent for intent in plan.intents if intent.plan_task_id == "goods_lookup")
    assert goods.answer_mode == AnswerMode.DETAIL
    assert goods.metric_name == ""
    assert goods.metric_column == ""
    assert goods.metric_formula == ""
    assert "spu_apply_create_time" in goods.required_evidence
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_relationship_chain_contract_does_not_require_unrequested_coupon_key():
    question = "最近 7 天有退款的订单，关联看一下对应商品发布时间。"
    pack = trade_refund_goods_pack()
    plan = compile_semantic_entity_chain_graph(question, pack)
    assert plan.intents
    assert "coupon_id" not in json.dumps(plan.evidence_contracts, ensure_ascii=False)
    assert "discount_rel_id" not in json.dumps(plan.evidence_contracts, ensure_ascii=False)
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="anchor_refund",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[
                        {
                            "seller_id": "100",
                            "sub_order_id": "sub_order_id_100",
                            "order_id": "order_id_100",
                            "spu_name": "商品A",
                            "refund_id": "refund_id_100",
                            "pt": "20260620",
                            "refund_status_name": "退款成功",
                            "refund_create_time": "2026-06-20 10:00:00",
                            "pay_amt": 10,
                        }
                    ],
                ),
            ),
            AgentTaskResult(
                task_id="order_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[
                        {
                            "seller_id": "100",
                            "sub_order_id": "sub_order_id_100",
                            "order_id": "order_id_100",
                            "spu_id": "spu_id_100",
                            "spu_name": "商品A",
                            "pt": "20260620",
                            "pay_amt": 100,
                            "order_detail_cnt": 1,
                        }
                    ],
                ),
            ),
            AgentTaskResult(
                task_id="goods_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[
                        {
                            "seller_id": "100",
                            "spu_id": "spu_id_100",
                            "spu_name": "商品A",
                            "spu_apply_create_time": "2026-06-01 10:00:00",
                            "spu_status_name": "已发布",
                            "pt": "20260620",
                        }
                    ],
                ),
            ),
        ],
        merged_query_bundle=QueryBundle(
            tables=["dwm_trade_refund_detail_di", "dwm_trade_order_detail_di", "dwm_goods_detail_df"],
            rows=[
                {"seller_id": "100", "sub_order_id": "sub_order_id_100", "refund_id": "refund_id_100", "order_detail_cnt": 1},
                {"seller_id": "100", "spu_id": "spu_id_100", "spu_apply_create_time": "2026-06-01 10:00:00"},
            ],
        ),
    )
    verified = EvidenceVerifier().verify(question, plan, run)
    assert verified.passed
    assert not any(gap.code == "MISSING_ENTITY_KEY" for gap in verified.gaps)


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
        question_understanding={
            "analysisGrain": "order",
            "rankingObjective": {
                "metricRef": "repay_amt",
                "ownerTable": "dwm_cs_repay_detail_df",
                "sourcePhrase": "赔付金额",
                "groupByColumn": "sub_order_id",
                "limit": 5,
            },
            "requestedMeasures": [
                {"metricRef": "pay_amt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单金额"},
                {
                    "metricRef": "refund_related_pay_amt",
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "sourcePhrase": "退款金额",
                },
                {"metricRef": "ticket_cnt", "ownerTable": "dwm_cs_ticket_detail_di", "sourcePhrase": "客服工单"},
            ],
        },
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


def test_llm_client_opens_circuit_after_timeout_and_fast_fails():
    class SlowModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            await asyncio.sleep(1.3)
            return type("Msg", (), {"content": "late"})()

    settings = get_settings().model_copy(
        update={
            "llm_request_timeout_seconds": 1,
            "llm_circuit_threshold": 1,
            "llm_circuit_cooldown_seconds": 30,
        }
    )
    client = LlmClient(settings)
    model = SlowModel()
    client._chat_model = lambda timeout_seconds=None: model

    first = client.chat("system", "user", fallback="fallback", timeout_seconds=1)
    second = client.chat("system", "user again", fallback="fallback", timeout_seconds=1)

    assert first == "fallback"
    assert second == "fallback"
    assert model.calls == 1
    assert "circuit_open" in client.last_error
    assert client.cache_trace()["circuit"]["open"]


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


def test_context_manager_protects_bi_slots_entities_and_corrections(tmp_path):
    settings = get_settings()
    manager = ContextManager(settings)
    state = {
        "question": "最近30天退款率最高的前5个商品，同时看工单量。",
        "requested_merchant_id": "seller_100",
        "thread_data": ThreadData(outputs_path=str(tmp_path)),
        "route_slots": {
            "timeWindow": {"days": 30, "raw": "最近30天"},
            "objectRefs": [{"refType": "spu_id", "value": "spu_1", "raw": "spu_1"}],
        },
        "fast_understanding": {
            "timeWindowDays": 30,
            "metricPhrases": ["退款率", "工单量"],
            "objectRefs": {"spu_id": ["spu_1"]},
        },
        "message_history": [
            {"role": "user", "text": "不对，退款率不是退款金额，要按退款单量 / 下单量。"},
        ],
        "memory_constraints": [
            {
                "memoryType": "correction",
                "summary": "退款率按退款单量/下单量解释",
                "metrics": ["refund_rate"],
                "enforcement": "required",
            }
        ],
        "plan": QueryPlan(
            question_understanding={
                "rankingObjective": {
                    "metricRef": "metric.refund_rate",
                    "sourcePhrase": "退款率",
                    "ownerTable": "ads_merchant_profile",
                    "groupByColumn": "spu_id",
                    "objectiveType": "ranking",
                },
                "requestedMeasures": [{"metricRef": "metric.cs_ticket_cnt_1d", "sourcePhrase": "工单量"}],
                "filters": [{"field": "spu_id", "value": "spu_1"}],
                "scopeConstraints": [{"ownerTable": "dwm_goods_detail_df", "field": "spu_id", "value": "spu_1", "required": True}],
                "requiredEvidenceIntents": ["ticket_count"],
                "analysisIntent": "risk_ranking",
                "timeWindowDays": 30,
            },
            intents=[
                QuestionIntent(
                    question="refund risk top spu",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_refund_rate",
                    preferred_table="ads_merchant_profile",
                    metric_name="refund_rate",
                    group_by_column="spu_id",
                    filter_column="spu_id",
                    filter_value="spu_1",
                    days=30,
                )
            ],
        ),
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="anchor_refund_rate",
                    success=True,
                    query_bundle=QueryBundle(rows=[{"spu_id": "spu_1", "refund_rate": 0.2}]),
                    entity_set=EntitySet(
                        task_id="anchor_refund_rate",
                        join_key="spu_id",
                        values=["spu_1", "spu_2"],
                        column_values={"spu_id": ["spu_1", "spu_2"]},
                        source_row_count=2,
                    ),
                )
            ],
            evidence_gaps=[EvidenceGap(code="MISSING_TICKET_EVIDENCE", task_id="ticket_lookup", reason="need ticket count")],
        ),
    }

    snapshot = manager.refresh_state(state, "verify_evidence_graph")
    protected = {fact.key: fact.value for fact in snapshot.protected_facts}

    assert protected["user_corrections"].startswith("不对，退款率")
    assert "metric.refund_rate" in protected["ranking_objective"]
    assert "metric.cs_ticket_cnt_1d" in protected["requested_measure:0"]
    assert "spu_id=spu_1" in protected["understanding_filters"]
    assert "task=anchor_refund_rate" in protected["reusable_entity_sets"]
    assert "MISSING_TICKET_EVIDENCE" in protected["evidence_gaps"]
    assert "退款率按退款单量" in protected["memory_constraints"]
    assert any(fact.category == "time_window" and "30" in fact.value for fact in snapshot.protected_facts)


def test_middleware_chain_summarizes_and_writes_workspace_manifest(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "context_window_tokens": 20,
            "context_compaction_threshold_ratio": 0.1,
            "context_compaction_target_ratio": 0.4,
        }
    )
    outputs = tmp_path / "threads" / "thread_mw" / "outputs"
    outputs.mkdir(parents=True)
    state = {
        "question": "最近7天 GMV、退款金额和工单量分别是多少？" * 80,
        "thread_data": ThreadData(thread_id="thread_mw", run_id="run_mw", workspace_path=str(outputs.parent), outputs_path=str(outputs)),
        "context_snapshots": [
            {
                "stage": "plan_query_graph",
                "summary": "important summary",
                "protectedFacts": [{"key": "question", "value": "GMV refund ticket"}],
            }
        ],
        "context_packages": [],
        "action_history": [],
        "middleware_events": [],
    }

    chain = MiddlewareChain([ContextBudgetMiddleware(settings), SummarizeMiddleware(settings), FileSystemContextMiddleware(settings)])
    state = chain.before_policy(state)

    assert state["context_budget_reports"][-1].over_budget
    assert state["context_compression_events"]
    assert state["context_compression_events"][-1].summary_artifact.path
    assert state["_runtime_context_stale"] is True
    assert (outputs / "workspace_manifest.json").exists()
    assert state["workspace_manifest"].entry_count >= 1
    assert any(event.code == "CONTEXT_SUMMARIZED" for event in state["middleware_events"])


def test_thread_context_service_restores_previous_artifacts_and_entities(tmp_path):
    outputs = tmp_path / "threads" / "thread_ctx" / "outputs"
    node_dir = outputs / "artifacts" / "node"
    node_dir.mkdir(parents=True)
    (outputs / "context_snapshot.json").write_text(json.dumps({"summary": "- plan_tables=dwm_trade_order_detail_di"}), encoding="utf-8")
    (outputs / "trace_replay.json").write_text(
        json.dumps({"runId": "run_prev", "question": "最近7天下单最多的前5个SPU", "answer": "Top SPU..."}),
        encoding="utf-8",
    )
    (node_dir / "agent_run_result.json").write_text(
        json.dumps(
            {
                "taskResults": [
                    {
                        "taskId": "anchor_top_spu",
                        "entitySet": {
                            "taskId": "anchor_top_spu",
                            "joinKey": "spu_id",
                            "values": ["spu_1", "spu_2"],
                            "sourceRowCount": 2,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state = {
        "thread_id": "thread_ctx",
        "run_id": "run_new",
        "checkpoint_thread_id": "thread_ctx:run_new",
        "thread_data": ThreadData(thread_id="thread_ctx", run_id="run_new", outputs_path=str(outputs)),
        "session_context": "",
    }

    context = ThreadContextService().restore(state)

    assert context["restored"]
    assert context["previousRunId"] == "run_prev"
    assert context["reusableEntitySets"][0]["joinKey"] == "spu_id"
    assert "reusableEntitySet" in state["session_context"]


def test_request_message_history_is_short_term_memory_context(tmp_path):
    request = ChatRequest.model_validate(
        {
            "message": "这些商品的下单量是多少？",
            "merchantId": "100",
            "messageHistory": [
                {"role": "user", "text": "最近30天退款金额最高的前5个商品"},
                {"role": "assistant", "text": "已找到 Top5 商品：spu_1、spu_2"},
                {"role": "user", "text": "这些商品的下单量是多少？"},
            ],
        }
    )
    assert len(request.message_history) == 3

    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        request.message,
        request.merchant_id,
        ChatContext(),
        None,
        "short_memory_thread",
        "short_memory_run",
        request.message_history,
    )

    state = workflow.inherit_context(state)

    assert "当前会话短期记忆" in state["session_context"]
    assert "最近30天退款金额最高的前5个商品" in state["session_context"]
    assert state["thread_context"]["messageHistory"][0]["role"] == "user"
    assert any("Short-term Memory" in step for step in state["thinking_steps"])


def test_old_message_history_uses_llm_summary_when_available(tmp_path):
    class FakeSummaryLlm:
        configured = True

        def __init__(self):
            self.calls = []

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls.append(
                {
                    "system": system_prompt,
                    "user": user_prompt,
                    "timeoutSeconds": timeout_seconds,
                }
            )
            return "## 旧会话压缩摘要\n- 已确认约束：最近30天。\n- 关键对象：spu_1、spu_2。\n- 未完成任务：继续补查下单量。"

    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    fake_llm = FakeSummaryLlm()
    workflow.planner.llm = fake_llm
    history = [
        {"role": "user", "text": "最近30天退款金额最高的前5个商品"},
        {"role": "assistant", "text": "已找到 Top 商品：spu_1、spu_2"},
        {"role": "user", "text": "这批商品发布时间是什么"},
        {"role": "assistant", "text": "发布时间分别是 2026-06-01 和 2026-06-02"},
        {"role": "user", "text": "先记住这些商品"},
        {"role": "assistant", "text": "已记录这些商品集合"},
        {"role": "user", "text": "那这些商品的下单量是多少"},
    ]
    state = workflow._initial_state(
        "那这些商品的下单量是多少",
        "100",
        ChatContext(),
        None,
        "short_memory_llm_thread",
        "short_memory_llm_run",
        history,
    )

    state = workflow.inherit_context(state)

    assert fake_llm.calls
    assert "旧会话压缩摘要" in state["session_context"]
    assert "spu_1、spu_2" in state["session_context"]
    assert "当前会话短期记忆" in state["session_context"]
    assert "那这些商品的下单量是多少" in state["session_context"]
    assert state["thread_context"]["messageHistorySummary"]["usedLlm"] is True
    assert state["thread_context"]["messageHistorySummary"]["summarySourceMessages"] == 1


def test_dynamic_context_middleware_injects_runtime_boundary(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_runtime_budget_chars": 1200})
    outputs = tmp_path / "threads" / "thread_dyn" / "outputs"
    outputs.mkdir(parents=True)
    state = {
        "question": "最近7天订单量是多少",
        "requested_merchant_id": "seller_100",
        "merchant": MerchantInfo(merchant_id="seller_100", merchant_name="测试商家"),
        "route_slots": RouteSlots(),
        "thread_context": {"restored": True, "previousQuestion": "上轮问题", "reusableEntitySets": [{"joinKey": "spu_id"}]},
        "workspace_manifest": {},
        "loaded_skills": ["trade"],
        "thread_data": ThreadData(thread_id="thread_dyn", run_id="run_dyn", outputs_path=str(outputs)),
        "middleware_events": [],
        "react_round": 1,
    }

    DynamicContextMiddleware(settings).before_policy(state)

    assert state["runtime_injection"]["merchant"]["merchantId"] == "seller_100"
    assert state["runtime_injection"]["threadContext"]["restored"]
    assert "currentDate" in state["runtime_context"]
    assert state["middleware_events"][-1].code == "RUNTIME_CONTEXT_INJECTED"


def test_dynamic_context_injects_standard_tool_failure_feedback():
    settings = get_settings()
    state = {
        "tool_call_results": [
            ToolCallExecutionResult(
                id="tool_bad",
                name="semantic_read",
                status="failed",
                error_type="INVALID_REF",
                error_message="semantic ref not found",
                retryable=True,
                recommended_action="semantic_grep",
                fallback_tools=["semantic_grep"],
            )
        ],
        "middleware_events": [],
    }

    DynamicContextMiddleware(settings).before_policy(state)

    feedback = state["runtime_injection"]["toolFeedback"][0]
    assert feedback["toolCallId"] == "tool_bad"
    assert feedback["toolName"] == "semantic_read"
    assert feedback["errorCode"] == "INVALID_REF"
    assert feedback["recommendedAction"] == "semantic_grep"
    assert "toolFeedback" in state["runtime_context"]


def test_dynamic_context_rerenders_after_summary_compaction(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    state = {
        "react_round": 2,
        "thread_data": ThreadData(thread_id="thread_ctx_stale", run_id="run_ctx_stale", outputs_path=str(tmp_path / "outputs")),
        "_runtime_context_stale": True,
        "middleware_events": [],
    }

    DynamicContextMiddleware(settings).before_policy(state)

    assert state["_runtime_context_stale"] is False
    assert state["middleware_events"][-1].metadata["rerenderedAfterCompaction"] is True


def test_memory_middleware_injects_and_store_writes_structured_memory(tmp_path):
    settings = get_settings().model_copy(
        update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 1600, "memory_backend": "file"}
    )
    store = StructuredMemoryStore(settings)
    state = {
        "question": "最近7天GMV是多少",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            question_understanding={"analysisIntent": "none"},
            intents=[
                QuestionIntent(
                    question="最近7天GMV是多少",
                    intent_type="VALID",
                    answer_mode="METRIC",
                    category=QuestionCategory.TRADE,
                    metric_name="gmv",
                    metric_resolution={"metricKey": "gmv_amt"},
                    days=7,
                )
            ],
        ),
        "answer": "最近7天GMV为100。",
        "middleware_events": [],
    }

    memory = store.update_from_state(state)
    assert memory["recentFocus"]["topMetrics"][0]["metric"] == "gmv_amt"
    assert memory["events"][0]["memoryTier"] == "retrieval"
    assert memory["preferences"][0]["memoryTier"] == "retrieval"
    assert not memory["coreMemoryProfile"]["corePreferences"]
    MemoryMiddleware(settings).before_policy(state)
    assert state["memory_injection"]["recentFocus"]["topMetrics"]
    assert not state["memory_injection"]["coreMemory"]["corePreferenceIds"]
    assert "answerPreview" in json.dumps(memory["events"][0], ensure_ascii=False)
    assert "tool_result" not in json.dumps(memory, ensure_ascii=False)


def test_habit_preferences_promote_to_core_only_after_repeat_or_explicit_signal(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    base_state = {
        "question": "最近7天退款率是多少",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近7天退款率是多少",
                    category=QuestionCategory.REFUND,
                    metric_resolution={"metricKey": "refund_rate"},
                    days=7,
                )
            ]
        ),
        "answer": "退款率为2%。",
    }

    first_memory = store.update_from_state(base_state)
    first_metric_pref = next(item for item in first_memory["preferences"] if item["key"] == "metric:refund_rate")
    assert first_metric_pref["memoryTier"] == "retrieval"
    assert first_metric_pref["hitCount"] == 1
    assert not first_memory["coreMemoryProfile"]["corePreferences"]

    second_memory = store.update_from_state(base_state)
    second_metric_pref = next(item for item in second_memory["preferences"] if item["key"] == "metric:refund_rate")
    assert second_metric_pref["memoryTier"] == "core"
    assert second_metric_pref["hitCount"] >= 2
    assert second_memory["coreMemoryProfile"]["corePreferences"]

    explicit_memory = store.update_from_state(
        {
            "question": "以后看售后风险默认优先看最近7天工单量",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="以后看售后风险默认优先看最近7天工单量",
                        category=QuestionCategory.CS_TICKET,
                        metric_resolution={"metricKey": "ticket_count"},
                        days=7,
                    )
                ]
            ),
            "answer": "已记录售后风险分析偏好。",
        }
    )
    ticket_pref = next(item for item in explicit_memory["preferences"] if item["key"] == "metric:ticket_count")
    assert ticket_pref["memoryTier"] == "core"


def test_memory_injection_uses_token_budget_by_default(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    store = StructuredMemoryStore(settings)
    store.update_from_state(
        {
            "question": "最近7天退款率是多少",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近7天退款率是多少",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                        days=7,
                    )
                ]
            ),
            "answer": "退款率为2%。",
        }
    )

    selected = store.select_for_question(
        {
            "question": "最近7天退款率",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"}, days=7)]),
        }
    )

    trace = selected["memoryInjectionTrace"]
    assert trace["budgetTokens"] == 1200
    assert trace["budgetUsedTokens"] <= 1200


def test_memory_token_budget_estimation_is_conservative_for_chinese(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    chinese_text = "退款率" * 300

    assert estimate_memory_tokens(chinese_text) >= 900
    assert estimate_text_tokens(chinese_text) >= 900
    assert estimate_memory_tokens(truncate_memory_text_by_tokens(chinese_text, 200)) <= 200
    assert memory_budget_tokens(settings, budget_chars=1200) == 300


def test_memory_store_does_not_persist_sql_rows_or_tool_trace(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 1600})
    store = StructuredMemoryStore(settings)
    state = {
        "question": "最近7天退款金额是多少",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近7天退款金额是多少",
                    intent_type="VALID",
                    answer_mode="METRIC",
                    category=QuestionCategory.REFUND,
                    metric_name="退款金额",
                    metric_resolution={"metricKey": "refund_related_pay_amt"},
                    days=7,
                )
            ],
        ),
        "answer": "最近7天退款金额为100。",
        "query_bundle": QueryBundle(
            sql="SELECT secret_column FROM dwm_trade_refund_detail_di",
            rows=[{"secret_column": "should_not_persist", "refund_related_pay_amt": 100}],
        ),
        "tool_context": "tool_result: full doris rows should_not_persist",
        "node_tool_traces": [{"sql": "SELECT secret_column FROM table"}],
    }

    memory = store.update_from_state(state)
    serialized = json.dumps(memory, ensure_ascii=False)
    assert "refund_related_pay_amt" in serialized
    assert "SELECT secret_column" not in serialized
    assert "should_not_persist" not in serialized
    assert "tool_result" not in serialized


def test_memory_correction_is_prioritized_and_marks_conflict(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    old_state = {
        "question": "最近7天退款金额是多少",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近7天退款金额是多少",
                    intent_type="VALID",
                    answer_mode="METRIC",
                    category=QuestionCategory.REFUND,
                    metric_name="退款金额",
                    metric_resolution={"metricKey": "refund_related_pay_amt"},
                    days=7,
                )
            ],
        ),
        "answer": "最近7天退款金额为100。",
    }
    store.update_from_state(old_state)
    correction_state = {
        "question": "不对，不是退款金额，是退款率，以后看售后风险要按退款率。",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="不对，不是退款金额，是退款率",
                    intent_type="VALID",
                    answer_mode="DERIVED",
                    category=QuestionCategory.REFUND,
                    metric_name="退款率",
                    metric_resolution={"metricKey": "refund_rate"},
                    days=7,
                )
            ],
        ),
        "answer": "已记录售后风险偏好。",
    }

    memory = store.update_from_state(correction_state)
    selected = store.select_for_question(
        {
            "question": "最近售后风险怎么看",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"})]),
        },
        budget_chars=2400,
    )
    assert selected["relevantCorrections"]
    assert selected["coreMemory"]["coreCorrectionIds"]
    assert selected["memoryInjectionTrace"]["coreMemoryCount"] >= 1
    assert "退款率" in selected["relevantCorrections"][0]["correctionText"]
    assert memory["conflicts"]
    assert any(float(item.get("confidence") or 1) <= 0.35 for item in memory["events"] if item.get("memoryType") != "correction")


def test_metric_definition_dispute_does_not_override_standard_metric_memory(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    dispute_state = {
        "question": "退款率不是这么算的，应该用退款单数除以下单订单数。",
        "requested_merchant_id": "seller_100",
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="退款率不是这么算的，应该用退款单数除以下单订单数。",
                    intent_type="VALID",
                    answer_mode="DERIVED",
                    category=QuestionCategory.REFUND,
                    metric_name="退款率",
                    metric_resolution={"metricKey": "refund_rate"},
                    days=7,
                )
            ],
        ),
        "answer": "当前标准退款率仍以语义层口径为准，已记录为口径争议待确认。",
    }

    memory = store.update_from_state(dispute_state)
    event = memory["events"][-1]
    assert event["memoryType"] == "metric_dispute"
    assert event["confidence"] < 0.8
    assert not memory["facts"]
    assert not memory["conflicts"]
    assert not memory["preferences"]

    selected = store.select_for_question(
        {
            "question": "退款率口径是什么",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"})]),
        },
        budget_chars=2400,
    )
    assert not selected["relevantCorrections"]
    assert selected["relevantMetricDisputes"]
    assert "不覆盖语义层" in selected["relevantMetricDisputes"][0]["governanceInstruction"]


def test_pending_memory_is_candidate_only_and_not_required_constraint(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["events"] = [
        {
            "eventId": "mem_pending_refund_rate",
            "memoryType": "correction",
            "question": "以后售后风险按退款率看",
            "correctionText": "以后售后风险按退款率看",
            "topics": ["REFUND"],
            "metrics": ["refund_rate"],
            "confidence": 0.95,
            "status": "pending",
            "createdAt": "2026-07-01T00:00:00",
        }
    ]
    store.save("seller_100", memory)

    selected = store.select_for_question(
        {
            "question": "最近售后风险怎么看",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"})]),
        },
        budget_chars=2400,
    )

    assert not selected["relevantCorrections"]
    assert selected["candidateMemories"]
    assert selected["memoryInjectionTrace"]["candidateIds"] == ["mem_pending_refund_rate"]
    constraints = build_memory_constraints(selected)
    assert not any(item.get("type") == "metric_correction" for item in constraints)
    assert not any(item.get("enforcement") == "required" for item in constraints)


def test_memory_ingestion_creates_candidate_knowledge_suggestion_and_past_case(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 3200})
    store = StructuredMemoryStore(settings)
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天退款率是多少",
                intent_type="VALID",
                answer_mode="DERIVED",
                category=QuestionCategory.REFUND,
                metric_name="退款率",
                metric_resolution={"metricKey": "refund_rate", "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate"},
                preferred_table="dwm_trade_refund_detail_di",
                days=7,
            )
        ]
    )
    state = {
        "question": "不对，以后退款率口径要重点确认。",
        "requested_merchant_id": "seller_100",
        "plan": plan,
        "answer": "已记录为口径待确认。",
        "chat_bi_completed": True,
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="refund_rate",
                    success=True,
                    query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"refund_rate": 0.02}]),
                )
            ]
        ),
        "recall_bundle": RecallBundle(
            items=[
                RecallItem(
                    doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate",
                    source_type="SEMANTIC_METRIC",
                    metadata={"semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate"},
                )
            ]
        ),
    }

    memory = store.update_from_state(state)
    assert memory["knowledgeSuggestions"]
    assert memory["knowledgeSuggestions"][-1]["status"] == "candidate"
    assert any(item["memoryType"] == "past_case" for item in memory["events"])

    selected = store.select_for_question(
        {
            "question": "最近7天退款率是多少",
            "requested_merchant_id": "seller_100",
            "plan": plan,
        },
        budget_chars=3200,
    )
    assert selected["relevantPastCases"]
    assert selected["relevantPastCases"][0]["casePayload"]["semanticRefIds"]


def test_memory_ingestion_creates_procedure_from_repair_trace(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 3200})
    store = StructuredMemoryStore(settings)
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天退款率是多少",
                intent_type="VALID",
                answer_mode="DERIVED",
                category=QuestionCategory.REFUND,
                metric_name="退款率",
                metric_resolution={"metricKey": "refund_rate"},
                days=7,
            )
        ]
    )
    state = {
        "question": "最近7天退款率是多少",
        "requested_merchant_id": "seller_100",
        "plan": plan,
        "answer": "退款率为2%。",
        "chat_bi_completed": True,
        "sql_repair_reviewed": True,
        "planner_repair_requests": [{"suggestedAction": "repair_graph", "reason": "missing relationship"}],
        "action_history": [{"action": "retrieve_knowledge", "status": "success"}],
        "agent_run_result": AgentRunResult(
            sql_repairs=[SqlRepairAttempt(task_id="refund_rate", original_sql="select 1", repaired_sql="select 2", reason="fix alias")]
        ),
    }

    memory = store.update_from_state(state)

    assert any(item["memoryType"] == "procedure" for item in memory["events"])
    procedure = next(item for item in memory["events"] if item["memoryType"] == "procedure")
    assert procedure["source"] == "planner_repair"
    assert procedure["status"] == "approved"
    assert procedure["casePayload"]["repairActions"]


def test_memory_knowledge_governance_reviews_publishes_and_indexes_suggestion(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file"})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["knowledgeSuggestions"] = [
        {
            "suggestionId": "ks_refund_rate",
            "suggestionType": "metric",
            "status": "candidate",
            "topic": "电商退货",
            "metricName": "refund_rate",
            "sourceTable": "dwm_trade_refund_detail_di",
            "createdAt": "2026-07-01T00:00:00",
        }
    ]
    store.save("seller_100", memory)
    service = MemoryKnowledgeGovernanceService(
        settings,
        memory_store=store,
        topic_assets=FakeSuggestionTopicAssets(),
        governance_service=FakeSuggestionGovernanceService(),
    )

    reviewed = service.review_suggestion(
        "seller_100",
        "ks_refund_rate",
        KnowledgeSuggestionReviewRequest(approved=True, reviewer="tester", review_note="looks good", action="review"),
    )
    assert reviewed["status"] == "reviewed"

    approved = service.review_suggestion(
        "seller_100",
        "ks_refund_rate",
        KnowledgeSuggestionReviewRequest(approved=True, reviewer="tester", review_note="approve", action="approve"),
    )
    assert approved["status"] == "approved"
    assert approved["promotedMemoryFact"]["memoryType"] == "business_fact"
    assert approved["promotedMemoryFact"]["memoryTier"] == "core"
    assert "refund_rate" in approved["promotedMemoryFact"]["metrics"]
    approved_memory = store.load("seller_100")
    assert approved_memory["facts"]
    constraints = build_memory_constraints({"coreMemory": approved_memory["coreMemoryProfile"]})
    assert any(item["enforcement"] == "required" and "refund_rate" in item["targetMetrics"] for item in constraints)

    published = service.publish_suggestion("seller_100", "ks_refund_rate", reviewer="tester", review_note="publish")
    assert published["status"] == "PUBLISHED"
    assert published["suggestion"]["status"] == "published"
    assert published["suggestion"]["publishedRefId"] == "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate"

    indexed = service.mark_suggestion_indexed("seller_100", "ks_refund_rate")
    assert indexed["status"] == "INDEXED"
    assert indexed["suggestion"]["status"] == "indexed"
    assert indexed["suggestion"]["indexedAt"]


def test_memory_knowledge_governance_request_publish_and_run_jobs(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file"})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["knowledgeSuggestions"] = [
        {
            "suggestionId": "ks_refund_rate",
            "suggestionType": "metric",
            "status": "approved",
            "topic": "电商退货",
            "metricName": "refund_rate",
            "sourceTable": "dwm_trade_refund_detail_di",
            "createdAt": "2026-07-01T00:00:00",
        }
    ]
    store.save("seller_100", memory)
    service = MemoryKnowledgeGovernanceService(
        settings,
        memory_store=store,
        topic_assets=FakeSuggestionTopicAssets(),
        governance_service=FakeSuggestionGovernanceService(),
    )

    requested = service.request_publish_suggestion("seller_100", "ks_refund_rate", requested_by="ops_reviewer", review_note="ready")
    assert requested["status"] == "PUBLISH_REQUESTED"
    assert requested["suggestion"]["status"] == "publish_requested"
    assert requested["suggestion"]["publishRequestedAt"]
    assert requested["suggestion"]["publishRequestedBy"] == "ops_reviewer"

    jobs = service.run_publish_jobs("seller_100", reviewer="release_bot", auto_index=True)
    assert jobs["queuedCount"] == 1
    assert jobs["processedCount"] == 1
    assert jobs["results"][0]["status"] == "PUBLISHED"
    assert jobs["results"][0]["indexed"]["status"] == "INDEXED"

    saved = store.load("seller_100")
    suggestion = saved["knowledgeSuggestions"][0]
    assert suggestion["status"] == "indexed"
    assert suggestion["reviewer"] == "release_bot"
    assert suggestion["publishRequestedBy"] == "ops_reviewer"
    assert suggestion["publishedRefId"] == "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_rate"
    assert suggestion["indexedAt"]


def test_context_manifest_records_memory_semantic_refs_and_observability(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("最近7天订单量是多少？", "seller_100", None, None, "thread_ctx_manifest", "run_ctx_manifest")
    state["memory_injection_trace"] = {"selectedIds": ["mem_order_cnt"], "candidateIds": ["mem_pending"]}
    state["recall_bundle"] = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:电商交易:dwm_trade_order_detail_di:metric:order_detail_cnt",
                source_type="SEMANTIC_METRIC",
                metadata={"semanticRefId": "semantic:电商交易:dwm_trade_order_detail_di:metric:order_detail_cnt"},
            )
        ]
    )
    state["plan"] = QueryPlan(
        intents=[
            QuestionIntent(
                plan_task_id="order_cnt",
                category=QuestionCategory.TRADE,
                preferred_table="dwm_trade_order_detail_di",
                metric_resolution={
                    "metricKey": "order_detail_cnt",
                    "semanticRefId": "semantic:电商交易:dwm_trade_order_detail_di:metric:order_detail_cnt",
                },
            )
        ]
    )

    package = workflow.prepare_scoped_context_package(
        state,
        "plan_query_graph",
        "PlannerAgent",
        allowed_tables=["dwm_trade_order_detail_di"],
        allowed_metrics=["order_detail_cnt"],
    )
    manifest = state["active_context_manifest"]
    summary = observability_summary(state)

    assert manifest["contextPackageId"] == package.package_id
    assert manifest["memoryIds"] == ["mem_order_cnt", "mem_pending"]
    assert "semantic:电商交易:dwm_trade_order_detail_di:metric:order_detail_cnt" in manifest["semanticRefIds"]
    assert summary["contextHash"] == manifest["contextHash"]


def test_answer_safe_memory_injection_removes_past_cases_from_answer_context():
    safe = answer_safe_memory_injection(
        {
            "relevantPastCases": [{"id": "case_1"}],
            "relevantProcedures": [{"id": "proc_1"}],
            "candidateMemories": [{"id": "mem_pending"}],
            "relevantPreferences": [{"id": "pref_1"}],
            "memoryInjectionTrace": {"selectedIds": ["case_1"], "candidates": [{"memoryId": "case_1"}]},
        }
    )

    assert "relevantPastCases" not in safe
    assert "relevantProcedures" not in safe
    assert "candidateMemories" not in safe
    assert safe["relevantPreferences"]
    assert "candidates" not in safe["memoryInjectionTrace"]


def test_golden_questions_and_evaluation_observability_shape():
    assert len(GOLDEN_QUESTIONS) >= 5
    assert any(case["id"] == "scm_inbound_7d" for case in GOLDEN_QUESTIONS)

    record = evaluation_observability_record(
        {
            "harness": {
                "observability": {
                    "selectedMemoryIds": ["mem_1"],
                    "semanticRefIds": ["semantic:metric"],
                    "contextHash": "ctxhash",
                    "validationGaps": [],
                    "evidenceGaps": [{"code": "MISSING"}],
                    "repairCount": 1,
                },
                "knowledgeRetrieval": {"sourceRefs": ["semantic:metric"], "rounds": [{"backend": "es"}]},
            }
        }
    )

    assert record["selectedMemoryIds"] == ["mem_1"]
    assert record["evidenceGapCount"] == 1
    assert record["repairCount"] == 1


def test_golden_case_loader_reads_jsonl_catalog():
    cases = GoldenCaseLoader(get_settings()).load()

    assert len(cases) >= 50
    assert any(case["id"] == "refund_rate_top_goods_ticket_30d" for case in cases)
    assert all(case.get("question") for case in cases)


def test_golden_evaluation_service_scores_layers_and_governance_items(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    service = GoldenEvaluationService(settings)
    request = GoldenEvaluationRequest(
        merchant_id="seller_100",
        case_ids=["refund_rate_top_goods_ticket_30d"],
        persist_report=True,
        persist_governance_items=True,
    )

    def fake_runner(question, merchant_id, context=None, listener=None, thread_id="", run_id="", message_history=None):
        return ChatResponse(
            id="answer_eval_1",
            answer="最近30天退款率最高的商品已结合工单量一起分析。",
            debug_trace={
                "harness": {
                    "observability": {"validationGaps": [], "evidenceGaps": [], "repairCount": 0},
                    "knowledgeRetrieval": {
                        "sourceRefs": ["SEMANTIC_METRIC:refund_rate", "SEMANTIC_METRIC:cs_ticket_cnt_1d"],
                        "rounds": [{"backend": "es", "sourceType": "SEMANTIC_METRIC", "topic": "REFUND CS"}],
                    },
                },
                "questionUnderstanding": {
                    "rankingObjective": {"metricRef": "refund_rate", "groupByColumn": "spu_id"},
                    "requestedMeasures": [{"metricRef": "cs_ticket_cnt_1d"}],
                    "requiredEvidenceIntents": ["refund_rate", "ticket_count"],
                    "timeWindowDays": 30,
                },
                "planIntents": [
                    {"taskId": "anchor_refund_rate", "preferredTable": "ads_merchant_profile", "metricName": "refund_rate", "days": 30},
                    {"taskId": "ticket_lookup", "preferredTable": "dwm_cs_ticket_detail_di", "metricName": "cs_ticket_cnt_1d", "days": 30},
                ],
                "dependencies": [{"anchorTaskId": "anchor_refund_rate", "dependentTaskId": "ticket_lookup", "joinKey": "spu_id"}],
                "taskResults": [
                    {"taskId": "anchor_refund_rate", "success": True, "queryBundle": {"failed": False, "tables": ["ads_merchant_profile"]}},
                    {"taskId": "ticket_lookup", "success": True, "queryBundle": {"failed": False, "tables": ["dwm_cs_ticket_detail_di"]}},
                ],
                "verifiedEvidence": {"passed": True, "coveredEvidence": ["refund_rate", "ticket_count"], "blockingGaps": []},
                "evidenceGaps": [],
                "planningAssetPack": {"metrics": ["refund_rate", "cs_ticket_cnt_1d"], "tables": ["ads_merchant_profile", "dwm_cs_ticket_detail_di"]},
            },
        )

    report = service.evaluate(request, fake_runner)

    assert report["accuracy"] == 1.0
    assert report["recallAccuracy"] == 1.0
    assert report["queryGraphAccuracy"] == 1.0
    assert report["sqlSuccessRate"] == 1.0
    assert report["evidenceCoverageRate"] == 1.0
    assert report["answerAccuracy"] == 1.0
    assert report["reportPath"]
    report_path = Path(report["reportPath"])
    assert report_path.exists()
    report_path.unlink(missing_ok=True)
    assert report["results"][0]["traceExcerpt"]["planIntents"]
    assert report["results"][0]["traceExcerpt"]["taskResults"]

    fail_request = GoldenEvaluationRequest(
        merchant_id="seller_100",
        cases_path=str(settings.resources_root / "evaluation" / "golden_cases.jsonl"),
        case_ids=["refund_rate_top_goods_ticket_30d"],
        persist_report=False,
        persist_governance_items=True,
    )

    def bad_runner(question, merchant_id, context=None, listener=None, thread_id="", run_id="", message_history=None):
        return ChatResponse(
            id="answer_eval_bad",
            answer="已查询。",
            debug_trace={
                "harness": {"knowledgeRetrieval": {"sourceRefs": [], "rounds": []}, "observability": {"evidenceGaps": [{"code": "MISSING"}]}},
                "questionUnderstanding": {"timeWindowDays": 30},
                "planIntents": [],
                "dependencies": [],
                "taskResults": [],
                "verifiedEvidence": {"passed": False, "blockingGaps": [{"code": "MISSING_TICKET_EVIDENCE"}]},
                "evidenceGaps": [{"code": "MISSING_TICKET_EVIDENCE", "severity": "blocking"}],
            },
        )

    failed = service.evaluate(fail_request, bad_runner)

    assert failed["accuracy"] == 0.0
    assert failed["governanceItems"]
    assert {item["failedLayer"] for item in failed["governanceItems"]} >= {"recall", "queryGraph", "sql", "evidence", "answer"}
    governance_path = Path(failed["governancePath"])
    assert governance_path.exists()
    governance_path.unlink(missing_ok=True)


def test_golden_evaluation_reads_trace_replay_style_plan_and_tasks(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    service = GoldenEvaluationService(settings)
    case = {
        "id": "trade_order_count_7d",
        "question": "最近7天订单量是多少？",
        "expectedIntent": "metric_query",
        "expectedTopics": ["TRADE"],
        "expectedSourceTypes": ["SEMANTIC_METRIC"],
        "expectedMetrics": ["order_detail_cnt"],
        "expectedTables": ["dwm_trade_order_detail_di"],
        "expectedTimeWindowDays": 7,
        "answerMustMention": ["订单"],
    }
    trace = {
        "answer": "最近7天订单量查询成功。",
        "harness": {
            "knowledgeRetrieval": {
                "sourceRefs": ["SEMANTIC_METRIC:order_detail_cnt"],
                "rounds": [{"topic": "TRADE", "sourceType": "SEMANTIC_METRIC"}],
            },
            "observability": {"evidenceGaps": [], "validationGaps": [], "repairCount": 0},
        },
        "plan": {
            "questionUnderstanding": {
                "timeWindowDays": 7,
                "rankingObjective": {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di"},
            },
            "intents": [
                {
                    "taskId": "anchor_order",
                    "planTaskId": "anchor_order",
                    "preferredTable": "dwm_trade_order_detail_di",
                    "metricName": "order_detail_cnt",
                    "days": 7,
                }
            ],
            "dependencies": [],
            "agentTrace": ["planner=semantic_metric_fallback"],
            "compilerTrace": ["ANCHOR_METRIC:order_detail_cnt:dwm_trade_order_detail_di"],
        },
        "tasks": [
            {
                "taskId": "anchor_order",
                "success": True,
                "queryBundle": {
                    "failed": False,
                    "tables": ["dwm_trade_order_detail_di"],
                    "rows": [{"order_detail_cnt": 12}],
                    "summary": "ok",
                },
            }
        ],
        "validation": {"valid": True, "gaps": []},
        "verifiedEvidence": {"passed": True, "coveredEvidence": ["order_detail_cnt"], "blockingGaps": []},
        "evidenceGaps": [],
        "planningAssetPack": {"metrics": ["order_detail_cnt"], "tables": ["dwm_trade_order_detail_di"]},
    }

    result = service.evaluate_case(
        case,
        "100",
        lambda question, merchant_id, **kwargs: ChatResponse(id="answer_trace_style", answer="最近7天订单量查询成功。", debug_trace=trace),
    )

    assert result["passed"]
    assert result["layers"]["queryGraph"]["passed"]
    assert result["layers"]["sql"]["passed"]
    assert result["traceExcerpt"]["planIntents"][0]["preferredTable"] == "dwm_trade_order_detail_di"
    assert result["traceExcerpt"]["taskResults"][0]["queryBundle"]["rowCount"] == 1


def test_memory_constraints_block_unapplied_metric_correction(tmp_path):
    memory_injection = {
        "relevantCorrections": [
            {
                "id": "mem_refund_rate",
                "memoryType": "correction",
                "correctionText": "不对，不是退款金额，是退款率，以后看售后风险要按退款率。",
                "topics": ["REFUND"],
                "metrics": ["refund_rate"],
                "confidence": 0.95,
                "hitReasons": ["correction_priority"],
            }
        ]
    }
    constraints = build_memory_constraints(memory_injection)
    assert constraints[0]["enforcement"] == "required"

    asset_pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="dwm_trade_refund_detail_di",
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "refund_related_pay_amt", "refund_rate"],
            )
        ],
        metrics=[
            PlanningAssetEntry(key="refund_related_pay_amt", table="dwm_trade_refund_detail_di"),
            PlanningAssetEntry(key="refund_rate", table="dwm_trade_refund_detail_di"),
        ],
    )
    plan = QueryPlan(
        question_understanding={"analysisIntent": "risk_ranking"},
        intents=[
            QuestionIntent(
                question="最近售后风险怎么看",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_risk",
                preferred_table="dwm_trade_refund_detail_di",
                metric_column="refund_related_pay_amt",
                metric_name="退款金额",
                metric_resolution={"metricKey": "refund_related_pay_amt"},
                group_by_column="seller_id",
            )
        ],
    )

    result = QueryGraphValidator().validate("最近售后风险怎么看", plan, asset_pack, constraints)
    assert any(gap.code == "MEMORY_CONSTRAINT_UNAPPLIED" and "refund_rate" in gap.evidence for gap in result.gaps)

    fixed_plan = plan.model_copy(
        update={
            "intents": [
                plan.intents[0].model_copy(
                    update={
                        "metric_column": "refund_rate",
                        "metric_name": "退款率",
                        "metric_resolution": {"metricKey": "refund_rate"},
                    }
                )
            ]
        }
    )
    fixed_result = QueryGraphValidator().validate("最近售后风险怎么看", fixed_plan, asset_pack, constraints)
    assert not any(gap.code == "MEMORY_CONSTRAINT_UNAPPLIED" for gap in fixed_result.gaps)


def test_evidence_verifier_reports_memory_constraint_and_metric_dispute():
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refund_risk",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[{"seller_id": "seller_100", "refund_related_pay_amt": 100}],
                ),
            )
        ],
        merged_query_bundle=QueryBundle(
            tables=["dwm_trade_refund_detail_di"],
            rows=[{"seller_id": "seller_100", "refund_related_pay_amt": 100}],
        ),
    )
    plan = QueryPlan(
        question_understanding={"analysisIntent": "risk_ranking"},
        intents=[
            QuestionIntent(
                question="最近售后风险怎么看",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_risk",
                metric_name="退款金额",
                metric_resolution={"metricKey": "refund_related_pay_amt"},
            )
        ],
    )
    constraints = build_memory_constraints(
        {
            "relevantCorrections": [
                {
                    "id": "mem_refund_rate",
                    "memoryType": "correction",
                    "correctionText": "以后看售后风险要按退款率。",
                    "topics": ["REFUND"],
                    "metrics": ["refund_rate"],
                    "confidence": 0.95,
                }
            ]
        }
    )

    verified = EvidenceVerifier().verify("最近售后风险怎么看", plan, run_result, constraints)
    assert any(gap.code == "MEMORY_CONSTRAINT_UNAPPLIED" and gap.severity == "blocking" for gap in verified.gaps)
    assert not verified.passed

    dispute_constraints = build_memory_constraints(
        {
            "relevantMetricDisputes": [
                {
                    "id": "mem_dispute",
                    "memoryType": "metric_dispute",
                    "question": "退款率不是这么算的，应该用退款单数除以下单订单数。",
                    "metrics": ["refund_rate"],
                    "confidence": 0.45,
                    "governanceInstruction": "口径争议信号，不覆盖语义层/指标中心标准定义。",
                }
            ]
        }
    )
    dispute_plan = plan.model_copy(
        update={
            "intents": [
                plan.intents[0].model_copy(
                    update={
                        "metric_name": "退款率",
                        "metric_resolution": {"metricKey": "refund_rate"},
                    }
                )
            ]
        }
    )
    disputed = EvidenceVerifier().verify("退款率口径是什么", dispute_plan, run_result, dispute_constraints)
    assert disputed.passed
    assert any(gap.code == "MEMORY_METRIC_DISPUTE_REQUIRES_CLARIFICATION" and gap.severity == "warning" for gap in disputed.gaps)


def test_summarize_middleware_flushes_runtime_checkpoint_before_compaction(tmp_path):
    outputs = tmp_path / "threads" / "thread_checkpoint" / "outputs"
    outputs.mkdir(parents=True)
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_window_tokens": 1000})
    state = {
        "question": "最近售后风险怎么看",
        "run_id": "run_checkpoint",
        "thread_id": "thread_checkpoint",
        "thread_data": ThreadData(thread_id="thread_checkpoint", run_id="run_checkpoint", outputs_path=str(outputs)),
        "context_budget_reports": [
            {
                "stage": "policy_round_1",
                "estimatedTokens": 1200,
                "overBudget": True,
            }
        ],
        "context_snapshots": [],
        "workspace_manifest": WorkspaceManifest(),
        "plan": QueryPlan(question_understanding={"analysisIntent": "risk_ranking"}),
        "agent_run_result": AgentRunResult(evidence_gaps=[EvidenceGap(code="MEMORY_CONSTRAINT_UNAPPLIED", evidence="refund_rate")]),
        "memory_constraints": [
            {
                "id": "mem_refund_rate",
                "type": "metric_correction",
                "enforcement": "required",
                "targetMetrics": ["refund_rate"],
                "instruction": "售后风险按退款率看",
            }
        ],
        "memory_constraint_trace": {"constraintCount": 1, "requiredCount": 1},
        "runtime_injection": {"currentDate": "2026-07-05"},
        "thread_context": {},
        "action_history": [],
        "run_steps": [],
        "middleware_events": [],
    }

    SummarizeMiddleware(settings).before_policy(state)

    assert state["summary_context"]
    assert state["runtime_checkpoints"]
    checkpoint_path = Path(state["runtime_checkpoints"][0]["path"])
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["memoryConstraints"][0]["targetMetrics"] == ["refund_rate"]
    assert checkpoint["evidenceGaps"][0]["code"] == "MEMORY_CONSTRAINT_UNAPPLIED"
    assert state["middleware_events"][-1].code == "CONTEXT_SUMMARIZED"


def test_memory_vector_candidates_use_rrf_fusion_without_trusting_unknown_ids():
    candidates = [
        MemoryRetrievalCandidate(
            memory_id="mem_refund_amount",
            memory_type="query_event",
            score=5.0,
            reasons=["metric_overlap:refund_related_pay_amt"],
            payload={"topics": ["REFUND"], "metrics": ["refund_related_pay_amt"]},
        ),
        MemoryRetrievalCandidate(
            memory_id="mem_refund_rate",
            memory_type="correction",
            score=2.0,
            reasons=["correction_priority"],
            payload={"topics": ["REFUND"], "metrics": ["refund_rate"]},
        ),
        MemoryRetrievalCandidate(
            memory_id="mem_ticket",
            memory_type="business_fact",
            score=4.5,
            reasons=["topic_overlap:REFUND"],
            payload={"topics": ["REFUND"], "metrics": ["ticket_cnt"]},
        ),
    ]

    fused = boost_vector_candidates(candidates, ["mem_refund_rate", "unknown_es_id"])

    assert fused[0].memory_id == "mem_refund_rate"
    assert "vector_rrf_match" in fused[0].reasons
    assert "unknown_es_id" not in [item.memory_id for item in fused]


def test_memory_retrieval_prefers_related_topic_and_metrics(tmp_path):
    settings = get_settings().model_copy(
        update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 2400, "memory_backend": "file"}
    )
    store = StructuredMemoryStore(settings)
    store.update_from_state(
        {
            "question": "最近7天退款率是多少",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近7天退款率是多少",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                        days=7,
                    )
                ]
            ),
            "answer": "退款率为2%。",
        }
    )
    store.update_from_state(
        {
            "question": "最近10天商品审核拒绝明细",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近10天商品审核拒绝明细",
                        category=QuestionCategory.GOODS,
                        metric_resolution={"metricKey": "goods_audit_reject_cnt"},
                        days=10,
                    )
                ]
            ),
            "answer": "商品审核拒绝明细已返回。",
        }
    )

    selected = store.select_for_question(
        {
            "question": "最近30天售后退款风险",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近30天售后退款风险",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                        days=30,
                    )
                ]
            ),
        },
        budget_chars=2400,
    )
    serialized = json.dumps(selected["relevantEvents"] + selected["relevantPreferences"], ensure_ascii=False)
    assert "refund_rate" in serialized
    assert selected["memoryInjectionTrace"]["candidateCount"] >= 2
    assert selected["memoryInjectionTrace"]["selectedIds"]


def test_memory_budget_trimming_preserves_core_preferences(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 1200})
    store = StructuredMemoryStore(settings)
    store.update_from_state(
        {
            "question": "最近7天退款率是多少",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近7天退款率是多少",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                        days=7,
                    )
                ]
            ),
            "answer": "退款率为2%。",
        }
    )
    for index in range(8):
        store.update_from_state(
            {
                "question": "第%s次看退款相关趋势和异常" % index,
                "requested_merchant_id": "seller_100",
                "plan": QueryPlan(
                    intents=[
                        QuestionIntent(
                            question="第%s次看退款相关趋势和异常" % index,
                            category=QuestionCategory.REFUND,
                            metric_resolution={"metricKey": "refund_rate"},
                            days=30,
                        )
                    ]
                ),
                "answer": "补充事件%s" % index,
            }
        )

    selected = store.select_for_question(
        {
            "question": "最近退款率怎么看",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"}, days=7)]),
        },
        budget_chars=1200,
    )

    assert selected["coreMemory"]["corePreferenceIds"]
    assert any(item["memoryTier"] == "core" for item in selected["relevantPreferences"])
    assert selected["memoryInjectionTrace"]["coreMemoryCount"] >= 1


def test_memory_retrieval_filters_expired_and_role_restricted_items(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["events"] = [
        {
            "eventId": "mem_expired_refund_rate",
            "memoryType": "correction",
            "question": "以后售后风险按退款率看",
            "correctionText": "以后售后风险按退款率看",
            "topics": ["REFUND"],
            "metrics": ["refund_rate"],
            "confidence": 0.95,
            "status": "approved",
            "retentionDays": 1,
            "createdAt": "2026-01-01T00:00:00",
        }
    ]
    memory["facts"] = [
        {
            "factId": "fact_admin_only",
            "memoryType": "business_fact",
            "content": "手机号字段只能管理员看",
            "topics": ["REFUND"],
            "metrics": ["refund_rate"],
            "confidence": 0.9,
            "status": "approved",
            "visibility": "restricted",
            "allowedRoles": ["merchant_admin"],
            "createdAt": "2026-07-01T00:00:00",
        }
    ]
    store.save("seller_100", memory)

    analyst_selected = store.select_for_question(
        {
            "question": "最近售后退款率怎么看",
            "requested_merchant_id": "seller_100",
            "access_role": "merchant_analyst",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近售后退款率怎么看",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                    )
                ]
            ),
        },
        budget_chars=2400,
    )
    assert not analyst_selected["relevantFacts"]
    assert analyst_selected["memoryInjectionTrace"]["filteredReasons"]["expired"] >= 1
    assert analyst_selected["memoryInjectionTrace"]["filteredReasons"]["role_filtered"] >= 1

    admin_selected = store.select_for_question(
        {
            "question": "最近售后退款率怎么看",
            "requested_merchant_id": "seller_100",
            "access_role": "merchant_admin",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近售后退款率怎么看",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                    )
                ]
            ),
        },
        budget_chars=2400,
    )
    assert admin_selected["relevantFacts"]
    assert admin_selected["relevantFacts"][0]["id"] == "fact_admin_only"


def test_memory_management_service_patches_deletes_and_cleans_expired(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["events"] = [
        {
            "eventId": "mem_refund_correction",
            "memoryType": "correction",
            "question": "以后售后风险按退款率看",
            "correctionText": "以后售后风险按退款率看",
            "topics": ["REFUND"],
            "metrics": ["refund_rate"],
            "confidence": 0.95,
            "status": "approved",
            "createdAt": "2026-07-01T00:00:00",
        },
        {
            "eventId": "mem_expired_query",
            "memoryType": "query_event",
            "question": "很久以前查过商品审核",
            "topics": ["GOODS"],
            "metrics": ["goods_audit_reject_cnt"],
            "confidence": 0.7,
            "retentionDays": 1,
            "createdAt": "2000-01-01T00:00:00",
        },
    ]
    store.save("seller_100", memory)
    service = MemoryManagementService(settings, store)

    patched = service.patch_item("seller_100", "mem_refund_correction", {"status": "disabled"})
    assert patched["success"]
    assert patched["item"]["status"] == "disabled"

    selected = store.select_for_question(
        {
            "question": "最近售后退款率怎么看",
            "requested_merchant_id": "seller_100",
            "memory_eval_context": {"topics": ["REFUND"], "metrics": ["refund_rate"]},
        },
        budget_chars=2400,
    )
    assert "mem_refund_correction" not in selected["memoryInjectionTrace"]["selectedIds"]
    assert selected["memoryInjectionTrace"]["filteredReasons"]["inactive"] >= 1

    cleanup = service.cleanup_expired("seller_100")
    assert cleanup["cleanedCount"] == 1
    assert cleanup["cleaned"][0]["memoryId"] == "mem_expired_query"
    cleaned_memory = store.load("seller_100")
    expired = next(item for item in cleaned_memory["events"] if item["eventId"] == "mem_expired_query")
    assert expired["status"] == "deleted"

    deleted = service.delete_item("seller_100", "mem_refund_correction", hard_delete=True)
    assert deleted["status"] == "HARD_DELETED"
    assert not any(item["eventId"] == "mem_refund_correction" for item in store.load("seller_100")["events"])


def test_memory_recall_evaluation_reports_expected_hits_and_false_positives(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file", "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_100")
    memory["events"] = [
        {
            "eventId": "mem_refund_rate_case",
            "memoryType": "correction",
            "question": "售后风险优先看退款率",
            "correctionText": "售后风险优先看退款率",
            "topics": ["REFUND"],
            "metrics": ["refund_rate"],
            "confidence": 0.94,
            "status": "approved",
            "createdAt": "2026-07-01T00:00:00",
        },
        {
            "eventId": "mem_goods_audit_case",
            "memoryType": "query_event",
            "question": "商品审核拒绝明细",
            "topics": ["GOODS"],
            "metrics": ["goods_audit_reject_cnt"],
            "confidence": 0.7,
            "status": "approved",
            "createdAt": "2026-07-01T00:00:00",
        },
    ]
    store.save("seller_100", memory)

    result = MemoryManagementService(settings, store).evaluate_recall(
        "seller_100",
        [
            {
                "caseId": "refund_risk",
                "question": "最近售后退款率风险怎么看",
                "topics": ["REFUND"],
                "metrics": ["refund_rate"],
                "expectedMemoryIds": ["mem_refund_rate_case"],
                "unexpectedMemoryIds": ["mem_goods_audit_case"],
            }
        ],
        budget_chars=2400,
    )

    assert result["passed"]
    assert result["hitRate"] == 1.0
    assert result["falsePositiveCount"] == 0
    assert result["results"][0]["hitMemoryIds"] == ["mem_refund_rate_case"]


def test_feedback_service_updates_memory_store(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "context_memory_budget_chars": 2400})
    store = StructuredMemoryStore(settings)
    pending_store = PendingAnswerStore()
    pending_store.put(
        PendingAnswer(
            id="ans_mem_1",
            question="最近7天GMV是多少",
            answer="最近7天GMV为100。",
            merchant_id="seller_100",
            merchant_name="测试商家",
            category_name="TRADE",
            doris_tables="ads_merchant_profile",
            suggested_questions="[]",
        )
    )

    class FakeAnswerRepository:
        def insert_answer(self, pending, adopted=False, liked=False, disliked=False):
            return True

        def update_feedback(self, answer_id, adopted, liked, disliked):
            return None

    service = FeedbackService(FakeAnswerRepository(), pending_store, store)
    assert service.apply_feedback("ans_mem_1", adopted=True, liked=True, disliked=False) is True
    memory = store.load("seller_100")
    assert memory["events"]
    assert "adopted" in memory["events"][-1]["feedbackSignal"]
    assert memory["recentFocus"]["topMetrics"]


class FakeEnterpriseMemoryRepository:
    def __init__(self, initial=None, fail=False, memory_items_by_id=None):
        self.items = dict(initial or {})
        self.memory_items_by_id = dict(memory_items_by_id or {})
        self.fail = fail
        self.load_count = 0
        self.load_items_count = 0
        self.save_count = 0
        self.hit_deltas = {}

    def load_memory(self, merchant_id):
        self.load_count += 1
        if self.fail:
            raise RuntimeError("mysql down")
        return self.items.get(merchant_id) or {
            "merchantId": merchant_id,
            "events": [],
            "preferences": [],
            "facts": [],
            "conflicts": [],
            "recentFocus": {},
        }

    def load_memory_items(self, merchant_id, memory_ids):
        self.load_items_count += 1
        if self.fail:
            raise RuntimeError("mysql down")
        result = {
            "merchantId": merchant_id,
            "events": [],
            "preferences": [],
            "facts": [],
            "conflicts": [],
            "recentFocus": {},
        }
        memory = self.items.get(merchant_id) or {}
        for group in ["events", "preferences", "facts"]:
            for item in memory.get(group) or []:
                memory_id = item.get("eventId") or item.get("preferenceId") or item.get("factId")
                if memory_id in memory_ids:
                    result[group].append(item)
        for memory_id in memory_ids:
            item = self.memory_items_by_id.get(memory_id)
            if not isinstance(item, dict):
                continue
            if item.get("eventId"):
                result["events"].append(item)
            elif item.get("preferenceId"):
                result["preferences"].append(item)
            elif item.get("factId"):
                result["facts"].append(item)
        return result

    def save_memory(self, merchant_id, payload):
        self.save_count += 1
        if self.fail:
            raise RuntimeError("mysql down")
        self.items[merchant_id] = payload
        return payload

    def apply_hit_deltas(self, deltas):
        self.hit_deltas.update(deltas)
        return len(deltas)


class FakeEnterpriseMemoryCache:
    def __init__(self, cached=None):
        self.cached = cached
        self.values = {}
        self.hit_deltas = {}
        self.get_count = 0
        self.set_count = 0
        self.invalidated = False

    def get_json(self, key):
        self.get_count += 1
        if self.cached is not None:
            return self.cached
        return self.values.get(key)

    def set_json(self, key, value):
        self.set_count += 1
        self.values[key] = value

    def invalidate_merchant(self, merchant_id):
        self.invalidated = True

    def increment_hit_delta(self, memory_id, merchant_id=""):
        self.hit_deltas[memory_id] = self.hit_deltas.get(memory_id, 0) + 1

    def drain_hit_deltas(self):
        return {memory_id: {"memoryId": memory_id, "hitCount": count} for memory_id, count in self.hit_deltas.items()}

    def backend_name(self):
        return "fake"


class FakeMemoryVectorIndex:
    def __init__(self, ids=None, enabled=False):
        self.ids = list(ids or [])
        self._enabled = enabled
        self.search_count = 0
        self.sync_count = 0

    def enabled(self):
        return self._enabled

    def search(self, merchant_id, query_text):
        self.search_count += 1
        return self.ids

    def sync_memory(self, memory):
        self.sync_count += 1
        return {"success": True, "upserted": 1}


class FakeEsResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http %s" % self.status_code)


class FakeMemoryEsApi:
    def __init__(self):
        self.index_exists = False
        self.docs = {}
        self.mapping = None

    def head(self, url, **kwargs):
        return FakeEsResponse(200 if self.index_exists else 404, {})

    def put(self, url, json=None, **kwargs):
        self.index_exists = True
        self.mapping = json
        return FakeEsResponse(200, {"acknowledged": True})

    def post(self, url, json=None, data=None, **kwargs):
        if url.endswith("/_bulk"):
            return self._bulk(data)
        if url.endswith("/_search"):
            return self._search(json or {})
        raise AssertionError("unexpected url %s" % url)

    def _bulk(self, data):
        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
        index = 0
        while index < len(lines):
            action = json.loads(lines[index])
            if "index" in action:
                meta = action["index"]
                doc = json.loads(lines[index + 1])
                self.docs[str(meta["_id"])] = doc
                index += 2
                continue
            if "update" in action:
                meta = action["update"]
                patch = json.loads(lines[index + 1]).get("doc") or {}
                current = dict(self.docs.get(str(meta["_id"])) or {})
                current.update(patch)
                self.docs[str(meta["_id"])] = current
                index += 2
                continue
            if "delete" in action:
                meta = action["delete"]
                self.docs.pop(str(meta["_id"]), None)
                index += 1
                continue
            raise AssertionError("unexpected bulk action %s" % action)
        return FakeEsResponse(200, {"errors": False})

    def _search(self, body):
        size = int(body.get("size") or 10)
        hits = []
        for doc_id, source in self.docs.items():
            if self._matches_query(source, body.get("query") or {}):
                hits.append({"_id": doc_id, "_source": dict(source)})
        sort_specs = body.get("sort") or []
        for spec in reversed(sort_specs):
            if not isinstance(spec, dict):
                continue
            field, options = next(iter(spec.items()))
            reverse = str((options or {}).get("order") or "asc") == "desc"
            hits.sort(key=lambda hit: str((hit.get("_source") or {}).get(field) or ""), reverse=reverse)
        return FakeEsResponse(200, {"hits": {"hits": hits[:size]}})

    def _matches_query(self, source, query):
        if not query:
            return True
        if "bool" in query:
            bool_query = query["bool"] or {}
            filters = list(bool_query.get("filter") or [])
            musts = list(bool_query.get("must") or [])
            return all(self._matches_clause(source, clause) for clause in filters + musts)
        return self._matches_clause(source, query)

    def _matches_clause(self, source, clause):
        if not clause:
            return True
        if "term" in clause:
            field, value = next(iter((clause.get("term") or {}).items()))
            return str(source.get(field) or "") == str(value)
        if "terms" in clause:
            field, values = next(iter((clause.get("terms") or {}).items()))
            return str(source.get(field) or "") in {str(item) for item in values or []}
        if "bool" in clause:
            return self._matches_query(source, clause)
        return True


class FakeSuggestionTopicAssets:
    def __init__(self):
        self.publish_calls = []
        self.patch_calls = []

    def stage_knowledge_suggestion_patch(self, topic, table_name, suggestion):
        self.patch_calls.append((topic, table_name, dict(suggestion)))
        return {
            "success": True,
            "status": "PATCH_STAGED",
            "topic": topic,
            "tableName": table_name,
            "suggestionId": suggestion.get("suggestionId"),
            "changes": [{"operation": "upsert", "sourceSuggestionId": suggestion.get("suggestionId")}],
        }

    def verify_published_suggestion(self, topic, table_name, suggestion_id):
        return {
            "success": True,
            "status": "VERIFIED",
            "topic": topic,
            "tableName": table_name,
            "suggestionId": suggestion_id,
        }

    def publish(self, topic, table_name, approved, reviewer, review_note):
        self.publish_calls.append((topic, table_name, approved, reviewer, review_note))
        return {
            "success": True,
            "status": "PUBLISHED",
            "topic": topic,
            "tableName": table_name,
            "publishMode": "scoped_incremental",
            "publishScope": {"topic": topic, "table": table_name},
        }


class FakeSuggestionGovernanceService:
    def __init__(self, publishable=True):
        self.publishable = publishable

    def preflight_publish(self, topic, table):
        return {
            "success": True,
            "publishable": self.publishable,
            "status": "PREFLIGHT_PASSED" if self.publishable else "PREFLIGHT_FAILED",
            "topic": topic,
            "tableName": table,
        }

    def after_publish(self, topic, table, reviewer="", review_note=""):
        return {
            "success": True,
            "status": "GOVERNED_PUBLISHED",
            "topic": topic,
            "tableName": table,
            "publishMode": "scoped_incremental",
            "publishScope": {"topic": topic, "table": table},
            "semanticCatalogVersion": {"semanticVersion": "semantic-1"},
        }


def test_memory_store_factory_switches_enterprise_backend(tmp_path):
    default_settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    file_settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "file"})
    es_settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "es"})
    hybrid_settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "memory_backend": "hybrid"})

    assert isinstance(create_memory_store(default_settings), EnterpriseMemoryStore)
    assert isinstance(create_memory_store(file_settings), StructuredMemoryStore)
    assert isinstance(create_memory_store(es_settings), EnterpriseMemoryStore)
    assert isinstance(create_memory_store(hybrid_settings), EnterpriseMemoryStore)


def test_memory_es_repository_round_trip_and_soft_delete(monkeypatch):
    api = FakeMemoryEsApi()
    monkeypatch.setattr("merchant_ai.services.memory.requests.head", api.head)
    monkeypatch.setattr("merchant_ai.services.memory.requests.put", api.put)
    monkeypatch.setattr("merchant_ai.services.memory.requests.post", api.post)
    settings = get_settings().model_copy(
        update={
            "memory_backend": "es",
            "es_enabled": True,
            "es_base_url": "http://fake-es:9200",
            "memory_es_index": "merchant_memory_test",
            "memory_vector_enabled": False,
        }
    )
    repo = MemoryEsRepository(settings)
    payload = {
        "merchantId": "seller_100",
        "recentFocus": {"summary": "常看退款率"},
        "events": [
            {
                "eventId": "mem_refund_rate",
                "memoryType": "correction",
                "question": "以后售后风险按退款率看",
                "correctionText": "以后售后风险按退款率看",
                "topics": ["REFUND"],
                "metrics": ["refund_rate"],
                "confidence": 0.95,
                "status": "approved",
                "createdAt": "2026-07-01T00:00:00",
            }
        ],
        "preferences": [
            {
                "preferenceId": "pref_window_7",
                "memoryType": "preference",
                "key": "timeWindow:7",
                "value": "7天",
                "source": "manual",
                "approvedBy": "tester",
            }
        ],
        "facts": [{"factId": "fact_refund", "memoryType": "business_fact", "content": "退款率优先"}],
        "conflicts": [{"conflictId": "conflict_1", "winnerId": "mem_refund_rate", "reason": "newer correction"}],
        "knowledgeSuggestions": [{"suggestionId": "ks_refund_rate", "metricName": "refund_rate", "topic": "REFUND", "status": "candidate"}],
    }

    repo.save_memory("seller_100", payload)
    loaded = repo.load_memory("seller_100")
    assert loaded["events"][0]["eventId"] == "mem_refund_rate"
    assert loaded["events"][0]["memoryTier"] == "core"
    assert loaded["events"][0]["memoryClass"] == "correction"
    assert loaded["preferences"][0]["preferenceId"] == "pref_window_7"
    assert loaded["preferences"][0]["memoryTier"] == "core"
    assert loaded["facts"][0]["factId"] == "fact_refund"
    assert loaded["coreMemoryProfile"]["corePreferences"] or loaded["coreMemoryProfile"]["coreCorrections"]
    assert loaded["knowledgeSuggestions"][0]["suggestionId"] == "ks_refund_rate"
    assert loaded["recentFocus"]["summary"] == "常看退款率"

    payload["events"] = []
    payload["knowledgeSuggestions"] = []
    repo.save_memory("seller_100", payload)
    loaded_after_delete = repo.load_memory("seller_100")
    assert not loaded_after_delete["events"]
    assert not loaded_after_delete["knowledgeSuggestions"]


def test_memory_es_repository_async_vector_index_defers_embedding(monkeypatch):
    api = FakeMemoryEsApi()
    monkeypatch.setattr("merchant_ai.services.memory.requests.head", api.head)
    monkeypatch.setattr("merchant_ai.services.memory.requests.put", api.put)
    monkeypatch.setattr("merchant_ai.services.memory.requests.post", api.post)
    embedding_calls = []
    monkeypatch.setattr(
        "merchant_ai.services.memory.MemoryVectorIndex._embed_text",
        lambda self, text: embedding_calls.append(text) or [0.1, 0.2],
    )
    settings = get_settings().model_copy(
        update={
            "memory_backend": "es",
            "es_enabled": True,
            "es_base_url": "http://fake-es:9200",
            "memory_es_index": "merchant_memory_test",
            "memory_vector_enabled": True,
            "memory_index_async": True,
            "embedding_api_key": "test-key",
            "embedding_dims": 2,
        }
    )

    class CapturingMemoryEsRepository(MemoryEsRepository):
        def __init__(self, settings):
            super().__init__(settings)
            self.scheduled = None

        def _schedule_vector_index(self, merchant_id, memory):
            self.scheduled = (merchant_id, memory)

    repo = CapturingMemoryEsRepository(settings)
    repo.save_memory(
        "seller_100",
        {
            "merchantId": "seller_100",
            "events": [
                {
                    "eventId": "mem_refund_rate",
                    "memoryType": "query_event",
                    "question": "最近7天退款率是多少",
                    "topics": ["REFUND"],
                    "metrics": ["refund_rate"],
                    "confidence": 0.55,
                    "createdAt": "2026-07-01T00:00:00",
                }
            ],
        },
    )

    doc_id = "memory_item:seller_100:event:mem_refund_rate"
    assert doc_id in api.docs
    assert "content_vector" not in api.docs[doc_id]
    assert not embedding_calls
    assert repo.scheduled is not None

    merchant_id, memory = repo.scheduled
    result = repo.sync_vector_index(merchant_id, memory)

    assert result["updated"] == 1
    assert embedding_calls
    assert api.docs[doc_id]["content_vector"] == [0.1, 0.2]


def test_memory_es_repository_apply_hit_deltas_updates_item_fields(monkeypatch):
    api = FakeMemoryEsApi()
    monkeypatch.setattr("merchant_ai.services.memory.requests.head", api.head)
    monkeypatch.setattr("merchant_ai.services.memory.requests.put", api.put)
    monkeypatch.setattr("merchant_ai.services.memory.requests.post", api.post)
    settings = get_settings().model_copy(
        update={
            "memory_backend": "es",
            "es_enabled": True,
            "es_base_url": "http://fake-es:9200",
            "memory_es_index": "merchant_memory_test",
            "memory_vector_enabled": False,
        }
    )
    repo = MemoryEsRepository(settings)
    repo.save_memory(
        "seller_100",
        {
            "merchantId": "seller_100",
            "events": [
                {
                    "eventId": "mem_refund_rate",
                    "memoryType": "correction",
                    "question": "以后售后风险按退款率看",
                    "topics": ["REFUND"],
                    "metrics": ["refund_rate"],
                    "confidence": 0.95,
                    "status": "approved",
                    "createdAt": "2026-07-01T00:00:00",
                }
            ],
        },
    )

    assert repo.apply_hit_deltas({"mem_refund_rate": {"merchantId": "seller_100", "hitCount": 2, "decayScore": 0.88}}) == 1
    loaded = repo.load_memory("seller_100")
    assert loaded["events"][0]["hitCount"] == 2
    assert float(loaded["events"][0]["decayScore"]) == 0.88


def test_memory_query_hash_uses_structured_slots_not_raw_question():
    base_context = {
        "question": "最近7天退款率是多少",
        "topics": {"退款售后"},
        "metrics": {"refund_rate"},
        "timeWindows": {7},
        "analysisIntent": "metric_query",
    }
    same_structure_context = {
        "question": "帮我看一下近一周售后退款率",
        "topics": {"退款售后"},
        "metrics": {"refund_rate"},
        "timeWindows": {7},
        "analysisIntent": "metric_query",
    }
    different_object_context = {
        **same_structure_context,
        "objectRefs": {"spuId": ["spu_2"]},
    }

    assert memory_query_hash("seller_100", base_context) == memory_query_hash("seller_100", same_structure_context)
    assert memory_query_hash("seller_100", base_context) != memory_query_hash("seller_100", different_object_context)


def test_enterprise_memory_store_writes_governed_events_preferences_facts_and_conflicts(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "es",
            "memory_redis_enabled": False,
            "memory_vector_enabled": False,
        }
    )
    repo = FakeEnterpriseMemoryRepository()
    store = EnterpriseMemoryStore(settings, repository=repo, hot_cache=FakeEnterpriseMemoryCache(), vector_index=FakeMemoryVectorIndex())
    store.update_from_state(
        {
            "question": "最近7天退款金额是多少",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="最近7天退款金额是多少",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_related_pay_amt"},
                        days=7,
                    )
                ]
            ),
            "answer": "最近7天退款金额为100。",
        }
    )
    memory = store.update_from_state(
        {
            "question": "不对，不是退款金额，是退款率，以后看售后风险要按退款率。",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        question="不对，不是退款金额，是退款率。",
                        category=QuestionCategory.REFUND,
                        metric_resolution={"metricKey": "refund_rate"},
                        days=7,
                    )
                ]
            ),
            "answer": "已记录售后风险偏好。",
        }
    )

    assert repo.save_count >= 2
    assert memory["events"][-1]["memoryType"] == "correction"
    assert memory["preferences"]
    assert memory["facts"]
    assert memory["conflicts"]
    assert memory["facts"][-1]["source"] == "correction"


def test_enterprise_memory_cache_hit_skips_repository_and_vector(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "es",
            "memory_redis_enabled": True,
            "memory_vector_enabled": True,
        }
    )
    cached = {
        "merchantId": "seller_100",
        "relevantEvents": [{"id": "mem_cached", "memoryType": "query_event"}],
        "relevantPreferences": [],
        "relevantFacts": [],
        "relevantCorrections": [],
        "relevantMetricDisputes": [],
        "memoryInjectionTrace": {"selectedIds": ["mem_cached"]},
    }
    repo = FakeEnterpriseMemoryRepository()
    cache = FakeEnterpriseMemoryCache(cached=cached)
    vector = FakeMemoryVectorIndex(ids=["mem_should_not_query"], enabled=True)
    store = EnterpriseMemoryStore(settings, repository=repo, hot_cache=cache, vector_index=vector)

    selected = store.select_for_question({"question": "最近7天GMV", "requested_merchant_id": "seller_100"}, budget_chars=1200)

    assert selected["memoryInjectionTrace"]["cacheHit"] is True
    assert repo.load_count == 0
    assert vector.search_count == 0
    assert cache.hit_deltas["mem_cached"] == 1


def test_enterprise_memory_vector_ids_load_authoritative_items_and_ignore_unknown_ids(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "es",
            "memory_redis_enabled": False,
            "memory_vector_enabled": True,
        }
    )
    repo = FakeEnterpriseMemoryRepository(
        initial={
            "seller_100": {
                "merchantId": "seller_100",
                "events": [],
                "preferences": [],
                "facts": [],
                "conflicts": [],
                "recentFocus": {},
            }
        },
        memory_items_by_id={
            "mem_vector_refund_rate": {
                "eventId": "mem_vector_refund_rate",
                "memoryType": "correction",
                "question": "以后看售后风险要优先看退款率",
                "answerPreview": "已记录退款率偏好。",
                "correctionText": "售后风险优先看退款率",
                "topics": [QuestionCategory.REFUND],
                "metrics": ["refund_rate"],
                "confidence": 0.9,
                "source": "answer_run",
                "createdAt": "2026-01-01T00:00:00",
            }
        }
    )
    store = EnterpriseMemoryStore(
        settings,
        repository=repo,
        hot_cache=FakeEnterpriseMemoryCache(),
        vector_index=FakeMemoryVectorIndex(ids=["mem_vector_refund_rate", "unknown_es_id"], enabled=True),
    )

    selected = store.select_for_question(
        {
            "question": "售后风险怎么看",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.REFUND, metric_resolution={"metricKey": "refund_rate"})]),
        },
        budget_chars=2400,
    )

    serialized = json.dumps(selected, ensure_ascii=False)
    assert selected["memoryInjectionTrace"]["vectorCandidateCount"] == 2
    assert selected["memoryInjectionTrace"]["vectorLoadedCount"] == 1
    assert "mem_vector_refund_rate" in serialized
    assert "refund_rate" in serialized
    assert "unknown_es_id" not in serialized
    assert repo.load_count >= 1
    assert repo.load_items_count == 1


def test_enterprise_memory_hybrid_falls_back_to_json_when_es_unavailable(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "hybrid",
            "memory_redis_enabled": False,
            "memory_vector_enabled": False,
        }
    )
    store = EnterpriseMemoryStore(
        settings,
        repository=FakeEnterpriseMemoryRepository(fail=True),
        hot_cache=FakeEnterpriseMemoryCache(),
        vector_index=FakeMemoryVectorIndex(),
    )

    memory = store.update_from_state(
        {
            "question": "最近7天GMV是多少",
            "requested_merchant_id": "seller_100",
            "plan": QueryPlan(intents=[QuestionIntent(category=QuestionCategory.TRADE, metric_resolution={"metricKey": "gmv_amt"}, days=7)]),
            "answer": "最近7天GMV为100。",
        }
    )

    assert memory["storageBackend"] == "json_fallback"
    assert StructuredMemoryStore(settings).memory_path("seller_100").exists()


def test_context_assembler_offloads_large_payload(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "context_planner_budget_chars": 1000,
            "tool_result_offload_chars": 500,
        }
    )
    outputs = tmp_path / "threads" / "thread_asm" / "outputs"
    outputs.mkdir(parents=True)
    state = {
        "thread_data": ThreadData(thread_id="thread_asm", run_id="run_asm", outputs_path=str(outputs)),
        "context_assembly_reports": [],
    }
    payload = {
        "question": "最近7天退款明细",
        "plannerToolResults": [{"result": {"content": "x" * 3000}, "artifact": {"relativePath": "old.json"}}],
    }

    compacted = ContextAssembler(settings).assemble_payload(state, "planner_round", "PlannerAgent", payload, budget_chars=1000)

    assert compacted["plannerToolResults"]["offloaded"]
    assert state["context_assembly_reports"][-1].compacted
    artifact_path = outputs / "artifacts" / "context" / "planner_round_plannerToolResults.json"
    assert artifact_path.exists()


def test_context_allocator_preserves_critical_sections_under_tiny_budget(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "context_planner_budget_chars": 900,
            "tool_result_offload_chars": 300,
        }
    )
    outputs = tmp_path / "threads" / "thread_alloc" / "outputs"
    outputs.mkdir(parents=True)
    state = {
        "thread_data": ThreadData(thread_id="thread_alloc", run_id="run_alloc", outputs_path=str(outputs)),
        "context_assembly_reports": [],
    }
    payload = {
        "question": "最近30天退款率最高的前5个商品，同时看工单量。",
        "status": "planning",
        "memoryConstraints": [{"memoryType": "correction", "summary": "退款率按退款单量/下单量", "enforcement": "required"}],
        "evidenceGaps": [{"code": "MISSING_TICKET_EVIDENCE", "taskId": "ticket_lookup"}],
        "runtimeInjection": {"merchant": {"merchantId": "seller_100"}, "threadContext": {"reusableEntitySets": [{"valuesPreview": ["spu_1"]}]}},
        "plannerToolResults": [{"result": {"content": "x" * 5000}}],
        "dataRows": [{"spu_id": "spu_%d" % idx, "refund_rate": idx / 100} for idx in range(100)],
        "trace": [{"round": idx, "observation": "y" * 300} for idx in range(30)],
    }

    compacted = ContextAssembler(settings).assemble_payload(state, "planner_round", "PlannerAgent", payload, budget_chars=900)

    assert compacted["question"]
    assert compacted["memoryConstraints"]
    assert compacted["evidenceGaps"]
    assert compacted["_contextOverBudget"]["artifact"]
    assert compacted["_contextAllocation"]["fullPayloadArtifact"]
    assert "plannerToolResults" in compacted["_contextAllocation"]["trimmedSections"]
    assert state["context_assembly_reports"][-1].compacted
    assert (outputs / "artifacts" / "context" / "planner_round_PlannerAgent_payload.json").exists()


def test_tool_call_recovery_patches_dangling_tool_result():
    state = {
        "tool_call_results": [ToolCallExecutionResult(id="tool_1", name="semantic_read", status="running")],
        "tool_call_ledger": [],
        "tool_call_recovery_events": [],
        "middleware_events": [],
    }

    ToolCallRecoveryMiddleware().before_policy(state)

    assert state["tool_call_results"][0].status == "failed"
    assert state["tool_call_results"][0].error_type == "MISSING_TOOL_RESULT"
    assert state["tool_call_ledger"][0].tool_call_id == "tool_1"
    assert state["tool_call_recovery_events"][0].action == "patch_missing_terminal_result"


def test_tool_call_recovery_inserts_synthetic_result_for_missing_tool_call_result():
    state = {
        "tool_call_requests": [ToolCallRequest(id="tool_missing", name="semantic_read", args={"refId": "semantic:x"})],
        "tool_call_results": [],
        "tool_call_ledger": [],
        "tool_call_recovery_events": [],
        "middleware_events": [],
    }

    ToolCallRecoveryMiddleware().before_policy(state)

    assert state["tool_call_results"][0].id == "tool_missing"
    assert state["tool_call_results"][0].status == "failed"
    assert state["tool_call_results"][0].error_type == "MISSING_TOOL_RESULT"
    assert state["tool_call_ledger"][0].tool_call_id == "tool_missing"
    assert state["tool_call_recovery_events"][0].action == "patch_missing_tool_result"
    assert state["middleware_events"][0].code == "MISSING_TOOL_RESULT_PATCHED"


def test_tool_output_budget_middleware_offloads_large_tool_result(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "middleware_tool_output_budget_chars": 1000,
        }
    )
    state = {
        "thread_data": ThreadData(thread_id="thread_tool_budget", run_id="run_tool_budget", outputs_path=str(tmp_path / "outputs")),
        "tool_call_results": [
            ToolCallExecutionResult(
                id="tool_big",
                name="artifact_read",
                status="success",
                result={"text": "x" * 2000},
            )
        ],
        "middleware_events": [],
    }

    ToolOutputBudgetMiddleware(settings).before_policy(state)

    result = state["tool_call_results"][0]
    assert result.result["_offloaded"]
    assert result.result["truncated"]
    assert Path(result.result["artifactRef"]["path"]).exists()
    assert state["tool_output_budget_reports"][0]["toolCallId"] == "tool_big"
    assert state["middleware_events"][-1].code == "TOOL_OUTPUT_BUDGET_APPLIED"


def test_token_usage_middleware_records_estimated_usage():
    settings = get_settings()
    state = {
        "react_round": 2,
        "question": "最近7天GMV",
        "runtime_context": "runtime" * 20,
        "memory_context": "memory",
        "summary_context": "",
        "tool_call_results": [ToolCallExecutionResult(id="tool_1", name="semantic_read", status="success")],
        "agent_run_result": AgentRunResult(task_results=[AgentTaskResult(task_id="node_1", success=True)]),
        "middleware_events": [],
    }

    TokenUsageMiddleware(settings).before_policy(state)

    assert state["token_usage_reports"][0]["stage"] == "policy_round_2"
    assert state["token_usage_reports"][0]["estimatedInputTokens"] > 0
    assert state["middleware_events"][-1].code == "TOKEN_USAGE_ESTIMATED"


def test_safety_finish_reason_middleware_records_llm_timeout_and_forced_stop():
    state = {
        "planner_provider_error": "timeout: provider call exceeded 8 seconds",
        "forced_tool_loop_stop_message": "[FORCED STOP] repeated tool calls",
        "middleware_loop_blocked": True,
        "middleware_events": [],
    }

    SafetyFinishReasonMiddleware().before_policy(state)

    reasons = {item["finishReason"] for item in state["safety_finish_reasons"]}
    assert "timeout" in reasons
    assert "forced_stop" in reasons or "tool_loop_hard_stop" in reasons
    assert state["middleware_events"][-1].code == "SAFETY_FINISH_REASON_RECORDED"


def test_run_budget_middleware_blocks_when_action_budget_exhausted():
    settings = get_settings().model_copy(update={"run_budget_max_actions": 2})
    state = {
        "run_started_at_ms": int(time.time() * 1000),
        "action_history": [AgentActionTrace(action="route_topic"), AgentActionTrace(action="retrieve_knowledge")],
        "trace_spans": [],
        "tool_runtime_events": [],
        "token_usage_reports": [],
        "middleware_events": [],
        "safety_finish_reasons": [],
        "chat_bi_completed": False,
        "answer": "",
    }

    RunBudgetMiddleware(settings).before_policy(state)

    assert state["run_budget_exhausted"]
    assert state["chat_bi_completed"]
    assert state["run_budget_report"]["breaches"] == ["actions"]
    assert state["middleware_events"][-1].code == "RUN_BUDGET_EXHAUSTED"


def test_run_budget_middleware_counts_llm_doris_tool_and_tokens():
    settings = get_settings().model_copy(
        update={
            "run_budget_max_llm_calls": 1,
            "run_budget_max_doris_queries": 1,
            "run_budget_max_tool_calls": 1,
            "run_budget_max_estimated_tokens": 100,
        }
    )
    state = {
        "run_started_at_ms": int(time.time() * 1000),
        "action_history": [],
        "trace_spans": [
            TraceSpan(kind="llm", name="planner"),
            TraceSpan(kind="sql", name="execute_sql"),
        ],
        "tool_runtime_events": [{"eventType": "tool.started"}],
        "token_usage_reports": [{"estimatedInputTokens": 120}],
        "middleware_events": [],
        "safety_finish_reasons": [],
        "chat_bi_completed": False,
        "answer": "",
    }

    RunBudgetMiddleware(settings).before_policy(state)

    assert state["run_budget_exhausted"]
    assert set(state["run_budget_report"]["breaches"]) == {"llm_calls", "doris_queries", "tool_calls", "estimated_tokens"}


def test_run_budget_middleware_uses_peak_context_tokens_not_cumulative_rounds():
    settings = get_settings().model_copy(update={"run_budget_max_estimated_tokens": 60000})
    state = {
        "run_started_at_ms": int(time.time() * 1000),
        "action_history": [],
        "trace_spans": [],
        "tool_runtime_events": [],
        "token_usage_reports": [
            {"estimatedInputTokens": 32763},
            {"estimatedInputTokens": 37060},
        ],
        "middleware_events": [],
        "safety_finish_reasons": [],
        "chat_bi_completed": False,
        "answer": "",
    }

    RunBudgetMiddleware(settings).before_policy(state)

    assert not state.get("run_budget_exhausted")
    assert state["run_budget_report"]["usage"]["estimatedTokens"] == 37060
    assert state["run_budget_report"]["usage"]["peakEstimatedTokens"] == 37060
    assert state["run_budget_report"]["breaches"] == []


def test_clarification_middleware_intercepts_ask_human_as_virtual_tool_call():
    state = {
        "qa_id": "qa_clarify",
        "question": "删除最近30天订单数据",
        "human_clarification_required": True,
        "human_clarification_question": "写操作需要人工确认。",
        "human_clarification_stage": "BUSINESS_SCOPE",
        "human_clarification_type": "write_operation",
        "human_clarification_options": ["取消操作", "提交工单"],
        "tool_call_results": [],
        "tool_call_ledger": [],
        "middleware_events": [],
        "tool_runtime_events": [],
    }
    decision = AgentDecision(selected_action="ask_human", selected_node="human_in_loop", available_actions=["ask_human"])

    ClarificationMiddleware().before_action(state, decision)

    assert state["_clarification_tool_intercepted"]
    assert state["clarification_command"]["goto"] == "END"
    assert state["clarification_tool_message"]["toolName"] == "ask_clarification"
    assert state["clarification_tool_message"]["question"] == "写操作需要人工确认。"
    assert state["tool_call_results"][0].name == "ask_clarification"
    assert state["tool_call_results"][0].tool_message["type"] == "clarification_request"
    assert state["tool_call_ledger"][0].tool_name == "ask_clarification"
    assert state["middleware_events"][-1].code == "CLARIFICATION_TOOL_INTERCEPTED"


def test_cancellation_middleware_marks_run_canceled(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    runs_dir = tmp_path / "run_events" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "run_cancel.json").write_text(json.dumps({"runId": "run_cancel", "status": "CANCELED"}), encoding="utf-8")
    state = {"run_id": "run_cancel", "middleware_events": [], "chat_bi_completed": False, "answer": ""}

    CancellationMiddleware(settings).before_policy(state)

    assert state["run_canceled"]
    assert state["chat_bi_completed"]
    assert "取消" in state["answer"]
    assert state["middleware_events"][-1].code == "RUN_CANCELED"


def test_loop_guard_blocks_repeated_action_pattern():
    settings = get_settings().model_copy(update={"middleware_loop_guard_threshold": 3})
    state = {
        "action_history": [
            AgentActionTrace(action="retrieve_knowledge"),
            AgentActionTrace(action="retrieve_knowledge"),
            AgentActionTrace(action="retrieve_knowledge"),
        ],
        "middleware_events": [],
    }

    LoopGuardMiddleware(settings).before_policy(state)

    assert state["middleware_loop_blocked"]
    assert state["query_graph_validated"]
    assert state["query_graph_validation_result"].gaps[0].code == "LOOP_DETECTED"


def test_loop_guard_warns_and_hard_stops_repeated_tool_calls():
    settings = get_settings().model_copy(
        update={
            "middleware_tool_repeat_warning_threshold": 3,
            "middleware_tool_repeat_hard_stop_threshold": 5,
            "middleware_tool_type_warning_threshold": 30,
            "middleware_tool_type_hard_stop_threshold": 50,
        }
    )
    warning_state = {
        "tool_call_requests": [
            ToolCallRequest(id="call_%d" % index, name="semantic_read", args={"refId": "metric.gmv"})
            for index in range(3)
        ],
        "middleware_events": [],
    }

    LoopGuardMiddleware(settings).before_policy(warning_state)

    assert not warning_state.get("middleware_loop_blocked")
    assert "semantic_read" in warning_state["pending_tool_loop_warnings"][0]
    assert warning_state["middleware_events"][-1].code == "TOOL_CALL_LOOP_WARNING"

    hard_stop_state = {
        "tool_call_requests": [
            ToolCallRequest(id="call_%d" % index, name="semantic_read", args={"refId": "metric.gmv"})
            for index in range(5)
        ],
        "middleware_events": [],
    }

    LoopGuardMiddleware(settings).before_policy(hard_stop_state)

    assert hard_stop_state["middleware_loop_blocked"]
    assert hard_stop_state["chat_bi_completed"]
    assert hard_stop_state["tool_call_requests"] == []
    assert "FORCED STOP" in hard_stop_state["forced_tool_loop_stop_message"]
    assert hard_stop_state["query_graph_validation_result"].gaps[0].code == "TOOL_CALL_LOOP_DETECTED"
    assert hard_stop_state["middleware_events"][-1].code == "TOOL_CALL_LOOP_HARD_STOP"


def test_loop_guard_tracks_tool_calls_in_thread_window_across_rounds():
    settings = get_settings().model_copy(
        update={
            "middleware_tool_repeat_warning_threshold": 3,
            "middleware_tool_repeat_hard_stop_threshold": 5,
            "middleware_tool_loop_window_size": 20,
        }
    )
    state = {"thread_id": "thread_loop", "middleware_events": [], "tool_loop_history": {}}

    for index in range(3):
        state["tool_call_requests"] = [ToolCallRequest(id="call_%d" % index, name="read_file", args={"path": "/tmp/a.md", "noise": index})]
        LoopGuardMiddleware(settings).before_policy(state)

    assert not state.get("middleware_loop_blocked")
    assert "read_file" in state["pending_tool_loop_warnings"][0]
    assert len(state["tool_loop_history"]["thread_loop"]) == 3


def test_loop_guard_does_not_recount_existing_planner_tool_calls_across_policy_rounds():
    settings = get_settings().model_copy(
        update={
            "middleware_tool_repeat_warning_threshold": 3,
            "middleware_tool_repeat_hard_stop_threshold": 5,
            "middleware_tool_loop_window_size": 20,
        }
    )
    state = {
        "thread_id": "thread_planner_tool",
        "plan": QueryPlan(
            planner_tool_calls=[
                {
                    "id": "planner_call_1",
                    "name": "emit_question_understanding",
                    "args": {"questionUnderstanding": {"timeWindowDays": 7}},
                }
            ]
        ),
        "middleware_events": [],
        "tool_loop_history": {},
    }

    chain = MiddlewareChain([LoopGuardMiddleware(settings)])
    for _index in range(5):
        state = chain.before_policy(state)

    assert not state.get("middleware_loop_blocked")
    assert state.get("pending_tool_loop_warnings", []) == []
    assert len(state["tool_loop_history"]["thread_planner_tool"]) == 1


def test_tool_loop_warning_is_injected_into_runtime_context():
    settings = get_settings()
    state = {
        "pending_tool_loop_warnings": ["工具 semantic_read 使用相同参数已重复 3 次，请停止重复调用。"],
        "middleware_events": [],
    }

    DynamicContextMiddleware(settings).before_policy(state)

    assert "semantic_read" in state["runtime_context"]
    assert state["pending_tool_loop_warnings"] == []


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


def test_tool_failure_registry_separates_merchant_thread_and_target():
    registry = ToolFailureRegistry(repeat_threshold=2, circuit_threshold=5)
    args_a = {"taskId": "anchor", "sql": "SELECT * FROM x", "merchantId": "seller_a", "threadId": "thread_a"}
    args_b = {"taskId": "anchor", "sql": "SELECT * FROM x", "merchantId": "seller_b", "threadId": "thread_a"}
    registry.record_failure("execute_sql", args_a, "TIMEOUT", "slow", service_name="doris", target="primary")
    registry.record_failure("execute_sql", args_a, "TIMEOUT", "slow", service_name="doris", target="primary")

    blocked_a = registry.should_block("execute_sql", args_a, service_name="doris", target="primary")
    blocked_b = registry.should_block("execute_sql", args_b, service_name="doris", target="primary")

    assert blocked_a is not None
    assert blocked_a.merchant_id == "seller_a"
    assert blocked_a.target == "primary"
    assert blocked_b is None


def test_tool_failure_registry_releases_tool_after_cooldown():
    registry = ToolFailureRegistry(repeat_threshold=5, circuit_threshold=1, cooldown_seconds=1)
    args = {"taskId": "anchor", "sql": "SELECT * FROM x"}
    registry.record_failure("execute_sql", args, "TIMEOUT", "slow")
    blocked = registry.should_block("execute_sql", {"taskId": "other"})
    assert blocked is not None
    assert blocked.open
    blocked.open_until_ms = 1
    assert registry.should_block("execute_sql", {"taskId": "other"}) is None


def test_tool_failure_registry_half_open_probe_closes_on_success():
    registry = ToolFailureRegistry(repeat_threshold=5, circuit_threshold=1, cooldown_seconds=1)
    args = {"taskId": "anchor", "sql": "SELECT * FROM x"}
    registry.record_failure("execute_sql", args, "TIMEOUT", "slow", service_name="doris", target="primary")
    circuit = registry.should_block("execute_sql", args, service_name="doris", target="primary")
    assert circuit is not None
    circuit.open_until_ms = 1

    assert registry.should_block("execute_sql", args, service_name="doris", target="primary") is None
    half_open = registry.trace()["circuits"][0]
    assert half_open["state"] == "half_open"
    registry.record_success("execute_sql", args, service_name="doris", target="primary")
    closed = registry.trace()["circuits"][0]
    assert closed["state"] == "closed"
    assert closed["open"] is False


def test_entity_transfer_does_not_mix_order_id_into_sub_order_id_dependency():
    dep = PlanDependency(
        anchor_task_id="order_detail",
        dependent_task_id="refund_lookup",
        join_key="seller_id+sub_order_id",
        anchor_column="seller_id+sub_order_id",
        dependent_column="seller_id+sub_order_id",
    )
    parent_result = AgentTaskResult(
        task_id="order_detail",
        success=True,
        query_bundle=QueryBundle(rows=[]),
        entity_set=EntitySet(
            task_id="order_detail",
            join_key="order_id",
            values=["order_id_100"],
            column_values={"order_id": ["order_id_100"], "sub_order_id": ["sub_order_id_100"]},
        ),
    )

    values = multi_entity_transfer_values(dep, [], parent_result, {"seller_id", "sub_order_id"}, 200)

    assert values == {"sub_order_id": ["sub_order_id_100"]}
    assert "order_id" not in values


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
    assert results[0].idempotency_key
    assert results[0].params_hash
    assert results[1].status == "failed"
    assert results[1].error_type == "INVALID_ARGUMENT"
    assert results[1].error_code == "INVALID_ARGUMENT"
    assert results[1].tool_message["toolCallId"] == "call_2"
    assert results[1].tool_message["toolName"] == "bad_tool"
    assert results[1].tool_message["errorCode"] == "INVALID_ARGUMENT"
    assert results[1].tool_message["details"]["toolCallId"] == "call_2"
    assert results[1].idempotency_key
    assert registry.trace()["failures"]


def test_tool_failure_feedback_keeps_runtime_envelope_for_prompt():
    result = ToolCallExecutionResult(
        id="tool_1",
        name="semantic_read",
        status="failed",
        error_type="TIMEOUT",
        error_message="semantic_read timed out after 5s",
        timeout_type="read_timeout",
        retryable=True,
        recommended_action="retry_or_use_semantic_grep",
        fallback_tools=["semantic_grep"],
        service_name="semantic",
        params_hash="params_1",
    )

    payload = serialize_tool_execution_result(result)

    assert payload["errorCode"] == "TIMEOUT"
    assert payload["retryable"] is True
    assert payload["recommendedAction"] == "retry_or_use_semantic_grep"
    assert payload["fallbackTools"] == ["semantic_grep"]
    assert payload["details"]["toolCallId"] == "tool_1"
    assert payload["details"]["serviceName"] == "semantic"
    assert result.tool_message["errorType"] == "TIMEOUT"
    assert result.tool_message["details"]["timeoutType"] == "read_timeout"


def test_tool_error_details_are_tool_specific_and_prompt_compacted():
    details = tool_error_details(
        "execute_sql",
        {
            "sql": "select * from dwd_order where seller_id = 1 " + "and order_id is not null " * 80,
            "queryId": "query_1",
            "failedStage": "execute",
        },
        "UNKNOWN_COLUMN",
        "unknown column: bad_col",
        call_id="sql_1",
    )

    assert details["toolKind"] == "doris"
    assert details["toolCallId"] == "sql_1"
    assert details["sqlHash"]
    assert len(details["sqlPreview"]) <= 500
    assert details["queryId"] == "query_1"

    compacted = compact_tool_failure_for_prompt(
        {
            "id": "sql_1",
            "name": "execute_sql",
            "status": "failed",
            "errorType": "UNKNOWN_COLUMN",
            "errorMessage": "unknown column: bad_col",
            "retryable": False,
            "recommendedAction": "repair_sql",
            "fallbackTools": ["answer_with_gap"],
            "details": details,
        }
    )

    assert compacted["errorCode"] == "UNKNOWN_COLUMN"
    assert compacted["recommendedAction"] == "repair_sql"
    assert len(compacted["details"]["sqlPreview"]) <= 500


def test_tool_runtime_service_execute_many_runs_parallel_and_records_batch_events():
    settings = get_settings().model_copy(update={"tool_max_concurrency": 2, "tool_rate_limit_enabled": False})
    runtime = ToolRuntimeService(settings)

    def slow(args):
        time.sleep(0.2)
        return {"value": args["value"]}

    started = time.monotonic()
    results = runtime.execute_many(
        [
            ToolCallRequest(id="call_1", name="semantic_read", args={"value": 1}),
            ToolCallRequest(id="call_2", name="artifact_read", args={"value": 2}),
        ],
        {"semantic_read": slow, "artifact_read": slow},
    )
    elapsed = time.monotonic() - started

    assert [item.status for item in results] == ["success", "success"]
    assert elapsed < 0.35
    event_types = [item["eventType"] for item in runtime.trace()["events"]]
    assert "tool.parallel.batch_started" in event_types
    assert "tool.parallel.batch_finished" in event_types


def test_execute_sql_runtime_policy_leaves_repair_to_node_worker():
    settings = get_settings()
    policy = ToolRuntimePolicyRegistry(settings).policy_for("execute_sql")
    assert policy.max_retries == 0
    assert "MEM_ALLOC_FAILED" in policy.non_retryable_errors
    assert not policy.retryable_errors


def test_tool_runtime_service_caches_successful_result():
    settings = get_settings().model_copy(update={"cache_enabled": True, "semantic_cache_ttl_seconds": 60})
    runtime = ToolRuntimeService(settings)
    calls = {"count": 0}

    def handler(args):
        calls["count"] += 1
        return {"value": args["value"], "count": calls["count"]}

    policy = ToolCachePolicy(enabled=True, namespace="semantic_test", ttl_seconds=60)
    first = runtime.execute("semantic_read", {"value": 7, "semanticVersion": "v1"}, handler, cache_policy=policy)
    second = runtime.execute("semantic_read", {"value": 7, "semanticVersion": "v1"}, handler, cache_policy=policy)

    assert first.status == "success"
    assert second.status == "success"
    assert second.cache_hit
    assert second.result["count"] == 1
    assert calls["count"] == 1
    metrics = runtime.trace()["metrics"]["tools"][0]
    assert metrics["cacheHits"] == 1


def test_tool_runtime_service_exposes_stable_idempotency_key():
    settings = get_settings().model_copy(update={"cache_enabled": False})
    runtime = ToolRuntimeService(settings)

    def handler(args):
        return {"value": args["value"]}

    first = runtime.execute("semantic_read", {"value": 7}, handler, call_id="tool_call_1")
    second = runtime.execute("semantic_read", {"value": 7}, handler, call_id="tool_call_1")
    changed = runtime.execute("semantic_read", {"value": 8}, handler, call_id="tool_call_1")

    assert first.status == second.status == changed.status == "success"
    assert first.idempotency_key == second.idempotency_key
    assert first.params_hash == second.params_hash
    assert first.idempotency_key != changed.idempotency_key
    assert first.runtime_events[0]["eventType"] == "tool.started"
    assert first.runtime_events[0]["payload"]["idempotencyKey"] == first.idempotency_key


def test_tool_runtime_service_classifies_tool_timeout_and_emits_heartbeat():
    settings = get_settings().model_copy(
        update={
            "tool_rate_limit_enabled": False,
            "tool_heartbeat_interval_seconds": 0.1,
            "doris_read_timeout_seconds": 1,
        }
    )
    runtime = ToolRuntimeService(settings)

    def slow_handler(args):
        time.sleep(1.4)
        return {"ok": True}

    result = runtime.execute("execute_sql", {"sql": "SELECT 1"}, slow_handler, target_kind="doris")

    assert result.status == "failed"
    assert result.error_type == "TIMEOUT"
    assert result.timeout_type == "tool_timeout"
    event_types = [item["eventType"] for item in result.runtime_events]
    assert "tool.started" in event_types
    assert "tool.heartbeat" in event_types


def test_tool_runtime_service_classifies_connect_and_read_timeouts():
    settings = get_settings().model_copy(update={"tool_rate_limit_enabled": False})
    runtime = ToolRuntimeService(settings)

    def connect_timeout(args):
        raise TimeoutError("connection timed out while connecting to Doris")

    def read_timeout(args):
        raise TimeoutError("read timed out waiting for Doris result")

    connect_result = runtime.execute("execute_sql", {"sql": "SELECT 1"}, connect_timeout, target_kind="doris")
    read_result = runtime.execute("execute_sql", {"sql": "SELECT 1"}, read_timeout, target_kind="doris")

    assert connect_result.error_type == "TIMEOUT"
    assert connect_result.timeout_type == "connect_timeout"
    assert read_result.error_type == "TIMEOUT"
    assert read_result.timeout_type == "read_timeout"


def test_redis_ttl_cache_falls_back_to_memory_when_unavailable(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "redis_enabled": True,
            "redis_cache_enabled": True,
            "redis_url": "redis://127.0.0.1:1/0",
            "redis_socket_timeout_seconds": 0.05,
            "cache_memory_max_entries": 8,
        }
    )
    cache = build_ttl_cache("unit_redis_fallback", settings, 60)

    cache.set("key", {"value": 1})
    assert cache.get("key") == {"value": 1}
    trace = cache.trace()
    assert trace["backend"] == "redis+memory_fallback"
    assert trace["fallback"]["hits"] == 1


def test_tool_runtime_redis_cache_store_falls_back_to_memory_when_unavailable(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "redis_enabled": True,
            "redis_cache_enabled": True,
            "redis_rate_limit_enabled": True,
            "redis_url": "redis://127.0.0.1:1/0",
            "redis_socket_timeout_seconds": 0.05,
            "semantic_cache_ttl_seconds": 60,
        }
    )
    runtime = ToolRuntimeService(settings)
    calls = {"count": 0}

    def handler(args):
        calls["count"] += 1
        return {"value": args["value"], "count": calls["count"]}

    policy = ToolCachePolicy(enabled=True, namespace="semantic_test", ttl_seconds=60)
    runtime.execute("semantic_read", {"value": 7, "semanticVersion": "v1"}, handler, cache_policy=policy)
    second = runtime.execute("semantic_read", {"value": 7, "semanticVersion": "v1"}, handler, cache_policy=policy)

    assert second.cache_hit
    assert calls["count"] == 1
    assert runtime.trace()["cache"]["backend"] == "redis+memory_fallback"


def test_tool_runtime_service_rate_limit_blocks_without_calling_handler():
    settings = get_settings().model_copy(update={"tool_rate_limit_enabled": True, "tool_default_qps": 1})
    runtime = ToolRuntimeService(settings)
    calls = {"count": 0}

    def handler(args):
        calls["count"] += 1
        return {"rows": [{"ok": True}]}

    first = runtime.execute("execute_sql", {"sql": "SELECT 1"}, handler, target_kind="doris")
    second = runtime.execute("execute_sql", {"sql": "SELECT 2"}, handler, target_kind="doris")

    assert first.status == "success"
    assert second.status == "blocked"
    assert second.error_type == "RATE_LIMITED"
    assert calls["count"] == 1


def test_tool_runtime_service_returns_structured_tool_message_and_recovery_action():
    settings = get_settings().model_copy(update={"tool_rate_limit_enabled": False})
    runtime = ToolRuntimeService(settings)

    def handler(args):
        raise RuntimeError("Unknown column 'refund_amt'")

    result = runtime.execute(
        "execute_sql",
        {"sql": "SELECT refund_amt FROM t", "merchantId": "seller_100", "threadId": "thread_1"},
        handler,
        target_kind="doris",
    )

    assert result.status == "failed"
    assert result.error_type == "UNKNOWN_COLUMN"
    assert result.service_name == "doris"
    assert result.circuit_key
    assert result.retryable is False
    assert result.recommended_action == "semantic_recall_or_graph_repair"
    assert "semantic_read" in result.fallback_tools
    assert result.tool_message["errorCode"] == "UNKNOWN_COLUMN"
    assert result.tool_message["recommendedAction"] == "semantic_recall_or_graph_repair"
    event_types = [item["eventType"] for item in runtime.trace()["events"]]
    assert "tool.failed" in event_types
    assert "tool.recovery.recommended" in event_types


def test_tool_runtime_service_circuit_half_open_events():
    settings = get_settings().model_copy(
        update={
            "tool_rate_limit_enabled": False,
            "tool_circuit_threshold": 1,
            "tool_circuit_cooldown_seconds": 1,
        }
    )
    runtime = ToolRuntimeService(settings)
    calls = {"count": 0}

    def fail_once(args):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("timeout")
        return {"ok": True}

    first = runtime.execute("semantic_read", {"refId": "metric.refund_rate"}, fail_once)
    assert first.status == "failed"
    circuit = runtime.failure_registry.trace()["circuits"][0]
    assert circuit["state"] == "open"
    runtime.failure_registry.circuits[circuit["circuitKey"]].open_until_ms = 1

    second = runtime.execute("semantic_read", {"refId": "metric.refund_rate"}, fail_once)
    assert second.status == "success"
    event_types = [item["eventType"] for item in runtime.trace()["events"]]
    assert "tool.circuit.half_open" in event_types
    assert "tool.circuit.closed" in event_types


def test_round_robin_load_balancer_cycles_targets():
    balancer = RoundRobinLoadBalancer(
        {
            "llm": [
                LoadBalancerTarget(name="a", endpoint="http://a"),
                LoadBalancerTarget(name="b", endpoint="http://b"),
            ]
        }
    )

    assert [balancer.select("llm").name for _ in range(4)] == ["a", "b", "a", "b"]


def test_context_package_has_stable_hash_for_same_refs(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = ContextManager(settings)
    state = {
        "run_id": "run_1",
        "thread_id": "thread_1",
        "question": "最近7天订单量是多少",
        "requested_merchant_id": "100",
        "planning_asset_pack": PlanningAssetPack(),
    }

    first = manager.package(state, "planner", "PlannerAgent")
    second = manager.package(state, "planner", "PlannerAgent")

    assert first.context_hash
    assert first.context_hash == second.context_hash
    assert first.context_delta.context_hash == first.context_hash


def test_planner_overrides_need_more_when_semantic_catalog_is_sufficient():
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量。"
    pack = profile_daily_pack()
    llm = FakeNeedMoreThenProfilePlannerLlm()
    plan, requests, _ = QueryGraphPlanner(llm).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert plan.intents
    assert llm.calls == 2
    assert any("planner.need_more_overridden_by_semantic_catalog" in item for item in plan.agent_trace)


def test_planner_falls_back_to_semantic_metric_when_llm_returns_empty_understanding():
    class EmptyUnderstandingLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {}

    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近7天订单量呢"
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE],
    )
    plan, requests, _ = QueryGraphPlanner(EmptyUnderstandingLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])
    assert not requests
    assert not plan.intents
    assert "planner.no_valid_llm_understanding" in plan.agent_trace


def test_semantic_metric_fallback_skips_multi_domain_detail_question():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近30天退款状态为处理中或异常的订单，关联看商品发布时间和订单金额。"
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )

    plan = compile_semantic_metric_fallback_graph(question, pack)

    assert not plan.intents
    assert any(item.startswith("planner.semantic_metric_fallback.skipped") for item in plan.agent_trace)


def test_planner_timeout_can_fallback_to_semantic_entity_id_graph():
    class TimeoutLlm:
        configured = True
        last_error = "timeout: provider call exceeded 20 seconds"
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {}

    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。"
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )

    plan, requests, reason = QueryGraphPlanner(TimeoutLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])

    assert not requests
    assert "PLANNER_LLM_TIMEOUT" in reason
    assert plan.intents
    assert plan.intents[0].preferred_table == "dwm_trade_order_detail_di"
    assert plan.intents[0].filter_column == "order_id"
    assert plan.intents[0].filter_value == "order_id_100"
    tables = {intent.preferred_table for intent in plan.intents}
    assert tables == {"dwm_trade_order_detail_di"}
    assert "planner=entity_id_semantic_fallback" in plan.agent_trace


def test_planner_uses_semantic_fast_path_for_topn_metric_question_without_llm():
    class ExplodingLlm:
        configured = True
        last_error = ""
        error_events = []

        def tool_chat(self, *args, **kwargs):
            raise AssertionError("planner LLM should not be called")

        def tool_json_chat(self, *args, **kwargs):
            raise AssertionError("planner LLM should not be called")

        def json_chat(self, *args, **kwargs):
            raise AssertionError("planner LLM should not be called")

    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间，帮我判断哪些是高风险新品。"
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )

    plan, requests, reason = QueryGraphPlanner(ExplodingLlm()).plan(question, [], "", recall_bundle_empty(), pack, [], [])

    assert reason == "SEMANTIC_FAST_PATH"
    assert not requests
    assert plan.intents
    assert plan.question_understanding["rankingObjective"]["metricRef"] == "refund_rate"
    assert plan.question_understanding["rankingObjective"]["limit"] == 10
    assert any(item.get("metricRef") == "order_detail_cnt" for item in plan.question_understanding["requestedMeasures"])
    assert any(item.get("metricRef") == "pay_amt" for item in plan.question_understanding["requestedMeasures"])
    assert any(intent.preferred_table == "dwm_goods_detail_df" for intent in plan.intents)
    assert any("planner.semantic_fast_path=topn_metric" in item for item in plan.agent_trace)


def test_asset_pack_defers_recalled_profile_table_until_metric_understanding_requests_it():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近30天GMV和退款金额走势是否正常？"
    keywords = KeywordExtractService().extract(question)
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        keywords,
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.REFUND],
    )

    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    pack = builder.compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND],
    )

    assert "ads_merchant_profile" not in pack.known_tables()

    traces = builder.expand_for_question_understanding(
        pack,
        {
            "rankingObjective": {
                "metricRef": "order_gmv_amt_1d",
                "ownerTable": "ads_merchant_profile",
                "sourcePhrase": "GMV",
            },
            "requestedMeasures": [
                {"metricRef": "refund_amt_1d", "ownerTable": "ads_merchant_profile", "sourcePhrase": "退款金额"},
            ],
        },
    )

    assert any("metric_request_table:order_gmv_amt_1d->ads_merchant_profile" in item for item in traces)
    assert "ads_merchant_profile" in pack.known_tables()
    assert any(metric.table == "ads_merchant_profile" and metric.key == "order_gmv_amt_1d" for metric in pack.metrics)


def test_asset_pack_expands_profile_from_recalled_metric_phrase_when_llm_metric_ref_is_wrong():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近7天优惠金额、优惠订单量和 GMV 分别是多少？"
    keywords = KeywordExtractService().extract(question)
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        keywords,
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.COUPON],
    )

    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    pack = builder.compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.COUPON],
    )

    assert "ads_merchant_profile" not in pack.known_tables()

    understanding = {
        "analysisGrain": "merchant",
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "GMV",
            "groupByColumn": "seller_id",
        },
        "requestedMeasures": [
            {"metricRef": "discount_amt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "优惠金额"},
        ],
        "timeWindowDays": 7,
    }
    traces = builder.expand_for_question_understanding(pack, understanding)
    assert any("metric_request_table:pay_amt->ads_merchant_profile" in item for item in traces)
    assert "ads_merchant_profile" in pack.known_tables()

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    assert plan.intents
    assert plan.intents[0].preferred_table == "ads_merchant_profile"
    assert plan.intents[0].metric_name == "order_gmv_amt_1d"
    assert plan.intents[0].metric_resolution.get("resolutionSource") == "semantic_recall_evidence"


def test_planner_timeout_can_fallback_to_multi_metric_trend_graph():
    class TimeoutLlm:
        configured = True
        last_error = "timeout: provider call exceeded 20 seconds"
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {}

    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近30天GMV和退款金额走势是否正常？"
    keywords = KeywordExtractService().extract(question)
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        keywords,
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.REFUND],
    )
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND],
    )

    plan, requests, reason = QueryGraphPlanner(TimeoutLlm()).plan(question, [], "", recall, pack, [], [])

    assert not requests
    assert "PLANNER_LLM_TIMEOUT" in reason
    assert not plan.intents
    assert any(item.startswith("PLANNER_LLM_TIMEOUT:") for item in plan.agent_trace)


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


def test_simple_metric_node_uses_structured_fast_path_without_llm_sql():
    settings = get_settings()
    llm = FakeLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "sub_order_id"],
            )
        ]
    )
    intent = QuestionIntent(
        question="最近7天订单量呢",
        intent_type="VALID",
        answer_mode=AnswerMode.METRIC,
        plan_task_id="anchor_order",
        task_role=TaskRole.ANCHOR,
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="llm_plan_bound_first",
        group_by_column="seller_id",
        metric_column="sub_order_id",
        metric_name="order_detail_cnt",
        metric_formula="COUNT(DISTINCT `sub_order_id`)",
        output_keys=["seller_id"],
        required_evidence=["seller_id", "sub_order_id"],
        days=7,
        limit=1,
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    decision = worker._last_sql_draft_decisions[intent.plan_task_id]
    assert llm.calls == 0
    assert decision.source == "structured_fast_path"
    assert "COUNT(DISTINCT `sub_order_id`) AS `order_detail_cnt`" in sql
    assert "`seller_id` = '100'" in sql
    assert "`pt` >=" in sql
    assert SqlValidationService().validate(sql, pack).valid


def test_topn_anchor_uses_structured_fast_path_without_llm_sql():
    settings = get_settings()
    llm = FakeLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "spu_id", "spu_name", "sub_order_id"],
            )
        ]
    )
    intent = QuestionIntent(
        question="最近7天下单最多的前3个SPU",
        intent_type="VALID",
        answer_mode=AnswerMode.TOPN,
        plan_task_id="anchor_order",
        task_role=TaskRole.ANCHOR,
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="llm_plan_bound_first",
        group_by_column="spu_id",
        metric_column="sub_order_id",
        metric_name="order_detail_cnt",
        metric_formula="COUNT(DISTINCT `sub_order_id`)",
        output_keys=["seller_id", "spu_id", "spu_name"],
        required_evidence=["seller_id", "spu_id", "sub_order_id"],
        days=7,
        limit=3,
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    decision = worker._last_sql_draft_decisions[intent.plan_task_id]
    assert llm.calls == 0
    assert decision.source == "structured_fast_path"
    assert "GROUP BY" in sql
    assert "COUNT(DISTINCT `sub_order_id`) AS `order_detail_cnt`" in sql


def test_node_execution_batch_records_failed_task_without_blocking_success(monkeypatch):
    settings = get_settings().model_copy(update={"max_concurrent_sub_agents": 2, "agent_node_timeout_seconds": 2})
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="GMV",
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_gmv",
                preferred_table="ads_merchant_profile",
            ),
            QuestionIntent(
                question="退款金额",
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                plan_task_id="metric_refund",
                preferred_table="ads_merchant_profile",
            ),
        ]
    )

    def fake_execute_node(intent, *_args, **_kwargs):
        if intent.plan_task_id == "metric_refund":
            return AgentTaskResult(
                task_id=intent.plan_task_id,
                success=False,
                query_bundle=QueryBundle(failed=True, error="simulated failure"),
            )
        return AgentTaskResult(
            task_id=intent.plan_task_id,
            success=True,
            query_bundle=QueryBundle(rows=[{"metric": 1}], tables=[intent.preferred_table]),
        )

    monkeypatch.setattr(worker, "execute_node", fake_execute_node)

    results, batch = worker._execute_ready_batch(
        ["metric_gmv", "metric_refund"],
        {intent.plan_task_id: intent for intent in plan.intents},
        {},
        plan,
        "100",
        PlanningAssetPack(),
        "最近7天 GMV 和退款金额分别是多少？",
        "",
    )

    assert results["metric_gmv"].success
    assert results["metric_refund"].query_bundle.failed
    assert batch.completed_task_ids == ["metric_gmv"]
    assert batch.failed_task_ids == ["metric_refund"]


def test_trend_objective_compiles_to_group_agg():
    class TrendUnderstandingLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, system_prompt, user_prompt, fallback=None):
            return {
                "status": "UNDERSTOOD",
                "questionUnderstanding": {
                    "analysisGrain": "day",
                    "analysisIntent": "trend_check",
                    "requiresExplanation": True,
                    "requiredEvidenceIntents": [
                        {
                            "semanticLabel": "trend_context",
                            "reason": "需要按天查看趋势",
                            "requiredLevel": "required",
                            "suggestedMetricRefs": ["order_detail_cnt"],
                            "suggestedDomains": ["trade"],
                        }
                    ],
                    "rankingObjective": {
                        "metricRef": "order_detail_cnt",
                        "sourcePhrase": "订单量走势",
                        "ownerTable": "dwm_trade_order_detail_di",
                        "objectiveType": "trend_anchor",
                        "groupByColumn": "pt",
                        "order": "desc",
                        "limit": 30,
                    },
                    "requestedMeasures": [],
                    "filters": [],
                    "timeWindowDays": 30,
                },
            }

    plan, requests, _ = QueryGraphPlanner(TrendUnderstandingLlm()).plan(
        "最近30天订单量走势是否正常？",
        [],
        "",
        recall_bundle_empty(),
        trade_refund_goods_pack(),
        [],
        [],
    )
    assert not requests
    assert plan.intents
    assert plan.intents[0].answer_mode == AnswerMode.GROUP_AGG
    assert plan.intents[0].group_by_column == "pt"


def test_formula_reconciliation_prunes_missing_or_branch_without_hardcoding():
    formula = "SUM(CASE WHEN spu_status_code = 1 OR spu_status_name = '上架' THEN 1 ELSE 0 END)"
    reconciled = reconcile_metric_formula_for_schema(
        formula,
        ["spu_status_code", "spu_status_name"],
        {"spu_status_name"},
        "goods_online_detail_cnt",
        "dwm_goods_detail_df",
    )
    assert reconciled.formula
    assert "spu_status_name" in reconciled.formula
    assert "spu_status_code" not in reconciled.formula
    assert reconciled.rewritten
    assert reconciled.missing_source_columns == ["spu_status_code"]
    assert compile_metric_formula(formula, {"spu_status_name"}) == reconciled.formula


def test_goods_online_metric_uses_live_schema_reconciled_formula_for_aggregate():
    understanding = {
        "analysisGrain": "product",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "goods_online_detail_cnt",
            "ownerTable": "dwm_goods_detail_df",
            "sourcePhrase": "发布成功商品数",
            "groupByColumn": "spu_name",
            "limit": 10,
        },
        "requestedMeasures": [],
        "filters": [],
        "timeWindowDays": 30,
    }
    pack = trade_refund_goods_pack(include_missing_goods_metric=True)
    plan = QuestionUnderstandingCompiler().compile("最近30天发布成功商品数最高的商品", understanding, pack)
    assert plan.intents
    intent = plan.intents[0]
    assert intent.metric_name == "goods_online_detail_cnt"
    assert intent.metric_column == "spu_status_name"
    assert "spu_status_code" not in intent.metric_formula
    assert "spu_status_name" in intent.metric_formula
    assert intent.metric_resolution["droppedSourceColumns"] == ["spu_status_code"]
    assert QueryGraphValidator().validate("最近30天发布成功商品数最高的商品", plan, pack).valid


def test_structured_sql_uses_reconciled_goods_formula():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = trade_refund_goods_pack(include_missing_goods_metric=True)
    intent = QuestionIntent(
        question="发布成功商品数",
        intent_type="VALID",
        answer_mode="GROUP_AGG",
        plan_task_id="goods_online",
        preferred_table="dwm_goods_detail_df",
        sql_strategy="structured_first",
        group_by_column="spu_name",
        metric_column="spu_status_name",
        metric_name="goods_online_detail_cnt",
        metric_formula="SUM(CASE WHEN spu_status_name = '上架' THEN 1 ELSE 0 END)",
        output_keys=["seller_id", "spu_name"],
        required_evidence=["spu_name", "spu_status_name"],
    )
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "spu_status_name" in sql
    assert "spu_status_code" not in sql
    assert SqlValidationService().validate(sql, pack).valid


def test_bind_node_sql_parameters_uses_backend_merchant_and_entities():
    pack = trade_refund_goods_pack()
    intent = QuestionIntent(
        question="最近30天退款金额最高商品",
        intent_type="VALID",
        answer_mode="TOPN",
        plan_task_id="refund_top",
        preferred_table="dwm_trade_refund_detail_di",
        filter_column="spu_name",
        filter_value="商品A,商品B",
        group_by_column="spu_name",
    )
    context = NodeExecutionContext(
        merchant_id="100",
        upstream_entity_sets=[
            EntitySet(join_key="spu_name", values=["商品A", "商品B"], column_values={"spu_name": ["商品A", "商品B"]})
        ],
    )
    sql = (
        "SELECT `spu_name`, SUM(`pay_amt`) AS `refund_amt` "
        "FROM `dwm_trade_refund_detail_di` "
        "WHERE `seller_id` = '999' AND `spu_name` IN ('商品A', '商品B') "
        "GROUP BY `spu_name`"
    )

    bound_sql, params, error = bind_node_sql_parameters(sql, intent, pack, context)

    assert error == ""
    assert "`seller_id` = %s" in bound_sql
    assert "`spu_name` IN (%s, %s)" in bound_sql
    assert params == ["100", "商品A", "商品B"]


def test_bind_node_sql_parameters_uses_ast_for_aliased_predicates():
    pack = trade_refund_goods_pack()
    intent = QuestionIntent(
        question="最近30天退款金额最高商品",
        intent_type="VALID",
        answer_mode="TOPN",
        plan_task_id="refund_top",
        preferred_table="dwm_trade_refund_detail_di",
        filter_column="spu_name",
        filter_value="商品A,商品B",
        group_by_column="spu_name",
    )
    context = NodeExecutionContext(merchant_id="100")
    sql = (
        "SELECT r.`spu_name`, SUM(r.`pay_amt`) AS `refund_amt` "
        "FROM `dwm_trade_refund_detail_di` r "
        "WHERE r.`seller_id` = '999' AND (r.`spu_name` IN ('旧商品') OR r.`refund_status_name` = '退款成功') "
        "GROUP BY r.`spu_name`"
    )

    bound_sql, params, error = bind_node_sql_parameters(sql, intent, pack, context)

    assert error == ""
    assert "`r`.`seller_id` = %s" in bound_sql
    assert "`r`.`spu_name` IN (%s, %s)" in bound_sql
    assert params == ["100", "商品A", "商品B"]


def test_merge_task_result_bundles_aligns_rows_by_entity_key():
    merged = merge_task_result_bundles(
        [
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[
                        {"spu_id": "spu_1", "refund_amt": 100},
                        {"spu_id": "spu_2", "refund_amt": 80},
                    ],
                ),
            ),
            AgentTaskResult(
                task_id="order_cnt",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[
                        {"spu_id": "spu_1", "order_cnt": 20},
                        {"spu_id": "spu_2", "order_cnt": 10},
                    ],
                ),
            ),
            AgentTaskResult(
                task_id="goods_publish",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[{"spu_id": "spu_1", "publish_time": "2026-06-01"}],
                ),
            ),
        ]
    )

    by_spu = {row["spu_id"]: row for row in merged.rows}
    assert merged.summary.startswith("按实体键 spu_id 合并")
    assert by_spu["spu_1"]["refund_amt"] == 100
    assert by_spu["spu_1"]["order_cnt"] == 20
    assert by_spu["spu_1"]["publish_time"] == "2026-06-01"
    assert by_spu["spu_2"]["refund_amt"] == 80
    assert by_spu["spu_2"]["order_cnt"] == 10


def test_merge_task_result_bundles_preserves_entity_time_grain():
    merged = merge_task_result_bundles(
        [
            AgentTaskResult(
                task_id="gmv",
                success=True,
                query_bundle=QueryBundle(
                    rows=[
                        {"spu_id": "spu_1", "pt": "2026-07-10", "gmv": 10},
                        {"spu_id": "spu_1", "pt": "2026-07-11", "gmv": 20},
                    ]
                ),
            ),
            AgentTaskResult(
                task_id="refund",
                success=True,
                query_bundle=QueryBundle(
                    rows=[
                        {"spu_id": "spu_1", "pt": "2026-07-10", "refund_amt": 1},
                        {"spu_id": "spu_1", "pt": "2026-07-11", "refund_amt": 2},
                    ]
                ),
            ),
        ]
    )

    assert len(merged.rows) == 2
    by_day = {(row["spu_id"], row["pt"]): row for row in merged.rows}
    assert by_day[("spu_1", "2026-07-10")]["gmv"] == 10
    assert by_day[("spu_1", "2026-07-10")]["refund_amt"] == 1
    assert by_day[("spu_1", "2026-07-11")]["gmv"] == 20
    assert by_day[("spu_1", "2026-07-11")]["refund_amt"] == 2


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
                sql_strategy="llm_first_debug",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天", execution_mode="subagent")
    assert result.task_results[0].success
    sql = result.task_results[0].query_bundle.sql
    assert "order_gmv_amt_1d" in sql
    assert "ads_merchant_profile" in sql
    assert result.sql_draft_decisions[0].source == "llm_plan_bound"
    assert result.node_plan_contracts[0].allowed_columns
    assert result.node_plan_critiques[0].valid


def test_node_worker_records_isolated_subagent_checkpoint(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    worker = NodeWorkerExecutor(GoodPlanBoundSqlLlm(), FakeDoris(), SqlValidationService(), settings)
    worker.with_artifact_root(str(tmp_path / "artifacts"))
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前5天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv",
                preferred_table="ads_merchant_profile",
                sql_strategy="llm_first_debug",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天", execution_mode="subagent")
    task = result.task_results[0]

    assert task.sub_agent_run_id.startswith("sub_anchor_gmv")
    assert Path(task.sub_agent_checkpoint_path).exists()
    assert "/subagents/" in task.query_bundle.offloaded_files[0]
    checkpoint = json.loads(Path(task.sub_agent_checkpoint_path).read_text(encoding="utf-8"))
    assert checkpoint["status"] == "success"
    assert checkpoint["contextPackage"]["taskId"] == "anchor_gmv"
    assert checkpoint["contextPackage"]["subagentEnabled"] is True
    assert checkpoint["contextPackage"]["runtimeMode"] == "independent_node_react"


def test_node_worker_can_read_artifact_before_sql_draft(tmp_path):
    class FileReadingSqlLlm:
        configured = True
        last_error = ""
        error_events = []

        def __init__(self):
            self.file_context_seen = False

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, timeout_seconds=None, tool_choice=None):
            if tool_choice == "draft_sql":
                return {"content": "", "toolCalls": [{"id": "draft", "name": "draft_sql", "args": self._draft_args(user_prompt)}]}
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "read_hint",
                        "name": "artifact_read",
                        "args": {"path": "planner/hint.txt", "maxChars": 1000, "reason": "need SQL hint"},
                    }
                ],
            }

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, timeout_seconds=None):
            return self._draft_args(user_prompt)

        def _draft_args(self, user_prompt):
            payload = json.loads(user_prompt)
            self.file_context_seen = "order_gmv_amt_1d" in str(payload.get("fileToolResults") or "")
            return {
                "sql": (
                    "SELECT `seller_id`, `pt`, SUM(`order_gmv_amt_1d`) AS `order_gmv_amt_1d` "
                    "FROM `ads_merchant_profile` WHERE `seller_id` = '100' "
                    "AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
                    "GROUP BY `seller_id`, `pt` ORDER BY `order_gmv_amt_1d` DESC LIMIT 5"
                ),
                "reason": "used artifact hint",
            }

    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "agent_node_file_tool_rounds": 1})
    llm = FileReadingSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    worker.with_artifact_root(str(tmp_path / "artifacts"))
    worker.artifact_store.write_text("planner", "hint.txt", "metric column is order_gmv_amt_1d", preview_chars=0)
    pack = profile_daily_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV最高的前5天",
                intent_type="VALID",
                answer_mode="TOPN",
                plan_task_id="anchor_gmv_file",
                preferred_table="ads_merchant_profile",
                sql_strategy="llm_first_debug",
                group_by_column="pt",
                metric_column="order_gmv_amt_1d",
                metric_name="order_gmv_amt_1d",
                output_keys=["seller_id", "pt"],
                days=30,
                limit=5,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "最近30天GMV最高的前5天", execution_mode="subagent")

    assert result.task_results[0].success
    assert llm.file_context_seen
    assert result.task_results[0].file_tool_results
    assert any(trace.tool_name == "node_file_context_tools" for trace in result.node_tool_traces)


def test_node_agent_accepts_window_sql_cte_alias_from_llm():
    settings = get_settings()
    llm = WindowRefundSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = trade_refund_goods_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="每个商品取最近一笔退款记录",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="latest_refund_per_goods",
                preferred_table="dwm_trade_refund_detail_di",
                sql_strategy="llm_first_debug",
                output_keys=["seller_id", "spu_name", "refund_id", "refund_create_time", "pay_amt"],
                required_evidence=["seller_id", "spu_name", "refund_id", "refund_create_time", "pay_amt"],
                days=30,
                limit=50,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "每个商品取最近一笔退款记录")

    assert result.task_results[0].success
    assert llm.calls == 1
    sql = result.task_results[0].query_bundle.sql
    assert "ROW_NUMBER()" in sql
    assert "PARTITION BY `spu_name`" in sql
    assert result.sql_draft_decisions[0].source == "llm_plan_bound"
    assert result.task_results[0].validation_results[-1].valid


def test_detail_node_uses_structured_fast_path_before_llm():
    settings = get_settings()
    llm = FakeLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = trade_refund_goods_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="查询子订单明细",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                filter_column="sub_order_id",
                filter_value="sub_order_id_100",
                output_keys=["seller_id", "sub_order_id", "order_id", "spu_id", "spu_name", "pt"],
                required_evidence=["seller_id", "sub_order_id", "order_id", "spu_id", "spu_name", "pt"],
                days=7,
                limit=20,
            )
        ]
    )
    result = worker.execute_plan("100", plan, pack, "", "查询子订单明细")
    assert result.task_results[0].success
    assert llm.calls == 0
    assert result.sql_draft_decisions[0].source == "structured_fast_path"
    assert any(item.tool_name == "draft_structured_sql_fast_path" for item in result.node_tool_traces)


def test_dependent_structured_sql_filters_only_relationship_keys():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = trade_refund_goods_pack()
    parent = AgentTaskResult(
        task_id="anchor_order",
        success=True,
        query_bundle=QueryBundle(
            tables=["dwm_trade_order_detail_di"],
            rows=[
                {
                    "seller_id": "100",
                    "sub_order_id": "sub_order_id_100",
                    "order_id": "order_id_100",
                    "spu_id": "spu_id_100",
                    "spu_name": "order_spu_name",
                    "pt": "2026-06-19",
                }
            ],
        ),
        entity_set=EntitySet(
            task_id="anchor_order",
            join_key="sub_order_id",
            values=["sub_order_id_100"],
            column_values={
                "sub_order_id": ["sub_order_id_100"],
                "order_id": ["order_id_100"],
                "spu_name": ["order_spu_name"],
                "pt": ["2026-06-19"],
            },
        ),
    )
    intent = QuestionIntent(
        question="查退款",
        intent_type="VALID",
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="refund_lookup",
        task_role=TaskRole.DEPENDENT,
        preferred_table="dwm_trade_refund_detail_di",
        group_by_column="sub_order_id",
        metric_column="refund_id",
        metric_name="refund_bill_cnt",
        output_keys=["seller_id", "sub_order_id", "order_id"],
        required_evidence=["sub_order_id", "order_id", "refund_id", "refund_status_name", "spu_name", "pt"],
        depends_on_task_ids=["anchor_order"],
        days=7,
    )
    plan = QueryPlan(
        intents=[intent],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="refund_lookup",
                join_key="sub_order_id",
                anchor_column="seller_id+sub_order_id",
                dependent_column="seller_id+sub_order_id",
            )
        ],
    )
    context = worker._node_context("refund_lookup", intent, {"anchor_order": parent}, plan, "100", "查退款", pack)
    sql = worker._draft_structured_sql(intent, pack, context)
    assert "`sub_order_id` IN ('sub_order_id_100')" in sql
    assert "`spu_name` IN" not in sql
    assert "`order_id` IN" not in sql
    assert "`pt` IN" not in sql


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
                sql_strategy="llm_first_debug",
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


def test_detail_doris_resource_error_uses_split_query_fallback():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    settings.agent_doris_split_query_enabled = True
    settings.agent_doris_split_chunk_days = 7
    settings.agent_doris_split_max_chunks = 3
    settings.agent_doris_split_max_concurrency = 3
    doris = DetailSplitFallbackDoris()
    worker = NodeWorkerExecutor(FakeLlm(), doris, SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "order_id", "sub_order_id", "spu_name"],
            )
        ]
    )
    intent = QuestionIntent(
        question="最近21天订单明细",
        intent_type="VALID",
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="detail_orders",
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="structured_first",
        output_keys=["order_id", "sub_order_id"],
        required_evidence=["spu_name", "pt"],
        days=21,
        limit=3,
    )

    result = worker.execute_node(intent, pack, "", NodeExecutionContext(merchant_id="100"))

    assert result.success
    assert result.query_bundle.original_row_count == 3
    assert result.query_bundle.runtime_events[0]["event"] == "split_query_fallback_started"
    assert result.query_bundle.runtime_events[0]["executionMode"] == "parallel_chunks"
    assert result.query_bundle.runtime_events[0]["maxConcurrency"] == 3
    assert any(item.tool_name == "execute_sql_split_fallback" for item in result.node_tool_traces)
    assert len([sql for sql in doris.sqls if "DATE_SUB(CURDATE(), INTERVAL 7 DAY)" in sql]) >= 1
    assert len(doris.sqls) >= 2


def test_detail_split_query_fallback_runs_chunks_in_parallel():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    settings.agent_doris_split_query_enabled = True
    settings.agent_doris_split_chunk_days = 7
    settings.agent_doris_split_max_chunks = 3
    settings.agent_doris_split_max_concurrency = 3
    doris = SlowDetailSplitFallbackDoris(delay_seconds=0.2)
    worker = NodeWorkerExecutor(FakeLlm(), doris, SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "order_id", "sub_order_id", "spu_name"],
            )
        ]
    )
    intent = QuestionIntent(
        question="最近21天订单明细",
        intent_type="VALID",
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="detail_orders",
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="structured_first",
        output_keys=["order_id", "sub_order_id"],
        required_evidence=["spu_name", "pt"],
        days=21,
        limit=3,
    )

    started = time.monotonic()
    result = worker.execute_node(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    elapsed = time.monotonic() - started

    assert result.success
    assert elapsed < 0.5
    events = result.query_bundle.runtime_events
    assert events[0]["executionMode"] == "parallel_chunks"
    assert events[0]["maxConcurrency"] == 3
    assert events[-1]["chunksSucceeded"] == 3
    assert events[-1]["executionMode"] == "parallel_chunks"


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
                sql_strategy="llm_first_debug",
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
                sql_strategy="llm_first_debug",
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
                sql_strategy="llm_first_debug",
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
                sql_strategy="llm_first_debug",
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
    assert "DATE_SUB(CURDATE(), INTERVAL 30 DAY)" in result.task_results[0].query_bundle.sql
    assert "DATE_FORMAT" not in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_structured_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_entity_dimension_topn_structured_sql_filters_blank_group_key():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=["seller_id", "pt", "spu_id", "spu_name", "ticket_id"],
            )
        ]
    )
    intent = QuestionIntent(
        question="客服工单量最高商品",
        intent_type="VALID",
        answer_mode=AnswerMode.TOPN,
        plan_task_id="anchor_ticket",
        task_role=TaskRole.ANCHOR,
        preferred_table="dwm_cs_ticket_detail_di",
        sql_strategy="structured_first",
        group_by_column="spu_id",
        metric_column="ticket_id",
        metric_name="ticket_cnt",
        metric_formula="COUNT(DISTINCT `ticket_id`)",
        output_keys=["seller_id", "spu_id", "spu_name"],
        days=60,
        limit=10,
    )

    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))

    assert "`spu_id` IS NOT NULL" in sql
    assert "`spu_id` != ''" in sql
    assert "GROUP BY `seller_id`, `spu_id`, `spu_name`" in sql


def test_node_agent_missing_entity_key_filter_uses_structured_fallback():
    settings = get_settings()
    settings.agent_sql_repair_rounds = 1
    llm = MissingEntityKeyFilterSqlLlm()
    worker = NodeWorkerExecutor(llm, FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=["seller_id", "pt", "spu_id", "spu_name", "ticket_id"],
            )
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近60天客服工单量最高的前10个商品",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_ticket",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_cs_ticket_detail_di",
                sql_strategy="llm_first_debug",
                group_by_column="spu_id",
                metric_column="ticket_id",
                metric_name="ticket_cnt",
                metric_formula="COUNT(DISTINCT `ticket_id`)",
                output_keys=["seller_id", "spu_id", "spu_name"],
                required_evidence=["seller_id", "spu_id", "spu_name", "ticket_id"],
                days=60,
                limit=10,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "最近60天客服工单量最高的前10个商品")

    assert result.task_results[0].success
    assert result.sql_repairs[0].error_code == "MISSING_ENTITY_KEY_FILTER"
    assert "`spu_id` IS NOT NULL" in result.task_results[0].query_bundle.sql
    assert "`spu_id` != ''" in result.task_results[0].query_bundle.sql
    assert any(item.tool_name == "draft_structured_sql_fallback" for item in result.node_tool_traces)
    assert not any(item.tool_name == "repair_sql" for item in result.node_tool_traces)
    assert llm.calls == 1


def test_entity_set_from_rows_ignores_blank_entity_values():
    intent = QuestionIntent(
        answer_mode=AnswerMode.TOPN,
        plan_task_id="anchor_ticket",
        group_by_column="spu_id",
    )
    rows = [
        {"spu_id": None, "spu_name": None, "ticket_id": "ticket_1"},
        {"spu_id": "", "spu_name": "", "ticket_id": "ticket_2"},
        {"spu_id": "   ", "spu_name": "   ", "ticket_id": "ticket_3"},
        {"spu_id": "spu_id_001", "spu_name": "商品A", "ticket_id": "ticket_4"},
    ]

    entity_set = entity_set_from_rows("anchor_ticket", intent, rows, 200)

    assert entity_set.values == ["spu_id_001"]
    assert entity_set.column_values["spu_id"] == ["spu_id_001"]
    assert entity_set.column_values["spu_name"] == ["商品A"]


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
    package = answer_data_package("最近30天GMV最高的前5天，同时看退款金额和工单量", plan, result)
    disclosed = {item.get("metricKey") for item in package["metricDisclosures"]}
    assert {"order_gmv_amt_1d", "refund_amt_1d", "cs_ticket_cnt_1d"} <= disclosed


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
    result = worker.execute_plan("100", plan, pack, "", "并发测试", execution_mode="subagent")
    elapsed = time.monotonic() - started
    assert [item.success for item in result.task_results] == [True, True]
    assert elapsed < 0.85
    assert result.node_execution_batches
    assert set(result.node_execution_batches[0].submitted_task_ids) == {"node_a", "node_b"}
    assert result.node_execution_batches[0].max_concurrency == 2


def test_node_worker_limits_parallel_subagents_by_task_cap():
    settings = get_settings()
    settings.max_concurrent_sub_agents = 10
    settings.max_sub_agent_tasks = 2
    settings.agent_node_timeout_seconds = 5
    worker = NodeWorkerExecutor(EmptySqlLlm(), SlowDoris(delay_seconds=0.1), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="table_a", columns=["seller_id"]),
            PlanningAssetEntry(table="table_b", columns=["seller_id"]),
            PlanningAssetEntry(table="table_c", columns=["seller_id"]),
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
            QuestionIntent(
                question="C",
                intent_type="VALID",
                answer_mode="DETAIL",
                plan_task_id="node_c",
                preferred_table="table_c",
                output_keys=["seller_id"],
                sql="SELECT `seller_id` FROM `table_c` WHERE `seller_id` = '100' LIMIT 1",
            ),
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "并发限制测试", execution_mode="subagent")

    assert [item.success for item in result.task_results] == [True, True, True]
    assert result.node_execution_batches[0].max_concurrency == 2
    assert any(event["event"] == "node.concurrency_limited" for event in result.node_execution_batches[0].runtime_events)
    concurrency_event = next(
        event for event in result.node_execution_batches[0].runtime_events if event["event"] == "node.concurrency_limited"
    )
    assert concurrency_event["requestedConcurrency"] == 10


def test_node_worker_isolates_slow_parallel_node_timeout():
    class TableDelayDoris:
        def __init__(self):
            self.calls = []

        def query(self, sql, params=None):
            self.calls.append(sql)
            if "table_b" in sql:
                time.sleep(1.25)
            return [{"seller_id": "100"}]

    settings = get_settings()
    settings.max_concurrent_sub_agents = 2
    settings.agent_node_timeout_seconds = 1
    settings.agent_node_poll_interval_seconds = 0.1
    repository = TableDelayDoris()
    worker = NodeWorkerExecutor(EmptySqlLlm(), repository, SqlValidationService(), settings)
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

    result = worker.execute_plan("100", plan, pack, "", "慢节点隔离")
    by_id = {item.task_id: item for item in result.task_results}

    assert by_id["node_a"].success
    assert not by_id["node_b"].success
    assert "超时" in by_id["node_b"].summary
    assert result.node_execution_batches[0].completed_task_ids == ["node_a"]
    assert result.node_execution_batches[0].timed_out_task_ids == ["node_b"]
    assert by_id["node_b"].query_bundle.runtime_events[0]["timeoutType"] == "node_timeout"
    batch_events = [item["event"] for item in result.node_execution_batches[0].runtime_events]
    assert "node.heartbeat" in batch_events
    assert "node.timeout" in batch_events
    time.sleep(0.4)
    assert len([sql for sql in repository.calls if "table_b" in sql]) == 1


def test_node_worker_resume_skips_successful_prior_node_result():
    class CountingDoris:
        def __init__(self):
            self.sqls = []

        def query(self, sql, params=None):
            self.sqls.append(sql)
            return [{"seller_id": "100"}]

    repo = CountingDoris()
    settings = get_settings()
    settings.max_concurrent_sub_agents = 1
    worker = NodeWorkerExecutor(EmptySqlLlm(), repo, SqlValidationService(), settings)
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
    prior = AgentTaskResult(
        task_id="node_a",
        success=True,
        summary="cached node_a",
        query_bundle=QueryBundle(sql="SELECT 1", tables=["table_a"], rows=[{"seller_id": "100"}]),
    )

    result = worker.execute_plan("100", plan, pack, "", "resume", resume_task_results=[prior])

    assert result.resumed_task_ids == ["node_a"]
    assert "node_a" in result.node_execution_batches[0].resumed_task_ids
    assert [item.task_id for item in result.task_results] == ["node_a", "node_b"]
    assert len(repo.sqls) == 1
    assert "table_b" in repo.sqls[0]


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
        sql_strategy="llm_first_debug",
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
    assert ("ticket_entity_expand", "repay_lookup") in dependency_pairs
    depends_by_task = {intent.plan_task_id: intent.depends_on_task_ids for intent in plan.intents}
    assert depends_by_task["refund_lookup"] == ["ticket_entity_expand"]
    assert depends_by_task["repay_lookup"] == ["ticket_entity_expand"]


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


def test_relationship_graph_repair_inserts_bridge_for_aggregate_parent_missing_key():
    question = "最近45天供应链入库量前10的商品，后续下单表现和退款表现怎么样。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "spu_id", "spu_name", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "refund_id", "pt"]),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            )
        ],
    )
    broken = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="order_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_order_detail_di",
                group_by_column="spu_id",
                output_keys=["seller_id", "spu_id", "spu_name"],
                depends_on_task_ids=["anchor_scm"],
            ),
            QuestionIntent(
                question=question,
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="refund_lookup",
                task_role="DEPENDENT",
                preferred_table="dwm_trade_refund_detail_di",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                depends_on_task_ids=["order_lookup"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="order_lookup",
                dependent_task_id="refund_lookup",
                join_key="sub_order_id",
                anchor_column="seller_id+sub_order_id",
                dependent_column="seller_id+sub_order_id",
            )
        ],
    )
    repaired = repair_dependency_key_production_gaps(question, broken, pack, [])
    bridge = next(intent for intent in repaired.intents if intent.answer_mode == AnswerMode.DETAIL)
    assert bridge.preferred_table == "dwm_trade_order_detail_di"
    assert "sub_order_id" in bridge.output_keys
    assert any(dep.anchor_task_id == "order_lookup" and dep.dependent_task_id == bridge.plan_task_id for dep in repaired.dependencies)
    assert any(dep.anchor_task_id == bridge.plan_task_id and dep.dependent_task_id == "refund_lookup" for dep in repaired.dependencies)
    assert QueryGraphValidator().validate(question, repaired, pack).valid


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


def test_graph_sanity_rejects_ranking_metric_as_dependent_root():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="ads_merchant_profile", columns=["seller_id", "order_detail_cnt", "ship_timeout_order_cnt_1d", "pt"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "ship_timeout_order_cnt_1d",
                "resolvedMetricRef": "ship_timeout_order_cnt_1d",
                "objectiveType": "metric_total",
            },
            "requestedMeasures": [],
        },
        intents=[
            QuestionIntent(
                question="最近7天发货超时订单量是多少",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="ads_merchant_profile",
                metric_name="order_detail_cnt",
                metric_column="order_detail_cnt",
                output_keys=["seller_id"],
            ),
            QuestionIntent(
                question="最近7天发货超时订单量是多少",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="ship_timeout_metric",
                task_role=TaskRole.DEPENDENT,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
                metric_column="ship_timeout_order_cnt_1d",
                output_keys=["seller_id"],
                depends_on_task_ids=["anchor_order"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="ship_timeout_metric",
                join_key="seller_id",
                anchor_column="seller_id",
                dependent_column="seller_id",
            )
        ],
    )

    result = QueryGraphValidator().validate("最近7天发货超时订单量是多少", plan, pack)
    reflection = PlannerReflectionAgent().reflect("最近7天发货超时订单量是多少", plan, pack)

    assert result.valid
    assert not reflection.passed
    codes = {issue.get("code") for issue in reflection.issues}
    assert "ROOT_METRIC_NOT_ROOT" in codes
    assert "DEPENDENCY_NOT_ENTITY_FILTER" in codes


def test_planner_reflection_allows_derived_ranking_metric_with_component_dependencies():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "refund_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "spu_name", "sub_order_id", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="refund_rate", table="dwm_trade_refund_detail_di", columns=["refund_id"]),
            PlanningAssetEntry(key="refund_bill_cnt", table="dwm_trade_refund_detail_di", columns=["refund_id"]),
            PlanningAssetEntry(key="order_detail_cnt", table="dwm_trade_order_detail_di", columns=["sub_order_id"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "refund_rate",
                "ownerTable": "dwm_trade_refund_detail_di",
                "groupByColumn": "spu_name",
                "objectiveType": "ranking",
            }
        },
        intents=[
            QuestionIntent(
                question="退款单量",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="component_refund_refund_bill_cnt",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="refund_bill_cnt",
                group_by_column="spu_name",
                output_keys=["seller_id", "spu_name"],
            ),
            QuestionIntent(
                question="下单数",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="component_order_order_detail_cnt",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                group_by_column="spu_name",
                output_keys=["seller_id", "spu_name"],
            ),
            QuestionIntent(
                question="退款率",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                plan_task_id="derived_refund_rate",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="refund_rate",
                group_by_column="spu_name",
                output_keys=["seller_id", "spu_name"],
                depends_on_task_ids=["component_refund_refund_bill_cnt", "component_order_order_detail_cnt"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="component_refund_refund_bill_cnt",
                dependent_task_id="derived_refund_rate",
                join_key="spu_name",
                anchor_column="spu_name",
                dependent_column="spu_name",
                relation_type="DERIVED_COMPONENT",
            ),
            PlanDependency(
                anchor_task_id="component_order_order_detail_cnt",
                dependent_task_id="derived_refund_rate",
                join_key="spu_name",
                anchor_column="spu_name",
                dependent_column="spu_name",
                relation_type="DERIVED_COMPONENT",
            ),
        ],
    )

    reflection = PlannerReflectionAgent().reflect("最近30天退款率最高的前10个商品", plan, pack)

    assert "ROOT_METRIC_NOT_ROOT" not in {issue.get("code") for issue in reflection.issues}


def test_graph_sanity_allows_dependency_with_entity_key():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "sub_order_id", "refund_id", "pt"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "resolvedMetricRef": "order_detail_cnt",
            },
            "requestedMeasures": [{"metricRef": "refund_bill_cnt", "resolvedMetricRef": "refund_bill_cnt"}],
        },
        intents=[
            QuestionIntent(
                question="订单看退款",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                metric_column="sub_order_id",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
            ),
            QuestionIntent(
                question="订单看退款",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="refund_lookup",
                task_role=TaskRole.DEPENDENT,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="refund_bill_cnt",
                metric_column="refund_id",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                depends_on_task_ids=["anchor_order"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="refund_lookup",
                join_key="seller_id,sub_order_id",
                anchor_column="seller_id,sub_order_id",
                dependent_column="seller_id,sub_order_id",
            )
        ],
    )

    result = QueryGraphValidator().validate("订单看退款", plan, pack)

    assert result.valid


def test_graph_sanity_requires_target_grain_output():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "sub_order_id", "bill_id", "pt"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "repay_bill_cnt",
                "resolvedMetricRef": "repay_bill_cnt",
                "groupByColumn": "spu_id",
            }
        },
        intents=[
            QuestionIntent(
                question="赔付单量较高商品",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_repay",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_cs_repay_detail_df",
                metric_name="repay_bill_cnt",
                metric_column="bill_id",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
            )
        ],
    )

    result = QueryGraphValidator().validate("赔付单量较高商品", plan, pack)
    reflection = PlannerReflectionAgent().reflect("赔付单量较高商品", plan, pack)

    assert result.valid
    assert not reflection.passed
    assert "TARGET_GRAIN_NOT_OUTPUT" in {issue.get("code") for issue in reflection.issues}


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


def test_lead_policy_fast_path_skips_reflection_before_validation():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": False,
        "query_graph_validated": False,
        "latency_optimization": {"eligible": True, "mode": "fast_path_verified_graph"},
        "react_round": 4,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近7天GMV是多少",
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.METRIC,
                    plan_task_id="gmv_metric",
                    preferred_table="ads_merchant_profile",
                )
            ]
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "validate_graph"
    assert "reflect_plan" not in decision.available_actions


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


def test_lead_policy_reunderstands_contract_mismatch_from_validator():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "react_round": 5,
        "query_graph_plan_attempts": 0,
        "query_graph_retrieve_count": 0,
        "query_graph_repair_attempts": 0,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="有优惠券的订单有退款占多少",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="anchor_order",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ]
        ),
        "planner_reflection": PlannerReflectionResult(passed=True),
        "query_graph_validation_result": GraphValidationResult(
            valid=False,
            gaps=[GraphValidationGap(code="SCOPE_NOT_NARROWING", evidence="dwm_trade_order_detail_di")],
            repairable=True,
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)
    assert decision.selected_action == "plan_graph"
    assert "repair_graph" not in decision.available_actions


def test_lead_policy_reunderstands_missing_explicit_object_filter():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "react_round": 5,
        "query_graph_plan_attempts": 0,
        "query_graph_retrieve_count": 0,
        "query_graph_repair_attempts": 0,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="查询订单 order_id_100 的订单明细，并查看退款金额。",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_refund",
                    preferred_table="dwm_trade_refund_detail_di",
                )
            ]
        ),
        "planner_reflection": PlannerReflectionResult(passed=True),
        "query_graph_validation_result": GraphValidationResult(
            valid=False,
            gaps=[GraphValidationGap(code="OBJECT_REF_FILTER_MISSING", evidence="order_id=order_id_100")],
            repairable=True,
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "plan_graph"
    assert "execute_graph" not in decision.available_actions


def test_lead_policy_reunderstands_invalid_ratio_numerator_before_retrieval():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "react_round": 5,
        "query_graph_plan_attempts": 0,
        "query_graph_retrieve_count": 0,
        "query_graph_repair_attempts": 0,
        "pending_knowledge_requests": [],
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="有优惠券的订单有退款占多少",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="anchor_order",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ]
        ),
        "planner_reflection": PlannerReflectionResult(passed=True),
        "query_graph_validation_result": GraphValidationResult(
            valid=False,
            gaps=[GraphValidationGap(code="CALCULATION_NUMERATOR_SAME_AS_DENOMINATOR", evidence="占多少")],
            repairable=True,
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "plan_graph"
    assert "retrieve_knowledge" not in decision.available_actions


def test_lead_policy_verifies_evidence_even_when_main_budget_exhausted():
    settings = get_settings()
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "react_round": settings.agent_main_rounds,
        "evidence_graph_verified": False,
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="anchor_order",
                    success=True,
                    query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_cnt": 1}]),
                )
            ]
        ),
    }

    decision = V2AgentPolicy(settings).decide(state)

    assert decision.selected_action == "verify_evidence"
    assert decision.selected_node == "verify_evidence_graph"
    assert decision.budget_exhausted


def test_lead_policy_can_dispatch_skill_worker_before_final_answer():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "sql_generated": True,
        "evidence_graph_verified": True,
        "chat_bi_completed": False,
        "react_round": 8,
        "plan": QueryPlan(
            question_understanding={
                "analysisIntent": "anomaly_check",
                "requiresExplanation": True,
                "analysisGrain": "day",
                "skillWorkflow": {"skillName": "bi_trend_attribution", "required": True},
            },
            intents=[
                QuestionIntent(
                    question="最近30天GMV是否异常？",
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.GROUP_AGG,
                    category=QuestionCategory.TRADE,
                    plan_task_id="gmv_trend",
                    metric_name="order_gmv_amt_1d",
                    group_by_column="pt",
                )
            ],
        ),
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="gmv_trend",
                    success=True,
                    query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100}]),
                )
            ],
            merged_query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100}]),
            verified_evidence=VerifiedEvidence(passed=True),
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "run_analysis_skill"
    assert decision.selected_node == "run_analysis_skill"
    assert "answer_data" in decision.available_actions


def test_lead_policy_does_not_dispatch_skill_for_plain_analysis_summary():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "sql_generated": True,
        "evidence_graph_verified": True,
        "chat_bi_completed": False,
        "react_round": 8,
        "plan": QueryPlan(
            question_understanding={"analysisIntent": "anomaly_check", "requiresExplanation": True, "analysisGrain": "day"},
            intents=[
                QuestionIntent(
                    question="最近30天GMV是否异常？",
                    intent_type=IntentType.VALID,
                    answer_mode=AnswerMode.GROUP_AGG,
                    category=QuestionCategory.TRADE,
                    plan_task_id="gmv_trend",
                    metric_name="order_gmv_amt_1d",
                    group_by_column="pt",
                )
            ],
        ),
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="gmv_trend",
                    success=True,
                    query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100}]),
                )
            ],
            merged_query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100}]),
            verified_evidence=VerifiedEvidence(passed=True),
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "answer_data"
    assert "run_analysis_skill" not in decision.available_actions[:1]


def test_lead_policy_executes_validated_graph_even_when_main_budget_exhausted():
    settings = get_settings()
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "react_round": settings.agent_main_rounds,
        "sql_generated": False,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近10天使用优惠券的订单中，有退货的订单占多少",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="order_lookup",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ]
        ),
        "query_graph_validation_result": GraphValidationResult(valid=True),
        "agent_run_result": AgentRunResult(task_results=[]),
    }

    decision = V2AgentPolicy(settings).decide(state)

    assert decision.selected_action == "execute_graph_direct"
    assert decision.selected_node == "execute_query_graph_direct"
    assert decision.budget_exhausted


def test_lead_policy_does_not_execute_invalid_graph_when_main_budget_exhausted():
    settings = get_settings()
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": True,
        "react_round": settings.agent_main_rounds,
        "sql_generated": False,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="最近10天使用优惠券的订单中，有退货的订单占多少",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="scope_event_ratio",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ]
        ),
        "query_graph_validation_result": GraphValidationResult(
            valid=False,
            gaps=[GraphValidationGap(code="CALCULATION_NUMERATOR_MISSING", evidence="有退货的订单")],
            repairable=True,
        ),
        "agent_run_result": AgentRunResult(task_results=[]),
    }

    decision = V2AgentPolicy(settings).decide(state)

    assert decision.selected_action == "answer_data"
    assert decision.selected_node == "answer_analysis"
    assert "execute_graph" not in decision.available_actions
    assert decision.budget_exhausted


def test_lead_policy_validates_after_reflection_repair_budget_exhausted():
    settings = get_settings()
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": False,
        "react_round": 8,
        "query_graph_repair_attempts": settings.agent_graph_repair_rounds,
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="查询订单 order_id_100 的订单明细",
                    intent_type="VALID",
                    answer_mode="DETAIL",
                    plan_task_id="anchor_order",
                    preferred_table="dwm_trade_order_detail_di",
                )
            ]
        ),
        "planner_reflection": PlannerReflectionResult(
            passed=False,
            issues=[{"code": "REPAIR_EXHAUSTED", "severity": "warning"}],
        ),
    }

    decision = V2AgentPolicy(settings).decide(state)

    assert decision.selected_action == "validate_graph"
    assert decision.selected_node == "validate_query_graph"


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
    assert decision.selected_action == "retrieve_knowledge"
    assert "reflect_plan" not in decision.available_actions


def test_lead_policy_stops_retrieval_when_knowledge_recall_stalls():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": False,
        "query_graph_validated": False,
        "react_round": 5,
        "query_graph_retrieve_count": 1,
        "query_graph_plan_attempts": 0,
        "pending_knowledge_requests": [
            KnowledgeRequest(
                type=KnowledgeRequestType.FIELD,
                query="客服工单和退款订单关联关系",
                reason="missing relationship evidence",
            )
        ],
        "lead_decision_context": {
            "progress": {
                "knowledgeRecallStalled": True,
                "newRecallRefsCount": 0,
            }
        },
        "plan": QueryPlan(),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "plan_graph"
    assert "retrieve_knowledge" not in decision.available_actions
    assert "answer_data" in decision.available_actions


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


def test_lead_policy_turns_semantic_repair_request_into_retrieve_action():
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
            issues=[{"code": "METRIC_RESOLUTION_NEEDED", "severity": "error"}],
            repair_reason="METRIC_RESOLUTION_NEEDED",
            repair_requests=[
                PlannerRepairRequest(
                    reason="METRIC_RESOLUTION_NEEDED",
                    action="semantic_read",
                    query="refund amount metric definition",
                )
            ],
        ),
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="退款金额",
                    intent_type="VALID",
                    answer_mode="GROUP_AGG",
                    plan_task_id="anchor_refund",
                    preferred_table="dwm_trade_refund_detail_di",
                )
            ]
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "retrieve_knowledge"
    assert "repair_graph" in decision.available_actions
    assert "repairRequests" in decision.reason


def test_lead_policy_turns_graph_repair_request_into_repair_action():
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
            issues=[{"code": "MISSING_EDGE", "severity": "error"}],
            repair_reason="MISSING_EDGE",
            repair_requests=[
                PlannerRepairRequest(
                    reason="MISSING_EDGE",
                    action="graph_repair",
                    query="add relationship edge",
                )
            ],
        ),
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="供应链商品看退款",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="anchor_scm",
                    preferred_table="dwm_scm_detail_di",
                )
            ]
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "repair_graph"
    assert "retrieve_knowledge" in decision.available_actions
    assert "repairRequests" in decision.reason


def test_lead_policy_repairs_structural_anchor_mismatch_before_reunderstand():
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "skills_loaded": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "query_graph_validated": False,
        "react_round": 7,
        "query_graph_plan_attempts": 1,
        "query_graph_repair_attempts": 0,
        "planner_reflection": PlannerReflectionResult(
            passed=False,
            issues=[{"code": "ROOT_METRIC_NOT_ROOT", "severity": "error", "taskId": "derived_refund_rate"}],
            repair_reason="ANCHOR_MISMATCH",
            repair_requests=[
                PlannerRepairRequest(
                    reason="ANCHOR_MISMATCH",
                    action="re_understand",
                    query="move refund_rate to root anchor",
                    task_id="derived_refund_rate",
                )
            ],
        ),
        "plan": QueryPlan(
            intents=[
                QuestionIntent(
                    question="退款率最高商品",
                    intent_type="VALID",
                    answer_mode="TOPN",
                    plan_task_id="derived_refund_rate",
                    preferred_table="dwm_trade_refund_detail_di",
                )
            ]
        ),
    }

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert decision.selected_action == "repair_graph"
    assert "plan_graph" not in decision.available_actions[:1]
    assert "ROOT_METRIC_NOT_ROOT" in decision.reason


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
        question_understanding={
            "analysisGrain": "order",
            "rankingObjective": {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di"},
            "requestedMeasures": [{"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款"}],
            "scopeConstraints": [],
            "filters": [],
        },
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
                    "suggestedTables": ["dwm_cs_ticket_detail_di"],
                    "suggestedFields": ["ticket_id"],
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


def test_sql_validator_allows_window_alias_from_cte_scope():
    pack = trade_refund_goods_pack()
    result = SqlValidationService().validate(
        """
        WITH ranked AS (
          SELECT
            seller_id,
            spu_name,
            refund_id,
            refund_create_time,
            pay_amt,
            pt,
            ROW_NUMBER() OVER(PARTITION BY spu_name ORDER BY refund_create_time DESC) AS rn
          FROM dwm_trade_refund_detail_di
          WHERE seller_id = '100'
            AND pt >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        )
        SELECT seller_id, spu_name, refund_id, refund_create_time, pay_amt, rn
        FROM ranked
        WHERE rn = 1
        LIMIT 50
        """,
        pack,
    )

    assert result.valid
    assert "ranked" in result.cte_names
    assert result.base_tables == ["dwm_trade_refund_detail_di"]


def test_sql_validator_still_rejects_unknown_real_column_inside_window():
    pack = trade_refund_goods_pack()
    result = SqlValidationService().validate(
        """
        WITH ranked AS (
          SELECT
            seller_id,
            spu_name,
            refund_id,
            ROW_NUMBER() OVER(PARTITION BY spu_name ORDER BY refund_created_at DESC) AS rn
          FROM dwm_trade_refund_detail_di
          WHERE seller_id = '100'
        )
        SELECT seller_id, spu_name, refund_id, rn
        FROM ranked
        WHERE rn = 1
        """,
        pack,
    )

    assert not result.valid
    assert result.error_code == "UNKNOWN_COLUMN"
    assert result.unknown_columns == ["dwm_trade_refund_detail_di.refund_created_at"]


def test_sql_validator_allows_derived_table_window_alias_scope():
    pack = trade_refund_goods_pack()
    result = SqlValidationService().validate(
        """
        SELECT r.seller_id, r.spu_name, r.refund_id, r.rn
        FROM (
          SELECT
            seller_id,
            spu_name,
            refund_id,
            ROW_NUMBER() OVER(PARTITION BY spu_name ORDER BY refund_create_time DESC) AS rn
          FROM dwm_trade_refund_detail_di
          WHERE seller_id = '100'
        ) r
        WHERE r.rn = 1
        """,
        pack,
    )

    assert result.valid
    assert result.base_tables == ["dwm_trade_refund_detail_di"]


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


def test_node_agent_anchors_relative_window_to_latest_partition_date():
    doris = CapturingFreshnessDoris("2026-06-24")
    settings = get_settings().model_copy(update={"agent_partition_date_anchor_enabled": True})
    worker = NodeWorkerExecutor(FakeLlm(), doris, SqlValidationService(), settings)
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
                question="最近7天订单量是多少？",
                intent_type="VALID",
                answer_mode="METRIC",
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                sql_strategy="structured_first",
                group_by_column="seller_id",
                metric_column="sub_order_id",
                metric_name="order_detail_cnt",
                metric_formula="COUNT(DISTINCT `sub_order_id`)",
                output_keys=["seller_id"],
                days=7,
                limit=1,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "最近7天订单量是多少？")

    assert result.task_results[0].success
    executed_sql = doris.sqls[-1]
    assert "CURDATE()" not in executed_sql
    assert "DATE_SUB('2026-06-24', INTERVAL 7 DAY)" in executed_sql
    assert len(result.task_results[0].query_bundle.rows) == 1
    assert any(item.tool_name == "anchor_partition_date" for item in result.node_tool_traces)


def test_node_agent_switches_to_realtime_fallback_when_offline_partition_is_stale():
    class RealtimeFallbackDoris:
        def __init__(self):
            self.sqls = []

        def query(self, sql, params=None):
            self.sqls.append(sql)
            if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
                return [{"min_pt": "20260601", "max_pt": "20260622"}]
            return [{"seller_id": "100", "sub_order_id": "sub_1", "order_detail_cnt": 1}]

    doris = RealtimeFallbackDoris()
    worker = NodeWorkerExecutor(FakeLlm(), doris, SqlValidationService(), get_settings())
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_realtime_di", columns=["seller_id", "sub_order_id", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_realtime_di",
                columns=["sub_order_id"],
                title="下单量",
                metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            )
        ],
        realtime_fallbacks=[
            PlanningAssetEntry(
                key="dwm_trade_order_detail_di",
                table="dwm_trade_order_realtime_di",
                title="订单实时明细兜底",
                metadata={"sourceTable": "dwm_trade_order_detail_di"},
            )
        ],
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="今天下单量",
                intent_type="VALID",
                answer_mode="GROUP_AGG",
                plan_task_id="today_order",
                preferred_table="dwm_trade_order_detail_di",
                sql_strategy="structured_first",
                metric_name="order_detail_cnt",
                metric_column="sub_order_id",
                metric_formula="COUNT(DISTINCT sub_order_id)",
                group_by_column="sub_order_id",
                output_keys=["seller_id", "sub_order_id"],
                days=1,
            )
        ]
    )

    result = worker.execute_plan("100", plan, pack, "", "今天下单量")

    assert result.task_results[0].success
    assert result.freshness_reports[0].status == "STALE_USE_REALTIME_FALLBACK"
    assert result.freshness_reports[0].fallback_table == "dwm_trade_order_realtime_di"
    assert any("dwm_trade_order_realtime_di" in sql for sql in doris.sqls if "MIN(`pt`)" not in sql)


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


def test_node_plan_contract_blocks_restricted_output_columns():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "buyer_phone", "pt"],
                metadata={
                    "merchantFilterColumn": "seller_id",
                    "rowAccessPolicy": {"scopeType": "merchant", "filterColumn": "seller_id", "valueSource": "merchant_id", "operator": "eq", "required": True},
                },
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="seller_id",
                table="dwm_trade_order_detail_di",
                metadata={"semantic": {"columnName": "seller_id", "visibilityPolicy": {"level": "public"}, "maskingPolicy": {"strategy": "none"}}},
            ),
            PlanningAssetEntry(
                key="buyer_phone",
                table="dwm_trade_order_detail_di",
                metadata={
                    "semantic": {
                        "columnName": "buyer_phone",
                        "visibilityPolicy": {"level": "restricted", "allowedRoles": ["merchant_admin"], "reason": "PII"},
                        "maskingPolicy": {"strategy": "full", "reason": "PII"},
                    }
                },
            ),
        ],
    )
    intent = QuestionIntent(
        question="看买家手机号",
        intent_type="VALID",
        answer_mode="DETAIL",
        plan_task_id="order_detail",
        preferred_table="dwm_trade_order_detail_di",
        output_keys=["seller_id", "buyer_phone"],
        sql_strategy="structured_first",
    )

    contract = worker._node_plan_contract(intent, pack, NodeExecutionContext(merchant_id="100"))
    critique = worker.node_plan_critic.review(contract)

    assert "buyer_phone" in contract.allowed_columns
    assert "buyer_phone" not in contract.visible_columns
    assert "buyer_phone" in contract.internal_only_columns
    assert not critique.valid
    assert any(issue["code"] == "PERMISSION_DENIED_OUTPUT_COLUMN" for issue in critique.issues)


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


def test_execute_node_masks_restricted_columns_for_allowed_role():
    settings = get_settings()

    class SensitiveDoris:
        def query(self, sql, params=None):
            if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
                return [{"min_pt": "20260601", "max_pt": "20260622"}]
            return [{"seller_id": "100", "buyer_phone": "13812345678", "pt": "2026-07-01"}]

    worker = NodeWorkerExecutor(FakeLlm(), SensitiveDoris(), SqlValidationService(), settings)
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "buyer_phone", "pt"],
                metadata={
                    "merchantFilterColumn": "seller_id",
                    "rowAccessPolicy": {"scopeType": "merchant", "filterColumn": "seller_id", "valueSource": "merchant_id", "operator": "eq", "required": True},
                },
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="buyer_phone",
                table="dwm_trade_order_detail_di",
                metadata={
                    "semantic": {
                        "columnName": "buyer_phone",
                        "visibilityPolicy": {"level": "restricted", "allowedRoles": ["merchant_admin"], "reason": "PII"},
                        "maskingPolicy": {"strategy": "full", "reason": "PII"},
                    }
                },
            )
        ],
    )
    intent = QuestionIntent(
        question="看买家手机号",
        intent_type="VALID",
        answer_mode="DETAIL",
        plan_task_id="order_detail",
        preferred_table="dwm_trade_order_detail_di",
        output_keys=["buyer_phone", "pt"],
        required_evidence=["buyer_phone"],
        sql_strategy="structured_first",
        days=7,
    )

    result = worker.execute_node(intent, pack, "", NodeExecutionContext(merchant_id="100", access_role="merchant_admin"))

    assert result.success
    assert result.query_bundle.rows[0]["buyer_phone"] == "***"


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


def test_compiler_projects_order_grain_repay_metric_to_product_grain():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_cs_repay_detail_df", columns=["seller_id", "sub_order_id", "bill_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "spu_id", "spu_name", "pt"]),
            PlanningAssetEntry(table="dwm_goods_detail_df", columns=["seller_id", "spu_id", "spu_name", "spu_apply_create_time", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "spu_name", "refund_id", "pay_amt", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="repay_bill_cnt",
                table="dwm_cs_repay_detail_df",
                columns=["bill_id"],
                metadata={"formula": "COUNT(DISTINCT bill_id)", "sourceColumns": ["bill_id"]},
            ),
            PlanningAssetEntry(
                key="refund_bill_cnt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_id"],
                metadata={"formula": "COUNT(DISTINCT refund_id)", "sourceColumns": ["refund_id"]},
            ),
            PlanningAssetEntry(
                key="pay_amt",
                table="dwm_trade_refund_detail_di",
                columns=["pay_amt"],
                metadata={"formula": "SUM(pay_amt)", "sourceColumns": ["pay_amt"]},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_repay_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="goods_order_by_spu_id",
                left_table="dwm_goods_detail_df",
                right_table="dwm_trade_order_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="goods_refund_by_spu_name",
                left_table="dwm_goods_detail_df",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_name", "rightColumn": "spu_name"},
                ],
            ),
        ],
    )
    understanding = {
        "analysisGrain": "product",
        "rankingObjective": {
            "metricRef": "repay_bill_cnt",
            "ownerTable": "dwm_cs_repay_detail_df",
            "sourcePhrase": "赔付单量",
            "groupByColumn": "spu_id",
            "limit": 10,
            "objectiveType": "ranking",
        },
        "requestedMeasures": [
            {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款量"},
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "SPU 发布时间",
                "requiredLevel": "required",
                "suggestedDomains": ["goods"],
                "suggestedTables": ["dwm_goods_detail_df"],
                "suggestedFields": ["spu_apply_create_time"],
            }
        ],
        "timeWindowDays": 60,
    }

    plan = QuestionUnderstandingCompiler().compile(
        "找到最近60天赔付单量较高的商品，看下对应的退款量、退款金额和商品发布时间。",
        understanding,
        pack,
    )

    projection = next(
        intent
        for intent in plan.intents
        if intent.answer_mode == AnswerMode.DERIVED
        and intent.metric_name == "repay_bill_cnt"
        and intent.group_by_column == "spu_id"
        and intent.metric_resolution.get("computeStrategy") == "projection_group_aggregate"
    )
    goods = next(intent for intent in plan.intents if intent.preferred_table == "dwm_goods_detail_df")
    deps_to_projection = {dep.anchor_task_id for dep in plan.dependencies if dep.dependent_task_id == projection.plan_task_id}
    deps_to_goods = {dep.anchor_task_id for dep in plan.dependencies if dep.dependent_task_id == goods.plan_task_id}

    assert deps_to_projection == {"anchor_repay", "order_bridge"}
    assert projection.plan_task_id in deps_to_goods
    assert any(item.startswith("PROJECT_ROOT_GROUP_BY:") for item in plan.compiler_trace)
    assert QueryGraphValidator().validate("赔付单量较高的商品", plan, pack).valid


def test_node_worker_projection_compute_groups_metric_rows_by_bridge_product():
    settings = get_settings()
    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), settings)
    intent = QuestionIntent(
        question="赔付单量较高的商品",
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.DERIVED,
        plan_task_id="projected_repay_bill_cnt_by_spu_id",
        task_role=TaskRole.DEPENDENT,
        metric_name="repay_bill_cnt",
        group_by_column="spu_id",
        limit=10,
        metric_resolution={
            "computeStrategy": "projection_group_aggregate",
            "metricKey": "repay_bill_cnt",
            "sourceMetricTaskId": "anchor_repay",
            "bridgeTaskId": "order_bridge",
            "sourceJoinKey": "sub_order_id",
            "bridgeJoinKey": "sub_order_id",
            "groupByColumn": "spu_id",
            "carryColumns": ["seller_id", "spu_id", "spu_name", "sub_order_id"],
            "sourceMetricAliases": ["repay_bill_cnt"],
        },
    )

    rows, error = worker._compute_derived_metric_rows(
        intent,
        NodeExecutionContext(
            upstream_rows=[
                {"__source_task_id": "anchor_repay", "sub_order_id": "sub_1", "repay_bill_cnt": 1},
                {"__source_task_id": "anchor_repay", "sub_order_id": "sub_2", "repay_bill_cnt": 1},
                {"__source_task_id": "anchor_repay", "sub_order_id": "sub_3", "repay_bill_cnt": 1},
                {"__source_task_id": "order_bridge", "sub_order_id": "sub_1", "seller_id": "100", "spu_id": "spu_a", "spu_name": "A"},
                {"__source_task_id": "order_bridge", "sub_order_id": "sub_2", "seller_id": "100", "spu_id": "spu_a", "spu_name": "A"},
                {"__source_task_id": "order_bridge", "sub_order_id": "sub_3", "seller_id": "100", "spu_id": "spu_b", "spu_name": "B"},
            ]
        ),
    )

    assert not error
    assert rows[0]["spu_id"] == "spu_a"
    assert rows[0]["repay_bill_cnt"] == 2
    assert rows[1]["spu_id"] == "spu_b"
    assert rows[1]["repay_bill_cnt"] == 1


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
        task_results=[
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"refund_related_pay_amt_raw": "121.50"}]),
            )
        ],
        merged_query_bundle=QueryBundle(tables=["dwm_trade_refund_detail_di"], rows=[{"refund_related_pay_amt_raw": "121.50"}], original_row_count=1),
    )
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "refund_top",
                "table": "dwm_trade_refund_detail_di",
                "columns": ["refund_related_pay_amt_raw"],
                "semanticLabel": "refund_metric",
                "requiredLevel": "required",
                "metricResolution": {
                    "requestedMetricRef": "refund_amt",
                    "metricKey": "",
                    "ownerTable": "dwm_trade_refund_detail_di",
                    "confidence": 0,
                    "resolutionSource": "unresolved",
                },
            }
        ],
    )
    verified = EvidenceVerifier().verify("看售后金额", plan, run)
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


def test_evidence_verifier_rejects_knowledge_ref_not_in_recall_bundle():
    plan = QueryPlan(
        evidence_contracts=[
            {
                "taskId": "knowledge_rule",
                "evidenceSource": "knowledge_ref",
                "knowledgeRefs": ["invented-rule-id"],
                "semanticLabel": "GMV business rule",
                "requiredLevel": "required",
            }
        ]
    )

    verified = EvidenceVerifier().verify("GMV 口径是什么", plan, AgentRunResult(), allowed_knowledge_refs=set())

    assert not verified.passed
    assert any(gap.code == "MISSING_REQUIRED_EVIDENCE" for gap in verified.gaps)


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


def test_hybrid_recall_returns_semantic_metric_evidence():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近45天优惠券金额投入最高的商品，退款率是否偏高？"
    recall = HybridRecallService(settings, topic_assets, WikiMemoryService(settings)).recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )

    metric_hits = [item for item in recall.items if str(item.source_type).upper() == "SEMANTIC_METRIC"]

    assert metric_hits
    assert any(item.metadata.get("metricKey") == "coupon_total_amt" for item in metric_hits)


def test_semantic_metric_resolver_uses_recalled_metric_evidence_for_bad_llm_ref():
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    question = "最近45天优惠券金额投入最高的商品，退款率是否偏高？"
    recall_service = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    recall = recall_service.recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    pack = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings)).compact(
        question,
        recall,
        [QuestionCategory.COUPON, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )

    resolution = SemanticMetricResolver(pack).resolve(
        question,
        "coupon_amt",
        "dwm_coupon_detail_di",
        "优惠券金额投入",
    )

    assert resolution.metric.key == "coupon_total_amt"
    assert resolution.resolution_source == "semantic_recall_evidence"
    assert resolution.confidence >= 0.9
    assert resolution.candidate_evidence[0]["recallEvidence"]["metricKey"] == "coupon_total_amt"


def test_semantic_metric_index_prefers_source_phrase_over_bad_llm_metric_ref():
    settings = get_settings()
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    index = SemanticMetricIndex(builder._all_metric_entries())
    gmv = index.resolve("pay_amt", "dwm_trade_order_detail_di", "GMV")
    assert gmv
    assert gmv.metric.table == "ads_merchant_profile"
    assert gmv.metric.key == "order_gmv_amt_1d"
    assert gmv.resolution_reason == "semantic_phrase_override"

    refund = index.resolve("pay_amt", "dwm_trade_refund_detail_di", "退款金额")
    assert refund
    assert refund.metric.table == "dwm_trade_refund_detail_di"
    assert refund.metric.key == "pay_amt"
    assert refund.resolution_reason == "semantic_metric_ref"


def test_semantic_metric_resolver_keeps_exact_refs_for_broad_detail_phrase():
    settings = get_settings()
    question = "查询退款单 refund_id_100 的退款明细，并关联订单和商品信息。"
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    pack = builder.compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.REFUND, QuestionCategory.TRADE, QuestionCategory.GOODS],
    )
    resolver = SemanticMetricResolver(pack)

    pay_amt = resolver.resolve(question, "pay_amt", "dwm_trade_refund_detail_di", "退款明细")
    sku_count = resolver.resolve(question, "sku_count", "dwm_trade_refund_detail_di", "退款明细")

    assert pay_amt.metric
    assert pay_amt.metric.key == "pay_amt"
    assert pay_amt.resolution_source == "semantic_metric_ref"
    assert sku_count.metric
    assert sku_count.metric.key == "sku_count"
    assert sku_count.resolution_source == "semantic_metric_ref"


def test_compiler_refund_detail_preserves_requested_metrics_and_valid_dependencies():
    settings = get_settings()
    question = "查询退款单 refund_id_100 的退款明细，并关联订单和商品信息。"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.REFUND, QuestionCategory.TRADE, QuestionCategory.GOODS],
    )
    understanding = {
        "analysisGrain": "refund",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "refund_bill_cnt",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款明细",
            "objectiveType": "detail_anchor",
            "groupByColumn": "refund_id",
            "limit": 100,
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单信息"},
            {"metricRef": "goods_cnt", "ownerTable": "dwm_goods_detail_df", "sourcePhrase": "商品信息"},
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款明细"},
            {"metricRef": "sku_count", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款明细"},
        ],
        "filters": [{"field": "refund_id", "value": "refund_id_100"}],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    result = QueryGraphValidator().validate(question, plan, pack)
    metric_pairs = {(intent.preferred_table, intent.metric_name) for intent in plan.intents if intent.metric_name}

    assert result.valid
    assert ("dwm_trade_refund_detail_di", "pay_amt") in metric_pairs
    assert ("dwm_trade_refund_detail_di", "sku_count") in metric_pairs
    assert not any(gap.code == "REQUESTED_MEASURE_NOT_PLANNED" for gap in result.gaps)
    assert not any(gap.code == "DEPENDENCY_KEY_NOT_PRODUCED" for gap in result.gaps)


def test_compiler_detail_anchor_uses_explicit_order_filter_even_when_metric_points_to_refund():
    question = "查询订单 order_id_100 的订单明细，并查看退货时间、退货用户、退款金额，再看下对应 SPU 什么时候发布的。"
    pack = trade_refund_goods_pack()
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单明细",
            "objectiveType": "detail_anchor",
            "groupByColumn": "order_id",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"}
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "商品发布时间",
                "requiredLevel": "required",
                "suggestedTables": ["dwm_goods_detail_df"],
                "suggestedFields": ["spu_apply_create_time"],
            }
        ],
        "filters": [{"field": "order_id", "value": "order_id_100"}],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    validation = QueryGraphValidator().validate(question, plan, pack)
    root = plan.intents[0]

    assert root.preferred_table == "dwm_trade_order_detail_di"
    assert root.answer_mode == AnswerMode.DETAIL
    assert root.filter_column == "order_id"
    assert root.filter_value == "order_id_100"
    assert any(intent.preferred_table == "dwm_trade_refund_detail_di" for intent in plan.intents)
    assert not any(gap.code == "OBJECT_REF_FILTER_MISSING" for gap in validation.gaps)
    assert validation.valid


def test_compiler_scope_phrase_does_not_override_explicit_order_anchor_metric():
    settings = get_settings()
    question = "最近20天优惠券活动订单的退款率和GMV表现怎么样？"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.COUPON, QuestionCategory.TRADE, QuestionCategory.REFUND],
    )
    understanding = {
        "analysisGrain": "product",
        "analysisIntent": "comparison",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "最近20天优惠券活动订单表现",
            "groupByColumn": "spu_name",
            "limit": 10,
        },
        "requestedMeasures": [
            {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款量"},
            {"metricRef": "discount_amt", "ownerTable": "dwm_coupon_detail_di", "sourcePhrase": "券金额投入"},
            {"metricRef": "refund_rate", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款率"},
        ],
        "scopeConstraints": [
            {
                "scopeId": "coupon_activity_orders",
                "sourcePhrase": "优惠券活动订单",
                "ownerTable": "dwm_coupon_detail_di",
                "metricRef": "coupon_amt",
                "entityGrain": "order",
                "targetDomain": "trade",
                "required": True,
            }
        ],
        "timeWindowDays": 20,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    assert plan.knowledge_requests
    recall_service = HybridRecallService(settings, TopicAssetService(settings), WikiMemoryService(settings))
    scoped_items = []
    for request in plan.knowledge_requests:
        scoped_recall = recall_service.recall(
            request.query,
            KeywordExtractService().extract(request.query),
            [],
            "",
            "100",
            [QuestionCategory.COUPON, QuestionCategory.TRADE, QuestionCategory.REFUND],
        )
        scoped_items.extend(scoped_recall.items)
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        RecallBundle(items=scoped_items),
        [QuestionCategory.COUPON, QuestionCategory.TRADE, QuestionCategory.REFUND],
    )
    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    result = QueryGraphValidator().validate(question, plan, pack)

    order_anchor = next(intent for intent in plan.intents if intent.plan_task_id == "order_lookup")
    assert order_anchor.preferred_table == "dwm_trade_order_detail_di"
    assert order_anchor.metric_name == "order_detail_cnt"
    assert plan.question_understanding["rankingObjective"]["resolvedMetricRef"] == "order_detail_cnt"
    assert any(
        item.get("metricRef") == "discount_amt" and item.get("resolvedMetricRef") in {"coupon_amt", "coupon_total_amt"}
        for item in plan.question_understanding.get("requestedMeasures", [])
    )
    assert result.valid


def test_compiler_does_not_turn_suggested_domains_into_query_nodes():
    settings = get_settings()
    question = "最近30天退款率最高的前10个商品，同时看下单数、退款金额和商品发布时间，帮我判断哪些是高风险新品。"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    understanding = {
        "analysisGrain": "product",
        "analysisIntent": "risk_ranking",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "refund_rate",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款率最高的前10个商品",
            "groupByColumn": "spu_id",
            "limit": 10,
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "下单数"},
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"},
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "new_product_identification",
                "reason": "需要商品发布时间判断新品",
                "requiredLevel": "required",
                "suggestedDomains": ["goods"],
            }
        ],
        "scopeConstraints": [],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    result = QueryGraphValidator().validate(question, plan, pack)
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)

    assert result.valid
    assert not any(intent.preferred_table == "dwm_goods_detail_df" for intent in plan.intents)
    assert not any(item.startswith("EVIDENCE_DOMAIN_REPAIR:") and "goods" in item for item in plan.compiler_trace)
    assert "DOMAIN_COVERAGE_GAP" not in {issue["code"] for issue in reflection.issues}


def test_compiler_binds_required_evidence_suggested_fields_to_matching_nodes():
    settings = get_settings()
    question = "查询订单 order_id_100 的订单明细，并查看退款金额，再看下对应 SPU 什么时候发布的。"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单明细",
            "objectiveType": "detail_anchor",
            "groupByColumn": "order_id",
            "limit": 20,
        },
        "requestedMeasures": [
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"}
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "商品发布时间",
                "requiredLevel": "required",
                "suggestedDomains": ["goods"],
                "suggestedTables": ["dwm_goods_detail_df"],
                "suggestedFields": ["spu_apply_create_time"],
            }
        ],
        "filters": [{"field": "order_id", "value": "order_id_100"}],
        "timeWindowDays": 90,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    goods_node = next(intent for intent in plan.intents if intent.preferred_table == "dwm_goods_detail_df")

    assert "spu_apply_create_time" in goods_node.output_keys
    assert "spu_apply_create_time" in goods_node.required_evidence
    assert QueryGraphValidator().validate(question, plan, pack).valid


def test_query_graph_validator_rejects_missing_required_field_evidence():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_goods_detail_df",
                columns=["seller_id", "spu_id", "spu_apply_create_time", "pt"],
            )
        ]
    )
    plan = QueryPlan(
        question_understanding={
            "requiredEvidenceIntents": [
                {
                    "semanticLabel": "商品发布时间",
                    "requiredLevel": "required",
                    "suggestedTables": ["dwm_goods_detail_df"],
                    "suggestedFields": ["spu_apply_create_time"],
                }
            ]
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="goods_lookup",
                preferred_table="dwm_goods_detail_df",
                output_keys=["seller_id", "spu_id"],
            )
        ],
    )

    validation = QueryGraphValidator().validate("商品发布时间", plan, pack)
    reflection = PlannerReflectionAgent().reflect("商品发布时间", plan, pack)

    assert validation.valid
    assert not reflection.passed
    assert any(issue.get("code") == "MISSING_REQUIRED_FIELD_EVIDENCE" for issue in reflection.issues)


def test_query_graph_validator_rejects_pending_knowledge_requests():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "order_id", "pt"],
            )
        ]
    )
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="order_metric",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_amt",
                metric_column="order_id",
                output_keys=["seller_id"],
            )
        ],
        knowledge_requests=[
            KnowledgeRequest(
                type=KnowledgeRequestType.METRIC,
                query="订单金额 语义指标口径 公式 来源字段",
                needed_for_task_id="order_metric",
                reason="metric evidence unresolved",
            )
        ],
    )

    validation = QueryGraphValidator().validate("订单金额", plan, pack)

    assert not validation.valid
    assert any(gap.code == "PENDING_KNOWLEDGE_REQUEST" for gap in validation.gaps)


def test_query_graph_validator_requires_explicit_object_ref_filter():
    question = "查询订单 order_id_100 的订单明细，并查看退款金额。"
    pack = trade_refund_goods_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_refund",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="order_id",
                output_keys=["seller_id", "order_id"],
            )
        ]
    )

    validation = QueryGraphValidator().validate(question, plan, pack)

    assert not validation.valid
    assert any(gap.code == "OBJECT_REF_FILTER_MISSING" and gap.evidence == "order_id=order_id_100" for gap in validation.gaps)


def test_query_graph_validator_accepts_explicit_object_ref_detail_filter():
    question = "查询订单 order_id_100 的订单明细，并查看退款金额。"
    pack = trade_refund_goods_pack()
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                filter_column="order_id",
                filter_value="order_id_100",
                output_keys=["seller_id", "order_id", "sub_order_id"],
            )
        ]
    )

    validation = QueryGraphValidator().validate(question, plan, pack)

    assert not any(gap.code == "OBJECT_REF_FILTER_MISSING" for gap in validation.gaps)


def test_compiler_preserves_parallel_detail_branches_with_topn_metric():
    question = "最近7天订单明细和退款明细都给我看一下，并找出退款金额最高的前3单。"
    pack = trade_refund_goods_pack()
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款金额",
            "objectiveType": "ranking",
            "groupByColumn": "order_id",
            "order": "desc",
            "limit": 3,
        },
        "requestedMeasures": [
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单明细"},
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款明细"},
        ],
        "requiredEvidenceIntents": [],
        "filters": [],
        "timeWindowDays": 7,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    validation = QueryGraphValidator().validate(question, plan, pack)
    detail_tables = {
        intent.preferred_table
        for intent in plan.intents
        if intent.answer_mode == AnswerMode.DETAIL and "detailEvidence=" in str(intent.analysis_note)
    }

    assert validation.valid
    assert {"dwm_trade_order_detail_di", "dwm_trade_refund_detail_di"} <= detail_tables
    assert any(intent.answer_mode == AnswerMode.TOPN and intent.metric_name == "pay_amt" for intent in plan.intents)
    assert any(item.startswith("DETAIL_EVIDENCE_BRANCH:") for item in plan.compiler_trace)
    assert not any(str(intent.analysis_note).startswith("missingDomain=") for intent in plan.intents)


def test_validator_rejects_missing_parallel_detail_branch():
    question = "最近7天订单明细和退款明细都给我看一下，并找出退款金额最高的前3单。"
    pack = trade_refund_goods_pack()
    plan = QueryPlan(
        question_understanding={
            "requestedMeasures": [
                {"metricRef": "pay_amt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单明细"},
                {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款明细"},
            ]
        },
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.TOPN,
                plan_task_id="anchor_refund",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="order_id",
                output_keys=["seller_id", "order_id", "pay_amt"],
            )
        ],
    )

    validation = QueryGraphValidator().validate(question, plan, pack)
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)

    assert validation.valid
    assert not reflection.passed
    assert {issue.get("code") for issue in reflection.issues} >= {"DETAIL_EVIDENCE_NOT_PLANNED"}


def test_compiler_does_not_turn_metric_phrases_into_detail_branches():
    question = "最近7天支付订单量和交易成功订单量分别是多少，差异大不大？"
    pack = trade_refund_goods_pack()
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "comparison",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "支付订单量",
            "groupByColumn": "pt",
        },
        "requestedMeasures": [
            {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "交易成功订单量",
                "groupByColumn": "pt",
            }
        ],
        "requiredEvidenceIntents": [],
        "filters": [],
        "timeWindowDays": 7,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)

    assert not any("detailEvidence=" in str(intent.analysis_note) for intent in plan.intents)


def test_compiler_does_not_repair_metric_domain_from_explanatory_evidence_domains():
    question = "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。"
    pack = trade_refund_goods_pack()
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "overview",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "发货超时订单量",
            "groupByColumn": "pt",
        },
        "requestedMeasures": [],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "发货超时规则与注意事项",
                "requiredLevel": "required",
                "suggestedDomains": ["scm", "order"],
                "reason": "需要规则说明，不是补 SCM 指标",
            }
        ],
        "filters": [],
        "timeWindowDays": 7,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)

    assert not any(str(item).startswith("EVIDENCE_DOMAIN_REPAIR") for item in plan.compiler_trace)


def test_coverage_critic_adds_recalled_metric_evidence_for_ship_timeout_order_count():
    question = "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。"
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "order_id", "pt"]),
            PlanningAssetEntry(table="ads_merchant_profile", columns=["seller_id", "pt", "ship_timeout_order_cnt_1d"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["order_id"],
                title="订单量",
                metadata={"metricKey": "order_detail_cnt", "sourceColumns": ["order_id"]},
            )
        ],
        metric_compaction={
            "recalledMetricEvidence": [
                {
                    "ownerTable": "ads_merchant_profile",
                    "metricKey": "ship_timeout_order_cnt_1d",
                    "businessName": "发货超时订单量",
                    "aliases": ["发货超时订单数", "超时发货订单量"],
                    "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
                    "sourceColumns": ["ship_timeout_order_cnt_1d"],
                    "formula": "SUM(ship_timeout_order_cnt_1d)",
                }
            ]
        },
    )
    understanding = {
        "analysisGrain": "merchant",
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单量",
        },
        "requestedMeasures": [],
        "requiredEvidenceIntents": [],
    }

    result = UnderstandingCoverageCritic().complete(question, understanding, pack)

    assert {
        "metricRef": "ship_timeout_order_cnt_1d",
        "ownerTable": "ads_merchant_profile",
        "sourcePhrase": "发货超时订单量",
        "completionSource": "recalled_metric_evidence",
        "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
    } in result.understanding["requestedMeasures"]
    assert any(item.startswith("UNDERSTANDING_RECALLED_METRIC_COMPLETION:") for item in result.trace)


def test_coverage_critic_prunes_requested_measure_that_duplicates_ranking_metric():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="ads_merchant_profile", columns=["merchant_id", "pt", "order_cnt_1d"]),
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pt", "sub_order_id"]),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_cnt_1d",
                table="ads_merchant_profile",
                columns=["order_cnt_1d"],
                title="总订单量",
                aliases=["订单量"],
                metadata={"metricKey": "order_cnt_1d", "sourceColumns": ["order_cnt_1d"]},
            ),
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="订单量",
                aliases=["下单数"],
                metadata={"metricKey": "order_detail_cnt", "sourceColumns": ["sub_order_id"]},
            ),
        ],
    )
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "order_cnt_1d",
            "ownerTable": "ads_merchant_profile",
            "sourcePhrase": "总订单量",
            "objectiveType": "metric_total",
        },
        "requestedMeasures": [
            {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "订单量",
            }
        ],
        "requiredEvidenceIntents": [],
    }

    result = UnderstandingCoverageCritic().complete("最近7天总订单量是多少？", understanding, pack)

    assert result.understanding["requestedMeasures"] == []
    assert any(item.startswith("UNDERSTANDING_OVER_COVERAGE_PRUNED:") for item in result.trace)


def test_compiler_treats_same_table_event_scope_anchor_as_population_contract():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=["seller_id", "ticket_id", "sub_order_id", "order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "order_id", "pt"],
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="ticket_cnt",
                table="dwm_cs_ticket_detail_di",
                columns=["ticket_id"],
                title="工单量",
                metadata={"sourceColumns": ["ticket_id"], "formula": "COUNT(DISTINCT ticket_id)"},
            ),
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="下单数",
                metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_ticket_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_ticket_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            )
        ],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "comparison",
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "有客服工单的订单",
            "groupByColumn": "sub_order_id",
            "limit": 100,
        },
        "requestedMeasures": [
            {"metricRef": "order_detail_cnt", "ownerTable": "dwm_trade_order_detail_di", "sourcePhrase": "订单数"}
        ],
        "scopeConstraints": [
            {
                "scopeId": "orders_with_ticket",
                "sourcePhrase": "有客服工单的订单",
                "ownerTable": "dwm_cs_ticket_detail_di",
                "metricRef": "ticket_cnt",
                "entityGrain": "order",
                "targetDomain": "ticket",
                "required": True,
            }
        ],
        "filters": [],
    }

    plan = QuestionUnderstandingCompiler().compile("最近30天有客服工单的订单里，哪些后来发生了退款或赔付？", understanding, pack)
    validation = QueryGraphValidator().validate("最近30天有客服工单的订单里，哪些后来发生了退款或赔付？", plan, pack)

    assert any(intent.plan_task_id == "anchor_ticket_scope" for intent in plan.intents)
    assert not any(gap.code == "SCOPE_NOT_NARROWING" for gap in validation.gaps)


def test_compiler_keeps_parallel_metrics_as_sibling_nodes_even_when_tables_are_related():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "sub_order_id", "refund_id", "pay_amt", "pt"],
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="订单量",
                metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            ),
            PlanningAssetEntry(
                key="pay_amt",
                table="dwm_trade_refund_detail_di",
                columns=["pay_amt"],
                title="退款金额",
                metadata={"sourceColumns": ["pay_amt"], "formula": "SUM(pay_amt)"},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            )
        ],
    )
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单量",
            "objectiveType": "metric_total",
            "groupByColumn": "seller_id",
            "order": "desc",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"}
        ],
        "scopeConstraints": [],
        "filters": [],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile("最近30天订单量和退款金额分别是多少", understanding, pack)
    refund_node = next(intent for intent in plan.intents if intent.metric_name == "pay_amt")

    assert refund_node.task_role == TaskRole.ANCHOR
    assert refund_node.depends_on_task_ids == []
    assert not any(dep.dependent_task_id == refund_node.plan_task_id for dep in plan.dependencies)
    assert any(item.startswith("GRAPH_ROLE:%s:sibling_metric" % refund_node.plan_task_id) for item in plan.compiler_trace)
    assert PlannerReflectionAgent().reflect("最近30天订单量和退款金额分别是多少", plan, pack).passed


def test_compiler_adds_daily_trend_context_for_time_window_metric():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "pt"],
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="订单量",
                metadata={"metricKey": "order_detail_cnt", "sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            )
        ],
    )
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单量",
            "objectiveType": "metric_total",
            "groupByColumn": "seller_id",
            "limit": 1,
        },
        "requestedMeasures": [],
        "scopeConstraints": [],
        "filters": [],
        "timeWindowDays": 7,
    }

    plan = QuestionUnderstandingCompiler().compile("最近7天订单量是多少？", understanding, pack)

    trend = next((intent for intent in plan.intents if intent.group_by_column == "pt"), None)
    assert trend is not None
    assert trend.answer_mode == AnswerMode.GROUP_AGG
    assert trend.metric_name == "order_detail_cnt"
    assert str((trend.metric_resolution or {}).get("visualization")) == "line_chart"
    assert any(item.startswith("DEFAULT_TREND_CONTEXT:") for item in plan.compiler_trace)


def test_compiler_keeps_top_entity_requested_metric_dependent():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "spu_id", "sub_order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "spu_id", "refund_id", "pt"],
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="下单量",
                metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            ),
            PlanningAssetEntry(
                key="refund_bill_cnt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_id"],
                title="退款量",
                metadata={"sourceColumns": ["refund_id"], "formula": "COUNT(DISTINCT refund_id)"},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_spu",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            )
        ],
    )
    understanding = {
        "analysisGrain": "product",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "下单量",
            "objectiveType": "ranking",
            "groupByColumn": "spu_id",
            "order": "desc",
            "limit": 3,
        },
        "requestedMeasures": [
            {"metricRef": "refund_bill_cnt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款量"}
        ],
        "scopeConstraints": [],
        "filters": [],
        "timeWindowDays": 90,
    }

    plan = QuestionUnderstandingCompiler().compile("查询前三 SPU 下单量，同时看下退货量", understanding, pack)
    refund_node = next(intent for intent in plan.intents if intent.metric_name == "refund_bill_cnt")

    assert refund_node.task_role == TaskRole.DEPENDENT
    assert refund_node.depends_on_task_ids
    assert any(dep.dependent_task_id == refund_node.plan_task_id for dep in plan.dependencies)
    assert any(item.startswith("GRAPH_ROLE:%s:dependent_metric" % refund_node.plan_task_id) for item in plan.compiler_trace)
    assert PlannerReflectionAgent().reflect("查询前三 SPU 下单量，同时看下退货量", plan, pack).passed


def test_planner_reflection_flags_sibling_metric_attached_as_dependency():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="dwm_trade_refund_detail_di", columns=["seller_id", "pay_amt", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="order_detail_cnt", table="dwm_trade_order_detail_di", columns=["sub_order_id"]),
            PlanningAssetEntry(key="pay_amt", table="dwm_trade_refund_detail_di", columns=["pay_amt"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "订单量",
            },
            "requestedMeasures": [
                {"metricRef": "pay_amt", "ownerTable": "dwm_trade_refund_detail_di", "sourcePhrase": "退款金额"}
            ],
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                group_by_column="seller_id",
                output_keys=["seller_id"],
            ),
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="refund_metric",
                task_role=TaskRole.DEPENDENT,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                group_by_column="seller_id",
                output_keys=["seller_id"],
                depends_on_task_ids=["anchor_order"],
            ),
        ],
        dependencies=[
            PlanDependency(
                anchor_task_id="anchor_order",
                dependent_task_id="refund_metric",
                join_key="seller_id",
                anchor_column="seller_id",
                dependent_column="seller_id",
                relation_type="LOOKUP",
            )
        ],
        compiler_trace=[
            "GRAPH_ROLE:anchor_order:primary_root:dwm_trade_order_detail_di.order_detail_cnt",
            "GRAPH_ROLE:refund_metric:sibling_metric:dwm_trade_refund_detail_di.pay_amt",
            "DEPENDENCY_SEMANTICS:refund_metric:parallel_evidence",
        ],
    )

    reflection = PlannerReflectionAgent().reflect("最近30天订单量和退款金额分别是多少", plan, pack)
    codes = {issue["code"] for issue in reflection.issues}

    assert "SIBLING_METRIC_WRONGLY_DEPENDENT" in codes
    assert "FAKE_DEPENDENCY" in codes
    assert reflection.repair_reason == "ANCHOR_MISMATCH"


def test_planner_reflection_flags_generic_root_when_recalled_measure_is_more_specific():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="ads_merchant_profile", columns=["merchant_id", "ship_timeout_order_cnt_1d", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="order_detail_cnt", table="dwm_trade_order_detail_di", columns=["sub_order_id"]),
            PlanningAssetEntry(key="ship_timeout_order_cnt_1d", table="ads_merchant_profile", columns=["ship_timeout_order_cnt_1d"]),
        ],
    )
    plan = QueryPlan(
        question_understanding={
            "rankingObjective": {
                "metricRef": "order_detail_cnt",
                "ownerTable": "dwm_trade_order_detail_di",
                "sourcePhrase": "订单量",
            },
            "requestedMeasures": [
                {
                    "metricRef": "ship_timeout_order_cnt_1d",
                    "ownerTable": "ads_merchant_profile",
                    "sourcePhrase": "发货超时订单量",
                    "completionSource": "recalled_metric_evidence",
                    "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
                }
            ],
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                output_keys=["seller_id"],
            ),
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="ship_timeout_metric",
                task_role=TaskRole.ANCHOR,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
                output_keys=["merchant_id"],
            ),
        ],
        compiler_trace=[
            "GRAPH_ROLE:anchor_order:primary_root:dwm_trade_order_detail_di.order_detail_cnt",
            "GRAPH_ROLE:ship_timeout_metric:sibling_metric:ads_merchant_profile.ship_timeout_order_cnt_1d",
            "DEPENDENCY_SEMANTICS:ship_timeout_metric:parallel_evidence",
        ],
    )

    reflection = PlannerReflectionAgent().reflect("平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。", plan, pack)

    assert any(issue["code"] == "ROOT_METRIC_NOT_MOST_SPECIFIC" for issue in reflection.issues)
    assert reflection.repair_reason == "ANCHOR_MISMATCH"


def test_repair_promotes_recalled_specific_metric_to_primary_root():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "sub_order_id", "pt"]),
            PlanningAssetEntry(table="ads_merchant_profile", columns=["merchant_id", "ship_timeout_order_cnt_1d", "pt"]),
        ],
        metrics=[
            PlanningAssetEntry(key="order_detail_cnt", table="dwm_trade_order_detail_di", columns=["sub_order_id"]),
            PlanningAssetEntry(
                key="ship_timeout_order_cnt_1d",
                table="ads_merchant_profile",
                columns=["ship_timeout_order_cnt_1d"],
                metadata={
                    "sourceColumns": ["ship_timeout_order_cnt_1d"],
                    "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
                },
            ),
        ],
    )
    understanding = {
        "analysisGrain": "merchant",
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "订单量",
            "objectiveType": "metric_total",
        },
        "requestedMeasures": [
            {
                "metricRef": "ship_timeout_order_cnt_1d",
                "ownerTable": "ads_merchant_profile",
                "sourcePhrase": "发货超时订单量",
                "completionSource": "recalled_metric_evidence",
                "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
            }
        ],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "shipping_timeout_order_population",
                "requiredLevel": "required",
                "suggestedDomains": ["order"],
                "suggestedMetricRefs": ["order_detail_cnt"],
            }
        ],
        "timeWindowDays": 7,
    }
    plan = QueryPlan(
        question_understanding=understanding,
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="anchor_order",
                task_role=TaskRole.ANCHOR,
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                output_keys=["seller_id"],
            ),
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="ship_timeout_metric",
                task_role=TaskRole.ANCHOR,
                preferred_table="ads_merchant_profile",
                metric_name="ship_timeout_order_cnt_1d",
                output_keys=["merchant_id"],
            ),
        ],
        compiler_trace=[
            "GRAPH_ROLE:anchor_order:primary_root:dwm_trade_order_detail_di.order_detail_cnt",
            "GRAPH_ROLE:ship_timeout_metric:sibling_metric:ads_merchant_profile.ship_timeout_order_cnt_1d",
            "DEPENDENCY_SEMANTICS:ship_timeout_metric:parallel_evidence",
        ],
    )

    repaired = repair_more_specific_root_metric(
        "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
        plan,
        pack,
        QuestionUnderstandingCompiler(),
    )

    ranking = repaired.question_understanding["rankingObjective"]
    assert ranking["metricRef"] == "ship_timeout_order_cnt_1d"
    assert ranking["ownerTable"] == "ads_merchant_profile"
    assert not repaired.question_understanding.get("requestedMeasures")
    assert repaired.question_understanding["requiredEvidenceIntents"][0]["suggestedMetricRefs"] == ["ship_timeout_order_cnt_1d"]
    assert repaired.intents[0].metric_name == "ship_timeout_order_cnt_1d"
    assert repaired.intents[0].preferred_table == "ads_merchant_profile"
    assert any(item.startswith("REPAIR_PROMOTE_ROOT_METRIC:") for item in repaired.compiler_trace)
    reflection = PlannerReflectionAgent().reflect(
        "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。",
        repaired,
        pack,
    )
    assert "ROOT_METRIC_NOT_MOST_SPECIFIC" not in {issue["code"] for issue in reflection.issues}


def test_compiler_adds_parallel_rule_evidence_branch_for_rule_data_question():
    question = "平台规则说发货超时要注意什么，同时看最近7天发货超时订单量。"
    pack = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="ads_merchant_profile", columns=["merchant_id", "pt", "ship_timeout_order_cnt_1d"])],
        metrics=[
            PlanningAssetEntry(
                key="ship_timeout_order_cnt_1d",
                table="ads_merchant_profile",
                columns=["ship_timeout_order_cnt_1d"],
                metadata={
                    "sourceColumns": ["ship_timeout_order_cnt_1d"],
                    "semanticRefId": "semantic:profile:ads_merchant_profile:metric:ship_timeout_order_cnt_1d",
                },
            )
        ],
        source_refs={
            "rule:shipping_timeout": RecallItem(
                doc_id="rule:shipping_timeout",
                title="平台发货规则",
                content="发货超时需要关注履约时效和物流信息。",
                source_type="RULE",
                answer_mode="RULE",
                fusion_score=5.0,
            )
        },
    )
    understanding = {
        "analysisGrain": "merchant",
        "analysisIntent": "overview",
        "requiresExplanation": True,
        "rankingObjective": {
            "metricRef": "ship_timeout_order_cnt_1d",
            "ownerTable": "ads_merchant_profile",
            "sourcePhrase": "发货超时订单量",
            "objectiveType": "metric_total",
            "groupByColumn": "merchant_id",
            "limit": 1,
        },
        "requestedMeasures": [],
        "requiredEvidenceIntents": [
            {
                "semanticLabel": "shipping_timeout_explanation",
                "requiredLevel": "required",
                "suggestedDomains": ["rule", "order"],
            }
        ],
        "timeWindowDays": 7,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)
    reflection = PlannerReflectionAgent().reflect(question, plan, pack)

    data_intents = [
        intent
        for intent in plan.intents
        if intent.answer_mode != AnswerMode.RULE and (intent.metric_resolution or {}).get("displayRole") != "trend_context"
    ]
    trend_intents = [intent for intent in plan.intents if (intent.metric_resolution or {}).get("displayRole") == "trend_context"]
    rule_intents = [intent for intent in plan.intents if intent.answer_mode == AnswerMode.RULE]
    assert len(data_intents) == 1
    assert data_intents[0].metric_name == "ship_timeout_order_cnt_1d"
    assert trend_intents
    assert len(rule_intents) == 1
    assert rule_intents[0].knowledge_ref_ids == ["rule:shipping_timeout"]
    assert not plan.dependencies
    assert any(item.startswith("RULE_EVIDENCE_BRANCH:") for item in plan.compiler_trace)
    assert "ANALYSIS_EVIDENCE_NOT_COVERED" not in {issue["code"] for issue in reflection.issues}
    assert "MISSING_KNOWLEDGE_REF" not in {issue["code"] for issue in reflection.issues}


def test_compiler_chains_multiple_scope_constraints_before_requested_measures():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_cs_ticket_detail_di",
                columns=["seller_id", "ticket_id", "sub_order_id", "order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "sub_order_id", "order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=["seller_id", "refund_id", "sub_order_id", "order_id", "pt"],
            ),
            PlanningAssetEntry(
                table="dwm_cs_repay_detail_df",
                columns=["seller_id", "bill_id", "sub_order_id", "order_id", "pt"],
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="ticket_cnt",
                table="dwm_cs_ticket_detail_di",
                columns=["ticket_id"],
                title="工单量",
                metadata={"sourceColumns": ["ticket_id"], "formula": "COUNT(DISTINCT ticket_id)"},
            ),
            PlanningAssetEntry(
                key="order_detail_cnt",
                table="dwm_trade_order_detail_di",
                columns=["sub_order_id"],
                title="订单量",
                metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
            ),
            PlanningAssetEntry(
                key="refund_bill_cnt",
                table="dwm_trade_refund_detail_di",
                columns=["refund_id"],
                title="退款量",
                metadata={"sourceColumns": ["refund_id"], "formula": "COUNT(DISTINCT refund_id)"},
            ),
            PlanningAssetEntry(
                key="repay_bill_cnt",
                table="dwm_cs_repay_detail_df",
                columns=["bill_id"],
                title="赔付单量",
                metadata={"sourceColumns": ["bill_id"], "formula": "COUNT(DISTINCT bill_id)"},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="order_ticket_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_ticket_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
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
            RelationshipEntry(
                relationship_id="order_repay_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_cs_repay_detail_df",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            ),
        ],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "rankingObjective": {
            "metricRef": "order_detail_cnt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "这些订单",
            "groupByColumn": "seller_id",
            "objectiveType": "metric_total",
            "limit": 1,
        },
        "requestedMeasures": [
            {"metricRef": "repay_bill_cnt", "ownerTable": "dwm_cs_repay_detail_df", "sourcePhrase": "是否发生赔付"}
        ],
        "scopeConstraints": [
            {
                "scopeId": "ticket_orders",
                "sourcePhrase": "客服工单里",
                "ownerTable": "dwm_cs_ticket_detail_di",
                "metricRef": "ticket_cnt",
                "entityGrain": "order",
                "targetDomain": "ticket",
                "required": True,
            },
            {
                "scopeId": "refund_orders",
                "sourcePhrase": "涉及退款",
                "ownerTable": "dwm_trade_refund_detail_di",
                "metricRef": "refund_bill_cnt",
                "entityGrain": "order",
                "targetDomain": "refund",
                "required": True,
            },
        ],
        "filters": [],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile(
        "最近30天客服工单里涉及退款的订单，同时看这些订单是否发生了赔付。",
        understanding,
        pack,
    )
    validation = QueryGraphValidator().validate(
        "最近30天客服工单里涉及退款的订单，同时看这些订单是否发生了赔付。",
        plan,
        pack,
    )

    task_ids = {intent.plan_task_id for intent in plan.intents}
    assert "anchor_ticket_scope" in task_ids
    assert "refund_scope" in task_ids
    assert dependency_path_exists_for_test(plan.dependencies, "anchor_ticket_scope", "refund_scope")
    repay = next(intent for intent in plan.intents if intent.metric_name == "repay_bill_cnt")
    assert dependency_path_exists_for_test(plan.dependencies, "refund_scope", repay.plan_task_id)
    assert not any(gap.code == "SCOPE_NOT_NARROWING" for gap in validation.gaps)
    assert validation.valid, [(gap.code, gap.evidence) for gap in validation.gaps]


def dependency_path_exists_for_test(dependencies: list[PlanDependency], source: str, target: str) -> bool:
    adjacency: dict[str, list[str]] = {}
    for dep in dependencies:
        adjacency.setdefault(dep.anchor_task_id, []).append(dep.dependent_task_id)
    pending = list(adjacency.get(source, []))
    seen: set[str] = set()
    while pending:
        task = pending.pop()
        if task == target:
            return True
        if task in seen:
            continue
        seen.add(task)
        pending.extend(adjacency.get(task, []))
    return False


def test_compiler_corrects_metric_semantic_mismatch_from_source_phrase():
    settings = get_settings()
    question = "最近30天GMV最高的前5天，同时看退款金额、赔付金额、工单量，判断是否存在异常波动。"
    topic_assets = TopicAssetService(settings)
    builder = PlanningAssetPackBuilder(topic_assets, SkillLoader(settings))
    recall_service = HybridRecallService(settings, topic_assets, WikiMemoryService(settings))
    recall = recall_service.recall(
        question,
        KeywordExtractService().extract(question),
        [],
        "",
        "100",
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )
    pack = builder.compact(
        question,
        recall,
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.CS_TICKET, QuestionCategory.COMPENSATION],
    )
    understanding = {
        "analysisGrain": "day",
        "analysisIntent": "anomaly_check",
        "rankingObjective": {
            "metricRef": "pay_amt",
            "ownerTable": "dwm_trade_order_detail_di",
            "sourcePhrase": "GMV",
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
    assert not plan.knowledge_requests
    assert plan.intents[0].preferred_table == "ads_merchant_profile"
    assert plan.intents[0].metric_name == "order_gmv_amt_1d"
    assert plan.intents[0].metric_resolution["resolutionSource"] == "semantic_recall_evidence"
    assert plan.intents[0].metric_resolution["metricEvidenceCandidates"][0]["metricKey"] == "order_gmv_amt_1d"
    assert any(item.startswith("METRIC_SEMANTIC_MISMATCH:") for item in plan.compiler_trace)


def test_compiler_binds_status_filter_to_matching_node_table():
    settings = get_settings()
    question = "最近30天退款状态为处理中或异常的订单，关联看商品发布时间和订单金额。"
    pack = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings)).compact(
        question,
        recall_bundle_empty(),
        [QuestionCategory.TRADE, QuestionCategory.REFUND, QuestionCategory.GOODS],
    )
    understanding = {
        "analysisGrain": "order",
        "analysisIntent": "none",
        "requiresExplanation": False,
        "requiredEvidenceIntents": [],
        "rankingObjective": {
            "metricRef": "refund_bill_cnt",
            "ownerTable": "dwm_trade_refund_detail_di",
            "sourcePhrase": "退款状态为处理中或异常的订单",
            "objectiveType": "detail_anchor",
            "groupByColumn": "sub_order_id",
            "order": "desc",
            "limit": 100,
        },
        "requestedMeasures": [],
        "filters": [{"field": "refund_status_name", "value": "处理中,异常"}],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile(question, understanding, pack)

    assert plan.intents
    anchor = plan.intents[0]
    assert anchor.filter_column == "refund_status_name"
    assert anchor.filter_value == "处理中,异常"
    assert "refund_status_name" in anchor.required_evidence
    assert any(item.startswith("FILTER_BOUND:anchor_refund:refund_status_name=") for item in plan.compiler_trace)


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


def test_materialized_profile_rate_metric_uses_standard_field_not_formula_components():
    pack = profile_daily_pack()
    pack.tables[0].columns.extend(["refund_rate_1d", "return_cnt_1d", "pay_order_cnt_1d"])
    pack.metrics.extend(
        [
            PlanningAssetEntry(
                key="return_cnt_1d",
                table="ads_merchant_profile",
                columns=["return_cnt_1d"],
                title="退货量",
                metadata={"sourceColumns": ["return_cnt_1d"], "formula": "SUM(return_cnt_1d)"},
            ),
            PlanningAssetEntry(
                key="pay_order_cnt_1d",
                table="ads_merchant_profile",
                columns=["pay_order_cnt_1d"],
                title="支付订单量",
                metadata={"sourceColumns": ["pay_order_cnt_1d"], "formula": "SUM(pay_order_cnt_1d)"},
            ),
            PlanningAssetEntry(
                key="refund_rate_1d",
                table="ads_merchant_profile",
                columns=["refund_rate_1d"],
                title="退货量占支付订单量比例",
                aliases=["退货率", "退款率", "退货比例"],
                metadata={
                    "sourceColumns": ["return_cnt_1d", "pay_order_cnt_1d"],
                    "formula": "return_cnt_1d / pay_order_cnt_1d",
                    "unit": "%",
                },
            ),
        ]
    )
    understanding = {
        "analysisGrain": "day",
        "rankingObjective": {
            "metricRef": "refund_rate_1d",
            "ownerTable": "ads_merchant_profile",
            "sourcePhrase": "退货比例",
            "groupByColumn": "pt",
            "order": "desc",
            "limit": 30,
        },
        "requestedMeasures": [],
        "filters": [],
        "timeWindowDays": 30,
    }

    plan = QuestionUnderstandingCompiler().compile("最近30天退货比例走势", understanding, pack)

    intent = next(intent for intent in plan.intents if intent.metric_name == "refund_rate_1d")
    assert intent.preferred_table == "ads_merchant_profile"
    assert intent.answer_mode != AnswerMode.DERIVED
    assert intent.metric_column == "refund_rate_1d"
    assert intent.metric_formula == "AVG(`refund_rate_1d`)"
    assert "return_cnt_1d" not in intent.required_evidence
    assert "pay_order_cnt_1d" not in intent.required_evidence
    assert not any(dep.relation_type == "DERIVED_COMPONENT" for dep in plan.dependencies)

    worker = NodeWorkerExecutor(FakeLlm(), FakeDoris(), SqlValidationService(), get_settings())
    sql = worker._draft_sql(intent, pack, "", NodeExecutionContext(merchant_id="100"))
    assert "AVG(`refund_rate_1d`) AS `refund_rate_1d`" in sql
    assert "return_cnt_1d / pay_order_cnt_1d" not in sql
    assert "return_cnt_1d" not in sql
    assert "pay_order_cnt_1d" not in sql


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
    assert "证据门禁" not in answer
    assert "说明" in answer
    assert "退款金额" in answer
    assert "dwm_trade_refund_detail_di" not in answer
    assert "pay_amt" not in answer


def test_answer_appends_lightweight_metric_disclosure_for_core_metric():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近30天GMV是多少？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="metric_gmv",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                metric_resolution={
                    "metricKey": "order_gmv_amt_1d",
                    "displayName": "GMV",
                    "formula": "SUM(order_gmv_amt_1d)",
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"seller_id": "100", "order_gmv_amt_1d": 1200}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_gmv", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = AnswerComposeService(LlmClient(get_settings())).compose(
        "最近30天GMV是多少？",
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        allow_llm=False,
    )

    assert "统计说明：" in answer
    assert "GMV按支付成功订单金额统计" in answer
    assert "时间为最近30天" in answer
    assert "范围为当前店铺" in answer


def test_answer_reconciliation_mode_for_backend_metric_mismatch():
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="为什么和后台看板 GMV 不一致？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="metric_gmv",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_column="order_gmv_amt_1d",
                metric_resolution={
                    "metricKey": "order_gmv_amt_1d",
                    "displayName": "GMV",
                    "formula": "SUM(order_gmv_amt_1d)",
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[{"seller_id": "100", "order_gmv_amt_1d": 1200}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="metric_gmv", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = AnswerComposeService(LlmClient(get_settings())).compose(
        "为什么和后台看板 GMV 不一致？",
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        allow_llm=False,
    )

    assert "口径对账" in answer
    assert "时间口径" in answer
    assert "状态口径" in answer
    assert "GMV 是否扣退款" in answer
    assert "商品粒度" in answer
    assert "数据更新" in answer


def test_answer_analysis_summary_uses_structured_analysis_intent_not_question_terms():
    class FakeAnalysisLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0
            self.payload = {}

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            self.payload = json.loads(user_prompt)
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
    assert "GMV" in summary
    assert "分析结论" not in summary
    assert llm.calls == 0


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


def test_answer_compose_uses_llm_for_plain_query_with_rows():
    class FakeAnswerLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            assert "不要输出 markdown 表格" in system_prompt
            return "最近30天退款金额最高的商品是 A，退款金额为 12.3。\n\n建议：\n- 优先查看该商品退款原因，确认是否和描述或履约有关。"

    llm = FakeAnswerLlm()
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["dwm_trade_refund_detail_di"],
            rows=[{"seller_id": "100", "spu_name": "A", "pay_amt": 12.3}],
        ),
        task_results=[
            AgentTaskResult(
                task_id="refund_top",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_refund_detail_di"],
                    rows=[{"seller_id": "100", "spu_name": "A", "pay_amt": 12.3}],
                ),
            )
        ],
    )
    plan = QueryPlan(
        question_understanding={"analysisIntent": "none", "requiresExplanation": False},
        intents=[
            QuestionIntent(
                question="最近30天退款金额最高的商品",
                intent_type="VALID",
                answer_mode=AnswerMode.TOPN,
                category=QuestionCategory.REFUND,
                preferred_table="dwm_trade_refund_detail_di",
                metric_name="pay_amt",
                metric_column="pay_amt",
                group_by_column="spu_name",
            )
        ],
    )
    answer = AnswerComposeService(llm).compose("最近30天退款金额最高的商品", MerchantInfo(merchant_id="100"), plan, run, "")
    assert "退款金额最高的商品是 A" in answer
    assert "已按当前口径" not in answer
    assert llm.calls == 1


def test_answer_compose_passes_profile_memory_and_data_to_llm_advice():
    class FakeAnswerLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0
            self.payload = {}

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            self.payload = json.loads(user_prompt)
            assert "businessContext" in user_prompt
            return "\n".join(
                [
                    "最近7天，订单量为 17，退款金额为 130元，咨询工单量为 5。",
                    "",
                    "建议：",
                    "- 结合近期售后关注，优先排查高退款商品和对应工单。",
                    "- 若订单量集中在少数商品，检查商品描述和履约承诺是否一致。",
                ]
            )

    question = "最近7天订单量、退款金额、咨询工单量分别是多少？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="order_metric",
                metric_name="order_detail_cnt",
                metric_resolution={"metricKey": "order_detail_cnt", "displayName": "订单量"},
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_metric",
                metric_name="pay_amt",
                metric_resolution={"metricKey": "pay_amt", "displayName": "退款金额"},
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.CS_TICKET,
                plan_task_id="ticket_metric",
                metric_name="cs_ticket_cnt_1d",
                metric_resolution={"metricKey": "cs_ticket_cnt_1d", "displayName": "咨询工单量"},
            ),
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="order_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_detail_cnt": 17}])),
            AgentTaskResult(task_id="refund_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "pay_amt": 130}])),
            AgentTaskResult(task_id="ticket_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "cs_ticket_cnt_1d": 5}])),
        ],
        merged_query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_detail_cnt": 17, "pay_amt": 130, "cs_ticket_cnt_1d": 5}]),
        verified_evidence=VerifiedEvidence(passed=True),
    )
    merchant = MerchantInfo(merchant_id="100", rows={"merchant_type_name": "认证商户", "is_unconditional_refund": 1})
    personalization_context = {
        "memoryContext": "近期经营记忆：该商家最近持续关注退款率、客服工单和售后风险。",
        "memoryInjection": {
            "recentFocus": {"summary": "近期重点关注退款率和客服工单", "topMetrics": ["退款率", "咨询工单量"]},
            "relevantPreferences": [{"memoryType": "metric_habit", "summary": "常看最近7天售后指标", "confidence": 0.86}],
        },
    }

    llm = FakeAnswerLlm()
    answer = AnswerComposeService(llm).compose(
        question,
        merchant,
        plan,
        run,
        "",
        personalization_context=personalization_context,
    )

    business_context = llm.payload["businessContext"]
    assert "认证商户" in business_context["merchantProfile"]
    assert "退款率" in business_context["memorySummary"]
    assert any(item["label"] == "订单量" and item["value"] == "17" for item in business_context["currentDataSignals"])
    assert "结合近期售后关注" in answer
    assert "继续追问" not in answer
    assert llm.calls == 1


def test_answer_compose_uses_analysis_summary_without_second_answer_llm_call():
    class FakeAnswerLlm:
        configured = True
        settings = get_settings()

        def __init__(self):
            self.calls = 0
            self.payload = {}

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            self.calls += 1
            return "不应该调用最终 Answer LLM"

    question = "最近7天GMV趋势"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_resolution={"metricKey": "order_gmv_amt_1d", "displayName": "GMV"},
                group_by_column="pt",
            )
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="gmv_trend",
                success=True,
                query_bundle=QueryBundle(
                    rows=[
                        {"pt": "2026-06-23", "order_gmv_amt_1d": 990},
                        {"pt": "2026-06-24", "order_gmv_amt_1d": 782.5},
                    ]
                ),
            )
        ],
        merged_query_bundle=QueryBundle(
            rows=[
                {"pt": "2026-06-23", "order_gmv_amt_1d": 990},
                {"pt": "2026-06-24", "order_gmv_amt_1d": 782.5},
            ]
        ),
        verified_evidence=VerifiedEvidence(passed=True),
    )
    bad_draft = "\n".join(
        [
            "分析结论：",
            "- 当前证据显示存在可解释的波动点，不能简单判断为业务为 0 或无异常。",
            "",
            "关键证据：",
            "- order_gmv_amt_1d 期初到期末下降。",
            "",
            "限制：",
            "- 可用行数较少，异常判断可信度有限。",
        ]
    )

    llm = FakeAnswerLlm()
    answer = AnswerComposeService(llm).compose(
        question,
        MerchantInfo(merchant_id="100"),
        plan,
        run,
        "",
        analysis_summary=bad_draft,
    )

    assert "这几天有波动" in answer
    assert "GMV 期初到期末下降" in answer
    assert "分析结论" not in answer
    assert "关键证据" not in answer
    assert "order_gmv_amt_1d" not in answer
    assert "限制" not in answer
    assert "建议结合下方趋势" in answer
    assert llm.calls == 0


def test_answer_sections_convert_daily_metric_rows_to_chart_series():
    settings = get_settings()
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="anchor_order",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                metric_resolution={"metricKey": "order_detail_cnt", "displayName": "订单量"},
            ),
            QuestionIntent(
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="trend_order_order_detail_cnt",
                preferred_table="dwm_trade_order_detail_di",
                metric_name="order_detail_cnt",
                group_by_column="pt",
                metric_resolution={"metricKey": "order_detail_cnt", "displayName": "订单量", "visualization": "line_chart"},
            ),
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="anchor_order",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[{"seller_id": "100", "order_detail_cnt": 17}],
                    original_row_count=1,
                ),
            ),
            AgentTaskResult(
                task_id="trend_order_order_detail_cnt",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_trade_order_detail_di"],
                    rows=[
                        {"seller_id": "100", "pt": "2026-06-23", "order_detail_cnt": 3},
                        {"seller_id": "100", "pt": "2026-06-24", "order_detail_cnt": 5},
                    ],
                    original_row_count=2,
                ),
            ),
        ]
    )

    sections = service.build_sections(plan, run)

    trend_section = next(section for section in sections if section.title == "订单量趋势")
    assert trend_section.data_rows == [
        {"metric_name": "订单量", "pt": "2026-06-23", "value": 3.0},
        {"metric_name": "订单量", "pt": "2026-06-24", "value": 5.0},
    ]


def test_contextual_suggestions_use_question_profile_memory_and_result():
    service = AnswerComposeService(LlmClient(get_settings()))
    intents = [
        QuestionIntent(
            intent_type="VALID",
            answer_mode=AnswerMode.METRIC,
            category=QuestionCategory.TRADE,
            plan_task_id="order",
            metric_name="order_detail_cnt",
            metric_resolution={"metricKey": "order_detail_cnt", "displayName": "订单量"},
        ),
        QuestionIntent(
            intent_type="VALID",
            answer_mode=AnswerMode.METRIC,
            category=QuestionCategory.REFUND,
            plan_task_id="refund",
            metric_name="refund_amt_1d",
            metric_resolution={"metricKey": "refund_amt_1d", "displayName": "退款金额"},
        ),
        QuestionIntent(
            intent_type="VALID",
            answer_mode=AnswerMode.METRIC,
            category=QuestionCategory.CS_TICKET,
            plan_task_id="ticket",
            metric_name="cs_ticket_cnt_1d",
            metric_resolution={"metricKey": "cs_ticket_cnt_1d", "displayName": "咨询工单量"},
        ),
    ]
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="order", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_detail_cnt": 17}])),
            AgentTaskResult(task_id="refund", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "refund_amt_1d": 130}])),
            AgentTaskResult(task_id="ticket", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "cs_ticket_cnt_1d": 5}])),
        ]
    )
    merchant = MerchantInfo(merchant_id="100", rows={"is_unconditional_refund": 1, "merchant_type_name": "企业商户"})
    personalization_context = {
        "memoryContext": "近期持续关注退款、客服工单和七天无理由售后风险。",
        "memoryInjection": {
            "recentFocus": {"topTopics": ["电商退货", "客服工单"], "topMetrics": ["退款金额", "咨询工单量"]}
        },
    }

    suggestions = service.contextual_suggestions(
        "最近7天订单量、退款金额、咨询工单量分别是多少？",
        intents,
        run_result=run,
        merchant=merchant,
        personalization_context=personalization_context,
    )

    assert suggestions[:3] == [
        "按商品拆解订单量、退款金额和工单量",
        "退款金额最高的商品是否也带来更多工单？",
        "工单最多的问题类型和订单状态是什么？",
    ]
    assert "我想查看保证金" not in suggestions[:3]


def test_contextual_suggestions_for_gmv_trend_prioritize_trade_drilldowns():
    service = AnswerComposeService(LlmClient(get_settings()))
    intents = [
        QuestionIntent(
            intent_type="VALID",
            answer_mode=AnswerMode.GROUP_AGG,
            category=QuestionCategory.TRADE,
            plan_task_id="gmv_trend",
            metric_name="order_gmv_amt_1d",
            group_by_column="pt",
            metric_resolution={"metricKey": "order_gmv_amt_1d", "displayName": "GMV"},
        )
    ]

    suggestions = service.contextual_suggestions(
        "最近7天GMV趋势",
        intents,
        run_result=AgentRunResult(
            task_results=[
                AgentTaskResult(
                    task_id="gmv_trend",
                    success=True,
                    query_bundle=QueryBundle(
                        rows=[
                            {"pt": "2026-06-23", "order_gmv_amt_1d": 990},
                            {"pt": "2026-06-24", "order_gmv_amt_1d": 782.5},
                        ]
                    ),
                )
            ]
        ),
        merchant=MerchantInfo(merchant_id="100", rows={"merchant_type_name": "企业商户"}),
        personalization_context={"memoryContext": "近期关注 GMV 下滑和退款影响。"},
    )

    assert suggestions[0] == "GMV变化主要来自订单量还是客单价？"
    assert any("退款金额" in item for item in suggestions[:4])
    assert "我想查看保证金" not in suggestions[:3]


def test_merchant_experience_package_surfaces_ux_helpers():
    service = AnswerComposeService(LlmClient(get_settings()))
    question = "最近7天GMV趋势"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                metric_name="order_gmv_amt_1d",
                group_by_column="pt",
                metric_resolution={
                    "metricKey": "order_gmv_amt_1d",
                    "displayName": "GMV",
                    "formula": "SUM(order_gmv_amt_1d)",
                    "sourceColumns": ["order_gmv_amt_1d"],
                },
            )
        ]
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[
            {"merchant_id": "100", "pt": "2026-07-01", "order_gmv_amt_1d": 100},
            {"merchant_id": "100", "pt": "2026-07-07", "order_gmv_amt_1d": 60},
        ],
        original_row_count=2,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_trend", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    sections = service.build_sections(plan, run)
    suggestions = service.contextual_suggestions(question, plan.intents, run_result=run, merchant=MerchantInfo(merchant_id="100"))

    package = service.merchant_experience(
        question,
        plan,
        run,
        merchant=MerchantInfo(merchant_id="100", merchant_name="测试商家"),
        sections=sections,
        suggestions=suggestions,
    )

    assert package["businessAdvice"]
    assert package["suggestedQuestions"]
    assert package["anomalyAlerts"][0]["metric"] == "GMV"
    assert package["metricDisclosures"][0]["displayName"] == "GMV"
    assert package["traceability"]["merchantId"] == "100"
    assert package["traceability"]["dataUpdatedAt"] == "2026-07-07"
    assert any(action["label"] == "拆解成交变化" for action in package["drillDownActions"])
    assert package["reportSubscriptionHint"]["enabled"] is True
    assert isinstance(package["clarificationHints"], list)


def test_response_includes_csv_download_for_large_results(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "result_csv_download_min_rows": 3,
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "导出最近7天GMV明细",
        "100",
        ChatContext(),
        None,
        "thread_csv_download",
        "run_csv_download",
    )
    state["answer"] = "结果较多，已提供 CSV 下载。"
    state["persisted"] = False
    state["response_context"] = ChatContext()
    state["plan"] = QueryPlan(
        intents=[
            QuestionIntent(
                question="导出最近7天GMV明细",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_detail",
                preferred_table="ads_merchant_profile",
                output_keys=["merchant_id", "pt", "order_gmv_amt_1d"],
            )
        ]
    )
    rows = [
        {"merchant_id": "100", "pt": "2026-07-01", "order_gmv_amt_1d": 100},
        {"merchant_id": "100", "pt": "2026-07-02", "order_gmv_amt_1d": 120},
        {"merchant_id": "100", "pt": "2026-07-03", "order_gmv_amt_1d": 90},
    ]
    bundle = QueryBundle(tables=["ads_merchant_profile"], rows=rows, original_row_count=len(rows))
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_detail", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    response = workflow.to_response(state)

    downloads = response.merchant_experience.get("downloadArtifacts") or []
    assert downloads
    assert downloads[0]["type"] == "csv"
    assert downloads[0]["rowCount"] == 3
    assert downloads[0]["downloadUrl"].startswith("merchant://")
    csv_path = Path(downloads[0]["path"])
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8-sig")
    assert "merchant_id,pt,order_gmv_amt_1d" in content
    assert "2026-07-03" in content


def test_merchant_profile_summary_combines_memory_focus_and_required_rules():
    service = MerchantProfileSummaryService()

    summary = service.summarize(
        merchant=MerchantInfo(merchant_id="100", merchant_name="测试商家"),
        memory_injection={
            "source": "structured_memory",
            "coreMemory": {
                "recentFocus": {
                    "topMetrics": ["GMV", "退款率"],
                    "topCategories": ["交易", "退款"],
                    "focusPattern": "关注 GMV 下滑和退款异常",
                }
            },
        },
        memory_constraints=[
            {
                "id": "fact_1",
                "type": "business_fact",
                "enforcement": "required",
                "instruction": "统计 GMV 时必须排除测试订单",
                "targetMetrics": ["GMV"],
                "source": "merchant_approved_fact",
            }
        ],
        route_slots=RouteSlots(time_window=RouteTimeWindow(days=7, raw="近7天")),
        fast_understanding=None,
    )

    assert summary["merchantId"] == "100"
    assert summary["defaultTimeWindowDays"] == 7
    assert summary["defaultTimeWindow"] == 7
    assert summary["preferredMetrics"] == ["GMV", "退款率"]
    assert summary["businessFocus"] == ["交易", "退款"]
    assert "退款率升高" in summary["recentRisks"]
    assert summary["confirmedRules"][0]["instruction"] == "统计 GMV 时必须排除测试订单"
    assert summary["confirmedRuleTexts"] == ["统计 GMV 时必须排除测试订单"]


def test_merchant_profile_store_persists_reviews_and_merges_runtime_summary(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path / "workspace")})
    store = MerchantProfileStore(settings)

    profile = store.upsert_profile(
        "100",
        {
            "defaultTimeWindow": 14,
            "preferredMetrics": ["GMV", "退款率"],
            "confirmedRuleTexts": ["有效订单排除已取消订单"],
            "recentRisks": ["退款率升高"],
            "businessFocus": ["售后"],
            "industryTags": ["服饰"],
        },
        reviewer="ops",
        review_status="reviewed",
    )
    approved = store.review_profile("100", approved=True, reviewer="lead", note="ok")
    merged = store.merge_runtime_summary(
        "100",
        {
            "merchantId": "100",
            "defaultTimeWindow": 7,
            "preferredMetrics": ["工单量"],
            "confirmedRules": [{"instruction": "统计 GMV 使用支付成功口径"}],
            "businessFocus": ["履约"],
        },
    )

    assert profile["defaultTimeWindow"] == 14
    assert approved["reviewStatus"] == "approved"
    assert merged["defaultTimeWindow"] == 14
    assert merged["preferredMetrics"][:3] == ["GMV", "退款率", "工单量"]
    assert "有效订单排除已取消订单" in merged["confirmedRuleTexts"]
    assert "退款率升高" in merged["recentRisks"]
    assert merged["industryTags"] == ["服饰"]
    assert merged["profileStore"]["enabled"] is True


def test_response_exposes_profile_freshness_and_security_audit(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "allowed_merchant_ids": "100,101",
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "最近7天GMV是多少",
        "100",
        ChatContext(),
        None,
        "thread_enterprise_surface",
        "run_enterprise_surface",
    )
    state["answer"] = "最近7天 GMV 是 100。"
    state["persisted"] = False
    state["response_context"] = ChatContext()
    state["merchant"] = MerchantInfo(merchant_id="100", merchant_name="测试商家")
    state["plan"] = QueryPlan(
        intents=[
            QuestionIntent(
                question="最近7天GMV是多少",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_metric",
                preferred_table="ads_merchant_profile",
            )
        ]
    )
    bundle = QueryBundle(tables=["ads_merchant_profile"], rows=[{"merchant_id": "100", "gmv": 100}], original_row_count=1)
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_metric", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    state["freshness_reports"] = [
        FreshnessCheckResult(
            task_id="gmv_metric",
            table="ads_merchant_profile",
            checked=True,
            status="AVAILABLE",
            requested_days=7,
            min_pt="20260701",
            max_pt="20260707",
        )
    ]
    state["merchant_profile_summary"] = {
        "merchantId": "100",
        "merchantName": "测试商家",
        "defaultTimeWindowDays": 7,
        "preferredMetrics": ["GMV"],
    }
    state["hypothesis_exploration"] = {
        "mode": "controlled_hypothesis_exploration",
        "hypotheses": [{"hypothesisId": "hyp_1", "title": "GMV 下跌来自交易规模变化"}],
        "budget": {"maxHypotheses": 3, "maxCandidateGraphs": 3},
    }
    state["candidate_query_graphs"] = {
        "mode": "candidate_query_graph_sandbox",
        "selectedCandidateId": "cand_hyp_1",
        "candidates": [{"candidateId": "cand_hyp_1", "status": "selected", "score": 80}],
    }
    state["latency_optimization"] = {
        "mode": "fast_path_verified_graph",
        "eligible": True,
        "reason": "single-node QueryGraph with no dependencies can use fast verified path",
        "skipNodes": ["reflect_query_graph", "run_analysis_skill", "answer_llm"],
        "preservedGuardrails": ["query_graph_validation", "readonly_sql", "evidence_verification"],
    }
    state["skill_lifecycle_records"] = [
        SkillLifecycleRecord(skill_name="gmv_drop_diagnosis", stage="matched", status="matched")
    ]

    response = workflow.to_response(state)

    merchant_experience = response.merchant_experience
    assert merchant_experience["merchantProfileSummary"]["preferredMetrics"] == ["GMV"]
    assert merchant_experience["dataFreshness"]["status"] == "checked"
    assert merchant_experience["dataFreshness"]["reports"][0]["table"] == "ads_merchant_profile"
    assert merchant_experience["dataFreshness"]["latestDataAt"] == "20260707"
    assert merchant_experience["dataFreshness"]["missingDataPolicy"].startswith("missing partition")
    assert merchant_experience["securityAudit"]["policy"] == "readonly_merchant_scoped"
    assert merchant_experience["securityAudit"]["tenantScoped"] is True
    assert merchant_experience["securityAudit"]["allowedMerchantCount"] == 2
    assert merchant_experience["securityAudit"]["rowLevelSecurity"]["enabled"] is True
    assert merchant_experience["securityAudit"]["sqlPolicy"]["readOnlyOnly"] is True
    assert merchant_experience["controlledReact"]["mode"] == "controlled_react_querygraph"
    assert merchant_experience["controlledReact"]["exploration"]["hypotheses"][0]["hypothesisId"] == "hyp_1"
    assert merchant_experience["controlledReact"]["candidateQueryGraphs"]["selectedCandidateId"] == "cand_hyp_1"
    assert merchant_experience["controlledReact"]["latencyOptimization"]["mode"] == "fast_path_verified_graph"
    assert "answer_llm" in merchant_experience["controlledReact"]["latencyOptimization"]["skipNodes"]
    assert "evidence_verification" in merchant_experience["controlledReact"]["guardrails"]
    assert merchant_experience["skillEcosystem"]["creator"]["enabled"] is True
    assert any(item["name"] == "gmv_drop_diagnosis" for item in merchant_experience["skillEcosystem"]["market"]["items"])
    assert merchant_experience["skillEcosystem"]["runtimeRecords"][0]["skillName"] == "gmv_drop_diagnosis"


def test_response_exposes_realtime_fallback_freshness_disclosure(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state("今天订单量是多少", "100", ChatContext(), None, "thread_freshness_fallback", "run_freshness_fallback")
    state["answer"] = "已切换实时表查看今天订单。"
    state["persisted"] = False
    state["response_context"] = ChatContext()
    state["merchant"] = MerchantInfo(merchant_id="100", merchant_name="测试商家")
    state["plan"] = QueryPlan()
    state["agent_run_result"] = AgentRunResult()
    state["freshness_reports"] = [
        FreshnessCheckResult(
            task_id="today_orders",
            table="dwm_trade_order_detail_di",
            checked=True,
            status="STALE_USE_REALTIME_FALLBACK",
            requested_days=1,
            max_pt="20260710",
            fallback_table="dwm_trade_order_realtime_di",
            reason="offline table stale; switch_to_realtime=dwm_trade_order_realtime_di",
        )
    ]

    response = workflow.to_response(state)

    freshness = response.merchant_experience["dataFreshness"]
    assert freshness["status"] == "realtime_fallback_used"
    assert freshness["offlineDelayDetected"] is True
    assert freshness["realtimeFallbackUsed"] is True
    assert freshness["answerDisclosure"]["required"] is True
    assert "不能把缺失数据直接解释为业务为 0" in "".join(freshness["notes"])


def test_access_control_denies_columns_and_writes_audit(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path / "workspace")})
    service = AccessControlService(settings)
    service.policy_path.parent.mkdir(parents=True, exist_ok=True)
    service.policy_path.write_text(
        json.dumps(
            {
                "allowedMerchantIds": ["100"],
                "tables": {
                    "ads_merchant_profile": {
                        "allowedRoles": ["merchant_analyst"],
                        "columns": {
                            "mobile": {"denied": True},
                            "buyer_mobile": {"mask": "partial"},
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    denied_contract = NodePlanContract(
        task_id="profile_sensitive",
        preferred_table="ads_merchant_profile",
        allowed_columns=["merchant_id", "mobile", "buyer_mobile"],
        visible_columns=["merchant_id", "mobile", "buyer_mobile"],
        access_role="merchant_analyst",
        merchant_id="100",
    )

    denied = service.authorize_contract(denied_contract, "SELECT mobile FROM ads_merchant_profile WHERE merchant_id = %s", run_id="run_acl")
    allowed_contract = denied_contract.model_copy(update={"visible_columns": ["merchant_id", "buyer_mobile"]})
    allowed = service.authorize_contract(
        allowed_contract,
        "SELECT buyer_mobile FROM ads_merchant_profile WHERE merchant_id = %s",
        run_id="run_acl",
    )
    audit = service.record_query_audit(allowed, row_count=3, status="success")

    assert denied.allowed is False
    assert denied.code == "COLUMN_DENIED"
    assert "mobile" in denied.denied_columns
    assert allowed.allowed is True
    assert allowed.masked_columns["buyer_mobile"] == "partial"
    assert audit["rowCount"] == 3
    assert service.audit_summary()["items"]


def test_response_exposes_human_loop_confirmation_card(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "帮我看一下店铺表现",
        "100",
        ChatContext(),
        None,
        "thread_human_loop_surface",
        "run_human_loop_surface",
    )
    workflow.request_human_clarification(
        state,
        "你想按哪个时间范围看？",
        "BUSINESS_SCOPE",
        "time_window",
        ["近7天", "昨天", "近30天"],
    )
    state["answer"] = "你想按哪个时间范围看？"
    state["persisted"] = False
    state["response_context"] = ChatContext()
    state["merchant"] = MerchantInfo(merchant_id="100", merchant_name="测试商家")

    response = workflow.to_response(state)

    human_loop = response.merchant_experience["humanLoop"]
    assert human_loop["status"] == "waiting_confirmation"
    assert human_loop["confirmationCard"]["title"] == "确认分析时间范围"
    assert human_loop["checkpoint"]["threadId"] == "thread_human_loop_surface"
    assert human_loop["knowledgeFeedback"]["reviewRequiredBeforePublish"] is True


def test_daily_report_includes_alerts_traceability_and_drilldowns():
    class DailyDoris:
        def query_one(self, sql, params):
            return {
                "merchant_name": "测试商家",
                "order_gmv_amt_1d": 1000,
                "order_user_cnt_1d": 20,
                "order_cnt_1d": 30,
                "trade_success_order_cnt_1d": 25,
                "refund_order_cnt_1d": 3,
                "refund_amt_1d": 120,
            }

    report = DailyReportService(DailyDoris()).report("100")

    assert report.anomaly_alerts
    assert report.drill_down_actions[0]["label"] == "查看退款商品"
    assert report.traceability["sourceTables"] == ["ads_merchant_profile"]
    assert len(report.suggestions) == len(set(report.suggestions))
    assert any("退款" in item for item in report.suggestions)


def test_clarification_prompt_is_merchant_friendly(tmp_path):
    workflow = create_workflow(get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)}))
    state = workflow._initial_state("帮我看一下", "100", ChatContext(), None, "thread_clarify", "run_clarify")

    prompt = workflow.build_scope_clarification_prompt(state)

    assert "最近7天" in prompt
    assert "交易、退款、客服或商品" in prompt


def test_answer_compose_keeps_summary_metric_authoritative_when_trend_is_partial():
    class MisreadingAnswerLlm:
        configured = True
        settings = get_settings()

        def chat(self, system_prompt, user_prompt, fallback="", timeout_seconds=None):
            assert "resultRole=summary" in system_prompt
            return (
                "最近7天咨询工单量是 5 单。\n\n"
                "从已提供的分日情况看：6月24日 3单，6月23日 1单，6月22日 1单。\n"
                "其余日期这里没有看到分日明细。\n"
                "建议优先回看 6月24日工单上升的咨询类型。"
            )

    plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type="VALID",
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.CS_TICKET,
                plan_task_id="anchor_ticket",
                preferred_table="ads_merchant_profile",
                metric_name="cs_ticket_cnt_1d",
                metric_resolution={"metricKey": "cs_ticket_cnt_1d", "displayName": "咨询工单量"},
            ),
            QuestionIntent(
                intent_type="VALID",
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.CS_TICKET,
                plan_task_id="trend_ticket_cs_ticket_cnt_1d",
                preferred_table="ads_merchant_profile",
                metric_name="cs_ticket_cnt_1d",
                group_by_column="pt",
                metric_resolution={"metricKey": "cs_ticket_cnt_1d", "displayName": "咨询工单量", "displayRole": "trend_context"},
            ),
        ]
    )
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(
            tables=["ads_merchant_profile"],
            rows=[
                {"merchant_id": "100", "cs_ticket_cnt_1d": 5},
                {"merchant_id": "100", "pt": "2026-06-22", "cs_ticket_cnt_1d": 1},
                {"merchant_id": "100", "pt": "2026-06-23", "cs_ticket_cnt_1d": 1},
                {"merchant_id": "100", "pt": "2026-06-24", "cs_ticket_cnt_1d": 3},
            ],
        ),
        task_results=[
            AgentTaskResult(
                task_id="anchor_ticket",
                success=True,
                query_bundle=QueryBundle(
                    tables=["ads_merchant_profile"],
                    rows=[{"merchant_id": "100", "cs_ticket_cnt_1d": 5}],
                    original_row_count=1,
                ),
            ),
            AgentTaskResult(
                task_id="trend_ticket_cs_ticket_cnt_1d",
                success=True,
                query_bundle=QueryBundle(
                    tables=["ads_merchant_profile"],
                    rows=[
                        {"merchant_id": "100", "pt": "2026-06-22", "cs_ticket_cnt_1d": 1},
                        {"merchant_id": "100", "pt": "2026-06-23", "cs_ticket_cnt_1d": 1},
                        {"merchant_id": "100", "pt": "2026-06-24", "cs_ticket_cnt_1d": 3},
                    ],
                    original_row_count=3,
                ),
            ),
        ],
    )

    answer = AnswerComposeService(MisreadingAnswerLlm()).compose("最近7天咨询工单量", MerchantInfo(merchant_id="100"), plan, run, "")

    assert "最近7天咨询工单量是 5 单" in answer
    assert "其余日期" not in answer
    assert "未带日期" not in answer
    assert "建议优先" not in answer
    assert "建议：" in answer
    assert "- 优先回看 6月24日工单上升的咨询类型。" in answer


def recall_bundle_empty():
    from merchant_ai.models import RecallBundle

    return RecallBundle()


def merge_recall_items_for_test(*groups):
    by_id = {}
    for group in groups:
        for item in group:
            current = by_id.get(item.doc_id)
            if current is None:
                by_id[item.doc_id] = item
                continue
            current_queries = list((current.metadata or {}).get("recallQueries") or [])
            item_queries = list((item.metadata or {}).get("recallQueries") or [])
            merged_queries = []
            for query in current_queries + item_queries:
                if query and query not in merged_queries:
                    merged_queries.append(query)
            base = item if item.fusion_score >= current.fusion_score else current
            metadata = {**(current.metadata or {}), **(item.metadata or {})}
            if merged_queries:
                metadata["recallQueries"] = merged_queries
                metadata["recallQuery"] = merged_queries[-1]
            by_id[item.doc_id] = base.model_copy(update={"fusion_score": max(current.fusion_score, item.fusion_score), "metadata": metadata})
    return list(by_id.values())


def trade_refund_goods_pack(include_missing_goods_metric=False):
    metrics = [
        PlanningAssetEntry(
            key="order_detail_cnt",
            table="dwm_trade_order_detail_di",
            columns=["sub_order_id"],
            title="下单数",
            metadata={"sourceColumns": ["sub_order_id"], "formula": "COUNT(DISTINCT sub_order_id)"},
        ),
        PlanningAssetEntry(
            key="refund_bill_cnt",
            table="dwm_trade_refund_detail_di",
            columns=["refund_id"],
            title="退款量",
            metadata={"sourceColumns": ["refund_id"], "formula": "COUNT(DISTINCT refund_id)"},
        ),
        PlanningAssetEntry(
            key="pay_amt",
            table="dwm_trade_refund_detail_di",
            columns=["pay_amt"],
            title="退款金额",
            metadata={"sourceColumns": ["pay_amt"], "formula": "SUM(pay_amt)"},
        ),
        PlanningAssetEntry(
            key="goods_publish_time",
            table="dwm_goods_detail_df",
            columns=["spu_apply_create_time"],
            title="商品发布时间",
            metadata={"sourceColumns": ["spu_apply_create_time"]},
        ),
    ]
    if include_missing_goods_metric:
        metrics.append(
            PlanningAssetEntry(
                key="goods_online_detail_cnt",
                table="dwm_goods_detail_df",
                columns=["spu_status_code", "spu_status_name"],
                title="上架商品量",
                metadata={
                    "sourceColumns": ["spu_status_code", "spu_status_name"],
                    "formula": "SUM(CASE WHEN spu_status_code = 1 OR spu_status_name = '上架' THEN 1 ELSE 0 END)",
                },
            )
        )
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="dwm_trade_order_detail_di",
                columns=["seller_id", "pt", "spu_id", "spu_name", "sub_order_id", "order_id", "discount_rel_id", "pay_amt"],
                title="订单明细表",
                aliases=["订单/子订单明细表", "订单明细", "子订单明细"],
            ),
            PlanningAssetEntry(
                table="dwm_trade_refund_detail_di",
                columns=[
                    "seller_id",
                    "pt",
                    "spu_name",
                    "sub_order_id",
                    "order_id",
                    "refund_id",
                    "pay_amt",
                    "refund_status_name",
                    "refund_create_time",
                ],
                title="退款明细表",
                aliases=["退款/售后明细表", "退款明细", "售后明细"],
            ),
            PlanningAssetEntry(
                table="dwm_goods_detail_df",
                columns=["seller_id", "pt", "spu_id", "spu_name", "spu_apply_create_time", "spu_status_name"],
                title="商品明细表",
                aliases=["商品明细", "SPU 明细"],
            ),
        ],
        metrics=metrics,
        relationships=[
            RelationshipEntry(
                relationship_id="order_refund_by_sub_order",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "sub_order_id", "rightColumn": "sub_order_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="order_goods_by_spu_id",
                left_table="dwm_trade_order_detail_di",
                right_table="dwm_goods_detail_df",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_id", "rightColumn": "spu_id"},
                ],
            ),
            RelationshipEntry(
                relationship_id="goods_refund_by_spu_name",
                left_table="dwm_goods_detail_df",
                right_table="dwm_trade_refund_detail_di",
                join_keys=[
                    {"leftColumn": "seller_id", "rightColumn": "seller_id"},
                    {"leftColumn": "spu_name", "rightColumn": "spu_name"},
                ],
            ),
        ],
    )


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
            "WHERE `%s` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL %d DAY) "
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


class WindowRefundSqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {
            "sql": (
                "WITH ranked AS ("
                "SELECT `seller_id`, `spu_name`, `refund_id`, `refund_create_time`, `pay_amt`, `pt`, "
                "ROW_NUMBER() OVER(PARTITION BY `spu_name` ORDER BY `refund_create_time` DESC) AS `rn` "
                "FROM `dwm_trade_refund_detail_di` "
                "WHERE `seller_id` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"
                ") "
                "SELECT `seller_id`, `spu_name`, `refund_id`, `refund_create_time`, `pay_amt`, `rn` "
                "FROM ranked WHERE `rn` = 1 LIMIT 50"
            ),
            "reason": "latest refund per goods via window function",
        }


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
                "WHERE `seller_id` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
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
                "WHERE `seller_id` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) "
                "GROUP BY `seller_id`, `spu_id` ORDER BY `order_gmv` DESC LIMIT 5"
            )
        }


class MissingEntityKeyFilterSqlLlm:
    configured = True
    last_error = ""
    error_events = []

    def __init__(self):
        self.calls = 0

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.calls += 1
        return {
            "sql": (
                "SELECT `seller_id`, `spu_id`, `spu_name`, COUNT(DISTINCT `ticket_id`) AS `ticket_cnt` "
                "FROM `dwm_cs_ticket_detail_di` "
                "WHERE `seller_id` = '100' AND `pt` >= DATE_SUB(CURDATE(), INTERVAL 60 DAY) "
                "GROUP BY `seller_id`, `spu_id`, `spu_name` ORDER BY `ticket_cnt` DESC LIMIT 10"
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
                "WHERE `seller_id` = '100' AND `pt` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 30 DAY), '%Y%m%d') "
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
                        "id": "call_ls_trade",
                        "name": "semantic_ls",
                        "args": {
                            "topic": "电商交易",
                            "query": "下单最多 SPU 订单指标",
                            "limit": 5,
                            "reason": "inspect semantic workspace manifest before reading detail assets",
                        },
                    }
                ],
            }
        if len(self.calls) == 2:
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
                    "sourcePhrase": "GMV",
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
                                "sourcePhrase": "GMV",
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
                    "sourcePhrase": "GMV",
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
                    "sourcePhrase": "优惠券金额投入",
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
                    {"metricRef": "goods_publish_time", "ownerTable": "dwm_goods_detail_df", "sourcePhrase": "商品发布时间"},
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
        self.calls = 0

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tools = tools
        self.tool_choice = tool_choice or ""
        return self

    async def ainvoke(self, messages):
        self.calls += 1
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


def test_file_run_event_store_persists_run_events_and_trace(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = AgentRunManager(settings)
    thread = manager.create_thread("100", "电商交易", ChatContext(topic="电商交易"))
    run = manager.create_run(thread.thread_id, "100", "最近7天GMV")
    manager.append_event(
        run.run_id,
        thread.thread_id,
        "run.step.completed",
        "PLAN_QUERY_GRAPH",
        {"stepId": "step_1", "toolCallId": "tool_1", "status": "success"},
    )
    response = ChatResponse(
        id="qa_1",
        answer="ok",
        debug_trace={
            "harness": {
                "performance": {"totalDurationMs": 12},
                "traceReplay": {"version": "v2", "path": "/tmp/trace_replay.json"},
            }
        },
    )
    manager.complete_run(run.run_id, response)

    reloaded = AgentRunManager(settings)
    loaded = reloaded.get_run(run.run_id)
    events = reloaded.events(run.run_id)
    trace = reloaded.trace(run.run_id)

    assert loaded is not None
    assert loaded.status == "COMPLETED"
    assert loaded.performance_summary["totalDurationMs"] == 12
    assert loaded.final_answer_hash
    assert loaded.resumable
    assert loaded.checkpoint_ref["backend"] == "sqlite"
    assert loaded.checkpoint_ref["checkpointThreadId"] == "%s:%s" % (thread.thread_id, run.run_id)
    assert loaded.artifact_refs
    assert any(event.step_id == "step_1" and event.tool_call_id == "tool_1" for event in events)
    assert trace is not None
    assert trace["harness"]["traceReplay"]["version"] == "v2"


def test_run_manager_lists_runs_and_builds_dashboard_from_store(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = AgentRunManager(settings)
    thread = manager.create_thread("100", "电商交易", ChatContext(topic="电商交易"))
    completed = manager.create_run(thread.thread_id, "100", "最近7天订单量")
    manager.complete_run(
        completed.run_id,
        ChatResponse(
            id="qa_completed",
            answer="订单量为 10。",
            debug_trace={"harness": {"performance": {"totalDurationMs": 25}}},
        ),
    )
    failed = manager.create_run(thread.thread_id, "100", "最近7天GMV")
    manager.fail_run(failed.run_id, "PLANNER_LLM_TIMEOUT")

    reloaded = AgentRunManager(settings)
    runs = reloaded.list_runs(limit=10, merchant_id="100")
    dashboard = reloaded.dashboard(limit=10, merchant_id="100")

    assert {run.run_id for run in runs} == {completed.run_id, failed.run_id}
    assert dashboard["totalRuns"] == 2
    assert dashboard["statusCounts"]["COMPLETED"] == 1
    assert dashboard["statusCounts"]["FAILED"] == 1
    assert dashboard["runs"][0]["runId"]
    assert dashboard["slowestRuns"][0]["durationMs"] >= 0
    assert dashboard["recentErrors"][0]["error"] == "PLANNER_LLM_TIMEOUT"
    assert run_duration_ms(next(run for run in runs if run.run_id == completed.run_id)) == 25


def test_run_manager_marks_memory_checkpointer_as_not_resumable(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "agent_checkpointer_backend": "memory",
        }
    )
    manager = AgentRunManager(settings)
    thread = manager.create_thread("100")
    run = manager.create_run(thread.thread_id, "100", "最近7天订单量")

    assert not run.resumable
    assert run.checkpoint_ref["backend"] == "memory"


def test_stream_service_emits_answer_delta_events(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = AgentRunManager(settings)

    def run_chat(message, merchant_id, context, listener, thread_id, run_id):
        listener("node.completed", "ANSWER_ANALYSIS", {"answerReady": True})
        return ChatResponse(id="qa_1", answer="abcdefghi", debug_trace={"harness": {"performance": {"totalDurationMs": 1}}})

    service = AgentRunStreamService(manager, run_chat, "100")
    events = list(service.stream(RunCreateRequest(message="最近7天订单量", merchant_id="100")))
    joined = "\n".join(events)
    assert "answer.delta" in joined
    assert "answer.completed" in joined
    assert '"event": "done"' in joined
    assert answer_chunks("abcdefghi", 4) == ["abcd", "efgh", "i"]
    run_id = next(event.run_id for events_for_run in manager.run_events.values() for event in events_for_run if event.event_type == "answer.delta")
    stored_types = [event.event_type for event in manager.events(run_id)]
    assert "answer.delta" in stored_types
    assert "answer.completed" in stored_types


def wait_for_run_status(manager: AgentRunManager, run_id: str, statuses: set[str], timeout_seconds: float = 2.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = manager.get_run(run_id)
        if run and str(getattr(run.status, "value", run.status)) in statuses:
            return run
        time.sleep(0.02)
    return manager.get_run(run_id)


def wait_for_run_event(manager: AgentRunManager, run_id: str, event_type: str, timeout_seconds: float = 2.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if any(event.event_type == event_type for event in manager.events(run_id)):
            return True
        time.sleep(0.02)
    return False


def test_async_run_service_completes_in_background(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = AgentRunManager(settings)

    def run_chat(message, merchant_id, context, listener, thread_id, run_id):
        listener("node.completed", "ANSWER", {"message": message})
        return ChatResponse(id="qa_async", answer="done", debug_trace={"harness": {"performance": {"totalDurationMs": 7}}})

    service = AgentAsyncRunService(manager, run_chat, "100", max_workers=1)
    run = service.submit(RunCreateRequest(message="最近7天订单量", merchant_id="100"))

    assert run.status == "QUEUED"
    completed = wait_for_run_status(manager, run.run_id, {"COMPLETED"})
    events = [event.event_type for event in manager.events(run.run_id)]

    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.answer and completed.answer.answer == "done"
    assert "run.queued" in events
    assert "run.worker.started" in events
    assert "run.completed" in events


def test_async_run_cancel_preserves_canceled_status_after_worker_finishes(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    manager = AgentRunManager(settings)
    started = Event()
    release = Event()

    def run_chat(message, merchant_id, context, listener, thread_id, run_id):
        started.set()
        release.wait(timeout=1)
        return ChatResponse(id="qa_async", answer="late answer", debug_trace={"harness": {"performance": {"totalDurationMs": 9}}})

    service = AgentAsyncRunService(manager, run_chat, "100", max_workers=1)
    run = service.submit(RunCreateRequest(message="慢查询", merchant_id="100"))
    assert started.wait(timeout=1)

    canceled = service.cancel(run.run_id)
    release.set()
    final = wait_for_run_status(manager, run.run_id, {"CANCELED"}, timeout_seconds=1.5)
    assert wait_for_run_event(manager, run.run_id, "run.completion_ignored", timeout_seconds=1.5)
    events = [event.event_type for event in manager.events(run.run_id)]

    assert canceled is not None
    assert final is not None
    assert final.status == "CANCELED"
    assert final.answer is None
    assert "run.cancel.requested" in events
    assert "run.completion_ignored" in events


def test_analysis_skill_runner_generates_evidence_bound_summary(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "skill_confirmation_required": False,
            "skill_worker_parallel_enabled": False,
        }
    )
    service = AnswerComposeService(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "anomaly_check",
            "requiresExplanation": True,
            "analysisGrain": "day",
        },
        intents=[
            QuestionIntent(
                question="最近30天GMV和退款金额走势是否正常？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_formula="SUM(order_gmv_amt_1d)",
                group_by_column="pt",
                metric_resolution={
                    "metricKey": "order_gmv_amt_1d",
                    "ownerTable": "ads_merchant_profile",
                    "formula": "SUM(order_gmv_amt_1d)",
                    "displayName": "GMV",
                },
            )
        ],
    )
    rows = [
        {"pt": "2026-06-01", "order_gmv_amt_1d": "100", "pay_amt": "10", "merchant_id": "100"},
        {"pt": "2026-06-02", "order_gmv_amt_1d": "180", "pay_amt": "20", "merchant_id": "100"},
        {"pt": "2026-06-03", "order_gmv_amt_1d": "420", "pay_amt": "90", "merchant_id": "100"},
    ]
    run = AgentRunResult(
        merged_query_bundle=QueryBundle(tables=["ads_merchant_profile"], rows=rows, original_row_count=len(rows)),
        verified_evidence=VerifiedEvidence(passed=True),
    )
    answer = service.summarize_analysis("最近30天GMV和退款金额走势是否正常？", plan, run, str(tmp_path))
    assert "GMV" in answer
    assert "关键证据" not in answer
    assert "限制" not in answer
    assert "建议" not in answer
    assert "GMV" in answer
    assert "order_gmv_amt_1d" not in answer
    assert "口径" not in answer
    trace = service.last_analysis_skill_trace
    assert trace["activated"]
    assert trace["workerType"] == "SKILL_WORKER"
    assert trace["executionMode"] == "isolated_skill_worker"
    assert trace["isolatedExecution"] is True
    assert trace["metadata"]["name"] == "bi_trend_attribution"
    assert trace["lifecycleStage"] == "completed"
    assert trace["isolatedRunId"].startswith("skill_bi_trend_attribution_")
    assert trace["reuseCandidate"] is True
    assert Path(trace["inputArtifact"]).exists()
    assert Path(trace["outputArtifact"]).exists()
    assert Path(trace["checkpointPath"]).exists()
    assert Path(trace["contextPackagePath"]).exists()
    checkpoint = json.loads(Path(trace["checkpointPath"]).read_text(encoding="utf-8"))
    assert checkpoint["workerType"] == "SKILL_WORKER"
    assert checkpoint["isolatedExecution"] is True
    assert checkpoint["contextPackage"]["skillName"] == "bi_trend_attribution"


def test_skill_worker_executes_complex_skill_with_isolated_context_package(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "llm_api_key": ""})
    worker = SkillWorkerExecutor(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={"analysisIntent": "risk_ranking", "requiresExplanation": True},
        intents=[
            QuestionIntent(
                question="最近30天退款率最高的商品，判断高风险新品。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                metric_name="refund_bill_cnt",
            )
        ],
    )
    bundle = QueryBundle(rows=[{"spu_id": "spu_1", "spu_apply_create_time": "2026-06-01", "refund_bill_cnt": 5}], original_row_count=1)
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="risk", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    result = worker.execute_answer_skill(
        "最近30天退款率最高的商品，判断高风险新品。",
        plan,
        run,
        str(tmp_path),
        skill_name="new_product_risk",
        merchant=MerchantInfo(merchant_id="100"),
        initial_trace={"matchedBy": "test_match"},
    )

    assert "新品风险分析" in result.answer
    assert result.trace["matchedBy"] == "test_match"
    assert result.trace["subAgentType"] == "SKILL_WORKER"
    assert result.trace["lifecycleStage"] == "completed"
    assert result.trace["requiresConfirmation"] is False
    assert result.trace["confirmed"] is True
    assert "awaiting_confirmation" not in result.trace["progress"]
    context = json.loads(Path(result.trace["contextPackagePath"]).read_text(encoding="utf-8"))
    assert context["packageType"] == "skill_worker_context"
    assert context["merchantId"] == "100"
    assert context["allowedTools"]["write_skill_output"] is True
    assert context["allowedTools"]["semantic_read"] is True
    assert context["allowedTools"]["artifact_grep"] is True
    assert context["fileContextTools"]["semantic_read"]
    assert context["verifiedRowCount"] == 1


def test_skill_worker_executes_merchant_sop_skill_with_isolated_context(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "llm_api_key": ""})
    worker = SkillWorkerExecutor(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={
            "analysisIntent": "diagnosis",
            "requiresExplanation": True,
            "skillWorkflow": {"skillName": "refund_rate_diagnosis", "required": True},
        },
        intents=[
            QuestionIntent(
                question="最近7天退款率为什么升高？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DERIVED,
                category=QuestionCategory.REFUND,
                metric_name="refund_rate",
                metric_formula="refund_bill_cnt / order_detail_cnt",
            )
        ],
    )
    bundle = QueryBundle(
        rows=[{"pt": "2026-07-10", "refund_bill_cnt": 12, "order_detail_cnt": 100, "refund_rate": 0.12}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="refund_rate", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    result = worker.execute_answer_skill(
        "最近7天退款率为什么升高？",
        plan,
        run,
        str(tmp_path),
        skill_name="refund_rate_diagnosis",
        merchant=MerchantInfo(merchant_id="100"),
        initial_trace={"matchedBy": "test_sop_match"},
    )

    assert "退款率升高归因" in result.answer
    assert result.trace["skillName"] == "refund_rate_diagnosis"
    assert result.trace["isolatedExecution"] is True
    assert result.trace["lifecycleStage"] == "completed"
    context = json.loads(Path(result.trace["contextPackagePath"]).read_text(encoding="utf-8"))
    assert context["skillName"] == "refund_rate_diagnosis"
    assert context["verifiedRowCount"] == 1


def test_skill_worker_executes_parallel_skills_with_isolated_contexts(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "max_concurrent_skill_workers": 2,
        }
    )
    worker = SkillWorkerExecutor(LlmClient(settings))
    plan = QueryPlan(
        question_understanding={"analysisIntent": "risk_ranking", "requiresExplanation": True},
        intents=[
            QuestionIntent(
                question="最近30天退款率最高的商品，判断高风险新品。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                metric_name="refund_bill_cnt",
            )
        ],
    )
    bundle = QueryBundle(
        rows=[{"spu_id": "spu_1", "spu_apply_create_time": "2026-06-01", "refund_bill_cnt": 5}],
        original_row_count=1,
    )
    run = AgentRunResult(
        task_results=[AgentTaskResult(task_id="risk", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )

    results = worker.execute_answer_skills(
        "最近30天退款率最高的商品，判断高风险新品。",
        plan,
        run,
        ["new_product_risk", "risk_analysis"],
        str(tmp_path),
        merchant=MerchantInfo(merchant_id="100"),
        initial_trace={"matchedBy": "unit_parallel"},
    )

    assert [result.trace["skillName"] for result in results] == ["new_product_risk", "risk_analysis"]
    assert all(result.answer for result in results)
    assert all(result.trace["parallelSkillBatch"] is True for result in results)
    assert all(result.trace["maxConcurrency"] == 2 for result in results)
    assert all(Path(result.trace["checkpointPath"]).exists() for result in results)


def test_workflow_run_analysis_skill_is_lead_agent_dispatchable_node(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "skill_confirmation_required": False,
            "skill_worker_parallel_enabled": False,
        }
    )
    workflow = create_workflow(settings)
    events = []
    state = workflow._initial_state(
        "最近30天GMV和退款金额走势是否正常？",
        "100",
        ChatContext(),
        lambda event_type, node, payload: events.append((event_type, node, payload)),
        "thread_skill_dispatch",
        "run_skill_dispatch",
    )
    state["event_listener"] = lambda event_type, node, payload: events.append((event_type, node, payload))
    state["routing_decision"] = RoutingDecision(route=QuestionRoute.BUSINESS)
    state["plan"] = QueryPlan(
        question_understanding={
            "analysisIntent": "anomaly_check",
            "requiresExplanation": True,
            "analysisGrain": "day",
            "skillWorkflow": {"skillName": "bi_trend_attribution", "required": True},
        },
        intents=[
            QuestionIntent(
                question="最近30天GMV和退款金额走势是否正常？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                metric_name="order_gmv_amt_1d",
                group_by_column="pt",
                preferred_table="ads_merchant_profile",
            )
        ],
    )
    rows = [
        {"pt": "2026-07-01", "order_gmv_amt_1d": "100", "pay_amt": "10", "merchant_id": "100"},
        {"pt": "2026-07-02", "order_gmv_amt_1d": "220", "pay_amt": "50", "merchant_id": "100"},
    ]
    bundle = QueryBundle(tables=["ads_merchant_profile"], rows=rows, original_row_count=len(rows))
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_trend", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    state["sql_generated"] = True
    state["evidence_graph_verified"] = True

    state = workflow.run_analysis_skill(state)

    assert state["analysis_summary"]
    assert state["skill_worker_completed"] is True
    assert state["analysis_skill_trace"]["workerType"] == "SKILL_WORKER"
    assert state["analysis_skill_trace"]["isolatedExecution"] is True
    assert state["skill_match"].skill_name
    assert state["skill_match"].status == "completed"
    assert state["skill_match"].headers[0]["whenToUse"]
    assert state["skill_match"].headers[0]["constraints"]
    assert state["skill_match"].headers[0]["requiredInputs"] or state["skill_match"].headers[0]["whenToUse"]
    assert state["skill_lifecycle_records"]
    lifecycle_stages = [payload["stage"] for event_type, _, payload in events if event_type == "skill.lifecycle"]
    assert "matched" in lifecycle_stages
    assert "isolated_execute" in lifecycle_stages
    assert "progress_synced" in lifecycle_stages
    assert "completed" in lifecycle_stages


def test_workflow_run_analysis_skill_can_parallel_candidate_skills(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "skill_worker_parallel_enabled": True,
            "max_concurrent_skill_workers": 2,
        }
    )
    workflow = create_workflow(settings)
    events = []
    state = workflow._initial_state(
        "最近30天退款率最高的商品，判断高风险新品。",
        "100",
        ChatContext(),
        lambda event_type, node, payload: events.append((event_type, node, payload)),
        "thread_skill_parallel",
        "run_skill_parallel",
    )
    state["event_listener"] = lambda event_type, node, payload: events.append((event_type, node, payload))
    state["routing_decision"] = RoutingDecision(route=QuestionRoute.BUSINESS)
    state["skill_match"] = SkillMatchState(
        skill_name="new_product_risk",
        status="matched",
        matched_by="unit_parallel",
        candidate_skills=["risk_analysis", "new_product_risk"],
        confirmed=True,
    )
    state["plan"] = QueryPlan(
        question_understanding={"analysisIntent": "risk_ranking", "requiresExplanation": True},
        intents=[
            QuestionIntent(
                question="最近30天退款率最高的商品，判断高风险新品。",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.DETAIL,
                category=QuestionCategory.GOODS,
                plan_task_id="risk",
                metric_name="refund_bill_cnt",
            )
        ],
    )
    bundle = QueryBundle(
        rows=[{"spu_id": "spu_1", "spu_apply_create_time": "2026-06-01", "refund_bill_cnt": 5}],
        original_row_count=1,
    )
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="risk", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    state["sql_generated"] = True
    state["evidence_graph_verified"] = True

    result = workflow.run_analysis_skill(state)

    trace = result["analysis_skill_trace"]
    assert result["skill_worker_completed"] is True
    assert trace["workerType"] == "SKILL_WORKER_BATCH"
    assert trace["parallelExecution"] is True
    assert trace["skillNames"] == ["new_product_risk", "risk_analysis"]
    assert trace["completedCount"] == 2
    assert "新品风险分析" in result["analysis_summary"]
    assert "风险分析" in result["analysis_summary"]
    assert len(trace["skillBatchResults"]) == 2
    assert any(record.skill_name == "new_product_risk" for record in result["skill_lifecycle_records"])
    assert any(record.skill_name == "risk_analysis" for record in result["skill_lifecycle_records"])
    lifecycle_stages = [payload["stage"] for event_type, _, payload in events if event_type == "skill.lifecycle"]
    assert "parallel_isolated_execute" in lifecycle_stages
    assert "completed" in lifecycle_stages


def test_workflow_skill_confirmation_gate_pauses_before_worker(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "skill_confirmation_required": True,
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state(
        "最近30天GMV是否异常？",
        "100",
        ChatContext(),
        None,
        "thread_skill_confirm",
        "run_skill_confirm",
    )
    state["routing_decision"] = RoutingDecision(route=QuestionRoute.BUSINESS)
    state["plan"] = QueryPlan(
        question_understanding={"analysisIntent": "anomaly_check", "requiresExplanation": True},
        intents=[
            QuestionIntent(
                question="最近30天GMV是否异常？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                metric_name="order_gmv_amt_1d",
                group_by_column="pt",
            )
        ],
    )
    bundle = QueryBundle(tables=["ads_merchant_profile"], rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": "100"}])
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_trend", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    state["sql_generated"] = True
    state["evidence_graph_verified"] = True

    result = workflow.run_analysis_skill(state)

    assert result["human_clarification_required"] is True
    assert result["human_clarification_type"] == "skill_confirm"
    assert result["skill_match"].status == "waiting_confirmation"
    assert result["analysis_summary"] == ""
    assert any(record.stage == "confirmation_required" for record in result["skill_lifecycle_records"])

    result["request_context"].pending_clarification_type = "skill_confirm"
    resumed = workflow.run_analysis_skill(result)

    assert resumed["skill_worker_completed"] is True
    assert resumed["analysis_skill_trace"]["requiresConfirmation"] is True
    assert resumed["analysis_skill_trace"]["confirmed"] is True
    assert "awaiting_confirmation" not in resumed["analysis_skill_trace"]["progress"]


def test_answer_skill_headers_expose_diana_style_contract():
    headers = answer_skill_headers(get_settings().resources_root / "runtime" / "agent_skills")
    trend = next(item for item in headers if item["name"] == "bi_trend_attribution")
    assert trend["whenToUse"]
    assert trend["constraints"]
    assert trend["path"].endswith("bi_trend_attribution/SKILL.md")


def test_skill_draft_service_keeps_free_exploration_pending_until_review(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path / "workspace")})
    service = SkillDraftService(settings, skill_root=tmp_path / "agent_skills")
    plan = QueryPlan(
        question_understanding={"analysisIntent": "diagnosis"},
        intents=[
            QuestionIntent(
                question="最近30天GMV和退款率异常原因",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                metric_name="order_gmv_amt_1d",
            ),
            QuestionIntent(
                question="最近30天GMV和退款率异常原因",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_trend",
                metric_name="refund_rate",
            ),
        ],
    )
    bundle = QueryBundle(rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100, "refund_rate": 0.12}])
    state = {
        "question": "最近30天GMV和退款率异常原因",
        "thread_id": "thread_draft",
        "run_id": "run_draft",
        "qa_id": "qa_draft",
        "merchant": MerchantInfo(merchant_id="100"),
        "plan": plan,
        "agent_run_result": AgentRunResult(
            task_results=[
                AgentTaskResult(task_id="gmv_trend", success=True, query_bundle=bundle),
                AgentTaskResult(task_id="refund_trend", success=True, query_bundle=bundle),
            ],
            merged_query_bundle=bundle,
            verified_evidence=VerifiedEvidence(passed=True),
        ),
        "evidence_graph_verified": True,
        "analysis_summary": "GMV上涨但退款率也升高。",
        "analysis_skill_trace": {"skillName": "risk_analysis", "lifecycleStage": "completed"},
    }

    draft = service.maybe_create_from_state(state)

    assert draft["status"] == "pending_review"
    assert draft["callable"] is False
    assert service.list_drafts("pending_review")[0]["draftId"] == draft["draftId"]
    reviewed = service.review_draft(
        draft["draftId"],
        SkillDraftReviewRequest(approved=True, reviewer="tester", review_note="ok"),
    )
    assert reviewed["status"] == "approved"
    assert reviewed["draft"]["callable"] is True
    assert reviewed["draft"]["publishedSkillName"]
    assert reviewed["draft"]["skillRegistry"]["status"] == "active"
    market = service.market()
    assert any(item["skillName"] == reviewed["draft"]["publishedSkillName"] for item in market["items"])
    installed = service.install_skill(reviewed["draft"]["publishedSkillName"], merchant_ids=["100"], traffic_percent=25)
    assert installed["skill"]["installScope"]["merchantIds"] == ["100"]
    assert installed["skill"]["grayRelease"]["trafficPercent"] == 25


def test_skill_evaluation_scores_trigger_cases(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path / "workspace"),
            "llm_api_key": "",
            "answer_skill_match_mode": "always",
        }
    )
    service = SkillEvaluationService(settings)

    result = service.evaluate(
        SkillEvaluationRequest(
            cases=[
                SkillEvaluationCase(
                    case_id="trend_should_trigger",
                    question="最近30天GMV是否异常，帮我分析原因",
                    expected_skill="bi_trend_attribution",
                    expect_trigger=True,
                    question_understanding={"analysisIntent": "anomaly_check", "requiresExplanation": True},
                    planned_evidence=[
                        {
                            "taskId": "gmv_trend",
                            "category": "TRADE",
                            "answerMode": "GROUP_AGG",
                            "metric": "order_gmv_amt_1d",
                            "groupBy": "pt",
                        }
                    ],
                    evidence_rows=[{"pt": "2026-07-01", "order_gmv_amt_1d": 100}],
                ),
                SkillEvaluationCase(
                    case_id="detail_should_not_trigger",
                    question="查一下订单 order_1 的状态",
                    expected_skill="",
                    expect_trigger=False,
                    question_understanding={"analysisIntent": "lookup"},
                    planned_evidence=[
                        {
                            "taskId": "order_detail",
                            "category": "TRADE",
                            "answerMode": "DETAIL",
                            "metric": "",
                            "table": "dwm_trade_order_detail_di",
                        }
                    ],
                    evidence_rows=[{"order_id": "order_1", "order_status": "已发货"}],
                ),
            ]
        )
    )

    assert result["total"] == 2
    assert result["passed"] == 2
    assert result["falsePositive"] == 0
    assert result["falseNegative"] == 0


def test_answer_package_hides_alternate_gmv_candidate_metrics():
    question = "最近7天GMV趋势"
    plan = QueryPlan(
        question_understanding={"requestedMeasures": [{"metricRef": "order_gmv_amt_1d"}]},
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_order",
                preferred_table="ads_merchant_profile",
                metric_name="order_gmv_amt_1d",
                metric_resolution={"metricKey": "order_gmv_amt_1d", "displayName": "GMV"},
                group_by_column="pt",
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_pay",
                preferred_table="ads_merchant_profile",
                metric_name="pay_gmv_amt_1d",
                metric_resolution={"metricKey": "pay_gmv_amt_1d", "displayName": "支付GMV", "sourcePhrase": "GMV"},
                group_by_column="pt",
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trade_success",
                preferred_table="ads_merchant_profile",
                metric_name="trade_success_gmv_amt_1d",
                metric_resolution={"metricKey": "trade_success_gmv_amt_1d", "displayName": "交易成功GMV", "sourcePhrase": "GMV"},
                group_by_column="pt",
            ),
        ],
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="gmv_order", success=True, query_bundle=QueryBundle(rows=[{"pt": "2026-06-22", "order_gmv_amt_1d": 1671}])),
            AgentTaskResult(task_id="gmv_pay", success=True, query_bundle=QueryBundle(rows=[{"pt": "2026-06-22", "pay_gmv_amt_1d": 1518}])),
            AgentTaskResult(task_id="gmv_trade_success", success=True, query_bundle=QueryBundle(rows=[{"pt": "2026-06-22", "trade_success_gmv_amt_1d": 1200}])),
        ],
        merged_query_bundle=QueryBundle(rows=[{"pt": "2026-06-22", "order_gmv_amt_1d": 1671}]),
        verified_evidence=VerifiedEvidence(passed=True),
    )

    package = answer_data_package(question, plan, run)
    disclosed = {item.get("metricKey") for item in package["metricDisclosures"]}
    section_metrics = {section.get("metricKey") for section in package["dataSections"]}

    assert disclosed == {"order_gmv_amt_1d"}
    assert section_metrics == {"order_gmv_amt_1d"}
    cleaned = sanitize_business_answer_text(
        "GMV从 1671 变化到 1200。\n同时，交易成功GMV从 1518 变化到 1200。",
        question,
        plan,
        run,
    )
    assert "交易成功GMV" not in cleaned
    assert "GMV从 1671" in cleaned


def test_multi_metric_answer_mentions_all_summary_metrics():
    service = AnswerComposeService(LlmClient(get_settings().model_copy(update={"llm_api_key": ""})))
    question = "最近7天订单量、退款金额、咨询工单量分别是多少？"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.TRADE,
                plan_task_id="order_metric",
                metric_name="order_detail_cnt",
                metric_resolution={"metricKey": "order_detail_cnt", "displayName": "订单量"},
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.REFUND,
                plan_task_id="refund_metric",
                metric_name="pay_amt",
                metric_resolution={"metricKey": "pay_amt", "displayName": "退款金额"},
            ),
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                category=QuestionCategory.CS_TICKET,
                plan_task_id="ticket_metric",
                metric_name="cs_ticket_cnt_1d",
                metric_resolution={"metricKey": "cs_ticket_cnt_1d", "displayName": "咨询工单量"},
            ),
        ]
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(task_id="order_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_detail_cnt": 17}])),
            AgentTaskResult(task_id="refund_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "pay_amt": 130}])),
            AgentTaskResult(task_id="ticket_metric", success=True, query_bundle=QueryBundle(rows=[{"seller_id": "100", "cs_ticket_cnt_1d": 5}])),
        ],
        merged_query_bundle=QueryBundle(rows=[{"seller_id": "100", "order_detail_cnt": 17, "pay_amt": 130, "cs_ticket_cnt_1d": 5}]),
        verified_evidence=VerifiedEvidence(passed=True),
    )

    answer = service.compose(question, MerchantInfo(merchant_id="100"), plan, run, "", allow_llm=False)

    assert "订单量为 17" in answer
    assert "退款金额为 130元" in answer
    assert "咨询工单量为 5" in answer


def test_context_package_keeps_minimal_refs_not_full_debug_trace(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    outputs = tmp_path / "threads" / "thread_1" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "trace_replay.json").write_text("{}", encoding="utf-8")
    state = {
        "question": "最近7天订单退款情况",
        "requested_merchant_id": "100",
        "thread_id": "thread_1",
        "run_id": "run_1",
        "thread_data": ThreadData(thread_id="thread_1", run_id="run_1", outputs_path=str(outputs)),
        "planning_asset_pack": PlanningAssetPack(
            tables=[PlanningAssetEntry(table="dwm_trade_order_detail_di", columns=["seller_id", "pt"])],
            metrics=[PlanningAssetEntry(key="order_cnt", table="dwm_trade_order_detail_di")],
        ),
        "agent_run_result": AgentRunResult(),
        "debugTrace": {"large": "should_not_be_copied"},
    }

    ContextManager(settings).refresh_state(state, "compact_assets")
    package = state["context_packages"][-1]
    payload = package.model_dump(by_alias=True)

    assert package.agent == "PlannerAgent"
    assert package.allowed_tables == ["dwm_trade_order_detail_di"]
    assert package.artifact_refs
    assert "debugTrace" not in json.dumps(payload, ensure_ascii=False)
    assert package.offload_reason


def test_schema_drift_report_identifies_missing_extra_and_type_changes(tmp_path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)})
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    semantic_schema = [
        {"columnName": "seller_id", "type": "bigint"},
        {"columnName": "order_amt", "type": "decimal(18,2)"},
        {"columnName": "semantic_only", "type": "varchar"},
    ]
    live_schema = [
        {"Field": "seller_id", "Type": "bigint"},
        {"Field": "order_amt", "Type": "double"},
        {"Field": "live_only", "Type": "varchar"},
    ]
    version = builder._semantic_catalog_version("电商交易", "fake_table", semantic_schema, live_schema)
    report = builder._schema_drift_report("电商交易", "fake_table", semantic_schema, live_schema, version)

    assert report.semantic_version.startswith("semantic-")
    assert report.schema_version.startswith("schema-")
    assert report.missing_live_columns == ["semantic_only"]
    assert report.extra_live_columns == ["live_only"]
    assert report.type_changed_columns == [{"column": "order_amt", "semanticType": "decimal", "liveType": "double"}]


def test_planner_reflection_emits_structured_repair_requests():
    reflection = PlannerReflectionAgent().reflect("最近7天订单退款情况", QueryPlan(), PlanningAssetPack())

    assert not reflection.passed
    assert reflection.repair_reason
    assert reflection.repair_requests
    assert reflection.repair_requests[0].stage == "planner_reflection"
    assert reflection.repair_requests[0].action in {"semantic_read", "re_understand", "graph_repair"}


def test_doris_repository_caches_repeated_selects(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_doris_select_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
        }
    )
    repo = DorisRepository(settings)

    class CountingDb:
        def __init__(self):
            self.calls = 0

        def query(self, sql, params=None):
            self.calls += 1
            return [{"seller_id": "100", "calls": self.calls}]

    repo.db = CountingDb()
    first = repo.query("SELECT * FROM `fake_table` WHERE `seller_id` = %s", ["100"])
    assert not repo.last_cache_hit
    second = repo.query("SELECT   * FROM `fake_table` WHERE `seller_id` = %s", ["100"])

    assert second == first
    assert repo.last_cache_hit
    assert repo.db.calls == 1
    assert repo.cache_trace()["hits"] == 1


def test_hybrid_recall_caches_by_question_and_topics(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_recall_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
        }
    )
    recall = HybridRecallService(settings, TopicAssetService(settings), WikiMemoryService(settings))
    keywords = type("Keywords", (), {"keywords": ["订单", "退款"]})()

    first = recall.recall("最近7天订单退款", keywords, [], "", "100", [QuestionCategory.TRADE, QuestionCategory.REFUND])
    second = recall.recall("最近7天订单退款", keywords, [], "", "100", [QuestionCategory.TRADE, QuestionCategory.REFUND])

    assert len(second.items) == len(first.items)
    assert recall.cache_trace()["recall"]["hits"] == 1


def test_es_retrieval_caches_successful_bundle(tmp_path, monkeypatch):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_recall_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
        }
    )
    retriever = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))
    calls = {"count": 0}

    def fake_search(query_text, topics, include_base_wiki=False):
        calls["count"] += 1
        return [
            RecallItem(
                doc_id="semantic:test",
                title="退款金额",
                content="退款金额指标",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=10,
            )
        ]

    monkeypatch.setattr(retriever, "_search", fake_search)
    monkeypatch.setattr(retriever, "_exact_metric_evidence", lambda query_text, topics: [])
    request = KnowledgeRetrievalRequest(query="最近7天退款金额", merchant_id="100", topic_categories=[])

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert first.source_refs == second.source_refs
    assert "semantic:test" in first.source_refs
    assert any(ref.endswith(":pay_amt") for ref in first.source_refs)
    assert calls["count"] == 1
    assert retriever.cache_trace()["esRecall"]["hits"] == 1


def test_asset_pack_cache_returns_deep_copy(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_asset_pack_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
        }
    )
    builder = PlanningAssetPackBuilder(TopicAssetService(settings), SkillLoader(settings))
    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:test",
                title="订单表",
                content="订单",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="电商交易",
                table="dwm_trade_order_detail_di",
                fusion_score=10,
            )
        ]
    )

    first = builder.compact("最近7天订单", recall, [QuestionCategory.TRADE])
    first.tables.append(PlanningAssetEntry(table="mutated_table"))
    second = builder.compact("最近7天订单", recall, [QuestionCategory.TRADE])

    assert "mutated_table" not in second.known_tables()
    assert builder.cache_trace()["assetPack"]["hits"] == 1


def test_llm_client_caches_successful_chat_response(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_llm_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
            "llm_api_key": "fake-key",
        }
    )
    llm = LlmClient(settings)

    class Result:
        content = "cached answer"

    class FakeModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return Result()

    fake = FakeModel()
    llm._model = fake

    first = llm.chat("system", "user")
    second = llm.chat("system", "user")

    assert first == second == "cached answer"
    assert fake.calls == 1
    assert llm.last_cache_hit
    assert llm.cache_trace()["hits"] == 1


def test_llm_client_caches_tool_json_chat_response(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_llm_ttl_seconds": 60,
            "cache_memory_max_entries": 8,
            "llm_api_key": "fake-key",
        }
    )
    llm = LlmClient(settings)
    model = ToolCallingAsyncModel()
    llm._model = model
    schema = sql_draft_tool().openai_schema()

    first = llm.tool_json_chat("system", "user", schema, {})
    second = llm.tool_json_chat("system", "user", schema, {})

    assert first == second == {"sql": "SELECT 1", "reason": "unit test"}
    assert model.calls == 1
    assert llm.last_cache_hit
    assert llm.cache_trace()["hits"] == 1


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


class DetailSplitFallbackDoris:
    def __init__(self):
        self.calls = 0
        self.sqls = []

    def query(self, sql, params=None):
        self.calls += 1
        self.sqls.append(sql)
        if self.calls == 1:
            raise RuntimeError("MEM_ALLOC_FAILED: query memory limit exceeded")
        return [
            {
                "seller_id": "100",
                "pt": "2026-06-20",
                "order_id": "order_%d" % self.calls,
                "sub_order_id": "sub_%d" % self.calls,
                "spu_name": "spu_%d" % self.calls,
            }
        ]


class SlowDetailSplitFallbackDoris:
    def __init__(self, delay_seconds=0.2):
        self.calls = 0
        self.sqls = []
        self.delay_seconds = delay_seconds
        self.lock = Lock()

    def query(self, sql, params=None):
        with self.lock:
            self.calls += 1
            call_no = self.calls
            self.sqls.append(sql)
        if call_no == 1:
            raise RuntimeError("MEM_ALLOC_FAILED: query memory limit exceeded")
        time.sleep(self.delay_seconds)
        return [
            {
                "seller_id": "100",
                "pt": "2026-06-20",
                "order_id": "order_%d" % call_no,
                "sub_order_id": "sub_%d" % call_no,
                "spu_name": "spu_%d" % call_no,
            }
        ]


class AlwaysMemoryFailDoris:
    def query(self, sql, params=None):
        raise RuntimeError("MEM_ALLOC_FAILED: query memory limit exceeded")


class FakeDorisWithFreshness:
    def query(self, sql, params=None):
        if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            return [{"min_pt": "20260601", "max_pt": "20260622"}]
        return [{"seller_id": "100", "sub_order_id": "sub_order_id_1", "order_cnt": 1}]


class CapturingFreshnessDoris:
    def __init__(self, max_pt: str = "2026-06-24"):
        self.max_pt = max_pt
        self.sqls = []

    def query(self, sql, params=None):
        self.sqls.append(sql)
        if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            return [{"min_pt": "2026-06-01", "max_pt": self.max_pt}]
        return [{"seller_id": "100", "order_detail_cnt": 12}]


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
