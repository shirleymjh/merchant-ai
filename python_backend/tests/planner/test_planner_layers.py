from merchant_ai.config import get_settings
from merchant_ai.models import (
    GraphValidationGap,
    KnowledgeRequest,
    KnowledgeRequestType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.planning import QueryGraphPlanner
from merchant_ai.services.planning_layers import PlanCompiler, PlanRepairer, UnderstandingExtractor


class NoLlm:
    def __init__(self):
        self.configured = False
        self.last_error = ""
        self.settings = get_settings()


def test_query_graph_planner_exposes_layered_components():
    planner = QueryGraphPlanner(NoLlm())

    assert isinstance(planner.understanding_extractor, UnderstandingExtractor)
    assert isinstance(planner.plan_compiler, PlanCompiler)
    assert isinstance(planner.plan_repairer, PlanRepairer)
    assert planner.graph_contract_validator is not None


def test_understanding_extractor_semantic_fast_path_is_used_without_llm():
    planner = QueryGraphPlanner(NoLlm())
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="table_alpha",
                table="table_alpha",
                columns=["entity_alpha", "axis_alpha", "value_alpha"],
                metadata={
                    "semanticDomain": "domain_alpha",
                    "questionCategory": "IDENTITY",
                    "timeColumn": "axis_alpha",
                    "defaultGroupByColumn": "axis_alpha",
                    "timeGrain": "time_alpha",
                },
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="axis_alpha",
                table="table_alpha",
                title="周期甲",
                aliases=["周期甲"],
                metadata={"semanticRole": "TIME", "grain": "time_alpha"},
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="metric_alpha",
                table="table_alpha",
                columns=["value_alpha"],
                title="指标甲",
                aliases=["指标甲"],
                metadata={
                    "formula": "SUM(value_alpha)",
                    "sourceColumns": ["value_alpha"],
                    "aggregationPolicy": "period_rollup",
                },
            )
        ],
    )

    plan, requests, reason = planner.plan("最近7个周期指标甲最高的前5个", [], "", None, pack, [], [])

    assert plan.intents
    assert requests == []
    assert reason == "SEMANTIC_FAST_PATH"
    assert "planner=semantic_topn_metric_fast_path" in plan.agent_trace


def test_plan_repairer_delegates_to_repair_boundary():
    class FakeLlm:
        configured = False
        last_error = ""

    class FakeCompiler:
        def compile(self, question, understanding, asset_pack):
            return QueryPlan()

    def unchanged(question, plan, asset_pack, *_args):
        return plan

    repairer = PlanRepairer(
        llm=FakeLlm(),
        compiler=FakeCompiler(),
        root_metric_repair=unchanged,
        dependency_key_repair=unchanged,
        missing_domain_repair=unchanged,
        llm_repair=lambda *_args: {},
        enrich_plan=lambda _question, plan, _asset_pack, _payload: plan,
    )
    repaired = repairer.repair("q", QueryPlan(), PlanningAssetPack(), [], [], "", None)

    assert "planner.repair.unavailable" in repaired.agent_trace


def test_plan_repairer_merges_knowledge_requests_without_discarding_graph_contract():
    class FakeLlm:
        configured = True
        last_error = ""

    class FakeCompiler:
        def compile(self, question, understanding, asset_pack):
            return QueryPlan()

    def unchanged(question, plan, asset_pack, *_args):
        return plan

    repairer = PlanRepairer(
        llm=FakeLlm(),
        compiler=FakeCompiler(),
        root_metric_repair=unchanged,
        dependency_key_repair=unchanged,
        missing_domain_repair=unchanged,
        llm_repair=lambda *_args: {
            "status": "NEED_MORE_KNOWLEDGE",
            "knowledgeRequests": [
                {
                    "type": "METRIC",
                    "query": "governed metric definition",
                    "neededForTaskId": "anchor_alpha",
                    "reason": "metric contract is incomplete",
                }
            ],
        },
        enrich_plan=lambda _question, plan, _asset_pack, _payload: plan,
    )
    original = QueryPlan(
        intents=[QuestionIntent(plan_task_id="anchor_alpha", preferred_table="table_alpha")],
        final_required_evidence=["metric_alpha"],
        question_understanding={"analysisIntent": "none"},
        agent_trace=["planner=initial"],
    )

    repaired = repairer.repair(
        "q",
        original,
        PlanningAssetPack(),
        [GraphValidationGap(code="METRIC_RESOLUTION_NEEDED")],
        [],
        "",
        None,
    )

    assert repaired is not original
    assert repaired.intents == original.intents
    assert repaired.final_required_evidence == ["metric_alpha"]
    assert repaired.question_understanding == {"analysisIntent": "none"}
    assert repaired.knowledge_requests[0].needed_for_task_id == "anchor_alpha"
    assert repaired.agent_trace == ["planner=initial", "planner.repair=llm_requested_knowledge"]


def test_plan_repairer_dedupes_only_identical_retrieval_semantics():
    class FakeLlm:
        configured = True
        last_error = ""

    class FakeCompiler:
        def compile(self, question, understanding, asset_pack):
            return QueryPlan()

    def unchanged(question, plan, asset_pack, *_args):
        return plan

    repairer = PlanRepairer(
        llm=FakeLlm(),
        compiler=FakeCompiler(),
        root_metric_repair=unchanged,
        dependency_key_repair=unchanged,
        missing_domain_repair=unchanged,
        llm_repair=lambda *_args: {
            "status": "NEED_MORE_KNOWLEDGE",
            "knowledgeRequests": [
                {
                    "type": "METRIC",
                    "query": "  GOVERNED   METRIC DEFINITION ",
                    "neededForTaskId": "anchor_alpha",
                    "sourcePhrase": "订单量",
                    "expectedRefs": ["semantic:metric:b", "semantic:metric:a"],
                    "reason": "new prose must not create another request",
                    "round": 7,
                    "requestKey": "runtime-only",
                },
                {
                    "type": "METRIC",
                    "query": "governed metric definition",
                    "neededForTaskId": "anchor_alpha",
                    "sourcePhrase": "支付订单量",
                    "expectedRefs": ["semantic:metric:a", "semantic:metric:b"],
                    "reason": "different source phrase is a distinct retrieval request",
                },
            ],
        },
        enrich_plan=lambda _question, plan, _asset_pack, _payload: plan,
    )
    original_request = KnowledgeRequest(
        type=KnowledgeRequestType.METRIC,
        query="governed metric definition",
        needed_for_task_id="anchor_alpha",
        source_phrase="订单量",
        expected_refs=["semantic:metric:a", "semantic:metric:b"],
        reason="initial prose",
    )
    original = QueryPlan(knowledge_requests=[original_request])

    repaired = repairer.repair(
        "q",
        original,
        PlanningAssetPack(),
        [GraphValidationGap(code="METRIC_RESOLUTION_NEEDED")],
        [],
        "",
        None,
    )

    assert len(repaired.knowledge_requests) == 2
    assert [item.source_phrase for item in repaired.knowledge_requests] == ["订单量", "支付订单量"]
    assert repaired.knowledge_requests[0].reason == "initial prose"


def test_trace_only_local_repair_does_not_preempt_structured_llm_repair():
    class FakeLlm:
        configured = True
        last_error = ""

    class FakeCompiler:
        def compile(self, question, understanding, asset_pack):
            return QueryPlan(
                intents=[QuestionIntent(plan_task_id="repaired_task", preferred_table="table_alpha")]
            )

    def trace_only(question, plan, asset_pack, *_args):
        plan.compiler_trace.append("local repair inspected the graph")
        return plan

    def unchanged(question, plan, asset_pack, *_args):
        return plan

    repairer = PlanRepairer(
        llm=FakeLlm(),
        compiler=FakeCompiler(),
        root_metric_repair=trace_only,
        dependency_key_repair=unchanged,
        missing_domain_repair=unchanged,
        llm_repair=lambda *_args: {
            "status": "UNDERSTOOD",
            "questionUnderstanding": {"analysisIntent": "none"},
        },
        enrich_plan=lambda _question, plan, _asset_pack, _payload: plan,
    )
    original = QueryPlan(
        intents=[QuestionIntent(plan_task_id="original_task", preferred_table="table_alpha")]
    )

    repaired = repairer.repair(
        "q",
        original,
        PlanningAssetPack(),
        [GraphValidationGap(code="MISSING_EVIDENCE_CONTRACT")],
        [],
        "",
        None,
    )

    assert repaired.intents[0].plan_task_id == "repaired_task"
    assert "planner.repair=llm_reunderstanding" in repaired.agent_trace
    assert "planner.repair=promote_more_specific_root_metric" not in repaired.agent_trace
