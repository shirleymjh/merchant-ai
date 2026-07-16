from merchant_ai.config import get_settings
from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack, QueryPlan
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
