from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentTaskResult,
    AnswerMode,
    GraphValidationGap,
    GraphValidationResult,
    IntentType,
    NodePlanCritiqueResult,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    ResolvedTimeRange,
)
from merchant_ai.services.assets import PlanningAssetPackBuilder, TopicAssetService
from merchant_ai.services.planning import QueryGraphPlanner
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.services.query import (
    ExecutionGraphPreparationRequired,
    ExecutionGraphValidationError,
    NodeWorkerExecutor,
    SqlValidationService,
    prepare_execution_graph,
)
from merchant_ai.services.retrieval import EsKnowledgeRetrievalService
from merchant_ai.services.semantic_metrics import seal_semantic_metric_resolution
from merchant_ai.services.time_semantics import (
    apply_time_window_contract_to_plan,
    resolve_time_window_contract,
)


class UnconfiguredLlm:
    configured = False


class NeverCalledRepository:
    def __init__(self) -> None:
        self.calls = 0

    def query(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("invalid or unprepared execution graphs must not reach Doris")


def governed_metric_intent(metric_key: str, label: str) -> QuestionIntent:
    return QuestionIntent(
        question=label,
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id=metric_key,
        preferred_table="merchant_daily",
        group_by_column="pt",
        metric_column=metric_key,
        metric_name=metric_key,
        metric_formula="SUM(`%s`)" % metric_key,
        output_keys=["pt"],
        required_evidence=["pt", metric_key],
        days=30,
        limit=30,
        sql_strategy="structured_first",
    )


def governed_multi_metric_graph() -> tuple[str, QueryPlan, PlanningAssetPack]:
    question = "最近30天指标甲和指标乙有什么变化？"
    plan = QueryPlan(
        intents=[
            governed_metric_intent("metric_a", "指标甲"),
            governed_metric_intent("metric_b", "指标乙"),
        ]
    )
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="merchant_daily",
                topic="test",
                columns=["merchant_id", "pt", "metric_a", "metric_b"],
                metadata={"timeColumn": "pt", "timeGrain": "day"},
            )
        ],
        fields=[
            PlanningAssetEntry(
                key="pt",
                table="merchant_daily",
                title="日期",
                metadata={"semanticRole": "TIME", "groupable": True},
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="metric_a",
                table="merchant_daily",
                title="指标甲",
                aliases=["指标甲"],
                columns=["metric_a"],
                metadata={
                    "formula": "SUM(metric_a)",
                    "sourceColumns": ["metric_a"],
                    "unit": "单",
                },
            ),
            PlanningAssetEntry(
                key="metric_b",
                table="merchant_daily",
                title="指标乙",
                aliases=["指标乙"],
                columns=["metric_b"],
                metadata={
                    "formula": "SUM(metric_b)",
                    "sourceColumns": ["metric_b"],
                    "unit": "单",
                },
            ),
        ],
    )
    return question, plan, pack


def test_prepare_execution_graph_is_pure_and_validates_the_merged_graph() -> None:
    question, plan, pack = governed_multi_metric_graph()
    original = plan.model_dump(by_alias=True, mode="json")

    prepared = prepare_execution_graph(question, plan, pack)

    assert plan.model_dump(by_alias=True, mode="json") == original
    assert len(plan.intents) == 2
    assert prepared.changed is True
    assert prepared.executable is True
    assert prepared.validation.valid is True
    assert prepared.validator_name == "QueryGraphValidator"
    assert prepared.source_plan_fingerprint != prepared.execution_plan_fingerprint
    assert len(prepared.plan.intents) == 1
    assert {spec["metricName"] for spec in prepared.plan.intents[0].metric_specs} == {
        "metric_a",
        "metric_b",
    }
    assert any("same_table_metric_merge" in note for note in prepared.optimization_notes)


def test_published_three_metric_question_preserves_asset_formulas_and_one_time_range() -> None:
    question = "最近30天订单量、退款金额和退货量分别是多少？"
    expected_metric_keys = {"order_cnt_1d", "refund_amt_1d", "return_cnt_1d"}
    settings = get_settings()
    topic_assets = TopicAssetService(settings)
    retrieval = EsKnowledgeRetrievalService(settings, topic_assets)

    candidates = retrieval._resolve_metric_candidates(question, [])
    candidates_by_key = {str(item.get("metricKey") or ""): item for item in candidates}

    assert len(candidates) == len(expected_metric_keys)
    assert set(candidates_by_key) == expected_metric_keys
    assert len({str(item.get("tableName") or "") for item in candidates}) == 1
    assert len({str(item.get("topic") or "") for item in candidates}) == 1

    owner_table = str(candidates[0]["tableName"])
    owner_topic = str(candidates[0]["topic"])
    published_asset = topic_assets.load_table_asset(owner_topic, owner_table)
    published_metrics = {
        str(item.get("metricKey") or ""): item
        for item in published_asset.get("metrics") or []
        if str(item.get("metricKey") or "") in expected_metric_keys
    }
    assert set(published_metrics) == expected_metric_keys
    for metric_key in expected_metric_keys:
        candidate = candidates_by_key[metric_key]
        published = published_metrics[metric_key]
        assert candidate["formula"] == published["formula"]
        assert candidate["sourceColumns"] == published["sourceColumns"]
        assert candidate["aggregationPolicy"] == published["aggregationPolicy"]

    asset_pack = PlanningAssetPack()
    PlanningAssetPackBuilder(topic_assets).expand_for_metric_catalog_resolution(
        asset_pack,
        question,
    )
    selected_assets = [
        {
            "semanticRefId": candidate["semanticRefId"],
            "metricRef": candidate["metricKey"],
            "ownerTable": candidate["tableName"],
            "sourcePhrase": candidate["matchedMetricLabel"],
        }
        for candidate in candidates
    ]
    plan = QueryGraphPlanner(UnconfiguredLlm())._compile_semantic_asset_selection_payload(
        question,
        {
            "status": "SELECTED",
            "queryContract": {
                "contractType": "independent_metrics",
                "timeWindowDays": 30,
            },
            "selectedRefs": [item["semanticRefId"] for item in selected_assets],
            "selectedAssets": selected_assets,
        },
        asset_pack,
    )

    intents_by_metric = {intent.metric_name: intent for intent in plan.intents}
    assert len(plan.intents) == len(expected_metric_keys)
    assert set(intents_by_metric) == expected_metric_keys
    for metric_key in expected_metric_keys:
        intent = intents_by_metric[metric_key]
        published = published_metrics[metric_key]
        assert intent.preferred_table == owner_table
        assert intent.metric_formula.replace("`", "") == published["formula"].replace("`", "")
        assert intent.metric_resolution["sourceColumns"] == published["sourceColumns"]
        assert intent.metric_resolution["aggregationPolicy"] == published["aggregationPolicy"]

    time_contract = resolve_time_window_contract(
        question,
        now=datetime(2026, 7, 16, 12, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    plan = apply_time_window_contract_to_plan(plan, time_contract)
    logical_time_ranges = [
        intent.time_range.model_dump(by_alias=True, mode="json") for intent in plan.intents
    ]
    assert logical_time_ranges
    assert all(time_range == logical_time_ranges[0] for time_range in logical_time_ranges)
    assert logical_time_ranges[0]["days"] == 30
    assert all(intent.days == 30 for intent in plan.intents)

    prepared = prepare_execution_graph(question, plan, asset_pack)

    assert prepared.executable is True
    assert len(prepared.plan.intents) == 1
    merged = prepared.plan.intents[0]
    specs_by_metric = {str(spec.get("metricName") or ""): spec for spec in merged.metric_specs}
    assert len(merged.metric_specs) == len(expected_metric_keys)
    assert set(specs_by_metric) == expected_metric_keys
    assert merged.days == 30
    assert merged.time_range == plan.intents[0].time_range
    for metric_key in expected_metric_keys:
        spec = specs_by_metric[metric_key]
        published = published_metrics[metric_key]
        assert str(spec["metricFormula"]).replace("`", "") == str(published["formula"]).replace("`", "")
        assert spec["sourceColumns"] == published["sourceColumns"]


def test_node_worker_refuses_to_normalize_the_callers_graph_in_place() -> None:
    question, plan, pack = governed_multi_metric_graph()
    original = plan.model_dump(by_alias=True, mode="json")
    repository = NeverCalledRepository()
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        repository,
        SqlValidationService(),
        get_settings(),
    )

    with pytest.raises(ExecutionGraphPreparationRequired):
        worker.execute_plan("100", plan, pack, "", question)

    assert plan.model_dump(by_alias=True, mode="json") == original
    assert repository.calls == 0


def test_invalid_prepared_graph_cannot_reach_node_execution() -> None:
    question, _, pack = governed_multi_metric_graph()
    invalid_plan = QueryPlan(
        intents=[
            governed_metric_intent("missing_metric", "不存在的指标"),
        ]
    )
    prepared = prepare_execution_graph(question, invalid_plan, pack)
    repository = NeverCalledRepository()
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        repository,
        SqlValidationService(),
        get_settings(),
    )

    assert prepared.validation.valid is False
    with pytest.raises(ExecutionGraphValidationError):
        worker.execute_plan(
            "100",
            prepared.plan,
            pack,
            "",
            question,
            execution_preparation=prepared,
        )

    assert repository.calls == 0


class StaleOfflineRepository:
    def __init__(self) -> None:
        self.sqls: list[str] = []
        self.realtime_max = datetime.now(ZoneInfo("Australia/Melbourne")).date().isoformat()

    def query(self, sql, *_args, **_kwargs):
        self.sqls.append(sql)
        if "MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            if "`merchant_realtime`" in sql:
                return [{"min_value": self.realtime_max, "max_value": self.realtime_max}]
            return [{"min_value": "2000-01-01", "max_value": "2000-01-01"}]
        raise AssertionError("runtime graph preparation must not execute business SQL")


class RecordingGraphValidator:
    def __init__(self, reject_realtime: bool = False) -> None:
        self.reject_realtime = reject_realtime
        self.tables: list[list[str]] = []

    def validate(self, _question, plan, _asset_pack, _memory_constraints):
        tables = [intent.preferred_table for intent in plan.intents]
        self.tables.append(tables)
        if self.reject_realtime and "merchant_realtime" in tables:
            return GraphValidationResult(
                valid=False,
                repairable=False,
                gaps=[
                    GraphValidationGap(
                        code="REALTIME_EXECUTION_GRAPH_REJECTED",
                        reason="focused test rejects the transformed graph",
                    )
                ],
            )
        return GraphValidationResult(valid=True, repairable=False)


class CapturingNodeWorker(NodeWorkerExecutor):
    def __init__(self, repository) -> None:
        super().__init__(
            UnconfiguredLlm(),
            repository,
            SqlValidationService(),
            get_settings(),
        )
        self.executed_tables: list[str] = []

    def execute_node(self, intent, _asset_pack, _knowledge_context, _context):
        self.executed_tables.append(intent.preferred_table)
        return AgentTaskResult(
            task_id=intent.plan_task_id,
            success=True,
            summary="captured",
            query_bundle=QueryBundle(
                tables=[intent.preferred_table],
                rows=[{"value": 1}],
                original_row_count=1,
                summary="captured",
            ),
        )


def realtime_fallback_graph() -> tuple[str, QueryPlan, PlanningAssetPack]:
    question = "runtime fallback contract test"
    source_ref = "semantic:test:merchant_offline:metric:order_cnt"
    target_ref = "semantic:test:merchant_realtime:metric:order_cnt"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question=question,
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="order_cnt",
                preferred_table="merchant_offline",
                metric_column="order_cnt",
                metric_name="order_cnt",
                metric_formula="SUM(order_cnt)",
                required_evidence=["order_cnt"],
                days=1,
                time_range=ResolvedTimeRange(
                    kind="rolling",
                    days=1,
                    calendar_anchor_policy="runtime_current_date",
                    data_as_of_policy="latest_available_partition",
                ),
                sql_strategy="structured_first",
                metric_resolution=seal_semantic_metric_resolution(
                    {
                        "semanticRefId": source_ref,
                        "metricKey": "order_cnt",
                        "ownerTable": "merchant_offline",
                        "formula": "SUM(order_cnt)",
                        "sourceColumns": ["order_cnt"],
                    }
                ),
            )
        ]
    )
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="merchant_offline",
                columns=["merchant_id", "pt", "order_cnt"],
                metadata={"merchantFilterColumn": "merchant_id", "timeColumn": "pt"},
            ),
            PlanningAssetEntry(
                table="merchant_realtime",
                columns=["merchant_id", "pt", "order_cnt"],
                metadata={"merchantFilterColumn": "merchant_id", "timeColumn": "pt"},
            ),
        ],
        metrics=[
            PlanningAssetEntry(
                key="order_cnt",
                table="merchant_offline",
                title="订单量",
                columns=["order_cnt"],
                source_ref_id=source_ref,
                metadata={"formula": "SUM(order_cnt)", "sourceColumns": ["order_cnt"]},
            ),
            PlanningAssetEntry(
                key="order_cnt",
                table="merchant_realtime",
                title="订单量",
                columns=["order_cnt"],
                source_ref_id=target_ref,
                metadata={"formula": "SUM(order_cnt)", "sourceColumns": ["order_cnt"]},
            ),
        ],
        realtime_fallbacks=[
            PlanningAssetEntry(
                key="merchant_offline",
                table="merchant_realtime",
                metadata={
                    "sourceTable": "merchant_offline",
                    "metricMappings": [
                        {
                            "sourceSemanticRefId": source_ref,
                            "targetSemanticRefId": target_ref,
                        }
                    ],
                },
            )
        ],
    )
    return question, plan, pack


def test_realtime_fallback_reprepares_and_binds_the_final_execution_graph() -> None:
    question, plan, pack = realtime_fallback_graph()
    original_fingerprint = query_graph_fingerprint(plan)
    validator = RecordingGraphValidator()
    repository = StaleOfflineRepository()
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph(
        "100",
        plan,
        pack,
        question,
        graph_validator=validator,
    )

    assert validator.tables == [["merchant_offline"], ["merchant_realtime"]]
    assert prepared.executable is True
    assert prepared.changed is True
    assert prepared.freshness_bound is True
    assert prepared.runtime_fallback_task_ids == ("order_cnt",)
    assert prepared.runtime_source_plan_fingerprint == original_fingerprint
    assert prepared.execution_plan_fingerprint != original_fingerprint
    assert prepared.plan.intents[0].preferred_table == "merchant_realtime"
    assert prepared.freshness_reports[0].status == "STALE_USE_REALTIME_FALLBACK"
    assert prepared.freshness_reports[0].table == "merchant_realtime"
    assert prepared.freshness_reports[0].fallback_from_table == "merchant_offline"
    assert prepared.freshness_reports[0].fallback_from_max_time_value == "2000-01-01"
    assert prepared.freshness_reports[0].max_pt == repository.realtime_max
    assert prepared.snapshot_alignment.status == "ALIGNED_COMPLETE"
    assert prepared.snapshot_alignment.common_anchor_time_value == repository.realtime_max
    assert prepared.snapshot_alignment.sources[0].table == "merchant_realtime"
    assert prepared.snapshot_alignment.sources[0].source_max_time_value == repository.realtime_max
    assert prepared.snapshot_alignment.sources[0].compatible is True
    assert query_graph_fingerprint(plan) == original_fingerprint

    result = worker.execute_plan(
        "100",
        prepared.plan,
        pack,
        "",
        question,
        execution_preparation=prepared,
    )

    assert worker.executed_tables == ["merchant_realtime"]
    assert result.executed_query_graph_fingerprint == prepared.execution_plan_fingerprint


def test_stale_realtime_fallback_cannot_be_reported_as_complete_snapshot() -> None:
    question, plan, pack = realtime_fallback_graph()
    repository = StaleOfflineRepository()
    repository.realtime_max = "2000-01-01"
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph("100", plan, pack, question)

    report = prepared.freshness_reports[0]
    assert prepared.plan.intents[0].preferred_table == "merchant_realtime"
    assert report.table == "merchant_realtime"
    assert report.status == "STALE_REALTIME_FALLBACK_STALE"
    assert prepared.snapshot_alignment.status == "ALIGNMENT_INCOMPLETE"
    assert prepared.snapshot_alignment.complete is False
    assert prepared.snapshot_alignment.sources[0].compatible is False


def test_failed_realtime_freshness_check_cannot_be_reported_as_complete_snapshot() -> None:
    class FailedRealtimeFreshnessRepository(StaleOfflineRepository):
        def query(self, sql, *_args, **_kwargs):
            if "MIN(`pt`)" in sql and "`merchant_realtime`" in sql:
                self.sqls.append(sql)
                raise RuntimeError("realtime freshness unavailable")
            return super().query(sql, *_args, **_kwargs)

    question, plan, pack = realtime_fallback_graph()
    repository = FailedRealtimeFreshnessRepository()
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph("100", plan, pack, question)

    report = prepared.freshness_reports[0]
    assert prepared.plan.intents[0].preferred_table == "merchant_realtime"
    assert report.table == "merchant_realtime"
    assert report.status == "STALE_REALTIME_FALLBACK_FRESHNESS_UNAVAILABLE"
    assert "realtime freshness unavailable" in report.reason
    assert prepared.snapshot_alignment.status == "ALIGNMENT_INCOMPLETE"
    assert prepared.snapshot_alignment.complete is False
    assert prepared.snapshot_alignment.sources[0].compatible is False


def test_invalid_realtime_execution_graph_is_rejected_before_node_dispatch() -> None:
    question, plan, pack = realtime_fallback_graph()
    validator = RecordingGraphValidator(reject_realtime=True)
    repository = StaleOfflineRepository()
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph(
        "100",
        plan,
        pack,
        question,
        graph_validator=validator,
    )

    assert validator.tables == [["merchant_offline"], ["merchant_realtime"]]
    assert prepared.executable is False
    assert prepared.validation.gaps[0].code == "REALTIME_EXECUTION_GRAPH_REJECTED"
    with pytest.raises(ExecutionGraphValidationError):
        worker.execute_plan(
            "100",
            prepared.plan,
            pack,
            "",
            question,
            execution_preparation=prepared,
        )

    assert worker.executed_tables == []
    assert len(repository.sqls) == 2


def test_stale_offline_source_without_production_mapping_exposes_gap_without_guessing_table() -> None:
    question, plan, pack = realtime_fallback_graph()
    pack.realtime_fallbacks = []
    validator = RecordingGraphValidator()
    repository = StaleOfflineRepository()
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph(
        "100",
        plan,
        pack,
        question,
        graph_validator=validator,
    )

    report = prepared.freshness_reports[0]
    assert prepared.plan.intents[0].preferred_table == "merchant_offline"
    assert prepared.runtime_fallback_task_ids == ()
    assert report.table == "merchant_offline"
    assert report.fallback_table == ""
    assert report.status == "STALE_NO_REALTIME_FALLBACK_MAPPING"
    assert "no governed production realtime fallback mapping" in report.reason
    assert prepared.snapshot_alignment.status == "ALIGNMENT_INCOMPLETE"
    assert prepared.snapshot_alignment.aligned is False
    assert prepared.snapshot_alignment.complete is False
    assert prepared.snapshot_alignment.sources[0].table == "merchant_offline"
    assert prepared.snapshot_alignment.sources[0].compatible is False
    assert validator.tables == [["merchant_offline"]]
    assert len(repository.sqls) == 1


def test_stale_metric_without_governed_realtime_metric_mapping_does_not_switch_table() -> None:
    question, plan, pack = realtime_fallback_graph()
    pack.realtime_fallbacks[0].metadata["metricMappings"] = []
    repository = StaleOfflineRepository()
    worker = CapturingNodeWorker(repository)

    prepared = worker.prepare_runtime_execution_graph("100", plan, pack, question)

    report = prepared.freshness_reports[0]
    assert prepared.plan.intents[0].preferred_table == "merchant_offline"
    assert prepared.runtime_fallback_task_ids == ()
    assert report.fallback_table == ""
    assert report.status == "STALE_NO_GOVERNED_REALTIME_METRIC_MAPPING"
    assert "requested metric has no governed production realtime mapping" in report.reason
    assert prepared.snapshot_alignment.complete is False
    assert prepared.snapshot_alignment.sources[0].compatible is False
    assert len(repository.sqls) == 1


def test_late_realtime_selection_fails_closed_instead_of_executing_a_new_table() -> None:
    class AlwaysValidNodeCritic:
        def review(self, contract):
            return NodePlanCritiqueResult(
                task_id=contract.task_id,
                valid=True,
                message="focused node contract passed",
            )

    question, plan, pack = realtime_fallback_graph()
    validator = RecordingGraphValidator()
    repository = StaleOfflineRepository()
    worker = NodeWorkerExecutor(
        UnconfiguredLlm(),
        repository,
        SqlValidationService(),
        get_settings(),
    )
    worker.node_contract_validator = AlwaysValidNodeCritic()
    logical_preparation = prepare_execution_graph(
        question,
        plan,
        pack,
        graph_validator=validator,
    )

    result = worker.execute_plan(
        "100",
        logical_preparation.plan,
        pack,
        "",
        question,
        execution_preparation=logical_preparation,
    )

    assert result.task_results[0].success is False
    assert "EXECUTION_GRAPH_CHANGED_AFTER_PREPARATION" in result.task_results[0].summary
    assert result.task_results[0].freshness_reports[0].status == "STALE_REQUIRES_GRAPH_REPREPARATION"
    assert len(repository.sqls) == 1
    assert "merchant_realtime" not in repository.sqls[0]
