from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionIntent,
    QueryPlan,
    RecallBundle,
    RecallItem,
)
from merchant_ai.services.assets import (
    PlanningAssetPackBuilder,
    TopicAssetService,
    planning_gaps_from_asset_traces,
)
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    QueryGraphValidator,
    compact_asset_planning_contract,
    compact_knowledge_request_gaps,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
    RouteSlotExtractor,
    TopicRouterService,
    planning_hints_from_extracted_keywords,
)


class GenericSemanticAssets:
    def __init__(self) -> None:
        self._tables = {
            "summary_topic": ["rollup_measure"],
            "fact_topic": ["event_fact"],
            "dimension_topic": ["dim_subject"],
        }
        self._assets = {
            "rollup_measure": {
                "tableUsageProfile": {
                    "businessLayer": "ADS",
                    "queryableByAgent": True,
                    "authorityLevel": 95,
                }
            },
            "event_fact": {
                "tableUsageProfile": {
                    "businessLayer": "DWM",
                    "queryableByAgent": True,
                    "authorityLevel": 90,
                }
            },
            "dim_subject": {
                "tableUsageProfile": {
                    "businessLayer": "DIM",
                    "queryableByAgent": True,
                    "authorityLevel": 99,
                }
            },
        }
        self._fields = {
            "rollup_measure": [{"columnName": "tenant_id", "role": "KEY"}],
            "event_fact": [
                {"columnName": "tenant_id", "role": "KEY"},
                {"columnName": "event_id", "role": "KEY"},
                {"columnName": "subject_id", "role": "KEY"},
            ],
            "dim_subject": [{"columnName": "subject_id", "role": "KEY"}],
        }
        self._metrics = {
            "rollup_measure": [
                {
                    "metricKey": "measure_rollup",
                    "businessName": "目标金额",
                    "aliases": ["目标金额"],
                    "detailMetricRef": "semantic:fact_topic:event_fact:metric:measure_value",
                }
            ],
            "event_fact": [
                {
                    "metricKey": "measure_value",
                    "businessName": "事实明细金额",
                    "aliases": ["事实明细金额"],
                    "tableKind": "detail_fact",
                    "metricIntent": "detail_drilldown",
                }
            ],
            "dim_subject": [],
        }

    def all_topic_names(self):
        return list(self._tables)

    def load_manifest(self, topic):
        return [{"tableName": table} for table in self._tables.get(topic, [])]

    def load_table_asset(self, _topic, table):
        return self._assets.get(table, {})

    def load_table_semantic_columns(self, _topic, table):
        return self._fields.get(table, [])

    def load_table_metrics(self, _topic, table):
        return self._metrics.get(table, [])

    def resolve_topic_category(self, topic):
        return topic


def generic_builder() -> PlanningAssetPackBuilder:
    builder = object.__new__(PlanningAssetPackBuilder)
    builder.topic_assets = GenericSemanticAssets()
    return builder


def generic_relationships():
    return [
        (
            "fact_topic",
            {
                "name": "summary_to_fact",
                "leftTable": "rollup_measure",
                "rightTable": "event_fact",
                "keys": [["tenant_id", "tenant_id"]],
            },
        ),
        (
            "dimension_topic",
            {
                "name": "fact_to_subject",
                "leftTable": "event_fact",
                "rightTable": "dim_subject",
                "keys": [["subject_id", "subject_id"]],
            },
        ),
    ]


def generic_hints():
    return {
        "metricPhrases": ["目标金额"],
        "dimensionKeywords": ["分析对象"],
        "dimensions": [
            {
                "phrase": "分析对象",
                "column": "subject_id",
                "topic": "dimension_topic",
                "ownerTable": "dim_subject",
                "role": "KEY",
                "source": "semantic_column",
            }
        ],
        "ranking": {
            "requested": True,
            "limit": 7,
            "order": "desc",
            "anchorMetricPhrase": "目标金额",
            "anchorMetricCandidates": ["measure_rollup"],
        },
    }


def recalled_fact_dimension_relationship() -> RecallBundle:
    return RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:dimension_topic:relationship:fact_to_subject",
                source_type="SEMANTIC_RELATIONSHIP",
                topic="dimension_topic",
                metadata={
                    "semanticRefId": "semantic:dimension_topic:relationship:fact_to_subject",
                    "leftTable": "event_fact",
                    "rightTable": "dim_subject",
                },
            )
        ]
    )


def test_explicit_metric_and_relationship_leaf_reads_enable_dimension_closure() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }

    tables, traces = builder._dimension_aware_seed_tables(
        question="目标金额最高的前7个分析对象",
        planning_hints=generic_hints(),
        precise_seed_tables={"rollup_measure"},
        precise_metric_seed_traces=[
            "precise_metric_seed:rollup_measure:measure_rollup:目标金额"
        ],
        recalled_relationship_tables={"event_fact", "dim_subject"},
        recalled_relationship_refs={
            "semantic:dimension_topic:relationship:fact_to_subject"
        },
        table_topic=table_topic,
        all_relationships=generic_relationships(),
        allow_profile=False,
    )

    assert tables == {"event_fact", "dim_subject"}
    assert "rollup_measure" not in tables
    assert "precise_metric_seed:event_fact:measure_value:目标金额" in traces
    assert "dimension_authority_seed:dim_subject" in traces
    assert "dimension_relationship_ref:semantic:dimension_topic:relationship:fact_to_subject" in traces


def test_simple_exact_metric_without_dimension_does_not_expand_relationships() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }

    tables, traces = builder._targeted_seed_tables(
        "目标金额是多少",
        recalled_fact_dimension_relationship(),
        ["summary_topic"],
        table_topic,
        explicit_tables={"rollup_measure"},
        planning_hints={"metricPhrases": ["目标金额"]},
        all_relationships=generic_relationships(),
    )

    assert tables == {"rollup_measure"}
    assert "targeted_seed_source=explicit_tables" in traces
    assert not any(item.startswith("dimension_relationship_path:") for item in traces)


def test_precise_rag_hit_cannot_enlarge_topic_manifest_boundary() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }
    outside_topic_hit = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:fact_topic:event_fact:metric:measure_value",
                source_type="SEMANTIC_METRIC",
                topic="fact_topic",
                table="event_fact",
                metadata={
                    "semanticKind": "METRIC",
                    "metricKey": "measure_value",
                    "tableName": "event_fact",
                },
            )
        ]
    )

    tables, traces = builder._targeted_seed_tables(
        "一个不包含已发布指标别名的问题",
        outside_topic_hit,
        ["summary_topic"],
        table_topic,
    )

    assert tables == set()
    assert "topic_manifest_boundary_rejected:event_fact" in traces


def test_missing_dimension_path_is_attached_to_planner_graph_as_typed_relationship_request() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }
    tables, traces = builder._dimension_aware_seed_tables(
        question="目标金额最高的前7个分析对象",
        planning_hints=generic_hints(),
        precise_seed_tables={"rollup_measure"},
        precise_metric_seed_traces=[
            "precise_metric_seed:rollup_measure:measure_rollup:目标金额"
        ],
        recalled_relationship_tables={"event_fact", "dim_subject"},
        recalled_relationship_refs=set(),
        table_topic=table_topic,
        all_relationships=generic_relationships()[:1],
        allow_profile=False,
    )
    gap = next(item for item in planning_gaps_from_asset_traces(traces) if item["code"] == "RELATIONSHIP_PATH_REQUIRED")
    assert tables == {"event_fact", "dim_subject"}
    assert gap["metricRefs"] == ["measure_value"]
    assert gap["dimensionRefs"] == ["subject_id"]
    assert gap["sourceOwner"] == "event_fact"
    assert gap["targetOwner"] == "dim_subject"
    assert gap["requiredSemantics"] == ["join_path", "canonical_entity", "cardinality", "fanout_policy"]

    class ConfiguredLlm:
        configured = True
        last_error = ""
        error_events = []

    pack = PlanningAssetPack(metric_compaction={"knowledgeRequestGaps": [gap]})
    planner = QueryGraphPlanner(ConfiguredLlm())
    planner_calls: list[str] = []
    staged_plan = QueryPlan(
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="rank_subjects",
                preferred_table="event_fact",
            )
        ]
    )

    planner._semantic_asset_selection_plan = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Semantic Selector must not produce the staged graph")
    )
    planner._semantic_candidate_hints = lambda *_args, **_kwargs: {}

    def main_planner_payload(*_args, **_kwargs):
        planner_calls.append("main_planner")
        return ({"status": "UNDERSTOOD", "reason": "staged graph planned"}, False, None, "")

    planner.understanding_extractor.initial_payload = main_planner_payload
    planner._compile_planner_payload = lambda *_args, **_kwargs: staged_plan
    plan, requests, reason = planner.plan(
        "目标金额最高的前7个分析对象",
        [],
        "",
        RecallBundle(),
        pack,
        [],
        [],
    )
    assert planner_calls == ["main_planner"]
    assert reason == "staged graph planned"
    assert plan.intents and plan.intents[0].plan_task_id == "rank_subjects"
    assert requests and all(str(item.type) == "RELATIONSHIP" for item in requests)
    assert plan.knowledge_requests == requests
    assert requests[0].request_key.startswith("asset_relationship:")
    assert "planner.relationship_obligations=attached" in plan.agent_trace
    validation = QueryGraphValidator().validate(
        "目标金额最高的前7个分析对象", plan, pack
    )
    assert not validation.valid
    assert "PENDING_KNOWLEDGE_REQUEST" in {item.code for item in validation.gaps}
    assert plan.intents and plan.intents[0].plan_task_id == "rank_subjects"


def test_compact_relationship_gap_keeps_executable_obligation_fields() -> None:
    gap = {
        "code": "RELATIONSHIP_PATH_REQUIRED",
        "requestKey": "relationship:measure:subject",
        "type": "RELATIONSHIP",
        "query": "find a governed path",
        "reason": "dimension owner differs from metric owner",
        "metricRefs": ["measure_value"],
        "dimensionRefs": ["subject_id"],
        "relationshipRefs": ["event_to_subject"],
        "dimensionPhrase": "分析对象",
        "sourceOwner": "event_fact",
        "sourceOwners": ["event_fact"],
        "targetOwner": "dim_subject",
        "requiredSemantics": ["join_path", "cardinality", "fanout_policy"],
    }

    compacted = compact_knowledge_request_gaps(
        PlanningAssetPack(metric_compaction={"knowledgeRequestGaps": [gap]})
    )

    assert compacted == [gap]


def test_cross_table_dependency_without_relationship_is_fail_closed() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(key="fact_a", table="fact_a", columns=["subject_id"]),
            PlanningAssetEntry(key="fact_b", table="fact_b", columns=["subject_id"]),
        ]
    )
    intents = {
        "a": QuestionIntent(
            intent_type=IntentType.VALID,
            answer_mode=AnswerMode.METRIC,
            plan_task_id="a",
            preferred_table="fact_a",
        ),
        "b": QuestionIntent(
            intent_type=IntentType.VALID,
            answer_mode=AnswerMode.METRIC,
            plan_task_id="b",
            preferred_table="fact_b",
        ),
    }
    dependency = PlanDependency(
        anchor_task_id="a",
        dependent_task_id="b",
        join_key="subject_id",
        anchor_column="subject_id",
        dependent_column="subject_id",
    )

    assert QueryGraphValidator()._relationship_supports(dependency, pack, intents) is False


def test_structured_ranking_contract_keeps_dimension_limit_order_and_anchor() -> None:
    question = "最近30天退款金额最高的前10个商品，分别看订单量、工单量和赔付金额。"
    hints = planning_hints_from_extracted_keywords(question, KeywordExtractService().extract(question))

    assert hints["dimensionKeywords"] == ["商品"]
    assert {item["column"] for item in hints["dimensions"]} == {"spu_id", "spu_name"}
    assert {item["ownerTable"] for item in hints["dimensions"]} == {"dwm_goods_detail_df"}
    assert hints["ranking"]["limit"] == 10
    assert hints["ranking"]["order"] == "desc"
    assert hints["ranking"]["anchorMetricPhrase"] == "退款金额"

    contract = compact_asset_planning_contract(
        PlanningAssetPack(metric_compaction={"questionStructure": hints})
    )
    assert contract["dimensionKeywords"] == ["商品"]
    assert contract["ranking"]["limit"] == 10
    assert contract["ranking"]["order"] == "desc"
    assert contract["ranking"]["anchorMetricPhrase"] == "退款金额"


def test_topic_route_broadens_through_asset_declared_detail_metric_lineage() -> None:
    assets = TopicAssetService(get_settings())
    question = "最近30天工单量最高的商品，查看他的商品发布时间"
    keywords = KeywordExtractService(assets).extract(question)
    slots = RouteSlotExtractor(assets).extract(question, keywords)
    decision = TopicRouterService(assets).route(question, keywords, route_slots=slots)

    mentions = {
        (item.kind, item.canonical_key, item.owner_table, str(item.topic), item.source)
        for item in keywords.mentions
    }
    assert (
        "metric",
        "cs_ticket_cnt_1d",
        "ads_merchant_profile",
        str(assets.resolve_topic_category("经营画像")),
        "semantic_metric",
    ) in mentions
    assert (
        "lineage",
        "ticket_cnt",
        "dwm_cs_ticket_detail_di",
        str(assets.resolve_topic_category("客服工单")),
        "semantic_metric_detail_ref",
    ) in mentions
    assert assets.resolve_topic_category("客服工单") in decision.candidate_topics
    assert assets.resolve_topic_category("商品管理") in decision.candidate_topics
    assert keywords.ambiguous_metric_keywords == []

    pack = PlanningAssetPackBuilder(assets).compact(
        question,
        RecallBundle(),
        decision.candidate_topics,
        planning_hints=planning_hints_from_extracted_keywords(question, keywords),
    )
    # Progressive disclosure keeps executable table details out of the initial
    # pack.  Route coverage is represented by the stable L0 manifest until the
    # Core Agent chooses and semantic_read loads exact table details.
    manifest_tables = {
        str(item.get("table") or "")
        for item in (pack.table_manifest or {}).get("tables") or []
    }
    assert {"dwm_cs_ticket_detail_di", "dwm_goods_detail_df"} <= manifest_tables
    assert pack.known_tables() == []
    assert pack.metrics == []


def test_metric_label_span_does_not_become_a_false_grouping_dimension() -> None:
    assets = TopicAssetService(get_settings())
    question = "最近30天商品申请量、审核拒绝量、质检不通过量和假货鉴定量有什么变化？"
    keywords = KeywordExtractService(assets).extract(question)

    assert keywords.dimension_keywords == []
    assert not [item for item in keywords.mentions if item.kind == "lineage"]
    assert set(keywords.topic_scores) == {str(assets.resolve_topic_category("经营画像"))}
