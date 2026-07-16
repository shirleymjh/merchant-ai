from __future__ import annotations

from merchant_ai.models import (
    AnswerMode,
    IntentType,
    PlanDependency,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionIntent,
    RecallBundle,
    RecallItem,
)
from merchant_ai.services.assets import (
    PlanningAssetPackBuilder,
    planning_gaps_from_asset_traces,
)
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    QueryGraphValidator,
    compact_asset_planning_contract,
)
from merchant_ai.services.routing import (
    KeywordExtractService,
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


def test_exact_summary_seed_keeps_recalled_relationship_and_uses_dimension_closure() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }

    tables, traces = builder._targeted_seed_tables(
        "目标金额最高的前7个分析对象",
        recalled_fact_dimension_relationship(),
        ["summary_topic"],
        table_topic,
        planning_hints=generic_hints(),
        all_relationships=generic_relationships(),
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
        planning_hints={"metricPhrases": ["目标金额"]},
        all_relationships=generic_relationships(),
    )

    assert tables == {"rollup_measure"}
    assert any(item == "precise_metric_seed:rollup_measure:measure_rollup:目标金额" for item in traces)
    assert not any(item.startswith("dimension_relationship_path:") for item in traces)


def test_missing_dimension_path_becomes_typed_relationship_request_before_any_plan() -> None:
    builder = generic_builder()
    table_topic = {
        "rollup_measure": "summary_topic",
        "event_fact": "fact_topic",
        "dim_subject": "dimension_topic",
    }
    tables, traces = builder._targeted_seed_tables(
        "目标金额最高的前7个分析对象",
        RecallBundle(),
        ["summary_topic"],
        table_topic,
        planning_hints=generic_hints(),
        all_relationships=generic_relationships()[:1],
    )
    gap = next(item for item in planning_gaps_from_asset_traces(traces) if item["code"] == "RELATIONSHIP_PATH_REQUIRED")
    assert tables == {"event_fact", "dim_subject"}
    assert gap["metricRefs"] == ["measure_value"]
    assert gap["dimensionRefs"] == ["subject_id"]
    assert gap["sourceOwner"] == "event_fact"
    assert gap["targetOwner"] == "dim_subject"
    assert gap["requiredSemantics"] == ["join_path", "canonical_entity", "cardinality", "fanout_policy"]

    class NeverCalledLlm:
        configured = True
        last_error = ""
        error_events = []

        def json_chat(self, *_args, **_kwargs):
            raise AssertionError("pre-plan relationship obligations must bypass the LLM")

    pack = PlanningAssetPack(metric_compaction={"knowledgeRequestGaps": [gap]})
    plan, requests, reason = QueryGraphPlanner(NeverCalledLlm()).plan(
        "目标金额最高的前7个分析对象",
        [],
        "",
        RecallBundle(),
        pack,
        [],
        [],
    )
    assert reason == "RELATIONSHIP_PATH_REQUIRED"
    assert not plan.intents
    assert requests and all(str(item.type) == "RELATIONSHIP" for item in requests)
    assert requests[0].request_key.startswith("asset_relationship:")


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
