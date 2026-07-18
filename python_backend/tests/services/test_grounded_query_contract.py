from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from merchant_ai.config import get_settings
from merchant_ai.services.assets import SemanticCatalogService, TopicAssetService
from merchant_ai.models import NodePlanContract
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContractBuilder,
    build_grounded_query_contract_from_refs,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.query import NodeWorkerExecutor
from merchant_ai.services.semantic_metrics import semantic_metric_temporal_contract_issue


def core_read(
    ref_id: str,
    kind: str,
    topic: str,
    table: str,
    payload: object,
) -> dict[str, object]:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "refId": ref_id,
        "kind": kind,
        "topic": topic,
        "table": table,
        "path": ref_id.replace("semantic:", "topics/").replace(":", "/") + ".json",
        "contentSnippet": content,
        "contentHash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def table_detail(topic: str, table: str, merchant_column: str = "merchant_id") -> dict[str, object]:
    return core_read(
        "semantic:%s:%s:detail" % (topic, table),
        "TABLE_DETAIL",
        topic,
        table,
        {
            "topic": topic,
            "tableName": table,
            "title": table,
            "dataGrain": "merchant_day_summary",
            "timeColumn": "pt",
            "merchantFilterColumn": merchant_column,
        },
    )


def metric_read(
    topic: str,
    table: str,
    metric_key: str,
    business_name: str,
    aliases: list[str],
    formula: str,
    source_columns: list[str],
    unit: str = "单",
    calculation_semantics: dict[str, object] | None = None,
    selection_policy: str = "period_window",
    applicable_time_grain: str = "period",
) -> dict[str, object]:
    return core_read(
        "semantic:%s:%s:metric:%s" % (topic, table, metric_key),
        "METRIC",
        topic,
        table,
        {
            "topic": topic,
            "tableName": table,
            "metric": {
                "metricKey": metric_key,
                "businessName": business_name,
                "aliases": aliases,
                "formula": formula,
                "sourceColumns": source_columns,
                "unit": unit,
                "metricGrain": "merchant_day_summary",
                "applicableTimeGrain": applicable_time_grain,
                "aggregationPolicy": "period_rollup",
                "timeColumn": "pt",
                "timeSemantics": {
                    "selectionPolicy": selection_policy,
                    "asOfPolicy": "latest_available_partition",
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
                },
                **(
                    {"calculationSemantics": calculation_semantics}
                    if calculation_semantics
                    else {}
                ),
            },
        },
    )


def column_read(
    topic: str,
    table: str,
    column: str,
    business_name: str,
    aliases: list[str],
    role: str = "DIMENSION",
    extra_definition: dict[str, object] | None = None,
) -> dict[str, object]:
    return core_read(
        "semantic:%s:%s:column:%s" % (topic, table, column),
        "COLUMN",
        topic,
        table,
        {
            "topic": topic,
            "tableName": table,
            "section": "columns",
            "key": column,
            "definition": {
                "columnName": column,
                "businessName": business_name,
                "description": business_name,
                "aliases": aliases,
                "role": role,
                **dict(extra_definition or {}),
            },
        },
    )


def entity_lookup_evidence() -> tuple[list[dict[str, object]], dict[str, object]]:
    order_topic = "电商交易"
    goods_topic = "商品管理"
    order_table = "fact_entity_detail"
    goods_table = "dim_related_entity"
    order_detail = table_detail(order_topic, order_table, merchant_column="seller_id")
    goods_detail = table_detail(goods_topic, goods_table, merchant_column="seller_id")
    entity_id = column_read(
        order_topic,
        order_table,
        "entity_id",
        "实体编号",
        ["实体ID"],
        role="ENTITY_UNIQUE_KEY",
        extra_definition={
            "isUniqueKey": True,
            "entityIdentity": "PRIMARY_ENTITY",
            "filterOperators": ["EQ", "IN"],
            "lookupTimePolicy": {"timeRequired": False, "mode": "IDENTITY_LOOKUP"},
        },
    )
    related_id = column_read(
        order_topic,
        order_table,
        "related_id",
        "关联实体编号",
        ["关联ID"],
        role="JOIN_KEY",
    )
    detail_status = column_read(
        order_topic,
        order_table,
        "detail_status",
        "明细状态",
        ["状态"],
    )
    published_at = column_read(
        goods_topic,
        goods_table,
        "published_at",
        "发布时间",
        ["发布于"],
        role="DATETIME",
    )
    relationship = core_read(
        "semantic:电商交易:relationships",
        "RELATIONSHIPS",
        order_topic,
        "",
        [
            {
                "name": "primary_to_related",
                "leftTable": order_table,
                "rightTable": goods_table,
                "joinType": "LEFT",
                "keys": [["seller_id", "seller_id"], ["related_id", "related_id"]],
                "grain": "primary_entity_related_entity",
                "cardinality": "MANY_TO_ONE",
                "fanoutPolicy": "PRESERVE_LEFT_GRAIN",
            },
            {
                "name": "related_to_primary_reverse",
                "leftTable": goods_table,
                "rightTable": order_table,
                "joinType": "LEFT",
                "keys": [["seller_id", "seller_id"], ["related_id", "related_id"]],
                "grain": "related_entity_primary_entity",
                "cardinality": "ONE_TO_MANY",
                "fanoutPolicy": "ALLOW_DECLARED_FANOUT",
            }
        ],
    )
    evidence = [
        order_detail,
        goods_detail,
        entity_id,
        related_id,
        detail_status,
        published_at,
        relationship,
    ]
    hints = {
        "tableRefs": [order_detail["refId"], goods_detail["refId"]],
        "selectedFieldRefs": [
            entity_id["refId"],
            related_id["refId"],
            detail_status["refId"],
            published_at["refId"],
        ],
        "entityFilters": [
            {
                "fieldRef": entity_id["refId"],
                "operator": "EQ",
                "literalValue": "entity_100",
                "requestedPhrase": "实体 entity_100",
            }
        ],
        "relationshipRefs": [relationship["refId"]],
        "analysisMode": "entity_lookup",
    }
    return evidence, hints


def test_builds_ready_same_table_multi_metric_contract_from_core_reads_only() -> None:
    topic = "经营画像"
    table = "ads_merchant_profile"
    evidence = [
        table_detail(topic, table),
        metric_read(
            topic,
            table,
            "order_cnt_1d",
            "总订单日汇总量",
            ["订单量", "订单数", "总订单数"],
            "SUM(order_cnt_1d)",
            ["order_cnt_1d"],
        ),
        metric_read(
            topic,
            table,
            "refund_amt_1d",
            "退款日汇总金额",
            ["退款金额", "退款额"],
            "SUM(refund_amt_1d)",
            ["refund_amt_1d"],
            unit="元",
        ),
    ]

    contract = GroundedQueryContractBuilder().build(
        "只查询最近30天的订单数和退款总额",
        [topic],
        evidence,
        binding_hints={
            "tableRefs": [evidence[0]["refId"]],
            "metricRefs": [evidence[1]["refId"], evidence[2]["refId"]],
            "labelRefs": {
                evidence[1]["refId"]: "订单数",
                evidence[2]["refId"]: "退款总额",
            },
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.ready is True
    assert contract.status == "READY"
    assert contract.execution_shape == "same_table_multi_metric"
    assert contract.primary_table == table
    assert {metric.metric_key for metric in contract.metrics} == {"order_cnt_1d", "refund_amt_1d"}
    assert {metric.table for metric in contract.metrics} == {table}
    assert contract.time_range.days == 30
    assert contract.time_range.anchor_policy == "latest_available_partition"
    assert contract.relationships == []
    assert set(contract.evidence_refs) == {str(item["refId"]) for item in evidence}
    assert contract.unresolved_gaps == []

    assets = TopicAssetService(get_settings())
    pack = materialize_grounded_asset_pack(contract, assets)
    assert pack.known_tables() == [table]
    assert {metric.key for metric in pack.metrics} == {"order_cnt_1d", "refund_amt_1d"}
    assert set(pack.known_columns(table)) == {"merchant_id", "pt", "order_cnt_1d", "refund_amt_1d"}
    published = assets.load_table_asset(topic, table)
    assert pack.tables[0].metadata["rowAccessPolicy"] == published["rowAccessPolicy"]
    assert pack.tables[0].metadata["resultAccessPolicies"] == published["resultAccessPolicies"]
    assert pack.tables[0].metadata["tableUsageProfile"] == {
        "contractStatus": published["tableUsageProfile"]["contractStatus"],
        "queryableByAgent": published["tableUsageProfile"]["queryableByAgent"],
    }
    assert "businessSummary" not in pack.tables[0].metadata
    assert "businessLayer" not in pack.tables[0].metadata["tableUsageProfile"]
    assert "topicRole" not in pack.tables[0].metadata["tableUsageProfile"]
    assert pack.tables[0].metadata["status"] == published["status"]
    assert pack.tables[0].metadata["version"] == published["version"]

    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    assert len(preparation.plan.intents) == 1
    assert preparation.plan.intents[0].group_by_column == ""
    assert {item["metricName"] for item in preparation.plan.intents[0].metric_specs} == {
        "order_cnt_1d",
        "refund_amt_1d",
    }
    assert "planner_llm_calls=0" in preparation.plan.agent_trace
    assert "GROUNDED_DIRECT_COMPILE:same_table_multi_metric" in preparation.plan.compiler_trace
    assert not any(
        item.startswith("execution_optimizer.same_table_metric_merge:")
        for item in preparation.plan.compiler_trace
    )
    intent = preparation.plan.intents[0]
    assert semantic_metric_temporal_contract_issue(intent.metric_resolution) == ""
    assert all(
        semantic_metric_temporal_contract_issue(spec) == ""
        for spec in intent.metric_specs
    )


def test_builds_typed_two_table_entity_lookup_without_metric_surrogate() -> None:
    evidence, hints = entity_lookup_evidence()
    contract = GroundedQueryContractBuilder().build(
        "查询实体 entity_100 的明细，再看关联对象什么时候发布",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.ready is True, [gap.model_dump() for gap in contract.unresolved_gaps]
    assert contract.query_shape == "ENTITY_LOOKUP"
    assert contract.execution_shape == "detail_join"
    assert contract.metrics == []
    assert [item.column for item in contract.selected_fields] == [
        "entity_id",
        "related_id",
        "detail_status",
        "published_at",
    ]
    assert contract.entity_filters[0].column == "entity_id"
    assert contract.entity_filters[0].operator == "EQ"
    assert contract.entity_filters[0].literal_value == "entity_100"
    assert contract.entity_filters[0].is_unique_key is True
    assert contract.entity_filters[0].lookup_time_policy["timeRequired"] is False
    assert contract.time_range.source == "default_days"
    assert contract.relationships[0].cardinality == "MANY_TO_ONE"
    assert contract.relationships[0].fanout_policy == "PRESERVE_LEFT_GRAIN"

    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [
        gap.model_dump() for gap in preparation.validation.gaps
    ]
    intent = preparation.plan.intents[0]
    assert intent.answer_mode == "DETAIL"
    assert intent.metric_name == ""
    assert intent.metric_formula == ""
    assert intent.metric_specs == []
    assert intent.filter_column == "entity_id"
    assert intent.filter_value == "entity_100"
    assert intent.output_keys == [
        "entity_id",
        "related_id",
        "detail_status",
        "published_at",
    ]


def test_order_detail_question_is_bound_as_generic_entity_lookup_not_metric() -> None:
    evidence, hints = entity_lookup_evidence()
    hints["entityFilters"][0]["literalValue"] = "order_id_100"
    hints["entityFilters"][0]["requestedPhrase"] = "订单 order_id_100"
    contract = GroundedQueryContractBuilder().build(
        "查询订单 order_id_100 的订单明细，再看对应商品什么时候发布",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.ready is True
    assert contract.query_shape == "ENTITY_LOOKUP"
    assert contract.metrics == []
    assert contract.entity_filters[0].literal_value == "order_id_100"
    assert contract.entity_filters[0].entity_identity == "PRIMARY_ENTITY"


def test_real_progressive_assets_compile_order_to_product_lookup_without_time_filter() -> None:
    settings = get_settings()
    assets = TopicAssetService(settings)
    catalog = SemanticCatalogService(assets)
    refs = [
        "semantic:电商交易:dwm_trade_order_detail_di:detail",
        "semantic:商品管理:dwm_goods_detail_df:detail",
        "semantic:电商交易:dwm_trade_order_detail_di:field:order_id",
        "semantic:电商交易:dwm_trade_order_detail_di:field:sub_order_id",
        "semantic:电商交易:dwm_trade_order_detail_di:field:spu_id",
        "semantic:商品管理:dwm_goods_detail_df:field:spu_apply_create_time",
        "semantic:商品管理:relationship:order_goods_by_spu_id",
    ]
    contract = build_grounded_query_contract_from_refs(
        "查询订单 order_id_100 的订单明细，再看对应商品什么时候发布",
        ["电商交易", "商品管理"],
        refs,
        catalog,
        binding_hints={
            "tableRefs": refs[:2],
            "selectedFields": [
                {"fieldRef": refs[2], "outputAlias": "order_id"},
                {"fieldRef": refs[3], "outputAlias": "sub_order_id"},
                {"fieldRef": refs[4], "outputAlias": "spu_id"},
                {
                    "fieldRef": refs[5],
                    "outputAlias": "spu_apply_create_time",
                },
            ],
            "entityFilters": [
                {
                    "fieldRef": refs[2],
                    "operator": "EQ",
                    "literalValue": "order_id_100",
                    "requestedPhrase": "订单 order_id_100",
                }
            ],
            "relationshipRefs": [refs[6]],
            "analysisMode": "ENTITY_LOOKUP",
        },
    )

    assert contract.ready is True, [
        gap.model_dump() for gap in contract.unresolved_gaps
    ]
    assert contract.metrics == []
    assert contract.time_range.source == "default_days"
    assert contract.time_range.explicit is False
    assert [item.name for item in contract.relationships] == [
        "order_goods_by_spu_id"
    ]
    assert contract.entity_filters[0].lookup_time_policy["mode"] == "unbounded"

    pack = materialize_grounded_asset_pack(contract, assets)
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [
        gap.model_dump() for gap in preparation.validation.gaps
    ]


def test_label_refs_cannot_substitute_for_typed_entity_filter() -> None:
    evidence, hints = entity_lookup_evidence()
    entity_filter = hints.pop("entityFilters")
    hints["labelRefs"] = {
        entity_filter[0]["fieldRef"]: "entity_100",
    }
    contract = GroundedQueryContractBuilder().build(
        "查询实体 entity_100 的明细",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.ready is False
    assert contract.entity_filters == []
    assert "ENTITY_FILTER_REQUIRED" in {gap.code for gap in contract.unresolved_gaps}


def test_single_table_detail_list_uses_typed_projection_and_explicit_time() -> None:
    topic = "通用明细"
    table = "fact_activity_detail"
    detail = table_detail(topic, table, merchant_column="tenant_id")
    activity_id = column_read(
        topic,
        table,
        "activity_id",
        "活动编号",
        ["活动ID"],
        role="ENTITY_KEY",
    )
    activity_status = column_read(
        topic,
        table,
        "activity_status",
        "活动状态",
        ["状态"],
    )
    contract = GroundedQueryContractBuilder().build(
        "查看最近30天活动明细",
        [topic],
        [detail, activity_id, activity_status],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "selectedFieldRefs": [activity_id["refId"], activity_status["refId"]],
            "analysisMode": "detail",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is True
    assert contract.query_shape == "DETAIL"
    assert contract.metrics == []
    assert contract.entity_filters == []
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid
    assert preparation.plan.intents[0].answer_mode == "DETAIL"


def test_relationship_endpoint_gap_drives_generic_binding_expansion() -> None:
    evidence, hints = entity_lookup_evidence()
    goods_detail_ref = hints["tableRefs"].pop()
    hints["selectedFieldRefs"] = hints["selectedFieldRefs"][:-1]
    evidence = [item for item in evidence if item["refId"] != goods_detail_ref]
    contract = GroundedQueryContractBuilder().build(
        "查询实体 entity_100 的明细并查看关联对象",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.status == "REVISE_BINDINGS"
    gap = next(
        item
        for item in contract.unresolved_gaps
        if item.code == "RELATIONSHIP_ENDPOINT_TABLE_BINDING_REQUIRED"
    )
    assert gap.search_scope == "READ_BINDINGS_THEN_TABLE_MANIFEST_THEN_TOPIC_INDEX"
    assert gap.required_capability == {
        "endpointTable": "dim_related_entity",
        "relationshipRef": "semantic:电商交易:relationships",
        "requiredSemanticRole": "RELATIONSHIP_ENDPOINT_TABLE",
    }


def test_detail_join_fails_closed_without_declared_fanout_policy() -> None:
    evidence, hints = entity_lookup_evidence()
    relationship = evidence[-1]
    payload = json.loads(str(relationship["contentSnippet"]))
    payload[0].pop("fanoutPolicy")
    replacement = core_read(
        str(relationship["refId"]),
        "RELATIONSHIPS",
        str(relationship["topic"]),
        "",
        payload,
    )
    evidence[-1] = replacement
    contract = GroundedQueryContractBuilder().build(
        "查询实体 entity_100 的明细和关联对象",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.ready is False
    assert "RELATIONSHIP_FANOUT_POLICY_REQUIRED" in {
        gap.code for gap in contract.unresolved_gaps
    }


def test_detail_contract_preserves_competing_relationships_for_sql_ast_proof() -> None:
    evidence, hints = entity_lookup_evidence()
    relationship = evidence[-1]
    payload = json.loads(str(relationship["contentSnippet"]))
    competing = dict(payload[0])
    competing["name"] = "primary_to_related_competing"
    payload.append(competing)
    evidence[-1] = core_read(
        str(relationship["refId"]),
        "RELATIONSHIPS",
        str(relationship["topic"]),
        "",
        payload,
    )
    contract = GroundedQueryContractBuilder().build(
        "查询实体 entity_100 的明细和关联对象",
        ["电商交易", "商品管理"],
        evidence,
        binding_hints=hints,
    )

    assert contract.status == "READY"
    assert contract.ready is True
    assert [item.name for item in contract.relationships] == [
        "primary_to_related",
        "related_to_primary_reverse",
        "primary_to_related_competing",
    ]


def test_model_style_binding_aliases_normalize_to_typed_contract_fields() -> None:
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail = table_detail(topic, table)
    order_metric = metric_read(
        topic,
        table,
        "order_cnt_1d",
        "总订单日汇总量",
        ["订单数"],
        "SUM(order_cnt_1d)",
        ["order_cnt_1d"],
    )
    refund_metric = metric_read(
        topic,
        table,
        "refund_amt_1d",
        "退款日汇总金额",
        ["退款总额"],
        "SUM(refund_amt_1d)",
        ["refund_amt_1d"],
        unit="元",
    )

    contract = GroundedQueryContractBuilder().build(
        "最近30天的订单数和退款总额是多少？",
        [topic],
        [detail, order_metric, refund_metric],
        binding_hints={
            "tableRef": detail["refId"],
            "metricBindings": [
                {"refId": order_metric["refId"], "phrase": "订单数"},
                {"metricRefId": refund_metric["refId"], "phrase": "退款总额"},
            ],
            "timeWindow": {"userPhrase": "最近30天"},
            "intent": "METRIC_SUMMARY",
        },
    )

    assert contract.ready is True
    assert contract.binding_hints.table_refs == [detail["refId"]]
    assert contract.binding_hints.metric_refs == [
        order_metric["refId"],
        refund_metric["refId"],
    ]
    assert contract.binding_hints.time_expression == "最近30天"
    assert contract.binding_hints.analysis_mode == "metric_total"
    assert contract.query_shape == "SCALAR"


def test_metric_binding_preserves_user_phrase_when_model_omits_label_refs() -> None:
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail = table_detail(topic, table)
    order_metric = metric_read(
        topic,
        table,
        "order_cnt_1d",
        "总订单日汇总量",
        ["订单数", "订单量"],
        "SUM(order_cnt_1d)",
        ["order_cnt_1d"],
    )
    refund_metric = metric_read(
        topic,
        table,
        "refund_amt_1d",
        "退款日汇总金额",
        ["退款总额", "退款额"],
        "SUM(refund_amt_1d)",
        ["refund_amt_1d"],
        unit="元",
    )

    contract = GroundedQueryContractBuilder().build(
        "最近30天的订单数和退款总额是多少？",
        [topic],
        [detail, order_metric, refund_metric],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [order_metric["refId"], refund_metric["refId"]],
            "timeExpression": "最近30天",
        },
    )

    assert [metric.requested_phrase for metric in contract.metrics] == [
        "订单数",
        "退款总额",
    ]


def test_scalar_limit_does_not_create_ranking_or_expose_merchant_scope_column() -> None:
    topic = "电商交易"
    table = "dwm_trade_order_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    metric = metric_read(
        topic,
        table,
        "order_cnt",
        "订单量",
        ["订单数"],
        "COUNT(DISTINCT order_id)",
        ["order_id"],
    )

    contract = GroundedQueryContractBuilder().build(
        "最近30天订单量",
        [topic],
        [detail, metric],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            # Compatibility callers may still send an execution limit.  It is
            # not a semantic ranking declaration and must not affect shape.
            "ranking": {"limit": 1},
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is True
    assert contract.query_shape == "SCALAR"
    assert contract.ranking.enabled is False
    assert contract.ranking.limit == 0

    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    intent = preparation.plan.intents[0]
    assert intent.answer_mode == "METRIC"
    assert intent.group_by_column == ""
    assert intent.output_keys == []
    assert "seller_id" not in intent.required_evidence


def test_merchant_scope_column_cannot_be_bound_as_group_dimension() -> None:
    topic = "电商交易"
    table = "dwm_trade_order_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    metric = metric_read(
        topic,
        table,
        "order_cnt",
        "订单量",
        ["订单数"],
        "COUNT(DISTINCT order_id)",
        ["order_id"],
    )
    seller = column_read(topic, table, "seller_id", "商家id", ["商家"], role="KEY")

    contract = GroundedQueryContractBuilder().build(
        "最近30天按商家看订单量",
        [topic],
        [detail, metric, seller],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "dimensionRefs": [seller["refId"]],
            "groupByRef": seller["refId"],
            "analysisMode": "grouped_metric",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is False
    assert "MERCHANT_SCOPE_DIMENSION_FORBIDDEN" in {
        gap.code for gap in contract.unresolved_gaps
    }


def test_builds_ranked_product_dimension_contract_for_ticket_metric() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    evidence = [
        table_detail(topic, table, merchant_column="seller_id"),
        metric_read(
            topic,
            table,
            "ticket_cnt",
            "客服工单明细量",
            ["客服工单明细数", "工单明细量", "按商品客服工单明细量"],
            "COUNT(DISTINCT ticket_id)",
            ["ticket_id"],
        ),
        column_read(topic, table, "spu_id", "商品id", ["商品", "spu_id"], role="KEY"),
    ]

    contract = GroundedQueryContractBuilder().build(
        "最近30天工单量最多的商品",
        [topic],
        evidence,
        binding_hints={
            "tableRefs": [evidence[0]["refId"]],
            "metricRefs": [evidence[1]["refId"]],
            "dimensionRefs": [evidence[2]["refId"]],
            "groupByRef": evidence[2]["refId"],
            "labelRefs": {
                evidence[1]["refId"]: "工单量",
                evidence[2]["refId"]: "商品",
            },
            "ranking": {"order": "desc", "limit": 1},
            "analysisMode": "topn",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.ready is True
    assert contract.execution_shape == "ranked_group"
    assert [metric.metric_key for metric in contract.metrics] == ["ticket_cnt"]
    assert [dimension.column for dimension in contract.dimensions] == ["spu_id"]
    assert contract.ranking.enabled is True
    assert contract.ranking.direction == "DESC"
    assert contract.ranking.limit == 1
    assert contract.ranking.metric_ref_id == contract.metrics[0].semantic_ref_id
    assert contract.ranking.dimension_ref_id == contract.dimensions[0].semantic_ref_id
    assert contract.time_range.days == 30
    assert contract.time_range.source == "relative_day_quantity"

    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    assert len(preparation.plan.intents) == 1
    assert preparation.plan.intents[0].answer_mode == "TOPN"
    assert preparation.plan.intents[0].group_by_column == "spu_id"
    assert preparation.plan.intents[0].limit == 1


def test_ticket_ranking_uses_product_id_without_ungoverned_name_label() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    metric = metric_read(
        topic,
        table,
        "ticket_cnt",
        "客服工单明细量",
        ["工单量"],
        "COUNT(DISTINCT ticket_id)",
        ["ticket_id"],
    )
    product_id = column_read(topic, table, "spu_id", "商品id", ["商品id"], role="KEY")
    product_name = column_read(topic, table, "spu_name", "商品名称", ["商品名称"])
    contract = GroundedQueryContractBuilder().build(
        "最近30天工单量最多的商品",
        [topic],
        [detail, metric, product_id, product_name],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "dimensionRefs": [product_id["refId"], product_name["refId"]],
            "groupByRef": product_id["refId"],
            "labelRefs": {
                metric["refId"]: "工单量",
                product_id["refId"]: "商品",
                product_name["refId"]: "商品名称",
            },
            "ranking": {"order": "desc", "limit": 10},
            "analysisMode": "topn",
        },
    )
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)

    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    intent = preparation.plan.intents[0]
    assert intent.group_by_column == "spu_id"
    assert "spu_name" not in intent.output_keys
    assert "labelColumns" not in intent.metric_resolution

    node_contract = NodePlanContract(
        task_id=intent.plan_task_id,
        preferred_table=table,
        visible_columns=["spu_id", "spu_name", "ticket_id"],
        metric_name=intent.metric_name,
        metric_formula=intent.metric_formula,
        metric_specs=intent.metric_specs,
        group_by_column="spu_id",
        output_keys=intent.output_keys,
        required_evidence=intent.required_evidence,
    )
    worker = object.__new__(NodeWorkerExecutor)
    sql = worker._draft_structured_aggregate_sql(
        intent,
        table,
        {"spu_id", "spu_name", "ticket_id"},
        "",
        node_contract,
    )

    assert "GROUP BY `spu_id`" in sql
    assert "`spu_name`" not in sql
    assert "GROUP BY `spu_id`, `spu_name`" not in sql


def test_ranked_multi_metric_uses_only_explicit_sort_metric_as_anchor() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    amount = metric_read(
        topic,
        table,
        "ticket_amt",
        "工单金额",
        ["金额"],
        "SUM(ticket_amt)",
        ["ticket_amt"],
        unit="元",
    )
    count = metric_read(
        topic,
        table,
        "ticket_cnt",
        "工单量",
        ["工单数"],
        "COUNT(DISTINCT ticket_id)",
        ["ticket_id"],
    )
    product = column_read(topic, table, "spu_id", "商品id", ["商品"], role="KEY")
    contract = GroundedQueryContractBuilder().build(
        "最近30天工单量最多的商品，同时返回工单金额",
        [topic],
        [detail, amount, count, product],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [amount["refId"], count["refId"]],
            "dimensionRefs": [product["refId"]],
            "groupByRef": product["refId"],
            "ranking": {
                "metricRef": count["refId"],
                "order": "desc",
                "limit": 10,
            },
            "analysisMode": "topn",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is True
    assert contract.query_shape == "RANKED"
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    intent = preparation.plan.intents[0]
    assert intent.metric_name == "ticket_cnt"
    assert [spec["metricName"] for spec in intent.metric_specs] == [
        "ticket_cnt",
        "ticket_amt",
    ]


def test_governed_field_count_distinct_compiles_without_published_metric() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    buyer = column_read(topic, table, "buyer_id", "买家id", ["买家", "买家id"], role="KEY")

    contract = GroundedQueryContractBuilder().build(
        "最近30天涉及多少个买家",
        [topic],
        [detail, buyer],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "fieldAggregations": [
                {
                    "fieldRef": buyer["refId"],
                    "aggregation": "COUNT_DISTINCT",
                    "requestedPhrase": "最近30天买家数",
                }
            ],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is True
    assert contract.execution_shape == "single_metric"
    assert len(contract.metrics) == 1
    metric = contract.metrics[0]
    assert metric.binding_type == "field_aggregation"
    assert metric.field_aggregation == "COUNT_DISTINCT"
    assert metric.source_field_ref_id == buyer["refId"]
    assert metric.metric_key == "count_distinct_buyer_id"
    assert metric.formula == "COUNT(DISTINCT `buyer_id`)"
    assert metric.source_columns == ["buyer_id"]
    assert metric.requested_phrase == "买家数"
    assert set(contract.evidence_refs) == {str(detail["refId"]), str(buyer["refId"])}

    pack = materialize_grounded_asset_pack(contract, TopicAssetService(get_settings()))
    assert len(pack.metrics) == 1
    assert pack.metrics[0].source_ref_id.startswith("grounded-field-aggregation:")
    assert pack.metrics[0].metadata["sourceFieldRefId"] == buyer["refId"]

    preparation = compile_grounded_query(contract, pack)
    assert preparation.validation.valid, [gap.model_dump() for gap in preparation.validation.gaps]
    assert len(preparation.plan.intents) == 1
    intent = preparation.plan.intents[0]
    assert intent.metric_name == "count_distinct_buyer_id"
    assert intent.metric_formula == "COUNT(DISTINCT `buyer_id`)"
    assert len(intent.metric_specs) == 1
    assert intent.metric_specs[0]["metricName"] == "count_distinct_buyer_id"
    assert intent.metric_specs[0]["metricFormula"] == "COUNT(DISTINCT `buyer_id`)"
    assert intent.metric_specs[0]["sourceColumns"] == ["buyer_id"]
    assert intent.metric_specs[0]["sourceFieldRefId"] == buyer["refId"]

    node_contract = NodePlanContract(
        task_id=intent.plan_task_id,
        preferred_table=table,
        allowed_columns=["seller_id", "pt", "buyer_id"],
        visible_columns=["buyer_id"],
        metric_name=intent.metric_name,
        metric_formula=intent.metric_formula,
        metric_specs=intent.metric_specs,
        metric_resolution=intent.metric_resolution,
    )
    worker = object.__new__(NodeWorkerExecutor)
    sql = worker._draft_structured_aggregate_sql(
        intent,
        table,
        {"seller_id", "pt", "buyer_id"},
        "",
        node_contract,
    )
    assert sql == (
        "SELECT COUNT(DISTINCT `buyer_id`) AS `count_distinct_buyer_id` "
        "FROM `dwm_cs_ticket_detail_di`"
    )


def test_field_aggregation_rejects_non_allowlisted_sum() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    buyer = column_read(topic, table, "buyer_id", "买家id", ["买家", "买家id"], role="KEY")

    contract = GroundedQueryContractBuilder().build(
        "最近30天买家字段求和",
        [topic],
        [detail, buyer],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "fieldAggregations": [
                {
                    "fieldRef": buyer["refId"],
                    "aggregation": "SUM",
                    "requestedPhrase": "买家字段求和",
                }
            ],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert contract.ready is False
    assert contract.metrics == []
    assert "FIELD_AGGREGATION_UNSUPPORTED" in {gap.code for gap in contract.unresolved_gaps}


def test_semantic_usage_policy_rejects_non_composable_period_rollup_without_phrase_rules() -> None:
    topic = "TEST_TOPIC"
    table = "daily_entity_summary"
    detail = table_detail(topic, table)
    metric = metric_read(
        topic,
        table,
        "daily_entity_count",
        "每日主体数",
        ["主体数量"],
        "SUM(daily_entity_count)",
        ["daily_entity_count"],
        calculation_semantics={
            "nativeTimeGrain": "DAY",
            "nativeWindowDays": 1,
            "timeRollupPolicy": "NOT_COMPOSABLE",
            "nativeGrainAnalysisModes": ["TREND", "TIME_SERIES"],
            "forbiddenAggregations": ["SUM"],
            "alternativeCapability": {
                "operation": "COUNT_DISTINCT",
                "requiredFieldRole": "KEY",
                "requiredTableGrain": "EVENT_DETAIL",
            },
        },
    )

    contract = GroundedQueryContractBuilder().build(
        "最近30天主体数量",
        [topic],
        [detail, metric],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert contract.status == "REVISE_BINDINGS"
    assert contract.ready is False
    gap = next(item for item in contract.unresolved_gaps if item.code == "TABLE_INSUFFICIENT")
    assert gap.required_capability == {
        "operation": "COUNT_DISTINCT",
        "requiredFieldRole": "KEY",
        "requiredTableGrain": "EVENT_DETAIL",
    }
    assert contract.rejected_bindings[0].table == table


def test_non_composable_daily_metric_remains_valid_at_native_trend_grain() -> None:
    topic = "TEST_TOPIC"
    table = "daily_entity_summary"
    detail = table_detail(topic, table)
    metric = metric_read(
        topic,
        table,
        "daily_entity_count",
        "每日主体数",
        ["主体数量"],
        "SUM(daily_entity_count)",
        ["daily_entity_count"],
        calculation_semantics={
            "nativeTimeGrain": "DAY",
            "nativeWindowDays": 1,
            "timeRollupPolicy": "NOT_COMPOSABLE",
            "nativeGrainAnalysisModes": ["TREND", "TIME_SERIES"],
            "forbiddenAggregations": ["SUM"],
        },
        selection_policy="per_time_grain",
        applicable_time_grain="day",
    )
    time_dimension = column_read(topic, table, "pt", "日期", ["日期", "时间"], role="TIME")

    contract = GroundedQueryContractBuilder().build(
        "最近30天每日主体趋势",
        [topic],
        [detail, metric, time_dimension],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "dimensionRefs": [time_dimension["refId"]],
            "groupByRef": time_dimension["refId"],
            "analysisMode": "trend",
            "timeExpression": "最近30天",
        },
    )

    assert contract.status == "READY"
    assert contract.ready is True
    assert contract.query_shape == "TREND"


def test_semantic_usage_policy_enforces_fixed_windows_and_required_components() -> None:
    topic = "TEST_TOPIC"
    table = "semantic_summary"
    detail = table_detail(topic, table)
    fixed = metric_read(
        topic,
        table,
        "fixed_window_metric",
        "固定窗口指标",
        [],
        "MAX(fixed_window_metric)",
        ["fixed_window_metric"],
        calculation_semantics={
            "nativeWindowDays": 30,
            "windowPolicy": "EXACT_ONLY",
            "alternativeCapability": {"windowPolicy": "ARBITRARY_PERIOD_RECOMPUTE"},
        },
    )
    weighted = metric_read(
        topic,
        table,
        "weighted_metric",
        "加权指标",
        [],
        "SUM(value_sum) / SUM(weight_sum)",
        ["value_sum"],
        calculation_semantics={
            "timeRollupPolicy": "WEIGHTED_AVERAGE",
            "requiredComponents": ["value_sum", "weight_sum"],
            "requiredWeightRef": "weight_sum",
            "alternativeCapability": {"requiredComponents": ["value_sum", "weight_sum"]},
        },
    )

    fixed_contract = GroundedQueryContractBuilder().build(
        "最近45天固定窗口指标",
        [topic],
        [detail, fixed],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [fixed["refId"]],
            "analysisMode": "metric_total",
            "timeExpression": "最近45天",
        },
    )
    weighted_contract = GroundedQueryContractBuilder().build(
        "最近30天加权指标",
        [topic],
        [detail, weighted],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [weighted["refId"]],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert fixed_contract.status == "REVISE_BINDINGS"
    assert "WINDOW_MISMATCH" in fixed_contract.unresolved_gaps[0].message
    assert weighted_contract.status == "REVISE_BINDINGS"
    assert "REQUIRED_COMPONENTS_MISSING:weight_sum" in weighted_contract.unresolved_gaps[0].message


def test_period_metric_without_explicit_time_fails_closed_for_clarification() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    metric = metric_read(
        topic,
        table,
        "ticket_cnt",
        "客服工单明细量",
        ["工单量"],
        "COUNT(DISTINCT ticket_id)",
        ["ticket_id"],
    )
    product_id = column_read(topic, table, "spu_id", "商品id", ["商品id"], role="KEY")

    contract = GroundedQueryContractBuilder().build(
        "工单量最多的商品",
        [topic],
        [detail, metric, product_id],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "dimensionRefs": [product_id["refId"]],
            "groupByRef": product_id["refId"],
            "ranking": {"order": "desc", "limit": 1},
            "analysisMode": "topn",
        },
    )

    assert contract.ready is False
    assert contract.time_range.source == "default_days"
    assert {gap.code for gap in contract.unresolved_gaps} == {"TIME_RANGE_REQUIRED"}


def test_column_ref_alias_resolves_only_when_canonical_field_was_read() -> None:
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail = table_detail(topic, table, merchant_column="seller_id")
    metric = metric_read(
        topic,
        table,
        "ticket_cnt",
        "客服工单明细量",
        ["工单量"],
        "COUNT(DISTINCT ticket_id)",
        ["ticket_id"],
    )
    product_id = column_read(topic, table, "spu_id", "商品id", ["商品id"], role="KEY")
    canonical_ref = str(product_id["refId"]).replace(":column:", ":field:")
    product_id["refId"] = canonical_ref

    contract = GroundedQueryContractBuilder().build(
        "最近30天工单量最多的商品",
        [topic],
        [detail, metric, product_id],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [metric["refId"]],
            "dimensionRefs": [canonical_ref.replace(":field:", ":column:")],
            "groupByRef": canonical_ref.replace(":field:", ":column:"),
            "ranking": {"order": "desc", "limit": 1},
            "analysisMode": "topn",
        },
    )

    assert contract.ready is True
    assert contract.binding_hints.group_by_ref == canonical_ref
    assert contract.dimensions[0].semantic_ref_id == canonical_ref


def test_cross_table_binding_fails_closed_until_relationship_is_read() -> None:
    order_topic = "电商交易"
    goods_topic = "商品管理"
    order_table = "dwm_trade_order_detail_di"
    goods_table = "dwm_goods_detail_df"
    base_evidence = [
        table_detail(order_topic, order_table, merchant_column="seller_id"),
        table_detail(goods_topic, goods_table, merchant_column="seller_id"),
        metric_read(
            order_topic,
            order_table,
            "order_detail_cnt",
            "商品订单量",
            ["订单数", "按商品订单量"],
            "COUNT(DISTINCT sub_order_id)",
            ["sub_order_id"],
        ),
        column_read(goods_topic, goods_table, "spu_id", "商品id", ["商品", "spu_id"], role="KEY"),
    ]
    builder = GroundedQueryContractBuilder()
    base_hints = {
        "tableRefs": [base_evidence[0]["refId"], base_evidence[1]["refId"]],
        "metricRefs": [base_evidence[2]["refId"]],
        "dimensionRefs": [base_evidence[3]["refId"]],
        "groupByRef": base_evidence[3]["refId"],
        "labelRefs": {
            base_evidence[2]["refId"]: "订单数",
            base_evidence[3]["refId"]: "商品",
        },
        "analysisMode": "grouped_metric",
    }

    unresolved = builder.build(
        "最近30天按商品查看订单数",
        [order_topic, goods_topic],
        base_evidence,
        binding_hints=base_hints,
    )

    assert unresolved.ready is False
    assert "RELATIONSHIP_EVIDENCE_REQUIRED" in {gap.code for gap in unresolved.unresolved_gaps}

    relationship = core_read(
        "semantic:商品管理:relationships",
        "RELATIONSHIPS",
        goods_topic,
        "",
        [
            {
                "name": "goods_order_by_spu_id",
                "leftTable": goods_table,
                "rightTable": order_table,
                "joinType": "LEFT",
                "keys": [["seller_id", "seller_id"], ["spu_id", "spu_id"]],
                "grain": "spu_id_sub_order_id",
                "cardinality": "ONE_TO_MANY",
                "fanoutPolicy": "ALLOW_DECLARED_FANOUT",
            }
        ],
    )
    resolved = builder.build(
        "最近30天按商品查看订单数",
        [order_topic, goods_topic],
        [*base_evidence, relationship],
        binding_hints={
            **base_hints,
            "relationshipRefs": [relationship["refId"]],
        },
    )

    assert resolved.ready is True
    assert resolved.execution_shape == "multi_table"
    assert [item.name for item in resolved.relationships] == ["goods_order_by_spu_id"]
    assert str(relationship["refId"]) in resolved.evidence_refs


def test_build_from_refs_resolves_only_core_selected_semantic_files() -> None:
    assets = TopicAssetService(get_settings())
    catalog = SemanticCatalogService(assets)
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail_ref = "semantic:%s:%s:detail" % (topic, table)
    order_ref = "semantic:%s:%s:metric:order_cnt_1d" % (topic, table)
    refund_ref = "semantic:%s:%s:metric:refund_amt_1d" % (topic, table)

    contract = build_grounded_query_contract_from_refs(
        "只查询最近30天的订单数和退款总额",
        [topic],
        [detail_ref, order_ref, refund_ref],
        catalog,
        binding_hints={
            "tableRefs": [detail_ref],
            "metricRefs": [order_ref, refund_ref],
            "labelRefs": {order_ref: "订单数", refund_ref: "退款总额"},
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.ready is True
    assert contract.evidence_refs == [detail_ref, order_ref, refund_ref]
    assert [metric.metric_key for metric in contract.metrics] == ["order_cnt_1d", "refund_amt_1d"]


def test_real_daily_user_metric_policy_returns_revise_bindings_for_period_total() -> None:
    assets = TopicAssetService(get_settings())
    catalog = SemanticCatalogService(assets)
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail_ref = "semantic:%s:%s:detail" % (topic, table)
    metric_ref = "semantic:%s:%s:metric:order_user_cnt_1d" % (topic, table)

    contract = build_grounded_query_contract_from_refs(
        "最近30天下单用户量",
        [topic],
        [detail_ref, metric_ref],
        catalog,
        binding_hints={
            "tableRefs": [detail_ref],
            "metricRefs": [metric_ref],
            "labelRefs": {metric_ref: "下单用户量"},
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
    )

    assert contract.status == "REVISE_BINDINGS"
    gap = next(item for item in contract.unresolved_gaps if item.code == "TABLE_INSUFFICIENT")
    assert gap.required_capability["operation"] == "COUNT_DISTINCT"
    assert gap.required_capability["entityRole"] == "BUYER"


def test_build_from_refs_accepts_exact_field_ref_for_count_distinct() -> None:
    assets = TopicAssetService(get_settings())
    catalog = SemanticCatalogService(assets)
    topic = "客服工单"
    table = "dwm_cs_ticket_detail_di"
    detail_ref = "semantic:%s:%s:detail" % (topic, table)
    buyer_ref = "semantic:%s:%s:field:buyer_id" % (topic, table)

    contract = build_grounded_query_contract_from_refs(
        "最近30天涉及多少个买家",
        [topic],
        [detail_ref, buyer_ref],
        catalog,
        binding_hints={
            "tableRefs": [detail_ref],
            "fieldAggregations": [
                {
                    "fieldRef": buyer_ref,
                    "aggregation": "COUNT_DISTINCT",
                    "requestedPhrase": "买家数",
                }
            ],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.ready is True
    assert contract.evidence_refs == [detail_ref, buyer_ref]
    assert contract.metrics[0].source_field_ref_id == buyer_ref
    assert contract.metrics[0].formula == "COUNT(DISTINCT `buyer_id`)"


def test_mixed_metric_time_policies_require_revised_bindings_before_compile() -> None:
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail = table_detail(topic, table)
    period_metric = metric_read(
        topic,
        table,
        "refund_rate_by_pay_order",
        "周期退款率",
        ["退款率"],
        "SUM(return_cnt_1d) / NULLIF(SUM(pay_order_cnt_1d), 0)",
        ["return_cnt_1d", "pay_order_cnt_1d"],
        unit="%",
    )
    daily_metric = metric_read(
        topic,
        table,
        "refund_rate_1d",
        "每日退款率",
        ["每日退款率"],
        "MAX(refund_rate_1d)",
        ["refund_rate_1d"],
        unit="%",
        selection_policy="per_time_grain",
        applicable_time_grain="day",
    )
    time_dimension = column_read(topic, table, "pt", "业务日期", ["日期"], role="TIME")

    contract = GroundedQueryContractBuilder().build(
        "最近30天退款率为什么高",
        [topic],
        [detail, period_metric, daily_metric, time_dimension],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [period_metric["refId"], daily_metric["refId"]],
            "dimensionRefs": [time_dimension["refId"]],
            "groupByRef": time_dimension["refId"],
            "analysisMode": "trend",
            "timeExpression": "最近30天",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.status == "REVISE_BINDINGS"
    codes = {gap.code for gap in contract.unresolved_gaps}
    assert "INCOMPATIBLE_METRIC_TIME_POLICIES" in codes
    assert "METRIC_TIME_GRAIN_MISMATCH" in codes
    mixed_gap = next(
        gap
        for gap in contract.unresolved_gaps
        if gap.code == "INCOMPATIBLE_METRIC_TIME_POLICIES"
    )
    assert mixed_gap.required_capability == {
        "singleTimeSelectionPolicy": True,
        "availableTimeSelectionPolicies": ["per_time_grain", "period_window"],
        "splitExecutionRequired": True,
    }


def test_multi_day_per_time_grain_metric_requires_time_grouping() -> None:
    topic = "经营画像"
    table = "ads_merchant_profile"
    detail = table_detail(topic, table)
    daily_metric = metric_read(
        topic,
        table,
        "refund_rate_1d",
        "每日退款率",
        ["每日退款率"],
        "MAX(refund_rate_1d)",
        ["refund_rate_1d"],
        unit="%",
        selection_policy="per_time_grain",
        applicable_time_grain="day",
    )

    contract = GroundedQueryContractBuilder().build(
        "最近30天每日退款率",
        [topic],
        [detail, daily_metric],
        binding_hints={
            "tableRefs": [detail["refId"]],
            "metricRefs": [daily_metric["refId"]],
            "analysisMode": "metric_total",
            "timeExpression": "最近30天",
        },
        now=datetime(2026, 7, 17, 2, 0, tzinfo=timezone.utc),
    )

    assert contract.status == "REVISE_BINDINGS"
    gap = next(
        gap
        for gap in contract.unresolved_gaps
        if gap.code == "METRIC_TIME_GRAIN_MISMATCH"
    )
    assert gap.required_capability["timeSelectionPolicy"] == "period_window"
    assert gap.resolution == "REVISE_BINDINGS"
