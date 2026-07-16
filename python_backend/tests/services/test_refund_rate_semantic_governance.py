import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService, semantic_catalog_conflict_detection
from merchant_ai.services.retrieval import EsKnowledgeRetrievalService


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_by_key(asset: dict, metric_key: str) -> dict:
    return next(metric for metric in asset.get("metrics") or [] if metric.get("metricKey") == metric_key)


def test_published_refund_rate_contracts_have_one_bare_alias_owner():
    assets = TopicAssetService(get_settings())
    profile = assets.load_table_asset("经营画像", "ads_merchant_profile")
    refund_detail = assets.load_table_asset("电商退货", "dwm_trade_refund_detail_di")

    period = metric_by_key(profile, "return_rate_by_order")
    assert profile["sampleEvidenceGovernance"]["enforcedAtLoad"] is True
    assert profile["profiles"] == []
    assert not any("samples=[" in str(item.get("evidence") or "") for item in profile["metrics"])
    assert period["formula"] == "SUM(return_cnt_1d) / NULLIF(SUM(order_cnt_1d), 0)"
    assert period["sourceColumns"] == ["return_cnt_1d", "order_cnt_1d"]
    assert period["temporalVariants"]["dailySeriesMetricKey"] == "refund_rate_1d"
    assert {"退款率", "退货率"} <= set(period["aliases"])

    daily = metric_by_key(profile, "refund_rate_1d")
    assert daily["formula"] == "MAX(refund_rate_1d)"
    assert daily["aggregation"] == "MAX"
    assert daily["aggregationPolicy"] == "daily_value_only"
    assert daily["temporalVariants"]["periodSummaryMetricKey"] == "return_rate_by_order"
    assert "canonicalMetricKey" not in daily
    assert "aliasOf" not in daily
    assert "禁止" in daily["selectionGuidance"]

    direct = metric_by_key(profile, "direct_refund_rate_by_pay_order")
    assert "直接退款率" in direct["aliases"]
    assert "退款率" not in direct["aliases"]
    assert "退货率" not in direct["aliases"]

    product = metric_by_key(refund_detail, "product_refund_order_share")
    assert product["businessName"] == "商品退款订单占比"
    assert "退款率" not in product["aliases"]
    assert "退货率" not in product["aliases"]
    assert not any(metric.get("metricKey") == "refund_rate" for metric in refund_detail.get("metrics") or [])

    bare_alias_owners = []
    for topic in assets.all_topic_names():
        for manifest_item in assets.load_manifest(topic):
            table = str(manifest_item.get("tableName") or "")
            for metric in assets.load_table_metrics(topic, table):
                if "退款率" in (metric.get("aliases") or []):
                    bare_alias_owners.append((topic, table, metric.get("metricKey")))
    assert bare_alias_owners == [("经营画像", "ads_merchant_profile", "return_rate_by_order")]


def test_catalog_blocks_cross_topic_ratio_alias_with_different_families(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    assets = TopicAssetService(settings)
    write_json(
        tmp_path / "topics" / "profile" / "tables" / "merchant_daily" / "asset.json",
        {
            "topic": "profile",
            "tableName": "merchant_daily",
            "metrics": [
                {
                    "metricKey": "merchant_return_rate",
                    "canonicalMetricKey": "merchant_return_rate",
                    "formula": "SUM(return_cnt) / SUM(order_cnt)",
                    "unit": "%",
                    "aliases": ["退款率"],
                }
            ],
        },
    )
    write_json(
        tmp_path / "topics" / "refund" / "tables" / "refund_detail" / "asset.json",
        {
            "topic": "refund",
            "tableName": "refund_detail",
            "metrics": [
                {
                    "metricKey": "product_refund_share",
                    "canonicalMetricKey": "product_refund_share",
                    "formula": "refund_cnt / order_cnt",
                    "unit": "%",
                    "aliases": ["退款率"],
                }
            ],
        },
    )

    report = semantic_catalog_conflict_detection(assets)

    assert any(
        item.get("type") == "global_ratio_alias_conflict" and item.get("alias") == "退款率"
        for item in report["conflicts"]
    )


def test_catalog_blocks_cross_topic_unmapped_ratio_term_alias(tmp_path):
    settings = get_settings().model_copy(update={"topic_path": str(tmp_path / "topics")})
    assets = TopicAssetService(settings)
    write_json(
        tmp_path / "topics" / "profile" / "tables" / "merchant_daily" / "asset.json",
        {
            "topic": "profile",
            "tableName": "merchant_daily",
            "metrics": [
                {
                    "metricKey": "merchant_return_rate",
                    "formula": "SUM(return_cnt) / SUM(order_cnt)",
                    "unit": "%",
                    "aliases": ["退款率"],
                }
            ],
        },
    )
    write_json(
        tmp_path / "topics" / "refund" / "tables" / "refund_detail" / "asset.json",
        {
            "topic": "refund",
            "tableName": "refund_detail",
            "terms": [{"term": "未治理退款口径", "aliases": ["退款率"]}],
        },
    )

    report = semantic_catalog_conflict_detection(assets)

    assert "global_ratio_alias_conflict" in {item.get("type") for item in report["conflicts"]}


def test_refund_rate_retrieval_exposes_governed_temporal_candidates_without_selecting_one():
    settings = get_settings()
    retrieval = EsKnowledgeRetrievalService(settings, TopicAssetService(settings))

    def metric_keys(question: str) -> list[str]:
        return [str(item.get("metricKey") or "") for item in retrieval._resolve_metric_candidates(question, [])]

    period_keys = metric_keys("最近30天退款率")
    assert {"return_rate_by_order", "refund_rate_1d"} <= set(period_keys)

    daily_keys = metric_keys("最近7天每天退款率走势")
    assert {"return_rate_by_order", "refund_rate_1d"} <= set(daily_keys)

    direct_keys = metric_keys("最近30天直接退款率")
    assert direct_keys[0] == "direct_refund_rate_by_pay_order"

    product_keys = metric_keys("最近30天 SPU 退款订单占比")
    assert product_keys[0] == "product_refund_order_share"


def test_store_summary_owns_bare_order_and_gmv_aliases_while_detail_aliases_remain_qualified():
    settings = get_settings()
    assets = TopicAssetService(settings)
    order_detail = assets.load_table_asset("电商交易", "dwm_trade_order_detail_di")

    detail_order = metric_by_key(order_detail, "order_detail_cnt")
    assert "订单量" not in detail_order["aliases"]
    assert {"商品订单量", "按商品订单量", "订单明细下单数"} <= set(detail_order["aliases"])

    detail_gmv = metric_by_key(order_detail, "pay_amt")
    assert "GMV" not in detail_gmv["aliases"]
    assert {"订单GMV", "商品GMV", "按商品GMV", "订单明细支付金额"} <= set(detail_gmv["aliases"])

    bare_owners = {"GMV": [], "订单量": []}
    for topic in assets.all_topic_names():
        for manifest_item in assets.load_manifest(topic):
            table = str(manifest_item.get("tableName") or "")
            for metric in assets.load_table_metrics(topic, table):
                aliases = set(str(alias) for alias in metric.get("aliases") or [])
                for alias in bare_owners:
                    if alias in aliases:
                        bare_owners[alias].append((topic, table, metric.get("metricKey")))

    assert bare_owners == {
        "GMV": [("经营画像", "ads_merchant_profile", "order_gmv_amt_1d")],
        "订单量": [("经营画像", "ads_merchant_profile", "order_cnt_1d")],
    }

    retrieval = EsKnowledgeRetrievalService(settings, assets)

    def top_metric(question: str) -> str:
        candidates = retrieval._resolve_metric_candidates(question, [])
        return str(candidates[0].get("metricKey") or "")

    assert top_metric("最近30天 GMV 是多少") == "order_gmv_amt_1d"
    assert top_metric("最近30天订单量是多少") == "order_cnt_1d"
    assert top_metric("最近30天按商品 GMV") == "pay_amt"
    assert top_metric("最近30天按商品订单量") == "order_detail_cnt"


def test_store_summary_owns_bare_compensation_and_ticket_rate_aliases():
    assets = TopicAssetService(get_settings())
    profile = assets.load_table_asset("经营画像", "ads_merchant_profile")
    repay = assets.load_table_asset("客服理赔", "dwm_cs_repay_detail_df")
    ticket = assets.load_table_asset("客服工单", "dwm_cs_ticket_detail_di")

    compensation_rate = metric_by_key(profile, "merchant_compensation_rate_by_order")
    assert compensation_rate["formula"] == "SUM(seller_repay_order_cnt_1d) / NULLIF(SUM(order_cnt_1d), 0)"
    assert compensation_rate["aggregationPolicy"] == "ratio_of_sums"
    assert compensation_rate["aliasConflictScope"] == "GLOBAL"

    ticket_rate = metric_by_key(profile, "merchant_ticket_rate_by_order")
    assert ticket_rate["formula"] == "SUM(cs_ticket_cnt_1d) / NULLIF(SUM(order_cnt_1d), 0)"
    assert ticket_rate["aggregationPolicy"] == "ratio_of_sums"
    assert ticket_rate["aliasConflictScope"] == "GLOBAL"

    detail_compensation_aliases = set(metric_by_key(repay, "compensation_rate")["aliases"])
    detail_ticket_aliases = set(metric_by_key(ticket, "ticket_rate")["aliases"])
    assert not {"赔付率", "理赔率"} & detail_compensation_aliases
    assert not {"工单率", "客服工单率"} & detail_ticket_aliases

    owners = {"赔付率": [], "理赔率": [], "工单率": [], "客服工单率": []}
    for topic in assets.all_topic_names():
        for manifest_item in assets.load_manifest(topic):
            table = str(manifest_item.get("tableName") or "")
            for metric in assets.load_table_metrics(topic, table):
                aliases = set(str(alias) for alias in metric.get("aliases") or [])
                for alias in owners:
                    if alias in aliases:
                        owners[alias].append((topic, table, metric.get("metricKey")))

    assert owners == {
        "赔付率": [("经营画像", "ads_merchant_profile", "merchant_compensation_rate_by_order")],
        "理赔率": [("经营画像", "ads_merchant_profile", "merchant_compensation_rate_by_order")],
        "工单率": [("经营画像", "ads_merchant_profile", "merchant_ticket_rate_by_order")],
        "客服工单率": [("经营画像", "ads_merchant_profile", "merchant_ticket_rate_by_order")],
    }

    retrieval = EsKnowledgeRetrievalService(get_settings(), assets)
    assert retrieval._resolve_metric_candidates("最近30天赔付率", [])[0]["metricKey"] == "merchant_compensation_rate_by_order"
    assert retrieval._resolve_metric_candidates("最近30天工单率", [])[0]["metricKey"] == "merchant_ticket_rate_by_order"
