from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta

from scripts.seed_yshopping_doris import (
    APPEAL_ROW_COUNT,
    DEPOSIT_ROW_COUNT,
    REALISTIC_ORDER_COUNT,
    REFUND_ROW_COUNT,
    REPAY_ROW_COUNT,
    SCM_ROW_COUNT,
    TICKET_ROW_COUNT,
    build_seed_model,
)


BASE_DATE = date(2026, 7, 18)


def test_seed_model_preserves_cross_table_product_identity() -> None:
    model = build_seed_model(BASE_DATE)
    goods_ids = {
        str(row["spu_id"])
        for row in model["dwm_goods_detail_df"]
    }

    assert goods_ids
    assert all(value.isdigit() for value in goods_ids)
    for table in [
        "dwm_trade_order_detail_di",
        "dwm_cs_ticket_detail_di",
        "dwm_scm_detail_di",
        "dwd_merchant_appeal_detail_df",
    ]:
        table_ids = {str(row["spu_id"]) for row in model[table]}
        assert table_ids
        assert table_ids <= goods_ids


def test_seed_model_preserves_order_ticket_refund_and_repay_lineage() -> None:
    model = build_seed_model(BASE_DATE)
    order_by_sub = {
        str(row["sub_order_id"]): row
        for row in model["dwm_trade_order_detail_di"]
    }
    ticket_by_id = {
        str(row["ticket_id"]): row
        for row in model["dwm_cs_ticket_detail_di"]
    }

    for ticket in model["dwm_cs_ticket_detail_di"]:
        order = order_by_sub[str(ticket["sub_order_id"])]
        assert str(ticket["spu_id"]) == str(order["spu_id"])
        assert ticket["spu_name"] == order["spu_name"]

    for refund in model["dwm_trade_refund_detail_di"]:
        order = order_by_sub[str(refund["sub_order_id"])]
        assert refund["spu_name"] == order["spu_name"]

    for repay in model["dwm_cs_repay_detail_df"]:
        ticket = ticket_by_id[str(repay["ticket_id"])]
        order = order_by_sub[str(repay["sub_order_id"])]
        assert repay["order_id"] == order["order_id"]
        assert ticket["sub_order_id"] == order["sub_order_id"]


def test_seed_model_is_large_and_has_one_meaningful_recent_ticket_leader() -> None:
    model = build_seed_model(BASE_DATE)

    assert len(model["dwm_trade_order_detail_di"]) == REALISTIC_ORDER_COUNT
    assert len(model["dwm_trade_refund_detail_di"]) == REFUND_ROW_COUNT
    assert len(model["dwm_cs_ticket_detail_di"]) == TICKET_ROW_COUNT
    assert len(model["dwm_cs_repay_detail_df"]) == REPAY_ROW_COUNT
    assert len(model["dwm_scm_detail_di"]) == SCM_ROW_COUNT
    assert len(model["dwd_merchant_deposit_recharge_df"]) == DEPOSIT_ROW_COUNT
    assert len(model["dwd_merchant_appeal_detail_df"]) == APPEAL_ROW_COUNT

    recent_start = BASE_DATE - timedelta(days=29)
    counts = Counter(
        str(row["spu_id"])
        for row in model["dwm_cs_ticket_detail_di"]
        if recent_start
        <= datetime.strptime(str(row["pt"])[:10], "%Y-%m-%d").date()
        <= BASE_DATE
    )
    leaders = [spu_id for spu_id, count in counts.items() if count == max(counts.values())]

    assert max(counts.values()) > 1
    assert leaders == ["1"]
