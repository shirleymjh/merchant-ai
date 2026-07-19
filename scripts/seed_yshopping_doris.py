#!/usr/bin/env python3

import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional


CONTAINER = "merchant-ai-rag-work-fe-1"
MYSQL_BASE = [
    "docker",
    "exec",
    "-i",
    CONTAINER,
    "mysql",
    "-N",
    "-B",
    "-uroot",
    "-P9030",
    "-h127.0.0.1",
]
DATABASE = "yshopping"
OUTPUT_SQL = Path("scripts/generated_yshopping_seed.sql")
TABLE_ROW_COUNT = 100
PROFILE_TABLE_ROW_COUNT = 100
DEMO_TABLES = [
    "ads_merchant_profile",
    "dim_merchant_df",
    "dwm_trade_order_detail_di",
    "dwm_trade_refund_detail_di",
    "dwm_cs_ticket_detail_di",
    "dwm_cs_repay_detail_df",
    "dwm_coupon_detail_di",
    "dwm_goods_detail_df",
    "dwm_scm_detail_di",
    "dwd_merchant_deposit_recharge_df",
    "dwd_merchant_appeal_detail_df",
]
BATCH_SIZE = 10
REALISTIC_ORDER_COUNT = 600
REFUND_ROW_COUNT = 160
TICKET_ROW_COUNT = 300
TICKET_TOP_PRODUCT_BOOST = 30
REPAY_ROW_COUNT = 90
COUPON_ROW_COUNT = 240
SCM_ROW_COUNT = 240
DEPOSIT_ROW_COUNT = 30
APPEAL_ROW_COUNT = 28

PRODUCT_CATALOG = [
    ("spu_id_001", "Urban Tote Bag Black", "sku_id_001", "Urban Tote Bag Black", 399.00),
    ("spu_id_002", "Retro Running Shoes White Blue", "sku_id_002", "Retro Running Shoes White Blue", 349.00),
    ("spu_id_003", "Sun Protection Shirt Beige", "sku_id_003", "Sun Protection Shirt Beige", 159.00),
    ("spu_id_004", "Lightweight Denim Jacket", "sku_id_004", "Lightweight Denim Jacket", 269.00),
    ("spu_id_005", "Commuter Backpack Grey", "sku_id_005", "Commuter Backpack Grey", 229.00),
    ("spu_id_006", "Cotton Casual Pants Khaki", "sku_id_006", "Cotton Casual Pants Khaki", 189.00),
    ("spu_id_007", "Minimal Leather Belt Brown", "sku_id_007", "Minimal Leather Belt Brown", 99.00),
    ("spu_id_008", "Travel Makeup Pouch", "sku_id_008", "Travel Makeup Pouch", 69.00),
    ("spu_id_009", "Cooling Knit T Shirt", "sku_id_009", "Cooling Knit T Shirt", 89.00),
    ("spu_id_010", "Canvas Bucket Hat", "sku_id_010", "Canvas Bucket Hat", 79.00),
    ("spu_id_011", "Linen Midi Skirt", "sku_id_011", "Linen Midi Skirt", 169.00),
    ("spu_id_012", "Soft Sole Sandals", "sku_id_012", "Soft Sole Sandals", 129.00),
    ("spu_id_013", "Silk Touch Scarf", "sku_id_013", "Silk Touch Scarf", 59.00),
    ("spu_id_014", "Daily Crossbody Bag", "sku_id_014", "Daily Crossbody Bag", 199.00),
    ("spu_id_015", "Slim Fit Polo Navy", "sku_id_015", "Slim Fit Polo Navy", 139.00),
    ("spu_id_016", "Outdoor Windbreaker", "sku_id_016", "Outdoor Windbreaker", 299.00),
    ("spu_id_017", "Pleated Wide Leg Pants", "sku_id_017", "Pleated Wide Leg Pants", 219.00),
    ("spu_id_018", "Bamboo Fiber Socks 5 Pack", "sku_id_018", "Bamboo Fiber Socks 5 Pack", 49.00),
    ("spu_id_019", "Classic White Sneakers", "sku_id_019", "Classic White Sneakers", 259.00),
    ("spu_id_020", "Ribbed Tank Top", "sku_id_020", "Ribbed Tank Top", 79.00),
    ("spu_id_021", "Waterproof Phone Pouch", "sku_id_021", "Waterproof Phone Pouch", 39.00),
    ("spu_id_022", "Compact Umbrella", "sku_id_022", "Compact Umbrella", 59.00),
    ("spu_id_023", "Yoga Training Leggings", "sku_id_023", "Yoga Training Leggings", 149.00),
    ("spu_id_024", "Quick Dry Sports Shorts", "sku_id_024", "Quick Dry Sports Shorts", 99.00),
    ("spu_id_025", "Woven Straw Bag", "sku_id_025", "Woven Straw Bag", 129.00),
    ("spu_id_026", "UV Arm Sleeves", "sku_id_026", "UV Arm Sleeves", 29.00),
    ("spu_id_027", "Leather Card Holder", "sku_id_027", "Leather Card Holder", 89.00),
    ("spu_id_028", "Loose Fit Hoodie", "sku_id_028", "Loose Fit Hoodie", 199.00),
    ("spu_id_029", "Daily Hair Clip Set", "sku_id_029", "Daily Hair Clip Set", 35.00),
    ("spu_id_030", "Lightweight Travel Slippers", "sku_id_030", "Lightweight Travel Slippers", 45.00),
    ("spu_id_031", "Cotton Pajama Set", "sku_id_031", "Cotton Pajama Set", 179.00),
    ("spu_id_032", "Slim Ankle Boots", "sku_id_032", "Slim Ankle Boots", 329.00),
    ("spu_id_033", "Minimal Watch Strap", "sku_id_033", "Minimal Watch Strap", 79.00),
    ("spu_id_034", "Foldable Shopping Bag", "sku_id_034", "Foldable Shopping Bag", 25.00),
    ("spu_id_100", "spu_name_100", "sku_id_100", "sku_name_100", 121.50),
]


def run_mysql(sql: str, database: Optional[str] = None) -> str:
    cmd = MYSQL_BASE.copy()
    if database:
        cmd.extend(["-D", database])
    cmd.extend(["-e", sql])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_tables() -> list[str]:
    rows = run_mysql(f"SHOW TABLES FROM {DATABASE};")
    existing = {line.strip() for line in rows.splitlines() if line.strip()}
    return [table for table in DEMO_TABLES if table in existing]


def get_columns() -> dict[str, list[dict[str, str]]]:
    sql = (
        "SELECT table_name, column_name, data_type, is_nullable, column_key, ordinal_position, "
        "COALESCE(CAST(character_maximum_length AS STRING), '') "
        "FROM information_schema.columns "
        f"WHERE table_schema='{DATABASE}' "
        "ORDER BY table_name, ordinal_position;"
    )
    rows = run_mysql(sql)
    by_table: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in rows.splitlines():
        parts = line.split("\t")
        while len(parts) < 7:
            parts.append("")
        table_name, column_name, data_type, is_nullable, column_key, ordinal_position, max_length = parts[:7]
        by_table[table_name].append(
            {
                "name": column_name,
                "type": data_type.lower(),
                "nullable": is_nullable,
                "key": column_key,
                "position": ordinal_position,
                "max_length": max_length,
            }
        )
    return by_table


def get_current_date() -> date:
    value = run_mysql("SELECT CURRENT_DATE();")
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def sql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def format_sql_value(value):
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{sql_escape(str(value))}'"


def money(value: float) -> float:
    return round(value + 0.000001, 2)


def id_value(prefix: str, seq: int) -> str:
    return f"{prefix}_{seq:03d}"


def datetime_string(pt_value: date, hour: int, minute: int, second: int = 0) -> str:
    return f"{pt_value:%Y-%m-%d} {hour % 24:02d}:{minute % 60:02d}:{second % 60:02d}"


def apply_length_limit(value, column: dict[str, str]):
    if not isinstance(value, str):
        return value
    max_length = column.get("max_length", "")
    if max_length.isdigit():
        limit = int(max_length)
        return value[:limit]
    return value


def random_time_string(pt_value: date, seed: int) -> str:
    rng = random.Random(seed)
    return f"{pt_value:%Y-%m-%d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def is_integer_type(data_type: str) -> bool:
    return data_type in {"bigint", "int", "integer", "smallint", "tinyint", "largeint"}


def is_decimal_type(data_type: str) -> bool:
    return data_type in {"double", "float", "decimal", "decimalv3"}


def column_names(columns: list[dict[str, str]]) -> set[str]:
    return {column["name"].lower() for column in columns}


def trailing_number(value) -> int:
    text = str(value or "")
    cursor = len(text)
    while cursor > 0 and text[cursor - 1].isdigit():
        cursor -= 1
    return int(text[cursor:]) if cursor < len(text) else 0


def coerce_for_column(value, column: dict[str, str]):
    if value is None:
        return ""
    data_type = column["type"]
    if isinstance(value, date) and data_type == "date":
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if is_integer_type(data_type):
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return int(round(value))
        return trailing_number(value)
    if is_decimal_type(data_type):
        if isinstance(value, (int, float)):
            return money(float(value))
        try:
            return money(float(str(value)))
        except ValueError:
            return 0
    return value


def row_from_map(table: str, columns: list[dict[str, str]], values: dict, row_index: int) -> list:
    pt_value = values.get("pt")
    if isinstance(pt_value, str):
        try:
            pt_value = datetime.strptime(pt_value[:10], "%Y-%m-%d").date()
        except ValueError:
            pt_value = date.today()
    elif not isinstance(pt_value, date):
        pt_value = date.today()
    row = []
    for column in columns:
        name = column["name"].lower()
        value = values.get(name)
        if value is None:
            value = values.get(column["name"])
        if value is None:
            value = seed_value(table, column, row_index, pt_value)
        row.append(apply_length_limit(coerce_for_column(value, column), column))
    return row


def product_for_index(index: int) -> dict:
    raw = PRODUCT_CATALOG[index % len(PRODUCT_CATALOG)]
    # The repo DDL models dwm_goods_detail_df.spu_id as BIGINT, while order/scm
    # tables expose it as text/varchar. Keep seed data cross-table comparable by
    # using the numeric business id everywhere instead of prefixed demo strings.
    spu_id = str(trailing_number(raw[0]))
    sku_id = str(trailing_number(raw[2]))
    return {
        "spu_id": spu_id,
        "spu_name": raw[1],
        "sku_id": sku_id,
        "sku_name": raw[3],
        "price": raw[4],
        "seq": int(spu_id or 0),
    }


def choose_product(rng: random.Random) -> dict:
    weights = []
    for index, _product in enumerate(PRODUCT_CATALOG):
        if index < 5:
            weights.append(10)
        elif index < 12:
            weights.append(6)
        elif index < 24:
            weights.append(3)
        elif index == len(PRODUCT_CATALOG) - 1:
            weights.append(1)
        else:
            weights.append(1.4)
    return product_for_index(rng.choices(range(len(PRODUCT_CATALOG)), weights=weights, k=1)[0])


def add_common_aliases(row: dict) -> dict:
    """Populate likely real-schema aliases while keeping simplified demo schemas compatible."""
    if "pay_amt" in row:
        row.setdefault("order_pay_amt", row["pay_amt"])
        row.setdefault("pay_amount", row["pay_amt"])
        row.setdefault("actual_pay_amt", row["pay_amt"])
        row.setdefault("order_gmv_amt", row["pay_amt"])
    if "refund_amt" in row:
        row.setdefault("pay_amt", row["refund_amt"])
        row.setdefault("refund_amount", row["refund_amt"])
        row.setdefault("refund_pay_amt", row["refund_amt"])
    if "sub_order_create_time" in row:
        row.setdefault("create_time", row["sub_order_create_time"])
        row.setdefault("order_create_time", row["sub_order_create_time"])
    if "refund_create_time" in row:
        row.setdefault("create_time", row["refund_create_time"])
    return row


def build_orders(base_date: date, rng: random.Random) -> list[dict]:
    orders: list[dict] = []
    # The app's default recent-detail window ends at the latest closed business day.
    special_pt = base_date - timedelta(days=1)
    special = add_common_aliases({
        "pt": special_pt,
        "order_id": "order_id_100",
        "sub_order_id": "sub_order_id_100",
        "buyer_id": "buyer_id_100",
        "buyer_name": "buyer_name_100",
        "seller_id": "100",
        "seller_name": "杭州云尚优选商贸有限公司",
        "sub_order_status_name": "交易成功",
        "order_status_name": "交易成功",
        "spu_id": "100",
        "spu_name": "spu_name_100",
        "sku_id": "100",
        "sku_name": "sku_name_100",
        "sku_cnt": 1,
        "quantity": 1,
        "pay_amt": 121.50,
        "pay_status_name": "支付成功",
        "address_city_name": "杭州",
        "receiver_city_name": "杭州",
        "sub_order_create_time": datetime_string(special_pt, 8, 27, 9),
    })
    orders.append(special)
    statuses = ["交易成功", "已发货", "待发货", "已签收", "交易关闭"]
    cities = ["杭州", "上海", "苏州", "南京", "宁波", "广州", "深圳", "北京", "成都"]
    seq = 1
    while len(orders) < REALISTIC_ORDER_COUNT:
        if seq == 100:
            seq += 1
            continue
        pt = base_date - timedelta(days=rng.randrange(0, 90))
        product = choose_product(rng)
        qty = rng.choices([1, 2, 3], weights=[82, 15, 3], k=1)[0]
        discount = rng.choice([0, 0, 0, 5, 10, 20])
        amount = money(max(9, product["price"] * qty - discount))
        status = rng.choices(statuses, weights=[58, 18, 12, 8, 4], k=1)[0]
        paid = status != "交易关闭"
        coupon_id = id_value("coupon_id", seq) if paid and discount > 0 else ""
        order = add_common_aliases({
            "pt": pt,
            "order_id": id_value("order_id", seq),
            "sub_order_id": id_value("sub_order_id", seq),
            "buyer_id": id_value("buyer_id", 1000 + rng.randrange(1, 180)),
            "buyer_name": id_value("buyer_name", 1000 + rng.randrange(1, 180)),
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "sub_order_status_name": status,
            "order_status_name": status,
            "spu_id": product["spu_id"],
            "spu_name": product["spu_name"],
            "sku_id": product["sku_id"],
            "sku_name": product["sku_name"],
            "sku_cnt": qty,
            "quantity": qty,
            "discount_amt": discount,
            "discount_id": coupon_id,
            "discount_rel_id": coupon_id,
            "discount_type_code": 1 if coupon_id else 0,
            "discount_type_name": "优惠券" if coupon_id else "",
            "is_use": 1 if coupon_id else 0,
            "pay_amt": amount if paid else 0,
            "pay_status_name": "支付成功" if paid else "未支付",
            "address_city_name": rng.choices(cities, weights=[38, 14, 9, 8, 7, 7, 7, 5, 5], k=1)[0],
            "receiver_city_name": rng.choices(cities, weights=[38, 14, 9, 8, 7, 7, 7, 5, 5], k=1)[0],
            "sub_order_create_time": datetime_string(pt, rng.randrange(8, 23), rng.randrange(0, 60), rng.randrange(0, 60)),
            "discount_create_time": datetime_string(pt, rng.randrange(8, 23), rng.randrange(0, 60), rng.randrange(0, 60)),
            "discount_modify_time": datetime_string(pt, rng.randrange(8, 23), rng.randrange(0, 60), rng.randrange(0, 60)),
        })
        orders.append(order)
        seq += 1
    return orders


def build_refunds(base_date: date, orders: list[dict], rng: random.Random) -> list[dict]:
    reasons = ["尺码不合适", "七天无理由", "商品瑕疵", "发错颜色", "物流破损", "买家拍错", "未按约定时间发货"]
    statuses = ["退款成功", "待商家处理", "退款关闭", "平台介入中"]
    refund_orders = []
    for order in orders:
        if order["order_id"] == "order_id_100":
            refund_orders.append(order)
            continue
        if order["pay_amt"] <= 0:
            continue
        product_seq = trailing_number(order["spu_id"])
        propensity = 0.10
        if product_seq in {2, 12, 19, 23, 32}:
            propensity += 0.14
        if product_seq in {1, 4, 5, 16}:
            propensity += 0.08
        if order["sub_order_status_name"] in {"待发货", "已发货"}:
            propensity += 0.04
        if rng.random() < propensity:
            refund_orders.append(order)
    target = REFUND_ROW_COUNT
    if len(refund_orders) > target:
        keep_special = [order for order in refund_orders if order["order_id"] == "order_id_100"]
        others = [order for order in refund_orders if order["order_id"] != "order_id_100"]
        refund_orders = keep_special + rng.sample(others, target - len(keep_special))
    elif len(refund_orders) < target:
        candidates = [order for order in orders if order["pay_amt"] > 0 and order not in refund_orders]
        refund_orders.extend(rng.sample(candidates, min(target - len(refund_orders), len(candidates))))
    refunds: list[dict] = []
    for index, order in enumerate(sorted(refund_orders, key=lambda item: item["sub_order_create_time"]), start=1):
        if order["order_id"] == "order_id_100":
            status = "退款成功"
            amount = order["pay_amt"]
            refund_pt = datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date()
            refund_time = datetime_string(refund_pt, 10, 21, 49)
            reason = "商品瑕疵"
            refund_id = "refund_id_100"
        else:
            status = rng.choices(statuses, weights=[72, 15, 8, 5], k=1)[0]
            ratio = rng.choices([1.0, 0.5, 0.35, 0.2], weights=[42, 30, 18, 10], k=1)[0]
            amount = money(order["pay_amt"] * ratio)
            order_pt = datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date()
            refund_pt = min(base_date, order_pt + timedelta(days=rng.randrange(0, 8)))
            refund_time = datetime_string(refund_pt, rng.randrange(9, 23), rng.randrange(0, 60), rng.randrange(0, 60))
            reason = rng.choice(reasons)
            refund_id = id_value("refund_id", index)
        refunds.append(add_common_aliases({
            "pt": refund_pt,
            "refund_id": refund_id,
            "order_id": order["order_id"],
            "sub_order_id": order["sub_order_id"],
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "buyer_id": order["buyer_id"],
            "buyer_name": order["buyer_name"],
            "spu_id": order["spu_id"],
            "spu_name": order["spu_name"],
            "sku_id": order["sku_id"],
            "sku_title": order["sku_name"],
            "sku_name": order["sku_name"],
            "refund_status_name": status,
            "refund_reason": reason,
            "refund_amt": amount,
            "discount_id": order.get("discount_rel_id") or order.get("discount_id") or "",
            "refund_create_time": refund_time,
        }))
    return refunds


def build_goods(base_date: date, orders: list[dict], rng: random.Random) -> list[dict]:
    first_order_by_spu: dict[str, date] = {}
    snapshot_pt = base_date - timedelta(days=1)
    for order in orders:
        pt = datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date()
        first_order_by_spu[order["spu_id"]] = min(first_order_by_spu.get(order["spu_id"], pt), pt)
    rows = []
    for index, _raw in enumerate(PRODUCT_CATALOG):
        product = product_for_index(index)
        spu_id = product["spu_id"]
        spu_name = product["spu_name"]
        sku_id = product["sku_id"]
        sku_name = product["sku_name"]
        first_order = first_order_by_spu.get(spu_id, base_date - timedelta(days=30))
        apply_date = max(base_date - timedelta(days=180), first_order - timedelta(days=rng.randrange(7, 45)))
        status = "已上架" if rng.random() > 0.08 else rng.choice(["审核中", "审核拒绝"])
        rows.append({
            "pt": snapshot_pt,
            "seller_id": "100",
            "spu_id": spu_id,
            "spu_name": spu_name,
            "sku_id": sku_id,
            "sku_name": sku_name,
            "goods_name": spu_name,
            "spu_status_name": status,
            "audit_operate_type_name": "平台审核" if status != "审核中" else "商家提交审核",
            "is_audit_pass": 1 if status == "已上架" else 0,
            "audit_remark": "审核通过" if status == "已上架" else "资料待补充或主图需优化",
            "spu_apply_create_time": datetime_string(apply_date, rng.randrange(8, 18), rng.randrange(0, 60)),
            "create_time": datetime_string(apply_date, rng.randrange(8, 18), rng.randrange(0, 60)),
        })
    return rows


def build_tickets(base_date: date, orders: list[dict], refunds: list[dict], rng: random.Random) -> list[dict]:
    paid_orders = [order for order in orders if order["pay_amt"] > 0]
    base_count = max(1, TICKET_ROW_COUNT - TICKET_TOP_PRODUCT_BOOST)
    source_orders = rng.sample(paid_orders, min(base_count, len(paid_orders)))
    recent_cutoff = base_date - timedelta(days=25)
    top_product_orders = [
        order
        for order in paid_orders
        if str(order.get("spu_id")) == "1"
        and datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date() >= recent_cutoff
    ]
    if not top_product_orders:
        top_product_orders = [
            order
            for order in paid_orders
            if datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date() >= recent_cutoff
        ]
    if top_product_orders:
        boosted_order = max(top_product_orders, key=lambda item: str(item["pt"]))
        source_orders.extend([boosted_order] * TICKET_TOP_PRODUCT_BOOST)
    refund_by_sub = {refund["sub_order_id"]: refund for refund in refunds}
    titles = ["物流进度咨询", "退款处理进度咨询", "商品质量反馈", "尺码咨询", "优惠券无法使用", "发票开具咨询"]
    rows = []
    for index, order in enumerate(source_orders, start=1):
        has_refund = order["sub_order_id"] in refund_by_sub
        pt = min(base_date, datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date() + timedelta(days=rng.randrange(0, 5)))
        title = "退款处理进度咨询" if has_refund else rng.choice(titles)
        status = rng.choices(["已关闭", "处理中", "待商家回复"], weights=[68, 22, 10], k=1)[0]
        rows.append({
            "pt": pt,
            "ticket_id": id_value("ticket_id", index),
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "buyer_id": order["buyer_id"],
            "buyer_name": order["buyer_name"],
            "order_id": order["order_id"],
            "sub_order_id": order["sub_order_id"],
            "spu_id": order["spu_id"],
            "spu_name": order["spu_name"],
            "ticket_title": title,
            "content": f"{title}，关联订单{order['order_id']}",
            "ticket_status_name": status,
            "priority_name": "高" if has_refund and rng.random() < 0.35 else rng.choice(["中", "低"]),
            "is_reopen": 1 if rng.random() < 0.08 else 0,
            "is_reminder": 1 if rng.random() < 0.18 else 0,
            "ticket_score": money(rng.uniform(3.6, 5.0)) if status == "已关闭" else 0,
            "ticket_create_time": datetime_string(pt, rng.randrange(9, 22), rng.randrange(0, 60)),
            "create_time": datetime_string(pt, rng.randrange(9, 22), rng.randrange(0, 60)),
        })
    return rows


def build_repay(base_date: date, tickets: list[dict], orders: list[dict], rng: random.Random) -> list[dict]:
    order_by_sub = {order["sub_order_id"]: order for order in orders}
    candidates = [ticket for ticket in tickets if ticket.get("sub_order_id") in order_by_sub]
    rows = []
    selected = rng.sample(candidates, min(REPAY_ROW_COUNT, len(candidates)))
    for index, ticket in enumerate(selected, start=1):
        order = order_by_sub[ticket["sub_order_id"]]
        pt = min(base_date, datetime.strptime(str(ticket["pt"])[:10], "%Y-%m-%d").date() + timedelta(days=rng.randrange(0, 3)))
        repay_amt = money(min(order["pay_amt"], rng.choice([20, 30, 50, 80, 100])))
        rows.append({
            "pt": pt,
            "bill_id": id_value("bill_id", index),
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "buyer_id": ticket["buyer_id"],
            "buyer_name": ticket["buyer_name"],
            "ticket_id": ticket["ticket_id"],
            "order_id": order["order_id"],
            "sub_order_id": order["sub_order_id"],
            "is_return": 1 if "退款" in ticket["ticket_title"] else 0,
            "repay_amt": repay_amt,
            "repay_status_code": rng.choice([1, 2, 3, 3, 3, 4]),
            "repay_status_name": rng.choices(["审批完成", "审批中", "已驳回", "已取消"], weights=[70, 15, 10, 5], k=1)[0],
            "pay_status_code": rng.choice([1, 3, 3, 3]),
            "pay_status_name": rng.choices(["打款成功", "打款中"], weights=[85, 15], k=1)[0],
            "pay_way_code": rng.choice([1, 2, 3]),
            "pay_way_name": rng.choice(["现金", "优惠券", "平台好物币"]),
            "cause_id": "售后体验补偿",
            "reason_code": 100 + index,
            "level1_reason_name": "售后服务",
            "level2_reason_name": "退款处理",
            "level3_reason_name": "体验补偿",
            "content": f"关联工单{ticket['ticket_id']}，按规则赔付",
            "create_time": datetime_string(pt, rng.randrange(9, 22), rng.randrange(0, 60)),
            "modify_time": datetime_string(pt, rng.randrange(9, 22), rng.randrange(0, 60)),
        })
    return rows


def build_coupons(base_date: date, orders: list[dict], rng: random.Random) -> list[dict]:
    buyers = sorted({order["buyer_id"] for order in orders})
    rows = []
    templates = [("满200减20券", 20, 200), ("满300减30券", 30, 300), ("老客复购10元券", 10, 99), ("夏季包邮券", 8, 59)]
    used_coupon_orders = [
        order
        for order in orders
        if order.get("discount_rel_id") and order.get("pay_amt", 0) > 0
    ]
    used_coupon_orders = sorted(used_coupon_orders, key=lambda item: (str(item.get("pt")), str(item.get("sub_order_id"))))
    for index, order in enumerate(used_coupon_orders, start=1):
        order_pt = datetime.strptime(str(order["pt"])[:10], "%Y-%m-%d").date()
        pt = max(base_date - timedelta(days=90), order_pt - timedelta(days=rng.randrange(0, 3)))
        title, amt, threshold = rng.choice(templates)
        rows.append({
            "pt": pt,
            "coupon_id": order["discount_rel_id"],
            "user_id": order["buyer_id"],
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "template_id": id_value("template_id", (index % 8) + 1),
            "template_title": title,
            "activity_id": id_value("activity_id", (index % 5) + 1),
            "activity_name": "夏季焕新活动" if index % 2 else "老客复购活动",
            "coupon_amt": order.get("discount_amt") or amt,
            "threshold": threshold,
            "discount_way_code": 1,
            "discount_way_name": "满减",
            "coupon_send_status_code": 2,
            "coupon_send_status_name": "已核销",
            "is_receive": 1,
            "is_voucher": 0,
            "coupon_create_time": datetime_string(pt, rng.randrange(8, 22), rng.randrange(0, 60)),
            "coupon_modify_time": datetime_string(order_pt, rng.randrange(8, 22), rng.randrange(0, 60)),
            "snap_create_time": datetime_string(pt, rng.randrange(8, 22), rng.randrange(0, 60)),
            "snap_modify_time": datetime_string(order_pt, rng.randrange(8, 22), rng.randrange(0, 60)),
            "template_create_time": datetime_string(pt - timedelta(days=7), 10, 0),
            "template_modify_time": datetime_string(pt - timedelta(days=1), 10, 0),
            "activity_start_time": datetime_string(pt - timedelta(days=3), 0, 0),
            "activity_expire_time": datetime_string(min(base_date, pt + timedelta(days=14)), 23, 59, 59),
            "activity_create_time": datetime_string(pt - timedelta(days=5), 10, 0),
            "activity_modify_time": datetime_string(pt - timedelta(days=1), 10, 0),
            "coupon_content": f"{title}，满{threshold}减{amt}",
        })
    next_index = len(rows) + 1
    while len(rows) < COUPON_ROW_COUNT:
        pt = base_date - timedelta(days=rng.randrange(0, 90))
        title, amt, threshold = rng.choice(templates)
        rows.append({
            "pt": pt,
            "coupon_id": id_value("coupon_id", 1000 + next_index),
            "user_id": rng.choice(buyers),
            "seller_id": "100",
            "seller_name": "杭州云尚优选商贸有限公司",
            "template_id": id_value("template_id", (next_index % 8) + 1),
            "template_title": title,
            "activity_id": id_value("activity_id", (next_index % 5) + 1),
            "activity_name": "夏季焕新活动" if next_index % 2 else "老客复购活动",
            "coupon_amt": amt,
            "threshold": threshold,
            "discount_way_code": 1,
            "discount_way_name": "满减",
            "coupon_send_status_code": rng.choice([1, 3]),
            "coupon_send_status_name": rng.choice(["已领取", "已过期"]),
            "is_receive": 1,
            "is_voucher": 0,
            "coupon_create_time": datetime_string(pt, rng.randrange(8, 22), rng.randrange(0, 60)),
            "template_create_time": datetime_string(pt - timedelta(days=7), 10, 0),
            "activity_start_time": datetime_string(pt - timedelta(days=3), 0, 0),
            "activity_expire_time": datetime_string(min(base_date, pt + timedelta(days=14)), 23, 59, 59),
            "coupon_content": f"{title}，满{threshold}减{amt}",
        })
        next_index += 1
    return rows


def build_scm(base_date: date, rng: random.Random) -> list[dict]:
    rows = []
    for index in range(1, SCM_ROW_COUNT + 1):
        product = product_for_index(index)
        pt = base_date - timedelta(days=rng.randrange(0, 90))
        inbound_cnt = rng.choice([20, 30, 40, 50, 60, 80, 100, 120])
        rows.append({
            "pt": pt,
            "inbound_id": id_value("inbound_id", index),
            "seller_id": "100",
            "spu_id": product["spu_id"],
            "sku_id": product["sku_id"],
            "inbound_status_name": rng.choices(["已入库", "待质检", "部分入库"], weights=[72, 15, 13], k=1)[0],
            "inbound_cnt": inbound_cnt,
            "warehouse_id": rng.choice(["WH-HZ-01", "WH-HZ-02", "WH-SH-01"]),
            "check_status_name": rng.choices(["质检通过", "待质检", "质检异常"], weights=[80, 12, 8], k=1)[0],
            "identify_result_name": rng.choices(["正品", "待鉴定", "异常"], weights=[88, 8, 4], k=1)[0],
            "outbound_id": id_value("outbound_id", index) if rng.random() < 0.45 else "",
            "inbound_create_time": datetime_string(pt, rng.randrange(8, 20), rng.randrange(0, 60)),
        })
    return rows


def build_deposits(base_date: date, rng: random.Random) -> list[dict]:
    rows = []
    for index in range(1, DEPOSIT_ROW_COUNT + 1):
        pt = base_date - timedelta(days=rng.randrange(0, 90))
        rows.append({
            "pt": pt,
            "merchant_id": "100",
            "user_id": "u100",
            "deposit_recharge_id": id_value("deposit_recharge_id", index),
            "trans_id": id_value("trans_id", index),
            "currency": "CNY",
            "deposit_recharge_amt": rng.choice([500, 800, 1000, 1500, 2000]),
            "trans_voucher": f"https://example.com/voucher/{index}",
            "remark": "保证金补缴" if index % 2 else "活动保证金充值",
            "create_time": datetime_string(pt, rng.randrange(9, 18), rng.randrange(0, 60)),
            "modify_time": datetime_string(pt, rng.randrange(9, 18), rng.randrange(0, 60)),
        })
    return rows


def build_appeals(base_date: date, goods: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    selected = rng.sample(goods, min(APPEAL_ROW_COUNT, len(goods)))
    for index, product in enumerate(selected, start=1):
        pt = base_date - timedelta(days=rng.randrange(0, 90))
        rows.append({
            "pt": pt,
            "appeal_id": id_value("appeal_id", index),
            "merchant_id": "100",
            "spu_id": product["spu_id"],
            "spu_name": product["spu_name"],
            "level1_category_code": 10,
            "level1_category_name": "服饰箱包",
            "level2_category_code": 100 + index,
            "level2_category_name": "时尚配饰",
            "level3_category_code": 1000 + index,
            "appeal_status_code": rng.choice([1, 1, 2, 3]),
            "appeal_status_name": rng.choices(["通过", "驳回", "取消"], weights=[70, 20, 10], k=1)[0],
            "apply_type_code": 1,
            "apply_type_name": "商品管理",
            "reason": "补充品牌授权和商品实拍图后发起申诉",
            "images_url": "https://example.com/appeal/image.jpg",
            "create_time": datetime_string(pt, rng.randrange(9, 20), rng.randrange(0, 60)),
            "modify_time": datetime_string(pt, rng.randrange(9, 20), rng.randrange(0, 60)),
        })
    return rows


def build_profile_rows(
        base_date: date,
        pt_values: list[date],
        orders: list[dict],
        refunds: list[dict],
        tickets: list[dict],
        repays: list[dict],
        coupons: list[dict],
        goods: list[dict],
        deposits: list[dict],
        appeals: list[dict],
        scm_rows: list[dict],
) -> list[dict]:
    orders_by_pt = defaultdict(list)
    refunds_by_pt = defaultdict(list)
    tickets_by_pt = defaultdict(list)
    repays_by_pt = defaultdict(list)
    coupons_by_pt = defaultdict(list)
    deposits_by_pt = defaultdict(list)
    appeals_by_pt = defaultdict(list)
    scm_by_pt = defaultdict(list)
    for collection, target in [
        (orders, orders_by_pt),
        (refunds, refunds_by_pt),
        (tickets, tickets_by_pt),
        (repays, repays_by_pt),
        (coupons, coupons_by_pt),
        (deposits, deposits_by_pt),
        (appeals, appeals_by_pt),
        (scm_rows, scm_by_pt),
    ]:
        for row in collection:
            target[datetime.strptime(str(row["pt"])[:10], "%Y-%m-%d").date()].append(row)
    goods_apply_by_pt = defaultdict(list)
    for row in goods:
        create_date = datetime.strptime(str(row["spu_apply_create_time"])[:10], "%Y-%m-%d").date()
        goods_apply_by_pt[create_date].append(row)
    rows = []
    for pt in pt_values:
        day_orders = orders_by_pt[pt]
        paid_orders = [row for row in day_orders if row.get("pay_amt", 0) > 0]
        success_orders = [row for row in paid_orders if row.get("sub_order_status_name") in {"交易成功", "已签收"}]
        day_refunds = refunds_by_pt[pt]
        success_refunds = [row for row in day_refunds if row.get("refund_status_name") == "退款成功"]
        day_tickets = tickets_by_pt[pt]
        day_repays = repays_by_pt[pt]
        day_coupons = coupons_by_pt[pt]
        online_goods = [
            row for row in goods
            if row.get("spu_status_name") == "已上架"
            and datetime.strptime(str(row["spu_apply_create_time"])[:10], "%Y-%m-%d").date() <= pt
        ]
        pay_gmv = money(sum(row.get("pay_amt", 0) for row in paid_orders))
        success_gmv = money(sum(row.get("pay_amt", 0) for row in success_orders))
        refund_amt = money(sum(row.get("refund_amt", row.get("pay_amt", 0)) for row in day_refunds))
        row = {
            "pt": pt,
            "merchant_id": "100",
            "user_id": "u100",
            "merchant_type_name": "POP商家",
            "brand_type_name": "品牌授权商家",
            "balance_type_name": "平台结算",
            "mobile": "13800001000",
            "company_name": "杭州云尚优选商贸有限公司",
            "license_id": "91330100MA1000001X",
            "is_unconditional_refund": 1,
            "is_invoice": 1,
            "contact_name": "何林",
            "business_address": "杭州市西湖区文三路88号",
            "send_address": "杭州市余杭区仓前街道1号仓",
            "refnd_address": "杭州市余杭区售后中心2号库",
            "bank_name": "招商银行杭州西湖支行",
            "bank_account": "6225888888881000",
            "account_type_name": "对公账户",
            "poundage_discount": 0.006,
            "deposit_amt": 8800,
            "order_cnt_1d": len(day_orders),
            "order_user_cnt_1d": len({row["buyer_id"] for row in day_orders}),
            "order_gmv_amt_1d": pay_gmv,
            "pay_order_cnt_1d": len(paid_orders),
            "pay_gmv_amt_1d": pay_gmv,
            "trade_success_order_cnt_1d": len(success_orders),
            "trade_success_gmv_amt_1d": success_gmv,
            "avg_pay_order_amt_1d": money(pay_gmv / len(paid_orders)) if paid_orders else 0,
            "ship_timeout_order_cnt_1d": max(0, len([row for row in paid_orders if row.get("sub_order_status_name") == "待发货"]) // 4),
            "signed_order_cnt_1d": len([row for row in paid_orders if row.get("sub_order_status_name") in {"已签收", "交易成功"}]),
            "delivery_timeout_order_cnt_1d": max(0, len(paid_orders) // 10),
            "refund_amt_1d": refund_amt,
            "return_success_amt_1d": money(sum(row.get("refund_amt", 0) for row in success_refunds)),
            "return_success_cnt_1d": len(success_refunds),
            "return_cnt_1d": len(day_refunds),
            "direct_refund_cnt_1d": len([row for row in day_refunds if row.get("refund_reason") in {"七天无理由", "买家拍错"}]),
            "refund_rate_1d": round(len(day_refunds) / len(paid_orders), 4) if paid_orders else 0,
            "cs_ticket_cnt_1d": len(day_tickets),
            "ticket_reopen_cnt_1d": sum(row.get("is_reopen", 0) for row in day_tickets),
            "ticket_reminder_cnt_1d": sum(row.get("is_reminder", 0) for row in day_tickets),
            "ticket_close_cnt_1d": len([row for row in day_tickets if row.get("ticket_status_name") == "已关闭"]),
            "avg_ticket_score_1d": money(sum(row.get("ticket_score", 0) for row in day_tickets) / len(day_tickets)) if day_tickets else 0,
            "seller_repay_order_cnt_1d": len(day_repays),
            "seller_repay_amt_1d": money(sum(row.get("repay_amt", 0) for row in day_repays)),
            "pay_success_discount_order_cnt_1d": len([row for row in day_coupons if row.get("coupon_send_status_name") == "已核销"]),
            "pay_success_discount_amt_1d": money(sum(row.get("coupon_amt", 0) for row in day_coupons if row.get("coupon_send_status_name") == "已核销")),
            "trade_success_discount_order_cnt_1d": len([row for row in day_coupons if row.get("coupon_send_status_name") == "已核销"]),
            "trade_success_discount_amt_1d": money(sum(row.get("coupon_amt", 0) for row in day_coupons if row.get("coupon_send_status_name") == "已核销")),
            "pay_discount_rate_1d": round(
                sum(row.get("coupon_amt", 0) for row in day_coupons if row.get("coupon_send_status_name") == "已核销") / pay_gmv,
                4) if pay_gmv else 0,
            "goods_audit_reject_cnt_1d": len([row for row in goods_apply_by_pt[pt] if row.get("spu_status_name") == "审核拒绝"]),
            "goods_audit_pass_cnt_1d": len([row for row in goods_apply_by_pt[pt] if row.get("spu_status_name") == "已上架"]),
            "goods_online_cnt_1d": len(online_goods),
            "goods_apply_cnt_1d": len(goods_apply_by_pt[pt]),
            "deposit_pay_cnt_1d": len(deposits_by_pt[pt]),
            "appeal_success_cnt_1d": len([row for row in appeals_by_pt[pt] if row.get("appeal_status_name") == "通过"]),
            "appeal_cnt_1d": len(appeals_by_pt[pt]),
            "punish_cnt_1d": 1 if len(day_refunds) > 4 and len(day_tickets) > 2 else 0,
            "scm_performance_cnt_1d": len(scm_by_pt[pt]),
        }
        rows.append(row)
    return rows


def build_dim_rows(base_date: date) -> list[dict]:
    return [{
        "pt": base_date,
        "merchant_id": "100",
        "user_id": "u100",
        "merchant_type_name": "POP商家",
        "brand_type_name": "品牌授权商家",
        "balance_type_name": "平台结算",
        "mobile": "13800001000",
        "company_name": "杭州云尚优选商贸有限公司",
        "license_id": "91330100MA1000001X",
        "contact_name": "何林",
        "business_address": "杭州市西湖区文三路88号",
        "send_address": "杭州市余杭区仓前街道1号仓",
        "refnd_address": "杭州市余杭区售后中心2号库",
        "bank_name": "招商银行杭州西湖支行",
        "bank_account": "6225888888881000",
        "account_type_name": "对公账户",
        "ship_model_name": "商家自发货",
        "is_invoice": 1,
        "is_unconditional_refund": 1,
        "init_deposit_amt": 10000,
        "deposit_freeze": 1200,
        "deposit_amt": 8800,
        "min_poundage": 0.003,
        "max_poundage": 0.008,
        "poundage_discount": 0.006,
    }]


def build_seed_model(base_date: date) -> dict[str, list[dict]]:
    rng = random.Random(20260620)
    pt_values = [base_date - timedelta(days=offset) for offset in range(PROFILE_TABLE_ROW_COUNT - 1, -1, -1)]
    orders = build_orders(base_date, rng)
    refunds = build_refunds(base_date, orders, rng)
    goods = build_goods(base_date, orders, rng)
    tickets = build_tickets(base_date, orders, refunds, rng)
    repays = build_repay(base_date, tickets, orders, rng)
    coupons = build_coupons(base_date, orders, rng)
    scm_rows = build_scm(base_date, rng)
    deposits = build_deposits(base_date, rng)
    appeals = build_appeals(base_date, goods, rng)
    seed_model = {
        "ads_merchant_profile": build_profile_rows(
            base_date, pt_values, orders, refunds, tickets, repays, coupons, goods, deposits, appeals, scm_rows),
        "dim_merchant_df": build_dim_rows(base_date),
        "dwm_trade_order_detail_di": orders,
        "dwm_trade_refund_detail_di": refunds,
        "dwm_goods_detail_df": goods,
        "dwm_cs_ticket_detail_di": tickets,
        "dwm_cs_repay_detail_df": repays,
        "dwm_coupon_detail_di": coupons,
        "dwm_scm_detail_di": scm_rows,
        "dwd_merchant_deposit_recharge_df": deposits,
        "dwd_merchant_appeal_detail_df": appeals,
    }
    validate_seed_model(seed_model, base_date)
    return seed_model


def validate_seed_model(seed_model: dict[str, list[dict]], base_date: date) -> None:
    """Fail before TRUNCATE when generated cross-table entities do not align."""

    orders = seed_model["dwm_trade_order_detail_di"]
    refunds = seed_model["dwm_trade_refund_detail_di"]
    tickets = seed_model["dwm_cs_ticket_detail_di"]
    repays = seed_model["dwm_cs_repay_detail_df"]
    goods = seed_model["dwm_goods_detail_df"]
    scm_rows = seed_model["dwm_scm_detail_di"]
    appeals = seed_model["dwd_merchant_appeal_detail_df"]

    order_by_sub = {str(row["sub_order_id"]): row for row in orders}
    ticket_by_id = {str(row["ticket_id"]): row for row in tickets}
    goods_by_spu = {str(row["spu_id"]): row for row in goods}

    def require_numeric_product_id(row: dict, table: str) -> str:
        spu_id = str(row.get("spu_id") or "")
        if not spu_id.isdigit():
            raise ValueError(f"{table} has non-canonical spu_id={spu_id!r}")
        if spu_id not in goods_by_spu:
            raise ValueError(f"{table} spu_id={spu_id!r} is absent from goods")
        return spu_id

    for order in orders:
        require_numeric_product_id(order, "orders")
    for row in [*scm_rows, *appeals]:
        require_numeric_product_id(row, "product_dimension")

    for ticket in tickets:
        spu_id = require_numeric_product_id(ticket, "tickets")
        order = order_by_sub.get(str(ticket.get("sub_order_id") or ""))
        if order is None:
            raise ValueError(f"ticket {ticket.get('ticket_id')} has no source order")
        if spu_id != str(order.get("spu_id")) or ticket.get("spu_name") != order.get("spu_name"):
            raise ValueError(f"ticket {ticket.get('ticket_id')} product differs from source order")

    for refund in refunds:
        order = order_by_sub.get(str(refund.get("sub_order_id") or ""))
        if order is None:
            raise ValueError(f"refund {refund.get('refund_id')} has no source order")
        if refund.get("spu_name") != order.get("spu_name"):
            raise ValueError(f"refund {refund.get('refund_id')} product differs from source order")

    for repay in repays:
        ticket = ticket_by_id.get(str(repay.get("ticket_id") or ""))
        order = order_by_sub.get(str(repay.get("sub_order_id") or ""))
        if ticket is None or order is None:
            raise ValueError(f"repay {repay.get('bill_id')} has broken ticket/order lineage")
        if repay.get("order_id") != order.get("order_id"):
            raise ValueError(f"repay {repay.get('bill_id')} order differs from source order")

    recent_start = base_date - timedelta(days=29)
    recent_counts: dict[str, int] = defaultdict(int)
    for ticket in tickets:
        pt = datetime.strptime(str(ticket["pt"])[:10], "%Y-%m-%d").date()
        if recent_start <= pt <= base_date:
            recent_counts[str(ticket["spu_id"])] += 1
    if not recent_counts or max(recent_counts.values()) <= 1:
        raise ValueError("recent ticket data must contain a meaningful product ranking")
    top_count = max(recent_counts.values())
    if sum(1 for value in recent_counts.values() if value == top_count) != 1:
        raise ValueError("recent ticket data must have one deterministic top product")


def seed_value(table: str, column: dict[str, str], row_index: int, pt_value: date):
    name = column["name"].lower()
    data_type = column["type"]
    base_number = row_index + 1

    if name == "pt":
        return pt_value.strftime("%Y-%m-%d")

    if "time" in name and data_type in {"varchar", "char", "string", "text", "datetime", "timestamp"}:
        return random_time_string(pt_value, hash((table, name, row_index)) & 0xFFFFFFFF)

    if name in {"seller_id", "merchant_id"}:
        if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
            return 100
        return "100"

    if name == "user_id":
        return f"user_{table}_{base_number:03d}"

    if name.endswith("_id") or "_id" in name:
        if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
            return 100000 + base_number
        return f"{name}_{base_number:03d}"

    if data_type == "date":
        return pt_value.strftime("%Y-%m-%d")

    if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
        if name.startswith("is_"):
            return base_number % 2
        if name.endswith("_cnt") or "_cnt_" in name or name.endswith("_code"):
            return (base_number % 9) + 1
        if "status" in name:
            return (base_number % 5) + 1
        if "score" in name:
            return (base_number % 5) + 1
        if "amt" in name or "price" in name or "salary" in name:
            return 1000 + base_number * 7
        return 100 + base_number

    if data_type in {"double", "float", "decimal"}:
        return round(10.5 + base_number * 1.11, 2)

    if "mobile" in name:
        return f"1380000{base_number:04d}"

    if "email" in name:
        return f"seed{base_number}@example.com"

    if "url" in name:
        return f"https://example.com/{table}/{name}/{base_number}"

    if "json" in name:
        return json.dumps({"table": table, "row": base_number, "column": name}, ensure_ascii=True)

    if "list" in name:
        return json.dumps([f"{name}_{base_number}", f"{name}_{base_number + 1}"], ensure_ascii=True)

    if "address" in name:
        return f"Address_{base_number}_Road_{pt_value:%m%d}"

    if "note" in name or "remark" in name or "desc" in name or "content" in name or "reason" in name:
        return f"{name}_{table}_{base_number}"

    if "name" in name:
        return f"{name}_{base_number}"

    if "title" in name:
        return f"{name}_{base_number}"

    if "period" in name:
        return pt_value.strftime("%Y-%m-%d")

    if name == "currency":
        return "CNY"

    if "voucher" in name:
        return f"voucher_{base_number}"

    return f"{name}_{base_number}"


def build_rows(
        table: str,
        columns: list[dict[str, str]],
        pt_values: list[date],
        seed_model: dict[str, list[dict]],
) -> list[list]:
    if table in seed_model:
        return [row_from_map(table, columns, row, row_index) for row_index, row in enumerate(seed_model[table])]
    row_count = PROFILE_TABLE_ROW_COUNT if table == "ads_merchant_profile" else TABLE_ROW_COUNT
    rows: list[list] = []
    for row_index in range(row_count):
        pt_value = pt_values[row_index % len(pt_values)]
        row = [apply_length_limit(seed_value(table, column, row_index, pt_value), column) for column in columns]
        rows.append(row)
    return rows


def alter_partitions(tables: list[str]) -> None:
    try:
        run_mysql("ADMIN SET FRONTEND CONFIG ('dynamic_partition_check_interval_seconds' = '5');", DATABASE)
    except subprocess.CalledProcessError:
        return
    properties = (
        "'dynamic_partition.enable'='true',"
        "'dynamic_partition.time_unit'='DAY',"
        "'dynamic_partition.start'='-120',"
        "'dynamic_partition.end'='3',"
        "'dynamic_partition.prefix'='p',"
        "'dynamic_partition.create_history_partition'='true',"
        "'dynamic_partition.history_partition_num'='120'"
    )
    for table in tables:
        try:
            run_mysql(f"ALTER TABLE {DATABASE}.{table} SET ({properties});", DATABASE)
        except subprocess.CalledProcessError:
            continue


def build_sql(tables: list[str], columns_by_table: dict[str, list[dict[str, str]]], base_date: date) -> str:
    max_days = max(TABLE_ROW_COUNT, PROFILE_TABLE_ROW_COUNT)
    pt_values = [base_date - timedelta(days=offset) for offset in range(max_days - 1, -1, -1)]
    seed_model = build_seed_model(base_date)
    statements = ["SET enable_insert_strict = true;"]

    for table in tables:
        columns = columns_by_table[table]
        column_names = ", ".join(f"`{column['name']}`" for column in columns)
        rows = build_rows(table, columns, pt_values, seed_model)
        statements.append(f"TRUNCATE TABLE {DATABASE}.{table};")
        for offset in range(0, len(rows), BATCH_SIZE):
            chunk = rows[offset: offset + BATCH_SIZE]
            values_sql = [
                "(" + ", ".join(format_sql_value(value) for value in row) + ")"
                for row in chunk
            ]
            statements.append(
                f"INSERT INTO {DATABASE}.{table} ({column_names}) VALUES\n" + ",\n".join(values_sql) + ";"
            )

    return "\n\n".join(statements) + "\n"


def execute_sql_statements(sql_text: str) -> None:
    statements = [segment.strip() for segment in sql_text.split(";") if segment.strip()]
    for statement in statements:
        run_mysql(statement + ";", DATABASE)


def verify_counts(tables: list[str]) -> None:
    for table in tables:
        count_sql = (
            f"SELECT '{table}', COUNT(*), MIN(pt), MAX(pt) FROM {DATABASE}.{table};"
            if table != "dim_merchant_df"
            else f"SELECT '{table}', COUNT(*), MIN(pt), MAX(pt) FROM {DATABASE}.{table};"
        )
        print(run_mysql(count_sql, DATABASE))


def main() -> int:
    tables = get_tables()
    columns_by_table = get_columns()
    base_date = get_current_date()

    alter_partitions(tables)
    time.sleep(8)

    sql_text = build_sql(tables, columns_by_table, base_date)
    OUTPUT_SQL.write_text(sql_text, encoding="utf-8")

    execute_sql_statements(sql_text)

    verify_counts(tables)
    print(f"SQL written to {OUTPUT_SQL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
