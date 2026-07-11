from __future__ import annotations

import re
from typing import Any


def answer_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_metric_value_for_answer(value: Any, metric_key: str, label: str = "") -> str:
    text = format_cell(value)
    numeric = answer_numeric_value(value)
    if numeric is None:
        return text
    metric_text = "%s %s" % (metric_key or "", label or "")
    if re.search(r"(gmv|amt|amount|金额|赔付|退款|优惠|补贴)", metric_text, flags=re.I):
        if float(numeric).is_integer():
            return "%s元" % int(numeric)
        return ("%s元" % ("%.2f" % numeric)).replace(".00元", "元")
    return text


def extract_question_time_phrase(question: str) -> str:
    text = str(question or "")
    for pattern in [
        r"最近\s*\d+\s*[天日周月]",
        r"近\s*\d+\s*[天日周月]",
        r"过去\s*\d+\s*[天日周月]",
        r"昨天",
        r"今日",
        r"今天",
        r"本周",
        r"本月",
    ]:
        match = re.search(pattern, text)
        if match:
            return re.sub(r"\s+", "", match.group(0))
    return ""


def humanize_column_name(column: str) -> str:
    text = str(column or "").strip()
    dictionary = {
        "order": "订单",
        "detail": "明细",
        "cnt": "数量",
        "amt": "金额",
        "gmv": "GMV",
        "refund": "退款",
        "return": "退货",
        "rate": "比例",
        "pay": "支付",
        "user": "用户",
        "ticket": "工单",
        "goods": "商品",
        "spu": "商品",
        "create": "创建",
        "time": "时间",
    }
    parts = [dictionary.get(part, "") for part in re.split(r"[_\s]+", text.lower())]
    label = "".join(part for part in parts if part)
    return label or text


def identifier_like_column(column: str) -> bool:
    text = str(column or "").strip().lower()
    return text in {"seller_id", "merchant_id", "user_id", "pt"} or text.endswith("_id") or text.endswith("_no")


def format_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ")[:80]
