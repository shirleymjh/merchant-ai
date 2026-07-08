from __future__ import annotations

from typing import Any, Dict, List


GOLDEN_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "scm_inbound_7d",
        "question": "最近 7 天入库量怎么样",
        "expectedIntent": "metric_query",
        "expectedTopics": ["SCM"],
        "expectedSourceTypes": ["SEMANTIC_METRIC"],
        "expectedMetrics": ["inbound_cnt"],
    },
    {
        "id": "trade_order_count_7d",
        "question": "最近7天订单量是多少？",
        "expectedIntent": "metric_query",
        "expectedTopics": ["TRADE"],
        "expectedSourceTypes": ["SEMANTIC_METRIC"],
        "expectedMetrics": ["order_detail_cnt"],
    },
    {
        "id": "refund_metric_dispute",
        "question": "退款率口径是不是退款单数除以下单订单数？",
        "expectedIntent": "rule_data_mixed",
        "expectedTopics": ["REFUND"],
        "expectedDisclosure": ["metric_dispute"],
    },
    {
        "id": "cross_table_refund_goods",
        "question": "最近 7 天有退款的订单，关联看一下对应商品发布时间。",
        "expectedIntent": "multi_hop",
        "expectedTopics": ["REFUND", "GOODS"],
        "expectedSourceTypes": ["SEMANTIC_RELATIONSHIP", "SEMANTIC_METRIC"],
    },
    {
        "id": "platform_rule_only",
        "question": "商家入驻资质规则是什么？",
        "expectedIntent": "rule_only",
        "expectedTopics": ["PLATFORM_RULE"],
        "expectedSourceTypes": ["BASE_WIKI"],
    },
]


def evaluation_observability_record(debug_trace: Dict[str, Any]) -> Dict[str, Any]:
    harness = (debug_trace or {}).get("harness") or {}
    observability = harness.get("observability") or (debug_trace or {}).get("observability") or {}
    knowledge = harness.get("knowledgeRetrieval") or {}
    return {
        "selectedMemoryIds": observability.get("selectedMemoryIds") or [],
        "semanticRefIds": observability.get("semanticRefIds") or [],
        "contextHash": observability.get("contextHash") or "",
        "validationGapCount": len(observability.get("validationGaps") or []),
        "evidenceGapCount": len(observability.get("evidenceGaps") or []),
        "repairCount": int(observability.get("repairCount") or 0),
        "recallSourceRefs": knowledge.get("sourceRefs") or [],
        "recallRounds": knowledge.get("rounds") or [],
    }
