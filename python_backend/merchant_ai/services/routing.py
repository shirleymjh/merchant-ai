from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from merchant_ai.models import (
    ExtractedKeywords,
    QuestionCategory,
    QuestionRoute,
    RecallBundle,
    RouteObjectRef,
    RouteSlots,
    RouteTimeWindow,
    RouteTopicCandidate,
    RoutingDecision,
    TOPIC_TO_CATEGORY,
    TopicRoutingDecision,
)


BUSINESS_KEYWORDS: Dict[QuestionCategory, List[str]] = {
    QuestionCategory.TRADE: [
        "订单",
        "子订单",
        "下单",
        "下单数",
        "下单量",
        "订单数",
        "订单量",
        "order",
        "order_detail_cnt",
        "销量",
        "交易",
        "gmv",
        "GMV",
        "支付",
        "成交",
        "客单价",
        "签收",
        "发货超时",
        "物流超时",
    ],
    QuestionCategory.REFUND: ["退款", "退货", "售后", "退款率", "退货率", "refund", "refund_rate", "refund_bill_cnt"],
    QuestionCategory.CS_TICKET: ["工单", "客服", "催单", "二次开启", "评价分", "ticket", "cs_ticket"],
    QuestionCategory.COMPENSATION: ["赔付", "赔款", "理赔", "补偿", "repay", "compensation"],
    QuestionCategory.COUPON: ["优惠", "优惠券", "券", "券活动", "折扣", "补贴", "coupon", "activity"],
    QuestionCategory.GOODS: ["商品", "审核", "上架", "spu", "sku", "类目", "资质", "新发布", "goods"],
    QuestionCategory.SCM: ["供应链", "履约", "入库", "质检", "鉴定", "出库", "仓库", "scm"],
    QuestionCategory.MERCHANT_OTHER: ["保证金", "申诉", "处罚", "费率", "结算"],
    QuestionCategory.IDENTITY: ["营业执照", "统一社会信用代码", "公司名称", "联系人", "地址", "银行卡", "开户行", "发票"],
    QuestionCategory.PLATFORM_RULE: ["规则", "处罚规则", "平台规则", "要求", "标准"],
}

ACTION_KEYWORDS = [
    "为什么",
    "原因",
    "影响",
    "分析",
    "对比",
    "环比",
    "同比",
    "同时",
    "分别",
    "并且",
    "综合",
    "关联",
    "对应",
    "趋势",
    "走势",
    "变化",
    "同步",
    "上升",
    "下降",
    "波动",
    "异常",
    "风险",
]
TIME_PATTERNS = [
    re.compile(r"(最近|近|过去|前)?\s*\d{1,3}\s*[天日]"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(周|星期|礼拜)"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(个月|月)"),
]


class KeywordExtractService:
    def extract(self, question: str) -> ExtractedKeywords:
        text = question or ""
        business: List[str] = []
        for words in BUSINESS_KEYWORDS.values():
            for word in words:
                if word.lower() in text.lower() and word not in business:
                    business.append(word)
        action = [word for word in ACTION_KEYWORDS if word in text]
        time_words: List[str] = []
        for pattern in TIME_PATTERNS:
            time_words.extend(match.group(0).strip() for match in pattern.finditer(text))
        for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月", "这个月"]:
            if word in text and word not in time_words:
                time_words.append(word)
        keywords = []
        for item in business + time_words + action:
            if item and item not in keywords:
                keywords.append(item)
        return ExtractedKeywords(
            keywords=keywords,
            business_keywords=business,
            time_keywords=time_words,
            action_keywords=action,
        )


class QuestionRoutingService:
    def route(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> RoutingDecision:
        normalized = (question or "").strip().lower()
        if not normalized:
            return RoutingDecision(route=QuestionRoute.INVALID, reason="空问题")
        if re.match(r"^(你好|您好|hi|hello|hey|在吗|嗨|哈喽|早上好|下午好|晚上好)[!！。,.，\s]*$", normalized, re.I):
            return RoutingDecision(route=QuestionRoute.GREETING, reason="寒暄问题")
        if self._is_store_overview_question(normalized):
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=True, reason="店铺整体经营问题")
        if self._is_ambiguous_question(normalized, keywords, recall_bundle):
            return RoutingDecision(route=QuestionRoute.INVALID, reason="问题表达不明确，建议补充业务对象或查询目标")
        simple_detail = self._is_simple_detail_lookup(normalized, keywords, recall_bundle)
        complex_question = (not simple_detail) and (
            len(normalized) >= 24
            or any(word in normalized for word in ACTION_KEYWORDS)
            or self._has_multiple_time_ranges(normalized)
            or self._matched_domain_count(normalized) >= 2
            or (recall_bundle and len(recall_bundle.items) >= 3 and not recall_bundle.has_strong_match() and len(normalized) >= 24)
        )
        return RoutingDecision(
            route=QuestionRoute.BUSINESS,
            complex=complex_question,
            reason="业务问题，单一明细查询" if simple_detail else ("业务问题，可能需要进一步拆解" if complex_question else "业务问题"),
        )

    def _is_ambiguous_question(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> bool:
        has_signal = (
            (keywords is not None and bool(keywords.business_keywords))
            or (recall_bundle is not None and recall_bundle.has_strong_match())
            or self._matched_domain_count(question) > 0
        )
        if has_signal:
            return False
        if len(question) <= 2:
            return True
        if re.match(r"^(这个|那个|这个呢|那个呢|在吗|看下|看一下|帮我看下|帮我看看)[!！。,.，\s]*$", question):
            return True
        if not keywords.business_keywords and question in {"分析", "分析问题", "原因", "看看原因", "看下原因", "是否异常"}:
            return True
        if any(item in question for item in ["我最近怎么样", "经营情况怎么样", "帮我看看经营情况", "店铺最近怎么样"]):
            return True
        return bool(re.match(r"^(什么情况|啥情况|怎么回事|什么意思|怎么看|怎么办|怎么弄|为什么|有问题|异常了?)[!！。,.，\s]*$", question))

    def _has_multiple_time_ranges(self, question: str) -> bool:
        return sum(1 for pattern in TIME_PATTERNS for _ in pattern.finditer(question)) >= 2

    def _has_any_time_range(self, question: str) -> bool:
        return self._has_multiple_time_ranges(question) or any(pattern.search(question) for pattern in TIME_PATTERNS) or any(
            word in question for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月", "这个月"]
        )

    def _is_simple_detail_lookup(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> bool:
        if not any(word in question for word in ["明细", "详情", "列表", "记录", "单号", "流水"]):
            return False
        if not self._has_any_time_range(question) or self._has_multiple_time_ranges(question):
            return False
        if any(word in question for word in ACTION_KEYWORDS):
            return False
        if self._matched_domain_count(question) >= 2:
            return False
        if keywords and any(any(flag in action for flag in ["分析", "对比", "优化", "判断", "解释", "排查"]) for action in keywords.action_keywords):
            return False
        return not recall_bundle or not recall_bundle.items or all((item.answer_mode or "").upper() == "DETAIL" for item in recall_bundle.items)

    def _is_store_overview_question(self, question: str) -> bool:
        return any(word in question for word in ["店铺整体", "整体经营", "经营概况", "经营情况", "店铺情况", "店铺概况"]) or (
            any(word in question for word in ["店铺", "商家", "我店"])
            and any(word in question for word in ["整体", "经营", "概况", "情况", "怎么样", "异常", "关注"])
        )

    def _matched_domain_count(self, question: str) -> int:
        return sum(1 for words in BUSINESS_KEYWORDS.values() if any(word.lower() in question.lower() for word in words))


class RouteSlotExtractor:
    OBJECT_PATTERNS = [
        ("sub_order_id", re.compile(r"(?<![A-Za-z0-9_])sub_order_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("order_id", re.compile(r"(?<![A-Za-z0-9_])order_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("spu_id", re.compile(r"(?<![A-Za-z0-9_])spu_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("sku_id", re.compile(r"(?<![A-Za-z0-9_])sku_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("refund_id", re.compile(r"(?<![A-Za-z0-9_])refund_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("ticket_id", re.compile(r"(?<![A-Za-z0-9_])ticket_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("bill_id", re.compile(r"(?<![A-Za-z0-9_])bill_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
        ("coupon_id", re.compile(r"(?<![A-Za-z0-9_])coupon_id[_:=：-]+[A-Za-z0-9_-]+", re.I)),
    ]
    OBJECT_TOPICS = {
        "order_id": [QuestionCategory.TRADE],
        "sub_order_id": [QuestionCategory.TRADE],
        "spu_id": [QuestionCategory.GOODS, QuestionCategory.TRADE],
        "sku_id": [QuestionCategory.GOODS, QuestionCategory.TRADE],
        "refund_id": [QuestionCategory.REFUND],
        "ticket_id": [QuestionCategory.CS_TICKET],
        "bill_id": [QuestionCategory.COMPENSATION],
        "coupon_id": [QuestionCategory.COUPON],
    }
    WRITE_TERMS = ["删除", "修改", "更新", "创建", "重建", "写入", "导入", "新增", "truncate", "drop", "insert", "update", "delete"]
    RISK_TERMS = ["平台规则", "规则", "处罚", "罚", "资质", "营业执照", "保证金", "敏感"]

    def extract(self, question: str, keywords: ExtractedKeywords) -> RouteSlots:
        text = question or ""
        object_refs = self._object_refs(text)
        time_window = self._time_window(text, keywords)
        operation = "write_requested" if any(term.lower() in text.lower() for term in self.WRITE_TERMS) else "read"
        analysis_signals = self._analysis_signals(keywords)
        topic_candidates = self._topic_candidates(text, object_refs)
        warnings: List[str] = []
        risk_level = self._risk_level(text, operation)
        if operation == "write_requested":
            warnings.append("WRITE_OPERATION_REQUESTED")
        if not topic_candidates:
            warnings.append("NO_EXPLICIT_TOPIC")
        if len(topic_candidates) >= 5:
            warnings.append("BROAD_TOPIC_SET")
        confidence = self._confidence(topic_candidates, object_refs, time_window, warnings)
        return RouteSlots(
            object_refs=object_refs,
            time_window=time_window,
            operation=operation,
            risk_level=risk_level,
            topic_candidates=topic_candidates,
            analysis_signals=analysis_signals,
            route_confidence=confidence,
            route_warnings=warnings,
        )

    def _object_refs(self, text: str) -> List[RouteObjectRef]:
        refs: List[RouteObjectRef] = []
        seen: Set[tuple[str, str]] = set()
        for ref_type, pattern in self.OBJECT_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                value = raw.replace("：", "_").replace(":", "_").replace("=", "_").replace("-", "_")
                if ref_type == "order_id" and value.lower().startswith("sub_order_id"):
                    continue
                identity = (ref_type, value.lower())
                if identity in seen:
                    continue
                seen.add(identity)
                refs.append(RouteObjectRef(ref_type=ref_type, value=value, raw=raw, confidence=0.95))
        return refs

    def _time_window(self, text: str, keywords: ExtractedKeywords) -> RouteTimeWindow:
        raw = (keywords.time_keywords[0] if keywords and keywords.time_keywords else "") or self._first_time_expression(text)
        days = extract_days(text, default=0)
        return RouteTimeWindow(days=days, raw=raw, needs_freshness_check=days > 0 and days <= 2)

    def _first_time_expression(self, text: str) -> str:
        for pattern in TIME_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()
        for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月", "这个月"]:
            if word in text:
                return word
        return ""

    def _analysis_signals(self, keywords: ExtractedKeywords) -> List[str]:
        if keywords and keywords.action_keywords:
            return ["weak_analysis_hint"]
        return []

    def _topic_candidates(self, text: str, object_refs: List[RouteObjectRef]) -> List[RouteTopicCandidate]:
        by_topic: Dict[QuestionCategory, Dict[str, object]] = {}
        lowered = text.lower()
        for category, words in BUSINESS_KEYWORDS.items():
            evidence = [word for word in words if word.lower() in lowered]
            if not evidence:
                continue
            by_topic[category] = {"score": len(evidence), "evidence": evidence[:8]}
        for ref in object_refs:
            for category in self.OBJECT_TOPICS.get(ref.ref_type, []):
                payload = by_topic.setdefault(category, {"score": 0, "evidence": []})
                payload["score"] = int(payload.get("score") or 0) + 2
                evidence = list(payload.get("evidence") or [])
                if ref.ref_type not in evidence:
                    evidence.append(ref.ref_type)
                payload["evidence"] = evidence[:8]
        ordered = []
        for category in topic_domain_order():
            payload = by_topic.get(category)
            if not payload:
                continue
            ordered.append(
                RouteTopicCandidate(
                    topic=category,
                    score=int(payload.get("score") or 0),
                    evidence=[str(item) for item in payload.get("evidence") or []],
                )
            )
        return ordered

    def _risk_level(self, text: str, operation: str) -> str:
        if operation == "write_requested":
            return "high_risk"
        if any(term in text for term in self.RISK_TERMS):
            return "rule_sensitive"
        return "normal"

    def _confidence(
        self,
        topic_candidates: List[RouteTopicCandidate],
        object_refs: List[RouteObjectRef],
        time_window: RouteTimeWindow,
        warnings: List[str],
    ) -> float:
        top_score = max([item.score for item in topic_candidates] or [0])
        confidence = 0.35 + 0.08 * min(top_score, 5) + 0.06 * min(len(topic_candidates), 4)
        if object_refs:
            confidence += 0.08
        if time_window.days:
            confidence += 0.04
        if warnings:
            confidence -= 0.08
        return max(0.0, min(0.95, round(confidence, 2)))


def topic_domain_order() -> List[QuestionCategory]:
    return [
        QuestionCategory.TRADE,
        QuestionCategory.REFUND,
        QuestionCategory.GOODS,
        QuestionCategory.CS_TICKET,
        QuestionCategory.COMPENSATION,
        QuestionCategory.COUPON,
        QuestionCategory.SCM,
        QuestionCategory.MERCHANT_OTHER,
        QuestionCategory.IDENTITY,
        QuestionCategory.PLATFORM_RULE,
    ]


class TopicRouterService:
    def route(
        self,
        question: str,
        keywords: ExtractedKeywords,
        context_topic: str = "",
        route_slots: Optional[RouteSlots] = None,
    ) -> TopicRoutingDecision:
        text = question or ""
        if context_topic and context_topic in TOPIC_TO_CATEGORY:
            primary = TOPIC_TO_CATEGORY[context_topic]
            return TopicRoutingDecision(
                primary_topic=primary,
                candidate_topics=[primary],
                confidence=0.82,
                reason="继承会话 Topic；primaryTopic 仅兼容字段，不表示 anchor",
            )

        scores: Dict[QuestionCategory, int] = {}
        for category, words in BUSINESS_KEYWORDS.items():
            scores[category] = sum(1 for word in words if word.lower() in text.lower())
        if route_slots:
            for candidate in route_slots.topic_candidates:
                try:
                    category = QuestionCategory(candidate.topic)
                except Exception:
                    continue
                scores[category] = max(scores.get(category, 0), int(candidate.score or 0))
        candidates = self._explicit_topics(scores)
        if not candidates:
            return TopicRoutingDecision(
                primary_topic=QuestionCategory.UNKNOWN,
                clarification_required=False,
                reason="未识别出显式业务 topic；保持开放 scope，交由后续 LLM/知识检索发现缺口",
            )
        top_score = max(scores.get(category, 0) for category in candidates)
        confidence = min(0.95, 0.45 + 0.08 * len(candidates) + 0.08 * top_score)
        primary_topic = route_primary_topic(candidates)
        return TopicRoutingDecision(
            primary_topic=primary_topic,
            candidate_topics=candidates,
            dimension_topics=[] if primary_topic == QuestionCategory.UNKNOWN else candidates[1:],
            confidence=confidence,
            clarification_required=False,
            reason=(
                "按显式业务词选择候选 topic；多 topic 时 primaryTopic 保持 UNKNOWN，"
                "不表示 anchor，避免把召回范围误当主 anchor"
                if primary_topic == QuestionCategory.UNKNOWN
                else "按显式业务词选择 topic；primaryTopic 仅兼容字段，不表示 anchor"
            ),
        )

    def _explicit_topics(self, scores: Dict[QuestionCategory, int]) -> List[QuestionCategory]:
        return [category for category in topic_domain_order() if scores.get(category, 0) > 0]


def route_primary_topic(candidates: List[QuestionCategory]) -> QuestionCategory:
    """Only a single-topic route can safely expose a compatibility primary topic."""
    return candidates[0] if len(candidates) == 1 else QuestionCategory.UNKNOWN


def extract_days(question: str, default: int = 7) -> int:
    text = question or ""
    for pattern, multiplier in [
        (re.compile(r"(最近|近|过去|前)\s*(\d{1,3})\s*[天日]"), 1),
        (re.compile(r"(最近|近|过去|前)\s*(\d{1,2})\s*(周|星期|礼拜)"), 7),
        (re.compile(r"(最近|近|过去|前)\s*(\d{1,2})\s*个月"), 30),
    ]:
        match = pattern.search(text)
        if match:
            return max(1, min(int(match.group(2)) * multiplier, 365))
    if "昨天" in text or "昨日" in text:
        return 1
    if "今天" in text or "今日" in text:
        return 1
    return default
