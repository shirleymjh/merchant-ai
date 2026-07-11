from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from merchant_ai.models import (
    ExtractedKeywords,
    KeywordMention,
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
    "建议",
    "优化",
    "改善",
    "怎么办",
    "排查",
    "明细",
    "详情",
    "最高",
    "最低",
    "最多",
    "最少",
]
DIMENSION_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "商品": {"key": "spu_id", "topic": QuestionCategory.GOODS},
    "spu": {"key": "spu_id", "topic": QuestionCategory.GOODS},
    "sku": {"key": "sku_id", "topic": QuestionCategory.GOODS},
    "类目": {"key": "category", "topic": QuestionCategory.GOODS},
    "渠道": {"key": "channel", "topic": QuestionCategory.TRADE},
    "日期": {"key": "pt", "topic": QuestionCategory.UNKNOWN},
    "每天": {"key": "pt", "topic": QuestionCategory.UNKNOWN},
    "按天": {"key": "pt", "topic": QuestionCategory.UNKNOWN},
    "退款原因": {"key": "refund_reason", "topic": QuestionCategory.REFUND},
    "退货原因": {"key": "refund_reason", "topic": QuestionCategory.REFUND},
    "工单类型": {"key": "ticket_type", "topic": QuestionCategory.CS_TICKET},
    "处罚类型": {"key": "punish_type", "topic": QuestionCategory.MERCHANT_OTHER},
}
RANKING_PATTERNS = [
    re.compile(r"前\s*\d+"),
    re.compile(r"top\s*\d+", re.I),
    re.compile(r"最高|最低|最多|最少|排名|排行"),
]
GENERIC_METRIC_ALIASES = {
    "金额",
    "数量",
    "次数",
    "总量",
    "占比",
    "比例",
    "平均值",
    "最大值",
    "最小值",
    "订单",
    "支付",
    "交易",
    "退款",
    "退货",
    "商品",
    "优惠",
    "工单",
    "客服",
    "赔付",
    "保证金",
}
TIME_PATTERNS = [
    re.compile(r"(最近|近|过去|前)?\s*\d{1,3}\s*[天日]"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(周|星期|礼拜)"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(个月|月)"),
]


class KeywordExtractService:
    def __init__(self, topic_assets: Any = None):
        self.topic_assets = topic_assets
        self._semantic_metrics = self._build_semantic_metric_lexicon(topic_assets)
        self._semantic_matcher = PhraseMatcher(list(self._semantic_metrics))

    def reload_semantic_lexicon(self) -> None:
        self._semantic_metrics = self._build_semantic_metric_lexicon(self.topic_assets)
        self._semantic_matcher = PhraseMatcher(list(self._semantic_metrics))

    def extract(self, question: str) -> ExtractedKeywords:
        text = question or ""
        normalized = normalize_keyword_text(text)
        topic_mentions = self._topic_mentions(normalized)
        metric_mentions = self._metric_mentions(normalized)
        dimension_mentions = self._dimension_mentions(normalized)
        action = longest_distinct_matches(normalized, ACTION_KEYWORDS)
        time_words: List[str] = []
        for pattern in TIME_PATTERNS:
            time_words.extend(match.group(0).strip() for match in pattern.finditer(text))
        for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月", "这个月"]:
            if word in text and word not in time_words:
                time_words.append(word)
        ranking = [match.group(0).strip() for pattern in RANKING_PATTERNS for match in pattern.finditer(normalized)]
        ranking = dedupe_ordered(ranking)
        ambiguous_metrics = ambiguous_metric_phrases(metric_mentions)
        all_mentions = dedupe_mentions([*metric_mentions, *dimension_mentions, *topic_mentions])
        negated_segments = extract_negated_segments(normalized)
        excluded_mentions = [
            item
            for item in all_mentions
            if any(item.phrase and item.phrase in segment for segment in negated_segments)
        ]
        mentions = [item for item in all_mentions if item not in excluded_mentions]
        metric_mentions = [item for item in metric_mentions if item in mentions]
        dimension_mentions = [item for item in dimension_mentions if item in mentions]
        topic_mentions = [item for item in topic_mentions if item in mentions]
        topic_scores = self._topic_scores(mentions)
        topic_keywords = dedupe_ordered([item.phrase for item in topic_mentions])
        metric_keywords = dedupe_ordered([item.phrase for item in metric_mentions])
        dimension_keywords = dedupe_ordered([item.phrase for item in dimension_mentions])
        business = (
            dedupe_ordered([*metric_keywords, *dimension_keywords, *topic_keywords])
            if self._semantic_metrics
            else legacy_business_matches(text)
        )
        keywords: List[str] = []
        keyword_actions = action if self._semantic_metrics else legacy_recall_actions(action)
        keyword_ranking = ranking if self._semantic_metrics else []
        for item in business + time_words + keyword_actions + keyword_ranking:
            if item and item not in keywords:
                keywords.append(item)
        analysis_intent = classify_analysis_intent(normalized, action, ranking)
        confidence = keyword_confidence(topic_scores, metric_mentions, dimension_mentions, normalized, ambiguous_metrics)
        unresolved = [
            phrase
            for phrase in ["上面", "上述", "这个", "那个", "它们", "这些", "前面"]
            if phrase in normalized and not metric_mentions
        ]
        return ExtractedKeywords(
            normalized_question=normalized,
            keywords=keywords,
            business_keywords=business,
            topic_keywords=topic_keywords,
            metric_keywords=metric_keywords,
            dimension_keywords=dimension_keywords,
            time_keywords=time_words,
            action_keywords=action,
            ranking_keywords=ranking,
            mentions=mentions,
            topic_scores={key: round(value, 3) for key, value in topic_scores.items()},
            analysis_intent=analysis_intent,
            confidence=confidence,
            unresolved_phrases=unresolved,
            excluded_topics=dedupe_topics(
                [item.topic for item in excluded_mentions if item.topic != QuestionCategory.UNKNOWN]
            ),
            excluded_metric_keywords=dedupe_ordered(
                [item.phrase for item in excluded_mentions if item.kind == "metric"]
            ),
            ambiguous_metric_keywords=ambiguous_metrics,
        )

    def _topic_mentions(self, normalized: str) -> List[KeywordMention]:
        mentions: List[KeywordMention] = []
        for category, words in BUSINESS_KEYWORDS.items():
            for phrase in longest_distinct_matches(normalized, words):
                mentions.append(
                    KeywordMention(
                        phrase=phrase,
                        canonical_key=category.value,
                        display_name=phrase,
                        kind="topic",
                        topic=category,
                        score=1.5,
                        source="routing_lexicon",
                    )
                )
        return mentions

    def _metric_mentions(self, normalized: str) -> List[KeywordMention]:
        if not self._semantic_metrics:
            return []
        selected_phrases = self._semantic_matcher.match(normalized)
        mentions: List[KeywordMention] = []
        for phrase in selected_phrases:
            for entry in self._semantic_metrics.get(normalize_keyword_text(phrase), [])[:4]:
                mentions.append(
                    KeywordMention(
                        phrase=phrase,
                        canonical_key=str(entry.get("metricKey") or ""),
                        display_name=str(entry.get("businessName") or phrase),
                        kind="metric",
                        topic=entry.get("topic") or QuestionCategory.UNKNOWN,
                        score=float(entry.get("score") or 3.0),
                        source="semantic_metric",
                    )
                )
        return mentions

    def _dimension_mentions(self, normalized: str) -> List[KeywordMention]:
        mentions: List[KeywordMention] = []
        for phrase in longest_distinct_matches(normalized, list(DIMENSION_KEYWORDS)):
            entry = DIMENSION_KEYWORDS[phrase.lower() if phrase.lower() in DIMENSION_KEYWORDS else phrase]
            mentions.append(
                KeywordMention(
                    phrase=phrase,
                    canonical_key=str(entry["key"]),
                    display_name=phrase,
                    kind="dimension",
                    topic=entry["topic"],
                    score=2.0,
                    source="dimension_lexicon",
                )
            )
        return mentions

    def _topic_scores(self, mentions: List[KeywordMention]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        seen: Set[tuple[str, str, str]] = set()
        for mention in mentions:
            if mention.topic == QuestionCategory.UNKNOWN:
                continue
            mention_topic = keyword_topic_value(mention.topic)
            identity = (mention.kind, mention.canonical_key, mention_topic)
            if identity in seen:
                continue
            seen.add(identity)
            scores[mention_topic] += mention.score
        return dict(scores)

    def _build_semantic_metric_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = TOPIC_TO_CATEGORY.get(topic_name, QuestionCategory.UNKNOWN)
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for metric in topic_assets.load_table_metrics(topic_name, table):
                    metric_key = str(metric.get("canonicalMetricKey") or metric.get("metricKey") or "")
                    business_name = str(metric.get("businessName") or metric_key)
                    aliases = [business_name, metric_key, str(metric.get("metricKey") or ""), *(metric.get("aliases") or [])]
                    for alias in aliases:
                        phrase = normalize_keyword_text(str(alias or ""))
                        if not valid_metric_alias(phrase):
                            continue
                        payload = {
                            "metricKey": metric_key,
                            "businessName": business_name,
                            "topic": category,
                            "table": table,
                            "score": 3.0 if phrase == normalize_keyword_text(business_name) else 2.8,
                        }
                        if not any(
                            item.get("metricKey") == metric_key and item.get("topic") == category
                            for item in lexicon[phrase]
                        ):
                            lexicon[phrase].append(payload)
        return dict(lexicon)


def normalize_keyword_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"\s+", " ", text)


class PhraseMatcher:
    END = "__phrases__"

    def __init__(self, phrases: List[str]):
        self.trie: Dict[str, Any] = {}
        for phrase in phrases:
            normalized = normalize_keyword_text(phrase)
            if not normalized:
                continue
            node = self.trie
            for character in normalized:
                node = node.setdefault(character, {})
            node.setdefault(self.END, []).append(normalized)

    def match(self, text: str) -> List[str]:
        normalized = normalize_keyword_text(text)
        matches: List[tuple[int, int, str]] = []
        for start in range(len(normalized)):
            node = self.trie
            longest: Optional[tuple[int, int, str]] = None
            for end in range(start, len(normalized)):
                node = node.get(normalized[end])
                if node is None:
                    break
                for phrase in node.get(self.END, []):
                    if valid_phrase_boundary(normalized, start, end + 1, phrase):
                        longest = (start, end + 1, phrase)
            if longest:
                matches.append(longest)
        selected: List[tuple[int, int, str]] = []
        occupied: List[tuple[int, int]] = []
        for start, end, phrase in sorted(matches, key=lambda item: (item[1] - item[0], -item[0]), reverse=True):
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            selected.append((start, end, phrase))
            occupied.append((start, end))
        selected.sort(key=lambda item: item[0])
        return dedupe_ordered([item[2] for item in selected])


def valid_phrase_boundary(text: str, start: int, end: int, phrase: str) -> bool:
    if not re.fullmatch(r"[a-z0-9_ -]+", phrase):
        return True
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before and (before.isascii() and (before.isalnum() or before == "_"))) and not (
        after and (after.isascii() and (after.isalnum() or after == "_"))
    )


def valid_metric_alias(phrase: str) -> bool:
    if not phrase or phrase in GENERIC_METRIC_ALIASES:
        return False
    compact = phrase.replace("_", "").replace(" ", "")
    return len(compact) >= 2


def phrase_spans(text: str, phrase: str) -> List[tuple[int, int]]:
    normalized_phrase = normalize_keyword_text(phrase)
    if not normalized_phrase:
        return []
    if len(normalized_phrase) == 1 and normalized_phrase != text.strip():
        return []
    if re.fullmatch(r"[a-z0-9_ -]+", normalized_phrase):
        pattern = re.compile(r"(?<![a-z0-9_])%s(?![a-z0-9_])" % re.escape(normalized_phrase), re.I)
        return [(match.start(), match.end()) for match in pattern.finditer(text)]
    return [(match.start(), match.end()) for match in re.finditer(re.escape(normalized_phrase), text)]


def longest_distinct_matches(text: str, candidates: List[str]) -> List[str]:
    normalized = normalize_keyword_text(text)
    ranked = sorted(
        {str(candidate) for candidate in candidates if str(candidate or "").strip()},
        key=lambda value: (len(normalize_keyword_text(value)), value),
        reverse=True,
    )
    selected: List[tuple[int, int, str]] = []
    occupied: List[tuple[int, int]] = []
    for candidate in ranked:
        for start, end in phrase_spans(normalized, candidate):
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            selected.append((start, end, normalized[start:end]))
            occupied.append((start, end))
            break
    selected.sort(key=lambda item: item[0])
    return dedupe_ordered([item[2] for item in selected])


def dedupe_ordered(items: List[str]) -> List[str]:
    result: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def legacy_business_matches(text: str) -> List[str]:
    matches: List[str] = []
    lowered = str(text or "").lower()
    for words in BUSINESS_KEYWORDS.values():
        for word in words:
            if word.lower() in lowered and word not in matches:
                matches.append(word)
    return matches


def legacy_recall_actions(actions: List[str]) -> List[str]:
    added_actions = {"建议", "优化", "改善", "怎么办", "排查", "明细", "详情", "最高", "最低", "最多", "最少"}
    return [action for action in actions if action not in added_actions]


def dedupe_mentions(items: List[KeywordMention]) -> List[KeywordMention]:
    result: List[KeywordMention] = []
    seen: Set[tuple[str, str, str, str]] = set()
    for item in items:
        identity = (item.phrase, item.kind, item.canonical_key, keyword_topic_value(item.topic))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def keyword_topic_value(topic: Any) -> str:
    return str(getattr(topic, "value", topic) or QuestionCategory.UNKNOWN.value)


def extract_negated_segments(text: str) -> List[str]:
    segments: List[str] = []
    pattern = re.compile(r"(?:不看|不要看|排除|不包含|不考虑|忽略|去掉)([^，。；,;]+)")
    for match in pattern.finditer(text):
        value = str(match.group(1) or "").strip()
        if value:
            segments.append(value)
    return segments


def dedupe_topics(items: List[Any]) -> List[QuestionCategory]:
    result: List[QuestionCategory] = []
    for item in items:
        try:
            topic = QuestionCategory(keyword_topic_value(item))
        except ValueError:
            continue
        if topic not in result:
            result.append(topic)
    return result


def classify_analysis_intent(text: str, actions: List[str], ranking: List[str]) -> str:
    if any(term in text for term in ["为什么", "原因", "归因", "影响因素", "导致"]):
        return "attribution"
    if any(term in text for term in ["占比", "比例", "比率", "占了多少", "占多少"]):
        return "ratio"
    if ranking:
        return "ranking"
    if any(term in text for term in ["同比", "环比", "对比", "比较", "同时", "分别", "关联", "相关"]):
        return "comparison"
    if any(term in text for term in ["异常", "是否正常", "风险", "波动"]):
        return "anomaly"
    if any(term in text for term in ["趋势", "走势", "变化", "上升", "下降", "同步"]):
        return "trend"
    if any(term in text for term in ["建议", "优化", "改善", "怎么办"]):
        return "advice"
    if any(term in text for term in ["明细", "详情", "列表", "记录", "单号", "流水"]):
        return "detail"
    return "analysis" if actions else "lookup"


def keyword_confidence(
    topic_scores: Dict[str, float],
    metric_mentions: List[KeywordMention],
    dimension_mentions: List[KeywordMention],
    normalized_question: str,
    ambiguous_metrics: Optional[List[str]] = None,
) -> float:
    if not normalized_question:
        return 0.0
    confidence = 0.25
    if topic_scores:
        confidence += min(0.35, max(topic_scores.values()) * 0.07)
    if metric_mentions:
        confidence += 0.25
    if dimension_mentions:
        confidence += 0.08
    if len(topic_scores) > 4:
        confidence -= 0.08
    if ambiguous_metrics:
        confidence -= min(0.3, 0.12 * len(ambiguous_metrics))
    return max(0.0, min(0.98, round(confidence, 2)))


def ambiguous_metric_phrases(items: List[KeywordMention]) -> List[str]:
    candidates: Dict[str, Set[tuple[str, str]]] = defaultdict(set)
    for item in items:
        candidates[item.phrase].add((item.canonical_key, keyword_topic_value(item.topic)))
    return [phrase for phrase, identities in candidates.items() if phrase and len(identities) > 1]


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
            or self._matched_domain_count(normalized, keywords) >= 2
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
            or self._matched_domain_count(question, keywords) > 0
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
        if self._matched_domain_count(question, keywords) >= 2:
            return False
        if keywords and any(any(flag in action for flag in ["分析", "对比", "优化", "判断", "解释", "排查"]) for action in keywords.action_keywords):
            return False
        return not recall_bundle or not recall_bundle.items or all((item.answer_mode or "").upper() == "DETAIL" for item in recall_bundle.items)

    def _is_store_overview_question(self, question: str) -> bool:
        return any(word in question for word in ["店铺整体", "整体经营", "经营概况", "经营情况", "店铺情况", "店铺概况"]) or (
            any(word in question for word in ["店铺", "商家", "我店"])
            and any(word in question for word in ["整体", "经营", "概况", "情况", "怎么样", "异常", "关注"])
        )

    def _matched_domain_count(self, question: str, keywords: Optional[ExtractedKeywords] = None) -> int:
        if keywords and keywords.topic_scores:
            return len([score for score in keywords.topic_scores.values() if score > 0])
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
        topic_candidates = self._topic_candidates(text, object_refs, keywords)
        warnings: List[str] = []
        risk_level = self._risk_level(text, operation)
        if operation == "write_requested":
            warnings.append("WRITE_OPERATION_REQUESTED")
        if not topic_candidates:
            warnings.append("NO_EXPLICIT_TOPIC")
        if len(topic_candidates) >= 5:
            warnings.append("BROAD_TOPIC_SET")
        if keywords and keywords.ambiguous_metric_keywords:
            warnings.append("AMBIGUOUS_METRIC")
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

    def _topic_candidates(
        self,
        text: str,
        object_refs: List[RouteObjectRef],
        keywords: Optional[ExtractedKeywords] = None,
    ) -> List[RouteTopicCandidate]:
        by_topic: Dict[QuestionCategory, Dict[str, object]] = {}
        lowered = text.lower()
        if keywords is not None:
            for category_value, score in keywords.topic_scores.items():
                try:
                    category = QuestionCategory(category_value)
                except ValueError:
                    continue
                evidence = [
                    item.phrase
                    for item in keywords.mentions
                    if item.topic == category and item.phrase
                ]
                by_topic[category] = {"score": float(score), "evidence": dedupe_ordered(evidence)[:8]}
        else:
            for category, words in BUSINESS_KEYWORDS.items():
                evidence = [word for word in words if word.lower() in lowered]
                if not evidence:
                    continue
                by_topic[category] = {"score": float(len(evidence)), "evidence": evidence[:8]}
        for ref in object_refs:
            for category in self.OBJECT_TOPICS.get(ref.ref_type, []):
                payload = by_topic.setdefault(category, {"score": 0, "evidence": []})
                payload["score"] = float(payload.get("score") or 0) + 2.0
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
                    score=float(payload.get("score") or 0),
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
        top_score = max([item.score for item in topic_candidates] or [0.0])
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
        context_topics: Optional[List[QuestionCategory]] = None,
    ) -> TopicRoutingDecision:
        text = question or ""
        inherited_topics = dedupe_topics(list(context_topics or []))
        if not inherited_topics and context_topic:
            for item in re.split(r"[、,，|/]", context_topic):
                category = TOPIC_TO_CATEGORY.get(item.strip())
                if category and category not in inherited_topics:
                    inherited_topics.append(category)
        if inherited_topics and not (keywords and keywords.topic_scores):
            primary = route_primary_topic(inherited_topics)
            return TopicRoutingDecision(
                primary_topic=primary,
                candidate_topics=inherited_topics,
                confidence=0.82,
                reason="继承会话 Topic 集合；多 Topic 时 primaryTopic 保持 UNKNOWN，不表示 anchor",
            )

        scores: Dict[QuestionCategory, float] = {}
        if keywords is not None:
            for category_value, score in keywords.topic_scores.items():
                try:
                    scores[QuestionCategory(category_value)] = float(score)
                except ValueError:
                    continue
        else:
            for category, words in BUSINESS_KEYWORDS.items():
                scores[category] = float(sum(1 for word in words if word.lower() in text.lower()))
        if route_slots:
            for candidate in route_slots.topic_candidates:
                try:
                    category = QuestionCategory(candidate.topic)
                except Exception:
                    continue
                scores[category] = max(scores.get(category, 0.0), float(candidate.score or 0.0))
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

    def _explicit_topics(self, scores: Dict[QuestionCategory, float]) -> List[QuestionCategory]:
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
