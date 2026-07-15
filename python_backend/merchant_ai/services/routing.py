from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
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
RANKING_PATTERNS = [
    re.compile(r"前\s*\d+"),
    re.compile(r"top\s*\d+", re.I),
    re.compile(r"最高|最低|最多|最少|排名|排行"),
]
TIME_PATTERNS = [
    re.compile(r"(最近|近|过去|前)?\s*\d{1,3}\s*[天日]"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(周|星期|礼拜)"),
    re.compile(r"(最近|近|过去|前)?\s*\d{1,2}\s*(个月|月)"),
]


def default_topic_assets() -> Any:
    try:
        from merchant_ai.config import get_settings
        from merchant_ai.services.assets import TopicAssetService

        return TopicAssetService(get_settings())
    except Exception:
        return None


class KeywordExtractService:
    def __init__(self, topic_assets: Any = None):
        self.topic_assets = topic_assets or default_topic_assets()
        self._semantic_metrics = self._build_semantic_metric_lexicon(self.topic_assets)
        self._semantic_matcher = PhraseMatcher(list(self._semantic_metrics))
        self._semantic_topics = self._build_semantic_topic_lexicon(self.topic_assets)
        self._semantic_topic_matcher = PhraseMatcher(list(self._semantic_topics))
        self._semantic_dimensions = self._build_semantic_dimension_lexicon(self.topic_assets)
        self._semantic_dimension_matcher = PhraseMatcher(list(self._semantic_dimensions))

    def reload_semantic_lexicon(self) -> None:
        self._semantic_metrics = self._build_semantic_metric_lexicon(self.topic_assets)
        self._semantic_matcher = PhraseMatcher(list(self._semantic_metrics))
        self._semantic_topics = self._build_semantic_topic_lexicon(self.topic_assets)
        self._semantic_topic_matcher = PhraseMatcher(list(self._semantic_topics))
        self._semantic_dimensions = self._build_semantic_dimension_lexicon(self.topic_assets)
        self._semantic_dimension_matcher = PhraseMatcher(list(self._semantic_dimensions))

    def business_surface_signal(self, question: str) -> Dict[str, Any]:
        """Detect whether the text touches governed business assets without resolving them."""

        normalized = normalize_keyword_text(question or "")
        hits = set()
        for matcher in [self._semantic_topic_matcher, self._semantic_matcher, self._semantic_dimension_matcher]:
            try:
                hits.update(str(item) for item in matcher.match(normalized))
            except Exception:
                continue
        return {
            "hasBusinessDomainPhrase": bool(hits),
            "businessSurfaceSignalCount": len(hits),
        }

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
        for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月"]:
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
            if (self._semantic_metrics or self._semantic_topics or self._semantic_dimensions)
            else []
        )
        keywords: List[str] = []
        keyword_actions = action if business else []
        keyword_ranking = ranking if business else []
        for item in business + time_words + keyword_actions + keyword_ranking:
            if item and item not in keywords:
                keywords.append(item)
        analysis_intent = classify_analysis_intent(normalized, action, ranking)
        confidence = keyword_confidence(topic_scores, metric_mentions, dimension_mentions, normalized, ambiguous_metrics)
        unresolved: List[str] = []
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
        if self._semantic_topics:
            return self._semantic_topic_mentions(normalized)
        return []

    def _semantic_topic_mentions(self, normalized: str) -> List[KeywordMention]:
        mentions: List[KeywordMention] = []
        for phrase in self._semantic_topic_matcher.match(normalized):
            for entry in self._semantic_topics.get(normalize_keyword_text(phrase), [])[:6]:
                mentions.append(
                    KeywordMention(
                        phrase=phrase,
                        canonical_key=str(entry.get("topic") or QuestionCategory.UNKNOWN.value),
                        display_name=phrase,
                        kind="topic",
                        topic=entry.get("category") or QuestionCategory.UNKNOWN,
                        score=float(entry.get("score") or 1.5),
                        source="semantic_topic",
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
        if self._semantic_dimensions:
            return self._semantic_dimension_mentions(normalized)
        return []

    def _semantic_dimension_mentions(self, normalized: str) -> List[KeywordMention]:
        mentions: List[KeywordMention] = []
        for phrase in self._semantic_dimension_matcher.match(normalized):
            for entry in self._semantic_dimensions.get(normalize_keyword_text(phrase), [])[:6]:
                topic = entry.get("category") or QuestionCategory.UNKNOWN
                mentions.append(
                    KeywordMention(
                        phrase=phrase,
                        canonical_key=str(entry.get("column") or ""),
                        display_name=phrase,
                        kind="dimension",
                        topic=topic,
                        score=float(entry.get("score") or 2.0),
                        source="semantic_column",
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
                    aliases = semantic_metric_alias_phrases(
                        [business_name, metric_key, str(metric.get("metricKey") or ""), *(metric.get("aliases") or [])]
                    )
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

    def _build_semantic_topic_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = TOPIC_TO_CATEGORY.get(topic_name, QuestionCategory.UNKNOWN)
            for phrase in semantic_alias_phrases([topic_name]):
                add_semantic_lexicon_entry(lexicon, phrase, {"topic": topic_name, "category": category, "score": 1.8})
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                labels = [
                    table,
                    str(manifest_item.get("tableComment") or ""),
                    str(manifest_item.get("dataGrain") or ""),
                ]
                for phrase in semantic_alias_phrases(labels):
                    add_semantic_lexicon_entry(lexicon, phrase, {"topic": topic_name, "category": category, "table": table, "score": 1.6})
                for metric in topic_assets.load_table_metrics(topic_name, table):
                    labels = [
                        str(metric.get("businessName") or ""),
                        str(metric.get("metricKey") or ""),
                        *[str(alias) for alias in metric.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        add_semantic_lexicon_entry(lexicon, phrase, {"topic": topic_name, "category": category, "table": table, "score": 1.4})
                for field in topic_assets.load_table_semantic_columns(topic_name, table):
                    labels = [
                        str(field.get("businessName") or ""),
                        str(field.get("columnName") or ""),
                        *[str(alias) for alias in field.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        add_semantic_lexicon_entry(lexicon, phrase, {"topic": topic_name, "category": category, "table": table, "score": 1.2})
                for term in topic_assets.load_table_terms(topic_name, table):
                    labels = [
                        str(term.get("term") or ""),
                        *[str(alias) for alias in term.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        add_semantic_lexicon_entry(lexicon, phrase, {"topic": topic_name, "category": category, "table": table, "score": 1.5})
        return dict(lexicon)

    def _build_semantic_dimension_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = TOPIC_TO_CATEGORY.get(topic_name, QuestionCategory.UNKNOWN)
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                for field in topic_assets.load_table_semantic_columns(topic_name, table):
                    column = str(field.get("columnName") or "")
                    if not column:
                        continue
                    role = str(field.get("role") or "").upper()
                    if role not in {"KEY", "DIMENSION", "TIME"}:
                        continue
                    labels = [
                        str(field.get("businessName") or ""),
                        column,
                        str(field.get("description") or ""),
                        *[str(alias) for alias in field.get("aliases") or []],
                    ]
                    score = 2.2
                    for phrase in semantic_alias_phrases(labels):
                        add_semantic_lexicon_entry(
                            lexicon,
                            phrase,
                            {"topic": topic_name, "category": category, "table": table, "column": column, "score": score},
                        )
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
    if not phrase:
        return False
    compact = phrase.replace("_", "").replace(" ", "")
    return len(compact) >= 2


def add_semantic_lexicon_entry(lexicon: Dict[str, List[Dict[str, Any]]], phrase: str, payload: Dict[str, Any]) -> None:
    key = normalize_keyword_text(phrase)
    if not valid_metric_alias(key):
        return
    identity = (
        str(payload.get("topic") or ""),
        str(payload.get("table") or ""),
        str(payload.get("column") or payload.get("metricKey") or ""),
    )
    if any(
        (
            str(item.get("topic") or ""),
            str(item.get("table") or ""),
            str(item.get("column") or item.get("metricKey") or ""),
        )
        == identity
        for item in lexicon[key]
    ):
        return
    lexicon[key].append(payload)


def semantic_alias_phrases(labels: List[str]) -> List[str]:
    phrases: List[str] = []
    for label in labels:
        text = normalize_keyword_text(str(label or ""))
        if not text:
            continue
        phrases.append(text)
    return dedupe_ordered(phrases)


def semantic_metric_alias_phrases(labels: List[str]) -> List[str]:
    phrases = semantic_alias_phrases(labels)
    generated: List[str] = []
    for phrase in phrases:
        compact = phrase.replace(" ", "")
        if len(compact) <= 2:
            continue
        generated.append(compact)
        if compact.endswith("元") and len(compact) > 3:
            generated.append(compact[:-1])
        for prefix in ["订单", "交易", "支付成功", "退款订单", "退款关联", "售后关联", "每日"]:
            if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
                generated.append(compact[len(prefix):])
        if "退款" in compact and "支付金额" in compact:
            generated.append("退款金额")
        if "支付" in compact and "金额" in compact:
            generated.append("支付金额")
    return dedupe_ordered([*phrases, *generated])


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
        if keywords and keywords.time_keywords and keywords.action_keywords:
            return False
        return True

    def _has_multiple_time_ranges(self, question: str) -> bool:
        return sum(1 for pattern in TIME_PATTERNS for _ in pattern.finditer(question)) >= 2

    def _has_any_time_range(self, question: str) -> bool:
        return self._has_multiple_time_ranges(question) or any(pattern.search(question) for pattern in TIME_PATTERNS) or any(
            word in question for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月"]
        )

    def _is_simple_detail_lookup(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> bool:
        if not any(word in question for word in ["明细", "详情", "列表", "记录", "单号", "流水"]):
            return False
        has_object_ref = any(pattern.search(question) for _ref_type, pattern in RouteSlotExtractor.OBJECT_PATTERNS)
        if (not self._has_any_time_range(question) and not has_object_ref) or self._has_multiple_time_ranges(question):
            return False
        if any(word in question for word in ACTION_KEYWORDS):
            return False
        if self._matched_domain_count(question, keywords) >= 2:
            return False
        if keywords and any(any(flag in action for flag in ["分析", "对比", "优化", "判断", "解释", "排查"]) for action in keywords.action_keywords):
            return False
        return not recall_bundle or not recall_bundle.items or all((item.answer_mode or "").upper() == "DETAIL" for item in recall_bundle.items)

    def _matched_domain_count(self, question: str, keywords: Optional[ExtractedKeywords] = None) -> int:
        if keywords and keywords.topic_scores:
            return len([score for score in keywords.topic_scores.values() if score > 0])
        return 0


class SemanticPreflightRouteClassifier:
    ROUTES = {"GREETING", "BUSINESS_CHAT", "BUSINESS_TASK", "INVALID", "CLARIFICATION_REPLY", "UNSUPPORTED_WRITE"}

    def __init__(self, settings: Any, llm: Any = None):
        self.settings = settings
        if llm is not None:
            self.llm = llm
        else:
            model = (
                str(getattr(settings, "preflight_semantic_route_model", "") or "")
                or str(getattr(settings, "llm_fast_model", "") or "")
                or str(getattr(settings, "openai_model", "") or "")
            )
            try:
                from merchant_ai.services.llm import LlmClient

                self.llm = LlmClient(
                    settings,
                    model_name=model,
                    api_key=str(getattr(settings, "preflight_llm_api_key", "") or ""),
                    base_url=str(getattr(settings, "preflight_llm_base_url", "") or ""),
                )
            except Exception:
                self.llm = None

    def classify(
        self,
        question: str,
        keywords: ExtractedKeywords,
        route_slots: "RouteSlots",
        pending_context: bool = False,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.settings, "preflight_semantic_route_enabled", False)):
            return {"enabled": False, "status": "disabled"}
        if not self.llm or not getattr(self.llm, "configured", False) or not hasattr(self.llm, "json_chat"):
            return {"enabled": True, "status": "unavailable"}
        system_prompt = (
            "你是商家经营助手的第一步轻量语义路由器。"
            "只判断用户当前输入的意图，不做业务分析、不查询数据、不输出建议。"
            "严格输出 JSON。route 只能是 GREETING、BUSINESS_CHAT、BUSINESS_TASK、INVALID、CLARIFICATION_REPLY。"
            "BUSINESS_CHAT 表示包含经营/商家表达但只是闲聊、情绪、泛泛讨论，没有明确查数或分析任务。"
            "BUSINESS_TASK 表示用户要求查询、排行、诊断、对比、明细、原因分析或经营建议。"
            "如果存在 pendingContext 且当前输入像补充条件、确认或选择，优先 CLARIFICATION_REPLY。"
        )
        payload = {
            "question": str(question or "")[:800],
            "pendingContext": bool(pending_context),
            "ruleSignals": {
                "businessKeywords": list(getattr(keywords, "business_keywords", []) or [])[:12],
                "metricKeywords": list(getattr(keywords, "metric_keywords", []) or [])[:12],
                "topicKeywords": list(getattr(keywords, "topic_keywords", []) or [])[:12],
                "dimensionKeywords": list(getattr(keywords, "dimension_keywords", []) or [])[:8],
                "timeKeywords": list(getattr(keywords, "time_keywords", []) or [])[:8],
                "actionKeywords": list(getattr(keywords, "action_keywords", []) or [])[:8],
                "rankingKeywords": list(getattr(keywords, "ranking_keywords", []) or [])[:8],
                "analysisIntent": getattr(keywords, "analysis_intent", ""),
            },
            "routeSlots": route_slots.model_dump(by_alias=True) if hasattr(route_slots, "model_dump") else {},
            "outputSchema": {
                "route": "GREETING|BUSINESS_CHAT|BUSINESS_TASK|INVALID|CLARIFICATION_REPLY",
                "confidence": "0.0-1.0",
                "reason": "short Chinese reason",
                "signals": {
                    "hasBusinessDomain": "boolean",
                    "hasMetric": "boolean",
                    "hasTimeWindow": "boolean",
                    "hasObject": "boolean",
                    "hasActionIntent": "boolean",
                    "isCasualOrEmotional": "boolean",
                },
                "missingSlots": ["metric", "timeWindow", "object", "analysisGoal"],
            },
        }
        try:
            result = self.llm.json_chat(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, default=str),
                fallback={},
                timeout_seconds=self._timeout_seconds(),
            )
        except Exception as exc:
            return {"enabled": True, "status": "failed", "error": str(exc)[:240]}
        if not isinstance(result, dict):
            return {"enabled": True, "status": "invalid_result"}
        provider_error = str(getattr(self.llm, "last_error", "") or "")
        if not result and provider_error:
            return {
                "enabled": True,
                "status": "failed",
                "error": provider_error[:240],
                "failureType": "provider_error",
            }
        route = str(result.get("route") or "").strip().upper()
        if route not in self.ROUTES:
            return {"enabled": True, "status": "invalid_route", "raw": result}
        try:
            confidence = float(result.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "enabled": True,
            "status": "success",
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(result.get("reason") or "")[:300],
            "signals": result.get("signals") if isinstance(result.get("signals"), dict) else {},
            "missingSlots": result.get("missingSlots") if isinstance(result.get("missingSlots"), list) else [],
        }

    def classify_surface(
        self,
        question: str,
        surface_signals: Dict[str, Any],
        pending_context: bool = False,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.settings, "preflight_semantic_route_enabled", False)):
            return {"enabled": False, "status": "disabled"}
        if not self.llm or not getattr(self.llm, "configured", False) or not hasattr(self.llm, "json_chat"):
            return {"enabled": True, "status": "unavailable"}
        system_prompt = (
            "你是商家经营助手的入口闸门，只判断是否进入后续业务链路。"
            "不要选择 Topic，不要解析指标口径，不要选择表，不要生成 SQL。"
            "严格输出 JSON。route 只能是 GREETING、BUSINESS_TASK、BUSINESS_CHAT、INVALID、CLARIFICATION_REPLY、UNSUPPORTED_WRITE。"
            "BUSINESS_TASK 表示值得进入后续 Topic/RAG/Planner 链路；它不要求指标、时间、口径已经完整。"
            "只要问题包含商家经营相关对象、业务域、指标样式、时间范围、明细查询、趋势/情况/表现等表达，就应优先输出 BUSINESS_TASK。"
            "不要因为用户表达简短、指标不完整、时间不完整、只说“情况/表现/怎么样”而判 INVALID；后续 Topic 和澄清节点会处理缺口。"
            "INVALID 只用于明显非商家经营场景，或完全没有业务对象/经营目标的问题。"
            "BUSINESS_CHAT 只用于助手能力说明、经营概念闲聊、无需查数或分析的问题；包含订单、退款、商品、工单、履约、赔付、优惠券等经营对象时不要输出 BUSINESS_CHAT。"
            "如果存在 pendingContext 且当前输入像补充条件、确认或选择，优先 CLARIFICATION_REPLY。"
            "如果用户要求删除、修改、更新、创建、写入、导入、重建等写操作，输出 UNSUPPORTED_WRITE。"
        )
        payload = {
            "question": str(question or "")[:800],
            "pendingContext": bool(pending_context),
            "surfaceSignals": surface_signals,
            "decisionHints": [
                "最近7天订单和退款情况 -> BUSINESS_TASK",
                "昨天客服工单怎么样 -> BUSINESS_TASK",
                "帮我看一下商品审核情况 -> BUSINESS_TASK",
                "今天天气怎么样 -> INVALID",
                "你能做什么 -> BUSINESS_CHAT",
            ],
            "outputSchema": {
                "route": "GREETING|BUSINESS_TASK|BUSINESS_CHAT|INVALID|CLARIFICATION_REPLY|UNSUPPORTED_WRITE",
                "confidence": "0.0-1.0",
                "intentKind": "chat|business_task|business_chat|clarification_reply|unsupported_write|invalid",
                "missingInfo": ["business_scope", "metric", "time_window", "object", "analysis_goal"],
                "clarificationQuestion": "short question when route is INVALID",
                "reason": "short Chinese reason",
            },
        }
        try:
            result = self.llm.json_chat(
                system_prompt,
                json.dumps(payload, ensure_ascii=False, default=str),
                fallback={},
                timeout_seconds=self._timeout_seconds(),
            )
        except Exception as exc:
            return {"enabled": True, "status": "failed", "error": str(exc)[:240]}
        if not isinstance(result, dict):
            return {"enabled": True, "status": "invalid_result"}
        provider_error = str(getattr(self.llm, "last_error", "") or "")
        if not result and provider_error:
            return {"enabled": True, "status": "failed", "error": provider_error[:240], "failureType": "provider_error"}
        route = str(result.get("route") or "").strip().upper()
        if route not in self.ROUTES:
            return {"enabled": True, "status": "invalid_route", "raw": result}
        try:
            confidence = float(result.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "enabled": True,
            "status": "success",
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "intentKind": str(result.get("intentKind") or result.get("intent_kind") or "").strip(),
            "missingInfo": result.get("missingInfo") if isinstance(result.get("missingInfo"), list) else [],
            "clarificationQuestion": str(result.get("clarificationQuestion") or result.get("clarification_question") or "")[:300],
            "reason": str(result.get("reason") or "")[:300],
        }

    def _timeout_seconds(self) -> int:
        configured = int(getattr(self.settings, "preflight_semantic_route_timeout_seconds", 3) or 3)
        max_timeout = int(getattr(self.settings, "preflight_semantic_route_max_timeout_seconds", 5) or 5)
        return max(3, min(configured, max_timeout))


@dataclass
class PreflightUnderstanding:
    keywords: ExtractedKeywords
    route_slots: RouteSlots
    rule_route: RoutingDecision
    semantic_trace: Dict[str, Any]
    routing_decision: RoutingDecision
    trace: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    surface_signals: Dict[str, Any] = dataclass_field(default_factory=dict)
    clarification_question: str = ""


class PreflightUnderstandingService:
    """Small-model-first entry gate. It does not resolve Topic, metric, or table."""

    def __init__(
        self,
        settings: Any,
        keyword_service: KeywordExtractService,
        routing_service: QuestionRoutingService,
        slot_extractor: "RouteSlotExtractor",
        semantic_classifier: SemanticPreflightRouteClassifier,
    ):
        self.settings = settings
        self.keyword_service = keyword_service
        self.routing_service = routing_service
        self.slot_extractor = slot_extractor
        self.semantic_classifier = semantic_classifier

    def understand(self, question: str, pending_context: bool = False) -> PreflightUnderstanding:
        surface_signals = self.surface_signals(question)
        surface_signals["pendingContext"] = bool(pending_context)
        rule_route = self.hard_gate_route(question, surface_signals, pending_context)
        semantic_trace = self.semantic_preflight_trace(question, rule_route, surface_signals, pending_context)
        routing_decision = self.merge_gate_routes(rule_route, semantic_trace, surface_signals, pending_context)
        route_slots = RouteSlots(
            operation="write_requested" if surface_signals.get("writeOperation") else "read",
            risk_level="high_risk" if surface_signals.get("writeOperation") else "normal",
            route_confidence=float(surface_signals.get("confidence") or 0.0),
            route_warnings=["PREFLIGHT_SURFACE_ONLY"] + (["WRITE_OPERATION_REQUESTED"] if surface_signals.get("writeOperation") else []),
        )
        keywords = ExtractedKeywords()
        trace = [
            {
                "stage": "preflight_surface_gate",
                "surfaceSignals": surface_signals,
                "ruleRoute": enum_route(rule_route.route),
            },
            {
                "stage": "semantic_preflight_route",
                "ruleRoute": enum_route(rule_route.route),
                "finalRoute": enum_route(routing_decision.route),
                "semantic": semantic_trace,
            },
        ]
        return PreflightUnderstanding(
            keywords=keywords,
            route_slots=route_slots,
            rule_route=rule_route,
            semantic_trace=semantic_trace,
            routing_decision=routing_decision,
            trace=trace,
            surface_signals=surface_signals,
            clarification_question=str(semantic_trace.get("clarificationQuestion") or ""),
        )

    def semantic_preflight_trace(
        self,
        question: str,
        rule_route: RoutingDecision,
        surface_signals: Dict[str, Any],
        pending_context: bool = False,
    ) -> Dict[str, Any]:
        if rule_route.reason in {"空问题", "检测到写操作请求，当前只支持只读查询和分析", "寒暄问题"}:
            return {
                "enabled": bool(getattr(self.settings, "preflight_semantic_route_enabled", False)),
                "status": "skipped_rule_terminal",
                "reason": rule_route.reason,
            }
        if self.surface_business_task_sufficient(surface_signals, pending_context):
            return {
                "enabled": bool(getattr(self.settings, "preflight_semantic_route_enabled", False)),
                "status": "skipped_surface_business",
                "route": "BUSINESS_TASK",
                "confidence": float(surface_signals.get("confidence") or 0.0),
                "reason": "surface signals are sufficient for the Topic/RAG chain",
            }
        return self.semantic_classifier.classify_surface(
            question,
            surface_signals,
            pending_context=pending_context,
        )

    def surface_business_task_sufficient(self, signals: Dict[str, Any], pending_context: bool = False) -> bool:
        if pending_context:
            return True
        if signals.get("empty") or signals.get("greeting") or signals.get("assistantChat") or signals.get("writeOperation"):
            return False
        if signals.get("hasObjectRef") or signals.get("hasBusinessMetricLikePhrase"):
            return True
        if signals.get("hasMetricLikePhrase") and (
            signals.get("hasBusinessDomainPhrase")
            or signals.get("hasTimeExpression")
            or signals.get("hasAnalysisIntent")
        ):
            return True
        if signals.get("hasBusinessDomainPhrase") and signals.get("hasAnalysisIntent"):
            return True
        return int(signals.get("businessSurfaceSignalCount") or 0) >= 2

    def surface_signals(self, question: str) -> Dict[str, Any]:
        text = str(question or "").strip()
        lowered = text.lower()
        business_surface = self.keyword_service.business_surface_signal(text)
        has_time = bool(any(pattern.search(text) for pattern in TIME_PATTERNS)) or any(
            word in text for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月"]
        )
        has_object_ref = bool(any(pattern.search(text) for _ref_type, pattern in RouteSlotExtractor.OBJECT_PATTERNS))
        raw_analysis_intent = bool(any(term in text for term in ACTION_KEYWORDS))
        write_operation = bool(any(term.lower() in lowered for term in RouteSlotExtractor.WRITE_TERMS))
        greeting = bool(re.match(r"^(你好|您好|hi|hello|hey|在吗|嗨|哈喽|早上好|下午好|晚上好)[!！。,.，\s]*$", lowered, re.I))
        assistant_chat_phrase = bool(
            any(term in text for term in ["你是谁", "你能做什么", "你可以做什么", "你会什么", "怎么用", "如何使用"])
        )
        business_metric_like = bool(
            re.search(
                r"(gmv|销售额|成交额|支付金额|客单价|订单量|单量|下单量|退款率|退款金额|退货率|售后率|"
                r"赔付率|赔付金额|工单率|优惠券|转化率|履约率|发货率|审核通过率|商品数|库存)",
                lowered,
                re.I,
            )
        )
        generic_metric_like = bool(re.search(r"(金额|数量|率|趋势|排行|top|最高|最低|最多|最少|多少)", lowered, re.I))
        metric_like = bool(business_metric_like or (generic_metric_like and business_surface.get("hasBusinessDomainPhrase")))
        assistant_chat = bool(
            assistant_chat_phrase
            and not (
                business_surface.get("hasBusinessDomainPhrase")
                or has_time
                or has_object_ref
                or metric_like
            )
        )
        confidence = 0.2
        has_analysis_intent = bool(
            raw_analysis_intent
            and (
                business_surface.get("hasBusinessDomainPhrase")
                or business_metric_like
                or has_object_ref
            )
        )
        if greeting or assistant_chat or write_operation:
            confidence = 0.98
        elif has_time or has_object_ref or has_analysis_intent or metric_like or business_surface.get("hasBusinessDomainPhrase"):
            confidence = 0.72
        return {
            "empty": not bool(text),
            "greeting": greeting,
            "assistantChat": assistant_chat,
            "writeOperation": write_operation,
            "hasTimeExpression": has_time,
            "hasObjectRef": has_object_ref,
            "hasMetricLikePhrase": metric_like,
            "hasBusinessMetricLikePhrase": business_metric_like,
            "hasGenericMetricLikePhrase": generic_metric_like,
            "hasAnalysisIntent": has_analysis_intent,
            **business_surface,
            "confidence": confidence,
        }

    def hard_gate_route(self, question: str, signals: Dict[str, Any], pending_context: bool) -> RoutingDecision:
        if signals.get("empty"):
            return RoutingDecision(route=QuestionRoute.INVALID, reason="空问题")
        if signals.get("writeOperation"):
            return RoutingDecision(route=QuestionRoute.INVALID, reason="检测到写操作请求，当前只支持只读查询和分析")
        if signals.get("greeting") or signals.get("assistantChat"):
            return RoutingDecision(route=QuestionRoute.GREETING, reason="寒暄问题")
        if pending_context:
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=False, reason="上一轮存在澄清/确认上下文，进入完整上下文承接")
        return RoutingDecision(route=QuestionRoute.INVALID, reason="等待小模型入口判断")

    def merge_gate_routes(
        self,
        rule_route: RoutingDecision,
        semantic_trace: Dict[str, Any],
        signals: Dict[str, Any],
        pending_context: bool,
    ) -> RoutingDecision:
        if rule_route.reason in {"空问题", "检测到写操作请求，当前只支持只读查询和分析", "寒暄问题"}:
            return rule_route
        has_business_surface = bool(signals.get("hasObjectRef") or signals.get("hasMetricLikePhrase") or signals.get("hasAnalysisIntent"))
        has_business_task_surface = bool(
            has_business_surface
            or (
                signals.get("hasBusinessDomainPhrase")
                and (signals.get("hasTimeExpression") or signals.get("hasObjectRef") or signals.get("hasMetricLikePhrase") or signals.get("hasAnalysisIntent"))
            )
        )
        if pending_context and semantic_trace.get("status") != "success":
            return rule_route
        if not semantic_trace or semantic_trace.get("status") != "success":
            if has_business_task_surface:
                return RoutingDecision(route=QuestionRoute.BUSINESS, complex=bool(signals.get("hasAnalysisIntent")), reason="入口 surface signal 足够，进入后续 Topic/RAG 链路")
            return RoutingDecision(route=QuestionRoute.INVALID, reason="入口信息不足，需要补充业务范围或查询目标")
        semantic_route = str(semantic_trace.get("route") or "")
        confidence = float(semantic_trace.get("confidence") or 0)
        min_conf = float(getattr(self.settings, "preflight_semantic_route_min_confidence", 0.62) or 0.62)
        if pending_context and semantic_route == "CLARIFICATION_REPLY" and confidence >= min_conf:
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=False, reason="语义路由：上一轮澄清/确认承接回复")
        if confidence < min_conf:
            if has_business_task_surface:
                return RoutingDecision(route=QuestionRoute.BUSINESS, complex=bool(signals.get("hasAnalysisIntent")), reason="入口 surface signal 覆盖低置信小模型，进入后续 Topic/RAG 链路")
            return RoutingDecision(route=QuestionRoute.INVALID, reason="小模型入口置信度不足，需要补充业务范围或查询目标")
        if semantic_route == "GREETING":
            return RoutingDecision(route=QuestionRoute.GREETING, complex=False, reason="入口判断：轻量对话，不触发查数")
        if semantic_route == "BUSINESS_CHAT":
            if has_business_task_surface:
                return RoutingDecision(route=QuestionRoute.BUSINESS, complex=bool(signals.get("hasAnalysisIntent")), reason="入口 surface signal 覆盖小模型闲聊判断，进入后续 Topic/RAG 链路")
            if not has_business_surface and not signals.get("greeting") and not signals.get("assistantChat"):
                return RoutingDecision(route=QuestionRoute.INVALID, complex=False, reason=str(semantic_trace.get("reason") or "入口判断：需要补充业务范围或查询目标"))
            return RoutingDecision(route=QuestionRoute.GREETING, complex=False, reason="入口判断：轻量业务对话，不触发查数")
        if semantic_route == "UNSUPPORTED_WRITE":
            return RoutingDecision(route=QuestionRoute.INVALID, complex=False, reason="入口判断：当前只支持只读查询和分析")
        if semantic_route in {"BUSINESS_TASK", "CLARIFICATION_REPLY"}:
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=bool(signals.get("hasAnalysisIntent")), reason="入口判断：业务任务，进入后续 Topic/RAG 链路")
        if semantic_route == "INVALID" and has_business_task_surface:
            return RoutingDecision(route=QuestionRoute.BUSINESS, complex=bool(signals.get("hasAnalysisIntent")), reason="入口 surface signal 覆盖小模型 INVALID，进入后续 Topic/RAG 链路")
        return RoutingDecision(route=QuestionRoute.INVALID, complex=False, reason=str(semantic_trace.get("reason") or "入口判断：需要补充业务范围或查询目标"))

def enum_route(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


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
        for word in ["昨天", "昨日", "今天", "今日", "上周", "本周", "这周", "上个月", "本月"]:
            if word in text:
                return word
        return ""

    def _analysis_signals(self, keywords: ExtractedKeywords) -> List[str]:
        analysis_actions = {
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
            "最高",
            "最低",
            "最多",
            "最少",
        }
        if keywords and any(action in analysis_actions for action in keywords.action_keywords):
            return ["weak_analysis_hint"]
        return []

    def _topic_candidates(
        self,
        text: str,
        object_refs: List[RouteObjectRef],
        keywords: Optional[ExtractedKeywords] = None,
    ) -> List[RouteTopicCandidate]:
        by_topic: Dict[QuestionCategory, Dict[str, object]] = {}
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
        if self._risk_level(text, "read") == "rule_sensitive":
            payload = by_topic.setdefault(QuestionCategory.PLATFORM_RULE, {"score": 0, "evidence": []})
            payload["score"] = max(float(payload.get("score") or 0), 2.0)
            evidence = list(payload.get("evidence") or [])
            if "rule_sensitive" not in evidence:
                evidence.append("rule_sensitive")
            payload["evidence"] = evidence[:8]
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
                clarification_required=True,
                reason="未识别出显式业务 topic；默认先确认分析范围，开放诊断问题由 OpenDiagnosticPolicy 接管",
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
