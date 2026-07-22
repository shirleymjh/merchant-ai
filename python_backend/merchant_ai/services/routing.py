from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set

from merchant_ai.models import (
    ExtractedKeywords,
    KeywordMention,
    QuestionCategory,
    QuestionRoute,
    RecallBundle,
    RouteObjectRef,
    RouteLexicalSpan,
    RouteSpanType,
    RouteSlots,
    RouteTimeWindow,
    RouteTopicCandidate,
    RoutingDecision,
    TopicRoutingDecision,
)
from merchant_ai.services.grounded_runtime_budget import GroundedRuntimeBudgetExceeded
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.text_parsing import (
    collapse_whitespace,
    contains_any_literal,
    is_ascii_identifier,
    is_ascii_word_phrase,
    iter_ascii_digit_spans,
    literal_spans,
    parse_prefixed_reference,
    split_on_characters,
)
from merchant_ai.services.time_semantics import extract_temporal_lexical_spans, resolve_time_range


_ROUTING_LANGUAGE = load_language_policy().routing
ACTION_KEYWORDS = list(_ROUTING_LANGUAGE.action_markers)


def route_span_overlaps(left: RouteLexicalSpan, right: RouteLexicalSpan) -> bool:
    return int(left.start) < int(right.end) and int(right.start) < int(left.end)


def extract_ranking_spans(text: str) -> List[RouteLexicalSpan]:
    value = str(text or "")
    candidates: List[RouteLexicalSpan] = []
    for source, prefixes, case_sensitive in (
        ("ordinal_prefix", _ROUTING_LANGUAGE.ranking_ordinal_prefixes, True),
        ("top_n", _ROUTING_LANGUAGE.ranking_top_prefixes, False),
    ):
        for prefix in prefixes:
            for start, end in _prefixed_number_spans(value, prefix, case_sensitive=case_sensitive):
                candidates.append(
                    RouteLexicalSpan(
                        span_type=RouteSpanType.RANKING,
                        start=start,
                        end=end,
                        text=value[start:end].strip(),
                        source=source,
                    )
                )
    for start, end, operator in literal_spans(value, _ROUTING_LANGUAGE.ranking_operators):
        candidates.append(
            RouteLexicalSpan(
                span_type=RouteSpanType.RANKING,
                start=start,
                end=end,
                text=operator.strip(),
                source="ranking_operator",
            )
        )
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.source))
    accepted: List[RouteLexicalSpan] = []
    for candidate in candidates:
        if any(route_span_overlaps(candidate, current) for current in accepted):
            continue
        accepted.append(candidate)
    return accepted


def _prefixed_number_spans(value: str, prefix: str, *, case_sensitive: bool) -> List[tuple[int, int]]:
    haystack = value if case_sensitive else value.casefold()
    needle = prefix if case_sensitive else prefix.casefold()
    spans: List[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        number_start = start + len(prefix)
        while number_start < len(value) and value[number_start].isspace():
            number_start += 1
        number_end = number_start
        while number_end < len(value) and value[number_end].isascii() and value[number_end].isdigit():
            number_end += 1
        if number_end > number_start:
            spans.append((start, number_end))
            cursor = number_end
        else:
            cursor = start + max(1, len(prefix))
    return spans


def extract_route_lexical_spans(text: str) -> List[RouteLexicalSpan]:
    """Resolve temporal/ranking ambiguity through typed span precedence.

    Temporal syntax owns an overlapping interval before ranking syntax is
    considered. A numeric prefix with a time unit is therefore temporal,
    while the same prefix before a non-time entity remains Top-N syntax.
    """

    temporal = extract_temporal_lexical_spans(text)
    ranking_candidates = extract_ranking_spans(text)
    ranking = [
        candidate
        for candidate in ranking_candidates
        if not any(route_span_overlaps(candidate, occupied) for occupied in temporal)
    ]
    return sorted([*temporal, *ranking], key=lambda item: (item.start, item.end, item.span_type))


def planning_hints_from_extracted_keywords(
    question: str,
    keywords: ExtractedKeywords | None,
) -> Dict[str, Any]:
    """Project routing evidence into the typed obligations used by planning.

    The projection deliberately consumes governed keyword mentions and typed
    lexical spans.  Asset compaction can therefore reason about a requested
    dimension or Top-N contract without re-interpreting business words from the
    raw question.
    """

    keywords = keywords or ExtractedKeywords()
    normalized = str(keywords.normalized_question or normalize_keyword_text(question or ""))
    metric_mentions = [item for item in keywords.mentions if item.kind == "metric" and item.phrase]
    dimension_mentions = [item for item in keywords.mentions if item.kind == "dimension" and item.phrase]
    metric_phrases = dedupe_ordered(
        [*[item.phrase for item in metric_mentions], *list(keywords.metric_keywords or [])]
    )
    dimensions: List[Dict[str, Any]] = []
    dimension_seen: Set[tuple[str, str, str, str]] = set()
    for mention in dimension_mentions:
        identity = (
            str(mention.phrase or ""),
            str(mention.canonical_key or ""),
            str(getattr(mention.topic, "value", mention.topic) or ""),
            str(mention.owner_table or ""),
        )
        if not identity[0] or not identity[1] or identity in dimension_seen:
            continue
        dimension_seen.add(identity)
        dimensions.append(
            {
                "phrase": identity[0],
                "column": identity[1],
                "topic": identity[2],
                "ownerTable": identity[3],
                "role": str(mention.semantic_role or ""),
                "source": str(mention.source or ""),
            }
        )

    ranking_spans = [
        span
        for span in keywords.lexical_spans
        if span.span_type == RouteSpanType.RANKING
    ]
    ranking: Dict[str, Any] = {}
    if ranking_spans:
        limit = 0
        order = ""
        operator = ""
        for span in ranking_spans:
            if span.source in {"ordinal_prefix", "top_n"} and not limit:
                digit_span = next(iter_ascii_digit_spans(span.text), None)
                limit = int(digit_span[2]) if digit_span else 0
            if span.source == "ranking_operator":
                operator = span.text
                order = str(_ROUTING_LANGUAGE.ranking_operators.get(span.text) or "")
        if limit and not order:
            order = _ROUTING_LANGUAGE.ranking_default_order
        anchor_phrase = nearest_metric_phrase_to_ranking(normalized, metric_phrases, ranking_spans)
        anchor_candidates = dedupe_ordered(
            [item.canonical_key for item in metric_mentions if item.phrase == anchor_phrase and item.canonical_key]
        )
        ranking = {
            "requested": True,
            "limit": limit,
            "order": order,
            "operator": operator,
            "anchorMetricPhrase": anchor_phrase,
            "anchorMetricCandidates": anchor_candidates,
            "spans": [span.model_dump(by_alias=True) for span in ranking_spans],
        }
        ranking = {key: value for key, value in ranking.items() if value not in (None, "", [], {})}

    return {
        key: value
        for key, value in {
            "metricPhrases": metric_phrases,
            "dimensionKeywords": dedupe_ordered(list(keywords.dimension_keywords or [])),
            "dimensions": dimensions,
            "ranking": ranking,
            "analysisIntent": str(keywords.analysis_intent or ""),
        }.items()
        if value not in (None, "", [], {})
    }


def nearest_metric_phrase_to_ranking(
    normalized_question: str,
    metric_phrases: List[str],
    ranking_spans: List[RouteLexicalSpan],
) -> str:
    """Bind a ranking clause to the closest governed metric mention."""

    if not metric_phrases or not ranking_spans:
        return ""
    rank_start = min(int(span.start) for span in ranking_spans)
    candidates: List[tuple[int, int, str]] = []
    for phrase in metric_phrases:
        normalized_phrase = normalize_keyword_text(phrase)
        if not normalized_phrase:
            continue
        start = normalized_question.find(normalized_phrase)
        while start >= 0:
            end = start + len(normalized_phrase)
            distance = rank_start - end if end <= rank_start else start - rank_start
            candidates.append((0 if end <= rank_start else 1, abs(distance), phrase))
            start = normalized_question.find(normalized_phrase, start + 1)
    if not candidates:
        return metric_phrases[0]
    return min(candidates, key=lambda item: (item[0], item[1], metric_phrases.index(item[2])))[2]


def default_topic_assets() -> Any:
    try:
        from merchant_ai.config import get_settings
        from merchant_ai.services.assets import TopicAssetService

        return TopicAssetService(get_settings())
    except Exception:
        return None


def load_asset_topic_contract(topic_assets: Any, topic: str) -> Dict[str, Any]:
    loader = getattr(topic_assets, "load_topic_contract", None)
    if callable(loader):
        try:
            contract = loader(topic)
            return contract if isinstance(contract, dict) else {}
        except Exception:
            return {}
    return {}


def resolve_asset_topic_category(topic_assets: Any, value: Any) -> QuestionCategory:
    resolver = getattr(topic_assets, "resolve_topic_category", None)
    if callable(resolver):
        try:
            return QuestionCategory(resolver(value))
        except Exception:
            pass
    raw = str(getattr(value, "value", value) or "").strip()
    return QuestionCategory(raw or QuestionCategory.UNKNOWN)


def asset_entry_topic_categories(topic_assets: Any, owner_topic: str, entry: Dict[str, Any]) -> List[QuestionCategory]:
    """Resolve owner and linked topics declared by one semantic asset entry."""

    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    linked_topics = entry.get("linkedTopics") or []
    metadata_linked_topics = metadata.get("linkedTopics") or []
    if not isinstance(linked_topics, list):
        linked_topics = [linked_topics]
    if not isinstance(metadata_linked_topics, list):
        metadata_linked_topics = [metadata_linked_topics]
    raw_values: List[Any] = [
        entry.get("ownerTopic"),
        entry.get("topic"),
        owner_topic,
        *linked_topics,
        *metadata_linked_topics,
    ]
    result: List[QuestionCategory] = []
    for value in raw_values:
        if not str(value or "").strip():
            continue
        category = resolve_asset_topic_category(topic_assets, value)
        if category != QuestionCategory.UNKNOWN and category not in result:
            result.append(category)
    return result


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
        topic_hits = set(self._semantic_topic_matcher.match(normalized))
        metric_hits = set(self._semantic_matcher.match(normalized))
        dimension_hits = set(self._semantic_dimension_matcher.match(normalized))
        hits = topic_hits | metric_hits | dimension_hits
        return {
            "hasBusinessDomainPhrase": bool(hits),
            "businessSurfaceSignalCount": len(hits),
            "hasPublishedMetricPhrase": bool(metric_hits),
            "hasPublishedTopicPhrase": bool(topic_hits),
            "hasPublishedDimensionPhrase": bool(dimension_hits),
        }

    def extract(self, question: str) -> ExtractedKeywords:
        text = question or ""
        normalized = normalize_keyword_text(text)
        dimension_mentions = self._dimension_mentions(normalized)
        metric_mentions = self._metric_mentions(normalized)
        dimension_mentions = independent_dimension_mentions(normalized, dimension_mentions, metric_mentions)
        lineage_mentions = self._metric_lineage_mentions(normalized, dimension_mentions)
        topic_mentions = self._topic_mentions(normalized)
        governed_phrases = {
            (item.kind, normalize_keyword_text(item.phrase))
            for item in [*metric_mentions, *dimension_mentions]
            if item.phrase
        }
        topic_mentions = [
            item
            for item in topic_mentions
            if not (
                item.source == "semantic_topic_metric"
                and ("metric", normalize_keyword_text(item.phrase)) in governed_phrases
            )
            and not (
                item.source == "semantic_topic_dimension"
                and ("dimension", normalize_keyword_text(item.phrase)) in governed_phrases
            )
        ]
        action = longest_distinct_matches(normalized, ACTION_KEYWORDS)
        lexical_spans = extract_route_lexical_spans(normalized)
        time_words = dedupe_ordered(
            [span.text for span in lexical_spans if span.span_type == RouteSpanType.TEMPORAL]
        )
        ranking = dedupe_ordered(
            [span.text for span in lexical_spans if span.span_type == RouteSpanType.RANKING]
        )
        ambiguous_metrics = ambiguous_metric_phrases(metric_mentions)
        all_mentions = dedupe_mentions([*metric_mentions, *dimension_mentions, *lineage_mentions, *topic_mentions])
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
        analysis_intent = classify_analysis_intent(action, lexical_spans)
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
            lexical_spans=lexical_spans,
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
                        source="semantic_topic_%s" % str(entry.get("source") or "contract"),
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
                topic_candidates = entry.get("topicCandidates") or [entry.get("topic")]
                for topic in topic_candidates:
                    mentions.append(
                        KeywordMention(
                            phrase=phrase,
                            canonical_key=str(entry.get("metricKey") or ""),
                            display_name=str(entry.get("businessName") or phrase),
                            kind="metric",
                            topic=topic or QuestionCategory.UNKNOWN,
                            owner_table=str(entry.get("table") or ""),
                            semantic_role="METRIC",
                            score=float(entry.get("score") or 3.0),
                            source="semantic_metric",
                        )
                    )
        return mentions

    def _metric_lineage_mentions(
        self,
        normalized: str,
        dimension_mentions: List[KeywordMention],
    ) -> List[KeywordMention]:
        """Resolve a dimensional metric to its governed detail serving path.

        A profile aggregate may own the fast summary metric while a detail fact
        table owns the requested grouping dimension.  That lineage should
        narrow discovery to the table that can answer the composed question,
        not blindly open the profile, dimension-master, and detail Topics.
        """

        causal_detail_requested = contains_any_literal(normalized, _ROUTING_LANGUAGE.causal_markers)
        if not dimension_mentions and not causal_detail_requested:
            return []
        mentions: List[KeywordMention] = []
        required_dimensions = metric_detail_grouping_dimensions(normalized, dimension_mentions)
        for phrase in self._semantic_matcher.match(normalized):
            for entry in self._semantic_metrics.get(normalize_keyword_text(phrase), [])[:4]:
                target, source = self._dimension_compatible_metric_entry(
                    entry,
                    required_dimensions,
                    allow_without_dimensions=causal_detail_requested,
                )
                if source != "semantic_metric_detail_ref":
                    continue
                for topic in target.get("topicCandidates") or [target.get("topic")]:
                    mentions.append(
                        KeywordMention(
                            phrase=phrase,
                            canonical_key=str(target.get("metricKey") or ""),
                            display_name=str(target.get("businessName") or phrase),
                            kind="lineage",
                            topic=topic or QuestionCategory.UNKNOWN,
                            owner_table=str(target.get("table") or ""),
                            semantic_role="METRIC_LINEAGE",
                            score=float(target.get("score") or 3.0),
                            source=source,
                        )
                    )
        return mentions

    def _dimension_compatible_metric_entry(
        self,
        entry: Dict[str, Any],
        dimension_mentions: List[KeywordMention],
        allow_without_dimensions: bool = False,
    ) -> tuple[Dict[str, Any], str]:
        """Follow an asset-declared detail metric only when it covers requested dimensions.

        Routing uses the published metric lineage and canonical semantic column
        identities.  It does not infer a detail table from business words or a
        physical table name.
        """

        if self.topic_assets is None or (not dimension_mentions and not allow_without_dimensions):
            return entry, "semantic_metric"
        detail_ref = str(entry.get("detailMetricRef") or entry.get("drilldownMetricRef") or "").strip()
        identity = semantic_metric_reference_identity(detail_ref)
        if not identity:
            return entry, "semantic_metric"
        target_topic, target_table, target_metric_key = identity
        requested_by_phrase: Dict[str, Set[str]] = defaultdict(set)
        for mention in dimension_mentions:
            phrase = normalize_keyword_text(mention.phrase)
            canonical_key = str(mention.canonical_key or "").strip()
            if phrase and canonical_key:
                requested_by_phrase[phrase].add(canonical_key)
        if not requested_by_phrase and not allow_without_dimensions:
            return entry, "semantic_metric"
        try:
            target_columns = {
                str(item.get("columnName") or "").strip()
                for item in self.topic_assets.load_table_semantic_columns(target_topic, target_table)
                if isinstance(item, dict) and str(item.get("columnName") or "").strip()
            }
            target_metric = next(
                (
                    item
                    for item in self.topic_assets.load_table_metrics(target_topic, target_table)
                    if str(item.get("canonicalMetricKey") or item.get("metricKey") or "").strip()
                    == target_metric_key
                ),
                None,
            )
        except Exception:
            return entry, "semantic_metric"
        if not target_metric or (
            requested_by_phrase
            and not all(keys.intersection(target_columns) for keys in requested_by_phrase.values())
        ):
            return entry, "semantic_metric"
        target_category = resolve_asset_topic_category(self.topic_assets, target_topic)
        target_topics = asset_entry_topic_categories(self.topic_assets, target_topic, target_metric)
        return (
            {
                **entry,
                "metricKey": target_metric_key,
                "businessName": str(target_metric.get("businessName") or target_metric_key),
                "topic": target_topics[0] if target_topics else target_category,
                "topicCandidates": target_topics or [target_category],
                "table": target_table,
            },
            "semantic_metric_detail_ref",
        )

    def _dimension_mentions(self, normalized: str) -> List[KeywordMention]:
        if self._semantic_dimensions:
            return self._semantic_dimension_mentions(normalized)
        return []

    def _semantic_dimension_mentions(self, normalized: str) -> List[KeywordMention]:
        mentions: List[KeywordMention] = []
        for phrase in self._semantic_dimension_matcher.match(normalized):
            for entry in self._semantic_dimensions.get(normalize_keyword_text(phrase), [])[:6]:
                topic_candidates = entry.get("topicCandidates") or [entry.get("category")]
                for topic in topic_candidates:
                    mentions.append(
                        KeywordMention(
                            phrase=phrase,
                            canonical_key=str(entry.get("column") or ""),
                            display_name=phrase,
                            kind="dimension",
                            topic=topic or QuestionCategory.UNKNOWN,
                            owner_table=str(entry.get("table") or ""),
                            semantic_role=str(entry.get("role") or ""),
                            score=float(entry.get("score") or 2.0),
                            source="semantic_column",
                        )
                    )
        return mentions

    def _topic_scores(self, mentions: List[KeywordMention]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        seen: Set[tuple[str, str, str]] = set()
        lineage_mentions = [item for item in mentions if item.kind == "lineage"]
        lineage_phrases = {normalize_keyword_text(item.phrase) for item in lineage_mentions if item.phrase}
        lineage_target_columns: Set[str] = set()
        if self.topic_assets is not None:
            for lineage in lineage_mentions:
                topic_names = self.topic_assets.topic_names_for_categories([lineage.topic])
                for topic_name in topic_names:
                    try:
                        lineage_target_columns.update(
                            str(item.get("columnName") or "").strip()
                            for item in self.topic_assets.load_table_semantic_columns(
                                topic_name,
                                lineage.owner_table,
                            )
                            if isinstance(item, dict) and str(item.get("columnName") or "").strip()
                        )
                    except Exception:
                        continue
        for mention in mentions:
            if mention.topic == QuestionCategory.UNKNOWN:
                continue
            mention_topic = keyword_topic_value(mention.topic)
            identity = (mention.kind, mention.canonical_key, mention_topic)
            if identity in seen:
                continue
            seen.add(identity)
            weight = 1.0
            # detailMetricRef is a strong discovery/ranking signal, not a
            # deterministic table-selection rule.  Keep the aggregate owner
            # and dimension master as lower-ranked L0 candidates so the Core
            # Agent can compare their business summaries and choose the table.
            if mention.kind == "lineage":
                weight = 2.0
            elif mention.kind == "metric" and normalize_keyword_text(mention.phrase) in lineage_phrases:
                weight = 0.4
            elif mention.kind == "dimension" and mention.canonical_key in lineage_target_columns:
                weight = 0.4
            scores[mention_topic] += mention.score * weight
        return dict(scores)

    def _build_semantic_metric_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = resolve_asset_topic_category(topic_assets, topic_name)
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                if not table:
                    continue
                for metric in topic_assets.load_table_metrics(topic_name, table):
                    topic_candidates = asset_entry_topic_categories(topic_assets, topic_name, metric)
                    metric_key = str(metric.get("canonicalMetricKey") or metric.get("metricKey") or "")
                    business_name = str(metric.get("businessName") or metric_key)
                    aliases = semantic_metric_alias_phrases(
                        [
                            business_name,
                            metric_key,
                            str(metric.get("metricKey") or ""),
                            str(metric.get("displayName") or ""),
                            str(metric.get("disambiguationName") or ""),
                            str(metric.get("naturalName") or ""),
                            str(metric.get("naturalAlias") or ""),
                            str(metric.get("originalBusinessName") or ""),
                            *(metric.get("aliases") or []),
                        ]
                    )
                    for alias in aliases:
                        phrase = normalize_keyword_text(str(alias or ""))
                        if not valid_metric_alias(phrase):
                            continue
                        payload = {
                            "metricKey": metric_key,
                            "businessName": business_name,
                            "topic": topic_candidates[0] if topic_candidates else category,
                            "topicCandidates": topic_candidates or [category],
                            "table": table,
                            "score": 3.0 if phrase == normalize_keyword_text(business_name) else 2.8,
                            "detailMetricRef": str(
                                metric.get("detailMetricRef")
                                or metric.get("drilldownMetricRef")
                                or metric.get("detail_metric_ref")
                                or ""
                            ),
                        }
                        if not any(
                            item.get("metricKey") == metric_key
                            and item.get("topicCandidates") == (topic_candidates or [category])
                            for item in lexicon[phrase]
                        ):
                            lexicon[phrase].append(payload)
        return dict(lexicon)

    def _build_semantic_topic_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = resolve_asset_topic_category(topic_assets, topic_name)
            contract = load_asset_topic_contract(topic_assets, topic_name)
            topic_phrases = [
                topic_name,
                str(contract.get("displayName") or ""),
                *[str(item) for item in contract.get("aliases") or []],
            ]
            for phrase in semantic_alias_phrases(topic_phrases):
                add_semantic_lexicon_entry(
                    lexicon,
                    phrase,
                    {"topic": topic_name, "category": category, "score": 1.8, "source": "contract"},
                )
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                labels = [
                    table,
                    str(manifest_item.get("tableComment") or ""),
                    str(manifest_item.get("dataGrain") or ""),
                ]
                for phrase in semantic_alias_phrases(labels):
                    add_semantic_lexicon_entry(
                        lexicon,
                        phrase,
                        {
                            "topic": topic_name,
                            "category": category,
                            "table": table,
                            "score": 1.6,
                            "source": "manifest",
                        },
                    )
                for metric in topic_assets.load_table_metrics(topic_name, table):
                    metric_topics = asset_entry_topic_categories(topic_assets, topic_name, metric)
                    labels = [
                        str(metric.get("businessName") or ""),
                        str(metric.get("metricKey") or ""),
                        *[str(alias) for alias in metric.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        for metric_topic in metric_topics or [category]:
                            add_semantic_lexicon_entry(
                                lexicon,
                                phrase,
                                {
                                    "topic": topic_name,
                                    "category": metric_topic,
                                    "table": table,
                                    "score": 1.4,
                                    "source": "metric",
                                },
                            )
                for field in topic_assets.load_table_semantic_columns(topic_name, table):
                    labels = [
                        str(field.get("businessName") or ""),
                        str(field.get("columnName") or ""),
                        *[str(alias) for alias in field.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        add_semantic_lexicon_entry(
                            lexicon,
                            phrase,
                            {
                                "topic": topic_name,
                                "category": category,
                                "table": table,
                                "score": 1.2,
                                "source": "dimension",
                            },
                        )
                for term in topic_assets.load_table_terms(topic_name, table):
                    term_topics = asset_entry_topic_categories(topic_assets, topic_name, term)
                    labels = [
                        str(term.get("term") or ""),
                        *[str(alias) for alias in term.get("aliases") or []],
                    ]
                    for phrase in semantic_alias_phrases(labels):
                        for term_topic in term_topics or [category]:
                            add_semantic_lexicon_entry(
                                lexicon,
                                phrase,
                                {
                                    "topic": topic_name,
                                    "category": term_topic,
                                    "table": table,
                                    "score": 1.5,
                                    "source": "term",
                                },
                            )
        return dict(lexicon)

    def _build_semantic_dimension_lexicon(self, topic_assets: Any) -> Dict[str, List[Dict[str, Any]]]:
        if topic_assets is None:
            return {}
        lexicon: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for topic_name in topic_assets.all_topic_names():
            category = resolve_asset_topic_category(topic_assets, topic_name)
            for manifest_item in topic_assets.load_manifest(topic_name):
                table = str(manifest_item.get("tableName") or "")
                for field in topic_assets.load_table_semantic_columns(topic_name, table):
                    topic_candidates = asset_entry_topic_categories(topic_assets, topic_name, field)
                    column = str(field.get("columnName") or "")
                    if not column:
                        continue
                    role = str(field.get("role") or "").upper()
                    if str(field.get("metricFormula") or "").strip() or field.get("aggregationPolicy"):
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
                            {
                                "topic": topic_name,
                                "category": topic_candidates[0] if topic_candidates else category,
                                "topicCandidates": topic_candidates or [category],
                                "table": table,
                                "column": column,
                                "role": role,
                                "score": score,
                            },
                        )
        return dict(lexicon)


def normalize_keyword_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return collapse_whitespace(text)


def metric_detail_grouping_dimensions(
    normalized_question: str,
    dimension_mentions: List[KeywordMention],
) -> List[KeywordMention]:
    """Keep the grouping dimension as the detail-lineage requirement.

    The ranking dimension must be covered by the serving fact, while a separate
    enrichment dimension may legitimately live in another Topic. Requiring
    every enrichment on the fact table would discard a valid lineage.
    """

    if not dimension_mentions:
        return []
    ranking_spans = extract_ranking_spans(normalized_question)
    grouping_phrases = {
        normalize_keyword_text(item.phrase)
        for item in dimension_mentions
        if item.phrase
        and _dimension_is_grouping_target(
            normalized_question,
            normalize_keyword_text(item.phrase),
            ranking_spans,
        )
    }
    if not grouping_phrases:
        return dimension_mentions
    return [
        item
        for item in dimension_mentions
        if normalize_keyword_text(item.phrase) in grouping_phrases
    ]


def _dimension_is_grouping_target(
    question: str,
    phrase: str,
    ranking_spans: List[RouteLexicalSpan],
) -> bool:
    for start, _end in phrase_spans(question, phrase):
        if any(question[:start].endswith(prefix) for prefix in _ROUTING_LANGUAGE.grouping_prefixes):
            return True
        for ranking in ranking_spans:
            if ranking.end > start:
                continue
            between = question[ranking.end:start].strip()
            if not between or between in _ROUTING_LANGUAGE.ranking_link_particles:
                return True
    return False


def semantic_metric_reference_identity(ref_id: str) -> Optional[tuple[str, str, str]]:
    parts = parse_prefixed_reference(
        ref_id,
        prefix="semantic:",
        separator=":",
        part_count=4,
    )
    if not parts or parts[2] != "metric":
        return None
    topic, table, _kind, metric_key = (str(item).strip() for item in parts)
    return (topic, table, metric_key) if topic and table and metric_key else None


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
    if not is_ascii_word_phrase(phrase, extras=(" ", "-")):
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
    return semantic_alias_phrases(labels)


def phrase_spans(text: str, phrase: str) -> List[tuple[int, int]]:
    normalized_phrase = normalize_keyword_text(phrase)
    if not normalized_phrase:
        return []
    if len(normalized_phrase) == 1 and normalized_phrase != text.strip():
        return []
    word_boundary = is_ascii_word_phrase(normalized_phrase, extras=(" ", "-"))
    return [
        (start, end)
        for start, end, _matched in literal_spans(
            text,
            [normalized_phrase],
            case_sensitive=False,
            ascii_word_boundary=word_boundary,
        )
    ]


def independent_dimension_mentions(
    normalized_question: str,
    dimensions: List[KeywordMention],
    metrics: List[KeywordMention],
) -> List[KeywordMention]:
    """Drop dimension labels that occur only inside a governed metric label."""

    metric_spans = [
        span
        for phrase in dedupe_ordered([item.phrase for item in metrics])
        for span in phrase_spans(normalized_question, phrase)
    ]
    if not metric_spans:
        return dimensions
    independent_phrases: Set[str] = set()
    for phrase in dedupe_ordered([item.phrase for item in dimensions]):
        occurrences = phrase_spans(normalized_question, phrase)
        if any(
            not any(metric_start <= start and end <= metric_end for metric_start, metric_end in metric_spans)
            for start, end in occurrences
        ):
            independent_phrases.add(normalize_keyword_text(phrase))
    return [
        item
        for item in dimensions
        if normalize_keyword_text(item.phrase) in independent_phrases
    ]


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
    terminators = frozenset("，。；,;")
    for start, end, _prefix in literal_spans(text, _ROUTING_LANGUAGE.negation_prefixes):
        cursor = end
        while cursor < len(text) and text[cursor] not in terminators:
            cursor += 1
        value = text[end:cursor].strip()
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


def classify_analysis_intent(
    actions: List[str],
    lexical_spans: List[RouteLexicalSpan],
) -> str:
    """Expose only analysis intent that routing can prove structurally.

    A typed ranking span has an operator/position contract and may therefore be
    surfaced as ``ranking``. Action tokens are retained as lexical facts, but
    their business meaning belongs to Planner understanding rather than a
    routing synonym table.
    """

    if any(span.span_type == RouteSpanType.RANKING for span in lexical_spans):
        return "ranking"
    return "unresolved" if actions else "lookup"


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


def _is_governed_greeting(value: str) -> bool:
    text = str(value or "").strip().casefold()
    for phrase in _ROUTING_LANGUAGE.greeting_phrases:
        normalized_phrase = phrase.casefold()
        if not text.startswith(normalized_phrase):
            continue
        suffix = text[len(normalized_phrase) :]
        if all(
            character.isspace() or character in _ROUTING_LANGUAGE.greeting_suffix_characters
            for character in suffix
        ):
            return True
    return False


def _object_reference_spans(text: str, column: str) -> List[tuple[int, int, str]]:
    value = str(text or "")
    haystack = value.casefold()
    needle = str(column or "").casefold()
    if not needle:
        return []
    spans: List[tuple[int, int, str]] = []
    cursor = 0
    separators = frozenset(":=：-")
    while cursor < len(value):
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        if start > 0 and value[start - 1].isascii() and (
            value[start - 1].isalnum() or value[start - 1] == "_"
        ):
            cursor = start + 1
            continue
        separator_cursor = start + len(needle)
        while separator_cursor < len(value) and value[separator_cursor].isspace():
            separator_cursor += 1
        if separator_cursor >= len(value):
            break
        if value[separator_cursor] == "_":
            value_start = separator_cursor + 1
        elif value[separator_cursor] in separators:
            value_start = separator_cursor + 1
            while value_start < len(value) and value[value_start].isspace():
                value_start += 1
        else:
            cursor = start + 1
            continue
        if value_start >= len(value) or not (
            value[value_start].isascii() and value[value_start].isalnum()
        ):
            cursor = start + 1
            continue
        end = value_start + 1
        while end < len(value) and value[end].isascii() and (
            value[end].isalnum() or value[end] in {"_", "-"}
        ):
            end += 1
        spans.append((start, end, value[start:end]))
        cursor = end
    return spans


class QuestionRoutingService:
    def __init__(self, topic_assets: Any = None):
        self.topic_assets = topic_assets or default_topic_assets()
        self._slot_extractor = RouteSlotExtractor(self.topic_assets)

    def route(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> RoutingDecision:
        normalized = (question or "").strip().lower()
        if not normalized:
            return RoutingDecision(route=QuestionRoute.INVALID, reason="空问题")
        if _is_governed_greeting(normalized):
            return RoutingDecision(route=QuestionRoute.GREETING, reason="寒暄问题")
        if self._is_ambiguous_question(normalized, keywords, recall_bundle):
            return RoutingDecision(route=QuestionRoute.INVALID, reason="问题表达不明确，建议补充业务对象或查询目标")
        simple_detail = self._is_simple_detail_lookup(normalized, keywords, recall_bundle)
        complex_question = (not simple_detail) and (
            len(normalized) >= 24
            or contains_any_literal(normalized, _ROUTING_LANGUAGE.action_markers)
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
        return len(extract_temporal_lexical_spans(question)) >= 2

    def _has_any_time_range(self, question: str) -> bool:
        return bool(extract_temporal_lexical_spans(question))

    def _is_simple_detail_lookup(self, question: str, keywords: ExtractedKeywords, recall_bundle: RecallBundle) -> bool:
        if not contains_any_literal(question, _ROUTING_LANGUAGE.detail_markers):
            return False
        has_object_ref = self._slot_extractor.has_object_ref(question)
        if (not self._has_any_time_range(question) and not has_object_ref) or self._has_multiple_time_ranges(question):
            return False
        non_detail_actions = [
            marker
            for marker in _ROUTING_LANGUAGE.action_markers
            if marker not in _ROUTING_LANGUAGE.detail_markers
        ]
        if contains_any_literal(question, non_detail_actions):
            return False
        if self._matched_domain_count(question, keywords) >= 2:
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
            "BUSINESS_CHAT 只用于助手能力说明、概念闲聊、无需查数或分析的问题；"
            "surfaceSignals 命中发布资产时不要输出 BUSINESS_CHAT。"
            "如果存在 pendingContext 且当前输入像补充条件、确认或选择，优先 CLARIFICATION_REPLY。"
            "如果用户要求删除、修改、更新、创建、写入、导入、重建等写操作，输出 UNSUPPORTED_WRITE。"
        )
        payload = {
            "question": str(question or "")[:800],
            "pendingContext": bool(pending_context),
            "surfaceSignals": surface_signals,
            "decisionHints": {
                "publishedAssetSignal": "BUSINESS_TASK",
                "pendingContextSelection": "CLARIFICATION_REPLY",
                "unsupportedWriteSignal": "UNSUPPORTED_WRITE",
                "assistantCapabilityQuestion": "BUSINESS_CHAT",
                "unrelatedWithoutAssetSignal": "INVALID",
            },
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
        routing_service: Optional[QuestionRoutingService] = None,
        slot_extractor: Optional["RouteSlotExtractor"] = None,
        semantic_classifier: Optional[SemanticPreflightRouteClassifier] = None,
    ):
        self.settings = settings
        self.keyword_service = keyword_service
        # ``routing_service`` was part of the legacy full-understanding
        # constructor but the online preflight has never called it.  Keep the
        # positional parameter only for compatibility; do not retain a second
        # routing authority in the surface gate.
        del routing_service
        self.slot_extractor = slot_extractor or RouteSlotExtractor(
            keyword_service.topic_assets
        )
        self.semantic_classifier = semantic_classifier or (
            SemanticPreflightRouteClassifier(settings)
        )

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
        del pending_context
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
        has_time = bool(extract_temporal_lexical_spans(text))
        has_object_ref = self.slot_extractor.has_object_ref(text)
        lexical_spans = extract_route_lexical_spans(text)
        has_typed_ranking = any(span.span_type == RouteSpanType.RANKING for span in lexical_spans)
        write_operation = contains_any_literal(lowered, _ROUTING_LANGUAGE.write_markers, case_sensitive=False)
        greeting = _is_governed_greeting(lowered)
        assistant_chat_phrase = contains_any_literal(text, _ROUTING_LANGUAGE.assistant_chat_phrases)
        business_metric_like = bool(business_surface.get("hasPublishedMetricPhrase"))
        metric_like = business_metric_like
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
            has_typed_ranking
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
            "hasTypedRankingSpan": has_typed_ranking,
            "hasMetricLikePhrase": metric_like,
            "hasBusinessMetricLikePhrase": business_metric_like,
            "hasGenericMetricLikePhrase": False,
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
        # A phrase resolved from the published semantic directory is already a
        # governed business-surface fact.  When the small model is unavailable
        # (or misreads the turn), preserve that fact and let Topic/Planner own
        # the later interpretation instead of rejecting the request here.
        has_business_surface = bool(
            signals.get("hasBusinessDomainPhrase")
            or signals.get("hasObjectRef")
            or signals.get("hasMetricLikePhrase")
            or signals.get("hasAnalysisIntent")
        )
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
    WRITE_TERMS = _ROUTING_LANGUAGE.write_markers
    RISK_TERMS = _ROUTING_LANGUAGE.risk_markers

    def __init__(self, topic_assets: Any = None):
        self.topic_assets = topic_assets or default_topic_assets()
        self._object_patterns, self._object_topics = self._build_object_contracts(self.topic_assets)

    def has_object_ref(self, text: str) -> bool:
        return any(_object_reference_spans(str(text or ""), ref_type) for ref_type in self._object_patterns)

    def _build_object_contracts(
        self,
        topic_assets: Any,
    ) -> tuple[List[str], Dict[str, List[QuestionCategory]]]:
        """Compile object references only from formally declared entity contracts."""

        if topic_assets is None:
            return [], {}
        topics: Dict[str, List[QuestionCategory]] = defaultdict(list)
        try:
            topic_names = topic_assets.all_topic_names()
        except Exception:
            return [], {}
        for topic_name in topic_names:
            category = resolve_asset_topic_category(topic_assets, topic_name)
            try:
                manifest = topic_assets.load_manifest(topic_name)
            except Exception:
                manifest = []
            for item in manifest:
                table = str(item.get("tableName") or "") if isinstance(item, dict) else ""
                if not table:
                    continue
                try:
                    fields = topic_assets.load_table_semantic_columns(topic_name, table)
                except Exception:
                    fields = []
                for field in fields:
                    if not isinstance(field, dict) or not (
                        str(field.get("canonicalEntityRef") or "").strip()
                        or field.get("isUniqueEntityKey") is True
                    ):
                        continue
                    column = str(field.get("columnName") or "").strip()
                    if not is_ascii_identifier(column):
                        continue
                    if category not in topics[column]:
                        topics[column].append(category)
        patterns = sorted(topics, key=lambda value: (-len(value), value))
        return patterns, dict(topics)

    def extract(self, question: str, keywords: ExtractedKeywords) -> RouteSlots:
        text = question or ""
        object_refs = self._object_refs(text)
        time_window = self._time_window(text, keywords)
        operation = (
            "write_requested"
            if contains_any_literal(text, self.WRITE_TERMS, case_sensitive=False)
            else "read"
        )
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
        for ref_type in self._object_patterns:
            for _start, _end, raw in _object_reference_spans(text, ref_type):
                value = raw.replace("：", "_").replace(":", "_").replace("=", "_").replace("-", "_")
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
        spans = extract_temporal_lexical_spans(text)
        return spans[0].text if spans else ""

    def _analysis_signals(self, keywords: ExtractedKeywords) -> List[str]:
        if keywords and any(
            span.span_type == RouteSpanType.RANKING
            for span in keywords.lexical_spans
        ):
            return ["typed_ranking_span"]
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
        for ref in object_refs:
            for category in self._object_topics.get(ref.ref_type, []):
                payload = by_topic.setdefault(category, {"score": 0, "evidence": []})
                payload["score"] = float(payload.get("score") or 0) + 2.0
                evidence = list(payload.get("evidence") or [])
                if ref.ref_type not in evidence:
                    evidence.append(ref.ref_type)
                payload["evidence"] = evidence[:8]
        ordered = []
        for category in topic_domain_order(by_topic):
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
        if contains_any_literal(text, self.RISK_TERMS):
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


def topic_domain_order(scores: Optional[Dict[QuestionCategory, Any]] = None) -> List[QuestionCategory]:
    """Return a deterministic, score-first order for any asset category set."""

    values = list((scores or {}).keys())
    return sorted(
        values,
        key=lambda category: (
            -float(((scores or {}).get(category) or {}).get("score") or 0)
            if isinstance((scores or {}).get(category), dict)
            else -float((scores or {}).get(category) or 0),
            str(category),
        ),
    )


class SemanticTopicRouterService:
    """Asset-bounded Topic scope selection over the published L0 directory.

    Published metric, dimension, entity and Topic signals establish the
    deterministic candidate floor.  The LLM may resolve ambiguity or add/remove
    unprotected candidates inside that published card set, but it cannot erase
    an exact governed metric owner.  Query planning still belongs to the Core.
    """

    STATUSES = {"RESOLVED", "AMBIGUOUS", "UNSUPPORTED"}

    def __init__(self, settings: Any, topic_assets: Any = None, llm: Any = None):
        self.settings = settings
        self.topic_assets = topic_assets or default_topic_assets()
        if llm is not None:
            self.llm = llm
        else:
            explicit_topic_model = str(
                getattr(settings, "topic_semantic_route_model", "") or ""
            ).strip()
            model = (
                explicit_topic_model
                or str(getattr(settings, "llm_fast_model", "") or "")
                or str(getattr(settings, "preflight_semantic_route_model", "") or "")
                or str(getattr(settings, "openai_model", "") or "")
            )
            try:
                from merchant_ai.services.llm import LlmClient

                explicit_topic_base_url = str(
                    getattr(settings, "topic_semantic_route_base_url", "")
                    or ""
                ).strip()
                explicit_topic_api_key = str(
                    getattr(settings, "topic_semantic_route_api_key", "")
                    or ""
                ).strip()
                # An explicitly configured Topic model belongs to the Topic
                # provider. If it has no dedicated endpoint/key, inherit the
                # main provider instead of pairing it with preflight/Kimi.
                if explicit_topic_model or explicit_topic_base_url or explicit_topic_api_key:
                    route_base_url = (
                        explicit_topic_base_url
                        or str(getattr(settings, "openai_base_url", "") or "")
                    )
                    route_api_key = (
                        explicit_topic_api_key
                        or str(getattr(settings, "openai_api_key", "") or "")
                    )
                else:
                    route_base_url = (
                        str(getattr(settings, "preflight_llm_base_url", "") or "")
                        or str(getattr(settings, "openai_base_url", "") or "")
                    )
                    route_api_key = (
                        str(getattr(settings, "preflight_llm_api_key", "") or "")
                        or str(getattr(settings, "openai_api_key", "") or "")
                    )
                extra_body = (
                    {"thinking": {"type": "disabled"}}
                    if model.lower().startswith("kimi-for-coding")
                    and "api.kimi.com/coding" in route_base_url.lower()
                    else None
                )
                self.llm = LlmClient(
                    settings,
                    model_name=model,
                    api_key=route_api_key,
                    base_url=route_base_url,
                    extra_body=extra_body,
                    max_tokens=400,
                )
            except Exception:
                self.llm = None

    def route(
        self,
        question: str,
        keywords: Optional[ExtractedKeywords] = None,
        route_slots: Optional[RouteSlots] = None,
        **_: Any,
    ) -> TopicRoutingDecision:
        """Compatibility entry point for semantic routing with safe fallback."""

        return self.route_with_budget(
            question,
            keywords=keywords,
            route_slots=route_slots,
        )

    def route_with_budget(
        self,
        question: str,
        *,
        keywords: Optional[ExtractedKeywords] = None,
        route_slots: Optional[RouteSlots] = None,
        runtime_budget: Any = None,
    ) -> TopicRoutingDecision:
        cards = self.topic_cards()
        topic_names = [str(item.get("topic") or "") for item in cards if item.get("topic")]
        if not topic_names:
            return TopicRoutingDecision(
                primary_topic=QuestionCategory.UNKNOWN,
                candidate_topics=[],
                confidence=0.0,
                clarification_required=True,
                routing_mode="topic_catalog_empty",
                selection_mode="semantic_llm",
                selection_evidence={
                    "router": "semantic_topic_llm",
                    "status": "catalog_empty",
                    "keywordRoutingUsed": False,
                },
                reason="没有已发布 Topic，无法建立语义检索范围",
            )

        asset_baseline = self._published_asset_baseline(
            question,
            keywords,
            route_slots,
            cards,
            topic_names,
        )
        protected_topics = list(
            asset_baseline.get("protectedTopicNames") or []
        )
        candidate_topics = list(
            asset_baseline.get("candidateTopicNames") or []
        )
        selection_evidence = dict(
            asset_baseline.get("assetSelectionEvidence") or {}
        )
        high_confidence = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.settings,
                        "route_topic_high_confidence",
                        0.75,
                    )
                    or 0.75
                ),
            ),
        )
        # A unique published metric owner is already stronger routing evidence
        # than another model pass. Topic is only a recall priority here; the
        # governed Contract and ACL still validate every actual query asset.
        if (
            len(protected_topics) == 1
            and candidate_topics == protected_topics
            and float(asset_baseline.get("assetConfidence") or 0.0)
            >= high_confidence
            and str(selection_evidence.get("queryShape") or "")
            == "summary_or_total"
            and len(selection_evidence.get("matchedMetrics") or []) == 1
        ):
            decision = self._deterministic_fallback_decision(
                question,
                keywords,
                route_slots=route_slots,
                asset_baseline=asset_baseline,
                status="exact_metric_asset_match",
                reason="唯一已发布指标归属已确定",
            )
            evidence = dict(decision.selection_evidence or {})
            evidence.update(
                {
                    "router": "published_asset_exact_metric",
                    "status": "resolved_without_llm",
                    "modelCallSkipped": True,
                    "assetConfidence": float(
                        asset_baseline.get("assetConfidence") or 0.0
                    ),
                }
            )
            return decision.model_copy(
                update={
                    "confidence": float(
                        asset_baseline.get("assetConfidence") or 0.0
                    ),
                    "clarification_required": False,
                    "routing_mode": "semantic_topic_exact_asset",
                    "selection_mode": "automatic_asset_exact",
                    "selection_evidence": evidence,
                    "reason": "唯一已发布指标归属已确定；跳过 Topic 模型调用",
                }
            )
        # The model may compare published Topic cards, except a detail owner
        # that the formal metric lineage explicitly marks as inappropriate for
        # this exact summary request. This prevents a bare store total from
        # drifting to a detail COUNT merely because both share an alias.
        suppressed_topic_names = set(
            asset_baseline.get("suppressedTopicNames") or []
        )
        model_topic_names = [
            topic
            for topic in topic_names
            if topic not in suppressed_topic_names
        ]
        model_cards = [
            card
            for card in cards
            if str(card.get("topic") or "") in set(model_topic_names)
        ]

        if not self._llm_available():
            return self._deterministic_fallback_decision(
                question,
                keywords,
                route_slots=route_slots,
                asset_baseline=asset_baseline,
                status="llm_unavailable",
                reason="Topic LLM 不可用",
            )

        attempts = max(
            1,
            min(2, int(getattr(self.settings, "topic_semantic_route_max_attempts", 2) or 2)),
        )
        minimum_confidence = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(self.settings, "topic_semantic_route_min_confidence", 0.55)
                    or 0.55
                ),
            ),
        )
        validation_error = ""
        last_payload: Dict[str, Any] = {}
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                payload = self._call_model(
                    question,
                    model_cards,
                    model_topic_names,
                    asset_baseline=asset_baseline,
                    validation_error=validation_error,
                    runtime_budget=runtime_budget,
                    attempt=attempt,
                )
            except Exception as exc:
                if isinstance(exc, GroundedRuntimeBudgetExceeded):
                    raise
                last_error = "%s:%s" % (type(exc).__name__, str(exc)[:300])
                break
            last_payload = payload if isinstance(payload, dict) else {}
            provider_error = str(getattr(self.llm, "last_error", "") or "")
            if not last_payload and provider_error:
                last_error = provider_error
                break
            normalized, validation_error = self._validate_payload(
                last_payload,
                model_topic_names,
            )
            if normalized is None:
                last_error = validation_error
                continue
            if (
                normalized["status"] == "RESOLVED"
                and normalized["confidence"] < minimum_confidence
                and attempt < attempts
            ):
                validation_error = (
                    "上一次返回 RESOLVED 但置信度 %.2f 低于 %.2f。"
                    "请改为 AMBIGUOUS 并返回所有合理 Topic，或提高有依据的置信度。"
                    % (normalized["confidence"], minimum_confidence)
                )
                continue
            if (
                normalized["status"] == "RESOLVED"
                and normalized["confidence"] < minimum_confidence
            ):
                normalized = {
                    **normalized,
                    "status": "AMBIGUOUS",
                    "ambiguityReason": (
                        normalized.get("ambiguityReason")
                        or "Topic 模型最终置信度低于 %.2f，按歧义候选处理"
                        % minimum_confidence
                    ),
                }
            return self._decision_from_payload(
                normalized,
                topic_names,
                asset_baseline=asset_baseline,
                attempts=attempt,
            )

        provider_error = str(getattr(self.llm, "last_error", "") or "")
        return self._deterministic_fallback_decision(
            question,
            keywords,
            route_slots=route_slots,
            asset_baseline=asset_baseline,
            status="llm_failed",
            reason=(
                "Topic LLM 返回无效结果"
            ),
            detail=(last_error or provider_error or str(last_payload)[:300]),
        )

    def _published_asset_baseline(
        self,
        question: str,
        keywords: Optional[ExtractedKeywords],
        route_slots: Optional[RouteSlots],
        cards: List[Dict[str, Any]],
        topic_names: List[str],
    ) -> Dict[str, Any]:
        """Build one shared candidate floor for both model and fallback paths."""

        extracted = keywords or ExtractedKeywords(
            normalized_question=str(question or "")
        )
        slots = route_slots or RouteSlots()
        asset_decision = TopicRouterService(self.topic_assets).route(
            question,
            extracted,
            route_slots=slots,
        )
        asset_categories = list(asset_decision.recall_topics())
        slot_categories = [
            item.topic
            for item in slots.topic_candidates
            if item.topic != QuestionCategory.UNKNOWN
        ]
        topic_name_set = set(topic_names)

        def names_for_categories(
            categories: List[QuestionCategory],
        ) -> List[str]:
            loader = getattr(
                self.topic_assets,
                "topic_names_for_categories",
                None,
            )
            if callable(loader):
                try:
                    return [
                        str(item)
                        for item in loader(categories)
                        if str(item) in topic_name_set
                    ]
                except Exception:
                    pass
            values: List[str] = []
            for category in categories:
                raw = str(getattr(category, "value", category) or "")
                if raw in topic_name_set and raw not in values:
                    values.append(raw)
            return values

        asset_topic_names = names_for_categories(asset_categories)
        slot_topic_names = names_for_categories(slot_categories)
        selection_evidence = dict(asset_decision.selection_evidence or {})
        serving_topics = [
            str(item or "").strip()
            for item in selection_evidence.get("servingTopics") or []
            if str(item or "").strip() in topic_name_set
        ]
        business_topics = [
            str(item or "").strip()
            for item in selection_evidence.get("businessTopics") or []
            if str(item or "").strip() in topic_name_set
        ]
        detail_requested = (
            str(selection_evidence.get("queryShape") or "")
            == "detail_or_breakdown"
        )
        declared_topics = dedupe_ordered(
            [
                *serving_topics,
                *(business_topics if detail_requested else []),
            ]
        )
        protected_topic_names = self._protected_metric_topic_names(
            extracted,
            topic_name_set,
        )
        suppressed_topic_names: List[str] = []
        if not detail_requested:
            for metric in selection_evidence.get("matchedMetrics") or []:
                if not isinstance(metric, dict):
                    continue
                metric_intent = str(
                    metric.get("metricIntent") or ""
                ).strip()
                metric_grain = str(
                    metric.get("metricGrain") or ""
                ).strip()
                if (
                    metric_intent != "store_summary"
                    and metric_grain != "merchant_day_summary"
                ):
                    continue
                serving_topic = str(
                    metric.get("servingTopic") or ""
                ).strip()
                if serving_topic not in protected_topic_names:
                    continue
                for topic in metric.get("businessTopics") or []:
                    topic_name = str(topic or "").strip()
                    if (
                        topic_name
                        and topic_name != serving_topic
                        and topic_name in topic_name_set
                        and topic_name not in slot_topic_names
                        and topic_name not in suppressed_topic_names
                    ):
                        suppressed_topic_names.append(topic_name)
        card_scores = self._topic_card_fallback_scores(question, cards)
        card_by_topic = {
            str(card.get("topic") or ""): card
            for card in cards
            if str(card.get("topic") or "")
        }
        ranked_card_topics = [
            topic
            for topic, score in sorted(
                card_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if score > 0
            and topic not in suppressed_topic_names
            and (
                not detail_requested
                or bool(
                    (card_by_topic.get(topic) or {}).get(
                        "supportsDetail"
                    )
                )
            )
        ]
        if ranked_card_topics:
            top_score = card_scores[ranked_card_topics[0]]
            threshold = max(0.9, top_score * 0.6)
            ranked_card_topics = [
                topic
                for topic in ranked_card_topics
                if card_scores[topic] >= threshold
            ][:4]
        candidate_topic_names = [
            topic
            for topic in dedupe_ordered(
            [
                *protected_topic_names,
                *asset_topic_names,
                *slot_topic_names,
                *declared_topics,
                *ranked_card_topics,
            ]
            )
            if topic not in suppressed_topic_names
            or topic in protected_topic_names
        ]
        return {
            "candidateTopicNames": candidate_topic_names,
            "protectedTopicNames": protected_topic_names,
            "assetSignalTopicNames": dedupe_ordered(
                [*asset_topic_names, *slot_topic_names, *declared_topics]
            ),
            "suppressedTopicNames": suppressed_topic_names,
            "assetRoutingMode": str(asset_decision.routing_mode or ""),
            "assetConfidence": float(asset_decision.confidence or 0.0),
            "assetSelectionEvidence": selection_evidence,
            "topicCardScores": {
                topic: round(score, 4)
                for topic, score in sorted(
                    card_scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if score > 0
            },
        }

    def _protected_metric_topic_names(
        self,
        keywords: ExtractedKeywords,
        published_topics: Set[str],
    ) -> List[str]:
        mentions = list(keywords.mentions or [])
        lineage_phrases = {
            normalize_keyword_text(item.phrase)
            for item in mentions
            if item.kind == "lineage" and item.phrase
        }
        protected_categories: List[QuestionCategory] = []
        for mention in mentions:
            phrase = normalize_keyword_text(mention.phrase)
            exact_metric = (
                mention.kind == "lineage"
                or (
                    mention.kind == "metric"
                    and phrase not in lineage_phrases
                )
            )
            if (
                not exact_metric
                or mention.topic == QuestionCategory.UNKNOWN
                or not str(mention.source or "").startswith("semantic_metric")
            ):
                continue
            if mention.topic not in protected_categories:
                protected_categories.append(mention.topic)
        loader = getattr(
            self.topic_assets,
            "topic_names_for_categories",
            None,
        )
        if callable(loader):
            try:
                return dedupe_ordered(
                    [
                        str(item)
                        for item in loader(protected_categories)
                        if str(item) in published_topics
                    ]
                )
            except Exception:
                pass
        return dedupe_ordered(
            [
                str(getattr(item, "value", item) or "")
                for item in protected_categories
                if str(getattr(item, "value", item) or "")
                in published_topics
            ]
        )

    def _deterministic_fallback_decision(
        self,
        question: str,
        keywords: Optional[ExtractedKeywords],
        *,
        route_slots: Optional[RouteSlots] = None,
        asset_baseline: Optional[Dict[str, Any]] = None,
        status: str,
        reason: str,
        detail: str = "",
    ) -> TopicRoutingDecision:
        """Degrade to published semantic keyword evidence, never all Topics.

        The fallback reuses the asset-generated keyword scores.  It does not
        contain business-name conditionals and it remains only a seed search
        scope; the Core still has to read and bind formal semantic assets.
        """

        fallback = TopicRouterService(self.topic_assets).route(
            question,
            keywords or ExtractedKeywords(normalized_question=question),
            route_slots=route_slots,
        )
        cards = self.topic_cards()
        topic_names = [
            str(item.get("topic") or "")
            for item in cards
            if item.get("topic")
        ]
        baseline = asset_baseline or self._published_asset_baseline(
            question,
            keywords,
            route_slots,
            cards,
            topic_names,
        )
        max_candidates = max(
            1,
            min(
                4,
                int(
                    getattr(
                        self.settings,
                        "route_topic_max_candidates",
                        4,
                    )
                    or 4
                ),
            ),
        )
        candidate_names = list(
            baseline.get("candidateTopicNames") or []
        )[:max_candidates]
        candidates = dedupe_topics(
            [
                resolve_asset_topic_category(self.topic_assets, topic)
                for topic in candidate_names
            ]
        )
        candidates = [
            item
            for item in candidates
            if item != QuestionCategory.UNKNOWN
        ]
        evidence = {
            **dict(fallback.selection_evidence or {}),
            "router": "semantic_topic_llm",
            "status": status,
            "keywordRoutingUsed": True,
            "assetSignalBaselineUsed": True,
            "fallbackRouter": "published_asset_signal_baseline",
            "candidateTopicNames": candidate_names,
            "protectedTopicNames": list(
                baseline.get("protectedTopicNames") or []
            ),
            "suppressedTopicNames": list(
                baseline.get("suppressedTopicNames") or []
            ),
            "topicCardScores": dict(
                baseline.get("topicCardScores") or {}
            ),
            "modelSelectedTopics": [],
            "publishedTopicCount": len(cards),
            "detail": str(detail or "")[:300],
        }
        return TopicRoutingDecision(
            primary_topic=QuestionCategory.UNKNOWN,
            candidate_topics=candidates,
            dimension_topics=[],
            confidence=float(fallback.confidence or 0.0),
            clarification_required=not candidates,
            routing_mode=(
                "semantic_topic_asset_fallback"
                if candidates
                else "semantic_topic_open_discovery"
            ),
            workspace_topics=candidates,
            selection_mode="semantic_llm_degraded_asset_fallback",
            selection_evidence=evidence,
            reason=(
                "%s；已使用发布语义资产生成的确定性候选 Topic：%s"
                % (reason, "、".join(str(item.value) for item in candidates))
                if candidates
                else "%s；没有足够的已发布语义信号，保留开放发现但不预加载全部 Topic"
                % reason
            ),
        )

    @staticmethod
    def _topic_card_fallback_scores(
        question: str,
        cards: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Rank published L0 Topic cards without business-name conditionals."""

        query = normalize_keyword_text(question or "")
        if not query:
            return {}
        query_ngrams = {
            query[index : index + width]
            for width in (2, 3)
            for index in range(max(0, len(query) - width + 1))
            if not query[index : index + width].isdigit()
        }

        def normalized_values(values: Any) -> List[str]:
            return list(
                dict.fromkeys(
                    normalize_keyword_text(value)
                    for value in (values or [])
                    if normalize_keyword_text(value)
                )
            )

        def approximate_ratio(label: str) -> float:
            if len(label) < 3 or len(query) < 2:
                return 0.0
            minimum = max(2, len(label) - 2)
            maximum = min(len(query), len(label) + 2)
            best = 0.0
            for width in range(minimum, maximum + 1):
                for index in range(len(query) - width + 1):
                    best = max(
                        best,
                        SequenceMatcher(
                            None,
                            label,
                            query[index : index + width],
                        ).ratio(),
                    )
            return best

        scores: Dict[str, float] = {}
        for card in cards:
            topic = str(card.get("topic") or "").strip()
            if not topic:
                continue
            score = 0.0
            for values, weight in (
                (
                    [
                        card.get("topic"),
                        card.get("displayName"),
                        *(card.get("aliases") or []),
                    ],
                    5.0,
                ),
                (card.get("metricAliases") or [], 4.5),
                (card.get("ruleAliases") or [], 3.5),
                (card.get("columnAliases") or [], 2.0),
            ):
                for label in normalized_values(values):
                    if label in query:
                        score += weight + min(len(label), 8) * 0.15
                        continue
                    similarity = approximate_ratio(label)
                    if similarity >= 0.62:
                        score += weight * similarity * 0.7
            searchable = normalize_keyword_text(
                " ".join(
                    [
                        *(card.get("capabilitySummaries") or []),
                        *(card.get("dataGrains") or []),
                    ]
                )
            )
            if searchable and query_ngrams:
                card_ngrams = {
                    searchable[index : index + width]
                    for width in (2, 3)
                    for index in range(
                        max(0, len(searchable) - width + 1)
                    )
                }
                score += min(
                    4.0,
                    len(query_ngrams.intersection(card_ngrams)) * 0.35,
                )
            if score > 0:
                scores[topic] = score
        return scores

    def topic_cards(self) -> List[Dict[str, Any]]:
        """Build compact, question-independent Topic cards from published L0 assets."""

        cards: List[Dict[str, Any]] = []
        names_loader = getattr(self.topic_assets, "all_topic_names", None)
        topic_names = list(names_loader() or []) if callable(names_loader) else []
        for topic in topic_names:
            contract = load_asset_topic_contract(self.topic_assets, topic)
            manifest_loader = getattr(self.topic_assets, "load_manifest", None)
            try:
                manifest = list(manifest_loader(topic) or []) if callable(manifest_loader) else []
            except Exception:
                manifest = []
            summaries: List[str] = []
            grains: List[str] = []
            metric_aliases: List[str] = []
            column_aliases: List[str] = []
            rule_aliases: List[str] = []
            preferred_for: List[str] = []
            supports_detail = False
            for item in manifest:
                if not isinstance(item, dict):
                    continue
                summary = str(item.get("businessSummary") or "").strip()
                grain = str(item.get("dataGrain") or "").strip()
                if summary and summary not in summaries:
                    summaries.append(summary)
                if grain and grain not in grains:
                    grains.append(grain)
                for intent in item.get("preferredFor") or []:
                    normalized_intent = str(intent or "").strip().upper()
                    if (
                        normalized_intent
                        and normalized_intent not in preferred_for
                    ):
                        preferred_for.append(normalized_intent)
                supports_detail = bool(
                    supports_detail
                    or item.get("supportsDetail")
                    or any(
                        intent in {"DETAIL", "TOPN"}
                        for intent in preferred_for
                    )
                )
                navigation = (
                    item.get("navigationHints")
                    if isinstance(item.get("navigationHints"), dict)
                    else {}
                )
                for key, target in (
                    ("metrics", metric_aliases),
                    ("columns", column_aliases),
                    ("rules", rule_aliases),
                ):
                    for entry in navigation.get(key) or []:
                        if not isinstance(entry, dict):
                            continue
                        for alias in [
                            entry.get("key"),
                            *(entry.get("aliases") or []),
                        ]:
                            normalized = str(alias or "").strip()
                            if normalized and normalized not in target:
                                target.append(normalized)
            cards.append(
                {
                    "topic": str(topic),
                    "displayName": str(contract.get("displayName") or topic),
                    "aliases": [str(item) for item in contract.get("aliases") or []][:8],
                    "capabilitySummaries": summaries[:6],
                    "dataGrains": grains[:6],
                    "supportsDetail": supports_detail,
                    "preferredFor": preferred_for[:12],
                    "metricAliases": metric_aliases[:80],
                    "columnAliases": column_aliases[:80],
                    "ruleAliases": rule_aliases[:40],
                }
            )
        return cards

    def _call_model(
        self,
        question: str,
        cards: List[Dict[str, Any]],
        topic_names: List[str],
        *,
        asset_baseline: Dict[str, Any],
        validation_error: str,
        runtime_budget: Any,
        attempt: int,
    ) -> Dict[str, Any]:
        timeout_seconds = self._timeout_seconds(runtime_budget)
        if runtime_budget is not None:
            runtime_budget.consume_llm_call(name="semantic_topic_router")
        system_prompt = self._system_prompt(topic_names)
        user_payload = {
            "question": str(question or "")[:1200],
            "topicDirectory": cards,
            "publishedSignalBaseline": {
                "candidateTopics": list(
                    asset_baseline.get("candidateTopicNames") or []
                ),
                "protectedTopics": list(
                    asset_baseline.get("protectedTopicNames") or []
                ),
                "excludedTopics": list(
                    asset_baseline.get("suppressedTopicNames") or []
                ),
                "instruction": (
                    "protectedTopics 来自精确发布指标资产，不能删除；"
                    "excludedTopics 是该指标声明的明细下钻 Topic，当前汇总问题不能恢复；"
                    "其余候选可依据 Topic 卡片保留或删除。"
                ),
            },
            "instruction": (
                "只选择后续应检索的 Topic 集合。不要输出指标、维度、时间、操作、表、主 Topic、"
                "支持 Topic、JOIN 或 SQL 计划。"
            ),
            "outputSchema": {
                "status": "RESOLVED|AMBIGUOUS|UNSUPPORTED",
                "relevantTopics": topic_names,
                "confidence": "0.0-1.0",
                "ambiguityReason": "仅 AMBIGUOUS 或 UNSUPPORTED 时填写",
            },
        }
        if validation_error:
            user_payload["repair"] = {
                "previousResultProblem": validation_error,
                "allowedTopics": topic_names,
            }
        tool = self._selection_tool(topic_names)

        def invoke() -> Dict[str, Any]:
            if self._use_tool_json():
                return self.llm.tool_json_chat(
                    system_prompt,
                    json.dumps(user_payload, ensure_ascii=False, default=str),
                    tool,
                    fallback={},
                    timeout_seconds=timeout_seconds,
                )
            if hasattr(self.llm, "json_chat"):
                return self.llm.json_chat(
                    system_prompt,
                    json.dumps(user_payload, ensure_ascii=False, default=str),
                    fallback={},
                    timeout_seconds=timeout_seconds,
                )
            return {}

        if runtime_budget is None:
            return invoke()
        with runtime_budget.stage("llm.topic_route.attempt_%d" % attempt):
            return invoke()

    def _use_tool_json(self) -> bool:
        return hasattr(self.llm, "tool_json_chat")

    def _system_prompt(self, topic_names: List[str]) -> str:
        examples = self._few_shot_examples(topic_names)
        return (
            "你是企业 BI 系统的 Topic 语义路由器。你的唯一任务是判断原始问题需要在哪些已发布 "
            "Topic 中检索语义资产。Topic 只是检索范围，不代表主表、事实锚点或执行顺序。"
            "必须阅读完整 Topic Directory 的业务能力和边界，不得按关键词命中次数做分类。"
            "publishedSignalBaseline.protectedTopics 来自精确命中的正式指标资产，必须全部保留；"
            "publishedSignalBaseline.excludedTopics 是当前汇总指标正式声明不适用的明细下钻 Topic，不得为了相似词恢复；"
            "你只能在给定 Topic Directory 候选中调整其他 Topic，不能扩展到目录之外。"
            "可以选择一个或多个 Topic；问题跨域时把所有相关 Topic 放进同一个 relevantTopics 数组。"
            "当问题询问商家级汇总指标、经营趋势或多个经营指标，且不要求商品、订单、工单等明细维度时，"
            "如果目录中存在明确承载商家-日期聚合指标的画像或汇总 Topic，必须将其加入 relevantTopics；"
            "相关业务事实 Topic 可以同时保留作为语义补充。反之，当问题要求明细、实体下钻、维度拆分或排行时，"
            "画像或汇总 Topic 不能替代承载该粒度的事实 Topic。"
            "不要解析或输出指标、维度、时间范围、聚合方式、排行方式、表、字段、JOIN、Contract 或 SQL。"
            "只能返回目录中真实存在的 Topic 名称，不能翻译、缩写或创造 Topic。"
            "如果有多个合理范围，返回 AMBIGUOUS 并包含所有合理 Topic；如果目录完全不支持，返回 UNSUPPORTED。"
            "不要因为用户问法口语化、同义表达或缺少标准术语就缩小范围。\n"
            "可选 Topic：%s\n"
            "示例：\n%s"
            % ("、".join(topic_names), "\n".join(examples))
        )

    @staticmethod
    def _few_shot_examples(topic_names: List[str]) -> List[str]:
        if not topic_names:
            return ["目录为空时输出 UNSUPPORTED，不得创造 Topic。"]
        return [
            "当一个问题需要目录中多个已发布能力时，返回覆盖这些能力的全部 Topic；"
            "不要指定主次，也不要规划查询。",
            "当证据不足以在候选 Topic 间消歧时返回 AMBIGUOUS，并仅列出目录中的候选。",
        ]

    @staticmethod
    def _selection_tool(topic_names: List[str]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "select_relevant_topics",
                "description": "只返回需要检索的已发布 Topic 集合",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["RESOLVED", "AMBIGUOUS", "UNSUPPORTED"],
                        },
                        "relevantTopics": {
                            "type": "array",
                            "items": {"type": "string", "enum": topic_names},
                            "uniqueItems": True,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "ambiguityReason": {"type": "string", "maxLength": 300},
                    },
                    "required": ["status", "relevantTopics", "confidence"],
                    "additionalProperties": False,
                },
            },
        }

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        topic_names: List[str],
    ) -> tuple[Optional[Dict[str, Any]], str]:
        status = str(payload.get("status") or "").strip().upper()
        if status not in self.STATUSES:
            return None, "status 必须是 RESOLVED、AMBIGUOUS 或 UNSUPPORTED"
        raw_topics = payload.get("relevantTopics")
        if not isinstance(raw_topics, list):
            return None, "relevantTopics 必须是数组"
        selected = dedupe_ordered([str(item or "").strip() for item in raw_topics if str(item or "").strip()])
        invalid = [item for item in selected if item not in topic_names]
        if invalid:
            return None, "返回了未发布 Topic：%s" % "、".join(invalid)
        if status in {"RESOLVED", "AMBIGUOUS"} and not selected:
            return None, "%s 必须至少返回一个 relevantTopics" % status
        if status == "UNSUPPORTED" and selected:
            return None, "UNSUPPORTED 不应返回 relevantTopics"
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            return None, "confidence 必须是 0 到 1 的数字"
        return (
            {
                "status": status,
                "relevantTopics": selected,
                "confidence": confidence,
                "ambiguityReason": str(payload.get("ambiguityReason") or "")[:300],
            },
            "",
        )

    def _decision_from_payload(
        self,
        payload: Dict[str, Any],
        all_topic_names: List[str],
        *,
        asset_baseline: Dict[str, Any],
        attempts: int,
    ) -> TopicRoutingDecision:
        model_selected_names = list(payload.get("relevantTopics") or [])
        protected_names = list(
            asset_baseline.get("protectedTopicNames") or []
        )
        selected_names = dedupe_ordered(
            [*model_selected_names, *protected_names]
        )
        categories = [
            resolve_asset_topic_category(self.topic_assets, item)
            for item in selected_names
        ]
        categories = dedupe_topics(
            [item for item in categories if item != QuestionCategory.UNKNOWN]
        )
        status = str(payload.get("status") or "")
        unsupported = status == "UNSUPPORTED" and not protected_names
        if status == "UNSUPPORTED" and protected_names:
            status = "RESOLVED"
        reason = str(payload.get("ambiguityReason") or "").strip()
        if unsupported:
            reason = reason or "已发布 Topic 目录不支持该问题"
        elif protected_names and not model_selected_names:
            reason = "Topic 模型未保留精确指标资产，系统已恢复受保护的正式 Topic"
        elif status == "AMBIGUOUS":
            reason = reason or "Topic LLM 无法唯一收敛，保留所有合理 Topic 作为检索范围"
        else:
            reason = "Topic LLM 已选择相关语义检索范围；未指定主次或执行计划"
        return TopicRoutingDecision(
            primary_topic=QuestionCategory.UNKNOWN,
            candidate_topics=categories,
            dimension_topics=[],
            confidence=float(payload.get("confidence") or 0.0),
            clarification_required=unsupported,
            routing_mode=(
                "semantic_topic_unsupported"
                if unsupported
                else "semantic_topic_ambiguous"
                if status == "AMBIGUOUS"
                else "semantic_topic_scope"
            ),
            workspace_topics=categories,
            selection_mode="semantic_llm",
            selection_evidence={
                "router": "semantic_topic_llm",
                "status": status.lower(),
                "modelSelectedTopics": model_selected_names,
                "candidateTopicNames": list(
                    asset_baseline.get("candidateTopicNames") or []
                ),
                "protectedTopicNames": protected_names,
                "suppressedTopicNames": list(
                    asset_baseline.get("suppressedTopicNames") or []
                ),
                "assetSignalTopicNames": list(
                    asset_baseline.get("assetSignalTopicNames") or []
                ),
                "assetSignalBaselineUsed": True,
                "publishedTopicCount": len(all_topic_names),
                "llmAttempts": attempts,
                "keywordRoutingUsed": bool(
                    asset_baseline.get("assetSignalTopicNames")
                    or protected_names
                ),
            },
            reason=reason,
        )

    def _open_directory_decision(
        self,
        topic_names: List[str],
        *,
        status: str,
        reason: str,
        detail: str = "",
    ) -> TopicRoutingDecision:
        categories = dedupe_topics(
            [
                resolve_asset_topic_category(self.topic_assets, item)
                for item in topic_names
            ]
        )
        categories = [item for item in categories if item != QuestionCategory.UNKNOWN]
        return TopicRoutingDecision(
            primary_topic=QuestionCategory.UNKNOWN,
            candidate_topics=categories,
            dimension_topics=[],
            confidence=0.0,
            clarification_required=False,
            routing_mode="semantic_topic_open_directory",
            workspace_topics=categories,
            selection_mode="semantic_llm_degraded",
            selection_evidence={
                "router": "semantic_topic_llm",
                "status": status,
                "publishedTopicCount": len(topic_names),
                "modelSelectedTopics": [],
                "keywordRoutingUsed": False,
                "detail": str(detail or "")[:300],
            },
            reason=reason,
        )

    def _llm_available(self) -> bool:
        return bool(
            self.llm
            and getattr(self.llm, "configured", False)
            and (hasattr(self.llm, "tool_json_chat") or hasattr(self.llm, "json_chat"))
        )

    def _timeout_seconds(self, runtime_budget: Any = None) -> float:
        configured = max(
            1,
            int(getattr(self.settings, "topic_semantic_route_timeout_seconds", 12) or 12),
        )
        if runtime_budget is None:
            return float(configured)
        return float(
            runtime_budget.clamp_timeout_seconds(
                configured,
                minimum_seconds=1,
                operation="topic_route_timeout",
            )
        )


class TopicRouterService:
    def __init__(self, topic_assets: Any = None):
        self.topic_assets = topic_assets or default_topic_assets()

    def route(
        self,
        question: str,
        keywords: ExtractedKeywords,
        context_topic: str = "",
        route_slots: Optional[RouteSlots] = None,
        context_topics: Optional[List[QuestionCategory]] = None,
        context_locked: bool = False,
    ) -> TopicRoutingDecision:
        selection_evidence = self.selection_evidence(question, keywords)
        inherited_topics = dedupe_topics(list(context_topics or []))
        if not inherited_topics and context_topic:
            for item in split_on_characters(context_topic, "、,，|/"):
                category = resolve_asset_topic_category(self.topic_assets, item.strip())
                if category and category not in inherited_topics:
                    inherited_topics.append(category)
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
        if selection_evidence.get("queryShape") == "detail_or_breakdown":
            # The published detailMetricRef is formal lineage, not a lexical
            # guess.  Promote its business owner into the seed workspace and
            # avoid retaining an aggregate serving Topic when its only signal
            # came from the metric mention that was just redirected to detail.
            lineage_score = max([float(value or 0.0) for value in scores.values()] or [1.0])
            business_categories: List[QuestionCategory] = []
            for topic_name in selection_evidence.get("businessTopics") or []:
                category = resolve_asset_topic_category(self.topic_assets, topic_name)
                if category == QuestionCategory.UNKNOWN:
                    continue
                business_categories.append(category)
                scores[category] = max(scores.get(category, 0.0), lineage_score)
            for topic_name in selection_evidence.get("servingTopics") or []:
                category = resolve_asset_topic_category(self.topic_assets, topic_name)
                if category in business_categories:
                    continue
                non_metric_evidence = any(
                    item.kind != "metric" and item.topic == category
                    for item in list(getattr(keywords, "mentions", None) or [])
                )
                if not non_metric_evidence:
                    scores.pop(category, None)
        candidates = self._explicit_topics(scores)
        if inherited_topics and (context_locked or not candidates):
            # Merchant users do not pre-select an internal Topic.  The previous
            # inferred workspace is therefore only a continuation prior.  A
            # genuinely user-locked scope is the sole case where it remains a
            # hard boundary in the face of new per-turn signals.
            outside_topics = [item for item in candidates if item not in inherited_topics]
            primary = route_primary_topic(inherited_topics)
            return TopicRoutingDecision(
                primary_topic=primary,
                candidate_topics=inherited_topics,
                confidence=0.82 if context_locked else 0.62,
                routing_mode="explicit_topic_scope" if context_locked else "inferred_continuation",
                selection_mode="automatic",
                selection_evidence=selection_evidence,
                reason=(
                    "继承用户明确锁定的 Topic 边界；问题中的其他 Topic 信号不会静默扩域：%s"
                    % ",".join(str(item.value) for item in outside_topics)
                    if outside_topics
                    else (
                        "继承用户明确锁定的 Topic 边界"
                        if context_locked
                        else "当前问题没有新的 Topic 证据，将上一轮推断范围作为弱延续先验"
                    )
                ),
            )
        if not candidates:
            return TopicRoutingDecision(
                primary_topic=QuestionCategory.UNKNOWN,
                clarification_required=False,
                routing_mode="open_discovery",
                selection_mode="automatic_discovery",
                selection_evidence=selection_evidence,
                reason="未解析出资产 Topic；保留开放范围进入全局语义召回，不猜测业务分类",
            )
        top_score = max(scores.get(category, 0) for category in candidates)
        confidence = min(0.95, 0.45 + 0.08 * len(candidates) + 0.08 * top_score)
        if len(candidates) > 1:
            first = candidates[0]
            second_score = float(scores.get(candidates[1], 0.0) or 0.0)
            first_score = float(scores.get(first, 0.0) or 0.0)
            if first_score >= 2.5 and (
                first_score - second_score >= 2.0
                or first_score >= max(1.0, second_score) * 1.8
            ):
                return TopicRoutingDecision(
                    primary_topic=first,
                    candidate_topics=[first],
                    dimension_topics=[],
                    confidence=confidence,
                    clarification_required=False,
                    routing_mode="seed_topic",
                    selection_mode="automatic",
                    selection_evidence=selection_evidence,
                    reason=(
                        self._selection_reason(selection_evidence)
                        or "最高分 Topic 明显领先，先作为 Seed Topic；其余弱信号只保留在 Topic Index，"
                        "不在初始召回阶段自动扩成多 Topic workspace"
                    ),
                )
        primary_topic = route_primary_topic(candidates)
        return TopicRoutingDecision(
            primary_topic=primary_topic,
            candidate_topics=candidates,
            dimension_topics=[] if primary_topic == QuestionCategory.UNKNOWN else candidates[1:],
            confidence=confidence,
            clarification_required=False,
            selection_mode="automatic",
            selection_evidence=selection_evidence,
            reason=(
                self._selection_reason(selection_evidence)
                or "按显式业务词选择候选 topic；多 topic 时 primaryTopic 保持 UNKNOWN，"
                "不表示 anchor，避免把召回范围误当主 anchor"
                if primary_topic == QuestionCategory.UNKNOWN
                else self._selection_reason(selection_evidence)
                or "按显式业务词选择 topic；primaryTopic 仅兼容字段，不表示 anchor"
            ),
        )

    def _explicit_topics(self, scores: Dict[QuestionCategory, float]) -> List[QuestionCategory]:
        return [category for category in topic_domain_order(scores) if scores.get(category, 0) > 0]

    def selection_evidence(
        self,
        question: str,
        keywords: Optional[ExtractedKeywords],
    ) -> Dict[str, Any]:
        """Build an auditable Topic decision without exposing model chain-of-thought.

        ``servingTopics`` describe where a governed metric can be queried now;
        ``businessTopics`` describe the metric's declared detail/business owner.
        Keeping the two separate prevents a profile serving table from silently
        redefining the user's business intent.
        """

        normalized = normalize_keyword_text(question or "")
        detail_requested = bool(
            (keywords and (keywords.dimension_keywords or keywords.ranking_keywords))
            or contains_any_literal(normalized, _ROUTING_LANGUAGE.detail_markers)
            or contains_any_literal(normalized, _ROUTING_LANGUAGE.causal_markers)
        )
        shape = "detail_or_breakdown" if detail_requested else "summary_or_total"
        metric_evidence: List[Dict[str, Any]] = []
        serving_topics: List[str] = []
        business_topics: List[str] = []
        serving_tables: List[str] = []
        seen: Set[tuple[str, str, str]] = set()

        for mention in list(getattr(keywords, "mentions", None) or []):
            if mention.kind != "metric" or not mention.canonical_key or not mention.owner_table:
                continue
            topic_names = self.topic_assets.topic_names_for_categories([mention.topic])
            serving_topic = str((topic_names or [keyword_topic_value(mention.topic)])[0] or "")
            identity = (serving_topic, str(mention.owner_table), str(mention.canonical_key))
            if identity in seen:
                continue
            seen.add(identity)
            metric = {}
            try:
                metric = next(
                    (
                        item
                        for item in self.topic_assets.load_table_metrics(serving_topic, mention.owner_table)
                        if str(item.get("canonicalMetricKey") or item.get("metricKey") or "")
                        == str(mention.canonical_key)
                    ),
                    {},
                )
            except Exception:
                metric = {}
            detail_ref = str(
                metric.get("detailMetricRef")
                or metric.get("drilldownMetricRef")
                or ""
            ).strip()
            detail_identity = semantic_metric_reference_identity(detail_ref)
            declared_business_topics: List[str] = []
            for value in [metric.get("ownerTopic"), detail_identity[0] if detail_identity else ""]:
                topic = str(value or "").strip()
                if topic and topic not in declared_business_topics:
                    declared_business_topics.append(topic)
                if topic and topic not in business_topics:
                    business_topics.append(topic)
            if not declared_business_topics and serving_topic:
                declared_business_topics.append(serving_topic)
                if serving_topic not in business_topics:
                    business_topics.append(serving_topic)
            if serving_topic and serving_topic not in serving_topics:
                serving_topics.append(serving_topic)
            if mention.owner_table not in serving_tables:
                serving_tables.append(mention.owner_table)
            metric_evidence.append(
                {
                    "phrase": mention.phrase,
                    "metricKey": mention.canonical_key,
                    "servingTopic": serving_topic,
                    "servingTable": mention.owner_table,
                    "businessTopics": declared_business_topics,
                    "detailMetricRef": detail_ref,
                    "metricIntent": str(metric.get("metricIntent") or ""),
                    "metricGrain": str(metric.get("metricGrain") or ""),
                    "applicableTimeGrain": str(metric.get("applicableTimeGrain") or ""),
                    "aggregationPolicy": str(metric.get("aggregationPolicy") or ""),
                }
            )

        same_table_summary = bool(
            shape == "summary_or_total"
            and metric_evidence
            and len(serving_topics) == 1
            and len(serving_tables) == 1
        )
        return {
            "selectionMode": "automatic",
            "queryShape": shape,
            "servingTopics": serving_topics,
            "businessTopics": business_topics,
            "servingTables": serving_tables,
            "sameTableSummaryCandidate": same_table_summary,
            "matchedMetrics": metric_evidence[:12],
            "policy": (
                "summary queries may seed a governed aggregate workspace; "
                "breakdown/ranking/detail queries prefer declared detail lineage"
            ),
        }

    @staticmethod
    def _selection_reason(evidence: Dict[str, Any]) -> str:
        metrics = list(evidence.get("matchedMetrics") or [])
        if not metrics:
            return ""
        business_topics = [str(item) for item in evidence.get("businessTopics") or [] if str(item)]
        serving_topics = [str(item) for item in evidence.get("servingTopics") or [] if str(item)]
        tables = [str(item) for item in evidence.get("servingTables") or [] if str(item)]
        if evidence.get("sameTableSummaryCandidate") and serving_topics and tables:
            return (
                "自动 Topic 选择：问题是无明细维度的汇总查询，匹配指标可由同一张已治理汇总表 "
                f"{tables[0]} 提供，因此以 {serving_topics[0]} 作为取数 Seed；"
                f"指标业务归属仍保留为 {('、'.join(business_topics) or serving_topics[0])}，"
                "如后续要求排行、拆分或明细则沿 detailMetricRef 切换工作区"
            )
        if evidence.get("queryShape") == "detail_or_breakdown" and business_topics:
            return (
                "自动 Topic 选择：问题包含明细、维度或排行要求，优先按指标声明的 detailMetricRef "
                f"进入业务工作区 {('、'.join(business_topics))}，而不是停留在汇总画像表"
            )
        return "自动 Topic 选择：根据已发布指标、表能力、查询粒度和明细 lineage 选择候选工作区"


def route_primary_topic(candidates: List[QuestionCategory]) -> QuestionCategory:
    """Only a single-topic route can safely expose a compatibility primary topic."""
    return candidates[0] if len(candidates) == 1 else QuestionCategory.UNKNOWN


def extract_days(question: str, default: int = 7) -> int:
    resolved = resolve_time_range(question, default_days=default)
    if int(default or 0) <= 0 and str(getattr(resolved, "source", "") or "") == "default_days":
        return 0
    return max(1, int(resolved.days or default or 1))
