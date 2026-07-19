from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class LanguagePolicyError(RuntimeError):
    """Raised when deterministic language parsing lacks a governed contract."""


@dataclass(frozen=True)
class TemporalLanguagePolicy:
    prefixes: tuple[str, ...]
    units: Mapping[str, str]
    named_windows: Mapping[str, str]
    named_window_semantics: Mapping[str, str]
    comparison_named_semantics: frozenset[str]
    comparison_markers: tuple[str, ...]
    year_over_year_markers: tuple[str, ...]
    period_over_period_markers: tuple[str, ...]
    previous_period_markers: tuple[str, ...]
    natural_month_markers: tuple[str, ...]
    rolling_month_prefixes: tuple[str, ...]
    labels: Mapping[str, str]
    daily_grain_markers: tuple[str, ...]
    time_series_markers: tuple[str, ...]
    coordinated_separator_characters: frozenset[str]
    comparison_separator_characters: frozenset[str]


@dataclass(frozen=True)
class RoutingLanguagePolicy:
    action_markers: tuple[str, ...]
    ranking_ordinal_prefixes: tuple[str, ...]
    ranking_top_prefixes: tuple[str, ...]
    ranking_operators: Mapping[str, str]
    ranking_default_order: str
    ranking_link_particles: tuple[str, ...]
    grouping_prefixes: tuple[str, ...]
    detail_markers: tuple[str, ...]
    causal_markers: tuple[str, ...]
    negation_prefixes: tuple[str, ...]
    greeting_phrases: tuple[str, ...]
    greeting_suffix_characters: frozenset[str]
    assistant_chat_phrases: tuple[str, ...]
    write_markers: tuple[str, ...]
    risk_markers: tuple[str, ...]
    follow_up_prefixes: tuple[str, ...]
    follow_up_phrases: tuple[str, ...]
    scope_lock_markers: tuple[str, ...]
    correction_markers: tuple[str, ...]
    entity_assignment_operators: tuple[str, ...]
    time_clarification_options: tuple[str, ...]
    memory_authoring_markers: tuple[str, ...]


@dataclass(frozen=True)
class AnswerLanguagePolicy:
    definition_markers: tuple[str, ...]
    diagnosis_markers: tuple[str, ...]
    formula_explanation_headings: tuple[str, ...]
    negated_assertion_markers: tuple[str, ...]
    missing_evidence_prefixes: tuple[str, ...]
    evidence_noun: str
    rule_evidence_headings: tuple[str, ...]
    trend_direction_markers: Mapping[str, tuple[str, ...]]
    previous_day_comparison_markers: tuple[str, ...]
    threshold_markers: tuple[str, ...]
    principal_markers: tuple[str, ...]
    ranking_counter_characters: frozenset[str]
    position_column_labels: tuple[str, ...]
    direct_value_exclusion_markers: tuple[str, ...]
    direct_value_markers: tuple[str, ...]


@dataclass(frozen=True)
class LanguagePolicy:
    policy_id: str
    revision: str
    temporal: TemporalLanguagePolicy
    routing: RoutingLanguagePolicy
    answer: AnswerLanguagePolicy


def packaged_language_policy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "policies" / "language_policy.json"


@lru_cache(maxsize=4)
def load_language_policy(path: str = "") -> LanguagePolicy:
    source = Path(path).expanduser().resolve() if str(path or "").strip() else packaged_language_policy_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LanguagePolicyError("language policy is unavailable or invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise LanguagePolicyError("language policy root or schemaVersion is invalid")
    temporal = _mapping(payload.get("temporal"), "temporal")
    routing = _mapping(payload.get("routing"), "routing")
    answer = _mapping(payload.get("answer"), "answer")
    temporal_units = _text_mapping(temporal.get("units"), "temporal.units")
    named_windows = _text_mapping(temporal.get("namedWindows"), "temporal.namedWindows")
    named_window_semantics = _text_mapping(
        temporal.get("namedWindowSemantics"),
        "temporal.namedWindowSemantics",
    )
    ranking_operators = _text_mapping(routing.get("rankingOperators"), "routing.rankingOperators")
    if any(value not in {"day", "week", "month"} for value in temporal_units.values()):
        raise LanguagePolicyError("temporal.units contains an unsupported canonical unit")
    if any(value not in {"day", "week", "month"} for value in named_windows.values()):
        raise LanguagePolicyError("temporal.namedWindows contains an unsupported canonical unit")
    if set(named_windows) != set(named_window_semantics):
        raise LanguagePolicyError("temporal named window units and semantics must declare identical phrases")
    comparison_named_semantics = frozenset(
        _text_tuple(
            temporal.get("comparisonNamedSemantics"),
            "temporal.comparisonNamedSemantics",
        )
    )
    if not comparison_named_semantics <= set(named_window_semantics.values()):
        raise LanguagePolicyError("temporal.comparisonNamedSemantics references an undeclared semantic")
    temporal_labels = _text_mapping(temporal.get("labels"), "temporal.labels")
    required_temporal_labels = {
        "default_rolling_days",
        "year_over_year",
        "previous_month",
        "previous_week",
        "previous_day",
        "previous_days",
    }
    if not required_temporal_labels <= set(temporal_labels):
        raise LanguagePolicyError("temporal.labels is missing a required canonical label")
    if any(value not in {"asc", "desc"} for value in ranking_operators.values()):
        raise LanguagePolicyError("routing.rankingOperators contains an unsupported order")
    ranking_default_order = _required_text(
        routing.get("rankingDefaultOrder"),
        "routing.rankingDefaultOrder",
    )
    if ranking_default_order not in {"asc", "desc"}:
        raise LanguagePolicyError("routing.rankingDefaultOrder is unsupported")
    trend_direction_markers = _text_tuple_mapping(
        answer.get("trendDirectionMarkers"),
        "answer.trendDirectionMarkers",
    )
    if set(trend_direction_markers) != {"up", "down", "flat"}:
        raise LanguagePolicyError("answer.trendDirectionMarkers must declare up, down, and flat")
    return LanguagePolicy(
        policy_id=_required_text(payload.get("policyId"), "policyId"),
        revision=_required_text(payload.get("revision"), "revision"),
        temporal=TemporalLanguagePolicy(
            prefixes=_text_tuple(temporal.get("prefixes"), "temporal.prefixes"),
            units=temporal_units,
            named_windows=named_windows,
            named_window_semantics=named_window_semantics,
            comparison_named_semantics=comparison_named_semantics,
            comparison_markers=_text_tuple(temporal.get("comparisonMarkers"), "temporal.comparisonMarkers"),
            year_over_year_markers=_text_tuple(
                temporal.get("yearOverYearMarkers"),
                "temporal.yearOverYearMarkers",
            ),
            period_over_period_markers=_text_tuple(
                temporal.get("periodOverPeriodMarkers"),
                "temporal.periodOverPeriodMarkers",
            ),
            previous_period_markers=_text_tuple(
                temporal.get("previousPeriodMarkers"),
                "temporal.previousPeriodMarkers",
            ),
            natural_month_markers=_text_tuple(
                temporal.get("naturalMonthMarkers"),
                "temporal.naturalMonthMarkers",
            ),
            rolling_month_prefixes=_text_tuple(
                temporal.get("rollingMonthPrefixes"),
                "temporal.rollingMonthPrefixes",
            ),
            labels=temporal_labels,
            daily_grain_markers=_text_tuple(temporal.get("dailyGrainMarkers"), "temporal.dailyGrainMarkers"),
            time_series_markers=_text_tuple(temporal.get("timeSeriesMarkers"), "temporal.timeSeriesMarkers"),
            coordinated_separator_characters=frozenset(
                _required_text(
                    temporal.get("coordinatedSeparatorCharacters"),
                    "temporal.coordinatedSeparatorCharacters",
                )
            ),
            comparison_separator_characters=frozenset(
                _required_text(
                    temporal.get("comparisonSeparatorCharacters"),
                    "temporal.comparisonSeparatorCharacters",
                )
            ),
        ),
        routing=RoutingLanguagePolicy(
            action_markers=_text_tuple(routing.get("actionMarkers"), "routing.actionMarkers"),
            ranking_ordinal_prefixes=_text_tuple(
                routing.get("rankingOrdinalPrefixes"),
                "routing.rankingOrdinalPrefixes",
            ),
            ranking_top_prefixes=_text_tuple(
                routing.get("rankingTopPrefixes"),
                "routing.rankingTopPrefixes",
            ),
            ranking_operators=ranking_operators,
            ranking_default_order=ranking_default_order,
            ranking_link_particles=_text_tuple(
                routing.get("rankingLinkParticles"),
                "routing.rankingLinkParticles",
            ),
            grouping_prefixes=_text_tuple(routing.get("groupingPrefixes"), "routing.groupingPrefixes"),
            detail_markers=_text_tuple(routing.get("detailMarkers"), "routing.detailMarkers"),
            causal_markers=_text_tuple(routing.get("causalMarkers"), "routing.causalMarkers"),
            negation_prefixes=_text_tuple(routing.get("negationPrefixes"), "routing.negationPrefixes"),
            greeting_phrases=_text_tuple(routing.get("greetingPhrases"), "routing.greetingPhrases"),
            greeting_suffix_characters=frozenset(
                _required_text(routing.get("greetingSuffixCharacters"), "routing.greetingSuffixCharacters")
            ),
            assistant_chat_phrases=_text_tuple(
                routing.get("assistantChatPhrases"),
                "routing.assistantChatPhrases",
            ),
            write_markers=_text_tuple(routing.get("writeMarkers"), "routing.writeMarkers"),
            risk_markers=_text_tuple(routing.get("riskMarkers"), "routing.riskMarkers"),
            follow_up_prefixes=_text_tuple(routing.get("followUpPrefixes"), "routing.followUpPrefixes"),
            follow_up_phrases=_text_tuple(routing.get("followUpPhrases"), "routing.followUpPhrases"),
            scope_lock_markers=_text_tuple(routing.get("scopeLockMarkers"), "routing.scopeLockMarkers"),
            correction_markers=_text_tuple(routing.get("correctionMarkers"), "routing.correctionMarkers"),
            entity_assignment_operators=_text_tuple(
                routing.get("entityAssignmentOperators"),
                "routing.entityAssignmentOperators",
            ),
            time_clarification_options=_text_tuple(
                routing.get("timeClarificationOptions"),
                "routing.timeClarificationOptions",
            ),
            memory_authoring_markers=_text_tuple(
                routing.get("memoryAuthoringMarkers"),
                "routing.memoryAuthoringMarkers",
            ),
        ),
        answer=AnswerLanguagePolicy(
            definition_markers=_text_tuple(answer.get("definitionMarkers"), "answer.definitionMarkers"),
            diagnosis_markers=_text_tuple(answer.get("diagnosisMarkers"), "answer.diagnosisMarkers"),
            formula_explanation_headings=_text_tuple(
                answer.get("formulaExplanationHeadings"),
                "answer.formulaExplanationHeadings",
            ),
            negated_assertion_markers=_text_tuple(
                answer.get("negatedAssertionMarkers"),
                "answer.negatedAssertionMarkers",
            ),
            missing_evidence_prefixes=_text_tuple(
                answer.get("missingEvidencePrefixes"),
                "answer.missingEvidencePrefixes",
            ),
            evidence_noun=_required_text(answer.get("evidenceNoun"), "answer.evidenceNoun"),
            rule_evidence_headings=_text_tuple(
                answer.get("ruleEvidenceHeadings"),
                "answer.ruleEvidenceHeadings",
            ),
            trend_direction_markers=trend_direction_markers,
            previous_day_comparison_markers=_text_tuple(
                answer.get("previousDayComparisonMarkers"),
                "answer.previousDayComparisonMarkers",
            ),
            threshold_markers=_text_tuple(answer.get("thresholdMarkers"), "answer.thresholdMarkers"),
            principal_markers=_text_tuple(answer.get("principalMarkers"), "answer.principalMarkers"),
            ranking_counter_characters=frozenset(
                _required_text(
                    answer.get("rankingCounterCharacters"),
                    "answer.rankingCounterCharacters",
                )
            ),
            position_column_labels=_text_tuple(
                answer.get("positionColumnLabels"),
                "answer.positionColumnLabels",
            ),
            direct_value_exclusion_markers=_text_tuple(
                answer.get("directValueExclusionMarkers"),
                "answer.directValueExclusionMarkers",
            ),
            direct_value_markers=_text_tuple(
                answer.get("directValueMarkers"),
                "answer.directValueMarkers",
            ),
        ),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not value:
        raise LanguagePolicyError("language policy %s must be a non-empty object" % label)
    return value


def _required_text(value: Any, label: str) -> str:
    text = str(value or "")
    if not text:
        raise LanguagePolicyError("language policy %s must be a non-empty string" % label)
    return text


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise LanguagePolicyError("language policy %s must be a non-empty array" % label)
    result = tuple(str(item or "").strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise LanguagePolicyError("language policy %s contains empty or duplicate values" % label)
    return result


def _text_mapping(value: Any, label: str) -> Mapping[str, str]:
    mapping = _mapping(value, label)
    result = {
        str(key or "").strip(): str(item or "").strip()
        for key, item in mapping.items()
    }
    if any(not key or not item for key, item in result.items()):
        raise LanguagePolicyError("language policy %s contains empty keys or values" % label)
    return result


def _text_tuple_mapping(value: Any, label: str) -> Mapping[str, tuple[str, ...]]:
    mapping = _mapping(value, label)
    return {
        str(key or "").strip(): _text_tuple(item, "%s.%s" % (label, key))
        for key, item in mapping.items()
        if str(key or "").strip()
    }
