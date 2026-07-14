from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field

from merchant_ai.config import Settings
from merchant_ai.models import APIModel, AnswerMode, FastUnderstandingResult, QueryPlan


class CapabilityFeatures(APIModel):
    intent_kind: str = "unknown"
    complexity: str = "unknown"
    analysis_intent: str = "lookup"
    metric_count: int = 0
    domain_count: int = 0
    node_count: int = 0
    dependency_count: int = 0
    requires_explanation: bool = False
    needs_planner: bool = True
    confidence: float = 0.0
    published_metric_count: int = 0
    answer_modes: List[str] = Field(default_factory=list)


class CapabilityContract(APIModel):
    capability_id: str
    version: str = "1.0.0"
    enabled: bool = True
    intent_kinds: List[str] = Field(default_factory=list)
    complexities: List[str] = Field(default_factory=list)
    analysis_intents: List[str] = Field(default_factory=list)
    answer_modes: List[str] = Field(default_factory=list)
    min_metrics: int = 0
    min_domains: int = 0
    max_metrics: int = 0
    max_domains: int = 0
    max_nodes: int = 0
    max_dependencies: int = 0
    allow_explanation: bool = False
    allow_planner_required: bool = False
    require_published_metrics: bool = False
    min_confidence: float = 0.0
    max_llm_calls: int = 0
    target_p95_ms: int = 0
    risk_level: str = "low"

    def evaluate(self, features: CapabilityFeatures) -> "CapabilityDecision":
        reasons: List[str] = []
        if not self.enabled:
            reasons.append("capability_disabled")
        if self.intent_kinds and features.intent_kind not in self.intent_kinds:
            reasons.append("unsupported_intent_kind:%s" % features.intent_kind)
        if self.complexities and features.complexity not in self.complexities:
            reasons.append("unsupported_complexity:%s" % features.complexity)
        if self.analysis_intents and features.analysis_intent not in self.analysis_intents:
            reasons.append("unsupported_analysis_intent:%s" % features.analysis_intent)
        unsupported_modes = sorted(set(features.answer_modes) - set(self.answer_modes)) if self.answer_modes else []
        if unsupported_modes:
            reasons.append("unsupported_answer_modes:%s" % ",".join(unsupported_modes))
        if features.metric_count < self.min_metrics:
            reasons.append("metrics_below_minimum:%s<%s" % (features.metric_count, self.min_metrics))
        if features.domain_count < self.min_domains:
            reasons.append("domains_below_minimum:%s<%s" % (features.domain_count, self.min_domains))
        for name, value, maximum in [
            ("metrics", features.metric_count, self.max_metrics),
            ("domains", features.domain_count, self.max_domains),
            ("nodes", features.node_count, self.max_nodes),
            ("dependencies", features.dependency_count, self.max_dependencies),
        ]:
            if maximum > 0 and value > maximum:
                reasons.append("%s_over_limit:%s>%s" % (name, value, maximum))
        if features.requires_explanation and not self.allow_explanation:
            reasons.append("explanation_not_supported")
        if features.needs_planner and not self.allow_planner_required:
            reasons.append("planner_required")
        if self.require_published_metrics and features.published_metric_count < max(1, features.metric_count):
            reasons.append("published_metric_contract_missing")
        if features.confidence and features.confidence < self.min_confidence:
            reasons.append("confidence_below_threshold")
        return CapabilityDecision(
            capability_id=self.capability_id,
            capability_version=self.version,
            eligible=not reasons,
            reasons=reasons,
            max_llm_calls=self.max_llm_calls,
            target_p95_ms=self.target_p95_ms,
            risk_level=self.risk_level,
            features=features,
        )


class CapabilityDecision(APIModel):
    capability_id: str = ""
    capability_version: str = ""
    eligible: bool = False
    reasons: List[str] = Field(default_factory=list)
    max_llm_calls: int = 0
    target_p95_ms: int = 0
    risk_level: str = ""
    features: CapabilityFeatures = Field(default_factory=CapabilityFeatures)


class CapabilityRegistry:
    def __init__(self, version: str, contracts: Iterable[CapabilityContract], source: str = "defaults"):
        self.version = version
        self.source = source
        self._contracts = {contract.capability_id: contract for contract in contracts}

    @classmethod
    def from_settings(cls, settings: Optional[Settings]) -> "CapabilityRegistry":
        path = settings.resources_root / "runtime" / "capabilities.json" if settings else None
        return cls.load(path)

    @classmethod
    def load(cls, path: Optional[Path]) -> "CapabilityRegistry":
        if path and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                contracts = [CapabilityContract.model_validate(item) for item in payload.get("capabilities") or []]
                if contracts:
                    return cls(str(payload.get("version") or "1.0.0"), contracts, source=str(path))
            except (OSError, ValueError, TypeError):
                pass
        return cls("1.0.0", default_capability_contracts())

    def contract(self, capability_id: str) -> Optional[CapabilityContract]:
        return self._contracts.get(str(capability_id or ""))

    def evaluate(self, capability_id: str, features: CapabilityFeatures) -> CapabilityDecision:
        contract = self.contract(capability_id)
        if not contract:
            return CapabilityDecision(
                capability_id=capability_id,
                capability_version=self.version,
                eligible=False,
                reasons=["capability_not_registered"],
                features=features,
            )
        return contract.evaluate(features)

    def catalog(self) -> List[Dict[str, Any]]:
        return [contract.model_dump(by_alias=True) for contract in self._contracts.values()]


def features_from_fast_understanding(fast: FastUnderstandingResult | None) -> CapabilityFeatures:
    if not fast:
        return CapabilityFeatures()
    return CapabilityFeatures(
        intent_kind=str(fast.intent_kind or "unknown"),
        complexity=str(fast.complexity or "unknown"),
        analysis_intent=str(fast.analysis_intent or "lookup"),
        metric_count=len(set(str(item) for item in fast.metric_phrases if str(item))),
        domain_count=len(set(str(item) for item in fast.topics if str(item))),
        node_count=1,
        dependency_count=0,
        requires_explanation=str(fast.analysis_intent or "lookup") not in {"", "lookup", "metric"},
        needs_planner=bool(fast.needs_planner),
        confidence=float(fast.confidence or 0.0),
        answer_modes=[AnswerMode.METRIC.value],
    )


def features_from_query_plan(plan: QueryPlan) -> CapabilityFeatures:
    understanding = plan.question_understanding or {}
    ranking = understanding.get("rankingObjective") or understanding.get("ranking_objective") or {}
    requested = understanding.get("requestedMeasures") or understanding.get("requested_measures") or []
    metric_refs = {
        str(item)
        for item in [
            ranking.get("metricRef") if isinstance(ranking, dict) else "",
            *[
                measure.get("metricRef") or measure.get("metric_ref")
                for measure in requested
                if isinstance(measure, dict)
            ],
        ]
        if str(item or "")
    }
    if not metric_refs:
        metric_refs = {
            str((intent.metric_resolution or {}).get("metricKey") or intent.metric_name or intent.metric_column)
            for intent in plan.intents
            if str((intent.metric_resolution or {}).get("metricKey") or intent.metric_name or intent.metric_column)
        }
    domains = {str(intent.category) for intent in plan.intents if str(intent.category) not in {"", "UNKNOWN"}}
    published = sum(
        1
        for intent in plan.intents
        if (intent.metric_resolution or {}).get("semanticRefId")
        or (intent.metric_resolution or {}).get("semanticContractHash")
        or intent.knowledge_ref_ids
    )
    raw_analysis_intent = str(understanding.get("analysisIntent") or understanding.get("analysis_intent") or "").strip()
    if raw_analysis_intent in {"", "none"}:
        analysis_intent = "ranking" if isinstance(ranking, dict) and ranking else "lookup"
    else:
        analysis_intent = raw_analysis_intent
    requires_explanation = bool(
        understanding.get("requiresExplanation")
        or understanding.get("requires_explanation")
        or analysis_intent not in {"", "none", "lookup", "metric", "ranking"}
    )
    answer_modes = sorted({str(getattr(intent.answer_mode, "value", intent.answer_mode)) for intent in plan.intents})
    if requires_explanation:
        intent_kind = "analysis"
    elif plan.dependencies:
        intent_kind = "multi_hop"
    elif answer_modes and set(answer_modes) <= {AnswerMode.DETAIL.value}:
        intent_kind = "detail_lookup"
    else:
        intent_kind = "metric_query"
    complexity = "complex" if requires_explanation or plan.dependencies else "simple" if len(plan.intents) <= 1 else "medium"
    confidence_values = [
        float((intent.metric_resolution or {}).get("confidence") or 0.0)
        for intent in plan.intents
        if (intent.metric_resolution or {}).get("confidence") is not None
    ]
    return CapabilityFeatures(
        intent_kind=intent_kind,
        complexity=complexity,
        analysis_intent=analysis_intent,
        metric_count=len(metric_refs),
        domain_count=len(domains),
        node_count=len(plan.intents),
        dependency_count=len(plan.dependencies),
        requires_explanation=requires_explanation,
        needs_planner=requires_explanation or bool(plan.dependencies),
        confidence=min(confidence_values) if confidence_values else 0.0,
        published_metric_count=published,
        answer_modes=answer_modes,
    )


def default_capability_contracts() -> List[CapabilityContract]:
    return [
        CapabilityContract(
            capability_id="metric_fast_entry",
            intent_kinds=["metric_query"],
            complexities=["simple"],
            analysis_intents=["lookup", "metric"],
            answer_modes=[AnswerMode.METRIC.value],
            min_metrics=1,
            min_domains=1,
            max_metrics=1,
            max_domains=1,
            max_nodes=1,
            max_dependencies=0,
            allow_explanation=False,
            allow_planner_required=False,
            max_llm_calls=0,
            target_p95_ms=3000,
            risk_level="low",
        ),
        CapabilityContract(
            capability_id="semantic_plan_fast",
            intent_kinds=["metric_query"],
            complexities=["simple"],
            analysis_intents=["lookup", "metric", "ranking"],
            answer_modes=[
                AnswerMode.METRIC.value,
                AnswerMode.TOPN.value,
                AnswerMode.GROUP_AGG.value,
            ],
            min_metrics=1,
            min_domains=1,
            max_metrics=1,
            max_domains=1,
            max_nodes=1,
            max_dependencies=0,
            allow_explanation=False,
            allow_planner_required=False,
            require_published_metrics=True,
            max_llm_calls=0,
            target_p95_ms=5000,
            risk_level="low",
        ),
        CapabilityContract(
            capability_id="semantic_detail_fast",
            intent_kinds=["detail_lookup"],
            complexities=["simple"],
            analysis_intents=["lookup"],
            answer_modes=[AnswerMode.DETAIL.value],
            min_domains=1,
            max_metrics=0,
            max_domains=1,
            max_nodes=1,
            max_dependencies=0,
            allow_explanation=False,
            allow_planner_required=False,
            require_published_metrics=False,
            max_llm_calls=0,
            target_p95_ms=5000,
            risk_level="low",
        ),
        CapabilityContract(
            capability_id="independent_multi_metric_trend",
            intent_kinds=["metric_query", "analysis"],
            complexities=["medium", "complex"],
            analysis_intents=["lookup", "metric", "trend_check", "comparison"],
            answer_modes=[AnswerMode.GROUP_AGG.value],
            min_metrics=2,
            min_domains=1,
            max_metrics=4,
            max_domains=4,
            max_nodes=4,
            max_dependencies=0,
            allow_explanation=True,
            allow_planner_required=True,
            require_published_metrics=True,
            max_llm_calls=0,
            target_p95_ms=6000,
            risk_level="medium",
        ),
        CapabilityContract(
            capability_id="semantic_topn_graph",
            intent_kinds=["multi_hop", "analysis"],
            complexities=["complex"],
            analysis_intents=["lookup", "ranking", "risk_ranking"],
            answer_modes=[AnswerMode.TOPN.value, AnswerMode.GROUP_AGG.value, AnswerMode.DETAIL.value, AnswerMode.DERIVED.value],
            min_metrics=1,
            min_domains=1,
            max_metrics=4,
            max_domains=4,
            max_nodes=8,
            max_dependencies=8,
            allow_explanation=True,
            allow_planner_required=True,
            require_published_metrics=True,
            max_llm_calls=0,
            target_p95_ms=8000,
            risk_level="medium",
        ),
    ]
