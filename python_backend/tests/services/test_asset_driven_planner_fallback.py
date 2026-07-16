from __future__ import annotations

from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack, QueryPlan, RecallBundle, RecallItem
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    compile_asset_driven_multi_metric_fallback_graph,
    planned_metric_identities,
)


def metric(table: str, key: str, label: str, column: str, topic: str = "RUNTIME_DOMAIN") -> PlanningAssetEntry:
    return PlanningAssetEntry(
        key=key,
        table=table,
        topic=topic,
        title=label,
        aliases=[label],
        columns=[column],
        source_ref_id=f"semantic:{topic}:{table}:metric:{key}",
        metadata={
            "businessName": label,
            "formula": f"SUM({column})",
            "sourceColumns": [column],
        },
    )


def table(name: str, time_column: str, columns: list[str], topic: str = "RUNTIME_DOMAIN") -> PlanningAssetEntry:
    return PlanningAssetEntry(
        key=name,
        table=name,
        topic=topic,
        columns=[time_column, *columns],
        source_ref_id=f"semantic:{topic}:{name}:table",
        metadata={"timeColumn": time_column},
    )


def evidence(item: PlanningAssetEntry, phrase: str) -> dict[str, object]:
    return {
        "ownerTable": item.table,
        "metricKey": item.key,
        "semanticRefId": item.source_ref_id,
        "matchedMetricLabel": phrase,
        "metricResolutionType": "exact_alias",
        "metricResolutionConfidence": 0.97,
        "metricResolutionAmbiguous": False,
    }


def test_asset_driven_timeout_fallback_compiles_cross_table_time_series_from_contracts_only() -> None:
    alpha = metric("fact_alpha", "alpha_value", "Alpha value", "alpha_amount")
    beta = metric("fact_alpha", "beta_value", "Beta value", "beta_amount")
    gamma = metric("fact_gamma", "gamma_value", "Gamma value", "gamma_amount", topic="SECOND_DOMAIN")
    irrelevant = [
        metric("fact_noise", f"noise_{index}", f"Noise {index}", f"noise_col_{index}", topic="NOISE_DOMAIN")
        for index in range(80)
    ]
    pack = PlanningAssetPack(
        tables=[
            table("fact_alpha", "event_day", ["alpha_amount", "beta_amount"]),
            table("fact_gamma", "observed_at", ["gamma_amount"], topic="SECOND_DOMAIN"),
            table("fact_noise", "noise_day", [f"noise_col_{index}" for index in range(80)], topic="NOISE_DOMAIN"),
        ],
        metrics=[alpha, beta, gamma, *irrelevant],
        metric_compaction={
            "fastUnderstanding": {
                "intentKind": "analysis",
                "analysisIntent": "anomaly",
                "metricPhrases": ["Alpha value", "Beta value", "Gamma value"],
                "timeWindowDays": 13,
                "timeRange": {
                    "kind": "rolling",
                    "startDate": "2026-01-01",
                    "endDate": "2026-01-13",
                    "days": 13,
                    "explicit": True,
                    "source": "runtime_contract",
                },
            },
            "recalledMetricEvidence": [
                evidence(alpha, "Alpha value"),
                evidence(beta, "Beta value"),
                evidence(gamma, "Gamma value"),
            ],
        },
    )

    plan = compile_asset_driven_multi_metric_fallback_graph("opaque request text", pack)

    assert planned_metric_identities(plan.intents) == {
        ("fact_alpha", "alpha_value"),
        ("fact_alpha", "beta_value"),
        ("fact_gamma", "gamma_value"),
    }
    assert {(intent.preferred_table, intent.group_by_column) for intent in plan.intents} == {
        ("fact_alpha", "event_day"),
        ("fact_gamma", "observed_at"),
    }
    assert all(intent.days == 13 and intent.time_range.days == 13 for intent in plan.intents)
    assert plan.question_understanding["timeRange"]["source"] == "runtime_contract"
    assert "planner=asset_driven_multi_metric_failure_fallback" in plan.agent_trace
    assert plan.question_understanding["rankingObjective"] == {}


def test_asset_driven_fallback_does_not_promote_the_first_unranked_metric() -> None:
    alpha = metric("fact_runtime", "alpha_value", "Alpha value", "alpha_amount")
    beta = metric("fact_runtime", "beta_value", "Beta value", "beta_amount")
    runtime_table = table("fact_runtime", "event_day", ["alpha_amount", "beta_amount"])

    def compile_with_order(metric_phrases: list[str]):
        return compile_asset_driven_multi_metric_fallback_graph(
            "opaque",
            PlanningAssetPack(
                tables=[runtime_table],
                metrics=[alpha, beta],
                metric_compaction={
                    "fastUnderstanding": {
                        "analysisIntent": "trend",
                        "metricPhrases": metric_phrases,
                        "timeWindowDays": 7,
                        "timeRange": {"kind": "rolling", "days": 7},
                    },
                    "recalledMetricEvidence": [
                        evidence(alpha, "Alpha value"),
                        evidence(beta, "Beta value"),
                    ],
                },
            ),
        )

    forward = compile_with_order(["Alpha value", "Beta value"])
    reversed_order = compile_with_order(["Beta value", "Alpha value"])

    assert forward.question_understanding["rankingObjective"] == {}
    assert reversed_order.question_understanding["rankingObjective"] == {}
    assert {
        (item["ownerTable"], item["metricRef"])
        for item in forward.question_understanding["requestedMeasures"]
    } == {
        (item["ownerTable"], item["metricRef"])
        for item in reversed_order.question_understanding["requestedMeasures"]
    }


def test_asset_driven_timeout_fallback_compiles_overview_from_structured_understanding() -> None:
    first = metric("fact_runtime", "first_measure", "First measure", "first_value")
    second = metric("fact_runtime", "second_measure", "Second measure", "second_value")
    pack = PlanningAssetPack(
        tables=[table("fact_runtime", "business_day", ["first_value", "second_value"])],
        metrics=[first, second],
    )
    understanding = {
        "analysisIntent": "overview",
        "timeWindowDays": 9,
        "timeRange": {"kind": "rolling", "days": 9, "source": "prior_understanding"},
        "rankingObjective": {
            "metricRef": first.key,
            "ownerTable": first.table,
            "sourcePhrase": first.title,
            "semanticRefId": first.source_ref_id,
        },
        "requestedMeasures": [
            {
                "metricRef": second.key,
                "ownerTable": second.table,
                "sourcePhrase": second.title,
                "semanticRefId": second.source_ref_id,
            }
        ],
    }

    plan = compile_asset_driven_multi_metric_fallback_graph(
        "text deliberately contains no metric labels or duration",
        pack,
        structured_understanding=understanding,
    )

    assert planned_metric_identities(plan.intents) == {
        ("fact_runtime", "first_measure"),
        ("fact_runtime", "second_measure"),
    }
    assert all(intent.group_by_column == "" for intent in plan.intents)
    assert all(intent.days == 9 and intent.time_range.source == "prior_understanding" for intent in plan.intents)
    assert plan.question_understanding["rankingObjective"]["objectiveType"] == "metric_total"


def test_asset_driven_timeout_fallback_fails_closed_when_one_metric_is_not_unique() -> None:
    known = metric("fact_known", "known_measure", "Known measure", "known_value")
    duplicate_a = metric("fact_a", "shared_a", "Shared measure", "shared_a")
    duplicate_b = metric("fact_b", "shared_b", "Shared measure", "shared_b")
    pack = PlanningAssetPack(
        tables=[
            table("fact_known", "known_day", ["known_value"]),
            table("fact_a", "a_day", ["shared_a"]),
            table("fact_b", "b_day", ["shared_b"]),
        ],
        metrics=[known, duplicate_a, duplicate_b],
        metric_compaction={
            "fastUnderstanding": {
                "analysisIntent": "trend",
                "metricPhrases": ["Known measure", "Shared measure"],
                "timeWindowDays": 5,
                "timeRange": {"kind": "rolling", "days": 5},
            },
            "recalledMetricEvidence": [evidence(known, "Known measure")],
        },
    )

    plan = compile_asset_driven_multi_metric_fallback_graph("opaque", pack)

    assert not plan.intents
    assert "planner.asset_driven_fallback.unresolved_structured_metrics:2" in plan.agent_trace
    assert "ASSET_DRIVEN_FALLBACK_FAIL_CLOSED" in plan.compiler_trace


def test_failure_fallback_does_not_call_model_again_or_render_an_asset_prompt() -> None:
    first = metric("fact_runtime", "first_measure", "First measure", "first_value")
    second = metric("fact_runtime", "second_measure", "Second measure", "second_value")
    pack = PlanningAssetPack(
        tables=[table("fact_runtime", "event_day", ["first_value", "second_value"])],
        metrics=[first, second],
        metric_compaction={
            "fastUnderstanding": {
                "analysisIntent": "trend",
                "metricPhrases": ["First measure", "Second measure"],
                "timeWindowDays": 7,
                "timeRange": {"kind": "rolling", "days": 7},
            },
            "recalledMetricEvidence": [
                evidence(first, "First measure"),
                evidence(second, "Second measure"),
            ],
        },
    )

    class ModelMustNotRun:
        configured = True
        last_error = "timeout: provider call exceeded contract"
        error_events: list[object] = []

        def json_chat(self, *args, **kwargs):
            raise AssertionError("failure fallback must not call or prompt a model")

    planner = QueryGraphPlanner(ModelMustNotRun())
    plan, requests, reason = planner.understanding_extractor.failure_fallback_plan(
        "opaque",
        pack,
        "PLANNER_LLM_TIMEOUT: provider timeout",
    )

    assert plan.intents
    assert not requests
    assert reason == "SEMANTIC_FAST_PATH"
    assert "planner.asset_driven_multi_metric_fallback_after_llm_failure" in plan.agent_trace


def test_configured_planner_calls_llm_selection_path_before_any_deterministic_candidate() -> None:
    first = metric("fact_runtime", "first_measure", "First measure", "first_value")
    second = metric("fact_runtime", "second_measure", "Second measure", "second_value")
    pack = PlanningAssetPack(
        tables=[table("fact_runtime", "event_day", ["first_value", "second_value"])],
        metrics=[first, second],
    )

    class ConfiguredModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

    planner = QueryGraphPlanner(ConfiguredModel())
    selected = compile_asset_driven_multi_metric_fallback_graph(
        "opaque",
        pack,
        structured_understanding={
            "analysisIntent": "overview",
            "timeWindowDays": 7,
            "rankingObjective": {
                "metricRef": first.key,
                "ownerTable": first.table,
                "semanticRefId": first.source_ref_id,
            },
            "requestedMeasures": [
                {
                    "metricRef": second.key,
                    "ownerTable": second.table,
                    "semanticRefId": second.source_ref_id,
                }
            ],
        },
    )
    calls: list[str] = []

    def llm_selection(*args, **kwargs):
        calls.append("semantic_asset_selection_llm")
        return selected, {"status": "SELECTED", "reason": "llm selected published refs"}

    planner._semantic_asset_selection_plan = llm_selection
    planner._semantic_fast_path = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("configured Planner must not enter deterministic fast path first")
    )

    plan, requests, reason = planner.plan("opaque", [], "", RecallBundle(), pack, [], [])

    assert calls == ["semantic_asset_selection_llm"]
    assert plan.intents
    assert not requests
    assert reason == "llm selected published refs"


def test_configured_non_provider_failure_fails_closed_without_deterministic_activation() -> None:
    class InvalidModel:
        configured = True
        last_error = "json_parse_error: invalid planner output"
        error_events: list[object] = []

    planner = QueryGraphPlanner(InvalidModel())
    planner._semantic_asset_selection_plan = lambda *args, **kwargs: (
        QueryPlan(),
        {"status": "INVALID", "queryContract": {"contractType": "requires_planner"}},
    )
    planner.understanding_extractor.initial_payload = lambda *args, **kwargs: ({}, False, None, "")
    planner.understanding_extractor.failure_fallback_plan = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("non-provider failure must not activate deterministic recovery")
    )

    plan, requests, reason = planner.plan(
        "opaque",
        [],
        "",
        RecallBundle(),
        PlanningAssetPack(),
        [],
        [],
    )

    assert not plan.intents
    assert not requests
    assert "PLANNER_JSON_PARSE_ERROR" in reason
    assert "planner.recovery_candidate.not_activated=non_provider_failure" in plan.agent_trace


def test_configured_provider_failure_returns_explicit_validated_recovery_candidate() -> None:
    first = metric("fact_runtime", "first_measure", "First measure", "first_value")
    second = metric("fact_runtime", "second_measure", "Second measure", "second_value")
    pack = PlanningAssetPack(
        tables=[table("fact_runtime", "event_day", ["first_value", "second_value"])],
        metrics=[first, second],
        metric_compaction={
            "fastUnderstanding": {
                "analysisIntent": "trend",
                "metricPhrases": ["First measure", "Second measure"],
                "timeWindowDays": 7,
                "timeRange": {"kind": "rolling", "days": 7},
            },
            "recalledMetricEvidence": [
                evidence(first, "First measure"),
                evidence(second, "Second measure"),
            ],
        },
    )

    class TimeoutModel:
        configured = True
        last_error = "timeout: provider call exceeded contract"
        error_events: list[object] = []

    planner = QueryGraphPlanner(TimeoutModel())
    planner._semantic_asset_selection_plan = lambda *args, **kwargs: (
        QueryPlan(),
        {"status": "INVALID", "queryContract": {"contractType": "requires_planner"}},
    )
    planner.understanding_extractor.initial_payload = lambda *args, **kwargs: ({}, False, None, "")

    plan, requests, reason = planner.plan("opaque", [], "", RecallBundle(), pack, [], [])

    assert plan.intents
    assert not requests
    assert reason == "TYPED_SEMANTIC_RECOVERY_CANDIDATE"
    assert "planner.recovery_candidate.trigger=provider_failure" in plan.agent_trace
    assert plan.question_understanding["recoveryCandidate"]["validated"] is True


def test_semantic_candidate_protocol_separates_exact_and_partial_labels_and_only_publishes_readable_refs() -> None:
    exact = metric("fact_daily", "target_measure", "Target measure", "target_value")
    exact.metadata.update(
        {
            "metricGrain": "entity_day_summary",
            "tableKind": "aggregate_profile",
            "metricIntent": "summary",
        }
    )
    partial = metric(
        "dim_runtime",
        "qualified_target_measure",
        "Qualified target measure",
        "qualified_target_value",
    )
    partial.metadata.update(
        {
            "metricGrain": "entity_dimension",
            "tableKind": "dimension",
            "metricIntent": "dimension_summary",
        }
    )
    pack = PlanningAssetPack(
        tables=[
            table("fact_daily", "event_day", ["target_value"]),
            table("dim_runtime", "snapshot_day", ["qualified_target_value"]),
        ],
        metrics=[exact, partial],
    )
    planner = QueryGraphPlanner(type("NoModel", (), {"configured": False})())

    def card(item: PlanningAssetEntry, candidate_id: str) -> dict[str, object]:
        return planner._semantic_selection_candidate_card(
            RecallItem(
                doc_id=item.source_ref_id,
                title=item.title,
                source_type="SEMANTIC_METRIC",
                topic=item.topic,
                table=item.table,
                metadata={
                    "semanticRefId": item.source_ref_id,
                    "semanticKind": "METRIC",
                    "metricKey": item.key,
                    "metricResolutionType": "vector_recall",
                },
            ),
            pack,
            candidate_id,
            ["Target measure"],
        )

    exact_card = card(exact, "M0")
    partial_card = card(partial, "M1")
    groups = planner._semantic_selection_candidate_groups(
        ["Target measure"],
        [exact_card, partial_card],
    )

    assert exact_card["matchType"] == "exact_label"
    assert exact_card["retrievalMatchType"] == "vector_recall"
    assert partial_card["matchType"] == "partial_label"
    assert exact_card["grain"] == "entity_day_summary"
    assert exact_card["tableKind"] == "aggregate_profile"
    assert groups == [{"phrase": "Target measure", "candidateIds": ["M0"]}]
    readable_refs = [item["ref"] for item in exact_card["readableRefs"]]
    assert readable_refs == [
        exact.source_ref_id,
        "semantic:RUNTIME_DOMAIN:fact_daily:asset",
    ]
    assert all(":field:" not in ref and not ref.endswith(":table") for ref in readable_refs)
