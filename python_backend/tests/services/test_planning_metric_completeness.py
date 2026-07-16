from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    IntentType,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
    RecallItem,
    RelationshipEntry,
)
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.planning import (
    EvidenceContractBuilder,
    QueryGraphContractValidator,
    QueryGraphValidator,
    QueryGraphPlanner,
    QuestionUnderstandingCompiler,
    SemanticMetricResolution,
    detail_evidence_requests_from_understanding,
    freeze_metric_obligation_ledger,
    more_specific_requested_measure_than_ranking,
    query_plan_question_coverage_gaps,
)


class UnconfiguredLlm:
    configured = False
    last_error = ""
    error_events = []


def test_detail_branch_is_controlled_by_typed_result_mode() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="runtime_fact",
                table="runtime_fact",
                columns=["tenant_id", "event_id", "measure_value"],
                metadata={"tableKind": "detail_fact"},
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="event_count",
                table="runtime_fact",
                aliases=["localized aggregate label"],
                metadata={"formula": "COUNT(DISTINCT event_id)"},
            )
        ],
    )
    metric_understanding = {
        "requestedMeasures": [
            {
                "metricRef": "event_count",
                "ownerTable": "runtime_fact",
                "sourcePhrase": "localized aggregate label",
                "resultMode": "metric",
            }
        ]
    }
    detail_understanding = {
        "requestedMeasures": [
            {
                "metricRef": "event_count",
                "ownerTable": "runtime_fact",
                "sourcePhrase": "localized row request",
                "resultMode": "detail",
            }
        ]
    }

    assert detail_evidence_requests_from_understanding(metric_understanding, pack) == []
    assert detail_evidence_requests_from_understanding(detail_understanding, pack) == [
        ("runtime_fact", "localized row request")
    ]


def test_semantic_selector_planning_contract_uses_current_runtime_understanding():
    planner = QueryGraphPlanner(UnconfiguredLlm())

    payload = planner._semantic_asset_selection_payload(
        "runtime question",
        RecallBundle(),
        PlanningAssetPack(),
        {
            "fastUnderstanding": {
                "intentKind": "metric_query",
                "analysisIntent": "lookup",
                "metricPhrases": ["runtime metric"],
                "timeWindowDays": 7,
            }
        },
    )

    assert payload["planningContract"]["intentKind"] == "metric_query"
    assert payload["planningContract"]["timeWindowDays"] == 7
    assert payload["metricPhrases"] == ["runtime metric"]


def test_partial_semantic_selection_with_clarification_cannot_compile():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    request_payload = {
        "retrievedCandidates": [
            {"id": "M0", "ref": "semantic:test:profile:metric:orders"},
            {"id": "M1", "ref": "semantic:test:profile:metric:refund_rate"},
        ],
        "candidateGroups": [
            {"phrase": "订单量", "candidateIds": ["M0"]},
            {"phrase": "退款率", "candidateIds": ["M1"]},
        ],
    }
    payload = planner._normalize_semantic_selection_payload(
        {
            "action": "ask_human",
            "selectedRefs": ["M0"],
            "clarifications": [
                {
                    "phrase": "退款率",
                    "question": "请选择退款率口径",
                    "options": [{"ref": "M1", "label": "退款率"}],
                }
            ],
        },
        request_payload,
        allow_read=False,
    )

    assert payload["status"] == "NEED_CLARIFICATION"
    assert payload["_uncoveredPhrases"] == ["退款率"]

    plan = planner._compile_semantic_asset_selection_payload("订单量和退款率", payload, PlanningAssetPack())
    assert not plan.intents
    assert "SEMANTIC_SELECTION_INCOMPLETE:退款率" in plan.compiler_trace
    assert "请选择退款率口径" in plan.clarification_needs


def test_complete_selected_refs_override_empty_ask_human_shell():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    request_payload = {
        "retrievedCandidates": [
            {"id": "M0", "ref": "semantic:test:profile:metric:orders"},
        ],
        "candidateGroups": [
            {"phrase": "订单量", "candidateIds": ["M0"]},
        ],
    }

    payload = planner._normalize_semantic_selection_payload(
        {
            "action": "ask_human",
            "selectedRefs": ["M0"],
            "clarifications": [],
        },
        request_payload,
        allow_read=False,
    )

    assert payload["status"] == "SELECTED"
    assert payload["action"] == "select"
    assert payload["selectedRefs"] == ["semantic:test:profile:metric:orders"]


def test_question_coverage_uses_frozen_metric_phrases_and_recalled_bindings():
    pack = PlanningAssetPack(
        metric_compaction={
            "fastUnderstanding": {"metricPhrases": ["订单量", "GMV", "退款金额"]},
            "recalledMetricEvidence": [
                {"matchedMetricLabel": "订单量", "ownerTable": "profile", "metricKey": "orders"},
                {"matchedMetricLabel": "GMV", "ownerTable": "profile", "metricKey": "gmv"},
                {"matchedMetricLabel": "退款金额", "ownerTable": "profile", "metricKey": "refund"},
            ],
        }
    )
    plan = QueryPlan(
        question_understanding={
            "selectedMetrics": [
                {"sourcePhrase": "退款金额", "ownerTable": "profile", "metricRef": "refund"}
            ]
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                plan_task_id="refund",
                preferred_table="profile",
                metric_name="refund",
                metric_column="refund",
            )
        ],
    )

    gaps = query_plan_question_coverage_gaps("最近7天看三个指标走势", plan, pack)
    assert {
        gap.evidence
        for gap in gaps
        if gap.code in {"METRIC_OBLIGATION_UNRESOLVED", "METRIC_OBLIGATION_NOT_PLANNED"}
    } == {"订单量", "GMV"}


def test_metric_obligation_ledger_freezes_every_fast_understanding_phrase_with_a_structured_decision():
    pack = PlanningAssetPack(
        metric_compaction={
            "fastUnderstanding": {
                "metricPhrases": ["GMV", "退款金额", "冻结金额"],
            },
            "recalledMetricEvidence": [
                {
                    "matchedMetricLabel": "GMV",
                    "ownerTable": "profile",
                    "metricKey": "gmv",
                    "semanticRefId": "semantic:test:profile:metric:gmv",
                },
                {
                    "matchedMetricLabel": "退款金额",
                    "ownerTable": "refund_detail",
                    "metricKey": "refund_amount",
                    "semanticRefId": "semantic:test:refund_detail:metric:refund_amount",
                    "metricResolutionAmbiguous": True,
                },
                {
                    "matchedMetricLabel": "退款金额",
                    "ownerTable": "refund_summary",
                    "metricKey": "refund_amount",
                    "semanticRefId": "semantic:test:refund_summary:metric:refund_amount",
                    "metricResolutionAmbiguous": True,
                },
            ],
        }
    )
    understanding = {
        "selectedMetrics": [
            {
                "sourcePhrase": "GMV",
                "ownerTable": "profile",
                "metricRef": "gmv",
                "semanticRefId": "semantic:test:profile:metric:gmv",
            }
        ]
    }

    frozen = freeze_metric_obligation_ledger(understanding, pack)
    ledger = frozen["metricObligations"]

    assert [item["sourcePhrase"] for item in ledger] == ["GMV", "退款金额", "冻结金额"]
    assert [item["decision"] for item in ledger] == ["selected", "ambiguous", "unresolved"]
    assert len({item["obligationId"] for item in ledger}) == 3
    assert freeze_metric_obligation_ledger(frozen, pack)["metricObligations"] == ledger
    repaired_payload = {
        "metricPhrases": ["GMV"],
        "selectedMetrics": understanding["selectedMetrics"],
        "metricObligations": ledger,
    }
    assert [
        item["sourcePhrase"]
        for item in freeze_metric_obligation_ledger(
            repaired_payload,
            PlanningAssetPack(),
        )["metricObligations"]
    ] == ["GMV", "退款金额", "冻结金额"]


def test_graph_contract_rejects_selected_metric_shrink_against_the_frozen_obligation_ledger():
    pack = PlanningAssetPack(
        metric_compaction={
            "fastUnderstanding": {"metricPhrases": ["GMV", "退款金额", "冻结金额"]},
            "recalledMetricEvidence": [
                {"matchedMetricLabel": "GMV", "ownerTable": "profile", "metricKey": "gmv"},
                {
                    "matchedMetricLabel": "退款金额",
                    "ownerTable": "refund_detail",
                    "metricKey": "refund_amount",
                    "metricResolutionAmbiguous": True,
                },
                {
                    "matchedMetricLabel": "退款金额",
                    "ownerTable": "refund_summary",
                    "metricKey": "refund_amount",
                    "metricResolutionAmbiguous": True,
                },
            ],
        }
    )
    plan = QueryPlan(
        question_understanding={
            "selectedMetrics": [
                {"sourcePhrase": "GMV", "ownerTable": "profile", "metricRef": "gmv"}
            ]
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="gmv",
                preferred_table="profile",
                metric_name="gmv",
            )
        ],
    )

    gaps = QueryGraphContractValidator().validate(plan, pack)

    assert {(gap.code, gap.evidence) for gap in gaps} >= {
        ("METRIC_OBLIGATION_AMBIGUOUS", "退款金额"),
        ("METRIC_OBLIGATION_UNRESOLVED", "冻结金额"),
    }


def test_graph_validator_rejects_a_selected_obligation_that_has_no_executable_node():
    pack = PlanningAssetPack(
        metric_compaction={
            "fastUnderstanding": {"metricPhrases": ["GMV", "退款金额"]},
        }
    )
    plan = QueryPlan(
        question_understanding={
            "selectedMetrics": [
                {"sourcePhrase": "GMV", "ownerTable": "profile", "metricRef": "gmv"},
                {"sourcePhrase": "退款金额", "ownerTable": "profile", "metricRef": "refund_amount"},
            ]
        },
        intents=[
            QuestionIntent(
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="gmv",
                preferred_table="profile",
                metric_name="gmv",
            )
        ],
    )

    result = QueryGraphValidator().validate("看 GMV 和退款金额", plan, pack)

    assert ("METRIC_OBLIGATION_NOT_PLANNED", "退款金额") in {
        (gap.code, gap.evidence) for gap in result.gaps
    }


def test_root_metric_is_not_promoted_by_phrase_substring_without_an_explicit_asset_relation_and_question_evidence():
    base_ref = "semantic:test:profile:metric:gmv"
    variant_ref = "semantic:test:profile:metric:net_gmv"
    ranking = {
        "metricRef": "gmv",
        "ownerTable": "profile",
        "sourcePhrase": "GMV",
        "semanticRefId": base_ref,
    }
    measure = {
        "metricRef": "net_gmv",
        "ownerTable": "profile",
        "sourcePhrase": "net GMV",
        "semanticRefId": variant_ref,
    }
    unrelated_pack = PlanningAssetPack(
        metrics=[
            PlanningAssetEntry(key="gmv", table="profile", title="GMV", source_ref_id=base_ref),
            PlanningAssetEntry(key="net_gmv", table="profile", title="net GMV", source_ref_id=variant_ref),
        ]
    )

    assert more_specific_requested_measure_than_ranking(
        ranking,
        {"requestedMeasures": [measure]},
        question="GMV 是多少",
        asset_pack=unrelated_pack,
    ) == {}

    related_pack = PlanningAssetPack(
        metrics=[
            PlanningAssetEntry(key="gmv", table="profile", title="GMV", source_ref_id=base_ref),
            PlanningAssetEntry(
                key="net_gmv",
                table="profile",
                title="net GMV",
                source_ref_id=variant_ref,
                metadata={"variantOf": base_ref},
            ),
        ]
    )
    assert more_specific_requested_measure_than_ranking(
        ranking,
        {"requestedMeasures": [measure]},
        question="net GMV 是多少",
        asset_pack=related_pack,
    ) == measure


def test_semantic_resolution_fills_truth_policies_from_enriched_time_semantics():
    metric = PlanningAssetEntry(
        key="snapshot_value",
        table="profile",
        title="Snapshot value",
        source_ref_id="semantic:test:profile:metric:snapshot_value",
        metadata={
            "sourceColumns": ["snapshot_value"],
            "timeColumn": "snapshot_pt",
            "metricGrain": "merchant_snapshot",
            "aggregationPolicy": "latest_value_only",
            "applicableTimeGrain": "day",
            "timeSemantics": {
                "selectionPolicy": "latest_as_of",
                "missingDataPolicy": "disclose_unknown",
                "zeroValuePolicy": "preserve_observed_zero",
            },
        },
    )

    payload = SemanticMetricResolution(
        requested_metric_ref="snapshot_value",
        source_phrase="Snapshot value",
        metric=metric,
        confidence=1.0,
        resolution_source="semantic_metric_ref",
    ).payload()

    assert payload["timeColumn"] == "snapshot_pt"
    assert payload["missingValuePolicy"] == "disclose_unknown"
    assert payload["zeroValueMeaning"] == "preserve_observed_zero"


def test_semantic_selector_candidate_card_keeps_the_temporal_execution_contract():
    ref = "semantic:test:profile:metric:snapshot_value"
    metric = PlanningAssetEntry(
        key="snapshot_value",
        table="profile",
        title="Snapshot value",
        source_ref_id=ref,
        metadata={
            "metricGrain": "merchant_snapshot",
            "timeColumn": "snapshot_pt",
            "aggregationPolicy": "latest_value_only",
            "applicableTimeGrain": "day",
            "timeSemantics": {
                "selectionPolicy": "latest_as_of",
                "missingDataPolicy": "disclose_unknown",
                "zeroValuePolicy": "preserve_observed_zero",
            },
        },
    )
    pack = PlanningAssetPack(metrics=[metric])
    item = RecallItem(
        doc_id=ref,
        title="Snapshot value",
        table="profile",
        metadata={"semanticRefId": ref, "metricKey": "snapshot_value"},
    )

    card = QueryGraphPlanner(UnconfiguredLlm())._semantic_selection_candidate_card(
        item,
        pack,
        "M0",
        ["Snapshot value"],
    )

    assert card["metricGrain"] == "merchant_snapshot"
    assert card["timeColumn"] == "snapshot_pt"
    assert card["applicableTimeGrain"] == "day"
    assert card["timeSemantics"]["selectionPolicy"] == "latest_as_of"


def test_selector_group_decision_binds_a_context_wrapped_original_obligation_without_substring_inference():
    ref = "semantic:test:profile:metric:signal_value"
    metric = PlanningAssetEntry(
        key="signal_value",
        table="profile",
        title="Signal value",
        source_ref_id=ref,
    )
    planner = QueryGraphPlanner(UnconfiguredLlm())
    decisions = planner._semantic_selection_metric_decisions(
        {
            "selectedRefs": [ref],
            "_retrievedCandidates": [{"id": "M0", "ref": ref}],
            "_candidateGroups": [
                {"phrase": "recent-window Signal value", "candidateIds": ["M0"]}
            ],
        },
        [
            {
                "metric": metric,
                "sourcePhrase": "Signal value",
                "semanticRefId": ref,
            }
        ],
    )
    frozen = freeze_metric_obligation_ledger(
        {"metricCandidateDecisions": decisions},
        PlanningAssetPack(
            metric_compaction={
                "fastUnderstanding": {
                    "metricPhrases": ["recent-window Signal value"]
                }
            }
        ),
    )

    assert frozen["metricObligations"][0]["decision"] == "selected"
    assert frozen["metricObligations"][0]["selectedMetrics"][0]["metricRef"] == "signal_value"


def test_time_ranked_primary_metric_keeps_same_grain_measures_as_parallel_siblings():
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table="table_a",
                columns=["tenant_key", "time_axis", "value_a", "entity_link"],
                metadata={"timeColumn": "time_axis"},
            ),
            PlanningAssetEntry(
                table="table_b",
                columns=["tenant_key", "time_axis", "value_b", "entity_link"],
                metadata={"timeColumn": "time_axis"},
            ),
        ],
        fields=[
            PlanningAssetEntry(
                key="time_axis",
                table="table_a",
                metadata={"semantic": {"role": "TIME"}},
            ),
            PlanningAssetEntry(
                key="time_axis",
                table="table_b",
                metadata={"semantic": {"role": "TIME"}},
            ),
        ],
        entity_keys=[
            PlanningAssetEntry(
                key="entity_link",
                table="table_a",
                metadata={"semantic": {"role": "KEY"}},
            ),
            PlanningAssetEntry(
                key="entity_link",
                table="table_b",
                metadata={"semantic": {"role": "KEY"}},
            ),
        ],
        relationships=[
            RelationshipEntry(
                relationship_id="entity_edge",
                left_table="table_a",
                right_table="table_b",
                join_keys=[{"leftColumn": "entity_link", "rightColumn": "entity_link"}],
            )
        ],
        metrics=[
            PlanningAssetEntry(
                key="metric_a",
                table="table_a",
                title="Signal A",
                source_ref_id="semantic:test:table_a:metric:metric_a",
                metadata={
                    "formula": "SUM(value_a)",
                    "sourceColumns": ["value_a"],
                    "timeColumn": "time_axis",
                },
            ),
            PlanningAssetEntry(
                key="metric_b",
                table="table_b",
                title="Signal B",
                source_ref_id="semantic:test:table_b:metric:metric_b",
                metadata={
                    "formula": "SUM(value_b)",
                    "sourceColumns": ["value_b"],
                    "timeColumn": "time_axis",
                },
            ),
        ],
        metric_compaction={
            "fastUnderstanding": {"metricPhrases": ["Signal A", "Signal B"]}
        },
    )
    understanding = {
        "analysisGrain": "time",
        "analysisIntent": "trend",
        "rankingObjective": {
            "metricRef": "metric_a",
            "ownerTable": "table_a",
            "sourcePhrase": "Signal A",
            "objectiveType": "topn",
            "groupByColumn": "time_axis",
            "order": "desc",
            "limit": 5,
            "producesFilterSet": True,
            "resultMode": "metric",
        },
        "requestedMeasures": [
            {
                "metricRef": "metric_b",
                "ownerTable": "table_b",
                "sourcePhrase": "Signal B",
                "resultMode": "metric",
            }
        ],
        "scopeConstraints": [],
        "filters": [],
        "timeWindowDays": 30,
        "suppressDefaultTrendContext": True,
    }

    plan = QuestionUnderstandingCompiler().compile(
        "Show Signal A and Signal B for the recent 30 days",
        understanding,
        pack,
    )

    secondary = next(intent for intent in plan.intents if intent.metric_name == "metric_b")
    assert secondary.group_by_column == "time_axis"
    assert secondary.depends_on_task_ids == []
    assert secondary.limit >= 30
    assert not plan.dependencies
    assert any(
        marker.endswith(":sibling_metric:table_b.metric_b")
        for marker in plan.compiler_trace
        if marker.startswith("GRAPH_ROLE:")
    )


def test_metric_specs_each_become_required_evidence_contracts():
    intent = QuestionIntent(
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.GROUP_AGG,
        plan_task_id="merged",
        preferred_table="profile",
        group_by_column="pt",
        metric_name="orders",
        metric_column="orders",
        metric_specs=[
            {"metricName": "orders", "metricColumn": "orders", "metricFormula": "SUM(orders)"},
            {"metricName": "gmv", "metricColumn": "gmv", "metricFormula": "SUM(gmv)"},
            {"metricName": "refund", "metricColumn": "refund", "metricFormula": "SUM(refund)"},
        ],
    )
    builder = EvidenceContractBuilder()
    contracts = builder.contracts_from_intents([intent])
    plan = QueryPlan(
        intents=[intent],
        evidence_contracts=contracts,
        final_required_evidence=builder.final_evidence_labels([intent]),
    )
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="merged",
                success=True,
                query_bundle=QueryBundle(tables=["profile"], rows=[{"pt": "2026-07-01", "orders": 10}]),
            )
        ],
        merged_query_bundle=QueryBundle(rows=[{"pt": "2026-07-01", "orders": 10}]),
    )

    assert [contract["semanticLabel"] for contract in contracts] == ["orders", "gmv", "refund"]
    verified = EvidenceVerifier().verify("订单量、GMV、退款金额", plan, run)
    assert not verified.passed
    assert {gap.evidence for gap in verified.gaps} >= {"gmv", "refund"}


def daily_value_metric_pack() -> tuple[PlanningAssetPack, str]:
    ref = "semantic:test:profile:metric:refund_rate_1d"
    return (
        PlanningAssetPack(
            tables=[
                PlanningAssetEntry(
                    key="profile",
                    table="profile",
                    columns=["merchant_id", "pt", "refund_rate_1d"],
                    metadata={"timeColumn": "pt", "dataGrain": "merchant_day_summary"},
                )
            ],
            metrics=[
                PlanningAssetEntry(
                    key="refund_rate_1d",
                    table="profile",
                    columns=["refund_rate_1d"],
                    title="每日退款率",
                    aliases=["每日退款率"],
                    metadata={
                        "formula": "AVG(refund_rate_1d)",
                        "sourceColumns": ["refund_rate_1d"],
                        "aggregationPolicy": "daily_value_only",
                        "applicableTimeGrain": "day",
                    },
                    source_ref_id=ref,
                )
            ],
        ),
        ref,
    )


def daily_value_selection_payload(ref: str) -> dict:
    return {
        "status": "SELECTED",
        "action": "select",
        "queryContract": {"contractType": "independent_metrics", "timeWindowDays": 30},
        "selectedRefs": [ref],
        "selectedAssets": [
            {
                "semanticRefId": ref,
                "metricRef": "refund_rate_1d",
                "ownerTable": "profile",
                "sourcePhrase": "每日退款率",
            }
        ],
    }


def test_daily_value_only_metric_rejects_multi_day_single_value_compilation():
    pack, ref = daily_value_metric_pack()
    plan = QueryGraphPlanner(UnconfiguredLlm())._compile_semantic_asset_selection_payload(
        "最近30天退款率是多少",
        daily_value_selection_payload(ref),
        pack,
    )

    assert not plan.intents
    assert any("DAILY_VALUE_PERIOD_ROLLUP_UNSAFE" in item for item in plan.compiler_trace)
    assert "planner.semantic_asset_selection.requires_full_planner" in plan.agent_trace


def test_daily_value_only_metric_uses_structured_time_series_contract_and_forces_pt():
    pack, ref = daily_value_metric_pack()
    payload = daily_value_selection_payload(ref)
    payload["planningContract"] = {
        "analysisIntent": "trend",
        "timeGrain": "day",
        "timeWindowDays": 30,
    }
    plan = QueryGraphPlanner(UnconfiguredLlm())._compile_semantic_asset_selection_payload(
        "opaque wording without time-series keywords",
        payload,
        pack,
    )

    assert plan.intents
    assert all(intent.group_by_column == "pt" for intent in plan.intents)
    assert all(intent.metric_resolution.get("aggregationPolicy") == "daily_value_only" for intent in plan.intents)
    assert all(intent.metric_resolution.get("applicableTimeGrain") == "day" for intent in plan.intents)
    assert "SEMANTIC_SELECTION_DAILY_VALUE_FORCED_PT" in plan.compiler_trace


def test_bare_gmv_does_not_expand_qualified_asset_metrics():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    profile_ref = "semantic:test:profile:metric:order_gmv"
    pack = PlanningAssetPack(
        metrics=[
            PlanningAssetEntry(
                key="order_gmv",
                title="订单GMV",
                aliases=["GMV", "订单GMV"],
                source_ref_id=profile_ref,
            ),
            PlanningAssetEntry(
                key="pay_gmv",
                title="支付GMV",
                aliases=["支付GMV"],
                source_ref_id="semantic:test:profile:metric:pay_gmv",
            ),
            PlanningAssetEntry(
                key="trade_success_gmv",
                title="交易成功GMV",
                aliases=["交易成功GMV"],
                source_ref_id="semantic:test:profile:metric:trade_success_gmv",
            ),
        ]
    )
    candidates = [
        {
            "id": "M0",
            "ref": profile_ref,
            "metricKey": "order_gmv",
            "name": "订单GMV",
            "aliases": ["GMV", "订单GMV"],
            "matched": "GMV",
        }
    ]

    expanded = planner._semantic_selection_add_asset_candidates(
        "最近30天GMV是多少",
        pack,
        candidates,
        ["GMV"],
        8,
    )

    assert [item["ref"] for item in expanded] == [profile_ref]


def test_candidate_group_requires_exact_metric_phrase_match():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    candidates = [
        {
            "id": "M0",
            "metricKey": "order_cnt_1d",
            "name": "订单量",
            "aliases": ["订单量"],
            "matched": "订单量",
        },
        {
            "id": "M1",
            "metricKey": "return_rate_by_order",
            "name": "商家退货率",
            "aliases": ["退货量占订单量比例"],
            "matched": "退货量占订单量比例",
        },
    ]

    groups = planner._semantic_selection_candidate_groups(["订单量"], candidates)

    assert groups == [{"phrase": "订单量", "candidateIds": ["M0"]}]


def test_candidate_group_binds_context_wrapped_phrase_to_longest_complete_asset_label():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    candidates = [
        {
            "id": "M0",
            "metricKey": "metric_base",
            "name": "基础指标",
            "aliases": ["指标"],
            "matched": "指标",
            "matchType": "exact_label",
        },
        {
            "id": "M1",
            "metricKey": "metric_qualified",
            "name": "限定指标",
            "aliases": ["限定指标"],
            "matched": "限定指标",
            "matchType": "exact_label",
        },
    ]

    groups = planner._semantic_selection_candidate_groups(["最近窗口限定指标"], candidates)

    assert groups == [{"phrase": "最近窗口限定指标", "candidateIds": ["M1"]}]


def test_metric_phrases_are_deduplicated_by_normalized_text():
    planner = QueryGraphPlanner(UnconfiguredLlm())
    pack = PlanningAssetPack(
        metrics=[
            PlanningAssetEntry(
                key="order_gmv",
                title="订单GMV",
                aliases=["GMV"],
                source_ref_id="semantic:test:profile:metric:order_gmv",
            )
        ]
    )

    phrases = planner._semantic_selection_metric_phrases(
        "最近30天GMV是多少",
        {"fastUnderstanding": {"metricPhrases": ["gmv", "GMV"]}},
        pack,
    )

    assert len(phrases) == 1
    assert phrases[0].lower() == "gmv"
