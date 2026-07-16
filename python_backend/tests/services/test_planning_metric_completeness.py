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
)
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.planning import (
    EvidenceContractBuilder,
    QueryGraphPlanner,
    query_plan_question_coverage_gaps,
)


class UnconfiguredLlm:
    configured = False
    last_error = ""
    error_events = []


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
    assert {gap.evidence for gap in gaps if gap.code == "QUESTION_METRIC_NOT_COVERED"} == {"订单量", "GMV"}


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
