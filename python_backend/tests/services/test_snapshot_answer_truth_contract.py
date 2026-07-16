from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    IntentType,
    NodePlanContract,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    SnapshotAlignmentContract,
    SnapshotSourceWindow,
)
from merchant_ai.services.answer import (
    AnswerComposeService,
    deterministic_structured_answer,
    verified_answer_context,
)
from merchant_ai.services.evidence import EvidenceVerifier


def metric_plan_and_run(value: int) -> tuple[str, QueryPlan, AgentRunResult]:
    question = "最近30天指标是多少？"
    intent = QuestionIntent(
        question=question,
        intent_type=IntentType.VALID,
        answer_mode=AnswerMode.METRIC,
        plan_task_id="metric_task",
        preferred_table="fact_primary",
        metric_name="metric_value",
        metric_column="metric_value",
        output_keys=["metric_value"],
        metric_resolution={
            "metricKey": "metric_value",
            "displayName": "经营指标",
            "unit": "个",
        },
    )
    bundle = QueryBundle(
        tables=["fact_primary"],
        rows=[{"metric_value": value}],
        original_row_count=1,
    )
    task = AgentTaskResult(
        task_id=intent.plan_task_id,
        success=True,
        query_bundle=bundle,
        node_plan_contract=NodePlanContract(
            task_id=intent.plan_task_id,
            preferred_table=intent.preferred_table,
            metric_column=intent.metric_column,
            metric_name=intent.metric_name,
            output_keys=intent.output_keys,
            visible_columns=intent.output_keys,
        ),
    )
    return question, QueryPlan(intents=[intent]), AgentRunResult(
        task_results=[task],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
    )


def complete_alignment(anchor: str = "20260710") -> SnapshotAlignmentContract:
    return SnapshotAlignmentContract(
        status="ALIGNED_COMPLETE",
        strategy="common_latest_partition",
        aligned=True,
        complete=True,
        common_anchor_time_value=anchor,
        sources=[
            SnapshotSourceWindow(
                task_id="metric_task",
                table="fact_primary",
                status="ALIGNED",
                compatible=True,
                coverage_complete=True,
                effective_start_time_value="20260611",
                effective_end_time_value="20260710",
            )
        ],
    )


def partial_alignment() -> SnapshotAlignmentContract:
    return SnapshotAlignmentContract(
        status="ALIGNED_PARTIAL_COVERAGE",
        strategy="common_latest_partition",
        aligned=True,
        complete=False,
        common_anchor_time_value="20260710",
        disclosure_required=True,
        sources=[
            SnapshotSourceWindow(
                task_id="metric_task",
                table="fact_primary",
                status="ALIGNED",
                compatible=True,
                coverage_complete=True,
                effective_start_time_value="20260611",
                effective_end_time_value="20260710",
            ),
            SnapshotSourceWindow(
                task_id="supporting_task",
                table="fact_supporting",
                status="PARTIAL_COVERAGE",
                compatible=True,
                coverage_complete=False,
                source_min_time_value="20260620",
                source_max_time_value="20260709",
                effective_start_time_value="20260611",
                effective_end_time_value="20260710",
                reason="requested window is not fully available",
            ),
        ],
    )


def verify_run(question: str, plan: QueryPlan, run: AgentRunResult):
    verified = EvidenceVerifier().verify(question, plan, run)
    run.verified_evidence = verified
    run.evidence_gaps = verified.gaps
    run.partial_answer_reason = verified.partial_answer_reason
    return verified


def test_partial_snapshot_creates_blocking_structured_gap_and_required_disclosures():
    question, plan, run = metric_plan_and_run(12)
    run.snapshot_alignment = partial_alignment()

    verified = verify_run(question, plan, run)

    gap = next(item for item in verified.gaps if item.code == "SNAPSHOT_SOURCE_COVERAGE_INCOMPLETE")
    assert not verified.passed
    assert gap.severity == "blocking"
    assert gap.source == "freshness"
    assert gap.disclosure_required
    assert gap.task_id == "supporting_task"
    assert gap.details["commonAnchorTimeValue"] == "20260710"
    assert gap.details["coverageComplete"] is False
    assert gap.missing_time_range == "20260611..20260710"
    assert verified.answer_guard_required
    assert any("数据截至 2026-07-10" in item for item in verified.required_disclosures)
    assert any("相关结果不可用" in item and "缺失解释为 0" in item for item in verified.required_disclosures)


def test_unaligned_snapshot_without_source_details_still_fails_closed():
    question, plan, run = metric_plan_and_run(12)
    run.snapshot_alignment = SnapshotAlignmentContract(
        status="ALIGNMENT_FAILED",
        strategy="common_latest_partition",
        aligned=False,
        complete=False,
        reason="no common window",
    )

    verified = verify_run(question, plan, run)

    gap = next(item for item in verified.gaps if item.code == "SNAPSHOT_ALIGNMENT_INCOMPLETE")
    assert not verified.passed
    assert gap.source == "freshness"
    assert gap.disclosure_required
    assert any("未完成统一时间对齐" in item for item in verified.required_disclosures)


def test_complete_common_anchor_is_disclosed_by_deterministic_answer_and_prompt_context():
    question, plan, run = metric_plan_and_run(12)
    run.snapshot_alignment = complete_alignment()
    verified = verify_run(question, plan, run)

    answer = deterministic_structured_answer(question, plan, run)
    context = verified_answer_context(question, plan, run)

    assert verified.passed
    assert "经营指标为 12个" in answer
    assert "数据截至 2026-07-10" in answer
    assert context.freshness == {
        "status": "ALIGNED_COMPLETE",
        "strategy": "common_latest_partition",
        "aligned": True,
        "complete": True,
        "commonAnchorTimeValue": "2026-07-10",
        "disclosureRequired": True,
        "unavailableSourceCount": 0,
    }
    assert "sources" not in context.freshness


def test_partial_snapshot_suppresses_a_zero_row_in_deterministic_answer():
    question, plan, run = metric_plan_and_run(0)
    run.snapshot_alignment = partial_alignment()
    verify_run(question, plan, run)

    answer = deterministic_structured_answer(question, plan, run)

    assert "经营指标为 0" not in answer
    assert "相关结果不可用" in answer
    assert "不能把缺失解释为 0" in answer


def test_final_answer_cannot_reintroduce_a_zero_claim_for_partial_snapshot():
    question, plan, run = metric_plan_and_run(0)
    run.snapshot_alignment = partial_alignment()
    verify_run(question, plan, run)
    service = AnswerComposeService(object())

    answer = service._finalize_answer("最近30天经营指标为 0个。", question, plan, run)

    assert "经营指标为 0个" not in answer
    assert "相关结果不可用" in answer
    assert service.last_answer_claim_trace["passed"] is True


def test_common_anchor_disclosure_is_trusted_by_final_claim_verification():
    question, plan, run = metric_plan_and_run(12)
    run.snapshot_alignment = complete_alignment("2026-07-10 00:00:00")
    verify_run(question, plan, run)
    service = AnswerComposeService(object())

    answer = service._finalize_answer(
        deterministic_structured_answer(question, plan, run),
        question,
        plan,
        run,
    )

    assert "经营指标为 12个" in answer
    assert "数据截至 2026-07-10" in answer
    assert service.last_answer_claim_trace["passed"] is True


def test_common_anchor_is_also_trusted_on_the_generic_claim_verifier_path():
    question = "核对本轮结果"
    plan = QueryPlan()
    run = AgentRunResult(snapshot_alignment=complete_alignment())
    verify_run(question, plan, run)
    service = AnswerComposeService(object())

    answer = service._finalize_answer("本轮结果已完成核对。", question, plan, run)

    assert "数据截至 2026-07-10" in answer
    assert service.last_answer_claim_trace["passed"] is True
