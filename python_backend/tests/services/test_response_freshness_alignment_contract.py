from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    AgentRunResult,
    FreshnessCheckResult,
    SnapshotAlignmentContract,
    SnapshotSourceWindow,
)


def test_response_freshness_uses_common_anchor_not_newest_individual_source():
    workflow = create_workflow(get_settings())
    run_result = AgentRunResult(
        snapshot_alignment=SnapshotAlignmentContract(
            status="ALIGNED_PARTIAL_COVERAGE",
            strategy="common_latest_partition",
            aligned=True,
            complete=False,
            common_anchor_time_value="20260710",
            disclosure_required=True,
            sources=[
                SnapshotSourceWindow(
                    task_id="recharge",
                    table="deposit_recharge",
                    source_max_time_value="20260710",
                    effective_start_time_value="20260704",
                    effective_end_time_value="20260710",
                    compatible=True,
                    coverage_complete=True,
                ),
                SnapshotSourceWindow(
                    task_id="profile",
                    table="merchant_profile",
                    source_max_time_value="20260711",
                    effective_start_time_value="20260704",
                    effective_end_time_value="20260710",
                    compatible=True,
                    coverage_complete=False,
                    reason="the unified anchor is not covered by all requested fields",
                ),
            ],
        )
    )
    state = {
        "agent_run_result": run_result,
        "freshness_reports": [
            FreshnessCheckResult(task_id="recharge", table="deposit_recharge", checked=True, max_time_value="20260710"),
            FreshnessCheckResult(task_id="profile", table="merchant_profile", checked=True, max_time_value="20260711"),
        ],
    }

    freshness = workflow.data_freshness_for_response(state, [])

    assert freshness["status"] == "aligned_partial_coverage"
    assert freshness["latestDataAt"] == "20260710"
    assert freshness["commonAnchorTimeValue"] == "20260710"
    assert [item["sourceLatestDataAt"] for item in freshness["sourceCutoffs"]] == ["20260710", "20260711"]
    assert freshness["answerDisclosure"]["required"] is True
    assert any("统一时间窗口" in note for note in freshness["notes"])
