from merchant_ai.models import AgentRunResult, AgentTaskResult, QueryBundle, QueryPlan
from merchant_ai.services.evidence import EvidenceVerifier


TABLE = "aggregate_metrics"
COLUMN = "metric_value"


def task_result(task_id: str, value: int) -> AgentTaskResult:
    return AgentTaskResult(
        task_id=task_id,
        success=True,
        query_bundle=QueryBundle(
            tables=[TABLE],
            rows=[{COLUMN: value}],
        ),
    )


def verify_contract(contract: dict, task_results: list[AgentTaskResult]):
    return EvidenceVerifier().verify(
        "compare the current window with the previous window",
        QueryPlan(evidence_contracts=[contract]),
        AgentRunResult(
            task_results=task_results,
            merged_query_bundle=QueryBundle(
                tables=[TABLE],
                rows=[{COLUMN: result.query_bundle.rows[0][COLUMN]} for result in task_results],
            ),
        ),
    )


def test_missing_previous_window_task_cannot_be_covered_by_current_window_on_same_table():
    verified = verify_contract(
        {
            "taskId": "previous_window",
            "table": TABLE,
            "columns": [COLUMN],
            "semanticLabel": "previous window metric",
        },
        [task_result("current_window", 120)],
    )

    assert verified.passed is False
    assert "previous window metric" not in verified.covered_evidence
    assert any(
        gap.code == "MISSING_REQUIRED_EVIDENCE" and gap.task_id == "previous_window"
        for gap in verified.gaps
    )


def test_task_scoped_contract_uses_exact_task_when_current_and_previous_share_table():
    verified = verify_contract(
        {
            "taskId": "previous_window",
            "table": TABLE,
            "columns": [COLUMN],
            "semanticLabel": "previous window metric",
        },
        [task_result("current_window", 120), task_result("previous_window", 100)],
    )

    assert verified.passed is True
    assert "previous window metric" in verified.covered_evidence


def test_legacy_table_only_contract_falls_back_when_table_result_is_unique():
    verified = verify_contract(
        {
            "table": TABLE,
            "columns": [COLUMN],
            "semanticLabel": "legacy metric",
        },
        [task_result("only_window", 100)],
    )

    assert verified.passed is True
    assert "legacy metric" in verified.covered_evidence


def test_legacy_table_only_contract_fails_when_table_result_is_ambiguous():
    verified = verify_contract(
        {
            "table": TABLE,
            "columns": [COLUMN],
            "semanticLabel": "legacy metric",
        },
        [task_result("current_window", 120), task_result("previous_window", 100)],
    )

    assert verified.passed is False
    assert "legacy metric" not in verified.covered_evidence
    assert any(gap.code == "MISSING_REQUIRED_EVIDENCE" for gap in verified.gaps)
