from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.query_graph_contract import record_graph_validation
from merchant_ai.graph.workflow import MerchantQaWorkflow
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    EvidenceGap,
    FreshnessCheckResult,
    GraphValidationResult,
    IntentType,
    PlannerReflectionResult,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    QuestionRoute,
    RoutingDecision,
    SnapshotAlignmentContract,
    SqlRepairAttempt,
    VerifiedEvidence,
)


def _validated_recovery_state(run_result: AgentRunResult) -> dict:
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                question="近三天和今天有多少活跃用户",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="active_users",
                preferred_table="dwm_active_user_di",
            )
        ]
    )
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "fast_understood": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "query_graph_reflected": True,
        "planner_reflection": PlannerReflectionResult(passed=True),
        "query_graph_plan_attempts": 1,
        "query_graph_retrieve_count": 1,
        "query_graph_supplemental_retrieve_count": 0,
        "query_graph_repair_attempts": 0,
        "sql_generated": True,
        "sql_repair_reviewed": True,
        "react_round": 8,
        "plan": plan,
        "agent_run_result": run_result,
    }
    record_graph_validation(state, GraphValidationResult(valid=True), plan)
    return state


def test_main_agent_observation_exposes_bounded_typed_execution_signals() -> None:
    zero_bundle = QueryBundle(tables=["dwm_active_user_di"], rows=[], original_row_count=0)
    failed_bundle = QueryBundle(
        tables=["dwm_active_user_rt"],
        failed=True,
        error="unknown table prefix",
    )
    repair = SqlRepairAttempt(
        task_id="today_active_users",
        round=1,
        error_code="UNKNOWN_TABLE",
        status="failed",
        progressed=True,
        observation="prefix did not match the governed table",
    )
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="recent_active_users",
                success=True,
                query_bundle=zero_bundle,
            ),
            AgentTaskResult(
                task_id="today_active_users",
                success=False,
                query_bundle=failed_bundle,
                sql_repairs=[repair],
            ),
        ],
        freshness_reports=[
            FreshnessCheckResult(
                task_id="today_active_users",
                table="dwm_active_user_di",
                checked=True,
                requested_days=1,
                status="STALE_REQUIRES_GRAPH_REPREPARATION",
                max_time_value="2026-07-15",
                fallback_table="dwm_active_user_rt",
                coverage_complete=False,
                reason="offline source does not cover today",
            )
        ],
        snapshot_alignment=SnapshotAlignmentContract(
            status="ALIGNMENT_INCOMPLETE",
            aligned=False,
            complete=False,
            reason="offline and realtime windows are not aligned",
        ),
        sql_repairs=[repair],
    )
    workflow = object.__new__(MerchantQaWorkflow)

    observation = workflow.main_agent_observation(
        {
            "react_round": 8,
            "topic_routed": True,
            "data_discovered": True,
            "planning_assets_compacted": True,
            "sql_generated": True,
            "plan": QueryPlan(),
            "agent_run_result": run_result,
        }
    )

    execution = observation["executionObservations"]
    assert execution["tasks"] == [
        {
            "taskId": "recent_active_users",
            "tables": ["dwm_active_user_di"],
            "rowCount": 0,
            "failed": False,
            "error": "",
        },
        {
            "taskId": "today_active_users",
            "tables": ["dwm_active_user_rt"],
            "rowCount": 0,
            "failed": True,
            "error": "unknown table prefix",
        },
    ]
    assert execution["zeroRowTaskIds"] == ["recent_active_users"]
    assert execution["freshness"][0] == {
        "taskId": "today_active_users",
        "table": "dwm_active_user_di",
        "requestedDays": 1,
        "status": "STALE_REQUIRES_GRAPH_REPREPARATION",
        "maxTimeValue": "2026-07-15",
        "fallbackTable": "dwm_active_user_rt",
        "coverageComplete": False,
        "alignmentStatus": "",
        "reason": "offline source does not cover today",
    }
    assert execution["snapshotAlignment"]["status"] == "ALIGNMENT_INCOMPLETE"
    assert execution["sqlRepairAttempts"][0]["errorCode"] == "UNKNOWN_TABLE"
    assert "executionFailedTasks=today_active_users" in observation["summary"]
    assert "zeroRowTasks=recent_active_users" in observation["summary"]
    assert "fallback=dwm_active_user_rt" in observation["summary"]
    assert "snapshotAlignment=ALIGNMENT_INCOMPLETE(complete=False)" in observation["summary"]
    assert "sqlRepairAttempts=1" in observation["summary"]


def test_zero_rows_reopens_plan_and_retrieval_after_evidence_verification() -> None:
    gap = EvidenceGap(code="ZERO_ROWS", task_id="active_users", reason="query returned zero rows")
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="active_users",
                success=True,
                query_bundle=QueryBundle(tables=["dwm_active_user_di"], rows=[]),
            )
        ],
        evidence_gaps=[gap],
        verified_evidence=VerifiedEvidence(passed=False, gaps=[gap], blocking_gaps=[gap]),
    )
    state = _validated_recovery_state(run_result)
    state.update(
        {
            "evidence_graph_verified": True,
            "verification_status": "failed",
            "evidence_accepted": False,
        }
    )

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert "plan_graph" in decision.available_actions
    assert "retrieve_knowledge" in decision.available_actions
    assert "answer_data" in decision.available_actions


def test_incomplete_freshness_reopens_plan_alongside_mandatory_verification() -> None:
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="active_users",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_active_user_di"],
                    rows=[{"day": "2026-07-15", "active_users": 10}],
                ),
            )
        ],
        freshness_reports=[
            FreshnessCheckResult(
                task_id="active_users",
                table="dwm_active_user_di",
                checked=True,
                requested_days=4,
                status="AVAILABLE",
                max_time_value="2026-07-15",
                coverage_complete=False,
            )
        ],
        snapshot_alignment=SnapshotAlignmentContract(
            status="ALIGNMENT_INCOMPLETE",
            aligned=False,
            complete=False,
        ),
    )
    state = _validated_recovery_state(run_result)
    state.update(
        {
            "evidence_graph_verified": False,
            "verification_status": "not_run",
            "evidence_accepted": False,
        }
    )

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert "verify_evidence" in decision.available_actions
    assert "plan_graph" in decision.available_actions
    assert "retrieve_knowledge" in decision.available_actions
    assert "answer_data" not in decision.available_actions


def test_final_sql_failure_reopens_plan_without_bypassing_repair_or_verification() -> None:
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="active_users",
                success=False,
                query_bundle=QueryBundle(
                    tables=["dwm_active_user_di"],
                    failed=True,
                    error="unknown column active_user_cnt",
                ),
            )
        ]
    )
    state = _validated_recovery_state(run_result)
    state.update(
        {
            "sql_repair_reviewed": False,
            "evidence_graph_verified": False,
            "verification_status": "not_run",
            "evidence_accepted": False,
        }
    )

    decision = V2AgentPolicy(get_settings()).decide(state)

    assert "repair_sql" in decision.available_actions
    assert "verify_evidence" in decision.available_actions
    assert "plan_graph" in decision.available_actions
    assert "retrieve_knowledge" in decision.available_actions
    assert "answer_data" not in decision.available_actions


def test_core_can_retry_empty_plan_within_reunderstand_budget() -> None:
    settings = get_settings().model_copy(
        update={"agent_plan_rounds": 1, "agent_graph_repair_rounds": 2}
    )
    state = {
        "routing_decision": RoutingDecision(route=QuestionRoute.BUSINESS),
        "topic_routed": True,
        "fast_understood": True,
        "data_discovered": True,
        "planning_assets_compacted": True,
        "core_managed_filesystem": True,
        "query_graph_plan_attempts": 1,
        "query_graph_supplemental_retrieve_count": 0,
        "react_round": 5,
        "plan": QueryPlan(),
        "agent_run_result": AgentRunResult(),
    }
    policy = V2AgentPolicy(settings)

    retry = policy.decide(state)

    assert "plan_graph" in retry.available_actions
    assert "retrieve_knowledge" in retry.available_actions

    state["query_graph_plan_attempts"] = 3
    exhausted = policy.decide(state)
    assert "plan_graph" not in exhausted.available_actions
