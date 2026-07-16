from __future__ import annotations

from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    ChatContext,
    IntentType,
    QueryBundle,
    QueryPlan,
    QuestionCategory,
    QuestionIntent,
    RouteSlots,
    SubAgentDelegationPlan,
    SubAgentDelegationTask,
    VerifiedEvidence,
)
from merchant_ai.services.distributed_workers import normalize_subagent_result
from merchant_ai.services.tools import delegate_subagent_tool


def test_delegate_subagent_tool_exposes_bounded_lead_contract():
    tool = delegate_subagent_tool(["document_analysis", "python_batch"])
    parameters = tool.openai_schema()["function"]["parameters"]

    assert tool.name == "delegate_subagent"
    assert parameters["properties"]["tasks"]["items"]["properties"]["taskKind"]["enum"] == [
        "document_analysis",
        "python_batch",
    ]
    assert set(parameters["required"]) == {
        "tasks",
        "parallel",
        "isolationMode",
        "readArtifactPolicy",
        "failureStrategy",
        "reason",
    }


def test_normalized_subagent_failure_contract_recommends_retry():
    result = normalize_subagent_result("python_batch", "timeout", {}, "provider timeout")

    assert result["status"] == "timeout"
    assert result["summary"] == "provider timeout"
    assert result["evidenceRefs"] == []
    assert result["artifactRefs"] == []
    assert result["gaps"][0]["code"] == "SUBAGENT_ERROR"
    assert result["recommendedNextAction"] == "retry_or_switch_strategy"
    assert result["retryable"] is True


def test_policy_offers_general_delegation_for_attachment():
    settings = get_settings().model_copy(update={"lead_agent_autonomous_enabled": False})
    policy = V2AgentPolicy(settings)
    state = {
        "data_discovered": False,
        "topic_routed": True,
        "fast_understood": True,
        "route_slots": RouteSlots(),
        "request_context": ChatContext(offloaded_files=["report.md"]),
        "question": "分析附件里的经营问题",
    }

    decision = policy.decide(state)

    assert decision.selected_action == "lead_arbitrate"
    assert {"delegate_subagent", "retrieve_knowledge"} <= set(decision.available_actions)


def test_policy_does_not_delegate_without_real_attachment_input():
    settings = get_settings().model_copy(update={"lead_agent_autonomous_enabled": False})
    decision = V2AgentPolicy(settings).decide(
        {
            "data_discovered": False,
            "topic_routed": True,
            "fast_understood": True,
            "route_slots": RouteSlots(),
            "request_context": ChatContext(),
            "question": "帮我分析一份还没有上传的报告",
        }
    )

    assert decision.selected_action == "retrieve_knowledge"
    assert "delegate_subagent" not in decision.available_actions


def test_workflow_delegates_document_and_records_uniform_result(tmp_path: Path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "distributed_subagents_enabled": False,
            "openai_api_key": "",
        }
    )
    workflow = create_workflow(settings)
    assert workflow.lead_llm_action_catalog(["run_analysis_skill", "delegate_subagent", "answer_data"]) == [
        "run_analysis_skill",
        "delegate_subagent",
        "answer_data",
    ]
    context = ChatContext()
    state = workflow._initial_state("总结附件", "100", context, None, "thread_delegate", "run_delegate")
    document = Path(state["thread_data"].workspace_path) / "report.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("GMV 增长 12%，退款率上升。", encoding="utf-8")
    context.offloaded_files = [str(document)]

    result_state = workflow.delegate_subagent(state)

    assert result_state["subagent_delegation_attempted"] is True
    assert result_state["subagent_delegation_completed"] is True
    assert result_state["subagent_delegation_plan"]["tasks"][0]["taskKind"] == "document_analysis"
    result = result_state["subagent_delegation_results"][0]
    assert result["status"] == "partial"
    assert result["recommendedNextAction"] == "return_partial_to_lead_agent"
    assert result["summary"]
    assert result["payload"]["fallbackUsed"] is True
    assert result["gaps"][0]["code"] == "DOCUMENT_LLM_UNAVAILABLE"
    assert set(
        [
            "status",
            "summary",
            "evidenceRefs",
            "artifactRefs",
            "gaps",
            "recommendedNextAction",
            "retryable",
        ]
    ).issubset(result)
    observation = result_state["main_agent_observations"][-1]
    assert observation["stage"] == "delegate_subagent"
    assert observation["summary"].startswith("0/1 Sub-Agent tasks completed; partial=1")
    assert "[partial:DOCUMENT_LLM_UNAVAILABLE]" in result_state["analysis_summary"]
    assert result_state["agent_run_result"].degraded_reasons[0]["code"] == "DOCUMENT_LLM_UNAVAILABLE"
    assert result_state["agent_run_result"].evidence_gaps[0].code == "DOCUMENT_LLM_UNAVAILABLE"
    assert result_state["agent_run_result"].verified_evidence.warning_gaps[0].code == "DOCUMENT_LLM_UNAVAILABLE"
    response = workflow.to_response(result_state)
    assert response.debug_trace["degradedReasons"][0]["code"] == "DOCUMENT_LLM_UNAVAILABLE"
    assert response.debug_trace["evidenceGaps"][0]["code"] == "DOCUMENT_LLM_UNAVAILABLE"
    guarded = workflow.answer_service._apply_answer_guard("文档摘录已生成。", result_state["agent_run_result"])
    assert "Sub-Agent 未完整执行" in guarded


def test_parallel_delegation_counts_only_completed_outcomes_as_success(tmp_path: Path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "distributed_subagents_enabled": False,
            "max_sub_agent_tasks": 2,
            "max_concurrent_sub_agents": 2,
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state("并行分析", "100", ChatContext(), None, "thread_parallel", "run_parallel")
    state["agent_run_result"].verified_evidence = VerifiedEvidence(passed=True)
    state["evidence_graph_verified"] = True
    plan = SubAgentDelegationPlan(
        parallel=True,
        failure_strategy="continue_partial",
        tasks=[
            SubAgentDelegationTask(task_kind="analysis_worker", objective="完整分析"),
            SubAgentDelegationTask(task_kind="document_analysis", objective="文档分析"),
        ],
    )
    workflow._allowed_delegation_kinds = lambda current: ["analysis_worker", "document_analysis"]
    workflow._build_delegation_plan = lambda current, allowed: plan

    def execute_task(current, task, failure_strategy, read_artifact_policy):
        if task.task_kind == "analysis_worker":
            return normalize_subagent_result(task.task_kind, "completed", {"answer": "完整结果"})
        return normalize_subagent_result(
            task.task_kind,
            "partial",
            {
                "answer": "摘录结果",
                "fallbackUsed": True,
                "gaps": [{"code": "DOCUMENT_LLM_UNAVAILABLE", "message": "provider unavailable"}],
            },
        )

    workflow._execute_delegation_task = execute_task

    result_state = workflow.delegate_subagent(state)

    statuses = sorted(item["status"] for item in result_state["subagent_delegation_results"])
    assert statuses == ["completed", "partial"]
    observation = result_state["main_agent_observations"][-1]
    assert observation["summary"] == "1/2 Sub-Agent tasks completed; partial=1 failed=0"
    assert "DOCUMENT_LLM_UNAVAILABLE" in {
        item["code"] for item in result_state["agent_run_result"].degraded_reasons
    }
    assert result_state["agent_run_result"].verified_evidence.passed is True
    assert result_state["agent_run_result"].verified_evidence.warning_gaps[0].code == "DOCUMENT_LLM_UNAVAILABLE"
    assert result_state["run_steps"][-1].status == "partial"


def test_workflow_delegates_generic_analysis_worker(tmp_path: Path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "distributed_subagents_enabled": False,
            "openai_api_key": "",
        }
    )
    workflow = create_workflow(settings)
    state = workflow._initial_state("最近30天GMV是否异常？", "100", ChatContext(), None, "thread_analysis_delegate", "run_analysis_delegate")
    state["plan"] = QueryPlan(
        question_understanding={"analysisIntent": "anomaly_check", "requiresExplanation": True, "analysisGrain": "day"},
        intents=[
            QuestionIntent(
                question="最近30天GMV是否异常？",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.GROUP_AGG,
                category=QuestionCategory.TRADE,
                plan_task_id="gmv_trend",
                metric_name="order_gmv_amt_1d",
                group_by_column="pt",
                preferred_table="ads_merchant_profile",
            )
        ],
    )
    bundle = QueryBundle(
        tables=["ads_merchant_profile"],
        rows=[
            {"pt": "2026-07-01", "order_gmv_amt_1d": 100},
            {"pt": "2026-07-02", "order_gmv_amt_1d": 220},
        ],
        original_row_count=2,
    )
    state["agent_run_result"] = AgentRunResult(
        task_results=[AgentTaskResult(task_id="gmv_trend", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )
    state["sql_generated"] = True
    state["evidence_graph_verified"] = True
    state["evidence_accepted"] = True

    result_state = workflow.delegate_subagent(state)

    assert result_state["subagent_delegation_plan"]["tasks"][0]["taskKind"] == "analysis_worker"
    result = result_state["subagent_delegation_results"][0]
    assert result["status"] == "completed"
    assert "ANALYSIS_WORKER" in str(result)
    assert result_state["analysis_summary"]
