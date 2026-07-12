from __future__ import annotations

from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.graph.policy import V2AgentPolicy
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import ChatContext, RouteSlots
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
        "fast_metric_attempted": False,
        "route_slots": RouteSlots(),
        "request_context": ChatContext(offloaded_files=["report.md"]),
        "question": "分析附件里的经营问题",
    }

    decision = policy.decide(state)

    assert decision.selected_action == "delegate_subagent"
    assert "retrieve_knowledge" in decision.available_actions


def test_policy_does_not_delegate_without_real_attachment_input():
    settings = get_settings().model_copy(update={"lead_agent_autonomous_enabled": False})
    decision = V2AgentPolicy(settings).decide(
        {
            "data_discovered": False,
            "topic_routed": True,
            "fast_understood": True,
            "fast_metric_attempted": False,
            "route_slots": RouteSlots(),
            "request_context": ChatContext(),
            "question": "帮我分析一份还没有上传的报告",
        }
    )

    assert decision.selected_action == "try_fast_metric"
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
    assert result["status"] == "failed"
    assert result["recommendedNextAction"] in {"retry_or_switch_strategy", "fallback_to_lead_agent"}
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
    assert result_state["main_agent_observations"][-1]["stage"] == "delegate_subagent"
