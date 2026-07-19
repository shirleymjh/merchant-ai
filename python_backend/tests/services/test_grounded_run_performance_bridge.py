from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import ChatContext, ChatResponse
from merchant_ai.services.runs import AgentRunManager, run_duration_ms


def test_runtime_budget_populates_run_performance_summary(tmp_path) -> None:
    settings = get_settings().model_copy(
        update={"harness_workspace_path": str(tmp_path)}
    )
    manager = AgentRunManager(settings)
    thread = manager.create_thread(
        "merchant-1",
        "topic",
        ChatContext(topic="topic"),
    )
    run = manager.create_run(
        thread.thread_id,
        "merchant-1",
        "question",
    )

    manager.complete_run(
        run.run_id,
        ChatResponse(
            answer="verified",
            debug_trace={
                "harness": {
                    "runtimeBudget": {
                        "elapsedMs": 42.5,
                        "usage": {"toolCalls": 3},
                    }
                }
            },
        ),
    )

    completed = manager.get_run(run.run_id)
    assert completed is not None
    assert completed.performance_summary["totalDurationMs"] == 42.5
    assert completed.performance_summary["runtimeBudget"]["usage"] == {
        "toolCalls": 3
    }
    assert run_duration_ms(completed) == 42.5
