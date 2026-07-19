from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_budget(
    *,
    duration: float = 90,
    llm: int = 8,
    tools: int = 60,
    doris: int = 12,
    monotonic: FakeClock | None = None,
    wall: FakeClock | None = None,
) -> GroundedRuntimeBudget:
    return GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=duration,
            max_llm_calls=llm,
            max_tool_calls=tools,
            max_doris_queries=doris,
        ),
        monotonic_clock=monotonic or FakeClock(),
        wall_clock=wall or FakeClock(1_700_000_000),
    )


def test_budget_uses_existing_settings_for_complex_and_fast_profiles() -> None:
    settings = SimpleNamespace(
        run_budget_max_duration_seconds=90,
        run_budget_fast_duration_seconds=25,
        run_budget_max_llm_calls=7,
        run_budget_max_tool_calls=41,
        run_budget_max_doris_queries=9,
    )

    complex_budget = GroundedRuntimeBudget.from_settings(settings)
    fast_budget = GroundedRuntimeBudget.from_settings(settings, fast_path=True)

    assert complex_budget.limits.as_dict() == {
        "maxDurationSeconds": 90.0,
        "maxLlmCalls": 7,
        "maxToolCalls": 41,
        "maxDorisQueries": 9,
        "profile": "complex",
    }
    assert fast_budget.limits.max_duration_seconds == 25
    assert fast_budget.limits.profile == "fast"


def test_report_tracks_llm_turns_tools_by_name_and_doris_queries() -> None:
    monotonic = FakeClock(10)
    wall = FakeClock(1_700_000_000)
    budget = make_budget(monotonic=monotonic, wall=wall)

    budget.consume_llm_call(name="core", turns=2)
    budget.consume_llm_call(name="sql_repair")
    budget.consume_llm_turns(2, name="core")
    budget.consume_tool_call("read_file", count=2)
    budget.consume_tool_call("submit_grounded_contract")
    budget.consume_doris_query(name="deterministic_query", count=2)
    monotonic.advance(1.25)

    report = budget.report()

    assert report["elapsedMs"] == 1250
    assert report["deadlineEpochMs"] == 1_700_000_090_000
    assert report["usage"] == {
        "llmCalls": 2,
        "llmTurns": 5,
        "llmCallsByName": {"core": 1, "sql_repair": 1},
        "llmTurnsByName": {"core": 4, "sql_repair": 1},
        "toolCalls": 3,
        "toolCallsByName": {"read_file": 2, "submit_grounded_contract": 1},
        "dorisQueries": 2,
        "dorisQueriesByName": {"deterministic_query": 2},
    }
    assert report["remaining"] == {
        "durationMs": 88750,
        "llmCalls": 6,
        "toolCalls": 57,
        "dorisQueries": 10,
    }
    json.dumps(report)


@pytest.mark.parametrize(
    ("consume", "breach"),
    [
        (lambda item: item.consume_llm_call(), "llm_calls"),
        (lambda item: item.consume_tool_call("read_file"), "tool_calls"),
        (lambda item: item.consume_doris_query(), "doris_queries"),
    ],
)
def test_count_limits_allow_exact_limit_and_reject_next_call(consume, breach: str) -> None:
    budget = make_budget(llm=1, tools=1, doris=1)

    consume(budget)
    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        consume(budget)

    assert raised.value.breaches == (breach,)
    report = budget.report()
    assert report["exhausted"] is True
    assert breach in report["breaches"]
    assert report["deniedAttempts"][-1]["breaches"] == [breach]


def test_reaching_one_count_limit_does_not_block_other_or_finalization_work() -> None:
    budget = make_budget(llm=1, tools=2, doris=1)

    budget.consume_llm_call()
    budget.checkpoint()
    budget.consume_tool_call("finalize_evidence")

    assert budget.report()["usage"]["toolCalls"] == 1


def test_duration_is_enforced_at_checkpoints_and_before_counted_operations() -> None:
    monotonic = FakeClock(5)
    budget = make_budget(duration=2, monotonic=monotonic)
    monotonic.advance(2)

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        budget.checkpoint()
    with pytest.raises(GroundedRuntimeBudgetExceeded):
        budget.consume_tool_call("read_file")

    assert raised.value.breaches == ("duration",)
    assert budget.report()["usage"]["toolCalls"] == 0
    assert budget.report()["remaining"]["durationMs"] == 0


def test_timeout_minimum_denies_external_call_when_less_than_one_second_remains() -> None:
    monotonic = FakeClock()
    budget = make_budget(duration=5, monotonic=monotonic)
    monotonic.advance(4.25)

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        budget.clamp_timeout_seconds(
            30,
            minimum_seconds=1,
            operation="doris_query_timeout",
        )

    assert raised.value.breaches == ("duration",)
    assert budget.report()["deniedAttempts"][-1]["operation"] == (
        "doris_query_timeout"
    )


def test_timeout_minimum_clamps_without_rounding_past_remaining_budget() -> None:
    monotonic = FakeClock()
    budget = make_budget(duration=5, monotonic=monotonic)
    monotonic.advance(2.4)

    timeout_seconds = budget.clamp_timeout_seconds(
        30,
        minimum_seconds=1,
        operation="doris_query_timeout",
    )

    assert timeout_seconds == pytest.approx(2.6)


def test_stage_context_aggregates_repeated_durations_and_errors() -> None:
    monotonic = FakeClock(100)
    budget = make_budget(monotonic=monotonic)

    with budget.stage("recall"):
        monotonic.advance(0.25)
    with pytest.raises(ValueError) as exc_info:
        with budget.stage("recall"):
            monotonic.advance(0.75)
            raise ValueError("broken")
    assert "broken" in str(exc_info.value)

    report = budget.report()
    assert report["stages"]["recall"] == {
        "calls": 2,
        "completed": 2,
        "successes": 1,
        "errors": 1,
        "active": 0,
        "completedDurationMs": 1000,
        "activeDurationMs": 0,
        "totalDurationMs": 1000,
        "maxDurationMs": 750,
        "lastDurationMs": 750,
    }


def test_stage_context_enforces_duration_after_successful_external_work() -> None:
    monotonic = FakeClock()
    budget = make_budget(duration=1, monotonic=monotonic)

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        with budget.stage("core_llm"):
            monotonic.advance(1.1)

    assert raised.value.breaches == ("duration",)
    assert budget.report()["stages"]["core_llm"]["successes"] == 1


def test_report_includes_active_parallel_stage_time() -> None:
    monotonic = FakeClock()
    budget = make_budget(monotonic=monotonic)
    first = budget.start_stage("query_execution")
    monotonic.advance(0.2)
    second = budget.start_stage("query_execution")
    monotonic.advance(0.3)

    active_report = budget.report()["stages"]["query_execution"]
    assert active_report["active"] == 2
    assert active_report["activeDurationMs"] == 800
    assert active_report["maxDurationMs"] == 500

    budget.end_stage(first)
    budget.end_stage(second)
    final_report = budget.report()["stages"]["query_execution"]
    assert final_report["completed"] == 2
    assert final_report["totalDurationMs"] == 800


def test_finish_freezes_elapsed_time_and_produces_serializable_report() -> None:
    monotonic = FakeClock(20)
    wall = FakeClock(1_700_000_000)
    budget = make_budget(monotonic=monotonic, wall=wall)
    monotonic.advance(3)
    wall.advance(3)

    report = budget.finish()
    monotonic.advance(100)
    wall.advance(100)

    assert report["status"] == "finished"
    assert report["elapsedMs"] == 3000
    assert report["finishedAtEpochMs"] == 1_700_000_003_000
    assert budget.report()["elapsedMs"] == 3000
    json.dumps(report)


def test_parallel_reservations_never_overbook_the_shared_tool_budget() -> None:
    budget = make_budget(tools=25)

    def reserve() -> bool:
        try:
            budget.consume_tool_call("parallel_goal")
        except GroundedRuntimeBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(lambda _: reserve(), range(100)))

    assert sum(accepted) == 25
    assert budget.report()["usage"]["toolCalls"] == 25
    assert budget.report()["remaining"]["toolCalls"] == 0


def test_invalid_bulk_counts_fail_without_changing_usage() -> None:
    budget = make_budget()

    with pytest.raises(ValueError):
        budget.consume_tool_call("read_file", count=0)
    with pytest.raises(ValueError):
        budget.consume_doris_query(count=-1)
    with pytest.raises(ValueError):
        budget.consume_llm_call(turns=0)

    assert budget.report()["usage"]["toolCalls"] == 0
    assert budget.report()["usage"]["dorisQueries"] == 0
    assert budget.report()["usage"]["llmCalls"] == 0
