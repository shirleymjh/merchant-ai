from __future__ import annotations

import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterator, Mapping, Optional


Clock = Callable[[], float]


@dataclass(frozen=True)
class GroundedRuntimeBudgetLimits:
    """Hard limits shared by one grounded-runtime invocation."""

    max_duration_seconds: float = 90.0
    max_llm_calls: int = 16
    max_tool_calls: int = 60
    max_doris_queries: int = 12
    profile: str = "complex"

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        fast_path: bool = False,
    ) -> "GroundedRuntimeBudgetLimits":
        duration_setting = "run_budget_fast_duration_seconds" if fast_path else "run_budget_max_duration_seconds"
        duration_default = 25 if fast_path else 90
        return cls(
            max_duration_seconds=max(
                0.001,
                float(getattr(settings, duration_setting, duration_default) or duration_default),
            ),
            max_llm_calls=max(1, int(getattr(settings, "run_budget_max_llm_calls", 16) or 16)),
            max_tool_calls=max(1, int(getattr(settings, "run_budget_max_tool_calls", 60) or 60)),
            max_doris_queries=max(1, int(getattr(settings, "run_budget_max_doris_queries", 12) or 12)),
            profile="fast" if fast_path else "complex",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "maxDurationSeconds": self.max_duration_seconds,
            "maxLlmCalls": self.max_llm_calls,
            "maxToolCalls": self.max_tool_calls,
            "maxDorisQueries": self.max_doris_queries,
            "profile": self.profile,
        }


class GroundedRuntimeBudgetExceeded(RuntimeError):
    """Raised before an operation would exceed a grounded-runtime limit."""

    def __init__(self, breaches: list[str], report: Mapping[str, Any]):
        self.breaches = tuple(breaches)
        self.report = dict(report)
        super().__init__("grounded runtime budget exhausted: %s" % ", ".join(breaches))


@dataclass(frozen=True)
class _ActiveStage:
    name: str
    started_monotonic: float


@dataclass
class _StageAggregate:
    calls: int = 0
    completed: int = 0
    successes: int = 0
    errors: int = 0
    completed_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    last_duration_seconds: float = 0.0


class GroundedRuntimeBudget:
    """Thread-safe budget enforcement and timing telemetry for grounded runs.

    Counters are reserved before the corresponding external operation starts.
    This makes the limits safe when independent grounded goals execute in
    parallel and means failed LLM/tool/Doris attempts still consume budget.
    """

    def __init__(
        self,
        limits: GroundedRuntimeBudgetLimits,
        *,
        monotonic_clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ) -> None:
        self.limits = limits
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._lock = RLock()
        self._started_monotonic = float(self._monotonic_clock())
        self._started_epoch_ms = int(float(self._wall_clock()) * 1000)
        self._finished_monotonic: Optional[float] = None
        self._finished_epoch_ms: Optional[int] = None

        self._llm_calls = 0
        self._llm_turns = 0
        self._llm_calls_by_name: Counter[str] = Counter()
        self._llm_turns_by_name: Counter[str] = Counter()
        self._tool_calls = 0
        self._tool_calls_by_name: Counter[str] = Counter()
        self._doris_queries = 0
        self._doris_queries_by_name: Counter[str] = Counter()

        self._stage_sequence = 0
        self._active_stages: Dict[int, _ActiveStage] = {}
        self._stage_aggregates: Dict[str, _StageAggregate] = defaultdict(_StageAggregate)
        self._denied_attempts: list[Dict[str, Any]] = []

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        fast_path: bool = False,
        monotonic_clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ) -> "GroundedRuntimeBudget":
        return cls(
            GroundedRuntimeBudgetLimits.from_settings(settings, fast_path=fast_path),
            monotonic_clock=monotonic_clock,
            wall_clock=wall_clock,
        )

    @property
    def deadline_epoch_ms(self) -> int:
        return self._started_epoch_ms + int(self.limits.max_duration_seconds * 1000)

    def elapsed_seconds(self) -> float:
        with self._lock:
            return self._elapsed_seconds_locked()

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.max_duration_seconds - self.elapsed_seconds())

    def clamp_timeout_seconds(
        self,
        requested_seconds: Any = None,
        *,
        minimum_seconds: float = 0.0,
        operation: str = "timeout_clamp",
    ) -> float:
        """Clamp one blocking operation to this run's remaining wall time.

        ``minimum_seconds`` is useful for clients whose timeout API only
        accepts whole seconds.  When the run has less than that minimum left,
        the operation is denied before the external call starts instead of
        rounding the timeout up past the shared deadline.
        """

        with self._lock:
            elapsed = self._elapsed_seconds_locked()
            self._raise_for_breaches_locked(
                ["duration"]
                if elapsed >= self.limits.max_duration_seconds
                else [],
                operation=operation,
            )
            remaining = max(
                0.0,
                self.limits.max_duration_seconds - elapsed,
            )
            try:
                minimum = max(0.0, float(minimum_seconds))
            except (TypeError, ValueError):
                minimum = 0.0
            if remaining < minimum:
                self._raise_for_breaches_locked(
                    ["duration"],
                    operation=operation,
                )
            try:
                requested = float(requested_seconds)
            except (TypeError, ValueError):
                requested = 0.0
            if requested <= 0:
                return remaining
            return min(max(requested, minimum), remaining)

    def checkpoint(self) -> None:
        """Fail if the wall-time deadline has passed.

        Count limits are checked by their atomic consume methods so reaching an
        LLM limit does not prevent a no-LLM finalization step from completing.
        """

        with self._lock:
            self._raise_for_breaches_locked(self._duration_breaches_locked(), operation="checkpoint")

    def consume_llm_call(self, *, name: str = "core", turns: int = 1) -> None:
        turns = _positive_count(turns, "turns")
        normalized_name = _counter_name(name)
        with self._lock:
            breaches = self._duration_breaches_locked()
            if self._llm_calls + 1 > self.limits.max_llm_calls:
                breaches.append("llm_calls")
            self._raise_for_breaches_locked(breaches, operation="llm:%s" % normalized_name)
            self._llm_calls += 1
            self._llm_turns += turns
            self._llm_calls_by_name[normalized_name] += 1
            self._llm_turns_by_name[normalized_name] += turns

    def consume_llm_turns(self, count: int = 1, *, name: str = "core") -> None:
        """Record extra logical turns that did not create another provider call."""

        count = _positive_count(count, "count")
        normalized_name = _counter_name(name)
        with self._lock:
            self._raise_for_breaches_locked(self._duration_breaches_locked(), operation="llm_turn:%s" % normalized_name)
            self._llm_turns += count
            self._llm_turns_by_name[normalized_name] += count

    def consume_tool_call(self, name: str, *, count: int = 1) -> None:
        count = _positive_count(count, "count")
        normalized_name = _counter_name(name)
        with self._lock:
            breaches = self._duration_breaches_locked()
            if self._tool_calls + count > self.limits.max_tool_calls:
                breaches.append("tool_calls")
            self._raise_for_breaches_locked(breaches, operation="tool:%s" % normalized_name)
            self._tool_calls += count
            self._tool_calls_by_name[normalized_name] += count

    def consume_doris_query(self, *, name: str = "execute_sql", count: int = 1) -> None:
        count = _positive_count(count, "count")
        normalized_name = _counter_name(name)
        with self._lock:
            breaches = self._duration_breaches_locked()
            if self._doris_queries + count > self.limits.max_doris_queries:
                breaches.append("doris_queries")
            self._raise_for_breaches_locked(breaches, operation="doris:%s" % normalized_name)
            self._doris_queries += count
            self._doris_queries_by_name[normalized_name] += count

    def start_stage(self, name: str) -> int:
        normalized_name = _counter_name(name)
        with self._lock:
            self._raise_for_breaches_locked(self._duration_breaches_locked(), operation="stage:%s" % normalized_name)
            self._stage_sequence += 1
            token = self._stage_sequence
            self._active_stages[token] = _ActiveStage(
                name=normalized_name,
                started_monotonic=float(self._monotonic_clock()),
            )
            self._stage_aggregates[normalized_name].calls += 1
            return token

    def end_stage(self, token: int, *, error: bool = False) -> None:
        with self._lock:
            active = self._active_stages.pop(token, None)
            if active is None:
                raise KeyError("unknown or completed grounded-runtime stage token: %s" % token)
            duration = max(0.0, float(self._monotonic_clock()) - active.started_monotonic)
            aggregate = self._stage_aggregates[active.name]
            aggregate.completed += 1
            aggregate.errors += int(error)
            aggregate.successes += int(not error)
            aggregate.completed_duration_seconds += duration
            aggregate.max_duration_seconds = max(aggregate.max_duration_seconds, duration)
            aggregate.last_duration_seconds = duration

    @contextmanager
    def stage(self, name: str, *, enforce_on_exit: bool = True) -> Iterator["GroundedRuntimeBudget"]:
        token = self.start_stage(name)
        try:
            yield self
        except BaseException:
            self.end_stage(token, error=True)
            raise
        else:
            self.end_stage(token)
            if enforce_on_exit:
                self.checkpoint()

    def finish(self) -> Dict[str, Any]:
        """Freeze elapsed time and return the final JSON-serializable report."""

        with self._lock:
            if self._finished_monotonic is None:
                self._finished_monotonic = float(self._monotonic_clock())
                self._finished_epoch_ms = int(float(self._wall_clock()) * 1000)
            return self._report_locked()

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return self._report_locked()

    def _elapsed_seconds_locked(self) -> float:
        end = self._finished_monotonic
        if end is None:
            end = float(self._monotonic_clock())
        return max(0.0, end - self._started_monotonic)

    def _duration_breaches_locked(self) -> list[str]:
        if self._elapsed_seconds_locked() >= self.limits.max_duration_seconds:
            return ["duration"]
        return []

    def _raise_for_breaches_locked(self, breaches: list[str], *, operation: str) -> None:
        breaches = list(dict.fromkeys(breaches))
        if not breaches:
            return
        self._denied_attempts.append(
            {
                "operation": operation,
                "breaches": breaches,
                "elapsedMs": _milliseconds(self._elapsed_seconds_locked()),
            }
        )
        self._denied_attempts = self._denied_attempts[-32:]
        raise GroundedRuntimeBudgetExceeded(breaches, self._report_locked())

    def _report_locked(self) -> Dict[str, Any]:
        elapsed_seconds = self._elapsed_seconds_locked()
        breaches = self._current_breaches_locked(elapsed_seconds)
        active_duration_by_name: Dict[str, float] = defaultdict(float)
        active_count_by_name: Dict[str, int] = defaultdict(int)
        active_max_duration_by_name: Dict[str, float] = defaultdict(float)
        now = self._finished_monotonic
        if now is None:
            now = float(self._monotonic_clock())
        for active in self._active_stages.values():
            active_duration = max(0.0, now - active.started_monotonic)
            active_count_by_name[active.name] += 1
            active_duration_by_name[active.name] += active_duration
            active_max_duration_by_name[active.name] = max(
                active_max_duration_by_name[active.name],
                active_duration,
            )

        stages: Dict[str, Dict[str, Any]] = {}
        for name in sorted(self._stage_aggregates):
            aggregate = self._stage_aggregates[name]
            active_duration = active_duration_by_name[name]
            stages[name] = {
                "calls": aggregate.calls,
                "completed": aggregate.completed,
                "successes": aggregate.successes,
                "errors": aggregate.errors,
                "active": active_count_by_name[name],
                "completedDurationMs": _milliseconds(aggregate.completed_duration_seconds),
                "activeDurationMs": _milliseconds(active_duration),
                "totalDurationMs": _milliseconds(aggregate.completed_duration_seconds + active_duration),
                "maxDurationMs": _milliseconds(
                    max(aggregate.max_duration_seconds, active_max_duration_by_name[name])
                ),
                "lastDurationMs": _milliseconds(aggregate.last_duration_seconds),
            }

        remaining = {
            "durationMs": _milliseconds(max(0.0, self.limits.max_duration_seconds - elapsed_seconds)),
            "llmCalls": max(0, self.limits.max_llm_calls - self._llm_calls),
            "toolCalls": max(0, self.limits.max_tool_calls - self._tool_calls),
            "dorisQueries": max(0, self.limits.max_doris_queries - self._doris_queries),
        }
        return {
            "startedAtEpochMs": self._started_epoch_ms,
            "finishedAtEpochMs": self._finished_epoch_ms,
            "deadlineEpochMs": self.deadline_epoch_ms,
            "elapsedMs": _milliseconds(elapsed_seconds),
            "limits": self.limits.as_dict(),
            "usage": {
                "llmCalls": self._llm_calls,
                "llmTurns": self._llm_turns,
                "llmCallsByName": dict(sorted(self._llm_calls_by_name.items())),
                "llmTurnsByName": dict(sorted(self._llm_turns_by_name.items())),
                "toolCalls": self._tool_calls,
                "toolCallsByName": dict(sorted(self._tool_calls_by_name.items())),
                "dorisQueries": self._doris_queries,
                "dorisQueriesByName": dict(sorted(self._doris_queries_by_name.items())),
            },
            "remaining": remaining,
            "stages": stages,
            "breaches": breaches,
            "exhausted": bool(breaches),
            "reason": (
                "grounded runtime budget exhausted: %s" % ", ".join(breaches)
                if breaches
                else ""
            ),
            "status": "finished" if self._finished_monotonic is not None else "running",
            "deniedAttempts": [dict(item) for item in self._denied_attempts],
        }

    def _current_breaches_locked(self, elapsed_seconds: float) -> list[str]:
        breaches: list[str] = []
        if elapsed_seconds >= self.limits.max_duration_seconds:
            breaches.append("duration")
        if self._llm_calls >= self.limits.max_llm_calls:
            breaches.append("llm_calls")
        if self._tool_calls >= self.limits.max_tool_calls:
            breaches.append("tool_calls")
        if self._doris_queries >= self.limits.max_doris_queries:
            breaches.append("doris_queries")
        return breaches


def _counter_name(value: Any) -> str:
    return str(value or "unknown").strip() or "unknown"


def _positive_count(value: int, field: str) -> int:
    count = int(value)
    if count <= 0:
        raise ValueError("%s must be greater than zero" % field)
    return count


def _milliseconds(seconds: float) -> float:
    return round(max(0.0, float(seconds)) * 1000, 3)
