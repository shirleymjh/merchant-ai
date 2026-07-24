from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.models import (
    AgentRunResult,
    ExtractedKeywords,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    RecallBundle,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.answer import AnswerComposeService
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedRuntimeBudgetMiddleware,
)
from merchant_ai.services.grounded_conversation_state import (
    GroundedConversationResolution,
)
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession


class _Catalog:
    @staticmethod
    def read(*, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        del max_chars, offset
        if path == "topics/orders/manifest.json":
            return {
                "success": True,
                "refId": "semantic:orders:manifest",
                "path": path,
                "kind": "TOPIC_MANIFEST",
                "topic": "orders",
                "content": '{"topic":"orders","tables":[{"tableName":"orders"}]}',
            }
        return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}


class _BudgetKernel:
    def __init__(self) -> None:
        self.new_session_calls = 0
        self.execute_calls = 0

    def new_session(
        self,
        question: str,
        merchant_id: str,
        **kwargs: Any,
    ) -> GroundedRuntimeSession:
        self.new_session_calls += 1
        return GroundedRuntimeSession(
            session_id="budget-session",
            question=question,
            merchant_id=merchant_id,
            merchant=kwargs.get("merchant") or MerchantInfo(merchant_id=merchant_id),
        )

    @staticmethod
    def route_topic(session: GroundedRuntimeSession) -> TopicRoutingDecision:
        session.keywords = ExtractedKeywords(keywords=["orders"])
        session.routing = TopicRoutingDecision(
            primary_topic="orders",
            candidate_topics=["orders"],
            routing_mode="seed_topic",
        )
        session.workspace_topics = ["orders"]
        return session.routing

    @staticmethod
    def recall_navigation(
        session: GroundedRuntimeSession,
        **_: Any,
    ) -> RecallBundle:
        session.recall = RecallBundle()
        return session.recall

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        **_: Any,
    ) -> AgentRunResult:
        self.execute_calls += 1
        raise AssertionError("Doris execution must not start after budget denial")


class _CapturingTimeoutKernel(_BudgetKernel):
    def __init__(self) -> None:
        super().__init__()
        self.received_runtime_budgets: list[GroundedRuntimeBudget | None] = []

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        **kwargs: Any,
    ) -> AgentRunResult:
        del session
        self.execute_calls += 1
        self.received_runtime_budgets.append(kwargs.get("runtime_budget"))
        return AgentRunResult()

    @staticmethod
    def verify_active(session: GroundedRuntimeSession) -> VerifiedEvidence:
        del session
        return VerifiedEvidence(passed=True)


class _BudgetGraph:
    def __init__(
        self,
        action: str,
        tools: list[Any],
        middleware: list[Any],
    ) -> None:
        self.action = action
        self.tools = {item.name: item for item in tools}
        self.budget_middleware = next(item for item in middleware if isinstance(item, GroundedRuntimeBudgetMiddleware))
        self.last_context: Any = None

    def invoke(self, payload: dict[str, Any], *, config: Any, context: Any) -> None:
        del payload, config
        self.last_context = context
        runtime = SimpleNamespace(context=context)
        if self.action == "model":
            request = SimpleNamespace(runtime=runtime)
            self.budget_middleware.wrap_model_call(request, lambda _: object())
            self.budget_middleware.wrap_model_call(request, lambda _: object())
            return
        if self.action == "tool":
            request = SimpleNamespace(
                runtime=runtime,
                tool_call={"name": "retrieve_knowledge"},
            )
            self.budget_middleware.wrap_tool_call(request, lambda _: object())
            self.budget_middleware.wrap_tool_call(request, lambda _: object())
            return
        raise AssertionError("unknown graph action: %s" % self.action)


class _BudgetFactory:
    def __init__(self, action: str) -> None:
        self.action = action
        self.graph: _BudgetGraph | None = None

    def __call__(self, **kwargs: Any) -> _BudgetGraph:
        self.graph = _BudgetGraph(
            self.action,
            kwargs["tools"],
            kwargs["middleware"],
        )
        return self.graph


class _StandaloneConversationAuthority:
    @staticmethod
    def resolve(question: str, **_: Any) -> GroundedConversationResolution:
        normalized = str(question or "").strip()
        return GroundedConversationResolution(
            original_question=normalized,
            effective_question=normalized,
            status="STANDALONE",
            source="TEST_STRUCTURED_SEMANTIC_REVIEW",
        )


def _limits(**updates: Any) -> GroundedRuntimeBudgetLimits:
    values = {
        "max_duration_seconds": 30,
        "max_llm_calls": 2,
        "max_tool_calls": 2,
        "max_doris_queries": 2,
    }
    values.update(updates)
    return GroundedRuntimeBudgetLimits(**values)


def _runtime(action: str, kernel: _BudgetKernel) -> GroundedDeepAgentRuntime:
    settings = SimpleNamespace(
        run_budget_max_duration_seconds=30,
        run_budget_max_llm_calls=1,
        run_budget_max_tool_calls=1,
        run_budget_max_doris_queries=1,
    )
    return GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=_Catalog(),
        settings=settings,
        agent_factory=_BudgetFactory(action),
        backend=object(),
        conversation_online_authority=_StandaloneConversationAuthority(),
    )


@pytest.mark.parametrize(
    ("action", "expected_breach"),
    [
        ("model", "llm_calls"),
        ("tool", "tool_calls"),
    ],
)
def test_run_converts_count_budget_exhaustion_to_operational_failure(
    action: str,
    expected_breach: str,
) -> None:
    kernel = _BudgetKernel()
    runtime = _runtime(action, kernel)

    response = runtime.run("query orders", "merchant-1")

    harness = response.debug_trace["harness"]
    failure = harness["operationalFailure"]
    assert failure["code"] == "GROUNDED_RUNTIME_BUDGET_EXHAUSTED"
    assert expected_breach in failure["breaches"]
    assert harness["runtimeBudget"]["status"] == "finished"
    assert response.data_rows == []
    assert "未完成或未验证的结果" in response.answer
    assert runtime.deep_agent_graph.last_context is not None
    assert (
        runtime.deep_agent_graph.last_context.session.execution_graph_replan_evidence
        == {}
    )


def test_core_model_request_timeout_is_clamped_to_remaining_run_budget() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=5))
    middleware = GroundedRuntimeBudgetMiddleware()
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    original = SimpleNamespace(
        runtime=runtime,
        model_settings={"timeout": 60, "temperature": 0},
    )

    def override(**updates: Any) -> Any:
        return SimpleNamespace(
            runtime=original.runtime,
            model_settings=updates["model_settings"],
            override=override,
        )

    original.override = override
    captured: list[Any] = []

    middleware.wrap_model_call(
        original,
        lambda request: captured.append(request) or object(),
    )

    assert len(captured) == 1
    settings = captured[0].model_settings
    assert settings["temperature"] == 0
    assert 0 < settings["timeout"] <= 5
    assert original.model_settings["timeout"] == 60


def test_core_model_request_without_timeout_uses_per_attempt_hard_cap() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=90))
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_call_timeout_seconds=20,
            grounded_core_model_retry_attempts=2,
        )
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    original = SimpleNamespace(runtime=runtime, model_settings={})

    def override(**updates: Any) -> Any:
        return SimpleNamespace(
            runtime=original.runtime,
            model_settings=updates["model_settings"],
            override=override,
        )

    original.override = override
    captured: list[Any] = []

    middleware.wrap_model_call(
        original,
        lambda request: captured.append(request) or object(),
    )

    assert len(captured) == 1
    assert 0 < captured[0].model_settings["timeout"] <= 20


def test_core_model_timeout_retries_once_and_counts_each_provider_call() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=90))
    retry_delays: list[float] = []
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_call_timeout_seconds=20,
            grounded_core_model_retry_attempts=2,
            grounded_core_model_retry_base_delay_ms=500,
            grounded_core_model_retry_max_delay_ms=4000,
        ),
        sleeper=retry_delays.append,
        jitter=lambda lower, upper: upper,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    original = SimpleNamespace(runtime=runtime, model_settings={})

    def override(**updates: Any) -> Any:
        return SimpleNamespace(
            runtime=original.runtime,
            model_settings=updates["model_settings"],
            override=override,
        )

    original.override = override
    attempts: list[float] = []

    def handler(request: Any) -> object:
        attempts.append(float(request.model_settings["timeout"]))
        if len(attempts) == 1:
            raise TimeoutError("provider read operation timed out")
        return object()

    result = middleware.wrap_model_call(original, handler)

    assert result is not None
    assert len(attempts) == 2
    assert retry_delays == [0.5]
    assert all(0 < timeout <= 20 for timeout in attempts)
    report = budget.report()
    assert report["usage"]["actionsByName"] == {
        "grounded_core": 1
    }
    assert report["usage"]["llmCallsByName"] == {"grounded_core": 2}
    assert report["stages"]["llm.grounded_core"]["calls"] == 2
    assert report["stages"]["llm.grounded_core"]["errors"] == 1
    assert report["stages"]["llm.grounded_core"]["successes"] == 1
    assert report["stages"]["llm.grounded_core.attempt_1"]["errors"] == 1
    assert report["stages"]["llm.grounded_core.attempt_2"]["successes"] == 1
    assert report["stages"]["llm.grounded_core.retry_backoff"]["calls"] == 1


def test_core_model_token_budget_counts_each_retry_input_and_output_usage() -> None:
    budget = GroundedRuntimeBudget(
        _limits(
            max_llm_calls=2,
            max_estimated_tokens=100,
        )
    )
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_retry_attempts=2,
            grounded_core_model_retry_base_delay_ms=0,
        ),
        provider_token_counter=lambda messages, system, tools: 11,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(
        runtime=runtime,
        model_settings={},
        messages=[SimpleNamespace(content="question")],
        tools=[],
        system_message=None,
    )
    calls = 0

    def handler(_: Any) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("provider timed out")
        return SimpleNamespace(
            result=[
                SimpleNamespace(
                    content="answer",
                    usage_metadata={"output_tokens": 7},
                )
            ]
        )

    middleware.wrap_model_call(request, handler)

    report = budget.report()
    assert report["usage"]["actions"] == 1
    assert report["usage"]["llmCalls"] == 2
    assert report["usage"]["estimatedTokens"] == 29
    assert report["usage"]["estimatedTokensByName"] == {
        "grounded_core.input.injected_provider_counter": 22,
        "grounded_core.output.provider_usage": 7,
    }


def test_core_model_circuit_fast_fails_until_cooldown_elapses() -> None:
    now = {"value": 1_000.0}
    budget = GroundedRuntimeBudget(
        _limits(
            max_actions=6,
            max_llm_calls=6,
        )
    )
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_retry_attempts=1,
            llm_circuit_threshold=2,
            llm_circuit_cooldown_seconds=30,
        ),
        wall_clock=lambda: now["value"],
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(runtime=runtime, model_settings={})
    provider_calls = 0

    def failing_handler(_: Any) -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise TimeoutError("provider timed out")

    for _ in range(2):
        with pytest.raises(TimeoutError):
            middleware.wrap_model_call(request, failing_handler)

    with pytest.raises(RuntimeError) as circuit_open:
        middleware.wrap_model_call(request, failing_handler)
    assert "circuit is open" in str(circuit_open.value)
    assert provider_calls == 2
    assert middleware._circuit_report()["open"] is True

    now["value"] += 31
    result = middleware.wrap_model_call(request, lambda _: object())

    assert result is not None
    assert middleware._circuit_report()["open"] is False


def test_core_model_rate_limit_uses_bounded_backoff_then_retries() -> None:
    class RateLimitError(RuntimeError):
        status_code = 429

    budget = GroundedRuntimeBudget(
        _limits(max_duration_seconds=90, max_llm_calls=3)
    )
    retry_delays: list[float] = []
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_call_timeout_seconds=20,
            grounded_core_model_retry_attempts=3,
            grounded_core_model_retry_base_delay_ms=200,
            grounded_core_model_retry_max_delay_ms=1000,
        ),
        sleeper=retry_delays.append,
        jitter=lambda lower, upper: upper,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(runtime=runtime, model_settings={})
    calls = 0

    def handler(_: Any) -> object:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError("too many requests")
        return object()

    result = middleware.wrap_model_call(request, handler)

    assert result is not None
    assert calls == 3
    assert retry_delays == [0.2, 0.4]
    assert budget.report()["usage"]["llmCalls"] == 3


def test_core_model_retry_backoff_cannot_escape_run_deadline() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=0.1))
    retry_delays: list[float] = []
    middleware = GroundedRuntimeBudgetMiddleware(
        SimpleNamespace(
            grounded_core_model_retry_attempts=2,
            grounded_core_model_retry_base_delay_ms=500,
            grounded_core_model_retry_max_delay_ms=4000,
        ),
        sleeper=retry_delays.append,
        jitter=lambda lower, upper: upper,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(runtime=runtime, model_settings={})
    calls = 0

    def handler(_: Any) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timed out")

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        middleware.wrap_model_call(request, handler)

    assert raised.value.breaches == ("duration",)
    assert calls == 1
    assert retry_delays == []


def test_core_model_does_not_retry_non_timeout_provider_error() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=90))
    middleware = GroundedRuntimeBudgetMiddleware()
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(runtime=runtime, model_settings={})
    calls = 0

    def handler(_: Any) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider rejected request")

    with pytest.raises(RuntimeError) as exc_info:
        middleware.wrap_model_call(request, handler)
    assert "provider rejected request" in str(exc_info.value)

    assert calls == 1
    assert budget.report()["usage"]["llmCalls"] == 1


def test_core_model_timeout_does_not_retry_without_llm_call_budget() -> None:
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=90, max_llm_calls=1))
    middleware = GroundedRuntimeBudgetMiddleware(SimpleNamespace(grounded_core_model_retry_attempts=2))
    runtime = SimpleNamespace(context=SimpleNamespace(budget=budget))
    request = SimpleNamespace(runtime=runtime, model_settings={})
    calls = 0

    def handler(_: Any) -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timed out")

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        middleware.wrap_model_call(request, handler)

    assert raised.value.breaches == ("llm_calls",)
    assert calls == 1
    assert budget.report()["usage"]["llmCalls"] == 1


def test_budget_exhaustion_before_session_creation_returns_controlled_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DeadlineClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 2.0

    budget = GroundedRuntimeBudget(
        _limits(max_duration_seconds=1),
        monotonic_clock=_DeadlineClock(),
    )
    monkeypatch.setattr(
        GroundedRuntimeBudget,
        "from_settings",
        classmethod(lambda cls, settings, **kwargs: budget),
    )
    kernel = _BudgetKernel()
    runtime = _runtime("model", kernel)

    response = runtime.run("query orders", "merchant-1")

    harness = response.debug_trace["harness"]
    assert kernel.new_session_calls == 0
    assert harness["operationalFailure"]["breaches"] == ["duration"]
    assert harness["runtimeBudget"]["status"] == "finished"
    assert "未完成或未验证的结果" in response.answer


def test_operational_budget_failure_takes_precedence_over_stale_answer_state() -> None:
    kernel = _BudgetKernel()
    state = kernel.new_session("query orders", "merchant-1")
    state.answer = "partial answer that must not escape"
    session = GroundedDeepAgentSession(
        runtime=state,
        operational_failure={
            "code": "GROUNDED_RUNTIME_BUDGET_EXHAUSTED",
            "breaches": ["duration"],
        },
        runtime_budget_report={"status": "finished", "exhausted": True},
    )

    response = GroundedDeepAgentRuntime._governed_response(
        session,
        "budget-thread",
        "budget-run",
    )

    assert response.answer != state.answer
    assert "未完成或未验证的结果" in response.answer
    assert response.data_rows == []
    assert response.debug_trace["harness"]["operationalFailure"]["code"] == ("GROUNDED_RUNTIME_BUDGET_EXHAUSTED")


def test_non_budget_operational_failure_preserves_real_error_code() -> None:
    kernel = _BudgetKernel()
    state = kernel.new_session("query orders", "merchant-1")
    session = GroundedDeepAgentSession(
        runtime=state,
        operational_failure={
            "code": "POPULATION_PRE_EXECUTION_REJECTED",
            "failureDisposition": "OPERATIONAL_TERMINAL",
            "retryable": False,
        },
        runtime_budget_report={
            "status": "finished",
            "exhausted": False,
        },
    )

    response = GroundedDeepAgentRuntime._governed_response(
        session,
        "population-thread",
        "population-run",
    )

    assert "POPULATION_PRE_EXECUTION_REJECTED" in response.answer
    assert "运行预算内完成" not in response.answer
    assert response.debug_trace["harness"]["operationalFailure"]["code"] == (
        "POPULATION_PRE_EXECUTION_REJECTED"
    )


def test_incomplete_terminal_state_returns_controlled_failure_instead_of_500() -> None:
    kernel = _BudgetKernel()
    state = kernel.new_session("query orders", "merchant-1")
    session = GroundedDeepAgentSession(runtime=state)

    response = GroundedDeepAgentRuntime._governed_response(
        session,
        "incomplete-thread",
        "incomplete-run",
    )

    failure = response.debug_trace["harness"]["operationalFailure"]
    assert failure["code"] == "GROUNDED_CORE_INCOMPLETE_TERMINAL_STATE"
    assert "GROUNDED_CORE_INCOMPLETE_TERMINAL_STATE" in response.answer


class _AnswerLlm:
    configured = True
    settings = SimpleNamespace(llm_answer_timeout_seconds=3)

    def __init__(self) -> None:
        self.calls = 0
        self.timeouts: list[float] = []

    def chat(self, *args: Any, **kwargs: Any) -> str:
        del args
        self.calls += 1
        self.timeouts.append(float(kwargs["timeout_seconds"]))
        return ""


def test_answer_composer_reserves_budget_and_clamps_provider_timeout() -> None:
    llm = _AnswerLlm()
    composer = AnswerComposeService(llm)  # type: ignore[arg-type]
    budget = GroundedRuntimeBudget(_limits(max_duration_seconds=1, max_llm_calls=1))
    plan = QueryPlan()
    run_result = AgentRunResult(merged_query_bundle=QueryBundle(rows=[{"value": 1}], tables=["orders"]))

    composer._compose_llm_business_answer(
        "why did orders change",
        plan,
        run_result,
        "",
        MerchantInfo(merchant_id="merchant-1"),
        None,
        runtime_budget=budget,
    )
    with pytest.raises(GroundedRuntimeBudgetExceeded):
        composer._compose_llm_business_answer(
            "why did orders change",
            plan,
            run_result,
            "",
            MerchantInfo(merchant_id="merchant-1"),
            None,
            runtime_budget=budget,
        )

    assert llm.calls == 1
    assert 0 < llm.timeouts[0] <= 1
    report = budget.report()
    assert report["usage"]["llmCallsByName"] == {"answer_composer": 1}
    assert report["stages"]["llm.answer_composer"]["successes"] == 1
