from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    ChatContext,
    ChatResponse,
    MerchantInfo,
    QuestionRoute,
    RoutingDecision,
    UserIdentity,
)
from merchant_ai.services import runtime_factory
from merchant_ai.services.runtime_factory import GroundedApplicationRuntime, RuntimeServices


class FakeCore:
    def __init__(self) -> None:
        self.deep_agent_graph = SimpleNamespace(get_state=lambda config: SimpleNamespace())
        self.semantic_catalog = object()
        self.calls: list[dict[str, Any]] = []

    def run(self, question: str, merchant_id: str, **kwargs: Any) -> ChatResponse:
        self.calls.append({"question": question, "merchantId": merchant_id, **kwargs})
        return ChatResponse(answer="42", debug_trace={"harness": {}})


class FakeCheckpointManager:
    def deep_agent_ref(self, thread_id: str, run_id: str) -> dict[str, Any]:
        return {
            "threadId": thread_id,
            "runId": run_id,
            "checkpointNamespace": "deepagent",
            "resumable": True,
        }

    def config_for_deep_agent(self, thread_id: str, run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": "deepagent"}}


class FakePreflight:
    def __init__(
        self,
        route: QuestionRoute = QuestionRoute.BUSINESS,
        *,
        semantic_route: str = "BUSINESS_TASK",
        surface_signals: dict[str, Any] | None = None,
        fail: bool = False,
    ) -> None:
        self.route = route
        self.semantic_route = semantic_route
        self.surface_signals = dict(surface_signals or {})
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def understand(self, question: str, pending_context: bool = False) -> Any:
        self.calls.append(
            {
                "question": question,
                "pendingContext": pending_context,
            }
        )
        if self.fail:
            raise RuntimeError("preflight unavailable")
        return SimpleNamespace(
            routing_decision=RoutingDecision(
                route=self.route,
                reason="fake preflight decision",
            ),
            semantic_trace={
                "status": "success",
                "route": self.semantic_route,
                "confidence": 0.95,
            },
            surface_signals=dict(self.surface_signals),
            clarification_question="",
        )


def facade(
    core: FakeCore | None,
    *,
    preflight: FakePreflight | None = None,
) -> GroundedApplicationRuntime:
    settings = get_settings().model_copy(update={"merchant_id": "m-default"})
    merchant_service = SimpleNamespace(
        current_merchant=lambda merchant_id: MerchantInfo(merchant_id=merchant_id)
    )
    return GroundedApplicationRuntime(
        settings=settings,
        core=core,
        services=RuntimeServices(
            topic_assets=object(),
            recall_service=object(),
            knowledge_retriever=object(),
            doris_repository=object(),
            access_control=object(),
            merchant_service=merchant_service,
            answer_repository=object(),
            pending_store=object(),
            keyword_service=object(),
            preflight_understanding=(preflight or FakePreflight()),
            answer_service=object(),
            memory_store=object(),
            merchant_profile_store=object(),
            recall_cache_clearers=(),
        ),
        checkpoint_manager=FakeCheckpointManager(),
        unavailable_reason="required grounded model authority is not configured",
    )


def test_deepagent_factory_branch_never_imports_legacy_workflow(monkeypatch: Any) -> None:
    sentinel = object()
    monkeypatch.setattr(runtime_factory, "create_grounded_runtime", lambda settings: sentinel)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "merchant_ai.graph.workflow":
            raise AssertionError("deepagent branch imported legacy workflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    settings = get_settings().model_copy(update={"agent_mode": "deepagent"})

    assert runtime_factory.create_runtime(settings) is sentinel


def test_legacy_workflow_cannot_be_selected_as_online_query_authority() -> None:
    settings = get_settings().model_copy(update={"agent_mode": "legacy"})

    with pytest.raises(ValueError) as exc_info:
        runtime_factory.create_runtime(settings)
    assert "grounded deepagent only" in str(exc_info.value)


def test_facade_adapts_api_signature_without_workflow_state() -> None:
    core = FakeCore()
    outer = facade(core)
    context = ChatContext(
        user_identity=UserIdentity(
            merchant_id="m-1",
            role="merchant_owner",
            permissions=["chat.run"],
        )
    )
    events: list[tuple[str, str, dict[str, Any]]] = []

    response = outer.run(
        "最近30天订单量",
        "m-1",
        context,
        lambda event_type, node, payload: events.append((event_type, node, payload)),
        "thread_1",
        "run_1",
        [],
    )

    assert response.answer == "42"
    assert response.context == context
    assert response.debug_trace["harness"]["runtime"] == "grounded_deepagent"
    assert response.debug_trace["harness"]["legacyFallbackUsed"] is False
    assert core.calls[0]["access_role"] == "merchant_admin"
    assert core.calls[0]["user_scope"]["merchantId"] == "m-1"
    assert [event[0] for event in events] == ["runtime.started", "answer.ready"]


def test_facade_greeting_fast_path_does_not_invoke_grounded_core() -> None:
    core = FakeCore()
    preflight = FakePreflight(
        QuestionRoute.GREETING,
        semantic_route="GREETING",
    )
    outer = facade(core, preflight=preflight)
    events: list[tuple[str, str, dict[str, Any]]] = []

    response = outer.run(
        "你好",
        "m-1",
        listener=lambda event_type, node, payload: events.append(
            (event_type, node, payload)
        ),
        thread_id="thread_greeting",
        run_id="run_greeting",
    )

    assert response.category_name == "GREETING"
    assert "商家 AI 助手" in response.answer
    assert response.debug_trace["harness"]["coreInvoked"] is False
    assert response.debug_trace["harness"]["preflight"]["route"] == "GREETING"
    assert core.calls == []
    assert [item[0] for item in events] == ["runtime.started", "answer.ready"]
    assert events[-1][1] == "PREFLIGHT_ROUTE"


def test_facade_business_chat_fast_path_works_without_grounded_core() -> None:
    preflight = FakePreflight(
        QuestionRoute.GREETING,
        semantic_route="BUSINESS_CHAT",
    )
    outer = facade(None, preflight=preflight)

    response = outer.run(
        "最近生意是不是越来越难做了",
        "m-1",
        thread_id="thread_chat",
        run_id="run_chat",
    )

    assert response.category_name == "BUSINESS_CHAT"
    assert response.debug_trace["harness"]["runtimePath"] == (
        "preflight_fast_path"
    )
    assert response.debug_trace["harness"]["coreInvoked"] is False


def test_facade_write_request_returns_fast_clarification_without_core() -> None:
    preflight = FakePreflight(
        QuestionRoute.INVALID,
        semantic_route="UNSUPPORTED_WRITE",
        surface_signals={"writeOperation": True},
    )
    outer = facade(None, preflight=preflight)

    response = outer.run(
        "删除昨天的订单",
        "m-1",
        thread_id="thread_write",
        run_id="run_write",
    )

    assert response.category_name == "UNSUPPORTED_WRITE"
    assert response.clarification is not None
    assert response.clarification.type == "write_operation"
    assert response.context is not None
    assert response.context.pending_clarification_type == "write_operation"
    assert response.debug_trace["harness"]["coreInvoked"] is False


def test_facade_preflight_failure_falls_open_to_grounded_core() -> None:
    core = FakeCore()
    outer = facade(core, preflight=FakePreflight(fail=True))
    events: list[tuple[str, str, dict[str, Any]]] = []

    response = outer.run(
        "最近30天订单量",
        "m-1",
        listener=lambda event_type, node, payload: events.append(
            (event_type, node, payload)
        ),
        thread_id="thread_fallback",
        run_id="run_fallback",
    )

    assert response.answer == "42"
    assert len(core.calls) == 1
    assert response.debug_trace["harness"]["coreInvoked"] is True
    assert response.debug_trace["harness"]["preflight"]["status"] == (
        "FAILED_OPEN_TO_CORE"
    )
    assert [item[0] for item in events] == [
        "runtime.started",
        "runtime.preflight_failed",
        "answer.ready",
    ]


def test_facade_marks_message_history_as_full_context_for_preflight() -> None:
    core = FakeCore()
    preflight = FakePreflight(QuestionRoute.BUSINESS)
    outer = facade(core, preflight=preflight)

    outer.run(
        "那昨天呢",
        "m-1",
        thread_id="thread_followup",
        run_id="run_followup",
        message_history=[
            SimpleNamespace(role="user", text="最近7天订单量"),
            SimpleNamespace(role="assistant", text="已完成查询"),
        ],
    )

    assert preflight.calls == [
        {
            "question": "那昨天呢",
            "pendingContext": True,
        }
    ]
    assert len(core.calls) == 1


def test_fastapi_production_entry_uses_runtime_factory() -> None:
    source_path = Path(__file__).resolve().parents[2] / "app/main.py"
    source = source_path.read_text(encoding="utf-8")
    assert "from merchant_ai.services.runtime_factory import create_runtime" in source
    assert "create_workflow" not in source


def test_facade_exposes_neutral_services_without_legacy_execution_fields() -> None:
    outer = facade(FakeCore())

    assert hasattr(outer, "services")
    for forbidden in ("planner", "node_worker", "asset_builder", "graph", "checkpoint_manager"):
        assert not hasattr(outer, forbidden)


def test_deep_agent_timeout_has_shared_sixty_second_floor() -> None:
    settings = get_settings().model_copy(
        update={
            "llm_request_timeout_seconds": 12,
            "llm_lead_timeout_seconds": 20,
            "llm_analysis_timeout_seconds": 30,
        }
    )

    assert runtime_factory._deep_agent_timeout_seconds(settings) == 60


def test_deep_agent_timeout_honors_larger_configured_budget() -> None:
    settings = get_settings().model_copy(
        update={
            "llm_request_timeout_seconds": 12,
            "llm_lead_timeout_seconds": 75,
            "llm_analysis_timeout_seconds": 30,
        }
    )

    assert runtime_factory._deep_agent_timeout_seconds(settings) == 75
