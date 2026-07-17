from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from merchant_ai.config import get_settings
from merchant_ai.models import ChatContext, ChatResponse, MerchantInfo, UserIdentity
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


def facade(core: FakeCore) -> GroundedApplicationRuntime:
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
            answer_service=object(),
            memory_store=object(),
            merchant_profile_store=object(),
            recall_cache_clearers=(),
        ),
        checkpoint_manager=FakeCheckpointManager(),
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


def test_fastapi_production_entry_uses_runtime_factory() -> None:
    source = Path("python_backend/app/main.py").read_text(encoding="utf-8")
    assert "from merchant_ai.services.runtime_factory import create_runtime" in source
    assert "create_workflow" not in source


def test_facade_exposes_neutral_services_without_legacy_execution_fields() -> None:
    outer = facade(FakeCore())

    assert hasattr(outer, "services")
    for forbidden in ("planner", "node_worker", "asset_builder", "graph", "checkpoint_manager"):
        assert not hasattr(outer, forbidden)
