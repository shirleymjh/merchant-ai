from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

from merchant_ai.config import get_settings
from merchant_ai.models import (
    ChatContext,
    ChatResponse,
    MerchantInfo,
    QuestionRoute,
    RoutingDecision,
    UserIdentity,
)
from merchant_ai.services.runtime_factory import (
    GroundedApplicationRuntime,
    RuntimeServices,
)
from merchant_ai.services.security import identity_scope_hash


class _Core:
    def run(self, question: str, merchant_id: str, **_: Any) -> ChatResponse:
        return ChatResponse(
            answer="最近7天订单总数为 42 单。",
            category_name="经营画像",
            doris_tables=["ads_merchant_profile"],
            debug_trace={
                "harness": {
                    "verifiedQueryArtifactIds": ["verified-query-1"],
                    "answerCoverage": {"passed": True},
                }
            },
        )


class _Preflight:
    def understand(self, question: str, pending_context: bool = False) -> Any:
        del question, pending_context
        return SimpleNamespace(
            routing_decision=RoutingDecision(
                route=QuestionRoute.BUSINESS,
                reason="business",
            ),
            semantic_trace={"route": "BUSINESS_TASK"},
            surface_signals={},
        )


class _PendingStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.answer = None

    def put(self, answer: Any) -> None:
        if self.fail:
            raise OSError("pending spool unavailable")
        self.answer = answer


class _AnswerRepository:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.answer = None

    def insert_answer(self, answer: Any, **_: Any) -> bool:
        if self.fail:
            raise RuntimeError("mysql unavailable")
        self.answer = answer
        return True


class _MemoryStore:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.state = None

    def update_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("memory unavailable")
        self.state = state
        return {"written": True}


class _BlockingMemoryStore(_MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def update_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        self.release.wait(timeout=2)
        return super().update_from_state(state)


class _CheckpointManager:
    def deep_agent_ref(self, thread_id: str, run_id: str) -> dict[str, Any]:
        return {"threadId": thread_id, "runId": run_id}

    def close(self) -> None:
        return None


def _runtime(
    *,
    pending_store: _PendingStore,
    answer_repository: _AnswerRepository,
    memory_store: _MemoryStore,
    memory_curator_async: bool = False,
) -> GroundedApplicationRuntime:
    settings = get_settings().model_copy(
        update={
            "merchant_id": "m-1",
            "memory_curator_enabled": True,
            "memory_curator_async": memory_curator_async,
        }
    )
    services = RuntimeServices(
        topic_assets=object(),
        recall_service=object(),
        knowledge_retriever=object(),
        doris_repository=object(),
        access_control=object(),
        merchant_service=SimpleNamespace(
            current_merchant=lambda merchant_id: MerchantInfo(
                merchant_id=merchant_id,
                merchant_name="测试商家",
            )
        ),
        answer_repository=answer_repository,
        pending_store=pending_store,
        keyword_service=object(),
        preflight_understanding=_Preflight(),
        answer_service=object(),
        memory_store=memory_store,
        merchant_profile_store=object(),
        recall_cache_clearers=(),
    )
    return GroundedApplicationRuntime(
        settings=settings,
        core=_Core(),
        services=services,
        checkpoint_manager=_CheckpointManager(),
    )


def test_verified_answer_is_written_to_pending_mysql_and_memory() -> None:
    pending_store = _PendingStore()
    answer_repository = _AnswerRepository()
    memory_store = _MemoryStore()
    runtime = _runtime(
        pending_store=pending_store,
        answer_repository=answer_repository,
        memory_store=memory_store,
    )
    identity = UserIdentity(
        merchant_id="m-1",
        user_id="u-1",
        role="merchant_owner",
        permissions=["chat.run", "memory.write"],
    )

    response = runtime.run(
        "最近7天订单总数",
        "m-1",
        ChatContext(
            days=7,
            metric_keys=["order_cnt"],
            topic="经营画像",
            user_identity=identity,
        ),
        thread_id="thread-persist",
        run_id="run-persist",
    )

    assert response.answer == "最近7天订单总数为 42 单。"
    assert response.persisted is True
    assert pending_store.answer is not None
    assert pending_store.answer.id == response.id
    assert pending_store.answer.thread_id == "thread-persist"
    assert pending_store.answer.identity_scope_hash == identity_scope_hash(
        identity,
        "m-1",
    )
    assert answer_repository.answer.id == response.id
    assert (
        memory_store.state["agent_run_result"].verified_evidence.passed
        is True
    )
    assert response.debug_trace["harness"]["persistence"] == {
        "pendingAnswerWritten": True,
        "answerRepositoryWritten": True,
        "memoryWritten": True,
        "memoryHookStatus": "completed",
        "verifiedQueryArtifactIds": ["verified-query-1"],
        "verifiedRuleArtifactIds": [],
        "feedbackPending": True,
        "runId": "run-persist",
    }


def test_personal_memory_hook_runs_without_verified_query_artifact() -> None:
    pending_store = _PendingStore()
    answer_repository = _AnswerRepository()
    memory_store = _MemoryStore()
    runtime = _runtime(
        pending_store=pending_store,
        answer_repository=answer_repository,
        memory_store=memory_store,
    )
    identity = UserIdentity(
        merchant_id="m-1",
        user_id="u-1",
        role="merchant_owner",
        permissions=["chat.run", "memory.write"],
    )
    response = runtime._finalize_response(
        ChatResponse(answer="好的，我会在个人偏好范围内使用。"),
        question="我平时更喜欢按周展示趋势",
        merchant_id="m-1",
        context=ChatContext(user_identity=identity),
        listener=None,
        thread_id="thread-preference",
        run_id="run-preference",
        preflight=SimpleNamespace(surface_signals={}),
        core_invoked=False,
        message_history=[],
    )

    assert memory_store.state is not None
    assert memory_store.state["question"] == "我平时更喜欢按周展示趋势"
    assert memory_store.state["message_history"] == []
    assert response.debug_trace["harness"]["personalMemoryHook"] == {
        "status": "completed",
        "written": True,
        "outcome": "stored",
        "mode": "inline",
        "refreshPolicy": "EVERY_INTERACTION",
    }


def test_next_turn_waits_for_same_users_pending_background_memory_hook() -> None:
    memory_store = _BlockingMemoryStore()
    runtime = _runtime(
        pending_store=_PendingStore(),
        answer_repository=_AnswerRepository(),
        memory_store=memory_store,
        memory_curator_async=True,
    )
    identity = UserIdentity(
        merchant_id="m-1",
        user_id="u-1",
        role="merchant_owner",
        permissions=["chat.run", "memory.write"],
    )
    context = ChatContext(user_identity=identity)
    response = runtime._finalize_response(
        ChatResponse(answer="好的。"),
        question="我平时更喜欢按周展示趋势",
        merchant_id="m-1",
        context=context,
        listener=None,
        thread_id="thread-first",
        run_id="run-first",
        preflight=SimpleNamespace(surface_signals={}),
        core_invoked=False,
        message_history=[],
    )

    assert response.debug_trace["harness"]["personalMemoryHook"]["status"] == "scheduled"
    assert memory_store.started.wait(timeout=1)
    awaited = Event()

    def await_hook() -> None:
        runtime._await_pending_personal_memory_hooks(
            "m-1",
            context,
            listener=None,
        )
        awaited.set()

    waiter = Thread(target=await_hook)
    waiter.start()
    assert not awaited.wait(timeout=0.05)
    memory_store.release.set()
    assert awaited.wait(timeout=1)
    waiter.join(timeout=1)
    assert memory_store.state is not None
    runtime.close()


def test_persistence_failure_does_not_replace_verified_answer() -> None:
    runtime = _runtime(
        pending_store=_PendingStore(fail=True),
        answer_repository=_AnswerRepository(fail=True),
        memory_store=_MemoryStore(fail=True),
    )

    response = runtime.run(
        "最近7天订单总数",
        "m-1",
        ChatContext(),
        thread_id="thread-degraded",
        run_id="run-degraded",
    )

    assert response.answer == "最近7天订单总数为 42 单。"
    assert response.persisted is False
    harness = response.debug_trace["harness"]
    assert "pendingAnswerPersistenceError" in harness
    assert "answerRepositoryPersistenceError" in harness
    assert harness["persistence"]["feedbackPending"] is False
