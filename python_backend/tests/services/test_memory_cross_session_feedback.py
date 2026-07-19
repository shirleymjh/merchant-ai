from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import PendingAnswer, UserIdentity
from merchant_ai.services.answer import FeedbackService
from merchant_ai.services.memory import StructuredMemoryStore
from merchant_ai.services.repositories import PendingAnswerStore
from merchant_ai.services.security import identity_scope_hash


class _AnswerRepository:
    def insert_answer(self, pending, adopted=False, liked=False, disliked=False):
        return True

    def update_feedback(self, answer_id, adopted, liked, disliked):
        return None


def _identity(user_id: str) -> UserIdentity:
    return UserIdentity(
        user_id=user_id,
        merchant_id="seller_100",
        role="merchant_operator",
        store_ids=["S1"],
        permissions=["memory.read"],
    )


def _pending(answer_id: str, identity: UserIdentity) -> PendingAnswer:
    return PendingAnswer(
        id=answer_id,
        question="最近7天退款情况怎么样",
        answer="最近7天退款趋势已核验。",
        merchant_id="seller_100",
        merchant_name="merchant",
        category_name="REFUND",
        doris_tables="refund_table",
        suggested_questions="[]",
        thread_id="source_thread",
        user_id=identity.user_id,
        identity_scope_hash=identity_scope_hash(identity, identity.merchant_id),
        store_ids=identity.store_ids,
        permissions=identity.permissions,
    )


def _recall_state(identity: UserIdentity):
    return {
        "question": "再看最近7天退款情况",
        "requested_merchant_id": "seller_100",
        "user_identity": identity.model_dump(by_alias=True),
        "access_role": "merchant_analyst",
        "memory_eval_context": {"topics": ["REFUND"], "timeWindows": [7]},
    }


def test_feedback_attribution_survives_store_recreation_and_is_recalled_in_new_session(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "file",
            "memory_query_understanding_enabled": False,
        }
    )
    identity = _identity("user_a")
    first_process = PendingAnswerStore(settings)
    first_process.put(_pending("answer_restart", identity))

    restarted_process = PendingAnswerStore(settings)
    restored = restarted_process.get("answer_restart")
    assert restored is not None
    assert restored.user_id == "user_a"
    assert restored.thread_id == "source_thread"

    memory_store = StructuredMemoryStore(settings)
    feedback = FeedbackService(_AnswerRepository(), restarted_process, memory_store)
    assert feedback.apply_feedback(
        "answer_restart",
        adopted=True,
        liked=True,
        disliked=False,
        identity=identity,
    ) is True

    new_session_store = StructuredMemoryStore(settings)
    same_principal = new_session_store.select_for_question(_recall_state(identity), budget_chars=2400)
    assert same_principal["memoryInjectionTrace"]["selectedIds"]
    injected_personal_memory = [
        *same_principal["relevantEvents"],
        *same_principal["relevantPreferences"],
    ]
    assert any(item["scope"]["userId"] == "user_a" for item in injected_personal_memory)

    other_principal = new_session_store.select_for_question(
        _recall_state(_identity("user_b")),
        budget_chars=2400,
    )
    assert other_principal["memoryInjectionTrace"]["selectedIds"] == []
    assert other_principal["relevantEvents"] == []
    assert other_principal["relevantPreferences"] == []


def test_feedback_dedup_and_negative_mutation_are_principal_scoped(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "file",
            "memory_query_understanding_enabled": False,
        }
    )
    store = StructuredMemoryStore(settings)
    pending_store = PendingAnswerStore(settings)
    service = FeedbackService(_AnswerRepository(), pending_store, store)
    user_a = _identity("user_a")
    user_b = _identity("user_b")

    pending_store.put(_pending("answer_a_like", user_a))
    pending_store.put(_pending("answer_b_like", user_b))
    assert service.apply_feedback("answer_a_like", True, True, False, identity=user_a)
    assert service.apply_feedback("answer_b_like", True, True, False, identity=user_b)

    before_dislike = store.load("seller_100")
    feedback_events = [item for item in before_dislike["events"] if item.get("source") == "feedback"]
    assert {item["scope"]["userId"] for item in feedback_events} == {"user_a", "user_b"}
    preference_b = next(
        item
        for item in before_dislike["preferences"]
        if item.get("scope", {}).get("userId") == "user_b" and item.get("topics") == ["REFUND"]
    )
    confidence_b = preference_b["confidence"]

    pending_store.put(_pending("answer_a_dislike", user_a))
    assert service.apply_feedback("answer_a_dislike", False, False, True, identity=user_a)

    after_dislike = store.load("seller_100")
    preference_b_after = next(
        item
        for item in after_dislike["preferences"]
        if item.get("scope", {}).get("userId") == "user_b" and item.get("topics") == ["REFUND"]
    )
    assert preference_b_after["confidence"] == confidence_b
