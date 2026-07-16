from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import (
    append_memory_recall_outcomes_to_run_result,
    create_workflow,
)
from merchant_ai.models import AgentRunResult, ChatContext, QuestionRoute, RoutingDecision
from merchant_ai.services.memory import EnterpriseMemoryStore, StructuredMemoryStore
from merchant_ai.services.middleware import MemoryMiddleware


def memory_settings(tmp_path, backend: str = "file"):
    return get_settings().model_copy(
        update={
            "agent_checkpointer_backend": "memory",
            "cache_enabled": False,
            "harness_workspace_path": str(tmp_path),
            "llm_api_key": "",
            "memory_backend": backend,
            "memory_redis_enabled": False,
            "memory_vector_enabled": False,
        }
    )


def business_memory_state(workflow, suffix: str):
    state = workflow._initial_state(
        "最近7天退款率怎么样",
        "seller_100",
        ChatContext(),
        None,
        "thread_%s" % suffix,
        "run_%s" % suffix,
    )
    state["routing_decision"] = RoutingDecision(route=QuestionRoute.BUSINESS)
    state["topic_routed"] = True
    return state


def correction_injection():
    return {
        "merchantId": "seller_100",
        "source": "test",
        "relevantCorrections": [
            {
                "id": "memory_refund_rate",
                "memoryType": "correction",
                "correctionText": "退款率按退款订单数除以支付订单数计算",
                "metrics": ["refund_rate"],
                "topics": ["REFUND"],
                "confidence": 0.95,
                "status": "approved",
            }
        ],
        "memoryInjectionTrace": {
            "candidateCount": 1,
            "selectedIds": ["memory_refund_rate"],
        },
    }


class EmptyStore:
    def select_for_question(self, *_args, **_kwargs):
        return {
            "merchantId": "seller_100",
            "source": "test",
            "memoryInjectionTrace": {"candidateCount": 0, "selectedIds": []},
        }

    def render_injection(self, _payload):
        return ""


class CorrectionStore:
    def __init__(self, render_fails: bool = False):
        self.render_fails = render_fails

    def select_for_question(self, *_args, **_kwargs):
        return correction_injection()

    def render_injection(self, _payload):
        if self.render_fails:
            raise RuntimeError("injected renderer outage")
        return "governed correction context"


def test_successful_empty_memory_recall_is_not_an_operational_failure(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    workflow.memory_store = EmptyStore()

    state = workflow.recall_memory(business_memory_state(workflow, "empty"))

    assert state["memory_recalled"] is True
    assert state["memory_recall_status"] == "empty"
    assert state["memory_injection_trace"]["usableSnapshot"] is True
    assert state["memory_recall_issues"] == []
    assert state["memory_constraints"] == []


def test_backend_failure_is_failed_not_empty_and_reaches_evidence(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))

    class FailedStore:
        def select_for_question(self, *_args, **_kwargs):
            raise RuntimeError("injected memory backend outage")

    workflow.memory_store = FailedStore()
    state = workflow.recall_memory(business_memory_state(workflow, "failed"))

    assert state["memory_recall_status"] == "failed"
    assert state["memory_injection_trace"]["usableSnapshot"] is False
    assert state["memory_recall_issues"][0]["code"] == "MEMORY_RECALL_FAILED"
    assert state["memory_constraints"] == []

    run_result = AgentRunResult()
    append_memory_recall_outcomes_to_run_result(state, run_result)

    assert run_result.evidence_gaps[0].code == "MEMORY_RECALL_FAILED"
    assert run_result.evidence_gaps[0].severity == "blocking"
    assert run_result.evidence_gaps[0].disclosure_required is True
    assert run_result.degraded_reasons[0]["stage"] == "memory_recall"

    state["agent_run_result"] = run_result
    response = workflow.to_response(state)
    memory_debug = response.debug_trace["harness"]["memory"]["retrieval"]
    assert memory_debug["status"] == "failed"
    assert memory_debug["issues"][0]["code"] == "MEMORY_RECALL_FAILED"
    assert response.debug_trace["evidenceGaps"][0]["code"] == "MEMORY_RECALL_FAILED"


def test_memory_failure_projection_is_idempotent_across_answer_boundaries(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    state = business_memory_state(workflow, "projection_idempotency")
    state["memory_recall_status"] = "failed"
    state["memory_injection_trace"] = {
        "status": "failed",
        "usableSnapshot": False,
        "issues": [
            {
                "code": "MEMORY_RECALL_FAILED",
                "message": "injected memory backend outage",
                "backend": "primary",
                "lane": "primary",
                "stage": "acquire",
                "severity": "blocking",
                "resolved": False,
            }
        ],
    }
    run_result = AgentRunResult()

    append_memory_recall_outcomes_to_run_result(state, run_result)
    append_memory_recall_outcomes_to_run_result(state, run_result)
    append_memory_recall_outcomes_to_run_result(state, run_result)

    assert [item["code"] for item in run_result.degraded_reasons] == ["MEMORY_RECALL_FAILED"]
    assert [item.code for item in run_result.evidence_gaps] == ["MEMORY_RECALL_FAILED"]


def test_failed_memory_recall_cannot_short_circuit_into_fast_answer(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    state = business_memory_state(workflow, "fast_block")
    state["memory_recalled"] = True
    state["memory_recall_status"] = "failed"
    state["memory_injection_trace"] = {"status": "failed", "usableSnapshot": False}

    state = workflow.try_fast_metric(state)

    assert state["fast_metric_attempted"] is True
    assert state["fast_metric_completed"] is False
    assert state["fast_metric_response"] is None
    assert state["answer"] == ""
    assert state["last_action_result"].status != "success"


def test_optional_profile_summary_failure_preserves_snapshot_and_constraints(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    workflow.memory_store = CorrectionStore()

    class FailedSummary:
        def summarize(self, **_kwargs):
            raise RuntimeError("injected profile summary outage")

    workflow.merchant_profile_summary_service = FailedSummary()
    state = workflow.recall_memory(business_memory_state(workflow, "profile_summary"))

    assert state["memory_recall_status"] == "degraded"
    assert state["memory_injection_trace"]["selectedIds"] == ["memory_refund_rate"]
    assert state["memory_injection"]["relevantCorrections"][0]["id"] == "memory_refund_rate"
    assert state["memory_constraints"][0]["enforcement"] == "required"
    assert state["memory_injection_trace"]["enrichmentStatus"]["profileSummary"] == "failed"
    assert any(item["code"] == "MEMORY_PROFILE_SUMMARY_FAILED" for item in state["memory_recall_issues"])


def test_constraint_compilation_failure_fails_closed_but_keeps_raw_snapshot(monkeypatch, tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    workflow.memory_store = CorrectionStore()

    def fail_constraints(_injection):
        raise RuntimeError("injected constraint compiler outage")

    monkeypatch.setattr("merchant_ai.graph.workflow.build_memory_constraints", fail_constraints)
    state = workflow.recall_memory(business_memory_state(workflow, "constraints"))

    assert state["memory_recall_status"] == "failed"
    assert state["memory_injection"] == {}
    assert state["memory_constraints"] == []
    assert state["memory_injection_raw_snapshot"]["relevantCorrections"][0]["id"] == "memory_refund_rate"
    assert state["memory_injection_trace"]["selectedIds"] == ["memory_refund_rate"]
    assert state["memory_recall_issues"][-1]["code"] == "MEMORY_CONSTRAINT_COMPILATION_FAILED"


def test_optional_renderer_failure_preserves_snapshot_and_constraints(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    workflow.memory_store = CorrectionStore(render_fails=True)

    state = workflow.recall_memory(business_memory_state(workflow, "renderer"))

    assert state["memory_recall_status"] == "degraded"
    assert state["memory_injection_trace"]["selectedIds"] == ["memory_refund_rate"]
    assert state["memory_constraints"][0]["id"] == "memory_refund_rate"
    assert state["memory_context"] == ""
    assert state["memory_injection_trace"]["enrichmentStatus"]["render"] == "failed"
    assert any(item["code"] == "MEMORY_RENDER_FAILED" for item in state["memory_recall_issues"])


def test_fast_understand_summary_refresh_failure_keeps_recalled_contract(tmp_path):
    workflow = create_workflow(memory_settings(tmp_path))
    workflow.memory_store = CorrectionStore()
    state = workflow.recall_memory(business_memory_state(workflow, "fast_summary"))

    class FailedSummary:
        def summarize(self, **_kwargs):
            raise RuntimeError("injected refresh outage")

    workflow.merchant_profile_summary_service = FailedSummary()
    state = workflow.fast_understand(state)

    assert state["fast_understood"] is True
    assert state["memory_recall_status"] == "degraded"
    assert state["memory_constraints"][0]["id"] == "memory_refund_rate"
    assert state["memory_injection_trace"]["selectedIds"] == ["memory_refund_rate"]
    assert any(
        item["stage"] == "fast_understand_profile_summary"
        for item in state["memory_recall_issues"]
    )


class FakeRepository:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def load_memory(self, merchant_id):
        if self.fail:
            raise RuntimeError("injected primary outage")
        return {
            "merchantId": merchant_id,
            "events": [],
            "preferences": [],
            "facts": [],
            "recentFocus": {},
        }

    def apply_hit_deltas(self, _deltas):
        return 0


class FakeCache:
    def __init__(self, cached=None):
        self.cached = cached

    def get_json(self, _key):
        return self.cached

    def set_json(self, _key, _value):
        return None

    def backend_name(self):
        return "fake_cache"


class FakeVector:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def search(self, _merchant_id, _query):
        if self.fail:
            raise RuntimeError("injected vector outage")
        return []


def test_enterprise_primary_fallback_is_degraded_not_empty(tmp_path):
    settings = memory_settings(tmp_path, backend="es")
    store = EnterpriseMemoryStore(
        settings,
        repository=FakeRepository(fail=True),
        hot_cache=FakeCache(),
        vector_index=FakeVector(),
        fallback_store=StructuredMemoryStore(settings),
    )

    selected = store.select_for_question(
        {"question": "最近7天退款率", "requested_merchant_id": "seller_100"},
        budget_tokens=400,
    )
    trace = selected["memoryInjectionTrace"]

    assert trace["status"] == "degraded"
    assert trace["usableSnapshot"] is True
    assert trace["issues"][0]["code"] == "MEMORY_PRIMARY_BACKEND_FAILED"
    assert trace["issues"][0]["fallbackUsed"] is True
    assert trace["issues"][0]["resolved"] is True


def test_vector_lane_failure_is_degraded_instead_of_zero_match(tmp_path):
    settings = memory_settings(tmp_path, backend="es").model_copy(update={"memory_vector_enabled": True})
    store = EnterpriseMemoryStore(
        settings,
        repository=FakeRepository(),
        hot_cache=FakeCache(),
        vector_index=FakeVector(fail=True),
        fallback_store=StructuredMemoryStore(settings),
    )

    selected = store.select_for_question(
        {"question": "最近7天退款率", "requested_merchant_id": "seller_100"},
        budget_tokens=400,
    )
    trace = selected["memoryInjectionTrace"]

    assert trace["status"] == "degraded"
    assert trace["issues"][0]["code"] == "MEMORY_VECTOR_RETRIEVAL_FAILED"


def test_usage_side_effect_failure_does_not_discard_cached_recall(tmp_path):
    settings = memory_settings(tmp_path, backend="es")

    class UsageFailStore(EnterpriseMemoryStore):
        def record_usage(self, _merchant_id, _memory_ids):
            raise RuntimeError("injected usage write outage")

    store = UsageFailStore(
        settings,
        repository=FakeRepository(),
        hot_cache=FakeCache(cached=correction_injection()),
        vector_index=FakeVector(),
        fallback_store=StructuredMemoryStore(settings),
    )

    selected = store.select_for_question(
        {"question": "最近7天退款率", "requested_merchant_id": "seller_100"},
        budget_tokens=400,
    )
    trace = selected["memoryInjectionTrace"]

    assert trace["status"] == "degraded"
    assert trace["selectedIds"] == ["memory_refund_rate"]
    assert selected["relevantCorrections"][0]["id"] == "memory_refund_rate"
    assert trace["issues"][0]["code"] == "MEMORY_USAGE_RECORD_FAILED"
    assert trace["issues"][0]["details"]["answerImpact"] is False


def test_locked_failed_snapshot_is_not_reported_ready_by_middleware(tmp_path):
    middleware = MemoryMiddleware(memory_settings(tmp_path))

    class NeverCalledStore:
        def select_for_question(self, *_args, **_kwargs):
            raise AssertionError("locked failed snapshot must not be reported as a usable retry result")

        def render_injection(self, _payload):
            raise AssertionError("failed snapshot must not be rendered")

    middleware.memory_store = NeverCalledStore()
    state = {
        "topic_routed": True,
        "memory_recalled": True,
        "_memory_snapshot_locked": True,
        "memory_injection": {},
        "memory_injection_trace": {
            "status": "failed",
            "usableSnapshot": False,
            "issues": [
                {
                    "code": "MEMORY_RECALL_FAILED",
                    "message": "backend unavailable",
                    "severity": "blocking",
                    "resolved": False,
                }
            ],
        },
        "memory_recall_status": "failed",
        "memory_recall_issues": [],
        "memory_constraints": [],
        "middleware_events": [],
    }

    result = middleware.before_policy(state)

    assert result["_memory_middleware_snapshot_ready"] is False
    assert result["middleware_events"][-1].code == "MEMORY_REQUEST_SNAPSHOT_UNAVAILABLE"
