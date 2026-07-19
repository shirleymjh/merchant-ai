from types import SimpleNamespace

from merchant_ai.config import get_settings
from merchant_ai.models import AnswerMode, QueryPlan, QuestionIntent
from merchant_ai.services.memory import (
    KnowledgeCuratorService,
    KnowledgeSuggestionGovernanceService,
    MemoryIngestionService,
    MemoryWriteGate,
    StructuredMemoryStore,
    default_memory_status,
    filter_memory_items,
    is_metric_definition_dispute,
    knowledge_suggestion_proposed_scope,
    memory_item_counts,
)


class _Store:
    def __init__(self, payload):
        self.payload = payload

    def load(self, merchant_id):
        return self.payload

    def save(self, merchant_id, payload):
        self.payload = payload
        return payload


class _CuratorLlm:
    configured = True

    def tool_json_chat(self, *args, **kwargs):
        return {
            "shouldExtract": True,
            "reason": "explicit personal preference",
            "candidates": [
                {
                    "kind": "personal_preference",
                    "scope": "personal",
                    "title": "区域关注偏好",
                    "content": "我只关注新加坡和印尼的指标。",
                    "evidenceQuote": "我只关注新加坡和印尼的指标",
                    "topic": "",
                    "metricName": "",
                    "aliases": [],
                    "confidence": 0.96,
                    "reason": "the user explicitly stated an individual focus preference",
                }
            ],
        }


class _MixedCuratorLlm:
    configured = True

    def tool_json_chat(self, *args, **kwargs):
        return {
            "shouldExtract": True,
            "reason": "the turn contains independent knowledge and personal signals",
            "candidates": [
                {
                    "kind": "merchant_rule",
                    "scope": "merchant",
                    "title": "售后风险关注规则",
                    "content": "本店售后风险不按退款金额判断。",
                    "evidenceQuote": "不对，这次不是退款金额",
                    "topic": "",
                    "metricName": "refund_rate",
                    "aliases": [],
                    "confidence": 0.94,
                    "reason": "explicit merchant correction",
                },
                {
                    "kind": "personal_preference",
                    "scope": "personal",
                    "title": "售后风险指标偏好",
                    "content": "以后看售后风险默认按退款率。",
                    "evidenceQuote": "以后看售后风险默认按退款率",
                    "topic": "",
                    "metricName": "refund_rate",
                    "aliases": [],
                    "confidence": 0.96,
                    "reason": "explicit personal preference",
                },
            ],
        }


def _governance_service(payload):
    service = object.__new__(KnowledgeSuggestionGovernanceService)
    service.memory_store = _Store(payload)
    return service


def test_personal_memory_uses_automatic_quarantine_not_human_review():
    gate = MemoryWriteGate()
    event = {
        "eventId": "memory_gap",
        "memoryType": "business_focus",
        "question": "最近关注客服工单趋势",
        "answerPreview": "本轮数据不完整",
        "confidence": 0.65,
        "scope": {"merchantId": "seller_1"},
        "answerEvidenceChecked": True,
        "answerVerified": False,
        "answerWithGap": True,
    }

    policy = gate.evaluate(event, "seller_1")
    stored = gate.apply(dict(event), "seller_1")

    assert policy["action"] == "quarantine"
    assert policy["allowed"] is True
    assert policy["humanReviewRequired"] is False
    assert stored["status"] == "quarantined"
    assert stored["reviewStatus"] == "evidence_gap"


def test_metric_corrections_are_knowledge_sources_not_active_personal_memory():
    assert default_memory_status("correction", source="answer_run") == "quarantined"
    assert default_memory_status("metric_dispute", source="answer_run") == "quarantined"
    assert default_memory_status("business_focus", source="answer_run") == "active"


def test_curator_routes_personal_preference_directly_to_memory(tmp_path):
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_curator_enabled": True,
            "memory_curator_min_confidence": 0.7,
        }
    )
    curator = KnowledgeCuratorService(settings, llm=_CuratorLlm())
    event = {
        "eventId": "preference_turn",
        "memoryType": "business_focus",
        "question": "请记住，我只关注新加坡和印尼的指标。",
        "topics": [],
        "metrics": [],
        "confidence": 0.8,
        "scope": {"merchantId": "seller_1", "userId": "user_1"},
    }

    suggestions, trace = curator.extract(
        {"question": event["question"], "message_history": []},
        event,
    )

    assert suggestions == []
    assert trace["candidateCount"] == 0
    assert trace["personalMemoryCandidateCount"] == 1
    preference = trace["personalMemoryCandidates"][0]
    assert preference["status"] == "active"
    assert preference["reviewStatus"] == "auto"
    assert preference["writePolicy"]["humanReviewRequired"] is False
    assert preference["scope"]["userId"] == "user_1"


def test_derived_metric_selection_is_not_itself_a_definition_dispute() -> None:
    plan = QueryPlan(intents=[QuestionIntent(answer_mode=AnswerMode.DERIVED)])

    assert not is_metric_definition_dispute(
        "不对，这次不是退款金额，是退款率。",
        ["refund_rate"],
        plan=plan,
    )
    assert is_metric_definition_dispute(
        "退款率应该用退款单数 / 订单数。",
        ["refund_rate"],
        plan=plan,
    )


def test_mixed_correction_keeps_independent_curated_personal_preference(tmp_path) -> None:
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "memory_backend": "file",
            "memory_curator_enabled": True,
        }
    )
    store = StructuredMemoryStore(settings)
    store.ingestion_service = MemoryIngestionService(
        settings,
        curator=KnowledgeCuratorService(settings, llm=_MixedCuratorLlm()),
    )

    memory = store.update_from_state(
        {
            "question": "不对，这次不是退款金额；以后看售后风险默认按退款率。",
            "requested_merchant_id": "seller_1",
            "plan": QueryPlan(
                intents=[
                    QuestionIntent(
                        answer_mode=AnswerMode.DERIVED,
                        metric_resolution={"metricKey": "refund_rate"},
                    )
                ]
            ),
            "answer": "已分别记录知识候选与个人偏好。",
        }
    )

    assert memory["events"][-1]["memoryType"] == "correction"
    assert memory["events"][-1]["status"] == "quarantined"
    assert memory["knowledgeSuggestions"]
    preference = next(
        item
        for item in memory["preferences"]
        if item["memoryType"] == "user_preference"
    )
    assert preference["memoryTier"] == "core"
    assert preference["status"] == "active"
    assert "退款率" in preference["value"]


def test_quarantined_memory_is_hidden_from_active_management_view():
    payload = {
        "merchantId": "seller_1",
        "events": [
            {"eventId": "active_1", "status": "active", "question": "关注工单"},
            {"eventId": "quarantine_1", "status": "quarantined", "question": "未验证结论"},
        ],
        "preferences": [],
        "facts": [],
    }

    active = filter_memory_items(payload, active_only=True)
    counts = memory_item_counts(payload)

    assert [item["eventId"] for item in active["events"]] == ["active_1"]
    assert counts["active"] == 1
    assert counts["quarantined"] == 1


def test_private_or_unclassified_knowledge_cannot_enter_shared_publish_flow():
    payload = {
        "merchantId": "seller_1",
        "knowledgeSuggestions": [
            {
                "suggestionId": "merchant_rule",
                "status": "approved",
                "scopeType": "merchant",
                "payload": {"proposedScope": "merchant"},
            },
            {
                "suggestionId": "legacy_rule",
                "status": "approved",
                "payload": {},
            },
        ],
    }
    service = _governance_service(payload)

    assert knowledge_suggestion_proposed_scope(payload["knowledgeSuggestions"][0]) == "merchant"
    assert knowledge_suggestion_proposed_scope(payload["knowledgeSuggestions"][1]) == "legacy_unclassified"
    assert service.review_suggestion(
        "seller_1",
        "merchant_rule",
        SimpleNamespace(action="approve", approved=True, reviewer="ops", review_note=""),
    )["status"] == "PRIVATE_MEMORY_NOT_PUBLISHABLE"
    assert service.request_publish_suggestion("seller_1", "merchant_rule")["status"] == "PRIVATE_MEMORY_NOT_PUBLISHABLE"
    assert service.publish_suggestion("seller_1", "merchant_rule")["status"] == "PRIVATE_MEMORY_NOT_PUBLISHABLE"
    assert service.mark_suggestion_indexed("seller_1", "merchant_rule")["status"] == "PRIVATE_MEMORY_NOT_PUBLISHABLE"
    assert service.publish_suggestion("seller_1", "legacy_rule")["status"] == "KNOWLEDGE_SCOPE_CLASSIFICATION_REQUIRED"


def test_publish_job_never_processes_private_or_unclassified_candidates():
    payload = {
        "merchantId": "seller_1",
        "knowledgeSuggestions": [
            {"suggestionId": "merchant_rule", "status": "publish_requested", "scopeType": "merchant"},
            {"suggestionId": "legacy_rule", "status": "publish_requested"},
        ],
    }
    service = _governance_service(payload)

    result = service.run_publish_jobs("seller_1")

    assert result["success"] is True
    assert result["queuedCount"] == 0
    assert result["processedCount"] == 0
    assert result["skippedScopeCount"] == 2
