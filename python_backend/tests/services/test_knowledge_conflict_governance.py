from copy import deepcopy

from merchant_ai.config import get_settings
from merchant_ai.models import KnowledgeSuggestionReviewRequest
from merchant_ai.services.assets import apply_resolved_knowledge_conflicts_to_asset
from merchant_ai.services.memory import (
    KnowledgeConflictService,
    KnowledgeSuggestionGovernanceService,
    StructuredMemoryStore,
)


class _ConflictLlm:
    configured = True

    def tool_json_chat(self, _system, _user, _schema, _context, timeout_seconds):
        assert timeout_seconds > 0
        return {
            "assessments": [
                {
                    "existingKnowledgeId": "refund-rate-rule",
                    "relation": "conflict",
                    "confidence": 0.97,
                    "reason": "同一指标在相同范围内使用了不同分子和分母",
                    "recommendedAction": "review",
                    "mergedContent": "",
                }
            ]
        }


class _TopicAssets:
    def all_topic_names(self):
        return ["交易"]

    def load_manifest(self, topic):
        assert topic == "交易"
        return [{"tableName": "fact_trade"}]

    def load_table_asset(self, topic, table):
        assert (topic, table) == ("交易", "fact_trade")
        return {
            "metrics": [],
            "terms": [],
            "knowledgeRules": [
                {
                    "ruleId": "refund-rate-rule",
                    "title": "退款率口径",
                    "content": "退款率按退款订单数除以支付订单数计算",
                    "alwaysApply": True,
                }
            ],
        }


class _StaticConflictService:
    def __init__(self, report):
        self.report = report

    def check(self, merchant_id, suggestion, memory):
        assert merchant_id == "seller_1"
        assert suggestion["suggestionId"] == "ks_1"
        assert isinstance(memory, dict)
        return deepcopy(self.report)


def _settings(tmp_path):
    return get_settings().model_copy(
        update={
            "memory_backend": "file",
            "harness_workspace_path": str(tmp_path / "workspace"),
            "topic_path": str(tmp_path / "topics"),
            "rule_knowledge_path": str(tmp_path / "rules"),
            "es_enabled": False,
            "knowledge_conflict_enabled": True,
        }
    )


def _platform_suggestion(status="candidate"):
    return {
        "suggestionId": "ks_1",
        "suggestionType": "business_rule",
        "status": status,
        "scopeType": "platform",
        "topic": "交易",
        "sourceTable": "fact_trade",
        "metricName": "退款率口径",
        "payload": {
            "proposedScope": "platform",
            "correctionText": "退款率按退款金额除以支付金额计算",
        },
    }


def _conflict_report(report_id="conflict-report-1"):
    return {
        "enabled": True,
        "status": "confirmation_required",
        "reportId": report_id,
        "candidateText": "退款率按退款金额除以支付金额计算",
        "candidateScope": "platform",
        "resolutionOptions": ["replace", "merge", "cancel"],
        "matches": [
            {
                "existingKnowledgeId": "refund-rate-rule",
                "existingText": "退款率按退款订单数除以支付订单数计算",
                "title": "退款率口径",
                "scope": "platform",
                "kind": "rule",
                "topic": "交易",
                "tableName": "fact_trade",
                "relation": "conflict",
                "confidence": 0.97,
                "mergedContent": "",
            }
        ],
    }


def _governance(tmp_path, status="candidate"):
    settings = _settings(tmp_path)
    store = StructuredMemoryStore(settings)
    memory = store.empty_memory("seller_1")
    memory["knowledgeSuggestions"] = [_platform_suggestion(status)]
    store.save("seller_1", memory)
    service = KnowledgeSuggestionGovernanceService(settings, memory_store=store)
    service.conflict_service = _StaticConflictService(_conflict_report())
    return service, store


def test_two_stage_check_uses_public_candidates_and_structured_judgement(tmp_path):
    settings = _settings(tmp_path)
    service = KnowledgeConflictService(
        settings,
        llm=_ConflictLlm(),
        topic_assets=_TopicAssets(),
    )
    suggestion = _platform_suggestion()
    memory = {
        "preferences": [
            {
                "preferenceId": "private-pref",
                "status": "active",
                "value": "退款率按退款金额除以支付金额计算",
            }
        ],
        "facts": [],
        "knowledgeSuggestions": [suggestion],
    }

    report = service.check("seller_1", suggestion, memory)

    assert report["status"] == "confirmation_required"
    assert report["reportId"]
    assert report["resolutionOptions"] == ["replace", "merge", "cancel"]
    assert [item["existingKnowledgeId"] for item in report["matches"]] == [
        "refund-rate-rule"
    ]
    assert report["matches"][0]["relation"] == "conflict"
    assert not any(
        item["existingKnowledgeId"] == "private-pref"
        for item in report["matches"]
    )


def test_operator_approval_is_blocked_until_current_conflict_report_is_resolved(
    tmp_path,
):
    service, store = _governance(tmp_path)

    blocked = service.review_suggestion(
        "seller_1",
        "ks_1",
        KnowledgeSuggestionReviewRequest(
            approved=True,
            action="approve",
            reviewer="ops",
            review_note="reviewing conflict",
        ),
    )

    assert blocked["success"] is False
    assert blocked["status"] == "CONFLICT_CONFIRMATION_REQUIRED"
    assert blocked["allowedResolutions"] == [
        "replace",
        "merge",
        "cancel",
    ]
    persisted = store.load("seller_1")["knowledgeSuggestions"][0]
    assert persisted["status"] == "candidate"
    assert persisted["conflictReviewStatus"] == "required"
    assert persisted["payload"]["conflictCheck"]["reportId"] == "conflict-report-1"

    approved = service.review_suggestion(
        "seller_1",
        "ks_1",
        KnowledgeSuggestionReviewRequest(
            approved=True,
            action="approve",
            reviewer="ops",
            review_note="merge the two rules",
            conflict_resolution="merge",
            conflict_report_id="conflict-report-1",
            merged_content="退款率同时保留订单口径，并新增退款金额占比作为独立指标",
        ),
    )

    assert approved["success"] is True
    assert approved["status"] == "approved"
    assert approved["suggestion"]["conflictReviewStatus"] == "resolved"
    assert (
        approved["suggestion"]["payload"]["correctionText"]
        == "退款率同时保留订单口径，并新增退款金额占比作为独立指标"
    )


def test_stale_conflict_report_cannot_be_used_for_approval(tmp_path):
    service, _store = _governance(tmp_path)

    result = service.review_suggestion(
        "seller_1",
        "ks_1",
        KnowledgeSuggestionReviewRequest(
            approved=True,
            reviewer="ops",
            conflict_resolution="keep_both",
            conflict_report_id="old-report",
        ),
    )

    assert result["success"] is False
    assert result["status"] == "STALE_CONFLICT_REPORT"
    assert result["expectedConflictReportId"] == "conflict-report-1"


def test_direct_publish_is_fail_closed_when_conflict_is_unresolved(tmp_path):
    service, store = _governance(tmp_path, status="approved")

    result = service.publish_suggestion("seller_1", "ks_1", reviewer="ops")

    assert result["success"] is False
    assert result["status"] == "CONFLICT_CONFIRMATION_REQUIRED"
    persisted = store.load("seller_1")["knowledgeSuggestions"][0]
    assert persisted["status"] == "approved"
    assert persisted["conflictReviewStatus"] == "required"


def test_publish_job_does_not_index_a_conflict_blocked_candidate(tmp_path):
    service, _store = _governance(tmp_path, status="publish_requested")

    def unexpected_index(*_args, **_kwargs):
        raise AssertionError("conflict-blocked knowledge must not be indexed")

    service.mark_suggestion_indexed = unexpected_index
    result = service.run_publish_jobs("seller_1", reviewer="ops")

    assert result["processedCount"] == 1
    assert result["results"][0]["success"] is False
    assert result["results"][0]["status"] == "CONFLICT_CONFIRMATION_REQUIRED"


def test_merge_resolution_removes_the_old_rule_from_pending_asset():
    asset = {
        "metrics": [],
        "terms": [],
        "knowledgeRules": [
            {
                "ruleId": "refund-rate-rule",
                "title": "退款率口径",
                "content": "退款率按退款订单数除以支付订单数计算",
            }
        ],
    }
    suggestion = _platform_suggestion("approved")
    suggestion["payload"].update(
        {
            "conflictResolution": "merge",
            "conflictResolutionReportId": "conflict-report-1",
            "conflictCheck": _conflict_report(),
        }
    )

    result = apply_resolved_knowledge_conflicts_to_asset(
        asset,
        "交易",
        "fact_trade",
        suggestion,
        "rule",
    )

    assert result["success"] is True
    assert result["removedKnowledgeIds"] == ["refund-rate-rule"]
    assert result["asset"]["knowledgeRules"] == []
