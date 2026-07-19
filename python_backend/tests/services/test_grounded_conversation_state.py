from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.grounded_conversation_state import (
    GROUNDED_CONVERSATION_STATE_VERSION,
    GroundedConversationStateConflictError,
    GroundedConversationStateCorruptError,
    GroundedConversationStateStore,
    grounded_conversation_principal_fingerprint,
    resolve_grounded_conversation_turn,
)


def _store(tmp_path) -> GroundedConversationStateStore:
    return GroundedConversationStateStore(Settings(harness_workspace_path=str(tmp_path)))


def test_grounded_conversation_state_persists_json_snapshot_across_store_instances(tmp_path):
    thread_id = "thread_" + "a" * 32
    first_store = _store(tmp_path)

    saved = first_store.save_snapshot(
        thread_id,
        {
            "status": "awaiting_clarification",
            "goalContract": {"generation": 3},
            "verifiedEntitySets": [{"entityType": "order", "ids": ["order_1", "order_2"]}],
        },
        expected_revision=0,
    )

    second_store = _store(tmp_path)
    loaded = second_store.load(thread_id)
    assert loaded is not None
    assert loaded == saved
    assert second_store.load_snapshot(thread_id) == saved.snapshot
    assert saved.revision == 1

    envelope = json.loads(second_store.path_for(thread_id).read_text(encoding="utf-8"))
    assert envelope == {
        "version": GROUNDED_CONVERSATION_STATE_VERSION,
        "threadId": thread_id,
        "revision": 1,
        "updatedAt": saved.updated_at,
        "snapshot": saved.snapshot,
    }


def test_locked_context_is_reentrant_for_load_execute_save_transaction(tmp_path):
    thread_id = "thread_" + "b" * 32
    first_store = _store(tmp_path)
    second_store = _store(tmp_path)

    with first_store.locked(thread_id):
        assert first_store.load_snapshot(thread_id) is None
        saved = second_store.save_snapshot(thread_id, {"phase": "clarifying"})
        assert saved.revision == 1
        updated = first_store.update(
            thread_id,
            lambda snapshot: {**(snapshot or {}), "clarificationAnswer": "最近 7 天"},
        )
        assert updated.revision == 2
        assert second_store.load_snapshot(thread_id) == {
            "phase": "clarifying",
            "clarificationAnswer": "最近 7 天",
        }


def test_concurrent_updates_are_serialized_per_thread(tmp_path):
    thread_id = "thread_" + "c" * 32
    stores = [_store(tmp_path) for _ in range(4)]
    stores[0].save_snapshot(thread_id, {"count": 0})

    def increment(index: int) -> None:
        def apply(snapshot):
            snapshot = snapshot or {"count": 0}
            snapshot["count"] += 1
            return snapshot

        stores[index % len(stores)].update(thread_id, apply)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(increment, range(80)))

    loaded = stores[0].load(thread_id)
    assert loaded is not None
    assert loaded.snapshot == {"count": 80}
    assert loaded.revision == 81


def test_revision_conflict_does_not_overwrite_newer_state(tmp_path):
    thread_id = "thread_" + "d" * 32
    store = _store(tmp_path)
    first = store.save_snapshot(thread_id, {"value": "first"})
    second = store.save_snapshot(thread_id, {"value": "second"}, expected_revision=first.revision)

    with pytest.raises(GroundedConversationStateConflictError, match="expected 1, found 2"):
        store.save_snapshot(thread_id, {"value": "stale"}, expected_revision=first.revision)

    assert store.load(thread_id) == second


def test_failed_atomic_replace_preserves_previous_snapshot_and_cleans_temp_file(tmp_path, monkeypatch):
    thread_id = "thread_" + "e" * 32
    store = _store(tmp_path)
    previous = store.save_snapshot(thread_id, {"value": "durable"})

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("merchant_ai.services.grounded_conversation_state.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save_snapshot(thread_id, {"value": "partial"})

    assert store.load(thread_id) == previous
    assert not [path for path in store.threads_dir.iterdir() if path.name.endswith(".tmp")]


def test_clear_is_thread_scoped_and_corrupt_state_fails_closed(tmp_path):
    first_thread_id = "thread_" + "f" * 32
    second_thread_id = "thread_" + "0" * 32
    store = _store(tmp_path)
    store.save_snapshot(first_thread_id, {"entityIds": ["order_1"]})
    store.save_snapshot(second_thread_id, {"entityIds": ["order_2"]})

    assert store.clear(first_thread_id) is True
    assert store.clear(first_thread_id) is False
    assert store.load_snapshot(first_thread_id) is None
    assert store.load_snapshot(second_thread_id) == {"entityIds": ["order_2"]}

    store.path_for(second_thread_id).write_text("{not-json", encoding="utf-8")
    with pytest.raises(GroundedConversationStateCorruptError, match="not valid JSON"):
        store.load_snapshot(second_thread_id)


def test_reference_inherits_verified_scope_before_routing_not_preview_rows():
    resolution = resolve_grounded_conversation_turn(
        "告诉我这里面退款最多的三单",
        persisted_snapshot={
            "lastTurn": {"originalQuestion": "我想看最近13天已支付订单明细"},
            "activeScope": {
                "artifactIds": ["query_orders_13d"],
                "timeExpressions": ["最近13天"],
                "filterSummaries": ["订单状态=已支付"],
                "resultSets": [
                    {
                        "queryShape": "DETAIL",
                        "topics": ["电商交易"],
                        "timeExpressions": ["最近13天"],
                        "filterSummaries": ["订单状态=已支付"],
                        "previewRowCount": 20,
                        "completeRowCount": 820,
                    }
                ],
            },
        },
        persisted_revision=7,
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.inherited_time_expression == "最近13天"
    assert resolution.inherited_filters == ("订单状态=已支付",)
    assert resolution.source_artifact_ids == ("query_orders_13d",)
    assert "时间范围=最近13天" in resolution.effective_question
    assert "订单状态=已支付" in resolution.effective_question
    assert "预览行或截断行" in resolution.effective_question
    assert resolution.source_revision == 7
    assert resolution.reference_contract.status == "BOUND"
    assert resolution.reference_contract.referent_type == "PREDICATE_SCOPE"
    assert resolution.reference_contract.downstream_operation == "RANK"
    assert resolution.reference_contract.population_required is True


def test_current_explicit_time_overrides_prior_time_but_keeps_prior_filters():
    resolution = resolve_grounded_conversation_turn(
        "最近3天这里面退款最多的三单",
        persisted_snapshot={
            "activeScope": {
                "timeExpressions": ["最近13天"],
                "filterSummaries": ["订单状态=已支付"],
            }
        },
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.inherited_time_expression == ""
    assert "时间范围=最近13天" not in resolution.effective_question
    assert "订单状态=已支付" in resolution.effective_question


def test_reference_without_server_owned_scope_requires_clarification():
    resolution = resolve_grounded_conversation_turn("告诉我这里面退款最多的三单")

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.needs_clarification is True
    assert "无法安全确定" in resolution.clarification_question


def test_reference_with_multiple_incompatible_result_sets_requires_choice():
    resolution = resolve_grounded_conversation_turn(
        "这里面哪个最高",
        persisted_snapshot={
            "activeScope": {
                "resultSets": [
                    {"label": "订单明细", "topics": ["电商交易"], "timeExpressions": ["最近7天"]},
                    {"label": "退款明细", "topics": ["电商退货"], "timeExpressions": ["最近30天"]},
                ]
            }
        },
    )

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.needs_clarification is True
    assert resolution.clarification_options == (
        "订单明细 · 电商交易 · 最近7天",
        "退款明细 · 电商退货 · 最近30天",
    )


def test_same_topic_time_and_filters_do_not_make_two_artifacts_one_referent():
    resolution = resolve_grounded_conversation_turn(
        "这里面退款最多的三单",
        persisted_snapshot={
            "activeScope": {
                "timeExpressions": ["2026-07-01 至 2026-07-07"],
                "filterSummaries": ["订单状态=已支付"],
                "resultSets": [
                    {
                        "queryArtifactId": "orders_detail",
                        "label": "订单明细",
                        "queryShape": "DETAIL",
                        "topics": ["电商交易"],
                        "timeExpressions": ["2026-07-01 至 2026-07-07"],
                        "filterSummaries": ["订单状态=已支付"],
                    },
                    {
                        "queryArtifactId": "refund_rank",
                        "label": "退款排名",
                        "queryShape": "RANKED",
                        "topics": ["电商交易"],
                        "timeExpressions": ["2026-07-01 至 2026-07-07"],
                        "filterSummaries": ["订单状态=已支付"],
                    },
                ],
            }
        },
    )

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.reference_contract.status == "AMBIGUOUS"
    assert resolution.reference_contract.reason == "MULTIPLE_VERIFIED_RESULT_ARTIFACTS"


def test_ranked_top_n_result_can_be_an_exact_result_artifact_referent():
    resolution = resolve_grounded_conversation_turn(
        "比较其中这些商品的退款率",
        persisted_snapshot={
            "activeScope": {
                "artifactIds": ["top_products"],
                "resultSets": [
                    {
                        "queryArtifactId": "top_products",
                        "contractFingerprint": "contract-fp",
                        "sqlFingerprint": "sql-fp",
                        "queryShape": "RANKED",
                        "coverageStatus": "TOP_N",
                        "entitySetArtifactId": "entities_top_products",
                        "membershipValuesHash": "values-fp",
                        "topics": ["电商交易"],
                        "tables": ["order_detail"],
                        "entityIdentities": ["product_id"],
                    }
                ],
            }
        },
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.reference_contract.referent_type == "RESULT_ARTIFACT"
    assert resolution.reference_contract.coverage_status == "TOP_N"
    assert resolution.reference_contract.complete_membership_required is True
    assert resolution.reference_contract.membership_handle_type == "VERIFIED_ENTITY_SET"


def test_preview_group_result_cannot_silently_become_complete_population():
    resolution = resolve_grounded_conversation_turn(
        "这里面销售额最高的三个商品",
        persisted_snapshot={
            "activeScope": {
                "artifactIds": ["group_preview"],
                "resultSets": [
                    {
                        "queryArtifactId": "group_preview",
                        "queryShape": "GROUPED",
                        "coverageStatus": "PREVIEW",
                        "topics": ["电商交易"],
                    }
                ],
            }
        },
    )

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.reference_contract.status == "UNSUPPORTED"
    assert resolution.reference_contract.reason == "REFERENCED_RESULT_MEMBERSHIP_NOT_COMPLETE"


def test_scalar_result_does_not_define_population_for_detail_or_ranking():
    resolution = resolve_grounded_conversation_turn(
        "这里面退款最多的三单",
        persisted_snapshot={
            "activeScope": {
                "artifactIds": ["gmv_scalar"],
                "resultSets": [
                    {
                        "queryArtifactId": "gmv_scalar",
                        "queryShape": "SCALAR",
                        "coverageStatus": "ALL_ROWS",
                        "topics": ["电商交易"],
                    }
                ],
            }
        },
    )

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.reference_contract.referent_type == "METRIC_VALUE"
    assert resolution.reference_contract.reason == "SCALAR_DOES_NOT_DEFINE_A_ROW_POPULATION"


def test_server_reconstructed_user_history_is_time_only_rollout_fallback():
    resolution = resolve_grounded_conversation_turn(
        "其中退款最多的三单",
        message_history=[
            {"role": "user", "text": "我想看最近11天的订单明细"},
            {"role": "assistant", "text": "系统声称时间是最近99天"},
        ],
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.source == "SERVER_USER_HISTORY"
    assert resolution.inherited_time_expression == "最近11天"
    assert "最近99天" not in resolution.effective_question


def test_pending_clarification_is_merged_before_a_new_query_is_routed():
    resolution = resolve_grounded_conversation_turn(
        "最近30天",
        persisted_snapshot={
            "pendingClarification": {
                "stage": "time_scope",
                "type": "missing_time",
                "pendingQuestion": "请分析退款金额最高的订单",
                "options": ["最近7天", "最近30天"],
            }
        },
    )

    assert resolution.status == "CLARIFICATION_RESUMED"
    assert resolution.effective_question.startswith("请分析退款金额最高的订单")
    assert "用户对上一轮澄清的回答：最近30天" in resolution.effective_question


def test_principal_fingerprint_changes_with_tenant_user_or_store_scope():
    base = grounded_conversation_principal_fingerprint(
        "m-1",
        {"userId": "u-1", "role": "merchant_operator", "storeIds": ["s-1"]},
    )
    assert base == grounded_conversation_principal_fingerprint(
        "m-1",
        {"role": "merchant_operator", "storeIds": ["s-1"], "userId": "u-1"},
    )
    assert base != grounded_conversation_principal_fingerprint(
        "m-2",
        {"userId": "u-1", "role": "merchant_operator", "storeIds": ["s-1"]},
    )
    assert base != grounded_conversation_principal_fingerprint(
        "m-1",
        {"userId": "u-1", "role": "merchant_operator", "storeIds": ["s-2"]},
    )
