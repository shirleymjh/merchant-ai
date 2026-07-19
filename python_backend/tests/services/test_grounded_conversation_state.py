from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationReferentCandidate,
    ConversationSemanticProviderOutput,
    build_conversation_semantic_resolver_request,
    review_conversation_semantics,
)
from merchant_ai.services.grounded_conversation_state import (
    GROUNDED_CONVERSATION_STATE_VERSION,
    GroundedConversationStateConflictError,
    GroundedConversationStateCorruptError,
    GroundedConversationStateStore,
    grounded_conversation_principal_fingerprint,
    resolve_grounded_conversation_turn,
)


def _store(tmp_path: Any) -> GroundedConversationStateStore:
    return GroundedConversationStateStore(
        Settings(harness_workspace_path=str(tmp_path))
    )


def test_grounded_conversation_state_persists_json_snapshot_across_store_instances(
    tmp_path: Any,
) -> None:
    thread_id = "thread_" + "a" * 32
    first_store = _store(tmp_path)

    saved = first_store.save_snapshot(
        thread_id,
        {
            "status": "awaiting_clarification",
            "goalContract": {"generation": 3},
            "verifiedEntitySets": [
                {"entityType": "entity", "ids": ["entity_1", "entity_2"]}
            ],
        },
        expected_revision=0,
    )

    second_store = _store(tmp_path)
    loaded = second_store.load(thread_id)
    assert loaded is not None
    assert loaded == saved
    assert second_store.load_snapshot(thread_id) == saved.snapshot
    assert saved.revision == 1

    envelope = json.loads(
        second_store.path_for(thread_id).read_text(encoding="utf-8")
    )
    assert envelope == {
        "version": GROUNDED_CONVERSATION_STATE_VERSION,
        "threadId": thread_id,
        "revision": 1,
        "updatedAt": saved.updated_at,
        "snapshot": saved.snapshot,
    }


def test_locked_context_is_reentrant_for_load_execute_save_transaction(
    tmp_path: Any,
) -> None:
    thread_id = "thread_" + "b" * 32
    first_store = _store(tmp_path)
    second_store = _store(tmp_path)

    with first_store.locked(thread_id):
        assert first_store.load_snapshot(thread_id) is None
        saved = second_store.save_snapshot(
            thread_id,
            {"phase": "clarifying"},
        )
        assert saved.revision == 1
        updated = first_store.update(
            thread_id,
            lambda snapshot: {
                **(snapshot or {}),
                "clarificationAnswer": "bounded answer",
            },
        )
        assert updated.revision == 2
        assert second_store.load_snapshot(thread_id) == {
            "phase": "clarifying",
            "clarificationAnswer": "bounded answer",
        }


def test_concurrent_updates_are_serialized_per_thread(tmp_path: Any) -> None:
    thread_id = "thread_" + "c" * 32
    stores = [_store(tmp_path) for _ in range(4)]
    stores[0].save_snapshot(thread_id, {"count": 0})

    def increment(index: int) -> None:
        def apply(snapshot: dict[str, Any] | None) -> dict[str, Any]:
            value = snapshot or {"count": 0}
            value["count"] += 1
            return value

        stores[index % len(stores)].update(thread_id, apply)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(increment, range(80)))

    loaded = stores[0].load(thread_id)
    assert loaded is not None
    assert loaded.snapshot == {"count": 80}
    assert loaded.revision == 81


def test_revision_conflict_does_not_overwrite_newer_state(tmp_path: Any) -> None:
    thread_id = "thread_" + "d" * 32
    store = _store(tmp_path)
    first = store.save_snapshot(thread_id, {"value": "first"})
    second = store.save_snapshot(
        thread_id,
        {"value": "second"},
        expected_revision=first.revision,
    )

    with pytest.raises(GroundedConversationStateConflictError) as captured:
        store.save_snapshot(
            thread_id,
            {"value": "stale"},
            expected_revision=first.revision,
        )

    assert "expected 1, found 2" in str(captured.value)
    assert store.load(thread_id) == second


def test_failed_atomic_replace_preserves_previous_snapshot_and_cleans_temp_file(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = "thread_" + "e" * 32
    store = _store(tmp_path)
    previous = store.save_snapshot(thread_id, {"value": "durable"})

    def fail_replace(source: Any, destination: Any) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(
        "merchant_ai.services.grounded_conversation_state.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError) as captured:
        store.save_snapshot(thread_id, {"value": "partial"})

    assert "replace failed" in str(captured.value)
    assert store.load(thread_id) == previous
    assert not [
        path
        for path in store.threads_dir.iterdir()
        if path.name.endswith(".tmp")
    ]


def test_clear_is_thread_scoped_and_corrupt_state_fails_closed(
    tmp_path: Any,
) -> None:
    first_thread_id = "thread_" + "f" * 32
    second_thread_id = "thread_" + "0" * 32
    store = _store(tmp_path)
    store.save_snapshot(first_thread_id, {"entityIds": ["entity_1"]})
    store.save_snapshot(second_thread_id, {"entityIds": ["entity_2"]})

    assert store.clear(first_thread_id) is True
    assert store.clear(first_thread_id) is False
    assert store.load_snapshot(first_thread_id) is None
    assert store.load_snapshot(second_thread_id) == {
        "entityIds": ["entity_2"]
    }

    store.path_for(second_thread_id).write_text(
        "{not-json",
        encoding="utf-8",
    )
    with pytest.raises(GroundedConversationStateCorruptError) as captured:
        store.load_snapshot(second_thread_id)
    assert "not valid JSON" in str(captured.value)


def test_thread_file_name_is_sanitized_without_changing_thread_identity(
    tmp_path: Any,
) -> None:
    store = _store(tmp_path)
    thread_id = "scope/with unicode/会话"
    saved = store.save_snapshot(thread_id, {"value": "retained"})

    path = store.path_for(thread_id)
    assert path.parent == store.threads_dir
    assert "/" not in path.name
    assert store.load(thread_id) == saved


def _candidate(
    artifact_id: str = "artifact-a",
    *,
    label: str = "Result A",
    query_shape: str = "DETAIL",
    coverage_status: str = "ALL_ROWS",
    membership: bool = True,
    time_scope_labels: tuple[str, ...] = ("retained-time-scope",),
) -> ConversationReferentCandidate:
    return ConversationReferentCandidate(
        artifact_id=artifact_id,
        contract_fingerprint="contract-%s" % artifact_id,
        sql_fingerprint="sql-%s" % artifact_id,
        query_shape=query_shape,
        coverage_status=coverage_status,
        label=label,
        topic_ids=("topic-a",),
        table_ids=("table-a",),
        goal_ids=("goal-a",),
        entity_identities=("entity-a",),
        data_grains=("grain-a",),
        time_scope_labels=time_scope_labels,
        filter_scope_labels=("retained-filter-scope",),
        membership_handle_type=("VERIFIED_ENTITY_SET" if membership else ""),
        membership_handle_id=("entity-set-%s" % artifact_id if membership else ""),
        membership_values_hash=("values-%s" % artifact_id if membership else ""),
        snapshot_semantics="ABSOLUTE_PREDICATE_SNAPSHOT",
    )


@dataclass
class _Provider:
    output: Any = None
    failure: Exception | None = None
    authority_fingerprint: str = "conversation-reviewer"

    def resolve_conversation_reference(
        self,
        request: Any,
        *,
        timeout_seconds: float,
    ) -> Any:
        del request, timeout_seconds
        if self.failure is not None:
            raise self.failure
        return self.output


def _semantic_review(
    question: str,
    candidates: tuple[ConversationReferentCandidate, ...],
    *,
    reference_detected: bool = True,
    ambiguous: bool = False,
    selected_artifact_id: str | None = None,
    referent_type: ConversationReferenceType = (
        ConversationReferenceType.PREDICATE_SCOPE
    ),
    downstream_operation: ConversationDownstreamOperation = (
        ConversationDownstreamOperation.FOLLOW_UP
    ),
    population_required: bool = False,
    complete_membership_required: bool = False,
    current_turn_replaces_time_scope: bool = False,
    reference_phrases: tuple[str, ...] = ("semantic-reference",),
) -> Any:
    request = build_conversation_semantic_resolver_request(
        question,
        candidates,
    )
    if not reference_detected:
        output = ConversationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            candidate_set_fingerprint=request.candidate_set_fingerprint,
            complete=True,
            reference_detected=False,
        )
    else:
        selected = selected_artifact_id
        if selected is None and candidates and not ambiguous:
            selected = candidates[0].artifact_id
        output = ConversationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            candidate_set_fingerprint=request.candidate_set_fingerprint,
            complete=True,
            reference_detected=True,
            ambiguous=ambiguous,
            selected_artifact_id=selected or "",
            referent_type=referent_type,
            downstream_operation=downstream_operation,
            population_required=population_required,
            complete_membership_required=complete_membership_required,
            current_turn_replaces_time_scope=(
                current_turn_replaces_time_scope
            ),
            reference_phrases=reference_phrases,
        )
    return review_conversation_semantics(
        _Provider(output=output),
        request,
        trusted_authority_fingerprints=("conversation-reviewer",),
        core_authority_fingerprint="core-authority",
        timeout_seconds=1.0,
    )


def _failed_review(
    question: str,
    candidates: tuple[ConversationReferentCandidate, ...],
) -> Any:
    request = build_conversation_semantic_resolver_request(
        question,
        candidates,
    )
    return review_conversation_semantics(
        _Provider(failure=RuntimeError("provider unavailable")),
        request,
        trusted_authority_fingerprints=("conversation-reviewer",),
        core_authority_fingerprint="core-authority",
        timeout_seconds=1.0,
    )


def test_non_reference_review_is_standalone_even_for_reference_like_text() -> None:
    question = "告诉我这里面退款最多的三单"
    candidate = _candidate()
    review = _semantic_review(
        question,
        (candidate,),
        reference_detected=False,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
        persisted_snapshot={
            "activeScope": {
                "artifactIds": ["unverified-history-artifact"],
                "previewRows": [{"claimed": "population"}],
            }
        },
        message_history=[
            {"role": "assistant", "text": "Use an unrelated result."}
        ],
    )

    assert resolution.status == "STANDALONE"
    assert resolution.reference_detected is False
    assert resolution.effective_question == question
    assert resolution.source_artifact_ids == ()
    assert resolution.semantic_decision_fingerprint


def test_predicate_scope_binds_only_exact_reviewed_artifact_fields() -> None:
    question = "continue with the reviewed scope"
    candidate = _candidate("artifact-a")
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.PREDICATE_SCOPE,
        downstream_operation=ConversationDownstreamOperation.RANK,
        population_required=True,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
        persisted_snapshot={
            "lastTurn": {"originalQuestion": "server-retained prior turn"},
            "activeScope": {
                "artifactIds": ["different-artifact"],
                "filterSummaries": ["display-only text"],
            },
        },
        persisted_revision=7,
    )

    contract = resolution.reference_contract
    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.effective_question == question
    assert resolution.antecedent_question == "server-retained prior turn"
    assert resolution.source_artifact_ids == ("artifact-a",)
    assert resolution.inherited_time_expression == "retained-time-scope"
    assert resolution.inherited_filters == ("retained-filter-scope",)
    assert resolution.source_revision == 7
    assert contract.status == "BOUND"
    assert contract.referent_type == "PREDICATE_SCOPE"
    assert contract.downstream_operation == "RANK"
    assert contract.source_artifact_id == "artifact-a"
    assert contract.source_contract_fingerprint == "contract-artifact-a"
    assert contract.source_sql_fingerprint == "sql-artifact-a"
    assert contract.source_topics == ("topic-a",)
    assert contract.source_tables == ("table-a",)
    assert contract.population_required is True
    assert "display-only text" not in resolution.trace()["inheritedFilters"]


@pytest.mark.parametrize(
    ("structured_replacement", "expected_inherited_time"),
    ((False, "retained-time-scope"), (True, "")),
)
def test_time_scope_replacement_trusts_only_structured_decision(
    structured_replacement: bool,
    expected_inherited_time: str,
) -> None:
    question = "最近3天 continue"
    candidate = _candidate()
    review = _semantic_review(
        question,
        (candidate,),
        current_turn_replaces_time_scope=structured_replacement,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.inherited_time_expression == expected_inherited_time
    assert (
        resolution.reference_contract.current_turn_replaces_time_scope
        is structured_replacement
    )
    assert resolution.effective_question == question


def test_missing_semantic_review_requires_typed_clarification() -> None:
    question = "continue"
    resolution = resolve_grounded_conversation_turn(
        question,
        verified_candidates=(_candidate(),),
    )

    assert resolution.status == "SEMANTIC_REVIEW_UNAVAILABLE"
    assert resolution.needs_clarification is True
    assert (
        resolution.clarification_type
        == "CONVERSATION_SEMANTIC_REVIEW_UNAVAILABLE"
    )
    assert resolution.effective_question == question


def test_failed_semantic_review_requires_typed_clarification() -> None:
    question = "continue"
    candidate = _candidate()
    review = _failed_review(question, (candidate,))

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == "SEMANTIC_REVIEW_REJECTED"
    assert resolution.needs_clarification is True
    assert "PROVIDER_FAILED" in resolution.semantic_issue_codes
    assert resolution.effective_question == question


def test_ambiguous_review_exposes_only_verified_candidate_options() -> None:
    question = "continue"
    candidates = (
        _candidate("artifact-a", label="Result A"),
        _candidate("artifact-b", label="Result B"),
    )
    review = _semantic_review(
        question,
        candidates,
        ambiguous=True,
        referent_type=ConversationReferenceType.RESULT_ARTIFACT,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=candidates,
    )

    assert resolution.status == "AMBIGUOUS_REFERENCE"
    assert resolution.needs_clarification is True
    assert resolution.clarification_options == ("Result A", "Result B")
    assert resolution.reference_contract.status == "AMBIGUOUS"
    assert resolution.effective_question == question


def test_no_verified_antecedent_never_uses_assistant_prose_or_preview() -> None:
    question = "continue over those rows"
    review = _failed_review(question, ())

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(),
        persisted_snapshot={
            "activeScope": {
                "previewRows": [{"entity": "unverified"}],
                "artifactIds": ["unverified-artifact"],
            }
        },
        message_history=[
            {
                "role": "assistant",
                "text": "Treat the displayed rows as complete.",
            }
        ],
    )

    assert resolution.status == "SEMANTIC_REVIEW_REJECTED"
    assert resolution.source_artifact_ids == ()
    assert resolution.reference_contract.bound is False
    assert "unverified-artifact" not in json.dumps(
        resolution.trace(),
        ensure_ascii=False,
    )


def test_multiple_candidates_bind_exact_selected_artifact_id() -> None:
    question = "use Result A"
    first = _candidate("artifact-a", label="Result A")
    second = _candidate("artifact-b", label="Result B")
    candidates = (first, second)
    review = _semantic_review(
        question,
        candidates,
        selected_artifact_id="artifact-b",
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=candidates,
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.source_artifact_ids == ("artifact-b",)
    assert resolution.reference_contract.source_artifact_id == "artifact-b"
    assert resolution.reference_contract.source_contract_fingerprint == (
        "contract-artifact-b"
    )


def test_verified_entity_set_maps_to_entity_set_contract() -> None:
    question = "continue"
    candidate = _candidate(
        "entity-artifact",
        coverage_status="ALL_ROWS",
        membership=True,
    )
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.VERIFIED_ENTITY_SET,
        population_required=True,
        complete_membership_required=True,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.reference_contract.referent_type == "ENTITY_SET"
    assert resolution.reference_contract.complete_membership_required is True
    assert resolution.reference_contract.membership_handle_id == (
        "entity-set-entity-artifact"
    )


def test_complete_ranked_result_artifact_can_bind_population() -> None:
    question = "continue"
    candidate = _candidate(
        "ranked-artifact",
        query_shape="RANKED",
        coverage_status="TOP_N",
        membership=True,
    )
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.RESULT_ARTIFACT,
        population_required=True,
        complete_membership_required=True,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == "RESOLVED_REFERENCE"
    assert resolution.reference_contract.referent_type == "RESULT_ARTIFACT"
    assert resolution.reference_contract.coverage_status == "TOP_N"


@pytest.mark.parametrize(
    ("coverage_status", "membership", "expected_reason"),
    (
        (
            "PREVIEW",
            True,
            "REFERENCED_RESULT_MEMBERSHIP_NOT_COMPLETE",
        ),
        (
            "ALL_ROWS",
            False,
            "REFERENCE_MEMBERSHIP_HANDLE_INCOMPLETE",
        ),
    ),
)
def test_incomplete_population_artifact_requires_clarification(
    coverage_status: str,
    membership: bool,
    expected_reason: str,
) -> None:
    question = "continue"
    candidate = _candidate(
        coverage_status=coverage_status,
        membership=membership,
    )
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.RESULT_ARTIFACT,
        population_required=True,
        complete_membership_required=True,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == "UNSAFE_REFERENCE"
    assert resolution.needs_clarification is True
    assert resolution.reference_contract.reason == expected_reason
    assert resolution.reference_contract.bound is False


@pytest.mark.parametrize(
    ("population_required", "expected_status"),
    ((False, "RESOLVED_REFERENCE"), (True, "UNSAFE_REFERENCE")),
)
def test_metric_value_never_silently_becomes_population(
    population_required: bool,
    expected_status: str,
) -> None:
    question = "continue"
    candidate = _candidate(query_shape="SCALAR")
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.METRIC_VALUE,
        population_required=population_required,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == expected_status
    assert resolution.reference_contract.referent_type == "METRIC_VALUE"
    if population_required:
        assert resolution.reference_contract.reason == (
            "METRIC_VALUE_DOES_NOT_DEFINE_POPULATION"
        )


def test_unsupported_referent_type_requires_clarification() -> None:
    question = "continue"
    candidate = _candidate()
    review = _semantic_review(
        question,
        (candidate,),
        referent_type=ConversationReferenceType.COMPARISON_BASELINE,
        downstream_operation=ConversationDownstreamOperation.COMPARE,
    )

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(candidate,),
    )

    assert resolution.status == "UNSAFE_REFERENCE"
    assert resolution.reference_contract.reason == (
        "REFERENT_TYPE_NOT_EXECUTABLE"
    )


def test_review_cannot_be_replayed_against_different_candidate_set() -> None:
    question = "continue"
    first = _candidate("artifact-a")
    second = _candidate("artifact-b")
    review = _semantic_review(question, (first,))

    resolution = resolve_grounded_conversation_turn(
        question,
        semantic_review=review,
        verified_candidates=(second,),
    )

    assert resolution.status == "SEMANTIC_REVIEW_BINDING_MISMATCH"
    assert resolution.needs_clarification is True
    assert resolution.source_artifact_ids == ()


def test_pending_clarification_resumes_as_separate_structured_fields() -> None:
    answer = "bounded clarification answer"
    pending_question = "original pending question"
    resolution = resolve_grounded_conversation_turn(
        answer,
        persisted_snapshot={
            "pendingClarification": {
                "stage": "scope-stage",
                "type": "scope-type",
                "pendingQuestion": pending_question,
                "options": ["option-a", "option-b"],
            }
        },
    )

    assert resolution.status == "CLARIFICATION_RESUMED"
    assert resolution.effective_question == answer
    assert resolution.pending_clarification_question == pending_question
    assert resolution.pending_clarification_answer == answer
    assert resolution.pending_clarification_stage == "scope-stage"
    assert resolution.pending_clarification_type == "scope-type"
    assert resolution.pending_clarification_options == (
        "option-a",
        "option-b",
    )


def test_request_context_pending_clarification_is_deterministic() -> None:
    resolution = resolve_grounded_conversation_turn(
        "answer",
        request_context=SimpleNamespace(
            pending_question="pending question",
            pending_clarification_stage="pending-stage",
            pending_clarification_type="pending-type",
            pending_clarification_options=["option-a"],
        ),
    )

    assert resolution.status == "CLARIFICATION_RESUMED"
    assert resolution.pending_clarification_question == "pending question"
    assert resolution.pending_clarification_answer == "answer"
    assert resolution.pending_clarification_options == ("option-a",)


def test_principal_fingerprint_changes_with_tenant_user_or_store_scope() -> None:
    base = grounded_conversation_principal_fingerprint(
        "m-1",
        {
            "userId": "u-1",
            "role": "merchant_operator",
            "storeIds": ["s-1"],
        },
    )
    assert base == grounded_conversation_principal_fingerprint(
        "m-1",
        {
            "role": "merchant_operator",
            "storeIds": ["s-1"],
            "userId": "u-1",
        },
    )
    assert base != grounded_conversation_principal_fingerprint(
        "m-2",
        {
            "userId": "u-1",
            "role": "merchant_operator",
            "storeIds": ["s-1"],
        },
    )
    assert base != grounded_conversation_principal_fingerprint(
        "m-1",
        {
            "userId": "u-1",
            "role": "merchant_operator",
            "storeIds": ["s-2"],
        },
    )
