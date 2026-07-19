from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationReferentCandidate,
    ConversationSemanticIssueCode,
    ConversationSemanticProviderOutput,
    build_conversation_semantic_resolver_request,
    conversation_referent_candidates_fingerprint,
    review_conversation_semantics,
)


def _candidate(
    artifact_id: str = "artifact-a",
    *,
    coverage_status: str = "ALL_ROWS",
) -> ConversationReferentCandidate:
    return ConversationReferentCandidate(
        artifact_id=artifact_id,
        contract_fingerprint="contract-%s" % artifact_id,
        sql_fingerprint="sql-%s" % artifact_id,
        query_shape="DETAIL",
        coverage_status=coverage_status,
        label="retained result %s" % artifact_id,
        topic_ids=("topic-a",),
        table_ids=("table-a",),
        goal_ids=("goal-a",),
        entity_identities=("entity-a",),
        data_grains=("grain-a",),
        time_scope_labels=("time-scope-a",),
        filter_scope_labels=("filter-scope-a",),
        membership_handle_type="VERIFIED_ENTITY_SET",
        membership_handle_id="entity-set-%s" % artifact_id,
        membership_values_hash="values-%s" % artifact_id,
        snapshot_semantics="ABSOLUTE_PREDICATE_SNAPSHOT",
    )


def _request(*candidates: ConversationReferentCandidate):
    return build_conversation_semantic_resolver_request(
        "continue the analysis over the referenced result",
        candidates or (_candidate(),),
    )


def _reference_output(request, **updates: Any):
    values = {
        "request_fingerprint": request.request_fingerprint,
        "question_fingerprint": request.question_fingerprint,
        "candidate_set_fingerprint": request.candidate_set_fingerprint,
        "complete": True,
        "reference_detected": True,
        "selected_artifact_id": request.candidates[0].artifact_id,
        "referent_type": ConversationReferenceType.PREDICATE_SCOPE,
        "downstream_operation": ConversationDownstreamOperation.RANK,
        "population_required": True,
        "complete_membership_required": False,
        "reference_phrases": ("referenced result",),
    }
    values.update(updates)
    return ConversationSemanticProviderOutput(**values)


@dataclass
class _Provider:
    output: Any
    authority_fingerprint: str = "reviewer-authority"
    delay_seconds: float = 0.0
    failure: Exception | None = None
    mutate_request: bool = False

    def resolve_conversation_reference(
        self,
        request,
        *,
        timeout_seconds: float,
    ):
        del timeout_seconds
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.mutate_request:
            object.__setattr__(request, "question", "mutated")
        if self.failure is not None:
            raise self.failure
        return self.output


def _review(provider: _Provider, request=None, **updates: Any):
    values = {
        "trusted_authority_fingerprints": ("reviewer-authority",),
        "core_authority_fingerprint": "core-authority",
        "timeout_seconds": 0.2,
    }
    values.update(updates)
    return review_conversation_semantics(
        provider,
        request or _request(),
        **values,
    )


def _issue_codes(review) -> set[ConversationSemanticIssueCode]:
    return {issue.code for issue in review.issues}


def test_structured_population_reference_is_accepted() -> None:
    request = _request()
    review = _review(_Provider(_reference_output(request)), request)

    assert review.accepted
    assert review.decision is not None
    assert review.decision.selected_artifact_id == "artifact-a"
    assert review.decision.population_required is True
    assert review.decision_fingerprint


def test_structured_non_reference_is_accepted_without_inheritance() -> None:
    request = _request()
    output = ConversationSemanticProviderOutput(
        request_fingerprint=request.request_fingerprint,
        question_fingerprint=request.question_fingerprint,
        candidate_set_fingerprint=request.candidate_set_fingerprint,
        complete=True,
        reference_detected=False,
    )
    review = _review(_Provider(output), request)

    assert review.accepted
    assert review.decision is not None
    assert review.decision.referent_type == ConversationReferenceType.NONE


def test_candidate_fingerprint_is_order_independent() -> None:
    first = _candidate("artifact-a")
    second = _candidate("artifact-b")

    assert conversation_referent_candidates_fingerprint((first, second)) == (
        conversation_referent_candidates_fingerprint((second, first))
    )
    assert _request(first, second).request_fingerprint == (
        _request(second, first).request_fingerprint
    )


@pytest.mark.parametrize(
    ("authority", "trusted", "core", "code"),
    (
        (
            "",
            ("reviewer-authority",),
            "core-authority",
            ConversationSemanticIssueCode.AUTHORITY_REQUIRED,
        ),
        (
            "unknown-authority",
            ("reviewer-authority",),
            "core-authority",
            ConversationSemanticIssueCode.AUTHORITY_UNTRUSTED,
        ),
        (
            "reviewer-authority",
            ("reviewer-authority",),
            "reviewer-authority",
            ConversationSemanticIssueCode.PROVIDER_NOT_INDEPENDENT,
        ),
    ),
)
def test_provider_authority_must_be_trusted_and_independent(
    authority: str,
    trusted: tuple[str, ...],
    core: str,
    code: ConversationSemanticIssueCode,
) -> None:
    request = _request()
    review = _review(
        _Provider(_reference_output(request), authority_fingerprint=authority),
        request,
        trusted_authority_fingerprints=trusted,
        core_authority_fingerprint=core,
    )

    assert not review.accepted
    assert code in _issue_codes(review)


@pytest.mark.parametrize(
    ("updates", "code"),
    (
        (
            {"request_fingerprint": "another-request"},
            ConversationSemanticIssueCode.REQUEST_BINDING_MISMATCH,
        ),
        (
            {"question_fingerprint": "another-question"},
            ConversationSemanticIssueCode.QUESTION_BINDING_MISMATCH,
        ),
        (
            {"candidate_set_fingerprint": "another-set"},
            ConversationSemanticIssueCode.CANDIDATE_BINDING_MISMATCH,
        ),
        (
            {"complete": False},
            ConversationSemanticIssueCode.PROVIDER_OUTPUT_INCOMPLETE,
        ),
        (
            {"selected_artifact_id": "artifact-outside-ledger"},
            ConversationSemanticIssueCode.SELECTED_ARTIFACT_UNKNOWN,
        ),
        (
            {
                "ambiguous": True,
                "selected_artifact_id": "artifact-a",
            },
            ConversationSemanticIssueCode.AMBIGUOUS_SELECTION_PRESENT,
        ),
        (
            {"referent_type": ConversationReferenceType.NONE},
            ConversationSemanticIssueCode.REFERENT_TYPE_REQUIRED,
        ),
        (
            {
                "downstream_operation": (
                    ConversationDownstreamOperation.UNSPECIFIED
                )
            },
            ConversationSemanticIssueCode.OPERATION_REQUIRED,
        ),
        (
            {
                "population_required": False,
                "complete_membership_required": True,
            },
            ConversationSemanticIssueCode.MEMBERSHIP_REQUIREMENT_INVALID,
        ),
    ),
)
def test_invalid_or_replayed_provider_decision_fails_closed(
    updates: dict[str, Any],
    code: ConversationSemanticIssueCode,
) -> None:
    request = _request()
    review = _review(
        _Provider(_reference_output(request, **updates)),
        request,
    )

    assert not review.accepted
    assert code in _issue_codes(review)


def test_non_reference_cannot_smuggle_reference_fields() -> None:
    request = _request()
    output = _reference_output(
        request,
        reference_detected=False,
    )
    review = _review(_Provider(output), request)

    assert not review.accepted
    assert ConversationSemanticIssueCode.NON_REFERENCE_FIELDS_PRESENT in (
        _issue_codes(review)
    )


def test_ambiguous_reference_is_accepted_only_without_selection() -> None:
    request = _request(_candidate("artifact-a"), _candidate("artifact-b"))
    output = _reference_output(
        request,
        ambiguous=True,
        selected_artifact_id="",
    )
    review = _review(_Provider(output), request)

    assert review.accepted
    assert review.decision is not None
    assert review.decision.ambiguous is True


def test_provider_timeout_failure_and_invalid_output_fail_closed() -> None:
    request = _request()
    timed_out = _review(
        _Provider(_reference_output(request), delay_seconds=0.1),
        request,
        timeout_seconds=0.01,
    )
    failed = _review(
        _Provider(None, failure=RuntimeError("provider failed")),
        request,
    )
    invalid = _review(_Provider("not-json"), request)

    assert ConversationSemanticIssueCode.PROVIDER_TIMEOUT in _issue_codes(
        timed_out
    )
    assert ConversationSemanticIssueCode.PROVIDER_FAILED in _issue_codes(
        failed
    )
    assert ConversationSemanticIssueCode.PROVIDER_OUTPUT_INVALID in (
        _issue_codes(invalid)
    )


def test_provider_cannot_mutate_request() -> None:
    request = _request()
    review = _review(
        _Provider(_reference_output(request), mutate_request=True),
        request,
    )

    assert not review.accepted
    assert ConversationSemanticIssueCode.REQUEST_MUTATED in _issue_codes(
        review
    )


def test_extra_provider_fields_are_rejected() -> None:
    request = _request()
    payload = _reference_output(request).model_dump(
        by_alias=True,
        mode="json",
    )
    payload["untrustedField"] = "value"
    review = _review(_Provider(payload), request)

    assert not review.accepted
    assert ConversationSemanticIssueCode.PROVIDER_OUTPUT_INVALID in (
        _issue_codes(review)
    )
