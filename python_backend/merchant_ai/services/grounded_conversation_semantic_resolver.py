from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence

from pydantic import ConfigDict, ValidationError, model_validator

from merchant_ai.models import APIModel


class ConversationReferenceType(str, Enum):
    NONE = "NONE"
    PREDICATE_SCOPE = "PREDICATE_SCOPE"
    VERIFIED_ENTITY_SET = "VERIFIED_ENTITY_SET"
    RESULT_ARTIFACT = "RESULT_ARTIFACT"
    METRIC_VALUE = "METRIC_VALUE"
    COMPARISON_BASELINE = "COMPARISON_BASELINE"


class ConversationDownstreamOperation(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    RANK = "RANK"
    DETAIL = "DETAIL"
    TREND = "TREND"
    COMPARE = "COMPARE"
    EXPLAIN = "EXPLAIN"
    AGGREGATE = "AGGREGATE"
    ANALYZE = "ANALYZE"
    FOLLOW_UP = "FOLLOW_UP"


class ConversationSemanticIssueCode(str, Enum):
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    AUTHORITY_UNTRUSTED = "AUTHORITY_UNTRUSTED"
    PROVIDER_NOT_INDEPENDENT = "PROVIDER_NOT_INDEPENDENT"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    PROVIDER_OUTPUT_INVALID = "PROVIDER_OUTPUT_INVALID"
    PROVIDER_OUTPUT_INCOMPLETE = "PROVIDER_OUTPUT_INCOMPLETE"
    REQUEST_MUTATED = "REQUEST_MUTATED"
    REQUEST_BINDING_MISMATCH = "REQUEST_BINDING_MISMATCH"
    QUESTION_BINDING_MISMATCH = "QUESTION_BINDING_MISMATCH"
    CANDIDATE_BINDING_MISMATCH = "CANDIDATE_BINDING_MISMATCH"
    NON_REFERENCE_FIELDS_PRESENT = "NON_REFERENCE_FIELDS_PRESENT"
    AMBIGUOUS_SELECTION_PRESENT = "AMBIGUOUS_SELECTION_PRESENT"
    REFERENT_TYPE_REQUIRED = "REFERENT_TYPE_REQUIRED"
    OPERATION_REQUIRED = "OPERATION_REQUIRED"
    SELECTED_ARTIFACT_REQUIRED = "SELECTED_ARTIFACT_REQUIRED"
    SELECTED_ARTIFACT_UNKNOWN = "SELECTED_ARTIFACT_UNKNOWN"
    MEMBERSHIP_REQUIREMENT_INVALID = "MEMBERSHIP_REQUIREMENT_INVALID"
    AMBIGUOUS_RETRIEVAL_QUESTION_PRESENT = (
        "AMBIGUOUS_RETRIEVAL_QUESTION_PRESENT"
    )


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _stable_fingerprint(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _unique_text(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    normalized = tuple(_text(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicate values" % field_name)
    return normalized


class ConversationReferentCandidate(_StrictFrozenModel):
    """Population-safe projection of one server-retained prior result."""

    artifact_id: str
    contract_fingerprint: str
    sql_fingerprint: str = ""
    query_shape: str
    coverage_status: str
    label: str = ""
    topic_ids: tuple[str, ...] = ()
    table_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()
    entity_identities: tuple[str, ...] = ()
    data_grains: tuple[str, ...] = ()
    time_scope_labels: tuple[str, ...] = ()
    filter_scope_labels: tuple[str, ...] = ()
    membership_handle_type: str = ""
    membership_handle_id: str = ""
    membership_values_hash: str = ""
    snapshot_semantics: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "ConversationReferentCandidate":
        for field_name in (
            "artifact_id",
            "contract_fingerprint",
            "query_shape",
            "coverage_status",
        ):
            if not _text(getattr(self, field_name)):
                raise ValueError("%s must not be empty" % field_name)
        for field_name in (
            "topic_ids",
            "table_ids",
            "goal_ids",
            "entity_identities",
            "data_grains",
            "time_scope_labels",
            "filter_scope_labels",
        ):
            _unique_text(getattr(self, field_name), field_name)
        if bool(self.membership_handle_id) != bool(
            self.membership_handle_type
        ):
            raise ValueError(
                "membership handle type and id must be declared together"
            )
        return self


class ConversationSemanticResolverRequest(_StrictFrozenModel):
    protocol_version: Literal["conversation_semantic_resolver_request.v1"] = (
        "conversation_semantic_resolver_request.v1"
    )
    request_fingerprint: str
    question: str
    question_fingerprint: str
    candidate_set_fingerprint: str
    candidates: tuple[ConversationReferentCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "ConversationSemanticResolverRequest":
        for field_name in (
            "request_fingerprint",
            "question",
            "question_fingerprint",
            "candidate_set_fingerprint",
        ):
            if not _text(getattr(self, field_name)):
                raise ValueError("%s must not be empty" % field_name)
        _unique_text(
            tuple(candidate.artifact_id for candidate in self.candidates),
            "candidates.artifact_id",
        )
        return self


class ConversationSemanticProviderOutput(_StrictFrozenModel):
    protocol_version: Literal["conversation_semantic_provider_output.v1"] = (
        "conversation_semantic_provider_output.v1"
    )
    request_fingerprint: str
    question_fingerprint: str
    candidate_set_fingerprint: str
    complete: bool
    reference_detected: bool
    ambiguous: bool = False
    selected_artifact_id: str = ""
    referent_type: ConversationReferenceType = ConversationReferenceType.NONE
    downstream_operation: ConversationDownstreamOperation = (
        ConversationDownstreamOperation.UNSPECIFIED
    )
    population_required: bool = False
    complete_membership_required: bool = False
    current_turn_replaces_time_scope: bool = False
    reference_phrases: tuple[str, ...] = ()
    retrieval_question: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "ConversationSemanticProviderOutput":
        for field_name in (
            "request_fingerprint",
            "question_fingerprint",
            "candidate_set_fingerprint",
        ):
            if not _text(getattr(self, field_name)):
                raise ValueError("%s must not be empty" % field_name)
        _unique_text(self.reference_phrases, "reference_phrases")
        if len(_text(self.retrieval_question)) > 1200:
            raise ValueError("retrieval_question must not exceed 1200 characters")
        return self


class ConversationSemanticProvider(Protocol):
    @property
    def authority_fingerprint(self) -> str: ...

    def resolve_conversation_reference(
        self,
        request: ConversationSemanticResolverRequest,
        *,
        timeout_seconds: float,
    ) -> ConversationSemanticProviderOutput | Mapping[str, Any] | str: ...


class ConversationSemanticIssue(_StrictFrozenModel):
    code: ConversationSemanticIssueCode
    message: str


class ConversationSemanticReview(_StrictFrozenModel):
    accepted: bool
    authority_fingerprint: str = ""
    request_fingerprint: str
    decision_fingerprint: str = ""
    decision: Optional[ConversationSemanticProviderOutput] = None
    issues: tuple[ConversationSemanticIssue, ...] = ()


def conversation_referent_candidates_fingerprint(
    candidates: Sequence[ConversationReferentCandidate],
) -> str:
    return _stable_fingerprint(
        [
            candidate.model_dump(by_alias=True, mode="json")
            for candidate in sorted(
                candidates,
                key=lambda item: item.artifact_id,
            )
        ]
    )


def build_conversation_semantic_resolver_request(
    question: str,
    candidates: Sequence[ConversationReferentCandidate],
) -> ConversationSemanticResolverRequest:
    normalized_question = _text(question)
    if not normalized_question:
        raise ValueError("question must not be empty")
    normalized_candidates = tuple(
        sorted(candidates, key=lambda item: item.artifact_id)
    )
    question_fingerprint = _stable_fingerprint(normalized_question)
    candidate_fingerprint = conversation_referent_candidates_fingerprint(
        normalized_candidates
    )
    request_payload = {
        "protocolVersion": "conversation_semantic_resolver_request.v1",
        "question": normalized_question,
        "questionFingerprint": question_fingerprint,
        "candidateSetFingerprint": candidate_fingerprint,
        "candidates": [
            candidate.model_dump(by_alias=True, mode="json")
            for candidate in normalized_candidates
        ],
    }
    return ConversationSemanticResolverRequest(
        request_fingerprint=_stable_fingerprint(request_payload),
        question=normalized_question,
        question_fingerprint=question_fingerprint,
        candidate_set_fingerprint=candidate_fingerprint,
        candidates=normalized_candidates,
    )


def _issue(
    code: ConversationSemanticIssueCode,
    message: str,
) -> ConversationSemanticIssue:
    return ConversationSemanticIssue(code=code, message=message)


def _review(
    request: ConversationSemanticResolverRequest,
    *,
    authority_fingerprint: str = "",
    decision: ConversationSemanticProviderOutput | None = None,
    issues: Sequence[ConversationSemanticIssue] = (),
) -> ConversationSemanticReview:
    issue_tuple = tuple(issues)
    return ConversationSemanticReview(
        accepted=not issue_tuple and decision is not None,
        authority_fingerprint=_text(authority_fingerprint),
        request_fingerprint=request.request_fingerprint,
        decision_fingerprint=(
            _stable_fingerprint(decision) if decision is not None else ""
        ),
        decision=decision,
        issues=issue_tuple,
    )


def _parse_output(value: Any) -> ConversationSemanticProviderOutput:
    if isinstance(value, ConversationSemanticProviderOutput):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return ConversationSemanticProviderOutput.model_validate(value)


def _validate_decision(
    request: ConversationSemanticResolverRequest,
    decision: ConversationSemanticProviderOutput,
) -> tuple[ConversationSemanticIssue, ...]:
    issues: list[ConversationSemanticIssue] = []
    if decision.request_fingerprint != request.request_fingerprint:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.REQUEST_BINDING_MISMATCH,
                "Provider output is not bound to this resolver request.",
            )
        )
    if decision.question_fingerprint != request.question_fingerprint:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.QUESTION_BINDING_MISMATCH,
                "Provider output is not bound to the current question.",
            )
        )
    if (
        decision.candidate_set_fingerprint
        != request.candidate_set_fingerprint
    ):
        issues.append(
            _issue(
                ConversationSemanticIssueCode.CANDIDATE_BINDING_MISMATCH,
                "Provider output is not bound to the retained candidate set.",
            )
        )
    if not decision.complete:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.PROVIDER_OUTPUT_INCOMPLETE,
                "Provider did not complete the semantic decision.",
            )
        )
    if not decision.reference_detected:
        if any(
            (
                decision.ambiguous,
                bool(decision.selected_artifact_id),
                decision.referent_type != ConversationReferenceType.NONE,
                decision.downstream_operation
                != ConversationDownstreamOperation.UNSPECIFIED,
                decision.population_required,
                decision.complete_membership_required,
                bool(decision.reference_phrases),
                bool(_text(decision.retrieval_question)),
            )
        ):
            issues.append(
                _issue(
                    ConversationSemanticIssueCode.NON_REFERENCE_FIELDS_PRESENT,
                    "A non-reference decision contains reference-only fields.",
                )
            )
        return tuple(issues)
    if decision.referent_type == ConversationReferenceType.NONE:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.REFERENT_TYPE_REQUIRED,
                "A detected reference requires a structured referent type.",
            )
        )
    if (
        decision.downstream_operation
        == ConversationDownstreamOperation.UNSPECIFIED
    ):
        issues.append(
            _issue(
                ConversationSemanticIssueCode.OPERATION_REQUIRED,
                "A detected reference requires a downstream operation.",
            )
        )
    if decision.complete_membership_required and not decision.population_required:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.MEMBERSHIP_REQUIREMENT_INVALID,
                "Complete membership cannot be required without a population.",
            )
        )
    if decision.ambiguous:
        if decision.selected_artifact_id:
            issues.append(
                _issue(
                    ConversationSemanticIssueCode.AMBIGUOUS_SELECTION_PRESENT,
                    "An ambiguous reference cannot select an artifact.",
                )
            )
        if _text(decision.retrieval_question):
            issues.append(
                _issue(
                    ConversationSemanticIssueCode.AMBIGUOUS_RETRIEVAL_QUESTION_PRESENT,
                    "An ambiguous reference cannot produce a standalone retrieval question.",
                )
            )
        return tuple(issues)
    if not decision.selected_artifact_id:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.SELECTED_ARTIFACT_REQUIRED,
                "An unambiguous reference must select one retained artifact.",
            )
        )
        return tuple(issues)
    known_ids = {candidate.artifact_id for candidate in request.candidates}
    if decision.selected_artifact_id not in known_ids:
        issues.append(
            _issue(
                ConversationSemanticIssueCode.SELECTED_ARTIFACT_UNKNOWN,
                "The provider selected an artifact outside the retained set.",
            )
        )
    return tuple(issues)


def review_conversation_semantics(
    provider: ConversationSemanticProvider,
    request: ConversationSemanticResolverRequest,
    *,
    trusted_authority_fingerprints: Sequence[str],
    core_authority_fingerprint: str = "",
    timeout_seconds: float = 8.0,
) -> ConversationSemanticReview:
    authority = _text(getattr(provider, "authority_fingerprint", ""))
    if not authority:
        return _review(
            request,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.AUTHORITY_REQUIRED,
                    "Conversation semantic provider has no authority identity.",
                ),
            ),
        )
    trusted = {
        _text(value)
        for value in trusted_authority_fingerprints
        if _text(value)
    }
    if authority not in trusted:
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.AUTHORITY_UNTRUSTED,
                    "Conversation semantic provider authority is not trusted.",
                ),
            ),
        )
    if authority == _text(core_authority_fingerprint):
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.PROVIDER_NOT_INDEPENDENT,
                    "Core and conversation semantic review must use distinct authorities.",
                ),
            ),
        )
    before = request.model_dump(by_alias=True, mode="json")
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        provider.resolve_conversation_reference,
        request,
        timeout_seconds=max(0.001, float(timeout_seconds)),
    )
    try:
        raw_output = future.result(
            timeout=max(0.001, float(timeout_seconds))
        )
    except FutureTimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.PROVIDER_TIMEOUT,
                    "Conversation semantic provider exceeded its budget.",
                ),
            ),
        )
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.PROVIDER_FAILED,
                    "%s:%s" % (type(exc).__name__, str(exc)[:300]),
                ),
            ),
        )
    else:
        executor.shutdown(wait=True, cancel_futures=True)
    after = request.model_dump(by_alias=True, mode="json")
    if before != after:
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.REQUEST_MUTATED,
                    "Conversation semantic provider mutated its request.",
                ),
            ),
        )
    try:
        decision = _parse_output(raw_output)
    except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        return _review(
            request,
            authority_fingerprint=authority,
            issues=(
                _issue(
                    ConversationSemanticIssueCode.PROVIDER_OUTPUT_INVALID,
                    "%s:%s" % (type(exc).__name__, str(exc)[:300]),
                ),
            ),
        )
    issues = _validate_decision(request, decision)
    return _review(
        request,
        authority_fingerprint=authority,
        decision=decision,
        issues=issues,
    )
