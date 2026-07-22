from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from merchant_ai.config import Settings
from merchant_ai.models import ResultCoverage
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationReferentCandidate,
    ConversationSemanticReview,
    build_conversation_semantic_resolver_request,
)


GROUNDED_CONVERSATION_STATE_VERSION = "grounded_conversation_state.v1"


class GroundedConversationStateError(RuntimeError):
    """Base error for durable Grounded conversation state."""


class GroundedConversationStateCorruptError(GroundedConversationStateError):
    """The persisted state cannot be trusted or restored."""


class GroundedConversationStateConflictError(GroundedConversationStateError):
    """The state changed after the caller loaded it."""


@dataclass(frozen=True)
class GroundedConversationState:
    thread_id: str
    snapshot: dict[str, Any]
    revision: int
    updated_at: str
    version: str = GROUNDED_CONVERSATION_STATE_VERSION


StateUpdater = Callable[[Optional[dict[str, Any]]], Mapping[str, Any]]


@dataclass(frozen=True)
class GroundedConversationResolution:
    """Server-side interpretation of one turn before Topic routing.

    ``effective_question`` remains byte-semantically equal to the normalized
    current user utterance. ``retrieval_question`` is a non-authoritative,
    retrieval-only contextualization. Cross-turn execution authority travels
    only through the typed reference Contract and trace fields.
    """

    original_question: str
    effective_question: str
    retrieval_question: str = ""
    status: str = "STANDALONE"
    reference_detected: bool = False
    reference_phrases: tuple[str, ...] = ()
    antecedent_question: str = ""
    inherited_time_expression: str = ""
    inherited_filters: tuple[str, ...] = ()
    source_revision: int = 0
    source_artifact_ids: tuple[str, ...] = ()
    source: str = ""
    clarification_question: str = ""
    clarification_options: tuple[str, ...] = ()
    clarification_type: str = ""
    semantic_authority_fingerprint: str = ""
    semantic_request_fingerprint: str = ""
    semantic_decision_fingerprint: str = ""
    semantic_candidate_set_fingerprint: str = ""
    semantic_issue_codes: tuple[str, ...] = ()
    pending_clarification_stage: str = ""
    pending_clarification_type: str = ""
    pending_clarification_question: str = ""
    pending_clarification_answer: str = ""
    pending_clarification_options: tuple[str, ...] = ()
    reference_contract: "GroundedReferenceContractResolution" = field(
        default_factory=lambda: GroundedReferenceContractResolution()
    )

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification_question)

    def trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "referenceDetected": self.reference_detected,
            "referencePhrases": list(self.reference_phrases),
            "antecedentQuestion": self.antecedent_question,
            "inheritedTimeExpression": self.inherited_time_expression,
            "inheritedFilters": list(self.inherited_filters),
            "sourceRevision": self.source_revision,
            "sourceArtifactIds": list(self.source_artifact_ids),
            "source": self.source,
            "originalQuestion": self.original_question,
            "effectiveQuestion": self.effective_question,
            "retrievalQuestion": (
                self.retrieval_question or self.effective_question
            ),
            "needsClarification": self.needs_clarification,
            "clarificationType": self.clarification_type,
            "semanticAuthorityFingerprint": self.semantic_authority_fingerprint,
            "semanticRequestFingerprint": self.semantic_request_fingerprint,
            "semanticDecisionFingerprint": self.semantic_decision_fingerprint,
            "semanticCandidateSetFingerprint": self.semantic_candidate_set_fingerprint,
            "semanticIssueCodes": list(self.semantic_issue_codes),
            "pendingClarification": {
                "stage": self.pending_clarification_stage,
                "type": self.pending_clarification_type,
                "question": self.pending_clarification_question,
                "answer": self.pending_clarification_answer,
                "options": list(self.pending_clarification_options),
            },
            "referenceContract": self.reference_contract.trace(),
        }


@dataclass(frozen=True)
class GroundedReferenceContractResolution:
    """Typed meaning of a cross-turn BI reference.

    A conversational antecedent is not always a row population. Keeping
    predicate scopes, exact entity sets, result artifacts and metric values
    distinct prevents downstream planning from coercing every reference into
    an entity list or a repeated time filter.
    """

    status: str = "NONE"
    referent_type: str = "NONE"
    downstream_operation: str = "UNSPECIFIED"
    source_artifact_id: str = ""
    source_contract_fingerprint: str = ""
    source_sql_fingerprint: str = ""
    source_query_shape: str = ""
    source_topics: tuple[str, ...] = ()
    source_tables: tuple[str, ...] = ()
    source_goal_ids: tuple[str, ...] = ()
    source_entity_identities: tuple[str, ...] = ()
    source_data_grains: tuple[str, ...] = ()
    coverage_status: str = "UNKNOWN"
    snapshot_semantics: str = "ABSOLUTE_PREDICATE_SNAPSHOT"
    population_required: bool = False
    complete_membership_required: bool = False
    membership_handle_type: str = ""
    membership_handle_id: str = ""
    membership_values_hash: str = ""
    current_turn_replaces_time_scope: bool = False
    reason: str = ""

    @property
    def bound(self) -> bool:
        return self.status == "BOUND" and bool(self.referent_type)

    def trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "referentType": self.referent_type,
            "downstreamOperation": self.downstream_operation,
            "sourceArtifactId": self.source_artifact_id,
            "sourceContractFingerprint": self.source_contract_fingerprint,
            "sourceSqlFingerprint": self.source_sql_fingerprint,
            "sourceQueryShape": self.source_query_shape,
            "sourceTopics": list(self.source_topics),
            "sourceTables": list(self.source_tables),
            "sourceGoalIds": list(self.source_goal_ids),
            "sourceEntityIdentities": list(self.source_entity_identities),
            "sourceDataGrains": list(self.source_data_grains),
            "coverageStatus": self.coverage_status,
            "snapshotSemantics": self.snapshot_semantics,
            "populationRequired": self.population_required,
            "completeMembershipRequired": self.complete_membership_required,
            "membershipHandleType": self.membership_handle_type,
            "membershipHandleId": self.membership_handle_id,
            "membershipValuesHash": self.membership_values_hash,
            "currentTurnReplacesTimeScope": self.current_turn_replaces_time_scope,
            "reason": self.reason,
        }


def grounded_conversation_principal_fingerprint(
    merchant_id: str,
    user_scope: Mapping[str, Any] | None,
) -> str:
    """Fingerprint the authorization scope that is allowed to reuse a state."""

    scope = dict(user_scope or {})
    payload = {
        "merchantId": str(merchant_id or scope.get("merchantId") or "").strip(),
        "userId": str(scope.get("userId") or scope.get("user_id") or "").strip(),
        "role": str(scope.get("role") or "").strip(),
        "region": str(scope.get("region") or "").strip(),
        "storeIds": sorted(
            {
                str(item).strip()
                for item in (scope.get("storeIds") or scope.get("store_ids") or [])
                if str(item).strip()
            }
        ),
        "permissions": sorted(
            {
                str(item).strip()
                for item in (scope.get("permissions") or [])
                if str(item).strip()
            }
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_grounded_conversation_turn(
    question: str,
    *,
    semantic_review: ConversationSemanticReview | None = None,
    verified_candidates: Sequence[ConversationReferentCandidate] = (),
    persisted_snapshot: Mapping[str, Any] | None = None,
    persisted_revision: int = 0,
    message_history: Sequence[Any] | None = None,
    request_context: Any = None,
) -> GroundedConversationResolution:
    """Resolve one turn only from a trusted structured semantic review.

    Persisted assistant prose, visible previews and untyped message history are
    never candidate authority. The provider decides natural-language meaning;
    this function only validates fingerprints, performs an exact artifact-id
    lookup and projects retained structured fields into a reference Contract.
    """

    original = str(question or "").strip()
    if not original:
        raise ValueError("conversation question is required")
    del message_history
    snapshot = dict(persisted_snapshot or {})
    revision = max(0, int(persisted_revision or 0))
    antecedent = _server_retained_antecedent(snapshot)
    active_scope = snapshot.get("activeScope")
    has_persisted_artifact_claim = bool(
        isinstance(active_scope, Mapping)
        and (
            active_scope.get("artifactIds")
            or active_scope.get("sourceArtifacts")
        )
    )
    pending = _pending_clarification(snapshot, request_context)
    if pending:
        pending_question = str(pending.get("pendingQuestion") or "").strip()
        if pending_question:
            return GroundedConversationResolution(
                original_question=original,
                effective_question=original,
                status="CLARIFICATION_RESUMED",
                antecedent_question=pending_question,
                source_revision=revision,
                source="SERVER_PENDING_CLARIFICATION",
                pending_clarification_stage=str(
                    pending.get("stage") or ""
                ).strip(),
                pending_clarification_type=str(
                    pending.get("type") or ""
                ).strip(),
                pending_clarification_question=pending_question,
                pending_clarification_answer=original,
                pending_clarification_options=_text_tuple(
                    pending.get("options") or ()
                ),
            )

    candidates = tuple(verified_candidates or ())
    if any(
        not isinstance(candidate, ConversationReferentCandidate)
        for candidate in candidates
    ):
        return _semantic_clarification(
            original,
            status="SEMANTIC_CANDIDATE_SET_INVALID",
            clarification_type="CONVERSATION_CANDIDATE_SET_INVALID",
            clarification_question=(
                "服务端保留的上一轮结果范围无法通过完整性校验。"
                "请重新说明本轮所需的数据范围。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=semantic_review,
        )
    try:
        request = build_conversation_semantic_resolver_request(
            original,
            candidates,
        )
    except (TypeError, ValueError):
        return _semantic_clarification(
            original,
            status="SEMANTIC_CANDIDATE_SET_INVALID",
            clarification_type="CONVERSATION_CANDIDATE_SET_INVALID",
            clarification_question=(
                "服务端保留的上一轮结果范围无法通过完整性校验。"
                "请重新说明本轮所需的数据范围。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=semantic_review,
        )

    if not candidates and not has_persisted_artifact_claim:
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="STANDALONE",
            antecedent_question=antecedent,
            source_revision=revision,
            source="NO_RETAINED_VERIFIED_ARTIFACT",
        )

    semantic_fields = _semantic_trace_fields(semantic_review, request)
    if semantic_review is None:
        return _semantic_clarification(
            original,
            status="SEMANTIC_REVIEW_UNAVAILABLE",
            clarification_type="CONVERSATION_SEMANTIC_REVIEW_UNAVAILABLE",
            clarification_question=(
                "本轮的跨轮语义复核暂不可用。为避免沿用错误范围，"
                "请明确说明是否需要基于上一轮结果继续。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=None,
            request=request,
        )
    decision = semantic_review.decision
    if (
        not semantic_review.accepted
        or decision is None
        or semantic_review.issues
    ):
        return _semantic_clarification(
            original,
            status="SEMANTIC_REVIEW_REJECTED",
            clarification_type="CONVERSATION_SEMANTIC_REVIEW_REJECTED",
            clarification_question=(
                "本轮的跨轮语义复核没有形成可信结论。"
                "请明确说明要沿用的上一轮结果范围。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=semantic_review,
            request=request,
        )
    if not _review_matches_request(semantic_review, request):
        return _semantic_clarification(
            original,
            status="SEMANTIC_REVIEW_BINDING_MISMATCH",
            clarification_type="CONVERSATION_SEMANTIC_REVIEW_BINDING_MISMATCH",
            clarification_question=(
                "跨轮语义结论与当前问题或候选结果不一致。"
                "请重新说明本轮所需的数据范围。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=semantic_review,
            request=request,
        )

    if not decision.reference_detected:
        if not _non_reference_decision_is_clean(decision):
            return _semantic_clarification(
                original,
                status="SEMANTIC_REVIEW_BINDING_MISMATCH",
                clarification_type=(
                    "CONVERSATION_SEMANTIC_REVIEW_BINDING_MISMATCH"
                ),
                clarification_question=(
                    "跨轮语义结论包含互相冲突的结构字段。"
                    "请重新说明本轮问题。"
                ),
                revision=revision,
                antecedent=antecedent,
                review=semantic_review,
                request=request,
            )
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="STANDALONE",
            source_revision=revision,
            source="VERIFIED_CONVERSATION_SEMANTIC_REVIEW",
            **semantic_fields,
        )

    if decision.ambiguous:
        return GroundedConversationResolution(
            original_question=original,
            effective_question=original,
            status="AMBIGUOUS_REFERENCE",
            reference_detected=True,
            reference_phrases=tuple(decision.reference_phrases),
            antecedent_question=antecedent,
            source_revision=revision,
            source="VERIFIED_CONVERSATION_SEMANTIC_REVIEW",
            clarification_question=(
                "本轮引用可能对应多个已验证结果，请选择要继续使用的结果。"
            ),
            clarification_options=_candidate_options(candidates),
            clarification_type="CONVERSATION_REFERENCE_AMBIGUOUS",
            reference_contract=GroundedReferenceContractResolution(
                status="AMBIGUOUS",
                referent_type="AMBIGUOUS",
                downstream_operation=_protocol_value(
                    decision.downstream_operation
                ),
                population_required=decision.population_required,
                complete_membership_required=(
                    decision.complete_membership_required
                ),
                current_turn_replaces_time_scope=(
                    decision.current_turn_replaces_time_scope
                ),
                reason="SEMANTIC_REVIEW_AMBIGUOUS",
            ),
            **semantic_fields,
        )

    candidates_by_id = {
        candidate.artifact_id: candidate for candidate in candidates
    }
    candidate = candidates_by_id.get(decision.selected_artifact_id)
    if candidate is None:
        return _semantic_clarification(
            original,
            status="SEMANTIC_REVIEW_BINDING_MISMATCH",
            clarification_type="CONVERSATION_SELECTED_ARTIFACT_UNVERIFIED",
            clarification_question=(
                "跨轮语义结论引用了不在服务端已验证集合中的结果。"
                "请重新选择上一轮结果范围。"
            ),
            revision=revision,
            antecedent=antecedent,
            review=semantic_review,
            request=request,
            reference_detected=True,
            reference_phrases=tuple(decision.reference_phrases),
        )

    referent_type = _contract_referent_type(decision.referent_type)
    if not referent_type:
        return _unsafe_reference_clarification(
            original,
            decision=decision,
            candidate=candidate,
            revision=revision,
            antecedent=antecedent,
            semantic_fields=semantic_fields,
            reason="REFERENT_TYPE_NOT_EXECUTABLE",
        )
    unsafe_reason = _reference_candidate_failure(
        referent_type,
        decision.population_required,
        decision.complete_membership_required,
        candidate,
    )
    if unsafe_reason:
        return _unsafe_reference_clarification(
            original,
            decision=decision,
            candidate=candidate,
            revision=revision,
            antecedent=antecedent,
            semantic_fields=semantic_fields,
            reason=unsafe_reason,
        )

    reference_contract = GroundedReferenceContractResolution(
        status="BOUND",
        referent_type=referent_type,
        downstream_operation=_protocol_value(
            decision.downstream_operation
        ),
        source_artifact_id=candidate.artifact_id,
        source_contract_fingerprint=candidate.contract_fingerprint,
        source_sql_fingerprint=candidate.sql_fingerprint,
        source_query_shape=candidate.query_shape,
        source_topics=tuple(candidate.topic_ids),
        source_tables=tuple(candidate.table_ids),
        source_goal_ids=tuple(candidate.goal_ids),
        source_entity_identities=tuple(candidate.entity_identities),
        source_data_grains=tuple(candidate.data_grains),
        coverage_status=candidate.coverage_status,
        snapshot_semantics=(
            candidate.snapshot_semantics
            or "ABSOLUTE_PREDICATE_SNAPSHOT"
        ),
        population_required=decision.population_required,
        complete_membership_required=(
            decision.complete_membership_required
        ),
        membership_handle_type=candidate.membership_handle_type,
        membership_handle_id=candidate.membership_handle_id,
        membership_values_hash=candidate.membership_values_hash,
        current_turn_replaces_time_scope=(
            decision.current_turn_replaces_time_scope
        ),
        reason="VERIFIED_SEMANTIC_REVIEW_BINDING",
    )
    inherited_times = (
        ()
        if decision.current_turn_replaces_time_scope
        else tuple(candidate.time_scope_labels)
    )
    return GroundedConversationResolution(
        original_question=original,
        effective_question=original,
        retrieval_question=(
            str(decision.retrieval_question or "").strip() or original
        ),
        status="RESOLVED_REFERENCE",
        reference_detected=True,
        reference_phrases=tuple(decision.reference_phrases),
        antecedent_question=antecedent,
        inherited_time_expression=(
            inherited_times[0] if len(inherited_times) == 1 else ""
        ),
        inherited_filters=tuple(candidate.filter_scope_labels),
        source_revision=revision,
        source_artifact_ids=(candidate.artifact_id,),
        source="VERIFIED_CONVERSATION_SEMANTIC_REVIEW",
        reference_contract=reference_contract,
        **semantic_fields,
    )


def _semantic_trace_fields(
    review: ConversationSemanticReview | None,
    request: Any,
) -> dict[str, Any]:
    return {
        "semantic_authority_fingerprint": (
            str(review.authority_fingerprint or "") if review else ""
        ),
        "semantic_request_fingerprint": request.request_fingerprint,
        "semantic_decision_fingerprint": (
            str(review.decision_fingerprint or "") if review else ""
        ),
        "semantic_candidate_set_fingerprint": (
            request.candidate_set_fingerprint
        ),
        "semantic_issue_codes": tuple(
            _protocol_value(issue.code)
            for issue in (review.issues if review else ())
        ),
    }


def _semantic_clarification(
    original: str,
    *,
    status: str,
    clarification_type: str,
    clarification_question: str,
    revision: int,
    antecedent: str,
    review: ConversationSemanticReview | None,
    request: Any = None,
    reference_detected: bool = False,
    reference_phrases: tuple[str, ...] = (),
) -> GroundedConversationResolution:
    semantic_fields = (
        _semantic_trace_fields(review, request)
        if request is not None
        else {
            "semantic_authority_fingerprint": (
                str(review.authority_fingerprint or "") if review else ""
            ),
            "semantic_decision_fingerprint": (
                str(review.decision_fingerprint or "") if review else ""
            ),
            "semantic_issue_codes": tuple(
                _protocol_value(issue.code) for issue in (
                    review.issues if review else ()
                )
            ),
        }
    )
    return GroundedConversationResolution(
        original_question=original,
        effective_question=original,
        status=status,
        reference_detected=reference_detected,
        reference_phrases=reference_phrases,
        antecedent_question=antecedent,
        source_revision=revision,
        source="CONVERSATION_SEMANTIC_REVIEW",
        clarification_question=clarification_question,
        clarification_type=clarification_type,
        **semantic_fields,
    )


def _review_matches_request(
    review: ConversationSemanticReview,
    request: Any,
) -> bool:
    decision = review.decision
    if decision is None:
        return False
    return bool(
        review.authority_fingerprint
        and review.decision_fingerprint
        and review.request_fingerprint == request.request_fingerprint
        and decision.complete
        and decision.request_fingerprint == request.request_fingerprint
        and decision.question_fingerprint == request.question_fingerprint
        and decision.candidate_set_fingerprint
        == request.candidate_set_fingerprint
    )


def _non_reference_decision_is_clean(decision: Any) -> bool:
    return not any(
        (
            decision.ambiguous,
            bool(decision.selected_artifact_id),
            decision.referent_type != ConversationReferenceType.NONE,
            decision.downstream_operation
            != ConversationDownstreamOperation.UNSPECIFIED,
            decision.population_required,
            decision.complete_membership_required,
            decision.current_turn_replaces_time_scope,
            bool(decision.reference_phrases),
            bool(str(decision.retrieval_question or "").strip()),
        )
    )


def _contract_referent_type(
    referent_type: ConversationReferenceType,
) -> str:
    mappings = {
        ConversationReferenceType.PREDICATE_SCOPE.value: "PREDICATE_SCOPE",
        ConversationReferenceType.VERIFIED_ENTITY_SET.value: "ENTITY_SET",
        ConversationReferenceType.RESULT_ARTIFACT.value: "RESULT_ARTIFACT",
        ConversationReferenceType.METRIC_VALUE.value: "METRIC_VALUE",
    }
    return mappings.get(_protocol_value(referent_type), "")


def _protocol_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _reference_candidate_failure(
    referent_type: str,
    population_required: bool,
    complete_membership_required: bool,
    candidate: ConversationReferentCandidate,
) -> str:
    if referent_type == "METRIC_VALUE" and population_required:
        return "METRIC_VALUE_DOES_NOT_DEFINE_POPULATION"
    membership_population = bool(
        population_required
        and referent_type in {"ENTITY_SET", "RESULT_ARTIFACT"}
    )
    if membership_population and not complete_membership_required:
        return "COMPLETE_MEMBERSHIP_REQUIREMENT_MISSING"
    membership_required = bool(
        referent_type == "ENTITY_SET" or complete_membership_required
    )
    if not membership_required:
        return ""
    complete_coverages = {
        ResultCoverage.ALL_ROWS.value,
        ResultCoverage.TOP_N.value,
        "COMPLETE",
        "EXACT_ENTITY_SET",
    }
    if candidate.coverage_status not in complete_coverages:
        return "REFERENCED_RESULT_MEMBERSHIP_NOT_COMPLETE"
    if not (
        candidate.membership_handle_type
        and candidate.membership_handle_id
        and candidate.membership_values_hash
    ):
        return "REFERENCE_MEMBERSHIP_HANDLE_INCOMPLETE"
    return ""


def _unsafe_reference_clarification(
    original: str,
    *,
    decision: Any,
    candidate: ConversationReferentCandidate,
    revision: int,
    antecedent: str,
    semantic_fields: Mapping[str, Any],
    reason: str,
) -> GroundedConversationResolution:
    return GroundedConversationResolution(
        original_question=original,
        effective_question=original,
        status="UNSAFE_REFERENCE",
        reference_detected=True,
        reference_phrases=tuple(decision.reference_phrases),
        antecedent_question=antecedent,
        source_revision=revision,
        source_artifact_ids=(candidate.artifact_id,),
        source="VERIFIED_CONVERSATION_SEMANTIC_REVIEW",
        clarification_question=(
            "已选择的上一轮结果不能安全满足本轮引用要求。"
            "请改为明确的数据范围或重新选择结果。"
        ),
        clarification_options=(candidate.label or candidate.artifact_id,),
        clarification_type="CONVERSATION_REFERENCE_UNSAFE",
        reference_contract=GroundedReferenceContractResolution(
            status="UNSUPPORTED",
                referent_type=(
                    _contract_referent_type(decision.referent_type)
                    or _protocol_value(decision.referent_type)
                ),
            downstream_operation=_protocol_value(
                decision.downstream_operation
            ),
            source_artifact_id=candidate.artifact_id,
            source_contract_fingerprint=candidate.contract_fingerprint,
            source_sql_fingerprint=candidate.sql_fingerprint,
            source_query_shape=candidate.query_shape,
            coverage_status=candidate.coverage_status,
            population_required=decision.population_required,
            complete_membership_required=(
                decision.complete_membership_required
            ),
            membership_handle_type=candidate.membership_handle_type,
            membership_handle_id=candidate.membership_handle_id,
            membership_values_hash=candidate.membership_values_hash,
            current_turn_replaces_time_scope=(
                decision.current_turn_replaces_time_scope
            ),
            reason=reason,
        ),
        **dict(semantic_fields),
    )


def _candidate_options(
    candidates: Sequence[ConversationReferentCandidate],
) -> tuple[str, ...]:
    return _text_tuple(
        candidate.label or candidate.artifact_id
        for candidate in candidates
    )


def _text_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    return tuple(
        dict.fromkeys(
            str(value or "").strip()
            for value in (values or ())
            if str(value or "").strip()
        )
    )


def _server_retained_antecedent(snapshot: Mapping[str, Any]) -> str:
    last_turn = snapshot.get("lastTurn")
    if not isinstance(last_turn, Mapping):
        return ""
    return str(last_turn.get("originalQuestion") or "").strip()


def _pending_clarification(snapshot: Mapping[str, Any], request_context: Any) -> dict[str, Any]:
    pending = snapshot.get("pendingClarification")
    if isinstance(pending, Mapping) and str(pending.get("pendingQuestion") or "").strip():
        return dict(pending)
    if request_context is None:
        return {}
    question = str(getattr(request_context, "pending_question", "") or "").strip()
    stage = str(getattr(request_context, "pending_clarification_stage", "") or "").strip()
    if not question or not stage:
        return {}
    return {
        "pendingQuestion": question,
        "stage": stage,
        "type": str(getattr(request_context, "pending_clarification_type", "") or ""),
        "options": list(getattr(request_context, "pending_clarification_options", None) or []),
    }


class GroundedConversationStateStore:
    """Durable, thread-scoped business state for the Grounded runtime.

    Each thread is stored in one versioned JSON envelope under the harness
    workspace. Writes use a same-directory temporary file and ``os.replace``
    so readers observe either the old or the new complete snapshot.

    ``locked(thread_id)`` can wrap an entire load -> execute -> save business
    transaction. Store methods are re-entrant inside that context for the same
    thread. The lock combines an in-process ``RLock`` with ``flock`` so separate
    store instances and worker processes sharing the workspace are serialized.
    """

    _registry_lock = threading.Lock()
    _thread_mutexes: dict[str, threading.RLock] = {}
    _held_file_locks = threading.local()

    def __init__(self, settings: Settings, root: Path | None = None):
        self.root = Path(root) if root is not None else settings.resolved_workspace_path / "grounded_conversation_state"
        self.threads_dir = self.root / "threads"
        self.locks_dir = self.root / "locks"
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def load(self, thread_id: str) -> Optional[GroundedConversationState]:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=False):
            return self._load_unlocked(normalized_thread_id)

    def load_snapshot(self, thread_id: str) -> Optional[dict[str, Any]]:
        state = self.load(thread_id)
        return state.snapshot if state is not None else None

    def save(
        self,
        thread_id: str,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        normalized_snapshot = self._normalize_snapshot(snapshot)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            return self._save_unlocked(normalized_thread_id, normalized_snapshot, current)

    def save_snapshot(
        self,
        thread_id: str,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        return self.save(thread_id, snapshot, expected_revision=expected_revision)

    def update(
        self,
        thread_id: str,
        updater: StateUpdater,
        *,
        expected_revision: int | None = None,
    ) -> GroundedConversationState:
        """Atomically load, transform, and persist one thread snapshot."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            current_snapshot = self._json_copy(current.snapshot) if current is not None else None
            replacement = updater(current_snapshot)
            normalized_snapshot = self._normalize_snapshot(replacement)
            return self._save_unlocked(normalized_thread_id, normalized_snapshot, current)

    def clear(self, thread_id: str, *, expected_revision: int | None = None) -> bool:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            current = self._load_unlocked(normalized_thread_id)
            self._check_expected_revision(current, expected_revision)
            path = self.path_for(normalized_thread_id)
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            self._fsync_directory(path.parent)
            return True

    def exists(self, thread_id: str) -> bool:
        return self.load(thread_id) is not None

    @contextmanager
    def locked(self, thread_id: str) -> Iterator[None]:
        """Hold the exclusive thread lock across a complete business transaction."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self._thread_file_lock(normalized_thread_id, exclusive=True):
            yield

    def path_for(self, thread_id: str) -> Path:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        return self.threads_dir / (self._thread_file_stem(normalized_thread_id) + ".json")

    def lock_path_for(self, thread_id: str) -> Path:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        return self.locks_dir / (self._thread_file_stem(normalized_thread_id) + ".lock")

    def _load_unlocked(self, thread_id: str) -> Optional[GroundedConversationState]:
        path = self.path_for(thread_id)
        try:
            encoded = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GroundedConversationStateError("failed to read Grounded conversation state") from exc

        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GroundedConversationStateCorruptError("Grounded conversation state is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise GroundedConversationStateCorruptError("Grounded conversation state envelope must be an object")

        version = str(payload.get("version") or "")
        persisted_thread_id = str(payload.get("threadId") or "")
        snapshot = payload.get("snapshot")
        revision = payload.get("revision")
        updated_at = str(payload.get("updatedAt") or "")
        if version != GROUNDED_CONVERSATION_STATE_VERSION:
            raise GroundedConversationStateCorruptError("unsupported Grounded conversation state version: %s" % version)
        if persisted_thread_id != thread_id:
            raise GroundedConversationStateCorruptError("Grounded conversation state thread scope mismatch")
        if not isinstance(snapshot, dict):
            raise GroundedConversationStateCorruptError("Grounded conversation snapshot must be an object")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise GroundedConversationStateCorruptError("Grounded conversation state revision is invalid")
        if not updated_at:
            raise GroundedConversationStateCorruptError("Grounded conversation state updatedAt is missing")
        return GroundedConversationState(
            thread_id=thread_id,
            snapshot=snapshot,
            revision=revision,
            updated_at=updated_at,
            version=version,
        )

    def _save_unlocked(
        self,
        thread_id: str,
        snapshot: dict[str, Any],
        current: GroundedConversationState | None,
    ) -> GroundedConversationState:
        state = GroundedConversationState(
            thread_id=thread_id,
            snapshot=snapshot,
            revision=(current.revision if current is not None else 0) + 1,
            updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        payload = {
            "version": state.version,
            "threadId": state.thread_id,
            "revision": state.revision,
            "updatedAt": state.updated_at,
            "snapshot": state.snapshot,
        }
        self._atomic_write_json(self.path_for(thread_id), payload)
        return state

    @contextmanager
    def _thread_file_lock(self, thread_id: str, *, exclusive: bool) -> Iterator[None]:
        lock_path = self.lock_path_for(thread_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_key = str(lock_path.resolve())
        mutex = self._mutex_for(lock_key)
        with mutex:
            held = self._held_locks()
            if lock_key in held:
                if exclusive and not held[lock_key]:
                    raise GroundedConversationStateError("cannot upgrade a shared Grounded conversation state lock")
                yield
                return

            with lock_path.open("a+b") as lock_handle:
                operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_handle.fileno(), operation)
                held[lock_key] = exclusive
                try:
                    yield
                finally:
                    held.pop(lock_key, None)
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _mutex_for(cls, lock_key: str) -> threading.RLock:
        with cls._registry_lock:
            mutex = cls._thread_mutexes.get(lock_key)
            if mutex is None:
                mutex = threading.RLock()
                cls._thread_mutexes[lock_key] = mutex
            return mutex

    @classmethod
    def _held_locks(cls) -> dict[str, bool]:
        held = getattr(cls._held_file_locks, "locks", None)
        if held is None:
            held = {}
            cls._held_file_locks.locks = held
        return held

    @staticmethod
    def _check_expected_revision(
        current: GroundedConversationState | None,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is None:
            return
        actual_revision = current.revision if current is not None else 0
        if actual_revision != expected_revision:
            raise GroundedConversationStateConflictError(
                "Grounded conversation state revision conflict: expected %s, found %s"
                % (expected_revision, actual_revision)
            )

    @staticmethod
    def _normalize_thread_id(thread_id: str) -> str:
        value = str(thread_id or "").strip()
        if not value:
            raise ValueError("thread_id is required")
        if len(value) > 512 or any(ord(character) < 32 for character in value):
            raise ValueError("thread_id is invalid")
        return value

    @staticmethod
    def _normalize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Grounded conversation snapshot must be a JSON mapping")
        try:
            encoded = json.dumps(dict(snapshot), ensure_ascii=False, separators=(",", ":"))
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise TypeError("Grounded conversation snapshot must contain only JSON values") from exc
        if not isinstance(decoded, dict):
            raise TypeError("Grounded conversation snapshot must be a JSON mapping")
        return decoded

    @staticmethod
    def _json_copy(snapshot: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))

    @staticmethod
    def _thread_file_stem(thread_id: str) -> str:
        safe_prefix = "".join(
            character
            if (
                character.isascii()
                and (character.isalnum() or character in "_.-")
            )
            else "_"
            for character in thread_id
        ).strip("._-")[:80] or "thread"
        digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:16]
        return "%s--%s" % (safe_prefix, digest)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            ".%s.%s.%s.%s.tmp" % (path.name, os.getpid(), threading.get_ident(), uuid.uuid4().hex)
        )
        encoded = json.dumps(dict(payload), ensure_ascii=False, indent=2, separators=(",", ": ")) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            GroundedConversationStateStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            # Some mounted filesystems do not support directory fsync. The
            # same-directory replace remains atomic even in that environment.
            pass
        finally:
            os.close(descriptor)
