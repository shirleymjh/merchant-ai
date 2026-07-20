from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import ConfigDict, Field, ValidationError, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    parse_original_question_goal_contract,
)
from merchant_ai.services.grounded_population_verifier import (
    GoalDeclarationPopulationVerificationInput,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationSemanticExpectation,
    PopulationSemanticReview,
    goal_population_verification_input,
    population_question_fingerprint,
)


class PopulationSemanticReviewerIssueCode(str, Enum):
    EFFECTIVE_QUESTION_MISMATCH = "EFFECTIVE_QUESTION_MISMATCH"
    PROVIDER_AUTHORITY_REQUIRED = "PROVIDER_AUTHORITY_REQUIRED"
    PROVIDER_AUTHORITY_UNTRUSTED = "PROVIDER_AUTHORITY_UNTRUSTED"
    PROVIDER_NOT_INDEPENDENT = "PROVIDER_NOT_INDEPENDENT"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    PROVIDER_OUTPUT_INVALID = "PROVIDER_OUTPUT_INVALID"
    PROVIDER_OUTPUT_INCOMPLETE = "PROVIDER_OUTPUT_INCOMPLETE"
    PROVIDER_REQUEST_MUTATED = "PROVIDER_REQUEST_MUTATED"
    PROVIDER_REQUEST_BINDING_MISMATCH = "PROVIDER_REQUEST_BINDING_MISMATCH"
    PROVIDER_QUESTION_BINDING_MISMATCH = "PROVIDER_QUESTION_BINDING_MISMATCH"
    PROVIDER_SKELETON_BINDING_MISMATCH = "PROVIDER_SKELETON_BINDING_MISMATCH"
    PROVIDER_GOAL_MISSING = "PROVIDER_GOAL_MISSING"
    PROVIDER_GOAL_UNKNOWN = "PROVIDER_GOAL_UNKNOWN"
    PROVIDER_GOAL_DUPLICATE = "PROVIDER_GOAL_DUPLICATE"
    PROVIDER_SCOPE_REQUIRED = "PROVIDER_SCOPE_REQUIRED"
    PROVIDER_SCOPE_UNEXPECTED = "PROVIDER_SCOPE_UNEXPECTED"
    PROVIDER_SOURCE_GOAL_REQUIRED = "PROVIDER_SOURCE_GOAL_REQUIRED"
    PROVIDER_SOURCE_GOAL_UNEXPECTED = "PROVIDER_SOURCE_GOAL_UNEXPECTED"
    PROVIDER_SOURCE_GOAL_UNKNOWN = "PROVIDER_SOURCE_GOAL_UNKNOWN"
    PROVIDER_SOURCE_GOAL_SELF_REFERENCE = "PROVIDER_SOURCE_GOAL_SELF_REFERENCE"


_EXTERNAL_POPULATION_SCOPES = {
    "VERIFIED_ENTITY_SET",
    "VERIFIED_PREDICATE_SCOPE",
    "VERIFIED_RESULT_ARTIFACT",
}


def population_semantic_model_required(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
) -> bool:
    """Return whether the Goal ledger claims an external verified population.

    Ordinary rankings over their own current-query rows (ALL_MATCHING_ROWS)
    and typed same-query Goal lineage (SAME_AS_GOAL) are verified
    deterministically by the runtime and SQL lineage gates.  Only scopes that
    claim a verified external entity, predicate, or result artifact need the
    isolated semantic model.
    """

    parsed = parse_original_question_goal_contract(contract)
    return any(
        _text(getattr(goal, "population_scope", "")).upper()
        in _EXTERNAL_POPULATION_SCOPES
        for goal in parsed.goals
    )


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _stable_fingerprint(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_unique(values: Sequence[str], field_name: str) -> None:
    normalized = [_text(value) for value in values]
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicate values" % field_name)


_SKELETON_STRUCTURE_FIELDS = {
    "METRIC": ("resultRole",),
    "DIMENSION": ("usage",),
    "TIME_WINDOW": (
        "timeExpression",
        "start",
        "end",
        "timezone",
        "granularity",
        "appliesToGoalIds",
    ),
    "COMPARISON": (
        "comparisonType",
        "leftGoalIds",
        "rightGoalIds",
    ),
    "ENTITY": ("role", "sourceGoalIds"),
    "DEPENDENCY": (
        "dependencyType",
        "upstreamGoalIds",
        "downstreamGoalIds",
    ),
    "RULE": ("requestedAction",),
    "DETAIL": ("inputGoalIds",),
    "RANKING": (
        "metricGoalIds",
        "dimensionGoalIds",
        "direction",
        "limit",
        "limitSource",
    ),
    "ANALYSIS": (
        "analysisType",
        "inputGoalIds",
        "baselineGoalIds",
    ),
}


class PopulationGoalSkeleton(_StrictFrozenModel):
    """Minimal non-executable Goal view exposed to the independent provider."""

    goal_id: str
    kind: str
    label: str
    required: bool
    source_spans: tuple[str, ...] = ()
    depends_on_goal_ids: tuple[str, ...] = ()
    answer_structure: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationGoalSkeleton":
        if not _text(self.goal_id):
            raise ValueError("goal_id must not be empty")
        kind = _text(self.kind).upper()
        if kind not in _SKELETON_STRUCTURE_FIELDS:
            raise ValueError("unsupported structural Goal kind")
        if not _text(self.label):
            raise ValueError("label must not be empty")
        _require_unique(self.source_spans, "source_spans")
        _require_unique(self.depends_on_goal_ids, "depends_on_goal_ids")
        allowed = set(_SKELETON_STRUCTURE_FIELDS[kind])
        if not set(self.answer_structure).issubset(allowed):
            raise ValueError("answer_structure contains a non-allowlisted field")
        for value in self.answer_structure.values():
            if isinstance(value, Mapping):
                raise ValueError("answer_structure must not contain nested objects")
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, (Mapping, list, tuple, set)) for item in value
            ):
                raise ValueError("answer_structure lists must contain scalar values")
        return self


class PopulationSemanticReviewerRequest(_StrictFrozenModel):
    protocol_version: Literal["population_semantic_reviewer_request.v1"] = (
        "population_semantic_reviewer_request.v1"
    )
    request_fingerprint: str
    effective_question: str
    question_fingerprint: str
    goal_skeleton_fingerprint: str
    goal_skeleton: tuple[PopulationGoalSkeleton, ...]

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationSemanticReviewerRequest":
        if not _text(self.request_fingerprint):
            raise ValueError("request_fingerprint must not be empty")
        if not _text(self.effective_question):
            raise ValueError("effective_question must not be empty")
        if not _text(self.question_fingerprint):
            raise ValueError("question_fingerprint must not be empty")
        if not _text(self.goal_skeleton_fingerprint):
            raise ValueError("goal_skeleton_fingerprint must not be empty")
        if not self.goal_skeleton:
            raise ValueError("goal_skeleton must not be empty")
        _require_unique(
            tuple(item.goal_id for item in self.goal_skeleton),
            "goal_skeleton.goal_id",
        )
        return self


class PopulationSemanticProviderDecision(_StrictFrozenModel):
    goal_id: str
    gate_required: bool
    scope_kind: PopulationScopeKind | None = None
    source_goal_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationSemanticProviderDecision":
        if not _text(self.goal_id):
            raise ValueError("goal_id must not be empty")
        _require_unique(self.source_goal_ids, "source_goal_ids")
        return self


class PopulationSemanticProviderOutput(_StrictFrozenModel):
    protocol_version: Literal["population_semantic_provider_output.v1"] = (
        "population_semantic_provider_output.v1"
    )
    request_fingerprint: str
    question_fingerprint: str
    goal_skeleton_fingerprint: str
    complete: bool
    decisions: tuple[PopulationSemanticProviderDecision, ...]


class PopulationSemanticReviewProvider(Protocol):
    """Injected structured provider with no tool, SQL, catalog, or asset surface."""

    @property
    def authority_fingerprint(self) -> str: ...

    def review_population_semantics(
        self,
        request: PopulationSemanticReviewerRequest,
        *,
        timeout_seconds: float,
    ) -> PopulationSemanticProviderOutput | Mapping[str, Any] | str: ...


class PopulationSemanticReviewerIssue(_StrictFrozenModel):
    code: PopulationSemanticReviewerIssueCode
    message: str
    goal_id: str = ""
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    blocking: Literal[True] = True


class PopulationSemanticReviewerOutcome(_StrictFrozenModel):
    passed: bool
    review: PopulationSemanticReview | None = None
    request: PopulationSemanticReviewerRequest | None = None
    request_fingerprint: str = ""
    goal_skeleton_fingerprint: str = ""
    provider_authority_fingerprint: str = ""
    provider_output_fingerprint: str = ""
    review_fingerprint: str = ""
    provider_invoked: bool = False
    issues: tuple[PopulationSemanticReviewerIssue, ...] = ()


def population_goal_skeleton(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
) -> tuple[PopulationGoalSkeleton, ...]:
    """Project the Core contract through a population-blind field allowlist."""

    parsed = parse_original_question_goal_contract(contract)
    skeleton: list[PopulationGoalSkeleton] = []
    for goal in parsed.goals:
        payload = goal.model_dump(by_alias=True, mode="json")
        kind = _text(payload.get("kind")).upper()
        allowed_fields = _SKELETON_STRUCTURE_FIELDS.get(kind, ())
        answer_structure: dict[str, Any] = {}
        for field_name in allowed_fields:
            value = payload.get(field_name)
            if value in (None, "", [], {}, ()):
                continue
            answer_structure[field_name] = value
        skeleton.append(
            PopulationGoalSkeleton(
                goal_id=_text(payload.get("goalId")),
                kind=kind,
                label=_text(payload.get("label")),
                required=bool(payload.get("required", True)),
                source_spans=tuple(payload.get("sourceSpans") or ()),
                depends_on_goal_ids=tuple(
                    payload.get("dependsOnGoalIds") or ()
                ),
                answer_structure=answer_structure,
            )
        )
    return tuple(sorted(skeleton, key=lambda item: item.goal_id))


def population_goal_skeleton_fingerprint(
    skeleton: Sequence[PopulationGoalSkeleton],
) -> str:
    return _stable_fingerprint(
        [
            item.model_dump(by_alias=True, mode="json")
            for item in sorted(skeleton, key=lambda value: value.goal_id)
        ]
    )


def population_semantic_reviewer_request_fingerprint(
    request: PopulationSemanticReviewerRequest,
) -> str:
    payload = request.model_dump(by_alias=True, mode="json")
    payload["requestFingerprint"] = ""
    return _stable_fingerprint(payload)


def population_semantic_review_fingerprint(
    review: PopulationSemanticReview,
) -> str:
    return _stable_fingerprint(review)


def build_population_semantic_reviewer_request(
    effective_question: str,
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
) -> PopulationSemanticReviewerRequest:
    parsed = parse_original_question_goal_contract(contract)
    question = str(effective_question or "").strip()
    if question != parsed.question.strip():
        raise ValueError("effective original question does not match Goal Contract")
    skeleton = population_goal_skeleton(parsed)
    skeleton_fingerprint = population_goal_skeleton_fingerprint(skeleton)
    request = PopulationSemanticReviewerRequest(
        request_fingerprint="pending",
        effective_question=question,
        question_fingerprint=population_question_fingerprint(question),
        goal_skeleton_fingerprint=skeleton_fingerprint,
        goal_skeleton=skeleton,
    )
    return request.model_copy(
        update={
            "request_fingerprint": population_semantic_reviewer_request_fingerprint(
                request
            )
        }
    )


class IndependentPopulationSemanticReviewer:
    """Fail-closed adapter around an independent structured semantic provider."""

    def __init__(
        self,
        provider: PopulationSemanticReviewProvider,
        *,
        trusted_provider_authority_fingerprints: Sequence[str],
        timeout_seconds: float,
    ) -> None:
        self.provider = provider
        self.trusted_provider_authority_fingerprints = tuple(
            _text(value)
            for value in trusted_provider_authority_fingerprints
            if _text(value)
        )
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def review(
        self,
        *,
        effective_question: str,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        declaration_author_fingerprint: str,
    ) -> PopulationSemanticReviewerOutcome:
        try:
            request = build_population_semantic_reviewer_request(
                effective_question,
                contract,
            )
        except Exception as exc:
            return _failed_outcome(
                request=None,
                provider_authority_fingerprint="",
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.EFFECTIVE_QUESTION_MISMATCH,
                        "The effective original question and Goal Contract could not be bound: %s"
                        % _bounded_error(exc),
                        path="effectiveQuestion",
                    ),
                ),
            )

        try:
            authority = _text(self.provider.authority_fingerprint)
        except Exception as exc:
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint="",
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_REQUIRED,
                        "The semantic provider authority could not be read: %s"
                        % _bounded_error(exc),
                        path="provider.authorityFingerprint",
                    ),
                ),
            )
        authority_issues: list[PopulationSemanticReviewerIssue] = []
        if not authority:
            authority_issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_REQUIRED,
                    "The semantic provider requires a server-owned authority fingerprint.",
                    path="provider.authorityFingerprint",
                )
            )
        if authority not in set(self.trusted_provider_authority_fingerprints):
            authority_issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_UNTRUSTED,
                    "The semantic provider authority is not trusted by the server.",
                    path="provider.authorityFingerprint",
                )
            )
        if authority and authority == _text(declaration_author_fingerprint):
            authority_issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_NOT_INDEPENDENT,
                    "The semantic reviewer and Core declaration require distinct authorities.",
                    path="provider.authorityFingerprint",
                )
            )
        if authority_issues:
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=False,
                issues=tuple(authority_issues),
            )

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="population-semantic-review",
        )
        provider_request = request.model_copy(deep=True)
        future = executor.submit(
            self.provider.review_population_semantics,
            provider_request,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            raw_output = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=True,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_TIMEOUT,
                        "The independent semantic provider exceeded its declared timeout.",
                    ),
                ),
            )
        except Exception as exc:
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=True,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_FAILED,
                        "The independent semantic provider failed: %s"
                        % _bounded_error(exc),
                    ),
                ),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if (
            population_semantic_reviewer_request_fingerprint(provider_request)
            != request.request_fingerprint
            or population_goal_skeleton_fingerprint(
                provider_request.goal_skeleton
            )
            != request.goal_skeleton_fingerprint
        ):
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=True,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_REQUEST_MUTATED,
                        "The provider mutated its isolated semantic review request copy.",
                    ),
                ),
            )

        try:
            output = _parse_provider_output(raw_output)
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=True,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_OUTPUT_INVALID,
                        "The provider did not return the strict structured output contract: %s"
                        % _bounded_error(exc),
                    ),
                ),
            )
        output_fingerprint = _stable_fingerprint(output)
        issues = _validate_provider_output(request, output)
        if issues:
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=True,
                provider_output_fingerprint=output_fingerprint,
                issues=tuple(issues),
            )
        review = _build_review(request, output, authority, output_fingerprint)
        return PopulationSemanticReviewerOutcome(
            passed=True,
            review=review,
            request=request,
            request_fingerprint=request.request_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            provider_authority_fingerprint=authority,
            provider_output_fingerprint=output_fingerprint,
            review_fingerprint=population_semantic_review_fingerprint(review),
            provider_invoked=True,
            issues=(),
        )

    def attest_declared_independent(
        self,
        *,
        effective_question: str,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        declaration_author_fingerprint: str,
    ) -> PopulationSemanticReviewerOutcome:
        """Attest a contract that declares no cross-population dependency.

        This is the deterministic fast path used for ordinary scalar, grouped,
        ranking and detail queries.  It preserves the independent reviewer
        authority and the same cryptographic bindings without spending a model
        call to rediscover that every Goal is non-gated.
        """

        try:
            request = build_population_semantic_reviewer_request(
                effective_question,
                contract,
            )
            authority = _text(self.provider.authority_fingerprint)
        except Exception as exc:
            return _failed_outcome(
                request=None,
                provider_authority_fingerprint="",
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_REQUIRED,
                        "The deterministic population attestation could not be bound: %s"
                        % _bounded_error(exc),
                    ),
                ),
            )
        if (
            not authority
            or authority
            not in set(self.trusted_provider_authority_fingerprints)
            or authority == _text(declaration_author_fingerprint)
        ):
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_UNTRUSTED,
                        "The deterministic population attestation authority is not trusted or independent.",
                    ),
                ),
            )
        output = PopulationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            complete=True,
            decisions=tuple(
                PopulationSemanticProviderDecision(
                    goal_id=item.goal_id,
                    gate_required=False,
                )
                for item in request.goal_skeleton
            ),
        )
        output_fingerprint = _stable_fingerprint(output)
        review = _build_review(
            request,
            output,
            authority,
            output_fingerprint,
        )
        return PopulationSemanticReviewerOutcome(
            passed=True,
            review=review,
            request=request,
            request_fingerprint=request.request_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            provider_authority_fingerprint=authority,
            provider_output_fingerprint=output_fingerprint,
            review_fingerprint=population_semantic_review_fingerprint(review),
            provider_invoked=False,
            issues=(),
        )

    def attest_declared_current_query(
        self,
        *,
        effective_question: str,
        contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
        declaration_author_fingerprint: str,
    ) -> PopulationSemanticReviewerOutcome:
        """Deterministically attest non-external current-query populations.

        SAME_AS_GOAL remains population-gated for later graph/SQL lineage
        verification, but it does not spend a model call: its complete source
        Goal set is already typed and validated in the Goal Contract.  An
        external VERIFIED_* scope is never accepted through this fast path.
        """

        try:
            parsed = parse_original_question_goal_contract(contract)
            request = build_population_semantic_reviewer_request(
                effective_question,
                parsed,
            )
            authority = _text(self.provider.authority_fingerprint)
        except Exception as exc:
            return _failed_outcome(
                request=None,
                provider_authority_fingerprint="",
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_REQUIRED,
                        "The deterministic current-query population attestation could not be bound: %s"
                        % _bounded_error(exc),
                    ),
                ),
            )
        if population_semantic_model_required(parsed):
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_FAILED,
                        "An external verified population requires the independent semantic model.",
                    ),
                ),
            )
        if (
            not authority
            or authority
            not in set(self.trusted_provider_authority_fingerprints)
            or authority == _text(declaration_author_fingerprint)
        ):
            return _failed_outcome(
                request=request,
                provider_authority_fingerprint=authority,
                provider_invoked=False,
                issues=(
                    _issue(
                        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_UNTRUSTED,
                        "The deterministic population attestation authority is not trusted or independent.",
                    ),
                ),
            )
        goal_map = parsed.goal_map()
        decisions: list[PopulationSemanticProviderDecision] = []
        for item in request.goal_skeleton:
            goal = goal_map[item.goal_id]
            same_query_population = (
                _text(getattr(goal, "population_scope", "")).upper()
                == PopulationScopeKind.SAME_AS_GOAL.value
            )
            decisions.append(
                PopulationSemanticProviderDecision(
                    goal_id=item.goal_id,
                    gate_required=same_query_population,
                    scope_kind=(
                        PopulationScopeKind.SAME_AS_GOAL
                        if same_query_population
                        else None
                    ),
                    source_goal_ids=(
                        tuple(getattr(goal, "population_goal_ids", ()) or ())
                        if same_query_population
                        else ()
                    ),
                )
            )
        output = PopulationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            complete=True,
            decisions=tuple(decisions),
        )
        output_fingerprint = _stable_fingerprint(output)
        review = _build_review(
            request,
            output,
            authority,
            output_fingerprint,
        )
        return PopulationSemanticReviewerOutcome(
            passed=True,
            review=review,
            request=request,
            request_fingerprint=request.request_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            provider_authority_fingerprint=authority,
            provider_output_fingerprint=output_fingerprint,
            review_fingerprint=population_semantic_review_fingerprint(review),
            provider_invoked=False,
            issues=(),
        )


def review_goal_contract_population_semantics(
    *,
    effective_question: str,
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    declaration_author_fingerprint: str,
    provider: PopulationSemanticReviewProvider,
    trusted_provider_authority_fingerprints: Sequence[str],
    timeout_seconds: float,
) -> PopulationSemanticReviewerOutcome:
    return IndependentPopulationSemanticReviewer(
        provider,
        trusted_provider_authority_fingerprints=(
            trusted_provider_authority_fingerprints
        ),
        timeout_seconds=timeout_seconds,
    ).review(
        effective_question=effective_question,
        contract=contract,
        declaration_author_fingerprint=declaration_author_fingerprint,
    )


def goal_declaration_population_input_from_review(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    outcome: PopulationSemanticReviewerOutcome,
    *,
    declaration_author_fingerprint: str,
    trusted_semantic_verifier_fingerprints: Sequence[str],
) -> GoalDeclarationPopulationVerificationInput:
    """Adapt a successful independent review into the online Goal gate input."""

    if not outcome.passed or outcome.review is None or outcome.request is None:
        raise ValueError("a passed independent population semantic review is required")
    parsed = parse_original_question_goal_contract(contract)
    rebuilt = build_population_semantic_reviewer_request(parsed.question, parsed)
    if rebuilt.request_fingerprint != outcome.request_fingerprint:
        raise ValueError("semantic review request does not match the Goal Contract")
    consumer_goal_ids = tuple(
        item.consumer_goal_id for item in outcome.review.expectations
    )
    return goal_population_verification_input(
        parsed,
        semantic_review=outcome.review,
        declaration_author_fingerprint=declaration_author_fingerprint,
        trusted_semantic_verifier_fingerprints=(
            trusted_semantic_verifier_fingerprints
        ),
        consumer_goal_ids=consumer_goal_ids,
    )


def _parse_provider_output(raw: Any) -> PopulationSemanticProviderOutput:
    if isinstance(raw, PopulationSemanticProviderOutput):
        return raw.model_copy(deep=True)
    if isinstance(raw, str):
        parsed = json.loads(raw)
    elif isinstance(raw, Mapping):
        parsed = dict(raw)
    else:
        raise TypeError("provider output must be a structured object or JSON object")
    if not isinstance(parsed, Mapping):
        raise TypeError("provider output JSON must contain an object")
    return PopulationSemanticProviderOutput.model_validate(parsed)


def _validate_provider_output(
    request: PopulationSemanticReviewerRequest,
    output: PopulationSemanticProviderOutput,
) -> list[PopulationSemanticReviewerIssue]:
    issues: list[PopulationSemanticReviewerIssue] = []
    if not output.complete:
        issues.append(
            _issue(
                PopulationSemanticReviewerIssueCode.PROVIDER_OUTPUT_INCOMPLETE,
                "The provider did not attest complete Goal skeleton review.",
                path="complete",
            )
        )
    bindings = (
        (
            PopulationSemanticReviewerIssueCode.PROVIDER_REQUEST_BINDING_MISMATCH,
            output.request_fingerprint,
            request.request_fingerprint,
            "requestFingerprint",
        ),
        (
            PopulationSemanticReviewerIssueCode.PROVIDER_QUESTION_BINDING_MISMATCH,
            output.question_fingerprint,
            request.question_fingerprint,
            "questionFingerprint",
        ),
        (
            PopulationSemanticReviewerIssueCode.PROVIDER_SKELETON_BINDING_MISMATCH,
            output.goal_skeleton_fingerprint,
            request.goal_skeleton_fingerprint,
            "goalSkeletonFingerprint",
        ),
    )
    for code, actual, expected, path in bindings:
        if actual != expected:
            issues.append(
                _issue(
                    code,
                    "The provider output is bound to a different semantic review input.",
                    path=path,
                )
            )

    skeleton_goal_ids = {item.goal_id for item in request.goal_skeleton}
    decisions: dict[str, PopulationSemanticProviderDecision] = {}
    duplicate_goal_ids: set[str] = set()
    for decision in output.decisions:
        if decision.goal_id in decisions:
            duplicate_goal_ids.add(decision.goal_id)
        else:
            decisions[decision.goal_id] = decision
    for goal_id in sorted(duplicate_goal_ids):
        issues.append(
            _issue(
                PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_DUPLICATE,
                "The provider returned more than one population decision for a Goal.",
                goal_id=goal_id,
                path="decisions",
            )
        )
    for goal_id in sorted(skeleton_goal_ids - set(decisions)):
        issues.append(
            _issue(
                PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_MISSING,
                "The provider omitted a Goal from its complete population review.",
                goal_id=goal_id,
                path="decisions",
            )
        )
    for goal_id in sorted(set(decisions) - skeleton_goal_ids):
        issues.append(
            _issue(
                PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_UNKNOWN,
                "The provider introduced a Goal absent from the population-blind skeleton.",
                goal_id=goal_id,
                path="decisions",
            )
        )
    for goal_id in sorted(skeleton_goal_ids & set(decisions)):
        decision = decisions[goal_id]
        scope_kind = _text(decision.scope_kind)
        if decision.gate_required and not scope_kind:
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SCOPE_REQUIRED,
                    "A population-gated Goal requires a typed scope kind.",
                    goal_id=goal_id,
                    path="decisions.scopeKind",
                )
            )
        if not decision.gate_required and (
            scope_kind or decision.source_goal_ids
        ):
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SCOPE_UNEXPECTED,
                    "A non-gated Goal must not carry population scope claims.",
                    goal_id=goal_id,
                    path="decisions",
                )
            )
        source_goal_ids = set(decision.source_goal_ids)
        for source_goal_id in sorted(source_goal_ids - skeleton_goal_ids):
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_UNKNOWN,
                    "A population source Goal is absent from the reviewed skeleton.",
                    goal_id=goal_id,
                    path="decisions.sourceGoalIds",
                    details={"sourceGoalId": source_goal_id},
                )
            )
        if goal_id in source_goal_ids:
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_SELF_REFERENCE,
                    "A Goal cannot use itself as its upstream population.",
                    goal_id=goal_id,
                    path="decisions.sourceGoalIds",
                )
            )
        source_required = scope_kind in {
            PopulationScopeKind.SAME_AS_GOAL.value,
            PopulationScopeKind.VERIFIED_ENTITY_SET.value,
        }
        source_forbidden = scope_kind in {
            PopulationScopeKind.UNIVERSE.value,
            PopulationScopeKind.INDEPENDENT.value,
            PopulationScopeKind.PREDICATE_SCOPE.value,
            PopulationScopeKind.VERIFIED_RESULT_ARTIFACT.value,
        }
        if decision.gate_required and source_required and not source_goal_ids:
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_REQUIRED,
                    "This population scope requires an explicit upstream Goal.",
                    goal_id=goal_id,
                    path="decisions.sourceGoalIds",
                )
            )
        if decision.gate_required and source_forbidden and source_goal_ids:
            issues.append(
                _issue(
                    PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_UNEXPECTED,
                    "This population scope must not claim a current-turn source Goal.",
                    goal_id=goal_id,
                    path="decisions.sourceGoalIds",
                )
            )
    return _dedupe_issues(issues)


def _build_review(
    request: PopulationSemanticReviewerRequest,
    output: PopulationSemanticProviderOutput,
    authority: str,
    output_fingerprint: str,
) -> PopulationSemanticReview:
    decisions = {item.goal_id: item for item in output.decisions}
    expectations: list[PopulationSemanticExpectation] = []
    exact_membership_kinds = {
        PopulationScopeKind.SAME_AS_GOAL.value,
        PopulationScopeKind.VERIFIED_ENTITY_SET.value,
        PopulationScopeKind.VERIFIED_RESULT_ARTIFACT.value,
    }
    for skeleton_goal in request.goal_skeleton:
        decision = decisions[skeleton_goal.goal_id]
        if not decision.gate_required or decision.scope_kind is None:
            continue
        scope_kind = PopulationScopeKind(str(decision.scope_kind))
        expectation_fingerprint = _stable_fingerprint(
            {
                "requestFingerprint": request.request_fingerprint,
                "goalId": skeleton_goal.goal_id,
                "scopeKind": scope_kind.value,
                "sourceGoalIds": sorted(decision.source_goal_ids),
            }
        )
        expectations.append(
            PopulationSemanticExpectation(
                expectation_id="population_expectation_%s"
                % expectation_fingerprint,
                consumer_goal_id=skeleton_goal.goal_id,
                expected_scope=PopulationScopeDescriptor(
                    scope_id="semantic_review:%s" % skeleton_goal.goal_id,
                    kind=scope_kind,
                    source_goal_ids=tuple(sorted(decision.source_goal_ids)),
                    complete_membership_required=(
                        scope_kind.value in exact_membership_kinds
                    ),
                ),
            )
        )
    review_id = "population_review_%s" % _stable_fingerprint(
        {
            "requestFingerprint": request.request_fingerprint,
            "providerAuthorityFingerprint": authority,
            "providerOutputFingerprint": output_fingerprint,
        }
    )
    return PopulationSemanticReview(
        review_id=review_id,
        question_fingerprint=request.question_fingerprint,
        goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
        provider_request_fingerprint=request.request_fingerprint,
        provider_output_fingerprint=output_fingerprint,
        verifier_fingerprint=authority,
        complete=True,
        expectations=tuple(expectations),
    )


def _failed_outcome(
    *,
    request: PopulationSemanticReviewerRequest | None,
    provider_authority_fingerprint: str,
    provider_invoked: bool,
    issues: Sequence[PopulationSemanticReviewerIssue],
    provider_output_fingerprint: str = "",
) -> PopulationSemanticReviewerOutcome:
    review: PopulationSemanticReview | None = None
    review_fingerprint = ""
    if request is not None and provider_authority_fingerprint:
        review = PopulationSemanticReview(
            review_id="population_review_failed_%s"
            % _stable_fingerprint(
                {
                    "requestFingerprint": request.request_fingerprint,
                    "providerAuthorityFingerprint": (
                        provider_authority_fingerprint
                    ),
                    "issueCodes": sorted(str(item.code) for item in issues),
                }
            ),
            question_fingerprint=request.question_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            provider_request_fingerprint=request.request_fingerprint,
            provider_output_fingerprint=provider_output_fingerprint,
            verifier_fingerprint=provider_authority_fingerprint,
            complete=False,
            expectations=(),
        )
        review_fingerprint = population_semantic_review_fingerprint(review)
    return PopulationSemanticReviewerOutcome(
        passed=False,
        review=review,
        request=request,
        request_fingerprint=(request.request_fingerprint if request else ""),
        goal_skeleton_fingerprint=(
            request.goal_skeleton_fingerprint if request else ""
        ),
        provider_authority_fingerprint=provider_authority_fingerprint,
        provider_output_fingerprint=provider_output_fingerprint,
        review_fingerprint=review_fingerprint,
        provider_invoked=provider_invoked,
        issues=tuple(_dedupe_issues(issues)),
    )


def _issue(
    code: PopulationSemanticReviewerIssueCode,
    message: str,
    *,
    goal_id: str = "",
    path: str = "",
    details: Mapping[str, Any] | None = None,
) -> PopulationSemanticReviewerIssue:
    return PopulationSemanticReviewerIssue(
        code=code,
        message=message,
        goal_id=goal_id,
        path=path,
        details=dict(details or {}),
    )


def _dedupe_issues(
    issues: Sequence[PopulationSemanticReviewerIssue],
) -> list[PopulationSemanticReviewerIssue]:
    retained: list[PopulationSemanticReviewerIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        identity = (str(issue.code), issue.goal_id, issue.path)
        if identity in seen:
            continue
        seen.add(identity)
        retained.append(issue)
    return retained


def _bounded_error(error: Exception) -> str:
    return ("%s: %s" % (type(error).__name__, str(error)))[:500]
