from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from merchant_ai.services.grounded_population_semantic_reviewer import (
    IndependentPopulationSemanticReviewer,
    PopulationSemanticProviderDecision,
    PopulationSemanticProviderOutput,
    PopulationSemanticReviewerIssueCode,
    build_population_semantic_reviewer_request,
    goal_declaration_population_input_from_review,
    population_goal_skeleton,
    population_goal_skeleton_fingerprint,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationScopeKind,
    PopulationSemanticVerifier,
)


QUESTION = "Return detail rows, then rank those rows by a measure"
CORE_AUTHORITY = "core-declaration-authority"
REVIEWER_AUTHORITY = "independent-semantic-reviewer-authority"
DETAIL_GOAL = "detail.rows"
METRIC_GOAL = "metric.value"
RANKING_GOAL = "ranking.rows"


def _contract(
    population_scope: str = "SAME_AS_GOAL",
    *,
    label: str = "rank the detail rows",
):
    ranking = {
        "goalId": RANKING_GOAL,
        "kind": "RANKING",
        "label": label,
        "sourceSpans": ["rank those rows by a measure"],
        "metricGoalIds": [METRIC_GOAL],
        "direction": "DESC",
        "limit": 3,
        "populationScope": population_scope,
        "rationale": "Core-owned rationale must not reach the reviewer",
        "semanticRefIds": ["asset-ref-must-not-reach-reviewer"],
    }
    if population_scope in {"SAME_AS_GOAL", "VERIFIED_ENTITY_SET"}:
        ranking["populationGoalIds"] = [DETAIL_GOAL]
    return {
        "question": QUESTION,
        "goals": [
            {
                "goalId": DETAIL_GOAL,
                "kind": "DETAIL",
                "label": "detail rows",
                "sourceSpans": ["detail rows"],
                "requiredFieldRefIds": ["field-ref-must-not-reach-reviewer"],
            },
            {
                "goalId": METRIC_GOAL,
                "kind": "METRIC",
                "label": "measure",
                "sourceSpans": ["measure"],
                "metricRefId": "metric-ref-must-not-reach-reviewer",
            },
            ranking,
        ],
    }


def _decision(
    goal_id: str,
    *,
    gate_required: bool = False,
    scope_kind: PopulationScopeKind | None = None,
    source_goal_ids: tuple[str, ...] = (),
) -> PopulationSemanticProviderDecision:
    return PopulationSemanticProviderDecision(
        goal_id=goal_id,
        gate_required=gate_required,
        scope_kind=scope_kind,
        source_goal_ids=source_goal_ids,
    )


def _complete_decisions(
    *,
    scope_kind: PopulationScopeKind = PopulationScopeKind.SAME_AS_GOAL,
    source_goal_ids: tuple[str, ...] = (DETAIL_GOAL,),
) -> tuple[PopulationSemanticProviderDecision, ...]:
    return (
        _decision(DETAIL_GOAL),
        _decision(METRIC_GOAL),
        _decision(
            RANKING_GOAL,
            gate_required=True,
            scope_kind=scope_kind,
            source_goal_ids=source_goal_ids,
        ),
    )


class StructuredProvider:
    def __init__(
        self,
        *,
        authority: str = REVIEWER_AUTHORITY,
        decisions: tuple[PopulationSemanticProviderDecision, ...] | None = None,
        complete: bool = True,
        response_mutations: dict | None = None,
        return_json: bool = False,
    ) -> None:
        self._authority = authority
        self.decisions = decisions or _complete_decisions()
        self.complete = complete
        self.response_mutations = dict(response_mutations or {})
        self.return_json = return_json
        self.requests = []

    @property
    def authority_fingerprint(self) -> str:
        return self._authority

    def review_population_semantics(self, request, *, timeout_seconds):
        self.requests.append((request, timeout_seconds))
        payload = PopulationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            goal_skeleton_fingerprint=request.goal_skeleton_fingerprint,
            complete=self.complete,
            decisions=self.decisions,
        ).model_dump(by_alias=True, mode="json")
        payload.update(self.response_mutations)
        return json.dumps(payload) if self.return_json else payload


class SleepingProvider(StructuredProvider):
    def review_population_semantics(self, request, *, timeout_seconds):
        self.requests.append((request, timeout_seconds))
        time.sleep(0.05)
        return {}


class FailedProvider(StructuredProvider):
    def review_population_semantics(self, request, *, timeout_seconds):
        self.requests.append((request, timeout_seconds))
        raise RuntimeError("provider unavailable")


class MutatingProvider(StructuredProvider):
    def review_population_semantics(self, request, *, timeout_seconds):
        request.goal_skeleton[0].answer_structure["populationScope"] = (
            "provider mutation"
        )
        return super().review_population_semantics(
            request,
            timeout_seconds=timeout_seconds,
        )


def _review(provider, *, contract=None, question: str = QUESTION, timeout=0.5):
    return IndependentPopulationSemanticReviewer(
        provider,
        trusted_provider_authority_fingerprints=(REVIEWER_AUTHORITY,),
        timeout_seconds=timeout,
    ).review(
        effective_question=question,
        contract=contract or _contract(),
        declaration_author_fingerprint=CORE_AUTHORITY,
    )


def _issue_codes(outcome) -> set[str]:
    return {str(issue.code) for issue in outcome.issues}


def _all_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_all_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def test_provider_receives_only_effective_question_and_population_blind_skeleton() -> None:
    provider = StructuredProvider()

    outcome = _review(provider)

    assert outcome.passed is True
    request, timeout_seconds = provider.requests[0]
    payload = request.model_dump(by_alias=True, mode="json")
    keys = _all_keys(payload)
    forbidden_keys = {
        "populationScope",
        "populationGoalIds",
        "populationFingerprint",
        "sourceArtifactIds",
        "semanticRefIds",
        "metricRefId",
        "dimensionRefId",
        "requiredFieldRefIds",
        "relationshipRefs",
        "lineageRefs",
        "rationale",
        "sql",
        "tables",
        "artifacts",
    }
    assert keys.isdisjoint(forbidden_keys)
    assert request.effective_question == QUESTION
    assert timeout_seconds == pytest.approx(0.5)
    ranking = next(
        item for item in request.goal_skeleton if item.goal_id == RANKING_GOAL
    )
    assert ranking.answer_structure == {
        "metricGoalIds": [METRIC_GOAL],
        "direction": "DESC",
        "limit": 3,
        "limitSource": "USER_EXPLICIT",
    }


def test_different_core_population_claims_produce_the_same_reviewer_skeleton() -> None:
    dependent = population_goal_skeleton(_contract("SAME_AS_GOAL"))
    independent = population_goal_skeleton(_contract("ALL_MATCHING_ROWS"))

    assert dependent == independent
    assert population_goal_skeleton_fingerprint(dependent) == (
        population_goal_skeleton_fingerprint(independent)
    )


def test_successful_structured_review_binds_question_skeleton_request_and_authority() -> None:
    provider = StructuredProvider(return_json=True)

    outcome = _review(provider)

    assert outcome.passed is True
    assert outcome.review is not None
    assert outcome.review.complete is True
    assert outcome.review.verifier_fingerprint == REVIEWER_AUTHORITY
    assert outcome.review.question_fingerprint == outcome.request.question_fingerprint
    assert outcome.review.goal_skeleton_fingerprint == (
        outcome.request.goal_skeleton_fingerprint
    )
    assert outcome.review.provider_request_fingerprint == (
        outcome.request.request_fingerprint
    )
    assert outcome.review.provider_output_fingerprint == (
        outcome.provider_output_fingerprint
    )
    expectation = outcome.review.expectations[0]
    assert expectation.consumer_goal_id == RANKING_GOAL
    assert expectation.expected_scope.kind == "SAME_AS_GOAL"
    assert expectation.expected_scope.source_goal_ids == (DETAIL_GOAL,)
    assert expectation.expected_scope.complete_membership_required is True


def test_successful_review_adapts_to_the_goal_declaration_population_gate() -> None:
    provider = StructuredProvider()
    contract = _contract("SAME_AS_GOAL")
    outcome = _review(provider, contract=contract)

    gate_input = goal_declaration_population_input_from_review(
        contract,
        outcome,
        declaration_author_fingerprint=CORE_AUTHORITY,
        trusted_semantic_verifier_fingerprints=(REVIEWER_AUTHORITY,),
    )
    result = PopulationSemanticVerifier().verify_goal_declaration(gate_input)

    assert gate_input.goal_skeleton_fingerprint == (
        outcome.goal_skeleton_fingerprint
    )
    assert result.passed is True, [gap.model_dump() for gap in result.gaps]


def test_metric_with_explicit_time_window_needs_no_population_gate() -> None:
    question = "最近7天订单总数是多少？"
    metric_goal_id = "metric.order_count"
    time_goal_id = "time.last_7_days"
    contract = {
        "question": question,
        "goals": [
            {
                "goalId": metric_goal_id,
                "kind": "METRIC",
                "label": "订单总数",
                "sourceSpans": ["订单总数"],
            },
            {
                "goalId": time_goal_id,
                "kind": "TIME_WINDOW",
                "label": "最近7天",
                "sourceSpans": ["最近7天"],
                "timeExpression": "最近7天",
                "appliesToGoalIds": [metric_goal_id],
            },
        ],
    }
    provider = StructuredProvider(
        decisions=(
            _decision(metric_goal_id, gate_required=False),
            _decision(time_goal_id, gate_required=False),
        )
    )

    outcome = _review(provider, contract=contract, question=question)

    assert outcome.passed is True, [
        issue.model_dump() for issue in outcome.issues
    ]
    assert all(decision.gate_required is False for decision in provider.decisions)
    assert outcome.review is not None
    assert outcome.review.expectations == ()

    gate_input = goal_declaration_population_input_from_review(
        contract,
        outcome,
        declaration_author_fingerprint=CORE_AUTHORITY,
        trusted_semantic_verifier_fingerprints=(REVIEWER_AUTHORITY,),
    )
    result = PopulationSemanticVerifier().verify_goal_declaration(gate_input)

    assert gate_input.declarations == ()
    assert result.passed is True, [gap.model_dump() for gap in result.gaps]
    assert result.attestation.gate_open is True


@pytest.mark.parametrize(
    ("scope_kind", "source_goal_ids", "complete_membership"),
    [
        (PopulationScopeKind.UNIVERSE, (), False),
        (PopulationScopeKind.INDEPENDENT, (), False),
        (PopulationScopeKind.SAME_AS_GOAL, (DETAIL_GOAL,), True),
        (PopulationScopeKind.VERIFIED_ENTITY_SET, (DETAIL_GOAL,), True),
        (PopulationScopeKind.PREDICATE_SCOPE, (), False),
        (PopulationScopeKind.VERIFIED_RESULT_ARTIFACT, (), True),
    ],
)
def test_reviewer_supports_each_structural_population_kind(
    scope_kind,
    source_goal_ids,
    complete_membership,
) -> None:
    provider = StructuredProvider(
        decisions=_complete_decisions(
            scope_kind=scope_kind,
            source_goal_ids=source_goal_ids,
        )
    )

    outcome = _review(provider)

    assert outcome.passed is True, [issue.model_dump() for issue in outcome.issues]
    scope = outcome.review.expectations[0].expected_scope
    assert scope.kind == scope_kind.value
    assert scope.source_goal_ids == source_goal_ids
    assert scope.complete_membership_required is complete_membership


def test_same_provider_and_core_authority_is_rejected_without_invocation() -> None:
    provider = StructuredProvider(authority=CORE_AUTHORITY)
    reviewer = IndependentPopulationSemanticReviewer(
        provider,
        trusted_provider_authority_fingerprints=(CORE_AUTHORITY,),
        timeout_seconds=0.5,
    )

    outcome = reviewer.review(
        effective_question=QUESTION,
        contract=_contract(),
        declaration_author_fingerprint=CORE_AUTHORITY,
    )

    assert outcome.passed is False
    assert outcome.provider_invoked is False
    assert provider.requests == []
    assert _issue_codes(outcome) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_NOT_INDEPENDENT.value
    }


def test_untrusted_provider_authority_is_rejected_without_invocation() -> None:
    provider = StructuredProvider(authority="untrusted-reviewer")

    outcome = _review(provider)

    assert outcome.passed is False
    assert outcome.provider_invoked is False
    assert provider.requests == []
    assert _issue_codes(outcome) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_AUTHORITY_UNTRUSTED.value
    }


def test_provider_timeout_fails_closed() -> None:
    outcome = _review(SleepingProvider(), timeout=0.01)

    assert outcome.passed is False
    assert outcome.review is not None
    assert outcome.review.complete is False
    assert _issue_codes(outcome) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_TIMEOUT.value
    }


def test_provider_exception_and_malformed_output_fail_closed() -> None:
    failed = _review(FailedProvider())
    malformed = _review(
        StructuredProvider(response_mutations={"unexpectedSql": "select secret"})
    )

    assert _issue_codes(failed) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_FAILED.value
    }
    assert _issue_codes(malformed) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_OUTPUT_INVALID.value
    }


def test_provider_cannot_mutate_the_retained_population_blind_request() -> None:
    outcome = _review(MutatingProvider())

    assert outcome.passed is False
    assert _issue_codes(outcome) == {
        PopulationSemanticReviewerIssueCode.PROVIDER_REQUEST_MUTATED.value
    }
    assert "populationScope" not in _all_keys(
        outcome.request.model_dump(by_alias=True, mode="json")
    )


@pytest.mark.parametrize(
    ("mutations", "expected_code"),
    [
        (
            {"requestFingerprint": "stale-request"},
            PopulationSemanticReviewerIssueCode.PROVIDER_REQUEST_BINDING_MISMATCH,
        ),
        (
            {"questionFingerprint": "stale-question"},
            PopulationSemanticReviewerIssueCode.PROVIDER_QUESTION_BINDING_MISMATCH,
        ),
        (
            {"goalSkeletonFingerprint": "stale-skeleton"},
            PopulationSemanticReviewerIssueCode.PROVIDER_SKELETON_BINDING_MISMATCH,
        ),
    ],
)
def test_stale_provider_bindings_fail_closed(mutations, expected_code) -> None:
    outcome = _review(StructuredProvider(response_mutations=mutations))

    assert outcome.passed is False
    assert expected_code.value in _issue_codes(outcome)


def test_missing_duplicate_and_unknown_goal_decisions_fail_closed() -> None:
    missing = _review(
        StructuredProvider(
            decisions=tuple(
                item
                for item in _complete_decisions()
                if item.goal_id != METRIC_GOAL
            )
        )
    )
    duplicate = _review(
        StructuredProvider(
            decisions=(*_complete_decisions(), _decision(DETAIL_GOAL))
        )
    )
    unknown = _review(
        StructuredProvider(
            decisions=(*_complete_decisions(), _decision("unknown.goal"))
        )
    )

    assert PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_MISSING.value in (
        _issue_codes(missing)
    )
    assert PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_DUPLICATE.value in (
        _issue_codes(duplicate)
    )
    assert PopulationSemanticReviewerIssueCode.PROVIDER_GOAL_UNKNOWN.value in (
        _issue_codes(unknown)
    )


@pytest.mark.parametrize(
    ("decision", "expected_code"),
    [
        (
            _decision(RANKING_GOAL, gate_required=True),
            PopulationSemanticReviewerIssueCode.PROVIDER_SCOPE_REQUIRED,
        ),
        (
            _decision(
                RANKING_GOAL,
                scope_kind=PopulationScopeKind.SAME_AS_GOAL,
                source_goal_ids=(DETAIL_GOAL,),
            ),
            PopulationSemanticReviewerIssueCode.PROVIDER_SCOPE_UNEXPECTED,
        ),
        (
            _decision(
                RANKING_GOAL,
                gate_required=True,
                scope_kind=PopulationScopeKind.SAME_AS_GOAL,
            ),
            PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_REQUIRED,
        ),
        (
            _decision(
                RANKING_GOAL,
                gate_required=True,
                scope_kind=PopulationScopeKind.INDEPENDENT,
                source_goal_ids=(DETAIL_GOAL,),
            ),
            PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_UNEXPECTED,
        ),
        (
            _decision(
                RANKING_GOAL,
                gate_required=True,
                scope_kind=PopulationScopeKind.SAME_AS_GOAL,
                source_goal_ids=("missing.goal",),
            ),
            PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_UNKNOWN,
        ),
        (
            _decision(
                RANKING_GOAL,
                gate_required=True,
                scope_kind=PopulationScopeKind.SAME_AS_GOAL,
                source_goal_ids=(RANKING_GOAL,),
            ),
            PopulationSemanticReviewerIssueCode.PROVIDER_SOURCE_GOAL_SELF_REFERENCE,
        ),
    ],
)
def test_invalid_scope_structures_fail_closed(decision, expected_code) -> None:
    outcome = _review(
        StructuredProvider(
            decisions=(
                _decision(DETAIL_GOAL),
                _decision(METRIC_GOAL),
                decision,
            )
        )
    )

    assert outcome.passed is False
    assert expected_code.value in _issue_codes(outcome)


def test_effective_question_mismatch_fails_before_provider_invocation() -> None:
    provider = StructuredProvider()

    outcome = _review(provider, question="different effective question")

    assert outcome.passed is False
    assert outcome.provider_invoked is False
    assert provider.requests == []
    assert _issue_codes(outcome) == {
        PopulationSemanticReviewerIssueCode.EFFECTIVE_QUESTION_MISMATCH.value
    }


def test_review_cannot_be_reused_after_nonpopulation_goal_skeleton_changes() -> None:
    original = _contract()
    outcome = _review(StructuredProvider(), contract=original)
    changed = _contract(label="different answer target")

    with pytest.raises(ValueError) as exc_info:
        goal_declaration_population_input_from_review(
            changed,
            outcome,
            declaration_author_fingerprint=CORE_AUTHORITY,
            trusted_semantic_verifier_fingerprints=(REVIEWER_AUTHORITY,),
        )
    assert "does not match" in str(exc_info.value)


def test_request_fingerprint_is_deterministic_and_bound_to_exact_skeleton() -> None:
    left = build_population_semantic_reviewer_request(QUESTION, _contract())
    right = build_population_semantic_reviewer_request(QUESTION, _contract())
    changed = build_population_semantic_reviewer_request(
        QUESTION,
        _contract(label="different answer target"),
    )

    assert left.request_fingerprint == right.request_fingerprint
    assert left.request_fingerprint != changed.request_fingerprint
    assert left.goal_skeleton_fingerprint != changed.goal_skeleton_fingerprint


def test_semantic_reviewer_source_has_no_regular_expression_dependency() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "merchant_ai/services/grounded_population_semantic_reviewer.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    blocked_modules = {"re", "regex"}
    imported_modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    blocked_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in blocked_modules
    ]

    assert imported_modules.isdisjoint(blocked_modules)
    assert blocked_calls == []
