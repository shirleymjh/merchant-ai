from __future__ import annotations

import pytest
from pydantic import ValidationError

from merchant_ai.services.grounded_exploration_protocol import (
    AdvisoryExplorationArtifact,
    AnalysisPlan,
    AnalysisPlanStep,
    EvidenceRequest,
    ExplorationScopeSignature,
    GroundedExplorationAssignment,
    GroundedExplorationLedgerState,
    HypothesisProposal,
    StoppingAssessment,
    append_ledger_event,
    build_ledger_event,
    evidence_request_fingerprint,
    validate_advisory_exploration_artifact,
    validate_grounded_exploration_assignment,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
)


def goal_contract() -> OriginalQuestionGoalContract:
    return OriginalQuestionGoalContract(
        question="Explore a reported change and retain the requested measure.",
        goals=[
            AnalysisQuestionGoal(
                goal_id="analysis.primary",
                label="Explore the reported change",
                analysis_type="OPEN_EXPLORATION",
                input_goal_ids=["measure.requested"],
            ),
            MetricQuestionGoal(
                goal_id="measure.requested",
                label="Retain the requested measure",
            ),
        ],
    )


def assignment(
    *,
    authorized_goal_ids: tuple[str, ...] = ("analysis.primary",),
    explicit_goal_ids: tuple[str, ...] = (),
    source_artifacts: tuple[str, ...] = ("artifact.allowed",),
) -> GroundedExplorationAssignment:
    contract = goal_contract()
    return GroundedExplorationAssignment(
        assignment_id="assignment.one",
        objective="Explore competing explanations for the verified change.",
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
        authorized_goal_ids=authorized_goal_ids,
        explicit_exploration_goal_ids=explicit_goal_ids,
        population_scope_fingerprint="population.base",
        allowed_narrower_population_fingerprints=("population.narrow",),
        time_scope_fingerprint="time.base",
        allowed_narrower_time_fingerprints=("time.narrow",),
        source_artifact_fingerprints=source_artifacts,
    )


def hypothesis(
    *,
    population_fingerprint: str = "population.base",
    time_fingerprint: str = "time.base",
) -> HypothesisProposal:
    return HypothesisProposal(
        hypothesis_id="hypothesis.one",
        falsifiable_statement="A measurable change is concentrated in one comparison segment.",
        premises=("The source artifact is comparable across the requested scope.",),
        expected_observations=("The segment comparison differs from its reference.",),
        falsifying_observations=("The segment comparison remains within its reference.",),
        goal_ids=("analysis.primary",),
        population_scope_fingerprint=population_fingerprint,
        time_scope_fingerprint=time_fingerprint,
        competing_explanations=("The apparent change is distributed across segments.",),
    )


def scope(
    kind: str,
    *,
    relation: str = "INHERIT",
    fingerprint: str | None = None,
    parent_fingerprint: str | None = None,
) -> ExplorationScopeSignature:
    base = "%s.base" % kind
    return ExplorationScopeSignature(
        relation=relation,
        fingerprint=fingerprint or base,
        parent_fingerprint=parent_fingerprint or base,
    )


def request(
    *,
    request_id: str = "request.one",
    population_scope: ExplorationScopeSignature | None = None,
    time_scope: ExplorationScopeSignature | None = None,
    source_artifacts: tuple[str, ...] = ("artifact.allowed",),
    rationale: str = "Test the falsifiable expectation.",
) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=request_id,
        capability="COMPARE_GROUPS",
        evidence_shape="COMPARISON_RESULT",
        goal_ids=("analysis.primary",),
        hypothesis_ids=("hypothesis.one",),
        population_scope=population_scope or scope("population"),
        time_scope=time_scope or scope("time"),
        source_artifact_fingerprints=source_artifacts,
        rationale=rationale,
    )


def artifact(
    *,
    requests: tuple[EvidenceRequest, ...] | None = None,
    hypothesis_item: HypothesisProposal | None = None,
    source_artifacts: tuple[str, ...] = ("artifact.allowed",),
) -> AdvisoryExplorationArtifact:
    evidence_requests = requests if requests is not None else (request(),)
    return AdvisoryExplorationArtifact(
        artifact_id="advisory.one",
        assignment_id="assignment.one",
        hypotheses=(hypothesis_item or hypothesis(),),
        evidence_requests=evidence_requests,
        analysis_plan=AnalysisPlan(
            plan_id="plan.one",
            steps=(
                AnalysisPlanStep(
                    step_id="step.one",
                    goal_ids=("analysis.primary",),
                    evidence_request_ids=tuple(item.request_id for item in evidence_requests),
                    hypothesis_ids=("hypothesis.one",),
                ),
            ),
            terminal_step_ids=("step.one",),
        ),
        stopping_assessment=StoppingAssessment(
            decision="CONTINUE",
            goal_ids=("analysis.primary",),
            unresolved_hypothesis_ids=("hypothesis.one",),
            outstanding_request_ids=tuple(item.request_id for item in evidence_requests),
            rationale="Requested evidence has not been returned.",
        ),
        source_artifact_fingerprints=source_artifacts,
        advisory_notes=("Root must authorize any evidence acquisition.",),
    )


def validation_codes(result: object) -> set[str]:
    return {issue.code for issue in result.issues}


def test_assignment_allows_analysis_or_explicitly_authorized_exploration_goals() -> None:
    contract = goal_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)

    analysis_only = validate_grounded_exploration_assignment(
        assignment(),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )
    explicit_non_analysis = validate_grounded_exploration_assignment(
        assignment(
            authorized_goal_ids=("analysis.primary", "measure.requested"),
            explicit_goal_ids=("measure.requested",),
        ),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )
    unapproved_non_analysis = validate_grounded_exploration_assignment(
        assignment(authorized_goal_ids=("measure.requested",)),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )

    assert analysis_only.valid is True
    assert explicit_non_analysis.valid is True
    assert unapproved_non_analysis.valid is False
    assert "GOAL_NOT_EXPLORATION_AUTHORIZED" in validation_codes(unapproved_non_analysis)


def test_assignment_fails_for_unknown_goal_stale_contract_or_untrusted_artifact() -> None:
    contract = goal_contract()
    result = validate_grounded_exploration_assignment(
        assignment(
            authorized_goal_ids=("analysis.missing",),
            source_artifacts=("artifact.untrusted",),
        ),
        contract,
        expected_goal_contract_fingerprint="different.contract",
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )

    assert result.valid is False
    assert validation_codes(result) == {
        "ARTIFACT_NOT_IN_SESSION_ALLOWLIST",
        "GOAL_CONTRACT_FINGERPRINT_MISMATCH",
        "UNKNOWN_EXPLORATION_GOAL",
    }


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("tableName", "physical_source"),
        ("fieldName", "physical_attribute"),
        ("metricFormula", "executable_expression"),
        ("sql", "executable_query"),
        ("tenantCondition", "access_predicate"),
    ],
)
def test_evidence_request_has_no_executable_data_selection_surface(
    field_name: str,
    field_value: str,
) -> None:
    payload = request().model_dump(by_alias=True, mode="json")
    payload[field_name] = field_value

    with pytest.raises(ValidationError) as exc_info:
        EvidenceRequest.model_validate(payload)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_scope_signatures_allow_inheritance_or_pre_authorized_narrowing() -> None:
    contract = goal_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)
    narrow_request = request(
        population_scope=scope(
            "population",
            relation="NARROW",
            fingerprint="population.narrow",
        ),
        time_scope=scope("time"),
    )
    result = validate_advisory_exploration_artifact(
        assignment(),
        artifact(requests=(narrow_request,)),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )

    assert result.valid is True


def test_scope_validation_rejects_unapproved_expansion_and_parent_mismatch() -> None:
    contract = goal_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)
    outside_request = request(
        population_scope=scope(
            "population",
            relation="NARROW",
            fingerprint="population.outside",
        ),
        time_scope=scope(
            "time",
            relation="NARROW",
            fingerprint="time.narrow",
            parent_fingerprint="time.different-parent",
        ),
    )
    result = validate_advisory_exploration_artifact(
        assignment(),
        artifact(
            requests=(outside_request,),
            hypothesis_item=hypothesis(population_fingerprint="population.outside"),
        ),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )

    assert result.valid is False
    assert {
        "POPULATION_SCOPE_EXPANSION_OR_MISMATCH",
        "POPULATION_SCOPE_EXPANSION_OR_UNAUTHORIZED_SCOPE",
        "TIME_SCOPE_PARENT_MISMATCH",
    }.issubset(validation_codes(result))


def test_artifact_references_only_session_allowlisted_artifacts() -> None:
    contract = goal_contract()
    fingerprint = original_question_goal_contract_fingerprint(contract)
    result = validate_advisory_exploration_artifact(
        assignment(),
        artifact(
            requests=(request(source_artifacts=("artifact.untrusted",)),),
            source_artifacts=("artifact.untrusted",),
        ),
        contract,
        expected_goal_contract_fingerprint=fingerprint,
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )

    assert result.valid is False
    assert "ARTIFACT_NOT_IN_SESSION_ALLOWLIST" in validation_codes(result)


def test_evidence_request_fingerprint_is_semantic_stable_and_deduplicated() -> None:
    first = request(request_id="request.one", rationale="first explanation")
    second = request(request_id="request.two", rationale="different explanation")

    assert evidence_request_fingerprint(first) == evidence_request_fingerprint(second)

    contract = goal_contract()
    result = validate_advisory_exploration_artifact(
        assignment(),
        artifact(requests=(first, second)),
        contract,
        expected_goal_contract_fingerprint=original_question_goal_contract_fingerprint(contract),
        session_artifact_fingerprint_allowlist=("artifact.allowed",),
    )
    assert result.valid is False
    assert "DUPLICATE_EVIDENCE_REQUEST" in validation_codes(result)


def test_analysis_plan_dependencies_are_known_and_acyclic() -> None:
    with pytest.raises(ValidationError) as unknown_step_error:
        AnalysisPlan(
            plan_id="plan.unknown",
            steps=(
                AnalysisPlanStep(
                    step_id="step.one",
                    goal_ids=("analysis.primary",),
                    evidence_request_ids=("request.one",),
                    depends_on_step_ids=("step.missing",),
                ),
            ),
            terminal_step_ids=("step.one",),
        )
    assert "unknown step" in str(unknown_step_error.value)

    with pytest.raises(ValidationError) as cycle_error:
        AnalysisPlan(
            plan_id="plan.cycle",
            steps=(
                AnalysisPlanStep(
                    step_id="step.one",
                    goal_ids=("analysis.primary",),
                    evidence_request_ids=("request.one",),
                    depends_on_step_ids=("step.two",),
                ),
                AnalysisPlanStep(
                    step_id="step.two",
                    goal_ids=("analysis.primary",),
                    evidence_request_ids=("request.two",),
                    depends_on_step_ids=("step.one",),
                ),
            ),
            terminal_step_ids=("step.two",),
        )
    assert "acyclic" in str(cycle_error.value)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("authority", "VERIFIED"),
        ("verificationStatus", "VERIFIED"),
        ("publishableAsFinal", True),
        ("causalConclusionAllowed", True),
        ("finalAnswer", "impermissible publication payload"),
        ("causalConclusion", "impermissible conclusion payload"),
    ],
)
def test_worker_artifact_cannot_claim_verified_final_or_causal_authority(
    field_name: str,
    field_value: object,
) -> None:
    payload = artifact().model_dump(by_alias=True, mode="json")
    payload[field_name] = field_value

    with pytest.raises(ValidationError):
        AdvisoryExplorationArtifact.model_validate(payload)


def test_ledger_is_hash_chained_cas_guarded_and_append_only() -> None:
    original = GroundedExplorationLedgerState(
        assignment_id="assignment.one",
        revision=0,
    )
    first = build_ledger_event(
        event_id="event.one",
        assignment_id="assignment.one",
        sequence=1,
        event_type="ASSIGNMENT_ACCEPTED",
        actor="ROOT_KERNEL",
        payload_fingerprint="payload.one",
    )
    next_state = append_ledger_event(original, first, expected_revision=0)

    assert original.revision == 0
    assert original.events == ()
    assert next_state.revision == 1
    assert next_state.events == (first,)
    assert next_state.head_event_fingerprint == first.event_fingerprint

    with pytest.raises(ValueError) as revision_error:
        append_ledger_event(next_state, first, expected_revision=0)
    assert "revision conflict" in str(revision_error.value)

    wrong_parent = build_ledger_event(
        event_id="event.two",
        assignment_id="assignment.one",
        sequence=2,
        event_type="HYPOTHESIS_PROPOSED",
        actor="EXPLORATION_WORKER",
        payload_fingerprint="payload.two",
        previous_event_fingerprint="different.head",
    )
    with pytest.raises(ValueError) as head_error:
        append_ledger_event(next_state, wrong_parent, expected_revision=1)
    assert "current head" in str(head_error.value)
