from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import OriginalQuestionGoalContract


ExplorationCapability = Literal[
    "DESCRIBE_DISTRIBUTION",
    "COMPARE_GROUPS",
    "TRACE_TIME_CHANGE",
    "FIND_OUTLIER_CANDIDATES",
    "MEASURE_ASSOCIATION",
    "CHECK_ROBUSTNESS",
    "ASSESS_DATA_QUALITY",
    "REQUEST_EXISTING_ARTIFACT",
]

EvidenceShape = Literal[
    "SUMMARY_STATISTICS",
    "COMPARISON_RESULT",
    "TIME_SERIES",
    "CANDIDATE_SET",
    "ASSOCIATION_RESULT",
    "ROBUSTNESS_RESULT",
    "QUALITY_REPORT",
    "EXISTING_ARTIFACT",
]


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_text(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


def _require_unique(values: Sequence[str], field_name: str) -> None:
    normalized = [str(value or "").strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicates" % field_name)


def _stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class GroundedExplorationAssignment(_StrictFrozenModel):
    """Root-issued authority boundary for one advisory exploration worker."""

    protocol_version: Literal["grounded_exploration.v1"] = "grounded_exploration.v1"
    assignment_id: str
    objective: str
    goal_contract_fingerprint: str
    authorized_goal_ids: tuple[str, ...]
    explicit_exploration_goal_ids: tuple[str, ...] = ()
    population_scope_fingerprint: str
    allowed_narrower_population_fingerprints: tuple[str, ...] = ()
    time_scope_fingerprint: str
    allowed_narrower_time_fingerprints: tuple[str, ...] = ()
    source_artifact_fingerprints: tuple[str, ...] = ()
    output_authority: Literal["ADVISORY"] = "ADVISORY"

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedExplorationAssignment":
        _require_text(self.assignment_id, "assignment_id")
        _require_text(self.objective, "objective")
        _require_text(self.goal_contract_fingerprint, "goal_contract_fingerprint")
        _require_text(self.population_scope_fingerprint, "population_scope_fingerprint")
        _require_text(self.time_scope_fingerprint, "time_scope_fingerprint")
        if not self.authorized_goal_ids:
            raise ValueError("authorized_goal_ids must not be empty")
        for field_name, values in (
            ("authorized_goal_ids", self.authorized_goal_ids),
            ("explicit_exploration_goal_ids", self.explicit_exploration_goal_ids),
            (
                "allowed_narrower_population_fingerprints",
                self.allowed_narrower_population_fingerprints,
            ),
            ("allowed_narrower_time_fingerprints", self.allowed_narrower_time_fingerprints),
            ("source_artifact_fingerprints", self.source_artifact_fingerprints),
        ):
            _require_unique(values, field_name)
        if not set(self.explicit_exploration_goal_ids).issubset(self.authorized_goal_ids):
            raise ValueError("explicit exploration goals must be authorized")
        if self.population_scope_fingerprint in self.allowed_narrower_population_fingerprints:
            raise ValueError("a narrower population fingerprint must differ from its parent")
        if self.time_scope_fingerprint in self.allowed_narrower_time_fingerprints:
            raise ValueError("a narrower time fingerprint must differ from its parent")
        return self


class ExplorationScopeSignature(_StrictFrozenModel):
    """A non-executable signature proving that a request inherits or narrows scope."""

    relation: Literal["INHERIT", "NARROW"]
    fingerprint: str
    parent_fingerprint: str

    @model_validator(mode="after")
    def validate_structure(self) -> "ExplorationScopeSignature":
        _require_text(self.fingerprint, "fingerprint")
        _require_text(self.parent_fingerprint, "parent_fingerprint")
        if self.relation == "INHERIT" and self.fingerprint != self.parent_fingerprint:
            raise ValueError("an inherited scope fingerprint must equal its parent")
        if self.relation == "NARROW" and self.fingerprint == self.parent_fingerprint:
            raise ValueError("a narrowed scope fingerprint must differ from its parent")
        return self


class HypothesisProposal(_StrictFrozenModel):
    """A falsifiable advisory claim. It is never a verified conclusion."""

    hypothesis_id: str
    falsifiable_statement: str
    premises: tuple[str, ...]
    expected_observations: tuple[str, ...]
    falsifying_observations: tuple[str, ...]
    goal_ids: tuple[str, ...]
    population_scope_fingerprint: str
    time_scope_fingerprint: str
    competing_explanations: tuple[str, ...]
    claim_status: Literal["HYPOTHESIS"] = "HYPOTHESIS"

    @model_validator(mode="after")
    def validate_structure(self) -> "HypothesisProposal":
        _require_text(self.hypothesis_id, "hypothesis_id")
        _require_text(self.falsifiable_statement, "falsifiable_statement")
        _require_text(self.population_scope_fingerprint, "population_scope_fingerprint")
        _require_text(self.time_scope_fingerprint, "time_scope_fingerprint")
        for field_name, values in (
            ("premises", self.premises),
            ("expected_observations", self.expected_observations),
            ("falsifying_observations", self.falsifying_observations),
            ("goal_ids", self.goal_ids),
            ("competing_explanations", self.competing_explanations),
        ):
            if not values:
                raise ValueError("%s must not be empty" % field_name)
            _require_unique(values, field_name)
        return self


class EvidenceRequest(_StrictFrozenModel):
    """Capability request with no executable table, field, formula, SQL, or ACL surface."""

    request_id: str
    capability: ExplorationCapability
    evidence_shape: EvidenceShape
    goal_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...] = ()
    population_scope: ExplorationScopeSignature
    time_scope: ExplorationScopeSignature
    source_artifact_fingerprints: tuple[str, ...] = ()
    depends_on_request_ids: tuple[str, ...] = ()
    rationale: str = ""
    authority: Literal["ADVISORY_REQUEST"] = "ADVISORY_REQUEST"

    @model_validator(mode="after")
    def validate_structure(self) -> "EvidenceRequest":
        _require_text(self.request_id, "request_id")
        if not self.goal_ids:
            raise ValueError("goal_ids must not be empty")
        for field_name, values in (
            ("goal_ids", self.goal_ids),
            ("hypothesis_ids", self.hypothesis_ids),
            ("source_artifact_fingerprints", self.source_artifact_fingerprints),
            ("depends_on_request_ids", self.depends_on_request_ids),
        ):
            _require_unique(values, field_name)
        if self.request_id in self.depends_on_request_ids:
            raise ValueError("an evidence request cannot depend on itself")
        return self


class AnalysisPlanStep(_StrictFrozenModel):
    step_id: str
    goal_ids: tuple[str, ...]
    evidence_request_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...] = ()
    depends_on_step_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "AnalysisPlanStep":
        _require_text(self.step_id, "step_id")
        if not self.goal_ids:
            raise ValueError("goal_ids must not be empty")
        if not self.evidence_request_ids:
            raise ValueError("evidence_request_ids must not be empty")
        for field_name, values in (
            ("goal_ids", self.goal_ids),
            ("evidence_request_ids", self.evidence_request_ids),
            ("hypothesis_ids", self.hypothesis_ids),
            ("depends_on_step_ids", self.depends_on_step_ids),
        ):
            _require_unique(values, field_name)
        if self.step_id in self.depends_on_step_ids:
            raise ValueError("an analysis step cannot depend on itself")
        return self


class AnalysisPlan(_StrictFrozenModel):
    plan_id: str
    steps: tuple[AnalysisPlanStep, ...]
    terminal_step_ids: tuple[str, ...]
    authority: Literal["ADVISORY"] = "ADVISORY"

    @model_validator(mode="after")
    def validate_graph(self) -> "AnalysisPlan":
        _require_text(self.plan_id, "plan_id")
        if not self.steps:
            raise ValueError("steps must not be empty")
        step_ids = [step.step_id for step in self.steps]
        _require_unique(step_ids, "step_ids")
        _require_unique(self.terminal_step_ids, "terminal_step_ids")
        known = set(step_ids)
        if not self.terminal_step_ids or not set(self.terminal_step_ids).issubset(known):
            raise ValueError("terminal steps must reference known step ids")
        dependencies = {step.step_id: set(step.depends_on_step_ids) for step in self.steps}
        if any(not refs.issubset(known) for refs in dependencies.values()):
            raise ValueError("analysis step dependency references an unknown step")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError("analysis plan dependencies must be acyclic")
            visiting.add(step_id)
            for dependency_id in dependencies[step_id]:
                visit(dependency_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            visit(step_id)
        return self


class StoppingAssessment(_StrictFrozenModel):
    decision: Literal[
        "CONTINUE",
        "STOP_SUFFICIENT",
        "STOP_BUDGET",
        "STOP_NO_PROGRESS",
        "STOP_SAFETY",
    ]
    goal_ids: tuple[str, ...]
    supported_hypothesis_ids: tuple[str, ...] = ()
    challenged_hypothesis_ids: tuple[str, ...] = ()
    unresolved_hypothesis_ids: tuple[str, ...] = ()
    outstanding_request_ids: tuple[str, ...] = ()
    rationale: str
    authority: Literal["ADVISORY"] = "ADVISORY"

    @model_validator(mode="after")
    def validate_structure(self) -> "StoppingAssessment":
        if not self.goal_ids:
            raise ValueError("goal_ids must not be empty")
        _require_text(self.rationale, "rationale")
        for field_name, values in (
            ("goal_ids", self.goal_ids),
            ("supported_hypothesis_ids", self.supported_hypothesis_ids),
            ("challenged_hypothesis_ids", self.challenged_hypothesis_ids),
            ("unresolved_hypothesis_ids", self.unresolved_hypothesis_ids),
            ("outstanding_request_ids", self.outstanding_request_ids),
        ):
            _require_unique(values, field_name)
        hypothesis_states = (
            set(self.supported_hypothesis_ids),
            set(self.challenged_hypothesis_ids),
            set(self.unresolved_hypothesis_ids),
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(hypothesis_states)
            for right in hypothesis_states[index + 1 :]
        ):
            raise ValueError("a hypothesis cannot have multiple stopping states")
        return self


class AdvisoryExplorationArtifact(_StrictFrozenModel):
    """Worker output that cannot represent verified evidence or a final answer."""

    protocol_version: Literal["grounded_exploration.v1"] = "grounded_exploration.v1"
    artifact_id: str
    assignment_id: str
    artifact_kind: Literal["ADVISORY_EXPLORATION"] = "ADVISORY_EXPLORATION"
    authority: Literal["ADVISORY"] = "ADVISORY"
    verification_status: Literal["UNVERIFIED"] = "UNVERIFIED"
    publishable_as_final: Literal[False] = False
    causal_conclusion_allowed: Literal[False] = False
    hypotheses: tuple[HypothesisProposal, ...]
    evidence_requests: tuple[EvidenceRequest, ...]
    analysis_plan: AnalysisPlan | None = None
    stopping_assessment: StoppingAssessment
    source_artifact_fingerprints: tuple[str, ...] = ()
    advisory_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "AdvisoryExplorationArtifact":
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.assignment_id, "assignment_id")
        if not self.hypotheses:
            raise ValueError("hypotheses must not be empty")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        request_ids = [item.request_id for item in self.evidence_requests]
        _require_unique(hypothesis_ids, "hypothesis_ids")
        _require_unique(request_ids, "request_ids")
        _require_unique(self.source_artifact_fingerprints, "source_artifact_fingerprints")
        _require_unique(self.advisory_notes, "advisory_notes")
        known_hypotheses = set(hypothesis_ids)
        known_requests = set(request_ids)
        for request in self.evidence_requests:
            if not set(request.hypothesis_ids).issubset(known_hypotheses):
                raise ValueError("evidence request references an unknown hypothesis")
            if not set(request.depends_on_request_ids).issubset(known_requests):
                raise ValueError("evidence request dependency is unknown")
        _validate_request_dependency_graph(self.evidence_requests)
        if self.analysis_plan is not None:
            for step in self.analysis_plan.steps:
                if not set(step.hypothesis_ids).issubset(known_hypotheses):
                    raise ValueError("analysis plan references an unknown hypothesis")
                if not set(step.evidence_request_ids).issubset(known_requests):
                    raise ValueError("analysis plan references an unknown evidence request")
        assessment_hypotheses = (
            set(self.stopping_assessment.supported_hypothesis_ids)
            | set(self.stopping_assessment.challenged_hypothesis_ids)
            | set(self.stopping_assessment.unresolved_hypothesis_ids)
        )
        if not assessment_hypotheses.issubset(known_hypotheses):
            raise ValueError("stopping assessment references an unknown hypothesis")
        if not set(self.stopping_assessment.outstanding_request_ids).issubset(known_requests):
            raise ValueError("stopping assessment references an unknown evidence request")
        return self


def _validate_request_dependency_graph(requests: Sequence[EvidenceRequest]) -> None:
    dependencies = {
        request.request_id: set(request.depends_on_request_ids) for request in requests
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(request_id: str) -> None:
        if request_id in visited:
            return
        if request_id in visiting:
            raise ValueError("evidence request dependencies must be acyclic")
        visiting.add(request_id)
        for dependency_id in dependencies[request_id]:
            visit(dependency_id)
        visiting.remove(request_id)
        visited.add(request_id)

    for request_id in dependencies:
        visit(request_id)


def evidence_request_fingerprint(request: EvidenceRequest) -> str:
    """Stable semantic identity; request ids and prose do not defeat deduplication."""

    return _stable_fingerprint(
        {
            "capability": request.capability,
            "evidenceShape": request.evidence_shape,
            "goalIds": sorted(request.goal_ids),
            "hypothesisIds": sorted(request.hypothesis_ids),
            "populationScope": request.population_scope.model_dump(by_alias=True),
            "timeScope": request.time_scope.model_dump(by_alias=True),
            "sourceArtifactFingerprints": sorted(request.source_artifact_fingerprints),
        }
    )


class ExplorationProtocolIssue(_StrictFrozenModel):
    code: str
    message: str
    path: str = ""
    blocking: Literal[True] = True


class ExplorationProtocolValidationResult(_StrictFrozenModel):
    valid: bool
    issues: tuple[ExplorationProtocolIssue, ...] = ()


def _issue(code: str, message: str, path: str = "") -> ExplorationProtocolIssue:
    return ExplorationProtocolIssue(code=code, message=message, path=path)


def _goal_kind(goal: Any) -> str:
    return str(getattr(goal, "kind", "") or "").strip().upper()


def validate_grounded_exploration_assignment(
    assignment: GroundedExplorationAssignment,
    goal_contract: OriginalQuestionGoalContract,
    *,
    expected_goal_contract_fingerprint: str,
    session_artifact_fingerprint_allowlist: Sequence[str],
) -> ExplorationProtocolValidationResult:
    issues: list[ExplorationProtocolIssue] = []
    if assignment.goal_contract_fingerprint != str(expected_goal_contract_fingerprint or "").strip():
        issues.append(
            _issue(
                "GOAL_CONTRACT_FINGERPRINT_MISMATCH",
                "The assignment does not bind the active Goal Contract.",
                "goalContractFingerprint",
            )
        )
    goals = goal_contract.goal_map()
    explicit = set(assignment.explicit_exploration_goal_ids)
    for goal_id in assignment.authorized_goal_ids:
        goal = goals.get(goal_id)
        if goal is None:
            issues.append(
                _issue(
                    "UNKNOWN_EXPLORATION_GOAL",
                    "The assignment references a Goal that is absent from the Goal Contract.",
                    "authorizedGoalIds",
                )
            )
        elif _goal_kind(goal) != "ANALYSIS" and goal_id not in explicit:
            issues.append(
                _issue(
                    "GOAL_NOT_EXPLORATION_AUTHORIZED",
                    "Only ANALYSIS Goals or explicitly authorized exploration Goals may be delegated.",
                    "authorizedGoalIds",
                )
            )
    session_artifacts = {
        str(value or "").strip()
        for value in session_artifact_fingerprint_allowlist
        if str(value or "").strip()
    }
    for fingerprint in assignment.source_artifact_fingerprints:
        if fingerprint not in session_artifacts:
            issues.append(
                _issue(
                    "ARTIFACT_NOT_IN_SESSION_ALLOWLIST",
                    "An assignment source artifact is not present in the session allowlist.",
                    "sourceArtifactFingerprints",
                )
            )
    return ExplorationProtocolValidationResult(valid=not issues, issues=tuple(issues))


def _validate_scope_signature(
    signature: ExplorationScopeSignature,
    *,
    parent_fingerprint: str,
    allowed_narrower_fingerprints: Sequence[str],
    scope_name: str,
) -> list[ExplorationProtocolIssue]:
    issues: list[ExplorationProtocolIssue] = []
    if signature.parent_fingerprint != parent_fingerprint:
        issues.append(
            _issue(
                "%s_PARENT_MISMATCH" % scope_name,
                "The requested scope is not derived from the assignment scope.",
            )
        )
        return issues
    if signature.relation == "INHERIT" and signature.fingerprint != parent_fingerprint:
        issues.append(
            _issue(
                "%s_INHERITANCE_MISMATCH" % scope_name,
                "An inherited scope must retain the assignment fingerprint.",
            )
        )
    if signature.relation == "NARROW" and signature.fingerprint not in set(
        allowed_narrower_fingerprints
    ):
        issues.append(
            _issue(
                "%s_EXPANSION_OR_UNAUTHORIZED_SCOPE" % scope_name,
                "A narrowed scope must use a Root-authorized descendant fingerprint.",
            )
        )
    return issues


def validate_advisory_exploration_artifact(
    assignment: GroundedExplorationAssignment,
    artifact: AdvisoryExplorationArtifact,
    goal_contract: OriginalQuestionGoalContract,
    *,
    expected_goal_contract_fingerprint: str,
    session_artifact_fingerprint_allowlist: Sequence[str],
) -> ExplorationProtocolValidationResult:
    issues = list(
        validate_grounded_exploration_assignment(
            assignment,
            goal_contract,
            expected_goal_contract_fingerprint=expected_goal_contract_fingerprint,
            session_artifact_fingerprint_allowlist=session_artifact_fingerprint_allowlist,
        ).issues
    )
    if artifact.assignment_id != assignment.assignment_id:
        issues.append(
            _issue(
                "ASSIGNMENT_ID_MISMATCH",
                "The advisory artifact was produced for a different assignment.",
                "assignmentId",
            )
        )
    goals = goal_contract.goal_map()
    authorized_goals = set(assignment.authorized_goal_ids)
    artifact_goal_paths: list[tuple[str, str]] = []
    for index, hypothesis in enumerate(artifact.hypotheses):
        artifact_goal_paths.extend(
            (goal_id, "hypotheses.%s.goalIds" % index) for goal_id in hypothesis.goal_ids
        )
        if hypothesis.population_scope_fingerprint not in {
            assignment.population_scope_fingerprint,
            *assignment.allowed_narrower_population_fingerprints,
        }:
            issues.append(
                _issue(
                    "POPULATION_SCOPE_EXPANSION_OR_MISMATCH",
                    "A hypothesis is outside the assignment population scope.",
                    "hypotheses.%s.populationScopeFingerprint" % index,
                )
            )
        if hypothesis.time_scope_fingerprint not in {
            assignment.time_scope_fingerprint,
            *assignment.allowed_narrower_time_fingerprints,
        }:
            issues.append(
                _issue(
                    "TIME_SCOPE_EXPANSION_OR_MISMATCH",
                    "A hypothesis is outside the assignment time scope.",
                    "hypotheses.%s.timeScopeFingerprint" % index,
                )
            )
    for index, request in enumerate(artifact.evidence_requests):
        artifact_goal_paths.extend(
            (goal_id, "evidenceRequests.%s.goalIds" % index) for goal_id in request.goal_ids
        )
        issues.extend(
            _validate_scope_signature(
                request.population_scope,
                parent_fingerprint=assignment.population_scope_fingerprint,
                allowed_narrower_fingerprints=assignment.allowed_narrower_population_fingerprints,
                scope_name="POPULATION_SCOPE",
            )
        )
        issues.extend(
            _validate_scope_signature(
                request.time_scope,
                parent_fingerprint=assignment.time_scope_fingerprint,
                allowed_narrower_fingerprints=assignment.allowed_narrower_time_fingerprints,
                scope_name="TIME_SCOPE",
            )
        )
    if artifact.analysis_plan is not None:
        for index, step in enumerate(artifact.analysis_plan.steps):
            artifact_goal_paths.extend(
                (goal_id, "analysisPlan.steps.%s.goalIds" % index)
                for goal_id in step.goal_ids
            )
    artifact_goal_paths.extend(
        (goal_id, "stoppingAssessment.goalIds")
        for goal_id in artifact.stopping_assessment.goal_ids
    )
    for goal_id, path in artifact_goal_paths:
        if goal_id not in goals:
            issues.append(
                _issue(
                    "UNKNOWN_EXPLORATION_GOAL",
                    "The advisory artifact references a Goal absent from the Goal Contract.",
                    path,
                )
            )
        elif goal_id not in authorized_goals:
            issues.append(
                _issue(
                    "GOAL_OUTSIDE_ASSIGNMENT",
                    "The advisory artifact references a Goal outside its assignment.",
                    path,
                )
            )
    session_artifacts = {
        str(value or "").strip()
        for value in session_artifact_fingerprint_allowlist
        if str(value or "").strip()
    }
    referenced_artifacts = set(assignment.source_artifact_fingerprints)
    referenced_artifacts.update(artifact.source_artifact_fingerprints)
    for request in artifact.evidence_requests:
        referenced_artifacts.update(request.source_artifact_fingerprints)
    for fingerprint in referenced_artifacts:
        if fingerprint not in session_artifacts:
            issues.append(
                _issue(
                    "ARTIFACT_NOT_IN_SESSION_ALLOWLIST",
                    "A referenced artifact fingerprint is not present in the session allowlist.",
                    "sourceArtifactFingerprints",
                )
            )
    fingerprints: dict[str, str] = {}
    for request in artifact.evidence_requests:
        fingerprint = evidence_request_fingerprint(request)
        existing_request_id = fingerprints.get(fingerprint)
        if existing_request_id is not None:
            issues.append(
                _issue(
                    "DUPLICATE_EVIDENCE_REQUEST",
                    "Evidence requests with the same stable capability fingerprint must be deduplicated.",
                    request.request_id,
                )
            )
        else:
            fingerprints[fingerprint] = request.request_id
    return ExplorationProtocolValidationResult(valid=not issues, issues=tuple(issues))


LedgerEventType = Literal[
    "ASSIGNMENT_ACCEPTED",
    "HYPOTHESIS_PROPOSED",
    "EVIDENCE_REQUESTED",
    "ANALYSIS_PLAN_PROPOSED",
    "STOPPING_ASSESSED",
    "ADVISORY_ARTIFACT_EMITTED",
]


class LedgerEvent(_StrictFrozenModel):
    event_id: str
    assignment_id: str
    sequence: int = Field(ge=1)
    event_type: LedgerEventType
    actor: Literal["ROOT_KERNEL", "EXPLORATION_WORKER"]
    authority: Literal["ADVISORY"] = "ADVISORY"
    payload_fingerprint: str
    source_artifact_fingerprints: tuple[str, ...] = ()
    previous_event_fingerprint: str = ""
    event_fingerprint: str

    @model_validator(mode="after")
    def validate_structure(self) -> "LedgerEvent":
        _require_text(self.event_id, "event_id")
        _require_text(self.assignment_id, "assignment_id")
        _require_text(self.payload_fingerprint, "payload_fingerprint")
        _require_text(self.event_fingerprint, "event_fingerprint")
        _require_unique(self.source_artifact_fingerprints, "source_artifact_fingerprints")
        if self.event_fingerprint != ledger_event_fingerprint(self):
            raise ValueError("event_fingerprint does not match the event content")
        return self


def ledger_event_fingerprint(event: LedgerEvent) -> str:
    return _stable_fingerprint(
        {
            "eventId": event.event_id,
            "assignmentId": event.assignment_id,
            "sequence": event.sequence,
            "eventType": event.event_type,
            "actor": event.actor,
            "authority": event.authority,
            "payloadFingerprint": event.payload_fingerprint,
            "sourceArtifactFingerprints": sorted(event.source_artifact_fingerprints),
            "previousEventFingerprint": event.previous_event_fingerprint,
        }
    )


def build_ledger_event(
    *,
    event_id: str,
    assignment_id: str,
    sequence: int,
    event_type: LedgerEventType,
    actor: Literal["ROOT_KERNEL", "EXPLORATION_WORKER"],
    payload_fingerprint: str,
    source_artifact_fingerprints: Sequence[str] = (),
    previous_event_fingerprint: str = "",
) -> LedgerEvent:
    payload = {
        "eventId": event_id,
        "assignmentId": assignment_id,
        "sequence": sequence,
        "eventType": event_type,
        "actor": actor,
        "authority": "ADVISORY",
        "payloadFingerprint": payload_fingerprint,
        "sourceArtifactFingerprints": sorted(source_artifact_fingerprints),
        "previousEventFingerprint": previous_event_fingerprint,
    }
    return LedgerEvent(
        event_id=event_id,
        assignment_id=assignment_id,
        sequence=sequence,
        event_type=event_type,
        actor=actor,
        payload_fingerprint=payload_fingerprint,
        source_artifact_fingerprints=tuple(source_artifact_fingerprints),
        previous_event_fingerprint=previous_event_fingerprint,
        event_fingerprint=_stable_fingerprint(payload),
    )


class GroundedExplorationLedgerState(_StrictFrozenModel):
    assignment_id: str
    revision: int = Field(ge=0)
    events: tuple[LedgerEvent, ...] = ()
    head_event_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_chain(self) -> "GroundedExplorationLedgerState":
        _require_text(self.assignment_id, "assignment_id")
        if self.revision != len(self.events):
            raise ValueError("ledger revision must equal its event count")
        previous = ""
        event_ids: set[str] = set()
        for index, event in enumerate(self.events, start=1):
            if event.assignment_id != self.assignment_id:
                raise ValueError("ledger event assignment_id mismatch")
            if event.sequence != index:
                raise ValueError("ledger event sequence is not contiguous")
            if event.previous_event_fingerprint != previous:
                raise ValueError("ledger event chain fingerprint mismatch")
            if event.event_id in event_ids:
                raise ValueError("ledger event ids must be unique")
            event_ids.add(event.event_id)
            previous = event.event_fingerprint
        if self.head_event_fingerprint != previous:
            raise ValueError("ledger head fingerprint mismatch")
        return self


def append_ledger_event(
    state: GroundedExplorationLedgerState,
    event: LedgerEvent,
    *,
    expected_revision: int,
) -> GroundedExplorationLedgerState:
    """CAS append returning a new state; existing events are never rewritten."""

    if expected_revision != state.revision:
        raise ValueError("ledger revision conflict")
    if event.assignment_id != state.assignment_id:
        raise ValueError("ledger event assignment_id mismatch")
    if event.sequence != state.revision + 1:
        raise ValueError("ledger event sequence must append at the current tail")
    if event.previous_event_fingerprint != state.head_event_fingerprint:
        raise ValueError("ledger event previous fingerprint must match the current head")
    if any(existing.event_id == event.event_id for existing in state.events):
        raise ValueError("ledger event id already exists")
    return GroundedExplorationLedgerState(
        assignment_id=state.assignment_id,
        revision=state.revision + 1,
        events=(*state.events, event),
        head_event_fingerprint=event.event_fingerprint,
    )


def advisory_exploration_artifact_fingerprint(
    artifact: AdvisoryExplorationArtifact,
) -> str:
    return _stable_fingerprint(artifact.model_dump(by_alias=True, mode="json"))
