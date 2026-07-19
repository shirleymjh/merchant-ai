from __future__ import annotations

import hashlib
import json
import uuid
from threading import RLock
from typing import Any, Callable, Literal, Protocol, Sequence

from pydantic import ConfigDict, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_exploration_protocol import (
    AdvisoryExplorationArtifact,
    GroundedExplorationAssignment,
    GroundedExplorationLedgerState,
    ExplorationProtocolIssue,
    ExplorationProtocolValidationResult,
    EvidenceRequest,
    advisory_exploration_artifact_fingerprint,
    append_ledger_event,
    build_ledger_event,
    evidence_request_fingerprint,
    validate_advisory_exploration_artifact,
    validate_grounded_exploration_assignment,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_subagent_runtime import (
    IsolatedSubagentJob,
    IsolatedSubagentRuntime,
)


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _required_text(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


def _unique_text(values: Sequence[str], field_name: str) -> None:
    normalized = [str(value or "").strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicates" % field_name)


def _fingerprint(payload: Any) -> str:
    if isinstance(payload, APIModel):
        value = payload.model_dump(by_alias=True, mode="json")
    else:
        value = payload
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class VerifiedExplorationObservation(_StrictFrozenModel):
    """Root-projected observation from an already verified artifact.

    The view deliberately has no physical data locator or executable expression.
    It gives an advisory worker context without mounting query capabilities.
    """

    observation_id: str
    statement: str
    evidence_refs: tuple[str, ...] = ()
    authority: Literal["ROOT_VERIFIED_INPUT"] = "ROOT_VERIFIED_INPUT"

    @model_validator(mode="after")
    def validate_structure(self) -> "VerifiedExplorationObservation":
        _required_text(self.observation_id, "observation_id")
        _required_text(self.statement, "statement")
        _unique_text(self.evidence_refs, "evidence_refs")
        return self


class VerifiedExplorationSourceView(_StrictFrozenModel):
    """Minimal advisory projection produced by a trusted artifact catalog."""

    artifact_id: str
    artifact_fingerprint: str
    verification_status: Literal["VERIFIED"] = "VERIFIED"
    goal_ids: tuple[str, ...]
    observations: tuple[VerifiedExplorationObservation, ...] = ()
    authority: Literal["ROOT_VERIFIED_INPUT"] = "ROOT_VERIFIED_INPUT"

    @model_validator(mode="after")
    def validate_structure(self) -> "VerifiedExplorationSourceView":
        _required_text(self.artifact_id, "artifact_id")
        _required_text(self.artifact_fingerprint, "artifact_fingerprint")
        if not self.goal_ids:
            raise ValueError("goal_ids must not be empty")
        _unique_text(self.goal_ids, "goal_ids")
        observation_ids = [item.observation_id for item in self.observations]
        _unique_text(observation_ids, "observation_ids")
        return self


class VerifiedExplorationArtifactCatalog(Protocol):
    """Trust boundary that exposes only verified, Root-projected artifacts."""

    def resolve_verified(
        self,
        artifact_ids: Sequence[str],
    ) -> Sequence[VerifiedExplorationSourceView]: ...


class InMemoryVerifiedExplorationArtifactCatalog:
    """Small reference catalog for tests and single-process composition."""

    def __init__(self, artifacts: Sequence[VerifiedExplorationSourceView] = ()) -> None:
        artifact_ids = [item.artifact_id for item in artifacts]
        _unique_text(artifact_ids, "artifact_ids")
        self._artifacts = {item.artifact_id: item for item in artifacts}

    def resolve_verified(
        self,
        artifact_ids: Sequence[str],
    ) -> Sequence[VerifiedExplorationSourceView]:
        return tuple(
            self._artifacts[artifact_id]
            for artifact_id in artifact_ids
            if artifact_id in self._artifacts
        )


class GroundedExplorationScopeAuthority(_StrictFrozenModel):
    """Root-issued scope authority; the coordinator never invents scope lineage."""

    population_scope_fingerprint: str
    allowed_narrower_population_fingerprints: tuple[str, ...] = ()
    time_scope_fingerprint: str
    allowed_narrower_time_fingerprints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedExplorationScopeAuthority":
        _required_text(
            self.population_scope_fingerprint,
            "population_scope_fingerprint",
        )
        _required_text(self.time_scope_fingerprint, "time_scope_fingerprint")
        _unique_text(
            self.allowed_narrower_population_fingerprints,
            "allowed_narrower_population_fingerprints",
        )
        _unique_text(
            self.allowed_narrower_time_fingerprints,
            "allowed_narrower_time_fingerprints",
        )
        if (
            self.population_scope_fingerprint
            in self.allowed_narrower_population_fingerprints
        ):
            raise ValueError("a narrower population scope must differ from its parent")
        if self.time_scope_fingerprint in self.allowed_narrower_time_fingerprints:
            raise ValueError("a narrower time scope must differ from its parent")
        return self


class GroundedExplorationAssignmentSpec(_StrictFrozenModel):
    assignment_id: str
    objective: str
    authorized_goal_ids: tuple[str, ...]
    explicit_exploration_goal_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...]
    scope_authority: GroundedExplorationScopeAuthority

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedExplorationAssignmentSpec":
        _required_text(self.assignment_id, "assignment_id")
        _required_text(self.objective, "objective")
        if not self.authorized_goal_ids:
            raise ValueError("authorized_goal_ids must not be empty")
        if not self.source_artifact_ids:
            raise ValueError("source_artifact_ids must not be empty")
        _unique_text(self.authorized_goal_ids, "authorized_goal_ids")
        _unique_text(
            self.explicit_exploration_goal_ids,
            "explicit_exploration_goal_ids",
        )
        _unique_text(self.source_artifact_ids, "source_artifact_ids")
        if not set(self.explicit_exploration_goal_ids).issubset(
            self.authorized_goal_ids
        ):
            raise ValueError("explicit exploration goals must be authorized")
        return self


class ExplorationGoalView(_StrictFrozenModel):
    goal_id: str
    kind: str
    label: str
    required: bool
    depends_on_goal_ids: tuple[str, ...] = ()
    population_scope: str = ""
    population_goal_ids: tuple[str, ...] = ()
    authority: Literal["ROOT_GOAL_CONTRACT"] = "ROOT_GOAL_CONTRACT"


class GroundedExplorationWorkerInvocation(_StrictFrozenModel):
    """The complete and only input surface mounted into an advisory worker."""

    assignment: GroundedExplorationAssignment
    question: str
    authorized_goals: tuple[ExplorationGoalView, ...]
    verified_sources: tuple[VerifiedExplorationSourceView, ...]
    output_contract: Literal["AdvisoryExplorationArtifact"] = (
        "AdvisoryExplorationArtifact"
    )
    query_capability_mounted: Literal[False] = False
    publication_capability_mounted: Literal[False] = False


class GroundedExplorationWorker(Protocol):
    def run(
        self,
        invocation: GroundedExplorationWorkerInvocation,
    ) -> AdvisoryExplorationArtifact: ...


class IsolatedGroundedExplorationWorker:
    """Adapter from the generic isolation harness to the advisory protocol.

    No tools, Skills, permissions, nested agents, semantic catalog, or executor
    are mounted. Output must be one strict JSON object conforming to the closed
    advisory artifact schema.
    """

    def __init__(
        self,
        runtime: IsolatedSubagentRuntime,
        *,
        parent_thread_id: str,
        model_timeout_seconds: float | None = None,
        on_progress: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.parent_thread_id = _required_text(parent_thread_id, "parent_thread_id")
        if model_timeout_seconds is not None and float(model_timeout_seconds) <= 0:
            raise ValueError("model_timeout_seconds must be positive")
        self.model_timeout_seconds = model_timeout_seconds
        self.on_progress = on_progress

    def run(
        self,
        invocation: GroundedExplorationWorkerInvocation,
    ) -> AdvisoryExplorationArtifact:
        assignment_id = invocation.assignment.assignment_id
        job = IsolatedSubagentJob(
            job_id="exploration_%s" % assignment_id,
            thread_id="%s__exploration_%s"
            % (self.parent_thread_id, assignment_id),
            system_prompt=(
                "You are an isolated advisory exploration worker. Use only the "
                "Root-issued Goal and verified observation views in the payload. "
                "Return exactly one JSON object conforming to the supplied schema. "
                "You may propose falsifiable hypotheses, an advisory analysis plan, "
                "a stopping assessment, and abstract EvidenceRequest capabilities. "
                "You have no authority to choose physical tables or fields, define "
                "metric formulas, write SQL, acquire evidence, run queries, expand "
                "scope, publish an answer, or claim verification or causality. Every "
                "request remains pending until the Root explicitly approves it."
            ),
            user_payload={
                "invocation": invocation.model_dump(by_alias=True, mode="json"),
                "resultSchema": AdvisoryExplorationArtifact.model_json_schema(
                    by_alias=True
                ),
            },
            backend=None,
            tools=[],
            skills=[],
            middleware=[],
            permissions=[],
            subagents=[],
            model_timeout_seconds=self.model_timeout_seconds,
        )
        result = self.runtime.run(job, on_progress=self.on_progress)
        try:
            payload = json.loads(result.raw_output)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_WORKER_OUTPUT_NOT_JSON",
                "The isolated worker did not return one JSON object.",
            ) from exc
        if not isinstance(payload, dict):
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_WORKER_OUTPUT_NOT_OBJECT",
                "The isolated worker output must be a JSON object.",
            )
        try:
            return AdvisoryExplorationArtifact.model_validate(payload)
        except Exception as exc:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_WORKER_OUTPUT_INVALID",
                "The isolated worker output violates the advisory artifact contract.",
            ) from exc


class PendingExplorationCapabilityRequest(_StrictFrozenModel):
    """Non-executable handoff that only the Root may translate into capabilities."""

    capability_request_id: str
    assignment_id: str
    advisory_artifact_id: str
    request_fingerprint: str
    request: EvidenceRequest
    status: Literal["PENDING_ROOT_APPROVAL"] = "PENDING_ROOT_APPROVAL"
    root_decision_required: Literal[True] = True
    executable: Literal[False] = False
    query_dispatched: Literal[False] = False

    @model_validator(mode="after")
    def validate_structure(self) -> "PendingExplorationCapabilityRequest":
        _required_text(self.assignment_id, "assignment_id")
        _required_text(self.advisory_artifact_id, "advisory_artifact_id")
        expected_fingerprint = evidence_request_fingerprint(self.request)
        if self.request_fingerprint != expected_fingerprint:
            raise ValueError("request_fingerprint does not match the evidence request")
        expected_id = "capability_request.%s" % expected_fingerprint
        if self.capability_request_id != expected_id:
            raise ValueError("capability_request_id does not match the request fingerprint")
        return self


class AcceptedAdvisoryExplorationArtifact(_StrictFrozenModel):
    artifact_id: str
    artifact_fingerprint: str
    pending_capability_request_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "AcceptedAdvisoryExplorationArtifact":
        _required_text(self.artifact_id, "artifact_id")
        _required_text(self.artifact_fingerprint, "artifact_fingerprint")
        _unique_text(
            self.pending_capability_request_ids,
            "pending_capability_request_ids",
        )
        return self


class GroundedExplorationCoordinatorState(_StrictFrozenModel):
    assignment: GroundedExplorationAssignment
    source_artifacts: tuple[VerifiedExplorationSourceView, ...]
    ledger: GroundedExplorationLedgerState
    accepted_artifacts: tuple[AcceptedAdvisoryExplorationArtifact, ...] = ()
    pending_capability_requests: tuple[PendingExplorationCapabilityRequest, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedExplorationCoordinatorState":
        assignment_id = self.assignment.assignment_id
        if self.ledger.assignment_id != assignment_id:
            raise ValueError("ledger assignment mismatch")
        _unique_text(
            [item.artifact_id for item in self.source_artifacts],
            "source_artifact_ids",
        )
        _unique_text(
            [item.artifact_id for item in self.accepted_artifacts],
            "accepted_artifact_ids",
        )
        _unique_text(
            [item.capability_request_id for item in self.pending_capability_requests],
            "capability_request_ids",
        )
        _unique_text(
            [item.request_fingerprint for item in self.pending_capability_requests],
            "pending_request_fingerprints",
        )
        if tuple(
            item.artifact_fingerprint for item in self.source_artifacts
        ) != self.assignment.source_artifact_fingerprints:
            raise ValueError("source artifact fingerprints do not match the assignment")
        if any(
            item.assignment_id != assignment_id
            for item in self.pending_capability_requests
        ):
            raise ValueError("capability request assignment mismatch")
        pending_by_id = {
            item.capability_request_id: item
            for item in self.pending_capability_requests
        }
        accepted_by_id = {
            item.artifact_id: item for item in self.accepted_artifacts
        }
        for accepted in self.accepted_artifacts:
            if any(
                request_id not in pending_by_id
                or pending_by_id[request_id].advisory_artifact_id
                != accepted.artifact_id
                for request_id in accepted.pending_capability_request_ids
            ):
                raise ValueError("accepted artifact capability request binding mismatch")
        if any(
            item.advisory_artifact_id not in accepted_by_id
            for item in self.pending_capability_requests
        ):
            raise ValueError("pending capability request references an unknown artifact")
        return self


class GroundedExplorationStateStore(Protocol):
    def create(self, state: GroundedExplorationCoordinatorState) -> bool: ...

    def load(self, assignment_id: str) -> GroundedExplorationCoordinatorState | None: ...

    def compare_and_swap(
        self,
        assignment_id: str,
        *,
        expected_revision: int,
        replacement: GroundedExplorationCoordinatorState,
    ) -> bool: ...


class InMemoryGroundedExplorationStateStore:
    """Reference CAS store. A durable adapter can implement the same three calls."""

    def __init__(self) -> None:
        self._states: dict[str, GroundedExplorationCoordinatorState] = {}
        self._lock = RLock()

    def create(self, state: GroundedExplorationCoordinatorState) -> bool:
        assignment_id = state.assignment.assignment_id
        with self._lock:
            if assignment_id in self._states:
                return False
            self._states[assignment_id] = state
            return True

    def load(self, assignment_id: str) -> GroundedExplorationCoordinatorState | None:
        with self._lock:
            return self._states.get(assignment_id)

    def compare_and_swap(
        self,
        assignment_id: str,
        *,
        expected_revision: int,
        replacement: GroundedExplorationCoordinatorState,
    ) -> bool:
        with self._lock:
            current = self._states.get(assignment_id)
            if current is None or current.ledger.revision != expected_revision:
                return False
            if replacement.assignment.assignment_id != assignment_id:
                return False
            self._states[assignment_id] = replacement
            return True


class GroundedExplorationAssignmentReceipt(_StrictFrozenModel):
    assignment: GroundedExplorationAssignment
    ledger_revision: int
    source_artifact_ids: tuple[str, ...]
    source_artifact_fingerprints: tuple[str, ...]
    next_action: Literal["RUN_ISOLATED_ADVISORY_WORKER"] = (
        "RUN_ISOLATED_ADVISORY_WORKER"
    )


class GroundedExplorationRunReport(_StrictFrozenModel):
    status: Literal["ADVISORY_ACCEPTED", "IDEMPOTENT_REPLAY"]
    assignment_id: str
    ledger_revision: int
    artifact: AdvisoryExplorationArtifact
    artifact_fingerprint: str
    pending_capability_requests: tuple[PendingExplorationCapabilityRequest, ...]
    authority: Literal["ADVISORY"] = "ADVISORY"
    publishable_as_final: Literal[False] = False
    query_executed: Literal[False] = False
    next_action: Literal["ROOT_REVIEW_CAPABILITY_REQUESTS"] = (
        "ROOT_REVIEW_CAPABILITY_REQUESTS"
    )


class GroundedExplorationCoordinatorError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: Sequence[ExplorationProtocolIssue] = (),
    ) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message
        self.issues = tuple(issues)


class GroundedExplorationCoordinator:
    """Root-side coordinator for advisory, evidence-grounded exploration."""

    def __init__(
        self,
        *,
        artifact_catalog: VerifiedExplorationArtifactCatalog,
        state_store: GroundedExplorationStateStore,
        worker: GroundedExplorationWorker,
        identifier_factory: Callable[[str], str] | None = None,
    ) -> None:
        self.artifact_catalog = artifact_catalog
        self.state_store = state_store
        self.worker = worker
        self.identifier_factory = identifier_factory or self._default_identifier

    def issue_assignment(
        self,
        goal_contract: OriginalQuestionGoalContract,
        spec: GroundedExplorationAssignmentSpec,
    ) -> GroundedExplorationAssignmentReceipt:
        sources = self._resolve_sources(spec.source_artifact_ids, goal_contract)
        goal_contract_fingerprint = original_question_goal_contract_fingerprint(
            goal_contract
        )
        assignment = GroundedExplorationAssignment(
            assignment_id=spec.assignment_id,
            objective=spec.objective,
            goal_contract_fingerprint=goal_contract_fingerprint,
            authorized_goal_ids=spec.authorized_goal_ids,
            explicit_exploration_goal_ids=spec.explicit_exploration_goal_ids,
            population_scope_fingerprint=(
                spec.scope_authority.population_scope_fingerprint
            ),
            allowed_narrower_population_fingerprints=(
                spec.scope_authority.allowed_narrower_population_fingerprints
            ),
            time_scope_fingerprint=spec.scope_authority.time_scope_fingerprint,
            allowed_narrower_time_fingerprints=(
                spec.scope_authority.allowed_narrower_time_fingerprints
            ),
            source_artifact_fingerprints=tuple(
                item.artifact_fingerprint for item in sources
            ),
        )
        validation = validate_grounded_exploration_assignment(
            assignment,
            goal_contract,
            expected_goal_contract_fingerprint=goal_contract_fingerprint,
            session_artifact_fingerprint_allowlist=tuple(
                item.artifact_fingerprint for item in sources
            ),
        )
        self._require_valid(validation, "EXPLORATION_ASSIGNMENT_REJECTED")
        accepted_event = build_ledger_event(
            event_id=self.identifier_factory("assignment_event"),
            assignment_id=assignment.assignment_id,
            sequence=1,
            event_type="ASSIGNMENT_ACCEPTED",
            actor="ROOT_KERNEL",
            payload_fingerprint=_fingerprint(assignment),
            source_artifact_fingerprints=assignment.source_artifact_fingerprints,
        )
        ledger = append_ledger_event(
            GroundedExplorationLedgerState(
                assignment_id=assignment.assignment_id,
                revision=0,
            ),
            accepted_event,
            expected_revision=0,
        )
        state = GroundedExplorationCoordinatorState(
            assignment=assignment,
            source_artifacts=sources,
            ledger=ledger,
        )
        if not self.state_store.create(state):
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_ASSIGNMENT_ALREADY_EXISTS",
                "An exploration assignment with this id already exists.",
            )
        return GroundedExplorationAssignmentReceipt(
            assignment=assignment,
            ledger_revision=ledger.revision,
            source_artifact_ids=tuple(item.artifact_id for item in sources),
            source_artifact_fingerprints=assignment.source_artifact_fingerprints,
        )

    def run_assignment(
        self,
        assignment_id: str,
        goal_contract: OriginalQuestionGoalContract,
        *,
        expected_revision: int,
    ) -> GroundedExplorationRunReport:
        state = self._required_state(assignment_id)
        if state.ledger.revision != expected_revision:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_LEDGER_REVISION_CONFLICT",
                "The expected revision does not match the exploration ledger.",
            )
        active_fingerprint = original_question_goal_contract_fingerprint(goal_contract)
        validation = validate_grounded_exploration_assignment(
            state.assignment,
            goal_contract,
            expected_goal_contract_fingerprint=active_fingerprint,
            session_artifact_fingerprint_allowlist=tuple(
                item.artifact_fingerprint for item in state.source_artifacts
            ),
        )
        self._require_valid(validation, "EXPLORATION_ASSIGNMENT_STALE")
        current_sources = self._resolve_sources(
            tuple(item.artifact_id for item in state.source_artifacts),
            goal_contract,
        )
        if current_sources != state.source_artifacts:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_SOURCE_ARTIFACT_CHANGED",
                "A bound verified source artifact changed after assignment issuance.",
            )
        invocation = self._worker_invocation(
            state.assignment,
            goal_contract,
            current_sources,
        )
        try:
            artifact = self.worker.run(invocation)
        except GroundedExplorationCoordinatorError:
            raise
        except Exception as exc:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_WORKER_FAILED",
                "The isolated advisory exploration worker failed.",
            ) from exc
        if not isinstance(artifact, AdvisoryExplorationArtifact):
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_WORKER_OUTPUT_TYPE_REJECTED",
                "The worker may return only AdvisoryExplorationArtifact.",
            )
        artifact_validation = validate_advisory_exploration_artifact(
            state.assignment,
            artifact,
            goal_contract,
            expected_goal_contract_fingerprint=active_fingerprint,
            session_artifact_fingerprint_allowlist=tuple(
                item.artifact_fingerprint for item in current_sources
            ),
        )
        self._require_valid(
            artifact_validation,
            "ADVISORY_EXPLORATION_ARTIFACT_REJECTED",
        )
        if not artifact.source_artifact_fingerprints:
            raise GroundedExplorationCoordinatorError(
                "ADVISORY_ARTIFACT_SOURCE_BINDING_REQUIRED",
                "The advisory artifact must bind at least one verified source artifact.",
            )
        artifact_fingerprint = advisory_exploration_artifact_fingerprint(artifact)
        prior = next(
            (
                item
                for item in state.accepted_artifacts
                if item.artifact_id == artifact.artifact_id
            ),
            None,
        )
        if prior is not None:
            if prior.artifact_fingerprint != artifact_fingerprint:
                raise GroundedExplorationCoordinatorError(
                    "ADVISORY_ARTIFACT_ID_COLLISION",
                    "An accepted artifact id was reused with different content.",
                )
            pending_by_id = {
                item.capability_request_id: item
                for item in state.pending_capability_requests
            }
            return GroundedExplorationRunReport(
                status="IDEMPOTENT_REPLAY",
                assignment_id=assignment_id,
                ledger_revision=state.ledger.revision,
                artifact=artifact,
                artifact_fingerprint=artifact_fingerprint,
                pending_capability_requests=tuple(
                    pending_by_id[request_id]
                    for request_id in prior.pending_capability_request_ids
                ),
            )
        pending = self._pending_requests(state, artifact)
        ledger = self._append_artifact_events(state.ledger, artifact)
        accepted = AcceptedAdvisoryExplorationArtifact(
            artifact_id=artifact.artifact_id,
            artifact_fingerprint=artifact_fingerprint,
            pending_capability_request_ids=tuple(
                item.capability_request_id for item in pending
            ),
        )
        replacement = GroundedExplorationCoordinatorState(
            assignment=state.assignment,
            source_artifacts=state.source_artifacts,
            ledger=ledger,
            accepted_artifacts=(*state.accepted_artifacts, accepted),
            pending_capability_requests=(
                *state.pending_capability_requests,
                *pending,
            ),
        )
        if not self.state_store.compare_and_swap(
            assignment_id,
            expected_revision=expected_revision,
            replacement=replacement,
        ):
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_LEDGER_REVISION_CONFLICT",
                "The exploration ledger changed before the advisory artifact could append.",
            )
        return GroundedExplorationRunReport(
            status="ADVISORY_ACCEPTED",
            assignment_id=assignment_id,
            ledger_revision=ledger.revision,
            artifact=artifact,
            artifact_fingerprint=artifact_fingerprint,
            pending_capability_requests=pending,
        )

    def pending_capability_requests(
        self,
        assignment_id: str,
    ) -> tuple[PendingExplorationCapabilityRequest, ...]:
        """Root tool point: review only; this service intentionally cannot execute."""

        return self._required_state(assignment_id).pending_capability_requests

    def state(self, assignment_id: str) -> GroundedExplorationCoordinatorState:
        """Root persistence/tool point for audit and durable adapter integration."""

        return self._required_state(assignment_id)

    def _resolve_sources(
        self,
        artifact_ids: Sequence[str],
        goal_contract: OriginalQuestionGoalContract,
    ) -> tuple[VerifiedExplorationSourceView, ...]:
        requested = tuple(str(item or "").strip() for item in artifact_ids)
        try:
            raw_sources = self.artifact_catalog.resolve_verified(requested)
            parsed = tuple(
                item
                if isinstance(item, VerifiedExplorationSourceView)
                else VerifiedExplorationSourceView.model_validate(item)
                for item in raw_sources
            )
        except Exception as exc:
            raise GroundedExplorationCoordinatorError(
                "VERIFIED_EXPLORATION_SOURCE_UNAVAILABLE",
                "The trusted artifact catalog could not resolve verified source views.",
            ) from exc
        indexed: dict[str, VerifiedExplorationSourceView] = {}
        for item in parsed:
            if item.artifact_id in indexed:
                raise GroundedExplorationCoordinatorError(
                    "VERIFIED_EXPLORATION_SOURCE_AMBIGUOUS",
                    "The trusted artifact catalog returned a duplicate artifact id.",
                )
            indexed[item.artifact_id] = item
        if set(indexed) != set(requested):
            raise GroundedExplorationCoordinatorError(
                "VERIFIED_EXPLORATION_SOURCE_INCOMPLETE",
                "Every requested source must resolve through the verified artifact catalog.",
            )
        known_goal_ids = set(goal_contract.goal_map())
        ordered = tuple(indexed[artifact_id] for artifact_id in requested)
        if any(
            goal_id not in known_goal_ids
            for item in ordered
            for goal_id in item.goal_ids
        ):
            raise GroundedExplorationCoordinatorError(
                "VERIFIED_EXPLORATION_SOURCE_GOAL_MISMATCH",
                "A verified source references a Goal outside the active Goal Contract.",
            )
        return ordered

    @staticmethod
    def _worker_invocation(
        assignment: GroundedExplorationAssignment,
        goal_contract: OriginalQuestionGoalContract,
        sources: tuple[VerifiedExplorationSourceView, ...],
    ) -> GroundedExplorationWorkerInvocation:
        goals = goal_contract.goal_map()
        authorized_goals = tuple(
            ExplorationGoalView(
                goal_id=goal_id,
                kind=str(getattr(goals[goal_id], "kind", "") or ""),
                label=str(getattr(goals[goal_id], "label", "") or ""),
                required=bool(getattr(goals[goal_id], "required", True)),
                depends_on_goal_ids=tuple(
                    getattr(goals[goal_id], "depends_on_goal_ids", ()) or ()
                ),
                population_scope=str(
                    getattr(goals[goal_id], "population_scope", "") or ""
                ),
                population_goal_ids=tuple(
                    getattr(goals[goal_id], "population_goal_ids", ()) or ()
                ),
            )
            for goal_id in assignment.authorized_goal_ids
        )
        return GroundedExplorationWorkerInvocation(
            assignment=assignment,
            question=goal_contract.question,
            authorized_goals=authorized_goals,
            verified_sources=sources,
        )

    def _pending_requests(
        self,
        state: GroundedExplorationCoordinatorState,
        artifact: AdvisoryExplorationArtifact,
    ) -> tuple[PendingExplorationCapabilityRequest, ...]:
        existing = {
            item.request_fingerprint for item in state.pending_capability_requests
        }
        pending: list[PendingExplorationCapabilityRequest] = []
        for request in artifact.evidence_requests:
            request_fingerprint = evidence_request_fingerprint(request)
            if request_fingerprint in existing:
                raise GroundedExplorationCoordinatorError(
                    "DUPLICATE_EVIDENCE_REQUEST_ACROSS_LEDGER",
                    "An equivalent evidence capability request is already pending.",
                )
            existing.add(request_fingerprint)
            pending.append(
                PendingExplorationCapabilityRequest(
                    capability_request_id="capability_request.%s"
                    % request_fingerprint,
                    assignment_id=state.assignment.assignment_id,
                    advisory_artifact_id=artifact.artifact_id,
                    request_fingerprint=request_fingerprint,
                    request=request,
                )
            )
        return tuple(pending)

    def _append_artifact_events(
        self,
        initial: GroundedExplorationLedgerState,
        artifact: AdvisoryExplorationArtifact,
    ) -> GroundedExplorationLedgerState:
        ledger = initial
        payloads: list[tuple[str, Any]] = []
        payloads.extend(("HYPOTHESIS_PROPOSED", item) for item in artifact.hypotheses)
        payloads.extend(("EVIDENCE_REQUESTED", item) for item in artifact.evidence_requests)
        if artifact.analysis_plan is not None:
            payloads.append(("ANALYSIS_PLAN_PROPOSED", artifact.analysis_plan))
        payloads.append(("STOPPING_ASSESSED", artifact.stopping_assessment))
        payloads.append(("ADVISORY_ARTIFACT_EMITTED", artifact))
        for event_type, payload in payloads:
            event = build_ledger_event(
                event_id=self.identifier_factory("exploration_event"),
                assignment_id=artifact.assignment_id,
                sequence=ledger.revision + 1,
                event_type=event_type,  # type: ignore[arg-type]
                actor="EXPLORATION_WORKER",
                payload_fingerprint=_fingerprint(payload),
                source_artifact_fingerprints=(
                    artifact.source_artifact_fingerprints
                    if event_type == "ADVISORY_ARTIFACT_EMITTED"
                    else ()
                ),
                previous_event_fingerprint=ledger.head_event_fingerprint,
            )
            ledger = append_ledger_event(
                ledger,
                event,
                expected_revision=ledger.revision,
            )
        return ledger

    def _required_state(
        self,
        assignment_id: str,
    ) -> GroundedExplorationCoordinatorState:
        normalized = _required_text(assignment_id, "assignment_id")
        state = self.state_store.load(normalized)
        if state is None:
            raise GroundedExplorationCoordinatorError(
                "EXPLORATION_ASSIGNMENT_NOT_FOUND",
                "The requested exploration assignment does not exist.",
            )
        return state

    @staticmethod
    def _require_valid(
        result: ExplorationProtocolValidationResult,
        code: str,
    ) -> None:
        if result.valid:
            return
        raise GroundedExplorationCoordinatorError(
            code,
            "The grounded exploration protocol validation failed.",
            issues=result.issues,
        )

    @staticmethod
    def _default_identifier(prefix: str) -> str:
        return "%s.%s" % (prefix, uuid.uuid4().hex)
