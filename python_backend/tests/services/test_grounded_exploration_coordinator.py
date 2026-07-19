from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from merchant_ai.services.grounded_exploration_coordinator import (
    GroundedExplorationAssignmentSpec,
    GroundedExplorationCoordinator,
    GroundedExplorationCoordinatorError,
    GroundedExplorationScopeAuthority,
    InMemoryGroundedExplorationStateStore,
    InMemoryVerifiedExplorationArtifactCatalog,
    IsolatedGroundedExplorationWorker,
    PendingExplorationCapabilityRequest,
    VerifiedExplorationObservation,
    VerifiedExplorationSourceView,
)
from merchant_ai.services.grounded_exploration_protocol import (
    AdvisoryExplorationArtifact,
    AnalysisPlan,
    AnalysisPlanStep,
    EvidenceRequest,
    ExplorationScopeSignature,
    HypothesisProposal,
    StoppingAssessment,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_subagent_runtime import (
    IsolatedSubagentJob,
    IsolatedSubagentResult,
    IsolatedSubagentRuntime,
)


def _goal_contract(*, question: str = "Explore the verified change.") -> OriginalQuestionGoalContract:
    return OriginalQuestionGoalContract(
        question=question,
        goals=[
            AnalysisQuestionGoal(
                goal_id="goal.analysis",
                label="Explore the verified change",
                analysis_type="OPEN_EXPLORATION",
                input_goal_ids=["goal.measure"],
            ),
            MetricQuestionGoal(
                goal_id="goal.measure",
                label="Retain the requested measure",
            ),
        ],
    )


def _source(*, fingerprint: str = "source.fingerprint") -> VerifiedExplorationSourceView:
    return VerifiedExplorationSourceView(
        artifact_id="source.artifact",
        artifact_fingerprint=fingerprint,
        goal_ids=("goal.measure",),
        observations=(
            VerifiedExplorationObservation(
                observation_id="observation.one",
                statement="The verified artifact reports a measurable change.",
                evidence_refs=("evidence.one",),
            ),
        ),
    )


def _spec() -> GroundedExplorationAssignmentSpec:
    return GroundedExplorationAssignmentSpec(
        assignment_id="assignment.one",
        objective="Explore competing explanations for the verified change.",
        authorized_goal_ids=("goal.analysis",),
        source_artifact_ids=("source.artifact",),
        scope_authority=GroundedExplorationScopeAuthority(
            population_scope_fingerprint="population.base",
            allowed_narrower_population_fingerprints=("population.narrow",),
            time_scope_fingerprint="time.base",
            allowed_narrower_time_fingerprints=("time.narrow",),
        ),
    )


def _scope(kind: str) -> ExplorationScopeSignature:
    fingerprint = "%s.base" % kind
    return ExplorationScopeSignature(
        relation="INHERIT",
        fingerprint=fingerprint,
        parent_fingerprint=fingerprint,
    )


def _artifact(
    *,
    artifact_id: str = "advisory.one",
    request_id: str = "request.one",
    rationale: str = "Acquire evidence only after Root approval.",
    source_artifacts: tuple[str, ...] = ("source.fingerprint",),
) -> AdvisoryExplorationArtifact:
    hypothesis = HypothesisProposal(
        hypothesis_id="hypothesis.one",
        falsifiable_statement="The verified change is concentrated in one comparison.",
        premises=("The verified input is comparable within its bound scope.",),
        expected_observations=("A comparison result differs from its reference.",),
        falsifying_observations=("The comparison remains within its reference.",),
        goal_ids=("goal.analysis",),
        population_scope_fingerprint="population.base",
        time_scope_fingerprint="time.base",
        competing_explanations=("The apparent change is distributed.",),
    )
    evidence_request = EvidenceRequest(
        request_id=request_id,
        capability="COMPARE_GROUPS",
        evidence_shape="COMPARISON_RESULT",
        goal_ids=("goal.analysis",),
        hypothesis_ids=("hypothesis.one",),
        population_scope=_scope("population"),
        time_scope=_scope("time"),
        source_artifact_fingerprints=source_artifacts,
        rationale=rationale,
    )
    return AdvisoryExplorationArtifact(
        artifact_id=artifact_id,
        assignment_id="assignment.one",
        hypotheses=(hypothesis,),
        evidence_requests=(evidence_request,),
        analysis_plan=AnalysisPlan(
            plan_id="plan.one",
            steps=(
                AnalysisPlanStep(
                    step_id="step.one",
                    goal_ids=("goal.analysis",),
                    evidence_request_ids=(request_id,),
                    hypothesis_ids=("hypothesis.one",),
                ),
            ),
            terminal_step_ids=("step.one",),
        ),
        stopping_assessment=StoppingAssessment(
            decision="CONTINUE",
            goal_ids=("goal.analysis",),
            unresolved_hypothesis_ids=("hypothesis.one",),
            outstanding_request_ids=(request_id,),
            rationale="The requested evidence has not been acquired.",
        ),
        source_artifact_fingerprints=source_artifacts,
    )


class _Worker:
    def __init__(self, artifact: AdvisoryExplorationArtifact | object) -> None:
        self.artifact = artifact
        self.invocations: list[Any] = []

    def run(self, invocation: Any) -> Any:
        self.invocations.append(invocation)
        return self.artifact


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return "%s.%s" % (prefix, self.value)


def _coordinator(
    worker: _Worker,
    *,
    catalog: Any | None = None,
    store: Any | None = None,
) -> tuple[GroundedExplorationCoordinator, InMemoryGroundedExplorationStateStore]:
    actual_store = store or InMemoryGroundedExplorationStateStore()
    return (
        GroundedExplorationCoordinator(
            artifact_catalog=catalog
            or InMemoryVerifiedExplorationArtifactCatalog((_source(),)),
            state_store=actual_store,
            worker=worker,
            identifier_factory=_Ids(),
        ),
        actual_store,
    )


def _issue(
    coordinator: GroundedExplorationCoordinator,
    contract: OriginalQuestionGoalContract | None = None,
) -> Any:
    return coordinator.issue_assignment(contract or _goal_contract(), _spec())


def test_assignment_is_signed_from_active_goal_contract_and_verified_sources() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)

    receipt = _issue(coordinator)
    state = coordinator.state("assignment.one")

    assert receipt.ledger_revision == 1
    assert receipt.source_artifact_ids == ("source.artifact",)
    assert receipt.source_artifact_fingerprints == ("source.fingerprint",)
    assert state.ledger.events[0].event_type == "ASSIGNMENT_ACCEPTED"
    assert state.ledger.events[0].actor == "ROOT_KERNEL"
    assert state.assignment.output_authority == "ADVISORY"
    assert worker.invocations == []


def test_assignment_fails_closed_when_verified_catalog_is_incomplete() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(
        worker,
        catalog=InMemoryVerifiedExplorationArtifactCatalog(),
    )

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        _issue(coordinator)

    assert captured.value.code == "VERIFIED_EXPLORATION_SOURCE_INCOMPLETE"
    assert worker.invocations == []


def test_assignment_rejects_catalog_entry_without_verified_status() -> None:
    class UnverifiedCatalog:
        def resolve_verified(self, artifact_ids: Any) -> Any:
            payload = _source().model_dump(by_alias=True, mode="json")
            payload["verificationStatus"] = "UNVERIFIED"
            return (payload,)

    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker, catalog=UnverifiedCatalog())

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        _issue(coordinator)

    assert captured.value.code == "VERIFIED_EXPLORATION_SOURCE_UNAVAILABLE"
    assert worker.invocations == []


def test_bound_source_projection_is_revalidated_before_worker_runs() -> None:
    class MutableCatalog:
        source = _source()

        def resolve_verified(self, artifact_ids: Any) -> Any:
            return (self.source,)

    catalog = MutableCatalog()
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker, catalog=catalog)
    receipt = _issue(coordinator)
    catalog.source = VerifiedExplorationSourceView(
        artifact_id="source.artifact",
        artifact_fingerprint="source.fingerprint",
        goal_ids=("goal.measure",),
        observations=(
            VerifiedExplorationObservation(
                observation_id="observation.one",
                statement="A changed projection with the same outer fingerprint.",
            ),
        ),
    )

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "EXPLORATION_SOURCE_ARTIFACT_CHANGED"
    assert worker.invocations == []


def test_valid_worker_artifact_is_cas_appended_and_requests_root_approval() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)

    report = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=receipt.ledger_revision,
    )
    state = coordinator.state("assignment.one")

    assert report.status == "ADVISORY_ACCEPTED"
    assert report.authority == "ADVISORY"
    assert report.publishable_as_final is False
    assert report.query_executed is False
    assert report.ledger_revision == 6
    assert [item.event_type for item in state.ledger.events] == [
        "ASSIGNMENT_ACCEPTED",
        "HYPOTHESIS_PROPOSED",
        "EVIDENCE_REQUESTED",
        "ANALYSIS_PLAN_PROPOSED",
        "STOPPING_ASSESSED",
        "ADVISORY_ARTIFACT_EMITTED",
    ]
    assert len(report.pending_capability_requests) == 1
    pending = report.pending_capability_requests[0]
    assert pending.status == "PENDING_ROOT_APPROVAL"
    assert pending.root_decision_required is True
    assert pending.executable is False
    assert pending.query_dispatched is False
    assert coordinator.pending_capability_requests("assignment.one") == (pending,)
    invocation = worker.invocations[0]
    assert invocation.query_capability_mounted is False
    assert invocation.publication_capability_mounted is False
    assert invocation.verified_sources == (_source(),)


def test_pending_request_contract_has_no_executable_selection_surface() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)
    report = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=receipt.ledger_revision,
    )
    payload = report.pending_capability_requests[0].model_dump(
        by_alias=True,
        mode="json",
    )

    forbidden_keys = {
        "table",
        "tableName",
        "field",
        "fieldName",
        "formula",
        "metricFormula",
        "sql",
        "finalAnswer",
    }

    def collect_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(
                *(collect_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(collect_keys(item) for item in value))
        return set()

    assert forbidden_keys.isdisjoint(collect_keys(payload))
    payload["sql"] = "impermissible"
    with pytest.raises(ValidationError):
        PendingExplorationCapabilityRequest.model_validate(payload)


def test_stale_goal_contract_is_rejected_before_worker_runs() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(question="A changed active question."),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "EXPLORATION_ASSIGNMENT_STALE"
    assert worker.invocations == []


def test_non_advisory_worker_output_is_rejected_without_state_change() -> None:
    worker = _Worker({"finalAnswer": "impermissible"})
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "EXPLORATION_WORKER_OUTPUT_TYPE_REJECTED"
    assert coordinator.state("assignment.one").ledger.revision == 1


def test_advisory_artifact_must_explicitly_bind_a_verified_source() -> None:
    worker = _Worker(_artifact(source_artifacts=()))
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "ADVISORY_ARTIFACT_SOURCE_BINDING_REQUIRED"
    assert coordinator.state("assignment.one").ledger.revision == 1


def test_worker_failure_is_structured_and_never_mutates_state() -> None:
    class FailingWorker:
        def run(self, invocation: Any) -> Any:
            raise RuntimeError("worker failed")

    coordinator, _ = _coordinator(FailingWorker())  # type: ignore[arg-type]
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "EXPLORATION_WORKER_FAILED"
    assert coordinator.state("assignment.one").ledger.revision == 1


class _ConflictStore(InMemoryGroundedExplorationStateStore):
    def compare_and_swap(
        self,
        assignment_id: str,
        *,
        expected_revision: int,
        replacement: Any,
    ) -> bool:
        return False


def test_cas_conflict_never_partially_appends_artifact_or_pending_request() -> None:
    worker = _Worker(_artifact())
    store = _ConflictStore()
    coordinator, _ = _coordinator(worker, store=store)
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert captured.value.code == "EXPLORATION_LEDGER_REVISION_CONFLICT"
    state = coordinator.state("assignment.one")
    assert state.ledger.revision == 1
    assert state.accepted_artifacts == ()
    assert state.pending_capability_requests == ()


def test_exact_artifact_replay_is_idempotent() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)
    first = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=receipt.ledger_revision,
    )
    second = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=first.ledger_revision,
    )

    assert second.status == "IDEMPOTENT_REPLAY"
    assert second.ledger_revision == first.ledger_revision
    assert second.pending_capability_requests == first.pending_capability_requests
    assert coordinator.state("assignment.one").ledger.revision == first.ledger_revision


def test_equivalent_request_across_new_artifact_is_rejected() -> None:
    worker = _Worker(_artifact())
    coordinator, _ = _coordinator(worker)
    receipt = _issue(coordinator)
    first = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=receipt.ledger_revision,
    )
    worker.artifact = _artifact(
        artifact_id="advisory.two",
        request_id="request.two",
        rationale="Different prose must not defeat semantic deduplication.",
    )

    with pytest.raises(GroundedExplorationCoordinatorError) as captured:
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=first.ledger_revision,
        )

    assert captured.value.code == "DUPLICATE_EVIDENCE_REQUEST_ACROSS_LEDGER"
    assert coordinator.state("assignment.one").ledger.revision == first.ledger_revision


class _FakeIsolatedRuntime:
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output
        self.job: Any = None

    def run(self, job: Any, *, on_progress: Any = None) -> IsolatedSubagentResult:
        self.job = job
        return IsolatedSubagentResult(
            job_id=job.job_id,
            thread_id=job.thread_id,
            checkpoint={},
            raw_output=self.raw_output,
            update_count=1,
        )


def test_isolated_worker_mounts_zero_capabilities_and_parses_only_advisory_json() -> None:
    runtime = _FakeIsolatedRuntime(
        json.dumps(_artifact().model_dump(by_alias=True, mode="json"))
    )
    worker = IsolatedGroundedExplorationWorker(
        runtime,  # type: ignore[arg-type]
        parent_thread_id="thread.one",
    )
    coordinator, _ = _coordinator(worker)  # type: ignore[arg-type]
    receipt = _issue(coordinator)

    report = coordinator.run_assignment(
        "assignment.one",
        _goal_contract(),
        expected_revision=receipt.ledger_revision,
    )

    assert report.status == "ADVISORY_ACCEPTED"
    assert runtime.job.tools == []
    assert runtime.job.skills == []
    assert runtime.job.permissions == []
    assert runtime.job.subagents == []
    assert runtime.job.middleware == []
    assert runtime.job.backend is None
    assert runtime.job.user_payload["invocation"]["queryCapabilityMounted"] is False
    assert runtime.job.user_payload["invocation"]["publicationCapabilityMounted"] is False


@pytest.mark.parametrize(
    "raw_output",
    [
        "not json",
        json.dumps([_artifact().model_dump(by_alias=True, mode="json")]),
        json.dumps(
            {
                **_artifact().model_dump(by_alias=True, mode="json"),
                "finalAnswer": "impermissible",
            }
        ),
    ],
)
def test_isolated_worker_rejects_non_object_or_expanded_output(raw_output: str) -> None:
    runtime = _FakeIsolatedRuntime(raw_output)
    worker = IsolatedGroundedExplorationWorker(
        runtime,  # type: ignore[arg-type]
        parent_thread_id="thread.one",
    )
    coordinator, _ = _coordinator(worker)  # type: ignore[arg-type]
    receipt = _issue(coordinator)

    with pytest.raises(GroundedExplorationCoordinatorError):
        coordinator.run_assignment(
            "assignment.one",
            _goal_contract(),
            expected_revision=receipt.ledger_revision,
        )

    assert coordinator.state("assignment.one").ledger.revision == 1


def test_generic_isolated_runtime_builds_safe_agent_name_without_pattern_matching() -> None:
    captured: dict[str, Any] = {}

    class Graph:
        def stream(self, payload: Any, *, config: Any, stream_mode: str) -> list[Any]:
            return []

        def get_state(self, config: Any) -> Any:
            return type("Snapshot", (), {"values": {"messages": []}})()

    def factory(**kwargs: Any) -> Graph:
        captured.update(kwargs)
        return Graph()

    runtime = IsolatedSubagentRuntime(
        model=object(),
        agent_factory=factory,
        checkpointer=object(),
    )
    runtime.run(
        IsolatedSubagentJob(
            job_id="job-one/二 three",
            thread_id="thread.one",
            system_prompt="Return an advisory object.",
            user_payload={},
            backend=None,
        )
    )

    name = captured["name"]
    segment = name.removeprefix("isolated_worker_")
    assert name.startswith("isolated_worker_")
    assert segment
    assert len(segment) <= 48
    assert all(
        "a" <= character <= "z"
        or "A" <= character <= "Z"
        or "0" <= character <= "9"
        or character == "_"
        for character in segment
    )


def test_generic_isolated_runtime_binds_requested_provider_timeout() -> None:
    captured: dict[str, Any] = {}

    class Model:
        def bind(self, **kwargs: Any) -> Any:
            captured["binding"] = dict(kwargs)
            return "bounded-model"

    class Graph:
        def stream(self, payload: Any, *, config: Any, stream_mode: str) -> list[Any]:
            return []

        def get_state(self, config: Any) -> Any:
            return type("Snapshot", (), {"values": {"messages": []}})()

    def factory(**kwargs: Any) -> Graph:
        captured["model"] = kwargs["model"]
        return Graph()

    runtime = IsolatedSubagentRuntime(
        model=Model(),
        agent_factory=factory,
        checkpointer=object(),
    )
    runtime.run(
        IsolatedSubagentJob(
            job_id="job-timeout",
            thread_id="thread.timeout",
            system_prompt="Return an advisory object.",
            user_payload={},
            backend=None,
            model_timeout_seconds=3.5,
        )
    )

    assert captured["binding"] == {"timeout": 3.5}
    assert captured["model"] == "bounded-model"
