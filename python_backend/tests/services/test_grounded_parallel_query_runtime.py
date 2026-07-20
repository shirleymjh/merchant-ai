from __future__ import annotations

import hashlib
import json
import time
from threading import Event, RLock
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.models import (
    AgentTaskResult,
    AgentRunResult,
    DataSnapshotContract,
    EvidenceGap,
    QueryBundle,
    QueryPlan,
    SqlValidationResult,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedParallelExecutionSpec,
)
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionEdgeSpec,
    GroundedExecutionGraphReceipt,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeSession,
    GroundedVerifiedQueryArtifact,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class CapturingAgentFactory:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace()


class ConcurrentBranchKernel(GroundedRuntimeKernel):
    """Small execution fake that retains the real branch/adoption kernel."""

    def __init__(
        self,
        *,
        failed_query_ids: set[str] | None = None,
        evidence_gapped_query_ids: set[str] | None = None,
        internal_failed_query_ids: set[str] | None = None,
        require_overlap: bool = True,
        data_snapshot: DataSnapshotContract | None = None,
        returned_snapshot_by_query_id: dict[str, DataSnapshotContract] | None = None,
    ) -> None:
        super().__init__(
            object(),
            keyword_service=object(),
            topic_router=object(),
        )
        self.failed_query_ids = set(failed_query_ids or set())
        self.evidence_gapped_query_ids = set(
            evidence_gapped_query_ids or set()
        )
        self.internal_failed_query_ids = set(
            internal_failed_query_ids or set()
        )
        self.require_overlap = require_overlap
        self.data_snapshot = data_snapshot
        self.returned_snapshot_by_query_id = dict(
            returned_snapshot_by_query_id or {}
        )
        self.snapshot_capture_calls = 0
        self.probe_lock = RLock()
        self.two_workers_active = Event()
        self.active_workers = 0
        self.max_active_workers = 0
        self.execute_query_ids: list[str] = []
        self.execute_runtime_budgets: dict[
            str, GroundedRuntimeBudget | None
        ] = {}
        self.verify_query_ids: list[str] = []
        self.seen_session_ids: dict[str, str] = {}

    def execute_active(
        self,
        session: GroundedRuntimeSession,
        **kwargs: Any,
    ) -> AgentRunResult:
        query_id = str(session.user_scope["queryId"])
        with self.probe_lock:
            self.active_workers += 1
            self.max_active_workers = max(
                self.max_active_workers,
                self.active_workers,
            )
            self.execute_query_ids.append(query_id)
            self.execute_runtime_budgets[query_id] = kwargs.get("runtime_budget")
            self.seen_session_ids[query_id] = session.session_id
            if self.active_workers >= 2:
                self.two_workers_active.set()
        try:
            if self.require_overlap and not self.two_workers_active.wait(timeout=3):
                raise AssertionError("parallel branch workers did not overlap")
            # Keep the overlap observable even on a fast CI worker.
            time.sleep(0.03)
            if query_id in self.internal_failed_query_ids:
                raise RuntimeError("planned branch failure: %s" % query_id)
            if query_id in self.failed_query_ids:
                failed_bundle = QueryBundle(
                    failed=True,
                    error="DORIS_ERROR: planned branch failure",
                )
                return AgentRunResult(
                    task_results=[
                        AgentTaskResult(
                            task_id=query_id,
                            success=False,
                            query_bundle=failed_bundle,
                            validation_results=[
                                SqlValidationResult(
                                    valid=False,
                                    error_code="DORIS_ERROR",
                                    message="planned branch failure",
                                )
                            ],
                        )
                    ],
                    merged_query_bundle=failed_bundle,
                )
            session.active_generation += int(
                session.user_scope.get("generationDelta", 0)
            )
            session.user_scope["executionMarker"] = query_id
            result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"query_id": query_id, "value": 1}],
                    tables=["table_%s" % query_id],
                    data_snapshot=(
                        self.returned_snapshot_by_query_id.get(query_id)
                        or kwargs.get("data_snapshot_contract")
                        or DataSnapshotContract()
                    ),
                )
            )
            session.run_result = result
            return result
        finally:
            with self.probe_lock:
                self.active_workers -= 1

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        del semantic_activation_fingerprint
        self.snapshot_capture_calls += 1
        return (
            self.data_snapshot.model_copy(deep=True)
            if self.data_snapshot is not None
            else DataSnapshotContract(
                unsupported_reason="TEST_SNAPSHOT_CAPABILITY_UNAVAILABLE"
            )
        )

    def verify_active(self, session: GroundedRuntimeSession) -> VerifiedEvidence:
        query_id = str(session.user_scope["queryId"])
        self.verify_query_ids.append(query_id)
        assert session.run_result is not None
        assert session.active_contract is not None
        if query_id in self.evidence_gapped_query_ids:
            verified = VerifiedEvidence(
                passed=False,
                blocking_gaps=[
                    EvidenceGap(
                        code="RESULT_TIME_WINDOW_NOT_PROVEN",
                        reason="returned rows do not prove the requested window",
                        severity="blocking",
                    )
                ],
                partial_answer_reason="requested time window is not proven",
            )
            session.verified_evidence = verified
            session.phase = "VERIFICATION_GAPPED"
            return verified
        verified = VerifiedEvidence(passed=True, covered_evidence=[query_id])
        artifact = GroundedVerifiedQueryArtifact(
            artifact_id="artifact_%s" % query_id,
            generation=session.active_generation,
            attempt_id=session.active_attempt_id,
            contract_fingerprint=grounded_query_contract_fingerprint(
                session.active_contract
            ),
            sql_fingerprint=hashlib.sha256(query_id.encode("utf-8")).hexdigest(),
            contract=session.active_contract,
            plan=session.active_plan or QueryPlan(),
            run_result=session.run_result,
            verified_evidence=verified,
            output_columns=["query_id", "value"],
        )
        session.verified_evidence = verified
        session.verified_query_ledger.append(artifact)
        return verified

    @staticmethod
    def latest_verified_query_artifact(
        session: GroundedRuntimeSession,
    ) -> GroundedVerifiedQueryArtifact | None:
        return session.verified_query_ledger[-1] if session.verified_query_ledger else None


def _runtime(kernel: ConcurrentBranchKernel) -> GroundedDeepAgentRuntime:
    return GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=object(),
        parallel_max_workers=2,
        agent_factory=CapturingAgentFactory(),
        backend=object(),
    )


def _prepared_context(
    kernel: ConcurrentBranchKernel,
    *,
    generations: tuple[int, int] = (1, 1),
    generation_deltas: tuple[int, int] = (0, 0),
    budget: GroundedRuntimeBudget | None = None,
) -> tuple[
    GroundedDeepAgentRunContext,
    GroundedRuntimeSession,
    GroundedRuntimeSession,
    GroundedRuntimeSession,
]:
    parent = kernel.new_session(
        "compare two independent metrics",
        "merchant-1",
        session_id="parent-session",
        user_scope={"tenantMarker": "parent"},
    )
    first = kernel.fork_query_branch(parent, "first")
    second = kernel.fork_query_branch(parent, "second")
    for query_id, branch, generation, delta in (
        ("first", first, generations[0], generation_deltas[0]),
        ("second", second, generations[1], generation_deltas[1]),
    ):
        branch.user_scope.update(
            {
                "queryId": query_id,
                "generationDelta": delta,
            }
        )
        branch.active_generation = generation
        branch.active_attempt_id = "attempt_%s" % query_id
        branch.active_execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
        branch.active_contract = GroundedQueryContract(
            question="metric %s" % query_id,
            status="READY",
            query_shape="SCALAR",
        )
        branch.active_plan = QueryPlan(agent_trace=[query_id])

    deep_session = GroundedDeepAgentSession(
        runtime=parent,
        parallel_branches={"first": first, "second": second},
        parallel_branch_goal_ids={
            "first": ["goal.first"],
            "second": ["goal.second"],
        },
    )
    return (
        GroundedDeepAgentRunContext(
            thread_id="parallel-thread",
            run_id="parallel-run",
            session=deep_session,
            budget=budget,
        ),
        parent,
        first,
        second,
    )


def _execute_batch(
    runtime: GroundedDeepAgentRuntime,
    context: GroundedDeepAgentRunContext,
) -> dict[str, Any]:
    tools = {item.name: item for item in runtime.tools}
    return json.loads(
        tools["execute_grounded_query_batch"].func(
            queries=[
                GroundedParallelExecutionSpec(query_id="first"),
                GroundedParallelExecutionSpec(query_id="second"),
            ],
            reason="independent goals",
            runtime=SimpleNamespace(context=context),
        )
    )


def _execute_query_ids(
    runtime: GroundedDeepAgentRuntime,
    context: GroundedDeepAgentRunContext,
    query_ids: list[str],
) -> dict[str, Any]:
    tools = {item.name: item for item in runtime.tools}
    return json.loads(
        tools["execute_grounded_query_batch"].func(
            queries=[
                GroundedParallelExecutionSpec(query_id=query_id)
                for query_id in query_ids
            ],
            reason="selected graph frontier",
            runtime=SimpleNamespace(context=context),
        )
    )


def _snapshot(mode: str) -> DataSnapshotContract:
    return DataSnapshotContract(
        datasource_fingerprint="datasource",
        datasource_environment="test",
        data_epoch="epoch-1",
        consistency_mode=mode,
        semantic_activation_fingerprint="graph-semantic-v1",
        cache_generation="cache-v1",
    )


def _freeze_test_graph(
    context: GroundedDeepAgentRunContext,
    *,
    contract_scope: bool,
) -> None:
    context.session.question_goal_contract = OriginalQuestionGoalContract(
        question=context.session.runtime.question,
        goals=[
            MetricQuestionGoal(goal_id="goal.first", label="first"),
            MetricQuestionGoal(goal_id="goal.second", label="second"),
        ],
    )
    context.session.execution_graph_fingerprint = "graph-semantic-v1"
    context.session.execution_graph_receipt = GroundedExecutionGraphReceipt(
        graph_id="graph-test",
        version=1,
        fingerprint="graph-semantic-v1",
        discovery_snapshot_fingerprint="discovery-v1",
        node_ids={"first_node": "first", "second_node": "second"},
        parallel_frontier=["first", "second"],
    )
    context.session.execution_graph_edges = (
        [
            GroundedExecutionEdgeSpec(
                source_client_key="first_node",
                target_client_key="second_node",
                dependency_mode="CONTRACT_SCOPE",
            )
        ]
        if contract_scope
        else []
    )


def test_parallel_batch_overlaps_workers_and_adopts_only_verified_success() -> None:
    kernel = ConcurrentBranchKernel(
        failed_query_ids={"second"},
        data_snapshot=_snapshot("OBSERVED_EPOCH"),
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=False)

    result = _execute_batch(runtime, context)

    assert kernel.max_active_workers == 2
    assert set(kernel.execute_query_ids) == {"first", "second"}
    assert kernel.verify_query_ids == ["first"]
    assert result["status"] == "PARTIAL"
    assert result["replanRequired"] is True
    assert result["nextAction"] == "REOPEN_GRAPH_FOR_RECOVERY"
    assert len(result["replanEvidenceSet"]) == 1
    assert result["replanEvidenceSetFingerprint"]
    assert result["executedInParallel"] is True
    assert result["adoptedArtifactIds"] == ["artifact_first"]
    assert [item["status"] for item in result["queries"]] == [
        "VERIFIED",
        "REPLAN_REQUIRED",
    ]
    assert [item.artifact_id for item in parent.verified_query_ledger] == [
        "artifact_first"
    ]
    assert context.session.artifact_goal_ids == {
        "artifact_first": ["goal.first"]
    }


def test_internal_parallel_failure_is_terminal_without_partial_adoption() -> None:
    kernel = ConcurrentBranchKernel(
        internal_failed_query_ids={"second"}
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)

    result = _execute_batch(runtime, context)

    assert result["status"] == "OPERATIONAL_FAILURE"
    assert parent.verified_query_ledger == []
    assert context.session.execution_graph_replan_evidence == {}
    failed = next(
        item for item in result["queries"]
        if item["queryId"] == "second"
    )
    assert failed["failureDisposition"] == "OPERATIONAL_TERMINAL"
    assert failed["replanEvidence"] == {}


def test_parallel_verification_gap_reopens_only_gapped_node() -> None:
    kernel = ConcurrentBranchKernel(
        evidence_gapped_query_ids={"second"},
        data_snapshot=_snapshot("OBSERVED_EPOCH"),
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=False)

    result = _execute_batch(runtime, context)

    assert result["status"] == "PARTIAL"
    assert result["replanRequired"] is True
    gapped = next(
        item for item in result["queries"] if item["queryId"] == "second"
    )
    assert gapped["status"] == "REPLAN_REQUIRED"
    assert gapped["failureDisposition"] == "EVIDENCE_GAPPED"
    assert gapped["code"] == "RESULT_TIME_WINDOW_NOT_PROVEN"
    assert gapped["replanEvidence"]["triggerKind"] == "DATA_GAP"
    assert gapped["replanEvidence"]["sourceQueryNodeId"] == "second"
    assert [item.artifact_id for item in parent.verified_query_ledger] == [
        "artifact_first"
    ]


def test_contract_scope_batch_is_blocked_before_doris_without_atomic_snapshot() -> None:
    kernel = ConcurrentBranchKernel(
        require_overlap=False,
        data_snapshot=_snapshot("OBSERVED_EPOCH"),
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=True)

    result = _execute_batch(runtime, context)

    assert result["status"] == "BLOCKED"
    assert result["code"] == "MULTI_QUERY_SNAPSHOT_CONTRACT_UNSATISFIED"
    assert result["snapshotIssues"] == [
        "ATOMIC_MULTI_QUERY_SNAPSHOT_UNSUPPORTED"
    ]
    assert result["snapshotRequirement"]["requireAtomicMultiQuery"] is True
    assert kernel.execute_query_ids == []
    assert parent.verified_query_ledger == []


def test_contract_scope_cannot_bypass_atomic_gate_by_executing_one_node_at_a_time() -> None:
    kernel = ConcurrentBranchKernel(
        require_overlap=False,
        data_snapshot=_snapshot("OBSERVED_EPOCH"),
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=True)

    result = _execute_query_ids(runtime, context, ["first"])

    assert result["status"] == "BLOCKED"
    assert result["snapshotIssues"] == [
        "ATOMIC_MULTI_QUERY_SNAPSHOT_UNSUPPORTED"
    ]
    assert kernel.execute_query_ids == []
    assert parent.verified_query_ledger == []


def test_independent_graph_nodes_share_observed_epoch_and_still_run_in_parallel() -> None:
    kernel = ConcurrentBranchKernel(data_snapshot=_snapshot("OBSERVED_EPOCH"))
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=False)

    result = _execute_batch(runtime, context)

    assert result["status"] == "VERIFIED"
    assert result["executedInParallel"] is True
    assert kernel.max_active_workers == 2
    assert len(parent.verified_query_ledger) == 2


def test_independent_nodes_executed_separately_reuse_one_frozen_snapshot() -> None:
    kernel = ConcurrentBranchKernel(
        require_overlap=False,
        data_snapshot=_snapshot("OBSERVED_EPOCH"),
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=False)

    first = _execute_query_ids(runtime, context, ["first"])
    second = _execute_query_ids(runtime, context, ["second"])

    assert first["status"] == "VERIFIED"
    assert second["status"] == "VERIFIED"
    assert kernel.snapshot_capture_calls == 1
    assert [
        artifact.artifact_id for artifact in parent.verified_query_ledger
    ] == ["artifact_first", "artifact_second"]


def test_contract_scope_batch_runs_when_adapter_proves_one_as_of_snapshot() -> None:
    kernel = ConcurrentBranchKernel(data_snapshot=_snapshot("AS_OF_READ"))
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=True)

    result = _execute_batch(runtime, context)

    assert result["status"] == "VERIFIED"
    assert kernel.max_active_workers == 2
    assert len(parent.verified_query_ledger) == 2


def test_snapshot_mismatch_after_execution_prevents_parent_adoption() -> None:
    shared = _snapshot("AS_OF_READ")
    changed = shared.model_copy(update={"data_epoch": "epoch-2"})
    kernel = ConcurrentBranchKernel(
        data_snapshot=shared,
        returned_snapshot_by_query_id={"second": changed},
    )
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel)
    _freeze_test_graph(context, contract_scope=True)

    result = _execute_batch(runtime, context)

    assert result["status"] == "BLOCKED"
    assert result["code"] == "MULTI_QUERY_SNAPSHOT_POSTFLIGHT_FAILED"
    assert result["snapshotIssues"] == ["DATA_SNAPSHOT_MISMATCH"]
    assert result["adoptedArtifactIds"] == []
    assert parent.verified_query_ledger == []


def test_parallel_branches_keep_generation_and_session_mutations_isolated() -> None:
    kernel = ConcurrentBranchKernel()
    runtime = _runtime(kernel)
    context, parent, first, second = _prepared_context(
        kernel,
        generations=(4, 9),
        generation_deltas=(2, 3),
    )
    parent.active_generation = 30

    result = _execute_batch(runtime, context)

    assert result["status"] == "VERIFIED"
    assert kernel.max_active_workers == 2
    assert kernel.seen_session_ids["first"] != kernel.seen_session_ids["second"]
    assert first.active_generation == 6
    assert second.active_generation == 12
    assert first.user_scope["executionMarker"] == "first"
    assert second.user_scope["executionMarker"] == "second"
    assert parent.user_scope == {"tenantMarker": "parent"}
    # Adoption advances the parent once; it does not copy either branch's
    # generation counter or branch-local user scope.
    assert parent.active_generation == 31
    assert [item.generation for item in parent.verified_query_ledger] == [6, 12]
    assert context.session.parallel_branches == {}
    assert context.session.parallel_branch_goal_ids == {}

    first.verified_query_ledger[0].output_columns.append("branch_only_mutation")
    assert "branch_only_mutation" not in parent.verified_query_ledger[0].output_columns


def test_parallel_workers_share_one_runtime_budget_for_doris_timeout_clamping() -> None:
    budget = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=30,
            max_llm_calls=2,
            max_tool_calls=10,
            max_doris_queries=2,
        )
    )
    kernel = ConcurrentBranchKernel()
    runtime = _runtime(kernel)
    context, _, _, _ = _prepared_context(kernel, budget=budget)

    result = _execute_batch(runtime, context)

    assert result["status"] == "VERIFIED"
    assert kernel.execute_runtime_budgets == {
        "first": budget,
        "second": budget,
    }
    assert budget.report()["usage"]["dorisQueries"] == 2


def test_parallel_doris_budget_exhaustion_aborts_batch_without_partial_adoption() -> None:
    budget = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=30,
            max_llm_calls=2,
            max_tool_calls=10,
            max_doris_queries=1,
        )
    )
    kernel = ConcurrentBranchKernel(require_overlap=False)
    runtime = _runtime(kernel)
    context, parent, _, _ = _prepared_context(kernel, budget=budget)

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        _execute_batch(runtime, context)

    report = budget.report()
    assert raised.value.breaches == ("doris_queries",)
    assert report["usage"]["dorisQueries"] == 1
    assert sum(report["usage"]["dorisQueriesByName"].values()) == 1
    assert any(
        "doris_queries" in attempt["breaches"]
        for attempt in report["deniedAttempts"]
    )
    assert len(kernel.execute_query_ids) == 1
    assert parent.verified_query_ledger == []
