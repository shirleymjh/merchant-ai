from __future__ import annotations

import json
import multiprocessing
import os
import traceback
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    ClarificationRequest,
    DataSnapshotContract,
    QueryBundle,
    SqlValidationResult,
)
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    _build_graph_revision_base_session_checkpoint,
    _execution_graph_node_runtime_states,
    _authorized_verified_query_artifacts,
    _published_query_artifact_digests,
)
from merchant_ai.services.grounded_execution_graph import (
    build_grounded_execution_graph_replan_evidence,
    discovery_evidence_snapshot_fingerprint,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeSession,
    verified_query_artifact_integrity_fingerprint,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)
from merchant_ai.services.grounded_graph_revision_journal import (
    GroundedGraphRevisionTransactionJournal,
)
from merchant_ai.services.grounded_subagent_runtime import (
    GroundedSubagentBudget,
    GroundedSubagentDispatchPlan,
    GroundedSubagentEvidenceRequirement,
    GroundedSubagentGoalContract,
    GroundedSubagentTaskOutcome,
    issue_grounded_subagent_capability_grant,
)
from tests.services.test_grounded_branch_scoped_runtime import (
    _context,
    _freeze_reopenable_execution_graph,
    _propose_test_execution_graph,
    _runtime,
    _set_frozen_branch_evidence_kind,
)


def _tools(runtime) -> dict[str, Any]:
    return {item.name: item for item in runtime.tools}


def _two_node_graph(runtime, context) -> dict[str, Any]:
    _tools(runtime)["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.first",
                    label="first metric",
                ),
                MetricQuestionGoal(
                    goal_id="metric.second",
                    label="second metric",
                ),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    return _propose_test_execution_graph(
        runtime,
        context,
        nodes=[
            {
                "clientKey": "first",
                "goalIds": ["metric.first"],
                "topicScope": ["电商交易"],
                "evidencePaths": ["topics/电商交易/tables/orders/metrics/order_count.json"],
            },
            {
                "clientKey": "second",
                "goalIds": ["metric.second"],
                "topicScope": ["电商交易"],
                "evidencePaths": ["topics/电商交易/tables/orders/metrics/order_count.json"],
            },
        ],
    )


class _RevisionCrash(RuntimeError):
    pass


class _IdempotentPopulationRevisionGate:
    def __init__(self, base_receipt_fingerprint: str) -> None:
        self.active_receipt_fingerprint = base_receipt_fingerprint
        self.revision_calls: list[tuple[str, str]] = []

    @staticmethod
    def register_run(**_kwargs: Any) -> None:
        return None

    def revise_graph(
        self,
        *,
        previous_graph_receipt_fingerprint: str,
        revised_graph_receipt,
        **kwargs: Any,
    ):
        del kwargs
        self.revision_calls.append(
            (
                previous_graph_receipt_fingerprint,
                revised_graph_receipt.receipt_fingerprint,
            )
        )
        if self.active_receipt_fingerprint == revised_graph_receipt.receipt_fingerprint:
            return SimpleNamespace(
                accepted=True,
                code="IDEMPOTENT",
                message="already committed",
            )
        if self.active_receipt_fingerprint != previous_graph_receipt_fingerprint:
            return SimpleNamespace(
                accepted=False,
                code="POPULATION_CAS_CONFLICT",
                message="stale population base",
            )
        self.active_receipt_fingerprint = revised_graph_receipt.receipt_fingerprint
        return SimpleNamespace(
            accepted=True,
            code="COMMITTED",
            message="committed",
        )


def _install_population_revision_base(
    context,
    frozen: dict[str, Any],
) -> PopulationDynamicGraphReceipt:
    receipt = seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id=frozen["receipt"]["graphId"],
            graph_version=frozen["receipt"]["version"],
            graph_fingerprint=frozen["receipt"]["fingerprint"],
            nodes=tuple(
                PopulationDynamicGraphNode(
                    query_node_id=query_id,
                    consumer_goal_ids=(context.session.query_branch_contexts[query_id].spec.goal_ids),
                )
                for query_id in frozen["clientNodeIds"].values()
            ),
        )
    )
    context.session.population_graph_receipt = receipt.model_copy(deep=True)
    return receipt


def _failed_revision_payload(
    context,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    receipt = context.session.execution_graph_receipt
    goal_contract = context.session.question_goal_contract
    assert receipt is not None
    assert goal_contract is not None
    triggers = []
    goal_ids = ("metric.first", "metric.second")
    for query_id in frozen["clientNodeIds"].values():
        context.session.query_branch_contexts[query_id].status = "FAILED"
        evidence = build_grounded_execution_graph_replan_evidence(
            trigger_kind="EXECUTION_ERROR",
            source_stage="EXECUTION",
            source_query_node_id=query_id,
            code="DORIS_ERROR",
            graph_receipt=receipt,
            details={"failureCodes": ["DORIS_ERROR"]},
        )
        context.session.execution_graph_replan_evidence[evidence.evidence_id] = evidence
        triggers.append(evidence)
    evidence_ref = "semantic:电商交易:orders:metric:order_count"
    return {
        "baseGraphId": frozen["receipt"]["graphId"],
        "baseVersion": frozen["receipt"]["version"],
        "baseFingerprint": frozen["receipt"]["fingerprint"],
        "triggerEvidenceSet": [
            {
                "evidenceId": item.evidence_id,
                "evidenceFingerprint": item.evidence_fingerprint,
            }
            for item in triggers
        ],
        "graph": {
            "baseVersion": frozen["receipt"]["version"],
            "goalContractFingerprint": (original_question_goal_contract_fingerprint(goal_contract)),
            "discoverySnapshotFingerprint": (
                discovery_evidence_snapshot_fingerprint(context.session.core_semantic_evidence)
            ),
            "nodes": [
                {
                    "clientKey": "%s_recovery" % key,
                    "goalIds": [goal_id],
                    "topicScope": ["电商交易"],
                    "evidenceRefIds": [evidence_ref],
                }
                for key, goal_id in zip(
                    ("first", "second"),
                    goal_ids,
                )
            ],
        },
    }


def _carried_verified_branch_revision_payload(
    context,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    receipt = context.session.execution_graph_receipt
    proposal = context.session.execution_graph_proposal
    assert receipt is not None
    assert proposal is not None
    failed_query_id = frozen["clientNodeIds"]["second"]
    context.session.query_branch_contexts[failed_query_id].status = "FAILED"
    evidence = build_grounded_execution_graph_replan_evidence(
        trigger_kind="EXECUTION_ERROR",
        source_stage="EXECUTION",
        source_query_node_id=failed_query_id,
        code="DORIS_ERROR",
        graph_receipt=receipt,
        details={"failureCodes": ["DORIS_ERROR"]},
    )
    context.session.execution_graph_replan_evidence[
        evidence.evidence_id
    ] = evidence
    carried = next(
        item for item in proposal.nodes if item.client_key == "first"
    )
    failed = next(
        item for item in proposal.nodes if item.client_key == "second"
    )
    replacement = failed.model_dump(by_alias=True, mode="json")
    replacement["clientKey"] = "second_recovery"
    return {
        "baseGraphId": receipt.graph_id,
        "baseVersion": receipt.version,
        "baseFingerprint": receipt.fingerprint,
        "triggerEvidenceSet": [
            {
                "evidenceId": evidence.evidence_id,
                "evidenceFingerprint": evidence.evidence_fingerprint,
            }
        ],
        "graph": {
            "baseVersion": receipt.version,
            "goalContractFingerprint": (
                proposal.goal_contract_fingerprint
            ),
            "discoverySnapshotFingerprint": (
                proposal.discovery_snapshot_fingerprint
            ),
            "nodes": [
                carried.model_dump(by_alias=True, mode="json"),
                replacement,
            ],
            "edges": [],
        },
    }


def _restore_base_session(
    target,
    source,
    workspace: GroundedContextWorkspace,
) -> None:
    target.context_workspace = workspace
    target.core_semantic_evidence = json.loads(json.dumps(source.core_semantic_evidence, ensure_ascii=False))
    target.question_goal_contract = (
        source.question_goal_contract.model_copy(deep=True) if source.question_goal_contract is not None else None
    )
    target.execution_graph_generation = source.execution_graph_generation
    target.execution_graph_fingerprint = source.execution_graph_fingerprint
    target.execution_graph_proposal = (
        source.execution_graph_proposal.model_copy(deep=True) if source.execution_graph_proposal is not None else None
    )
    target.execution_graph_receipt = (
        source.execution_graph_receipt.model_copy(deep=True) if source.execution_graph_receipt is not None else None
    )
    target.execution_graph_edges = [item.model_copy(deep=True) for item in source.execution_graph_edges]
    target.population_graph_receipt = (
        source.population_graph_receipt.model_copy(deep=True) if source.population_graph_receipt is not None else None
    )


def _spawn_graph_revision_recovery(
    workspace_path: str,
    base_population_fingerprint: str,
    expected_artifact_id: str,
    result_queue: Any,
) -> None:
    try:
        runtime, kernel, catalog = _runtime(
            require_parallel_overlap=False
        )
        settings = Settings(harness_workspace_path=workspace_path)
        runtime.settings = settings
        runtime.population_gate_enforced = True
        runtime.population_execution_gate = (
            _IdempotentPopulationRevisionGate(
                base_population_fingerprint
            )
        )
        kernel.route_topic = lambda _session: None  # type: ignore[attr-defined]
        kernel.recall_navigation = (  # type: ignore[attr-defined]
            lambda _session: None
        )
        for topic in ("电商交易", "电商退货"):
            catalog.documents[
                "topics/%s/manifest.json" % topic
            ] = {
                "refId": "semantic:%s:manifest" % topic,
                "kind": "TOPIC_MANIFEST",
                "topic": topic,
                "content": json.dumps(
                    {"topic": topic},
                    ensure_ascii=False,
                ),
            }

        captured: dict[str, Any] = {}

        class _RecoveryBootstrapGraph:
            @staticmethod
            def invoke(
                payload: dict[str, Any],
                *,
                config: Any = None,
                context: Any = None,
            ) -> None:
                del config
                captured["initialContext"] = json.loads(
                    payload["messages"][0]["content"]
                )
                captured["session"] = context.session
                context.session.runtime.clarification = (
                    ClarificationRequest(
                        question="recovery acceptance complete",
                        stage="acceptance",
                        type="recovery_acceptance",
                    )
                )

        runtime.deep_agent_graph = _RecoveryBootstrapGraph()
        response = runtime._run_once(
            "two independent metrics",
            "merchant-1",
            access_role="merchant",
            user_scope={},
            thread_id="branch-thread",
            run_id="branch-run",
        )
        session = captured["session"]
        initial_context = captured["initialContext"]
        bootstrap_recovery = next(
            item
            for item in session.execution_graph_history
            if item.get("status")
            == "JOURNAL_RECOVERY_COMPLETED_AT_BOOTSTRAP"
        )
        reports = list(bootstrap_recovery["transactions"])
        receipt = session.execution_graph_receipt
        population_receipt = session.population_graph_receipt
        workspace = session.context_workspace
        assert receipt is not None
        assert population_receipt is not None
        assert workspace is not None
        result_queue.put(
            {
                "ok": True,
                "pid": os.getpid(),
                "responseClarificationType": (
                    response.clarification.type
                    if response.clarification is not None
                    else ""
                ),
                "reports": reports,
                "goalIds": [
                    item.goal_id
                    for item in (
                        session.question_goal_contract.goals
                        if session.question_goal_contract
                        is not None
                        else []
                    )
                ],
                "receipt": receipt.model_dump(
                    by_alias=True,
                    mode="json",
                ),
                "populationReceipt": population_receipt.model_dump(
                    by_alias=True,
                    mode="json",
                ),
                "branchStatuses": {
                    query_node_id: branch.status
                    for query_node_id, branch in (
                        session.query_branch_contexts.items()
                    )
                },
                "ledgerArtifactIds": [
                    item.artifact_id
                    for item in session.runtime.verified_query_ledger
                ],
                "authorizedArtifactIds": [
                    item.artifact_id
                    for item in _authorized_verified_query_artifacts(
                        session
                    )
                ],
                "artifactGoalIds": dict(
                    session.artifact_goal_ids
                ),
                "restoredExecutionState": initial_context[
                    "restoredExecutionState"
                ],
                "pendingCount": len(
                    GroundedGraphRevisionTransactionJournal(
                        workspace
                    ).discover_pending()
                ),
                "expectedArtifactId": expected_artifact_id,
            }
        )
    except BaseException:
        result_queue.put(
            {
                "ok": False,
                "pid": os.getpid(),
                "error": traceback.format_exc(),
            }
        )


def test_new_runtime_rolls_forward_each_graph_revision_crash_window(
    tmp_path,
    crash_stage: str,
    pending_phase: str,
    expected_population_calls: int,
) -> None:
    first_runtime, first_kernel, _ = _runtime(require_parallel_overlap=False)
    first_context = _context(
        first_kernel,
        "two independent metrics",
    )
    frozen = _two_node_graph(first_runtime, first_context)
    base_population = _install_population_revision_base(
        first_context,
        frozen,
    )
    revision = _failed_revision_payload(first_context, frozen)
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    first_workspace = GroundedContextWorkspace.open(
        settings,
        thread_id=first_context.thread_id,
        run_id=first_context.run_id,
        merchant_id="merchant-1",
        access_role="merchant",
        user_scope={},
        question=first_context.session.runtime.question,
    )
    first_context.session.context_workspace = first_workspace
    checkpoint_context = _context(
        first_kernel,
        "two independent metrics",
    )
    _restore_base_session(
        checkpoint_context.session,
        first_context.session,
        first_workspace,
    )
    population_gate = _IdempotentPopulationRevisionGate(base_population.receipt_fingerprint)
    crashed_transaction_ids: list[str] = []

    def crash(stage: str, transaction_id: str) -> None:
        if stage != crash_stage:
            return
        crashed_transaction_ids.append(transaction_id)
        raise _RevisionCrash(stage)

    first_runtime.settings = settings
    first_runtime.population_gate_enforced = True
    first_runtime.population_execution_gate = population_gate
    first_runtime.graph_revision_fault_injector = crash

    crashed = False
    try:
        _tools(first_runtime)["revise_grounded_execution_graph"].func(
            revision=revision,
            runtime=SimpleNamespace(context=first_context),
        )
    except _RevisionCrash:
        crashed = True
    assert crashed is True
    assert len(crashed_transaction_ids) == 1

    pending = GroundedGraphRevisionTransactionJournal(
        GroundedContextWorkspace.open(
            settings,
            thread_id=first_context.thread_id,
            run_id=first_context.run_id,
            merchant_id="merchant-1",
            access_role="merchant",
            user_scope={},
            question=first_context.session.runtime.question,
        )
    ).discover_pending()
    assert len(pending) == 1
    assert pending[0].transaction_id == crashed_transaction_ids[0]
    assert pending[0].phase == pending_phase

    restarted_runtime, restarted_kernel, _ = _runtime(require_parallel_overlap=False)
    restarted_context = _context(
        restarted_kernel,
        "two independent metrics",
    )
    restarted_workspace = GroundedContextWorkspace.open(
        settings,
        thread_id=restarted_context.thread_id,
        run_id=restarted_context.run_id,
        merchant_id="merchant-1",
        access_role="merchant",
        user_scope={},
        question=restarted_context.session.runtime.question,
    )
    _restore_base_session(
        restarted_context.session,
        checkpoint_context.session,
        restarted_workspace,
    )
    restarted_runtime.settings = settings
    restarted_runtime.population_gate_enforced = True
    restarted_runtime.population_execution_gate = population_gate

    recovered = json.loads(
        _tools(restarted_runtime)["revise_grounded_execution_graph"].func(
            revision=revision,
            runtime=SimpleNamespace(context=restarted_context),
        )
    )

    assert recovered["status"] == "REVISED"
    assert recovered["recovered"] is True
    assert recovered["receipt"]["version"] == 2
    assert restarted_context.session.execution_graph_receipt is not None
    assert restarted_context.session.execution_graph_receipt.fingerprint == recovered["receipt"]["fingerprint"]
    assert len(population_gate.revision_calls) == expected_population_calls
    assert GroundedGraphRevisionTransactionJournal(restarted_workspace).discover_pending() == ()


def test_recovery_hydrates_subgoal_generation_without_promoting_advisory_coverage() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "two independent metrics")
    _two_node_graph(runtime, context)
    population_receipt = _install_population_revision_base(
        context,
        {
            "receipt": context.session.execution_graph_receipt.model_dump(
                by_alias=True,
                mode="json",
            ),
            "clientNodeIds": dict(
                context.session.execution_graph_receipt.node_ids
            ),
        },
    )
    sub_goal = GroundedSubagentGoalContract(
        sub_goal_id="subgoal.recovery.audit",
        parent_goal_ids=["metric.first"],
        objective="Inspect one bounded recovery hypothesis.",
        required_outputs=["finding"],
        input_artifact_refs=[],
        evidence_requirements=[
            GroundedSubagentEvidenceRequirement(
                requirement_id="recovery.refs",
                description="Return exact refs for Root review.",
                accepted_ref_types=["SEMANTIC_REF"],
            )
        ],
        allowed_capabilities=["READ_CONTEXT"],
        budget=GroundedSubagentBudget(
            max_tool_calls=2,
            timeout_seconds=10,
        ),
        generation=1,
    )
    grant = issue_grounded_subagent_capability_grant(
        sub_goal,
        allowed_tool_names=["grep", "ls", "read_file"],
    )
    outcome = GroundedSubagentTaskOutcome(
        sub_goal_id=sub_goal.sub_goal_id,
        generation=1,
        status="COMPLETED",
        grant=grant,
        advisory_output={
            "summary": "advisory only",
            "finding": "semantic:topic-a:metric",
            "evidenceRefs": ["semantic:topic-a:metric"],
            "gaps": [],
            "recommendedNextAction": "ROOT_REVIEW",
            "proposedSubGoals": [],
            "evidenceGaps": [],
        },
    )
    context.session.subagent_dispatches = [
        {
            "dispatchId": "subdispatch.recovery",
            "parallel": False,
            "status": "COMPLETED",
            "tasks": [
                outcome.model_dump(by_alias=True, mode="json")
            ],
        }
    ]
    proposal = context.session.execution_graph_proposal
    receipt = context.session.execution_graph_receipt
    assert proposal is not None and receipt is not None
    checkpoint = _build_graph_revision_base_session_checkpoint(
        context.session,
        execution_proposal=proposal,
        execution_receipt=receipt,
        population_receipt=population_receipt,
        node_states=_execution_graph_node_runtime_states(
            context.session,
            receipt,
        ),
    )

    restarted_runtime, restarted_kernel, _ = _runtime(
        require_parallel_overlap=False
    )
    restarted_context = _context(
        restarted_kernel,
        "two independent metrics",
    )
    restored = restarted_runtime._restore_graph_revision_base_session(
        restarted_context.session,
        checkpoint,
        runtime_budget=None,
    )

    assert restored is True
    assert restarted_context.session.subagent_dispatches == (
        context.session.subagent_dispatches
    )
    assert restarted_context.session.goal_coverage_result == {}
    assert restarted_context.session.runtime.verified_query_ledger == []
    repeated = json.loads(
        _tools(restarted_runtime)["delegate_grounded_tasks"].func(
            plan=GroundedSubagentDispatchPlan(tasks=[sub_goal]),
            runtime=SimpleNamespace(context=restarted_context),
        )
    )
    assert repeated["code"] == "SUBAGENT_GOAL_GENERATION_INVALID"
    assert repeated["issues"][0]["expectedGeneration"] == 2
    assert restarted_context.session.goal_coverage_result == {}


def test_same_runtime_finishes_journal_after_execution_switch_without_repeating_population_or_graph_revision(
    tmp_path,
) -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(
        kernel,
        "two independent metrics",
    )
    frozen = _two_node_graph(runtime, context)
    base_population = _install_population_revision_base(
        context,
        frozen,
    )
    revision = _failed_revision_payload(context, frozen)
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace")
    )
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id=context.thread_id,
        run_id=context.run_id,
        merchant_id="merchant-1",
        access_role="merchant",
        user_scope={},
        question=context.session.runtime.question,
    )
    context.session.context_workspace = workspace
    population_gate = _IdempotentPopulationRevisionGate(
        base_population.receipt_fingerprint
    )
    history_count_before = len(
        context.session.execution_graph_history
    )
    revision_count_before = (
        context.session.execution_graph_revision_count
    )
    crashed_transaction_ids: list[str] = []

    def crash(stage: str, transaction_id: str) -> None:
        if stage != "AFTER_EXECUTION_SWITCH":
            return
        crashed_transaction_ids.append(transaction_id)
        raise _RevisionCrash(stage)

    runtime.settings = settings
    runtime.population_gate_enforced = True
    runtime.population_execution_gate = population_gate
    runtime.graph_revision_fault_injector = crash

    with pytest.raises(_RevisionCrash):
        _tools(runtime)["revise_grounded_execution_graph"].func(
            revision=revision,
            runtime=SimpleNamespace(context=context),
        )

    switched_receipt = context.session.execution_graph_receipt
    assert switched_receipt is not None
    assert switched_receipt.version == frozen["receipt"]["version"] + 1
    assert (
        context.session.execution_graph_revision_count
        == revision_count_before + 1
    )
    assert (
        len(context.session.execution_graph_history)
        == history_count_before + 1
    )
    assert len(population_gate.revision_calls) == 1
    pending = GroundedGraphRevisionTransactionJournal(
        workspace
    ).discover_pending()
    assert len(pending) == 1
    assert pending[0].transaction_id == crashed_transaction_ids[0]
    assert pending[0].phase == "POPULATION_COMMITTED"

    runtime.graph_revision_fault_injector = None
    recovered = json.loads(
        _tools(runtime)["revise_grounded_execution_graph"].func(
            revision=revision,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert recovered["status"] == "REVISED"
    assert recovered["recovered"] is True
    assert recovered["receipt"]["version"] == switched_receipt.version
    assert context.session.execution_graph_receipt is not None
    assert (
        context.session.execution_graph_receipt.fingerprint
        == switched_receipt.fingerprint
    )
    assert (
        context.session.execution_graph_revision_count
        == revision_count_before + 1
    )
    assert (
        len(context.session.execution_graph_history)
        == history_count_before + 1
    )
    assert len(population_gate.revision_calls) == 1
    assert (
        GroundedGraphRevisionTransactionJournal(
            workspace
        ).discover_pending()
        == ()
    )
