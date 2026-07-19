from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from merchant_ai.models import QueryPlan, QuestionIntent
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RuleQuestionGoal,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_rule_artifact import (
    GroundedRuleEvidenceRef,
    GroundedVerifiedRuleArtifact,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeSession,
)
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlValidationResult,
)


class _QuerylessPopulationGate:
    def __init__(self) -> None:
        self.completion_calls = 0

    def require_graph_complete(self, *, reference):
        self.completion_calls += 1
        raise AssertionError(
            "A verified zero-scope Goal must not require a query graph"
        )


class _GoalCaptureGate:
    def __init__(self) -> None:
        self.commit_run_authority = ""
        self.gate_id_run_authority = ""

    def commit_goal(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        exact_question: str,
        goal_contract: OriginalQuestionGoalContract,
    ):
        self.commit_run_authority = run_authority_fingerprint
        contract_fingerprint = (
            original_question_goal_contract_fingerprint(goal_contract)
        )
        return SimpleNamespace(
            accepted=True,
            code="GOAL_COMMITTED",
            stage=PopulationVerificationStage.GOAL_DECLARATION,
            transition=SimpleNamespace(
                state=SimpleNamespace(
                    goal_attestation=_goal_attestation(
                        contract_fingerprint
                    )
                )
            ),
        )

    def gate_id(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
    ) -> str:
        self.gate_id_run_authority = run_authority_fingerprint
        return "population-gate-captured"


class _PreReferenceCaptureGate:
    def __init__(self) -> None:
        self.run_authority_fingerprint = ""

    def build_pre_execution_reference(
        self,
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
        graph_receipt,
        node,
    ) -> PopulationPreExecutionReference:
        self.run_authority_fingerprint = run_authority_fingerprint
        return seal_population_pre_execution_reference(
            PopulationPreExecutionReference(
                gate_id="population-gate-pre-reference",
                context_owner_fingerprint=(
                    context_owner_fingerprint
                ),
                run_authority_fingerprint=(
                    run_authority_fingerprint
                ),
                goal_contract_fingerprint=(
                    goal_contract_fingerprint
                ),
                graph_receipt=graph_receipt,
                node=node,
            )
        )


def _goal_attestation(
    goal_contract_fingerprint: str,
) -> PopulationVerificationAttestation:
    candidate = PopulationVerificationAttestation(
        stage=PopulationVerificationStage.GOAL_DECLARATION,
        passed=True,
        gate_open=True,
        input_fingerprint="input-fingerprint",
        goal_contract_fingerprint=goal_contract_fingerprint,
        accepted_scopes=(),
    )
    return candidate.model_copy(
        update={
            "attestation_fingerprint": population_attestation_fingerprint(
                candidate
            )
        }
    )


def test_rule_only_goal_with_verified_empty_population_scope_finalizes_without_query_graph() -> None:
    question = "当前发布规则要求怎样处理？"
    rule_ref_id = "semantic:rules:policy:chunk:0001"
    contract = OriginalQuestionGoalContract(
        question=question,
        goals=[
            RuleQuestionGoal(
                goal_id="rule.policy",
                label="发布规则",
                rule_ref_ids=[rule_ref_id],
            )
        ],
    )
    contract_fingerprint = original_question_goal_contract_fingerprint(
        contract
    )
    runtime_session = GroundedRuntimeSession(
        session_id="queryless-rule-session",
        question=question,
        merchant_id="merchant-1",
    )
    runtime_session.verified_rule_ledger.append(
        GroundedVerifiedRuleArtifact(
            artifact_id="rule-artifact-1",
            question=question,
            goal_contract_fingerprint=contract_fingerprint,
            goal_ids=["rule.policy"],
            evidence_refs=[
                GroundedRuleEvidenceRef(
                    ref_id=rule_ref_id,
                    source_type="GOVERNED_RULE",
                    content_hash="content-fingerprint",
                    content="已发布的可验证规则内容。",
                )
            ],
            rule_context="已发布的可验证规则内容。",
            verification_passed=True,
            created_at="2026-07-19T00:00:00Z",
        )
    )
    session = GroundedDeepAgentSession(
        runtime=runtime_session,
        question_goal_contract=contract,
        population_goal_gate_id="population-gate-queryless",
        population_goal_gate_result={"accepted": True},
        population_goal_attestation=_goal_attestation(
            contract_fingerprint
        ),
    )
    gate = _QuerylessPopulationGate()
    runtime = GroundedDeepAgentRuntime.__new__(
        GroundedDeepAgentRuntime
    )
    runtime.population_gate_enforced = True
    runtime.population_execution_gate = gate

    coverage = runtime._require_complete_goal_coverage(session)

    assert coverage.finalization_allowed is True
    assert runtime_session.verified_query_ledger == []
    assert session.population_pre_execution_references == {}
    assert gate.completion_calls == 0


def test_goal_gate_run_authority_comes_from_server_workspace() -> None:
    question = "当前发布规则要求怎样处理？"
    contract = OriginalQuestionGoalContract(
        question=question,
        goals=[
            RuleQuestionGoal(
                goal_id="rule.policy",
                label="发布规则",
            )
        ],
    )
    workspace = GroundedContextWorkspace(
        root=Path("/workspace/run-authority"),
        artifacts_root=Path("/workspace/run-authority/artifacts"),
        staging_root=Path("/workspace/run-authority/staging"),
        core_scratch_root=Path("/workspace/run-authority/core"),
        subagents_root=Path("/workspace/run-authority/subagents"),
        thread_fingerprint="thread-fingerprint",
        run_fingerprint="run-fingerprint",
        owner_fingerprint="owner-fingerprint",
        request_fingerprint="server-request-fingerprint",
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="run-authority-session",
            question=question,
            merchant_id="merchant-1",
        ),
        context_workspace=workspace,
    )
    gate = _GoalCaptureGate()
    runtime = GroundedDeepAgentRuntime.__new__(
        GroundedDeepAgentRuntime
    )
    runtime.population_gate_enforced = True
    runtime.population_execution_gate = gate
    tools = {item.name: item for item in runtime._build_tools()}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=contract,
            runtime=SimpleNamespace(
                context=GroundedDeepAgentRunContext(
                    thread_id="thread-id",
                    run_id="run-id",
                    session=session,
                )
            ),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert gate.commit_run_authority == workspace.request_fingerprint
    assert gate.gate_id_run_authority == workspace.request_fingerprint
    assert session.population_goal_gate_id == "population-gate-captured"
    assert session.population_goal_attestation is not None


def test_pre_execution_run_authority_comes_from_server_workspace() -> None:
    question = "返回已定义指标。"
    contract = OriginalQuestionGoalContract(
        question=question,
        goals=[
            MetricQuestionGoal(
                goal_id="metric.primary",
                label="已定义指标",
            )
        ],
    )
    workspace = GroundedContextWorkspace(
        root=Path("/workspace/pre-reference"),
        artifacts_root=Path("/workspace/pre-reference/artifacts"),
        staging_root=Path("/workspace/pre-reference/staging"),
        core_scratch_root=Path("/workspace/pre-reference/core"),
        subagents_root=Path("/workspace/pre-reference/subagents"),
        thread_fingerprint="thread-fingerprint",
        run_fingerprint="run-fingerprint",
        owner_fingerprint="owner-fingerprint",
        request_fingerprint="server-request-fingerprint",
    )
    execution_session = GroundedRuntimeSession(
        session_id="pre-reference-session",
        question=question,
        merchant_id="merchant-1",
        active_generation=1,
        active_attempt_id="attempt-1",
        active_contract=GroundedQueryContract(
            question=question,
            topics=["governed-topic"],
            status="READY",
            query_shape="SCALAR",
        ),
        active_preparation=SimpleNamespace(
            plan=QueryPlan(
                intents=[
                    QuestionIntent(plan_task_id="query-node-1")
                ]
            )
        ),
        active_sql_validation=GroundedSqlValidationResult(
            valid=True,
            ast_fingerprint="sql-ast-fingerprint",
        ),
    )
    session = GroundedDeepAgentSession(
        runtime=execution_session,
        context_workspace=workspace,
        question_goal_contract=contract,
        active_goal_ids=["metric.primary"],
    )
    gate = _PreReferenceCaptureGate()
    runtime = GroundedDeepAgentRuntime.__new__(
        GroundedDeepAgentRuntime
    )
    runtime.population_gate_enforced = True
    runtime.population_execution_gate = gate

    kwargs = runtime._population_execution_kwargs(
        session,
        execution_session,
    )

    reference = kwargs["population_pre_execution_reference"]
    assert gate.run_authority_fingerprint == workspace.request_fingerprint
    assert (
        reference.run_authority_fingerprint
        == workspace.request_fingerprint
    )
    assert kwargs["population_query_node_id"] == "query-node-1"
