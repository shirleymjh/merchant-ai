from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import AgentRunResult, QueryBundle, QueryPlan, VerifiedEvidence
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeAttempt,
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


class GoalGateKernel:
    def __init__(self) -> None:
        self.verify_portfolio_calls = 0
        self.compose_answer_calls = 0
        self.propose_contract_calls = 0
        self.activate_contract_calls = 0

    @staticmethod
    def new_session(question: str, merchant_id: str) -> GroundedRuntimeSession:
        return GroundedRuntimeSession(
            session_id="goal-gate-session",
            question=question,
            merchant_id=merchant_id,
        )

    def verify_portfolio(
        self,
        session: GroundedRuntimeSession,
    ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
        self.verify_portfolio_calls += 1
        return (
            QueryPlan(),
            AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"revenue": 120, "orders": 8}],
                    tables=["orders"],
                )
            ),
            VerifiedEvidence(passed=True),
            [artifact.artifact_id for artifact in session.verified_query_ledger],
        )

    def compose_answer(
        self,
        session: GroundedRuntimeSession,
        *,
        allow_llm: bool,
    ) -> str:
        self.compose_answer_calls += 1
        session.answer = "verified answer"
        session.answer_artifact_ids = [
            artifact.artifact_id for artifact in session.verified_query_ledger
        ]
        return session.answer

    def propose_contract(
        self,
        session: GroundedRuntimeSession,
        evidence: list[dict[str, Any]],
        hints: Any,
        **kwargs: Any,
    ) -> GroundedRuntimeAttempt:
        self.propose_contract_calls += 1
        return GroundedRuntimeAttempt(
            attempt_id="semantic-mismatch-attempt",
            contract=GroundedQueryContract(
                question=session.question,
                status="READY",
                query_shape="SCALAR",
                evidence_refs=[item["refId"] for item in evidence],
            ),
        )

    def activate_contract(
        self,
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        self.activate_contract_calls += 1
        raise AssertionError("a semantically mismatched Contract must not be activated")


def _runtime(kernel: GoalGateKernel) -> GroundedDeepAgentRuntime:
    return GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=object(),
        agent_factory=CapturingAgentFactory(),
        backend=object(),
    )


def _context(
    kernel: GoalGateKernel,
    *,
    question: str = "return revenue and orders",
) -> GroundedDeepAgentRunContext:
    return GroundedDeepAgentRunContext(
        thread_id="goal-gate-thread",
        run_id="goal-gate-run",
        session=GroundedDeepAgentSession(
            runtime=kernel.new_session(question, "merchant-1")
        ),
    )


def _tools(runtime: GroundedDeepAgentRuntime) -> dict[str, Any]:
    return {item.name: item for item in runtime.tools}


def _declare_metric_goals(
    tools: dict[str, Any],
    context: GroundedDeepAgentRunContext,
) -> None:
    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.revenue",
                        label="revenue",
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="orders",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert result["status"] == "ACCEPTED"


def _verified_artifact(
    *,
    artifact_id: str = "artifact-revenue",
    question: str = "return revenue and orders",
) -> GroundedVerifiedQueryArtifact:
    contract = GroundedQueryContract(
        question=question,
        status="READY",
        query_shape="SCALAR",
        evidence_refs=["semantic:table:orders"],
    )
    return GroundedVerifiedQueryArtifact(
        artifact_id=artifact_id,
        generation=1,
        contract_fingerprint=grounded_query_contract_fingerprint(contract),
        sql_fingerprint="f" * 64,
        contract=contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[{"revenue": 120}],
                tables=["orders"],
            )
        ),
        verified_evidence=VerifiedEvidence(passed=True),
    )


def test_missing_required_goal_blocks_both_finalization_and_answer_composition() -> None:
    kernel = GoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel)
    tools = _tools(runtime)
    _declare_metric_goals(tools, context)
    artifact = _verified_artifact()
    context.session.runtime.verified_query_ledger.append(artifact)
    context.session.artifact_goal_ids[artifact.artifact_id] = ["metric.revenue"]

    finalized = json.loads(
        tools["finalize_evidence_collection"].func(
            reason="all data collected",
            runtime=SimpleNamespace(context=context),
        )
    )
    composed = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert finalized["status"] == "EVIDENCE_INCOMPLETE"
    assert finalized["code"] == "ORIGINAL_QUESTION_GOALS_UNCOVERED"
    assert finalized["missingRequiredGoalIds"] == ["metric.orders"]
    assert composed["status"] == "GOAL_COVERAGE_INCOMPLETE"
    assert composed["code"] == "ORIGINAL_QUESTION_GOALS_UNCOVERED"
    assert composed["missingRequiredGoalIds"] == ["metric.orders"]
    assert context.session.data_collection_sealed is False
    assert context.session.goal_coverage_result["finalizationAllowed"] is False
    assert kernel.verify_portfolio_calls == 0
    assert kernel.compose_answer_calls == 0


def test_complete_verified_goal_coverage_allows_finalization_and_composition() -> None:
    kernel = GoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel)
    tools = _tools(runtime)
    _declare_metric_goals(tools, context)
    artifact = _verified_artifact(artifact_id="artifact-complete")
    context.session.runtime.verified_query_ledger.append(artifact)
    context.session.artifact_goal_ids[artifact.artifact_id] = [
        "metric.revenue",
        "metric.orders",
    ]

    finalized = json.loads(
        tools["finalize_evidence_collection"].func(
            reason="all data collected",
            runtime=SimpleNamespace(context=context),
        )
    )
    composed = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert finalized["status"] == "EVIDENCE_COLLECTION_SEALED"
    assert finalized["goalCoverage"]["coveredGoalIds"] == [
        "metric.revenue",
        "metric.orders",
    ]
    assert composed == {
        "status": "ANSWERED",
        "answer": "verified answer",
        "verifiedQueryArtifactIds": ["artifact-complete"],
    }
    assert context.session.data_collection_sealed is True
    assert kernel.verify_portfolio_calls == 1
    assert kernel.compose_answer_calls == 1


def test_proposal_is_blocked_when_assigned_goal_semantics_are_not_in_contract() -> None:
    kernel = GoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel, question="return revenue")
    tools = _tools(runtime)
    semantic_table_ref = "semantic:table:orders"
    semantic_metric_ref = "semantic:metric:revenue"
    context.session.core_semantic_evidence = [
        {
            "refId": semantic_table_ref,
            "kind": "TABLE_DETAIL",
            "topic": "orders",
            "table": "orders",
            "contentSnippet": "{}",
            "contentHash": "hash",
        }
    ]
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.revenue",
                        label="revenue",
                        metric_ref_id=semantic_metric_ref,
                    )
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    proposed = json.loads(
        tools["propose_grounded_contract"].func(
            read_ref_ids=[semantic_table_ref],
            binding_hints={"tableRefs": [semantic_table_ref]},
            goal_ids=["metric.revenue"],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert proposed["status"] == "BLOCKED"
    assert proposed["code"] == "QUERY_GOAL_ASSIGNMENT_MISMATCH"
    assert proposed["issues"] == [
        {
            "code": "QUERY_GOAL_SEMANTIC_REF_MISMATCH",
            "goalId": "metric.revenue",
            "declaredSemanticRefIds": [semantic_metric_ref],
            "contractEvidenceRefs": [semantic_table_ref],
        }
    ]
    assert kernel.propose_contract_calls == 1
    assert kernel.activate_contract_calls == 0
    assert context.session.active_goal_ids == []


def test_goal_contract_cannot_change_after_query_start() -> None:
    kernel = GoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel, question="return revenue")
    tools = _tools(runtime)
    initial = OriginalQuestionGoalContract(
        question=context.session.runtime.question,
        goals=[
            MetricQuestionGoal(
                goal_id="metric.revenue",
                label="revenue",
            )
        ],
    )
    first = json.loads(
        tools["declare_original_question_goals"].func(
            contract=initial,
            runtime=SimpleNamespace(context=context),
        )
    )
    assert first["status"] == "ACCEPTED"
    context.session.runtime.attempts.append(
        GroundedRuntimeAttempt(
            attempt_id="query-started",
            contract=GroundedQueryContract(
                question=context.session.runtime.question,
                status="READY",
                query_shape="SCALAR",
            ),
        )
    )

    redeclared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    *initial.goals,
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="orders",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert redeclared["status"] == "REJECTED"
    assert redeclared["code"] == "GOAL_CONTRACT_IMMUTABLE_AFTER_QUERY_START"
    assert len(redeclared["contractFingerprint"]) == 64
    assert list(context.session.question_goal_contract.goal_map()) == [
        "metric.revenue"
    ]
