from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    AgentRunResult,
    AnswerClaimVerification,
    QueryBundle,
    QueryPlan,
    ResolvedTimeRange,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_outcome_completion import (
    OutcomeCompletionDecision,
    OutcomeCompletionStatus,
    OutcomeEvidenceKind,
    UserOutcomeAssessment,
    outcome_attestation_matches,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    TimeWindowQuestionGoal,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRankingBinding,
)
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


class ScalarBindingGoalGateKernel(GoalGateKernel):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: dict[str, GroundedRuntimeAttempt] = {}

    def propose_contract(
        self,
        session: GroundedRuntimeSession,
        evidence: list[dict[str, Any]],
        hints: Any,
        **kwargs: Any,
    ) -> GroundedRuntimeAttempt:
        del kwargs
        self.propose_contract_calls += 1
        normalized_hints = (
            hints
            if isinstance(hints, GroundedBindingHints)
            else GroundedBindingHints.model_validate(hints)
        )
        attempt = GroundedRuntimeAttempt(
            attempt_id="scalar-binding-attempt",
            contract=GroundedQueryContract(
                question=session.question,
                status="READY",
                query_shape=(
                    "DETAIL" if normalized_hints.selected_fields else "SCALAR"
                ),
                binding_hints=normalized_hints,
                evidence_refs=[item["refId"] for item in evidence],
                time_range=ResolvedTimeRange(
                    days=7,
                    explicit=True,
                    source="last_n_days",
                ),
            ),
        )
        self.attempts[attempt.attempt_id] = attempt
        return attempt

    def activate_contract(
        self,
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        del session
        self.activate_contract_calls += 1
        attempt = self.attempts[attempt_id]
        attempt.activated = True
        attempt.activation_status = "ACTIVATED"
        attempt.active_generation = 1
        attempt.next_action = "EXECUTE_GROUNDED_QUERY"
        return attempt


class OutcomeCompletionKernel(GoalGateKernel):
    def compose_answer(
        self,
        session: GroundedRuntimeSession,
        *,
        allow_llm: bool,
    ) -> str:
        del allow_llm
        self.compose_answer_calls += 1
        session.answer = "Revenue is 120."
        session.answer_plan = QueryPlan()
        session.answer_run_result = AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[{"revenue": 120}],
                tables=["orders"],
            ),
            verified_evidence=VerifiedEvidence(passed=True),
            answer_claim_verification=AnswerClaimVerification(passed=True),
        )
        session.answer_verified_evidence = VerifiedEvidence(passed=True)
        session.answer_artifact_ids = [
            artifact.artifact_id for artifact in session.verified_query_ledger
        ]
        return session.answer


class CapturingOutcomeCompletionProvider:
    def __init__(self, decision: OutcomeCompletionDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, **kwargs: Any) -> OutcomeCompletionDecision:
        self.calls.append(dict(kwargs))
        return self.decision.model_copy(deep=True)


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

    assert finalized["status"] == "SKILL_INPUT_SNAPSHOT_READY"
    assert finalized["goalCoverage"]["coveredGoalIds"] == [
        "metric.revenue",
        "metric.orders",
    ]
    assert composed["status"] == "ANSWERED"
    assert composed["answer"] == "verified answer"
    assert composed["verifiedQueryArtifactIds"] == ["artifact-complete"]
    assert composed["goalAnswerCoverage"]["passed"] is True
    assert composed["goalAnswerCoverage"]["mappedGoalIds"] == [
        "metric.revenue",
        "metric.orders",
    ]
    assert context.session.data_collection_sealed is False
    assert kernel.verify_portfolio_calls == 1
    assert kernel.compose_answer_calls == 1


def test_ranked_compose_uses_internal_artifact_renderer_not_core_supplied_spans() -> None:
    kernel = GoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel, question="top 3 products")
    tools = _tools(runtime)
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.sales",
                        label="sales",
                        required=False,
                    ),
                    RankingQuestionGoal(
                        goal_id="ranking.top3",
                        label="top 3 products",
                        metric_goal_ids=["metric.sales"],
                        limit=3,
                        population_scope="ALL_MATCHING_ROWS",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"
    contract = GroundedQueryContract(
        question=context.session.runtime.question,
        status="READY",
        query_shape="RANKED",
        evidence_refs=["semantic:metric:sales"],
        metrics=[
            GroundedMetricBinding(
                requested_phrase="sales",
                semantic_ref_id="semantic:metric:sales",
                topic="orders",
                table="orders",
                metric_key="sales",
            )
        ],
        ranking=GroundedRankingBinding(
            enabled=True,
            direction="DESC",
            limit=3,
            metric_ref_id="semantic:metric:sales",
        ),
    )
    artifact = GroundedVerifiedQueryArtifact(
        artifact_id="artifact-ranked",
        generation=1,
        contract_fingerprint=grounded_query_contract_fingerprint(contract),
        sql_fingerprint="a" * 64,
        contract=contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[
                    {"product": "Product A", "sales": 30},
                    {"product": "Product B", "sales": 20},
                    {"product": "Product C", "sales": 10},
                ],
                tables=["orders"],
                result_coverage="TOP_N",
            )
        ),
        verified_evidence=VerifiedEvidence(passed=True),
        ranking_semantics_verified=True,
        output_columns=["product", "sales"],
    )
    context.session.runtime.verified_query_ledger.append(artifact)
    context.session.artifact_goal_ids[artifact.artifact_id] = [
        "metric.sales",
        "ranking.top3",
    ]

    composed = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert composed["status"] == "ANSWERED"
    assert "Product A" in composed["answer"]
    assert "| product | sales |" in composed["answer"]
    assert composed["goalAnswerCoverage"]["passed"] is True
    ranking_binding = next(
        item
        for item in composed["goalAnswerCoverage"]["bindings"]
        if item["goalId"] == "ranking.top3"
    )
    assert ranking_binding["renderer"] == "VERIFIED_RANKING_RENDERER"
    schema = tools["compose_verified_answer"].tool_call_schema.model_json_schema()
    assert "goal_answer_bindings" not in schema.get("properties", {})
    assert "goalAnswerBindings" not in schema.get("properties", {})


def test_goal_contract_is_once_only_and_idempotent() -> None:
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

    replayed = json.loads(
        tools["declare_original_question_goals"].func(
            contract=initial,
            runtime=SimpleNamespace(context=context),
        )
    )
    assert replayed["status"] == "ALREADY_DECLARED"
    assert replayed["contractFingerprint"] == first["contractFingerprint"]

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
    assert redeclared["code"] == "GOAL_CONTRACT_ALREADY_COMMITTED"
    assert len(redeclared["contractFingerprint"]) == 64
    assert list(context.session.question_goal_contract.goal_map()) == [
        "metric.revenue"
    ]


def test_complex_compose_uses_outcome_completion_instead_of_hard_goal_gate() -> None:
    decision = OutcomeCompletionDecision(
        overall_status=OutcomeCompletionStatus.PARTIAL,
        outcomes=[
            UserOutcomeAssessment(
                outcome_id="revenue",
                requirement="return revenue",
                status=OutcomeCompletionStatus.SATISFIED,
                evidence_kind=OutcomeEvidenceKind.DATA,
                query_artifact_ids=["artifact-revenue"],
                evidence_refs=["semantic:table:orders"],
            ),
            UserOutcomeAssessment(
                outcome_id="orders",
                requirement="return orders",
                status=OutcomeCompletionStatus.INSUFFICIENT_EVIDENCE,
                evidence_kind=OutcomeEvidenceKind.DATA,
                missing_reason="orders are not yet supported by verified evidence",
            ),
        ],
        missing_requirements=[
            "orders are not yet supported by verified evidence"
        ],
    )
    provider = CapturingOutcomeCompletionProvider(decision)
    kernel = OutcomeCompletionKernel()
    runtime = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=object(),
        outcome_completion_provider=provider,
        agent_factory=CapturingAgentFactory(),
        backend=object(),
    )
    context = _context(kernel)
    tools = _tools(runtime)
    _declare_metric_goals(tools, context)
    artifact = _verified_artifact()
    context.session.runtime.verified_query_ledger.append(artifact)
    context.session.artifact_goal_ids[artifact.artifact_id] = [
        "metric.revenue"
    ]

    blocked = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert blocked["status"] == "OUTCOME_COMPLETION_INCOMPLETE"
    assert blocked["missingRequirements"] == [
        "orders are not yet supported by verified evidence"
    ]
    assert kernel.compose_answer_calls == 1
    assert len(provider.calls) == 1
    assert context.session.runtime.answer == ""

    accepted = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            accept_partial=True,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert accepted["status"] == "ANSWERED"
    assert accepted["partial"] is True
    assert "### 未完成项" in accepted["answer"]
    assert outcome_attestation_matches(
        accepted["answer"], context.session.outcome_completion_result
    )


def test_simple_scalar_compose_does_not_call_outcome_evaluator() -> None:
    provider = CapturingOutcomeCompletionProvider(
        OutcomeCompletionDecision(
            overall_status=OutcomeCompletionStatus.SATISFIED,
            outcomes=[],
        )
    )
    kernel = OutcomeCompletionKernel()
    runtime = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=object(),
        outcome_completion_provider=provider,
        agent_factory=CapturingAgentFactory(),
        backend=object(),
    )
    context = _context(kernel, question="return revenue")
    tools = _tools(runtime)
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.revenue",
                        label="revenue",
                    )
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"
    artifact = _verified_artifact(question=context.session.runtime.question)
    context.session.runtime.verified_query_ledger.append(artifact)
    context.session.artifact_goal_ids[artifact.artifact_id] = [
        "metric.revenue"
    ]

    composed = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert composed["status"] == "ANSWERED"
    assert provider.calls == []
