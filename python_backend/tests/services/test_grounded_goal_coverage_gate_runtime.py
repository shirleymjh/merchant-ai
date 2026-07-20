from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    AgentRunResult,
    QueryBundle,
    QueryPlan,
    ResolvedTimeRange,
    VerifiedEvidence,
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


def _propose_scalar_binding_contract(
    *,
    metric_goal_refs: list[str],
    binding_hints: dict[str, Any],
) -> tuple[dict[str, Any], ScalarBindingGoalGateKernel]:
    kernel = ScalarBindingGoalGateKernel()
    runtime = _runtime(kernel)
    context = _context(kernel, question="最近7天查询指标")
    tools = _tools(runtime)
    table_ref = "semantic:经营画像:merchant_profile_daily:detail"
    hint_refs = [
        *list(binding_hints.get("metricRefs") or []),
        *[
            str(item.get("fieldRef") or "")
            for item in (binding_hints.get("fieldAggregations") or [])
        ],
        *[
            str(item.get("fieldRef") or "")
            for item in (binding_hints.get("selectedFields") or [])
        ],
    ]
    read_ref_ids = list(
        dict.fromkeys(
            [table_ref, *metric_goal_refs, *hint_refs]
        )
    )
    context.session.core_semantic_evidence = [
        {
            "refId": ref_id,
            "kind": (
                "TABLE_DETAIL"
                if ref_id == table_ref
                else "METRIC"
                if ":metric:" in ref_id
                else "COLUMN"
            ),
            "topic": "经营画像",
            "table": "merchant_profile_daily",
            "contentSnippet": "{}",
            "contentHash": "hash",
        }
        for ref_id in read_ref_ids
    ]
    metric_goal_ids = [
        "metric.output_%s" % (index + 1)
        for index in range(len(metric_goal_refs))
    ]
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    *[
                        MetricQuestionGoal(
                            goal_id=goal_id,
                            label="指标%s" % (index + 1),
                            metric_ref_id=metric_goal_refs[index],
                        )
                        for index, goal_id in enumerate(metric_goal_ids)
                    ],
                    TimeWindowQuestionGoal(
                        goal_id="time.recent_7_days",
                        label="最近7天",
                        time_expression="最近7天",
                        days=7,
                        applies_to_goal_ids=metric_goal_ids,
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"
    result = json.loads(
        tools["propose_grounded_contract"].func(
            read_ref_ids=read_ref_ids,
            binding_hints={
                "tableRefs": [table_ref],
                "timeExpression": "最近7天",
                **binding_hints,
            },
            goal_ids=[*metric_goal_ids, "time.recent_7_days"],
            runtime=SimpleNamespace(context=context),
        )
    )
    return result, kernel


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
    assert composed["status"] == "ANSWERED"
    assert composed["answer"] == "verified answer"
    assert composed["verifiedQueryArtifactIds"] == ["artifact-complete"]
    assert composed["goalAnswerCoverage"]["passed"] is True
    assert composed["goalAnswerCoverage"]["mappedGoalIds"] == [
        "metric.revenue",
        "metric.orders",
    ]
    assert context.session.data_collection_sealed is True
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


def test_scalar_metric_proposal_rejects_unrequested_selected_field() -> None:
    metric_ref = "semantic:经营画像:merchant_profile_daily:metric:order_cnt_1d"
    source_field_ref = (
        "semantic:经营画像:merchant_profile_daily:field:order_cnt_1d"
    )

    proposed, kernel = _propose_scalar_binding_contract(
        metric_goal_refs=[metric_ref],
        binding_hints={
            "metricRefs": [metric_ref],
            "selectedFields": [
                {
                    "fieldRef": source_field_ref,
                    "outputAlias": "order_cnt_1d_source",
                }
            ],
        },
    )

    assert proposed["status"] == "BLOCKED"
    assert proposed["code"] == "QUERY_GOAL_ASSIGNMENT_MISMATCH"
    assert (
        proposed["nextAction"]
        == "REMOVE_EXTRA_OUTPUT_BINDINGS_AND_RESUBMIT"
    )
    assert proposed["issues"] == [
        {
            "code": "SCALAR_METRIC_EXTRA_OUTPUT_NOT_REQUESTED",
            "goalIds": ["metric.output_1", "time.recent_7_days"],
            "assignedMetricGoalIds": ["metric.output_1"],
            "unexpectedSelectedFieldRefs": [source_field_ref],
            "metricRefs": [metric_ref],
            "fieldAggregationRefs": [],
            "mixedMetricBindingModes": False,
            "expectedMetricOutputCount": 1,
            "submittedMetricOutputCount": 1,
            "nextAction": "REMOVE_EXTRA_OUTPUT_BINDINGS_AND_RESUBMIT",
            "instruction": (
                "A scalar metric branch may bind only one metric output per "
                "assigned METRIC goal plus necessary time semantics. Remove "
                "selectedFields; when a published metricRef already covers the "
                "goal, do not also project or aggregate its source field."
            ),
        }
    ]
    assert kernel.activate_contract_calls == 0


def test_scalar_metric_proposal_rejects_mixed_published_and_field_aggregation() -> None:
    published_ref = (
        "semantic:经营画像:merchant_profile_daily:metric:order_cnt_1d"
    )
    field_ref = "semantic:经营画像:merchant_profile_daily:field:buyer_id"

    proposed, kernel = _propose_scalar_binding_contract(
        metric_goal_refs=[published_ref, field_ref],
        binding_hints={
            "metricRefs": [published_ref],
            "fieldAggregations": [
                {
                    "fieldRef": field_ref,
                    "aggregation": "COUNT_DISTINCT",
                    "requestedPhrase": "买家数",
                }
            ],
        },
    )

    issue = proposed["issues"][0]
    assert proposed["code"] == "QUERY_GOAL_ASSIGNMENT_MISMATCH"
    assert issue["code"] == "SCALAR_METRIC_EXTRA_OUTPUT_NOT_REQUESTED"
    assert issue["mixedMetricBindingModes"] is True
    assert issue["expectedMetricOutputCount"] == 2
    assert issue["submittedMetricOutputCount"] == 2
    assert issue["metricRefs"] == [published_ref]
    assert issue["fieldAggregationRefs"] == [field_ref]
    assert issue["nextAction"] == "REMOVE_EXTRA_OUTPUT_BINDINGS_AND_RESUBMIT"
    assert kernel.activate_contract_calls == 0


def test_scalar_metric_proposal_rejects_more_metric_outputs_than_goals() -> None:
    requested_ref = (
        "semantic:经营画像:merchant_profile_daily:metric:order_cnt_1d"
    )
    extra_ref = (
        "semantic:经营画像:merchant_profile_daily:metric:refund_order_cnt_1d"
    )

    proposed, kernel = _propose_scalar_binding_contract(
        metric_goal_refs=[requested_ref],
        binding_hints={"metricRefs": [requested_ref, extra_ref]},
    )

    issue = proposed["issues"][0]
    assert issue["code"] == "SCALAR_METRIC_EXTRA_OUTPUT_NOT_REQUESTED"
    assert issue["mixedMetricBindingModes"] is False
    assert issue["expectedMetricOutputCount"] == 1
    assert issue["submittedMetricOutputCount"] == 2
    assert issue["metricRefs"] == [requested_ref, extra_ref]
    assert kernel.activate_contract_calls == 0


def test_scalar_metric_proposal_allows_one_field_aggregation_for_one_goal() -> None:
    field_ref = "semantic:经营画像:merchant_profile_daily:field:buyer_id"

    proposed, kernel = _propose_scalar_binding_contract(
        metric_goal_refs=[field_ref],
        binding_hints={
            "fieldAggregations": [
                {
                    "fieldRef": field_ref,
                    "aggregation": "COUNT_DISTINCT",
                    "requestedPhrase": "买家数",
                }
            ]
        },
    )

    assert proposed["status"] == "READY"
    assert proposed["activated"] is True
    assert proposed["assignedGoalIds"] == [
        "metric.output_1",
        "time.recent_7_days",
    ]
    assert kernel.activate_contract_calls == 1


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
