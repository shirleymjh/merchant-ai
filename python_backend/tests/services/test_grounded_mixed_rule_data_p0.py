from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    AgentRunResult,
    QueryBundle,
    QueryPlan,
    RecallBundle,
    RecallItem,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    _phase_visible_tools,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    parse_original_question_goal_contract,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_rule_artifact import (
    GroundedVerifiedRuleArtifact,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeSession,
    GroundedVerifiedQueryArtifact,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


RULE_REF = "semantic:rules:published:chunk:0001"
FIELD_REF = "semantic:topic:records:field:record_id"


class _Factory:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace()


class _Kernel:
    def __init__(self) -> None:
        self.compose_calls = 0

    def publish_rule_artifact(
        self,
        session: GroundedRuntimeSession,
        artifact: GroundedVerifiedRuleArtifact,
    ) -> GroundedVerifiedRuleArtifact:
        session.verified_rule_ledger.append(artifact.model_copy(deep=True))
        return artifact.model_copy(deep=True)

    def verify_portfolio(
        self,
        session: GroundedRuntimeSession,
    ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
        return (
            QueryPlan(),
            AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"record_id": "record-1"}],
                    tables=["records"],
                    result_coverage="ALL_ROWS",
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
        del allow_llm
        self.compose_calls += 1
        session.answer = "已完成数据查询。"
        session.answer_artifact_ids = [
            artifact.artifact_id for artifact in session.verified_query_ledger
        ]
        return session.answer

    def compose_rule_answer(self, session: GroundedRuntimeSession) -> str:
        session.answer = "纯规则回答"
        return session.answer


def _runtime() -> tuple[GroundedDeepAgentRuntime, _Kernel]:
    kernel = _Kernel()
    runtime = GroundedDeepAgentRuntime(
        kernel,  # type: ignore[arg-type]
        lead_model=object(),
        semantic_catalog=object(),
        agent_factory=_Factory(),
        backend=object(),
    )
    return runtime, kernel


def _goal_contract(question: str) -> OriginalQuestionGoalContract:
    return parse_original_question_goal_contract(
        {
            "question": question,
            "goals": [
                {
                    "goalId": "rule.published",
                    "kind": "rule",
                    "label": "已发布规则",
                    "ruleRefIds": [RULE_REF],
                },
                {
                    "goalId": "time.recent",
                    "kind": "time_window",
                    "label": "用户指定时间范围",
                    "timeExpression": "用户指定的最近时间范围",
                    "appliesToGoalIds": ["detail.records"],
                },
                {
                    "goalId": "detail.records",
                    "kind": "detail",
                    "label": "数据明细",
                    "requiredFieldRefIds": [FIELD_REF],
                    "inputGoalIds": ["time.recent"],
                },
            ],
        }
    )


def _context(
    question: str = "说明已发布规则，并返回用户指定时间范围内的数据明细",
) -> GroundedDeepAgentRunContext:
    rule = RecallItem(
        doc_id=RULE_REF,
        title="已发布规则",
        content="仅依据当前已发布的正式规则执行。",
        source_type="GOVERNED_RULE",
        metadata={
            "status": "PUBLISHED",
            "visibilityPolicy": {"level": "merchant"},
        },
    )
    return GroundedDeepAgentRunContext(
        thread_id="mixed-thread",
        run_id="mixed-run",
        session=GroundedDeepAgentSession(
            runtime=GroundedRuntimeSession(
                session_id="mixed-session",
                question=question,
                merchant_id="merchant-1",
                recall=RecallBundle(items=[rule]),
            )
        ),
    )


def _declare_and_publish_rule(
    runtime: GroundedDeepAgentRuntime,
    context: GroundedDeepAgentRunContext,
) -> dict[str, Any]:
    tools = {item.name: item for item in runtime.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=_goal_contract(context.session.runtime.question),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"
    return json.loads(
        tools["publish_verified_rule_evidence"].func(
            goal_ids=["rule.published"],
            rule_ref_ids=[RULE_REF],
            runtime=SimpleNamespace(context=context),
        )
    )


def _query_artifact(question: str) -> GroundedVerifiedQueryArtifact:
    contract = GroundedQueryContract(
        question=question,
        status="READY",
        query_shape="DETAIL",
        evidence_refs=[FIELD_REF],
    )
    return GroundedVerifiedQueryArtifact(
        artifact_id="query-detail",
        generation=1,
        contract_fingerprint=grounded_query_contract_fingerprint(contract),
        sql_fingerprint="a" * 64,
        contract=contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[{"record_id": "record-1"}],
                tables=["records"],
                result_coverage="ALL_ROWS",
                original_row_count=1,
            )
        ),
        verified_evidence=VerifiedEvidence(passed=True),
        output_columns=["record_id"],
        output_semantic_refs={"record_id": FIELD_REF},
    )


def test_published_rule_does_not_close_data_tools_for_mixed_goals() -> None:
    runtime, _ = _runtime()
    context = _context()
    published = _declare_and_publish_rule(runtime, context)

    visible, _ = _phase_visible_tools(context.session, runtime.tools)
    visible_names = {item.name for item in visible}

    assert published["status"] == "RULE_EVIDENCE_VERIFIED"
    assert published["nextAction"] == "CONTINUE_GROUNDED_DATA_COLLECTION"
    assert published["remainingRequiredGoalIds"] == [
        "time.recent",
        "detail.records",
    ]
    assert "propose_grounded_execution_graph" in visible_names
    assert "query_data" in visible_names
    assert "query_batch" in visible_names
    assert "compose_verified_rule_answer" not in visible_names


def test_mixed_answer_fails_closed_when_rule_or_data_is_missing() -> None:
    runtime, kernel = _runtime()
    tools = {item.name: item for item in runtime.tools}

    missing_data = _context()
    _declare_and_publish_rule(runtime, missing_data)
    data_result = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=missing_data),
        )
    )

    missing_rule = _context()
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=_goal_contract(missing_rule.session.runtime.question),
            runtime=SimpleNamespace(context=missing_rule),
        )
    )
    assert declared["status"] == "ACCEPTED"
    query_artifact = _query_artifact(missing_rule.session.runtime.question)
    missing_rule.session.runtime.verified_query_ledger.append(query_artifact)
    missing_rule.session.artifact_goal_ids[query_artifact.artifact_id] = [
        "time.recent",
        "detail.records",
    ]
    rule_result = json.loads(
        tools["compose_verified_answer"].func(
            allow_llm=False,
            runtime=SimpleNamespace(context=missing_rule),
        )
    )

    assert data_result["status"] == "GOAL_COVERAGE_INCOMPLETE"
    assert data_result["missingRequiredGoalIds"] == [
        "time.recent",
        "detail.records",
    ]
    assert rule_result["status"] == "GOAL_COVERAGE_INCOMPLETE"
    assert rule_result["missingRequiredGoalIds"] == ["rule.published"]
    assert kernel.compose_calls == 0


def test_mixed_finalizer_binds_rule_and_query_artifacts_once() -> None:
    runtime, kernel = _runtime()
    context = _context()
    tools = {item.name: item for item in runtime.tools}
    published = _declare_and_publish_rule(runtime, context)
    query_artifact = _query_artifact(context.session.runtime.question)
    context.session.runtime.verified_query_ledger.append(query_artifact)
    context.session.artifact_goal_ids[query_artifact.artifact_id] = [
        "time.recent",
        "detail.records",
    ]

    finalized = json.loads(
        tools["finalize_evidence_collection"].func(
            reason="all required evidence is verified",
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
    assert composed["status"] == "ANSWERED"
    assert composed["verifiedRuleArtifactIds"] == [published["artifactId"]]
    assert composed["verifiedQueryArtifactIds"] == [query_artifact.artifact_id]
    assert "### 规则依据" in composed["answer"]
    assert "仅依据当前已发布的正式规则执行" in composed["answer"]
    assert "record-1" in composed["answer"]
    bindings = {
        item["goalId"]: item
        for item in composed["goalAnswerCoverage"]["bindings"]
    }
    assert set(bindings) == {
        "rule.published",
        "time.recent",
        "detail.records",
    }
    assert bindings["rule.published"]["renderer"] == (
        "VERIFIED_RULE_ARTIFACT_RENDERER"
    )
    assert bindings["detail.records"]["renderer"] == (
        "VERIFIED_DETAIL_RENDERER"
    )
    assert composed["goalAnswerCoverage"]["passed"] is True
    assert kernel.compose_calls == 1
