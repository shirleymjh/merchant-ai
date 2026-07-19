from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.config import Settings
from merchant_ai.models import QueryPlan, QuestionIntent
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    _authorized_verified_query_artifacts,
    _published_query_artifact_digests,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from tests.services.test_grounded_verified_artifact_publication import (
    _kernel_and_session,
)


class _AgentFactory:
    def __call__(self, **kwargs: Any) -> object:
        del kwargs
        return SimpleNamespace()


class _SemanticCatalog:
    @staticmethod
    def read(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "success": False,
            "error": "TEST_SEMANTIC_READ_NOT_AVAILABLE",
        }


class _PopulationPostGate:
    def __init__(
        self,
        session: GroundedDeepAgentSession,
        *,
        accepted: bool,
    ) -> None:
        self.session = session
        self.accepted = accepted
        self.post_calls = 0
        self.staged_artifact_ids: list[str] = []
        self.parent_ledger_sizes: list[int] = []

    @staticmethod
    def build_pre_execution_reference(
        *,
        context_owner_fingerprint: str,
        run_authority_fingerprint: str,
        goal_contract_fingerprint: str,
        graph_receipt: Any,
        node: Any,
    ) -> PopulationPreExecutionReference:
        return seal_population_pre_execution_reference(
            PopulationPreExecutionReference(
                gate_id="test-gate",
                context_owner_fingerprint=context_owner_fingerprint,
                run_authority_fingerprint=run_authority_fingerprint,
                goal_contract_fingerprint=goal_contract_fingerprint,
                graph_receipt=graph_receipt,
                node=node,
            )
        )

    def commit_node_post_result(
        self,
        *,
        reference: PopulationPreExecutionReference,
    ) -> object:
        del reference
        self.post_calls += 1
        self.staged_artifact_ids = sorted(
            self.session.population_staged_query_artifacts
        )
        self.parent_ledger_sizes.append(
            len(self.session.runtime.verified_query_ledger)
        )
        return SimpleNamespace(
            accepted=self.accepted,
            code=(
                "TEST_POST_ACCEPTED"
                if self.accepted
                else "TEST_POST_REJECTED"
            ),
            stage="POST_RESULT",
        )


def _online_context(
    tmp_path: Any,
    *,
    accepted: bool,
) -> tuple[
    GroundedDeepAgentRuntime,
    GroundedDeepAgentRunContext,
    _PopulationPostGate,
    Any,
]:
    kernel, runtime_session, executor = _kernel_and_session(
        verifier_passed=True
    )
    query_node_id = "node.primary"
    goal_id = "goal.primary"
    plan = QueryPlan(
        intents=[QuestionIntent(plan_task_id=query_node_id)]
    )
    runtime_session.active_plan = plan.model_copy(deep=True)
    runtime_session.active_preparation = SimpleNamespace(
        executable=True,
        plan=plan.model_copy(deep=True),
        asset_pack_fingerprint="semantic-activation-1",
    )
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace")
    )
    runtime = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=_SemanticCatalog(),
        settings=settings,
        agent_factory=_AgentFactory(),
        backend=object(),
    )
    goal_contract = OriginalQuestionGoalContract(
        question=runtime_session.question,
        goals=[
            MetricQuestionGoal(
                goal_id=goal_id,
                label="primary",
            )
        ],
    )
    deep_session = GroundedDeepAgentSession(
        runtime=runtime_session,
        question_goal_contract=goal_contract,
        active_goal_ids=[goal_id],
    )
    deep_session.context_workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-primary",
        run_id="run-primary",
        merchant_id=runtime_session.merchant_id,
        access_role=runtime_session.access_role,
        user_scope=runtime_session.user_scope,
        question=runtime_session.question,
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-primary",
        run_id="run-primary",
        session=deep_session,
    )
    gate = _PopulationPostGate(
        deep_session,
        accepted=accepted,
    )
    runtime.population_gate_enforced = True
    runtime.population_execution_gate = gate
    return runtime, context, gate, executor


def _execute_serial(
    runtime: GroundedDeepAgentRuntime,
    context: GroundedDeepAgentRunContext,
) -> dict[str, Any]:
    tool = next(
        item
        for item in runtime.tools
        if item.name == "execute_grounded_query"
    )
    return json.loads(
        tool.func(
            reason="execute active governed contract",
            runtime=SimpleNamespace(context=context),
        )
    )


def test_serial_population_post_rejection_leaves_zero_parent_authority(
    tmp_path: Any,
) -> None:
    runtime, context, gate, executor = _online_context(
        tmp_path,
        accepted=False,
    )

    result = _execute_serial(runtime, context)

    state = context.session.runtime
    assert result["status"] == "BLOCKED"
    assert result["code"] == "POPULATION_POST_RESULT_REJECTED"
    assert executor.publish_calls == 1
    assert gate.post_calls == 1
    assert len(gate.staged_artifact_ids) == 1
    assert gate.parent_ledger_sizes == [0]
    assert context.session.population_staged_query_artifacts == {}
    assert state.verified_query_ledger == []
    assert state.verified_entity_sets == []
    assert context.session.artifact_goal_ids == {}
    assert context.session.population_artifact_query_node_ids == {}
    assert _authorized_verified_query_artifacts(context.session) == []
    assert _published_query_artifact_digests(context.session) == {}
    assert runtime._verified_conversation_scope(context.session) == {}
    assert state.verified_evidence is None
    assert state.run_result is None
    assert state.publication_authority_run_result is None
    assert state.publication_authority_fingerprint == ""
    assert state.answer_artifact_ids == []
    assert all(
        item.status == "POST_AUTHORIZATION_REJECTED"
        and item.receipt == {}
        for item in state.pending_query_publications
    )


def test_serial_population_post_acceptance_adopts_exactly_once(
    tmp_path: Any,
) -> None:
    runtime, context, gate, executor = _online_context(
        tmp_path,
        accepted=True,
    )

    result = _execute_serial(runtime, context)

    state = context.session.runtime
    assert result["status"] == "VERIFIED"
    assert executor.publish_calls == 1
    assert gate.post_calls == 1
    assert len(gate.staged_artifact_ids) == 1
    assert gate.parent_ledger_sizes == [0]
    assert context.session.population_staged_query_artifacts == {}
    assert len(state.verified_query_ledger) == 1
    artifact = state.verified_query_ledger[0]
    assert artifact.artifact_id == gate.staged_artifact_ids[0]
    assert result["queryArtifactId"] == artifact.artifact_id
    assert result["populationPostGate"]["accepted"] is True
    assert context.session.artifact_goal_ids == {
        artifact.artifact_id: ["goal.primary"]
    }
    assert context.session.population_artifact_query_node_ids == {
        artifact.artifact_id: "node.primary"
    }
    assert [
        item.artifact_id
        for item in _authorized_verified_query_artifacts(context.session)
    ] == [artifact.artifact_id]
