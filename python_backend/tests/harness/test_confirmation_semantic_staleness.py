from merchant_ai.config import get_settings
from merchant_ai.graph.workflow import create_workflow
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    ChatContext,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    SkillMatchState,
    VerifiedEvidence,
)


def test_confirmation_resume_rejects_stale_semantic_assets(monkeypatch, tmp_path):
    workflow = create_workflow(get_settings().model_copy(update={"harness_workspace_path": str(tmp_path)}))
    source = workflow._initial_state("placeholder question", "tenant_1", ChatContext(), None, "thread_1", "run_1")
    source["planning_asset_pack"] = PlanningAssetPack(
        tables=[PlanningAssetEntry(table="table_alpha", topic="topic_alpha")]
    )
    source["plan"] = QueryPlan(
        intents=[
            QuestionIntent(
                question="placeholder question",
                intent_type="VALID",
                answer_mode="METRIC",
                plan_task_id="task_alpha",
                preferred_table="table_alpha",
            )
        ]
    )
    source["agent_run_result"] = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="task_alpha",
                success=True,
                query_bundle=QueryBundle(rows=[{"metric_alpha": 1}]),
            )
        ],
        verified_evidence=VerifiedEvidence(passed=True),
    )
    source["skill_match"] = SkillMatchState()
    monkeypatch.setattr(workflow.recall_service.topic_assets, "semantic_source_hash", lambda _topics: "hash_v1")

    workflow.persist_confirmation_evidence(source)

    resumed_context = ChatContext(
        pending_question="placeholder question",
        confirmation_token=source["confirmation_token"],
        confirmation_run_id="run_1",
    )
    resumed = workflow._initial_state("confirm", "tenant_1", resumed_context, None, "thread_1", "run_2")
    monkeypatch.setattr(workflow.recall_service.topic_assets, "semantic_source_hash", lambda _topics: "hash_v2")

    assert workflow.restore_confirmation_evidence(resumed) is False
    assert resumed["confirmation_restore_status"]["code"] == "CONFIRMATION_SEMANTIC_VERSION_CHANGED"
    assert resumed["confirmation_evidence_reused"] is False
    assert resumed["plan"].intents == []
    workflow.checkpoint_manager.close()
