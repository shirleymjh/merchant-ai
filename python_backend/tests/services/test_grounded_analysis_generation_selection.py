from merchant_ai.services.grounded_analysis_artifact import (
    GroundedDerivedAnalysisArtifact,
    GroundedDerivedAnalysisVerification,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentSession,
    _latest_verified_analysis_artifacts,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession
from merchant_ai.services.grounded_subagent_runtime import (
    GroundedVerifiedSkillArtifact,
)


def _analysis(artifact_id: str) -> GroundedDerivedAnalysisArtifact:
    return GroundedDerivedAnalysisArtifact(
        artifact_id=artifact_id,
        goal_contract_fingerprint="goal-contract",
        analysis_goal_id="analysis.main",
        analysis_type="ANOMALY",
        publication_status="PROVED",
        result_ref="result:%s" % artifact_id,
        result={"artifact": artifact_id},
        verified_evidence=GroundedDerivedAnalysisVerification(
            publication_status="PROVED",
            evidence_refs=["result:%s" % artifact_id],
        ),
    )


def _skill(
    artifact_id: str,
    *,
    generation: int,
) -> GroundedVerifiedSkillArtifact:
    return GroundedVerifiedSkillArtifact(
        artifact_id="skill-%s" % generation,
        skill_name="analysis",
        skill_run_id="run-%s" % generation,
        sub_goal_id="analysis-subgoal",
        parent_goal_ids=["analysis.main"],
        generation=generation,
        skill_contract_fingerprint="contract-%s" % generation,
        skill_definition_sha256="definition",
        input_artifact_ids=["query-input"],
        input_artifact_fingerprints={"query-input": "fingerprint"},
        derived_analysis_artifact_ids=[artifact_id],
        structured_output_fingerprint="output-%s" % generation,
    ).with_ledger_fingerprint()


def test_only_latest_successful_analysis_generation_is_selected() -> None:
    old = _analysis("analysis-old")
    latest = _analysis("analysis-latest")
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="session",
            question="分析异常",
            merchant_id="merchant",
        ),
        verified_analysis_ledger=[old, latest],
        verified_skill_ledger=[
            _skill(old.artifact_id, generation=1),
            _skill(latest.artifact_id, generation=2),
        ],
    )

    selected = _latest_verified_analysis_artifacts(session)

    assert [item.artifact_id for item in selected] == [
        "analysis-latest"
    ]


def test_analysis_linked_only_to_invalid_skill_receipt_is_not_selected() -> None:
    artifact = _analysis("analysis-untrusted")
    invalid = _skill(artifact.artifact_id, generation=1).model_copy(
        update={"ledger_fingerprint": "tampered"}
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="session",
            question="分析异常",
            merchant_id="merchant",
        ),
        verified_analysis_ledger=[artifact],
        verified_skill_ledger=[invalid],
    )

    assert _latest_verified_analysis_artifacts(session) == []
