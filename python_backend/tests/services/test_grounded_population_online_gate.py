from __future__ import annotations

import ast
import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    DataSnapshotContract,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_execution_identity import (
    grounded_data_snapshot_fingerprint,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationExecutionGraphBinding,
    PopulationExecutionNodeBinding,
    PopulationGateCoordinator,
    PopulationGatePhase,
    PopulationGateState,
    PopulationResultSelection,
    seal_population_execution_graph_binding,
    seal_population_gate_state,
)
from merchant_ai.services.grounded_population_online_gate import (
    GroundedWorkspacePopulationGateStateStore,
    PopulationOnlineGateFacade,
    PopulationOnlineGateStorageError,
    PopulationOnlineLedgerError,
    PopulationZeroToolModelDecision,
    PublishedGroundedPopulationLedgerReader,
    StructuredPopulationSemanticModelProvider,
)
from merchant_ai.services.grounded_population_semantic_reviewer import (
    IndependentPopulationSemanticReviewer,
    PopulationSemanticProviderDecision,
    build_population_semantic_reviewer_request,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationConstraintEvidence,
    PopulationConstraintKind,
    PopulationExecutionClaim,
    PopulationLineageMechanism,
    PopulationLineageProof,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedVerifiedQueryArtifact,
    verified_query_artifact_integrity_fingerprint,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


QUESTION = "Return detail rows, then rank those rows by a measure"
DETAIL_GOAL = "detail.rows"
METRIC_GOAL = "metric.value"
RANKING_GOAL = "ranking.rows"
SOURCE_NODE = "node.source"
RANKING_NODE = "node.ranking"
SEMANTIC_AUTHORITY = "semantic-review-authority"
LINEAGE_AUTHORITY = "lineage-review-authority"
LEDGER_AUTHORITY = "published-ledger-authority"
CORE_AUTHORITY = "core-goal-authority"


def _stable_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _settings(tmp_path: Path) -> Settings:
    return Settings(harness_workspace_path=str(tmp_path / "runtime"))


def _workspace(settings: Settings) -> GroundedContextWorkspace:
    return GroundedContextWorkspace.open(
        settings,
        thread_id="population-online-thread",
        run_id="population-online-run",
        merchant_id="population-online-merchant",
        access_role="merchant_analyst",
        user_scope={"userId": "population-online-user"},
        question=QUESTION,
    )


def _goal_contract() -> dict[str, Any]:
    return {
        "question": QUESTION,
        "goals": [
            {
                "goalId": DETAIL_GOAL,
                "kind": "DETAIL",
                "label": "detail rows",
                "sourceSpans": ["detail rows"],
            },
            {
                "goalId": METRIC_GOAL,
                "kind": "METRIC",
                "label": "measure",
                "sourceSpans": ["measure"],
            },
            {
                "goalId": RANKING_GOAL,
                "kind": "RANKING",
                "label": "rank those rows",
                "sourceSpans": ["rank those rows by a measure"],
                "metricGoalIds": [METRIC_GOAL],
                "direction": "DESC",
                "limit": 3,
                "populationScope": "SAME_AS_GOAL",
                "populationGoalIds": [DETAIL_GOAL],
            },
        ],
    }


def _provider_decisions() -> tuple[PopulationSemanticProviderDecision, ...]:
    return (
        PopulationSemanticProviderDecision(
            goal_id=DETAIL_GOAL,
            gate_required=False,
        ),
        PopulationSemanticProviderDecision(
            goal_id=METRIC_GOAL,
            gate_required=False,
        ),
        PopulationSemanticProviderDecision(
            goal_id=RANKING_GOAL,
            gate_required=True,
            scope_kind=PopulationScopeKind.SAME_AS_GOAL,
            source_goal_ids=(DETAIL_GOAL,),
        ),
    )


class _StructuredModel:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.schema = None
        self.method = ""
        self.strict = False
        self.messages: list[Any] = []

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str,
        strict: bool,
    ) -> "_StructuredModel":
        self.schema = schema
        self.method = method
        self.strict = strict
        return self

    def invoke(self, messages: Any) -> Any:
        self.messages = list(messages)
        return self.output


def _sealed_attestation(
    stage: PopulationVerificationStage,
    *,
    goal_fingerprint: str,
    graph_fingerprint: str = "",
    previous_fingerprint: str = "",
) -> PopulationVerificationAttestation:
    pending = PopulationVerificationAttestation(
        stage=stage,
        passed=True,
        gate_open=True,
        input_fingerprint=_stable_hash(
            {
                "stage": stage.value,
                "goal": goal_fingerprint,
                "graph": graph_fingerprint,
            }
        ),
        goal_contract_fingerprint=goal_fingerprint,
        graph_fingerprint=graph_fingerprint,
        previous_attestation_fingerprint=previous_fingerprint,
    )
    return pending.model_copy(
        update={
            "attestation_fingerprint": population_attestation_fingerprint(
                pending
            )
        }
    )


def _initial_state(gate_id: str = "gate-state") -> PopulationGateState:
    goal_fingerprint = _stable_hash({"goal": gate_id})
    attestation = _sealed_attestation(
        PopulationVerificationStage.GOAL_DECLARATION,
        goal_fingerprint=goal_fingerprint,
    )
    return seal_population_gate_state(
        PopulationGateState(
            gate_id=gate_id,
            revision=1,
            phase=PopulationGatePhase.GOAL_DECLARATION,
            goal_contract_fingerprint=goal_fingerprint,
            goal_attestation=attestation,
        )
    )


def _next_pre_state(
    state: PopulationGateState,
    *,
    candidate: str,
) -> PopulationGateState:
    graph_fingerprint = _stable_hash({"candidate": candidate})
    binding = seal_population_execution_graph_binding(
        PopulationExecutionGraphBinding(
            graph_id="graph-%s" % candidate,
            graph_version=1,
            graph_fingerprint=graph_fingerprint,
            nodes=(
                PopulationExecutionNodeBinding(
                    query_node_id="node-%s" % candidate,
                    consumer_goal_ids=("goal-%s" % candidate,),
                    generation=1,
                    attempt_id="attempt-%s" % candidate,
                    query_contract_fingerprint=_stable_hash(
                        {"contract": candidate}
                    ),
                    sql_ast_fingerprint=_stable_hash({"ast": candidate}),
                    snapshot_fingerprint=_stable_hash(
                        {"snapshot": candidate}
                    ),
                ),
            ),
        )
    )
    pre_attestation = _sealed_attestation(
        PopulationVerificationStage.PRE_EXECUTION,
        goal_fingerprint=state.goal_contract_fingerprint,
        graph_fingerprint=graph_fingerprint,
        previous_fingerprint=(
            state.goal_attestation.attestation_fingerprint
        ),
    )
    return seal_population_gate_state(
        state.model_copy(
            update={
                "revision": 2,
                "phase": PopulationGatePhase.PRE_EXECUTION,
                "graph_fingerprint": graph_fingerprint,
                "graph_binding": binding,
                "pre_execution_attestation": pre_attestation,
                "state_fingerprint": "",
            }
        )
    )


def test_zero_tool_provider_exposes_only_exact_question_and_blind_skeleton() -> None:
    request = build_population_semantic_reviewer_request(
        QUESTION,
        _goal_contract(),
    )
    model = _StructuredModel(
        PopulationZeroToolModelDecision(
            complete=True,
            decisions=_provider_decisions(),
        )
    )
    provider = StructuredPopulationSemanticModelProvider(
        model,
        authority_fingerprint=SEMANTIC_AUTHORITY,
    )

    output = provider.review_population_semantics(
        request,
        timeout_seconds=2.0,
    )

    assert model.schema is PopulationZeroToolModelDecision
    assert model.method == "json_schema"
    assert model.strict is True
    supplied = json.loads(model.messages[1][1])
    assert set(supplied) == {"question", "goalSkeleton"}
    assert supplied["question"] == QUESTION
    serialized = json.dumps(supplied, ensure_ascii=False)
    for forbidden in (
        "requestFingerprint",
        "questionFingerprint",
        "goalSkeletonFingerprint",
        "populationScope",
        "populationGoalIds",
        "graphFingerprint",
        "queryNodeId",
        "sql",
        "tables",
        "artifacts",
    ):
        assert forbidden not in serialized
    assert output.request_fingerprint == request.request_fingerprint
    assert output.question_fingerprint == request.question_fingerprint
    assert output.goal_skeleton_fingerprint == (
        request.goal_skeleton_fingerprint
    )


def test_workspace_store_recovers_after_restart_and_keeps_attestation_immutable(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    state = _initial_state()
    first = GroundedWorkspacePopulationGateStateStore(workspace)

    assert first.create_population_gate(state) is True
    restarted = GroundedWorkspacePopulationGateStateStore(workspace)
    recovered = restarted.load_population_gate(state.gate_id)

    assert recovered == state
    gate_root = restarted.gate_checkpoint_path(state.gate_id)
    attestation_files = list(gate_root.glob("attestation_*.json"))
    state_files = list(gate_root.glob("state_*.json"))
    assert len(attestation_files) == 1
    assert len(state_files) == 1
    assert attestation_files[0].stat().st_mode & 0o777 == 0o400
    assert state_files[0].stat().st_mode & 0o777 == 0o400
    assert (gate_root / "head.json").stat().st_mode & 0o777 == 0o600


def test_workspace_store_revision_cas_is_process_safe_across_instances(
    tmp_path: Path,
) -> None:
    workspace = _workspace(_settings(tmp_path))
    first = GroundedWorkspacePopulationGateStateStore(workspace)
    second = GroundedWorkspacePopulationGateStateStore(workspace)
    state = _initial_state("gate-cas")
    assert first.create_population_gate(state)

    candidates = tuple(
        _next_pre_state(state, candidate=candidate)
        for candidate in ("a", "b")
    )

    def commit(item: tuple[GroundedWorkspacePopulationGateStateStore, Any]):
        store, candidate = item
        return store.compare_and_swap_population_gate(
            gate_id=state.gate_id,
            expected_revision=1,
            expected_state_fingerprint=state.state_fingerprint,
            next_state=candidate,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(commit, ((first, candidates[0]), (second, candidates[1])))
        )

    assert sorted(outcomes) == [False, True]
    recovered = GroundedWorkspacePopulationGateStateStore(
        workspace
    ).load_population_gate(state.gate_id)
    assert recovered is not None and recovered.revision == 2
    assert recovered.state_fingerprint in {
        item.state_fingerprint for item in candidates
    }


def test_workspace_store_cannot_replace_a_prior_attestation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(_settings(tmp_path))
    store = GroundedWorkspacePopulationGateStateStore(workspace)
    state = _initial_state("gate-immutable-attestation")
    assert store.create_population_gate(state)
    candidate = _next_pre_state(state, candidate="immutable")
    replacement_goal = _sealed_attestation(
        PopulationVerificationStage.GOAL_DECLARATION,
        goal_fingerprint=state.goal_contract_fingerprint,
    ).model_copy(update={"question_fingerprint": "changed-question"})
    replacement_goal = replacement_goal.model_copy(
        update={
            "attestation_fingerprint": population_attestation_fingerprint(
                replacement_goal
            )
        }
    )
    forged = seal_population_gate_state(
        candidate.model_copy(
            update={
                "goal_attestation": replacement_goal,
                "state_fingerprint": "",
            }
        )
    )

    with pytest.raises(
        PopulationOnlineGateStorageError,
        match="POPULATION_GATE_IMMUTABLE_BINDING_CHANGED",
    ):
        store.compare_and_swap_population_gate(
            gate_id=state.gate_id,
            expected_revision=1,
            expected_state_fingerprint=state.state_fingerprint,
            next_state=forged,
        )

    assert store.load_population_gate(state.gate_id) == state


def test_workspace_store_rejects_tampered_revision_and_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    store = GroundedWorkspacePopulationGateStateStore(workspace)
    state = _initial_state("gate-tamper")
    assert store.create_population_gate(state)
    state_file = next(
        store.gate_checkpoint_path(state.gate_id).glob("state_*.json")
    )
    state_file.chmod(0o600)
    state_file.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(
        PopulationOnlineGateStorageError,
        match="POPULATION_GATE_IMMUTABLE_HASH_MISMATCH",
    ):
        store.load_population_gate(state.gate_id)

    other_settings = _settings(tmp_path / "symlink-case")
    other_workspace = _workspace(other_settings)
    outside = tmp_path / "outside-checkpoints"
    outside.mkdir()
    (other_workspace.root / "checkpoints").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises((OSError, PopulationOnlineGateStorageError)):
        GroundedWorkspacePopulationGateStateStore(other_workspace)
    assert list(outside.iterdir()) == []


def test_workspace_store_does_not_reset_when_head_is_missing(
    tmp_path: Path,
) -> None:
    workspace = _workspace(_settings(tmp_path))
    store = GroundedWorkspacePopulationGateStateStore(workspace)
    state = _initial_state("gate-missing-head")
    assert store.create_population_gate(state)
    (store.gate_checkpoint_path(state.gate_id) / "head.json").unlink()

    with pytest.raises(
        PopulationOnlineGateStorageError,
        match="POPULATION_GATE_HEAD_MISSING_WITH_HISTORY",
    ):
        store.load_population_gate(state.gate_id)
    with pytest.raises(
        PopulationOnlineGateStorageError,
        match="POPULATION_GATE_HEAD_MISSING_WITH_HISTORY",
    ):
        store.create_population_gate(state)


def _published_query_artifact(
    settings: Settings,
    workspace: GroundedContextWorkspace,
    *,
    query_contract: GroundedQueryContract,
    sql_ast_fingerprint: str,
    snapshot: DataSnapshotContract,
    coverage: str = "TOP_N",
    truncated: bool = False,
    publication_status: str = "PUBLISHED",
) -> GroundedVerifiedQueryArtifact:
    rows = [
        {"entity_id": "order-a", "value": 9},
        {"entity_id": "order-b", "value": 7},
        {"entity_id": "order-c", "value": 5},
    ]
    store = WorkspaceArtifactStore(settings, workspace.artifacts_root)
    tag = _stable_hash(
        {
            "coverage": coverage,
            "truncated": truncated,
            "publicationStatus": publication_status,
        }
    )[:16]
    rows_artifact = store.write_json(
        "query_results",
        "%s.rows.json" % tag,
        rows,
        preview_chars=0,
        immutable=True,
    )
    sql_artifact = store.write_text(
        "query_results",
        "%s.sql" % tag,
        "SELECT governed_result",
        preview_chars=0,
        immutable=True,
    )
    verified = VerifiedEvidence(passed=True)
    verified_payload = verified.model_dump(by_alias=True, mode="json")
    verified_sha256 = _stable_hash(verified_payload)
    contract_fingerprint = grounded_query_contract_fingerprint(
        query_contract
    )
    generation = 7
    attempt_id = "attempt-ranking"
    exact_count = len(rows) if not truncated else 9
    snapshot_payload = snapshot.model_dump(by_alias=True, mode="json")
    publication_fingerprint = _stable_hash(
        {
            "tag": tag,
            "contract": contract_fingerprint,
            "ast": sql_ast_fingerprint,
        }
    )
    manifest_payload = {
        "schemaVersion": 2,
        "artifactKind": "GROUNDED_QUERY_RESULT",
        "publicationStatus": "VERIFIED",
        "artifactFingerprint": publication_fingerprint,
        "executionGeneration": generation,
        "executionAttemptId": attempt_id,
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_ast_fingerprint,
        "sqlSha256": sql_artifact["sha256"],
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": (
            snapshot.semantic_activation_fingerprint
        ),
        "dataSnapshot": snapshot_payload,
        "resultCoverage": coverage,
        "resultIsTruncated": truncated,
        "storedRowCount": len(rows),
        "exactResultRowCount": exact_count,
        "verifiedEvidence": verified_payload,
        "verifiedEvidenceSha256": verified_sha256,
        "rowsArtifact": {
            key: rows_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
        "sqlArtifact": {
            key: sql_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
    }
    manifest_artifact = store.write_json(
        "query_results",
        "%s.manifest.json" % tag,
        manifest_payload,
        preview_chars=0,
        immutable=True,
    )
    receipt = {
        "artifactFingerprint": publication_fingerprint,
        "queryManifestSha256": manifest_artifact["sha256"],
        "rowsSha256": rows_artifact["sha256"],
        "sqlSha256": sql_artifact["sha256"],
        "manifestContentAddress": manifest_artifact["contentAddress"],
        "rowsContentAddress": rows_artifact["contentAddress"],
        "sqlContentAddress": sql_artifact["contentAddress"],
        "manifestRelativePath": manifest_artifact["relativePath"],
        "rowsRelativePath": rows_artifact["relativePath"],
        "sqlRelativePath": sql_artifact["relativePath"],
        "manifestRef": manifest_artifact["merchantUri"],
        "rowsRef": rows_artifact["merchantUri"],
        "sqlRef": sql_artifact["merchantUri"],
        "storedRowCount": len(rows),
        "exactResultRowCount": exact_count,
        "resultCoverage": coverage,
        "resultIsTruncated": truncated,
        "executionGeneration": generation,
        "attemptFingerprint": hashlib.sha256(
            attempt_id.encode("utf-8")
        ).hexdigest(),
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_ast_fingerprint,
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": (
            snapshot.semantic_activation_fingerprint
        ),
        "dataSnapshotFingerprint": _stable_hash(snapshot_payload),
        "verifiedEvidenceSha256": verified_sha256,
    }
    artifact = GroundedVerifiedQueryArtifact(
        artifact_id="query-artifact-%s" % tag,
        generation=generation,
        attempt_id=attempt_id,
        contract_fingerprint=contract_fingerprint,
        sql_fingerprint=sql_ast_fingerprint,
        contract=query_contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=rows,
                result_coverage=coverage,
                is_truncated=truncated,
                data_snapshot=snapshot,
            )
        ),
        verified_evidence=verified,
        publication_status=publication_status,
        result_artifact_receipts=[receipt],
        output_columns=["entity_id", "value"],
    )
    artifact.ledger_fingerprint = (
        verified_query_artifact_integrity_fingerprint(artifact)
    )
    return artifact


def _pre_scope(
    snapshot_fingerprint: str,
) -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="resolved-ranking-population",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(DETAIL_GOAL,),
        population_fingerprint=_stable_hash(
            {"population": "verified-detail-result"}
        ),
        entity_identity_ref="semantic:entity:order",
        grain_fingerprint=_stable_hash({"grain": "order"}),
        snapshot_fingerprint=snapshot_fingerprint,
        constraints=(
            PopulationConstraintEvidence(
                fingerprint=_stable_hash({"constraint": "time"}),
                kind=PopulationConstraintKind.TIME,
            ),
            PopulationConstraintEvidence(
                fingerprint=_stable_hash({"constraint": "membership"}),
                kind=PopulationConstraintKind.ENTITY_MEMBERSHIP,
            ),
        ),
        complete_membership_required=True,
    )


def _online_environment(
    tmp_path: Path,
    *,
    coverage: str = "TOP_N",
    truncated: bool = False,
    publication_status: str = "PUBLISHED",
) -> dict[str, Any]:
    settings = _settings(tmp_path)
    workspace = _workspace(settings)
    state_store = GroundedWorkspacePopulationGateStateStore(workspace)
    ledger: list[GroundedVerifiedQueryArtifact] = []
    reader = PublishedGroundedPopulationLedgerReader(
        settings=settings,
        workspace=workspace,
        state_store=state_store,
        ledger_provider=lambda: tuple(ledger),
        authority_fingerprint=LEDGER_AUTHORITY,
    )
    model = _StructuredModel(
        PopulationZeroToolModelDecision(
            complete=True,
            decisions=_provider_decisions(),
        )
    )
    semantic_provider = StructuredPopulationSemanticModelProvider(
        model,
        authority_fingerprint=SEMANTIC_AUTHORITY,
    )
    reviewer = IndependentPopulationSemanticReviewer(
        semantic_provider,
        trusted_provider_authority_fingerprints=(SEMANTIC_AUTHORITY,),
        timeout_seconds=2.0,
    )
    coordinator = PopulationGateCoordinator(
        state_store=state_store,
        ledger_reader=reader,
        trusted_semantic_verifier_fingerprints=(SEMANTIC_AUTHORITY,),
        trusted_lineage_verifier_fingerprints=(LINEAGE_AUTHORITY,),
        trusted_artifact_verifier_fingerprints=(LEDGER_AUTHORITY,),
        trusted_ledger_authority_fingerprints=(LEDGER_AUTHORITY,),
    )
    facade = PopulationOnlineGateFacade(
        semantic_reviewer=reviewer,
        coordinator=coordinator,
        declaration_author_fingerprint=CORE_AUTHORITY,
    )
    gate_id = "population-online-gate"
    goal = facade.commit_goal(
        gate_id=gate_id,
        expected_revision=0,
        exact_question=QUESTION,
        goal_contract=_goal_contract(),
    )
    assert goal.accepted is True, goal.model_dump()
    goal_state = goal.transition.state
    query_contract = GroundedQueryContract(
        question=QUESTION,
        status="READY",
        query_shape="RANKED",
    )
    query_contract_fingerprint = grounded_query_contract_fingerprint(
        query_contract
    )
    sql_ast_fingerprint = _stable_hash({"ast": "ranking-node"})
    snapshot = DataSnapshotContract(
        datasource_fingerprint=_stable_hash({"datasource": "primary"}),
        datasource_environment="test",
        consistency_mode="UNSUPPORTED",
        semantic_activation_fingerprint=_stable_hash(
            {"semanticActivation": "v1"}
        ),
        cache_generation="cache-generation-1",
        captured_at="2026-07-19T00:00:00Z",
        unsupported_reason="TEST_SNAPSHOT_UNAVAILABLE",
    )
    snapshot_fingerprint = grounded_data_snapshot_fingerprint(snapshot)
    scope = _pre_scope(snapshot_fingerprint)
    declaration_fingerprint = next(
        item.declaration_scope_fingerprint
        for item in goal_state.goal_attestation.accepted_scopes
        if item.consumer_goal_id == RANKING_GOAL
    )
    proof = PopulationLineageProof(
        proof_id="ranking-population-proof",
        mechanism=PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE,
        verifier_fingerprint=LINEAGE_AUTHORITY,
        verified=True,
        graph_fingerprint=_stable_hash({"graph": "dynamic"}),
        query_node_id=RANKING_NODE,
        generation=7,
        attempt_id="attempt-ranking",
        query_contract_fingerprint=query_contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        source_population_fingerprint=scope.population_fingerprint,
        result_population_fingerprint=scope.population_fingerprint,
        source_goal_ids=(DETAIL_GOAL,),
        source_node_ids=(SOURCE_NODE,),
        preserved_constraints=scope.constraints,
        source_entity_identity_ref=scope.entity_identity_ref,
        result_entity_identity_ref=scope.entity_identity_ref,
        source_grain_fingerprint=scope.grain_fingerprint,
        result_grain_fingerprint=scope.grain_fingerprint,
        source_snapshot_fingerprint=snapshot_fingerprint,
        result_snapshot_fingerprint=snapshot_fingerprint,
        complete_membership=True,
    )
    graph_fingerprint = proof.graph_fingerprint
    graph_binding = seal_population_execution_graph_binding(
        PopulationExecutionGraphBinding(
            graph_id="dynamic-graph",
            graph_version=1,
            graph_fingerprint=graph_fingerprint,
            nodes=(
                PopulationExecutionNodeBinding(
                    query_node_id=SOURCE_NODE,
                    consumer_goal_ids=(DETAIL_GOAL,),
                    generation=3,
                    attempt_id="attempt-source",
                    query_contract_fingerprint=_stable_hash(
                        {"contract": "source"}
                    ),
                    sql_ast_fingerprint=_stable_hash({"ast": "source"}),
                    snapshot_fingerprint=snapshot_fingerprint,
                ),
                PopulationExecutionNodeBinding(
                    query_node_id=RANKING_NODE,
                    consumer_goal_ids=(RANKING_GOAL,),
                    generation=7,
                    attempt_id="attempt-ranking",
                    query_contract_fingerprint=(
                        query_contract_fingerprint
                    ),
                    sql_ast_fingerprint=sql_ast_fingerprint,
                    snapshot_fingerprint=snapshot_fingerprint,
                ),
            ),
        )
    )
    claim = PopulationExecutionClaim(
        consumer_goal_id=RANKING_GOAL,
        query_node_id=RANKING_NODE,
        generation=7,
        attempt_id="attempt-ranking",
        declaration_scope_fingerprint=declaration_fingerprint,
        required_scope=scope,
        effective_scope=scope,
        query_contract_fingerprint=query_contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        lineage_proofs=(proof,),
    )
    pre = facade.authorize_pre_execution(
        gate_id=gate_id,
        expected_revision=1,
        graph_binding=graph_binding,
        claims=(claim,),
    )
    assert pre.accepted is True, pre.model_dump()
    artifact = _published_query_artifact(
        settings,
        workspace,
        query_contract=query_contract,
        sql_ast_fingerprint=sql_ast_fingerprint,
        snapshot=snapshot,
        coverage=coverage,
        truncated=truncated,
        publication_status=publication_status,
    )
    ledger.append(artifact)
    return {
        "settings": settings,
        "workspace": workspace,
        "stateStore": state_store,
        "reader": reader,
        "ledger": ledger,
        "artifact": artifact,
        "facade": facade,
        "gateId": gate_id,
        "goal": goal,
        "pre": pre,
        "graphFingerprint": graph_fingerprint,
    }


def test_facade_keeps_goal_topology_open_and_supports_per_node_attempts(
    tmp_path: Path,
) -> None:
    environment = _online_environment(tmp_path)
    goal_state = environment["goal"].transition.state
    pre_state = environment["pre"].transition.state

    assert goal_state.graph_fingerprint == ""
    assert goal_state.graph_binding is None
    assert [node.generation for node in pre_state.graph_binding.nodes] == [3, 7]
    assert [node.attempt_id for node in pre_state.graph_binding.nodes] == [
        "attempt-source",
        "attempt-ranking",
    ]
    for method_name in ("authorize_pre_execution", "commit_post_result"):
        parameters = inspect.signature(
            getattr(PopulationOnlineGateFacade, method_name)
        ).parameters
        assert "previous_attestation" not in parameters
        assert "attestation" not in parameters


def test_published_ledger_and_facade_complete_post_result_from_store_chain(
    tmp_path: Path,
) -> None:
    environment = _online_environment(tmp_path)
    observed_reads: list[str] = []
    original_read = environment["reader"].artifact_store.read

    def audited_read(path: str, **kwargs: Any):
        observed_reads.append(path)
        return original_read(path, **kwargs)

    environment["reader"].artifact_store.read = audited_read
    snapshot = environment["reader"].snapshot_population_artifacts(
        gate_id=environment["gateId"],
        goal_contract_fingerprint=(
            environment["pre"].transition.state.goal_contract_fingerprint
        ),
        graph_fingerprint=environment["graphFingerprint"],
    )

    assert len(snapshot.entries) == 1
    entry = snapshot.entries[0]
    assert entry.receipt.query_node_id == RANKING_NODE
    assert entry.receipt.generation == 7
    assert entry.receipt.attempt_id == "attempt-ranking"
    assert entry.receipt.evidence.query_contract_fingerprint == (
        environment["artifact"].contract_fingerprint
    )
    assert entry.receipt.evidence.sql_ast_fingerprint == (
        environment["artifact"].sql_fingerprint
    )
    assert observed_reads
    assert not any(path.endswith(".sql") for path in observed_reads)
    post = environment["facade"].commit_post_result(
        gate_id=environment["gateId"],
        expected_revision=2,
        selections=(
            PopulationResultSelection(
                consumer_goal_id=RANKING_GOAL,
                query_node_id=RANKING_NODE,
                ledger_artifact_id=entry.ledger_artifact_id,
                receipt_fingerprint=entry.receipt.receipt_fingerprint,
            ),
        ),
    )

    assert post.accepted is True, post.model_dump()
    recovered = GroundedWorkspacePopulationGateStateStore(
        environment["workspace"]
    ).load_population_gate(environment["gateId"])
    assert recovered.phase == PopulationGatePhase.POST_RESULT.value
    assert recovered.revision == 3


def test_ledger_ignores_orphan_files_and_non_published_artifacts(
    tmp_path: Path,
) -> None:
    environment = _online_environment(
        tmp_path,
        publication_status="PENDING",
    )
    WorkspaceArtifactStore(
        environment["settings"],
        environment["workspace"].artifacts_root,
    ).write_json(
        "query_results",
        "orphan-published-looking.json",
        {"publicationStatus": "VERIFIED"},
        preview_chars=0,
        immutable=True,
    )

    snapshot = environment["reader"].snapshot_population_artifacts(
        gate_id=environment["gateId"],
        goal_contract_fingerprint=(
            environment["pre"].transition.state.goal_contract_fingerprint
        ),
        graph_fingerprint=environment["graphFingerprint"],
    )

    assert snapshot.entries == ()


def test_ledger_rejects_nested_artifact_tampering_before_projection(
    tmp_path: Path,
) -> None:
    environment = _online_environment(tmp_path)
    environment["artifact"].result_artifact_receipts[0]["rowsSha256"] = (
        hashlib.sha256(b"tampered-ledger-receipt").hexdigest()
    )

    with pytest.raises(
        PopulationOnlineLedgerError,
        match="POPULATION_LEDGER_ARTIFACT_INTEGRITY_INVALID",
    ):
        environment["reader"].snapshot_population_artifacts(
            gate_id=environment["gateId"],
            goal_contract_fingerprint=(
                environment[
                    "pre"
                ].transition.state.goal_contract_fingerprint
            ),
            graph_fingerprint=environment["graphFingerprint"],
        )


def test_preview_cannot_finalize_population_gate(tmp_path: Path) -> None:
    environment = _online_environment(
        tmp_path,
        coverage="PREVIEW",
        truncated=True,
    )
    snapshot = environment["reader"].snapshot_population_artifacts(
        gate_id=environment["gateId"],
        goal_contract_fingerprint=(
            environment["pre"].transition.state.goal_contract_fingerprint
        ),
        graph_fingerprint=environment["graphFingerprint"],
    )
    entry = snapshot.entries[0]

    post = environment["facade"].commit_post_result(
        gate_id=environment["gateId"],
        expected_revision=2,
        selections=(
            PopulationResultSelection(
                consumer_goal_id=RANKING_GOAL,
                query_node_id=RANKING_NODE,
                ledger_artifact_id=entry.ledger_artifact_id,
                receipt_fingerprint=entry.receipt.receipt_fingerprint,
            ),
        ),
    )

    assert post.accepted is False
    assert post.code == "RESULT_COVERAGE_INCOMPLETE"
    assert {
        str(issue.code) for issue in post.transition.issues
    }.issuperset(
        {"RESULT_COVERAGE_INCOMPLETE", "RESULT_TRUNCATED"}
    )


def test_cross_node_result_replay_is_rejected(tmp_path: Path) -> None:
    environment = _online_environment(tmp_path)
    snapshot = environment["reader"].snapshot_population_artifacts(
        gate_id=environment["gateId"],
        goal_contract_fingerprint=(
            environment["pre"].transition.state.goal_contract_fingerprint
        ),
        graph_fingerprint=environment["graphFingerprint"],
    )
    entry = snapshot.entries[0]

    replay = environment["facade"].commit_post_result(
        gate_id=environment["gateId"],
        expected_revision=2,
        selections=(
            PopulationResultSelection(
                consumer_goal_id=RANKING_GOAL,
                query_node_id=SOURCE_NODE,
                ledger_artifact_id=entry.ledger_artifact_id,
                receipt_fingerprint=entry.receipt.receipt_fingerprint,
            ),
        ),
    )

    assert replay.accepted is False
    assert replay.code == "RESULT_BINDING_MISMATCH"
    retained = environment["stateStore"].load_population_gate(
        environment["gateId"]
    )
    assert retained.phase == PopulationGatePhase.PRE_EXECUTION.value
    assert retained.revision == 2


def test_online_gate_source_has_no_regex_or_process_local_authority() -> None:
    source_path = Path(
        "python_backend/merchant_ai/services/grounded_population_online_gate.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "re" for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module == "re"
        )
        for node in ast.walk(tree)
    )
    assert "self._states" not in source
    assert "bind_tools" not in source
    assert "result_artifact_receipts" in source
    assert "verified_query_artifact_integrity_valid" in source
