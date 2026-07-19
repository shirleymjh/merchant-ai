from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merchant_ai.services.grounded_population_gate_coordinator import (
    InMemoryPopulationGateStateStore,
    PopulationArtifactLedgerEntry,
    PopulationArtifactLedgerSnapshot,
    PopulationExecutionGraphBinding,
    PopulationExecutionNodeBinding,
    PopulationGateCode,
    PopulationGateCoordinator,
    PopulationGatePhase,
    PopulationGoalDeclarationCommand,
    PopulationPostResultCommand,
    PopulationPreExecutionCommand,
    PopulationPublishedArtifactReceipt,
    PopulationResultSelection,
    population_artifact_ledger_snapshot_fingerprint,
    population_gate_state_fingerprint,
    seal_population_artifact_ledger_entry,
    seal_population_artifact_ledger_snapshot,
    seal_population_execution_graph_binding,
    seal_population_published_artifact_receipt,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationConstraintEvidence,
    PopulationConstraintKind,
    PopulationDeclaration,
    PopulationExecutionClaim,
    PopulationLineageMechanism,
    PopulationLineageProof,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationSemanticExpectation,
    PopulationSemanticReview,
    population_attestation_fingerprint,
)


GATE_ID = "gate-1"
GENERATION = 7
ATTEMPT_ID = "attempt-1"
QUESTION_FP = "question-fingerprint"
GOAL_FP = "goal-contract-fingerprint"
GRAPH_FP = "execution-graph-fingerprint"
SEMANTIC_VERIFIER = "semantic-verifier-authority"
LINEAGE_VERIFIER = "lineage-verifier-authority"
ARTIFACT_VERIFIER = "artifact-verifier-authority"
LEDGER_AUTHORITY = "ledger-authority"
CORE_AUTHOR = "core-authority"
CONSUMER_GOAL = "consumer-goal"
SECOND_CONSUMER_GOAL = "second-consumer-goal"
SOURCE_GOAL = "source-goal"
CONSUMER_NODE = "consumer-node"
SECOND_CONSUMER_NODE = "second-consumer-node"
SOURCE_NODE = "source-node"
QUERY_CONTRACT_FP = "query-contract-fingerprint"
SQL_AST_FP = "validated-sql-ast-fingerprint"
SNAPSHOT_FP = "data-snapshot-fingerprint"
POPULATION_FP = "population-fingerprint"
LEDGER_ARTIFACT_ID = "ledger-artifact-1"


class _LedgerReader:
    def __init__(
        self,
        *,
        authority_fingerprint: str = LEDGER_AUTHORITY,
    ) -> None:
        self._authority_fingerprint = authority_fingerprint
        self.snapshot: PopulationArtifactLedgerSnapshot | None = None
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    @property
    def authority_fingerprint(self) -> str:
        return self._authority_fingerprint

    def snapshot_population_artifacts(
        self,
        *,
        gate_id: str,
        goal_contract_fingerprint: str,
        graph_fingerprint: str,
    ) -> PopulationArtifactLedgerSnapshot:
        self.calls.append(
            {
                "gate_id": gate_id,
                "goal_contract_fingerprint": goal_contract_fingerprint,
                "graph_fingerprint": graph_fingerprint,
            }
        )
        if self.error is not None:
            raise self.error
        if self.snapshot is None:
            raise RuntimeError("ledger snapshot is unavailable")
        return self.snapshot


def _coordinator(
    reader: _LedgerReader | None = None,
) -> tuple[PopulationGateCoordinator, _LedgerReader]:
    ledger = reader or _LedgerReader()
    return (
        PopulationGateCoordinator(
            state_store=InMemoryPopulationGateStateStore(),
            ledger_reader=ledger,
            trusted_semantic_verifier_fingerprints=(SEMANTIC_VERIFIER,),
            trusted_lineage_verifier_fingerprints=(LINEAGE_VERIFIER,),
            trusted_artifact_verifier_fingerprints=(ARTIFACT_VERIFIER,),
            trusted_ledger_authority_fingerprints=(LEDGER_AUTHORITY,),
        ),
        ledger,
    )


def _declared_scope() -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="declared-scope",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(SOURCE_GOAL,),
        complete_membership_required=True,
    )


def _resolved_scope(
    *,
    population_fingerprint: str = POPULATION_FP,
    entity_identity_ref: str = "entity-identity",
    constraints: tuple[PopulationConstraintEvidence, ...] | None = None,
) -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="resolved-scope",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(SOURCE_GOAL,),
        population_fingerprint=population_fingerprint,
        entity_identity_ref=entity_identity_ref,
        grain_fingerprint="entity-grain",
        snapshot_fingerprint=SNAPSHOT_FP,
        constraints=constraints
        if constraints is not None
        else (
            PopulationConstraintEvidence(
                fingerprint="time-constraint",
                kind=PopulationConstraintKind.TIME,
            ),
            PopulationConstraintEvidence(
                fingerprint="membership-constraint",
                kind=PopulationConstraintKind.ENTITY_MEMBERSHIP,
            ),
        ),
        complete_membership_required=True,
    )


def _goal_command(
    *,
    expected_revision: int = 0,
    semantic_verifier: str = SEMANTIC_VERIFIER,
) -> PopulationGoalDeclarationCommand:
    scope = _declared_scope()
    return PopulationGoalDeclarationCommand(
        gate_id=GATE_ID,
        expected_revision=expected_revision,
        goal_contract_fingerprint=GOAL_FP,
        question_fingerprint=QUESTION_FP,
        declaration_author_fingerprint=CORE_AUTHOR,
        semantic_review=PopulationSemanticReview(
            review_id="semantic-review",
            question_fingerprint=QUESTION_FP,
            verifier_fingerprint=semantic_verifier,
            complete=True,
            expectations=(
                PopulationSemanticExpectation(
                    expectation_id="population-expectation",
                    consumer_goal_id=CONSUMER_GOAL,
                    expected_scope=scope,
                ),
            ),
        ),
        declarations=(
            PopulationDeclaration(
                consumer_goal_id=CONSUMER_GOAL,
                declared_scope=scope,
            ),
        ),
    )


def _graph_binding(
    *,
    graph_fingerprint: str = GRAPH_FP,
    generation: int = GENERATION,
    attempt_id: str = ATTEMPT_ID,
    query_contract_fingerprint: str = QUERY_CONTRACT_FP,
    sql_ast_fingerprint: str = SQL_AST_FP,
    snapshot_fingerprint: str = SNAPSHOT_FP,
    consumer_goal_ids: tuple[str, ...] = (CONSUMER_GOAL,),
) -> PopulationExecutionGraphBinding:
    return seal_population_execution_graph_binding(
        PopulationExecutionGraphBinding(
            graph_id="graph-1",
            graph_version=3,
            graph_fingerprint=graph_fingerprint,
            nodes=(
                PopulationExecutionNodeBinding(
                    query_node_id=SOURCE_NODE,
                    consumer_goal_ids=(SOURCE_GOAL,),
                    generation=GENERATION - 1,
                    attempt_id="source-attempt",
                    query_contract_fingerprint="source-query-contract",
                    sql_ast_fingerprint="source-sql-ast",
                    snapshot_fingerprint=snapshot_fingerprint,
                ),
                PopulationExecutionNodeBinding(
                    query_node_id=CONSUMER_NODE,
                    consumer_goal_ids=consumer_goal_ids,
                    generation=generation,
                    attempt_id=attempt_id,
                    query_contract_fingerprint=(
                        query_contract_fingerprint
                    ),
                    sql_ast_fingerprint=sql_ast_fingerprint,
                    snapshot_fingerprint=snapshot_fingerprint,
                ),
            ),
        )
    )


def _claim(
    goal_attestation,
    *,
    effective_scope: PopulationScopeDescriptor | None = None,
    preserved_constraints: tuple[PopulationConstraintEvidence, ...]
    | None = None,
    query_node_id: str = CONSUMER_NODE,
    generation: int = GENERATION,
    attempt_id: str = ATTEMPT_ID,
    query_contract_fingerprint: str = QUERY_CONTRACT_FP,
    sql_ast_fingerprint: str = SQL_AST_FP,
) -> PopulationExecutionClaim:
    required = _resolved_scope()
    effective = effective_scope or required
    proof = PopulationLineageProof(
        proof_id="population-proof",
        mechanism=PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE,
        verifier_fingerprint=LINEAGE_VERIFIER,
        verified=True,
        graph_fingerprint=GRAPH_FP,
        query_node_id=query_node_id,
        generation=generation,
        attempt_id=attempt_id,
        query_contract_fingerprint=query_contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        source_population_fingerprint=required.population_fingerprint,
        result_population_fingerprint=effective.population_fingerprint,
        source_goal_ids=required.source_goal_ids,
        source_node_ids=(SOURCE_NODE,),
        preserved_constraints=preserved_constraints
        if preserved_constraints is not None
        else required.constraints,
        source_entity_identity_ref=required.entity_identity_ref,
        result_entity_identity_ref=effective.entity_identity_ref,
        source_grain_fingerprint=required.grain_fingerprint,
        result_grain_fingerprint=effective.grain_fingerprint,
        source_snapshot_fingerprint=required.snapshot_fingerprint,
        result_snapshot_fingerprint=effective.snapshot_fingerprint,
        complete_membership=True,
    )
    declaration_scope_fingerprint = next(
        item.declaration_scope_fingerprint
        for item in goal_attestation.accepted_scopes
        if item.consumer_goal_id == CONSUMER_GOAL
    )
    return PopulationExecutionClaim(
        consumer_goal_id=CONSUMER_GOAL,
        query_node_id=query_node_id,
        generation=generation,
        attempt_id=attempt_id,
        declaration_scope_fingerprint=declaration_scope_fingerprint,
        required_scope=required,
        effective_scope=effective,
        query_contract_fingerprint=query_contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        lineage_proofs=(proof,),
    )


def _pre_command(
    goal_state,
    *,
    graph_binding: PopulationExecutionGraphBinding | None = None,
    claim: PopulationExecutionClaim | None = None,
    expected_revision: int = 1,
    goal_fingerprint: str = GOAL_FP,
) -> PopulationPreExecutionCommand:
    return PopulationPreExecutionCommand(
        gate_id=GATE_ID,
        expected_revision=expected_revision,
        goal_contract_fingerprint=goal_fingerprint,
        graph_binding=graph_binding or _graph_binding(),
        claims=(claim or _claim(goal_state.goal_attestation),),
    )


def _prepare_pre_gate() -> tuple[
    PopulationGateCoordinator,
    _LedgerReader,
    object,
]:
    coordinator, ledger = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())
    assert goal.committed is True
    pre = coordinator.authorize_pre_execution(_pre_command(goal.state))
    assert pre.committed is True, pre.model_dump()
    return coordinator, ledger, pre.state


def _ledger_snapshot(
    pre_state,
    *,
    publication_status: str = "PUBLISHED",
    coverage: PopulationArtifactCoverage = PopulationArtifactCoverage.TOP_N,
    verified: bool = True,
    immutable: bool = True,
    truncated: bool = False,
    stored_row_count: int = 3,
    exact_result_row_count: int = 3,
    authority: str = LEDGER_AUTHORITY,
    generation: int = GENERATION,
    attempt_id: str = ATTEMPT_ID,
    goal_fingerprint: str = GOAL_FP,
    graph_fingerprint: str = GRAPH_FP,
    query_node_id: str = CONSUMER_NODE,
    covered_goal_ids: tuple[str, ...] = (CONSUMER_GOAL,),
) -> PopulationArtifactLedgerSnapshot:
    proof_fingerprints = (
        pre_state.pre_execution_attestation.accepted_proof_fingerprints
    )
    evidence = PopulationArtifactEvidence(
        artifact_id="result-artifact",
        artifact_fingerprint="result-content-address",
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=coverage,
        population_fingerprint=POPULATION_FP,
        verifier_fingerprint=ARTIFACT_VERIFIER,
        verified=verified,
        immutable=immutable,
        goal_contract_fingerprint=goal_fingerprint,
        graph_fingerprint=graph_fingerprint,
        query_contract_fingerprint=QUERY_CONTRACT_FP,
        sql_ast_fingerprint=SQL_AST_FP,
        snapshot_fingerprint=SNAPSHOT_FP,
        lineage_proof_fingerprints=proof_fingerprints,
    )
    receipt = seal_population_published_artifact_receipt(
        PopulationPublishedArtifactReceipt(
            ledger_artifact_id=LEDGER_ARTIFACT_ID,
            publication_status=publication_status,
            generation=generation,
            attempt_id=attempt_id,
            goal_contract_fingerprint=goal_fingerprint,
            graph_fingerprint=graph_fingerprint,
            query_node_id=query_node_id,
            covered_consumer_goal_ids=covered_goal_ids,
            result_is_truncated=truncated,
            stored_row_count=stored_row_count,
            exact_result_row_count=exact_result_row_count,
            evidence=evidence,
        )
    )
    entry = seal_population_artifact_ledger_entry(
        PopulationArtifactLedgerEntry(
            ledger_artifact_id=LEDGER_ARTIFACT_ID,
            publication_status=publication_status,
            receipt=receipt,
        )
    )
    return seal_population_artifact_ledger_snapshot(
        PopulationArtifactLedgerSnapshot(
            ledger_id="ledger-1",
            ledger_authority_fingerprint=authority,
            ledger_revision=9,
            goal_contract_fingerprint=goal_fingerprint,
            graph_fingerprint=graph_fingerprint,
            entries=(entry,),
        )
    )


def _post_command(
    receipt_fingerprint: str,
    *,
    ledger_artifact_id: str = LEDGER_ARTIFACT_ID,
    consumer_goal_id: str = CONSUMER_GOAL,
    query_node_id: str = CONSUMER_NODE,
    expected_revision: int = 2,
    goal_fingerprint: str = GOAL_FP,
    graph_fingerprint: str = GRAPH_FP,
) -> PopulationPostResultCommand:
    return PopulationPostResultCommand(
        gate_id=GATE_ID,
        expected_revision=expected_revision,
        goal_contract_fingerprint=goal_fingerprint,
        graph_fingerprint=graph_fingerprint,
        selections=(
            PopulationResultSelection(
                consumer_goal_id=consumer_goal_id,
                query_node_id=query_node_id,
                ledger_artifact_id=ledger_artifact_id,
                receipt_fingerprint=receipt_fingerprint,
            ),
        ),
    )


def _post_from_snapshot(
    coordinator: PopulationGateCoordinator,
    ledger: _LedgerReader,
    snapshot: PopulationArtifactLedgerSnapshot,
):
    ledger.snapshot = snapshot
    receipt_fingerprint = snapshot.entries[0].receipt.receipt_fingerprint
    return coordinator.commit_post_result(
        _post_command(receipt_fingerprint)
    )


def _issue_codes(result) -> set[str]:
    return {str(issue.code) for issue in result.issues}


def _two_consumer_goal_command() -> PopulationGoalDeclarationCommand:
    first_scope = _declared_scope()
    second_scope = first_scope.model_copy(update={"scope_id": "second-scope"})
    base = _goal_command()
    return base.model_copy(
        update={
            "semantic_review": base.semantic_review.model_copy(
                update={
                    "expectations": (
                        PopulationSemanticExpectation(
                            expectation_id="first-expectation",
                            consumer_goal_id=CONSUMER_GOAL,
                            expected_scope=first_scope,
                        ),
                        PopulationSemanticExpectation(
                            expectation_id="second-expectation",
                            consumer_goal_id=SECOND_CONSUMER_GOAL,
                            expected_scope=second_scope,
                        ),
                    )
                }
            ),
            "declarations": (
                PopulationDeclaration(
                    consumer_goal_id=CONSUMER_GOAL,
                    declared_scope=first_scope,
                ),
                PopulationDeclaration(
                    consumer_goal_id=SECOND_CONSUMER_GOAL,
                    declared_scope=second_scope,
                ),
            ),
        }
    )


def _node_specific_claim(
    goal_attestation,
    *,
    consumer_goal_id: str,
    query_node_id: str,
    generation: int,
    attempt_id: str,
    query_contract_fingerprint: str,
    sql_ast_fingerprint: str,
    snapshot_fingerprint: str,
    population_fingerprint: str,
) -> PopulationExecutionClaim:
    constraints = (
        PopulationConstraintEvidence(
            fingerprint="%s-time" % consumer_goal_id,
            kind=PopulationConstraintKind.TIME,
        ),
        PopulationConstraintEvidence(
            fingerprint="%s-membership" % consumer_goal_id,
            kind=PopulationConstraintKind.ENTITY_MEMBERSHIP,
        ),
    )
    scope = PopulationScopeDescriptor(
        scope_id="%s-resolved" % consumer_goal_id,
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=(SOURCE_GOAL,),
        population_fingerprint=population_fingerprint,
        entity_identity_ref="shared-entity-identity",
        grain_fingerprint="shared-grain",
        snapshot_fingerprint=snapshot_fingerprint,
        constraints=constraints,
        complete_membership_required=True,
    )
    declaration_fingerprint = next(
        item.declaration_scope_fingerprint
        for item in goal_attestation.accepted_scopes
        if item.consumer_goal_id == consumer_goal_id
    )
    return PopulationExecutionClaim(
        consumer_goal_id=consumer_goal_id,
        query_node_id=query_node_id,
        generation=generation,
        attempt_id=attempt_id,
        declaration_scope_fingerprint=declaration_fingerprint,
        required_scope=scope,
        effective_scope=scope,
        query_contract_fingerprint=query_contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        lineage_proofs=(
            PopulationLineageProof(
                proof_id="%s-proof" % consumer_goal_id,
                mechanism=PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE,
                verifier_fingerprint=LINEAGE_VERIFIER,
                verified=True,
                graph_fingerprint=GRAPH_FP,
                query_node_id=query_node_id,
                generation=generation,
                attempt_id=attempt_id,
                query_contract_fingerprint=query_contract_fingerprint,
                sql_ast_fingerprint=sql_ast_fingerprint,
                source_population_fingerprint=population_fingerprint,
                result_population_fingerprint=population_fingerprint,
                source_goal_ids=(SOURCE_GOAL,),
                source_node_ids=(query_node_id,),
                preserved_constraints=constraints,
                source_entity_identity_ref="shared-entity-identity",
                result_entity_identity_ref="shared-entity-identity",
                source_grain_fingerprint="shared-grain",
                result_grain_fingerprint="shared-grain",
                source_snapshot_fingerprint=snapshot_fingerprint,
                result_snapshot_fingerprint=snapshot_fingerprint,
                complete_membership=True,
            ),
        ),
    )


def _prepare_two_node_pre_gate():
    coordinator, ledger = _coordinator()
    goal = coordinator.commit_goal_declaration(_two_consumer_goal_command())
    assert goal.committed is True
    nodes = (
        PopulationExecutionNodeBinding(
            query_node_id=CONSUMER_NODE,
            consumer_goal_ids=(CONSUMER_GOAL,),
            generation=4,
            attempt_id="first-node-attempt",
            query_contract_fingerprint="first-query-contract",
            sql_ast_fingerprint="first-sql-ast",
            snapshot_fingerprint="first-snapshot",
        ),
        PopulationExecutionNodeBinding(
            query_node_id=SECOND_CONSUMER_NODE,
            consumer_goal_ids=(SECOND_CONSUMER_GOAL,),
            generation=11,
            attempt_id="second-node-attempt",
            query_contract_fingerprint="second-query-contract",
            sql_ast_fingerprint="second-sql-ast",
            snapshot_fingerprint="second-snapshot",
        ),
    )
    graph = seal_population_execution_graph_binding(
        PopulationExecutionGraphBinding(
            graph_id="two-node-graph",
            graph_version=5,
            graph_fingerprint=GRAPH_FP,
            nodes=nodes,
        )
    )
    claims = tuple(
        _node_specific_claim(
            goal.state.goal_attestation,
            consumer_goal_id=consumer_goal_id,
            query_node_id=node.query_node_id,
            generation=node.generation,
            attempt_id=node.attempt_id,
            query_contract_fingerprint=node.query_contract_fingerprint,
            sql_ast_fingerprint=node.sql_ast_fingerprint,
            snapshot_fingerprint=node.snapshot_fingerprint,
            population_fingerprint=population_fingerprint,
        )
        for node, consumer_goal_id, population_fingerprint in (
            (nodes[0], CONSUMER_GOAL, "first-population"),
            (nodes[1], SECOND_CONSUMER_GOAL, "second-population"),
        )
    )
    pre = coordinator.authorize_pre_execution(
        PopulationPreExecutionCommand(
            gate_id=GATE_ID,
            expected_revision=1,
            goal_contract_fingerprint=GOAL_FP,
            graph_binding=graph,
            claims=claims,
        )
    )
    assert pre.committed is True, pre.model_dump()
    return coordinator, ledger, pre.state, nodes


def _two_node_ledger_snapshot(pre_state, nodes, *, cross_attempts: bool = False):
    scope_map = {
        item.consumer_goal_id: item
        for item in pre_state.pre_execution_attestation.accepted_scopes
    }
    specifications = (
        (nodes[0], CONSUMER_GOAL, "first-ledger-artifact"),
        (nodes[1], SECOND_CONSUMER_GOAL, "second-ledger-artifact"),
    )
    entries = []
    selections = []
    for index, (node, consumer_goal_id, ledger_artifact_id) in enumerate(
        specifications
    ):
        scope = scope_map[consumer_goal_id]
        execution_identity = nodes[1 - index] if cross_attempts else node
        evidence = PopulationArtifactEvidence(
            artifact_id="%s-result" % consumer_goal_id,
            artifact_fingerprint="%s-content" % consumer_goal_id,
            artifact_kind=PopulationArtifactKind.QUERY_RESULT,
            coverage=PopulationArtifactCoverage.TOP_N,
            population_fingerprint=scope.population_fingerprint,
            verifier_fingerprint=ARTIFACT_VERIFIER,
            verified=True,
            immutable=True,
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            query_contract_fingerprint=node.query_contract_fingerprint,
            sql_ast_fingerprint=node.sql_ast_fingerprint,
            snapshot_fingerprint=node.snapshot_fingerprint,
            lineage_proof_fingerprints=scope.proof_fingerprints,
        )
        receipt = seal_population_published_artifact_receipt(
            PopulationPublishedArtifactReceipt(
                ledger_artifact_id=ledger_artifact_id,
                publication_status="PUBLISHED",
                generation=execution_identity.generation,
                attempt_id=execution_identity.attempt_id,
                goal_contract_fingerprint=GOAL_FP,
                graph_fingerprint=GRAPH_FP,
                query_node_id=node.query_node_id,
                covered_consumer_goal_ids=(consumer_goal_id,),
                result_is_truncated=False,
                stored_row_count=3,
                exact_result_row_count=3,
                evidence=evidence,
            )
        )
        entries.append(
            seal_population_artifact_ledger_entry(
                PopulationArtifactLedgerEntry(
                    ledger_artifact_id=ledger_artifact_id,
                    publication_status="PUBLISHED",
                    receipt=receipt,
                )
            )
        )
        selections.append(
            PopulationResultSelection(
                consumer_goal_id=consumer_goal_id,
                query_node_id=node.query_node_id,
                ledger_artifact_id=ledger_artifact_id,
                receipt_fingerprint=receipt.receipt_fingerprint,
            )
        )
    snapshot = seal_population_artifact_ledger_snapshot(
        PopulationArtifactLedgerSnapshot(
            ledger_id="two-node-ledger",
            ledger_authority_fingerprint=LEDGER_AUTHORITY,
            ledger_revision=14,
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            entries=tuple(entries),
        )
    )
    return snapshot, tuple(selections)


def test_three_stage_gate_commits_an_immutable_cas_bound_attestation_chain() -> None:
    coordinator, ledger = _coordinator()

    goal = coordinator.commit_goal_declaration(_goal_command())
    assert goal.committed is True
    assert goal.state.revision == 1
    assert goal.state.phase == PopulationGatePhase.GOAL_DECLARATION.value
    assert goal.state.state_fingerprint == population_gate_state_fingerprint(
        goal.state
    )

    pre = coordinator.authorize_pre_execution(_pre_command(goal.state))
    assert pre.committed is True
    assert pre.state.revision == 2
    assert pre.state.phase == PopulationGatePhase.PRE_EXECUTION.value
    assert pre.state.pre_execution_attestation.previous_attestation_fingerprint == (
        pre.state.goal_attestation.attestation_fingerprint
    )

    ledger.snapshot = _ledger_snapshot(pre.state)
    receipt = ledger.snapshot.entries[0].receipt
    post = coordinator.commit_post_result(
        _post_command(receipt.receipt_fingerprint)
    )

    assert post.committed is True, post.model_dump()
    assert post.state.revision == 3
    assert post.state.phase == PopulationGatePhase.POST_RESULT.value
    assert post.state.post_result_attestation.previous_attestation_fingerprint == (
        post.state.pre_execution_attestation.attestation_fingerprint
    )
    assert post.state.post_result_attestation.attestation_fingerprint == (
        population_attestation_fingerprint(post.state.post_result_attestation)
    )
    assert post.state.ledger_snapshot_fingerprint == (
        ledger.snapshot.snapshot_fingerprint
    )
    assert post.state.published_receipt_fingerprints == (
        receipt.receipt_fingerprint,
    )
    assert ledger.calls == [
        {
            "gate_id": GATE_ID,
            "goal_contract_fingerprint": GOAL_FP,
            "graph_fingerprint": GRAPH_FP,
        }
    ]


def test_two_dynamic_nodes_with_distinct_generations_and_attempts_commit() -> None:
    coordinator, ledger, pre_state, nodes = _prepare_two_node_pre_gate()
    assert nodes[0].generation != nodes[1].generation
    assert nodes[0].attempt_id != nodes[1].attempt_id
    assert "generation" not in pre_state.model_dump()
    assert "attempt_id" not in pre_state.model_dump()
    assert "generation" not in pre_state.graph_binding.model_dump(
        exclude={"nodes"}
    )
    assert "attempt_id" not in pre_state.graph_binding.model_dump(
        exclude={"nodes"}
    )
    ledger.snapshot, selections = _two_node_ledger_snapshot(pre_state, nodes)

    result = coordinator.commit_post_result(
        PopulationPostResultCommand(
            gate_id=GATE_ID,
            expected_revision=2,
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            selections=selections,
        )
    )

    assert result.committed is True, result.model_dump()
    assert len(result.state.published_receipt_fingerprints) == 2


def test_two_dynamic_nodes_reject_cross_bound_generation_and_attempt() -> None:
    coordinator, ledger, pre_state, nodes = _prepare_two_node_pre_gate()
    ledger.snapshot, selections = _two_node_ledger_snapshot(
        pre_state,
        nodes,
        cross_attempts=True,
    )

    result = coordinator.commit_post_result(
        PopulationPostResultCommand(
            gate_id=GATE_ID,
            expected_revision=2,
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            selections=selections,
        )
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.RESULT_BINDING_MISMATCH.value
    assert _issue_codes(result) == {
        PopulationGateCode.RESULT_BINDING_MISMATCH.value
    }
    assert coordinator.get_state(GATE_ID).revision == 2


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("query_contract_fingerprint", "cross-bound-query-contract"),
        ("sql_ast_fingerprint", "cross-bound-sql-ast"),
        ("snapshot_fingerprint", "cross-bound-snapshot"),
    ],
)
def test_post_result_rejects_sealed_evidence_cross_bound_to_another_node(
    field_name: str,
    field_value: str,
) -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    snapshot = _ledger_snapshot(pre_state)
    original_entry = snapshot.entries[0]
    evidence = original_entry.receipt.evidence.model_copy(
        update={field_name: field_value}
    )
    receipt = seal_population_published_artifact_receipt(
        original_entry.receipt.model_copy(
            update={"evidence": evidence, "receipt_fingerprint": ""}
        )
    )
    entry = seal_population_artifact_ledger_entry(
        original_entry.model_copy(
            update={"receipt": receipt, "entry_fingerprint": ""}
        )
    )
    ledger.snapshot = seal_population_artifact_ledger_snapshot(
        snapshot.model_copy(
            update={"entries": (entry,), "snapshot_fingerprint": ""}
        )
    )

    result = coordinator.commit_post_result(
        _post_command(receipt.receipt_fingerprint)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.RESULT_BINDING_MISMATCH.value
    assert PopulationGateCode.RESULT_BINDING_MISMATCH.value in _issue_codes(result)


def test_goal_rejection_never_creates_gate_state() -> None:
    coordinator, _ = _coordinator()

    result = coordinator.commit_goal_declaration(
        _goal_command(semantic_verifier="untrusted-semantic-authority")
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.VERIFICATION_REJECTED.value
    assert coordinator.get_state(GATE_ID) is None


def test_concurrent_goal_commits_have_exactly_one_cas_winner() -> None:
    coordinator, _ = _coordinator()
    command = _goal_command()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(coordinator.commit_goal_declaration, (command, command)))

    assert sum(item.committed for item in results) == 1
    assert coordinator.get_state(GATE_ID).revision == 1


@pytest.mark.parametrize(
    ("override", "expected_code"),
    [
        ({"expected_revision": 0}, PopulationGateCode.CAS_REVISION_MISMATCH),
        (
            {"goal_fingerprint": "other-goal-contract"},
            PopulationGateCode.BINDING_MISMATCH,
        ),
    ],
)
def test_pre_execution_is_cas_bound_to_goal_state(override, expected_code) -> None:
    coordinator, _ = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())

    result = coordinator.authorize_pre_execution(
        _pre_command(goal.state, **override)
    )

    assert result.committed is False
    assert result.code == expected_code.value
    assert coordinator.get_state(GATE_ID).revision == 1


@pytest.mark.parametrize(
    ("graph_override", "claim_override"),
    [
        ({"consumer_goal_ids": (SOURCE_GOAL,)}, {}),
        (
            {},
            {"query_contract_fingerprint": "different-query-contract"},
        ),
        ({}, {"generation": GENERATION + 1}),
        ({}, {"attempt_id": "different-attempt"}),
        ({}, {"sql_ast_fingerprint": "different-sql-ast"}),
        (
            {"snapshot_fingerprint": "different-snapshot"},
            {},
        ),
        ({}, {"query_node_id": "missing-query-node"}),
    ],
)
def test_pre_execution_rejects_claims_not_bound_to_the_structured_graph(
    graph_override,
    claim_override,
) -> None:
    coordinator, _ = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())
    graph = _graph_binding(**graph_override)
    claim = _claim(goal.state.goal_attestation, **claim_override)

    result = coordinator.authorize_pre_execution(
        _pre_command(goal.state, graph_binding=graph, claim=claim)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.GRAPH_BINDING_INVALID.value
    assert PopulationGateCode.GRAPH_BINDING_INVALID.value in _issue_codes(result)


def test_pre_execution_rejects_a_tampered_graph_binding_fingerprint() -> None:
    coordinator, _ = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())
    graph = _graph_binding().model_copy(
        update={"graph_version": 99},
    )

    result = coordinator.authorize_pre_execution(
        _pre_command(goal.state, graph_binding=graph)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.GRAPH_BINDING_INVALID.value


@pytest.mark.parametrize(
    "proof_override",
    [
        {"graph_fingerprint": "replayed-graph"},
        {"query_node_id": "replayed-node"},
        {"generation": GENERATION + 1},
        {"attempt_id": "replayed-attempt"},
        {"query_contract_fingerprint": "replayed-query-contract"},
        {"sql_ast_fingerprint": "replayed-sql-ast"},
        {"source_snapshot_fingerprint": "replayed-source-snapshot"},
        {"result_snapshot_fingerprint": "replayed-result-snapshot"},
    ],
)
def test_coordinator_rejects_lineage_proof_replay_before_execution(
    proof_override,
) -> None:
    coordinator, _ = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())
    claim = _claim(goal.state.goal_attestation)
    replayed_proof = claim.lineage_proofs[0].model_copy(
        update=proof_override
    )
    replayed_claim = claim.model_copy(
        update={"lineage_proofs": (replayed_proof,)}
    )

    result = coordinator.authorize_pre_execution(
        _pre_command(goal.state, claim=replayed_claim)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.GRAPH_BINDING_INVALID.value
    assert PopulationGateCode.GRAPH_BINDING_INVALID.value in _issue_codes(result)


def test_pre_execution_independent_verifier_blocks_time_only_population_downgrade() -> None:
    coordinator, _ = _coordinator()
    goal = coordinator.commit_goal_declaration(_goal_command())
    time_only = (
        PopulationConstraintEvidence(
            fingerprint="time-constraint",
            kind=PopulationConstraintKind.TIME,
        ),
    )
    downgraded = _resolved_scope(
        population_fingerprint="time-only-population",
        entity_identity_ref="different-identity",
        constraints=time_only,
    )
    claim = _claim(
        goal.state.goal_attestation,
        effective_scope=downgraded,
        preserved_constraints=time_only,
    )

    result = coordinator.authorize_pre_execution(
        _pre_command(goal.state, claim=claim)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.VERIFICATION_REJECTED.value
    assert result.state.revision == 1


@pytest.mark.parametrize(
    ("snapshot_override", "expected_issue"),
    [
        (
            {"publication_status": "PREVIEW"},
            PopulationGateCode.RESULT_NOT_PUBLISHED,
        ),
        (
            {"coverage": PopulationArtifactCoverage.PREVIEW},
            PopulationGateCode.RESULT_COVERAGE_INCOMPLETE,
        ),
        (
            {"coverage": PopulationArtifactCoverage.PARTIAL},
            PopulationGateCode.RESULT_COVERAGE_INCOMPLETE,
        ),
        ({"verified": False}, PopulationGateCode.RESULT_NOT_VERIFIED),
        ({"immutable": False}, PopulationGateCode.RESULT_MUTABLE),
        ({"truncated": True}, PopulationGateCode.RESULT_TRUNCATED),
        (
            {"exact_result_row_count": 4},
            PopulationGateCode.RESULT_COUNT_MISMATCH,
        ),
        (
            {"generation": GENERATION + 1},
            PopulationGateCode.RESULT_BINDING_MISMATCH,
        ),
        (
            {"attempt_id": "other-attempt"},
            PopulationGateCode.RESULT_BINDING_MISMATCH,
        ),
        (
            {"goal_fingerprint": "other-goal"},
            PopulationGateCode.LEDGER_BINDING_MISMATCH,
        ),
        (
            {"graph_fingerprint": "other-graph"},
            PopulationGateCode.LEDGER_BINDING_MISMATCH,
        ),
        (
            {"query_node_id": SOURCE_NODE},
            PopulationGateCode.RESULT_BINDING_MISMATCH,
        ),
        (
            {"covered_goal_ids": (SOURCE_GOAL,)},
            PopulationGateCode.RESULT_BINDING_MISMATCH,
        ),
    ],
)
def test_post_result_fails_closed_for_nonfinal_or_misbound_ledger_artifacts(
    snapshot_override,
    expected_issue,
) -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    snapshot = _ledger_snapshot(pre_state, **snapshot_override)

    result = _post_from_snapshot(coordinator, ledger, snapshot)

    assert result.committed is False
    assert expected_issue.value in _issue_codes(result)
    assert coordinator.get_state(GATE_ID).revision == 2


def test_post_result_rejects_untrusted_injected_ledger_authority_before_read() -> None:
    reader = _LedgerReader(authority_fingerprint="untrusted-ledger")
    coordinator, _ = _coordinator(reader)
    goal = coordinator.commit_goal_declaration(_goal_command())
    pre = coordinator.authorize_pre_execution(_pre_command(goal.state))
    reader.snapshot = _ledger_snapshot(pre.state)

    result = coordinator.commit_post_result(
        _post_command(
            reader.snapshot.entries[0].receipt.receipt_fingerprint
        )
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.LEDGER_AUTHORITY_UNTRUSTED.value
    assert reader.calls == []


def test_post_result_rejects_unledgered_and_wrong_receipt_selections() -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    ledger.snapshot = _ledger_snapshot(pre_state)

    unledgered = coordinator.commit_post_result(
        _post_command(
            "unledgered-receipt",
            ledger_artifact_id="unledgered-artifact",
        )
    )
    wrong_receipt = coordinator.commit_post_result(
        _post_command("wrong-receipt-fingerprint")
    )

    assert unledgered.committed is False
    assert unledgered.code == PopulationGateCode.RESULT_NOT_IN_LEDGER.value
    assert wrong_receipt.committed is False
    assert wrong_receipt.code == PopulationGateCode.RESULT_RECEIPT_MISMATCH.value


def test_post_result_rejects_tampered_snapshot_entry_or_receipt() -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    snapshot = _ledger_snapshot(pre_state)
    receipt = snapshot.entries[0].receipt.model_copy(
        update={"stored_row_count": 2}
    )
    entry = snapshot.entries[0].model_copy(update={"receipt": receipt})
    ledger.snapshot = snapshot.model_copy(update={"entries": (entry,)})

    result = coordinator.commit_post_result(
        _post_command(receipt.receipt_fingerprint)
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.LEDGER_SNAPSHOT_INVALID.value


def test_post_result_rejects_a_tampered_ledger_snapshot_fingerprint() -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    ledger.snapshot = _ledger_snapshot(pre_state).model_copy(
        update={"ledger_revision": 10}
    )

    result = coordinator.commit_post_result(
        _post_command(
            ledger.snapshot.entries[0].receipt.receipt_fingerprint
        )
    )

    assert result.committed is False
    assert result.code == PopulationGateCode.LEDGER_SNAPSHOT_INVALID.value
    assert ledger.snapshot.snapshot_fingerprint != (
        population_artifact_ledger_snapshot_fingerprint(ledger.snapshot)
    )


def test_post_result_is_cas_bound_and_cannot_be_replayed() -> None:
    coordinator, ledger, pre_state = _prepare_pre_gate()
    ledger.snapshot = _ledger_snapshot(pre_state)
    command = _post_command(
        ledger.snapshot.entries[0].receipt.receipt_fingerprint
    )

    first = coordinator.commit_post_result(command)
    replay = coordinator.commit_post_result(command)

    assert first.committed is True
    assert replay.committed is False
    assert replay.code == PopulationGateCode.CAS_REVISION_MISMATCH.value
    assert coordinator.get_state(GATE_ID).revision == 3


def test_coordinator_source_contains_no_regex_or_sql_text_interpretation() -> None:
    source_path = Path(
        "python_backend/merchant_ai/services/grounded_population_gate_coordinator.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"re", "regex"}
    ]
    string_field_names = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"sql", "sql_text", "query_text"}
    }

    assert "re" not in imported_roots
    assert "regex" not in imported_roots
    assert calls == []
    assert string_field_names == set()
