from __future__ import annotations

import ast
from pathlib import Path

import pytest

from merchant_ai.services.grounded_goal_contract import (
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
)
from merchant_ai.services.grounded_population_verifier import (
    GoalDeclarationPopulationVerificationInput,
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationConstraintEvidence,
    PopulationConstraintKind,
    PopulationDeclaration,
    PopulationExecutionClaim,
    PopulationGapCode,
    PopulationLineageMechanism,
    PopulationLineageProof,
    PopulationResultEvidence,
    PopulationScopeDescriptor,
    PopulationScopeKind,
    PopulationSemanticExpectation,
    PopulationSemanticReview,
    PopulationSemanticVerifier,
    PostResultPopulationVerificationInput,
    PreExecutionPopulationVerificationInput,
    goal_population_verification_input,
    population_attestation_fingerprint,
    population_declaration_scope_fingerprint,
    population_declarations_from_goal_contract,
    population_question_fingerprint,
)


QUESTION = "A detail result followed by a ranking over that result"
QUESTION_FP = population_question_fingerprint(QUESTION)
GOAL_FP = "goal-contract-fingerprint"
GRAPH_FP = "query-graph-fingerprint"
SEMANTIC_VERIFIER = "semantic-verifier-build"
LINEAGE_VERIFIER = "sql-ast-lineage-verifier-build"
ARTIFACT_VERIFIER = "immutable-artifact-verifier-build"
CORE_AUTHOR = "core-declaration-build"
CONSUMER_GOAL = "ranking.result"
SOURCE_GOAL = "detail.rows"
POPULATION_FP = "full-population-fingerprint"
QUERY_CONTRACT_FP = "query-contract-fingerprint"
SQL_AST_FP = "validated-sql-ast-fingerprint"
SNAPSHOT_FP = "data-snapshot-fingerprint"


def _declaration_scope(
    *,
    kind: PopulationScopeKind = PopulationScopeKind.SAME_AS_GOAL,
    source_goal_ids: tuple[str, ...] = (SOURCE_GOAL,),
    source_artifact_ids: tuple[str, ...] = (),
    complete: bool = True,
) -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="declared-scope",
        kind=kind,
        source_goal_ids=source_goal_ids,
        source_artifact_ids=source_artifact_ids,
        complete_membership_required=complete,
    )


def _semantic_review(
    scope: PopulationScopeDescriptor | None = None,
    *,
    verifier_fingerprint: str = SEMANTIC_VERIFIER,
    complete: bool = True,
) -> PopulationSemanticReview:
    expected = scope or _declaration_scope()
    return PopulationSemanticReview(
        review_id="review-1",
        question_fingerprint=QUESTION_FP,
        verifier_fingerprint=verifier_fingerprint,
        complete=complete,
        expectations=(
            PopulationSemanticExpectation(
                expectation_id="expectation-1",
                consumer_goal_id=CONSUMER_GOAL,
                expected_scope=expected,
            ),
        ),
    )


def _goal_gate(
    declaration_scope: PopulationScopeDescriptor | None = None,
    *,
    review_scope: PopulationScopeDescriptor | None = None,
    review_verifier: str = SEMANTIC_VERIFIER,
):
    scope = declaration_scope or _declaration_scope()
    return PopulationSemanticVerifier().verify_goal_declaration(
        GoalDeclarationPopulationVerificationInput(
            question_fingerprint=QUESTION_FP,
            goal_contract_fingerprint=GOAL_FP,
            declaration_author_fingerprint=CORE_AUTHOR,
            semantic_review=_semantic_review(
                review_scope,
                verifier_fingerprint=review_verifier,
            ),
            trusted_semantic_verifier_fingerprints=(SEMANTIC_VERIFIER,),
            declarations=(
                PopulationDeclaration(
                    consumer_goal_id=CONSUMER_GOAL,
                    declared_scope=scope,
                ),
            ),
        )
    )


def _resolved_scope(
    *,
    kind: PopulationScopeKind = PopulationScopeKind.SAME_AS_GOAL,
    population_fingerprint: str = POPULATION_FP,
    entity_identity_ref: str = "entity.identity",
    grain_fingerprint: str = "entity-grain",
    constraints: tuple[PopulationConstraintEvidence, ...] | None = None,
    source_artifact_ids: tuple[str, ...] = (),
    complete: bool = True,
) -> PopulationScopeDescriptor:
    return PopulationScopeDescriptor(
        scope_id="resolved-scope",
        kind=kind,
        source_goal_ids=(SOURCE_GOAL,)
        if kind == PopulationScopeKind.SAME_AS_GOAL
        else (),
        source_artifact_ids=source_artifact_ids,
        population_fingerprint=population_fingerprint,
        entity_identity_ref=entity_identity_ref,
        grain_fingerprint=grain_fingerprint,
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
        complete_membership_required=complete,
    )


def _lineage_proof(
    *,
    mechanism: PopulationLineageMechanism = (
        PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE
    ),
    source_scope: PopulationScopeDescriptor | None = None,
    result_scope: PopulationScopeDescriptor | None = None,
    preserved_constraints: tuple[PopulationConstraintEvidence, ...] | None = None,
    artifacts: tuple[PopulationArtifactEvidence, ...] = (),
) -> PopulationLineageProof:
    required = source_scope or _resolved_scope()
    effective = result_scope or required
    return PopulationLineageProof(
        proof_id="proof-1",
        mechanism=mechanism,
        verifier_fingerprint=LINEAGE_VERIFIER,
        verified=True,
        source_population_fingerprint=required.population_fingerprint,
        result_population_fingerprint=effective.population_fingerprint,
        source_goal_ids=required.source_goal_ids,
        source_node_ids=("node-detail",),
        preserved_constraints=preserved_constraints
        if preserved_constraints is not None
        else required.constraints,
        artifact_evidence=artifacts,
        source_entity_identity_ref=required.entity_identity_ref,
        result_entity_identity_ref=effective.entity_identity_ref,
        source_grain_fingerprint=required.grain_fingerprint,
        result_grain_fingerprint=effective.grain_fingerprint,
        source_snapshot_fingerprint=required.snapshot_fingerprint,
        result_snapshot_fingerprint=effective.snapshot_fingerprint,
        complete_membership=True,
    )


def _pre_gate(
    goal_result,
    *,
    required_scope: PopulationScopeDescriptor | None = None,
    effective_scope: PopulationScopeDescriptor | None = None,
    proof: PopulationLineageProof | None = None,
):
    required = required_scope or _resolved_scope()
    effective = effective_scope or required
    lineage = proof or _lineage_proof(
        source_scope=required,
        result_scope=effective,
    )
    declaration_fp = (
        goal_result.attestation.accepted_scopes[0].declaration_scope_fingerprint
    )
    return PopulationSemanticVerifier().verify_pre_execution(
        PreExecutionPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            declaration_attestation=goal_result.attestation,
            trusted_lineage_verifier_fingerprints=(LINEAGE_VERIFIER,),
            trusted_artifact_verifier_fingerprints=(ARTIFACT_VERIFIER,),
            claims=(
                PopulationExecutionClaim(
                    consumer_goal_id=CONSUMER_GOAL,
                    query_node_id="node-ranking",
                    declaration_scope_fingerprint=declaration_fp,
                    required_scope=required,
                    effective_scope=effective,
                    query_contract_fingerprint=QUERY_CONTRACT_FP,
                    sql_ast_fingerprint=SQL_AST_FP,
                    lineage_proofs=(lineage,),
                ),
            ),
        )
    )


def _result_artifact(
    *,
    coverage: PopulationArtifactCoverage = PopulationArtifactCoverage.TOP_N,
    population_fingerprint: str = POPULATION_FP,
    lineage_proof_fingerprints: tuple[str, ...] = (),
) -> PopulationArtifactEvidence:
    return PopulationArtifactEvidence(
        artifact_id="result-artifact",
        artifact_fingerprint="result-artifact-content-address",
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=coverage,
        population_fingerprint=population_fingerprint,
        verifier_fingerprint=ARTIFACT_VERIFIER,
        verified=True,
        immutable=True,
        goal_contract_fingerprint=GOAL_FP,
        graph_fingerprint=GRAPH_FP,
        query_contract_fingerprint=QUERY_CONTRACT_FP,
        sql_ast_fingerprint=SQL_AST_FP,
        snapshot_fingerprint=SNAPSHOT_FP,
        lineage_proof_fingerprints=lineage_proof_fingerprints,
    )


def _gap_codes(result) -> set[str]:
    return {str(gap.code) for gap in result.gaps}


def test_goal_declaration_accepts_independently_verified_same_goal_population() -> None:
    result = _goal_gate()

    assert result.passed is True
    assert result.gate_open is True
    assert result.attestation.accepted_scopes[0].scope_kind == "SAME_AS_GOAL"
    assert result.attestation.attestation_fingerprint == (
        population_attestation_fingerprint(result.attestation)
    )


def test_goal_declaration_blocks_dependent_population_downgrade_to_independent() -> None:
    result = _goal_gate(
        _declaration_scope(
            kind=PopulationScopeKind.INDEPENDENT,
            source_goal_ids=(),
            complete=False,
        )
    )

    assert result.passed is False
    assert PopulationGapCode.POPULATION_SCOPE_KIND_MISMATCH.value in _gap_codes(
        result
    )
    assert PopulationGapCode.POPULATION_SOURCE_GOAL_MISMATCH.value in _gap_codes(
        result
    )


def test_goal_declaration_requires_a_trusted_distinct_semantic_authority() -> None:
    result = _goal_gate(review_verifier=CORE_AUTHOR)

    assert result.passed is False
    assert _gap_codes(result) == {
        PopulationGapCode.SEMANTIC_REVIEW_UNTRUSTED.value,
        PopulationGapCode.SEMANTIC_REVIEW_NOT_INDEPENDENT.value,
    }


@pytest.mark.parametrize(
    "mechanism",
    [
        PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
        PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE,
        PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE,
    ],
)
def test_pre_execution_accepts_verified_same_query_population_lineage(
    mechanism: PopulationLineageMechanism,
) -> None:
    goal_result = _goal_gate()
    required = _resolved_scope()
    result = _pre_gate(
        goal_result,
        required_scope=required,
        proof=_lineage_proof(
            mechanism=mechanism,
            source_scope=required,
            result_scope=required,
        ),
    )

    assert result.passed is True
    assert result.gate_open is True
    assert result.attestation.accepted_scopes[0].population_fingerprint == (
        POPULATION_FP
    )
    assert result.attestation.accepted_proof_fingerprints


def test_pre_execution_blocks_time_filter_copy_that_loses_entity_population() -> None:
    goal_result = _goal_gate()
    required = _resolved_scope()
    effective = _resolved_scope(
        population_fingerprint="time-filter-only-population",
        entity_identity_ref="different.entity.identity",
    )
    time_only = (
        PopulationConstraintEvidence(
            fingerprint="time-constraint",
            kind=PopulationConstraintKind.TIME,
        ),
    )
    proof = _lineage_proof(
        mechanism=PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE,
        source_scope=required,
        result_scope=effective,
        preserved_constraints=time_only,
    )

    result = _pre_gate(
        goal_result,
        required_scope=required,
        effective_scope=effective,
        proof=proof,
    )

    assert result.passed is False
    assert PopulationGapCode.POPULATION_DEGRADED_TO_TIME_FILTER.value in _gap_codes(
        result
    )
    assert PopulationGapCode.CONSTRAINT_LINEAGE_INCOMPLETE.value in _gap_codes(
        result
    )
    assert PopulationGapCode.ENTITY_MAPPING_REQUIRED.value in _gap_codes(result)


def test_pre_execution_accepts_verified_entity_set_artifact() -> None:
    source_artifact_id = "verified-entity-set"
    declaration = _declaration_scope(
        kind=PopulationScopeKind.VERIFIED_ENTITY_SET,
        source_goal_ids=(),
        source_artifact_ids=(source_artifact_id,),
        complete=True,
    )
    goal_result = _goal_gate(declaration, review_scope=declaration)
    required = _resolved_scope(
        kind=PopulationScopeKind.VERIFIED_ENTITY_SET,
        source_artifact_ids=(source_artifact_id,),
    )
    artifact = PopulationArtifactEvidence(
        artifact_id=source_artifact_id,
        artifact_fingerprint="entity-set-content-address",
        artifact_kind=PopulationArtifactKind.ENTITY_SET,
        coverage=PopulationArtifactCoverage.EXACT_ENTITY_SET,
        population_fingerprint=POPULATION_FP,
        verifier_fingerprint=ARTIFACT_VERIFIER,
        verified=True,
        immutable=True,
        goal_contract_fingerprint="source-goal-contract",
        graph_fingerprint="source-query-graph",
        query_contract_fingerprint="source-query-contract",
        sql_ast_fingerprint="source-sql-ast",
        snapshot_fingerprint=SNAPSHOT_FP,
    )
    proof = _lineage_proof(
        mechanism=PopulationLineageMechanism.VERIFIED_ENTITY_SET_ARTIFACT,
        source_scope=required,
        result_scope=required,
        artifacts=(artifact,),
    )

    result = _pre_gate(goal_result, required_scope=required, proof=proof)

    assert result.passed is True, [gap.model_dump() for gap in result.gaps]
    assert result.attestation.artifact_fingerprints == (
        "entity-set-content-address",
    )


def test_pre_execution_rejects_preview_result_as_population_artifact() -> None:
    source_artifact_id = "verified-result"
    declaration = _declaration_scope(
        kind=PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
        source_goal_ids=(),
        source_artifact_ids=(source_artifact_id,),
        complete=True,
    )
    goal_result = _goal_gate(declaration, review_scope=declaration)
    required = _resolved_scope(
        kind=PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
        source_artifact_ids=(source_artifact_id,),
    )
    preview = PopulationArtifactEvidence(
        artifact_id=source_artifact_id,
        artifact_fingerprint="preview-content-address",
        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
        coverage=PopulationArtifactCoverage.PREVIEW,
        population_fingerprint=POPULATION_FP,
        verifier_fingerprint=ARTIFACT_VERIFIER,
        verified=True,
        immutable=True,
    )
    proof = _lineage_proof(
        mechanism=PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT,
        source_scope=required,
        result_scope=required,
        artifacts=(preview,),
    )

    result = _pre_gate(goal_result, required_scope=required, proof=proof)

    assert result.passed is False
    assert PopulationGapCode.ARTIFACT_COVERAGE_INCOMPLETE.value in _gap_codes(
        result
    )


def test_pre_execution_accepts_complete_verified_result_artifact() -> None:
    source_artifact_id = "verified-result"
    declaration = _declaration_scope(
        kind=PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
        source_goal_ids=(),
        source_artifact_ids=(source_artifact_id,),
        complete=True,
    )
    goal_result = _goal_gate(declaration, review_scope=declaration)
    required = _resolved_scope(
        kind=PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
        source_artifact_ids=(source_artifact_id,),
    )
    artifact = PopulationArtifactEvidence(
        artifact_id=source_artifact_id,
        artifact_fingerprint="complete-result-content-address",
        artifact_kind=PopulationArtifactKind.RESULT_RELATION,
        coverage=PopulationArtifactCoverage.COMPLETE,
        population_fingerprint=POPULATION_FP,
        verifier_fingerprint=ARTIFACT_VERIFIER,
        verified=True,
        immutable=True,
    )
    proof = _lineage_proof(
        mechanism=PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT,
        source_scope=required,
        result_scope=required,
        artifacts=(artifact,),
    )

    result = _pre_gate(goal_result, required_scope=required, proof=proof)

    assert result.passed is True, [gap.model_dump() for gap in result.gaps]


def test_post_result_accepts_top_n_artifact_bound_to_pre_execution_attestation() -> None:
    goal_result = _goal_gate()
    pre_result = _pre_gate(goal_result)
    result = PopulationSemanticVerifier().verify_post_result(
        PostResultPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            pre_execution_attestation=pre_result.attestation,
            trusted_artifact_verifier_fingerprints=(ARTIFACT_VERIFIER,),
            results=(
                PopulationResultEvidence(
                    consumer_goal_id=CONSUMER_GOAL,
                    query_node_id="node-ranking",
                    result_artifact=_result_artifact(
                        lineage_proof_fingerprints=(
                            pre_result.attestation.accepted_proof_fingerprints
                        )
                    ),
                    lineage_proof_fingerprints=(
                        pre_result.attestation.accepted_proof_fingerprints
                    ),
                ),
            ),
        )
    )

    assert result.passed is True, [gap.model_dump() for gap in result.gaps]
    assert result.attestation.previous_attestation_fingerprint == (
        pre_result.attestation.attestation_fingerprint
    )


def test_post_result_blocks_preview_even_when_query_and_population_match() -> None:
    goal_result = _goal_gate()
    pre_result = _pre_gate(goal_result)
    result = PopulationSemanticVerifier().verify_post_result(
        PostResultPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            pre_execution_attestation=pre_result.attestation,
            trusted_artifact_verifier_fingerprints=(ARTIFACT_VERIFIER,),
            results=(
                PopulationResultEvidence(
                    consumer_goal_id=CONSUMER_GOAL,
                    query_node_id="node-ranking",
                    result_artifact=_result_artifact(
                        coverage=PopulationArtifactCoverage.PREVIEW,
                        lineage_proof_fingerprints=(
                            pre_result.attestation.accepted_proof_fingerprints
                        ),
                    ),
                    lineage_proof_fingerprints=(
                        pre_result.attestation.accepted_proof_fingerprints
                    ),
                ),
            ),
        )
    )

    assert result.passed is False
    assert PopulationGapCode.ARTIFACT_COVERAGE_INCOMPLETE.value in _gap_codes(
        result
    )


def test_post_result_blocks_missing_pre_execution_lineage_binding() -> None:
    goal_result = _goal_gate()
    pre_result = _pre_gate(goal_result)
    result = PopulationSemanticVerifier().verify_post_result(
        PostResultPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            pre_execution_attestation=pre_result.attestation,
            trusted_artifact_verifier_fingerprints=(ARTIFACT_VERIFIER,),
            results=(
                PopulationResultEvidence(
                    consumer_goal_id=CONSUMER_GOAL,
                    query_node_id="node-ranking",
                    result_artifact=_result_artifact(
                        lineage_proof_fingerprints=("different-proof",)
                    ),
                    lineage_proof_fingerprints=("different-proof",),
                ),
            ),
        )
    )

    assert result.passed is False
    assert PopulationGapCode.RESULT_LINEAGE_ATTESTATION_MISMATCH.value in (
        _gap_codes(result)
    )


def test_goal_contract_adapter_preserves_typed_population_dependency() -> None:
    contract = parse_original_question_goal_contract(
        {
            "question": QUESTION,
            "goals": [
                {
                    "goalId": SOURCE_GOAL,
                    "kind": "DETAIL",
                    "label": "source rows",
                },
                {
                    "goalId": "metric.value",
                    "kind": "METRIC",
                    "label": "rank metric",
                },
                {
                    "goalId": CONSUMER_GOAL,
                    "kind": "RANKING",
                    "label": "rank source rows",
                    "metricGoalIds": ["metric.value"],
                    "limit": 3,
                    "populationScope": "SAME_AS_GOAL",
                    "populationGoalIds": [SOURCE_GOAL],
                },
            ],
        }
    )
    declarations = population_declarations_from_goal_contract(contract)
    scope = declarations[0].declared_scope
    review = PopulationSemanticReview(
        review_id="review-adapter",
        question_fingerprint=population_question_fingerprint(contract.question),
        verifier_fingerprint=SEMANTIC_VERIFIER,
        complete=True,
        expectations=(
            PopulationSemanticExpectation(
                expectation_id="expectation-adapter",
                consumer_goal_id=CONSUMER_GOAL,
                expected_scope=scope,
            ),
        ),
    )

    request = goal_population_verification_input(
        contract,
        semantic_review=review,
        declaration_author_fingerprint=CORE_AUTHOR,
        trusted_semantic_verifier_fingerprints=(SEMANTIC_VERIFIER,),
    )
    result = PopulationSemanticVerifier().verify(request)

    assert request.goal_contract_fingerprint == (
        original_question_goal_contract_fingerprint(contract)
    )
    assert scope.kind == "SAME_AS_GOAL"
    assert scope.source_goal_ids == (SOURCE_GOAL,)
    assert result.passed is True


def test_universe_and_independent_population_kinds_are_not_interchangeable() -> None:
    universe = _declaration_scope(
        kind=PopulationScopeKind.UNIVERSE,
        source_goal_ids=(),
        complete=False,
    )
    independent = _declaration_scope(
        kind=PopulationScopeKind.INDEPENDENT,
        source_goal_ids=(),
        complete=False,
    )

    result = _goal_gate(independent, review_scope=universe)

    assert result.passed is False
    assert _gap_codes(result) == {
        PopulationGapCode.POPULATION_SCOPE_KIND_MISMATCH.value
    }


def test_changed_prior_attestation_is_rejected_before_execution() -> None:
    goal_result = _goal_gate()
    altered = goal_result.attestation.model_copy(update={"accepted_scopes": ()})

    result = PopulationSemanticVerifier().verify_pre_execution(
        PreExecutionPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            declaration_attestation=altered,
            trusted_lineage_verifier_fingerprints=(LINEAGE_VERIFIER,),
        )
    )

    assert result.passed is False
    assert PopulationGapCode.PRIOR_ATTESTATION_INVALID.value in _gap_codes(result)


def test_caller_cannot_skip_an_attested_population_gate() -> None:
    goal_result = _goal_gate()

    result = PopulationSemanticVerifier().verify_pre_execution(
        PreExecutionPopulationVerificationInput(
            goal_contract_fingerprint=GOAL_FP,
            graph_fingerprint=GRAPH_FP,
            declaration_attestation=goal_result.attestation,
            trusted_lineage_verifier_fingerprints=(LINEAGE_VERIFIER,),
            required_consumer_goal_ids=("different.goal",),
        )
    )

    assert result.passed is False
    assert PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH.value in (
        _gap_codes(result)
    )


def test_declaration_fingerprint_is_order_independent_for_source_refs() -> None:
    left = PopulationScopeDescriptor(
        scope_id="left",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=("goal.a", "goal.b"),
        source_artifact_ids=("artifact.a", "artifact.b"),
        complete_membership_required=True,
    )
    right = PopulationScopeDescriptor(
        scope_id="right",
        kind=PopulationScopeKind.SAME_AS_GOAL,
        source_goal_ids=("goal.b", "goal.a"),
        source_artifact_ids=("artifact.b", "artifact.a"),
        complete_membership_required=True,
    )

    assert population_declaration_scope_fingerprint(left) == (
        population_declaration_scope_fingerprint(right)
    )


def test_population_verifier_source_has_no_regular_expression_dependency() -> None:
    source_path = Path(
        "python_backend/merchant_ai/services/grounded_population_verifier.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    blocked_modules = {"re", "regex"}
    imported_modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        str(node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    blocked_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in blocked_modules
    ]

    assert imported_modules.isdisjoint(blocked_modules)
    assert blocked_calls == []
