from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Sequence, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    original_question_goal_contract_fingerprint,
    parse_original_question_goal_contract,
)


class PopulationVerificationStage(str, Enum):
    GOAL_DECLARATION = "GOAL_DECLARATION"
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_RESULT = "POST_RESULT"


class PopulationScopeKind(str, Enum):
    """Closed structural population kinds; business universes remain asset-defined."""

    UNIVERSE = "UNIVERSE"
    INDEPENDENT = "INDEPENDENT"
    SAME_AS_GOAL = "SAME_AS_GOAL"
    VERIFIED_ENTITY_SET = "VERIFIED_ENTITY_SET"
    PREDICATE_SCOPE = "PREDICATE_SCOPE"
    VERIFIED_RESULT_ARTIFACT = "VERIFIED_RESULT_ARTIFACT"


class PopulationConstraintKind(str, Enum):
    TIME = "TIME"
    PREDICATE = "PREDICATE"
    ENTITY_MEMBERSHIP = "ENTITY_MEMBERSHIP"
    RELATION = "RELATION"
    GOVERNED_SCOPE = "GOVERNED_SCOPE"


class PopulationLineageMechanism(str, Enum):
    DIRECT_SCOPE = "DIRECT_SCOPE"
    SAME_QUERY_PREDICATE_LINEAGE = "SAME_QUERY_PREDICATE_LINEAGE"
    SAME_QUERY_CTE_LINEAGE = "SAME_QUERY_CTE_LINEAGE"
    SAME_QUERY_SEMI_JOIN_LINEAGE = "SAME_QUERY_SEMI_JOIN_LINEAGE"
    VERIFIED_ENTITY_SET_ARTIFACT = "VERIFIED_ENTITY_SET_ARTIFACT"
    VERIFIED_RESULT_ARTIFACT = "VERIFIED_RESULT_ARTIFACT"


class PopulationArtifactKind(str, Enum):
    ENTITY_SET = "ENTITY_SET"
    QUERY_RESULT = "QUERY_RESULT"
    RESULT_RELATION = "RESULT_RELATION"


class PopulationArtifactCoverage(str, Enum):
    ALL_ROWS = "ALL_ROWS"
    COMPLETE = "COMPLETE"
    EXACT_ENTITY_SET = "EXACT_ENTITY_SET"
    TOP_N = "TOP_N"
    PREVIEW = "PREVIEW"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class PopulationGapCode(str, Enum):
    SEMANTIC_REVIEW_REQUIRED = "SEMANTIC_REVIEW_REQUIRED"
    SEMANTIC_REVIEW_UNTRUSTED = "SEMANTIC_REVIEW_UNTRUSTED"
    SEMANTIC_REVIEW_NOT_INDEPENDENT = "SEMANTIC_REVIEW_NOT_INDEPENDENT"
    SEMANTIC_REVIEW_INCOMPLETE = "SEMANTIC_REVIEW_INCOMPLETE"
    SEMANTIC_REVIEW_BINDING_MISSING = "SEMANTIC_REVIEW_BINDING_MISSING"
    GOAL_SKELETON_FINGERPRINT_MISMATCH = "GOAL_SKELETON_FINGERPRINT_MISMATCH"
    DECLARATION_AUTHORITY_REQUIRED = "DECLARATION_AUTHORITY_REQUIRED"
    QUESTION_FINGERPRINT_MISMATCH = "QUESTION_FINGERPRINT_MISMATCH"
    GOAL_CONTRACT_FINGERPRINT_REQUIRED = "GOAL_CONTRACT_FINGERPRINT_REQUIRED"
    POPULATION_DECLARATION_MISSING = "POPULATION_DECLARATION_MISSING"
    POPULATION_DECLARATION_UNEXPECTED = "POPULATION_DECLARATION_UNEXPECTED"
    POPULATION_DECLARATION_DUPLICATE = "POPULATION_DECLARATION_DUPLICATE"
    POPULATION_SCOPE_KIND_MISMATCH = "POPULATION_SCOPE_KIND_MISMATCH"
    POPULATION_SOURCE_GOAL_MISMATCH = "POPULATION_SOURCE_GOAL_MISMATCH"
    POPULATION_SOURCE_ARTIFACT_MISMATCH = "POPULATION_SOURCE_ARTIFACT_MISMATCH"
    POPULATION_SCOPE_FINGERPRINT_MISMATCH = "POPULATION_SCOPE_FINGERPRINT_MISMATCH"
    POPULATION_ENTITY_IDENTITY_MISMATCH = "POPULATION_ENTITY_IDENTITY_MISMATCH"
    POPULATION_GRAIN_MISMATCH = "POPULATION_GRAIN_MISMATCH"
    POPULATION_GRAIN_REQUIRED = "POPULATION_GRAIN_REQUIRED"
    POPULATION_ENTITY_IDENTITY_REQUIRED = "POPULATION_ENTITY_IDENTITY_REQUIRED"
    POPULATION_COMPLETENESS_MISMATCH = "POPULATION_COMPLETENESS_MISMATCH"
    POPULATION_SOURCE_REQUIRED = "POPULATION_SOURCE_REQUIRED"
    PRIOR_ATTESTATION_REQUIRED = "PRIOR_ATTESTATION_REQUIRED"
    PRIOR_ATTESTATION_INVALID = "PRIOR_ATTESTATION_INVALID"
    PRIOR_ATTESTATION_STAGE_MISMATCH = "PRIOR_ATTESTATION_STAGE_MISMATCH"
    PRIOR_ATTESTATION_SCOPE_MISMATCH = "PRIOR_ATTESTATION_SCOPE_MISMATCH"
    GRAPH_FINGERPRINT_REQUIRED = "GRAPH_FINGERPRINT_REQUIRED"
    EXECUTION_CLAIM_MISSING = "EXECUTION_CLAIM_MISSING"
    EXECUTION_CLAIM_UNEXPECTED = "EXECUTION_CLAIM_UNEXPECTED"
    EXECUTION_CLAIM_DUPLICATE = "EXECUTION_CLAIM_DUPLICATE"
    QUERY_NODE_REQUIRED = "QUERY_NODE_REQUIRED"
    QUERY_GENERATION_REQUIRED = "QUERY_GENERATION_REQUIRED"
    QUERY_ATTEMPT_REQUIRED = "QUERY_ATTEMPT_REQUIRED"
    QUERY_CONTRACT_FINGERPRINT_REQUIRED = "QUERY_CONTRACT_FINGERPRINT_REQUIRED"
    SQL_AST_FINGERPRINT_REQUIRED = "SQL_AST_FINGERPRINT_REQUIRED"
    POPULATION_FINGERPRINT_REQUIRED = "POPULATION_FINGERPRINT_REQUIRED"
    LINEAGE_PROOF_REQUIRED = "LINEAGE_PROOF_REQUIRED"
    LINEAGE_PROOF_UNTRUSTED = "LINEAGE_PROOF_UNTRUSTED"
    LINEAGE_PROOF_UNVERIFIED = "LINEAGE_PROOF_UNVERIFIED"
    LINEAGE_GRAPH_MISMATCH = "LINEAGE_GRAPH_MISMATCH"
    LINEAGE_QUERY_NODE_MISMATCH = "LINEAGE_QUERY_NODE_MISMATCH"
    LINEAGE_GENERATION_MISMATCH = "LINEAGE_GENERATION_MISMATCH"
    LINEAGE_ATTEMPT_MISMATCH = "LINEAGE_ATTEMPT_MISMATCH"
    LINEAGE_QUERY_CONTRACT_MISMATCH = "LINEAGE_QUERY_CONTRACT_MISMATCH"
    LINEAGE_SQL_AST_MISMATCH = "LINEAGE_SQL_AST_MISMATCH"
    LINEAGE_SOURCE_SNAPSHOT_MISMATCH = "LINEAGE_SOURCE_SNAPSHOT_MISMATCH"
    LINEAGE_RESULT_SNAPSHOT_MISMATCH = "LINEAGE_RESULT_SNAPSHOT_MISMATCH"
    LINEAGE_MECHANISM_INVALID = "LINEAGE_MECHANISM_INVALID"
    LINEAGE_SOURCE_POPULATION_MISMATCH = "LINEAGE_SOURCE_POPULATION_MISMATCH"
    LINEAGE_RESULT_POPULATION_MISMATCH = "LINEAGE_RESULT_POPULATION_MISMATCH"
    LINEAGE_SOURCE_GOAL_MISMATCH = "LINEAGE_SOURCE_GOAL_MISMATCH"
    CONSTRAINT_LINEAGE_INCOMPLETE = "CONSTRAINT_LINEAGE_INCOMPLETE"
    POPULATION_DEGRADED_TO_TIME_FILTER = "POPULATION_DEGRADED_TO_TIME_FILTER"
    ENTITY_MAPPING_REQUIRED = "ENTITY_MAPPING_REQUIRED"
    GRAIN_MAPPING_REQUIRED = "GRAIN_MAPPING_REQUIRED"
    SNAPSHOT_ALIGNMENT_REQUIRED = "SNAPSHOT_ALIGNMENT_REQUIRED"
    SNAPSHOT_FINGERPRINT_REQUIRED = "SNAPSHOT_FINGERPRINT_REQUIRED"
    MEMBERSHIP_INCOMPLETE = "MEMBERSHIP_INCOMPLETE"
    ARTIFACT_EVIDENCE_REQUIRED = "ARTIFACT_EVIDENCE_REQUIRED"
    ARTIFACT_UNTRUSTED = "ARTIFACT_UNTRUSTED"
    ARTIFACT_UNVERIFIED = "ARTIFACT_UNVERIFIED"
    ARTIFACT_NOT_IMMUTABLE = "ARTIFACT_NOT_IMMUTABLE"
    ARTIFACT_KIND_MISMATCH = "ARTIFACT_KIND_MISMATCH"
    ARTIFACT_COVERAGE_INCOMPLETE = "ARTIFACT_COVERAGE_INCOMPLETE"
    ARTIFACT_POPULATION_MISMATCH = "ARTIFACT_POPULATION_MISMATCH"
    ARTIFACT_GOAL_CONTRACT_MISMATCH = "ARTIFACT_GOAL_CONTRACT_MISMATCH"
    ARTIFACT_GRAPH_MISMATCH = "ARTIFACT_GRAPH_MISMATCH"
    ARTIFACT_QUERY_CONTRACT_MISMATCH = "ARTIFACT_QUERY_CONTRACT_MISMATCH"
    ARTIFACT_SQL_AST_MISMATCH = "ARTIFACT_SQL_AST_MISMATCH"
    ARTIFACT_SNAPSHOT_MISMATCH = "ARTIFACT_SNAPSHOT_MISMATCH"
    RESULT_EVIDENCE_MISSING = "RESULT_EVIDENCE_MISSING"
    RESULT_EVIDENCE_UNEXPECTED = "RESULT_EVIDENCE_UNEXPECTED"
    RESULT_EVIDENCE_DUPLICATE = "RESULT_EVIDENCE_DUPLICATE"
    RESULT_LINEAGE_ATTESTATION_MISMATCH = "RESULT_LINEAGE_ATTESTATION_MISMATCH"


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _require_text(value: Any, field_name: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


def _normalized_unique(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(_text(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError("%s must not contain empty values" % field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError("%s must not contain duplicate values" % field_name)
    return normalized


def _stable_fingerprint(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def population_question_fingerprint(question: str) -> str:
    """Retain exact question identity without interpreting any language."""

    return _stable_fingerprint({"question": str(question or "")})


class PopulationConstraintEvidence(_StrictFrozenModel):
    fingerprint: str
    kind: PopulationConstraintKind
    semantic_ref_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationConstraintEvidence":
        _require_text(self.fingerprint, "fingerprint")
        _normalized_unique(self.semantic_ref_ids, "semantic_ref_ids")
        return self


class PopulationScopeDescriptor(_StrictFrozenModel):
    """A language-neutral description of one intended or executable population."""

    scope_id: str
    kind: PopulationScopeKind
    source_goal_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...] = ()
    population_fingerprint: str = ""
    entity_identity_ref: str = ""
    grain_fingerprint: str = ""
    snapshot_fingerprint: str = ""
    constraints: tuple[PopulationConstraintEvidence, ...] = ()
    complete_membership_required: bool = False

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationScopeDescriptor":
        _require_text(self.scope_id, "scope_id")
        _normalized_unique(self.source_goal_ids, "source_goal_ids")
        _normalized_unique(self.source_artifact_ids, "source_artifact_ids")
        fingerprints = tuple(item.fingerprint for item in self.constraints)
        _normalized_unique(fingerprints, "constraints.fingerprint")
        return self


class PopulationSemanticExpectation(_StrictFrozenModel):
    expectation_id: str
    consumer_goal_id: str
    expected_scope: PopulationScopeDescriptor

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationSemanticExpectation":
        _require_text(self.expectation_id, "expectation_id")
        _require_text(self.consumer_goal_id, "consumer_goal_id")
        return self


class PopulationSemanticReview(_StrictFrozenModel):
    review_id: str
    question_fingerprint: str
    goal_skeleton_fingerprint: str = ""
    provider_request_fingerprint: str = ""
    provider_output_fingerprint: str = ""
    verifier_fingerprint: str
    complete: bool
    expectations: tuple[PopulationSemanticExpectation, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationSemanticReview":
        _require_text(self.review_id, "review_id")
        _require_text(self.question_fingerprint, "question_fingerprint")
        _require_text(self.verifier_fingerprint, "verifier_fingerprint")
        return self


class PopulationDeclaration(_StrictFrozenModel):
    consumer_goal_id: str
    declared_scope: PopulationScopeDescriptor

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationDeclaration":
        _require_text(self.consumer_goal_id, "consumer_goal_id")
        return self


class PopulationArtifactEvidence(_StrictFrozenModel):
    artifact_id: str
    artifact_fingerprint: str
    artifact_kind: PopulationArtifactKind
    coverage: PopulationArtifactCoverage
    population_fingerprint: str
    verifier_fingerprint: str
    verified: bool
    immutable: bool
    goal_contract_fingerprint: str = ""
    graph_fingerprint: str = ""
    query_contract_fingerprint: str = ""
    sql_ast_fingerprint: str = ""
    snapshot_fingerprint: str = ""
    lineage_proof_fingerprints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationArtifactEvidence":
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.artifact_fingerprint, "artifact_fingerprint")
        _require_text(self.population_fingerprint, "population_fingerprint")
        _require_text(self.verifier_fingerprint, "verifier_fingerprint")
        _normalized_unique(
            self.lineage_proof_fingerprints,
            "lineage_proof_fingerprints",
        )
        return self


class PopulationLineageProof(_StrictFrozenModel):
    proof_id: str
    mechanism: PopulationLineageMechanism
    verifier_fingerprint: str
    verified: bool
    graph_fingerprint: str = ""
    query_node_id: str = ""
    generation: int = Field(default=0, ge=0)
    attempt_id: str = ""
    query_contract_fingerprint: str = ""
    sql_ast_fingerprint: str = ""
    source_population_fingerprint: str
    result_population_fingerprint: str
    source_goal_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    preserved_constraints: tuple[PopulationConstraintEvidence, ...] = ()
    artifact_evidence: tuple[PopulationArtifactEvidence, ...] = ()
    source_entity_identity_ref: str = ""
    result_entity_identity_ref: str = ""
    entity_mapping_fingerprint: str = ""
    relationship_ref_ids: tuple[str, ...] = ()
    source_grain_fingerprint: str = ""
    result_grain_fingerprint: str = ""
    grain_mapping_fingerprint: str = ""
    source_snapshot_fingerprint: str = ""
    result_snapshot_fingerprint: str = ""
    snapshot_alignment_fingerprint: str = ""
    complete_membership: bool = False

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationLineageProof":
        _require_text(self.proof_id, "proof_id")
        _require_text(self.verifier_fingerprint, "verifier_fingerprint")
        _normalized_unique(self.source_goal_ids, "source_goal_ids")
        _normalized_unique(self.source_node_ids, "source_node_ids")
        _normalized_unique(self.relationship_ref_ids, "relationship_ref_ids")
        fingerprints = tuple(item.fingerprint for item in self.preserved_constraints)
        _normalized_unique(fingerprints, "preserved_constraints.fingerprint")
        artifact_ids = tuple(item.artifact_id for item in self.artifact_evidence)
        _normalized_unique(artifact_ids, "artifact_evidence.artifact_id")
        return self


class PopulationExecutionClaim(_StrictFrozenModel):
    consumer_goal_id: str
    query_node_id: str
    generation: int = Field(default=0, ge=0)
    attempt_id: str = ""
    declaration_scope_fingerprint: str
    required_scope: PopulationScopeDescriptor
    effective_scope: PopulationScopeDescriptor
    query_contract_fingerprint: str
    sql_ast_fingerprint: str
    lineage_proofs: tuple[PopulationLineageProof, ...]

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationExecutionClaim":
        _require_text(self.consumer_goal_id, "consumer_goal_id")
        proof_ids = tuple(item.proof_id for item in self.lineage_proofs)
        _normalized_unique(proof_ids, "lineage_proofs.proof_id")
        return self


class PopulationResultEvidence(_StrictFrozenModel):
    consumer_goal_id: str
    query_node_id: str
    result_artifact: PopulationArtifactEvidence
    lineage_proof_fingerprints: tuple[str, ...]

    @model_validator(mode="after")
    def validate_structure(self) -> "PopulationResultEvidence":
        _require_text(self.consumer_goal_id, "consumer_goal_id")
        _require_text(self.query_node_id, "query_node_id")
        _normalized_unique(
            self.lineage_proof_fingerprints,
            "lineage_proof_fingerprints",
        )
        return self


class PopulationScopeAttestation(_StrictFrozenModel):
    consumer_goal_id: str
    scope_kind: PopulationScopeKind
    source_goal_ids: tuple[str, ...] = ()
    declaration_scope_fingerprint: str
    population_fingerprint: str = ""
    entity_identity_ref: str = ""
    grain_fingerprint: str = ""
    constraint_fingerprints: tuple[str, ...] = ()
    complete_membership_required: bool = False
    query_node_id: str = ""
    generation: int = Field(default=0, ge=0)
    attempt_id: str = ""
    query_contract_fingerprint: str = ""
    sql_ast_fingerprint: str = ""
    snapshot_fingerprint: str = ""
    source_artifact_ids: tuple[str, ...] = ()
    proof_fingerprints: tuple[str, ...] = ()


class PopulationVerificationGap(_StrictFrozenModel):
    code: PopulationGapCode
    stage: PopulationVerificationStage
    message: str
    consumer_goal_id: str = ""
    query_node_id: str = ""
    proof_id: str = ""
    artifact_id: str = ""
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    blocking: Literal[True] = True


class PopulationVerificationAttestation(_StrictFrozenModel):
    verifier_version: Literal["population_semantic_verifier.v1"] = (
        "population_semantic_verifier.v1"
    )
    stage: PopulationVerificationStage
    passed: bool
    gate_open: bool
    input_fingerprint: str
    goal_contract_fingerprint: str
    question_fingerprint: str = ""
    graph_fingerprint: str = ""
    accepted_scopes: tuple[PopulationScopeAttestation, ...] = ()
    accepted_proof_fingerprints: tuple[str, ...] = ()
    artifact_fingerprints: tuple[str, ...] = ()
    gap_codes: tuple[PopulationGapCode, ...] = ()
    previous_attestation_fingerprint: str = ""
    attestation_fingerprint: str = ""


class GoalDeclarationPopulationVerificationInput(_StrictFrozenModel):
    stage: Literal[PopulationVerificationStage.GOAL_DECLARATION] = (
        PopulationVerificationStage.GOAL_DECLARATION
    )
    question_fingerprint: str
    goal_skeleton_fingerprint: str = ""
    goal_contract_fingerprint: str
    declaration_author_fingerprint: str
    semantic_review: PopulationSemanticReview | None = None
    trusted_semantic_verifier_fingerprints: tuple[str, ...]
    declarations: tuple[PopulationDeclaration, ...] = ()


class PreExecutionPopulationVerificationInput(_StrictFrozenModel):
    stage: Literal[PopulationVerificationStage.PRE_EXECUTION] = (
        PopulationVerificationStage.PRE_EXECUTION
    )
    goal_contract_fingerprint: str
    graph_fingerprint: str
    declaration_attestation: PopulationVerificationAttestation | None = None
    trusted_lineage_verifier_fingerprints: tuple[str, ...]
    trusted_artifact_verifier_fingerprints: tuple[str, ...] = ()
    required_consumer_goal_ids: tuple[str, ...] = ()
    consumer_scope_selection_explicit: bool = False
    claims: tuple[PopulationExecutionClaim, ...] = ()


class PostResultPopulationVerificationInput(_StrictFrozenModel):
    stage: Literal[PopulationVerificationStage.POST_RESULT] = (
        PopulationVerificationStage.POST_RESULT
    )
    goal_contract_fingerprint: str
    graph_fingerprint: str
    pre_execution_attestation: PopulationVerificationAttestation | None = None
    trusted_artifact_verifier_fingerprints: tuple[str, ...]
    required_consumer_goal_ids: tuple[str, ...] = ()
    consumer_scope_selection_explicit: bool = False
    results: tuple[PopulationResultEvidence, ...] = ()


PopulationVerificationInput: TypeAlias = Annotated[
    GoalDeclarationPopulationVerificationInput
    | PreExecutionPopulationVerificationInput
    | PostResultPopulationVerificationInput,
    Field(discriminator="stage"),
]


_INPUT_ADAPTER = TypeAdapter(PopulationVerificationInput)


class PopulationVerificationResult(_StrictFrozenModel):
    stage: PopulationVerificationStage
    passed: bool
    gate_open: bool
    gaps: tuple[PopulationVerificationGap, ...] = ()
    attestation: PopulationVerificationAttestation


_DEPENDENT_SCOPES = {
    PopulationScopeKind.SAME_AS_GOAL.value,
    PopulationScopeKind.VERIFIED_ENTITY_SET.value,
    PopulationScopeKind.PREDICATE_SCOPE.value,
    PopulationScopeKind.VERIFIED_RESULT_ARTIFACT.value,
}

_MECHANISMS_BY_SCOPE = {
    PopulationScopeKind.UNIVERSE.value: {
        PopulationLineageMechanism.DIRECT_SCOPE.value,
    },
    PopulationScopeKind.INDEPENDENT.value: {
        PopulationLineageMechanism.DIRECT_SCOPE.value,
    },
    PopulationScopeKind.SAME_AS_GOAL.value: {
        PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE.value,
        PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE.value,
        PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE.value,
        PopulationLineageMechanism.VERIFIED_ENTITY_SET_ARTIFACT.value,
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value,
    },
    PopulationScopeKind.PREDICATE_SCOPE.value: {
        PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE.value,
        PopulationLineageMechanism.SAME_QUERY_CTE_LINEAGE.value,
        PopulationLineageMechanism.SAME_QUERY_SEMI_JOIN_LINEAGE.value,
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value,
    },
    PopulationScopeKind.VERIFIED_ENTITY_SET.value: {
        PopulationLineageMechanism.VERIFIED_ENTITY_SET_ARTIFACT.value,
    },
    PopulationScopeKind.VERIFIED_RESULT_ARTIFACT.value: {
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value,
    },
}

_COMPLETE_COVERAGE = {
    PopulationArtifactCoverage.ALL_ROWS.value,
    PopulationArtifactCoverage.COMPLETE.value,
    PopulationArtifactCoverage.EXACT_ENTITY_SET.value,
    PopulationArtifactCoverage.TOP_N.value,
}


def population_declaration_scope_fingerprint(scope: PopulationScopeDescriptor) -> str:
    """Fingerprint only declaration-time semantics, not late execution bindings."""

    return _stable_fingerprint(
        {
            "kind": str(scope.kind),
            "sourceGoalIds": sorted(scope.source_goal_ids),
            "sourceArtifactIds": sorted(scope.source_artifact_ids),
            "populationFingerprint": scope.population_fingerprint,
            "entityIdentityRef": scope.entity_identity_ref,
            "grainFingerprint": scope.grain_fingerprint,
            "snapshotFingerprint": scope.snapshot_fingerprint,
            "constraintFingerprints": sorted(
                item.fingerprint for item in scope.constraints
            ),
            "completeMembershipRequired": scope.complete_membership_required,
        }
    )


def population_lineage_proof_fingerprint(proof: PopulationLineageProof) -> str:
    return _stable_fingerprint(proof)


def population_attestation_fingerprint(
    attestation: PopulationVerificationAttestation,
) -> str:
    payload = attestation.model_dump(by_alias=True, mode="json")
    payload["attestationFingerprint"] = ""
    return _stable_fingerprint(payload)


def population_declarations_from_goal_contract(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    consumer_goal_ids: Sequence[str] = (),
) -> tuple[PopulationDeclaration, ...]:
    """Adapt existing Goal Contract population declarations without inferring prose.

    By default only ranking or explicitly dependent population goals are
    projected. Callers may provide consumer IDs to project another goal kind.
    """

    parsed = parse_original_question_goal_contract(contract)
    requested = {_text(value) for value in consumer_goal_ids if _text(value)}
    declarations: list[PopulationDeclaration] = []
    kind_map = {
        "ALL_MATCHING_ROWS": PopulationScopeKind.INDEPENDENT,
        "SAME_AS_GOAL": PopulationScopeKind.SAME_AS_GOAL,
        "VERIFIED_ENTITY_SET": PopulationScopeKind.VERIFIED_ENTITY_SET,
        "VERIFIED_PREDICATE_SCOPE": PopulationScopeKind.PREDICATE_SCOPE,
        "VERIFIED_RESULT_ARTIFACT": PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
    }
    for goal in parsed.goals:
        include = (
            goal.goal_id in requested
            if requested
            else isinstance(goal, RankingQuestionGoal)
            or bool(goal.population_goal_ids)
            or str(goal.population_scope) != "ALL_MATCHING_ROWS"
        )
        if not include:
            continue
        scope_kind = kind_map[str(goal.population_scope)]
        declarations.append(
            PopulationDeclaration(
                consumer_goal_id=goal.goal_id,
                declared_scope=PopulationScopeDescriptor(
                    scope_id="goal:%s:population" % goal.goal_id,
                    kind=scope_kind,
                    source_goal_ids=tuple(goal.population_goal_ids),
                    complete_membership_required=scope_kind
                    in {
                        PopulationScopeKind.SAME_AS_GOAL,
                        PopulationScopeKind.VERIFIED_ENTITY_SET,
                        PopulationScopeKind.VERIFIED_RESULT_ARTIFACT,
                    },
                ),
            )
        )
    return tuple(declarations)


def goal_population_verification_input(
    contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    *,
    semantic_review: PopulationSemanticReview,
    declaration_author_fingerprint: str,
    trusted_semantic_verifier_fingerprints: Sequence[str],
    consumer_goal_ids: Sequence[str] = (),
) -> GoalDeclarationPopulationVerificationInput:
    parsed = parse_original_question_goal_contract(contract)
    return GoalDeclarationPopulationVerificationInput(
        question_fingerprint=population_question_fingerprint(parsed.question),
        goal_skeleton_fingerprint=semantic_review.goal_skeleton_fingerprint,
        goal_contract_fingerprint=original_question_goal_contract_fingerprint(
            parsed
        ),
        declaration_author_fingerprint=declaration_author_fingerprint,
        semantic_review=semantic_review,
        trusted_semantic_verifier_fingerprints=tuple(
            trusted_semantic_verifier_fingerprints
        ),
        declarations=population_declarations_from_goal_contract(
            parsed,
            consumer_goal_ids=consumer_goal_ids,
        ),
    )


class PopulationSemanticVerifier:
    """Independent fail-closed population gate for three lifecycle stages.

    This verifier never parses user language or SQL. Independent semantic and
    AST/artifact authorities must supply typed, fingerprinted facts.
    """

    def verify(
        self,
        request: PopulationVerificationInput | Mapping[str, Any],
    ) -> PopulationVerificationResult:
        parsed = (
            request
            if isinstance(
                request,
                (
                    GoalDeclarationPopulationVerificationInput,
                    PreExecutionPopulationVerificationInput,
                    PostResultPopulationVerificationInput,
                ),
            )
            else _INPUT_ADAPTER.validate_python(request)
        )
        if isinstance(parsed, GoalDeclarationPopulationVerificationInput):
            return self.verify_goal_declaration(parsed)
        if isinstance(parsed, PreExecutionPopulationVerificationInput):
            return self.verify_pre_execution(parsed)
        return self.verify_post_result(parsed)

    def verify_goal_declaration(
        self,
        request: GoalDeclarationPopulationVerificationInput,
    ) -> PopulationVerificationResult:
        stage = PopulationVerificationStage.GOAL_DECLARATION
        gaps: list[PopulationVerificationGap] = []
        accepted_scopes: list[PopulationScopeAttestation] = []
        if not _text(request.goal_contract_fingerprint):
            gaps.append(
                _gap(
                    PopulationGapCode.GOAL_CONTRACT_FINGERPRINT_REQUIRED,
                    stage,
                    "Population declarations must bind the active Goal Contract.",
                    path="goalContractFingerprint",
                )
            )
        if not _text(request.declaration_author_fingerprint):
            gaps.append(
                _gap(
                    PopulationGapCode.DECLARATION_AUTHORITY_REQUIRED,
                    stage,
                    "Population declarations must identify the Core declaration authority.",
                    path="declarationAuthorFingerprint",
                )
            )
        review = request.semantic_review
        if review is None:
            gaps.append(
                _gap(
                    PopulationGapCode.SEMANTIC_REVIEW_REQUIRED,
                    stage,
                    "An independent semantic population review is required.",
                    path="semanticReview",
                )
            )
            expectations: tuple[PopulationSemanticExpectation, ...] = ()
        else:
            expectations = review.expectations
            trusted = set(request.trusted_semantic_verifier_fingerprints)
            if review.verifier_fingerprint not in trusted:
                gaps.append(
                    _gap(
                        PopulationGapCode.SEMANTIC_REVIEW_UNTRUSTED,
                        stage,
                        "The semantic population review was not produced by a trusted verifier.",
                        path="semanticReview.verifierFingerprint",
                    )
                )
            if review.verifier_fingerprint == request.declaration_author_fingerprint:
                gaps.append(
                    _gap(
                        PopulationGapCode.SEMANTIC_REVIEW_NOT_INDEPENDENT,
                        stage,
                        "The population declaration and semantic review require distinct authorities.",
                        path="semanticReview.verifierFingerprint",
                    )
                )
            if review.question_fingerprint != request.question_fingerprint:
                gaps.append(
                    _gap(
                        PopulationGapCode.QUESTION_FINGERPRINT_MISMATCH,
                        stage,
                        "The semantic review is bound to a different original question.",
                        path="semanticReview.questionFingerprint",
                    )
                )
            if request.goal_skeleton_fingerprint:
                if (
                    not review.goal_skeleton_fingerprint
                    or not review.provider_request_fingerprint
                    or not review.provider_output_fingerprint
                ):
                    gaps.append(
                        _gap(
                            PopulationGapCode.SEMANTIC_REVIEW_BINDING_MISSING,
                            stage,
                            "The semantic review lacks sealed skeleton or provider request/output bindings.",
                            path="semanticReview",
                        )
                    )
                elif (
                    review.goal_skeleton_fingerprint
                    != request.goal_skeleton_fingerprint
                ):
                    gaps.append(
                        _gap(
                            PopulationGapCode.GOAL_SKELETON_FINGERPRINT_MISMATCH,
                            stage,
                            "The semantic review was produced from a different Goal skeleton.",
                            path="semanticReview.goalSkeletonFingerprint",
                        )
                    )
            if not review.complete:
                gaps.append(
                    _gap(
                        PopulationGapCode.SEMANTIC_REVIEW_INCOMPLETE,
                        stage,
                        "The semantic verifier did not attest complete population review.",
                        path="semanticReview.complete",
                    )
                )

        expectation_map, expectation_duplicates = _unique_by_consumer(
            expectations,
            lambda item: item.consumer_goal_id,
        )
        declaration_map, declaration_duplicates = _unique_by_consumer(
            request.declarations,
            lambda item: item.consumer_goal_id,
        )
        for consumer_goal_id in sorted(expectation_duplicates | declaration_duplicates):
            gaps.append(
                _gap(
                    PopulationGapCode.POPULATION_DECLARATION_DUPLICATE,
                    stage,
                    "A consumer Goal has duplicate population declarations or expectations.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(set(expectation_map) - set(declaration_map)):
            gaps.append(
                _gap(
                    PopulationGapCode.POPULATION_DECLARATION_MISSING,
                    stage,
                    "Core omitted a population declaration required by the independent semantic review.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(set(declaration_map) - set(expectation_map)):
            gaps.append(
                _gap(
                    PopulationGapCode.POPULATION_DECLARATION_UNEXPECTED,
                    stage,
                    "Core declared population semantics absent from the complete independent review.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(set(expectation_map) & set(declaration_map)):
            expected = expectation_map[consumer_goal_id].expected_scope
            declared = declaration_map[consumer_goal_id].declared_scope
            comparison_gaps = _compare_declaration_scopes(
                expected,
                declared,
                stage=stage,
                consumer_goal_id=consumer_goal_id,
            )
            gaps.extend(comparison_gaps)
            if not comparison_gaps:
                accepted_scopes.append(
                    PopulationScopeAttestation(
                        consumer_goal_id=consumer_goal_id,
                        scope_kind=declared.kind,
                        source_goal_ids=tuple(sorted(declared.source_goal_ids)),
                        declaration_scope_fingerprint=(
                            population_declaration_scope_fingerprint(declared)
                        ),
                        population_fingerprint=declared.population_fingerprint,
                        entity_identity_ref=declared.entity_identity_ref,
                        grain_fingerprint=declared.grain_fingerprint,
                        constraint_fingerprints=tuple(
                            sorted(
                                item.fingerprint for item in declared.constraints
                            )
                        ),
                        complete_membership_required=(
                            declared.complete_membership_required
                        ),
                        source_artifact_ids=tuple(
                            sorted(declared.source_artifact_ids)
                        ),
                    )
                )
        return _result(
            request,
            stage=stage,
            goal_contract_fingerprint=request.goal_contract_fingerprint,
            question_fingerprint=request.question_fingerprint,
            gaps=gaps,
            accepted_scopes=accepted_scopes,
        )

    def verify_pre_execution(
        self,
        request: PreExecutionPopulationVerificationInput,
    ) -> PopulationVerificationResult:
        stage = PopulationVerificationStage.PRE_EXECUTION
        gaps: list[PopulationVerificationGap] = []
        prior = request.declaration_attestation
        prior_valid = _validate_prior_attestation(
            prior,
            expected_stage=PopulationVerificationStage.GOAL_DECLARATION,
            goal_contract_fingerprint=request.goal_contract_fingerprint,
            graph_fingerprint="",
            current_stage=stage,
        )
        gaps.extend(prior_valid)
        if not _text(request.graph_fingerprint):
            gaps.append(
                _gap(
                    PopulationGapCode.GRAPH_FINGERPRINT_REQUIRED,
                    stage,
                    "Pre-execution population verification must bind the frozen execution graph.",
                    path="graphFingerprint",
                )
            )
        prior_scopes = {
            item.consumer_goal_id: item
            for item in prior.accepted_scopes
        } if prior is not None else {}
        attested_consumers = set(prior_scopes)
        requested_consumers = set(request.required_consumer_goal_ids)
        explicit_selection = bool(
            request.consumer_scope_selection_explicit
        )
        selection_mismatch = (
            not requested_consumers.issubset(attested_consumers)
            if explicit_selection
            else bool(requested_consumers)
            and requested_consumers != attested_consumers
        )
        if selection_mismatch:
            gaps.append(
                _gap(
                    PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                    stage,
                    "A node cannot add population gates absent from Goal declaration.",
                    details={
                        "attestedConsumerGoalIds": sorted(attested_consumers),
                        "requestedConsumerGoalIds": sorted(requested_consumers),
                    },
                )
            )
        required_consumers = (
            requested_consumers
            if explicit_selection
            else attested_consumers
        )
        claim_map, duplicate_consumers = _unique_by_consumer(
            request.claims,
            lambda item: item.consumer_goal_id,
        )
        for consumer_goal_id in sorted(duplicate_consumers):
            gaps.append(
                _gap(
                    PopulationGapCode.EXECUTION_CLAIM_DUPLICATE,
                    stage,
                    "A consumer Goal has duplicate execution population claims.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(required_consumers - set(claim_map)):
            gaps.append(
                _gap(
                    PopulationGapCode.EXECUTION_CLAIM_MISSING,
                    stage,
                    "A required population declaration has no execution claim.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(set(claim_map) - required_consumers):
            gaps.append(
                _gap(
                    PopulationGapCode.EXECUTION_CLAIM_UNEXPECTED,
                    stage,
                    "An execution claim is outside this gate's required population set.",
                    consumer_goal_id=consumer_goal_id,
                )
            )

        accepted_scopes: list[PopulationScopeAttestation] = []
        accepted_proofs: list[str] = []
        artifact_fingerprints: list[str] = []
        for consumer_goal_id in sorted(required_consumers & set(claim_map)):
            claim = claim_map[consumer_goal_id]
            claim_gaps: list[PopulationVerificationGap] = []
            prior_scope = prior_scopes.get(consumer_goal_id)
            if prior_scope is None or (
                claim.declaration_scope_fingerprint
                != prior_scope.declaration_scope_fingerprint
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The execution claim does not bind the accepted declaration scope.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif (
                str(claim.required_scope.kind) != str(prior_scope.scope_kind)
                or set(claim.required_scope.source_goal_ids)
                != set(prior_scope.source_goal_ids)
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The resolved execution scope changed declared kind or Goal lineage.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif prior_scope.source_artifact_ids and set(
                claim.required_scope.source_artifact_ids
            ) != set(prior_scope.source_artifact_ids):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The resolved execution scope changed declared source artifacts.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif (
                claim.required_scope.complete_membership_required
                != prior_scope.complete_membership_required
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The execution claim changed the declared completeness requirement.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif (
                prior_scope.entity_identity_ref
                and claim.required_scope.entity_identity_ref
                != prior_scope.entity_identity_ref
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The execution claim changed the declared entity identity.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif (
                prior_scope.grain_fingerprint
                and claim.required_scope.grain_fingerprint
                != prior_scope.grain_fingerprint
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The execution claim changed the declared population grain.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            elif prior_scope.constraint_fingerprints and not set(
                prior_scope.constraint_fingerprints
            ).issubset(
                item.fingerprint for item in claim.required_scope.constraints
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The execution claim dropped declaration-time population constraints.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.query_node_id):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.QUERY_NODE_REQUIRED,
                        stage,
                        "A population execution claim requires a query node.",
                        consumer_goal_id=consumer_goal_id,
                    )
                )
            if claim.generation < 1:
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.QUERY_GENERATION_REQUIRED,
                        stage,
                        "A population execution claim requires its query-node generation.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.attempt_id):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.QUERY_ATTEMPT_REQUIRED,
                        stage,
                        "A population execution claim requires its query-node attempt identity.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.query_contract_fingerprint):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.QUERY_CONTRACT_FINGERPRINT_REQUIRED,
                        stage,
                        "A population execution claim must bind the frozen Query Contract.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.sql_ast_fingerprint):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.SQL_AST_FINGERPRINT_REQUIRED,
                        stage,
                        "Population lineage must bind the validated SQL AST, not SQL prose.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.required_scope.population_fingerprint) or not _text(
                claim.effective_scope.population_fingerprint
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.POPULATION_FINGERPRINT_REQUIRED,
                        stage,
                        "Resolved source and effective populations require full-scope fingerprints.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.required_scope.grain_fingerprint) or not _text(
                claim.effective_scope.grain_fingerprint
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.POPULATION_GRAIN_REQUIRED,
                        stage,
                        "Resolved populations require source and effective grain fingerprints.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not _text(claim.required_scope.snapshot_fingerprint) or not _text(
                claim.effective_scope.snapshot_fingerprint
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.SNAPSHOT_FINGERPRINT_REQUIRED,
                        stage,
                        "Population verification requires source and effective data snapshot fingerprints.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if (
                str(claim.required_scope.kind)
                == PopulationScopeKind.VERIFIED_ENTITY_SET.value
                and (
                    not _text(claim.required_scope.entity_identity_ref)
                    or not _text(claim.effective_scope.entity_identity_ref)
                )
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.POPULATION_ENTITY_IDENTITY_REQUIRED,
                        stage,
                        "A verified entity-set population requires source and target entity identities.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if (
                str(claim.required_scope.kind)
                in {
                    PopulationScopeKind.VERIFIED_ENTITY_SET.value,
                    PopulationScopeKind.VERIFIED_RESULT_ARTIFACT.value,
                }
                and not claim.required_scope.source_artifact_ids
            ):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.POPULATION_SOURCE_REQUIRED,
                        stage,
                        "This population kind requires an explicit verified source artifact.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if str(claim.required_scope.kind) != str(claim.effective_scope.kind):
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.POPULATION_SCOPE_KIND_MISMATCH,
                        stage,
                        "The effective execution scope changed the required population kind.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )
            if not claim.lineage_proofs:
                claim_gaps.append(
                    _gap(
                        PopulationGapCode.LINEAGE_PROOF_REQUIRED,
                        stage,
                        "Population preservation requires an independently verified lineage proof.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=claim.query_node_id,
                    )
                )

            successful_proofs: list[PopulationLineageProof] = []
            failed_proof_gaps: list[PopulationVerificationGap] = []
            for proof in claim.lineage_proofs:
                proof_gaps = _verify_lineage_proof(
                    claim,
                    proof,
                    graph_fingerprint=request.graph_fingerprint,
                    trusted_lineage_verifiers=request.trusted_lineage_verifier_fingerprints,
                    trusted_artifact_verifiers=request.trusted_artifact_verifier_fingerprints,
                )
                if proof_gaps:
                    failed_proof_gaps.extend(proof_gaps)
                else:
                    successful_proofs.append(proof)
            if claim.lineage_proofs and not successful_proofs:
                claim_gaps.extend(failed_proof_gaps)

            gaps.extend(claim_gaps)
            if claim_gaps:
                continue
            proof_fingerprints = tuple(
                sorted(
                    population_lineage_proof_fingerprint(proof)
                    for proof in successful_proofs
                )
            )
            accepted_proofs.extend(proof_fingerprints)
            accepted_artifacts = tuple(
                sorted(
                    {
                        artifact.artifact_id
                        for proof in successful_proofs
                        for artifact in proof.artifact_evidence
                    }
                )
            )
            artifact_fingerprints.extend(
                artifact.artifact_fingerprint
                for proof in successful_proofs
                for artifact in proof.artifact_evidence
            )
            accepted_scopes.append(
                PopulationScopeAttestation(
                    consumer_goal_id=consumer_goal_id,
                    scope_kind=claim.effective_scope.kind,
                    source_goal_ids=tuple(
                        sorted(claim.required_scope.source_goal_ids)
                    ),
                    declaration_scope_fingerprint=(
                        claim.declaration_scope_fingerprint
                    ),
                    population_fingerprint=(
                        claim.effective_scope.population_fingerprint
                    ),
                    entity_identity_ref=claim.effective_scope.entity_identity_ref,
                    grain_fingerprint=claim.effective_scope.grain_fingerprint,
                    constraint_fingerprints=tuple(
                        sorted(
                            item.fingerprint
                            for item in claim.effective_scope.constraints
                        )
                    ),
                    complete_membership_required=(
                        claim.required_scope.complete_membership_required
                    ),
                    query_node_id=claim.query_node_id,
                    generation=claim.generation,
                    attempt_id=claim.attempt_id,
                    query_contract_fingerprint=claim.query_contract_fingerprint,
                    sql_ast_fingerprint=claim.sql_ast_fingerprint,
                    snapshot_fingerprint=(
                        claim.effective_scope.snapshot_fingerprint
                    ),
                    source_artifact_ids=accepted_artifacts,
                    proof_fingerprints=proof_fingerprints,
                )
            )
        return _result(
            request,
            stage=stage,
            goal_contract_fingerprint=request.goal_contract_fingerprint,
            graph_fingerprint=request.graph_fingerprint,
            previous_attestation=prior,
            gaps=gaps,
            accepted_scopes=accepted_scopes,
            accepted_proof_fingerprints=accepted_proofs,
            artifact_fingerprints=artifact_fingerprints,
        )

    def verify_post_result(
        self,
        request: PostResultPopulationVerificationInput,
    ) -> PopulationVerificationResult:
        stage = PopulationVerificationStage.POST_RESULT
        gaps: list[PopulationVerificationGap] = []
        prior = request.pre_execution_attestation
        gaps.extend(
            _validate_prior_attestation(
                prior,
                expected_stage=PopulationVerificationStage.PRE_EXECUTION,
                goal_contract_fingerprint=request.goal_contract_fingerprint,
                graph_fingerprint=request.graph_fingerprint,
                current_stage=stage,
            )
        )
        prior_scopes = {
            item.consumer_goal_id: item
            for item in prior.accepted_scopes
        } if prior is not None else {}
        attested_consumers = set(prior_scopes)
        requested_consumers = set(request.required_consumer_goal_ids)
        explicit_selection = bool(
            request.consumer_scope_selection_explicit
        )
        selection_mismatch = (
            not requested_consumers.issubset(attested_consumers)
            if explicit_selection
            else bool(requested_consumers)
            and requested_consumers != attested_consumers
        )
        if selection_mismatch:
            gaps.append(
                _gap(
                    PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                    stage,
                    "A node cannot add population gates absent from its PRE attestation.",
                    details={
                        "attestedConsumerGoalIds": sorted(attested_consumers),
                        "requestedConsumerGoalIds": sorted(requested_consumers),
                    },
                )
            )
        required_consumers = (
            requested_consumers
            if explicit_selection
            else attested_consumers
        )
        result_map, duplicates = _unique_by_consumer(
            request.results,
            lambda item: item.consumer_goal_id,
        )
        for consumer_goal_id in sorted(duplicates):
            gaps.append(
                _gap(
                    PopulationGapCode.RESULT_EVIDENCE_DUPLICATE,
                    stage,
                    "A consumer Goal has duplicate result population evidence.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(required_consumers - set(result_map)):
            gaps.append(
                _gap(
                    PopulationGapCode.RESULT_EVIDENCE_MISSING,
                    stage,
                    "A required population has no verified result artifact.",
                    consumer_goal_id=consumer_goal_id,
                )
            )
        for consumer_goal_id in sorted(set(result_map) - required_consumers):
            gaps.append(
                _gap(
                    PopulationGapCode.RESULT_EVIDENCE_UNEXPECTED,
                    stage,
                    "A result is outside this gate's required population set.",
                    consumer_goal_id=consumer_goal_id,
                )
            )

        accepted_scopes: list[PopulationScopeAttestation] = []
        artifact_fingerprints: list[str] = []
        for consumer_goal_id in sorted(required_consumers & set(result_map)):
            result = result_map[consumer_goal_id]
            prior_scope = prior_scopes.get(consumer_goal_id)
            result_gaps: list[PopulationVerificationGap] = []
            if prior_scope is None:
                result_gaps.append(
                    _gap(
                        PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                        stage,
                        "The result has no accepted pre-execution population scope.",
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=result.query_node_id,
                    )
                )
            else:
                if result.query_node_id != prior_scope.query_node_id:
                    result_gaps.append(
                        _gap(
                            PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                            stage,
                            "The result was produced by a different query node.",
                            consumer_goal_id=consumer_goal_id,
                            query_node_id=result.query_node_id,
                        )
                    )
                if not set(prior_scope.proof_fingerprints).issubset(
                    result.lineage_proof_fingerprints
                ):
                    result_gaps.append(
                        _gap(
                            PopulationGapCode.RESULT_LINEAGE_ATTESTATION_MISMATCH,
                            stage,
                            "The result artifact does not retain accepted pre-execution lineage proofs.",
                            consumer_goal_id=consumer_goal_id,
                            query_node_id=result.query_node_id,
                        )
                    )
                if set(result.lineage_proof_fingerprints) != set(
                    result.result_artifact.lineage_proof_fingerprints
                ):
                    result_gaps.append(
                        _gap(
                            PopulationGapCode.RESULT_LINEAGE_ATTESTATION_MISMATCH,
                            stage,
                            "Result lineage references are not sealed into the immutable artifact evidence.",
                            consumer_goal_id=consumer_goal_id,
                            query_node_id=result.query_node_id,
                        )
                    )
                result_gaps.extend(
                    _verify_result_artifact(
                        result.result_artifact,
                        prior_scope,
                        goal_contract_fingerprint=request.goal_contract_fingerprint,
                        graph_fingerprint=request.graph_fingerprint,
                        trusted_artifact_verifiers=(
                            request.trusted_artifact_verifier_fingerprints
                        ),
                        consumer_goal_id=consumer_goal_id,
                        query_node_id=result.query_node_id,
                    )
                )
            gaps.extend(result_gaps)
            if not result_gaps and prior_scope is not None:
                accepted_scopes.append(prior_scope)
                artifact_fingerprints.append(
                    result.result_artifact.artifact_fingerprint
                )
        return _result(
            request,
            stage=stage,
            goal_contract_fingerprint=request.goal_contract_fingerprint,
            graph_fingerprint=request.graph_fingerprint,
            previous_attestation=prior,
            gaps=gaps,
            accepted_scopes=accepted_scopes,
            accepted_proof_fingerprints=(
                prior.accepted_proof_fingerprints if prior is not None else ()
            ),
            artifact_fingerprints=artifact_fingerprints,
        )


def _compare_declaration_scopes(
    expected: PopulationScopeDescriptor,
    declared: PopulationScopeDescriptor,
    *,
    stage: PopulationVerificationStage,
    consumer_goal_id: str,
) -> list[PopulationVerificationGap]:
    gaps: list[PopulationVerificationGap] = []
    if str(expected.kind) != str(declared.kind):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SCOPE_KIND_MISMATCH,
                stage,
                "Core changed the independently verified population kind.",
                consumer_goal_id=consumer_goal_id,
                details={
                    "expected": str(expected.kind),
                    "declared": str(declared.kind),
                },
            )
        )
    if set(expected.source_goal_ids) != set(declared.source_goal_ids):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SOURCE_GOAL_MISMATCH,
                stage,
                "Core changed the upstream Goal population references.",
                consumer_goal_id=consumer_goal_id,
                details={
                    "expected": sorted(expected.source_goal_ids),
                    "declared": sorted(declared.source_goal_ids),
                },
            )
        )
    if expected.source_artifact_ids and set(expected.source_artifact_ids) != set(
        declared.source_artifact_ids
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SOURCE_ARTIFACT_MISMATCH,
                stage,
                "Core changed the verified source artifact population references.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    if (
        expected.population_fingerprint
        and expected.population_fingerprint != declared.population_fingerprint
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SCOPE_FINGERPRINT_MISMATCH,
                stage,
                "Core changed the expected population fingerprint.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    if (
        expected.entity_identity_ref
        and expected.entity_identity_ref != declared.entity_identity_ref
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_ENTITY_IDENTITY_MISMATCH,
                stage,
                "Core changed the expected population entity identity.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    if expected.grain_fingerprint and (
        expected.grain_fingerprint != declared.grain_fingerprint
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_GRAIN_MISMATCH,
                stage,
                "Core changed the expected population grain.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    if (
        expected.complete_membership_required
        != declared.complete_membership_required
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_COMPLETENESS_MISMATCH,
                stage,
                "Core changed the expected population completeness requirement.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    if (
        str(expected.kind) == PopulationScopeKind.SAME_AS_GOAL.value
        and not expected.source_goal_ids
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SOURCE_REQUIRED,
                stage,
                "SAME_AS_GOAL requires an explicit upstream Goal population.",
                consumer_goal_id=consumer_goal_id,
            )
        )
    return gaps


def _verify_lineage_proof(
    claim: PopulationExecutionClaim,
    proof: PopulationLineageProof,
    *,
    graph_fingerprint: str,
    trusted_lineage_verifiers: Sequence[str],
    trusted_artifact_verifiers: Sequence[str],
) -> list[PopulationVerificationGap]:
    stage = PopulationVerificationStage.PRE_EXECUTION
    consumer_goal_id = claim.consumer_goal_id
    common = {
        "consumer_goal_id": consumer_goal_id,
        "query_node_id": claim.query_node_id,
        "proof_id": proof.proof_id,
    }
    gaps: list[PopulationVerificationGap] = []
    if proof.verifier_fingerprint not in set(trusted_lineage_verifiers):
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_PROOF_UNTRUSTED,
                stage,
                "The population lineage proof was not produced by a trusted AST/lineage verifier.",
                **common,
            )
        )
    if not proof.verified:
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_PROOF_UNVERIFIED,
                stage,
                "The population lineage proof is not verified.",
                **common,
            )
        )
    execution_bindings = (
        (
            PopulationGapCode.LINEAGE_GRAPH_MISMATCH,
            proof.graph_fingerprint,
            graph_fingerprint,
            "frozen execution graph",
        ),
        (
            PopulationGapCode.LINEAGE_QUERY_NODE_MISMATCH,
            proof.query_node_id,
            claim.query_node_id,
            "query node",
        ),
        (
            PopulationGapCode.LINEAGE_GENERATION_MISMATCH,
            proof.generation,
            claim.generation,
            "query generation",
        ),
        (
            PopulationGapCode.LINEAGE_ATTEMPT_MISMATCH,
            proof.attempt_id,
            claim.attempt_id,
            "query attempt",
        ),
        (
            PopulationGapCode.LINEAGE_QUERY_CONTRACT_MISMATCH,
            proof.query_contract_fingerprint,
            claim.query_contract_fingerprint,
            "frozen Query Contract",
        ),
        (
            PopulationGapCode.LINEAGE_SQL_AST_MISMATCH,
            proof.sql_ast_fingerprint,
            claim.sql_ast_fingerprint,
            "validated SQL AST",
        ),
    )
    for code, actual, expected, label in execution_bindings:
        if not actual or actual != expected:
            gaps.append(
                _gap(
                    code,
                    stage,
                    "The lineage proof is not bound to the active %s." % label,
                    **common,
                )
            )
    allowed = _MECHANISMS_BY_SCOPE.get(str(claim.required_scope.kind), set())
    if str(proof.mechanism) not in allowed:
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_MECHANISM_INVALID,
                stage,
                "The lineage mechanism cannot prove this population kind.",
                details={
                    "scopeKind": str(claim.required_scope.kind),
                    "mechanism": str(proof.mechanism),
                },
                **common,
            )
        )
    if (
        proof.source_population_fingerprint
        != claim.required_scope.population_fingerprint
    ):
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_SOURCE_POPULATION_MISMATCH,
                stage,
                "The proof does not originate from the full required population.",
                **common,
            )
        )
    if (
        proof.result_population_fingerprint
        != claim.effective_scope.population_fingerprint
    ):
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_RESULT_POPULATION_MISMATCH,
                stage,
                "The proof does not terminate at the effective query population.",
                **common,
            )
        )
    if str(claim.required_scope.kind) in _DEPENDENT_SCOPES and (
        claim.required_scope.population_fingerprint
        != claim.effective_scope.population_fingerprint
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SCOPE_FINGERPRINT_MISMATCH,
                stage,
                "A dependent population was changed instead of preserved.",
                **common,
            )
        )
    required_source_goals = set(claim.required_scope.source_goal_ids)
    if required_source_goals and not required_source_goals.issubset(
        proof.source_goal_ids
    ):
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_SOURCE_GOAL_MISMATCH,
                stage,
                "The proof omits an upstream Goal that defines the population.",
                details={"requiredSourceGoalIds": sorted(required_source_goals)},
                **common,
            )
        )
    required_constraints = {
        item.fingerprint for item in claim.required_scope.constraints
    }
    preserved_constraints = {
        item.fingerprint for item in proof.preserved_constraints
    }
    if not required_constraints.issubset(preserved_constraints):
        gaps.append(
            _gap(
                PopulationGapCode.CONSTRAINT_LINEAGE_INCOMPLETE,
                stage,
                "The proof omits constraints from the full upstream population.",
                details={
                    "missingConstraintFingerprints": sorted(
                        required_constraints - preserved_constraints
                    )
                },
                **common,
            )
        )

    source_identity = (
        proof.source_entity_identity_ref
        or claim.required_scope.entity_identity_ref
    )
    result_identity = (
        proof.result_entity_identity_ref
        or claim.effective_scope.entity_identity_ref
    )
    if source_identity != result_identity and not (
        proof.entity_mapping_fingerprint and proof.relationship_ref_ids
    ):
        gaps.append(
            _gap(
                PopulationGapCode.ENTITY_MAPPING_REQUIRED,
                stage,
                "Different population entity identities require a governed relationship mapping.",
                **common,
            )
        )
    source_grain = (
        proof.source_grain_fingerprint
        or claim.required_scope.grain_fingerprint
    )
    result_grain = (
        proof.result_grain_fingerprint
        or claim.effective_scope.grain_fingerprint
    )
    if source_grain != result_grain and not proof.grain_mapping_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.GRAIN_MAPPING_REQUIRED,
                stage,
                "Different population grains require an independently verified grain mapping.",
                **common,
            )
        )
    source_snapshot = proof.source_snapshot_fingerprint
    result_snapshot = proof.result_snapshot_fingerprint
    if source_snapshot != claim.required_scope.snapshot_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_SOURCE_SNAPSHOT_MISMATCH,
                stage,
                "The lineage proof source snapshot differs from the required population snapshot.",
                **common,
            )
        )
    if result_snapshot != claim.effective_scope.snapshot_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.LINEAGE_RESULT_SNAPSHOT_MISMATCH,
                stage,
                "The lineage proof result snapshot differs from the effective population snapshot.",
                **common,
            )
        )
    if source_snapshot != result_snapshot and not proof.snapshot_alignment_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.SNAPSHOT_ALIGNMENT_REQUIRED,
                stage,
                "Different data snapshots require an independently verified alignment contract.",
                **common,
            )
        )
    if claim.required_scope.complete_membership_required and not proof.complete_membership:
        gaps.append(
            _gap(
                PopulationGapCode.MEMBERSHIP_INCOMPLETE,
                stage,
                "The proof does not preserve complete upstream membership.",
                **common,
            )
        )

    artifact_mechanism = str(proof.mechanism) in {
        PopulationLineageMechanism.VERIFIED_ENTITY_SET_ARTIFACT.value,
        PopulationLineageMechanism.VERIFIED_RESULT_ARTIFACT.value,
    }
    if artifact_mechanism:
        if not proof.artifact_evidence:
            gaps.append(
                _gap(
                    PopulationGapCode.ARTIFACT_EVIDENCE_REQUIRED,
                    stage,
                    "Cross-node population lineage requires a verified immutable artifact.",
                    **common,
                )
            )
        expected_artifact_kind = (
            PopulationArtifactKind.ENTITY_SET.value
            if str(proof.mechanism)
            == PopulationLineageMechanism.VERIFIED_ENTITY_SET_ARTIFACT.value
            else "RESULT"
        )
        artifact_passed = False
        artifact_gaps: list[PopulationVerificationGap] = []
        for artifact in proof.artifact_evidence:
            current = _verify_source_artifact(
                artifact,
                expected_artifact_kind=expected_artifact_kind,
                expected_population_fingerprint=(
                    claim.required_scope.population_fingerprint
                ),
                expected_source_artifact_ids=(
                    claim.required_scope.source_artifact_ids
                ),
                trusted_artifact_verifiers=trusted_artifact_verifiers,
                consumer_goal_id=consumer_goal_id,
                query_node_id=claim.query_node_id,
                proof_id=proof.proof_id,
            )
            if current:
                artifact_gaps.extend(current)
            else:
                artifact_passed = True
        if proof.artifact_evidence and not artifact_passed:
            gaps.extend(artifact_gaps)
        expected_artifact_ids = set(claim.required_scope.source_artifact_ids)
        provided_artifact_ids = {
            artifact.artifact_id for artifact in proof.artifact_evidence
        }
        if expected_artifact_ids and not expected_artifact_ids.issubset(
            provided_artifact_ids
        ):
            gaps.append(
                _gap(
                    PopulationGapCode.POPULATION_SOURCE_ARTIFACT_MISMATCH,
                    stage,
                    "The proof does not retain every source artifact that defines the population.",
                    details={
                        "missingSourceArtifactIds": sorted(
                            expected_artifact_ids - provided_artifact_ids
                        )
                    },
                    **common,
                )
            )

    if _is_time_filter_degradation(claim, proof, gaps):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_DEGRADED_TO_TIME_FILTER,
                stage,
                "Only time constraints survived; the upstream entity/relation population was not preserved.",
                **common,
            )
        )
    return _dedupe_gaps(gaps)


def _verify_source_artifact(
    artifact: PopulationArtifactEvidence,
    *,
    expected_artifact_kind: str,
    expected_population_fingerprint: str,
    expected_source_artifact_ids: Sequence[str],
    trusted_artifact_verifiers: Sequence[str],
    consumer_goal_id: str,
    query_node_id: str,
    proof_id: str,
) -> list[PopulationVerificationGap]:
    stage = PopulationVerificationStage.PRE_EXECUTION
    common = {
        "consumer_goal_id": consumer_goal_id,
        "query_node_id": query_node_id,
        "proof_id": proof_id,
        "artifact_id": artifact.artifact_id,
    }
    gaps: list[PopulationVerificationGap] = []
    if artifact.verifier_fingerprint not in set(trusted_artifact_verifiers):
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_UNTRUSTED,
                stage,
                "The population artifact was not verified by a trusted authority.",
                **common,
            )
        )
    if not artifact.verified:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_UNVERIFIED,
                stage,
                "The source population artifact is not verified.",
                **common,
            )
        )
    if not artifact.immutable:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_NOT_IMMUTABLE,
                stage,
                "The source population artifact is not immutable.",
                **common,
            )
        )
    if expected_artifact_kind == PopulationArtifactKind.ENTITY_SET.value:
        kind_valid = str(artifact.artifact_kind) == expected_artifact_kind
    else:
        kind_valid = str(artifact.artifact_kind) in {
            PopulationArtifactKind.QUERY_RESULT.value,
            PopulationArtifactKind.RESULT_RELATION.value,
        }
    if not kind_valid:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_KIND_MISMATCH,
                stage,
                "The artifact kind cannot carry the required population lineage.",
                **common,
            )
        )
    if str(artifact.coverage) not in _COMPLETE_COVERAGE:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_COVERAGE_INCOMPLETE,
                stage,
                "PREVIEW, PARTIAL, or unknown artifact coverage cannot prove a complete population.",
                **common,
            )
        )
    if artifact.population_fingerprint != expected_population_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_POPULATION_MISMATCH,
                stage,
                "The artifact belongs to a different population.",
                **common,
            )
        )
    if expected_source_artifact_ids and artifact.artifact_id not in set(
        expected_source_artifact_ids
    ):
        gaps.append(
            _gap(
                PopulationGapCode.POPULATION_SOURCE_ARTIFACT_MISMATCH,
                stage,
                "The proof substituted a different source artifact.",
                **common,
            )
        )
    return gaps


def _verify_result_artifact(
    artifact: PopulationArtifactEvidence,
    prior_scope: PopulationScopeAttestation,
    *,
    goal_contract_fingerprint: str,
    graph_fingerprint: str,
    trusted_artifact_verifiers: Sequence[str],
    consumer_goal_id: str,
    query_node_id: str,
) -> list[PopulationVerificationGap]:
    stage = PopulationVerificationStage.POST_RESULT
    common = {
        "consumer_goal_id": consumer_goal_id,
        "query_node_id": query_node_id,
        "artifact_id": artifact.artifact_id,
    }
    gaps: list[PopulationVerificationGap] = []
    if artifact.verifier_fingerprint not in set(trusted_artifact_verifiers):
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_UNTRUSTED,
                stage,
                "The result artifact was not verified by a trusted authority.",
                **common,
            )
        )
    if not artifact.verified:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_UNVERIFIED,
                stage,
                "The result artifact is not verified.",
                **common,
            )
        )
    if not artifact.immutable:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_NOT_IMMUTABLE,
                stage,
                "The result artifact is not immutable.",
                **common,
            )
        )
    if str(artifact.artifact_kind) not in {
        PopulationArtifactKind.QUERY_RESULT.value,
        PopulationArtifactKind.RESULT_RELATION.value,
    }:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_KIND_MISMATCH,
                stage,
                "Final population evidence must be a verified result artifact.",
                **common,
            )
        )
    if str(artifact.coverage) not in _COMPLETE_COVERAGE:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_COVERAGE_INCOMPLETE,
                stage,
                "PREVIEW, PARTIAL, or unknown results cannot finalize population semantics.",
                **common,
            )
        )
    if not set(prior_scope.proof_fingerprints).issubset(
        artifact.lineage_proof_fingerprints
    ):
        gaps.append(
            _gap(
                PopulationGapCode.RESULT_LINEAGE_ATTESTATION_MISMATCH,
                stage,
                "The immutable result artifact omits accepted population lineage proofs.",
                **common,
            )
        )
    if artifact.population_fingerprint != prior_scope.population_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.ARTIFACT_POPULATION_MISMATCH,
                stage,
                "The result population differs from the accepted execution population.",
                **common,
            )
        )
    comparisons = (
        (
            PopulationGapCode.ARTIFACT_GOAL_CONTRACT_MISMATCH,
            artifact.goal_contract_fingerprint,
            goal_contract_fingerprint,
            "Goal Contract",
        ),
        (
            PopulationGapCode.ARTIFACT_GRAPH_MISMATCH,
            artifact.graph_fingerprint,
            graph_fingerprint,
            "execution graph",
        ),
        (
            PopulationGapCode.ARTIFACT_QUERY_CONTRACT_MISMATCH,
            artifact.query_contract_fingerprint,
            prior_scope.query_contract_fingerprint,
            "Query Contract",
        ),
        (
            PopulationGapCode.ARTIFACT_SQL_AST_MISMATCH,
            artifact.sql_ast_fingerprint,
            prior_scope.sql_ast_fingerprint,
            "validated SQL AST",
        ),
        (
            PopulationGapCode.ARTIFACT_SNAPSHOT_MISMATCH,
            artifact.snapshot_fingerprint,
            prior_scope.snapshot_fingerprint,
            "data snapshot",
        ),
    )
    for code, actual, expected, label in comparisons:
        if not actual or actual != expected:
            gaps.append(
                _gap(
                    code,
                    stage,
                    "The result artifact is not bound to the accepted %s." % label,
                    **common,
                )
            )
    return gaps


def _is_time_filter_degradation(
    claim: PopulationExecutionClaim,
    proof: PopulationLineageProof,
    proof_gaps: Sequence[PopulationVerificationGap],
) -> bool:
    if str(claim.required_scope.kind) not in _DEPENDENT_SCOPES:
        return False
    if str(proof.mechanism) != (
        PopulationLineageMechanism.SAME_QUERY_PREDICATE_LINEAGE.value
    ):
        return False
    kinds = {str(item.kind) for item in proof.preserved_constraints}
    if kinds != {PopulationConstraintKind.TIME.value}:
        return False
    unsafe_codes = {
        PopulationGapCode.LINEAGE_SOURCE_POPULATION_MISMATCH.value,
        PopulationGapCode.LINEAGE_RESULT_POPULATION_MISMATCH.value,
        PopulationGapCode.POPULATION_SCOPE_FINGERPRINT_MISMATCH.value,
        PopulationGapCode.CONSTRAINT_LINEAGE_INCOMPLETE.value,
        PopulationGapCode.ENTITY_MAPPING_REQUIRED.value,
        PopulationGapCode.GRAIN_MAPPING_REQUIRED.value,
    }
    return bool({str(item.code) for item in proof_gaps} & unsafe_codes)


def _validate_prior_attestation(
    attestation: PopulationVerificationAttestation | None,
    *,
    expected_stage: PopulationVerificationStage,
    goal_contract_fingerprint: str,
    graph_fingerprint: str,
    current_stage: PopulationVerificationStage,
) -> list[PopulationVerificationGap]:
    if attestation is None:
        return [
            _gap(
                PopulationGapCode.PRIOR_ATTESTATION_REQUIRED,
                current_stage,
                "The previous population gate attestation is required.",
            )
        ]
    gaps: list[PopulationVerificationGap] = []
    if str(attestation.stage) != expected_stage.value:
        gaps.append(
            _gap(
                PopulationGapCode.PRIOR_ATTESTATION_STAGE_MISMATCH,
                current_stage,
                "The supplied population attestation is from the wrong lifecycle stage.",
            )
        )
    if (
        not attestation.passed
        or not attestation.gate_open
        or attestation.attestation_fingerprint
        != population_attestation_fingerprint(attestation)
    ):
        gaps.append(
            _gap(
                PopulationGapCode.PRIOR_ATTESTATION_INVALID,
                current_stage,
                "The previous population gate did not pass or its attestation was altered.",
            )
        )
    if attestation.goal_contract_fingerprint != goal_contract_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                current_stage,
                "The previous attestation belongs to a different Goal Contract.",
            )
        )
    if graph_fingerprint and attestation.graph_fingerprint != graph_fingerprint:
        gaps.append(
            _gap(
                PopulationGapCode.PRIOR_ATTESTATION_SCOPE_MISMATCH,
                current_stage,
                "The previous attestation belongs to a different execution graph.",
            )
        )
    return gaps


def _unique_by_consumer(
    values: Sequence[Any],
    consumer_id: Any,
) -> tuple[dict[str, Any], set[str]]:
    mapped: dict[str, Any] = {}
    duplicates: set[str] = set()
    for item in values:
        key = _text(consumer_id(item))
        if key in mapped:
            duplicates.add(key)
        else:
            mapped[key] = item
    return mapped, duplicates


def _gap(
    code: PopulationGapCode,
    stage: PopulationVerificationStage,
    message: str,
    *,
    consumer_goal_id: str = "",
    query_node_id: str = "",
    proof_id: str = "",
    artifact_id: str = "",
    path: str = "",
    details: Mapping[str, Any] | None = None,
) -> PopulationVerificationGap:
    return PopulationVerificationGap(
        code=code,
        stage=stage,
        message=message,
        consumer_goal_id=consumer_goal_id,
        query_node_id=query_node_id,
        proof_id=proof_id,
        artifact_id=artifact_id,
        path=path,
        details=dict(details or {}),
    )


def _dedupe_gaps(
    gaps: Sequence[PopulationVerificationGap],
) -> list[PopulationVerificationGap]:
    retained: list[PopulationVerificationGap] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for gap in gaps:
        identity = (
            str(gap.code),
            gap.consumer_goal_id,
            gap.query_node_id,
            gap.proof_id,
            gap.artifact_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        retained.append(gap)
    return retained


def _result(
    request: PopulationVerificationInput,
    *,
    stage: PopulationVerificationStage,
    goal_contract_fingerprint: str,
    gaps: Sequence[PopulationVerificationGap],
    question_fingerprint: str = "",
    graph_fingerprint: str = "",
    previous_attestation: PopulationVerificationAttestation | None = None,
    accepted_scopes: Sequence[PopulationScopeAttestation] = (),
    accepted_proof_fingerprints: Sequence[str] = (),
    artifact_fingerprints: Sequence[str] = (),
) -> PopulationVerificationResult:
    normalized_gaps = tuple(_dedupe_gaps(gaps))
    passed = not normalized_gaps
    attestation = PopulationVerificationAttestation(
        stage=stage,
        passed=passed,
        gate_open=passed,
        input_fingerprint=_stable_fingerprint(request),
        goal_contract_fingerprint=goal_contract_fingerprint,
        question_fingerprint=question_fingerprint,
        graph_fingerprint=graph_fingerprint,
        accepted_scopes=tuple(
            sorted(accepted_scopes, key=lambda item: item.consumer_goal_id)
        ),
        accepted_proof_fingerprints=tuple(
            sorted(set(accepted_proof_fingerprints))
        ),
        artifact_fingerprints=tuple(sorted(set(artifact_fingerprints))),
        gap_codes=tuple(gap.code for gap in normalized_gaps),
        previous_attestation_fingerprint=(
            previous_attestation.attestation_fingerprint
            if previous_attestation is not None
            else ""
        ),
    )
    attestation = attestation.model_copy(
        update={
            "attestation_fingerprint": population_attestation_fingerprint(
                attestation
            )
        }
    )
    return PopulationVerificationResult(
        stage=stage,
        passed=passed,
        gate_open=passed,
        gaps=normalized_gaps,
        attestation=attestation,
    )
