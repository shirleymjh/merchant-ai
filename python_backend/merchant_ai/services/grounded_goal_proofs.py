from __future__ import annotations

from typing import Any, Mapping, Sequence

from merchant_ai.models import ResultCoverage
from merchant_ai.services.grounded_goal_contract import (
    ComparisonQuestionGoal,
    DependencyQuestionGoal,
    DetailQuestionGoal,
    GoalProofResolution,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
)


def derive_query_artifact_goal_resolutions(
    *,
    goal_contract: OriginalQuestionGoalContract,
    artifact: Any,
    assigned_goal_ids: Sequence[str],
    artifact_goal_ids: Mapping[str, Sequence[str]],
    all_artifacts: Sequence[Any],
) -> list[GoalProofResolution | dict[str, Any]]:
    """Derive only proofs that are mechanically present in query artifacts.

    Interpretive ANALYSIS/RULE goals are intentionally absent.  They require a
    dedicated verified artifact and can never be completed merely because a
    query artifact was assigned their goal ID.
    """

    goal_map = goal_contract.goal_map()
    artifact_id = str(getattr(artifact, "artifact_id", "") or "")
    contract = getattr(artifact, "contract", None)
    run_result = getattr(artifact, "run_result", None)
    bundle = getattr(run_result, "merged_query_bundle", None)
    rows = list(getattr(bundle, "rows", None) or [])
    output_columns = list(getattr(artifact, "output_columns", None) or [])
    evidence_refs = list(getattr(contract, "evidence_refs", None) or [])
    resolutions: list[GoalProofResolution | dict[str, Any]] = []
    for goal_id in _dedupe_strings(assigned_goal_ids):
        goal = goal_map.get(goal_id)
        if isinstance(goal, DetailQuestionGoal):
            if not output_columns:
                continue
            result_coverage = _bundle_result_coverage(bundle)
            detail_complete = (
                result_coverage == ResultCoverage.ALL_ROWS.value
                and not bool(getattr(bundle, "failed", False))
                and not bool(getattr(bundle, "is_truncated", False))
            )
            if not detail_complete:
                resolutions.append(
                    {
                        "goalId": goal_id,
                        "goalKind": "DETAIL",
                        "resolution": "INSUFFICIENT_EVIDENCE",
                        "proofType": "QUERY_RESULT_COVERAGE_INCOMPLETE",
                        "evidenceRefs": evidence_refs,
                        "outputFields": output_columns,
                        "rowSetRef": artifact_id,
                        "rowCount": len(rows),
                        "reason": (
                            "detail result coverage is %s; a capped, truncated, "
                            "or unclassified row set cannot prove all requested rows"
                            % result_coverage
                        ),
                        "details": {
                            "resultCoverage": result_coverage,
                            "visibleRowCount": len(rows),
                            "isTruncated": bool(
                                getattr(bundle, "is_truncated", False)
                            ),
                        },
                    }
                )
                continue
            resolutions.append(
                {
                    "goalId": goal_id,
                    "goalKind": "DETAIL",
                    "resolution": "PROVED",
                    "proofType": "VERIFIED_QUERY_ROW_SET",
                    "evidenceRefs": evidence_refs,
                    "outputFields": output_columns,
                    "rowSetRef": artifact_id,
                    "rowCount": int(
                        getattr(bundle, "original_row_count", 0) or len(rows)
                    ),
                }
            )
        elif isinstance(goal, RankingQuestionGoal):
            ranking = getattr(contract, "ranking", None)
            if (
                str(getattr(contract, "query_shape", "") or "").upper()
                != "RANKED"
                or not bool(getattr(ranking, "enabled", False))
            ):
                continue
            ranking_receipt = _ranking_execution_receipt(
                artifact,
                contract=contract,
                bundle=bundle,
            )
            population_proof = _ranking_population_proof(
                goal,
                contract=contract,
                artifact_goal_ids=artifact_goal_ids,
                all_artifacts=all_artifacts,
            )
            if not ranking_receipt or population_proof is None:
                continue
            population_scope, population_goal_ids, population_lineage_refs = (
                population_proof
            )
            resolutions.append(
                {
                    "goalId": goal_id,
                    "goalKind": "RANKING",
                    "resolution": "PROVED",
                    "proofType": "VERIFIED_ORDERED_ROW_SET",
                    "evidenceRefs": _dedupe_strings(
                        [*evidence_refs, *ranking_receipt]
                    ),
                    "orderByGoalIds": list(goal.metric_goal_ids),
                    "direction": str(getattr(ranking, "direction", "") or "").upper(),
                    "limit": int(getattr(ranking, "limit", 0) or 0),
                    "rowSetRef": artifact_id,
                    "populationScope": population_scope,
                    "populationGoalIds": population_goal_ids,
                    "populationLineageRefs": population_lineage_refs,
                }
            )
        elif isinstance(goal, ComparisonQuestionGoal):
            resolution = _ranked_comparison_resolution(
                goal,
                artifact_id=artifact_id,
                contract=contract,
                evidence_refs=evidence_refs,
                ranking_receipt=_ranking_execution_receipt(
                    artifact,
                    contract=contract,
                    bundle=bundle,
                ),
            )
            if resolution is not None:
                resolutions.append(resolution)
        elif isinstance(goal, DependencyQuestionGoal):
            resolution = _dependency_resolution(
                goal,
                artifact_id=artifact_id,
                artifact_goal_ids=artifact_goal_ids,
                all_artifacts=all_artifacts,
                evidence_refs=evidence_refs,
            )
            if resolution is not None:
                resolutions.append(resolution)
    return resolutions


def _ranked_comparison_resolution(
    goal: ComparisonQuestionGoal,
    *,
    artifact_id: str,
    contract: Any,
    evidence_refs: Sequence[str],
    ranking_receipt: Sequence[str],
) -> dict[str, Any] | None:
    ranking = getattr(contract, "ranking", None)
    if (
        str(getattr(contract, "query_shape", "") or "").upper() != "RANKED"
        or not bool(getattr(ranking, "enabled", False))
        or not ranking_receipt
    ):
        return None
    comparison_type = _canonical_capability(getattr(goal, "comparison_type", ""))
    if comparison_type not in {
        "RANK",
        "RANKING",
        "TOPN",
        "TOP_1",
        "RANK_DESC_TOP_1",
        "RANK_ASC_TOP_1",
    } and not comparison_type.startswith(("RANK_", "TOP_")):
        return None
    direction = str(getattr(ranking, "direction", "") or "").upper()
    limit = int(getattr(ranking, "limit", 0) or 0)
    return {
        "goalId": goal.goal_id,
        "goalKind": "COMPARISON",
        "resolution": "PROVED",
        "proofType": "VERIFIED_RANKING_RESULT",
        "evidenceRefs": _dedupe_strings([*evidence_refs, *ranking_receipt]),
        "comparisonType": comparison_type,
        "operandGoalIds": [*goal.left_goal_ids, *goal.right_goal_ids],
        "comparisonMethod": "ORDER_BY_%s_LIMIT_%d" % (direction, limit),
        "resultRef": artifact_id,
    }


def _ranking_execution_receipt(
    artifact: Any,
    *,
    contract: Any,
    bundle: Any,
) -> list[str]:
    """Accept only ranking semantics sealed by the Kernel execution path."""

    verified = getattr(artifact, "verified_evidence", None)
    if not bool(getattr(verified, "passed", False)):
        return []
    if not bool(getattr(artifact, "ranking_semantics_verified", False)):
        return []
    ranking = getattr(contract, "ranking", None)
    limit = int(getattr(ranking, "limit", 0) or 0)
    rows = list(getattr(bundle, "rows", None) or [])
    if limit <= 0 or len(rows) > limit:
        return []
    if _bundle_result_coverage(bundle) != ResultCoverage.TOP_N.value:
        return []
    if bool(getattr(bundle, "failed", False)) or bool(
        getattr(bundle, "is_truncated", False)
    ):
        return []

    contract_fingerprint = str(
        getattr(artifact, "contract_fingerprint", "") or ""
    ).strip()
    sql_fingerprint = str(
        getattr(artifact, "sql_fingerprint", "") or ""
    ).strip()
    validation = getattr(artifact, "sql_validation", None)
    if validation is not None:
        if (
            not bool(getattr(validation, "valid", False))
            or not str(getattr(validation, "ast_fingerprint", "") or "").strip()
            or (
                contract_fingerprint
                and str(
                    getattr(validation, "contract_fingerprint", "") or ""
                ).strip()
                != contract_fingerprint
            )
            or (
                sql_fingerprint
                and str(getattr(validation, "ast_fingerprint", "") or "").strip()
                != sql_fingerprint
            )
        ):
            return []
    return _dedupe_strings(
        [
            "query-artifact:%s"
            % str(getattr(artifact, "artifact_id", "") or ""),
            "contract-fingerprint:%s" % contract_fingerprint,
            "sql-fingerprint:%s" % sql_fingerprint,
            (
                "sql-validation:%s"
                % str(getattr(validation, "ast_fingerprint", "") or "")
                if validation is not None
                else "deterministic-ranking-contract:%s" % contract_fingerprint
            ),
        ]
    )


def _ranking_population_proof(
    goal: RankingQuestionGoal,
    *,
    contract: Any,
    artifact_goal_ids: Mapping[str, Sequence[str]],
    all_artifacts: Sequence[Any],
) -> tuple[str, list[str], list[str]] | None:
    """Derive population only from executable artifact lineage.

    Goal declarations describe intent; they are never evidence.  In
    particular, ordinary semantic refs cannot prove that a downstream query
    used the row population produced by another goal.
    """

    reference = getattr(contract, "reference_scope", None)
    if bool(getattr(reference, "enabled", False)) and bool(
        getattr(reference, "population_required", False)
    ):
        if not bool(getattr(reference, "executable", False)):
            return None
        scope_by_type = {
            "PREDICATE_SCOPE": "VERIFIED_PREDICATE_SCOPE",
            "ENTITY_SET": "VERIFIED_ENTITY_SET",
            "RESULT_ARTIFACT": "VERIFIED_RESULT_ARTIFACT",
        }
        population_scope = scope_by_type.get(
            str(getattr(reference, "referent_type", "") or "").upper()
        )
        if not population_scope:
            return None
        source_artifact_id = str(
            getattr(reference, "source_artifact_id", "") or ""
        ).strip()
        source_contract_fingerprint = str(
            getattr(reference, "source_contract_fingerprint", "") or ""
        ).strip()
        source_sql_fingerprint = str(
            getattr(reference, "source_sql_fingerprint", "") or ""
        ).strip()
        if not source_artifact_id or not source_contract_fingerprint:
            return None
        return (
            population_scope,
            [],
            _dedupe_strings(
                [
                    "query-artifact:%s" % source_artifact_id,
                    "contract-fingerprint:%s" % source_contract_fingerprint,
                    "sql-fingerprint:%s" % source_sql_fingerprint,
                ]
            ),
        )

    if goal.population_scope == "ALL_MATCHING_ROWS":
        return "ALL_MATCHING_ROWS", [], []

    upstream_bindings = list(
        getattr(contract, "upstream_entity_bindings", None) or []
    )
    if not upstream_bindings:
        return None
    artifacts_by_id = {
        str(getattr(item, "artifact_id", "") or ""): item
        for item in all_artifacts
        if str(getattr(item, "artifact_id", "") or "")
    }
    population_artifact_ids: set[str] = set()
    for population_goal_id in goal.population_goal_ids:
        matching = {
            artifact_id
            for artifact_id, goal_ids in artifact_goal_ids.items()
            if population_goal_id in set(goal_ids or [])
        }
        if not matching:
            return None
        population_artifact_ids.update(matching)
    bound_sources = {
        str(getattr(item, "source_query_artifact_id", "") or "")
        for item in upstream_bindings
        if str(getattr(item, "source_query_artifact_id", "") or "")
    }
    if not population_artifact_ids or not population_artifact_ids.issubset(
        bound_sources
    ):
        return None
    if any(
        artifact_id not in artifacts_by_id
        or not bool(
            getattr(
                getattr(artifacts_by_id[artifact_id], "verified_evidence", None),
                "passed",
                False,
            )
        )
        for artifact_id in population_artifact_ids
    ):
        return None
    lineage: list[str] = []
    for item in upstream_bindings:
        source_id = str(
            getattr(item, "source_query_artifact_id", "") or ""
        ).strip()
        if source_id not in population_artifact_ids:
            continue
        lineage.extend(
            [
                "query-artifact:%s" % source_id,
                "contract-fingerprint:%s"
                % str(getattr(item, "source_contract_fingerprint", "") or ""),
                "sql-fingerprint:%s"
                % str(getattr(item, "source_sql_fingerprint", "") or ""),
                "entity-set:%s"
                % str(getattr(item, "entity_set_artifact_id", "") or ""),
            ]
        )
    return (
        goal.population_scope,
        list(goal.population_goal_ids),
        _dedupe_strings(lineage),
    )


def _dependency_resolution(
    goal: DependencyQuestionGoal,
    *,
    artifact_id: str,
    artifact_goal_ids: Mapping[str, Sequence[str]],
    all_artifacts: Sequence[Any],
    evidence_refs: Sequence[str],
) -> dict[str, Any] | None:
    upstream_ids = _artifacts_covering_goals(
        goal.upstream_goal_ids,
        artifact_goal_ids,
    )
    downstream_ids = _artifacts_covering_goals(
        goal.downstream_goal_ids,
        artifact_goal_ids,
    )
    if not upstream_ids and set(goal.upstream_goal_ids).intersection(
        artifact_goal_ids.get(artifact_id, ())
    ):
        upstream_ids = [artifact_id]
    if not downstream_ids and set(goal.downstream_goal_ids).intersection(
        artifact_goal_ids.get(artifact_id, ())
    ):
        downstream_ids = [artifact_id]
    known_ids = {
        str(getattr(item, "artifact_id", "") or "") for item in all_artifacts
    }
    upstream_ids = [item for item in upstream_ids if item in known_ids]
    downstream_ids = [item for item in downstream_ids if item in known_ids]
    if not upstream_ids or not downstream_ids:
        return None
    lineage_refs = _dedupe_strings(
        [
            *evidence_refs,
            *[
                ref
                for item in all_artifacts
                if str(getattr(item, "artifact_id", "") or "")
                in set([*upstream_ids, *downstream_ids])
                for refs in (getattr(item, "output_lineage", {}) or {}).values()
                for ref in refs
            ],
        ]
    )
    if not lineage_refs:
        return None
    return {
        "goalId": goal.goal_id,
        "goalKind": "DEPENDENCY",
        "resolution": "PROVED",
        "proofType": "VERIFIED_ARTIFACT_LINEAGE",
        "evidenceRefs": list(evidence_refs),
        "upstreamArtifactIds": upstream_ids,
        "downstreamArtifactIds": downstream_ids,
        "lineageRefs": lineage_refs,
    }


def _artifacts_covering_goals(
    goal_ids: Sequence[str],
    artifact_goal_ids: Mapping[str, Sequence[str]],
) -> list[str]:
    required = set(goal_ids)
    return [
        artifact_id
        for artifact_id, assigned in artifact_goal_ids.items()
        if required.intersection(assigned)
    ]


def _canonical_capability(value: Any) -> str:
    characters: list[str] = []
    separator_pending = False
    for character in str(value or "").strip().upper():
        codepoint = ord(character)
        is_ascii_symbol = (
            ord("A") <= codepoint <= ord("Z")
            or ord("0") <= codepoint <= ord("9")
        )
        if is_ascii_symbol:
            if separator_pending and characters and characters[-1] != "_":
                characters.append("_")
            characters.append(character)
            separator_pending = False
        elif characters:
            separator_pending = True
    return "".join(characters)


def _bundle_result_coverage(bundle: Any) -> str:
    value = getattr(bundle, "result_coverage", ResultCoverage.UNKNOWN.value)
    return str(getattr(value, "value", value) or ResultCoverage.UNKNOWN.value).upper()


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
