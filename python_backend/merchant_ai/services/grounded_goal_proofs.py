from __future__ import annotations

from typing import Any, Mapping, Sequence

from merchant_ai.models import ResultCoverage
from merchant_ai.services.grounded_goal_contract import (
    ComparisonQuestionGoal,
    DependencyQuestionGoal,
    DetailQuestionGoal,
    DimensionQuestionGoal,
    EntityQuestionGoal,
    GoalProofResolution,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    TimeWindowQuestionGoal,
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
    output_semantic_refs = _artifact_output_semantic_refs(artifact)
    evidence_refs = list(getattr(contract, "evidence_refs", None) or [])
    resolutions: list[GoalProofResolution | dict[str, Any]] = []
    for goal_id in _dedupe_strings(assigned_goal_ids):
        goal = goal_map.get(goal_id)
        if isinstance(goal, TimeWindowQuestionGoal):
            time_proof = _time_window_resolution(
                goal_id=goal_id,
                contract=contract,
                evidence_refs=evidence_refs,
            )
            if time_proof is not None:
                resolutions.append(time_proof)
        elif isinstance(goal, DetailQuestionGoal):
            if not output_columns:
                continue
            result_coverage = _bundle_result_coverage(bundle)
            detail_complete = (
                result_coverage == ResultCoverage.ALL_ROWS.value
                and not bool(getattr(bundle, "failed", False))
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
                        "outputSemanticRefs": output_semantic_refs,
                        "rowSetRef": artifact_id,
                        "rowCount": len(rows),
                        "reason": (
                            "detail result coverage is %s; a capped or unclassified "
                            "row set cannot prove all requested rows"
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
                    "outputSemanticRefs": output_semantic_refs,
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
            metric_goal_ids, dimension_goal_ids, binding_details = (
                _ranking_goal_bindings(
                    goal,
                    goal_map=goal_map,
                    contract=contract,
                )
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
                    "orderByGoalIds": metric_goal_ids,
                    "dimensionGoalIds": dimension_goal_ids,
                    "rankingMetricRefId": str(
                        getattr(ranking, "metric_ref_id", "") or ""
                    ).strip(),
                    "rankingDimensionRefId": str(
                        getattr(ranking, "dimension_ref_id", "") or ""
                    ).strip(),
                    "direction": str(getattr(ranking, "direction", "") or "").upper(),
                    "limit": int(getattr(ranking, "limit", 0) or 0),
                    "rowSetRef": artifact_id,
                    "populationScope": population_scope,
                    "populationGoalIds": population_goal_ids,
                    "populationLineageRefs": population_lineage_refs,
                    "details": binding_details,
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


def _time_window_resolution(
    *,
    goal_id: str,
    contract: Any,
    evidence_refs: Sequence[str],
) -> dict[str, Any] | None:
    """Build a TIME_WINDOW proof exclusively from the executed Contract."""

    time_range = getattr(contract, "time_range", None)
    if time_range is None:
        return None
    start = str(
        getattr(time_range, "execution_start_date", "")
        or getattr(time_range, "execution_start_value", "")
        or getattr(time_range, "start_date", "")
        or ""
    ).strip()
    end = str(
        getattr(time_range, "execution_end_date", "")
        or getattr(time_range, "execution_end_value", "")
        or getattr(time_range, "end_date", "")
        or ""
    ).strip()
    hints = getattr(contract, "binding_hints", None)
    expression = str(
        getattr(time_range, "label", "")
        or getattr(hints, "time_expression", "")
        or ""
    ).strip()
    actual = {
        "timeExpression": expression,
        "start": start,
        "end": end,
        "timezone": str(getattr(time_range, "timezone", "") or "").strip(),
        "granularity": str(getattr(time_range, "granularity", "") or "").strip(),
        "days": int(getattr(time_range, "days", 0) or 0),
        "label": str(getattr(time_range, "label", "") or "").strip(),
        "explicit": bool(getattr(time_range, "explicit", False)),
        "calendarAnchorPolicy": str(
            getattr(time_range, "calendar_anchor_policy", "") or ""
        ).strip(),
        "dataAsOfPolicy": str(
            getattr(time_range, "data_as_of_policy", "") or ""
        ).strip(),
        "windowRole": str(
            getattr(time_range, "window_role", "") or ""
        ).strip(),
        "timeRangeKind": str(getattr(time_range, "kind", "") or "").strip(),
    }
    if not actual["explicit"] or actual["days"] <= 0:
        return {
            "goalId": goal_id,
            "goalKind": "TIME_WINDOW",
            "resolution": "INSUFFICIENT_EVIDENCE",
            "proofType": "QUERY_TIME_RANGE_NOT_EXPLICIT",
            "evidenceRefs": list(evidence_refs),
            **actual,
            "reason": (
                "the verified query artifact does not contain an explicit "
                "positive executed time window"
            ),
            "details": {
                "explicit": actual["explicit"],
                "days": actual["days"],
            },
        }
    return {
        "goalId": goal_id,
        "goalKind": "TIME_WINDOW",
        "resolution": "PROVED",
        "proofType": "VERIFIED_QUERY_TIME_RANGE",
        "evidenceRefs": list(evidence_refs),
        **actual,
    }


def _artifact_output_semantic_refs(artifact: Any) -> list[str]:
    values: list[Any] = []
    semantic_map = getattr(artifact, "output_semantic_refs", None) or {}
    if isinstance(semantic_map, Mapping):
        values.extend(semantic_map.values())
    lineage = getattr(artifact, "output_lineage", None) or {}
    if isinstance(lineage, Mapping):
        for refs in lineage.values():
            if isinstance(refs, (str, bytes)):
                values.append(refs)
            elif isinstance(refs, Sequence):
                values.extend(refs)
    return _dedupe_strings(values)


def _ranking_goal_bindings(
    goal: RankingQuestionGoal,
    *,
    goal_map: Mapping[str, Any],
    contract: Any,
) -> tuple[list[str], list[str], dict[str, Any]]:
    ranking = getattr(contract, "ranking", None)
    metric_ref = str(getattr(ranking, "metric_ref_id", "") or "").strip()
    dimension_ref = str(
        getattr(ranking, "dimension_ref_id", "") or ""
    ).strip()
    metric_ids, metric_mode = _goal_ids_matching_ref(
        metric_ref,
        goal.metric_goal_ids,
        goal_map=goal_map,
        contract_bindings=list(getattr(contract, "metrics", None) or []),
        allowed_goal_types=(MetricQuestionGoal,),
    )
    dimension_ids, dimension_mode = _goal_ids_matching_ref(
        dimension_ref,
        goal.dimension_goal_ids,
        goal_map=goal_map,
        contract_bindings=list(getattr(contract, "dimensions", None) or []),
        allowed_goal_types=(DimensionQuestionGoal, EntityQuestionGoal),
    )
    return (
        metric_ids,
        dimension_ids,
        {
            "metricBindingMode": metric_mode,
            "dimensionBindingMode": dimension_mode,
        },
    )


def _goal_ids_matching_ref(
    actual_ref: str,
    declared_goal_ids: Sequence[str],
    *,
    goal_map: Mapping[str, Any],
    contract_bindings: Sequence[Any],
    allowed_goal_types: tuple[type, ...],
) -> tuple[list[str], str]:
    if not actual_ref:
        return [], "ACTUAL_REF_MISSING"
    actual = _canonical_semantic_ref(actual_ref)
    matches = [
        goal_id
        for goal_id in declared_goal_ids
        if isinstance(goal_map.get(goal_id), allowed_goal_types)
        and actual in _goal_refs(goal_map[goal_id])
    ]
    if matches:
        return _dedupe_strings(matches), "SEMANTIC_REF_MATCH"

    binding_refs = [
        _canonical_semantic_ref(getattr(item, "semantic_ref_id", ""))
        for item in contract_bindings
        if str(getattr(item, "semantic_ref_id", "") or "").strip()
    ]
    if (
        len(declared_goal_ids) == 1
        and len(binding_refs) == 1
        and binding_refs[0] == actual
        and isinstance(goal_map.get(declared_goal_ids[0]), allowed_goal_types)
    ):
        return [declared_goal_ids[0]], "UNIQUE_CONTRACT_BINDING"
    return [], "NO_GOAL_REF_MATCH"


def _goal_refs(goal: Any) -> set[str]:
    values = list(getattr(goal, "semantic_ref_ids", None) or [])
    for field_name in ("metric_ref_id", "dimension_ref_id", "entity_ref_id"):
        value = str(getattr(goal, field_name, "") or "").strip()
        if value:
            values.append(value)
    return {
        value
        for value in (_canonical_semantic_ref(item) for item in values)
        if value
    }


def _canonical_semantic_ref(value: Any) -> str:
    return str(value or "").strip().casefold().replace(":column:", ":field:")


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
