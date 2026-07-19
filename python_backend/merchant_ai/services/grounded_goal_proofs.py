from __future__ import annotations

from typing import Any, Mapping, Sequence

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
            resolutions.append(
                {
                    "goalId": goal_id,
                    "goalKind": "DETAIL",
                    "resolution": "PROVED",
                    "proofType": "VERIFIED_QUERY_ROW_SET",
                    "evidenceRefs": evidence_refs,
                    "outputFields": output_columns,
                    "rowSetRef": artifact_id,
                    "rowCount": len(rows),
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
            resolutions.append(
                {
                    "goalId": goal_id,
                    "goalKind": "RANKING",
                    "resolution": "PROVED",
                    "proofType": "VERIFIED_ORDERED_ROW_SET",
                    "evidenceRefs": evidence_refs,
                    "orderByGoalIds": list(goal.metric_goal_ids),
                    "direction": str(getattr(ranking, "direction", "") or "").upper(),
                    "limit": int(getattr(ranking, "limit", 0) or 0),
                    "rowSetRef": artifact_id,
                }
            )
        elif isinstance(goal, ComparisonQuestionGoal):
            resolution = _ranked_comparison_resolution(
                goal,
                artifact_id=artifact_id,
                contract=contract,
                evidence_refs=evidence_refs,
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
) -> dict[str, Any] | None:
    ranking = getattr(contract, "ranking", None)
    if (
        str(getattr(contract, "query_shape", "") or "").upper() != "RANKED"
        or not bool(getattr(ranking, "enabled", False))
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
        "evidenceRefs": list(evidence_refs),
        "comparisonType": comparison_type,
        "operandGoalIds": [*goal.left_goal_ids, *goal.right_goal_ids],
        "comparisonMethod": "ORDER_BY_%s_LIMIT_%d" % (direction, limit),
        "resultRef": artifact_id,
    }


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
    import re

    return re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(value or "").strip().upper(),
    ).strip("_")


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
