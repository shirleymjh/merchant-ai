#!/usr/bin/env python3
"""Architecture-focused acceptance runner for real merchant questions.

The runner is intentionally dry by default.  It imports the FastAPI app and
executes real Doris/LLM work only when ``MERCHANT_AI_ACCEPTANCE_REAL=1``.
This keeps the question catalogue and trace parser useful while the grounded
runtime integration is still being completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REAL_RUN_ENV = "MERCHANT_AI_ACCEPTANCE_REAL"
MERCHANT_ID_ENV = "MERCHANT_AI_ACCEPTANCE_MERCHANT_ID"
CASE_IDS_ENV = "MERCHANT_AI_ACCEPTANCE_CASE_IDS"
INCLUDE_SUPPLEMENTAL_ENV = "MERCHANT_AI_ACCEPTANCE_INCLUDE_SUPPLEMENTAL"
OUTPUT_ENV = "MERCHANT_AI_ACCEPTANCE_OUTPUT"


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    question: str
    suite: str
    expected_goal_kinds: tuple[str, ...]
    requires_analysis_artifact: bool = False
    requires_entity_lineage: bool = False


USER_ACCEPTANCE_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        "real_01_order_refund_impact",
        "最近30天订单量和退款金额分别是多少，并分析退款有没有影响订单表现。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_02_deposit_refund_relationship",
        "最近10天保证金缴纳流水和退款金额有没有明显关系？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_03_top_sales_then_refund",
        "上个月销售额最高的前3个商品，以及这些商品最近7天退款量是多少？",
        "user15",
        ("METRIC", "TIME_WINDOW", "RANKING", "ENTITY", "DEPENDENCY"),
        requires_entity_lineage=True,
    ),
    AcceptanceCase(
        "real_04_ticket_status_reminder_trend",
        "最近7天按工单状态统计工单量，再看催单工单量有没有升高。",
        "user15",
        ("METRIC", "DIMENSION", "TIME_WINDOW", "RANKING", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_05_reject_detail_and_top_goods",
        "最近10天商品审核拒绝明细给我看一下，再告诉我拒绝最多的商品有哪些。",
        "user15",
        ("DETAIL", "RANKING", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "real_06_orders_returns_ship_timeout_cause",
        "最近30天订单量、退货量、发货超时订单量为什么一起上升？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_07_fulfilment_timeout_reminder_cause",
        "最近7天履约量、发货超时订单量和催单工单量分别是多少，并分析是不是履约问题导致催单增加。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_08_top_refund_goods_then_details",
        "最近10天退款最多的商品有哪些？再给我看这些商品对应的退款明细。",
        "user15",
        ("RANKING", "ENTITY", "DETAIL", "DEPENDENCY", "TIME_WINDOW"),
        requires_entity_lineage=True,
    ),
    AcceptanceCase(
        "real_09_pay_vs_trade_success",
        "最近7天支付订单量和交易成功订单量分别是多少，差异大不大？",
        "user15",
        ("METRIC", "COMPARISON", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "real_10_deposit_appeal_punish_anomaly",
        "最近30天保证金充值流水、申诉次数和处罚次数分别是多少，有没有异常？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_11_order_refund_details_top3",
        "最近7天订单明细和退款明细都给我看一下，并找出退款金额最高的前3单。",
        "user15",
        ("DETAIL", "RANKING", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "real_12_goods_audit_listing_analysis",
        "最近7天商品审核通过量、审核拒绝量和上架商品量分别是多少，帮我分析商品侧有没有问题。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_13_ticket_details_by_status",
        "最近10天工单明细给我看一下，再按工单状态统计数量。",
        "user15",
        ("DETAIL", "DIMENSION", "RANKING", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "real_14_gmv_refund_ship_timeout",
        "最近30天 GMV 下降是不是和退款金额、发货超时订单量有关？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_15_coupon_orders_gmv_lift",
        "最近7天优惠金额、优惠订单量和 GMV 分别是多少，优惠有没有带来成交提升？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        requires_analysis_artifact=True,
    ),
)


SUPPLEMENTAL_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        "base_single_metric",
        "最近7天订单量是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "base_same_table_two_metrics",
        "最近7天订单量和 GMV 分别是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "base_cross_table",
        "最近7天订单量和发货超时订单量分别是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "base_multiple_details",
        "最近7天订单明细和工单明细分别给我看一下。",
        "supplemental",
        ("DETAIL", "TIME_WINDOW"),
    ),
    AcceptanceCase(
        "base_topn_entity_chain",
        "最近30天退款金额最高的前5个商品，再看这些商品最近7天的工单量。",
        "supplemental",
        ("RANKING", "ENTITY", "DEPENDENCY", "METRIC", "TIME_WINDOW"),
        requires_entity_lineage=True,
    ),
    AcceptanceCase(
        "base_rule",
        "商家入驻资质规则是什么？",
        "supplemental",
        ("RULE",),
    ),
)

ALL_CASES: tuple[AcceptanceCase, ...] = USER_ACCEPTANCE_CASES + SUPPLEMENTAL_CASES


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _pick(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in _sequence(value) if str(item or "").strip()]


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value or "").strip()))


def _all_scalar_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _all_scalar_text(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for child in value for item in _all_scalar_text(child)]
    return [str(value)] if value is not None else []


def _goal_contract(trace: Mapping[str, Any], harness: Mapping[str, Any]) -> dict[str, Any]:
    for owner in (harness, trace):
        for key in (
            "originalQuestionGoalContract",
            "questionGoalContract",
            "goalContract",
            "original_question_goal_contract",
        ):
            candidate = _mapping(owner.get(key))
            if candidate:
                return candidate
    return {}


def _goal_kinds(trace: Mapping[str, Any], harness: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    kinds_by_id: dict[str, str] = {}
    direct = _pick(harness, "goalKinds", "goal_kinds", default=_pick(trace, "goalKinds", "goal_kinds", default=[]))
    direct_kinds: list[str] = []
    if isinstance(direct, Mapping):
        for goal_id, kind in direct.items():
            if str(goal_id or "") and str(kind or ""):
                kinds_by_id[str(goal_id)] = str(kind).upper()
    else:
        direct_kinds.extend(item.upper() for item in _strings(direct))

    contract = _goal_contract(trace, harness)
    raw_goals = _pick(contract, "goals", default=_pick(harness, "originalQuestionGoals", "goals", default=[]))
    for raw in _sequence(raw_goals):
        goal = _mapping(raw)
        goal_id = str(_pick(goal, "goalId", "goal_id", "id", default="") or "")
        kind = str(_pick(goal, "kind", "goalKind", "goal_kind", default="") or "").upper()
        if goal_id and kind:
            kinds_by_id[goal_id] = kind

    goal_coverage = _mapping(_pick(harness, "goalCoverage", "goal_coverage", default={}))
    explicit_by_id = _mapping(_pick(goal_coverage, "goalKindsByGoalId", "goal_kinds_by_goal_id", default={}))
    for goal_id, kind in explicit_by_id.items():
        if str(goal_id or "") and str(kind or ""):
            kinds_by_id[str(goal_id)] = str(kind).upper()

    answer_coverage = _mapping(_pick(harness, "answerCoverage", "answer_coverage", default={}))
    for raw in _sequence(_pick(answer_coverage, "bindings", default=[])):
        binding = _mapping(raw)
        goal_id = str(_pick(binding, "goalId", "goal_id", default="") or "")
        kind = str(_pick(binding, "goalKind", "goal_kind", default="") or "").upper()
        if goal_id and kind:
            kinds_by_id[goal_id] = kind

    return sorted(set(direct_kinds).union(kinds_by_id.values())), kinds_by_id


def _normalized_goal(raw: Any) -> dict[str, Any]:
    goal = _mapping(raw)
    return {
        "goalId": str(_pick(goal, "goalId", "goal_id", "id", default="") or ""),
        "kind": str(_pick(goal, "kind", "goalKind", "goal_kind", default="") or "").upper(),
        "required": bool(_pick(goal, "required", default=True)),
        "dependsOnGoalIds": _strings(
            _pick(goal, "dependsOnGoalIds", "depends_on_goal_ids", default=[])
        ),
        "populationScope": str(
            _pick(goal, "populationScope", "population_scope", default="") or ""
        ).upper(),
        "populationGoalIds": _strings(
            _pick(goal, "populationGoalIds", "population_goal_ids", default=[])
        ),
        "upstreamGoalIds": _strings(
            _pick(goal, "upstreamGoalIds", "upstream_goal_ids", default=[])
        ),
        "downstreamGoalIds": _strings(
            _pick(goal, "downstreamGoalIds", "downstream_goal_ids", default=[])
        ),
        "dependencyType": str(
            _pick(goal, "dependencyType", "dependency_type", default="") or ""
        ).upper(),
        "artifactKind": str(
            _pick(goal, "artifactKind", "artifact_kind", default="") or ""
        ).upper(),
    }


def _goals(trace: Mapping[str, Any], harness: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = _goal_contract(trace, harness)
    raw_goals = _pick(
        contract,
        "goals",
        default=_pick(harness, "originalQuestionGoals", "goals", default=[]),
    )
    return [_normalized_goal(raw) for raw in _sequence(raw_goals)]


def _normalized_branch(raw: Any) -> dict[str, Any]:
    branch = _mapping(raw)
    history = _sequence(_pick(branch, "statusHistory", "status_history", "history", "transitions", default=[]))
    last_gaps = _sequence(_pick(branch, "lastGaps", "last_gaps", default=[]))
    lineage_text = " ".join(_all_scalar_text((history, last_gaps, branch.get("lineage"), branch.get("entityLineage"))))
    lineage_wait = bool(
        _pick(branch, "lineageWaitObserved", "lineage_wait_observed", default=False)
        or "WAITING_VERIFIED_ENTITY_SET" in lineage_text.upper()
        or "WAITING_FOR_LINEAGE" in lineage_text.upper()
    )
    upstream_bindings = _sequence(
        _pick(branch, "upstreamEntityBindings", "upstream_entity_bindings", "entityLineage", default=[])
    )
    upstream_entity_artifact_ids: list[str] = []
    for raw_binding in upstream_bindings:
        binding = _mapping(raw_binding)
        upstream_entity_artifact_ids.extend(
            _strings(_pick(binding, "entitySetArtifactId", "entity_set_artifact_id", "artifactId", default=[]))
        )
    return {
        "queryId": str(_pick(branch, "queryId", "query_id", "branchId", "branch_id", default="") or ""),
        "goalIds": _strings(_pick(branch, "goalIds", "goal_ids", default=[])),
        "topicScope": _strings(_pick(branch, "topicScope", "topic_scope", "topics", default=[])),
        "status": str(_pick(branch, "status", "branchStatus", "branch_status", default="") or ""),
        "dependencyQueryIds": _strings(
            _pick(branch, "dependencyQueryIds", "dependency_query_ids", "upstreamQueryIds", default=[])
        ),
        "dependencyGoalIds": _strings(
            _pick(branch, "dependencyGoalIds", "dependency_goal_ids", "upstreamGoalIds", default=[])
        ),
        "verifiedArtifactIds": _strings(
            _pick(branch, "verifiedArtifactIds", "verified_artifact_ids", default=[])
        ),
        "upstreamEntityArtifactIds": _dedupe(upstream_entity_artifact_ids),
        "lineageWaitObserved": lineage_wait,
        "statusHistory": history,
    }


def _analysis_artifact_ids(harness: Mapping[str, Any], goal_coverage: Mapping[str, Any], answer_coverage: Mapping[str, Any]) -> list[str]:
    artifact_ids = _strings(
        _pick(harness, "verifiedAnalysisArtifactIds", "analysisArtifactIds", "verified_analysis_artifact_ids", default=[])
    )
    for raw in _sequence(_pick(harness, "analysisArtifacts", "analysis_artifacts", default=[])):
        artifact = _mapping(raw)
        artifact_ids.extend(_strings(_pick(artifact, "artifactId", "artifact_id", default=[])))

    proof_types_by_goal = _mapping(
        _pick(goal_coverage, "resolutionProofTypesByGoalId", "resolution_proof_types_by_goal_id", default={})
    )
    artifact_ids_by_goal = _mapping(
        _pick(goal_coverage, "resolutionArtifactIdsByGoalId", "resolution_artifact_ids_by_goal_id", default={})
    )
    for goal_id, proof_types in proof_types_by_goal.items():
        normalized = {item.upper() for item in _strings(proof_types)}
        if "DETERMINISTIC_DERIVED_ANALYSIS" in normalized:
            artifact_ids.extend(_strings(artifact_ids_by_goal.get(goal_id)))

    for raw in _sequence(_pick(answer_coverage, "bindings", default=[])):
        binding = _mapping(raw)
        renderer = str(_pick(binding, "renderer", default="") or "").upper()
        if renderer == "VERIFIED_ANALYSIS_ARTIFACT_RENDERER":
            artifact_ids.extend(_strings(_pick(binding, "artifactIds", "artifact_ids", default=[])))
    return _dedupe(artifact_ids)


def parse_acceptance_response(
    case: AcceptanceCase,
    response_payload: Mapping[str, Any] | Any,
    *,
    status_code: int = 200,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    """Normalize one API/runtime response into the stable acceptance report."""

    if hasattr(response_payload, "model_dump"):
        response_payload = response_payload.model_dump(by_alias=True)
    body = _mapping(response_payload)
    trace = _mapping(_pick(body, "debugTrace", "debug_trace", default={}))
    harness = _mapping(trace.get("harness"))
    goal_coverage = _mapping(_pick(harness, "goalCoverage", "goal_coverage", default={}))
    answer_coverage = _mapping(_pick(harness, "answerCoverage", "answer_coverage", default={}))
    operational_failure = _mapping(
        _pick(harness, "operationalFailure", "operational_failure", default={})
    )
    budget = _mapping(_pick(harness, "runtimeBudget", "runtime_budget", default={}))
    usage = _mapping(_pick(budget, "usage", default={}))
    goal_kinds, goal_kinds_by_id = _goal_kinds(trace, harness)
    goals = _goals(trace, harness)

    branches = [
        _normalized_branch(raw)
        for raw in _sequence(_pick(harness, "queryBranches", "query_branches", "branches", default=[]))
    ]
    query_artifact_ids = _strings(
        _pick(harness, "verifiedQueryArtifactIds", "verified_query_artifact_ids", default=[])
    )
    rule_artifact_ids = _strings(
        _pick(harness, "verifiedRuleArtifactIds", "verified_rule_artifact_ids", default=[])
    )
    coverage_artifact_ids = _strings(_pick(goal_coverage, "artifactIds", "artifact_ids", default=[]))
    analysis_artifact_ids = _analysis_artifact_ids(harness, goal_coverage, answer_coverage)
    all_artifact_ids = _dedupe(
        [*query_artifact_ids, *rule_artifact_ids, *analysis_artifact_ids, *coverage_artifact_ids]
    )

    topic_routing = _mapping(_pick(harness, "topicRouting", "topic_routing", default={}))
    if not topic_routing:
        topic_routing = _mapping(_pick(trace, "topicRouting", "topic_routing", "routing", default={}))
    if not topic_routing and str(_pick(body, "categoryName", "category_name", default="") or ""):
        topic_routing = {"displaySummary": str(_pick(body, "categoryName", "category_name", default=""))}

    stages = _mapping(_pick(budget, "stages", default={}))
    stage_timings_ms = {
        str(name): float(_pick(_mapping(report), "totalDurationMs", "durationMs", default=0.0) or 0.0)
        for name, report in stages.items()
    }
    answer = str(_pick(body, "answer", default="") or "")
    clarification = _mapping(_pick(body, "clarification", default={}))

    summary: dict[str, Any] = {
        "caseId": case.case_id,
        "suite": case.suite,
        "question": case.question,
        "httpStatus": int(status_code),
        "status": "PENDING",
        "answer": answer,
        "topicRouting": topic_routing,
        "goalKinds": goal_kinds,
        "goalKindsById": goal_kinds_by_id,
        "goals": goals,
        "branches": branches,
        "artifactIds": all_artifact_ids,
        "queryArtifactIds": _dedupe(query_artifact_ids),
        "ruleArtifactIds": _dedupe(rule_artifact_ids),
        "analysisArtifactIds": analysis_artifact_ids,
        "usage": {
            "dorisQueries": int(_pick(usage, "dorisQueries", "doris_queries", default=0) or 0),
            "llmCalls": int(_pick(usage, "llmCalls", "llm_calls", default=0) or 0),
            "toolCalls": int(_pick(usage, "toolCalls", "tool_calls", default=0) or 0),
        },
        "stageTimingsMs": stage_timings_ms,
        "elapsedMs": float(elapsed_ms if elapsed_ms is not None else _pick(budget, "elapsedMs", default=0.0) or 0.0),
        "goalCoverage": goal_coverage,
        "answerCoverage": answer_coverage,
        "operationalFailure": operational_failure,
        "clarification": clarification,
        "architectureViolations": [],
        "architecturePassed": False,
        "performanceViolations": [],
        "performancePassed": False,
    }
    violations = architecture_violations(case, summary)
    latency_violations = performance_violations(case, summary)
    summary["architectureViolations"] = violations
    summary["architecturePassed"] = not violations
    summary["performanceViolations"] = latency_violations
    summary["performancePassed"] = not latency_violations
    if status_code >= 400:
        summary["status"] = "HTTP_ERROR"
    elif operational_failure:
        summary["status"] = "OPERATIONAL_FAILURE"
    elif clarification:
        summary["status"] = "CLARIFICATION"
    else:
        summary["status"] = "PASS" if not violations else "FAIL"
    return summary


def _coverage_value(coverage: Mapping[str, Any], camel: str, snake: str, default: Any = None) -> Any:
    return _pick(coverage, camel, snake, default=default)


def _typed_analysis_insufficiency(summary: Mapping[str, Any]) -> bool:
    goal_coverage = _mapping(summary.get("goalCoverage"))
    answer_coverage = _mapping(summary.get("answerCoverage"))
    kinds_by_id = _mapping(summary.get("goalKindsById"))
    resolution_by_id = _mapping(
        _coverage_value(goal_coverage, "resolutionByGoalId", "resolution_by_goal_id", {})
    )
    reasons = _mapping(
        _coverage_value(goal_coverage, "insufficiencyReasonByGoalId", "insufficiency_reason_by_goal_id", {})
    )
    refs_by_id = _mapping(
        _coverage_value(
            goal_coverage,
            "resolutionEvidenceRefsByGoalId",
            "resolution_evidence_refs_by_goal_id",
            {},
        )
    )
    analysis_goal_ids = {goal_id for goal_id, kind in kinds_by_id.items() if str(kind).upper() == "ANALYSIS"}
    if not analysis_goal_ids:
        return False
    bindings = {
        str(_pick(binding, "goalId", "goal_id", default="")): _mapping(binding)
        for binding in (_mapping(raw) for raw in _sequence(answer_coverage.get("bindings")))
    }
    for goal_id in analysis_goal_ids:
        binding = bindings.get(goal_id, {})
        if (
            str(resolution_by_id.get(goal_id, "")).upper() == "INSUFFICIENT_EVIDENCE"
            and str(reasons.get(goal_id, "")).strip()
            and _strings(refs_by_id.get(goal_id))
            and str(_pick(binding, "resolution", default="")).upper() == "INSUFFICIENT_EVIDENCE"
            and str(_pick(binding, "renderer", default="")).upper() == "VERIFIED_INSUFFICIENCY_RENDERER"
            and str(_pick(binding, "insufficiencyRef", "insufficiency_ref", default="")).strip()
        ):
            return True
    return False


def _goal_coverage_map(
    goal_coverage: Mapping[str, Any],
    camel: str,
    snake: str,
) -> dict[str, Any]:
    return _mapping(_coverage_value(goal_coverage, camel, snake, {}))


def _branch_relation_is_declared(
    *,
    upstream_goal_id: str,
    downstream_goal_id: str,
    branches: Sequence[Mapping[str, Any]],
) -> bool:
    upstream_query_ids = {
        str(branch.get("queryId") or "")
        for branch in branches
        if upstream_goal_id in _strings(branch.get("goalIds"))
    }
    downstream_branches = [
        branch
        for branch in branches
        if downstream_goal_id in _strings(branch.get("goalIds"))
    ]
    if not upstream_query_ids or not downstream_branches:
        return False
    if any(
        str(branch.get("queryId") or "") in upstream_query_ids
        for branch in downstream_branches
    ):
        return True
    return all(
        bool(
            upstream_query_ids.intersection(
                _strings(branch.get("dependencyQueryIds"))
            )
        )
        for branch in downstream_branches
    )


def _typed_lineage_relations(goals: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    for goal in goals:
        goal_id = str(goal.get("goalId") or "")
        kind = str(goal.get("kind") or "").upper()
        if kind == "RANKING" and str(goal.get("populationScope") or "") != "ALL_MATCHING_ROWS":
            relations.extend(
                (population_goal_id, goal_id, "RANKING_POPULATION")
                for population_goal_id in _strings(goal.get("populationGoalIds"))
            )
        if kind != "DEPENDENCY":
            continue
        dependency_type = str(goal.get("dependencyType") or "")
        artifact_kind = str(goal.get("artifactKind") or "")
        if dependency_type in {"CONTRACT_SCOPE", "PREDICATE_SCOPE"}:
            continue
        if not (
            dependency_type in {"ENTITY_CHAIN", "RESULT_CHAIN"}
            or artifact_kind
            in {
                "ENTITY_SET",
                "RESULT_ARTIFACT",
                "VERIFIED_ENTITY_SET",
                "VERIFIED_RESULT_ARTIFACT",
            }
        ):
            continue
        relations.extend(
            (upstream_goal_id, downstream_goal_id, goal_id)
            for upstream_goal_id in _strings(goal.get("upstreamGoalIds"))
            for downstream_goal_id in _strings(goal.get("downstreamGoalIds"))
        )
    return list(dict.fromkeys(relations))


def architecture_violations(case: AcceptanceCase, summary: Mapping[str, Any]) -> list[str]:
    """Return only architectural safety failures; numeric values are never asserted."""

    violations: list[str] = []
    status_code = int(summary.get("httpStatus") or 0)
    operational_failure = _mapping(summary.get("operationalFailure"))
    clarification = _mapping(summary.get("clarification"))
    answer = str(summary.get("answer") or "")
    final_answer = bool(answer and status_code < 400 and not operational_failure and not clarification)

    if operational_failure and not str(_pick(operational_failure, "code", default="") or ""):
        violations.append("OPERATIONAL_FAILURE_MUST_BE_TYPED")
    if not final_answer:
        return violations

    goal_coverage = _mapping(summary.get("goalCoverage"))
    answer_coverage = _mapping(summary.get("answerCoverage"))
    if not bool(_coverage_value(goal_coverage, "passed", "passed", False)):
        violations.append("FINAL_ANSWER_REQUIRES_PASSED_GOAL_COVERAGE")
    if not bool(_coverage_value(goal_coverage, "finalizationAllowed", "finalization_allowed", False)):
        violations.append("FINAL_ANSWER_REQUIRES_GOAL_FINALIZATION_GATE")
    if not bool(_coverage_value(answer_coverage, "passed", "passed", False)):
        violations.append("FINAL_ANSWER_REQUIRES_PASSED_ANSWER_COVERAGE")
    source = str(_coverage_value(answer_coverage, "source", "source", "") or "")
    if source not in {"compose_verified_answer", "compose_verified_rule_answer", "run_skill"}:
        violations.append("FINAL_ANSWER_SOURCE_IS_NOT_TRUSTED")
    expected_fingerprint = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    actual_fingerprint = str(
        _coverage_value(answer_coverage, "answerFingerprint", "answer_fingerprint", "") or ""
    )
    if actual_fingerprint != expected_fingerprint:
        violations.append("FINAL_ANSWER_ATTESTATION_FINGERPRINT_MISMATCH")

    goal_kinds = {str(item).upper() for item in _sequence(summary.get("goalKinds"))}
    if not goal_kinds:
        violations.append("GOAL_KINDS_NOT_EXPOSED")
    missing_expected_kinds = {
        str(item).upper() for item in case.expected_goal_kinds
    } - goal_kinds
    if missing_expected_kinds:
        violations.append("EXPECTED_GOAL_KINDS_MISSING")

    goals = [_mapping(raw) for raw in _sequence(summary.get("goals"))]
    goal_by_id = {
        str(goal.get("goalId") or ""): goal
        for goal in goals
        if str(goal.get("goalId") or "")
    }
    if not goal_by_id or len(goal_by_id) != len(goals):
        violations.append("TYPED_GOAL_CONTRACT_NOT_EXPOSED")
    required_goal_ids = {
        goal_id
        for goal_id, goal in goal_by_id.items()
        if bool(goal.get("required", True))
    }

    artifact_ids = set(_strings(summary.get("artifactIds")))
    if not artifact_ids:
        violations.append("FINAL_ANSWER_HAS_NO_VERIFIED_ARTIFACT")

    coverage_required_goal_ids = set(
        _strings(
            _coverage_value(
                goal_coverage,
                "requiredGoalIds",
                "required_goal_ids",
                [],
            )
        )
    )
    if required_goal_ids != coverage_required_goal_ids:
        violations.append("GOAL_COVERAGE_REQUIRED_SET_MISMATCH")
    resolutions = _goal_coverage_map(
        goal_coverage,
        "resolutionByGoalId",
        "resolution_by_goal_id",
    )
    proof_types_by_goal = _goal_coverage_map(
        goal_coverage,
        "resolutionProofTypesByGoalId",
        "resolution_proof_types_by_goal_id",
    )
    artifact_ids_by_goal = _goal_coverage_map(
        goal_coverage,
        "resolutionArtifactIdsByGoalId",
        "resolution_artifact_ids_by_goal_id",
    )
    evidence_refs_by_goal = _goal_coverage_map(
        goal_coverage,
        "resolutionEvidenceRefsByGoalId",
        "resolution_evidence_refs_by_goal_id",
    )
    insufficiency_reasons = _goal_coverage_map(
        goal_coverage,
        "insufficiencyReasonByGoalId",
        "insufficiency_reason_by_goal_id",
    )
    for goal_id in required_goal_ids:
        resolution = str(resolutions.get(goal_id) or "").upper()
        if resolution not in {"PROVED", "INSUFFICIENT_EVIDENCE"}:
            violations.append("REQUIRED_GOAL_RESOLUTION_NOT_TYPED")
            continue
        if resolution == "PROVED":
            proof_types = _strings(proof_types_by_goal.get(goal_id))
            proved_artifact_ids = set(_strings(artifact_ids_by_goal.get(goal_id)))
            if not proof_types:
                violations.append("PROVED_GOAL_REQUIRES_TYPED_PROOF")
            if not proved_artifact_ids:
                violations.append("PROVED_GOAL_REQUIRES_VERIFIED_ARTIFACT")
            elif not proved_artifact_ids.issubset(artifact_ids):
                violations.append("GOAL_PROOF_REFERENCES_UNVERIFIED_ARTIFACT")
        elif not (
            str(insufficiency_reasons.get(goal_id) or "").strip()
            and _strings(evidence_refs_by_goal.get(goal_id))
        ):
            violations.append("INSUFFICIENT_GOAL_REQUIRES_TYPED_EVIDENCE_GAP")

    answer_required_goal_ids = set(
        _strings(
            _coverage_value(
                answer_coverage,
                "requiredGoalIds",
                "required_goal_ids",
                [],
            )
        )
    )
    if answer_required_goal_ids != required_goal_ids:
        violations.append("ANSWER_COVERAGE_REQUIRED_SET_MISMATCH")
    binding_by_goal_id: dict[str, dict[str, Any]] = {}
    for raw in _sequence(answer_coverage.get("bindings")):
        binding = _mapping(raw)
        goal_id = str(_pick(binding, "goalId", "goal_id", default="") or "")
        if not goal_id or goal_id in binding_by_goal_id:
            violations.append("ANSWER_BINDING_GOAL_ID_INVALID")
            continue
        binding_by_goal_id[goal_id] = binding
        supplied_artifacts = set(
            _strings(_pick(binding, "artifactIds", "artifact_ids", default=[]))
        )
        supplied_refs = set(
            _strings(_pick(binding, "evidenceRefs", "evidence_refs", default=[]))
        )
        if not supplied_artifacts.issubset(artifact_ids):
            violations.append("ANSWER_BINDING_REFERENCES_UNVERIFIED_ARTIFACT")
        allowed_artifacts = set(_strings(artifact_ids_by_goal.get(goal_id)))
        allowed_refs = set(_strings(evidence_refs_by_goal.get(goal_id)))
        if not supplied_artifacts.issubset(allowed_artifacts) or not supplied_refs.issubset(allowed_refs):
            violations.append("ANSWER_BINDING_REFERENCES_DIFFERENT_GOAL_EVIDENCE")
        resolution = str(resolutions.get(goal_id) or "").upper()
        if str(_pick(binding, "resolution", default="") or "").upper() != resolution:
            violations.append("ANSWER_BINDING_RESOLUTION_MISMATCH")
        if resolution == "PROVED" and not (
            supplied_artifacts.intersection(allowed_artifacts)
            or supplied_refs.intersection(allowed_refs)
        ):
            violations.append("ANSWER_BINDING_REQUIRES_FINAL_RESULT_EVIDENCE")
        if resolution == "INSUFFICIENT_EVIDENCE" and str(
            _pick(binding, "insufficiencyRef", "insufficiency_ref", default="") or ""
        ) not in allowed_refs:
            violations.append("ANSWER_BINDING_REQUIRES_TYPED_INSUFFICIENCY_REF")
    if required_goal_ids - set(binding_by_goal_id):
        violations.append("FINAL_ANSWER_MISSING_REQUIRED_GOAL_BINDING")

    for goal in goals:
        if str(goal.get("kind") or "") != "RANKING":
            continue
        goal_id = str(goal.get("goalId") or "")
        population_scope = str(goal.get("populationScope") or "")
        valid_population_scopes = {
            "ALL_MATCHING_ROWS",
            "SAME_AS_GOAL",
            "VERIFIED_ENTITY_SET",
            "VERIFIED_PREDICATE_SCOPE",
            "VERIFIED_RESULT_ARTIFACT",
        }
        if not population_scope:
            violations.append("RANKING_POPULATION_SCOPE_NOT_DECLARED")
        elif population_scope not in valid_population_scopes:
            violations.append("RANKING_POPULATION_SCOPE_INVALID")
        population_goal_ids = set(_strings(goal.get("populationGoalIds")))
        if not population_goal_ids.issubset(goal_by_id):
            violations.append("RANKING_POPULATION_GOAL_UNKNOWN")
        if population_scope in {"SAME_AS_GOAL", "VERIFIED_ENTITY_SET"} and not population_goal_ids:
            violations.append("RANKING_POPULATION_GOAL_REQUIRED")
        if population_scope in {
            "ALL_MATCHING_ROWS",
            "VERIFIED_PREDICATE_SCOPE",
            "VERIFIED_RESULT_ARTIFACT",
        } and population_goal_ids:
            violations.append("RANKING_POPULATION_GOAL_UNEXPECTED")
        if any(
            str(goal_by_id[population_goal_id].get("kind") or "")
            not in {"DETAIL", "ENTITY", "RANKING"}
            for population_goal_id in population_goal_ids.intersection(goal_by_id)
        ):
            violations.append("RANKING_POPULATION_GOAL_KIND_INVALID")
        if population_goal_ids and any(
            str(resolutions.get(goal_id) or "").upper() != "PROVED"
            for goal_id in population_goal_ids
        ):
            violations.append("RANKING_POPULATION_GOALS_NOT_PROVED")
        proof_types = {
            item.upper() for item in _strings(proof_types_by_goal.get(goal_id))
        }
        if (
            str(resolutions.get(goal_id) or "").upper() == "PROVED"
            and "VERIFIED_ORDERED_ROW_SET" not in proof_types
        ):
            violations.append("RANKING_REQUIRES_VERIFIED_POPULATION_PROOF")
        if population_scope != "ALL_MATCHING_ROWS" and not _strings(
            evidence_refs_by_goal.get(goal_id)
        ):
            violations.append("RANKING_POPULATION_LINEAGE_NOT_ATTESTED")

    dependency_goals = [
        goal for goal in goals if str(goal.get("kind") or "") == "DEPENDENCY"
    ]
    for goal in dependency_goals:
        goal_id = str(goal.get("goalId") or "")
        referenced_goal_ids = {
            *_strings(goal.get("upstreamGoalIds")),
            *_strings(goal.get("downstreamGoalIds")),
        }
        if not referenced_goal_ids or not referenced_goal_ids.issubset(goal_by_id):
            violations.append("DEPENDENCY_GOAL_RELATION_INCOMPLETE")
        if referenced_goal_ids and any(
            str(resolutions.get(reference_goal_id) or "").upper() != "PROVED"
            for reference_goal_id in referenced_goal_ids
        ):
            violations.append("DEPENDENCY_RELATION_GOALS_NOT_PROVED")
        proof_types = {
            item.upper() for item in _strings(proof_types_by_goal.get(goal_id))
        }
        if (
            str(resolutions.get(goal_id) or "").upper() == "PROVED"
            and "VERIFIED_ARTIFACT_LINEAGE" not in proof_types
        ):
            violations.append("DEPENDENCY_REQUIRES_VERIFIED_ARTIFACT_LINEAGE")

    if case.requires_analysis_artifact:
        has_analysis_artifact = bool(_strings(summary.get("analysisArtifactIds")))
        if not has_analysis_artifact and not _typed_analysis_insufficiency(summary):
            violations.append("ANALYSIS_REQUIRES_VERIFIED_ARTIFACT_OR_TYPED_INSUFFICIENCY")

    branches = [_mapping(raw) for raw in _sequence(summary.get("branches"))]
    for branch in branches:
        if not (
            str(branch.get("queryId") or "")
            and _strings(branch.get("goalIds"))
            and _strings(branch.get("topicScope"))
            and str(branch.get("status") or "")
        ):
            violations.append("BRANCH_SCOPE_OR_STATUS_INCOMPLETE")
            break
        if str(branch.get("status") or "").upper() != "VERIFIED" or not _strings(
            branch.get("verifiedArtifactIds")
        ):
            violations.append("DECLARED_BRANCH_REQUIRES_VERIFIED_RESULT")
            break

    kinds_by_id = _mapping(summary.get("goalKindsById"))
    query_goal_kinds = {"METRIC", "DIMENSION", "TIME_WINDOW", "ENTITY", "DETAIL", "RANKING"}
    query_goal_ids = {goal_id for goal_id, kind in kinds_by_id.items() if str(kind).upper() in query_goal_kinds}
    assigned_goal_ids = {goal_id for branch in branches for goal_id in _strings(branch.get("goalIds"))}
    if branches and query_goal_ids - assigned_goal_ids:
        violations.append("QUERY_GOALS_MUST_BE_ASSIGNED_TO_BRANCH")

    lineage_relations = _typed_lineage_relations(goals)
    if case.requires_entity_lineage and not lineage_relations:
        violations.append("ENTITY_LINEAGE_RELATION_NOT_DECLARED")
    if branches and any(
        not _branch_relation_is_declared(
            upstream_goal_id=upstream_goal_id,
            downstream_goal_id=downstream_goal_id,
            branches=branches,
        )
        for upstream_goal_id, downstream_goal_id, _ in lineage_relations
    ):
        violations.append("CROSS_NODE_LINEAGE_REQUIRES_EXECUTION_DEPENDENCY")

    return _dedupe(violations)


def performance_violations(
    case: AcceptanceCase,
    summary: Mapping[str, Any],
) -> list[str]:
    """Enforce stable call-count budgets without asserting provider latency."""

    if case.case_id != "base_single_metric" or not str(
        summary.get("answer") or ""
    ).strip():
        return []
    usage = _mapping(summary.get("usage"))
    violations: list[str] = []
    if int(usage.get("llmCalls") or 0) > 3:
        violations.append("SINGLE_METRIC_LLM_CALL_BUDGET_EXCEEDED")
    if int(usage.get("toolCalls") or 0) > 3:
        violations.append("SINGLE_METRIC_TOOL_CALL_BUDGET_EXCEEDED")
    if int(usage.get("dorisQueries") or 0) != 1:
        violations.append("SINGLE_METRIC_DORIS_QUERY_COUNT_INVALID")
    return violations


def select_cases(case_ids: Sequence[str] | None = None, *, include_supplemental: bool = True) -> list[AcceptanceCase]:
    available = ALL_CASES if include_supplemental else USER_ACCEPTANCE_CASES
    selected = {item.strip() for item in (case_ids or []) if item.strip()}
    if not selected:
        return list(available)
    known = {case.case_id for case in available}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError("unknown acceptance case IDs: %s" % ", ".join(unknown))
    return [case for case in available if case.case_id in selected]


def run_real_cases(cases: Sequence[AcceptanceCase], merchant_id: str) -> list[dict[str, Any]]:
    """Execute through the public FastAPI contract. Called only after the env gate."""

    from fastapi.testclient import TestClient

    from app.main import app, run_manager

    results: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for case in cases:
            started = time.perf_counter()
            existing_run_ids = set(run_manager.runs)
            try:
                response = client.post(
                    "/api/chat",
                    json={"message": case.question, "merchantId": merchant_id},
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                payload = response.json() if response.content else {}
                new_runs = [
                    run
                    for run_id, run in run_manager.runs.items()
                    if run_id not in existing_run_ids
                    and str(run.merchant_id) == merchant_id
                    and str(run.question) == case.question
                ]
                if len(new_runs) == 1 and new_runs[0].trace_path:
                    trace_path = Path(new_runs[0].trace_path)
                    if trace_path.is_file():
                        trace = json.loads(trace_path.read_text(encoding="utf-8"))
                        if isinstance(trace, Mapping):
                            payload = {
                                **_mapping(payload),
                                "debugTrace": dict(trace),
                            }
                results.append(
                    parse_acceptance_response(
                        case,
                        payload,
                        status_code=response.status_code,
                        elapsed_ms=elapsed_ms,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "caseId": case.case_id,
                        "suite": case.suite,
                        "question": case.question,
                        "status": "ERROR",
                        "answer": "",
                        "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
                        "operationalFailure": {
                            "code": "ACCEPTANCE_RUNNER_EXCEPTION",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "architecturePassed": False,
                        "architectureViolations": ["RUNNER_COULD_NOT_PARSE_RUNTIME_RESPONSE"],
                    }
                )
    return results


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the case catalogue and exit")
    parser.add_argument("--case", action="append", default=[], help="run one case ID; repeat as needed")
    parser.add_argument("--output", default=os.getenv(OUTPUT_ENV, ""), help="real-run JSON output path")
    return parser.parse_args(argv)


def _catalogue_payload(cases: Sequence[AcceptanceCase]) -> dict[str, Any]:
    return {
        "mode": "DRY_RUN",
        "realRunEnv": REAL_RUN_ENV,
        "caseCount": len(cases),
        "userCaseCount": sum(case.suite == "user15" for case in cases),
        "supplementalCaseCount": sum(case.suite == "supplemental" for case in cases),
        "cases": [asdict(case) for case in cases],
        "message": f"No Doris/LLM calls were made. Set {REAL_RUN_ENV}=1 to execute.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env_case_ids = [item for item in os.getenv(CASE_IDS_ENV, "").split(",") if item.strip()]
    include_supplemental = _env_true(INCLUDE_SUPPLEMENTAL_ENV, default=True)
    cases = select_cases([*env_case_ids, *args.case], include_supplemental=include_supplemental)
    if args.list or not _env_true(REAL_RUN_ENV):
        print(json.dumps(_catalogue_payload(cases), ensure_ascii=False, indent=2))
        return 0

    merchant_id = os.getenv(MERCHANT_ID_ENV, "100").strip() or "100"
    results = run_real_cases(cases, merchant_id)
    report = {
        "mode": "REAL",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "merchantId": merchant_id,
        "caseCount": len(results),
        "passed": sum(item.get("status") == "PASS" for item in results),
        "results": results,
    }
    output = args.output or ".merchant-ai/acceptance_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output_path.resolve()), **{key: report[key] for key in ("caseCount", "passed")}}, ensure_ascii=False))
    return 0 if all(item.get("status") == "PASS" for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
