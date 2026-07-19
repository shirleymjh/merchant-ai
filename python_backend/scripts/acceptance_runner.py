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
    minimum_branch_count: int = 0
    requires_analysis_artifact: bool = False
    requires_entity_lineage: bool = False


USER_ACCEPTANCE_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        "real_01_order_refund_impact",
        "最近30天订单量和退款金额分别是多少，并分析退款有没有影响订单表现。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=1,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_02_deposit_refund_relationship",
        "最近10天保证金缴纳流水和退款金额有没有明显关系？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_03_top_sales_then_refund",
        "上个月销售额最高的前3个商品，以及这些商品最近7天退款量是多少？",
        "user15",
        ("METRIC", "TIME_WINDOW", "RANKING", "ENTITY", "DEPENDENCY"),
        minimum_branch_count=2,
        requires_entity_lineage=True,
    ),
    AcceptanceCase(
        "real_04_ticket_status_reminder_trend",
        "最近7天按工单状态统计工单量，再看催单工单量有没有升高。",
        "user15",
        ("METRIC", "DIMENSION", "TIME_WINDOW", "RANKING", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_05_reject_detail_and_top_goods",
        "最近10天商品审核拒绝明细给我看一下，再告诉我拒绝最多的商品有哪些。",
        "user15",
        ("DETAIL", "RANKING", "TIME_WINDOW"),
        minimum_branch_count=2,
    ),
    AcceptanceCase(
        "real_06_orders_returns_ship_timeout_cause",
        "最近30天订单量、退货量、发货超时订单量为什么一起上升？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_07_fulfilment_timeout_reminder_cause",
        "最近7天履约量、发货超时订单量和催单工单量分别是多少，并分析是不是履约问题导致催单增加。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_08_top_refund_goods_then_details",
        "最近10天退款最多的商品有哪些？再给我看这些商品对应的退款明细。",
        "user15",
        ("RANKING", "ENTITY", "DETAIL", "DEPENDENCY", "TIME_WINDOW"),
        minimum_branch_count=2,
        requires_entity_lineage=True,
    ),
    AcceptanceCase(
        "real_09_pay_vs_trade_success",
        "最近7天支付订单量和交易成功订单量分别是多少，差异大不大？",
        "user15",
        ("METRIC", "COMPARISON", "TIME_WINDOW"),
        minimum_branch_count=1,
    ),
    AcceptanceCase(
        "real_10_deposit_appeal_punish_anomaly",
        "最近30天保证金充值流水、申诉次数和处罚次数分别是多少，有没有异常？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_11_order_refund_details_top3",
        "最近7天订单明细和退款明细都给我看一下，并找出退款金额最高的前3单。",
        "user15",
        ("DETAIL", "RANKING", "TIME_WINDOW"),
        minimum_branch_count=2,
    ),
    AcceptanceCase(
        "real_12_goods_audit_listing_analysis",
        "最近7天商品审核通过量、审核拒绝量和上架商品量分别是多少，帮我分析商品侧有没有问题。",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=1,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_13_ticket_details_by_status",
        "最近10天工单明细给我看一下，再按工单状态统计数量。",
        "user15",
        ("DETAIL", "DIMENSION", "RANKING", "TIME_WINDOW"),
        minimum_branch_count=2,
    ),
    AcceptanceCase(
        "real_14_gmv_refund_ship_timeout",
        "最近30天 GMV 下降是不是和退款金额、发货超时订单量有关？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
    AcceptanceCase(
        "real_15_coupon_orders_gmv_lift",
        "最近7天优惠金额、优惠订单量和 GMV 分别是多少，优惠有没有带来成交提升？",
        "user15",
        ("METRIC", "TIME_WINDOW", "ANALYSIS"),
        minimum_branch_count=2,
        requires_analysis_artifact=True,
    ),
)


SUPPLEMENTAL_CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase(
        "base_single_metric",
        "最近7天订单量是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
        minimum_branch_count=1,
    ),
    AcceptanceCase(
        "base_same_table_two_metrics",
        "最近7天订单量和 GMV 分别是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
        minimum_branch_count=1,
    ),
    AcceptanceCase(
        "base_cross_table",
        "最近7天订单量和发货超时订单量分别是多少？",
        "supplemental",
        ("METRIC", "TIME_WINDOW"),
        minimum_branch_count=2,
    ),
    AcceptanceCase(
        "base_multiple_details",
        "最近7天订单明细和工单明细分别给我看一下。",
        "supplemental",
        ("DETAIL", "TIME_WINDOW"),
        minimum_branch_count=2,
    ),
    AcceptanceCase(
        "base_topn_entity_chain",
        "最近30天退款金额最高的前5个商品，再看这些商品最近7天的工单量。",
        "supplemental",
        ("RANKING", "ENTITY", "DEPENDENCY", "METRIC", "TIME_WINDOW"),
        minimum_branch_count=2,
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
    }
    violations = architecture_violations(case, summary)
    summary["architectureViolations"] = violations
    summary["architecturePassed"] = not violations
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
    artifact_ids = set(_strings(summary.get("artifactIds")))
    if not artifact_ids:
        violations.append("FINAL_ANSWER_HAS_NO_VERIFIED_ARTIFACT")
    for raw in _sequence(answer_coverage.get("bindings")):
        binding = _mapping(raw)
        supplied = set(_strings(_pick(binding, "artifactIds", "artifact_ids", default=[])))
        if not supplied.issubset(artifact_ids):
            violations.append("ANSWER_BINDING_REFERENCES_UNVERIFIED_ARTIFACT")
            break

    if case.requires_analysis_artifact:
        has_analysis_artifact = bool(_strings(summary.get("analysisArtifactIds")))
        if not has_analysis_artifact and not _typed_analysis_insufficiency(summary):
            violations.append("ANALYSIS_REQUIRES_VERIFIED_ARTIFACT_OR_TYPED_INSUFFICIENCY")

    branches = [_mapping(raw) for raw in _sequence(summary.get("branches"))]
    if len(branches) < case.minimum_branch_count:
        violations.append("INDEPENDENT_GOALS_REQUIRE_DECLARED_BRANCHES")
    for branch in branches:
        if not (
            str(branch.get("queryId") or "")
            and _strings(branch.get("goalIds"))
            and _strings(branch.get("topicScope"))
            and str(branch.get("status") or "")
        ):
            violations.append("BRANCH_SCOPE_OR_STATUS_INCOMPLETE")
            break

    kinds_by_id = _mapping(summary.get("goalKindsById"))
    query_goal_kinds = {"METRIC", "DIMENSION", "TIME_WINDOW", "ENTITY", "DEPENDENCY", "DETAIL", "RANKING"}
    query_goal_ids = {goal_id for goal_id, kind in kinds_by_id.items() if str(kind).upper() in query_goal_kinds}
    assigned_goal_ids = {goal_id for branch in branches for goal_id in _strings(branch.get("goalIds"))}
    if query_goal_ids - assigned_goal_ids:
        violations.append("QUERY_GOALS_MUST_BE_ASSIGNED_TO_BRANCH")

    if case.requires_entity_lineage:
        branch_by_id = {str(branch.get("queryId") or ""): branch for branch in branches}
        dependent = [branch for branch in branches if _strings(branch.get("dependencyQueryIds"))]
        lineage_safe = False
        for branch in dependent:
            dependencies = _strings(branch.get("dependencyQueryIds"))
            upstream_verified = all(
                dependency in branch_by_id and _strings(branch_by_id[dependency].get("verifiedArtifactIds"))
                for dependency in dependencies
            )
            lineage_bound = bool(
                branch.get("lineageWaitObserved")
                and (upstream_verified or _strings(branch.get("upstreamEntityArtifactIds")))
            )
            lineage_safe = lineage_safe or lineage_bound
        if not lineage_safe:
            violations.append("ENTITY_CHAIN_MUST_WAIT_FOR_VERIFIED_LINEAGE")

    return _dedupe(violations)


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

    from app.main import app

    results: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post(
                    "/api/chat",
                    json={"message": case.question, "merchantId": merchant_id},
                )
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                payload = response.json() if response.content else {}
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
