from __future__ import annotations

from typing import Any, Dict, List, Set

from merchant_ai.models import (
    AgentTaskResult,
    EvidenceGap,
    NodeExecutionContext,
    NodePlanContract,
)
from merchant_ai.services.query_sql_binding import (
    normalize_identifier,
    sql_has_bound_scope_column_values,
)


def contract_gaps_from_task_results(task_results: List[AgentTaskResult]) -> List[EvidenceGap]:
    gaps: List[EvidenceGap] = []
    for task_result in task_results:
        critique = task_result.node_plan_critique
        if not critique or critique.valid or not critique.graph_repairable:
            continue
        gaps.append(
            EvidenceGap(
                code=critique.code or "PLAN_CONTRACT_MISMATCH",
                task_id=task_result.task_id,
                evidence=contract_issue_evidence(task_result),
                reason=critique.message,
                severity="error",
                disclosure_required=True,
                source="execution_contract_validator",
                answer_instruction="当前 node plan contract 与执行要求不一致，应先修 QueryGraph，不要把它解释成无数据。",
            )
        )
    return gaps


def sql_repair_gaps_from_task_results(task_results: List[AgentTaskResult]) -> List[EvidenceGap]:
    """Expose terminal SQL repair states as typed evidence gaps."""

    gaps: List[EvidenceGap] = []
    for task_result in task_results:
        attempts = list(task_result.sql_repairs or [])
        if not attempts or not task_result.query_bundle.failed:
            continue
        terminal = attempts[-1]
        if terminal.error_code == "REPAIR_NO_PROGRESS" or terminal.status == "no_progress":
            code = "REPAIR_NO_PROGRESS"
            reason = terminal.error_message or "SQL repair did not change the canonical SQL state"
        elif terminal.exhausted:
            code = "SQL_REPAIR_EXHAUSTED"
            reason = terminal.observation or terminal.error_message or "SQL repair budget was exhausted"
        else:
            continue
        gaps.append(
            EvidenceGap(
                code=code,
                task_id=task_result.task_id,
                evidence=terminal.state_fingerprint or terminal.input_sql_hash,
                reason=reason,
                severity="blocking",
                disclosure_required=True,
                source="node_sql_repair",
                answer_instruction="SQL 修复没有形成新的可执行查询；禁止把失败结果解释为无数据或指标为 0。",
                suggested_action="replan_or_fix_sql_contract",
                details={
                    "gapCode": code,
                    "taskId": task_result.task_id,
                    "repairRound": terminal.round,
                    "sourceErrorCode": terminal.source_error_code or terminal.error_code,
                    "inputSqlHash": terminal.input_sql_hash,
                    "outputSqlHash": terminal.output_sql_hash,
                    "contractHash": terminal.contract_hash,
                    "stateFingerprint": terminal.state_fingerprint,
                    "status": terminal.status,
                    "exhausted": terminal.exhausted,
                },
            )
        )
    return gaps


def contract_issue_evidence(task_result: AgentTaskResult) -> str:
    critique = task_result.node_plan_critique
    contract = task_result.node_plan_contract
    issues = getattr(critique, "issues", []) or []
    if issues:
        first = issues[0]
        evidence = str(first.get("evidence") or first.get("code") or "")
        if evidence:
            return evidence[:240]
    if contract and contract.preferred_table:
        return "%s:%s" % (contract.preferred_table, ",".join(contract.allowed_columns[:8]))
    return ""


def tenant_filter_columns(contract: NodePlanContract) -> Set[str]:
    column = normalize_identifier(contract.merchant_filter_column)
    return {column} if column else set()


def tenant_scope_binding_error(
    bound_sql: str,
    params: List[Any],
    contract: NodePlanContract,
    context: NodeExecutionContext,
) -> str:
    tenant_columns = tenant_filter_columns(contract)
    if not tenant_columns:
        return ""
    merchant_id = str(context.merchant_id or "").strip()
    if not merchant_id:
        return "缺少当前请求 merchant_id，不能执行带商家域的数据查询"
    tenant_column = next(iter(tenant_columns))
    if not sql_has_bound_scope_column_values(
        bound_sql,
        tenant_column,
        params,
        [merchant_id],
        contract.preferred_table,
    ):
        return "SQL 商家过滤没有被后端参数绑定到当前 merchant，禁止执行跨商家风险查询"
    if contract.authorized_region and contract.region_filter_column:
        if not sql_has_bound_scope_column_values(
            bound_sql,
            contract.region_filter_column,
            params,
            [contract.authorized_region],
            contract.preferred_table,
        ):
            return "SQL Region 过滤没有被后端参数绑定，禁止执行跨 Region 查询"
    if contract.authorized_store_ids and contract.store_filter_column:
        if not sql_has_bound_scope_column_values(
            bound_sql,
            contract.store_filter_column,
            params,
            list(contract.authorized_store_ids),
            contract.preferred_table,
        ):
            return "SQL 门店过滤没有被后端参数绑定，禁止执行跨门店查询"
    return ""


def collect_degraded_reasons(task_results: List[AgentTaskResult]) -> List[Dict[str, Any]]:
    reasons: List[Dict[str, Any]] = []
    seen = set()
    for task_result in task_results:
        for trace in task_result.node_tool_traces:
            code = trace.error_type or degraded_code_for_tool(trace.tool_name, trace.status)
            if not code:
                continue
            key = (task_result.task_id, trace.tool_name, code, trace.repair_round)
            if key in seen:
                continue
            seen.add(key)
            reasons.append(
                {
                    "taskId": task_result.task_id,
                    "tool": trace.tool_name,
                    "status": trace.status,
                    "code": code,
                    "reason": trace.output_summary,
                    "repairRound": trace.repair_round,
                }
            )
    return reasons


def degraded_code_for_tool(tool_name: str, status: str) -> str:
    name = str(tool_name or "")
    state = str(status or "").lower()
    if state in {"failed", "error", "skipped"}:
        return "TOOL_%s" % state.upper()
    if name in {
        "draft_structured_sql_fallback",
        "draft_resource_safe_sql_fallback",
        "execute_sql_split_fallback",
        "select_realtime_fallback",
    }:
        return name.upper()
    return ""
