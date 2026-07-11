from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTask,
    AgentTaskResult,
    AnswerMode,
    EntitySet,
    EvidenceCheckResult,
    EvidenceGap,
    FreshnessCheckResult,
    IntentType,
    NodeAgentContext,
    NodeExecutionBatch,
    NodeExecutionContext,
    NodePlanContract,
    NodePlanCritiqueResult,
    NodeTaskProfile,
    NodeToolCall,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    ReActStep,
    SqlDraftDecision,
    SqlRepairAttempt,
    SqlValidationResult,
    TaskRole,
    ToolCallRequest,
    ToolCachePolicy,
    ToolCallExecutionResult,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import (
    default_row_access_policy,
    normalize_column_display_policy,
    normalize_masking_policy,
    normalize_row_access_policy,
    normalize_visibility_policy,
)
from merchant_ai.services.formulas import (
    compile_metric_formula as compile_reconciled_metric_formula,
    formula_columns as reconciled_formula_columns,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.planning import EvidenceContractBuilder
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.query_contracts import (
    collect_degraded_reasons,
    contract_gaps_from_task_results,
    tenant_scope_binding_error,
)
from merchant_ai.services.repositories import DorisRepository
from merchant_ai.services.runtime_state import NodeTaskState, create_runtime_state_store, node_task_idempotency_key
from merchant_ai.services.query_sql_binding import (
    append_note,
    bind_node_sql_parameters,
    blank_entity_value,
    has_merchant_filter_predicate,
    is_dependent_context_column,
    parse_partition_date,
    partition_is_stale_for_near_realtime,
    quote_identifier,
    realtime_fallback_for_table,
    split_detail_sql_by_pt_windows,
    sql_has_bound_merchant_filter,
    sql_literal,
)
from merchant_ai.services.query_security import (
    DEFAULT_ACCESS_ROLE,
    apply_column_masks,
    configured_contract_detail_columns,
    configured_default_detail_columns,
    role_allowed_for_column,
    table_asset_metadata,
    table_field_semantics,
)
from merchant_ai.services.tool_runtime import ToolFailureRegistry, ToolRuntimePolicyRegistry, ToolRuntimeService, classify_timeout_type
from merchant_ai.services.tools import artifact_file_tool_definitions, canonical_tool_registry, node_runtime_tool_schemas, semantic_file_tool_definitions, sql_draft_tool, sql_repair_tool


SQL_BUILTIN_IDENTIFIERS = {"current_date", "current_timestamp", "current_time", "curdate", "now"}
STRUCTURED_FALLBACK_ERROR_CODES = {
    "SQL_EMPTY",
    "PARSE_ERROR",
    "UNKNOWN_COLUMN",
    "UNKNOWN_CONTRACT_COLUMN",
    "UNKNOWN_BASE_TABLE",
    "OUT_OF_NODE_TABLE_SCOPE",
    "MISSING_MERCHANT_FILTER",
    "MISSING_PARTITION_FILTER",
    "INVALID_PARTITION_FILTER",
    "MISSING_OUTPUT_KEY",
    "MISSING_ENTITY_KEY_FILTER",
    "MEM_ALLOC_FAILED",
    "TIMEOUT",
    "SQL_SYNTAX",
    "DORIS_ERROR",
}
STRICT_STRUCTURED_FALLBACK_CODES = {
    "MISSING_OUTPUT_KEY",
    "UNKNOWN_CONTRACT_COLUMN",
    "INVALID_PARTITION_FILTER",
    "MISSING_ENTITY_KEY_FILTER",
}
RESOURCE_CONSTRAINED_DORIS_ERRORS = {"MEM_ALLOC_FAILED", "TIMEOUT"}
class SqlValidationService:
    def validate(self, sql: str, asset_pack: PlanningAssetPack) -> SqlValidationResult:
        normalized = (sql or "").strip()
        if not normalized:
            return SqlValidationResult(valid=False, error_code="SQL_EMPTY", message="SQL 为空")
        if ";" in normalized.rstrip(";"):
            return SqlValidationResult(valid=False, error_code="MULTI_STATEMENT", message="SQL 不允许包含多语句")
        try:
            parsed = sqlglot.parse_one(normalized, read="doris")
        except Exception as exc:
            return SqlValidationResult(valid=False, error_code="PARSE_ERROR", message="SQL 解析失败: %s" % str(exc)[:200])
        if not isinstance(parsed, (exp.Select, exp.Union)) and not parsed.find(exp.Select):
            return SqlValidationResult(valid=False, error_code="NOT_SELECT", message="只允许 SELECT/WITH 查询")
        forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create)
        if any(parsed.find(kind) for kind in forbidden):
            return SqlValidationResult(valid=False, error_code="UNSAFE_SQL", message="SQL 包含写操作或 DDL")

        cte_names = {cte.alias for cte in parsed.find_all(exp.CTE) if cte.alias}
        table_aliases: Dict[str, str] = {}
        base_tables: List[str] = []
        for table in parsed.find_all(exp.Table):
            name = table.name
            if not name or name in cte_names:
                continue
            if name not in base_tables:
                base_tables.append(name)
            alias = table.alias_or_name
            if alias:
                table_aliases[alias] = name
            table_aliases[name] = name

        known_tables = set(asset_pack.known_tables())
        unknown_tables = sorted(table for table in base_tables if known_tables and table not in known_tables)
        if unknown_tables:
            return SqlValidationResult(
                valid=False,
                error_code="UNKNOWN_BASE_TABLE",
                message="SQL 引用了 assetPack 外的真实表: %s" % unknown_tables,
                base_tables=base_tables,
                cte_names=sorted(cte_names),
                unknown_tables=unknown_tables,
            )

        unknown_columns = sql_scope_unknown_columns(parsed, asset_pack)
        if unknown_columns:
            return SqlValidationResult(
                valid=False,
                error_code="UNKNOWN_COLUMN",
                message="SQL 引用了未知字段: %s" % sorted(set(unknown_columns)),
                base_tables=base_tables,
                cte_names=sorted(cte_names),
                unknown_columns=sorted(set(unknown_columns)),
            )

        return SqlValidationResult(valid=True, base_tables=base_tables, cte_names=sorted(cte_names), message="passed")


class NodeAgent:
    """Per-node agent facade that selects tools by node task kind."""

    TOOL_REGISTRY = {
        "inspect_schema": "inspect asset/live schema available for this node",
        "resolve_columns": "resolve required columns and output keys",
        "contract_critic": "check whether node plan contract is executable before SQL draft",
        "check_freshness": "check pt freshness/fallback risk",
        "choose_sql_strategy": "choose plan-bound LLM SQL or structured fallback",
        "draft_structured_sql": "draft safe one-table structured SQL",
        "draft_llm_sql": "draft one-table SQL with LLM bound to node plan contract",
        "semantic_ls": "list semantic files when node needs exact field/rule detail",
        "semantic_read": "read semantic file detail on demand",
        "semantic_grep": "search semantic files on demand",
        "artifact_ls": "list current run artifacts on demand",
        "artifact_read": "read current run artifact on demand",
        "artifact_grep": "search current run artifacts on demand",
        "validate_sql": "validate SQL with sqlglot and node scope",
        "execute_sql": "execute SQL in Doris",
        "repair_sql": "repair SQL only, never QueryGraph",
        "summarize_node_result": "summarize rows, entity set, and gaps",
    }

    def __init__(self, worker: "NodeWorkerExecutor"):
        self.worker = worker

    def execute(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        return self.worker._execute_node_with_tools(intent, asset_pack, knowledge_context, context)

    def tool_chain_for_intent(self, intent: QuestionIntent, context: NodeExecutionContext) -> NodeAgentContext:
        task_kind = self._task_kind(intent)
        tools = ["inspect_schema", "resolve_columns", "contract_critic", "check_freshness", "choose_sql_strategy"]
        if intent.sql_strategy == "structured_first":
            tools.append("draft_structured_sql")
            if self.worker.llm.configured:
                tools.append("draft_llm_sql")
        else:
            if self.worker.llm.configured:
                tools.append("draft_llm_sql")
            tools.append("draft_structured_sql")
        if self.worker.llm.configured:
            tools.extend(["artifact_ls", "artifact_read", "artifact_grep"])
            if getattr(self.worker, "semantic_catalog", None) is not None:
                tools.extend(["semantic_ls", "semantic_read", "semantic_grep"])
        tools.extend(["validate_sql", "execute_sql", "repair_sql", "summarize_node_result"])
        if intent.task_role == TaskRole.DEPENDENT and not context.upstream_entity_sets:
            task_kind = "DEPENDENT_LOOKUP_WITHOUT_CONTEXT"
        return NodeAgentContext(task_id=intent.plan_task_id, task_kind=task_kind, selected_tools=tools)

    def task_profile_for_intent(self, intent: QuestionIntent, context: NodeExecutionContext) -> NodeTaskProfile:
        agent_context = self.tool_chain_for_intent(intent, context)
        risk_controls = ["single_table_scope", "schema_validation", "plan_contract_bound_sql"]
        if intent.sql_strategy == "structured_first":
            risk_controls.append("structured_sql_first")
        else:
            risk_controls.append("llm_plan_bound_first")
        if intent.task_role == TaskRole.DEPENDENT:
            risk_controls.append("upstream_entity_filter")
        if intent.group_by_column == "pt":
            risk_controls.append("pt_partition_grouping")
        return NodeTaskProfile(
            task_id=intent.plan_task_id,
            task_kind=agent_context.task_kind,
            sql_strategy=intent.sql_strategy or "llm_plan_bound_first",
            selected_tools=agent_context.selected_tools,
            reason="%s node uses %s" % (agent_context.task_kind, intent.sql_strategy or "llm_plan_bound_first"),
            risk_controls=risk_controls,
        )

    def _task_kind(self, intent: QuestionIntent) -> str:
        if intent.task_role == TaskRole.DEPENDENT:
            return "DEPENDENT_LOOKUP"
        if intent.answer_mode == AnswerMode.TOPN:
            return "TOPN"
        if intent.group_by_column == "pt":
            return "TREND"
        if intent.answer_mode == AnswerMode.GROUP_AGG:
            return "GROUP_AGG"
        if intent.answer_mode == AnswerMode.DETAIL:
            return "DETAIL"
        return str(intent.answer_mode or "QUERY")


class NodePlanCritic:
    """Lightweight execution gate for a single node plan contract."""

    def review(self, contract: NodePlanContract) -> NodePlanCritiqueResult:
        issues: List[Dict[str, Any]] = []
        allowed = set(contract.allowed_columns)
        visible = set(contract.visible_columns)
        required = set(contract.required_columns)
        if not contract.preferred_table:
            issues.append(issue("PLAN_CONTRACT_MISMATCH", "node contract has no preferred table"))
        if not allowed:
            issues.append(issue("PLAN_CONTRACT_MISMATCH", "node contract has no allowed columns"))
        if self._metric_required(contract):
            metric_columns = formula_columns(contract.metric_formula, allowed)
            if contract.metric_column and contract.metric_column not in allowed:
                issues.append(issue("MISSING_METRIC_COLUMN", "metricColumn is not available in node schema", contract.metric_column))
            elif contract.metric_formula and not metric_columns:
                issues.append(issue("MISSING_METRIC_COLUMN", "metricFormula has no resolvable source columns", contract.metric_formula))
            elif not contract.metric_column and not contract.metric_formula:
                issues.append(issue("MISSING_METRIC_COLUMN", "metric node has no metricColumn or metricFormula"))
        if self._group_required(contract) and not contract.group_by_column and not contract.output_keys:
            issues.append(issue("MISSING_GROUP_BY_COLUMN", "aggregate node has no groupByColumn or outputKeys"))
        if contract.group_by_column and contract.group_by_column not in allowed:
            issues.append(issue("MISSING_GROUP_BY_COLUMN", "groupByColumn is not available in node schema", contract.group_by_column))
        missing_output = [column for column in contract.output_keys if column and column not in allowed]
        if missing_output:
            issues.append(issue("MISSING_OUTPUT_KEY", "outputKeys are not available in node schema", ",".join(missing_output)))
        denied_output = [column for column in contract.output_keys if column and column in allowed and column not in visible]
        if denied_output:
            issues.append(issue("PERMISSION_DENIED_OUTPUT_COLUMN", "outputKeys contain columns blocked by semantic access policy", ",".join(denied_output)))
        denied_group = bool(contract.group_by_column and contract.group_by_column in allowed and contract.group_by_column not in visible)
        if denied_group:
            issues.append(issue("PERMISSION_DENIED_GROUP_BY_COLUMN", "groupByColumn is blocked by semantic access policy", contract.group_by_column))
        if contract.task_role == TaskRole.DEPENDENT.value and not any(
            item.get("values") or item.get("columnValues") or item.get("column_values") for item in contract.upstream_entity_sets
        ):
            issues.append(issue("MISSING_UPSTREAM_ENTITY", "dependent node has no upstream entity set"))
        evidence_gaps = [
            column
            for column in contract.required_evidence
            if column
            and column not in allowed
            and column not in required
            and column != contract.metric_column
            and column != contract.metric_name
            and column not in metric_resolution_aliases(contract.metric_resolution)
        ]
        if evidence_gaps:
            issues.append(
                issue(
                    "CONTRACT_REQUIRED_EVIDENCE_GAP",
                    "requiredEvidence contains columns not available in node contract",
                    ",".join(evidence_gaps),
                )
            )
        denied_evidence = [
            column
            for column in contract.required_evidence
            if column
            and column in allowed
            and column not in visible
            and column not in {contract.metric_column, contract.merchant_filter_column}
        ]
        if denied_evidence:
            issues.append(
                issue(
                    "PERMISSION_DENIED_REQUIRED_EVIDENCE",
                    "requiredEvidence contains columns blocked by semantic access policy",
                    ",".join(denied_evidence),
                )
            )
        if not issues:
            return NodePlanCritiqueResult(task_id=contract.task_id, valid=True, message="contract passed")
        primary = issues[0]
        return NodePlanCritiqueResult(
            task_id=contract.task_id,
            valid=False,
            code=str(primary.get("code") or "PLAN_CONTRACT_MISMATCH"),
            message=str(primary.get("reason") or "node plan contract mismatch"),
            issues=issues,
            graph_repairable=True,
        )

    def _metric_required(self, contract: NodePlanContract) -> bool:
        return bool(
            contract.metric_column
            or contract.metric_formula
            or contract.metric_name
            or contract.answer_mode == AnswerMode.METRIC.value
        )

    def _group_required(self, contract: NodePlanContract) -> bool:
        return contract.answer_mode in {AnswerMode.TOPN.value, AnswerMode.GROUP_AGG.value}


class NodeWorkerExecutor:
    def __init__(
        self,
        llm: LlmClient,
        doris_repository: DorisRepository,
        validator: SqlValidationService,
        settings: Settings,
        semantic_catalog: Any = None,
    ):
        self.llm = llm
        self.doris_repository = doris_repository
        self.validator = validator
        self.settings = settings
        self.semantic_catalog = semantic_catalog
        self.node_agent = NodeAgent(self)
        self.prompt_assembler = PromptAssembler()
        self._prompt_traces_by_task: Dict[str, Dict[str, Any]] = {}
        self.tool_runtime_policies = ToolRuntimePolicyRegistry(settings)
        self.tool_failure_registry = ToolFailureRegistry(
            repeat_threshold=settings.tool_failure_repeat_threshold,
            circuit_threshold=settings.tool_circuit_threshold,
            cooldown_seconds=settings.tool_circuit_cooldown_seconds,
        )
        self.tool_runtime_service = ToolRuntimeService(
            settings,
            policy_registry=self.tool_runtime_policies,
            failure_registry=self.tool_failure_registry,
            tool_registry=canonical_tool_registry(NodeAgent.TOOL_REGISTRY),
        )
        self.artifact_store = WorkspaceArtifactStore(settings)
        self.runtime_state_store = create_runtime_state_store(settings)
        self.node_plan_critic = NodePlanCritic()
        self.access_control = AccessControlService(settings)
        self._last_sql_draft_decisions: Dict[str, SqlDraftDecision] = {}
        self._last_node_file_tool_results: Dict[str, List[Dict[str, Any]]] = {}

    def with_artifact_root(self, root: str) -> None:
        self.artifact_store = self.artifact_store.with_root(root)

    def execute_plan(
        self,
        merchant_id: str,
        plan: QueryPlan,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        question: str,
        resume_task_results: Optional[List[AgentTaskResult]] = None,
        run_id: str = "",
    ) -> AgentRunResult:
        optimize_query_plan_for_execution(plan, asset_pack)
        result = AgentRunResult()
        execution_run_id = run_id or "inline_%s" % uuid.uuid4().hex[:16]
        executable = [intent for intent in plan.intents if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE]
        tasks_by_id = {intent.plan_task_id or "node_%s" % (index + 1): intent for index, intent in enumerate(executable)}
        completed: Dict[str, AgentTaskResult] = {}
        for prior in resume_task_results or []:
            if prior.task_id in tasks_by_id and prior.success and not prior.query_bundle.failed:
                completed[prior.task_id] = prior
        result.resumed_task_ids = list(completed.keys())
        pending = {task_id: intent for task_id, intent in tasks_by_id.items() if task_id not in completed}
        if completed:
            result.node_execution_batches.append(
                NodeExecutionBatch(
                    batch_id="resume_%03d" % (int(time.time() * 1000) % 1000),
                    ready_task_ids=list(completed.keys()),
                    completed_task_ids=list(completed.keys()),
                    resumed_task_ids=list(completed.keys()),
                    max_concurrency=0,
                    timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
                )
            )
        while pending:
            ready_ids = [
                task_id
                for task_id, intent in pending.items()
                if all(parent in completed for parent in intent.depends_on_task_ids)
                and (intent.task_role != TaskRole.DEPENDENT or intent.depends_on_task_ids)
            ]
            if not ready_ids:
                ready_ids = list(pending.keys())
            batch_results, batch_trace = self._execute_ready_batch(ready_ids, pending, completed, plan, merchant_id, asset_pack, question, knowledge_context, execution_run_id)
            result.node_execution_batches.append(batch_trace)
            for task_id, task_result in batch_results.items():
                completed[task_id] = task_result
                pending.pop(task_id, None)

        for index, (task_id, task_result) in enumerate(completed.items()):
            intent = tasks_by_id[task_id]
            result.tasks.append(
                AgentTask(
                    task_id=task_id,
                    plan_index=index,
                    sub_agent_type="NODE_WORKER",
                    instruction=intent.question,
                    depends_on=intent.depends_on_task_ids,
                    plan_dependencies=[dep for dep in plan.dependencies if dep.dependent_task_id == task_id],
                )
            )
            result.task_results.append(task_result)
            result.query_bundles.append(task_result.query_bundle)
            result.sql_repairs.extend(task_result.sql_repairs)
            result.node_tool_traces.extend(task_result.node_tool_traces)
            if task_result.node_task_profile.task_id:
                result.node_task_profiles.append(task_result.node_task_profile)
            result.freshness_reports.extend(task_result.freshness_reports)
            if task_result.node_plan_contract.task_id:
                result.node_plan_contracts.append(task_result.node_plan_contract)
                result.node_plan_critiques.append(task_result.node_plan_critique)
            if task_result.sql_draft_decision.task_id:
                result.sql_draft_decisions.append(task_result.sql_draft_decision)

        result.merged_query_bundle = merge_task_result_bundles(result.task_results)
        result.evidence_check = self._check_dependency_coverage(plan, result.task_results)
        result.evidence_gaps = contract_gaps_from_task_results(result.task_results) + [
            EvidenceGap(code="DEPENDENCY_GAP", task_id=gap, reason=gap) for gap in result.evidence_check.gaps
        ]
        result.degraded_reasons = collect_degraded_reasons(result.task_results)
        return result

    def _execute_ready_batch(
        self,
        ready_ids: List[str],
        pending: Dict[str, QuestionIntent],
        completed: Dict[str, AgentTaskResult],
        plan: QueryPlan,
        merchant_id: str,
        asset_pack: PlanningAssetPack,
        question: str,
        knowledge_context: str,
        run_id: str = "",
    ) -> tuple[Dict[str, AgentTaskResult], NodeExecutionBatch]:
        started = time.perf_counter()
        run_id = run_id or "inline_%s" % uuid.uuid4().hex[:16]
        configured_workers = max(1, int(self.settings.max_concurrent_sub_agents or 1))
        task_cap = max(1, int(getattr(self.settings, "max_sub_agent_tasks", configured_workers) or configured_workers))
        max_workers = max(1, min(configured_workers, task_cap, len(ready_ids)))
        results: Dict[str, AgentTaskResult] = {}
        batch = NodeExecutionBatch(
            batch_id="batch_%03d" % (len(ready_ids) + int(time.time() * 1000) % 1000),
            ready_task_ids=list(ready_ids),
            max_concurrency=max_workers,
            timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
        )
        if max_workers < min(configured_workers, len(ready_ids)):
            batch.runtime_events.append(
                {
                    "event": "node.concurrency_limited",
                    "requestedConcurrency": configured_workers,
                    "effectiveConcurrency": max_workers,
                    "maxSubAgentTasks": task_cap,
                    "readyTaskCount": len(ready_ids),
                }
            )
        for start in range(0, len(ready_ids), max_workers):
            chunk_ids = ready_ids[start : start + max_workers]
            chunk_futures = {}
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                for task_id in chunk_ids:
                    node_args = {"taskId": task_id, "table": pending[task_id].preferred_table}
                    blocked = self.tool_failure_registry.should_block("node_agent", node_args)
                    if blocked:
                        batch.blocked_task_ids.append(task_id)
                        batch.failed_task_ids.append(task_id)
                        results[task_id] = failed_result(task_id, pending[task_id], "node_agent blocked by circuit breaker: %s" % blocked.reason)
                        continue
                    self.runtime_state_store.enqueue_node_task(
                        NodeTaskState(
                            run_id=run_id,
                            task_id=task_id,
                            status="queued",
                            idempotency_key=node_task_idempotency_key(run_id, task_id, pending[task_id].preferred_table),
                            payload={
                                "preferredTable": pending[task_id].preferred_table,
                                "dependsOn": list(pending[task_id].depends_on_task_ids or []),
                                "answerMode": str(pending[task_id].answer_mode),
                            },
                        )
                    )
                    claimed = self.runtime_state_store.claim_node_task(
                        run_id,
                        task_id,
                        lease_owner="node_worker",
                        lease_seconds=max(1, self.settings.agent_node_timeout_seconds),
                    )
                    if not claimed:
                        batch.blocked_task_ids.append(task_id)
                        batch.failed_task_ids.append(task_id)
                        results[task_id] = failed_result(task_id, pending[task_id], "node task could not acquire execution lease")
                        continue
                    context = self._node_context(task_id, pending[task_id], completed, plan, merchant_id, question, asset_pack)
                    context = self._prepare_subagent_context(task_id, pending[task_id], context, asset_pack)
                    context.context_package["parentRunId"] = run_id
                    future = executor.submit(self.execute_node, pending[task_id], asset_pack, knowledge_context, context)
                    chunk_futures[future] = task_id
                    batch.submitted_task_ids.append(task_id)
                if not chunk_futures:
                    continue
                done = set()
                not_done = set(chunk_futures.keys())
                timeout_seconds = max(1, int(self.settings.agent_node_timeout_seconds or 1))
                poll_interval = max(0.1, float(getattr(self.settings, "agent_node_poll_interval_seconds", 5.0) or 5.0))
                deadline = time.perf_counter() + timeout_seconds
                next_heartbeat_at = time.perf_counter() + poll_interval
                while not_done and time.perf_counter() < deadline:
                    remaining = max(0.0, deadline - time.perf_counter())
                    wait_for = min(remaining, max(0.05, next_heartbeat_at - time.perf_counter()))
                    just_done, still_running = wait(not_done, timeout=wait_for, return_when=FIRST_COMPLETED)
                    done.update(just_done)
                    not_done = set(still_running)
                    now = time.perf_counter()
                    if not_done and now >= next_heartbeat_at:
                        batch.runtime_events.append(
                            {
                                "event": "node.heartbeat",
                                "runningTaskIds": [chunk_futures[future] for future in not_done],
                                "elapsedMs": int((now - started) * 1000),
                                "timeoutSeconds": timeout_seconds,
                                "pollIntervalSeconds": poll_interval,
                            }
                        )
                        next_heartbeat_at = now + poll_interval
                for future in done:
                    task_id = chunk_futures[future]
                    node_args = {"taskId": task_id, "table": pending[task_id].preferred_table}
                    try:
                        task_result = future.result(timeout=0)
                        if task_result.query_bundle.failed:
                            self.tool_failure_registry.record_failure("node_agent", node_args, "NODE_FAILED", task_result.query_bundle.error or task_result.summary)
                            batch.failed_task_ids.append(task_id)
                            self.runtime_state_store.complete_node_task(run_id, task_id, "failed", {"error": task_result.query_bundle.error or task_result.summary})
                        else:
                            self.tool_failure_registry.record_success("node_agent", node_args)
                            batch.completed_task_ids.append(task_id)
                            self.runtime_state_store.complete_node_task(run_id, task_id, "completed", {"rows": task_result.query_bundle.effective_row_count()})
                    except Exception as exc:
                        self.tool_failure_registry.record_failure("node_agent", node_args, "ERROR", str(exc))
                        task_result = failed_result(task_id, pending[task_id], "NodeWorker 执行异常: %s" % str(exc)[:200])
                        batch.failed_task_ids.append(task_id)
                        self.runtime_state_store.complete_node_task(run_id, task_id, "failed", {"error": str(exc)[:500]})
                    task_result.task_id = task_id
                    results[task_id] = task_result
                for future in not_done:
                    task_id = chunk_futures[future]
                    grace_seconds = max(0, int(getattr(self.settings, "agent_node_timeout_grace_seconds", 60) or 60))
                    batch.runtime_events.append(
                        {
                            "event": "node.timeout",
                            "taskId": task_id,
                            "timeoutSeconds": self.settings.agent_node_timeout_seconds,
                            "timeoutType": "node_timeout",
                            "hardStopGraceSeconds": grace_seconds,
                        }
                    )
                    self.tool_failure_registry.record_failure(
                        "node_agent",
                        {"taskId": task_id, "table": pending[task_id].preferred_table},
                        "TIMEOUT",
                        "node execution timed out",
                    )
                    batch.timed_out_task_ids.append(task_id)
                    batch.failed_task_ids.append(task_id)
                    results[task_id] = timed_out_result(task_id, pending[task_id], self.settings.agent_node_timeout_seconds)
                    self.runtime_state_store.complete_node_task(run_id, task_id, "timeout", {"timeoutSeconds": self.settings.agent_node_timeout_seconds})
                    future.cancel()
            finally:
                for future in chunk_futures:
                    if not future.done():
                        future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
        for task_id in ready_ids:
            if task_id not in results:
                self.tool_failure_registry.record_failure(
                    "node_agent",
                    {"taskId": task_id, "table": pending[task_id].preferred_table},
                    "TIMEOUT",
                    "node execution timed out",
                )
                results[task_id] = timed_out_result(task_id, pending[task_id], self.settings.agent_node_timeout_seconds)
                batch.timed_out_task_ids.append(task_id)
                batch.failed_task_ids.append(task_id)
        batch.completed_task_ids = list(dict.fromkeys(batch.completed_task_ids))
        batch.failed_task_ids = list(dict.fromkeys(batch.failed_task_ids))
        batch.timed_out_task_ids = list(dict.fromkeys(batch.timed_out_task_ids))
        batch.duration_ms = int((time.perf_counter() - started) * 1000)
        return results, batch

    def _prepare_subagent_context(
        self,
        task_id: str,
        intent: QuestionIntent,
        context: NodeExecutionContext,
        asset_pack: PlanningAssetPack,
    ) -> NodeExecutionContext:
        safe_task_id = sanitize_node_artifact_name(task_id or intent.plan_task_id or "node")
        run_id = "sub_%s_%s" % (safe_task_id, uuid.uuid4().hex[:10])
        workspace = Path(self.artifact_store.root) / "subagents" / safe_task_id / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        checkpoint_path = workspace / "checkpoint.json"
        context_package = {
            "subAgentRunId": run_id,
            "taskId": task_id,
            "taskRole": str(intent.task_role),
            "subagentEnabled": False,
            "answerMode": str(intent.answer_mode),
            "preferredTable": intent.preferred_table,
            "allowedColumns": asset_pack.known_columns(intent.preferred_table)[:64],
            "upstreamEntitySets": [item.model_dump(by_alias=True) for item in context.upstream_entity_sets[:4]],
            "workspacePath": str(workspace),
            "checkpointPath": str(checkpoint_path),
        }
        next_context = context.model_copy(
            update={
                "sub_agent_run_id": run_id,
                "workspace_path": str(workspace),
                "checkpoint_path": str(checkpoint_path),
                "context_package": context_package,
            }
        )
        self._write_subagent_checkpoint(next_context, intent, "started", {"allowedColumns": context_package["allowedColumns"]})
        return next_context

    def _node_context(
        self,
        task_id: str,
        intent: QuestionIntent,
        completed: Dict[str, AgentTaskResult],
        plan: QueryPlan,
        merchant_id: str,
        question: str,
        asset_pack: PlanningAssetPack,
    ) -> NodeExecutionContext:
        upstream_rows: List[Dict[str, Any]] = []
        entity_sets: List[EntitySet] = []
        dependent_columns = set(asset_pack.known_columns(intent.preferred_table))
        for dep in plan.dependencies:
            if dep.dependent_task_id != task_id:
                continue
            parent_result = completed.get(dep.anchor_task_id)
            if not parent_result:
                continue
            rows = self._task_rows_for_context(parent_result, include_artifacts=intent.answer_mode == AnswerMode.DERIVED)
            if intent.answer_mode == AnswerMode.DERIVED:
                upstream_rows.extend([dict(row, __source_task_id=dep.anchor_task_id) for row in rows])
            else:
                upstream_rows.extend(rows)
            key, dependent_key = choose_entity_transfer_key(dep, rows, parent_result, dependent_columns)
            column_values = multi_entity_transfer_values(dep, rows, parent_result, dependent_columns, self.settings.agent_max_entity_values)
            values = list(column_values.get(dependent_key, [])) if dependent_key else []
            if not values and key:
                for row in rows:
                    value = row.get(key)
                    if key in row and not blank_entity_value(value) and value not in values:
                        values.append(value)
            allowed_dependent_keys = dependency_allowed_dependent_entity_keys(dep, dependent_columns)
            if parent_result.entity_set and parent_result.entity_set.values:
                if parent_result.entity_set.join_key in dependent_columns:
                    if allowed_dependent_keys and parent_result.entity_set.join_key not in allowed_dependent_keys:
                        pass
                    elif dependent_key and parent_result.entity_set.join_key != dependent_key:
                        pass
                    else:
                        dependent_key = parent_result.entity_set.join_key
                        merged_values = list(values)
                        for value in parent_result.entity_set.values:
                            if not blank_entity_value(value) and value not in merged_values:
                                merged_values.append(value)
                        values = merged_values
                        column_values[dependent_key] = merged_values[: self.settings.agent_max_entity_values]
            truncated = len(values) > self.settings.agent_max_entity_values
            truncated = truncated or any(len(items) > self.settings.agent_max_entity_values for items in column_values.values())
            missing_reason = ""
            if not values and not any(column_values.values()):
                if rows and not key:
                    missing_reason = "JOIN_KEY_NOT_PRODUCED"
                elif rows:
                    missing_reason = "JOIN_KEY_VALUES_EMPTY"
                elif parent_result.query_bundle.failed:
                    missing_reason = "UPSTREAM_SQL_FAILED"
                else:
                    missing_reason = "UPSTREAM_ZERO_ROWS"
            entity_sets.append(
                EntitySet(
                    task_id=dep.anchor_task_id,
                    join_key=dependent_key,
                    values=[value for value in values if not blank_entity_value(value)][: self.settings.agent_max_entity_values],
                    column_values={
                        column: [value for value in items if not blank_entity_value(value)][: self.settings.agent_max_entity_values]
                        for column, items in column_values.items()
                        if any(not blank_entity_value(value) for value in items)
                    },
                    truncated=truncated,
                    source_row_count=len(rows),
                    source_key=key,
                    requested_join_key=dep.anchor_column or dep.join_key,
                    missing_reason=missing_reason,
                )
            )
        return NodeExecutionContext(
            merchant_id=merchant_id,
            access_role=DEFAULT_ACCESS_ROLE,
            question=question or intent.question,
            upstream_entity_sets=entity_sets,
            upstream_rows=upstream_rows if intent.answer_mode == AnswerMode.DERIVED else upstream_rows[: self.settings.tool_result_preview_rows],
        )

    def _task_rows_for_context(self, task_result: AgentTaskResult, include_artifacts: bool = False) -> List[Dict[str, Any]]:
        if not include_artifacts:
            return list(task_result.query_bundle.rows)
        for path in task_result.query_bundle.offloaded_files:
            if not str(path).endswith("_rows.json"):
                continue
            try:
                payload = json.loads(self.artifact_store.read(str(path), max_chars=20_000_000).get("content") or "[]")
            except Exception:
                continue
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
        return list(task_result.query_bundle.rows)

    def execute_node(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        if intent.answer_mode == AnswerMode.DERIVED:
            result = self._execute_derived_node(intent, asset_pack, context)
        else:
            result = self.node_agent.execute(intent, asset_pack, knowledge_context, context)
        return self._finalize_subagent_result(result, intent, context)

    def _execute_derived_node(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        tool_traces: List[NodeToolCall] = []
        profile = NodeTaskProfile(
            task_id=intent.plan_task_id,
            task_kind="DERIVED_METRIC",
            sql_strategy="derived_compute",
            selected_tools=["load_component_results", "compute_derived_metric", "summarize_node_result"],
            reason="compute semantic metric from upstream component metrics",
            risk_controls=["semantic_metric_contract", "upstream_component_evidence"],
            contract_status="passed",
            sql_draft_source="derived_compute",
        )
        record_tool(
            tool_traces,
            intent,
            "load_component_results",
            "success" if context.upstream_rows else "failed",
            "upstreamRows=%s" % len(context.upstream_rows),
            "componentMetrics=%s" % ",".join(str(item.get("metricKey") or "") for item in (intent.metric_resolution or {}).get("componentMetrics") or []),
        )
        rows, error = self._compute_derived_metric_rows(intent, context)
        if error:
            record_tool(tool_traces, intent, "compute_derived_metric", "failed", intent.metric_name, error, "DERIVED_METRIC_FAILED")
            return AgentTaskResult(
                task_id=intent.plan_task_id,
                success=False,
                summary=error,
                query_bundle=QueryBundle(tables=[], failed=True, error=error, summary=error),
                react_trace=[ReActStep(round=1, reason=error, action="compute_derived_metric.failed", observation=intent.metric_name)],
                node_tool_traces=tool_traces,
                node_task_profile=profile,
            )
        artifact_paths = self._write_node_artifacts(intent.plan_task_id, "-- derived semantic metric: %s" % intent.metric_name, rows, context.workspace_path)
        preview_rows = rows[: max(0, self.settings.context_artifact_inline_max_rows)]
        entity_set = entity_set_from_rows(intent.plan_task_id, intent, rows, self.settings.agent_max_entity_values)
        record_tool(
            tool_traces,
            intent,
            "compute_derived_metric",
            "success",
            intent.metric_formula,
            "rows=%s metric=%s" % (len(rows), intent.metric_name),
        )
        record_tool(
            tool_traces,
            intent,
            "summarize_node_result",
            "success",
            "rows=%s" % len(rows),
            "entityKey=%s values=%s artifacts=%s" % (entity_set.join_key, len(entity_set.values), len(artifact_paths)),
        )
        return AgentTaskResult(
            task_id=intent.plan_task_id,
            success=True,
            summary="计算派生指标返回 %s 行" % len(rows),
            query_bundle=QueryBundle(
                tables=[],
                rows=preview_rows,
                original_row_count=len(rows),
                summary="计算派生指标返回 %s 行" % len(rows),
                offloaded_files=artifact_paths,
            ),
            react_trace=[
                ReActStep(round=1, reason="读取组件指标结果", action="load_component_results", observation="rows=%s" % len(context.upstream_rows)),
                ReActStep(round=2, reason="按语义层公式计算派生指标", action="compute_derived_metric", observation="rows=%s" % len(rows)),
            ],
            entity_set=entity_set,
            node_tool_traces=tool_traces,
            node_task_profile=profile,
        )

    def _compute_derived_metric_rows(self, intent: QuestionIntent, context: NodeExecutionContext) -> Tuple[List[Dict[str, Any]], str]:
        resolution = intent.metric_resolution or {}
        if str(resolution.get("computeStrategy") or "") == "projection_group_aggregate":
            return self._compute_projection_group_aggregate_rows(intent, context)
        components = [item for item in resolution.get("componentMetrics") or [] if isinstance(item, dict)]
        if len(components) < 2:
            return [], "DERIVED_METRIC_COMPONENTS_MISSING"
        group_key = intent.group_by_column or str(resolution.get("groupByColumn") or "")
        if not group_key:
            return [], "DERIVED_METRIC_GROUP_KEY_MISSING"
        component_keys = [str(item.get("metricKey") or "") for item in components if item.get("metricKey")]
        grouped: Dict[Any, Dict[str, Any]] = {}
        for row in context.upstream_rows:
            group_value = row.get(group_key)
            if blank_entity_value(group_value):
                continue
            target = grouped.setdefault(group_value, {group_key: group_value})
            for metric_key in component_keys:
                value = row.get(metric_key)
                if value is None:
                    value = first_present_value(row, metric_alias_candidates(metric_key))
                number = numeric_value(value)
                if number is None:
                    continue
                target[metric_key] = float(target.get(metric_key) or 0) + number
        if not grouped:
            return [], "DERIVED_METRIC_NO_JOINED_COMPONENT_ROWS"
        metric_name = intent.metric_name or "derived_metric"
        unit = str(resolution.get("unit") or "")
        rows: List[Dict[str, Any]] = []
        numerator_key = component_keys[0]
        denominator_key = component_keys[1]
        for values in grouped.values():
            if any(key not in values for key in component_keys[:2]):
                continue
            numerator = numeric_value(values.get(numerator_key))
            denominator = numeric_value(values.get(denominator_key))
            if numerator is None or denominator in {None, 0}:
                continue
            derived = numerator / denominator
            if unit == "%":
                derived *= 100
            row = dict(values)
            row[metric_name] = round(derived, 6)
            rows.append(row)
        rows.sort(key=lambda item: numeric_value(item.get(metric_name)) or 0, reverse=True)
        limit = int(intent.limit or 0)
        if limit > 0:
            rows = rows[:limit]
        return rows, "" if rows else "DERIVED_METRIC_ZERO_ROWS"

    def _compute_projection_group_aggregate_rows(self, intent: QuestionIntent, context: NodeExecutionContext) -> Tuple[List[Dict[str, Any]], str]:
        resolution = intent.metric_resolution or {}
        metric_task_id = str(resolution.get("sourceMetricTaskId") or "")
        bridge_task_id = str(resolution.get("bridgeTaskId") or "")
        source_join_key = str(resolution.get("sourceJoinKey") or "")
        bridge_join_key = str(resolution.get("bridgeJoinKey") or source_join_key)
        group_key = intent.group_by_column or str(resolution.get("groupByColumn") or "")
        metric_name = intent.metric_name or str(resolution.get("metricKey") or "metric_value")
        aliases = [
            str(item)
            for item in resolution.get("sourceMetricAliases") or []
            if str(item or "")
        ] or metric_alias_candidates(metric_name)
        carry_columns = [
            str(item)
            for item in resolution.get("carryColumns") or []
            if str(item or "")
        ]
        if not metric_task_id or not bridge_task_id or not source_join_key or not bridge_join_key or not group_key:
            return [], "PROJECTION_AGGREGATE_CONTRACT_MISSING"
        metric_rows = [
            row for row in context.upstream_rows if str(row.get("__source_task_id") or "") == metric_task_id
        ]
        bridge_rows = [
            row for row in context.upstream_rows if str(row.get("__source_task_id") or "") == bridge_task_id
        ]
        if not metric_rows or not bridge_rows:
            return [], "PROJECTION_AGGREGATE_UPSTREAM_ROWS_MISSING"
        bridge_by_join: Dict[Any, List[Dict[str, Any]]] = {}
        for row in bridge_rows:
            join_value = row.get(bridge_join_key)
            if blank_entity_value(join_value):
                continue
            bridge_by_join.setdefault(join_value, []).append(row)
        grouped: Dict[Any, Dict[str, Any]] = {}
        for metric_row in metric_rows:
            join_value = metric_row.get(source_join_key)
            if blank_entity_value(join_value):
                continue
            metric_value = first_present_value(metric_row, aliases)
            number = numeric_value(metric_value)
            if number is None:
                continue
            for bridge_row in bridge_by_join.get(join_value, []):
                group_value = bridge_row.get(group_key)
                if blank_entity_value(group_value):
                    continue
                target = grouped.setdefault(group_value, {group_key: group_value})
                target[metric_name] = float(target.get(metric_name) or 0) + number
                for column in carry_columns:
                    if column == group_key:
                        continue
                    value = bridge_row.get(column)
                    if not blank_entity_value(value) and column not in target:
                        target[column] = value
        if not grouped:
            return [], "PROJECTION_AGGREGATE_ZERO_ROWS"
        rows = list(grouped.values())
        rows.sort(key=lambda item: numeric_value(item.get(metric_name)) or 0, reverse=True)
        limit = int(intent.limit or 0)
        if limit > 0:
            rows = rows[:limit]
        return rows, "" if rows else "PROJECTION_AGGREGATE_ZERO_ROWS"

    def _write_node_artifacts(self, task_id: str, sql: str, rows: List[Dict[str, Any]], workspace_path: str = "") -> List[str]:
        safe_task_id = task_id or "node"
        store = self.artifact_store.with_root(workspace_path) if workspace_path else self.artifact_store
        sql_artifact = store.write_text("sql", "%s.sql" % safe_task_id, sql, preview_chars=0)
        rows_artifact = store.write_json("sql_results", "%s_rows.json" % safe_task_id, rows, preview_chars=0)
        return [path for path in [sql_artifact.get("path"), rows_artifact.get("path")] if path]

    def _finalize_subagent_result(
        self,
        result: AgentTaskResult,
        intent: QuestionIntent,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        if not context.sub_agent_run_id:
            return result
        result.sub_agent_run_id = context.sub_agent_run_id
        result.sub_agent_checkpoint_path = context.checkpoint_path
        result.sub_agent_workspace = context.workspace_path
        result.sub_agent_context = dict(context.context_package or {})
        result.file_tool_results = list(self._last_node_file_tool_results.pop(intent.plan_task_id, []) or [])
        if result.file_tool_results:
            result.react_trace.insert(
                0,
                ReActStep(
                    round=0,
                    reason="NodeWorker 按需读取文件上下文",
                    action="file_context_tools",
                    observation="toolResults=%d" % len(result.file_tool_results),
                ),
            )
        self._write_subagent_checkpoint(
            context,
            intent,
            "success" if result.success and not result.query_bundle.failed else "failed",
            {
                "summary": result.summary,
                "rows": result.query_bundle.effective_row_count(),
                "failed": result.query_bundle.failed,
                "error": result.query_bundle.error,
                "fileToolResults": result.file_tool_results[:8],
            },
        )
        return result

    def _write_subagent_checkpoint(
        self,
        context: NodeExecutionContext,
        intent: QuestionIntent,
        status: str,
        payload: Dict[str, Any],
    ) -> None:
        if not context.checkpoint_path:
            return
        try:
            path = Path(context.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "subAgentRunId": context.sub_agent_run_id,
                "taskId": intent.plan_task_id,
                "status": status,
                "contextPackage": context.context_package,
                "payload": payload,
                "updatedAtMs": int(time.time() * 1000),
            }
            path.write_text(json.dumps(checkpoint, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
            run_id = str(context.context_package.get("parentRunId") or context.sub_agent_run_id or "inline")
            self.runtime_state_store.upsert_node_task(
                NodeTaskState(
                    run_id=run_id,
                    task_id=intent.plan_task_id,
                    status=status,
                    idempotency_key=node_task_idempotency_key(run_id, intent.plan_task_id, intent.preferred_table),
                    attempts=1,
                    lease_owner=context.sub_agent_run_id,
                    payload={
                        "preferredTable": intent.preferred_table,
                        "answerMode": str(intent.answer_mode),
                        "summary": payload.get("summary") or "",
                        "rows": payload.get("rows") or 0,
                        "failed": bool(payload.get("failed")),
                        "checkpointPath": str(path),
                    },
                )
            )
        except Exception:
            return

    def _execute_node_with_tools(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        tool_traces: List[NodeToolCall] = []
        agent_context = self.node_agent.tool_chain_for_intent(intent, context)
        node_task_profile = self.node_agent.task_profile_for_intent(intent, context)
        record_tool(
            tool_traces,
            intent,
            "choose_sql_strategy",
            "success",
            agent_context.task_kind,
            "%s via %s" % (intent.sql_strategy or "llm_first", ",".join(agent_context.selected_tools[:6])),
        )
        self._record_schema_tools(tool_traces, intent, asset_pack)
        contract = self._node_plan_contract(intent, asset_pack, context)
        critique_started = time.perf_counter()
        critique = self.node_plan_critic.review(contract)
        critique_duration_ms = int((time.perf_counter() - critique_started) * 1000)
        node_task_profile.contract_status = "passed" if critique.valid else "failed"
        node_task_profile.contract_critique_reason = critique.message or critique.code
        record_tool(
            tool_traces,
            intent,
            "contract_critic",
            "success" if critique.valid else "failed",
            contract.preferred_table,
            critique.message or "contract passed",
            critique.code,
            duration_ms=critique_duration_ms,
        )
        trace: List[ReActStep] = [
            ReActStep(round=1, reason="根据 QueryGraph node 生成 SQL", action="sql_draft.plan", observation=intent.preferred_table)
        ]
        if not critique.valid:
            freshness = FreshnessCheckResult(
                task_id=intent.plan_task_id,
                table=intent.preferred_table,
                checked=False,
                status="SKIPPED",
                requested_days=int(intent.days or 0),
                reason="node plan contract failed; skip freshness and SQL execution",
            )
            message = "%s：%s" % (critique.code or "PLAN_CONTRACT_MISMATCH", critique.message)
            trace.append(ReActStep(round=1, reason=message, action="contract_critic.failed", observation=intent.preferred_table))
            record_tool(tool_traces, intent, "check_freshness", "SKIPPED", intent.preferred_table, freshness.reason)
            record_tool(tool_traces, intent, "summarize_node_result", "failed", "node plan contract", message, critique.code)
            return AgentTaskResult(
                success=False,
                summary=message,
                query_bundle=QueryBundle(tables=[intent.preferred_table] if intent.preferred_table else [], failed=True, error=message, summary=message),
                react_trace=trace,
                node_tool_traces=tool_traces,
                node_task_profile=node_task_profile,
                freshness_reports=[freshness],
                node_plan_contract=contract,
                node_plan_critique=critique,
            )
        freshness_started = time.perf_counter()
        freshness = self._check_freshness(intent, asset_pack, context)
        freshness_duration_ms = int((time.perf_counter() - freshness_started) * 1000)
        record_tool(
            tool_traces,
            intent,
            "check_freshness",
            freshness.status or "skipped",
            intent.preferred_table,
            freshness.reason or freshness.max_pt or freshness.status,
            freshness.status if freshness.status not in {"AVAILABLE", "SKIPPED", "NO_PT_COLUMN"} else "",
            duration_ms=freshness_duration_ms,
        )
        fallback_intent = self._maybe_realtime_fallback_intent(intent, asset_pack, freshness)
        if fallback_intent:
            freshness.status = "STALE_USE_REALTIME_FALLBACK"
            freshness.fallback_table = fallback_intent.preferred_table
            freshness.reason = "%s; switch_to_realtime=%s" % (freshness.reason or "offline table stale", fallback_intent.preferred_table)
            record_tool(
                tool_traces,
                intent,
                "select_realtime_fallback",
                "success",
                intent.preferred_table,
                "fallbackTable=%s maxPt=%s" % (fallback_intent.preferred_table, freshness.max_pt),
                "STALE_USE_REALTIME_FALLBACK",
            )
            trace.append(
                ReActStep(
                    round=2,
                    reason="离线表近实时分区滞后，切换到实时 fallback 表",
                    action="select_realtime_fallback",
                    observation="%s -> %s" % (intent.preferred_table, fallback_intent.preferred_table),
                )
            )
            intent = fallback_intent
            self._record_schema_tools(tool_traces, intent, asset_pack)
            contract = self._node_plan_contract(intent, asset_pack, context)
            critique_started = time.perf_counter()
            critique = self.node_plan_critic.review(contract)
            critique_duration_ms = int((time.perf_counter() - critique_started) * 1000)
            node_task_profile.contract_status = "passed" if critique.valid else "failed"
            node_task_profile.contract_critique_reason = critique.message or critique.code
            record_tool(
                tool_traces,
                intent,
                "contract_critic",
                "success" if critique.valid else "failed",
                contract.preferred_table,
                critique.message or "fallback contract passed",
                critique.code,
                duration_ms=critique_duration_ms,
            )
            if not critique.valid:
                message = "%s：%s" % (critique.code or "REALTIME_FALLBACK_CONTRACT_MISMATCH", critique.message)
                record_tool(tool_traces, intent, "summarize_node_result", "failed", "realtime fallback contract", message, critique.code)
                return AgentTaskResult(
                    success=False,
                    summary=message,
                    query_bundle=QueryBundle(tables=[intent.preferred_table] if intent.preferred_table else [], failed=True, error=message, summary=message),
                    react_trace=trace,
                    node_tool_traces=tool_traces,
                    node_task_profile=node_task_profile,
                    freshness_reports=[freshness],
                    node_plan_contract=contract,
                    node_plan_critique=critique,
                )
        draft_tool = self._draft_tool_name(intent)
        draft_started = time.perf_counter()
        sql = self._draft_sql(intent, asset_pack, knowledge_context, context, contract)
        draft_duration_ms = int((time.perf_counter() - draft_started) * 1000)
        draft_decision = self._last_sql_draft_decisions.pop(intent.plan_task_id, SqlDraftDecision(task_id=intent.plan_task_id))
        file_tool_results = self._last_node_file_tool_results.get(intent.plan_task_id, [])
        if file_tool_results:
            record_tool(
                tool_traces,
                intent,
                "node_file_context_tools",
                "success",
                "ContextPackage=%s" % context.sub_agent_run_id,
                "rounds=%d" % len(file_tool_results),
            )
        node_task_profile.sql_draft_source = draft_decision.source
        if draft_decision.source == "structured_fast_path":
            draft_tool = "draft_structured_sql_fast_path"
        draft_summary = trim_sql(sql)
        draft_prompt = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "draft"), None)
        if draft_prompt:
            draft_summary = append_prompt_marker(draft_summary, draft_prompt)
        draft_tool_schema = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "draft_tool"), None)
        if draft_tool_schema:
            draft_summary = append_tool_schema_marker(draft_summary, draft_tool_schema)
        record_tool(
            tool_traces,
            intent,
            draft_tool,
            "success" if sql else "failed",
            intent.preferred_table,
            draft_summary,
            "SQL_EMPTY" if not sql else "",
            duration_ms=draft_duration_ms,
        )
        if draft_decision.structured_fallback_used:
            record_tool(
                tool_traces,
                intent,
                "draft_structured_sql_fallback",
                "success" if sql else "failed",
                draft_decision.fallback_reason,
                trim_sql(sql),
                "SQL_EMPTY" if not sql else "",
            )
        validation_results: List[SqlValidationResult] = []
        repair_attempts: List[SqlRepairAttempt] = []
        for round_index in range(self.settings.agent_sql_repair_rounds + 1):
            validation_started = time.perf_counter()
            validation = self.validator.validate(sql, asset_pack)
            validation = self._node_scope_validation(validation, intent, sql, asset_pack)
            validation = self._contract_scope_validation(validation, intent, sql, contract)
            validation_duration_ms = int((time.perf_counter() - validation_started) * 1000)
            validation_results.append(validation)
            record_tool(
                tool_traces,
                intent,
                "validate_sql",
                "success" if validation.valid else "failed",
                trim_sql(sql),
                validation.message,
                validation.error_code,
                round_index,
                duration_ms=validation_duration_ms,
            )
            trace.append(
                ReActStep(
                    round=2 + round_index * 3,
                    reason="执行前安全校验",
                    action="validate_sql",
                    observation="passed" if validation.valid else "%s: %s" % (validation.error_code, validation.message),
                )
            )
            if not validation.valid:
                structured_attempt = self._structured_fallback_attempt(sql, validation, intent, asset_pack, context)
                if structured_attempt and round_index < self.settings.agent_sql_repair_rounds:
                    repair_attempts.append(structured_attempt)
                    record_tool(
                        tool_traces,
                        intent,
                        "draft_structured_sql_fallback",
                        "success",
                        validation.error_code,
                        trim_sql(structured_attempt.repaired_sql),
                        validation.error_code,
                        round_index + 1,
                    )
                    trace.append(
                        ReActStep(
                            round=3 + round_index * 3,
                            reason="改用结构化一表 SQL 收敛查询",
                            action="draft_structured_sql_fallback",
                            observation=structured_attempt.error_code,
                        )
                    )
                    sql = structured_attempt.repaired_sql
                    continue
                if validation.error_code in STRICT_STRUCTURED_FALLBACK_CODES:
                    return AgentTaskResult(
                        success=False,
                        summary=validation.message,
                        query_bundle=QueryBundle(sql=sql, tables=validation.base_tables, failed=True, error=validation.message),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                if round_index >= self.settings.agent_sql_repair_rounds:
                    return AgentTaskResult(
                        success=False,
                        summary=validation.message,
                        query_bundle=QueryBundle(sql=sql, tables=validation.base_tables, failed=True, error=validation.message),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                repair_started = time.perf_counter()
                repaired = self._repair_sql(sql, validation, intent, asset_pack, context)
                repair_duration_ms = int((time.perf_counter() - repair_started) * 1000)
                repair_attempts.append(repaired)
                repair_summary = trim_sql(repaired.repaired_sql)
                repair_prompt = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "repair"), None)
                if repair_prompt:
                    repair_summary = append_prompt_marker(repair_summary, repair_prompt)
                repair_tool_schema = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "repair_tool"), None)
                if repair_tool_schema:
                    repair_summary = append_tool_schema_marker(repair_summary, repair_tool_schema)
                record_tool(
                    tool_traces,
                    intent,
                    "repair_sql",
                    "success" if repaired.repaired_sql else "failed",
                    validation.error_code,
                    repair_summary,
                    validation.error_code,
                    round_index + 1,
                    duration_ms=repair_duration_ms,
                )
                if not repaired.repaired_sql:
                    return AgentTaskResult(
                        success=False,
                        summary=validation.message,
                        query_bundle=QueryBundle(sql=sql, tables=validation.base_tables, failed=True, error=validation.message),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                sql = repaired.repaired_sql
                continue
            bound_sql, sql_params, binding_error = bind_node_sql_parameters(sql, intent, asset_pack, context)
            tenant_binding_error = tenant_scope_binding_error(bound_sql, sql_params, contract, context)
            if tenant_binding_error:
                binding_error = binding_error or tenant_binding_error
            if binding_error:
                record_tool(
                    tool_traces,
                    intent,
                    "bind_sql_params",
                    "failed",
                    trim_sql(sql),
                    binding_error,
                    "SQL_PARAM_BINDING_FAILED",
                    round_index,
                )
                structured_attempt = self._structured_fallback_attempt(
                    sql,
                    validation.model_copy(update={"valid": False, "error_code": "SQL_PARAM_BINDING_FAILED", "message": binding_error}),
                    intent,
                    asset_pack,
                    context,
                )
                if structured_attempt and round_index < self.settings.agent_sql_repair_rounds:
                    repair_attempts.append(structured_attempt)
                    record_tool(
                        tool_traces,
                        intent,
                        "draft_structured_sql_fallback",
                        "success",
                        "SQL_PARAM_BINDING_FAILED",
                        trim_sql(structured_attempt.repaired_sql),
                        "SQL_PARAM_BINDING_FAILED",
                        round_index + 1,
                    )
                    sql = structured_attempt.repaired_sql
                    continue
                return AgentTaskResult(
                    success=False,
                    summary=binding_error,
                    query_bundle=QueryBundle(sql=sql, tables=validation.base_tables, failed=True, error=binding_error),
                    react_trace=trace,
                    sql_repairs=repair_attempts,
                    validation_results=validation_results,
                    node_tool_traces=tool_traces,
                    node_task_profile=node_task_profile,
                    freshness_reports=[freshness],
                    node_plan_contract=contract,
                    node_plan_critique=critique,
                    sql_draft_decision=draft_decision,
                )
            anchored_sql, anchor_date = self._apply_partition_date_anchor(bound_sql, intent, freshness)
            if anchor_date:
                bound_sql = anchored_sql
                record_tool(
                    tool_traces,
                    intent,
                    "anchor_partition_date",
                    "success",
                    intent.preferred_table,
                    "CURDATE anchored to max_pt=%s" % anchor_date,
                    "PARTITION_DATE_ANCHOR",
                    round_index,
                )
            record_tool(
                tool_traces,
                intent,
                "bind_sql_params",
                "success",
                trim_sql(bound_sql),
                "params=%d merchantBound=%s" % (len(sql_params), sql_has_bound_merchant_filter(bound_sql, asset_pack.known_columns(intent.preferred_table))),
                "",
                round_index,
            )
            access_decision = self.access_control.authorize_contract(contract, bound_sql, run_id=context.sub_agent_run_id)
            record_tool(
                tool_traces,
                intent,
                "access_control",
                "success" if access_decision.allowed else "failed",
                intent.preferred_table,
                access_decision.message or "ACL passed columns=%d masks=%d" % (len(access_decision.checked_columns), len(access_decision.masked_columns)),
                access_decision.code,
                round_index,
            )
            if not access_decision.allowed:
                message = "%s：%s" % (access_decision.code or "ACCESS_DENIED", access_decision.message or "query access denied")
                trace.append(ReActStep(round=3 + round_index * 3, reason="权限校验失败", action="access_control.failed", observation=message))
                return AgentTaskResult(
                    success=False,
                    summary=message,
                    query_bundle=QueryBundle(sql=bound_sql, tables=[intent.preferred_table] if intent.preferred_table else [], failed=True, error=message, summary=message),
                    react_trace=trace,
                    sql_repairs=repair_attempts,
                    validation_results=validation_results,
                    node_tool_traces=tool_traces,
                    node_task_profile=node_task_profile,
                    freshness_reports=[freshness],
                    node_plan_contract=contract,
                    node_plan_critique=critique,
                    sql_draft_decision=draft_decision,
                )
            contract.masked_columns = dict(access_decision.masked_columns or {})
            execute_args = {
                "taskId": intent.plan_task_id,
                "table": intent.preferred_table,
                "sql": trim_sql(bound_sql, 1000),
                "paramCount": len(sql_params),
            }
            query_started = time.perf_counter()
            runtime_result = self.tool_runtime_service.execute(
                "execute_sql",
                execute_args,
                lambda _args: {
                    "rows": self.doris_repository.query(bound_sql, sql_params),
                    "cacheHit": bool(getattr(self.doris_repository, "last_cache_hit", False)),
                    "cacheKey": str(getattr(self.doris_repository, "last_cache_key", "") or ""),
                },
                call_id=intent.plan_task_id or "execute_sql",
                target_kind="doris",
            )
            query_duration_ms = runtime_result.duration_ms or int((time.perf_counter() - query_started) * 1000)
            if runtime_result.status != "success":
                self.access_control.record_query_audit(access_decision, status=runtime_result.status or "failed")
                message = runtime_result.error_message or "Doris 查询失败"
                error_type = runtime_result.error_type or "SQL_EXECUTION_FAILED"
                if error_type == "TIMEOUT" and "超时" not in message:
                    message = "Doris 查询超时: %s" % message
                record_tool(tool_traces, intent, "execute_sql", runtime_result.status or "failed", trim_sql(bound_sql), message, error_type, round_index, duration_ms=query_duration_ms)
                trace.append(ReActStep(round=3 + round_index * 3, reason="Doris 执行失败", action="query_doris.failed", observation="%s: %s" % (error_type, message[:200])))
                split_result = self._split_detail_query_fallback(
                    bound_sql,
                    sql_params,
                    error_type,
                    message,
                    intent,
                    asset_pack,
                    context,
                    validation.base_tables,
                    contract,
                    query_duration_ms,
                )
                if split_result is not None:
                    rows = list(split_result["rows"])
                    cache_hit = bool(split_result.get("cacheHit"))
                    cache_key = str(split_result.get("cacheKey") or "")
                    split_duration_ms = int(split_result.get("durationMs") or query_duration_ms)
                    split_events = list(split_result.get("runtimeEvents") or [])
                    display_rows = apply_column_masks(rows, contract)
                    self.access_control.record_query_audit(access_decision, row_count=len(rows), status="success_split_fallback")
                    artifact_paths = self._write_node_artifacts(intent.plan_task_id, bound_sql, display_rows, context.workspace_path)
                    preview_rows = display_rows[: max(0, self.settings.context_artifact_inline_max_rows)]
                    entity_set = entity_set_from_rows(intent.plan_task_id, intent, rows, self.settings.agent_max_entity_values)
                    record_tool(
                        tool_traces,
                        intent,
                        "execute_sql_split_fallback",
                        "success",
                        trim_sql(bound_sql),
                        "rows=%s chunks=%s sourceError=%s cacheHit=%s" % (len(rows), len(split_events), error_type, cache_hit),
                        error_type,
                        round_index,
                        duration_ms=split_duration_ms,
                    )
                    trace.append(ReActStep(round=4 + round_index * 3, reason="Doris 超时/资源错误后按时间窗口拆分明细查询", action="query_doris.split_fallback", observation="rows=%s chunks=%s" % (len(rows), len(split_events))))
                    return AgentTaskResult(
                        success=True,
                        summary="Doris 原查询失败后拆分查询返回 %s 行" % len(rows),
                        query_bundle=QueryBundle(
                            sql=bound_sql,
                            params=sql_params,
                            tables=validation.base_tables,
                            rows=preview_rows,
                            original_row_count=len(rows),
                            summary="Doris 原查询失败后拆分查询返回 %s 行" % len(rows),
                            offloaded_files=artifact_paths,
                            duration_ms=split_duration_ms,
                            cache_hit=cache_hit,
                            cache_key=cache_key,
                            runtime_events=split_events,
                        ),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        entity_set=entity_set,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                if error_type not in {"CIRCUIT_OPEN", "RATE_LIMITED", "TIMEOUT"}:
                    policy = doris_error_policy(error_type)
                    structured_attempt = self._structured_fallback_attempt(
                        sql,
                        validation.model_copy(update={"valid": False, "error_code": error_type, "message": message}),
                        intent,
                        asset_pack,
                        context,
                    )
                    if policy["structured_fallback"] and structured_attempt and round_index < self.settings.agent_sql_repair_rounds:
                        repair_attempts.append(structured_attempt)
                        record_tool(
                            tool_traces,
                            intent,
                            "draft_structured_sql_fallback",
                            "success",
                            error_type,
                            trim_sql(structured_attempt.repaired_sql),
                            error_type,
                            round_index + 1,
                        )
                        sql = structured_attempt.repaired_sql
                        continue
                    resource_attempt = self._resource_safe_fallback_attempt(sql, error_type, message, intent, asset_pack, context)
                    if policy["resource_fallback"] and resource_attempt and round_index < self.settings.agent_sql_repair_rounds:
                        repair_attempts.append(resource_attempt)
                        record_tool(
                            tool_traces,
                            intent,
                            "draft_resource_safe_sql_fallback",
                            "success",
                            error_type,
                            trim_sql(resource_attempt.repaired_sql),
                            error_type,
                            round_index + 1,
                        )
                        sql = resource_attempt.repaired_sql
                        continue
                return AgentTaskResult(
                    success=False,
                    summary="%s: %s" % (error_type, message[:200]),
                    query_bundle=QueryBundle(sql=bound_sql, params=sql_params, tables=validation.base_tables, failed=True, error=message),
                    react_trace=trace,
                    sql_repairs=repair_attempts,
                    validation_results=validation_results,
                    node_tool_traces=tool_traces,
                    node_task_profile=node_task_profile,
                    freshness_reports=[freshness],
                    node_plan_contract=contract,
                    node_plan_critique=critique,
                    sql_draft_decision=draft_decision,
                )
            try:
                rows = list(runtime_result.result.get("rows") or [])
                cache_hit = bool(runtime_result.result.get("cacheHit") or getattr(self.doris_repository, "last_cache_hit", False))
                cache_key = str(runtime_result.result.get("cacheKey") or getattr(self.doris_repository, "last_cache_key", "") or "")
                record_tool(
                    tool_traces,
                    intent,
                    "execute_sql",
                    "success",
                    trim_sql(bound_sql),
                    "rows=%s durationMs=%s cacheHit=%s" % (len(rows), query_duration_ms, cache_hit),
                    "",
                    round_index,
                    duration_ms=query_duration_ms,
                )
                trace.append(ReActStep(round=3 + round_index * 3, reason="读取 Doris", action="query_doris", observation="rows=%s" % len(rows)))
                entity_set = entity_set_from_rows(intent.plan_task_id, intent, rows, self.settings.agent_max_entity_values)
                display_rows = apply_column_masks(rows, contract)
                self.access_control.record_query_audit(access_decision, row_count=len(rows), status="success")
                artifact_paths = self._write_node_artifacts(intent.plan_task_id, bound_sql, display_rows, context.workspace_path)
                preview_rows = display_rows[: max(0, self.settings.context_artifact_inline_max_rows)]
                record_tool(
                    tool_traces,
                    intent,
                    "summarize_node_result",
                    "success",
                    "rows=%s" % len(rows),
                    "entityKey=%s values=%s artifacts=%s" % (entity_set.join_key, len(entity_set.values), len(artifact_paths)),
                )
                return AgentTaskResult(
                    success=True,
                    summary="返回 %s 行" % len(rows),
                    query_bundle=QueryBundle(
                        sql=bound_sql,
                        params=sql_params,
                        tables=validation.base_tables,
                        rows=preview_rows,
                        original_row_count=len(rows),
                        summary="返回 %s 行" % len(rows),
                        offloaded_files=artifact_paths,
                        duration_ms=query_duration_ms,
                        cache_hit=cache_hit,
                        cache_key=cache_key,
                    ),
                    react_trace=trace,
                    sql_repairs=repair_attempts,
                    validation_results=validation_results,
                    entity_set=entity_set,
                    node_tool_traces=tool_traces,
                    node_task_profile=node_task_profile,
                    freshness_reports=[freshness],
                    node_plan_contract=contract,
                    node_plan_critique=critique,
                    sql_draft_decision=draft_decision,
                )
            except Exception as exc:
                query_duration_ms = int((time.perf_counter() - query_started) * 1000) if "query_started" in locals() else 0
                error_text = str(exc)
                doris_error_code = classify_doris_error(error_text)
                policy = doris_error_policy(doris_error_code)
                self.tool_failure_registry.record_failure("execute_sql", execute_args, doris_error_code, error_text)
                self.access_control.record_query_audit(access_decision, status="failed_exception")
                record_tool(tool_traces, intent, "execute_sql", "failed", trim_sql(bound_sql), error_text[:240], doris_error_code, round_index)
                trace.append(ReActStep(round=3 + round_index * 3, reason="读取 Doris 失败", action="query_doris", observation=error_text[:240]))
                split_result = self._split_detail_query_fallback(
                    bound_sql,
                    sql_params,
                    doris_error_code,
                    error_text,
                    intent,
                    asset_pack,
                    context,
                    validation.base_tables,
                    contract,
                    query_duration_ms,
                )
                if split_result is not None:
                    rows = list(split_result["rows"])
                    cache_hit = bool(split_result.get("cacheHit"))
                    cache_key = str(split_result.get("cacheKey") or "")
                    split_duration_ms = int(split_result.get("durationMs") or query_duration_ms)
                    split_events = list(split_result.get("runtimeEvents") or [])
                    display_rows = apply_column_masks(rows, contract)
                    self.access_control.record_query_audit(access_decision, row_count=len(rows), status="success_split_fallback")
                    artifact_paths = self._write_node_artifacts(intent.plan_task_id, bound_sql, display_rows, context.workspace_path)
                    preview_rows = display_rows[: max(0, self.settings.context_artifact_inline_max_rows)]
                    entity_set = entity_set_from_rows(intent.plan_task_id, intent, rows, self.settings.agent_max_entity_values)
                    record_tool(
                        tool_traces,
                        intent,
                        "execute_sql_split_fallback",
                        "success",
                        trim_sql(bound_sql),
                        "rows=%s chunks=%s sourceError=%s cacheHit=%s" % (len(rows), len(split_events), doris_error_code, cache_hit),
                        doris_error_code,
                        round_index,
                        duration_ms=split_duration_ms,
                    )
                    trace.append(ReActStep(round=4 + round_index * 3, reason="Doris 超时/资源错误后按时间窗口拆分明细查询", action="query_doris.split_fallback", observation="rows=%s chunks=%s" % (len(rows), len(split_events))))
                    return AgentTaskResult(
                        success=True,
                        summary="Doris 原查询失败后拆分查询返回 %s 行" % len(rows),
                        query_bundle=QueryBundle(
                            sql=bound_sql,
                            params=sql_params,
                            tables=validation.base_tables,
                            rows=preview_rows,
                            original_row_count=len(rows),
                            summary="Doris 原查询失败后拆分查询返回 %s 行" % len(rows),
                            offloaded_files=artifact_paths,
                            duration_ms=split_duration_ms,
                            cache_hit=cache_hit,
                            cache_key=cache_key,
                            runtime_events=split_events,
                        ),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        entity_set=entity_set,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                structured_attempt = self._structured_fallback_attempt(
                    sql,
                    validation.model_copy(update={"valid": False, "error_code": doris_error_code, "message": error_text}),
                    intent,
                    asset_pack,
                    context,
                )
                if policy["structured_fallback"] and structured_attempt and round_index < self.settings.agent_sql_repair_rounds:
                    repair_attempts.append(structured_attempt)
                    record_tool(
                        tool_traces,
                        intent,
                        "draft_structured_sql_fallback",
                        "success",
                        doris_error_code,
                        trim_sql(structured_attempt.repaired_sql),
                        doris_error_code,
                        round_index + 1,
                    )
                    trace.append(
                        ReActStep(
                            round=4 + round_index * 3,
                            reason="Doris 失败后改用结构化 SQL 收敛字段和过滤",
                            action="draft_structured_sql_fallback",
                            observation=error_text[:160],
                        )
                    )
                    sql = structured_attempt.repaired_sql
                    continue
                resource_attempt = self._resource_safe_fallback_attempt(sql, doris_error_code, error_text, intent, asset_pack, context)
                if policy["resource_fallback"] and resource_attempt and round_index < self.settings.agent_sql_repair_rounds:
                    repair_attempts.append(resource_attempt)
                    record_tool(
                        tool_traces,
                        intent,
                        "draft_resource_safe_sql_fallback",
                        "success",
                        doris_error_code,
                        trim_sql(resource_attempt.repaired_sql),
                        doris_error_code,
                        round_index + 1,
                    )
                    trace.append(
                        ReActStep(
                            round=4 + round_index * 3,
                            reason="Doris 资源错误后改用资源保护 SQL",
                            action="draft_resource_safe_sql_fallback",
                            observation=error_text[:160],
                        )
                    )
                    sql = resource_attempt.repaired_sql
                    continue
                if round_index >= self.settings.agent_sql_repair_rounds or not policy["llm_repair"]:
                    return AgentTaskResult(
                        success=False,
                        summary="Doris 查询失败: %s" % error_text[:200],
                        query_bundle=QueryBundle(sql=bound_sql, params=sql_params, tables=validation.base_tables, failed=True, error=error_text, duration_ms=query_duration_ms),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                repair_started = time.perf_counter()
                repaired = self._repair_sql(sql, validation.model_copy(update={"valid": False, "error_code": doris_error_code, "message": error_text}), intent, asset_pack, context)
                repair_duration_ms = int((time.perf_counter() - repair_started) * 1000)
                repair_attempts.append(repaired)
                repair_summary = trim_sql(repaired.repaired_sql)
                repair_prompt = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "repair"), None)
                if repair_prompt:
                    repair_summary = append_prompt_marker(repair_summary, repair_prompt)
                repair_tool_schema = self._prompt_traces_by_task.pop(prompt_trace_key(intent, "repair_tool"), None)
                if repair_tool_schema:
                    repair_summary = append_tool_schema_marker(repair_summary, repair_tool_schema)
                record_tool(
                    tool_traces,
                    intent,
                    "repair_sql",
                    "success" if repaired.repaired_sql else "failed",
                    doris_error_code,
                    repair_summary,
                    doris_error_code,
                    round_index + 1,
                    duration_ms=repair_duration_ms,
                )
                if not repaired.repaired_sql:
                    return AgentTaskResult(
                        success=False,
                        summary="Doris 查询失败: %s" % error_text[:200],
                        query_bundle=QueryBundle(sql=bound_sql, params=sql_params, tables=validation.base_tables, failed=True, error=error_text, duration_ms=query_duration_ms),
                        react_trace=trace,
                        sql_repairs=repair_attempts,
                        validation_results=validation_results,
                        node_tool_traces=tool_traces,
                        node_task_profile=node_task_profile,
                        freshness_reports=[freshness],
                        node_plan_contract=contract,
                        node_plan_critique=critique,
                        sql_draft_decision=draft_decision,
                    )
                sql = repaired.repaired_sql
        result = failed_result(intent.plan_task_id, intent, "NodeWorker 未能完成执行")
        result.node_tool_traces = tool_traces
        result.node_task_profile = node_task_profile
        result.freshness_reports = [freshness]
        result.node_plan_contract = contract
        result.node_plan_critique = critique
        result.sql_draft_decision = draft_decision
        return result

    def _draft_sql(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
        contract: Optional[NodePlanContract] = None,
    ) -> str:
        contract = contract or self._node_plan_contract(intent, asset_pack, context)
        decision = SqlDraftDecision(task_id=intent.plan_task_id)
        if intent.sql:
            decision.source = "provided_sql"
            decision.reason = "intent already contains SQL"
            self._last_sql_draft_decisions[intent.plan_task_id] = decision
            return intent.sql.strip()
        if intent.sql_strategy == "structured_first":
            structured_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract)
            decision.source = "structured_first"
            decision.reason = "intent.sql_strategy=structured_first"
            self._last_sql_draft_decisions[intent.plan_task_id] = decision
            if structured_sql:
                return structured_sql
        if self._use_structured_fast_path(intent, contract, context):
            structured_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract)
            if structured_sql:
                decision.source = "structured_fast_path"
                decision.reason = "low-risk node contract can be compiled deterministically"
                self._last_sql_draft_decisions[intent.plan_task_id] = decision
                return structured_sql
        sql = ""
        if self.llm.configured:
            decision.llm_attempted = True
            draft_args = {
                "taskId": intent.plan_task_id,
                "table": intent.preferred_table,
                "answerMode": str(intent.answer_mode),
                "taskRole": str(intent.task_role),
                "strategy": intent.sql_strategy or "llm_plan_bound_first",
            }
            blocked = self.tool_failure_registry.should_block("draft_llm_sql", draft_args)
            if blocked:
                decision.structured_fallback_used = True
                decision.source = "structured_fallback"
                decision.fallback_reason = "draft_llm_sql blocked by circuit breaker: %s" % blocked.reason
                structured_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract)
                self._last_sql_draft_decisions[intent.plan_task_id] = decision
                return structured_sql
            tool = sql_draft_tool()
            prompt = self.prompt_assembler.render(
                "node.sql_draft",
                sections={
                    "node_context_policy": (
                    "NodeAgent 只接收当前 nodePlanContract；必须基于 contract 写 SQL。"
                    "不能改表、不能猜字段、不能 join、不能修改 QueryGraph。"
                ),
            },
        )
            file_context, file_results = self._node_file_tool_loop(intent, context, contract, knowledge_context)
            if file_results:
                self._last_node_file_tool_results[intent.plan_task_id] = file_results
            user_payload = self._node_llm_payload(intent, context, contract, file_context)
            try:
                if hasattr(self.llm, "tool_json_chat"):
                    payload = self.llm.tool_json_chat(prompt.system_prompt, user_payload, tool.openai_schema(), {})
                else:
                    payload = self.llm.json_chat(prompt.system_prompt, user_payload, {})
            except Exception as exc:
                self.tool_failure_registry.record_failure("draft_llm_sql", draft_args, "PROVIDER_ERROR", str(exc))
                payload = {}
            self._prompt_traces_by_task[prompt_trace_key(intent, "draft")] = prompt.trace()
            self._prompt_traces_by_task[prompt_trace_key(intent, "draft_tool")] = tool.trace_schema()
            sql = str(payload.get("sql") or "").strip()
            if sql:
                self.tool_failure_registry.record_success("draft_llm_sql", draft_args)
                decision.source = "llm_plan_bound"
                decision.reason = str(payload.get("reason") or "LLM drafted SQL from node plan contract")
            else:
                error_type = "PROVIDER_ERROR" if self.llm.last_error else "SQL_EMPTY"
                self.tool_failure_registry.record_failure("draft_llm_sql", draft_args, error_type, self.llm.last_error or "LLM returned empty SQL")
        if sql:
            self._last_sql_draft_decisions[intent.plan_task_id] = decision
            return sql
        decision.structured_fallback_used = True
        decision.source = "structured_fallback"
        decision.fallback_reason = self.llm.last_error or "LLM unavailable or returned empty SQL"
        decision.reason = "fallback to deterministic single-table SQL builder"
        self._last_sql_draft_decisions[intent.plan_task_id] = decision
        return self._draft_structured_sql(intent, asset_pack, context, contract=contract)

    def _use_structured_fast_path(
        self,
        intent: QuestionIntent,
        contract: NodePlanContract,
        context: NodeExecutionContext,
    ) -> bool:
        if intent.sql or intent.sql_strategy in {"structured_first", "llm_first_debug"}:
            return False
        if intent.answer_mode == AnswerMode.DETAIL:
            return bool(intent.output_keys or intent.required_evidence or intent.filter_column)
        if (
            intent.answer_mode == AnswerMode.METRIC
            and intent.task_role != TaskRole.DEPENDENT
            and not context.upstream_entity_sets
            and (intent.metric_formula or intent.metric_column or intent.metric_specs)
        ):
            return True
        if (
            intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG}
            and intent.task_role != TaskRole.DEPENDENT
            and not context.upstream_entity_sets
            and intent.group_by_column
            and (intent.metric_formula or intent.metric_column or intent.metric_specs)
        ):
            return True
        if intent.task_role == TaskRole.DEPENDENT and context.upstream_entity_sets:
            return intent.answer_mode in {AnswerMode.GROUP_AGG, AnswerMode.METRIC, AnswerMode.TOPN}
        resolution_source = str((intent.metric_resolution or {}).get("resolutionSource") or "")
        if "schema_reconciled" in resolution_source:
            return True
        return False

    def _node_file_tool_loop(
        self,
        intent: QuestionIntent,
        context: NodeExecutionContext,
        contract: NodePlanContract,
        knowledge_context: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        max_rounds = int(getattr(self.settings, "agent_node_file_tool_rounds", 0) or 0)
        if max_rounds <= 0 or not self.llm.configured or not hasattr(self.llm, "tool_chat"):
            return "", []
        tools = artifact_file_tool_definitions()
        if self.semantic_catalog is not None:
            tools = semantic_file_tool_definitions() + tools
        tool_schemas = [tool.openai_schema() for tool in tools]
        handlers = self._file_tool_handlers(context.workspace_path)
        results: List[Dict[str, Any]] = []
        calls_trace: List[Dict[str, Any]] = []
        for round_index in range(max_rounds):
            payload = {
                "question": context.question or intent.question,
                "nodePlanContract": contract.model_dump(by_alias=True),
                "subAgentContextPackage": context.context_package,
                "knowledgePreview": knowledge_context[:4000],
                "previousToolResults": compact_tool_results_for_prompt(results),
                "instruction": (
                    "如果 nodePlanContract/knowledgePreview 已足够写 SQL，不要调用工具。"
                    "如果缺字段、公式、关系说明或需要查看 artifact，再调用 semantic_read/artifact_read/grep。"
                    "只能读取当前任务相关文件，最多调用少量工具。"
                ),
            }
            llm_result = self.llm.tool_chat(
                "你是 NodeWorker 的文件上下文选择器。按需读取文件，目标是补齐当前单节点 SQL 所需上下文。",
                json.dumps(payload, ensure_ascii=False, default=str),
                tool_schemas,
                {"content": "", "toolCalls": []},
                timeout_seconds=min(8, int(getattr(self.settings, "llm_request_timeout_seconds", 20) or 20)),
            )
            calls = [
                ToolCallRequest(id=str(call.get("id") or "node_file_%d_%d" % (round_index, idx)), name=str(call.get("name") or ""), args=call.get("args") or {})
                for idx, call in enumerate(llm_result.get("toolCalls") or [])
                if str(call.get("name") or "") in handlers
            ][:4]
            if not calls:
                break
            cache_policies = {
                "semantic_ls": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "semantic_read": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "semantic_grep": ToolCachePolicy(enabled=True, namespace="semantic_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_ls": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_read": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
                "artifact_grep": ToolCachePolicy(enabled=True, namespace="artifact_tool", ttl_seconds=self.settings.semantic_cache_ttl_seconds),
            }
            executed = self.tool_runtime_service.execute_many(calls, handlers, cache_policies=cache_policies)
            serialized = [serialize_tool_execution_result(item) for item in executed]
            calls_trace.extend([call.model_dump(by_alias=True) for call in calls])
            results.extend(serialized)
        if not results:
            return "", []
        text = "## NodeWorker File Tool Results\n%s" % json.dumps(compact_tool_results_for_prompt(results), ensure_ascii=False, default=str)
        return text[-6000:], [{"calls": calls_trace, "results": results}]

    def _file_tool_handlers(self, subagent_workspace: str = "") -> Dict[str, Any]:
        artifact_store = self.artifact_store
        handlers: Dict[str, Any] = {
            "artifact_ls": lambda args: {"items": artifact_store.ls(namespace=str(args.get("namespace") or ""), limit=int(args.get("limit") or 50))},
            "artifact_read": lambda args: artifact_store.read(
                path=str(args.get("path") or ""),
                offset=int(args.get("offset") or 0),
                max_chars=min(int(args.get("maxChars") or self.settings.context_file_inline_max_chars), self.settings.context_file_inline_max_chars),
            ),
            "artifact_grep": lambda args: {"hits": artifact_store.grep(query=str(args.get("query") or ""), limit=int(args.get("limit") or 20))},
        }
        if self.semantic_catalog is not None:
            handlers.update(
                {
                    "semantic_ls": lambda args: {
                        "items": self.semantic_catalog.ls(
                            topic=str(args.get("topic") or ""),
                            query=str(args.get("query") or ""),
                            limit=int(args.get("limit") or 20),
                        )
                    },
                    "semantic_read": lambda args: self.semantic_catalog.read(
                        ref_id=str(args.get("refId") or ""),
                        path=str(args.get("path") or ""),
                        max_chars=min(int(args.get("maxChars") or self.settings.context_file_inline_max_chars), self.settings.context_file_inline_max_chars),
                        offset=int(args.get("offset") or 0),
                    ),
                    "semantic_grep": lambda args: {
                        "hits": self.semantic_catalog.grep(
                            query=str(args.get("query") or ""),
                            topic=str(args.get("topic") or ""),
                            limit=int(args.get("limit") or 20),
                        )
                    },
                }
            )
        return handlers

    def _node_llm_payload(
        self,
        intent: QuestionIntent,
        context: NodeExecutionContext,
        contract: NodePlanContract,
        file_context: str = "",
    ) -> str:
        return json.dumps(
            {
                "nodePlanContract": contract.model_dump(by_alias=True),
                "selectMustInclude": contract_select_required_columns(contract),
                "upstreamPreviewRows": context.upstream_rows[:10],
                "subAgentContextPackage": context.context_package,
                "fileToolResults": file_context[:6000],
                "availableToolSchemas": node_runtime_tool_schemas(NodeAgent.TOOL_REGISTRY, self.node_agent.tool_chain_for_intent(intent, context).selected_tools),
                "contextScope": [
                    "nodePlanContract",
                    "upstream preview rows",
                    "file tool results selected by this NodeWorker",
                ],
                "rules": (
                    "只生成 SELECT/WITH 查询；禁止 DDL/DML；只能查询 nodePlanContract.preferredTable；不要 join 其他表；"
                    "只能使用 nodePlanContract.allowedColumns 里的真实字段；不要使用 contract 外字段或表。"
                    "SELECT/输出字段只能来自 nodePlanContract.visibleColumns；internalOnlyColumns 只能用于商家/权限过滤或内部公式，不得直接输出。"
                    "如果字段在 nodePlanContract.maskedColumns 里，不要试图规避或还原脱敏策略。"
                    "必须使用 nodePlanContract.merchantFilterColumn 做商家过滤；"
                    "如果 nodePlanContract.metricSpecs 不为空，SELECT 必须输出每个 metricSpec 的 metricName；公式只能使用 metricSpec.sourceColumns/metricFormula。"
                    "如果问题或 upstreamEntitySets 提供了 sub_order_id/spu_id/refund_id/ticket_id/bill_id，必须用这些分桶/主键过滤；"
                    "selectMustInclude 是强制 SELECT 输出列，必须逐个原样出现在 SELECT 中；"
                    "QueryGraph outputKeys 是传给 dependent 的实体键，必须原样出现在 SELECT 结果中，不能只放在 WHERE/GROUP BY；"
                    "GROUP_AGG/TOPN 必须按 outputKeys 和 groupByColumn 分组并输出，不能丢失 coupon_id/spu_id/spu_name/sub_order_id/order_id/ticket_id/bill_id 等实体键；"
                    "如果表有 pt 且不是快照维表豁免，必须使用 nodePlanContract.days 生成时间窗过滤。"
                    "pt 是 Doris DATE 分区列，时间窗必须写成 `pt` >= DATE_SUB(CURDATE(), INTERVAL N DAY)，不要使用 DATE_FORMAT('%Y%m%d')。"
                    "没有倒排索引时不要假设索引存在；只选择 requiredColumns 中需要字段和过滤字段；明细 LIMIT <= 20。"
                    "不能修改 contract 里的指标、粒度、依赖或证据要求。"
                ),
            },
            ensure_ascii=False,
            default=str,
        )

    def _node_local_context(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> Dict[str, Any]:
        table = intent.preferred_table
        return {
            "taskId": intent.plan_task_id,
            "taskRole": intent.task_role,
            "answerMode": intent.answer_mode,
            "table": table,
            "requiredColumns": self._node_required_columns(intent, asset_pack).get(table, []),
            "outputKeys": intent.output_keys,
            "entityFilters": [
                {
                    "taskId": entity.task_id,
                    "joinKey": entity.join_key,
                    "columns": sorted(entity.column_values.keys()),
                    "truncated": entity.truncated,
                    "missingReason": entity.missing_reason,
                }
                for entity in context.upstream_entity_sets
            ],
        }

    def _record_schema_tools(self, traces: List[NodeToolCall], intent: QuestionIntent, asset_pack: PlanningAssetPack) -> None:
        columns = asset_pack.known_columns(intent.preferred_table)
        record_tool(
            traces,
            intent,
            "inspect_schema",
            "success" if columns else "failed",
            intent.preferred_table,
            "columns=%s schemaSource=%s" % (len(columns), asset_pack.schema_source.get(intent.preferred_table, "asset")),
            "" if columns else "SCHEMA_MISSING",
        )
        requested = self._node_required_columns(intent, asset_pack).get(intent.preferred_table, [])
        record_tool(
            traces,
            intent,
            "resolve_columns",
            "success" if requested else "warning",
            ",".join(intent.required_evidence[:12]),
            ",".join(requested[:16]),
            "" if requested else "NO_REQUIRED_COLUMNS",
        )

    def _check_freshness(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> FreshnessCheckResult:
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        if not table:
            return FreshnessCheckResult(task_id=intent.plan_task_id, table=table, status="SKIPPED", reason="no preferred table")
        partition_anchor_enabled = bool(getattr(self.settings, "agent_partition_date_anchor_enabled", False))
        if int(intent.days or 0) > 2 and not partition_anchor_enabled:
            return FreshnessCheckResult(
                task_id=intent.plan_task_id,
                table=table,
                checked=False,
                status="SKIPPED",
                requested_days=int(intent.days or 0),
                reason="freshness check is only required for near-real-time windows",
            )
        if "pt" not in columns:
            return FreshnessCheckResult(
                task_id=intent.plan_task_id,
                table=table,
                checked=False,
                status="NO_PT_COLUMN",
                requested_days=int(intent.days or 0),
                reason="table has no pt partition column in asset pack",
            )
        where = ""
        params: List[Any] = []
        if "seller_id" in columns:
            where = " WHERE `seller_id` = %s"
            params.append(context.merchant_id)
        elif "merchant_id" in columns:
            where = " WHERE `merchant_id` = %s"
            params.append(context.merchant_id)
        sql = "SELECT MIN(`pt`) AS `min_pt`, MAX(`pt`) AS `max_pt` FROM `%s`%s" % (table, where)
        try:
            rows = self.doris_repository.query(sql, params)
        except Exception as exc:
            return FreshnessCheckResult(
                task_id=intent.plan_task_id,
                table=table,
                checked=True,
                status="CHECK_FAILED",
                requested_days=int(intent.days or 0),
                reason=str(exc)[:200],
            )
        first = rows[0] if rows else {}
        min_pt = str(first.get("min_pt") or first.get("MIN(`pt`)") or "")
        max_pt = str(first.get("max_pt") or first.get("MAX(`pt`)") or "")
        status = "AVAILABLE" if max_pt else "ZERO_ROWS"
        reason = "max_pt=%s" % max_pt if max_pt else "freshness check returned no partition value"
        return FreshnessCheckResult(
            task_id=intent.plan_task_id,
            table=table,
            checked=True,
            status=status,
            requested_days=int(intent.days or 0),
            min_pt=min_pt,
            max_pt=max_pt,
            reason=reason,
        )

    def _apply_partition_date_anchor(self, sql: str, intent: QuestionIntent, freshness: FreshnessCheckResult) -> tuple[str, str]:
        if not bool(getattr(self.settings, "agent_partition_date_anchor_enabled", False)):
            return sql, ""
        if not freshness.checked or not freshness.max_pt or "CURDATE()" not in str(sql or "").upper():
            return sql, ""
        anchor = parse_partition_date(freshness.max_pt)
        if not anchor:
            return sql, ""
        anchor_text = anchor.isoformat()
        anchored_sql = re.sub(r"\bCURDATE\(\)", "'%s'" % anchor_text, str(sql or ""), flags=re.I)
        if anchored_sql == sql:
            return sql, ""
        freshness.reason = append_note(freshness.reason, "relative time anchored to max_pt=%s" % anchor_text)
        return anchored_sql, anchor_text

    def _maybe_realtime_fallback_intent(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        freshness: FreshnessCheckResult,
    ) -> Optional[QuestionIntent]:
        if not freshness.checked or not freshness.max_pt or not intent.preferred_table:
            return None
        fallback = realtime_fallback_for_table(asset_pack, intent.preferred_table)
        if not fallback:
            return None
        fallback_table = fallback.table or str(fallback.metadata.get("realtimeTable") or fallback.metadata.get("fallbackTable") or "")
        if not fallback_table or fallback_table == intent.preferred_table:
            return None
        if fallback_table not in set(asset_pack.known_tables()):
            return None
        if not partition_is_stale_for_near_realtime(freshness.max_pt, int(intent.days or 0)):
            return None
        updated_resolution = dict(intent.metric_resolution or {})
        if updated_resolution.get("ownerTable") == intent.preferred_table:
            updated_resolution["ownerTable"] = fallback_table
        return intent.model_copy(
            update={
                "preferred_table": fallback_table,
                "metric_resolution": updated_resolution,
                "analysis_note": append_note(
                    intent.analysis_note,
                    "离线表 %s 分区 max_pt=%s，已切换实时 fallback 表 %s"
                    % (intent.preferred_table, freshness.max_pt, fallback_table),
                ),
            }
        )

    def _draft_tool_name(self, intent: QuestionIntent) -> str:
        if intent.sql:
            return "draft_structured_sql"
        if intent.sql_strategy == "structured_first":
            return "draft_structured_sql"
        if self.llm.configured:
            return "draft_llm_sql"
        return "draft_structured_sql"

    def _structured_fallback_attempt(
        self,
        sql: str,
        validation: SqlValidationResult,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> SqlRepairAttempt | None:
        structured_sql = self._draft_structured_sql(intent, asset_pack, context)
        if not structured_sql or equivalent_sql(sql, structured_sql):
            return None
        error_code = validation.error_code or "STRUCTURED_FALLBACK"
        if error_code not in STRUCTURED_FALLBACK_ERROR_CODES:
            return None
        return SqlRepairAttempt(
            task_id=intent.plan_task_id,
            round=1,
            original_sql=sql,
            repaired_sql=structured_sql,
            error_code=error_code,
            error_message=validation.message,
            success=True,
        )

    def _resource_safe_fallback_attempt(
        self,
        sql: str,
        error_code: str,
        error_text: str,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> SqlRepairAttempt | None:
        if error_code not in RESOURCE_CONSTRAINED_DORIS_ERRORS:
            return None
        safe_sql = self._draft_structured_sql(intent, asset_pack, context, resource_safe=True)
        if not safe_sql or equivalent_sql(sql, safe_sql):
            return None
        return SqlRepairAttempt(
            task_id=intent.plan_task_id,
            round=1,
            original_sql=sql,
            repaired_sql=safe_sql,
            error_code=error_code,
            error_message="Doris resource error; switched to resource-safe SQL: %s" % error_text[:180],
            success=True,
        )

    def _split_detail_query_fallback(
        self,
        bound_sql: str,
        sql_params: List[Any],
        error_code: str,
        error_text: str,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
        tables: List[str],
        contract: NodePlanContract,
        previous_duration_ms: int,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self.settings, "agent_doris_split_query_enabled", True)):
            return None
        if error_code not in RESOURCE_CONSTRAINED_DORIS_ERRORS:
            return None
        if intent.answer_mode != AnswerMode.DETAIL:
            return None
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        if "pt" not in columns or int(intent.days or 0) <= 0:
            return None
        safe_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract, resource_safe=True)
        if not safe_sql:
            return None
        safe_bound_sql, safe_params, binding_error = bind_node_sql_parameters(safe_sql, intent, asset_pack, context)
        if binding_error:
            return None
        chunk_days = max(1, int(getattr(self.settings, "agent_doris_split_chunk_days", 7) or 7))
        max_chunks = max(1, int(getattr(self.settings, "agent_doris_split_max_chunks", 6) or 6))
        max_concurrency = max(1, int(getattr(self.settings, "agent_doris_split_max_concurrency", 3) or 3))
        limit = structured_limit(intent.limit, detail=True, resource_safe=True)
        split_sqls = split_detail_sql_by_pt_windows(safe_bound_sql, int(intent.days or 0), chunk_days, max_chunks, limit)
        if not split_sqls:
            return None
        events: List[Dict[str, Any]] = [
            {
                "event": "split_query_fallback_started",
                "sourceErrorCode": error_code,
                "sourceError": str(error_text or "")[:240],
                "originalDurationMs": previous_duration_ms,
                "chunkDays": chunk_days,
                "maxChunks": max_chunks,
                "chunkCount": len(split_sqls),
                "maxConcurrency": min(max_concurrency, len(split_sqls)),
                "executionMode": "parallel_chunks",
                "limit": limit,
            }
        ]
        cache_hit = False
        cache_keys: List[str] = []
        started = time.perf_counter()
        chunk_results: Dict[int, List[Dict[str, Any]]] = {}

        def query_chunk(index: int, chunk_sql: str) -> Dict[str, Any]:
            chunk_started = time.perf_counter()
            chunk_rows = self.doris_repository.query(chunk_sql, safe_params)
            return {
                "chunkIndex": index,
                "rows": list(chunk_rows or []),
                "cacheHit": bool(getattr(self.doris_repository, "last_cache_hit", False)),
                "cacheKey": str(getattr(self.doris_repository, "last_cache_key", "") or ""),
                "durationMs": int((time.perf_counter() - chunk_started) * 1000),
                "sql": trim_sql(chunk_sql, 600),
            }

        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(split_sqls))) as executor:
            futures = {
                executor.submit(query_chunk, index, chunk_sql): (index, chunk_sql)
                for index, chunk_sql in enumerate(split_sqls, start=1)
            }
            for future in as_completed(futures):
                index, chunk_sql = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    events.append(
                        {
                            "event": "split_query_chunk_failed",
                            "chunkIndex": index,
                            "errorCode": classify_doris_error(str(exc)),
                            "error": str(exc)[:240],
                            "sql": trim_sql(chunk_sql, 600),
                        }
                    )
                    continue
                chunk_rows = payload["rows"]
                chunk_results[index] = chunk_rows
                cache_hit = cache_hit or bool(payload["cacheHit"])
                if payload["cacheKey"]:
                    cache_keys.append(payload["cacheKey"])
                events.append(
                    {
                        "event": "split_query_chunk_succeeded",
                        "chunkIndex": index,
                        "rows": len(chunk_rows),
                        "cacheHit": bool(payload["cacheHit"]),
                        "durationMs": payload["durationMs"],
                        "sql": payload["sql"],
                    }
                )
        rows: List[Dict[str, Any]] = []
        for index in sorted(chunk_results):
            if len(rows) >= limit:
                break
            remaining = max(0, limit - len(rows))
            rows.extend(chunk_results[index][:remaining])
        if not rows:
            return None
        events.append(
            {
                "event": "split_query_fallback_finished",
                "rows": len(rows),
                "chunksAttempted": len([item for item in events if str(item.get("event")) in {"split_query_chunk_succeeded", "split_query_chunk_failed"}]),
                "chunksSucceeded": len([item for item in events if str(item.get("event")) == "split_query_chunk_succeeded"]),
                "chunksFailed": len([item for item in events if str(item.get("event")) == "split_query_chunk_failed"]),
                "executionMode": "parallel_chunks",
                "tables": tables,
            }
        )
        return {
            "rows": rows[:limit],
            "cacheHit": cache_hit,
            "cacheKey": ",".join(cache_keys[:3]),
            "durationMs": int((time.perf_counter() - started) * 1000),
            "runtimeEvents": events,
        }

    def _draft_structured_sql(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
        contract: Optional[NodePlanContract] = None,
        resource_safe: bool = False,
    ) -> str:
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        if not table or not columns:
            return ""
        contract = contract or self._node_plan_contract(intent, asset_pack, context)
        where = self._structured_where(intent, table, columns, context, entity_value_limit=50 if resource_safe else 200)
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        if intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
            return self._draft_structured_aggregate_sql(intent, table, columns, where_sql, contract, resource_safe=resource_safe)
        select_columns = self._structured_detail_columns(intent, columns, contract, resource_safe=resource_safe)
        return "SELECT %s FROM `%s`%s LIMIT %d" % (
            ", ".join(quote_identifier(column) for column in select_columns),
            table,
            where_sql,
            structured_limit(intent.limit, detail=True, resource_safe=resource_safe),
        )

    def _draft_structured_aggregate_sql(
        self,
        intent: QuestionIntent,
        table: str,
        columns: set,
        where_sql: str,
        contract: NodePlanContract,
        resource_safe: bool = False,
    ) -> str:
        group_columns = self._structured_group_columns(intent, columns, contract)
        select_parts = [quote_identifier(column) for column in group_columns]
        metric_parts = structured_metric_select_parts(intent, table, columns)
        if metric_parts is None:
            return ""
        if metric_parts:
            for index, (metric_expr, metric_alias) in enumerate(metric_parts):
                select_parts.append("%s AS `%s`" % (metric_expr, metric_alias))
                if index == 0:
                    order_expr = "`%s` DESC" % metric_alias
        else:
            count_alias = count_alias_for_table(table)
            select_parts.append("COUNT(*) AS `%s`" % count_alias)
            order_expr = "`%s` DESC" % count_alias
        if not resource_safe:
            for column in self._structured_context_columns(intent, columns, group_columns, contract):
                select_parts.append("MAX(`%s`) AS `%s`" % (column, column))
        if not group_columns:
            return "SELECT %s FROM `%s`%s" % (", ".join(select_parts), table, where_sql)
        return "SELECT %s FROM `%s`%s GROUP BY %s ORDER BY %s LIMIT %d" % (
            ", ".join(select_parts),
            table,
            where_sql,
            ", ".join(quote_identifier(column) for column in group_columns),
            order_expr,
            structured_limit(intent.limit, detail=False, resource_safe=resource_safe),
        )

    def _structured_detail_columns(self, intent: QuestionIntent, columns: set, contract: NodePlanContract, resource_safe: bool = False) -> List[str]:
        visible_columns = set(contract.visible_columns or [])
        preferred = []
        for column in intent.output_keys + intent.required_evidence + [intent.filter_column, intent.group_by_column, intent.metric_column]:
            if column and column in columns and column in visible_columns and column not in preferred:
                preferred.append(column)
        for spec in metric_specs_for_intent(intent, intent.preferred_table):
            for column in metric_spec_source_columns(spec, columns):
                if column and column in columns and column in visible_columns and column not in preferred:
                    preferred.append(column)
        configured = configured_contract_detail_columns(contract, columns, visible_columns)
        for column in configured:
            if column not in preferred:
                preferred.append(column)
        if resource_safe:
            for column in ["seller_id", "merchant_id", "pt"]:
                if column in visible_columns and column not in preferred:
                    preferred.append(column)
            return preferred[:12] or sorted(visible_columns)[:8]
        for column in ["seller_id", "merchant_id", "pt", "order_id", "sub_order_id", "spu_id", "spu_name", "refund_id", "ticket_id", "bill_id", "coupon_id", "pay_amt", "repay_amt"]:
            if column in visible_columns and column not in preferred:
                preferred.append(column)
        return preferred[:24] or sorted(visible_columns)[:16]

    def _structured_group_columns(self, intent: QuestionIntent, columns: set, contract: NodePlanContract) -> List[str]:
        visible_columns = set(contract.visible_columns or [])
        group_columns: List[str] = []
        allowed_output_keys = self._aggregate_output_group_keys(intent)
        for column in [intent.group_by_column] + allowed_output_keys:
            if column and column in columns and column in visible_columns and column not in group_columns:
                group_columns.append(column)
        if "seller_id" in group_columns and len(group_columns) > 1:
            group_columns.remove("seller_id")
            group_columns.insert(0, "seller_id")
        if intent.group_by_column == "pt":
            return [column for column in ["seller_id", "merchant_id", "pt"] if column in group_columns or column == "pt" and column in visible_columns]
        if intent.task_role == TaskRole.DEPENDENT:
            return group_columns[:10]
        return group_columns[:6]

    def _structured_context_columns(self, intent: QuestionIntent, columns: set, group_columns: List[str], contract: NodePlanContract) -> List[str]:
        if intent.task_role != TaskRole.DEPENDENT:
            return []
        visible_columns = set(contract.visible_columns or [])
        blocked = set(group_columns) | {intent.metric_column}
        context_columns: List[str] = []
        for column in intent.required_evidence + intent.output_keys:
            if column in blocked or column not in columns or column not in visible_columns or column in context_columns:
                continue
            if aggregate_context_column_allowed(column):
                context_columns.append(column)
        return context_columns[:8]

    def _aggregate_output_group_keys(self, intent: QuestionIntent) -> List[str]:
        keys = []
        for column in intent.output_keys:
            if not aggregate_group_key_allowed(intent, column):
                continue
            if column in {"seller_id", "merchant_id", "sub_order_id", "order_id", "ticket_id", "spu_id", "spu_name", "coupon_id", "discount_rel_id"}:
                keys.append(column)
        if intent.task_role == TaskRole.DEPENDENT:
            context_keys: List[str] = []
            for column in intent.required_evidence:
                if column in keys or column in context_keys:
                    continue
                if is_dependent_context_column(column) and aggregate_group_key_allowed(intent, column):
                    context_keys.append(column)
            if context_keys:
                priority_entity_keys = {
                    "seller_id",
                    "merchant_id",
                    "sub_order_id",
                    "order_id",
                    "ticket_id",
                    "bill_id",
                    "refund_id",
                    "spu_id",
                    "spu_name",
                    "coupon_id",
                    "discount_rel_id",
                }
                insert_at = sum(1 for column in keys if column in priority_entity_keys)
                keys = keys[:insert_at] + context_keys + keys[insert_at:]
        return keys

    def _structured_where(
        self,
        intent: QuestionIntent,
        table: str,
        columns: set,
        context: NodeExecutionContext,
        entity_value_limit: int = 200,
    ) -> List[str]:
        where: List[str] = []
        if "seller_id" in columns:
            where.append("`seller_id` = %s" % sql_literal(context.merchant_id))
        elif "merchant_id" in columns:
            where.append("`merchant_id` = %s" % sql_literal(context.merchant_id))
        elif "shop_id" in columns:
            where.append("`shop_id` = %s" % sql_literal(context.merchant_id))
        if intent.filter_column and intent.filter_column in columns and intent.filter_value:
            where.append(filter_predicate(intent.filter_column, intent.filter_value))
        if aggregate_entity_key_requires_non_empty_filter(intent, columns):
            column = intent.group_by_column
            where.append("`%s` IS NOT NULL" % column)
            where.append("`%s` != ''" % column)
        applied_entity_columns: set[str] = set()
        for entity in context.upstream_entity_sets:
            for column, values in entity.column_values.items():
                if not values or column not in columns or column in applied_entity_columns:
                    continue
                where.append("`%s` IN (%s)" % (column, ", ".join(sql_literal(value) for value in values[:entity_value_limit])))
                applied_entity_columns.add(column)
            if entity.values and entity.join_key in columns and entity.join_key not in applied_entity_columns:
                where.append("`%s` IN (%s)" % (entity.join_key, ", ".join(sql_literal(value) for value in entity.values[:entity_value_limit])))
                applied_entity_columns.add(entity.join_key)
        if "pt" in columns:
            if table == "dwm_goods_detail_df" and intent.task_role == TaskRole.DEPENDENT:
                merchant_filter = "`seller_id` = %s" % sql_literal(context.merchant_id) if "seller_id" in columns else "1=1"
                where.append("`pt` = (SELECT MAX(`pt`) FROM `%s` WHERE %s)" % (table, merchant_filter))
            elif not any("`pt`" in predicate for predicate in where):
                where.append("`pt` >= DATE_SUB(CURDATE(), INTERVAL %d DAY)" % max(intent.days or 7, 1))
        return where

    def _repair_sql(
        self,
        sql: str,
        validation: SqlValidationResult,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> SqlRepairAttempt:
        attempt = SqlRepairAttempt(
            task_id=intent.plan_task_id,
            round=1,
            original_sql=sql,
            error_code=validation.error_code,
            error_message=validation.message,
        )
        if not self.llm.configured:
            return attempt
        repair_args = {
            "taskId": intent.plan_task_id,
            "table": intent.preferred_table,
            "errorCode": validation.error_code,
            "sql": trim_sql(sql, 500),
        }
        blocked = self.tool_failure_registry.should_block("repair_sql", repair_args)
        if blocked:
            attempt.error_message = "repair_sql blocked by circuit breaker: %s" % blocked.reason
            return attempt
        tool = sql_repair_tool()
        prompt = self.prompt_assembler.render(
            "node.sql_repair",
            sections={
                "repair_policy": (
                    "只能基于 nodePlanContract 修 SQL。UNKNOWN_COLUMN、MEM_ALLOC_FAILED、timeout 后只能收敛字段、分区、实体键和 limit，"
                    "不能扩大扫描，不能修改 QueryGraph、preferredTable、指标或依赖。"
                ),
            },
        )
        contract = self._node_plan_contract(intent, asset_pack, context)
        user_payload = json.dumps(
            {
                "failedSql": sql,
                "error": validation.model_dump(by_alias=True),
                "nodePlanContract": contract.model_dump(by_alias=True),
                "rules": "只修复当前 SQL；保持单表、商家过滤、pt/实体过滤和 LIMIT；不要新增字段或表。",
            },
            ensure_ascii=False,
            default=str,
        )
        try:
            if hasattr(self.llm, "tool_json_chat"):
                payload = self.llm.tool_json_chat(prompt.system_prompt, user_payload, tool.openai_schema(), {})
            else:
                payload = self.llm.json_chat(prompt.system_prompt, user_payload, {})
        except Exception as exc:
            self.tool_failure_registry.record_failure("repair_sql", repair_args, "PROVIDER_ERROR", str(exc))
            payload = {}
        self._prompt_traces_by_task[prompt_trace_key(intent, "repair")] = prompt.trace()
        self._prompt_traces_by_task[prompt_trace_key(intent, "repair_tool")] = tool.trace_schema()
        attempt.repaired_sql = str(payload.get("sql") or "").strip()
        attempt.success = bool(attempt.repaired_sql)
        if attempt.success:
            self.tool_failure_registry.record_success("repair_sql", repair_args)
        else:
            error_type = "PROVIDER_ERROR" if self.llm.last_error else "SQL_EMPTY"
            self.tool_failure_registry.record_failure("repair_sql", repair_args, error_type, self.llm.last_error or "LLM returned empty repair SQL")
        return attempt

    def _node_scope_validation(
        self,
        validation: SqlValidationResult,
        intent: QuestionIntent,
        sql: str,
        asset_pack: PlanningAssetPack,
    ) -> SqlValidationResult:
        if not validation.valid or not intent.preferred_table:
            return validation
        out_of_scope = [table for table in validation.base_tables if table != intent.preferred_table]
        if out_of_scope:
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "OUT_OF_NODE_TABLE_SCOPE",
                    "message": "NodeWorker 只能查询 preferredTable=%s，不能引用表: %s" % (intent.preferred_table, out_of_scope),
                    "unknown_tables": out_of_scope,
                }
            )
        columns = set(asset_pack.known_columns(intent.preferred_table))
        normalized = " ".join((sql or "").lower().split())
        if ({"seller_id", "merchant_id", "shop_id"} & columns) and not has_merchant_filter_predicate(sql, columns):
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "MISSING_MERCHANT_FILTER",
                    "message": "Node SQL 必须包含可绑定的 seller_id / merchant_id / shop_id 商家过滤",
                }
            )
        if "pt" in columns and not self._pt_safety_exempt(intent) and "pt" not in normalized:
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "MISSING_PARTITION_FILTER",
                    "message": "Node SQL 必须包含 pt 分区过滤或 pt 分组",
                }
            )
        if "pt" in columns and invalid_pt_date_filter(sql):
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "INVALID_PARTITION_FILTER",
                    "message": "pt 是 Doris DATE 分区列，时间窗必须直接使用 DATE_SUB/CURDATE DATE 比较，不能包成 DATE_FORMAT(..., '%Y%m%d') 字符串",
                }
            )
        return validation

    def _contract_scope_validation(
        self,
        validation: SqlValidationResult,
        intent: QuestionIntent,
        sql: str,
        contract: NodePlanContract,
    ) -> SqlValidationResult:
        if not validation.valid:
            return validation
        try:
            parsed = sqlglot.parse_one((sql or "").strip(), read="doris")
        except Exception:
            return validation
        allowed = set(contract.allowed_columns)
        visible = set(contract.visible_columns)
        internal_only = set(contract.internal_only_columns)
        if allowed:
            metric_aliases = contract_metric_aliases(contract)
            unknown_contract_columns = sql_scope_unknown_contract_columns(parsed, allowed, metric_aliases)
            if unknown_contract_columns:
                return validation.model_copy(
                    update={
                        "valid": False,
                        "error_code": "UNKNOWN_CONTRACT_COLUMN",
                        "message": "SQL 使用了 nodePlanContract.allowedColumns 外字段: %s"
                        % sorted(set(unknown_contract_columns)),
                        "unknown_columns": sorted(set(unknown_contract_columns)),
                    }
                )
        selected = selected_output_names(parsed)
        if selected and internal_only:
            denied = [column for column in selected if column in internal_only]
            if denied:
                return validation.model_copy(
                    update={
                        "valid": False,
                        "error_code": "PERMISSION_DENIED_OUTPUT_COLUMN",
                        "message": "SQL 试图输出受语义层权限约束的字段: %s" % sorted(set(denied)),
                    }
                )
        if intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
            required_outputs = [column for column in ([contract.group_by_column] + contract.output_keys) if column]
            missing = [column for column in required_outputs if column in allowed and column not in selected]
            required_metric_aliases = list(contract_metric_aliases(contract))
            missing.extend(alias for alias in required_metric_aliases if alias and alias not in selected)
            if missing:
                return validation.model_copy(
                    update={
                        "valid": False,
                        "error_code": "MISSING_OUTPUT_KEY",
                        "message": "聚合 SQL 缺少 contract 要求输出字段: %s" % sorted(set(missing)),
                    }
                )
            denied_required = [column for column in required_outputs if column and column in allowed and column not in visible]
            if denied_required:
                return validation.model_copy(
                    update={
                        "valid": False,
                        "error_code": "PERMISSION_DENIED_OUTPUT_COLUMN",
                        "message": "聚合 SQL 依赖了受语义层权限约束的输出字段: %s" % sorted(set(denied_required)),
                    }
                )
            if aggregate_entity_key_requires_non_empty_filter(intent, allowed) and not has_non_empty_filter(sql, contract.group_by_column):
                return validation.model_copy(
                    update={
                        "valid": False,
                        "error_code": "MISSING_ENTITY_KEY_FILTER",
                        "message": "实体维度 TOPN/GROUP_AGG 必须过滤空 groupByColumn，避免空实体桶影响排名和依赖传递",
                    }
                )
        return validation

    def _pt_safety_exempt(self, intent: QuestionIntent) -> bool:
        return intent.preferred_table == "dwm_goods_detail_df" and intent.task_role == TaskRole.DEPENDENT

    def _node_plan_contract(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> NodePlanContract:
        table = intent.preferred_table
        table_columns = asset_pack.known_columns(table)
        required_columns = self._node_required_columns(intent, asset_pack).get(table, [])
        allowed_columns = list(required_columns)
        contract_output_keys = self._node_contract_output_keys(intent, set(table_columns))
        for column in formula_columns(intent.metric_formula, set(table_columns)):
            if column not in allowed_columns:
                allowed_columns.append(column)
        for spec in metric_specs_for_intent(intent, table):
            for column in metric_spec_source_columns(spec, set(table_columns)):
                if column not in allowed_columns:
                    allowed_columns.append(column)
        for entity in context.upstream_entity_sets:
            for column in entity.column_values:
                if column in table_columns and column not in allowed_columns:
                    allowed_columns.append(column)
            if entity.join_key and entity.join_key in table_columns and entity.join_key not in allowed_columns:
                allowed_columns.append(entity.join_key)
        table_metadata = table_asset_metadata(asset_pack, table)
        merchant_filter_column = str(table_metadata.get("merchantFilterColumn") or "")
        if not merchant_filter_column:
            if "seller_id" in table_columns:
                merchant_filter_column = "seller_id"
            elif "merchant_id" in table_columns:
                merchant_filter_column = "merchant_id"
            elif "shop_id" in table_columns:
                merchant_filter_column = "shop_id"
        row_scope_policy = normalize_row_access_policy(table_metadata.get("rowAccessPolicy") or default_row_access_policy(merchant_filter_column))
        access_role = str(context.access_role or DEFAULT_ACCESS_ROLE)
        field_semantics = table_field_semantics(asset_pack, table)
        visible_columns: List[str] = []
        internal_only_columns: List[str] = []
        column_access_policy: Dict[str, Dict[str, Any]] = {}
        column_display_policy: Dict[str, Dict[str, Any]] = {}
        masked_columns: Dict[str, str] = {}
        protected_internal = {merchant_filter_column, str(row_scope_policy.get("filterColumn") or "")} - {""}
        for column in allowed_columns:
            semantic = field_semantics.get(column, {})
            visibility_policy = normalize_visibility_policy(semantic.get("visibilityPolicy") or {})
            masking_policy = normalize_masking_policy(semantic.get("maskingPolicy") or {})
            policy = {
                **visibility_policy,
                "maskingStrategy": masking_policy.get("strategy") or "none",
                "maskingReason": masking_policy.get("reason") or "",
            }
            column_access_policy[column] = policy
            column_display_policy[column] = normalize_column_display_policy(semantic)
            permitted = role_allowed_for_column(visibility_policy, access_role)
            if permitted:
                visible_columns.append(column)
            elif column in protected_internal or column in required_columns or column == intent.metric_column:
                internal_only_columns.append(column)
            if permitted and masking_policy.get("strategy") not in {"", "none"}:
                masked_columns[column] = str(masking_policy.get("strategy") or "none")
        return NodePlanContract(
            task_id=intent.plan_task_id,
            question=context.question,
            preferred_table=table,
            allowed_columns=allowed_columns,
            visible_columns=visible_columns,
            internal_only_columns=internal_only_columns,
            required_columns=required_columns,
            metric_column=intent.metric_column,
            metric_name=intent.metric_name,
            metric_formula=intent.metric_formula,
            metric_specs=metric_specs_for_intent(intent, table),
            group_by_column=intent.group_by_column,
            output_keys=contract_output_keys,
            required_evidence=intent.required_evidence,
            days=int(intent.days or 0),
            limit=int(intent.limit or 0),
            merchant_id=context.merchant_id,
            merchant_filter_column=merchant_filter_column,
            access_role=access_role,
            row_scope_policy=row_scope_policy,
            column_access_policy=column_access_policy,
            column_display_policy=column_display_policy,
            masked_columns=masked_columns,
            answer_mode=enum_text(intent.answer_mode),
            task_role=enum_text(intent.task_role),
            sql_strategy=intent.sql_strategy or "llm_plan_bound_first",
            upstream_entity_sets=[item.model_dump(by_alias=True) for item in context.upstream_entity_sets],
            metric_resolution=intent.metric_resolution,
        )

    def _node_contract_output_keys(self, intent: QuestionIntent, columns: set) -> List[str]:
        if intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
            keys: List[str] = []
            for column in [intent.group_by_column] + self._aggregate_output_group_keys(intent):
                if column and column in columns and column not in keys:
                    keys.append(column)
            return keys
        return [column for column in intent.output_keys if column and column in columns]

    def _node_asset_tables(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Dict[str, List[str]]:
        names = self._node_table_names(intent, asset_pack)
        return {table: asset_pack.known_columns(table)[:100] for table in names}

    def _node_required_columns(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Dict[str, List[str]]:
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        requested: List[str] = []
        for item in intent.required_evidence + intent.output_keys + [intent.filter_column, intent.group_by_column, intent.metric_column]:
            if item and item in columns and item not in requested:
                requested.append(item)
        for spec in metric_specs_for_intent(intent, table):
            for item in metric_spec_source_columns(spec, columns):
                if item and item in columns and item not in requested:
                    requested.append(item)
        if intent.answer_mode == AnswerMode.DETAIL:
            for item in configured_default_detail_columns(asset_pack, table, columns):
                if item and item not in requested:
                    requested.append(item)
        for item in [
            "seller_id",
            "merchant_id",
            "pt",
            "sub_order_id",
            "order_id",
            "spu_id",
            "spu_name",
            "refund_id",
            "ticket_id",
            "bill_id",
            "coupon_id",
            "discount_rel_id",
            "pay_amt",
            "repay_amt",
        ]:
            if item in columns and item not in requested:
                requested.append(item)
        if not requested:
            requested = asset_pack.known_columns(table)[:16]
        return {table: requested[:32]} if table else {}

    def _node_access_hints(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Dict[str, Dict[str, Any]]:
        hints: Dict[str, Dict[str, Any]] = {}
        for table in self._node_table_names(intent, asset_pack):
            columns = set(asset_pack.known_columns(table))
            hints[table] = table_access_hint(table, columns)
        return hints

    def _node_relationships(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> List[Dict[str, Any]]:
        names = set(self._node_table_names(intent, asset_pack))
        relationships = []
        for rel in asset_pack.relationships:
            if rel.left_table in names or rel.right_table in names:
                relationships.append(
                    {
                        "relationshipId": rel.relationship_id,
                        "leftTable": rel.left_table,
                        "rightTable": rel.right_table,
                        "joinKeys": rel.join_keys,
                        "description": rel.description[:120],
                    }
                )
            if len(relationships) >= 16:
                break
        return relationships

    def _node_skills(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> List[Dict[str, Any]]:
        skills = []
        for skill in asset_pack.skills:
            skills.append(
                {
                    "domain": skill.domain,
                    "retrievalHints": skill.retrieval_hints[:4],
                    "fieldWarnings": skill.field_warnings[:4],
                    "answerGuidelines": skill.answer_guidelines[:4],
                }
            )
        return skills

    def _node_table_names(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> List[str]:
        names: List[str] = []
        if intent.preferred_table:
            names.append(intent.preferred_table)
        if not names:
            names = asset_pack.known_tables()[:1]
        return names[:1]

    def _check_dependency_coverage(self, plan: QueryPlan, task_results: List[AgentTaskResult]) -> EvidenceCheckResult:
        gaps = []
        for dep in plan.dependencies:
            anchor_ok = any(item.task_id == dep.anchor_task_id and item.success for item in task_results)
            dep_ok = any(item.task_id == dep.dependent_task_id and item.success for item in task_results)
            if not anchor_ok or not dep_ok:
                gaps.append("%s->%s 未完整覆盖" % (dep.anchor_task_id, dep.dependent_task_id))
        return EvidenceCheckResult(passed=not gaps, summary="证据图校验通过" if not gaps else "证据图存在缺口", gaps=gaps)


def merge_query_bundles(bundles: List[QueryBundle]) -> QueryBundle:
    rows: List[Dict[str, Any]] = []
    tables: List[str] = []
    first_error = ""
    for bundle in bundles:
        for table in bundle.tables:
            if table not in tables:
                tables.append(table)
        if bundle.failed and not first_error:
            first_error = bundle.error or bundle.summary
        if not bundle.failed:
            rows.extend(bundle.rows)
    return QueryBundle(
        tables=tables,
        rows=rows[:200],
        original_row_count=len(rows),
        failed=bool(bundles) and not any(not item.failed for item in bundles),
        error=first_error,
        summary="合并 %s 个 NodeWorker 结果" % len(bundles),
        duration_ms=sum(int(bundle.duration_ms or 0) for bundle in bundles),
        cache_hit=any(bundle.cache_hit for bundle in bundles),
    )


ENTITY_MERGE_KEY_PRIORITY = [
    "spu_id",
    "sku_id",
    "sub_order_id",
    "order_id",
    "refund_id",
    "ticket_id",
    "bill_id",
    "coupon_id",
    "discount_rel_id",
    "pt",
]


def merge_task_result_bundles(task_results: List[AgentTaskResult]) -> QueryBundle:
    bundles = [item.query_bundle for item in task_results]
    merge_key = choose_merge_entity_key(task_results)
    if not merge_key:
        return merge_query_bundles(bundles)
    tables = merged_bundle_tables(bundles)
    first_error = first_bundle_error(bundles)
    merged_by_key: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    scalar_rows: List[Tuple[str, Dict[str, Any]]] = []
    for task_result in task_results:
        bundle = task_result.query_bundle
        if bundle.failed:
            continue
        task_id = task_result.task_id or task_result.node_task_profile.task_id or "node"
        for row in bundle.rows or []:
            if merge_key not in row or blank_entity_value(row.get(merge_key)):
                if len(bundle.rows or []) == 1:
                    scalar_rows.append((task_id, row))
                continue
            key_value = row.get(merge_key)
            if key_value not in merged_by_key:
                merged_by_key[key_value] = {merge_key: key_value}
                order.append(key_value)
            merge_row_fields(merged_by_key[key_value], row, task_id, merge_key)
    merged_rows = [merged_by_key[key] for key in order]
    if scalar_rows and merged_rows:
        for task_id, row in scalar_rows:
            for merged in merged_rows:
                merge_row_fields(merged, row, task_id, merge_key)
    elif scalar_rows:
        for task_id, row in scalar_rows:
            target: Dict[str, Any] = {}
            merge_row_fields(target, row, task_id, merge_key)
            if target:
                merged_rows.append(target)
    if not merged_rows:
        return merge_query_bundles(bundles)
    return QueryBundle(
        tables=tables,
        rows=merged_rows[:200],
        original_row_count=len(merged_rows),
        failed=bool(bundles) and not any(not item.failed for item in bundles),
        error=first_error,
        summary="按实体键 %s 合并 %s 个 NodeWorker 结果" % (merge_key, len(bundles)),
        duration_ms=sum(int(bundle.duration_ms or 0) for bundle in bundles),
        cache_hit=any(bundle.cache_hit for bundle in bundles),
    )


def choose_merge_entity_key(task_results: List[AgentTaskResult]) -> str:
    key_hits: Dict[str, int] = {}
    for task_result in task_results:
        bundle = task_result.query_bundle
        if bundle.failed:
            continue
        row_keys = set()
        for row in bundle.rows or []:
            row_keys.update(str(key) for key in row.keys() if not blank_entity_value(row.get(key)))
        for key in ENTITY_MERGE_KEY_PRIORITY:
            if key in row_keys:
                key_hits[key] = key_hits.get(key, 0) + 1
    for key in ENTITY_MERGE_KEY_PRIORITY:
        if key_hits.get(key, 0) >= 2:
            return key
    for key in ENTITY_MERGE_KEY_PRIORITY:
        if key_hits.get(key, 0) == 1 and len([item for item in task_results if not item.query_bundle.failed]) == 1:
            return key
    return ""


def merge_row_fields(target: Dict[str, Any], row: Dict[str, Any], task_id: str, merge_key: str) -> None:
    for key, value in row.items():
        if key == merge_key:
            continue
        if key not in target or blank_entity_value(target.get(key)):
            target[key] = value
            continue
        if target.get(key) == value or blank_entity_value(value):
            continue
        namespaced = "%s__%s" % (task_id, key)
        if namespaced not in target:
            target[namespaced] = value


def merged_bundle_tables(bundles: List[QueryBundle]) -> List[str]:
    tables: List[str] = []
    for bundle in bundles:
        for table in bundle.tables:
            if table not in tables:
                tables.append(table)
    return tables


def first_bundle_error(bundles: List[QueryBundle]) -> str:
    for bundle in bundles:
        if bundle.failed and (bundle.error or bundle.summary):
            return bundle.error or bundle.summary
    return ""


def optimize_query_plan_for_execution(plan: QueryPlan, asset_pack: PlanningAssetPack) -> None:
    """Merge structurally equivalent metric nodes before NodeWorker execution."""

    if not plan or len(plan.intents) < 2:
        return
    intents_by_id = {intent.plan_task_id: intent for intent in plan.intents if intent.plan_task_id}
    groups: Dict[Tuple[Any, ...], List[QuestionIntent]] = {}
    for intent in plan.intents:
        key = same_table_metric_merge_key(intent, asset_pack, intents_by_id)
        if key:
            groups.setdefault(key, []).append(intent)
    task_id_map: Dict[str, str] = {}
    removed_task_ids: Set[str] = set()
    merge_notes: List[str] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        primary = primary_metric_intent(group)
        primary_id = primary.plan_task_id
        if not primary_id:
            continue
        ordered_group = [primary] + [intent for intent in group if intent is not primary]
        merged_ids = [intent.plan_task_id for intent in group if intent.plan_task_id]
        specs: List[Dict[str, Any]] = []
        output_keys: List[str] = []
        required_evidence: List[str] = []
        knowledge_ref_ids: List[str] = []
        depends_on_task_ids: List[str] = []
        question_parts: List[str] = []
        limit = int(primary.limit or 0)
        for intent in ordered_group:
            specs.extend(metric_specs_for_intent(intent, intent.preferred_table))
            output_keys.extend(intent.output_keys)
            required_evidence.extend(intent.required_evidence)
            depends_on_task_ids.extend(intent.depends_on_task_ids)
            for spec in metric_specs_for_intent(intent, intent.preferred_table):
                required_evidence.extend([str(item) for item in spec.get("sourceColumns") or [] if item])
                metric_name = str(spec.get("metricName") or "")
                if metric_name:
                    required_evidence.append(metric_name)
            knowledge_ref_ids.extend(intent.knowledge_ref_ids)
            if intent.question and intent.question not in question_parts:
                question_parts.append(intent.question)
            limit = max(limit, int(intent.limit or 0))
        if primary.group_by_column and primary.group_by_column not in output_keys:
            output_keys.insert(0, primary.group_by_column)
        primary.metric_specs = dedupe_metric_specs(specs)
        primary.output_keys = dedupe_strings(output_keys)
        primary.required_evidence = dedupe_strings(required_evidence)
        primary.knowledge_ref_ids = dedupe_strings(knowledge_ref_ids)
        primary.depends_on_task_ids = dedupe_strings(dep for dep in depends_on_task_ids if dep not in merged_ids)
        primary.sql_strategy = "structured_first"
        if primary.answer_mode == AnswerMode.TOPN and primary.limit:
            primary.limit = primary.limit
        else:
            primary.limit = limit or primary.limit
        if len(question_parts) > 1:
            primary.question = "；".join(question_parts)
        for intent in group:
            if intent is primary:
                continue
            if intent.plan_task_id:
                task_id_map[intent.plan_task_id] = primary_id
                removed_task_ids.add(intent.plan_task_id)
        merge_notes.append("execution_optimizer.same_table_metric_merge:%s->%s" % ("+".join(merged_ids), primary_id))
    if not removed_task_ids:
        return
    plan.intents = [intent for intent in plan.intents if intent.plan_task_id not in removed_task_ids]
    rewrite_plan_dependencies(plan, task_id_map)
    plan.evidence_contracts = EvidenceContractBuilder().contracts_from_intents(plan.intents)
    plan.final_required_evidence = EvidenceContractBuilder().final_evidence_labels(plan.intents)
    plan.agent_trace.extend(merge_notes)
    optimizer_notes = plan.compiler_trace if isinstance(plan.compiler_trace, list) else []
    optimizer_notes.extend(merge_notes)
    plan.compiler_trace = optimizer_notes


def primary_metric_intent(group: List[QuestionIntent]) -> QuestionIntent:
    for intent in group:
        if intent.answer_mode == AnswerMode.TOPN and not intent.depends_on_task_ids:
            return intent
    for intent in group:
        if not intent.depends_on_task_ids:
            return intent
    return group[0]


def same_table_metric_merge_key(
    intent: QuestionIntent,
    asset_pack: PlanningAssetPack,
    intents_by_id: Dict[str, QuestionIntent] | None = None,
) -> Tuple[Any, ...] | None:
    base_key = same_table_metric_base_key(intent, asset_pack)
    if not base_key:
        return None
    dependency_key = tuple(sorted(intent.depends_on_task_ids))
    if intents_by_id:
        dependency_key = tuple(
            sorted(
                task_id
                for task_id in intent.depends_on_task_ids
                if same_table_metric_base_key(intents_by_id.get(task_id, QuestionIntent()), asset_pack) != base_key
            )
        )
    role_key = enum_text(intent.task_role) if dependency_key else ""
    return (*base_key, role_key, dependency_key)


def same_table_metric_base_key(intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Tuple[Any, ...] | None:
    if intent.intent_type != IntentType.VALID:
        return None
    if intent.answer_mode not in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
        return None
    if not intent.preferred_table or intent.sql:
        return None
    if not (intent.metric_column or intent.metric_formula or intent.metric_specs):
        return None
    columns = set(asset_pack.known_columns(intent.preferred_table))
    if not columns:
        return None
    if intent.group_by_column and intent.group_by_column not in columns:
        return None
    return (
        intent.preferred_table,
        intent.group_by_column,
        intent.filter_column,
        str(intent.filter_value or ""),
        int(intent.days or 0),
    )


def rewrite_plan_dependencies(plan: QueryPlan, task_id_map: Dict[str, str]) -> None:
    for intent in plan.intents:
        intent.depends_on_task_ids = dedupe_strings([task_id_map.get(task_id, task_id) for task_id in intent.depends_on_task_ids])
    rewritten: List[Any] = []
    seen: Set[Tuple[str, str, str, str, str]] = set()
    for dep in plan.dependencies:
        dep.anchor_task_id = task_id_map.get(dep.anchor_task_id, dep.anchor_task_id)
        dep.dependent_task_id = task_id_map.get(dep.dependent_task_id, dep.dependent_task_id)
        if dep.anchor_task_id == dep.dependent_task_id:
            continue
        key = (dep.anchor_task_id, dep.dependent_task_id, dep.join_key, dep.anchor_column, dep.dependent_column)
        if key in seen:
            continue
        seen.add(key)
        rewritten.append(dep)
    plan.dependencies = rewritten


def rewrite_evidence_contracts(plan: QueryPlan, task_id_map: Dict[str, str]) -> None:
    for contract in plan.evidence_contracts:
        if not isinstance(contract, dict):
            continue
        task_id = str(contract.get("taskId") or contract.get("task_id") or "")
        if task_id in task_id_map:
            contract["taskId"] = task_id_map[task_id]
            contract["task_id"] = task_id_map[task_id]


def metric_specs_for_intent(intent: QuestionIntent, table: str) -> List[Dict[str, Any]]:
    if intent.metric_specs:
        return [normalize_metric_spec(spec, intent, table) for spec in intent.metric_specs if isinstance(spec, dict)]
    if not (intent.metric_column or intent.metric_formula or intent.metric_name):
        return []
    return [normalize_metric_spec({}, intent, table)]


def normalize_metric_spec(spec: Dict[str, Any], intent: QuestionIntent, table: str) -> Dict[str, Any]:
    metric_column = str(spec.get("metricColumn") or spec.get("metric_column") or intent.metric_column or "")
    metric_formula = str(spec.get("metricFormula") or spec.get("metric_formula") or intent.metric_formula or "")
    metric_name = str(spec.get("metricName") or spec.get("metric_name") or intent.metric_name or "")
    if not metric_name:
        metric_name = metric_alias_for_values(metric_column, table)
    source_columns = [
        str(item)
        for item in spec.get("sourceColumns")
        or spec.get("source_columns")
        or (intent.metric_resolution or {}).get("sourceColumns")
        or (intent.metric_resolution or {}).get("source_columns")
        or []
        if item
    ]
    return {
        "metricName": metric_name,
        "metricColumn": metric_column,
        "metricFormula": metric_formula,
        "sourceColumns": dedupe_strings(source_columns),
        "sourceTaskId": str(spec.get("sourceTaskId") or spec.get("source_task_id") or intent.plan_task_id or ""),
    }


def dedupe_metric_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for spec in specs:
        normalized = normalize_metric_spec(spec, QuestionIntent(), "")
        key = (
            str(normalized.get("metricName") or ""),
            str(normalized.get("metricColumn") or ""),
            str(normalized.get("metricFormula") or ""),
        )
        if key in seen or not any(key):
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def structured_metric_select_parts(intent: QuestionIntent, table: str, columns: set) -> List[Tuple[str, str]] | None:
    parts: List[Tuple[str, str]] = []
    for spec in metric_specs_for_intent(intent, table):
        metric_alias = str(spec.get("metricName") or "")
        metric_formula = str(spec.get("metricFormula") or "")
        metric_column = str(spec.get("metricColumn") or "")
        metric_expr = compile_metric_formula(metric_formula, columns)
        if metric_formula and not metric_expr:
            return None
        if metric_expr:
            parts.append((metric_expr, metric_alias or "metric_value"))
            continue
        if metric_column and metric_column in columns:
            alias = metric_alias or metric_alias_for_values(metric_column, table)
            if is_count_metric_alias(alias):
                parts.append(("COUNT(DISTINCT `%s`)" % metric_column, alias))
            else:
                parts.append(("SUM(`%s`)" % metric_column, alias))
    return parts


def metric_spec_source_columns(spec: Dict[str, Any], columns: set) -> List[str]:
    found = [str(item) for item in spec.get("sourceColumns") or [] if str(item) in columns]
    metric_column = str(spec.get("metricColumn") or "")
    if metric_column and metric_column in columns and metric_column not in found:
        found.append(metric_column)
    for column in formula_columns(str(spec.get("metricFormula") or ""), columns):
        if column not in found:
            found.append(column)
    return found


def metric_alias_for_values(metric_column: str, table: str) -> str:
    if metric_column == "pay_amt" and "refund" in table:
        return "refund_related_pay_amt"
    if metric_column == "pay_amt":
        return "order_pay_amt"
    if metric_column == "repay_amt":
        return "repay_amt"
    if metric_column:
        return "sum_%s" % metric_column
    return "metric_value"


def metric_alias_candidates(metric_key: str) -> List[str]:
    text = str(metric_key or "")
    aliases = {
        "order_detail_cnt": ["order_detail_cnt", "order_cnt", "sub_order_cnt", "cnt", "count"],
        "refund_bill_cnt": ["refund_bill_cnt", "refund_cnt", "cnt", "count"],
        "ticket_bill_cnt": ["ticket_bill_cnt", "ticket_cnt", "cnt", "count"],
        "repay_bill_cnt": ["repay_bill_cnt", "repay_cnt", "cnt", "count"],
    }
    return dedupe_strings([text] + aliases.get(text, []))


def first_present_value(row: Dict[str, Any], aliases: List[str]) -> Any:
    for alias in aliases:
        if alias in row:
            return row.get(alias)
    return None


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def dedupe_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def record_tool(
    traces: List[NodeToolCall],
    intent: QuestionIntent,
    tool_name: str,
    status: str,
    input_summary: str = "",
    output_summary: str = "",
    error_type: str = "",
    repair_round: int = 0,
    duration_ms: int = 0,
) -> None:
    traces.append(
        NodeToolCall(
            task_id=intent.plan_task_id,
            tool_name=tool_name,
            status=status,
            input_summary=str(input_summary or "")[:240],
            output_summary=str(output_summary or "")[:360],
            error_type=str(error_type or "")[:120],
            repair_round=repair_round,
            duration_ms=max(0, int(duration_ms or 0)),
        )
    )


def serialize_tool_execution_result(result: ToolCallExecutionResult) -> Dict[str, Any]:
    return {
        "id": result.id,
        "name": result.name,
        "status": result.status,
        "result": result.result,
        "errorType": result.error_type,
        "errorCode": result.error_code or result.error_type,
        "errorMessage": result.error_message,
        "retryable": result.retryable,
        "recommendedAction": result.recommended_action,
        "fallbackTools": list(result.fallback_tools),
        "details": result.details,
        "durationMs": result.duration_ms,
        "attempts": result.attempts,
        "cacheHit": result.cache_hit,
    }


def compact_tool_failure_for_prompt(item: Dict[str, Any], max_detail_chars: int = 1200) -> Dict[str, Any]:
    details = item.get("details") or {}
    if not isinstance(details, dict):
        details = {"value": str(details)[:max_detail_chars]}
    compact_details: Dict[str, Any] = {}
    for key, value in details.items():
        if value in ("", None, [], {}):
            continue
        text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
        if key.lower() in {"sqlpreview", "messagepreview", "stderr", "stdout", "traceback"} and len(text) > 500:
            text = text[:500]
        compact_details[key] = text if len(text) <= max_detail_chars else text[:max_detail_chars]
    result = item.get("result") or {}
    artifact_ref = {}
    if isinstance(result, dict):
        artifact_ref = result.get("artifactRef") or result.get("artifact") or {}
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "errorType": item.get("errorType"),
        "errorCode": item.get("errorCode") or item.get("errorType"),
        "errorMessage": str(item.get("errorMessage") or "")[:500],
        "retryable": item.get("retryable"),
        "recommendedAction": item.get("recommendedAction"),
        "fallbackTools": item.get("fallbackTools") or [],
        "details": compact_details,
        "artifactRef": artifact_ref,
    }


def compact_tool_results_for_prompt(results: List[Dict[str, Any]], max_items: int = 6, max_chars: int = 6000) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for item in results[-max_items:]:
        if str(item.get("status") or "") in {"failed", "error", "blocked", "timeout", "rate_limited", "circuit_blocked"}:
            compacted.append(compact_tool_failure_for_prompt(item))
            continue
        payload = {
            "name": item.get("name"),
            "status": item.get("status"),
            "errorType": item.get("errorType"),
            "errorCode": item.get("errorCode") or item.get("errorType"),
            "errorMessage": item.get("errorMessage"),
            "retryable": item.get("retryable"),
            "recommendedAction": item.get("recommendedAction"),
            "fallbackTools": item.get("fallbackTools") or [],
            "details": item.get("details") or {},
        }
        result = item.get("result") or {}
        if isinstance(result, dict):
            for key in ["relativePath", "merchantUri", "truncated", "estimatedChars", "nextContentOffsetChars"]:
                if key in result:
                    payload[key] = result.get(key)
            if "content" in result:
                payload["content"] = str(result.get("content") or "")[:1800]
            if "items" in result:
                payload["items"] = result.get("items")[:8] if isinstance(result.get("items"), list) else result.get("items")
            if "hits" in result:
                payload["hits"] = result.get("hits")[:8] if isinstance(result.get("hits"), list) else result.get("hits")
        if len(json.dumps(payload, ensure_ascii=False, default=str)) > max_chars:
            payload["content"] = str(payload.get("content") or "")[: max(200, max_chars // 2)]
            payload.pop("items", None)
            payload.pop("hits", None)
        compacted.append(payload)
    return compacted


def issue(code: str, reason: str, evidence: str = "") -> Dict[str, Any]:
    return {"code": code, "severity": "error", "reason": reason, "evidence": evidence}


def enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def formula_columns(formula: str, known_columns: Set[str]) -> List[str]:
    return reconciled_formula_columns(formula, known_columns)


def metric_resolution_aliases(resolution: Dict[str, Any]) -> Set[str]:
    aliases: Set[str] = set()
    for key in ["metricKey", "requestedMetricRef", "displayName"]:
        value = resolution.get(key)
        if value:
            aliases.add(str(value))
    for column in resolution.get("sourceColumns") or []:
        if column:
                aliases.add(str(column))
    return aliases


def contract_metric_aliases(contract: NodePlanContract) -> Set[str]:
    aliases: Set[str] = set()
    for spec in contract.metric_specs:
        metric_name = str(spec.get("metricName") or spec.get("metric_name") or "")
        if metric_name:
            aliases.add(metric_name)
            continue
        metric_column = str(spec.get("metricColumn") or spec.get("metric_column") or "")
        if metric_column:
            aliases.add(metric_alias_for_values(metric_column, contract.preferred_table))
    if contract.metric_name:
        aliases.add(contract.metric_name)
    elif contract.metric_column:
        aliases.add(metric_alias_for_values(contract.metric_column, contract.preferred_table))
    resolution = contract.metric_resolution or {}
    metric_key = str(resolution.get("metricKey") or resolution.get("metric_key") or "")
    if metric_key:
        aliases.add(metric_key)
    return {alias for alias in aliases if alias}


def sql_scope_unknown_columns(parsed: exp.Expression, asset_pack: PlanningAssetPack) -> List[str]:
    unknown_columns: List[str] = []
    for scope in traverse_scope(parsed):
        select_aliases = immediate_select_aliases(scope.expression)
        selected_sources = getattr(scope, "selected_sources", {}) or {}
        for column in scope.columns:
            column_name = column.name
            if not column_name or column_name.lower() in SQL_BUILTIN_IDENTIFIERS:
                continue
            if is_select_alias_reference(column, select_aliases):
                continue
            source_alias, source = resolve_scope_column_source(scope, column)
            if source is None:
                continue
            if is_derived_scope_source(source):
                output_names = source_output_names(source, asset_pack)
                if output_names is not None and column_name not in output_names:
                    unknown_columns.append("%s.%s" % (source_alias or "derived", column_name))
                continue
            if isinstance(source, exp.Table):
                base_table = source.name
                known_columns = set(asset_pack.known_columns(base_table))
                if known_columns and column_name not in known_columns:
                    unknown_columns.append("%s.%s" % (source_alias or base_table, column_name))
                continue
            if not selected_sources and len(set(asset_pack.known_tables())) == 1:
                base_table = next(iter(asset_pack.known_tables()))
                known_columns = set(asset_pack.known_columns(base_table))
                if known_columns and column_name not in known_columns:
                    unknown_columns.append("%s.%s" % (base_table, column_name))
    return sorted(set(unknown_columns))


def sql_scope_unknown_contract_columns(parsed: exp.Expression, allowed: Set[str], metric_aliases: Set[str]) -> List[str]:
    unknown_columns: List[str] = []
    for scope in traverse_scope(parsed):
        select_aliases = immediate_select_aliases(scope.expression)
        for column in scope.columns:
            column_name = column.name
            if not column_name or column_name.lower() in SQL_BUILTIN_IDENTIFIERS:
                continue
            if is_select_alias_reference(column, select_aliases):
                continue
            _source_alias, source = resolve_scope_column_source(scope, column)
            if source is not None and is_derived_scope_source(source):
                output_names = source_output_names(source)
                if output_names is None or column_name in output_names or column_name in metric_aliases:
                    continue
            if column_name not in allowed:
                unknown_columns.append(column_name)
    return sorted(set(unknown_columns))


def resolve_scope_column_source(scope: Any, column: exp.Column) -> Tuple[str, Any]:
    selected_sources = getattr(scope, "selected_sources", {}) or {}
    table_alias = column.table
    if table_alias:
        pair = selected_sources.get(table_alias)
        return table_alias, pair[1] if pair else None
    if len(selected_sources) == 1:
        source_alias, pair = next(iter(selected_sources.items()))
        return source_alias, pair[1]
    for source_alias, pair in selected_sources.items():
        source = pair[1]
        if is_derived_scope_source(source):
            output_names = source_output_names(source)
            if output_names is None or column.name in output_names:
                return source_alias, source
    return "", None


def is_derived_scope_source(source: Any) -> bool:
    return hasattr(source, "expression") and hasattr(source, "selected_sources")


def source_output_names(source: Any, asset_pack: Optional[PlanningAssetPack] = None) -> Optional[Set[str]]:
    if not is_derived_scope_source(source):
        return None
    names = selected_output_names(source.expression)
    if names:
        return names
    if not getattr(source, "stars", None):
        return set()
    expanded: Set[str] = set()
    for _, pair in (getattr(source, "selected_sources", {}) or {}).items():
        child = pair[1]
        if isinstance(child, exp.Table):
            if asset_pack is None:
                return None
            expanded.update(asset_pack.known_columns(child.name))
            continue
        child_names = source_output_names(child, asset_pack)
        if child_names is None:
            return None
        expanded.update(child_names)
    return expanded or None


def immediate_select_aliases(expression: exp.Expression) -> Set[str]:
    if not isinstance(expression, exp.Select):
        return set()
    return {item.alias for item in expression.expressions if getattr(item, "alias", "")}


def selected_output_names(parsed: exp.Expression) -> Set[str]:
    names: Set[str] = set()
    select = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    if not select:
        return names
    for expression in select.expressions:
        alias = getattr(expression, "alias", "")
        if alias:
            names.add(alias)
            continue
        if isinstance(expression, exp.Column):
            names.add(expression.name)
            continue
        column = expression.find(exp.Column)
        if column and not expression.find(exp.AggFunc):
            names.add(column.name)
    return names


def contract_select_required_columns(contract: NodePlanContract) -> List[str]:
    columns: List[str] = []
    for column in [contract.group_by_column] + list(contract.output_keys):
        if column and column not in columns:
            columns.append(column)
    return columns


def invalid_pt_date_filter(sql: str) -> bool:
    upper = (sql or "").upper()
    if "PT" not in upper:
        return False
    return "DATE_FORMAT" in upper and "%Y%M%D" in upper


def is_select_alias_reference(column: exp.Column, select_aliases: set) -> bool:
    if column.name not in select_aliases:
        return False
    return bool(column.find_ancestor(exp.Order, exp.Group, exp.Having))


def trim_sql(sql: str, limit: int = 260) -> str:
    text = " ".join((sql or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def sanitize_node_artifact_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "node")).strip("._")
    return text or "node"


def prompt_trace_key(intent: QuestionIntent, phase: str) -> str:
    return "%s:%s" % (intent.plan_task_id or intent.preferred_table or "node", phase)


def append_prompt_marker(summary: str, trace: Dict[str, Any]) -> str:
    marker = "prompt=%s@%s" % (trace.get("promptId") or "", trace.get("version") or "")
    if not summary:
        return marker
    if marker in summary:
        return summary
    return "%s | %s" % (summary, marker)


def append_tool_schema_marker(summary: str, schema: Dict[str, Any]) -> str:
    name = str(schema.get("name") or "")
    if not name:
        return summary
    marker = "tool=%s" % name
    if not summary:
        return marker
    if marker in summary:
        return summary
    return "%s | %s" % (summary, marker)


def classify_doris_error(error_text: str) -> str:
    lower = (error_text or "").lower()
    if "unknown column" in lower or "unknown" in lower and "column" in lower:
        return "UNKNOWN_COLUMN"
    if "mem_alloc_failed" in lower or "memory" in lower:
        return "MEM_ALLOC_FAILED"
    if "timeout" in lower or "timed out" in lower:
        return "TIMEOUT"
    if "syntax" in lower or "parse" in lower:
        return "SQL_SYNTAX"
    return "DORIS_ERROR"


def doris_error_policy(error_code: str) -> Dict[str, bool]:
    code = error_code or "DORIS_ERROR"
    if code in RESOURCE_CONSTRAINED_DORIS_ERRORS:
        return {"structured_fallback": False, "resource_fallback": True, "llm_repair": False}
    if code == "UNKNOWN_COLUMN":
        return {"structured_fallback": True, "resource_fallback": False, "llm_repair": True}
    if code == "SQL_SYNTAX":
        return {"structured_fallback": True, "resource_fallback": False, "llm_repair": True}
    return {"structured_fallback": True, "resource_fallback": False, "llm_repair": True}


def structured_limit(limit: int, detail: bool, resource_safe: bool = False) -> int:
    raw_limit = max(int(limit or 20), 1)
    if resource_safe:
        return min(raw_limit, 10 if detail else 20)
    return min(raw_limit, 50 if detail else 100)


def equivalent_sql(left: str, right: str) -> bool:
    return " ".join((left or "").split()).lower() == " ".join((right or "").split()).lower()


def filter_predicate(column: str, value: Any) -> str:
    raw = str(value or "")
    if "," in raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        return "`%s` IN (%s)" % (column, ", ".join(sql_literal(item) for item in values))
    return "`%s` = %s" % (column, sql_literal(value))


def count_alias_for_table(table: str) -> str:
    if "refund" in table:
        return "refund_cnt"
    if "ticket" in table:
        return "ticket_cnt"
    if "repay" in table:
        return "repay_cnt"
    if "coupon" in table:
        return "coupon_cnt"
    if "scm" in table:
        return "scm_cnt"
    if "goods" in table:
        return "goods_cnt"
    return "order_cnt"


FORMULA_ALLOWED_TOKENS = {
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "DISTINCT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "NULLIF",
    "COALESCE",
    "IFNULL",
    "CAST",
    "AS",
    "DECIMAL",
    "DOUBLE",
    "SIGNED",
    "UNSIGNED",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
}


def compile_metric_formula(formula: str, columns: set) -> str:
    return compile_reconciled_metric_formula(formula, columns)


def metric_alias_for_intent(intent: QuestionIntent, table: str) -> str:
    if intent.metric_name:
        return intent.metric_name
    if intent.metric_column == "pay_amt" and "refund" in table:
        return "refund_related_pay_amt"
    if intent.metric_column == "pay_amt":
        return "order_pay_amt"
    if intent.metric_column == "repay_amt":
        return "repay_amt"
    if intent.metric_column:
        return "sum_%s" % intent.metric_column
    return "metric_value"


def is_count_metric_alias(alias: str) -> bool:
    text = (alias or "").lower()
    return text.endswith("_cnt") or text.endswith("_count") or "count" in text


def entity_set_from_rows(task_id: str, intent: QuestionIntent, rows: List[Dict[str, Any]], max_values: int) -> EntitySet:
    key = intent.group_by_column or intent.filter_column or "sub_order_id"
    if rows and key not in rows[0]:
        for candidate in ["sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id", "refund_id", "pt"]:
            if candidate in rows[0]:
                key = candidate
                break
    values = []
    column_values: Dict[str, List[Any]] = {}
    for row in rows:
        value = row.get(key)
        if not blank_entity_value(value) and value not in values:
            values.append(value)
        for column in ["sub_order_id", "order_id", "spu_id", "spu_name", "ticket_id", "refund_id", "bill_id", "coupon_id", "discount_rel_id", "pt"]:
            if column not in row:
                continue
            candidate_value = row.get(column)
            if blank_entity_value(candidate_value):
                continue
            items = column_values.setdefault(column, [])
            if candidate_value not in items:
                items.append(candidate_value)
    truncated = len(values) > max_values or any(len(items) > max_values for items in column_values.values())
    return EntitySet(
        task_id=task_id or "",
        join_key=key,
        values=values[:max_values],
        column_values={column: items[:max_values] for column, items in column_values.items() if items},
        truncated=truncated,
        source_row_count=len(rows),
    )


def upstream_missing_reason(entity_sets: List[EntitySet]) -> str:
    if any(entity.values or entity.column_values for entity in entity_sets):
        return ""
    reasons = [entity.missing_reason for entity in entity_sets if entity.missing_reason]
    for reason in ["JOIN_KEY_NOT_PRODUCED", "JOIN_KEY_VALUES_EMPTY", "UPSTREAM_SQL_FAILED", "UPSTREAM_ZERO_ROWS"]:
        if reason in reasons:
            return reason
    return "UPSTREAM_ENTITY_MISSING"


def aggregate_entity_key_requires_non_empty_filter(intent: QuestionIntent, columns: set) -> bool:
    if intent.answer_mode not in {AnswerMode.TOPN, AnswerMode.GROUP_AGG}:
        return False
    column = intent.group_by_column or ""
    if not column or column not in columns:
        return False
    return column in entity_dimension_columns()


def entity_dimension_columns() -> set[str]:
    return {
        "sub_order_id",
        "order_id",
        "spu_id",
        "spu_name",
        "ticket_id",
        "bill_id",
        "refund_id",
        "coupon_id",
        "discount_rel_id",
    }


def has_non_empty_filter(sql: str, column: str) -> bool:
    if not sql or not column:
        return False
    normalized = " ".join((sql or "").replace("`", "").lower().split())
    col = column.lower()
    has_not_null = "%s is not null" % col in normalized
    has_not_empty = (
        "%s != ''" % col in normalized
        or "%s <> ''" % col in normalized
        or "length(%s) > 0" % col in normalized
        or "char_length(%s) > 0" % col in normalized
    )
    return has_not_null and has_not_empty



def aggregate_group_key_allowed(intent: QuestionIntent, column: str) -> bool:
    if not column:
        return False
    group_by = intent.group_by_column or ""
    if column in {"seller_id", "merchant_id"}:
        return True
    if column == group_by:
        return True
    if group_by == "pt":
        return column == "pt"
    companion_keys = {
        "spu_id": {"spu_name"},
        "spu_name": {"spu_id"},
        "coupon_id": {"discount_rel_id"},
        "discount_rel_id": {"coupon_id"},
    }
    if column in companion_keys.get(group_by, set()):
        return True
    detail_keys = {"sub_order_id", "order_id", "ticket_id", "bill_id", "refund_id", "coupon_id", "discount_rel_id", "pt"}
    if column in detail_keys:
        return intent.task_role == TaskRole.DEPENDENT and column in set(intent.required_evidence)
    return column in {"spu_id", "spu_name"}


def aggregate_context_column_allowed(column: str) -> bool:
    text = (column or "").lower()
    return any(token in text for token in ["status", "time", "name", "reason", "type"])


def dependent_skip_message(reason: str) -> str:
    if reason == "JOIN_KEY_NOT_PRODUCED":
        return "JOIN_KEY_NOT_PRODUCED：上游节点未产出 dependency join key，跳过 dependent 节点执行"
    if reason == "JOIN_KEY_VALUES_EMPTY":
        return "JOIN_KEY_VALUES_EMPTY：上游节点 join key 值为空，跳过 dependent 节点执行"
    if reason == "UPSTREAM_SQL_FAILED":
        return "UPSTREAM_SQL_FAILED：上游节点 SQL 失败，跳过 dependent 节点执行"
    if reason == "UPSTREAM_ZERO_ROWS":
        return "UPSTREAM_ZERO_ROWS：上游节点返回 0 行，跳过 dependent 节点执行"
    return "上游实体集缺失，跳过 dependent 节点执行"


def choose_entity_transfer_key(
    dep: Any,
    rows: List[Dict[str, Any]],
    parent_result: AgentTaskResult,
    dependent_columns: set,
) -> tuple:
    row_keys = set(rows[0].keys()) if rows else set()
    preferred_pairs = paired_join_tokens(dep.anchor_column, dep.dependent_column) + paired_join_tokens(dep.join_key, dep.join_key)
    preferred_pairs.sort(key=lambda item: (item[0] in {"seller_id", "merchant_id"}, item[0]))
    for key, dep_key in preferred_pairs:
        if key in {"seller_id", "merchant_id"} or dep_key in {"seller_id", "merchant_id"}:
            continue
        if key in row_keys and dep_key in dependent_columns:
            return key, dep_key
    if parent_result.entity_set:
        entity_key = parent_result.entity_set.join_key
        if entity_key in dependent_columns and entity_key not in {"seller_id", "merchant_id"}:
            return entity_key, entity_key
    for key in ["sub_order_id", "order_id", "spu_id", "spu_name", "refund_id", "ticket_id", "bill_id", "coupon_id", "pt"]:
        if key in row_keys and key in dependent_columns:
            return key, key
    return "", dep.dependent_column or dep.join_key


def multi_entity_transfer_values(
    dep: Any,
    rows: List[Dict[str, Any]],
    parent_result: AgentTaskResult,
    dependent_columns: set,
    max_values: int,
) -> Dict[str, List[Any]]:
    column_values: Dict[str, List[Any]] = {}
    pairs = paired_join_tokens(dep.anchor_column, dep.dependent_column) + paired_join_tokens(dep.join_key, dep.join_key)
    allowed_dependent_keys = dependency_allowed_dependent_entity_keys(dep, dependent_columns)
    if not rows:
        if not parent_result.entity_set:
            return {}
        return {
            column: values[:max_values]
            for column, values in parent_result.entity_set.column_values.items()
            if column in allowed_dependent_keys and any(not blank_entity_value(value) for value in values)
        }
    row_keys = set(rows[0].keys())
    for key, dep_key in pairs:
        if key in {"seller_id", "merchant_id"} or dep_key in {"seller_id", "merchant_id"}:
            continue
        if key not in row_keys or dep_key not in dependent_columns:
            continue
        values = column_values.setdefault(dep_key, [])
        for row in rows:
            value = row.get(key)
            if not blank_entity_value(value) and value not in values:
                values.append(value)
    if parent_result.entity_set:
        for column, values in parent_result.entity_set.column_values.items():
            if column in dependent_columns and column not in {"seller_id", "merchant_id"}:
                if allowed_dependent_keys and column not in allowed_dependent_keys:
                    continue
                target_values = column_values.setdefault(column, [])
                for value in values:
                    if not blank_entity_value(value) and value not in target_values:
                        target_values.append(value)
        if parent_result.entity_set.join_key in dependent_columns and parent_result.entity_set.join_key not in {"seller_id", "merchant_id"}:
            if not allowed_dependent_keys or parent_result.entity_set.join_key in allowed_dependent_keys:
                target_values = column_values.setdefault(parent_result.entity_set.join_key, [])
                for value in parent_result.entity_set.values:
                    if not blank_entity_value(value) and value not in target_values:
                        target_values.append(value)
    return {
        column: [value for value in values if not blank_entity_value(value)][:max_values]
        for column, values in column_values.items()
        if any(not blank_entity_value(value) for value in values)
    }


def dependency_allowed_dependent_entity_keys(dep: Any, dependent_columns: set) -> set[str]:
    pairs = paired_join_tokens(dep.anchor_column, dep.dependent_column) + paired_join_tokens(dep.join_key, dep.join_key)
    return {
        dep_key
        for key, dep_key in pairs
        if key not in {"seller_id", "merchant_id"} and dep_key not in {"seller_id", "merchant_id"} and dep_key in dependent_columns
    }


def paired_join_tokens(anchor_value: str, dependent_value: str) -> List[Tuple[str, str]]:
    anchor_tokens = split_join_tokens(anchor_value)
    dependent_tokens = split_join_tokens(dependent_value)
    pairs: List[Tuple[str, str]] = []
    for index, key in enumerate(anchor_tokens):
        dep_key = dependent_tokens[index] if index < len(dependent_tokens) else key
        pair = (key, dep_key)
        if key and dep_key and pair not in pairs:
            pairs.append(pair)
    return pairs


def split_join_tokens(value: str) -> List[str]:
    if not value:
        return []
    tokens: List[str] = []
    for piece in str(value).replace("+", ",").split(","):
        token = piece.strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def is_repairable_doris_error(error_text: str) -> bool:
    return doris_error_policy(classify_doris_error(error_text)).get("llm_repair", False)


def table_access_hint(table: str, columns: set) -> Dict[str, Any]:
    catalog: Dict[str, Dict[str, Any]] = {
        "dwm_trade_order_detail_di": {
            "uniqueKeys": ["sub_order_id", "pt"],
            "distributionKeys": ["sub_order_id"],
            "bestEqualityFilters": ["seller_id", "sub_order_id", "pt"],
            "fallbackFilters": ["seller_id", "order_id", "pt"],
            "invertedIndexes": [],
        },
        "dwm_trade_refund_detail_di": {
            "uniqueKeys": ["refund_id", "pt"],
            "distributionKeys": ["refund_id"],
            "bestEqualityFilters": ["seller_id", "refund_id", "pt"],
            "fallbackFilters": ["seller_id", "sub_order_id", "order_id", "pt"],
            "invertedIndexes": [],
        },
        "dwm_goods_detail_df": {
            "tableKind": "snapshot_dimension",
            "uniqueKeys": ["spu_id", "pt"],
            "distributionKeys": ["spu_id"],
            "bestEqualityFilters": ["seller_id", "spu_id", "pt"],
            "fallbackFilters": ["seller_id", "spu_name", "pt"],
            "invertedIndexes": [],
            "timeWindowPolicy": "do_not_apply_question_window_for_dependent_lookup; use latest pt for seller_id + spu_id",
        },
        "dwm_cs_ticket_detail_di": {
            "uniqueKeys": ["ticket_id", "pt"],
            "distributionKeys": ["ticket_id"],
            "bestEqualityFilters": ["seller_id", "ticket_id", "pt"],
            "fallbackFilters": ["seller_id", "sub_order_id", "order_id", "pt"],
            "invertedIndexes": [],
        },
        "dwm_cs_repay_detail_df": {
            "tableKind": "snapshot_fact",
            "uniqueKeys": ["bill_id", "pt"],
            "distributionKeys": ["bill_id"],
            "bestEqualityFilters": ["seller_id", "bill_id", "pt"],
            "fallbackFilters": ["seller_id", "sub_order_id", "order_id", "pt"],
            "invertedIndexes": [],
        },
        "dwm_coupon_detail_di": {
            "uniqueKeys": ["coupon_id", "pt"],
            "distributionKeys": ["coupon_id"],
            "bestEqualityFilters": ["seller_id", "coupon_id", "pt"],
            "fallbackFilters": ["seller_id", "order_id", "sub_order_id", "pt"],
            "invertedIndexes": [],
        },
        "dwm_scm_detail_di": {
            "uniqueKeys": ["scm_id", "pt"],
            "distributionKeys": ["scm_id"],
            "bestEqualityFilters": ["seller_id", "scm_id", "pt"],
            "fallbackFilters": ["seller_id", "spu_id", "pt"],
            "invertedIndexes": [],
        },
    }
    hint = catalog.get(table, {"uniqueKeys": ["pt"], "distributionKeys": [], "bestEqualityFilters": ["pt"], "fallbackFilters": [], "invertedIndexes": []})
    payload: Dict[str, Any] = {}
    for key, value in hint.items():
        if isinstance(value, list):
            payload[key] = [column for column in value if column in columns]
        else:
            payload[key] = value
    return payload


def timed_out_result(task_id: str, intent: QuestionIntent, timeout_seconds: int) -> AgentTaskResult:
    result = failed_result(task_id, intent, "NodeWorker 超时：超过 %s 秒未返回" % timeout_seconds)
    result.query_bundle.runtime_events.append(
        {
            "event": "node.timeout",
            "taskId": task_id,
            "timeoutSeconds": timeout_seconds,
            "timeoutType": classify_timeout_type("node execution timed out", source="node"),
        }
    )
    return result


def failed_result(task_id: str, intent: QuestionIntent, message: str) -> AgentTaskResult:
    return AgentTaskResult(
        task_id=task_id or intent.plan_task_id,
        success=False,
        summary=message,
        query_bundle=QueryBundle(sql=intent.sql, tables=[intent.preferred_table] if intent.preferred_table else [], failed=True, error=message),
        react_trace=[ReActStep(round=1, reason="NodeWorker 失败", action="node_worker.failed", observation=message[:240])],
        node_task_profile=NodeTaskProfile(
            task_id=task_id or intent.plan_task_id,
            task_kind="FAILED_BEFORE_TOOL_CHAIN",
            sql_strategy=intent.sql_strategy or "llm_first",
            reason=message[:240],
            risk_controls=["single_table_scope"],
        ),
    )
