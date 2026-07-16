from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextvars import copy_context
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
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
    GraphValidationResult,
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
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
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
    equivalent_formula_text,
    formula_columns as reconciled_formula_columns,
)
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.planning import (
    EvidenceContractBuilder,
    QueryGraphValidator,
    query_plan_question_coverage_gaps,
)
from merchant_ai.services.planning_layers import GraphContractValidator
from merchant_ai.services.prompts import PromptAssembler
from merchant_ai.services.query_contracts import (
    collect_degraded_reasons,
    contract_gaps_from_task_results,
    tenant_scope_binding_error,
)
from merchant_ai.services.repositories import DorisRepository
from merchant_ai.services.semantic_metrics import seal_semantic_metric_resolution, semantic_metric_contract_issue
from merchant_ai.services.runtime_state import NodeTaskState, create_runtime_state_store, node_task_idempotency_key
from merchant_ai.services.distributed_workers import DistributedSubAgentClient
from merchant_ai.services.query_sql_binding import (
    add_sql_where_condition,
    append_note,
    bind_node_sql_parameters,
    blank_entity_value,
    has_merchant_filter_predicate,
    normalize_inclusive_relative_window_sql,
    parse_partition_date,
    partition_is_stale_for_near_realtime,
    quote_identifier,
    realtime_fallback_for_table,
    split_detail_sql_by_time_windows,
    sql_has_bound_merchant_filter,
    sql_literal,
)
from merchant_ai.services.query_security import (
    DEFAULT_ACCESS_ROLE,
    apply_column_masks,
    configured_contract_detail_columns,
    declared_result_access_policy,
    configured_default_detail_columns,
    role_allowed_for_column,
    table_asset_metadata,
    table_field_semantics,
)
from merchant_ai.services.time_semantics import (
    CALENDAR_ANCHOR_POLICY,
    LATEST_PARTITION_ANCHOR_POLICY,
    latest_partition_window_predicate,
    time_window_contract_payload,
)
from merchant_ai.services.tool_runtime import (
    ToolFailureRegistry,
    ToolRuntimePolicyRegistry,
    ToolRuntimeService,
    classify_timeout_type,
    current_tool_cancel_event,
)
from merchant_ai.services.tools import artifact_file_tool_definitions, canonical_tool_registry, node_runtime_tool_schemas, semantic_file_tool_definitions, sql_draft_tool, sql_repair_tool


SQL_BUILTIN_IDENTIFIERS = {"current_date", "current_timestamp", "current_time", "curdate", "now"}


class ExecutionGraphPreparationError(RuntimeError):
    """Raised before SQL dispatch when the execution graph contract is not ready."""


class ExecutionGraphPreparationRequired(ExecutionGraphPreparationError):
    """The caller supplied a graph that still changes under execution normalization."""


class ExecutionGraphValidationError(ExecutionGraphPreparationError):
    """The normalized execution graph did not pass the graph contract validator."""


@dataclass(frozen=True)
class ExecutionGraphPreparation:
    """Immutable hand-off contract between graph normalization and NodeWorker.

    ``plan`` is a deep copy of the caller's QueryGraph.  The remaining hashes bind
    the validation result to the exact question, semantic asset pack and normalized
    graph that may be dispatched to SQL workers.
    """

    plan: QueryPlan
    validation: GraphValidationResult
    source_plan_fingerprint: str
    execution_plan_fingerprint: str
    question_fingerprint: str
    asset_pack_fingerprint: str
    changed: bool
    optimization_notes: Tuple[str, ...] = ()
    validator_name: str = ""
    freshness_reports: Tuple[FreshnessCheckResult, ...] = ()
    runtime_fallback_task_ids: Tuple[str, ...] = ()
    runtime_source_plan_fingerprint: str = ""

    @property
    def executable(self) -> bool:
        return bool(self.validation.valid)

    def require_executable(self) -> QueryPlan:
        if self.validation.valid:
            return self.plan
        gap_codes = [str(gap.code or "") for gap in self.validation.gaps if str(gap.code or "")]
        raise ExecutionGraphValidationError(
            "normalized execution graph is invalid; gaps=%s"
            % (",".join(gap_codes[:8]) or "QUERY_GRAPH_VALIDATION_FAILED")
        )

    @property
    def freshness_bound(self) -> bool:
        """Whether runtime freshness was evaluated for this execution hand-off."""

        return bool(self.runtime_source_plan_fingerprint)


def execution_question_fingerprint(question: str) -> str:
    return hashlib.sha256(str(question or "").encode("utf-8")).hexdigest()


def execution_asset_pack_fingerprint(asset_pack: PlanningAssetPack) -> str:
    payload = asset_pack.model_dump(by_alias=True, mode="json")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sql_references_filter_column(sql: str, column: str) -> bool:
    value = str(column or "").strip().strip("`")
    return bool(value and re.search(r"(?<![A-Za-z0-9_])`?%s`?\s*(?:=|IN\s*\()" % re.escape(value), sql or "", flags=re.I))


def sql_filters_column(sql: str, column: str) -> bool:
    value = str(column or "").strip().strip("`")
    if not value:
        return False
    predicate = r"(?<![A-Za-z0-9_])`?%s`?\s*(?:=|!=|<>|<|>|<=|>=|BETWEEN\b|IN\s*\()" % re.escape(value)
    return bool(re.search(predicate, sql or "", flags=re.I))
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
        "check_freshness": "check the declared time column and fallback risk",
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
        if intent_is_time_series(intent):
            risk_controls.append("declared_time_grouping")
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
        if intent_is_time_series(intent):
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
            governance_issue = metric_execution_contract_issue(contract)
            if governance_issue:
                issues.append(issue("UNGOVERNED_METRIC", governance_issue, contract.metric_name or contract.metric_column))
            binding_issue = semantic_metric_binding_issue(contract)
            if binding_issue:
                issues.append(issue("SEMANTIC_METRIC_BINDING_DRIFT", binding_issue, contract.metric_name or contract.metric_column))
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
        computation_columns = set(formula_columns(contract.metric_formula, allowed))
        for spec in contract.metric_specs:
            computation_columns.update(metric_spec_source_columns(spec, allowed))
        denied_evidence = [
            column
            for column in contract.required_evidence
            if column
            and column in allowed
            and column not in visible
            and column not in {contract.metric_column, contract.merchant_filter_column}
            and column not in computation_columns
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
            or contract.metric_specs
            or contract.answer_mode
            in {AnswerMode.METRIC.value, AnswerMode.TOPN.value, AnswerMode.GROUP_AGG.value, AnswerMode.DERIVED.value}
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
        self.distributed_subagent_client = (
            DistributedSubAgentClient(settings, state_store=self.runtime_state_store)
            if bool(settings.distributed_subagents_enabled)
            else None
        )
        self.node_plan_critic = NodePlanCritic()
        self.access_control = AccessControlService(settings)

    def with_artifact_root(self, root: str) -> None:
        self.artifact_store.set_context_root(root)

    def prepare_runtime_execution_graph(
        self,
        merchant_id: str,
        plan: QueryPlan,
        asset_pack: PlanningAssetPack,
        question: str,
        *,
        graph_validator: Any = None,
        memory_constraints: Optional[List[Dict[str, Any]]] = None,
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: Optional[Dict[str, Any]] = None,
    ) -> ExecutionGraphPreparation:
        """Bind freshness routing to the exact graph that may reach SQL.

        The logical graph is normalized and validated first.  Freshness may then
        select an asset-declared realtime table.  Any such selection creates a new
        graph version, rebuilds its evidence contract, and runs the complete graph
        preparation/validation pipeline again.  The returned preparation is the
        only graph hand-off accepted by :meth:`execute_plan`.
        """

        initial = prepare_execution_graph(
            question,
            plan,
            asset_pack,
            graph_validator,
            memory_constraints or [],
        )
        if not initial.executable:
            return initial

        runtime_source_fingerprint = initial.execution_plan_fingerprint
        runtime_plan = initial.plan.model_copy(deep=True)
        reports: List[FreshnessCheckResult] = []
        fallback_task_ids: List[str] = []
        scope = dict(user_scope or {})
        for index, intent in enumerate(runtime_plan.intents):
            if intent.intent_type != IntentType.VALID or intent.answer_mode == AnswerMode.RULE:
                continue
            context = NodeExecutionContext(
                merchant_id=merchant_id,
                effective_user_id=str(scope.get("userId") or scope.get("user_id") or ""),
                authorized_region=str(scope.get("region") or ""),
                authorized_store_ids=[
                    str(item)
                    for item in (scope.get("storeIds") or scope.get("store_ids") or [])
                    if str(item or "").strip()
                ],
                access_role=access_role or DEFAULT_ACCESS_ROLE,
                question=question or intent.question,
                context_package={"userScope": scope},
            )
            report = self._check_freshness(intent, asset_pack, context)
            fallback_intent = self._maybe_realtime_fallback_intent(intent, asset_pack, report)
            if fallback_intent is not None:
                report.status = "STALE_USE_REALTIME_FALLBACK"
                report.fallback_table = fallback_intent.preferred_table
                report.reason = "%s; switch_to_realtime=%s" % (
                    report.reason or "offline table stale",
                    fallback_intent.preferred_table,
                )
                runtime_plan.intents[index] = fallback_intent
                fallback_task_ids.append(intent.plan_task_id)
            reports.append(report)

        final = initial
        if fallback_task_ids:
            evidence_builder = EvidenceContractBuilder()
            runtime_plan.evidence_contracts = evidence_builder.contracts_from_intents(runtime_plan.intents)
            runtime_plan.final_required_evidence = evidence_builder.final_evidence_labels(runtime_plan.intents)
            runtime_plan.agent_trace.append(
                "execution_runtime.realtime_fallback=%s" % ",".join(fallback_task_ids)
            )
            final = prepare_execution_graph(
                question,
                runtime_plan,
                asset_pack,
                graph_validator,
                memory_constraints or [],
            )

        return replace(
            final,
            changed=bool(final.changed or fallback_task_ids),
            freshness_reports=tuple(report.model_copy(deep=True) for report in reports),
            runtime_fallback_task_ids=tuple(fallback_task_ids),
            runtime_source_plan_fingerprint=runtime_source_fingerprint,
        )

    def execute_plan(
        self,
        merchant_id: str,
        plan: QueryPlan,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        question: str,
        resume_task_results: Optional[List[AgentTaskResult]] = None,
        run_id: str = "",
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: Optional[Dict[str, Any]] = None,
        execution_mode: str = "auto",
        execution_preparation: Optional[ExecutionGraphPreparation] = None,
    ) -> AgentRunResult:
        plan = require_normalized_execution_plan(
            plan,
            asset_pack,
            question,
            execution_preparation=execution_preparation,
        )
        result = AgentRunResult(executed_query_graph_fingerprint=query_graph_fingerprint(plan))
        prepared_freshness_by_task = {
            report.task_id: report.model_copy(deep=True)
            for report in (execution_preparation.freshness_reports if execution_preparation else ())
            if report.task_id
        }
        execution_run_id = run_id or "inline_%s" % uuid.uuid4().hex[:16]
        executable = [intent for intent in plan.intents if intent.intent_type == IntentType.VALID and intent.answer_mode != AnswerMode.RULE]
        tasks_by_id = {intent.plan_task_id or "node_%s" % (index + 1): intent for index, intent in enumerate(executable)}
        contract_hashes = {
            task_id: node_execution_contract_hash(intent, plan, merchant_id, access_role, user_scope or {}, asset_pack)
            for task_id, intent in tasks_by_id.items()
        }
        completed: Dict[str, AgentTaskResult] = {}
        for prior in resume_task_results or []:
            expected_hash = contract_hashes.get(prior.task_id, "")
            if (
                expected_hash
                and prior.execution_contract_hash == expected_hash
                and prior.success
                and not prior.query_bundle.failed
            ):
                completed[prior.task_id] = prior
            elif prior.task_id in tasks_by_id:
                result.resume_rejected_task_ids.append(prior.task_id)
        result.resumed_task_ids = list(completed.keys())
        pending = {task_id: intent for task_id, intent in tasks_by_id.items() if task_id not in completed}
        dag_evidence_gaps: List[EvidenceGap] = []
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
                blocked_results, blocked_gaps, blocked_batch = self._fail_closed_dag_batch(
                    pending,
                    tasks_by_id,
                )
                result.node_execution_batches.append(blocked_batch)
                dag_evidence_gaps.extend(blocked_gaps)
                for task_id, task_result in blocked_results.items():
                    task_result.execution_contract_hash = contract_hashes.get(task_id, "")
                    completed[task_id] = task_result
                pending.clear()
                break
            batch_results, batch_trace = self._execute_ready_batch(
                ready_ids,
                pending,
                completed,
                plan,
                merchant_id,
                asset_pack,
                question,
                knowledge_context,
                execution_run_id,
                access_role=access_role,
                user_scope=user_scope,
                execution_mode=execution_mode,
                prepared_freshness_by_task=prepared_freshness_by_task,
            )
            result.node_execution_batches.append(batch_trace)
            for task_id, task_result in batch_results.items():
                task_result.execution_contract_hash = contract_hashes.get(task_id, "")
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

        result.merged_query_bundle = merge_task_result_bundles(result.task_results, self.artifact_store)
        result.evidence_check = self._check_dependency_coverage(plan, result.task_results)
        dependency_check_gaps = list(result.evidence_check.gaps)
        if dag_evidence_gaps:
            result.evidence_check.passed = False
            result.evidence_check.summary = "NodeWorker DAG 存在不可执行的依赖缺口"
            result.evidence_check.gaps.extend(
                "%s:%s:%s" % (gap.code, gap.task_id, gap.reason)
                for gap in dag_evidence_gaps
            )
        lineage_gaps: List[EvidenceGap] = []
        if not result.merged_query_bundle.lineage_complete:
            lineage_gaps.append(
                EvidenceGap(
                    code="RESULT_LINEAGE_INCOMPLETE",
                    evidence=",".join(result.merged_query_bundle.source_row_counts.keys()),
                    reason=result.merged_query_bundle.summary,
                    severity="blocking",
                    disclosure_required=True,
                    source="result_lineage",
                    answer_instruction="完整结果工件不可用，禁止基于预览行计算、排序或声明完整结论。",
                    suggested_action="restore_full_artifact_or_rerun_query",
                )
            )
        result.evidence_gaps = dag_evidence_gaps + lineage_gaps + contract_gaps_from_task_results(result.task_results) + [
            EvidenceGap(code="DEPENDENCY_GAP", task_id=gap, reason=gap) for gap in dependency_check_gaps
        ]
        result.degraded_reasons = collect_degraded_reasons(result.task_results)
        return result

    def _fail_closed_dag_batch(
        self,
        pending: Dict[str, QuestionIntent],
        tasks_by_id: Dict[str, QuestionIntent],
    ) -> tuple[Dict[str, AgentTaskResult], List[EvidenceGap], NodeExecutionBatch]:
        pending_ids = list(pending.keys())
        known_task_ids = set(tasks_by_id)
        unresolved_ids: Set[str] = set()
        direct_reasons: Dict[str, str] = {}
        for task_id, intent in pending.items():
            dependencies = list(intent.depends_on_task_ids or [])
            missing_dependencies = [dependency for dependency in dependencies if dependency not in known_task_ids]
            if intent.task_role == TaskRole.DEPENDENT and not dependencies:
                unresolved_ids.add(task_id)
                direct_reasons[task_id] = "DEPENDENT task has no dependency task ids"
            elif missing_dependencies:
                unresolved_ids.add(task_id)
                direct_reasons[task_id] = "dependency task reference does not exist: %s" % ", ".join(missing_dependencies)

        changed = True
        while changed:
            changed = False
            for task_id, intent in pending.items():
                if task_id in unresolved_ids:
                    continue
                unresolved_dependencies = [
                    dependency
                    for dependency in intent.depends_on_task_ids
                    if dependency in unresolved_ids
                ]
                if not unresolved_dependencies:
                    continue
                unresolved_ids.add(task_id)
                direct_reasons[task_id] = "dependency chain includes unresolved task(s): %s" % ", ".join(unresolved_dependencies)
                changed = True

        task_failures: List[Dict[str, Any]] = []
        results: Dict[str, AgentTaskResult] = {}
        gaps: List[EvidenceGap] = []
        for task_id, intent in pending.items():
            dependencies = list(intent.depends_on_task_ids or [])
            if task_id in unresolved_ids:
                code = "UNRESOLVED_DEPENDENCY"
                reason = direct_reasons[task_id]
            else:
                code = "CYCLIC_GRAPH"
                reason = "dependency closure is blocked by a cycle among pending tasks"
            message = "%s: %s" % (code, reason)
            task_result = failed_result(task_id, intent, message)
            task_result.node_task_profile.task_kind = "DAG_BLOCKED"
            task_result.node_task_profile.contract_status = code
            task_result.query_bundle.runtime_events.append(
                {
                    "event": "node.dag_blocked",
                    "taskId": task_id,
                    "errorCode": code,
                    "dependsOn": dependencies,
                    "reason": reason,
                }
            )
            results[task_id] = task_result
            gaps.append(
                EvidenceGap(
                    code=code,
                    task_id=task_id,
                    evidence=",".join(dependencies),
                    reason=reason,
                    severity="blocking",
                    disclosure_required=True,
                    source="node_scheduler",
                    answer_instruction="QueryGraph 依赖不可执行；不要运行 SQL，也不要把缺失结果解释成业务数据为 0。",
                    details={
                        "gapCode": code,
                        "taskId": task_id,
                        "dependsOn": dependencies,
                        "knownTaskIds": sorted(known_task_ids),
                    },
                )
            )
            task_failures.append(
                {
                    "taskId": task_id,
                    "errorCode": code,
                    "dependsOn": dependencies,
                    "reason": reason,
                }
            )

        batch = NodeExecutionBatch(
            batch_id="dag_blocked_%03d" % (int(time.time() * 1000) % 1000),
            ready_task_ids=[],
            failed_task_ids=pending_ids,
            blocked_task_ids=pending_ids,
            max_concurrency=0,
            timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
            runtime_events=[
                {
                    "event": "node.dag_fail_closed",
                    "status": "failed",
                    "blockedTaskIds": pending_ids,
                    "taskFailures": task_failures,
                }
            ],
        )
        return results, gaps, batch

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
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: Optional[Dict[str, Any]] = None,
        execution_mode: str = "auto",
        prepared_freshness_by_task: Optional[Dict[str, FreshnessCheckResult]] = None,
    ) -> tuple[Dict[str, AgentTaskResult], NodeExecutionBatch]:
        started = time.perf_counter()
        run_id = run_id or "inline_%s" % uuid.uuid4().hex[:16]
        configured_workers = max(1, int(self.settings.max_concurrent_sub_agents or 1))
        task_cap = max(1, int(getattr(self.settings, "max_sub_agent_tasks", configured_workers) or configured_workers))
        max_workers = max(1, min(configured_workers, task_cap, len(ready_ids)))
        results: Dict[str, AgentTaskResult] = {}
        requested_execution_mode = str(execution_mode or "auto").lower()
        batch = NodeExecutionBatch(
            batch_id="batch_%03d" % (len(ready_ids) + int(time.time() * 1000) % 1000),
            ready_task_ids=list(ready_ids),
            max_concurrency=max_workers,
            timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
        )
        batch.runtime_events.append({"event": "node.execution_tier", "requestedMode": requested_execution_mode})
        recovered_tasks = self.runtime_state_store.recover_expired_node_tasks(max_attempts=3)
        if recovered_tasks:
            batch.runtime_events.append({"event": "node.expired_leases_recovered", "count": recovered_tasks})
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
            future_modes: Dict[Any, str] = {}
            future_cancel_events: Dict[Any, Event] = {}
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
                    task_mode = self._task_execution_mode(requested_execution_mode, pending[task_id], question)
                    context = self._node_context(
                        task_id,
                        pending[task_id],
                        completed,
                        plan,
                        merchant_id,
                        question,
                        asset_pack,
                        access_role=access_role,
                        user_scope=user_scope,
                    )
                    prepared_freshness = (prepared_freshness_by_task or {}).get(task_id)
                    if prepared_freshness is not None:
                        context.context_package["preparedFreshnessReport"] = prepared_freshness.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                    context.cancel_event = Event()
                    distributed = bool(self.distributed_subagent_client and task_mode == "subagent")
                    if distributed:
                        context = self._prepare_subagent_context(task_id, pending[task_id], context, asset_pack)
                        context.context_package.update({"parentRunId": run_id, "runtimeMode": "distributed_subagent", "subagentEnabled": True})
                        future = submit_with_current_context(
                            executor,
                            self._execute_distributed_node,
                            run_id,
                            pending[task_id],
                            asset_pack,
                            knowledge_context,
                            context,
                        )
                        task_mode = "distributed_subagent"
                    else:
                        self.runtime_state_store.enqueue_node_task(
                            NodeTaskState(
                                run_id=run_id,
                                task_id=task_id,
                                status="queued",
                                idempotency_key=node_task_idempotency_key(run_id, task_id, pending[task_id].preferred_table),
                                payload={
                                    "taskKind": "query_node",
                                    "preferredTable": pending[task_id].preferred_table,
                                    "dependsOn": list(pending[task_id].depends_on_task_ids or []),
                                    "answerMode": str(pending[task_id].answer_mode),
                                },
                            )
                        )
                        lease_owner = "node_subagent" if task_mode == "subagent" else "node_direct_worker"
                        claimed = self.runtime_state_store.claim_node_task(
                            run_id,
                            task_id,
                            lease_owner=lease_owner,
                            lease_seconds=max(1, self.settings.agent_node_timeout_seconds),
                        )
                        if not claimed:
                            batch.blocked_task_ids.append(task_id)
                            batch.failed_task_ids.append(task_id)
                            results[task_id] = failed_result(task_id, pending[task_id], "node task could not acquire execution lease")
                            continue
                        if task_mode == "subagent":
                            context = self._prepare_subagent_context(task_id, pending[task_id], context, asset_pack)
                            context.context_package["parentRunId"] = run_id
                            future = submit_with_current_context(
                                executor,
                                self._run_isolated_subagent,
                                pending[task_id],
                                asset_pack,
                                knowledge_context,
                                context,
                            )
                        else:
                            context.context_package.update({"parentRunId": run_id, "runtimeMode": "direct_node_worker", "subagentEnabled": False})
                            future = submit_with_current_context(
                                executor,
                                self.execute_node,
                                pending[task_id],
                                asset_pack,
                                knowledge_context,
                                context,
                            )
                    chunk_futures[future] = task_id
                    future_modes[future] = task_mode
                    future_cancel_events[future] = context.cancel_event
                    batch.submitted_task_ids.append(task_id)
                    batch.runtime_events.append({"event": "node.task_dispatched", "taskId": task_id, "executionMode": task_mode})
                if not chunk_futures:
                    continue
                done = set()
                not_done = set(chunk_futures.keys())
                timeout_seconds = max(1, int(self.settings.agent_node_timeout_seconds or 1))
                poll_interval = max(0.1, float(getattr(self.settings, "agent_node_poll_interval_seconds", 5.0) or 5.0))
                deadline = time.perf_counter() + timeout_seconds
                next_heartbeat_at = time.perf_counter() + poll_interval
                run_was_canceled = False
                while not_done and time.perf_counter() < deadline:
                    if self.runtime_state_store.run_canceled(run_id):
                        run_was_canceled = True
                        for running_future in not_done:
                            future_cancel_events[running_future].set()
                        batch.runtime_events.append(
                            {
                                "event": "node.run_canceled",
                                "runningTaskIds": [chunk_futures[future] for future in not_done],
                            }
                        )
                        break
                    remaining = max(0.0, deadline - time.perf_counter())
                    wait_for = min(remaining, max(0.05, next_heartbeat_at - time.perf_counter()))
                    just_done, still_running = wait(not_done, timeout=wait_for, return_when=FIRST_COMPLETED)
                    done.update(just_done)
                    not_done = set(still_running)
                    now = time.perf_counter()
                    if not_done and now >= next_heartbeat_at:
                        for running_future in not_done:
                            if future_modes.get(running_future) == "distributed_subagent":
                                continue
                            self.runtime_state_store.heartbeat_node_task(
                                run_id,
                                chunk_futures[running_future],
                                "node_subagent" if future_modes.get(running_future) == "subagent" else "node_direct_worker",
                                lease_seconds=max(1, self.settings.agent_node_timeout_seconds),
                            )
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
                    future_cancel_events[future].set()
                    if run_was_canceled:
                        batch.failed_task_ids.append(task_id)
                        results[task_id] = cancelled_result(pending[task_id])
                        self.runtime_state_store.complete_node_task(run_id, task_id, "canceled", {"reason": "run canceled"})
                        future.cancel()
                        continue
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
                        future_cancel_events[future].set()
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

    def _task_execution_mode(self, requested_mode: str, intent: QuestionIntent, question: str) -> str:
        if requested_mode in {"direct", "subagent"}:
            return requested_mode
        score = 0
        if intent.task_role == TaskRole.DEPENDENT or intent.depends_on_task_ids:
            score += 2
        if intent.answer_mode in {AnswerMode.DERIVED, AnswerMode.DETAIL, AnswerMode.TOPN}:
            score += 1
        if re.search(r"原因|归因|为什么|诊断|异常|分析|建议|下钻|关联", "%s %s" % (question or "", intent.question or ""), re.I):
            score += 2
        if intent.answer_mode in {AnswerMode.METRIC, AnswerMode.GROUP_AGG} and not intent.depends_on_task_ids:
            score -= 1
        return "subagent" if score >= 2 else "direct"

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
            **dict(context.context_package or {}),
            "subAgentRunId": run_id,
            "taskId": task_id,
            "taskRole": str(intent.task_role),
            "subagentEnabled": True,
            "runtimeMode": "independent_node_react",
            "maxRounds": max(1, int(self.settings.max_sub_agent_rounds or 1)),
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

    def _run_isolated_subagent(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        observations: List[Dict[str, Any]] = []
        max_rounds = max(1, int(self.settings.max_sub_agent_rounds or 1))
        result = AgentTaskResult(task_id=intent.plan_task_id)
        for round_index in range(1, max_rounds + 1):
            if context_is_cancelled(context):
                return cancelled_result(intent)
            result = self.execute_node(intent, asset_pack, knowledge_context, context)
            observations.append(
                {
                    "round": round_index,
                    "failed": bool(result.query_bundle.failed),
                    "rows": result.query_bundle.effective_row_count(),
                    "error": str(result.query_bundle.error or "")[:300],
                }
            )
            if not result.query_bundle.failed or not result.query_bundle.error:
                break
            if context_is_cancelled(context):
                return cancelled_result(intent)
            if not any(term in str(result.query_bundle.error).lower() for term in ["timeout", "tempor", "connection", "retry"]):
                break
        context.context_package["observations"] = observations
        context.context_package["roundsUsed"] = len(observations)
        self._write_subagent_checkpoint(
            context,
            intent,
            "success" if not result.query_bundle.failed else "failed",
            {"observations": observations, "roundsUsed": len(observations)},
        )
        return result

    def _execute_distributed_node(
        self,
        run_id: str,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        knowledge_context: str,
        context: NodeExecutionContext,
    ) -> AgentTaskResult:
        if not self.distributed_subagent_client:
            return failed_result(intent.plan_task_id, intent, "distributed sub-agent client is not configured")
        result = self.distributed_subagent_client.execute(
            run_id,
            intent.plan_task_id,
            "query_node",
            {
                "intent": intent.model_dump(by_alias=True),
                "assetPack": asset_pack.model_dump(by_alias=True),
                "knowledgeContext": knowledge_context,
                "context": context.model_dump(by_alias=True),
            },
            timeout_seconds=max(1, int(self.settings.agent_node_timeout_seconds or 1)),
        )
        if result.status != "completed":
            return failed_result(intent.plan_task_id, intent, result.error or "distributed node failed: %s" % result.status)
        return AgentTaskResult.model_validate(result.result)

    def _node_context(
        self,
        task_id: str,
        intent: QuestionIntent,
        completed: Dict[str, AgentTaskResult],
        plan: QueryPlan,
        merchant_id: str,
        question: str,
        asset_pack: PlanningAssetPack,
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: Optional[Dict[str, Any]] = None,
    ) -> NodeExecutionContext:
        upstream_rows: List[Dict[str, Any]] = []
        entity_sets: List[EntitySet] = []
        dependent_columns = set(asset_pack.known_columns(intent.preferred_table))
        dependent_metadata = table_asset_metadata(asset_pack, intent.preferred_table)
        excluded_transfer_columns = {
            str(dependent_metadata.get("merchantFilterColumn") or ""),
            str((dependent_metadata.get("rowAccessPolicy") or {}).get("filterColumn") or "")
            if isinstance(dependent_metadata.get("rowAccessPolicy"), dict)
            else "",
        } - {""}
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
            key, dependent_key = choose_entity_transfer_key(
                dep,
                rows,
                parent_result,
                dependent_columns,
                excluded_transfer_columns,
            )
            column_values = multi_entity_transfer_values(
                dep,
                rows,
                parent_result,
                dependent_columns,
                self.settings.agent_max_entity_values,
                excluded_transfer_columns,
            )
            values = list(column_values.get(dependent_key, [])) if dependent_key else []
            if not values and key:
                for row in rows:
                    value = row.get(key)
                    if key in row and not blank_entity_value(value) and value not in values:
                        values.append(value)
            allowed_dependent_keys = dependency_allowed_dependent_entity_keys(
                dep,
                dependent_columns,
                excluded_transfer_columns,
            )
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
            effective_user_id=str((user_scope or {}).get("userId") or (user_scope or {}).get("user_id") or ""),
            authorized_region=str((user_scope or {}).get("region") or ""),
            authorized_store_ids=[
                str(item)
                for item in ((user_scope or {}).get("storeIds") or (user_scope or {}).get("store_ids") or [])
                if str(item or "").strip()
            ],
            access_role=access_role or DEFAULT_ACCESS_ROLE,
            question=question or intent.question,
            upstream_entity_sets=entity_sets,
            upstream_rows=upstream_rows if intent.answer_mode == AnswerMode.DERIVED else upstream_rows[: self.settings.tool_result_preview_rows],
            context_package={"userScope": dict(user_scope or {})},
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
        if context_is_cancelled(context):
            return cancelled_result(intent)
        context.runtime_scratch = {}
        try:
            if intent.answer_mode == AnswerMode.DERIVED:
                result = self._execute_derived_node(intent, asset_pack, context)
            else:
                result = self.node_agent.execute(intent, asset_pack, knowledge_context, context)
            file_tool_results = list(context.runtime_scratch.get("file_tool_results") or [])
            if file_tool_results and not result.file_tool_results:
                result.file_tool_results = file_tool_results
            if context_is_cancelled(context):
                return cancelled_result(intent)
            return self._finalize_subagent_result(result, intent, context)
        finally:
            context.runtime_scratch = {}

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
        zero_fill_numerator = derived_ratio_numerator_zero_fill_allowed(intent, components)
        for values in grouped.values():
            if denominator_key not in values:
                continue
            numerator = numeric_value(values.get(numerator_key))
            if numerator is None and zero_fill_numerator:
                numerator = 0
                values[numerator_key] = 0
            denominator = numeric_value(values.get(denominator_key))
            if numerator is None or denominator in {None, 0}:
                continue
            derived = numerator / denominator
            if unit == "%":
                derived *= 100
            row = dict(values)
            row[metric_name] = round(derived, 6)
            rows.append(row)
        if intent_is_time_series(intent):
            rows.sort(key=lambda item: str(item.get(group_key) or ""))
        else:
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
        result.file_tool_results = list(result.file_tool_results or [])
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
        freshness = self._prepared_freshness_report(intent, context)
        freshness_prepared = freshness is not None
        if freshness is None:
            freshness = self._check_freshness(intent, asset_pack, context)
        freshness_duration_ms = 0 if freshness_prepared else int((time.perf_counter() - freshness_started) * 1000)
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
        if freshness.status == "STALE_USE_REALTIME_FALLBACK":
            record_tool(
                tool_traces,
                intent,
                "select_realtime_fallback",
                "success",
                freshness.table,
                "fallbackTable=%s maxPt=%s" % (intent.preferred_table, freshness.max_pt),
                "STALE_USE_REALTIME_FALLBACK",
            )
            trace.append(
                ReActStep(
                    round=2,
                    reason="离线表近实时分区滞后，切换到实时 fallback 表",
                    action="select_realtime_fallback",
                    observation="%s -> %s" % (freshness.table, intent.preferred_table),
                )
            )
        else:
            fallback_intent = self._maybe_realtime_fallback_intent(intent, asset_pack, freshness)
            if fallback_intent is not None:
                freshness.status = "STALE_REQUIRES_GRAPH_REPREPARATION"
                freshness.fallback_table = fallback_intent.preferred_table
                freshness.reason = "%s; blocked_unprepared_realtime=%s" % (
                    freshness.reason or "offline table stale",
                    fallback_intent.preferred_table,
                )
                message = (
                    "EXECUTION_GRAPH_CHANGED_AFTER_PREPARATION：freshness selected realtime table %s "
                    "but the validated execution graph still targets %s"
                    % (fallback_intent.preferred_table, intent.preferred_table)
                )
                node_task_profile.contract_status = "EXECUTION_GRAPH_CHANGED_AFTER_PREPARATION"
                node_task_profile.contract_critique_reason = message
                trace.append(
                    ReActStep(
                        round=2,
                        reason=message,
                        action="select_realtime_fallback.blocked",
                        observation="%s -> %s" % (intent.preferred_table, fallback_intent.preferred_table),
                    )
                )
                record_tool(
                    tool_traces,
                    intent,
                    "select_realtime_fallback",
                    "failed",
                    intent.preferred_table,
                    message,
                    "EXECUTION_GRAPH_CHANGED_AFTER_PREPARATION",
                )
                record_tool(
                    tool_traces,
                    intent,
                    "summarize_node_result",
                    "failed",
                    "runtime execution graph",
                    message,
                    "EXECUTION_GRAPH_CHANGED_AFTER_PREPARATION",
                )
                return AgentTaskResult(
                    success=False,
                    summary=message,
                    query_bundle=QueryBundle(
                        tables=[intent.preferred_table] if intent.preferred_table else [],
                        failed=True,
                        error=message,
                        summary=message,
                    ),
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
        draft_decision = pop_node_runtime_value(
            context,
            "sql_draft_decision",
            SqlDraftDecision(task_id=intent.plan_task_id),
        )
        file_tool_results = list(context.runtime_scratch.get("file_tool_results") or [])
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
        draft_prompt = pop_node_prompt_trace(context, prompt_trace_key(intent, "draft"))
        if draft_prompt:
            draft_summary = append_prompt_marker(draft_summary, draft_prompt)
        draft_tool_schema = pop_node_prompt_trace(context, prompt_trace_key(intent, "draft_tool"))
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
            sql = self._enforce_identity_scope_sql(sql, contract, context)
            sql = normalize_inclusive_relative_window_sql(sql, intent.days)
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
                repair_prompt = pop_node_prompt_trace(context, prompt_trace_key(intent, "repair"))
                if repair_prompt:
                    repair_summary = append_prompt_marker(repair_summary, repair_prompt)
                repair_tool_schema = pop_node_prompt_trace(context, prompt_trace_key(intent, "repair_tool"))
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
                "params=%d tenantBound=%s"
                % (
                    len(sql_params),
                    sql_has_bound_merchant_filter(bound_sql, {contract.merchant_filter_column} - {""}),
                ),
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
            if context_is_cancelled(context):
                return cancelled_result(intent)
            query_started = time.perf_counter()
            query_cancel_event = Event()
            runtime_result = self.tool_runtime_service.execute(
                "execute_sql",
                execute_args,
                lambda _args: {
                    "rows": doris_query_with_cancellation(
                        self.doris_repository,
                        bound_sql,
                        sql_params,
                        cancel_events=[context.cancel_event, query_cancel_event],
                        timeout_seconds=self.settings.doris_read_timeout_seconds,
                    ),
                    "cacheHit": bool(getattr(self.doris_repository, "last_cache_hit", False)),
                    "cacheKey": str(getattr(self.doris_repository, "last_cache_key", "") or ""),
                },
                call_id=intent.plan_task_id or "execute_sql",
                target_kind="doris",
                cancel_event=query_cancel_event,
            )
            query_duration_ms = runtime_result.duration_ms or int((time.perf_counter() - query_started) * 1000)
            if context_is_cancelled(context):
                return cancelled_result(intent)
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
                    freshness,
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
                display_rows = annotate_time_window_result_rows(apply_column_masks(rows, contract), intent)
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
                    display_rows = annotate_time_window_result_rows(apply_column_masks(rows, contract), intent)
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
                repair_prompt = pop_node_prompt_trace(context, prompt_trace_key(intent, "repair"))
                if repair_prompt:
                    repair_summary = append_prompt_marker(repair_summary, repair_prompt)
                repair_tool_schema = pop_node_prompt_trace(context, prompt_trace_key(intent, "repair_tool"))
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
            context.runtime_scratch["sql_draft_decision"] = decision
            return intent.sql.strip()
        if intent.sql_strategy == "structured_first":
            structured_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract)
            decision.source = "structured_first"
            decision.reason = "intent.sql_strategy=structured_first"
            context.runtime_scratch["sql_draft_decision"] = decision
            if structured_sql:
                return structured_sql
        if self._use_structured_fast_path(intent, contract, context):
            structured_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract)
            if structured_sql:
                decision.source = "structured_fast_path"
                decision.reason = "low-risk node contract can be compiled deterministically"
                context.runtime_scratch["sql_draft_decision"] = decision
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
                context.runtime_scratch["sql_draft_decision"] = decision
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
                context.runtime_scratch["file_tool_results"] = file_results
            user_payload = self._node_llm_payload(intent, context, contract, file_context)
            try:
                if hasattr(self.llm, "tool_json_chat"):
                    payload = self.llm.tool_json_chat(prompt.system_prompt, user_payload, tool.openai_schema(), {})
                else:
                    payload = self.llm.json_chat(prompt.system_prompt, user_payload, {})
            except Exception as exc:
                self.tool_failure_registry.record_failure("draft_llm_sql", draft_args, "PROVIDER_ERROR", str(exc))
                payload = {}
            set_node_prompt_trace(context, prompt_trace_key(intent, "draft"), prompt.trace())
            set_node_prompt_trace(context, prompt_trace_key(intent, "draft_tool"), tool.trace_schema())
            sql = str(payload.get("sql") or "").strip()
            if sql:
                self.tool_failure_registry.record_success("draft_llm_sql", draft_args)
                decision.source = "llm_plan_bound"
                decision.reason = str(payload.get("reason") or "LLM drafted SQL from node plan contract")
            else:
                error_type = "PROVIDER_ERROR" if self.llm.last_error else "SQL_EMPTY"
                self.tool_failure_registry.record_failure("draft_llm_sql", draft_args, error_type, self.llm.last_error or "LLM returned empty SQL")
        if sql:
            context.runtime_scratch["sql_draft_decision"] = decision
            return sql
        decision.structured_fallback_used = True
        decision.source = "structured_fallback"
        decision.fallback_reason = self.llm.last_error or "LLM unavailable or returned empty SQL"
        decision.reason = "fallback to deterministic single-table SQL builder"
        context.runtime_scratch["sql_draft_decision"] = decision
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
                    "upstreamEntitySets 声明了依赖键和值时，必须严格使用声明的 joinKey/columnValues 过滤；"
                    "selectMustInclude 是强制 SELECT 输出列，必须逐个原样出现在 SELECT 中；"
                    "QueryGraph outputKeys 是传给 dependent 的实体键，必须原样出现在 SELECT 结果中，不能只放在 WHERE/GROUP BY；"
                    "GROUP_AGG/TOPN 必须按 contract 声明的 outputKeys 和 groupByColumn 分组并原样输出；"
                    "timeWindowContract 非空时，必须使用其中的 partitionColumn、tenantColumn、anchorPolicy 和 days 生成过滤。"
                    "相对时间窗锚定 preferredTable 在授权主体过滤后的 MAX(partitionColumn)，不要用 CURDATE()/CURRENT_DATE；"
                    "显式日期且 anchorPolicy=calendar 时才用固定日期 BETWEEN；不要使用 DATE_FORMAT('%Y%m%d')。"
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

    def _prepared_freshness_report(
        self,
        intent: QuestionIntent,
        context: NodeExecutionContext,
    ) -> Optional[FreshnessCheckResult]:
        """Return only a freshness report bound to the current execution node."""

        raw = (context.context_package or {}).get("preparedFreshnessReport")
        if not raw:
            return None
        try:
            report = raw if isinstance(raw, FreshnessCheckResult) else FreshnessCheckResult.model_validate(raw)
        except Exception:
            return None
        if report.task_id and report.task_id != intent.plan_task_id:
            return None
        if report.table == intent.preferred_table:
            return report.model_copy(deep=True)
        if (
            report.status == "STALE_USE_REALTIME_FALLBACK"
            and report.fallback_table == intent.preferred_table
        ):
            return report.model_copy(deep=True)
        return None

    def _check_freshness(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> FreshnessCheckResult:
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        table_metadata = table_asset_metadata(asset_pack, table)
        time_column = str(table_metadata.get("timeColumn") or "")
        merchant_filter_column = str(table_metadata.get("merchantFilterColumn") or "")
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
        if not time_column or time_column not in columns:
            return FreshnessCheckResult(
                task_id=intent.plan_task_id,
                table=table,
                checked=False,
                status="NO_TIME_COLUMN",
                requested_days=int(intent.days or 0),
                reason="table has no declared timeColumn in the semantic asset",
            )
        where = ""
        params: List[Any] = []
        if merchant_filter_column and merchant_filter_column in columns:
            where = " WHERE `%s` = %%s" % merchant_filter_column
            params.append(context.merchant_id)
        sql = "SELECT MIN(`%s`) AS `min_value`, MAX(`%s`) AS `max_value` FROM `%s`%s" % (
            time_column,
            time_column,
            table,
            where,
        )
        try:
            rows = doris_query_with_cancellation(
                self.doris_repository,
                sql,
                params,
                cancel_events=[context.cancel_event],
                timeout_seconds=min(5, int(self.settings.doris_read_timeout_seconds or 5)),
            )
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
        min_pt = str(first.get("min_value") or "")
        max_pt = str(first.get("max_value") or "")
        status = "AVAILABLE" if max_pt else "ZERO_ROWS"
        reason = "max_pt=%s" % max_pt if max_pt else "freshness check returned no partition value"
        return FreshnessCheckResult(
            task_id=intent.plan_task_id,
            table=table,
            checked=True,
            status=status,
            pt_column=time_column,
            requested_days=int(intent.days or 0),
            min_pt=min_pt,
            max_pt=max_pt,
            reason=reason,
        )

    def _apply_partition_date_anchor(self, sql: str, intent: QuestionIntent, freshness: FreshnessCheckResult) -> tuple[str, str]:
        if not bool(getattr(self.settings, "agent_partition_date_anchor_enabled", False)):
            return sql, ""
        if intent.time_range.start_date and intent.time_range.end_date and intent.time_range.anchor_policy == "calendar":
            return sql, ""
        if not freshness.checked or not freshness.max_pt:
            return sql, ""
        anchor = parse_partition_date(freshness.max_pt)
        if not anchor:
            return sql, ""
        anchor_text = anchor.isoformat()
        if not re.search(r"\b(CURDATE\(\)|CURRENT_DATE(?:\(\))?)", str(sql or ""), flags=re.I):
            return sql, ""
        normalized_sql = normalize_inclusive_relative_window_sql(str(sql or ""), intent.days)
        anchored_sql = re.sub(r"\b(CURDATE\(\)|CURRENT_DATE(?:\(\))?)", "'%s'" % anchor_text, normalized_sql, flags=re.I)
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
        if updated_resolution:
            updated_resolution = governed_realtime_metric_resolution(
                updated_resolution,
                intent.preferred_table,
                fallback_table,
                fallback.metadata or {},
                asset_pack,
            )
            if not updated_resolution:
                return None
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
        freshness: FreshnessCheckResult,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self.settings, "agent_doris_split_query_enabled", True)):
            return None
        if error_code not in RESOURCE_CONSTRAINED_DORIS_ERRORS:
            return None
        if intent.answer_mode != AnswerMode.DETAIL:
            return None
        table = intent.preferred_table
        columns = set(asset_pack.known_columns(table))
        table_metadata = table_asset_metadata(asset_pack, table)
        time_column = str(table_metadata.get("timeColumn") or "")
        if not time_column or time_column not in columns or int(intent.days or 0) <= 0:
            return None
        safe_sql = self._draft_structured_sql(intent, asset_pack, context, contract=contract, resource_safe=True)
        if not safe_sql:
            return None
        base_validation = self.validator.validate(safe_sql, asset_pack)
        base_validation = self._node_scope_validation(base_validation, intent, safe_sql, asset_pack)
        base_validation = self._contract_scope_validation(base_validation, intent, safe_sql, contract)
        if not base_validation.valid:
            return None
        chunk_days = max(1, int(getattr(self.settings, "agent_doris_split_chunk_days", 7) or 7))
        max_chunks = max(1, int(getattr(self.settings, "agent_doris_split_max_chunks", 6) or 6))
        max_concurrency = max(1, int(getattr(self.settings, "agent_doris_split_max_concurrency", 3) or 3))
        limit = structured_limit(intent.limit, detail=True, resource_safe=True)
        anchor = parse_partition_date(freshness.max_pt) if freshness.max_pt else None
        if not anchor:
            return None
        split_sqls = split_detail_sql_by_time_windows(
            safe_sql,
            int(intent.days or 0),
            chunk_days,
            max_chunks,
            limit,
            time_column,
            anchor_date=anchor.isoformat() if anchor else "",
        )
        if not split_sqls:
            return None
        anchor_date = ""
        chunk_queries: List[Tuple[str, List[Any]]] = []
        for split_sql in split_sqls:
            split_validation = self.validator.validate(split_sql, asset_pack)
            split_validation = self._node_scope_validation(split_validation, intent, split_sql, asset_pack)
            split_validation = self._contract_scope_validation(split_validation, intent, split_sql, contract)
            if not split_validation.valid:
                return None
            safe_bound_sql, safe_params, binding_error = bind_node_sql_parameters(split_sql, intent, asset_pack, context)
            tenant_binding_error = tenant_scope_binding_error(safe_bound_sql, safe_params, contract, context)
            if tenant_binding_error:
                binding_error = binding_error or tenant_binding_error
            if binding_error:
                return None
            anchored_sql, split_anchor_date = self._apply_partition_date_anchor(safe_bound_sql, intent, freshness)
            access_decision = self.access_control.authorize_contract(contract, anchored_sql, run_id=context.sub_agent_run_id)
            if not access_decision.allowed:
                return None
            chunk_queries.append((anchored_sql, safe_params))
            anchor_date = anchor_date or split_anchor_date
        if not chunk_queries:
            return None
        events: List[Dict[str, Any]] = [
            {
                "event": "split_query_fallback_started",
                "sourceErrorCode": error_code,
                "sourceError": str(error_text or "")[:240],
                "originalDurationMs": previous_duration_ms,
                "chunkDays": chunk_days,
                "maxChunks": max_chunks,
                "chunkCount": len(chunk_queries),
                "maxConcurrency": min(max_concurrency, len(chunk_queries)),
                "executionMode": "parallel_chunks",
                "limit": limit,
                "anchorPartitionDate": anchor_date,
            }
        ]
        cache_hit = False
        cache_keys: List[str] = []
        started = time.perf_counter()
        chunk_results: Dict[int, List[Dict[str, Any]]] = {}
        split_cancel_event = Event()

        def query_chunk(index: int, chunk_sql: str, chunk_params: List[Any]) -> Dict[str, Any]:
            chunk_started = time.perf_counter()
            chunk_rows = doris_query_with_cancellation(
                self.doris_repository,
                chunk_sql,
                chunk_params,
                cancel_events=[context.cancel_event, split_cancel_event],
                timeout_seconds=self.settings.doris_read_timeout_seconds,
            )
            return {
                "chunkIndex": index,
                "rows": list(chunk_rows or []),
                "cacheHit": bool(getattr(self.doris_repository, "last_cache_hit", False)),
                "cacheKey": str(getattr(self.doris_repository, "last_cache_key", "") or ""),
                "durationMs": int((time.perf_counter() - chunk_started) * 1000),
                "sql": trim_sql(chunk_sql, 600),
            }

        executor = ThreadPoolExecutor(max_workers=min(max_concurrency, len(chunk_queries)))
        try:
            futures = {
                submit_with_current_context(executor, query_chunk, index, chunk_sql, chunk_params): (index, chunk_sql)
                for index, (chunk_sql, chunk_params) in enumerate(chunk_queries, start=1)
            }
            try:
                for future in as_completed(
                    futures,
                    timeout=max(1, int(self.settings.doris_read_timeout_seconds or 1)),
                ):
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
            except TimeoutError:
                split_cancel_event.set()
                events.append(
                    {
                        "event": "split_query_timeout",
                        "timeoutSeconds": self.settings.doris_read_timeout_seconds,
                    }
                )
            finally:
                for future in futures:
                    if not future.done():
                        future.cancel()
        finally:
            split_cancel_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
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
        where = self._structured_where(
            intent,
            table,
            columns,
            context,
            contract,
            entity_value_limit=50 if resource_safe else 200,
        )
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
        metric_parts = structured_metric_select_parts(intent, table, columns, contract.metric_specs)
        if metric_parts is None:
            return ""
        if not metric_parts:
            return ""
        for index, (metric_expr, metric_alias) in enumerate(metric_parts):
            select_parts.append("%s AS `%s`" % (metric_expr, metric_alias))
            if index == 0:
                order_expr = "`%s` DESC" % metric_alias
        resolution = intent.metric_resolution or {}
        if (
            str(resolution.get("displayRole") or resolution.get("display_role") or "").lower() == "trend_context"
            and intent.group_by_column
            and intent.group_by_column in group_columns
        ):
            order_expr = "%s ASC" % quote_identifier(intent.group_by_column)
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
            return preferred[:12] or sorted(visible_columns)[:8]
        return preferred[:24] or sorted(visible_columns)[:16]

    def _structured_group_columns(self, intent: QuestionIntent, columns: set, contract: NodePlanContract) -> List[str]:
        visible_columns = set(contract.visible_columns or [])
        group_columns: List[str] = []
        allowed_output_keys = self._aggregate_output_group_keys(intent)
        for column in [intent.group_by_column] + allowed_output_keys:
            if column and column in columns and column in visible_columns and column not in group_columns:
                group_columns.append(column)
        if intent.task_role == TaskRole.DEPENDENT:
            return group_columns[:10]
        return group_columns[:6]

    def _structured_context_columns(self, intent: QuestionIntent, columns: set, group_columns: List[str], contract: NodePlanContract) -> List[str]:
        if intent.task_role != TaskRole.DEPENDENT:
            return []
        visible_columns = set(contract.visible_columns or [])
        blocked = set(group_columns) | {intent.metric_column}
        context_columns: List[str] = []
        for column in intent.output_keys:
            if column in blocked or column not in columns or column not in visible_columns or column in context_columns:
                continue
            context_columns.append(column)
        return context_columns[:8]

    def _aggregate_output_group_keys(self, intent: QuestionIntent) -> List[str]:
        keys = []
        for column in intent.output_keys:
            if not aggregate_group_key_allowed(intent, column):
                continue
            keys.append(column)
        return keys

    def _structured_where(
        self,
        intent: QuestionIntent,
        table: str,
        columns: set,
        context: NodeExecutionContext,
        contract: NodePlanContract,
        entity_value_limit: int = 200,
    ) -> List[str]:
        where: List[str] = []
        merchant_column = str(contract.merchant_filter_column or "")
        if merchant_column and merchant_column in columns:
            where.append("`%s` = %s" % (merchant_column, sql_literal(context.merchant_id)))
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
        time_contract = dict(contract.time_window_contract or {})
        partition_column = str(time_contract.get("partitionColumn") or "")
        days = int(time_contract.get("days") or intent.days or getattr(intent.time_range, "days", 0) or 0)
        if (
            partition_column in columns
            and intent.time_range.start_date
            and intent.time_range.end_date
            and intent.time_range.anchor_policy == CALENDAR_ANCHOR_POLICY
        ):
            if not any(sql_references_filter_column(predicate, partition_column) for predicate in where):
                where.append(
                    "`%s` BETWEEN %s AND %s"
                    % (partition_column, sql_literal(intent.time_range.start_date), sql_literal(intent.time_range.end_date))
                )
        elif partition_column in columns and days > 0:
            if not any(sql_references_filter_column(predicate, partition_column) for predicate in where):
                where.append(
                    latest_partition_window_predicate(
                        table,
                        days,
                        partition_column=partition_column,
                        tenant_column=str(time_contract.get("tenantColumn") or merchant_column),
                        tenant_value_sql=sql_literal(context.merchant_id) if (time_contract.get("tenantColumn") or merchant_column) else "",
                        offset_days=int(getattr(intent.time_range, "offset_days", 0) or 0),
                    )
                )
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
                "rules": "只修复当前 SQL；保持单表、契约声明的租户/时间/实体过滤和 LIMIT；不要新增字段或表。",
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
        set_node_prompt_trace(context, prompt_trace_key(intent, "repair"), prompt.trace())
        set_node_prompt_trace(context, prompt_trace_key(intent, "repair_tool"), tool.trace_schema())
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
        if contract.merchant_filter_column and not has_merchant_filter_predicate(sql, {contract.merchant_filter_column}):
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "MISSING_MERCHANT_FILTER",
                    "message": "Node SQL must filter by nodePlanContract.merchantFilterColumn",
                }
            )
        time_contract = dict(contract.time_window_contract or {})
        partition_column = str(time_contract.get("partitionColumn") or "")
        if partition_column and not sql_filters_column(sql, partition_column):
            return validation.model_copy(
                update={
                    "valid": False,
                    "error_code": "MISSING_PARTITION_FILTER",
                    "message": "Node SQL must filter the partitionColumn declared by timeWindowContract",
                }
            )
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

    def _node_plan_contract(
        self,
        intent: QuestionIntent,
        asset_pack: PlanningAssetPack,
        context: NodeExecutionContext,
    ) -> NodePlanContract:
        table = intent.preferred_table
        table_columns = asset_pack.known_columns(table)
        metric_contract = compile_node_metric_contract(intent, asset_pack, set(table_columns))
        required_columns = self._node_required_columns(intent, asset_pack).get(table, [])
        allowed_columns = list(required_columns)
        contract_output_keys = self._node_contract_output_keys(intent, set(table_columns))
        for column in formula_columns(str(metric_contract.get("formula") or intent.metric_formula), set(table_columns)):
            if column not in allowed_columns:
                allowed_columns.append(column)
        effective_metric_specs = metric_specs_for_contract(intent, table, metric_contract, asset_pack, set(table_columns))
        for spec in effective_metric_specs:
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
        row_scope_policy = normalize_row_access_policy(table_metadata.get("rowAccessPolicy") or default_row_access_policy(merchant_filter_column))
        region_filter_column = str(table_metadata.get("regionFilterColumn") or "")
        store_filter_column = str(table_metadata.get("storeFilterColumn") or "")
        time_column = str(table_metadata.get("timeColumn") or "")
        execution_scope_columns = [merchant_filter_column, region_filter_column, store_filter_column]
        if time_column in table_columns and (intent.days or intent.time_range.start_date or intent.time_range.end_date):
            execution_scope_columns.append(time_column)
        for column in execution_scope_columns:
            if column and column in table_columns and column not in allowed_columns:
                allowed_columns.append(column)
        access_role = str(context.access_role or DEFAULT_ACCESS_ROLE)
        field_semantics = table_field_semantics(asset_pack, table)
        aggregate_result_aliases = {
            str(spec.get("metricName") or spec.get("metric_name") or "")
            for spec in effective_metric_specs
            if str(spec.get("metricName") or spec.get("metric_name") or "")
        }
        aggregate_result_aliases.update(
            str(value)
            for value in [
                metric_contract.get("metricKey"),
                intent.metric_name,
                (metric_contract.get("resolution") or {}).get("metricKey"),
            ]
            if str(value or "")
        )
        aggregate_output_columns = set(contract_output_keys)
        if intent.group_by_column:
            aggregate_output_columns.add(intent.group_by_column)
        visible_columns: List[str] = []
        internal_only_columns: List[str] = []
        column_access_policy: Dict[str, Dict[str, Any]] = {}
        column_display_policy: Dict[str, Dict[str, Any]] = {}
        masked_columns: Dict[str, str] = {}
        protected_internal = {merchant_filter_column, str(row_scope_policy.get("filterColumn") or "")} - {""}
        for column in allowed_columns:
            semantic = field_semantics.get(column, {})
            result_role = ""
            if intent.answer_mode in {AnswerMode.TOPN, AnswerMode.GROUP_AGG, AnswerMode.METRIC}:
                if column in aggregate_result_aliases:
                    result_role = "METRIC"
                elif column in aggregate_output_columns:
                    result_role = (
                        "TIME"
                        if column == time_column
                        else str(semantic.get("semanticRole") or semantic.get("role") or "").upper()
                    )
            result_policy = declared_result_access_policy(table_metadata, result_role)
            visibility_policy = normalize_visibility_policy(
                (result_policy.get("visibilityPolicy") if result_policy else None)
                or semantic.get("visibilityPolicy")
                or {}
            )
            masking_policy = normalize_masking_policy(
                (result_policy.get("maskingPolicy") if result_policy else None)
                or semantic.get("maskingPolicy")
                or {}
            )
            policy = {
                **visibility_policy,
                "maskingStrategy": masking_policy.get("strategy") or "none",
                "maskingReason": masking_policy.get("reason") or "",
            }
            column_access_policy[column] = policy
            column_display_policy[column] = normalize_column_display_policy(result_policy or semantic)
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
            metric_name=str(metric_contract.get("metricKey") or intent.metric_name),
            metric_formula=str(metric_contract.get("formula") or intent.metric_formula),
            metric_specs=effective_metric_specs,
            group_by_column=intent.group_by_column,
            output_keys=contract_output_keys,
            required_evidence=intent.required_evidence,
            days=int(intent.days or 0),
            limit=int(intent.limit or 0),
            merchant_id=context.merchant_id,
            merchant_filter_column=merchant_filter_column,
            effective_user_id=context.effective_user_id,
            authorized_region=context.authorized_region,
            authorized_store_ids=list(context.authorized_store_ids),
            region_filter_column=region_filter_column,
            store_filter_column=store_filter_column,
            access_role=access_role,
            row_scope_policy=row_scope_policy,
            column_access_policy=column_access_policy,
            column_display_policy=column_display_policy,
            masked_columns=masked_columns,
            answer_mode=enum_text(intent.answer_mode),
            task_role=enum_text(intent.task_role),
            sql_strategy=intent.sql_strategy or "llm_plan_bound_first",
            upstream_entity_sets=[item.model_dump(by_alias=True) for item in context.upstream_entity_sets],
            metric_resolution=dict(metric_contract.get("resolution") or intent.metric_resolution),
            metric_governance_mode=str(metric_contract.get("mode") or "legacy_unsealed"),
            time_window_contract=self._intent_time_window_contract(
                intent,
                table,
                table_columns,
                merchant_filter_column,
                time_column,
            ),
        )

    def _intent_time_window_contract(
        self,
        intent: QuestionIntent,
        table: str,
        table_columns: List[str],
        merchant_filter_column: str,
        time_column: str,
    ) -> Dict[str, Any]:
        if time_column not in set(table_columns) or not (intent.days or intent.time_range.start_date or intent.time_range.end_date):
            return {}
        if intent.time_range.start_date or intent.time_range.end_date:
            contract = time_window_contract_payload(intent.time_range, table, time_column, merchant_filter_column)
            if contract.get("anchorPolicy") != CALENDAR_ANCHOR_POLICY:
                contract["anchorPolicy"] = LATEST_PARTITION_ANCHOR_POLICY
                contract["executionRule"] = "relative windows must anchor to MAX(partitionColumn) after tenant filter"
            if intent.days and not contract.get("days"):
                contract["days"] = int(intent.days or 0)
            return contract
        days = int(intent.days or 0)
        return {
            "kind": "rolling",
            "label": "最近%d天" % days,
            "days": days,
            "anchorPolicy": LATEST_PARTITION_ANCHOR_POLICY,
            "partitionColumn": time_column,
            "table": table,
            "tenantColumn": merchant_filter_column,
            "executionRule": "relative windows must anchor to MAX(partitionColumn) after tenant filter",
        }

    def _enforce_identity_scope_sql(
        self,
        sql: str,
        contract: NodePlanContract,
        context: NodeExecutionContext,
    ) -> str:
        scoped_sql = str(sql or "")
        context.context_package["regionFilterColumn"] = contract.region_filter_column
        context.context_package["storeFilterColumn"] = contract.store_filter_column
        conditions: List[str] = []
        if contract.region_filter_column and context.authorized_region and not sql_references_filter_column(scoped_sql, contract.region_filter_column):
            conditions.append("%s = %s" % (quote_identifier(contract.region_filter_column), sql_literal(context.authorized_region)))
        if contract.store_filter_column and context.authorized_store_ids and not sql_references_filter_column(scoped_sql, contract.store_filter_column):
            values = ", ".join(sql_literal(item) for item in context.authorized_store_ids[:200])
            conditions.append("%s IN (%s)" % (quote_identifier(contract.store_filter_column), values))
        for condition in conditions:
            scoped_sql = add_sql_where_condition(scoped_sql, condition)
        return scoped_sql

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
        table_asset = next((item for item in asset_pack.tables if item.table == table), None)
        metadata = dict(getattr(table_asset, "metadata", {}) or {})
        semantic_defaults = [
            str(metadata.get("merchantFilterColumn") or ""),
            str(metadata.get("timeColumn") or ""),
        ]
        for item in semantic_defaults:
            if item in columns and item not in requested:
                requested.append(item)
        if not requested:
            requested = asset_pack.known_columns(table)[:16]
        return {table: requested[:32]} if table else {}

    def _node_access_hints(self, intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Dict[str, Dict[str, Any]]:
        hints: Dict[str, Dict[str, Any]] = {}
        for table in self._node_table_names(intent, asset_pack):
            columns = set(asset_pack.known_columns(table))
            hints[table] = semantic_table_access_hint(asset_pack, table, columns)
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


def merge_query_bundles(
    bundles: List[QueryBundle],
    source_ids: Optional[List[str]] = None,
    artifact_store: Optional[WorkspaceArtifactStore] = None,
) -> QueryBundle:
    rows: List[Dict[str, Any]] = []
    tables: List[str] = []
    first_error = ""
    source_ids = source_ids or ["node_%s" % (index + 1) for index in range(len(bundles))]
    source_row_counts: Dict[str, int] = {}
    source_artifact_refs: Dict[str, List[str]] = {}
    offloaded_files: List[str] = []
    incomplete_sources: List[str] = []
    for index, bundle in enumerate(bundles):
        source_id = source_ids[index] if index < len(source_ids) else "node_%s" % (index + 1)
        source_row_counts[source_id] = bundle.effective_row_count()
        source_artifact_refs[source_id] = list(bundle.offloaded_files or [])
        offloaded_files.extend(path for path in bundle.offloaded_files or [] if path not in offloaded_files)
        for table in bundle.tables:
            if table not in tables:
                tables.append(table)
        if bundle.failed and not first_error:
            first_error = bundle.error or bundle.summary
        if not bundle.failed:
            complete_rows, complete = query_bundle_complete_rows(bundle)
            if not complete:
                incomplete_sources.append(source_id)
                continue
            rows.extend(complete_rows)
    if incomplete_sources:
        return incomplete_merge_bundle(
            tables,
            bundles,
            first_error,
            source_row_counts,
            source_artifact_refs,
            offloaded_files,
            incomplete_sources,
        )
    merged_artifact = write_merged_rows_artifact(artifact_store, rows, source_ids, "concatenated")
    if merged_artifact and merged_artifact not in offloaded_files:
        offloaded_files.append(merged_artifact)
    return QueryBundle(
        tables=tables,
        rows=rows[:200],
        original_row_count=len(rows),
        offloaded_files=offloaded_files,
        source_row_counts=source_row_counts,
        source_artifact_refs=source_artifact_refs,
        failed=bool(bundles) and not any(not item.failed for item in bundles),
        error=first_error,
        summary="合并 %s 个 NodeWorker 结果" % len(bundles),
        duration_ms=sum(int(bundle.duration_ms or 0) for bundle in bundles),
        cache_hit=any(bundle.cache_hit for bundle in bundles),
    )


def annotate_time_window_result_rows(rows: List[Dict[str, Any]], intent: QuestionIntent) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    time_range = intent.time_range
    role = str(getattr(time_range, "window_role", "") or (intent.metric_resolution or {}).get("timeWindowRole") or "").strip()
    label = str(getattr(time_range, "label", "") or "").strip()
    if not role and not label:
        return rows
    annotated: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if role:
            item["__timeWindowRole"] = role
        if label:
            item["__timeWindowLabel"] = label
        comparison_type = str(getattr(time_range, "comparison_type", "") or "").strip()
        if comparison_type:
            item["__timeWindowComparisonType"] = comparison_type
        annotated.append(item)
    return annotated


def merge_task_result_bundles(
    task_results: List[AgentTaskResult],
    artifact_store: Optional[WorkspaceArtifactStore] = None,
) -> QueryBundle:
    bundles = [item.query_bundle for item in task_results]
    source_ids = [item.task_id or item.node_task_profile.task_id or "node_%s" % (index + 1) for index, item in enumerate(task_results)]
    merge_keys = choose_merge_entity_keys(task_results)
    if not merge_keys:
        return merge_query_bundles(bundles, source_ids, artifact_store)
    tables = merged_bundle_tables(bundles)
    first_error = first_bundle_error(bundles)
    source_row_counts = {source_id: bundle.effective_row_count() for source_id, bundle in zip(source_ids, bundles)}
    source_artifact_refs = {source_id: list(bundle.offloaded_files or []) for source_id, bundle in zip(source_ids, bundles)}
    offloaded_files = dedupe_strings(path for bundle in bundles for path in bundle.offloaded_files or [])
    complete_rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    incomplete_sources: List[str] = []
    for source_id, bundle in zip(source_ids, bundles):
        if bundle.failed:
            continue
        complete_rows, complete = query_bundle_complete_rows(bundle)
        if not complete:
            incomplete_sources.append(source_id)
        else:
            complete_rows_by_task[source_id] = complete_rows
    if incomplete_sources:
        return incomplete_merge_bundle(
            tables,
            bundles,
            first_error,
            source_row_counts,
            source_artifact_refs,
            offloaded_files,
            incomplete_sources,
        )
    merged_by_key: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    scalar_rows: List[Tuple[str, Dict[str, Any]]] = []
    for task_result in task_results:
        bundle = task_result.query_bundle
        if bundle.failed:
            continue
        task_id = task_result.task_id or task_result.node_task_profile.task_id or "node"
        task_rows = complete_rows_by_task.get(task_id, list(bundle.rows or []))
        for row in task_rows:
            if any(key not in row or blank_entity_value(row.get(key)) for key in merge_keys):
                if len(task_rows) == 1:
                    scalar_rows.append((task_id, row))
                continue
            key_value = tuple(row.get(key) for key in merge_keys)
            if key_value not in merged_by_key:
                merged_by_key[key_value] = {key: row.get(key) for key in merge_keys}
                order.append(key_value)
            merge_row_fields(merged_by_key[key_value], row, task_id, merge_keys)
    merged_rows = [merged_by_key[key] for key in order]
    if scalar_rows and merged_rows:
        for task_id, row in scalar_rows:
            for merged in merged_rows:
                merge_row_fields(merged, row, task_id, merge_keys)
    elif scalar_rows:
        for task_id, row in scalar_rows:
            target: Dict[str, Any] = {}
            merge_row_fields(target, row, task_id, merge_keys)
            if target:
                merged_rows.append(target)
    if not merged_rows:
        return merge_query_bundles(bundles, source_ids, artifact_store)
    merged_artifact = write_merged_rows_artifact(artifact_store, merged_rows, source_ids, "entity_joined")
    if merged_artifact and merged_artifact not in offloaded_files:
        offloaded_files.append(merged_artifact)
    return QueryBundle(
        tables=tables,
        rows=merged_rows[:200],
        original_row_count=len(merged_rows),
        offloaded_files=offloaded_files,
        source_row_counts=source_row_counts,
        source_artifact_refs=source_artifact_refs,
        failed=bool(bundles) and not any(not item.failed for item in bundles),
        error=first_error,
        summary="按实体键 %s 合并 %s 个 NodeWorker 结果" % ("+".join(merge_keys), len(bundles)),
        duration_ms=sum(int(bundle.duration_ms or 0) for bundle in bundles),
        cache_hit=any(bundle.cache_hit for bundle in bundles),
    )


def query_bundle_complete_rows(bundle: QueryBundle, max_artifact_bytes: int = 20_000_000) -> Tuple[List[Dict[str, Any]], bool]:
    preview_rows = [row for row in bundle.rows or [] if isinstance(row, dict)]
    expected = max(0, int(bundle.original_row_count or 0))
    if not expected or len(preview_rows) >= expected:
        return preview_rows[:expected] if expected else preview_rows, True
    for raw_path in bundle.offloaded_files or []:
        path = Path(str(raw_path or ""))
        if not path.name.endswith("_rows.json"):
            continue
        try:
            if not path.is_file() or path.stat().st_size > max_artifact_bytes:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        artifact_rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        if len(artifact_rows) >= expected:
            return artifact_rows[:expected], True
    return preview_rows, False


def write_merged_rows_artifact(
    artifact_store: Optional[WorkspaceArtifactStore],
    rows: List[Dict[str, Any]],
    source_ids: List[str],
    kind: str,
) -> str:
    if artifact_store is None or len(rows) <= 200:
        return ""
    fingerprint = hashlib.sha256(
        json.dumps({"sources": source_ids, "kind": kind, "rowCount": len(rows)}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    artifact = artifact_store.write_json(
        "sql_results",
        "merged_%s_%s_rows.json" % (kind, fingerprint),
        rows,
        preview_chars=0,
    )
    return str(artifact.get("path") or "")


def incomplete_merge_bundle(
    tables: List[str],
    bundles: List[QueryBundle],
    first_error: str,
    source_row_counts: Dict[str, int],
    source_artifact_refs: Dict[str, List[str]],
    offloaded_files: List[str],
    incomplete_sources: List[str],
) -> QueryBundle:
    message = "MERGE_INPUT_INCOMPLETE: full rows unavailable for %s" % ",".join(incomplete_sources)
    return QueryBundle(
        tables=tables,
        rows=[],
        original_row_count=sum(source_row_counts.values()),
        offloaded_files=offloaded_files,
        lineage_complete=False,
        source_row_counts=source_row_counts,
        source_artifact_refs=source_artifact_refs,
        failed=True,
        error=first_error or message,
        summary=message,
        duration_ms=sum(int(bundle.duration_ms or 0) for bundle in bundles),
        cache_hit=any(bundle.cache_hit for bundle in bundles),
        runtime_events=[{"code": "MERGE_INPUT_INCOMPLETE", "sourceTaskIds": incomplete_sources}],
    )


def choose_merge_entity_key(task_results: List[AgentTaskResult]) -> str:
    keys = choose_merge_entity_keys(task_results)
    return keys[0] if keys else ""


def choose_merge_entity_keys(task_results: List[AgentTaskResult]) -> List[str]:
    candidates: List[List[str]] = []
    for task_result in task_results:
        bundle = task_result.query_bundle
        if bundle.failed or not bundle.rows or not task_result.entity_set:
            continue
        declared = dedupe_strings(
            [
                task_result.entity_set.join_key,
                *task_result.entity_set.column_values.keys(),
            ]
        )
        row_keys = set.intersection(*(set(row.keys()) for row in bundle.rows)) if bundle.rows else set()
        candidates.append([key for key in declared if key in row_keys])
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[0][:1]
    common = set(candidates[0])
    for declared in candidates[1:]:
        common &= set(declared)
    return [key for key in candidates[0] if key in common]


def merge_row_fields(target: Dict[str, Any], row: Dict[str, Any], task_id: str, merge_keys: List[str]) -> None:
    lineage = target.setdefault("__fieldLineage", {})
    conflicts = target.setdefault("__fieldConflicts", {})
    for key, value in row.items():
        if str(key).startswith("__"):
            continue
        if key in merge_keys:
            continue
        owners = lineage.setdefault(str(key), [])
        if task_id not in owners:
            owners.append(task_id)
        if key not in target or blank_entity_value(target.get(key)):
            target[key] = value
            continue
        if target.get(key) == value or blank_entity_value(value):
            continue
        namespaced = "%s__%s" % (task_id, key)
        if namespaced not in target:
            target[namespaced] = value
        conflict_values = conflicts.setdefault(str(key), [])
        existing = {str(item.get("taskId") or ""): item for item in conflict_values if isinstance(item, dict)}
        first_owner = next((owner for owner in owners if owner != task_id), "")
        if first_owner and first_owner not in existing:
            conflict_values.append({"taskId": first_owner, "value": target.get(key)})
        if task_id not in existing:
            conflict_values.append({"taskId": task_id, "value": value})


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


def optimize_query_plan_for_execution(plan: QueryPlan, asset_pack: PlanningAssetPack) -> QueryPlan:
    """Return a normalized execution graph without mutating the planned graph."""

    optimized = (plan or QueryPlan()).model_copy(deep=True)
    _optimize_query_plan_for_execution_in_place(optimized, asset_pack)
    return optimized


def _optimize_query_plan_for_execution_in_place(plan: QueryPlan, asset_pack: PlanningAssetPack) -> None:
    """Merge structurally equivalent metric nodes on an isolated graph copy."""

    if len(plan.intents) < 2:
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


def prepare_execution_graph(
    question: str,
    plan: QueryPlan,
    asset_pack: PlanningAssetPack,
    graph_validator: Any = None,
    memory_constraints: Optional[List[Dict[str, Any]]] = None,
) -> ExecutionGraphPreparation:
    """Normalize a copy and validate the exact graph that NodeWorker may execute."""

    source_fingerprint = query_graph_fingerprint(plan)
    normalized = optimize_query_plan_for_execution(plan, asset_pack)
    execution_fingerprint = query_graph_fingerprint(normalized)
    validator = graph_validator or QueryGraphValidator()
    facade = validator if isinstance(validator, GraphContractValidator) else GraphContractValidator(validator)
    validation = facade.validate(
        question,
        normalized,
        asset_pack,
        memory_constraints or [],
    )
    validation = execution_graph_validation_with_question_coverage(
        question,
        normalized,
        asset_pack,
        validation,
    )
    optimization_notes = tuple(
        str(item)
        for item in normalized.compiler_trace
        if str(item).startswith("execution_optimizer.")
    )
    wrapped_validator = getattr(facade, "validator", validator)
    return ExecutionGraphPreparation(
        plan=normalized,
        validation=validation,
        source_plan_fingerprint=source_fingerprint,
        execution_plan_fingerprint=execution_fingerprint,
        question_fingerprint=execution_question_fingerprint(question),
        asset_pack_fingerprint=execution_asset_pack_fingerprint(asset_pack),
        changed=source_fingerprint != execution_fingerprint,
        optimization_notes=optimization_notes,
        validator_name=type(wrapped_validator).__name__,
    )


def execution_graph_validation_with_question_coverage(
    question: str,
    plan: QueryPlan,
    asset_pack: PlanningAssetPack,
    validation: GraphValidationResult,
) -> GraphValidationResult:
    """Apply the same question-coverage gate used by the workflow validator."""

    coverage_gaps = query_plan_question_coverage_gaps(question, plan, asset_pack)
    if not coverage_gaps:
        return validation
    gaps = list(validation.gaps or [])
    seen = {(gap.code, gap.task_id, gap.evidence) for gap in gaps}
    for gap in coverage_gaps:
        identity = (gap.code, gap.task_id, gap.evidence)
        if identity in seen:
            continue
        gaps.append(gap)
        seen.add(identity)
    return validation.model_copy(
        update={
            "valid": False,
            "gaps": gaps,
            "repairable": False,
        }
    )


def require_normalized_execution_plan(
    plan: QueryPlan,
    asset_pack: PlanningAssetPack,
    question: str,
    *,
    execution_preparation: Optional[ExecutionGraphPreparation] = None,
) -> QueryPlan:
    """Fail before worker dispatch if normalization or preparation is stale.

    The optional typed preparation is the strict production hand-off: it proves
    that the normalized graph passed validation.  The raw-plan path remains for
    low-level worker tests, but it may only receive an already-normalized graph and
    never performs the former in-place optimization.
    """

    candidate = plan
    if execution_preparation is not None:
        if execution_preparation.question_fingerprint != execution_question_fingerprint(question):
            raise ExecutionGraphPreparationRequired("execution preparation belongs to a different question")
        if execution_preparation.asset_pack_fingerprint != execution_asset_pack_fingerprint(asset_pack):
            raise ExecutionGraphPreparationRequired("execution preparation belongs to a different semantic asset pack")
        candidate = execution_preparation.require_executable()
        if query_graph_fingerprint(plan) != execution_preparation.execution_plan_fingerprint:
            raise ExecutionGraphPreparationRequired("supplied plan does not match the prepared execution graph")
        if query_graph_fingerprint(candidate) != execution_preparation.execution_plan_fingerprint:
            raise ExecutionGraphPreparationRequired("prepared execution graph changed after validation")

    normalized_again = optimize_query_plan_for_execution(candidate, asset_pack)
    if query_graph_fingerprint(normalized_again) != query_graph_fingerprint(candidate):
        raise ExecutionGraphPreparationRequired(
            "QueryGraph must be normalized and validated before NodeWorker.execute_plan"
        )
    return candidate


def node_execution_contract_hash(
    intent: QuestionIntent,
    plan: QueryPlan,
    merchant_id: str,
    access_role: str,
    user_scope: Dict[str, Any],
    asset_pack: Optional[PlanningAssetPack] = None,
) -> str:
    """Bind resumable node results to the exact execution contract, not only task id."""

    dependencies = [
        item.model_dump(by_alias=True)
        for item in plan.dependencies
        if item.anchor_task_id == intent.plan_task_id or item.dependent_task_id == intent.plan_task_id
    ]
    dependencies.sort(key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    node_contract = {
        "intent": intent.model_dump(by_alias=True),
        "dependencies": dependencies,
    }
    scope = {
        "merchantId": str(merchant_id or ""),
        "accessRole": str(access_role or ""),
        "userScope": user_scope or {},
    }
    asset_versions = {}
    if asset_pack is not None:
        relevant_tables = {intent.preferred_table, *[str(item) for item in getattr(intent, "candidate_tables", []) or []]}
        asset_versions = {
            table: version.model_dump(by_alias=True)
            for table, version in asset_pack.semantic_catalog_version.items()
            if table in relevant_tables
        }
    payload = {
        "planFingerprint": hashlib.sha256(
            json.dumps(plan.model_dump(by_alias=True), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "nodeContractHash": hashlib.sha256(
            json.dumps(node_contract, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "scope": scope,
        "timeRange": {
            "days": int(intent.days or 0),
            "filterColumn": intent.filter_column,
            "filterValue": intent.filter_value,
        },
        "assetVersion": asset_versions,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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
        str(getattr(intent.time_range, "window_role", "") or ""),
        int(getattr(intent.time_range, "offset_days", 0) or 0),
        str(getattr(intent.time_range, "start_date", "") or ""),
        str(getattr(intent.time_range, "end_date", "") or ""),
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


def compile_node_metric_contract(
    intent: QuestionIntent,
    asset_pack: PlanningAssetPack,
    table_columns: Set[str],
) -> Dict[str, Any]:
    resolution = dict(intent.metric_resolution or {})
    declared_mode = str(
        resolution.get("metricGovernanceMode")
        or resolution.get("metric_governance_mode")
        or ""
    )
    if resolution:
        semantic_ref_id = str(resolution.get("semanticRefId") or "")
        claims_local_compilation = bool(
            declared_mode == "compiled_local"
            or semantic_ref_id.startswith("semantic:compiled_local:")
        )
        if claims_local_compilation:
            normalized = seal_local_execution_resolution(resolution, intent, table_columns)
            if not normalized:
                return {"mode": "legacy_unsealed", "resolution": resolution}
            return {
                "mode": "compiled_local",
                "metricKey": str(normalized.get("metricKey") or intent.metric_name),
                "formula": str(normalized.get("formula") or intent.metric_formula),
                "sourceColumns": list(normalized.get("sourceColumns") or []),
                "resolution": normalized,
            }
        claims_published_semantics = bool(
            semantic_ref_id.startswith("semantic:")
            or resolution.get("semanticContract")
            or resolution.get("semanticContractHash")
            or str(resolution.get("governanceStatus") or "").lower() == "published"
            or declared_mode == "published_semantic"
        )
        if claims_published_semantics:
            return {
                "mode": "published_semantic",
                "metricKey": str(resolution.get("metricKey") or intent.metric_name),
                "formula": str(resolution.get("formula") or intent.metric_formula),
                "sourceColumns": list(resolution.get("sourceColumns") or []),
                "resolution": resolution,
            }
        if local_resolution_complete(resolution, intent.preferred_table, table_columns):
            normalized = seal_local_execution_resolution(resolution, intent, table_columns)
            return {
                "mode": "compiled_local",
                "metricKey": str(normalized.get("metricKey") or intent.metric_name),
                "formula": str(normalized.get("formula") or intent.metric_formula),
                "sourceColumns": list(normalized.get("sourceColumns") or []),
                "resolution": normalized,
            }
        return {"mode": "legacy_unsealed", "resolution": resolution}

    metric = local_metric_entry(intent, asset_pack)
    metadata = dict(getattr(metric, "metadata", {}) or {}) if metric else {}
    metric_key = str(
        intent.metric_name
        or getattr(metric, "key", "")
        or intent.metric_column
        or ""
    )
    formula = str(
        intent.metric_formula
        or metadata.get("formula")
        or metadata.get("metricFormula")
        or ""
    ).strip()
    aggregation_source = "explicit_formula" if intent.metric_formula else ("asset_formula" if formula else "")
    source_columns = dedupe_strings(
        [
            str(item)
            for item in metadata.get("sourceColumns")
            or metadata.get("source_columns")
            or getattr(metric, "columns", [])
            or []
            if str(item) in table_columns
        ]
    )
    if formula:
        source_columns = dedupe_strings(source_columns + formula_columns(formula, table_columns))
    if not formula and intent.metric_column in table_columns:
        aggregation = normalized_local_aggregation(metadata.get("aggregation") or metadata.get("agg"))
        if aggregation:
            formula = local_aggregation_formula(aggregation, intent.metric_column)
            aggregation_source = "asset_aggregation"
        source_columns = [intent.metric_column] if formula else []
    if not metric_key or not formula or not source_columns or not compile_metric_formula(formula, table_columns):
        return {"mode": "legacy_unsealed", "resolution": {}}
    compiled = {
        "semanticRefId": "semantic:compiled_local:%s:metric:%s" % (intent.preferred_table, metric_key),
        "metricKey": metric_key,
        "ownerTable": intent.preferred_table,
        "formula": formula,
        "sourceColumns": source_columns,
        "metricGovernanceMode": "compiled_local",
        "contractProvenance": {
            "kind": "execution_contract",
            "ownerTable": intent.preferred_table,
            "metricKey": metric_key,
            "taskId": intent.plan_task_id,
        },
        "resolutionSource": "compiled_local",
        "localCompilationPolicy": aggregation_source,
    }
    compiled = seal_semantic_metric_resolution(compiled, force=True)
    return {
        "mode": "compiled_local",
        "metricKey": metric_key,
        "formula": formula,
        "sourceColumns": source_columns,
        "resolution": compiled,
    }


def seal_local_execution_resolution(
    resolution: Dict[str, Any],
    intent: QuestionIntent,
    table_columns: Set[str],
) -> Dict[str, Any]:
    updated = dict(resolution or {})
    metric_key = str(updated.get("metricKey") or intent.metric_name or intent.metric_column or "")
    owner_table = str(updated.get("ownerTable") or intent.preferred_table or "")
    formula = str(updated.get("formula") or intent.metric_formula or "")
    source_columns = [str(item) for item in updated.get("sourceColumns") or [] if str(item)]
    if not source_columns and formula:
        source_columns = formula_columns(formula, table_columns)
    if not metric_key or owner_table != intent.preferred_table or not formula or not source_columns:
        return {}
    if any(column not in table_columns for column in source_columns) or not compile_metric_formula(formula, table_columns):
        return {}
    provenance = updated.get("contractProvenance")
    if not isinstance(provenance, dict):
        provenance = {
            "kind": "execution_contract",
            "ownerTable": owner_table,
            "metricKey": metric_key,
            "taskId": intent.plan_task_id,
        }
    updated.update(
        {
            "semanticRefId": str(updated.get("semanticRefId") or "semantic:compiled_local:%s:metric:%s" % (owner_table, metric_key)),
            "metricKey": metric_key,
            "ownerTable": owner_table,
            "formula": formula,
            "sourceColumns": source_columns,
            "metricGovernanceMode": "compiled_local",
            "contractProvenance": provenance,
            "resolutionSource": "compiled_local",
            "localCompilationPolicy": str(
                updated.get("localCompilationPolicy")
                or updated.get("aggregationSource")
                or "provided_execution_contract"
            ),
        }
    )
    return seal_semantic_metric_resolution(updated, force=True)


def local_resolution_complete(resolution: Dict[str, Any], table: str, table_columns: Set[str]) -> bool:
    formula = str(resolution.get("formula") or "")
    source_columns = [str(item) for item in resolution.get("sourceColumns") or [] if str(item)]
    return bool(
        resolution.get("metricKey")
        and str(resolution.get("ownerTable") or "") == table
        and formula
        and source_columns
        and all(column in table_columns for column in source_columns)
        and compile_metric_formula(formula, table_columns)
    )


def local_metric_entry(intent: QuestionIntent, asset_pack: PlanningAssetPack) -> Any:
    candidates = []
    requested = {str(item) for item in [intent.metric_name, intent.metric_column] if str(item)}
    for metric in asset_pack.metrics:
        if metric.table != intent.preferred_table:
            continue
        names = {str(metric.key or ""), str(metric.title or ""), *[str(item) for item in metric.aliases or []]}
        columns = {str(item) for item in metric.columns or []}
        if requested & (names | columns):
            candidates.append(metric)
    identities = {(str(item.table), str(item.key), str(item.source_ref_id)) for item in candidates}
    return candidates[0] if len(identities) == 1 else None


def normalized_local_aggregation(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "sum": "sum",
        "avg": "avg",
        "average": "avg",
        "count": "count",
        "count_distinct": "count_distinct",
        "distinct_count": "count_distinct",
        "max": "max",
        "min": "min",
    }
    return aliases.get(text, "")


def local_aggregation_formula(aggregation: str, column: str) -> str:
    if aggregation == "count_distinct":
        return "COUNT(DISTINCT `%s`)" % column
    if aggregation in {"sum", "avg", "count", "max", "min"}:
        return "%s(`%s`)" % (aggregation.upper(), column)
    return ""


def metric_specs_for_contract(
    intent: QuestionIntent,
    table: str,
    metric_contract: Dict[str, Any],
    asset_pack: PlanningAssetPack,
    table_columns: Set[str],
) -> List[Dict[str, Any]]:
    specs = metric_specs_for_intent(intent, table)
    if not specs and metric_contract.get("formula"):
        specs = [
            {
                "metricName": str(metric_contract.get("metricKey") or ""),
                "metricColumn": intent.metric_column,
                "metricFormula": str(metric_contract.get("formula") or ""),
                "sourceColumns": list(metric_contract.get("sourceColumns") or []),
                "sourceTaskId": intent.plan_task_id,
            }
        ]
    compiled_specs: List[Dict[str, Any]] = []
    for spec in specs:
        compiled = dict(spec)
        if not compiled.get("metricFormula"):
            spec_intent = intent.model_copy(
                update={
                    "metric_name": str(compiled.get("metricName") or ""),
                    "metric_column": str(compiled.get("metricColumn") or ""),
                    "metric_formula": "",
                    "metric_specs": [],
                    "metric_resolution": {},
                }
            )
            spec_contract = compile_node_metric_contract(spec_intent, asset_pack, table_columns)
            if spec_contract.get("mode") == "compiled_local":
                compiled["metricName"] = str(spec_contract.get("metricKey") or compiled.get("metricName") or "")
                compiled["metricFormula"] = str(spec_contract.get("formula") or "")
                compiled["sourceColumns"] = list(spec_contract.get("sourceColumns") or [])
        elif not compiled.get("sourceColumns"):
            compiled["sourceColumns"] = formula_columns(str(compiled.get("metricFormula") or ""), table_columns)
        compiled_specs.append(compiled)
    return compiled_specs


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
    normalized = {
        "metricName": metric_name,
        "metricColumn": metric_column,
        "metricFormula": metric_formula,
        "sourceColumns": dedupe_strings(source_columns),
        "sourceTaskId": str(spec.get("sourceTaskId") or spec.get("source_task_id") or intent.plan_task_id or ""),
    }
    resolution = intent.metric_resolution or {}
    presentation_fields = {
        "displayName": ("displayName", "display_name"),
        "naturalName": ("naturalName", "natural_name"),
        "description": ("description",),
        "unit": ("unit",),
        "valueFormat": ("valueFormat", "value_format"),
        "decimalPlaces": ("decimalPlaces", "decimal_places"),
        "metricType": ("metricType", "metric_type"),
        "aggregationPolicy": ("aggregationPolicy", "aggregation_policy"),
        "applicableTimeGrain": ("applicableTimeGrain", "applicable_time_grain"),
        "semanticRefId": ("semanticRefId", "semantic_ref_id"),
        "ownerTable": ("ownerTable", "owner_table"),
    }
    for target, aliases in presentation_fields.items():
        value = next(
            (
                source.get(alias)
                for source in [spec, resolution]
                for alias in aliases
                if source.get(alias) not in (None, "", [], {})
            ),
            None,
        )
        if value not in (None, "", [], {}):
            normalized[target] = value
    if table and not normalized.get("ownerTable"):
        normalized["ownerTable"] = table
    return normalized


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


def structured_metric_select_parts(
    intent: QuestionIntent,
    table: str,
    columns: set,
    contract_specs: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, str]] | None:
    parts: List[Tuple[str, str]] = []
    for spec in contract_specs or metric_specs_for_intent(intent, table):
        metric_alias = str(spec.get("metricName") or "")
        metric_formula = str(spec.get("metricFormula") or "")
        metric_column = str(spec.get("metricColumn") or "")
        metric_expr = compile_metric_formula(metric_formula, columns)
        if metric_formula and not metric_expr:
            return None
        if metric_expr:
            parts.append((metric_expr, metric_alias or "metric_value"))
            continue
        if metric_column:
            return None
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
    if metric_column:
        return "sum_%s" % metric_column
    return "metric_value"


def metric_alias_candidates(metric_key: str) -> List[str]:
    text = str(metric_key or "")
    aliases = ["cnt", "count"] if text.endswith("_cnt") or text.endswith("_count") else []
    return dedupe_strings([text] + aliases)


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


def governed_metric_contract_issue(resolution: Dict[str, Any], table: str = "") -> str:
    payload = resolution or {}
    components = [item for item in payload.get("componentMetrics") or [] if isinstance(item, dict)]
    if components:
        if not all(
            str(item.get("semanticRefId") or "").startswith("semantic:")
            and item.get("metricKey")
            and item.get("ownerTable")
            and item.get("sourceColumns")
            and item.get("formula")
            for item in components
        ):
            return "derived metric components are missing governed semantic references"
    return semantic_metric_contract_issue(payload, table)


def metric_execution_contract_issue(contract: NodePlanContract) -> str:
    mode = str(contract.metric_governance_mode or "legacy_unsealed")
    if mode == "published_semantic":
        return governed_metric_contract_issue(contract.metric_resolution, contract.preferred_table)
    if mode == "compiled_local":
        resolution = contract.metric_resolution or {}
        if str(resolution.get("ownerTable") or "") != contract.preferred_table:
            return "locally compiled metric ownerTable does not match the execution table"
        if not str(resolution.get("metricKey") or ""):
            return "locally compiled metric has no metricKey"
        source_columns = [str(item) for item in resolution.get("sourceColumns") or [] if str(item)]
        if not source_columns:
            return "locally compiled metric has no sourceColumns"
        if any(column not in set(contract.allowed_columns) for column in source_columns):
            return "locally compiled metric sourceColumns are outside the node schema"
        formula = str(resolution.get("formula") or "")
        if not formula or not compile_metric_formula(formula, set(contract.allowed_columns)):
            return "locally compiled metric formula is not executable against the node schema"
        semantic_ref_id = str(resolution.get("semanticRefId") or "")
        provenance = resolution.get("contractProvenance") or {}
        if not semantic_ref_id.startswith("semantic:compiled_local:"):
            return "locally compiled metric has no local semantic reference"
        if not isinstance(provenance, dict) or str(provenance.get("kind") or "") not in {
            "execution_contract",
            "planning_asset",
        }:
            return "locally compiled metric has no verified provenance"
        sealed_issue = semantic_metric_contract_issue(resolution, contract.preferred_table)
        if sealed_issue:
            return sealed_issue
        for spec in contract.metric_specs or []:
            spec_formula = str(spec.get("metricFormula") or "")
            spec_columns = [str(item) for item in spec.get("sourceColumns") or [] if str(item)]
            if not spec_formula or not spec_columns:
                return "locally compiled multi-metric contract is incomplete"
            if any(column not in set(contract.allowed_columns) for column in spec_columns):
                return "locally compiled multi-metric sourceColumns are outside the node schema"
            if not compile_metric_formula(spec_formula, set(contract.allowed_columns)):
                return "locally compiled multi-metric formula is not executable against the node schema"
        return ""
    return "metric execution contract is unsealed and has no verified local provenance"


def semantic_metric_binding_issue(contract: NodePlanContract) -> str:
    resolution = contract.metric_resolution or {}
    if len(contract.metric_specs or []) > 1:
        return ""
    metric_key = str(resolution.get("metricKey") or "")
    formula = str(resolution.get("formula") or "")
    source_columns = {str(item) for item in resolution.get("sourceColumns") or [] if str(item)}
    if contract.metric_name and metric_key and contract.metric_name != metric_key:
        return "node metricName differs from the sealed semantic metricKey"
    if contract.metric_formula and formula and not equivalent_formula_text(contract.metric_formula, formula):
        return "node metricFormula differs from the sealed semantic formula"
    if contract.metric_column and source_columns and contract.metric_column not in source_columns:
        return "node metricColumn is outside the sealed semantic sourceColumns"
    return ""


def governed_realtime_metric_resolution(
    resolution: Dict[str, Any],
    source_table: str,
    fallback_table: str,
    fallback_metadata: Dict[str, Any],
    asset_pack: PlanningAssetPack,
) -> Dict[str, Any]:
    """Switch a metric table only through an explicit semantic metric mapping."""
    source_ref = str(resolution.get("semanticRefId") or "")
    metric_key = str(resolution.get("metricKey") or "")
    raw_mappings = fallback_metadata.get("metricMappings") or fallback_metadata.get("semanticMetricMappings") or []
    if isinstance(raw_mappings, dict):
        raw_mappings = [
            {"sourceSemanticRefId": key, "targetSemanticRefId": value}
            for key, value in raw_mappings.items()
        ]
    mapping = next(
        (
            item
            for item in raw_mappings
            if isinstance(item, dict)
            and str(item.get("sourceSemanticRefId") or item.get("sourceMetricRef") or "") in {source_ref, metric_key}
        ),
        None,
    )
    if not mapping:
        return {}
    target_ref = str(mapping.get("targetSemanticRefId") or mapping.get("targetMetricRef") or "")
    target = next(
        (
            item
            for item in asset_pack.metrics
            if item.table == fallback_table and target_ref in {item.source_ref_id, item.key}
        ),
        None,
    )
    if not target:
        return {}
    metadata = dict(target.metadata or {})
    source_columns = [str(item) for item in metadata.get("sourceColumns") or target.columns or [] if str(item)]
    formula = str(metadata.get("formula") or metadata.get("metricFormula") or "")
    if not source_columns or not formula:
        return {}
    updated = {
        **resolution,
        "metricKey": target.key,
        "ownerTable": fallback_table,
        "sourceColumns": source_columns,
        "formula": formula,
        "unit": str(metadata.get("unit") or resolution.get("unit") or ""),
        "displayName": target.title or str(metadata.get("businessName") or target.key),
        "description": str(metadata.get("description") or ""),
        "semanticRefId": target.source_ref_id,
        "fallbackFromSemanticRefId": source_ref,
        "fallbackFromTable": source_table,
        "resolutionSource": "%s+governed_realtime_fallback" % str(resolution.get("resolutionSource") or "semantic"),
    }
    return seal_semantic_metric_resolution(updated, force=True)


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


def intent_is_time_series(intent: QuestionIntent) -> bool:
    resolution = dict(intent.metric_resolution or {})
    time_contract = resolution.get("timeWindowContract") if isinstance(resolution.get("timeWindowContract"), dict) else {}
    time_column = str(
        resolution.get("timeColumn")
        or resolution.get("time_column")
        or time_contract.get("partitionColumn")
        or ""
    )
    result_role = str(resolution.get("displayRole") or resolution.get("resultRole") or "").lower()
    return result_role in {"trend", "trend_context", "time_series"} or bool(
        time_column and intent.group_by_column == time_column
    )


def entity_set_from_rows(task_id: str, intent: QuestionIntent, rows: List[Dict[str, Any]], max_values: int) -> EntitySet:
    declared_columns = dedupe_strings(
        [intent.group_by_column, intent.filter_column, *intent.output_keys, *intent.required_evidence]
    )
    row_columns = set(rows[0].keys()) if rows else set()
    key = next((column for column in declared_columns if column in row_columns), "")
    values = []
    column_values: Dict[str, List[Any]] = {}
    for row in rows:
        if key:
            value = row.get(key)
            if not blank_entity_value(value) and value not in values:
                values.append(value)
        for column in declared_columns:
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
    resolution = dict(intent.metric_resolution or {})
    return bool(resolution.get("groupByNonEmptyRequired"))


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
    if column == group_by:
        return True
    metric_columns = {intent.metric_column}
    for spec in metric_specs_for_intent(intent, intent.preferred_table):
        metric_columns.update(str(item) for item in spec.get("sourceColumns") or [])
        metric_columns.add(str(spec.get("metricName") or ""))
    return column in set(intent.output_keys) and column not in metric_columns


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
    excluded_dependent_columns: Optional[set] = None,
) -> tuple:
    row_keys = set(rows[0].keys()) if rows else set()
    excluded = set(excluded_dependent_columns or set())
    preferred_pairs = dependency_join_pairs(dep)
    for key, dep_key in preferred_pairs:
        if dep_key in excluded:
            continue
        if key in row_keys and dep_key in dependent_columns:
            return key, dep_key
    if parent_result.entity_set:
        source_key = parent_result.entity_set.join_key
        for key, dep_key in preferred_pairs:
            if dep_key in excluded:
                continue
            if source_key == key and dep_key in dependent_columns:
                return key, dep_key
    return "", ""


def multi_entity_transfer_values(
    dep: Any,
    rows: List[Dict[str, Any]],
    parent_result: AgentTaskResult,
    dependent_columns: set,
    max_values: int,
    excluded_dependent_columns: Optional[set] = None,
) -> Dict[str, List[Any]]:
    column_values: Dict[str, List[Any]] = {}
    pairs = dependency_join_pairs(dep)
    allowed_dependent_keys = dependency_allowed_dependent_entity_keys(
        dep,
        dependent_columns,
        excluded_dependent_columns,
    )
    if not rows:
        if not parent_result.entity_set:
            return {}
        for source_key, dependent_key in pairs:
            values = list(parent_result.entity_set.column_values.get(source_key) or [])
            if source_key == parent_result.entity_set.join_key:
                values.extend(parent_result.entity_set.values or [])
            if dependent_key not in allowed_dependent_keys:
                continue
            for value in values:
                if blank_entity_value(value):
                    continue
                bucket = column_values.setdefault(dependent_key, [])
                if value not in bucket:
                    bucket.append(value)
        return {column: values[:max_values] for column, values in column_values.items() if values}
    row_keys = set(rows[0].keys())
    for key, dep_key in pairs:
        if key not in row_keys or dep_key not in dependent_columns:
            continue
        values = column_values.setdefault(dep_key, [])
        for row in rows:
            value = row.get(key)
            if not blank_entity_value(value) and value not in values:
                values.append(value)
    if parent_result.entity_set:
        for source_key, dependent_key in pairs:
            if dependent_key not in allowed_dependent_keys:
                continue
            values = list(parent_result.entity_set.column_values.get(source_key) or [])
            if source_key == parent_result.entity_set.join_key:
                values.extend(parent_result.entity_set.values or [])
            target_values = column_values.setdefault(dependent_key, [])
            for value in values:
                if not blank_entity_value(value) and value not in target_values:
                    target_values.append(value)
    return {
        column: [value for value in values if not blank_entity_value(value)][:max_values]
        for column, values in column_values.items()
        if any(not blank_entity_value(value) for value in values)
    }


def dependency_allowed_dependent_entity_keys(
    dep: Any,
    dependent_columns: set,
    excluded_dependent_columns: Optional[set] = None,
) -> set[str]:
    excluded = set(excluded_dependent_columns or set())
    return {
        dep_key
        for _, dep_key in dependency_join_pairs(dep)
        if dep_key in dependent_columns and dep_key not in excluded
    }


def dependency_join_pairs(dep: Any) -> List[Tuple[str, str]]:
    """Return only join mappings explicitly declared by the validated graph."""

    pairs: List[Tuple[str, str]] = []
    for pair in paired_join_tokens(dep.anchor_column, dep.dependent_column) + paired_join_tokens(dep.join_key, dep.join_key):
        if pair not in pairs:
            pairs.append(pair)
    return pairs


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


def derived_ratio_numerator_zero_fill_allowed(intent: QuestionIntent, components: List[Dict[str, Any]]) -> bool:
    formula = str(intent.metric_formula or (intent.metric_resolution or {}).get("formula") or "")
    if "/" not in formula or len(components) < 2:
        return False
    numerator = components[0] if isinstance(components[0], dict) else {}
    numerator_formula = str(numerator.get("formula") or "").strip().lower()
    if not numerator_formula.startswith(("count(", "count (", "sum(", "sum (")):
        return False
    return bool(intent.group_by_column)


def is_repairable_doris_error(error_text: str) -> bool:
    return doris_error_policy(classify_doris_error(error_text)).get("llm_repair", False)


def semantic_table_access_hint(asset_pack: PlanningAssetPack, table: str, columns: set) -> Dict[str, Any]:
    entry = next((item for item in asset_pack.tables if item.table == table), None)
    metadata = dict(getattr(entry, "metadata", {}) or {})
    physical = dict(metadata.get("physicalTableMetadata") or {})
    usage = dict(metadata.get("tableUsageProfile") or {})
    primary = list(physical.get("primaryKeyColumns") or metadata.get("primaryKeyColumns") or [])
    partition = list(physical.get("partitionColumns") or metadata.get("partitionColumns") or [])
    bucket = list(physical.get("bucketColumns") or metadata.get("bucketColumns") or [])
    tenant = str(metadata.get("merchantFilterColumn") or "")
    time_column = str(metadata.get("timeColumn") or "")
    entity_keys = [item.key for item in asset_pack.entity_keys if item.table == table]
    return {
        "tableKind": str(usage.get("businessLayer") or metadata.get("dataGrain") or ""),
        "uniqueKeys": [item for item in dedupe_strings(primary + entity_keys + partition) if item in columns],
        "distributionKeys": [item for item in dedupe_strings(bucket) if item in columns],
        "bestEqualityFilters": [item for item in dedupe_strings([tenant, *primary, *entity_keys, time_column]) if item in columns],
        "fallbackFilters": [item for item in dedupe_strings([tenant, *partition, time_column]) if item in columns],
        "invertedIndexes": [item for item in physical.get("invertedIndexColumns") or [] if item in columns],
        "timeWindowPolicy": str(metadata.get("timeWindowPolicy") or usage.get("timeWindowPolicy") or ""),
    }


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


def submit_with_current_context(executor: ThreadPoolExecutor, fn: Any, *args: Any):
    context = copy_context()
    return executor.submit(context.run, fn, *args)


def doris_query_with_cancellation(
    repository: Any,
    sql: str,
    params: Optional[List[Any]] = None,
    cancel_events: Optional[List[Any]] = None,
    timeout_seconds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = getattr(repository, "query")
    effective_cancel_events = [event for event in (cancel_events or []) if event is not None]
    runtime_cancel_event = current_tool_cancel_event()
    if runtime_cancel_event is not None and runtime_cancel_event not in effective_cancel_events:
        effective_cancel_events.append(runtime_cancel_event)
    try:
        signature = inspect.signature(query)
        supports_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        supports_cancellation = supports_kwargs or "cancel_events" in signature.parameters
    except (TypeError, ValueError):
        supports_cancellation = False
    if supports_cancellation:
        return query(
            sql,
            params,
            cancel_events=effective_cancel_events,
            timeout_seconds=timeout_seconds,
        )
    return query(sql, params)


def set_node_prompt_trace(context: NodeExecutionContext, key: str, payload: Dict[str, Any]) -> None:
    traces = context.runtime_scratch.setdefault("prompt_traces", {})
    traces[str(key)] = payload


def pop_node_prompt_trace(context: NodeExecutionContext, key: str) -> Optional[Dict[str, Any]]:
    traces = context.runtime_scratch.get("prompt_traces") or {}
    return traces.pop(str(key), None)


def pop_node_runtime_value(context: NodeExecutionContext, key: str, default: Any = None) -> Any:
    return context.runtime_scratch.pop(str(key), default)


def context_is_cancelled(context: NodeExecutionContext) -> bool:
    event = getattr(context, "cancel_event", None)
    return bool(event is not None and event.is_set())


def cancelled_result(intent: QuestionIntent) -> AgentTaskResult:
    result = failed_result(intent.plan_task_id, intent, "NodeWorker 已取消，停止后续查询与重试")
    result.query_bundle.runtime_events.append({"event": "node.cancelled", "taskId": intent.plan_task_id})
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
