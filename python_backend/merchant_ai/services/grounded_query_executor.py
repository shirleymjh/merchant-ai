from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import sqlglot
from sqlglot import exp

from merchant_ai.config import Settings
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.models import (
    AgentRunResult,
    AgentTask,
    AgentTaskResult,
    EvidenceCheckResult,
    EvidenceGap,
    NodePlanContract,
    NodeTaskProfile,
    QueryBundle,
    QueryPlan,
    ReActStep,
    SqlValidationResult,
)
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import (
    default_row_access_policy,
    normalize_row_access_policy,
)
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.query_sql_binding import quote_identifier, sql_literal
from merchant_ai.services.query_security import apply_column_masks
from merchant_ai.services.time_semantics import (
    CALENDAR_ANCHOR_POLICY,
    LATEST_PARTITION_ANCHOR_POLICY,
    latest_as_of_partition_predicate_sql,
    latest_partition_window_predicate,
)


@dataclass(frozen=True)
class GroundedSqlCompilation:
    sql: str
    table: str
    metric_aliases: tuple[str, ...]
    group_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    node_contract: NodePlanContract


class GroundedQueryExecutionKernel:
    """Deterministic Data Engine for an activated GroundedQueryContract.

    The executor has no Planner, NodeAgent, ReAct SQL drafting, critic, repair
    loop, or workflow dependency.  It projects the already-grounded semantic
    formulae into one governed SQL statement, validates the exact table and
    columns, applies tenant/access scope, executes Doris, and returns the
    ordinary evidence models consumed by the verifier and answer renderer.
    """

    def __init__(
        self,
        doris_repository: Any,
        settings: Settings,
        *,
        access_control: AccessControlService | None = None,
    ) -> None:
        self.doris_repository = doris_repository
        self.settings = settings
        self.access_control = access_control or AccessControlService(settings)

    def execute_contract(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: Any,
        question: str,
        *,
        run_id: str = "",
        access_role: str = "merchant_analyst",
        user_scope: dict[str, Any] | None = None,
        execution_preparation: Any = None,
    ) -> AgentRunResult:
        if execution_preparation is None or not bool(
            getattr(execution_preparation, "executable", False)
        ):
            raise RuntimeError("grounded execution requires a validated preparation")
        if not contract.ready:
            raise RuntimeError("grounded execution requires an active READY Contract")
        if len(plan.intents) != 1:
            raise RuntimeError("grounded direct execution requires exactly one compiled node")

        intent = plan.intents[0]
        compilation = self.compile_sql(
            merchant_id,
            contract,
            plan,
            asset_pack,
            access_role=access_role,
            user_scope=user_scope or {},
        )
        validation = self.validate_sql(compilation.sql, asset_pack)
        if not validation.valid:
            return self._failed_result(
                plan,
                compilation,
                validation,
                validation.error_code or "GROUNDED_SQL_VALIDATION_FAILED",
                validation.message,
            )

        decision = self.access_control.authorize_contract(
            compilation.node_contract,
            compilation.sql,
            run_id=run_id,
        )
        if not decision.allowed:
            denied = SqlValidationResult(
                valid=False,
                error_code=decision.code or "ACCESS_DENIED",
                message=decision.message or "grounded query access denied",
                base_tables=[compilation.table],
            )
            return self._failed_result(
                plan,
                compilation,
                denied,
                denied.error_code,
                denied.message,
            )

        started = time.perf_counter()
        try:
            rows = self.doris_repository.query(
                compilation.sql,
                timeout_seconds=max(
                    1,
                    int(getattr(self.settings, "doris_read_timeout_seconds", 30) or 30),
                ),
            )
            rows = apply_column_masks(
                [dict(row) for row in rows],
                compilation.node_contract.model_copy(
                    update={"masked_columns": dict(decision.masked_columns or {})}
                ),
            )
            for row in rows:
                row.setdefault(
                    "__timeWindowRole",
                    str(intent.time_range.window_role or "primary"),
                )
        except Exception as exc:
            failed = SqlValidationResult(
                valid=False,
                error_code="DORIS_ERROR",
                message=str(exc)[:500],
                base_tables=[compilation.table],
            )
            return self._failed_result(
                plan,
                compilation,
                failed,
                "DORIS_ERROR",
                str(exc)[:500],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        task_id = intent.plan_task_id
        bundle = QueryBundle(
            sql=compilation.sql,
            tables=[compilation.table],
            rows=rows,
            original_row_count=len(rows),
            source_row_counts={task_id: len(rows)},
            duration_ms=duration_ms,
            cache_hit=bool(getattr(self.doris_repository, "last_cache_hit", False)),
            cache_key=str(getattr(self.doris_repository, "last_cache_key", "") or ""),
            runtime_events=[
                {
                    "event": "grounded_data_engine.executed",
                    "status": "success",
                    "taskId": task_id,
                    "table": compilation.table,
                    "rowCount": len(rows),
                    "plannerLlmCalls": 0,
                    "sqlLlmCalls": 0,
                }
            ],
        )
        task_result = AgentTaskResult(
            task_id=task_id,
            sub_agent_type="GROUNDED_DATA_ENGINE",
            success=True,
            summary="Grounded Data Engine executed %d row(s)" % len(rows),
            query_bundle=bundle,
            validation_results=[validation],
            react_trace=[
                ReActStep(
                    round=1,
                    reason="Execute the activated GroundedQueryContract deterministically",
                    action="grounded_data_engine.execute_sql",
                    observation="table=%s;rows=%d" % (compilation.table, len(rows)),
                )
            ],
            node_task_profile=NodeTaskProfile(
                task_id=task_id,
                task_kind="GROUNDED_DATA_ENGINE",
                sql_strategy="grounded_deterministic",
                selected_tools=[
                    "compile_grounded_sql",
                    "validate_grounded_sql",
                    "authorize_grounded_query",
                    "execute_doris",
                ],
                reason="READY GroundedQueryContract is compiled without a NodeAgent",
                risk_controls=[
                    "contract_bound_table",
                    "contract_bound_columns",
                    "published_metric_formula",
                    "tenant_scope",
                    "read_only_sql",
                ],
                contract_status="passed",
                sql_draft_source="grounded_deterministic",
            ),
            node_plan_contract=compilation.node_contract,
        )
        self.access_control.record_query_audit(
            decision,
            row_count=len(rows),
            status="success",
        )
        return AgentRunResult(
            executed_query_graph_fingerprint=query_graph_fingerprint(plan),
            tasks=[
                AgentTask(
                    task_id=task_id,
                    plan_index=0,
                    sub_agent_type="GROUNDED_DATA_ENGINE",
                    instruction=question,
                )
            ],
            task_results=[task_result],
            query_bundles=[bundle.model_copy(deep=True)],
            merged_query_bundle=bundle.model_copy(deep=True),
            evidence_check=EvidenceCheckResult(
                passed=True,
                summary="Grounded Data Engine execution completed",
            ),
            node_task_profiles=[task_result.node_task_profile.model_copy(deep=True)],
            node_plan_contracts=[compilation.node_contract.model_copy(deep=True)],
        )

    def compile_sql(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: Any,
        *,
        access_role: str,
        user_scope: dict[str, Any],
    ) -> GroundedSqlCompilation:
        intent = plan.intents[0]
        table = str(intent.preferred_table or contract.primary_table or "").strip()
        table_binding = next(
            (item for item in contract.tables if item.table == table),
            None,
        )
        if not table or table_binding is None:
            raise RuntimeError("grounded execution table is not bound by the Contract")
        columns = set(asset_pack.known_columns(table))
        if not columns:
            raise RuntimeError("grounded execution table has no projected schema")

        metric_parts: list[str] = []
        metric_aliases: list[str] = []
        required_columns: list[str] = []
        for metric in contract.metrics:
            if metric.table != table:
                raise RuntimeError("grounded direct execution cannot infer a cross-table metric")
            formula = compile_metric_formula(metric.formula, columns)
            if not formula:
                raise RuntimeError(
                    "published metric formula cannot compile against projected schema: %s"
                    % metric.semantic_ref_id
                )
            alias = str(metric.metric_key or "").strip()
            if not alias or alias in metric_aliases:
                raise RuntimeError("grounded metric aliases must be non-empty and unique")
            metric_parts.append("%s AS %s" % (formula, quote_identifier(alias)))
            metric_aliases.append(alias)
            required_columns.extend(metric.source_columns)

        group_bindings = [
            item for item in contract.dimensions if item.usage == "group_by"
        ]
        if len(group_bindings) > 1:
            raise RuntimeError("grounded direct execution supports one explicit group dimension")
        group_columns: list[str] = []
        if group_bindings:
            dimension = group_bindings[0]
            if dimension.table != table or dimension.column not in columns:
                raise RuntimeError("grounded group dimension is outside the execution table")
            if dimension.column == table_binding.merchant_filter_column:
                raise RuntimeError("merchant scope column cannot be a business dimension")
            group_columns.append(dimension.column)
            required_columns.append(dimension.column)

        where: list[str] = []
        merchant_column = str(table_binding.merchant_filter_column or "").strip()
        if merchant_column:
            if merchant_column not in columns:
                raise RuntimeError("merchant scope column is absent from projected schema")
            where.append(
                "%s = %s" % (quote_identifier(merchant_column), sql_literal(merchant_id))
            )
            required_columns.append(merchant_column)

        table_metadata = self._table_metadata(asset_pack, table)
        region_column = str(table_metadata.get("regionFilterColumn") or "").strip()
        region = str(user_scope.get("region") or "").strip()
        if region:
            if not region_column or region_column not in columns:
                raise RuntimeError("authorized region scope has no declared semantic filter column")
            where.append(
                "%s = %s" % (quote_identifier(region_column), sql_literal(region))
            )
            required_columns.append(region_column)
        store_ids = [
            str(item)
            for item in (user_scope.get("storeIds") or user_scope.get("store_ids") or [])
            if str(item or "").strip()
        ]
        store_column = str(table_metadata.get("storeFilterColumn") or "").strip()
        if store_ids:
            if not store_column or store_column not in columns:
                raise RuntimeError("authorized store scope has no declared semantic filter column")
            where.append(
                "%s IN (%s)"
                % (
                    quote_identifier(store_column),
                    ", ".join(sql_literal(item) for item in store_ids[:200]),
                )
            )
            required_columns.append(store_column)

        time_column = self._time_column(contract, table_binding.time_column)
        if time_column:
            if time_column not in columns:
                raise RuntimeError("metric time column is absent from projected schema")
            where.append(
                self._time_predicate(
                    contract,
                    table,
                    time_column,
                    merchant_column,
                    merchant_id,
                )
            )
            required_columns.append(time_column)

        if group_columns:
            group_column = group_columns[0]
            where.extend(
                [
                    "%s IS NOT NULL" % quote_identifier(group_column),
                    "%s != ''" % quote_identifier(group_column),
                ]
            )

        select_parts = [quote_identifier(column) for column in group_columns]
        select_parts.extend(metric_parts)
        sql = "SELECT %s FROM %s" % (
            ", ".join(select_parts),
            quote_identifier(table),
        )
        if where:
            sql += " WHERE " + " AND ".join("(%s)" % item for item in where if item)
        if group_columns:
            sql += " GROUP BY " + ", ".join(
                quote_identifier(column) for column in group_columns
            )
        if contract.ranking.enabled:
            ranking_metric = next(
                (
                    metric.metric_key
                    for metric in contract.metrics
                    if metric.semantic_ref_id == contract.ranking.metric_ref_id
                ),
                "",
            )
            if not ranking_metric:
                raise RuntimeError("grounded ranking metric is not bound")
            sql += " ORDER BY %s DESC LIMIT %d" % (
                quote_identifier(ranking_metric),
                max(1, int(contract.ranking.limit or 1)),
            )
        elif group_columns:
            sql += " LIMIT %d" % max(1, int(intent.limit or 20))

        required = tuple(self._dedupe(required_columns))
        allowed_columns = self._dedupe([*columns])
        node_contract = NodePlanContract(
            task_id=intent.plan_task_id,
            question=contract.question,
            preferred_table=table,
            allowed_columns=allowed_columns,
            visible_columns=list(group_columns),
            internal_only_columns=[
                column
                for column in required
                if column not in group_columns
            ],
            required_columns=list(required),
            metric_column=intent.metric_column,
            metric_name=intent.metric_name,
            metric_formula=intent.metric_formula,
            metric_specs=[dict(item) for item in intent.metric_specs],
            group_by_column=intent.group_by_column,
            output_keys=list(intent.output_keys),
            required_evidence=list(intent.required_evidence),
            days=int(intent.days or 0),
            limit=int(intent.limit or 0),
            merchant_id=merchant_id,
            merchant_filter_column=merchant_column,
            effective_user_id=str(
                user_scope.get("userId") or user_scope.get("user_id") or ""
            ),
            authorized_region=region,
            authorized_store_ids=store_ids,
            region_filter_column=region_column,
            store_filter_column=store_column,
            access_role=access_role or "merchant_analyst",
            row_scope_policy=normalize_row_access_policy(
                table_metadata.get("rowAccessPolicy")
                or default_row_access_policy(merchant_column)
            ),
            answer_mode=str(intent.answer_mode),
            task_role=str(intent.task_role),
            sql_strategy="grounded_deterministic",
            metric_resolution=dict(intent.metric_resolution or {}),
            metric_governance_mode="grounded_query_contract",
            time_window_contract={
                "kind": contract.time_range.kind,
                "label": contract.time_range.label,
                "days": contract.time_range.days,
                "startDate": contract.time_range.start_date,
                "endDate": contract.time_range.end_date,
                "anchorPolicy": contract.time_range.anchor_policy,
                "partitionColumn": time_column,
                "tenantColumn": merchant_column,
            }
            if time_column
            else {},
        )
        return GroundedSqlCompilation(
            sql=sql,
            table=table,
            metric_aliases=tuple(metric_aliases),
            group_columns=tuple(group_columns),
            required_columns=required,
            node_contract=node_contract,
        )

    @staticmethod
    def validate_sql(sql: str, asset_pack: Any) -> SqlValidationResult:
        normalized = str(sql or "").strip()
        if not normalized:
            return SqlValidationResult(
                valid=False,
                error_code="SQL_EMPTY",
                message="Grounded SQL is empty",
            )
        if ";" in normalized.rstrip(";"):
            return SqlValidationResult(
                valid=False,
                error_code="MULTI_STATEMENT",
                message="Grounded execution accepts one SELECT statement",
            )
        try:
            parsed = sqlglot.parse_one(normalized, read="doris")
        except Exception as exc:
            return SqlValidationResult(
                valid=False,
                error_code="PARSE_ERROR",
                message="Grounded SQL parse failed: %s" % str(exc)[:200],
            )
        if not isinstance(parsed, (exp.Select, exp.Union)) and not parsed.find(exp.Select):
            return SqlValidationResult(
                valid=False,
                error_code="NOT_SELECT",
                message="Only SELECT queries are allowed",
            )
        if any(
            parsed.find(kind)
            for kind in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create)
        ):
            return SqlValidationResult(
                valid=False,
                error_code="UNSAFE_SQL",
                message="Grounded SQL contains a write or DDL operation",
            )

        cte_names = {cte.alias for cte in parsed.find_all(exp.CTE) if cte.alias}
        base_tables = GroundedQueryExecutionKernel._dedupe(
            table.name
            for table in parsed.find_all(exp.Table)
            if table.name and table.name not in cte_names
        )
        known_tables = set(asset_pack.known_tables())
        unknown_tables = [table for table in base_tables if table not in known_tables]
        if unknown_tables:
            return SqlValidationResult(
                valid=False,
                error_code="UNKNOWN_BASE_TABLE",
                message="Grounded SQL references a table outside the Contract",
                base_tables=base_tables,
                unknown_tables=unknown_tables,
            )
        allowed_columns = {
            column
            for table in base_tables
            for column in asset_pack.known_columns(table)
        }
        aliases = {
            alias.alias
            for alias in parsed.find_all(exp.Alias)
            if alias.alias
        }
        unknown_columns = sorted(
            {
                column.name
                for column in parsed.find_all(exp.Column)
                if column.name and column.name not in allowed_columns and column.name not in aliases
            }
        )
        if unknown_columns:
            return SqlValidationResult(
                valid=False,
                error_code="UNKNOWN_COLUMN",
                message="Grounded SQL references columns outside the Contract",
                base_tables=base_tables,
                unknown_columns=unknown_columns,
            )
        return SqlValidationResult(
            valid=True,
            base_tables=base_tables,
            cte_names=sorted(cte_names),
            message="passed",
        )

    def _failed_result(
        self,
        plan: QueryPlan,
        compilation: GroundedSqlCompilation,
        validation: SqlValidationResult,
        code: str,
        message: str,
        *,
        duration_ms: int = 0,
    ) -> AgentRunResult:
        intent = plan.intents[0]
        bundle = QueryBundle(
            sql=compilation.sql,
            tables=[compilation.table],
            failed=True,
            error="%s: %s" % (code, message),
            summary=message,
            duration_ms=duration_ms,
            runtime_events=[
                {
                    "event": "grounded_data_engine.failed",
                    "status": "failed",
                    "code": code,
                    "taskId": intent.plan_task_id,
                    "table": compilation.table,
                }
            ],
        )
        task_result = AgentTaskResult(
            task_id=intent.plan_task_id,
            sub_agent_type="GROUNDED_DATA_ENGINE",
            success=False,
            summary=bundle.error,
            query_bundle=bundle,
            validation_results=[validation],
            node_plan_contract=compilation.node_contract,
            node_task_profile=NodeTaskProfile(
                task_id=intent.plan_task_id,
                task_kind="GROUNDED_DATA_ENGINE",
                sql_strategy="grounded_deterministic",
                contract_status=code,
                sql_draft_source="grounded_deterministic",
            ),
        )
        return AgentRunResult(
            executed_query_graph_fingerprint=query_graph_fingerprint(plan),
            tasks=[
                AgentTask(
                    task_id=intent.plan_task_id,
                    plan_index=0,
                    sub_agent_type="GROUNDED_DATA_ENGINE",
                    instruction=intent.question,
                )
            ],
            task_results=[task_result],
            query_bundles=[bundle.model_copy(deep=True)],
            merged_query_bundle=bundle.model_copy(deep=True),
            evidence_check=EvidenceCheckResult(
                passed=False,
                summary=message,
                gaps=["%s:%s" % (code, message)],
            ),
            evidence_gaps=[
                EvidenceGap(
                    code=code,
                    task_id=intent.plan_task_id,
                    evidence=compilation.sql[:240],
                    reason=message,
                )
            ],
            node_plan_contracts=[compilation.node_contract.model_copy(deep=True)],
        )

    @staticmethod
    def _table_metadata(asset_pack: Any, table: str) -> dict[str, Any]:
        entry = next(
            (item for item in asset_pack.tables if str(item.table or "") == table),
            None,
        )
        return dict(getattr(entry, "metadata", {}) or {}) if entry is not None else {}

    @staticmethod
    def _time_column(contract: GroundedQueryContract, table_default: str) -> str:
        columns = GroundedQueryExecutionKernel._dedupe(
            metric.time_column or table_default
            for metric in contract.metrics
            if metric.table == contract.primary_table
        )
        if len(columns) > 1:
            raise RuntimeError("grounded metrics have incompatible time columns")
        return columns[0] if columns else str(table_default or "")

    @staticmethod
    def _time_predicate(
        contract: GroundedQueryContract,
        table: str,
        time_column: str,
        merchant_column: str,
        merchant_id: str,
    ) -> str:
        policies = GroundedQueryExecutionKernel._dedupe(
            str(metric.time_semantics.get("selectionPolicy") or "period_window")
            for metric in contract.metrics
        )
        if len(policies) > 1:
            raise RuntimeError("grounded metrics have incompatible time selection policies")
        policy = policies[0] if policies else "period_window"
        time_range = contract.time_range
        if policy == "latest_as_of":
            return latest_as_of_partition_predicate_sql(
                table,
                time_column,
                anchor_value_sql=(sql_literal(time_range.end_date) if time_range.end_date else ""),
                tenant_column=merchant_column,
                tenant_value_sql=(sql_literal(merchant_id) if merchant_column else ""),
            )
        if policy == "per_time_grain" and contract.query_shape == "SCALAR" and time_range.days > 1:
            raise RuntimeError(
                "published metric requires per-time-grain execution for a multi-day window"
            )
        if (
            time_range.anchor_policy == LATEST_PARTITION_ANCHOR_POLICY
            and time_range.days > 0
        ):
            return latest_partition_window_predicate(
                table,
                time_range.days,
                partition_column=time_column,
                tenant_column=merchant_column,
                tenant_value_sql=(sql_literal(merchant_id) if merchant_column else ""),
                offset_days=time_range.offset_days,
            )
        if time_range.start_date and time_range.end_date:
            return "%s BETWEEN %s AND %s" % (
                quote_identifier(time_column),
                sql_literal(time_range.start_date),
                sql_literal(time_range.end_date),
            )
        if time_range.anchor_policy == CALENDAR_ANCHOR_POLICY:
            raise RuntimeError("calendar time range is missing explicit start/end bounds")
        raise RuntimeError("grounded time semantics cannot be compiled")

    @staticmethod
    def _dedupe(values: Iterable[Any]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result
