from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import time
import uuid
import fcntl
from dataclasses import dataclass, field as dataclass_field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from merchant_ai.config import Settings
from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.graph.query_graph_contract import query_graph_fingerprint
from merchant_ai.models import (
    AgentRunResult,
    AgentTask,
    AgentTaskResult,
    DataSnapshotContract,
    EvidenceCheckResult,
    EvidenceGap,
    EntitySet,
    EntityFilterObligation,
    EntityFilterVerificationProof,
    EntityReference,
    NodePlanContract,
    NodeTaskProfile,
    QueryBundle,
    QueryPlan,
    ReActStep,
    ResultCoverage,
    SqlValidationResult,
    VerifiedEvidence,
)
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.context_filesystem import merchant_uri_for_artifact
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspaceError,
    _atomic_write_at,
    _open_directory_beneath,
    _read_regular_file_at,
    grounded_context_owner_fingerprint,
    validated_grounded_query_artifact_roots,
)
from merchant_ai.services.assets import (
    default_row_access_policy,
    normalize_row_access_policy,
)
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.entity_contracts import (
    canonical_entity_values,
    entity_comparison_policy,
    entity_filter_contract_hash,
    entity_filter_sql_hash,
    entity_value_hash,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    grounded_detail_relationship_candidates,
)
from merchant_ai.services.grounded_runtime_budget import GroundedRuntimeBudget
from merchant_ai.services.grounded_execution_identity import (
    GroundedNodeExecutionIdentitySeal,
    GroundedRunExecutionIdentitySeal,
    build_grounded_node_execution_identity,
    build_grounded_run_execution_identity,
    grounded_data_snapshot_identity,
    require_grounded_execution_identity_live,
)
from merchant_ai.services.grounded_result_streaming import (
    GroundedResultArtifactReceipt,
    GroundedResultStreamLimits,
    GroundedResultStreamMaterializer,
    GroundedResultStreamingError,
    grounded_canonical_json_sha256,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    GroundedPopulationRuntimeGateError,
    PopulationExecutorNodeEvidence,
    PopulationPreExecutionReference,
)
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.doris_physical_plan_governance import (
    DorisPhysicalPlanGovernor,
    PartitionPruningRequirement,
    PhysicalPlanAssessment,
    PhysicalPlanGap,
)
from merchant_ai.services.query_sql_binding import quote_identifier, sql_literal
from merchant_ai.services.query_security import apply_column_masks
from merchant_ai.services.tool_runtime import ExecutionIdentity
from merchant_ai.services.time_semantics import (
    LATEST_AVAILABLE_PARTITION_DATA_AS_OF_POLICY,
    latest_as_of_partition_predicate_sql,
    latest_partition_window_predicate,
)

DEFAULT_ACCESS_ROLE = load_authorization_policy().default_access_role


@dataclass(frozen=True)
class GroundedSqlCompilation:
    sql: str
    table: str
    tables: tuple[str, ...]
    metric_aliases: tuple[str, ...]
    group_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    node_contract: NodePlanContract
    access_contracts: tuple[NodePlanContract, ...] = ()


@dataclass
class _GroundedStreamingRowState:
    """Bounded pre-mask state retained while the full result streams."""

    preview_limit: int
    raw_preview_rows: list[dict[str, Any]] = dataclass_field(
        default_factory=list
    )
    row_count: int = 0
    filter_column_missing: bool = False
    observed_filter_values: set[str] = dataclass_field(default_factory=set)


@dataclass(frozen=True)
class _GroundedExecutionIdentityContext:
    execution_identity: ExecutionIdentity
    run_identity: GroundedRunExecutionIdentitySeal
    node_identity: GroundedNodeExecutionIdentitySeal
    user_scope: dict[str, Any]
    reference_scope: Any
    graph_fingerprint: str
    query_node_id: str
    goal_contract_fingerprint: str
    query_contract_fingerprint: str
    sql_ast_fingerprint: str
    access_contracts: tuple[NodePlanContract, ...]


class GroundedQueryExecutionKernel:
    """Data Engine for the two explicitly governed grounded execution lanes.

    An extremely simple published scalar metric may arrive through the strict
    deterministic fast path. Every other query arrives as a complete SQL AST
    authored by the single Core LLM. The executor has no Planner, NodeAgent,
    SQL-drafting ReAct loop, critic, repair workflow, or business query
    templates; it only validates, injects trusted access scope, executes Doris,
    and returns evidence models.
    """

    def __init__(
        self,
        doris_repository: Any,
        settings: Settings,
        *,
        access_control: AccessControlService | None = None,
        population_execution_gate: (
            GroundedPopulationExecutionGate | None
        ) = None,
    ) -> None:
        self.doris_repository = doris_repository
        self.settings = settings
        self.access_control = access_control or AccessControlService(settings)
        self.population_execution_gate = population_execution_gate

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        capture = getattr(self.doris_repository, "capture_data_snapshot", None)
        if not callable(capture):
            return DataSnapshotContract(
                unsupported_reason="DATA_SNAPSHOT_CAPABILITY_UNAVAILABLE"
            )
        snapshot = capture(str(semantic_activation_fingerprint or "").strip())
        if isinstance(snapshot, DataSnapshotContract):
            return snapshot
        return DataSnapshotContract.model_validate(snapshot)

    def execute_contract(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: Any,
        question: str,
        *,
        run_id: str = "",
        artifact_root: str = "",
        context_owner_fingerprint: str = "",
        access_role: str = DEFAULT_ACCESS_ROLE,
        user_scope: dict[str, Any] | None = None,
        execution_reference_scope: Any = None,
        execution_goal_contract_fingerprint: str = "",
        expected_semantic_activation_fingerprint: str = "",
        population_pre_execution_reference: (
            PopulationPreExecutionReference | None
        ) = None,
        population_query_node_id: str = "",
        execution_preparation: Any = None,
        runtime_budget: GroundedRuntimeBudget | None = None,
        data_snapshot_contract: DataSnapshotContract | None = None,
        execution_generation: int = 0,
        execution_attempt_id: str = "",
        cancel_events: Iterable[Any] | None = None,
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
        stream_query = getattr(
            self.doris_repository,
            "stream_query_batches",
            None,
        )
        streaming_artifact_mode = bool(
            artifact_root and callable(stream_query)
        )
        candidate_validation = getattr(
            execution_preparation,
            "candidate_validation",
            None,
        )
        if isinstance(candidate_validation, GroundedSqlValidationResult):
            compilation = self.compile_core_sql_candidate(
                merchant_id,
                contract,
                plan,
                asset_pack,
                candidate_validation,
                access_role=access_role,
                user_scope=user_scope or {},
            )
            validation = self.validate_sql(compilation.sql, asset_pack)
        else:
            compilation = self.compile_sql(
                merchant_id,
                contract,
                plan,
                asset_pack,
                access_role=access_role,
                user_scope=user_scope or {},
                complete_result_artifact=streaming_artifact_mode,
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

        access_contracts = compilation.access_contracts or (compilation.node_contract,)
        decisions = []
        for item in access_contracts:
            if isinstance(candidate_validation, GroundedSqlValidationResult):
                decision = self.access_control.authorize_contract(
                    item,
                    compilation.sql,
                    run_id=run_id,
                    checked_columns_override=item.required_columns,
                )
            else:
                decision = self.access_control.authorize_contract(
                    item,
                    compilation.sql,
                    run_id=run_id,
                )
            decisions.append(decision)
        denied_decision = next((item for item in decisions if not item.allowed), None)
        if denied_decision is not None:
            denied = SqlValidationResult(
                valid=False,
                error_code=denied_decision.code or "ACCESS_DENIED",
                message=denied_decision.message or "grounded query access denied",
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                denied,
                denied.error_code,
                denied.message,
            )

        governed_detail = bool(
            contract.query_shape in {"DETAIL", "ENTITY_LOOKUP"}
            and str(execution_goal_contract_fingerprint or "").strip()
        )
        if governed_detail and not artifact_root:
            unavailable = SqlValidationResult(
                valid=False,
                error_code="QUERY_RESULT_COMPLETE_ARTIFACT_REQUIRED",
                message=(
                    "governed detail execution requires a complete result "
                    "artifact root"
                ),
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                unavailable,
                unavailable.error_code,
                unavailable.message,
            )
        if (
            artifact_root
            and contract.query_shape in {"DETAIL", "ENTITY_LOOKUP"}
            and not streaming_artifact_mode
        ):
            unavailable = SqlValidationResult(
                valid=False,
                error_code="QUERY_RESULT_STREAMING_REQUIRED",
                message=(
                    "complete detail publication requires a streaming Doris "
                    "repository"
                ),
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                unavailable,
                unavailable.error_code,
                unavailable.message,
            )

        validated_artifact_roots: tuple[Path, Path] | None = None
        if artifact_root:
            try:
                validated_artifact_roots = (
                    validated_grounded_query_artifact_roots(
                        self.settings.resolved_workspace_path,
                        artifact_root,
                    )
                )
            except (GroundedContextWorkspaceError, OSError) as exc:
                invalid_root = SqlValidationResult(
                    valid=False,
                    error_code="QUERY_RESULT_ARTIFACT_ROOT_INVALID",
                    message=str(exc)[:500],
                    base_tables=list(compilation.tables),
                )
                return self._failed_result(
                    plan,
                    compilation,
                    invalid_root,
                    invalid_root.error_code,
                    invalid_root.message,
                )

        query_timeout_seconds = max(
            1,
            int(getattr(self.settings, "doris_read_timeout_seconds", 30) or 30),
        )
        if runtime_budget is not None:
            # Clamp as close as possible to the external call. Contract
            # compilation, SQL validation and access checks may already have
            # consumed part of the shared run deadline. Doris accepts an
            # integer timeout, so fail before querying when less than one
            # whole second remains rather than rounding beyond the deadline.
            query_timeout_seconds = max(
                1,
                int(
                    runtime_budget.clamp_timeout_seconds(
                        query_timeout_seconds,
                        minimum_seconds=1.0,
                        operation="doris_query_timeout",
                    )
                ),
            )

        normalized_user_scope = dict(user_scope or {})
        normalized_activation_fingerprint = str(
            expected_semantic_activation_fingerprint or ""
        ).strip()
        try:
            active_data_snapshot = self._prepare_data_snapshot(
                data_snapshot_contract,
                expected_semantic_activation_fingerprint=(
                    normalized_activation_fingerprint
                ),
                governed=bool(
                    artifact_root
                    and str(execution_goal_contract_fingerprint or "").strip()
                ),
            )
            execution_identity_context = (
                self._prepare_execution_identity_context(
                    merchant_id=merchant_id,
                    access_role=access_role,
                    user_scope=normalized_user_scope,
                    reference_scope=execution_reference_scope,
                    context_owner_fingerprint=context_owner_fingerprint,
                    goal_contract_fingerprint=(
                        execution_goal_contract_fingerprint
                    ),
                    plan=plan,
                    compilation=compilation,
                    contract=contract,
                    candidate_validation=candidate_validation,
                    data_snapshot=active_data_snapshot,
                    execution_generation=execution_generation,
                    execution_attempt_id=execution_attempt_id,
                    execution_query_node_id=population_query_node_id,
                    expected_semantic_activation_fingerprint=(
                        normalized_activation_fingerprint
                    ),
                )
                if artifact_root
                and str(execution_goal_contract_fingerprint or "").strip()
                else None
            )
            if execution_identity_context is not None:
                self._require_execution_identity_live(
                    "PRE_EXECUTION",
                    execution_identity_context,
                    context_owner_fingerprint=context_owner_fingerprint,
                    merchant_id=merchant_id,
                    access_role=access_role,
                    user_scope=normalized_user_scope,
                    reference_scope=execution_reference_scope,
                    data_snapshot=active_data_snapshot,
                    execution_generation=execution_generation,
                    execution_attempt_id=execution_attempt_id,
                    goal_contract_fingerprint=(
                        execution_goal_contract_fingerprint
                    ),
                    query_contract_fingerprint=(
                        grounded_query_contract_fingerprint(contract)
                    ),
                    sql_ast_fingerprint=(
                        execution_identity_context.sql_ast_fingerprint
                    ),
                )
            self._authorize_population_pre_execution(
                population_pre_execution_reference,
                contract=contract,
                compilation=compilation,
                plan=plan,
                data_snapshot=active_data_snapshot,
                actual_sql_ast_fingerprint=(
                    execution_identity_context.sql_ast_fingerprint
                    if execution_identity_context is not None
                    else self._sql_evidence_fingerprint(
                        grounded_query_contract_fingerprint(contract),
                        compilation.sql,
                        candidate_validation,
                    )
                ),
                context_owner_fingerprint=context_owner_fingerprint,
                goal_contract_fingerprint=(
                    execution_goal_contract_fingerprint
                ),
                execution_generation=execution_generation,
                execution_attempt_id=execution_attempt_id,
                population_query_node_id=population_query_node_id,
            )
        except Exception as exc:
            error_code = (
                "POPULATION_PRE_EXECUTION_REJECTED"
                if isinstance(exc, GroundedPopulationRuntimeGateError)
                else "QUERY_EXECUTION_IDENTITY_FAILED"
            )
            identity_failure = SqlValidationResult(
                valid=False,
                error_code=error_code,
                message=str(exc)[:500],
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                identity_failure,
                identity_failure.error_code,
                identity_failure.message,
                data_snapshot=(
                    data_snapshot_contract
                    if data_snapshot_contract is not None
                    else DataSnapshotContract()
                ),
            )

        physical_plan_assessment = self._assess_physical_plan(
            contract,
            compilation,
            asset_pack,
            timeout_seconds=query_timeout_seconds,
        )
        if physical_plan_assessment and not bool(
            physical_plan_assessment.get("executable")
        ):
            blocking = next(
                (
                    item
                    for item in physical_plan_assessment.get("gaps") or []
                    if isinstance(item, dict) and item.get("blocking")
                ),
                {},
            )
            code = str(
                blocking.get("code")
                or "PHYSICAL_PLAN_NOT_EXECUTABLE"
            )
            message = str(
                blocking.get("message")
                or "Doris physical-plan governance rejected the query"
            )
            physical_failure = SqlValidationResult(
                valid=False,
                error_code=code,
                message=message,
                base_tables=list(compilation.tables),
            )
            for decision in decisions:
                self.access_control.record_query_audit(
                    decision,
                    row_count=0,
                    status="physical_plan_rejected",
                )
            return self._failed_result(
                plan,
                compilation,
                physical_failure,
                code,
                message,
                data_snapshot=active_data_snapshot,
                physical_plan_assessment=physical_plan_assessment,
            )
        if runtime_budget is not None:
            query_timeout_seconds = max(
                1,
                int(
                    runtime_budget.clamp_timeout_seconds(
                        query_timeout_seconds,
                        minimum_seconds=1.0,
                        operation="doris_query_timeout_after_explain",
                    )
                ),
            )

        masked_columns = {
            key: value
            for decision in decisions
            for key, value in (decision.masked_columns or {}).items()
        }
        if isinstance(candidate_validation, GroundedSqlValidationResult):
            masked_columns = self._candidate_output_masks(
                {
                    "%s.%s" % (access_contract.preferred_table, key): value
                    for access_contract, decision in zip(
                        access_contracts,
                        decisions,
                    )
                    for key, value in (decision.masked_columns or {}).items()
                },
                candidate_validation,
            )
        masked_contract = compilation.node_contract.model_copy(
            update={"masked_columns": masked_columns}
        )
        streamed_rows_receipt: GroundedResultArtifactReceipt | None = None
        fetched_row_count = 0
        artifact_exact_row_count = 0
        started = time.perf_counter()
        try:
            if streaming_artifact_mode:
                if validated_artifact_roots is None:
                    raise RuntimeError(
                        "QUERY_RESULT_ARTIFACT_ROOT_VALIDATION_REQUIRED"
                    )
                _, staging_root = validated_artifact_roots
                limits = self._result_stream_limits()
                stream_state = _GroundedStreamingRowState(
                    preview_limit=limits.preview_rows
                )
                source_batches = self._repository_stream_batches(
                    compilation.sql,
                    batch_size=limits.fetch_batch_rows,
                    cancel_events=cancel_events,
                    timeout_seconds=query_timeout_seconds,
                    data_snapshot_contract=active_data_snapshot,
                )
                transformed_batches = self._masked_stream_batches(
                    source_batches,
                    state=stream_state,
                    node_contract=compilation.node_contract,
                    masked_contract=masked_contract,
                    time_window_role=str(
                        intent.time_range.window_role or "primary"
                    ),
                )
                streamed_rows_receipt = GroundedResultStreamMaterializer(
                    staging_root
                ).materialize_batches(
                    transformed_batches,
                    artifact_id="stream_query_%s" % uuid.uuid4().hex,
                    limits=limits,
                    cancel_events=cancel_events,
                )
                fetched_row_count = streamed_rows_receipt.exact_row_count
                artifact_exact_row_count = fetched_row_count
                rows = [
                    dict(row)
                    for row in streamed_rows_receipt.preview_rows
                ]
                result_raw_rows = list(stream_state.raw_preview_rows)
                (
                    result_coverage,
                    result_is_truncated,
                    exact_result_row_count,
                ) = self._classify_streamed_result(
                    contract,
                    intent,
                    compilation.sql,
                    streamed_rows_receipt,
                    core_sql_candidate=isinstance(
                        candidate_validation,
                        GroundedSqlValidationResult,
                    ),
                )
                entity_filter_verification = (
                    self._candidate_entity_filter_verification(
                        compilation.node_contract,
                        compilation.sql,
                    )
                    if isinstance(
                        candidate_validation,
                        GroundedSqlValidationResult,
                    )
                    else self._streaming_entity_filter_verification(
                        compilation.node_contract,
                        stream_state,
                        compilation.sql,
                    )
                    if contract.query_shape in {"DETAIL", "ENTITY_LOOKUP"}
                    else self._deterministic_metric_filter_verification(
                        compilation.node_contract,
                        compilation.sql,
                        fetched_row_count,
                    )
                )
            else:
                query_kwargs: dict[str, Any] = {
                    "timeout_seconds": query_timeout_seconds,
                }
                query_signature = inspect.signature(
                    self.doris_repository.query
                )
                if "data_snapshot_contract" in query_signature.parameters:
                    query_kwargs["data_snapshot_contract"] = (
                        active_data_snapshot
                    )
                if "semantic_request_fingerprint" in query_signature.parameters:
                    query_kwargs["semantic_request_fingerprint"] = (
                        grounded_query_contract_fingerprint(contract)
                    )
                if "scope_fingerprint" in query_signature.parameters:
                    query_kwargs["scope_fingerprint"] = (
                        str(context_owner_fingerprint or "").strip()
                        or grounded_context_owner_fingerprint(
                            merchant_id,
                            access_role,
                            normalized_user_scope,
                        )
                    )
                raw_rows = [
                    dict(row)
                    for row in self.doris_repository.query(
                        compilation.sql,
                        **query_kwargs,
                    )
                ]
                fetched_row_count = len(raw_rows)
                (
                    result_raw_rows,
                    result_coverage,
                    result_is_truncated,
                    exact_result_row_count,
                ) = self._classify_result_rows(
                    contract,
                    intent,
                    compilation.sql,
                    raw_rows,
                    core_sql_candidate=isinstance(
                        candidate_validation,
                        GroundedSqlValidationResult,
                    ),
                )
                artifact_exact_row_count = len(result_raw_rows)
                entity_filter_verification = (
                    self._candidate_entity_filter_verification(
                        compilation.node_contract,
                        compilation.sql,
                    )
                    if isinstance(
                        candidate_validation,
                        GroundedSqlValidationResult,
                    )
                    else self._entity_filter_verification(
                        compilation.node_contract,
                        result_raw_rows,
                        compilation.sql,
                    )
                    if contract.query_shape in {"DETAIL", "ENTITY_LOOKUP"}
                    else self._deterministic_metric_filter_verification(
                        compilation.node_contract,
                        compilation.sql,
                        len(raw_rows),
                    )
                )
                rows = apply_column_masks(
                    result_raw_rows,
                    masked_contract,
                )
                for row in rows:
                    row.setdefault(
                        "__timeWindowRole",
                        str(intent.time_range.window_role or "primary"),
                    )
        except GroundedResultStreamingError as exc:
            failed = SqlValidationResult(
                valid=False,
                error_code=exc.code.value,
                message=str(exc)[:500],
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                failed,
                failed.error_code,
                failed.message,
                duration_ms=int((time.perf_counter() - started) * 1000),
                data_snapshot=active_data_snapshot,
            )
        except GroundedContextWorkspaceError as exc:
            failed = SqlValidationResult(
                valid=False,
                error_code="QUERY_RESULT_ARTIFACT_ROOT_INVALID",
                message=str(exc)[:500],
                base_tables=list(compilation.tables),
            )
            return self._failed_result(
                plan,
                compilation,
                failed,
                failed.error_code,
                failed.message,
                duration_ms=int((time.perf_counter() - started) * 1000),
                data_snapshot=active_data_snapshot,
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
                data_snapshot=active_data_snapshot,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        task_id = intent.plan_task_id
        pending_artifact_receipt: dict[str, Any] = {}
        if artifact_root:
            try:
                pending_artifact_receipt = self._stage_grounded_result_artifacts(
                    artifact_root=artifact_root,
                    run_id=run_id,
                    task_id=task_id,
                    context_owner_fingerprint=context_owner_fingerprint,
                    contract=contract,
                    compilation=compilation,
                    execution_preparation=execution_preparation,
                    semantic_activation_fingerprint=(
                        normalized_activation_fingerprint
                    ),
                    data_snapshot=active_data_snapshot,
                    rows=rows,
                    streamed_rows_receipt=streamed_rows_receipt,
                    artifact_exact_row_count=artifact_exact_row_count,
                    result_coverage=result_coverage,
                    result_is_truncated=result_is_truncated,
                    exact_result_row_count=exact_result_row_count,
                    execution_generation=execution_generation,
                    execution_attempt_id=execution_attempt_id,
                    execution_identity_context=(
                        execution_identity_context
                    ),
                    cancel_events=cancel_events,
                )
            except Exception as exc:
                message = "%s:%s" % (type(exc).__name__, str(exc)[:300])
                artifact_validation = SqlValidationResult(
                    valid=False,
                    error_code="QUERY_RESULT_ARTIFACT_STAGING_FAILED",
                    message=message,
                    base_tables=list(compilation.tables),
                )
                for decision in decisions:
                    self.access_control.record_query_audit(
                        decision,
                        row_count=fetched_row_count,
                        status="artifact_persistence_failed",
                    )
                return self._failed_result(
                    plan,
                    compilation,
                    artifact_validation,
                    artifact_validation.error_code,
                    artifact_validation.message,
                    duration_ms=duration_ms,
                    data_snapshot=active_data_snapshot,
                )
        bundle = QueryBundle(
            sql=compilation.sql,
            tables=list(compilation.tables),
            rows=rows,
            # Execution only creates server-private staging. No path or
            # reference becomes Core-visible before independent verification.
            offloaded_files=[],
            original_row_count=exact_result_row_count,
            is_truncated=result_is_truncated,
            result_coverage=result_coverage,
            source_row_counts={task_id: fetched_row_count},
            source_artifact_refs={},
            duration_ms=duration_ms,
            cache_hit=bool(getattr(self.doris_repository, "last_cache_hit", False)),
            cache_key=str(getattr(self.doris_repository, "last_cache_key", "") or ""),
            data_snapshot=active_data_snapshot,
            runtime_events=[
                {
                    "event": "grounded_data_engine.executed",
                    "status": "success",
                    "taskId": task_id,
                    "table": compilation.table,
                    "rowCount": len(rows),
                    "previewRowCount": len(rows),
                    "fetchedRowCount": fetched_row_count,
                    "resultCoverage": result_coverage,
                    "resultArtifactCoverage": (
                        "ALL_ROWS"
                        if streamed_rows_receipt is not None
                        else "INLINE_RESULT"
                    ),
                    "resultArtifactComplete": bool(
                        streamed_rows_receipt is not None
                    ),
                    "resultRowCountExact": result_coverage
                    in {
                        ResultCoverage.ALL_ROWS.value,
                        ResultCoverage.TOP_N.value,
                    },
                    "plannerLlmCalls": 0,
                    **(
                        {
                            "_serverPrivatePendingResultArtifact": (
                                pending_artifact_receipt
                            )
                        }
                        if pending_artifact_receipt
                        else {}
                    ),
                    "sqlLlmCalls": (
                        1
                        if isinstance(candidate_validation, GroundedSqlValidationResult)
                        else 0
                    ),
                    **(
                        {
                            "physicalPlanAssessmentId": str(
                                physical_plan_assessment.get("assessmentId")
                                or ""
                            ),
                            "physicalPlanStatus": str(
                                physical_plan_assessment.get("status") or ""
                            ),
                        }
                        if physical_plan_assessment
                        else {}
                    ),
                }
            ],
        )
        task_result = AgentTaskResult(
            task_id=task_id,
            sub_agent_type="GROUNDED_DATA_ENGINE",
            success=True,
            summary=(
                "Grounded Data Engine executed %d row(s); retained %d preview row(s)"
                % (fetched_row_count, len(rows))
            ),
            query_bundle=bundle,
            validation_results=[validation],
            react_trace=[
                ReActStep(
                    round=1,
                    reason=(
                        "Execute the Core-authored SQL validated against the active GroundedQueryContract"
                        if isinstance(candidate_validation, GroundedSqlValidationResult)
                        else "Execute the activated GroundedQueryContract deterministically"
                    ),
                    action=(
                        "grounded_data_engine.execute_core_sql"
                        if isinstance(candidate_validation, GroundedSqlValidationResult)
                        else "grounded_data_engine.execute_sql"
                    ),
                    observation="table=%s;rows=%d;preview=%d"
                    % (compilation.table, fetched_row_count, len(rows)),
                )
            ],
            node_task_profile=NodeTaskProfile(
                task_id=task_id,
                task_kind="GROUNDED_DATA_ENGINE",
                sql_strategy=(
                    "core_llm_grounded_sql"
                    if isinstance(candidate_validation, GroundedSqlValidationResult)
                    else "grounded_deterministic"
                ),
                selected_tools=[
                    "compile_grounded_sql",
                    "validate_grounded_sql",
                    "authorize_grounded_query",
                    "execute_doris",
                ],
                reason=(
                    "Core authored the complete SQL; the runtime only validated and scoped it"
                    if isinstance(candidate_validation, GroundedSqlValidationResult)
                    else "READY GroundedQueryContract is compiled without a NodeAgent"
                ),
                risk_controls=[
                    "contract_bound_table",
                    "contract_bound_columns",
                    "published_metric_formula",
                    "tenant_scope",
                    "read_only_sql",
                ],
                contract_status="passed",
                sql_draft_source=(
                    "core_llm"
                    if isinstance(candidate_validation, GroundedSqlValidationResult)
                    else "grounded_deterministic"
                ),
            ),
            node_plan_contract=compilation.node_contract,
            entity_filter_verification=entity_filter_verification,
            entity_set=self._sealed_raw_entity_outputs(
                contract,
                result_raw_rows,
                task_id,
                source_row_count=fetched_row_count,
                force_truncated=fetched_row_count > len(result_raw_rows),
            ),
        )
        for decision in decisions:
            self.access_control.record_query_audit(
                decision,
                row_count=fetched_row_count,
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
            # These are independent bundle projections over one immutable row
            # population.  Copying the full rows for task/query/merged views
            # would triple resident result memory before artifact offload.
            query_bundles=[bundle.model_copy(deep=False)],
            merged_query_bundle=bundle.model_copy(deep=False),
            evidence_check=EvidenceCheckResult(
                passed=False,
                summary=(
                    "Grounded Data Engine execution completed; independent EvidenceVerifier pending"
                ),
            ),
            node_task_profiles=[task_result.node_task_profile.model_copy(deep=True)],
            node_plan_contracts=[
                item.model_copy(deep=True)
                for item in (
                    compilation.access_contracts
                    or (compilation.node_contract,)
                )
            ],
            physical_plan_assessment=physical_plan_assessment,
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
        complete_result_artifact: bool = False,
    ) -> GroundedSqlCompilation:
        if (
            contract.upstream_entity_bindings
            or contract.binding_hints.upstream_entity_bindings
        ):
            raise RuntimeError(
                "upstream entity dependencies remain Core SQL and serial-chain owned"
            )
        if contract.query_shape in {"DETAIL", "ENTITY_LOOKUP"}:
            return self._compile_detail_sql(
                merchant_id,
                contract,
                plan,
                asset_pack,
                access_role=access_role,
                user_scope=user_scope,
                complete_result_artifact=complete_result_artifact,
            )
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
        primary_group_columns: list[str] = []
        if group_bindings:
            dimension = group_bindings[0]
            if dimension.table != table or dimension.column not in columns:
                raise RuntimeError("grounded group dimension is outside the execution table")
            if dimension.column == table_binding.merchant_filter_column:
                raise RuntimeError("merchant scope column cannot be a business dimension")
            group_columns.append(dimension.column)
            primary_group_columns.append(dimension.column)
            required_columns.append(dimension.column)

        selected_projections: list[tuple[str, str]] = []
        if contract.query_shape == "RANKED":
            selected_aliases: set[str] = set()
            for field in contract.selected_fields:
                output_alias = str(field.output_alias or field.column or "").strip()
                if (
                    field.table != table
                    or field.column not in columns
                    or not output_alias
                    or len(output_alias) > 128
                    or any(ord(character) < 32 for character in output_alias)
                ):
                    raise RuntimeError(
                        "grounded ranked label field is outside the one-table deterministic projection"
                    )
                if output_alias in selected_aliases or output_alias in {
                    *metric_aliases,
                    *primary_group_columns,
                }:
                    raise RuntimeError(
                        "grounded ranked output aliases must be non-empty and unique"
                    )
                selected_aliases.add(output_alias)
                if field.column == table_binding.merchant_filter_column:
                    raise RuntimeError(
                        "merchant scope column cannot be a ranked output label"
                    )
                if field.column not in group_columns:
                    group_columns.append(field.column)
                selected_projections.append((field.column, output_alias))
                required_columns.append(field.column)

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
                    ", ".join(sql_literal(item) for item in store_ids),
                )
            )
            required_columns.append(store_column)

        time_column = self._time_column(
            contract,
            table,
            table_binding.time_column,
        )
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
        pruning_column = self._partition_pruning_column(contract, table)
        if pruning_column and pruning_column != time_column:
            if pruning_column not in columns:
                raise RuntimeError(
                    "declared partition pruning column is absent from projected schema"
                )
            pruning_predicate = self._partition_pruning_predicate(
                contract,
                table,
                pruning_column,
            )
            if pruning_predicate:
                where.append(pruning_predicate)
                required_columns.append(pruning_column)

        for entity_filter in contract.entity_filters:
            if entity_filter.table != table:
                raise RuntimeError(
                    "grounded deterministic metric filter is outside the execution table"
                )
            if entity_filter.column not in columns:
                raise RuntimeError(
                    "grounded deterministic metric filter field is absent from projected schema"
                )
            if entity_filter.operator not in set(entity_filter.allowed_operators):
                raise RuntimeError(
                    "grounded deterministic metric filter operator is not declared"
                )
            where.append(
                self._literal_filter_predicate(
                    entity_filter.column,
                    entity_filter.operator,
                    entity_filter.literal_value,
                )
            )
            required_columns.append(entity_filter.column)

        if group_columns:
            group_column = group_columns[0]
            where.extend(
                [
                    "%s IS NOT NULL" % quote_identifier(group_column),
                    "%s != ''" % quote_identifier(group_column),
                ]
            )

        select_parts = [
            quote_identifier(column) for column in primary_group_columns
        ]
        select_parts.extend(
            "%s AS %s"
            % (quote_identifier(column), quote_identifier(output_alias))
            for column, output_alias in selected_projections
        )
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
            ranking_direction = str(contract.ranking.direction or "").strip().upper()
            if ranking_direction not in {"ASC", "DESC"}:
                raise RuntimeError("grounded ranking direction is not executable")
            sql += " ORDER BY %s %s LIMIT %d" % (
                quote_identifier(ranking_metric),
                ranking_direction,
                max(1, int(contract.ranking.limit or 1)),
            )
        # Unranked GROUPED/TREND results are evidence-complete by default.
        # Applying an unordered LIMIT here silently drops valid groups/time
        # buckets and makes a deterministic query incomplete. Bounded Top-N
        # remains explicit through the governed RANKED Contract above.

        required = tuple(self._dedupe(required_columns))
        allowed_columns = self._dedupe([*columns])
        primary_filter = contract.entity_filters[0] if contract.entity_filters else None
        node_contract = NodePlanContract(
            task_id=intent.plan_task_id,
            question=contract.question,
            preferred_table=table,
            allowed_columns=allowed_columns,
            visible_columns=[
                *primary_group_columns,
                *(output_alias for _, output_alias in selected_projections),
            ],
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
            filter_column=primary_filter.column if primary_filter else "",
            filter_values=self._entity_filter_values(contract.entity_filters),
            entity_filter_obligations=[
                item.model_copy(deep=True)
                for item in plan.entity_filter_obligations
            ],
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
            access_role=access_role or DEFAULT_ACCESS_ROLE,
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
                "calendarAnchorPolicy": contract.time_range.calendar_anchor_policy,
                "dataAsOfPolicy": contract.time_range.data_as_of_policy,
                "partitionColumn": time_column,
                "tenantColumn": merchant_column,
            }
            if time_column
            else {},
        )
        return GroundedSqlCompilation(
            sql=sql,
            table=table,
            tables=(table,),
            metric_aliases=tuple(metric_aliases),
            group_columns=tuple(group_columns),
            required_columns=required,
            node_contract=node_contract,
            access_contracts=(node_contract,),
        )

    def compile_core_sql_candidate(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: Any,
        validation: GroundedSqlValidationResult,
        *,
        access_role: str,
        user_scope: dict[str, Any],
    ) -> GroundedSqlCompilation:
        """Bind trusted execution scope around an already validated Core SQL AST.

        Core owns the complete SQL topology. This method never adds a business
        aggregate, grouping, ranking, join, window, CTE, or fallback template.
        It only injects runtime access predicates and projects the accepted SQL
        lineage into the existing ACL/evidence contracts.
        """

        if not validation.valid or not validation.canonical_sql:
            raise RuntimeError("Core SQL candidate has not passed grounded AST validation")
        if len(plan.intents) != 1:
            raise RuntimeError("Core SQL execution requires one evidence intent")
        referenced_tables = self._dedupe(validation.referenced_tables)
        if not referenced_tables:
            raise RuntimeError("Core SQL candidate has no grounded base table")
        table_bindings = {item.table: item for item in contract.tables}
        missing_bindings = [
            table for table in referenced_tables if table not in table_bindings
        ]
        if missing_bindings:
            raise RuntimeError(
                "Core SQL references tables outside the active Contract: %s"
                % ",".join(missing_bindings)
            )

        scoped_sql = self._inject_candidate_execution_scope(
            validation.canonical_sql,
            merchant_id,
            contract,
            asset_pack,
            user_scope,
        )
        intent = plan.intents[0]
        primary = (
            contract.primary_table
            if contract.primary_table in referenced_tables
            else referenced_tables[0]
        )
        referenced_by_table: dict[str, list[str]] = {
            table: [] for table in referenced_tables
        }
        for qualified in validation.referenced_columns:
            table, separator, column = str(qualified or "").partition(".")
            if separator and table in referenced_by_table and column:
                referenced_by_table[table].append(column)
        visible_by_table: dict[str, list[str]] = {
            table: [] for table in referenced_tables
        }
        for lineage in validation.output_lineage.values():
            for qualified in lineage:
                table, separator, column = str(qualified or "").partition(".")
                if separator and table in visible_by_table and column:
                    visible_by_table[table].append(column)

        obligations = self._candidate_entity_filter_obligations(
            contract,
            intent.plan_task_id,
        )
        store_ids = self._dedupe(
            user_scope.get("storeIds") or user_scope.get("store_ids") or []
        )
        region = str(user_scope.get("region") or "").strip()
        access_contracts: list[NodePlanContract] = []
        for table in referenced_tables:
            binding = table_bindings[table]
            metadata = self._table_metadata(asset_pack, table)
            columns = self._dedupe(asset_pack.known_columns(table))
            if not columns:
                raise RuntimeError(
                    "Core SQL table has no progressively disclosed schema: %s" % table
                )
            merchant_column = str(binding.merchant_filter_column or "").strip()
            if not merchant_column or merchant_column not in columns:
                raise RuntimeError(
                    "Core SQL table has no governed merchant scope column: %s" % table
                )
            region_column = str(metadata.get("regionFilterColumn") or "").strip()
            store_column = str(metadata.get("storeFilterColumn") or "").strip()
            if region and (not region_column or region_column not in columns):
                raise RuntimeError(
                    "authorized region scope has no declared semantic filter column: %s"
                    % table
                )
            if store_ids and (not store_column or store_column not in columns):
                raise RuntimeError(
                    "authorized store scope has no declared semantic filter column: %s"
                    % table
                )
            required = self._dedupe(
                [
                    *referenced_by_table[table],
                    *[
                        item.column
                        for item in contract.entity_filters
                        if item.table == table
                    ],
                    merchant_column,
                    region_column if region else "",
                    store_column if store_ids else "",
                ]
            )
            visible = self._dedupe(visible_by_table[table])
            table_filter = next(
                (item for item in contract.entity_filters if item.table == table),
                None,
            )
            filter_output_alias = ""
            if table_filter is not None:
                expected = "%s.%s" % (table_filter.table, table_filter.column)
                filter_output_alias = next(
                    (
                        alias
                        for alias, lineage in validation.output_lineage.items()
                        if expected in lineage
                    ),
                    "",
                )
            table_metric_specs = [
                dict(item)
                for item in intent.metric_specs
                if str(
                    item.get("ownerTable")
                    or item.get("owner_table")
                    or intent.preferred_table
                    or ""
                )
                == table
            ]
            access_contracts.append(
                NodePlanContract(
                    task_id=intent.plan_task_id,
                    question=contract.question,
                    preferred_table=table,
                    allowed_columns=columns,
                    visible_columns=visible,
                    internal_only_columns=[
                        column for column in required if column not in visible
                    ],
                    required_columns=required,
                    metric_column=(
                        intent.metric_column if intent.preferred_table == table else ""
                    ),
                    metric_name=(
                        intent.metric_name if intent.preferred_table == table else ""
                    ),
                    metric_formula=(
                        intent.metric_formula if intent.preferred_table == table else ""
                    ),
                    metric_specs=(
                        [dict(item) for item in intent.metric_specs]
                        if table == primary
                        else table_metric_specs
                    ),
                    group_by_column=(
                        intent.group_by_column
                        if intent.group_by_column in visible
                        else ""
                    ),
                    filter_column=filter_output_alias,
                    filter_values=self._entity_filter_values(
                        item
                        for item in contract.entity_filters
                        if table == primary or item.table == table
                    ),
                    entity_filter_obligations=[
                        item
                        for item in obligations
                        if table == primary or item.reference.table == table
                    ],
                    output_keys=(
                        list(validation.output_columns) if table == primary else []
                    ),
                    required_evidence=(
                        list(validation.output_columns) if table == primary else []
                    ),
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
                    access_role=access_role or DEFAULT_ACCESS_ROLE,
                    row_scope_policy=normalize_row_access_policy(
                        metadata.get("rowAccessPolicy")
                        or default_row_access_policy(merchant_column)
                    ),
                    answer_mode=str(intent.answer_mode),
                    task_role=str(intent.task_role),
                    sql_strategy="core_llm_grounded_sql",
                    metric_resolution=(
                        dict(intent.metric_resolution or {})
                        if intent.preferred_table == table
                        else {}
                    ),
                    metric_governance_mode="grounded_query_contract",
                    time_window_contract={
                        "kind": contract.time_range.kind,
                        "label": contract.time_range.label,
                        "days": contract.time_range.days,
                        "startDate": contract.time_range.start_date,
                        "endDate": contract.time_range.end_date,
                        "calendarAnchorPolicy": contract.time_range.calendar_anchor_policy,
                        "dataAsOfPolicy": contract.time_range.data_as_of_policy,
                        "partitionColumn": binding.time_column,
                        "tenantColumn": merchant_column,
                    }
                    if binding.time_column
                    else {},
                )
            )
        primary_contract = next(
            item for item in access_contracts if item.preferred_table == primary
        )
        return GroundedSqlCompilation(
            sql=scoped_sql,
            table=primary,
            tables=tuple(referenced_tables),
            metric_aliases=tuple(metric.metric_key for metric in contract.metrics),
            group_columns=tuple(
                item.column
                for item in contract.dimensions
                if item.usage == "group_by"
            ),
            required_columns=tuple(
                self._dedupe(
                    column
                    for table in referenced_tables
                    for column in referenced_by_table[table]
                )
            ),
            node_contract=primary_contract,
            access_contracts=tuple(access_contracts),
        )

    def _inject_candidate_execution_scope(
        self,
        sql: str,
        merchant_id: str,
        contract: GroundedQueryContract,
        asset_pack: Any,
        user_scope: dict[str, Any],
    ) -> str:
        """Inject trusted access and upstream entity predicates.

        Core owns query semantics and topology. The kernel owns secret runtime
        values and injects them exactly like tenant scope, including into the
        ON clause of a RIGHT-side table in a LEFT JOIN.
        """

        if not str(merchant_id or "").strip():
            raise RuntimeError("trusted execution scope is missing merchant_id")
        try:
            parsed = sqlglot.parse_one(sql, read="doris")
        except Exception as exc:
            raise RuntimeError("accepted Core SQL could not be reparsed") from exc
        bindings = {item.table: item for item in contract.tables}
        upstream_refs = {
            item.target_field_ref
            for item in contract.upstream_entity_bindings
        }
        upstream_filters_by_table: dict[str, list[Any]] = {}
        for item in contract.entity_filters:
            if item.semantic_ref_id in upstream_refs:
                upstream_filters_by_table.setdefault(item.table, []).append(item)
        region = str(user_scope.get("region") or "").strip()
        store_ids = self._dedupe(
            user_scope.get("storeIds") or user_scope.get("store_ids") or []
        )
        try:
            scopes = list(traverse_scope(parsed))
        except Exception as exc:
            raise RuntimeError("Core SQL scope injection could not resolve aliases") from exc
        for scope in scopes:
            select = scope.expression
            if not isinstance(select, exp.Select):
                continue
            for join in select.args.get("joins") or []:
                side = str(join.args.get("side") or "").strip().upper()
                if side in {"RIGHT", "FULL"}:
                    raise RuntimeError(
                        "RIGHT/FULL JOIN cannot receive fail-closed tenant scope injection; "
                        "rewrite it as an equivalent governed INNER/LEFT JOIN"
                    )
            where_predicates: list[exp.Expression] = []
            for raw_alias, pair in (
                getattr(scope, "selected_sources", {}) or {}
            ).items():
                source = pair[1]
                if not isinstance(source, exp.Table):
                    continue
                table = str(source.name or "").strip()
                binding = bindings.get(table)
                if binding is None:
                    continue
                alias = str(raw_alias or source.alias_or_name or table).strip()
                columns = set(asset_pack.known_columns(table))
                merchant_column = str(binding.merchant_filter_column or "").strip()
                if not merchant_column or merchant_column not in columns:
                    raise RuntimeError(
                        "grounded table lacks an injectable merchant scope: %s" % table
                    )
                predicates: list[exp.Expression] = [
                    exp.EQ(
                        this=exp.column(merchant_column, table=alias),
                        expression=exp.Literal.string(str(merchant_id)),
                    )
                ]
                for entity_filter in upstream_filters_by_table.get(table, []):
                    if entity_filter.column not in columns:
                        raise RuntimeError(
                            "upstream entity field is absent from grounded schema: %s.%s"
                            % (table, entity_filter.column)
                        )
                    if entity_filter.operator != "IN" or not isinstance(
                        entity_filter.literal_value,
                        (list, tuple),
                    ) or not entity_filter.literal_value:
                        raise RuntimeError(
                            "upstream entity binding requires a non-empty typed IN set"
                        )
                    predicates.append(
                        exp.In(
                            this=exp.column(entity_filter.column, table=alias),
                            expressions=[
                                exp.convert(value)
                                for value in entity_filter.literal_value
                            ],
                        )
                    )
                metadata = self._table_metadata(asset_pack, table)
                if region:
                    region_column = str(
                        metadata.get("regionFilterColumn") or ""
                    ).strip()
                    if not region_column or region_column not in columns:
                        raise RuntimeError(
                            "authorized region scope cannot be injected for table %s"
                            % table
                        )
                    predicates.append(
                        exp.EQ(
                            this=exp.column(region_column, table=alias),
                            expression=exp.Literal.string(region),
                        )
                    )
                if store_ids:
                    store_column = str(
                        metadata.get("storeFilterColumn") or ""
                    ).strip()
                    if not store_column or store_column not in columns:
                        raise RuntimeError(
                            "authorized store scope cannot be injected for table %s"
                            % table
                        )
                    predicates.append(
                        exp.In(
                            this=exp.column(store_column, table=alias),
                            expressions=[
                                exp.Literal.string(value) for value in store_ids
                            ],
                        )
                    )
                predicate = self._and_expressions(predicates)
                parent = source.parent
                if isinstance(parent, exp.Join) and parent.this is source:
                    side = str(parent.args.get("side") or "").strip().upper()
                    if side == "LEFT":
                        on = parent.args.get("on")
                        if not isinstance(on, exp.Expression):
                            raise RuntimeError(
                                "LEFT JOIN is missing its governed ON predicate"
                            )
                        parent.set("on", exp.and_(on, predicate))
                        continue
                where_predicates.append(predicate)
            if where_predicates:
                injected = self._and_expressions(where_predicates)
                existing = select.args.get("where")
                if isinstance(existing, exp.Where):
                    injected = exp.and_(existing.this, injected)
                select.set("where", exp.Where(this=injected))
        return parsed.sql(
            dialect="doris",
            pretty=False,
            normalize=False,
            comments=False,
        )

    @staticmethod
    def _sealed_raw_entity_outputs(
        contract: GroundedQueryContract,
        raw_rows: list[dict[str, Any]],
        task_id: str,
        *,
        source_row_count: int | None = None,
        force_truncated: bool = False,
    ) -> EntitySet | None:
        """Retain pre-mask entity outputs for later verified publication."""

        entity_outputs: dict[str, str] = {}
        for dimension in contract.dimensions:
            if (
                dimension.usage == "group_by"
                and dimension.column
                and dimension.entity_identity
            ):
                entity_outputs[dimension.column] = dimension.entity_identity
        for field in contract.selected_fields:
            output = field.output_alias or field.column
            if output and field.entity_identity:
                entity_outputs[output] = field.entity_identity
        if not entity_outputs:
            return None
        column_values: dict[str, list[Any]] = {}
        truncated = bool(force_truncated)
        for output in entity_outputs:
            values_by_key: dict[str, Any] = {}
            for row in raw_rows:
                if output not in row or row.get(output) is None:
                    continue
                value = row.get(output)
                key = "%s:%r" % (type(value).__name__, value)
                values_by_key.setdefault(key, value)
            ordered = [values_by_key[key] for key in sorted(values_by_key)]
            if len(ordered) > 5000:
                truncated = True
                ordered = ordered[:5000]
            column_values[output] = ordered
        first_column = next(iter(entity_outputs), "")
        return EntitySet(
            task_id=task_id,
            join_key=first_column,
            values=list(column_values.get(first_column) or []),
            column_values=column_values,
            truncated=truncated,
            source_row_count=(
                len(raw_rows)
                if source_row_count is None
                else max(0, int(source_row_count))
            ),
            source_key="grounded_pre_mask_verified_candidate",
        )

    @staticmethod
    def _and_expressions(expressions: Iterable[exp.Expression]) -> exp.Expression:
        items = [item for item in expressions if isinstance(item, exp.Expression)]
        if not items:
            raise RuntimeError("execution scope predicate is empty")
        result = items[0]
        for item in items[1:]:
            result = exp.and_(result, item)
        return result

    @staticmethod
    def _candidate_entity_filter_obligations(
        contract: GroundedQueryContract,
        task_id: str,
    ) -> list[EntityFilterObligation]:
        obligations: list[EntityFilterObligation] = []
        for index, item in enumerate(contract.entity_filters):
            values = (
                list(item.literal_value)
                if item.operator == "IN"
                and isinstance(item.literal_value, (list, tuple))
                else [item.literal_value]
            )
            obligations.append(
                EntityFilterObligation(
                    obligation_id="grounded_core_sql_entity_%d" % (index + 1),
                    task_id=task_id,
                    required=True,
                    reference=EntityReference(
                        semantic_ref_id=item.semantic_ref_id,
                        field=item.column,
                        table=item.table,
                        raw_label=item.requested_phrase,
                        raw_value=str(item.literal_value),
                        values=values,
                        comparison_policy=item.operator.lower(),
                        source="grounded_core_sql_candidate",
                        confidence=1.0,
                        status="bound",
                        time_scope_explicit=bool(contract.time_range.explicit),
                        lookup_time_policy=dict(item.lookup_time_policy),
                    ),
                    status="bound",
                    reason="validated mandatory predicate in Core SQL AST",
                )
            )
        return obligations

    @staticmethod
    def _candidate_output_masks(
        qualified_masks: dict[str, str],
        validation: GroundedSqlValidationResult,
    ) -> dict[str, str]:
        output: dict[str, str] = {}
        strength = {"partial": 1, "hash": 2, "full": 3}
        for alias, lineage in validation.output_lineage.items():
            strategies = [
                str(qualified_masks.get(item) or "").strip().lower()
                for item in lineage
                if str(qualified_masks.get(item) or "").strip()
            ]
            if not strategies:
                continue
            output[alias] = max(
                strategies,
                key=lambda item: strength.get(item, 0),
            )
        return output

    @staticmethod
    def _candidate_entity_filter_verification(
        contract: NodePlanContract,
        sql: str,
    ) -> EntityFilterVerificationProof:
        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations:
            return EntityFilterVerificationProof(
                task_id=contract.task_id,
                status="not_required",
            )
        contract_hash = entity_filter_contract_hash(contract)
        requested = {
            value
            for obligation in obligations
            for value in canonical_entity_values(
                obligation.reference.values,
                entity_comparison_policy(obligation.reference),
            )
        }
        return EntityFilterVerificationProof(
            task_id=contract.task_id,
            obligation_id=obligations[0].obligation_id,
            status="verified",
            verified=True,
            coverage_complete=True,
            contract_hash=contract_hash,
            sql_hash=entity_filter_sql_hash(sql),
            requested_value_hashes=sorted(
                entity_value_hash(item, contract_hash) for item in requested
            ),
            row_count=0,
            reason="Grounded SQL AST proved each typed entity predicate mandatory before execution",
        )

    def _compile_detail_sql(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: Any,
        *,
        access_role: str,
        user_scope: dict[str, Any],
        complete_result_artifact: bool = False,
    ) -> GroundedSqlCompilation:
        intent = plan.intents[0]
        if contract.metrics:
            raise RuntimeError("grounded detail execution cannot contain metrics")
        tables = self._dedupe(
            [contract.primary_table]
            + [item.table for item in contract.selected_fields]
            + [item.table for item in contract.entity_filters]
        )
        if not tables or len(tables) > 2:
            raise RuntimeError("grounded detail execution supports one or two tables")
        table_bindings = {item.table: item for item in contract.tables}
        if any(table not in table_bindings for table in tables):
            raise RuntimeError("detail table is not bound by the Contract")
        columns_by_table = {
            table: set(asset_pack.known_columns(table)) for table in tables
        }
        if any(not columns for columns in columns_by_table.values()):
            raise RuntimeError("detail table has no projected schema")
        aliases = {table: "t%d" % index for index, table in enumerate(tables)}

        output_aliases: set[str] = set()
        select_parts: list[str] = []
        required_by_table: dict[str, list[str]] = {table: [] for table in tables}
        for field_binding in contract.selected_fields:
            if field_binding.table not in columns_by_table:
                raise RuntimeError("selected field table is outside detail execution scope")
            if field_binding.column not in columns_by_table[field_binding.table]:
                raise RuntimeError("selected detail field is absent from projected schema")
            output_alias = str(
                field_binding.output_alias or field_binding.column
            ).strip()
            if not output_alias or output_alias in output_aliases:
                raise RuntimeError("detail output aliases must be non-empty and unique")
            output_aliases.add(output_alias)
            select_parts.append(
                "%s.%s AS %s"
                % (
                    aliases[field_binding.table],
                    quote_identifier(field_binding.column),
                    quote_identifier(output_alias),
                )
            )
            required_by_table[field_binding.table].append(field_binding.column)
        if not select_parts:
            raise RuntimeError("detail execution requires exact selected fields")

        primary = tables[0]
        from_sql = "%s %s" % (quote_identifier(primary), aliases[primary])
        join_scoped_tables: set[str] = set()
        if len(tables) == 2:
            secondary = tables[1]
            relationship_candidates = grounded_detail_relationship_candidates(
                primary,
                set(tables),
                contract.relationships,
            )
            if len(relationship_candidates) != 1:
                raise RuntimeError(
                    "detail join requires exactly one direction-safe relationship proof"
                )
            relationship = relationship_candidates[0]
            join_type = str(relationship.join_type or "INNER").upper()
            predicates: list[str] = []
            for left_column, right_column in relationship.keys:
                if left_column not in columns_by_table[relationship.left_table]:
                    raise RuntimeError("relationship left key is absent from projected schema")
                if right_column not in columns_by_table[relationship.right_table]:
                    raise RuntimeError("relationship right key is absent from projected schema")
                predicates.append(
                    "%s.%s = %s.%s"
                    % (
                        aliases[relationship.left_table],
                        quote_identifier(left_column),
                        aliases[relationship.right_table],
                        quote_identifier(right_column),
                    )
                )
                required_by_table[relationship.left_table].append(left_column)
                required_by_table[relationship.right_table].append(right_column)
            if not predicates:
                raise RuntimeError("detail relationship has no join keys")
            if join_type == "LEFT":
                secondary_binding = table_bindings[secondary]
                secondary_merchant = str(
                    secondary_binding.merchant_filter_column or ""
                ).strip()
                if (
                    not secondary_merchant
                    or secondary_merchant not in columns_by_table[secondary]
                ):
                    raise RuntimeError(
                        "detail secondary table has no governed merchant scope column"
                    )
                predicates.append(
                    "%s.%s = %s"
                    % (
                        aliases[secondary],
                        quote_identifier(secondary_merchant),
                        sql_literal(merchant_id),
                    )
                )
                required_by_table[secondary].append(secondary_merchant)
                for time_predicate, time_column in self._detail_time_predicates(
                    contract,
                    secondary,
                    secondary_binding.time_column,
                    aliases[secondary],
                ):
                    if time_column not in columns_by_table[secondary]:
                        raise RuntimeError(
                            "detail time/pruning field is absent from projected schema"
                        )
                    predicates.append(time_predicate)
                    required_by_table[secondary].append(time_column)
                join_scoped_tables.add(secondary)
            from_sql += " %s JOIN %s %s ON %s" % (
                join_type,
                quote_identifier(secondary),
                aliases[secondary],
                " AND ".join(predicates),
            )

        where: list[str] = []
        for entity_filter in contract.entity_filters:
            if entity_filter.table not in columns_by_table:
                raise RuntimeError("entity filter table is outside detail execution scope")
            if entity_filter.column not in columns_by_table[entity_filter.table]:
                raise RuntimeError("entity filter field is absent from projected schema")
            if entity_filter.operator not in set(entity_filter.allowed_operators):
                raise RuntimeError("entity filter operator is not declared by field semantics")
            where.append(
                self._detail_filter_predicate(
                    aliases[entity_filter.table],
                    entity_filter.column,
                    entity_filter.operator,
                    entity_filter.literal_value,
                )
            )
            required_by_table[entity_filter.table].append(entity_filter.column)

        for table in tables:
            binding = table_bindings[table]
            merchant_column = str(binding.merchant_filter_column or "").strip()
            if not merchant_column or merchant_column not in columns_by_table[table]:
                raise RuntimeError("detail table has no governed merchant scope column")
            if table not in join_scoped_tables:
                where.append(
                    "%s.%s = %s"
                    % (
                        aliases[table],
                        quote_identifier(merchant_column),
                        sql_literal(merchant_id),
                    )
                )
            required_by_table[table].append(merchant_column)
            if table not in join_scoped_tables:
                for time_predicate, time_column in self._detail_time_predicates(
                    contract,
                    table,
                    binding.time_column,
                    aliases[table],
                ):
                    if time_column not in columns_by_table[table]:
                        raise RuntimeError(
                            "detail time/pruning field is absent from projected schema"
                        )
                    where.append(time_predicate)
                    required_by_table[table].append(time_column)

        sql = "SELECT %s FROM %s" % (", ".join(select_parts), from_sql)
        if where:
            sql += " WHERE " + " AND ".join("(%s)" % item for item in where)
        if not complete_result_artifact:
            # Legacy inline execution fetches one sentinel beyond the display
            # cap. In complete artifact mode the cursor must reach source EOF;
            # the display limit applies only to the bounded Core projection.
            sql += " LIMIT %d" % (self._detail_display_limit(intent) + 1)

        access_contracts: list[NodePlanContract] = []
        for table in tables:
            binding = table_bindings[table]
            metadata = self._table_metadata(asset_pack, table)
            visible = [
                item.column for item in contract.selected_fields if item.table == table
            ]
            required = self._dedupe(required_by_table[table])
            table_filter = next(
                (item for item in contract.entity_filters if item.table == table),
                None,
            )
            filter_output_alias = ""
            if table_filter is not None:
                filter_output_alias = next(
                    (
                        item.output_alias or item.column
                        for item in contract.selected_fields
                        if item.table == table
                        and item.column == table_filter.column
                    ),
                    table_filter.column,
                )
            access_contracts.append(
                NodePlanContract(
                    task_id=intent.plan_task_id,
                    question=contract.question,
                    preferred_table=table,
                    allowed_columns=self._dedupe(columns_by_table[table]),
                    visible_columns=visible,
                    internal_only_columns=[
                        column for column in required if column not in visible
                    ],
                    required_columns=required,
                    output_keys=visible,
                    required_evidence=visible,
                    filter_column=filter_output_alias,
                    filter_values=self._entity_filter_values(
                        item
                        for item in contract.entity_filters
                        if item.table == table
                    ),
                    entity_filter_obligations=[
                        item
                        for item in plan.entity_filter_obligations
                        if item.reference.table == table
                    ],
                    days=int(intent.days or 0),
                    limit=int(intent.limit or 100),
                    merchant_id=merchant_id,
                    merchant_filter_column=binding.merchant_filter_column,
                    effective_user_id=str(
                        user_scope.get("userId") or user_scope.get("user_id") or ""
                    ),
                    access_role=access_role or DEFAULT_ACCESS_ROLE,
                    row_scope_policy=normalize_row_access_policy(
                        metadata.get("rowAccessPolicy")
                        or default_row_access_policy(binding.merchant_filter_column)
                    ),
                    answer_mode=str(intent.answer_mode),
                    task_role=str(intent.task_role),
                    sql_strategy="grounded_deterministic",
                    metric_governance_mode="grounded_query_contract",
                )
            )
        primary_contract = next(
            item for item in access_contracts if item.preferred_table == primary
        )
        return GroundedSqlCompilation(
            sql=sql,
            table=primary,
            tables=tuple(tables),
            metric_aliases=(),
            group_columns=(),
            required_columns=tuple(
                self._dedupe(
                    column
                    for table in tables
                    for column in required_by_table[table]
                )
            ),
            node_contract=primary_contract,
            access_contracts=tuple(access_contracts),
        )

    @staticmethod
    def _detail_filter_predicate(
        table_alias: str,
        column: str,
        operator: str,
        literal_value: Any,
    ) -> str:
        left = "%s.%s" % (table_alias, quote_identifier(column))
        operators = {
            "EQ": "=",
            "NE": "!=",
            "GT": ">",
            "GTE": ">=",
            "LT": "<",
            "LTE": "<=",
        }
        if operator == "IN":
            if not isinstance(literal_value, (list, tuple)) or not literal_value:
                raise RuntimeError("IN entity filter requires a non-empty literal list")
            return "%s IN (%s)" % (
                left,
                ", ".join(sql_literal(item) for item in literal_value),
            )
        sql_operator = operators.get(operator)
        if not sql_operator:
            raise RuntimeError("unsupported grounded entity filter operator")
        return "%s %s %s" % (left, sql_operator, sql_literal(literal_value))

    @staticmethod
    def _literal_filter_predicate(
        column: str,
        operator: str,
        literal_value: Any,
    ) -> str:
        left = quote_identifier(column)
        operators = {
            "EQ": "=",
            "NE": "!=",
            "GT": ">",
            "GTE": ">=",
            "LT": "<",
            "LTE": "<=",
        }
        if operator == "IN":
            if not isinstance(literal_value, (list, tuple)) or not literal_value:
                raise RuntimeError("IN entity filter requires a non-empty literal list")
            if len(literal_value) > 100:
                raise RuntimeError("IN entity filter exceeds the deterministic value bound")
            return "%s IN (%s)" % (
                left,
                ", ".join(sql_literal(item) for item in literal_value),
            )
        sql_operator = operators.get(operator)
        if not sql_operator:
            raise RuntimeError("unsupported grounded entity filter operator")
        return "%s %s %s" % (left, sql_operator, sql_literal(literal_value))

    @classmethod
    def _deterministic_metric_filter_verification(
        cls,
        contract: NodePlanContract,
        sql: str,
        row_count: int,
    ) -> EntityFilterVerificationProof:
        """Bind generated literal predicates to the executed SQL and Contract.

        Aggregate rows do not expose the internal filter field, so result-row
        identity comparison would be impossible without changing the requested
        grain.  Instead, this lane proves that every exact Contract obligation
        is a mandatory top-level conjunct in the generated SELECT and seals the
        proof to both the NodePlanContract and executed SQL hashes.
        """

        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations:
            return EntityFilterVerificationProof(
                task_id=contract.task_id,
                status="not_required",
            )
        contract_hash = entity_filter_contract_hash(contract)
        requested = {
            value
            for obligation in obligations
            for value in canonical_entity_values(
                obligation.reference.values,
                entity_comparison_policy(obligation.reference),
            )
        }
        verified = cls._sql_contains_all_literal_filter_obligations(sql, obligations)
        return EntityFilterVerificationProof(
            task_id=contract.task_id,
            obligation_id=obligations[0].obligation_id,
            status="verified" if verified else "failed",
            code="" if verified else "ENTITY_FILTER_SQL_PREDICATE_MISSING",
            verified=verified,
            coverage_complete=verified,
            contract_hash=contract_hash,
            sql_hash=entity_filter_sql_hash(sql),
            requested_value_hashes=sorted(
                entity_value_hash(item, contract_hash) for item in requested
            ),
            row_count=max(0, int(row_count or 0)),
            reason=(
                "Deterministic compiler proved every typed literal filter as a mandatory SQL conjunct"
                if verified
                else "Executed deterministic SQL does not contain every typed literal filter obligation"
            ),
        )

    @classmethod
    def _sql_contains_all_literal_filter_obligations(
        cls,
        sql: str,
        obligations: list[EntityFilterObligation],
    ) -> bool:
        try:
            parsed = sqlglot.parse_one(sql, read="doris")
        except Exception:
            return False
        where = parsed.args.get("where") if isinstance(parsed, exp.Select) else None
        if not isinstance(where, exp.Where):
            return False
        actual = cls._and_conjuncts(where.this)
        for obligation in obligations:
            reference = obligation.reference
            operator = str(reference.comparison_policy or "").strip().upper()
            values = list(reference.values or [])
            literal: Any = values if operator == "IN" else values[0] if values else None
            try:
                predicate = cls._literal_filter_predicate(
                    reference.field,
                    operator,
                    literal,
                )
                expected_select = sqlglot.parse_one(
                    "SELECT 1 WHERE %s" % predicate,
                    read="doris",
                )
            except Exception:
                return False
            expected_where = expected_select.args.get("where")
            if not isinstance(expected_where, exp.Where):
                return False
            if not any(item == expected_where.this for item in actual):
                return False
        return True

    @classmethod
    def _and_conjuncts(cls, expression: exp.Expression) -> list[exp.Expression]:
        current = expression.this if isinstance(expression, exp.Paren) else expression
        if isinstance(current, exp.And):
            return [
                *cls._and_conjuncts(current.this),
                *cls._and_conjuncts(current.expression),
            ]
        return [current]

    @staticmethod
    def _entity_filter_verification(
        contract: NodePlanContract,
        rows: list[dict[str, Any]],
        sql: str,
    ) -> EntityFilterVerificationProof:
        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations:
            return EntityFilterVerificationProof(
                task_id=contract.task_id,
                status="not_required",
            )
        reference = obligations[0].reference
        policy = entity_comparison_policy(reference)
        requested = canonical_entity_values(contract.filter_values, policy)
        observed = canonical_entity_values(
            [row.get(contract.filter_column) for row in rows if contract.filter_column in row],
            policy,
        )
        contract_hash = entity_filter_contract_hash(contract)
        base = {
            "task_id": contract.task_id,
            "obligation_id": obligations[0].obligation_id,
            "field": contract.filter_column,
            "comparison_policy": policy,
            "contract_hash": contract_hash,
            "sql_hash": entity_filter_sql_hash(sql),
            "requested_value_hashes": sorted(
                entity_value_hash(item, contract_hash) for item in requested
            ),
            "observed_value_hashes": sorted(
                entity_value_hash(item, contract_hash) for item in observed
            ),
            "row_count": len(rows),
            "coverage_complete": True,
        }
        if not contract.filter_column or not requested:
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_CONTRACT_INVALID",
                reason="grounded entity filter lacks an executable field/value",
            )
        if any(contract.filter_column not in row for row in rows):
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_RESULT_UNVERIFIABLE",
                reason="detail rows do not expose the governed entity identity",
            )
        unexpected = observed - requested
        if unexpected:
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_RESULT_MISMATCH",
                unexpected_value_count=len(unexpected),
                reason="detail result contains identities outside the requested filter",
            )
        missing = requested - observed
        return EntityFilterVerificationProof(
            **base,
            verified=True,
            status="verified",
            missing_values=sorted(missing),
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
        unknown_columns = GroundedQueryExecutionKernel._scope_unknown_columns(
            parsed,
            asset_pack,
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

    @staticmethod
    def _scope_unknown_columns(
        parsed: exp.Expression,
        asset_pack: Any,
    ) -> list[str]:
        """Resolve columns per SQL alias/scope instead of unioning table schemas."""

        unknown: set[str] = set()
        try:
            scopes = list(traverse_scope(parsed))
        except Exception:
            return ["<scope-analysis-failed>"]
        for scope in scopes:
            selected_sources = getattr(scope, "selected_sources", {}) or {}
            sources = {
                str(alias or "").lower(): pair[1]
                for alias, pair in selected_sources.items()
            }
            select_aliases = {
                str(item.alias or "").lower()
                for item in (
                    scope.expression.expressions
                    if isinstance(scope.expression, exp.Select)
                    else []
                )
                if isinstance(item, exp.Alias) and item.alias
            }

            def source_has_column(source: Any, column_name: str) -> bool:
                if isinstance(source, exp.Table):
                    known = {
                        str(item or "").lower()
                        for item in asset_pack.known_columns(source.name)
                    }
                    return column_name in known
                expression = getattr(source, "expression", None)
                outputs = {
                    str(item or "").lower()
                    for item in getattr(expression, "named_selects", []) or []
                }
                return column_name in outputs

            for column in getattr(scope, "columns", []) or []:
                if isinstance(column.this, exp.Star):
                    continue
                name = str(column.name or "").lower()
                qualifier = str(column.table or "").lower()
                if not name:
                    continue
                if not qualifier and name in select_aliases:
                    continue
                if qualifier:
                    source = sources.get(qualifier)
                    if source is None or not source_has_column(source, name):
                        unknown.add("%s.%s" % (qualifier, name))
                    continue
                matches = [
                    alias
                    for alias, source in sources.items()
                    if source_has_column(source, name)
                ]
                if len(matches) != 1:
                    unknown.add(
                        "%s:%s"
                        % ("ambiguous" if len(matches) > 1 else "unknown", name)
                    )
        return sorted(unknown)

    def _assess_physical_plan(
        self,
        contract: GroundedQueryContract,
        compilation: GroundedSqlCompilation,
        asset_pack: Any,
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        explain = getattr(self.doris_repository, "explain_verbose", None)
        if not callable(explain):
            return {}
        formal_assets = [
            dict(getattr(item, "metadata", {}) or {})
            for item in getattr(asset_pack, "tables", []) or []
            if str(getattr(item, "table", "") or "")
            in set(compilation.tables)
        ]
        time_field = contract.time_field
        partition_requirements: list[PartitionPruningRequirement] = []
        if (
            time_field.table
            and time_field.partition_pruning_column
            and time_field.semantic_ref_id
            and time_field.partition_pruning_policy
            in {"EXACT_EQUIVALENT", "SAFE_SUPERSET"}
        ):
            partition_requirements.append(
                PartitionPruningRequirement(
                    table=time_field.table,
                    partition_column=time_field.partition_pruning_column,
                    semantic_evidence_ref=time_field.semantic_ref_id,
                )
            )
        if not partition_requirements and len(set(compilation.tables)) < 2:
            return {}
        try:
            explain_kwargs: dict[str, Any] = {}
            if "timeout_seconds" in inspect.signature(explain).parameters:
                explain_kwargs["timeout_seconds"] = max(
                    1,
                    int(timeout_seconds),
                )
            explain_payload = explain(compilation.sql, **explain_kwargs)
            assessment = DorisPhysicalPlanGovernor().assess(
                sql=compilation.sql,
                formal_assets=formal_assets,
                explain_payload=explain_payload,
                partition_requirements=partition_requirements,
            )
        except Exception as exc:
            blocking = bool(partition_requirements)
            sql_fingerprint = hashlib.sha256(
                str(compilation.sql or "").strip().encode("utf-8")
            ).hexdigest()
            assessment = PhysicalPlanAssessment(
                assessment_id="physical_plan_%s" % sql_fingerprint[:16],
                status="INVALID" if blocking else "GAPPED",
                executable=not blocking,
                sql_fingerprint=sql_fingerprint,
                gaps=[
                    PhysicalPlanGap(
                        code="PHYSICAL_PLAN_EXPLAIN_FAILED",
                        message=(
                            "Doris EXPLAIN VERBOSE failed: %s"
                            % str(exc)[:400]
                        ),
                        blocking=blocking,
                        expected_evidence="Doris EXPLAIN VERBOSE receipt",
                    )
                ],
            )
        return assessment.model_dump(by_alias=True, mode="json")

    def _failed_result(
        self,
        plan: QueryPlan,
        compilation: GroundedSqlCompilation,
        validation: SqlValidationResult,
        code: str,
        message: str,
        *,
        duration_ms: int = 0,
        data_snapshot: DataSnapshotContract | None = None,
        physical_plan_assessment: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        intent = plan.intents[0]
        bundle = QueryBundle(
            sql=compilation.sql,
            tables=list(compilation.tables),
            failed=True,
            error="%s: %s" % (code, message),
            summary=message,
            duration_ms=duration_ms,
            data_snapshot=data_snapshot or DataSnapshotContract(),
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
            physical_plan_assessment=dict(physical_plan_assessment or {}),
        )

    def _prepare_data_snapshot(
        self,
        supplied: DataSnapshotContract | None,
        *,
        expected_semantic_activation_fingerprint: str,
        governed: bool,
    ) -> DataSnapshotContract:
        """Capture or revalidate after SQL/ACL and before the first query."""

        expected_activation = str(
            expected_semantic_activation_fingerprint or ""
        ).strip()
        if governed and not expected_activation:
            raise RuntimeError(
                "QUERY_EXECUTION_SEMANTIC_ACTIVATION_REQUIRED"
            )
        capture = getattr(
            self.doris_repository,
            "capture_data_snapshot",
            None,
        )
        revalidate = getattr(
            self.doris_repository,
            "revalidate_data_snapshot",
            None,
        )
        if supplied is None:
            if callable(capture):
                observed = capture(expected_activation)
                snapshot = (
                    observed
                    if isinstance(observed, DataSnapshotContract)
                    else DataSnapshotContract.model_validate(observed)
                )
            else:
                snapshot = DataSnapshotContract(
                    semantic_activation_fingerprint=expected_activation,
                    unsupported_reason=(
                        "DATA_SNAPSHOT_CAPABILITY_UNAVAILABLE"
                    ),
                )
        else:
            snapshot = supplied.model_copy(deep=True)
            if callable(revalidate):
                observed = revalidate(snapshot)
                if isinstance(observed, DataSnapshotContract):
                    snapshot = observed.model_copy(deep=True)
                elif observed is not None and observed is not True:
                    snapshot = DataSnapshotContract.model_validate(observed)
            elif callable(capture):
                current = capture(expected_activation)
                current_snapshot = (
                    current
                    if isinstance(current, DataSnapshotContract)
                    else DataSnapshotContract.model_validate(current)
                )
                if grounded_data_snapshot_identity(
                    current_snapshot
                ) != grounded_data_snapshot_identity(snapshot):
                    raise RuntimeError("DATA_SNAPSHOT_REVALIDATION_MISMATCH")
            elif governed:
                raise RuntimeError(
                    "DATA_SNAPSHOT_REVALIDATION_CAPABILITY_REQUIRED"
                )
        if expected_activation and str(
            snapshot.semantic_activation_fingerprint or ""
        ).strip() != expected_activation:
            raise RuntimeError(
                "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_MISMATCH"
            )
        return snapshot

    def _prepare_execution_identity_context(
        self,
        *,
        merchant_id: str,
        access_role: str,
        user_scope: Mapping[str, Any],
        reference_scope: Any,
        context_owner_fingerprint: str,
        goal_contract_fingerprint: str,
        plan: QueryPlan,
        compilation: GroundedSqlCompilation,
        contract: GroundedQueryContract,
        candidate_validation: Any,
        data_snapshot: DataSnapshotContract,
        execution_generation: int,
        execution_attempt_id: str,
        execution_query_node_id: str,
        expected_semantic_activation_fingerprint: str,
    ) -> _GroundedExecutionIdentityContext:
        goal_fingerprint = str(goal_contract_fingerprint or "").strip()
        if not self._valid_sha256(goal_fingerprint):
            raise RuntimeError(
                "QUERY_EXECUTION_GOAL_CONTRACT_FINGERPRINT_INVALID"
            )
        expected_owner = grounded_context_owner_fingerprint(
            merchant_id,
            access_role,
            user_scope,
        )
        if expected_owner != str(context_owner_fingerprint or "").strip():
            raise RuntimeError("QUERY_EXECUTION_CONTEXT_OWNER_MISMATCH")
        activation_fingerprint = str(
            expected_semantic_activation_fingerprint or ""
        ).strip()
        if not activation_fingerprint or activation_fingerprint != str(
            data_snapshot.semantic_activation_fingerprint or ""
        ).strip():
            raise RuntimeError(
                "QUERY_EXECUTION_SNAPSHOT_ACTIVATION_MISMATCH"
            )
        execution_identity = self._server_execution_identity(
            merchant_id=merchant_id,
            access_role=access_role,
            user_scope=user_scope,
            data_snapshot=data_snapshot,
        )
        run_identity = build_grounded_run_execution_identity(
            context_owner_fingerprint=context_owner_fingerprint,
            execution_identity=execution_identity,
            user_scope=user_scope,
            reference_scope=reference_scope,
            datasource_fingerprint=data_snapshot.datasource_fingerprint,
            cache_generation=data_snapshot.cache_generation,
        )
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        sql_ast_fingerprint = self._sql_evidence_fingerprint(
            contract_fingerprint,
            compilation.sql,
            candidate_validation,
        )
        graph_fingerprint = query_graph_fingerprint(plan)
        access_contracts = tuple(
            compilation.access_contracts
            or (compilation.node_contract,)
        )
        query_node_id = str(execution_query_node_id or "").strip() or (
            plan.intents[0].plan_task_id
        )
        node_identity = build_grounded_node_execution_identity(
            run_identity=run_identity,
            graph_fingerprint=graph_fingerprint,
            query_node_id=query_node_id,
            generation=int(execution_generation or 0),
            attempt_id=str(execution_attempt_id or "").strip(),
            goal_contract_fingerprint=goal_fingerprint,
            query_contract_fingerprint=contract_fingerprint,
            sql_ast_fingerprint=sql_ast_fingerprint,
            data_snapshot=data_snapshot,
            access_contracts=access_contracts,
        )
        return _GroundedExecutionIdentityContext(
            execution_identity=execution_identity,
            run_identity=run_identity,
            node_identity=node_identity,
            user_scope=dict(user_scope),
            reference_scope=reference_scope,
            graph_fingerprint=graph_fingerprint,
            query_node_id=query_node_id,
            goal_contract_fingerprint=goal_fingerprint,
            query_contract_fingerprint=contract_fingerprint,
            sql_ast_fingerprint=sql_ast_fingerprint,
            access_contracts=access_contracts,
        )

    def _require_execution_identity_live(
        self,
        stage: str,
        context: _GroundedExecutionIdentityContext,
        *,
        context_owner_fingerprint: str,
        merchant_id: str,
        access_role: str,
        user_scope: Mapping[str, Any],
        reference_scope: Any,
        data_snapshot: DataSnapshotContract,
        execution_generation: int,
        execution_attempt_id: str,
        goal_contract_fingerprint: str,
        query_contract_fingerprint: str,
        sql_ast_fingerprint: str,
    ) -> None:
        require_grounded_execution_identity_live(
            stage=stage,
            stored_run_identity=context.run_identity,
            stored_node_identity=context.node_identity,
            context_owner_fingerprint=context_owner_fingerprint,
            execution_identity=self._server_execution_identity(
                merchant_id=merchant_id,
                access_role=access_role,
                user_scope=user_scope,
                data_snapshot=data_snapshot,
            ),
            user_scope=user_scope,
            reference_scope=reference_scope,
            datasource_fingerprint=data_snapshot.datasource_fingerprint,
            cache_generation=data_snapshot.cache_generation,
            graph_fingerprint=context.graph_fingerprint,
            query_node_id=context.query_node_id,
            generation=int(execution_generation or 0),
            attempt_id=str(execution_attempt_id or "").strip(),
            goal_contract_fingerprint=str(
                goal_contract_fingerprint or ""
            ).strip(),
            query_contract_fingerprint=str(
                query_contract_fingerprint or ""
            ).strip(),
            sql_ast_fingerprint=str(sql_ast_fingerprint or "").strip(),
            data_snapshot=data_snapshot,
            access_contracts=context.access_contracts,
        )

    def _authorize_population_pre_execution(
        self,
        reference: PopulationPreExecutionReference | None,
        *,
        contract: GroundedQueryContract,
        compilation: GroundedSqlCompilation,
        plan: QueryPlan,
        data_snapshot: DataSnapshotContract,
        actual_sql_ast_fingerprint: str,
        context_owner_fingerprint: str,
        goal_contract_fingerprint: str,
        execution_generation: int,
        execution_attempt_id: str,
        population_query_node_id: str,
    ) -> None:
        gate = self.population_execution_gate
        # Population authorization is query-scoped.  A standalone query has
        # no upstream verified entity/result-set reference and must not be
        # rejected merely because a gate is configured globally.
        if reference is None:
            return
        if gate is None:
            raise GroundedPopulationRuntimeGateError(
                "POPULATION_PRE_EXECUTION_REJECTED"
            )
        if not isinstance(gate, GroundedPopulationExecutionGate) or not isinstance(
            reference,
            PopulationPreExecutionReference,
        ):
            raise GroundedPopulationRuntimeGateError(
                "POPULATION_PRE_EXECUTION_REJECTED"
            )
        current_node_id = str(population_query_node_id or "").strip() or str(
            plan.intents[0].plan_task_id or ""
        ).strip()
        node_reference = reference.node
        expected = {
            "contextOwnerFingerprint": str(
                context_owner_fingerprint or ""
            ).strip(),
            "goalContractFingerprint": str(
                goal_contract_fingerprint or ""
            ).strip(),
            "queryNodeId": current_node_id,
            "generation": int(execution_generation or 0),
            "attemptId": str(execution_attempt_id or "").strip(),
            "queryContractFingerprint": (
                grounded_query_contract_fingerprint(contract)
            ),
        }
        observed = {
            "contextOwnerFingerprint": (
                reference.context_owner_fingerprint
            ),
            "goalContractFingerprint": (
                reference.goal_contract_fingerprint
            ),
            "queryNodeId": node_reference.query_node_id,
            "generation": node_reference.generation,
            "attemptId": node_reference.attempt_id,
            "queryContractFingerprint": (
                node_reference.query_contract_fingerprint
            ),
        }
        if observed != expected:
            raise GroundedPopulationRuntimeGateError(
                "POPULATION_PRE_EXECUTION_REJECTED"
            )
        current_evidence = PopulationExecutorNodeEvidence(
            query_node_id=current_node_id,
            contract=contract,
            compilation=compilation,
            data_snapshot=data_snapshot,
            actual_sql_ast_fingerprint=actual_sql_ast_fingerprint,
        )
        result = gate.authorize_node(
            reference=reference,
            execution=current_evidence,
        )
        if not bool(getattr(result, "accepted", False)):
            raise GroundedPopulationRuntimeGateError(
                "POPULATION_PRE_EXECUTION_REJECTED",
                result=result,
            )

    @staticmethod
    def _server_execution_identity(
        *,
        merchant_id: str,
        access_role: str,
        user_scope: Mapping[str, Any],
        data_snapshot: DataSnapshotContract,
    ) -> ExecutionIdentity:
        scope = dict(user_scope or {})
        row_policy = scope.get("rowPolicy") or scope.get("row_policy") or {}
        if not isinstance(row_policy, Mapping):
            row_policy = {}
        return ExecutionIdentity.from_server_context(
            merchant_id=merchant_id,
            tenant_id=str(
                scope.get("tenantId") or scope.get("tenant_id") or ""
            ),
            principal_id=str(
                scope.get("principalId")
                or scope.get("principal_id")
                or scope.get("userId")
                or scope.get("user_id")
                or ""
            ),
            role=access_role,
            permissions=scope.get("permissions") or (),
            region=str(scope.get("region") or ""),
            store_ids=(
                scope.get("storeIds") or scope.get("store_ids") or ()
            ),
            row_policy=dict(row_policy),
            datasource_environment=data_snapshot.datasource_environment,
            semantic_activation_fingerprint=(
                data_snapshot.semantic_activation_fingerprint
            ),
        )

    @staticmethod
    def _sql_evidence_fingerprint(
        contract_fingerprint: str,
        sql: str,
        candidate_validation: Any,
    ) -> str:
        candidate_fingerprint = str(
            getattr(candidate_validation, "ast_fingerprint", "") or ""
        ).strip()
        if candidate_fingerprint:
            return candidate_fingerprint
        return hashlib.sha256(
            ("%s:%s" % (contract_fingerprint, sql)).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        candidate = str(value or "")
        return len(candidate) == 64 and all(
            character in "0123456789abcdef" for character in candidate
        )

    @staticmethod
    def _cancellation_requested(events: Iterable[Any]) -> bool:
        return any(
            bool(getattr(event, "is_set", lambda: False)())
            for event in events
            if event is not None
        )

    def _result_stream_limits(self) -> GroundedResultStreamLimits:
        return GroundedResultStreamLimits(
            preview_rows=max(
                1,
                int(self.settings.context_artifact_inline_max_rows or 1),
            ),
            fetch_batch_rows=max(
                1,
                int(self.settings.grounded_result_stream_fetch_batch_rows),
            ),
            max_rows=max(
                1,
                int(self.settings.grounded_result_stream_max_rows),
            ),
            max_bytes=max(
                2,
                int(self.settings.grounded_result_stream_max_bytes),
            ),
        )

    def _repository_stream_batches(
        self,
        sql: str,
        *,
        batch_size: int,
        cancel_events: Iterable[Any] | None,
        timeout_seconds: int,
        data_snapshot_contract: DataSnapshotContract,
    ) -> Iterator[Iterable[Mapping[str, Any]]]:
        stream = getattr(self.doris_repository, "stream_query_batches", None)
        if not callable(stream):
            raise RuntimeError("QUERY_RESULT_STREAMING_REQUIRED")
        signature = inspect.signature(stream)
        supports_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        candidate_kwargs = {
            "batch_size": batch_size,
            "cancel_events": cancel_events,
            "timeout_seconds": timeout_seconds,
            "data_snapshot_contract": data_snapshot_contract,
        }
        kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if supports_kwargs or key in signature.parameters
        }
        return iter(stream(sql, **kwargs))

    def _masked_stream_batches(
        self,
        batches: Iterator[Iterable[Mapping[str, Any]]],
        *,
        state: _GroundedStreamingRowState,
        node_contract: NodePlanContract,
        masked_contract: NodePlanContract,
        time_window_role: str,
    ) -> Iterator[list[dict[str, Any]]]:
        try:
            for batch in batches:
                raw_batch = [dict(row) for row in batch]
                self._observe_streaming_rows(
                    state,
                    raw_batch,
                    node_contract,
                )
                masked_batch = apply_column_masks(
                    raw_batch,
                    masked_contract,
                )
                for row in masked_batch:
                    row.setdefault("__timeWindowRole", time_window_role)
                yield masked_batch
        finally:
            close = getattr(batches, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _observe_streaming_rows(
        state: _GroundedStreamingRowState,
        rows: Sequence[dict[str, Any]],
        contract: NodePlanContract,
    ) -> None:
        remaining = max(
            0,
            state.preview_limit - len(state.raw_preview_rows),
        )
        if remaining:
            state.raw_preview_rows.extend(
                dict(row) for row in rows[:remaining]
            )
        state.row_count += len(rows)
        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations:
            return
        filter_column = str(contract.filter_column or "")
        policy = entity_comparison_policy(obligations[0].reference)
        for row in rows:
            if not filter_column or filter_column not in row:
                state.filter_column_missing = True
                continue
            state.observed_filter_values.update(
                canonical_entity_values([row.get(filter_column)], policy)
            )

    @staticmethod
    def _streaming_entity_filter_verification(
        contract: NodePlanContract,
        state: _GroundedStreamingRowState,
        sql: str,
    ) -> EntityFilterVerificationProof:
        obligations = [
            item
            for item in contract.entity_filter_obligations
            if item.required and item.status == "bound"
        ]
        if not obligations:
            return EntityFilterVerificationProof(
                task_id=contract.task_id,
                status="not_required",
            )
        reference = obligations[0].reference
        policy = entity_comparison_policy(reference)
        requested = canonical_entity_values(contract.filter_values, policy)
        observed = set(state.observed_filter_values)
        contract_hash = entity_filter_contract_hash(contract)
        base = {
            "task_id": contract.task_id,
            "obligation_id": obligations[0].obligation_id,
            "field": contract.filter_column,
            "comparison_policy": policy,
            "contract_hash": contract_hash,
            "sql_hash": entity_filter_sql_hash(sql),
            "requested_value_hashes": sorted(
                entity_value_hash(item, contract_hash)
                for item in requested
            ),
            "observed_value_hashes": sorted(
                entity_value_hash(item, contract_hash)
                for item in observed
            ),
            "row_count": state.row_count,
            "coverage_complete": True,
        }
        if not contract.filter_column or not requested:
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_CONTRACT_INVALID",
                reason=(
                    "grounded entity filter lacks an executable field/value"
                ),
            )
        if state.filter_column_missing:
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_RESULT_UNVERIFIABLE",
                reason=(
                    "detail rows do not expose the governed entity identity"
                ),
            )
        unexpected = observed - requested
        if unexpected:
            return EntityFilterVerificationProof(
                **base,
                status="failed",
                code="ENTITY_FILTER_RESULT_MISMATCH",
                unexpected_value_count=len(unexpected),
                reason=(
                    "detail result contains identities outside the requested filter"
                ),
            )
        return EntityFilterVerificationProof(
            **base,
            verified=True,
            status="verified",
            missing_values=sorted(requested - observed),
        )

    @classmethod
    def _classify_streamed_result(
        cls,
        contract: GroundedQueryContract,
        intent: Any,
        sql: str,
        receipt: GroundedResultArtifactReceipt,
        *,
        core_sql_candidate: bool,
    ) -> tuple[str, bool, int]:
        query_shape = str(contract.query_shape or "").upper()
        if query_shape == "RANKED":
            coverage = ResultCoverage.TOP_N.value
        elif query_shape not in {"DETAIL", "ENTITY_LOOKUP"}:
            coverage = (
                cls._core_sql_non_ranked_coverage(sql)
                if core_sql_candidate
                else ResultCoverage.ALL_ROWS.value
            )
        elif core_sql_candidate:
            coverage = cls._core_sql_detail_coverage(
                sql,
                receipt.exact_row_count,
            )
        else:
            coverage = ResultCoverage.ALL_ROWS.value
        semantic_complete = coverage in {
            ResultCoverage.ALL_ROWS.value,
            ResultCoverage.TOP_N.value,
        }
        return (
            coverage,
            bool(receipt.preview_is_truncated or not semantic_complete),
            receipt.exact_row_count if semantic_complete else 0,
        )

    @staticmethod
    def _execution_identity_authority_payload(
        context: _GroundedExecutionIdentityContext,
    ) -> dict[str, Any]:
        execution_identity = context.execution_identity
        reference_scope = context.reference_scope
        if hasattr(reference_scope, "model_dump"):
            reference_scope = reference_scope.model_dump(
                by_alias=True,
                mode="json",
            )
        elif isinstance(reference_scope, Mapping):
            reference_scope = dict(reference_scope)
        else:
            reference_scope = {}
        return {
            "executionIdentity": {
                "merchantId": execution_identity.merchant_id,
                "tenantId": execution_identity.tenant_id,
                "principalId": execution_identity.principal_id,
                "role": execution_identity.role,
                "permissions": list(execution_identity.permissions),
                "region": execution_identity.region,
                "storeIds": list(execution_identity.store_ids),
                "rowPolicyFingerprint": (
                    execution_identity.row_policy_fingerprint
                ),
                "datasourceEnvironment": (
                    execution_identity.datasource_environment
                ),
                "semanticActivationFingerprint": (
                    execution_identity.semantic_activation_fingerprint
                ),
            },
            "runIdentity": context.run_identity.model_dump(
                by_alias=True,
                mode="json",
            ),
            "nodeIdentity": context.node_identity.model_dump(
                by_alias=True,
                mode="json",
            ),
            "userScope": dict(context.user_scope),
            "referenceScope": reference_scope,
            "graphFingerprint": context.graph_fingerprint,
            "queryNodeId": context.query_node_id,
            "goalContractFingerprint": (
                context.goal_contract_fingerprint
            ),
            "queryContractFingerprint": (
                context.query_contract_fingerprint
            ),
            "sqlAstFingerprint": context.sql_ast_fingerprint,
            "accessContracts": [
                item.model_dump(by_alias=True, mode="json")
                for item in context.access_contracts
            ],
        }

    @staticmethod
    def _execution_identity_context_from_authority(
        payload: Mapping[str, Any],
    ) -> _GroundedExecutionIdentityContext:
        raw_execution_identity = dict(
            payload.get("executionIdentity") or {}
        )
        execution_identity = ExecutionIdentity(
            merchant_id=str(raw_execution_identity.get("merchantId") or ""),
            tenant_id=str(raw_execution_identity.get("tenantId") or ""),
            principal_id=str(
                raw_execution_identity.get("principalId") or ""
            ),
            role=str(raw_execution_identity.get("role") or ""),
            permissions=tuple(
                str(item)
                for item in raw_execution_identity.get("permissions") or []
            ),
            region=str(raw_execution_identity.get("region") or ""),
            store_ids=tuple(
                str(item)
                for item in raw_execution_identity.get("storeIds") or []
            ),
            row_policy_fingerprint=str(
                raw_execution_identity.get("rowPolicyFingerprint") or ""
            ),
            datasource_environment=str(
                raw_execution_identity.get("datasourceEnvironment") or ""
            ),
            semantic_activation_fingerprint=str(
                raw_execution_identity.get(
                    "semanticActivationFingerprint"
                )
                or ""
            ),
        )
        return _GroundedExecutionIdentityContext(
            execution_identity=execution_identity,
            run_identity=GroundedRunExecutionIdentitySeal.model_validate(
                payload.get("runIdentity") or {}
            ),
            node_identity=GroundedNodeExecutionIdentitySeal.model_validate(
                payload.get("nodeIdentity") or {}
            ),
            user_scope=dict(payload.get("userScope") or {}),
            reference_scope=dict(payload.get("referenceScope") or {}),
            graph_fingerprint=str(
                payload.get("graphFingerprint") or ""
            ),
            query_node_id=str(payload.get("queryNodeId") or ""),
            goal_contract_fingerprint=str(
                payload.get("goalContractFingerprint") or ""
            ),
            query_contract_fingerprint=str(
                payload.get("queryContractFingerprint") or ""
            ),
            sql_ast_fingerprint=str(
                payload.get("sqlAstFingerprint") or ""
            ),
            access_contracts=tuple(
                NodePlanContract.model_validate(item)
                for item in payload.get("accessContracts") or []
            ),
        )

    def _stage_grounded_result_artifacts(
        self,
        *,
        artifact_root: str,
        run_id: str,
        task_id: str,
        context_owner_fingerprint: str,
        contract: GroundedQueryContract,
        compilation: GroundedSqlCompilation,
        execution_preparation: Any,
        semantic_activation_fingerprint: str,
        data_snapshot: DataSnapshotContract,
        rows: list[dict[str, Any]],
        streamed_rows_receipt: GroundedResultArtifactReceipt | None,
        artifact_exact_row_count: int,
        result_coverage: str,
        result_is_truncated: bool,
        exact_result_row_count: int,
        execution_generation: int,
        execution_attempt_id: str,
        execution_identity_context: (
            _GroundedExecutionIdentityContext | None
        ),
        cancel_events: Iterable[Any] | None,
    ) -> dict[str, Any]:
        cancellation_events = tuple(cancel_events or ())
        if self._cancellation_requested(cancellation_events):
            raise RuntimeError("QUERY_RESULT_ARTIFACT_STAGING_CANCELLED")
        try:
            publication_root, staging_root = (
                validated_grounded_query_artifact_roots(
                    self.settings.resolved_workspace_path,
                    artifact_root,
                )
            )
        except GroundedContextWorkspaceError as exc:
            raise RuntimeError(str(exc)) from exc
        if not str(context_owner_fingerprint or "").strip():
            raise RuntimeError("QUERY_RESULT_CONTEXT_OWNER_REQUIRED")
        generation = int(execution_generation or 0)
        attempt_id = str(execution_attempt_id or "").strip()
        if generation <= 0:
            raise RuntimeError("QUERY_RESULT_EXECUTION_GENERATION_REQUIRED")
        if not attempt_id:
            raise RuntimeError("QUERY_RESULT_EXECUTION_ATTEMPT_REQUIRED")
        store = WorkspaceArtifactStore(self.settings, staging_root)
        rows_canonical_hash = grounded_canonical_json_sha256(rows)
        sql_hash = hashlib.sha256(
            str(compilation.sql or "").encode("utf-8")
        ).hexdigest()
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        activation_fingerprint = str(
            semantic_activation_fingerprint or ""
        ).strip()
        if not activation_fingerprint:
            # Direct legacy executor tests do not create publication authority.
            # Governed online calls always provide the sealed activation.
            activation_fingerprint = str(
                getattr(
                    execution_preparation,
                    "asset_pack_fingerprint",
                    "",
                )
                or ""
            ).strip()
        if not activation_fingerprint:
            raise RuntimeError(
                "QUERY_RESULT_SEMANTIC_ACTIVATION_FINGERPRINT_REQUIRED"
            )
        candidate_validation = getattr(
            execution_preparation,
            "candidate_validation",
            None,
        )
        sql_evidence_fingerprint = str(
            getattr(candidate_validation, "ast_fingerprint", "") or ""
        ).strip()
        if not sql_evidence_fingerprint:
            sql_evidence_fingerprint = hashlib.sha256(
                ("%s:%s" % (contract_fingerprint, compilation.sql)).encode(
                    "utf-8"
                )
            ).hexdigest()
        snapshot_identity = self._artifact_snapshot_identity(data_snapshot)
        snapshot_activation_fingerprint = str(
            snapshot_identity.get("semanticActivationFingerprint") or ""
        )
        if (
            snapshot_activation_fingerprint
            and snapshot_activation_fingerprint != activation_fingerprint
        ):
            raise RuntimeError(
                "QUERY_RESULT_SNAPSHOT_ACTIVATION_MISMATCH"
            )

        if streamed_rows_receipt is not None:
            if (
                not streamed_rows_receipt.complete
                or not streamed_rows_receipt.active
                or not streamed_rows_receipt.immutable
                or streamed_rows_receipt.coverage != "ALL_ROWS"
                or streamed_rows_receipt.exact_row_count
                != max(0, int(artifact_exact_row_count or 0))
            ):
                raise RuntimeError("QUERY_RESULT_STREAM_RECEIPT_INCOMPLETE")
            rows_artifact = {
                "success": True,
                "relativePath": streamed_rows_receipt.rows_relative_path,
                "sha256": streamed_rows_receipt.rows_canonical_sha256,
                "contentAddress": streamed_rows_receipt.content_address,
                "bytes": streamed_rows_receipt.byte_count,
                "immutable": True,
            }
            artifact_coverage = "ALL_ROWS"
            artifact_complete = True
            artifact_row_count = streamed_rows_receipt.exact_row_count
        else:
            rows_text = json.dumps(
                rows,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            rows_bytes = rows_text.encode("utf-8")
            rows_sha256 = hashlib.sha256(rows_bytes).hexdigest()
            rows_artifact = {
                "success": True,
                "relativePath": "",
                "sha256": rows_sha256,
                "contentAddress": "sha256:%s" % rows_sha256,
                "bytes": len(rows_bytes),
                "immutable": True,
                "_content": rows_text,
            }
            artifact_complete = str(result_coverage or "") in {
                ResultCoverage.ALL_ROWS.value,
                ResultCoverage.TOP_N.value,
            }
            artifact_coverage = (
                "ALL_ROWS" if artifact_complete else "INACTIVE_PARTIAL"
            )
            artifact_row_count = len(rows)

        if artifact_row_count != max(
            0,
            int(artifact_exact_row_count or 0),
        ):
            raise RuntimeError("QUERY_RESULT_ARTIFACT_ROW_COUNT_MISMATCH")
        seal_identity: dict[str, Any] = {}
        if execution_identity_context is not None:
            seal_identity = {
                "runExecutionIdentity": (
                    execution_identity_context.run_identity.model_dump(
                        by_alias=True,
                        mode="json",
                    )
                ),
                "nodeExecutionIdentity": (
                    execution_identity_context.node_identity.model_dump(
                        by_alias=True,
                        mode="json",
                    )
                ),
            }
        result_identity = {
            "contractFingerprint": contract_fingerprint,
            "goalContractFingerprint": (
                execution_identity_context.goal_contract_fingerprint
                if execution_identity_context is not None
                else ""
            ),
            "sqlSha256": sql_hash,
            "sqlEvidenceFingerprint": sql_evidence_fingerprint,
            "rowsCanonicalSha256": rows_canonical_hash,
            "contextOwnerFingerprint": str(
                context_owner_fingerprint or ""
            ),
            "semanticActivationFingerprint": activation_fingerprint,
            "dataSnapshot": snapshot_identity,
            "resultCoverage": str(result_coverage or ""),
            "resultIsTruncated": bool(result_is_truncated),
            "storedRowCount": len(rows),
            "exactResultRowCount": max(0, int(exact_result_row_count or 0)),
            "previewRowCount": len(rows),
            "artifactRowCount": artifact_row_count,
            "artifactByteCount": max(
                0,
                int(rows_artifact.get("bytes") or 0),
            ),
            "artifactRowsSha256": str(
                rows_artifact.get("sha256") or ""
            ),
            "artifactContentAddress": str(
                rows_artifact.get("contentAddress") or ""
            ),
            "artifactCoverage": artifact_coverage,
            "artifactComplete": bool(artifact_complete),
            "streamingMaterialized": bool(
                streamed_rows_receipt is not None
            ),
            "executionGeneration": generation,
            "executionAttemptId": attempt_id,
            "runFingerprint": hashlib.sha256(
                str(run_id or "").encode("utf-8")
            ).hexdigest(),
            "taskFingerprint": hashlib.sha256(
                str(task_id or "").encode("utf-8")
            ).hexdigest(),
            **seal_identity,
        }
        pending_id = "pending_query_%s" % hashlib.sha256(
            json.dumps(
                result_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        namespace = pending_id
        if streamed_rows_receipt is None:
            rows_artifact = store.write_text(
                namespace,
                "rows.json",
                str(rows_artifact.pop("_content")),
                preview_chars=0,
                immutable=True,
            )
        sql_artifact = store.write_text(
            namespace,
            "query.sql",
            compilation.sql,
            preview_chars=0,
            immutable=True,
        )
        if not rows_artifact.get("success") or not sql_artifact.get("success"):
            raise RuntimeError("QUERY_RESULT_CONTENT_WRITE_FAILED")
        if self._cancellation_requested(cancellation_events):
            raise RuntimeError("QUERY_RESULT_ARTIFACT_STAGING_CANCELLED")
        pending_manifest = {
            "schemaVersion": 3,
            "artifactKind": "GROUNDED_QUERY_RESULT_PENDING",
            "pendingArtifactId": pending_id,
            **result_identity,
            "rowsArtifact": self._private_artifact_receipt(rows_artifact),
            "sqlArtifact": self._private_artifact_receipt(sql_artifact),
            "rowsSha256": rows_artifact.get("sha256"),
        }
        pending_manifest_artifact = store.write_json(
            namespace,
            "pending.manifest.json",
            pending_manifest,
            preview_chars=0,
            immutable=True,
        )
        if not pending_manifest_artifact.get("success"):
            raise RuntimeError("QUERY_RESULT_PENDING_MANIFEST_WRITE_FAILED")
        if self._cancellation_requested(cancellation_events):
            raise RuntimeError("QUERY_RESULT_ARTIFACT_STAGING_CANCELLED")
        # This receipt is deliberately server-private. The runtime kernel
        # removes it from QueryBundle events before the result can reach Core.
        # Absolute roots are retained solely so the verifier gate can reopen
        # and validate the exact staged bytes without trusting caller paths.
        private_receipt = {
            "pendingArtifactId": pending_id,
            "publicationRoot": str(publication_root),
            "stagingRoot": str(staging_root),
            "identity": result_identity,
            "rowsArtifact": self._private_artifact_receipt(rows_artifact),
            "sqlArtifact": self._private_artifact_receipt(sql_artifact),
            "pendingManifestArtifact": self._private_artifact_receipt(
                pending_manifest_artifact
            ),
        }
        if execution_identity_context is not None:
            private_receipt["executionIdentityAuthority"] = (
                self._execution_identity_authority_payload(
                    execution_identity_context
                )
            )
        return private_receipt

    def publish_pending_result_artifact(
        self,
        pending_receipt: dict[str, Any],
        *,
        verified_evidence: VerifiedEvidence,
        expected_generation: int,
        expected_attempt_id: str,
        expected_contract_fingerprint: str,
        expected_sql_fingerprint: str,
        expected_context_owner_fingerprint: str,
        expected_semantic_activation_fingerprint: str,
        expected_data_snapshot: DataSnapshotContract,
        expected_result_coverage: str,
        expected_result_is_truncated: bool,
        expected_stored_row_count: int,
        expected_exact_result_row_count: int,
        expected_rows_canonical_sha256: str,
        expected_goal_contract_fingerprint: str = "",
        expected_merchant_id: str = "",
        expected_access_role: str = "",
        expected_user_scope: Mapping[str, Any] | None = None,
        expected_reference_scope: Any = None,
    ) -> dict[str, Any]:
        """Verify exact staged bytes and publish a verifier-bound receipt.

        Files do not authorize consumption by existing on disk. Only the
        returned receipt, committed into the verified ledger by the kernel,
        is a visibility capability for the read backend and Sandbox.
        """

        if not verified_evidence.passed:
            raise RuntimeError("QUERY_RESULT_VERIFIED_EVIDENCE_REQUIRED")
        receipt = dict(pending_receipt or {})
        identity = dict(receipt.get("identity") or {})
        expected_identity = {
            "contractFingerprint": str(expected_contract_fingerprint or ""),
            "sqlEvidenceFingerprint": str(expected_sql_fingerprint or ""),
            "contextOwnerFingerprint": str(
                expected_context_owner_fingerprint or ""
            ),
            "semanticActivationFingerprint": str(
                expected_semantic_activation_fingerprint or ""
            ),
            "dataSnapshot": self._artifact_snapshot_identity(
                expected_data_snapshot
            ),
            "resultCoverage": str(expected_result_coverage or ""),
            "resultIsTruncated": bool(expected_result_is_truncated),
            "storedRowCount": max(0, int(expected_stored_row_count or 0)),
            "exactResultRowCount": max(
                0,
                int(expected_exact_result_row_count or 0),
            ),
            "executionGeneration": int(expected_generation or 0),
            "executionAttemptId": str(expected_attempt_id or ""),
            "rowsCanonicalSha256": str(
                expected_rows_canonical_sha256 or ""
            ),
        }
        if str(expected_goal_contract_fingerprint or "").strip():
            expected_identity["goalContractFingerprint"] = str(
                expected_goal_contract_fingerprint or ""
            ).strip()
        for key, expected in expected_identity.items():
            if identity.get(key) != expected:
                raise RuntimeError(
                    "QUERY_RESULT_PENDING_BINDING_MISMATCH:%s" % key
                )
        publication_root, staging_root = self._validated_artifact_roots(
            receipt
        )
        staging_store = WorkspaceArtifactStore(self.settings, staging_root)
        pending_manifest_text = self._read_private_staged_artifact(
            staging_store,
            dict(receipt.get("pendingManifestArtifact") or {}),
        )
        try:
            pending_manifest = json.loads(pending_manifest_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "QUERY_RESULT_PENDING_MANIFEST_INVALID"
            ) from exc
        pending_id = str(receipt.get("pendingArtifactId") or "").strip()
        if (
            not pending_id
            or pending_manifest.get("pendingArtifactId") != pending_id
            or any(
                pending_manifest.get(key) != value
                for key, value in identity.items()
            )
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_MANIFEST_MISMATCH")
        rows_artifact_receipt = dict(receipt.get("rowsArtifact") or {})
        sql_artifact_receipt = dict(receipt.get("sqlArtifact") or {})
        if pending_manifest.get("rowsArtifact") != rows_artifact_receipt:
            raise RuntimeError("QUERY_RESULT_PENDING_ROWS_BINDING_MISMATCH")
        if pending_manifest.get("sqlArtifact") != sql_artifact_receipt:
            raise RuntimeError("QUERY_RESULT_PENDING_SQL_BINDING_MISMATCH")
        if str(sql_artifact_receipt.get("sha256") or "") != str(
            identity.get("sqlSha256") or ""
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_SQL_SHA_MISMATCH")
        if (
            identity.get("artifactComplete") is not True
            or identity.get("artifactCoverage") != "ALL_ROWS"
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_ARTIFACT_INCOMPLETE")
        artifact_binding = {
            "sha256": str(identity.get("artifactRowsSha256") or ""),
            "contentAddress": str(
                identity.get("artifactContentAddress") or ""
            ),
            "bytes": max(0, int(identity.get("artifactByteCount") or 0)),
        }
        if any(
            rows_artifact_receipt.get(key) != value
            for key, value in artifact_binding.items()
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_FULL_ROWS_MISMATCH")

        authority_payload = dict(
            receipt.get("executionIdentityAuthority") or {}
        )
        if str(expected_goal_contract_fingerprint or "").strip():
            if not authority_payload:
                raise RuntimeError(
                    "QUERY_RESULT_EXECUTION_IDENTITY_AUTHORITY_REQUIRED"
                )
            execution_identity_context = (
                self._execution_identity_context_from_authority(
                    authority_payload
                )
            )
            self._require_execution_identity_live(
                "PRE_PUBLICATION",
                execution_identity_context,
                context_owner_fingerprint=(
                    expected_context_owner_fingerprint
                ),
                merchant_id=expected_merchant_id,
                access_role=expected_access_role,
                user_scope=dict(expected_user_scope or {}),
                reference_scope=expected_reference_scope,
                data_snapshot=expected_data_snapshot,
                execution_generation=expected_generation,
                execution_attempt_id=expected_attempt_id,
                goal_contract_fingerprint=(
                    expected_goal_contract_fingerprint
                ),
                query_contract_fingerprint=(
                    expected_contract_fingerprint
                ),
                sql_ast_fingerprint=expected_sql_fingerprint,
            )

        verified_payload = verified_evidence.model_dump(
            by_alias=True,
            mode="json",
        )
        verified_sha = self._stable_artifact_hash(verified_payload)
        publication_fingerprint = hashlib.sha256(
            ("%s:%s" % (pending_id, verified_sha)).encode("utf-8")
        ).hexdigest()
        published_rows = self._publish_staged_artifact_streaming(
            staging_root,
            publication_root,
            rows_artifact_receipt,
            "result_%s_rows.json" % publication_fingerprint,
            expected_json_row_count=max(
                0,
                int(identity.get("artifactRowCount") or 0),
            ),
        )
        published_sql = self._publish_staged_artifact_streaming(
            staging_root,
            publication_root,
            sql_artifact_receipt,
            "result_%s.sql" % publication_fingerprint,
        )
        final_manifest = {
            "schemaVersion": 3,
            "artifactKind": "GROUNDED_QUERY_RESULT",
            "publicationStatus": "VERIFIED",
            "artifactFingerprint": publication_fingerprint,
            "pendingArtifactId": pending_id,
            **identity,
            "verifiedEvidence": verified_payload,
            "verifiedEvidenceSha256": verified_sha,
            "pendingManifestSha256": str(
                dict(receipt.get("pendingManifestArtifact") or {}).get(
                    "sha256"
                )
                or ""
            ),
            "rowsArtifact": self._manifest_child_receipt(published_rows),
            "sqlArtifact": self._manifest_child_receipt(published_sql),
        }
        manifest_text = json.dumps(
            final_manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        exact_manifest_sha256 = hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest()
        publication_store = WorkspaceArtifactStore(
            self.settings,
            publication_root,
        )
        published_manifest = publication_store.write_text(
            "query_results",
            "result_%s.manifest.json" % exact_manifest_sha256,
            manifest_text,
            preview_chars=0,
            immutable=True,
        )
        if (
            not published_manifest.get("success")
            or published_manifest.get("sha256")
            != exact_manifest_sha256
        ):
            raise RuntimeError("QUERY_RESULT_MANIFEST_PUBLICATION_FAILED")
        return {
            "artifactFingerprint": publication_fingerprint,
            "pendingArtifactId": pending_id,
            "manifestRelativePath": str(
                published_manifest.get("relativePath") or ""
            ),
            "manifestRef": str(published_manifest.get("merchantUri") or ""),
            "rowsRelativePath": str(
                published_rows.get("relativePath") or ""
            ),
            "rowsRef": str(published_rows.get("merchantUri") or ""),
            "sqlRelativePath": str(
                published_sql.get("relativePath") or ""
            ),
            "sqlRef": str(published_sql.get("merchantUri") or ""),
            "queryManifestSha256": str(
                published_manifest.get("sha256") or ""
            ),
            # Compatibility projection for existing response formatting.
            "manifestSha256": str(published_manifest.get("sha256") or ""),
            "rowsSha256": str(published_rows.get("sha256") or ""),
            "sqlSha256": str(published_sql.get("sha256") or ""),
            "manifestContentAddress": str(
                published_manifest.get("contentAddress") or ""
            ),
            "rowsContentAddress": str(
                published_rows.get("contentAddress") or ""
            ),
            "sqlContentAddress": str(
                published_sql.get("contentAddress") or ""
            ),
            "storedRowCount": int(identity.get("storedRowCount") or 0),
            "artifactRowCount": int(
                identity.get("artifactRowCount") or 0
            ),
            "artifactByteCount": int(
                identity.get("artifactByteCount") or 0
            ),
            "artifactCoverage": str(
                identity.get("artifactCoverage") or ""
            ),
            "artifactComplete": bool(identity.get("artifactComplete")),
            "exactResultRowCount": int(
                identity.get("exactResultRowCount") or 0
            ),
            "resultCoverage": str(identity.get("resultCoverage") or ""),
            "resultIsTruncated": bool(
                identity.get("resultIsTruncated")
            ),
            "executionGeneration": int(
                identity.get("executionGeneration") or 0
            ),
            "attemptFingerprint": hashlib.sha256(
                str(identity.get("executionAttemptId") or "").encode(
                    "utf-8"
                )
            ).hexdigest(),
            "contractFingerprint": str(
                identity.get("contractFingerprint") or ""
            ),
            "sqlEvidenceFingerprint": str(
                identity.get("sqlEvidenceFingerprint") or ""
            ),
            "contextOwnerFingerprint": str(
                identity.get("contextOwnerFingerprint") or ""
            ),
            "semanticActivationFingerprint": str(
                identity.get("semanticActivationFingerprint") or ""
            ),
            "dataSnapshotFingerprint": self._stable_artifact_hash(
                identity.get("dataSnapshot") or {}
            ),
            "verifiedEvidenceSha256": verified_sha,
        }

    def _validated_artifact_roots(
        self,
        receipt: dict[str, Any],
    ) -> tuple[Path, Path]:
        try:
            publication_root, staging_root = (
                validated_grounded_query_artifact_roots(
                    self.settings.resolved_workspace_path,
                    str(receipt.get("publicationRoot") or ""),
                )
            )
        except GroundedContextWorkspaceError as exc:
            raise RuntimeError(str(exc)) from exc
        received_staging_root = Path(
            os.path.abspath(str(receipt.get("stagingRoot") or ""))
        )
        if staging_root != received_staging_root:
            raise RuntimeError("QUERY_RESULT_STAGING_ROOT_MISMATCH")
        return publication_root, staging_root

    def _publish_staged_artifact_streaming(
        self,
        staging_root: Path,
        publication_root: Path,
        artifact: dict[str, Any],
        final_name: str,
        *,
        expected_json_row_count: int | None = None,
    ) -> dict[str, Any]:
        """Hash and atomically link staged bytes without loading them in RAM."""

        expected_sha = str(artifact.get("sha256") or "").strip()
        expected_bytes = max(0, int(artifact.get("bytes") or 0))
        if (
            len(expected_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha
            )
            or str(artifact.get("contentAddress") or "")
            != "sha256:%s" % expected_sha
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_RECEIPT_INVALID")
        source_components = self._artifact_relative_components(
            str(artifact.get("relativePath") or "")
        )
        final_component = self._artifact_file_component(final_name)
        source_parent_descriptor = -1
        source_descriptor = -1
        publication_descriptor = -1
        publication_locked = False
        temporary_name = ""
        try:
            source_parent_descriptor = self._open_workspace_directory(
                staging_root,
                source_components[:-1],
            )
            source_name = source_components[-1]
            source_marker = self._immutable_marker_name(source_name)
            marker_digest = _read_regular_file_at(
                source_parent_descriptor,
                source_marker,
            ).decode("ascii").strip()
            if marker_digest != expected_sha:
                raise RuntimeError(
                    "QUERY_RESULT_PENDING_IMMUTABLE_MARKER_MISMATCH"
                )
            source_descriptor = os.open(
                source_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=source_parent_descriptor,
            )
            if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
                raise RuntimeError("QUERY_RESULT_PENDING_ARTIFACT_INVALID")
            if expected_json_row_count is None:
                actual_sha, actual_bytes = self._hash_descriptor(
                    source_descriptor
                )
                actual_rows = None
            else:
                (
                    actual_sha,
                    actual_bytes,
                    actual_rows,
                ) = self._hash_json_rows_descriptor(source_descriptor)
            if (
                actual_sha != expected_sha
                or actual_bytes != expected_bytes
                or (
                    expected_json_row_count is not None
                    and actual_rows != expected_json_row_count
                )
            ):
                raise RuntimeError(
                    "QUERY_RESULT_PENDING_ARTIFACT_SHA_MISMATCH"
                )
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            publication_descriptor = self._open_workspace_directory(
                publication_root,
                ("query_results",),
            )
            # The precreated publication directory itself is the cross-thread
            # and cross-process lock target.  This avoids racing to create a
            # lock path while retaining the already validated directory fd.
            fcntl.flock(publication_descriptor, fcntl.LOCK_EX)
            publication_locked = True
            final_marker = self._immutable_marker_name(final_component)
            try:
                final_marker_digest = _read_regular_file_at(
                    publication_descriptor,
                    final_marker,
                ).decode("ascii").strip()
            except FileNotFoundError:
                final_marker_digest = ""
            if final_marker_digest and final_marker_digest != expected_sha:
                raise RuntimeError("QUERY_RESULT_PUBLICATION_MARKER_CONFLICT")
            if not final_marker_digest:
                _atomic_write_at(
                    publication_descriptor,
                    final_marker,
                    ("%s\n" % expected_sha).encode("ascii"),
                    error_code="QUERY_RESULT_PUBLICATION_MARKER_FAILED",
                )
            try:
                existing_descriptor = os.open(
                    final_component,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=publication_descriptor,
                )
            except FileNotFoundError:
                existing_descriptor = -1
            if existing_descriptor >= 0:
                try:
                    actual_sha, actual_bytes = self._hash_descriptor(
                        existing_descriptor
                    )
                finally:
                    os.close(existing_descriptor)
                if (
                    actual_sha != expected_sha
                    or actual_bytes != expected_bytes
                ):
                    raise RuntimeError(
                        "QUERY_RESULT_PUBLICATION_CONTENT_CONFLICT"
                    )
            else:
                temporary_name = ".artifact-write-%s.tmp" % uuid.uuid4().hex
                temporary_descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=publication_descriptor,
                )
                digest = hashlib.sha256()
                byte_count = 0
                try:
                    while True:
                        chunk = os.read(source_descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        byte_count += len(chunk)
                        self._write_all(temporary_descriptor, chunk)
                    os.fsync(temporary_descriptor)
                finally:
                    os.close(temporary_descriptor)
                if (
                    digest.hexdigest() != expected_sha
                    or byte_count != expected_bytes
                ):
                    raise RuntimeError(
                        "QUERY_RESULT_PENDING_ARTIFACT_SHA_MISMATCH"
                    )
                os.link(
                    temporary_name,
                    final_component,
                    src_dir_fd=publication_descriptor,
                    dst_dir_fd=publication_descriptor,
                    follow_symlinks=False,
                )
                os.unlink(temporary_name, dir_fd=publication_descriptor)
                temporary_name = ""
                os.fsync(publication_descriptor)
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "QUERY_RESULT_CONTENT_PUBLICATION_FAILED:%s" % type(exc).__name__
            ) from exc
        finally:
            if temporary_name and publication_descriptor >= 0:
                try:
                    os.unlink(temporary_name, dir_fd=publication_descriptor)
                except OSError:
                    pass
            if publication_locked and publication_descriptor >= 0:
                fcntl.flock(publication_descriptor, fcntl.LOCK_UN)
            if publication_descriptor >= 0:
                os.close(publication_descriptor)
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if source_parent_descriptor >= 0:
                os.close(source_parent_descriptor)
        relative_path = "query_results/%s" % final_component
        return {
            "success": True,
            "relativePath": relative_path,
            "merchantUri": merchant_uri_for_artifact(
                relative_path,
                namespace="query_results",
            ),
            "sha256": expected_sha,
            "contentAddress": "sha256:%s" % expected_sha,
            "bytes": expected_bytes,
            "immutable": True,
        }

    def _open_workspace_directory(
        self,
        root: Path,
        components: tuple[str, ...],
        *,
        create: bool = False,
    ) -> int:
        trusted_workspace = self.settings.resolved_workspace_path.resolve(
            strict=True
        )
        lexical_root = Path(os.path.abspath(str(root)))
        try:
            root_components = tuple(
                lexical_root.relative_to(trusted_workspace).parts
            )
        except ValueError as exc:
            raise RuntimeError(
                "QUERY_RESULT_ARTIFACT_ROOT_OUTSIDE_WORKSPACE"
            ) from exc
        return _open_directory_beneath(
            trusted_workspace,
            (*root_components, *components),
            create=create,
        )

    @staticmethod
    def _artifact_relative_components(relative_path: str) -> tuple[str, ...]:
        path = Path(str(relative_path or ""))
        components = tuple(path.parts)
        if (
            path.is_absolute()
            or not components
            or any(
                not component
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
                for component in components
            )
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_PATH_INVALID")
        return components

    @staticmethod
    def _artifact_file_component(name: str) -> str:
        value = str(name or "")
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise RuntimeError("QUERY_RESULT_PUBLICATION_NAME_INVALID")
        return value

    @staticmethod
    def _immutable_marker_name(file_name: str) -> str:
        return ".artifact-immutable-%s.sha256" % hashlib.sha256(
            file_name.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _hash_descriptor(descriptor: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        return digest.hexdigest(), byte_count

    @staticmethod
    def _hash_json_rows_descriptor(
        descriptor: int,
    ) -> tuple[str, int, int]:
        """Hash and count one canonical top-level JSON object array."""

        digest = hashlib.sha256()
        byte_count = 0
        row_count = 0
        started = False
        completed = False
        expect_value = True
        after_comma = False
        depth = 0
        in_string = False
        escaped = False
        whitespace = {9, 10, 13, 32}
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            for character in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == 92:
                        escaped = True
                    elif character == 34:
                        in_string = False
                    continue
                if completed:
                    if character not in whitespace:
                        raise RuntimeError(
                            "QUERY_RESULT_PENDING_ROWS_JSON_INVALID"
                        )
                    continue
                if not started:
                    if character in whitespace:
                        continue
                    if character != 91:
                        raise RuntimeError(
                            "QUERY_RESULT_PENDING_ROWS_JSON_INVALID"
                        )
                    started = True
                    continue
                if depth:
                    if character == 34:
                        in_string = True
                    elif character in {91, 123}:
                        depth += 1
                    elif character in {93, 125}:
                        depth -= 1
                        if depth == 0:
                            expect_value = False
                    continue
                if character in whitespace:
                    continue
                if expect_value:
                    if character == 93 and row_count == 0 and not after_comma:
                        completed = True
                        continue
                    if character != 123:
                        raise RuntimeError(
                            "QUERY_RESULT_PENDING_ROWS_JSON_INVALID"
                        )
                    row_count += 1
                    depth = 1
                    after_comma = False
                    continue
                if character == 44:
                    expect_value = True
                    after_comma = True
                    continue
                if character == 93:
                    completed = True
                    continue
                raise RuntimeError(
                    "QUERY_RESULT_PENDING_ROWS_JSON_INVALID"
                )
        if (
            not started
            or not completed
            or depth
            or in_string
            or escaped
            or expect_value and row_count > 0
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_ROWS_JSON_INVALID")
        return digest.hexdigest(), byte_count, row_count

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short artifact publication write")
            remaining = remaining[written:]

    @staticmethod
    def _private_artifact_receipt(
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "relativePath": str(artifact.get("relativePath") or ""),
            "sha256": str(artifact.get("sha256") or ""),
            "contentAddress": str(artifact.get("contentAddress") or ""),
            "bytes": max(0, int(artifact.get("bytes") or 0)),
        }

    @staticmethod
    def _manifest_child_receipt(
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "relativePath": str(artifact.get("relativePath") or ""),
            "merchantUri": str(artifact.get("merchantUri") or ""),
            "sha256": str(artifact.get("sha256") or ""),
            "contentAddress": str(artifact.get("contentAddress") or ""),
            "bytes": max(0, int(artifact.get("bytes") or 0)),
        }

    @staticmethod
    def _read_private_staged_artifact(
        store: WorkspaceArtifactStore,
        artifact: dict[str, Any],
    ) -> str:
        relative_path = str(artifact.get("relativePath") or "").strip()
        expected_bytes = max(0, int(artifact.get("bytes") or 0))
        expected_sha = str(artifact.get("sha256") or "").strip()
        expected_address = str(artifact.get("contentAddress") or "").strip()
        if (
            not relative_path
            or not expected_sha
            or expected_address != "sha256:%s" % expected_sha
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_RECEIPT_INVALID")
        result = store.read(
            relative_path,
            max_chars=max(1, expected_bytes + 1),
            require_immutable=True,
        )
        if not result.get("success"):
            raise RuntimeError(
                "QUERY_RESULT_PENDING_ARTIFACT_INVALID:%s"
                % str(result.get("error") or "READ_FAILED")
            )
        content = str(result.get("content") or "")
        encoded = content.encode("utf-8")
        if (
            bool(result.get("truncated"))
            or len(encoded) != expected_bytes
            or hashlib.sha256(encoded).hexdigest() != expected_sha
            or result.get("contentAddress") != expected_address
        ):
            raise RuntimeError("QUERY_RESULT_PENDING_ARTIFACT_SHA_MISMATCH")
        return content

    @staticmethod
    def _artifact_snapshot_identity(
        data_snapshot: DataSnapshotContract,
    ) -> dict[str, Any]:
        # Cache eligibility and artifact identity are different concerns.
        # ``cache_identity()`` intentionally returns an empty mapping when the
        # datasource cannot promise a reusable epoch.  The publication gate
        # must nevertheless bind every server-observed datasource and
        # semantic identity so an UNSUPPORTED snapshot cannot cross an
        # activation or datasource boundary.
        return {
            "datasourceFingerprint": str(
                data_snapshot.datasource_fingerprint or ""
            ),
            "datasourceEnvironment": str(
                data_snapshot.datasource_environment or ""
            ),
            "dataEpoch": str(data_snapshot.data_epoch or ""),
            "consistencyMode": str(
                data_snapshot.consistency_mode or "UNSUPPORTED"
            ),
            "semanticActivationFingerprint": str(
                data_snapshot.semantic_activation_fingerprint or ""
            ),
            "cacheGeneration": str(
                data_snapshot.cache_generation or ""
            ),
            "capturedAt": str(data_snapshot.captured_at or ""),
            "unsupportedReason": str(
                data_snapshot.unsupported_reason or ""
            ),
        }

    @staticmethod
    def _stable_artifact_hash(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_json_sha256(value: Any) -> str:
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for chunk in encoder.iterencode(value):
            digest.update(chunk.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _detail_display_limit(intent: Any) -> int:
        return max(1, min(int(getattr(intent, "limit", 0) or 100), 1000))

    @classmethod
    def _classify_result_rows(
        cls,
        contract: GroundedQueryContract,
        intent: Any,
        sql: str,
        raw_rows: list[dict[str, Any]],
        *,
        core_sql_candidate: bool,
    ) -> tuple[list[dict[str, Any]], str, bool, int]:
        """Classify row-set coverage without deriving it from equal counts.

        Deterministic detail SQL deliberately fetches a sentinel row.  Core
        SQL remains immutable, so its AST is inspected conservatively: an
        exhausted sole outer LIMIT can prove completeness, while a saturated,
        nested, offset, or unparseable limit is only a preview.
        """

        query_shape = str(getattr(contract, "query_shape", "") or "").upper()
        if query_shape == "RANKED":
            return (
                list(raw_rows),
                ResultCoverage.TOP_N.value,
                False,
                len(raw_rows),
            )
        if query_shape not in {"DETAIL", "ENTITY_LOOKUP"}:
            coverage = (
                cls._core_sql_non_ranked_coverage(sql)
                if core_sql_candidate
                else ResultCoverage.ALL_ROWS.value
            )
            complete = coverage == ResultCoverage.ALL_ROWS.value
            return (
                list(raw_rows),
                coverage,
                not complete,
                len(raw_rows) if complete else 0,
            )

        if not core_sql_candidate:
            display_limit = cls._detail_display_limit(intent)
            if len(raw_rows) > display_limit:
                return (
                    list(raw_rows[:display_limit]),
                    ResultCoverage.PREVIEW.value,
                    True,
                    0,
                )
            return (
                list(raw_rows),
                ResultCoverage.ALL_ROWS.value,
                False,
                len(raw_rows),
            )

        coverage = cls._core_sql_detail_coverage(sql, len(raw_rows))
        complete = coverage == ResultCoverage.ALL_ROWS.value
        return (
            list(raw_rows),
            coverage,
            not complete,
            len(raw_rows) if complete else 0,
        )

    @staticmethod
    def _core_sql_detail_coverage(sql: str, returned_row_count: int) -> str:
        try:
            expression = sqlglot.parse_one(sql, read="mysql")
        except Exception:
            return ResultCoverage.PREVIEW.value

        limits = list(expression.find_all(exp.Limit))
        outer_limit = expression.args.get("limit")
        outer_offset = expression.args.get("offset")
        if not limits:
            return ResultCoverage.ALL_ROWS.value
        if len(limits) != 1 or outer_limit is None or outer_offset is not None:
            return ResultCoverage.PREVIEW.value
        limit_expression = getattr(outer_limit, "expression", None)
        if not isinstance(limit_expression, exp.Literal) or not limit_expression.is_int:
            return ResultCoverage.PREVIEW.value
        try:
            limit_value = int(limit_expression.this)
        except (TypeError, ValueError):
            return ResultCoverage.PREVIEW.value
        if limit_value <= 0 or returned_row_count >= limit_value:
            return ResultCoverage.PREVIEW.value
        return ResultCoverage.ALL_ROWS.value

    @staticmethod
    def _core_sql_non_ranked_coverage(sql: str) -> str:
        """A LIMIT on aggregate/grouped Core SQL cannot prove full coverage."""

        try:
            expression = sqlglot.parse_one(sql, read="mysql")
        except Exception:
            return ResultCoverage.PREVIEW.value
        if any(True for _ in expression.find_all(exp.Limit)):
            return ResultCoverage.PREVIEW.value
        return ResultCoverage.ALL_ROWS.value

    @staticmethod
    def _table_metadata(asset_pack: Any, table: str) -> dict[str, Any]:
        entry = next(
            (item for item in asset_pack.tables if str(item.table or "") == table),
            None,
        )
        return dict(getattr(entry, "metadata", {}) or {}) if entry is not None else {}

    @staticmethod
    def _time_column(
        contract: GroundedQueryContract,
        table: str,
        table_default: str,
    ) -> str:
        if contract.time_field.table == table and contract.time_field.column:
            return str(contract.time_field.column)
        columns = GroundedQueryExecutionKernel._dedupe(
            metric.time_column or table_default
            for metric in contract.metrics
            if metric.table == table
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
        if (
            contract.time_field.table == table
            and contract.time_field.column == time_column
            and contract.time_field.time_role != "PARTITION"
        ):
            return GroundedQueryExecutionKernel._bounded_time_predicate(
                quote_identifier(time_column),
                contract,
                role=contract.time_field.role,
            )
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
            time_range.data_as_of_policy
            == LATEST_AVAILABLE_PARTITION_DATA_AS_OF_POLICY
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
        raise RuntimeError("grounded calendar window is missing explicit start/end bounds")

    @staticmethod
    def _bounded_time_predicate(
        qualified_column: str,
        contract: GroundedQueryContract,
        *,
        role: str,
        lower_expansion_days: int = 0,
        upper_expansion_days: int = 0,
    ) -> str:
        start_raw = str(
            contract.time_range.execution_start_date
            or contract.time_range.start_date
            or ""
        ).strip()
        end_raw = str(
            contract.time_range.execution_end_date
            or contract.time_range.end_date
            or ""
        ).strip()
        try:
            start = date.fromisoformat(start_raw[:10]) - timedelta(
                days=max(0, int(lower_expansion_days or 0))
            )
            end = date.fromisoformat(end_raw[:10]) + timedelta(
                days=max(0, int(upper_expansion_days or 0))
            )
        except ValueError as exc:
            raise RuntimeError(
                "explicit business time range lacks canonical date bounds"
            ) from exc
        normalized_role = str(role or "").strip().upper()
        if normalized_role in {"DATETIME", "TIMESTAMP", "TIME"}:
            exclusive_end = end + timedelta(days=1)
            return "%s >= %s AND %s < %s" % (
                qualified_column,
                sql_literal("%s 00:00:00" % start.isoformat()),
                qualified_column,
                sql_literal("%s 00:00:00" % exclusive_end.isoformat()),
            )
        return "%s BETWEEN %s AND %s" % (
            qualified_column,
            sql_literal(start.isoformat()),
            sql_literal(end.isoformat()),
        )

    @staticmethod
    def _partition_pruning_column(
        contract: GroundedQueryContract,
        table: str,
    ) -> str:
        time_field = contract.time_field
        if (
            time_field.table == table
            and time_field.partition_pruning_column
            and time_field.partition_pruning_policy
            in {"EXACT_EQUIVALENT", "SAFE_SUPERSET"}
        ):
            return str(time_field.partition_pruning_column)
        return ""

    @staticmethod
    def _partition_pruning_predicate(
        contract: GroundedQueryContract,
        table: str,
        column: str,
        *,
        table_alias: str = "",
    ) -> str:
        time_field = contract.time_field
        if time_field.table != table or column != time_field.partition_pruning_column:
            return ""
        qualified = (
            "%s.%s" % (table_alias, quote_identifier(column))
            if table_alias
            else quote_identifier(column)
        )
        return GroundedQueryExecutionKernel._bounded_time_predicate(
            qualified,
            contract,
            role="DATE",
            lower_expansion_days=time_field.partition_lower_expansion_days,
            upper_expansion_days=time_field.partition_upper_expansion_days,
        )

    @staticmethod
    def _detail_time_predicates(
        contract: GroundedQueryContract,
        table: str,
        table_default: str,
        table_alias: str,
    ) -> list[tuple[str, str]]:
        if not contract.time_range.explicit:
            return []
        time_field = contract.time_field
        if time_field.semantic_ref_id:
            if time_field.table != table or not time_field.column:
                return []
            business_column = str(time_field.column)
            business_predicate = GroundedQueryExecutionKernel._bounded_time_predicate(
                "%s.%s" % (table_alias, quote_identifier(business_column)),
                contract,
                role=time_field.role,
            )
        else:
            business_column = str(table_default or "").strip()
            if not business_column:
                return []
            business_predicate = GroundedQueryExecutionKernel._bounded_time_predicate(
                "%s.%s" % (table_alias, quote_identifier(business_column)),
                contract,
                role="DATE",
            )
        predicates = [(business_predicate, business_column)]
        pruning_column = GroundedQueryExecutionKernel._partition_pruning_column(
            contract,
            table,
        )
        if pruning_column and pruning_column != business_column:
            pruning_predicate = (
                GroundedQueryExecutionKernel._partition_pruning_predicate(
                    contract,
                    table,
                    pruning_column,
                    table_alias=table_alias,
                )
            )
            if pruning_predicate:
                predicates.append((pruning_predicate, pruning_column))
        return predicates

    @staticmethod
    def _dedupe(values: Iterable[Any]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _entity_filter_values(filters: Iterable[Any]) -> list[Any]:
        values: list[Any] = []
        for item in filters:
            literal = item.literal_value
            if item.operator == "IN" and isinstance(
                literal,
                (list, tuple, set, frozenset),
            ):
                values.extend(literal)
            else:
                values.append(literal)
        return values
