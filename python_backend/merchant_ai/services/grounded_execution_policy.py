from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import Field

from merchant_ai.models import APIModel
from merchant_ai.services.formulas import compile_metric_formula
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.semantic_metrics import semantic_metric_temporal_contract_issue


class GroundedFastPathReason(str, Enum):
    """Stable, domain-independent reasons for selecting the LLM SQL path."""

    CONTRACT_NOT_READY = "CONTRACT_NOT_READY"
    QUERY_SHAPE_NOT_SCALAR = "QUERY_SHAPE_NOT_SCALAR"
    TABLE_COUNT_NOT_ONE = "TABLE_COUNT_NOT_ONE"
    METRIC_COUNT_NOT_ONE = "METRIC_COUNT_NOT_ONE"
    METRIC_NOT_PUBLISHED = "METRIC_NOT_PUBLISHED"
    METRIC_TABLE_MISMATCH = "METRIC_TABLE_MISMATCH"
    DIMENSIONS_PRESENT = "DIMENSIONS_PRESENT"
    GROUPING_PRESENT = "GROUPING_PRESENT"
    RANKING_PRESENT = "RANKING_PRESENT"
    SELECTED_FIELDS_PRESENT = "SELECTED_FIELDS_PRESENT"
    ENTITY_FILTERS_PRESENT = "ENTITY_FILTERS_PRESENT"
    RELATIONSHIPS_PRESENT = "RELATIONSHIPS_PRESENT"
    FIELD_AGGREGATION_PRESENT = "FIELD_AGGREGATION_PRESENT"
    MULTI_WINDOW_PRESENT = "MULTI_WINDOW_PRESENT"
    METRIC_SOURCE_COLUMNS_MISSING = "METRIC_SOURCE_COLUMNS_MISSING"
    METRIC_FORMULA_NOT_EXECUTABLE = "METRIC_FORMULA_NOT_EXECUTABLE"
    METRIC_TIME_SEMANTICS_NOT_EXECUTABLE = "METRIC_TIME_SEMANTICS_NOT_EXECUTABLE"
    QUERY_SHAPE_NOT_DETERMINISTIC = "QUERY_SHAPE_NOT_DETERMINISTIC"
    PRIMARY_TABLE_MISMATCH = "PRIMARY_TABLE_MISMATCH"
    METRICS_PRESENT = "METRICS_PRESENT"
    GROUP_DIMENSION_COUNT_NOT_ONE = "GROUP_DIMENSION_COUNT_NOT_ONE"
    DIMENSION_TABLE_MISMATCH = "DIMENSION_TABLE_MISMATCH"
    RANKING_NOT_ENABLED = "RANKING_NOT_ENABLED"
    RANKING_DIRECTION_NOT_SUPPORTED = "RANKING_DIRECTION_NOT_SUPPORTED"
    RANKING_LIMIT_INVALID = "RANKING_LIMIT_INVALID"
    RANKING_METRIC_NOT_BOUND = "RANKING_METRIC_NOT_BOUND"
    RANKING_DIMENSION_NOT_BOUND = "RANKING_DIMENSION_NOT_BOUND"
    SELECTED_FIELDS_MISSING = "SELECTED_FIELDS_MISSING"
    SELECTED_FIELD_BINDING_INVALID = "SELECTED_FIELD_BINDING_INVALID"
    SELECTED_FIELD_ALIAS_DUPLICATE = "SELECTED_FIELD_ALIAS_DUPLICATE"
    ENTITY_FILTERS_MISSING = "ENTITY_FILTERS_MISSING"
    ENTITY_FILTER_BINDING_INVALID = "ENTITY_FILTER_BINDING_INVALID"
    ENTITY_FILTER_OPERATOR_NOT_DECLARED = "ENTITY_FILTER_OPERATOR_NOT_DECLARED"
    ENTITY_FILTER_OPERATOR_NOT_SUPPORTED = "ENTITY_FILTER_OPERATOR_NOT_SUPPORTED"
    ENTITY_FILTER_VALUE_LIMIT_EXCEEDED = "ENTITY_FILTER_VALUE_LIMIT_EXCEEDED"
    ENTITY_FILTER_HINT_MISMATCH = "ENTITY_FILTER_HINT_MISMATCH"
    UPSTREAM_ENTITY_BINDINGS_PRESENT = "UPSTREAM_ENTITY_BINDINGS_PRESENT"
    TYPED_ENTITY_FILTER_MISSING = "TYPED_ENTITY_FILTER_MISSING"
    EXECUTION_SHAPE_NOT_DETERMINISTIC = "EXECUTION_SHAPE_NOT_DETERMINISTIC"
    METRICS_MISSING = "METRICS_MISSING"
    METRIC_BINDING_DUPLICATE = "METRIC_BINDING_DUPLICATE"
    METRIC_ALIAS_DUPLICATE = "METRIC_ALIAS_DUPLICATE"
    METRIC_TIME_COLUMN_MISSING = "METRIC_TIME_COLUMN_MISSING"
    METRIC_TIME_COLUMN_INCOMPATIBLE = "METRIC_TIME_COLUMN_INCOMPATIBLE"
    METRIC_TIME_POLICY_INCOMPATIBLE = "METRIC_TIME_POLICY_INCOMPATIBLE"
    METRIC_ANCHOR_POLICY_INCOMPATIBLE = "METRIC_ANCHOR_POLICY_INCOMPATIBLE"
    DIMENSION_BINDING_INVALID = "DIMENSION_BINDING_INVALID"
    TREND_DIMENSION_NOT_TEMPORAL = "TREND_DIMENSION_NOT_TEMPORAL"
    RANKING_LIMIT_EXCEEDS_MAXIMUM = "RANKING_LIMIT_EXCEEDS_MAXIMUM"
    FIELD_AGGREGATION_BINDING_INVALID = "FIELD_AGGREGATION_BINDING_INVALID"
    FIELD_AGGREGATION_NOT_ALLOWLISTED = "FIELD_AGGREGATION_NOT_ALLOWLISTED"


class GroundedExecutionMode(str, Enum):
    """Execution authority selected after a Contract becomes READY."""

    UNDECIDED = "UNDECIDED"
    DETERMINISTIC_METRIC = "DETERMINISTIC_METRIC"
    DETERMINISTIC_MULTI_METRIC = "DETERMINISTIC_MULTI_METRIC"
    DETERMINISTIC_GROUPED = "DETERMINISTIC_GROUPED"
    DETERMINISTIC_TREND = "DETERMINISTIC_TREND"
    DETERMINISTIC_RANKED = "DETERMINISTIC_RANKED"
    DETERMINISTIC_ENTITY_LOOKUP = "DETERMINISTIC_ENTITY_LOOKUP"
    CORE_SQL_REQUIRED = "CORE_SQL_REQUIRED"


class GroundedExecutionReason(str, Enum):
    """Stable routing reasons exposed to Core and runtime observers."""

    CONTRACT_NOT_READY = "CONTRACT_NOT_READY"
    SINGLE_METRIC_FAST_PATH_ELIGIBLE = "SINGLE_METRIC_FAST_PATH_ELIGIBLE"
    SAME_TABLE_MULTI_METRIC_ELIGIBLE = "SAME_TABLE_MULTI_METRIC_ELIGIBLE"
    GROUPED_DETERMINISTIC_ELIGIBLE = "GROUPED_DETERMINISTIC_ELIGIBLE"
    TREND_DETERMINISTIC_ELIGIBLE = "TREND_DETERMINISTIC_ELIGIBLE"
    RANKED_DETERMINISTIC_ELIGIBLE = "RANKED_DETERMINISTIC_ELIGIBLE"
    ENTITY_LOOKUP_DETERMINISTIC_ELIGIBLE = "ENTITY_LOOKUP_DETERMINISTIC_ELIGIBLE"
    COMPLEX_QUERY_REQUIRES_CORE_SQL = "COMPLEX_QUERY_REQUIRES_CORE_SQL"


DETERMINISTIC_EXECUTION_MODES = frozenset(
    {
        GroundedExecutionMode.DETERMINISTIC_METRIC,
        GroundedExecutionMode.DETERMINISTIC_MULTI_METRIC,
        GroundedExecutionMode.DETERMINISTIC_GROUPED,
        GroundedExecutionMode.DETERMINISTIC_TREND,
        GroundedExecutionMode.DETERMINISTIC_RANKED,
        GroundedExecutionMode.DETERMINISTIC_ENTITY_LOOKUP,
    }
)


class GroundedFastPathDecision(APIModel):
    """Pure policy result; it neither compiles nor executes a query."""

    eligible: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    reason_details: dict[str, str] = Field(default_factory=dict)


class GroundedDeterministicExecutionDecision(GroundedFastPathDecision):
    """Capability-based execution decision for all deterministic shapes."""

    execution_mode: GroundedExecutionMode = GroundedExecutionMode.UNDECIDED
    execution_reason_codes: list[str] = Field(default_factory=list)


def evaluate_single_metric_fast_path(
    contract: GroundedQueryContract,
) -> GroundedFastPathDecision:
    """Return whether a Contract fits the legacy published-scalar strict subset.

    Runtime routing uses :func:`evaluate_deterministic_execution`, which owns
    the broader capability matrix.  This compatibility helper intentionally
    continues to reject field-derived aggregates and literal filters so older
    callers cannot silently gain authority. It never inspects lexical business
    signals.
    """

    reasons: list[str] = []
    details: dict[str, str] = {}

    def reject(reason: GroundedFastPathReason, detail: str = "") -> None:
        code = reason.value
        if code not in reasons:
            reasons.append(code)
        if detail:
            details[code] = detail

    if not contract.ready:
        reject(GroundedFastPathReason.CONTRACT_NOT_READY)

    if str(contract.query_shape or "").strip().upper() != "SCALAR":
        reject(GroundedFastPathReason.QUERY_SHAPE_NOT_SCALAR)

    if len(contract.tables) != 1:
        reject(GroundedFastPathReason.TABLE_COUNT_NOT_ONE)

    if len(contract.metrics) != 1:
        reject(GroundedFastPathReason.METRIC_COUNT_NOT_ONE)

    grouped_dimensions = [
        dimension for dimension in contract.dimensions if str(dimension.usage or "").strip().lower() == "group_by"
    ]
    hints = contract.binding_hints
    if contract.dimensions or hints.dimension_refs:
        reject(GroundedFastPathReason.DIMENSIONS_PRESENT)
    if grouped_dimensions or str(hints.group_by_ref or "").strip():
        reject(GroundedFastPathReason.GROUPING_PRESENT)

    ranking = contract.ranking
    if (
        ranking.enabled
        or bool(str(ranking.direction or "").strip())
        or int(ranking.limit or 0) != 0
        or bool(str(ranking.metric_ref_id or "").strip())
        or bool(str(ranking.dimension_ref_id or "").strip())
        or bool(str(hints.ranking.order or "").strip())
        or int(hints.ranking.limit or 0) != 0
        or bool(str(hints.ranking.metric_ref or "").strip())
    ):
        reject(GroundedFastPathReason.RANKING_PRESENT)

    if contract.selected_fields or hints.selected_fields:
        reject(GroundedFastPathReason.SELECTED_FIELDS_PRESENT)
    if contract.entity_filters or hints.entity_filters:
        reject(GroundedFastPathReason.ENTITY_FILTERS_PRESENT)
    if contract.relationships or hints.relationship_refs:
        reject(GroundedFastPathReason.RELATIONSHIPS_PRESENT)

    if _has_multiple_windows(contract):
        reject(GroundedFastPathReason.MULTI_WINDOW_PRESENT)

    if len(contract.metrics) == 1:
        metric = contract.metrics[0]
        binding_type = str(metric.binding_type or "").strip().lower()
        if (
            binding_type == "field_aggregation"
            or bool(str(metric.field_aggregation or "").strip())
            or bool(str(metric.source_field_ref_id or "").strip())
            or bool(hints.field_aggregations)
        ):
            reject(GroundedFastPathReason.FIELD_AGGREGATION_PRESENT)

        if binding_type != "published_metric" or not str(metric.semantic_ref_id or "").startswith("semantic:"):
            reject(GroundedFastPathReason.METRIC_NOT_PUBLISHED)

        if len(contract.tables) == 1 and metric.table != contract.tables[0].table:
            reject(GroundedFastPathReason.METRIC_TABLE_MISMATCH)

        if not metric.source_columns:
            reject(GroundedFastPathReason.METRIC_SOURCE_COLUMNS_MISSING)

        if not compile_metric_formula(metric.formula, set(metric.source_columns)):
            reject(GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE)

        temporal_issue = semantic_metric_temporal_contract_issue(metric.model_dump(by_alias=False))
        if temporal_issue:
            reject(
                GroundedFastPathReason.METRIC_TIME_SEMANTICS_NOT_EXECUTABLE,
                temporal_issue,
            )

    return GroundedFastPathDecision(
        eligible=not reasons,
        reason_codes=reasons,
        reason_details=details,
    )


def evaluate_deterministic_execution(
    contract: GroundedQueryContract,
) -> GroundedDeterministicExecutionDecision:
    """Select a deterministic compiler path from governed Contract structure.

    The policy is deliberately lexical- and domain-independent.  It only
    admits shapes that the current direct compiler can execute without
    inventing joins, bindings, ranking semantics, or entity predicates.
    """

    shape = str(contract.query_shape or "").strip().upper()
    if shape == "SCALAR" and len(contract.metrics) == 1:
        return _execution_decision(
            contract,
            _evaluate_metric_query_fast_path(
                contract,
                allowed_execution_shapes={"", "single_metric"},
                required_group_dimensions=0,
            ),
            GroundedExecutionMode.DETERMINISTIC_METRIC,
            GroundedExecutionReason.SINGLE_METRIC_FAST_PATH_ELIGIBLE,
        )
    if shape == "SCALAR":
        return _execution_decision(
            contract,
            _evaluate_metric_query_fast_path(
                contract,
                allowed_execution_shapes={"", "same_table_multi_metric"},
                required_group_dimensions=0,
            ),
            GroundedExecutionMode.DETERMINISTIC_MULTI_METRIC,
            GroundedExecutionReason.SAME_TABLE_MULTI_METRIC_ELIGIBLE,
        )
    if shape == "GROUPED":
        return _execution_decision(
            contract,
            _evaluate_metric_query_fast_path(
                contract,
                allowed_execution_shapes={
                    "",
                    "grouped_metric",
                    "same_table_multi_metric",
                },
                required_group_dimensions=1,
            ),
            GroundedExecutionMode.DETERMINISTIC_GROUPED,
            GroundedExecutionReason.GROUPED_DETERMINISTIC_ELIGIBLE,
        )
    if shape == "TREND":
        return _execution_decision(
            contract,
            _evaluate_metric_query_fast_path(
                contract,
                allowed_execution_shapes={
                    "",
                    "grouped_metric",
                    "same_table_multi_metric",
                },
                required_group_dimensions=1,
                require_temporal_dimension=True,
            ),
            GroundedExecutionMode.DETERMINISTIC_TREND,
            GroundedExecutionReason.TREND_DETERMINISTIC_ELIGIBLE,
        )
    if shape == "RANKED":
        return _execution_decision(
            contract,
            _evaluate_ranked_fast_path(contract),
            GroundedExecutionMode.DETERMINISTIC_RANKED,
            GroundedExecutionReason.RANKED_DETERMINISTIC_ELIGIBLE,
        )
    if shape == "ENTITY_LOOKUP":
        return _execution_decision(
            contract,
            _evaluate_entity_lookup_fast_path(contract),
            GroundedExecutionMode.DETERMINISTIC_ENTITY_LOOKUP,
            GroundedExecutionReason.ENTITY_LOOKUP_DETERMINISTIC_ELIGIBLE,
        )
    return _execution_decision(
        contract,
        GroundedFastPathDecision(
            eligible=False,
            reason_codes=[GroundedFastPathReason.QUERY_SHAPE_NOT_DETERMINISTIC.value],
        ),
        GroundedExecutionMode.UNDECIDED,
        GroundedExecutionReason.COMPLEX_QUERY_REQUIRES_CORE_SQL,
    )


def _execution_decision(
    contract: GroundedQueryContract,
    decision: GroundedFastPathDecision,
    eligible_mode: GroundedExecutionMode,
    eligible_reason: GroundedExecutionReason,
) -> GroundedDeterministicExecutionDecision:
    if not contract.ready or contract.status != "READY":
        mode = GroundedExecutionMode.UNDECIDED
        execution_reasons = [GroundedExecutionReason.CONTRACT_NOT_READY.value]
    elif decision.eligible:
        mode = eligible_mode
        execution_reasons = [eligible_reason.value]
    else:
        mode = GroundedExecutionMode.CORE_SQL_REQUIRED
        execution_reasons = [
            GroundedExecutionReason.COMPLEX_QUERY_REQUIRES_CORE_SQL.value,
            *decision.reason_codes,
        ]
    return GroundedDeterministicExecutionDecision(
        eligible=decision.eligible,
        reason_codes=list(decision.reason_codes),
        reason_details=dict(decision.reason_details),
        execution_mode=mode,
        execution_reason_codes=execution_reasons,
    )


def _evaluate_metric_query_fast_path(
    contract: GroundedQueryContract,
    *,
    allowed_execution_shapes: set[str],
    required_group_dimensions: int,
    require_temporal_dimension: bool = False,
    allow_ranking: bool = False,
) -> GroundedFastPathDecision:
    """Admit only one-table aggregate projections supported by the compiler.

    The accepted topology is one SELECT over one governed table, zero or one
    explicit GROUP BY column, one or more executable governed aggregates, and
    optional literal predicates that are already exact Contract bindings.
    Relationships, upstream entity dependencies, comparison windows, inferred
    dimensions, and any execution shape that signals a join/window/CTE remain
    Core-owned.
    """

    reasons: list[str] = []
    details: dict[str, str] = {}

    def reject(reason: GroundedFastPathReason, detail: str = "") -> None:
        if reason.value not in reasons:
            reasons.append(reason.value)
        if detail:
            details[reason.value] = detail

    if not contract.ready:
        reject(GroundedFastPathReason.CONTRACT_NOT_READY)

    execution_shape = str(contract.execution_shape or "").strip().lower()
    if execution_shape not in allowed_execution_shapes:
        reject(
            GroundedFastPathReason.EXECUTION_SHAPE_NOT_DETERMINISTIC,
            execution_shape,
        )

    if len(contract.tables) != 1:
        reject(GroundedFastPathReason.TABLE_COUNT_NOT_ONE)
    if not contract.metrics:
        reject(GroundedFastPathReason.METRICS_MISSING)
    if contract.selected_fields or contract.binding_hints.selected_fields:
        reject(GroundedFastPathReason.SELECTED_FIELDS_PRESENT)
    if contract.relationships or contract.binding_hints.relationship_refs:
        reject(GroundedFastPathReason.RELATIONSHIPS_PRESENT)
    if (
        contract.upstream_entity_bindings
        or contract.binding_hints.upstream_entity_bindings
    ):
        reject(GroundedFastPathReason.UPSTREAM_ENTITY_BINDINGS_PRESENT)
    if _has_multiple_windows(contract):
        reject(GroundedFastPathReason.MULTI_WINDOW_PRESENT)
    if not allow_ranking and _ranking_present(contract):
        reject(GroundedFastPathReason.RANKING_PRESENT)

    table = contract.tables[0].table if len(contract.tables) == 1 else ""
    if table and contract.primary_table and contract.primary_table != table:
        reject(GroundedFastPathReason.PRIMARY_TABLE_MISMATCH)
    _validate_literal_entity_filters(contract, table, reject)

    grouped = [
        item
        for item in contract.dimensions
        if str(item.usage or "").strip().lower() == "group_by"
    ]
    if required_group_dimensions == 0:
        if contract.dimensions or contract.binding_hints.dimension_refs:
            reject(GroundedFastPathReason.DIMENSIONS_PRESENT)
        if grouped or str(contract.binding_hints.group_by_ref or "").strip():
            reject(GroundedFastPathReason.GROUPING_PRESENT)
    elif (
        len(contract.dimensions) != required_group_dimensions
        or len(grouped) != required_group_dimensions
    ):
        reject(GroundedFastPathReason.GROUP_DIMENSION_COUNT_NOT_ONE)

    group_dimension = grouped[0] if len(grouped) == 1 else None
    if group_dimension is not None:
        semantic_ref_id = str(group_dimension.semantic_ref_id or "").strip()
        if (
            not semantic_ref_id.startswith("semantic:")
            or not str(group_dimension.column or "").strip()
            or group_dimension.table != table
        ):
            reject(GroundedFastPathReason.DIMENSION_BINDING_INVALID)
        hint_refs = {
            str(item or "").strip()
            for item in contract.binding_hints.dimension_refs
            if str(item or "").strip()
        }
        if hint_refs and hint_refs != {semantic_ref_id}:
            reject(GroundedFastPathReason.DIMENSION_BINDING_INVALID)
        group_by_ref = str(contract.binding_hints.group_by_ref or "").strip()
        if group_by_ref and group_by_ref != semantic_ref_id:
            reject(GroundedFastPathReason.DIMENSION_BINDING_INVALID)
        table_binding = contract.tables[0] if len(contract.tables) == 1 else None
        if (
            table_binding is not None
            and table_binding.merchant_filter_column
            and group_dimension.column == table_binding.merchant_filter_column
        ):
            reject(GroundedFastPathReason.DIMENSION_BINDING_INVALID)
        if require_temporal_dimension:
            role = str(group_dimension.role or "").strip().upper()
            governed_time_column = (
                str(table_binding.time_column or "").strip()
                if table_binding is not None
                else ""
            )
            if role not in {"TIME", "DATE", "DATETIME", "TIMESTAMP"} and (
                not governed_time_column
                or group_dimension.column != governed_time_column
            ):
                reject(GroundedFastPathReason.TREND_DIMENSION_NOT_TEMPORAL)

    metric_refs: list[str] = []
    metric_aliases: list[str] = []
    for metric in contract.metrics:
        metric_refs.append(str(metric.semantic_ref_id or "").strip())
        metric_aliases.append(str(metric.metric_key or "").strip())
        _validate_metric_binding(
            contract,
            metric,
            table,
            reject,
            allow_native_time_grain=require_temporal_dimension,
        )
    _validate_metric_set_time_compatibility(contract, table, reject)
    if len(set(metric_refs)) != len(metric_refs):
        reject(GroundedFastPathReason.METRIC_BINDING_DUPLICATE)
    normalized_metric_aliases = [alias.casefold() for alias in metric_aliases]
    if any(not alias for alias in metric_aliases) or len(
        set(normalized_metric_aliases)
    ) != len(normalized_metric_aliases):
        reject(GroundedFastPathReason.METRIC_ALIAS_DUPLICATE)

    hinted_metric_refs = {
        str(item or "").strip()
        for item in contract.binding_hints.metric_refs
        if str(item or "").strip()
    }
    expected_published_refs = {
        str(metric.semantic_ref_id or "").strip()
        for metric in contract.metrics
        if str(metric.binding_type or "").strip().lower() == "published_metric"
    }
    if hinted_metric_refs and hinted_metric_refs != expected_published_refs:
        reject(GroundedFastPathReason.METRIC_BINDING_DUPLICATE)

    return GroundedFastPathDecision(
        eligible=not reasons,
        reason_codes=reasons,
        reason_details=details,
    )


def _evaluate_ranked_fast_path(
    contract: GroundedQueryContract,
) -> GroundedFastPathDecision:
    base = _evaluate_metric_query_fast_path(
        contract,
        allowed_execution_shapes={"", "ranked_group"},
        required_group_dimensions=1,
        allow_ranking=True,
    )
    reasons = list(base.reason_codes)
    details = dict(base.reason_details)

    def reject(reason: GroundedFastPathReason, detail: str = "") -> None:
        if reason.value not in reasons:
            reasons.append(reason.value)
        if detail:
            details[reason.value] = detail

    grouped = [item for item in contract.dimensions if str(item.usage or "").strip().lower() == "group_by"]

    ranking = contract.ranking
    if not ranking.enabled:
        reject(GroundedFastPathReason.RANKING_NOT_ENABLED)
    if str(ranking.direction or "").strip().upper() not in {"ASC", "DESC"}:
        reject(GroundedFastPathReason.RANKING_DIRECTION_NOT_SUPPORTED)
    if int(ranking.limit or 0) <= 0:
        reject(GroundedFastPathReason.RANKING_LIMIT_INVALID)
    if int(ranking.limit or 0) > 1000:
        reject(GroundedFastPathReason.RANKING_LIMIT_EXCEEDS_MAXIMUM)
    metric_refs = {metric.semantic_ref_id for metric in contract.metrics}
    if not ranking.metric_ref_id or ranking.metric_ref_id not in metric_refs:
        reject(GroundedFastPathReason.RANKING_METRIC_NOT_BOUND)
    if len(grouped) != 1 or ranking.dimension_ref_id != grouped[0].semantic_ref_id:
        reject(GroundedFastPathReason.RANKING_DIMENSION_NOT_BOUND)
    return GroundedFastPathDecision(
        eligible=not reasons,
        reason_codes=reasons,
        reason_details=details,
    )


def _evaluate_entity_lookup_fast_path(
    contract: GroundedQueryContract,
) -> GroundedFastPathDecision:
    reasons: list[str] = []

    def reject(reason: GroundedFastPathReason, detail: str = "") -> None:
        if reason.value not in reasons:
            reasons.append(reason.value)

    if not contract.ready:
        reject(GroundedFastPathReason.CONTRACT_NOT_READY)
    execution_shape = str(contract.execution_shape or "").strip().lower()
    if execution_shape not in {"", "entity_lookup"}:
        reject(GroundedFastPathReason.EXECUTION_SHAPE_NOT_DETERMINISTIC)
    if len(contract.tables) != 1:
        reject(GroundedFastPathReason.TABLE_COUNT_NOT_ONE)
    if contract.metrics or contract.binding_hints.metric_refs:
        reject(GroundedFastPathReason.METRICS_PRESENT)
    if contract.dimensions or contract.binding_hints.dimension_refs or contract.binding_hints.group_by_ref:
        reject(GroundedFastPathReason.DIMENSIONS_PRESENT)
    if contract.relationships or contract.binding_hints.relationship_refs:
        reject(GroundedFastPathReason.RELATIONSHIPS_PRESENT)
    if (
        contract.upstream_entity_bindings
        or contract.binding_hints.upstream_entity_bindings
    ):
        reject(GroundedFastPathReason.UPSTREAM_ENTITY_BINDINGS_PRESENT)
    if _ranking_present(contract):
        reject(GroundedFastPathReason.RANKING_PRESENT)
    if _has_multiple_windows(contract):
        reject(GroundedFastPathReason.MULTI_WINDOW_PRESENT)
    if not contract.selected_fields:
        reject(GroundedFastPathReason.SELECTED_FIELDS_MISSING)
    if not contract.entity_filters:
        reject(GroundedFastPathReason.ENTITY_FILTERS_MISSING)

    table = contract.tables[0].table if len(contract.tables) == 1 else ""
    if table and contract.primary_table != table:
        reject(GroundedFastPathReason.PRIMARY_TABLE_MISMATCH)
    aliases: list[str] = []
    for field in contract.selected_fields:
        alias = str(field.output_alias or field.column or "").strip()
        aliases.append(alias)
        if (
            not str(field.semantic_ref_id or "").startswith("semantic:")
            or not str(field.column or "").strip()
            or field.table != table
            or not alias
        ):
            reject(GroundedFastPathReason.SELECTED_FIELD_BINDING_INVALID)
    normalized_aliases = [alias.casefold() for alias in aliases]
    if len(set(normalized_aliases)) != len(normalized_aliases):
        reject(GroundedFastPathReason.SELECTED_FIELD_ALIAS_DUPLICATE)

    typed_filter = False
    for entity_filter in contract.entity_filters:
        operator = str(entity_filter.operator or "").strip().upper()
        allowed = {str(item or "").strip().upper() for item in entity_filter.allowed_operators}
        literal = entity_filter.literal_value
        if (
            not str(entity_filter.semantic_ref_id or "").startswith("semantic:")
            or not str(entity_filter.column or "").strip()
            or entity_filter.table != table
            or literal is None
            or (operator == "IN" and (not isinstance(literal, (list, tuple)) or not literal))
            or (operator != "IN" and isinstance(literal, (list, tuple, dict, set)))
        ):
            reject(GroundedFastPathReason.ENTITY_FILTER_BINDING_INVALID)
        if not operator or operator not in allowed:
            reject(GroundedFastPathReason.ENTITY_FILTER_OPERATOR_NOT_DECLARED)
        if operator not in {"EQ", "NE", "GT", "GTE", "LT", "LTE", "IN"}:
            reject(GroundedFastPathReason.ENTITY_FILTER_OPERATOR_NOT_SUPPORTED)
        if operator == "IN" and isinstance(literal, (list, tuple)) and len(literal) > 100:
            reject(GroundedFastPathReason.ENTITY_FILTER_VALUE_LIMIT_EXCEEDED)
        if entity_filter.is_unique_key or str(entity_filter.entity_identity or "").strip():
            typed_filter = True
    if contract.entity_filters and not typed_filter:
        reject(GroundedFastPathReason.TYPED_ENTITY_FILTER_MISSING)

    return GroundedFastPathDecision(
        eligible=not reasons,
        reason_codes=reasons,
    )


def _validate_metric_binding(
    contract: GroundedQueryContract,
    metric: Any,
    table: str,
    reject: Any,
    *,
    allow_native_time_grain: bool = False,
) -> None:
    binding_type = str(metric.binding_type or "").strip().lower()
    if binding_type == "field_aggregation":
        _validate_field_aggregation_metric(contract, metric, table, reject)
        return
    if (
        bool(str(metric.field_aggregation or "").strip())
        or bool(str(metric.source_field_ref_id or "").strip())
    ):
        reject(GroundedFastPathReason.FIELD_AGGREGATION_BINDING_INVALID)
    if binding_type != "published_metric" or not str(metric.semantic_ref_id or "").startswith("semantic:"):
        reject(GroundedFastPathReason.METRIC_NOT_PUBLISHED)
    if table and metric.table != table:
        reject(GroundedFastPathReason.METRIC_TABLE_MISMATCH)
    if not metric.source_columns:
        reject(GroundedFastPathReason.METRIC_SOURCE_COLUMNS_MISSING)
    if not compile_metric_formula(metric.formula, set(metric.source_columns)):
        reject(GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE)
    temporal_issue = semantic_metric_temporal_contract_issue(metric.model_dump(by_alias=False))
    if temporal_issue and not (
        allow_native_time_grain and _supports_native_grain_trend(metric)
    ):
        reject(
            GroundedFastPathReason.METRIC_TIME_SEMANTICS_NOT_EXECUTABLE,
            temporal_issue,
        )


def _validate_field_aggregation_metric(
    contract: GroundedQueryContract,
    metric: Any,
    table: str,
    reject: Any,
) -> None:
    """Admit only the exact COUNT derivation published by a COLUMN capability.

    Core chooses the already-read COLUMN ref and an allowlisted operator.  The
    Contract builder derives every remaining executable field.  Rechecking the
    complete derivation here prevents a hand-edited READY Contract from using
    the field-aggregation lane as arbitrary formula authority.
    """

    semantic_ref = str(metric.semantic_ref_id or "").strip()
    source_ref = str(metric.source_field_ref_id or "").strip()
    aggregation = str(metric.field_aggregation or "").strip().upper()
    source_columns = [
        str(item or "").strip()
        for item in metric.source_columns
        if str(item or "").strip()
    ]
    column = source_columns[0] if len(source_columns) == 1 else ""
    expected_formula = (
        "COUNT(DISTINCT `%s`)" % column
        if aggregation == "COUNT_DISTINCT" and column
        else "COUNT(`%s`)" % column
        if aggregation == "COUNT" and column
        else ""
    )
    expected_metric_key = (
        "%s_%s"
        % (
            "count_distinct" if aggregation == "COUNT_DISTINCT" else "count",
            column,
        )
        if aggregation in {"COUNT", "COUNT_DISTINCT"} and column
        else ""
    )
    table_binding = next(
        (item for item in contract.tables if item.table == table),
        None,
    )
    hint_matches = [
        item
        for item in contract.binding_hints.field_aggregations
        if str(item.field_ref or "").strip() == source_ref
        and _normalized_field_aggregation(item.aggregation) == aggregation
    ]
    capabilities = dict(metric.calculation_capabilities or {})
    raw_allowed = capabilities.get("allowedAggregations") or capabilities.get(
        "allowed_aggregations"
    )
    if isinstance(raw_allowed, str):
        raw_allowed = [raw_allowed]
    allowed = {
        _normalized_field_aggregation(item)
        for item in (raw_allowed or [])
        if _normalized_field_aggregation(item)
    }
    declared = _normalized_field_aggregation(
        capabilities.get("declaredAggregation")
        or capabilities.get("declared_aggregation")
    )

    if (
        not semantic_ref.startswith("semantic:")
        or source_ref != semantic_ref
        or aggregation not in {"COUNT", "COUNT_DISTINCT"}
        or len(source_columns) != 1
        or not expected_formula
        or str(metric.formula or "").strip() != expected_formula
        or str(metric.metric_key or "").strip() != expected_metric_key
        or metric.table != table
        or (
            table_binding is not None
            and table_binding.merchant_filter_column
            and column == table_binding.merchant_filter_column
        )
        or len(hint_matches) != 1
    ):
        reject(GroundedFastPathReason.FIELD_AGGREGATION_BINDING_INVALID)
    if aggregation not in allowed or declared != aggregation:
        reject(GroundedFastPathReason.FIELD_AGGREGATION_NOT_ALLOWLISTED)
    if not compile_metric_formula(metric.formula, set(source_columns)):
        reject(GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE)


def _validate_literal_entity_filters(
    contract: GroundedQueryContract,
    table: str,
    reject: Any,
) -> None:
    """Validate ordinary, same-table literal predicates without planning them."""

    filters = list(contract.entity_filters)
    hints = list(contract.binding_hints.entity_filters)
    if not filters and not hints:
        return
    if len(filters) != len(hints):
        reject(GroundedFastPathReason.ENTITY_FILTER_HINT_MISMATCH)

    unmatched_hints = list(hints)
    table_binding = next(
        (item for item in contract.tables if item.table == table),
        None,
    )
    for entity_filter in filters:
        operator = str(entity_filter.operator or "").strip().upper()
        allowed = {
            str(item or "").strip().upper()
            for item in entity_filter.allowed_operators
            if str(item or "").strip()
        }
        literal = entity_filter.literal_value
        if (
            not str(entity_filter.semantic_ref_id or "").startswith("semantic:")
            or not str(entity_filter.column or "").strip()
            or entity_filter.table != table
            or (
                table_binding is not None
                and table_binding.merchant_filter_column
                and entity_filter.column == table_binding.merchant_filter_column
            )
            or not _literal_filter_value_valid(operator, literal)
        ):
            reject(GroundedFastPathReason.ENTITY_FILTER_BINDING_INVALID)
        if not operator or operator not in allowed:
            reject(GroundedFastPathReason.ENTITY_FILTER_OPERATOR_NOT_DECLARED)
        if operator not in {"EQ", "NE", "GT", "GTE", "LT", "LTE", "IN"}:
            reject(GroundedFastPathReason.ENTITY_FILTER_OPERATOR_NOT_SUPPORTED)
        if operator == "IN" and isinstance(literal, (list, tuple)) and len(literal) > 100:
            reject(GroundedFastPathReason.ENTITY_FILTER_VALUE_LIMIT_EXCEEDED)

        matching_index = next(
            (
                index
                for index, hint in enumerate(unmatched_hints)
                if str(hint.field_ref or "").strip()
                == str(entity_filter.semantic_ref_id or "").strip()
                and _normalized_filter_operator(hint.operator) == operator
                and hint.literal_value == literal
            ),
            None,
        )
        if matching_index is None:
            reject(GroundedFastPathReason.ENTITY_FILTER_HINT_MISMATCH)
        else:
            unmatched_hints.pop(matching_index)
    if unmatched_hints:
        reject(GroundedFastPathReason.ENTITY_FILTER_HINT_MISMATCH)


def _literal_filter_value_valid(operator: str, value: Any) -> bool:
    def scalar(item: Any) -> bool:
        if item is None or isinstance(item, bool):
            return False
        if isinstance(item, float) and not math.isfinite(item):
            return False
        if isinstance(item, str):
            return bool(item.strip())
        return isinstance(item, (int, float))

    if operator == "IN":
        return bool(
            isinstance(value, (list, tuple))
            and value
            and len(value) <= 100
            and all(scalar(item) for item in value)
        )
    return scalar(value)


def _normalized_field_aggregation(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return {
        "COUNT": "COUNT",
        "COUNT_DISTINCT": "COUNT_DISTINCT",
        "DISTINCT_COUNT": "COUNT_DISTINCT",
        "COUNTDISTINCT": "COUNT_DISTINCT",
    }.get(normalized, "")


def _normalized_filter_operator(value: Any) -> str:
    normalized = str(value or "").strip().upper().replace(" ", "_")
    return {
        "=": "EQ",
        "==": "EQ",
        "EQ": "EQ",
        "!=": "NE",
        "NE": "NE",
        ">": "GT",
        "GT": "GT",
        ">=": "GTE",
        "GTE": "GTE",
        "<": "LT",
        "LT": "LT",
        "<=": "LTE",
        "LTE": "LTE",
        "IN": "IN",
    }.get(normalized, "")


def _validate_metric_set_time_compatibility(
    contract: GroundedQueryContract,
    table: str,
    reject: Any,
) -> None:
    if not contract.metrics:
        return
    table_binding = next(
        (item for item in contract.tables if item.table == table),
        None,
    )
    default_time_column = (
        str(table_binding.time_column or "").strip()
        if table_binding is not None
        else ""
    )
    time_columns = {
        str(metric.time_column or default_time_column).strip()
        for metric in contract.metrics
        if str(metric.time_column or default_time_column).strip()
    }
    if len(time_columns) != 1:
        reject(
            GroundedFastPathReason.METRIC_TIME_COLUMN_MISSING
            if not time_columns
            else GroundedFastPathReason.METRIC_TIME_COLUMN_INCOMPATIBLE
        )

    selection_policies = {
        str(
            dict(metric.time_semantics or {}).get("selectionPolicy")
            or dict(metric.time_semantics or {}).get("selection_policy")
            or "period_window"
        )
        .strip()
        .lower()
        for metric in contract.metrics
    }
    if len(selection_policies) > 1:
        reject(GroundedFastPathReason.METRIC_TIME_POLICY_INCOMPATIBLE)

    anchor_policies = {
        str(
            dict(metric.time_semantics or {}).get("asOfPolicy")
            or dict(metric.time_semantics or {}).get("as_of_policy")
            or metric.anchor_policy
            or ""
        )
        .strip()
        .lower()
        for metric in contract.metrics
        if str(
            dict(metric.time_semantics or {}).get("asOfPolicy")
            or dict(metric.time_semantics or {}).get("as_of_policy")
            or metric.anchor_policy
            or ""
        ).strip()
    }
    contract_anchor = str(contract.time_range.anchor_policy or "").strip().lower()
    if len(anchor_policies) > 1 or (
        len(anchor_policies) == 1
        and contract_anchor
        and contract_anchor not in anchor_policies
    ):
        reject(GroundedFastPathReason.METRIC_ANCHOR_POLICY_INCOMPATIBLE)


def _supports_native_grain_trend(metric: Any) -> bool:
    time_semantics = dict(metric.time_semantics or {})
    selection_policy = str(
        time_semantics.get("selectionPolicy")
        or time_semantics.get("selection_policy")
        or ""
    ).strip().lower()
    capabilities = dict(metric.calculation_capabilities or {})
    modes = capabilities.get("nativeGrainAnalysisModes") or capabilities.get(
        "native_grain_analysis_modes"
    )
    normalized_modes = {
        str(item or "").strip().upper()
        for item in (modes if isinstance(modes, (list, tuple, set)) else [modes])
        if str(item or "").strip()
    }
    return bool(
        selection_policy == "per_time_grain"
        and str(metric.applicable_time_grain or "").strip()
        and normalized_modes.intersection({"TREND", "TIME_SERIES", "TIMESERIES"})
    )


def _ranking_present(contract: GroundedQueryContract) -> bool:
    ranking = contract.ranking
    hints = contract.binding_hints.ranking
    return bool(
        ranking.enabled
        or str(ranking.direction or "").strip()
        or int(ranking.limit or 0) != 0
        or str(ranking.metric_ref_id or "").strip()
        or str(ranking.dimension_ref_id or "").strip()
        or str(hints.order or "").strip()
        or int(hints.limit or 0) != 0
        or str(hints.metric_ref or "").strip()
    )


def _has_multiple_windows(contract: GroundedQueryContract) -> bool:
    """Detect comparison/secondary-window semantics from structured state."""

    time_range = contract.time_range
    window_role = str(time_range.window_role or "").strip().lower()
    return bool(
        window_role not in {"", "primary"}
        or str(time_range.comparison_type or "").strip()
        or int(time_range.offset_days or 0) != 0
        or _contains_multiple_declared_windows(contract.binding_hints)
    )


def _contains_multiple_declared_windows(binding_hints: Any) -> bool:
    """Remain fail-closed if the Contract protocol later adds window arrays."""

    for attribute in ("time_ranges", "time_windows", "comparison_windows"):
        value = getattr(binding_hints, attribute, None)
        if isinstance(value, (list, tuple)) and len(value) > 1:
            return True
    return False
