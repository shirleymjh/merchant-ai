from __future__ import annotations

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
    METRIC_TIME_SEMANTICS_NOT_EXECUTABLE = (
        "METRIC_TIME_SEMANTICS_NOT_EXECUTABLE"
    )


class GroundedExecutionMode(str, Enum):
    """Execution authority selected after a Contract becomes READY."""

    UNDECIDED = "UNDECIDED"
    DETERMINISTIC_METRIC = "DETERMINISTIC_METRIC"
    CORE_SQL_REQUIRED = "CORE_SQL_REQUIRED"


class GroundedExecutionReason(str, Enum):
    """Stable routing reasons exposed to Core and runtime observers."""

    CONTRACT_NOT_READY = "CONTRACT_NOT_READY"
    SINGLE_METRIC_FAST_PATH_ELIGIBLE = "SINGLE_METRIC_FAST_PATH_ELIGIBLE"
    COMPLEX_QUERY_REQUIRES_CORE_SQL = "COMPLEX_QUERY_REQUIRES_CORE_SQL"


class GroundedFastPathDecision(APIModel):
    """Pure policy result; it neither compiles nor executes a query."""

    eligible: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    reason_details: dict[str, str] = Field(default_factory=dict)


def evaluate_single_metric_fast_path(
    contract: GroundedQueryContract,
) -> GroundedFastPathDecision:
    """Return whether a grounded Contract is safe for deterministic execution.

    This gate intentionally consumes only normalized Contract fields and
    published metric capabilities. It never inspects the question, metric
    name, table name, aliases, or other lexical business signals.
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
        dimension
        for dimension in contract.dimensions
        if str(dimension.usage or "").strip().lower() == "group_by"
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

        if (
            binding_type != "published_metric"
            or not str(metric.semantic_ref_id or "").startswith("semantic:")
        ):
            reject(GroundedFastPathReason.METRIC_NOT_PUBLISHED)

        if len(contract.tables) == 1 and metric.table != contract.tables[0].table:
            reject(GroundedFastPathReason.METRIC_TABLE_MISMATCH)

        if not metric.source_columns:
            reject(GroundedFastPathReason.METRIC_SOURCE_COLUMNS_MISSING)

        if not compile_metric_formula(metric.formula, set(metric.source_columns)):
            reject(GroundedFastPathReason.METRIC_FORMULA_NOT_EXECUTABLE)

        temporal_issue = semantic_metric_temporal_contract_issue(
            metric.model_dump(by_alias=False)
        )
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
