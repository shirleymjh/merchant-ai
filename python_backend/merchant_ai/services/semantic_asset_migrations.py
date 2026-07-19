from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError


TEMPORAL_DEFAULTS_BY_AGGREGATION_POLICY: Dict[str, Dict[str, str]] = {
    "period_rollup": {
        "applicableTimeGrain": "period",
        "selectionPolicy": "period_window",
    },
    "period_recompute": {
        "applicableTimeGrain": "period",
        "selectionPolicy": "period_window",
    },
    "ratio_of_sums": {
        "applicableTimeGrain": "period",
        "selectionPolicy": "period_window",
    },
    "daily_value_only": {
        "applicableTimeGrain": "day",
        "selectionPolicy": "per_time_grain",
    },
    "latest_value_only": {
        "applicableTimeGrain": "day",
        "selectionPolicy": "latest_as_of",
    },
}

def migrate_published_semantic_asset(asset: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Materialize executable contracts without guessing business metric identities.

    The migration is intentionally policy-driven: it only expands an already
    declared aggregation policy and table time column. Existing declarations
    always win, and unsupported/inconsistent assets remain validation errors.
    """

    migrated = deepcopy(asset or {})
    changes: List[str] = []
    errors: List[str] = []
    table = str(migrated.get("tableName") or "").strip()
    table_time_column = str(migrated.get("timeColumn") or "").strip()
    metrics = [metric for metric in migrated.get("metrics") or [] if isinstance(metric, dict)]
    metric_keys = {
        str(metric.get("metricKey") or metric.get("key") or "").strip()
        for metric in metrics
        if str(metric.get("metricKey") or metric.get("key") or "").strip()
    }
    physical_columns = {
        str(column.get("columnName") or column.get("Field") or column.get("name") or "").strip()
        for column in migrated.get("schemaColumns") or migrated.get("schema") or []
        if isinstance(column, dict)
        and str(column.get("columnName") or column.get("Field") or column.get("name") or "").strip()
    }

    semantic_columns = migrated.get("semanticColumns") or []
    has_entity_fields = any(
        bool(
            str(field.get("canonicalEntityRef") or "").strip()
            or field.get("isUniqueEntityKey") is True
            or field.get("filterOperators")
        )
        for field in semantic_columns
        if isinstance(field, dict)
    )
    if has_entity_fields and table_time_column and migrated.get("entityLookupPolicy") is None:
        # Fail closed for ID lookups with no user time scope. A default window
        # would silently change lookup semantics, while `clarify` preserves the
        # table's declared time axis without inventing a duration.
        migrated["entityLookupPolicy"] = {"mode": "clarify", "timeColumn": table_time_column}
        changes.append("%s.entityLookupPolicy" % (table or "asset"))

    for metric in metrics:
        metric_key = str(metric.get("metricKey") or metric.get("key") or "").strip()
        aggregation_policy = str(metric.get("aggregationPolicy") or "").strip().lower()
        defaults = TEMPORAL_DEFAULTS_BY_AGGREGATION_POLICY.get(aggregation_policy)
        if not defaults:
            errors.append("%s.%s: unsupported or missing aggregationPolicy" % (table, metric_key or "<unknown>"))
            continue
        if not table_time_column and not str(metric.get("timeColumn") or "").strip():
            errors.append("%s.%s: no declared metric/table timeColumn" % (table, metric_key or "<unknown>"))
            continue

        prefix = "%s.%s" % (table, metric_key or "<unknown>")
        dependency_refs = _canonical_metric_dependencies(
            metric,
            metric_keys,
            metric_key,
            physical_columns,
        )
        if dependency_refs:
            existing_requires_metrics = _metric_dependencies_for_keys(
                metric,
                ("requiresMetrics",),
            )
            if dependency_refs != existing_requires_metrics:
                metric["requiresMetrics"] = dependency_refs
                changes.append(prefix + ".requiresMetrics")
            source_columns = [
                str(value or "").strip()
                for value in metric.get("sourceColumns") or metric.get("source_columns") or []
                if str(value or "").strip()
            ]
            physical_source_columns = [
                value for value in source_columns if value not in set(dependency_refs)
            ]
            if physical_source_columns != source_columns:
                metric["sourceColumns"] = physical_source_columns
                changes.append(prefix + ".sourceColumns")
        if not str(metric.get("applicableTimeGrain") or "").strip():
            metric["applicableTimeGrain"] = defaults["applicableTimeGrain"]
            changes.append(prefix + ".applicableTimeGrain")
        if not str(metric.get("timeColumn") or "").strip() and table_time_column:
            metric["timeColumn"] = table_time_column
            changes.append(prefix + ".timeColumn")

        raw_time_semantics = metric.get("timeSemantics")
        if raw_time_semantics is None:
            raw_time_semantics = {}
            metric["timeSemantics"] = raw_time_semantics
        if not isinstance(raw_time_semantics, dict):
            errors.append(prefix + ": timeSemantics is not an object")
            continue
        semantic_defaults = {
            "selectionPolicy": defaults["selectionPolicy"],
            "asOfPolicy": "latest_available_partition",
            "missingDataPolicy": "disclose_unknown",
            "zeroValuePolicy": "preserve_observed_zero",
        }
        for key, value in semantic_defaults.items():
            if not str(raw_time_semantics.get(key) or "").strip():
                raw_time_semantics[key] = value
                changes.append(prefix + ".timeSemantics." + key)

        if aggregation_policy == "ratio_of_sums":
            formula = str(metric.get("formula") or metric.get("metricFormula") or "").strip()
            ratio_columns = _simple_ratio_columns(formula)
            if ratio_columns:
                numerator, denominator = ratio_columns
                metric["formula"] = "SUM(%s) / NULLIF(SUM(%s), 0)" % (numerator, denominator)
                changes.append(prefix + ".formula")

    return migrated, changes, errors


def _canonical_metric_dependencies(
    metric: Dict[str, Any],
    metric_keys: set[str],
    metric_key: str,
    physical_columns: set[str],
) -> List[str]:
    """Materialize exact metric-to-metric lineage without business inference.

    Only canonical metric keys declared directly, retained by semantic cleanup,
    or referenced as SQL columns in the published formula are eligible.  A
    similarly named business phrase can never create a dependency.
    """

    declared = _declared_metric_dependencies(metric)
    result = [ref for ref in declared if ref and ref != metric_key]
    candidates: List[str] = []
    cleanup = metric.get("semanticCleanup")
    if isinstance(cleanup, dict):
        candidates.extend(
            str(value or "").strip()
            for value in cleanup.get("droppedNonSchemaSourceColumns") or []
        )
    candidates.extend(
        value
        for value in (
            str(item or "").strip()
            for item in metric.get("sourceColumns") or metric.get("source_columns") or []
        )
        if value not in physical_columns
    )
    candidates.extend(
        value
        for value in _formula_column_names(
            str(metric.get("formula") or metric.get("metricFormula") or "")
        )
        if value not in physical_columns
    )
    for candidate in candidates:
        if not candidate or candidate == metric_key or candidate not in metric_keys:
            continue
        if candidate not in result:
            result.append(candidate)
    return result


def _declared_metric_dependencies(metric: Dict[str, Any]) -> List[str]:
    return _metric_dependencies_for_keys(
        metric,
        ("requiresMetrics", "metricDependencies", "externalMetricRefs"),
    )


def _metric_dependencies_for_keys(
    metric: Dict[str, Any],
    keys: tuple[str, ...],
) -> List[str]:
    result: List[str] = []
    for key in keys:
        for value in metric.get(key) or []:
            if isinstance(value, dict):
                ref = str(value.get("metricRef") or value.get("metricKey") or "").strip()
            else:
                ref = str(value or "").strip()
            if ref and ref not in result:
                result.append(ref)
    return result


def _formula_column_names(formula: str) -> List[str]:
    if not formula:
        return []
    try:
        expression = parse_one(formula, read="mysql")
    except (ParseError, ValueError, TypeError):
        return []
    result: List[str] = []
    for column in expression.find_all(exp.Column):
        name = str(column.name or "").strip()
        if name and name not in result:
            result.append(name)
    return result


def _simple_ratio_columns(formula: str) -> tuple[str, str] | None:
    """Return columns only for the exact governed ``column / NULLIF(column, 0)`` shape."""

    if not formula:
        return None
    try:
        expression = parse_one(formula, read="mysql")
    except (ParseError, ValueError, TypeError):
        return None
    if not isinstance(expression, exp.Div):
        return None
    numerator = expression.this
    denominator_guard = expression.expression
    if not isinstance(numerator, exp.Column) or not isinstance(denominator_guard, exp.Nullif):
        return None
    denominator = denominator_guard.this
    zero = denominator_guard.expression
    if not isinstance(denominator, exp.Column) or not isinstance(zero, exp.Literal):
        return None
    if zero.is_string or str(zero.this).strip() != "0":
        return None
    if numerator.table or denominator.table:
        return None
    return numerator.name, denominator.name
