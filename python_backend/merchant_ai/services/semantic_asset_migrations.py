from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, List, Tuple


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

ENTITY_ROLES = frozenset({"KEY", "ENTITY", "ENTITY_KEY", "PRIMARY_KEY", "IDENTIFIER"})
SIMPLE_RATIO_FORMULA = re.compile(
    r"\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*/\s*"
    r"NULLIF\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*,\s*0\s*\)\s*",
    flags=re.IGNORECASE,
)


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

    semantic_columns = migrated.get("semanticColumns") or []
    has_entity_fields = any(
        str(field.get("role") or field.get("semanticRole") or "").strip().upper() in ENTITY_ROLES
        for field in semantic_columns
        if isinstance(field, dict)
    )
    if has_entity_fields and table_time_column and migrated.get("entityLookupPolicy") is None:
        # Fail closed for ID lookups with no user time scope. A default window
        # would silently change lookup semantics, while `clarify` preserves the
        # table's declared time axis without inventing a duration.
        migrated["entityLookupPolicy"] = {"mode": "clarify", "timeColumn": table_time_column}
        changes.append("%s.entityLookupPolicy" % (table or "asset"))

    for metric in migrated.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
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
            match = SIMPLE_RATIO_FORMULA.fullmatch(formula)
            if match:
                metric["formula"] = "SUM(%s) / NULLIF(SUM(%s), 0)" % (match.group(1), match.group(2))
                changes.append(prefix + ".formula")

    return migrated, changes, errors

