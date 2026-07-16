from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable


PERIOD_WINDOW_AGGREGATION_POLICIES = frozenset(
    {
        "period_rollup",
        "period_recompute",
        "ratio_of_sums",
    }
)
LATEST_AS_OF_AGGREGATION_POLICIES = frozenset({"latest_value_only"})
PER_GRAIN_AGGREGATION_POLICIES = frozenset({"daily_value_only"})
TEMPORAL_SELECTION_BY_AGGREGATION_POLICY = {
    **{policy: "period_window" for policy in PERIOD_WINDOW_AGGREGATION_POLICIES},
    **{policy: "latest_as_of" for policy in LATEST_AS_OF_AGGREGATION_POLICIES},
    **{policy: "per_time_grain" for policy in PER_GRAIN_AGGREGATION_POLICIES},
}
EXECUTABLE_AS_OF_POLICIES_BY_SELECTION = {
    "period_window": frozenset({"calendar", "latest_available_partition"}),
    "latest_as_of": frozenset({"calendar", "latest_available_partition", "latest_observation"}),
    "per_time_grain": frozenset({"calendar", "latest_available_partition"}),
}
EXECUTABLE_MISSING_DATA_POLICIES = frozenset({"disclose_unknown", "fail_closed"})
EXECUTABLE_ZERO_VALUE_POLICIES = frozenset({"preserve_observed_zero"})


def metric_aggregation_policy(payload: Dict[str, Any]) -> str:
    sealed = (payload or {}).get("semanticContract") or (payload or {}).get("semantic_contract") or {}
    return str(
        (payload or {}).get("aggregationPolicy")
        or (payload or {}).get("aggregation_policy")
        or (sealed.get("aggregationPolicy") if isinstance(sealed, dict) else "")
        or (sealed.get("aggregation_policy") if isinstance(sealed, dict) else "")
        or ""
    ).strip().lower()


def metric_time_selection_policy(payload: Dict[str, Any]) -> str:
    """Resolve the generic temporal operator declared by a metric asset.

    New assets may state ``timeSemantics.selectionPolicy`` explicitly.  The
    aggregation-policy mapping keeps already-published assets executable while
    remaining independent of business metric names, tables, and columns.
    """

    time_semantics = metric_time_semantics(payload)
    if isinstance(time_semantics, dict):
        declared = str(
            time_semantics.get("selectionPolicy")
            or time_semantics.get("selection_policy")
            or ""
        ).strip().lower()
        if declared:
            return declared
    return TEMPORAL_SELECTION_BY_AGGREGATION_POLICY.get(metric_aggregation_policy(payload), "")


def metric_time_semantics(payload: Dict[str, Any]) -> Dict[str, Any]:
    sealed = (payload or {}).get("semanticContract") or (payload or {}).get("semantic_contract") or {}
    raw = (
        (payload or {}).get("timeSemantics")
        or (payload or {}).get("time_semantics")
        or (sealed.get("timeSemantics") if isinstance(sealed, dict) else {})
        or {}
    )
    return dict(raw) if isinstance(raw, dict) else {}


def metric_missing_data_policy(payload: Dict[str, Any]) -> str:
    semantics = metric_time_semantics(payload)
    sealed = (payload or {}).get("semanticContract") or (payload or {}).get("semantic_contract") or {}
    return str(
        (payload or {}).get("missingValuePolicy")
        or (payload or {}).get("missing_value_policy")
        or semantics.get("missingDataPolicy")
        or semantics.get("missing_data_policy")
        or (sealed.get("missingValuePolicy") if isinstance(sealed, dict) else "")
        or ""
    ).strip().lower()


def metric_zero_value_policy(payload: Dict[str, Any]) -> str:
    semantics = metric_time_semantics(payload)
    sealed = (payload or {}).get("semanticContract") or (payload or {}).get("semantic_contract") or {}
    return str(
        (payload or {}).get("zeroValueMeaning")
        or (payload or {}).get("zero_value_meaning")
        or semantics.get("zeroValuePolicy")
        or semantics.get("zero_value_policy")
        or (sealed.get("zeroValueMeaning") if isinstance(sealed, dict) else "")
        or ""
    ).strip().lower()


def semantic_metric_temporal_contract_issue(payload: Dict[str, Any]) -> str:
    """Validate the executable part of a published metric contract."""

    aggregation_policy = metric_aggregation_policy(payload)
    expected_selection = TEMPORAL_SELECTION_BY_AGGREGATION_POLICY.get(aggregation_policy, "")
    if not expected_selection:
        return "published metric has no supported aggregationPolicy execution contract"
    if not str((payload or {}).get("metricGrain") or (payload or {}).get("metric_grain") or "").strip():
        return "published metric has no metricGrain execution contract"
    applicable_grain = str(
        (payload or {}).get("applicableTimeGrain")
        or (payload or {}).get("applicable_time_grain")
        or ""
    ).strip().lower()
    if not applicable_grain:
        return "published metric has no applicableTimeGrain execution contract"
    if aggregation_policy in PER_GRAIN_AGGREGATION_POLICIES and applicable_grain != "day":
        return "daily-value metric is not executable outside day grain"
    if not str((payload or {}).get("timeColumn") or (payload or {}).get("time_column") or "").strip():
        return "published metric has no timeColumn execution contract"
    semantics = metric_time_semantics(payload)
    if not semantics:
        return "published metric has no timeSemantics execution contract"
    selection = metric_time_selection_policy(payload)
    if selection != expected_selection:
        return "published metric time selection conflicts with aggregationPolicy"
    as_of_policy = str(
        semantics.get("asOfPolicy") or semantics.get("as_of_policy") or ""
    ).strip().lower()
    if as_of_policy not in EXECUTABLE_AS_OF_POLICIES_BY_SELECTION.get(selection, frozenset()):
        return "published metric asOfPolicy is not executable for its time selection"
    if metric_missing_data_policy(payload) not in EXECUTABLE_MISSING_DATA_POLICIES:
        return "published metric missingDataPolicy is not executable"
    if metric_zero_value_policy(payload) not in EXECUTABLE_ZERO_VALUE_POLICIES:
        return "published metric zeroValuePolicy is not executable"
    return ""


def semantic_metric_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the immutable, execution-relevant part of a semantic resolution."""
    source_columns = payload.get("sourceColumns") or payload.get("source_columns") or []
    return {
        "semanticRefId": str(payload.get("semanticRefId") or payload.get("semantic_ref_id") or "").strip(),
        "metricKey": str(payload.get("metricKey") or payload.get("metric_key") or "").strip(),
        "ownerTable": str(payload.get("ownerTable") or payload.get("owner_table") or "").strip(),
        "formula": normalize_formula(payload.get("formula")),
        "sourceColumns": normalize_text_list(source_columns),
        "unit": str(payload.get("unit") or "").strip(),
        "metricGrain": str(payload.get("metricGrain") or payload.get("metric_grain") or "").strip().lower(),
        "timeColumn": str(payload.get("timeColumn") or payload.get("time_column") or "").strip(),
        "aggregationPolicy": metric_aggregation_policy(payload),
        "applicableTimeGrain": str(
            payload.get("applicableTimeGrain")
            or payload.get("applicable_time_grain")
            or ""
        ).strip().lower(),
        "timeSemantics": normalize_json_value(
            metric_time_semantics(payload)
        ),
        "missingValuePolicy": metric_missing_data_policy(payload),
        "zeroValueMeaning": metric_zero_value_policy(payload),
    }


def seal_semantic_metric_resolution(payload: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    """Attach a stable snapshot so later stages cannot silently change a metric."""
    updated = dict(payload or {})
    contract = semantic_metric_contract(updated)
    if not semantic_metric_contract_complete(contract):
        return updated
    if force or not isinstance(updated.get("semanticContract"), dict):
        updated["semanticContract"] = contract
        updated["semanticContractHash"] = semantic_metric_contract_hash(contract)
    return updated


def semantic_metric_contract_issue(payload: Dict[str, Any], execution_table: str = "") -> str:
    current = semantic_metric_contract(payload or {})
    if not semantic_metric_contract_complete(current):
        return "semantic metric contract is incomplete"
    sealed = (payload or {}).get("semanticContract")
    sealed_hash = str((payload or {}).get("semanticContractHash") or "")
    if not isinstance(sealed, dict) or not sealed_hash:
        return "semantic metric contract is not sealed"
    canonical_sealed = semantic_metric_contract(sealed)
    if not semantic_metric_contract_complete(canonical_sealed):
        return "sealed semantic metric contract is incomplete"
    if semantic_metric_contract_hash(canonical_sealed) != sealed_hash:
        return "sealed semantic metric contract hash is invalid"
    if current != canonical_sealed:
        return "semantic metric contract drifted after resolution"
    if execution_table and current["ownerTable"] != execution_table:
        return "semantic metric ownerTable does not match the execution table"
    return ""


def semantic_metric_contract_hash(contract: Dict[str, Any]) -> str:
    canonical = json.dumps(semantic_metric_contract(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_metric_contract_complete(contract: Dict[str, Any]) -> bool:
    return bool(
        str(contract.get("semanticRefId") or "").startswith("semantic:")
        and contract.get("metricKey")
        and contract.get("ownerTable")
        and contract.get("formula")
        and contract.get("sourceColumns")
    )


def normalize_text_list(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def normalize_formula(value: Any) -> str:
    # SQL compilation quotes governed identifiers for safe execution. That is
    # representational, not semantic, so `column` and column compare equally.
    # Operators, predicates, functions, and literals remain intact, ensuring
    # genuine formula changes still invalidate the sealed contract.
    return " ".join(str(value or "").replace("`", "").strip().split())


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
