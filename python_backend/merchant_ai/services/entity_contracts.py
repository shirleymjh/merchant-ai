from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any, Dict, Iterable, Set

from merchant_ai.models import EntityReference, NodePlanContract


def entity_comparison_policy(reference: EntityReference) -> str:
    policy = str(reference.comparison_policy or "").strip().lower()
    if policy:
        return policy
    if reference.value_type == "integer":
        return "integer"
    if reference.value_type == "number":
        return "decimal"
    return "exact"


def canonical_entity_value(value: Any, policy: str) -> str:
    normalized_policy = str(policy or "exact").strip().lower()
    if normalized_policy == "integer":
        try:
            number = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return "invalid:%s" % str(value)
        if not number.is_finite() or number != number.to_integral_value():
            return "invalid:%s" % str(value)
        return str(int(number))
    if normalized_policy in {"decimal", "number", "numeric"}:
        try:
            number = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return "invalid:%s" % str(value)
        if not number.is_finite():
            return "invalid:%s" % str(value)
        normalized = format(number.normalize(), "f")
        return "0" if normalized in {"-0", "-0.0"} else normalized
    text = str(value)
    if normalized_policy in {"case_insensitive", "casefold"}:
        return text.casefold()
    if normalized_policy in {"trimmed", "trim"}:
        return text.strip()
    if normalized_policy in {"trimmed_case_insensitive", "trim_casefold"}:
        return text.strip().casefold()
    return text


def canonical_entity_values(values: Iterable[Any], policy: str) -> Set[str]:
    return {
        canonical_entity_value(value, policy)
        for value in values or []
        if value is not None and str(value) != ""
    }


def entity_value_display_map(values: Iterable[Any], policy: str) -> Dict[str, Any]:
    return {
        canonical_entity_value(value, policy): value
        for value in values or []
        if value is not None and str(value) != ""
    }


def entity_filter_contract_hash(contract: NodePlanContract) -> str:
    payload = {
        "taskId": contract.task_id,
        "preferredTable": contract.preferred_table,
        "filterColumn": contract.filter_column,
        "filterValues": contract.filter_values,
        "obligations": [item.model_dump(by_alias=True) for item in contract.entity_filter_obligations],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def entity_filter_sql_hash(sql: str) -> str:
    return hashlib.sha256(str(sql or "").strip().encode("utf-8")).hexdigest()


def entity_value_hash(value: str, contract_hash: str) -> str:
    return hashlib.sha256((str(contract_hash or "") + ":" + value).encode("utf-8")).hexdigest()
