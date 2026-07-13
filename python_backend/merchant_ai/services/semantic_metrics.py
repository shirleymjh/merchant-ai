from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable


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
