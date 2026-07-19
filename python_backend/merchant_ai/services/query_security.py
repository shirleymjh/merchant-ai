from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Set, Tuple

from merchant_ai.models import NodePlanContract, PlanningAssetPack
from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.services.assets import (
    normalize_column_display_policy,
    normalize_masking_policy,
    normalize_visibility_policy,
)


DEFAULT_ACCESS_ROLE = load_authorization_policy().default_access_role


def table_asset_metadata(asset_pack: PlanningAssetPack, table: str) -> Dict[str, Any]:
    for entry in asset_pack.tables:
        if (entry.table or entry.key) == table and isinstance(entry.metadata, dict):
            return entry.metadata
    return {}


def table_field_semantics(asset_pack: PlanningAssetPack, table: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entry in asset_pack.fields:
        if entry.table != table or not entry.key:
            continue
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        semantic = metadata.get("semantic") if isinstance(metadata.get("semantic"), dict) else metadata
        if isinstance(semantic, dict):
            result[entry.key] = semantic
    return result


def declared_result_access_policy(table_metadata: Dict[str, Any], semantic_role: str) -> Dict[str, Any]:
    """Return an explicitly declared result policy for a semantic output role.

    A result policy is intentionally separate from raw-column access.  For
    example, a governed aggregate may be publishable while its physical source
    column remains unavailable to detail queries.  Missing roles fail closed.
    """

    policies = table_metadata.get("resultAccessPolicies")
    if not isinstance(policies, dict):
        return {}
    role = str(semantic_role or "").strip().upper()
    declared = policies.get(role) or policies.get("*")
    if not role or not isinstance(declared, dict):
        return {}
    return {
        "semanticRole": role,
        "visibilityPolicy": normalize_visibility_policy(declared.get("visibilityPolicy") or {}),
        "maskingPolicy": normalize_masking_policy(declared.get("maskingPolicy") or {}),
        **normalize_column_display_policy(declared),
    }


def configured_default_detail_columns(asset_pack: PlanningAssetPack, table: str, columns: Set[str]) -> List[str]:
    configured: List[Tuple[int, str]] = []
    for column, semantic in table_field_semantics(asset_pack, table).items():
        if column not in columns:
            continue
        display_policy = normalize_column_display_policy(semantic)
        scenarios = {str(item).lower() for item in display_policy.get("displayScenarios") or []}
        if not display_policy.get("defaultVisible") and "detail" not in scenarios:
            continue
        configured.append((int(display_policy.get("displayPriority") or 1000), column))
    configured.sort(key=lambda item: (item[0], item[1]))
    return [column for _, column in configured]


def configured_contract_detail_columns(contract: NodePlanContract, columns: Set[str], visible_columns: Set[str]) -> List[str]:
    configured: List[Tuple[int, str]] = []
    for column, policy in (contract.column_display_policy or {}).items():
        if column not in columns or column not in visible_columns:
            continue
        display_policy = normalize_column_display_policy(policy)
        scenarios = {str(item).lower() for item in display_policy.get("displayScenarios") or []}
        if not display_policy.get("defaultVisible") and "detail" not in scenarios:
            continue
        configured.append((int(display_policy.get("displayPriority") or 1000), column))
    configured.sort(key=lambda item: (item[0], item[1]))
    return [column for _, column in configured]


def role_allowed_for_column(policy: Dict[str, Any], access_role: str) -> bool:
    level = str(policy.get("level") or "public")
    roles = [str(item) for item in policy.get("allowedRoles") or [] if str(item or "").strip()]
    if level == "public":
        return True
    if level == "hidden":
        return False
    if not roles:
        return False
    return str(access_role or DEFAULT_ACCESS_ROLE) in roles


def contract_masked_columns_map(contract: NodePlanContract) -> Dict[str, str]:
    return {
        str(column): str(strategy)
        for column, strategy in (contract.masked_columns or {}).items()
        if str(column or "").strip() and str(strategy or "").strip()
    }


def mask_value(value: Any, strategy: str) -> Any:
    if value is None:
        return None
    text = str(value)
    kind = str(strategy or "none").lower()
    if kind == "full":
        return "***"
    if kind == "hash":
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if kind == "partial":
        if len(text) <= 2:
            return "*" * len(text)
        if len(text) <= 6:
            return text[:1] + "*" * (len(text) - 2) + text[-1:]
        return text[:2] + "*" * max(1, len(text) - 4) + text[-2:]
    return value


def apply_column_masks(rows: List[Dict[str, Any]], contract: NodePlanContract) -> List[Dict[str, Any]]:
    masked_columns = contract_masked_columns_map(contract)
    if not rows or not masked_columns:
        return list(rows)
    result: List[Dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        for column, strategy in masked_columns.items():
            if column in next_row:
                next_row[column] = mask_value(next_row.get(column), strategy)
        result.append(next_row)
    return result
