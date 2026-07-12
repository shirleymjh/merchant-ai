from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from merchant_ai.config import Settings
from merchant_ai.models import NodePlanContract


@dataclass
class AccessDecision:
    allowed: bool = True
    code: str = ""
    message: str = ""
    table: str = ""
    checked_columns: List[str] = field(default_factory=list)
    denied_columns: List[str] = field(default_factory=list)
    masked_columns: Dict[str, str] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)


class AccessControlService:
    """Table/column ACL and query audit for merchant BI execution."""

    def __init__(self, settings: Settings, root: Path | None = None):
        self.settings = settings
        self.root = root or (settings.resolved_workspace_path / "ops" / "access_control")
        self.policy_path = self.root / "merchant_acl.json"
        self.audit_path = self.root / "query_audit.jsonl"

    def authorize_contract(self, contract: NodePlanContract, sql: str = "", run_id: str = "") -> AccessDecision:
        policy = self._load_policy()
        table = contract.preferred_table
        role = contract.access_role or "merchant_analyst"
        merchant_id = contract.merchant_id or self.settings.merchant_id
        table_policy = self._table_policy(policy, table)
        allowed_merchants = self._string_set(policy.get("allowedMerchantIds") or policy.get("allowedMerchants"))
        denied_tables = self._string_set(policy.get("deniedTables"))
        allowed_tables = self._string_set(policy.get("allowedTables"))
        if allowed_merchants and merchant_id not in allowed_merchants:
            return self._deny("MERCHANT_SCOPE_DENIED", "merchant is not allowed by ACL", contract, sql, run_id, [])
        if table in denied_tables:
            return self._deny("TABLE_DENIED", "table is explicitly denied by ACL", contract, sql, run_id, [])
        if allowed_tables and table not in allowed_tables:
            return self._deny("TABLE_NOT_ALLOWED", "table is not in ACL allowlist", contract, sql, run_id, [])
        table_roles = self._string_set(table_policy.get("allowedRoles"))
        if table_roles and role not in table_roles:
            return self._deny("TABLE_ROLE_DENIED", "role cannot access table", contract, sql, run_id, [])
        sql_columns = self._columns_from_sql(sql, contract.allowed_columns)
        checked_columns = sorted(sql_columns or set(contract.required_columns or []) or set(contract.visible_columns or []))
        column_policies = table_policy.get("columns") if isinstance(table_policy.get("columns"), dict) else {}
        denied_columns: List[str] = []
        masked_columns = dict(contract.masked_columns or {})
        for column in checked_columns:
            if column in set(contract.internal_only_columns or []) and column not in set(contract.required_columns or []):
                denied_columns.append(column)
                continue
            column_policy = column_policies.get(column) if isinstance(column_policies, dict) else {}
            if isinstance(column_policy, dict):
                if bool(column_policy.get("denied")):
                    denied_columns.append(column)
                    continue
                allowed_roles = self._string_set(column_policy.get("allowedRoles"))
                if allowed_roles and role not in allowed_roles:
                    denied_columns.append(column)
                    continue
                strategy = str(column_policy.get("mask") or column_policy.get("maskingStrategy") or "")
                if strategy and strategy != "none":
                    masked_columns[column] = strategy
        if denied_columns:
            return self._deny("COLUMN_DENIED", "one or more columns are denied by ACL", contract, sql, run_id, checked_columns, denied_columns)
        return self._allow(contract, sql, run_id, checked_columns, masked_columns)

    def record_query_audit(self, decision: AccessDecision, *, row_count: int = 0, status: str = "") -> Dict[str, Any]:
        payload = dict(decision.audit or {})
        payload["rowCount"] = int(row_count or 0)
        payload["status"] = status or ("allowed" if decision.allowed else "denied")
        payload["writtenAt"] = datetime.utcnow().isoformat() + "Z"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload

    def audit_summary(self, limit: int = 20) -> Dict[str, Any]:
        if not self.audit_path.exists():
            return {"auditPath": str(self.audit_path), "items": []}
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()[-max(1, limit):]
        items = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return {"auditPath": str(self.audit_path), "items": items}

    def _load_policy(self) -> Dict[str, Any]:
        if not self.policy_path.exists():
            return {}
        try:
            payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _table_policy(self, policy: Dict[str, Any], table: str) -> Dict[str, Any]:
        tables = policy.get("tables") if isinstance(policy.get("tables"), dict) else {}
        payload = tables.get(table) if isinstance(tables, dict) else {}
        return payload if isinstance(payload, dict) else {}

    def _allow(self, contract: NodePlanContract, sql: str, run_id: str, checked_columns: List[str], masked_columns: Dict[str, str]) -> AccessDecision:
        audit = self._audit_payload(contract, sql, run_id, "allowed", "", checked_columns, [], masked_columns)
        return AccessDecision(
            allowed=True,
            table=contract.preferred_table,
            checked_columns=checked_columns,
            masked_columns=masked_columns,
            audit=audit,
        )

    def _deny(
        self,
        code: str,
        message: str,
        contract: NodePlanContract,
        sql: str,
        run_id: str,
        checked_columns: List[str],
        denied_columns: List[str] | None = None,
    ) -> AccessDecision:
        denied = sorted(denied_columns or [])
        audit = self._audit_payload(contract, sql, run_id, "denied", code, checked_columns, denied, contract.masked_columns or {})
        decision = AccessDecision(
            allowed=False,
            code=code,
            message=message,
            table=contract.preferred_table,
            checked_columns=checked_columns,
            denied_columns=denied,
            masked_columns=dict(contract.masked_columns or {}),
            audit=audit,
        )
        self.record_query_audit(decision, status="denied")
        return decision

    def _audit_payload(
        self,
        contract: NodePlanContract,
        sql: str,
        run_id: str,
        status: str,
        code: str,
        checked_columns: List[str],
        denied_columns: List[str],
        masked_columns: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "eventType": "query_access",
            "runId": run_id,
            "taskId": contract.task_id,
            "merchantId": contract.merchant_id,
            "effectiveUserId": contract.effective_user_id,
            "region": contract.authorized_region,
            "storeIds": list(contract.authorized_store_ids),
            "accessRole": contract.access_role,
            "table": contract.preferred_table,
            "checkedColumns": checked_columns,
            "deniedColumns": denied_columns,
            "maskedColumns": sorted((masked_columns or {}).keys()),
            "status": status,
            "code": code,
            "sqlHash": hashlib.sha256(str(sql or "").encode("utf-8")).hexdigest()[:16] if sql else "",
            "createdAt": datetime.utcnow().isoformat() + "Z",
        }

    def _columns_from_sql(self, sql: str, allowed_columns: List[str]) -> Set[str]:
        if not sql:
            return set()
        lowered = str(sql or "").lower()
        result = set()
        for column in allowed_columns or []:
            text = str(column or "").strip()
            if text and re.search(r"(?<![a-zA-Z0-9_])`?%s`?(?![a-zA-Z0-9_])" % re.escape(text.lower()), lowered):
                result.add(text)
        return result

    def _string_set(self, values: Any) -> Set[str]:
        if isinstance(values, str):
            return {item.strip() for item in values.split(",") if item.strip()}
        if isinstance(values, list):
            return {str(item).strip() for item in values if str(item).strip()}
        return set()
