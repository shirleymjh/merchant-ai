from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

from merchant_ai.config import Settings
from merchant_ai.models import NodePlanContract


@dataclass
class AccessDecision:
    allowed: bool = False
    code: str = ""
    message: str = ""
    table: str = ""
    checked_columns: List[str] = field(default_factory=list)
    denied_columns: List[str] = field(default_factory=list)
    masked_columns: Dict[str, str] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccessPolicyLoadResult:
    available: bool
    policy: Dict[str, Any] = field(default_factory=dict)
    code: str = ""
    message: str = ""
    schema_version: int = 0


class AccessControlService:
    """Table/column ACL and query audit for merchant BI execution."""

    POLICY_SCHEMA_VERSION = 1
    DENY_BY_DEFAULT = "DENY"

    def __init__(self, settings: Settings, root: Path | None = None):
        self.settings = settings
        self.root = root or (settings.resolved_workspace_path / "ops" / "access_control")
        self.policy_path = self.root / "merchant_acl.json"
        self.audit_path = self.root / "query_audit.jsonl"

    def authorize_contract(
        self,
        contract: NodePlanContract,
        sql: str = "",
        run_id: str = "",
        checked_columns_override: Optional[Iterable[str]] = None,
    ) -> AccessDecision:
        loaded_policy = self._load_policy()
        if not loaded_policy.available:
            return self._deny(
                loaded_policy.code or "ACL_POLICY_UNAVAILABLE",
                loaded_policy.message or "ACL policy is unavailable",
                contract,
                sql,
                run_id,
                [],
            )
        policy = loaded_policy.policy
        table = contract.preferred_table
        role = str(contract.access_role or "").strip()
        merchant_id = str(contract.merchant_id or "").strip()
        table_policy = self._table_policy(policy, table)
        allowed_merchants = self._string_set(policy.get("allowedMerchantIds"))
        denied_tables = self._string_set(policy.get("deniedTables"))
        allowed_tables = self._string_set(policy.get("allowedTables"))
        policy_tables = policy.get("tables") if isinstance(policy.get("tables"), dict) else {}
        if merchant_id not in allowed_merchants:
            return self._deny("MERCHANT_SCOPE_DENIED", "merchant is not allowed by ACL", contract, sql, run_id, [])
        if table in denied_tables:
            return self._deny("TABLE_DENIED", "table is explicitly denied by ACL", contract, sql, run_id, [])
        if table not in allowed_tables and table not in policy_tables:
            return self._deny("TABLE_NOT_ALLOWED", "table is not in ACL allowlist", contract, sql, run_id, [])
        table_roles = self._string_set(table_policy.get("allowedRoles"))
        if role not in table_roles:
            return self._deny("TABLE_ROLE_DENIED", "role cannot access table", contract, sql, run_id, [])
        if checked_columns_override is not None:
            allowed = set(contract.allowed_columns or [])
            override_columns = {
                str(column or "").strip()
                for column in checked_columns_override
                if str(column or "").strip()
            }
            out_of_contract_columns = sorted(override_columns - allowed)
            if out_of_contract_columns:
                return self._deny(
                    "COLUMN_DENIED",
                    "one or more checked columns are outside the active contract",
                    contract,
                    sql,
                    run_id,
                    sorted(override_columns),
                    out_of_contract_columns,
                )
            checked_columns = sorted(override_columns)
        else:
            try:
                sql_columns = self._columns_from_sql(sql, contract.allowed_columns)
            except ValueError:
                return self._deny(
                    "ACL_SQL_PARSE_FAILED",
                    "SQL identifiers cannot be verified by ACL",
                    contract,
                    sql,
                    run_id,
                    [],
                )
            checked_columns = sorted(
                sql_columns
                or set(contract.required_columns or [])
                or set(contract.visible_columns or [])
            )
        column_policies = table_policy.get("columns") if isinstance(table_policy.get("columns"), dict) else {}
        allowed_column_values = table_policy.get("allowedColumns")
        has_column_allowlist = isinstance(allowed_column_values, list)
        allowed_columns = self._string_set(allowed_column_values)
        denied_columns: List[str] = []
        masked_columns = dict(contract.masked_columns or {})
        for column in checked_columns:
            if has_column_allowlist and column not in allowed_columns:
                denied_columns.append(column)
                continue
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

    def _load_policy(self) -> AccessPolicyLoadResult:
        if not self.policy_path.exists():
            return self._policy_unavailable("ACL policy file does not exist")
        try:
            payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self._policy_unavailable("ACL policy file cannot be read or parsed")
        if not isinstance(payload, dict):
            return self._policy_unavailable("ACL policy root must be an object")
        version = payload.get("schemaVersion")
        if isinstance(version, bool) or not isinstance(version, int):
            return self._policy_unavailable("ACL policy schemaVersion is required")
        if version != self.POLICY_SCHEMA_VERSION:
            return self._policy_unavailable("ACL policy schemaVersion is unsupported")
        if payload.get("defaultEffect") != self.DENY_BY_DEFAULT:
            return self._policy_unavailable("ACL policy defaultEffect must be DENY")
        validation_error = self._policy_validation_error(payload)
        if validation_error:
            return self._policy_unavailable(validation_error)
        return AccessPolicyLoadResult(
            available=True,
            policy=payload,
            schema_version=version,
        )

    @staticmethod
    def _policy_unavailable(message: str) -> AccessPolicyLoadResult:
        return AccessPolicyLoadResult(
            available=False,
            code="ACL_POLICY_UNAVAILABLE",
            message=message,
        )

    def _policy_validation_error(self, policy: Dict[str, Any]) -> str:
        for field_name in ("allowedMerchantIds", "allowedTables", "deniedTables"):
            if not self._is_string_list(policy.get(field_name, [])):
                return "ACL policy %s must be a string list" % field_name
        tables = policy.get("tables", {})
        if not isinstance(tables, dict):
            return "ACL policy tables must be an object"
        for table_name, table_policy in tables.items():
            if not isinstance(table_name, str) or not table_name.strip():
                return "ACL policy table keys must be non-empty strings"
            if not isinstance(table_policy, dict):
                return "ACL table policy must be an object"
            if not self._is_string_list(table_policy.get("allowedRoles", [])):
                return "ACL table allowedRoles must be a string list"
            if "allowedColumns" in table_policy and not self._is_string_list(table_policy.get("allowedColumns")):
                return "ACL table allowedColumns must be a string list"
            columns = table_policy.get("columns", {})
            if not isinstance(columns, dict):
                return "ACL table columns must be an object"
            for column_name, column_policy in columns.items():
                if not isinstance(column_name, str) or not column_name.strip():
                    return "ACL column keys must be non-empty strings"
                if not isinstance(column_policy, dict):
                    return "ACL column policy must be an object"
                if "allowedRoles" in column_policy and not self._is_string_list(column_policy.get("allowedRoles")):
                    return "ACL column allowedRoles must be a string list"
        return ""

    @staticmethod
    def _is_string_list(values: Any) -> bool:
        return isinstance(values, list) and all(
            isinstance(item, str) and bool(item.strip()) for item in values
        )

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
        try:
            self.record_query_audit(decision, status="denied")
        except (OSError, UnicodeError) as exc:
            # Audit storage failure must never turn an authorization rejection
            # into an exception or, worse, an allow decision.
            decision.audit["auditWriteStatus"] = "failed"
            decision.audit["auditWriteErrorType"] = type(exc).__name__
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
        sql_text = str(sql)
        try:
            expression = sqlglot.parse_one(sql_text, read="doris")
        except (sqlglot.errors.ParseError, ValueError, TypeError):
            normalized_sql = self._normalize_dbapi_positional_parameters(sql_text)
            if normalized_sql == sql_text:
                raise ValueError("SQL identifier parsing failed")
            try:
                expression = sqlglot.parse_one(normalized_sql, read="doris")
            except (sqlglot.errors.ParseError, ValueError, TypeError) as exc:
                raise ValueError("SQL identifier parsing failed") from exc
        allowed_by_normalized = {
            str(column).strip().casefold(): str(column).strip()
            for column in allowed_columns or []
            if str(column).strip()
        }
        return {
            allowed_by_normalized[normalized]
            for column in expression.find_all(exp.Column)
            if (normalized := str(column.name or "").strip().casefold()) in allowed_by_normalized
        }

    @staticmethod
    def _normalize_dbapi_positional_parameters(sql: str) -> str:
        try:
            tokens = sqlglot.Tokenizer(dialect="doris").tokenize(sql)
        except sqlglot.errors.TokenError:
            return sql
        replacements: list[tuple[int, int]] = []
        for index, token in enumerate(tokens[:-1]):
            next_token = tokens[index + 1]
            if (
                token.token_type == TokenType.MOD
                and token.text == "%"
                and next_token.token_type == TokenType.VAR
                and next_token.text == "s"
                and next_token.start == token.end + 1
            ):
                replacements.append((token.start, next_token.end + 1))
        if not replacements:
            return sql
        parts: list[str] = []
        cursor = 0
        for start, end in replacements:
            parts.append(sql[cursor:start])
            parts.append("?")
            cursor = end
        parts.append(sql[cursor:])
        return "".join(parts)

    def _string_set(self, values: Any) -> Set[str]:
        if isinstance(values, list):
            return {str(item).strip() for item in values if str(item).strip()}
        return set()
