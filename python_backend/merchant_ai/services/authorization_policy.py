from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class AuthorizationPolicyError(RuntimeError):
    """Raised when the governed authorization policy cannot be trusted."""


@dataclass(frozen=True)
class IdentityRolePolicy:
    access_role: str
    principal_roles: tuple[str, ...]
    is_ops: bool


@dataclass(frozen=True)
class AuthorizationPolicy:
    policy_id: str
    revision: str
    default_identity_role: str
    default_access_role: str
    ops_principal_role: str
    identity_roles: Mapping[str, IdentityRolePolicy]
    access_role_permissions: Mapping[str, frozenset[str]]

    def normalize_identity_role(self, role: str) -> str:
        candidate = str(role or "").strip()
        if candidate in self.identity_roles:
            return candidate
        return self.default_identity_role

    def require_identity_role(self, role: str) -> str:
        candidate = str(role or "").strip()
        if candidate not in self.identity_roles:
            raise AuthorizationPolicyError("identity role is not declared by the active authorization policy")
        return candidate

    def identity_role_policy(self, role: str, *, strict: bool = False) -> IdentityRolePolicy:
        candidate = self.require_identity_role(role) if strict else self.normalize_identity_role(role)
        return self.identity_roles[candidate]

    def access_role_for_identity(self, role: str, *, strict: bool = False) -> str:
        return self.identity_role_policy(role, strict=strict).access_role

    def permissions_for_access_role(self, role: str) -> frozenset[str]:
        return self.access_role_permissions.get(str(role or "").strip(), frozenset())


def packaged_authorization_policy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "policies" / "authorization_policy.json"


@lru_cache(maxsize=8)
def load_authorization_policy(path: str = "") -> AuthorizationPolicy:
    source = Path(path).expanduser().resolve() if str(path or "").strip() else packaged_authorization_policy_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorizationPolicyError("authorization policy is unavailable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AuthorizationPolicyError("authorization policy root must be an object")
    if payload.get("schemaVersion") != 1:
        raise AuthorizationPolicyError("authorization policy schemaVersion is unsupported")

    defaults = _required_mapping(payload, "defaults")
    identity_payload = _required_mapping(payload, "identityRoles")
    access_payload = _required_mapping(payload, "accessRoles")
    identity_roles: dict[str, IdentityRolePolicy] = {}
    for name, raw in identity_payload.items():
        role_name = _required_text(name, "identity role name")
        role_payload = _mapping(raw, "identity role")
        access_role = _required_text(role_payload.get("accessRole"), "identity role accessRole")
        principal_roles = _text_tuple(role_payload.get("principalRoles"), "identity role principalRoles")
        if not principal_roles:
            raise AuthorizationPolicyError("identity role principalRoles cannot be empty")
        identity_roles[role_name] = IdentityRolePolicy(
            access_role=access_role,
            principal_roles=principal_roles,
            is_ops=bool(role_payload.get("isOps", False)),
        )

    access_role_permissions: dict[str, frozenset[str]] = {}
    for name, raw in access_payload.items():
        role_name = _required_text(name, "access role name")
        role_payload = _mapping(raw, "access role")
        access_role_permissions[role_name] = frozenset(
            _text_tuple(role_payload.get("permissions"), "access role permissions")
        )

    default_identity_role = _required_text(defaults.get("identityRole"), "defaults.identityRole")
    default_access_role = _required_text(defaults.get("accessRole"), "defaults.accessRole")
    ops_principal_role = _required_text(defaults.get("opsPrincipalRole"), "defaults.opsPrincipalRole")
    if default_identity_role not in identity_roles:
        raise AuthorizationPolicyError("default identity role is not declared")
    if default_access_role not in access_role_permissions:
        raise AuthorizationPolicyError("default access role is not declared")
    for role_policy in identity_roles.values():
        if role_policy.access_role not in access_role_permissions:
            raise AuthorizationPolicyError("identity role references an undeclared access role")
        if any(role not in access_role_permissions for role in role_policy.principal_roles):
            raise AuthorizationPolicyError("identity role references an undeclared principal role")
    if ops_principal_role not in access_role_permissions:
        raise AuthorizationPolicyError("ops principal role is not declared")

    return AuthorizationPolicy(
        policy_id=_required_text(payload.get("policyId"), "policyId"),
        revision=_required_text(payload.get("revision"), "revision"),
        default_identity_role=default_identity_role,
        default_access_role=default_access_role,
        ops_principal_role=ops_principal_role,
        identity_roles=identity_roles,
        access_role_permissions=access_role_permissions,
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping(payload.get(key), key)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not value:
        raise AuthorizationPolicyError(f"authorization policy {label} must be a non-empty object")
    return value


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise AuthorizationPolicyError(f"authorization policy {label} must be a non-empty string")
    return text


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AuthorizationPolicyError(f"authorization policy {label} must be an array")
    result = tuple(dict.fromkeys(str(item or "").strip() for item in value if str(item or "").strip()))
    if len(result) != len(value):
        raise AuthorizationPolicyError(f"authorization policy {label} contains empty or duplicate values")
    return result
