from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set

from fastapi import HTTPException

from merchant_ai.config import Settings
from merchant_ai.models import UserIdentity
from merchant_ai.services.authorization_policy import load_authorization_policy


def identity_scope_payload(identity: Any = None, merchant_id: str = "") -> Dict[str, Any]:
    """Return the stable authorization scope that owns thread-scoped context."""

    if hasattr(identity, "model_dump"):
        payload = identity.model_dump(by_alias=True)
    elif isinstance(identity, dict):
        payload = dict(identity)
    else:
        payload = {}
    return {
        "merchantId": str(payload.get("merchantId") or payload.get("merchant_id") or merchant_id or "").strip(),
        "userId": str(payload.get("userId") or payload.get("user_id") or "").strip(),
        "role": str(payload.get("role") or load_authorization_policy().default_identity_role).strip(),
        "storeIds": sorted(
            {
                str(item).strip()
                for item in (payload.get("storeIds") or payload.get("store_ids") or [])
                if str(item).strip()
            }
        ),
        "permissions": sorted(
            {
                str(item).strip()
                for item in (payload.get("permissions") or [])
                if str(item).strip()
            }
        ),
    }


def identity_scope_hash(identity: Any = None, merchant_id: str = "") -> str:
    payload = identity_scope_payload(identity, merchant_id)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Permission(str, Enum):
    CHAT_RUN = "chat.run"
    RUN_READ = "run.read"
    RUN_CANCEL = "run.cancel"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    ASSET_WRITE = "asset.write"
    OPS_READ = "ops.read"
    OPS_WRITE = "ops.write"


_ACTIVE_AUTHORIZATION_POLICY = load_authorization_policy()
ROLE_PERMISSIONS = {
    role: (
        set(Permission)
        if "*" in permissions
        else {Permission(value) for value in permissions}
    )
    for role, permissions in _ACTIVE_AUTHORIZATION_POLICY.access_role_permissions.items()
}
IDENTITY_ROLES = set(_ACTIVE_AUTHORIZATION_POLICY.identity_roles)


def resolve_authenticated_identity(
    settings: Settings,
    authorization: str = "",
    requested_identity: Optional[UserIdentity] = None,
) -> Optional[UserIdentity]:
    """Resolve trusted identity from an HS256 JWT; request identity is dev-only input."""
    token = str(authorization or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        if not settings.identity_jwt_secret:
            raise HTTPException(status_code=503, detail="identity JWT secret is not configured")
        claims = decode_hs256_jwt(token, settings)
        return identity_from_claims(claims)
    if settings.identity_auth_required:
        raise HTTPException(status_code=401, detail="authenticated merchant identity is required")
    if requested_identity is None:
        return None
    # Local development remains convenient, but never accepts arbitrary role names.
    role = _ACTIVE_AUTHORIZATION_POLICY.normalize_identity_role(requested_identity.role)
    return requested_identity.model_copy(update={"role": role})


def decode_hs256_jwt(token: str, settings: Settings) -> Dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="invalid identity token")
    header = decode_jwt_segment(parts[0])
    claims = decode_jwt_segment(parts[1])
    if header.get("alg") != "HS256":
        raise HTTPException(status_code=401, detail="unsupported identity token algorithm")
    expected = hmac.new(
        settings.identity_jwt_secret.encode("utf-8"),
        (parts[0] + "." + parts[1]).encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        signature = base64.urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid identity token signature") from exc
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="invalid identity token signature")
    now = int(time.time())
    if claims.get("exp") is not None and int(claims["exp"]) <= now:
        raise HTTPException(status_code=401, detail="identity token expired")
    if claims.get("nbf") is not None and int(claims["nbf"]) > now:
        raise HTTPException(status_code=401, detail="identity token is not active")
    if settings.identity_jwt_issuer and str(claims.get("iss") or "") != settings.identity_jwt_issuer:
        raise HTTPException(status_code=401, detail="invalid identity token issuer")
    audience = claims.get("aud")
    audiences = set(audience if isinstance(audience, list) else [audience])
    if settings.identity_jwt_audience and settings.identity_jwt_audience not in audiences:
        raise HTTPException(status_code=401, detail="invalid identity token audience")
    return claims


def decode_jwt_segment(value: str) -> Dict[str, Any]:
    try:
        payload = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid identity token") from exc


def identity_from_claims(claims: Dict[str, Any]) -> UserIdentity:
    role = str(claims.get("role") or _ACTIVE_AUTHORIZATION_POLICY.default_identity_role)
    if role not in IDENTITY_ROLES:
        raise HTTPException(status_code=403, detail="identity role is not allowed")
    store_ids = claims.get("storeIds") or claims.get("store_ids") or []
    permissions = claims.get("permissions") or []
    return UserIdentity(
        user_id=str(claims.get("sub") or claims.get("userId") or ""),
        merchant_id=str(claims.get("merchantId") or claims.get("merchant_id") or ""),
        display_name=str(claims.get("name") or claims.get("displayName") or ""),
        role=role,
        region=str(claims.get("region") or claims.get("Region") or ""),
        language=str(claims.get("language") or "zh-CN"),
        store_ids=[str(item) for item in store_ids if str(item or "").strip()] if isinstance(store_ids, list) else [],
        permissions=[str(item) for item in permissions if str(item or "").strip()] if isinstance(permissions, list) else [],
    )


@dataclass(frozen=True)
class Principal:
    merchant_id: str = ""
    roles: Set[str] = field(default_factory=lambda: {_ACTIVE_AUTHORIZATION_POLICY.default_access_role})
    permissions: Set[Permission] = field(default_factory=set)
    is_ops: bool = False

    def has_permission(self, permission: Permission) -> bool:
        if self.is_ops:
            return True
        if permission in self.permissions:
            return True
        return any(permission in ROLE_PERMISSIONS.get(role, set()) for role in self.roles)


def merchant_principal(merchant_id: str, roles: Iterable[str] | None = None) -> Principal:
    normalized_roles = {
        str(role or "").strip()
        for role in (roles or [_ACTIVE_AUTHORIZATION_POLICY.default_access_role])
        if str(role or "").strip()
    }
    return Principal(
        merchant_id=str(merchant_id or "").strip(),
        roles=normalized_roles or {_ACTIVE_AUTHORIZATION_POLICY.default_access_role},
    )


def ops_principal() -> Principal:
    return Principal(roles={_ACTIVE_AUTHORIZATION_POLICY.ops_principal_role}, is_ops=True)


def principal_from_authenticated_identity(identity: UserIdentity) -> Principal:
    """Translate a verified API identity into the authorization principal model.

    ``merchantId`` from a request is never used here.  A normal merchant
    principal is scoped only by the merchant claim in the verified identity;
    platform operators remain the sole cross-merchant identity role.
    """

    role = str(identity.role or _ACTIVE_AUTHORIZATION_POLICY.default_identity_role).strip()
    role_policy = _ACTIVE_AUTHORIZATION_POLICY.identity_role_policy(role, strict=True)
    if role_policy.is_ops:
        return Principal(roles=set(role_policy.principal_roles), is_ops=True)
    merchant_id = str(identity.merchant_id or "").strip()
    if not merchant_id:
        raise HTTPException(status_code=403, detail="authenticated identity has no merchant scope")
    explicit_permissions: Set[Permission] = set()
    for value in identity.permissions:
        try:
            explicit_permissions.add(Permission(str(value)))
        except ValueError:
            continue
    return Principal(
        merchant_id=merchant_id,
        roles=set(role_policy.principal_roles),
        permissions=explicit_permissions,
    )


def authorize_authenticated_merchant_access(
    settings: Settings,
    identity: UserIdentity,
    merchant_id: str,
    permission: Permission,
) -> str:
    """Authorize one merchant target using only a previously verified identity."""

    return authorize_merchant_access(
        settings,
        principal_from_authenticated_identity(identity),
        merchant_id,
        permission,
    )


def authorize_merchant_access(settings: Settings, principal: Principal, merchant_id: str, permission: Permission) -> str:
    target = (merchant_id or settings.merchant_id or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="merchantId is required")
    if not settings.merchant_allowed(target):
        raise HTTPException(status_code=403, detail="merchantId is not allowed")
    if principal.merchant_id and principal.merchant_id != target and not principal.is_ops:
        raise HTTPException(status_code=403, detail="principal cannot access merchantId")
    if not principal.has_permission(permission):
        raise HTTPException(status_code=403, detail="permission denied: %s" % permission.value)
    return target
