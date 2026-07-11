from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Set

from fastapi import HTTPException

from merchant_ai.config import Settings


class Permission(str, Enum):
    CHAT_RUN = "chat.run"
    RUN_READ = "run.read"
    RUN_CANCEL = "run.cancel"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    ASSET_WRITE = "asset.write"
    OPS_READ = "ops.read"
    OPS_WRITE = "ops.write"


ROLE_PERMISSIONS = {
    "merchant_analyst": {
        Permission.CHAT_RUN,
        Permission.RUN_READ,
        Permission.MEMORY_READ,
    },
    "merchant_admin": {
        Permission.CHAT_RUN,
        Permission.RUN_READ,
        Permission.RUN_CANCEL,
        Permission.MEMORY_READ,
        Permission.MEMORY_WRITE,
    },
    "ops_admin": set(Permission),
}


@dataclass(frozen=True)
class Principal:
    merchant_id: str = ""
    roles: Set[str] = field(default_factory=lambda: {"merchant_analyst"})
    is_ops: bool = False

    def has_permission(self, permission: Permission) -> bool:
        if self.is_ops:
            return True
        return any(permission in ROLE_PERMISSIONS.get(role, set()) for role in self.roles)


def merchant_principal(merchant_id: str, roles: Iterable[str] | None = None) -> Principal:
    normalized_roles = {str(role or "").strip() for role in (roles or ["merchant_analyst"]) if str(role or "").strip()}
    return Principal(merchant_id=str(merchant_id or "").strip(), roles=normalized_roles or {"merchant_analyst"})


def ops_principal() -> Principal:
    return Principal(roles={"ops_admin"}, is_ops=True)


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
