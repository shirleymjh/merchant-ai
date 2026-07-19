from __future__ import annotations

import json

import pytest

from merchant_ai.services.authorization_policy import (
    AuthorizationPolicyError,
    load_authorization_policy,
    packaged_authorization_policy_path,
)


def test_packaged_authorization_policy_is_closed_and_referentially_valid() -> None:
    policy = load_authorization_policy()

    assert policy.policy_id
    assert policy.revision
    assert policy.default_identity_role in policy.identity_roles
    assert policy.default_access_role in policy.access_role_permissions
    assert all(
        item.access_role in policy.access_role_permissions
        and all(role in policy.access_role_permissions for role in item.principal_roles)
        for item in policy.identity_roles.values()
    )


def test_unknown_request_role_uses_policy_declared_default() -> None:
    policy = load_authorization_policy()

    assert policy.access_role_for_identity("undeclared-role") == policy.default_access_role


def test_invalid_policy_reference_fails_closed(tmp_path) -> None:
    payload = json.loads(packaged_authorization_policy_path().read_text(encoding="utf-8"))
    payload["identityRoles"][payload["defaults"]["identityRole"]]["accessRole"] = "missing-role"
    path = tmp_path / "invalid-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuthorizationPolicyError) as exc_info:
        load_authorization_policy(str(path))

    assert "undeclared access role" in str(exc_info.value)
