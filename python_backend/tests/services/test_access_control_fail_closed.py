from __future__ import annotations

import json

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import NodePlanContract
from merchant_ai.services.access_control import AccessControlService, AccessDecision


def contract() -> NodePlanContract:
    return NodePlanContract(
        task_id="acl-contract",
        merchant_id="merchant-a",
        preferred_table="fact_events",
        access_role="analyst",
        allowed_columns=["merchant_id", "event_id", "event_id_suffix"],
        required_columns=["merchant_id", "event_id"],
        visible_columns=["event_id"],
    )


def write_policy(root, payload: object) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "merchant_acl.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def explicit_allow_policy() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "defaultEffect": "DENY",
        "allowedMerchantIds": ["merchant-a"],
        "tables": {
            "fact_events": {
                "allowedRoles": ["analyst"],
                "allowedColumns": ["merchant_id", "event_id"],
            }
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        [],
        {},
        {"schemaVersion": 999, "defaultEffect": "DENY"},
        {"schemaVersion": 1, "defaultEffect": "ALLOW"},
        {
            "schemaVersion": 1,
            "defaultEffect": "DENY",
            "allowedMerchantIds": "merchant-a",
        },
    ],
)
def test_invalid_or_unsupported_policy_is_unavailable_and_denied(tmp_path, payload) -> None:
    write_policy(tmp_path, payload)
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(contract(), "SELECT event_id FROM fact_events")

    assert decision.allowed is False
    assert decision.code == "ACL_POLICY_UNAVAILABLE"
    assert decision.audit["status"] == "denied"
    assert service.audit_summary()["items"][-1]["code"] == "ACL_POLICY_UNAVAILABLE"


def test_missing_policy_is_unavailable_and_denied(tmp_path) -> None:
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(contract(), "SELECT event_id FROM fact_events")

    assert decision.allowed is False
    assert decision.code == "ACL_POLICY_UNAVAILABLE"
    assert "does not exist" in decision.message


def test_uninitialized_access_decision_is_denied_by_default() -> None:
    assert AccessDecision().allowed is False


def test_malformed_json_policy_is_unavailable_and_denied(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "merchant_acl.json").write_text("{", encoding="utf-8")
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(contract(), "SELECT event_id FROM fact_events")

    assert decision.allowed is False
    assert decision.code == "ACL_POLICY_UNAVAILABLE"
    assert "parsed" in decision.message


def test_valid_empty_deny_by_default_policy_does_not_allow_any_merchant(tmp_path) -> None:
    write_policy(tmp_path, {"schemaVersion": 1, "defaultEffect": "DENY"})
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(contract(), "SELECT event_id FROM fact_events")

    assert decision.allowed is False
    assert decision.code == "MERCHANT_SCOPE_DENIED"


def test_contract_without_merchant_scope_cannot_fall_back_to_process_default(
    tmp_path,
) -> None:
    write_policy(tmp_path, explicit_allow_policy())
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(
        contract().model_copy(update={"merchant_id": ""}),
        "SELECT event_id FROM fact_events",
    )

    assert decision.allowed is False
    assert decision.code == "MERCHANT_SCOPE_DENIED"


def test_explicit_merchant_table_role_and_column_allowlist_allows_query(tmp_path) -> None:
    write_policy(tmp_path, explicit_allow_policy())
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(
        contract(),
        "SELECT event_id FROM fact_events WHERE merchant_id = 'merchant-a'",
    )

    assert decision.allowed is True
    assert decision.checked_columns == ["event_id", "merchant_id"]


def test_column_identifiers_are_ast_matched_not_substring_matched(tmp_path) -> None:
    write_policy(tmp_path, explicit_allow_policy())
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(
        contract(),
        "SELECT event_id_suffix FROM fact_events WHERE merchant_id = 'merchant-a'",
    )

    assert decision.allowed is False
    assert decision.code == "COLUMN_DENIED"
    assert decision.checked_columns == ["event_id_suffix", "merchant_id"]
    assert decision.denied_columns == ["event_id_suffix"]


def test_checked_column_override_cannot_silently_drop_contract_external_column(
    tmp_path,
) -> None:
    write_policy(tmp_path, explicit_allow_policy())
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(
        contract(),
        "SELECT secret_value FROM fact_events",
        checked_columns_override=["secret_value"],
    )

    assert decision.allowed is False
    assert decision.code == "COLUMN_DENIED"
    assert decision.denied_columns == ["secret_value"]


def test_unparseable_sql_is_denied_when_acl_must_extract_columns(tmp_path) -> None:
    write_policy(tmp_path, explicit_allow_policy())
    service = AccessControlService(get_settings(), root=tmp_path)

    decision = service.authorize_contract(contract(), "SELECT (")

    assert decision.allowed is False
    assert decision.code == "ACL_SQL_PARSE_FAILED"
