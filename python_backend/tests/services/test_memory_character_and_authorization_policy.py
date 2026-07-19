from merchant_ai.services.authorization_policy import load_authorization_policy
from merchant_ai.services.memory import (
    contains_sensitive_identifier,
    default_memory_allowed_roles,
    extract_metric_like_terms,
    extract_time_windows,
    memory_visible_to_role,
)


def test_sensitive_identifier_scanner_covers_governed_privacy_shapes() -> None:
    assert contains_sensitive_identifier("card 4111111111111111")
    assert contains_sensitive_identifier("identifier 123-45-6789")
    assert contains_sensitive_identifier("contact analyst@example.com")
    assert not contains_sensitive_identifier("metric 12345 and analyst-at-example")


def test_memory_roles_come_from_authorization_permissions() -> None:
    policy = load_authorization_policy()
    expected = sorted(
        role
        for role, permissions in policy.access_role_permissions.items()
        if "memory.read" in permissions or "*" in permissions
    )

    assert default_memory_allowed_roles("procedure") == expected
    assert all(
        memory_visible_to_role(
            {"visibility": "planner_only", "allowedRoles": expected},
            role,
        )
        for role in expected
    )
    assert not memory_visible_to_role(
        {"visibility": "merchant", "allowedRoles": expected},
        "undeclared_role",
    )


def test_memory_recall_does_not_guess_metrics_from_ascii_words() -> None:
    assert extract_metric_like_terms("show revenue and margin") == set()
    assert extract_metric_like_terms("semantic:trade:summary:metric:net_revenue") == {
        "net_revenue"
    }


def test_memory_time_windows_reuse_typed_temporal_spans() -> None:
    assert extract_time_windows("最近7天") == [7]
    assert extract_time_windows("近2周") == [14]
