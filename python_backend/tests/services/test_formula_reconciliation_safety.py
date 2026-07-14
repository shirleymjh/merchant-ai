from merchant_ai.services.formulas import reconcile_metric_formula_for_schema


def test_formula_reconciliation_rejects_unrelated_or_branch_projection():
    reconciled = reconcile_metric_formula_for_schema(
        "SUM(CASE WHEN is_vip = 1 OR order_amt > 100 THEN 1 ELSE 0 END)",
        ["is_vip", "order_amt"],
        {"order_amt"},
        "priority_order_cnt",
        "orders",
    )

    assert reconciled.formula == ""
    assert reconciled.rewritten is False
    assert reconciled.missing_source_columns == ["is_vip"]


def test_formula_reconciliation_allows_same_field_code_name_projection():
    reconciled = reconcile_metric_formula_for_schema(
        "SUM(CASE WHEN status_code = 1 OR status_name = 'active' THEN 1 ELSE 0 END)",
        ["status_code", "status_name"],
        {"status_name"},
        "active_cnt",
        "entities",
    )

    assert reconciled.formula
    assert "status_code" not in reconciled.formula
    assert "status_name" in reconciled.formula
    assert reconciled.rewritten is True
