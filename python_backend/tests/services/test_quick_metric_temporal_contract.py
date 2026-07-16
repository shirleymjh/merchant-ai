from __future__ import annotations

from types import SimpleNamespace

import pytest

from merchant_ai.services.quick_metrics import (
    compile_semantic_quick_metric,
    quick_metric_response,
    quick_metric_supports_temporal_mode,
    resolve_quick_metric_temporal_variant,
)


def semantic_metric(
    key: str,
    label: str,
    formula: str,
    compiled_formula: str,
    *,
    policy: str = "",
    grain: str = "",
    variants: dict | None = None,
    linked_variant_of: str = "",
) -> dict:
    return {
        "key": key,
        "label": label,
        "unit": "%",
        "formula": formula,
        "compiled_formula": compiled_formula,
        "source_columns": ["snapshot_rate"] if "snapshot_rate" in formula else ["event_count", "base_count"],
        "terms": [label, key],
        "table": "runtime_measurements",
        "time_column": "event_day",
        "tenant_column": "tenant_key",
        "topic": "runtime_domain",
        "aggregation_policy": policy,
        "applicable_time_grain": grain,
        "temporal_variants": variants or {},
        "linked_variant_of": linked_variant_of,
    }


def extracted_metric_phrase(phrase: str, analysis_intent: str = "lookup") -> SimpleNamespace:
    return SimpleNamespace(
        metric_keywords=[phrase],
        ranking_keywords=[],
        unresolved_phrases=[],
        mentions=[],
        analysis_intent=analysis_intent,
    )


def test_compiler_preserves_published_temporal_governance_metadata() -> None:
    contract = compile_semantic_quick_metric(
        {
            "metricKey": "runtime_snapshot_ratio",
            "displayName": "快照脉冲率",
            "formula": "MAX(snapshot_rate)",
            "sourceColumns": ["snapshot_rate"],
            "aggregationPolicy": "daily_value_only",
            "applicableTimeGrain": "day",
            "temporalVariants": {"publishedPeriodContract": "runtime_period_ratio"},
            "linkedVariantOf": "semantic:runtime_domain:runtime_measurements:metric:runtime_family",
        },
        "runtime_domain",
        "runtime_measurements",
        "event_day",
        "tenant_key",
    )

    assert contract is not None
    assert contract["aggregation_policy"] == "daily_value_only"
    assert contract["applicable_time_grain"] == "day"
    assert contract["temporal_variants"] == {"publishedPeriodContract": "runtime_period_ratio"}
    assert contract["linked_variant_of"].endswith(":runtime_family")


@pytest.mark.parametrize(
    ("policy", "grain", "mode", "supported"),
    [
        ("daily_value_only", "day", "exact_day", True),
        ("daily_value_only", "day", "daily_series", True),
        ("daily_value_only", "day", "period_summary", False),
        ("daily_value_only", "", "exact_day", False),
        ("ratio_of_sums", "period", "period_summary", True),
        ("ratio_of_sums", "period", "daily_series", False),
        ("ratio_of_sums", "", "daily_series", True),
        ("ratio_of_sums", "", "period_summary", True),
    ],
)
def test_temporal_support_is_a_function_of_published_policy_and_grain(
    policy: str,
    grain: str,
    mode: str,
    supported: bool,
) -> None:
    metric = semantic_metric(
        "arbitrary_measure",
        "任意指标",
        "MAX(snapshot_rate)" if policy == "daily_value_only" else "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "MAX(`snapshot_rate`)" if policy == "daily_value_only" else "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy=policy,
        grain=grain,
    )

    assert quick_metric_supports_temporal_mode(metric, mode) is supported


def test_period_summary_resolves_published_variant_and_never_queries_interval_daily_max() -> None:
    daily = semantic_metric(
        "runtime_snapshot_ratio",
        "快照脉冲率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
        variants={"publishedPeriodContract": "runtime_period_ratio"},
    )
    period = semantic_metric(
        "runtime_period_ratio",
        "周期脉冲率",
        "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy="ratio_of_sums",
    )
    calls: list[str] = []

    class Repository:
        def query(self, sql, params=None):
            calls.append(sql)
            if "GROUP BY" in sql:
                return [
                    {"time_dimension": "2026-07-14", "value": 0.2},
                    {"time_dimension": "2026-07-15", "value": 0.3},
                ]
            return [{"value": 0.25}]

    response = quick_metric_response(
        "最近2天快照脉冲率是多少？",
        "tenant-period",
        Repository(),
        extracted_metric_phrase("快照脉冲率"),
        [daily, period],
    )

    assert response is not None
    assert len(calls) == 2
    assert all("MAX(`snapshot_rate`)" not in sql for sql in calls)
    assert all("SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)" in sql for sql in calls)
    assert response.debug_trace["temporalMode"] == "period_summary"
    assert response.debug_trace["temporalMetricRedirected"] is True
    assert response.debug_trace["semanticMetric"]["metricKey"] == "runtime_period_ratio"


def test_daily_series_uses_grouped_daily_value_without_interval_scalar_query() -> None:
    daily = semantic_metric(
        "runtime_snapshot_ratio_series",
        "快照序列率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )
    calls: list[str] = []

    class Repository:
        def query(self, sql, params=None):
            calls.append(sql)
            return [
                {"time_dimension": "2026-07-14", "value": 0.2},
                {"time_dimension": "2026-07-15", "value": 0.3},
            ]

    response = quick_metric_response(
        "最近2天快照序列率趋势",
        "tenant-series",
        Repository(),
        extracted_metric_phrase("快照序列率", analysis_intent="trend"),
        [daily],
    )

    assert response is not None
    assert len(calls) == 1
    assert "GROUP BY `event_day`" in calls[0]
    assert "MAX(`snapshot_rate`)" in calls[0]
    assert "最新日值为 30.00%" in response.answer
    assert response.data_sections[1].data_rows == [
        {
            "metric_name": "快照序列率",
            "value": 0.3,
            "time_dimension": "2026-07-15",
        }
    ]


def test_period_daily_value_without_published_variant_returns_to_planner_before_query() -> None:
    daily = semantic_metric(
        "isolated_snapshot_ratio",
        "孤立快照率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("unsupported daily-value period summary must return to Planner")

    response = quick_metric_response(
        "最近2天孤立快照率是多少？",
        "tenant-unsupported",
        UnexpectedRepository(),
        extracted_metric_phrase("孤立快照率"),
        [daily],
    )

    assert response is None


def test_comparison_window_returns_to_planner_before_query() -> None:
    period = semantic_metric(
        "comparison_period_ratio",
        "对照脉冲率",
        "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy="ratio_of_sums",
    )

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("a one-window quick executor cannot answer a period comparison")

    response = quick_metric_response(
        "最近2天对照脉冲率与前2天相比是多少？",
        "tenant-comparison",
        UnexpectedRepository(),
        extracted_metric_phrase("对照脉冲率", analysis_intent="comparison"),
        [period],
    )

    assert response is None


def test_linked_variant_of_can_publish_the_period_side_of_a_metric_family() -> None:
    daily = semantic_metric(
        "linked_snapshot_ratio",
        "关联快照率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )
    period = semantic_metric(
        "linked_period_ratio",
        "关联周期率",
        "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy="ratio_of_sums",
        linked_variant_of="semantic:runtime_domain:runtime_measurements:metric:linked_snapshot_ratio",
    )

    resolved = resolve_quick_metric_temporal_variant(daily, [daily, period], "period_summary")

    assert resolved is period


def test_ambiguous_period_variants_are_not_arbitrarily_selected() -> None:
    daily = semantic_metric(
        "ambiguous_snapshot_ratio",
        "歧义快照率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
        variants={"candidateA": "period_ratio_a", "candidateB": "period_ratio_b"},
    )
    period_a = semantic_metric(
        "period_ratio_a",
        "周期口径甲",
        "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy="ratio_of_sums",
    )
    period_b = semantic_metric(
        "period_ratio_b",
        "周期口径乙",
        "SUM(event_count) / NULLIF(SUM(base_count), 0)",
        "SUM(`event_count`) / NULLIF(SUM(`base_count`), 0)",
        policy="ratio_of_sums",
    )

    assert resolve_quick_metric_temporal_variant(daily, [daily, period_a, period_b], "period_summary") is None
