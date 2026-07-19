from __future__ import annotations

from types import SimpleNamespace

import pytest

from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.quick_metrics import (
    compile_semantic_quick_metric,
    quick_metric_response,
    quick_metric_supports_temporal_mode,
    quick_metric_time_filter,
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


def test_structured_artifact_refs_disable_quick_metric_before_query() -> None:
    metric = semantic_metric(
        "runtime_attachment_metric",
        "附件指标",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )

    class UnexpectedRepository:
        def query(self, sql, params=None):
            raise AssertionError("artifact-bearing requests must use the governed analysis path")

    response = quick_metric_response(
        "最近2天附件指标趋势",
        "tenant-artifact",
        UnexpectedRepository(),
        extracted_metric_phrase("附件指标", analysis_intent="trend"),
        [metric],
        artifact_refs=[{"uri": "merchant://request/input.csv"}],
    )

    assert response is None


def test_missing_structured_keywords_disable_quick_metric_before_query() -> None:
    metric = semantic_metric(
        "runtime_structured_gate_metric",
        "结构门指标",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )

    class UnexpectedRepository:
        def query(self, sql, params=None):
            raise AssertionError("quick metric requires structured keyword evidence")

    response = quick_metric_response(
        "最近2天结构门指标趋势",
        "tenant-structured-gate",
        UnexpectedRepository(),
        semantic_metrics=[metric],
    )

    assert response is None


def test_compiler_preserves_published_temporal_governance_metadata() -> None:
    contract = compile_semantic_quick_metric(
        {
            "metricKey": "runtime_snapshot_ratio",
            "displayName": "快照脉冲率",
            "formula": "MAX(snapshot_rate)",
            "sourceColumns": ["snapshot_rate"],
            "timeColumn": "metric_event_day",
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
    assert contract["time_column"] == "metric_event_day"
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
        ("latest_value_only", "day", "exact_day", True),
        ("latest_value_only", "day", "period_summary", True),
        ("latest_value_only", "day", "daily_series", False),
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


def test_latest_value_only_uses_one_latest_as_of_partition_and_never_sums_the_window() -> None:
    snapshot = semantic_metric(
        "snapshot_balance",
        "快照余额",
        "SUM(event_count)",
        "SUM(`event_count`)",
        policy="latest_value_only",
        grain="day",
    )
    snapshot["unit"] = "元"
    calls: list[tuple[str, list[str]]] = []

    class Repository:
        def query(self, sql, params=None):
            calls.append((sql, list(params or [])))
            return [{"time_dimension": "2026-07-11", "value": 110.0}]

    response = quick_metric_response(
        "最近30天快照余额是多少？",
        "tenant-latest-as-of",
        Repository(),
        extracted_metric_phrase("快照余额"),
        [snapshot],
    )

    assert response is not None
    assert len(calls) == 1
    sql, params = calls[0]
    assert "`event_day` = (SELECT MAX(`event_day`)" in sql
    assert "BETWEEN" not in sql
    assert params == ["tenant-latest-as-of", "tenant-latest-as-of"]
    assert "截至 2026-07-11" in response.answer
    assert "未做跨日加总" in response.answer
    assert "合计" not in response.answer
    assert response.debug_trace["summarySemantics"] == "latest_as_of_value"
    assert response.debug_trace["timeWindowContract"]["executionStartValue"] == "2026-07-11"
    assert response.debug_trace["timeWindowContract"]["executionEndValue"] == "2026-07-11"


def test_latest_value_only_null_is_missing_evidence_not_business_zero() -> None:
    snapshot = semantic_metric(
        "null_snapshot_balance",
        "空快照余额",
        "SUM(event_count)",
        "SUM(`event_count`)",
        policy="latest_value_only",
        grain="day",
    )

    class Repository:
        def query(self, sql, params=None):
            return [{"time_dimension": "2026-07-11", "value": None}]

    response = quick_metric_response(
        "最近30天空快照余额是多少？",
        "tenant-null-latest",
        Repository(),
        extracted_metric_phrase("空快照余额"),
        [snapshot],
    )

    assert response is None


def test_daily_value_series_null_returns_to_query_graph_instead_of_inventing_zero() -> None:
    daily = semantic_metric(
        "null_daily_ratio",
        "空序列率",
        "MAX(snapshot_rate)",
        "MAX(`snapshot_rate`)",
        policy="daily_value_only",
        grain="day",
    )

    class Repository:
        def query(self, sql, params=None):
            return [
                {"time_dimension": "2026-07-14", "value": 0.2},
                {"time_dimension": "2026-07-15", "value": None},
            ]

    response = quick_metric_response(
        "最近2天空序列率趋势",
        "tenant-null-series",
        Repository(),
        extracted_metric_phrase("空序列率", analysis_intent="trend"),
        [daily],
    )

    assert response is None


def test_latest_value_only_exact_date_uses_the_resolved_as_of_value() -> None:
    snapshot = semantic_metric(
        "dated_snapshot_balance",
        "日期快照余额",
        "SUM(event_count)",
        "SUM(`event_count`)",
        policy="latest_value_only",
        grain="day",
    )
    calls: list[tuple[str, list[str]]] = []

    class Repository:
        def query(self, sql, params=None):
            calls.append((sql, list(params or [])))
            return [{"time_dimension": "2026-07-10", "value": 80.0}]

    response = quick_metric_response(
        "2026-07-10日期快照余额是多少？",
        "tenant-exact-as-of",
        Repository(),
        extracted_metric_phrase("日期快照余额"),
        [snapshot],
    )

    assert response is not None
    assert len(calls) == 1
    sql, params = calls[0]
    assert "`event_day` = (SELECT MAX(`event_day`)" in sql
    assert "`event_day` <= %s" in sql
    assert "BETWEEN" not in sql
    assert params == ["tenant-exact-as-of", "tenant-exact-as-of", "2026-07-10"]


@pytest.mark.parametrize(
    ("resolved_field", "resolved_anchor"),
    [
        ("execution_end_value", "20260712"),
        ("execution_end_date", "2026-07-12"),
    ],
)
def test_latest_value_only_prefers_the_resolved_execution_anchor(
    resolved_field: str,
    resolved_anchor: str,
) -> None:
    time_range = ResolvedTimeRange(
        kind="rolling",
        anchor_policy="latest_partition",
        **{resolved_field: resolved_anchor},
    )

    predicate, params = quick_metric_time_filter(
        time_range,
        "arbitrary_snapshot_table",
        "arbitrary_partition",
        "arbitrary_tenant",
        "tenant-resolved-anchor",
        aggregation_policy="latest_value_only",
    )

    assert predicate == (
        "`arbitrary_partition` = (SELECT MAX(`arbitrary_partition`) "
        "FROM `arbitrary_snapshot_table` WHERE `arbitrary_tenant` = %s "
        "AND `arbitrary_partition` <= %s)"
    )
    assert params == ["tenant-resolved-anchor", resolved_anchor]


def test_latest_value_calendar_as_of_uses_the_same_max_not_after_operator_as_query_graph() -> None:
    time_range = ResolvedTimeRange(
        kind="exact_date",
        anchor_policy="calendar",
        end_date="2026-07-12",
        execution_end_value="20260712",
    )

    predicate, params = quick_metric_time_filter(
        time_range,
        "arbitrary_snapshot_table",
        "arbitrary_partition",
        "arbitrary_tenant",
        "tenant-calendar-anchor",
        aggregation_policy="latest_value_only",
        as_of_policy="calendar",
    )

    assert predicate == (
        "`arbitrary_partition` = (SELECT MAX(`arbitrary_partition`) "
        "FROM `arbitrary_snapshot_table` WHERE `arbitrary_tenant` = %s "
        "AND `arbitrary_partition` <= %s)"
    )
    assert params == ["tenant-calendar-anchor", "20260712"]


def test_latest_value_only_daily_trend_returns_to_query_graph() -> None:
    snapshot = semantic_metric(
        "trend_snapshot_balance",
        "趋势快照余额",
        "SUM(event_count)",
        "SUM(`event_count`)",
        policy="latest_value_only",
        grain="day",
    )

    class UnexpectedRepository:
        def query(self, *_args, **_kwargs):
            raise AssertionError("snapshot trend requires the QueryGraph time-series executor")

    response = quick_metric_response(
        "最近7天趋势快照余额走势",
        "tenant-snapshot-trend",
        UnexpectedRepository(),
        extracted_metric_phrase("趋势快照余额", analysis_intent="trend"),
        [snapshot],
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
