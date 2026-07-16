import json
from datetime import date, timedelta

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.answer import (
    AnswerComposeService,
    aligned_trend_sync_analysis,
    answer_prompt_sections,
    lightweight_answer_contract_verification,
    metric_series_groups_for_intent,
    multi_trend_metric_sentence,
    query_bundle_rows_for_trend,
)


def trend_plan_and_run(tmp_path, series, preview_limit=20, persist_artifacts=True):
    intents = []
    tasks = []
    for metric_key, payload in series.items():
        label = payload["label"]
        rows = [
            {"pt": point_date, metric_key: value}
            for point_date, value in payload["points"]
        ]
        task_id = "trend_%s" % metric_key
        intent = QuestionIntent(
            question="趋势是否同步上升？",
            intent_type="VALID",
            answer_mode=AnswerMode.GROUP_AGG,
            plan_task_id=task_id,
            preferred_table="trend_table",
            metric_name=metric_key,
            metric_column=metric_key,
            group_by_column="pt",
            metric_resolution={
                "metricKey": metric_key,
                "displayName": label,
                "aggregationPolicy": payload.get("aggregationPolicy", ""),
            },
        )
        artifacts = []
        if persist_artifacts:
            path = tmp_path / ("%s_rows.json" % task_id)
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            artifacts = [str(path)]
        bundle = QueryBundle(
            rows=rows[:preview_limit],
            original_row_count=len(rows),
            offloaded_files=artifacts,
        )
        intents.append(intent)
        tasks.append(AgentTaskResult(task_id=task_id, success=True, query_bundle=bundle))
    plan = QueryPlan(intents=intents)
    run = AgentRunResult(
        task_results=tasks,
        merged_query_bundle=QueryBundle(rows=[]),
        verified_evidence=VerifiedEvidence(passed=True),
    )
    return plan, run


def daily_points(values, start=date(2026, 6, 1)):
    return [((start + timedelta(days=index)).isoformat(), value) for index, value in enumerate(values)]


def test_trend_answer_reads_full_rows_artifact_without_expanding_prompt_preview(tmp_path):
    values = list(range(30))
    values[-1] = 100
    plan, run = trend_plan_and_run(
        tmp_path,
        {"refund_rate": {"label": "退款率", "points": daily_points(values)}},
    )

    answer = multi_trend_metric_sentence("最近30天退款率走势怎么样？哪一天波动最大？", plan, run)
    sections = AnswerComposeService(object()).build_sections(plan, run)
    prompt_sections = answer_prompt_sections(plan, run)

    assert "变化到 2026-06-30" in answer
    assert "最大单日变化点是 退款率 在 2026-06-30" in answer
    assert "变化到 2026-06-20" not in answer
    assert len(run.task_results[0].query_bundle.rows) == 20
    assert len(prompt_sections[0]["rows"]) == 10
    assert len(sections[0].data_rows) == 30

    summary_intent = QuestionIntent(
        question="最近30天汇总量和退款率走势怎么样？",
        intent_type="VALID",
        answer_mode=AnswerMode.METRIC,
        plan_task_id="summary_total",
        metric_name="summary_total",
        metric_column="summary_total",
        metric_resolution={"metricKey": "summary_total", "displayName": "汇总量"},
    )
    summary_bundle = QueryBundle(rows=[{"summary_total": 1000}], original_row_count=1)
    plan.intents.insert(0, summary_intent)
    run.task_results.insert(0, AgentTaskResult(task_id="summary_total", success=True, query_bundle=summary_bundle))
    verification = lightweight_answer_contract_verification(
        "最近30天汇总量和退款率走势怎么样？",
        plan,
        run,
        "最近30天，汇总量为 1000，退款率最新值为 100%。",
    )

    assert verification is not None
    assert verification.passed is True


def test_trend_answer_does_not_treat_truncated_preview_as_complete_series(tmp_path):
    plan, run = trend_plan_and_run(
        tmp_path,
        {"refund_rate": {"label": "退款率", "points": daily_points(list(range(30)))}},
        persist_artifacts=False,
    )

    answer = multi_trend_metric_sentence("最近30天退款率走势怎么样？", plan, run)

    assert "按日数据未加载完整" in answer
    assert "变化到 2026-06-20" not in answer
    assert "最大单日变化点" not in answer


def test_period_rollup_series_reports_governed_period_total(tmp_path):
    plan, run = trend_plan_and_run(
        tmp_path,
        {
            "metric_flow": {
                "label": "流量指标",
                "aggregationPolicy": "period_rollup",
                "points": daily_points([1000, 2000, 0, 500, 1500]),
            }
        },
    )

    answer = multi_trend_metric_sentence("最近5天流量指标表现如何？", plan, run)

    assert "流量指标周期合计为 5000" in answer
    assert "从 2026-06-01 的 1000 变化到 2026-06-05 的 1500" in answer


def test_snapshot_and_daily_value_policies_never_sum_the_series(tmp_path):
    plan, run = trend_plan_and_run(
        tmp_path,
        {
            "metric_snapshot": {
                "label": "快照指标",
                "aggregationPolicy": "latest_value_only",
                "points": daily_points([8700, 8800]),
            },
            "metric_daily": {
                "label": "日值指标",
                "aggregationPolicy": "daily_value_only",
                "points": daily_points([0.1, 0.2]),
            },
        },
    )

    answer = multi_trend_metric_sentence("最近2天快照指标和日值指标表现如何？", plan, run)

    assert "快照指标截至 2026-06-02 为 8800" in answer
    assert "17500" not in answer
    assert "日值指标周期合计" not in answer


def test_sync_up_uses_aligned_daily_direction_coverage_not_only_first_and_last(tmp_path):
    plan, run = trend_plan_and_run(
        tmp_path,
        {
            "metric_a": {"label": "指标A", "points": daily_points([1, 2, 3, 4, 5])},
            "metric_b": {"label": "指标B", "points": daily_points([1, 2, 1, 2, 3])},
            "metric_c": {"label": "指标C", "points": daily_points([1, 2, 3, 2, 3])},
        },
    )

    answer = multi_trend_metric_sentence("最近5天指标A、指标B和指标C是否同步上升？", plan, run)

    assert "没有同步上升" in answer
    assert "同步覆盖率 50%（2/4 个共同可比日变化）" in answer
    assert "日期对齐覆盖率 100%（4/4）" in answer


def test_sync_alignment_skips_missing_dates_instead_of_filling_zero(tmp_path):
    start = date(2026, 6, 1)
    all_dates = [(start + timedelta(days=index)).isoformat() for index in range(4)]
    plan, run = trend_plan_and_run(
        tmp_path,
        {
            "metric_a": {"label": "指标A", "points": list(zip(all_dates, [1, 2, 3, 4]))},
            "metric_b": {"label": "指标B", "points": [(all_dates[0], 1), (all_dates[2], 3), (all_dates[3], 4)]},
            "metric_c": {"label": "指标C", "points": list(zip(all_dates, [1, 2, 3, 4]))},
        },
    )
    groups = []
    for intent, task in zip(plan.intents, run.task_results):
        rows, complete = query_bundle_rows_for_trend(task.query_bundle)
        groups.extend(
            {**group, "seriesComplete": complete}
            for group in metric_series_groups_for_intent(plan, intent, rows)
        )

    analysis = aligned_trend_sync_analysis(groups, "up")
    answer = multi_trend_metric_sentence("最近4天指标A、指标B和指标C是否同步上升？", plan, run)

    assert analysis["expectedIntervals"] == 3
    assert analysis["comparableIntervals"] == 1
    assert analysis["matchingIntervals"] == 1
    assert "共同日期不足，暂不能判断是否同步上升" in answer
    assert "1/3 个相邻日区间可比" in answer
    assert "缺失日期未按 0 处理" in answer


def test_multi_metric_extreme_changes_are_reported_per_metric_without_cross_unit_ranking(tmp_path):
    plan, run = trend_plan_and_run(
        tmp_path,
        {
            "metric_a": {"label": "指标A", "points": daily_points([1, 5, 6])},
            "metric_b": {"label": "指标B", "points": daily_points([1000, 1001, 1600])},
        },
    )

    answer = multi_trend_metric_sentence("最近3天指标A和指标B哪一天波动最大？", plan, run)

    assert "指标A 在 2026-06-02" in answer
    assert "指标B 在 2026-06-03" in answer
    assert "不同量纲不做横向大小比较" in answer
