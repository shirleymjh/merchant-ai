from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    AnswerMode,
    IntentType,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    VerifiedEvidence,
)
from merchant_ai.services.answer import (
    answer_skill_headers,
    contextual_business_suggestions,
    metric_disclosures,
    render_structured_skill_answer,
    select_answer_skill,
)
from merchant_ai.services.answer_formatting import format_metric_value_for_answer
from merchant_ai.services.llm import LlmClient
from merchant_ai.services.skill_worker import SkillWorkerExecutor


def _write_placeholder_skill(resources_root: Path) -> Path:
    skill_dir = resources_root / "runtime" / "agent_skills" / "placeholder_signal_review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: placeholder_signal_review
description: Review a placeholder metric using verified evidence only.
title: 占位信号复核
executionMode: structured_renderer
renderer: verified_evidence
---

# Placeholder Signal Review

## Activation Contract

Use only when the structured plan declares this resource name.
""",
        encoding="utf-8",
    )
    return skill_dir


def _placeholder_plan(skill_name: str = "") -> QueryPlan:
    understanding = {"analysisIntent": "diagnosis", "requiresExplanation": True}
    if skill_name:
        understanding["skillWorkflow"] = {"skillName": skill_name, "required": True}
    return QueryPlan(
        question_understanding=understanding,
        intents=[
            QuestionIntent(
                question="检查占位信号",
                intent_type=IntentType.VALID,
                answer_mode=AnswerMode.METRIC,
                plan_task_id="placeholder_task",
                metric_name="opaque_refund_gmv_amt",
                metric_resolution={
                    "metricKey": "opaque_refund_gmv_amt",
                    "displayName": "占位信号",
                    "description": "按占位事件统计",
                    "unit": "星",
                    "valueFormat": "decimal",
                    "sourceColumnLabels": {"opaque_refund_gmv_amt": "占位信号"},
                },
            )
        ],
    )


def _placeholder_run() -> AgentRunResult:
    bundle = QueryBundle(rows=[{"opaque_refund_gmv_amt": 12}], original_row_count=1)
    return AgentRunResult(
        task_results=[AgentTaskResult(task_id="placeholder_task", success=True, query_bundle=bundle)],
        query_bundles=[bundle],
        merged_query_bundle=bundle,
        verified_evidence=VerifiedEvidence(passed=True),
    )


def test_skill_selection_only_accepts_runtime_header_declarations(tmp_path: Path):
    resources_root = tmp_path / "resources"
    _write_placeholder_skill(resources_root)
    headers = answer_skill_headers(resources_root / "runtime" / "agent_skills")

    assert [item["name"] for item in headers] == ["placeholder_signal_review"]
    assert headers[0]["executionMode"] == "structured_renderer"
    assert headers[0]["renderer"] == "verified_evidence"
    assert select_answer_skill(
        _placeholder_plan("placeholder_signal_review"), skill_headers=headers
    ) == "placeholder_signal_review"
    assert select_answer_skill(_placeholder_plan("not_installed"), skill_headers=headers) == ""
    assert select_answer_skill(_placeholder_plan(), skill_headers=headers) == ""


def test_skill_worker_uses_frontmatter_execution_metadata(tmp_path: Path, monkeypatch):
    resources_root = tmp_path / "resources"
    _write_placeholder_skill(resources_root)
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path / "workspace"),
            "llm_api_key": "",
            "distributed_subagents_enabled": False,
        }
    )
    monkeypatch.setattr(type(settings), "resources_root", property(lambda _settings: resources_root))

    result = SkillWorkerExecutor(LlmClient(settings)).execute_answer_skill(
        "检查占位信号",
        _placeholder_plan("placeholder_signal_review"),
        _placeholder_run(),
        skill_name="placeholder_signal_review",
    )

    assert "占位信号复核" in result.answer
    assert "占位信号=12星" in result.answer
    assert result.trace["executionMode"] == "structured_renderer"
    assert result.trace["renderer"] == "verified_evidence"


def test_skill_worker_does_not_default_when_no_resource_was_matched(tmp_path: Path):
    settings = get_settings().model_copy(update={"harness_workspace_path": str(tmp_path), "llm_api_key": ""})

    result = SkillWorkerExecutor(LlmClient(settings)).execute_answer_skill(
        "任意问题",
        _placeholder_plan(),
        _placeholder_run(),
        skill_name="",
    )

    assert result.answer == ""
    assert result.trace["matchStatus"] == "no_match"
    assert result.trace["lifecycleStage"] == "skipped"


def test_metric_formatting_uses_metadata_not_metric_identifier_tokens():
    assert format_metric_value_for_answer(123, "opaque_refund_gmv_amt", "任意名称") == "123"
    assert format_metric_value_for_answer(
        123,
        "opaque_refund_gmv_amt",
        "任意名称",
        {"unit": "星", "valueFormat": "decimal"},
    ) == "123星"
    assert format_metric_value_for_answer(
        0.125,
        "opaque_value",
        "占位比率",
        {"unit": "%", "valueFormat": "percent"},
    ) == "12.5%"


def test_metric_disclosure_preserves_placeholder_display_contract():
    plan = _placeholder_plan()

    disclosures = metric_disclosures(plan, VerifiedEvidence(passed=True))

    assert disclosures == [
        {
            "metricKey": "opaque_refund_gmv_amt",
            "displayName": "占位信号",
            "description": "按占位事件统计",
            "unit": "星",
            "valueFormat": "decimal",
            "sourceColumnLabels": {"opaque_refund_gmv_amt": "占位信号"},
        }
    ]


def test_suggestions_are_derived_from_verified_placeholder_labels():
    plan = _placeholder_plan()

    suggestions = contextual_business_suggestions(
        "检查占位信号",
        plan.intents,
        run_result=_placeholder_run(),
    )

    assert suggestions[:3] == [
        "查看占位信号按时间维度的变化",
        "按已验证维度拆解占位信号",
        "核对占位信号波动区间对应的明细",
    ]
    assert all("退款" not in item and "GMV" not in item for item in suggestions)


def test_structured_renderer_fails_closed_without_resource_metadata():
    assert render_structured_skill_answer(
        "verified_evidence",
        {"dataRows": [{"opaque_refund_gmv_amt": 12}]},
    ) == ""
