from __future__ import annotations

import json
import time
from typing import Any, Dict

import pytest

from merchant_ai.config import Settings, get_settings
from merchant_ai.graph.workflow import planner_degraded_state
from merchant_ai.models import PlanningAssetPack, QueryPlan
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    QueryGraphValidator,
    append_prompt_trace,
    planner_operational_trace_reason,
)
from merchant_ai.services.planning_tooling import planner_failure_gap_code
from merchant_ai.services.tools import question_understanding_tool


SEMANTIC_REF = "semantic:test:fact_0:metric:measure_0"
TABLE_DETAIL_REF = "semantic:test:fact_0:detail"
GROUP_FIELD_REF = "semantic:test:fact_0:field:entity_id"


class SemanticCatalog:
    def read(self, ref_id: str, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "success": True,
            "refId": ref_id,
            "content": '{"metricKey":"measure_0","ownerTable":"fact_0"}',
        }

    def ls(self, **_kwargs: Any) -> list[Dict[str, Any]]:
        return []

    def grep(self, *_args: Any, **_kwargs: Any) -> list[Dict[str, Any]]:
        return []


class ThirdRoundTransientLlm:
    configured = True
    error_events: list[str] = []

    def __init__(self, recover: bool) -> None:
        self.recover = recover
        self.last_error = ""
        self.calls: list[Dict[str, Any]] = []

    def tool_chat(
        self,
        _system_prompt: str,
        user_prompt: str,
        _tools: list[Dict[str, Any]],
        fallback: Dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        tool_choice: str | None = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "userPrompt": user_prompt,
                "payload": json.loads(user_prompt),
                "timeoutSeconds": timeout_seconds,
                "toolChoice": tool_choice,
            }
        )
        call_number = len(self.calls)
        if call_number == 1:
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "load_read_schema",
                        "name": "load_tool_schemas",
                        "args": {"toolNames": ["semantic_read"]},
                    }
                ],
            }
        if call_number == 2:
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "read_metric",
                        "name": "semantic_read",
                        "args": {"refId": SEMANTIC_REF, "maxChars": 1000},
                    },
                    {
                        "id": "read_owner_detail",
                        "name": "semantic_read",
                        "args": {"refId": TABLE_DETAIL_REF, "maxChars": 1000},
                    },
                    {
                        "id": "read_group_field",
                        "name": "semantic_read",
                        "args": {"refId": GROUP_FIELD_REF, "maxChars": 1000},
                    },
                ],
            }
        if call_number == 3 or not self.recover:
            self.last_error = "timeout: stale adapter error"
            raise TimeoutError("provider call exceeded 20 seconds")
        return {
            "content": "",
            "toolCalls": [
                {
                    "id": "emit_after_retry",
                    "name": "emit_question_understanding",
                    "args": {
                        "status": "UNDERSTOOD",
                        "reason": "reused the already-read semantic contract",
                        "questionUnderstanding": {
                            "analysisGrain": "entity",
                            "analysisIntent": "none",
                            "requiresExplanation": False,
                            "requiredEvidenceIntents": [],
                            "rankingObjective": {
                                "metricRef": "measure_0",
                                "ownerTable": "fact_0",
                                "sourcePhrase": "governed measure",
                                "groupByColumn": "entity_id",
                                "order": "desc",
                                "limit": 10,
                            },
                            "requestedMeasures": [],
                            "filters": [],
                            "semanticQuery": {
                                "resultMode": "metric",
                                "filterNodes": [],
                                "rootFilterNodeId": "",
                                "selectRefIds": [],
                                "measureRefIds": [SEMANTIC_REF],
                                "dimensionRefIds": [],
                                "sourceRefIds": [],
                                "relationshipRefIds": [],
                                "joinStrategy": "auto",
                                "orderBy": [],
                                "limit": 0,
                                "bindingStatus": "unresolved",
                            },
                            "timeWindowDays": 30,
                        },
                    },
                }
            ],
        }


def planner_for(tmp_path: Any, llm: Any, **settings_updates: Any) -> QueryGraphPlanner:
    settings = get_settings().model_copy(
        update={
            "agent_planner_tool_rounds": 3,
            "agent_deferred_tool_schema_enabled": True,
            "agent_planner_transient_retries": 1,
            "llm_planner_timeout_seconds": 20,
            "llm_answer_timeout_seconds": 10,
            **settings_updates,
        }
    )
    return QueryGraphPlanner(
        llm,
        semantic_catalog=SemanticCatalog(),
        artifact_store=WorkspaceArtifactStore(settings, tmp_path),
        settings=settings,
    )


def runtime_budget(seconds: float = 90.0) -> Dict[str, Any]:
    return {"runtimeBudget": {"deadlineEpochMs": time.time() * 1000.0 + seconds * 1000.0}}


def run_tool_loop(planner: QueryGraphPlanner, budget_seconds: float = 90.0) -> Dict[str, Any]:
    return planner._llm_understand_with_semantic_tools(
        "planner system prompt",
        {"filesystemContextPolicy": {"entry": "adaptive"}},
        question_understanding_tool(False),
        False,
        require_semantic_read_before_emit=True,
        prompt_budget=30_000,
        planner_context=runtime_budget(budget_seconds),
    )


def test_third_tool_round_timeout_retries_same_context_and_recovers_without_blocking_gap(tmp_path: Any) -> None:
    llm = ThirdRoundTransientLlm(recover=True)
    payload = run_tool_loop(planner_for(tmp_path, llm))

    assert payload["status"] == "UNDERSTOOD"
    assert len(llm.calls) == 4
    assert llm.calls[2]["userPrompt"] == llm.calls[3]["userPrompt"]
    assert llm.calls[2]["toolChoice"] == "emit_question_understanding"
    assert any(item.get("name") == "semantic_read" for item in llm.calls[2]["payload"]["plannerToolResults"])
    assert payload["_plannerLoadedRefs"] == sorted(
        [SEMANTIC_REF, TABLE_DETAIL_REF, GROUP_FIELD_REF]
    )
    assert payload["_plannerRecoveredAfterRetry"] is True
    attempts = payload["_plannerOperationalAttempts"]
    assert [(item["round"], item["errorCode"], item["retryScheduled"]) for item in attempts] == [
        (3, "PLANNER_LLM_TIMEOUT", True)
    ]
    assert attempts[0]["status"] == "recovered"
    assert attempts[0]["recoveredOnAttempt"] == 2

    plan = QueryPlan()
    append_prompt_trace(
        plan,
        {
            "_promptTrace": {"promptId": "planner.question_understanding", "version": "test"},
            "_promptStats": {
                "operationalAttempts": attempts,
                "recoveredAfterTransientFailure": True,
            },
        },
    )
    assert planner_failure_gap_code(plan) == ""


def test_repeated_tool_round_timeout_fails_closed_and_preserves_original_typed_gap(tmp_path: Any) -> None:
    llm = ThirdRoundTransientLlm(recover=False)
    payload = run_tool_loop(planner_for(tmp_path, llm))

    assert payload["_plannerFailFast"] is True
    assert len(llm.calls) == 4
    assert payload["_plannerLoadedRefs"] == sorted(
        [SEMANTIC_REF, TABLE_DETAIL_REF, GROUP_FIELD_REF]
    )
    attempts = payload["_plannerOperationalAttempts"]
    assert [item["errorCode"] for item in attempts] == ["PLANNER_LLM_TIMEOUT", "PLANNER_LLM_TIMEOUT"]
    assert attempts[0]["retryScheduled"] is True
    assert attempts[1]["retrySkippedReason"] == "retry_budget_exhausted"

    failed_plan = QueryPlan(
        agent_trace=[planner_operational_trace_reason(attempts)],
        planner_prompt_stats={"operationalAttempts": attempts},
    )
    validation = QueryGraphValidator().validate("governed question", failed_plan, PlanningAssetPack())
    assert [gap.code for gap in validation.gaps] == ["PLANNER_LLM_TIMEOUT"]
    degraded = planner_degraded_state("", failed_plan)
    assert degraded["active"] is True
    assert degraded["code"] == "PLANNER_LLM_TIMEOUT"


def test_planner_403_is_not_retried(tmp_path: Any) -> None:
    class ForbiddenLlm:
        configured = True
        last_error = "HTTP 403 forbidden"

    planner = planner_for(tmp_path, ForbiddenLlm())
    calls = 0

    def invoke(_timeout_seconds: int) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    _, attempts, retry_count = planner._invoke_planner_llm_with_transient_retry(
        invoke,
        bool,
        runtime_budget(),
        retry_limit=1,
        observation_context={"phase": "direct_understanding"},
    )

    assert calls == 1
    assert retry_count == 0
    assert attempts[0]["errorCode"] == "PLANNER_PROVIDER_ERROR"
    assert attempts[0]["retryable"] is False


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("", "PLANNER_EMPTY_RESPONSE"),
        ("Connection reset by peer", "PLANNER_PROVIDER_ERROR"),
        ("HTTP 503 service unavailable", "PLANNER_PROVIDER_ERROR"),
    ],
)
def test_planner_empty_connection_and_5xx_retry_once_then_recover(
    tmp_path: Any,
    failure: str,
    expected_code: str,
) -> None:
    class TransientLlm:
        configured = True
        last_error = ""

    llm = TransientLlm()
    planner = planner_for(tmp_path, llm)
    calls = 0

    def invoke(_timeout_seconds: int) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            llm.last_error = failure
            return {}
        llm.last_error = ""
        return {"status": "UNDERSTOOD"}

    result, attempts, retry_count = planner._invoke_planner_llm_with_transient_retry(
        invoke,
        bool,
        runtime_budget(),
        retry_limit=1,
        observation_context={"phase": "direct_understanding"},
    )

    assert result == {"status": "UNDERSTOOD"}
    assert calls == 2
    assert retry_count == 1
    assert attempts[0]["errorCode"] == expected_code
    assert attempts[0]["status"] == "recovered"


def test_planner_retry_requires_global_budget_for_full_timeout_and_answer_reserve(tmp_path: Any) -> None:
    class TimeoutLlm:
        configured = True
        last_error = "timeout: provider call exceeded 20 seconds"

    planner = planner_for(tmp_path, TimeoutLlm())
    calls = 0

    def invoke(_timeout_seconds: int) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    _, attempts, retry_count = planner._invoke_planner_llm_with_transient_retry(
        invoke,
        bool,
        runtime_budget(25),
        retry_limit=1,
        observation_context={"phase": "direct_understanding"},
    )

    assert calls == 1
    assert retry_count == 0
    assert attempts[0]["retrySkippedReason"] == "insufficient_run_budget"
    assert Settings.model_fields["agent_planner_transient_retries"].default == 1
