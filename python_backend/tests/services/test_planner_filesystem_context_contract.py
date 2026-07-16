from __future__ import annotations

import json
from typing import Any

from merchant_ai.config import get_settings
from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack
from merchant_ai.services.planning import QueryGraphPlanner


def planner_pack(metric_count: int = 36) -> PlanningAssetPack:
    tables = [
        PlanningAssetEntry(
            key=f"fact_{index}",
            table=f"fact_{index}",
            topic=f"DOMAIN_{index}",
            columns=["event_day", *[f"value_{item}" for item in range(metric_count)]],
            source_ref_id=f"semantic:DOMAIN_{index}:fact_{index}:table",
            metadata={
                "timeColumn": "event_day",
                "oversizedSchemaSentinel": "FULL_SCHEMA_MUST_NOT_ENTER_L0" * 200,
            },
        )
        for index in range(3)
    ]
    metrics = [
        PlanningAssetEntry(
            key=f"measure_{index}",
            table=f"fact_{index % 3}",
            topic=f"DOMAIN_{index % 3}",
            title=f"Measure {index}",
            aliases=[f"Measure alias {index}"],
            columns=[f"value_{index}"],
            source_ref_id=f"semantic:DOMAIN_{index % 3}:fact_{index % 3}:metric:measure_{index}",
            metadata={
                "formula": f"SUM(value_{index})",
                "sourceColumns": [f"value_{index}"],
                "oversizedDefinitionSentinel": "FULL_METRIC_DETAIL_MUST_NOT_ENTER_L0" * 200,
            },
        )
        for index in range(metric_count)
    ]
    return PlanningAssetPack(
        tables=tables,
        metrics=metrics,
        metric_compaction={
            "targetedSeed": {
                "trace": [
                    "precise_metric_seed:fact_0:measure_0:Requested zero",
                    "precise_metric_seed:fact_1:measure_1:Requested one",
                ]
            },
            "recalledMetricEvidence": [
                {
                    "ownerTable": "fact_0",
                    "metricKey": "measure_0",
                    "semanticRefId": "semantic:DOMAIN_0:fact_0:metric:measure_0",
                    "matchedMetricLabel": "Requested zero",
                    "metricResolutionType": "exact_alias",
                    "metricResolutionConfidence": 0.97,
                    "metricResolutionAmbiguous": False,
                },
                {
                    "ownerTable": "fact_1",
                    "metricKey": "measure_1",
                    "semanticRefId": "semantic:DOMAIN_1:fact_1:metric:measure_1",
                    "matchedMetricLabel": "Requested one",
                    "metricResolutionType": "exact_alias",
                    "metricResolutionConfidence": 0.97,
                    "metricResolutionAmbiguous": False,
                },
            ],
            "fastUnderstanding": {
                "intentKind": "analysis",
                "analysisIntent": "trend",
                "metricPhrases": ["Requested zero", "Requested one"],
                "timeWindowDays": 11,
                "timeRange": {
                    "kind": "rolling",
                    "days": 11,
                    "anchorPolicy": "latest_available_partition",
                    "source": "structured_test_contract",
                },
            },
        },
    )


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def test_planner_l0_prompt_is_bounded_and_does_not_inline_asset_details() -> None:
    class CaptureModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.calls: list[tuple[str, str, dict[str, Any]]] = []

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, **kwargs):
            self.calls.append((system_prompt, user_prompt, tool_schema))
            return {}

    settings = get_settings().model_copy(update={"agent_planner_prompt_budget_chars": 20000})
    model = CaptureModel()
    planner = QueryGraphPlanner(model, settings=settings)
    payload = planner._llm_understand(
        "opaque request whose words are not used to trim the L0 index",
        planner_pack(),
        [],
        [],
        use_tool_loop=False,
    )

    assert model.calls
    assert payload["_promptStats"]["totalChars"] <= settings.agent_planner_prompt_budget_chars
    user_payload = json.loads(model.calls[0][1])
    catalog = user_payload["semanticCatalog"]
    assert catalog["mode"] == "filesystem_workspace_index"
    assert not nested_keys(catalog).intersection(
        {"formula", "sourceColumns", "columns", "aliases", "keyColumns", "joinKeys", "schema"}
    )
    serialized = json.dumps(user_payload, ensure_ascii=False)
    assert "FULL_SCHEMA_MUST_NOT_ENTER_L0" not in serialized
    assert "FULL_METRIC_DETAIL_MUST_NOT_ENTER_L0" not in serialized
    assert user_payload["planningContract"]["timeWindowDays"] == 11
    assert catalog["candidateBudget"]["availableMetrics"] == 36


def test_planner_l0_candidate_refs_do_not_change_with_question_wording() -> None:
    planner = QueryGraphPlanner(type("NoModel", (), {"configured": False})())
    pack = planner_pack()
    first = planner._understanding_payload("completely unrelated wording A", pack, [], [], False, False, None)
    second = planner._understanding_payload("completely unrelated wording B", pack, [], [], False, False, None)

    first_refs = [item["sourceRefId"] for item in first["semanticCatalog"]["candidateMetrics"]]
    second_refs = [item["sourceRefId"] for item in second["semanticCatalog"]["candidateMetrics"]]
    assert first_refs == second_refs
    assert first_refs[:2] == [
        "semantic:DOMAIN_0:fact_0:metric:measure_0",
        "semantic:DOMAIN_1:fact_1:metric:measure_1",
    ]


def test_planner_prompt_budget_fails_closed_without_calling_model() -> None:
    class ModelMustNotRun:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def tool_json_chat(self, *args, **kwargs):
            raise AssertionError("over-budget Planner prompt must not be sent")

    planner = QueryGraphPlanner(
        ModelMustNotRun(),
        settings=get_settings().model_copy(update={"agent_planner_prompt_budget_chars": 100}),
    )
    payload = planner._llm_understand("opaque", planner_pack(), [], [], use_tool_loop=False)

    assert payload["_plannerContextOverBudget"] is True
    assert payload["_promptStats"]["budgetPolicy"] == "fail_closed"
    assert payload["status"] == "NEED_MORE_KNOWLEDGE"


def test_adaptive_planner_reads_exact_ref_before_emitting_understanding() -> None:
    selected_ref = "semantic:DOMAIN_0:fact_0:metric:measure_0"

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            assert ref_id == selected_ref
            return {
                "success": True,
                "refId": ref_id,
                "content": '{"metricKey":"measure_0","formula":"SUM(value_0)"}',
            }

        def ls(self, **kwargs):
            return []

        def grep(self, **kwargs):
            return []

    class ProgressiveModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []
            self.prompt_sizes: list[int] = []

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, **kwargs):
            payload = json.loads(user_prompt)
            self.payloads.append(payload)
            self.prompt_sizes.append(
                len(system_prompt) + len(user_prompt) + len(json.dumps(tools, ensure_ascii=False))
            )
            if len(self.payloads) == 1:
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
            if len(self.payloads) == 2:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "read_selected_metric",
                            "name": "semantic_read",
                            "args": {"refId": selected_ref, "maxChars": 1000},
                        }
                    ],
                }
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "emit_after_read",
                        "name": "emit_question_understanding",
                        "args": {
                            "status": "UNDERSTOOD",
                            "reason": "selected ref was read",
                            "questionUnderstanding": {
                                "analysisGrain": "time",
                                "analysisIntent": "trend_check",
                                "requiresExplanation": False,
                                "requiredEvidenceIntents": [],
                                "rankingObjective": {
                                    "metricRef": "measure_0",
                                    "ownerTable": "fact_0",
                                    "sourcePhrase": "Requested zero",
                                    "groupByColumn": "event_day",
                                },
                                "requestedMeasures": [],
                                "filters": [],
                                "timeWindowDays": 11,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20000,
            "agent_planner_tool_rounds": 3,
            "agent_deferred_tool_schema_enabled": True,
        }
    )
    model = ProgressiveModel()
    planner = QueryGraphPlanner(model, semantic_catalog=SemanticCatalog(), settings=settings)
    payload = planner._llm_understand(
        "opaque",
        planner_pack(),
        [],
        [],
        use_tool_loop=True,
        filesystem_context_entry="adaptive",
    )

    assert payload["status"] == "UNDERSTOOD"
    assert selected_ref in payload["_plannerLoadedRefs"]
    assert model.payloads[0]["filesystemContextPolicy"]["mustReadBeforeEmit"] is True
    assert "formula" not in nested_keys(model.payloads[0]["semanticCatalog"])
    assert any(
        "SUM(value_0)" in json.dumps(item, ensure_ascii=False)
        for item in model.payloads[2]["plannerToolResults"]
    ), model.prompt_sizes
    assert max(model.prompt_sizes) <= settings.agent_planner_prompt_budget_chars


def test_active_planner_round_keeps_multiple_exact_metric_reads_under_budget() -> None:
    selected_refs = [
        f"semantic:DOMAIN_{index % 3}:fact_{index % 3}:metric:measure_{index}"
        for index in range(5)
    ]

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            index = int(ref_id.rsplit("measure_", 1)[-1])
            table_index = index % 3
            return {
                "success": True,
                "refId": ref_id,
                "path": f"topics/DOMAIN_{table_index}/tables/fact_{table_index}/asset.json#metric:measure_{index}",
                "kind": "METRIC",
                "topic": f"DOMAIN_{table_index}",
                "table": f"fact_{table_index}",
                "content": json.dumps(
                    {
                        "metric": {
                            "metricKey": f"measure_{index}",
                            "businessName": f"Measure {index}",
                            "aliases": [f"Requested {index}", *[f"Alias {item}" for item in range(20)]],
                            "formula": f"SUM(value_{index})",
                            "sourceColumns": [f"value_{index}"],
                            "aggregationPolicy": "sum",
                            "selectionGuidance": "runtime governed metric definition " * 20,
                            "irrelevantGovernanceEnvelope": "MUST_BE_COMPACTED" * 300,
                        }
                    }
                ),
            }

        def ls(self, **kwargs):
            return []

        def grep(self, **kwargs):
            return []

    class MultiReadModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []
            self.prompt_sizes: list[int] = []

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, **kwargs):
            payload = json.loads(user_prompt)
            self.payloads.append(payload)
            self.prompt_sizes.append(
                len(system_prompt) + len(user_prompt) + len(json.dumps(tools, ensure_ascii=False))
            )
            if len(self.payloads) == 1:
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
            if len(self.payloads) == 2:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": f"read_{index}",
                            "name": "semantic_read",
                            "args": {"refId": ref_id, "maxChars": 20_000},
                        }
                        for index, ref_id in enumerate(selected_refs)
                    ],
                }
            results = payload["plannerToolResults"]
            assert {str((item.get("result") or {}).get("refId") or "") for item in results} == set(selected_refs)
            assert "MUST_BE_COMPACTED" not in json.dumps(results, ensure_ascii=False)
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "emit_after_multi_read",
                        "name": "emit_question_understanding",
                        "args": {
                            "status": "UNDERSTOOD",
                            "questionUnderstanding": {
                                "analysisGrain": "time",
                                "analysisIntent": "trend_check",
                                "requiresExplanation": False,
                                "requiredEvidenceIntents": [],
                                "rankingObjective": {
                                    "metricRef": "measure_0",
                                    "ownerTable": "fact_0",
                                    "sourcePhrase": "Requested 0",
                                    "groupByColumn": "event_day",
                                },
                                "requestedMeasures": [
                                    {
                                        "metricRef": f"measure_{index}",
                                        "ownerTable": f"fact_{index % 3}",
                                        "sourcePhrase": f"Requested {index}",
                                        "groupByColumn": "event_day",
                                    }
                                    for index in range(1, 5)
                                ],
                                "filters": [],
                                "timeWindowDays": 11,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20_000,
            "agent_planner_tool_rounds": 3,
            "agent_deferred_tool_schema_enabled": True,
        }
    )
    model = MultiReadModel()
    planner = QueryGraphPlanner(model, semantic_catalog=SemanticCatalog(), settings=settings)

    payload = planner._llm_understand(
        "opaque",
        planner_pack(),
        [],
        [],
        use_tool_loop=True,
        filesystem_context_entry="adaptive",
    )

    assert payload["status"] == "UNDERSTOOD"
    assert set(payload["_plannerLoadedRefs"]) == set(selected_refs)
    assert max(model.prompt_sizes) <= settings.agent_planner_prompt_budget_chars
    assert payload["_promptStats"]["toolRounds"][-1]["compaction"] == "active_semantic_results"
