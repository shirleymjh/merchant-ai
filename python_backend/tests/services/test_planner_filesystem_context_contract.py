from __future__ import annotations

import json
from typing import Any

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import PlanningAssetEntry, PlanningAssetPack
from merchant_ai.services.planning import (
    QueryGraphPlanner,
    compact_semantic_candidate_hints_for_prompt,
    filesystem_workspace_index_catalog,
)
from merchant_ai.services.planning_tooling import compact_tool_result_for_prompt


def empty_semantic_query(result_mode: str = "metric") -> dict[str, Any]:
    return {
        "resultMode": result_mode,
        "filterNodes": [],
        "rootFilterNodeId": "",
        "selectRefIds": [],
        "measureRefIds": [],
        "dimensionRefIds": [],
        "sourceRefIds": [],
        "relationshipRefIds": [],
        "joinStrategy": "auto",
        "orderBy": [],
        "limit": 0,
        "bindingStatus": "unresolved",
    }


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
        table_manifest={
            "mode": "stable_topic_table_manifest",
            "questionIndependent": True,
            "topics": [
                {
                    "topic": f"DOMAIN_{index}",
                    "topicId": f"DOMAIN_{index}",
                    "manifestRefId": f"semantic:DOMAIN_{index}:manifest",
                    "path": f"topics/DOMAIN_{index}/manifest.json",
                    "tables": [
                        {
                            "topic": f"DOMAIN_{index}",
                            "table": f"fact_{index}",
                            "sourceRefId": f"semantic:DOMAIN_{index}:fact_{index}:asset",
                            "path": f"topics/DOMAIN_{index}/tables/fact_{index}/asset.json",
                        }
                    ],
                }
                for index in range(3)
            ],
            "tables": [
                {
                    "topic": f"DOMAIN_{index}",
                    "table": f"fact_{index}",
                    "sourceRefId": f"semantic:DOMAIN_{index}:fact_{index}:asset",
                    "path": f"topics/DOMAIN_{index}/tables/fact_{index}/asset.json",
                }
                for index in range(3)
            ],
            "tableCount": 3,
        },
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
                    "calendarAnchorPolicy": "runtime_current_date",
                    "dataAsOfPolicy": "latest_available_partition",
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


def test_topic_table_manifest_is_stable_when_rag_candidates_change() -> None:
    table_manifest = {
        "mode": "stable_topic_table_manifest",
        "questionIndependent": True,
        "topics": [
            {
                "topic": "DOMAIN",
                "topicId": "DOMAIN",
                "manifestRefId": "semantic:DOMAIN:manifest",
                "tables": [
                    {"topic": "DOMAIN", "table": "fact_a", "sourceRefId": "semantic:DOMAIN:fact_a:asset"},
                    {"topic": "DOMAIN", "table": "fact_b", "sourceRefId": "semantic:DOMAIN:fact_b:asset"},
                ],
            }
        ],
        "tables": [
            {"topic": "DOMAIN", "table": "fact_a", "sourceRefId": "semantic:DOMAIN:fact_a:asset"},
            {"topic": "DOMAIN", "table": "fact_b", "sourceRefId": "semantic:DOMAIN:fact_b:asset"},
        ],
        "tableCount": 2,
    }
    first = PlanningAssetPack(
        table_manifest=table_manifest,
        tables=[PlanningAssetEntry(key="fact_a", table="fact_a", topic="DOMAIN", source_ref_id="rag:a")],
    )
    second = PlanningAssetPack(
        table_manifest=table_manifest,
        tables=[PlanningAssetEntry(key="fact_b", table="fact_b", topic="DOMAIN", source_ref_id="rag:b")],
    )

    first_catalog = filesystem_workspace_index_catalog(first, "question one")
    second_catalog = filesystem_workspace_index_catalog(second, "unrelated question two")

    assert first_catalog["tableManifest"] == second_catalog["tableManifest"]
    assert [item["table"] for item in first_catalog["tableManifest"]["tables"]] == ["fact_a", "fact_b"]
    assert not {"tables", "candidateMetrics", "relationships"}.intersection(first_catalog)
    assert all(
        set(item) == {"topic", "table", "title", "detailRefId", "detailPath"}
        for item in first_catalog["tableManifest"]["tables"]
    )


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
    assert catalog["tableManifest"]["questionIndependent"] is True
    assert [item["table"] for item in catalog["tableManifest"]["tables"]] == ["fact_0", "fact_1", "fact_2"]
    assert not nested_keys(catalog).intersection(
        {"formula", "sourceColumns", "columns", "aliases", "keyColumns", "joinKeys", "schema"}
    )
    serialized = json.dumps(user_payload, ensure_ascii=False)
    assert "FULL_SCHEMA_MUST_NOT_ENTER_L0" not in serialized
    assert "FULL_METRIC_DETAIL_MUST_NOT_ENTER_L0" not in serialized
    assert user_payload["planningContract"]["timeWindowDays"] == 11
    assert set(catalog["candidateBudget"]) == {"availableTables"}
    assert catalog["candidateBudget"]["availableTables"] == 3
    assert not {"tables", "candidateMetrics", "relationships"}.intersection(catalog)
    assert all(
        set(item) == {"topic", "table", "title", "detailRefId", "detailPath"}
        for item in catalog["tableManifest"]["tables"]
    )


def test_planner_l0_table_detail_refs_do_not_change_with_question_wording() -> None:
    planner = QueryGraphPlanner(type("NoModel", (), {"configured": False})())
    pack = planner_pack()
    first = planner._understanding_payload("completely unrelated wording A", pack, [], [], False, False, None)
    second = planner._understanding_payload("completely unrelated wording B", pack, [], [], False, False, None)

    first_tables = first["semanticCatalog"]["tableManifest"]["tables"]
    second_tables = second["semanticCatalog"]["tableManifest"]["tables"]
    first_refs = [item["detailRefId"] for item in first_tables]
    second_refs = [item["detailRefId"] for item in second_tables]
    assert first_refs == second_refs
    assert first_refs[:2] == [
        "semantic:DOMAIN_0:fact_0:detail",
        "semantic:DOMAIN_1:fact_1:detail",
    ]
    assert all(set(item) == {"topic", "table", "title", "detailRefId", "detailPath"} for item in first_tables)


def test_deepagent_core_owns_semantic_tools_and_planner_only_consumes_read_ledger() -> None:
    detail_ref = "semantic:DOMAIN_0:fact_0:detail"
    metric_ref = "semantic:DOMAIN_0:fact_0:metric:measure_0"
    schema_ref = "semantic:DOMAIN_0:fact_0:schema"

    class CoreManagedModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.tool_json_calls: list[tuple[str, dict[str, Any]]] = []
            self.tool_chat_calls = 0

        def tool_chat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            self.tool_chat_calls += 1
            raise AssertionError("core-managed Planner must not start a semantic tool loop")

        def tool_json_chat(self, system_prompt, user_prompt, tool_schema, fallback=None, **kwargs):
            del tool_schema, fallback, kwargs
            self.tool_json_calls.append((system_prompt, json.loads(user_prompt)))
            return {
                "status": "UNDERSTOOD",
                "reason": "compiled only from Core-read evidence",
                "questionUnderstanding": {
                    "analysisGrain": "metric",
                    "analysisIntent": "none",
                    "requiresExplanation": False,
                    "requiredEvidenceIntents": [],
                    "anchorMetric": {
                        "metricRef": "measure_0",
                        "sourcePhrase": "Requested zero",
                        "ownerTable": "fact_0",
                        "objectiveType": "metric_total",
                        "groupByColumn": "event_day",
                        "order": "desc",
                        "limit": 1,
                    },
                    "supportMetrics": [],
                    "metricCandidateDecisions": [],
                    "calculationIntents": [],
                    "scopeConstraints": [],
                    "filters": [],
                    "semanticQuery": empty_semantic_query(),
                    "timeWindowDays": 11,
                },
            }

    evidence = [
        {
            "refId": detail_ref,
            "path": "topics/DOMAIN_0/tables/fact_0/detail.json",
            "kind": "TABLE_DETAIL",
            "topic": "DOMAIN_0",
            "table": "fact_0",
            "contentSnippet": '{"tableName":"fact_0"}',
            "contentHash": "detail-hash",
        },
        {
            "refId": metric_ref,
            "path": "topics/DOMAIN_0/tables/fact_0/metrics/measure_0.json",
            "kind": "METRIC",
            "topic": "DOMAIN_0",
            "table": "fact_0",
            "contentSnippet": '{"metricKey":"measure_0","formula":"SUM(value_0)"}',
            "contentHash": "metric-hash",
        },
        {
            "refId": schema_ref,
            "path": "topics/DOMAIN_0/tables/fact_0/schema.json",
            "kind": "SCHEMA",
            "topic": "DOMAIN_0",
            "table": "fact_0",
            "contentSnippet": '{"columns":[{"columnName":"event_day"}]}',
            "contentHash": "schema-hash",
        },
    ]
    model = CoreManagedModel()
    planner = QueryGraphPlanner(
        model,
        semantic_catalog=object(),
        settings=get_settings().model_copy(
            update={"agent_planner_tool_rounds": 12, "agent_planner_prompt_budget_chars": 20_000}
        ),
    )
    planner_context = {
        "coreManagedFilesystem": True,
        "coreSemanticEvidence": evidence,
    }

    assert planner._initial_semantic_tool_entry("opaque", planner_pack(), [], planner_context) == ""
    payload = planner._llm_understand(
        "Requested zero",
        planner_pack(),
        [],
        [],
        planner_context=planner_context,
        use_tool_loop=False,
    )

    assert payload["status"] == "UNDERSTOOD"
    assert model.tool_chat_calls == 0
    assert len(model.tool_json_calls) == 1
    system_prompt, user_payload = model.tool_json_calls[0]
    assert "core_managed_filesystem" in system_prompt
    assert user_payload["filesystemAuthority"]["owner"] == "DeepAgent Core"
    assert user_payload["filesystemAuthority"]["hiddenPlannerToolLoop"] == "disabled"
    assert [item["refId"] for item in user_payload["coreSemanticEvidence"]] == [
        detail_ref,
        metric_ref,
        schema_ref,
    ]
    assert "filesystemContextPolicy" not in user_payload


def test_hidden_planner_file_tool_loop_is_disabled_by_default() -> None:
    assert Settings.model_fields["agent_planner_tool_rounds"].default == 0
    assert Settings.model_fields["planner_filesystem_context_mode"].default == "off"

    # Older dependency-injection stubs may not expose the new flag. Missing
    # configuration must not silently restore the pre-DeepAgent tool loop.
    planner = object.__new__(QueryGraphPlanner)
    planner.settings = type("LegacySettings", (), {})()
    assert planner._filesystem_context_mode() == "off"


def test_core_read_evidence_gate_requires_exact_metric_and_grouping_schema() -> None:
    planner = QueryGraphPlanner(type("NoModel", (), {"configured": False})())
    understanding = {
        "anchorMetric": {
            "metricRef": "measure_0",
            "ownerTable": "fact_0",
            "groupByColumn": "event_day",
        },
        "supportMetrics": [],
        "scopeConstraints": [],
        "semanticQuery": empty_semantic_query(),
    }
    detail_only = {
        "coreManagedFilesystem": True,
        "coreSemanticEvidence": [
            {
                "refId": "semantic:DOMAIN_0:fact_0:detail",
                "kind": "TABLE_DETAIL",
            }
        ],
    }

    errors = planner._core_semantic_evidence_errors(understanding, detail_only)

    assert "read exact metric definition for fact_0.measure_0" in errors
    assert "read exact column or schema for fact_0.event_day" in errors

    complete = {
        "coreManagedFilesystem": True,
        "coreSemanticEvidence": [
            *detail_only["coreSemanticEvidence"],
            {"refId": "semantic:DOMAIN_0:fact_0:metric:measure_0", "kind": "METRIC"},
            {"refId": "semantic:DOMAIN_0:fact_0:schema", "kind": "SCHEMA"},
        ],
    }
    assert planner._core_semantic_evidence_errors(understanding, complete) == []


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


def test_semantic_candidate_prompt_hints_keep_refs_without_copying_asset_definitions() -> None:
    hints = {
        "authority": "advisory_only",
        "candidateRefs": [f"semantic:DOMAIN:fact:metric:measure_{index}" for index in range(14)],
        "metricPhrases": ["Requested measure"],
        "provenance": "rag_and_semantic_search",
        "policy": "LONG_POLICY" * 500,
        "candidates": [
            {
                "id": f"M{index}",
                "ref": f"semantic:DOMAIN:fact:metric:measure_{index}",
                "kind": "METRIC",
                "name": f"Measure {index}",
                "metricKey": f"measure_{index}",
                "table": "fact",
                "topic": "DOMAIN",
                "matched": "Requested measure",
                "matchType": "partial_label",
                "confidence": 0.8,
                "readableRefs": [
                    {
                        "ref": f"semantic:DOMAIN:fact:metric:measure_{index}",
                        "type": "metric",
                        "label": f"Measure {index}",
                    }
                ],
                "aliases": ["OVERSIZED_ALIAS" * 300],
                "description": "OVERSIZED_DESCRIPTION" * 300,
                "temporalVariants": [{"oversized": "OVERSIZED_TEMPORAL" * 300}],
                "selectionGuidance": "OVERSIZED_GUIDANCE" * 300,
            }
            for index in range(14)
        ],
    }

    compact = compact_semantic_candidate_hints_for_prompt(hints, budget_level=1)

    assert compact["candidateRefs"] == hints["candidateRefs"]
    assert "candidates" not in compact
    serialized = json.dumps(compact, ensure_ascii=False)
    assert "OVERSIZED_" not in serialized
    assert compact["policy"] == "search_coordinates_only; choose from tableManifest after semantic_read"

    uncompressed_index = compact_semantic_candidate_hints_for_prompt(hints, budget_level=0)
    assert len(uncompressed_index["candidates"]) == 14
    assert "OVERSIZED_" not in json.dumps(uncompressed_index, ensure_ascii=False)


def test_tool_loop_prompt_compacts_advisory_candidates_under_total_budget() -> None:
    class NoProviderModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

    settings = get_settings().model_copy(update={"agent_planner_prompt_budget_chars": 30_000})
    planner = QueryGraphPlanner(NoProviderModel(), settings=settings)
    hints = {
        "authority": "advisory_only",
        "candidateRefs": [f"semantic:DOMAIN:fact_0:metric:measure_{index}" for index in range(14)],
        "metricPhrases": ["Requested measure"],
        "provenance": "rag_and_semantic_search",
        "candidates": [
            {
                "id": f"M{index}",
                "ref": f"semantic:DOMAIN:fact_0:metric:measure_{index}",
                "kind": "METRIC",
                "name": f"Measure {index}",
                "metricKey": f"measure_{index}",
                "table": "fact_0",
                "topic": "DOMAIN",
                "matched": "Requested measure",
                "matchType": "partial_label",
                "confidence": 0.8,
                "aliases": ["LARGE_ALIAS" * 500],
                "description": "LARGE_DESCRIPTION" * 500,
            }
            for index in range(14)
        ],
    }

    payload = planner._llm_understand(
        "Requested measure by subject",
        planner_pack(),
        [],
        [],
        planner_context={"semanticCandidateHints": hints},
        use_tool_loop=True,
        filesystem_context_entry="adaptive",
    )

    assert payload.get("_plannerContextOverBudget") is not True
    assert payload["_promptStats"]["totalChars"] <= settings.agent_planner_prompt_budget_chars
    assert payload["_promptStats"]["toolSchemaChars"] > 0


def test_metric_tool_compaction_never_drops_the_execution_contract() -> None:
    metric = {
        "metricKey": "opaque_metric",
        "businessName": "Opaque metric with a deliberately long display name",
        "formula": "SUM(numerator) / NULLIF(SUM(denominator), 0)",
        "sourceColumns": ["numerator", "denominator"],
        "metricGrain": "tenant_event",
        "aggregationPolicy": "ratio_of_sums",
        "applicableTimeGrain": "period",
        "timeColumn": "event_day",
        "timeSemantics": {
            "selectionPolicy": "period_window",
            "asOfPolicy": "latest_available_partition",
            "missingDataPolicy": "disclose_unknown",
            "zeroValuePolicy": "preserve_observed_zero",
        },
        "description": "x" * 5000,
    }
    result = compact_tool_result_for_prompt(
        {
            "success": True,
            "refId": "semantic:domain:fact:metric:opaque_metric",
            "kind": "METRIC",
            "content": json.dumps({"metric": metric}),
        },
        220,
    )

    compact = result["metric"]
    assert result["executionContractPreserved"] is True
    for key in [
        "formula",
        "sourceColumns",
        "metricGrain",
        "aggregationPolicy",
        "applicableTimeGrain",
        "timeColumn",
        "timeSemantics",
    ]:
        assert compact[key] == metric[key]


def test_adaptive_planner_reads_exact_ref_before_emitting_understanding() -> None:
    detail_ref = "semantic:DOMAIN_0:fact_0:detail"
    selected_ref = "semantic:DOMAIN_0:fact_0:metric:measure_0"
    schema_ref = "semantic:DOMAIN_0:fact_0:schema"

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            assert ref_id in {detail_ref, selected_ref, schema_ref}
            if ref_id == detail_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "TABLE_DETAIL",
                    "content": json.dumps(
                        {
                            "table": "fact_0",
                            "metricsRefId": "semantic:DOMAIN_0:fact_0:metrics",
                            "schemaRefId": schema_ref,
                        }
                    ),
                }
            if ref_id == schema_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "SCHEMA",
                    "content": '{"columns":[{"name":"event_day","type":"date"}]}',
                }
            return {
                "success": True,
                "refId": ref_id,
                "kind": "METRIC",
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
                            "id": "read_selected_table_detail",
                            "name": "semantic_read",
                            "args": {"refId": detail_ref, "maxChars": 1000},
                        }
                    ],
                }
            if len(self.payloads) == 3:
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
            if len(self.payloads) == 4:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "read_selected_schema",
                            "name": "semantic_read",
                            "args": {"refId": schema_ref, "maxChars": 1000},
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
                                    "semanticQuery": empty_semantic_query(),
                                    "timeWindowDays": 11,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20000,
            "agent_planner_tool_rounds": 5,
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

    assert payload["status"] == "UNDERSTOOD", (
        payload.get("reason"), payload.get("_promptStats"), len(model.payloads),
        model.payloads[-1].get("plannerToolResults") if model.payloads else [],
    )
    assert set(payload["_plannerLoadedRefs"]) == {detail_ref, selected_ref, schema_ref}
    assert set(payload["_plannerSemanticReadRefs"]) == {detail_ref, selected_ref, schema_ref}
    assert model.payloads[0]["filesystemContextPolicy"]["mustReadBeforeEmit"] is True
    assert "formula" not in nested_keys(model.payloads[0]["semanticCatalog"])
    assert any(
        "SUM(value_0)" in json.dumps(item, ensure_ascii=False)
        for item in model.payloads[-1]["plannerToolResults"]
    ), model.prompt_sizes
    assert max(model.prompt_sizes) <= settings.agent_planner_prompt_budget_chars


def test_manifest_read_alone_cannot_authorize_final_table_selection() -> None:
    manifest_ref = "semantic:DOMAIN_0:manifest"
    detail_ref = "semantic:DOMAIN_0:fact_0:detail"
    metric_ref = "semantic:DOMAIN_0:fact_0:metric:measure_0"
    schema_ref = "semantic:DOMAIN_0:fact_0:schema"

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            if ref_id == manifest_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "TOPIC_MANIFEST",
                    "content": '{"tables":[{"tableName":"fact_0"}]}',
                }
            if ref_id == detail_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "TABLE_DETAIL",
                    "content": json.dumps(
                        {
                            "table": "fact_0",
                            "metricsRefId": "semantic:DOMAIN_0:fact_0:metrics",
                            "schemaRefId": schema_ref,
                        }
                    ),
                }
            if ref_id == schema_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "SCHEMA",
                    "content": '{"columns":[{"name":"event_day","type":"date"}]}',
                }
            assert ref_id == metric_ref
            return {
                "success": True,
                "refId": ref_id,
                "kind": "METRIC",
                "content": '{"metricKey":"measure_0","formula":"SUM(value_0)"}',
            }

        def ls(self, **kwargs):
            return []

        def grep(self, **kwargs):
            return []

    class ManifestThenMetricModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, **kwargs):
            payload = json.loads(user_prompt)
            self.payloads.append(payload)
            round_number = len(self.payloads)
            if round_number == 1:
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
            if round_number == 2:
                return {
                    "content": "",
                    "toolCalls": [{"id": "read_manifest", "name": "semantic_read", "args": {"refId": manifest_ref}}],
                }
            if round_number == 3:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "emit_too_early",
                            "name": "emit_question_understanding",
                            "args": {
                                "status": "UNDERSTOOD",
                                "questionUnderstanding": {
                                    "analysisGrain": "metric",
                                    "analysisIntent": "none",
                                    "requiresExplanation": False,
                                    "requiredEvidenceIntents": [],
                                    "rankingObjective": {},
                                    "requestedMeasures": [],
                                    "filters": [],
                                    "semanticQuery": empty_semantic_query(),
                                    "timeWindowDays": 7,
                                },
                            },
                        }
                    ],
                }
            if round_number == 4:
                assert any(
                    item.get("errorType") == "SEMANTIC_EVIDENCE_READ_REQUIRED"
                    for item in payload.get("plannerToolResults") or []
                )
                return {
                    "content": "",
                    "toolCalls": [{"id": "read_detail", "name": "semantic_read", "args": {"refId": detail_ref}}],
                }
            if round_number == 5:
                return {
                    "content": "",
                    "toolCalls": [{"id": "read_metric", "name": "semantic_read", "args": {"refId": metric_ref}}],
                }
            if round_number == 6:
                return {
                    "content": "",
                    "toolCalls": [{"id": "read_schema", "name": "semantic_read", "args": {"refId": schema_ref}}],
                }
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "emit_after_detail",
                        "name": "emit_question_understanding",
                        "args": {
                            "status": "UNDERSTOOD",
                            "questionUnderstanding": {
                                "analysisGrain": "metric",
                                "analysisIntent": "none",
                                "requiresExplanation": False,
                                "requiredEvidenceIntents": [],
                                "rankingObjective": {
                                    "metricRef": "measure_0",
                                    "ownerTable": "fact_0",
                                    "sourcePhrase": "Requested zero",
                                    "resultMode": "metric",
                                },
                                "requestedMeasures": [],
                                "filters": [],
                                "semanticQuery": empty_semantic_query(),
                                "timeWindowDays": 7,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20_000,
            "agent_planner_tool_rounds": 7,
            "agent_deferred_tool_schema_enabled": True,
        }
    )
    model = ManifestThenMetricModel()
    planner = QueryGraphPlanner(model, semantic_catalog=SemanticCatalog(), settings=settings)

    payload = planner._llm_understand(
        "opaque",
        planner_pack(),
        [],
        [],
        use_tool_loop=True,
        filesystem_context_entry="adaptive",
    )

    assert payload["status"] == "UNDERSTOOD", (
        payload.get("reason"), payload.get("_promptStats"), len(model.payloads),
        model.payloads[-1].get("plannerToolResults") if model.payloads else [],
    )
    assert set(payload["_plannerSemanticReadRefs"]) == {manifest_ref, detail_ref, metric_ref, schema_ref}
    assert set(payload["_plannerSemanticDetailReadRefs"]) == {detail_ref, metric_ref, schema_ref}


def test_active_planner_round_keeps_multiple_exact_metric_reads_under_budget() -> None:
    selected_refs = [
        f"semantic:DOMAIN_{index % 3}:fact_{index % 3}:metric:measure_{index}"
        for index in range(5)
    ]
    detail_refs = [f"semantic:DOMAIN_{index}:fact_{index}:detail" for index in range(3)]
    schema_refs = [f"semantic:DOMAIN_{index}:fact_{index}:schema" for index in range(3)]

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            if ref_id in detail_refs:
                table_index = int(ref_id.split(":", 3)[2].rsplit("_", 1)[-1])
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "TABLE_DETAIL",
                    "topic": f"DOMAIN_{table_index}",
                    "table": f"fact_{table_index}",
                    "content": json.dumps(
                        {
                            "table": f"fact_{table_index}",
                            "metricsRefId": f"semantic:DOMAIN_{table_index}:fact_{table_index}:metrics",
                            "schemaRefId": f"semantic:DOMAIN_{table_index}:fact_{table_index}:schema",
                        }
                    ),
                }
            if ref_id in schema_refs:
                table_index = int(ref_id.split(":", 3)[2].rsplit("_", 1)[-1])
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "SCHEMA",
                    "topic": f"DOMAIN_{table_index}",
                    "table": f"fact_{table_index}",
                    "content": '{"columns":[{"name":"event_day","type":"date"}]}',
                }
            assert ref_id in selected_refs
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
                            "id": f"read_detail_{index}",
                            "name": "semantic_read",
                            "args": {"refId": ref_id, "maxChars": 20_000},
                        }
                        for index, ref_id in enumerate(detail_refs)
                    ],
                }
            if len(self.payloads) == 3:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": f"read_metric_{index}",
                            "name": "semantic_read",
                            "args": {"refId": ref_id, "maxChars": 20_000},
                        }
                        for index, ref_id in enumerate(selected_refs)
                    ],
                }
            if len(self.payloads) == 4:
                results = payload["plannerToolResults"]
                assert "MUST_BE_COMPACTED" not in json.dumps(results, ensure_ascii=False)
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": f"read_schema_{index}",
                            "name": "semantic_read",
                            "args": {"refId": ref_id, "maxChars": 20_000},
                        }
                        for index, ref_id in enumerate(schema_refs)
                    ],
                }
            results = payload["plannerToolResults"]
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
                                    "resultMode": "metric",
                                },
                                "requestedMeasures": [
                                    {
                                        "metricRef": f"measure_{index}",
                                        "ownerTable": f"fact_{index % 3}",
                                        "sourcePhrase": f"Requested {index}",
                                        "groupByColumn": "event_day",
                                        "resultMode": "metric",
                                    }
                                    for index in range(1, 5)
                                ],
                                    "filters": [],
                                    "semanticQuery": empty_semantic_query(),
                                    "timeWindowDays": 11,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20_000,
            "agent_planner_tool_rounds": 5,
            "agent_deferred_tool_schema_enabled": True,
            "tool_rate_limit_enabled": False,
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

    assert payload["status"] == "UNDERSTOOD", (
        payload.get("reason"),
        payload.get("_plannerSemanticReadRefs"),
        [
            item
            for item in payload.get("_plannerToolResults") or []
            if item.get("errorType") == "SEMANTIC_EVIDENCE_READ_REQUIRED"
        ],
    )
    expected_read_refs = set(detail_refs + selected_refs + schema_refs)
    assert set(payload["_plannerLoadedRefs"]) == expected_read_refs
    assert set(payload["_plannerSemanticReadRefs"]) == expected_read_refs
    assert max(model.prompt_sizes) <= settings.agent_planner_prompt_budget_chars
    assert payload["_promptStats"]["toolRounds"][-1]["compaction"] == "active_semantic_results"


def test_terminal_planner_round_forces_validated_emit_and_retries_invalid_structure() -> None:
    detail_ref = "semantic:DOMAIN_0:fact_0:detail"
    selected_ref = "semantic:DOMAIN_0:fact_0:metric:measure_0"
    schema_ref = "semantic:DOMAIN_0:fact_0:schema"

    class SemanticCatalog:
        def read(self, ref_id="", path="", max_chars=0, offset=0):
            if ref_id == detail_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "TABLE_DETAIL",
                    "topic": "DOMAIN_0",
                    "table": "fact_0",
                    "content": json.dumps(
                        {
                            "table": "fact_0",
                            "metricsRefId": "semantic:DOMAIN_0:fact_0:metrics",
                            "schemaRefId": schema_ref,
                        }
                    ),
                }
            if ref_id == schema_ref:
                return {
                    "success": True,
                    "refId": ref_id,
                    "kind": "SCHEMA",
                    "topic": "DOMAIN_0",
                    "table": "fact_0",
                    "content": '{"columns":[{"name":"event_day","type":"date"}]}',
                }
            assert ref_id == selected_ref
            return {
                "success": True,
                "refId": ref_id,
                "kind": "METRIC",
                "topic": "DOMAIN_0",
                "table": "fact_0",
                "content": json.dumps(
                    {
                        "metric": {
                            "metricKey": "measure_0",
                            "businessName": "Measure 0",
                            "formula": "SUM(value_0)",
                            "sourceColumns": ["value_0"],
                        }
                    }
                ),
            }

        def ls(self, **kwargs):
            return []

        def grep(self, **kwargs):
            return []

    class InvalidThenValidModel:
        configured = True
        last_error = ""
        error_events: list[object] = []

        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []
            self.tool_choices: list[str] = []

        def tool_chat(self, system_prompt, user_prompt, tools, fallback=None, tool_choice=None, **kwargs):
            self.payloads.append(json.loads(user_prompt))
            self.tool_choices.append(str(tool_choice or ""))
            round_number = len(self.payloads)
            if round_number == 1:
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
            if round_number == 2:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "read_selected_detail",
                            "name": "semantic_read",
                            "args": {"refId": detail_ref, "maxChars": 1000},
                        }
                    ],
                }
            if round_number == 3:
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
            if round_number == 4:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "read_selected_schema",
                            "name": "semantic_read",
                            "args": {"refId": schema_ref, "maxChars": 1000},
                        }
                    ],
                }
            if round_number == 5:
                return {
                    "content": "",
                    "toolCalls": [
                        {
                            "id": "invalid_emit",
                            "name": "emit_question_understanding",
                            "args": {"status": "UNDERSTOOD", "reason": "missing understanding"},
                        }
                    ],
                }
            return {
                "content": "",
                "toolCalls": [
                    {
                        "id": "repaired_emit",
                        "name": "emit_question_understanding",
                        "args": {
                            "status": "UNDERSTOOD",
                            "reason": "repaired against validator feedback",
                            "questionUnderstanding": {
                                "analysisGrain": "time",
                                "analysisIntent": "trend_check",
                                "requiresExplanation": False,
                                "requiredEvidenceIntents": [],
                                "anchorMetric": {
                                    "metricRef": "measure_0",
                                    "ownerTable": "fact_0",
                                    "sourcePhrase": "Requested zero",
                                    "groupByColumn": "event_day",
                                },
                                    "supportMetrics": [],
                                    "semanticQuery": empty_semantic_query(),
                                    "timeWindowDays": 11,
                            },
                        },
                    }
                ],
            }

    settings = get_settings().model_copy(
        update={
            "agent_planner_prompt_budget_chars": 20_000,
            "agent_planner_tool_rounds": 5,
            "agent_planner_invalid_output_retries": 1,
            "agent_deferred_tool_schema_enabled": True,
        }
    )
    model = InvalidThenValidModel()
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
    assert payload["questionUnderstanding"]["rankingObjective"]["metricRef"] == "measure_0"
    assert len(payload["_plannerInvalidOutputAttempts"]) == 1
    assert model.tool_choices == [
        "",
        "",
        "",
        "",
        "emit_question_understanding",
        "emit_question_understanding",
    ]
    assert any(
        item.get("errorType") == "INVALID_STRUCTURED_OUTPUT"
        for item in model.payloads[-1]["plannerToolResults"]
    )
