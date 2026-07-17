from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import merchant_ai.services.deep_agent_runtime as runtime_module
from merchant_ai.config import get_settings
from merchant_ai.graph.policy import AgentActionRegistry
from merchant_ai.models import RecallBundle, RecallItem, RouteSlots, RouteTimeWindow
from merchant_ai.services.deep_agent_runtime import (
    DeepAgentWorkflowAdapter,
    ReadOnlySemanticBackend,
    _DianaLeadSession,
    _ResultSink,
)


class _TopicAssetsStub:
    settings = SimpleNamespace()

    @staticmethod
    def resolve_topic_category(topic: str) -> str:
        return topic

    @staticmethod
    def load_manifest(topic: str) -> list[dict[str, Any]]:
        assert topic == "OPERATIONS"
        return [
            {
                "tableName": "ads_merchant_profile",
                "tableComment": "Merchant-day order and refund aggregates",
                # L1/L2 material deliberately exists in the source asset. Topic
                # entry disclosure must still remain L0-only.
                "metrics": [
                    {"metricKey": "order_cnt_1d", "formula": "SUM(order_cnt_1d)"},
                    {"metricKey": "refund_amt_1d", "formula": "SUM(refund_amt_1d)"},
                ],
                "columns": [{"name": "merchant_id"}, {"name": "pt"}],
                "schema": {"pt": "date"},
            }
        ]


def _adapter() -> DeepAgentWorkflowAdapter:
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    adapter.domain_workflow = SimpleNamespace(
        policy=SimpleNamespace(
            max_main_actions=16,
            registry=AgentActionRegistry(),
        )
    )
    return adapter


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def _topic_scoped_recall_bundle() -> RecallBundle:
    return RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:OPERATIONS:ads_merchant_profile:metric:order_cnt_1d",
                title="Order count",
                content=(
                    "Daily governed order count. Exact formula and source columns "
                    "must be opened through read_file before compilation."
                ),
                source_type="SEMANTIC_METRIC",
                topic="OPERATIONS",
                table="ads_merchant_profile",
                fusion_score=0.97,
                metadata={
                    "semanticRefId": "semantic:OPERATIONS:ads_merchant_profile:metric:order_cnt_1d",
                    "semanticPath": (
                        "topics/OPERATIONS/tables/ads_merchant_profile/metrics/order_cnt_1d.json"
                    ),
                    "semanticKind": "METRIC",
                    # Candidate disclosure is navigation-only; these fields must
                    # not leak into the initial Core observation.
                    "formula": "SUM(order_cnt_1d)",
                    "sourceColumns": ["order_cnt_1d"],
                },
            ),
            RecallItem(
                doc_id="semantic:RETURNS:fact_refund:metric:refund_amt",
                title="Out-of-scope refund amount",
                content="A semantically similar hit from another Topic.",
                source_type="SEMANTIC_METRIC",
                topic="RETURNS",
                table="fact_refund",
                fusion_score=0.99,
                metadata={
                    "semanticRefId": "semantic:RETURNS:fact_refund:metric:refund_amt",
                    "semanticPath": "topics/RETURNS/tables/fact_refund/metrics/refund_amt.json",
                    "semanticKind": "METRIC",
                },
            ),
        ],
        top_score=0.99,
    )


def test_topic_entry_observation_discloses_l0_and_thin_recall_candidates_together() -> None:
    """Target: first post-Topic observation is useful without becoming an asset dump."""

    adapter = _adapter()
    session = _DianaLeadSession(
        state={
            "react_round": 1,
            "topic_workspace": {
                "mode": "topic_workspace",
                "topics": ["OPERATIONS"],
            },
            "recall_bundle": _topic_scoped_recall_bundle(),
        },
        sink=_ResultSink(),
        observation={"summary": "Topic selected and fresh recall completed"},
    )

    payload = adapter._turn_payload(session)

    # The two disclosures must be present in the same first observation after
    # Topic convergence: L0 is the browse map; recallCandidates are thin search
    # hints. Neither is trusted execution evidence yet.
    assert payload["semanticDisclosure"]["layer"] == "L0"
    assert payload["tableManifest"]["tables"] == [
        {
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "title": "Merchant-day order and refund aggregates",
            "detailRefId": "semantic:OPERATIONS:ads_merchant_profile:detail",
            "detailPath": "topics/OPERATIONS/tables/ads_merchant_profile/detail.json",
        }
    ]
    assert [item["refId"] for item in payload["recallCandidates"]] == [
        "semantic:OPERATIONS:ads_merchant_profile:metric:order_cnt_1d"
    ]
    candidate = payload["recallCandidates"][0]
    assert {
        "refId",
        "path",
        "kind",
        "topic",
        "table",
        "snippet",
    }.issubset(candidate)
    assert candidate["path"].endswith("/metrics/order_cnt_1d.json")
    assert candidate["kind"] == "METRIC"
    assert candidate["topic"] == "OPERATIONS"
    assert candidate["table"] == "ads_merchant_profile"
    assert payload["coreSemanticEvidence"]["readCount"] == 0

    forbidden = {
        "content",
        "formula",
        "sourceColumns",
        "metrics",
        "columns",
        "schema",
        "rules",
        "relationships",
    }
    assert not (_nested_keys(payload["recallCandidates"]) & forbidden)


def test_ranking_without_explicit_time_clarifies_before_initial_recall() -> None:
    state = {
        "route_slots": RouteSlots(
            time_window=RouteTimeWindow(days=0, raw=""),
            analysis_signals=["typed_ranking_span"],
        )
    }

    assert DeepAgentWorkflowAdapter._ranking_requires_time_clarification(state) is True

    state["route_slots"].time_window = RouteTimeWindow(days=30, raw="最近30天")
    assert DeepAgentWorkflowAdapter._ranking_requires_time_clarification(state) is False


def test_deepagent_core_action_catalog_hides_legacy_asset_and_planner_pipeline_actions() -> None:
    """Target: legacy compatibility actions never become Core ReAct affordances."""

    domain = _GroundedToolDomainStub()
    domain.policy.decide = lambda _state: SimpleNamespace(
        available_actions=[
            "fast_understand",
            "retrieve_knowledge",
            "compact_assets",
            "plan_graph",
            "query_metric",
            "reflect_plan",
            "validate_graph",
            "repair_graph",
        ]
    )
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    session = _DianaLeadSession(
        state={"react_round": 1, "topic_workspace": {"topics": ["OPERATIONS"]}},
        sink=_ResultSink(),
        observation={"summary": "Topic is ready"},
        table_manifest_disclosed=True,
    )

    payload = adapter._prepare_turn(session)
    exposed = [item["actionId"] for item in payload["actionCatalog"]]

    assert session.available_actions == ()
    assert "fast_understand" not in exposed
    assert "retrieve_knowledge" not in exposed
    assert "compact_assets" not in exposed
    assert "plan_graph" not in exposed
    assert "query_metric" not in exposed
    assert "reflect_plan" not in exposed
    assert "validate_graph" not in exposed
    assert "repair_graph" not in exposed
    assert payload["planningAuthority"]["mode"] == "grounded_query_contract"
    assert payload["planningAuthority"]["legacyPlanningDisabled"] is True


def test_deepagent_registers_grounded_contract_tools_without_meta_action_schema(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Target: grounding/compilation are first-class tools, not action-id arguments."""

    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_module, "create_deep_agent", fake_create_deep_agent)
    settings = get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "ops_path": str(tmp_path / "ops" / "runtime.json"),
        }
    )
    domain = SimpleNamespace(
        settings=settings,
        checkpoint_manager=SimpleNamespace(saver=lambda: object()),
        graph=object(),
        policy=SimpleNamespace(max_main_actions=16, registry=AgentActionRegistry()),
    )
    adapter = DeepAgentWorkflowAdapter(
        domain,
        SimpleNamespace(configured=True, chat_model=lambda: object()),
        SimpleNamespace(topic_assets=_TopicAssetsStub()),
    )

    tools = {item.name: item for item in captured["tools"]}
    assert "inspect_diana_state" in tools
    assert "retrieve_knowledge" in tools
    assert "commit_grounded_query_contract" in tools
    assert "compile_grounded_query" in tools

    for name in ("commit_grounded_query_contract", "compile_grounded_query"):
        schema = tools[name].tool_call_schema.model_json_schema()
        serialized = json.dumps(schema, ensure_ascii=False).lower()
        assert "action_id" not in schema.get("properties", {})
        assert "planning_assets_compacted" not in serialized
        assert "runtime" not in serialized
        assert "session" not in serialized

    # Keep a live reference so facade assignment cannot make the constructed
    # adapter disappear under aggressive test optimizers/type checkers.
    assert adapter.deep_agent_graph is not None


class _GroundedToolRegistryStub:
    @staticmethod
    def get(action_id: str) -> Any:
        return SimpleNamespace(
            id=action_id,
            node=action_id,
            agent="Runtime",
            description=action_id,
        )

    @staticmethod
    def actions(action_ids: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
        return [{"id": action_id} for action_id in action_ids]


class _GroundedToolMiddlewareStub:
    @staticmethod
    def after_action(state: dict[str, Any]) -> dict[str, Any]:
        return state

    @staticmethod
    def before_policy(state: dict[str, Any]) -> dict[str, Any]:
        return state


class _GroundedToolDomainStub:
    """Minimal domain seam proving the native tools hand off grounded state."""

    def __init__(self) -> None:
        self.policy = SimpleNamespace(
            max_main_actions=16,
            registry=_GroundedToolRegistryStub(),
            decide=lambda _state: SimpleNamespace(available_actions=[]),
        )
        self.middleware_chain = _GroundedToolMiddlewareStub()
        self.commit_input: dict[str, Any] = {}
        self.compile_input: dict[str, Any] = {}

    @staticmethod
    def materialize_plan_clarification(state: dict[str, Any]) -> None:
        del state

    @staticmethod
    def refresh_execution_tier_policy(state: dict[str, Any]) -> None:
        del state

    @staticmethod
    def main_agent_observation(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "grounded query compiled" if state.get("plan") else "grounded contract committed"
        }

    @staticmethod
    def build_lead_decision_context(
        state: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        del state, observation
        return {}

    def commit_grounded_query_contract(
        self,
        state: dict[str, Any],
        binding_hints: dict[str, Any],
    ) -> dict[str, Any]:
        self.commit_input = {
            "core_semantic_evidence": list(state.get("core_semantic_evidence") or []),
            "binding_hints": dict(binding_hints),
        }
        read_refs = {
            str(item.get("refId") or "")
            for item in self.commit_input["core_semantic_evidence"]
        }
        selected_refs = {
            *binding_hints.get("tableRefs", []),
            *binding_hints.get("metricRefs", []),
        }
        assert selected_refs <= read_refs
        state["grounded_query_contract"] = {
            "status": "READY",
            "question": state["question"],
            "primaryTable": "ads_merchant_profile",
            "evidenceRefs": sorted(selected_refs),
        }
        state["grounded_asset_pack"] = {
            "source": "successful_core_read_file_calls",
            "evidenceRefs": sorted(selected_refs),
        }
        return state

    def compile_grounded_query(self, state: dict[str, Any]) -> dict[str, Any]:
        # This is the architecture assertion: the native compile seam is ready
        # without the legacy broad PlanningAssetPack phase.
        assert state.get("planning_assets_compacted") is False
        assert "planning_asset_pack" not in state
        self.compile_input = {
            "grounded_query_contract": state["grounded_query_contract"],
            "grounded_asset_pack": state["grounded_asset_pack"],
            "core_semantic_evidence": list(state.get("core_semantic_evidence") or []),
        }
        state["plan"] = {
            "compiledBy": "compile_grounded_query",
            "contractEvidenceRefs": list(
                state["grounded_query_contract"].get("evidenceRefs") or []
            ),
        }
        return state


def test_native_grounded_compile_uses_committed_contract_and_core_reads_without_compaction() -> None:
    """Target: QueryGraph compilation bypasses plannerContext/PlanningAssetPack."""

    detail_ref = "semantic:OPERATIONS:ads_merchant_profile:detail"
    metric_ref = "semantic:OPERATIONS:ads_merchant_profile:metric:order_cnt_1d"
    evidence = [
        {
            "refId": detail_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/detail.json",
            "kind": "TABLE_DETAIL",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": '{"tableName":"ads_merchant_profile"}',
            "contentHash": "detail-hash",
        },
        {
            "refId": metric_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
            "kind": "METRIC",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": '{"metricKey":"order_cnt_1d","formula":"SUM(order_cnt_1d)"}',
            "contentHash": "metric-hash",
        },
    ]
    domain = _GroundedToolDomainStub()
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    session = _DianaLeadSession(
        state={
            "question": "最近30天订单数",
            "react_round": 1,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["OPERATIONS"]},
            "planning_assets_compacted": False,
        },
        sink=_ResultSink(),
        table_manifest_disclosed=True,
        core_semantic_evidence=evidence,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(session=session))

    commit_tool = adapter._build_commit_grounded_contract_tool()
    commit_tool.func(
        table_refs=[detail_ref],
        metric_refs=[metric_ref],
        dimension_refs=[],
        group_by_ref="",
        label_refs=[],
        relationship_refs=[],
        ranking_order="",
        limit=0,
        analysis_mode="metric_summary",
        time_expression="最近30天",
        reason="same-table governed metric",
        runtime=runtime,
    )

    assert [item["refId"] for item in domain.commit_input["core_semantic_evidence"]] == [
        detail_ref,
        metric_ref,
    ]
    assert session.state["grounded_query_contract"]["evidenceRefs"] == [detail_ref, metric_ref]
    assert session.state["grounded_asset_pack"]["source"] == "successful_core_read_file_calls"
    assert session.state["planning_assets_compacted"] is False

    compile_tool = adapter._build_compile_grounded_query_tool()
    raw_result = compile_tool.func(
        reason="compile the committed grounded contract",
        runtime=runtime,
    )
    result = json.loads(raw_result)

    assert result["status"] == "ACTION_REQUIRED"
    assert domain.compile_input["grounded_query_contract"] is session.state["grounded_query_contract"]
    assert domain.compile_input["grounded_asset_pack"] is session.state["grounded_asset_pack"]
    assert [
        item["refId"] for item in domain.compile_input["core_semantic_evidence"]
    ] == [detail_ref, metric_ref]
    assert session.state["plan"]["compiledBy"] == "compile_grounded_query"
    assert session.state["planning_assets_compacted"] is False


def test_grounding_commit_rejects_partial_exact_read_with_executable_next_path() -> None:
    detail_ref = "semantic:OPERATIONS:ads_merchant_profile:detail"
    metric_ref = "semantic:OPERATIONS:ads_merchant_profile:metric:order_cnt_1d"
    evidence = [
        {
            "refId": detail_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/detail.json",
            "kind": "TABLE_DETAIL",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": '{"tableName":"ads_merchant_profile"}',
            "contentHash": "detail-hash",
            "contentComplete": True,
        },
        {
            "refId": metric_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
            "kind": "METRIC",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": "{\n",
            "contentHash": "partial-hash",
            "contentComplete": False,
        },
    ]
    domain = _GroundedToolDomainStub()
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    session = _DianaLeadSession(
        state={
            "question": "最近30天订单数",
            "react_round": 1,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["OPERATIONS"]},
        },
        sink=_ResultSink(),
        table_manifest_disclosed=True,
        core_semantic_evidence=evidence,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(session=session))

    raw = adapter._build_commit_grounded_contract_tool().func(
        table_refs=[detail_ref],
        metric_refs=[metric_ref],
        dimension_refs=[],
        group_by_ref="",
        label_refs={},
        relationship_refs=[],
        ranking_order="",
        limit=0,
        analysis_mode="metric_summary",
        time_expression="最近30天",
        reason="partial read must not bind",
        runtime=runtime,
    )
    result = json.loads(raw)

    required = result["groundingReadRequired"]["requiredReads"]
    assert required == [
        {
            "binding": "metricRefs",
            "submittedRef": metric_ref,
            "expectedKind": "METRIC",
            "observedKind": "METRIC",
            "readPath": "/knowledge/topics/OPERATIONS/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
            "indexPath": "/knowledge/topics/OPERATIONS/tables/ads_merchant_profile/metrics/index.json",
            "instruction": "Read the exact definition file",
        }
    ]
    assert domain.commit_input == {}


def test_revise_bindings_exposes_compatible_read_field_before_topic_expansion() -> None:
    rejected_detail = "semantic:OPERATIONS:ads_merchant_profile:detail"
    rejected_metric = "semantic:OPERATIONS:ads_merchant_profile:metric:daily_user_count"
    buyer_ref = "semantic:OPERATIONS:order_detail:field:buyer_id"
    rejection = {
        "fingerprint": "rejected-1",
        "code": "TABLE_INSUFFICIENT",
        "topic": "OPERATIONS",
        "table": "ads_merchant_profile",
        "refIds": [rejected_detail, rejected_metric],
        "requiredCapability": {
            "operation": "COUNT_DISTINCT",
            "entityRole": "BUYER",
            "requiredFieldRole": "KEY",
        },
    }
    buyer_content = json.dumps(
        {
            "topic": "OPERATIONS",
            "tableName": "order_detail",
            "definition": {
                "columnName": "buyer_id",
                "role": "KEY",
                "calculationSemantics": {
                    "semanticEntityRole": "BUYER",
                    "allowedAggregations": ["COUNT_DISTINCT"],
                },
            },
        }
    )
    adapter = _adapter()
    session = _DianaLeadSession(
        state={
            "question": "period unique buyers",
            "react_round": 2,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["OPERATIONS"]},
            "grounded_query_contract": {
                "status": "REVISE_BINDINGS",
                "ready": False,
                "unresolvedGaps": [
                    {
                        "code": "TABLE_INSUFFICIENT",
                        "requiredCapability": rejection["requiredCapability"],
                    }
                ],
                "rejectedBindings": [rejection],
            },
            "grounded_rejected_bindings": [rejection],
        },
        sink=_ResultSink(),
        table_manifest_disclosed=True,
        core_semantic_evidence=[
            {
                "refId": buyer_ref,
                "path": "topics/OPERATIONS/tables/order_detail/columns/buyer_id.json",
                "kind": "COLUMN",
                "topic": "OPERATIONS",
                "table": "order_detail",
                "contentSnippet": buyer_content,
                "contentHash": "buyer-hash",
                "contentComplete": True,
            }
        ],
    )

    payload = adapter._turn_payload(session)

    assert payload["bindingRevision"]["searchRequired"] is False
    assert payload["bindingRevision"]["compatibleReadBindings"] == [
        {
            "refId": buyer_ref,
            "path": "topics/OPERATIONS/tables/order_detail/columns/buyer_id.json",
            "kind": "COLUMN",
            "topic": "OPERATIONS",
            "table": "order_detail",
        }
    ]


def test_grounding_commit_blocks_reusing_semantically_rejected_table() -> None:
    detail_ref = "semantic:OPERATIONS:ads_merchant_profile:detail"
    metric_ref = "semantic:OPERATIONS:ads_merchant_profile:metric:daily_user_count"
    rejection = {
        "fingerprint": "rejected-1",
        "code": "TABLE_INSUFFICIENT",
        "topic": "OPERATIONS",
        "table": "ads_merchant_profile",
        "refIds": [detail_ref, metric_ref],
        "requiredCapability": {
            "operation": "COUNT_DISTINCT",
            "entityRole": "BUYER",
            "requiredFieldRole": "KEY",
        },
    }
    evidence = [
        {
            "refId": detail_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/detail.json",
            "kind": "TABLE_DETAIL",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": '{"tableName":"ads_merchant_profile"}',
            "contentHash": "detail-hash",
            "contentComplete": True,
        },
        {
            "refId": metric_ref,
            "path": "topics/OPERATIONS/tables/ads_merchant_profile/metrics/daily_user_count.json",
            "kind": "METRIC",
            "topic": "OPERATIONS",
            "table": "ads_merchant_profile",
            "contentSnippet": json.dumps(
                {
                    "tableName": "ads_merchant_profile",
                    "metric": {
                        "metricKey": "daily_user_count",
                        "formula": "SUM(daily_user_count)",
                        "calculationSemantics": {
                            "timeRollupPolicy": "NOT_COMPOSABLE"
                        },
                    },
                }
            ),
            "contentHash": "metric-hash",
            "contentComplete": True,
        },
    ]
    domain = _GroundedToolDomainStub()
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    session = _DianaLeadSession(
        state={
            "question": "period unique buyers",
            "react_round": 2,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["OPERATIONS"]},
            "grounded_rejected_bindings": [rejection],
        },
        sink=_ResultSink(),
        table_manifest_disclosed=True,
        core_semantic_evidence=evidence,
    )
    runtime = SimpleNamespace(context=SimpleNamespace(session=session))

    raw = adapter._build_commit_grounded_contract_tool().func(
        table_refs=[detail_ref],
        metric_refs=[metric_ref],
        dimension_refs=[],
        group_by_ref="",
        label_refs={},
        relationship_refs=[],
        ranking_order="",
        limit=0,
        analysis_mode="metric_total",
        time_expression="last 30 days",
        reason="retry rejected binding",
        runtime=runtime,
    )
    result = json.loads(raw)

    assert result["groundingCommitBlocked"]["code"] == "REJECTED_BINDING_REUSED"
    assert result["groundingCommitBlocked"]["rejectedTable"] == "ads_merchant_profile"
    assert domain.commit_input == {}


def test_semantic_read_loop_guard_blocks_identical_no_progress_calls() -> None:
    catalog = SimpleNamespace(
        read=lambda **_kwargs: {
            "success": False,
            "error": "SEMANTIC_REF_NOT_FOUND",
        }
    )
    backend = ReadOnlySemanticBackend(catalog)
    session = _DianaLeadSession(
        state={
            "topic_workspace": {"mode": "topic_workspace", "topics": ["OPERATIONS"]},
        },
        sink=_ResultSink(),
    )

    with backend.scope_to_session(session):
        results = [
            backend.read(
                "/topics/OPERATIONS/tables/missing/detail.json",
                offset=0,
                limit=2000,
            )
            for _ in range(4)
        ]

    assert [item.error for item in results[:3]] == [
        "SEMANTIC_REF_NOT_FOUND",
        "SEMANTIC_REF_NOT_FOUND",
        "SEMANTIC_REF_NOT_FOUND",
    ]
    assert "SEMANTIC_TOOL_NO_PROGRESS_BLOCKED" in str(results[3].error)
    assert session.state["semantic_tool_loop_guard"]["count"] == 4
