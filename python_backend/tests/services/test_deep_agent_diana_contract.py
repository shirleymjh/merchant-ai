from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import merchant_ai.services.deep_agent_runtime as runtime_module
from deepagents.middleware.filesystem import FilesystemMiddleware
from merchant_ai.services.deep_agent_runtime import (
    DeepAgentWorkflowAdapter,
    ReadOnlyRunArtifactBackend,
    ReadOnlySemanticBackend,
    _DianaLeadSession,
    _ResultSink,
)


class _SemanticCatalogStub:
    def __init__(self) -> None:
        self.read_calls: list[str] = []
        self.ls_calls: list[str] = []
        self.grep_calls: list[tuple[str, str]] = []

    def read(self, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        del max_chars, offset
        self.read_calls.append(path)
        if path.strip("/") == "topics/index.json":
            return {
                "success": True,
                "refId": "semantic:topics:index",
                "path": "topics/index.json",
                "kind": "TOPIC_INDEX",
                "topic": "",
                "content": '{"topics":[{"topic":"FINANCE","manifestPath":"topics/FINANCE/manifest.json"}]}\n',
            }
        parts = [part for part in path.strip("/").split("/") if part]
        topic = parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""
        return {
            "success": True,
            "refId": f"semantic:{topic}:manifest",
            "path": path.strip("/"),
            "kind": "TOPIC_MANIFEST",
            "topic": topic,
            "content": '{"kind":"manifest"}\n',
        }

    @staticmethod
    def topic_index_ref() -> dict[str, Any]:
        return {
            "content": '{"topics":[{"topic":"FINANCE","manifestPath":"topics/FINANCE/manifest.json"}]}\n'
        }

    def ls(self, path: str, limit: int) -> list[dict[str, Any]]:
        del limit
        self.ls_calls.append(path)
        return [{"path": f"{path.strip('/')}/manifest.json", "estimatedChars": 32}]

    def grep(self, query: str, topic: str, limit: int, path: str = "") -> list[dict[str, Any]]:
        del limit, path
        self.grep_calls.append((query, topic))
        return [
            {
                "path": f"topics/{topic}/manifest.json",
                "summary": f"match in {topic}",
            }
        ]


class _EvidenceSemanticCatalogStub(_SemanticCatalogStub):
    def read(self, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        del max_chars, offset
        normalized = path.strip("/")
        self.read_calls.append(normalized)
        parts = [part for part in normalized.split("/") if part]
        topic = parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""
        table = parts[3] if len(parts) >= 4 and parts[2] == "tables" else ""
        if normalized.endswith("/detail.json"):
            kind = "TABLE_DETAIL"
            ref_id = f"semantic:{topic}:{table}:detail"
            content = '{"tableName":"%s","children":["metrics","schema"]}\n' % table
        elif "/metrics/" in normalized:
            metric_key = parts[-1].removesuffix(".json")
            kind = "METRIC"
            ref_id = f"semantic:{topic}:{table}:metric:{metric_key}"
            content = '{"metricKey":"%s","formula":"SUM(amount)"}\n' % metric_key
        elif normalized.endswith("/schema.json"):
            kind = "SCHEMA"
            ref_id = f"semantic:{topic}:{table}:schema"
            content = '{"columns":[{"name":"event_day","type":"date"}]}\n'
        else:
            kind = "TOPIC_MANIFEST"
            ref_id = f"semantic:{topic}:manifest"
            content = '{"kind":"manifest"}\n'
        return {
            "success": True,
            "refId": ref_id,
            "path": normalized,
            "kind": kind,
            "topic": topic,
            "table": table,
            "content": content,
        }


class _ActionRegistryStub:
    @staticmethod
    def get(action_id: str) -> Any:
        return SimpleNamespace(
            id=action_id,
            node="plan_query_graph",
            agent="PlannerAgent",
            description="build QueryGraph",
        )

    @staticmethod
    def actions(action_ids: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
        return [{"id": action_id} for action_id in action_ids]


class _MiddlewareChainStub:
    @staticmethod
    def before_action(state: dict[str, Any], decision: Any) -> dict[str, Any]:
        del decision
        return state

    @staticmethod
    def capture_action(state: dict[str, Any], decision: Any) -> dict[str, Any]:
        del decision
        return state

    @staticmethod
    def after_action(state: dict[str, Any]) -> dict[str, Any]:
        return state

    @staticmethod
    def before_policy(state: dict[str, Any]) -> dict[str, Any]:
        return state


class _PlanActionDomainStub:
    def __init__(self) -> None:
        registry = _ActionRegistryStub()
        self.policy = SimpleNamespace(
            max_main_actions=8,
            registry=registry,
            decide=lambda _state: SimpleNamespace(available_actions=[]),
        )
        self.middleware_chain = _MiddlewareChainStub()
        self.handler_calls = 0

    @staticmethod
    def materialize_plan_clarification(state: dict[str, Any]) -> None:
        del state

    @staticmethod
    def refresh_execution_tier_policy(state: dict[str, Any]) -> None:
        del state

    @staticmethod
    def main_agent_observation(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {"summary": "plan action completed"}

    @staticmethod
    def build_lead_decision_context(state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        del state, observation
        return {}

    @staticmethod
    def ensure_terminal_planning_gap(state: dict[str, Any], decision: Any) -> None:
        del state, decision

    def plan_query_graph(self, state: dict[str, Any]) -> dict[str, Any]:
        self.handler_calls += 1
        state["plan_graph_handler_called"] = True
        return state


class _TopicAssetsStub:
    settings = SimpleNamespace()

    @staticmethod
    def resolve_topic_category(topic: str) -> str:
        return topic

    @staticmethod
    def load_manifest(topic: str) -> list[dict[str, Any]]:
        assert topic == "SALES"
        return [
            {
                "tableName": "fact_order",
                "tableComment": "Orders",
                # Sensitive L1+ fields deliberately present in the source. The
                # L0 disclosure must ignore all of them.
                "metrics": [{"name": "gmv", "formula": "sum(amount)"}],
                "columns": [{"name": "event_day"}],
                "schema": {"event_day": "date"},
                "rules": ["exclude test merchants"],
                "relationships": [{"to": "dim_merchant"}],
            }
        ]


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


def test_diana_planner_remains_governed_core_action_not_deepagent_subagent(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_module, "create_deep_agent", fake_create_deep_agent)
    domain = SimpleNamespace(
        settings=SimpleNamespace(resolved_ops_path=tmp_path / "ops" / "runtime.json"),
        checkpoint_manager=SimpleNamespace(saver=lambda: object()),
        graph=object(),
    )
    lead_llm = SimpleNamespace(configured=True, chat_model=lambda: object())

    adapter = DeepAgentWorkflowAdapter(domain, lead_llm, SimpleNamespace())

    assert adapter.deep_agent_graph is not None
    assert "Planner is not a subagent" in captured["system_prompt"]
    assert "routing, planning authority, QueryGraph validation" in captured["system_prompt"]
    assert [subagent["name"] for subagent in captured["subagents"]] == ["general-purpose"]
    assert all("planner" not in subagent["name"].lower() for subagent in captured["subagents"])
    assert "no authority to plan QueryGraph" in captured["subagents"][0]["system_prompt"]
    assert "planner_subagent" not in DeepAgentWorkflowAdapter.MIGRATED_COMPONENTS
    assert "native ls, read_file and grep" in captured["system_prompt"]
    assert "Planner does not browse or deep-read" in captured["system_prompt"]
    assert "every plan as a revisable hypothesis" in captured["system_prompt"]
    assert "Zero rows and failed queries" in captured["system_prompt"]


def test_deepagent_large_tool_results_offload_at_twenty_thousand_tokens() -> None:
    parameter = inspect.signature(FilesystemMiddleware.__init__).parameters[
        "tool_token_limit_before_evict"
    ]

    assert parameter.default == 20_000
    assert "artifact_offload" in DeepAgentWorkflowAdapter.MIGRATED_COMPONENTS


def test_deepagent_owns_short_term_context_but_not_governed_merchant_memory(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_module, "create_deep_agent", fake_create_deep_agent)
    domain = SimpleNamespace(
        settings=SimpleNamespace(resolved_ops_path=tmp_path / "ops" / "runtime.json"),
        checkpoint_manager=SimpleNamespace(saver=lambda: object()),
        graph=object(),
    )

    adapter = DeepAgentWorkflowAdapter(
        domain,
        SimpleNamespace(configured=True, chat_model=lambda: object()),
        SimpleNamespace(),
    )

    assert "run_context_schema" in DeepAgentWorkflowAdapter.MIGRATED_COMPONENTS
    assert "merchant_personal_memory_store" in DeepAgentWorkflowAdapter.DOMAIN_GOVERNED_COMPONENTS
    assert "shared_knowledge_publish_governance" in DeepAgentWorkflowAdapter.DOMAIN_GOVERNED_COMPONENTS
    assert captured["context_schema"] is runtime_module.DeepAgentRunContext
    # Deep Agents MemoryMiddleware loads editable AGENTS.md content into the
    # system prompt. It cannot preserve the separate personal-memory and shared
    # knowledge lifecycles, so it must not replace the domain stores.
    assert "memory" not in captured
    assert adapter.backend.routes["/artifacts/"] is adapter.artifact_backend
    assert "short-term conversation messages" in captured["system_prompt"]
    assert "Personal merchant memory" in captured["system_prompt"]
    assert "separate knowledge candidates" in captured["system_prompt"]
    assert "never copy either personal memory or shared knowledge" in captured["system_prompt"]
    assert "mounted read-only at /artifacts" in captured["system_prompt"]
    assert any(
        permission.mode == "deny"
        and permission.operations == ["write"]
        and permission.paths == ["/memory", "/memory/**"]
        for permission in captured["permissions"]
    )
    assert any(
        permission.mode == "deny"
        and permission.operations == ["write"]
        and permission.paths == ["/artifacts", "/artifacts/**"]
        for permission in captured["permissions"]
    )


def test_semantic_backend_uses_topic_index_and_opened_manifests_to_grow_workspace() -> None:
    catalog = _SemanticCatalogStub()
    backend = ReadOnlySemanticBackend(catalog)

    assert "TOPIC_SCOPE_REQUIRED" in (backend.ls("/topics").error or "")
    assert "TOPIC_SCOPE_REQUIRED" in (backend.read("/topics/SALES/manifest.json").error or "")
    assert "TOPIC_SCOPE_REQUIRED" in (backend.grep("orders", "/topics/SALES").error or "")
    assert catalog.read_calls == []
    assert catalog.ls_calls == []
    assert catalog.grep_calls == []

    state = {"topic_workspace": {"topics": ["SALES"]}}
    with backend.scope_to_state(state):
        topic_listing = backend.ls("/topics")
        assert topic_listing.error is None
        assert [entry["path"] for entry in topic_listing.entries or []] == [
            "/topics/index.json",
            "/topics/SALES/",
        ]

        index_read = backend.read("/topics/index.json")
        assert index_read.error is None

        active_read = backend.read("/topics/SALES/manifest.json")
        assert active_read.error is None
        assert active_read.file_data and "manifest" in active_read.file_data["content"]

        opened_read = backend.read("/topics/FINANCE/manifest.json")
        opened_ls = backend.ls("/topics/FINANCE")
        opened_grep = backend.grep("revenue", "/topics/FINANCE")
        assert opened_read.error is None
        assert opened_ls.error is None
        assert opened_grep.error is None
        assert state["semantic_workspace_opened_topics"] == ["FINANCE"]
        assert state["topic_workspace"]["effectiveTopics"] == ["SALES", "FINANCE"]

    assert catalog.read_calls == [
        "topics/index.json",
        "topics/SALES/manifest.json",
        "topics/FINANCE/manifest.json",
    ]
    assert catalog.ls_calls == []
    assert catalog.grep_calls == [("revenue", "FINANCE")]


def test_semantic_backend_canonicalizes_alias_paths_and_honors_user_topic_lock() -> None:
    catalog = _SemanticCatalogStub()
    backend = ReadOnlySemanticBackend(catalog)
    state = {
        "topic_workspace": {
            "mode": "explicit_topic_scope",
            "topics": ["SALES"],
            "isolated": True,
            "expansionPolicy": "user_locked",
        }
    }

    with backend.scope_to_state(state):
        denied_before_index = backend.read("runtime/topics/FINANCE/manifest.json")
        assert "TOPIC_SCOPE_LOCKED" in (denied_before_index.error or "")
        assert backend.read("resources/runtime/topics/index.json").error is None
        denied_after_index = backend.read("resources/runtime/topics/FINANCE/manifest.json")

    assert "TOPIC_SCOPE_LOCKED" in (denied_after_index.error or "")
    assert state.get("semantic_workspace_opened_topics") in (None, [])
    assert catalog.read_calls == ["topics/index.json"]


def test_deepagent_tool_schemas_never_expose_runtime_or_trusted_session() -> None:
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    inspect_tool = DeepAgentWorkflowAdapter._build_inspect_tool(adapter)
    action_tool = DeepAgentWorkflowAdapter._build_action_tool(adapter)

    inspect_schema = inspect_tool.tool_call_schema.model_json_schema()
    action_schema = action_tool.tool_call_schema.model_json_schema()

    assert inspect_schema["properties"] == {}
    assert set(action_schema["properties"]) == {"action_id", "reason"}
    serialized = str({"inspect": inspect_schema, "action": action_schema}).lower()
    assert "runtime" not in serialized
    assert "session" not in serialized
    assert "merchant_id" not in serialized
    assert "thread_id" not in serialized
    assert "run_id" not in serialized


def test_active_run_artifacts_are_mounted_read_only_with_path_confinement(tmp_path: Path) -> None:
    outputs = tmp_path / "threads" / "thread_1" / "runs" / "run_1" / "outputs"
    planner_dir = outputs / "artifacts" / "planner"
    planner_dir.mkdir(parents=True)
    (planner_dir / "query_graph.json").write_text(
        '{"nodes":[{"taskId":"ticket_count"}]}\n',
        encoding="utf-8",
    )
    session = _DianaLeadSession(
        state={
            "thread_data": SimpleNamespace(outputs_path=str(outputs)),
            "topic_workspace": {"topics": ["SALES"]},
        },
        sink=_ResultSink(),
    )
    scope = ReadOnlySemanticBackend(_SemanticCatalogStub())
    backend = ReadOnlyRunArtifactBackend(SimpleNamespace(resolved_workspace_path=tmp_path))

    assert "ARTIFACT_SCOPE_REQUIRED" in (backend.read("/planner/query_graph.json").error or "")
    with scope.scope_to_session(session):
        assert [item["path"] for item in backend.ls("/").entries or []] == ["/planner/"]
        assert [item["path"] for item in backend.ls("/planner").entries or []] == [
            "/planner/query_graph.json"
        ]
        read_result = backend.read("/planner/query_graph.json")
        assert read_result.file_data and "ticket_count" in read_result.file_data["content"]
        matches = backend.grep("ticket_count", "/planner")
        assert [(item["path"], item["line"]) for item in matches.matches or []] == [
            ("/planner/query_graph.json", 1)
        ]
        assert backend.write("/planner/query_graph.json", "replace").error == "ARTIFACT_BACKEND_READ_ONLY"
        assert backend.read("../../outside.txt").error == "ARTIFACT_PATH_OUTSIDE_ROOT"


def test_adapter_settings_assignment_remains_a_domain_workflow_facade() -> None:
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    original = SimpleNamespace(name="original")
    replacement = SimpleNamespace(name="replacement")
    domain = SimpleNamespace(settings=original)
    adapter.domain_workflow = domain

    adapter.settings = replacement

    assert domain.settings is replacement
    assert "settings" not in adapter.__dict__


def test_only_successful_core_read_file_calls_enter_trusted_semantic_evidence() -> None:
    catalog = _EvidenceSemanticCatalogStub()
    backend = ReadOnlySemanticBackend(catalog)
    session = _DianaLeadSession(
        state={"topic_workspace": {"topics": ["SALES"]}},
        sink=_ResultSink(),
    )

    with backend.scope_to_session(session):
        backend.ls("/topics/SALES")
        backend.grep("orders", "/topics/SALES")
        assert session.core_semantic_evidence == []

        detail = backend.read("/topics/SALES/tables/fact_order/detail.json", offset=0, limit=20)
        assert detail.error is None
        backend.read("/topics/SALES/tables/fact_order/detail.json", offset=0, limit=20)
        denied = backend.read("/topics/FINANCE/tables/fact_other/detail.json")

    assert "TOPIC_SCOPE_DENIED" in (denied.error or "")
    assert len(session.core_semantic_evidence) == 1
    evidence = session.core_semantic_evidence[0]
    assert evidence["refId"] == "semantic:SALES:fact_order:detail"
    assert evidence["path"] == "topics/SALES/tables/fact_order/detail.json"
    assert evidence["kind"] == "TABLE_DETAIL"
    assert evidence["topic"] == "SALES"
    assert evidence["table"] == "fact_order"
    assert "children" in evidence["contentSnippet"]
    assert len(evidence["contentHash"]) == 64
    assert evidence["offset"] == 0
    assert evidence["limit"] == 20
    assert evidence["contentComplete"] is True
    assert catalog.ls_calls == []
    assert catalog.grep_calls == [("orders", "SALES")]
    assert catalog.read_calls == [
        "topics/SALES/tables/fact_order/detail.json",
        "topics/SALES/tables/fact_order/detail.json",
    ]


def test_plan_graph_rejects_until_core_reads_table_detail_and_exact_definition() -> None:
    catalog = _EvidenceSemanticCatalogStub()
    backend = ReadOnlySemanticBackend(catalog)
    domain = _PlanActionDomainStub()
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace()
    session = _DianaLeadSession(
        state={
            "react_round": 1,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["SALES"]},
        },
        sink=_ResultSink(),
        available_actions=("plan_graph",),
        observation={"summary": "assets compacted"},
        table_manifest_disclosed=True,
    )

    with backend.scope_to_session(session):
        backend.read("/topics/SALES/tables/fact_order/detail.json")

    rejected = adapter._execute_action(session, "plan_graph", "compile the selected table")

    assert rejected["status"] == "ACTION_REJECTED"
    assert rejected["error"] == "CORE_SEMANTIC_EVIDENCE_REQUIRED"
    assert rejected["missingSemanticEvidence"] == ["EXACT_DEFINITION_OR_SCHEMA_REQUIRED"]
    assert "read_file" in rejected["next"]
    assert domain.handler_calls == 0
    assert session.state["core_semantic_evidence"] == session.core_semantic_evidence
    assert session.state["core_managed_filesystem"] is True
    assert rejected["coreSemanticEvidence"]["readCount"] == 1
    assert rejected["coreSemanticEvidence"]["contractProposalReady"] is False
    assert "contentSnippet" not in rejected["coreSemanticEvidence"]["refs"][0]

    with backend.scope_to_session(session):
        backend.read("/topics/SALES/tables/fact_order/metrics/gmv.json")

    completed = adapter._execute_action(session, "plan_graph", "compile from Core-read refs")

    assert completed["status"] == "ACTION_REQUIRED"
    assert domain.handler_calls == 1
    assert session.state["plan_graph_handler_called"] is True
    assert len(session.state["core_semantic_evidence"]) == 2
    assert completed["coreSemanticEvidence"]["contractProposalReady"] is True
    assert completed["coreSemanticEvidence"]["missingForContractProposal"] == []
    assert session.state["lead_decisions"][-1].source == "deepagent_core_react"


def test_runtime_continuation_is_not_traced_as_a_core_model_decision() -> None:
    domain = _PlanActionDomainStub()
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.domain_workflow = domain
    adapter.semantic_catalog = SimpleNamespace()
    session = _DianaLeadSession(
        state={
            "react_round": 1,
            "topic_workspace": {"mode": "topic_workspace", "topics": ["SALES"]},
        },
        sink=_ResultSink(),
        available_actions=("plan_graph",),
        observation={"summary": "runtime continuation"},
        table_manifest_disclosed=True,
        core_semantic_evidence=[
            {
                "refId": "semantic:SALES:fact_order:detail",
                "path": "topics/SALES/tables/fact_order/detail.json",
                "kind": "TABLE_DETAIL",
                "topic": "SALES",
                "table": "fact_order",
                "contentSnippet": "detail",
                "contentHash": "detail-hash",
            },
            {
                "refId": "semantic:SALES:fact_order:metric:gmv",
                "path": "topics/SALES/tables/fact_order/metrics/gmv.json",
                "kind": "METRIC",
                "topic": "SALES",
                "table": "fact_order",
                "contentSnippet": "metric",
                "contentHash": "metric-hash",
            },
        ],
    )

    adapter._execute_action(
        session,
        "plan_graph",
        "runtime fail-closed continuation after DeepAgent stopped",
        decision_source="runtime_fail_closed",
    )

    assert session.state["lead_decisions"][-1].source == "runtime_fail_closed"


def test_first_semantic_disclosure_is_strict_l0_table_manifest_and_is_emitted_once() -> None:
    adapter = object.__new__(DeepAgentWorkflowAdapter)
    adapter.semantic_catalog = SimpleNamespace(topic_assets=_TopicAssetsStub())
    adapter.domain_workflow = SimpleNamespace(
        policy=SimpleNamespace(max_main_actions=8, registry=SimpleNamespace())
    )
    session = _DianaLeadSession(
        state={
            "react_round": 1,
            "topic_workspace": {
                "mode": "topic_workspace",
                "topics": ["SALES"],
            },
        },
        sink=_ResultSink(),
        available_actions=(),
        observation={"summary": "topic selected"},
    )

    first = DeepAgentWorkflowAdapter._turn_payload(adapter, session)
    second = DeepAgentWorkflowAdapter._turn_payload(adapter, session)

    assert "tableManifest" in first
    assert "tableManifest" not in second
    assert "semanticDisclosure" not in second
    assert session.table_manifest_disclosed is True
    assert first["semanticDisclosure"] == {
        "layer": "L0",
        "contains": ["topic", "table", "title", "businessSummary", "detailRefId", "detailPath"],
        "omits": ["metrics", "columns", "schema", "rules", "relationships"],
        "next": "choose a table, then read /knowledge/<detailPath>",
    }
    manifest = first["tableManifest"]
    assert manifest["tableCount"] == 1
    assert manifest["tables"] == [
        {
            "topic": "SALES",
            "table": "fact_order",
            "title": "Orders",
            "detailRefId": "semantic:SALES:fact_order:detail",
            "detailPath": "topics/SALES/tables/fact_order/detail.json",
        }
    ]
    assert not _nested_keys(manifest).intersection(
        {"metrics", "formula", "columns", "schema", "rules", "relationships"}
    )
