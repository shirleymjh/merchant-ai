from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from merchant_ai.models import (
    ClarificationRequest,
    ExtractedKeywords,
    MerchantInfo,
    RecallBundle,
    RecallItem,
    TopicRoutingDecision,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedCoreToolBoundaryMiddleware,
    GroundedSemanticBackend,
    _thin_recall,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeAttempt,
    GroundedRuntimeSession,
)


class FakeSemanticCatalog:
    def read(self, *, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        if path == "topics/客服工单/manifest.json":
            content = '{"topic":"客服工单","tables":[{"tableName":"tickets"}]}'
            return {
                "success": True,
                "refId": "semantic:客服工单:manifest",
                "path": path,
                "kind": "TOPIC_MANIFEST",
                "topic": "客服工单",
                "content": content,
            }
        if path == "topics/客服工单/tables/tickets/detail.json":
            content = '{"topic":"客服工单","tableName":"tickets"}'
            return {
                "success": True,
                "refId": "semantic:客服工单:tickets:detail",
                "path": path,
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "content": content,
            }
        return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}

    @staticmethod
    def ls(path: str, limit: int) -> list[dict[str, Any]]:
        return [
            {
                "path": "topics/客服工单/tables/tickets/detail.json",
                "estimatedChars": 100,
            }
        ]

    @staticmethod
    def grep(query: str, topic: str, limit: int, path: str) -> list[dict[str, Any]]:
        return [
            {
                "path": "topics/客服工单/tables/tickets/detail.json",
                "snippets": ["工单明细表"],
            }
        ]


class FakeKernel:
    def __init__(self):
        self.route_calls = 0
        self.recall_queries: list[str] = []
        self.propose_calls = 0
        self.compile_calls = 0

    def new_session(self, question: str, merchant_id: str, **kwargs: Any) -> GroundedRuntimeSession:
        return GroundedRuntimeSession(
            session_id="s1",
            question=question,
            merchant_id=merchant_id,
            merchant=kwargs.get("merchant") or MerchantInfo(merchant_id=merchant_id),
        )

    def route_topic(self, session: GroundedRuntimeSession) -> TopicRoutingDecision:
        self.route_calls += 1
        session.keywords = ExtractedKeywords(keywords=["工单"])
        session.routing = TopicRoutingDecision(
            primary_topic="SERVICE",
            candidate_topics=["SERVICE"],
            routing_mode="seed_topic",
        )
        session.workspace_topics = ["客服工单"]
        return session.routing

    def recall_navigation(self, session: GroundedRuntimeSession, *, query: str = "", **kwargs: Any) -> RecallBundle:
        self.recall_queries.append(query or session.question)
        bundle = RecallBundle(
            items=[
                RecallItem(
                    doc_id="semantic:客服工单:tickets:detail",
                    title="客服工单明细表",
                    content="包含工单与商品字段",
                    source_type="TABLE_DETAIL",
                    topic="客服工单",
                    table="tickets",
                    fusion_score=8.0,
                    metadata={
                        "semanticRefId": "semantic:客服工单:tickets:detail",
                        "semanticPath": "topics/客服工单/tables/tickets/detail.json",
                    },
                )
            ]
        )
        session.recall = bundle
        return bundle

    def propose_contract(self, session: GroundedRuntimeSession, evidence: list[dict[str, Any]], hints: dict[str, Any], **kwargs: Any) -> GroundedRuntimeAttempt:
        self.propose_calls += 1
        assert evidence[0]["refId"] == "semantic:客服工单:tickets:detail"
        attempt = GroundedRuntimeAttempt(
            attempt_id="a1",
            contract=GroundedQueryContract(
                question=session.question,
                topics=["客服工单"],
                status="READY",
                query_shape="SCALAR",
            ),
        )
        session.attempts.append(attempt)
        return attempt

    def compile_candidate(self, session: GroundedRuntimeSession, attempt_id: str) -> GroundedRuntimeAttempt:
        self.compile_calls += 1
        attempt = session.attempts[-1]
        attempt.compile_status = "VALID"
        attempt.activated = True
        return attempt

    @staticmethod
    def request_clarification(session: GroundedRuntimeSession, question: str, **kwargs: Any) -> ClarificationRequest:
        request = ClarificationRequest(
            question=question,
            stage=kwargs["stage"],
            type=kwargs["clarification_type"],
            options=kwargs.get("options") or [],
            pending_question=session.question,
        )
        session.clarification = request
        return request


class FakeGraph:
    def __init__(self, tools: list[Any], action: str = "clarify"):
        self.tools = {item.name: item for item in tools}
        self.action = action
        self.invocations: list[dict[str, Any]] = []
        self.configs: list[Any] = []

    def invoke(self, payload: dict[str, Any], *, config: Any, context: Any) -> None:
        self.invocations.append(payload)
        self.configs.append(config)
        if self.action == "clarify":
            self.tools["ask_human"].func(
                question="请问要查询最近多少天？",
                stage="time_binding",
                clarification_type="TIME_RANGE_REQUIRED",
                options=["最近7天", "最近30天"],
                runtime=SimpleNamespace(context=context),
            )


class CapturingFactory:
    def __init__(self, action: str = "clarify", fail: bool = False):
        self.action = action
        self.fail = fail
        self.kwargs: dict[str, Any] = {}
        self.graph: Any = None

    def __call__(self, **kwargs: Any) -> Any:
        if self.fail:
            raise ValueError("boom")
        self.kwargs = kwargs
        self.graph = FakeGraph(kwargs["tools"], self.action)
        return self.graph


def runtime(factory: CapturingFactory, kernel: FakeKernel | None = None) -> GroundedDeepAgentRuntime:
    return GroundedDeepAgentRuntime(
        kernel or FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=factory,
    )


def test_runtime_source_has_no_legacy_or_action_catalog_dependencies() -> None:
    source = Path(
        "python_backend/merchant_ai/services/grounded_deep_agent_runtime.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "MerchantQaWorkflow",
        "V2AgentPolicy",
        "StateGraph",
        "inspect_diana_state",
        "actionCatalog",
        "run_diana_action",
        "graph.workflow",
        "graph.policy",
    ):
        assert forbidden not in source


def test_initialization_keeps_one_core_native_filesystem_skills_and_isolated_subagent() -> None:
    factory = CapturingFactory()
    runtime(factory)

    assert {item.name for item in factory.kwargs["tools"]} == {
        "retrieve_knowledge",
        "propose_grounded_contract",
        "execute_grounded_query",
        "compose_verified_answer",
        "ask_human",
    }
    contract_tool = next(
        item
        for item in factory.kwargs["tools"]
        if item.name == "propose_grounded_contract"
    )
    contract_schema = json.dumps(
        contract_tool.tool_call_schema.model_json_schema(),
        ensure_ascii=False,
    )
    assert "tableRefs" in contract_schema
    assert "metricRefs" in contract_schema
    assert "timeExpression" in contract_schema
    assert factory.kwargs["backend"] is not None
    assert len(factory.kwargs["middleware"]) == 1
    assert isinstance(
        factory.kwargs["middleware"][0],
        GroundedCoreToolBoundaryMiddleware,
    )
    assert [item["name"] for item in factory.kwargs["subagents"]] == [
        "general-purpose"
    ]
    assert factory.kwargs["subagents"][0]["tools"] == []
    assert "execute SQL" in factory.kwargs["subagents"][0]["system_prompt"]


def test_run_bootstraps_topic_and_scoped_recall_into_first_core_context() -> None:
    factory = CapturingFactory()
    kernel = FakeKernel()
    outer = runtime(factory, kernel)

    response = outer.run("工单量最多的商品", "m-1")

    assert kernel.route_calls == 1
    assert kernel.recall_queries == ["工单量最多的商品"]
    first = json.loads(factory.graph.invocations[0]["messages"][0]["content"])
    assert first["trustedExecutionScope"]["merchantScopeBound"] is True
    assert first["trustedExecutionScope"]["merchantId"] == "m-1"
    assert "automatically binds" in first["trustedExecutionScope"]["tenantFilterPolicy"]
    assert first["topicL0Manifests"][0]["topic"] == "客服工单"
    assert first["thinRecallCandidates"][0]["refId"] == "semantic:客服工单:tickets:detail"
    assert response.clarification is not None
    assert response.debug_trace["harness"]["legacyFallbackUsed"] is False


def test_semantic_backend_records_only_complete_exact_reads() -> None:
    backend = GroundedSemanticBackend(
        FakeSemanticCatalog(),
        reader_is_core=lambda: True,
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="s1",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )

    with backend.scope(session):
        result = backend.read(
            "/topics/客服工单/tables/tickets/detail.json",
            offset=0,
            limit=2000,
        )

    assert result.error is None
    assert session.core_semantic_evidence[0]["refId"] == "semantic:客服工单:tickets:detail"
    assert session.core_semantic_evidence[0]["contentComplete"] is True


def test_subagent_semantic_reads_never_enter_core_submission_ledger() -> None:
    identity = {"core": False}
    backend = GroundedSemanticBackend(
        FakeSemanticCatalog(),
        reader_is_core=lambda: identity["core"],
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="s1",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )

    with backend.scope(session):
        subagent_read = backend.read(
            "/topics/客服工单/tables/tickets/detail.json",
            limit=2000,
        )
        identity["core"] = True
        core_read = backend.read(
            "/topics/客服工单/tables/tickets/detail.json",
            limit=2000,
        )

    assert subagent_read.error is None
    assert core_read.error is None
    assert len(session.core_semantic_evidence) == 1
    assert session.core_semantic_evidence[0]["refId"] == "semantic:客服工单:tickets:detail"


def test_root_core_read_middleware_records_exact_complete_file() -> None:
    backend = GroundedSemanticBackend(
        FakeSemanticCatalog(),
        reader_is_core=lambda: False,
    )
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="s1",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    context = GroundedDeepAgentRunContext(thread_id="t1", run_id="r1", session=session)
    request = SimpleNamespace(
        tool_call={
            "id": "call-1",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/客服工单/tables/tickets/detail.json",
                "offset": 0,
                "limit": 2000,
            },
        },
        runtime=SimpleNamespace(context=context),
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(
            content="file content",
            tool_call_id="call-1",
            name="read_file",
        ),
    )

    assert result.status == "success"
    assert [item["refId"] for item in session.core_semantic_evidence] == [
        "semantic:客服工单:tickets:detail"
    ]


def test_task_dispatch_is_blocked_before_subagent_execution() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(
        GroundedSemanticBackend(FakeSemanticCatalog())
    )
    called = {"handler": False}
    request = SimpleNamespace(
        tool_call={"id": "call-task", "name": "task", "args": {}},
        runtime=SimpleNamespace(context=None),
    )

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(content="unexpected", tool_call_id="call-task")

    result = middleware.wrap_tool_call(request, handler)

    assert result.status == "error"
    assert called["handler"] is False


def test_full_table_asset_is_never_exposed_by_grounded_filesystem_or_thin_recall() -> None:
    backend = GroundedSemanticBackend(FakeSemanticCatalog())
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="s1",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    with backend.scope(session):
        denied = backend.read(
            "/topics/客服工单/tables/tickets/asset.json",
            limit=2000,
        )
    assert str(denied.error or "").startswith("FULL_TABLE_ASSET_DENIED")

    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:客服工单:tickets:asset",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="客服工单",
                table="tickets",
                metadata={
                    "semanticRefId": "semantic:客服工单:tickets:asset",
                    "semanticPath": "topics/客服工单/tables/tickets/asset.json",
                    "semanticKind": "TABLE_ASSET",
                },
            ),
            RecallItem(
                doc_id="semantic:客服工单:tickets:detail",
                source_type="SEMANTIC_TABLE_ASSET",
                topic="客服工单",
                table="tickets",
                metadata={
                    "semanticRefId": "semantic:客服工单:tickets:detail",
                    "semanticPath": "topics/客服工单/tables/tickets/detail.json",
                    "semanticKind": "TABLE_DETAIL",
                },
            ),
        ]
    )
    assert [item["refId"] for item in _thin_recall(recall, 8)] == [
        "semantic:客服工单:tickets:detail"
    ]


def test_typed_retrieve_and_contract_tools_use_kernel_without_action_dispatch() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("工单量", "m-1")
    kernel.route_topic(kernel_session)
    session = GroundedDeepAgentSession(
        runtime=kernel_session,
        core_semantic_evidence=[
            {
                "refId": "semantic:客服工单:tickets:detail",
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "contentSnippet": "{}",
                "contentHash": "hash",
            }
        ],
    )
    context = GroundedDeepAgentRunContext(thread_id="t1", run_id="r1", session=session)
    tools = {item.name: item for item in outer.tools}

    recall_result = json.loads(
        tools["retrieve_knowledge"].func(
            query="商品字段",
            reason="need product dimension",
            runtime=SimpleNamespace(context=context),
        )
    )
    contract_result = json.loads(
        tools["propose_grounded_contract"].func(
            read_ref_ids=["semantic:客服工单:tickets:detail"],
            binding_hints={"tableRefs": ["semantic:客服工单:tickets:detail"]},
            auto_compile=True,
            runtime=SimpleNamespace(context=context),
        )
    )

    assert recall_result["status"] == "OK"
    assert kernel.recall_queries[-1] == "商品字段"
    assert contract_result["activated"] is True
    assert kernel.propose_calls == 1
    assert kernel.compile_calls == 1


def test_internal_runtime_failure_cannot_be_disguised_as_user_clarification() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("工单量", "m-1")
    context = GroundedDeepAgentRunContext(
        thread_id="t1",
        run_id="r1",
        session=GroundedDeepAgentSession(runtime=kernel_session),
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["ask_human"].func(
            question="请提供系统本应注入的 merchant_id",
            stage="contract_binding",
            clarification_type="SYSTEM_BLOCKER",
            options=[],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "REJECTED"
    assert result["code"] == "INTERNAL_FAILURE_IS_NOT_USER_CLARIFICATION"
    assert kernel_session.clarification is None


def test_initialization_failure_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="Grounded DeepAgent initialization failed"):
        runtime(CapturingFactory(fail=True))

    with pytest.raises(RuntimeError, match="model is not configured"):
        GroundedDeepAgentRuntime(
            FakeKernel(),
            lead_model=None,
            semantic_catalog=FakeSemanticCatalog(),
            agent_factory=CapturingFactory(),
        )


def test_checkpoint_config_factory_is_used_without_rewriting_namespace() -> None:
    factory = CapturingFactory()
    expected = {
        "configurable": {
            "thread_id": "t-fixed",
            "checkpoint_ns": "deepagent",
            "run_id": "r-fixed",
        }
    }
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        checkpoint_config_factory=lambda thread_id, run_id: {
            **expected,
            "observed": {"threadId": thread_id, "runId": run_id},
        },
        agent_factory=factory,
    )

    outer.run("工单量", "m-1", thread_id="t-fixed", run_id="r-fixed")

    assert factory.graph.configs[0]["configurable"] == expected["configurable"]
    assert factory.graph.configs[0]["observed"] == {
        "threadId": "t-fixed",
        "runId": "r-fixed",
    }
