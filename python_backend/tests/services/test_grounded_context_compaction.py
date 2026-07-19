from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_compaction import (
    ProviderAwareContextTokenCounter,
    build_grounded_recovery_payload,
)
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedContextManagementMiddleware,
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedRunFilesystemBackend,
    GroundedSemanticBackend,
    _phase_visible_tools,
)
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionGraphReceipt,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeSession,
    GroundedVerifiedQueryArtifact,
    verified_query_artifact_integrity_fingerprint,
)


def _settings(
    root: Path,
    *,
    context_window_tokens: int = 1_000,
    threshold_ratio: float = 0.85,
    target_ratio: float = 0.4,
    page_chars: int = 12_000,
) -> Settings:
    return Settings(
        harness_workspace_path=str(root),
        context_window_tokens=context_window_tokens,
        context_compaction_threshold_ratio=threshold_ratio,
        context_compaction_target_ratio=target_ratio,
        context_file_inline_max_chars=page_chars,
    )


def _workspace(settings: Settings) -> GroundedContextWorkspace:
    return GroundedContextWorkspace.open(
        settings,
        thread_id="thread-context",
        run_id="run-context",
        merchant_id="merchant-context",
        access_role="merchant_analyst",
        user_scope={"tenantId": "tenant-context", "userId": "user-context"},
        question="question-context",
    )


def _query_artifact() -> GroundedVerifiedQueryArtifact:
    contract = GroundedQueryContract(
        question="question-context",
        status="READY",
        query_shape="DETAIL",
    )
    receipt = {
        "artifactFingerprint": "a" * 64,
        "manifestRef": "merchant://artifact/query_results/manifest",
        "rowsRef": "merchant://artifact/query_results/rows",
        "storedRowCount": 5,
        "resultCoverage": "ALL_ROWS",
        "rowsSha256": "b" * 64,
        "manifestSha256": "c" * 64,
    }
    bundle = QueryBundle(
        rows=[{"value": index} for index in range(5)],
        offloaded_files=["/private/runtime/query-result.json"],
        original_row_count=5,
        result_coverage="ALL_ROWS",
        runtime_events=[{"resultArtifact": receipt}],
    )
    artifact = GroundedVerifiedQueryArtifact(
        artifact_id="query-artifact-context",
        generation=2,
        attempt_id="attempt-context",
        contract_fingerprint="d" * 64,
        sql_fingerprint="e" * 64,
        contract=contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(merged_query_bundle=bundle),
        verified_evidence=VerifiedEvidence(passed=True),
        execution_mode="CORE_SQL_REQUIRED",
        output_columns=["value"],
    )
    artifact.ledger_fingerprint = (
        verified_query_artifact_integrity_fingerprint(artifact)
    )
    return artifact


def _session(settings: Settings) -> GroundedDeepAgentSession:
    runtime = GroundedRuntimeSession(
        session_id="session-context",
        question="question-context",
        merchant_id="merchant-context",
        phase="ACTIVE_COMPILED",
        active_generation=2,
        active_attempt_id="attempt-context",
        active_contract=GroundedQueryContract(
            question="question-context",
            status="READY",
            query_shape="DETAIL",
        ),
    )
    artifact = _query_artifact()
    runtime.verified_query_ledger = [artifact]
    return GroundedDeepAgentSession(
        runtime=runtime,
        context_workspace=_workspace(settings),
        context_artifact_inline_max_rows=2,
        question_goal_contract=OriginalQuestionGoalContract(
            question="question-context",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.context",
                    label="context metric",
                )
            ],
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:topic:table:metric:value",
                "path": "topics/topic/tables/table/metrics/value.json",
                "kind": "METRIC",
                "topic": "topic",
                "table": "table",
                "contentHash": "f" * 64,
                "contentComplete": True,
                "contentSnippet": "full semantic body remains kernel-side",
            }
        ],
        execution_graph_generation=3,
        execution_graph_fingerprint="1" * 64,
        execution_graph_receipt=GroundedExecutionGraphReceipt(
            graph_id="graph-context",
            version=3,
            fingerprint="1" * 64,
            discovery_snapshot_fingerprint="2" * 64,
            node_ids={"node": "query-node"},
        ),
        artifact_goal_ids={
            artifact.artifact_id: ["metric.context"],
        },
    )


def _request(
    session: GroundedDeepAgentSession,
    messages: list[Any],
) -> Any:
    request = SimpleNamespace(
        messages=messages,
        tools=[],
        system_message=HumanMessage(content="system-context"),
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="thread-context",
                run_id="run-context",
                session=session,
            )
        ),
    )

    def override(**updates: Any) -> Any:
        values = dict(request.__dict__)
        values.update(updates)
        return SimpleNamespace(**values)

    request.override = override
    return request


def test_context_below_token_watermark_is_not_compacted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = _session(settings)
    messages = [HumanMessage(content="raw-context" * 200)]
    request = _request(session, messages)
    captured: dict[str, Any] = {}
    middleware = GroundedContextManagementMiddleware(
        settings,
        provider_token_counter=lambda _messages, _system, _tools: 849,
    )

    middleware.wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    assert captured["request"].messages == messages
    report = session.core_context_reports[-1]
    assert report["compactionTriggered"] is False
    assert report["decision"] == "KEEP_FULL_CONTEXT_BELOW_WATERMARK"
    assert report["beforeUsageRatio"] == 0.849
    assert not list(
        session.context_workspace.core_scratch_root.glob(
            "context/recovery_*.json"
        )
    )


def test_context_at_watermark_persists_recovery_and_compacts_to_target(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    session = _session(settings)
    messages = [
        HumanMessage(content="raw-user-context" * 500),
        AIMessage(content="raw-core-reasoning" * 500),
    ]
    original_contents = [message.content for message in messages]
    request = _request(session, messages)
    captured: dict[str, Any] = {}

    def count(current: list[Any], _system: Any, _tools: list[Any]) -> int:
        return 900 if len(current) > 1 else 350

    middleware = GroundedContextManagementMiddleware(
        settings,
        provider_token_counter=count,
    )
    middleware.wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    updated_messages = captured["request"].messages
    assert len(updated_messages) == 1
    summary = json.loads(updated_messages[0].content)
    summary_payload = summary["contextRecoverySummary"]
    report = session.core_context_reports[-1]
    assert report["compactionTriggered"] is True
    assert report["afterTokens"] == 350
    assert report["afterUsageRatio"] == 0.35
    assert report["targetAchieved"] is True
    assert report["tokenCount"]["authority"] == "PROVIDER_MODEL"
    assert report["rawCheckpointPreserved"] is True
    assert [message.content for message in messages] == original_contents
    assert summary_payload["recoveryArtifactRef"].startswith(
        "/workspace/context/recovery_"
    )

    recovery_path = (
        session.context_workspace.core_scratch_root
        / summary_payload["recoveryArtifactRef"].removeprefix("/workspace/")
    )
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    rebuilt = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )
    assert rebuilt["recoveryFingerprint"] == recovery["recoveryFingerprint"]
    assert recovery["identityBinding"]["ownerFingerprint"] == (
        session.context_workspace.owner_fingerprint
    )
    assert recovery["goalContract"]["contract"]["goals"][0]["goalId"] == (
        "metric.context"
    )
    assert recovery["semanticReceipts"][0]["refId"] == (
        "semantic:topic:table:metric:value"
    )
    assert recovery["executionGraph"]["receipt"]["graphId"] == (
        "graph-context"
    )
    assert recovery["queryArtifactReceipts"][0]["queryArtifactId"] == (
        "query-artifact-context"
    )
    assert recovery["phase"]["runtimePhase"] == "ACTIVE_COMPILED"

    immutable_read = WorkspaceArtifactStore(
        settings,
        session.context_workspace.core_scratch_root,
    ).read(
        str(recovery_path.relative_to(session.context_workspace.core_scratch_root)),
        require_immutable=True,
    )
    assert immutable_read["success"] is True


def test_token_counter_prefers_current_model_counter_and_labels_fallback() -> None:
    class ProviderModel:
        calls = 0

        def get_num_tokens_from_messages(
            self,
            messages: list[Any],
            *,
            tools: list[Any],
        ) -> int:
            self.calls += 1
            assert messages
            assert tools
            return 123

    model = ProviderModel()
    provider = ProviderAwareContextTokenCounter(model).count(
        [HumanMessage(content="hello")],
        HumanMessage(content="system"),
        [{"name": "tool"}],
    )
    fallback = ProviderAwareContextTokenCounter(object()).count(
        [HumanMessage(content="你好")],
        HumanMessage(content="system"),
        [],
    )

    assert provider.tokens == 123
    assert provider.authority == "PROVIDER_MODEL"
    assert provider.fallback_used is False
    assert model.calls == 1
    assert fallback.tokens > 0
    assert fallback.source == "conservative_utf8_bytes_estimate"
    assert fallback.authority == "CONSERVATIVE_FALLBACK"
    assert fallback.fallback_used is True


def test_recovery_never_turns_unpublished_runtime_events_into_authority(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    artifact = session.runtime.verified_query_ledger[0]

    unpublished = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )["queryArtifactReceipts"][0]

    assert unpublished["publicationStatus"] == "VERIFIED_IN_MEMORY"
    assert unpublished["resultArtifacts"] == []

    committed_receipt = dict(
        artifact.run_result.merged_query_bundle.runtime_events[0][
            "resultArtifact"
        ]
    )
    artifact.publication_status = "PUBLISHED"
    artifact.result_artifact_receipts = [committed_receipt]
    artifact.ledger_fingerprint = (
        verified_query_artifact_integrity_fingerprint(artifact)
    )
    published = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )["queryArtifactReceipts"][0]

    assert published["publicationStatus"] == "PUBLISHED"
    assert published["resultArtifacts"] == [committed_receipt]


def test_recovery_omits_a_query_artifact_after_ledger_tampering(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    artifact = session.runtime.verified_query_ledger[0]
    artifact.contract_fingerprint = "0" * 64

    recovery = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )

    assert recovery["queryArtifactReceipts"] == []


def test_frozen_graph_reopens_only_artifact_filesystem_after_verified_query(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    session.runtime.phase = "EXECUTED"
    session.runtime.active_contract = None
    session.query_branch_contexts = {
        "node": SimpleNamespace(status="EXECUTED")
    }
    tools = [
        SimpleNamespace(name=name)
        for name in (
            "ls",
            "read_file",
            "grep",
            "retrieve_knowledge",
            "finalize_evidence_collection",
        )
    ]

    visible, _ = _phase_visible_tools(session, tools)
    names = {item.name for item in visible}

    assert {"ls", "read_file", "grep"}.issubset(names)
    assert "retrieve_knowledge" not in names


@pytest.mark.parametrize("coverage", ["PREVIEW", "ALL_ROWS"])
def test_governed_response_returns_configured_preview_and_safe_artifact_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coverage: str,
) -> None:
    session = _session(_settings(tmp_path))
    artifact = session.runtime.verified_query_ledger[0]
    bundle = artifact.run_result.merged_query_bundle
    bundle.result_coverage = coverage
    bundle.is_truncated = coverage == "PREVIEW"
    session.runtime.answer = "verified answer"
    session.runtime.answer_artifact_ids = [artifact.artifact_id]
    session.runtime.run_result = artifact.run_result
    monkeypatch.setattr(
        GroundedDeepAgentRuntime,
        "_answer_is_attested",
        staticmethod(lambda _session: True),
    )

    response = GroundedDeepAgentRuntime._governed_response(
        session,
        "thread-context",
        "run-context",
    )

    assert len(response.data_rows) == 2
    assert len(response.data_sections[0].data_rows) == 2
    assert response.data_sections[0].preview_row_count == 2
    assert response.data_sections[0].original_row_count == 5
    assert response.data_sections[0].original_row_count_exact is True
    assert response.data_sections[0].result_coverage == coverage
    assert response.data_sections[0].has_more is True
    assert response.data_sections[0].offloaded_files
    assert all(
        item.startswith("merchant://")
        for item in response.data_sections[0].offloaded_files
    )
    serialized = response.model_dump_json(by_alias=True)
    assert "/private/runtime/query-result.json" not in serialized


def test_artifact_backend_pages_only_verified_immutable_files(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, page_chars=32)
    session = _session(settings)
    root = session.context_workspace.artifacts_root
    artifact = WorkspaceArtifactStore(settings, root).write_text(
        "query_results",
        "large.txt",
        "0123456789" * 20,
        immutable=True,
    )
    unverified = root / "query_results" / "unverified.txt"
    unverified.write_text("must remain hidden", encoding="utf-8")
    backend = GroundedRunFilesystemBackend(
        root_kind="artifacts",
        read_only=True,
        settings=settings,
        allowed_artifact_digests={
            artifact["relativePath"]: artifact["sha256"]
        },
    )
    scope = GroundedSemanticBackend(object())

    with scope.scope(session):
        listing = backend.ls("/artifacts/query_results")
        assert [item.path for item in listing.entries] == [
            "/query_results/large.txt"
        ]
        assert not backend.glob(
            "*/unverified.txt",
            "/artifacts",
        ).matches
        assert not backend.grep(
            "must remain hidden",
            "/artifacts",
        ).matches
        assert backend.read(
            "/artifacts/query_results/unverified.txt"
        ).error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"

        first = backend.read(
            "/artifacts/%s" % artifact["relativePath"],
            offset=0,
            limit=500,
        )
        assert first.file_data is not None
        assert len(first.file_data["content"]) == 32
        assert first.file_data["nextContentOffsetChars"] == 32
        second = backend.read(
            "/artifacts/%s" % artifact["relativePath"],
            offset=32,
            limit=500,
        )
        assert second.file_data is not None
        assert second.file_data["content"] == ("23456789012345678901234567890123")
