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
    build_grounded_model_recovery_message,
    build_grounded_recovery_payload,
    compact_summary_to_reference_only,
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
    build_grounded_execution_graph_replan_evidence,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)
from merchant_ai.services.grounded_goal_contract import (
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedContractGap,
    GroundedQueryContract,
    GroundedRejectedBinding,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeAttempt,
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
    artifact.ledger_fingerprint = verified_query_artifact_integrity_fingerprint(artifact)
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
    assert not list(session.context_workspace.core_scratch_root.glob("context/recovery_*.json"))


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
    assert summary_payload["recoveryArtifactRef"].startswith("/workspace/context/recovery_")

    recovery_path = session.context_workspace.core_scratch_root / summary_payload["recoveryArtifactRef"].removeprefix(
        "/workspace/"
    )
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    rebuilt = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )
    assert rebuilt["recoveryFingerprint"] == recovery["recoveryFingerprint"]
    assert recovery["identityBinding"]["ownerFingerprint"] == (session.context_workspace.owner_fingerprint)
    assert recovery["goalContract"]["contract"]["goals"][0]["goalId"] == ("metric.context")
    assert recovery["semanticReceipts"][0]["refId"] == ("semantic:topic:table:metric:value")
    assert recovery["executionGraph"]["receipt"]["graphId"] == ("graph-context")
    assert recovery["queryArtifactReceipts"][0]["queryArtifactId"] == ("query-artifact-context")
    assert recovery["phase"]["runtimePhase"] == "ACTIVE_COMPILED"

    immutable_read = WorkspaceArtifactStore(
        settings,
        session.context_workspace.core_scratch_root,
    ).read(
        str(recovery_path.relative_to(session.context_workspace.core_scratch_root)),
        require_immutable=True,
    )
    assert immutable_read["success"] is True


def test_recovery_keeps_full_control_summary_until_safety_threshold(
    tmp_path: Path,
) -> None:
    """The target ratio is an optimization, not a reason to drop control state."""

    settings = _settings(tmp_path)
    session = _session(settings)
    request = _request(
        session,
        [
            HumanMessage(content="raw-user-context" * 500),
            AIMessage(content="raw-core-reasoning" * 500),
        ],
    )
    captured: dict[str, Any] = {}

    def count(current: list[Any], _system: Any, _tools: list[Any]) -> int:
        if len(current) > 1:
            return 900
        try:
            payload = json.loads(str(current[0].content))
        except (TypeError, ValueError, IndexError):
            return 300
        summary = payload.get("contextRecoverySummary")
        # A full recovery summary is deliberately kept in the target/threshold
        # band.  A reference-only summary is made artificially small so the
        # test fails if the middleware incorrectly chooses it for optimization.
        return 700 if isinstance(summary, dict) and "goalContract" in summary else 300

    middleware = GroundedContextManagementMiddleware(
        settings,
        provider_token_counter=count,
    )
    middleware.wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    summary = json.loads(captured["request"].messages[0].content)
    summary_payload = summary["contextRecoverySummary"]
    report = session.core_context_reports[-1]

    assert report["compactionTriggered"] is True
    assert report["recoveryMode"] == "FULL"
    assert report["afterTokens"] == 700
    assert report["afterUsageRatio"] == 0.7
    assert report["targetAchieved"] is False
    assert summary_payload["goalContract"]["goals"][0]["goalId"] == (
        "metric.context"
    )
    assert summary_payload["semanticReceipts"][0]["refId"] == (
        "semantic:topic:table:metric:value"
    )


def test_recovery_artifact_keeps_graph_revision_and_population_authority(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    receipt = session.execution_graph_receipt
    assert receipt is not None
    trigger = build_grounded_execution_graph_replan_evidence(
        trigger_kind="TABLE_DELAY",
        source_stage="DATASOURCE",
        source_query_node_id="query-node",
        code="TABLE_FRESHNESS_COVERAGE_INCOMPLETE",
        graph_receipt=receipt,
        details={"freshnessStatus": "COVERAGE_INCOMPLETE"},
    )
    session.execution_graph_replan_evidence[trigger.evidence_id] = trigger
    session.execution_graph_revision_count = 1
    session.execution_graph_max_revision_count = 2
    session.execution_graph_used_replan_fingerprints = ["used-trigger-fingerprint"]
    session.population_graph_receipt = seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id=receipt.graph_id,
            graph_version=receipt.version,
            graph_fingerprint=receipt.fingerprint,
            nodes=(
                PopulationDynamicGraphNode(
                    query_node_id="query-node",
                    consumer_goal_ids=("metric.context",),
                ),
            ),
        )
    )

    payload = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )
    graph = payload["executionGraph"]

    assert graph["revisionCount"] == 1
    assert graph["maximumRevisionCount"] == 2
    assert graph["replanEvidence"][0]["evidenceId"] == (trigger.evidence_id)
    assert graph["replanEvidence"][0]["evidenceFingerprint"] == (trigger.evidence_fingerprint)
    assert graph["usedReplanEvidenceFingerprints"] == ["used-trigger-fingerprint"]
    assert graph["populationAttestation"]["attestationReceiptFingerprint"] == (
        session.population_graph_receipt.receipt_fingerprint
    )
    assert graph["populationAttestation"]["derivedFromExecutionGraph"] is True


def test_recovery_keeps_latest_unresolved_contract_and_graph_rejection(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    time_ref = "semantic:profile:daily:field:pt"
    unresolved = GroundedQueryContract(
        question=session.runtime.question,
        status="UNRESOLVED",
        query_shape="SCALAR",
        binding_hints=GroundedBindingHints(
            table_refs=["semantic:profile:daily:detail"],
            metric_refs=["semantic:profile:daily:metric:orders"],
            time_expression="最近7天",
            time_field_ref=time_ref,
        ),
        unresolved_gaps=[
            GroundedContractGap(
                code="TIME_FIELD_REF_NOT_READ",
                message="The selected time field has not been read.",
                evidence_kind="COLUMN",
                resolution="READ_EXACT_TIME_FIELD_OR_REMOVE_OPTIONAL_HINT",
                search_scope="CURRENT_TABLE_COLUMNS",
                required_capability={
                    "readNext": [
                        {
                            "refId": time_ref,
                            "path": (
                                "topics/profile/tables/daily/columns/pt.json"
                            ),
                        }
                    ],
                    "candidateTimeFieldRefs": [time_ref],
                },
                rejected_ref_ids=[time_ref],
            )
        ],
        rejected_bindings=[
            GroundedRejectedBinding(
                fingerprint="rejected-time-binding",
                code="TIME_FIELD_REF_NOT_READ",
                ref_ids=[time_ref],
                reason="The selected time field has not been read.",
            )
        ],
    )
    session.runtime.phase = "CONTRACT_PROPOSED"
    session.runtime.active_contract = None
    session.runtime.attempts = [
        GroundedRuntimeAttempt(
            attempt_id="attempt-unresolved",
            contract=unresolved,
            status="UNRESOLVED",
            next_action="RESOLVE_CONTRACT",
        )
    ]
    session.latest_graph_rejection = {
        "status": "REJECTED",
        "code": "EXECUTION_GRAPH_INVALID",
        "issues": [
            {
                "code": "GRAPH_NODE_GOAL_UNKNOWN",
                "nodeKey": "node-orders",
            }
        ],
        "nextAction": "REVISE_EXECUTION_GRAPH_FROM_CURRENT_DISCOVERY",
    }

    payload = build_grounded_recovery_payload(
        session,
        thread_id="thread-context",
        run_id="run-context",
    )

    attempt = payload["latestUnresolvedContractAttempt"]
    assert attempt["attemptId"] == "attempt-unresolved"
    assert attempt["status"] == "UNRESOLVED"
    assert attempt["nextAction"] == (
        "CHOOSE_SAFE_REPAIR_AND_SUBMIT_NEW_VERSION"
    )
    assert attempt["acceptedBindingHints"]["timeFieldRef"] == time_ref
    assert attempt["gaps"][0]["code"] == "TIME_FIELD_REF_NOT_READ"
    assert attempt["gaps"][0]["rejectedRefIds"] == [time_ref]
    assert attempt["rejectedRefIds"] == [time_ref]
    assert attempt["readNext"] == [
        {
            "refId": time_ref,
            "path": "topics/profile/tables/daily/columns/pt.json",
            "sourceGapCode": "TIME_FIELD_REF_NOT_READ",
        }
    ]
    assert attempt["repairOptions"][0]["gapCode"] == (
        "TIME_FIELD_REF_NOT_READ"
    )
    assert attempt["repairOptions"][0]["type"] == (
        "READ_EXACT_TIME_FIELD_OR_REMOVE_OPTIONAL_HINT"
    )
    assert attempt["repairDirective"]["status"] == "REPAIR_REQUIRED"
    assert attempt["repairDirective"]["repairType"] == "EVIDENCE"
    assert payload["latestGraphRejection"]["code"] == (
        "EXECUTION_GRAPH_INVALID"
    )

    artifact_ref = "/workspace/context/recovery-test.json"
    for message in (
        build_grounded_model_recovery_message(payload, artifact_ref),
        compact_summary_to_reference_only(payload, artifact_ref),
    ):
        summary = json.loads(message.content)["contextRecoverySummary"]
        assert summary["latestUnresolvedContractAttempt"]["attemptId"] == (
            "attempt-unresolved"
        )
        assert summary["latestUnresolvedContractAttempt"]["readNext"][0][
            "refId"
        ] == time_ref
        assert summary["latestGraphRejection"]["nextAction"] == (
            "REVISE_EXECUTION_GRAPH_FROM_CURRENT_DISCOVERY"
        )


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

    committed_receipt = dict(artifact.run_result.merged_query_bundle.runtime_events[0]["resultArtifact"])
    artifact.publication_status = "PUBLISHED"
    artifact.result_artifact_receipts = [committed_receipt]
    artifact.ledger_fingerprint = verified_query_artifact_integrity_fingerprint(artifact)
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


def test_complete_goal_coverage_exposes_only_final_answer_tool(
    tmp_path: Path,
) -> None:
    session = _session(_settings(tmp_path))
    session.runtime.phase = "EXECUTED"
    session.runtime.active_contract = None
    session.query_branch_contexts = {"node": SimpleNamespace(status="EXECUTED")}
    tools = [
        SimpleNamespace(name=name)
        for name in (
            "ls",
            "read_file",
            "grep",
            "retrieve_knowledge",
            "finalize_evidence_collection",
            "compose_verified_answer",
        )
    ]

    visible, _ = _phase_visible_tools(session, tools)
    names = {item.name for item in visible}

    assert names == {"compose_verified_answer"}


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
    assert all(item.startswith("merchant://") for item in response.data_sections[0].offloaded_files)
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
        allowed_artifact_digests={artifact["relativePath"]: artifact["sha256"]},
    )
    scope = GroundedSemanticBackend(object())

    with scope.scope(session):
        listing = backend.ls("/artifacts/query_results")
        assert [item.path for item in listing.entries] == ["/query_results/large.txt"]
        assert not backend.glob(
            "*/unverified.txt",
            "/artifacts",
        ).matches
        assert not backend.grep(
            "must remain hidden",
            "/artifacts",
        ).matches
        assert backend.read("/artifacts/query_results/unverified.txt").error == "GROUNDED_CONTEXT_FILE_NOT_ALLOWED"

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
