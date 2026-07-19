from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    ClarificationRequest,
    DataSnapshotContract,
    ExtractedKeywords,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    RecallBundle,
    RecallItem,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_conversation_state import (
    GroundedConversationResolution,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedContextManagementMiddleware,
    GroundedCoreToolBoundaryMiddleware,
    GroundedRuntimeBudgetMiddleware,
    GroundedSemanticBackend,
    _core_visible_binding_hints,
    _grounded_contract_sql_obligations,
    _grounded_semantic_read_control,
    _parallel_goal_dependency_issues,
    _phase_visible_tools,
    _skill_output_contract_issues,
    _thin_recall,
)
from merchant_ai.services.grounded_answer_coverage import answer_fingerprint
from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedContractGap,
    GroundedEntityFilterBinding,
    GroundedEntityFilterHint,
    GroundedQueryContract,
    GroundedUpstreamEntityBinding,
    GroundedUpstreamEntityHint,
)
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeAttempt,
    GroundedRuntimeSession,
    GroundedRuntimeSqlCandidateAttempt,
    GroundedVerifiedQueryArtifact,
    verified_query_artifact_integrity_fingerprint,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    DependencyQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    TimeWindowQuestionGoal,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.grounded_subagent_runtime import IsolatedSubagentResult


def _stable_test_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _publish_skill_test_artifact(
    settings: Settings,
    runtime: GroundedRuntimeSession,
    *,
    thread_id: str,
    run_id: str,
) -> GroundedContextWorkspace:
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id=thread_id,
        run_id=run_id,
        merchant_id=runtime.merchant_id,
        access_role=runtime.access_role,
        user_scope=runtime.user_scope,
        question=runtime.question,
    )
    store = WorkspaceArtifactStore(settings, workspace.artifacts_root)
    rows = list(runtime.answer_run_result.merged_query_bundle.rows)
    rows_artifact = store.write_json(
        "query_results",
        "skill-test.rows.json",
        rows,
        preview_chars=0,
        immutable=True,
    )
    sql_text = "SELECT governed_columns FROM verified_source"
    sql_artifact = store.write_text(
        "query_results",
        "skill-test.sql",
        sql_text,
        preview_chars=0,
        immutable=True,
    )
    attempt_id = "skill-test-attempt"
    generation = 1
    contract = runtime.active_contract
    assert contract is not None
    contract_fingerprint = grounded_query_contract_fingerprint(contract)
    sql_fingerprint = hashlib.sha256(b"skill-test-sql-evidence").hexdigest()
    semantic_fingerprint = hashlib.sha256(b"skill-test-semantic-activation").hexdigest()
    datasource_fingerprint = hashlib.sha256(b"skill-test-datasource").hexdigest()
    verified = runtime.answer_verified_evidence
    assert verified is not None and verified.passed
    verified_payload = verified.model_dump(by_alias=True, mode="json")
    verified_sha256 = _stable_test_hash(verified_payload)
    snapshot_contract = DataSnapshotContract(
        datasource_fingerprint=datasource_fingerprint,
        datasource_environment="test",
        consistency_mode="UNSUPPORTED",
        semantic_activation_fingerprint=semantic_fingerprint,
        cache_generation="test-generation",
        captured_at="2026-07-19T00:00:00Z",
        unsupported_reason="TEST_DATASOURCE_SNAPSHOT_UNAVAILABLE",
    )
    runtime.answer_run_result.merged_query_bundle.data_snapshot = snapshot_contract.model_copy(deep=True)
    data_snapshot = snapshot_contract.model_dump(
        by_alias=True,
        mode="json",
    )
    artifact_fingerprint = hashlib.sha256(("%s:%s" % (thread_id, run_id)).encode("utf-8")).hexdigest()
    manifest_payload = {
        "schemaVersion": 2,
        "artifactKind": "GROUNDED_QUERY_RESULT",
        "publicationStatus": "VERIFIED",
        "artifactFingerprint": artifact_fingerprint,
        "executionGeneration": generation,
        "executionAttemptId": attempt_id,
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_fingerprint,
        "sqlSha256": sql_artifact["sha256"],
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": semantic_fingerprint,
        "dataSnapshot": data_snapshot,
        "resultCoverage": "ALL_ROWS",
        "resultIsTruncated": False,
        "storedRowCount": len(rows),
        "exactResultRowCount": len(rows),
        "verifiedEvidence": verified_payload,
        "verifiedEvidenceSha256": verified_sha256,
        "rowsArtifact": {
            key: rows_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
        "sqlArtifact": {
            key: sql_artifact[key]
            for key in (
                "relativePath",
                "merchantUri",
                "sha256",
                "contentAddress",
                "bytes",
            )
        },
    }
    manifest_artifact = store.write_json(
        "query_results",
        "skill-test.manifest.json",
        manifest_payload,
        preview_chars=0,
        immutable=True,
    )
    receipt = {
        "artifactFingerprint": artifact_fingerprint,
        "queryManifestSha256": manifest_artifact["sha256"],
        "rowsSha256": rows_artifact["sha256"],
        "sqlSha256": sql_artifact["sha256"],
        "manifestContentAddress": manifest_artifact["contentAddress"],
        "rowsContentAddress": rows_artifact["contentAddress"],
        "sqlContentAddress": sql_artifact["contentAddress"],
        "manifestRelativePath": manifest_artifact["relativePath"],
        "rowsRelativePath": rows_artifact["relativePath"],
        "sqlRelativePath": sql_artifact["relativePath"],
        "manifestRef": manifest_artifact["merchantUri"],
        "rowsRef": rows_artifact["merchantUri"],
        "sqlRef": sql_artifact["merchantUri"],
        "storedRowCount": len(rows),
        "exactResultRowCount": len(rows),
        "resultCoverage": "ALL_ROWS",
        "resultIsTruncated": False,
        "executionGeneration": generation,
        "attemptFingerprint": hashlib.sha256(attempt_id.encode("utf-8")).hexdigest(),
        "contractFingerprint": contract_fingerprint,
        "sqlEvidenceFingerprint": sql_fingerprint,
        "contextOwnerFingerprint": workspace.owner_fingerprint,
        "semanticActivationFingerprint": semantic_fingerprint,
        "dataSnapshotFingerprint": _stable_test_hash(data_snapshot),
        "verifiedEvidenceSha256": verified_sha256,
    }
    artifact_id = "query-artifact-%s" % artifact_fingerprint[:16]
    published_artifact = GroundedVerifiedQueryArtifact(
        artifact_id=artifact_id,
        generation=generation,
        attempt_id=attempt_id,
        contract_fingerprint=contract_fingerprint,
        sql_fingerprint=sql_fingerprint,
        contract=contract.model_copy(deep=True),
        plan=runtime.answer_plan.model_copy(deep=True),
        run_result=runtime.answer_run_result.model_copy(deep=True),
        verified_evidence=verified.model_copy(deep=True),
        publication_status="PUBLISHED",
        result_artifact_receipts=[receipt],
        output_columns=[str(key) for row in rows for key in row if not str(key).startswith("__")],
    )
    published_artifact.ledger_fingerprint = verified_query_artifact_integrity_fingerprint(published_artifact)
    runtime.verified_query_ledger = [published_artifact]
    runtime.answer_artifact_ids = [artifact_id]
    return workspace


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

    def propose_contract(
        self, session: GroundedRuntimeSession, evidence: list[dict[str, Any]], hints: dict[str, Any], **kwargs: Any
    ) -> GroundedRuntimeAttempt:
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

    def activate_contract(self, session: GroundedRuntimeSession, attempt_id: str) -> GroundedRuntimeAttempt:
        self.compile_calls += 1
        attempt = session.attempts[-1]
        is_core_sql = attempt.execution_mode == "CORE_SQL_REQUIRED"
        attempt.compile_status = "NOT_APPLICABLE_CORE_SQL_REQUIRED" if is_core_sql else "VALID"
        attempt.activation_status = "ACTIVATED"
        if is_core_sql:
            attempt.next_action = "SUBMIT_GROUNDED_SQL_CANDIDATE"
        else:
            attempt.execution_mode = "DETERMINISTIC_METRIC"
            attempt.execution_reason_codes = ["SINGLE_METRIC_FAST_PATH_ELIGIBLE"]
            attempt.fast_path_eligible = True
            attempt.next_action = "EXECUTE_GROUNDED_QUERY"
        attempt.activated = True
        session.active_generation = max(1, session.active_generation)
        attempt.active_generation = session.active_generation
        return attempt

    @staticmethod
    def submit_sql_candidate(
        session: GroundedRuntimeSession,
        sql: str,
        **kwargs: Any,
    ) -> GroundedRuntimeSqlCandidateAttempt:
        assert "SELECT" in sql.upper()
        assert kwargs["rationale"]
        return GroundedRuntimeSqlCandidateAttempt(
            candidate_id="sql-1",
            active_generation=1,
            status="ACCEPTED",
            next_action="EXECUTE_GROUNDED_QUERY",
            ast_fingerprint="a" * 64,
            contract_fingerprint="c" * 64,
            output_columns=["ticket_count"],
        )

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


class StandaloneConversationAuthority:
    def resolve(
        self,
        question: str,
        **kwargs: Any,
    ) -> GroundedConversationResolution:
        del kwargs
        normalized = str(question or "").strip()
        return GroundedConversationResolution(
            original_question=normalized,
            effective_question=normalized,
            status="STANDALONE",
            source="TEST_STRUCTURED_CONVERSATION_REVIEW",
        )


class ClarifyingConversationAuthority:
    def resolve(
        self,
        question: str,
        **kwargs: Any,
    ) -> GroundedConversationResolution:
        del kwargs
        normalized = str(question or "").strip()
        return GroundedConversationResolution(
            original_question=normalized,
            effective_question=normalized,
            status="SEMANTIC_REVIEW_UNAVAILABLE",
            source="TEST_STRUCTURED_CONVERSATION_REVIEW",
            clarification_question="请明确是否沿用上一轮结果。",
            clarification_type=("CONVERSATION_SEMANTIC_REVIEW_UNAVAILABLE"),
        )


def runtime(factory: CapturingFactory, kernel: FakeKernel | None = None) -> GroundedDeepAgentRuntime:
    return GroundedDeepAgentRuntime(
        kernel or FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=factory,
        conversation_online_authority=StandaloneConversationAuthority(),
    )


def declare_single_metric_goal(
    tools: dict[str, Any],
    context: GroundedDeepAgentRunContext,
    *,
    goal_id: str = "metric.primary",
    label: str = "主指标",
) -> None:
    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id=goal_id,
                        label=label,
                    )
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert result["status"] == "ACCEPTED"


def test_runtime_source_has_no_legacy_or_action_catalog_dependencies() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "merchant_ai/services/grounded_deep_agent_runtime.py"
    )
    source = source_path.read_text(encoding="utf-8")
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


def test_initialization_keeps_skill_bodies_out_of_parent_core() -> None:
    factory = CapturingFactory()
    runtime(factory)

    assert {item.name for item in factory.kwargs["tools"]} == {
        "declare_original_question_goals",
        "propose_grounded_execution_graph",
        "reopen_grounded_execution_graph_discovery",
        "revise_grounded_execution_graph",
        "retrieve_knowledge",
        "publish_verified_rule_evidence",
        "compose_verified_rule_answer",
        "propose_grounded_contract",
        "prepare_grounded_query_batch",
        "submit_grounded_sql_candidate",
        "execute_grounded_query",
        "execute_grounded_query_batch",
        "publish_verified_entity_set",
        "delegate_grounded_exploration",
        "finalize_evidence_collection",
        "compose_verified_answer",
        "run_skill",
        "ask_human",
    }

    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="visibility",
            question="主指标是多少",
            merchant_id="merchant-1",
            workspace_topics=["客服工单"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="主指标是多少",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.primary",
                    label="主指标",
                )
            ],
        ),
    )
    visible, _ = _phase_visible_tools(
        session,
        factory.kwargs["tools"],
    )
    visible_names = {item.name for item in visible}
    assert "propose_grounded_execution_graph" in visible_names
    assert "declare_grounded_query_branches" not in visible_names
    assert "declare_grounded_query_branches" not in {item.name for item in factory.kwargs["tools"]}
    contract_tool = next(item for item in factory.kwargs["tools"] if item.name == "propose_grounded_contract")
    contract_schema = json.dumps(
        contract_tool.tool_call_schema.model_json_schema(),
        ensure_ascii=False,
    )
    assert "tableRefs" in contract_schema
    assert "metricRefs" in contract_schema
    assert "timeExpression" in contract_schema
    compose_tool = next(item for item in factory.kwargs["tools"] if item.name == "compose_verified_answer")
    assert compose_tool.return_direct is True
    assert factory.kwargs["backend"] is not None
    assert [item.name for item in factory.kwargs["middleware"]] == [
        "GroundedContextManagementMiddleware",
        "GroundedRuntimeBudgetMiddleware",
        "GroundedCoreToolBoundaryMiddleware",
    ]
    assert isinstance(
        factory.kwargs["middleware"][0],
        GroundedContextManagementMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][1],
        GroundedRuntimeBudgetMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][2],
        GroundedCoreToolBoundaryMiddleware,
    )
    assert [item["name"] for item in factory.kwargs["subagents"]] == ["general-purpose"]
    assert factory.kwargs["subagents"][0]["tools"] == []
    assert factory.kwargs["subagents"][0]["skills"] is None
    assert factory.kwargs["skills"] is None
    assert "execute SQL" in factory.kwargs["subagents"][0]["system_prompt"]
    for deterministic_mode in (
        "DETERMINISTIC_METRIC",
        "DETERMINISTIC_MULTI_METRIC",
        "DETERMINISTIC_GROUPED",
        "DETERMINISTIC_TREND",
        "DETERMINISTIC_RANKED",
        "DETERMINISTIC_ENTITY_LOOKUP",
    ):
        assert deterministic_mode in factory.kwargs["system_prompt"]
    assert "never plan goals or impose an execution order" in factory.kwargs["system_prompt"]


def test_discovery_read_control_never_uses_fixed_leaf_counts_to_freeze() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="discovery",
            question="分析当前问题",
            merchant_id="merchant-1",
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:topic:table:detail",
                "kind": "TABLE_DETAIL",
                "topic": "topic",
                "contentHash": "table-hash",
                "contentComplete": True,
            },
            {
                "refId": "semantic:topic:table:metric:value",
                "kind": "METRIC",
                "topic": "topic",
                "contentHash": "metric-hash",
                "contentComplete": True,
            },
            {
                "refId": "semantic:topic:table:column:id",
                "kind": "COLUMN",
                "topic": "topic",
                "contentHash": "column-hash",
                "contentComplete": True,
            },
        ],
    )

    control = _grounded_semantic_read_control(session)

    assert control["status"] == "DISCOVERY_OPEN"
    assert control["retrievalClosed"] is False
    assert control["nextAction"] == ("CONTINUE_DISCOVERY_OR_PROPOSE_CONTRACT_OR_GRAPH")


def test_discovery_evidence_is_never_silently_fifo_evicted() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="evidence-ledger",
            question="分析当前问题",
            merchant_id="merchant-1",
        )
    )
    backend = GroundedSemanticBackend(FakeSemanticCatalog())

    for index in range(70):
        backend._retain_evidence(
            session,
            {
                "refId": "semantic:topic:table:field:%s" % index,
                "path": "topics/topic/table/fields/%s.json" % index,
                "kind": "COLUMN",
                "topic": "topic",
                "contentHash": "hash-%s" % index,
                "contentComplete": True,
            },
        )

    assert len(session.core_semantic_evidence) == 70
    assert session.core_semantic_evidence[0]["refId"] == ("semantic:topic:table:field:0")


def test_upstream_entity_values_are_hidden_from_parent_core_contract_response() -> None:
    target_ref = "semantic:商品管理:goods:field:spu_id"
    contract = GroundedQueryContract(
        question="查看该商品",
        status="READY",
        binding_hints=GroundedBindingHints(
            entity_filters=[
                GroundedEntityFilterHint(
                    field_ref=target_ref,
                    operator="IN",
                    literal_value=["secret-spu-1", "secret-spu-2"],
                )
            ],
            upstream_entity_bindings=[
                GroundedUpstreamEntityHint(
                    entity_set_artifact_id="entity_set_1",
                    target_field_ref=target_ref,
                )
            ],
        ),
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id=target_ref,
                topic="商品管理",
                table="goods",
                column="spu_id",
                operator="IN",
                literal_value=["secret-spu-1", "secret-spu-2"],
                entity_identity="entity:product",
                allowed_operators=["IN"],
            )
        ],
        upstream_entity_bindings=[
            GroundedUpstreamEntityBinding(
                entity_set_artifact_id="entity_set_1",
                source_query_artifact_id="query_artifact_1",
                source_column="spu_id",
                source_semantic_ref_id="semantic:tickets:field:spu_id",
                source_entity_identity="entity:product",
                target_field_ref=target_ref,
                target_table="goods",
                target_column="spu_id",
                target_entity_identity="entity:product",
                value_count=2,
                values_hash="sealed-values-hash",
            )
        ],
    )

    payload = {
        "acceptedBindingHints": _core_visible_binding_hints(contract),
        "sqlObligations": _grounded_contract_sql_obligations(contract),
    }
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "secret-spu-1" not in encoded
    assert "secret-spu-2" not in encoded
    assert payload["sqlObligations"]["entityFilters"][0]["runtimeInjected"] is True
    assert payload["sqlObligations"]["entityFilters"][0]["entitySetArtifactId"] == ("entity_set_1")


def test_run_bootstraps_topic_and_scoped_recall_into_first_core_context() -> None:
    factory = CapturingFactory()
    kernel = FakeKernel()
    outer = runtime(factory, kernel)

    response = outer.run("工单量最多的商品", "m-1")

    assert kernel.route_calls == 1
    assert kernel.recall_queries == ["工单量最多的商品"]
    first = json.loads(factory.graph.invocations[0]["messages"][0]["content"])
    assert first["userInputRequirements"]["explicitTimeExpression"] is False
    assert first["trustedExecutionScope"]["merchantScopeBound"] is True
    assert first["trustedExecutionScope"]["merchantId"] == "m-1"
    assert "automatically binds" in first["trustedExecutionScope"]["tenantFilterPolicy"]
    assert first["topicL0Manifests"][0]["topic"] == "客服工单"
    assert first["thinRecallCandidates"][0]["refId"] == "semantic:客服工单:tickets:detail"
    goal_policy = first["originalQuestionGoalPolicy"]
    assert goal_policy["queryTopologyDecision"] == ("LATE_BOUND_AFTER_FORMAL_EVIDENCE")
    assert goal_policy["executionGraphFreezePoint"] == ("IMMEDIATELY_BEFORE_QUERY_PREPARATION")
    assert "branchDeclarationRequiredBeforeRetrieval" not in goal_policy
    assert "Do not freeze query branches" in first["instructions"]
    assert "availableSkillHeaders" not in first
    assert first["analysisSkillPolicy"] == {
        "lifecyclePhase": "post_query_analysis",
        "requiresGroundedContract": True,
        "requiresExecutedQuery": True,
        "requiresVerifiedEvidence": True,
        "mayInfluenceSemanticBindings": False,
        "mayExecuteSql": False,
        "headersDisclosedAfterEvidenceFinalizationOnly": True,
    }
    assert response.clarification is not None
    assert response.debug_trace["harness"]["legacyFallbackUsed"] is False


def test_conversation_typed_clarification_stops_before_route_and_core() -> None:
    factory = CapturingFactory()
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    outer.conversation_online_authority = ClarifyingConversationAuthority()

    response = outer.run("这里面退款最多的三单", "m-1")

    assert kernel.route_calls == 0
    assert kernel.recall_queries == []
    assert factory.graph.invocations == []
    assert response.clarification is not None
    assert response.clarification.type == ("CONVERSATION_SEMANTIC_REVIEW_UNAVAILABLE")


def test_provider_timeout_is_returned_as_controlled_operational_failure() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory, FakeKernel())

    class TimeoutGraph:
        @staticmethod
        def invoke(payload: dict[str, Any], *, config: Any, context: Any) -> None:
            del payload, config, context
            raise TimeoutError("provider read operation timed out")

    outer.deep_agent_graph = TimeoutGraph()
    response = outer.run("工单量", "m-1")
    failure = response.debug_trace["harness"]["operationalFailure"]

    assert failure["code"] == "GROUNDED_PROVIDER_TIMEOUT"
    assert failure["retryable"] is True
    assert "模型调用时限" in response.answer


def test_post_answer_tail_timeout_cannot_overwrite_verified_answer() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory, FakeKernel())

    class AnswerThenTimeoutGraph:
        @staticmethod
        def invoke(payload: dict[str, Any], *, config: Any, context: Any) -> None:
            del payload, config
            context.session.runtime.answer = "已验证答案"
            context.session.answer_coverage_result = {
                "passed": True,
                "source": "compose_verified_answer",
                "answerFingerprint": answer_fingerprint("已验证答案"),
            }
            raise TimeoutError("unnecessary tail turn timed out")

    outer.deep_agent_graph = AnswerThenTimeoutGraph()
    response = outer.run("工单量", "m-1")

    assert response.answer == "已验证答案"
    assert "operationalFailure" not in response.debug_trace["harness"]


def test_plain_core_answer_cannot_bypass_return_direct_compose_attestation() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory, FakeKernel())

    class PlainAnswerGraph:
        @staticmethod
        def invoke(payload: dict[str, Any], *, config: Any, context: Any) -> None:
            del payload, config
            # Simulates an ordinary AIMessage/free-text tail trying to become
            # final without compose_verified_answer.
            context.session.runtime.answer = "我直接声称这是最终答案。"

    outer.deep_agent_graph = PlainAnswerGraph()
    with pytest.raises(RuntimeError) as exc_info:
        outer.run("工单量", "m-1")
    assert "without a matching answer-coverage" in str(exc_info.value)


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


def test_topic_expansion_is_driven_by_structured_search_scope_not_gap_code() -> None:
    relationship_ref = "semantic:alpha:relationship:edge_one"
    runtime_state = GroundedRuntimeSession(
        session_id="s-expand",
        question="cross topic detail",
        merchant_id="m-1",
        workspace_topics=["alpha"],
        attempts=[
            GroundedRuntimeAttempt(
                attempt_id="attempt-1",
                contract=GroundedQueryContract(
                    question="cross topic detail",
                    topics=["alpha"],
                    status="REVISE_BINDINGS",
                    unresolved_gaps=[
                        GroundedContractGap(
                            code="ARBITRARY_CAPABILITY_GAP",
                            message="endpoint is outside current workspace",
                            topic="alpha",
                            resolution="REVISE_BINDINGS",
                            search_scope=("READ_BINDINGS_THEN_TABLE_MANIFEST_THEN_TOPIC_INDEX"),
                            required_capability={
                                "relationshipRef": relationship_ref,
                                "endpointTable": "beta_detail",
                            },
                        )
                    ],
                ),
            )
        ],
    )
    session = GroundedDeepAgentSession(
        runtime=runtime_state,
        core_semantic_evidence=[
            {
                "refId": relationship_ref,
                "topic": "alpha",
                "kind": "RELATIONSHIP",
            }
        ],
    )

    assert session.can_expand_topic() is True
    session.mark_topic_expanded()
    assert session.can_expand_topic() is False


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
    assert [item["refId"] for item in session.core_semantic_evidence] == ["semantic:客服工单:tickets:detail"]


def test_core_read_uses_same_successful_catalog_receipt_without_second_read() -> None:
    class OneShotCatalog(FakeSemanticCatalog):
        calls = 0

        def read(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls > 1:
                return {"success": False, "error": "TRANSIENT_FAILURE"}
            return super().read(**kwargs)

    catalog = OneShotCatalog()
    backend = GroundedSemanticBackend(
        catalog,
        reader_is_core=lambda: False,
    )
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="receipt-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="receipt-thread",
        run_id="receipt-run",
        session=session,
    )
    request = SimpleNamespace(
        tool_call={
            "id": "call-receipt",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/客服工单/tables/tickets/detail.json",
                "offset": 0,
                "limit": 2000,
            },
        },
        runtime=SimpleNamespace(context=context),
    )

    def handler(_: Any) -> ToolMessage:
        read_result = backend.read(
            "/topics/客服工单/tables/tickets/detail.json",
            0,
            2000,
        )
        assert read_result.error is None
        return ToolMessage(
            content="file content",
            tool_call_id="call-receipt",
            name="read_file",
        )

    with backend.scope(session):
        result = middleware.wrap_tool_call(request, handler)

    assert result.status == "success"
    assert catalog.calls == 1
    assert [item["refId"] for item in session.core_semantic_evidence] == ["semantic:客服工单:tickets:detail"]


def test_duplicate_complete_semantic_read_returns_receipt_without_backend_call() -> None:
    backend = GroundedSemanticBackend(FakeSemanticCatalog())
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="duplicate-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:客服工单:tickets:detail",
                "path": "topics/客服工单/tables/tickets/detail.json",
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "contentSnippet": '{"tableName":"tickets"}',
                "contentHash": "hash",
                "contentComplete": True,
            }
        ],
    )
    request = SimpleNamespace(
        tool_call={
            "id": "call-duplicate",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/客服工单/tables/tickets/detail.json",
                "offset": 0,
                "limit": 2000,
            },
        },
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="duplicate-thread",
                run_id="duplicate-run",
                session=session,
            )
        ),
    )
    called = {"handler": False}

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(content="unexpected", tool_call_id="call-duplicate")

    result = middleware.wrap_tool_call(request, handler)
    payload = json.loads(str(result.content))

    assert called["handler"] is False
    assert payload["status"] == "ALREADY_READ"
    assert payload["receipt"]["refId"] == "semantic:客服工单:tickets:detail"


def test_ready_contract_closes_semantic_read_boundary() -> None:
    backend = GroundedSemanticBackend(FakeSemanticCatalog())
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    contract = GroundedQueryContract(
        question="工单量",
        status="READY",
        query_shape="RANKED",
    )
    runtime_state = GroundedRuntimeSession(
        session_id="ready-session",
        question="工单量",
        merchant_id="m-1",
        workspace_topics=["客服工单"],
        phase="ACTIVE_COMPILED",
        active_generation=1,
        active_attempt_id="attempt-ready",
        active_execution_mode=GroundedExecutionMode.DETERMINISTIC_RANKED,
        active_contract=contract,
    )
    session = GroundedDeepAgentSession(runtime=runtime_state)
    request = SimpleNamespace(
        tool_call={
            "id": "call-after-ready",
            "name": "read_file",
            "args": {"file_path": "/knowledge/topics/客服工单/tables/tickets/detail.json"},
        },
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="ready-thread",
                run_id="ready-run",
                session=session,
            )
        ),
    )
    called = {"handler": False}

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(content="unexpected", tool_call_id="call-after-ready")

    result = middleware.wrap_tool_call(request, handler)
    payload = json.loads(str(result.content))

    assert called["handler"] is False
    assert result.status == "error"
    assert payload["code"] == "GROUNDED_CONTRACT_READY"
    assert payload["readControl"]["status"] == "READY_TO_EXECUTE"


def test_large_semantic_read_is_offloaded_but_full_evidence_is_retained() -> None:
    large_content = json.dumps(
        {
            "topic": "客服工单",
            "tableName": "tickets",
            "section": "columns",
            "items": [{"key": "field_%04d" % index, "description": "x" * 80} for index in range(300)],
        },
        ensure_ascii=False,
    )

    class LargeCatalog(FakeSemanticCatalog):
        def read(self, *, path: str, max_chars: int, offset: int) -> dict[str, Any]:
            del max_chars, offset
            return {
                "success": True,
                "refId": "semantic:客服工单:tickets:columns",
                "path": path,
                "kind": "COLUMN_INDEX",
                "topic": "客服工单",
                "table": "tickets",
                "content": large_content,
            }

    backend = GroundedSemanticBackend(LargeCatalog())
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="large-session",
            question="工单字段",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="large-thread",
        run_id="large-run",
        session=session,
    )
    request = SimpleNamespace(
        tool_call={
            "id": "call-large",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/客服工单/tables/tickets/columns/index.json",
                "offset": 0,
                "limit": 2000,
            },
        },
        runtime=SimpleNamespace(context=context),
    )

    def handler(_: Any) -> ToolMessage:
        read_result = backend.read(
            "/topics/客服工单/tables/tickets/columns/index.json",
            0,
            2000,
        )
        assert read_result.error is None
        return ToolMessage(
            content=large_content,
            tool_call_id="call-large",
            name="read_file",
        )

    with backend.scope(session):
        result = middleware.wrap_tool_call(request, handler)
    payload = json.loads(str(result.content))

    assert len(large_content) > middleware.MAX_INLINE_READ_CHARS
    assert len(str(result.content)) < 3_000
    assert payload["status"] == "TOOL_RESULT_OFFLOADED"
    assert payload["receipt"]["refId"] == "semantic:客服工单:tickets:columns"
    assert len(session.core_semantic_evidence[0]["contentSnippet"]) == len(large_content)


def test_catalog_pagination_is_rejected_in_favor_of_targeted_grep() -> None:
    backend = GroundedSemanticBackend(FakeSemanticCatalog())
    middleware = GroundedCoreToolBoundaryMiddleware(backend)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="catalog-page-session",
            question="品牌字段",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    request = SimpleNamespace(
        tool_call={
            "id": "call-catalog-page",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/客服工单/tables/tickets/columns/index.json",
                "offset": 200,
                "limit": 200,
            },
        },
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="catalog-page-thread",
                run_id="catalog-page-run",
                session=session,
            )
        ),
    )
    called = {"handler": False}

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(content="unexpected", tool_call_id="call-catalog-page")

    result = middleware.wrap_tool_call(request, handler)
    payload = json.loads(str(result.content))

    assert called["handler"] is False
    assert result.status == "error"
    assert payload["code"] == "PAGINATED_CATALOG_SCAN_DENIED"
    assert "grep" in payload["message"]


def test_context_middleware_keeps_messages_below_watermark_and_hides_retrieval_tools_when_ready() -> None:
    contract = GroundedQueryContract(
        question="工单量",
        status="READY",
        query_shape="RANKED",
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="context-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
            phase="ACTIVE_COMPILED",
            active_generation=1,
            active_attempt_id="attempt-context",
            active_execution_mode=GroundedExecutionMode.DETERMINISTIC_RANKED,
            active_contract=contract,
        )
    )
    messages = [
        HumanMessage(content=json.dumps({"question": "工单量", "context": "x" * 2000})),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "read-old",
                    "name": "read_file",
                    "args": {"file_path": "/knowledge/old.json", "limit": 2000},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="z" * 20_000,
            tool_call_id="read-old",
            name="read_file",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-execute",
                    "name": "execute_grounded_query",
                    "args": {"reason": "ready"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"status":"VERIFIED"}',
            tool_call_id="call-execute",
            name="execute_grounded_query",
        ),
    ]
    tools = [
        SimpleNamespace(name="read_file", description="read", args_schema=None),
        SimpleNamespace(name="grep", description="grep", args_schema=None),
        SimpleNamespace(
            name="execute_grounded_query",
            description="execute",
            args_schema=None,
        ),
        SimpleNamespace(
            name="submit_grounded_sql_candidate",
            description="submit",
            args_schema=None,
        ),
    ]
    request = SimpleNamespace(
        messages=messages,
        tools=tools,
        system_message=HumanMessage(content="system"),
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="context-thread",
                run_id="context-run",
                session=session,
            )
        ),
    )

    def override(**updates: Any) -> Any:
        values = dict(request.__dict__)
        values.update(updates)
        return SimpleNamespace(**values)

    request.override = override
    captured: dict[str, Any] = {}

    def handler(updated_request: Any) -> str:
        captured["request"] = updated_request
        return "ok"

    result = GroundedContextManagementMiddleware().wrap_model_call(request, handler)
    updated = captured["request"]

    assert result == "ok"
    assert updated.messages == messages
    assert str(updated.messages[4].content) == '{"status":"VERIFIED"}'
    assert [_tool.name for _tool in updated.tools] == ["execute_grounded_query"]
    report = session.core_context_reports[-1]
    assert report["compactionTriggered"] is False
    assert report["savedChars"] == 0
    assert report["semanticReadMessagesCompacted"] == 0
    assert report["removedTools"] == [
        "grep",
        "read_file",
        "submit_grounded_sql_candidate",
    ]


def test_context_middleware_exposes_only_goal_transaction_on_first_model_turn() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="first-turn-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    tools = [
        SimpleNamespace(
            name=name,
            description=name,
            args_schema=None,
        )
        for name in (
            "declare_original_question_goals",
            "read_file",
            "grep",
            "propose_grounded_contract",
            "execute_grounded_query",
            "compose_verified_answer",
            "ask_human",
            "task",
            "write_todos",
        )
    ]
    request = SimpleNamespace(
        messages=[HumanMessage(content='{"question":"工单量"}')],
        tools=tools,
        system_message=HumanMessage(content="system"),
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="first-turn-thread",
                run_id="first-turn-run",
                session=session,
            )
        ),
    )

    def override(**updates: Any) -> Any:
        values = dict(request.__dict__)
        values.update(updates)
        return SimpleNamespace(**values)

    request.override = override
    captured: dict[str, Any] = {}

    def handler(updated_request: Any) -> str:
        captured["tools"] = updated_request.tools
        return "ok"

    GroundedContextManagementMiddleware().wrap_model_call(request, handler)

    assert [_tool.name for _tool in captured["tools"]] == [
        "declare_original_question_goals",
        "ask_human",
    ]
    assert session.core_context_reports[-1]["toolCountAfter"] == 2


def test_task_dispatch_is_blocked_before_subagent_execution() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
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
            RecallItem(
                doc_id="semantic:客服工单:tickets:metric:ticket_cnt",
                source_type="METRIC",
                topic="客服工单",
                table="tickets",
                metadata={
                    "semanticRefId": "semantic:客服工单:tickets:metric:ticket_cnt",
                    "semanticPath": "topics/客服工单/tables/tickets/asset.json#metric:ticket_cnt",
                    "semanticKind": "METRIC",
                },
            ),
        ]
    )
    thin = _thin_recall(recall, 8)
    assert [item["refId"] for item in thin] == [
        "semantic:客服工单:tickets:detail",
        "semantic:客服工单:tickets:metric:ticket_cnt",
    ]
    assert thin[1]["path"] == "topics/客服工单/tables/tickets/metrics/ticket_cnt.json"


def test_thin_recall_never_leaks_host_paths_and_keeps_rules_inline_only() -> None:
    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="/Users/example/resources/runtime/rules/platform_rules.md#chunk-1",
                source_type="GOVERNED_RULE",
                content="unsafe legacy rule",
            ),
            RecallItem(
                doc_id="semantic:rules:platform_rules:chunk:0001",
                source_type="GOVERNED_RULE",
                content="governed inline rule",
                metadata={"semanticPath": "rules/platform_rules.md"},
            ),
            RecallItem(
                doc_id="/private/tmp/untrusted",
                source_type="SEMANTIC_METRIC",
                metadata={"semanticPath": "/private/tmp/metric.json"},
            ),
        ]
    )

    thin = _thin_recall(recall, 8)

    assert thin == [
        {
            "refId": "semantic:rules:platform_rules:chunk:0001",
            "path": "",
            "kind": "GOVERNED_RULE",
            "topic": "",
            "table": "",
            "title": "",
            "snippet": "governed inline rule",
            "score": 0.0,
            "navigationMode": "INLINE_ONLY",
            "bindingEligible": False,
        }
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
    declare_single_metric_goal(tools, context)

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
            goal_ids=["metric.primary"],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert recall_result["status"] == "OK"
    assert kernel.recall_queries[-1] == "商品字段"
    assert contract_result["activated"] is True
    assert contract_result["activationStatus"] == "ACTIVATED"
    assert contract_result["executionMode"] == "DETERMINISTIC_METRIC"
    assert contract_result["executionReasonCodes"] == ["SINGLE_METRIC_FAST_PATH_ELIGIBLE"]
    assert contract_result["fastPathEligible"] is True
    assert contract_result["fastPathReasonCodes"] == []
    assert contract_result["nextAction"] == "EXECUTE_GROUNDED_QUERY"
    assert contract_result["semanticCoverage"] == {
        "status": "READY_TO_EXECUTE",
        "retrievalClosed": True,
        "blockingGapCount": 0,
        "nextAction": "EXECUTE_GROUNDED_QUERY",
    }
    assert contract_result["activeGeneration"] == 1
    assert len(contract_result["contractFingerprint"]) == 64
    assert "requiredFinalOutputAliases" in contract_result["sqlObligations"]
    assert kernel.propose_calls == 1
    assert kernel.compile_calls == 1


def test_semantic_goal_dependency_does_not_force_query_serialization() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("查询两个指标", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="t-batch-depends-on",
        run_id="r-batch-depends-on",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.top_products",
                        label="TopN 商品",
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="这些商品的退款",
                        depends_on_goal_ids=["metric.top_products"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    assert deep_session.question_goal_contract is not None
    assert (
        _parallel_goal_dependency_issues(
            deep_session.question_goal_contract,
            {
                "top-products": ["metric.top_products"],
                "refunds": ["metric.refunds"],
            },
        )
        == []
    )


def test_parallel_batch_rejects_dependency_goal_endpoints() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("查询商品退款链路", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="t-batch-dependency-goal",
        run_id="r-batch-dependency-goal",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.top_products",
                        label="TopN 商品",
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="这些商品的退款",
                    ),
                    DependencyQuestionGoal(
                        goal_id="dependency.product_refunds",
                        label="TopN 商品到退款的实体依赖",
                        upstream_goal_ids=["metric.top_products"],
                        downstream_goal_ids=["metric.refunds"],
                        artifact_kind="entity_set",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    result = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[
                {"queryId": "top-products", "goalIds": ["metric.top_products"]},
                {"queryId": "refunds", "goalIds": ["metric.refunds"]},
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "REJECTED"
    assert result["code"] == "PARALLEL_GOAL_DEPENDENCY_DETECTED"
    assert result["issues"] == [
        {
            "code": "BATCH_DEPENDENCY_GOAL_EDGE",
            "upstreamGoalId": "metric.top_products",
            "downstreamGoalId": "metric.refunds",
            "upstreamQueryIds": ["top-products"],
            "downstreamQueryIds": ["refunds"],
            "requiredExecution": "SERIAL",
            "direct": True,
            "pathGoalIds": ["metric.top_products", "metric.refunds"],
            "pathEdges": [
                {
                    "relationType": "DEPENDENCY_GOAL",
                    "upstreamGoalId": "metric.top_products",
                    "downstreamGoalId": "metric.refunds",
                    "declaredByGoalId": "dependency.product_refunds",
                    "dependencyGoalId": "dependency.product_refunds",
                }
            ],
            "dependencyGoalIds": ["dependency.product_refunds"],
            "dependencyGoalId": "dependency.product_refunds",
            "dependencyGoalQueryIds": [],
        }
    ]
    assert deep_session.parallel_branches == {}


def test_transitive_semantic_goal_dependencies_do_not_create_artifact_waits() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("查询三级依赖结果", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="t-batch-transitive-depends-on",
        run_id="r-batch-transitive-depends-on",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(goal_id="metric.c", label="上游 C"),
                    MetricQuestionGoal(
                        goal_id="metric.b",
                        label="中间 B",
                        depends_on_goal_ids=["metric.c"],
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.a",
                        label="下游 A",
                        depends_on_goal_ids=["metric.b"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    assert deep_session.question_goal_contract is not None
    assert (
        _parallel_goal_dependency_issues(
            deep_session.question_goal_contract,
            {
                "upstream-c": ["metric.c"],
                "downstream-a": ["metric.a"],
            },
        )
        == []
    )


def test_artifact_dependency_does_not_spread_through_semantic_only_edges() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("查询混合依赖结果", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="t-batch-transitive-dependency-goal",
        run_id="r-batch-transitive-dependency-goal",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(goal_id="metric.c", label="上游 C"),
                    MetricQuestionGoal(goal_id="metric.b", label="中间 B"),
                    MetricQuestionGoal(
                        goal_id="metric.a",
                        label="下游 A",
                        depends_on_goal_ids=["metric.b"],
                    ),
                    DependencyQuestionGoal(
                        goal_id="dependency.c_to_b",
                        label="C 到 B 的实体依赖",
                        upstream_goal_ids=["metric.c"],
                        downstream_goal_ids=["metric.b"],
                        artifact_kind="entity_set",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    assert deep_session.question_goal_contract is not None
    assert (
        _parallel_goal_dependency_issues(
            deep_session.question_goal_contract,
            {
                "upstream-c": ["metric.c"],
                "downstream-a": ["metric.a"],
            },
        )
        == []
    )
    assert deep_session.parallel_branches == {}


def test_parallel_batch_allows_mutually_independent_goals() -> None:
    class ParallelKernel(FakeKernel):
        @staticmethod
        def fork_query_branch(
            session: GroundedRuntimeSession,
            branch_key: str,
        ) -> GroundedRuntimeSession:
            return GroundedRuntimeSession(
                session_id=f"{session.session_id}:{branch_key}",
                question=session.question,
                merchant_id=session.merchant_id,
                merchant=session.merchant,
                workspace_topics=list(session.workspace_topics),
            )

    factory = CapturingFactory(action="none")
    kernel = ParallelKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("查询工单量和退款量", "m-1")
    evidence_ref = "semantic:客服工单:tickets:detail"
    deep_session = GroundedDeepAgentSession(
        runtime=kernel_session,
        core_semantic_evidence=[
            {
                "refId": evidence_ref,
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "contentSnippet": "{}",
                "contentHash": "hash",
            }
        ],
    )
    context = GroundedDeepAgentRunContext(
        thread_id="t-batch-independent",
        run_id="r-batch-independent",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(goal_id="metric.tickets", label="工单量"),
                    MetricQuestionGoal(goal_id="metric.refunds", label="退款量"),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"

    result = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[
                {
                    "queryId": "tickets",
                    "readRefIds": [evidence_ref],
                    "goalIds": ["metric.tickets"],
                },
                {
                    "queryId": "refunds",
                    "readRefIds": [evidence_ref],
                    "goalIds": ["metric.refunds"],
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "PREPARED"
    assert result["preparedCount"] == 2
    assert {item["queryId"] for item in result["queries"]} == {
        "tickets",
        "refunds",
    }
    assert all(item["status"] == "PREPARED" for item in result["queries"])
    assert set(deep_session.parallel_branches) == {"tickets", "refunds"}


def test_ready_core_sql_contract_is_activated_without_template_switch() -> None:
    class CoreSqlKernel(FakeKernel):
        def propose_contract(
            self,
            session: GroundedRuntimeSession,
            evidence: list[dict[str, Any]],
            hints: dict[str, Any],
            **kwargs: Any,
        ) -> GroundedRuntimeAttempt:
            attempt = super().propose_contract(session, evidence, hints, **kwargs)
            attempt.execution_mode = "CORE_SQL_REQUIRED"
            attempt.execution_reason_codes = [
                "COMPLEX_QUERY_REQUIRES_CORE_SQL",
                "QUERY_SHAPE_NOT_SCALAR",
            ]
            attempt.fast_path_reason_codes = ["QUERY_SHAPE_NOT_SCALAR"]
            attempt.next_action = "SUBMIT_GROUNDED_SQL_CANDIDATE"
            return attempt

    factory = CapturingFactory(action="none")
    kernel = CoreSqlKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("按商品统计工单量", "m-1")
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
    context = GroundedDeepAgentRunContext(
        thread_id="t-core-sql",
        run_id="r-core-sql",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}
    declare_single_metric_goal(tools, context)

    result = json.loads(
        tools["propose_grounded_contract"].func(
            read_ref_ids=["semantic:客服工单:tickets:detail"],
            binding_hints={"tableRefs": ["semantic:客服工单:tickets:detail"]},
            goal_ids=["metric.primary"],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert kernel.compile_calls == 1
    assert result["activated"] is True
    assert result["executionMode"] == "CORE_SQL_REQUIRED"
    assert result["compileStatus"] == "NOT_APPLICABLE_CORE_SQL_REQUIRED"
    assert result["fastPathReasonCodes"] == ["QUERY_SHAPE_NOT_SCALAR"]
    assert result["nextAction"] == "SUBMIT_GROUNDED_SQL_CANDIDATE"


def test_core_sql_tool_submits_complete_sql_without_template_dispatch() -> None:
    class TransactionalCoreSqlKernel(FakeKernel):
        def __init__(self) -> None:
            super().__init__()
            self.execute_calls = 0
            self.verify_calls = 0

        def execute_active(
            self,
            session: GroundedRuntimeSession,
            **kwargs: Any,
        ) -> AgentRunResult:
            del kwargs
            self.execute_calls += 1
            result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"ticket_count": 3}],
                    tables=["tickets"],
                )
            )
            session.run_result = result
            return result

        def verify_active(
            self,
            session: GroundedRuntimeSession,
        ) -> VerifiedEvidence:
            self.verify_calls += 1
            verified = VerifiedEvidence(passed=True)
            session.verified_evidence = verified
            assert session.run_result is not None
            contract = GroundedQueryContract(
                question=session.question,
                status="READY",
                query_shape="GROUPED",
            )
            session.verified_query_ledger.append(
                GroundedVerifiedQueryArtifact(
                    artifact_id="query_artifact_core_sql",
                    generation=1,
                    contract_fingerprint=grounded_query_contract_fingerprint(contract),
                    sql_fingerprint="a" * 64,
                    contract=contract,
                    plan=QueryPlan(),
                    run_result=session.run_result,
                    verified_evidence=verified,
                    output_columns=["ticket_count"],
                )
            )
            return verified

        @staticmethod
        def latest_verified_query_artifact(
            session: GroundedRuntimeSession,
        ) -> GroundedVerifiedQueryArtifact | None:
            return session.verified_query_ledger[-1] if session.verified_query_ledger else None

    factory = CapturingFactory(action="none")
    kernel = TransactionalCoreSqlKernel()
    outer = runtime(factory, kernel)
    kernel_session = outer.kernel.new_session("按商品统计工单量", "m-1")
    context = GroundedDeepAgentRunContext(
        thread_id="t-submit-core-sql",
        run_id="r-submit-core-sql",
        session=GroundedDeepAgentSession(runtime=kernel_session),
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["submit_grounded_sql_candidate"].func(
            sql="SELECT spu_id, COUNT(*) AS ticket_count FROM tickets GROUP BY spu_id",
            expected_generation=1,
            contract_fingerprint="c" * 64,
            rationale="Contract requires grouped Core SQL",
            evidence_ref_ids=["semantic:客服工单:tickets:detail"],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "VERIFIED"
    assert result["sqlCandidateStatus"] == "ACCEPTED"
    assert result["submittedAndExecuted"] is True
    assert result["nextAction"] == "PUBLISH_ENTITY_SET_OR_CONTINUE_QUERYING_OR_FINALIZE"
    assert result["outputColumns"] == ["ticket_count"]
    assert result["rowCount"] == 1
    assert kernel.execute_calls == 1
    assert kernel.verify_calls == 1


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
    with pytest.raises(RuntimeError) as factory_exc_info:
        runtime(CapturingFactory(fail=True))
    assert "Grounded DeepAgent initialization failed" in str(factory_exc_info.value)

    with pytest.raises(RuntimeError) as model_exc_info:
        GroundedDeepAgentRuntime(
            FakeKernel(),
            lead_model=None,
            semantic_catalog=FakeSemanticCatalog(),
            agent_factory=CapturingFactory(),
        )
    assert "model is not configured" in str(model_exc_info.value)


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
        conversation_online_authority=StandaloneConversationAuthority(),
        agent_factory=factory,
    )

    outer.run("工单量", "m-1", thread_id="t-fixed", run_id="r-fixed")

    assert factory.graph.configs[0]["configurable"] == expected["configurable"]
    assert factory.graph.configs[0]["observed"] == {
        "threadId": "t-fixed",
        "runId": "r-fixed",
    }


def test_run_skill_is_rejected_before_query_execution_and_verification() -> None:
    factory = CapturingFactory(action="none")
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        skill_root="python_backend/resources/runtime/agent_skills",
        agent_factory=factory,
    )
    kernel_session = outer.kernel.new_session("分析最近30天退款率", "m-1")
    context = GroundedDeepAgentRunContext(
        thread_id="thread-pre-query-skill",
        run_id="run-pre-query-skill",
        session=GroundedDeepAgentSession(runtime=kernel_session),
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["run_skill"].func(
            skill_name="refund-rate-diagnosis",
            objective="分析退款率",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "EVIDENCE_COLLECTION_NOT_SEALED"
    assert kernel_session.run_result is None
    assert context.session.skill_runs == []


def test_execute_tool_returns_revise_instruction_instead_of_crashing_core() -> None:
    class CompatibilityBlockedKernel(FakeKernel):
        @staticmethod
        def execute_active(session: GroundedRuntimeSession, **kwargs: Any) -> AgentRunResult:
            raise RuntimeError("grounded metrics have incompatible time selection policies")

    factory = CapturingFactory(action="none")
    outer = runtime(factory, CompatibilityBlockedKernel())
    kernel_session = outer.kernel.new_session("最近30天退款率", "m-1")
    context = GroundedDeepAgentRunContext(
        thread_id="thread-execution-blocked",
        run_id="run-execution-blocked",
        session=GroundedDeepAgentSession(runtime=kernel_session),
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["execute_grounded_query"].func(
            reason="执行退款率查询",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "EXECUTION_REVISE_REQUIRED"
    assert result["nextAction"] == "REVISE_BINDINGS"
    assert "Do not retry the same bindings" in result["instruction"]


def test_skill_headers_are_disclosed_only_after_portfolio_finalization() -> None:
    class VerifiedKernel(FakeKernel):
        @staticmethod
        def execute_active(session: GroundedRuntimeSession, **kwargs: Any) -> AgentRunResult:
            result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"refund_rate": 0.2}],
                    tables=["merchant_profile"],
                )
            )
            session.run_result = result
            return result

        @staticmethod
        def verify_active(session: GroundedRuntimeSession) -> VerifiedEvidence:
            verified = VerifiedEvidence(passed=True)
            session.verified_evidence = verified
            assert session.run_result is not None
            contract = GroundedQueryContract(
                question=session.question,
                status="READY",
                query_shape="SCALAR",
            )
            session.verified_query_ledger.append(
                GroundedVerifiedQueryArtifact(
                    artifact_id="query_artifact_1",
                    generation=1,
                    contract_fingerprint=grounded_query_contract_fingerprint(contract),
                    sql_fingerprint="f" * 64,
                    contract=contract,
                    plan=QueryPlan(),
                    run_result=session.run_result,
                    verified_evidence=verified,
                )
            )
            return verified

        @staticmethod
        def latest_verified_query_artifact(
            session: GroundedRuntimeSession,
        ) -> GroundedVerifiedQueryArtifact | None:
            return session.verified_query_ledger[-1] if session.verified_query_ledger else None

        @staticmethod
        def verify_portfolio(
            session: GroundedRuntimeSession,
        ) -> tuple[QueryPlan, AgentRunResult, VerifiedEvidence, list[str]]:
            assert session.run_result is not None
            assert session.verified_evidence is not None
            return (
                QueryPlan(),
                session.run_result,
                session.verified_evidence,
                ["query_artifact_1"],
            )

    factory = CapturingFactory(action="none")
    outer = GroundedDeepAgentRuntime(
        VerifiedKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        skill_root="python_backend/resources/runtime/agent_skills",
        agent_factory=factory,
    )
    kernel_session = outer.kernel.new_session("最近30天退款率", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="thread-skill-headers",
        run_id="run-skill-headers",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.refund_rate",
                        label="退款率",
                    ),
                    TimeWindowQuestionGoal(
                        goal_id="time.recent_30_days",
                        label="最近30天",
                        time_expression="最近30天",
                        applies_to_goal_ids=["metric.refund_rate"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "ACCEPTED"
    deep_session.active_goal_ids = [
        "metric.refund_rate",
        "time.recent_30_days",
    ]

    result = json.loads(
        tools["execute_grounded_query"].func(
            reason="执行已激活查询",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "VERIFIED"
    assert deep_session.analysis_skill_headers_disclosed is False
    assert "availableAnalysisSkillHeaders" not in result

    finalized = json.loads(
        tools["finalize_evidence_collection"].func(
            reason="原问题所需数据已经齐全",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert finalized["status"] == "EVIDENCE_COLLECTION_SEALED"
    assert deep_session.analysis_skill_headers_disclosed is True
    assert deep_session.data_collection_sealed is True
    assert any(item["name"] == "refund-rate-diagnosis" for item in finalized["availableAnalysisSkillHeaders"])
    assert all(item["lifecyclePhase"] == "post_query_analysis" for item in finalized["availableAnalysisSkillHeaders"])


def test_skill_output_contract_rejects_governed_formula_drift() -> None:
    metric_ref = "semantic:经营画像:profile:metric:refund_rate"
    plan = QueryPlan(
        intents=[
            QuestionIntent(
                metric_specs=[
                    {
                        "semanticRefId": metric_ref,
                        "metricFormula": "SUM(return_cnt) / NULLIF(SUM(pay_order_cnt), 0)",
                    }
                ]
            )
        ]
    )
    structured = {
        "answerMarkdown": "退款率分析。",
        "observations": [],
        "semanticDisclosures": [
            {
                "metricRef": metric_ref,
                "formula": "(SUM(return_cnt) + SUM(direct_refund_cnt)) / SUM(pay_order_cnt)",
            }
        ],
        "derivedFacts": [],
        "hypotheses": [],
        "recommendations": [],
        "evidenceRefs": [],
        "gaps": [],
        "executionConfidence": 0.5,
    }

    issues = _skill_output_contract_issues(structured, plan)

    assert any(item["code"] == "GOVERNED_FORMULA_DRIFT" for item in issues)


def test_run_skill_uses_generic_isolated_subagent_checkpoint_progress_and_artifact(
    tmp_path: Path,
) -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    settings = Settings(harness_workspace_path=str(tmp_path / "runtime"))
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        checkpoint_config_factory=lambda thread_id, run_id: {
            "configurable": {"thread_id": thread_id, "run_id": run_id}
        },
        skill_root="python_backend/resources/runtime/agent_skills",
        skill_run_root=str(tmp_path / "skill-runs"),
        settings=settings,
        agent_factory=factory,
    )

    class FakeIsolatedRuntime:
        job: Any = None

        def run(self, job: Any, *, on_progress: Any) -> IsolatedSubagentResult:
            self.job = job
            on_progress("subagent", "started", job.job_id)
            on_progress("subagent_step", "running", "read_file")
            on_progress("subagent", "completed", "updates=1")
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={
                    "threadId": job.thread_id,
                    "runId": job.job_id,
                    "checkpointNamespace": "",
                },
                raw_output=json.dumps(
                    {
                        "answerMarkdown": "基于已验证证据完成风险分析。",
                        "observations": [],
                        "semanticDisclosures": [],
                        "derivedFacts": [],
                        "hypotheses": [],
                        "recommendations": [],
                        "evidenceRefs": [],
                        "gaps": [],
                        "executionConfidence": 0.88,
                    },
                    ensure_ascii=False,
                ),
                update_count=1,
            )

    isolated = FakeIsolatedRuntime()
    outer.subagent_runtime = isolated
    kernel_session = kernel.new_session("分析最近30天经营风险", "m-1")
    kernel_session.workspace_topics = ["客服工单"]
    kernel_session.active_contract = GroundedQueryContract(
        question=kernel_session.question,
        topics=["客服工单"],
        status="READY",
        query_shape="SCALAR",
    )
    kernel_session.active_plan = QueryPlan()
    kernel_session.run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(rows=[{"risk_value": 1}], tables=["tickets"])
    )
    kernel_session.verified_evidence = VerifiedEvidence(passed=True)
    kernel_session.answer_plan = kernel_session.active_plan
    kernel_session.answer_run_result = kernel_session.run_result
    kernel_session.answer_verified_evidence = kernel_session.verified_evidence
    workspace = _publish_skill_test_artifact(
        settings,
        kernel_session,
        thread_id="thread-skill",
        run_id="run-parent",
    )
    session = GroundedDeepAgentSession(
        runtime=kernel_session,
        context_workspace=workspace,
        analysis_skill_headers_disclosed=True,
        data_collection_sealed=True,
    )
    events: list[tuple[str, str, dict[str, Any]]] = []
    context = GroundedDeepAgentRunContext(
        thread_id="thread-skill",
        run_id="run-parent",
        session=session,
        listener=lambda event_type, node, payload: events.append((event_type, node, payload)),
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["run_skill"].func(
            skill_name="risk-analysis",
            objective="基于已验证证据输出风险优先级",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "SKILL_COMPLETED"
    assert result["executionConfidence"] == 0.88
    assert "artifact" not in result
    assert "workspace" not in result
    skill_workspaces = list(workspace.subagents_root.glob("skill/*"))
    assert len(skill_workspaces) == 1
    assert (skill_workspaces[0] / "result.json").is_file()
    skill_input = json.loads((skill_workspaces[0] / "input.json").read_text(encoding="utf-8"))
    assert "dataRows" not in skill_input
    assert skill_input["verifiedArtifactAccess"]["inlineRows"] is False
    assert "sqlArtifact" not in json.dumps(
        skill_input["verifiedArtifactAccess"],
        ensure_ascii=False,
    )
    assert str(workspace.artifacts_root) not in json.dumps(
        skill_input,
        ensure_ascii=False,
    )
    assert result["checkpoint"]["threadId"].startswith("thread-skill__skill_")
    assert isolated.job.skills == []
    assert isolated.job.user_payload["mountedSkill"] == "/skills/risk-analysis/SKILL.md"
    assert "generic isolated subagent" in isolated.job.system_prompt
    assert [item[0] for item in events] == ["skill.progress"] * len(events)
    assert any(item[2]["stage"] == "subagent_step" for item in events)
    assert kernel_session.answer == "基于已验证证据完成风险分析。"
    assert kernel_session.run_result.skill_lifecycle_records[0].matched_by == ("core_llm_skill_header")


def test_grounded_exploration_subagent_is_advisory_and_mounts_no_capabilities() -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        agent_factory=factory,
    )

    class ExplorationRuntime:
        job: Any = None

        def run(self, job: Any, *, on_progress: Any) -> IsolatedSubagentResult:
            self.job = job
            assignment = job.user_payload["invocation"]["assignment"]
            assignment_id = assignment["assignmentId"]
            assert assignment["objective"] == ("Explore competing explanations from verified evidence.")
            population = assignment["populationScopeFingerprint"]
            time_scope = assignment["timeScopeFingerprint"]
            source_fingerprints = assignment["sourceArtifactFingerprints"]
            on_progress("subagent", "started", job.job_id)
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={"threadId": job.thread_id, "runId": job.job_id},
                raw_output=json.dumps(
                    {
                        "artifactId": "advisory.test",
                        "assignmentId": assignment_id,
                        "hypotheses": [
                            {
                                "hypothesisId": "hypothesis.test",
                                "falsifiableStatement": ("The verified change is concentrated in one comparison."),
                                "premises": ["The verified input is comparable."],
                                "expectedObservations": ["A comparison differs from its reference."],
                                "falsifyingObservations": ["The comparison does not differ."],
                                "goalIds": ["analysis.primary"],
                                "populationScopeFingerprint": population,
                                "timeScopeFingerprint": time_scope,
                                "competingExplanations": ["The change is distributed across comparisons."],
                            }
                        ],
                        "evidenceRequests": [
                            {
                                "requestId": "request.test",
                                "capability": "COMPARE_GROUPS",
                                "evidenceShape": "COMPARISON_RESULT",
                                "goalIds": ["analysis.primary"],
                                "hypothesisIds": ["hypothesis.test"],
                                "populationScope": {
                                    "relation": "INHERIT",
                                    "fingerprint": population,
                                    "parentFingerprint": population,
                                },
                                "timeScope": {
                                    "relation": "INHERIT",
                                    "fingerprint": time_scope,
                                    "parentFingerprint": time_scope,
                                },
                                "sourceArtifactFingerprints": source_fingerprints,
                            }
                        ],
                        "stoppingAssessment": {
                            "decision": "CONTINUE",
                            "goalIds": ["analysis.primary"],
                            "unresolvedHypothesisIds": ["hypothesis.test"],
                            "outstandingRequestIds": ["request.test"],
                            "rationale": "The comparison evidence is still required.",
                        },
                        "sourceArtifactFingerprints": source_fingerprints,
                    },
                    ensure_ascii=False,
                ),
                update_count=1,
            )

    isolated = ExplorationRuntime()
    outer.subagent_runtime = isolated
    kernel_session = kernel.new_session("分析已验证变化", "m-1")
    contract = GroundedQueryContract(
        question=kernel_session.question,
        status="READY",
        query_shape="SCALAR",
    )
    run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(
            rows=[{"measure": 10}],
            original_row_count=1,
        )
    )
    verified = VerifiedEvidence(
        passed=True,
        covered_evidence=["evidence.measure"],
    )
    query_artifact = GroundedVerifiedQueryArtifact(
        artifact_id="artifact.measure",
        generation=1,
        contract_fingerprint="contract.fingerprint",
        sql_fingerprint="sql.fingerprint",
        contract=contract,
        plan=QueryPlan(),
        run_result=run_result,
        verified_evidence=verified,
        output_columns=["measure"],
    )
    kernel_session.verified_query_ledger = [query_artifact]
    deep_session = GroundedDeepAgentSession(
        runtime=kernel_session,
        artifact_goal_ids={"artifact.measure": ["metric.measure"]},
        question_goal_contract=OriginalQuestionGoalContract(
            question=kernel_session.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.measure",
                    label="verified measure",
                ),
                AnalysisQuestionGoal(
                    goal_id="analysis.primary",
                    label="explore the verified change",
                    analysis_type="OPEN_EXPLORATION",
                    input_goal_ids=["metric.measure"],
                ),
            ],
        ),
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-exploration",
        run_id="run-exploration",
        session=deep_session,
    )
    visible, _ = _phase_visible_tools(deep_session, outer.tools)
    assert "delegate_grounded_exploration" in {item.name for item in visible}

    tools = {item.name: item for item in outer.tools}
    result = json.loads(
        tools["delegate_grounded_exploration"].func(
            analysis_goal_ids=["analysis.primary"],
            source_query_artifact_ids=["artifact.measure"],
            objective="Explore competing explanations from verified evidence.",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ADVISORY_ACCEPTED"
    assert result["authority"] == "ADVISORY"
    assert result["publishableAsFinal"] is False
    assert result["queryExecuted"] is False
    assert result["pendingCapabilityRequests"][0]["executable"] is False
    assert result["pendingCapabilityRequests"][0]["queryDispatched"] is False
    assert isolated.job.backend is None
    assert isolated.job.tools == []
    assert isolated.job.skills == []
    assert isolated.job.permissions == []
    assert isolated.job.subagents == []
    assert isolated.job.model_timeout_seconds == 15.0
    assert len(deep_session.exploration_states) == 1
    assert len(deep_session.exploration_reports) == 1


def test_skill_repairs_once_inside_isolation_without_query_mutation(
    tmp_path: Path,
) -> None:
    factory = CapturingFactory(action="none")
    kernel = FakeKernel()
    settings = Settings(harness_workspace_path=str(tmp_path / "runtime"))
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        skill_root="python_backend/resources/runtime/agent_skills",
        skill_run_root=str(tmp_path / "skill-runs"),
        settings=settings,
        agent_factory=factory,
    )

    class RepairingRuntime:
        jobs: list[Any] = []

        def run(self, job: Any, *, on_progress: Any) -> IsolatedSubagentResult:
            self.jobs.append(job)
            on_progress("subagent", "started", job.job_id)
            if len(self.jobs) == 1:
                raw = json.dumps(
                    {"answerMarkdown": "1. 未验证指标为 999。"},
                    ensure_ascii=False,
                )
            else:
                raw = json.dumps(
                    {
                        "answerMarkdown": "当前仅报告已验证查询结果，未补充新的归因结论。",
                        "observations": [],
                        "semanticDisclosures": [],
                        "derivedFacts": [],
                        "hypotheses": [],
                        "recommendations": [],
                        "evidenceRefs": [],
                        "gaps": ["缺少可验证的归因证据"],
                        "executionConfidence": 0.6,
                    },
                    ensure_ascii=False,
                )
            on_progress("subagent", "completed", "updates=1")
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={"threadId": job.thread_id, "runId": job.job_id},
                raw_output=raw,
                update_count=1,
            )

    repairing = RepairingRuntime()
    outer.subagent_runtime = repairing
    kernel_session = kernel.new_session("分析最近30天经营风险", "m-1")
    kernel_session.active_contract = GroundedQueryContract(
        question=kernel_session.question,
        topics=["客服工单"],
        status="READY",
        query_shape="SCALAR",
    )
    kernel_session.active_plan = QueryPlan()
    kernel_session.run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(rows=[{"risk_value": 1}], tables=["tickets"])
    )
    kernel_session.verified_evidence = VerifiedEvidence(passed=True)
    kernel_session.answer_plan = kernel_session.active_plan
    kernel_session.answer_run_result = kernel_session.run_result
    kernel_session.answer_verified_evidence = kernel_session.verified_evidence
    workspace = _publish_skill_test_artifact(
        settings,
        kernel_session,
        thread_id="thread-skill-repair",
        run_id="run-skill-repair",
    )
    deep_session = GroundedDeepAgentSession(
        runtime=kernel_session,
        context_workspace=workspace,
        analysis_skill_headers_disclosed=True,
        data_collection_sealed=True,
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-skill-repair",
        run_id="run-skill-repair",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["run_skill"].func(
            skill_name="risk-analysis",
            objective="基于已验证证据分析风险",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "SKILL_COMPLETED"
    assert result["repairAttempted"] is True
    assert result["queryMutationAllowed"] is False
    assert len(repairing.jobs) == 2
    assert repairing.jobs[1].user_payload["repairAttempt"] == 1
    blocked = json.loads(
        tools["retrieve_knowledge"].func(
            query="再查更多指标",
            reason="修复 Skill",
            runtime=SimpleNamespace(context=context),
        )
    )
    assert blocked["status"] == "POST_QUERY_SKILL_BOUNDARY_CLOSED"


def test_skill_second_failure_returns_verified_fallback_without_third_attempt(
    tmp_path: Path,
) -> None:
    class FallbackKernel(FakeKernel):
        @staticmethod
        def compose_answer(session: GroundedRuntimeSession, *, allow_llm: bool) -> str:
            assert allow_llm is False
            session.answer = "已返回确定性已验证查询结果。"
            return session.answer

    factory = CapturingFactory(action="none")
    kernel = FallbackKernel()
    settings = Settings(harness_workspace_path=str(tmp_path / "runtime"))
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        checkpointer=object(),
        skill_root="python_backend/resources/runtime/agent_skills",
        skill_run_root=str(tmp_path / "skill-runs"),
        settings=settings,
        agent_factory=factory,
    )

    class AlwaysInvalidRuntime:
        jobs: list[Any] = []

        def run(self, job: Any, *, on_progress: Any) -> IsolatedSubagentResult:
            self.jobs.append(job)
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={"threadId": job.thread_id, "runId": job.job_id},
                raw_output=json.dumps(
                    {"answerMarkdown": "未经验证的结果为 999。"},
                    ensure_ascii=False,
                ),
                update_count=1,
            )

    invalid = AlwaysInvalidRuntime()
    outer.subagent_runtime = invalid
    kernel_session = kernel.new_session("分析最近30天经营风险", "m-1")
    kernel_session.active_contract = GroundedQueryContract(
        question=kernel_session.question,
        topics=["客服工单"],
        status="READY",
        query_shape="SCALAR",
    )
    kernel_session.active_plan = QueryPlan()
    kernel_session.run_result = AgentRunResult(
        merged_query_bundle=QueryBundle(rows=[{"risk_value": 1}], tables=["tickets"])
    )
    kernel_session.verified_evidence = VerifiedEvidence(passed=True)
    kernel_session.answer_plan = kernel_session.active_plan
    kernel_session.answer_run_result = kernel_session.run_result
    kernel_session.answer_verified_evidence = kernel_session.verified_evidence
    workspace = _publish_skill_test_artifact(
        settings,
        kernel_session,
        thread_id="thread-skill-fallback",
        run_id="run-skill-fallback",
    )
    deep_session = GroundedDeepAgentSession(
        runtime=kernel_session,
        context_workspace=workspace,
        analysis_skill_headers_disclosed=True,
        data_collection_sealed=True,
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-skill-fallback",
        run_id="run-skill-fallback",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["run_skill"].func(
            skill_name="risk-analysis",
            objective="基于已验证证据分析风险",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "SKILL_FALLBACK_ANSWERED"
    assert result["answerMarkdown"] == "已返回确定性已验证查询结果。"
    assert result["queryMutationAllowed"] is False
    assert len(invalid.jobs) == 2
    assert kernel_session.answer == result["answerMarkdown"]
