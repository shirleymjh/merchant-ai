from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

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
    SubAgentResultEnvelope,
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
    GroundedToolCallRepairMiddleware,
    GroundedTrustedSessionContextMiddleware,
    _core_visible_binding_hints,
    _grounded_contract_sql_obligations,
    _grounded_semantic_read_control,
    _latest_grounded_repair_tool_exchange,
    _metric_recall_slots,
    _augment_batch_query_request_semantic_refs,
    _normalize_detail_goal_binding_hints,
    _normalize_scalar_metric_query_request,
    _read_exact_core_semantic_path,
    _record_governed_query_branch_outcome,
    _simple_scalar_goal_contract,
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
from merchant_ai.services.grounded_query_branches import (
    GroundedBranchBudget,
    GroundedBranchBudgetLimits,
    GroundedQueryBranchContext,
    GroundedQueryBranchSpec,
)
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
    DetailQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    OriginalQuestionGoalDeclaration,
    RankingQuestionGoal,
    RuleQuestionGoal,
    TimeWindowQuestionGoal,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)
from merchant_ai.services.grounded_subagent_runtime import IsolatedSubagentResult
from merchant_ai.services.grounded_subagent_runtime import (
    GroundedSubagentBudget,
    GroundedSubagentDispatchPlan,
    GroundedSubagentEvidenceRequirement,
    GroundedSubagentGoalContract,
    GroundedSkillRunContract,
)
from merchant_ai.services.query_request import QueryRequest
from merchant_ai.services.query_request import (
    QueryOutcome,
    QueryOutcomeStatus,
    StructuredQueryObservation,
)


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


def _verified_skill_contract(
    session: GroundedDeepAgentSession,
    *,
    skill_name: str = "risk-analysis",
    sub_goal_id: str = "skill.risk",
    generation: int = 1,
    snapshot_generation: int = 1,
) -> GroundedSkillRunContract:
    artifact_ids = list(session.runtime.answer_artifact_ids)
    assert artifact_ids
    session.question_goal_contract = OriginalQuestionGoalContract(
        question=session.runtime.question,
        goals=[
            MetricQuestionGoal(
                goal_id="metric.skill.input",
                label="Skill verified input",
            )
        ],
    )
    session.artifact_goal_ids = {artifact_id: ["metric.skill.input"] for artifact_id in artifact_ids}
    session.analysis_skill_headers_disclosed = True
    session.skill_input_snapshot_generation = snapshot_generation
    session.data_collection_sealed = False
    return GroundedSkillRunContract(
        sub_goal_id=sub_goal_id,
        parent_goal_ids=["metric.skill.input"],
        skill_name=skill_name,
        objective="基于已验证证据执行正式分析 Skill",
        required_outputs=["observations", "gaps"],
        input_artifact_ids=artifact_ids,
        evidence_requirements=[
            GroundedSubagentEvidenceRequirement(
                requirement_id="verified.skill.inputs",
                description="Use only selected verified query artifacts.",
                accepted_ref_types=["VERIFIED_QUERY_ARTIFACT"],
            )
        ],
        budget=GroundedSubagentBudget(
            max_tool_calls=8,
            timeout_seconds=45,
        ),
        input_snapshot_generation=snapshot_generation,
        generation=generation,
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


class FakeScalarSemanticCatalog(FakeSemanticCatalog):
    def read(self, *, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        del max_chars, offset
        if path == "topics/经营画像/tables/ads_merchant_profile/detail.json":
            content = json.dumps(
                {
                    "topic": "经营画像",
                    "tableName": "ads_merchant_profile",
                    "timeColumn": "pt",
                    "merchantFilterColumn": "merchant_id",
                },
                ensure_ascii=False,
            )
            return {
                "success": True,
                "refId": "semantic:经营画像:ads_merchant_profile:detail",
                "path": path,
                "kind": "TABLE_DETAIL",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "content": content,
            }
        if path == ("topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json"):
            content = json.dumps(
                {
                    "metricKey": "order_cnt_1d",
                    "businessName": "订单量",
                    "formula": "SUM(order_cnt_1d)",
                },
                ensure_ascii=False,
            )
            return {
                "success": True,
                "refId": ("semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"),
                "path": path,
                "kind": "METRIC",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "content": content,
            }
        if path == "topics/经营画像/tables/ads_merchant_profile/columns/pt.json":
            content = json.dumps(
                {
                    "topic": "经营画像",
                    "tableName": "ads_merchant_profile",
                    "section": "columns",
                    "key": "pt",
                    "definition": {
                        "columnName": "pt",
                        "businessName": "业务日期",
                        "role": "TIME",
                    },
                },
                ensure_ascii=False,
            )
            return {
                "success": True,
                "refId": "semantic:经营画像:ads_merchant_profile:field:pt",
                "path": path,
                "kind": "COLUMN",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "content": content,
            }
        return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}


class FakeMultiMetricSemanticCatalog(FakeScalarSemanticCatalog):
    def read(self, *, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        del max_chars, offset
        if path == "topics/电商退货/tables/refunds/detail.json":
            return {
                "success": True,
                "refId": "semantic:电商退货:refunds:detail",
                "path": path,
                "kind": "TABLE_DETAIL",
                "topic": "电商退货",
                "table": "refunds",
                "content": json.dumps(
                    {
                        "topic": "电商退货",
                        "tableName": "refunds",
                        "timeColumn": "pt",
                    },
                    ensure_ascii=False,
                ),
            }
        if path == "topics/电商退货/tables/refunds/metrics/refund_amt.json":
            return {
                "success": True,
                "refId": "semantic:电商退货:refunds:metric:refund_amt",
                "path": path,
                "kind": "METRIC",
                "topic": "电商退货",
                "table": "refunds",
                "content": json.dumps(
                    {
                        "metric": {
                            "metricKey": "refund_amt",
                            "businessName": "退款金额",
                            "aliases": ["退款金额"],
                            "timeColumn": "pt",
                        }
                    },
                    ensure_ascii=False,
                ),
            }
        if path == "topics/电商退货/tables/refunds/columns/pt.json":
            return {
                "success": True,
                "refId": "semantic:电商退货:refunds:field:pt",
                "path": path,
                "kind": "COLUMN",
                "topic": "电商退货",
                "table": "refunds",
                "content": json.dumps(
                    {
                        "key": "pt",
                        "definition": {
                            "columnName": "pt",
                            "businessName": "业务日期",
                            "role": "TIME",
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        return super().read(path=path, max_chars=2_000_000, offset=0)


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
            retrieval_question=(
                kwargs.get("retrieval_question") or question
            ),
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
        self.recall_queries.append(
            query or session.retrieval_question or session.question
        )
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
        self.contexts: list[Any] = []

    def invoke(self, payload: dict[str, Any], *, config: Any, context: Any) -> None:
        self.invocations.append(payload)
        self.configs.append(config)
        self.contexts.append(context)
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


class ContextualizedConversationAuthority:
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
            retrieval_question="最近7天的退款情况",
            status="RESOLVED_REFERENCE",
            reference_detected=True,
            source="TEST_STRUCTURED_CONVERSATION_REVIEW",
        )


def runtime(factory: CapturingFactory, kernel: FakeKernel | None = None) -> GroundedDeepAgentRuntime:
    return GroundedDeepAgentRuntime(
        kernel or FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=factory,
        conversation_online_authority=StandaloneConversationAuthority(),
    )


def model_call_request(
    messages: list[Any],
    *,
    session: GroundedDeepAgentSession,
) -> Any:
    request = SimpleNamespace(
        messages=messages,
        tools=[],
        system_message=HumanMessage(content="system"),
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="tool-repair-thread",
                run_id="tool-repair-run",
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


def test_query_data_normalizes_only_provably_redundant_scalar_bindings() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="normalize-query-request",
            question="最近7天订单量是多少",
            merchant_id="m-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="最近7天订单量是多少",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单量",
                ),
                TimeWindowQuestionGoal(
                    goal_id="time.recent_7_days",
                    label="最近7天",
                    time_expression="最近7天",
                    days=7,
                    applies_to_goal_ids=["metric.orders"],
                ),
            ],
        ),
    )
    metric_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    request = QueryRequest.model_validate(
        {
            "queryId": "orders.recent_7_days",
            "goalIds": ["metric.orders", "time.recent_7_days"],
            "bindingHints": {
                "metricRefs": [metric_ref],
                "fieldAggregations": [
                    {
                        "fieldRef": "semantic:经营画像:ads_merchant_profile:field:order_cnt_1d",
                        "aggregation": "SUM",
                    }
                ],
                "selectedFields": [
                    {
                        "fieldRef": "semantic:经营画像:ads_merchant_profile:field:order_cnt_1d",
                    }
                ],
            },
        }
    )

    normalized = _normalize_scalar_metric_query_request(session, request)

    assert normalized.binding_hints.metric_refs == [metric_ref]
    assert normalized.binding_hints.field_aggregations == []
    assert normalized.binding_hints.selected_fields == []
    assert request.binding_hints.field_aggregations
    assert request.binding_hints.selected_fields


def test_query_data_keeps_mixed_bindings_when_metric_refs_do_not_cover_goals() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="keep-ambiguous-query-request",
            question="订单量和买家数是多少",
            merchant_id="m-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="订单量和买家数是多少",
            goals=[
                MetricQuestionGoal(goal_id="metric.orders", label="订单量"),
                MetricQuestionGoal(goal_id="metric.buyers", label="买家数"),
            ],
        ),
    )
    request = QueryRequest.model_validate(
        {
            "queryId": "orders-and-buyers",
            "goalIds": ["metric.orders", "metric.buyers"],
            "bindingHints": {
                "metricRefs": ["semantic:经营画像:profile:metric:order_cnt_1d"],
                "fieldAggregations": [
                    {
                        "fieldRef": "semantic:电商交易:orders:field:buyer_id",
                        "aggregation": "COUNT_DISTINCT",
                    }
                ],
            },
        }
    )

    normalized = _normalize_scalar_metric_query_request(session, request)

    assert normalized == request


def test_query_data_updates_frozen_branch_and_unlocks_verified_dependency() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="query-data-branch-sync",
            question="先查商品再查工单",
            merchant_id="m-1",
        )
    )
    source = GroundedQueryBranchContext(
        spec=GroundedQueryBranchSpec(
            query_id="source",
            goal_ids=["goal.source"],
            topic_scope=["商品"],
        ),
        runtime=None,
        budget=GroundedBranchBudget(
            "source",
            GroundedBranchBudgetLimits(),
        ),
    )
    dependent = GroundedQueryBranchContext(
        spec=GroundedQueryBranchSpec(
            query_id="dependent",
            goal_ids=["goal.dependent"],
            topic_scope=["客服工单"],
        ),
        runtime=None,
        budget=GroundedBranchBudget(
            "dependent",
            GroundedBranchBudgetLimits(),
        ),
        dependency_query_ids=["source"],
        status="WAITING_VERIFIED_ARTIFACT",
    )
    session.query_branch_contexts = {
        "source": source,
        "dependent": dependent,
    }

    _record_governed_query_branch_outcome(
        session,
        QueryRequest(query_id="source", goal_ids=["goal.source"]),
        QueryOutcome(
            status=QueryOutcomeStatus.VERIFIED,
            query_id="source",
            artifact_ids=["artifact.source"],
            covered_goal_ids=["goal.source"],
        ),
    )

    assert source.status == "VERIFIED"
    assert source.verified_artifact_ids == ["artifact.source"]
    assert dependent.status == "DECLARED"


def test_query_data_records_reasoning_gap_on_frozen_branch() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="query-data-branch-gap",
            question="查订单量",
            merchant_id="m-1",
        )
    )
    branch = GroundedQueryBranchContext(
        spec=GroundedQueryBranchSpec(
            query_id="orders",
            goal_ids=["goal.orders"],
            topic_scope=["经营画像"],
        ),
        runtime=None,
        budget=GroundedBranchBudget(
            "orders",
            GroundedBranchBudgetLimits(),
        ),
    )
    session.query_branch_contexts = {"orders": branch}

    _record_governed_query_branch_outcome(
        session,
        QueryRequest(query_id="orders", goal_ids=["goal.orders"]),
        QueryOutcome(
            status=QueryOutcomeStatus.NEEDS_REASONING,
            query_id="orders",
            observation=StructuredQueryObservation(
                stage="CONTRACT",
                code="TIME_FIELD_REF_NOT_READ",
                retryable=True,
                gaps=[{"code": "TIME_FIELD_REF_NOT_READ"}],
            ),
        ),
    )

    assert branch.status == "CONTRACT_GAPPED"
    assert branch.last_gaps == [{"code": "TIME_FIELD_REF_NOT_READ"}]


def test_simple_scalar_goal_contract_keeps_query_graph_and_delegation_choices() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="simple-scalar-tools",
            question="最近7天订单量是多少",
            merchant_id="m-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="最近7天订单量是多少",
            goals=[
                MetricQuestionGoal(goal_id="metric.orders", label="订单量"),
                TimeWindowQuestionGoal(
                    goal_id="time.recent_7_days",
                    label="最近7天",
                    time_expression="最近7天",
                    days=7,
                    applies_to_goal_ids=["metric.orders"],
                ),
            ],
        ),
    )

    visible, _ = _phase_visible_tools(session, outer.tools)
    visible_names = {item.name for item in visible}

    assert _simple_scalar_goal_contract(session) is True
    assert "query_data" in visible_names
    assert "propose_grounded_execution_graph" in visible_names
    assert "query_batch" in visible_names
    assert "delegate_grounded_tasks" in visible_names


def test_simple_scalar_goal_declaration_prefetches_exact_metric_assets() -> None:
    factory = CapturingFactory(action="none")
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeScalarSemanticCatalog(),
        agent_factory=factory,
        conversation_online_authority=StandaloneConversationAuthority(),
    )
    question = "最近7天订单量是多少"
    metric_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="simple-scalar-prefetch",
            question=question,
            merchant_id="m-1",
            workspace_topics=["经营画像"],
            recall=RecallBundle(
                items=[
                    RecallItem(
                        doc_id=metric_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="经营画像",
                        table="ads_merchant_profile",
                        metadata={
                            "semanticRefId": metric_ref,
                            "semanticPath": ("topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json"),
                            "metricResolutionType": "exact_semantic_label",
                            "metricResolutionConfidence": 0.97,
                            "metricResolutionAmbiguous": False,
                        },
                    )
                ]
            ),
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-prefetch",
        run_id="run-prefetch",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单量",
                    ),
                    TimeWindowQuestionGoal(
                        goal_id="time.recent_7_days",
                        label="最近7天",
                        time_expression="最近7天",
                        days=7,
                        applies_to_goal_ids=["metric.orders"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert result["simpleScalarPrefetch"]["status"] == "PREFETCHED"
    assert result["nextAction"] == "QUERY_DATA_WITH_PREFETCHED_ASSETS"
    assert {item["refId"] for item in session.core_semantic_evidence} == {
        "semantic:经营画像:ads_merchant_profile:detail",
        metric_ref,
        "semantic:经营画像:ads_merchant_profile:field:pt",
    }
    assert result["simpleScalarPrefetch"]["timeFieldRef"] == ("semantic:经营画像:ads_merchant_profile:field:pt")
    visible, _ = _phase_visible_tools(session, outer.tools)
    visible_names = {item.name for item in visible}
    assert "query_data" in visible_names
    assert "read_file" not in visible_names
    assert "retrieve_knowledge" not in visible_names


def test_goal_declaration_removes_presentation_directive_fake_rule() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    question = "最近7天订单明细和工单明细分别给我看一下。"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="presentation-rule-normalization",
            question=question,
            merchant_id="m-1",
            workspace_topics=["电商交易", "客服工单"],
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-presentation-rule",
        run_id="run-presentation-rule",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalDeclaration(
                goals=[
                    TimeWindowQuestionGoal(
                        goal_id="time.recent_7_days",
                        label="最近7天",
                        source_spans=["最近7天"],
                        time_expression="最近7天",
                        days=7,
                        applies_to_goal_ids=["detail.orders", "detail.tickets"],
                    ),
                    DetailQuestionGoal(
                        goal_id="detail.orders",
                        label="订单明细",
                        source_spans=["订单明细"],
                    ),
                    DetailQuestionGoal(
                        goal_id="detail.tickets",
                        label="工单明细",
                        source_spans=["工单明细"],
                    ),
                    RuleQuestionGoal(
                        goal_id="rule.separate_views",
                        label="订单明细和工单明细分别展示",
                        source_spans=["分别给我看一下"],
                        requested_action="分别返回两个明细结果",
                    ),
                ]
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert result["goalNormalization"] == {
        "code": "PRESENTATION_DIRECTIVE_RULE_REMOVED",
        "removedGoalIds": ["rule.separate_views"],
    }
    assert session.question_goal_contract is not None
    assert [goal.kind for goal in session.question_goal_contract.goals] == [
        "TIME_WINDOW",
        "DETAIL",
        "DETAIL",
    ]
    assert result["nextAction"] == "DISCOVER_SEMANTIC_EVIDENCE"


def test_batch_request_keeps_explicit_time_field_in_semantic_read_set() -> None:
    request = QueryRequest(
        query_id="detail.orders",
        read_ref_ids=["semantic:电商交易:dwm_trade_order_detail_di:detail"],
        binding_hints=GroundedBindingHints(time_field_ref=("semantic:电商交易:dwm_trade_order_detail_di:column:pt")),
    )

    normalized = _augment_batch_query_request_semantic_refs(request)

    assert normalized.read_ref_ids == [
        "semantic:电商交易:dwm_trade_order_detail_di:detail",
        "semantic:电商交易:dwm_trade_order_detail_di:field:pt",
    ]
    assert request.read_ref_ids == ["semantic:电商交易:dwm_trade_order_detail_di:detail"]


def test_batch_request_adds_declared_semantic_table_binding_to_read_set() -> None:
    request = QueryRequest(
        query_id="detail.orders",
        binding_hints=GroundedBindingHints(
            table_refs=["semantic:电商交易:dwm_trade_order_detail_di:detail"],
            time_field_ref=("semantic:电商交易:dwm_trade_order_detail_di:field:pt"),
        ),
    )

    normalized = _augment_batch_query_request_semantic_refs(request)

    assert normalized.read_ref_ids == [
        "semantic:电商交易:dwm_trade_order_detail_di:detail",
        "semantic:电商交易:dwm_trade_order_detail_di:field:pt",
    ]


def test_batch_request_does_not_treat_physical_aliases_as_semantic_reads() -> None:
    request = QueryRequest(
        query_id="detail.orders",
        semantic_paths=["semantic:电商交易:dwm_trade_order_detail_di:detail"],
        binding_hints=GroundedBindingHints(
            table_refs=["dwm_trade_order_detail_di"],
            time_field_ref="pt",
            time_expression="最近7天",
        ),
    )

    normalized = _augment_batch_query_request_semantic_refs(request)

    assert normalized.read_ref_ids == []
    assert normalized.binding_hints.table_refs == ["dwm_trade_order_detail_di"]
    assert normalized.binding_hints.time_field_ref == "pt"


def test_generic_detail_goal_does_not_trust_model_all_fields_flag() -> None:
    question = "最近7天订单明细和工单明细分别给我看一下。"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="generic-detail-goal",
            question=question,
            merchant_id="m-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question=question,
            goals=[
                DetailQuestionGoal(
                    goal_id="detail.orders",
                    label="订单明细",
                    request_all_fields=True,
                )
            ],
        ),
    )

    normalized = _normalize_detail_goal_binding_hints(
        session,
        ["detail.orders"],
        GroundedBindingHints(),
    )

    assert normalized.analysis_mode == "DETAIL"
    assert normalized.detail_projection_mode == "DEFAULT"


def test_multi_metric_goal_contract_keeps_graph_and_batch_tools() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="multi-metric-tools",
            question="订单量和退款金额是多少",
            merchant_id="m-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="订单量和退款金额是多少",
            goals=[
                MetricQuestionGoal(goal_id="metric.orders", label="订单量"),
                MetricQuestionGoal(goal_id="metric.refunds", label="退款金额"),
            ],
        ),
    )

    visible, _ = _phase_visible_tools(
        session,
        [*outer.tools, SimpleNamespace(name="task")],
    )
    visible_names = {item.name for item in visible}

    assert _simple_scalar_goal_contract(session) is False
    assert "propose_grounded_execution_graph" in visible_names
    assert "query_batch" in visible_names
    assert "delegate_grounded_tasks" in visible_names
    assert "task" in visible_names


def test_governed_response_exposes_goal_contract_for_trace_audit() -> None:
    goal_contract = OriginalQuestionGoalContract(
        question="最近7天订单量是多少",
        goals=[
            MetricQuestionGoal(goal_id="metric.orders", label="订单量"),
            TimeWindowQuestionGoal(
                goal_id="time.recent_7_days",
                label="最近7天",
                time_expression="最近7天",
                days=7,
                applies_to_goal_ids=["metric.orders"],
            ),
        ],
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="trace-goal-contract",
            question=goal_contract.question,
            merchant_id="m-1",
        ),
        question_goal_contract=goal_contract,
        operational_failure={
            "code": "TEST_TERMINAL",
            "message": "test response",
            "retryable": False,
        },
    )

    response = GroundedDeepAgentRuntime._governed_response(
        session,
        "thread-trace-goals",
        "run-trace-goals",
    )
    harness = response.debug_trace["harness"]

    assert harness["goalKinds"] == ["METRIC", "TIME_WINDOW"]
    assert harness["originalQuestionGoalContract"] == goal_contract.model_dump(
        by_alias=True,
        mode="json",
    )


def test_goal_declaration_binds_trusted_question_without_model_copy() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    kernel_session = outer.kernel.new_session("最近7天订单总数", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="thread-goal-trusted-question",
        run_id="run-goal-trusted-question",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalDeclaration(
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单总数",
                    ),
                    TimeWindowQuestionGoal(
                        goal_id="time.recent_7_days",
                        label="最近7天",
                        time_expression="最近7天",
                        days=7,
                        applies_to_goal_ids=["metric.orders"],
                    ),
                ]
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert deep_session.question_goal_contract is not None
    assert deep_session.question_goal_contract.question == "最近7天订单总数"


def test_goal_declaration_does_not_invoke_population_gate() -> None:
    class ExplodingPopulationGate:
        def commit_goal(self, **_: Any) -> Any:
            raise AssertionError("Goal declaration must not call population authority")

    factory = CapturingFactory(action="none")
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=factory,
        population_execution_gate=ExplodingPopulationGate(),
        population_gate_enforced=True,
    )
    kernel_session = outer.kernel.new_session("最近10天订单明细", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="thread-goal-no-population",
        run_id="run-goal-no-population",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    DetailQuestionGoal(
                        goal_id="detail.orders",
                        label="最近10天订单明细",
                    )
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert deep_session.population_goal_gate_result == {}
    assert deep_session.population_goal_attestation is None


def test_goal_declaration_rejects_premature_population_binding() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    kernel_session = outer.kernel.new_session("在这些订单中找退款最高的三单", "m-1")
    deep_session = GroundedDeepAgentSession(runtime=kernel_session)
    context = GroundedDeepAgentRunContext(
        thread_id="thread-goal-population-rejected",
        run_id="run-goal-population-rejected",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=kernel_session.question,
                goals=[
                    DetailQuestionGoal(
                        goal_id="detail.orders",
                        label="这些订单",
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refund_amount",
                        label="退款金额",
                    ),
                    RankingQuestionGoal(
                        goal_id="ranking.refunds",
                        label="退款最高三单",
                        metric_goal_ids=["metric.refund_amount"],
                        limit=3,
                        population_scope="SAME_AS_GOAL",
                        population_goal_ids=["detail.orders"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "REJECTED"
    assert result["code"] == "GOAL_EXECUTION_SCOPE_FORBIDDEN"
    assert deep_session.question_goal_contract is None


def test_runtime_source_has_no_legacy_or_action_catalog_dependencies() -> None:
    source_path = Path(__file__).resolve().parents[2] / "merchant_ai/services/grounded_deep_agent_runtime.py"
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

    services_root = source_path.parent
    for online_module in (
        services_root / "runtime_factory.py",
        services_root / "grounded_runtime_kernel.py",
        services_root / "grounded_query_contract.py",
        services_root / "grounded_query_executor.py",
    ):
        online_source = online_module.read_text(encoding="utf-8")
        for forbidden_import in (
            "from merchant_ai.graph.workflow import",
            "from merchant_ai.services.planning import",
            "from merchant_ai.services.query import NodeWorkerExecutor",
        ):
            assert forbidden_import not in online_source


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
        "publish_verified_entity_set",
        "delegate_grounded_tasks",
        "delegate_grounded_exploration",
        "finalize_evidence_collection",
        "compose_verified_answer",
        "load_skill",
        "run_skill",
        "ask_human",
        "query_data",
        "query_batch",
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
    query_tool = next(item for item in factory.kwargs["tools"] if item.name == "query_data")
    query_schema = json.dumps(
        query_tool.tool_call_schema.model_json_schema(),
        ensure_ascii=False,
    )
    assert "tableRefs" in query_schema
    assert "metricRefs" in query_schema
    assert "timeExpression" in query_schema
    goal_tool = next(item for item in factory.kwargs["tools"] if item.name == "declare_original_question_goals")
    goal_schema = json.dumps(
        goal_tool.tool_call_schema.model_json_schema(),
        ensure_ascii=False,
    )
    for execution_field in (
        "populationScope",
        "populationGoalIds",
        "metricRefId",
        "dimensionRefId",
        "entityRefId",
        "requiredFieldRefIds",
        "artifactKind",
    ):
        assert execution_field not in goal_schema
    compose_tool = next(item for item in factory.kwargs["tools"] if item.name == "compose_verified_answer")
    # Completion gaps are ordinary ReAct observations so Core can decide
    # whether to query, clarify, or disclose an accepted partial result.
    assert compose_tool.return_direct is False
    ask_human_tool = next(item for item in factory.kwargs["tools"] if item.name == "ask_human")
    assert ask_human_tool.return_direct is False
    assert factory.kwargs["backend"] is not None
    assert [item.name for item in factory.kwargs["middleware"]] == [
        "GroundedTrustedSessionContextMiddleware",
        "GroundedToolCallRepairMiddleware",
        "GroundedContextManagementMiddleware",
        "GroundedRuntimeBudgetMiddleware",
        "GroundedCoreToolBoundaryMiddleware",
    ]
    assert isinstance(
        factory.kwargs["middleware"][0],
        GroundedTrustedSessionContextMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][1],
        GroundedToolCallRepairMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][2],
        GroundedContextManagementMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][3],
        GroundedRuntimeBudgetMiddleware,
    )
    assert isinstance(
        factory.kwargs["middleware"][4],
        GroundedCoreToolBoundaryMiddleware,
    )
    assert [item["name"] for item in factory.kwargs["subagents"]] == [
        "general-purpose",
        "grounded-researcher",
    ]
    assert factory.kwargs["subagents"][0]["tools"] == []
    assert factory.kwargs["subagents"][0]["skills"] is None
    researcher = factory.kwargs["subagents"][1]
    assert researcher["tools"] == []
    assert researcher["response_format"] is SubAgentResultEnvelope
    # Deep Agents supplies the native task's filesystem middleware from the
    # raw spec's permissions; the harness must not inject a duplicate.
    assert researcher.get("middleware") in (None, [])
    assert "READ_CONTEXT" in researcher["description"]
    assert factory.kwargs["skills"] is None
    assert "execute SQL" in factory.kwargs["subagents"][0]["system_prompt"]
    assert '<prompt id="grounded.native.general_purpose"' in (
        factory.kwargs["subagents"][0]["system_prompt"]
    )
    assert '<prompt id="grounded.native.researcher"' in (
        factory.kwargs["subagents"][1]["system_prompt"]
    )
    assert '<prompt id="grounded.core.system"' in factory.kwargs[
        "system_prompt"
    ]
    assert "Use query_data for one governed query" in factory.kwargs["system_prompt"]
    assert "capabilities and constraints, not as a prewritten procedure" in (
        factory.kwargs["system_prompt"]
    )
    assert "not alternate planning authorities" in factory.kwargs[
        "system_prompt"
    ]


def test_compose_tool_ends_only_after_answer_attestation() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(SimpleNamespace())
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="compose-command",
            question="return revenue",
            merchant_id="merchant-1",
            answer="verified answer",
        ),
        answer_coverage_result={
            "passed": True,
            "source": "compose_verified_answer",
            "answerFingerprint": answer_fingerprint("verified answer"),
        },
    )
    request = SimpleNamespace(
        tool_call={
            "name": "compose_verified_answer",
            "id": "compose-call",
            "args": {"allow_llm": False},
        },
        runtime=SimpleNamespace(
            context=SimpleNamespace(session=session)
        ),
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(
            content=json.dumps(
                {"status": "ANSWERED", "answer": "verified answer"}
            ),
            name="compose_verified_answer",
            tool_call_id="compose-call",
        ),
    )

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update["messages"][0].tool_call_id == "compose-call"


def test_compose_gap_remains_a_react_observation() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(SimpleNamespace())
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="compose-observation",
            question="return revenue and orders",
            merchant_id="merchant-1",
        )
    )
    request = SimpleNamespace(
        tool_call={
            "name": "compose_verified_answer",
            "id": "compose-gap-call",
            "args": {"allow_llm": False},
        },
        runtime=SimpleNamespace(
            context=SimpleNamespace(session=session)
        ),
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(
            content=json.dumps(
                {"status": "OUTCOME_COMPLETION_INCOMPLETE"}
            ),
            name="compose_verified_answer",
            tool_call_id="compose-gap-call",
        ),
    )

    assert isinstance(result, ToolMessage)
    assert json.loads(result.content)["status"] == (
        "OUTCOME_COMPLETION_INCOMPLETE"
    )


def test_production_settings_expose_governed_query_facade(
    tmp_path: Path,
) -> None:
    factory = CapturingFactory(action="none")
    settings = Settings(
        harness_workspace_path=str(tmp_path / "runtime"),
    )
    GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=factory,
        settings=settings,
        skill_run_root=str(tmp_path / "skill-runs"),
        conversation_online_authority=StandaloneConversationAuthority(),
    )

    names = {item.name for item in factory.kwargs["tools"]}
    assert {"query_data", "query_batch"}.issubset(names)
    query_schema = next(
        item for item in factory.kwargs["tools"] if item.name == "query_data"
    ).tool_call_schema.model_json_schema()
    assert "merchantId" not in json.dumps(query_schema, ensure_ascii=False)


def test_core_agent_selects_skill_placement_with_harness_resource_guards(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "skills"
    core_skill = skill_root / "short-check"
    complex_skill = skill_root / "complex-diagnosis"
    core_skill.mkdir(parents=True)
    complex_skill.mkdir(parents=True)
    (core_skill / "SKILL.md").write_text(
        """---
name: short-check
description: Check one bounded verified value.
executionPlacement: AUTO
executionMode: structured_renderer
---
Compare the verified value with the governed threshold and cite both.
""",
        encoding="utf-8",
    )
    (complex_skill / "SKILL.md").write_text(
        """---
name: complex-diagnosis
description: Run a long multi-step diagnosis.
executionPlacement: AUTO
executionMode: python_script
script: scripts/diagnose.py
---
Run the isolated diagnosis procedure.
""",
        encoding="utf-8",
    )
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeSemanticCatalog(),
        agent_factory=CapturingFactory(action="none"),
        skill_root=str(skill_root),
        skill_run_root=str(tmp_path / "skill-runs"),
        conversation_online_authority=StandaloneConversationAuthority(),
    )

    headers = {item["name"]: item for item in outer.skill_headers}
    assert headers["short-check"]["executionPlacement"] == "AUTO"
    assert headers["complex-diagnosis"]["executionPlacement"] == "AUTO"
    assert "coreProcedure" not in headers["short-check"]
    assert "coreProcedure" not in headers["complex-diagnosis"]

    deep_session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="complex-skill-isolation",
            question="诊断复杂指标",
            merchant_id="merchant-1",
            workspace_topics=["客服工单"],
        ),
        analysis_skill_headers_disclosed=True,
        skill_input_snapshot_generation=1,
        question_goal_contract=OriginalQuestionGoalContract(
            question="诊断复杂指标",
            goals=[MetricQuestionGoal(goal_id="metric.primary", label="复杂指标")],
        ),
    )
    deep_session.runtime.answer_artifact_ids = ["artifact-skill-input"]
    context = GroundedDeepAgentRunContext(
        thread_id="thread-complex-skill",
        run_id="run-complex-skill",
        session=deep_session,
    )
    load_tool = {item.name: item for item in outer.tools}["load_skill"]
    core_loaded = json.loads(
        load_tool.func(
            skill_name="short-check",
            reason="This invocation is one bounded comparison.",
            runtime=SimpleNamespace(context=context),
        )
    )
    forced_isolation = json.loads(
        load_tool.func(
            skill_name="complex-diagnosis",
            reason="This invocation needs the declared script.",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert core_loaded["status"] == "CORE_SKILL_LOADED"
    assert "governed threshold" in core_loaded["procedure"]
    assert core_loaded["decisionAuthority"] == ("CORE_AGENT_WITH_HARNESS_GUARDS")
    assert forced_isolation["status"] == "SKILL_REQUIRES_SUBAGENT"
    assert forced_isolation["decisionAuthority"] == "HARNESS_RESOURCE_GUARD"

    prepared = outer._prepare_grounded_subagent_task(
        context,
        task=GroundedSubagentGoalContract(
            sub_goal_id="subgoal.complex.skill",
            parent_goal_ids=["metric.primary"],
            objective="隔离执行复杂诊断 Skill",
            required_outputs=["diagnosis"],
            input_artifact_refs=[],
            evidence_requirements=[
                GroundedSubagentEvidenceRequirement(
                    requirement_id="diagnosis.refs",
                    description="Return evidence refs.",
                )
            ],
            allowed_capabilities=["READ_CONTEXT"],
            budget=GroundedSubagentBudget(max_tool_calls=2, timeout_seconds=15),
            generation=1,
            skill_names=["complex-diagnosis"],
        ),
        execute_branch=lambda **kwargs: "{}",
    )
    assert prepared.grant.skill_names == ["complex-diagnosis"]
    assert prepared.job.user_payload["mountedSkill"] == ("/skills/complex-diagnosis/SKILL.md")


def test_topology_repair_keeps_rebinding_and_execution_graph_tools_visible() -> None:
    outer = runtime(CapturingFactory(action="none"))
    kernel_session = GroundedRuntimeSession(
        session_id="topology-repair",
        question="最近7天订单总数和退款单量",
        merchant_id="merchant-1",
        phase="QUERY_REPAIR_REQUIRED",
    )
    contract = GroundedQueryContract(
        question=kernel_session.question,
        status="REVISE_BINDINGS",
        query_shape="MULTI_TABLE",
        unresolved_gaps=[
            GroundedContractGap(
                code="INDEPENDENT_QUERY_SPLIT_REQUIRED",
                message="independent metrics must be split",
                evidence_kind="QUERY_TOPOLOGY",
                resolution=("REBIND_TO_COMPATIBLE_SINGLE_TABLE_OR_PROPOSE_EXECUTION_GRAPH"),
                required_capability={
                    "relationshipEvidenceRequired": False,
                    "tableGroups": [
                        {"table": "ads_merchant_profile"},
                        {"table": "dwm_trade_refund_detail_di"},
                    ],
                },
            )
        ],
    )
    kernel_session.attempts = [
        GroundedRuntimeAttempt(
            attempt_id="attempt-topology",
            contract=contract,
            status=contract.status,
            next_action=("REBIND_COMPATIBLE_CONTRACT_OR_PROPOSE_EXECUTION_GRAPH"),
        )
    ]
    session = GroundedDeepAgentSession(
        runtime=kernel_session,
        question_goal_contract=OriginalQuestionGoalContract(
            question=kernel_session.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单总数",
                ),
                MetricQuestionGoal(
                    goal_id="metric.refunds",
                    label="退款单量",
                ),
            ],
        ),
    )

    control = _grounded_semantic_read_control(session)
    visible, _ = _phase_visible_tools(
        session,
        [*outer.tools, SimpleNamespace(name="read_file")],
    )
    visible_names = {item.name for item in visible}

    assert control["status"] == "REPAIR_REQUIRED"
    assert control["repairType"] == "TOPOLOGY"
    assert control["relationshipEvidenceRequired"] is False
    assert "query_data" in visible_names
    assert "query_batch" in visible_names
    assert "propose_grounded_execution_graph" in visible_names
    assert "retrieve_knowledge" in visible_names
    assert "read_file" in visible_names
    assert "ask_human" in visible_names


def test_context_recovery_preserves_latest_repair_tool_call_pair() -> None:
    tool_call_id = "call_contract_repair"
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "query_data",
                "args": {
                    "request": {
                        "queryId": "metric.orders",
                        "readRefIds": ["semantic:topic:table:metric:orders"],
                        "goalIds": ["metric.orders"],
                    },
                },
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        name="query_data",
        tool_call_id=tool_call_id,
        content=json.dumps(
            {
                "status": "REVISE_BINDINGS",
                "repairDirective": {
                    "status": "REPAIR_REQUIRED",
                    "repairType": "BINDING",
                },
            }
        ),
    )

    exchange = _latest_grounded_repair_tool_exchange([HumanMessage(content="最近7天订单数"), ai_message, tool_message])

    assert len(exchange) == 2
    assert exchange[0].tool_calls[0]["id"] == tool_call_id
    assert exchange[1].tool_call_id == tool_call_id
    assert json.loads(exchange[1].content)["repairDirective"]["repairType"] == "BINDING"


def test_context_recovery_preserves_query_batch_observation_pair() -> None:
    tool_call_id = "call_query_batch_observation"
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "query_batch",
                "args": {"requests": []},
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        name="query_batch",
        tool_call_id=tool_call_id,
        content=json.dumps(
            {
                "status": "NEEDS_REASONING",
                "outcomes": [
                    {
                        "queryId": "detail.orders",
                        "status": "NEEDS_REASONING",
                        "observation": {
                            "code": "CORE_SQL_REQUIRED",
                            "repairReceipt": {"receiptId": "receipt-1"},
                        },
                    }
                ],
            }
        ),
    )

    exchange = _latest_grounded_repair_tool_exchange(
        [HumanMessage(content="最近7天订单明细"), ai_message, tool_message]
    )

    assert len(exchange) == 2
    assert exchange[0].tool_calls[0]["name"] == "query_batch"
    assert json.loads(exchange[1].content)["outcomes"][0]["observation"]["code"] == "CORE_SQL_REQUIRED"


def test_verified_query_can_compose_without_analysis_skill_snapshot() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory)
    kernel_session = GroundedRuntimeSession(
        session_id="verified-compose-visibility",
        question="最近7天订单总数是多少？",
        merchant_id="merchant-1",
    )
    contract = GroundedQueryContract(
        question=kernel_session.question,
        status="READY",
        query_shape="SCALAR",
    )
    kernel_session.verified_query_ledger = [
        GroundedVerifiedQueryArtifact(
            artifact_id="artifact.orders",
            generation=1,
            contract_fingerprint="contract.orders",
            sql_fingerprint="sql.orders",
            contract=contract,
            plan=QueryPlan(),
            run_result=AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"order_count": 12}],
                    result_coverage="ALL_ROWS",
                    original_row_count=1,
                )
            ),
            verified_evidence=VerifiedEvidence(passed=True),
            output_columns=["order_count"],
        )
    ]
    session = GroundedDeepAgentSession(
        runtime=kernel_session,
        question_goal_contract=OriginalQuestionGoalContract(
            question=kernel_session.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单总数",
                )
            ],
        ),
        artifact_goal_ids={"artifact.orders": ["metric.orders"]},
    )

    visible, _ = _phase_visible_tools(session, outer.tools)
    visible_names = {item.name for item in visible}

    assert visible_names == {"compose_verified_answer"}

    session.runtime.clarification = ClarificationRequest(
        question="请选择统计口径",
        stage="metric_scope",
        type="METRIC_SCOPE_REQUIRED",
    )
    visible_after_clarification, _ = _phase_visible_tools(
        session,
        outer.tools,
    )
    assert visible_after_clarification == []


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


def test_goal_evidence_closure_removes_navigation_without_query_shape_rules() -> None:
    question = "最近7天订单量和 GMV 分别是多少？"
    table_ref = "semantic:经营画像:ads_merchant_profile:detail"
    order_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    gmv_ref = "semantic:经营画像:ads_merchant_profile:metric:order_gmv_amt_1d"
    time_ref = "semantic:经营画像:ads_merchant_profile:field:pt"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="goal-evidence-closure",
            question=question,
            merchant_id="merchant-1",
            workspace_topics=["经营画像"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question=question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单量",
                    source_spans=["订单量"],
                ),
                MetricQuestionGoal(
                    goal_id="metric.gmv",
                    label="GMV",
                    source_spans=["GMV"],
                ),
                TimeWindowQuestionGoal(
                    goal_id="time.window",
                    label="最近7天",
                    source_spans=["最近7天"],
                    time_expression="最近7天",
                    applies_to_goal_ids=["metric.orders", "metric.gmv"],
                ),
            ],
        ),
        core_semantic_evidence=[
            {
                "refId": table_ref,
                "path": "topics/经营画像/tables/ads_merchant_profile/detail.json",
                "kind": "TABLE_DETAIL",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "contentSnippet": json.dumps(
                    {
                        "tableName": "ads_merchant_profile",
                        "title": "商家经营核心指标日汇总表",
                        "timeColumn": "pt",
                    },
                    ensure_ascii=False,
                ),
                "contentComplete": True,
            },
            {
                "refId": order_ref,
                "path": "topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
                "kind": "METRIC",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "contentSnippet": json.dumps(
                    {
                        "metric": {
                            "metricKey": "order_cnt_1d",
                            "aliases": ["订单量"],
                        }
                    },
                    ensure_ascii=False,
                ),
                "contentComplete": True,
            },
            {
                "refId": gmv_ref,
                "path": "topics/经营画像/tables/ads_merchant_profile/metrics/order_gmv_amt_1d.json",
                "kind": "METRIC",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "contentSnippet": json.dumps(
                    {
                        "metric": {
                            "metricKey": "order_gmv_amt_1d",
                            "aliases": ["GMV"],
                        }
                    },
                    ensure_ascii=False,
                ),
                "contentComplete": True,
            },
            {
                "refId": time_ref,
                "path": "topics/经营画像/tables/ads_merchant_profile/columns/pt.json",
                "kind": "COLUMN",
                "topic": "经营画像",
                "table": "ads_merchant_profile",
                "contentSnippet": json.dumps(
                    {
                        "key": "pt",
                        "definition": {
                            "columnName": "pt",
                            "businessName": "业务日期",
                            "role": "TIME",
                        },
                    },
                    ensure_ascii=False,
                ),
                "contentComplete": True,
            },
        ],
    )

    control = _grounded_semantic_read_control(session)

    assert control["status"] == "DECLARED_GOALS_EVIDENCE_BOUND"
    assert control["retrievalClosed"] is True
    assert "topologyRequirement" not in control
    assert control["goalBindings"] == {
        "metric.orders": [order_ref],
        "metric.gmv": [gmv_ref],
    }
    assert control["bindingSeed"]["bindingHints"] == {
        "tableRefs": [table_ref],
        "metricRefs": [order_ref, gmv_ref],
        "dimensionRefs": [],
        "timeFieldRefs": [time_ref],
        "timeFieldRef": time_ref,
    }

    tools = [
        SimpleNamespace(name=name)
        for name in (
            "grep",
            "read_file",
            "retrieve_knowledge",
            "query_data",
            "query_batch",
            "propose_grounded_execution_graph",
        )
    ]
    visible, _ = _phase_visible_tools(session, tools)

    assert {item.name for item in visible} == {
        "query_data",
        "query_batch",
        "propose_grounded_execution_graph",
    }


def test_cross_table_independent_metric_prefetch_keeps_graph_choice() -> None:
    factory = CapturingFactory(action="none")
    outer = GroundedDeepAgentRuntime(
        FakeKernel(),
        lead_model=object(),
        semantic_catalog=FakeMultiMetricSemanticCatalog(),
        agent_factory=factory,
        conversation_online_authority=StandaloneConversationAuthority(),
    )
    question = "最近7天订单量和退款金额分别是多少？"
    order_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    refund_ref = "semantic:电商退货:refunds:metric:refund_amt"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="cross-table-prefetch",
            question=question,
            merchant_id="m-1",
            workspace_topics=["经营画像", "电商退货"],
            recall=RecallBundle(
                items=[
                    RecallItem(
                        doc_id=order_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="经营画像",
                        table="ads_merchant_profile",
                        metadata={
                            "semanticRefId": order_ref,
                            "semanticPath": "topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
                            "matchedMetricLabel": "订单量",
                            "metricResolutionType": "exact_semantic_label",
                            "metricResolutionConfidence": 0.97,
                        },
                    ),
                    RecallItem(
                        doc_id=refund_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="电商退货",
                        table="refunds",
                        metadata={
                            "semanticRefId": refund_ref,
                            "semanticPath": "topics/电商退货/tables/refunds/metrics/refund_amt.json",
                            "matchedMetricLabel": "退款金额",
                            "metricResolutionType": "exact_semantic_label",
                            "metricResolutionConfidence": 0.97,
                        },
                    ),
                ]
            ),
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-cross-table-prefetch",
        run_id="run-cross-table-prefetch",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单量",
                        source_spans=["订单量"],
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="退款金额",
                        source_spans=["退款金额"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "ACCEPTED"
    assert result["semanticPrefetch"]["status"] == "PREFETCHED"
    assert len(result["semanticPrefetch"]["items"]) == 2
    visible, _ = _phase_visible_tools(
        session,
        [
            SimpleNamespace(name=name)
            for name in (
                "query_data",
                "query_batch",
                "propose_grounded_execution_graph",
                "read_file",
                "retrieve_knowledge",
            )
        ],
    )
    assert {item.name for item in visible} == {
        "query_data",
        "query_batch",
        "propose_grounded_execution_graph",
    }


def test_goal_declaration_reports_missing_recall_without_side_effects() -> None:
    class GapTrackingKernel(FakeKernel):
        def __init__(self) -> None:
            super().__init__()
            self.gap_calls = 0

        def recall_goal_gaps(
            self,
            session: GroundedRuntimeSession,
            requests: list[dict[str, Any]],
            *,
            max_requests: int,
        ) -> list[dict[str, Any]]:
            del session, requests, max_requests
            self.gap_calls += 1
            return []

    kernel = GapTrackingKernel()
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeMultiMetricSemanticCatalog(),
        agent_factory=CapturingFactory(action="none"),
        conversation_online_authority=StandaloneConversationAuthority(),
    )
    question = "订单量和退款金额分别是多少？"
    order_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="goal-recall-no-side-effect",
            question=question,
            merchant_id="m-1",
            workspace_topics=["经营画像"],
            recall=RecallBundle(
                items=[
                    RecallItem(
                        doc_id=order_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="经营画像",
                        table="ads_merchant_profile",
                        metadata={
                            "semanticRefId": order_ref,
                            "semanticPath": ("topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json"),
                            "matchedMetricLabel": "订单量",
                        },
                    )
                ]
            ),
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-goal-recall-no-side-effect",
        run_id="run-goal-recall-no-side-effect",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}

    result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单量",
                        source_spans=["订单量"],
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="退款金额",
                        source_spans=["退款金额"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["goalRecallCoverage"]["missingGoalIds"] == ["metric.refunds"]
    assert result["semanticPrefetch"] == {
        "status": "SKIPPED",
        "mode": "UNIQUE_GOAL_METRIC",
        "code": "GOAL_RECALL_COVERAGE_INCOMPLETE",
        "missingGoalIds": ["metric.refunds"],
    }
    assert kernel.gap_calls == 0
    assert session.core_semantic_evidence == []


def test_core_can_supplement_only_missing_goal_from_live_receipt() -> None:
    class SupplementalKernel(FakeKernel):
        def __init__(self) -> None:
            super().__init__()
            self.gap_requests: list[dict[str, Any]] = []

        def recall_goal_gaps(
            self,
            session: GroundedRuntimeSession,
            requests: list[dict[str, Any]],
            *,
            max_requests: int,
        ) -> list[dict[str, Any]]:
            attempted = list(requests)[:max_requests]
            self.gap_requests.extend(attempted)
            refund_ref = "semantic:电商退货:refunds:metric:refund_amt"
            session.recall = RecallBundle(
                items=[
                    *session.recall.items,
                    RecallItem(
                        doc_id=refund_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="电商退货",
                        table="refunds",
                        metadata={
                            "semanticRefId": refund_ref,
                            "semanticPath": ("topics/电商退货/tables/refunds/metrics/refund_amt.json"),
                            "matchedMetricLabel": "退款金额",
                            "metricResolutionType": "exact_semantic_label",
                            "metricResolutionConfidence": 0.97,
                            "metricResolutionAmbiguous": False,
                            "targetGoalIds": ["metric.refunds"],
                        },
                    ),
                ]
            )
            return [
                {
                    "requestId": item["requestId"],
                    "status": "COMPLETED",
                    "candidateCount": 1,
                }
                for item in attempted
            ]

    kernel = SupplementalKernel()
    outer = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=FakeMultiMetricSemanticCatalog(),
        agent_factory=CapturingFactory(action="none"),
        conversation_online_authority=StandaloneConversationAuthority(),
    )
    question = "订单量和退款金额分别是多少？"
    order_ref = "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="goal-recall-supplement",
            question=question,
            merchant_id="m-1",
            workspace_topics=["经营画像", "电商退货"],
            recall=RecallBundle(
                items=[
                    RecallItem(
                        doc_id=order_ref,
                        source_type="SEMANTIC_METRIC",
                        topic="经营画像",
                        table="ads_merchant_profile",
                        metadata={
                            "semanticRefId": order_ref,
                            "semanticPath": ("topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json"),
                            "matchedMetricLabel": "订单量",
                            "metricResolutionType": "exact_semantic_label",
                            "metricResolutionConfidence": 0.97,
                            "metricResolutionAmbiguous": False,
                        },
                    )
                ]
            ),
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-goal-recall-supplement",
        run_id="run-goal-recall-supplement",
        session=session,
    )
    tools = {item.name: item for item in outer.tools}
    declared = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单量",
                        source_spans=["订单量"],
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="退款金额",
                        source_spans=["退款金额"],
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    supplemented = json.loads(
        tools["retrieve_knowledge"].func(
            query="退款金额",
            reason="补齐缺失 Goal 的候选语义资产",
            coverage_receipt_id=(declared["goalRecallCoverage"]["receiptId"]),
            goal_ids=["metric.refunds"],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert supplemented["status"] == "COMPLETE"
    assert supplemented["goalRecallCoverage"]["missingGoalIds"] == []
    assert supplemented["semanticPrefetch"]["status"] == "PREFETCHED"
    assert len(supplemented["semanticPrefetch"]["items"]) == 2
    assert len(kernel.gap_requests) == 1
    assert kernel.gap_requests[0]["targetGoalIds"] == ["metric.refunds"]

    rejected = json.loads(
        tools["retrieve_knowledge"].func(
            query="订单量",
            reason="不应重复召回已覆盖 Goal",
            coverage_receipt_id=(supplemented["goalRecallCoverage"]["receiptId"]),
            goal_ids=["metric.orders"],
            runtime=SimpleNamespace(context=context),
        )
    )
    assert rejected["status"] == "REJECTED"
    assert rejected["code"] == "GOAL_RECALL_SUPPLEMENT_NOT_REQUIRED"


def test_goal_dependency_keeps_graph_capability_after_prefetch_closure() -> None:
    question = "先找出高退款商品，再统计这些商品的退款金额"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="dependency-graph-capability",
            question=question,
            merchant_id="m-1",
            workspace_topics=["电商退货"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question=question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.top_products",
                    label="高退款商品",
                    source_spans=["高退款商品"],
                ),
                MetricQuestionGoal(
                    goal_id="metric.refund_amount",
                    label="这些商品的退款金额",
                    source_spans=["退款金额"],
                    depends_on_goal_ids=["metric.top_products"],
                ),
            ],
        ),
    )
    table_ref = "semantic:电商退货:refunds:detail"
    session.core_semantic_evidence = [
        {
            "refId": table_ref,
            "kind": "TABLE_DETAIL",
            "topic": "电商退货",
            "table": "refunds",
            "contentSnippet": json.dumps({"tableName": "refunds"}),
            "contentComplete": True,
        },
        {
            "refId": "semantic:电商退货:refunds:metric:refund_amt",
            "kind": "METRIC",
            "topic": "电商退货",
            "table": "refunds",
            "contentSnippet": json.dumps(
                {"metric": {"aliases": ["退款金额"]}},
                ensure_ascii=False,
            ),
            "contentComplete": True,
        },
        {
            "refId": "semantic:电商退货:refunds:metric:high_refund_products",
            "kind": "METRIC",
            "topic": "电商退货",
            "table": "refunds",
            "contentSnippet": json.dumps(
                {"metric": {"aliases": ["高退款商品"]}},
                ensure_ascii=False,
            ),
            "contentComplete": True,
        },
    ]
    tools = [
        SimpleNamespace(name=name)
        for name in (
            "query_data",
            "query_batch",
            "propose_grounded_execution_graph",
            "read_file",
            "retrieve_knowledge",
        )
    ]

    visible, _ = _phase_visible_tools(session, tools)

    assert "propose_grounded_execution_graph" in {item.name for item in visible}


def test_detail_only_discovery_switches_to_query_batch_after_table_evidence() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="detail-batch-ready",
            question="最近7天订单明细和最近10天退款明细分别给我看一下。",
            merchant_id="merchant-1",
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="最近7天订单明细和最近10天退款明细分别给我看一下。",
            goals=[
                DetailQuestionGoal(
                    goal_id="g_order_detail",
                    label="最近7天订单明细",
                ),
                TimeWindowQuestionGoal(
                    goal_id="g_order_time",
                    label="最近7天",
                    time_expression="最近7天",
                    applies_to_goal_ids=["g_order_detail"],
                ),
                DetailQuestionGoal(
                    goal_id="g_refund_detail",
                    label="最近10天退款明细",
                ),
                TimeWindowQuestionGoal(
                    goal_id="g_refund_time",
                    label="最近10天",
                    time_expression="最近10天",
                    applies_to_goal_ids=["g_refund_detail"],
                ),
            ],
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:电商交易:orders:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商交易",
                "table": "orders",
                "contentComplete": True,
            },
            {
                "refId": "semantic:电商退货:refunds:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商退货",
                "table": "refunds",
                "contentComplete": True,
            },
        ],
    )

    control = _grounded_semantic_read_control(session)

    assert control["status"] == "READY_FOR_QUERY_BATCH"
    assert control["nextAction"] == "CORE_REACT_DECIDES"
    assert control["decisionMode"] == "CALLER_REACT"


def test_query_readiness_and_batch_observations_are_advisory_to_react_caller() -> None:
    question = "最近7天订单明细和最近10天退款明细分别给我看一下。"
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="batch-read-control",
            question=question,
            merchant_id="merchant-1",
            workspace_topics=["电商交易", "电商退货"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question=question,
            goals=[
                DetailQuestionGoal(goal_id="g.orders", label="订单明细"),
                TimeWindowQuestionGoal(
                    goal_id="g.orders.time",
                    label="最近7天",
                    time_expression="最近7天",
                    applies_to_goal_ids=["g.orders"],
                ),
                DetailQuestionGoal(goal_id="g.refunds", label="退款明细"),
                TimeWindowQuestionGoal(
                    goal_id="g.refunds.time",
                    label="最近10天",
                    time_expression="最近10天",
                    applies_to_goal_ids=["g.refunds"],
                ),
            ],
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:电商交易:orders:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商交易",
                "table": "orders",
                "contentComplete": True,
            },
            {
                "refId": "semantic:电商退货:refunds:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商退货",
                "table": "refunds",
                "contentComplete": True,
            },
        ],
    )
    tools = [
        SimpleNamespace(name=name)
        for name in (
            "ls",
            "grep",
            "read_file",
            "retrieve_knowledge",
            "query_data",
            "query_batch",
        )
    ]

    visible, _ = _phase_visible_tools(session, tools)
    assert {item.name for item in visible} == set(
        {
            "ls",
            "grep",
            "read_file",
            "retrieve_knowledge",
            "query_data",
            "query_batch",
        }
    )

    session.latest_query_batch_observations = [
        {
            "queryId": "orders",
            "observation": {
                "stage": "CONTRACT",
                "code": "SEMANTIC_REF_NOT_READ",
                "readNext": [
                    {
                        "refId": "semantic:电商交易:orders:field:pt",
                        "path": "topics/电商交易/tables/orders/columns/pt.json",
                    }
                ],
            },
        }
    ]
    visible, _ = _phase_visible_tools(session, tools)
    assert {item.name for item in visible} == set(
        {
            "ls",
            "grep",
            "read_file",
            "retrieve_knowledge",
            "query_data",
            "query_batch",
        }
    )

    session.latest_query_batch_observations = [
        {
            "queryId": "refunds",
            "observation": {
                "stage": "CONTRACT",
                "code": "DEPENDENT_QUERY_REQUIRES_SERIAL_EXECUTION",
                "nextActions": ["RETRY_QUERY_DATA_SERIAL"],
            },
        }
    ]
    visible, _ = _phase_visible_tools(session, tools)
    assert {item.name for item in visible} == set(
        {
            "ls",
            "grep",
            "read_file",
            "retrieve_knowledge",
            "query_data",
            "query_batch",
        }
    )


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
    assert factory.graph.invocations[0]["messages"][0]["content"] == (
        "工单量最多的商品"
    )
    first = factory.graph.contexts[0].session.trusted_bootstrap_context
    assert "question" not in first
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
        "headersDisclosedAfterVerifiedInputSnapshot": True,
        "queryCollectionClosedBySkill": False,
        "authoritativeSkillExecution": "SERIAL_ONLY",
        "placementRule": {
            "decisionAuthority": "CORE_AGENT",
            "CORE": "choose load_skill for a short bounded invocation",
            "SUBAGENT": "choose run_skill for a complex or long invocation",
            "harnessOverride": ("scripts, oversized procedures and hard isolation requirements"),
        },
        "maxVerifiedSkillArtifacts": 4,
        "skillProseCountsAsGoalCoverage": False,
        "retryRule": "SAME_SUB_GOAL_NEXT_GENERATION",
    }
    root_prompt_trace = response.debug_trace["harness"]["promptManagement"][
        "root"
    ]
    assert root_prompt_trace["promptId"] == "grounded.core.system"
    assert root_prompt_trace["version"] == "v1"
    assert root_prompt_trace["templateFingerprint"]
    assert root_prompt_trace["renderFingerprint"]
    assert response.clarification is not None
    assert response.debug_trace["harness"]["legacyFallbackUsed"] is False


def test_run_uses_retrieval_question_without_mutating_core_question() -> None:
    factory = CapturingFactory()
    kernel = FakeKernel()
    outer = runtime(factory, kernel)
    outer.conversation_online_authority = ContextualizedConversationAuthority()

    outer.run("那退款呢？", "m-1")

    assert kernel.recall_queries == ["最近7天的退款情况"]
    assert factory.graph.invocations[0]["messages"][0]["content"] == "那退款呢？"
    first = factory.graph.contexts[0].session.trusted_bootstrap_context
    assert "question" not in first
    assert "retrievalQuestion" not in first["trustedConversationContext"][
        "resolution"
    ]


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


def test_unresolved_topic_route_opens_discovery_without_merchant_selection() -> None:
    class NoTopicKernel(FakeKernel):
        def route_topic(
            self,
            session: GroundedRuntimeSession,
        ) -> TopicRoutingDecision:
            self.route_calls += 1
            session.routing = TopicRoutingDecision(
                clarification_required=True,
                routing_mode="semantic_topic_open_discovery",
                reason="no bounded published Topic candidate",
            )
            session.workspace_topics = []
            return session.routing

        def recall_navigation(
            self,
            session: GroundedRuntimeSession,
            *,
            query: str = "",
            **kwargs: Any,
        ) -> RecallBundle:
            raise AssertionError("recall must not run without a Topic scope")

    factory = CapturingFactory(action="none")
    kernel = NoTopicKernel()
    outer = runtime(factory, kernel)

    response = outer.run("今天天气怎么样？", "m-1")

    assert kernel.route_calls == 1
    assert len(factory.graph.invocations) == 1
    assert factory.graph.invocations[0]["messages"][0]["content"] == (
        "今天天气怎么样？"
    )
    first = factory.graph.contexts[0].session.trusted_bootstrap_context
    assert first["topicRouting"]["routingMode"] == "open_discovery"
    assert first["topicRouting"]["clarificationRequired"] is False
    assert first["topicL0Manifests"] == []
    assert response.clarification is None


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


def test_transient_provider_failure_is_returned_as_controlled_operational_failure() -> None:
    factory = CapturingFactory(action="none")
    outer = runtime(factory, FakeKernel())

    class RateLimitError(RuntimeError):
        status_code = 429

    class RateLimitedGraph:
        @staticmethod
        def invoke(payload: dict[str, Any], *, config: Any, context: Any) -> None:
            del payload, config, context
            raise RateLimitError("too many requests")

    outer.deep_agent_graph = RateLimitedGraph()
    response = outer.run("工单量", "m-1")
    failure = response.debug_trace["harness"]["operationalFailure"]

    assert failure["code"] == "GROUNDED_PROVIDER_TRANSIENT_FAILURE"
    assert failure["reason"] == "rate_limit"
    assert failure["retryable"] is True
    assert "模型服务暂时不可用" in response.answer


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


def test_plain_core_answer_cannot_bypass_compose_attestation() -> None:
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
    response = outer.run("工单量", "m-1")

    assert response.answer != "我直接声称这是最终答案。"
    assert response.debug_trace["harness"]["operationalFailure"]["code"] == "GROUNDED_ANSWER_ATTESTATION_MISSING"


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


def test_topic_index_allows_navigation_without_a_structured_gap() -> None:
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

    assert session.can_expand_topic() is False
    session.topic_index_read = True
    assert session.can_expand_topic() is True
    session.mark_topic_expanded()
    assert session.can_expand_topic() is True


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


def test_tool_loop_guard_fuses_duplicate_call_and_unlocks_after_progress() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="tool-loop-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="工单量",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.tickets",
                    label="工单量",
                )
            ],
        ),
    )
    context = GroundedDeepAgentRunContext(
        thread_id="tool-loop-thread",
        run_id="tool-loop-run",
        session=session,
    )
    calls = {"count": 0}

    def invoke(call_id: str) -> ToolMessage:
        request = SimpleNamespace(
            tool_call={
                "id": call_id,
                "name": "ls",
                "args": {"path": "/knowledge/topics/客服工单"},
            },
            runtime=SimpleNamespace(context=context),
        )

        def handler(_: Any) -> ToolMessage:
            calls["count"] += 1
            return ToolMessage(
                content="manifest.json",
                name="ls",
                tool_call_id=call_id,
            )

        return middleware.wrap_tool_call(request, handler)

    first = invoke("call-ls-1")
    second = invoke("call-ls-2")
    second_payload = json.loads(str(second.content))

    assert first.status == "success"
    assert calls["count"] == 1
    assert second.status == "error"
    assert second.tool_call_id == "call-ls-2"
    assert second_payload["code"] == "TOOL_CALL_NO_PROGRESS"
    assert second_payload["previousToolCallId"] == "call-ls-1"

    visible, _ = _phase_visible_tools(
        session,
        [
            SimpleNamespace(name="ls"),
            SimpleNamespace(name="query_data"),
        ],
    )
    assert {item.name for item in visible} == {"query_data"}

    session.core_semantic_evidence.append(
        {
            "refId": "semantic:客服工单:tickets:detail",
            "contentHash": "new-evidence",
            "contentComplete": True,
        }
    )
    visible_after_progress, _ = _phase_visible_tools(
        session,
        [
            SimpleNamespace(name="ls"),
            SimpleNamespace(name="query_data"),
        ],
    )
    assert "ls" in {item.name for item in visible_after_progress}


def test_tool_loop_guard_is_not_limited_to_filesystem_tools() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="generic-tool-loop-session",
            question="工单量",
            merchant_id="m-1",
        )
    )
    context = GroundedDeepAgentRunContext(
        thread_id="generic-tool-loop-thread",
        run_id="generic-tool-loop-run",
        session=session,
    )
    calls = {"count": 0}

    def invoke(call_id: str, reason: str) -> ToolMessage:
        request = SimpleNamespace(
            tool_call={
                "id": call_id,
                "name": "query_data",
                "args": {
                    "read_ref_ids": ["semantic:metric:tickets"],
                    "binding_hints": {"metricRefs": ["semantic:metric:tickets"]},
                    "goal_ids": ["metric.tickets"],
                    "reason": reason,
                },
            },
            runtime=SimpleNamespace(context=context),
        )

        def handler(_: Any) -> ToolMessage:
            calls["count"] += 1
            return ToolMessage(
                content='{"status":"REJECTED"}',
                name="query_data",
                tool_call_id=call_id,
            )

        return middleware.wrap_tool_call(request, handler)

    invoke("call-contract-1", "first wording")
    repeated = invoke("call-contract-2", "paraphrased wording")

    assert calls["count"] == 1
    assert json.loads(str(repeated.content))["code"] == ("TOOL_CALL_NO_PROGRESS")


def test_tool_loop_guard_warns_then_hard_blocks_repeated_tool_type() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(
        GroundedSemanticBackend(FakeSemanticCatalog()),
        settings=SimpleNamespace(
            middleware_loop_guard_threshold=3,
            middleware_tool_repeat_warning_threshold=5,
            middleware_tool_repeat_hard_stop_threshold=6,
            middleware_tool_type_warning_threshold=3,
            middleware_tool_type_hard_stop_threshold=4,
            middleware_tool_loop_window_size=6,
        ),
    )
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="tool-type-loop-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="工单量",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.tickets",
                    label="工单量",
                )
            ],
        ),
    )
    context = GroundedDeepAgentRunContext(
        thread_id="tool-type-loop-thread",
        run_id="tool-type-loop-run",
        session=session,
    )
    handler_calls = 0

    def invoke(call_number: int) -> ToolMessage:
        request = SimpleNamespace(
            tool_call={
                "id": "call-ls-%s" % call_number,
                "name": "ls",
                "args": {
                    "path": (
                        "/knowledge/topics/客服工单/%s"
                        % call_number
                    )
                },
            },
            runtime=SimpleNamespace(context=context),
        )

        def handler(_: Any) -> ToolMessage:
            nonlocal handler_calls
            handler_calls += 1
            return ToolMessage(
                content="ok",
                name="ls",
                tool_call_id="call-ls-%s" % call_number,
            )

        return middleware.wrap_tool_call(request, handler)

    assert invoke(1).status == "success"
    assert invoke(2).status == "success"
    warning = invoke(3)
    hard_stop = invoke(4)

    assert handler_calls == 2
    assert json.loads(str(warning.content))["code"] == (
        "TOOL_LOOP_WARNING"
    )
    assert json.loads(str(hard_stop.content))["code"] == (
        "TOOL_LOOP_HARD_STOP"
    )
    assert session.tool_loop_blocked == {"ls"}

    visible, _ = _phase_visible_tools(
        session,
        [
            SimpleNamespace(name="ls"),
            SimpleNamespace(name="query_data"),
        ],
    )
    assert {item.name for item in visible} == {"query_data"}


def test_tool_call_exception_is_recovered_with_original_call_id() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="tool-recovery-session",
            question="工单量",
            merchant_id="m-1",
        )
    )
    request = SimpleNamespace(
        tool_call={
            "id": "call-tool-error",
            "name": "retrieve_knowledge",
            "args": {"query": "工单量", "reason": "find metric"},
        },
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="tool-recovery-thread",
                run_id="tool-recovery-run",
                session=session,
            )
        ),
    )

    result = middleware.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    payload = json.loads(str(result.content))

    assert result.status == "error"
    assert result.tool_call_id == "call-tool-error"
    assert payload["code"] == "TOOL_CALL_FAILED"
    assert payload["toolCallId"] == "call-tool-error"


def test_ready_contract_does_not_close_react_semantic_read_boundary() -> None:
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

    assert called["handler"] is True
    assert result.status != "error"


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


def test_tool_call_repair_injects_interrupted_result_without_mutating_checkpoint() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="tool-repair-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    messages = [
        HumanMessage(content="查询工单量"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-query",
                    "name": "query_data",
                    "args": {"request": {"queryId": "tickets"}},
                    "type": "tool_call",
                }
            ],
        ),
        HumanMessage(content="继续"),
    ]
    request = model_call_request(messages, session=session)
    captured: dict[str, Any] = {}

    def handler(updated: Any) -> str:
        captured["request"] = updated
        return "ok"

    result = GroundedToolCallRepairMiddleware().wrap_model_call(
        request,
        handler,
    )

    assert result == "ok"
    assert request.messages is messages
    assert len(messages) == 3
    repaired = captured["request"].messages
    assert [item.type for item in repaired] == ["human", "ai", "tool", "human"]
    assert repaired[2].tool_call_id == "call-query"
    assert repaired[2].name == "query_data"
    assert repaired[2].status == "error"
    assert "interrupted" in str(repaired[2].content)
    report = session.tool_call_recovery_reports[-1]
    assert report["status"] == "REPAIRED"
    assert report["injectedToolResultCount"] == 1
    assert report["isolatedToolResultCount"] == 0
    assert report["checkpointMutated"] is False


def test_tool_call_repair_preserves_valid_parallel_result_and_fills_only_missing_one() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="parallel-tool-repair-session",
            question="同时读取两个结果",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    existing_result = ToolMessage(
        content='{"status":"VERIFIED"}',
        tool_call_id="call-query",
        name="query_data",
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-query",
                    "name": "query_data",
                    "args": {"request": {"queryId": "tickets"}},
                    "type": "tool_call",
                },
                {
                    "id": "call-read",
                    "name": "read_file",
                    "args": {"file_path": "/knowledge/topic.json"},
                    "type": "tool_call",
                },
            ],
        ),
        existing_result,
    ]
    request = model_call_request(messages, session=session)
    captured: dict[str, Any] = {}

    GroundedToolCallRepairMiddleware().wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    repaired = captured["request"].messages
    assert [item.type for item in repaired] == ["ai", "tool", "tool"]
    assert repaired[1].tool_call_id == "call-read"
    assert repaired[1].status == "error"
    assert repaired[2] is existing_result
    report = session.tool_call_recovery_reports[-1]
    assert [
        item["toolCallId"]
        for item in report["injectedToolResults"]
    ] == ["call-read"]


def test_tool_call_repair_isolates_orphan_duplicate_and_out_of_order_results() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="orphan-tool-repair-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    valid_result = ToolMessage(
        content='{"status":"VERIFIED"}',
        tool_call_id="call-query",
        name="query_data",
    )
    messages = [
        HumanMessage(content="查询工单量"),
        ToolMessage(
            content="orphan",
            tool_call_id="call-orphan",
            name="read_file",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-query",
                    "name": "query_data",
                    "args": {"request": {"queryId": "tickets"}},
                    "type": "tool_call",
                }
            ],
        ),
        valid_result,
        ToolMessage(
            content="duplicate",
            tool_call_id="call-query",
            name="query_data",
        ),
        HumanMessage(content="下一步"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-late",
                    "name": "read_file",
                    "args": {"file_path": "/knowledge/topic.json"},
                    "type": "tool_call",
                }
            ],
        ),
        HumanMessage(content="工具还没完成"),
        ToolMessage(
            content="late",
            tool_call_id="call-late",
            name="read_file",
        ),
    ]
    request = model_call_request(messages, session=session)
    captured: dict[str, Any] = {}

    GroundedToolCallRepairMiddleware().wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    repaired = captured["request"].messages
    assert repaired[:5] == [
        messages[0],
        messages[2],
        valid_result,
        messages[5],
        messages[6],
    ]
    assert repaired[5].type == "tool"
    assert repaired[5].tool_call_id == "call-late"
    assert repaired[5].status == "error"
    assert repaired[6] is messages[7]
    report = session.tool_call_recovery_reports[-1]
    assert report["injectedToolResultCount"] == 1
    assert report["isolatedToolResultCount"] == 3
    assert {
        item["reason"]
        for item in report["isolatedToolResults"]
    } == {
        "ORPHAN_TOOL_RESULT",
        "DUPLICATE_TOOL_RESULT",
        "OUT_OF_ORDER_TOOL_RESULT",
    }


def test_tool_call_repair_leaves_well_formed_history_unchanged() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="valid-tool-history-session",
            question="工单量",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        )
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-query",
                    "name": "query_data",
                    "args": {"request": {"queryId": "tickets"}},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"status":"VERIFIED"}',
            tool_call_id="call-query",
            name="query_data",
        ),
    ]
    request = model_call_request(messages, session=session)
    captured: dict[str, Any] = {}

    GroundedToolCallRepairMiddleware().wrap_model_call(
        request,
        lambda updated: captured.setdefault("request", updated),
    )

    assert captured["request"] is request
    assert session.tool_call_recovery_reports == []


def test_context_middleware_keeps_messages_and_tools_available_to_react_caller() -> None:
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
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="工单量",
            goals=[MetricQuestionGoal(goal_id="metric.tickets", label="工单量")],
        ),
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
                    "name": "query_data",
                    "args": {"request": {"queryId": "tickets"}},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"status":"VERIFIED"}',
            tool_call_id="call-execute",
            name="query_data",
        ),
    ]
    tools = [
        SimpleNamespace(name="read_file", description="read", args_schema=None),
        SimpleNamespace(name="grep", description="grep", args_schema=None),
        SimpleNamespace(
            name="query_data",
            description="execute",
            args_schema=None,
        ),
        SimpleNamespace(
            name="query_batch",
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
    assert [_tool.name for _tool in updated.tools] == [
        "read_file",
        "grep",
        "query_data",
        "query_batch",
    ]
    report = session.core_context_reports[-1]
    assert report["compactionTriggered"] is False
    assert report["savedChars"] == 0
    assert report["semanticReadMessagesCompacted"] == 0
    assert report["removedTools"] == []


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
            "query_data",
            "query_batch",
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


def test_native_task_rejects_unguarded_dispatch_before_subagent_execution() -> None:
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
    called = {"handler": False}
    request = SimpleNamespace(
        tool_call={
            "id": "call-task",
            "name": "task",
            "args": {"subagent_type": "general-purpose", "description": "{}"},
        },
        runtime=SimpleNamespace(context=None),
    )

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(content="unexpected", tool_call_id="call-task")

    result = middleware.wrap_tool_call(request, handler)

    assert result.status == "error"
    assert json.loads(result.content)["code"] == "NATIVE_TASK_SUBAGENT_TYPE_NOT_ALLOWED"
    assert called["handler"] is False


def test_native_task_accepts_only_read_context_contract() -> None:
    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="native-task-boundary",
            question="主指标是多少",
            merchant_id="merchant-1",
            workspace_topics=["客服工单"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="主指标是多少",
            goals=[MetricQuestionGoal(goal_id="metric.primary", label="主指标")],
        ),
    )
    contract = _subagent_goal()
    description = json.dumps(
        {
            "protocol": "grounded_native_task.v1",
            "subGoalContract": contract.model_dump(by_alias=True, mode="json"),
        },
        ensure_ascii=False,
    )
    middleware = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(FakeSemanticCatalog()))
    request = SimpleNamespace(
        tool_call={
            "id": "call-native-task",
            "name": "task",
            "args": {
                "subagent_type": "grounded-researcher",
                "description": description,
            },
        },
        runtime=SimpleNamespace(
            context=GroundedDeepAgentRunContext(
                thread_id="native-task-thread",
                run_id="native-task-run",
                session=session,
            )
        ),
    )

    called = {"handler": False}

    def handler(_: Any) -> ToolMessage:
        called["handler"] = True
        return ToolMessage(
            content=json.dumps({"status": "COMPLETED"}),
            tool_call_id="call-native-task",
        )

    result = middleware.wrap_tool_call(request, handler)

    assert called["handler"] is True
    assert json.loads(result.content)["status"] == "COMPLETED"


def _subagent_goal(
    *,
    sub_goal_id: str = "subgoal.read.metric",
    generation: int = 1,
    parent_goal_ids: list[str] | None = None,
) -> GroundedSubagentGoalContract:
    return GroundedSubagentGoalContract(
        sub_goal_id=sub_goal_id,
        parent_goal_ids=parent_goal_ids or ["metric.primary"],
        objective="隔离阅读当前 Topic 证据并返回候选引用",
        required_outputs=["finding"],
        input_artifact_refs=[],
        evidence_requirements=[
            GroundedSubagentEvidenceRequirement(
                requirement_id="evidence.semantic.refs",
                description="Return exact semantic refs for Root review.",
                accepted_ref_types=["SEMANTIC_REF"],
            )
        ],
        allowed_capabilities=["READ_CONTEXT"],
        budget=GroundedSubagentBudget(
            max_tool_calls=3,
            timeout_seconds=9,
        ),
        generation=generation,
    )


def test_core_dynamically_dispatches_goal_bound_task_with_exact_capability_grant() -> None:
    factory = CapturingFactory()
    outer = runtime(factory)
    deep_session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="subagent-dispatch",
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
    context = GroundedDeepAgentRunContext(
        thread_id="thread-subagent-dispatch",
        run_id="run-subagent-dispatch",
        session=deep_session,
    )

    class Isolated:
        def __init__(self) -> None:
            self.jobs: list[Any] = []

        def run(self, job: Any, *, on_progress: Any = None) -> IsolatedSubagentResult:
            self.jobs.append(job)
            if on_progress is not None:
                on_progress("subagent", "started", job.job_id)
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={"runId": job.job_id},
                raw_output=json.dumps(
                    {
                        "summary": "候选引用已定位，等待 Root 复核。",
                        "finding": "semantic:客服工单:tickets:detail",
                        "evidenceRefs": ["semantic:客服工单:tickets:detail"],
                        "gaps": [],
                        "recommendedNextAction": "ROOT_READ_EXACT_REF",
                        "proposedSubGoals": [],
                        "evidenceGaps": [],
                    },
                    ensure_ascii=False,
                ),
                update_count=2,
            )

    isolated = Isolated()
    outer.subagent_runtime = isolated  # type: ignore[assignment]
    tools = {item.name: item for item in outer.tools}
    response = json.loads(
        tools["delegate_grounded_tasks"].func(
            plan=GroundedSubagentDispatchPlan(
                tasks=[_subagent_goal()],
                reason="Long semantic investigation benefits from isolation.",
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert response["status"] == "COMPLETED"
    assert response["outputAuthority"] == "ADVISORY"
    assert response["tasks"][0]["subGoalId"] == "subgoal.read.metric"
    assert response["tasks"][0]["generation"] == 1
    assert response["tasks"][0]["advisoryOutput"]["finding"].startswith("semantic:")
    assert len(isolated.jobs) == 1
    job = isolated.jobs[0]
    assert job.capability_grant.fingerprint_valid()
    assert job.capability_grant.parent_goal_ids == ["metric.primary"]
    assert job.capability_grant.generation == 1
    assert job.capability_grant.allowed_tool_names == ["grep", "ls", "read_file"]
    assert job.capability_grant.query_branch_ids == []
    assert job.capability_grant.skill_names == []
    assert job.capability_grant.max_tool_calls == 3
    assert job.model_timeout_seconds == 9
    assert job.tools == []
    assert job.user_payload["promptTrace"]["promptId"] == (
        "grounded.subagent.system"
    )
    assert deep_session.core_semantic_evidence == []


def test_subagent_retry_requires_next_generation_and_parent_goal_binding() -> None:
    factory = CapturingFactory()
    outer = runtime(factory)
    deep_session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="subagent-generation",
            question="主指标是多少",
            merchant_id="merchant-1",
            workspace_topics=["客服工单"],
        ),
        question_goal_contract=OriginalQuestionGoalContract(
            question="主指标是多少",
            goals=[MetricQuestionGoal(goal_id="metric.primary", label="主指标")],
        ),
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-subagent-generation",
        run_id="run-subagent-generation",
        session=deep_session,
    )

    class Isolated:
        def run(self, job: Any, *, on_progress: Any = None) -> IsolatedSubagentResult:
            return IsolatedSubagentResult(
                job_id=job.job_id,
                thread_id=job.thread_id,
                checkpoint={},
                raw_output=json.dumps(
                    {
                        "summary": "done",
                        "finding": "ref",
                        "evidenceRefs": [],
                        "gaps": [],
                        "recommendedNextAction": "ROOT_REVIEW",
                        "proposedSubGoals": [],
                        "evidenceGaps": [],
                    }
                ),
                update_count=1,
            )

    outer.subagent_runtime = Isolated()  # type: ignore[assignment]
    delegate = {item.name: item for item in outer.tools}["delegate_grounded_tasks"]

    first = json.loads(
        delegate.func(
            plan=GroundedSubagentDispatchPlan(tasks=[_subagent_goal()]),
            runtime=SimpleNamespace(context=context),
        )
    )
    repeated_generation = json.loads(
        delegate.func(
            plan=GroundedSubagentDispatchPlan(tasks=[_subagent_goal()]),
            runtime=SimpleNamespace(context=context),
        )
    )
    next_generation = json.loads(
        delegate.func(
            plan=GroundedSubagentDispatchPlan(tasks=[_subagent_goal(generation=2)]),
            runtime=SimpleNamespace(context=context),
        )
    )
    unknown_parent = json.loads(
        delegate.func(
            plan=GroundedSubagentDispatchPlan(
                tasks=[
                    _subagent_goal(
                        sub_goal_id="subgoal.unknown.parent",
                        parent_goal_ids=["metric.unknown"],
                    )
                ]
            ),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert first["status"] == "COMPLETED"
    assert repeated_generation["code"] == "SUBAGENT_GOAL_GENERATION_INVALID"
    assert repeated_generation["issues"][0]["expectedGeneration"] == 2
    assert next_generation["status"] == "COMPLETED"
    assert unknown_parent["code"] == "SUBAGENT_PARENT_GOAL_UNKNOWN"


def test_query_subagent_receives_only_one_prepared_branch_execution_tool() -> None:
    factory = CapturingFactory()
    outer = runtime(factory)
    root_runtime = GroundedRuntimeSession(
        session_id="root-query-subagent",
        question="复杂指标是多少",
        merchant_id="merchant-1",
        workspace_topics=["客服工单"],
    )
    branch_runtime = GroundedRuntimeSession(
        session_id="branch-query-subagent",
        question="计算复杂指标",
        merchant_id="merchant-1",
        workspace_topics=["客服工单"],
        active_generation=3,
        active_execution_mode=GroundedExecutionMode.CORE_SQL_REQUIRED,
        active_contract=GroundedQueryContract(
            question="计算复杂指标",
            topics=["客服工单"],
            status="READY",
            query_shape="COMPLEX",
            evidence_refs=["semantic:客服工单:tickets:detail"],
        ),
    )
    branch_context = GroundedQueryBranchContext(
        spec=GroundedQueryBranchSpec(
            query_id="query.complex",
            objective="计算复杂指标",
            goal_ids=["metric.primary"],
            topic_scope=["客服工单"],
            evidence_ref_ids=["semantic:客服工单:tickets:detail"],
        ),
        runtime=branch_runtime,
        budget=GroundedBranchBudget(
            "query.complex",
            GroundedBranchBudgetLimits(),
        ),
        status="PREPARED",
    )
    deep_session = GroundedDeepAgentSession(
        runtime=root_runtime,
        question_goal_contract=OriginalQuestionGoalContract(
            question="复杂指标是多少",
            goals=[MetricQuestionGoal(goal_id="metric.primary", label="复杂指标")],
        ),
        query_branch_contexts={"query.complex": branch_context},
        parallel_branches={"query.complex": branch_runtime},
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-query-subagent",
        run_id="run-query-subagent",
        session=deep_session,
    )
    calls: list[dict[str, Any]] = []

    def execute_branch(**kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "status": "VERIFIED",
                "queryId": kwargs["queries"][0].query_id,
            }
        )

    prepared = outer._prepare_grounded_subagent_task(
        context,
        task=GroundedSubagentGoalContract(
            sub_goal_id="subgoal.query.complex",
            parent_goal_ids=["metric.primary"],
            objective="隔离完成复杂 SQL 推理并执行已准备分支",
            required_outputs=["queryReceipt"],
            input_artifact_refs=["query-branch:query.complex"],
            evidence_requirements=[
                GroundedSubagentEvidenceRequirement(
                    requirement_id="verified.query.receipt",
                    description="Return the verified branch receipt.",
                    accepted_ref_types=["VERIFIED_QUERY_ARTIFACT"],
                )
            ],
            allowed_capabilities=["QUERY_BRANCH"],
            budget=GroundedSubagentBudget(max_tool_calls=2, timeout_seconds=20),
            generation=1,
            query_branch_ids=["query.complex"],
        ),
        execute_branch=execute_branch,
    )

    assert branch_context.runtime is branch_runtime
    assert [item.name for item in prepared.job.tools] == ["execute_assigned_query"]
    assert prepared.grant.query_branch_ids == ["query.complex"]
    assert prepared.grant.allowed_tool_names == ["execute_assigned_query"]
    assert prepared.grant.artifact_ids == []
    assert prepared.job.user_payload["queryBranch"]["queryId"] == "query.complex"
    result = json.loads(
        prepared.job.tools[0].func(
            sql="SELECT governed_metric FROM tickets",
            rationale="Use the exact prepared Contract.",
            evidence_ref_ids=["semantic:客服工单:tickets:detail"],
        )
    )

    assert result == {"status": "VERIFIED", "queryId": "query.complex"}
    assert len(calls) == 1
    assert [item.query_id for item in calls[0]["queries"]] == ["query.complex"]
    assert calls[0]["runtime"].context is context


def test_subagent_uses_shared_query_data_with_goal_scope_and_root_adoption() -> None:
    outer = runtime(CapturingFactory(action="none"))
    root_runtime = GroundedRuntimeSession(
        session_id="root-shared-query-data",
        question="分析复杂指标",
        merchant_id="merchant-1",
        workspace_topics=["客服工单"],
    )
    deep_session = GroundedDeepAgentSession(
        runtime=root_runtime,
        question_goal_contract=OriginalQuestionGoalContract(
            question="分析复杂指标",
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.primary",
                    label="复杂指标",
                ),
                MetricQuestionGoal(
                    goal_id="metric.outside",
                    label="未授权指标",
                ),
            ],
        ),
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-shared-query-data",
        run_id="run-shared-query-data",
        session=deep_session,
    )
    calls: list[dict[str, Any]] = []

    def shared_query_data(**kwargs: Any) -> str:
        calls.append(kwargs)
        request = kwargs["request"]
        scoped_context = kwargs["runtime"].context
        assert scoped_context.session.runtime is not root_runtime
        assert scoped_context.session.runtime.merchant_id == "merchant-1"
        if len(calls) == 1:
            return json.dumps(
                {
                    "status": "NEEDS_REASONING",
                    "queryId": request.query_id,
                    "observation": {
                        "stage": "SQL_GENERATION",
                        "code": "CORE_SQL_CANDIDATE_REQUIRED",
                        "retryable": True,
                        "repairReceipt": {
                            "version": "query_repair_receipt.v1",
                            "receiptId": "repair-subagent-1",
                            "queryId": request.query_id,
                            "callerId": "subagent-bound",
                            "stage": "SQL_GENERATION",
                            "code": "CORE_SQL_CANDIDATE_REQUIRED",
                            "attemptCount": 1,
                            "contractGeneration": 1,
                            "contractFingerprint": "c" * 64,
                            "sqlAstFingerprint": "",
                            "allowedNextActions": ["SUBMIT_SQL_CANDIDATE"],
                            "receiptFingerprint": "test",
                        },
                    },
                }
            )
        assert request.repair_receipt is not None
        return json.dumps(
            {
                "status": "VERIFIED",
                "queryId": request.query_id,
                "artifactIds": ["artifact-subagent-query"],
            }
        )

    outer.kernel.adopt_verified_branches = (  # type: ignore[attr-defined]
        lambda parent, branches: [SimpleNamespace(artifact_id="artifact-subagent-query")]
    )
    prepared = outer._prepare_grounded_subagent_task(
        context,
        task=GroundedSubagentGoalContract(
            sub_goal_id="subgoal.shared.query",
            parent_goal_ids=["metric.primary"],
            objective="隔离完成复杂查询并自行处理 Observation",
            required_outputs=["verifiedQueryArtifact"],
            input_artifact_refs=[],
            evidence_requirements=[
                GroundedSubagentEvidenceRequirement(
                    requirement_id="verified.query",
                    description="Return one verified query artifact.",
                    accepted_ref_types=["VERIFIED_QUERY_ARTIFACT"],
                )
            ],
            allowed_capabilities=["READ_CONTEXT", "QUERY_DATA"],
            budget=GroundedSubagentBudget(
                max_tool_calls=3,
                timeout_seconds=20,
            ),
            generation=1,
        ),
        execute_branch=lambda **kwargs: "{}",
        query_data_call=shared_query_data,
    )

    assert prepared.grant.capabilities == ["READ_CONTEXT", "QUERY_DATA"]
    assert prepared.grant.allowed_tool_names == [
        "grep",
        "ls",
        "query_data",
        "read_file",
    ]
    assert [item.name for item in prepared.job.tools] == ["query_data"]
    query_tool = prepared.job.tools[0]
    first = json.loads(
        query_tool.func(
            request={
                "queryId": "query.subagent.complex",
                "goalIds": ["metric.primary"],
            }
        )
    )
    assert first["status"] == "NEEDS_REASONING"
    verified = json.loads(
        query_tool.func(
            request={
                "queryId": "query.subagent.complex",
                "goalIds": ["metric.primary"],
                "sqlCandidate": {
                    "sql": "SELECT COUNT(*) AS metric_value FROM tickets",
                    "rationale": "Implement the returned Contract.",
                },
                "repairReceipt": first["observation"]["repairReceipt"],
            }
        )
    )
    denied = json.loads(
        query_tool.func(
            request={
                "queryId": "query.outside",
                "goalIds": ["metric.outside"],
            }
        )
    )

    assert verified["status"] == "VERIFIED"
    assert verified["adoptedIntoRoot"] is True
    assert deep_session.artifact_goal_ids["artifact-subagent-query"] == ["metric.primary"]
    assert denied["status"] == "DENIED"
    assert denied["code"] == "SUBAGENT_QUERY_GOAL_SCOPE_MISMATCH"
    assert len(calls) == 2


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


def test_multi_metric_thin_recall_retains_candidates_for_each_metric_phrase() -> None:
    keywords = SimpleNamespace(
        mentions=[
            SimpleNamespace(
                kind="metric",
                phrase="订单总数",
                canonical_key="order_cnt_1d",
                owner_table="ads_merchant_profile",
            ),
            SimpleNamespace(
                kind="metric",
                phrase="退款单量",
                canonical_key="direct_refund_cnt_1d",
                owner_table="ads_merchant_profile",
            ),
            SimpleNamespace(
                kind="metric",
                phrase="退款单量",
                canonical_key="refund_bill_cnt",
                owner_table="dwm_trade_refund_detail_di",
            ),
        ],
        metric_keywords=["订单总数", "退款单量"],
    )
    slots = _metric_recall_slots(keywords)
    assert [item["phrase"] for item in slots] == [
        "订单总数",
        "退款单量",
    ]
    recall = RecallBundle(
        items=[
            RecallItem(
                doc_id="semantic:经营画像:ads_merchant_profile:metric:refund_rate_by_pay_order",
                title="退款率",
                content="周期汇总退款率",
                source_type="SEMANTIC_METRIC",
                topic="经营画像",
                table="ads_merchant_profile",
                fusion_score=0.99,
                metadata={
                    "semanticRefId": "semantic:经营画像:ads_merchant_profile:metric:refund_rate_by_pay_order",
                    "semanticPath": "topics/经营画像/tables/ads_merchant_profile/metrics/refund_rate_by_pay_order.json",
                    "semanticKind": "METRIC",
                    "metricKey": "refund_rate_by_pay_order",
                    "aliases": ["退款率"],
                },
            ),
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:product_refund_order_share",
                title="商品退款订单占比",
                content="商品退款订单占比",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=0.98,
                metadata={
                    "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:product_refund_order_share",
                    "semanticPath": "topics/电商退货/tables/dwm_trade_refund_detail_di/metrics/product_refund_order_share.json",
                    "semanticKind": "METRIC",
                    "metricKey": "product_refund_order_share",
                    "aliases": ["商品退款订单占比"],
                },
            ),
            RecallItem(
                doc_id="semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d",
                title="订单总数",
                content="订单总数",
                source_type="SEMANTIC_METRIC",
                topic="经营画像",
                table="ads_merchant_profile",
                fusion_score=0.40,
                metadata={
                    "semanticRefId": "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d",
                    "semanticPath": "topics/经营画像/tables/ads_merchant_profile/metrics/order_cnt_1d.json",
                    "semanticKind": "METRIC",
                    "metricKey": "order_cnt_1d",
                    "aliases": ["订单总数"],
                },
            ),
            RecallItem(
                doc_id="semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_bill_cnt",
                title="退款单量",
                content="退款单量",
                source_type="SEMANTIC_METRIC",
                topic="电商退货",
                table="dwm_trade_refund_detail_di",
                fusion_score=0.30,
                metadata={
                    "semanticRefId": "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_bill_cnt",
                    "semanticPath": "topics/电商退货/tables/dwm_trade_refund_detail_di/metrics/refund_bill_cnt.json",
                    "semanticKind": "METRIC",
                    "metricKey": "refund_bill_cnt",
                    "aliases": ["退款单量"],
                },
            ),
            RecallItem(
                doc_id="semantic:经营画像:ads_merchant_profile:metric:direct_refund_cnt_1d",
                title="直接退款量",
                content="退款单量",
                source_type="SEMANTIC_METRIC",
                topic="经营画像",
                table="ads_merchant_profile",
                fusion_score=0.20,
                metadata={
                    "semanticRefId": "semantic:经营画像:ads_merchant_profile:metric:direct_refund_cnt_1d",
                    "semanticPath": "topics/经营画像/tables/ads_merchant_profile/metrics/direct_refund_cnt_1d.json",
                    "semanticKind": "METRIC",
                    "metricKey": "direct_refund_cnt_1d",
                    "aliases": ["退款单量"],
                },
            ),
        ]
    )

    thin = _thin_recall(recall, 4, metric_slots=slots)
    refs = [item["refId"] for item in thin]

    assert "semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d" in refs
    assert "semantic:经营画像:ads_merchant_profile:metric:direct_refund_cnt_1d" in refs
    assert "semantic:电商退货:dwm_trade_refund_detail_di:metric:refund_bill_cnt" in refs
    assert thin[0]["metricSlotPhrases"] == ["订单总数"]


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


def test_compat_batch_reuses_core_evidence_when_semantic_ref_is_passed_as_path() -> None:
    class RejectingCatalog(FakeSemanticCatalog):
        def read(self, *, path: str, max_chars: int, offset: int) -> dict[str, Any]:
            raise AssertionError("already-read evidence must be reused")

    session = GroundedDeepAgentSession(
        runtime=GroundedRuntimeSession(
            session_id="batch-evidence-reuse",
            question="查询工单明细",
            merchant_id="m-1",
            workspace_topics=["客服工单"],
        ),
        core_semantic_evidence=[
            {
                "refId": "semantic:客服工单:tickets:detail",
                "path": "knowledge/topics/客服工单/tables/tickets/detail.json",
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "contentSnippet": "{}",
                "contentHash": "hash",
                "contentComplete": True,
            }
        ],
    )
    evidence, newly_read = _read_exact_core_semantic_path(
        GroundedSemanticBackend(RejectingCatalog()),
        session,
        "semantic:客服工单:tickets:detail",
    )

    assert newly_read is False
    assert evidence["refId"] == "semantic:客服工单:tickets:detail"


def test_query_data_collapses_contract_execution_and_evidence() -> None:
    class QueryDataKernel(FakeKernel):
        def execute_active(
            self,
            session: GroundedRuntimeSession,
            **kwargs: Any,
        ) -> AgentRunResult:
            del kwargs
            result = AgentRunResult(
                merged_query_bundle=QueryBundle(
                    rows=[{"ticket_count": 3}],
                    tables=["tickets"],
                )
            )
            session.run_result = result
            return result

        @staticmethod
        def verify_active(
            session: GroundedRuntimeSession,
        ) -> VerifiedEvidence:
            verified = VerifiedEvidence(passed=True)
            session.verified_evidence = verified
            assert session.run_result is not None
            contract = GroundedQueryContract(
                question=session.question,
                topics=["客服工单"],
                status="READY",
                query_shape="SCALAR",
            )
            session.verified_query_ledger.append(
                GroundedVerifiedQueryArtifact(
                    artifact_id="query_artifact_query_data",
                    generation=1,
                    contract_fingerprint=(grounded_query_contract_fingerprint(contract)),
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
    kernel = QueryDataKernel()
    outer = runtime(factory, kernel)
    kernel_session = kernel.new_session("工单量是多少", "m-1")
    detail_ref = "semantic:客服工单:tickets:detail"
    deep_session = GroundedDeepAgentSession(
        runtime=kernel_session,
        core_semantic_evidence=[
            {
                "refId": detail_ref,
                "kind": "TABLE_DETAIL",
                "topic": "客服工单",
                "table": "tickets",
                "contentSnippet": "{}",
                "contentHash": "hash",
            }
        ],
    )
    context = GroundedDeepAgentRunContext(
        thread_id="thread-query-data",
        run_id="run-query-data",
        session=deep_session,
    )
    tools = {item.name: item for item in outer.tools}
    declare_single_metric_goal(tools, context)

    result = json.loads(
        tools["query_data"].func(
            request={
                "queryId": "query.ticket_count",
                "goalIds": ["metric.primary"],
                "readRefIds": [detail_ref],
                "bindingHints": {"tableRefs": [detail_ref]},
                "reason": "answer the scalar metric goal",
            },
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "VERIFIED"
    assert result["queryId"] == "query.ticket_count"
    assert result["artifactIds"] == ["query_artifact_query_data"]
    assert result["coveredGoalIds"] == ["metric.primary"]
    assert result["rowCount"] == 1
    assert kernel.propose_calls == 1


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

    assert result["status"] == "SKILL_RUN_CONTRACT_REQUIRED"
    assert kernel_session.run_result is None
    assert context.session.skill_runs == []


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
            contract=_verified_skill_contract(session),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "VERIFIED_SKILL_ARTIFACT_PUBLISHED"
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
    assert isolated.job.user_payload["promptTrace"]["promptId"] == (
        "grounded.skill_subagent.system"
    )
    assert "generic isolated subagent" in isolated.job.system_prompt
    assert [item[0] for item in events] == ["skill.progress"] * len(events)
    assert any(item[2]["stage"] == "subagent_step" for item in events)
    assert kernel_session.answer == ""
    assert len(session.verified_skill_ledger) == 1
    assert session.verified_skill_ledger[0].integrity_valid()
    assert set(session.goal_coverage_result["artifactIds"]) == set(kernel_session.answer_artifact_ids)
    assert result["verifiedSkillArtifactId"] not in str(session.goal_coverage_result)
    assert kernel_session.run_result.skill_lifecycle_records[0].matched_by == ("core_llm_skill_header")

    base_contract = _verified_skill_contract(session)
    stale_snapshot = json.loads(
        tools["run_skill"].func(
            contract=base_contract.model_copy(
                update={
                    "sub_goal_id": "skill.stale.snapshot",
                    "input_snapshot_generation": 2,
                }
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    outside_snapshot = json.loads(
        tools["run_skill"].func(
            contract=base_contract.model_copy(
                update={
                    "sub_goal_id": "skill.outside.snapshot",
                    "input_artifact_ids": ["query-artifact-not-frozen"],
                }
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert stale_snapshot["status"] == "SKILL_INPUT_SNAPSHOT_STALE"
    assert outside_snapshot["status"] == "SKILL_INPUT_SNAPSHOT_SCOPE_MISMATCH"

    session.skill_execution_in_progress = True
    concurrent = json.loads(
        tools["run_skill"].func(
            contract=_verified_skill_contract(
                session,
                skill_name="ratio-analysis",
                sub_goal_id="skill.concurrent",
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    session.skill_execution_in_progress = False
    assert concurrent["status"] == "SKILL_EXECUTION_IN_PROGRESS"

    # Simulate a Root query/local replan followed by a refreshed immutable
    # Skill-input snapshot. The next Skill generation must bind snapshot 2.
    session.skill_input_snapshot_generation = 2
    generation_two = json.loads(
        tools["run_skill"].func(
            contract=_verified_skill_contract(
                session,
                generation=2,
                snapshot_generation=2,
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    skill_b = json.loads(
        tools["run_skill"].func(
            contract=_verified_skill_contract(
                session,
                skill_name="merchant-daily-briefing",
                sub_goal_id="skill.briefing",
                snapshot_generation=2,
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    skill_c = json.loads(
        tools["run_skill"].func(
            contract=_verified_skill_contract(
                session,
                skill_name="gmv-drop-diagnosis",
                sub_goal_id="skill.gmv",
                snapshot_generation=2,
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    skill_limit = json.loads(
        tools["run_skill"].func(
            contract=_verified_skill_contract(
                session,
                skill_name="refund-rate-diagnosis",
                sub_goal_id="skill.refund",
                snapshot_generation=2,
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert generation_two["status"] == "VERIFIED_SKILL_ARTIFACT_PUBLISHED"
    assert skill_b["status"] == "VERIFIED_SKILL_ARTIFACT_PUBLISHED"
    assert skill_c["status"] == "VERIFIED_SKILL_ARTIFACT_PUBLISHED"
    assert skill_limit["status"] == "VERIFIED_SKILL_ARTIFACT_LIMIT_REACHED"
    assert len(session.verified_skill_ledger) == 4
    assert all(item.integrity_valid() for item in session.verified_skill_ledger)
    assert kernel_session.answer == ""
    assert all(item.artifact_id not in str(session.goal_coverage_result) for item in session.verified_skill_ledger)


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
            contract=_verified_skill_contract(deep_session),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "VERIFIED_SKILL_ARTIFACT_PUBLISHED"
    assert result["repairAttempted"] is True
    assert result["queryMutationAllowed"] is False
    assert len(repairing.jobs) == 2
    assert repairing.jobs[1].user_payload["repairAttempt"] == 1
    continued = json.loads(
        tools["retrieve_knowledge"].func(
            query="再查更多指标",
            reason="修复 Skill",
            runtime=SimpleNamespace(context=context),
        )
    )
    assert continued["status"] != "SKILL_EXECUTION_IN_PROGRESS"


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
            contract=_verified_skill_contract(deep_session),
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "SKILL_VERIFICATION_FAILED"
    assert result["queryMutationAllowed"] is False
    assert len(invalid.jobs) == 2
    assert kernel_session.answer == ""
