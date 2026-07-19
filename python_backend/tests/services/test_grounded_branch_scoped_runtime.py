from __future__ import annotations

import hashlib
import json
from threading import Event, RLock
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    AgentRunResult,
    MerchantInfo,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedCoreToolBoundaryMiddleware,
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
    GroundedSemanticBackend,
    _semantic_payload_summary,
)
from merchant_ai.services.grounded_goal_contract import (
    AnalysisQuestionGoal,
    DependencyQuestionGoal,
    DetailQuestionGoal,
    DimensionQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
    TimeWindowQuestionGoal,
    original_question_goal_contract_fingerprint,
)
from merchant_ai.services.grounded_execution_graph import (
    discovery_evidence_snapshot_fingerprint,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedContractGap,
    GroundedQueryContract,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeAttempt,
    GroundedRuntimeSession,
    GroundedVerifiedQueryArtifact,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class _Catalog:
    def __init__(self) -> None:
        self.documents = {
            "topics/电商交易/tables/orders/detail.json": {
                "refId": "semantic:电商交易:orders:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商交易",
                "table": "orders",
                "content": json.dumps(
                    {
                        "topic": "电商交易",
                        "tableName": "orders",
                        "semanticNavigation": {
                            "source": "published_asset",
                            "questionIndependent": True,
                            "bindingEvidence": False,
                            "publishedCounts": {"metrics": 1, "columns": 0},
                            "advertisedCounts": {"metrics": 1, "columns": 0},
                            "metricLeaves": [
                                {
                                    "key": "order_count",
                                    "aliases": ["订单量", "总订单量"],
                                    "refId": "semantic:电商交易:orders:metric:order_count",
                                    "path": "topics/电商交易/tables/orders/metrics/order_count.json",
                                }
                            ],
                            "columnLeaves": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            "topics/电商交易/tables/orders/metrics/order_count.json": {
                "refId": "semantic:电商交易:orders:metric:order_count",
                "kind": "METRIC",
                "topic": "电商交易",
                "table": "orders",
                "content": '{"tableName":"orders","metric":{"metricKey":"order_count","formula":"COUNT(DISTINCT order_id)"}}',
            },
            "topics/电商退货/tables/refunds/detail.json": {
                "refId": "semantic:电商退货:refunds:detail",
                "kind": "TABLE_DETAIL",
                "topic": "电商退货",
                "table": "refunds",
                "content": '{"topic":"电商退货","tableName":"refunds"}',
            },
            "topics/电商退货/tables/refunds/metrics/refund_amount.json": {
                "refId": "semantic:电商退货:refunds:metric:refund_amount",
                "kind": "METRIC",
                "topic": "电商退货",
                "table": "refunds",
                "content": '{"tableName":"refunds","metric":{"metricKey":"refund_amount","formula":"SUM(refund_amount)"}}',
            },
        }

    def read(self, **kwargs: Any) -> dict[str, Any]:
        path = str(kwargs.get("path") or "")
        ref_id = str(kwargs.get("ref_id") or "")
        if ref_id:
            match = next(
                ((item_path, item) for item_path, item in self.documents.items() if item["refId"] == ref_id),
                None,
            )
            if match is None:
                return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}
            path, document = match
        else:
            document = self.documents.get(path)
        if document is None:
            return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}
        return {
            "success": True,
            "path": path,
            **dict(document),
        }


class _BranchKernel:
    def __init__(self, *, require_parallel_overlap: bool = True) -> None:
        self.lock = RLock()
        self.active_prepares = 0
        self.max_active_prepares = 0
        self.two_prepares = Event()
        self.seen_topics: dict[str, list[str]] = {}
        self.require_parallel_overlap = require_parallel_overlap

    @staticmethod
    def new_session(
        question: str,
        merchant_id: str,
        **kwargs: Any,
    ) -> GroundedRuntimeSession:
        return GroundedRuntimeSession(
            session_id=str(kwargs.get("session_id") or "root"),
            question=question,
            merchant_id=merchant_id,
            merchant=kwargs.get("merchant") or MerchantInfo(merchant_id=merchant_id),
            workspace_topics=["电商交易", "电商退货"],
        )

    def fork_query_branch(
        self,
        session: GroundedRuntimeSession,
        branch_id: str,
        **kwargs: Any,
    ) -> GroundedRuntimeSession:
        return GroundedRuntimeSession(
            session_id="%s:%s" % (session.session_id, branch_id),
            question=str(kwargs.get("objective") or session.question),
            merchant_id=session.merchant_id,
            merchant=session.merchant,
            workspace_topics=list(kwargs.get("workspace_topics") or []),
        )

    def propose_contract(
        self,
        session: GroundedRuntimeSession,
        evidence: list[dict[str, Any]],
        hints: Any,
        **kwargs: Any,
    ) -> GroundedRuntimeAttempt:
        del hints
        with self.lock:
            self.active_prepares += 1
            self.max_active_prepares = max(
                self.max_active_prepares,
                self.active_prepares,
            )
            if self.active_prepares >= 2:
                self.two_prepares.set()
        try:
            if self.require_parallel_overlap:
                assert self.two_prepares.wait(timeout=2)
            topics = list(kwargs.get("topics") or [])
            self.seen_topics[session.session_id] = topics
            assert evidence
            assert {item["topic"] for item in evidence} == set(topics)
            ready = any(item["kind"] == "METRIC" for item in evidence)
            contract = GroundedQueryContract(
                question=session.question,
                topics=topics,
                status="READY" if ready else "UNRESOLVED",
                query_shape="SCALAR",
                evidence_refs=[item["refId"] for item in evidence],
                unresolved_gaps=(
                    []
                    if ready
                    else [
                        GroundedContractGap(
                            code="METRIC_EVIDENCE_REQUIRED",
                            message="read the advertised metric leaf",
                            evidence_kind="METRIC",
                            topic=topics[0],
                        )
                    ]
                ),
            )
            attempt = GroundedRuntimeAttempt(
                attempt_id="attempt:%s" % session.session_id,
                contract=contract,
            )
            session.attempts.append(attempt)
            return attempt
        finally:
            with self.lock:
                self.active_prepares -= 1

    @staticmethod
    def activate_contract(
        session: GroundedRuntimeSession,
        attempt_id: str,
    ) -> GroundedRuntimeAttempt:
        del attempt_id
        attempt = session.attempts[-1]
        attempt.activated = True
        attempt.activation_status = "ACTIVATED"
        attempt.compile_status = "VALID"
        attempt.execution_mode = "DETERMINISTIC_METRIC"
        attempt.active_generation = 1
        session.active_generation = 1
        session.active_attempt_id = attempt.attempt_id
        session.active_contract = attempt.contract
        session.active_execution_mode = "DETERMINISTIC_METRIC"
        session.phase = "ACTIVE_COMPILED"
        return attempt

    @staticmethod
    def execute_active(
        session: GroundedRuntimeSession,
        **kwargs: Any,
    ) -> AgentRunResult:
        del kwargs
        result = AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[{"branch": session.session_id, "value": 1}],
                tables=[session.session_id],
            )
        )
        session.run_result = result
        return result

    @staticmethod
    def verify_active(
        session: GroundedRuntimeSession,
    ) -> VerifiedEvidence:
        assert session.active_contract is not None
        assert session.run_result is not None
        verified = VerifiedEvidence(
            passed=True,
            covered_evidence=[session.session_id],
        )
        artifact = GroundedVerifiedQueryArtifact(
            artifact_id="artifact:%s" % session.session_id,
            generation=session.active_generation,
            attempt_id=session.active_attempt_id,
            contract_fingerprint=grounded_query_contract_fingerprint(session.active_contract),
            sql_fingerprint=hashlib.sha256(session.session_id.encode("utf-8")).hexdigest(),
            contract=session.active_contract,
            plan=QueryPlan(),
            run_result=session.run_result,
            verified_evidence=verified,
            output_columns=["branch", "value"],
        )
        session.verified_evidence = verified
        session.verified_query_ledger.append(artifact)
        return verified

    @staticmethod
    def latest_verified_query_artifact(
        session: GroundedRuntimeSession,
    ) -> GroundedVerifiedQueryArtifact | None:
        return session.verified_query_ledger[-1] if session.verified_query_ledger else None

    @staticmethod
    def adopt_verified_branches(
        session: GroundedRuntimeSession,
        branches: list[GroundedRuntimeSession],
    ) -> list[GroundedVerifiedQueryArtifact]:
        adopted = [
            artifact.model_copy(deep=True)
            for branch in branches
            for artifact in branch.verified_query_ledger
            if artifact.verified_evidence.passed
        ]
        session.verified_query_ledger.extend(adopted)
        return adopted


class _Factory:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace()


def _runtime(
    *,
    require_parallel_overlap: bool = True,
) -> tuple[GroundedDeepAgentRuntime, _BranchKernel, _Catalog]:
    kernel = _BranchKernel(require_parallel_overlap=require_parallel_overlap)
    catalog = _Catalog()
    runtime = GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=catalog,
        settings=SimpleNamespace(
            grounded_finalization_reserve_seconds=0,
            grounded_branch_max_duration_seconds=30,
        ),
        agent_factory=_Factory(),
        backend=object(),
    )
    return runtime, kernel, catalog


def _context(
    kernel: _BranchKernel,
    question: str,
) -> GroundedDeepAgentRunContext:
    state = kernel.new_session(question, "merchant-1", session_id="parent")
    state.workspace_topics = ["电商交易", "电商退货"]
    return GroundedDeepAgentRunContext(
        thread_id="branch-thread",
        run_id="branch-run",
        session=GroundedDeepAgentSession(runtime=state),
    )


def test_declared_branches_prepare_semantics_and_contracts_in_parallel() -> None:
    runtime, kernel, _ = _runtime()
    context = _context(kernel, "订单量和退款金额分别是多少")
    tools = {item.name: item for item in runtime.tools}
    goal_result = json.loads(
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.orders",
                        label="订单量",
                    ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="退款金额",
                    ),
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    )
    assert goal_result["nextAction"] == ("DISCOVER_SEMANTIC_EVIDENCE")
    declared = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "orders",
                    "objective": "订单量",
                    "goalIds": ["metric.orders"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "refunds",
                    "objective": "退款金额",
                    "goalIds": ["metric.refunds"],
                    "topicScope": ["电商退货"],
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["status"] == "FROZEN"

    prepared = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[
                {
                    "queryId": "orders",
                    "semanticPaths": [
                        "topics/电商交易/tables/orders/detail.json",
                        "topics/电商交易/tables/orders/metrics/order_count.json",
                    ],
                    "bindingHints": {
                        "tableRefs": ["semantic:电商交易:orders:detail"],
                        "metricRefs": ["semantic:电商交易:orders:metric:order_count"],
                    },
                },
                {
                    "queryId": "refunds",
                    "semanticPaths": [
                        "topics/电商退货/tables/refunds/detail.json",
                        "topics/电商退货/tables/refunds/metrics/refund_amount.json",
                    ],
                    "bindingHints": {
                        "tableRefs": ["semantic:电商退货:refunds:detail"],
                        "metricRefs": ["semantic:电商退货:refunds:metric:refund_amount"],
                    },
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert prepared["status"] == "PREPARED"
    assert prepared["compatMode"] == "BRANCH_SCOPED_V2"
    assert prepared["preparedInParallel"] is True
    assert kernel.max_active_prepares == 2
    assert context.session.core_semantic_evidence == []
    orders = context.session.query_branch_contexts["orders"]
    refunds = context.session.query_branch_contexts["refunds"]
    assert orders.effective_topics() == ["电商交易"]
    assert refunds.effective_topics() == ["电商退货"]
    assert all(item["topic"] == "电商交易" for item in orders.semantic_ledger.evidence())
    assert all(item["topic"] == "电商退货" for item in refunds.semantic_ledger.evidence())
    assert set(context.session.parallel_branches) == {"orders", "refunds"}

    executed = json.loads(
        tools["execute_grounded_query_batch"].func(
            queries=[{"queryId": "orders"}, {"queryId": "refunds"}],
            reason="independent metrics",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert executed["status"] == "VERIFIED"
    assert executed["executedInParallel"] is True
    assert len(context.session.runtime.verified_query_ledger) == 2
    assert orders.status == "VERIFIED"
    assert refunds.status == "VERIFIED"
    assert orders.budget.report()["usage"]["dorisQueries"] == 1
    assert refunds.budget.report()["usage"]["dorisQueries"] == 1


def test_entity_chain_downstream_waits_before_contract_preparation() -> None:
    runtime, kernel, _ = _runtime()
    context = _context(kernel, "先查商品集合，再用该集合查退款")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.top_products",
                    label="商品排行",
                ),
                    MetricQuestionGoal(
                        goal_id="metric.refunds",
                        label="商品集合退款",
                    ),
                    DependencyQuestionGoal(
                        goal_id="dependency.entity_set",
                        label="商品集合传递",
                        dependency_type="entity_chain",
                        artifact_kind="VERIFIED_ENTITY_SET",
                        upstream_goal_ids=["metric.top_products"],
                        downstream_goal_ids=["metric.refunds"],
                    ),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    declared = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "top-products",
                    "goalIds": ["metric.top_products"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "product-refunds",
                    "goalIds": ["metric.refunds"],
                    "topicScope": ["电商退货"],
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )
    assert declared["waitingForVerifiedEntitySetQueryIds"] == ["product-refunds"]

    result = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[{"queryId": "product-refunds"}],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "BLOCKED"
    assert result["queries"][0]["status"] == ("WAITING_VERIFIED_ENTITY_SET")
    assert context.session.query_branch_contexts["product-refunds"].runtime is None
    assert kernel.max_active_prepares == 0


def test_ranking_metric_and_dimension_goals_must_share_its_branch() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "销售额最高的前3个商品")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.sales",
                    label="销售额",
                ),
                DimensionQuestionGoal(
                    goal_id="dimension.product",
                    label="商品",
                ),
                RankingQuestionGoal(
                    goal_id="ranking.top_products",
                    label="销售额最高前3商品",
                    metric_goal_ids=["metric.sales"],
                    dimension_goal_ids=["dimension.product"],
                    limit=3,
                    population_scope="ALL_MATCHING_ROWS",
                ),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )

    rejected = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "sales-inputs",
                    "goalIds": ["metric.sales", "dimension.product"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "ranking",
                    "goalIds": ["ranking.top_products"],
                    "topicScope": ["电商交易"],
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert rejected["status"] == "REJECTED"
    assert {item["code"] for item in rejected["issues"]} == {"QUERY_BRANCH_STRUCTURAL_GOALS_NOT_COLOCATED"}

    accepted = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "ranking",
                    "goalIds": [
                        "metric.sales",
                        "dimension.product",
                        "ranking.top_products",
                    ],
                    "topicScope": ["电商交易"],
                }
            ],
            runtime=SimpleNamespace(context=context),
        )
    )
    assert accepted["status"] == "FROZEN"


def test_same_branch_prerequisites_are_local_not_entity_chain_dependencies() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(
        kernel,
        "最近10天订单明细，再找退款金额最高的前5单",
    )
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                TimeWindowQuestionGoal(
                    goal_id="time.recent10d",
                    label="最近10天",
                    time_expression="最近10天",
                    applies_to_goal_ids=[
                        "detail.orders",
                        "ranking.refunds",
                    ],
                ),
                DetailQuestionGoal(
                    goal_id="detail.orders",
                    label="订单明细",
                    depends_on_goal_ids=["time.recent10d"],
                ),
                MetricQuestionGoal(
                    goal_id="metric.refund_amount",
                    label="退款金额",
                ),
                DimensionQuestionGoal(
                    goal_id="dimension.order_id",
                    label="订单标识",
                ),
                RankingQuestionGoal(
                    goal_id="ranking.refunds",
                    label="退款最高前5单",
                    metric_goal_ids=["metric.refund_amount"],
                    dimension_goal_ids=["dimension.order_id"],
                    direction="DESC",
                    limit=5,
                    population_scope="ALL_MATCHING_ROWS",
                    depends_on_goal_ids=[
                        "time.recent10d",
                        "metric.refund_amount",
                        "dimension.order_id",
                    ],
                ),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )

    declared = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "orders",
                    "goalIds": ["time.recent10d", "detail.orders"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "refund-ranking",
                    "goalIds": [
                        "time.recent10d",
                        "metric.refund_amount",
                        "dimension.order_id",
                        "ranking.refunds",
                    ],
                    "topicScope": ["电商退货"],
                },
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert declared["status"] == "FROZEN"
    assert set(declared["readyQueryIds"]) == {
        "orders",
        "refund-ranking",
    }
    assert declared["waitingForVerifiedEntitySetQueryIds"] == []


def test_contract_scope_population_keeps_both_execution_nodes_ready() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(
        kernel,
        "我想看最近7天的订单明细，然后告诉我这里面退款最多的三单",
    )
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                TimeWindowQuestionGoal(
                    goal_id="time.recent7d",
                    label="最近7天",
                    time_expression="最近7天",
                    applies_to_goal_ids=[
                        "detail.orders",
                        "ranking.refunds",
                    ],
                ),
                DetailQuestionGoal(
                    goal_id="detail.orders",
                    label="订单明细",
                    depends_on_goal_ids=["time.recent7d"],
                ),
                MetricQuestionGoal(
                    goal_id="metric.refund_amount",
                    label="退款金额",
                ),
                DimensionQuestionGoal(
                    goal_id="dimension.order_id",
                    label="订单标识",
                ),
                RankingQuestionGoal(
                    goal_id="ranking.refunds",
                    label="这里面退款最多的三单",
                    metric_goal_ids=["metric.refund_amount"],
                    dimension_goal_ids=["dimension.order_id"],
                    limit=3,
                    population_scope="SAME_AS_GOAL",
                    population_goal_ids=["detail.orders"],
                ),
                DependencyQuestionGoal(
                    goal_id="dependency.order_population",
                    label="订单范围作为退款排名总体",
                    dependency_type="CONTRACT_SCOPE",
                    upstream_goal_ids=["detail.orders"],
                    downstream_goal_ids=["ranking.refunds"],
                    artifact_kind="",
                ),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    context.session.core_semantic_evidence = [
        {
            "refId": "semantic:trade:orders:detail",
            "path": "topics/trade/orders/detail.json",
            "topic": "电商交易",
            "contentHash": "orders-hash",
            "contentComplete": True,
        },
        {
            "refId": "semantic:refund:refunds:detail",
            "path": "topics/refund/refunds/detail.json",
            "topic": "电商退货",
            "contentHash": "refunds-hash",
            "contentComplete": True,
        },
    ]
    goal_contract = context.session.question_goal_contract
    assert goal_contract is not None

    frozen = json.loads(
        tools["propose_grounded_execution_graph"].func(
            proposal={
                "baseVersion": 0,
                "goalContractFingerprint": (
                    original_question_goal_contract_fingerprint(
                        goal_contract
                    )
                ),
                "discoverySnapshotFingerprint": (
                    discovery_evidence_snapshot_fingerprint(
                        context.session.core_semantic_evidence
                    )
                ),
                "nodes": [
                    {
                        "clientKey": "orders",
                        "goalIds": [
                            "time.recent7d",
                            "detail.orders",
                        ],
                        "topicScope": ["电商交易"],
                        "evidenceRefIds": [
                            "semantic:trade:orders:detail"
                        ],
                    },
                    {
                        "clientKey": "refund_ranking",
                        "goalIds": [
                            "time.recent7d",
                            "metric.refund_amount",
                            "dimension.order_id",
                            "ranking.refunds",
                        ],
                        "topicScope": ["电商交易", "电商退货"],
                        "evidenceRefIds": [
                            "semantic:trade:orders:detail",
                            "semantic:refund:refunds:detail",
                        ],
                    },
                ],
                "edges": [
                    {
                        "sourceClientKey": "orders",
                        "targetClientKey": "refund_ranking",
                        "dependencyMode": "CONTRACT_SCOPE",
                    }
                ],
            },
            runtime=SimpleNamespace(context=context),
        )
    )

    assert frozen["status"] == "FROZEN"
    orders_id = frozen["clientNodeIds"]["orders"]
    refunds_id = frozen["clientNodeIds"]["refund_ranking"]
    assert set(frozen["readyQueryIds"]) == {orders_id, refunds_id}
    assert frozen["waitingForVerifiedArtifactQueryIds"] == []
    refunds = context.session.query_branch_contexts[refunds_id]
    assert refunds.contract_scope_query_ids == [orders_id]
    assert refunds.dependency_query_ids == []
    assert refunds.runtime is not None


def test_detail_input_goals_are_colocated_but_analysis_inputs_may_branch() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    detail_context = _context(kernel, "订单明细")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=detail_context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单",
                ),
                DetailQuestionGoal(
                    goal_id="detail.orders",
                    label="订单明细",
                    input_goal_ids=["metric.orders"],
                ),
            ],
        ),
        runtime=SimpleNamespace(context=detail_context),
    )
    rejected = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "orders",
                    "goalIds": ["metric.orders"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "order-detail",
                    "goalIds": ["detail.orders"],
                    "topicScope": ["电商交易"],
                },
            ],
            runtime=SimpleNamespace(context=detail_context),
        )
    )
    assert rejected["issues"][0]["code"] == ("QUERY_BRANCH_STRUCTURAL_GOALS_NOT_COLOCATED")

    analysis_context = _context(kernel, "订单量和退款金额是否相关")
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=analysis_context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单量",
                ),
                MetricQuestionGoal(
                    goal_id="metric.refunds",
                    label="退款金额",
                ),
                AnalysisQuestionGoal(
                    goal_id="analysis.correlation",
                    label="相关性",
                    analysis_type="correlation",
                    input_goal_ids=["metric.orders", "metric.refunds"],
                ),
            ],
        ),
        runtime=SimpleNamespace(context=analysis_context),
    )
    accepted = json.loads(
        tools["declare_grounded_query_branches"].func(
            branches=[
                {
                    "queryId": "orders",
                    "goalIds": ["metric.orders"],
                    "topicScope": ["电商交易"],
                },
                {
                    "queryId": "refunds",
                    "goalIds": ["metric.refunds"],
                    "topicScope": ["电商退货"],
                },
            ],
            runtime=SimpleNamespace(context=analysis_context),
        )
    )
    assert accepted["status"] == "FROZEN"
    assert accepted["readyQueryIds"] == ["orders", "refunds"]


def test_l0_detail_path_yields_bounded_l1_navigation_then_exact_leaf_ready() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "订单量是多少")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(
                    goal_id="metric.orders",
                    label="订单量",
                )
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    tools["declare_grounded_query_branches"].func(
        branches=[
            {
                "queryId": "orders",
                "goalIds": ["metric.orders"],
                "topicScope": ["电商交易"],
            }
        ],
        runtime=SimpleNamespace(context=context),
    )

    first = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[
                {
                    "queryId": "orders",
                    "semanticPaths": ["topics/电商交易/tables/orders/detail.json"],
                    "bindingHints": {"tableRefs": ["semantic:电商交易:orders:detail"]},
                }
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert first["status"] == "BLOCKED"
    receipt = first["queries"][0]["semanticReceipts"][0]
    navigation = receipt["summary"]["semanticNavigation"]
    assert navigation["bindingEvidence"] is False
    assert navigation["advertisedCounts"] == {"metrics": 1, "columns": 0}
    assert navigation["metricLeaves"] == [
        {
            "key": "order_count",
            "aliases": ["订单量", "总订单量"],
            "refId": "semantic:电商交易:orders:metric:order_count",
            "path": "topics/电商交易/tables/orders/metrics/order_count.json",
        }
    ]

    second = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[
                {
                    "queryId": "orders",
                    "semanticPaths": [navigation["metricLeaves"][0]["path"]],
                    "bindingHints": {
                        "tableRefs": ["semantic:电商交易:orders:detail"],
                        "metricRefs": [navigation["metricLeaves"][0]["refId"]],
                    },
                }
            ],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert second["status"] == "PREPARED"
    branch = context.session.query_branch_contexts["orders"]
    assert branch.semantic_ledger.refs() == [
        "semantic:电商交易:orders:detail",
        "semantic:电商交易:orders:metric:order_count",
    ]
    assert branch.budget.report()["usage"]["semanticReads"] == 2


def test_l1_receipt_keeps_every_publisher_bounded_column_coordinate() -> None:
    column_leaves = [
        {
            "key": "column_%s" % index,
            "aliases": ["字段%s" % index],
            "refId": "semantic:topic:table:column:%s" % index,
            "path": "topics/topic/tables/table/columns/%s.json" % index,
        }
        for index in range(26)
    ]
    summary = _semantic_payload_summary(
        "TABLE_DETAIL",
        json.dumps(
            {
                "tableName": "table",
                "semanticNavigation": {
                    "source": "published_asset",
                    "questionIndependent": True,
                    "bindingEvidence": False,
                    "publishedCounts": {"metrics": 0, "columns": 26},
                    "advertisedCounts": {"metrics": 0, "columns": 26},
                    "metricLeaves": [],
                    "columnLeaves": column_leaves,
                },
            },
            ensure_ascii=False,
        ),
    )

    navigation = summary["semanticNavigation"]
    assert navigation["advertisedCounts"]["columns"] == 26
    assert len(navigation["columnLeaves"]) == 26
    assert navigation["columnLeaves"][-1]["key"] == "column_25"


def test_multi_branch_boundary_rejects_global_filesystem_retrieval() -> None:
    runtime, kernel, catalog = _runtime()
    context = _context(kernel, "订单量和退款金额")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(goal_id="metric.orders", label="订单量"),
                MetricQuestionGoal(goal_id="metric.refunds", label="退款金额"),
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    tools["declare_grounded_query_branches"].func(
        branches=[
            {
                "queryId": "orders",
                "goalIds": ["metric.orders"],
                "topicScope": ["电商交易"],
            },
            {
                "queryId": "refunds",
                "goalIds": ["metric.refunds"],
                "topicScope": ["电商退货"],
            },
        ],
        runtime=SimpleNamespace(context=context),
    )
    boundary = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(catalog))
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={
            "id": "read-1",
            "name": "read_file",
            "args": {"file_path": "/knowledge/topics/电商交易/tables/orders/detail.json"},
        },
    )

    result = boundary.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(AssertionError("global handler must not run")),
    )

    payload = json.loads(str(result.content))
    assert result.status == "error"
    assert payload["code"] == "EXECUTION_GRAPH_DISCOVERY_FROZEN"


def test_query_goal_allows_discovery_before_execution_graph_freeze() -> None:
    runtime, kernel, catalog = _runtime()
    context = _context(kernel, "订单量")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[MetricQuestionGoal(goal_id="metric.orders", label="订单量")],
        ),
        runtime=SimpleNamespace(context=context),
    )
    boundary = GroundedCoreToolBoundaryMiddleware(GroundedSemanticBackend(catalog))
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={
            "id": "read-before-branch",
            "name": "read_file",
            "args": {"file_path": "/knowledge/topics/电商交易/tables/orders/detail.json"},
        },
    )

    result = boundary.wrap_tool_call(
        request,
        lambda _: SimpleNamespace(
            content="table detail",
            status="success",
            name="read_file",
            tool_call_id="read-before-graph",
        ),
    )

    assert result.status != "error"
    assert context.session.core_semantic_evidence
    goal_contract = context.session.question_goal_contract
    assert goal_contract is not None
    evidence_ref = context.session.core_semantic_evidence[0]["refId"]

    frozen = json.loads(
        tools["propose_grounded_execution_graph"].func(
            proposal={
                "baseVersion": 0,
                "goalContractFingerprint": (
                    original_question_goal_contract_fingerprint(
                        goal_contract
                    )
                ),
                "discoverySnapshotFingerprint": (
                    discovery_evidence_snapshot_fingerprint(
                        context.session.core_semantic_evidence
                    )
                ),
                "nodes": [
                    {
                        "clientKey": "orders",
                        "goalIds": ["metric.orders"],
                        "topicScope": ["电商交易"],
                        "evidenceRefIds": [evidence_ref],
                    }
                ],
            },
            runtime=SimpleNamespace(context=context),
        )
    )
    assert frozen["status"] == "FROZEN"
    assert frozen["receipt"]["version"] == 1
    assert frozen["receipt"]["fingerprint"]
    query_id = frozen["clientNodeIds"]["orders"]
    inherited = context.session.query_branch_contexts[
        query_id
    ].semantic_ledger.refs()
    assert inherited == ["semantic:电商交易:orders:detail"]


def _freeze_reopenable_execution_graph(
    runtime: GroundedDeepAgentRuntime,
    context: GroundedDeepAgentRunContext,
    *,
    base_version: int = 0,
    evidence_ref_id: str = "semantic:discovery:metric",
) -> dict[str, Any]:
    tools = {item.name: item for item in runtime.tools}
    if context.session.question_goal_contract is None:
        tools["declare_original_question_goals"].func(
            contract=OriginalQuestionGoalContract(
                question=context.session.runtime.question,
                goals=[
                    MetricQuestionGoal(
                        goal_id="metric.primary",
                        label="primary metric",
                    )
                ],
            ),
            runtime=SimpleNamespace(context=context),
        )
    context.session.core_semantic_evidence.append(
        {
            "refId": evidence_ref_id,
            "path": "topics/电商交易/tables/orders/metrics/order_count.json",
            "topic": "电商交易",
            "contentHash": "hash:%s" % evidence_ref_id,
            "contentComplete": True,
        }
    )
    goal_contract = context.session.question_goal_contract
    assert goal_contract is not None
    return json.loads(
        tools["propose_grounded_execution_graph"].func(
            proposal={
                "baseVersion": base_version,
                "goalContractFingerprint": (
                    original_question_goal_contract_fingerprint(
                        goal_contract
                    )
                ),
                "discoverySnapshotFingerprint": (
                    discovery_evidence_snapshot_fingerprint(
                        context.session.core_semantic_evidence
                    )
                ),
                "nodes": [
                    {
                        "clientKey": "primary_query",
                        "goalIds": ["metric.primary"],
                        "topicScope": ["电商交易"],
                        "evidenceRefIds": [evidence_ref_id],
                    }
                ],
            },
            runtime=SimpleNamespace(context=context),
        )
    )


def test_gapped_unexecuted_graph_reopens_discovery_and_refreezes_with_cas() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "primary metric")
    tools = {item.name: item for item in runtime.tools}
    first = _freeze_reopenable_execution_graph(runtime, context)
    assert first["status"] == "FROZEN"
    assert first["receipt"]["version"] == 1
    query_id = first["clientNodeIds"]["primary_query"]
    branch = context.session.query_branch_contexts[query_id]
    branch.status = "CONTRACT_GAPPED"
    branch.last_gaps = [
        {
            "code": "BINDING_EVIDENCE_REQUIRED",
            "requiredCapability": {
                "kind": "SEMANTIC_EVIDENCE",
                "scope": "ACTIVE_GOAL",
            },
        }
    ]

    reopened = json.loads(
        tools["reopen_grounded_execution_graph_discovery"].func(
            graph_id=first["receipt"]["graphId"],
            version=first["receipt"]["version"],
            reason="A structured binding gap requires new evidence",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert reopened["status"] == "DISCOVERY_REOPENED"
    assert reopened["baseVersion"] == 1
    assert context.session.execution_graph_generation == 1
    assert context.session.execution_graph_receipt is None
    assert context.session.query_branch_contexts == {}
    assert context.session.execution_graph_history[-1]["status"] == (
        "GAPPED_REOPENED"
    )

    second = _freeze_reopenable_execution_graph(
        runtime,
        context,
        base_version=reopened["baseVersion"],
        evidence_ref_id="semantic:discovery:metric:revised",
    )

    assert second["status"] == "FROZEN"
    assert second["receipt"]["version"] == 2
    assert context.session.execution_graph_generation == 2
    assert second["receipt"]["graphId"] != first["receipt"]["graphId"]


def test_execution_graph_reopen_rejects_stale_identity_without_mutation() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "primary metric")
    tools = {item.name: item for item in runtime.tools}
    frozen = _freeze_reopenable_execution_graph(runtime, context)
    query_id = frozen["clientNodeIds"]["primary_query"]
    branch = context.session.query_branch_contexts[query_id]
    branch.status = "CONTRACT_GAPPED"
    branch.last_gaps = [{"code": "BINDING_EVIDENCE_REQUIRED"}]

    wrong_graph = json.loads(
        tools["reopen_grounded_execution_graph_discovery"].func(
            graph_id="graph_stale",
            version=frozen["receipt"]["version"],
            reason="structured gap",
            runtime=SimpleNamespace(context=context),
        )
    )
    wrong_version = json.loads(
        tools["reopen_grounded_execution_graph_discovery"].func(
            graph_id=frozen["receipt"]["graphId"],
            version=frozen["receipt"]["version"] + 1,
            reason="structured gap",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert wrong_graph["code"] == "EXECUTION_GRAPH_REOPEN_STALE"
    assert wrong_version["code"] == "EXECUTION_GRAPH_REOPEN_STALE"
    assert context.session.execution_graph_receipt is not None
    assert set(context.session.query_branch_contexts) == {query_id}


def test_execution_graph_reopen_requires_a_current_typed_contract_gap() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "primary metric")
    tools = {item.name: item for item in runtime.tools}
    frozen = _freeze_reopenable_execution_graph(runtime, context)

    rejected = json.loads(
        tools["reopen_grounded_execution_graph_discovery"].func(
            graph_id=frozen["receipt"]["graphId"],
            version=frozen["receipt"]["version"],
            reason="request more evidence",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert rejected["status"] == "REJECTED"
    assert rejected["code"] == "EXECUTION_GRAPH_TYPED_GAP_REQUIRED"
    assert context.session.execution_graph_receipt is not None


def test_execution_graph_reopen_is_forbidden_after_verified_artifact() -> None:
    runtime, kernel, _ = _runtime(require_parallel_overlap=False)
    context = _context(kernel, "primary metric")
    tools = {item.name: item for item in runtime.tools}
    frozen = _freeze_reopenable_execution_graph(runtime, context)
    query_id = frozen["clientNodeIds"]["primary_query"]
    branch = context.session.query_branch_contexts[query_id]
    branch.status = "CONTRACT_GAPPED"
    branch.last_gaps = [{"code": "BINDING_EVIDENCE_REQUIRED"}]
    branch.verified_artifact_ids = ["artifact:verified"]

    rejected = json.loads(
        tools["reopen_grounded_execution_graph_discovery"].func(
            graph_id=frozen["receipt"]["graphId"],
            version=frozen["receipt"]["version"],
            reason="request more evidence",
            runtime=SimpleNamespace(context=context),
        )
    )

    assert rejected["status"] == "REJECTED"
    assert rejected["code"] == (
        "EXECUTION_GRAPH_REOPEN_AFTER_EXECUTION_FORBIDDEN"
    )
    assert rejected["executedQueryIds"] == [query_id]
    assert context.session.execution_graph_receipt is not None
