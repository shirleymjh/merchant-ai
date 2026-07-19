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
    DetailQuestionGoal,
    DimensionQuestionGoal,
    MetricQuestionGoal,
    OriginalQuestionGoalContract,
    RankingQuestionGoal,
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
                (
                    (item_path, item)
                    for item_path, item in self.documents.items()
                    if item["refId"] == ref_id
                ),
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
            merchant=kwargs.get("merchant")
            or MerchantInfo(merchant_id=merchant_id),
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
            contract_fingerprint=grounded_query_contract_fingerprint(
                session.active_contract
            ),
            sql_fingerprint=hashlib.sha256(
                session.session_id.encode("utf-8")
            ).hexdigest(),
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
        return (
            session.verified_query_ledger[-1]
            if session.verified_query_ledger
            else None
        )

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
    kernel = _BranchKernel(
        require_parallel_overlap=require_parallel_overlap
    )
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
    assert goal_result["nextAction"] == (
        "DECLARE_QUERY_BRANCHES_BEFORE_RETRIEVAL"
    )
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
    assert declared["status"] == "DECLARED"

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
                        "metricRefs": [
                            "semantic:电商交易:orders:metric:order_count"
                        ],
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
                        "metricRefs": [
                            "semantic:电商退货:refunds:metric:refund_amount"
                        ],
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
    assert all(
        item["topic"] == "电商交易"
        for item in orders.semantic_ledger.evidence()
    )
    assert all(
        item["topic"] == "电商退货"
        for item in refunds.semantic_ledger.evidence()
    )
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
    context = _context(kernel, "先查商品排行，再查这些商品退款")
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
                    label="这些商品退款",
                    depends_on_goal_ids=["metric.top_products"],
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
    assert declared["waitingForVerifiedEntitySetQueryIds"] == [
        "product-refunds"
    ]

    result = json.loads(
        tools["prepare_grounded_query_batch"].func(
            queries=[{"queryId": "product-refunds"}],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert result["status"] == "BLOCKED"
    assert result["queries"][0]["status"] == (
        "WAITING_VERIFIED_ENTITY_SET"
    )
    assert context.session.query_branch_contexts[
        "product-refunds"
    ].runtime is None
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
    assert {
        item["code"] for item in rejected["issues"]
    } == {"QUERY_BRANCH_STRUCTURAL_GOALS_NOT_COLOCATED"}

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
    assert accepted["status"] == "DECLARED"


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
    assert rejected["issues"][0]["code"] == (
        "QUERY_BRANCH_STRUCTURAL_GOALS_NOT_COLOCATED"
    )

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
    assert accepted["status"] == "DECLARED"
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
                    "semanticPaths": [
                        "topics/电商交易/tables/orders/detail.json"
                    ],
                    "bindingHints": {
                        "tableRefs": ["semantic:电商交易:orders:detail"]
                    },
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
                        "metricRefs": [
                            navigation["metricLeaves"][0]["refId"]
                        ],
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
    boundary = GroundedCoreToolBoundaryMiddleware(
        GroundedSemanticBackend(catalog)
    )
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={
            "id": "read-1",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/电商交易/tables/orders/detail.json"
            },
        },
    )

    result = boundary.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(
            AssertionError("global handler must not run")
        ),
    )

    payload = json.loads(str(result.content))
    assert result.status == "error"
    assert payload["code"] == "BRANCH_SCOPED_RETRIEVAL_REQUIRED"


def test_query_goal_requires_branch_declaration_before_first_read() -> None:
    runtime, kernel, catalog = _runtime()
    context = _context(kernel, "订单量")
    tools = {item.name: item for item in runtime.tools}
    tools["declare_original_question_goals"].func(
        contract=OriginalQuestionGoalContract(
            question=context.session.runtime.question,
            goals=[
                MetricQuestionGoal(goal_id="metric.orders", label="订单量")
            ],
        ),
        runtime=SimpleNamespace(context=context),
    )
    boundary = GroundedCoreToolBoundaryMiddleware(
        GroundedSemanticBackend(catalog)
    )
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={
            "id": "read-before-branch",
            "name": "read_file",
            "args": {
                "file_path": "/knowledge/topics/电商交易/tables/orders/detail.json"
            },
        },
    )

    result = boundary.wrap_tool_call(
        request,
        lambda _: (_ for _ in ()).throw(
            AssertionError("read must wait for branch declaration")
        ),
    )

    payload = json.loads(str(result.content))
    assert result.status == "error"
    assert payload["code"] == "QUERY_BRANCH_DECLARATION_REQUIRED"
    assert context.session.core_semantic_evidence == []
