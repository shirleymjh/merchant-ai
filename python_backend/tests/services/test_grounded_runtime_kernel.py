from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.models import (
    AgentRunResult,
    ExtractedKeywords,
    GraphValidationGap,
    GraphValidationResult,
    KnowledgeBundle,
    MerchantInfo,
    PlanningAssetPack,
    QueryPlan,
    QuestionCategory,
    RecallBundle,
    RecallItem,
    ResolvedTimeRange,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedTableBinding,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeKernel


class FakeTopicAssets:
    @staticmethod
    def topic_names_for_categories(categories: list[QuestionCategory]) -> list[str]:
        return ["客服工单" for item in categories if item == QuestionCategory.SERVICE]


class FakeKeywordService:
    @staticmethod
    def extract(question: str) -> ExtractedKeywords:
        return ExtractedKeywords(
            normalized_question=question,
            keywords=["工单"],
            topic_scores={QuestionCategory.SERVICE.value: 5.0},
        )


class FakeRouter:
    @staticmethod
    def route(question: str, keywords: ExtractedKeywords) -> TopicRoutingDecision:
        assert question == keywords.normalized_question
        return TopicRoutingDecision(
            primary_topic=QuestionCategory.SERVICE,
            candidate_topics=[QuestionCategory.SERVICE],
            confidence=0.9,
            routing_mode="seed_topic",
        )


class FakeRecall:
    calls: int = 0

    def recall(self, question: str, keywords: Any, history: list[Any], context: str, merchant_id: str, topics: list[Any]) -> RecallBundle:
        self.calls += 1
        assert merchant_id == "m-1"
        assert topics == [QuestionCategory.SERVICE]
        return RecallBundle(
            items=[
                RecallItem(
                    doc_id="semantic:客服工单:manifest",
                    topic="客服工单",
                    source_type="TOPIC_MANIFEST",
                )
            ]
        )


class FakeKnowledgeRetriever:
    backend_name = "es"

    def __init__(self):
        self.requests: list[Any] = []

    def retrieve(self, request: Any) -> KnowledgeBundle:
        self.requests.append(request)
        suffix = "supplemental" if request.query == "商品字段" else "initial"
        return KnowledgeBundle(
            backend="es",
            recall_bundle=RecallBundle(
                items=[
                    RecallItem(
                        doc_id="semantic:客服工单:%s" % suffix,
                        topic="客服工单",
                        source_type="COLUMN" if suffix == "supplemental" else "TOPIC_MANIFEST",
                    )
                ]
            ),
        )


class QueueBuilder:
    def __init__(self, contracts: list[GroundedQueryContract]):
        self.contracts = list(contracts)

    def build(self, question: str, topics: list[str], evidence: list[dict[str, Any]], **kwargs: Any) -> GroundedQueryContract:
        contract = self.contracts.pop(0)
        assert contract.question == question
        return contract


class FakeCompiler:
    def __init__(self, valid: bool = True):
        self.valid = valid
        self.calls = 0

    def __call__(self, contract: GroundedQueryContract, pack: PlanningAssetPack) -> Any:
        self.calls += 1
        validation = GraphValidationResult(
            valid=self.valid,
            gaps=(
                []
                if self.valid
                else [GraphValidationGap(code="BROKEN", reason="candidate is invalid")]
            ),
        )
        return SimpleNamespace(
            validation=validation,
            plan=QueryPlan(agent_trace=[contract.question]),
            executable=self.valid,
        )


class FakeExecutor:
    execute_calls = 0

    def execute_contract(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        pack: PlanningAssetPack,
        question: str,
        **kwargs: Any,
    ) -> AgentRunResult:
        self.execute_calls += 1
        assert contract.ready is True
        assert kwargs["execution_preparation"].executable is True
        return AgentRunResult()


class FakeVerifier:
    @staticmethod
    def verify(question: str, plan: QueryPlan, run_result: AgentRunResult) -> VerifiedEvidence:
        return VerifiedEvidence(passed=True, covered_evidence=[question])


class FakeComposer:
    @staticmethod
    def compose(question: str, merchant: MerchantInfo, plan: QueryPlan, run_result: AgentRunResult, context: str, **kwargs: Any) -> str:
        return "%s:%s" % (merchant.merchant_id, question)


def contract(question: str, status: str) -> GroundedQueryContract:
    candidate = GroundedQueryContract(
        question=question,
        topics=["客服工单"],
        status=status,
        query_shape="SCALAR" if status == "READY" else "UNRESOLVED",
    )
    if status != "READY":
        return candidate
    candidate.primary_table = "tickets"
    candidate.tables = [GroundedTableBinding(topic="客服工单", table="tickets")]
    candidate.metrics = [
        GroundedMetricBinding(
            requested_phrase="工单量",
            semantic_ref_id="semantic:客服工单:tickets:metric:ticket_count",
            topic="客服工单",
            table="tickets",
            metric_key="ticket_count",
            formula="SUM(ticket_count)",
            source_columns=["ticket_count"],
            aggregation_policy="period_rollup",
            metric_grain="merchant_day",
            applicable_time_grain="period",
            time_column="event_day",
            time_semantics={
                "selectionPolicy": "period_window",
                "asOfPolicy": "calendar",
                "missingDataPolicy": "disclose_unknown",
                "zeroValuePolicy": "preserve_observed_zero",
            },
            binding_type="published_metric",
        )
    ]
    candidate.time_range = ResolvedTimeRange(
        days=30,
        explicit=True,
        window_role="primary",
    )
    return candidate


def kernel(*, builder: Any, compiler: Any, executor: Any | None = None) -> GroundedRuntimeKernel:
    return GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
        recall_service=FakeRecall(),
        contract_builder=builder,
        asset_materializer=lambda candidate, assets: PlanningAssetPack(),
        compiler=compiler,
        executor=executor,
        verifier=FakeVerifier(),
        answer_composer=FakeComposer(),
    )


def test_kernel_has_no_legacy_workflow_or_action_dependency() -> None:
    source = Path(
        "python_backend/merchant_ai/services/grounded_runtime_kernel.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "MerchantQaWorkflow",
        "V2AgentPolicy",
        "StateGraph",
        "run_diana_action",
    ):
        assert forbidden not in source


def test_route_and_recall_are_direct_typed_kernel_operations() -> None:
    runtime = kernel(
        builder=QueueBuilder([]),
        compiler=FakeCompiler(),
    )
    session = runtime.new_session("最近30天工单量", "m-1")

    routing = runtime.route_topic(session)
    recalled = runtime.recall_navigation(session)

    assert routing.primary_topic == QuestionCategory.SERVICE
    assert session.workspace_topics == ["客服工单"]
    assert recalled.items[0].doc_id == "semantic:客服工单:manifest"
    assert session.phase == "NAVIGATION_RECALLED"


def test_unified_retriever_is_preferred_with_strict_topic_scope_and_supplemental_merge() -> None:
    retriever = FakeKnowledgeRetriever()
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
        recall_service=retriever,
        contract_builder=QueueBuilder([]),
    )
    session = runtime.new_session("最近30天工单量", "m-1")
    runtime.route_topic(session)

    initial = runtime.recall_navigation(session)
    supplemental = runtime.recall_navigation(session, query="商品字段")

    assert retriever.requests[0].strict_topic_scope is True
    assert retriever.requests[0].topic_categories == [QuestionCategory.SERVICE]
    assert retriever.requests[1].query == "商品字段"
    assert retriever.requests[1].strict_topic_scope is True
    assert [item.doc_id for item in initial.items] == ["semantic:客服工单:initial"]
    assert {item.doc_id for item in supplemental.items} == {
        "semantic:客服工单:initial",
        "semantic:客服工单:supplemental",
    }


def test_unresolved_candidate_is_recorded_without_replacing_active_contract() -> None:
    question = "最近30天工单量"
    ready = contract(question, "READY")
    unresolved = contract(question, "UNRESOLVED")
    compiler = FakeCompiler()
    runtime = kernel(
        builder=QueueBuilder([ready, unresolved]),
        compiler=compiler,
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    first = runtime.propose_contract(session, [], {})
    runtime.activate_contract(session, first.attempt_id)
    active_generation = session.active_generation
    active_attempt_id = session.active_attempt_id

    second = runtime.propose_contract(session, [], {})
    compiled_second = runtime.activate_contract(session, second.attempt_id)

    assert compiled_second.compile_status == "SKIPPED_NOT_READY"
    assert compiler.calls == 1
    assert session.active_generation == active_generation
    assert session.active_attempt_id == active_attempt_id
    assert session.active_contract is not None
    assert session.active_contract.status == "READY"


def test_ready_contract_is_routed_before_any_compilation() -> None:
    question = "最近30天工单量"
    compiler = FakeCompiler()
    runtime = kernel(
        builder=QueueBuilder([contract(question, "READY")]),
        compiler=compiler,
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    attempt = runtime.propose_contract(session, [], {})

    assert attempt.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC
    assert attempt.fast_path_eligible is True
    assert attempt.fast_path_reason_codes == []
    assert attempt.execution_reason_codes == [
        "SINGLE_METRIC_FAST_PATH_ELIGIBLE"
    ]
    assert attempt.next_action == "ACTIVATE_DETERMINISTIC_METRIC"
    assert compiler.calls == 0


def test_complex_ready_contract_activates_scope_without_template_compilation() -> None:
    question = "最近30天按商品统计工单量"
    complex_contract = contract(question, "READY")
    complex_contract.query_shape = "GROUPED"
    complex_contract.dimensions = [
        GroundedDimensionBinding(
            requested_phrase="商品",
            semantic_ref_id="semantic:客服工单:tickets:column:spu_id",
            topic="客服工单",
            table="tickets",
            column="spu_id",
            usage="group_by",
        )
    ]
    compiler = FakeCompiler()
    materialized: list[str] = []
    runtime = kernel(builder=QueueBuilder([complex_contract]), compiler=compiler)
    runtime.asset_materializer = lambda candidate, assets: materialized.append(
        candidate.question
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    attempt = runtime.propose_contract(session, [], {})
    activated = runtime.activate_contract(session, attempt.attempt_id)

    assert activated.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert activated.fast_path_eligible is False
    assert set(activated.fast_path_reason_codes) >= {
        "QUERY_SHAPE_NOT_SCALAR",
        "DIMENSIONS_PRESENT",
        "GROUPING_PRESENT",
    }
    assert activated.execution_reason_codes[0] == (
        "COMPLEX_QUERY_REQUIRES_CORE_SQL"
    )
    assert activated.compile_status == "NOT_APPLICABLE_CORE_SQL_REQUIRED"
    assert activated.activation_status == "ACTIVATED"
    assert activated.next_action == "SUBMIT_GROUNDED_SQL_CANDIDATE"
    assert activated.activated is True
    assert compiler.calls == 0
    assert materialized == []
    assert session.active_execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert session.active_contract == complex_contract
    assert session.active_pack is None
    assert session.active_plan is None
    assert session.active_preparation is None
    assert session.phase == "ACTIVE_CORE_SQL_REQUIRED"

    with pytest.raises(RuntimeError, match="latest SQL candidate is not"):
        runtime.execute_active(session)


def test_complex_activation_replaces_old_compiled_artifacts_atomically() -> None:
    question = "最近30天工单量与商品分布"
    simple_contract = contract(question, "READY")
    complex_contract = contract(question, "READY")
    complex_contract.metrics.append(
        complex_contract.metrics[0].model_copy(
            update={
                "semantic_ref_id": "semantic:客服工单:tickets:metric:buyer_count",
                "metric_key": "buyer_count",
            }
        )
    )
    compiler = FakeCompiler()
    runtime = kernel(
        builder=QueueBuilder([simple_contract, complex_contract]),
        compiler=compiler,
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    first = runtime.propose_contract(session, [], {})
    runtime.activate_contract(session, first.attempt_id)
    first_generation = session.active_generation
    assert session.active_plan is not None

    second = runtime.propose_contract(session, [], {})
    runtime.activate_contract(session, second.attempt_id)

    assert compiler.calls == 1
    assert session.active_generation == first_generation + 1
    assert session.active_attempt_id == second.attempt_id
    assert session.active_execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert session.active_plan is None
    assert session.active_pack is None
    assert session.active_preparation is None


def test_invalid_compilation_never_partially_switches_active_generation() -> None:
    question = "最近30天工单量"
    runtime = kernel(
        builder=QueueBuilder([contract(question, "READY")]),
        compiler=FakeCompiler(valid=False),
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]
    attempt = runtime.propose_contract(session, [], {})

    compiled = runtime.activate_contract(session, attempt.attempt_id)

    assert compiled.compile_status == "VALIDATION_FAILED"
    assert compiled.validation_gaps[0]["code"] == "BROKEN"
    assert session.active_generation == 0
    assert session.active_contract is None
    assert session.active_pack is None
    assert session.active_plan is None


def test_valid_candidate_executes_verifies_and_answers_through_contract_executor() -> None:
    question = "最近30天工单量"
    executor = FakeExecutor()
    runtime = kernel(
        builder=QueueBuilder([contract(question, "READY")]),
        compiler=FakeCompiler(valid=True),
        executor=executor,
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]
    attempt = runtime.propose_contract(session, [], {})
    activated = runtime.activate_contract(session, attempt.attempt_id)

    run_result = runtime.execute_active(session)
    verified = runtime.verify_active(session)
    answer = runtime.compose_answer(session)

    assert activated.activated is True
    assert activated.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC
    assert activated.activation_status == "ACTIVATED"
    assert isinstance(run_result, AgentRunResult)
    assert verified.passed is True
    assert answer == "m-1:最近30天工单量"
    assert executor.execute_calls == 1
    assert session.phase == "ANSWERED"


def test_missing_runtime_services_fail_closed_and_clarification_is_typed() -> None:
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
    )
    session = runtime.new_session("工单量最多的商品", "m-1")

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        runtime.recall_navigation(session)

    clarification = runtime.request_clarification(
        session,
        "请问要查询最近多少天？",
        stage="time_binding",
        clarification_type="TIME_RANGE_REQUIRED",
        options=["最近7天", "最近30天"],
    )
    assert clarification.pending_question == session.question
    assert session.phase == "CLARIFICATION_REQUIRED"
