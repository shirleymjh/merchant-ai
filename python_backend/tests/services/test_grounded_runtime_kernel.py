from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.models import (
    AgentTaskResult,
    AgentRunResult,
    EntityFilterObligation,
    EntityFilterVerificationProof,
    EntityReference,
    ExtractedKeywords,
    GraphValidationGap,
    GraphValidationResult,
    KnowledgeBundle,
    MerchantInfo,
    NodePlanContract,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    QuestionIntent,
    QuestionCategory,
    RecallBundle,
    RecallItem,
    ResolvedTimeRange,
    RouteSlots,
    RouteTimeWindow,
    TopicRoutingDecision,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_query_contract import (
    GroundedBindingHints,
    GroundedContractGap,
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedFieldAggregationHint,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
    GroundedUpstreamEntityHint,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedVerifiedEntitySet,
    GroundedVerifiedQueryArtifact,
    _namespace_artifact_run,
)
from merchant_ai.services.entity_contracts import (
    canonical_entity_values,
    entity_comparison_policy,
    entity_filter_contract_hash,
    entity_filter_sql_hash,
    entity_value_hash,
)
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)


class FakeTopicAssets:
    @staticmethod
    def topic_names_for_categories(categories: list[QuestionCategory]) -> list[str]:
        return ["客服工单" for item in categories if item == QuestionCategory.SERVICE]


def test_independent_metric_contract_enters_topology_repair_phase() -> None:
    class TopologyContractBuilder:
        @staticmethod
        def build(*args: Any, **kwargs: Any) -> GroundedQueryContract:
            del args, kwargs
            return GroundedQueryContract(
                question="查询两个独立指标",
                status="REVISE_BINDINGS",
                query_shape="MULTI_TABLE",
                unresolved_gaps=[
                    GroundedContractGap(
                        code="INDEPENDENT_QUERY_SPLIT_REQUIRED",
                        message="split independent metrics",
                        evidence_kind="QUERY_TOPOLOGY",
                        resolution=(
                            "REBIND_TO_COMPATIBLE_SINGLE_TABLE_OR_PROPOSE_EXECUTION_GRAPH"
                        ),
                    )
                ],
            )

    kernel = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        route_slot_extractor=SimpleNamespace(),
        topic_router=FakeRouter(),
        contract_builder=TopologyContractBuilder(),
    )
    session = kernel.new_session("查询两个独立指标", "merchant-1")
    session.workspace_topics = ["客服工单"]

    attempt = kernel.propose_contract(session, [], {})

    assert session.phase == "QUERY_REPAIR_REQUIRED"
    assert attempt.repair_type == "TOPOLOGY"
    assert attempt.next_action == (
        "CHOOSE_SAFE_REPAIR_AND_SUBMIT_NEW_VERSION"
    )
    assert attempt.contract_version == 1
    assert attempt.parent_attempt_id == ""
    assert attempt.activated is False

    repaired = kernel.propose_contract(session, [], {})

    assert repaired.contract_version == 2
    assert repaired.parent_attempt_id == attempt.attempt_id
    assert repaired.parent_contract_fingerprint == (
        grounded_query_contract_fingerprint(attempt.contract)
    )


def test_core_sql_evidence_plan_preserves_dimension_and_selected_field_labels() -> None:
    contract = GroundedQueryContract(
        question="查询商品工单量和商品发布时间",
        status="READY",
        query_shape="DETAIL",
        primary_table="dwm_goods_detail_df",
        dimensions=[
            GroundedDimensionBinding(
                requested_phrase="商品",
                semantic_ref_id="semantic:goods:field:spu_id",
                topic="商品管理",
                table="dwm_goods_detail_df",
                column="spu_id",
                business_name="商品id",
            )
        ],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id="semantic:goods:field:publish_time",
                topic="商品管理",
                table="dwm_goods_detail_df",
                column="spu_apply_create_time",
                business_name="创建时间",
                aliases=["创建时间", "商品发布时间"],
                output_alias="spu_apply_create_time",
            )
        ],
    )
    validation = GroundedSqlValidationResult(
        valid=True,
        canonical_sql=(
            "SELECT spu_id, spu_apply_create_time FROM dwm_goods_detail_df"
        ),
        ast_fingerprint="a" * 64,
        referenced_tables=["dwm_goods_detail_df"],
        output_columns=["spu_id", "spu_apply_create_time"],
    )

    plan = GroundedRuntimeKernel._build_core_sql_evidence_plan(
        contract,
        validation,
    )

    assert plan.intents[0].metric_resolution["sourceColumnLabels"] == {
        "spu_id": "商品id",
        "spu_apply_create_time": "商品发布时间",
    }


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


class UpstreamCapturingBuilder:
    def build(
        self,
        question: str,
        topics: list[str],
        evidence: list[dict[str, Any]],
        **kwargs: Any,
    ) -> GroundedQueryContract:
        hints = kwargs["binding_hints"]
        entity_filter = hints.entity_filters[-1]
        target_ref = entity_filter.field_ref
        return GroundedQueryContract(
            question=question,
            topics=topics,
            status="READY",
            query_shape="ENTITY_LOOKUP",
            primary_table="goods",
            binding_hints=hints,
            tables=[
                GroundedTableBinding(
                    topic="商品管理",
                    table="goods",
                    detail_ref_id="semantic:商品管理:goods:detail",
                    merchant_filter_column="seller_id",
                )
            ],
            selected_fields=[
                GroundedSelectedFieldBinding(
                    semantic_ref_id="semantic:商品管理:goods:field:publish_time",
                    topic="商品管理",
                    table="goods",
                    column="publish_time",
                    output_alias="publish_time",
                )
            ],
            entity_filters=[
                GroundedEntityFilterBinding(
                    semantic_ref_id=target_ref,
                    topic="商品管理",
                    table="goods",
                    column="spu_id",
                    operator=entity_filter.operator,
                    literal_value=entity_filter.literal_value,
                    entity_identity="entity:product",
                    allowed_operators=["IN"],
                )
            ],
            evidence_refs=[
                "semantic:商品管理:goods:detail",
                target_ref,
                "semantic:商品管理:goods:field:publish_time",
            ],
        )


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

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

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
        self.last_kwargs = dict(kwargs)
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


class PortfolioComposer:
    def __init__(self) -> None:
        self.plan: QueryPlan | None = None
        self.run_result: AgentRunResult | None = None

    def compose(
        self,
        question: str,
        merchant: MerchantInfo,
        plan: QueryPlan,
        run_result: AgentRunResult,
        context: str,
        **kwargs: Any,
    ) -> str:
        self.plan = plan.model_copy(deep=True)
        self.run_result = run_result.model_copy(deep=True)
        return "portfolio-answer"


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
    assert retriever.requests[0].route_slots["timeWindow"]["days"] == 30
    assert retriever.requests[0].route_slots["topicCandidates"][0]["topic"] == (
        QuestionCategory.SERVICE.value
    )
    assert retriever.requests[0].intent_kind == "lookup"
    assert retriever.requests[0].complexity == "simple"
    assert retriever.requests[1].query == "商品字段"
    assert retriever.requests[1].strict_topic_scope is True
    assert [item.doc_id for item in initial.items] == ["semantic:客服工单:initial"]
    assert {item.doc_id for item in supplemental.items} == {
        "semantic:客服工单:initial",
        "semantic:客服工单:supplemental",
    }


def test_initial_route_extracts_keywords_and_slots_once_for_topic_and_recall() -> None:
    class CountingKeywordService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract(self, question: str) -> ExtractedKeywords:
            self.calls.append(question)
            return ExtractedKeywords(
                normalized_question=question,
                keywords=["工单"],
                topic_scores={QuestionCategory.SERVICE.value: 5.0},
            )

    class CountingSlotExtractor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ExtractedKeywords]] = []

        def extract(
            self,
            question: str,
            keywords: ExtractedKeywords,
        ) -> RouteSlots:
            self.calls.append((question, keywords))
            return RouteSlots(
                time_window=RouteTimeWindow(days=30, raw="最近30天")
            )

    class RecordingRouter:
        def __init__(self) -> None:
            self.keywords: ExtractedKeywords | None = None
            self.route_slots: RouteSlots | None = None

        def route_with_budget(
            self,
            question: str,
            *,
            keywords: ExtractedKeywords,
            route_slots: RouteSlots,
            runtime_budget: Any = None,
        ) -> TopicRoutingDecision:
            del question, runtime_budget
            self.keywords = keywords
            self.route_slots = route_slots
            return TopicRoutingDecision(
                primary_topic=QuestionCategory.SERVICE,
                candidate_topics=[QuestionCategory.SERVICE],
            )

    keywords = CountingKeywordService()
    slots = CountingSlotExtractor()
    router = RecordingRouter()
    retriever = FakeKnowledgeRetriever()
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=keywords,
        route_slot_extractor=slots,
        topic_router=router,
        recall_service=retriever,
        contract_builder=QueueBuilder([]),
    )
    session = runtime.new_session("最近30天工单量", "m-1")

    runtime.route_topic(session)
    runtime.recall_navigation(session)

    assert keywords.calls == ["最近30天工单量"]
    assert len(slots.calls) == 1
    assert router.keywords is session.keywords
    assert router.route_slots is session.route_slots
    assert retriever.requests[0].route_slots["timeWindow"]["days"] == 30


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
    assert attempt.next_action == "ACTIVATE_DETERMINISTIC_QUERY"
    assert compiler.calls == 0


def test_allowlisted_field_count_is_routed_to_deterministic_kernel_lane() -> None:
    question = "最近30天涉及多少买家"
    candidate = contract(question, "READY")
    field_ref = "semantic:客服工单:tickets:column:buyer_id"
    metric = candidate.metrics[0]
    metric.semantic_ref_id = field_ref
    metric.source_field_ref_id = field_ref
    metric.binding_type = "field_aggregation"
    metric.field_aggregation = "COUNT_DISTINCT"
    metric.metric_key = "count_distinct_buyer_id"
    metric.formula = "COUNT(DISTINCT `buyer_id`)"
    metric.source_columns = ["buyer_id"]
    metric.calculation_capabilities = {
        "allowedAggregations": ["COUNT_DISTINCT"],
        "declaredAggregation": "COUNT_DISTINCT",
    }
    candidate.binding_hints.field_aggregations = [
        GroundedFieldAggregationHint(
            field_ref=field_ref,
            aggregation="COUNT_DISTINCT",
        )
    ]
    runtime = kernel(
        builder=QueueBuilder([candidate]),
        compiler=FakeCompiler(),
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    attempt = runtime.propose_contract(session, [], {})

    assert attempt.execution_mode == GroundedExecutionMode.DETERMINISTIC_METRIC
    assert attempt.fast_path_eligible is True
    assert attempt.next_action == "ACTIVATE_DETERMINISTIC_QUERY"


def test_grouped_ready_contract_uses_deterministic_compilation() -> None:
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
    def materialize(candidate: GroundedQueryContract, assets: Any) -> PlanningAssetPack:
        materialized.append(candidate.question)
        return PlanningAssetPack()

    runtime.asset_materializer = materialize
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]

    attempt = runtime.propose_contract(session, [], {})
    activated = runtime.activate_contract(session, attempt.attempt_id)

    assert activated.execution_mode == GroundedExecutionMode.DETERMINISTIC_GROUPED
    assert activated.fast_path_eligible is True
    assert activated.fast_path_reason_codes == []
    assert activated.execution_reason_codes == [
        "GROUPED_DETERMINISTIC_ELIGIBLE"
    ]
    assert activated.compile_status == "VALID"
    assert activated.activation_status == "ACTIVATED"
    assert activated.next_action == "EXECUTE_GROUNDED_QUERY"
    assert activated.activated is True
    assert compiler.calls == 1
    assert materialized == [question]
    assert session.active_execution_mode == GroundedExecutionMode.DETERMINISTIC_GROUPED
    assert session.active_contract == complex_contract
    assert session.active_pack is not None
    assert session.active_plan is not None
    assert session.active_preparation is not None
    assert session.phase == "ACTIVE_COMPILED"


def test_multi_metric_activation_replaces_old_compiled_artifacts_atomically() -> None:
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

    assert compiler.calls == 2
    assert session.active_generation == first_generation + 1
    assert session.active_attempt_id == second.attempt_id
    assert session.active_execution_mode == GroundedExecutionMode.DETERMINISTIC_MULTI_METRIC
    assert session.active_plan is not None
    assert session.active_pack is not None
    assert session.active_preparation is not None


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
    assert "runtime_budget" not in executor.last_kwargs
    assert session.phase == "ANSWERED"


def test_serial_verified_artifact_requires_pre_ledger_authorization() -> None:
    question = "最近30天工单量"
    runtime = kernel(
        builder=QueueBuilder([contract(question, "READY")]),
        compiler=FakeCompiler(valid=True),
        executor=FakeExecutor(),
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]
    attempt = runtime.propose_contract(session, [], {})
    runtime.activate_contract(session, attempt.attempt_id)
    runtime.execute_active(session)

    with pytest.raises(RuntimeError) as raised:
        runtime.verify_active(
            session,
            pre_ledger_authorizer=lambda _artifact: False,
        )

    assert str(raised.value) == (
        "VERIFIED_QUERY_ARTIFACT_ADOPTION_REJECTED"
    )
    assert session.verified_query_ledger == []
    assert session.verified_evidence is None
    assert session.run_result is None
    assert runtime.latest_verified_query_artifact(session) is None
    assert session.answer == ""


def test_serial_verified_artifact_is_visible_after_authorization() -> None:
    question = "最近30天工单量"
    runtime = kernel(
        builder=QueueBuilder([contract(question, "READY")]),
        compiler=FakeCompiler(valid=True),
        executor=FakeExecutor(),
    )
    session = runtime.new_session(question, "m-1")
    session.workspace_topics = ["客服工单"]
    attempt = runtime.propose_contract(session, [], {})
    runtime.activate_contract(session, attempt.attempt_id)
    runtime.execute_active(session)
    staged_ids: list[str] = []

    verified = runtime.verify_active(
        session,
        pre_ledger_authorizer=lambda artifact: (
            staged_ids.append(artifact.artifact_id) is None
        ),
    )

    assert verified.passed is True
    assert len(staged_ids) == 1
    assert [item.artifact_id for item in session.verified_query_ledger] == (
        staged_ids
    )
    assert runtime.latest_verified_query_artifact(session) is not None


def test_kernel_forwards_optional_runtime_budget_to_contract_executor() -> None:
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
    runtime.activate_contract(session, attempt.attempt_id)
    shared_budget = object()

    runtime.execute_active(session, runtime_budget=shared_budget)

    assert executor.last_kwargs["runtime_budget"] is shared_budget


def test_missing_runtime_services_fail_closed_and_clarification_is_typed() -> None:
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
    )
    session = runtime.new_session("工单量最多的商品", "m-1")

    with pytest.raises(RuntimeError) as exc_info:
        runtime.recall_navigation(session)
    assert "refusing to fall back" in str(exc_info.value)

    clarification = runtime.request_clarification(
        session,
        "请问要查询最近多少天？",
        stage="time_binding",
        clarification_type="TIME_RANGE_REQUIRED",
        options=["最近7天", "最近30天"],
    )
    assert clarification.pending_question == session.question
    assert session.phase == "CLARIFICATION_REQUIRED"


def test_verified_entity_set_materializes_typed_downstream_filter_and_audit() -> None:
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
        contract_builder=UpstreamCapturingBuilder(),
    )
    session = runtime.new_session("查看该商品发布时间", "m-1")
    session.workspace_topics = ["商品管理"]
    source_contract = GroundedQueryContract(
        question="工单量最多的商品",
        topics=["客服工单"],
        status="READY",
        query_shape="RANKED",
    )
    source_contract_fingerprint = grounded_query_contract_fingerprint(
        source_contract
    )
    source_query = GroundedVerifiedQueryArtifact(
        artifact_id="query_artifact_top_product",
        generation=1,
        contract_fingerprint=source_contract_fingerprint,
        sql_fingerprint="source-sql-fingerprint",
        contract=source_contract,
        plan=QueryPlan(),
        run_result=AgentRunResult(),
        verified_evidence=VerifiedEvidence(passed=True),
        output_columns=["spu_id"],
        output_semantic_refs={
            "spu_id": "semantic:客服工单:tickets:field:spu_id"
        },
        output_entity_identities={"spu_id": "entity:product"},
        sealed_entity_values={"spu_id": ["spu-2", "spu-1"]},
    )
    entity_set = GroundedVerifiedEntitySet(
        artifact_id="entity_set_top_product",
        source_query_artifact_id=source_query.artifact_id,
        source_column="spu_id",
        source_semantic_ref_id=(
            "semantic:客服工单:tickets:field:spu_id"
        ),
        source_entity_identity="entity:product",
        values=["spu-1", "spu-2"],
        value_count=2,
        values_hash="set-values-hash",
    )
    session.verified_query_ledger.append(source_query)
    session.verified_entity_sets.append(entity_set)
    target_ref = "semantic:商品管理:goods:field:spu_id"
    hints = GroundedBindingHints(
        table_refs=["semantic:商品管理:goods:detail"],
        selected_fields=[
            {
                "fieldRef": "semantic:商品管理:goods:field:publish_time",
                "outputAlias": "publish_time",
            }
        ],
        upstream_entity_bindings=[
            GroundedUpstreamEntityHint(
                entity_set_artifact_id=entity_set.artifact_id,
                target_field_ref=target_ref,
                operator="IN",
            )
        ],
        analysis_mode="ENTITY_LOOKUP",
    )

    attempt = runtime.propose_contract(session, [], hints)

    assert attempt.contract.ready is True
    assert attempt.contract.entity_filters[0].literal_value == ["spu-1", "spu-2"]
    assert attempt.contract.entity_filters[0].operator == "IN"
    assert attempt.contract.upstream_entity_bindings[0].entity_set_artifact_id == (
        entity_set.artifact_id
    )
    assert attempt.contract.upstream_entity_bindings[0].source_contract_fingerprint == (
        source_contract_fingerprint
    )
    assert attempt.contract.upstream_entity_bindings[0].target_entity_identity == (
        "entity:product"
    )


def test_verified_portfolio_preserves_multiple_query_graphs_and_namespaces_tasks() -> None:
    runtime = GroundedRuntimeKernel(
        FakeTopicAssets(),
        keyword_service=FakeKeywordService(),
        topic_router=FakeRouter(),
    )
    session = runtime.new_session("工单最多商品及退款和发布时间", "m-1")
    for artifact_id, table, row in [
        (
            "query_artifact_top_product",
            "ticket_detail",
            {"spu_id": "spu-1", "ticket_count": 9},
        ),
        (
            "query_artifact_refund",
            "refund_detail",
            {"refund_amount": 88.5},
        ),
    ]:
        artifact_contract = GroundedQueryContract(
            question=session.question,
            status="READY",
            query_shape="SCALAR",
        )
        session.verified_query_ledger.append(
            GroundedVerifiedQueryArtifact(
                artifact_id=artifact_id,
                generation=len(session.verified_query_ledger) + 1,
                contract_fingerprint=grounded_query_contract_fingerprint(
                    artifact_contract
                ),
                sql_fingerprint="sql-%s" % artifact_id,
                contract=artifact_contract,
                plan=QueryPlan(
                    intents=[
                        QuestionIntent(
                            question=session.question,
                            plan_task_id="same_task_id",
                        )
                    ],
                    evidence_contracts=[
                        {
                            "taskId": "same_task_id",
                            "table": table,
                            "semanticLabel": next(iter(row)),
                            "requiredLevel": "required",
                            "columns": [next(iter(row))],
                        }
                    ],
                    final_evidence_column_hints={
                        "same_task_id": list(row)
                    },
                ),
                run_result=AgentRunResult(
                    merged_query_bundle=QueryBundle(
                        tables=[table],
                        rows=[row],
                    )
                ),
                verified_evidence=VerifiedEvidence(
                    passed=True,
                    covered_evidence=list(row),
                ),
                output_columns=list(row),
            )
        )

    plan, run_result, verified, artifact_ids = runtime.verified_portfolio(
        session
    )

    assert artifact_ids == [
        "query_artifact_top_product",
        "query_artifact_refund",
    ]
    assert verified.passed is True
    assert run_result.merged_query_bundle.tables == [
        "ticket_detail",
        "refund_detail",
    ]
    assert {
        row["__evidenceArtifactId"]
        for row in run_result.merged_query_bundle.rows
    } == set(artifact_ids)
    assert len({item.plan_task_id for item in plan.intents}) == 2
    assert {
        str(item.get("taskId") or "")
        for item in plan.evidence_contracts
    } == {item.plan_task_id for item in plan.intents}

    composer = PortfolioComposer()
    runtime.answer_composer = composer
    runtime.verifier = FakeVerifier()
    session.active_generation = 2
    session.active_contract = session.verified_query_ledger[-1].contract.model_copy(
        deep=True
    )
    session.active_plan = session.verified_query_ledger[-1].plan.model_copy(
        deep=True
    )
    session.active_pack = PlanningAssetPack()

    answer = runtime.compose_answer(session)

    assert answer == "portfolio-answer"
    assert composer.run_result is not None
    assert len(composer.run_result.merged_query_bundle.rows) == 2
    assert session.answer_artifact_ids == artifact_ids


def test_artifact_namespacing_preserves_immutable_entity_filter_proof_hash() -> None:
    reference = EntityReference(
        field="spu_id",
        table="dwm_goods_detail_df",
        values=["1"],
        comparison_policy="exact",
        status="bound",
    )
    contract = NodePlanContract(
        task_id="goods_lookup",
        preferred_table="dwm_goods_detail_df",
        filter_column="spu_id",
        filter_values=["1"],
        entity_filter_obligations=[
            EntityFilterObligation(
                obligation_id="goods_lookup_entity_1",
                task_id="goods_lookup",
                reference=reference,
                status="bound",
            )
        ],
    )
    old_hash = entity_filter_contract_hash(contract)
    policy = entity_comparison_policy(reference)
    requested = canonical_entity_values(contract.filter_values, policy)
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="goods_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[
                        {
                            "spu_id": "1",
                            "goods_publish_time": "2026-04-06 15:54:00",
                        }
                    ],
                    sql=(
                        "SELECT spu_id, spu_apply_create_time AS goods_publish_time "
                        "FROM dwm_goods_detail_df"
                    ),
                ),
                node_plan_contract=contract,
                entity_filter_verification=EntityFilterVerificationProof(
                    task_id="goods_lookup",
                    obligation_id="goods_lookup_entity_1",
                    field="spu_id",
                    comparison_policy=policy,
                    contract_hash=old_hash,
                    requested_value_hashes=sorted(
                        entity_value_hash(value, old_hash)
                        for value in requested
                    ),
                    verified=True,
                    coverage_complete=True,
                    status="verified",
                ),
            )
        ]
    )

    namespaced = _namespace_artifact_run(run, "artifact_goods")
    task_result = namespaced.task_results[0]
    proof = task_result.entity_filter_verification
    new_hash = entity_filter_contract_hash(task_result.node_plan_contract)

    assert task_result.task_id == "artifact_goods::goods_lookup"
    assert proof.task_id == "artifact_goods::goods_lookup"
    assert new_hash == old_hash
    assert proof.contract_hash == new_hash
    assert proof.requested_value_hashes == sorted(
        entity_value_hash(value, new_hash)
        for value in requested
    )


def test_artifact_namespacing_accepts_nested_in_filter_values() -> None:
    reference = EntityReference(
        field="spu_id",
        table="dwm_goods_detail_df",
        values=["1"],
        comparison_policy="exact",
        status="bound",
    )
    contract = NodePlanContract(
        task_id="goods_lookup",
        preferred_table="dwm_goods_detail_df",
        filter_column="",
        filter_values=[["1"]],
        entity_filter_obligations=[
            EntityFilterObligation(
                obligation_id="goods_lookup_entity_1",
                task_id="goods_lookup",
                reference=reference,
                status="bound",
            )
        ],
    )
    contract_hash = entity_filter_contract_hash(contract)
    requested = canonical_entity_values(reference.values, "exact")
    run = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="goods_lookup",
                success=True,
                query_bundle=QueryBundle(
                    tables=["dwm_goods_detail_df"],
                    rows=[{"spu_apply_create_time": "2026-04-06 15:54:00"}],
                    sql="SELECT spu_apply_create_time FROM dwm_goods_detail_df",
                ),
                node_plan_contract=contract,
                entity_filter_verification=EntityFilterVerificationProof(
                    task_id="goods_lookup",
                    obligation_id="goods_lookup_entity_1",
                    comparison_policy="exact",
                    contract_hash=contract_hash,
                    sql_hash=entity_filter_sql_hash(
                        "SELECT spu_apply_create_time FROM dwm_goods_detail_df"
                    ),
                    requested_value_hashes=sorted(
                        entity_value_hash(value, contract_hash)
                        for value in requested
                    ),
                    verified=True,
                    coverage_complete=True,
                    status="verified",
                ),
            )
        ]
    )

    namespaced = _namespace_artifact_run(run, "artifact_goods")
    task_result = namespaced.task_results[0]

    assert task_result.entity_filter_verification.contract_hash == (
        entity_filter_contract_hash(task_result.node_plan_contract)
    )
    assert canonical_entity_values(
        task_result.node_plan_contract.filter_values,
        "exact",
    ) == {"1"}
