from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    ClarificationRequest,
    ExtractedKeywords,
    MerchantInfo,
    PlanningAssetEntry,
    PlanningAssetPack,
    RecallBundle,
    ResolvedTimeRange,
    TopicRoutingDecision,
)
from merchant_ai.services.grounded_deep_agent_runtime import (
    GroundedDeepAgentRunContext,
    GroundedDeepAgentRuntime,
    GroundedDeepAgentSession,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedTableBinding,
)
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeKernel
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class _QueueBuilder:
    def __init__(self, contract: GroundedQueryContract):
        self.contract = contract

    def build(
        self,
        question: str,
        topics: list[str],
        evidence: list[dict[str, Any]],
        **_: Any,
    ) -> GroundedQueryContract:
        assert question == self.contract.question
        return self.contract.model_copy(deep=True)


class _NoTemplateCompiler:
    def __call__(
        self,
        contract: GroundedQueryContract,
        pack: PlanningAssetPack,
    ) -> Any:
        raise AssertionError("complex Core SQL must not use template compilation")


class _ExplodingValidator:
    @staticmethod
    def validate(candidate: Any, contract: GroundedQueryContract) -> Any:
        raise ValueError("validator exploded")


class _ManifestCatalog:
    @staticmethod
    def read(*, path: str, max_chars: int, offset: int) -> dict[str, Any]:
        if path == "topics/orders/manifest.json":
            return {
                "success": True,
                "refId": "semantic:orders:manifest",
                "path": path,
                "kind": "TOPIC_MANIFEST",
                "topic": "orders",
                "content": '{"topic":"orders","tables":[{"tableName":"fact_orders"}]}',
            }
        return {"success": False, "error": "SEMANTIC_REF_NOT_FOUND"}


class _IdleGraph:
    @staticmethod
    def invoke(payload: dict[str, Any], *, config: Any, context: Any) -> None:
        return None


class _CapturingFactory:
    def __init__(self):
        self.tools: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> _IdleGraph:
        self.tools = {item.name: item for item in kwargs["tools"]}
        return _IdleGraph()


class _ClarifyingGraph:
    def __init__(self, tools: list[Any]):
        self.tools = {item.name: item for item in tools}
        self.invocations: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any], *, config: Any, context: Any) -> None:
        self.invocations.append(payload)
        self.tools["ask_human"].func(
            question="请问要查询最近多少天？",
            stage="time_binding",
            clarification_type="TIME_RANGE_REQUIRED",
            options=["最近7天", "最近30天"],
            runtime=SimpleNamespace(context=context),
        )


class _ClarifyingFactory:
    def __init__(self):
        self.graph: _ClarifyingGraph | None = None

    def __call__(self, **kwargs: Any) -> _ClarifyingGraph:
        self.graph = _ClarifyingGraph(kwargs["tools"])
        return self.graph


class _RecallFailureKernel:
    def new_session(
        self,
        question: str,
        merchant_id: str,
        **kwargs: Any,
    ) -> Any:
        from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession

        return GroundedRuntimeSession(
            session_id="recall-failure",
            question=question,
            merchant_id=merchant_id,
            merchant=kwargs.get("merchant") or MerchantInfo(merchant_id=merchant_id),
        )

    @staticmethod
    def route_topic(session: Any) -> TopicRoutingDecision:
        session.keywords = ExtractedKeywords(keywords=["订单"])
        session.routing = TopicRoutingDecision(
            primary_topic="orders",
            candidate_topics=["orders"],
            routing_mode="seed_topic",
        )
        session.workspace_topics = ["orders"]
        return session.routing

    @staticmethod
    def recall_navigation(session: Any, **_: Any) -> RecallBundle:
        raise ConnectionError("ES unavailable")

    @staticmethod
    def request_clarification(
        session: Any,
        question: str,
        **kwargs: Any,
    ) -> ClarificationRequest:
        request = ClarificationRequest(
            question=question,
            stage=kwargs["stage"],
            type=kwargs["clarification_type"],
            options=kwargs.get("options") or [],
            pending_question=session.question,
        )
        session.clarification = request
        return request


def _grouped_contract() -> GroundedQueryContract:
    detail_ref = "semantic:orders:fact_orders:detail"
    metric_ref = "semantic:orders:fact_orders:metric:total_amount"
    dimension_ref = "semantic:orders:fact_orders:field:buyer_id"
    return GroundedQueryContract(
        status="READY",
        question="2026年6月按买家统计下单金额",
        topics=["orders"],
        query_shape="GROUPED",
        primary_table="fact_orders",
        tables=[
            GroundedTableBinding(
                topic="orders",
                table="fact_orders",
                time_column="event_date",
                merchant_filter_column="tenant_id",
                detail_ref_id=detail_ref,
            )
        ],
        metrics=[
            GroundedMetricBinding(
                requested_phrase="下单金额",
                semantic_ref_id=metric_ref,
                topic="orders",
                table="fact_orders",
                metric_key="total_amount",
                formula="SUM(amount)",
                source_columns=["amount"],
                time_column="event_date",
                binding_type="published_metric",
            )
        ],
        dimensions=[
            GroundedDimensionBinding(
                requested_phrase="买家",
                semantic_ref_id=dimension_ref,
                topic="orders",
                table="fact_orders",
                column="buyer_id",
                usage="group_by",
            )
        ],
        time_range=ResolvedTimeRange(
            explicit=True,
            start_date="2026-06-01",
            end_date="2026-06-30",
            days=30,
            window_role="primary",
        ),
        evidence_refs=[detail_ref, metric_ref, dimension_ref],
    )


def _grouped_pack() -> PlanningAssetPack:
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="fact_orders",
                table="fact_orders",
                topic="orders",
                columns=["tenant_id", "event_date", "buyer_id", "amount"],
            )
        ]
    )


def _activated_kernel(
    *,
    validator: Any | None = None,
) -> tuple[GroundedRuntimeKernel, Any, Any]:
    contract = _grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=_QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: _grouped_pack(),
        compiler=_NoTemplateCompiler(),
        sql_candidate_validator=validator,
    )
    session = kernel.new_session(contract.question, "merchant-1")
    session.workspace_topics = ["orders"]
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    return kernel, session, activated


def _valid_grouped_sql() -> str:
    return """
        SELECT o.buyer_id, SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
    """


def _submit_tool(
    kernel: GroundedRuntimeKernel,
    session: Any,
) -> tuple[Any, GroundedDeepAgentRunContext]:
    factory = _CapturingFactory()
    GroundedDeepAgentRuntime(
        kernel,
        lead_model=object(),
        semantic_catalog=_ManifestCatalog(),
        agent_factory=factory,
    )
    context = GroundedDeepAgentRunContext(
        thread_id="resilience-thread",
        run_id="resilience-run",
        session=GroundedDeepAgentSession(runtime=session),
    )
    return factory.tools["submit_grounded_sql_candidate"], context


def test_sql_parse_error_returns_a_typed_tool_result_instead_of_raising() -> None:
    kernel, session, activated = _activated_kernel()
    tool, context = _submit_tool(kernel, session)

    payload = json.loads(
        tool.func(
            sql="SELECT (",
            expected_generation=activated.active_generation,
            contract_fingerprint=grounded_query_contract_fingerprint(
                activated.contract
            ),
            rationale="验证 SQL parse 错误边界",
            evidence_ref_ids=[],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert payload["status"] == "REJECTED"
    assert payload["nextAction"] == "REPAIR_SQL"
    assert any(gap["code"] == "SQL_PARSE_ERROR" for gap in payload["gaps"])
    assert session.active_preparation is None


def test_submit_validator_exception_is_returned_as_a_structured_internal_block() -> None:
    kernel, session, activated = _activated_kernel(validator=_ExplodingValidator())
    tool, context = _submit_tool(kernel, session)

    payload = json.loads(
        tool.func(
            sql=_valid_grouped_sql(),
            expected_generation=activated.active_generation,
            contract_fingerprint=grounded_query_contract_fingerprint(
                activated.contract
            ),
            rationale="验证 validator 异常边界",
            evidence_ref_ids=[],
            runtime=SimpleNamespace(context=context),
        )
    )

    assert payload["status"] == "BLOCKED"
    assert payload["code"]
    assert payload["nextAction"] == "STOP_INTERNAL"
    assert session.active_preparation is None


def test_initial_recall_failure_degrades_to_empty_navigation_and_keeps_core_alive() -> None:
    factory = _ClarifyingFactory()
    runtime = GroundedDeepAgentRuntime(
        _RecallFailureKernel(),
        lead_model=object(),
        semantic_catalog=_ManifestCatalog(),
        agent_factory=factory,
    )

    response = runtime.run("查询订单量", "merchant-1")

    assert response.clarification is not None
    assert factory.graph is not None
    initial = json.loads(factory.graph.invocations[0]["messages"][0]["content"])
    assert initial["thinRecallCandidates"] == []


def test_repeated_accepted_sql_keeps_the_original_executable_candidate() -> None:
    kernel, session, activated = _activated_kernel()
    fingerprint = grounded_query_contract_fingerprint(activated.contract)

    first = kernel.submit_sql_candidate(
        session,
        _valid_grouped_sql(),
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )
    active_preparation = session.active_preparation
    second = kernel.submit_sql_candidate(
        session,
        _valid_grouped_sql(),
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    assert first.status == "ACCEPTED"
    assert second.ast_fingerprint == first.ast_fingerprint
    assert session.active_preparation is active_preparation
    assert session.active_preparation is not None
    assert session.active_sql_candidate is not None
    assert session.active_sql_validation is not None
    assert session.active_sql_validation.valid is True
