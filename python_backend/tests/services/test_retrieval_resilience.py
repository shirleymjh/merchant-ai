from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from merchant_ai.models import (
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
    RecallBundle,
    RecallItem,
)
from merchant_ai.services.retrieval import (
    ResilientKnowledgeRetrievalService,
)
from merchant_ai.services.tool_runtime import ToolFailureRegistry


class _Retrieval:
    def __init__(
        self,
        backend_name: str,
        *,
        result: KnowledgeBundle | None = None,
        error: Exception | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.result = result or KnowledgeBundle(backend=backend_name)
        self.error = error
        self.calls = 0

    def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeBundle:
        del request
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    def cache_trace(self) -> dict[str, Any]:
        return {"calls": self.calls}


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        es_base_url="http://es:9200",
        es_index="merchant_ai_recall",
        tool_failure_repeat_threshold=10,
        tool_circuit_threshold=2,
        tool_circuit_cooldown_seconds=60,
    )


def test_es_operational_failure_uses_local_hybrid_evidence() -> None:
    primary = _Retrieval(
        "es",
        error=TimeoutError("ES timed out"),
    )
    fallback = _Retrieval(
        "hybrid",
        result=KnowledgeBundle(
            backend="hybrid",
            recall_bundle=RecallBundle(
                items=[
                    RecallItem(
                        doc_id="semantic:orders:metric:order_count",
                        title="订单量",
                        content="订单量指标定义",
                        source_type="SEMANTIC_METRIC",
                        fusion_score=9,
                    )
                ]
            ),
            retrieval_status="success",
        ),
    )
    service = ResilientKnowledgeRetrievalService(
        primary,
        fallback,
        settings=_settings(),
    )

    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="最近7天订单量",
            merchant_id="merchant-1",
        )
    )

    assert primary.calls == 1
    assert fallback.calls == 1
    assert bundle.retrieval_status == "degraded"
    assert bundle.source_refs == [
        "semantic:orders:metric:order_count"
    ]
    assert bundle.retrieval_issues[0].fallback_used is True
    assert bundle.retrieval_issues[0].resolved is True


def test_es_circuit_skips_primary_until_registry_cooldown() -> None:
    primary = _Retrieval(
        "es",
        error=ConnectionError("ES unavailable"),
    )
    fallback = _Retrieval(
        "hybrid",
        result=KnowledgeBundle(
            backend="hybrid",
            retrieval_status="empty",
        ),
    )
    registry = ToolFailureRegistry(
        repeat_threshold=10,
        circuit_threshold=2,
        cooldown_seconds=60,
    )
    service = ResilientKnowledgeRetrievalService(
        primary,
        fallback,
        settings=_settings(),
        failure_registry=registry,
    )
    request = KnowledgeRetrievalRequest(
        query="订单量",
        merchant_id="merchant-1",
    )

    service.retrieve(request)
    service.retrieve(request)
    third = service.retrieve(request)

    assert primary.calls == 2
    assert fallback.calls == 3
    assert third.retrieval_issues[0].code == (
        "ES_RETRIEVAL_CIRCUIT_OPEN"
    )
    assert registry.trace()["circuits"][0]["open"] is True


def test_empty_es_result_is_valid_and_does_not_trigger_fallback() -> None:
    primary = _Retrieval(
        "es",
        result=KnowledgeBundle(
            backend="es",
            retrieval_status="empty",
        ),
    )
    fallback = _Retrieval("hybrid")
    service = ResilientKnowledgeRetrievalService(
        primary,
        fallback,
        settings=_settings(),
    )

    bundle = service.retrieve(
        KnowledgeRetrievalRequest(
            query="不存在的指标",
            merchant_id="merchant-1",
        )
    )

    assert bundle.retrieval_status == "empty"
    assert primary.calls == 1
    assert fallback.calls == 0
