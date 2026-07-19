from __future__ import annotations

from types import SimpleNamespace

import pytest

from merchant_ai.services.grounded_query_branches import (
    GroundedBranchBudget,
    GroundedBranchBudgetExceeded,
    GroundedBranchBudgetLimits,
    GroundedQueryBranchContext,
    GroundedQueryBranchSpec,
    GroundedSemanticReadLedger,
)
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetLimits,
)


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_branch_budget_charges_each_internal_read_to_parent_budget() -> None:
    parent = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=30,
            max_llm_calls=3,
            max_tool_calls=10,
            max_doris_queries=3,
        )
    )
    branch = GroundedBranchBudget(
        "orders",
        GroundedBranchBudgetLimits(
            max_semantic_reads=2,
            max_semantic_chars=100,
            max_contract_attempts=2,
            max_doris_queries=1,
            max_duration_seconds=30,
            finalization_reserve_seconds=0,
        ),
        parent=parent,
    )

    branch.consume_semantic_read(path="orders/detail.json", content_chars=20)
    branch.consume_semantic_read(path="orders/metric.json", content_chars=30)

    assert branch.report()["usage"] == {
        "semanticReads": 2,
        "semanticChars": 50,
        "contractAttempts": 0,
        "dorisQueries": 0,
    }
    assert parent.report()["usage"]["toolCallsByName"] == {
        "branch.orders.semantic_read": 2
    }
    with pytest.raises(GroundedBranchBudgetExceeded) as raised:
        branch.consume_semantic_read(path="orders/column.json", content_chars=1)
    assert raised.value.code == "BRANCH_SEMANTIC_READ_LIMIT"
    assert parent.report()["usage"]["toolCalls"] == 2


def test_branch_budget_limits_use_grounded_runtime_settings() -> None:
    limits = GroundedBranchBudgetLimits.from_settings(
        SimpleNamespace(
            grounded_branch_max_semantic_reads=5,
            grounded_branch_max_semantic_chars=12_000,
            grounded_branch_max_contract_attempts=2,
            grounded_branch_max_doris_queries=1,
            grounded_branch_max_duration_seconds=22,
            grounded_finalization_reserve_seconds=7,
        )
    )

    assert limits.as_dict() == {
        "maxSemanticReads": 5,
        "maxSemanticChars": 12_000,
        "maxContractAttempts": 2,
        "maxDorisQueries": 1,
        "maxDurationSeconds": 22.0,
        "finalizationReserveSeconds": 7.0,
    }


def test_branch_budget_keeps_doris_local_and_global_limits_consistent() -> None:
    parent = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(
            max_duration_seconds=30,
            max_llm_calls=3,
            max_tool_calls=10,
            max_doris_queries=2,
        )
    )
    branch = GroundedBranchBudget(
        "refunds",
        GroundedBranchBudgetLimits(
            max_doris_queries=1,
            finalization_reserve_seconds=0,
        ),
        parent=parent,
    )

    branch.consume_doris_query()

    assert branch.report()["usage"]["dorisQueries"] == 1
    assert parent.report()["usage"]["dorisQueriesByName"] == {
        "parallel.refunds": 1
    }
    with pytest.raises(GroundedBranchBudgetExceeded) as raised:
        branch.consume_doris_query()
    assert raised.value.code == "BRANCH_DORIS_QUERY_LIMIT"
    assert parent.report()["usage"]["dorisQueries"] == 1


def test_semantic_ledgers_logically_isolate_the_same_or_different_refs() -> None:
    first = GroundedSemanticReadLedger()
    second = GroundedSemanticReadLedger()
    first.retain(
        {
            "refId": "semantic:trade:orders:detail",
            "path": "topics/trade/orders/detail.json",
            "contentSnippet": "{}",
        }
    )
    second.retain(
        {
            "refId": "semantic:refund:refunds:detail",
            "path": "topics/refund/refunds/detail.json",
            "contentSnippet": "{}",
        }
    )

    assert first.refs() == ["semantic:trade:orders:detail"]
    assert second.refs() == ["semantic:refund:refunds:detail"]
    assert first.evidence()[0]["refId"] not in second.refs()


def test_branch_context_report_exposes_local_scope_and_budget_only() -> None:
    context = GroundedQueryBranchContext(
        spec=GroundedQueryBranchSpec(
            query_id="orders",
            objective="订单明细",
            goal_ids=["detail.orders"],
            topic_scope=["电商交易"],
        ),
        runtime=None,
        budget=GroundedBranchBudget(
            "orders",
            GroundedBranchBudgetLimits(finalization_reserve_seconds=0),
        ),
        dependency_query_ids=["top-products"],
        status="WAITING_VERIFIED_ENTITY_SET",
    )

    report = context.report()

    assert report["queryId"] == "orders"
    assert report["topicScope"] == ["电商交易"]
    assert report["dependencyQueryIds"] == ["top-products"]
    assert report["status"] == "WAITING_VERIFIED_ENTITY_SET"


def test_branch_budget_charges_only_active_work_not_declaration_or_queue_wait() -> None:
    clock = _ManualClock()
    branch = GroundedBranchBudget(
        "orders",
        GroundedBranchBudgetLimits(
            max_duration_seconds=1,
            finalization_reserve_seconds=0,
        ),
        monotonic_clock=clock,
    )

    clock.advance(65)

    assert branch.report()["elapsedMs"] == 0
    assert branch.report()["wallElapsedMs"] == 0

    with branch.stage("semantic_retrieval"):
        clock.advance(0.04)

    clock.advance(65)

    assert branch.report()["elapsedMs"] == 40
    branch.consume_doris_query()

    report = branch.report()
    assert report["elapsedMs"] == 40
    assert report["wallElapsedMs"] == 65_040
    assert report["usage"]["dorisQueries"] == 1
