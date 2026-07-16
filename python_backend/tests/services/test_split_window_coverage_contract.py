from __future__ import annotations

from dataclasses import replace
from threading import Lock

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AgentRunResult,
    AnswerMode,
    NodeExecutionContext,
    PlanningAssetEntry,
    PlanningAssetPack,
    QueryPlan,
    QuestionIntent,
)
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.query import (
    NodeWorkerExecutor,
    SqlValidationService,
    merge_task_result_bundles,
    split_window_coverage_gaps,
)
from merchant_ai.services.query_sql_binding import (
    build_split_detail_sql_plan,
    split_detail_sql_chunk_contract_error,
    split_window_coverage_contract,
)


class UnusedLlm:
    configured = False
    last_error = ""


class SplitFallbackDoris:
    def __init__(self, failed_chunk_interval: int | None = None):
        self.failed_chunk_interval = failed_chunk_interval
        self.original_failed = False
        self.sqls: list[str] = []
        self.lock = Lock()

    def query(self, sql, params=None):
        del params
        with self.lock:
            self.sqls.append(sql)
        if "SELECT MIN(`pt`)" in sql and "MAX(`pt`)" in sql:
            return [{"min_value": "2026-05-01", "max_value": "2026-06-20"}]
        with self.lock:
            if not self.original_failed:
                self.original_failed = True
                raise RuntimeError("MEM_ALLOC_FAILED: original detail query exceeded memory")
        failure_marker = "`pt` >= DATE_SUB('2026-06-20', INTERVAL '%d' DAY)" % int(self.failed_chunk_interval or 0)
        if self.failed_chunk_interval is not None and failure_marker in sql:
            raise RuntimeError("MEM_ALLOC_FAILED: injected split chunk failure")
        return [
            {
                "seller_id": "100",
                "pt": "2026-06-20",
                "order_id": "order_%d" % len(self.sqls),
                "sub_order_id": "sub_%d" % len(self.sqls),
                "spu_name": "商品",
            }
        ]


def detail_asset_pack() -> PlanningAssetPack:
    table = "dwm_trade_order_detail_di"
    columns = ["seller_id", "pt", "order_id", "sub_order_id", "spu_name"]
    semantic = {
        "visibilityPolicy": {"level": "public", "allowedRoles": []},
        "maskingPolicy": {"strategy": "none"},
        "defaultVisible": True,
        "displayScenarios": ["detail"],
    }
    return PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                table=table,
                columns=columns,
                metadata={"timeColumn": "pt", "merchantFilterColumn": "seller_id"},
            )
        ],
        fields=[
            PlanningAssetEntry(
                key=column,
                table=table,
                columns=[column],
                metadata={"semantic": semantic},
            )
            for column in columns
        ],
    )


def detail_intent(days: int) -> QuestionIntent:
    return QuestionIntent(
        question="最近%d天订单明细" % days,
        intent_type="VALID",
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="detail_orders",
        preferred_table="dwm_trade_order_detail_di",
        sql_strategy="structured_first",
        output_keys=["order_id", "sub_order_id"],
        required_evidence=["spu_name", "pt"],
        days=days,
        limit=10,
    )


def split_worker(doris: SplitFallbackDoris, *, max_chunks: int = 3) -> NodeWorkerExecutor:
    settings = get_settings().model_copy(
        update={
            "agent_sql_repair_rounds": 1,
            "agent_partition_date_anchor_enabled": True,
            "agent_doris_split_query_enabled": True,
            "agent_doris_split_chunk_days": 7,
            "agent_doris_split_max_chunks": max_chunks,
            "agent_doris_split_max_concurrency": 3,
        }
    )
    return NodeWorkerExecutor(UnusedLlm(), doris, SqlValidationService(), settings)


def finished_coverage(result) -> dict:
    finished = next(
        event
        for event in reversed(result.query_bundle.runtime_events)
        if event.get("event") == "split_query_fallback_finished"
    )
    return finished["coverage"]


def test_split_plan_preserves_full_window_obligation_when_max_chunks_truncates():
    plan = build_split_detail_sql_plan(
        "SELECT `order_id`, `pt` FROM `orders` WHERE `seller_id` = %s LIMIT 10",
        days=30,
        chunk_days=7,
        max_chunks=3,
        limit=10,
        time_column="pt",
        anchor_date="2026-06-20",
    )

    assert plan is not None
    coverage = split_window_coverage_contract(
        plan,
        [{"chunkIndex": chunk.index, "status": "succeeded", "rows": 1} for chunk in plan.chunks],
    )

    assert coverage["requestedWindow"] == {
        "startDate": "2026-05-22",
        "endDate": "2026-06-20",
        "days": 30,
        "anchorDate": "2026-06-20",
    }
    assert coverage["requiredChunkCount"] == 5
    assert coverage["plannedChunkCount"] == 3
    assert coverage["executedChunkCount"] == 3
    assert coverage["omittedChunkCount"] == 2
    assert coverage["truncated"] is True
    assert coverage["complete"] is False
    assert coverage["code"] == "SPLIT_WINDOW_COVERAGE_INCOMPLETE"


def test_split_chunk_contract_rejects_sql_not_derived_by_root_and_restriction():
    base_sql = "SELECT `order_id`, `pt` FROM `orders` WHERE `seller_id` = %s LIMIT 10"
    plan = build_split_detail_sql_plan(
        base_sql,
        days=21,
        chunk_days=7,
        max_chunks=3,
        limit=10,
        time_column="pt",
        anchor_date="2026-06-20",
    )

    assert plan is not None
    valid_code, _ = split_detail_sql_chunk_contract_error(base_sql, plan, plan.chunks[0], "pt", 10)
    tampered_chunk = replace(
        plan.chunks[0],
        sql=plan.chunks[0].sql.replace(" LIMIT 10", " AND 1 = 0 LIMIT 10"),
    )
    tampered_plan = replace(plan, chunks=(tampered_chunk, *plan.chunks[1:]))
    invalid_code, _ = split_detail_sql_chunk_contract_error(
        base_sql,
        tampered_plan,
        tampered_chunk,
        "pt",
        10,
    )

    assert valid_code == ""
    assert invalid_code == "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID"


def test_split_chunk_contract_rejects_window_metadata_not_matching_full_plan():
    base_sql = "SELECT `order_id`, `pt` FROM `orders` WHERE `seller_id` = %s LIMIT 10"
    plan = build_split_detail_sql_plan(
        base_sql,
        days=21,
        chunk_days=7,
        max_chunks=3,
        limit=10,
        time_column="pt",
        anchor_date="2026-06-20",
    )

    assert plan is not None
    tampered_chunk = replace(plan.chunks[1], start_date="2026-06-08")
    tampered_plan = replace(
        plan,
        chunks=(plan.chunks[0], tampered_chunk, plan.chunks[2]),
    )
    invalid_code, _ = split_detail_sql_chunk_contract_error(
        base_sql,
        tampered_plan,
        tampered_chunk,
        "pt",
        10,
    )

    assert invalid_code == "SPLIT_WINDOW_PLAN_INVALID"


def test_max_chunks_truncation_retains_partial_rows_but_fails_closed():
    intent = detail_intent(30)
    result = split_worker(SplitFallbackDoris(), max_chunks=3).execute_node(
        intent,
        detail_asset_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    coverage = finished_coverage(result)
    assert result.success is False
    assert result.query_bundle.failed is True
    assert result.query_bundle.lineage_complete is False
    assert result.query_bundle.rows
    assert "SPLIT_WINDOW_COVERAGE_INCOMPLETE" in result.query_bundle.error
    assert coverage["requiredChunkCount"] == 5
    assert coverage["plannedChunkCount"] == 3
    assert coverage["succeededChunkCount"] == 3
    assert coverage["omittedChunkCount"] == 2
    assert coverage["truncated"] is True
    assert coverage["complete"] is False


def test_one_failed_chunk_blocks_verification_even_when_two_chunks_return_rows():
    intent = detail_intent(21)
    result = split_worker(SplitFallbackDoris(failed_chunk_interval=13)).execute_node(
        intent,
        detail_asset_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    coverage = finished_coverage(result)
    assert result.success is False
    assert result.query_bundle.failed is True
    assert result.query_bundle.lineage_complete is False
    assert len(result.query_bundle.rows) == 2
    assert coverage["requiredChunkCount"] == 3
    assert coverage["executedChunkCount"] == 3
    assert coverage["succeededChunkCount"] == 2
    assert coverage["failedChunkCount"] == 1
    assert coverage["failedChunks"][0]["chunkIndex"] == 2
    assert coverage["complete"] is False

    gaps = split_window_coverage_gaps([result])
    merged = merge_task_result_bundles([result])
    run_result = AgentRunResult(
        task_results=[result],
        query_bundles=[result.query_bundle],
        merged_query_bundle=merged,
        evidence_gaps=gaps,
    )
    verified = EvidenceVerifier().verify(intent.question, QueryPlan(intents=[intent]), run_result)

    assert merged.lineage_complete is False
    assert verified.passed is False
    assert any(gap.code == "SPLIT_WINDOW_COVERAGE_INCOMPLETE" for gap in verified.blocking_gaps)


def test_all_required_chunks_succeed_only_then_coverage_is_complete():
    result = split_worker(SplitFallbackDoris(), max_chunks=3).execute_node(
        detail_intent(21),
        detail_asset_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    coverage = finished_coverage(result)
    assert result.success is True
    assert result.query_bundle.failed is False
    assert result.query_bundle.lineage_complete is True
    assert coverage["requiredChunkCount"] == 3
    assert coverage["executedChunkCount"] == 3
    assert coverage["failedChunkCount"] == 0
    assert coverage["truncated"] is False
    assert coverage["complete"] is True


def test_execute_plan_lifts_incomplete_split_coverage_into_blocking_evidence():
    intent = detail_intent(30)
    plan = QueryPlan(intents=[intent])
    run_result = split_worker(SplitFallbackDoris(), max_chunks=3).execute_plan(
        "100",
        plan,
        detail_asset_pack(),
        "",
        intent.question,
        execution_mode="direct",
    )

    verified = EvidenceVerifier().verify(intent.question, plan, run_result)

    assert run_result.merged_query_bundle.failed is True
    assert run_result.merged_query_bundle.lineage_complete is False
    assert any(gap.code == "SPLIT_WINDOW_COVERAGE_INCOMPLETE" for gap in run_result.evidence_gaps)
    assert verified.passed is False
    assert any(gap.code == "SPLIT_WINDOW_COVERAGE_INCOMPLETE" for gap in verified.blocking_gaps)
