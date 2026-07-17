from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_query_contract import (
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedTableBinding,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.grounded_query_executor import GroundedQueryExecutionKernel


class FakeDoris:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.last_cache_hit = False
        self.last_cache_key = ""

    def query(self, sql: str, *, timeout_seconds: int) -> list[dict[str, object]]:
        self.calls.append((sql, timeout_seconds))
        return [{"order_cnt_1d": 129, "refund_amt_1d": 4437.15}]


def scalar_contract() -> GroundedQueryContract:
    topic = "经营画像"
    table = "ads_merchant_profile"
    return GroundedQueryContract(
        status="READY",
        question="最近30天的订单数和退款总额是多少？",
        topics=[topic],
        query_shape="SCALAR",
        execution_shape="same_table_multi_metric",
        primary_table=table,
        tables=[
            GroundedTableBinding(
                topic=topic,
                table=table,
                title="商家经营画像",
                data_grain="merchant_day_summary",
                time_column="pt",
                merchant_filter_column="merchant_id",
                detail_ref_id="semantic:经营画像:ads_merchant_profile:detail",
            )
        ],
        metrics=[
            GroundedMetricBinding(
                requested_phrase="订单数",
                semantic_ref_id="semantic:经营画像:ads_merchant_profile:metric:order_cnt_1d",
                topic=topic,
                table=table,
                metric_key="order_cnt_1d",
                business_name="总订单日汇总量",
                formula="SUM(order_cnt_1d)",
                source_columns=["order_cnt_1d"],
                aggregation_policy="period_rollup",
                metric_grain="merchant_day_summary",
                applicable_time_grain="period",
                time_column="pt",
                unit="单",
                anchor_policy="latest_available_partition",
                time_semantics={
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                },
            ),
            GroundedMetricBinding(
                requested_phrase="退款总额",
                semantic_ref_id="semantic:经营画像:ads_merchant_profile:metric:refund_amt_1d",
                topic=topic,
                table=table,
                metric_key="refund_amt_1d",
                business_name="退款日汇总金额",
                formula="SUM(refund_amt_1d)",
                source_columns=["refund_amt_1d"],
                aggregation_policy="period_rollup",
                metric_grain="merchant_day_summary",
                applicable_time_grain="period",
                time_column="pt",
                unit="元",
                anchor_policy="latest_available_partition",
                time_semantics={
                    "selectionPolicy": "period_window",
                    "asOfPolicy": "latest_available_partition",
                },
            ),
        ],
        time_range=ResolvedTimeRange(
            kind="rolling",
            start_date="2026-06-18",
            end_date="2026-07-17",
            days=30,
            label="最近30天",
            anchor_policy="latest_available_partition",
            explicit=True,
        ),
    )


def test_direct_executor_has_no_node_agent_and_preserves_metric_labels(tmp_path) -> None:
    settings = get_settings()
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=AccessControlService(settings, root=tmp_path),
    )

    result = executor.execute_contract(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-direct",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    assert len(repository.calls) == 1
    sql = repository.calls[0][0]
    assert "SUM(`order_cnt_1d`) AS `order_cnt_1d`" in sql
    assert "SUM(`refund_amt_1d`) AS `refund_amt_1d`" in sql
    assert "`merchant_id` = '99999999999999999999999999999999'" in sql
    assert "SELECT MAX(`pt`)" in sql
    assert result.task_results[0].sub_agent_type == "GROUNDED_DATA_ENGINE"
    assert result.task_results[0].node_task_profile.sql_draft_source == "grounded_deterministic"
    assert result.task_results[0].query_bundle.rows == [
        {
            "order_cnt_1d": 129,
            "refund_amt_1d": 4437.15,
            "__timeWindowRole": "primary",
        }
    ]
    specs = preparation.plan.intents[0].metric_specs
    assert [item["displayName"] for item in specs] == ["订单数", "退款总额"]

    verified = EvidenceVerifier().verify(contract.question, preparation.plan, result)
    assert verified.passed, [gap.model_dump() for gap in verified.blocking_gaps]


def test_runtime_factory_does_not_construct_node_worker() -> None:
    source = (
        __import__("pathlib").Path(
            "python_backend/merchant_ai/services/runtime_factory.py"
        ).read_text(encoding="utf-8")
    )

    assert "NodeWorkerExecutor" not in source
    assert "NodeAgent" not in source
    assert "PlanningAssetPackBuilder" not in source
