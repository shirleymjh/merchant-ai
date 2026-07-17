from __future__ import annotations

from merchant_ai.config import get_settings
from merchant_ai.models import ResolvedTimeRange
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_query_contract import (
    GroundedEntityFilterBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
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


class FakeDetailDoris(FakeDoris):
    def query(self, sql: str, *, timeout_seconds: int) -> list[dict[str, object]]:
        self.calls.append((sql, timeout_seconds))
        return [
            {
                "entity_id": "entity_100",
                "related_id": "related_9",
                "detail_status": "completed",
                "published_at": "2026-01-05 10:30:00",
            }
        ]


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


def detail_lookup_contract() -> GroundedQueryContract:
    primary = "fact_entity_detail"
    related = "dim_related_entity"
    field_refs = {
        "entity_id": "semantic:domain:%s:field:entity_id" % primary,
        "related_id": "semantic:domain:%s:field:related_id" % primary,
        "detail_status": "semantic:domain:%s:field:detail_status" % primary,
        "published_at": "semantic:related:%s:field:published_at" % related,
    }
    relationship_ref = "semantic:domain:relationships"
    return GroundedQueryContract(
        status="READY",
        question="查询实体 entity_100 的明细，再看关联对象什么时候发布",
        topics=["domain", "related"],
        query_shape="ENTITY_LOOKUP",
        execution_shape="detail_join",
        primary_table=primary,
        tables=[
            GroundedTableBinding(
                topic="domain",
                table=primary,
                data_grain="entity_detail",
                time_column="pt",
                merchant_filter_column="seller_id",
                detail_ref_id="semantic:domain:%s:detail" % primary,
            ),
            GroundedTableBinding(
                topic="related",
                table=related,
                data_grain="related_entity",
                time_column="pt",
                merchant_filter_column="seller_id",
                detail_ref_id="semantic:related:%s:detail" % related,
            ),
        ],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id=field_refs["entity_id"],
                topic="domain",
                table=primary,
                column="entity_id",
                output_alias="entity_id",
                is_unique_key=True,
                entity_identity="PRIMARY_ENTITY",
                filter_operators=["EQ"],
                lookup_time_policy={"mode": "global"},
            ),
            GroundedSelectedFieldBinding(
                semantic_ref_id=field_refs["related_id"],
                topic="domain",
                table=primary,
                column="related_id",
                output_alias="related_id",
            ),
            GroundedSelectedFieldBinding(
                semantic_ref_id=field_refs["detail_status"],
                topic="domain",
                table=primary,
                column="detail_status",
                output_alias="detail_status",
            ),
            GroundedSelectedFieldBinding(
                semantic_ref_id=field_refs["published_at"],
                topic="related",
                table=related,
                column="published_at",
                output_alias="published_at",
            ),
        ],
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id=field_refs["entity_id"],
                topic="domain",
                table=primary,
                column="entity_id",
                operator="EQ",
                literal_value="entity_100",
                is_unique_key=True,
                entity_identity="PRIMARY_ENTITY",
                allowed_operators=["EQ"],
                lookup_time_policy={"mode": "global"},
            )
        ],
        relationships=[
            GroundedRelationshipBinding(
                semantic_ref_id=relationship_ref,
                topic="domain",
                name="primary_to_related",
                left_table=primary,
                right_table=related,
                join_type="LEFT",
                keys=[["seller_id", "seller_id"], ["related_id", "related_id"]],
                grain="primary_entity_related_entity",
                cardinality="MANY_TO_ONE",
                fanout_policy="PRESERVE_LEFT_GRAIN",
            )
        ],
        evidence_refs=[
            "semantic:domain:%s:detail" % primary,
            "semantic:related:%s:detail" % related,
            *field_refs.values(),
            relationship_ref,
        ],
        time_range=ResolvedTimeRange(
            source="default_days",
            days=7,
            explicit=False,
        ),
    )


def test_detail_executor_compiles_typed_entity_filter_and_governed_join(tmp_path) -> None:
    settings = get_settings()
    contract = detail_lookup_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDetailDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=AccessControlService(settings, root=tmp_path),
    )

    result = executor.execute_contract(
        "merchant_1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-detail",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    assert len(repository.calls) == 1
    sql = repository.calls[0][0]
    assert "COUNT(" not in sql and "SUM(" not in sql
    assert "FROM `fact_entity_detail` t0" in sql
    assert "LEFT JOIN `dim_related_entity` t1" in sql
    assert "t0.`related_id` = t1.`related_id`" in sql
    assert "t0.`entity_id` = 'entity_100'" in sql
    assert "t0.`seller_id` = 'merchant_1'" in sql
    assert "t1.`seller_id` = 'merchant_1'" in sql
    assert "BETWEEN" not in sql
    assert result.merged_query_bundle.tables == [
        "fact_entity_detail",
        "dim_related_entity",
    ]
    assert result.task_results[0].entity_filter_verification.verified is True
    verified = EvidenceVerifier().verify(contract.question, preparation.plan, result)
    assert verified.passed, [gap.model_dump() for gap in verified.blocking_gaps]
