from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    PlanningAssetEntry,
    PlanningAssetPack,
    ResolvedTimeRange,
)
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
    GroundedUpstreamEntityBinding,
)
from merchant_ai.services.grounded_query_executor import GroundedQueryExecutionKernel
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeKernel
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class QueueBuilder:
    def __init__(self, contract: GroundedQueryContract):
        self.contract = contract

    def build(self, question: str, topics: list[str], evidence: list[dict[str, Any]], **_: Any) -> GroundedQueryContract:
        assert question == self.contract.question
        return self.contract.model_copy(deep=True)


def explicit_test_access_control(
    settings,
    root: Path,
    contract: GroundedQueryContract,
) -> AccessControlService:
    root.mkdir(parents=True, exist_ok=True)
    (root / "merchant_acl.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "defaultEffect": "DENY",
                "allowedMerchantIds": ["merchant-1"],
                "tables": {
                    table.table: {"allowedRoles": ["merchant_analyst"]}
                    for table in contract.tables
                },
            }
        ),
        encoding="utf-8",
    )
    return AccessControlService(settings, root=root)


class NoTemplateCompiler:
    def __call__(self, contract: GroundedQueryContract, pack: PlanningAssetPack) -> Any:
        raise AssertionError("complex Core SQL must never invoke the template compiler")


class FakeDoris:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.sql = ""
        self.last_cache_hit = False
        self.last_cache_key = ""

    def query(self, sql: str, **_: Any) -> list[dict[str, Any]]:
        self.sql = sql
        return [dict(item) for item in self.rows]


def grouped_contract() -> GroundedQueryContract:
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
                entity_identity="entity:buyer",
                filter_operators=["EQ", "IN"],
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


def grouped_pack() -> PlanningAssetPack:
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


def upstream_filtered_contract() -> GroundedQueryContract:
    detail_ref = "semantic:orders:fact_orders:detail"
    metric_ref = "semantic:orders:fact_orders:metric:total_amount"
    buyer_ref = "semantic:orders:fact_orders:field:buyer_id"
    return GroundedQueryContract(
        status="READY",
        question="查询已验证买家集合的下单金额",
        topics=["orders"],
        query_shape="SCALAR",
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
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id=buyer_ref,
                topic="orders",
                table="fact_orders",
                column="buyer_id",
                operator="IN",
                literal_value=["b-1", "b-2"],
                entity_identity="entity:buyer",
                allowed_operators=["IN"],
            )
        ],
        upstream_entity_bindings=[
            GroundedUpstreamEntityBinding(
                entity_set_artifact_id="entity_set_test",
                source_query_artifact_id="query_artifact_test",
                source_contract_fingerprint="source-contract",
                source_sql_fingerprint="source-sql",
                source_column="buyer_id",
                source_semantic_ref_id="semantic:source:buyers:field:buyer_id",
                source_entity_identity="entity:buyer",
                target_field_ref=buyer_ref,
                target_table="fact_orders",
                target_column="buyer_id",
                target_entity_identity="entity:buyer",
                operator="IN",
                value_count=2,
                values_hash="values-hash",
            )
        ],
        time_range=ResolvedTimeRange(
            explicit=True,
            start_date="2026-06-01",
            end_date="2026-06-30",
            days=30,
            window_role="primary",
        ),
        evidence_refs=[detail_ref, metric_ref, buyer_ref],
    )


def test_complex_contract_accepts_core_sql_and_executes_without_template(
    tmp_path: Path,
) -> None:
    contract = grouped_contract()
    pack = grouped_pack()
    doris = FakeDoris([{"buyer_id": "b-1", "total_amount": 12.5}])
    settings = get_settings()
    executor = GroundedQueryExecutionKernel(
        doris,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
        ),
    )
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: pack,
        compiler=NoTemplateCompiler(),
        executor=executor,
        verifier=EvidenceVerifier(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    session.workspace_topics = ["orders"]

    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    submitted = kernel.submit_sql_candidate(
        session,
        """
        SELECT o.buyer_id, SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=grounded_query_contract_fingerprint(
            activated.contract
        ),
    )
    result = kernel.execute_active(session)
    verified = kernel.verify_active(session)

    assert activated.execution_mode == GroundedExecutionMode.CORE_SQL_REQUIRED
    assert activated.compile_status == "NOT_APPLICABLE_CORE_SQL_REQUIRED"
    assert submitted.status == "ACCEPTED"
    assert submitted.next_action == "EXECUTE_GROUNDED_QUERY"
    assert session.active_plan is not None
    assert session.active_plan.intents[0].sql_strategy == "core_llm_grounded_sql"
    assert result.merged_query_bundle.rows == [
        {"buyer_id": "b-1", "total_amount": 12.5, "__timeWindowRole": "primary"}
    ]
    assert "tenant_id" in doris.sql
    assert "merchant-1" in doris.sql
    assert "GROUP BY o.buyer_id" in doris.sql
    assert verified.passed is True, [
        item.model_dump(by_alias=True) for item in verified.blocking_gaps
    ]
    artifact = kernel.latest_verified_query_artifact(session)
    assert artifact is not None
    assert artifact.output_semantic_refs["buyer_id"].endswith(":field:buyer_id")
    entity_set = kernel.publish_verified_entity_set(
        session,
        artifact.artifact_id,
        "buyer_id",
    )
    assert entity_set.values == ["b-1"]
    assert entity_set.source_entity_identity == "entity:buyer"
    assert entity_set.truncated is False
    with pytest.raises(RuntimeError) as exc_info:
        kernel.execute_active(session)
    assert "SQL_EXECUTION_NO_PROGRESS" in str(exc_info.value)


def test_same_invalid_sql_is_fused_as_no_progress() -> None:
    contract = grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    sql = """
        SELECT o.buyer_id, SUM(o.amount) AS wrong_alias
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
    """

    fingerprint = grounded_query_contract_fingerprint(activated.contract)
    first = kernel.submit_sql_candidate(
        session,
        sql,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )
    second = kernel.submit_sql_candidate(
        session,
        sql,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    assert first.status == "REJECTED"
    assert first.next_action == "REPAIR_SQL"
    assert second.status == "NO_PROGRESS"
    assert second.validation_gaps[0]["code"] == "SQL_CANDIDATE_NO_PROGRESS"


def test_upstream_entity_set_is_injected_without_disclosing_values_to_core_sql(
    tmp_path: Path,
) -> None:
    contract = upstream_filtered_contract()
    doris = FakeDoris([{"total_amount": 22.5}])
    settings = get_settings()
    executor = GroundedQueryExecutionKernel(
        doris,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
        ),
    )
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
        executor=executor,
        verifier=EvidenceVerifier(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    contract_fingerprint = grounded_query_contract_fingerprint(
        activated.contract
    )
    forbidden = kernel.submit_sql_candidate(
        session,
        """
        SELECT SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
          AND o.buyer_id IN ('b-1')
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=contract_fingerprint,
    )
    submitted = kernel.submit_sql_candidate(
        session,
        """
        SELECT SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=contract_fingerprint,
    )

    result = kernel.execute_active(session)

    assert forbidden.status == "REJECTED"
    assert forbidden.validation_gaps[0]["code"] == (
        "SQL_RUNTIME_ENTITY_PREDICATE_FORBIDDEN"
    )
    assert submitted.status == "ACCEPTED"
    assert result.merged_query_bundle.rows[0]["total_amount"] == 22.5
    assert "buyer_id IN ('b-1', 'b-2')" in doris.sql
    assert "merchant-1" in doris.sql


def test_sql_submission_rejects_stale_contract_generation() -> None:
    contract = grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)

    with pytest.raises(RuntimeError) as exc_info:
        kernel.submit_sql_candidate(
            session,
            "SELECT 1",
            expected_generation=activated.active_generation + 1,
            expected_contract_fingerprint=grounded_query_contract_fingerprint(
                activated.contract
            ),
        )
    assert "SQL_CANDIDATE_STALE_CONTRACT" in str(exc_info.value)


def test_rejected_latest_candidate_invalidates_previous_accepted_sql() -> None:
    contract = grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
        executor=object(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    fingerprint = grounded_query_contract_fingerprint(activated.contract)
    accepted = kernel.submit_sql_candidate(
        session,
        """
        SELECT o.buyer_id, SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )
    rejected = kernel.submit_sql_candidate(
        session,
        """
        SELECT o.buyer_id, SUM(o.amount) AS wrong_alias
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    assert accepted.status == "ACCEPTED"
    assert rejected.status == "REJECTED"
    assert session.active_preparation is None
    assert session.active_sql_candidate is None
    with pytest.raises(RuntimeError) as exc_info:
        kernel.execute_active(session)
    assert "latest SQL candidate is not" in str(exc_info.value)


def test_identical_contract_reactivation_does_not_reset_generation_or_budget() -> None:
    contract = grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    first_proposal = kernel.propose_contract(session, [], {})
    first = kernel.activate_contract(session, first_proposal.attempt_id)
    fingerprint = grounded_query_contract_fingerprint(first.contract)
    invalid_sql = """
        SELECT o.buyer_id, SUM(o.amount) AS wrong_alias
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
    """
    kernel.submit_sql_candidate(
        session,
        invalid_sql,
        expected_generation=first.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    second_proposal = kernel.propose_contract(session, [], {})
    second = kernel.activate_contract(session, second_proposal.attempt_id)
    repeated = kernel.submit_sql_candidate(
        session,
        invalid_sql,
        expected_generation=second.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    assert second.active_generation == first.active_generation
    assert second.activation_status == "ACTIVE_CONTRACT_REUSED"
    assert repeated.status == "NO_PROGRESS"


def test_repair_exhaustion_is_stable_and_does_not_append_forever() -> None:
    contract = grouped_contract()
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: grouped_pack(),
        compiler=NoTemplateCompiler(),
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    fingerprint = grounded_query_contract_fingerprint(activated.contract)
    for index in range(3):
        rejected = kernel.submit_sql_candidate(
            session,
            """
            SELECT o.buyer_id, SUM(o.amount) AS wrong_%d
            FROM fact_orders o
            WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
            GROUP BY o.buyer_id
            """ % index,
            expected_generation=activated.active_generation,
            expected_contract_fingerprint=fingerprint,
        )
        assert rejected.status == "REJECTED"

    exhausted = kernel.submit_sql_candidate(
        session,
        "SELECT 1",
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )
    attempt_count = len(session.sql_candidate_attempts)
    repeated = kernel.submit_sql_candidate(
        session,
        "SELECT 2",
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    assert exhausted.status == "REPAIR_EXHAUSTED"
    assert repeated.candidate_id == exhausted.candidate_id
    assert len(session.sql_candidate_attempts) == attempt_count
    assert session.active_preparation is None


def test_left_join_right_scope_is_injected_into_on_clause() -> None:
    contract = GroundedQueryContract(
        status="READY",
        question="查询订单及商品标签",
        topics=["orders", "goods"],
        query_shape="DETAIL",
        primary_table="fact_orders",
        tables=[
            GroundedTableBinding(
                topic="orders",
                table="fact_orders",
                merchant_filter_column="tenant_id",
                detail_ref_id="semantic:orders:fact_orders:detail",
            ),
            GroundedTableBinding(
                topic="goods",
                table="dim_goods",
                merchant_filter_column="tenant_id",
                detail_ref_id="semantic:goods:dim_goods:detail",
            ),
        ],
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id="semantic:orders:fact_orders:field:order_id",
                topic="orders",
                table="fact_orders",
                column="order_id",
                output_alias="order_id",
            ),
            GroundedSelectedFieldBinding(
                semantic_ref_id="semantic:goods:dim_goods:field:label",
                topic="goods",
                table="dim_goods",
                column="label",
                output_alias="label",
            ),
        ],
        entity_filters=[
            GroundedEntityFilterBinding(
                semantic_ref_id="semantic:goods:dim_goods:field:goods_id",
                topic="goods",
                table="dim_goods",
                column="goods_id",
                operator="IN",
                literal_value=["g-1", "g-2"],
                entity_identity="entity:product",
                allowed_operators=["IN"],
            )
        ],
        upstream_entity_bindings=[
            GroundedUpstreamEntityBinding(
                entity_set_artifact_id="entity_set_goods",
                source_query_artifact_id="query_artifact_goods",
                source_column="goods_id",
                source_semantic_ref_id="semantic:source:field:goods_id",
                source_entity_identity="entity:product",
                target_field_ref="semantic:goods:dim_goods:field:goods_id",
                target_table="dim_goods",
                target_column="goods_id",
                target_entity_identity="entity:product",
                value_count=2,
                values_hash="goods-values-hash",
            )
        ],
        relationships=[
            GroundedRelationshipBinding(
                semantic_ref_id="semantic:orders:relationship:order_goods",
                topic="orders",
                name="order_goods",
                left_table="fact_orders",
                right_table="dim_goods",
                join_type="LEFT",
                keys=[["tenant_id", "tenant_id"], ["goods_id", "goods_id"]],
                grain="order_to_goods",
                cardinality="MANY_TO_ONE",
                fanout_policy="PRESERVE_LEFT_GRAIN",
            )
        ],
    )
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="fact_orders",
                table="fact_orders",
                columns=["tenant_id", "order_id", "goods_id"],
            ),
            PlanningAssetEntry(
                key="dim_goods",
                table="dim_goods",
                columns=["tenant_id", "goods_id", "label"],
            ),
        ]
    )
    executor = GroundedQueryExecutionKernel(FakeDoris([]), get_settings())

    scoped = executor._inject_candidate_execution_scope(
        """
        SELECT a.order_id, b.label
        FROM fact_orders a
        LEFT JOIN dim_goods b
          ON a.tenant_id = b.tenant_id AND a.goods_id = b.goods_id
        """,
        "merchant-1",
        contract,
        pack,
        {},
    )

    assert "b.tenant_id = 'merchant-1'" in scoped
    assert "b.goods_id IN ('g-1', 'g-2')" in scoped
    assert "a.tenant_id = 'merchant-1'" in scoped
    assert scoped.index("b.tenant_id = 'merchant-1'") < scoped.find(" WHERE ")
    assert scoped.index("b.goods_id IN ('g-1', 'g-2')") < scoped.find(" WHERE ")


def test_post_scope_validation_does_not_union_multitable_columns() -> None:
    pack = PlanningAssetPack(
        tables=[
            PlanningAssetEntry(
                key="table_a",
                table="table_a",
                columns=["id", "only_a"],
            ),
            PlanningAssetEntry(
                key="table_b",
                table="table_b",
                columns=["id", "only_b"],
            ),
        ]
    )

    result = GroundedQueryExecutionKernel.validate_sql(
        "SELECT b.only_a FROM table_a a JOIN table_b b ON a.id = b.id",
        pack,
    )

    assert result.valid is False
    assert result.error_code == "UNKNOWN_COLUMN"
    assert result.unknown_columns == ["b.only_a"]


@pytest.mark.parametrize(
    ("policy_payload", "expected_code"),
    [
        (
            {
                "schemaVersion": 1,
                "defaultEffect": "DENY",
                "allowedMerchantIds": ["merchant-1"],
                "deniedTables": ["fact_orders"],
                "tables": {
                    "fact_orders": {"allowedRoles": ["merchant_analyst"]}
                },
            },
            "TABLE_DENIED",
        ),
        (None, "ACL_POLICY_UNAVAILABLE"),
    ],
)
def test_access_denial_sets_terminal_guard(
    tmp_path: Path,
    policy_payload: dict[str, Any] | None,
    expected_code: str,
) -> None:
    contract = grouped_contract()
    pack = grouped_pack()
    settings = get_settings()
    acl_root = tmp_path / "acl"
    acl_root.mkdir(parents=True)
    if policy_payload is not None:
        (acl_root / "merchant_acl.json").write_text(
            json.dumps(policy_payload),
            encoding="utf-8",
        )
    executor = GroundedQueryExecutionKernel(
        FakeDoris([]),
        settings,
        access_control=AccessControlService(settings, root=acl_root),
    )
    kernel = GroundedRuntimeKernel(
        object(),
        keyword_service=object(),
        topic_router=object(),
        contract_builder=QueueBuilder(contract),
        asset_materializer=lambda _contract, _assets: pack,
        compiler=NoTemplateCompiler(),
        executor=executor,
    )
    session = kernel.new_session(contract.question, "merchant-1")
    proposed = kernel.propose_contract(session, [], {})
    activated = kernel.activate_contract(session, proposed.attempt_id)
    fingerprint = grounded_query_contract_fingerprint(activated.contract)
    kernel.submit_sql_candidate(
        session,
        """
        SELECT o.buyer_id, SUM(o.amount) AS total_amount
        FROM fact_orders o
        WHERE o.event_date BETWEEN '2026-06-01' AND '2026-06-30'
        GROUP BY o.buyer_id
        """,
        expected_generation=activated.active_generation,
        expected_contract_fingerprint=fingerprint,
    )

    result = kernel.execute_active(session)

    assert result.task_results[0].query_bundle.failed is True
    assert session.terminal_guard_code == expected_code
    with pytest.raises(RuntimeError) as raised:
        kernel.submit_sql_candidate(
            session,
            "SELECT 1",
            expected_generation=activated.active_generation,
            expected_contract_fingerprint=fingerprint,
        )
    assert str(raised.value) == "TERMINAL_GUARD:%s" % expected_code
