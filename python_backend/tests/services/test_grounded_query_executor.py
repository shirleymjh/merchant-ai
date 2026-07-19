from __future__ import annotations

import json
from pathlib import Path

import pytest

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import ResolvedTimeRange, ResultCoverage
from merchant_ai.services.access_control import AccessControlService
from merchant_ai.services.assets import TopicAssetService
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_query_contract import (
    GroundedDimensionBinding,
    GroundedEntityFilterBinding,
    GroundedEntityFilterHint,
    GroundedMetricBinding,
    GroundedQueryContract,
    GroundedRankingBinding,
    GroundedRelationshipBinding,
    GroundedSelectedFieldBinding,
    GroundedTableBinding,
    GroundedTimeFieldBinding,
    compile_deterministic_grounded_query,
    compile_grounded_query,
    materialize_grounded_asset_pack,
)
from merchant_ai.services.grounded_query_executor import GroundedQueryExecutionKernel
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)


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


class FakeCappedDetailDoris(FakeDoris):
    def query(self, sql: str, *, timeout_seconds: int) -> list[dict[str, object]]:
        self.calls.append((sql, timeout_seconds))
        return [
            {
                "entity_id": "entity_100",
                "related_id": "related_%d" % index,
                "detail_status": "completed",
                "published_at": "2026-01-05 10:30:00",
            }
            for index in range(101)
        ]


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def explicit_test_access_control(
    settings,
    root,
    contract: GroundedQueryContract,
    *,
    merchant_id: str,
    role: str,
) -> AccessControlService:
    root.mkdir(parents=True, exist_ok=True)
    (root / "merchant_acl.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "defaultEffect": "DENY",
                "allowedMerchantIds": [merchant_id],
                "tables": {
                    table.table: {"allowedRoles": [role]}
                    for table in contract.tables
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return AccessControlService(settings, root=root)


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
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
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
                    "missingDataPolicy": "disclose_unknown",
                    "zeroValuePolicy": "preserve_observed_zero",
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


def literal_filtered_scalar_contract() -> GroundedQueryContract:
    contract = scalar_contract()
    contract.question = "最近30天处理中和待处理的订单数"
    contract.execution_shape = "single_metric"
    contract.metrics = [contract.metrics[0]]
    field_ref = "semantic:经营画像:ads_merchant_profile:column:order_status"
    contract.entity_filters = [
        GroundedEntityFilterBinding(
            semantic_ref_id=field_ref,
            topic="经营画像",
            table="ads_merchant_profile",
            column="order_status",
            operator="IN",
            literal_value=["processing", "pending"],
            requested_phrase="处理中和待处理",
            allowed_operators=["EQ", "IN"],
        )
    ]
    contract.binding_hints.entity_filters = [
        GroundedEntityFilterHint(
            field_ref=field_ref,
            operator="IN",
            literal_value=["processing", "pending"],
            requested_phrase="处理中和待处理",
        )
    ]
    contract.evidence_refs = [field_ref]
    return contract


def ascending_ranked_contract() -> GroundedQueryContract:
    contract = scalar_contract()
    contract.question = "最近30天订单数最少的5个商家类型"
    contract.query_shape = "RANKED"
    contract.execution_shape = "ranked_group"
    contract.metrics = [contract.metrics[0]]
    dimension_ref = "semantic:经营画像:ads_merchant_profile:column:merchant_type"
    contract.dimensions = [
        GroundedDimensionBinding(
            requested_phrase="商家类型",
            semantic_ref_id=dimension_ref,
            topic="经营画像",
            table="ads_merchant_profile",
            column="merchant_type",
            business_name="商家类型",
            role="DIMENSION",
            usage="group_by",
        )
    ]
    contract.ranking = GroundedRankingBinding(
        enabled=True,
        direction="ASC",
        limit=5,
        metric_ref_id=contract.metrics[0].semantic_ref_id,
        dimension_ref_id=dimension_ref,
    )
    return contract


def test_direct_executor_has_no_node_agent_and_preserves_metric_labels(tmp_path) -> None:
    settings = get_settings()
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
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


def test_grounded_query_stages_before_verified_manifest_publication(
    tmp_path: Path,
) -> None:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(
        contract,
        TopicAssetService(settings),
    )
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
            merchant_id="merchant-1",
            role="merchant_admin",
        ),
    )
    artifact_root = (
        settings.resolved_workspace_path
        / "threads"
        / "thread-1"
        / "runs"
        / "run-1"
        / "outputs"
        / "artifacts"
    )

    result = executor.execute_contract(
        "merchant-1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-1",
        artifact_root=str(artifact_root),
        context_owner_fingerprint="owner-fingerprint",
        access_role="merchant_admin",
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="attempt-1",
    )

    bundle = result.merged_query_bundle
    assert bundle.failed is False
    assert bundle.offloaded_files == []
    assert bundle.source_artifact_refs == {}
    assert not list(artifact_root.rglob("*.json"))
    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    staging_root = Path(pending["stagingRoot"])
    rows_path = staging_root / pending["rowsArtifact"]["relativePath"]
    pending_manifest_path = (
        staging_root
        / pending["pendingManifestArtifact"]["relativePath"]
    )
    stored_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    pending_manifest = json.loads(
        pending_manifest_path.read_text(encoding="utf-8")
    )
    assert stored_rows == bundle.rows
    assert pending_manifest["artifactKind"] == "GROUNDED_QUERY_RESULT_PENDING"
    assert pending_manifest["contextOwnerFingerprint"] == "owner-fingerprint"

    verified = EvidenceVerifier().verify(
        contract.question,
        preparation.plan,
        result,
    )
    receipt = executor.publish_pending_result_artifact(
        pending,
        verified_evidence=verified,
        expected_generation=1,
        expected_attempt_id="attempt-1",
        expected_contract_fingerprint=(
            grounded_query_contract_fingerprint(contract)
        ),
        expected_sql_fingerprint=pending["identity"][
            "sqlEvidenceFingerprint"
        ],
        expected_context_owner_fingerprint="owner-fingerprint",
        expected_semantic_activation_fingerprint=(
            preparation.asset_pack_fingerprint
        ),
        expected_data_snapshot=bundle.data_snapshot,
        expected_result_coverage=str(bundle.result_coverage),
        expected_result_is_truncated=bundle.is_truncated,
        expected_stored_row_count=len(bundle.rows),
        expected_exact_result_row_count=bundle.original_row_count,
        expected_rows_canonical_sha256=hashlib.sha256(
            json.dumps(
                bundle.rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    )
    manifest_path = artifact_root / receipt["manifestRelativePath"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifactKind"] == "GROUNDED_QUERY_RESULT_VERIFIED"
    assert manifest["publicationStatus"] == "VERIFIED"
    assert receipt["queryManifestSha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert receipt["rowsSha256"] == manifest["rowsArtifact"]["sha256"]
    assert receipt["rowsRef"].startswith("merchant://artifact/query_results/")


def test_grounded_query_fails_closed_when_result_root_escapes_workspace(
    tmp_path: Path,
) -> None:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(
        contract,
        TopicAssetService(settings),
    )
    preparation = compile_grounded_query(contract, pack)
    executor = GroundedQueryExecutionKernel(
        FakeDoris(),
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
            merchant_id="merchant-1",
            role="merchant_admin",
        ),
    )

    result = executor.execute_contract(
        "merchant-1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        artifact_root=str(tmp_path / "outside"),
        context_owner_fingerprint="owner-fingerprint",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    assert result.merged_query_bundle.failed is True
    assert "QUERY_RESULT_ARTIFACT_STAGING_FAILED" in (
        result.merged_query_bundle.error
    )


def test_business_time_filter_and_proven_partition_pruning_are_separate(
    tmp_path,
) -> None:
    settings = get_settings()
    contract = scalar_contract()
    contract.time_field = GroundedTimeFieldBinding(
        semantic_ref_id="semantic:经营画像:ads_merchant_profile:field:pay_time",
        topic="经营画像",
        table="ads_merchant_profile",
        column="pay_time",
        business_name="支付时间",
        role="DATETIME",
        time_role="BUSINESS_EVENT",
        timezone="Australia/Melbourne",
        partition_pruning_column="pt",
        partition_pruning_policy="EXACT_EQUIVALENT",
    )
    contract.evidence_refs.append(contract.time_field.semantic_ref_id)
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
    )

    executor.execute_contract(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-business-time",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    sql = repository.calls[0][0]
    assert "`pay_time` >= '2026-06-18 00:00:00'" in sql
    assert "`pay_time` < '2026-07-18 00:00:00'" in sql
    assert "`pt` BETWEEN '2026-06-18' AND '2026-07-17'" in sql


def test_business_time_does_not_add_unproven_partition_predicate(tmp_path) -> None:
    settings = get_settings()
    contract = scalar_contract()
    contract.time_field = GroundedTimeFieldBinding(
        semantic_ref_id="semantic:经营画像:ads_merchant_profile:field:event_at",
        topic="经营画像",
        table="ads_merchant_profile",
        column="event_at",
        role="DATETIME",
        time_role="BUSINESS_EVENT",
        timezone="Australia/Melbourne",
    )
    contract.evidence_refs.append(contract.time_field.semantic_ref_id)
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
    )

    executor.execute_contract(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-business-time-no-pruning-proof",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    sql = repository.calls[0][0]
    assert "`event_at` >= '2026-06-18 00:00:00'" in sql
    assert "`pt` BETWEEN" not in sql
    assert "SELECT MAX(`pt`)" not in sql


def test_direct_executor_clamps_doris_timeout_to_shared_runtime_remaining_time(
    tmp_path,
) -> None:
    settings = get_settings()
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
    )
    clock = ManualClock()
    budget = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(max_duration_seconds=5),
        monotonic_clock=clock,
    )
    clock.value = 2.4

    executor.execute_contract(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-budget-timeout",
        access_role="merchant_admin",
        execution_preparation=preparation,
        runtime_budget=budget,
    )

    assert len(repository.calls) == 1
    assert repository.calls[0][1] == 2
    assert repository.calls[0][1] <= budget.remaining_seconds()


def test_direct_executor_fails_before_doris_when_runtime_has_under_one_second(
    tmp_path,
) -> None:
    settings = get_settings()
    contract = scalar_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
    )
    clock = ManualClock()
    budget = GroundedRuntimeBudget(
        GroundedRuntimeBudgetLimits(max_duration_seconds=5),
        monotonic_clock=clock,
    )
    clock.value = 4.25

    with pytest.raises(GroundedRuntimeBudgetExceeded) as raised:
        executor.execute_contract(
            "99999999999999999999999999999999",
            contract,
            preparation.plan,
            pack,
            contract.question,
            run_id="run-budget-denied",
            access_role="merchant_admin",
            execution_preparation=preparation,
            runtime_budget=budget,
        )

    assert raised.value.breaches == ("duration",)
    assert repository.calls == []


def test_direct_executor_compiles_and_proves_literal_metric_filters(tmp_path) -> None:
    settings = get_settings()
    contract = literal_filtered_scalar_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_deterministic_grounded_query(contract, pack)
    repository = FakeDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="99999999999999999999999999999999",
            role="merchant_admin",
        ),
    )

    result = executor.execute_contract(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-filtered-metric",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    sql = repository.calls[0][0]
    assert "`order_status` IN ('processing', 'pending')" in sql
    node_contract = result.task_results[0].node_plan_contract
    assert node_contract.filter_column == "order_status"
    assert node_contract.filter_values == ["processing", "pending"]
    assert len(node_contract.entity_filter_obligations) == 1
    proof = result.task_results[0].entity_filter_verification
    assert proof.verified is True
    assert proof.coverage_complete is True
    verified = EvidenceVerifier().verify(contract.question, preparation.plan, result)
    assert verified.passed, [gap.model_dump() for gap in verified.blocking_gaps]


def test_direct_executor_preserves_ascending_rank_direction(tmp_path) -> None:
    settings = get_settings()
    contract = ascending_ranked_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_deterministic_grounded_query(contract, pack)
    executor = GroundedQueryExecutionKernel(
        FakeDoris(),
        settings,
        access_control=AccessControlService(settings, root=tmp_path),
    )

    compilation = executor.compile_sql(
        "99999999999999999999999999999999",
        contract,
        preparation.plan,
        pack,
        access_role="merchant_admin",
        user_scope={},
    )

    assert "ORDER BY `order_cnt_1d` ASC LIMIT 5" in compilation.sql


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
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="merchant_1",
            role="merchant_admin",
        ),
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
    assert "LIMIT 101" in sql
    assert result.merged_query_bundle.tables == [
        "fact_entity_detail",
        "dim_related_entity",
    ]
    assert result.merged_query_bundle.result_coverage == ResultCoverage.ALL_ROWS.value
    assert result.merged_query_bundle.original_row_count == 1
    assert result.merged_query_bundle.is_truncated is False
    assert result.task_results[0].entity_filter_verification.verified is True
    verified = EvidenceVerifier().verify(contract.question, preparation.plan, result)
    assert verified.passed, [gap.model_dump() for gap in verified.blocking_gaps]


def test_detail_executor_uses_sentinel_row_to_mark_capped_result_as_preview(
    tmp_path,
) -> None:
    settings = get_settings()
    contract = detail_lookup_contract()
    pack = materialize_grounded_asset_pack(contract, TopicAssetService(settings))
    preparation = compile_grounded_query(contract, pack)
    repository = FakeCappedDetailDoris()
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path,
            contract,
            merchant_id="merchant_1",
            role="merchant_admin",
        ),
    )

    result = executor.execute_contract(
        "merchant_1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-detail-capped",
        access_role="merchant_admin",
        execution_preparation=preparation,
    )

    bundle = result.merged_query_bundle
    assert "LIMIT 101" in repository.calls[0][0]
    assert len(bundle.rows) == 100
    assert bundle.result_coverage == ResultCoverage.PREVIEW.value
    assert bundle.is_truncated is True
    assert bundle.original_row_count == 0
    assert bundle.effective_row_count() == 100
    assert bundle.runtime_events[0]["fetchedRowCount"] == 101


def test_preview_result_artifact_cannot_claim_complete_population(
    tmp_path: Path,
) -> None:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    contract = detail_lookup_contract()
    pack = materialize_grounded_asset_pack(
        contract,
        TopicAssetService(settings),
    )
    preparation = compile_grounded_query(contract, pack)
    executor = GroundedQueryExecutionKernel(
        FakeCappedDetailDoris(),
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
            merchant_id="merchant_1",
            role="merchant_admin",
        ),
    )
    artifact_root = settings.resolved_workspace_path / "artifacts"

    result = executor.execute_contract(
        "merchant_1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-preview",
        artifact_root=str(artifact_root),
        context_owner_fingerprint="owner-fingerprint",
        access_role="merchant_admin",
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="attempt-preview",
    )

    bundle = result.merged_query_bundle
    assert bundle.offloaded_files == []
    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    staging_root = Path(pending["stagingRoot"])
    manifest_path = (
        staging_root
        / pending["pendingManifestArtifact"]["relativePath"]
    )
    rows_path = staging_root / pending["rowsArtifact"]["relativePath"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    assert len(stored_rows) == 100
    assert manifest["resultCoverage"] == ResultCoverage.PREVIEW.value
    assert manifest["resultIsTruncated"] is True
    assert manifest["exactResultRowCount"] == 0


@pytest.mark.parametrize(
    ("sql", "returned_rows", "expected"),
    [
        ("SELECT entity_id FROM fact_entity_detail LIMIT 100", 100, "PREVIEW"),
        ("SELECT entity_id FROM fact_entity_detail LIMIT 100", 99, "ALL_ROWS"),
        (
            "SELECT entity_id FROM (SELECT entity_id FROM fact_entity_detail LIMIT 100) q",
            50,
            "PREVIEW",
        ),
        (
            "SELECT entity_id FROM fact_entity_detail LIMIT 100 OFFSET 1",
            20,
            "PREVIEW",
        ),
    ],
)
def test_core_sql_detail_coverage_is_fail_closed_for_ambiguous_limits(
    sql: str,
    returned_rows: int,
    expected: str,
) -> None:
    assert (
        GroundedQueryExecutionKernel._core_sql_detail_coverage(sql, returned_rows)
        == expected
    )
