from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from merchant_ai.config import Settings, get_settings
from merchant_ai.models import (
    DataSnapshotContract,
    ResolvedTimeRange,
    ResultCoverage,
    VerifiedEvidence,
)
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
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    PopulationPreExecutionNodeReference,
    PopulationPreExecutionReference,
    seal_population_pre_execution_reference,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)
from merchant_ai.services.grounded_result_streaming import (
    grounded_canonical_json_sha256,
)
from merchant_ai.services.grounded_context_workspace import GroundedContextWorkspace
from merchant_ai.services.grounded_runtime_budget import (
    GroundedRuntimeBudget,
    GroundedRuntimeBudgetExceeded,
    GroundedRuntimeBudgetLimits,
)
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
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


class FakeStreamingDetailDoris:
    def __init__(
        self,
        row_count: int = 257,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.row_count = row_count
        self.query_calls = 0
        self.stream_calls: list[dict[str, object]] = []
        self.events = events if events is not None else []
        self.last_cache_hit = False
        self.last_cache_key = ""

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        self.events.append("snapshot")
        return DataSnapshotContract(
            datasource_fingerprint="datasource-stream-test",
            datasource_environment="test",
            consistency_mode="UNSUPPORTED",
            semantic_activation_fingerprint=(
                semantic_activation_fingerprint
            ),
            cache_generation="generation-test",
            unsupported_reason="TEST_SNAPSHOT_NON_ATOMIC",
        )

    def revalidate_data_snapshot(
        self,
        snapshot: DataSnapshotContract,
    ) -> DataSnapshotContract:
        self.events.append("snapshot-revalidated")
        return snapshot.model_copy(deep=True)

    def query(self, sql: str, **kwargs: object) -> list[dict[str, object]]:
        del sql, kwargs
        self.query_calls += 1
        raise AssertionError("streaming artifact mode must not call query/fetchall")

    def stream_query_batches(
        self,
        sql: str,
        *,
        batch_size: int,
        cancel_events=None,
        timeout_seconds: int | None = None,
        data_snapshot_contract: DataSnapshotContract | None = None,
    ):
        self.events.append("stream")
        self.stream_calls.append(
            {
                "sql": sql,
                "batchSize": batch_size,
                "cancelEvents": tuple(cancel_events or ()),
                "timeoutSeconds": timeout_seconds,
                "snapshot": data_snapshot_contract,
            }
        )
        for start in range(0, self.row_count, batch_size):
            yield [
                {
                    "entity_id": "entity_100",
                    "related_id": "related_%d" % index,
                    "detail_status": "completed",
                    "published_at": "2026-01-05 10:30:00",
                }
                for index in range(start, min(self.row_count, start + batch_size))
            ]


class RecordingPopulationExecutionGate(GroundedPopulationExecutionGate):
    def __init__(self, events: list[str], *, accepted: bool = True) -> None:
        self.events = events
        self.accepted = accepted
        self.calls: list[dict[str, object]] = []

    def authorize_node(self, **kwargs):
        self.events.append("population")
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            accepted=self.accepted,
            code="ACCEPTED" if self.accepted else "REJECTED",
        )


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
                calendar_anchor_policy="runtime_current_date",
                data_as_of_policy="latest_available_partition",
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
                calendar_anchor_policy="runtime_current_date",
                data_as_of_policy="latest_available_partition",
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
            calendar_anchor_policy="runtime_current_date",
            data_as_of_policy="latest_available_partition",
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
    assert result.task_results[0].query_bundle.rows is (
        result.query_bundles[0].rows
    )
    assert result.query_bundles[0].rows is result.merged_query_bundle.rows
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
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_admin",
        user_scope={},
        question=contract.question,
    )
    artifact_root = workspace.artifacts_root

    result = executor.execute_contract(
        "merchant-1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-1",
        artifact_root=str(artifact_root),
        context_owner_fingerprint=workspace.owner_fingerprint,
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
    assert (
        pending_manifest["contextOwnerFingerprint"]
        == workspace.owner_fingerprint
    )

    verified = EvidenceVerifier().verify(
        contract.question,
        preparation.plan,
        result,
    )
    def publish_once(_: int) -> dict[str, object]:
        return executor.publish_pending_result_artifact(
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
            expected_context_owner_fingerprint=workspace.owner_fingerprint,
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

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(publish_once, range(12)))
    assert len(
        {str(item["queryManifestSha256"]) for item in receipts}
    ) == 1
    receipt = receipts[0]
    manifest_path = artifact_root / receipt["manifestRelativePath"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifactKind"] == "GROUNDED_QUERY_RESULT"
    assert manifest["publicationStatus"] == "VERIFIED"
    assert manifest_path.name == "result_%s.manifest.json" % receipt[
        "queryManifestSha256"
    ]
    assert receipt["queryManifestSha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert receipt["rowsSha256"] == manifest["rowsArtifact"]["sha256"]
    assert receipt["rowsRef"].startswith("merchant://artifact/query_results/")


def test_unsupported_snapshot_identity_still_blocks_boundary_mismatch(
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
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_admin",
        user_scope={},
        question=contract.question,
    )
    observed_snapshot = DataSnapshotContract(
        datasource_fingerprint="datasource-a",
        datasource_environment="production",
        consistency_mode="UNSUPPORTED",
        semantic_activation_fingerprint=(
            preparation.asset_pack_fingerprint
        ),
        cache_generation="cache-generation-a",
        captured_at="2026-07-19T00:00:00Z",
        unsupported_reason="ATOMIC_SNAPSHOT_UNAVAILABLE",
    )
    result = executor.execute_contract(
        "merchant-1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-1",
        artifact_root=str(workspace.artifacts_root),
        context_owner_fingerprint=workspace.owner_fingerprint,
        access_role="merchant_admin",
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="attempt-1",
        data_snapshot_contract=observed_snapshot,
    )
    bundle = result.merged_query_bundle
    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    snapshot_identity = pending["identity"]["dataSnapshot"]
    assert snapshot_identity["datasourceFingerprint"] == "datasource-a"
    assert snapshot_identity["datasourceEnvironment"] == "production"
    assert snapshot_identity["semanticActivationFingerprint"] == (
        preparation.asset_pack_fingerprint
    )
    assert snapshot_identity["cacheGeneration"] == "cache-generation-a"
    assert snapshot_identity["dataEpoch"] == ""
    assert snapshot_identity["consistencyMode"] == "UNSUPPORTED"
    verified = EvidenceVerifier().verify(
        contract.question,
        preparation.plan,
        result,
    )
    common = {
        "verified_evidence": verified,
        "expected_generation": 1,
        "expected_attempt_id": "attempt-1",
        "expected_contract_fingerprint": (
            grounded_query_contract_fingerprint(contract)
        ),
        "expected_sql_fingerprint": pending["identity"][
            "sqlEvidenceFingerprint"
        ],
        "expected_context_owner_fingerprint": workspace.owner_fingerprint,
        "expected_semantic_activation_fingerprint": (
            preparation.asset_pack_fingerprint
        ),
        "expected_result_coverage": str(bundle.result_coverage),
        "expected_result_is_truncated": bundle.is_truncated,
        "expected_stored_row_count": len(bundle.rows),
        "expected_exact_result_row_count": bundle.original_row_count,
        "expected_rows_canonical_sha256": hashlib.sha256(
            json.dumps(
                bundle.rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }
    mismatched_snapshots = [
        observed_snapshot.model_copy(
            update={"datasource_fingerprint": "datasource-b"}
        ),
        observed_snapshot.model_copy(
            update={"semantic_activation_fingerprint": "activation-b"}
        ),
    ]
    for mismatched_snapshot in mismatched_snapshots:
        with pytest.raises(RuntimeError) as raised:
            executor.publish_pending_result_artifact(
                pending,
                expected_data_snapshot=mismatched_snapshot,
                **common,
            )
        assert "QUERY_RESULT_PENDING_BINDING_MISMATCH:dataSnapshot" in str(
            raised.value
        )

    with pytest.raises(RuntimeError) as raised:
        executor.publish_pending_result_artifact(
            pending,
            expected_data_snapshot=observed_snapshot,
            **{
                **common,
                "expected_semantic_activation_fingerprint": "activation-b",
            },
        )
    assert (
        "QUERY_RESULT_PENDING_BINDING_MISMATCH:"
        "semanticActivationFingerprint"
    ) in str(raised.value)

    receipt = executor.publish_pending_result_artifact(
        pending,
        expected_data_snapshot=observed_snapshot,
        **common,
    )
    manifest = json.loads(
        (
            workspace.artifacts_root / receipt["manifestRelativePath"]
        ).read_text(encoding="utf-8")
    )
    assert manifest["dataSnapshot"] == snapshot_identity
    assert receipt["dataSnapshotFingerprint"] == hashlib.sha256(
        json.dumps(
            snapshot_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


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
    assert "QUERY_RESULT_ARTIFACT_ROOT_INVALID" in (
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


def test_detail_publication_fails_closed_without_streaming_repository(
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
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-preview",
        run_id="run-preview",
        merchant_id="merchant_1",
        access_role="merchant_admin",
        user_scope={},
        question=contract.question,
    )
    artifact_root = workspace.artifacts_root

    result = executor.execute_contract(
        "merchant_1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-preview",
        artifact_root=str(artifact_root),
        context_owner_fingerprint=workspace.owner_fingerprint,
        access_role="merchant_admin",
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="attempt-preview",
    )

    bundle = result.merged_query_bundle
    assert bundle.failed is True
    assert "QUERY_RESULT_STREAMING_REQUIRED" in bundle.error
    assert executor.doris_repository.calls == []
    assert not list(artifact_root.rglob("rows.json"))


def _execute_streaming_detail(
    tmp_path: Path,
    *,
    row_count: int = 257,
    preview_rows: int = 7,
    max_rows: int = 10_000,
    max_bytes: int = 16 * 1024 * 1024,
    cancel_events=None,
    population_gate: GroundedPopulationExecutionGate | None = None,
    include_population_reference: bool = False,
    population_query_node_id: str = "",
    population_reference_node_id: str = "",
    include_waiting_population_node: bool = False,
):
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace"),
        context_artifact_inline_max_rows=preview_rows,
        grounded_result_stream_fetch_batch_rows=13,
        grounded_result_stream_max_rows=max_rows,
        grounded_result_stream_max_bytes=max_bytes,
    )
    contract = detail_lookup_contract()
    pack = materialize_grounded_asset_pack(
        contract,
        TopicAssetService(settings),
    )
    preparation = compile_grounded_query(contract, pack)
    shared_events = getattr(population_gate, "events", None)
    repository = FakeStreamingDetailDoris(
        row_count=row_count,
        events=(shared_events if isinstance(shared_events, list) else None),
    )
    executor = GroundedQueryExecutionKernel(
        repository,
        settings,
        access_control=explicit_test_access_control(
            settings,
            tmp_path / "acl",
            contract,
            merchant_id="merchant_1",
            role="merchant_admin",
        ),
        population_execution_gate=population_gate,
    )
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-stream",
        run_id="run-stream",
        merchant_id="merchant_1",
        access_role="merchant_admin",
        user_scope={},
        question=contract.question,
    )
    goal_fingerprint = hashlib.sha256(b"goal-contract").hexdigest()
    current_population_node_id = str(
        population_reference_node_id
        or population_query_node_id
        or preparation.plan.intents[0].plan_task_id
    )
    dynamic_graph_nodes = [
        PopulationDynamicGraphNode(
            query_node_id=current_population_node_id,
            consumer_goal_ids=("goal-detail",),
        )
    ]
    if include_waiting_population_node:
        dynamic_graph_nodes.append(
            PopulationDynamicGraphNode(
                query_node_id="query-node-waiting",
                consumer_goal_ids=("goal-waiting",),
            )
        )
    population_reference = (
        seal_population_pre_execution_reference(
            PopulationPreExecutionReference(
                gate_id="population-gate-test",
                context_owner_fingerprint=workspace.owner_fingerprint,
                run_authority_fingerprint=workspace.request_fingerprint,
                goal_contract_fingerprint=goal_fingerprint,
                graph_receipt=seal_population_dynamic_graph_receipt(
                    PopulationDynamicGraphReceipt(
                        graph_id="dynamic-graph-test",
                        graph_version=1,
                        graph_fingerprint=hashlib.sha256(
                            b"dynamic-graph"
                        ).hexdigest(),
                        nodes=tuple(dynamic_graph_nodes),
                    )
                ),
                node=PopulationPreExecutionNodeReference(
                        query_node_id=current_population_node_id,
                        consumer_goal_ids=("goal-detail",),
                        generation=1,
                        attempt_id="attempt-stream",
                        query_contract_fingerprint=(
                            grounded_query_contract_fingerprint(contract)
                        ),
                    ),
            )
        )
        if include_population_reference
        else None
    )
    result = executor.execute_contract(
        "merchant_1",
        contract,
        preparation.plan,
        pack,
        contract.question,
        run_id="run-stream",
        artifact_root=str(workspace.artifacts_root),
        context_owner_fingerprint=workspace.owner_fingerprint,
        access_role="merchant_admin",
        user_scope={},
        execution_reference_scope={},
        execution_goal_contract_fingerprint=goal_fingerprint,
        expected_semantic_activation_fingerprint=(
            preparation.asset_pack_fingerprint
        ),
        population_pre_execution_reference=population_reference,
        population_query_node_id=population_query_node_id,
        execution_preparation=preparation,
        execution_generation=1,
        execution_attempt_id="attempt-stream",
        cancel_events=cancel_events,
    )
    return {
        "settings": settings,
        "contract": contract,
        "preparation": preparation,
        "repository": repository,
        "executor": executor,
        "workspace": workspace,
        "goalFingerprint": goal_fingerprint,
        "populationReference": population_reference,
        "result": result,
    }


def test_governed_detail_streams_complete_rows_and_retains_bounded_preview(
    tmp_path: Path,
) -> None:
    values = _execute_streaming_detail(tmp_path)
    repository = values["repository"]
    result = values["result"]
    bundle = result.merged_query_bundle

    assert bundle.failed is False
    assert repository.query_calls == 0
    assert len(repository.stream_calls) == 1
    assert repository.events == ["snapshot", "stream"]
    assert " LIMIT " not in repository.stream_calls[0]["sql"]
    assert len(bundle.rows) == 7
    assert bundle.original_row_count == 257
    assert bundle.source_row_counts == {
        result.tasks[0].task_id: 257,
    }
    assert bundle.result_coverage == ResultCoverage.ALL_ROWS.value
    assert bundle.is_truncated is True
    assert bundle.runtime_events[0]["fetchedRowCount"] == 257
    assert bundle.runtime_events[0]["previewRowCount"] == 7

    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    identity = pending["identity"]
    rows_path = (
        Path(pending["stagingRoot"])
        / pending["rowsArtifact"]["relativePath"]
    )
    stored_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    assert len(stored_rows) == 257
    assert stored_rows[:7] == bundle.rows
    assert identity["storedRowCount"] == 7
    assert identity["previewRowCount"] == 7
    assert identity["artifactRowCount"] == 257
    assert identity["exactResultRowCount"] == 257
    assert identity["artifactByteCount"] == len(rows_path.read_bytes())
    assert identity["artifactRowsSha256"] == hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
    assert identity["artifactContentAddress"] == (
        "sha256:%s" % identity["artifactRowsSha256"]
    )
    assert identity["artifactCoverage"] == "ALL_ROWS"
    assert identity["artifactComplete"] is True
    assert identity["streamingMaterialized"] is True
    assert identity["runExecutionIdentity"]["sealFingerprint"]
    assert identity["nodeExecutionIdentity"]["sealFingerprint"]
    verified = EvidenceVerifier().verify(
        values["contract"].question,
        values["preparation"].plan,
        result,
    )
    assert verified.passed, [
        gap.model_dump() for gap in verified.blocking_gaps
    ]


def test_population_pre_gate_runs_after_snapshot_and_before_doris_stream(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    gate = RecordingPopulationExecutionGate(events)
    values = _execute_streaming_detail(
        tmp_path,
        row_count=19,
        population_gate=gate,
        include_population_reference=True,
    )

    assert values["result"].merged_query_bundle.failed is False
    assert events == ["snapshot", "population", "stream"]
    assert len(gate.calls) == 1
    assert gate.calls[0]["reference"] == values["populationReference"]
    assert gate.calls[0]["execution"].data_snapshot == (
        values["result"].merged_query_bundle.data_snapshot
    )


def test_population_pre_gate_uses_server_graph_node_identity_with_waiting_peer(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    gate = RecordingPopulationExecutionGate(events)
    values = _execute_streaming_detail(
        tmp_path,
        row_count=19,
        population_gate=gate,
        include_population_reference=True,
        population_query_node_id="query-node-ready",
        include_waiting_population_node=True,
    )

    assert values["result"].merged_query_bundle.failed is False
    assert events == ["snapshot", "population", "stream"]
    reference = values["populationReference"]
    assert len(reference.graph_receipt.nodes) == 2
    assert reference.node.query_node_id == "query-node-ready"
    assert gate.calls[0]["execution"].query_node_id == "query-node-ready"
    assert (
        gate.calls[0]["execution"].query_node_id
        != values["preparation"].plan.intents[0].plan_task_id
    )


def test_population_server_graph_node_identity_mismatch_never_reaches_doris(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    gate = RecordingPopulationExecutionGate(events)
    values = _execute_streaming_detail(
        tmp_path,
        row_count=19,
        population_gate=gate,
        include_population_reference=True,
        population_query_node_id="query-node-other",
        population_reference_node_id="query-node-ready",
        include_waiting_population_node=True,
    )

    assert values["result"].merged_query_bundle.failed is True
    assert "POPULATION_PRE_EXECUTION_REJECTED" in (
        values["result"].merged_query_bundle.error
    )
    assert gate.calls == []
    assert values["repository"].query_calls == 0
    assert values["repository"].stream_calls == []


@pytest.mark.parametrize(
    ("inject_gate", "include_reference", "accepted", "expected_failure"),
    [
        (True, False, True, False),
        (False, True, True, True),
        (True, True, False, True),
    ],
)
def test_population_pre_gate_applies_only_when_query_binds_a_reference(
    tmp_path: Path,
    inject_gate: bool,
    include_reference: bool,
    accepted: bool,
    expected_failure: bool,
) -> None:
    events: list[str] = []
    gate = RecordingPopulationExecutionGate(
        events,
        accepted=accepted,
    )
    values = _execute_streaming_detail(
        tmp_path,
        row_count=19,
        population_gate=gate if inject_gate else None,
        include_population_reference=include_reference,
    )
    bundle = values["result"].merged_query_bundle

    assert bundle.failed is expected_failure
    if expected_failure:
        assert "POPULATION_PRE_EXECUTION_REJECTED" in bundle.error
        assert values["repository"].query_calls == 0
        assert values["repository"].stream_calls == []
        assert "stream" not in events
    else:
        assert gate.calls == []
        assert len(values["repository"].stream_calls) == 1
        assert events == ["snapshot", "stream"]


def test_publication_revalidates_identity_and_streams_exact_staged_rows(
    tmp_path: Path,
) -> None:
    values = _execute_streaming_detail(tmp_path, row_count=41)
    result = values["result"]
    bundle = result.merged_query_bundle
    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    receipt = values["executor"].publish_pending_result_artifact(
        pending,
        verified_evidence=VerifiedEvidence(passed=True),
        expected_generation=1,
        expected_attempt_id="attempt-stream",
        expected_contract_fingerprint=(
            grounded_query_contract_fingerprint(values["contract"])
        ),
        expected_sql_fingerprint=pending["identity"][
            "sqlEvidenceFingerprint"
        ],
        expected_context_owner_fingerprint=(
            values["workspace"].owner_fingerprint
        ),
        expected_semantic_activation_fingerprint=(
            values["preparation"].asset_pack_fingerprint
        ),
        expected_data_snapshot=bundle.data_snapshot,
        expected_result_coverage=str(bundle.result_coverage),
        expected_result_is_truncated=bundle.is_truncated,
        expected_stored_row_count=len(bundle.rows),
        expected_exact_result_row_count=bundle.original_row_count,
        expected_rows_canonical_sha256=(
            grounded_canonical_json_sha256(bundle.rows)
        ),
        expected_goal_contract_fingerprint=values["goalFingerprint"],
        expected_merchant_id="merchant_1",
        expected_access_role="merchant_admin",
        expected_user_scope={},
        expected_reference_scope={},
    )

    published_rows = (
        values["workspace"].artifacts_root
        / receipt["rowsRelativePath"]
    )
    assert len(json.loads(published_rows.read_text(encoding="utf-8"))) == 41
    assert receipt["storedRowCount"] == len(bundle.rows)
    assert receipt["artifactRowCount"] == 41
    assert receipt["artifactByteCount"] == len(published_rows.read_bytes())
    assert receipt["artifactCoverage"] == "ALL_ROWS"
    assert receipt["artifactComplete"] is True


def test_tampered_streamed_rows_cannot_publish(
    tmp_path: Path,
) -> None:
    values = _execute_streaming_detail(tmp_path, row_count=31)
    bundle = values["result"].merged_query_bundle
    pending = bundle.runtime_events[0][
        "_serverPrivatePendingResultArtifact"
    ]
    rows_path = (
        Path(pending["stagingRoot"])
        / pending["rowsArtifact"]["relativePath"]
    )
    rows_path.chmod(0o600)
    rows_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        values["executor"].publish_pending_result_artifact(
            pending,
            verified_evidence=VerifiedEvidence(passed=True),
            expected_generation=1,
            expected_attempt_id="attempt-stream",
            expected_contract_fingerprint=(
                grounded_query_contract_fingerprint(values["contract"])
            ),
            expected_sql_fingerprint=pending["identity"][
                "sqlEvidenceFingerprint"
            ],
            expected_context_owner_fingerprint=(
                values["workspace"].owner_fingerprint
            ),
            expected_semantic_activation_fingerprint=(
                values["preparation"].asset_pack_fingerprint
            ),
            expected_data_snapshot=bundle.data_snapshot,
            expected_result_coverage=str(bundle.result_coverage),
            expected_result_is_truncated=bundle.is_truncated,
            expected_stored_row_count=len(bundle.rows),
            expected_exact_result_row_count=bundle.original_row_count,
            expected_rows_canonical_sha256=(
                grounded_canonical_json_sha256(bundle.rows)
            ),
            expected_goal_contract_fingerprint=(
                values["goalFingerprint"]
            ),
            expected_merchant_id="merchant_1",
            expected_access_role="merchant_admin",
            expected_user_scope={},
            expected_reference_scope={},
        )
    assert "QUERY_RESULT_PENDING_ARTIFACT_SHA_MISMATCH" in str(
        raised.value
    )
    assert not list(
        values["workspace"].artifacts_root.glob(
            "query_results/*_rows.json"
        )
    )


def test_stream_quota_or_cancellation_never_stages_pending_artifact(
    tmp_path: Path,
) -> None:
    quota = _execute_streaming_detail(
        tmp_path / "quota",
        row_count=30,
        max_rows=5,
    )
    quota_bundle = quota["result"].merged_query_bundle
    assert quota_bundle.failed is True
    assert "RESULT_STREAM_ROW_QUOTA_EXCEEDED" in quota_bundle.error
    assert not list(
        quota["workspace"].staging_root.rglob("pending.manifest.json")
    )

    cancelled_event = Event()
    cancelled_event.set()
    cancelled = _execute_streaming_detail(
        tmp_path / "cancelled",
        row_count=30,
        cancel_events=[cancelled_event],
    )
    cancelled_bundle = cancelled["result"].merged_query_bundle
    assert cancelled_bundle.failed is True
    assert "RESULT_STREAM_CANCELLED" in cancelled_bundle.error
    assert not list(
        cancelled["workspace"].staging_root.rglob(
            "pending.manifest.json"
        )
    )


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
